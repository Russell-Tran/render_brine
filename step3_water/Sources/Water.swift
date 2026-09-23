// Renders a still image of open water on the GPU, one thread per pixel.
//
// A camera 3 m above a flat sea looks toward the horizon. For each pixel below
// the horizon, the kernel finds where that pixel's ray hits the sea, adds up a
// few sine waves there to get the slope of the surface, and shades it by how
// much it tilts toward the sun: deep blue when facing away, turquoise when
// facing the sun, plus a glint. Above the horizon is the step 2 sky gradient.
//
// The waves only change the shading, not the shape (like a bump map): the sea
// itself stays flat. Every pixel is independent, so this is an
// "embarrassingly parallel" job, the easiest kind of work for a GPU.

import CoreGraphics
import Foundation
import ImageIO
import Metal
import UniformTypeIdentifiers
import simd

// MARK: - Scene

/// One sine wave on the sea surface. Must match `Wave` in the kernel.
struct Wave: Equatable {
    var directionX: Float   // unit direction the wave travels, on the sea plane
    var directionZ: Float
    var wavenumber: Float   // 2π / wavelength, in radians per meter
    var amplitude: Float    // meters
    var phase: Float        // radians

    init(angleDegrees: Float, wavelength: Float, amplitude: Float, phase: Float = 0) {
        let a = angleDegrees * .pi / 180
        directionX = cos(a)
        directionZ = sin(a)
        wavenumber = 2 * .pi / wavelength
        self.amplitude = amplitude
        self.phase = phase
    }
}

/// Five waves, from long swells to short ripples, each heading a different way.
let defaultWaves = [
    Wave(angleDegrees: 75, wavelength: 14, amplitude: 0.35),
    Wave(angleDegrees: 110, wavelength: 8, amplitude: 0.20, phase: 1.3),
    Wave(angleDegrees: 40, wavelength: 4.5, amplitude: 0.10, phase: 2.1),
    Wave(angleDegrees: 95, wavelength: 2.2, amplitude: 0.05, phase: 0.7),
    Wave(angleDegrees: 150, wavelength: 1.3, amplitude: 0.03, phase: 4.0),
]

// Shared by the CPU reference and the kernel (interpolated into its source).
let cameraHeight: Float = 3          // meters above the sea
let tanHalfFOV: Float = 0.577        // tan(30°): a 60° vertical field of view
let tilt: Float = 0.15               // looks slightly down, putting the horizon ~37% from the top
let sunDirection = simd_normalize(SIMD3<Float>(0, 0.35, 1))  // ahead of the camera, 19° up (frame top is ~23°)
let sunDiskCosine: Float = 0.9994    // sun disk radius ≈ 2°
let specularPower: Float = 300
let specularStrength: Float = 0.9
let fogDensity: Float = 0.012        // per meter
let skySpread: Float = 4             // how quickly the sky goes from haze to blue

let skyBlue = SIMD3<Float>(140, 200, 235) / 255      // same as step 2
let deepBlue = SIMD3<Float>(8, 40, 90) / 255         // same as step 2
let turquoise = SIMD3<Float>(40, 190, 190) / 255
let haze = SIMD3<Float>(200, 225, 240) / 255
let sunColor = SIMD3<Float>(1, 0.97, 0.9)

// MARK: - CPU reference

/// Height of the sea at (x, z): the sum of every wave.
func waveHeight(x: Float, z: Float, waves: [Wave]) -> Float {
    waves.reduce(0) { h, w in h + w.amplitude * sin(w.wavenumber * (w.directionX * x + w.directionZ * z) + w.phase) }
}

/// Surface normal at (x, z), from the exact slope (derivative) of the waves.
func waveNormal(x: Float, z: Float, waves: [Wave]) -> SIMD3<Float> {
    var dx: Float = 0, dz: Float = 0
    for w in waves {
        let c = cos(w.wavenumber * (w.directionX * x + w.directionZ * z) + w.phase)
        dx += w.amplitude * w.wavenumber * w.directionX * c
        dz += w.amplitude * w.wavenumber * w.directionZ * c
    }
    return simd_normalize(SIMD3<Float>(-dx, 1, -dz))
}

/// The color of pixel (px, py), each channel from 0 to 1. The kernel below
/// does exactly the same math on the GPU.
func shadePixel(px: Int, py: Int, width: Int, height: Int, waves: [Wave]) -> SIMD3<Float> {
    let aspect = Float(width) / Float(height)
    let u = (Float(px) + 0.5) / Float(width)
    let v = (Float(py) + 0.5) / Float(height)
    let d = SIMD3<Float>((2 * u - 1) * aspect * tanHalfFOV, (1 - 2 * v) * tanHalfFOV - tilt, 1)
    let dn = simd_normalize(d)

    if d.y >= 0 {
        if simd_dot(dn, sunDirection) > sunDiskCosine { return sunColor }
        return simd_mix(haze, skyBlue, SIMD3(repeating: simd_clamp(dn.y * skySpread, 0, 1)))
    }

    let s = cameraHeight / -d.y            // where the ray meets the sea
    let n = waveNormal(x: d.x * s, z: d.z * s, waves: waves)
    let facing = simd_clamp(simd_dot(n, sunDirection), 0, 1)
    let halfway = simd_normalize(sunDirection - dn)
    let glint = pow(simd_clamp(simd_dot(n, halfway), 0, 1), specularPower) * specularStrength
    let water = simd_mix(deepBlue, turquoise, SIMD3(repeating: facing * facing)) + glint
    let fog = 1 - exp(-simd_length(d) * s * fogDensity)
    return simd_clamp(simd_mix(water, haze, SIMD3(repeating: fog)), .zero, SIMD3(repeating: 1))
}

// MARK: - GPU

// Must match `Params` in the kernel.
struct Params {
    var width: UInt32
    var height: UInt32
    var waveCount: UInt32
}

func metal(_ v: SIMD3<Float>) -> String { "float3(\(v.x), \(v.y), \(v.z))" }

let waterKernelSource = """
    #include <metal_stdlib>
    using namespace metal;

    struct Params { uint width; uint height; uint waveCount; };
    struct Wave { float directionX; float directionZ; float wavenumber; float amplitude; float phase; };

    constant float CAMERA_HEIGHT = \(cameraHeight);
    constant float TAN_HALF_FOV = \(tanHalfFOV);
    constant float TILT = \(tilt);
    constant float3 SUN = \(metal(sunDirection));
    constant float SUN_DISK_COSINE = \(sunDiskCosine);
    constant float SPECULAR_POWER = \(specularPower);
    constant float SPECULAR_STRENGTH = \(specularStrength);
    constant float FOG_DENSITY = \(fogDensity);
    constant float SKY_SPREAD = \(skySpread);
    constant float3 SKY_BLUE = \(metal(skyBlue));
    constant float3 DEEP_BLUE = \(metal(deepBlue));
    constant float3 TURQUOISE = \(metal(turquoise));
    constant float3 HAZE = \(metal(haze));
    constant float3 SUN_COLOR = \(metal(sunColor));

    // One thread per pixel, the same math as shadePixel() in Water.swift.
    kernel void water(device uchar4 *pixels [[buffer(0)]],
                      constant Params &p [[buffer(1)]],
                      constant Wave *waves [[buffer(2)]],
                      uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= p.width || gid.y >= p.height) return;
        float aspect = float(p.width) / float(p.height);
        float u = (float(gid.x) + 0.5) / float(p.width);
        float v = (float(gid.y) + 0.5) / float(p.height);
        float3 d = float3((2 * u - 1) * aspect * TAN_HALF_FOV, (1 - 2 * v) * TAN_HALF_FOV - TILT, 1);
        float3 dn = normalize(d);
        float3 color;

        if (d.y >= 0) {
            color = dot(dn, SUN) > SUN_DISK_COSINE
                ? SUN_COLOR
                : mix(HAZE, SKY_BLUE, clamp(dn.y * SKY_SPREAD, 0.0, 1.0));
        } else {
            float s = CAMERA_HEIGHT / -d.y;
            float x = d.x * s, z = d.z * s;
            float dx = 0, dz = 0;
            for (uint i = 0; i < p.waveCount; i++) {
                Wave w = waves[i];
                float c = cos(w.wavenumber * (w.directionX * x + w.directionZ * z) + w.phase);
                dx += w.amplitude * w.wavenumber * w.directionX * c;
                dz += w.amplitude * w.wavenumber * w.directionZ * c;
            }
            float3 n = normalize(float3(-dx, 1, -dz));
            float facing = clamp(dot(n, SUN), 0.0, 1.0);
            float3 halfway = normalize(SUN - dn);
            float glint = pow(clamp(dot(n, halfway), 0.0, 1.0), SPECULAR_POWER) * SPECULAR_STRENGTH;
            float3 water = mix(DEEP_BLUE, TURQUOISE, facing * facing) + glint;
            float fog = 1 - exp(-length(d) * s * FOG_DENSITY);
            color = clamp(mix(water, HAZE, fog), 0.0, 1.0);
        }
        pixels[gid.y * p.width + gid.x] = uchar4(uchar3(round(color * 255)), 255);
    }
    """

enum WaterError: Error, CustomStringConvertible {
    case noMetalDevice
    case kernelCompile(String)
    case outOfMemory(Int)
    case gpu(String)
    case png(String)
    case usage(String)

    var description: String {
        switch self {
        case .noMetalDevice: return "no Metal GPU found"
        case .kernelCompile(let detail): return "could not compile the kernel: \(detail)"
        case .outOfMemory(let bytes): return "could not allocate a \(grouped(bytes))-byte buffer"
        case .gpu(let detail): return "GPU error: \(detail)"
        case .png(let detail): return "could not write PNG: \(detail)"
        case .usage(let detail): return "\(detail)\n\(usageText)"
        }
    }
}

struct RGBA: Equatable, CustomStringConvertible {
    var r, g, b, a: UInt8
    var description: String { "(\(r), \(g), \(b), \(a))" }

    /// Rounds a 0...1 color to bytes the same way the kernel does.
    init(_ c: SIMD3<Float>) {
        func byte(_ v: Float) -> UInt8 { UInt8((v * 255).rounded()) }
        self.init(r: byte(c.x), g: byte(c.y), b: byte(c.z), a: 255)
    }
    init(r: UInt8, g: UInt8, b: UInt8, a: UInt8) { (self.r, self.g, self.b, self.a) = (r, g, b, a) }
}

/// An image whose pixels live in a GPU buffer the CPU can read directly (step 2).
struct GPUImage {
    let width: Int
    let height: Int
    let buffer: MTLBuffer

    var byteCount: Int { width * height * 4 }

    func pixel(x: Int, y: Int) -> RGBA {
        let p = buffer.contents().advanced(by: (y * width + x) * 4).assumingMemoryBound(to: UInt8.self)
        return RGBA(r: p[0], g: p[1], b: p[2], a: p[3])
    }
}

/// The GPU to use. (Needs CoreGraphics linked; the Makefile does that.)
func findDevice() throws -> MTLDevice {
    if let device = MTLCreateSystemDefaultDevice() ?? MTLCopyAllDevices().first {
        return device
    }
    throw WaterError.noMetalDevice
}

func renderWater(width: Int, height: Int, waves: [Wave] = defaultWaves, on device: MTLDevice) throws -> GPUImage {
    let pipeline: MTLComputePipelineState
    do {
        let options = MTLCompileOptions()
        // Precise math, so the GPU's sin/cos/pow stay close to the CPU reference.
        options.mathMode = .safe
        let library = try device.makeLibrary(source: waterKernelSource, options: options)
        guard let function = library.makeFunction(name: "water") else {
            throw WaterError.kernelCompile("no kernel named water")
        }
        pipeline = try device.makeComputePipelineState(function: function)
    } catch let error as WaterError {
        throw error
    } catch {
        throw WaterError.kernelCompile(error.localizedDescription)
    }

    let bytes = width * height * 4
    guard let pixels = device.makeBuffer(length: bytes, options: .storageModeShared) else {
        throw WaterError.outOfMemory(bytes)
    }
    // Metal won't bind an empty buffer, so pass one placeholder wave when there are none.
    let waveData = waves.isEmpty ? [Wave(angleDegrees: 0, wavelength: 1, amplitude: 0)] : waves
    guard let waveBuffer = device.makeBuffer(bytes: waveData, length: MemoryLayout<Wave>.stride * waveData.count,
                                             options: .storageModeShared),
          let queue = device.makeCommandQueue(),
          let commands = queue.makeCommandBuffer(),
          let encoder = commands.makeComputeCommandEncoder()
    else { throw WaterError.gpu("could not set up the GPU work") }

    var params = Params(width: UInt32(width), height: UInt32(height), waveCount: UInt32(waves.count))
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(pixels, offset: 0, index: 0)
    encoder.setBytes(&params, length: MemoryLayout<Params>.stride, index: 1)
    encoder.setBuffer(waveBuffer, offset: 0, index: 2)
    let simdWidth = pipeline.threadExecutionWidth
    let group = MTLSize(width: simdWidth, height: pipeline.maxTotalThreadsPerThreadgroup / simdWidth, depth: 1)
    encoder.dispatchThreads(MTLSize(width: width, height: height, depth: 1), threadsPerThreadgroup: group)
    encoder.endEncoding()
    commands.commit()
    commands.waitUntilCompleted()
    if let error = commands.error { throw WaterError.gpu(error.localizedDescription) }
    return GPUImage(width: width, height: height, buffer: pixels)
}

// MARK: - Saving (as in step 2: the PNG encoder reads the GPU buffer in place)

func savePNG(_ image: GPUImage, to url: URL) throws {
    try? FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    guard let provider = CGDataProvider(dataInfo: nil, data: image.buffer.contents(), size: image.byteCount,
                                        releaseData: { _, _, _ in }),
          let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
          let cgImage = CGImage(width: image.width, height: image.height, bitsPerComponent: 8, bitsPerPixel: 32,
                                bytesPerRow: image.width * 4, space: colorSpace,
                                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                                provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent),
          let destination = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)
    else { throw WaterError.png("could not set up the encoder for \(url.path)") }
    CGImageDestinationAddImage(destination, cgImage, nil)
    guard CGImageDestinationFinalize(destination) else { throw WaterError.png("could not write \(url.path)") }
}

// MARK: - Command line

struct Options: Equatable {
    var width = 1920
    var height = 1080
    var out = "renders/water.png"
}

let usageText = "usage: .build/water [--width N] [--height N] [--out PATH]"

func parseOptions(_ args: [String]) throws -> Options {
    var options = Options()
    var rest = args[...]
    while let flag = rest.popFirst() {
        guard let value = rest.popFirst() else { throw WaterError.usage("\(flag) needs a value") }
        switch flag {
        case "--width", "--height":
            guard let n = Int(value), n > 0, n <= 16384 else {
                throw WaterError.usage("\(flag) must be a whole number from 1 to 16384")
            }
            if flag == "--width" { options.width = n } else { options.height = n }
        case "--out":
            options.out = value
        default:
            throw WaterError.usage("unknown option \(flag)")
        }
    }
    return options
}

/// 2073600 -> "2,073,600", independent of the user's locale.
func grouped(_ n: Int) -> String {
    let digits = String(n)
    var result = ""
    for (i, c) in digits.enumerated() {
        if i > 0 && (digits.count - i) % 3 == 0 { result += "," }
        result.append(c)
    }
    return result
}

func summary(gpuName: String, image: GPUImage, waveCount: Int, savedTo path: String) -> String {
    [
        "GPU: \(gpuName)",
        "Rendered \(image.width) × \(image.height) pixels, one GPU thread per pixel "
            + "(\(grouped(image.width * image.height)) threads, none waiting on another)",
        "Each water pixel adds up \(waveCount) sine waves to find the surface slope, then shades it toward the sun",
        "Saved \(path)",
    ].joined(separator: "\n")
}
