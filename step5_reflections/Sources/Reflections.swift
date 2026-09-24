// Step 3's sea, now reflecting the sky, with smoother edges.
//
// Compiled together with ../step3_water/Sources/Water.swift, which provides the
// waves, camera, colors, waveNormal(), GPUImage, findDevice() and savePNG().
//
// What's new compared with step 3:
//  - Reflection. Each water pixel bounces its ray off the tilted surface and
//    looks up the sky in that direction (including the sun, which now makes
//    the glints instead of step 3's separate glint term).
//  - Fresnel. Water reflects about 2% of light when you look straight down and
//    nearly all of it at a glancing angle. That fraction blends the reflection
//    with the water's own color, so the sea gets brighter toward the horizon.
//  - Several samples per pixel. Each pixel is split into a k × k grid and the
//    colors are averaged, which smooths the jagged edges step 3 had.

import CoreGraphics
import Foundation
import Metal
import simd

/// How much brighter the sun disk is than the sky, so its reflection makes
/// glints even where the water reflects only a little.
let sunBrightness: Float = 6
/// Water reflects this fraction of light when viewed straight on.
let fresnelF0: Float = 0.02

// MARK: - CPU reference

/// Schlick's approximation of the Fresnel effect. `cosine` is the cosine of
/// the angle between the view ray and the surface normal: 1 looking straight
/// down, 0 at a grazing angle.
func fresnel(cosine: Float) -> Float {
    let m: Float = 1 - cosine
    let m5: Float = m * m * m * m * m
    return fresnelF0 + (1 - fresnelF0) * m5
}

/// The sky's color in a direction (a unit vector), including the bright sun disk.
func skyRadiance(_ direction: SIMD3<Float>) -> SIMD3<Float> {
    if simd_dot(direction, sunDirection) > sunDiskCosine { return sunColor * sunBrightness }
    let t: Float = simd_clamp(direction.y * skySpread, 0, 1)
    return simd_mix(haze, skyBlue, SIMD3(repeating: t))
}

/// Mirror `ray` off a surface with normal `normal`. If a steep wave would send
/// the reflection into the water, flip it back up (a common cheap fix).
func reflectOffWater(_ ray: SIMD3<Float>, normal: SIMD3<Float>) -> SIMD3<Float> {
    var r = simd_reflect(ray, normal)
    if r.y < 0 { r.y = -r.y }
    return simd_normalize(r)
}

/// Where the samples go inside a pixel: a k × k grid of cell centers, each
/// coordinate between 0 and 1.
func sampleOffsets(perSide k: Int) -> [SIMD2<Float>] {
    var offsets: [SIMD2<Float>] = []
    for j in 0..<k {
        for i in 0..<k {
            let x: Float = (Float(i) + 0.5) / Float(k)
            let y: Float = (Float(j) + 0.5) / Float(k)
            offsets.append(SIMD2(x, y))
        }
    }
    return offsets
}

/// The color seen through image position (sx, sy), measured in pixels from
/// the top-left corner. Channels are 0...1.
func shadeReflectionSample(sx: Float, sy: Float, width: Int, height: Int, waves: [Wave]) -> SIMD3<Float> {
    let aspect: Float = Float(width) / Float(height)
    let u: Float = sx / Float(width)
    let v: Float = sy / Float(height)
    let dx: Float = (2 * u - 1) * aspect * tanHalfFOV
    let dy: Float = (1 - 2 * v) * tanHalfFOV - tilt
    let d = SIMD3<Float>(dx, dy, 1)
    let dn = simd_normalize(d)
    let one = SIMD3<Float>(repeating: 1)

    if d.y >= 0 { return simd_min(skyRadiance(dn), one) }

    let s: Float = cameraHeight / -d.y
    let n = waveNormal(x: d.x * s, z: d.z * s, waves: waves)
    let facing: Float = simd_clamp(simd_dot(n, sunDirection), 0, 1)
    let body = simd_mix(deepBlue, turquoise, SIMD3(repeating: facing * facing))
    let reflected = skyRadiance(reflectOffWater(dn, normal: n))
    let f: Float = fresnel(cosine: simd_clamp(simd_dot(n, -dn), 0, 1))
    let water = simd_mix(body, reflected, SIMD3(repeating: f))
    let distance: Float = simd_length(d) * s
    let fog: Float = 1 - exp(-distance * fogDensity)
    return simd_clamp(simd_mix(water, haze, SIMD3(repeating: fog)), .zero, one)
}

/// The color of pixel (px, py): the average of its k × k samples.
func shadeReflectionPixel(px: Int, py: Int, width: Int, height: Int, waves: [Wave],
                          samplesPerSide k: Int) -> SIMD3<Float> {
    var sum = SIMD3<Float>(repeating: 0)
    for o in sampleOffsets(perSide: k) {
        sum += shadeReflectionSample(sx: Float(px) + o.x, sy: Float(py) + o.y,
                                     width: width, height: height, waves: waves)
    }
    return sum / Float(k * k)
}

// MARK: - GPU

// Must match `ReflectParams` in the kernel.
struct ReflectParams {
    var width: UInt32
    var height: UInt32
    var waveCount: UInt32
    var samplesPerSide: UInt32
}

let reflectionKernelSource = """
    #include <metal_stdlib>
    using namespace metal;

    struct ReflectParams { uint width; uint height; uint waveCount; uint samplesPerSide; };
    struct Wave { float directionX; float directionZ; float wavenumber; float amplitude; float phase; };

    constant float CAMERA_HEIGHT = \(cameraHeight);
    constant float TAN_HALF_FOV = \(tanHalfFOV);
    constant float TILT = \(tilt);
    constant float3 SUN = \(metal(sunDirection));
    constant float SUN_DISK_COSINE = \(sunDiskCosine);
    constant float SUN_BRIGHTNESS = \(sunBrightness);
    constant float FRESNEL_F0 = \(fresnelF0);
    constant float FOG_DENSITY = \(fogDensity);
    constant float SKY_SPREAD = \(skySpread);
    constant float3 SKY_BLUE = \(metal(skyBlue));
    constant float3 DEEP_BLUE = \(metal(deepBlue));
    constant float3 TURQUOISE = \(metal(turquoise));
    constant float3 HAZE = \(metal(haze));
    constant float3 SUN_COLOR = \(metal(sunColor));

    float fresnel(float cosine) {
        float m = 1 - cosine;
        return FRESNEL_F0 + (1 - FRESNEL_F0) * m * m * m * m * m;
    }

    float3 skyRadiance(float3 direction) {
        if (dot(direction, SUN) > SUN_DISK_COSINE) return SUN_COLOR * SUN_BRIGHTNESS;
        return mix(HAZE, SKY_BLUE, clamp(direction.y * SKY_SPREAD, 0.0, 1.0));
    }

    // The same math as shadeReflectionSample() in Reflections.swift.
    float3 shadeSample(float sx, float sy, constant ReflectParams &p, constant Wave *waves) {
        float aspect = float(p.width) / float(p.height);
        float u = sx / float(p.width);
        float v = sy / float(p.height);
        float3 d = float3((2 * u - 1) * aspect * TAN_HALF_FOV, (1 - 2 * v) * TAN_HALF_FOV - TILT, 1);
        float3 dn = normalize(d);
        if (d.y >= 0) return min(skyRadiance(dn), float3(1));

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
        float3 body = mix(DEEP_BLUE, TURQUOISE, facing * facing);
        float3 r = reflect(dn, n);
        if (r.y < 0) r.y = -r.y;
        float3 reflected = skyRadiance(normalize(r));
        float f = fresnel(clamp(dot(n, -dn), 0.0, 1.0));
        float3 water = mix(body, reflected, f);
        float fog = 1 - exp(-length(d) * s * FOG_DENSITY);
        return clamp(mix(water, HAZE, fog), 0.0, 1.0);
    }

    // One thread per pixel; each thread averages k × k samples inside its pixel.
    kernel void reflections(device uchar4 *pixels [[buffer(0)]],
                            constant ReflectParams &p [[buffer(1)]],
                            constant Wave *waves [[buffer(2)]],
                            uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= p.width || gid.y >= p.height) return;
        uint k = p.samplesPerSide;
        float3 sum = float3(0);
        for (uint j = 0; j < k; j++) {
            for (uint i = 0; i < k; i++) {
                float sx = float(gid.x) + (float(i) + 0.5) / float(k);
                float sy = float(gid.y) + (float(j) + 0.5) / float(k);
                sum += shadeSample(sx, sy, p, waves);
            }
        }
        float3 color = sum / float(k * k);
        pixels[gid.y * p.width + gid.x] = uchar4(uchar3(round(color * 255)), 255);
    }
    """

/// The reflection kernel, compiled once and reused.
final class ReflectionRenderer {
    let device: MTLDevice
    private let pipeline: MTLComputePipelineState
    private let queue: MTLCommandQueue
    private let waveBuffer: MTLBuffer
    private let waveCount: Int

    init(device: MTLDevice, waves: [Wave] = defaultWaves) throws {
        self.device = device
        do {
            let options = MTLCompileOptions()
            // Precise math, as in step 3 (newer SDKs call this mathMode = .safe).
            options.fastMathEnabled = false
            let library = try device.makeLibrary(source: reflectionKernelSource, options: options)
            guard let function = library.makeFunction(name: "reflections") else {
                throw WaterError.kernelCompile("no kernel named reflections")
            }
            pipeline = try device.makeComputePipelineState(function: function)
        } catch let error as WaterError {
            throw error
        } catch {
            throw WaterError.kernelCompile(error.localizedDescription)
        }
        // Metal won't bind an empty buffer, so pass one placeholder wave when there are none.
        let waveData = waves.isEmpty ? [Wave(angleDegrees: 0, wavelength: 1, amplitude: 0)] : waves
        guard let queue = device.makeCommandQueue(),
              let waveBuffer = device.makeBuffer(bytes: waveData, length: MemoryLayout<Wave>.stride * waveData.count,
                                                 options: .storageModeShared)
        else { throw WaterError.gpu("could not set up the GPU work") }
        self.queue = queue
        self.waveBuffer = waveBuffer
        self.waveCount = waves.count
    }

    func makePixelBuffer(width: Int, height: Int) throws -> MTLBuffer {
        let bytes = width * height * 4
        guard let buffer = device.makeBuffer(length: bytes, options: .storageModeShared) else {
            throw WaterError.outOfMemory(bytes)
        }
        return buffer
    }

    /// Renders into `pixels` and returns the GPU time in seconds (by the GPU's clock).
    @discardableResult
    func render(into pixels: MTLBuffer, width: Int, height: Int, samplesPerSide: Int) throws -> Double {
        guard let commands = queue.makeCommandBuffer(), let encoder = commands.makeComputeCommandEncoder() else {
            throw WaterError.gpu("could not create a command encoder")
        }
        var params = ReflectParams(width: UInt32(width), height: UInt32(height),
                                   waveCount: UInt32(waveCount), samplesPerSide: UInt32(samplesPerSide))
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(pixels, offset: 0, index: 0)
        encoder.setBytes(&params, length: MemoryLayout<ReflectParams>.stride, index: 1)
        encoder.setBuffer(waveBuffer, offset: 0, index: 2)
        let simdWidth = pipeline.threadExecutionWidth
        let group = MTLSize(width: simdWidth, height: pipeline.maxTotalThreadsPerThreadgroup / simdWidth, depth: 1)
        encoder.dispatchThreads(MTLSize(width: width, height: height, depth: 1), threadsPerThreadgroup: group)
        encoder.endEncoding()
        commands.commit()
        commands.waitUntilCompleted()
        if let error = commands.error { throw WaterError.gpu(error.localizedDescription) }
        return commands.gpuEndTime - commands.gpuStartTime
    }

    func renderImage(width: Int, height: Int, samplesPerSide: Int) throws -> GPUImage {
        let pixels = try makePixelBuffer(width: width, height: height)
        try render(into: pixels, width: width, height: height, samplesPerSide: samplesPerSide)
        return GPUImage(width: width, height: height, buffer: pixels)
    }
}

// MARK: - Timing (as in step 4)

/// Renders continuously for `seconds` so the GPU reaches full clock speed.
func warmUpGPU(_ renderer: ReflectionRenderer, seconds: Double) throws {
    let pixels = try renderer.makePixelBuffer(width: 1920, height: 1080)
    let start = DispatchTime.now().uptimeNanoseconds
    while Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9 < seconds {
        try renderer.render(into: pixels, width: 1920, height: 1080, samplesPerSide: 1)
    }
}

/// Median GPU time, in seconds, of `runs` renders after 3 untimed ones.
func medianGPUTime(_ renderer: ReflectionRenderer, width: Int, height: Int, samplesPerSide: Int,
                   runs: Int) throws -> Double {
    let pixels = try renderer.makePixelBuffer(width: width, height: height)
    for _ in 0..<3 {
        try renderer.render(into: pixels, width: width, height: height, samplesPerSide: samplesPerSide)
    }
    var times: [Double] = []
    for _ in 0..<runs {
        times.append(try renderer.render(into: pixels, width: width, height: height, samplesPerSide: samplesPerSide))
    }
    let sorted = times.sorted()
    let mid = sorted.count / 2
    return sorted.count % 2 == 1 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2
}

struct SampleTiming {
    let samplesPerSide: Int
    let gpuSeconds: Double
    var samplesPerPixel: Int { samplesPerSide * samplesPerSide }
}

func timingTable(_ timings: [SampleTiming], width: Int, height: Int) -> String {
    var lines = ["\(width) × \(height), median GPU time of 10 runs after a 500 ms warm-up:"]
    guard let base = timings.first else { return lines[0] }
    for t in timings {
        let ms: Double = t.gpuSeconds * 1000
        let work: Double = Double(t.samplesPerPixel) / Double(base.samplesPerPixel)
        let time: Double = t.gpuSeconds / base.gpuSeconds
        let name = t.samplesPerPixel == 1 ? "1 sample per pixel:" : "\(t.samplesPerPixel) samples per pixel:"
        let label = name.padding(toLength: 21, withPad: " ", startingAt: 0)
        lines.append(String(format: "  %@ %7.3f ms  (%.0f× the work, %.1f× the time)", label, ms, work, time))
    }
    return lines.joined(separator: "\n")
}
