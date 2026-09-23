// Paints a sky-to-deep-blue gradient on the GPU, one thread per pixel, into a
// buffer that the CPU then reads directly.
//
// This is the point of step 2: on Apple silicon the CPU and GPU share the same
// RAM (unified memory). With CUDA you'd cudaMalloc on the device, run the
// kernel, then cudaMemcpy the pixels back to the host. Here the GPU writes into
// a .storageModeShared buffer and the CPU, and even the PNG encoder, read that
// same memory. Nothing is copied between GPU and CPU.

import CoreGraphics
import Foundation
import ImageIO
import Metal
import UniformTypeIdentifiers

struct RGBA: Equatable, CustomStringConvertible {
    var r, g, b, a: UInt8
    var description: String { "(\(r), \(g), \(b), \(a))" }
}

let skyBlue = RGBA(r: 140, g: 200, b: 235, a: 255)
let deepBlue = RGBA(r: 8, g: 40, b: 90, a: 255)

enum PaintError: Error, CustomStringConvertible {
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

// MARK: - The gradient, on the CPU and on the GPU

/// The color of row `y`: sky blue at the top, deep blue at the bottom.
/// Integer math, so the GPU kernel below produces exactly the same bytes.
func gradientPixel(y: Int, height: Int) -> RGBA {
    guard height > 1 else { return skyBlue }
    let d = height - 1
    func mix(_ top: UInt8, _ bottom: UInt8) -> UInt8 {
        UInt8((Int(top) * (d - y) + Int(bottom) * y) / d)
    }
    return RGBA(r: mix(skyBlue.r, deepBlue.r), g: mix(skyBlue.g, deepBlue.g),
                b: mix(skyBlue.b, deepBlue.b), a: mix(skyBlue.a, deepBlue.a))
}

// Must match `Size` in the kernel source.
struct Size {
    var width: UInt32
    var height: UInt32
}

let gradientKernelSource = """
    #include <metal_stdlib>
    using namespace metal;

    struct Size { uint width; uint height; };

    constant int4 TOP = int4(\(skyBlue.r), \(skyBlue.g), \(skyBlue.b), \(skyBlue.a));
    constant int4 BOTTOM = int4(\(deepBlue.r), \(deepBlue.g), \(deepBlue.b), \(deepBlue.a));

    // One thread per pixel. gid is this thread's (x, y) in the image.
    // dispatchThreads never launches threads outside the image on Apple GPUs,
    // so the bounds check only matters if this is dispatched in whole
    // threadgroups (like a CUDA <<<blocks, threads>>> launch).
    kernel void gradient(device uchar4 *pixels [[buffer(0)]],
                         constant Size &size [[buffer(1)]],
                         uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= size.width || gid.y >= size.height) return;
        int d = int(size.height) - 1;
        int y = int(gid.y);
        int4 c = d > 0 ? (TOP * (d - y) + BOTTOM * y) / d : TOP;
        pixels[gid.y * size.width + gid.x] = uchar4(c);
    }
    """

/// An image whose pixels live in a GPU buffer the CPU can read directly.
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

/// The GPU to use: the system default, or else the first GPU Metal lists.
/// (Needs CoreGraphics linked; the Makefile does that.)
func findDevice() throws -> MTLDevice {
    if let device = MTLCreateSystemDefaultDevice() ?? MTLCopyAllDevices().first {
        return device
    }
    throw PaintError.noMetalDevice
}

/// Runs the gradient kernel, writing width × height RGBA pixels at the start of `buffer`.
func paintGradient(into buffer: MTLBuffer, width: Int, height: Int, on device: MTLDevice) throws {
    precondition(buffer.length >= width * height * 4, "buffer too small")
    let pipeline: MTLComputePipelineState
    do {
        let library = try device.makeLibrary(source: gradientKernelSource, options: nil)
        guard let function = library.makeFunction(name: "gradient") else {
            throw PaintError.kernelCompile("no kernel named gradient")
        }
        pipeline = try device.makeComputePipelineState(function: function)
    } catch let error as PaintError {
        throw error
    } catch {
        throw PaintError.kernelCompile(error.localizedDescription)
    }

    guard let queue = device.makeCommandQueue(),
          let commands = queue.makeCommandBuffer(),
          let encoder = commands.makeComputeCommandEncoder()
    else { throw PaintError.gpu("could not create a command encoder") }

    var size = Size(width: UInt32(width), height: UInt32(height))
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(buffer, offset: 0, index: 0)
    encoder.setBytes(&size, length: MemoryLayout<Size>.stride, index: 1)
    // Threadgroups one SIMD group wide (32 threads) and as tall as the kernel allows.
    let simdWidth = pipeline.threadExecutionWidth
    let group = MTLSize(width: simdWidth, height: pipeline.maxTotalThreadsPerThreadgroup / simdWidth, depth: 1)
    encoder.dispatchThreads(MTLSize(width: width, height: height, depth: 1), threadsPerThreadgroup: group)
    encoder.endEncoding()
    commands.commit()
    commands.waitUntilCompleted()
    if let error = commands.error { throw PaintError.gpu(error.localizedDescription) }
}

/// Allocates a shared buffer, paints the gradient into it, and returns it as an image.
func paintGradient(width: Int, height: Int, on device: MTLDevice) throws -> GPUImage {
    let bytes = width * height * 4
    guard let buffer = device.makeBuffer(length: bytes, options: .storageModeShared) else {
        throw PaintError.outOfMemory(bytes)
    }
    try paintGradient(into: buffer, width: width, height: height, on: device)
    return GPUImage(width: width, height: height, buffer: buffer)
}

// MARK: - Saving

/// Writes the image as a PNG. The encoder reads the GPU buffer in place:
/// the data provider wraps the buffer's memory instead of copying it.
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
    else { throw PaintError.png("could not set up the encoder for \(url.path)") }
    CGImageDestinationAddImage(destination, cgImage, nil)
    guard CGImageDestinationFinalize(destination) else { throw PaintError.png("could not write \(url.path)") }
}

// MARK: - Command line

struct Options: Equatable {
    var width = 1920
    var height = 1080
    var out = "renders/gradient.png"
}

let usageText = "usage: .build/paint_gpu [--width N] [--height N] [--out PATH]"

func parseOptions(_ args: [String]) throws -> Options {
    var options = Options()
    var rest = args[...]
    while let flag = rest.popFirst() {
        guard let value = rest.popFirst() else { throw PaintError.usage("\(flag) needs a value") }
        switch flag {
        case "--width", "--height":
            guard let n = Int(value), n > 0, n <= 16384 else {
                throw PaintError.usage("\(flag) must be a whole number from 1 to 16384")
            }
            if flag == "--width" { options.width = n } else { options.height = n }
        case "--out":
            options.out = value
        default:
            throw PaintError.usage("unknown option \(flag)")
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

func summary(gpuName: String, unifiedMemory: Bool, image: GPUImage, storageMode: String, savedTo path: String) -> String {
    [
        "GPU: \(gpuName) (unified memory: \(unifiedMemory ? "yes" : "no"))",
        "Painted \(image.width) × \(image.height) pixels, one GPU thread per pixel "
            + "(\(grouped(image.width * image.height)) threads)",
        "Pixel buffer: \(grouped(image.byteCount)) bytes, storage mode: \(storageMode)",
        "Bytes copied from GPU to CPU: 0 (the CPU read the pixels straight from the buffer the GPU wrote)",
        "Top-left pixel: \(image.pixel(x: 0, y: 0)), bottom-left pixel: \(image.pixel(x: 0, y: image.height - 1))",
        "Saved \(path)",
    ].joined(separator: "\n")
}

func storageModeName(_ mode: MTLStorageMode) -> String {
    switch mode {
    case .shared: return "shared (CPU and GPU use the same memory)"
    case .private: return "private (GPU only)"
    case .memoryless: return "memoryless"
    case .managed: return "managed"
    @unknown default: return "unknown"
    }
}
