// Times step 3's water render on the GPU's own clock.
//
// This file is compiled together with ../step3_water/Sources/Water.swift, so
// it times exactly the same kernel (waterKernelSource) with the same waves.
//
// How the timing works, and why:
//  - The kernel is compiled once, and the pixel buffer is allocated once per
//    size, so we time only the rendering.
//  - The GPU runs at a low clock speed when idle and only speeds up once it's
//    kept busy. On the M4 mini a 4K frame took ~3.7 ms from idle and ~1.9 ms
//    after ~200 ms of steady work. So before timing anything, we render
//    continuously for a set time (500 ms by default), and do a few untimed
//    runs at each new size.
//  - Each timed run is one command buffer. Its gpuStartTime and gpuEndTime say
//    when the GPU actually started and finished it, which excludes the time
//    the CPU spent setting it up.
//  - The median of the runs is reported, so a stray slow run doesn't skew it.

import Foundation
import Metal

struct Resolution: Equatable {
    let name: String
    let width: Int
    let height: Int
    var pixels: Int { width * height }
}

let benchmarkResolutions = [
    Resolution(name: "tiny", width: 256, height: 144),
    Resolution(name: "1080p", width: 1920, height: 1080),
    Resolution(name: "4K", width: 3840, height: 2160),
]

/// Step 3's water kernel, compiled once and reused for every run.
final class WaterRenderer {
    let device: MTLDevice
    private let pipeline: MTLComputePipelineState
    private let queue: MTLCommandQueue
    private let waveBuffer: MTLBuffer
    private let waveCount: Int

    init(device: MTLDevice, waves: [Wave] = defaultWaves) throws {
        self.device = device
        do {
            // Same compile options as step 3's renderWater().
            let options = MTLCompileOptions()
            options.fastMathEnabled = false
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

    /// Renders once and returns how long it took, in seconds: `gpu` by the
    /// GPU's clock, `wall` from submitting the work to getting it back.
    func render(into pixels: MTLBuffer, width: Int, height: Int) throws -> (gpu: Double, wall: Double) {
        guard let commands = queue.makeCommandBuffer(), let encoder = commands.makeComputeCommandEncoder() else {
            throw WaterError.gpu("could not create a command encoder")
        }
        var params = Params(width: UInt32(width), height: UInt32(height), waveCount: UInt32(waveCount))
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(pixels, offset: 0, index: 0)
        encoder.setBytes(&params, length: MemoryLayout<Params>.stride, index: 1)
        encoder.setBuffer(waveBuffer, offset: 0, index: 2)
        let simdWidth = pipeline.threadExecutionWidth
        let group = MTLSize(width: simdWidth, height: pipeline.maxTotalThreadsPerThreadgroup / simdWidth, depth: 1)
        encoder.dispatchThreads(MTLSize(width: width, height: height, depth: 1), threadsPerThreadgroup: group)
        encoder.endEncoding()

        let start = DispatchTime.now().uptimeNanoseconds
        commands.commit()
        commands.waitUntilCompleted()
        let wall = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
        if let error = commands.error { throw WaterError.gpu(error.localizedDescription) }
        return (commands.gpuEndTime - commands.gpuStartTime, wall)
    }

    func renderImage(width: Int, height: Int) throws -> GPUImage {
        let pixels = try makePixelBuffer(width: width, height: height)
        _ = try render(into: pixels, width: width, height: height)
        return GPUImage(width: width, height: height, buffer: pixels)
    }
}

struct SizeResult {
    let resolution: Resolution
    let gpuSeconds: [Double]
    let wallSeconds: [Double]
}

/// Renders 1080p frames back to back for at least `seconds`, so the GPU is at
/// full clock speed before timing starts. Returns how many frames it rendered.
@discardableResult
func warmUp(_ renderer: WaterRenderer, seconds: Double) throws -> Int {
    let pixels = try renderer.makePixelBuffer(width: 1920, height: 1080)
    let start = DispatchTime.now().uptimeNanoseconds
    var frames = 0
    while Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9 < seconds {
        _ = try renderer.render(into: pixels, width: 1920, height: 1080)
        frames += 1
    }
    return frames
}

/// Untimed runs at each new size before its timed runs.
let runsBeforeTiming = 3

func benchmark(_ renderer: WaterRenderer, at resolution: Resolution, runs: Int) throws -> SizeResult {
    let pixels = try renderer.makePixelBuffer(width: resolution.width, height: resolution.height)
    for _ in 0..<runsBeforeTiming {
        _ = try renderer.render(into: pixels, width: resolution.width, height: resolution.height)
    }
    var gpu: [Double] = []
    var wall: [Double] = []
    for _ in 0..<runs {
        let t = try renderer.render(into: pixels, width: resolution.width, height: resolution.height)
        gpu.append(t.gpu)
        wall.append(t.wall)
    }
    return SizeResult(resolution: resolution, gpuSeconds: gpu, wallSeconds: wall)
}

// MARK: - Statistics and the report

func median(_ values: [Double]) -> Double {
    precondition(!values.isEmpty, "median of nothing")
    let sorted = values.sorted()
    let mid = sorted.count / 2
    return sorted.count % 2 == 1 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2
}

/// Millions of pixels rendered per second.
func megapixelsPerSecond(pixels: Int, seconds: Double) -> Double {
    Double(pixels) / seconds / 1e6
}

func milliseconds(_ seconds: Double) -> String {
    String(format: "%.3f ms", seconds * 1000)
}

func pad(_ s: String, _ width: Int) -> String {
    s.count >= width ? s : s + String(repeating: " ", count: width - s.count)
}

/// "4K vs 1080p: 4.0× the pixels took 3.9× the time"
func scalingLine(from small: SizeResult, to large: SizeResult) -> String {
    let pixelRatio = Double(large.resolution.pixels) / Double(small.resolution.pixels)
    let timeRatio = median(large.gpuSeconds) / median(small.gpuSeconds)
    return String(format: "%@ vs %@: %.1f× the pixels took %.1f× the time",
                  large.resolution.name, small.resolution.name, pixelRatio, timeRatio)
}

func report(gpuName: String, results: [SizeResult], warmupMs: Int, runs: Int) -> String {
    var lines = [
        "GPU: \(gpuName)",
        "Warmed up with \(warmupMs) ms of rendering, then \(runs) timed runs per size. "
            + "Times are medians, by the GPU's own clock.",
        "",
        pad("size", 7) + pad("pixels", 12) + pad("GPU time", 12) + pad("fastest", 12)
            + pad("wall time", 12) + "throughput",
    ]
    for r in results {
        let gpu = median(r.gpuSeconds)
        lines.append(pad(r.resolution.name, 7)
            + pad(grouped(r.resolution.pixels), 12)
            + pad(milliseconds(gpu), 12)
            + pad(milliseconds(r.gpuSeconds.min()!), 12)
            + pad(milliseconds(median(r.wallSeconds)), 12)
            + String(format: "%.0f Mpixel/s", megapixelsPerSecond(pixels: r.resolution.pixels, seconds: gpu)))
    }
    if results.count >= 2 {
        lines.append("")
        for i in 1..<results.count {
            lines.append(scalingLine(from: results[i - 1], to: results[i]))
        }
    }
    return lines.joined(separator: "\n")
}

// MARK: - Command line

struct BenchOptions: Equatable {
    var warmupMs = 500
    var runs = 20
}

let benchUsage = "usage: .build/timing [--warmup-ms N] [--runs N]"

func parseBenchOptions(_ args: [String]) throws -> BenchOptions {
    var options = BenchOptions()
    var rest = args[...]
    while let flag = rest.popFirst() {
        guard let value = rest.popFirst() else { throw WaterError.usage("\(flag) needs a value\n\(benchUsage)") }
        switch flag {
        case "--warmup-ms":
            guard let n = Int(value), n >= 0, n <= 60000 else {
                throw WaterError.usage("--warmup-ms must be a whole number from 0 to 60000")
            }
            options.warmupMs = n
        case "--runs":
            guard let n = Int(value), n >= 1, n <= 1000 else {
                throw WaterError.usage("--runs must be a whole number from 1 to 1000")
            }
            options.runs = n
        default:
            throw WaterError.usage("unknown option \(flag)\n\(benchUsage)")
        }
    }
    return options
}
