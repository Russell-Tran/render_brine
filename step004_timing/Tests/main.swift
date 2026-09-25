// Tests for step 4. Unit tests check the statistics and the report with
// made-up timings; GPU tests check that the timed renderer draws exactly what
// step 3 draws, and that the timings it returns make sense.

import Foundation
import Metal

func fakeResult(_ name: String, _ width: Int, _ height: Int, gpuMs: [Double]) -> SizeResult {
    SizeResult(resolution: Resolution(name: name, width: width, height: height),
               gpuSeconds: gpuMs.map { $0 / 1000 },
               wallSeconds: gpuMs.map { ($0 + 0.2) / 1000 })
}

func sameBytes(_ a: GPUImage, _ b: GPUImage) -> Bool {
    a.width == b.width && a.height == b.height
        && memcmp(a.buffer.contents(), b.buffer.contents(), a.byteCount) == 0
}

section("statistics")
test("median of an odd count is the middle value") {
    expectEqual(median([3, 1, 2]), 2)
    expectEqual(median([5]), 5)
}
test("median of an even count averages the middle two") {
    expectEqual(median([4, 1, 3, 2]), 2.5)
}
test("median ignores one very slow run") {
    expectEqual(median([1, 1, 1, 1, 100]), 1)
}
test("throughput is pixels per second, in millions") {
    let mps: Double = megapixelsPerSecond(pixels: 2_073_600, seconds: 0.002)
    expect(abs(mps - 1036.8) < 1e-9, "got \(mps)")
}

section("report")
test("the report has one row per size with median, fastest and throughput") {
    let results = [
        fakeResult("1080p", 1920, 1080, gpuMs: [2.0, 2.2, 1.9]),
        fakeResult("4K", 3840, 2160, gpuMs: [7.8, 8.1, 8.0]),
    ]
    let text = report(gpuName: "Test GPU", results: results, warmupMs: 500, runs: 3)
    for line in [
        "GPU: Test GPU",
        "Warmed up with 500 ms of rendering, then 3 timed runs per size.",
        "1080p  2,073,600   2.000 ms    1.900 ms    2.200 ms    1037 Mpixel/s",
        "4K     8,294,400   8.000 ms    7.800 ms    8.200 ms    1037 Mpixel/s",
    ] {
        expect(text.contains(line), "missing line: \(line)\n\(text)")
    }
}
test("the scaling line compares pixels and time") {
    let small = fakeResult("1080p", 1920, 1080, gpuMs: [2.0])
    let large = fakeResult("4K", 3840, 2160, gpuMs: [7.8])
    expectEqual(scalingLine(from: small, to: large), "4K vs 1080p: 4.0× the pixels took 3.9× the time")
}
test("the benchmark sizes go from small to large") {
    let pixels = benchmarkResolutions.map { $0.pixels }
    expectEqual(pixels, pixels.sorted())
    expectEqual(benchmarkResolutions.map { $0.name }, ["tiny", "1080p", "4K"])
}

section("parseBenchOptions")
test("defaults to 500 ms of warm-up and 20 timed runs") {
    expectEqual(try parseBenchOptions([]), BenchOptions(warmupMs: 500, runs: 20))
}
test("reads --warmup-ms and --runs") {
    expectEqual(try parseBenchOptions(["--warmup-ms", "0", "--runs", "3"]), BenchOptions(warmupMs: 0, runs: 3))
}
test("rejects bad input") {
    expectThrows { _ = try parseBenchOptions(["--runs", "0"]) }
    expectThrows { _ = try parseBenchOptions(["--warmup-ms", "-1"]) }
    expectThrows { _ = try parseBenchOptions(["--warmup", "5"]) }
    expectThrows { _ = try parseBenchOptions(["--runs"]) }
    expectThrows { _ = try parseBenchOptions(["--size", "4K"]) }
}

section("GPU (this Mac)")
test("the timed renderer draws exactly what step 3 draws") {
    let device = try findDevice()
    let renderer = try WaterRenderer(device: device)
    for (w, h) in [(64, 36), (33, 17)] {
        let timed = try renderer.renderImage(width: w, height: h)
        let step3 = try renderWater(width: w, height: h, on: device)
        expect(sameBytes(timed, step3), "\(w) × \(h) differs from step 3's render")
    }
}
test("a benchmark returns one positive timing per run") {
    let renderer = try WaterRenderer(device: try findDevice())
    let r = try benchmark(renderer, at: Resolution(name: "small", width: 64, height: 36), runs: 4)
    expectEqual(r.gpuSeconds.count, 4)
    expectEqual(r.wallSeconds.count, 4)
    expect(r.gpuSeconds.allSatisfy { $0 > 0 }, "GPU times: \(r.gpuSeconds)")
}
test("GPU time fits inside the wall-clock time") {
    let renderer = try WaterRenderer(device: try findDevice())
    let r = try benchmark(renderer, at: Resolution(name: "1080p", width: 1920, height: 1080), runs: 5)
    for (gpu, wall) in zip(r.gpuSeconds, r.wallSeconds) {
        // Half a millisecond of slack for the two clocks being read at different moments.
        expect(gpu <= wall + 0.0005, "GPU \(milliseconds(gpu)) > wall \(milliseconds(wall))")
    }
}
test("warming up keeps rendering for at least the requested time") {
    let renderer = try WaterRenderer(device: try findDevice())
    let start = DispatchTime.now().uptimeNanoseconds
    let frames = try warmUp(renderer, seconds: 0.1)
    let elapsed = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
    expect(elapsed >= 0.1, "stopped after \(elapsed) s")
    expect(frames > 1, "rendered only \(frames) frame(s)")
}
test("a bigger image takes longer") {
    let renderer = try WaterRenderer(device: try findDevice())
    let small = try benchmark(renderer, at: Resolution(name: "small", width: 64, height: 36), runs: 5)
    let big = try benchmark(renderer, at: Resolution(name: "1080p", width: 1920, height: 1080), runs: 5)
    let smallMs: Double = median(small.gpuSeconds) * 1000
    let bigMs: Double = median(big.gpuSeconds) * 1000
    expect(bigMs > smallMs, "1080p took \(bigMs) ms, 64 × 36 took \(smallMs) ms")
}

finish()
