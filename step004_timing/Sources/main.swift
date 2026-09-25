// Step 4: time the step 3 water render at several sizes.

import Foundation

do {
    let options = try parseBenchOptions(Array(CommandLine.arguments.dropFirst()))
    let renderer = try WaterRenderer(device: try findDevice())
    try warmUp(renderer, seconds: Double(options.warmupMs) / 1000)
    var results: [SizeResult] = []
    for resolution in benchmarkResolutions {
        results.append(try benchmark(renderer, at: resolution, runs: options.runs))
    }
    print(report(gpuName: renderer.device.name, results: results, warmupMs: options.warmupMs, runs: options.runs))
} catch {
    FileHandle.standardError.write("timing: \(error)\n".data(using: .utf8)!)
    exit(1)
}
