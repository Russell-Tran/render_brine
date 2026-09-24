// Step 5: render the reflecting sea, then time it at 1, 4 and 16 samples per pixel.

import Foundation

do {
    let device = try findDevice()
    let renderer = try ReflectionRenderer(device: device)
    let (width, height) = (1920, 1080)

    let smooth = try renderer.renderImage(width: width, height: height, samplesPerSide: 4)
    try savePNG(smooth, to: URL(fileURLWithPath: "renders/reflections.png"))
    let rough = try renderer.renderImage(width: width, height: height, samplesPerSide: 1)
    try savePNG(rough, to: URL(fileURLWithPath: "renders/reflections_1spp.png"))
    print("GPU: \(device.name)")
    print("Saved renders/reflections.png (16 samples per pixel) and renders/reflections_1spp.png (1 sample per pixel)")
    print("")

    try warmUpGPU(renderer, seconds: 0.5)
    var timings: [SampleTiming] = []
    for k in [1, 2, 4] {
        let t = try medianGPUTime(renderer, width: width, height: height, samplesPerSide: k, runs: 10)
        timings.append(SampleTiming(samplesPerSide: k, gpuSeconds: t))
    }
    print(timingTable(timings, width: width, height: height))
} catch {
    FileHandle.standardError.write("reflections: \(error)\n".data(using: .utf8)!)
    exit(1)
}
