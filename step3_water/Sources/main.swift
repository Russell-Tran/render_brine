// Step 3: render a still image of water on the GPU and save it as a PNG.

import Foundation

do {
    let options = try parseOptions(Array(CommandLine.arguments.dropFirst()))
    let device = try findDevice()
    let image = try renderWater(width: options.width, height: options.height, on: device)
    try savePNG(image, to: URL(fileURLWithPath: options.out))
    print(summary(gpuName: device.name, image: image, waveCount: defaultWaves.count, savedTo: options.out))
} catch {
    FileHandle.standardError.write("water: \(error)\n".data(using: .utf8)!)
    exit(1)
}
