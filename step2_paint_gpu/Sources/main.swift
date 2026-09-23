// Step 2: paint one image on the GPU and save it as a PNG.

import Foundation

do {
    let options = try parseOptions(Array(CommandLine.arguments.dropFirst()))
    let device = try findDevice()
    let image = try paintGradient(width: options.width, height: options.height, on: device)
    try savePNG(image, to: URL(fileURLWithPath: options.out))
    print(summary(gpuName: device.name, unifiedMemory: device.hasUnifiedMemory, image: image,
                  storageMode: storageModeName(image.buffer.storageMode), savedTo: options.out))
} catch {
    FileHandle.standardError.write("paint_gpu: \(error)\n".data(using: .utf8)!)
    exit(1)
}
