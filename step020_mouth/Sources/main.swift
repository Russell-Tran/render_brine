// Step 20: one still of the lower arch, rendered on the GPU and saved as a PNG.
//
//   .build/mouth [width height samples-per-side out]

import Foundation

let args: [String] = Array(CommandLine.arguments.dropFirst())

/// The nth argument as a whole number, or a default.
func argument(_ n: Int, _ fallback: Int) -> Int {
    guard n < args.count, let v = Int(args[n]) else { return fallback }
    return v
}

let width: Int = argument(0, 1920)
let height: Int = argument(1, 1080)
let samples: Int = argument(2, 4)
let out: String = args.count > 3 ? args[3] : "renders/mouth.png"

do {
    let device = try findDevice()
    let start = Date()
    let (image, gpu) = try renderMouth(width: width, height: height, samples: samples, on: device)
    try savePNG(image, to: URL(fileURLWithPath: out))
    let wall: Double = Date().timeIntervalSince(start)
    print("GPU: \(device.name)")
    print("\(width) × \(height), \(samples * samples) samples per pixel, 14 teeth")
    print(String(format: "%.2f s on the GPU, %.2f s wall", gpu, wall))
    print("Saved \(out)")
} catch {
    FileHandle.standardError.write("mouth: \(error)\n".data(using: .utf8)!)
    exit(1)
}
