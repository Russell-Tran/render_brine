// Step 25: self-pollination in the common bean, one still, rendered on the GPU
// and saved as a PNG.
//
//   .build/bean [width height samples-per-side out]

import Foundation

let args: [String] = Array(CommandLine.arguments.dropFirst())

/// The nth argument as a whole number, or a default.
func argument(_ n: Int, _ fallback: Int) -> Int {
    guard n < args.count, let v = Int(args[n]) else { return fallback }
    return v
}

let width: Int = argument(0, 1920)
let height: Int = argument(1, 1080)
let samples: Int = argument(2, 3)
let out: String = args.count > 3 ? args[3] : "renders/bean.png"

do {
    let device = try findDevice()
    let start = Date()
    let scene = try BeanScene(FlowerModel(.none), on: device)
    let result = try renderBean(scene, width: width, height: height, samples: samples)
    try savePNG(result.bytes, width: width, height: height, to: URL(fileURLWithPath: out))
    let wall: Double = Date().timeIntervalSince(start)
    print("GPU: \(device.name)")
    print("\(width) × \(height), \(samples * samples) samples per pixel; inset \(magnificationLabel(width: width, height: height))")
    print(String(format: "%.2f s on the GPU, %.2f s wall", result.gpu, wall))
    print("Saved \(out)")
} catch {
    FileHandle.standardError.write("bean: \(error)\n".data(using: .utf8)!)
    exit(1)
}
