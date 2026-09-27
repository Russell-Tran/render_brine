// Step 22: step 20's still — same camera, same light — with a round electric
// toothbrush head pressed on #21 at the gum line, rendered on the GPU and
// saved as a PNG.
//
//   .build/brush [width height samples-per-side out]

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
let out: String = args.count > 3 ? args[3] : "renders/brush.png"

do {
    let device = try findDevice()
    let start = Date()
    let probe = try SceneProbe(device: device)
    let brush: Brush = try pressedBrush(probe: probe)
    let cast: Double = Date().timeIntervalSince(start)
    let (image, gpu) = try renderMouth(width: width, height: height, samples: samples,
                                       extra: brushExtra(brush), on: device)
    try savePNG(image, to: URL(fileURLWithPath: out))
    let wall: Double = Date().timeIntervalSince(start)
    let lengths: [Float] = brush.tufts.map { $0.length }
    print("GPU: \(device.name)")
    print(String(format: "%d tufts cast onto #21 in %.2f s: %.2f–%.2f mm long, rest length %.1f mm",
                 brush.tufts.count, cast, lengths.min()!, lengths.max()!, restLength))
    print("\(width) × \(height), \(samples * samples) samples per pixel")
    print(String(format: "%.2f s on the GPU, %.2f s wall", gpu, wall))
    print("Saved \(out)")
} catch {
    FileHandle.standardError.write("brush: \(error)\n".data(using: .utf8)!)
    exit(1)
}
