// Step 53: a gold-plated padlock, one still, rendered on the GPU and saved as a PNG.
//
//   .build/lock [width height samples-per-side out]

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
let out: String = args.count > 3 ? args[3] : "renders/gold_lock.png"

do {
    let device = try findDevice()
    let start = Date()
    let (image, gpu) = try renderLock(width: width, height: height, samples: samples, on: device)
    annotate(image)
    try savePNG(image, to: URL(fileURLWithPath: out))
    let wall: Double = Date().timeIntervalSince(start)
    let f0: SIMD3<Double> = reflectanceRGB(goldJC, cosTheta: 1)
    print("GPU: \(device.name)")
    print("\(width) × \(height), \(samples * samples) samples per pixel")
    print(String(format: "gold at normal incidence, linear sRGB (%.3f, %.3f, %.3f), from Johnson & Christy's n,k", f0.x, f0.y, f0.z))
    print(String(format: "inset %.4f µm per pixel; main view %.2f px per mm at the lock", insetMicrometresPerPixel(height: height),
                 mainPixelsPerMillimetre(height: height)))
    print(String(format: "%.2f s on the GPU, %.2f s wall", gpu, wall))
    print("Saved \(out)")
} catch {
    FileHandle.standardError.write("lock: \(error)\n".data(using: .utf8)!)
    exit(1)
}
