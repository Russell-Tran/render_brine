// Step 34: an ant smells and tastes honey — one still, rendered on the GPU, saved as a PNG.
//
//   .build/ant [width height samples-per-side out]

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
let out: String = args.count > 3 ? args[3] : "renders/ant_honey.png"

do {
    let device = try findDevice()
    let scene = try Scene(mutant: .none)
    let start = Date()
    let (image, gpu) = try renderAnt(width: width, height: height, samples: samples, scene: scene, on: device)
    annotate(image, scene: scene)
    try savePNG(image, to: URL(fileURLWithPath: out))
    let wall: Double = Date().timeIntervalSince(start)
    print("GPU: \(device.name)")
    print("\(width) × \(height), \(samples * samples) samples per pixel, \(scene.ant.shapes.count) ant shapes, \(scene.molecule.atoms.count) atoms, \(scene.volatiles.count) odour molecules in the inset")
    print(String(format: "drop %.2f mm across, %.2f mm high, contact angle %.0f°", 2 * dropBaseRadius, dropHeight, dropContactAngle * 180 / Float.pi))
    print(String(format: "ant length %.2f mm; main view %.2f µm per pixel, inset %.3f µm per pixel, molecule %.3f Å per pixel",
                 scene.ant.length, mainMillimetresPerPixel(width: width) * 1000, insetMicrometresPerPixel(height: height),
                 moleculeAngstromsPerPixel(height: height)))
    print(String(format: "%.2f s on the GPU, %.2f s wall", gpu, wall))
    print("Saved \(out)")
} catch {
    FileHandle.standardError.write("ant: \(error)\n".data(using: .utf8)!)
    exit(1)
}
