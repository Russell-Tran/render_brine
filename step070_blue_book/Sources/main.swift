// Step 70: a plain blue clothbound hardback, closed, one still, rendered on
// the GPU and saved as a PNG.
//
//   .build/book [width height samples-per-side out]

import Foundation
import simd

let args: [String] = Array(CommandLine.arguments.dropFirst())

func argument(_ n: Int, _ fallback: Int) -> Int {
    guard n < args.count, let v = Int(args[n]) else { return fallback }
    return v
}

let width: Int = argument(0, 1920)
let height: Int = argument(1, 1080)
let samples: Int = argument(2, 4)
let out: String = args.count > 3 ? args[3] : "renders/blue_book.png"

do {
    let device = try findDevice()
    let start = Date()
    let setup: StillSetup = stillSetup()
    let r = try BookRenderer(device: device, width: width, height: height, studio: setup.studio)
    let gpu: Double = try r.render(setup.scene, camera: setup.camera, samples: samples)
    annotate(r.image, setup: setup, renderer: r)
    try savePNG(r.image, to: URL(fileURLWithPath: out))
    let wall: Double = Date().timeIntervalSince(start)
    let L: BookLayout = setup.layout
    print("GPU: \(device.name)")
    print("\(width) × \(height), \(samples * samples) samples per pixel, \(setup.scene.prims.count) primitives")
    print(String(format: "text block %.3f mm (%d leaves × %.4f mm); book %.2f mm thick; squares %.3f mm",
                 L.T, leafCount, paperCaliper, L.top, L.s))
    print(String(format: "%.2f s on the GPU, %.2f s wall", gpu, wall))
    print("Saved \(out)")
} catch {
    FileHandle.standardError.write("book: \(error)\n".data(using: .utf8)!)
    exit(1)
}
