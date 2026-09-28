// Step 58: a plastic toy Stegosaurus, one still, rendered on the GPU and
// saved as a PNG.
//
//   .build/toy [width height samples-per-side out]

import Foundation
import simd

let args: [String] = Array(CommandLine.arguments.dropFirst())

func argument(_ n: Int, _ fallback: Int) -> Int {
    guard n < args.count, let v = Int(args[n]) else { return fallback }
    return v
}

let width: Int = argument(0, 1920)
let height: Int = argument(1, 1080)
let samples: Int = argument(2, 5)
let out: String = args.count > 3 ? args[3] : "renders/toy_stegosaurus.png"

do {
    let device = try findDevice()
    let start = Date()
    let setup: StillSetup = stillSetup(stegosaurusDesign())
    let r = try ToyRenderer(device: device, width: width, height: height, paints: setup.design.paints, studio: setup.studio)
    let gpu: Double = try r.render(setup.toy, camera: setup.camera, samples: samples)
    let len = setup.toy.extent(along: SIMD3<Float>(1, 0, 0))
    annotate(r.image, setup: setup, caption: stegoCaption(lengthMM: len.hi - len.lo))
    try savePNG(r.image, to: URL(fileURLWithPath: out))
    let wall: Double = Date().timeIntervalSince(start)
    let ex = setup.toy.extent(along: SIMD3<Float>(1, 0, 0))
    let ey = setup.toy.extent(along: SIMD3<Float>(0, 1, 0))
    let f0: SIMD3<Double> = surfaceF0()
    print("GPU: \(device.name)")
    print("\(width) × \(height), \(samples * samples) samples per pixel")
    print(String(format: "toy %.1f mm long, %.1f mm tall (spec 50.8–76.2 mm long)", ex.hi - ex.lo, ey.hi - ey.lo))
    print(String(format: "PVC n(550 nm) = %.5f → F0 = %.4f (R %.4f, B %.4f)", pvcIndex(nanometres: 550), f0.y, f0.x, f0.z))
    print(String(format: "%.2f s on the GPU, %.2f s wall", gpu, wall))
    print("Saved \(out)")
} catch {
    FileHandle.standardError.write("toy: \(error)\n".data(using: .utf8)!)
    exit(1)
}
