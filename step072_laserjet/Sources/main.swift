// Step 72: an HP LaserJet P1102w, one still, rendered on the GPU and saved
// as a PNG, with an inset of how it prints.
//
//   .build/printer [width height samples-per-side out]

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
let out: String = args.count > 3 ? args[3] : "renders/laserjet.png"

do {
    let device = try findDevice()
    let start = Date()
    let setup: StillSetup = stillSetup()
    let r = try PrinterRenderer(device: device, width: width, height: height, studio: setup.studio)
    let gpu: Double = try r.render(camera: setup.camera, samples: samples)
    let inset: InsetContent = buildInset()
    annotate(r.image, setup: setup, inset: inset)
    try savePNG(r.image, to: URL(fileURLWithPath: out))
    let wall: Double = Date().timeIntervalSince(start)
    print("GPU: \(device.name)")
    print("\(width) × \(height), \(samples * samples) samples per pixel")
    print(String(format: "printer %.0f × %.0f × %.0f mm; dots %.2f µm apart; %d toner dots in the inset",
                 drawnWidth(), drawnDepth(), drawnHeight(), inset.pitch, inset.dots.count))
    print(String(format: "%.2f s on the GPU, %.2f s wall", gpu, wall))
    print("Saved \(out)")
} catch {
    FileHandle.standardError.write("printer: \(error)\n".data(using: .utf8)!)
    exit(1)
}
