// Step 77: an ILCA 7 (Laser) sailing dinghy, one still, floating where
// Archimedes puts it, rendered on the GPU and saved as a PNG.
//
//   .build/sailboat [width height samples-per-side out]

import Foundation
import simd

let allArgs: [String] = CommandLine.arguments
let args: [String] = Array(allArgs.dropFirst())

func argument(_ n: Int, _ fallback: Int) -> Int {
    guard n < args.count, let v = Int(args[n]) else { return fallback }
    return v
}

let width: Int = argument(0, 1920)
let height: Int = argument(1, 1080)
let samples: Int = argument(2, 4)
let out: String = args.count > 3 ? args[3] : "renders/sailboat.png"

do {
    let device = try findDevice()
    let start = Date()
    let flo: Flotation = archimedes()
    let drawn: Double = drawnDraught(flo)
    let floated: Double = Date().timeIntervalSince(start)
    let setup: StillSetup = stillSetup()
    let r = try BoatRenderer(device: device, width: width, height: height, setup: setup, draught: drawn)
    let gpu: Double = try r.render(camera: setup.camera, samples: samples)
    annotate(r.image, setup: setup, report: Report(flotation: flo, drawn: drawn))
    try savePNG(r.image, to: URL(fileURLWithPath: out))
    let wall: Double = Date().timeIntervalSince(start)
    let displaced: Double = flo.volume * waterDensity
    print("GPU: \(device.name)")
    print("\(width) × \(height), \(samples * samples) samples per pixel")
    print(String(format: "mass %.1f kg; displaced %.5f m³ = %.2f kg of water at %.1f kg/m³; hull draught %.1f mm (drawn %.1f mm)",
                 flo.mass, flo.volume, displaced, waterDensity, flo.draught * 1000, drawn * 1000))
    print(String(format: "%d columns, %d inside intervals, %.2f s to float", flo.body.columns, flo.body.intervals.count, floated))
    print(String(format: "%.2f s on the GPU, %.2f s wall", gpu, wall))
    print("Saved \(out)")
} catch {
    FileHandle.standardError.write("sailboat: \(error)\n".data(using: .utf8)!)
    exit(1)
}
