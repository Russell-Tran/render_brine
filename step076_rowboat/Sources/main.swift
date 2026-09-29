// Step 76: a Whitehall pulling boat, one still, floating where Archimedes
// puts it, rendered on the GPU and saved as a PNG.
//
//   .build/rowboat [width height samples-per-side out]

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
let out: String = args.count > 3 ? args[3] : "renders/rowboat.png"

do {
    let device = try findDevice()
    let start = Date()
    let s: Float = boatScale()
    let buoy: Buoyancy = solveBuoyancy(scale: s)
    let draft: Float = drawnDraft(buoy)
    let layout: Layout = buildLayout(scale: s, drawnDraft: draft)
    let setup: StillSetup = stillSetup()
    let r = try BoatRenderer(device: device, width: width, height: height, studio: setup.studio, layout: layout, draft: draft)
    let gpu: Double = try r.render(camera: setup.camera, samples: samples)
    annotate(r.image, setup: setup, layout: layout, buoyancy: buoy, draft: draft)
    try savePNG(r.image, to: URL(fileURLWithPath: out))
    let wall: Double = Date().timeIntervalSince(start)
    print("GPU: \(device.name)")
    print("\(width) × \(height), \(samples * samples) samples per pixel")
    print(String(format: "mass %.2f kg (hull %.2f, oars %.2f); displaced %.2f L of sea water; draft %.1f mm (drawn %.1f)",
                 buoy.totalKilograms, buoy.hullKilograms, buoy.oarKilograms, buoy.displacedCubicMetres * 1000, buoy.draft, draft))
    print(String(format: "with one %.1f-kg rower %.1f mm; at 725 lb %.1f mm fresh, %.1f mm sea (Newfound %.1f mm)",
                 rowerKilograms, buoy.draftWithRower, buoy.draftAtCapacityFresh, buoy.draftAtCapacitySea, nwDraftAtCapacity))
    print(String(format: "oars %.0f mm (%.1f in; S&T rule %.1f in), span %.0f mm; hull Lipschitz bound %.3f",
                 layout.oars[0].length, layout.oars[0].length / inch, layout.oarLengthRule / inch, layout.span, layout.hull.lipschitz))
    print(String(format: "%.2f s on the GPU, %.2f s wall", gpu, wall))
    print("Saved \(out)")
} catch {
    FileHandle.standardError.write("rowboat: \(error)\n".data(using: .utf8)!)
    exit(1)
}
