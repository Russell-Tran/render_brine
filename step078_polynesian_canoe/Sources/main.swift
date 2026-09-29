// Step 78: a waʻa kaulua, the Hawaiian double-hulled voyaging canoe,
// extremely simplified, floating where Archimedes puts it. One still,
// rendered on the GPU and saved as a PNG.
//
//   .build/canoe [width height samples-per-side out]

import Foundation
import simd

var args: [String] = []
for (i, a) in CommandLine.arguments.enumerated() where i > 0 { args.append(a) }

func argument(_ n: Int, _ fallback: Int) -> Int {
    guard n < args.count, let v = Int(args[n]) else { return fallback }
    return v
}

let width: Int = argument(0, 1920)
let height: Int = argument(1, 1080)
let samples: Int = argument(2, 4)
let out: String = args.count > 3 ? args[3] : "renders/polynesian_canoe.png"

do {
    let device = try findDevice()
    let start = Date()
    let setup: StillSetup = stillSetup()
    let r = try CanoeRenderer(device: device, width: width, height: height, sky: setup.sky)
    let grid: ColumnGrid = try columnGrid(r, step: 0.01)
    let f: Flotation = solveDraft(grid)
    let sink: Float = placedDraft(f)
    let gpu: Double = try r.render(camera: setup.camera, samples: samples, sink: sink)
    let section: SectionInset = try sampleSection(r, sink: sink)
    let keel: [SIMD3<Float>] = try keelLine(r, sink: sink)
    annotate(r.image, setup: setup, flotation: f, sink: sink, section: section, keel: keel)
    try savePNG(r.image, to: URL(fileURLWithPath: out))
    let wall: Double = Date().timeIntervalSince(start)
    print("GPU: \(device.name)")
    print("\(width) × \(height), \(samples * samples) samples per pixel")
    print(String(format: "mass %.1f kg, sea water %.1f kg/m³ → %.4f m³ displaced; draft %.4f m (PVS %.3f m); ρV − M = %.3f kg",
                 canoeMass, seaWaterDensity, f.displaced, f.draft, pvsDraft, f.imbalance))
    let lwl: Float = try waterlineLength(r, sink: sink)
    print(String(format: "waterline %.3f m = %.2f ft (PVS 54 ft); at PVS's draft the hull would displace %.3f m³",
                 lwl, Double(lwl) / metresPerFoot, f.atPVSDraft))
    print(String(format: "sails %.2f m² (PVS %.2f m²), scaled ×%.4f from the drawing", sailArea(foresail) + sailArea(aftsail), pvsSailArea, sailScale))
    print(String(format: "%.2f s on the GPU, %.2f s wall", gpu, wall))
    print("Saved \(out)")
} catch {
    FileHandle.standardError.write("canoe: \(error)\n".data(using: .utf8)!)
    exit(1)
}
