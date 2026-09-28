// Step 52: the fibres of a cotton sock — the knit, the yarn, one fibre (one
// cell), its cut end, and the cellulose it is made of. One still, rendered on
// the GPU, saved as a PNG.
//
//   .build/cotton_sock [width height samples-per-side out]

import Foundation

let args: [String] = Array(CommandLine.arguments.dropFirst())

func argument(_ n: Int, _ fallback: Int) -> Int {
    guard n < args.count, let v = Int(args[n]) else { return fallback }
    return v
}

let width: Int = argument(0, 1920)
let height: Int = argument(1, 1080)
let samples: Int = argument(2, 3)
let out: String = args.count > 3 ? args[3] : "renders/cotton_sock.png"

do {
    let device = try findDevice()
    let start = Date()
    let scene = try Scene(mutant: .none)
    let built: Double = Date().timeIntervalSince(start)
    let (image, gpu) = try renderSock(width: width, height: height, samples: samples, scene: scene, on: device, progress: true)
    annotate(image, scene: scene)
    try savePNG(image, to: URL(fileURLWithPath: out))
    let wall: Double = Date().timeIntervalSince(start)
    print("GPU: \(device.name)")
    print("\(width) × \(height), \(samples * samples) samples per pixel")
    print(String(format: "knit: wales %.3f mm, courses %.3f mm, loop %.3f mm of yarn (measured %.2f), depth %.4f mm, yarn radius %.1f µm",
                 scene.knit.w, scene.knit.c, scene.knit.loopLength, measuredLoopLength, scene.knit.d, scene.knit.r * 1000))
    print(String(format: "yarn: %.1f fibres per section, %d on the drawn surface, helix %.1f°; %d hairs in the main view",
                 fibresPerYarnSection, scene.yarn.fibreCount, helixAngle(radius: surfaceRadius(scene.yarn.section)) * 180 / Float.pi,
                 scene.hairs.count))
    let s: CrossSection = scene.yarn.section
    print(String(format: "fibre: wall half-thickness %.2f µm, arc %.2f µm, lumen %.2f × %.2f µm, reach %.1f µm; ribbon bound %.3f",
                 s.halfThickness, 2 * s.halfArc * s.bendRadius, 2 * s.lumenHalfArc * s.bendRadius, 2 * s.lumenHalfThickness, s.reach, scene.ribbonK))
    print("cellulose: \(scene.molecule.atoms.count) atoms, \(scene.molecule.formula)")
    print(String(format: "scene built in %.1f s; %.1f s on the GPU, %.1f s wall", built, gpu, wall))
    print("Saved \(out)")
} catch {
    FileHandle.standardError.write("cotton_sock: \(error)\n".data(using: .utf8)!)
    exit(1)
}
