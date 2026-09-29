// Step 79, "A minute of ant world": the renderer. It reads ONLY the recorded
// world (records/world79.rec, checked against records/MANIFEST) and draws it
// with the shared ant (lib/ant/v1); it is built without the simulation.
//
//   .build/render scale              prediction P3: one 1080p frame at 1, 5,
//                                    10, 20 and 40 ants, per-ant bounding
//                                    spheres on and off, two cameras
//   .build/render still F out.png [wide|close]
//                                    film frame F from the record
//   .build/render ants N out.png [wide|close] [off]
//                                    the scaling scene with N ants
//
// Environment: WORLD_SAMPLES (samples per pixel side, default 2),
// WORLD_WIDTH (default 1920; height is 9/16 of it).

import Foundation
import simd

func envInt(_ name: String, _ fallback: Int) -> Int {
    if let s = ProcessInfo.processInfo.environment[name], let v = Int(s) { return v }
    return fallback
}

func fmt(_ x: Double, _ d: Int = 2) -> String { String(format: "%.\(d)f", x) }

let argv: [String] = CommandLine.arguments
var args: [String] = argv
if !args.isEmpty { args.removeFirst() }
let width: Int = envInt("WORLD_WIDTH", 1920)
let height: Int = width * 9 / 16
let samples: Int = envInt("WORLD_SAMPLES", 2)
let here = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)

func cameraNamed(_ s: String?) -> WorldCamera { s == "close" ? .close : .wide }

do {
    let film: FilmRecord = try filmLoadVerified(directory: here)
    let set = WorldSet(config: film.sim.config, grains: film.sim.grains)
    let renderer = try WorldRenderer(width: width, height: height, set: set)
    let lastField: [Float] = film.field(atFrame: film.frameCount - 1)

    switch args.first ?? "" {
    case "still":
        guard args.count >= 3, let f = Int(args[1]) else { print("usage: still F out.png [wide|close]"); exit(2) }
        let cast = WorldCast(record: film.sim)
        let ants: [AntV1.Posed] = cast.ants(atFrame: f).map { $0.posed }
        let scene = WorldFrameScene(camera: cameraNamed(args.count > 3 ? args[3] : nil), ants: ants,
                                    field: film.field(atFrame: f))
        let t0: Date = Date()
        let gpu: Double = try renderer.render(scene, samples: samples)
        try worldSavePNG(renderer.image, to: URL(fileURLWithPath: args[2]))
        print("frame \(f): \(ants.count) ants, \(film.sim.clock.label(atFrame: f)); GPU \(fmt(gpu)) s, wall \(fmt(Date().timeIntervalSince(t0))) s → \(args[2])")

    case "ants":
        guard args.count >= 3, let n = Int(args[1]) else { print("usage: ants N out.png [wide|close] [off]"); exit(2) }
        let cam: WorldCamera = cameraNamed(args.count > 3 ? args[3] : nil)
        renderer.bounds = !(args.count > 4 && args[4] == "off")
        let ants: [AntV1.Posed] = worldScalingAnts(count: n, camera: cam, config: film.sim.config)
        let gpu: Double = try renderer.render(WorldFrameScene(camera: cam, ants: ants, field: lastField), samples: samples)
        try worldSavePNG(renderer.image, to: URL(fileURLWithPath: args[2]))
        print("\(n) ants: GPU \(fmt(gpu)) s → \(args[2])")

    case "scale":
        // One frame, timed: the best of a few renders (the GPU is shared
        // with other work, so the minimum is the frame's own cost).
        let counts: [Int] = [1, 5, 10, 20, 40]
        let frames: Double = Double(SimClock.film.frameCount)
        print("P3: one \(width)×\(height) frame, \(samples * samples) samples per pixel, field of the film's last frame")
        for (name, cam) in [("wide (250 mm across)", WorldCamera.wide), ("close (50 mm across)", WorldCamera.close)] {
            print("\ncamera: \(name)")
            print("  ants | bounds ON: GPU s   wall s  h/1800 | bounds OFF: GPU s   wall s  h/1800 | OFF/ON")
            for n in counts {
                let ants: [AntV1.Posed] = worldScalingAnts(count: n, camera: cam, config: film.sim.config)
                let scene = WorldFrameScene(camera: cam, ants: ants, field: lastField)
                var row: String = String(format: "  %4d |", n)
                var on: Double = 0
                for b in [true, false] {
                    renderer.bounds = b
                    // Bounds off is slow enough that one run is its own
                    // warm-up; bounds on gets a warm-up and three runs.
                    if b { _ = try renderer.render(scene, samples: samples) }
                    var bestGPU: Double = .infinity
                    var bestWall: Double = .infinity
                    let repeats: Int = b ? 3 : 1
                    for _ in 0..<repeats {
                        let t0: Date = Date()
                        let g: Double = try renderer.render(scene, samples: samples)
                        bestWall = min(bestWall, Date().timeIntervalSince(t0))
                        bestGPU = min(bestGPU, g)
                    }
                    let hours: Double = bestWall * frames / 3600
                    row += String(format: "  %9.3f %8.3f %7.2f |", bestGPU, bestWall, hours)
                    if b { on = bestWall } else { row += String(format: " %5.1f×", bestWall / on) }
                }
                print(row)
            }
        }

    default:
        print("usage: render scale | still F out.png [wide|close] | ants N out.png [wide|close] [off]")
        exit(2)
    }
} catch {
    print("error: \(error)")
    exit(1)
}
