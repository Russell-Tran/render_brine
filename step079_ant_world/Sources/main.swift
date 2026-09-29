// Step 79, "A minute of ant world": the renderer and the film. It reads ONLY
// the recorded world (records/world79.rec, checked against records/MANIFEST)
// and draws it with the shared ant (lib/ant/v1); it is built without the
// simulation.
//
//   .build/render film [from to]     render the film's frames not yet on disk
//                                    (renders/frames/NNNN.png) — stop it any
//                                    time; run it again to carry on
//   .build/render count              the film's frame count (for make encode)
//   .build/render gif FROM TO STEP   a GIF excerpt from the frames on disk
//   .build/render verify F...        draw frames afresh and compare them with
//                                    the ones on disk (the resume check)
//   .build/render overlap [STEP]     3D interpenetration over the film
//   .build/render refs               draw the reference frames → records/refs
//   .build/render refcheck           draw them again, compare with records/refs
//   .build/render frame F out.png    one film frame, captions and all
//   .build/render scale              prediction P3 (part 4)
//   .build/render ants N out.png [wide|close] [off]
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
let store = FilmStore(directory: here.appendingPathComponent("renders/frames"))
let showcase: URL = here.appendingPathComponent("../showcase")
let logURL: URL = here.appendingPathComponent("renders/film_log.txt")

func cameraNamed(_ s: String?) -> WorldCamera {
    if let name = s, name.hasPrefix("close") { return .close }
    return .wide
}

func log(_ line: String) {
    print(line)
    let stamped: String = ISO8601DateFormatter().string(from: Date()) + " " + line + "\n"
    if let h = try? FileHandle(forWritingTo: logURL) {
        h.seekToEndOfFile()
        h.write(stamped.data(using: .utf8) ?? Data())
        h.closeFile()
    } else {
        try? FileManager.default.createDirectory(at: logURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        try? stamped.write(to: logURL, atomically: true, encoding: .utf8)
    }
}


do {
    let film: FilmRecord = try filmLoadVerified(directory: here)
    let plan = FilmPlan(film: film)

    switch args.first ?? "" {
    case "film":
        let from: Int = args.count > 1 ? (Int(args[1]) ?? 0) : 0
        let to: Int = args.count > 2 ? (Int(args[2]) ?? plan.frameCount) : plan.frameCount
        let list: [Int] = Array(max(from, 0)..<min(to, plan.frameCount))
        let missing: Int = list.filter { !store.has($0) }.count
        log("film: session start; frames \(from)–\(to - 1): \(missing) to render, \(list.count - missing) on disk; \(width)×\(height), \(samples * samples) spp")
        let frames = try FilmFrames(plan: plan, width: width, height: height, samples: samples)
        let t0: Date = Date()
        var done: Int = 0
        let result = try filmRender(frames, store: store, frames: list) { f, gpu in
            done += 1
            if done % 30 == 0 || done == missing {
                let el: Double = Date().timeIntervalSince(t0)
                let left: Double = el / Double(done) * Double(missing - done)
                print("  frame \(f): GPU \(fmt(gpu)) s; \(done)/\(missing) in \(fmt(el / 60, 1)) min, ~\(fmt(left / 60, 1)) min left")
            }
        }
        log("film: session end; rendered \(result.rendered), skipped \(result.skipped), \(fmt(result.seconds, 1)) s wall")
        let all: Bool = (0..<plan.frameCount).allSatisfy { store.has($0) }
        print(all ? "all \(plan.frameCount) frames are on disk: make encode" : "not all frames are on disk yet; run again to carry on")

    case "count":
        // The film's frame count, for the encoder (make encode).
        print(plan.frameCount)

    case "gif":
        guard args.count >= 4, let a = Int(args[1]), let b = Int(args[2]), let step = Int(args[3]) else {
            print("usage: gif FROM TO STEP [width] [delay-cs]"); exit(2)
        }
        let gw: Int = args.count > 4 ? (Int(args[4]) ?? 960) : 960
        let delay: Int = args.count > 5 ? (Int(args[5]) ?? 10) : 10
        try filmGIF(store: store, frames: Array(stride(from: a, to: b, by: step)), width: gw, delayCentiseconds: delay,
                    to: here.appendingPathComponent("renders/ant_world.gif"), copyTo: showcase.appendingPathComponent("ant_world.gif"))

    case "verify":
        // Draw the named frames afresh, in the order given, on a new
        // renderer, and compare with the frames on disk.
        let list: [Int] = args.dropFirst().compactMap { Int($0) }
        let frames = try FilmFrames(plan: plan, width: width, height: height, samples: samples)
        var worst: Int = 0
        for f in list {
            guard store.has(f) else { print("frame \(f): not on disk"); continue }
            try frames.draw(f)
            let disk = try store.read(f)
            let d: FilmChannelDifference = filmCompare(frames.renderer.image.bytes(), disk.rgba)
            let same: Bool = d.maxAbs.allSatisfy { $0 == 0 }
            worst = max(worst, d.maxAbs.max() ?? 0)
            print("frame \(f): \(same ? "identical to the frame on disk" : "DIFFERS: max \(d.maxAbs), mean \(d.meanAbs)")")
        }
        log("verify: \(list.count) frames drawn afresh; worst channel difference \(worst)")
        if worst > 0 { exit(1) }

    case "overlap":
        let step: Int = args.count > 1 ? (Int(args[1]) ?? 1) : 1
        let frames = try FilmFrames(plan: plan, width: 16, height: 16, samples: 1)
        var worst: Float = 0
        var framesInside: Int = 0
        var pairFrames: Int = 0
        var checked: Int = 0
        var antFrames: Int = 0
        var nearFood: Int = 0
        var worstAt: String = ""
        let cfg: SimWorldConfig = film.sim.config
        for f in stride(from: 0, to: plan.frameCount, by: step) {
            let ants = frames.ants(f)
            antFrames += ants.count
            let o: FilmOverlap = try filmOverlap(frames.renderer, ants: ants)
            checked += o.pairsChecked
            if o.pairsInside > 0 { framesInside += 1 }
            pairFrames += o.pairsInside
            for (i, _, _) in o.inside {
                let a: SimAntFrame = film.sim.ant(i, atFrame: plan.recordFrame(f))
                if simd_distance(a.position, cfg.sugar) < cfg.sugarRadius + SimConst.foodCrowdMargin { nearFood += 1 }
            }
            if o.deepest > worst { worst = o.deepest; worstAt = "frame \(f)" }
        }
        let sampled: Int = (plan.frameCount + step - 1) / step
        log("overlap (every \(step) frame(s), \(sampled) frames, \(antFrames) ant-frames): \(checked) close pairs checked; \(pairFrames) pair-frames inside one another (\(nearFood) of them at the food) in \(framesInside) frames; deepest \(fmt(Double(worst), 3)) mm at \(worstAt)")

    case "refs", "refcheck":
        let drawn: [(Int, [UInt8])] = try filmDrawReferences(plan)
        let dir: URL = here.appendingPathComponent(filmReferenceDirectory)
        for (f, rgba) in drawn {
            let url: URL = dir.appendingPathComponent(String(format: "%04d.png", f))
            if args[0] == "refs" {
                try filmSavePNG(rgba, width: filmReferenceWidth, height: filmReferenceHeight, to: url)
                print("reference frame \(f) → \(url.path)")
            } else {
                let ref = try filmReadPNG(url)
                let d: FilmChannelDifference = filmCompare(rgba, ref.rgba)
                print("frame \(f): max |diff| R G B \(d.maxAbs) /255, mean \(d.meanAbs.map { fmt($0, 4) }) /255 — \(d.within ? "within" : "OUTSIDE") the tolerance (mean < \(filmReferenceMeanTolerance), max < \(filmReferenceMaxTolerance))")
            }
        }

    case "frame":
        guard args.count >= 3, let f = Int(args[1]) else { print("usage: frame F out.png"); exit(2) }
        let frames = try FilmFrames(plan: plan, width: width, height: height, samples: samples)
        let t0: Date = Date()
        let gpu: Double = try frames.draw(f)
        try worldSavePNG(frames.renderer.image, to: URL(fileURLWithPath: args[2]))
        print("film frame \(f): \(frames.ants(f).count) ants, \(plan.clockLabel(f)), camera \(fmt(Double(plan.camera(f).width), 1)) mm across; GPU \(fmt(gpu)) s, wall \(fmt(Date().timeIntervalSince(t0))) s → \(args[2])")

    case "ants":
        guard args.count >= 3, let n = Int(args[1]) else { print("usage: ants N out.png [wide|close] [off]"); exit(2) }
        let set = WorldSet(config: film.sim.config, grains: film.sim.grains)
        let renderer = try WorldRenderer(width: width, height: height, set: set)
        let cam: WorldCamera = cameraNamed(args.count > 3 ? args[3] : nil)
        renderer.bounds = !(args.count > 4 && args[4] == "off")
        let ants: [AntV1.Posed] = worldScalingAnts(count: n, camera: cam, config: film.sim.config)
        let lastField: [Float] = film.field(atFrame: film.frameCount - 1)
        let gpu: Double = try renderer.render(WorldFrameScene(camera: cam, ants: ants, field: lastField), samples: samples)
        try worldSavePNG(renderer.image, to: URL(fileURLWithPath: args[2]))
        print("\(n) ants: GPU \(fmt(gpu)) s → \(args[2])")

    case "scale":
        // One frame, timed: the best of a few renders (the GPU is shared
        // with other work, so the minimum is the frame's own cost).
        let set = WorldSet(config: film.sim.config, grains: film.sim.grains)
        let renderer = try WorldRenderer(width: width, height: height, set: set)
        let lastField: [Float] = film.field(atFrame: film.frameCount - 1)
        let counts: [Int] = [1, 5, 10, 20, 40]
        let frameTotal: Double = Double(SimClock.film.frameCount)
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
                    let hours: Double = bestWall * frameTotal / 3600
                    row += String(format: "  %9.3f %8.3f %7.2f |", bestGPU, bestWall, hours)
                    if b { on = bestWall } else { row += String(format: " %5.1f×", bestWall / on) }
                }
                print(row)
            }
        }

    default:
        print("usage: render film [from to] | encode | gif FROM TO STEP | verify F... | overlap [STEP] | refs | refcheck | frame F out.png | scale | ants N out.png")
        exit(2)
    }
} catch {
    print("error: \(error)")
    exit(1)
}
