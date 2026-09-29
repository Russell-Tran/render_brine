// Tests for step 79's recorded world and its renderer (part 4, prediction P3's
// renderer). Called from main.swift after the simulation's tests.
//
// SIM_MUTANT=shrunkBounds halves every ant's bounding sphere; the "bounds on
// and off draw the same picture" test must catch it.

import Foundation
import simd

func renderTests() {
    let here = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    var saved: FilmRecord? = nil

    section("The recorded world (records/world79.rec)")

    test("the record's SHA-256 is the manifest's, and it loads") {
        saved = try filmLoadVerified(directory: here)
        if let s = saved {
            expectEqual(s.frameCount, 1800)
            expectEqual(s.sim.config.seed, UInt64(79))
            expect(s.sim.config.mutant == .none, "the film's record must be made without a mutant")
        }
    }
    guard let film = saved else {
        print("        (no record: run `make record` on the laptop, or restore records/world79.rec)")
        return
    }

    // Not a test (it depends on the machine): does this machine's run of seed
    // 79 reproduce the recorded world?
    let reproduces: Bool = mutant == .none && film.sim.encoded() == rec.encoded()
    print(reproduces ? "        this machine's run of seed 79 reproduces the recorded world bit for bit"
                     : "        this machine's run of seed 79 is NOT the recorded world (expected on another machine, or under a mutant)")

    test("stored fields every 10 frames and at the last; none clipped") {
        expectEqual(film.fieldFrames, FilmRecord.fieldFrames(frameCount: film.frameCount))
        expectEqual(film.fieldFrames.count, 181)
        var top: UInt16 = 0
        for v in film.fields { top = max(top, v.max() ?? 0) }
        expect(top < 65535, "a stored field hit the 16-bit ceiling")
        let peak: Float = Float(top) * FilmRecord.fieldQuantum
        print("        peak stored pheromone \(fmt(Double(peak), 3)) marks/mm² (ceiling 6.55)")
    }

    test("the CPU rebuild runs one stored field onto the next (within two storage quanta)") {
        // Machine-independent: from stored field k, run on to stored frame
        // k + 1; it must land on what the GPU stored there.
        var worst: Float = 0
        let picks: [Int] = [0, 20, 45, 70, 95, 120, 142, 149, 150, 160, 175, 179]
        for k in picks {
            let f: Int = film.fieldFrames[k + 1]
            let rebuilt: [Float] = film.field(atFrame: f, fromStored: k)
            let stored: [UInt16] = film.fields[k + 1]
            for c in 0..<rebuilt.count {
                let s: Float = Float(stored[c]) * FilmRecord.fieldQuantum
                worst = max(worst, abs(rebuilt[c] - s))
            }
        }
        expect(worst <= 2.05 * FilmRecord.fieldQuantum, "worst \(worst) marks/mm²")
        print("        worst difference over \(picks.count) strides: \(worst) marks/mm² (quantum \(FilmRecord.fieldQuantum))")
    }

    test("on the recording machine: the rebuilt field is the live field (within one quantum)") {
        guard reproduces else { print("        skipped: this run is not the recorded world"); return }
        for f in checkFrames.sorted() {
            guard let live = filmLive.fields[f] else { continue }
            let rebuilt: [Float] = film.field(atFrame: f)
            var worst: Float = 0
            for c in 0..<live.count { worst = max(worst, abs(rebuilt[c] - live[c])) }
            expect(worst <= 1.05 * FilmRecord.fieldQuantum, "frame \(f): \(worst)")
        }
    }

    section("Drawing the world with the shared ant (lib/ant/v1)")

    let set = WorldSet(config: film.sim.config, grains: film.sim.grains)
    let cast = WorldCast(record: film.sim)

    test("sugar grains: step 26's crystal at each simulated sieve size, lying on the ground") {
        expectEqual(set.grains.count, film.sim.grains.count)
        var worstSize: Float = 0
        var worstRest: Float = 0
        for (g, s) in zip(set.grains, film.sim.grains) {
            worstSize = max(worstSize, abs(g.sieveSize - s.size))
            let low: Float = g.vertices().map { $0.y }.min() ?? 1
            worstRest = max(worstRest, abs(low))
            let d: Float = simd_distance(SIMD2<Float>(g.centre.x, g.centre.z), film.sim.config.sugar)
            expect(d + g.boundRadius <= set.pile.w + 1e-4, "grain outside the pile's sphere")
        }
        expect(worstSize < 1e-4, "sieve size off by \(worstSize) mm")
        expect(worstRest < 1e-5, "a grain floats or sinks by \(worstRest) mm")
        print("        unit sieve size \(fmt(Double(sugarUnitSieve), 4)) mm; pile sphere r = \(fmt(Double(set.pile.w), 2)) mm")
    }

    test("posed ants stand where the record says; stance feet on the ground, nothing below it") {
        for f in [0, 300, 900, 1431, 1500, 1799] {
            for (id, p) in cast.ants(atFrame: f) {
                let a: SimAntFrame = film.sim.ant(id, atFrame: f)
                let off: Float = simd_distance(SIMD2<Float>(p.at.x, p.at.z), a.position)
                expect(off < 1e-3, "frame \(f) ant \(id) is \(off) mm from its record")
                let dy: Float = abs(simWrap(p.yaw - a.yaw))
                expect(dy < 1e-4, "frame \(f) ant \(id) heading off by \(dy)")
                for j in 0..<6 where p.stance[j] { expect(abs(p.feet[j].y) < 1e-5, "stance foot above ground") }
                expect(p.lowest > -0.005, "frame \(f) ant \(id) reaches \(p.lowest) mm into the ground")
            }
        }
    }

    // A small renderer for the picture tests.
    let small: WorldRenderer
    do { small = try WorldRenderer(width: 480, height: 270, set: set) } catch {
        test("the renderer compiles lib/ant/v1's Metal with the scene's kernel") { throw error }
        return
    }
    test("the renderer compiles lib/ant/v1's Metal with the scene's kernel") {}

    let lastFrame: Int = film.frameCount - 1
    let lastField: [Float] = film.field(atFrame: lastFrame)
    let recorded: [AntV1.Posed] = cast.ants(atFrame: lastFrame).map { $0.posed }

    test("per-ant bounding spheres on or off draw the same picture, bit for bit") {
        let twelve: [AntV1.Posed] = worldScalingAnts(count: 12, camera: WorldCamera.close, config: film.sim.config)
        let closeScene = WorldFrameScene(camera: WorldCamera.close, ants: twelve, field: lastField)
        let wideScene = WorldFrameScene(camera: WorldCamera.wide, ants: recorded, field: lastField)
        let scenes: [WorldFrameScene] = [closeScene, wideScene]
        for scene in scenes {
            small.bounds = true
            try small.render(scene, samples: 1)
            let on: [UInt8] = small.image.bytes()
            small.bounds = false
            try small.render(scene, samples: 1)
            let off: [UInt8] = small.image.bytes()
            small.bounds = true
            var differ: Int = 0
            for k in 0..<on.count where on[k] != off[k] { differ += 1 }
            expect(differ == 0, "\(differ) bytes differ")
        }
    }

    test("the drawn distance functions are honest: nothing lies within 0.75 × the distance") {
        let ants: [AntV1.Posed] = worldScalingAnts(count: 12, camera: .close, config: film.sim.config)
        var rng = SimRandom(seed: 7, stream: 99)
        var points: [SIMD3<Float>] = []
        let centres: [SIMD2<Float>] = ants.map { SIMD2<Float>($0.at.x, $0.at.z) } + [film.sim.config.sugar, film.sim.config.nest]
        for c in centres {
            for _ in 0..<150 {
                let x: Float = c.x + rng.float(-4, 4)
                let z: Float = c.y + rng.float(-4, 4)
                let y: Float = rng.float(-1.5, 2.0)
                points.append(SIMD3<Float>(x, y, z))
            }
        }
        let first: [SIMD4<Float>] = try small.probe(points, ants: ants)
        var targets: [SIMD3<Float>] = []
        var from: [Int] = []
        for (k, p) in points.enumerated() where first[k].x > 0.002 {
            for _ in 0..<6 {
                var u = SIMD3<Float>(rng.float(-1, 1), rng.float(-1, 1), rng.float(-1, 1))
                if simd_length(u) < 1e-3 { u = SIMD3<Float>(0, 1, 0) }
                u = simd_normalize(u)
                let r: Float = first[k].x * AntV1.stepScale * rng.float(0.2, 1.0)
                targets.append(p + u * r)
                from.append(k)
            }
        }
        let second: [SIMD4<Float>] = try small.probe(targets, ants: ants)
        var bad: Int = 0
        var worst: Float = 0
        for k in 0..<targets.count where second[k].x < 0 {
            bad += 1
            worst = min(worst, second[k].x)
        }
        expect(bad == 0, "\(bad) of \(targets.count) points inside a surface (deepest \(worst) mm)")
        print("        \(targets.count) points checked around \(ants.count) ants, the pile and the nest")
    }

    test("the nest entrance is a hole: empty inside, card around it") {
        let n: SIMD4<Float> = set.nest
        let pts: [SIMD3<Float>] = [
            SIMD3<Float>(n.x, -1.0, n.y),               // down the hole
            SIMD3<Float>(n.x + 0.5 * n.z, -3.0, n.y),    // deeper, off-centre
            SIMD3<Float>(n.x + n.z + 1.0, -0.1, n.y),    // in the card beside it
            SIMD3<Float>(n.x, -n.w - 0.1, n.y),          // below the hole's floor
        ]
        let d: [SIMD4<Float>] = try small.probe(pts, ants: [])
        expect(d[0].x > 0 && d[1].x > 0, "the hole is not empty: \(d[0].x), \(d[1].x)")
        expect(d[2].x < 0 && d[3].x < 0, "the card is not solid: \(d[2].x), \(d[3].x)")
    }

    test("the wide shot has the nest on the left, the sugar on the right, and draws them there") {
        let cam: WorldCamera = .wide
        let c: SimWorldConfig = film.sim.config
        let nestPx: SIMD2<Float> = cam.project(SIMD3<Float>(c.nest.x, 0, c.nest.y), width: 480, height: 270)
        let sugarPx: SIMD2<Float> = cam.project(SIMD3<Float>(c.sugar.x, 0.2, c.sugar.y), width: 480, height: 270)
        expect(nestPx.x < sugarPx.x, "nest at \(nestPx), sugar at \(sugarPx)")
        try small.render(WorldFrameScene(camera: cam, ants: [], field: []), samples: 1)
        let seen: SIMD4<Float> = small.image.seen(Int(sugarPx.x), Int(sugarPx.y))
        expect(seen.y == 3, "the pile's centre pixel shows material \(seen.y), not sugar")
        let hole: SIMD4<Float> = small.image.seen(Int(nestPx.x), Int(nestPx.y))
        let px: SIMD4<UInt8> = small.image.rgba(Int(nestPx.x), Int(nestPx.y))
        expect(hole.y == 1 && px.x < 60, "the nest entrance is not dark: \(px)")
    }

    test("the pheromone is drawn: the formed trail tints the card blue; no field, no tint") {
        func blue(_ field: [Float]) throws -> Int {
            try small.render(WorldFrameScene(camera: .wide, ants: [], field: field), samples: 1)
            var n: Int = 0
            for y in 0..<270 {
                for x in 0..<480 {
                    let p: SIMD4<UInt8> = small.image.rgba(x, y)
                    if Int(p.z) - Int(p.x) > 20 { n += 1 }
                }
            }
            return n
        }
        let with: Int = try blue(lastField)
        let without: Int = try blue([])
        expect(without == 0, "\(without) blue pixels with no pheromone")
        expect(with > 2000, "only \(with) blue pixels on the formed trail")
        print("        blue pixels at the last frame: \(with) of \(480 * 270); with no field: \(without)")
    }
}
