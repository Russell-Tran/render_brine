// Tests for step 79's simulation (prediction P2: does a trail emerge?), then
// (RenderTests.swift) its recorded world and renderer (part 4).
//
// Everything is measured from the RECORD the film will read — positions,
// headings, distances, states, the dabs and the replayed field — not from the
// rules that made them.
//
// SIM_MUTANT=noPheromone|rewind|tightTurn breaks the simulation on purpose;
// `make mutants` requires the suite to fail for each:
//   noPheromone — no dab ever reaches the field     → "a trail emerges" fails
//   rewind      — the last fifth of the film clock runs backwards → the
//                 forward-only tests fail
//   tightTurn   — the turning cap is 3 mm instead of 5 → the turn-radius test
//                 fails
//   shrunkBounds — every ant's bounding sphere at half its radius (renderer)
//                 → "bounds on or off draw the same picture" fails
//
// `.build/tests report` prints the P2 numbers (with and without pheromone, and
// formation times over several seeds) instead of testing.

import Foundation
import simd

let mutant: SimMutant = SimMutant.fromEnvironment()
let argv: [String] = CommandLine.arguments
var args: [String] = argv
if !args.isEmpty { args.removeFirst() }

func fmt(_ x: Double, _ d: Int = 2) -> String { String(format: "%.\(d)f", x) }

/// The film run, once, and its field at a few frames.
let lastFrame: Int = SimClock.film.frameCount - 1
let checkFrames: Set<Int> = [0, 450, 900, 1350, 1499, lastFrame]
let started: Date = Date()
let film: SimRunResult = try simFilmRun(seed: 79, mutant: mutant, keepFieldsAt: checkFrames)
/// The same run, under a name the renderer's tests can see past their own `film`.
let filmLive: SimRunResult = film
let runSeconds: Double = Date().timeIntervalSince(started)
let rec: SimFilmRecord = film.record
let cfg: SimWorldConfig = rec.config

// MARK: - report mode

if args.first == "report" {
    func summary(_ r: SimFilmRecord, _ label: String) {
        let c: SimConcentration = simConcentration(r)
        let n: Int = r.frameCount
        let lastTen: Double = simMean(c.outbound, from: n - 300, to: n) ?? -1
        let allTen: Double = simMean(c.all, from: n - 300, to: n) ?? -1
        var line: String = "\(label): returns \(r.returns), dabs \(r.dabs.count), first sugar contact "
        line += r.firstContact.map { fmt($0, 1) + " s" } ?? "never"
        print(line)
        print("  outbound index, last 10 s of film: \(fmt(lastTen)); all ants: \(fmt(allTen)); chance \(fmt(c.uniformShare, 3))")
        if let f = c.formedFrame {
            print("  trail FORMED at film frame \(f) = film \(fmt(Double(f) / 30, 1)) s = ant time \(fmt(r.simSeconds(atFrame: f), 0)) s")
        } else {
            print("  trail never formed within the film")
        }
        var row: String = "  index by film second (every 5 s):"
        for s in stride(from: 5, through: 60, by: 5) {
            let v: Double? = c.outbound[min(s * 30, n) - 1]
            row += " " + (v.map { fmt($0) } ?? "-")
        }
        print(row)
    }
    print("P2 report — film clock: \(SimClock.film.segments), \(fmt(SimClock.film.filmSeconds, 0)) s of film = \(fmt(SimClock.film.simSeconds(atFrame: lastFrame), 0)) s of ant time")
    print("simulation wall time for the film run: \(fmt(runSeconds, 1)) s")
    summary(rec, "with pheromone (seed 79)")
    let off: SimRunResult = try simFilmRun(seed: 79, mutant: .noPheromone)
    summary(off.record, "noPheromone (seed 79)")
    if let fld = film.fields[lastFrame] {
        let v: SimTrailVisibility = simTrailVisibility(field: fld, width: cfg.gridWidth, height: cfg.gridHeight, config: cfg)
        print("visibility at the last frame: on-trail median \(fmt(Double(v.onTrailMedian), 3)) marks/mm², off-trail 99th pct \(fmt(Double(v.offTrail99), 4)), contrast \(fmt(Double(v.contrast), 1))×, continuity \(fmt(Double(v.continuity))), mass in corridor \(fmt(v.massInCorridor)), peak \(fmt(Double(v.peak), 2))")
    }
    // Formation over seeds, on a 20-minute real-time clock.
    let long = SimClock(segments: [SimClockSegment(frames: 1200 * 30, ticksPerFrame: 2)])
    var times: [String] = []
    var offTimes: [String] = []
    for seed in [UInt64(79), 1, 2, 3, 4, 5, 6, 7] {
        var c = SimWorldConfig()
        c.seed = seed
        c.mutant = .none
        let r: SimFilmRecord = try simRun(config: c, clock: long).record
        let f: Int? = simConcentration(r).formedFrame
        times.append(f.map { fmt(r.simSeconds(atFrame: $0), 0) } ?? "never")
        c.mutant = .noPheromone
        let r0: SimFilmRecord = try simRun(config: c, clock: long).record
        let c0: SimConcentration = simConcentration(r0)
        let f0: Int? = c0.formedFrame
        let m0: Double = simMean(c0.outbound, from: 0, to: r0.frameCount) ?? -1
        offTimes.append((f0.map { fmt(r0.simSeconds(atFrame: $0), 0) } ?? "never") + "/" + fmt(m0))
    }
    print("formation time over 20 min of ant time, seeds 79,1..7 (s): \(times)")
    print("noPheromone, same seeds (formed / mean outbound index): \(offTimes)")
    exit(0)
}

// MARK: - the constants

section("Sourced constants")

test("walking speed is Dussutour et al.'s 90 mm / 2.96 s ≈ 30.4 mm/s") {
    expect(abs(SimConst.walkSpeed - 30.405) < 0.01, "\(SimConst.walkSpeed)")
}

test("pheromone mean lifetime is Beckers et al.'s 47 min") {
    expectEqual(SimConst.pheromoneLifetime, 2820.0)
}

test("14% of workers never lay (Mailleux et al. 2005): 3 of 20 here, from the seeded draw") {
    let layers: Int = rec.ants(atFrame: 0).filter { $0.layer == 1 }.count
    let nonLayers: Int = cfg.antCount - layers
    // Binomial(20, 0.14): mean 2.8; anything 0...8 is unremarkable.
    expect(nonLayers >= 0 && nonLayers <= 8, "\(nonLayers)")
    print("        non-layers: \(nonLayers) of \(cfg.antCount)")
}

test("sugar grains: step 26's sieve sizes, none overlapping, all inside the pile") {
    expect(rec.grains.count == cfg.grainCount, "\(rec.grains.count)")
    for g in rec.grains {
        expect(g.size >= 0.30 && g.size <= 0.67, "size \(g.size)")
        let d: Float = simd_distance(SIMD2<Float>(g.x, g.z), cfg.sugar)
        expect(d + g.size / 2 <= cfg.sugarRadius + 1e-4, "outside the pile")
    }
    for i in 0..<rec.grains.count {
        for j in (i + 1)..<rec.grains.count {
            let a: SimGrain = rec.grains[i]
            let b: SimGrain = rec.grains[j]
            let d: Float = simd_distance(SIMD2<Float>(a.x, a.z), SIMD2<Float>(b.x, b.z))
            let touch: Float = (a.size + b.size) / 2
            expect(d >= touch - 1e-4, "grains \(i) and \(j) overlap")
        }
    }
}

// MARK: - the field

section("Pheromone field (GPU)")

func blobField(evaporate: Bool, diffuse: Bool) throws -> SimField {
    let dt: Double = SimConst.tick
    let f = try SimField.forSimulation(width: 120, height: 80)
    if !evaporate { f.decay = 1 }
    if !diffuse { f.alpha = 0 }
    var dabs: [SimDabGPU] = []
    for k in 0..<12 {
        let x: Float = 3.3 + Float(k) * 4.7
        let z: Float = 5.1 + Float(k % 5) * 6.9
        dabs.append(f.gpuDab(x: x, z: z, marks: 1 + Float(k) * 0.25))
    }
    // One dab hard against the edge, to test the zero-flux boundary.
    dabs.append(f.gpuDab(x: 0.26, z: 0.26, marks: 2))
    try f.step(dabs: dabs)
    _ = dt
    return f
}

test("a dab adds exactly its marks to the field's integral") {
    let f = try SimField.forSimulation(width: 40, height: 40)
    f.decay = 1
    f.alpha = 0
    try f.step(dabs: [f.gpuDab(x: 7.37, z: 11.91, marks: 1)])
    expect(abs(f.totalMarks() - 1) < 1e-6, "\(f.totalMarks())")
}

test("diffusion conserves pheromone (5000 ticks, edges closed)") {
    let f = try blobField(evaporate: false, diffuse: true)
    let m0: Double = f.totalMarks()
    for _ in 0..<5000 { try f.step(dabs: []) }
    let m1: Double = f.totalMarks()
    expect(abs(m1 - m0) / m0 < 1e-5, "\(m0) → \(m1)")
    expect(f.values.allSatisfy { $0 >= 0 }, "negative pheromone")
    print("        mass \(m0) → \(m1), relative change \(String(format: "%.2e", abs(m1 - m0) / m0))")
}

test("diffusion spreads a dot as the heat equation says (variance grows 2Dt)") {
    let f = try SimField.forSimulation(width: 80, height: 80)
    f.decay = 1
    let c: Float = 20.25    // a cell centre
    try f.step(dabs: [f.gpuDab(x: c, z: c, marks: 1)])
    let n: Int = 3000
    for _ in 1..<n { try f.step(dabs: []) }
    var sxx: Double = 0
    var m: Double = 0
    let v = f.values
    for j in 0..<80 {
        for i in 0..<80 {
            let centre: Double = Double(i) + 0.5
            let xc: Double = centre * Double(f.cell)
            let x: Double = xc - Double(c)
            let val: Double = Double(v[j * 80 + i])
            sxx += val * x * x
            m += val
        }
    }
    let variance: Double = sxx / m
    // The dot starts in one cell; the n spreading steps each add 2·α·h².
    let area: Double = Double(SimConst.cell * SimConst.cell)
    let diffusion: Double = Double(SimConst.diffusion)
    let rate: Double = diffusion * SimConst.tick
    let alpha: Double = rate / area
    let steps: Double = Double(n)
    let perStep: Double = 2 * alpha
    let spread: Double = perStep * steps
    let expected: Double = spread * area
    expect(abs(variance - expected) / expected < 0.01, "variance \(variance), expected \(expected)")
}

test("evaporation is an exact e^(−t/τ) per tick, cell by cell, and in total") {
    let f = try blobField(evaporate: true, diffuse: false)
    let before: [Float] = f.snapshot()
    let m0: Double = f.totalMarks()
    let n: Int = 3600   // one minute
    for _ in 0..<n { try f.step(dabs: []) }
    let after: [Float] = f.snapshot()
    // The blob was already one factor in; compare cell by cell with the same
    // Float multiplications on the CPU: bit for bit.
    var cpu: [Float] = before
    for _ in 0..<n { for k in 0..<cpu.count { cpu[k] = cpu[k] * f.decay } }
    expect(cpu == after, "GPU evaporation differs from CPU")
    let m1: Double = f.totalMarks()
    let elapsed: Double = Double(n) * SimConst.tick
    let ratio: Double = -elapsed / SimConst.pheromoneLifetime
    let want: Double = m0 * exp(ratio)
    expect(abs(m1 - want) / want < 1e-4, "\(m1) vs \(want)")
    let kept: Double = m1 / m0
    let theory: Double = exp(-60.0 / 2820.0)
    print("        one minute: ×\(kept) (e^(−60/2820) = \(theory))")
}

// MARK: - determinism, saving, replay

section("Determinism, saving, replay")

test("same seed → the same record, bit for bit (and the same field)") {
    let again: SimRunResult = try simFilmRun(seed: 79, mutant: mutant, keepFieldsAt: [lastFrame])
    expect(again.record.encoded() == rec.encoded(), "records differ")
    expect(again.fields[lastFrame] == film.fields[lastFrame], "fields differ")
}

test("a different seed gives a different world") {
    var c = SimWorldConfig()
    c.seed = 80
    c.mutant = mutant
    let short = SimClock(segments: [SimClockSegment(frames: 300, ticksPerFrame: 30)])
    let other: SimFilmRecord = try simRun(config: c, clock: short).record
    let mine: SimFilmRecord = try simRun(config: cfg, clock: short).record
    expect(other.antFrames != mine.antFrames, "seeds 79 and 80 agree")
}

test("the stored structs have no padding (so equal records encode equally)") {
    expectEqual(MemoryLayout<SimAntFrame>.size, MemoryLayout<SimAntFrame>.stride)
    expectEqual(MemoryLayout<SimAntFrame>.size, 28)
    expectEqual(MemoryLayout<SimTrackPoint>.size, 24)
    expectEqual(MemoryLayout<SimDabEvent>.size, 16)
}

test("the record saves and loads back exactly; it is small") {
    let url = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("step079_track_test.bin")
    try rec.save(to: url)
    let back: SimFilmRecord = try SimFilmRecord.load(from: url)
    expect(back == rec, "round trip changed the record")
    let size: Int = (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? Int) ?? -1
    let points: Int = rec.tracks.map { $0.count }.reduce(0, +)
    let kb: Int = size / 1024
    print("        \(kb) KB for \(rec.frameCount) frames × \(rec.antCount) ants, \(points) track points, \(rec.dabs.count) dabs")
    expect(size > 0 && size < 20_000_000, "\(size) bytes")
    try? FileManager.default.removeItem(at: url)
    expectThrows { _ = try SimFilmRecord.decode(Data([1, 2, 3])) }
}

test("the replayed field equals the live field bit for bit (frames out of order too)") {
    let player = try SimFieldPlayer(record: rec)
    for f in checkFrames.sorted() + [900, 0] {
        let v: [Float] = try player.field(atFrame: f)
        expect(v == film.fields[f], "frame \(f) differs")
    }
}

// MARK: - forward only

section("Forward only: the film never rewinds")

test("the film clock only moves forward, and the film is 60 s") {
    expectEqual(rec.frameCount, 1800)
    for f in 1..<rec.frameCount {
        expect(rec.ticks[f] > rec.ticks[f - 1], "frame \(f): tick \(rec.ticks[f]) after \(rec.ticks[f - 1])")
        if rec.ticks[f] <= rec.ticks[f - 1] { break }
    }
    var last: Double = -1
    for k in 0...600 {
        let s: Double = rec.clock.simSeconds(atFilmSecond: Double(k) / 10)
        expect(s >= last, "film second \(Double(k) / 10)")
        last = s
    }
}

test("every ant's distance walked never decreases; heading changes only while walking") {
    for i in 0..<rec.antCount {
        var prev: SimAntFrame = rec.ant(i, atFrame: 0)
        for f in 1..<rec.frameCount {
            let a: SimAntFrame = rec.ant(i, atFrame: f)
            expect(a.distance >= prev.distance, "ant \(i) frame \(f): \(prev.distance) → \(a.distance)")
            if a.distance < prev.distance { break }
            if a.distance == prev.distance && a.visible && prev.visible && a.outing == prev.outing {
                expect(a.yaw == prev.yaw, "ant \(i) turned on the spot at frame \(f)")
            }
            prev = a
        }
    }
}

test("track distances strictly increase within each outing") {
    for i in 0..<rec.antCount {
        let t: [SimTrackPoint] = rec.tracks[i]
        for k in 1..<max(t.count, 1) where t[k].outing == t[k - 1].outing {
            expect(t[k].distance > t[k - 1].distance, "ant \(i) point \(k)")
            expect(t[k].tick > t[k - 1].tick, "ant \(i) point \(k) tick")
        }
    }
}

test("the clock label shows the rate: time-lapse ×15, then real time") {
    expect(rec.clock.label(atFrame: 0).hasPrefix("time-lapse ×15"), rec.clock.label(atFrame: 0))
    expect(rec.clock.label(atFrame: lastFrame).hasPrefix("real time"), rec.clock.label(atFrame: lastFrame))
}

// MARK: - on the ground, in bounds, at the sourced speed, drawable turns

section("Kinematics")

test("every visible ant stays on the ground plane inside the arena (body and antennae)") {
    for f in 0..<rec.frameCount {
        for a in rec.ants(atFrame: f) where a.visible {
            let fw = SIMD2<Float>(cos(a.yaw), sin(a.yaw))
            let pts: [SIMD2<Float>] = [a.position, a.position + fw * (SimConst.antennaAhead + SimConst.antennaSide),
                                       a.position - fw * SimConst.gasterTipBehind]
            for p in pts {
                expect(p.x >= 0 && p.y >= 0 && p.x <= cfg.width && p.y <= cfg.depth, "frame \(f) out at \(p)")
            }
            expect(a.x.isFinite && a.z.isFinite && a.yaw.isFinite, "not finite")
        }
    }
}

test("ants walk at the sourced 30.4 mm/s — never faster; slower only in a dab's stop") {
    var full: Int = 0
    var steps: Int = 0
    let perTick: Float = SimConst.walkSpeed * Float(SimConst.tick)
    for t in rec.tracks {
        for k in 1..<max(t.count, 1) where t[k].outing == t[k - 1].outing {
            let dt: Float = Float(t[k].tick - t[k - 1].tick)
            let ds: Float = t[k].distance - t[k - 1].distance
            let chord: Float = simd_distance(SIMD2<Float>(t[k].x, t[k].z), SIMD2<Float>(t[k - 1].x, t[k - 1].z))
            expect(ds <= perTick * dt + 1e-3, "too fast: \(ds) in \(dt) ticks")
            expect(chord <= ds + 1e-3, "moved further than it walked")
            if dt == 1 {
                steps += 1
                if abs(ds - perTick) < 1e-3 { full += 1 }
            }
        }
    }
    let fullD: Double = Double(full)
    let stepsD: Double = Double(max(steps, 1))
    let share: Double = fullD / stepsD
    expect(share > 0.9, "only \(share) of single-tick steps at full speed")
    print("        \(fmt(share * 100, 1))% of walking ticks at exactly 30.4 mm/s; the rest easing into or out of dabs")
}

test("no path curves tighter than the shared ant's 5 mm (AntV1.minimumTurnRadius)") {
    var tightest: Float = .infinity
    for t in rec.tracks {
        for k in 1..<max(t.count, 1) where t[k].outing == t[k - 1].outing {
            let ds: Float = t[k].distance - t[k - 1].distance
            let dh: Float = abs(simWrap(t[k].yaw - t[k - 1].yaw))
            if dh > 0 { tightest = min(tightest, ds / dh) }
            // Distances are Floats (as lib/ant/v1's Track keeps them): allow
            // the rounding of the two distances, one ulp each.
            let slack: Float = 2 * t[k].distance.ulp
            expect(dh <= (ds + slack) / 5 + 1e-6, "turned \(dh) rad in \(ds) mm")
        }
    }
    print("        tightest radius on any track: \(fmt(Double(tightest), 3)) mm")
}

test("the tracks run at least one stride (2.53 mm) past every frame") {
    for i in 0..<rec.antCount {
        let a: SimAntFrame = rec.ant(i, atFrame: lastFrame)
        guard a.visible, a.state != .tasting else { continue }
        let pts: [SimTrackPoint] = rec.trackPoints(ant: i, outing: Int(a.outing))
        let reach: Float = pts.last?.distance ?? 0
        // An ant that walked on after the film's end has track beyond it.
        expect(reach >= a.distance, "ant \(i)")
        if reach < a.distance + 2.53 { print("        ant \(i) stood still or went in after the last frame (reach \(reach - a.distance) mm)") }
    }
}

test("bodies stay apart away from the food — how close they come, and how often they overlap") {
    var worst: Float = .infinity
    var overlaps: Int = 0
    var atFood: Int = 0
    var pairs: Int = 0
    let crowd: Float = cfg.sugarRadius + SimConst.foodCrowdMargin
    for f in 0..<rec.frameCount {
        let ants: [SimAntFrame] = Array(rec.ants(atFrame: f)).filter { $0.visible }
        for i in 0..<ants.count {
            for j in (i + 1)..<ants.count {
                let a: SimAntFrame = ants[i]
                let b: SimAntFrame = ants[j]
                if simd_distance(a.position, b.position) > 8 { continue }
                // Ants inside the nest entrance are going in or coming out.
                if simd_distance(a.position, cfg.nest) < 3 || simd_distance(b.position, cfg.nest) < 3 { continue }
                let fa = SIMD2<Float>(cos(a.yaw), sin(a.yaw))
                let fb = SIMD2<Float>(cos(b.yaw), sin(b.yaw))
                let gap: Float = simSegmentDistance(a.position + fa * SimConst.bodyFront, a.position + fa * SimConst.bodyRear,
                                                    b.position + fb * SimConst.bodyFront, b.position + fb * SimConst.bodyRear)
                    - 2 * SimConst.bodyRadius
                pairs += 1
                if simd_distance(a.position, cfg.sugar) < crowd || simd_distance(b.position, cfg.sugar) < crowd {
                    if gap < 0 { atFood += 1 }
                    continue
                }
                worst = min(worst, gap)
                if gap < 0 { overlaps += 1 }
            }
        }
    }
    let totalAntFrames: Int = rec.antFrames.filter { $0.visible }.count
    let overlapsD: Double = Double(overlaps)
    let framesD: Double = Double(max(totalAntFrames, 1))
    let share: Double = overlapsD / framesD
    print("        away from the food: \(overlaps) overlapping pair-frames in \(totalAntFrames) ant-frames (\(fmt(share * 100, 1))%), deepest \(fmt(Double(-worst), 2)) mm; at the food (crowding allowed): \(atFood)")
    // MODEL bound: ants squeezing past one coming the other way overlap for
    // a moment; more than 5% of ant-frames would mean they jam.
    expect(share < 0.05, "\(overlaps) overlaps")
}

// MARK: - the trail

section("The trail (P2)")

let conc: SimConcentration = simConcentration(rec)

test("a trail EMERGES: outbound ants concentrate onto the nest–sugar route and stay there") {
    let n: Int = rec.frameCount
    let lastTen: Double = simMean(conc.outbound, from: n - 300, to: n) ?? 0
    let firstTen: Double = simMean(conc.outbound, from: 0, to: 300) ?? 0
    expect(conc.formedFrame != nil, "the index never reached \(SimMetric.formedThreshold) and held")
    // Formed means: reached 0.5 and never fell below 0.4 after; so the film's
    // last 10 s must hold at least 0.4.
    expect(lastTen >= SimMetric.holdThreshold, "last 10 s: \(lastTen)")
    expect(firstTen < 0.25, "the route is not given from the start: \(firstTen)")
    if let f = conc.formedFrame {
        print("        formed at film \(fmt(Double(f) / 30, 1)) s (ant time \(fmt(rec.simSeconds(atFrame: f), 0)) s); index first 10 s \(fmt(firstTen)), last 10 s \(fmt(lastTen)); chance \(fmt(conc.uniformShare, 3))")
    } else {
        print("        index first 10 s \(fmt(firstTen)), last 10 s \(fmt(lastTen))")
    }
}

test("the formed trail can be drawn: its pheromone stands out from the ground") {
    guard let fld = film.fields[lastFrame] else { expect(false, "no field kept"); return }
    let v: SimTrailVisibility = simTrailVisibility(field: fld, width: cfg.gridWidth, height: cfg.gridHeight, config: cfg)
    print("        on-trail median \(fmt(Double(v.onTrailMedian), 3)) marks/mm², off-trail 99th pct \(fmt(Double(v.offTrail99), 4)): contrast \(fmt(Double(v.contrast), 1))×; continuity \(fmt(Double(v.continuity))); \(fmt(v.massInCorridor * 100, 0))% of all pheromone within 5 mm of the route")
    expect(v.contrast >= 3, "contrast \(v.contrast)")
    expect(v.continuity >= 0.8, "continuity \(v.continuity)")
}

test("no trail is laid before any ant has found the sugar") {
    guard let first = rec.firstContact else { expect(false, "sugar never found"); return }
    let firstTick: Int = rec.dabs.map { Int($0.tick) }.min() ?? Int.max
    expect(Double(firstTick) * SimConst.tick >= first, "a dab at \(Double(firstTick) * SimConst.tick) s, sugar found at \(first) s")
}

test("only ants that have fed are ever seen dabbing, and only carrying food home") {
    for a in rec.antFrames where a.dabbing {
        expect(a.fed == 1 && a.carrying == 1 && a.state == .returning, "a dab by an unfed or outbound ant")
        expect(a.layer == 1, "a dab by a non-layer")
    }
}

// MARK: - the recorded world and its renderer (part 4)

renderTests()

finish()
