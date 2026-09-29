// Tests for lib/ant/v1, the shared ant. The anatomy and gait are checked on
// the drawn geometry (which feet are really on the ground, and whether they
// move while they are), on three kinds of walk a simulation might ask for:
// straight, a steady turn, and a wandering path with a stop in it. The
// distance function is checked from the same Metal source a scene pastes in.
// And the module is checked against its lineage: step 55's own Anatomy,
// Walk and Render sources are compiled into this test binary unchanged
// (read from ../../../step055_ant_trail_5, never edited), and the module
// must pose the same ant and draw the same distances.
//
// ANT_MUTANT=nonTripod|sliding|segments13 breaks the ant on purpose;
// `make mutants` requires the suite to fail for each.

import Foundation
import Metal
import simd

let mutant: AntV1.Mutant = {
    switch ProcessInfo.processInfo.environment["ANT_MUTANT"] {
    case "nonTripod": return .nonTripod
    case "sliding": return .sliding
    case "segments13": return .segments13
    default: return .none
    }
}()

typealias Posed = AntV1.Posed

// MARK: - the walks

/// A straight walk along +x through the origin, 30 mm, sampled every 5 µm.
let fineStep: Float = 0.005
let straightDistances: [Float] = (0..<6000).map { Float($0) * fineStep - 10 }
func straightWalk(_ m: AntV1.Mutant) -> [Posed] {
    straightDistances.map { d in
        AntV1.pose(at: SIMD2<Float>(d, 0), heading: 0, distance: d, time: d / AntV1.walkingSpeed, mutant: m)
    }
}
let straight: [Posed] = straightWalk(mutant)

/// A steady turn: a circle of radius `turnRadius`, recorded into a Track.
/// 25 mm is about six body lengths — tighter than a trail follower turns.
let turnRadius: Float = 25
func arcGround(_ s: Float, radius r: Float) -> AntV1.Ground {
    let h: Float = s / r
    return AntV1.Ground(SIMD2<Float>(r * sin(h), r * (1 - cos(h))), heading: h)
}
func recordTrack(from s0: Float, to s1: Float, step: Float, _ g: (Float) -> AntV1.Ground) -> AntV1.Track {
    var t = AntV1.Track()
    var s: Float = s0
    while s <= s1 { t.append(distance: s, g(s)); s += step }
    return t
}
let arcTrack: AntV1.Track = recordTrack(from: -5, to: 40, step: 0.05) { arcGround($0, radius: turnRadius) }
let arcWalk: [Posed] = (0..<5000).map { k in
    let d: Float = Float(k) * fineStep
    return AntV1.pose(distance: d, time: d / AntV1.walkingSpeed, track: arcTrack, mutant: mutant)
}

/// A wandering path, integrated as a simulation would (heading changes with
/// distance, position steps along it), with a dab stop in the middle: while
/// it stands, distance does not advance and time does.
/// Its tightest turn is λ / (2π · amplitude) = 6.4 mm, just inside the
/// module's limit (`AntV1.minimumTurnRadius`).
let wanderAmp: Float = 0.4          // rad
let wanderWave: Float = 16.0        // mm
func wanderHeading(_ s: Float) -> Float { wanderAmp * sin(2 * Float.pi * s / wanderWave) }
let wanderTrack: AntV1.Track = {
    var t = AntV1.Track()
    var p = SIMD2<Float>(0, 0)
    var s: Float = 0
    let ds: Float = 0.02
    t.append(distance: 0, AntV1.Ground(p, heading: wanderHeading(0)))
    while s < 40 {
        let h: Float = wanderHeading(s + ds / 2)
        p += SIMD2<Float>(cos(h), sin(h)) * ds
        s += ds
        t.append(distance: s, AntV1.Ground(p, heading: wanderHeading(s)))
    }
    return t
}()
/// The simulation's clock: walk at speed, stop for a dab centred at 0.5 s.
let dabCentre: Float = 0.5
let wanderTimes: [Float] = (0..<1000).map { Float($0) * 0.001 }
func wanderDistance(_ t: Float) -> Float {
    // Distance = speed × (t − the walking the stop has cost so far).
    let lost: Float = AntV1.plateauIntegral(t - dabCentre, hold: AntV1.dabHoldReal, ramp: AntV1.dabRampReal)
    return AntV1.walkingSpeed * (t - lost)
}
let wanderWalk: [Posed] = wanderTimes.map { t in
    let state = AntV1.State(dab: AntV1.dabStop(t - dabCentre).dab)
    return AntV1.pose(distance: wanderDistance(t), time: t, track: wanderTrack, state: state, mutant: mutant)
}

let walks: [(name: String, ants: [Posed])] = [("straight", straight), ("turning r = 25 mm", arcWalk), ("wandering, with a dab", wanderWalk)]

/// A foot is down when its lowest point is on the ground.
func down(_ a: Posed, _ j: Int) -> Bool { a.feet[j].y < 1e-6 }

// MARK: - anatomy

section("the worker, against the literature")

test("each antenna has 12 segments — scape plus an 11-segment funiculus — joined end to end, every pose") {
    for (name, ants) in walks {
        for a in stride(from: 0, to: ants.count, by: 37).map({ ants[$0] }) {
            expectEqual(a.model.antennae.count, 2)
            for i in 0..<2 {
                let segs: [AntV1.Shape] = a.shapes.filter { $0.part == .antenna && $0.index == i }
                expectEqual(segs.count, AntV1.workerAntennaSegments)
                var gap: Float = 0
                for k in 1..<segs.count { gap = max(gap, simd_distance(segs[k - 1].b, segs[k].a)) }
                expect(gap < 1e-5, "\(name): antenna \(i) segments part by \(gap) mm")
                expect(segs[0].light, "\(name): the scape is not the pale cuticle")
            }
        }
    }
    expectEqual(AntV1.workerAntennaSegments, 12)
}

test("six legs, every one joined to the mesosoma — none to the head, petiole or gaster") {
    for (name, ants) in walks {
        for a in stride(from: 0, to: ants.count, by: 97).map({ ants[$0] }) {
            expectEqual(a.model.legs.count, 6)
            let meso: [AntV1.Shape] = a.model.body.filter { $0.part == .mesosoma }
            let others: [AntV1.Shape] = a.model.body.filter { (s: AntV1.Shape) -> Bool in
                let p: AntV1.Part = s.part
                return p == .head || p == .petiole || p == .gaster
            }
            for leg in a.model.legs {
                expect(meso.contains { $0.contains(leg.root) }, "\(name): \(leg.name) root is not inside the mesosoma")
                expect(!others.contains { $0.contains(leg.root) }, "\(name): \(leg.name) root is inside another tagma")
                expect(simd_distance(leg.shapes[0].a, leg.root) < 1e-6, "\(leg.name) coxa does not start at its root")
            }
            expectEqual(a.shapes.filter { $0.part == .leg }.count, 6 * 9)
            expectEqual(Set(a.model.legs.map { $0.side > 0 ? "R" : "L" }).count, 2)
        }
    }
}

test("one petiole between mesosoma and gaster, as in all Formicinae; Seifert's head; 3.4–5.0 mm long") {
    let a: Posed = straight[0]
    let pet: [AntV1.Shape] = a.model.body.filter { $0.part == .petiole }
    expectEqual(pet.count, 1)
    expect(abs(AntV1.headWidth - 0.941) < 0.002, "CW \(AntV1.headWidth)")
    expect(abs(AntV1.headLength - 1.011) < 0.002, "CL \(AntV1.headLength)")
    expect(abs(AntV1.scapeLength / AntV1.cephalicSize - 0.979) < 1e-4)
    print(String(format: "        %.2f mm long", AntV1.baseLength))
    expect(AntV1.workerLengthRange.contains(AntV1.baseLength), "the ant is \(AntV1.baseLength) mm")
}

test("no leg is ever stretched: femur and tibia keep their lengths on every walk") {
    for (name, ants) in walks {
        var worst: Float = 0
        for a in ants {
            for (j, leg) in a.model.legs.enumerated() {
                let spec: AntV1.LegSpec = AntV1.legSpecs[j % 3]
                worst = max(worst, abs(simd_distance(leg.hip, leg.knee) - spec.femur))
                worst = max(worst, abs(simd_distance(leg.knee, leg.ankle) - spec.tibia))
            }
        }
        print(String(format: "        %@: worst femur/tibia error %.5f mm", name, worst))
        expect(worst < 1e-3, "\(name): a leg segment is off its length by \(worst) mm — a foot out of reach")
    }
}

// MARK: - the gait

section("the gait")

test("alternating tripod, read off the drawn feet: only whole tripods stand alone, and they take turns") {
    for (name, ants) in walks {
        var sequence: [Int] = []
        var previous: Int = -1
        var bad: Int = 0
        for a in ants {
            let set: Set<Int> = Set((0..<6).filter { down(a, $0) })
            var now: Int = -2
            if set.count == 6 { now = -1 } else if set == AntV1.tripodA { now = 0 } else if set == AntV1.tripodB { now = 1 }
            if now == -2 {
                bad += 1
                if bad < 3 { expect(false, "\(name): feet \(set.sorted()) down at \(a.distance) mm — not a tripod") }
                previous = -1
                continue
            }
            if now >= 0 && now != previous { sequence.append(now) }
            previous = now
        }
        for k in 1..<max(sequence.count, 1) where sequence[k] == sequence[k - 1] {
            expect(false, "\(name): tripod \(sequence[k]) stands alone twice running")
        }
        let strides: Float = (ants.last!.distance - ants[0].distance) / AntV1.strideLength
        print(String(format: "        %@: %d single-tripod phases in %.1f strides", name, sequence.count, strides))
        expect(sequence.contains(0) && sequence.contains(1), "\(name): never stands on one tripod")
        let phases: Float = Float(sequence.count)
        let wholeStrides: Float = strides.rounded(.down)
        expect(phases >= 2 * wholeStrides - 1, "\(name): only \(sequence.count) single-tripod phases")
        expect(bad == 0, "\(name): \(bad) poses with a non-tripod set of feet down")
    }
}

test("stance feet do not slide — straight, turning, wandering and through a stop: within 0.1 µm of where they landed") {
    for (name, ants) in walks {
        var worst: Float = 0
        var stances: Int = 0
        for j in 0..<6 {
            var landed: SIMD3<Float>? = nil
            for a in ants {
                if down(a, j) {
                    if let l = landed { worst = max(worst, simd_distance(l, a.feet[j])) } else { landed = a.feet[j]; stances += 1 }
                } else {
                    landed = nil
                }
            }
        }
        print(String(format: "        %@: %d stances, worst drift %.7f mm", name, stances, worst))
        expect(stances > 10, "\(name): only \(stances) stances")
        expect(worst < 1e-4, "\(name): a stance foot moved \(worst) mm across the ground")
    }
}

test("stance feet touch the ground (lowest point 0), swing feet lift, nothing goes below y = 0") {
    for (name, ants) in walks {
        var worstTouch: Float = 0
        var lowest: Float = 0
        var highestSwing: Float = 0
        for a in ants {
            for (j, leg) in a.model.legs.enumerated() {
                let tip: AntV1.Shape = leg.shapes[leg.shapes.count - 1].placed(scale: a.scale, yaw: a.yaw, at: a.at)
                let tipLow: Float = tip.extent(along: SIMD3(0, 1, 0)).lo
                if down(a, j) { worstTouch = max(worstTouch, abs(tipLow)) } else { highestSwing = max(highestSwing, a.feet[j].y) }
            }
            lowest = min(lowest, a.lowest)
        }
        print(String(format: "        %@: stance foot to ground %.6f mm, lowest %.6f mm, swing up to %.3f mm", name, worstTouch, lowest, highestSwing))
        expect(worstTouch < 1e-4, "\(name): a stance foot is \(worstTouch) mm off the ground")
        expect(lowest > -1e-4, "\(name): part of the ant is \(-lowest) mm below the ground")
        expect(highestSwing > 0.1, "\(name): swing feet barely lift")
    }
}

test("the gait and the geometry agree on which feet are down") {
    var mismatches: Int = 0
    for (_, ants) in walks { for a in ants { for j in 0..<6 where a.stance[j] != down(a, j) { mismatches += 1 } } }
    expectEqual(mismatches, 0)
}

test("when the ant stops, its legs stop: the whole leg set is frozen while distance holds still") {
    var standing: [Posed] = []
    for (k, t) in wanderTimes.enumerated() where abs(t - dabCentre) < AntV1.dabHoldReal / 2 { standing.append(wanderWalk[k]) }
    expect(standing.count > 20, "only \(standing.count) standing samples")
    var worst: Float = 0
    for a in standing { for (f, g) in zip(a.feet, standing[0].feet) { worst = max(worst, simd_distance(f, g)) } }
    print(String(format: "        %d samples standing: feet move %.7f mm", standing.count, worst))
    expect(worst < 1e-6, "feet move \(worst) mm while the ant stands")
}

test("the tightest turn the gait holds: r = \(AntV1.minimumTurnRadius) mm either way — no stretch, nothing underground, tripods, planted") {
    for sgn in [Float(1), Float(-1)] {
        let r: Float = AntV1.minimumTurnRadius
        let track: AntV1.Track = recordTrack(from: -5, to: 30, step: 0.02) { s in
            let g: AntV1.Ground = arcGround(s, radius: r)
            return AntV1.Ground(SIMD2<Float>(g.position.x, sgn * g.position.y), heading: sgn * g.heading)
        }
        var low: Float = 0
        var stretch: Float = 0
        var bad: Int = 0
        var drift: Float = 0
        var landed: [SIMD3<Float>?] = Array(repeating: nil, count: 6)
        for k in 0..<4000 {
            let a: Posed = AntV1.pose(distance: Float(k) * fineStep, time: 0, track: track, mutant: mutant)
            low = min(low, a.lowest)
            for (j, leg) in a.model.legs.enumerated() {
                stretch = max(stretch, abs(simd_distance(leg.knee, leg.ankle) - AntV1.legSpecs[j % 3].tibia))
                if down(a, j) {
                    if let l = landed[j] { drift = max(drift, simd_distance(l, a.feet[j])) } else { landed[j] = a.feet[j] }
                } else { landed[j] = nil }
            }
            let set: Set<Int> = Set((0..<6).filter { down(a, $0) })
            if !(set.count == 6 || set == AntV1.tripodA || set == AntV1.tripodB) { bad += 1 }
        }
        print(String(format: "        turning %@: lowest %.5f mm, worst tibia error %.5f mm, drift %.7f mm, %d non-tripod poses",
                     sgn > 0 ? "right" : "left", low, stretch, drift, bad))
        expect(low > -1e-4 && stretch < 1e-3 && bad == 0 && drift < 1e-4, "the gait breaks at r = \(r) mm")
    }
}

// MARK: - the dab and the antennae

section("the dab and the antennae")

test("at dab = 1 the gaster TIP meets the ground (0 ± 1 µm), the belly stays clear; walking, the gaster is clear") {
    let pressed: Posed = AntV1.pose(at: SIMD2(3, 1), heading: 0.7, distance: 3, time: 0.1, state: AntV1.State(dab: 1), mutant: mutant)
    let tipLow: Float = pressed.shapes.filter { $0.part == .gaster && $0.kind == .roundCone }.map { $0.extent(along: SIMD3(0, 1, 0)).lo }.min() ?? 1
    let belly: Float = pressed.shapes.first { $0.part == .gaster && $0.kind == .ellipsoid }!.extent(along: SIMD3(0, 1, 0)).lo
    let walking: Posed = AntV1.pose(at: SIMD2(3, 1), heading: 0.7, distance: 3, time: 0.1, mutant: mutant)
    let walkLow: Float = walking.shapes.filter { $0.part == .gaster }.map { $0.extent(along: SIMD3(0, 1, 0)).lo }.min() ?? 0
    print(String(format: "        pressed: tip %.6f mm, belly %.3f mm; walking: gaster %.3f mm clear", tipLow, belly, walkLow))
    expect(abs(tipLow) < 1e-3, "tip at \(tipLow) mm")
    expect(belly > 0.03, "belly at \(belly) mm")
    expect(walkLow > 0.1, "walking gaster only \(walkLow) mm clear")
}

test("a dab stop can fall where all six feet are down: changeoverDistance, as step 55 placed its sites") {
    for (s, off) in [(Float(0), Float(0)), (3.3, 0.25), (17.01, 0.9), (-4, 0.5)] {
        let c: Float = AntV1.changeoverDistance(atOrAfter: s, gaitOffset: off)
        let a: Posed = AntV1.pose(at: SIMD2(c, 0), heading: 0, distance: c, time: 0, state: AntV1.State(gaitOffset: off), mutant: mutant)
        let n: Int = (0..<6).filter { down(a, $0) }.count
        expect(c >= s - 1e-5 && c < s + AntV1.strideLength, "changeover \(c) for \(s)")
        expect(n == 6, "only \(n) feet down at the changeover \(c) mm (offset \(off))")
    }
    // Step 55's first dab site is one.
    let c55: Float = AntV1.changeoverDistance(atOrAfter: dabSites[0] - 0.01, gaitOffset: gaitOffset)
    expect(abs(c55 - dabSites[0]) < 1e-3, "step 55's dab site \(dabSites[0]), changeover \(c55)")
}

test("dabStop keeps the lineage's timing: a 0.2 s stop, the tip down only while standing still") {
    var worstDown: Float = 0
    for k in 0..<400 {
        let t: Float = Float(k) * 0.001 - 0.2
        let (speed, dab) = AntV1.dabStop(t)
        if dab > 0.999 { worstDown = max(worstDown, speed) }
    }
    expect(worstDown < 1e-6, "the tip is down while moving at \(worstDown) of speed")
    expect(AntV1.dabStop(0.101).speed == 1 && AntV1.dabStop(-0.101).speed == 1, "the stop lasts beyond 0.2 s")
    expect(AntV1.dabStop(0).speed == 0 && AntV1.dabStop(0).dab == 1)
    let lost: Float = AntV1.plateauIntegral(1, hold: AntV1.dabHoldReal, ramp: AntV1.dabRampReal)
    expect(abs(lost - AntV1.dabWalkingLost) < 1e-6, "a dab costs \(lost) s, not \(AntV1.dabWalkingLost)")
}

test("the antennae sweep just above the ground ahead of the head, and follow the clock") {
    var lowTip: Float = 10
    var zs: [Float] = []
    for a in straight {
        for ant in a.model.antennae {
            expect(ant.reached, "an antenna cannot reach its tip")
            let tip: AntV1.Shape = ant.segments[ant.segments.count - 1].placed(scale: 1, yaw: a.yaw, at: a.at)
            lowTip = min(lowTip, tip.extent(along: SIMD3(0, 1, 0)).lo)
        }
        zs.append(a.model.antennae[0].tipCentre.z)
    }
    let span: Float = (zs.max() ?? 0) - (zs.min() ?? 0)
    print(String(format: "        tip lowest %.3f mm; right tip sweeps %.2f mm side to side; %.2f sweeps a real second", lowTip, span, AntV1.sweepRate))
    expect(lowTip > 0 && lowTip < 0.06, "tip lowest at \(lowTip) mm")
    expect(span > 0.4, "the antenna sweeps only \(span) mm")
    let a0: Posed = AntV1.pose(at: .zero, heading: 0, distance: 0, time: 0, mutant: mutant)
    let half: Posed = AntV1.pose(at: .zero, heading: 0, distance: 0, time: 0, state: AntV1.State(sweepPhase: 0.25), mutant: mutant)
    expect(simd_distance(a0.model.antennae[0].tipCentre, half.model.antennae[0].tipCentre) > 0.2, "sweepPhase does nothing")
}

// MARK: - the API

section("the pose API")

test("deterministic: the same arguments give the same shapes bit for bit, whatever was posed in between") {
    let args: [(SIMD2<Float>, Float, Float, Float, AntV1.State)] = [
        (SIMD2(0, 0), 0, 0, 0, AntV1.State()),
        (SIMD2(12.3, -4.5), 2.1, 17.9, 3.3, AntV1.State(dab: 0.4, sweepPhase: 0.7, gaitOffset: 0.2, scale: 1.03)),
        (SIMD2(-100, 250), -1.2, 1234.5, 60, AntV1.State(dab: 1, scale: 0.96)),
    ]
    var first: [[AntV1.Shape]] = []
    for (p, h, d, t, s) in args { first.append(AntV1.pose(at: p, heading: h, distance: d, time: t, state: s, mutant: mutant).shapes) }
    _ = wanderWalk.count
    for (k, (p, h, d, t, s)) in args.enumerated().reversed() {
        let again: [AntV1.Shape] = AntV1.pose(at: p, heading: h, distance: d, time: t, state: s, mutant: mutant).shapes
        expect(again == first[k], "pose \(k) came out different the second time")
    }
    // Two tracks recorded the same way pose the same ant.
    let t2: AntV1.Track = recordTrack(from: -5, to: 40, step: 0.05) { arcGround($0, radius: turnRadius) }
    let y: Posed = AntV1.pose(distance: arcWalk[1540].distance, time: arcWalk[1540].time, track: t2, mutant: mutant)
    expect(y.shapes == arcWalk[1540].shapes, "same track, different ant")
}

test("the pose is where it was asked for: origin on the ground at the position, facing the heading") {
    let a: Posed = AntV1.pose(at: SIMD2(5, -2), heading: 1.0, distance: 3, time: 0, mutant: mutant)
    expect(simd_distance(a.at, SIMD3<Float>(5, 0, -2)) < 1e-6)
    let head: AntV1.Shape = a.shapes.first { $0.part == .head }!
    let dir: SIMD3<Float> = simd_normalize(head.a - a.at)
    let h: Float = atan2(dir.z, dir.x)
    expect(abs(h - 1.0) < 0.02, "the head points at \(h) rad, not 1.0")
    let scaled: Posed = AntV1.pose(at: .zero, heading: 0, distance: 0, time: 0, state: AntV1.State(scale: 1.04), mutant: mutant)
    let local: Float = AntV1.Model(body: scaled.model.body, legs: [], antennae: []).length
    let placedBody: [AntV1.Shape] = scaled.model.body.filter { $0.part != .eye }.map { $0.placed(scale: 1.04, yaw: 0, at: .zero) }
    let lo: Float = placedBody.map { $0.extent(along: SIMD3(1, 0, 0)).lo }.min() ?? 0
    let hi: Float = placedBody.map { $0.extent(along: SIMD3(1, 0, 0)).hi }.max() ?? 0
    let span: Float = hi - lo
    let want: Float = local * 1.04
    expect(abs(span - want) < 1e-4, "scale 1.04 makes \(span) mm from \(local)")
}

test("the Track fixes the past: it refuses to rewrite recorded distances, and interpolates between samples") {
    var t = AntV1.Track()
    expect(t.append(distance: 0, AntV1.Ground(SIMD2(0, 0), heading: 0)))
    expect(t.append(distance: 1, AntV1.Ground(SIMD2(1, 0), heading: 0.2)))
    expect(!t.append(distance: 1, AntV1.Ground(SIMD2(9, 9), heading: 3)), "rewrote a recorded distance")
    expect(!t.append(distance: 0.5, AntV1.Ground(SIMD2(9, 9), heading: 3)), "wrote into the past")
    let mid: AntV1.Ground = t.ground(0.5)
    let midX: Float = mid.position.x
    let midH: Float = mid.heading
    expect(abs(midX - 0.5) < 1e-6, "mid \(mid)")
    expect(abs(midH - 0.1) < 1e-6, "mid \(mid)")
    let ahead: AntV1.Ground = t.ground(2)
    let wantX: Float = 1 + cos(Float(0.2))
    let wantZ: Float = sin(Float(0.2))
    let offX: Float = abs(ahead.position.x - wantX)
    let offZ: Float = abs(ahead.position.y - wantZ)
    expect(offX < 1e-5 && offZ < 1e-5, "extrapolation \(ahead)")
    // Across ±π the heading goes the short way.
    var w = AntV1.Track()
    w.append(distance: 0, AntV1.Ground(.zero, heading: 3.1))
    w.append(distance: 1, AntV1.Ground(.zero, heading: -3.1))
    let h: Float = w.ground(0.5).heading
    expect(abs(abs(h) - Float.pi) < 0.01, "heading \(h) went the long way round")
    expect(t.isSettled(1 - AntV1.strideLength) && !t.isSettled(0.5))
}

// MARK: - the distance function

section("the distance function")

let gpuProbe: AntProbe? = try? AntProbe()
/// A few ants from the walks, placed apart, to probe together.
let probeAnts: [Posed] = [straight[1234], arcWalk[2500], wanderWalk[500], wanderWalk[800]]

test("outside the ant, no distance claims more room than a ray stepping ×\(AntV1.stepScale) allows") {
    guard let g = gpuProbe else { expect(false, "no GPU / the Metal source did not compile"); return }
    var rng = SystemRandomNumberGenerator()
    var a: [SIMD3<Float>] = []
    var b: [SIMD3<Float>] = []
    for k in 0..<200_000 {
        let c: SIMD3<Float> = probeAnts[k % probeAnts.count].at
        let span: SIMD3<Float> = k % 2 == 0 ? SIMD3(4.0, 1.4, 2.4) : SIMD3(2.6, 1.0, 1.0)
        let off = SIMD3<Float>(Float.random(in: -span.x...span.x, using: &rng), Float.random(in: 0...span.y, using: &rng),
                               Float.random(in: -span.z...span.z, using: &rng))
        let p: SIMD3<Float> = c + off
        let d: SIMD3<Float> = simd_normalize(SIMD3<Float>(Float.random(in: -1...1, using: &rng), Float.random(in: -1...1, using: &rng),
                                                          Float.random(in: -1...1, using: &rng)))
        let step: Float = k % 2 == 0 ? 0.01 : 0.004
        a.append(p)
        b.append(p + d * step)
    }
    do {
        let pa: [SIMD4<Float>] = try g.distance(a, ants: probeAnts)
        let pb: [SIMD4<Float>] = try g.distance(b, ants: probeAnts)
        var worst: Float = 0
        var n: Int = 0
        for i in 0..<a.count where pa[i].x > 0 && pb[i].x > 0 {
            worst = max(worst, abs(pa[i].x - pb[i].x) / simd_distance(a[i], b[i]))
            n += 1
        }
        print(String(format: "        %d pairs outside: worst over-report %.3f; the ray allows %.3f", n, worst, 1 / AntV1.stepScale))
        expect(n > 10_000, "only \(n) pairs outside the ant")
        expect(worst * AntV1.stepScale <= 1.0, "the ant's distance oversteps: \(worst)")
    } catch { expect(false, "\(error)") }
}

test("the room a ray takes is really empty: no shape reaches inside distance × \(AntV1.stepScale) of a point outside") {
    guard let g = gpuProbe else { expect(false, "no GPU"); return }
    var rng = SystemRandomNumberGenerator()
    var pts: [SIMD3<Float>] = []
    for k in 0..<20_000 {
        let c: SIMD3<Float> = probeAnts[k % probeAnts.count].at
        pts.append(c + SIMD3<Float>(Float.random(in: -3...3, using: &rng), Float.random(in: 0...1.4, using: &rng),
                                    Float.random(in: -2.2...2.2, using: &rng)))
    }
    do {
        let d: [SIMD4<Float>] = try g.distance(pts, ants: probeAnts)
        let shapes: [AntV1.Shape] = probeAnts.flatMap { $0.shapes }
        var violations: Int = 0
        var checked: Int = 0
        for (i, p) in pts.enumerated() where d[i].x > 0.002 {
            let r: Float = d[i].x * AntV1.stepScale
            for _ in 0..<8 {
                let u = SIMD3<Float>(Float.random(in: -1...1, using: &rng), Float.random(in: -1...1, using: &rng), Float.random(in: -1...1, using: &rng))
                if simd_length(u) > 1 { continue }
                let q: SIMD3<Float> = p + u * r
                checked += 1
                for s in shapes {
                    let (c, br) = s.bound
                    if simd_distance(c, q) > br { continue }
                    if s.contains(q) { violations += 1; break }
                }
            }
        }
        print("        \(checked) points inside the stepped balls, \(violations) inside a shape")
        expect(checked > 50_000, "only \(checked) checked")
        expectEqual(violations, 0)
    } catch { expect(false, "\(error)") }
}

test("the GPU draws the ant the CPU placed: every stance foot's lowest point is on the drawn surface") {
    guard let g = gpuProbe else { expect(false, "no GPU"); return }
    var pts: [SIMD3<Float>] = []
    for a in probeAnts { for j in 0..<6 where down(a, j) { pts.append(a.feet[j] + SIMD3<Float>(0, 0.0005, 0)) } }
    do {
        let q: [SIMD4<Float>] = try g.distance(pts, ants: probeAnts)
        let worst: Float = q.map { abs($0.x) }.max() ?? 1
        print(String(format: "        %d stance feet, drawn surface within %.4f mm of each", pts.count, worst))
        expect(!pts.isEmpty && worst < 0.003, "a stance foot is \(worst) mm from the drawn ant")
    } catch { expect(false, "\(error)") }
}

test("the cuticle: dark body, pale mandibles and scape, dark eyes, as step 32 drew them") {
    guard let g = gpuProbe else { expect(false, "no GPU"); return }
    let a: Posed = probeAnts[0]
    func onShape(_ pick: (AntV1.Shape) -> Bool) -> SIMD3<Float> {
        let s: AntV1.Shape = a.shapes.first(where: pick)!
        if s.kind != .roundCone { return s.a }
        let sum: SIMD3<Float> = s.a + s.b
        return sum / 2
    }
    // The eye's centre lies inside the head; probe just inside its outer face.
    let eye: AntV1.Shape = a.shapes.first { $0.part == .eye }!
    let outward: SIMD3<Float> = simd_cross(eye.xAxis, eye.yAxis) * (simd_dot(eye.a - a.at, AntV1.turn(SIMD3(0, 0, 1), a.yaw)) > 0 ? 1 : -1)
    let pts: [SIMD3<Float>] = [onShape { $0.part == .mesosoma }, onShape { $0.part == .mandible }, onShape { $0.part == .antenna && $0.light },
                               eye.a + outward * (eye.b.z * 0.9)]
    do {
        let s: [SIMD4<Float>] = try g.surface(pts, ants: [a])
        let alb: [SIMD3<Float>] = s.map { SIMD3<Float>($0.x, $0.y, $0.z) }
        expect(simd_distance(alb[0], AntV1.cuticleDark) < 1e-4, "mesosoma \(alb[0])")
        expect(simd_distance(alb[1], AntV1.cuticlePale) < 1e-4, "mandible \(alb[1])")
        expect(simd_distance(alb[2], AntV1.cuticlePale) < 1e-4, "scape \(alb[2])")
        expect(simd_distance(alb[3], AntV1.cuticleEye) < 1e-4, "eye \(alb[3])")
        expect(abs(s[3].w - AntV1.alphaEye) < 1e-6 && abs(s[0].w - AntV1.alphaDark) < 1e-6, "alphas \(s.map { $0.w })")
    } catch { expect(false, "\(error)") }
}

// MARK: - the lineage

section("the lineage: the module is step 55's ant")

/// Step 55's trail as a path: its `pose(x)` is where its ants stand at x.
func step55Path(_ s: Float) -> AntV1.Ground {
    let (at, yaw) = pose(s)
    return AntV1.Ground(SIMD2<Float>(at.x, at.z), heading: yaw)
}

/// The module's ant for step 55's ant 0 at its own clock τ: same path, same
/// distance, its clock turned into real seconds, its gait offset and dab.
func moduleFor55(tau: Float) -> Posed {
    let x: Float = trailX(tau)
    let state = AntV1.State(dab: dabAmount(tau), gaitOffset: gaitOffset)
    return AntV1.pose(distance: x, time: tau / slowdown, path: step55Path, state: state, mutant: mutant)
}

let lineageTaus: [Float] = (0..<1600).map { Float($0) * 0.01 - 8 }

test("the same pose gives step 55's shapes: every primitive within 0.1 µm (\(lineageTaus.count) poses, dabs included)") {
    var worst: Float = 0
    var worstFoot: Float = 0
    var kindMismatch: Int = 0
    var countMismatch: Int = 0
    var stanceMismatch: Int = 0
    for tau in lineageTaus {
        let old: WorldAnt = worldAnt(tau: tau, index: 0, mutant: .none)
        let new: Posed = moduleFor55(tau: tau)
        if old.shapes.count != new.shapes.count { countMismatch += 1; continue }
        for (o, n) in zip(old.shapes, new.shapes) {
            if o.kind.rawValue != n.kind.rawValue || o.part.rawValue != n.part.rawValue || o.light != n.light || o.index != n.index { kindMismatch += 1 }
            let ds: Float = max(simd_distance(o.a, n.a), simd_distance(o.b, n.b))
            let dr: Float = max(abs(o.ra - n.ra), abs(o.rb - n.rb))
            let dx: Float = max(simd_distance(o.xAxis, n.xAxis), simd_distance(o.yAxis, n.yAxis))
            worst = max(worst, ds, dr, dx)
        }
        for (f, g) in zip(old.feet, new.feet) { worstFoot = max(worstFoot, simd_distance(f, g)) }
        if old.stance != new.stance { stanceMismatch += 1 }
    }
    print(String(format: "        worst primitive difference %.2e mm, worst foot %.2e mm", worst, worstFoot))
    expectEqual(countMismatch, 0)
    expectEqual(kindMismatch, 0)
    expectEqual(stanceMismatch, 0)
    expect(worst < 1e-4, "the module's ant differs from step 55's by \(worst) mm")
    expect(worstFoot < 1e-4, "a foot differs from step 55's by \(worstFoot) mm")
}

test("the same constants as step 55: speed, stride, duty factor, tripods, lift, dab, sweep") {
    expectEqual(AntV1.walkingSpeed, realSpeed)
    expectEqual(AntV1.strideLength, strideLength)
    expectEqual(AntV1.dutyFactor, dutyFactor)
    expectEqual(AntV1.tripodA, tripodA)
    expectEqual(AntV1.tripodB, tripodB)
    expectEqual(AntV1.footLift, footLift)
    expectEqual(AntV1.baseLength, baseLength)
    expectEqual(AntV1.dabPauseReal, dabPauseReal)
    expectEqual(AntV1.dabFull.bend, dabFull.bend)
    expectEqual(AntV1.dabFull.flex, dabFull.flex)
    expect(abs(AntV1.sweepRate - sweepRate * slowdown) < 1e-5, "sweep \(AntV1.sweepRate) vs \(sweepRate * slowdown)")
    // The cuticle and step size as step 55's kernel text writes them: the
    // numbers on each named line, parsed.
    let kernel: String = kernelSource(camera: camera)
    func numbers(_ name: String) -> [Float] {
        guard let line = kernel.split(separator: "\n").first(where: { $0.contains("constant") && $0.contains(" \(name) = ") }) else { return [] }
        let rhs: Substring = line.split(separator: "=")[1]
        let cleaned: String = rhs.replacingOccurrences(of: "float3(", with: "").replacingOccurrences(of: ")", with: "").replacingOccurrences(of: ";", with: "")
        return cleaned.split(separator: ",").compactMap { Float($0.trimmingCharacters(in: .whitespaces)) }
    }
    func vec(_ v: SIMD3<Float>) -> [Float] { [v.x, v.y, v.z] }
    let expected: [(String, [Float])] = [("DARK", vec(AntV1.cuticleDark)), ("PALE", vec(AntV1.cuticlePale)), ("EYE", vec(AntV1.cuticleEye)),
                                         ("PALE_ALPHA", [AntV1.alphaPale]), ("STEP_SCALE", [AntV1.stepScale]), ("CUTICLE_F0", [AntV1.cuticleF0])]
    for (name, want) in expected {
        let got: [Float] = numbers(name)
        expect(got == want, "step 55's \(name) is \(got), the module's \(want)")
    }
    expect(kernel.contains("float k = group == 1 ? 0.10 : (group == 2 ? 0.04 : 0.012);"), "step 55's blends moved")
    expect(kernel.contains("return smin(smin(g1, g2, 0.03), g0, 0.012);"), "step 55's group join moved")
    expectEqual(AntV1.blendMesosoma, 0.10)
    expectEqual(AntV1.blendHead, 0.04)
    expectEqual(AntV1.blendOther, 0.012)
    expectEqual(AntV1.blendGroups, 0.03)
}

test("the GPU draws step 55's distances: the module's Metal and step 55's kernel agree at 60,000 points") {
    guard let g = gpuProbe else { expect(false, "no GPU"); return }
    var rng = SystemRandomNumberGenerator()
    do {
        let old = try TrailRenderer(width: 64, height: 13, mutant: .none, on: g.device)
        var worst: Float = 0
        var n: Int = 0
        for t in [Float(0.3), 2.2, 4.4] {
            let theirs: [WorldAnt] = antsInView(time: t, range: -40...40, mutant: .none)
            let ours: [Posed] = theirs.map { moduleFor55(tau: $0.tau) }
            var pts: [SIMD3<Float>] = []
            for k in 0..<20_000 {
                let c: SIMD3<Float> = theirs[k % theirs.count].at
                pts.append(c + SIMD3<Float>(Float.random(in: -3...3, using: &rng), Float.random(in: 0...1.4, using: &rng),
                                            Float.random(in: -2...2, using: &rng)))
            }
            let a: [SIMD3<Float>] = try old.probe(pts, time: t)
            let b: [SIMD4<Float>] = try g.distance(pts, ants: ours)
            for i in 0..<pts.count where a[i].z < 1 {
                worst = max(worst, abs(a[i].z - b[i].x))
                n += 1
            }
        }
        print(String(format: "        %d points within 1 mm of an ant: worst difference %.2e mm", n, worst))
        expect(n > 20_000, "only \(n) points near the ants")
        expect(worst < 1e-4, "the module draws the ant \(worst) mm away from step 55's")
    } catch { expect(false, "\(error)") }
}

finish()
