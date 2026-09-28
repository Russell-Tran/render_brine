// Tests for step 57 (step 44's, carried over and moved onto the wood, and
// new ones for the wood and for motion you can see). The gait is checked from
// the drawn geometry — which feet are actually on the wood, and whether they
// move while they are — not from the timetable that placed them; contact by
// measuring the gap between each foot and a dense sampling of the planed
// face, and penetration both ways (the tarsus's underside against the face,
// the face's points against the tarsus); the loop from the ants' positions
// and from rendered frames; the distance function and the wood from the same
// kernel source that draws the picture; the motion from rendered frames.
//
// TRAIL_MUTANT=nonTripod|sliding|rewind|sink|swapRoughness|frozen breaks the
// scene on purpose; `make mutants` requires the suite to fail for each.

import Foundation
import Metal
import simd

let mutant: Mutant = {
    switch ProcessInfo.processInfo.environment["TRAIL_MUTANT"] {
    case "nonTripod": return .nonTripod
    case "sliding": return .sliding
    case "rewind": return .rewind
    case "sink": return .sink
    case "swapRoughness": return .swapRoughness
    case "frozen": return .frozen
    default: return .none
    }
}()
// The wood reads the mutant too (before anything touches it).
let _setWood: Void = { woodMutant = mutant }()

func deg(_ r: Float) -> Float { r * 180 / Float.pi }

/// Holds the renderer so a failure to find a GPU is a test failure, not a crash.
final class GPUBox {
    let renderer: TrailRenderer? = {
        guard let d = try? findDevice() else { return nil }
        return try? TrailRenderer(width: 480, height: Int((480 * frameAspect).rounded()), mutant: mutant, on: d)
    }()
}
let gpu = GPUBox()

/// Fine time samples across one loop, and the ants (copy 0) at each.
let fineStep: Float = 0.005
let fineTimes: [Float] = (0..<Int(loopSeconds / fineStep)).map { Float($0) * fineStep }
let fine: [[WorldAnt]] = fineTimes.map { t in antPlans.map { worldAnt($0, copy: 0, time: t, mutant: mutant) } }

// MARK: - measuring contact against the wood, independently of how it was set

/// The gap between a sphere and the planed face, mm: the least distance from
/// the sphere's surface to points ON the face, sampled on a grid under the
/// sphere and refined round the nearest. Negative if the face is inside.
func sphereGap(_ c: SIMD3<Float>, _ r: Float) -> Float {
    func dist(_ dx: Float, _ dz: Float) -> Float {
        let x: Float = c.x + dx
        let z: Float = c.z + dz
        let q = SIMD3<Float>(x, woodHeight(x, z), z)
        let d: Float = simd_distance(q, c) - r
        return q.y > c.y ? -abs(d) - 1 : d      // face above the centre: deep inside
    }
    var best: Float = 1e9
    var bx: Float = 0
    var bz: Float = 0
    let n: Int = 10
    let step: Float = r * 1.2 / Float(n)
    for i in -n...n {
        for j in -n...n {
            let v: Float = dist(Float(i) * step, Float(j) * step)
            if v < best { best = v; bx = Float(i) * step; bz = Float(j) * step }
        }
    }
    var s: Float = step
    for _ in 0..<4 {
        s /= 4
        let cx: Float = bx
        let cz: Float = bz
        for i in -4...4 {
            for j in -4...4 {
                let v: Float = dist(cx + Float(i) * s, cz + Float(j) * s)
                if v < best { best = v; bx = cx + Float(i) * s; bz = cz + Float(j) * s }
            }
        }
    }
    return best
}

/// Foot j's round tip, in the world: centre and radius.
func tip(_ a: WorldAnt, _ j: Int) -> (centre: SIMD3<Float>, radius: Float) {
    let s: Shape = a.model.legs[j].shapes[a.model.legs[j].shapes.count - 1].placed(scale: antPlans[a.id].scale, yaw: a.yaw, at: a.at)
    return (s.b, s.rb)
}

/// The foot proper's gap to the wood: the last tarsomere's underside, every
/// 2 µm, and its round tip, against the face.
func footGap(_ a: WorldAnt, _ j: Int) -> Float {
    let last: Shape = a.model.legs[j].shapes[a.model.legs[j].shapes.count - 1].placed(scale: antPlans[a.id].scale, yaw: a.yaw, at: a.at)
    var g: Float = sphereGap(last.b, last.rb)
    for (q, _) in undersidePoints(last, spacing: 0.002) where q.y < woodHeightMax + 0.02 {
        g = min(g, q.y - woodHeight(q.x, q.z))
    }
    return g
}

/// A foot is down, read from the drawing, when the foot proper is within
/// 16 µm of the wood — under a pixel at the GIF's width (21 µm) — and is
/// not moving: it stands where it stood 5 ms before, or where it will stand
/// 5 ms on. A planted foot touches at the tightest moment of its stance and
/// may stand a few micrometres clear at others (Walk.swift, distalLift); a
/// test bounds that. A swinging foot is always moving.
let downGap: Float = 0.016
func down(_ a: WorldAnt, _ j: Int, before: WorldAnt, after: WorldAnt) -> Bool {
    let still: Bool = simd_distance(a.feet[j], before.feet[j]) < 1e-5 || simd_distance(a.feet[j], after.feet[j]) < 1e-5
    return still && footGap(a, j) < downGap
}

/// The drawing's reading of every fine sample, once (it is not cheap).
let fineDown: [[[Bool]]] = fine.enumerated().map { (k, ants) in
    ants.map { a in
        let t: Float = fineTimes[k]
        let before: WorldAnt = worldAnt(antPlans[a.id], copy: 0, time: t - fineStep, mutant: mutant)
        let after: WorldAnt = worldAnt(antPlans[a.id], copy: 0, time: t + fineStep, mutant: mutant)
        return (0..<6).map { down(a, $0, before: before, after: after) }
    }
}

/// How far any low part of an ant reaches INTO the wood, mm (0 if none):
/// the underside of every shape near the face, sampled every 2 µm, against
/// the face's height there.
func penetration(_ a: WorldAnt) -> Float {
    var worst: Float = 0
    for s in a.shapes where s.kind == .roundCone && s.extent(along: SIMD3(0, 1, 0)).lo < woodHeightMax + 1e-3 {
        for (q, _) in undersidePoints(s, spacing: 0.002) where q.y < woodHeightMax {
            worst = max(worst, woodHeight(q.x, q.z) - q.y)
        }
    }
    return worst
}

/// The other way round: points ON the face, every 3 µm under each low round
/// cone, that lie inside it. Returns the deepest (0 if none) and the count
/// of points looked at.
func faceInside(_ a: WorldAnt) -> (depth: Float, points: Int) {
    var worst: Float = 0
    var n: Int = 0
    for s in a.shapes where s.kind == .roundCone && s.extent(along: SIMD3(0, 1, 0)).lo < woodHeightMax + 1e-3 {
        let x0: Float = s.extent(along: SIMD3(1, 0, 0)).lo
        let x1: Float = s.extent(along: SIMD3(1, 0, 0)).hi
        let z0: Float = s.extent(along: SIMD3(0, 0, 1)).lo
        let z1: Float = s.extent(along: SIMD3(0, 0, 1)).hi
        var x: Float = x0
        while x <= x1 {
            var z: Float = z0
            while z <= z1 {
                let q = SIMD3<Float>(x, woodHeight(x, z), z)
                let d: Float = roundConeDistance(q, s.a, s.b, s.ra, s.rb)
                worst = max(worst, -d)
                n += 1
                z += 0.003
            }
            x += 0.003
        }
    }
    return (worst, n)
}

section("each ant, against the literature")

test("each antenna has 12 segments — scape plus an 11-segment funiculus — as a worker's does") {
    for k in stride(from: 0, to: fine.count, by: 97) {
        for a in fine[k] {
            expectEqual(a.model.antennae.count, 2)
            for i in 0..<2 {
                let drawn: Int = a.shapes.filter { $0.part == .antenna && $0.index == i }.count
                expectEqual(drawn, workerAntennaSegments)
            }
        }
    }
}

test("six legs, every one joined to the mesosoma — none to the head, petiole or gaster") {
    for k in stride(from: 0, to: fine.count, by: 97) {
        for a in fine[k] {
            expectEqual(a.model.legs.count, 6)
            let meso: [Shape] = a.model.body.filter { $0.part == .mesosoma }
            let others: [Shape] = a.model.body.filter { $0.part == .head || $0.part == .petiole || $0.part == .gaster }
            for leg in a.model.legs {
                expect(meso.contains { $0.contains(leg.root) }, "\(leg.name) root is not inside the mesosoma")
                expect(!others.contains { $0.contains(leg.root) }, "\(leg.name) root is inside another tagma")
                expect(simd_distance(leg.shapes[0].a, leg.root) < 1e-6, "\(leg.name) coxa does not start at its root")
            }
            expectEqual(a.shapes.filter { $0.part == .leg }.count, 6 * 9)
        }
    }
}

test("one petiole, a single node between mesosoma and gaster, as in all Formicinae") {
    for a in fine[0] {
        let pet: [Shape] = a.model.body.filter { $0.part == .petiole }
        expectEqual(pet.count, 1)
        let mesoBack: Float = a.model.body.filter { $0.part == .mesosoma }.map { $0.extent(along: SIMD3(1, 0, 0)).lo }.min() ?? 0
        let gasterFront: Float = a.model.body.filter { $0.part == .gaster }.map { $0.extent(along: SIMD3(1, 0, 0)).hi }.max() ?? 0
        expect(pet[0].a.x < mesoBack + 0.05 && pet[0].a.x > gasterFront - 0.05, "petiole out of place")
    }
}

test("the head is Seifert's: CS 976 µm, CL/CW 1.074, scape SL/CS 0.979") {
    expect(abs(headWidth - 0.941) < 0.002, "CW \(headWidth)")
    expect(abs(headLength - 1.011) < 0.002, "CL \(headLength)")
    expect(abs(scapeLength / cephalicSize - 0.979) < 1e-4)
    let head: Shape = bodyShapes(gasterBend: 0)[0]
    expect(abs(head.b.z * 2 - headWidth) < 1e-4, "drawn head width \(head.b.z * 2)")
}

test("every worker is 3.4–5.0 mm long, mandibles to gaster tip") {
    for p in antPlans {
        let l: Float = baseLength * p.scale
        print(String(format: "        ant %d: %.2f mm", p.id, l))
        expect(workerLengthRange.contains(l), "ant \(p.id) is \(l) mm")
    }
}

test("no leg is ever stretched or squashed: femur, tibia and tarsus keep their lengths") {
    var worst: Float = 0
    for ants in fine {
        for a in ants {
            for (j, leg) in a.model.legs.enumerated() {
                let spec: LegSpec = legSpecs[j % 3]
                worst = max(worst, abs(simd_distance(leg.hip, leg.knee) - spec.femur))
                worst = max(worst, abs(simd_distance(leg.knee, leg.ankle) - spec.tibia))
            }
        }
    }
    print(String(format: "        worst femur/tibia length error %.5f mm", worst))
    expect(worst < 1e-3, "a leg segment is off its length by \(worst) mm: the foot is out of reach")
}

section("the gait")

test("alternating tripod: whenever three feet are down they are one tripod, and the tripods take turns") {
    var singles: [String: Int] = [:]
    var bad: Int = 0
    for id in 0..<3 {
        var sequence: [Int] = []       // one entry per single-tripod phase: 0 = A alone, 1 = B alone
        var previous: Int = -1         // what the last sample was: -1 changeover, 0 A, 1 B
        for (k, _) in fine.enumerated() {
            let set: Set<Int> = Set((0..<6).filter { fineDown[k][id][$0] })
            var now: Int = -2
            if set.count == 6 { now = -1 }                       // both tripods: a changeover
            else if set == tripodA { now = 0 }
            else if set == tripodB { now = 1 }
            if now == -2 {
                bad += 1
                if bad < 4 { expect(false, "ant \(id) has feet \(set.sorted()) down — not a tripod") }
                previous = -1
                continue
            }
            if now >= 0 && now != previous { sequence.append(now) }
            previous = now
        }
        // Alternation: no tripod stands alone twice running — every phase
        // on one tripod is followed by a phase on the other.
        for k in 1..<max(sequence.count, 1) where sequence[k] == sequence[k - 1] {
            expect(false, "ant \(id): tripod \(sequence[k]) stands alone twice running (phase \(k))")
        }
        singles["ant \(id)"] = sequence.count
        expect(sequence.contains(0) && sequence.contains(1), "ant \(id) never stands on one of its tripods")
        // Two tripod phases per stride: 2 × strides walked in the loop.
        expect(sequence.count >= 2 * stridesPerGroup - 1, "ant \(id): only \(sequence.count) single-tripod phases")
    }
    print("        single-tripod phases per loop: \(singles)")
    expect(bad == 0, "\(bad) samples with a non-tripod set of feet down")
}

test("stance feet do not slide: a planted foot stays within 0.1 µm of where it landed") {
    // Planted: the timetable's stance, which the drawing confirms (the test
    // after next); the drawing alone cannot tell a foot resting from one a
    // tenth of a micrometre into its swing.
    var worst: Float = 0
    var stances: Int = 0
    for id in 0..<3 {
        for j in 0..<6 {
            var landed: SIMD3<Float>? = nil
            for ants in fine {
                let a: WorldAnt = ants[id]
                if a.stance[j] {
                    if let l = landed { worst = max(worst, simd_distance(l, a.feet[j])) } else { landed = a.feet[j]; stances += 1 }
                } else {
                    landed = nil
                }
            }
        }
    }
    print(String(format: "        %d stances, worst drift %.6f mm", stances, worst))
    expect(worst < 1e-4, "a stance foot moved \(worst) mm across the ground")
}

test("planted feet rest ON the wood's relief: each stance touches (gap 0 ± 1 µm) and never stands more than 16 µm clear") {
    // Group the fine samples by stance (ant, leg, where the foot is planted);
    // judge only stances seen whole, from touchdown to lift-off, in the loop.
    var tightest: [String: Float] = [:]
    var whole: Set<String> = []
    var loosest: Float = 0
    var gaps: [Float] = []
    var restHeights: [Float] = []
    var highestSwing: Float = 0
    for (k, ants) in fine.enumerated() {
        for a in ants {
            for j in 0..<6 {
                if a.stance[j] {
                    let g: Float = footGap(a, j)
                    let key: String = String(format: "%d-%d-%.4f-%.4f", a.id, j, a.feet[j].x, a.feet[j].z)
                    tightest[key] = min(tightest[key] ?? 1, g)
                    // Whole: began after the loop did and ended before it ends.
                    if k > 0 && k < fine.count - 1 && !fine[k - 1][a.id].stance[j] { whole.insert(key + "+in") }
                    if k > 0 && k < fine.count - 1 && !fine[k + 1][a.id].stance[j] { whole.insert(key + "+out") }
                    loosest = max(loosest, g)
                    if k % 4 == 0 { gaps.append(g) }
                    restHeights.append(a.feet[j].y)
                } else {
                    highestSwing = max(highestSwing, a.feet[j].y - woodHeight(a.feet[j].x, a.feet[j].z))
                }
            }
        }
    }
    let judged: [Float] = tightest.filter { whole.contains($0.key + "+in") && whole.contains($0.key + "+out") }.map { abs($0.value) }
    let worstTouch: Float = judged.max() ?? 1
    print("        \(judged.count) stances seen whole in the loop")
    gaps.sort()
    let median: Float = gaps.isEmpty ? 0 : gaps[gaps.count / 2]
    let p90: Float = gaps.isEmpty ? 0 : gaps[gaps.count * 9 / 10]
    let lo: Float = restHeights.min() ?? 0
    let hi: Float = restHeights.max() ?? 0
    print(String(format: "        %d stances: at its tightest moment each touches, worst |gap| %.5f mm; clearance through a stance median %.4f, 90%% %.4f, max %.4f mm",
                 tightest.count, worstTouch, median, p90, loosest))
    print(String(format: "        the feet stand between %+.4f and %+.4f mm (the relief); swing lift up to %.3f mm", lo, hi, highestSwing))
    expect(worstTouch < 1e-3, "a stance never comes within \(worstTouch) mm of the wood")
    expect(loosest < downGap, "a planted foot stands \(loosest) mm clear of the wood")
    // Standing on the relief, not on a flat plane: the feet come to rest at
    // many heights, spanning micrometres.
    expect(hi - lo > 0.005, "every foot stands at the same height (\(lo)…\(hi)): the relief is ignored")
    expect(highestSwing > 0.1, "swing feet barely lift: \(highestSwing) mm")
}

test("nothing ever goes into the wood: every low part's underside, every 2 µm, at every 5 ms of the loop") {
    var worst: Float = 0
    var belowPlane: Float = 0
    for ants in fine {
        for a in ants {
            worst = max(worst, penetration(a))
            belowPlane = min(belowPlane, a.lowest)
        }
    }
    print(String(format: "        deepest any underside reaches into the wood: %.6f mm (lowest point of any ant %+.4f mm: the face dips below 0)",
                 worst, belowPlane))
    expect(worst < 1e-3, "an ant reaches \(worst) mm into the wood")
}

test("and from the wood's side: points on the face, every 3 µm under every low part, never inside an ant (every GIF frame)") {
    var worst: Float = 0
    var total: Int = 0
    var frames: Int = 0
    let count: Int = Int((loopSeconds * 100).rounded()) / gifDelayCentiseconds
    for f in stride(from: 0, to: count, by: 5) {
        let t: Float = Float(f * gifDelayCentiseconds) / 100
        var n: Int = 0
        for a in antsInView(time: t, range: viewRange, mutant: mutant) {
            let r = faceInside(a)
            worst = max(worst, r.depth)
            n += r.points
        }
        total += n
        frames += 1
    }
    print(String(format: "        %d frames, %d face points (%.0f a frame): deepest inside an ant %.6f mm", frames, total,
                 Double(total) / Double(frames), worst))
    expect(worst < 1e-3, "the wood is \(worst) mm inside an ant")
}

test("the timetable and the drawing agree on which feet are down (except a swinging foot still within 16 µm)") {
    var plantedButUp: Int = 0
    var swingingButDown: Int = 0
    var excused: Int = 0
    for (k, ants) in fine.enumerated() {
        for a in ants {
            for j in 0..<6 {
                let d: Bool = fineDown[k][a.id][j]
                if a.stance[j] && !d { plantedButUp += 1 }
                if !a.stance[j] && d {
                    // Only allowed at the very start or end of a swing, when
                    // the foot is still or already within 16 µm of the wood.
                    let lift: Float = a.feet[j].y - woodHeight(a.feet[j].x, a.feet[j].z)
                    if lift < downGap + woodHeightMax * 2 { excused += 1 } else { swingingButDown += 1 }
                }
            }
        }
    }
    print("        planted but drawn up: \(plantedButUp); swinging but drawn down: \(swingingButDown) (+\(excused) at a swing's first or last few ms)")
    expectEqual(plantedButUp, 0)
    expectEqual(swingingButDown, 0)
}

section("the trail")

test("single file: every ant stays within 0.15 mm of the trail line and heads along it within 8°") {
    var worstZ: Float = 0
    var worstYaw: Float = 0
    for ants in fine {
        for a in ants {
            worstZ = max(worstZ, abs(a.at.z))
            worstYaw = max(worstYaw, abs(deg(a.yaw)))
        }
    }
    print(String(format: "        widest %.3f mm off the trail, heading up to %.1f°", worstZ, worstYaw))
    expect(worstZ <= 0.15, "an ant strays \(worstZ) mm")
    expect(worstYaw <= 8, "an ant heads \(worstYaw)° off the trail")
}

test("the antennae sweep the trail: each tip crosses within 0.1 mm of the line, always just above the wood") {
    for id in 0..<3 {
        for i in 0..<2 {
            var nearest: Float = 10
            var lowTip: Float = 10
            for ants in fine {
                let a: WorldAnt = ants[id]
                let ant: Antenna = a.model.antennae[i]
                expect(ant.reached, "ant \(id) antenna \(i) cannot reach its tip")
                let tip: Shape = ant.segments[ant.segments.count - 1].placed(scale: antPlans[id].scale, yaw: a.yaw, at: a.at)
                nearest = min(nearest, abs(tip.b.z))
                lowTip = min(lowTip, sphereGap(tip.b, tip.rb))
            }
            expect(nearest < 0.1, "ant \(id) antenna \(i) never comes nearer the trail than \(nearest) mm")
            expect(lowTip > 0 && lowTip < 0.06 + 2 * woodHeightMax, "ant \(id) antenna \(i) tip lowest at \(lowTip) mm above the wood")
        }
    }
}

/// The gaster tip's gap to the wood: the exact cone distance from points on
/// the face, over a wider patch than the dab was solved on.
func gasterGap(_ a: WorldAnt) -> Float {
    let t: Shape = a.shapes.filter { $0.part == .gaster && $0.kind == .roundCone }[0]
    return coneGap(t, near: SIMD2<Float>(t.b.x, t.b.z), reach: t.rb * 2.5)
}

test("each ant dabs once a loop: the gaster TIP meets the wood (0 ± 1 µm), belly clear, only while the ant stands still") {
    for id in 0..<3 {
        var touches: Int = 0
        var wasDown = false
        var lowest: Float = 10
        for (k, ants) in fine.enumerated() {
            let a: WorldAnt = ants[id]
            let tipLow: Float = a.shapes.filter { $0.part == .gaster && $0.kind == .roundCone }[0].extent(along: SIMD3(0, 1, 0)).lo
            let g: Float = tipLow > woodHeightMax + 0.002 ? tipLow : gasterGap(a)
            let belly: Float = a.shapes.first { $0.part == .gaster && $0.kind == .ellipsoid }!.extent(along: SIMD3(0, 1, 0)).lo
            expect(belly - woodHeightMax > 0.015, "ant \(id) touches with its belly (\(belly) mm)")
            lowest = min(lowest, g)
            let isDown: Bool = g < 1e-3
            if isDown {
                expect(antPlans[id].speed(fineTimes[k]) < 1e-5, "ant \(id) dabs while moving")
                if !wasDown { touches += 1 }
            }
            wasDown = isDown
        }
        expectEqual(touches, 1)
        expect(abs(lowest) < 1e-3, "ant \(id) gaster tip's least gap to the wood \(lowest) mm")
    }
}

test("the ants never touch one another, and keep a following gap") {
    var minGap: Float = 100
    var overlaps: Int = 0
    var rng = SystemRandomNumberGenerator()
    for t in stride(from: Float(0), to: loopSeconds, by: 0.03) {
        let ants: [WorldAnt] = antsInView(time: t, range: -12...12, mutant: mutant).sorted { $0.at.x < $1.at.x }
        for k in 1..<ants.count {
            let back: WorldAnt = ants[k - 1]
            let front: WorldAnt = ants[k]
            let tail: Float = front.model.body.map { $0.placed(scale: antPlans[front.id].scale, yaw: front.yaw, at: front.at).extent(along: SIMD3(1, 0, 0)).lo }.min() ?? 0
            let nose: Float = back.model.body.map { $0.placed(scale: antPlans[back.id].scale, yaw: back.yaw, at: back.at).extent(along: SIMD3(1, 0, 0)).hi }.max() ?? 0
            minGap = min(minGap, tail - nose)
            // Any shape of one inside any shape of the other?
            for s in back.shapes {
                let (cs, rs) = s.bound
                for u in front.shapes {
                    let (cu, ru) = u.bound
                    if simd_distance(cs, cu) > rs + ru { continue }
                    for _ in 0..<60 {
                        let p: SIMD3<Float> = cs + SIMD3<Float>(Float.random(in: -rs...rs, using: &rng), Float.random(in: -rs...rs, using: &rng),
                                                                Float.random(in: -rs...rs, using: &rng))
                        if s.contains(p) && u.contains(p) { overlaps += 1; break }
                    }
                }
            }
        }
    }
    print(String(format: "        closest nose-to-tail gap %.2f mm (nominal %.1f)", minGap, noseToTailGap))
    expect(overlaps == 0, "\(overlaps) overlapping shape pairs")
    expect(minGap > 1.0, "ants come within \(minGap) mm nose to tail")
}

test("at every frame at least three ants are in view") {
    for f in 0..<125 {
        let t: Float = Float(f) * loopSeconds / 125
        let n: Int = antsInView(time: t, range: viewRange, mutant: mutant).filter { a in
            viewRange.contains(a.at.x - 1) || viewRange.contains(a.at.x + 1)
        }.count
        expect(n >= 3, "frame \(f): only \(n) ants in view")
    }
}

section("the loop")

test("forward only: every ant's distance never decreases, and it gains exactly one period a loop") {
    for p in antPlans {
        var last: Float = p.distance(0, mutant: mutant)
        var worstBack: Float = 0
        for t in fineTimes.dropFirst() {
            let d: Float = p.distance(t, mutant: mutant)
            worstBack = max(worstBack, last - d)
            last = d
        }
        // Float rounding while standing still is ~1e-6 mm; a rewind is millimetres.
        expect(worstBack <= 1e-5, "ant \(p.id) goes back \(worstBack) mm")
        let gained: Float = p.distance(loopSeconds, mutant: mutant) - p.distance(0, mutant: mutant)
        expect(abs(gained - groupPeriod) < 1e-3, "ant \(p.id) gains \(gained) mm a loop, not \(groupPeriod)")
    }
}

test("seamless: at the loop's end each ant is exactly where the same ant ahead of it began") {
    var worst: Float = 0
    for p in antPlans {
        for m in -2...1 {
            let end: WorldAnt = worldAnt(p, copy: m, time: loopSeconds, mutant: mutant)
            let start: WorldAnt = worldAnt(p, copy: m + 1, time: 0, mutant: mutant)
            for (a, b) in zip(end.shapes, start.shapes) {
                worst = max(worst, simd_distance(a.a, b.a), simd_distance(a.b, b.b))
            }
        }
    }
    let m0: [Mark] = marks(time: 0, range: viewRange)
    let m1: [Mark] = marks(time: loopSeconds, range: viewRange)
    expectEqual(m0.count, m1.count)
    for (a, b) in zip(m0, m1) { worst = max(worst, simd_distance(a.point, b.point), abs(a.strength - b.strength)) }
    print(String(format: "        worst mismatch at the seam %.6f mm", worst))
    expect(worst < 1e-3, "the loop's end misses its start by \(worst) mm")
}

test("the stride divides the loop, and the GIF's frames divide it: the legs come round too") {
    let strides: Float = groupPeriod / strideLength
    expect(abs(strides - strides.rounded()) < 1e-4, "\(strides) strides a loop")
    expectEqual(Int((loopSeconds * 100).rounded()) % gifDelayCentiseconds, 0)
}

test("rendered, the frame after the last is the first: frame at t = loop matches frame 0") {
    guard let r = gpu.renderer else { expect(false, "no GPU"); return }
    do {
        _ = try r.render(time: 0, samples: 1)
        let n: Int = r.image.width * r.image.height * 4
        let first: [UInt8] = Array(UnsafeBufferPointer(start: r.image.pixels.contents().assumingMemoryBound(to: UInt8.self), count: n))
        _ = try r.render(time: loopSeconds, samples: 1)
        let p = r.image.pixels.contents().assumingMemoryBound(to: UInt8.self)
        var worst: Int = 0
        var differing: Int = 0
        for i in 0..<n {
            let d: Int = abs(Int(p[i]) - Int(first[i]))
            worst = max(worst, d)
            if d > 1 { differing += 1 }
        }
        print("        largest pixel difference \(worst)/255, \(differing) channels differ by more than 1")
        // The positions match to 10 nm; what is left is float rounding in the
        // march, which can flip a grazing ray at a silhouette: a handful of
        // pixels in a frame, never a visible jump.
        expect(differing < n / 10_000, "the seam shows: \(differing) channels differ, by up to \(worst)")
        // And a frame a third of the way through is NOT the first: something moves.
        _ = try r.render(time: loopSeconds / 3, samples: 1)
        var moved: Int = 0
        for i in stride(from: 0, to: n, by: 4) where abs(Int(p[i]) - Int(first[i])) > 20 { moved += 1 }
        expect(moved > 1000, "only \(moved) pixels changed a third of a loop on")
    } catch { expect(false, "\(error)") }
}

test("the numbers the caption quotes: 90 mm in 2.96 s, slowed about 15×, about 12 strides a second") {
    expect(abs(realSpeed - 30.4) < 0.1, "speed \(realSpeed)")
    expect(abs(slowdown - 15) < 0.6, "slowed \(slowdown)×")
    expect(stepFrequency > 9 && stepFrequency < 13, "step frequency \(stepFrequency) Hz")
    print(String(format: "        %.1f mm/s real, slowed %.2f×, stride %.2f mm, %.1f strides/s, dab stop %.2f s shown",
                 realSpeed, slowdown, strideLength, stepFrequency, dabWindow))
}

section("the distance function")

test("outside every surface, no distance claims more room than the ray allows") {
    guard let r = gpu.renderer else { expect(false, "no GPU"); return }
    var worst: [Int: Float] = [:]
    var rng = SystemRandomNumberGenerator()
    for t in [Float(0), 1.6, 4.1, 5.1, 9.4] {
        let ants: [WorldAnt] = antsInView(time: t, range: -40...40, mutant: mutant)
        var a: [SIMD3<Float>] = []
        var b: [SIMD3<Float>] = []
        for k in 0..<150_000 {
            // A third anywhere near the file, a third packed round one ant's
            // body, a third hugging the wood (within 60 µm of its face),
            // where the roughness's slopes are.
            let c: SIMD3<Float> = ants[k % ants.count].at
            let span: SIMD3<Float> = k % 3 == 0 ? SIMD3(4.0, 1.4, 2.4) : SIMD3(2.6, 1.0, 1.0)
            var p: SIMD3<Float> = c + SIMD3<Float>(Float.random(in: -span.x...span.x, using: &rng),
                                                   Float.random(in: 0...span.y, using: &rng) + 0.001,
                                                   Float.random(in: -span.z...span.z, using: &rng))
            if k % 3 == 2 {
                let x: Float = Float.random(in: -10.5...10.5, using: &rng)
                let z: Float = Float.random(in: -3...3, using: &rng)
                p = SIMD3<Float>(x, woodHeight(x, z) + Float.random(in: 0.0005...0.06, using: &rng), z)
            }
            let d: SIMD3<Float> = simd_normalize(SIMD3<Float>(Float.random(in: -1...1, using: &rng), Float.random(in: -1...1, using: &rng),
                                                              Float.random(in: -1...1, using: &rng)))
            let step: Float = k % 3 == 0 ? 0.01 : (k % 3 == 1 ? 0.004 : 0.002)
            a.append(p)
            b.append(p + d * step)
        }
        guard let pa = try? r.probe(a, time: t), let pb = try? r.probe(b, time: t) else { expect(false, "probe failed"); return }
        for i in 0..<a.count {
            let ma: Int = Int(pa[i].y)
            guard ma == Int(pb[i].y), pa[i].x > 0, pb[i].x > 0 else { continue }
            let v: Float = abs(pa[i].x - pb[i].x) / simd_distance(a[i], b[i])
            worst[ma] = max(worst[ma] ?? 0, v)
        }
    }
    print(String(format: "        worst over-report: wood %.2f, ant %.2f; the ray allows %.2f", worst[1] ?? 0, worst[2] ?? 0, 1 / stepScale))
    expect(worst.count == 2, "not every material was sampled: \(worst)")
    for (m, v) in worst { expect(v * stepScale <= 1.0, "material \(m) oversteps: \(v)") }
}

test("the wood the GPU draws is the wood the CPU rests the feet on: heights agree to 0.01 µm at 200,000 points") {
    guard let r = gpu.renderer else { expect(false, "no GPU"); return }
    var rng = SystemRandomNumberGenerator()
    var pts: [SIMD3<Float>] = []
    for _ in 0..<200_000 {
        pts.append(SIMD3<Float>(Float.random(in: -12...12, using: &rng), 0.3, Float.random(in: -6...6, using: &rng)))
    }
    guard let q = try? r.probe(pts, time: 0) else { expect(false, "probe failed"); return }
    var worst: Float = 0
    for (p, o) in zip(pts, q) { worst = max(worst, abs(o.w - woodHeight(p.x, p.z))) }
    print(String(format: "        worst CPU/GPU wood height difference %.2e mm", worst))
    expect(worst < 1e-5, "the kernel's wood differs from the CPU's by \(worst) mm")
}

test("the ants the GPU draws are the ants the CPU placed: a stance foot's lowest point is on the drawn surface") {
    guard let r = gpu.renderer else { expect(false, "no GPU"); return }
    let t: Float = 2.7
    let ants: [WorldAnt] = antsInView(time: t, range: -40...40, mutant: mutant)
    var pts: [SIMD3<Float>] = []
    for a in ants { for j in 0..<6 where a.stance[j] { pts.append(a.feet[j] + SIMD3<Float>(0, 0.0005, 0)) } }
    guard let q = try? r.probe(pts, time: t) else { expect(false, "probe failed"); return }
    var worst: Float = 0
    for p in q { worst = max(worst, abs(p.z)) }
    print(String(format: "        %d stance feet, drawn ant surface within %.4f mm of each", pts.count, worst))
    expect(!pts.isEmpty && worst < 0.003, "a stance foot is \(worst) mm from the drawn ant")
}

section("the wood: planed Scots pine, radial face (Pinkowski et al. 2016)")

/// Follow the grain: points of one radial coordinate, along z.
func alongGrain(_ r0: Float, _ z: Float) -> Float {
    r0 + z * tan(grainRunOut) + grainWaveAmplitude * sin(2 * Float.pi * z / grainWaveLength)
}

test("the rings: 2.3 mm apart on average across the frame, 31.6% latewood, the transition abrupt") {
    // Walk along the trail line (z = 0) in 1 µm steps and time the zones.
    var x: Float = -10.5
    var last: Float = latewood(x, 0)
    var boundaries: [Float] = []
    var lateLength: Float = 0
    var riseStart: Float? = nil
    var widestRise: Float = 0
    while x < 10.5 {
        x += 0.001
        let lw: Float = latewood(x, 0)
        if lw > 0.5 { lateLength += 0.001 }
        if last >= 0.5 && lw < 0.5 { boundaries.append(x) }            // latewood → next earlywood: the ring boundary
        if last <= 0.02 && lw > 0.02 { riseStart = x }
        if let s = riseStart, lw >= 0.98 { widestRise = max(widestRise, x - s); riseStart = nil }
        last = lw
    }
    let span: Float = (boundaries.last ?? 0) - (boundaries.first ?? 0)
    let mean: Float = span / Float(max(boundaries.count - 1, 1))
    let lateShare: Float = lateLength / 21.0
    print(String(format: "        %d ring boundaries in view, mean spacing %.3f mm (cited 2.3); latewood %.1f%% (cited 31.6%%); earlywood→latewood in %.3f mm",
                 boundaries.count, mean, lateShare * 100, widestRise))
    expect(boundaries.count >= 7, "only \(boundaries.count) rings in view")
    expect(abs(mean - ringWidth) < 0.12, "rings \(mean) mm apart")
    expect(abs(lateShare - latewoodFraction) < 0.03, "latewood share \(lateShare)")
    expect(widestRise < 0.25, "the transition takes \(widestRise) mm: not abrupt")
}

test("the roughness, measured as Pinkowski et al. did — along the grain, 12.5 mm, 5 × 2.5 mm: Ra, Rz, Rt per zone") {
    // Profiles that follow the grain through the middle of each zone.
    var rows: [(zone: String, ra: Float, rz: Float, rt: Float)] = []
    for (zone, target) in [("earlywood", Float(0.35)), ("latewood", Float(0.84))] {
        var ras: [Float] = []
        var rzs: [Float] = []
        var rts: [Float] = []
        for k in 1..<(ringCount - 1) {
            let r0: Float = ringStarts[k] + (ringStarts[k + 1] - ringStarts[k]) * target
            for lane in 0..<3 {
                let dz: Float = Float(lane) * 13
                var p: [Float] = []
                for i in 0..<12_500 {
                    let z: Float = Float(i) * 0.001 - 6.25 + dz - 13
                    p.append(woodHeight(alongGrain(r0, z), z))
                }
                let mean: Float = p.reduce(0, +) / Float(p.count)
                p = p.map { $0 - mean }
                ras.append(p.map { abs($0) }.reduce(0, +) / Float(p.count))
                var rz: Float = 0
                for s in 0..<5 {
                    let seg = p[(s * 2500)..<((s + 1) * 2500)]
                    rz += (seg.max()! - seg.min()!) / 5
                }
                rzs.append(rz)
                rts.append(p.max()! - p.min()!)
            }
        }
        func avg(_ v: [Float]) -> Float { v.reduce(0, +) / Float(v.count) }
        rows.append((zone, avg(ras) * 1000, avg(rzs) * 1000, avg(rts) * 1000))
    }
    let early = rows[0]
    let late = rows[1]
    print(String(format: "        earlywood Ra %.2f µm (cited 7.16), Rz %.1f (%.1f), Rt %.1f (%.1f)", early.ra, early.rz,
                 raEarlywood * 1000 * rzOverRaEarly, early.rt, raEarlywood * 1000 * rtOverRaEarly))
    print(String(format: "        latewood  Ra %.2f µm (cited 3.76), Rz %.1f (%.1f), Rt %.1f (%.1f)", late.ra, late.rz,
                 raLatewood * 1000 * rzOverRaLate, late.rt, raLatewood * 1000 * rtOverRaLate))
    expect(abs(early.ra / 7.16 - 1) < 0.06, "earlywood Ra \(early.ra) µm")
    expect(abs(late.ra / 3.76 - 1) < 0.06, "latewood Ra \(late.ra) µm")
    expect(abs(early.rz / (7.16 * rzOverRaEarly) - 1) < 0.15, "earlywood Rz \(early.rz) µm")
    expect(abs(late.rz / (3.76 * rzOverRaLate) - 1) < 0.15, "latewood Rz \(late.rz) µm")
    expect(abs(early.rt / (7.16 * rtOverRaEarly) - 1) < 0.2, "earlywood Rt \(early.rt) µm")
    expect(abs(late.rt / (3.76 * rtOverRaLate) - 1) < 0.2, "latewood Rt \(late.rt) µm")
    expect(early.ra > late.ra, "earlywood planed smoother than latewood: the wrong way round")
}

test("rays are left out honestly: one ray cell (15 µm) is under a pixel at the GIF's width") {
    let pixel: Float = millimetresPerPixel(width: gifWidth) * 1000
    print(String(format: "        a pixel is %.1f µm; a ray cell %.0f µm", pixel, rayCellHeightMicrometres))
    expect(rayCellHeightMicrometres < pixel)
}

section("the motion shows")

/// Render the frame at t and keep its pixels.
func frame(_ r: TrailRenderer, _ t: Float) throws -> [UInt8] {
    _ = try r.render(time: t, samples: 1)
    let n: Int = r.image.width * r.image.height * 4
    return Array(UnsafeBufferPointer(start: r.image.pixels.contents().assumingMemoryBound(to: UInt8.self), count: n))
}

test("visible motion: between every pair of consecutive GIF frames, hundreds of pixels change, and on the ants") {
    guard let r = gpu.renderer else { expect(false, "no GPU"); return }
    let w: Int = r.image.width
    let h: Int = r.image.height
    let dt: Float = Float(gifDelayCentiseconds) / 100
    var fewest: Int = Int.max
    var leastOnAnts: Float = 1
    do {
        for k in 0..<12 {
            let t: Float = Float(k) * loopSeconds / 12 + 0.37
            let a: [UInt8] = try frame(r, t)
            let b: [UInt8] = try frame(r, t + dt)
            // Where the ants are, in either frame: their projected shapes, padded 6 px.
            var onAnt = [Bool](repeating: false, count: w * h)
            for when in [t, t + dt] {
                // Where the ants truly are (not where a broken picture shows them).
                for ant in antsInView(time: when, range: viewRange, mutant: .none) {
                    for s in ant.shapes {
                        let c: SIMD2<Float> = project(s.bound.centre, width: w, height: h)
                        let rad: Float = s.bound.radius / millimetresPerPixel(width: w) + 6
                        let x0: Int = max(Int(c.x - rad), 0)
                        let x1: Int = min(Int(c.x + rad), w - 1)
                        let y0: Int = max(Int(c.y - rad), 0)
                        let y1: Int = min(Int(c.y + rad), h - 1)
                        if x0 > x1 || y0 > y1 { continue }
                        for y in y0...y1 { for x in x0...x1 { onAnt[y * w + x] = true } }
                    }
                }
            }
            var changed: Int = 0
            var changedOnAnts: Int = 0
            for i in 0..<(w * h) {
                let d: Int = abs(Int(a[4 * i]) - Int(b[4 * i])) + abs(Int(a[4 * i + 1]) - Int(b[4 * i + 1])) + abs(Int(a[4 * i + 2]) - Int(b[4 * i + 2]))
                if d > 36 {
                    changed += 1
                    if onAnt[i] { changedOnAnts += 1 }
                }
            }
            fewest = min(fewest, changed)
            if changed > 0 { leastOnAnts = min(leastOnAnts, Float(changedOnAnts) / Float(changed)) }
        }
    } catch { expect(false, "\(error)"); return }
    let share: Float = Float(fewest) / Float(w * h)
    print(String(format: "        at %d × %d: fewest pixels changing by > 12 levels per channel between consecutive frames %d (%.2f%% of the frame); at least %.0f%% of them on the ants",
                 w, h, fewest, share * 100, leastOnAnts * 100))
    expect(fewest >= 300, "only \(fewest) pixels change between consecutive frames: the motion does not show")
    expect(leastOnAnts > 0.9, "the change is not where the ants are")
}

test("a step reads: each stride carries a foot well over 10 px at the GIF's width, and the body moves every frame") {
    let px: Float = 1 / millimetresPerPixel(width: gifWidth)
    let stridePx: Float = strideLength * px
    let perFrame: Float = groupPeriod / Float(Int((loopSeconds * 100).rounded()) / gifDelayCentiseconds) * px
    let liftPx: Float = footLift * px
    print(String(format: "        stride %.0f px, body %.1f px a frame on average, swing lift %.1f px (the camera is 32° above the wood)",
                 stridePx, perFrame, liftPx))
    expect(stridePx >= 10, "a stride is only \(stridePx) px")
    expect(perFrame >= 2, "the ants move only \(perFrame) px a frame")
}

section("the picture")

test("keep text off the ants: at every GIF frame every part stays between the caption strip and the labels below") {
    let w: Int = gifWidth
    let h: Int = Int((Float(w) * frameAspect).rounded())
    let k: Float = Float(w) / 1200
    let top: Float = 160 * k            // caption lines and the wood's labels end above this
    let bottom: Float = Float(h) - 102 * k   // the trail's labels and the footfall diagram begin below this
    var highest: Float = 1e9
    var lowest: Float = -1e9
    let count: Int = Int((loopSeconds * 100).rounded()) / gifDelayCentiseconds
    for f in 0..<count {
        let t: Float = Float(f * gifDelayCentiseconds) / 100
        for a in antsInView(time: t, range: viewRange, mutant: mutant) {
            for s in a.shapes {
                let c: SIMD2<Float> = project(s.bound.centre, width: w, height: h)
                let r: Float = s.bound.radius / millimetresPerPixel(width: w)
                if c.x + r < 0 || c.x - r > Float(w) { continue }
                highest = min(highest, c.y - r)
                lowest = max(lowest, c.y + r)
            }
        }
    }
    print(String(format: "        ants span rows %.0f–%.0f px; text above %.0f and below %.0f", highest, lowest, top, bottom))
    expect(highest > top, "an ant reaches up into the caption strip (row \(highest))")
    expect(lowest < bottom, "an ant reaches down into the labels (row \(lowest))")
}

test("the scale bar is true everywhere: 1 mm along the trail is the same pixels near and far") {
    let w: Int = 1200
    let h: Int = Int((Float(w) * frameAspect).rounded())
    for p in [SIMD3<Float>(-8, 0, 0), SIMD3<Float>(3, 0.8, 1.5), SIMD3<Float>(7, 0, -1.8)] {
        let dx: Float = simd_distance(project(p, width: w, height: h), project(p + SIMD3<Float>(1, 0, 0), width: w, height: h))
        expect(abs(dx - 1 / millimetresPerPixel(width: w)) < 0.01, "1 mm is \(dx) px at \(p)")
    }
}

finish()
