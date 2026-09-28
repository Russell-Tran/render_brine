// Tests for a file of ants on a pheromone trail (step 44's tests, generalised
// to a file of `fileCount`, plus the visible-motion tests). The gait is
// checked from the drawn geometry — which feet are actually on the ground,
// and whether they move while they are — not from the timetable that placed
// them; the loop from the ants' positions and from rendered frames; the
// motion from rendered GIF-sized frames; the distance function from the
// same kernel source that draws the picture.
//
// TRAIL_MUTANT=nonTripod|sliding|rewind|subpixel breaks the scene on
// purpose; `make mutants` requires the suite to fail for each.

import Foundation
import Metal
import simd

let mutant: Mutant = {
    switch ProcessInfo.processInfo.environment["TRAIL_MUTANT"] {
    case "nonTripod": return .nonTripod
    case "sliding": return .sliding
    case "rewind": return .rewind
    case "subpixel": return .subpixel
    default: return .none
    }
}()

func deg(_ r: Float) -> Float { r * 180 / Float.pi }

let gifHeight: Int = Int((Float(gifWidth) * frameAspect).rounded())

/// Holds the renderers so a failure to find a GPU is a test failure, not a crash.
final class GPUBox {
    let device: MTLDevice? = try? findDevice()
    lazy var small: TrailRenderer? = {
        guard let d = device else { return nil }
        return try? TrailRenderer(width: 480, height: Int((480 * frameAspect).rounded()), mutant: mutant, on: d)
    }()
    /// At the GIF's own size: what the viewer sees.
    lazy var full: TrailRenderer? = {
        guard let d = device else { return nil }
        return try? TrailRenderer(width: gifWidth, height: gifHeight, mutant: mutant, on: d)
    }()
}
let gpu = GPUBox()

/// Fine time samples across one loop, and every ant in (or near) view at each.
let fineStep: Float = 0.005
let fineTimes: [Float] = (0..<Int((loopSeconds / fineStep).rounded())).map { Float($0) * fineStep }
let wideView: ClosedRange<Float> = (viewRange.lowerBound - 6)...(viewRange.upperBound + 6)
let fine: [[WorldAnt]] = fineTimes.map { antsInView(time: $0, range: wideView, mutant: mutant) }

/// One ant's whole crossing of the frame, on its own clock: since every ant
/// walks the same timetable, this is every ant's history.
let crossingTaus: [Float] = {
    var lo: Float = (wideView.lowerBound - 5) / cruise
    while trailX(lo) > wideView.lowerBound - 5 { lo -= 0.5 }
    var hi: Float = (wideView.upperBound + 5) / cruise
    while trailX(hi) < wideView.upperBound + 5 { hi += 0.5 }
    return (0..<Int((hi - lo) / fineStep)).map { lo + Float($0) * fineStep }
}()
let crossing: [WorldAnt] = crossingTaus.map { worldAnt(tau: $0, index: 0, mutant: mutant) }

/// A foot is down when its lowest point is on the ground.
func down(_ a: WorldAnt, _ j: Int) -> Bool { a.feet[j].y < 1e-6 }

/// An ant's pixels on screen at the GIF's size, through a given camera: the
/// box round every shape's bounding sphere.
func screenBox(_ a: WorldAnt, camera c: OrthoCamera) -> (x0: Int, y0: Int, x1: Int, y1: Int) {
    var x0: Float = 1e9, y0: Float = 1e9, x1: Float = -1e9, y1: Float = -1e9
    let pxPerMm: Float = Float(gifWidth) / (2 * c.halfWidth)
    for s in a.shapes {
        let (centre, r) = s.bound
        let p: SIMD2<Float> = project(centre, width: gifWidth, height: gifHeight, camera: c)
        let rp: Float = r * pxPerMm
        x0 = min(x0, p.x - rp); x1 = max(x1, p.x + rp)
        y0 = min(y0, p.y - rp); y1 = max(y1, p.y + rp)
    }
    return (max(Int(x0) - 3, 0), max(Int(y0) - 3, 0), min(Int(x1) + 3, gifWidth - 1), min(Int(y1) + 3, gifHeight - 1))
}

section("each ant, against the literature")

test("each antenna has 12 segments — scape plus an 11-segment funiculus — as a worker's does") {
    for k in stride(from: 0, to: crossing.count, by: 97) {
        let a: WorldAnt = crossing[k]
        expectEqual(a.model.antennae.count, 2)
        for i in 0..<2 {
            let drawn: Int = a.shapes.filter { $0.part == .antenna && $0.index == i }.count
            expectEqual(drawn, workerAntennaSegments)
        }
    }
}

test("six legs, every one joined to the mesosoma — none to the head, petiole or gaster") {
    for k in stride(from: 0, to: crossing.count, by: 97) {
        let a: WorldAnt = crossing[k]
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

test("one petiole, a single node between mesosoma and gaster, as in all Formicinae") {
    let a: WorldAnt = crossing[0]
    let pet: [Shape] = a.model.body.filter { $0.part == .petiole }
    expectEqual(pet.count, 1)
    let mesoBack: Float = a.model.body.filter { $0.part == .mesosoma }.map { $0.extent(along: SIMD3(1, 0, 0)).lo }.min() ?? 0
    let gasterFront: Float = a.model.body.filter { $0.part == .gaster }.map { $0.extent(along: SIMD3(1, 0, 0)).hi }.max() ?? 0
    expect(pet[0].a.x < mesoBack + 0.05 && pet[0].a.x > gasterFront - 0.05, "petiole out of place")
}

test("the head is Seifert's: CS 976 µm, CL/CW 1.074, scape SL/CS 0.979") {
    expect(abs(headWidth - 0.941) < 0.002, "CW \(headWidth)")
    expect(abs(headLength - 1.011) < 0.002, "CL \(headLength)")
    expect(abs(scapeLength / cephalicSize - 0.979) < 1e-4)
    let head: Shape = bodyShapes(gasterBend: 0)[0]
    expect(abs(head.b.z * 2 - headWidth) < 1e-4, "drawn head width \(head.b.z * 2)")
}

test("every worker is 3.4–5.0 mm long, mandibles to gaster tip") {
    let l: Float = baseLength * antScale
    print(String(format: "        %.2f mm", l))
    expect(workerLengthRange.contains(l), "the ants are \(l) mm")
}

test("no leg is ever stretched or squashed: femur and tibia keep their lengths, the whole way across") {
    var worst: Float = 0
    for a in crossing {
        for (j, leg) in a.model.legs.enumerated() {
            let spec: LegSpec = legSpecs[j % 3]
            worst = max(worst, abs(simd_distance(leg.hip, leg.knee) - spec.femur))
            worst = max(worst, abs(simd_distance(leg.knee, leg.ankle) - spec.tibia))
        }
    }
    print(String(format: "        worst femur/tibia length error %.5f mm", worst))
    expect(worst < 1e-3, "a leg segment is off its length by \(worst) mm: the foot is out of reach")
}

section("the gait")

test("alternating tripod: whenever three feet are down they are one tripod, and the tripods take turns") {
    var sequence: [Int] = []       // one entry per single-tripod phase: 0 = A alone, 1 = B alone
    var previous: Int = -1
    var bad: Int = 0
    for a in crossing {
        let set: Set<Int> = Set((0..<6).filter { down(a, $0) })
        var now: Int = -2
        if set.count == 6 { now = -1 }
        else if set == tripodA { now = 0 }
        else if set == tripodB { now = 1 }
        if now == -2 {
            bad += 1
            if bad < 4 { expect(false, "feet \(set.sorted()) down at x = \(a.distance) — not a tripod") }
            previous = -1
            continue
        }
        if now >= 0 && now != previous { sequence.append(now) }
        previous = now
    }
    for k in 1..<max(sequence.count, 1) where sequence[k] == sequence[k - 1] {
        expect(false, "tripod \(sequence[k]) stands alone twice running (phase \(k))")
    }
    let strides: Float = (crossing.last!.distance - crossing[0].distance) / strideLength
    print(String(format: "        %d single-tripod phases in %.1f strides across the frame", sequence.count, strides))
    expect(sequence.contains(0) && sequence.contains(1), "the ant never stands on one of its tripods")
    expect(Float(sequence.count) >= 2 * strides.rounded(.down) - 1, "only \(sequence.count) single-tripod phases")
    expect(bad == 0, "\(bad) samples with a non-tripod set of feet down")
}

test("stance feet do not slide: a foot on the ground stays within 0.1 µm of where it landed") {
    var worst: Float = 0
    var stances: Int = 0
    for j in 0..<6 {
        var landed: SIMD3<Float>? = nil
        for a in crossing {
            if down(a, j) {
                if let l = landed { worst = max(worst, simd_distance(l, a.feet[j])) } else { landed = a.feet[j]; stances += 1 }
            } else {
                landed = nil
            }
        }
    }
    print(String(format: "        %d stances, worst drift %.6f mm", stances, worst))
    expect(worst < 1e-4, "a stance foot moved \(worst) mm across the ground")
}

test("feet in stance touch the ground (lowest point 0), feet in swing lift, and nothing goes below it") {
    var worstTouch: Float = 0
    var lowest: Float = 0
    var highestSwing: Float = 0
    for a in crossing {
        for (j, leg) in a.model.legs.enumerated() {
            let tip: Shape = leg.shapes[leg.shapes.count - 1]
            let tipLow: Float = tip.placed(scale: antScale, yaw: a.yaw, at: a.at).extent(along: SIMD3(0, 1, 0)).lo
            if down(a, j) { worstTouch = max(worstTouch, abs(tipLow)) } else { highestSwing = max(highestSwing, a.feet[j].y) }
        }
        lowest = min(lowest, a.lowest)
    }
    print(String(format: "        stance foot to ground %.6f mm; lowest point of any ant %.6f mm; swing lift up to %.3f mm",
                 worstTouch, lowest, highestSwing))
    expect(worstTouch < 1e-4, "a stance foot is \(worstTouch) mm off the ground")
    expect(lowest > -1e-4, "part of an ant is \(-lowest) mm below the ground")
    expect(highestSwing > 0.1, "swing feet barely lift: \(highestSwing) mm")
}

test("the timetable and the geometry agree on which feet are down") {
    var mismatches: Int = 0
    for a in crossing { for j in 0..<6 where a.stance[j] != down(a, j) { mismatches += 1 } }
    expectEqual(mismatches, 0)
}

section("the file on the trail")

test("single file: every ant stays within 0.15 mm of the trail line and heads along it within 8°") {
    var worstZ: Float = 0
    var worstYaw: Float = 0
    for a in crossing {
        worstZ = max(worstZ, abs(a.at.z))
        worstYaw = max(worstYaw, abs(deg(a.yaw)))
    }
    print(String(format: "        widest %.3f mm off the trail, heading up to %.1f°", worstZ, worstYaw))
    expect(worstZ <= 0.15, "an ant strays \(worstZ) mm")
    expect(worstYaw <= 8, "an ant heads \(worstYaw)° off the trail")
}

test("the antennae sweep the trail: each tip crosses within 0.1 mm of the line, always just above the ground") {
    for i in 0..<2 {
        var nearest: Float = 10
        var lowTip: Float = 10
        for a in crossing {
            let ant: Antenna = a.model.antennae[i]
            expect(ant.reached, "antenna \(i) cannot reach its tip")
            let tip: Shape = ant.segments[ant.segments.count - 1].placed(scale: antScale, yaw: a.yaw, at: a.at)
            nearest = min(nearest, abs(tip.b.z - a.at.z))
            lowTip = min(lowTip, tip.extent(along: SIMD3(0, 1, 0)).lo)
        }
        expect(nearest < 0.1, "antenna \(i) never comes nearer the trail than \(nearest) mm")
        expect(lowTip > 0 && lowTip < 0.06, "antenna \(i) tip lowest at \(lowTip) mm")
    }
}

test("neighbours' antennae sweep out of step: each is a fixed fraction of a sweep behind the ant ahead") {
    // Ant k+1's clock is one loop ahead of ant k's, so its sweep leads by
    // sweepRate × loop sweeps; the fractional part is what shows.
    let lead: Float = (sweepRate * loopSeconds).truncatingRemainder(dividingBy: 1)
    print(String(format: "        neighbours %.2f of a sweep apart", lead))
    expect(lead > 0.15 && lead < 0.85, "neighbours sweep nearly in step (\(lead) of a sweep apart)")
    for ants in fine where ants.count > 1 {
        let s: [WorldAnt] = ants.sorted { $0.index < $1.index }
        for k in 1..<s.count { expect(abs(s[k].tau - s[k - 1].tau - loopSeconds) < 1e-3, "clocks not one loop apart") }
    }
}

test("each ant dabs at every site it passes: the gaster TIP meets the ground (0 ± 1 µm), belly clear, only while it stands still") {
    var touches: [Float] = []
    var wasDown = false
    var lowest: Float = 10
    for a in crossing {
        let g: Float = a.shapes.filter { $0.part == .gaster }.map { $0.extent(along: SIMD3(0, 1, 0)).lo }.min() ?? 1
        let belly: Float = a.shapes.first { $0.part == .gaster && $0.kind == .ellipsoid }!.extent(along: SIMD3(0, 1, 0)).lo
        expect(belly > 0.03, "the ant touches with its belly (\(belly) mm)")
        lowest = min(lowest, g)
        let isDown: Bool = g < 1e-3
        if isDown {
            expect(timetableSpeed(a.tau) < 1e-5, "the ant dabs while moving (x = \(a.distance))")
            if !wasDown { touches.append(a.distance) }
        }
        wasDown = isDown
    }
    print("        dabs at x = \(touches.map { String(format: "%.2f", $0) }) mm; sites \(dabSites.map { String(format: "%.2f", $0) })")
    expectEqual(touches.count, dabSites.count)
    for (x, site) in zip(touches, dabSites) { expect(abs(x - site) < 0.02, "dab at \(x), site at \(site)") }
    for site in dabSites { expect(viewRange.contains(site), "a dab site at \(site) mm is out of view") }
    expect(abs(lowest) < 1e-3, "gaster lowest \(lowest) mm")
}

test("the ants never touch one another, and keep a following gap") {
    var minGap: Float = 100
    var overlaps: Int = 0
    var rng = SystemRandomNumberGenerator()
    for k in stride(from: 0, to: fine.count, by: 6) {
        let ants: [WorldAnt] = fine[k].sorted { $0.at.x < $1.at.x }
        for k in 1..<ants.count {
            let back: WorldAnt = ants[k - 1]
            let front: WorldAnt = ants[k]
            let tail: Float = front.model.body.map { $0.placed(scale: antScale, yaw: front.yaw, at: front.at).extent(along: SIMD3(1, 0, 0)).lo }.min() ?? 0
            let nose: Float = back.model.body.map { $0.placed(scale: antScale, yaw: back.yaw, at: back.at).extent(along: SIMD3(1, 0, 0)).hi }.max() ?? 0
            minGap = min(minGap, tail - nose)
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
    print(String(format: "        closest nose-to-tail gap %.2f mm (cruising: %.2f)", minGap, spacing - baseLength))
    expect(overlaps == 0, "\(overlaps) overlapping shape pairs")
    expect(minGap > 1.0, "ants come within \(minGap) mm nose to tail")
}

test("they follow one headway apart: each ant passes any point exactly 0.4 s (real) after the one ahead") {
    for xp in [viewRange.lowerBound + 1, -3.3, 7.1, viewRange.upperBound - 1] {
        // When does the timetable pass xp? Bisection on the monotone X(τ).
        var lo: Float = (xp - 20) / cruise
        var hi: Float = (xp + 20) / cruise + 4 * dabLoss
        for _ in 0..<60 { let m: Float = (lo + hi) / 2; if trailX(m) < xp { lo = m } else { hi = m } }
        // Ant k passes when its clock reads that τ, so at picture time τ − k·loop.
        let passes: [Float] = (0..<3).map { k in lo - Float(k) * loopSeconds }
        for k in 1..<passes.count {
            let gapReal: Float = (passes[k - 1] - passes[k]) / slowdown
            expect(abs(gapReal - headwayReal) < 1e-4, "headway \(gapReal) s at x = \(xp)")
        }
    }
    let cruisingGap: Float = trailX(-40 / cruise + loopSeconds) - trailX(-40 / cruise)
    print(String(format: "        nose-to-nose while cruising %.2f mm = %.1f mm/s × %.2f s", cruisingGap, realSpeed, headwayReal))
    expect(abs(cruisingGap - spacing) < 1e-3, "cruising spacing \(cruisingGap)")
}

test("\(fileCount) ants in view at every frame: exactly \(fileCount) bodies, and never fewer whole-or-part") {
    var bodies: [Int: Int] = [:]
    var parts: [Int: Int] = [:]
    for ants in fine {
        // An ant within 0.01 mm of an edge is mid-crossing: it counts as half,
        // since at that instant its neighbour at the far edge is crossing too.
        var halves: Int = 0
        for a in ants where viewRange.contains(a.at.x) {
            let edge: Float = min(a.at.x - viewRange.lowerBound, viewRange.upperBound - a.at.x)
            halves += edge < 0.01 ? 1 : 2
        }
        let b: Int = (halves + 1) / 2
        let n: Int = ants.filter { a in
            let lo: Float = a.shapes.map { $0.extent(along: SIMD3(1, 0, 0)).lo }.min() ?? 0
            let hi: Float = a.shapes.map { $0.extent(along: SIMD3(1, 0, 0)).hi }.max() ?? 0
            return hi > viewRange.lowerBound && lo < viewRange.upperBound
        }.count
        bodies[b, default: 0] += 1
        parts[n, default: 0] += 1
    }
    print("        bodies in view: \(bodies.sorted { $0.key < $1.key }.map { "\($0.key) in \($0.value) samples" }); whole or in part: \(parts.sorted { $0.key < $1.key }.map { "\($0.key) in \($0.value)" })")
    expect(bodies.count == 1 && bodies[fileCount] != nil, "bodies in view: \(bodies)")
    expect((parts.keys.min() ?? 0) >= fileCount, "as few as \(parts.keys.min() ?? 0) ants in view")
}

section("the loop")

test("forward only: every ant's position never decreases, and after one loop it stands where the ant ahead stood") {
    for k in -3...3 {
        var last: Float = worldAnt(k, time: 0, mutant: mutant).distance
        var worstBack: Float = 0
        for t in fineTimes.dropFirst() {
            let d: Float = trailX(antTau(k, time: t, mutant: mutant))
            worstBack = max(worstBack, last - d)
            last = d
        }
        expect(worstBack <= 1e-5, "ant \(k) goes back \(worstBack) mm")
        let end: Float = trailX(antTau(k, time: loopSeconds, mutant: mutant))
        let ahead: Float = trailX(antTau(k + 1, time: 0, mutant: mutant))
        expect(abs(end - ahead) < 1e-3, "ant \(k) ends at \(end), the ant ahead began at \(ahead)")
    }
}

test("seamless: at the loop's end every ant, and every mark, is exactly where the one ahead began") {
    var worst: Float = 0
    for k in -4...3 {
        let end: WorldAnt = worldAnt(k, time: loopSeconds, mutant: mutant)
        let start: WorldAnt = worldAnt(k + 1, time: 0, mutant: mutant)
        for (a, b) in zip(end.shapes, start.shapes) {
            worst = max(worst, simd_distance(a.a, b.a), simd_distance(a.b, b.b))
        }
    }
    let m0: [Mark] = marks(time: 0, range: viewRange)
    let m1: [Mark] = marks(time: loopSeconds, range: viewRange)
    expectEqual(m0.count, m1.count)
    for (a, b) in zip(m0, m1) { worst = max(worst, simd_distance(a.point, b.point), abs(a.strength - b.strength)) }
    print(String(format: "        worst mismatch at the seam %.6f mm", worst))
    expect(worst < 1e-3, "the loop's end misses its start by \(worst) mm")
}

test("nothing pops: 1 ms apart, no part of any ant jumps and no mark flickers — across the seam and every dab") {
    var worstMove: Float = 0
    var worstGlow: Float = 0
    var moments: [Float] = [0, loopSeconds - 0.0005, loopSeconds / 2]
    for c in dabTaus {
        var u: Float = c.truncatingRemainder(dividingBy: loopSeconds)
        if u < 0 { u += loopSeconds }
        moments += [u - dabHold / 2, u, u + dabHold / 2, u - dabWindow / 2]
    }
    for t in moments {
        let a: [WorldAnt] = antsInView(time: t, range: viewRange, mutant: mutant).sorted { $0.index < $1.index }
        let b: [WorldAnt] = antsInView(time: t + 0.001, range: viewRange, mutant: mutant).sorted { $0.index < $1.index }
        for p in a {
            guard let q = b.first(where: { $0.index == p.index }) else { continue }
            for (s, u) in zip(p.shapes, q.shapes) { worstMove = max(worstMove, simd_distance(s.a, u.a), simd_distance(s.b, u.b)) }
        }
        let ma: [Mark] = marks(time: t, range: viewRange)
        let mb: [Mark] = marks(time: t + 0.001, range: viewRange)
        for (x, y) in zip(ma, mb) { worstGlow = max(worstGlow, abs(x.strength - y.strength)) }
    }
    print(String(format: "        in 1 ms: largest move %.4f mm, largest change in a mark's glow %.4f", worstMove, worstGlow))
    // A swinging foot at full speed covers ~0.012 mm in 1 ms of picture.
    expect(worstMove < 0.03, "a part jumps \(worstMove) mm in 1 ms")
    expect(worstGlow < 0.02, "a mark's glow jumps \(worstGlow)")
}

test("the GIF's frames divide the loop") {
    expectEqual(Int((loopSeconds * 100).rounded()) % gifDelayCentiseconds, 0)
}

test("rendered, the frame after the last is the first: frame at t = loop matches frame 0") {
    guard let r = gpu.small else { expect(false, "no GPU"); return }
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
        expect(differing < n / 10_000, "the seam shows: \(differing) channels differ, by up to \(worst)")
    } catch { expect(false, "\(error)") }
}

test("the numbers the caption quotes: 90 mm in 2.96 s, slowed 15×, 12 strides a second, 0.4 s and 12 mm apart") {
    expect(abs(realSpeed - 30.4) < 0.1, "speed \(realSpeed)")
    expect(abs(slowdown - 15) < 1e-6, "slowed \(slowdown)×")
    expect(stepFrequency > 9 && stepFrequency < 13, "step frequency \(stepFrequency) Hz")
    expect(abs(spacing - 12.16) < 0.01, "spacing \(spacing)")
    expect(abs(loopSeconds - 6) < 1e-5, "loop \(loopSeconds) s")
    print(String(format: "        %.1f mm/s real, slowed %.0f×, stride %.2f mm, %.1f strides/s, %.2f s = %.2f mm apart, dab stop %.2f s shown",
                 realSpeed, slowdown, strideLength, stepFrequency, headwayReal, spacing, dabWindow))
}

section("the motion can be seen (step 29's lesson)")

/// Pixels whose colour changes visibly (any channel by ≥ 24/255) between two
/// renders, counted inside each box.
func changedPixels(_ r: TrailRenderer, from t0: Float, to t1: Float, boxes: [(x0: Int, y0: Int, x1: Int, y1: Int)]) throws -> [Int] {
    _ = try r.render(time: t0, samples: 1)
    let n: Int = r.image.width * r.image.height * 4
    let first: [UInt8] = Array(UnsafeBufferPointer(start: r.image.pixels.contents().assumingMemoryBound(to: UInt8.self), count: n))
    _ = try r.render(time: t1, samples: 1)
    let p = r.image.pixels.contents().assumingMemoryBound(to: UInt8.self)
    return boxes.map { b in
        var c: Int = 0
        if b.x1 <= b.x0 || b.y1 <= b.y0 { return 0 }
        for y in b.y0...b.y1 {
            for x in b.x0...b.x1 {
                let i: Int = (y * r.image.width + x) * 4
                let d: Int = max(abs(Int(p[i]) - Int(first[i])), abs(Int(p[i + 1]) - Int(first[i + 1])), abs(Int(p[i + 2]) - Int(first[i + 2])))
                if d >= 24 { c += 1 }
            }
        }
        return c
    }
}

test("from one GIF frame to the next, every ant in view visibly changes — hundreds of pixels while it walks") {
    guard let r = gpu.full else { expect(false, "no GPU"); return }
    let dt: Float = Float(gifDelayCentiseconds) / 100
    var walkingLeast: Int = Int.max
    var stoppedLeast: Int = Int.max
    do {
        for f in stride(from: 0, to: Int((loopSeconds / dt).rounded()), by: 13) {
            let t: Float = Float(f) * dt
            let ants: [WorldAnt] = antsInView(time: t, range: viewRange, mutant: mutant).filter {
                $0.at.x > viewRange.lowerBound + 2.5 && $0.at.x < viewRange.upperBound - 2.5
            }
            let boxes = ants.map { screenBox($0, camera: r.camera) }
            let changed: [Int] = try changedPixels(r, from: t, to: t + dt, boxes: boxes)
            for (a, c) in zip(ants, changed) {
                if timetableSpeed(a.tau) > 0.5 * cruise { walkingLeast = min(walkingLeast, c) } else { stoppedLeast = min(stoppedLeast, c) }
            }
        }
    } catch { expect(false, "\(error)"); return }
    print("        per frame at \(gifWidth) × \(gifHeight): a walking ant changes at least \(walkingLeast) px; one stopping or stopped, \(stoppedLeast == Int.max ? 0 : stoppedLeast) px")
    expect(walkingLeast >= 300, "a walking ant changes only \(walkingLeast) pixels from one frame to the next")
    expect(stoppedLeast == Int.max || stoppedLeast >= 20, "a stopping ant changes only \(stoppedLeast) pixels")
}

test("the moves the caption names are at least 10 px on screen: a step, a body's advance, the dab") {
    guard let r = gpu.full else { expect(false, "no GPU"); return }
    let c: OrthoCamera = r.camera
    func px(_ p: SIMD3<Float>) -> SIMD2<Float> { project(p, width: gifWidth, height: gifHeight, camera: c) }
    // A step: how far each foot travels on screen in one swing.
    var leastStep: Float = 1e9
    for j in 0..<6 {
        var liftedAt: SIMD3<Float>? = nil
        for (k, a) in crossing.enumerated() where k > 0 {
            if !down(a, j) && down(crossing[k - 1], j) { liftedAt = crossing[k - 1].feet[j] }
            if down(a, j) && !down(crossing[k - 1], j), let l = liftedAt {
                leastStep = min(leastStep, simd_distance(px(l), px(a.feet[j])))
            }
        }
    }
    // The body's advance in one second of picture at cruise.
    let advance: Float = simd_distance(px(SIMD3<Float>(0, 0, 0)), px(SIMD3<Float>(cruise, 0, 0)))
    // The dab: how far the gaster tip travels on screen, walking to pressed.
    let walking: WorldAnt = worldAnt(tau: dabTaus[0] - dabWindow, index: 0, mutant: mutant)
    let pressed: WorldAnt = worldAnt(tau: dabTaus[0], index: 0, mutant: mutant)
    func tip(_ a: WorldAnt) -> SIMD3<Float> {
        let s: Shape = a.shapes.filter { $0.part == .gaster && $0.kind == .roundCone }[0]
        return s.b - a.at
    }
    let dab: Float = simd_distance(px(tip(walking)), px(tip(pressed)))
    print(String(format: "        least step %.1f px; body advances %.1f px per picture second; gaster tip moves %.1f px in a dab (%.1f px/mm)",
                 leastStep, advance, dab, Float(gifWidth) / (2 * c.halfWidth)))
    expect(leastStep >= 10, "a step is only \(leastStep) px on screen")
    expect(advance >= 10, "the body advances only \(advance) px a second")
    expect(dab >= 10, "the dab moves the gaster tip only \(dab) px on screen")
}

section("the distance function")

test("outside every surface, no distance claims more room than the ray allows") {
    guard let r = gpu.small else { expect(false, "no GPU"); return }
    var worst: [Int: Float] = [:]
    var rng = SystemRandomNumberGenerator()
    for t in [Float(0), 1.6, 4.1, 5.1] {
        let ants: [WorldAnt] = antsInView(time: t, range: -40...40, mutant: mutant)
        var a: [SIMD3<Float>] = []
        var b: [SIMD3<Float>] = []
        for k in 0..<100_000 {
            let c: SIMD3<Float> = ants[k % ants.count].at
            let span: SIMD3<Float> = k % 2 == 0 ? SIMD3(4.0, 1.4, 2.4) : SIMD3(2.6, 1.0, 1.0)
            let p: SIMD3<Float> = c + SIMD3<Float>(Float.random(in: -span.x...span.x, using: &rng),
                                                   Float.random(in: 0...span.y, using: &rng) + 0.001,
                                                   Float.random(in: -span.z...span.z, using: &rng))
            let d: SIMD3<Float> = simd_normalize(SIMD3<Float>(Float.random(in: -1...1, using: &rng), Float.random(in: -1...1, using: &rng),
                                                              Float.random(in: -1...1, using: &rng)))
            let step: Float = k % 2 == 0 ? 0.01 : 0.004
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
    print(String(format: "        worst over-report: table %.2f, ant %.2f; the ray allows %.2f", worst[1] ?? 0, worst[2] ?? 0, 1 / stepScale))
    expect(worst.count == 2, "not every material was sampled: \(worst)")
    for (m, v) in worst { expect(v * stepScale <= 1.0, "material \(m) oversteps: \(v)") }
}

test("the ants the GPU draws are the ants the CPU placed: a stance foot's lowest point is on the drawn surface") {
    guard let r = gpu.small else { expect(false, "no GPU"); return }
    let t: Float = 2.7
    let ants: [WorldAnt] = antsInView(time: t, range: -40...40, mutant: mutant)
    var pts: [SIMD3<Float>] = []
    for a in ants { for j in 0..<6 where down(a, j) { pts.append(a.feet[j] + SIMD3<Float>(0, 0.0005, 0)) } }
    guard let q = try? r.probe(pts, time: t) else { expect(false, "probe failed"); return }
    var worst: Float = 0
    for p in q { worst = max(worst, abs(p.z)) }
    print(String(format: "        %d stance feet, drawn ant surface within %.4f mm of each", pts.count, worst))
    expect(!pts.isEmpty && worst < 0.003, "a stance foot is \(worst) mm from the drawn ant")
}

section("the picture")

test("the scale bar is true everywhere: 1 mm along the trail is the same pixels near and far") {
    let w: Int = 1200
    let h: Int = Int((Float(w) * frameAspect).rounded())
    for p in [SIMD3<Float>(-8, 0, 0), SIMD3<Float>(3, 0.8, 1.5), SIMD3<Float>(7, 0, -1.8)] {
        let dx: Float = simd_distance(project(p, width: w, height: h), project(p + SIMD3<Float>(1, 0, 0), width: w, height: h))
        expect(abs(dx - 1 / millimetresPerPixel(width: w)) < 0.01, "1 mm is \(dx) px at \(p)")
    }
}

test("no text on the ants: the caption above and the legends below stay clear of every ant at every frame") {
    // Where the text blocks end and begin, from Annotate's own layout.
    let (captionBottom, legendTop) = textBands(width: gifWidth, height: gifHeight)
    var top: Float = 1e9
    var bottom: Float = -1e9
    for ants in fine {
        for a in ants {
            for s in a.shapes {
                let (c, r) = s.bound
                let p: SIMD2<Float> = project(c, width: gifWidth, height: gifHeight)
                let rp: Float = r / millimetresPerPixel(width: gifWidth)
                top = min(top, p.y - rp)
                bottom = max(bottom, p.y + rp)
            }
        }
    }
    print(String(format: "        ants span rows %.0f–%.0f; caption ends at %.0f, legends begin at %.0f", top, bottom, captionBottom, legendTop))
    expect(top > captionBottom + 4, "an ant reaches the caption")
    expect(bottom < legendTop - 4, "an ant reaches the legends")
}

finish()
