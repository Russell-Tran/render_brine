// Tests for step 44. The gait is checked from the drawn geometry — which feet
// are actually on the ground, and whether they move while they are — not
// from the timetable that placed them; the loop from the ants' positions and
// from rendered frames; the distance function from the same kernel source
// that draws the picture.
//
// TRAIL_MUTANT=nonTripod|sliding|rewind breaks the scene on purpose; `make
// mutants` requires the suite to fail for each.

import Foundation
import Metal
import simd

let mutant: Mutant = {
    switch ProcessInfo.processInfo.environment["TRAIL_MUTANT"] {
    case "nonTripod": return .nonTripod
    case "sliding": return .sliding
    case "rewind": return .rewind
    default: return .none
    }
}()

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

/// A foot is down when its lowest point is on the ground.
func down(_ a: WorldAnt, _ j: Int) -> Bool { a.feet[j].y < 1e-6 }

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
        for ants in fine {
            let a: WorldAnt = ants[id]
            let set: Set<Int> = Set((0..<6).filter { down(a, $0) })
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

test("stance feet do not slide: a foot on the ground stays within 0.1 µm of where it landed") {
    var worst: Float = 0
    var stances: Int = 0
    for id in 0..<3 {
        for j in 0..<6 {
            var landed: SIMD3<Float>? = nil
            for ants in fine {
                let a: WorldAnt = ants[id]
                if down(a, j) {
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

test("feet in stance touch the ground (lowest point 0), feet in swing lift, and nothing goes below it") {
    var worstTouch: Float = 0
    var lowest: Float = 0
    var highestSwing: Float = 0
    for ants in fine {
        for a in ants {
            for (j, leg) in a.model.legs.enumerated() {
                let tip: Shape = leg.shapes[leg.shapes.count - 1]
                let tipLow: Float = tip.placed(scale: antPlans[a.id].scale, yaw: a.yaw, at: a.at).extent(along: SIMD3(0, 1, 0)).lo
                if down(a, j) { worstTouch = max(worstTouch, abs(tipLow)) } else { highestSwing = max(highestSwing, a.feet[j].y) }
            }
            lowest = min(lowest, a.lowest)
        }
    }
    print(String(format: "        stance foot to ground %.6f mm; lowest point of any ant %.6f mm; swing lift up to %.3f mm",
                 worstTouch, lowest, highestSwing))
    expect(worstTouch < 1e-4, "a stance foot is \(worstTouch) mm off the ground")
    expect(lowest > -1e-4, "part of an ant is \(-lowest) mm below the ground")
    expect(highestSwing > 0.1, "swing feet barely lift: \(highestSwing) mm")
}

test("the timetable and the geometry agree on which feet are down") {
    var mismatches: Int = 0
    for ants in fine { for a in ants { for j in 0..<6 where a.stance[j] != down(a, j) { mismatches += 1 } } }
    expectEqual(mismatches, 0)
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

test("the antennae sweep the trail: each tip crosses within 0.1 mm of the line, always just above the ground") {
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
                lowTip = min(lowTip, tip.extent(along: SIMD3(0, 1, 0)).lo)
            }
            expect(nearest < 0.1, "ant \(id) antenna \(i) never comes nearer the trail than \(nearest) mm")
            expect(lowTip > 0 && lowTip < 0.06, "ant \(id) antenna \(i) tip lowest at \(lowTip) mm")
        }
    }
}

test("each ant dabs once a loop: the gaster TIP meets the ground (0 ± 1 µm), belly clear, only while the ant stands still") {
    for id in 0..<3 {
        var touches: Int = 0
        var wasDown = false
        var lowest: Float = 10
        for (k, ants) in fine.enumerated() {
            let a: WorldAnt = ants[id]
            let g: Float = a.shapes.filter { $0.part == .gaster }.map { $0.extent(along: SIMD3(0, 1, 0)).lo }.min() ?? 1
            let belly: Float = a.shapes.first { $0.part == .gaster && $0.kind == .ellipsoid }!.extent(along: SIMD3(0, 1, 0)).lo
            expect(belly > 0.03, "ant \(id) touches with its belly (\(belly) mm)")
            lowest = min(lowest, g)
            let isDown: Bool = g < 1e-3
            if isDown {
                expect(antPlans[id].speed(fineTimes[k]) < 1e-5, "ant \(id) dabs while moving")
                if !wasDown { touches += 1 }
            }
            wasDown = isDown
        }
        expectEqual(touches, 1)
        expect(abs(lowest) < 1e-3, "ant \(id) gaster lowest \(lowest) mm")
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
        for k in 0..<100_000 {
            // Half anywhere near the file, half packed round one ant's body.
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
    guard let r = gpu.renderer else { expect(false, "no GPU"); return }
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

finish()
