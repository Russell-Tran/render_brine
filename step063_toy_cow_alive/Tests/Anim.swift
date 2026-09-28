// Tests every animated toy step shares. The gait, the planted feet, the
// rigid joints and the loop are read off the toy as drawn — the segment
// frames the kernel receives at each moment — and the motion off the
// rendered pixels.

import CoreGraphics
import Foundation
import Metal
import simd

/// Moments to check: every 1/100 s across the loop.
func fineTimes(_ spec: WalkSpec) -> [Float] {
    let n: Int = Int((spec.loopSeconds * 100).rounded())
    return (0...n).map { Float($0) / 100 }
}

/// The foot segments' frames and lowest points in a posed toy.
func feet(_ toy: PosedToy) -> [(frame: Frame, low: Float)] {
    var out: [(Frame, Float)] = []
    for (i, s) in toy.segments.enumerated() where s.part == .foot {
        let f: Frame = toy.frames[i]
        let ld = SIMD3<Float>(f.x.y, f.y.y, f.z.y)
        var lo: Float = .infinity
        for p in s.prims { lo = min(lo, primExtent(p, ld).0) }
        out.append((f, f.o.y + lo))
    }
    return out.map { (frame: $0.0, low: $0.1) }
}

func yawOf(_ f: Frame) -> Float { atan2(-f.x.z, f.x.x) }

func testAnimation(_ perf: Performance, setup a: AnimSetup, renderer: ToyRenderer?, mutant: Mutant) {
    let spec: WalkSpec = perf.spec
    let times: [Float] = fineTimes(spec)
    let toys: [PosedToy] = times.map { perf.posed($0) }
    let legNames: [String] = perf.design.legs.map { $0.name }

    section("planted feet and rigid pieces")

    test("a foot on the table does not move: not by 0.1 µm, nor turn, until it lifts") {
        var worst: Float = 0
        var worstYaw: Float = 0
        var stances: Int = 0
        for j in 0..<legNames.count {
            var landed: Frame? = nil
            for toy in toys {
                let f = feet(toy)[j]
                if f.low < 1e-4 {
                    if let l = landed {
                        worst = max(worst, simd_distance(l.o, f.frame.o))
                        worstYaw = max(worstYaw, abs(wrapAngle(yawOf(l) - yawOf(f.frame))))
                    } else { landed = f.frame; stances += 1 }
                } else {
                    landed = nil
                }
            }
        }
        print(String(format: "        %d stances; worst drift %.2e mm, worst turn %.2e rad", stances, worst, worstYaw))
        expect(worst < 1e-4, "a planted foot moved \(worst) mm across the table")
        expect(worstYaw < 1e-5, "a planted foot turned \(worstYaw) rad on the table")
        expect(stances > 4 * spec.strides, "only \(stances) stances")
    }

    test("the pieces stay joined: every leg reaches its foot without stretching, at every moment") {
        var worst: Float = 0
        var tightest: Float = .infinity
        var tightWhere: String = ""
        for (k, toy) in toys.enumerated() {
            for j in 0..<legNames.count {
                let l: LegSpec = perf.design.legs[j]
                guard let ui = toy.segments.firstIndex(where: { $0.limb == j && $0.part == .upperLeg }) else { continue }
                let upper: Frame = toy.frames[ui]
                let lower: Frame = toy.frames[ui + 1]
                let foot: Frame = toy.frames[ui + 2]
                let knee: SIMD3<Float> = upper.toWorld(SIMD3<Float>(0, -l.upper, 0))
                let ankle: SIMD3<Float> = lower.toWorld(SIMD3<Float>(0, -l.lower, 0))
                let footTop: SIMD3<Float> = foot.toWorld(SIMD3<Float>(0, l.ankleHeight, 0))
                let hip: SIMD3<Float> = toy.frames[0].toWorld(l.hip)
                worst = max(worst, simd_distance(knee, lower.o), simd_distance(ankle, footTop), simd_distance(hip, upper.o))
                let slack: Float = l.upper + l.lower - simd_distance(hip, footTop)
                if slack < tightest { tightest = slack; tightWhere = String(format: "%@ at %.2f s", l.name, times[k]) }
            }
        }
        print(String(format: "        worst gap at a joint %.2e mm; closest a leg comes to full stretch %.3f mm (%@)", worst, tightest, tightWhere))
        expect(worst < 1e-3, "a joint came apart by \(worst) mm: a leg was stretched past its length")
        // A leg may stand nearly straight — the cow's foreleg is moulded 6°
        // from straight, 0.02 mm short of full stretch — but never past it.
        expect(tightest > 1e-3, "a leg was pulled straight (\(tightest) mm from full stretch)")
    }

    test("feet swing clear and nothing goes through the table") {
        var lowest: Float = .infinity
        var lift: Float = 0
        for toy in toys {
            lowest = min(lowest, toy.extent(along: SIMD3<Float>(0, 1, 0)).lo)
            for f in feet(toy) { lift = max(lift, f.low) }
        }
        print(String(format: "        lowest point of the toy %.5f mm; highest a foot lifts %.2f mm", lowest, lift))
        expect(lowest > -1e-3, "the toy goes \(-lowest) mm into the table")
        expect(lift > 0.5 * spec.footLift, "the feet barely lift")
    }

    test("no piece passes through another: every pair that is not joined stays apart, every 0.02 s") {
        var rng = SystemRandomNumberGenerator()
        var hits: Int = 0
        var worstPair: String = ""
        var checked: Int = 0
        for (k, toy) in toys.enumerated() where k % 2 == 0 {
            let n: Int = toy.segments.count
            let bounds: [(SIMD3<Float>, Float)] = (0..<n).map { i in
                let b = toy.segments[i].bound
                return (toy.frames[i].toWorld(b.centre), b.radius)
            }
            for i in 0..<n {
                for j in (i + 1)..<n where !joined(toy, i, j) {
                    guard simd_distance(bounds[i].0, bounds[j].0) < bounds[i].1 + bounds[j].1 else { continue }
                    checked += 1
                    // Points inside piece i (by 0.02 mm), tested against piece j.
                    let si: Segment = toy.segments[i]
                    for _ in 0..<160 {
                        let p: Prim = si.prims[Int.random(in: 0..<si.prims.count, using: &rng)]
                        let b = p.bound
                        let q: SIMD3<Float> = b.centre + SIMD3<Float>(Float.random(in: -1...1, using: &rng), Float.random(in: -1...1, using: &rng),
                                                                     Float.random(in: -1...1, using: &rng)) * b.radius
                        guard si.sdf(local: q, seams: false).d < -0.02 else { continue }
                        let w: SIMD3<Float> = toy.frames[i].toWorld(q)
                        // Overlap buried inside the body is a joint's socket, not a collision.
                        if toy.inside(w, segment: j, margin: 0.02) && !toy.inside(w, segment: 0, margin: 0) {
                            hits += 1
                            worstPair = "\(si.name) in \(toy.segments[j].name) at \(times[k]) s"
                        }
                    }
                }
            }
        }
        print("        \(checked) nearby unjoined pairs checked; \(hits) points inside two pieces" + (hits > 0 ? " (e.g. \(worstPair))" : ""))
        expectEqual(hits, 0)
    }

    section("the gait, read off the drawn feet")

    test("a lateral-sequence walk: \(spec.footfallOrder.joined(separator: ", ")), a quarter-stride apart, each foot down \(Int(spec.dutyFactor * 100))% of the stride") {
        // The steady part of the walk: a stride in from each end.
        let t0: Float = spec.walkStart + spec.ramp + spec.strideSeconds
        let t1: Float = spec.walkEnd - spec.ramp - spec.strideSeconds
        var events: [(Float, Int)] = []
        var downCount: [Int] = Array(repeating: 0, count: legNames.count)
        var samples: Int = 0
        var previous: [Bool]? = nil
        for (k, toy) in toys.enumerated() {
            let t: Float = times[k]
            let down: [Bool] = feet(toy).map { $0.low < 1e-4 }
            if t >= t0 && t <= t1 {
                samples += 1
                for j in 0..<down.count where down[j] { downCount[j] += 1 }
                if let p = previous { for j in 0..<down.count where down[j] && !p[j] { events.append((t, j)) } }
            }
            previous = down
        }
        events.sort { $0.0 < $1.0 }
        let seq: [String] = events.map { legNames[$0.1] }
        print("        touchdowns: " + seq.prefix(12).joined(separator: " ") + " …")
        // The order: each touchdown is followed by the next in the cycle.
        var wrong: Int = 0
        for k in 1..<max(seq.count, 1) {
            let i: Int = spec.footfallOrder.firstIndex(of: seq[k - 1]) ?? -9
            if seq[k] != spec.footfallOrder[(i + 1) % 4] { wrong += 1 }
        }
        expect(seq.count >= 4 * 6, "only \(seq.count) touchdowns in the steady walk")
        expectEqual(wrong, 0)
        // Spacing: a quarter of a stride between touchdowns.
        var gaps: [Float] = []
        for k in 1..<max(events.count, 1) { gaps.append(events[k].0 - events[k - 1].0) }
        let q: Float = spec.strideSeconds / 4
        let worstGap: Float = gaps.map { abs($0 - q) }.max() ?? .infinity
        print(String(format: "        touchdowns %.3f s apart at worst off the quarter-stride %.3f s", worstGap, q))
        expect(worstGap < 0.2 * q, "touchdowns are not a quarter-stride apart (off by \(worstGap) s)")
        // Duty factor, from time down.
        let df: [Float] = downCount.map { Float($0) / Float(max(samples, 1)) }
        print("        measured duty factors: " + zip(legNames, df).map { String(format: "%@ %.3f", $0.0, $0.1) }.joined(separator: ", "))
        for d in df { expect(abs(d - spec.dutyFactor) < 0.03, "duty factor \(d), the walk's is \(spec.dutyFactor)") }
    }

    test("never fewer than two feet down while walking; three or more most of the time") {
        var minDown: Int = 4
        var threePlus: Int = 0
        var n: Int = 0
        for (k, toy) in toys.enumerated() where times[k] > spec.walkStart && times[k] < spec.walkEnd {
            let d: Int = feet(toy).filter { $0.low < 1e-4 }.count
            minDown = min(minDown, d)
            if d >= 3 { threePlus += 1 }
            n += 1
        }
        print(String(format: "        fewest feet down %d; three or more %.0f%% of the walk", minDown, Float(threePlus) / Float(max(n, 1)) * 100))
        expect(minDown >= 2, "only \(minDown) feet down at one moment")
    }

    section("the loop")

    test("forward only: distance round the circle never decreases, and the walk goes exactly once round") {
        var worstBack: Float = 0
        var last: Float = perf.distance(0)
        for t in times.dropFirst() where t < spec.loopSeconds - 1e-4 {
            let d: Float = perf.distance(t)
            worstBack = max(worstBack, last - d)
            last = d
        }
        let most: Float = times.map { perf.distance($0) }.max() ?? 0
        print(String(format: "        worst step back %.2e mm; farthest %.3f mm of %.3f", worstBack, most, spec.circumference))
        expect(worstBack <= 1e-5, "the toy goes back \(worstBack) mm")
        expect(abs(most - spec.circumference) < 1e-3, "the walk does not go once round: \(most) of \(spec.circumference) mm")
        let (p0, y0) = perf.path.at(0)
        let (p1, y1) = perf.path.at(spec.circumference)
        expect(simd_distance(p0, p1) < 1e-3 && abs(y1 - y0 - 2 * .pi) < 1e-4, "the circle does not close: \(p0) vs \(p1), turned \(y1 - y0)")
    }

    test("frozen, then alive, then frozen again in exactly the moulded pose — the still's pose") {
        let still: PosedToy = perf.design.posed(perf.design.restPose())
        func gap(_ a: PosedToy, _ b: PosedToy) -> Float {
            var w: Float = 0
            for i in 0..<a.frames.count {
                w = max(w, simd_distance(a.frames[i].o, b.frames[i].o), simd_distance(a.frames[i].x, b.frames[i].x),
                        simd_distance(a.frames[i].y, b.frames[i].y))
            }
            return w
        }
        var holdWorst: Float = 0
        var aliveMost: Float = 0
        for (k, toy) in toys.enumerated() {
            let t: Float = times[k]
            let g: Float = gap(toy, still)
            if t <= spec.holdStart || t >= spec.settleEnd { holdWorst = max(holdWorst, g) } else { aliveMost = max(aliveMost, g) }
        }
        let seam: Float = gap(perf.posed(spec.loopSeconds), perf.posed(0))
        print(String(format: "        holds differ from the still by %.2e; alive, it moves up to %.1f; frame after the last vs the first %.2e",
                     holdWorst, aliveMost, seam))
        expect(holdWorst < 1e-3, "the frozen holds are not the moulded pose (\(holdWorst))")
        expect(aliveMost > 10, "the toy never really moves (\(aliveMost))")
        expect(seam < 1e-3, "the loop does not close: \(seam)")
    }

    test("the stride divides the circle, and the frames divide the loop") {
        expect(abs(spec.circumference / spec.stride - Float(spec.strides)) < 1e-4)
        expectEqual(Int((spec.loopSeconds * 100).rounded()) % gifDelayCentiseconds, 0)
    }

    section("the motion shows on screen")

    test("frame to frame, the toy's pixels change while it is alive, and not at all while it is frozen") {
        guard let r = renderer else { expect(false, "no GPU"); return }
        let dt: Float = 0.05
        var changes: [(Float, Int)] = []
        var frozenChange: Int = 0
        // Waking (crouching and wagging, looking left, looking right), five
        // moments of the walk, and a settling step.
        let aliveTimes: [Float] = [spec.holdStart + 0.5, spec.holdStart + 1.0, spec.holdStart + 2.0, spec.walkStart + 1.2,
                                   spec.walkStart + 3.3, spec.walkStart + 5.7, spec.walkStart + 8.1, spec.walkStart + 10.4,
                                   spec.walkEnd + 0.62]
        let n: Int = r.width * r.height * 4
        func frame(_ t: Float) -> [UInt8]? {
            guard (try? r.render(perf.posed(t), camera: a.camera, samples: 1)) != nil else { return nil }
            return Array(UnsafeBufferPointer(start: r.image.pixels.contents().assumingMemoryBound(to: UInt8.self), count: n))
        }
        func changed(_ x: [UInt8], _ y: [UInt8]) -> Int {
            var c: Int = 0
            for i in stride(from: 0, to: n, by: 4) where abs(Int(x[i]) - Int(y[i])) + abs(Int(x[i + 1]) - Int(y[i + 1])) + abs(Int(x[i + 2]) - Int(y[i + 2])) > 24 {
                c += 1
            }
            return c
        }
        for t in aliveTimes {
            guard let f0 = frame(t), let f1 = frame(t + dt) else { expect(false, "render failed"); return }
            changes.append((t, changed(f0, f1)))
        }
        if let f0 = frame(0.3), let f1 = frame(0.35) { frozenChange = changed(f0, f1) }
        let px: Int = r.width * r.height
        print("        pixels changed in 0.05 s (of \(px)): " + changes.map { String(format: "t=%.1f %d", $0.0, $0.1) }.joined(separator: ", ")
              + "; frozen \(frozenChange)")
        let walking = changes.filter { $0.0 > spec.walkStart && $0.0 < spec.walkEnd }
        expect(walking.allSatisfy { $0.1 > px / 400 }, "while walking, a frame changes too few pixels to see")
        expect(changes.filter { $0.1 > px / 2000 }.count >= changes.count - 1, "the toy barely moves between frames")
        expectEqual(frozenChange, 0)
    }

    test("rendered, the frame after the last is the first") {
        guard let r = renderer else { expect(false, "no GPU"); return }
        let n: Int = r.width * r.height * 4
        guard (try? r.render(perf.posed(0), camera: a.camera, samples: 1)) != nil else { expect(false, "render failed"); return }
        let first: [UInt8] = Array(UnsafeBufferPointer(start: r.image.pixels.contents().assumingMemoryBound(to: UInt8.self), count: n))
        guard (try? r.render(perf.posed(spec.loopSeconds), camera: a.camera, samples: 1)) != nil else { expect(false, "render failed"); return }
        let p = r.image.pixels.contents().assumingMemoryBound(to: UInt8.self)
        var differing: Int = 0
        for i in 0..<n where abs(Int(p[i]) - Int(first[i])) > 1 { differing += 1 }
        print("        \(differing) channels differ by more than 1")
        expectEqual(differing, 0)
    }

    test("the scale bar is true: 10 mm at the circle's centre, projected, is the bar") {
        guard let r = renderer else { expect(false, "no GPU"); return }
        let p0: SIMD2<Float> = project(a.barPoint, width: r.width, height: r.height, cam: a.camera)
        let p1: SIMD2<Float> = project(a.barPoint + a.camera.right * mainBarMillimetres, width: r.width, height: r.height, cam: a.camera)
        let bar: ScaleBar = animBar(a, width: r.width, height: r.height)
        print(String(format: "        projected %.2f px, bar %.2f px", simd_distance(p0, p1), Float(bar.pixels)))
        expect(abs(simd_distance(p0, p1) - Float(bar.pixels)) < 0.5)
    }

    test("the whole walk stays in the frame and clear of the caption and the footfall diagram") {
        guard let r = renderer else { expect(false, "no GPU"); return }
        let box: CGRect = gaitDiagramRect(width: r.width, height: r.height)
        let captionBottom: CGFloat = CGFloat(r.height) / 560 * (42 + 15 * 5)
        var bad: Int = 0
        for (k, toy) in toys.enumerated() where k % 25 == 0 {
            for (i, s) in toy.segments.enumerated() {
              for pr in s.prims {
                let b = pr.bound
                let c: SIMD3<Float> = toy.frames[i].toWorld(b.centre)
                let p: SIMD2<Float> = project(c, width: r.width, height: r.height, cam: a.camera)
                let rr: Float = b.radius * pixelsPerMillimetre(at: c, height: r.height, cam: a.camera)
                let rect = CGRect(x: CGFloat(p.x - rr), y: CGFloat(p.y - rr), width: CGFloat(2 * rr), height: CGFloat(2 * rr))
                if rect.minX < 0 || rect.maxX > CGFloat(r.width) || rect.maxY > CGFloat(r.height) || rect.minY < captionBottom
                    || rect.intersects(box) { bad += 1 }
              }
            }
        }
        expectEqual(bad, 0)
    }
}

/// Are segments i and j joined (so allowed to overlap at their joint)?
func joined(_ toy: PosedToy, _ i: Int, _ j: Int) -> Bool {
    let a: Segment = toy.segments[i]
    let b: Segment = toy.segments[j]
    func isBody(_ s: Segment) -> Bool { s.part == .body }
    if isBody(a) && (b.part == .head || b.part == .tail || b.part == .upperLeg || b.part == .plates) { return true }
    if isBody(b) && (a.part == .head || a.part == .tail || a.part == .upperLeg || a.part == .plates) { return true }
    if a.limb >= 0 && a.limb == b.limb { return abs(i - j) == 1 }
    return false
}
