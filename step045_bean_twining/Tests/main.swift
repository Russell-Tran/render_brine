// Tests for step 45. The bean's numbers are checked against Darwin and the
// floras, its twining against the direction Darwin recorded — on the stem as
// drawn, three ways — and its contact with the pole by probing the same
// distance function the GPU draws with. Then the loop: a loop later the scene
// must be the same scene one loop higher, and the film must only ever run
// forward.
//
// TWINE_MUTANT=left_handed|through_pole|rewind breaks it on purpose;
// `make mutants` requires the suite to fail for each.

import Foundation
import Metal
import simd

let mutant: Mutant = Mutant.fromEnvironment

final class Box {
    var renderer: PlantRenderer?
    var error: String = ""
    init() {
        do { renderer = try PlantRenderer(look: look, on: try findDevice()) } catch { self.error = "\(error)" }
    }
}
let box = Box()
if box.renderer == nil { print("no renderer: \(box.error)") }

func probe(_ s: PlantScene, _ pts: [V3], _ mask: UInt32 = allMaterials) -> [SIMD2<Float>]? {
    guard let r = box.renderer else { return nil }
    return try? r.probe(s, pts, mask: mask)
}

/// The stem as drawn in a scene: its chain.
func drawnStem(_ s: PlantScene) -> Chain? { s.chains.first { $0.material == .stem } }

/// Hours to test at: spread over a loop, off the frame lattice.
let testHours: [Float] = [0, 1.37, 4.02, 7.7]

// MARK: -

section("numbers, against the sources")

test("the revolution period is Darwin's: three circles of 2 h 0 m, 1 h 55 m, 1 h 55 m — 'the average of 1 hr. 57 m.'") {
    let minutes: Float = revolutionHours * 60
    print(String(format: "        revolution %.1f min", minutes))
    expect(abs(minutes - darwinAverageMinutes) < 0.5, "revolution is \(minutes) min")
}

test("winding one turn takes longer than one free revolution, by a ratio inside Darwin's two timed cases") {
    let lo: Float = darwinWindingRatios.min()!
    let hi: Float = darwinWindingRatios.max()!
    print(String(format: "        one turn round the pole: %.2f h (%.2f× the free revolution; Darwin %.2f–%.2f)",
                 gyreHours, gyreHours / revolutionHours, lo, hi))
    expect(gyreHours / revolutionHours >= lo && gyreHours / revolutionHours <= hi)
}

test("the label's hours are the loop's real duration: three turns round the pole") {
    let caption: String = timeCaption()
    print(String(format: "        loop = %.2f h; caption \"%@\"", loopHours, caption))
    expect(abs(loopHours - 3 * gyreHours) < 1e-4)
    expect(caption.contains(String(format: "%.0f hours", loopHours)))
}

test("the pole is a size beans are recorded twining round (Darwin 1875: 1/3-inch rods to 4-inch sticks; not 9 inches)") {
    expect(poleDiameter >= 8.4 && poleDiameter <= twinedDiameterRecorded)
    expect(poleDiameter < failedDiameterRecorded)
}

test("the leaf is trifoliate at the floras' sizes: petiole 4–9 cm, leaflets 4–16 × 2.5–11 cm") {
    expect(petioleRange.contains(petioleLength))
    for l in [terminalLeaflet, lateralLeaflet] {
        expect(leafletLengthRange.contains(l.x), "leaflet length \(l.x)")
        expect(leafletWidthRange.contains(l.y), "leaflet width \(l.y)")
    }
    // Three blades per leaf, measured on a scene.
    let s: PlantScene = beanScene(2.0, mutant)
    let n: Int = nodes(2.0, mutant).count
    expectEqual(s.leaflets.count, 3 * n)
}

section("which way round: anticlockwise seen from above (Darwin: 'against the sun')")

/// The wound part of the drawn stem, bottom to front.
func woundStem(_ t: Float) -> [V3] {
    let s: PlantScene = beanScene(t, mutant)
    guard let c = drawnStem(s) else { return [] }
    let n: Int = stem(t, mutant).frontIndex
    return Array(c.points.prefix(n))
}

test("seen from above, the stem goes round anticlockwise as it climbs, at every step") {
    var bad: Int = 0
    var total: Int = 0
    for t in testHours {
        let pts: [V3] = woundStem(t)
        // Anticlockwise seen from above: from +x towards −z. Up is +y, so this
        // is the right-handed angle about +y.
        let a: [Float] = unwrappedAngles(pts, origin: V3(0, 0, 0), axis: V3(0, 1, 0), reference: V3(1, 0, 0))
        for i in 1..<pts.count {
            total += 1
            let rises: Bool = pts[i].y > pts[i - 1].y
            let anticlockwise: Bool = a[i] > a[i - 1]
            if !(rises && anticlockwise) { bad += 1 }
        }
    }
    print("        \(total - bad) of \(total) steps up the stem turn anticlockwise from above")
    expect(total > 100 && bad == 0, "\(bad) steps turn the wrong way")
}

test("on the face of the pole towards the camera, the stem rises from left to right") {
    var good: Int = 0
    var total: Int = 0
    for t in testHours {
        let cam: Camera = camera(t)
        let pts: [V3] = woundStem(t)
        let toCam: V3 = simd_normalize(V3(cam.position.x, 0, cam.position.z))
        for i in 1..<pts.count {
            let a: V3 = pts[i - 1]
            let b: V3 = pts[i]
            // Both on the near half of the pole.
            guard simd_dot(V3(a.x, 0, a.z), toCam) > 0.5 * poleRadius,
                  simd_dot(V3(b.x, 0, b.z), toCam) > 0.5 * poleRadius else { continue }
            let pa: SIMD2<Float> = cam.project(a, width: 400, height: 520)
            let pb: SIMD2<Float> = cam.project(b, width: 400, height: 520)
            total += 1
            // Screen y grows downward: rising means pb.y < pa.y.
            if pb.y < pa.y && pb.x > pa.x { good += 1 }
        }
    }
    print("        \(good) of \(total) near-face steps rise left to right")
    expect(total > 20 && good == total, "\(total - good) near-face steps rise right to left")
}

test("it is a right-handed helix: the torsion of the wound stem is positive everywhere") {
    var bad: Int = 0
    var total: Int = 0
    for t in testHours {
        let pts: [V3] = woundStem(t)
        for i in 3..<pts.count {
            let d1: V3 = pts[i - 2] - pts[i - 3]
            let d2: V3 = pts[i - 1] - pts[i - 2]
            let d3: V3 = pts[i] - pts[i - 1]
            total += 1
            if simd_dot(simd_cross(d1, d2), d3) <= 0 { bad += 1 }
        }
    }
    print("        \(total - bad) of \(total) point quadruples have positive torsion")
    expect(total > 100 && bad == 0)
}

section("contact with the pole")

/// Points on the stem's surface: rings round its axis.
func stemSurface(_ c: Chain, from: Int, to: Int) -> [V3] {
    var out: [V3] = []
    for i in max(from, 1)..<min(to, c.points.count) {
        let a: V3 = c.points[i - 1]
        let b: V3 = c.points[i]
        let t: V3 = simd_normalize(b - a)
        let e1: V3 = perpendicular(to: t)
        let e2: V3 = simd_cross(t, e1)
        for k in 0..<3 {
            let p: V3 = a + (b - a) * (Float(k) / 3)
            let r: Float = c.radii[i - 1]
            for j in 0..<24 {
                let th: Float = Float(j) / 24 * 2 * Float.pi
                out.append(p + (e1 * cos(th) + e2 * sin(th)) * r)
            }
        }
    }
    return out
}

test("the wound stem touches the pole and never enters it — on the distance function") {
    for t in [Float(0.6), 5.3] {
        let s: PlantScene = beanScene(t, mutant)
        guard let c = drawnStem(s) else { expect(false, "no stem"); return }
        let front: Int = stem(t, mutant).frontIndex
        let clampStart: Int = front - samplesPerTurn * Int(tighteningTurns + 0.5)
        // Every point of the stem's surface, against the pole alone.
        let all: [V3] = stemSurface(c, from: 1, to: c.points.count)
        guard let d = probe(s, all, maskOf([.pole])) else { expect(false, "probe failed"); return }
        let deepest: Float = d.map { $0.x }.min()!
        // Along the clamped turns, each ring's nearest point is on the wood.
        let clamped: [V3] = stemSurface(c, from: 1, to: clampStart)
        guard let dc = probe(s, clamped, maskOf([.pole])) else { expect(false, "probe failed"); return }
        var worstGap: Float = 0
        var rings: Int = 0
        var i: Int = 0
        while i + 24 <= dc.count {
            let ringMin: Float = dc[i..<(i + 24)].map { $0.x }.min()!
            worstGap = max(worstGap, ringMin)
            rings += 1
            i += 24
        }
        print(String(format: "        hour %.1f: deepest stem point %.3f mm into the pole; clamped turns: widest gap %.3f mm over %d rings",
                     t, -deepest, worstGap, rings))
        expect(deepest > -0.05, "the stem enters the pole by \(-deepest) mm")
        expect(worstGap < 0.15, "the clamped stem stands \(worstGap) mm off the pole")
    }
}

test("the newest turn is laid loose and tightens onto the wood within a turn (Silk & Hubbard 1991; Isnard et al. 2009)") {
    let atFront: Float = windingRadius(0, mutant) - contactRadius(mutant)
    let turnBack: Float = windingRadius(2 * Float.pi * tighteningTurns, mutant) - contactRadius(mutant)
    print(String(format: "        clear of the wood: %.2f mm at the front, %.3f mm a turn behind", atFront, turnBack))
    expect(atFront > 1.0 && turnBack < 1e-4)
    // Monotone: it only ever tightens.
    var last: Float = .infinity
    for k in 0...100 {
        let r: Float = windingRadius(Float(k) / 100 * 2 * Float.pi * 1.5, mutant)
        expect(r <= last + 1e-6)
        last = r
    }
}

test("the spire is close at the front and opens to the mature pitch behind it (Darwin)") {
    let e: Float = 0.01
    let front: Float = (riseBehindFront(e) - riseBehindFront(0)) / e * 2 * Float.pi
    let behind: Float = (riseBehindFront(3 * Float.pi + e) - riseBehindFront(3 * Float.pi)) / e * 2 * Float.pi
    print(String(format: "        pitch at the front %.1f mm, 1.5 turns behind %.1f mm", front, behind))
    expect(abs(front - closeSpireFraction * maturePitch) < 0.5)
    expect(abs(behind - maturePitch) < 0.5)
}

section("growth")

test("the free tip sweeps round the pole anticlockwise, one turn per winding turn, and never touches it") {
    func tipAngle(_ t: Float) -> Float {
        let p: V3 = freeShoot(t, mutant).last!
        return atan2(-p.z, p.x)
    }
    var total: Float = 0
    let steps: Int = 60
    for k in 0..<steps {
        let t0: Float = gyreHours * Float(k) / Float(steps)
        let t1: Float = gyreHours * Float(k + 1) / Float(steps)
        var d: Float = tipAngle(t1) - tipAngle(t0)
        while d > Float.pi { d -= 2 * Float.pi }
        while d < -Float.pi { d += 2 * Float.pi }
        total += d
    }
    print(String(format: "        the tip turns %.3f turns in one winding period (%.2f h)", total / (2 * Float.pi), gyreHours))
    expect(abs(total - 2 * Float.pi) < 0.01, "tip turned \(total) rad")
    // The free shoot stays clear of the wood.
    var nearest: Float = .infinity
    for t in testHours {
        let free: [V3] = freeShoot(t, mutant)
        for (i, p) in free.enumerated() where i > 0 {
            let s: Float = freeLength * Float(free.count - 1 - i) / Float(free.count - 1)
            nearest = min(nearest, simd_length(V3(p.x, 0, p.z)) - poleRadius - stemRadiusFromApex(s))
        }
    }
    print(String(format: "        the free shoot stays %.2f mm or more off the pole", nearest))
    expect(nearest > 0)
}

test("leaves: one every internode (MODEL 10 cm of stem), alternate, each three-quarters of a turn round from the last") {
    let t: Float = 3.1
    let ns: [Node] = nodes(t, mutant).filter { $0.wound && frontAngle(t) - Float($0.index) * nodeAngleStep > 2 * Float.pi }
    guard ns.count >= 3, let c = drawnStem(beanScene(t, mutant)) else { expect(false, "too few nodes"); return }
    // Arc length along the drawn stem between consecutive mature nodes.
    func arcTo(_ p: V3) -> Float {
        var best: Float = .infinity
        var bestS: Float = 0
        var s: Float = 0
        for i in 1..<c.points.count {
            let a: V3 = c.points[i - 1]
            let b: V3 = c.points[i]
            let ab: V3 = b - a
            let h: Float = min(max(simd_dot(p - a, ab) / simd_dot(ab, ab), 0), 1)
            let d: Float = simd_distance(p, a + ab * h)
            if d < best { best = d; bestS = s + h * simd_length(ab) }
            s += simd_length(ab)
        }
        return bestS
    }
    var worst: Float = 0
    for i in 1..<ns.count {
        let gap: Float = arcTo(ns[i].position) - arcTo(ns[i - 1].position)
        worst = max(worst, abs(gap - internodeLength) / internodeLength)
        expect(ns[i].side == -ns[i - 1].side, "leaves \(i - 1) and \(i) on the same side")
        let turn: Float = atan2(-ns[i].position.z, ns[i].position.x) - atan2(-ns[i - 1].position.z, ns[i - 1].position.x)
        let c3: Float = cos(turn - 1.5 * Float.pi)
        expect(c3 > 0.999, "not three-quarters of a turn apart")
    }
    print(String(format: "        internode %.1f mm; drawn spacing within %.2f%% over %d mature nodes", internodeLength, worst * 100, ns.count))
    expect(worst < 0.02)
}

test("leaves open as they age: small and folded at the tip, full-size and spread lower down") {
    var lastG: Float = -1
    var lastFold: Float = 10
    for k in 0...40 {
        let a: Float = Float(k) * 0.5
        expect(leafGrowth(a) >= lastG)
        lastG = leafGrowth(a)
        let fold: Float = 1.4 + (0.18 - 1.4) * leafOpening(a)
        expect(fold <= lastFold)
        lastFold = fold
    }
    expect(leafGrowth(0) < 0.1 && leafGrowth(leafOpenHours) > 0.999)
}

section("the loop")

test("a loop later the scene is the same scene one loop higher (and so is the camera)") {
    var worst: Float = 0
    for t in testHours {
        let a: PlantScene = beanScene(t, mutant).shifted(by: loopRise)
        let b: PlantScene = beanScene(t + loopHours, mutant)
        worst = max(worst, a.largestDifference(from: b))
        let ca: Camera = camera(t)
        let cb: Camera = camera(t + loopHours)
        worst = max(worst, simd_distance(ca.position + V3(0, loopRise, 0), cb.position))
    }
    print(String(format: "        largest difference %.5f mm", worst))
    expect(worst < 0.01)
}

test("the film only runs forward: every frame later in the day and higher up than the last, the last into the first included") {
    var bad: Int = 0
    var steps: [Float] = []
    for f in 0..<frameCount {
        // The frame after the last is frame 0 of the next loop: one loop on.
        let t0: Float = hourOf(frame: f, mutant: mutant)
        let t1: Float = f + 1 < frameCount ? hourOf(frame: f + 1, mutant: mutant) : hourOf(frame: 0, mutant: mutant) + loopHours
        if !(t1 > t0) || !(frontHeight(t1) > frontHeight(t0)) { bad += 1 }
        steps.append(t1 - t0)
    }
    let spread: Float = steps.max()! - steps.min()!
    print(String(format: "        %d backward steps; step %.4f h, spread %.6f h", bad, steps[0], spread))
    expect(bad == 0, "\(bad) frames run backwards")
    expect(spread < 1e-4, "the steps are uneven: the join would show")
}

test("rendered, the frame one loop on matches frame 0") {
    guard let r = box.renderer else { expect(false, "no renderer"); return }
    let w: Int = 120
    let h: Int = 156
    let t0: Float = hourOf(frame: 0, mutant: mutant)
    guard let a = try? r.render(beanScene(t0, mutant), camera: camera(t0), width: w, height: h, samples: 1, tubeMin: 0),
          let b = try? r.render(beanScene(t0 + loopHours, mutant), camera: camera(t0 + loopHours), width: w, height: h,
                                samples: 1, tubeMin: 0) else { expect(false, "render failed"); return }
    var worst: Float = 0
    var off: Int = 0
    for i in 0..<(w * h) {
        let d: Float = simd_reduce_max(simd_abs(a.frame.rgba[i] - b.frame.rgba[i]))
        worst = max(worst, d)
        if d > 2.0 / 255 { off += 1 }
    }
    print(String(format: "        largest pixel difference %.4f; %d of %d pixels differ by more than 2/255", worst, off, w * h))
    // Not bit-for-bit: a loop on, every coordinate is 300 mm larger and a
    // float carries it to about 30 nm, which moves a few marched hits by a
    // hair. The scene test above holds the geometry to 10 µm.
    expect(worst < 0.05 && Float(off) < 0.02 * Float(w * h))
}

section("the distance function is a distance")

test("outside every surface it never claims more room than there is, beyond what the ray allows") {
    let t: Float = 2.3
    let s: PlantScene = beanScene(t, mutant)
    var rng = Lcg(state: 11)
    var a: [V3] = []
    var b: [V3] = []
    let step: Float = 0.25
    func add(_ p: V3) {
        a.append(p)
        b.append(p + rng.unit() * step)
    }
    let yf: Float = frontHeight(t)
    for _ in 0..<150_000 {
        add(V3(-180 + 360 * rng.next(), yf - 380 + 520 * rng.next(), -180 + 360 * rng.next()))
    }
    // Close to the leaves, where the estimate lives.
    for l in s.leaflets {
        for _ in 0..<1500 {
            let p: V3 = l.origin + l.u * (l.length * (-0.1 + 1.2 * rng.next())) + l.v * (l.halfWidth * (2.4 * rng.next() - 1.2))
                + l.w * (6 * rng.next() - 3)
            add(p)
        }
    }
    guard let da = probe(s, a), let db = probe(s, b) else { expect(false, "probe failed"); return }
    var worst = [Float](repeating: 0, count: 8)
    for i in 0..<a.count where da[i].y == db[i].y && da[i].x > 0 && db[i].x > 0 {
        let m: Int = Int(da[i].y)
        worst[m] = max(worst[m], abs(da[i].x - db[i].x) / step)
    }
    let names: [String] = ["-", "pole", "stem", "petiole", "leaf", "tendril", "twine", "stipule"]
    var line: String = "        worst over-report:"
    for m in 1...4 { line += String(format: " %@ %.3f", names[m], worst[m]) }
    print(line + String(format: "; the ray allows %.2f", 1 / stepScale))
    for m in 1...7 { expect(worst[m] * stepScale <= 1.0, "\(names[m]) oversteps: \(worst[m])") }
}

section("the picture")

test("frame 0 shows the pole, the stem wound on it, and green leaves") {
    guard let r = box.renderer else { expect(false, "no renderer"); return }
    let t: Float = 0
    guard let out = try? r.render(beanScene(t, mutant), camera: camera(t), width: 200, height: 260, samples: 1, tubeMin: 0)
    else { expect(false, "render failed"); return }
    var counts = [Int](repeating: 0, count: 8)
    var leafSum = SIMD3<Float>(0, 0, 0)
    for (i, s) in out.frame.seen.enumerated() {
        let m: Int = Int(s.x)
        counts[m] += 1
        if m == Material.leaf.rawValue {
            let c: SIMD4<Float> = out.frame.rgba[i]
            leafSum += SIMD3<Float>(c.x, c.y, c.z)
        }
    }
    let n: Float = Float(out.frame.seen.count)
    let leaf: SIMD3<Float> = leafSum / Float(max(counts[4], 1))
    print(String(format: "        pole %.1f%%, stem %.1f%%, leaf %.1f%%; leaf colour %.2f %.2f %.2f",
                 Float(counts[1]) / n * 100, Float(counts[2]) / n * 100, Float(counts[4]) / n * 100, leaf.x, leaf.y, leaf.z))
    expect(Float(counts[1]) / n > 0.03 && Float(counts[2]) / n > 0.005 && Float(counts[4]) / n > 0.03)
    expect(leaf.y > leaf.x && leaf.y > leaf.z, "leaves are not green")
}

finish()
