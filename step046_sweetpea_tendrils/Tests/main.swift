// Tests for step 46. The sweet pea's numbers are checked against the floras
// and Darwin; its tendrils against what a tendril is (a leaflet, so it grows
// from the leaf's tip) and what an anchored tendril must do (coil both ways,
// with one perversion and no net twist) — on the geometry as drawn; its grip on
// the net by probing the same distance function the GPU draws with. Then the
// loop: a loop later the scene must be the same one mesh square higher, and
// the film must only ever run forward.
//
// TENDRIL_MUTANT=no_perversion|stem_tendril|rewind breaks it on purpose;
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

/// Hours to test at: spread over a loop, off the frame lattice.
let testHours: [Float] = [0, 3.3, 9.71, 16.2, 21.05]

/// Points on the surface of a tube round a polyline.
func tubeSurface(_ pts: [V3], radius r: Float, around: Int = 12) -> [V3] {
    var out: [V3] = []
    if pts.count < 2 { return out }
    for i in 1..<pts.count {
        let a: V3 = pts[i - 1]
        let b: V3 = pts[i]
        guard simd_distance(a, b) > 1e-5 else { continue }
        let t: V3 = simd_normalize(b - a)
        let e1: V3 = perpendicular(to: t)
        let e2: V3 = simd_cross(t, e1)
        for k in 0..<2 {
            let p: V3 = a + (b - a) * (Float(k) / 2)
            for j in 0..<around {
                let th: Float = Float(j) / Float(around) * 2 * Float.pi
                out.append(p + (e1 * cos(th) + e2 * sin(th)) * r)
            }
        }
    }
    return out
}

// MARK: -

section("numbers, against the sources")

test("leaves: one pair of leaflets, 20–60 × 7–30 mm (Flora of China), stipules 1.5–2.5 cm (Flora of Pakistan)") {
    expect(leafletLengthRange.contains(leafletSize.x) && leafletWidthRange.contains(leafletSize.y))
    expect(stipuleRange.contains(stipuleLength))
    let t: Float = 5
    let s: PlantScene = sweetPeaScene(t, mutant)
    let n: Int = nodes(t).count
    expectEqual(s.leaflets.filter { $0.material == .leaf }.count, leafletsPerLeaf * n)
    expectEqual(s.leaflets.filter { $0.material == .stipule }.count, 2 * n)
}

test("the stem and the leaf stalks are winged (Flora of China: 'Stem climbing ... winged'; 'rachis winged')") {
    let s: PlantScene = sweetPeaScene(5, mutant)
    let stemWings: Int = s.ribbons.filter { $0.material == .stem }.count
    let stalkWings: Int = s.ribbons.filter { $0.material == .petiole }.count
    let stemSegments: Int = (s.chains.first { $0.material == .stem }?.points.count ?? 1) - 1
    print("        \(stemWings) wing panels on \(stemSegments) stem segments; \(stalkWings) on leaf stalks")
    expectEqual(stemWings, stemSegments)
    expect(stalkWings >= nodes(5).count)
    // Each wing lies at right angles to its segment — what makes it exact.
    for r in s.ribbons { expect(abs(simd_dot(r.n, simd_normalize(r.b - r.a))) < 1e-4) }
}

test("the frame says what it is: a relative of the garden pea, not the one you eat, with toxic seeds") {
    expect(relationCaption.contains("relative of the garden pea") && relationCaption.contains("not the pea you eat"))
    expect(toxicCaption.contains("toxic"))
    expect(timeCaption().contains(String(format: "%.0f hours", loopHours)))
    print("        \"\(relationCaption) \(toxicCaption)\"; \"\(timeCaption())\"")
}

test("the tendrils and young stem sweep round in the garden pea's time (Darwin: ellipses of 1 h 20 m and 1 h 30 m)") {
    let minutes: Float = nutationHours * 60
    print(String(format: "        one sweep %.1f min, %d in a loop", minutes, nutationsPerLoop))
    expect(peaEllipseMinutesRange.contains(minutes))
    expect(abs(loopHours / nutationHours - Float(nutationsPerLoop)) < 1e-4)
}

test("the free coil forms half a day to a day after the catch (Darwin: 'in half a day, or in a day or two')") {
    expect(contractionStartsAfter >= 7 && contractionStartsAfter <= 48)
    expect(contractionCompleteAfter > contractionStartsAfter && contractionCompleteAfter <= 48)
}

section("tendrils are leaflets: they grow from the leaf's tip")

test("every tendril starts where the leaflets sit, at the end of the leaf, well out from the stem") {
    var worst: Float = 0
    var nearest: Float = .infinity
    var count: Int = 0
    for t in testHours {
        let s: PlantScene = sweetPeaScene(t, mutant)
        let leafletOrigins: [V3] = s.leaflets.filter { $0.material == .leaf }.map { $0.origin }
        for n in nodes(t) {
            let td: TendrilGeometry = tendril(n, t, mutant)
            // The tendril's base is a leaflet pair's base.
            let d: Float = leafletOrigins.map { simd_distance($0, td.base) }.min() ?? .infinity
            worst = max(worst, d)
            let stem: V3 = stemPoint(n.y, t)
            nearest = min(nearest, simd_distance(V3(td.base.x, 0, td.base.z), V3(stem.x, 0, stem.z)))
            count += 1
            // And the drawn chain starts there.
            let drawn: Bool = s.chains.contains { $0.material == .tendril && simd_distance($0.points[0], td.base) < 1e-4 }
            expect(drawn, "no tendril chain starts at node \(n.index)'s tendril base")
        }
    }
    print(String(format: "        %d tendrils: base within %.4f mm of the leaflets' base, at least %.1f mm out from the stem",
                 count, worst, nearest))
    expect(worst < 1e-3, "a tendril starts \(worst) mm from its leaf's tip")
    expect(nearest > stemRadius + wingWidth, "a tendril starts on the stem")
}

section("the anchored free coil: two hands and one perversion")

/// The handedness runs of a coil: θ about the chord, measured right-handedly,
/// against distance along the chord. Returns the signs of the runs, and the
/// turns in each run.
func handednessRuns(_ pts: [V3]) -> (signs: [Int], turns: [Float], net: Float) {
    let a: V3 = pts.first!
    let e: V3 = simd_normalize(pts.last! - a)
    let theta: [Float] = unwrappedAngles(pts, origin: a, axis: e, reference: perpendicular(to: e))
    var signs: [Int] = []
    var turns: [Float] = []
    var runStart: Float = 0
    var started: Bool = false
    for i in 1..<pts.count {
        let q: V3 = pts[i] - a
        let rho: Float = simd_length(q - e * simd_dot(q, e))
        // Only where the coil is off its chord; the ends ramp in along it.
        guard rho > 0.5 * coilRadius else { continue }
        let dz: Float = simd_dot(pts[i] - pts[i - 1], e)
        let dth: Float = theta[i] - theta[i - 1]
        guard abs(dth) > 1e-3, abs(dz) > 1e-6 else { continue }
        let sgn: Int = (dth / dz) > 0 ? 1 : -1
        if !started { runStart = theta[i - 1]; started = true }
        if signs.last != sgn {
            if !signs.isEmpty { turns.append(abs(theta[i - 1] - runStart) / (2 * Float.pi)); runStart = theta[i - 1] }
            signs.append(sgn)
        }
    }
    if started { turns.append(abs(theta[theta.count - 1] - runStart) / (2 * Float.pi)) }
    return (signs, turns, (theta.last! - theta.first!) / (2 * Float.pi))
}

test("every sprung tendril's free coil has exactly one perversion, a right-handed helix one side and left-handed the other, with as many turns each way (Darwin) and no net twist") {
    var checked: Int = 0
    for t in testHours {
        for n in nodes(t) {
            let td: TendrilGeometry = tendril(n, t, mutant)
            guard td.contraction > 0.95 else { continue }
            let r = handednessRuns(td.free)
            checked += 1
            if checked == 1 {
                print(String(format: "        runs %@; turns %@; net twist %.3f turns", "\(r.signs)",
                             r.turns.map { String(format: "%.2f", $0) }.joined(separator: " / "), r.net))
            }
            expect(r.signs.count == 2, "node \(n.index): \(r.signs.count - 1) perversions")
            guard r.signs.count == 2 else { continue }
            expect(r.signs[0] == -r.signs[1], "both sides the same hand")
            expect(abs(r.turns[0] - r.turns[1]) < 0.25, "turns \(r.turns[0]) one way, \(r.turns[1]) the other")
            expect(r.turns[0] >= 1 && r.turns[1] >= 1, "under a turn a side")
            expect(abs(r.net) < 0.05, "net twist \(r.net) turns between two anchored ends")
        }
    }
    print("        \(checked) sprung tendrils checked")
    expect(checked >= 3)
}

test("the spring draws the leaf in: the coil keeps its length, and its chord shortens as it coils") {
    let n = NodeInfo(index: 3, y: 150, age: 0, side: -1)
    let tBirth: Float = 150 / growthRate
    var lastChord: Float = .infinity
    var worstLength: Float = 0
    var pull: Float = 0
    for k in 0...24 {
        let age: Float = contactAge + contractionStartsAfter + Float(k) / 24 * (contractionCompleteAfter - contractionStartsAfter)
        let nn = NodeInfo(index: n.index, y: n.y, age: age, side: n.side)
        let td: TendrilGeometry = tendril(nn, tBirth + age, mutant)
        let k0: Catch = tendrilCatch(nn, tBirth + age)
        let chord: Float = simd_distance(td.free.first!, td.free.last!)
        worstLength = max(worstLength, abs(polylineLength(td.free) - k0.reach) / k0.reach)
        expect(chord <= lastChord + 1e-4)
        lastChord = chord
        pull = simd_distance(td.leafTip, k0.leafTip)
    }
    print(String(format: "        coil length held to %.2f%%; the leaf tip is drawn %.1f mm towards the net", worstLength * 100, pull))
    expect(worstLength < 0.01)
    expect(pull > 5)
}

section("gripping the net")

test("a searching tendril never enters a string, and its tip arrives touching one exactly at contact") {
    var deepest: Float = .infinity
    var arrival: Float = .infinity
    let rw: Float = twineRadius + tendrilRadius
    for k in 0..<60 {
        let t: Float = loopHours * Float(k) / 60 + 0.013
        for n in nodes(t) where n.age < contactAge {
            let td: TendrilGeometry = tendril(n, t, mutant)
            let tip: V3 = td.free.last!
            let d: Float = simd_distance(V3(tip.x, 0, tip.z), V3(td.stringAxis.x, 0, td.stringAxis.z)) - rw
            deepest = min(deepest, d)
        }
    }
    // At the moment of contact.
    let n = NodeInfo(index: 4, y: 200, age: contactAge, side: 1)
    let tc: Float = 200 / growthRate + contactAge
    let td: TendrilGeometry = tendril(n, tc - 1e-4, mutant)
    let tip: V3 = td.free.last!
    arrival = simd_distance(V3(tip.x, 0, tip.z), V3(td.stringAxis.x, 0, td.stringAxis.z)) - rw
    print(String(format: "        before contact the tip stays %.2f mm or more off the string; at contact %.4f mm", deepest, arrival))
    expect(deepest > 0)
    expect(abs(arrival) < 0.01)
}

test("the coil round the string touches it all the way round and never enters it — on the distance function") {
    var worstIn: Float = 0
    var worstGap: Float = 0
    var rings: Int = 0
    for t in testHours {
        let s: PlantScene = sweetPeaScene(t, mutant)
        for n in nodes(t) where n.age > contactAge + wrapHours {
            let td: TendrilGeometry = tendril(n, t, mutant)
            let pts: [V3] = tubeSurface(td.wrap, radius: tendrilRadius, around: 16)
            guard let d = probe(s, pts, maskOf([.twine])) else { expect(false, "probe failed"); return }
            var i: Int = 0
            while i + 16 <= d.count {
                let ring: [Float] = d[i..<(i + 16)].map { $0.x }
                worstIn = min(worstIn, ring.min()!)
                worstGap = max(worstGap, ring.min()!)
                rings += 1
                i += 16
            }
        }
    }
    print(String(format: "        %d rings round the string: deepest %.3f mm in, widest gap %.3f mm", rings, -worstIn, worstGap))
    expect(rings > 100)
    expect(worstIn > -0.03, "a tendril enters the string by \(-worstIn) mm")
    expect(worstGap < 0.06, "the coil stands \(worstGap) mm off the string")
}

test("no part of any tendril passes through the net, at any hour of the loop") {
    var deepest: Float = .infinity
    for k in 0..<24 {
        let t: Float = loopHours * Float(k) / 24 + 0.37
        let s: PlantScene = sweetPeaScene(t, mutant)
        var pts: [V3] = []
        for n in nodes(t) {
            let td: TendrilGeometry = tendril(n, t, mutant)
            pts += tubeSurface(td.main, radius: tendrilRadius * 1.15, around: 8)
            pts += tubeSurface(td.catching, radius: tendrilRadius, around: 8)
            for b in td.sides { pts += tubeSurface(b, radius: tendrilRadius * 0.9, around: 8) }
        }
        guard let d = probe(s, pts, maskOf([.twine])) else { expect(false, "probe failed"); return }
        deepest = min(deepest, d.map { $0.x }.min() ?? .infinity)
    }
    print(String(format: "        deepest tendril point %.3f mm into a string", max(-deepest, 0)))
    expect(deepest > -0.03)
}

test("every stage is on screen somewhere in the loop: searching, curling round the string, anchored, coiling, sprung") {
    var seen = Set<Int>()
    for f in stride(from: 0, to: frameCount, by: 10) {
        let t: Float = hourOf(frame: f)
        let cam: Camera = camera(t)
        for n in nodes(t) {
            let td: TendrilGeometry = tendril(n, t, mutant)
            let p: SIMD2<Float> = cam.project(td.free[td.free.count / 2], width: 480, height: 624)
            if p.x > 0 && p.x < 480 && p.y > 0 && p.y < 624 { seen.insert(td.stage.rawValue) }
        }
    }
    print("        stages seen: \(seen.sorted())")
    expectEqual(seen.count, 5)
    // And in order, for one tendril.
    var last: Int = -1
    for k in 0...100 {
        let s: Int = stage(Float(k) * 0.5).rawValue
        expect(s >= last)
        last = s
    }
}

section("the loop")

test("a loop later the scene is the same scene one mesh square higher (and so is the camera)") {
    var worst: Float = 0
    for t in testHours {
        let a: PlantScene = sweetPeaScene(t, mutant).shifted(by: loopRise)
        let b: PlantScene = sweetPeaScene(t + loopHours, mutant)
        worst = max(worst, a.largestDifference(from: b))
        worst = max(worst, simd_distance(camera(t).position + V3(0, loopRise, 0), camera(t + loopHours).position))
    }
    print(String(format: "        largest difference %.5f mm", worst))
    expect(worst < 0.01)
    expect(abs(loopRise - meshSpacing) < 1e-4, "the net would not repeat with the loop")
}

test("the film only runs forward: every frame later and higher than the last, the last into the first included") {
    var bad: Int = 0
    var steps: [Float] = []
    for f in 0..<frameCount {
        let t0: Float = hourOf(frame: f, mutant: mutant)
        let t1: Float = f + 1 < frameCount ? hourOf(frame: f + 1, mutant: mutant) : hourOf(frame: 0, mutant: mutant) + loopHours
        if !(t1 > t0) || !(apexHeight(t1) > apexHeight(t0)) { bad += 1 }
        steps.append(t1 - t0)
    }
    let spread: Float = steps.max()! - steps.min()!
    print(String(format: "        %d backward steps; step %.4f h, spread %.6f h", bad, steps[0], spread))
    expect(bad == 0, "\(bad) frames run backwards")
    expect(spread < 1e-4)
}

test("rendered, the frame one loop on matches frame 0") {
    guard let r = box.renderer else { expect(false, "no renderer"); return }
    let w: Int = 120
    let h: Int = 156
    let t0: Float = hourOf(frame: 0, mutant: mutant)
    guard let a = try? r.render(sweetPeaScene(t0, mutant), camera: camera(t0), width: w, height: h, samples: 1, tubeMin: 0),
          let b = try? r.render(sweetPeaScene(t0 + loopHours, mutant), camera: camera(t0 + loopHours), width: w,
                                height: h, samples: 1, tubeMin: 0) else { expect(false, "render failed"); return }
    var worst: Float = 0
    var off: Int = 0
    for i in 0..<(w * h) {
        let d: Float = simd_reduce_max(simd_abs(a.frame.rgba[i] - b.frame.rgba[i]))
        worst = max(worst, d)
        if d > 2.0 / 255 { off += 1 }
    }
    print(String(format: "        largest pixel difference %.4f; %d of %d pixels differ by more than 2/255", worst, off, w * h))
    expect(worst < 0.05 && Float(off) < 0.02 * Float(w * h))
}

section("the distance function is a distance")

test("outside every surface it never claims more room than there is, beyond what the ray allows") {
    let t: Float = 9.71
    let s: PlantScene = sweetPeaScene(t, mutant)
    var rng = Lcg(state: 5)
    var a: [V3] = []
    var b: [V3] = []
    let step: Float = 0.1
    func add(_ p: V3) {
        a.append(p)
        b.append(p + rng.unit() * step)
    }
    let ya: Float = apexHeight(t)
    for _ in 0..<150_000 { add(V3(-110 + 220 * rng.next(), ya - 260 + 300 * rng.next(), -20 + 70 * rng.next())) }
    for l in s.leaflets {
        for _ in 0..<800 {
            add(l.origin + l.u * (l.length * (-0.1 + 1.2 * rng.next())) + l.v * (l.halfWidth * (2.4 * rng.next() - 1.2))
                + l.w * (4 * rng.next() - 2))
        }
    }
    // Close round the winged stem and stalks.
    for r in s.ribbons {
        for _ in 0..<40 { add(r.a + (r.b - r.a) * rng.next() + rng.unit() * (3 * rng.next())) }
    }
    guard let da = probe(s, a), let db = probe(s, b) else { expect(false, "probe failed"); return }
    var worst = [Float](repeating: 0, count: 8)
    for i in 0..<a.count where da[i].y == db[i].y && da[i].x > 0 && db[i].x > 0 {
        let m: Int = Int(da[i].y)
        worst[m] = max(worst[m], abs(da[i].x - db[i].x) / step)
    }
    let names: [String] = ["-", "pole", "stem", "petiole", "leaf", "tendril", "twine", "stipule"]
    var line: String = "        worst over-report:"
    for m in 2...7 { line += String(format: " %@ %.3f", names[m], worst[m]) }
    print(line + String(format: "; the ray allows %.2f", 1 / stepScale))
    for m in 1...7 { expect(worst[m] * stepScale <= 1.0, "\(names[m]) oversteps: \(worst[m])") }
}

section("the picture")

test("frame 0 shows the net, the winged stem, leaves, and tendrils") {
    guard let r = box.renderer else { expect(false, "no renderer"); return }
    let t: Float = 0
    guard let out = try? r.render(sweetPeaScene(t, mutant), camera: camera(t), width: 240, height: 312, samples: 1,
                                  tubeMin: 0.3) else { expect(false, "render failed"); return }
    var counts = [Int](repeating: 0, count: 8)
    for s in out.frame.seen { counts[Int(s.x)] += 1 }
    let n: Float = Float(out.frame.seen.count)
    print(String(format: "        twine %.1f%%, stem %.1f%%, leaf %.1f%%, tendril %.2f%%",
                 Float(counts[6]) / n * 100, Float(counts[2]) / n * 100, Float(counts[4]) / n * 100, Float(counts[5]) / n * 100))
    expect(Float(counts[6]) / n > 0.01 && Float(counts[2]) / n > 0.01 && Float(counts[4]) / n > 0.05)
    expect(Float(counts[5]) / n > 0.002)
}

finish()
