// Tests for step 66. The anatomy is checked against its tables and studies —
// the maxillary tooth sizes, the arch widths, the palate's depth, the rugae
// and the papilla — the numbering and its handedness against the renderer's
// own camera, the distance function against the definition of a distance, and
// the finished picture against the colour science it claims, by reading
// colours back off the rendered pixels.
//
// UPPER_MUTANT=mandibularSizes|mirrored|noRugae breaks the anatomy and
// MOUTH_MUTANT=flat|opaque the shading, on purpose; `make mutants` requires
// the suite to fail for each.

import Foundation
import Metal
import simd

let mutant: Mutant = {
    switch ProcessInfo.processInfo.environment["MOUTH_MUTANT"] {
    case "flat": return .flat
    case "opaque": return .opaque
    default: return .none
    }
}()

/// Holds the scene so a failure to build one is a test failure, not a crash.
final class SceneBox {
    var renderer: MouthRenderer?
    var opaque: MouthRenderer?
    var error: String = ""
    init() {
        do {
            let device = try findDevice()
            renderer = try MouthRenderer(mutant: mutant, on: device)
            opaque = try MouthRenderer(mutant: .opaque, on: device)
        } catch { self.error = "\(error)" }
    }
}
let box = SceneBox()

func probe(_ pts: [SIMD3<Float>]) -> [SIMD2<Float>]? {
    guard let r = box.renderer else { return nil }
    return try? r.probe(pts)
}

/// A point given in the arch's frame (Anatomy.swift), as the world has it.
func w(_ x: Float, _ y: Float, _ z: Float) -> SIMD3<Float> { world(SIMD3<Float>(x, y, z)) }

let placed: [PlacedTooth] = placeTeeth()

section("the teeth, against Wheeler's maxillary tables")

test("eight teeth per side, each row as Wheeler's 9th edition gives it for the UPPER jaw") {
    expectEqual(toothTable.count, 8)
    expectEqual(toothTable.map { $0.crownHeight }, [10.5, 9.0, 10.0, 8.5, 8.5, 7.5, 7.0, 6.5])
    expectEqual(toothTable.map { $0.width }, [8.5, 6.5, 7.5, 7.0, 7.0, 10.0, 9.0, 8.5])
    expectEqual(toothTable.map { $0.cervicalWidth }, [7.0, 5.0, 5.5, 5.0, 5.0, 8.0, 7.0, 6.5])
    expectEqual(toothTable.map { $0.depth }, [7.0, 6.0, 8.0, 9.0, 9.0, 11.0, 11.0, 10.0])
    expectEqual(toothTable.map { $0.cervicalDepth }, [6.0, 5.0, 7.0, 8.0, 8.0, 10.0, 10.0, 9.5])
    expectEqual(toothTable.map { $0.cejCurve }, [3.5, 3.0, 2.5, 1.0, 1.0, 1.0, 1.0, 1.0])
}

test("the upper teeth are not the lower: a central 8.5 mm wide, not 5.0; molars broader than long") {
    // The numbers that tell the jaws apart at a glance (step 20's lower table:
    // central 5.0, lateral 5.5, first molar 11.0 long and 10.5 across).
    let c: ToothSpec = toothTable[0]
    let l: ToothSpec = toothTable[1]
    expect(c.width == 8.5 && c.width > l.width + 1.5, "central \(c.width), lateral \(l.width)")
    for m in toothTable[5...7] { expect(m.depth > m.width, "\(m.name) should be broader buccolingually than long") }
    // In the upper arch the central's crown is the longest; in the lower, the canine's.
    expect(toothTable.map { $0.crownHeight }.max() == c.crownHeight, "the central should be the longest crown")
}

test("every crown is narrower at the neck than at its contacts, in both directions") {
    for t in toothTable {
        expect(t.cervicalWidth < t.width, "\(t.name) mesiodistal")
        expect(t.cervicalDepth < t.depth, "\(t.name) labiolingual")
    }
}

section("the arch, its numbers and its handedness")

test("sixteen teeth, #1 to #16 once each, the wisdom teeth #1 and #16 included") {
    expectEqual(placed.count, 16)
    expectEqual(placed.map { $0.number }.sorted(), Array(1...16))
    for n in [1, 16] {
        guard let i = toothIndex(n, in: placed) else { expect(false, "no #\(n)"); continue }
        expect(placed[i].spec.name == "third molar", "#\(n) is a \(placed[i].spec.name)")
    }
    for n in [8, 9] {
        guard let i = toothIndex(n, in: placed) else { expect(false, "no #\(n)"); continue }
        expect(placed[i].spec.name == "central incisor", "#\(n) is a \(placed[i].spec.name)")
    }
}

test("#1–#8 on the patient's right (+x), #9–#16 on the left, and seen from the front, #16 on the viewer's right") {
    for t in placed {
        let right: Bool = t.number <= 8
        expect(right ? t.centre.x > 0 : t.centre.x < 0, "#\(t.number) at x = \(t.centre.x)")
    }
    // Independent of the numbering: a camera in front of the patient, looking
    // back into the mouth with the renderer's own convention for "right".
    let front = Camera(position: SIMD3<Float>(0, -5, -80), target: SIMD3<Float>(0, 5, 20))
    guard let i16 = toothIndex(16, in: placed), let i1 = toothIndex(1, in: placed) else { expect(false, "missing"); return }
    let c16: SIMD2<Float> = placed[i16].centre
    let c1: SIMD2<Float> = placed[i1].centre
    let s16: Float = simd_dot(w(c16.x, 0, c16.y) - front.position, front.right)
    let s1: Float = simd_dot(w(c1.x, 0, c1.y) - front.position, front.right)
    expect(s16 > 0 && s1 < 0, "#16 should appear on the viewer's right (\(s16)), #1 on the left (\(s1))")
}

test("the two sides are mirror images, and neighbours touch along the arch") {
    for i in 0..<8 {
        let l: PlacedTooth = placed[i]
        let r: PlacedTooth = placed[i + 8]
        expect(abs(l.centre.x + r.centre.x) < 1e-4 && abs(l.centre.y - r.centre.y) < 1e-4, "\(l.spec.name) not mirrored")
    }
    for side in 0..<2 {
        for i in 0..<7 {
            let a: PlacedTooth = placed[side * 8 + i]
            let b: PlacedTooth = placed[side * 8 + i + 1]
            let gap: Float = simd_distance(a.centre, b.centre)
            let expected: Float = (a.spec.width + b.spec.width) / 2
            expect(gap <= expected + 1e-3 && gap > expected - 0.4, "\(a.spec.name)–\(b.spec.name): \(gap) vs \(expected)")
        }
    }
    let centrals: Float = simd_distance(placed[0].centre, placed[8].centre)
    // A chord across the midline, a touch shorter than the 8.5 mm of arc.
    expect(abs(centrals - toothTable[0].width) < 0.1, "the centrals are \(centrals) mm apart")
}

test("the arch lands on published upper widths: canines within a standard deviation, first molars on target") {
    // Hawley's circle, its radius the sum of three widths from the table,
    // is not told where the canines go. Wankhede et al. 2023 measured
    // 35.2 ± 2.0 mm cusp to cusp (mean of men and women); step 20's lower
    // arch put its canines 25 mm apart.
    let canines: Float = placed[10].centre.x - placed[2].centre.x
    let premolars: Float = placed[11].centre.x - placed[3].centre.x
    let molars: Float = placed[13].centre.x - placed[5].centre.x
    print(String(format: "        canines %.1f mm (published %.1f ± %.1f), first premolars %.1f (%.1f ± %.1f), first molars %.2f",
                 canines, publishedIntercanine.mean, publishedIntercanine.sd,
                 premolars, publishedInterpremolar.mean, publishedInterpremolar.sd, molars))
    expect(abs(canines - publishedIntercanine.mean) <= publishedIntercanine.sd, "intercanine \(canines) mm")
    expect(abs(premolars - publishedInterpremolar.mean) <= publishedInterpremolar.sd, "interpremolar \(premolars) mm")
    expect(abs(molars - firstMolarSpan) < 0.01, "first molars \(molars) mm apart")
    expect(abs(hawleyRadius - 22.5) < 1e-5, "Hawley radius \(hawleyRadius)")
}

test("every tooth faces out of the arch, away from the palate") {
    for t in placed {
        let toCentre = SIMD2<Float>(0, 25) - t.centre
        expect(simd_dot(t.outward, toCentre) < 0, "#\(t.number) faces in")
        expect(abs(simd_dot(t.outward, t.tangent)) < 1e-4, "#\(t.number) outward is not perpendicular")
    }
}

test("the crowns hang down: in the world a crown is below its gum, and the palate above both") {
    guard let i = toothIndex(12, in: placed) else { expect(false, "no #12"); return }
    let t: PlacedTooth = placed[i]      // the patient's left first premolar
    let c = SIMD3<Float>(t.centre.x, 0, t.centre.y)
    let out: SIMD2<Float> = t.outward * (t.spec.cervicalDepth / 2 + 1.0)
    let pts: [SIMD3<Float>] = [
        world(c + SIMD3<Float>(0, -3, 0)),                                     // mid crown
        world(c + SIMD3<Float>(out.x, -t.spec.crownHeight - 3, out.y)),        // beside the root
        world(c + SIMD3<Float>(0, 6, 0)),                                      // past the cusps
    ]
    guard let r = probe(pts) else { expect(false, "probe failed: \(box.error)"); return }
    expect(pts[0].y > 0 && r[0].x < 0 && r[0].y == 1, "mid crown at world y \(pts[0].y) should be enamel: \(r[0])")
    expect(pts[1].y > pts[0].y && r[1].x < 0 && r[1].y == 2, "above the crown, over the root, should be gum: \(r[1])")
    expect(pts[2].y < 0 && r[2].x > 0, "below the cusps should be air: \(r[2])")
}

section("the palate")

/// The palate's surface over (x, z) in the arch frame, found by probing the
/// world upward from the occlusal plane: the first point that is palate.
func palateSurface(_ x: Float, _ z: Float) -> Float? {
    let pts: [SIMD3<Float>] = (0..<4000).map { w(x, -Float($0) * 0.01, z) }
    guard let d = probe(pts), let hit = d.firstIndex(where: { $0.x < 0 }) else { return nil }
    return d[hit].y == 3 ? -Float(hit) * 0.01 : nil
}

test("the vault stands 15.6 mm above the first molars' palatal gum margins, at the raphe") {
    guard let i = toothIndex(14, in: placed) else { expect(false, "no #14"); return }
    let m: PlacedTooth = placed[i]
    guard let roof = palateSurface(0, m.centre.y) else { expect(false, "no palate over the midline at the first molars"); return }
    let margin: Float = -m.spec.crownHeight + gumMarginAboveCEJ
    let height: Float = margin - roof
    print(String(format: "        raphe %.2f mm beyond the margin line (Alaqeely et al.: %.2f ± 2.7)", height, palatalVaultHeight))
    expect(abs(height - palatalVaultHeight) < 0.3, "vault height \(height)")
}

test("four rugae each side, behind the incisors and no further back than the second premolars") {
    expectEqual(rugae.count, 2 * rugaePerSide)
    let incisorBack: Float = placed[0].centre.y + placed[0].spec.depth / 2
    let reach: Float = placed[4].centre.y
    for g in rugae {
        for p in g.points {
            expect(p.z > incisorBack && p.z <= reach + 0.01, "ruga point at z \(p.z): not between \(incisorBack) and \(reach)")
        }
    }
    let left: Int = rugae.filter { $0.points[0].x < 0 }.count
    expectEqual(left, rugaePerSide)
    for (g, len) in zip(stride(from: 0, to: rugae.count, by: 2).map { rugae[$0] }, rugaLengths) {
        let run: Float = simd_distance(SIMD2<Float>(g.points[0].x, g.points[0].z), SIMD2<Float>(g.points[2].x, g.points[2].z))
        expect(run >= 5 && run <= 10 && abs(run - len) < 0.01, "a primary ruga runs \(run) mm (Lysell: 5–10)")
    }
}

test("the rugae stand out of the palate as ridges: a crest a millimetre proud, valleys either side") {
    guard !rugae.isEmpty else { expect(false, "there are no rugae"); return }
    // Across each ruga, half way along its first leg: how far the palate's
    // surface stands proud of the smooth vault, on the crest and 1.6 mm to
    // either side. Neighbouring rugae are about 3 mm apart, so 1.6 mm is in
    // the valley between them.
    for (k, g) in rugae.enumerated() {
        let a = SIMD2<Float>(g.points[0].x, g.points[0].z)
        let b = SIMD2<Float>(g.points[1].x, g.points[1].z)
        let mid: SIMD2<Float> = (a + b) / 2
        let along: SIMD2<Float> = simd_normalize(b - a)
        let across = SIMD2<Float>(-along.y, along.x)
        var proud: [Float] = []
        for off in [Float(0), Float(-1.6), Float(1.6)] {
            let q: SIMD2<Float> = mid + across * off
            guard let s = palateSurface(q.x, q.y), let roof = vault.roof(q.x, q.y) else { proud.append(-9); continue }
            proud.append(s - roof)
        }
        if k % 2 == 0 { print(String(format: "        ruga %d: crest %.2f mm proud, sides %.2f and %.2f", k, proud[0], proud[1], proud[2])) }
        expect(proud[0] > 0.7, "ruga \(k) crest only \(proud[0]) mm proud")
        expect(proud[0] - max(proud[1], proud[2]) > 0.3, "ruga \(k) is not a ridge: \(proud)")
    }
}

test("the incisive papilla sits on the midline 12.6 mm behind the incisors' labial face") {
    let labial: Float = placed[0].centre.y - placed[0].spec.depth / 2
    expect(abs(incisivePapillaZ - labial - incisivePapillaDistance) < 1e-4, "papilla at z \(incisivePapillaZ)")
    guard let top = palateSurface(0, incisivePapillaZ), let roof = vault.roof(0, incisivePapillaZ)
    else { expect(false, "no palate at the papilla"); return }
    // It stands proud of the vault's own surface there (towards the tongue: +y here).
    print(String(format: "        papilla crest %.2f, the vault beneath it %.2f (arch frame)", top, roof))
    expect(top - roof > 0.6, "the papilla does not stand out: \(top) vs \(roof)")
}

section("colour and reflectance: step 20's measurements")

test("CIELAB conversion: white is white, mid-grey is 18.4%, and it round-trips") {
    let white: SIMD3<Float> = labToLinearSRGB(SIMD3(100, 0, 0))
    expect(simd_distance(white, SIMD3<Float>(1, 1, 1)) < 0.002, "L*100 → \(white)")
    let grey: SIMD3<Float> = labToLinearSRGB(SIMD3(50, 0, 0))
    expect(abs(grey.y - 0.1842) < 0.001, "L*50 → \(grey)")
    let back: SIMD3<Double> = linearSRGBToLab(labToLinearSRGB(gingivaLab))
    expect(simd_distance(back, gingivaLab) < 0.05, "gingiva round trip \(back)")
}

test("the colours are step 20's measured CIELAB: the same maxillary teeth, gingiva and incisal enamel") {
    expect(simd_distance(middleThirdLab[.central]!, SIMD3(73.0, -0.5, 14.5)) < 1e-9, "central")
    expect(simd_distance(middleThirdLab[.lateral]!, SIMD3(70.0, 0.2, 18.6)) < 1e-9, "lateral")
    expect(simd_distance(middleThirdLab[.canine]!, SIMD3(65.1, 1.4, 23.6)) < 1e-9, "canine")
    expect(simd_distance(middleThirdLab[.posterior]!, SIMD3(70.0, 0.2, 18.6)) < 1e-9, "posterior (MODEL)")
    // Step 20 took the mean of these two for its lower incisors.
    let step20Incisor = SIMD3<Double>(71.5, -0.15, 16.5)
    let mean: SIMD3<Double> = (middleThirdLab[.central]! + middleThirdLab[.lateral]!) / 2
    expect(simd_distance(mean, step20Incisor) < 0.051, "mean of central and lateral \(mean)")
    expect(simd_distance(gingivaLab, SIMD3(52.9, 23.3, 14.9)) < 1e-9, "gingiva, Ho et al. 2015")
    expect(simd_distance(incisalLab, SIMD3(73.5, 2.2, 11.9)) < 1e-9, "incisal, Wee et al. 2022")
    expect(simd_distance(cervicalShift, SIMD3(-4.0, 1.5, 5.0)) < 1e-9, "cervical shift")
    // Each tooth is handed the right row: centrals, laterals, canines, then the rest.
    let shades: [Float] = gpuTeeth(placed).map { $0.extra.y }
    expectEqual(Array(shades[0..<8]), [0, 1, 2, 3, 3, 3, 3, 3])
    expectEqual(Array(shades[8..<16]), [0, 1, 2, 3, 3, 3, 3, 3])
}

test("Fresnel from refractive indices: dry enamel 5.6%, saliva 2.0%, enamel under saliva 0.94%") {
    expect(abs(dryEnamelF0 - 0.0560) < 0.0002, "\(dryEnamelF0)")
    expect(abs(filmF0 - 0.0204) < 0.0002, "\(filmF0)")
    expect(abs(enamelUnderFilmF0 - 0.0094) < 0.0002, "\(enamelUnderFilmF0)")
}

section("the distance function is a distance")

/// The worst over-report per material among pairs of nearby points outside
/// every surface. A ray only ever asks from outside, so only those count.
func worstOverReport(_ centre: SIMD3<Float>, _ half: SIMD3<Float>, _ count: Int) -> [Float]? {
    var rng = SystemRandomNumberGenerator()
    var a: [SIMD3<Float>] = []
    var b: [SIMD3<Float>] = []
    for _ in 0..<count {
        let p = SIMD3<Float>(centre.x + Float.random(in: -half.x...half.x, using: &rng),
                             centre.y + Float.random(in: -half.y...half.y, using: &rng),
                             centre.z + Float.random(in: -half.z...half.z, using: &rng))
        let dir: SIMD3<Float> = simd_normalize(SIMD3<Float>(Float.random(in: -1...1, using: &rng),
                                                            Float.random(in: -1...1, using: &rng),
                                                            Float.random(in: -1...1, using: &rng)))
        a.append(p)
        b.append(p + dir * 0.05)
    }
    guard let da = probe(a), let db = probe(b) else { return nil }
    var worst: [Float] = [0, 0, 0, 0]
    for i in 0..<a.count where da[i].y == db[i].y && da[i].x > 0 && db[i].x > 0 {
        let m: Int = Int(da[i].y)
        worst[m] = max(worst[m], abs(da[i].x - db[i].x) / 0.05)
    }
    return worst
}

test("outside every surface — teeth, gum, palate — it never claims more room than the ray allows") {
    // The whole scene, world frame: the crowns hang below y = 0, the gum and
    // palate rise above.
    guard let worst = worstOverReport(SIMD3<Float>(0, 12, 26), SIMD3<Float>(38, 20, 38), 200_000)
    else { expect(false, "probe failed: \(box.error)"); return }
    print(String(format: "        worst over-report outside: enamel %.2f, gum %.2f, palate %.2f; the ray allows %.2f",
                 worst[1], worst[2], worst[3], 1 / stepScale))
    for m in 1...3 { expect(worst[m] * stepScale <= 1.0, "material \(m) oversteps: \(worst[m])") }
}

test("and close in, round the rugae and papilla, the sheared molars and the tuberosities") {
    let roof: Float = vault.roof(0, 16) ?? vault.rimY
    let regions: [(SIMD3<Float>, SIMD3<Float>)] = [
        (w(0, roof, 16), SIMD3<Float>(14, 3, 10)),                       // rugae and papilla
        (w(placed[5].centre.x, -4, placed[5].centre.y), SIMD3<Float>(8, 6, 8)),   // #14, the most sheared crown
        (w(placed[7].centre.x, -8, placed[7].centre.y + 8), SIMD3<Float>(10, 12, 12)), // behind #16
        (w(placed[15].centre.x, -8, placed[15].centre.y + 8), SIMD3<Float>(10, 12, 12)), // behind #1
    ]
    for (c, h) in regions {
        guard let worst = worstOverReport(c, h, 60_000) else { expect(false, "probe failed"); return }
        print(String(format: "        near (%.0f, %.0f, %.0f): enamel %.2f, gum %.2f, palate %.2f",
                     c.x, c.y, c.z, worst[1], worst[2], worst[3]))
        for m in 1...3 { expect(worst[m] * stepScale <= 1.0, "material \(m) oversteps near \(c): \(worst[m])") }
    }
}

section("the picture")

let render: (image: MouthImage, gpuSeconds: Double)? = {
    guard let r = box.renderer else { return nil }
    return try? r.render(width: 640, height: 360, samples: 2, camera: stillCamera)
}()

func lab(_ p: SIMD4<UInt8>) -> SIMD3<Double> {
    linearSRGBToLab(SIMD3<Float>(srgbByteToLinear(p.x), srgbByteToLinear(p.y), srgbByteToLinear(p.z)))
}

/// Mean CIELAB of the pixels whose centre saw enamel of these Universal
/// numbers in this band of crown height.
func meanLab(numbers: Set<Int>, t lo: Float, _ hi: Float) -> (lab: SIMD3<Double>, count: Int) {
    guard let img = render?.image else { return (.zero, 0) }
    var sum = SIMD3<Double>(0, 0, 0)
    var n: Int = 0
    for y in 0..<img.height {
        for x in 0..<img.width {
            let s: SIMD4<Float> = img.seen(x, y)
            guard s.x == 1, s.y >= lo, s.y < hi, numbers.contains(Int(s.w)) else { continue }
            sum += lab(img.rgba(x, y))
            n += 1
        }
    }
    return (n > 0 ? sum / Double(n) : .zero, n)
}

let everyTooth: Set<Int> = Set(1...16)

test("the frame holds teeth, gum and palate, and the corners are background") {
    guard let img = render?.image else { expect(false, "render failed: \(box.error)"); return }
    var counts: [Int] = [0, 0, 0, 0]
    var seenNumbers: Set<Int> = []
    for y in 0..<img.height {
        for x in 0..<img.width {
            let s: SIMD4<Float> = img.seen(x, y)
            counts[Int(s.x)] += 1
            if s.x == 1 { seenNumbers.insert(Int(s.w)) }
        }
    }
    let total: Double = Double(img.width * img.height)
    print(String(format: "        enamel %.0f%%, gum %.0f%%, palate %.0f%%, background %.0f%%; teeth in view: %@",
                 Double(counts[1]) / total * 100, Double(counts[2]) / total * 100,
                 Double(counts[3]) / total * 100, Double(counts[0]) / total * 100,
                 seenNumbers.sorted().map { "#\($0)" }.joined(separator: " ")))
    expect(Double(counts[1]) / total > 0.12, "enamel \(counts[1])")
    expect(Double(counts[2]) / total > 0.1, "gum \(counts[2])")
    expect(Double(counts[3]) / total > 0.08, "palate \(counts[3])")
    expect(seenNumbers.isSuperset(of: Set(3...14)), "the picture should show #3 to #14 at least: \(seenNumbers.sorted())")
    let corner: SIMD4<UInt8> = img.rgba(1, 1)
    expect(img.seen(1, 1).x == 0 && abs(Int(corner.x) - Int(backgroundDisplay * 255)) <= 1, "top-left corner \(corner)")
}

test("on the picture, #16's side is on the viewer's right: the still's camera stands off the patient's left") {
    guard let img = render?.image else { expect(false, "render failed"); return }
    var leftSide: [Float] = []
    var rightSide: [Float] = []
    for y in 0..<img.height {
        for x in 0..<img.width {
            let s: SIMD4<Float> = img.seen(x, y)
            guard s.x == 1 else { continue }
            if Int(s.w) >= 9 { leftSide.append(Float(x)) } else { rightSide.append(Float(x)) }
        }
    }
    let l: Float = leftSide.reduce(0, +) / Float(max(leftSide.count, 1))
    let r: Float = rightSide.reduce(0, +) / Float(max(rightSide.count, 1))
    print(String(format: "        patient's left teeth centred at x %.0f, right at %.0f", l, r))
    expect(l > r, "the patient's left (#9–#16) should be on the viewer's right")
}

test("realistic, not toothpaste-white: the middle third is ivory") {
    let m = meanLab(numbers: everyTooth, t: 0.35, 0.65)
    print(String(format: "        middle third renders at L* %.1f a* %.1f b* %.1f over %d pixels", m.lab.x, m.lab.y, m.lab.z, m.count))
    expect(m.count > 500)
    expect(m.lab.z > 6, "b* \(m.lab.z): white, not ivory")
    expect(m.lab.x < 94, "L* \(m.lab.x)")
}

test("on the picture, the neck is yellower than the middle of the crown (b*/L*, as step 20 learned)") {
    let neck = meanLab(numbers: everyTooth, t: 0.0, 0.25)
    let mid = meanLab(numbers: everyTooth, t: 0.45, 0.7)
    let yNeck: Double = neck.lab.z / neck.lab.x
    let yMid: Double = mid.lab.z / mid.lab.x
    print(String(format: "        neck b*/L* %.3f (%d px), middle %.3f (%d px)", yNeck, neck.count, yMid, mid.count))
    expect(neck.count > 100 && mid.count > 100)
    expect(yNeck > yMid * 1.08, "neck \(yNeck) vs middle \(yMid)")
}

test("on the picture, the canines are yellower than the centrals, as measured") {
    let canines = meanLab(numbers: [6, 11], t: 0.3, 0.7)
    let centrals = meanLab(numbers: [8, 9], t: 0.3, 0.7)
    let yc: Double = canines.lab.z / canines.lab.x
    let yi: Double = centrals.lab.z / centrals.lab.x
    print(String(format: "        canines b*/L* %.3f (%d px), centrals %.3f (%d px)", yc, canines.count, yi, centrals.count))
    expect(canines.count > 100 && centrals.count > 100)
    expect(yc > yi * 1.15, "canines \(yc) vs centrals \(yi)")
}

test("translucency, and only translucency, greys the thin edge: compared with the same scene opaque") {
    guard let img = render?.image, let flatR = box.opaque,
          let flat = try? flatR.render(width: 640, height: 360, samples: 2, camera: stillCamera).image
    else { expect(false, "render failed"); return }
    var withT = SIMD3<Double>(0, 0, 0)
    var without = SIMD3<Double>(0, 0, 0)
    var n: Int = 0
    for y in 0..<img.height {
        for x in 0..<img.width {
            let s: SIMD4<Float> = img.seen(x, y)
            guard s.x == 1, s.y >= 0.9, s.z <= 1 else { continue }
            withT += lab(img.rgba(x, y))
            without += lab(flat.rgba(x, y))
            n += 1
        }
    }
    guard n > 50 else { expect(false, "only \(n) incisal pixels"); return }
    withT /= Double(n)
    without /= Double(n)
    print(String(format: "        thin edge with translucency b* %.1f; opaque b* %.1f (%d pixels)", withT.z, without.z, n))
    expect(withT.z < without.z - 2.0, "translucency should blue the edge: \(withT.z) vs \(without.z)")
}

finish()
