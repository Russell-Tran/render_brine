// Tests for step 64: step 20's tests, kept where they still apply, and the
// wear — scored against Smith & Knight's Tooth Wear Index surface by surface,
// its facets held to being planes that face where the grinding comes from, its
// dentine to showing only where the enamel is worn through and flush with it,
// as attrition leaves it. Everything is read back from the kernel that drew
// the picture, not from the constants meant to produce it.
//
// MOUTH_MUTANT=flat|opaque|noDentin|roundFacets|scooped breaks it on purpose;
// `make mutants` requires the suite to fail for each.

import Foundation
import Metal
import simd

let mutant: Mutant = {
    switch ProcessInfo.processInfo.environment["MOUTH_MUTANT"] {
    case "flat": return .flat
    case "opaque": return .opaque
    case "noDentin": return .noDentin
    case "roundFacets": return .roundFacets
    case "scooped": return .scooped
    default: return .none
    }
}()

let device: MTLDeviceBox = MTLDeviceBox()

/// Holds the device so a failure to find one is a test failure, not a crash.
final class MTLDeviceBox {
    let device = try? findDevice()
}

section("the anatomy, against Wheeler's tables")

test("seven mandibular teeth per side, with the crown measurements in the table") {
    expectEqual(mandibularTeeth.count, 7)
    let heights: [Float] = mandibularTeeth.map { $0.crownHeight }
    let widths: [Float] = mandibularTeeth.map { $0.width }
    expectEqual(heights, [9.0, 9.5, 11.0, 8.5, 8.0, 7.5, 7.0])
    expectEqual(widths, [5.0, 5.5, 7.0, 7.0, 7.0, 11.0, 10.5])
    // The canine is the longest crown in the arch — it is what makes it a canine.
    expect(heights.max() == mandibularTeeth[2].crownHeight, "the canine should be the tallest crown")
}

test("every crown is narrower at the neck than at its contacts, in both directions") {
    for t in mandibularTeeth {
        expect(t.cervicalWidth < t.width, "\(t.name) mesiodistal")
        expect(t.cervicalDepth < t.depth, "\(t.name) labiolingual")
    }
}

test("the gum scallops more under the incisors than the molars, as the junction does") {
    expect(mandibularTeeth[0].cejCurve > mandibularTeeth[5].cejCurve)
}

section("the arch")

let placed: [PlacedTooth] = placeTeeth()

test("fourteen teeth, the two sides mirror images") {
    expectEqual(placed.count, 14)
    for i in 0..<7 {
        let r: PlacedTooth = placed[i]
        let l: PlacedTooth = placed[i + 7]
        expect(abs(r.centre.x + l.centre.x) < 1e-4 && abs(r.centre.y - l.centre.y) < 1e-4,
               "\(r.spec.name) is not mirrored: \(r.centre) vs \(l.centre)")
    }
}

test("neighbouring teeth touch: centres are half of each width apart, along the arch") {
    for side in 0..<2 {
        for i in 0..<6 {
            let a: PlacedTooth = placed[side * 7 + i]
            let b: PlacedTooth = placed[side * 7 + i + 1]
            let gap: Float = simd_distance(a.centre, b.centre)
            let expected: Float = (a.spec.width + b.spec.width) / 2
            // A chord is a touch shorter than the arc it spans.
            expect(gap <= expected + 1e-3 && gap > expected - 0.35,
                   "\(a.spec.name)–\(b.spec.name): \(gap) mm apart, arc says \(expected)")
        }
    }
    let centrals: Float = simd_distance(placed[0].centre, placed[7].centre)
    expect(abs(centrals - mandibularTeeth[0].width) < 0.05, "the two centrals are \(centrals) mm apart")
}

test("the arch lands at published widths: canines ~26 mm apart, first molars on target") {
    // The canine's cusp tip is its centre. Hawley's circle, whose radius came
    // from the table, puts it within a millimetre of the ~26 mm adult
    // mandibular intercanine width — without being told to.
    let canines: Float = placed[9].centre.x - placed[2].centre.x
    expect(canines > 24.0 && canines < 27.0, "intercanine width \(canines) mm")
    let molars: Float = placed[12].centre.x - placed[5].centre.x
    expect(abs(molars - firstMolarSpan) < 0.01, "first molars \(molars) mm apart")
}

test("every tooth faces out of the arch, away from the tongue") {
    for t in placed {
        let toCentre = SIMD2<Float>(0, 20) - t.centre
        expect(simd_dot(t.outward, toCentre) < 0, "\(t.spec.name) faces in")
        expect(abs(simd_dot(t.outward, t.tangent)) < 1e-4, "\(t.spec.name) outward is not perpendicular")
    }
}

test("the wisdom teeth, when asked for, come after the fourteen and change none of them") {
    // Step 21 turns them on; this still and step 22 leave them off, so the
    // fourteen must be the same teeth in the same places either way.
    let more: [PlacedTooth] = placeTeeth(thirdMolars: true)
    expectEqual(more.count, 16)
    for i in 0..<placed.count {
        expect(more[i].spec.name == placed[i].spec.name && more[i].centre == placed[i].centre
               && more[i].tangent == placed[i].tangent && more[i].outward == placed[i].outward,
               "tooth \(i) moved when the wisdom teeth were added")
    }
    let t17: PlacedTooth = more[tooth17Index]
    let t32: PlacedTooth = more[tooth32Index]
    expect(t17.spec.name == "third molar" && t17.side == -1, "index \(tooth17Index) should be #17, on −x")
    expect(t32.spec.name == "third molar" && t32.side == 1, "index \(tooth32Index) should be #32, on +x")
    expect(abs(t17.centre.x + t32.centre.x) < 1e-4 && abs(t17.centre.y - t32.centre.y) < 1e-4, "not mirrored")
    // Each touches its second molar, straight behind it on the arch's line.
    for (w, m) in [(t17, more[6]), (t32, more[13])] {
        let gap: Float = simd_distance(w.centre, m.centre)
        expect(abs(gap - (w.spec.width + m.spec.width) / 2) < 1e-3, "third molar is \(gap) mm from the second")
        expect(simd_dot(simd_normalize(w.centre - m.centre), m.tangent) > 0.99999, "not in line")
        expect(simd_dot(w.outward, SIMD2<Float>(0, 20) - w.centre) < 0, "third molar faces in")
    }
    expect(mandibularThirdMolar.cervicalWidth < mandibularThirdMolar.width
           && mandibularThirdMolar.cervicalDepth < mandibularThirdMolar.depth, "the neck should be narrower")
}

test("the kernel counts the teeth it is given: 14 by default, 16 with the wisdom teeth") {
    expect(kernelSource(mutant: .none).contains("constant uint TOOTH_COUNT = 14;"), "off")
    expect(kernelSource(mutant: .none, thirdMolars: true).contains("constant uint TOOTH_COUNT = 16;"), "on")
    // And the wisdom teeth really are in the scene: solid crowns, not the
    // tongue, the gum or air.
    guard let dev = device.device else { expect(false, "no GPU"); return }
    let more: [PlacedTooth] = placeTeeth(thirdMolars: true)
    let pts: [SIMD3<Float>] = [tooth17Index, tooth32Index].map {
        SIMD3<Float>(more[$0].centre.x, -3, more[$0].centre.y)
    }
    guard let on = try? probeScene(pts, thirdMolars: true, on: dev), let off = try? probeScene(pts, on: dev)
    else { expect(false, "probe failed"); return }
    for k in 0..<2 {
        expect(on[k].x < 0 && on[k].y == 1, "with them, wisdom tooth \(k) mid-crown is \(on[k])")
        expect(off[k].x > 0, "without them, wisdom tooth \(k)'s place should be empty: \(off[k])")
    }
}

section("colour and reflectance, derived rather than typed")

test("CIELAB conversion: white is white, mid-grey is 18.4% reflectance, and it round-trips") {
    let white: SIMD3<Float> = labToLinearSRGB(SIMD3(100, 0, 0))
    expect(simd_distance(white, SIMD3<Float>(1, 1, 1)) < 0.002, "L*100 → \(white)")
    let grey: SIMD3<Float> = labToLinearSRGB(SIMD3(50, 0, 0))
    expect(abs(grey.y - 0.1842) < 0.001, "L*50 → \(grey)")
    let back: SIMD3<Double> = linearSRGBToLab(labToLinearSRGB(gingivaLab))
    expect(simd_distance(back, gingivaLab) < 0.05, "gingiva round trip \(back)")
}

test("Fresnel from refractive indices: dry enamel 5.6%, saliva 2.0%, enamel under saliva 0.94%") {
    expect(abs(dryEnamelF0 - 0.0560) < 0.0002, "\(dryEnamelF0)")
    expect(abs(filmF0 - 0.0204) < 0.0002, "\(filmF0)")
    expect(abs(enamelUnderFilmF0 - 0.0094) < 0.0002, "\(enamelUnderFilmF0)")
    // A wet tooth reflects from its enamel a sixth of what a dry one does.
    expect(dryEnamelF0 / enamelUnderFilmF0 > 5.5)
}

test("the neck is yellower and redder than the middle third, and canines yellower than incisors") {
    let mid: SIMD3<Double> = middleThirdLab[.incisor]!
    let neck: SIMD3<Double> = mid + cervicalShift
    expect(neck.y > mid.y && neck.z > mid.z && neck.x < mid.x, "Hasegawa et al.'s direction")
    expect(middleThirdLab[.canine]!.z > middleThirdLab[.incisor]!.z)
    expect(middleThirdLab[.canine]!.x < middleThirdLab[.incisor]!.x)
}

section("the distance function is a distance")

test("outside every surface, it never claims more room than there is — worn as well as whole") {
    guard let dev = device.device else { expect(false, "no GPU"); return }
  for depth in [Float(0), finalWearDepth] {
    // Pairs of nearby points across the whole scene. A true distance function
    // changes by at most the distance moved; the ray trusts only `stepScale`
    // of each step, so the measured over-report must stay under 1/stepScale.
    //
    // Only pairs OUTSIDE the surfaces count, because a ray only ever asks from
    // outside. That exclusion is not a convenience: deep inside the tongue the
    // ellipsoid approximation has a singularity at the centre, reading 40×,
    // which no ray ever reaches. Outside, the first version read 3.0× on the
    // teeth and 3.4× on the gum — the streaks in the first renders.
    var rng = SystemRandomNumberGenerator()
    var a: [SIMD3<Float>] = []
    var b: [SIMD3<Float>] = []
    for _ in 0..<150_000 {
        let p = SIMD3<Float>(Float.random(in: -26...26, using: &rng),
                             Float.random(in: -22...4, using: &rng),
                             Float.random(in: -6...50, using: &rng))
        let dir: SIMD3<Float> = simd_normalize(SIMD3<Float>(Float.random(in: -1...1, using: &rng),
                                                            Float.random(in: -1...1, using: &rng),
                                                            Float.random(in: -1...1, using: &rng)))
        a.append(p)
        b.append(p + dir * 0.05)
    }
    guard let da = try? probeScene(a, depth: depth, mutant: mutant, on: dev),
          let db = try? probeScene(b, depth: depth, mutant: mutant, on: dev) else {
        expect(false, "probe failed"); return
    }
    var worst: [Float] = [0, 0, 0, 0, 0]
    for i in 0..<a.count where da[i].y == db[i].y && da[i].x > 0 && db[i].x > 0 {
        let m: Int = Int(da[i].y)
        worst[m] = max(worst[m], abs(da[i].x - db[i].x) / 0.05)
    }
    print(String(format: "        wear %.2f mm — worst over-report outside: enamel %.2f, gum %.2f, tongue %.2f, floor %.2f; the ray allows %.2f",
                 depth, worst[1], worst[2], worst[3], worst[4], 1 / stepScale))
    for m in 1...4 { expect(worst[m] * stepScale <= 1.0, "material \(m) oversteps at wear \(depth): \(worst[m])") }
  }
}

test("the teeth stand out of the gum, and the gum out of the jaw, where they should") {
    guard let dev = device.device else { expect(false, "no GPU"); return }
    let t: PlacedTooth = placed[3]       // the patient's left first premolar, #21
    let c = SIMD3<Float>(t.centre.x, 0, t.centre.y)
    let pts: [SIMD3<Float>] = [
        c + SIMD3<Float>(0, -3, 0),                                   // mid crown
        c + SIMD3<Float>(0, -t.spec.crownHeight - 3, 0),              // deep in the root
        c + SIMD3<Float>(0, -t.spec.crownHeight - 3, 0) + SIMD3<Float>(t.outward.x, 0, t.outward.y) * (t.spec.cervicalDepth / 2 + 1.0),
        c + SIMD3<Float>(0, 6, 0),                                    // air above the tooth
    ]
    guard let r = try? probeScene(pts, on: dev) else { expect(false, "probe failed"); return }
    expect(r[0].x < 0 && r[0].y == 1, "mid crown should be inside enamel: \(r[0])")
    expect(r[1].x < 0, "the root should be inside the scene: \(r[1])")
    expect(r[2].x < 0 && r[2].y == 2, "just outside the root should be gum: \(r[2])")
    expect(r[3].x > 0, "above the tooth should be air: \(r[3])")
}

section("the picture, unworn: step 20's colour science still holds")

// Step 20's arch, depth 0, through this step's kernel — the colour tests are
// about enamel as it grows, and the worn picture has lost the thin incisal
// edge they read (which a test below checks, too).
let render: (image: MouthImage, gpuSeconds: Double)? = {
    guard let dev = device.device else { return nil }
    return try? renderMouth(width: 640, height: 360, samples: 2, mutant: mutant, depth: 0, on: dev)
}()

/// Mean CIELAB of the rendered pixels whose centre saw enamel of these kinds
/// in this band of crown height.
func meanLab(kinds: Set<Int>, t lo: Float, _ hi: Float) -> (lab: SIMD3<Double>, count: Int) {
    guard let img = render?.image else { return (.zero, 0) }
    var sum = SIMD3<Double>(0, 0, 0)
    var n: Int = 0
    for y in 0..<img.height {
        for x in 0..<img.width {
            let s: SIMD4<Float> = img.seen(x, y)
            guard s.x == 1, s.y >= lo, s.y < hi, kinds.contains(Int(s.z)) else { continue }
            let p: SIMD4<UInt8> = img.rgba(x, y)
            let lin = SIMD3<Float>(srgbByteToLinear(p.x), srgbByteToLinear(p.y), srgbByteToLinear(p.z))
            sum += linearSRGBToLab(lin)
            n += 1
        }
    }
    return (n > 0 ? sum / Double(n) : .zero, n)
}

test("the frame holds teeth, gum and tongue, and the corners are background") {
    guard let img = render?.image else { expect(false, "render failed"); return }
    var counts: [Int] = [0, 0, 0, 0, 0]
    for y in 0..<img.height { for x in 0..<img.width { counts[Int(img.seen(x, y).x)] += 1 } }
    let total: Double = Double(img.width * img.height)
    let teeth: Double = Double(counts[1]) / total
    let gum: Double = Double(counts[2]) / total
    print("        enamel \(Int(teeth * 100))%, gum \(Int(gum * 100))%, tongue \(counts[3] * 100 / Int(total))%, background \(counts[0] * 100 / Int(total))%")
    expect(teeth > 0.15 && teeth < 0.6, "enamel covers \(teeth)")
    expect(gum > 0.1, "gum covers \(gum)")
    let corner: SIMD4<UInt8> = img.rgba(img.width - 2, 1)
    expect(img.seen(img.width - 2, 1).x == 0 && abs(Int(corner.x) - Int(backgroundDisplay * 255)) <= 1,
           "top-right corner should be background, got \(corner)")
}

test("realistic, not toothpaste-white: the middle third is ivory, not white") {
    let m = meanLab(kinds: [0, 1, 2, 3, 4, 5], t: 0.35, 0.65)
    print(String(format: "        middle third renders at L* %.1f a* %.1f b* %.1f over %d pixels", m.lab.x, m.lab.y, m.lab.z, m.count))
    expect(m.count > 500)
    expect(m.lab.z > 6, "b* \(m.lab.z): that is white, not ivory")
    expect(m.lab.x < 94, "L* \(m.lab.x)")
}

test("on the finished picture, the neck is yellower than the middle of the crown") {
    // Yellowness as b*/L*, not b* alone. b* falls as a surface darkens, and
    // the neck sits in the gum's shadow, so comparing raw b* counts shading as
    // colour — which is what the first version of this test did, and it read
    // the neck as barely yellower than the middle when it was not.
    let neck = meanLab(kinds: [0, 1, 2, 3, 4, 5], t: 0.0, 0.25)
    let mid = meanLab(kinds: [0, 1, 2, 3, 4, 5], t: 0.45, 0.7)
    let yNeck: Double = neck.lab.z / neck.lab.x
    let yMid: Double = mid.lab.z / mid.lab.x
    print(String(format: "        neck L* %.1f b* %.1f (b*/L* %.3f), middle L* %.1f b* %.1f (%.3f)",
                 neck.lab.x, neck.lab.z, yNeck, mid.lab.x, mid.lab.z, yMid))
    expect(neck.count > 100 && mid.count > 100)
    expect(yNeck > yMid * 1.08, "neck b*/L* \(yNeck) vs middle \(yMid)")
}

test("on the finished picture, the thin incisal edge is bluer and greyer than the body") {
    let edge = meanLab(kinds: [0, 1], t: 0.9, 1.01)
    let body = meanLab(kinds: [0, 1], t: 0.4, 0.7)
    print(String(format: "        incisal b* %.1f L* %.1f, body b* %.1f L* %.1f", edge.lab.z, edge.lab.x, body.lab.z, body.lab.x))
    expect(edge.count > 50 && body.count > 100)
    expect(edge.lab.z < body.lab.z - 2.0, "incisal b* \(edge.lab.z) vs body \(body.lab.z)")
}

test("translucency, and only translucency, greys the thin edge: compared with the same scene opaque") {
    // The incisal edge is already bluer than the body because its measured
    // colour is (Wee et al.'s b* 11.9 against the body's ~18) — so the test
    // above passes without any translucency at all, and the `opaque` mutant
    // walked straight past it. This one renders the scene a second time with
    // translucency switched off and requires the thin edge to be bluer in the
    // real render than in that one. Under the mutant both are opaque.
    guard let dev = device.device, let img = render?.image,
          let flat = try? renderMouth(width: 640, height: 360, samples: 2, mutant: .opaque, depth: 0, on: dev).image
    else { expect(false, "render failed"); return }
    var withT = SIMD3<Double>(0, 0, 0)
    var without = SIMD3<Double>(0, 0, 0)
    var n: Int = 0
    for y in 0..<img.height {
        for x in 0..<img.width {
            let s: SIMD4<Float> = img.seen(x, y)
            guard s.x == 1, s.y >= 0.9, s.z <= 1 else { continue }
            let a: SIMD4<UInt8> = img.rgba(x, y)
            let b: SIMD4<UInt8> = flat.rgba(x, y)
            withT += linearSRGBToLab(SIMD3<Float>(srgbByteToLinear(a.x), srgbByteToLinear(a.y), srgbByteToLinear(a.z)))
            without += linearSRGBToLab(SIMD3<Float>(srgbByteToLinear(b.x), srgbByteToLinear(b.y), srgbByteToLinear(b.z)))
            n += 1
        }
    }
    guard n > 50 else { expect(false, "only \(n) incisal pixels"); return }
    withT /= Double(n)
    without /= Double(n)
    print(String(format: "        thin edge with translucency L* %.1f b* %.1f; opaque L* %.1f b* %.1f (%d pixels)",
                 withT.x, withT.z, without.x, without.z, n))
    expect(withT.z < without.z - 2.0, "translucency should blue the edge: \(withT.z) vs \(without.z)")
}


section("the wear, as numbers")

test("the years come from the rate: 2.25 mm at 45 µm a year is 50 years, 1.5–3× ordinary chewing") {
    expect(abs(finalWearYears - finalWearDepth / bruxistWearRate) < 1e-4)
    expect(abs(finalWearYears - 50) < 0.01, "\(finalWearYears) years")
    // Korkut et al.'s untreated bruxists lose height faster than Lambrechts
    // measured on ordinary chewing contacts, molars and premolars both.
    expect(bruxistWearRate > normalMolarWearRate && bruxistWearRate > normalPremolarWearRate)
    expect(bruxistWearRate / normalMolarWearRate > 1.5 && bruxistWearRate / normalPremolarWearRate < 3.01)
}

test("the claimed depth is inside incisal stage 3: past substantial dentine loss, short of the pulp") {
    expect(finalWearDepth - incisalEnamel >= substantialDentineLoss, "only \(finalWearDepth - incisalEnamel) mm of dentine")
    expect(finalWearDepth < incisalEnamelToPulp - 1.0, "within a millimetre of the pulp")
    // Wall enamel thickens up the crown through Al-Zahawi's two points.
    expect(abs(wallEnamel(heightAboveJunction: 1, kind: .incisor) - facialEnamelAt1mm) < 1e-4)
    expect(abs(wallEnamel(heightAboveJunction: 5, kind: .incisor) - facialEnamelAt5mm) < 1e-4)
    expect(wallEnamel(heightAboveJunction: 0, kind: .incisor) == 0)
}

test("every facet leans outward, towards where the opposing teeth come from") {
    for t in placeTeeth() {
        let n: SIMD4<Float> = facetPlane(t, depth: 1)
        let out: Float = n.x * t.outward.x + n.z * t.outward.y
        expect(n.y > 0.9 && out > 0.1, "\(t.spec.name) facet normal \(n)")
        expect(abs(n.x * t.tangent.x + n.z * t.tangent.y) < 1e-4, "\(t.spec.name) leans along the arch")
    }
    // Where the grinding starts: on the outer half of every crown.
    for spec in mandibularTeeth {
        let a: SIMD3<Float> = facetAnchorLocal(spec)
        expect(a.y > 0, "\(spec.name) starts wearing on its inner half: \(a)")
    }
}

// The surveys: every tooth, dropped onto every 0.1 mm, unworn and worn.
let surveyStep: Float = 0.1
let unworn: [ToothSurvey]? = {
    guard let dev = device.device else { return nil }
    return try? surveyTeeth(depth: 0, step: surveyStep, mutant: mutant, on: dev)
}()
let worn: [ToothSurvey]? = {
    guard let dev = device.device else { return nil }
    return try? surveyTeeth(depth: finalWearDepth, step: surveyStep, mutant: mutant, on: dev)
}()

section("the wear, as drawn")

test("the CPU copy of step 20's crown top matches the kernel, so the anchors sit on the real surface") {
    guard let u = unworn else { expect(false, "survey failed"); return }
    var worst: Float = 0
    var n: Int = 0
    for s in u {
        for p in s.points where s.onOcclusalTable(p.uv) {
            guard let h = surfaceHeightCPU(s.spec, p.uv) else { continue }
            worst = max(worst, abs(h - p.height))
            n += 1
        }
    }
    print(String(format: "        %d points, worst disagreement %.4f mm", n, worst))
    expect(n > 10_000 && worst < 0.005, "worst \(worst) mm over \(n) points")
}

test("unworn, there is no facet and no dentine anywhere — dentine never shows through intact enamel") {
    guard let u = unworn else { expect(false, "survey failed"); return }
    let facets: Int = u.reduce(0) { $0 + $1.worn.count }
    expect(facets == 0, "\(facets) worn points on step 20's teeth")
}

test("every tooth is worn exactly the stated depth, straight down at its anchor") {
    guard let dev = device.device, let h0 = try? anchorHeights(depth: 0, mutant: mutant, on: dev),
          let h1 = try? anchorHeights(depth: finalWearDepth, mutant: mutant, on: dev) else {
        expect(false, "drop failed"); return
    }
    let placed: [PlacedTooth] = placeTeeth()
    for i in 0..<placed.count {
        let lost: Float = h0[i] - h1[i]
        expect(abs(lost - finalWearDepth) < 0.01, "\(placed[i].spec.name) \(i) lost \(lost) mm")
        expect(abs(h0[i] - facetAnchorLocal(placed[i].spec).z) < 0.005, "anchor off the surface by \(h0[i] - facetAnchorLocal(placed[i].spec).z)")
    }
}

test("the canine is ground 2.25 mm at its tip, and its crown stands at least 1.6 mm lower overall") {
    guard let u = unworn, let w = worn else { expect(false, "survey failed"); return }
    for k in [2, 9] {
        let before: Float = u[k].points.map { $0.height }.max() ?? 0
        let after: Float = w[k].points.map { $0.height }.max() ?? 0
        let tipBefore: Float = facetAnchorLocal(u[k].spec).z
        print(String(format: "        canine %d: highest point %.2f → %.2f mm (down %.2f); tip down %.2f at the anchor",
                     k, before, after, before - after, finalWearDepth))
        expect(abs(before - tipBefore) < 0.25, "the anchor should be at the canine's tip, got \(tipBefore) vs top \(before)")
        // The facet leans 20° labially, so its lingual edge — now the highest
        // point of the canine — stands higher than where the tip was cut: the
        // crown loses 2.25 mm at the tip (the anchor test) and 1.64 mm of its
        // overall height, which is what the label says.
        expect(abs((before - after) - canineHeightLoss) < 0.02, "the canine lost \(before - after) mm of height, label says \(canineHeightLoss)")
    }
}

test("every facet is a plane: worn points lie on one flat surface to within 10 µm") {
    guard let w = worn else { expect(false, "survey failed"); return }
    var worstRMS: Double = 0
    for s in w {
        guard let fit = s.facetFit() else { expect(false, "\(s.spec.name) \(s.index) has no facet"); continue }
        worstRMS = max(worstRMS, fit.rms)
        expect(fit.rms < 0.01 && fit.worst < 0.03,
               String(format: "%@ %d: rms %.4f, worst %.4f mm off its plane", s.spec.name, s.index, fit.rms, fit.worst))
    }
    print(String(format: "        worst rms off the plane: %.4f mm", worstRMS))
}

test("every facet faces as occlusal anatomy says: up, and outward by its stated lean") {
    guard let w = worn else { expect(false, "survey failed"); return }
    for s in w {
        guard let fit = s.facetFit() else { expect(false, "no facet"); continue }
        // y = c0 + c1 u + c2 v: the normal is (−c1, −c2, 1), normalised.
        let n: SIMD3<Double> = simd_normalize(SIMD3<Double>(-fit.c.y, -fit.c.z, 1))
        let lean: Double = acos(n.z) * 180 / Double.pi
        let stated: Double = Double(facetTiltDegrees(s.spec.kind))
        expect(abs(lean - stated) < 1.0, String(format: "%@: leans %.2f°, stated %.0f°", s.spec.name, lean, stated))
        expect(n.y > 0 && abs(n.x) < 0.02, String(format: "%@: faces (%.3f, %.3f) in (u, v)", s.spec.name, n.x, n.y))
    }
}

test("dentine shows only where the facet has cut through the enamel cap, and lies flush with it") {
    guard let w = worn else { expect(false, "survey failed"); return }
    var shown: Int = 0
    for s in w {
        guard let fit = s.facetFit() else { continue }
        for p in s.dentine {
            shown += 1
            let plane: Double = fit.c.x + fit.c.y * Double(p.uv.x) + fit.c.z * Double(p.uv.y)
            // Flush: on the facet's plane, not scooped below the enamel rim —
            // attrition's dentine "remains flat with no cupping" (Kaidonis).
            expect(abs(plane - Double(p.height)) < 0.01,
                   String(format: "%@: dentine %.3f mm off the facet at (%.1f, %.1f)", s.spec.name, plane - Double(p.height), p.uv.x, p.uv.y))
            // And only below the cap: at least the enamel's depth was ground off.
            let removed: Float = crownTopCPU(s.spec, p.uv) - p.height
            // (0.02: the kernel reads dentine 0.01 mm under the surface, along a
            // normal that leans up to 20°.)
            expect(removed >= topEnamel(s.spec.kind) - 0.02, "\(s.spec.name): dentine under \(removed) mm of wear")
        }
        // A rim of enamel stays round the dentine: "there is almost always a
        // rim of enamel at the worn surface margins" (Bardsley 2008).
        let rim: Int = s.worn.count - s.dentine.count
        expect(s.dentine.isEmpty || rim > 0, "\(s.spec.name) has no enamel rim")
    }
    print("        \(shown) points of exposed dentine")
    expect(shown > 500, "only \(shown) dentine points")
}

test("each surface scores the Tooth Wear Index stage this still claims") {
    guard let w = worn else { expect(false, "survey failed"); return }
    for s in w {
        let measured: Int = s.stage(depth: finalWearDepth)
        let claimed: Int = claimedStage[s.spec.kind]!
        if s.index < 7 {
            print(String(format: "        %-16@ %@ stage %d (claimed %d): dentine %.0f%% of the occlusal table, %.2f mm into it",
                         s.spec.name as NSString, twiSurface(s.spec.kind).rawValue, measured, claimed,
                         s.occlusalDentineFraction * 100, s.dentineLoss))
        }
        expect(measured == claimed, "\(s.spec.name) \(s.index): measured \(measured), claimed \(claimed)")
    }
}

test("wear only takes away: at every point, no tooth is ever higher after more years") {
    guard let dev = device.device else { expect(false, "no GPU"); return }
    var previous: [ToothSurvey]? = nil
    for depth in [Float(0), 0.4, 0.9, 1.5, 2.0, finalWearDepth] {
        guard let now = try? surveyTeeth(depth: depth, step: 0.25, mutant: mutant, on: dev) else {
            expect(false, "survey failed"); return
        }
        if let before = previous {
            for (a, b) in zip(before, now) {
                var heights: [SIMD2<Float>: Float] = [:]
                for p in a.points { heights[p.uv] = p.height }
                for p in b.points {
                    // 0.5 µm: the drop stops within 0.2 µm of a surface, so two drops onto the same unchanged surface can differ by that much.
                    if let h = heights[p.uv] { expect(p.height <= h + 5e-4, "\(a.spec.name) grew \(p.height - h) mm at wear \(depth)") }
                }
            }
        }
        previous = now
    }
}

section("the worn picture")

let wornRender: MouthImage? = {
    guard let dev = device.device else { return nil }
    return try? renderMouth(width: 640, height: 360, samples: 2, mutant: mutant, depth: finalWearDepth, on: dev).image
}()

func meanLab(_ img: MouthImage, _ want: (SIMD4<Float>) -> Bool) -> (lab: SIMD3<Double>, count: Int) {
    var sum = SIMD3<Double>(0, 0, 0)
    var n: Int = 0
    for y in 0..<img.height {
        for x in 0..<img.width where want(img.seen(x, y)) {
            let p: SIMD4<UInt8> = img.rgba(x, y)
            sum += linearSRGBToLab(SIMD3<Float>(srgbByteToLinear(p.x), srgbByteToLinear(p.y), srgbByteToLinear(p.z)))
            n += 1
        }
    }
    return (n > 0 ? sum / Double(n) : .zero, n)
}

test("on the picture, exposed dentine is darker, redder and yellower than the facet enamel around it") {
    guard let img = wornRender else { expect(false, "render failed"); return }
    let dentine = meanLab(img) { $0.x == 5 }
    let facet = meanLab(img) { $0.x == 6 }
    print(String(format: "        dentine L* %.1f a* %.1f b* %.1f (%d px); facet enamel L* %.1f a* %.1f b* %.1f (%d px)",
                 dentine.lab.x, dentine.lab.y, dentine.lab.z, dentine.count, facet.lab.x, facet.lab.y, facet.lab.z, facet.count))
    expect(dentine.count > 300 && facet.count > 300)
    let yd: Double = dentine.lab.z / dentine.lab.x
    let yf: Double = facet.lab.z / facet.lab.x
    expect(yd > yf * 1.25, String(format: "dentine b*/L* %.3f vs facet %.3f", yd, yf))
    expect(dentine.lab.y > facet.lab.y + 1.5, String(format: "dentine a* %.1f vs facet %.1f", dentine.lab.y, facet.lab.y))
}

test("worn, the thin translucent incisal edge is gone: no incisor shows its top tenth") {
    guard let img = wornRender else { expect(false, "render failed"); return }
    var edge: Int = 0
    for y in 0..<img.height { for x in 0..<img.width {
        let s: SIMD4<Float> = img.seen(x, y)
        if (s.x == 1 || s.x == 5 || s.x == 6) && s.z == 0 && s.y >= 0.9 { edge += 1 }
    } }
    expect(edge == 0, "\(edge) incisor pixels in the top tenth of the unworn crown")
}

test("facets are glossier than unworn enamel, dentine a little less so") {
    expect(facetRoughness < 0.3 && dentineRoughness < 0.3 && facetRoughness < dentineRoughness)
}

section("the labels and the inset")

test("each label's leader lands on a pixel showing what it names") {
    guard let w = worn, let img = wornRender, let t = labelTargets(w) else { expect(false, "no survey or render"); return }
    // (aux material: 5 dentine, 6 facet enamel; z is the crown kind.)
    let checks: [(String, SIMD3<Float>, (SIMD4<Float>) -> Bool)] = [
        ("exposed dentine", t.dentine, { $0.x == 5 }),
        ("wear facet", t.facet, { $0.x == 6 }),
        ("flattened canine", t.canine, { ($0.x == 5 || $0.x == 6) && $0.z == 1 }),
    ]
    for (name, p, ok) in checks {
        let q: SIMD2<Float> = stillCamera.project(p, width: img.width, height: img.height)
        let x: Int = Int(q.x)
        let y: Int = Int(q.y)
        guard x >= 1, y >= 1, x < img.width - 1, y < img.height - 1 else { expect(false, "\(name) is off the picture at \(q)"); continue }
        // The pixel under the dot, or one beside it: the dot is 7 px across.
        var hit: Bool = false
        for dy in -1...1 { for dx in -1...1 where ok(img.seen(x + dx, y + dy)) { hit = true } }
        expect(hit, "\(name) points at \(img.seen(x, y)) at \(q)")
    }
}

test("the inset's cut is to scale: the worn incisor is 2.25 mm shorter, dentine bare at the top, enamel either side") {
    guard let dev = device.device, let g = try? sampleSection(depth: finalWearDepth, perMillimetre: 20, on: dev) else {
        expect(false, "section failed"); return
    }
    // Down the column through the facet's anchor, where the wear is stated.
    let anchor: SIMD3<Float> = facetAnchorLocal(placeTeeth()[sectionToothIndex].spec)
    let col: Int = Int((anchor.y - sectionV.lowerBound) * 20)
    func topRow(worn: Bool) -> Int? { (0..<g.rows).first { g.at(col, $0, worn: worn) > 0 } }
    guard let now = topRow(worn: true), let was = topRow(worn: false) else { expect(false, "empty section"); return }
    let lost: Float = Float(now - was) / 20
    print(String(format: "        at the anchor the cut's top came down %.2f mm", lost))
    expect(abs(lost - finalWearDepth) < 0.1, "the cut lost \(lost) mm at the anchor")
    // Just under the facet's low (labial) edge: enamel, dentine, enamel across.
    let r: Int = now + 4
    let row: [Float] = (0..<g.columns).map { g.at($0, r, worn: true) }
    guard let first = row.firstIndex(where: { $0 > 0 }), let last = row.lastIndex(where: { $0 > 0 }) else { expect(false); return }
    expect(row[first] == 1 && row[last] == 1 && row[first...last].contains(2),
           "under the facet the cut reads \(row[first...last].map { Int($0) })")
    // And nowhere is there dentine outside the unworn outline, or tissue where there was none.
    for i in 0..<g.worn.count { expect(!(g.worn[i] > 0 && g.whole[i] == 0), "tissue appeared at \(i)") }
}

finish()
