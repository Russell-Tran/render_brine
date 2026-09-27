// Tests for step 20. The anatomy is checked against its tables, the distance
// function against the definition of a distance, and the finished picture
// against the colour science it claims — by reading colours back off the
// rendered pixels, not off the constants that were meant to produce them.
//
// MOUTH_MUTANT=flat|opaque breaks the shading on purpose; `make mutants`
// requires the suite to fail for each.

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

test("outside every surface, it never claims more room than there is, beyond what the ray allows") {
    guard let dev = device.device else { expect(false, "no GPU"); return }
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
    guard let da = try? probeScene(a, on: dev), let db = try? probeScene(b, on: dev) else {
        expect(false, "probe failed"); return
    }
    var worst: [Float] = [0, 0, 0, 0, 0]
    for i in 0..<a.count where da[i].y == db[i].y && da[i].x > 0 && db[i].x > 0 {
        let m: Int = Int(da[i].y)
        worst[m] = max(worst[m], abs(da[i].x - db[i].x) / 0.05)
    }
    print(String(format: "        worst over-report outside: enamel %.2f, gum %.2f, tongue %.2f, floor %.2f; the ray allows %.2f",
                 worst[1], worst[2], worst[3], worst[4], 1 / stepScale))
    for m in 1...4 { expect(worst[m] * stepScale <= 1.0, "material \(m) oversteps: \(worst[m])") }
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

section("the picture")

let render: (image: MouthImage, gpuSeconds: Double)? = {
    guard let dev = device.device else { return nil }
    return try? renderMouth(width: 640, height: 360, samples: 2, mutant: mutant, on: dev)
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
          let flat = try? renderMouth(width: 640, height: 360, samples: 2, mutant: .opaque, on: dev).image
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

finish()
