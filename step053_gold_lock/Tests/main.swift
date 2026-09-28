// Tests for step 53. The optical constants are checked against the papers,
// the conductor Fresnel equations against an independent derivation, the
// distance function against the definition of a distance, and the finished
// picture against the physics it claims — by reading colours and layer
// thicknesses back off the rendered pixels, not off the constants that were
// meant to produce them.
//
// LOCK_MUTANT=silver|dielectric|noNickel|thickGold breaks the step on purpose;
// `make mutants` requires the suite to fail for each.

import CoreGraphics
import Foundation
import Metal
import simd

let mutant: Mutant = activeMutant
if mutant != .none { print("MUTANT: \(mutant)") }

/// Holds the device so a failure to find one is a test failure, not a crash.
final class DeviceBox { let device = try? findDevice() }
let gpu = DeviceBox()

/// The textbook closed form for a conductor's reflectance, in real arithmetic
/// only: with s = sin θ, c = cos θ,
///     a² + b² = √((n² − k² − s²)² + 4n²k²),   a² = ½(that + n² − k² − s²)
///     Rs = (a² + b² − 2ac + c²) / (a² + b² + 2ac + c²)
///     Rp = Rs (a² + b² − 2as·t + s²t²) / (a² + b² + 2as·t + s²t²),  t = tan θ
/// A different derivation from the complex-number one the renderer uses, so
/// agreement between them is evidence, not an echo.
func closedFormReflectance(n: Double, k: Double, cosTheta c: Double) -> Double {
    let s2: Double = 1 - c * c
    let s: Double = s2.squareRoot()
    let d: Double = n * n - k * k - s2
    let root: Double = (d * d + 4 * n * n * k * k).squareRoot()
    let a2: Double = (root + d) / 2
    let a: Double = a2.squareRoot()
    let sumAB: Double = root
    let rs: Double = (sumAB - 2 * a * c + c * c) / (sumAB + 2 * a * c + c * c)
    if c < 1e-9 { return 1 }
    let t: Double = s / c
    let st: Double = s * t
    let rp: Double = rs * (sumAB - 2 * a * st + st * st) / (sumAB + 2 * a * st + st * st)
    return (rs + rp) / 2
}

// MARK: -

section("the optical constants, against Johnson & Christy")

test("the gold and silver rows are J&C's, in order of wavelength") {
    expectEqual(goldJC.count, 24)
    expectEqual(silverJC.count, 24)
    // Spot checks against the table (refractiveindex.info's copy of J&C 1972).
    let g = goldJC.first { abs($0.micrometres - 0.5486) < 1e-6 }
    expect(g != nil && g!.n == 0.43 && g!.k == 2.455, "gold at 548.6 nm: \(String(describing: g))")
    let s = silverJC.first { abs($0.micrometres - 0.6168) < 1e-6 }
    expect(s != nil && s!.n == 0.06 && s!.k == 4.152, "silver at 616.8 nm: \(String(describing: s))")
    for t in [goldJC, silverJC, nickelJC, brassQuerry] {
        for i in 1..<t.count { expect(t[i].micrometres > t[i - 1].micrometres, "rows out of order at \(i)") }
        expect(t[0].micrometres <= 0.380 && t[t.count - 1].micrometres >= 0.780, "table does not span 380–780 nm")
    }
    // Same photon energies in both, as in the paper.
    for i in 0..<goldJC.count { expect(goldJC[i].micrometres == silverJC[i].micrometres, "row \(i) energy differs") }
}

test("normal-incidence reflectance is ((n−1)² + k²) / ((n+1)² + k²), the value each row implies") {
    // Worked by hand from the rows: 450.9 nm (n 1.38, k 1.914) → 40.8%;
    // 548.6 nm (0.43, 2.455) → 78.7%; 659.5 nm (0.14, 3.697) → 96.3%.
    let cases: [(Double, Double)] = [(450.9, 0.4082), (548.6, 0.7869), (659.5, 0.9626)]
    for (nm, expected) in cases {
        let v = nk(goldJC, nanometres: nm)
        let r: Double = fresnelConductor(n: v.n, k: v.k, cosTheta: 1)
        print(String(format: "        gold at %.1f nm: R = %.4f (table implies %.4f)", nm, r, expected))
        expect(abs(r - expected) < 0.001, "R(\(nm)) = \(r), expected \(expected)")
    }
    // And a dielectric falls out as the k = 0 case: glass, n 1.5, 4.0%.
    expect(abs(fresnelConductor(n: 1.5, k: 0, cosTheta: 1, mutant: .none) - 0.04) < 1e-9)
}

test("the complex Fresnel equations agree with the real closed form at every angle") {
    var worst: Double = 0
    for row in goldJC + silverJC + nickelJC {
        for i in 0...90 {
            let theta: Double = Double(i) * 0.999 * Double.pi / 180
            let c: Double = cos(theta)
            let a: Double = fresnelConductor(n: row.n, k: row.k, cosTheta: c)
            let b: Double = closedFormReflectance(n: row.n, k: row.k, cosTheta: c)
            worst = max(worst, abs(a - b))
        }
    }
    print(String(format: "        worst disagreement over 69 rows × 91 angles: %.2e", worst))
    expect(worst < 1e-9, "the two derivations differ by \(worst)")
}

test("towards grazing incidence every metal's reflectance tends to 1") {
    for (name, t) in [("gold", goldJC), ("silver", silverJC)] {
        for nm in [450.0, 550.0, 650.0] {
            let v = nk(t, nanometres: nm)
            let r89: Double = fresnelConductor(n: v.n, k: v.k, cosTheta: cos(89.9 * Double.pi / 180))
            let r0: Double = fresnelConductor(n: v.n, k: v.k, cosTheta: 0)
            expect(r89 > 0.99 && abs(r0 - 1) < 1e-9, "\(name) at \(nm) nm: R(89.9°) = \(r89), R(90°) = \(r0)")
        }
    }
}

section("why gold is yellow, from the data")

test("gold reflects red far more than blue; silver reflects both") {
    func r(_ t: [NKRow], _ nm: Double) -> Double {
        let v = nk(t, nanometres: nm)
        return fresnelConductor(n: v.n, k: v.k, cosTheta: 1)
    }
    let gold: Double = r(goldJC, 650) / r(goldJC, 450)
    let silver: Double = r(silverJC, 650) / r(silverJC, 450)
    print(String(format: "        R(650)/R(450): gold %.2f, silver %.3f", gold, silver))
    expect(gold > 2.0, "gold ratio \(gold)")
    expect(silver < 1.05, "silver ratio \(silver)")
}

test("gold's interband edge is in the visible, near 2.4 eV; silver's is in the ultraviolet") {
    let au: Double = interbandOnsetEV(goldJC)
    let ag: Double = interbandOnsetEV(silverJC)
    print(String(format: "        steepest rise of ε₂ = 2nk: gold %.2f eV (%.0f nm), silver %.2f eV (%.0f nm)",
                 au, photonEV / au * 1000, ag, photonEV / ag * 1000))
    expect(au > 2.2 && au < 2.6, "gold edge \(au) eV")
    // The visible ends at ~380 nm = 3.26 eV.
    expect(ag > photonEV / 0.380, "silver edge \(ag) eV should be in the UV")
    // Romaniello & de Boeij's non-relativistic gold: onset ~3.5 eV → 354 nm, UV.
    expect(photonEV / 3.5 < 0.380, "without relativity gold's edge would be ultraviolet")
}

test("the colour pipeline: a perfect mirror is exactly white, and gold is yellow, silver nearly neutral") {
    let mirror: SIMD3<Double> = linearSRGB(xyz: xyz { _ in 1 }) / whiteThroughGrid
    expect(simd_distance(mirror, SIMD3<Double>(1, 1, 1)) < 1e-12, "mirror \(mirror)")
    let lab: SIMD3<Double> = linearSRGBToLab(mirror)
    expect(abs(lab.x - 100) < 0.01 && abs(lab.y) < 0.01 && abs(lab.z) < 0.01, "white Lab \(lab)")
    // The un-normalised sum is within 1% of white per channel: D65 at 10 nm.
    expect(simd_distance(whiteThroughGrid, SIMD3<Double>(1, 1, 1)) < 0.015, "grid white \(whiteThroughGrid)")
    let au: SIMD3<Double> = reflectanceRGB(goldJC, cosTheta: 1)
    let ag: SIMD3<Double> = reflectanceRGB(silverJC, cosTheta: 1)
    let auLab: SIMD3<Double> = linearSRGBToLab(au)
    let agLab: SIMD3<Double> = linearSRGBToLab(ag)
    print(String(format: "        gold F0 linear sRGB (%.3f, %.3f, %.3f), L*a*b* (%.1f, %.1f, %.1f)", au.x, au.y, au.z, auLab.x, auLab.y, auLab.z))
    print(String(format: "        silver F0 linear sRGB (%.3f, %.3f, %.3f), L*a*b* (%.1f, %.1f, %.1f)", ag.x, ag.y, ag.z, agLab.x, agLab.y, agLab.z))
    expect(au.x > au.y && au.y > au.z && au.x / au.z > 2.3, "gold should be red > green > blue: \(au)")
    expect(auLab.z > 30 && auLab.y > 0, "gold should be strongly yellow, a little red: \(auLab)")
    expect(abs(agLab.y) < 3 && abs(agLab.z) < 5, "silver should be close to neutral: \(agLab)")
    // At grazing incidence gold's colour washes out towards white.
    let grazing: SIMD3<Double> = reflectanceRGB(goldJC, cosTheta: 0.02)
    expect(linearSRGBToLab(grazing).z < auLab.z / 2, "grazing gold should be paler: \(grazing)")
}

test("the kernel's 129-entry table interpolates to within 0.1% of the direct spectral integral") {
    let table: [SIMD3<Float>] = fresnelTable(.none)
    var worst: Double = 0
    for i in 0..<500 {
        let c: Double = (Double(i) + 0.37) / 500
        let x: Double = c * Double(fresnelTableSize - 1)
        let j: Int = min(Int(x), fresnelTableSize - 2)
        let f: Double = x - Double(j)
        let a = SIMD3<Double>(Double(table[j].x), Double(table[j].y), Double(table[j].z))
        let b = SIMD3<Double>(Double(table[j + 1].x), Double(table[j + 1].y), Double(table[j + 1].z))
        let lerp: SIMD3<Double> = a + (b - a) * f
        let direct: SIMD3<Double> = reflectanceRGB(goldJC, cosTheta: c, mutant: .none)
        worst = max(worst, simd_abs(lerp - direct).max())
    }
    expect(worst < 0.001, "table error \(worst)")
}

test("the gold is optically bulk: tens of skin depths thick, so its n,k apply and the nickel cannot show") {
    var deepest: Double = 0
    for nm in spectralGrid { deepest = max(deepest, skinDepthNM(goldJC, nanometres: nm)) }
    let thickness: Double = Double(goldMicrometres) * 1000
    let roundTrip: Double = exp(-2 * thickness / deepest)
    print(String(format: "        deepest 1/e intensity depth %.1f nm (at 380 nm); gold %.0f nm = %.0f of them; round trip to the nickel %.1e",
                 deepest, thickness, thickness / deepest, roundTrip))
    expect(thickness / deepest > 10, "gold is only \(thickness / deepest) skin depths")
    expect(roundTrip < 1e-6)
}

section("the lock and its plating, against their sources")

test("the padlock is a 40 mm brass padlock: Master Lock 140D's published numbers, and the shackle built from them") {
    expectEqual(bodyWidth, 40)
    expectEqual(shackleDiameter, 6)
    expectEqual(shackleClearHeight, 22)
    expectEqual(shackleInsideWidth, 21)
    // The shape the kernel draws reproduces them: the clear span between legs,
    // and the clear height from the body's top to the inside of the bow.
    let span: Float = 2 * shackleLegOffset - shackleDiameter
    let clear: Float = shackleBowCentre + shackleLegOffset - shackleRadius - bodyHeight
    expect(abs(span - 21) < 1e-4 && abs(clear - 22) < 1e-4, "span \(span), clear height \(clear)")
    expect(2 * shackleLegOffset + shackleDiameter < bodyWidth, "the shackle must fit on the body")
}

test("the plating is gold on nickel on brass, at the cited thicknesses") {
    let stack: [Layer] = platingStack(mutant)
    expectEqual(stack.map { $0.name }, ["gold", "nickel", "brass"])
    expect(stack[0].micrometres == goldMicrometres, "gold is \(stack[0].micrometres) µm, cited \(goldMicrometres)")
    expect(goldMicrometres >= ftcGoldElectroplateMinimum && goldMicrometres < ftcHeavyGoldElectroplate,
           "gold must be 'gold electroplate' under 16 CFR 23.3: 0.175 µm up to heavy's 2.5 µm")
    expect(stack.count > 1 && stack[1].micrometres == nickelMicrometres)
}

test("the gold is thinner than a cotton fibre is wide — by thirty times") {
    let gold: Float = platingStack(mutant)[0].micrometres
    print(String(format: "        gold %.2f µm, cotton fibre %.0f µm: %.0f×", gold, cottonFibreMicrometres, cottonFibreMicrometres / gold))
    expect(gold * 10 < cottonFibreMicrometres, "gold \(gold) µm is not much thinner than a \(cottonFibreMicrometres) µm fibre")
}

section("the distance function is a distance")

/// Pairs of nearby points, each outside every surface, and the worst ratio of
/// the change in distance to the distance moved, by material.
func overReport(_ a: [SIMD3<Float>], _ step: Float, _ dev: MTLDevice) -> [Float]? {
    var rng = SystemRandomNumberGenerator()
    let b: [SIMD3<Float>] = a.map { p in
        let dir: SIMD3<Float> = simd_normalize(SIMD3<Float>(Float.random(in: -1...1, using: &rng),
                                                            Float.random(in: -1...1, using: &rng),
                                                            Float.random(in: -1...1, using: &rng)))
        return p + dir * step
    }
    guard let da = try? probeScene(a, on: dev), let db = try? probeScene(b, on: dev) else { return nil }
    var worst: [Float] = [0, 0, 0, 0]
    for i in 0..<a.count where da[i].y == db[i].y && da[i].x > 0 && db[i].x > 0 {
        let m: Int = Int(da[i].y)
        worst[m] = max(worst[m], abs(da[i].x - db[i].x) / step)
    }
    return worst
}

test("outside every surface — body, keyhole, shackle, table — no distance claims more room than the ray allows") {
    guard let dev = gpu.device else { expect(false, "no GPU"); return }
    var rng = SystemRandomNumberGenerator()
    var pts: [SIMD3<Float>] = []
    // The whole lock and the table round it.
    for _ in 0..<120_000 {
        let q = SIMD3<Float>(Float.random(in: -30...30, using: &rng), Float.random(in: -10...(lockOverallHeight + 8), using: &rng),
                             Float.random(in: 0...(bodyThickness + 8), using: &rng))
        pts.append(lockToWorld(q))
    }
    // Close round the keyway, the ring round the plug, and the shackle's joins.
    for _ in 0..<80_000 {
        let q = SIMD3<Float>(Float.random(in: -7...7, using: &rng), Float.random(in: -1.5...(keywayDepth + 1), using: &rng),
                             bodyThickness / 2 + Float.random(in: -7...7, using: &rng))
        pts.append(lockToWorld(q))
    }
    for _ in 0..<60_000 {
        let q = SIMD3<Float>(Float.random(in: -18...18, using: &rng), Float.random(in: (bodyHeight - 3)...(bodyHeight + 30), using: &rng),
                             bodyThickness / 2 + Float.random(in: -5...5, using: &rng))
        pts.append(lockToWorld(q))
    }
    guard let fine = overReport(pts, 0.02, dev), let coarse = overReport(pts, 0.3, dev) else { expect(false, "probe failed"); return }
    print(String(format: "        worst over-report outside: body %.3f, shackle %.3f, table %.3f (steps of 0.3 mm: %.3f, %.3f); the ray allows %.3f",
                 fine[1], fine[2], fine[3], coarse[1], coarse[2], 1 / stepScale))
    for m in 1...3 {
        expect(fine[m] * stepScale <= 1.0, "material \(m) oversteps: \(fine[m])")
        expect(coarse[m] * stepScale <= 1.0, "material \(m) oversteps at 0.3 mm: \(coarse[m])")
    }
}

test("exact where it should be, and the keyhole is a hole with a floor") {
    guard let dev = gpu.device else { expect(false, "no GPU"); return }
    let face: SIMD3<Float> = lockToWorld(SIMD3<Float>(0, bodyHeight / 2, bodyThickness))
    let bowTop: SIMD3<Float> = lockToWorld(SIMD3<Float>(0, shackleBowCentre + shackleLegOffset + shackleRadius, bodyThickness / 2))
    let t: Float = bodyThickness / 2
    let pts: [SIMD3<Float>] = [
        face + SIMD3<Float>(0, 5, 0),                                   // 5 mm above the face
        bowTop + lockV * 4,                                             // 4 mm beyond the bow
        lockToWorld(SIMD3<Float>(0, 3.0, t)),                           // inside the keyway slot
        lockToWorld(SIMD3<Float>(0, keywayDepth + 2, t)),               // past the slot's floor: brass
        lockToWorld(SIMD3<Float>(0, bodyHeight / 2, t)),                // mid-body
        lockToWorld(SIMD3<Float>(-shackleLegOffset, bodyHeight + 5, t)),// in a shackle leg
        lockToWorld(SIMD3<Float>(0, bodyHeight + 5, t)),                // between the legs: air
        lockToWorld(SIMD3<Float>(plugDiameter / 2, 0.3, t)),            // in the ring round the plug
    ]
    guard let r = try? probeScene(pts, on: dev) else { expect(false, "probe failed"); return }
    expect(abs(r[0].x - 5) < 1e-3 && r[0].y == 1, "above the face: \(r[0])")
    expect(abs(r[1].x - 4) < 1e-3 && r[1].y == 2, "beyond the bow: \(r[1])")
    expect(r[2].x > 0, "the keyway should be empty: \(r[2])")
    expect(r[3].x < 0 && r[3].y == 1, "beyond the slot's floor should be solid body: \(r[3])")
    expect(r[4].x < 0 && r[4].y == 1, "mid-body: \(r[4])")
    expect(r[5].x < 0 && r[5].y == 2, "shackle leg: \(r[5])")
    expect(r[6].x > 0, "between the legs: \(r[6])")
    expect(r[7].x > 0, "the ring round the plug should be open: \(r[7])")
}

section("the picture")

let render: (image: LockImage, gpuSeconds: Double)? = {
    guard let dev = gpu.device else { return nil }
    return try? renderLock(width: 1920, height: 1080, samples: 2, mutant: mutant, on: dev)
}()

func pixelLinear(_ img: LockImage, _ x: Int, _ y: Int) -> SIMD3<Double> {
    let p: SIMD4<UInt8> = img.rgba(x, y)
    return SIMD3<Double>(srgbByteToLinear(p.x), srgbByteToLinear(p.y), srgbByteToLinear(p.z))
}

test("the frame holds the lock and the table; the lock mirrors a light and mirrors its surroundings") {
    guard let img = render?.image else { expect(false, "render failed"); return }
    var lock: Int = 0
    var table: Int = 0
    var lit: Int = 0
    var surroundings: Int = 0
    var litLum: Double = 0
    var restLum: Double = 0
    var lums: [Double] = []
    for y in 0..<img.height {
        for x in 0..<img.width {
            let s: SIMD4<Float> = img.seen(x, y)
            if s.x == 3 { table += 1 }
            guard s.x == 1 || s.x == 2 else { continue }
            lock += 1
            let c: SIMD3<Double> = pixelLinear(img, x, y)
            let lum: Double = 0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z
            lums.append(lum)
            if s.w == 1 || s.w == 2 { lit += 1; litLum += lum } else { restLum += lum }
            if s.w == 3 || s.w == 4 { surroundings += 1 }
        }
    }
    let total: Double = Double(img.width * img.height)
    litLum /= Double(max(lit, 1))
    restLum /= Double(max(lock - lit, 1))
    print(String(format: "        lock %.1f%% of the frame, table %.1f%%; %d lock pixels mirror a light (mean luminance %.3f, the rest of the lock %.3f); %d mirror the table or the lock",
                 Double(lock) / total * 100, Double(table) / total * 100, lit, litLum, restLum, surroundings))
    expect(Double(lock) / total > 0.08 && Double(lock) / total < 0.4, "lock covers \(Double(lock) / total)")
    expect(Double(table) / total > 0.5)
    // A clear highlight in a bright studio: a large patch mirroring a light,
    // half as bright again as the rest of the metal — which is itself bright,
    // since the lock sits in a light tent. (Step 53's first, dark studio asked
    // for 4×; in a tent the rest of the metal mirrors white too, so that
    // measured the darkness of the room, not the highlight.) And the form
    // reads: the black flags put dark lines on it, so the darkest twentieth of
    // the metal is under a quarter as bright as the highlight.
    lums.sort()
    let dark: Double = lums.isEmpty ? 1 : lums[lums.count / 20]
    print(String(format: "        the darkest twentieth of the metal is below luminance %.3f", dark))
    expect(lit > 20_000, "the highlight: only \(lit) pixels mirror a light")
    expect(litLum > 1.5 * restLum, "the highlight is not clear: \(litLum) against \(restLum)")
    expect(dark < 0.25 * litLum, "no dark lines to show the form: \(dark) against \(litLum)")
    expect(surroundings > 20_000, "the reflection: only \(surroundings) pixels mirror the table or the lock")
}

/// Pixels of the flat front face whose every sample mirrored the key softbox:
/// the rendered colour, and the colour the constants alone predict for it.
func faceSamples() -> [(rendered: SIMD3<Double>, predicted: SIMD3<Double>, linear: SIMD3<Double>)] {
    guard let img = render?.image else { return [] }
    var out: [(SIMD3<Double>, SIMD3<Double>, SIMD3<Double>)] = []
    for y in stride(from: 0, to: img.height, by: 2) {
        for x in stride(from: 0, to: img.width, by: 2) {
            let s: SIMD4<Float> = img.seen(x, y)
            let l: SIMD4<Float> = img.seenLight(x, y)
            guard s.x == 1, s.z == 1, s.w == 1, l.y > 0.9999, l.x > 0 else { continue }
            // Gold's reflectance at this angle, from J&C's n,k through the
            // conductor Fresnel equations and the CIE observer — always gold,
            // always the true formula, whatever the render was fed — times
            // the white light it mirrored, through the kernel's tone curve.
            let f: SIMD3<Double> = reflectanceRGB(goldJC, cosTheta: Double(s.y), mutant: .none)
            let display: SIMD3<Double> = toneMap(f * Double(l.x))
            let predicted = SIMD3<Double>(srgbToLinear(display.x), srgbToLinear(display.y), srgbToLinear(display.z))
            let lin: SIMD3<Double> = pixelLinear(img, x, y)
            out.append((linearSRGBToLab(lin), linearSRGBToLab(predicted), lin))
        }
    }
    return out
}

func srgbToLinear(_ c: Double) -> Double { c <= 0.04045 ? c / 12.92 : pow((c + 0.055) / 1.055, 2.4) }

let face = faceSamples()

test("white light off the gold renders the colour the constants predict, within ΔE*ab 2") {
    guard face.count > 2_000 else { expect(false, "only \(face.count) face pixels mirror the key"); return }
    var sum: Double = 0
    var worst: Double = 0
    var meanR = SIMD3<Double>(0, 0, 0)
    var meanP = SIMD3<Double>(0, 0, 0)
    for f in face {
        let d: Double = deltaE(f.rendered, f.predicted)
        sum += d
        worst = max(worst, d)
        meanR += f.rendered
        meanP += f.predicted
    }
    let n: Double = Double(face.count)
    meanR /= n
    meanP /= n
    print(String(format: "        %d face pixels: rendered L*a*b* (%.1f, %.1f, %.1f), predicted (%.1f, %.1f, %.1f); mean ΔE %.2f, worst %.2f",
                 face.count, meanR.x, meanR.y, meanR.z, meanP.x, meanP.y, meanP.z, sum / n, worst))
    expect(sum / n < 1.0, "mean ΔE \(sum / n)")
    expect(worst < 2.0, "worst ΔE \(worst)")
}

test("on the finished picture, the gold reflects red far more than blue") {
    guard face.count > 2_000 else { expect(false, "no face pixels"); return }
    var m = SIMD3<Double>(0, 0, 0)
    for f in face { m += f.linear }
    m /= Double(face.count)
    print(String(format: "        mean linear sRGB of the lit face (%.3f, %.3f, %.3f): red/blue %.2f", m.x, m.y, m.z, m.x / m.z))
    expect(m.x / m.z > 1.8, "red/blue \(m.x / m.z)")
}

section("the inset and the scale bars, read off the annotated picture")

let annotated: Bool = {
    guard let img = render?.image else { return false }
    annotate(img, mutant: mutant)
    return true
}()

/// Runs of rows down the inset's measuring column, classified by which true
/// layer colour each pixel is nearest (within ΔE 4), top to bottom.
func insetRuns() -> [(name: String, rows: Int)] {
    guard annotated, let img = render?.image else { return [] }
    let r: CGRect = insetRect(height: img.height)
    let x: Int = insetMeasureColumn(height: img.height)
    let names: [String] = ["resin", "gold", "nickel", "brass"]
    let labs: [SIMD3<Double>] = [linearSRGBToLab(resinLinear), linearSRGBToLab(sectionColour("gold")),
                                 linearSRGBToLab(sectionColour("nickel")), linearSRGBToLab(sectionColour("brass"))]
    var runs: [(String, Int)] = []
    for y in Int(r.minY.rounded(.up))..<Int(r.maxY) {
        let lab: SIMD3<Double> = linearSRGBToLab(pixelLinear(img, x, y))
        var best: Int = -1
        var bestD: Double = 4
        for (i, l) in labs.enumerated() where deltaE(lab, l) < bestD { best = i; bestD = deltaE(lab, l) }
        let name: String = best >= 0 ? names[best] : "other"
        if let last = runs.last, last.0 == name { runs[runs.count - 1].1 += 1 } else { runs.append((name, 1)) }
    }
    return runs.map { (name: $0.0, rows: $0.1) }
}

let runs = insetRuns()

test("the inset shows resin, then gold, then nickel, then brass, each as thick as cited, to the pixel") {
    guard let img = render?.image else { expect(false, "render failed"); return }
    let umPerPx: Float = insetMicrometresPerPixel(height: img.height)
    let significant: [(name: String, rows: Int)] = runs.filter { $0.name != "other" }
    print("        down the column: " + significant.map { "\($0.name) \($0.rows) px" }.joined(separator: ", ")
          + String(format: "  (%.4f µm per pixel)", umPerPx))
    expectEqual(significant.map { $0.name }, ["resin", "gold", "nickel", "brass"])
    let gold: Int = significant.first { $0.name == "gold" }?.rows ?? 0
    let nickel: Int = significant.first { $0.name == "nickel" }?.rows ?? 0
    let goldPx: Float = goldMicrometres / umPerPx
    let nickelPx: Float = nickelMicrometres / umPerPx
    expect(abs(Float(gold) - goldPx) <= 1.0, "gold is \(gold) px, the cited \(goldMicrometres) µm is \(goldPx) px")
    expect(abs(Float(nickel) - nickelPx) <= 1.0, "nickel is \(nickel) px, the cited \(nickelMicrometres) µm is \(nickelPx) px")
}

/// The length of the run of ink-dark pixels along a scale bar's middle row.
func paintedLength(_ bar: ScaleBar) -> Int {
    guard let img = render?.image else { return 0 }
    let y: Int = Int(bar.y + 1.5 * CGFloat(img.height) / 1080)
    var n: Int = 0
    for x in max(Int(bar.x) - 4, 0)..<min(Int(bar.x + bar.pixels) + 5, img.width) {
        let p: SIMD4<UInt8> = img.rgba(x, y)
        if Int(p.x) + Int(p.y) + Int(p.z) < 3 * 45 { n += 1 }
    }
    return n
}

test("each scale bar is true: 10 mm at the lock, 5 µm in the inset, as painted") {
    guard annotated, let img = render?.image else { expect(false, "render failed"); return }
    let h: Int = img.height
    // Main view: ten millimetres square to the camera at the lock, projected.
    let a: SIMD2<Float> = project(lockCentre, width: img.width, height: h)
    let b: SIMD2<Float> = project(lockCentre + camera.right * mainBarMillimetres, width: img.width, height: h)
    let projected: Float = simd_distance(a, b)
    let main: ScaleBar = mainScaleBar(height: h)
    let inset: ScaleBar = insetScaleBar(height: h)
    let insetExpected: Float = insetBarMicrometres / insetMicrometresPerPixel(height: h)
    let mainPainted: Int = paintedLength(main)
    let insetPainted: Int = paintedLength(inset)
    print(String(format: "        10 mm: projected %.1f px, bar %.1f px, painted %d px; 5 µm: %.1f px, bar %.1f px, painted %d px",
                 projected, Float(main.pixels), mainPainted, insetExpected, Float(inset.pixels), insetPainted))
    expect(abs(Float(main.pixels) - projected) < 0.5, "main bar \(main.pixels) vs projected \(projected)")
    expect(abs(Float(inset.pixels) - insetExpected) < 0.01)
    expect(abs(Float(mainPainted) - projected) <= 2, "main bar painted \(mainPainted) px")
    expect(abs(Float(insetPainted) - insetExpected) <= 2, "inset bar painted \(insetPainted) px")
}

test("the inset is on the picture: the cotton fibre's circle at 15 µm, and the ring on the lock's face") {
    guard annotated, let img = render?.image else { expect(false, "render failed"); return }
    let r: CGRect = insetRect(height: img.height)
    let pxPerUm: Float = 1 / insetMicrometresPerPixel(height: img.height)
    // The fibre circle's leftmost point, on the row through its centre.
    let layers: [DrawnLayer] = insetLayers(height: img.height, mutant: mutant)
    let radius: Float = cottonFibreMicrometres / 2 * pxPerUm
    let cy: Int = Int(Float(layers[0].bottom) - radius - 0.9 * pxPerUm)
    let cx: Float = Float(r.minX + 0.30 * r.width)
    var light: Int = 0
    for dx in -3...3 {
        let p: SIMD4<UInt8> = img.rgba(Int(cx - radius) + dx, cy)
        if p.x > 150 { light += 1 }
    }
    expect(light > 0, "no fibre outline at its left edge")
    let cp: SIMD2<Float> = project(lockToWorld(cutPointLocal), width: img.width, height: img.height)
    expect(cp.x > 0 && cp.x < Float(r.minX) && cp.y > 0 && cp.y < Float(img.height), "cut point off the main view: \(cp)")
    let s: SIMD4<Float> = img.seen(Int(cp.x), Int(cp.y))
    expect(s.x == 1, "the cut point should be on the body: \(s)")
}

test("the render took \(String(format: "%.2f", render?.gpuSeconds ?? -1)) s on the GPU") {
    expect(render != nil)
}

finish()
