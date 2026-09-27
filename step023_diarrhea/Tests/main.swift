// Tests for step 23. The physiology is checked where it is decided — in the
// emitters and paths that place every particle — and the anatomy against the
// distance function the picture was drawn with, through the GPU probe. The
// finished frame is read back once, to check the picture agrees.
//
// GUT_MUTANT=tips|na-through-cells|lazy-colon breaks the physiology on
// purpose; `make mutants` requires the suite to fail for each.

import Foundation
import Metal
import simd

let mutant: Mutant = Mutant(rawValue: ProcessInfo.processInfo.environment["GUT_MUTANT"] ?? "none") ?? .none

/// Holds the renderer so a failure to find a GPU is a test failure, not a crash.
final class Box {
    let renderer: Renderer? = {
        guard let d = try? findDevice() else { return nil }
        return try? Renderer(device: d, layout: Layout(width: 640))
    }()
}
let gpu = Box()

func probe(_ pts: [SIMD4<Float>], _ t: Float) -> [SIMD2<Float>]? {
    guard let r = gpu.renderer else { return nil }
    return try? r.probe(pts, time: t)
}

section("the numbers, against their sources")

test("drawn particle sizes keep Shannon's order: Cl⁻ largest, then K⁺, Na⁺ smallest") {
    let cl: Float = radiusPm(.chloride) * panelMMPerPm
    let k: Float = radiusPm(.potassium) * panelMMPerPm
    let na: Float = radiusPm(.sodium) * panelMMPerPm
    expect(cl > k && k > na, "Cl \(cl) K \(k) Na \(na)")
    // And in proportion, not just in order.
    expect(abs(cl / na - 181.0 / 102.0) < 1e-4)
    expect(radiusPm(.water) == 140, "water is the 1.4 Å probe")
}

test("K⁺ is amber, not CPK purple, so it cannot be mistaken for Na⁺") {
    let na: SIMD4<Float> = speciesColour(.sodium)
    let k: SIMD4<Float> = speciesColour(.potassium)
    let d: Float = simd_distance(SIMD3(na.x, na.y, na.z), SIMD3(k.x, k.y, k.z))
    expect(d > 0.6, "Na⁺ and K⁺ colours only \(d) apart")
    expect(k.x > 0.8 && k.y > 0.5 && k.z < 0.3, "K⁺ should be amber: \(k)")
}

test("villus to crypt 3:1; ileum and colon inside the 3-6-9 limits, colon about twice as wide") {
    expect(abs(villusHeight / ilealCryptDepth - 3) < 1e-5)
    expect(2 * ileumRadius <= 30 && 2 * colonRadius <= 60)
    expect(colonRadius / ileumRadius > 1.8 && colonRadius / ileumRadius < 2.3)
}

section("the villi stop at the valve (read off the distance function)")

test("villi stand along the ileum, top and bottom, at every phase of the wave") {
    for panel in 0..<2 {
        for t: Float in [0, 2.4, 7.1] {
            var pts: [SIMD4<Float>] = []
            for x in villusXs() where x > gutStartX + 3 {
                let R: Float = ileumRadiusAt(x, panel: panel, t: t)
                for side: Float in [1, -1] {
                    pts.append(SIMD4(x, ileumAxisY + side * (R - villusHeight * 0.5), -0.3, Float(panel)))
                }
            }
            guard let r = probe(pts, t) else { expect(false, "probe failed"); return }
            let hits: Int = r.filter { $0.x < 0 && $0.y == 2 }.count
            expect(hits == pts.count, "panel \(panel) t \(t): \(hits) of \(pts.count) villus probes hit a villus")
        }
    }
}

test("no villus anywhere past the valve: 300,000 random points in the colon and the valve's lips") {
    var pts: [SIMD4<Float>] = []
    var rng = SystemRandomNumberGenerator()
    for i in 0..<300_000 {
        pts.append(SIMD4(Float.random(in: lipX...gutEndX, using: &rng),
                         Float.random(in: -32...26, using: &rng),
                         Float.random(in: -30...0, using: &rng), Float(i % 2)))
    }
    for t: Float in [0, 4.8] {
        guard let r = probe(pts, t) else { expect(false, "probe failed"); return }
        let villi: Int = r.filter { $0.x < 0 && $0.y == 2 }.count
        expect(villi == 0, "\(villi) points past the valve are inside a villus at t \(t)")
    }
}

test("the colon has crypts: open pits at the crypt sites, tissue between them") {
    var pts: [SIMD4<Float>] = []
    var between: [SIMD4<Float>] = []
    for k in 3..<40 {
        let x: Float = Float(k) * colonCryptPitch
        let xs: Float = x + colonCryptPitch * 0.5
        let R: Float = colonRadiusAt(x, panel: 0, t: 0)
        let Rs: Float = colonRadiusAt(xs, panel: 0, t: 0)
        pts.append(SIMD4(x, colonAxisY + R + colonCryptDepth * 0.5, -0.3, 0))
        between.append(SIMD4(xs, colonAxisY + Rs + colonCryptDepth * 0.5, -0.3, 0))
    }
    guard let a = probe(pts, 0), let b = probe(between, 0) else { expect(false, "probe failed"); return }
    expect(a.allSatisfy { $0.x > 0 }, "a crypt site is filled in")
    expect(b.allSatisfy { $0.x < 0 && $0.y == 3 }, "the lining between crypts is missing")
}

section("secretion: from the crypts, Cl⁻ through the cells, Na⁺ between them")

test("every secreted particle rises through a crypt, below the villus bases — none from a villus tip") {
    let es: [Emitter] = emitters(panel: 1, mutant: mutant).filter { $0.route == .crypt }
    expect(es.count >= 200, "only \(es.count) secretion emitters")
    let t: Float = 1.3
    var pts: [SIMD4<Float>] = []
    var deeper = true
    for e in es {
        let p: SIMD3<Float> = wallPath(e, age: cryptCrossing * 0.667, t: t, mutant: mutant)
        let R: Float = ileumRadiusAt(e.x, panel: 1, t: t)
        if abs(p.y - ileumAxisY) <= R { deeper = false }
        pts.append(SIMD4(p.x, p.y, -0.3, 1))
        // And the source is not on a villus: half a pitch from the nearest.
        let nearest: Float = villusXs().map { abs($0 - e.x) }.min() ?? 0
        expect(nearest > villusPitch * 0.4, "a secretion source sits \(nearest) mm from a villus")
    }
    expect(deeper, "secretion crosses the lining above the villus bases, not in a crypt")
    guard let r = probe(pts, t) else { expect(false, "probe failed"); return }
    let inTissue: Int = r.filter { $0.x < 0 }.count
    expect(inTissue == 0, "\(inTissue) of \(r.count) secretion paths are inside tissue where the crypt should be open")
}

test("villus tips are where the ileum absorbs: the absorbed water passes into a villus") {
    let es: [Emitter] = emitters(panel: 0, mutant: mutant).filter { $0.route == .villusAbsorb }
    let t: Float = 3.0
    let pts: [SIMD4<Float>] = es.map { e in
        let p: SIMD3<Float> = wallPath(e, age: 0.5, t: t, mutant: mutant)
        return SIMD4(p.x, p.y, -0.3, 0)
    }
    guard let r = probe(pts, t) else { expect(false, "probe failed"); return }
    expect(r.allSatisfy { $0.x < 0 && $0.y == 2 }, "absorbed water should be inside a villus just past its tip")
}

test("in the close-up, Cl⁻ crosses INSIDE the cells and Na⁺ only BETWEEN them") {
    let es: [Emitter] = insetEmitters(mutant: mutant)
    var pts: [SIMD4<Float>] = []
    var kinds: [(Species, Route)] = []
    for e in es where e.species != .water {
        for i in 0..<21 {
            let age: Float = Float(i) / 20
            let p: SIMD3<Float> = insetPath(e, age: age)
            guard p.y > insetBottom + 1, p.y < insetTop - 1 else { continue }
            pts.append(SIMD4(p.x, p.y, 0, 2))
            kinds.append((e.species, e.route))
        }
    }
    guard let r = probe(pts, 0) else { expect(false, "probe failed"); return }
    var clInside = 0, clTotal = 0, naOutside = 0, naTotal = 0
    for (i, kr) in kinds.enumerated() {
        if kr.0 == .chloride { clTotal += 1; if r[i].x < 0 { clInside += 1 } }
        if kr.0 == .sodium { naTotal += 1; if r[i].x > 0 { naOutside += 1 } }
    }
    expect(clTotal > 20 && clInside == clTotal, "Cl⁻ inside a cell at \(clInside) of \(clTotal) points")
    expect(naTotal > 20 && naOutside == naTotal, "Na⁺ outside every cell at \(naOutside) of \(naTotal) points")
    expect(es.filter { $0.species == .chloride }.allSatisfy { $0.route == .transcellular })
    expect(es.filter { $0.species == .sodium }.allSatisfy { $0.route == .paracellular })
}

test("order: every Cl⁻ is out before any Na⁺ passes a junction, and water crosses after both") {
    // The pulse phase at which each particle crosses the apical level.
    func crossing(_ e: Emitter) -> Float {
        var lo: Float = 0, hi: Float = 1
        for _ in 0..<40 {
            let mid: Float = (lo + hi) / 2
            if insetPath(e, age: mid).y < insetTop { lo = mid } else { hi = mid }
        }
        return e.phase + lo * e.life
    }
    let es: [Emitter] = insetEmitters(mutant: mutant)
    let cl: [Float] = es.filter { $0.species == .chloride }.map(crossing)
    let na: [Float] = es.filter { $0.species == .sodium }.map(crossing)
    let water: [Float] = es.filter { $0.species == .water }.map(crossing)
    expect(cl.max()! < na.min()!, "last Cl⁻ out at \(cl.max()!), first Na⁺ through at \(na.min()!)")
    expect(na.max()! < water.min()!, "last Na⁺ at \(na.max()!), first water at \(water.min()!)")
    // And in each crypt burst of the main panel, the same order.
    let b: [(Species, Float)] = burstOrder(panel: 1)
    expect(b[0].0 == .chloride && b[1].0 == .sodium && b[2].0 == .water && b[0].1 < b[1].1 && b[1].1 < b[2].1)
    // Everything crosses and is gone within its pulse, so pulses never mix.
    expect(es.allSatisfy { $0.phase + $0.life < 1 })
}

test("arrows are an honest count: every route shows its arrow for the same length of time") {
    let routes: [Route] = [.crypt, .villusAbsorb, .colonAbsorb]
    let durations: [Float] = routes.map { r in
        let w = arrowWindow(r)
        return (w.to - w.from) * w.life
    }
    expect(durations.allSatisfy { abs($0 - durations[0]) < 1e-5 }, "arrow durations \(durations)")
    for panel in 0..<2 {
        for e in emitters(panel: panel, mutant: mutant) where routes.contains(e.route) {
            expect(e.life == arrowWindow(e.route).life, "an emitter's life disagrees with its arrow window")
        }
    }
}

section("the stool is the stool table")

test("the cholera lumen holds Na⁺ : Cl⁻ : K⁺ : HCO₃⁻ in exactly 135 : 100 : 15 : 45") {
    var n: [Species: Int] = [:]
    for (s, c) in lumenPopulation(panel: 1) { n[s] = c }
    let na: Int = n[.sodium]!, cl: Int = n[.chloride]!, k: Int = n[.potassium]!, hco3: Int = n[.bicarbonate]!
    let s: StoolElectrolytes = choleraStool
    expect(na * s.chloride == cl * s.sodium && k * s.chloride == cl * s.potassium
           && hco3 * s.chloride == cl * s.bicarbonate, "counts \(na) \(cl) \(k) \(hco3)")
    // K⁺ is present but sparse.
    expect(k > 0 && k * 5 < na, "K⁺ \(k) against Na⁺ \(na)")
}

test("the particle mix passes the osmotic-gap test for SECRETORY diarrhea (< 50 mOsm/kg)") {
    var n: [Species: Float] = [:]
    for (s, c) in lumenPopulation(panel: 1) { n[s] = Float(c) }
    // Concentrations implied by the counts, scaled so Cl⁻ reads its measured 100.
    let perCl: Float = 100 / n[.chloride]!
    let na: Float = n[.sodium]! * perCl
    let k: Float = n[.potassium]! * perCl
    let gap: Float = osmoticGap(sodium: na, potassium: k)
    print(String(format: "        implied Na⁺ %.0f K⁺ %.0f → gap %.0f mOsm/kg", na, k, gap))
    expect(gap < secretoryGapLimit, "gap \(gap)")
    expect(abs(osmoticGap(sodium: 135, potassium: 15) - (-10)) < 1e-4)
    // Electroneutral to within the small unmeasured anion gap.
    let cations: Float = na + k
    let anions: Float = 100 + n[.bicarbonate]! * perCl
    expect(abs(cations - anions) <= 10, "cations \(cations) anions \(anions)")
}

section("water: in and out of the gut")

test("cholera pours more water in than comes back; normally more comes back than goes in") {
    let n = waterBudget(panel: 0, mutant: mutant)
    let d = waterBudget(panel: 1, mutant: mutant)
    print("        normal: \(n.into) in, \(n.out) back (colon \(n.colonOut)); cholera: \(d.into) in, \(d.out) back (colon \(d.colonOut))")
    expect(d.into > d.out, "cholera \(d.into) in vs \(d.out) out")
    expect(n.out > n.into, "normal \(n.out) out vs \(n.into) in")
}

test("the cholera colon is overwhelmed, not idle: it returns ~5/24 of the inflow, more than a normal colon") {
    let n = waterBudget(panel: 0, mutant: mutant)
    let d = waterBudget(panel: 1, mutant: mutant)
    let ratio: Float = Float(d.colonOut) / Float(d.into)
    let expected: Float = maximalColonAbsorbsLitres / choleraInflowLitres
    expect(d.colonOut > 0, "the cholera colon absorbs nothing")
    expect(abs(ratio - expected) < 0.02, "colon returns \(ratio) of the inflow, sources say \(expected)")
    // 5 L/day against 1.35 normally: running at its ceiling.
    expect(Float(d.colonOut) >= 3 * Float(n.colonOut), "cholera colon \(d.colonOut) vs normal \(n.colonOut)")
}

section("motion: a push, not a pinch, and a loop that only goes forward")

test("the relaxed segment is AHEAD of the ring, and the ring travels downstream") {
    for panel in 0..<2 {
        var dMin: Float = 0, dMax: Float = 0, gMin: Float = 1, gMax: Float = -1
        var d: Float = -wave(panel).wavelength / 2
        while d < wave(panel).wavelength / 2 {
            let g: Float = waveShape(d, panel: panel).total
            if g < gMin { gMin = g; dMin = d }
            if g > gMax { gMax = g; dMax = d }
            d += 0.1
        }
        expect(gMin < -0.25 && gMax > 0.08, "panel \(panel): squeeze \(gMin), relax \(gMax)")
        expect(dMax - dMin > 5 && dMax - dMin < 30, "panel \(panel): relaxation \(dMax - dMin) mm ahead of the ring")
        let a: Float = waveOffset(40, panel: panel, t: 1.0)
        let b: Float = waveOffset(40, panel: panel, t: 1.05)
        expect(b < a, "the ring should approach x = 40 from upstream")
    }
    expect(choleraWave.speed > normalWave.speed)
}

let frames: Int = 96

test("every lumen particle and stool lump moves forward, frame after frame, all loop") {
    var backwards = 0
    var checked = 0
    for f in 0..<frames {
        let t0: Float = frameTime(f, of: frames)
        let t1: Float = frameTime(f + 1, of: frames)
        for panel in 0..<2 {
            for (s, n) in lumenPopulation(panel: panel) {
                for i in 0..<n {
                    guard let a = lumenParticle(s, index: i, panel: panel, t: t0),
                          let b = lumenParticle(s, index: i, panel: panel, t: t1) else { continue }
                    let dx: Float = b.position.x - a.position.x
                    if abs(dx) > gutSpan / 2 || (panel == 0 && dx < -20) { continue }   // a wrap or a respawn
                    checked += 1
                    if dx <= 0 { backwards += 1 }
                }
            }
        }
        let l0: [SIMD4<Float>] = stoolLumps(at: t0).sorted { $0.x < $1.x }
        let l1: [SIMD4<Float>] = stoolLumps(at: t1).sorted { $0.x < $1.x }
        for j in 1..<(l0.count - 1) {
            checked += 1
            if l1[j].x <= l0[j].x && l1[j].x - l0[j].x > -lumpSpacing / 2 { backwards += 1 }
        }
    }
    expect(checked > 20_000, "only \(checked) steps checked")
    expect(backwards == 0, "\(backwards) steps went backwards")
}

test("the loop closes forward: frame N is frame 0, and time only increases") {
    for f in 0..<frames { expect(frameTime(f + 1, of: frames) > frameTime(f, of: frames)) }
    expect(abs(frameTime(frames, of: frames) - loopSeconds) < 1e-5)
    let a: [Particle] = particles(at: 0, mutant: mutant)
    let b: [Particle] = particles(at: loopSeconds, mutant: mutant)
    expectEqual(a.count, b.count)
    var worst: Float = 0
    for i in 0..<min(a.count, b.count) {
        worst = max(worst, simd_distance(a[i].position, b[i].position), abs(a[i].alpha - b[i].alpha) * 10)
    }
    expect(worst < 2e-3, "particles at T differ from t = 0 by \(worst)")
    let l0: [Float] = stoolLumps(at: 0).map { $0.x }.sorted()
    let l1: [Float] = stoolLumps(at: loopSeconds).map { $0.x }.sorted()
    expect(zip(l0, l1).allSatisfy { abs($0 - $1) < 1e-3 }, "stool at T \(l1) vs 0 \(l0)")
    for panel in 0..<2 {
        for x: Float in [-80, -40, 30, 70, 120] {
            let r0: Float = x < lipX ? ileumRadiusAt(x, panel: panel, t: 0) : colonRadiusAt(x, panel: panel, t: 0)
            let r1: Float = x < lipX ? ileumRadiusAt(x, panel: panel, t: loopSeconds) : colonRadiusAt(x, panel: panel, t: loopSeconds)
            expect(abs(r0 - r1) < 1e-3, "wall at x \(x) panel \(panel): \(r0) vs \(r1)")
        }
    }
}

section("the distance function is a distance")

test("outside every surface, it never claims more room than there is, beyond what the ray allows") {
    // Pairs of nearby points across the gut. A true distance changes by at
    // most the distance moved; the ray trusts only `stepScale` of each step,
    // so the measured over-report must stay under 1/stepScale. Only pairs
    // OUTSIDE the surfaces count, because a ray only ever asks from outside —
    // step 20 learned that interior points (the tongue's ellipsoid centre)
    // read singularities no ray reaches. And no slope is divided out here:
    // step 20's slope corrections switched off abruptly and made jumps of 60×.
    var rng = SystemRandomNumberGenerator()
    var worst: [Float] = [0, 0, 0, 0, 0, 0]
    let step: Float = 0.05
    for t: Float in [0.7, 3.9, 6.6] {
        var a: [SIMD4<Float>] = []
        var b: [SIMD4<Float>] = []
        for i in 0..<160_000 {
            let panel = Float(i % 2)
            let p = SIMD3<Float>(Float.random(in: gutStartX...gutEndX, using: &rng),
                                 Float.random(in: -34...28, using: &rng),
                                 Float.random(in: -30...3, using: &rng))
            let dir: SIMD3<Float> = simd_normalize(SIMD3(Float.random(in: -1...1, using: &rng),
                                                         Float.random(in: -1...1, using: &rng),
                                                         Float.random(in: -1...1, using: &rng)))
            let q: SIMD3<Float> = p + dir * step
            a.append(SIMD4(p, panel))
            b.append(SIMD4(q, panel))
        }
        guard let da = probe(a, t), let db = probe(b, t) else { expect(false, "probe failed"); return }
        for i in 0..<a.count where da[i].y == db[i].y && da[i].x > 0 && db[i].x > 0 {
            let m: Int = Int(da[i].y)
            worst[m] = max(worst[m], abs(da[i].x - db[i].x) / step)
        }
    }
    print(String(format: "        worst over-report outside: ileum %.2f, villi %.2f, colon %.2f, stool %.2f; the ray allows %.2f",
                 worst[1], worst[2], worst[3], worst[5], 1 / stepScale))
    for m in [1, 2, 3, 5] { expect(worst[m] * stepScale <= 1.0, "material \(m) oversteps: \(worst[m])") }
}

section("the picture")

test("the rendered frame agrees: villus pixels only upstream of the valve, both panels drawn") {
    guard let r = gpu.renderer else { expect(false, "no GPU"); return }
    guard (try? r.render(time: 2.0, samples: 1, mutant: mutant)) != nil else { expect(false, "render failed"); return }
    let f: Frame = r.frame
    var villusPixels = 0, strayVillus = 0, colonPixels = [0, 0], particlePixels = 0
    for y in 0..<f.layout.height {
        for x in 0..<f.layout.width {
            let s: SIMD4<Float> = f.seen(x, y)
            let mat: Int = Int(s.y) % 100
            if mat == 2 { villusPixels += 1; if s.z > lipX { strayVillus += 1 } }
            if mat == 3 && s.x >= 0 && s.x < 2 { colonPixels[Int(s.x)] += 1 }
            if s.w > 0 { particlePixels += 1 }
        }
    }
    print("        villus pixels \(villusPixels), past the valve \(strayVillus); colon pixels \(colonPixels); particle pixels \(particlePixels)")
    expect(villusPixels > 1000, "villi barely visible: \(villusPixels) pixels")
    expect(strayVillus == 0, "\(strayVillus) villus pixels past the valve")
    expect(colonPixels[0] > 5000 && colonPixels[1] > 5000)
    expect(particlePixels > 1000)
}

finish()
