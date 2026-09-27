// Tests for step 24. The physiology is checked where it is decided — in the
// emitters and lumen rules that place every particle — and the anatomy against
// the distance function the picture was drawn with, through the GPU probe. The
// finished frame is read back once, to check the picture agrees.
//
// GUT_MUTANT=lactase-working|chloride-on|low-gap breaks the physiology on
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

let frames: Int = 96
/// Every particle, at every one of `frames` frames of the loop, computed once.
let loopParticles: [[Particle]] = (0..<frames).map { particles(at: frameTime($0, of: frames), mutant: mutant) }

section("the numbers, against their sources")

test("drawn sizes keep their real order and the ions keep Shannon's proportions") {
    let cl: Float = drawnRadius(.chloride), k: Float = drawnRadius(.potassium), na: Float = drawnRadius(.sodium)
    expect(cl > k && k > na, "Cl \(cl) K \(k) Na \(na)")
    expect(abs(cl / na - 181.0 / 102.0) < 1e-4)
    // Among what is drawn: glucose > SCFA > water > K⁺ > Na⁺, as in life.
    let order: [Species] = [.glucose, .scfa, .water, .potassium, .sodium]
    for i in 1..<order.count {
        expect(drawnRadius(order[i - 1]) > drawnRadius(order[i]), "\(order[i - 1]) not larger than \(order[i])")
    }
    expect(drawnRadius(.glucose) == drawnRadius(.galactose))
}

test("the sugars wear step 17's colours, and galactose cannot be mistaken for chloride") {
    let g: SIMD4<Float> = speciesColour(.glucose), gal: SIMD4<Float> = speciesColour(.galactose)
    expect(simd_distance(SIMD3(g.x, g.y, g.z), SIMD3<Float>(0.96, 0.86, 0.42)) < 1e-5)
    expect(simd_distance(SIMD3(gal.x, gal.y, gal.z), SIMD3<Float>(0.62, 0.90, 0.52)) < 1e-5)
    let cl: SIMD4<Float> = speciesColour(.chloride)
    expect(simd_distance(SIMD3(cl.x, cl.y, cl.z), SIMD3(gal.x, gal.y, gal.z)) > 0.4)
}

test("two glasses: 24 g, the NIH consensus threshold for appreciable symptoms") {
    expect(lactoseGramsPerCup * Float(cups) == 24)
}

test("villus to crypt 3:1; ileum and colon inside the 3-6-9 limits, colon about twice as wide") {
    expect(abs(villusHeight / ilealCryptDepth - 3) < 1e-5)
    expect(2 * ileumRadius <= 30 && 2 * colonRadius <= 60)
    expect(colonRadius / ileumRadius > 1.8 && colonRadius / ileumRadius < 2.3)
}

section("the anatomy is step 23's (read off the distance function)")

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

section("lactase: the cut happens only where the enzyme is")

test("lactose is split only in the upper panel, at villus tips, and only at lactase cells in the close-up") {
    expect(lactasePresent(panel: 0, mutant: mutant) && !lactasePresent(panel: 1, mutant: mutant),
           "lactase present below")
    for panel in 0..<2 {
        let splits: [Emitter] = emitters(panel: panel, mutant: mutant).filter { $0.route == .villusSplit }
        if panel == 0 {
            expect(splits.count == 2 * upperSplitEvents, "upper panel has \(splits.count) split emitters")
        } else {
            expect(splits.isEmpty, "\(splits.count) lactose halves split in the lower panel, which has no lactase")
        }
        for e in splits {
            let nearest: Float = villusXs().map { abs($0 - e.x) }.min() ?? 99
            expect(nearest < 1e-4, "a split away from a villus tip")
        }
    }
    // The close-up: sugars only come apart over cells with lactase.
    for e in insetEmitters(mutant: mutant) where e.species == .glucose || e.species == .galactose {
        expect(insetHasLactase(e.tag), "a sugar split over cell \(e.tag), which has no lactase")
    }
    let waterCells: [Int] = insetEmitters(mutant: mutant).filter { $0.species == .water }.map { $0.tag }
    expect(!waterCells.isEmpty && waterCells.allSatisfy { !insetHasLactase($0) })
}

test("below, every glucose and galactose is still bonded as lactose — none is ever free") {
    var halves = 0, lone = 0
    for ps in loopParticles {
        var byItem: [Int: [Particle]] = [:]
        for p in ps where p.panel == 1 && (p.species == .glucose || p.species == .galactose) {
            halves += 1
            byItem[p.item, default: []].append(p)
        }
        for (_, pair) in byItem {
            let r: Float = drawnRadius(.glucose)
            if pair.count != 2 || pair[0].species == pair[1].species { lone += pair.count; continue }
            let d: Float = simd_distance(pair[0].position, pair[1].position)
            if abs(d - 2 * r * 0.92) > 0.05 { lone += 2 }
        }
    }
    expect(halves > 1000, "only \(halves) sugar spheres below")
    expect(lone == 0, "\(lone) sugar spheres below are not in a bonded lactose")
}

test("above, the split halves come apart into the villus; no lactose is left past the valve") {
    let es: [Emitter] = emitters(panel: 0, mutant: mutant).filter { $0.route == .villusSplit }
    let t: Float = 2.2
    var before: [SIMD4<Float>] = [], after: [SIMD4<Float>] = []
    for e in es {
        let a: SIMD3<Float> = wallPath(e, age: splitAge * 0.4, t: t)
        let b: SIMD3<Float> = wallPath(e, age: 0.8, t: t)
        before.append(SIMD4(a.x, a.y, -0.3, 0))
        after.append(SIMD4(b.x, b.y, -0.3, 0))
    }
    guard let ra = probe(before, t), let rb = probe(after, t) else { expect(false, "probe failed"); return }
    expect(ra.allSatisfy { $0.x > 0 }, "a lactose inside tissue before it reached the tip")
    let inside: Int = rb.filter { $0.x < 0 }.count
    expect(inside == rb.count, "only \(inside) of \(rb.count) split sugars went into the tissue")
    var stray = 0
    for ps in loopParticles {
        for p in ps where p.panel == 0 && p.route == .lumen && (p.species == .glucose || p.species == .galactose) {
            if p.position.x > lipX { stray += 1 }
        }
    }
    expect(stray == 0, "\(stray) lactose halves past the valve in the upper panel")
}

section("osmosis alone: no chloride, no ion leading the water")

test("no Cl⁻ secretion anywhere in the scene, and no Cl⁻ on screen at all") {
    for panel in 0..<2 {
        let es: [Emitter] = emitters(panel: panel, mutant: mutant)
        expect(!chlorideSecretion(panel: panel, mutant: mutant))
        expect(es.allSatisfy { $0.route != .crypt }, "panel \(panel) has crypt secretion")
        expect(es.allSatisfy { $0.species != .chloride }, "panel \(panel) emits Cl⁻")
    }
    let cl: Int = loopParticles.reduce(0) { $0 + $1.filter { $0.species == .chloride }.count }
    expect(cl == 0, "\(cl) chloride particles drawn over the loop")
}

test("below, water moves INTO the lumen, with no ion moving lumen-ward before or with it") {
    let n = waterBudget(panel: 0, mutant: mutant)
    let d = waterBudget(panel: 1, mutant: mutant)
    let lowerVillusOut: Int = d.out - d.colonOut
    print("        upper: \(n.into) in, \(n.out) back · lower: \(d.into) in, \(lowerVillusOut) back by the villi, \(d.colonOut) by the colon")
    expect(d.into > lowerVillusOut, "lower small intestine: \(d.into) in vs \(lowerVillusOut) out")
    expect(n.out > n.into, "upper: \(n.out) out vs \(n.into) in")
    // Everything that moves from the tissue into the lumen, anywhere, is water.
    for panel in 0..<2 {
        for e in emitters(panel: panel, mutant: mutant) where e.route == .crypt || e.route == .osmoticInflow {
            expect(e.species == .water, "a \(e.species) moves into the lumen in panel \(panel)")
            expect(panel == 1, "osmotic inflow in the upper panel")
        }
    }
    // In the close-up, only water goes up, and every ion comes down.
    for e in insetEmitters(mutant: mutant) {
        let up: Bool = insetPath(e, age: 0.9).y > insetPath(e, age: 0.1).y
        expect(up == (e.species == .water), "\(e.species) goes \(up ? "up" : "down") in the close-up")
    }
    // And the inflowing water really comes out of a villus into the open lumen.
    let es: [Emitter] = emitters(panel: 1, mutant: mutant).filter { $0.route == .osmoticInflow }
    let t: Float = 4.0
    var a: [SIMD4<Float>] = [], b: [SIMD4<Float>] = []
    for e in es {
        let p0: SIMD3<Float> = wallPath(e, age: 0.25, t: t), p1: SIMD3<Float> = wallPath(e, age: 0.98, t: t)
        a.append(SIMD4(p0.x, p0.y, -0.3, 1)); b.append(SIMD4(p1.x, p1.y, -0.3, 1))
    }
    guard let ra = probe(a, t), let rb = probe(b, t) else { expect(false, "probe failed"); return }
    expect(ra.allSatisfy { $0.x < 0 }, "inflowing water should start inside tissue")
    expect(rb.allSatisfy { $0.x > 0 }, "inflowing water should end in the lumen")
}

section("the stool: the osmotic gap")

test("the lower panel's particle mix has an osmotic gap over 125 — osmotic, not secretory") {
    let s: StoolMix = stool(mutant: mutant)
    let na: Float = s.concentration(s.sodium), k: Float = s.concentration(s.potassium)
    print(String(format: "        %d particles × %.0f mOsm/kg: implied Na⁺ %.1f K⁺ %.1f → gap %.0f mOsm/kg",
                 s.total, mOsmPerParticle, na, k, s.gap))
    expect(s.gap > osmoticGapLimit, "gap \(s.gap)")
    expect(abs(Float(s.total) * mOsmPerParticle - stoolOsmolality) <= mOsmPerParticle, "mix totals \(Float(s.total) * mOsmPerParticle)")
    // The measured lactulose-diarrhoea values it was built from (Hammer 1989).
    expect(abs(na - 33) < 1 && abs(k - 30) < 1, "Na⁺ \(na) K⁺ \(k), want 33 and 30")
    // Unforced check: ~30% of the organic acids ionized ≈ the cations they hold.
    let anions: Float = 0.3 * s.concentration(s.scfa)
    expect(abs(anions - (na + k)) / (na + k) < 0.06, "ionized organic acids \(anions) vs Na⁺ + K⁺ \(na + k)")
    expect(osmoticGap(sodium: 33, potassium: 30) == 164)
}

test("the upper panel is consistent: formed stool, gap between the cut-offs, no lactose, gas or SCFA") {
    let g: Float = osmoticGap(sodium: normalStoolSodium, potassium: normalStoolPotassium)
    expect(g == 80 && g > secretoryGapLimit && g < osmoticGapLimit, "normal stool gap \(g)")
    expect(stool(mutant: mutant).gap > g, "the lower stool should hold more unmeasured osmoles than normal")
    let made: Int = loopParticles.reduce(0) { $0 + $1.filter { $0.panel == 0 && ($0.species == .gas || $0.species == .scfa) }.count }
    expect(made == 0, "\(made) gas or SCFA particles in the upper panel")
}

test("what reaches the rectum below IS the stool mix, item for item") {
    var seen: [Species: Set<Int>] = [:]
    for ps in loopParticles {
        for p in ps where p.panel == 1 && p.route == .lumen && p.position.x > breakEndX + 2 && p.alpha > 0.5 {
            let s: Species = p.species == .galactose ? .glucose : p.species
            seen[s, default: []].insert(p.item)
        }
    }
    let s: StoolMix = stool(mutant: mutant)
    let got: (Int, Int, Int, Int) = (seen[.sodium]?.count ?? 0, seen[.potassium]?.count ?? 0,
                                     seen[.scfa]?.count ?? 0, seen[.glucose]?.count ?? 0)
    print("        at the rectum: Na⁺ \(got.0), K⁺ \(got.1), SCFA \(got.2), lactose \(got.3)")
    expect(got.0 == s.sodium && got.1 == s.potassium && got.2 == s.scfa && got.3 == s.lactose,
           "rectum \(got) vs mix \(s.sodium) \(s.potassium) \(s.scfa) \(s.lactose)")
    let counted = StoolMix(sodium: got.0, potassium: got.1, scfa: got.2, lactose: got.3)
    expect(counted.gap > osmoticGapLimit, "gap of what is drawn at the rectum: \(counted.gap)")
}

section("fermentation: only in the colon, only where bacteria met lactose")

test("every gas bubble and SCFA is in the lower colon, downstream of where its lactose met its bacterium") {
    var n = 0, bad = 0
    for ps in loopParticles {
        for p in ps where p.species == .gas || p.species == .scfa {
            n += 1
            let j: Int = (p.parent / 8) % 10_000
            guard p.panel == 1, p.position.x > lipX, p.parent == itemID(1, .lactose, j),
                  let xf = fermentX(j, mutant: mutant), p.position.x > xf - 1.5 else { bad += 1; continue }
        }
    }
    expect(n > 2000, "only \(n) gas/SCFA particles over the loop")
    expect(bad == 0, "\(bad) gas/SCFA particles appear where no bacterium met a lactose")
}

test("each lactose is taken up only when it has reached its bacterium") {
    var checked = 0, far = 0
    for ps in loopParticles {
        var lactoseAt: [Int: SIMD3<Float>] = [:], lactoseAlpha: [Int: Float] = [:], bug: [Int: [SIMD3<Float>]] = [:]
        for p in ps where p.panel == 1 && p.route == .lumen {
            if p.species == .glucose || p.species == .galactose {
                lactoseAt[p.item, default: .zero] += p.position * 0.5
                lactoseAlpha[p.item] = p.alpha
            }
            if p.species == .bacterium { bug[p.item, default: []].append(p.position) }
        }
        for (item, a) in lactoseAlpha where a < 0.6 {
            let j: Int = (item / 8) % 10_000
            guard let b = bug[itemID(1, .bacterium, j)], b.count == 3 else { continue }
            let centre: SIMD3<Float> = (b[0] + b[1] + b[2]) / 3
            checked += 1
            if simd_distance(centre, lactoseAt[item]!) > 1.0 { far += 1 }
        }
    }
    expect(checked > 20, "only \(checked) uptakes seen")
    expect(far == 0, "\(far) lactose fading away from their bacterium")
}

test("fermentation yields 3.7 SCFA per lactose; the colon takes some back, not all") {
    let made: Int = scfaProduced(mutant: mutant), back: Int = scfaAbsorbed(mutant: mutant)
    expect(Float(made) / Float(fermentedParcels) > 3.6 && Float(made) / Float(fermentedParcels) < 3.8)
    expect(back > 0 && back < made, "absorbed \(back) of \(made)")
    expect(made % 41 != 0, "the salvage permutation needs 41 coprime to \(made)")
    let chosen: Int = (0..<made).filter { scfaSalvaged($0, of: made, absorbed: back) }.count
    expect(chosen == back, "\(chosen) chosen, \(back) intended")
    // And they are seen going: into the colonic wall, in the lower panel only.
    var going: Set<Int> = []
    for ps in loopParticles { for p in ps where p.route == .scfaSalvage { expect(p.panel == 1); going.insert(p.item) } }
    expect(going.count > back * 3 / 4, "only \(going.count) of \(back) salvaged SCFAs seen leaving")
    let t: Float = 5.0
    var ends: [SIMD4<Float>] = []
    for x: Float in stride(from: 12, through: 84, by: 6) {
        let R: Float = colonRadiusAt(x, panel: 1, t: t)
        for side: Float in [1, -1] { ends.append(SIMD4(x, colonAxisY + side * (R + salvageDepth), -0.3, 1)) }
    }
    guard let r = probe(ends, t) else { expect(false, "probe failed"); return }
    expect(r.allSatisfy { $0.x < 0 && $0.y == 3 }, "a salvaged SCFA ends outside the colonic wall")
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
    expect(lowerWave.speed > normalWave.speed)
}

test("every lumen item and stool lump moves forward, frame after frame, all loop") {
    func centres(_ ps: [Particle]) -> [Int: (x: Float, panel: Int)] {
        var sum: [Int: (Float, Int, Int)] = [:]
        for p in ps where p.route == .lumen && p.item >= 0 && p.panel < 2 {
            let v = sum[p.item] ?? (0, 0, p.panel)
            sum[p.item] = (v.0 + p.position.x, v.1 + 1, p.panel)
        }
        return sum.mapValues { ($0.0 / Float($0.1), $0.2) }
    }
    var backwards = 0, checked = 0
    for f in 0..<frames {
        let a = centres(loopParticles[f])
        let b = centres(loopParticles[(f + 1) % frames])
        for (id, ca) in a {
            guard let cb = b[id] else { continue }
            let dx: Float = cb.x - ca.x
            if abs(dx) > gutSpan / 2 || (ca.panel == 0 && dx < -20) { continue }   // a wrap or a respawn
            checked += 1
            if dx <= 0 { backwards += 1 }
        }
        let t0: Float = frameTime(f, of: frames), t1: Float = frameTime(f + 1, of: frames)
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
        if a[i].species != b[i].species || a[i].item != b[i].item { worst = 99 }
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

test("arrows are an honest count: every route shows its arrow for the same length of time") {
    let routes: [Route] = [.crypt, .villusAbsorb, .osmoticInflow, .colonAbsorb]
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

section("the distance function is a distance")

test("outside every surface, it never claims more room than there is, beyond what the ray allows") {
    // As in step 23: pairs of nearby points, outside the surfaces only, and no
    // slope divided out. The over-report must stay under 1/stepScale.
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
            a.append(SIMD4(p, panel))
            b.append(SIMD4(p + dir * step, panel))
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

test("the rendered frame agrees: villi only upstream of the valve, bubbles and bacteria seen below only") {
    guard let r = gpu.renderer else { expect(false, "no GPU"); return }
    guard (try? r.render(time: 2.0, samples: 1, mutant: mutant)) != nil else { expect(false, "render failed"); return }
    let f: Frame = r.frame
    var villusPixels = 0, strayVillus = 0, gasPixels = [0, 0], particlePixels = 0
    for y in 0..<f.layout.height {
        for x in 0..<f.layout.width {
            let s: SIMD4<Float> = f.seen(x, y)
            let mat: Int = Int(s.y) % 100
            if mat == 2 { villusPixels += 1; if s.z > lipX { strayVillus += 1 } }
            if s.w > 0 { particlePixels += 1 }
            let panel: Int = Int(s.x)
            if panel == 0 || panel == 1, Int(s.w + 0.5) - 1 == Species.gas.rawValue { gasPixels[panel] += 1 }
        }
    }
    print("        villus pixels \(villusPixels), past the valve \(strayVillus); gas pixels \(gasPixels); particle pixels \(particlePixels)")
    expect(villusPixels > 1000, "villi barely visible: \(villusPixels) pixels")
    expect(strayVillus == 0, "\(strayVillus) villus pixels past the valve")
    expect(gasPixels[0] == 0 && gasPixels[1] > 50, "gas pixels \(gasPixels)")
    expect(particlePixels > 1000)
}

finish()
