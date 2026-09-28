// Tests for step 29: step 25's suite, copied (the still's flower is the
// animation's flower), then the growth — the tube only lengthens, reaches the
// ovule exactly when growth ends, stays inside the pistil at every frame, and
// the loop closes by dissolving forward to frame 0, never by rewinding; and
// the animation's first frame is step 25's picture but for the tube.
//
// BEAN_MUTANT=rewind|strays (step 29) or tube_leaves|open|all_free (step 25)
// breaks it on purpose; `make mutants` requires the suite to fail for each.
//
// Step 25's header follows.
//
// Tests for step 25. The flower's numbers are checked against the floras, and
// its shape against the claims the picture makes — by probing the same
// distance function the GPU draws with, not a copy of it. Then the finished
// picture is read back for the things a viewer must be able to see.
//
// BEAN_MUTANT=tube_leaves|open|all_free breaks the flower on purpose;
// `make mutants` requires the suite to fail for each.

import CoreGraphics
import Foundation
import ImageIO
import Metal
import simd

let mutant: Mutant = Mutant.fromEnvironment
let model = FlowerModel(mutant)

/// Holds the scene so a failure to build one is a test failure, not a crash.
final class SceneBox {
    var scene: BeanScene?
    var error: String = ""
    init() {
        do { scene = try BeanScene(model, on: try findDevice()) } catch { self.error = "\(error)" }
    }
}
let box = SceneBox()

func probe(_ pts: [SIMD3<Float>], _ mode: ProbeMode, grown: Float? = nil) -> [SIMD2<Float>]? {
    guard let s = box.scene else { return nil }
    return try? s.probe(pts, mode: mode, tubeGrown: grown)
}

/// Sphere-trace many rays at once through one probe mode. Returns, per ray,
/// whether it met a surface within `reach`.
func traceHits(from origins: [SIMD3<Float>], _ dirs: [SIMD3<Float>], mode: ProbeMode, reach: Float) -> [Bool]? {
    var t = [Float](repeating: 0.002, count: origins.count)
    var done = [Bool](repeating: false, count: origins.count)
    var hit = [Bool](repeating: false, count: origins.count)
    for _ in 0..<400 {
        let pts: [SIMD3<Float>] = (0..<origins.count).map { origins[$0] + dirs[$0] * t[$0] }
        guard let d = probe(pts, mode) else { return nil }
        for i in 0..<origins.count where !done[i] {
            if d[i].x < 0.001 { hit[i] = true; done[i] = true; continue }
            t[i] += d[i].x * stepScale
            if t[i] > reach { done[i] = true }
        }
        if done.allSatisfy({ $0 }) { break }
    }
    return hit
}

/// Evenly spread unit vectors (a Fibonacci sphere).
func sphereDirections(_ n: Int) -> [SIMD3<Float>] {
    (0..<n).map { i in
        let y: Float = 1 - 2 * (Float(i) + 0.5) / Float(n)
        let r: Float = (1 - y * y).squareRoot()
        let a: Float = Float(i) * 2.3999632
        return SIMD3<Float>(r * cos(a), y, r * sin(a))
    }
}

if box.scene == nil { print("no scene: \(box.error)") }

// MARK: -

section("sizes, against the floras")

test("the standard is 1–1.9 cm long along its midline (FTEA), measured on the distance function") {
    // In the plane of symmetry, walk out from inside the bud in every
    // direction and find where the banner starts. The banner's midline is the
    // curve through those points, over the top from calyx to tip.
    let centre = SIMD3<Float>(6.5, 1.5, 0)
    let angles: Int = 720
    let steps: Int = 900
    var pts: [SIMD3<Float>] = []
    for a in 0..<angles {
        let th: Float = Float(a) / Float(angles) * 2 * Float.pi
        let dir = SIMD3<Float>(cos(th), sin(th), 0)
        for k in 0..<steps { pts.append(centre + dir * (Float(k) * 0.01)) }
    }
    guard let d = probe(pts, .banner) else { expect(false, "probe failed"); return }
    var curve: [SIMD3<Float>?] = []
    for a in 0..<angles {
        var found: SIMD3<Float>? = nil
        for k in 0..<steps where d[a * steps + k].x < 0 { found = pts[a * steps + k]; break }
        curve.append(found)
    }
    var length: Float = 0
    for a in 0..<angles {
        if let p = curve[a], let q = curve[(a + 1) % angles], simd_distance(p, q) < 0.4 { length += simd_distance(p, q) }
    }
    print(String(format: "        standard midline %.1f mm", length))
    expect(standardLengthRange.contains(length), "standard is \(length) mm")
}

test("the keel's centreline is FTEA's ± 2.2 cm, and coils through 1–5 turns") {
    let len: Float = keelCentrelineLength
    print(String(format: "        keel centreline %.1f mm, %.2f turns", len, keelTurns))
    expect(abs(len - keelLengthCited) / keelLengthCited < 0.2, "keel is \(len) mm")
    expect(keelTurns >= keelTurnsCitedMinimum && keelTurns <= 5)
}

test("pollen grains are PalDat's size: 41–50 µm across the equator, 36–40 µm pole to pole") {
    expect(pollenEquatorialRange.contains(pollenEquatorialDiameter))
    expect(pollenPolarRange.contains(pollenPolarDiameter))
    // And the drawn grain, measured on the distance function by bisection.
    let g: PollenGrain = germinatingGrain
    let eq: SIMD3<Float> = simd_normalize(simd_cross(g.pole, SIMD3<Float>(0.3, 0.4, 0.8)))
    func extent(_ dir: SIMD3<Float>) -> Float? {
        var lo: Float = 0
        var hi: Float = 0.05
        for _ in 0..<30 {
            let mid: Float = (lo + hi) / 2
            guard let d = probe([g.centre + dir * mid], .grains) else { return nil }
            if d[0].x < 0 { lo = mid } else { hi = mid }
        }
        return lo
    }
    guard let a = extent(eq), let b = extent(-eq), let c = extent(g.pole), let e = extent(-g.pole) else {
        expect(false, "probe failed"); return
    }
    let equatorial: Float = (a + b) * 1000
    let polar: Float = (c + e) * 1000
    print(String(format: "        drawn grain %.1f µm across the equator, %.1f µm pole to pole", equatorial, polar))
    expect(equatorial >= 41 && equatorial <= 50, "equatorial \(equatorial) µm")
    expect(polar >= 36 && polar <= 40, "polar \(polar) µm")
}

test("ovules and their count: six, in a row, inside the ovary's cavity") {
    expect(model.ovules.count >= 4 && model.ovules.count <= 12)
    guard let d = probe(model.ovules, .pistil) else { expect(false, "probe failed"); return }
    for (i, v) in d.enumerated() { expect(v.x < 0, "ovule \(i) is outside the ovary: \(v.x)") }
    let ys: [Float] = model.ovules.map { $0.y }
    expect((ys.max()! - ys.min()!) < 1e-4, "not in a row")
}

section("the coil")

test("the style coils through at least 360° (FTEA), the keel through at least one turn") {
    let styleTurn: Float = turning(from: styleStartS, to: stigmaS)
    let keelStart: Float = spine.first(where: { $0.p.x >= keelStartX })!.s
    let keelTurn: Float = turning(from: keelStart, to: spine[spine.count - 1].s)
    print(String(format: "        style turns %.0f°, keel %.0f°", styleTurn, keelTurn))
    expect(styleTurn >= styleCoilCitedMinimumDegrees, "style turns only \(styleTurn)°")
    expect(keelTurn >= keelTurnsCitedMinimum * 360, "keel turns only \(keelTurn)°")
}

test("the style lies inside the keel all the way round the coil") {
    let pts: [SpinePoint] = spine.filter { $0.styleRadius > 0 }
    guard let d = probe(pts.map { $0.p }, .keelCavity) else { expect(false, "probe failed"); return }
    var worst: Float = -1e9
    for (i, p) in pts.enumerated() { worst = max(worst, d[i].x + p.styleRadius) }
    print(String(format: "        closest the style comes to the keel wall: %.3f mm", -worst))
    expect(worst < -0.05, "the style reaches within \(-worst) mm of the keel wall")
}

test("neighbouring turns of the keel do not touch") {
    var worst: Float = 1e9
    let outer: (SpinePoint) -> Float = { $0.keelRadius + petalThickness / 2 }
    for i in stride(from: 0, to: spine.count, by: 3) {
        for j in stride(from: i, to: spine.count, by: 3) where spine[j].s - spine[i].s > 3.0 {
            let gap: Float = simd_distance(spine[i].p, spine[j].p) - outer(spine[i]) - outer(spine[j])
            // Only pairs a turn apart: the spine is straight or turning away elsewhere.
            if abs(turning(from: spine[i].s, to: spine[j].s) - 360) < 60 { worst = min(worst, gap) }
        }
    }
    print(String(format: "        narrowest gap between turns %.3f mm", worst))
    expect(worst > 0, "turns overlap by \(-worst) mm")
}

section("the stamens: diadelphous, 9 + 1")

test("ten stamens: nine fused, one free, and the free one is the vexillary, on top") {
    expectEqual(model.stamens.count, 10)
    expectEqual(model.stamens.filter { $0.fused }.count, 9)
    let free: [Stamen] = model.stamens.filter { !$0.fused }
    expectEqual(free.count, 1)
    expect(free.first?.angleDegrees == 0, "the free stamen should sit on top, toward the banner")
    expectEqual(model.anthers.count, 10)
}

test("on the distance function: a continuous sheath between the nine, open over the vexillary") {
    let x: Float = 4.0
    func at(_ deg: Float) -> SIMD3<Float> {
        let a: Float = deg * Float.pi / 180
        return SIMD3<Float>(x, sheathRadius * cos(a), sheathRadius * sin(a))
    }
    // Midway between each pair of neighbouring fused filaments (36°…324°).
    let between: [Float] = (1..<9).map { Float($0) * 36 + 18 }
    let pts: [SIMD3<Float>] = between.map(at) + [at(0), at(18), at(-18)]
    guard let d = probe(pts, .stamens) else { expect(false, "probe failed"); return }
    for (i, deg) in between.enumerated() {
        expect(d[i].x < 0, "no tissue at \(deg)° between fused filaments: \(d[i].x)")
    }
    expect(d[8].x < 0, "no vexillary filament on top: \(d[8].x)")
    expect(d[9].x > 0 && d[10].x > 0, "the vexillary is joined to the sheath: \(d[9].x), \(d[10].x)")
}

test("the anthers are clustered round the stigma, inside the keel's tip") {
    for a in model.anthers {
        let dist: Float = simd_distance(a.centre, stigmaCentre)
        expect(dist < 1.3, "an anther is \(dist) mm from the stigma")
    }
    guard let d = probe(model.anthers.map { $0.centre }, .keelCavity) else { expect(false, "probe failed"); return }
    for v in d { expect(v.x < 0, "an anther is outside the keel") }
}

section("a closed bud")

test("the banner, wings and calyx close round the stigma: no way out") {
    let dirs: [SIMD3<Float>] = sphereDirections(600)
    var origins: [SIMD3<Float>] = []
    var all: [SIMD3<Float>] = []
    for o in [stigmaCentre, coilCentre, SIMD3<Float>(5.0, 0, 0)] {
        for d in dirs { origins.append(o); all.append(d) }
    }
    guard let hits = traceHits(from: origins, all, mode: .enclosure, reach: 30) else {
        expect(false, "probe failed"); return
    }
    let escaped: Int = hits.filter { !$0 }.count
    print("        \(escaped) of \(hits.count) rays escape")
    expect(Double(escaped) / Double(hits.count) < 0.01, "\(escaped) rays escape: the flower is open")
}

test("the petals' margins overlap: banner over wings, wings across each other") {
    expect(180 - bannerBottomGapHalfDegrees > wingFromDegrees, "banner and wings do not meet")
    expect(wingToDegrees > 180, "the wings do not meet underneath")
    expectEqual(bannerOpenAngle(model.mutant), 0)
}

section("the pollen tube")

let tube: Chain = model.tube
let entryIndex: Int = tube.points.firstIndex(where: { simd_distance($0, tubeEntry) < 1e-5 }) ?? 0

test("it starts at a pollen grain that sits on the stigma") {
    guard let g = probe([tube.points[0]], .grains),
          let s = probe([germinatingGrain.centre - germinatingGrain.pole * grainPolarRadius], .pistil)
    else { expect(false, "probe failed"); return }
    expect(abs(g[0].x) < 0.002, "the tube starts \(g[0].x) mm from a grain")
    expect(abs(s[0].x) < 0.006, "the germinating grain is \(s[0].x) mm off the stigma")
}

test("it lies inside the pistil all the way from the stigma to the ovule") {
    expect(entryIndex > 0, "no entry point")
    let inside: [SIMD3<Float>] = Array(tube.points[entryIndex...])
    guard let d = probe(inside, .pistil), let surf = probe(Array(tube.points[1..<entryIndex]), .pistil) else {
        expect(false, "probe failed"); return
    }
    var worst: Float = -1e9
    var worstAt: Int = 0
    for (i, v) in d.enumerated() where v.x > worst { worst = v.x; worstAt = i }
    print(String(format: "        %d points inside; the closest to the pistil's surface is %.4f mm in", inside.count, -worst))
    expect(worst < -pollenTubeRadius * 0.5, "the tube leaves the pistil at \(inside[worstAt]): \(worst) mm")
    // Before it enters, it runs over the stigma's surface — on it, not off in the air.
    for v in surf { expect(v.x < 0.012, "the tube lifts \(v.x) mm off the stigma") }
}

test("it follows the coiled style, and ends at an ovule") {
    guard let last = tube.points.last, let d = probe([last], .ovules) else { expect(false, "probe failed"); return }
    expect(abs(d[0].x) < 0.01, "the tube ends \(d[0].x) mm from an ovule")
    var longest: Float = 0
    for i in 1..<tube.points.count { longest = max(longest, simd_distance(tube.points[i], tube.points[i - 1])) }
    expect(longest < 0.06, "a gap of \(longest) mm in the tube")
    var length: Float = 0
    for i in 1..<tube.points.count { length += simd_distance(tube.points[i], tube.points[i - 1]) }
    let styleLength: Float = stigmaS - styleStartS
    print(String(format: "        tube %.1f mm long; the style is %.1f mm", length, styleLength))
    expect(length > styleLength, "the tube is shorter than the style it runs down")
}

section("scale")

test("the main view shows pollen at true scale, a few pixels across") {
    let ppm: Float = mainPixelsPerMillimetre(height: 1080)
    let grainPx: Float = pollenEquatorialDiameter * ppm
    print(String(format: "        main view %.1f px/mm: a grain is %.1f px", ppm, grainPx))
    expect(grainPx > 2 && grainPx < 6, "a grain is \(grainPx) px")
}

test("the inset's scale bar and magnification agree with its camera") {
    let side: Int = insetPixels(width: 1920, height: 1080)
    let ppm: Float = insetPixelsPerMillimetre(side: side)
    let bar: Float = insetBarMillimetres * ppm
    let mag: Float = magnification(width: 1920, height: 1080)
    let label: String = magnificationLabel(width: 1920, height: 1080)
    let labelled: Float = Float(label.dropFirst()) ?? -1
    // Independently: a 0.1 mm segment at the stigma, projected by the inset camera.
    let cam: Camera = insetCamera()
    let a: SIMD2<Float> = cam.project(insetTarget, width: side, height: side)
    let b: SIMD2<Float> = cam.project(insetTarget + cam.right * insetBarMillimetres, width: side, height: side)
    let projected: Float = simd_distance(a, b)
    print(String(format: "        inset %.0f px/mm, 100 µm bar %.1f px (projected %.1f), %@", ppm, bar, projected, label))
    expect(abs(bar - projected) < 0.5, "bar \(bar) px, projection \(projected) px")
    expect(abs(labelled - mag) < 0.051, "labelled \(label), actual \(mag)")
    expect(abs(mag - ppm / mainPixelsPerMillimetre(height: 1080)) < 1e-4)
    expect(pollenEquatorialDiameter * ppm > 25, "grains are only \(pollenEquatorialDiameter * ppm) px in the inset")
}

section("the distance function is a distance")

test("outside every surface, it never claims more room than there is, beyond what the ray allows") {
    // Pairs of nearby points. A true distance changes by at most the distance
    // moved; the ray trusts only `stepScale` of each step, so the measured
    // over-report must stay under 1/stepScale. Only pairs OUTSIDE surfaces
    // count, because a ray only ever asks from outside.
    guard box.scene != nil else { expect(false, "no scene"); return }
    var rng = Lcg(state: 7)
    var a: [SIMD3<Float>] = []
    var b: [SIMD3<Float>] = []
    let step: Float = 0.004
    func add(_ p: SIMD3<Float>) {
        a.append(p)
        b.append(p + rng.unit() * step)
    }
    for _ in 0..<120_000 {
        add(SIMD3<Float>(-1 + 14.5 * rng.next(), -2.2 + 8.8 * rng.next(), -2 + 4 * rng.next()))
    }
    let tip: SIMD3<Float> = box.scene!.layout.tipCentre
    let tipR: Float = box.scene!.layout.tipRadius
    for _ in 0..<80_000 { add(tip + rng.unit() * (tipR * rng.next())) }
    for _ in 0..<30_000 { add(SIMD3<Float>(0.5 + 7 * rng.next(), -1 + 2 * rng.next(), -1 + 1.2 * rng.next())) }
    guard let da = probe(a, .render), let db = probe(b, .render) else { expect(false, "probe failed"); return }
    var worst = [Float](repeating: 0, count: 14)
    for i in 0..<a.count where da[i].y == db[i].y && da[i].x > 0 && db[i].x > 0 {
        let m: Int = Int(da[i].y)
        worst[m] = max(worst[m], abs(da[i].x - db[i].x) / step)
    }
    let names: [String] = ["-", "banner", "wing", "keel", "green", "ovary", "ovule", "stamen", "anther",
                           "style", "hair", "pollen", "tube", "pod"]
    var line: String = "        worst over-report:"
    for m in 1...12 { line += String(format: " %@ %.2f", names[m], worst[m]) }
    print(line)
    print(String(format: "        the ray allows %.2f", 1 / stepScale))
    for m in 1...12 { expect(worst[m] * stepScale <= 1.0, "\(names[m]) oversteps: \(worst[m])") }
}

section("the picture")

let small: (bytes: [UInt8], gpu: Double, flower: LayerImage, inset: LayerImage)? = {
    guard let s = box.scene else { return nil }
    return try? renderBean(s, width: 960, height: 540, samples: 2)
}()

func lab(_ r: Float, _ g: Float, _ b: Float) -> SIMD3<Float> {
    func lin(_ c: Float) -> Float { c <= 0.04045 ? c / 12.92 : pow((c + 0.055) / 1.055, 2.4) }
    let R: Float = lin(r), G: Float = lin(g), B: Float = lin(b)
    let X: Float = (0.4124564 * R + 0.3575761 * G + 0.1804375 * B) / 0.95047
    let Y: Float = 0.2126729 * R + 0.7151522 * G + 0.0721750 * B
    let Z: Float = (0.0193339 * R + 0.1191920 * G + 0.9503041 * B) / 1.08883
    func f(_ t: Float) -> Float { t > 0.008856 ? cbrt(t) : t / 0.12842 + 4.0 / 29.0 }
    return SIMD3<Float>(116 * f(Y) - 16, 500 * (f(X) - f(Y)), 200 * (f(Y) - f(Z)))
}

test("the petals read white — bright and nearly colourless, not grey") {
    guard let r = small else { expect(false, "render failed"); return }
    let img: LayerImage = r.flower
    var sum = SIMD3<Float>(0, 0, 0)
    var n: Int = 0
    for y in 0..<img.height {
        for x in 0..<img.width {
            let s: SIMD4<Float> = img.seen(x, y)
            let m: Int = Int(s.x)
            guard (m == Material.banner.rawValue || m == Material.wing.rawValue) && s.y == 0 else { continue }
            let i: Int = (y * img.width + x) * 4
            sum += lab(Float(r.bytes[i]) / 255, Float(r.bytes[i + 1]) / 255, Float(r.bytes[i + 2]) / 255)
            n += 1
        }
    }
    guard n > 1000 else { expect(false, "only \(n) petal pixels"); return }
    let m: SIMD3<Float> = sum / Float(n)
    let chroma: Float = (m.y * m.y + m.z * m.z).squareRoot()
    print(String(format: "        banner and wings: L* %.1f, chroma %.1f over %d pixels", m.x, chroma, n))
    expect(m.x > 85, "petals at L* \(m.x) read grey")
    expect(chroma < 8, "petals at chroma \(chroma) are not white")
}

test("the coil, the stigma and the pollen are in the frame; the inset shows grains and the tube") {
    guard let r = small else { expect(false, "render failed"); return }
    func count(_ img: LayerImage, _ m: Material) -> Int {
        var n: Int = 0
        for y in 0..<img.height { for x in 0..<img.width where Int(img.seen(x, y).x) == m.rawValue { n += 1 } }
        return n
    }
    let total: Float = Float(r.flower.width * r.flower.height)
    let keel: Float = Float(count(r.flower, .keel)) / total
    let style: Int = count(r.flower, .style)
    let insetTotal: Float = Float(r.inset.width * r.inset.height)
    let pollen: Float = Float(count(r.inset, .pollen)) / insetTotal
    let tubePx: Int = count(r.inset, .tube)
    print(String(format: "        main: keel %.1f%%, style %d px; inset: pollen %.1f%%, tube %d px",
                 keel * 100, style, pollen * 100, tubePx))
    expect(keel > 0.02, "the keel covers \(keel)")
    expect(style > 150, "the style covers \(style) px")
    expect(pollen > 0.04, "pollen covers \(pollen) of the inset")
    expect(tubePx > 40, "the pollen tube covers \(tubePx) inset px")
}



// MARK: - step 29: the growth

section("step 29: the tube grows")

let tubeLength: Float = box.scene?.layout.tubeLength ?? 0
let tubeArc: [Float] = tubeArcLengths(tube.points)

test("the kernel's tube length is the tube's arc length, pore to micropyle") {
    expect(tubeLength > 0, "no tube length")
    expect(abs(tubeLength - tubeArc[tubeArc.count - 1]) < 1e-6, "layout \(tubeLength), chain \(tubeArc.last!)")
    print(String(format: "        tube %.2f mm; at %.0f h, %.2f mm/h", tubeLength, realGrowthHours,
                 growthRate(tubeLength: tubeLength)))
    // Williams 2012: angiosperm tubes grow about 10 µm/h to over 20 mm/h.
    let rate: Float = growthRate(tubeLength: tubeLength)
    expect(rate > 0.01 && rate < 20, "rate \(rate) mm/h is outside the angiosperm range")
    expect(loopSeconds >= 8 && loopSeconds <= 12, "the loop is \(loopSeconds) s")
}

test("the tube only lengthens, and reaches the ovule exactly when growth ends") {
    for n in [110, 60, 137] {
        let tl = Timeline(frameCount: n, tubeLength: tubeLength, mutant: mutant)
        let arrive: Int = tl.arrivalFrame
        expect(arrive > n / 2 && arrive < n - 2, "arrival on frame \(arrive) of \(n)")
        var prev: Float = -1
        for f in 0..<n {
            let st: FrameState = tl.state(f)
            expect(st.grown >= prev, "\(n) frames: the tube shrinks at frame \(f): \(prev) → \(st.grown) mm")
            prev = st.grown
            if f < arrive { expect(st.grown < tubeLength, "\(n) frames: the tube arrives early, at frame \(f)") }
            if f == arrive { expect(st.grown == tubeLength, "\(n) frames: at arrival the tube is \(st.grown) mm") }
        }
        expectEqual(tl.state(0).grown, 0)
        // And monotonic by construction, at a much finer step than any frame.
        var g: Float = -1
        for k in 0...10_000 {
            let v: Float = grownFraction(Float(k) / 10_000)
            expect(v >= g, "grownFraction falls at \(k)")
            g = v
        }
        expectEqual(grownFraction(1), 1)
    }
}

test("when it arrives the tube's tip is at the micropyle of an ovule, and not before") {
    let tl = Timeline(frameCount: 110, tubeLength: tubeLength, mutant: .none)
    let tip: SIMD3<Float> = grownTube(tube, tl.state(tl.arrivalFrame).grown).last!
    guard let d = probe([tip], .ovules) else { expect(false, "probe failed"); return }
    expect(abs(d[0].x) < 0.01, "at arrival the tip is \(d[0].x) mm from an ovule")
    expect(simd_distance(tip, micropyle(fertilisedOvule)) < 1e-4, "at arrival the tip is not at the micropyle")
    let before: SIMD3<Float> = grownTube(tube, tl.state(tl.arrivalFrame - 1).grown).last!
    expect(simd_distance(before, micropyle(fertilisedOvule)) > 0.05, "the tube is already at the ovule a frame early")
}

test("at every frame, the grown tube lies inside the pistil — its growing tip included") {
    let tl = Timeline(frameCount: 110, tubeLength: tubeLength, mutant: .none)
    var inside: [SIMD3<Float>] = []
    var insideFrame: [Int] = []
    var surface: [SIMD3<Float>] = []
    for f in 0..<tl.frameCount {
        let pts: [SIMD3<Float>] = grownTube(tube, tl.state(f).grown)
        for (i, p) in pts.enumerated() {
            if i >= entryIndex { inside.append(p); insideFrame.append(f) } else if i > 0 { surface.append(p) }
        }
    }
    guard let d = probe(inside, .pistil), let sd = probe(surface, .pistil) else { expect(false, "probe failed"); return }
    var worst: Float = -1e9
    var bad: Set<Int> = []
    for (i, v) in d.enumerated() {
        worst = max(worst, v.x)
        if v.x >= -pollenTubeRadius * 0.5 { bad.insert(insideFrame[i]) }
    }
    print(String(format: "        %d points over %d frames; the closest to the pistil's surface is %.4f mm in",
                 inside.count, tl.frameCount, -worst))
    expect(bad.isEmpty, "the tube leaves the pistil at frames \(bad.sorted())")
    // Before it enters, it runs on the grain it left or on the stigma — never
    // off in the air.
    guard let sg = probe(surface, .grains) else { expect(false, "probe failed"); return }
    for (i, v) in sd.enumerated() {
        expect(min(v.x, sg[i].x) < 0.012, "the tube lifts \(v.x) mm off the stigma, \(sg[i].x) mm off the grain")
    }
}

test("the kernel draws the tube exactly as far as it has grown") {
    guard box.scene != nil else { expect(false, "no scene"); return }
    var failures: Int = 0
    for grown in [Float(0.05), 0.4, 3.0, 9.7, tubeLength * 0.8] {
        var pts: [SIMD3<Float>] = []
        var arc: [Float] = []
        for (i, p) in tube.points.enumerated() { pts.append(p); arc.append(tubeArc[i]) }
        let tip: SIMD3<Float> = grownTube(tube, grown).last!
        guard let d = probe(pts + [tip], .tube, grown: grown) else { expect(false, "probe failed"); return }
        for i in 0..<pts.count {
            if arc[i] < grown - 0.001 && abs(d[i].x + pollenTubeRadius) > 1e-4 { failures += 1 }
            if arc[i] > grown + 0.02 && d[i].x <= 0.01 { failures += 1 }
        }
        expect(abs(d[pts.count].x + pollenTubeRadius) < 1e-4, "at \(grown) mm the tip reads \(d[pts.count].x)")
    }
    expectEqual(failures, 0)
}

test("the distance function stays a distance with the tube part-grown") {
    guard box.scene != nil else { expect(false, "no scene"); return }
    var rng = Lcg(state: 29)
    let step: Float = 0.004
    var worst: Float = 0
    var worstAt: Float = 0
    for grown in [Float(0.1), 2.0, 7.5, tubeLength * 0.9] {
        let tip: SIMD3<Float> = grownTube(tube, grown).last!
        var a: [SIMD3<Float>] = []
        var b: [SIMD3<Float>] = []
        for _ in 0..<40_000 {
            let p: SIMD3<Float> = tip + rng.unit() * (0.2 * rng.next())
            a.append(p)
            b.append(p + rng.unit() * step)
        }
        guard let da = probe(a, .render, grown: grown), let db = probe(b, .render, grown: grown) else {
            expect(false, "probe failed"); return
        }
        for i in 0..<a.count where da[i].y == db[i].y && da[i].x > 0 && db[i].x > 0 {
            let r: Float = abs(da[i].x - db[i].x) / step
            if r > worst { worst = r; worstAt = grown }
        }
    }
    print(String(format: "        worst over-report round the growing tip %.2f (at %.1f mm grown); the ray allows %.2f",
                 worst, worstAt, 1 / stepScale))
    expect(worst * stepScale <= 1.0, "oversteps \(worst) near the tip")
}

test("the loop closes by dissolving forward to frame 0, never by rewinding") {
    for n in [110, 60] {
        let tl = Timeline(frameCount: n, tubeLength: tubeLength, mutant: mutant)
        var prevFade: Float = 0
        var fading: Int = 0
        for f in 0..<n {
            let st: FrameState = tl.state(f)
            if tl.u(f) < holdEnd {
                expectEqual(st.fade, 0)
                continue
            }
            fading += 1
            expect(st.grown == tubeLength, "\(n) frames: the tube is \(st.grown) mm during the dissolve, at frame \(f)")
            expect(st.fade > prevFade && st.fade < 1, "\(n) frames: fade \(st.fade) after \(prevFade) at frame \(f)")
            prevFade = st.fade
        }
        expect(fading >= 5, "\(n) frames: only \(fading) frames of dissolve")
        expect(prevFade > 0.8, "\(n) frames: the last frame has dissolved only \(prevFade) toward frame 0")
    }
}

/// A small loop, rendered for the picture tests below.
let smallLoop: Loop? = {
    guard let s = box.scene, let fr = try? BeanFrames(s, width: 480, height: 270, samples: 1) else { return nil }
    return Loop(fr, frameCount: 40, delayCentiseconds: 25, mutant: mutant)
}()

func pixelDistance(_ a: [UInt8], _ b: [UInt8]) -> Double {
    var sum: Double = 0
    for i in 0..<a.count { sum += abs(Double(a[i]) - Double(b[i])) }
    return sum
}

test("the dissolve moves the picture steadily toward frame 0") {
    guard let loop = smallLoop, let first = try? loop.frame(0) else { expect(false, "render failed"); return }
    let tl: Timeline = loop.timeline
    guard let held = try? loop.frame(tl.arrivalFrame) else { expect(false, "render failed"); return }
    let total: Double = pixelDistance(held, first)
    expect(total > 0, "the grown tube changes nothing on screen")
    var prev: Double = total
    for f in tl.arrivalFrame..<tl.frameCount {
        guard let img = try? loop.frame(f) else { expect(false, "render failed"); return }
        let d: Double = pixelDistance(img, first)
        if tl.u(f) >= holdEnd { expect(d < prev, "frame \(f) is no closer to frame 0: \(d) after \(prev)") }
        prev = d
    }
    print(String(format: "        the last frame is %.0f%% of the way from the grown tube to frame 0", 100 * (1 - prev / total)))
    expect(prev < total * 0.25, "the last frame is still \(prev / total) of the way from frame 0")
}

test("frame 0 is step 25's composition: same camera, light and scene numbers") {
    // Step 25's values, copied here from its committed source.
    expectEqual(cameraDistance, 60)
    expectEqual(mainHalfHeight, 6.8)
    expectEqual(mainTarget, SIMD3<Float>(10.3, 1.0, 0))
    expectEqual(viewDirection, SIMD3<Float>(0.30, 0.52, 1.0))
    expectEqual(insetFieldDiameter, 0.56)
    expectEqual(keyDirection, simd_normalize(SIMD3<Float>(-0.3, 0.6, 0.85)))
    expectEqual(stepScale, 0.85)
    let still: FlowerModel = FlowerModel(.none)
    expectEqual(model.grains.count, still.grains.count)
    expectEqual(model.hairs.count, still.hairs.count)
    expectEqual(model.anthers.count, still.anthers.count)
}

test("with the tube fully grown, the picture is step 25's committed still, pixel for pixel") {
    guard let s = box.scene, let r = try? renderBean(s, width: 1920, height: 1080, samples: 3) else {
        expect(false, "render failed"); return
    }
    let url = URL(fileURLWithPath: "../showcase/bean.png")
    guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
          let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else { expect(false, "cannot read \(url.path)"); return }
    var ref = [UInt8](repeating: 0, count: 1920 * 1080 * 4)
    ref.withUnsafeMutableBytes { raw in
        let ctx = CGContext(data: raw.baseAddress, width: 1920, height: 1080, bitsPerComponent: 8, bytesPerRow: 1920 * 4,
                            space: CGColorSpace(name: CGColorSpace.sRGB)!,
                            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
        ctx?.draw(img, in: CGRect(x: 0, y: 0, width: 1920, height: 1080))
    }
    var differ: Int = 0
    for i in stride(from: 0, to: ref.count, by: 4)
    where ref[i] != r.bytes[i] || ref[i + 1] != r.bytes[i + 1] || ref[i + 2] != r.bytes[i + 2] { differ += 1 }
    print("        \(differ) of \(1920 * 1080) pixels differ from showcase/bean.png")
    expectEqual(differ, 0)
}

test("frame 0 differs from the still only round the tube") {
    guard let s = box.scene, let fr = try? BeanFrames(s, width: 640, height: 360, samples: 2) else {
        expect(false, "render failed"); return
    }
    fr.reuse = false
    guard let a = try? fr.render(grown: 0).bytes, let b = try? fr.render(grown: tubeLength).bytes else {
        expect(false, "render failed"); return
    }
    let main: PixelRect = fr.mainTubeRect(from: 0, to: tubeLength)
    let place: InsetPlacement = insetPlacement(width: 640, height: 360)
    var differ: Int = 0
    var outside: Int = 0
    for y in 0..<360 {
        for x in 0..<640 {
            let i: Int = (y * 640 + x) * 4
            if a[i] == b[i] && a[i + 1] == b[i + 1] && a[i + 2] == b[i + 2] { continue }
            differ += 1
            let inMain: Bool = x >= main.x0 && x < main.x1 && y >= main.y0 && y < main.y1
            let dx: Float = Float(x) + 0.5 - place.centre.x
            let dy: Float = Float(y) + 0.5 - place.centre.y
            let inInset: Bool = (dx * dx + dy * dy).squareRoot() < place.radius + 1
            if !inMain && !inInset { outside += 1 }
        }
    }
    print("        \(differ) pixels differ, \(outside) of them away from the tube")
    expect(differ > 50, "the tube hardly shows: \(differ) pixels")
    expectEqual(outside, 0)
}

test("re-rendering only the newly grown stretch gives the same picture as rendering it all") {
    guard let s = box.scene, let fr = try? BeanFrames(s, width: 480, height: 270, samples: 2),
          let whole = try? BeanFrames(s, width: 480, height: 270, samples: 2) else { expect(false, "render failed"); return }
    whole.reuse = false
    var worst: Int = 0
    for grown in [Float(0), 0.01, 0.03, 0.06, 0.1, 0.15, 0.2, 0.3, 0.9, 4.0, 12.0, 6.0, tubeLength] {
        guard let a = try? fr.render(grown: grown).bytes, let b = try? whole.render(grown: grown).bytes else {
            expect(false, "render failed"); return
        }
        var differ: Int = 0
        for i in 0..<a.count where a[i] != b[i] { differ += 1 }
        worst = max(worst, differ)
        expect(differ == 0, "at \(grown) mm, \(differ) bytes differ")
    }
    print("        worst: \(worst) bytes differ")
}

test("the growth shows in both views: the inset's tube and the main view's change grow with it") {
    guard let s = box.scene, let fr = try? BeanFrames(s, width: 640, height: 360, samples: 1) else {
        expect(false, "render failed"); return
    }
    fr.reuse = false
    var insetTube: [Int] = []
    var mainChange: [Int] = []
    var base: [UInt8] = []
    for grown in [Float(0), 0.1, 0.25, 5.0, tubeLength] {
        guard let r = try? fr.render(grown: grown) else { expect(false, "render failed"); return }
        var n: Int = 0
        for y in 0..<r.inset.height {
            for x in 0..<r.inset.width where Int(r.inset.seen(x, y).x) == Material.tube.rawValue { n += 1 }
        }
        insetTube.append(n)
        if base.isEmpty { base = r.bytes }
        var m: Int = 0
        let place: InsetPlacement = insetPlacement(width: 640, height: 360)
        for y in 0..<360 {
            for x in 0..<Int(place.centre.x - place.radius - 2) {
                let i: Int = (y * 640 + x) * 4
                if abs(Int(r.bytes[i]) - Int(base[i])) + abs(Int(r.bytes[i + 1]) - Int(base[i + 1])) > 6 { m += 1 }
            }
        }
        mainChange.append(m)
    }
    print("        inset tube pixels \(insetTube); main-view pixels changed \(mainChange)")
    expectEqual(insetTube[0], 0)
    expect(insetTube[1] > 0 && insetTube[2] > insetTube[1], "the inset does not show the tube growing")
    expect(mainChange[3] > mainChange[1] && mainChange[4] > mainChange[3], "the main view does not show it growing")
}

test("the clock runs from 0 to the full journey's hours, never backwards") {
    let tl = Timeline(frameCount: 110, tubeLength: tubeLength, mutant: .none)
    var prev: Float = -1
    for f in 0...tl.arrivalFrame {
        let h: Float = tl.hours(tl.state(f))
        expect(h >= prev, "the clock goes back at frame \(f)")
        prev = h
    }
    expectEqual(tl.hours(tl.state(0)), 0)
    expect(abs(prev - realGrowthHours) < 1e-5, "at arrival the clock reads \(prev) h")
    expectEqual(clockText(hours: 3.52), "3 h 30 min")
    expectEqual(clockText(hours: 10), "10 h 00 min")
}

finish()
