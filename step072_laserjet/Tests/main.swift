// Tests for step 72, the HP LaserJet P1102w. The printer's size is measured
// off the kernel's own distance function (trays closed, as HP measures) and
// held to HP's Table C-1; the parts HP names are found where they should
// be; the distance function is held to the definition of a distance; the
// inset's dots are measured off the picture and held to 600 dpi; the toner
// to Wikipedia's bounds; and the scale bars to the camera and the inset.
//
// PRINTER_MUTANT=wrongDPI|wrongSize|tonerTooBig breaks the step on purpose;
// `make mutants` requires the suite to fail for each.

import CoreGraphics
import Foundation
import Metal
import simd

let mutant: Mutant = activeMutant
if mutant != .none { print("MUTANT: \(mutant.rawValue)") }

let setup: StillSetup = stillSetup(mutant)
let renderer: PrinterRenderer? = {
    guard let d = try? findDevice() else { return nil }
    return try? PrinterRenderer(device: d, width: 1920, height: 1080, studio: setup.studio, mutant: mutant)
}()

section("the printer, against HP's user guide")

/// How far the printer reaches along a direction: sphere-trace many parallel
/// lines in from outside and take the farthest hit.
func reach(along axis: SIMD3<Float>, across u: SIMD3<Float>, _ v: SIMD3<Float>, uRange: ClosedRange<Float>, vRange: ClosedRange<Float>,
           centre: SIMD3<Float>, open: Bool) -> Float {
    guard let r = renderer else { return .nan }
    let n: Int = 40
    var starts: [SIMD3<Float>] = []
    for i in 0..<n {
        for j in 0..<n {
            let a: Float = uRange.lowerBound + (uRange.upperBound - uRange.lowerBound) * Float(i) / Float(n - 1)
            let b: Float = vRange.lowerBound + (vRange.upperBound - vRange.lowerBound) * Float(j) / Float(n - 1)
            starts.append(centre + axis * 600 + u * a + v * b)
        }
    }
    var t: [Float] = Array(repeating: 0, count: starts.count)
    var done: [Bool] = Array(repeating: false, count: starts.count)
    for _ in 0..<300 {
        let pts: [SIMD3<Float>] = (0..<starts.count).map { starts[$0] - axis * t[$0] }
        guard let pr = try? r.probe(pts, trayOpen: open) else { return .nan }
        for i in 0..<starts.count where !done[i] {
            let d: Float = pr[i].z
            if d < 0.002 { done[i] = true } else { t[i] += d * 0.95 }
            if t[i] > 1200 { done[i] = true; t[i] = .infinity }
        }
    }
    let best: Float = t.min() ?? .infinity
    return 600 - best
}

test("closed up, as HP measures it: \(Int(hpWidth)) × \(Int(hpDepth)) × \(Int(hpHeight)) mm (Table C-1), to half a millimetre") {
    let c = SIMD3<Float>(0, 0, 0)
    let X = SIMD3<Float>(1, 0, 0), Y = SIMD3<Float>(0, 1, 0), Z = SIMD3<Float>(0, 0, 1)
    let hs: Float = printerScale(mutant) * 1.1
    let yr: ClosedRange<Float> = 1...(hpHeight * hs)
    let xr: ClosedRange<Float> = (-hpWidth / 2 * hs)...(hpWidth / 2 * hs)
    let zr: ClosedRange<Float> = (-hpDepth / 2 * hs)...(hpDepth / 2 * hs)
    let right: Float = reach(along: X, across: Y, Z, uRange: yr, vRange: zr, centre: c, open: false)
    let left: Float = reach(along: -X, across: Y, Z, uRange: yr, vRange: zr, centre: c, open: false)
    let front: Float = reach(along: Z, across: X, Y, uRange: xr, vRange: yr, centre: c, open: false)
    let back: Float = reach(along: -Z, across: X, Y, uRange: xr, vRange: yr, centre: c, open: false)
    let top: Float = reach(along: Y, across: X, Z, uRange: xr, vRange: zr, centre: c, open: false)
    let w: Float = right + left
    let d: Float = front + back
    print(String(format: "        drawn %.2f × %.2f × %.2f mm; HP %.0f × %.0f × %.0f mm", w, d, top, hpWidth, hpDepth, hpHeight))
    expect(abs(w - hpWidth) < 0.5, "width \(w)")
    expect(abs(d - hpDepth) < 0.5, "depth \(d)")
    expect(abs(top - hpHeight) < 0.5, "height \(top)")
}

test("HP's weight is plausible for that box: 5.3 kg over its bounding volume is a light, hollow thing") {
    let litres: Float = hpWidth * hpDepth * hpHeight / 1e6
    let density: Float = hpWeightKilograms / litres
    print(String(format: "        %.1f L box, mean density %.2f kg/L (solid plastic is ~1.0)", litres, density))
    expect(density > 0.1 && density < 1.0)
}

test("the control panel as HP lays it out: wireless button, wireless light, attention light, ready light, cancel button, back to front") {
    guard let r = renderer else { expect(false, "no GPU"); return }
    let s: Float = printerScale(mutant)
    var seq: [Int] = []
    // Look down onto the panel: the first thing each vertical ray meets.
    var pts: [SIMD3<Float>] = []
    for i in 0..<400 { pts.append(SIMD3<Float>(-148 * s, (hpHeight + 5) * s, (-15 + Float(i) * 0.2) * s)) }
    var t: [Float] = Array(repeating: 0, count: pts.count)
    var mats: [Int] = Array(repeating: 0, count: pts.count)
    for _ in 0..<200 {
        let q: [SIMD3<Float>] = (0..<pts.count).map { pts[$0] - SIMD3<Float>(0, t[$0], 0) }
        guard let pr = try? r.probe(q) else { expect(false, "probe failed"); return }
        for i in 0..<pts.count where mats[i] == 0 {
            if pr[i].x < 0.002 { mats[i] = Int(pr[i].y) } else { t[i] += pr[i].x * 0.95 }
        }
    }
    for m in mats where m != 2 && m != 0 { if seq.last != m { seq.append(m) } }
    print("        along the panel: " + seq.map { ["", "", "", "button", "", "", "", "", "", "", "wireless light", "attention light", "ready light", "cancel button"][min($0, 13)] }.joined(separator: ", "))
    expectEqual(seq, [3, 10, 11, 12, 13])
}

section("the distance function")

test("outside every surface, no distance claims more room than the ray allows") {
    guard let r = renderer else { expect(false, "no GPU"); return }
    var rng = SystemRandomNumberGenerator()
    let s: Float = printerScale(mutant)
    var a: [SIMD3<Float>] = []
    for i in 0..<240_000 {
        if i % 2 == 0 {
            a.append(SIMD3<Float>(Float.random(in: -200...200, using: &rng) * s, Float.random(in: 0.001...215, using: &rng) * s,
                                  Float.random(in: -130...300, using: &rng) * s))
        } else if i % 4 == 1 {
            // Near the front and the top, where the cuts are.
            a.append(SIMD3<Float>(Float.random(in: -180...180, using: &rng) * s, Float.random(in: 0.001...200, using: &rng) * s,
                                  Float.random(in: 80...140, using: &rng) * s))
        } else {
            // Round the control panel and the power button.
            let panel: Bool = i % 8 == 3
            a.append(panel ? SIMD3<Float>(Float.random(in: -160...(-136), using: &rng) * s, Float.random(in: 186...200, using: &rng) * s,
                                          Float.random(in: -15...60, using: &rng) * s)
                           : SIMD3<Float>(Float.random(in: -150...(-126), using: &rng) * s, Float.random(in: 28...52, using: &rng) * s,
                                          Float.random(in: 105...125, using: &rng) * s))
        }
    }
    var worst: Float = 0
    var mats: Set<Int> = []
    for open in [true, false] {
        for step in [Float(0.004), 0.05, 0.5] {
            let b: [SIMD3<Float>] = a.map { p in
                let d: SIMD3<Float> = simd_normalize(SIMD3<Float>(Float.random(in: -1...1, using: &rng), Float.random(in: -1...1, using: &rng),
                                                                  Float.random(in: -1...1, using: &rng)))
                var q: SIMD3<Float> = p + d * step
                q.y = max(q.y, 0.0005)
                return q
            }
            guard let pa = try? r.probe(a, trayOpen: open), let pb = try? r.probe(b, trayOpen: open) else { expect(false, "probe failed"); return }
            for i in 0..<a.count {
                guard pa[i].x > 0, pb[i].x > 0 else { continue }
                mats.insert(Int(pa[i].y))
                let dist: Float = simd_distance(a[i], b[i])
                if Int(pa[i].y) == Int(pb[i].y) { worst = max(worst, abs(pa[i].x - pb[i].x) / dist) }
                if pa[i].z > 0 && pb[i].z > 0 { worst = max(worst, abs(pa[i].z - pb[i].z) / dist) }
            }
        }
    }
    print(String(format: "        worst |Δd|/|Δp| outside %.3f (the ray allows %.3f); materials seen %@", worst, 1 / stepScale,
                 mats.sorted().map { String($0) }.joined(separator: ",")))
    expect(worst * stepScale <= 1, "oversteps: \(worst)")
    for m in [1, 2, 3, 5] { expect(mats.contains(m), "material \(m) never sampled") }
}

section("how it prints: the inset")

let inset: InsetContent = buildInset(mutant)

test("the inset's grid is 600 dpi: dots 25.4 mm / 600 = 42.33 µm apart, in the drawn data") {
    let want: Float = 25_400 / 600
    var spacing: [Float] = []
    let cs: [SIMD2<Float>] = inset.dots.map { $0.centre }
    for (i, a) in cs.enumerated() {
        var best: Float = .infinity
        for (j, b) in cs.enumerated() where j != i { best = min(best, simd_distance(a, b)) }
        spacing.append(best)
    }
    let median: Float = spacing.sorted()[spacing.count / 2]
    print(String(format: "        %d dots; nearest-neighbour spacing %.2f µm; 600 dpi is %.2f µm", cs.count, median, want))
    expect(abs(median - want) < 0.01, "dots \(median) µm apart")
    expect(cs.count > 100)
}

test("toner particles inside Wikipedia's bounds (≈5 µm for 600 dpi … 14–16 µm for old toners), and far smaller than a dot") {
    var lo: Float = .infinity, hi: Float = 0
    for d in inset.dots { for p in d.particles { lo = min(lo, p.z); hi = max(hi, p.z) } }
    print(String(format: "        particles %.1f–%.1f µm across; a dot is %.1f µm", lo, hi, inset.pitch))
    expect(tonerSourcedRange.contains(lo) && tonerSourcedRange.contains(hi), "particles \(lo)–\(hi) µm")
    expect(hi < inset.pitch / 4, "particles as big as dots")
}

var rendered: Bool = false
if let r = renderer, (try? r.render(camera: setup.camera, samples: 1)) != nil {
    rendered = true
    annotate(r.image, setup: setup, inset: inset)
}

test("measured off the picture: the letter's edge steps by one dot, 42.33 µm, every time it steps") {
    guard rendered, let img = renderer?.image else { expect(false, "render failed"); return }
    let rect: CGRect = insetRect(height: img.height)
    let um: Float = insetMicrometresPerPixel(height: img.height)
    let pitch600: Float = 25_400 / 600
    // Rows through dot centres (the page grid, whatever pitch was drawn is
    // what the picture shows); the letter's right-hand edge on each.
    var edges: [Float] = []
    var y: Float = inset.gridOrigin.y
    while y < insetFieldMicrometres - 20 {
        let py: Int = Int(rect.maxY - CGFloat(y / um))
        if py > Int(rect.minY) + 30 && py < Int(rect.maxY) - 30 {
            // Rightmost pixel of a run of ≥ 12 dark ones.
            var run: Int = 0
            var edge: Int = -1
            for x in Int(rect.minX)..<Int(rect.maxX) {
                let p: SIMD4<UInt8> = img.rgba(x, py)
                let lum: Int = Int(p.x) + Int(p.y) + Int(p.z)
                if lum < 3 * 70 { run += 1; if run >= 12 { edge = x } } else { run = 0 }
            }
            if edge >= 0 { edges.append(Float(edge)) }
        }
        y += pitch600
    }
    var steps: [Float] = []
    for i in 1..<edges.count { let d: Float = abs(edges[i] - edges[i - 1]); if d > 6 { steps.append(d) } }
    let sorted: [Float] = steps.sorted()
    let median: Float = sorted.isEmpty ? 0 : sorted[sorted.count / 2]
    let wantPx: Float = pitch600 / um
    print(String(format: "        %d rows, %d steps; the median step %.1f px = %.1f µm; one dot at 600 dpi is %.1f px",
                 edges.count, steps.count, median, median * um, wantPx))
    expect(steps.count >= 4, "too few steps seen: \(steps.count)")
    expect(abs(median - wantPx) < 0.15 * wantPx, "the edge steps by \(median * um) µm, not \(pitch600)")
}

test("the scale bars are true: 100 mm at the printer's middle, projected; 100 µm in the inset") {
    guard rendered, let img = renderer?.image else { expect(false, "render failed"); return }
    let a: SIMD2<Float> = project(setup.centre, width: img.width, height: img.height, cam: setup.camera)
    let b: SIMD2<Float> = project(setup.centre + setup.camera.right * mainBarMillimetres, width: img.width, height: img.height, cam: setup.camera)
    let projected: Float = simd_distance(a, b)
    let bar: ScaleBar = mainScaleBar(setup, width: img.width, height: img.height)
    let insetBar: ScaleBar = insetScaleBar(height: img.height)
    let insetWant: Float = insetBarMicrometres / insetMicrometresPerPixel(height: img.height)
    var painted: Int = 0
    let yy: Int = Int(bar.y + 1.5)
    for x in max(Int(bar.x) - 4, 0)..<min(Int(bar.x + bar.pixels) + 5, img.width) {
        let p: SIMD4<UInt8> = img.rgba(x, yy)
        if Int(p.x) + Int(p.y) + Int(p.z) < 3 * 60 { painted += 1 }
    }
    print(String(format: "        100 mm: projected %.1f px, bar %.1f px, painted %d px; 100 µm: %.1f px, bar %.1f px",
                 projected, Float(bar.pixels), painted, insetWant, Float(insetBar.pixels)))
    expect(abs(Float(bar.pixels) - projected) < 0.5)
    expect(abs(Float(painted) - projected) <= 2.5)
    expect(abs(Float(insetBar.pixels) - insetWant) < 0.01)
}

section("the picture")

test("the picture shows the printer — matte black with soft reflections, its lights lit — beside the inset") {
    guard rendered, let img = renderer?.image else { expect(false, "render failed"); return }
    var counts: [Int: Int] = [:]
    var spec: Double = 0, all: Double = 0
    let rect: CGRect = insetRect(height: img.height)
    var overInset: Int = 0
    for y in 0..<img.height {
        for x in 0..<img.width {
            let a: SIMD4<Float> = img.seen(x, y)
            let m: Int = Int(a.x)
            counts[m, default: 0] += 1
            if m >= 2 && rect.contains(CGPoint(x: x, y: y)) { overInset += 1 }
            if m == 2 { let l: SIMD4<Float> = img.light(x, y); spec += Double(l.x); all += Double(l.y) }
        }
    }
    let total: Double = Double(img.width * img.height)
    let housing: Double = Double(counts[2] ?? 0) / total
    let share: Double = spec / max(all, 1e-9)
    print(String(format: "        housing %.1f%% of the frame; reflection's share of its light %.2f; lights %d, %d px; under the inset %d px",
                 housing * 100, share, counts[10] ?? 0, counts[12] ?? 0, overInset))
    expect(housing > 0.10, "the printer is too small in the frame")
    expect(share > 0.2 && share < 0.95, "the black plastic's reflections are \(share) of its light")
    expect((counts[10] ?? 0) > 4 && (counts[12] ?? 0) > 4, "the lights are not seen")
    expect(overInset == 0, "the inset covers the printer")
}

finish()
