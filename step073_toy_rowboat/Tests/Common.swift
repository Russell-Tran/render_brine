// Tests every toy-boat step shares (grown from steps 58–63's): the
// plastic's optics against Zhang et al., the distance function against the
// definition of a distance, the parting line, the size against Russell's
// spec, and the still — the scale bar, the inset and the picture, read back
// off the rendered pixels.

import CoreGraphics
import Foundation
import Metal
import simd

/// Holds the device so a failure to find one is a test failure, not a crash.
final class DeviceBox { let device = try? findDevice() }
let gpu = DeviceBox()

// MARK: - optics

func testOptics(_ renderer: ToyRenderer?, mutant: Mutant) {
    section("the plastic's gloss, from its refractive index")

    test("PVC's index is Zhang et al.'s: 1.54493 at 550 nm, falling from 1.561 at 400 nm to 1.535 at 780 nm") {
        expect(abs(pvcIndex(nanometres: 550) - 1.54493) < 1e-9, "n(550) = \(pvcIndex(nanometres: 550))")
        expect(abs(pvcIndex(nanometres: 400) - 1.56135) < 1e-9)
        expect(abs(pvcIndex(nanometres: 780) - 1.53526) < 1e-9)
        for i in 1..<pvcZhang.count {
            expect(pvcZhang[i].micrometres > pvcZhang[i - 1].micrometres, "rows out of order at \(i)")
            expect(pvcZhang[i].n < pvcZhang[i - 1].n, "PVC's n should fall with wavelength (normal dispersion) at \(i)")
        }
    }

    test("F0 = ((n − 1)/(n + 1))² = 4.58% for PVC; acrylic 3.94%, polystyrene 5.20%") {
        let f: Double = fresnelF0(1, pvcIndex(nanometres: 550))
        print(String(format: "        PVC F0 %.5f; PMMA %.5f; polystyrene %.5f", f, fresnelF0(1, pmmaIndex550),
                     fresnelF0(1, polystyreneIndex550)))
        // Worked by hand: (0.54493 / 2.54493)² = 0.21412² = 0.045848.
        expect(abs(f - 0.045848) < 2e-6, "F0 = \(f)")
        expect(abs(fresnelDielectric(n: 1.54493, cosTheta: 1) - f) < 1e-12, "the angle formula at normal incidence")
        expect(abs(fresnelDielectric(n: 1.54493, cosTheta: 0) - 1) < 1e-12, "at grazing, everything is mirrored")
        // Brewster's angle: p-polarised reflectance vanishes at tan θB = n.
        let thetaB: Double = atan(1.54493)
        let c: Double = cos(thetaB)
        let s2: Double = 1 - c * c
        let ct: Double = (1 - s2 / (1.54493 * 1.54493)).squareRoot()
        let rp: Double = (1.54493 * c - ct) / (1.54493 * c + ct)
        expect(abs(rp) < 1e-9, "r_p at Brewster's angle = \(rp)")
    }

    test("the render's surface mirrors with PVC's F0: the kernel's own table, read back from the GPU") {
        guard let r = renderer else { expect(false, "no GPU"); return }
        guard let g = try? r.fresnelOnGPU([1.0, 0.5, 0.0]) else { expect(false, "probe failed"); return }
        let n: SIMD3<Double> = surfaceIndex()
        let want0 = SIMD3<Double>(fresnelF0(1, n.x), fresnelF0(1, n.y), fresnelF0(1, n.z))
        let want5: Double = fresnelDielectric(n: n.y, cosTheta: 0.5)
        print(String(format: "        GPU F(cos 1) = (%.5f, %.5f, %.5f), want (%.5f, %.5f, %.5f); F(cos 0.5) %.5f, want %.5f",
                     g[0].x, g[0].y, g[0].z, want0.x, want0.y, want0.z, g[1].y, want5))
        expect(abs(Double(g[0].x) - want0.x) < 1e-5 && abs(Double(g[0].y) - want0.y) < 1e-5 && abs(Double(g[0].z) - want0.z) < 1e-5,
               "F0 on the GPU is \(g[0])")
        expect(abs(Double(g[1].y) - want5) < 2e-4, "F(60°) on the GPU is \(g[1].y), want \(want5)")
        expect(abs(g[2].y - 1) < 1e-6, "F at grazing on the GPU is \(g[2].y)")
    }
}

// MARK: - the distance function

/// Random points round the toy, packed densely near its surface too.
func samplePoints(_ toy: PosedToy, count: Int) -> [SIMD3<Float>] {
    var rng = SystemRandomNumberGenerator()
    let ex = toy.extent(along: SIMD3<Float>(1, 0, 0))
    let ey = toy.extent(along: SIMD3<Float>(0, 1, 0))
    let ez = toy.extent(along: SIMD3<Float>(0, 0, 1))
    var pts: [SIMD3<Float>] = []
    for k in 0..<count {
        if k % 2 == 0 {
            pts.append(SIMD3<Float>(Float.random(in: (ex.lo - 4)...(ex.hi + 4), using: &rng),
                                    Float.random(in: 0.001...(ey.hi + 4), using: &rng),
                                    Float.random(in: (ez.lo - 4)...(ez.hi + 4), using: &rng)))
        } else {
            // Near a random segment's frame origin and shapes.
            let i: Int = Int.random(in: 0..<toy.segments.count, using: &rng)
            let s: Segment = toy.segments[i]
            let p: Prim = s.prims[Int.random(in: 0..<s.prims.count, using: &rng)]
            let b = p.bound
            let q: SIMD3<Float> = b.centre + SIMD3<Float>(Float.random(in: -1...1, using: &rng), Float.random(in: -1...1, using: &rng),
                                                         Float.random(in: -1...1, using: &rng)) * (b.radius + 0.6)
            var w: SIMD3<Float> = toy.frames[i].toWorld(q)
            w.y = max(w.y, 0.001)
            pts.append(w)
        }
    }
    return pts
}

func testDistanceFunction(_ renderer: ToyRenderer?, toy: PosedToy, label: String) {
    test("\(label): outside every surface, no distance claims more room than the ray allows") {
        guard let r = renderer else { expect(false, "no GPU"); return }
        var rng = SystemRandomNumberGenerator()
        let a: [SIMD3<Float>] = samplePoints(toy, count: 160_000)
        var worst: [Int: Float] = [:]
        for step in [Float(0.01), 0.25] {
            let b: [SIMD3<Float>] = a.map { p in
                let d: SIMD3<Float> = simd_normalize(SIMD3<Float>(Float.random(in: -1...1, using: &rng), Float.random(in: -1...1, using: &rng),
                                                                  Float.random(in: -1...1, using: &rng)))
                var q: SIMD3<Float> = p + d * step
                q.y = max(q.y, 0.0005)
                return q
            }
            guard let pa = try? r.probe(a, toy: toy), let pb = try? r.probe(b, toy: toy) else { expect(false, "probe failed"); return }
            for i in 0..<a.count {
                // Only points outside count: the table, and the toy.
                let ma: Int = Int(pa[i].y)
                guard ma == Int(pb[i].y), pa[i].x > 0, pb[i].x > 0 else { continue }
                let v: Float = abs(pa[i].x - pb[i].x) / simd_distance(a[i], b[i])
                worst[ma] = max(worst[ma] ?? 0, v)
                // The toy's own distance too, even where the table is nearer.
                if pa[i].z > 0 && pb[i].z > 0 {
                    let u: Float = abs(pa[i].z - pb[i].z) / simd_distance(a[i], b[i])
                    worst[9] = max(worst[9] ?? 0, u)
                }
            }
        }
        print(String(format: "        worst |Δd|/|Δp| outside: table %.3f, toy (scene) %.3f, toy alone %.3f; the ray allows %.3f",
                     worst[1] ?? 0, worst[2] ?? 0, worst[9] ?? 0, 1 / stepScale))
        expect(worst[1] != nil && worst[2] != nil && worst[9] != nil, "not every material was sampled: \(worst)")
        for (m, v) in worst { expect(v * stepScale <= 1.0, "material \(m) oversteps: \(v)") }
    }

    test("\(label): the GPU's distance (with its bounding-sphere shortcut) is the CPU's, everywhere") {
        guard let r = renderer else { expect(false, "no GPU"); return }
        let pts: [SIMD3<Float>] = samplePoints(toy, count: 20_000)
        guard let g = try? r.probe(pts, toy: toy) else { expect(false, "probe failed"); return }
        var worst: Float = 0
        var segMismatch: Int = 0
        for (i, p) in pts.enumerated() {
            let c = toy.sdf(p)
            worst = max(worst, abs(c.d - g[i].z))
            if abs(c.d - g[i].z) < 1e-4 && c.segment != Int(g[i].w) && c.d < 0.5 { segMismatch += 1 }
        }
        print(String(format: "        worst CPU/GPU difference %.2e mm over %d points", worst, pts.count))
        expect(worst < 2e-3, "the GPU's toy differs from the CPU's by \(worst) mm")
    }
}

// MARK: - the parting line

/// How far a probe travelling along −dir (in segment i's frame) from a
/// start point gets before it meets the surface.
func surfaceHit(_ toy: PosedToy, segment i: Int, from start: SIMD3<Float>, dir: SIMD3<Float>) -> Float {
    var t: Float = 0
    for _ in 0..<600 {
        let q: SIMD3<Float> = start - dir * t
        let d: Float = toy.segments[i].sdf(local: q, seams: toy.seams).d
        if d < 1e-5 { return t }
        t += d * 0.9
    }
    return t
}

/// One place to measure the parting line: a world point on it, the
/// outward direction to look along, the direction square to the seam, and
/// how far either side of it the smooth surface is sampled.
struct SeamStation {
    var segment: Int
    var at: SIMD3<Float>
    var outward: SIMD3<Float>
    var across: SIMD3<Float>
    var name: String
    var offsets: (Float, Float) = (0.25, 0.5)
    var reach: Float = 20
}

/// The height of the parting-line ridge at a station: the surface on the
/// seam against the surface either side of it, where the ridge is not, with
/// the curvature taken out (a parabola through the two offsets).
func ridgeHeight(_ toy: PosedToy, _ s: SeamStation) -> Float {
    let f: Frame = toy.frames[s.segment]
    let q: SIMD3<Float> = f.toLocal(s.at)
    let dir: SIMD3<Float> = f.dirToLocal(simd_normalize(s.outward))
    let across: SIMD3<Float> = f.dirToLocal(simd_normalize(s.across))
    let start: SIMD3<Float> = q + dir * s.reach
    func out(_ o: Float) -> Float { s.reach - surfaceHit(toy, segment: s.segment, from: start + across * o, dir: dir) }
    let a1: Float = s.offsets.0
    let a2: Float = s.offsets.1
    let h0: Float = out(0)
    let h1: Float = (out(a1) + out(-a1)) / 2
    let h2: Float = (out(a2) + out(-a2)) / 2
    let c: Float = (h2 - h1) / (a2 * a2 - a1 * a1)
    let smooth: Float = h1 - c * a1 * a1
    return h0 - smooth
}

func testSeam(_ toy: PosedToy, stations: [SeamStation], where place: String) {
    section("the mould's parting line")
    test("a ridge \(seamHeight) mm proud runs where the mould's halves met: \(place)") {
        var report: [String] = []
        for s in stations {
            let h: Float = ridgeHeight(toy, s)
            report.append(String(format: "%@ %.3f", s.name, h))
            expect(h > 0.6 * seamHeight && h < 1.4 * seamHeight, "\(s.name): ridge \(h) mm, the seam is \(seamHeight)")
        }
        print("        ridge heights (mm): " + report.joined(separator: ", "))
    }
    test("the seam is a hairline: \(2 * seamHalfWidth) mm wide and \(seamHeight) mm high, well inside moulding's ±0.2 mm") {
        expect(2 * seamHalfWidth < 0.2 && seamHeight < 0.2)
        expect(seamHeight >= 0.01, "a seam under 10 µm would not catch the light")
    }
}

// MARK: - size

func testSize(_ toy: PosedToy, name: String) {
    section("size, against Russell's spec")
    test("the \(name) is 2–3 inches long, bow to stern: 50.8–76.2 mm") {
        let e = toy.extent(along: SIMD3<Float>(1, 0, 0))
        let len: Float = e.hi - e.lo
        print(String(format: "        %.2f mm long (%.2f in)", len, len / 25.4))
        expect(specLengthRange.contains(len), "length \(len) mm")
    }
    test("it stands on the table: its lowest point touches y = 0 and nothing goes below") {
        let low: Float = toy.extent(along: SIMD3<Float>(0, 1, 0)).lo
        print(String(format: "        lowest point %.4f mm", low))
        expect(low > -2e-3 && low < 0.05, "lowest point \(low) mm")
    }
}

// MARK: - the still's picture

func pixelLinear(_ img: ToyImage, _ x: Int, _ y: Int) -> SIMD3<Double> {
    let p: SIMD4<UInt8> = img.rgba(x, y)
    return SIMD3<Double>(srgbByteToLinear(p.x), srgbByteToLinear(p.y), srgbByteToLinear(p.z))
}

/// Runs of PVC and air along an inset row, read off the picture: each
/// pixel named by the colour it is, within ΔE 3 (CIELAB).
func insetRuns(_ img: ToyImage, row y: Int) -> [(String, Int, Int)] {
    let r: CGRect = insetRect(height: img.height)
    let pvcLab: SIMD3<Double> = linearSRGBToLab(SIMD3<Double>(massColour))
    let airLab: SIMD3<Double> = linearSRGBToLab(SIMD3<Double>(insetAir))
    var runs: [(String, Int, Int)] = []
    for x in Int(r.minX) + 3..<Int(r.maxX) - 3 {
        let lab: SIMD3<Double> = linearSRGBToLab(pixelLinear(img, x, y))
        let name: String = simd_distance(lab, pvcLab) < 3 ? "pvc" : (simd_distance(lab, airLab) < 3 ? "air" : "other")
        if let last = runs.last, last.0 == name { runs[runs.count - 1].2 += 1 } else { runs.append((name, x, 1)) }
    }
    // Drop what labels drew over it, then join what they split.
    var joined: [(String, Int, Int)] = []
    for r in runs where r.0 != "other" {
        if let last = joined.last, last.0 == r.0 { joined[joined.count - 1].2 += r.2 } else { joined.append(r) }
    }
    return joined
}

/// Renders the still (1 sample), checks the frame, annotates it, checks the
/// scale bar and where the cut is marked. Returns the annotated image for
/// the boat's own inset checks.
func testStillPicture(_ renderer: ToyRenderer?, setup s: StillSetup, caption: Caption, mutant: Mutant) -> ToyImage? {
    section("the picture")
    var rendered: Bool = false
    if let r = renderer, (try? r.render(s.toy, camera: s.camera, samples: 1)) != nil { rendered = true }

    test("the frame holds the toy large, on the table; its paint shows a sheen of the lights") {
        guard rendered, let img = renderer?.image else { expect(false, "render failed"); return }
        var toyPx: Int = 0
        var table: Int = 0
        var sheen: Int = 0
        var specSum: Double = 0
        var diffSum: Double = 0
        var fWorst: Double = 0
        var minX: Int = img.width
        var maxX: Int = 0
        for y in 0..<img.height {
            for x in 0..<img.width {
                let a: SIMD4<Float> = img.seen(x, y)
                if a.x == 1 { table += 1 }
                guard a.x == 2 else { continue }
                toyPx += 1
                minX = min(minX, x)
                maxX = max(maxX, x)
                let l: SIMD4<Float> = img.light(x, y)
                specSum += Double(l.x)
                diffSum += Double(l.y)
                if l.x > 0.25 * l.y && l.x > 0.02 { sheen += 1 }
                if a.w > 0.05 {
                    fWorst = max(fWorst, abs(Double(l.z) - fresnelDielectric(n: surfaceIndex().y, cosTheta: Double(a.w))))
                }
            }
        }
        let total: Double = Double(img.width * img.height)
        let span: Double = Double(maxX - minX) / Double(img.width)
        print(String(format: "        toy %.1f%% of the frame, %.0f%% of its width, table %.1f%%; mirrored/diffuse %.3f; %d sheen pixels; worst F error %.2e",
                     Double(toyPx) / total * 100, span * 100, Double(table) / total * 100, specSum / max(diffSum, 1e-9), sheen, fWorst))
        expect(Double(toyPx) / total > 0.05 && Double(toyPx) / total < 0.4, "toy covers \(Double(toyPx) / total)")
        expect(span > 0.4, "the toy should span over 40% of the frame's width: \(span)")
        expect(minX > 2, "the toy runs off the left edge")
        expect(Double(table) / total > 0.4, "table covers \(Double(table) / total)")
        expect(specSum / max(diffSum, 1e-9) > 0.04, "the surface mirrors too little: \(specSum / max(diffSum, 1e-9))")
        expect(sheen > 2_000, "only \(sheen) pixels show the paint's sheen")
        expect(fWorst < 2e-3, "the Fresnel the kernel used differs from PVC's by \(fWorst)")
    }

    if rendered, let r = renderer { annotate(r.image, setup: s, caption: caption) }

    test("the inset is on the picture: each of its pixels is PVC exactly where the distance function says the cut is solid") {
        guard rendered, let img = renderer?.image else { expect(false, "render failed"); return }
        let r: CGRect = insetRect(height: img.height)
        let pvcLab: SIMD3<Double> = linearSRGBToLab(SIMD3<Double>(massColour))
        var solid: Int = 0
        var agree: Int = 0
        var checked: Int = 0
        var y: Int = Int(r.minY) + 60
        while y < Int(r.maxY) - 60 {
            var x: Int = Int(r.minX) + 4
            while x < Int(r.maxX) - 4 {
                let p: SIMD3<Float> = insetPoint(CGFloat(x) + 0.5, CGFloat(y) + 0.5, cut: s.cut, height: img.height)
                let want: Bool = s.toy.sdf(p).d <= 0
                let lab: SIMD3<Double> = linearSRGBToLab(pixelLinear(img, x, y))
                let isPVC: Bool = simd_distance(lab, pvcLab) < 3
                // Labels are drawn over the inset: only count bare pixels.
                let isAir: Bool = simd_distance(lab, linearSRGBToLab(SIMD3<Double>(insetAir))) < 3
                if isPVC || isAir {
                    checked += 1
                    if isPVC == want { agree += 1 }
                    if want { solid += 1 }
                }
                x += 3
            }
            y += 3
        }
        let share: Double = Double(agree) / Double(max(checked, 1))
        let percent: Double = share * 100
        print(String(format: "        %d inset pixels checked, %d solid; %.3f%% agree with the distance function", checked, solid, percent))
        expect(solid > 800, "the cut shows almost no plastic: \(solid)")
        expect(share > 0.999, "the inset disagrees with the toy")
    }

    test("the scale bars are true: 10 mm at the toy, projected, is the bar as painted; the inset's bar is its scale") {
        guard rendered, let img = renderer?.image else { expect(false, "render failed"); return }
        let a: SIMD2<Float> = project(s.centre, width: img.width, height: img.height, cam: s.camera)
        let b: SIMD2<Float> = project(s.centre + s.camera.right * mainBarMillimetres, width: img.width, height: img.height, cam: s.camera)
        let projected: Float = simd_distance(a, b)
        let bar: ScaleBar = mainScaleBar(s, width: img.width, height: img.height)
        let y: Int = Int(bar.y + 1.5 * CGFloat(img.height) / 1080)
        var painted: Int = 0
        for x in max(Int(bar.x) - 4, 0)..<min(Int(bar.x + bar.pixels) + 5, img.width) {
            let p: SIMD4<UInt8> = img.rgba(x, y)
            if Int(p.x) + Int(p.y) + Int(p.z) < 3 * 60 { painted += 1 }
        }
        let insetBar: ScaleBar = insetScaleBar(height: img.height)
        let insetExpected: Float = insetBarMillimetres / insetMillimetresPerPixel(height: img.height)
        print(String(format: "        10 mm: projected %.1f px, bar %.1f px, painted %d px; inset %g mm: %.1f px, bar %.1f px",
                     projected, Float(bar.pixels), painted, insetBarMillimetres, insetExpected, Float(insetBar.pixels)))
        expect(abs(Float(bar.pixels) - projected) < 0.5, "bar \(bar.pixels) vs projected \(projected)")
        expect(abs(Float(painted) - projected) <= 2.5, "bar painted \(painted) px")
        expect(abs(Float(insetBar.pixels) - insetExpected) < 0.01)
    }

    test("the cut is marked on the toy, in view, left of the inset, and its outline lies on the toy's surface") {
        guard rendered, let img = renderer?.image else { expect(false, "render failed"); return }
        let r: CGRect = insetRect(height: img.height)
        expect(!s.cut.marks.isEmpty, "no outline")
        var worst: Float = 0
        for m in s.cut.marks {
            let p: SIMD2<Float> = project(m, width: img.width, height: img.height, cam: s.camera)
            expect(p.x > 0 && p.x < Float(r.minX) && p.y > 0 && p.y < Float(img.height), "cut mark off the view: \(p)")
            worst = max(worst, abs(s.toy.sdf(m).d))
            expect(abs(simd_dot(m - s.cut.centre, simd_cross(s.cut.right, s.cut.down))) < 1e-3, "a mark off the cut's plane")
        }
        print(String(format: "        %d outline points, worst %.3f mm off the surface", s.cut.marks.count, worst))
        expect(worst < 0.1, "the outline is \(worst) mm off the toy")
    }
    return rendered ? renderer?.image : nil
}
