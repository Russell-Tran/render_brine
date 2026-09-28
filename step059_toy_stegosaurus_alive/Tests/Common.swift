// Tests every toy step shares: the plastic's optics against Zhang et al.,
// the distance function against the definition of a distance, the parting
// line, the size against Russell's spec, and — for the stills — the scale
// bar, the inset and the picture, read back off the rendered pixels.

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
    for _ in 0..<400 {
        let q: SIMD3<Float> = start - dir * t
        let d: Float = toy.segments[i].sdf(local: q, seams: toy.seams).d
        if d < 1e-5 { return t }
        t += d * 0.9
    }
    return t
}

/// The height of the parting-line ridge at a point on a segment's mid-plane
/// (local z = 0), seen along a local direction `dir` (outward, square to z):
/// the surface at z = 0, against the surface continued from z = ±0.25 and
/// ±0.5, where the ridge is not, with the curvature taken out.
func ridgeHeight(_ toy: PosedToy, segment i: Int, at q: SIMD3<Float>, dir: SIMD3<Float>) -> Float {
    let start: SIMD3<Float> = q + dir * 6
    func out(_ z: Float) -> Float { 6 - surfaceHit(toy, segment: i, from: start + SIMD3<Float>(0, 0, z), dir: dir) }
    let h0: Float = out(0)
    let h1: Float = (out(0.25) + out(-0.25)) / 2
    let h2: Float = (out(0.5) + out(-0.5)) / 2
    let c: Float = (h1 - h2) * 16 / 3
    let smooth: Float = h1 + c / 16
    return h0 - smooth
}

func testSeam(_ toy: PosedToy, stations: [(segment: Int, at: SIMD3<Float>, dir: SIMD3<Float>, name: String)]) {
    section("the mould's parting line")
    test("a ridge \(seamHeight) mm proud runs round the toy in each piece's mid-plane: body, head and every leg") {
        var report: [String] = []
        for s in stations {
            let h: Float = ridgeHeight(toy, segment: s.segment, at: s.at, dir: s.dir)
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
    test("the \(name) is 2–3 inches long, nose to tail tip: 50.8–76.2 mm") {
        let fwd: SIMD3<Float> = simd_normalize(SIMD3<Float>(toy.frames[0].x.x, 0, toy.frames[0].x.z))
        let e = toy.extent(along: fwd)
        let len: Float = e.hi - e.lo
        print(String(format: "        %.2f mm long (%.2f in)", len, len / 25.4))
        expect(specLengthRange.contains(len), "length \(len) mm")
    }
    test("it stands on the table: every foot touches y = 0 and nothing goes below it") {
        var worst: Float = 0
        for (i, s) in toy.segments.enumerated() where s.part == .foot {
            let f: Frame = toy.frames[i]
            let ld = SIMD3<Float>(f.x.y, f.y.y, f.z.y)
            var lo: Float = .infinity
            for p in s.prims { lo = min(lo, primExtent(p, ld).0) }
            worst = max(worst, abs(f.o.y + lo))
        }
        let low: Float = toy.extent(along: SIMD3<Float>(0, 1, 0)).lo
        print(String(format: "        worst foot off the table %.5f mm; lowest point %.5f mm", worst, low))
        expect(worst < 1e-3, "a foot is \(worst) mm off the table")
        expect(low > -1e-3, "part of the toy is \(-low) mm below the table")
    }
    test("four legs, each of three rigid pieces") {
        let legs: Set<Int> = Set(toy.segments.filter { $0.limb >= 0 }.map { $0.limb })
        expectEqual(legs.count, 4)
        for l in legs {
            let parts: [Part] = toy.segments.filter { $0.limb == l }.map { $0.part }
            expectEqual(parts, [.upperLeg, .lowerLeg, .foot])
        }
    }
}

// MARK: - the still's picture

func pixelLinear(_ img: ToyImage, _ x: Int, _ y: Int) -> SIMD3<Double> {
    let p: SIMD4<UInt8> = img.rgba(x, y)
    return SIMD3<Double>(srgbByteToLinear(p.x), srgbByteToLinear(p.y), srgbByteToLinear(p.z))
}

func testStillPicture(_ renderer: ToyRenderer?, setup s: StillSetup, caption: Caption, mutant: Mutant) {
    section("the picture")
    var rendered: Bool = false
    if let r = renderer, (try? r.render(s.toy, camera: s.camera, samples: 1)) != nil { rendered = true }

    test("the frame holds the toy and the table; the toy's paint shows a sheen of the lights") {
        guard rendered, let img = renderer?.image else { expect(false, "render failed"); return }
        var toyPx: Int = 0
        var table: Int = 0
        var sheen: Int = 0
        var specSum: Double = 0
        var diffSum: Double = 0
        var fWorst: Double = 0
        for y in 0..<img.height {
            for x in 0..<img.width {
                let a: SIMD4<Float> = img.seen(x, y)
                if a.x == 1 { table += 1 }
                guard a.x == 2 else { continue }
                toyPx += 1
                let l: SIMD4<Float> = img.light(x, y)
                specSum += Double(l.x)
                diffSum += Double(l.y)
                if l.x > 0.25 * l.y && l.x > 0.02 { sheen += 1 }
                // The Fresnel the kernel used at this pixel's angle, against
                // the formula from PVC's n (only where the surface faces the
                // camera enough that the table is fine).
                if a.w > 0.05 {
                    fWorst = max(fWorst, abs(Double(l.z) - fresnelDielectric(n: surfaceIndex().y, cosTheta: Double(a.w))))
                }
            }
        }
        let total: Double = Double(img.width * img.height)
        print(String(format: "        toy %.1f%% of the frame, table %.1f%%; mirrored/diffuse light on the toy %.3f; %d pixels with a sheen; worst F error %.2e",
                     Double(toyPx) / total * 100, Double(table) / total * 100, specSum / max(diffSum, 1e-9), sheen, fWorst))
        expect(Double(toyPx) / total > 0.05 && Double(toyPx) / total < 0.4, "toy covers \(Double(toyPx) / total)")
        expect(Double(table) / total > 0.4, "table covers \(Double(table) / total)")
        expect(specSum / max(diffSum, 1e-9) > 0.04, "the surface mirrors too little: \(specSum / max(diffSum, 1e-9))")
        expect(sheen > 2_000, "only \(sheen) pixels show the paint's sheen")
        expect(fWorst < 2e-3, "the Fresnel the kernel used differs from PVC's by \(fWorst)")
    }

    if rendered, let r = renderer { annotate(r.image, setup: s, caption: caption) }

    test("the inset is on the picture: solid PVC under a paint layer as thick as the model says, to the pixel") {
        guard rendered, let img = renderer?.image else { expect(false, "render failed"); return }
        let r: CGRect = insetRect(height: img.height)
        let x: Int = Int(r.midX)
        let paintIndex: Int = s.design.legs[s.cutLimb].lowerPrims[0].paint
        let paintLab: SIMD3<Double> = linearSRGBToLab(SIMD3<Double>(s.design.paints[paintIndex].albedo))
        let pvcLab: SIMD3<Double> = linearSRGBToLab(SIMD3<Double>(plasticColour(s.design.paints[paintIndex])))
        var runs: [(String, Int)] = []
        for y in Int(r.minY) + 4..<Int(r.maxY) - 4 {
            let lab: SIMD3<Double> = linearSRGBToLab(pixelLinear(img, x, y))
            let name: String = simd_distance(lab, paintLab) < 3 ? "paint" : (simd_distance(lab, pvcLab) < 3 ? "pvc" : "other")
            if let last = runs.last, last.0 == name { runs[runs.count - 1].1 += 1 } else { runs.append((name, 1)) }
        }
        let sig: [(String, Int)] = runs.filter { $0.0 != "other" }
        let expected: Float = paintMicrometres / 1000 / insetMillimetresPerPixel(height: img.height)
        print("        down the inset's middle: " + sig.map { "\($0.0) \($0.1) px" }.joined(separator: ", ")
              + String(format: "  (paint should be %.1f px)", expected))
        expectEqual(sig.map { $0.0 }, ["paint", "pvc", "paint"])
        if sig.count == 3 {
            expect(abs(Float(sig[0].1) - expected) <= 1.5 && abs(Float(sig[2].1) - expected) <= 1.5,
                   "paint runs \(sig[0].1), \(sig[2].1) px; expected \(expected)")
            expect(sig[1].1 > 100, "the plastic under the paint should be solid across the cut: \(sig[1].1) px")
        }
    }

    test("the scale bar is true: 10 mm at the toy, projected, is the bar as painted") {
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
        print(String(format: "        10 mm: projected %.1f px, bar %.1f px, painted %d px; inset 0.5 mm: %.1f px, bar %.1f px",
                     projected, Float(bar.pixels), painted, insetExpected, Float(insetBar.pixels)))
        expect(abs(Float(bar.pixels) - projected) < 0.5, "bar \(bar.pixels) vs projected \(projected)")
        expect(abs(Float(painted) - projected) <= 2.5, "bar painted \(painted) px")
        expect(abs(Float(insetBar.pixels) - insetExpected) < 0.01)
    }

    test("the cut is on the leg, in view, left of the inset") {
        guard rendered, let img = renderer?.image else { expect(false, "render failed"); return }
        let c: Cut = cutFor(s)
        let p: SIMD2<Float> = project(c.centre, width: img.width, height: img.height, cam: s.camera)
        let r: CGRect = insetRect(height: img.height)
        expect(p.x > 0 && p.x < Float(r.minX) && p.y > 0 && p.y < Float(img.height), "cut point off the view: \(p)")
        expect(s.toy.sdf(c.centre).d < -0.5, "the cut's centre should be inside the leg")
    }
}
