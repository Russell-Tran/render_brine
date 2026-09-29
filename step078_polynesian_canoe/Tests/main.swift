// Tests for step 78, the waʻa kaulua. The canoe's size is measured off the
// kernel's own distance function and held to PVS's plan (length overall,
// beam, waterline length, sail area); its two hulls and eight ʻiako are
// counted; it floats by Archimedes — the displaced sea water, measured two
// independent ways, weighs what PVS says the loaded canoe weighs — and
// nothing but the hulls and the paddle's blade is in the water; the
// distance function is held to the definition of a distance; the scale
// bars to the camera and the inset; and the draft is measured off the
// finished picture.
//
// CANOE_MUTANT=floatsWrong|wrongSize|singleHull breaks the step on
// purpose; `make mutants` requires the suite to fail for each.

import CoreGraphics
import Foundation
import Metal
import simd

let mutant: Mutant = activeMutant
if mutant != .none { print("MUTANT: \(mutant.rawValue)") }

let setup: StillSetup = stillSetup(mutant)
let renderer: CanoeRenderer? = {
    guard let d = try? findDevice() else { return nil }
    return try? CanoeRenderer(device: d, width: 1920, height: 1080, sky: setup.sky, mutant: mutant)
}()

// Float the canoe exactly as main.swift does.
let grid: ColumnGrid? = renderer.flatMap { try? columnGrid($0, step: 0.01, mutant: mutant) }
let flotation: Flotation? = grid.map { solveDraft($0, mutant: mutant) }
let sink: Float = flotation.map { placedDraft($0, mutant: mutant) } ?? 0

section("the canoe, against PVS's plan")

/// How far the canoe reaches along a direction: sphere-trace many parallel
/// lines in from outside and take the farthest hit. `channel` picks the
/// probe's distance: 0 the whole canoe, 2 the hulls and manu.
func reach(along axis: SIMD3<Float>, across u: SIMD3<Float>, _ v: SIMD3<Float>, uRange: ClosedRange<Float>,
           vRange: ClosedRange<Float>, channel: Int) -> Float {
    guard let r = renderer else { return .nan }
    let n: Int = 64
    let far: Float = 40
    var starts: [SIMD3<Float>] = []
    for i in 0..<n {
        for j in 0..<n {
            let a: Float = uRange.lowerBound + (uRange.upperBound - uRange.lowerBound) * Float(i) / Float(n - 1)
            let b: Float = vRange.lowerBound + (vRange.upperBound - vRange.lowerBound) * Float(j) / Float(n - 1)
            let o: SIMD3<Float> = axis * far
            let du: SIMD3<Float> = u * a
            let dv: SIMD3<Float> = v * b
            starts.append(o + du + dv)
        }
    }
    var t: [Float] = Array(repeating: 0, count: starts.count)
    var done: [Bool] = Array(repeating: false, count: starts.count)
    for _ in 0..<400 {
        var pts: [SIMD3<Float>] = []
        pts.reserveCapacity(starts.count)
        for i in 0..<starts.count {
            let back: SIMD3<Float> = axis * t[i]
            pts.append(starts[i] - back)
        }
        guard let pr = try? r.probe(pts, sink: 0) else { return .nan }
        for i in 0..<starts.count where !done[i] {
            let d: Float = channel == 0 ? pr[i].x : pr[i].z
            if d < 0.0002 { done[i] = true } else { t[i] += d * 0.95 }
            if t[i] > 2 * far { done[i] = true; t[i] = .infinity }
        }
    }
    let best: Float = t.min() ?? .infinity
    return far - best
}

let X = SIMD3<Float>(1, 0, 0), Y = SIMD3<Float>(0, 1, 0), Z = SIMD3<Float>(0, 0, 1)

test("length overall 62 ft 4 in (18.999 m), manu tip to manu tip, to 1 cm") {
    let yr: ClosedRange<Float> = 0...5
    let zr: ClosedRange<Float> = -3...3
    let bow: Float = reach(along: X, across: Y, Z, uRange: yr, vRange: zr, channel: 2)
    let stern: Float = reach(along: -X, across: Y, Z, uRange: yr, vRange: zr, channel: 2)
    let loa: Float = bow + stern
    print(String(format: "        drawn %.4f m (%.2f ft); PVS %.4f m", loa, Double(loa) / metresPerFoot, pvsLengthOverall))
    let loaErr: Double = abs(Double(loa) - pvsLengthOverall)
    expect(loaErr < 0.01, "length overall \(loa)")
}

test("beam 17 ft 6 in (5.334 m) over the ʻiako, to 1 cm") {
    let xr: ClosedRange<Float> = -12...11
    let yr: ClosedRange<Float> = 1.2...1.9   // round the ʻiako, where the beam is
    let port: Float = reach(along: -Z, across: X, Y, uRange: xr, vRange: yr, channel: 0)
    let starboard: Float = reach(along: Z, across: X, Y, uRange: xr, vRange: yr, channel: 0)
    let beam: Float = port + starboard
    print(String(format: "        drawn %.4f m; PVS %.4f m", beam, pvsBeam))
    let beamErr: Double = abs(Double(beam) - pvsBeam)
    expect(beamErr < 0.01, "beam \(beam)")
}

test("two hulls, alike, side by side: cut across at midships below the waterline, two separate sections") {
    guard let r = renderer else { expect(false, "no GPU"); return }
    var pts: [SIMD3<Float>] = []
    for i in 0..<6000 {
        let z: Float = Float(i) * 0.001 - 3
        pts.append(SIMD3<Float>(0, -0.1, z))
    }
    guard let pr = try? r.probe(pts, sink: sink) else { expect(false, "probe failed"); return }
    var runs: [(Float, Float)] = []
    var inside: Bool = false
    for i in 0..<pts.count {
        let wet: Bool = pr[i].w < 0 && Int(pr[i].y) == 2
        if wet && !inside { runs.append((pts[i].z, pts[i].z)) }
        if wet { runs[runs.count - 1].1 = pts[i].z }
        inside = wet
    }
    print("        sections across: " + runs.map { String(format: "%.3f…%.3f m", $0.0, $0.1) }.joined(separator: ", "))
    expectEqual(runs.count, 2)
    if runs.count == 2 {
        let w0: Float = runs[0].1 - runs[0].0
        let w1: Float = runs[1].1 - runs[1].0
        expect(abs(w0 - w1) < 0.003, "hulls differ: \(w0) vs \(w1)")
        expect(runs[0].1 < -0.5 && runs[1].0 > 0.5, "hulls not apart")
    }
}

test("eight ʻiako (PVS), counted along the canoe outboard of the hulls") {
    guard let r = renderer else { expect(false, "no GPU"); return }
    let s: Float = canoeScale(mutant)
    var pts: [SIMD3<Float>] = []
    for i in 0..<22_000 {
        let fi: Float = Float(i)
        let mm: Float = fi * 0.001
        let u: Float = mm - 12
        let x: Float = u * s
        let y: Float = (gunwaleHeight + crossbeamRadius) * s
        let z: Float = (0.3 - crossbeamHalfSpan) * s
        pts.append(SIMD3<Float>(x, y, z))
    }
    guard let pr = try? r.parts(pts, sink: 0) else { expect(false, "probe failed"); return }
    var count: Int = 0
    var inside: Bool = false
    for p in pr {
        let hit: Bool = p.y < 0
        if hit && !inside { count += 1 }
        inside = hit
    }
    print("        \(count) crossbeams")
    expectEqual(count, crossbeamCount)
}

test("540 sq ft of sail (PVS), measured over both sails' planes, to 0.5%; the foresail the larger (Kāne)") {
    guard let r = renderer else { expect(false, "no GPU"); return }
    let s: Float = canoeScale(mutant)
    let step: Float = 0.01
    var areas: [Double] = []
    for sail in [foresail, aftsail] {
        let o = SIMD3<Float>(sail.mastX, deckTop, 0) * s
        let U = SIMD3<Float>(cos(sailYaw), 0, -sin(sailYaw))
        var pts: [SIMD3<Float>] = []
        var u: Float = -8 * s
        while u < 2 * s {
            var v: Float = step / 2
            while v < 15 * s {
                pts.append(o + U * u + SIMD3<Float>(0, v, 0))
                v += step
            }
            u += step
        }
        guard let pr = try? r.parts(pts, sink: 0) else { expect(false, "probe failed"); return }
        let n: Int = pr.filter { $0.x < 0 }.count
        let cell: Double = Double(step) * Double(step)
        areas.append(Double(n) * cell)
    }
    let total: Double = areas.reduce(0, +)
    let sqft: Double = total / (metresPerFoot * metresPerFoot)
    print(String(format: "        foresail %.2f m², aft sail %.2f m²: %.2f m² = %.1f sq ft; PVS 540 sq ft", areas[0], areas[1], total, sqft))
    expect(abs(total - pvsSailArea) / pvsSailArea < 0.005, "sail area \(total)")
    expect(areas[0] > areas[1], "the foresail is not the larger")
}

test("the manu rise higher at the stern than at the bow (Kāne)") {
    guard let r = renderer else { expect(false, "no GPU"); return }
    let s: Float = canoeScale(mutant)
    var pts: [SIMD3<Float>] = []
    for i in 0..<200 {
        for j in 0..<300 {
            let x: Float = 7 + Float(i) * 0.013
            let y: Float = Float(j) * 0.015
            pts.append(SIMD3<Float>(x * s, y * s, hullCentreZ * s))
            pts.append(SIMD3<Float>(-x * s, y * s, hullCentreZ * s))
        }
    }
    guard let pr = try? r.probe(pts, sink: 0) else { expect(false, "probe failed"); return }
    var bow: Float = 0, stern: Float = 0
    for i in 0..<pts.count where pr[i].z < 0 {
        if pts[i].x > 0 { bow = max(bow, pts[i].y) } else { stern = max(stern, pts[i].y) }
    }
    print(String(format: "        tips above the keel: bow %.2f m, stern %.2f m", bow, stern))
    expect(stern > bow + 0.5)
}

section("floating: Archimedes")

test("the solver's balance: ρ_sea × V(draft) = M, to 0.1 kg") {
    guard let f = flotation else { expect(false, "no flotation"); return }
    print(String(format: "        M %.2f kg; ρ %.1f kg/m³; V %.5f m³; draft %.4f m; ρV − M = %.4f kg",
                 canoeMass, seaWaterDensity, f.displaced, f.draft, f.imbalance))
    expect(abs(f.imbalance) < 0.1)
}

test("as drawn, the sea water pushed aside weighs the loaded canoe: a second grid (1.5 cm, shifted), to 0.5%") {
    guard let r = renderer else { expect(false, "no GPU"); return }
    guard let g2 = try? columnGrid(r, step: 0.015, offset: 0.25, mutant: mutant) else { expect(false, "columns failed"); return }
    let v: Double = g2.volume(below: sink)
    let err: Double = (seaWaterDensity * v - canoeMass) / canoeMass
    print(String(format: "        drawn draft %.4f m: %.4f m³ → %.1f kg of sea water for %.1f kg of canoe (%+.2f%%)",
                 sink, v, seaWaterDensity * v, canoeMass, err * 100))
    expect(abs(err) < 0.005, "out of balance by \(err * 100)%")
}

// Voxels below the drawn waterline, 2.5 cm on a side: shared by the next two tests.
let voxelStep: Float = 0.025
let voxels: (points: [SIMD3<Float>], probes: [SIMD4<Float>])? = {
    guard let r = renderer else { return nil }
    let s: Float = canoeScale(mutant)
    var pts: [SIMD3<Float>] = []
    let nx: Int = Int(21.6 * s / voxelStep)
    let nz: Int = Int(5.0 * s / voxelStep)
    let ny: Int = Int(1.6 * s / voxelStep)
    pts.reserveCapacity(nx * ny * nz)
    for i in 0..<nx {
        let x: Float = -11.8 * s + (Float(i) + 0.5) * voxelStep
        for k in 0..<nz {
            let z: Float = -2.5 * s + (Float(k) + 0.5) * voxelStep
            for j in 0..<ny {
                let y: Float = -(Float(j) + 0.5) * voxelStep
                pts.append(SIMD3<Float>(x, y, z))
            }
        }
    }
    guard let pr = try? r.probe(pts, sink: sink) else { return nil }
    return (pts, pr)
}()

test("counted another way — voxels of the canoe below the drawn waterline — the same weight, to 1.5%") {
    guard let vx = voxels else { expect(false, "no voxels"); return }
    var n: Int = 0
    for p in vx.probes where p.x < 0 { n += 1 }
    let cell: Double = Double(voxelStep) * Double(voxelStep) * Double(voxelStep)
    let v: Double = Double(n) * cell
    let err: Double = (seaWaterDensity * v - canoeMass) / canoeMass
    print(String(format: "        %d voxels = %.3f m³ → %.1f kg (%+.2f%%)", n, v, seaWaterDensity * v, err * 100))
    expect(abs(err) < 0.015, "out of balance by \(err * 100)%")
}

test("nothing but the hulls and the paddle's blade is in the water, and the hulls keep their freeboard") {
    guard let vx = voxels else { expect(false, "no voxels"); return }
    var dry: Int = 0
    for p in vx.probes where p.x < 0 && p.w >= 0 { dry += 1 }
    let freeboard: Float = gunwaleHeight * canoeScale(mutant) - sink
    print(String(format: "        %d voxels of dry parts under water; freeboard %.3f m", dry, freeboard))
    expectEqual(dry, 0)
    expect(freeboard > 0.3, "freeboard \(freeboard)")
}

test("the fitted hull floats as PVS's plan says: draft 2 ft 6 in to 2%, 54 ft on the waterline to 1%") {
    guard let r = renderer, let lwl = try? waterlineLength(r, sink: sink, mutant: mutant) else { expect(false, "no GPU"); return }
    let d: Double = Double(sink)
    print(String(format: "        drawn draft %.4f m (%.3f ft), PVS %.4f m; waterline %.3f m (%.2f ft), PVS %.3f m",
                 d, d / metresPerFoot, pvsDraft, lwl, Double(lwl) / metresPerFoot, pvsLengthWaterline))
    expect(abs(d - pvsDraft) / pvsDraft < 0.02, "draft \(d)")
    let lwlErr: Double = abs(Double(lwl) - pvsLengthWaterline) / pvsLengthWaterline
    expect(lwlErr < 0.01, "waterline \(lwl)")
}

section("the distance function")

test("outside every surface, no distance claims more room than the ray allows") {
    guard let r = renderer else { expect(false, "no GPU"); return }
    var rng = SystemRandomNumberGenerator()
    let s: Float = canoeScale(mutant)
    var a: [SIMD3<Float>] = []
    for i in 0..<240_000 {
        switch i % 4 {
        case 0:
            a.append(SIMD3<Float>(Float.random(in: -12.5...10.5, using: &rng), Float.random(in: -1.2...16, using: &rng),
                                  Float.random(in: -3.5...3.5, using: &rng)) * s)
        case 1:
            // Round the hulls and the manu.
            a.append(SIMD3<Float>(Float.random(in: -10...10, using: &rng), Float.random(in: -1...4, using: &rng),
                                  Float.random(in: -2.6...2.6, using: &rng)) * s)
        case 2:
            // Round the ʻiako, the deck and the paddle.
            a.append(SIMD3<Float>(Float.random(in: -11.8...6.5, using: &rng), Float.random(in: 0...3, using: &rng),
                                  Float.random(in: -2.9...2.9, using: &rng)) * s)
        default:
            // Round the sails and spars.
            a.append(SIMD3<Float>(Float.random(in: -6...6.5, using: &rng), Float.random(in: 1.5...15.5, using: &rng),
                                  Float.random(in: -0.6...2.4, using: &rng)) * s)
        }
    }
    var worst: Float = 0
    var mats: Set<Int> = []
    for step in [Float(0.002), 0.02, 0.2] {
        let b: [SIMD3<Float>] = a.map { p in
            let d: SIMD3<Float> = simd_normalize(SIMD3<Float>(Float.random(in: -1...1, using: &rng), Float.random(in: -1...1, using: &rng),
                                                              Float.random(in: -1...1, using: &rng)))
            return p + d * step
        }
        guard let pa = try? r.probe(a, sink: 0), let pb = try? r.probe(b, sink: 0) else { expect(false, "probe failed"); return }
        for i in 0..<a.count {
            guard pa[i].x > 0, pb[i].x > 0 else { continue }
            mats.insert(Int(pa[i].y))
            let dist: Float = simd_distance(a[i], b[i])
            if Int(pa[i].y) == Int(pb[i].y) { worst = max(worst, abs(pa[i].x - pb[i].x) / dist) }
            if pa[i].w > 0 && pb[i].w > 0 { worst = max(worst, abs(pa[i].w - pb[i].w) / dist) }
            if pa[i].z > 0 && pb[i].z > 0 { worst = max(worst, abs(pa[i].z - pb[i].z) / dist) }
        }
    }
    print(String(format: "        worst |Δd|/|Δp| outside %.3f (the ray allows %.3f); materials seen %@", worst, 1 / stepScale,
                 mats.sorted().map { String($0) }.joined(separator: ",")))
    let claimed: Float = worst * stepScale
    expect(claimed <= 1, "oversteps: \(worst)")
    for m in 2...8 { expect(mats.contains(m), "material \(m) never sampled") }
}

test("outside the hulls, the distance never exceeds the distance to the nearest point found inside them") {
    // A direct check of the lower bound: for points outside, d must not be
    // larger than the distance to any point known to be inside.
    guard let r = renderer, let vx = voxels else { expect(false, "no GPU"); return }
    var inside: [SIMD3<Float>] = []
    for (i, p) in vx.probes.enumerated() where p.w < 0 && i % 7 == 0 { inside.append(vx.points[i]) }
    var rng = SystemRandomNumberGenerator()
    var outside: [SIMD3<Float>] = []
    for _ in 0..<3000 {
        let q: SIMD3<Float> = inside[Int.random(in: 0..<inside.count, using: &rng)]
        let d: SIMD3<Float> = simd_normalize(SIMD3<Float>(Float.random(in: -1...1, using: &rng), Float.random(in: -1...1, using: &rng),
                                                          Float.random(in: -1...1, using: &rng)))
        outside.append(q + d * Float.random(in: 0.05...0.6, using: &rng))
    }
    guard let po = try? r.probe(outside, sink: sink) else { expect(false, "probe failed"); return }
    var worst: Float = -1
    for (i, p) in outside.enumerated() where po[i].w > 0 {
        var nearest: Float = .infinity
        for q in inside { nearest = min(nearest, simd_distance(p, q)) }
        worst = max(worst, po[i].w - nearest)
    }
    print(String(format: "        %d inside points; worst (claimed − nearest inside) %.4f m", inside.count, worst))
    expect(worst <= 0.0005, "a distance overstates: \(worst)")
}

var rendered: Bool = false
var section_: SectionInset? = nil
if let r = renderer, let f = flotation, (try? r.render(camera: setup.camera, samples: 1, sink: sink)) != nil {
    rendered = true
    section_ = try? sampleSection(r, sink: sink)
    let keel: [SIMD3<Float>] = (try? keelLine(r, sink: sink, mutant: mutant)) ?? []
    if let sec = section_ { annotate(r.image, setup: setup, flotation: f, sink: sink, section: sec, keel: keel, mutant: mutant) }
}

section("the picture")

test("the scale bars are true: 5 m at the canoe's middle, projected; 1 m in the inset") {
    guard rendered, let img = renderer?.image else { expect(false, "render failed"); return }
    let a: SIMD2<Float> = project(setup.centre, width: img.width, height: img.height, cam: setup.camera)
    let b: SIMD2<Float> = project(setup.centre + setup.camera.right * mainBarMetres, width: img.width, height: img.height, cam: setup.camera)
    let projected: Float = simd_distance(a, b)
    let bar: ScaleBar = mainScaleBar(setup, width: img.width, height: img.height)
    let insetBar: ScaleBar = insetScaleBar(height: img.height)
    let insetWant: Float = insetBarMetres * insetPixelsPerMetre(height: img.height)
    var painted: Int = 0
    let yy: Int = Int(bar.y + 1.5)
    for x in max(Int(bar.x) - 4, 0)..<min(Int(bar.x + bar.pixels) + 5, img.width) {
        let p: SIMD4<UInt8> = img.rgba(x, yy)
        if Int(p.x) + Int(p.y) + Int(p.z) > 3 * 235 { painted += 1 }
    }
    print(String(format: "        5 m: projected %.1f px, bar %.1f px, painted %d px; 1 m: %.1f px, bar %.1f px",
                 projected, Float(bar.pixels), painted, insetWant, Float(insetBar.pixels)))
    let barPx: Float = Float(bar.pixels)
    expect(abs(barPx - projected) < 0.5)
    let paintedPx: Float = Float(painted)
    expect(abs(paintedPx - projected) <= 2.5)
    let insetPx: Float = Float(insetBar.pixels)
    expect(abs(insetPx - insetWant) < 0.01)
}

test("measured off the inset: from the waterline to the keel is the drawn draft, to 2 px") {
    guard rendered, let img = renderer?.image else { expect(false, "render failed"); return }
    let k: Float = insetPixelsPerMetre(height: img.height)
    // Down the starboard hull's centre, where nothing is drawn over it: the
    // waterline is the blue rule; the keel is the last hull pixel.
    let col: Int = Int(insetPoint(z: hullCentreZ * canoeScale(mutant), y: 0, height: img.height).x)
    let r: CGRect = insetRect(height: img.height)
    var rule: Int = -1, keel: Int = -1
    for y in Int(r.minY) + 4..<Int(r.maxY) - 4 {
        let p: SIMD4<UInt8> = img.rgba(col, y)
        let pr: Int = Int(p.x), pg: Int = Int(p.y), pb: Int = Int(p.z)
        let nearR: Bool = abs(pr - 5) < 30
        let nearG: Bool = abs(pg - 84) < 30
        let nearB: Bool = abs(pb - 158) < 30
        let isRule: Bool = nearR && nearG && nearB
        if isRule && rule < 0 { rule = y }
        let isHull: Bool = Int(p.x) < 90 && Int(p.y) < 70 && Int(p.z) < 90
        // The keel: the last hull pixel of the run below the rule.
        if rule >= 0 && y > rule + 2 {
            if isHull { keel = y } else if keel >= 0 { break }
        }
    }
    let drawnPx: Float = Float(keel - rule)
    let wantPx: Float = sink * k
    print(String(format: "        waterline at row %d, keel at row %d: %.0f px = %.3f m; drawn draft %.3f m = %.1f px",
                 rule, keel, drawnPx, drawnPx / k, sink, wantPx))
    expect(rule > 0 && keel > rule)
    expect(abs(drawnPx - wantPx) <= 2.0, "the inset shows \(drawnPx / k) m")
}

func pictureTest() {
    guard rendered, let img = renderer?.image else { expect(false, "render failed"); return }
    var counts: [Int: Int] = [:]
    let rect: CGRect = insetRect(height: img.height)
    var overInset: Int = 0
    for y in 0..<img.height {
        for x in 0..<img.width {
            let m: Int = Int(img.seen(x, y).x)
            counts[m, default: 0] += 1
            if m >= 2 && rect.insetBy(dx: -30, dy: -60).contains(CGPoint(x: x, y: y)) { overInset += 1 }
        }
    }
    let total: Double = Double(img.width * img.height)
    func pc(_ m: Int) -> Double { Double(counts[m] ?? 0) / total * 100 }
    print(String(format: "        sky %.1f%%, sea %.1f%%, hulls %.2f%%, sails %.1f%%, hulls seen through the sea %.2f%%; near the inset %d px",
                 pc(0), pc(1), pc(2), pc(7), pc(12), overInset))
    expect(pc(0) > 10 && pc(1) > 10, "sky or sea missing")
    expect(pc(2) > 1 && pc(7) > 2.5, "the canoe is too small")
    expect(pc(12) > 0.2, "the hulls are not seen below the waterline")
    expect(overInset == 0, "the inset covers the canoe")
}

test("the picture shows the canoe, sky and sea, the hulls through the water, and the inset clear of the canoe", pictureTest)

finish()
