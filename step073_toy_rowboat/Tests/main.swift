// Tests for step 73, the toy rowing boat. The boat is read off the toy as
// drawn and checked against the builder's 14 ft Whitehall-type numbers at
// 1:65; the mould (walls, draft, parting line, no undercut) against
// Protolabs' guidelines; the optics, the distance function, the size and
// the picture by the shared tests.
//
// TOY_MUTANT=dielectricZero|noSeam|wrongSize|noOars breaks the step on
// purpose; `make mutants` requires the suite to fail for each.

import CoreGraphics
import Foundation
import Metal
import simd

let mutant: Mutant = activeMutant
if mutant != .none { print("MUTANT: \(mutant.rawValue)") }

let design: BoatDesign = boatDesign(mutant)
let setup: StillSetup = stillSetup(design, mutant: mutant)
let toy: PosedToy = setup.toy
let renderer: ToyRenderer? = {
    guard let d = gpu.device else { return nil }
    return try? ToyRenderer(device: d, width: 1920, height: 1080, paints: design.paints, studio: setup.studio, mutant: mutant)
}()
// The drawing's own hull numbers, at the true scale (not the mutant's).
let hull = RowboatHull(scale: 1.0 / 65.0)
let hullOnly = PosedToy(segments: [toy.segments[0]], frames: [toy.frames[0]], paints: toy.paints, seams: false)

section("the boat: a 14 ft Whitehall-type rowboat at 1:65")

test("length 14 ft 2 in, beam 51 in, depth 18 in, at 1:65: 66.4 × 20.0 × 7.08 mm, as drawn") {
    let len = hullOnly.extent(along: SIMD3<Float>(1, 0, 0))
    let wid = hullOnly.extent(along: SIMD3<Float>(0, 0, 1))
    let L: Float = len.hi - len.lo
    let B: Float = wid.hi - wid.lo
    // Depth: the gunwale's top over the keel, amidships (where the sheer is lowest).
    let x: Float = hull.sheerLowX
    let wall = wallAt(toy, cut: Cut(centre: SIMD3<Float>(x, 0, 0), right: SIMD3<Float>(0, 0, 1), down: SIMD3<Float>(0, -1, 0)), y: 5)
    var top: Float = 12
    let zTop: Float = (wall.outer + wall.inner) / 2
    while top > 0 && hullOnly.sdf(SIMD3<Float>(x, top, zTop)).d > 0 { top -= 0.001 }
    var keel: Float = 3
    while keel > -1 && hullOnly.sdf(SIMD3<Float>(x, keel, 0)).d <= 0 { keel -= 0.001 }
    let D: Float = top - keel
    let s: Float = 10.0 / 65.0
    print(String(format: "        length %.2f mm (want %.2f), beam %.2f (want %.2f), depth %.2f (want %.2f); L/B %.2f (want %.2f)",
                 L, rowboatLengthCM * s, B, rowboatBeamCM * s, D, rowboatDepthCM * s, L / B, rowboatLengthCM / rowboatBeamCM))
    expect(abs(L - rowboatLengthCM * s) < 0.015 * rowboatLengthCM * s, "length \(L)")
    expect(abs(B - rowboatBeamCM * s) < 0.02 * rowboatBeamCM * s, "beam \(B)")
    expect(abs(D - rowboatDepthCM * s) < 0.04 * rowboatDepthCM * s, "depth \(D)")
}

test("the sheer rises to the bow and the transom; the keel rocker rises to the forefoot; the transom is narrower than the beam") {
    let mid: Float = hull.sheerY(hull.sheerLowX)
    let bow: Float = hull.sheerY(30)
    let stern: Float = hull.sheerY(-30)
    let ex = hullOnly.extent(along: SIMD3<Float>(1, 0, 0))
    // The transom's width at its top, from the drawn toy.
    var z: Float = 12
    let xt: Float = ex.lo + 0.6
    let yt: Float = hull.sheerY(xt) - 0.8
    while z > 0 && hullOnly.sdf(SIMD3<Float>(xt, yt, z)).d > 0 { z -= 0.002 }
    print(String(format: "        sheer %.2f mm amidships, %.2f near the bow, %.2f near the stern; transom %.1f mm wide at the top against a %.1f mm beam",
                 mid, bow, stern, 2 * z, 2 * hull.halfBeam))
    expect(bow > mid + 1.5 && stern > mid + 0.5 && bow > stern, "the sheer should spring most at the bow")
    expect(2 * z > 0.3 * 2 * hull.halfBeam && 2 * z < 0.65 * 2 * hull.halfBeam, "a wine-glass transom, under two-thirds of the beam")
}

test("three thwarts stand across the hollow, wall to wall, and the hollow is open between them") {
    let ts: [Float] = hull.thwartX
    expectEqual(ts.count, 3)
    var report: [String] = []
    for x in ts {
        let yTop: Float = hull.sheerY(x) - hull.thwartBelowSheer
        // Across the boat just under the thwart's top: solid from side to side.
        var gaps: Int = 0
        let w: Float = hull.halfWidthAtSheer(x) - 2.5
        for i in 0...100 {
            let z: Float = -w + 2 * w * Float(i) / 100
            if hullOnly.sdf(SIMD3<Float>(x, yTop - 0.3, z)).d > 0 { gaps += 1 }
        }
        // Just above it: air over the middle.
        let above: Float = hullOnly.sdf(SIMD3<Float>(x, yTop + 0.3, 0)).d
        report.append(String(format: "x %.1f: top %.2f, %d gaps under, %.2f mm clear above", x, yTop, gaps, above))
        expectEqual(gaps, 0)
        expect(above > 0.2, "air over the thwart at x = \(x)")
    }
    for x in [Float(6), -12] {
        let floorY: Float = 2.6
        expect(hullOnly.sdf(SIMD3<Float>(x, floorY + 1.0, 0)).d > 0.5, "the hollow should be open at x = \(x)")
    }
    print("        " + report.joined(separator: "; "))
}

test("a pair of oars rest in the rowlocks, blades flat and outboard (GPU)") {
    guard let r = renderer else { expect(false, "no GPU"); return }
    let oars: [Segment] = toy.segments.filter { $0.part == .oars }
    var blades: Int = 0
    for o in oars { for p in o.prims where p.tag == .blade { blades += 1 } }
    let rowlocks: Int = toy.segments[0].prims.filter { $0.tag == .rowlock }.count
    expectEqual(oars.count, 1)
    expectEqual(blades, 2)
    expectEqual(rowlocks, 2)
    // Probe the kernel at each blade's middle and at each rowlock, the
    // world points computed from the design, not from the drawn segments.
    let s: Float = 10.0 / 65.0
    let len: Float = oarLengthCM * s
    let outboard: Float = len * (1 - oarInboardShare)
    let bladeLen: Float = oarBladeLengthCM * s
    let ang: Float = oarRestAngleDegrees * .pi / 180
    let rx: Float = hull.rowlockX
    let rz: Float = hull.halfWidthAtSheer(rx) - 0.55
    let y: Float = hull.sheerY(rx) + 0.75
    var pts: [SIMD3<Float>] = []
    for side in [Float(-1), 1] {
        let dir = SIMD3<Float>(-sin(ang), 0, side * cos(ang))
        let pivot = SIMD3<Float>(rx, y, side * rz)
        pts.append(pivot + dir * (outboard - bladeLen / 2))
        pts.append(pivot)
    }
    guard let d = try? r.probe(pts, toy: toy) else { expect(false, "probe failed"); return }
    print(String(format: "        toy distance at the blades %.2f, %.2f mm; at the rowlocks %.2f, %.2f mm; blade %.1f mm out from the gunwale",
                 d[0].z, d[2].z, d[1].z, d[3].z, abs(pts[0].z) - rz))
    expect(d[0].z < -0.2 && d[2].z < -0.2, "no blade where the oars should be")
    expect(d[1].z < -0.2 && d[3].z < -0.2, "no oar in a rowlock")
    expect(abs(pts[0].z) - rz > 10, "the blades should be well outboard")
}

section("the mould: one that opens up and down (Protolabs' guidelines)")

test("walls \(hull.wall) mm, inside the moulder's range; the draft outside and in is at least \(draftMostDegrees)°") {
    let dr = draftOnCut(toy, cut: setup.cut)
    print(String(format: "        on the cut: wall %.2f mm; outside leans %.1f°, inside %.1f° from vertical", dr.thickness, dr.outer, dr.inner))
    expect(mouldWallRange.contains(dr.thickness), "wall \(dr.thickness) mm")
    expect(dr.outer >= draftMostDegrees && dr.inner >= draftMostDegrees, "draft \(dr.outer)°, \(dr.inner)°")
    // The thwarts' faces lean too: wider at the floor than at the top.
    let x: Float = hull.thwartX[1]
    let yTop: Float = hull.sheerY(x) - hull.thwartBelowSheer
    func face(_ y: Float) -> Float {
        var f: Float = x + 4
        while f > x && hullOnly.sdf(SIMD3<Float>(f, y, 0)).d > 0 { f -= 0.0005 }
        return f
    }
    let lean: Float = atan((face(yTop - 1.8) - face(yTop - 0.3)) / 1.5) * 180 / .pi
    print(String(format: "        rowing thwart's forward face leans %.2f° (design %.1f°)", lean, draftMostDegrees))
    expect(lean > draftMostDegrees * 0.8, "the thwart's face leans \(lean)°")
}

test("no undercut: below the parting line the hull only narrows downwards, so the lower half can let it go") {
    var worst: Float = 0
    for x in [Float(-20), -10, 0, 10, 20] {
        let seam: Float = hull.sheerY(x) - 0.5
        let c = Cut(centre: SIMD3<Float>(x, 0, 0), right: SIMD3<Float>(0, 0, 1), down: SIMD3<Float>(0, -1, 0))
        var last: Float = -.infinity
        var y: Float = 1.2
        while y < seam {
            let z: Float = wallAt(toy, cut: c, y: y).outer
            worst = min(worst, z - last)
            last = z
            y += 0.1
        }
    }
    print(String(format: "        the most the hull's side ever steps back going up: %.4f mm", -worst))
    expect(worst > -0.004, "an undercut of \(-worst) mm")
}

testSize(toy, name: "rowing boat")
testOptics(renderer, mutant: mutant)

section("the distance function")
testDistanceFunction(renderer, toy: toy, label: "the still")

var stations: [SeamStation] = []
for (x, side, name) in [(Float(0), Float(-1), "port amidships"), (-15, 1, "starboard aft"), (18, -1, "port forward")] {
    let y: Float = hull.sheerY(x) - 0.5
    let z: Float = side * (hull.halfWidthAtSheer(x) - 0.2)
    stations.append(SeamStation(segment: 0, at: SIMD3<Float>(x, y, z), outward: SIMD3<Float>(0, 0, side),
                                across: SIMD3<Float>(0, 1, 0), name: name, offsets: (0.12, 0.24), reach: 6))
}
if let oarIndex = toy.segments.firstIndex(where: { $0.part == .oars }) {
    let f: Frame = toy.frames[oarIndex]
    let ang: Float = oarRestAngleDegrees * .pi / 180
    let rx: Float = hull.rowlockX
    let rz: Float = hull.halfWidthAtSheer(rx) - 0.55
    let dir = SIMD3<Float>(-sin(ang), 0, cos(ang))
    let p: SIMD3<Float> = SIMD3<Float>(rx, f.o.y, rz) + dir * 8
    let out: SIMD3<Float> = simd_normalize(simd_cross(SIMD3<Float>(0, 1, 0), dir))
    stations.append(SeamStation(segment: oarIndex, at: p, outward: out, across: SIMD3<Float>(0, 1, 0), name: "starboard oar",
                                offsets: (0.15, 0.3), reach: 3))
}
testSeam(toy, stations: stations, where: "round the gunwale, and along each oar")

let len = toy.extent(along: SIMD3<Float>(1, 0, 0))
let picture: ToyImage? = testStillPicture(renderer, setup: setup, caption: boatCaption(lengthMM: len.hi - len.lo), mutant: mutant)

test("the inset shows the hollow hull to scale: two walls \(hull.wall) mm thick, air between, read off the pixels") {
    guard let img = picture else { expect(false, "render failed"); return }
    let y: Float = 3.0
    let p: CGPoint = insetPixel(SIMD3<Float>(setup.cut.centre.x, y, 0), cut: setup.cut, height: img.height)
    let runs = insetRuns(img, row: Int(p.y))
    let sig = runs.filter { $0.0 != "other" }
    let dr = wallAt(toy, cut: setup.cut, y: y)
    let wantPx: Float = (dr.outer - dr.inner) / insetMillimetresPerPixel(height: img.height)
    print("        along the inset's row at y = 3.0 mm: " + sig.map { "\($0.0) \($0.2) px" }.joined(separator: ", ")
          + String(format: "  (each wall should be %.1f px)", wantPx))
    expectEqual(sig.map { $0.0 }, ["air", "pvc", "air", "pvc", "air"])
    if sig.count == 5 {
        expect(abs(Float(sig[1].2) - wantPx) <= 1.5 && abs(Float(sig[3].2) - wantPx) <= 1.5, "walls \(sig[1].2), \(sig[3].2) px")
    }
}

finish()
