// Tests for step 74, the toy sailboat. The boat is read off the toy as drawn
// and checked against the class's numbers (Wikipedia; lasersailingtips) at
// 1:60; the mould (a centreplane split: sail thickness, draft, no undercut
// for a sideways pull) against Protolabs' guidelines; the optics, the
// distance function, the size and the picture by the shared tests.
//
// TOY_MUTANT=dielectricZero|noSeam|wrongSize|noSail breaks the step on
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
// The drawing's own numbers, at the true scale (not the mutant's).
let geo = DinghyGeometry(scale: 1.0 / 60.0)
let s: Float = 10.0 / 60.0
let hullOnly = PosedToy(segments: [toy.segments[0]], frames: [toy.frames[0]], paints: toy.paints, seams: false)

section("the boat: a single-handed dinghy at 1:60")

test("hull 4.23 m long and 1.37 m in the beam (Wikipedia), at 1:60: 70.5 × 22.8 mm, as drawn") {
    let len = hullOnly.extent(along: SIMD3<Float>(1, 0, 0))
    let wid = hullOnly.extent(along: SIMD3<Float>(0, 0, 1))
    let L: Float = len.hi - len.lo
    let B: Float = wid.hi - wid.lo
    print(String(format: "        length %.2f mm (want %.2f), beam %.2f (want %.2f)", L, dinghyLengthCM * s, B, dinghyBeamCM * s))
    expect(abs(L - dinghyLengthCM * s) < 0.015 * dinghyLengthCM * s, "length \(L)")
    expect(abs(B - dinghyBeamCM * s) < 0.02 * dinghyBeamCM * s, "beam \(B)")
}

test("one triangular sail, its sides the class's: luff 5.13, leech 5.57, foot 2.74 m; 6.99 of its 7.06 m² (the rest is roach)") {
    var sails: [Prim] = []
    for seg in toy.segments { for p in seg.prims where p.tag == .sail { sails.append(p) } }
    expectEqual(sails.count, 1)
    guard let p = sails.first else { return }
    let a = SIMD2<Float>(p.size.x, p.size.y)
    let b = SIMD2<Float>(p.size.z, p.size.w)
    let c = SIMD2<Float>(p.b.x, p.b.y)
    let foot: Float = simd_distance(a, b) / s / 100
    let leech: Float = simd_distance(b, c) / s / 100
    let luff: Float = simd_distance(c, a) / s / 100
    let e1: SIMD2<Float> = b - a
    let e2: SIMD2<Float> = c - a
    let cross: Float = e1.x * e2.y - e1.y * e2.x
    let area: Float = abs(cross) / 2 / (s * s) / 10_000
    print(String(format: "        luff %.3f m, leech %.3f m, foot %.3f m; area %.3f m² (class %.2f m²); the triangle is in the centreplane: normal %@",
                 luff, leech, foot, area, sailAreaM2, "\(p.n)"))
    expect(abs(luff - 5.13) < 0.01 && abs(leech - 5.57) < 0.01 && abs(foot - 2.74) < 0.01, "sides")
    expect(area < sailAreaM2 && area > 0.98 * sailAreaM2, "area \(area)")
    expect(abs(abs(p.n.z) - 1) < 1e-6, "the sail should lie in the centreplane")
}

test("the sail is drawn where it should be, solid, \(sailThickness) mm thick, and joined to the mast and boom (GPU)") {
    guard let r = renderer else { expect(false, "no GPU"); return }
    let c = geo.sailCorners
    let o = SIMD3<Float>(geo.mastX, geo.deckY(geo.mastX), 0)
    let centroid2: SIMD2<Float> = (c.tack + c.head + c.clew) / 3
    let centroid: SIMD3<Float> = o + SIMD3<Float>(centroid2.x, centroid2.y, 0)
    // Along the luff, the sail and mast are one: solid all the way up.
    var pts: [SIMD3<Float>] = [centroid]
    for i in 0..<40 {
        let t: Float = Float(i) / 39
        let y: Float = c.tack.y + 0.5 + (c.head.y - c.tack.y - 2) * t
        pts.append(o + SIMD3<Float>(c.tack.x - 0.15, y, 0))
    }
    guard let d = try? r.probe(pts, toy: toy) else { expect(false, "probe failed"); return }
    let th: Float = thicknessAcross(toy, x: centroid.x, y: centroid.y)
    var gaps: Int = 0
    for i in 1..<pts.count where d[i].z > 0 { gaps += 1 }
    print(String(format: "        distance at the sail's centroid %.3f mm; thickness there %.3f mm; %d gaps along the luff", d[0].z, th, gaps))
    expect(d[0].z < -0.5, "no sail at its centroid")
    expect(abs(th - sailThickness) < 0.01, "thickness \(th)")
    expect(mouldWallRange.contains(th), "a moulder could not fill \(th) mm")
    expectEqual(gaps, 0)
}

test("the mast stands 5.86 m above the deck (6.16 m of tubes, 0.30 m in the step), at 1:60, drawn") {
    let rig: Int = toy.segments.firstIndex { $0.part == .rig }!
    let mastOnly = PosedToy(segments: [Segment(name: "mast", part: .rig, prims: toy.segments[rig].prims.filter { $0.tag == .mast },
                                               blend: 0, seam: false)],
                            frames: [toy.frames[rig]], paints: toy.paints, seams: false)
    let top: Float = mastOnly.extent(along: SIMD3<Float>(0, 1, 0)).hi
    let above: Float = top - geo.deckY(geo.mastX)
    let want: Float = (mastTubesCM - mastInStepCM) * s
    print(String(format: "        mast top %.2f mm above the deck (want %.2f)", above, want))
    expect(abs(above - want) < 0.05, "mast \(above)")
}

section("the mould: split along the centreplane (Protolabs' guidelines)")

test("no undercut for a sideways pull: from the centreplane out, every line across meets solid then air, never solid again") {
    var undercuts: Int = 0
    var lines: Int = 0
    let ex = toy.extent(along: SIMD3<Float>(1, 0, 0))
    let ey = toy.extent(along: SIMD3<Float>(0, 1, 0))
    for i in 0..<70 {
        let x: Float = ex.lo + 0.3 + (ex.hi - ex.lo - 0.6) * Float(i) / 69
        for j in 0..<90 {
            let y: Float = 0.05 + (ey.hi - 0.1) * Float(j) / 89
            for side in [Float(-1), 1] {
                var wasAir: Bool = false
                var z: Float = 0
                while z < 13 {
                    let solid: Bool = toy.sdf(SIMD3<Float>(x, y, side * z)).d <= 0
                    if solid && wasAir { undercuts += 1; break }
                    if !solid { wasAir = true }
                    z += 0.05
                }
                lines += 1
            }
        }
    }
    print("        \(lines) lines across the toy, \(undercuts) undercut")
    expectEqual(undercuts, 0)
}

test("the faces each half slides along lean away from the parting plane: deck, bottom and transom at least \(draftLeastDegrees)°") {
    func slope(_ p0: SIMD3<Float>, _ dir: SIMD3<Float>, _ zs: (Float, Float)) -> Float {
        // How far the surface moves along `dir` between two points out from
        // the centreplane, from the drawn toy.
        func hit(_ z: Float) -> Float {
            var t: Float = -3
            while t < 3 && hullOnly.sdf(p0 + dir * t + SIMD3<Float>(0, 0, z)).d > 0 { t += 0.0005 }
            return t
        }
        return atan((hit(zs.1) - hit(zs.0)) / (zs.1 - zs.0)) * 180 / .pi
    }
    let deck: Float = slope(SIMD3<Float>(0, geo.deckY(0), 0), SIMD3<Float>(0, -1, 0), (1, 3))
    let bottom: Float = slope(SIMD3<Float>(geo.keelLowX, 0, 0), SIMD3<Float>(0, 1, 0), (1, 3))
    let ex = hullOnly.extent(along: SIMD3<Float>(1, 0, 0))
    let transom: Float = slope(SIMD3<Float>(ex.lo, 3.5, 0), SIMD3<Float>(1, 0, 0), (1, 3))
    print(String(format: "        deck leans %.2f°, bottom %.2f°, transom %.2f° from square to the parting plane", deck, bottom, transom))
    expect(deck >= draftLeastDegrees && bottom >= draftLeastDegrees && transom >= draftLeastDegrees, "draft")
}

testSize(toy, name: "sailboat")
testOptics(renderer, mutant: mutant)

section("the distance function")
testDistanceFunction(renderer, toy: toy, label: "the still")

let rig: Int = toy.segments.firstIndex { $0.part == .rig }!
let mastMidY: Float = geo.deckY(geo.mastX) + 40
testSeam(toy, stations: [
    SeamStation(segment: 0, at: SIMD3<Float>(0, geo.deckY(0), 0), outward: SIMD3<Float>(0, 1, 0), across: SIMD3<Float>(0, 0, 1),
                name: "deck crown", offsets: (0.25, 0.5), reach: 3),
    SeamStation(segment: 0, at: SIMD3<Float>(-20, geo.deckY(-20), 0), outward: SIMD3<Float>(0, 1, 0), across: SIMD3<Float>(0, 0, 1),
                name: "after deck", offsets: (0.25, 0.5), reach: 3),
    SeamStation(segment: rig, at: SIMD3<Float>(geo.mastX, mastMidY, 0), outward: SIMD3<Float>(1, 0, 0), across: SIMD3<Float>(0, 0, 1),
                name: "mast, forward face", offsets: (0.12, 0.24), reach: 3),
], where: "over the deck's crown and up the mast, in the centreplane")

test("…and along the keel, where the bottom's V meets it (a straight-sided fit)") {
    let x: Float = geo.keelLowX
    func out(_ z: Float) -> Float {
        var y: Float = -1
        while y < 3 && toy.sdf(SIMD3<Float>(x, y, z)).d > 0 { y += 0.0002 }
        return -y
    }
    let h0: Float = out(0)
    let h1: Float = (out(0.25) + out(-0.25)) / 2
    let h2: Float = (out(0.5) + out(-0.5)) / 2
    let m: Float = (h1 - h2) / 0.25
    let ridge: Float = h0 - (h1 + m * 0.25)
    print(String(format: "        ridge under the keel %.3f mm", ridge))
    expect(ridge > 0.6 * seamHeight && ridge < 1.4 * seamHeight, "keel ridge \(ridge)")
}

let len = toy.extent(along: SIMD3<Float>(1, 0, 0))
let picture: ToyImage? = testStillPicture(renderer, setup: setup, caption: boatCaption(lengthMM: len.hi - len.lo), mutant: mutant)

test("the inset shows the solid hull and the sail to scale, read off the pixels") {
    guard let img = picture else { expect(false, "render failed"); return }
    let mm: Float = insetMillimetresPerPixel(height: img.height)
    let hullRow: CGPoint = insetPixel(SIMD3<Float>(setup.cut.centre.x, 1.6, 0), cut: setup.cut, height: img.height)
    let hullRuns = insetRuns(img, row: Int(hullRow.y))
    let sailRow: CGPoint = insetPixel(SIMD3<Float>(setup.cut.centre.x, 25.5, 0), cut: setup.cut, height: img.height)
    let sailRuns = insetRuns(img, row: Int(sailRow.y))
    // The world height the row's pixel centres sample.
    let rowY: Float = insetPoint(hullRow.x, CGFloat(Int(hullRow.y)) + 0.5, cut: setup.cut, height: img.height).y
    var zEdge: Float = 0
    while toy.sdf(SIMD3<Float>(setup.cut.centre.x, rowY, zEdge)).d <= 0 { zEdge += 0.001 }
    let hullPx: Float = 2 * zEdge / mm
    let sailPx: Float = thicknessAcross(toy, x: setup.cut.centre.x, y: 25.5) / mm
    print("        hull row: " + hullRuns.map { "\($0.0) \($0.2)" }.joined(separator: ", ")
          + String(format: " (solid %.1f px)", hullPx) + "; sail row: " + sailRuns.map { "\($0.0) \($0.2)" }.joined(separator: ", ")
          + String(format: " (sail %.1f px)", sailPx))
    expectEqual(hullRuns.map { $0.0 }, ["air", "pvc", "air"])
    expectEqual(sailRuns.map { $0.0 }, ["air", "pvc", "air"])
    if hullRuns.count == 3 { expect(abs(Float(hullRuns[1].2) - hullPx) <= 1.5, "hull \(hullRuns[1].2) px") }
    if sailRuns.count == 3 { expect(abs(Float(sailRuns[1].2) - sailPx) <= 1.5, "sail \(sailRuns[1].2) px") }
}

finish()
