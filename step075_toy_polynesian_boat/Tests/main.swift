// Tests for step 75, the toy voyaging canoe (waʻa kaulua). The canoe is read
// off the toy as drawn and checked against PVS's published Hōkūleʻa
// figures at 1:300; the two mouldings (no undercut for the platform's
// up-and-down mould, nor for each rig's flat one) against Protolabs'
// guidelines; the optics, the distance function, the size and the picture
// by the shared tests.
//
// TOY_MUTANT=dielectricZero|noSeam|wrongSize|singleHull breaks the step on
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
let geo = CanoeGeometry(scale: 1.0 / 300.0)
let s: Float = 1000.0 / 300.0
func only(_ tags: Set<PrimTag>, segment i: Int) -> PosedToy {
    var seg: Segment = toy.segments[i]
    seg.prims = seg.prims.filter { tags.contains($0.tag) }
    seg.patches = []
    return PosedToy(segments: [seg], frames: [toy.frames[i]], paints: toy.paints, seams: false)
}
let hullsAndManu: PosedToy = only([.hull, .manu], segment: 0)
let beams: PosedToy = only([.crossbeam], segment: 0)
var platformNoSweep: Segment = toy.segments[0]
platformNoSweep.prims = platformNoSweep.prims.filter { $0.tag != .sweep }
let platform = PosedToy(segments: [platformNoSweep], frames: [toy.frames[0]], paints: toy.paints, seams: false)

section("the canoe: Hōkūleʻa's proportions (PVS) at 1:300")

test("64 ft 9 in overall, manu to manu, and 19 ft 8 in across the ʻiako: 65.8 × 20.0 mm, as drawn") {
    let len = hullsAndManu.extent(along: SIMD3<Float>(1, 0, 0))
    let wid = beams.extent(along: SIMD3<Float>(0, 0, 1))
    let L: Float = len.hi - len.lo
    let B: Float = wid.hi - wid.lo
    print(String(format: "        length %.2f mm (want %.2f), beam %.2f (want %.2f)", L, canoeLengthM * s, B, canoeBeamM * s))
    expect(abs(L - canoeLengthM * s) < 0.015 * canoeLengthM * s, "length \(L)")
    expect(abs(B - canoeBeamM * s) < 0.015 * canoeBeamM * s, "beam \(B)")
}

test("two hulls, each 4.9 ft wide and 6.6 ft deep, their centres 14.2 ft apart, with open water between them (GPU)") {
    guard let r = renderer else { expect(false, "no GPU"); return }
    let hulls: Int = toy.segments[0].prims.filter { $0.tag == .hull }.count
    expectEqual(hulls, 2)
    // Across the canoe just under the gunwales, between two ʻiako: solid,
    // air, solid.
    let y: Float = geo.hullDepth - 0.7
    var pts: [SIMD3<Float>] = []
    for i in 0..<400 {
        let z: Float = -geo.halfBeam + 2 * geo.halfBeam * Float(i) / 399
        pts.append(SIMD3<Float>(3.0, y, z))
    }
    guard let d = try? r.probe(pts, toy: toy) else { expect(false, "probe failed"); return }
    var runs: [(Bool, Int, Int)] = []
    for (i, v) in d.enumerated() {
        let solid: Bool = v.z < 0
        if let last = runs.last, last.0 == solid { runs[runs.count - 1].2 = i } else { runs.append((solid, i, i)) }
    }
    let dz: Float = 2 * geo.halfBeam / 399
    let solids = runs.filter { $0.0 }
    let widths: [Float] = solids.map { Float($0.2 - $0.1 + 1) * dz }
    let centres: [Float] = solids.map { -geo.halfBeam + Float($0.1 + $0.2) / 2 * dz }
    print(String(format: "        across at %.1f mm up: %d solid runs, widths %@ mm, centres %@ mm", y, solids.count,
                 widths.map { String(format: "%.2f", $0) }.joined(separator: ", "), centres.map { String(format: "%.2f", $0) }.joined(separator: ", ")))
    expectEqual(solids.count, 2)
    if solids.count == 2 {
        let spacing: Float = centres[1] - centres[0]
        expect(abs(spacing - hullSpacingFt * metresPerFoot * s) < 0.2, "hull spacing \(spacing)")
        expect(abs(widths[0] - hullWidthFt * metresPerFoot * s) < 0.3, "hull width \(widths[0])")
    }
    var keel: Float = 2
    while keel > -1 && toy.sdf(SIMD3<Float>(0, keel, geo.hullOffset)).d <= 0 { keel -= 0.001 }
    var top: Float = geo.hullDepth - 1
    let zOut: Float = geo.hullOffset + geo.hullHalfWidth - 0.4
    while top < 12 && hullsAndManu.sdf(SIMD3<Float>(0, top, zOut)).d <= 0 { top += 0.001 }
    print(String(format: "        hull depth, keel to gunwale, %.2f mm (want %.2f)", top - keel, hullDepthFt * metresPerFoot * s))
    expect(abs(top - keel - hullDepthFt * metresPerFoot * s) < 0.25, "hull depth \(top - keel)")
}

test("eight ʻiako join the hulls: along each, solid from one end to the other (GPU)") {
    guard let r = renderer else { expect(false, "no GPU"); return }
    let n: Int = toy.segments[0].prims.filter { $0.tag == .crossbeam }.count
    expectEqual(n, 8)
    var broken: Int = 0
    for f in iakoFromBow {
        let x: Float = geo.fromBow(f)
        let pts: [SIMD3<Float>] = (0..<200).map { i in
            SIMD3<Float>(x, geo.iakoY, -geo.halfBeam + 0.6 + (2 * geo.halfBeam - 1.2) * Float(i) / 199)
        }
        guard let d = try? r.probe(pts, toy: toy) else { expect(false, "probe failed"); return }
        if d.contains(where: { $0.z > 0 }) { broken += 1 }
    }
    print("        \(broken) of 8 ʻiako broken")
    expectEqual(broken, 0)
}

test("the manu rise 4.0 ft at the bow and 6.0 ft at the stern above the gunwale: higher at the stern") {
    let manus: [Prim] = toy.segments[0].prims.filter { $0.tag == .manu }
    expect(manus.count == 4, "\(manus.count) manu")
    var bowRise: Float = 0
    var sternRise: Float = 0
    for m in manus {
        let one = PosedToy(segments: [Segment(name: "m", part: .hull, prims: [m], blend: 0, seam: false)], frames: [Frame.identity],
                           paints: toy.paints, seams: false)
        let top: Float = one.extent(along: SIMD3<Float>(0, 1, 0)).hi - geo.hullDepth
        let x: Float = one.extent(along: SIMD3<Float>(1, 0, 0)).hi
        if x > 0 { bowRise = max(bowRise, top) } else { sternRise = max(sternRise, top) }
    }
    print(String(format: "        manu ihu %.2f mm, manu hope %.2f mm above the gunwale (want %.2f, %.2f)", bowRise, sternRise,
                 geo.ft(manuIhuRiseFt), geo.ft(manuHopeRiseFt)))
    expect(abs(bowRise - geo.ft(manuIhuRiseFt)) < 0.1 && abs(sternRise - geo.ft(manuHopeRiseFt)) < 0.1, "manu heights")
    expect(sternRise > bowRise, "the manu rise higher at the stern")
}

test("two kia, 31 ft 2 in; each crab-claw sail between its ʻopeʻa (to 41 ft 5 in) and paepae, its edge hollowed (GPU)") {
    guard let r = renderer else { expect(false, "no GPU"); return }
    let rigs: [Int] = toy.segments.indices.filter { toy.segments[$0].part == .rig }
    expectEqual(rigs.count, 2)
    var report: [String] = []
    for i in rigs {
        let mast = only([.mast], segment: i).extent(along: SIMD3<Float>(0, 1, 0)).hi - geo.polaTop
        let spar = only([.spar], segment: i).extent(along: SIMD3<Float>(0, 1, 0)).hi - geo.polaTop
        let sails: [Prim] = toy.segments[i].prims.filter { $0.tag == .sail }
        expectEqual(sails.count, 1)
        guard let p = sails.first else { continue }
        // The hollow: the middle of the chord between the two tips is air,
        // a point inside the triangle near the tack is sail.
        let o: SIMD3<Float> = p.a
        let tipB = SIMD2<Float>(p.size.z, p.size.w)
        let tipC = SIMD2<Float>(p.b.x, p.b.y)
        let tack = SIMD2<Float>(p.size.x, p.size.y)
        let chordMid: SIMD2<Float> = (tipB + tipC) / 2
        let inner: SIMD2<Float> = tack + (chordMid - tack) * 0.5
        let pts: [SIMD3<Float>] = [o + SIMD3<Float>(chordMid.x, chordMid.y, 0), o + SIMD3<Float>(inner.x, inner.y, 0)]
        guard let d = try? r.probe(pts, toy: toy) else { expect(false, "probe failed"); return }
        report.append(String(format: "kia %.2f mm, ʻopeʻa %.2f mm; chord's middle %.2f mm outside, sail's middle %.2f", mast, spar, d[0].z, d[1].z))
        expect(abs(mast - mastHeightM * s) < 0.1, "kia \(mast) mm")
        expect(abs(spar - sparHeightM * s) < 0.1, "ʻopeʻa \(spar) mm")
        expect(d[0].z > 0.5, "the sail's edge between the tips should be hollowed")
        expect(d[1].z < -0.4, "no sail inside the claw")
    }
    print("        " + report.joined(separator: "; "))
}

section("the moulds: the platform up and down, each rig flat (Protolabs' guidelines)")

test("the platform has no undercut for an up-and-down mould: every vertical line meets it in one solid run") {
    var bad: Int = 0
    var lines: Int = 0
    let ex = platform.extent(along: SIMD3<Float>(1, 0, 0))
    let top: Float = platform.extent(along: SIMD3<Float>(0, 1, 0)).hi
    for i in 0..<130 {
        let x: Float = ex.lo + 0.2 + (ex.hi - ex.lo - 0.4) * Float(i) / 129
        for j in 0..<90 {
            let z: Float = -geo.halfBeam + 2 * geo.halfBeam * Float(j) / 89
            var runs: Int = 0
            var inside: Bool = false
            var y: Float = -0.05
            while y < top + 0.2 {
                let solid: Bool = platform.sdf(SIMD3<Float>(x, y, z)).d <= 0
                if solid && !inside { runs += 1 }
                inside = solid
                y += 0.04
            }
            lines += 1
            if runs > 1 { bad += 1 }
        }
    }
    print("        \(lines) vertical lines, \(bad) meeting the platform more than once")
    expectEqual(bad, 0)
}

test("each rig has no undercut for a mould split along its sail: every line across meets it in one solid run") {
    var bad: Int = 0
    var lines: Int = 0
    for i in toy.segments.indices where toy.segments[i].part == .rig {
        let one = PosedToy(segments: [toy.segments[i]], frames: [toy.frames[i]], paints: toy.paints, seams: false)
        let ex = one.extent(along: SIMD3<Float>(1, 0, 0))
        let ey = one.extent(along: SIMD3<Float>(0, 1, 0))
        for a in 0..<80 {
            let x: Float = ex.lo + (ex.hi - ex.lo) * Float(a) / 79
            for b in 0..<120 {
                let y: Float = ey.lo + (ey.hi - ey.lo) * Float(b) / 119
                var runs: Int = 0
                var inside: Bool = false
                var z: Float = -1.5
                while z < 1.5 {
                    let solid: Bool = one.sdf(SIMD3<Float>(x, y, z)).d <= 0
                    if solid && !inside { runs += 1 }
                    inside = solid
                    z += 0.02
                }
                lines += 1
                if runs > 1 { bad += 1 }
            }
        }
    }
    print("        \(lines) lines across the rigs, \(bad) undercut")
    expectEqual(bad, 0)
}

test("draft: the hulls' sides lean at least \(draftMostDegrees)° from vertical; the manu thin upwards by \(draftLeastDegrees)°; sails \(sailThickness75) mm") {
    func outside(_ y: Float) -> Float {
        var z: Float = geo.hullOffset + geo.hullHalfWidth + 1
        while z > geo.hullOffset && hullsAndManu.sdf(SIMD3<Float>(-8, y, z)).d > 0 { z -= 0.0005 }
        return z
    }
    let upper: Float = atan((outside(geo.hullDepth - 0.8) - outside(geo.hullDepth - 2.8)) / 2.0) * 180 / .pi
    let lower: Float = atan((outside(2.0) - outside(0.8)) / 1.2) * 180 / .pi
    let m: Prim = toy.segments[0].prims.first { $0.tag == .manu }!
    let taperDeg: Float = atan(m.f.w) * 180 / .pi
    let sail: Prim? = toy.segments.flatMap { $0.prims }.first { $0.tag == .sail }
    let th: Float = 2 * (sail?.ra ?? 0)
    print(String(format: "        hull sides lean %.1f° (upper) and %.1f° (the V); manu faces %.2f°; sail %.2f mm", upper, lower, taperDeg, th))
    expect(upper >= draftMostDegrees && lower >= draftMostDegrees, "hull draft")
    expect(taperDeg >= draftLeastDegrees - 1e-4, "manu draft \(taperDeg)")
    expect(mouldWallRange.contains(th), "sail \(th) mm")
}

testSize(toy, name: "voyaging canoe")
testOptics(renderer, mutant: mutant)

section("the distance function")
testDistanceFunction(renderer, toy: toy, label: "the still")

var stations: [SeamStation] = []
for (x, side, name) in [(Float(-8), Float(-1), "port hull, outside"), (8, 1, "starboard hull, outside"), (-8, 1, "starboard hull, inside")] {
    let y: Float = geo.hullDepth - 0.3
    let zc: Float = side * geo.hullOffset
    let out: Float = name.hasSuffix("inside") ? -side : side
    stations.append(SeamStation(segment: 0, at: SIMD3<Float>(x, y, zc + out * geo.hullHalfWidth), outward: SIMD3<Float>(0, 0, out),
                                across: SIMD3<Float>(0, 1, 0), name: name, offsets: (0.12, 0.22), reach: 2.5))
}
if let rigIndex = toy.segments.firstIndex(where: { $0.part == .rig }) {
    let mx: Float = geo.masts[0]
    stations.append(SeamStation(segment: rigIndex, at: SIMD3<Float>(mx, geo.polaTop + 15, 0), outward: SIMD3<Float>(1, 0, 0),
                                across: SIMD3<Float>(0, 0, 1), name: "fore kia", offsets: (0.12, 0.24), reach: 3))
}
testSeam(toy, stations: stations, where: "round each hull's gunwale, and up each rig's kia in its sail's plane")

let len = toy.extent(along: SIMD3<Float>(1, 0, 0))
let picture: ToyImage? = testStillPicture(renderer, setup: setup, caption: boatCaption(lengthMM: len.hi - len.lo), mutant: mutant)

test("the inset shows two hulls to scale, open water between them, read off the pixels") {
    guard let img = picture else { expect(false, "render failed"); return }
    let mm: Float = insetMillimetresPerPixel(height: img.height)
    let row: CGPoint = insetPixel(SIMD3<Float>(setup.cut.centre.x, 2.0, 0), cut: setup.cut, height: img.height)
    let runs = insetRuns(img, row: Int(row.y))
    let rowY: Float = insetPoint(row.x, CGFloat(Int(row.y)) + 0.5, cut: setup.cut, height: img.height).y
    var zEdge: Float = geo.hullOffset
    while toy.sdf(SIMD3<Float>(setup.cut.centre.x, rowY, zEdge)).d <= 0 { zEdge += 0.001 }
    var zIn: Float = geo.hullOffset
    while toy.sdf(SIMD3<Float>(setup.cut.centre.x, rowY, zIn)).d <= 0 { zIn -= 0.001 }
    let wantPx: Float = (zEdge - zIn) / mm
    print("        row at 2 mm up: " + runs.map { "\($0.0) \($0.2)" }.joined(separator: ", ") + String(format: " (each hull %.1f px)", wantPx))
    expectEqual(runs.map { $0.0 }, ["air", "pvc", "air", "pvc", "air"])
    if runs.count == 5 {
        expect(abs(Float(runs[1].2) - wantPx) <= 1.5 && abs(Float(runs[3].2) - wantPx) <= 1.5, "hulls \(runs[1].2), \(runs[3].2) px")
    }
}

finish()
