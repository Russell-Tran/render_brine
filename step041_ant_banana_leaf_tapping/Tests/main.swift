// Tests for step 41: step 40's tests, copied (run on the contact frame, where
// the scene is step 40's), then the motion as step 30 tests it — the touch at
// every touch-down, no penetration at any frame, a clear lift, a forward loop —
// and what must NOT move: the wax molecule, and the smell (none, every frame).
//
// Step 40's tests. The anatomy is checked against the literature it cites,
// the contact and the pores against the GPU's own distance functions — the
// same source that draws the picture — and the finished frame for what it
// shows where.
//
// ANT_MUTANT=segments13|hover|tastePores|odour|alkaneFormula|press|rewind breaks the scene
// on purpose; `make mutants` requires the suite to fail for each.

import Foundation
import Metal
import simd

let mutant: Mutant = {
    switch ProcessInfo.processInfo.environment["ANT_MUTANT"] {
    case "segments13": return .segments13
    case "hover": return .hover
    case "tastePores": return .tastePores
    case "odour": return .odour
    case "alkaneFormula": return .alkaneFormula
    case "press": return .press
    case "rewind": return .rewind
    default: return .none
    }
}()

/// Holds the device so a failure to find one is a test failure, not a crash.
final class DeviceBox {
    let device: MTLDevice? = try? findDevice()
}
let gpu = DeviceBox()

let scene: Scene? = try? Scene(mutant: mutant)
let library: MTLLibrary? = {
    guard let d = gpu.device, let s = scene else { return nil }
    return try? makeLibrary(d, s)
}()

/// Every frame of the GIF's loop, and the first — the touch-down, step 40's pose.
let frames: [FrameState] = scene.map { s in (0..<defaultFrames).map { s.frame(at: frameTime($0, of: defaultFrames)) } } ?? []
let contactFrame: FrameState? = frames.first

func probe(_ pts: [SIMD3<Float>], frame given: FrameState? = nil) -> [Probe]? {
    guard let d = gpu.device, let s = scene, let l = library, let f = given ?? contactFrame else { return nil }
    return try? probeScene(pts, scene: s, frame: f, library: l, on: d)
}

func deg(_ r: Float) -> Float { r * 180 / Float.pi }

section("the ant, against the literature")

test("the scene loads") {
    expect(scene != nil, "could not build the scene (are the Resources/ structure files there?)")
}

test("each antenna has 12 segments — scape plus an 11-segment funiculus — as a worker's does") {
    guard let s = contactFrame else { expect(false); return }
    expectEqual(s.ant.antennae.count, 2)
    for (i, a) in s.ant.antennae.enumerated() {
        expectEqual(a.segments.count, workerAntennaSegments)
        // Count what goes to the GPU, too: shapes tagged as this antenna.
        var drawn: Int = 0
        for x in s.ant.shapes where x.part == AntPart.antenna && x.index == i { drawn += 1 }
        expectEqual(drawn, 12)
    }
}

test("the antennae are elbowed, and the funiculus has no club") {
    guard let s = contactFrame else { expect(false); return }
    for a in s.ant.antennae {
        let scape: Shape = a.segments[0]
        let first: Shape = a.segments[1]
        let bend: Float = deg(acos(simd_dot(simd_normalize(scape.b - scape.a), simd_normalize(first.b - first.a))))
        expect(bend > 35, "elbow is only \(bend)°")
        expect(abs(simd_distance(scape.a, scape.b) - scapeLength) < 1e-4, "scape length")
        // Without a distinct club: no segment's end radius jumps by more than
        // 12% over the one before (a Camponotus-like club would be 40%+).
        for i in 2..<a.segments.count {
            let ratio: Float = a.segments[i].rb / a.segments[i - 1].rb
            expect(ratio < 1.12, "segment \(i) jumps by \(ratio)×")
        }
    }
}

test("the head is Seifert's: CS 976 µm, CL/CW 1.074, scape SL/CS 0.979") {
    expect(abs(headWidth - 0.941) < 0.002, "CW \(headWidth)")
    expect(abs(headLength - 1.011) < 0.002, "CL \(headLength)")
    expect(abs((headLength + headWidth) / 2 - 0.976) < 1e-4)
    expect(abs(scapeLength / cephalicSize - 0.979) < 1e-4)
    guard let s = contactFrame else { expect(false); return }
    let head: Shape = s.ant.body[0]
    expect(abs(head.b.z * 2 - headWidth) < 1e-4, "drawn head width \(head.b.z * 2)")
}

test("six legs, and every one joins the mesosoma — none the head, petiole or gaster") {
    guard let s = contactFrame else { expect(false); return }
    expectEqual(s.ant.legs.count, 6)
    let meso: [Shape] = s.ant.body.filter { $0.part == .mesosoma }
    let others: [Shape] = s.ant.body.filter { $0.part == .head || $0.part == .petiole || $0.part == .gaster }
    for leg in s.ant.legs {
        expect(meso.contains { $0.contains(leg.root) }, "\(leg.name) root \(leg.root) is not inside the mesosoma")
        expect(!others.contains { $0.contains(leg.root) }, "\(leg.name) root is inside another tagma")
        // The first segment starts at that root: the leg is attached, not floating.
        expect(simd_distance(leg.shapes[0].a, leg.root) < 1e-6, "\(leg.name) coxa does not start at its root")
        // And the tarsus tip stands on the table.
        let last: Shape = leg.shapes[leg.shapes.count - 1]
        expect(abs(last.b.y - last.rb) < 0.004, "\(leg.name) tarsus tip at \(last.b.y - last.rb) mm above the table")
    }
    let legShapes: Int = s.ant.shapes.filter { $0.part == .leg }.count
    expectEqual(legShapes, 6 * 9)
}

test("one petiole, a single node between mesosoma and gaster, as in all Formicinae") {
    guard let s = contactFrame else { expect(false); return }
    let pet: [Shape] = s.ant.body.filter { $0.part == .petiole }
    expectEqual(pet.count, 1)
    let mesoBack: Float = s.ant.body.filter { $0.part == .mesosoma }.map { $0.extent(along: SIMD3(1, 0, 0)).lo }.min() ?? 0
    let gasterFront: Float = s.ant.body.filter { $0.part == .gaster }.map { $0.extent(along: SIMD3(1, 0, 0)).hi }.max() ?? 0
    expect(pet[0].a.x < mesoBack + 0.05 && pet[0].a.x > gasterFront - 0.05,
           "petiole at x \(pet[0].a.x), mesosoma ends \(mesoBack), gaster starts \(gasterFront)")
}

test("the worker is 3.4–5.0 mm long, mandibles to gaster tip") {
    guard let s = contactFrame else { expect(false); return }
    let l: Float = s.ant.length
    print(String(format: "        length %.2f mm", l))
    expect(workerLengthRange.contains(l), "length \(l) mm")
}

section("the leaf")

test("the leaf's textures are Alayan et al.'s: pleats 7 mm / 250 µm, ridges 200 µm / 15 µm, and in the inset 20 µm / 2 µm and 1 µm / 0.2 µm") {
    expect(pleatSpacing == 7.0 && abs(pleatHeight - 0.250) < 1e-6)
    expect(abs(veinRidgeSpacing - 0.200) < 1e-6 && abs(veinRidgeHeight - 0.015) < 1e-6)
    expect(cellRidgeSpacing == 20 && cellRidgeHeight == 2 && waxBumpSpacing == 1 && abs(waxBumpHeight - 0.2) < 1e-6)
    // Measure what is drawn: walk across the veins and read the top and the
    // bottom back.
    var ridge: [Float] = []
    var bottom: [Float] = []
    let origin = SIMD2<Float>(3.0, 0.5)
    for k in 0..<4000 {
        let xz: SIMD2<Float> = origin + acrossVeins * (Float(k) * 0.002)       // 8 mm
        bottom.append(leafBottom(xz))
        ridge.append(leafTop(xz) - leafBottom(xz) - leafThickness)
    }
    let ridgeRange: Float = (ridge.max() ?? 0) - (ridge.min() ?? 0)
    let pleatRange: Float = (bottom.max() ?? 0) - (bottom.min() ?? 0)
    print(String(format: "        drawn: vein ridges %.1f µm high, pleats %.0f µm high over 8 mm", ridgeRange * 1000, pleatRange * 1000))
    expect(abs(ridgeRange - veinRidgeHeight) < 1e-4, "ridges \(ridgeRange)")
    expect(abs(pleatRange - pleatHeight) < 2e-3, "pleats \(pleatRange)")
    // Periods: the same height one spacing further on, and not half-way.
    let a = SIMD2<Float>(3.1, 0.2)
    expect(abs(leafTop(a) - leafTop(a + acrossVeins * pleatSpacing)) < 1e-4)
    let r0: Float = leafTop(a) - leafBottom(a)
    let r1: Float = leafTop(a + acrossVeins * veinRidgeSpacing) - leafBottom(a + acrossVeins * veinRidgeSpacing)
    expect(abs(r0 - r1) < 1e-5, "ridge period")
    // Along the veins nothing changes: the textures run with the veins.
    expect(abs(leafTop(a) - leafTop(a + veinDirection * 1.37)) < 1e-5)
}

test("the leaf rests on the card, is a thin sheet, and is water-repellent (117° contact angle)") {
    var lowest: Float = 9
    for k in 0..<2000 {
        let xz = SIMD2<Float>(2.6 + Float(k % 50) * 0.1, -2 + Float(k / 50) * 0.1)
        lowest = min(lowest, leafBottom(xz))
    }
    expect(abs(lowest) < 1e-3, "the leaf's lowest point is \(lowest) mm off the card")
    expect(leafThickness >= 0.1 && leafThickness <= 0.6, "thickness \(leafThickness)")
    expect(leafWaterContactAngle > 90 && abs(leafWaterContactAngle - 117) < 1e-6)
    expect(leafWaxLoad.lowerBound == 80 && leafWaxLoad.upperBound == 90)
    // The leaf lies beyond the cut, away from the ant: the head is on the
    // other side of the edge.
    let head = SIMD2<Float>(headCentre.x, headCentre.z)
    expect(simd_dot(head - cutEdgeA, cutEdgeInward) < 0, "the ant's head is over the leaf")
}

test("the leaf is green, its cut face a deeper green, and it has no smell to give") {
    let g: SIMD3<Float> = leafAlbedo
    expect(g.y > g.x && g.y > g.z, "leaf colour \(g) is not green")
    let c: SIMD3<Float> = leafCutAlbedo
    expect(c.y > c.x && c.y > c.z && c.y < g.y, "cut colour \(c)")
    let w: SIMD3<Float> = labToLinearSRGB(SIMD3<Float>(100, 0, 0))
    expect(simd_length(w - SIMD3<Float>(1, 1, 1)) < 0.01, "white is \(w)")
    expect(!objectHasOdour)
    expect(stomataPerSquareMillimetre == 70)
}

section("the touch")

test("the right antenna's tip touches the leaf: distance 0 within 2 µm, and no penetration") {
    guard let s = contactFrame else { expect(false); return }
    let a: Antenna = s.ant.antennae[0]
    let (point, normal) = contactPoint()
    // On the CPU: the contact point lies on the leaf's top, inside its edge.
    expect(abs(point.y - leafTop(SIMD2<Float>(point.x, point.z))) < 1e-5, "contact point is off the top")
    expect(-simd_dot(SIMD2<Float>(point.x, point.z) - cutEdgeA, cutEdgeInward) < -0.05, "contact is at the cut edge, not on the surface")
    // On the GPU: the lowest point of the tip is on the leaf.
    let lowest: SIMD3<Float> = a.tipCentre - normal * a.tipRadius
    guard let r = probe([lowest]) else { expect(false, "probe failed"); return }
    print(String(format: "        tip to leaf %.4f mm, tip to its own surface %.4f mm", r[0].food, r[0].ant))
    expect(abs(r[0].food) < 0.002, "the tip's nearest point is \(r[0].food) mm from the leaf")
    expect(abs(r[0].ant) < 0.002, "that point is not on the antenna: \(r[0].ant)")
    // No part of the antenna inside the leaf: sample a box round the tip.
    var pts: [SIMD3<Float>] = []
    var rng = SystemRandomNumberGenerator()
    for _ in 0..<60_000 {
        pts.append(point + SIMD3<Float>(Float.random(in: -0.15...0.15, using: &rng),
                                        Float.random(in: -0.05...0.25, using: &rng),
                                        Float.random(in: -0.15...0.15, using: &rng)))
    }
    guard let q = probe(pts) else { expect(false, "probe failed"); return }
    var worst: Float = 0
    for p in q where p.ant < 0 { worst = min(worst, p.food) }
    expect(worst > -0.002, "antenna reaches \(-worst) mm into the leaf")
}

test("no part of the ant is inside the leaf — not the left antenna, not a foot") {
    guard let s = contactFrame else { expect(false); return }
    var pts: [SIMD3<Float>] = []
    for shape in s.ant.shapes where shape.kind == .roundCone {
        for k in 0...20 {
            let t: Float = Float(k) / 20
            pts.append(shape.a + (shape.b - shape.a) * t)
        }
    }
    for a in s.ant.antennae { pts.append(a.tipCentre) }
    var worst: Float = 9
    for p in pts { worst = min(worst, leafSDF(p)) }
    // Skip the right tip, which touches by design: its centre is one tip
    // radius off, so every sampled centre line is at least that far.
    print(String(format: "        nearest ant centre line to the leaf: %.4f mm", worst))
    expect(worst > funiculusTipRadius * 0.9, "an ant part reaches into the leaf: \(worst)")
}

test("in the inset the taste hair's apex rests on a wax bump's top at the origin, and nowhere enters the leaf") {
    guard let s = scene else { expect(false); return }
    let h: Sensillum = s.hairs[0]
    expect(abs(h.tip.y - h.tipRadius) < 1e-6 && abs(h.tip.x) < 1e-6 && abs(h.tip.z) < 1e-6, "apex not over the origin")
    guard let r = probe([SIMD3<Float>(0, 0.0005, 0), SIMD3<Float>(0, -0.02, 0)]) else { expect(false, "probe failed"); return }
    print(String(format: "        leaf just above the contact: %.4f µm; just below: %.4f µm", r[0].insetGround, r[1].insetGround))
    expect(abs(r[0].insetGround) < 0.01, "the wax at the contact is \(r[0].insetGround) µm off")
    expect(r[1].insetGround < 0, "under the contact is not leaf")
    // Points throughout the hair: none inside the leaf or a guard cell.
    var pts: [SIMD3<Float>] = []
    var rng = SystemRandomNumberGenerator()
    for _ in 0..<40_000 {
        let t: Float = Float.random(in: 0...1, using: &rng)
        let d: SIMD3<Float> = simd_normalize(SIMD3<Float>(Float.random(in: -1...1, using: &rng), Float.random(in: -1...1, using: &rng),
                                                          Float.random(in: -1...1, using: &rng)))
        let rad: Float = (h.baseRadius + (h.tipRadius - h.baseRadius) * t) * Float.random(in: 0...1, using: &rng)
        pts.append(h.base + (h.tip - h.base) * t + d * rad)
    }
    for k in 0..<400 {
        let a: Float = Float(k) * 2.399
        let z: Float = -Float(k) / 400
        pts.append(h.tip + SIMD3<Float>(cos(a) * (1 - z * z).squareRoot(), z, sin(a) * (1 - z * z).squareRoot()) * h.tipRadius)
    }
    guard let q = probe(pts) else { expect(false, "probe failed"); return }
    var worst: Float = 9
    for p in q { worst = min(worst, min(p.insetGround, p.guardCells)) }
    print(String(format: "        deepest hair point into the leaf: %.4f µm", -min(worst, 0)))
    expect(worst > -0.01, "the hair enters the leaf by \(-worst) µm")
}

test("a stoma: two guard cells on the surface, meeting at the ends, a slit pore between them into the leaf") {
    guard let s = scene else { expect(false); return }
    let c: SIMD3<Float> = stomaCentre
    let ax: SIMD2<Float> = stomaAxis
    let ac = SIMD2<Float>(-ax.y, ax.x)
    // Aligned with the ridges (and so the veins): the axis is square to the
    // across-ridge direction.
    expect(abs(simd_dot(ax, insetAcrossRidges)) < 1e-5)
    let ground: Float = insetRidgeHeight(SIMD2<Float>(c.x, c.z))
    func at(_ along: Float, _ across: Float, _ up: Float) -> SIMD3<Float> {
        let xz: SIMD2<Float> = SIMD2<Float>(c.x, c.z) + ax * along + ac * across
        return SIMD3<Float>(xz.x, insetRidgeHeight(xz) + up, xz.y)
    }
    let pts: [SIMD3<Float>] = [
        at(0, stomaWidth * 0.25, 0.6),          // on a guard cell, above the ground: solid
        at(0, -stomaWidth * 0.25, 0.6),         // the other
        at(0, 0, -1.0),                         // in the pore, below the surface: open
        at(0, 0, 0.3),                          // in the slit between the cells: open
        at(stomaLength * 0.34, 0, 0.3),         // past the pore's end, where they meet: closed
        at(0, stomaWidth * 0.8, 0.6),           // beside the stoma: air
    ]
    guard let r = probe(pts) else { expect(false, "probe failed"); return }
    print(String(format: "        guard cells %.2f, %.2f; pore %.2f, slit %.2f; end %.2f; beside %.2f µm", r[0].guardCells, r[1].guardCells,
                 min(r[2].insetGround, r[2].guardCells), min(r[3].insetGround, r[3].guardCells), r[4].guardCells, r[5].guardCells))
    expect(r[0].guardCells < 0 && r[1].guardCells < 0, "no guard cells")
    expect(min(r[2].insetGround, r[2].guardCells) > 0, "the pore does not open into the leaf")
    expect(min(r[3].insetGround, r[3].guardCells) > 0, "no slit between the guard cells")
    expect(r[4].guardCells < 0, "the guard cells do not meet at the end")
    expect(r[5].guardCells > 0 && r[5].insetGround > 0, "the stoma is too wide")
    // Clear of the contact: every point of either guard cell (the same
    // ellipsoids the kernel draws, sampled over their surfaces) is microns
    // from the taste hair's apex, so the hair touches wax, not the stoma.
    let apex: SIMD3<Float> = s.hairs[0].tip - SIMD3<Float>(0, s.hairs[0].tipRadius, 0)
    var nearest: Float = 1e9
    for side in [Float(1), Float(-1)] {
        let cxz: SIMD2<Float> = SIMD2<Float>(c.x, c.z) + ac * side * (stomaWidth * 0.18)
        let centre = SIMD3<Float>(cxz.x, insetRidgeHeight(cxz) - 0.3, cxz.y)
        let r = SIMD3<Float>(stomaLength * 0.5, 1.5, stomaWidth * 0.32)
        for i in 0..<60 {
            for j in 0..<30 {
                let th: Float = Float(i) / 60 * 2 * Float.pi
                let ph: Float = Float(j) / 29 * Float.pi - Float.pi / 2
                let l = SIMD3<Float>(cos(ph) * cos(th), sin(ph), cos(ph) * sin(th)) * r
                let w: SIMD2<Float> = ax * l.x + ac * l.z
                nearest = min(nearest, simd_distance(apex, centre + SIMD3<Float>(w.x, l.y, w.y)))
            }
        }
    }
    print(String(format: "        stoma %.1f µm from the contact; its centre %.2f µm below the ridge line", nearest, ground - c.y))
    expect(nearest > 3, "the stoma is \(nearest) µm from the contact")
}

section("the two sense hairs")

/// Survey a hair's wall on the GPU: sample just under its outer surface on a
/// fine (t, φ) grid, call a sample a hole if the drawn distance says it is
/// outside the hair, and count connected holes. Also test the apex.
func poreSurvey(_ hairIndex: Int) -> (wallPores: Int, apexHole: Bool, largest: Int)? {
    guard let s = scene else { return nil }
    let h: Sensillum = s.hairs[hairIndex]
    let u: SIMD3<Float> = simd_normalize(h.tip - h.base)
    let helper: SIMD3<Float> = abs(u.y) < 0.9 ? SIMD3(0, 1, 0) : SIMD3(1, 0, 0)
    let e1: SIMD3<Float> = simd_normalize(simd_cross(u, helper))
    let e2: SIMD3<Float> = simd_cross(u, e1)
    let L: Float = h.length
    let step: Float = 0.012
    let rows: Int = Int(0.95 * L / step)
    let meanR: Float = (h.baseRadius + h.tipRadius) / 2
    let cols: Int = Int(2 * Float.pi * meanR / step)
    let depth: Float = 0.03
    var pts: [SIMD3<Float>] = []
    pts.reserveCapacity(rows * cols + 1)
    for i in 0..<rows {
        let t: Float = 0.05 * L + Float(i) * step
        let r: Float = h.baseRadius + (h.tipRadius - h.baseRadius) * (t / L) - depth
        for j in 0..<cols {
            let phi: Float = 2 * Float.pi * Float(j) / Float(cols)
            pts.append(h.base + u * t + (e1 * cos(phi) + e2 * sin(phi)) * r)
        }
    }
    pts.append(h.tip + u * (h.tipRadius - depth))
    guard let res = probe(pts) else { return nil }
    let dist: (Int) -> Float = { hairIndex == 0 ? res[$0].tasteHair : res[$0].smellHair }
    let hole: [Bool] = (0..<(rows * cols)).map { dist($0) > 0 }
    // Union–find over the grid, wrapping round the hair.
    var parent: [Int] = Array(0..<(rows * cols))
    func find(_ x: Int) -> Int {
        var x = x
        while parent[x] != x { parent[x] = parent[parent[x]]; x = parent[x] }
        return x
    }
    func join(_ a: Int, _ b: Int) { let ra = find(a), rb = find(b); if ra != rb { parent[ra] = rb } }
    for i in 0..<rows {
        for j in 0..<cols where hole[i * cols + j] {
            let right: Int = i * cols + (j + 1) % cols
            if hole[right] { join(i * cols + j, right) }
            if i + 1 < rows && hole[(i + 1) * cols + j] { join(i * cols + j, (i + 1) * cols + j) }
        }
    }
    var sizes: [Int: Int] = [:]
    for k in 0..<(rows * cols) where hole[k] { sizes[find(k), default: 0] += 1 }
    return (sizes.count, dist(rows * cols) > 0, sizes.values.max() ?? 0)
}

test("the taste hair has exactly one pore, and it is at the tip") {
    guard let s = scene, let survey = poreSurvey(0) else { expect(false, "survey failed"); return }
    print("        taste hair: \(survey.wallPores) wall pores drawn, tip pore \(survey.apexHole)")
    expectEqual(survey.wallPores, 0)
    expect(survey.apexHole, "no opening at the apex")
    expectEqual(s.hairs[0].wallPoreCount, 0)
    expect(s.hairs[0].tipPoreRadius > 0)
}

test("the smell hair has many pores in its wall and none at the tip") {
    guard let s = scene, let survey = poreSurvey(1) else { expect(false, "survey failed"); return }
    let spec: Int = s.hairs[1].wallPoreCount
    print("        smell hair: \(survey.wallPores) wall pores drawn (lattice \(spec)), tip pore \(survey.apexHole), largest \(survey.largest) samples")
    expect(survey.wallPores >= 50, "only \(survey.wallPores) wall pores")
    expect(Double(survey.wallPores) >= 0.85 * Double(spec), "\(survey.wallPores) of \(spec) lattice pores drawn")
    expect(!survey.apexHole, "the smell hair is open at the apex")
    // Pores, not slots: no hole bigger than a pore's area in samples, doubled
    // for the seam where two half-pores can meet.
    let r: Float = s.hairs[1].wallPoreRadius
    let area: Float = Float.pi * r * r / (0.012 * 0.012)
    expect(Float(survey.largest) < 2.5 * area, "a hole of \(survey.largest) samples against a pore of \(area)")
    // Gellert et al. 2022: ant basiconic wall pores 0.07–0.09 µm across.
    expect(2 * r >= 0.07 && 2 * r <= 0.09, "pore diameter \(2 * r) µm")
}

section("smell: an intact leaf")

test("no odour molecules reach the smell hair: an intact leaf gives off almost nothing") {
    guard let s = scene else { expect(false); return }
    expect(!objectHasOdour)
    expectEqual(s.volatiles.count, 0)
}

section("the molecule")

func dihedral(_ p0: SIMD3<Float>, _ p1: SIMD3<Float>, _ p2: SIMD3<Float>, _ p3: SIMD3<Float>) -> Float {
    let b1: SIMD3<Float> = p1 - p0, b2: SIMD3<Float> = p2 - p1, b3: SIMD3<Float> = p3 - p2
    let n1: SIMD3<Float> = simd_cross(b1, b2), n2: SIMD3<Float> = simd_cross(b2, b3)
    return deg(atan2(simd_dot(simd_cross(n1, n2), simd_normalize(b2)), simd_dot(n1, n2)))
}

test("hentriacontane is C31H64: 95 atoms, 94 bonds, no ring, every C four bonds and every H one") {
    guard let s = scene else { expect(false); return }
    let m: Molecule = s.chemistry.alkane
    let f: [String: Int] = m.formula
    expect(f["C"] == 31 && f["H"] == 64 && f.count == 2, "the alkane is \(f)")
    expectEqual(m.atoms.count, 95)
    expectEqual(m.bonds.count, 94)
    expectEqual(m.rings, 0)
    let v: [Int] = m.valences
    for (i, a) in m.atoms.enumerated() { expectEqual(v[i], a.element == "C" ? 4 : 1) }
    // CnH2n+2: a saturated alkane.
    expectEqual(f["H"] ?? 0, 2 * (f["C"] ?? 0) + 2)
}

test("its geometry is PubChem's hexadecane's: C–C 1.530, C–H 1.096 Å, C–C–C 113.1°, H–C–H 107.2°, and all-trans") {
    guard let s = scene else { expect(false); return }
    let m: Molecule = s.chemistry.alkane
    var neighbours: [[Int]] = Array(repeating: [], count: m.atoms.count)
    for (a, b) in m.bonds {
        neighbours[a].append(b); neighbours[b].append(a)
        let d: Float = simd_distance(m.atoms[a].position, m.atoms[b].position)
        let want: Float = m.atoms[a].element == "C" && m.atoms[b].element == "C" ? 1.530 : 1.096
        expect(abs(d - want) < 1e-3, "bond \(a)–\(b) is \(d) Å")
    }
    func angle(_ a: Int, _ c: Int, _ b: Int) -> Float {
        let u: SIMD3<Float> = simd_normalize(m.atoms[a].position - m.atoms[c].position)
        let w: SIMD3<Float> = simd_normalize(m.atoms[b].position - m.atoms[c].position)
        return deg(acos(min(max(simd_dot(u, w), -1), 1)))
    }
    var backbone: [Int] = []
    for i in 0..<m.atoms.count where m.atoms[i].element == "C" { backbone.append(i) }
    for k in 1..<(backbone.count - 1) {
        let ccc: Float = angle(backbone[k - 1], backbone[k], backbone[k + 1])
        expect(abs(ccc - 113.1) < 0.05, "C–C–C \(ccc)°")
        let hs: [Int] = neighbours[backbone[k]].filter { m.atoms[$0].element == "H" }
        expectEqual(hs.count, 2)
        if hs.count == 2 {
            let hch: Float = angle(hs[0], backbone[k], hs[1])
            expect(abs(hch - 107.2) < 0.05, "H–C–H \(hch)°")
        }
    }
    for k in 0..<(backbone.count - 3) {
        let d: Float = dihedral(m.atoms[backbone[k]].position, m.atoms[backbone[k + 1]].position,
                                m.atoms[backbone[k + 2]].position, m.atoms[backbone[k + 3]].position)
        expect(abs(abs(d) - 180) < 0.5, "backbone dihedral \(d)° is not trans")
    }
    // No two atoms that are not bonded closer than 1.7 Å (H···H across a
    // CH2 is 1.76 Å at these numbers).
    for i in 0..<m.atoms.count {
        for j in (i + 1)..<m.atoms.count where !neighbours[i].contains(j) {
            expect(simd_distance(m.atoms[i].position, m.atoms[j].position) > 1.7, "atoms \(i), \(j) crowd")
        }
    }
    let first: SIMD3<Float> = m.atoms[backbone[0]].position
    let last: SIMD3<Float> = m.atoms[backbone[backbone.count - 1]].position
    print(String(format: "        chain end to end %.1f Å", simd_distance(first, last)))
}

section("the distance functions are distances")

/// Points in a thin shell round each hair's wall, where the pores are.
func hairShellPoints(count: Int) -> [SIMD3<Float>] {
    guard let s = scene else { return [] }
    var rng = SystemRandomNumberGenerator()
    var out: [SIMD3<Float>] = []
    for h in s.hairs {
        let u: SIMD3<Float> = simd_normalize(h.tip - h.base)
        let e1: SIMD3<Float> = simd_normalize(simd_cross(u, SIMD3<Float>(1, 0, 0)))
        let e2: SIMD3<Float> = simd_cross(u, e1)
        for _ in 0..<count {
            let t: Float = Float.random(in: 0.1...1.0, using: &rng) * h.length
            let phi: Float = Float.random(in: 0...(2 * Float.pi), using: &rng)
            let r: Float = h.baseRadius + (h.tipRadius - h.baseRadius) * t / h.length + Float.random(in: -0.2...0.15, using: &rng)
            out.append(h.base + u * t + (e1 * cos(phi) + e2 * sin(phi)) * r)
        }
    }
    return out
}

/// Worst over-report, per material, over pairs of nearby points both outside
/// every surface — the only place a ray ever asks from. Inside is excluded on
/// purpose: rays never go there, and the ellipsoid approximation's interior
/// is not a distance at all.
func worstOverReport(box lo: SIMD3<Float>, _ hi: SIMD3<Float>, step: Float, inset: Bool,
                     points given: [SIMD3<Float>] = [], frame: FrameState? = nil) -> [Int: Float] {
    var rng = SystemRandomNumberGenerator()
    var a: [SIMD3<Float>] = []
    var b: [SIMD3<Float>] = []
    for k in 0..<(given.isEmpty ? 120_000 : given.count) {
        let p: SIMD3<Float> = given.isEmpty
            ? SIMD3<Float>(Float.random(in: lo.x...hi.x, using: &rng), Float.random(in: lo.y...hi.y, using: &rng),
                           Float.random(in: lo.z...hi.z, using: &rng))
            : given[k]
        let d: SIMD3<Float> = simd_normalize(SIMD3<Float>(Float.random(in: -1...1, using: &rng), Float.random(in: -1...1, using: &rng),
                                                          Float.random(in: -1...1, using: &rng)))
        a.append(p)
        b.append(p + d * step)
    }
    guard let pa = probe(a, frame: frame), let pb = probe(b, frame: frame) else { return [:] }
    var worst: [Int: Float] = [:]
    for i in 0..<a.count {
        let (da, db, ma, mb): (Float, Float, Int, Int) = inset
            ? (pa[i].insetDistance, pb[i].insetDistance, pa[i].insetMaterial, pb[i].insetMaterial)
            : (pa[i].mainDistance, pb[i].mainDistance, pa[i].mainMaterial, pb[i].mainMaterial)
        guard ma == mb, da > 0, db > 0 else { continue }
        let v: Float = abs(da - db) / step
        if v > (worst[ma] ?? 0) && ProcessInfo.processInfo.environment["ANT_DEBUG"] != nil { print("        m\(ma) \(v) at \(a[i]) d \(da)") }
        worst[ma] = max(worst[ma] ?? 0, v)
    }
    return worst
}

test("main view: outside every surface, no distance claims more room than the ray allows") {
    var w: [Int: Float] = worstOverReport(box: SIMD3(-2.3, 0, -1.8), SIMD3(4.5, 1.6, 2.4), step: 0.01, inset: false)
    // And a second pass packed round the body, where the parts blend. The
    // first version of this test found the petiole — a thin ellipsoid —
    // over-reporting 2.7× here; it is an exact rounded box now.
    for (m, v) in worstOverReport(box: SIMD3(-2.1, 0, -0.7), SIMD3(2.2, 1.3, 0.7), step: 0.005, inset: false) {
        w[m] = max(w[m] ?? 0, v)
    }
    // As step 30: round the tip and the leaf with the antenna at its highest.
    if let up = frames.max(by: { $0.lift < $1.lift }) {
        for (m, v) in worstOverReport(box: SIMD3(2.0, 0, 0.3), SIMD3(3.2, 1.2, 1.6), step: 0.004, inset: false, frame: up) {
            w[m] = max(w[m] ?? 0, v)
        }
    }
    print(String(format: "        worst over-report: table %.2f, ant %.2f, leaf %.2f; the ray allows %.2f",
                 w[1] ?? 0, w[2] ?? 0, w[3] ?? 0, 1 / stepScale))
    expect(w.count == 3, "not every material was sampled: \(w)")
    for (m, v) in w { expect(v * stepScale <= 1.0, "material \(m) oversteps: \(v)") }
}

test("inset: the hairs, the dome, the waxed ridges and the stoma are honest outside too") {
    let w: [Int: Float] = worstOverReport(box: SIMD3(-4, -0.5, -12), SIMD3(8, 12, 4), step: 0.02, inset: true)
    // A second pass concentrated round the tip, where the film and pore are.
    let t: [Int: Float] = worstOverReport(box: SIMD3(-1.5, -0.2, -1.5), SIMD3(1.5, 1.5, 1.5), step: 0.005, inset: true)
    // And a third in a thin shell round each hair's wall, down among the pores.
    let pores: [Int: Float] = worstOverReport(box: .zero, .zero, step: 0.004, inset: true, points: hairShellPoints(count: 80_000))
    var all: [Int: Float] = w
    for (m, v) in t { all[m] = max(all[m] ?? 0, v) }
    for (m, v) in pores { all[m] = max(all[m] ?? 0, v) }
    // As step 30: and with the antenna lifted, when the leaf is moved — both
    // round the hairs and round where the leaf has gone.
    if let up = frames.max(by: { $0.lift < $1.lift }), up.lift > 0 {
        let leafBox: SIMD3<Float> = up.leaf.translation
        for (m, v) in worstOverReport(box: SIMD3(-4, -0.5, -12) - leafBox, SIMD3(8, 12, 4) - leafBox, step: 0.02,
                                      inset: true, frame: up) {
            all[m] = max(all[m] ?? 0, v)
        }
        for (m, v) in worstOverReport(box: SIMD3(-4, -0.5, -8), SIMD3(8, 12, 4), step: 0.01, inset: true, frame: up) {
            all[m] = max(all[m] ?? 0, v)
        }
    }
    print(String(format: "        worst over-report: leaf %.2f, cuticle %.2f, taste hair %.2f, smell hair %.2f, guard cells %.2f",
                 all[1] ?? 0, all[2] ?? 0, all[3] ?? 0, all[4] ?? 0, all[8] ?? 0))
    expect(all.count >= 5, "not every material was sampled: \(all)")
    for (m, v) in all { expect(v * stepScale <= 1.0, "material \(m) oversteps: \(v)") }
}

section("the picture")

let render: AntImage? = {
    guard let d = gpu.device, let s = scene, let f = contactFrame else { return nil }
    return try? renderAnt(width: 960, height: 540, samples: 1, scene: s, frame: f, on: d).image
}()

test("the touch is visible: antenna and leaf both within 3 px of the projected contact") {
    guard let img = render, let f = contactFrame else { expect(false, "render failed"); return }
    let a: Antenna = f.ant.antennae[0]
    let p: SIMD2<Float> = projectMain(a.tipCentre - SIMD3<Float>(0, a.tipRadius, 0), width: img.width, height: img.height)
    var sawAnt = false
    var sawSugar = false
    for dy in -3...3 {
        for dx in -3...3 {
            let x: Int = Int(p.x) + dx
            let y: Int = Int(p.y) + dy
            let v: SIMD4<Float> = img.seen(x, y)
            if v.x == 1 && v.y == 2 { sawAnt = true }
            if v.x == 1 && v.y == 3 { sawSugar = true }
        }
    }
    expect(sawAnt && sawSugar, "near the contact: antenna \(sawAnt), leaf \(sawSugar)")
}

test("the inset shows both hairs, the wax and the stoma, and no odour; the molecule inset shows only C and H") {
    guard let img = render else { expect(false, "render failed"); return }
    var seen: Set<Int> = []
    var taste: Set<Int> = []
    var odourInset: Int = 0
    for y in 0..<img.height {
        for x in 0..<img.width {
            let v: SIMD4<Float> = img.seen(x, y)
            if v.x == 2 { seen.insert(Int(v.y)) }
            if v.x == 3 && v.y >= 1 { taste.insert(Int(v.y)) }
            if v.x == 4 { odourInset += 1 }
        }
    }
    expect(seen.isSuperset(of: [1, 2, 3, 4, 8]), "inset materials seen: \(seen.sorted())")
    expect(!seen.contains(6), "odour dots drawn in the inset")
    expectEqual(odourInset, 0)
    expect(taste.isSuperset(of: [1, 3, 9]), "molecule inset shows \(taste.sorted())")
    expect(!taste.contains(2), "oxygen in an alkane")
}

test("each scale bar matches its own view: 1 mm, 5 µm, 0.5 nm in that view's pixels") {
    let w: Int = 1920
    let h: Int = 1080
    // Move 1 mm across the main view and see how far the projection goes.
    let p = SIMD3<Float>(1, 0.3, 0.5)
    let dx: Float = simd_distance(projectMain(p, width: w, height: h), projectMain(p + mainCamera.right, width: w, height: h))
    expect(abs(dx - 1 / mainMillimetresPerPixel(width: w)) < 0.01, "1 mm is \(dx) px")
    let q = SIMD3<Float>(0, 1, 0)
    let du: Float = simd_distance(projectInset(q, height: h), projectInset(q + insetCamera.right * 5, height: h))
    expect(abs(du - 5 / insetMicrometresPerPixel(height: h)) < 0.01, "5 µm is \(du) px")
    // The molecule bar: its label's length in Å over the view's Å per pixel.
    let barPx: Float = moleculeBarNanometres * 10 / moleculeAngstromsPerPixel(height: h)
    expect(abs(barPx * moleculeAngstromsPerPixel(height: h) - 10 * moleculeBarNanometres) < 1e-3)
    expect(barPx < 2 * moleculeRadius * Float(h) * 0.8, "the bar does not fit its inset")
    let odourAPerPx: Float = odourField / (2 * odourRadius * Float(h))
    expect(odourBarNanometres * 10 / odourAPerPx < 2 * odourRadius * Float(h) * 0.8, "the odour bar does not fit")
    // The odour inset sits clear of the other two.
    let hf = Float(h)
    expect(simd_distance(odourCentre, insetCentre) * hf > (odourRadius + insetRadius) * hf + 8)
    expect(simd_distance(odourCentre, moleculeCentre) * hf > (odourRadius + moleculeRadius) * hf + 8)
    // The three views differ in scale by factors of about 150 and 5000: they
    // could never share a bar.
    let mmPerPxMain: Float = mainMillimetresPerPixel(width: w)
    let mmPerPxInset: Float = insetMicrometresPerPixel(height: h) / 1000
    expect(mmPerPxMain / mmPerPxInset > 100)
}

section("the tap (step 30's, on a banana leaf)")

test("the rate is the literature's, and the slow-down is stated: 4 strokes/s, 2.5 s a tap, ×10") {
    expect(realStrokeRange.contains(realStrokesPerSecond))
    expect(abs(slowdown - 10) < 1e-5, "slowed ×\(slowdown)")
    expect(loopSeconds >= 8 && loopSeconds <= 12, "loop \(loopSeconds) s")
    // The GIF plays the loop at its real length: frames × delay = loop.
    expect(abs(Float(defaultFrames * defaultDelayCentiseconds) / 100 - loopSeconds) < 1e-4)
    expectEqual(defaultFrames % tapsPerLoop, 0)
}

test("every frame: both antennae keep 12 segments, joined end to end, the same lengths") {
    guard let c = contactFrame else { expect(false); return }
    var worstGap: Float = 0
    var worstLength: Float = 0
    for f in frames {
        expectEqual(f.ant.antennae.count, 2)
        for (i, a) in f.ant.antennae.enumerated() {
            expectEqual(a.segments.count, 12)
            expectEqual(f.ant.shapes.filter { $0.part == .antenna && $0.index == i }.count, 12)
            worstGap = max(worstGap, simd_distance(a.segments[0].a, a.socket))
            for k in 1..<a.segments.count {
                worstGap = max(worstGap, simd_distance(a.segments[k].a, a.segments[k - 1].b))
            }
            for k in 0..<a.segments.count {
                let l0: Float = simd_distance(c.ant.antennae[i].segments[k].a, c.ant.antennae[i].segments[k].b)
                let l1: Float = simd_distance(a.segments[k].a, a.segments[k].b)
                worstLength = max(worstLength, abs(l1 / l0 - 1))
            }
        }
    }
    print(String(format: "        worst joint gap %.2e mm, worst segment length change %.3f%%", worstGap, worstLength * 100))
    expect(worstGap < 1e-5, "a joint opens by \(worstGap) mm")
    expect(worstLength < 0.01, "a segment changes length by \(worstLength * 100)%")
}

/// The leaf's distance from the right antenna at one frame, on the GPU: the
/// tip's lowest point (along the contact normal), and the deepest any
/// antenna point reaches into the leaf among points packed round the tip.
func tipSurvey(_ f: FrameState, points n: Int) -> (lowest: Float, deepest: Float)? {
    guard let s = scene else { return nil }
    let a: Antenna = f.ant.antennae[0]
    let lowest: SIMD3<Float> = a.tipCentre - s.contact.normal * a.tipRadius
    var pts: [SIMD3<Float>] = [lowest]
    var rng = SystemRandomNumberGenerator()
    for _ in 0..<n {
        pts.append(a.tipCentre + SIMD3<Float>(Float.random(in: -0.16...0.16, using: &rng),
                                              Float.random(in: -0.12...0.16, using: &rng),
                                              Float.random(in: -0.16...0.16, using: &rng)))
    }
    guard let q = probe(pts, frame: f) else { return nil }
    var deepest: Float = 1
    for p in q.dropFirst() where p.ant < 0 { deepest = min(deepest, p.food) }
    return (q[0].food, deepest)
}

/// The exact clearance between a ball and the leaf's top, on the CPU: the
/// nearest of the top surface's points on a 2 µm grid round the ball, minus
/// its radius. (The GPU's leaf distance is a safe bound, up to 6% short.)
func exactClearance(centre c: SIMD3<Float>, radius r: Float) -> Float {
    var best: Float = 1e9
    for i in -80...80 {
        for j in -80...80 {
            let xz = SIMD2<Float>(c.x + Float(i) * 0.002, c.z + Float(j) * 0.002)
            let p = SIMD3<Float>(xz.x, leafTop(xz), xz.y)
            best = min(best, simd_distance(p, c))
        }
    }
    return best - r
}

test("at every touch-down the tip meets the leaf: 0 within 2 µm, on the GPU and exactly on the CPU") {
    let downs: [FrameState] = frames.filter { $0.lift == 0 }
    expect(downs.count >= tapsPerLoop, "only \(downs.count) touching frames")
    var worst: Float = 0
    var worstExact: Float = 0
    for f in downs {
        guard let r = tipSurvey(f, points: 0) else { expect(false, "probe failed"); return }
        worst = max(worst, abs(r.lowest))
        let a: Antenna = f.ant.antennae[0]
        worstExact = max(worstExact, abs(exactClearance(centre: a.tipCentre, radius: a.tipRadius)))
    }
    print(String(format: "        %d touching frames; worst tip-to-leaf %.4f mm (GPU), %.4f mm (exact)", downs.count, worst, worstExact))
    expect(worst < 0.002, "a touching tip is \(worst) mm from the leaf")
    expect(worstExact < 0.002, "a touching tip is \(worstExact) mm from the leaf's surface")
}

test("at no frame does any part of the antenna go into the leaf") {
    var worst: Float = 1
    for f in frames {
        guard let r = tipSurvey(f, points: 12_000) else { expect(false, "probe failed"); return }
        worst = min(worst, r.deepest)
    }
    print(String(format: "        %d frames × 12,000 points: nearest antenna point to the inside of the leaf %.4f mm", frames.count, worst))
    expect(worst > -0.002, "the antenna reaches \(-worst) mm into the leaf")
}

test("at no frame is any part of the ant inside the leaf — not the left antenna as it sways, not a foot") {
    var worst: Float = 9
    for f in frames {
        for shape in f.ant.shapes where shape.kind == .roundCone {
            for k in 0...20 {
                let t: Float = Float(k) / 20
                worst = min(worst, leafSDF(shape.a + (shape.b - shape.a) * t))
            }
        }
    }
    print(String(format: "        nearest ant centre line to the leaf over the loop: %.4f mm", worst))
    expect(worst > funiculusTipRadius * 0.9, "an ant part reaches into the leaf: \(worst)")
}

test("lifted, the tip clears the leaf — by the lift, and never less than 0.03 mm near the top") {
    let ups: [FrameState] = frames.filter { $0.lift > 0.5 * liftHeight }
    expect(!ups.isEmpty, "no lifted frames")
    let k: Float = 1 / (1 + leafMaxSlope * leafMaxSlope).squareRoot()
    var least: Float = 1
    var worstExact: Float = 0
    for f in ups {
        guard let r = tipSurvey(f, points: 4000) else { expect(false, "probe failed"); return }
        let a: Antenna = f.ant.antennae[0]
        let exact: Float = exactClearance(centre: a.tipCentre, radius: a.tipRadius)
        least = min(least, exact)
        worstExact = max(worstExact, abs(exact - f.lift))
        // The GPU's bound: never more than the truth, never less than K of it.
        expect(r.lowest <= f.lift + 0.002 && r.lowest >= k * f.lift - 0.002,
               "lift \(f.lift) mm but the GPU puts the tip \(r.lowest) mm up")
        expect(r.deepest > 0, "a lifted antenna point is in the leaf")
    }
    print(String(format: "        %d lifted frames, least clearance %.3f mm (lift %.3f mm at the top); exact clearance vs lift, worst %.4f mm",
                 ups.count, least, liftHeight, worstExact))
    expect(worstExact < 0.002, "the clearance and the lift disagree by \(worstExact) mm")
    expect(least > 0.03)
}

/// Points throughout the taste hair, and round its apex.
func tasteHairPoints(_ h: Sensillum, count: Int) -> [SIMD3<Float>] {
    var pts: [SIMD3<Float>] = []
    var rng = SystemRandomNumberGenerator()
    for _ in 0..<count {
        let t: Float = Float.random(in: 0...1, using: &rng)
        let d: SIMD3<Float> = simd_normalize(SIMD3<Float>(Float.random(in: -1...1, using: &rng), Float.random(in: -1...1, using: &rng),
                                                          Float.random(in: -1...1, using: &rng)))
        let rad: Float = (h.baseRadius + (h.tipRadius - h.baseRadius) * t) * Float.random(in: 0...1, using: &rng)
        pts.append(h.base + (h.tip - h.base) * t + d * rad)
    }
    for k in 0..<400 {
        let a: Float = Float(k) * 2.399
        let z: Float = -Float(k) / 400
        pts.append(h.tip + SIMD3<Float>(cos(a) * (1 - z * z).squareRoot(), z, sin(a) * (1 - z * z).squareRoot()) * h.tipRadius)
    }
    return pts
}

test("inset (GPU): the taste hair meets the wax at every touch-down, leaves it when lifted, and never enters the leaf") {
    guard let s = scene else { expect(false); return }
    let h: Sensillum = s.hairs[0]
    let apexLow = SIMD3<Float>(h.tip.x, h.tip.y - h.tipRadius, h.tip.z)
    let pts: [SIMD3<Float>] = [apexLow] + tasteHairPoints(h, count: 6000)
    var worstDown: Float = 0
    var deepest: Float = 9
    var worstDrop: Float = 0
    var worstTurn: Float = 0
    for f in frames {
        guard let r = probe(pts, frame: f) else { expect(false, "probe failed"); return }
        for p in r { deepest = min(deepest, min(p.insetGround, p.guardCells)) }
        if f.lift == 0 {
            worstDown = max(worstDown, abs(r[0].insetGround))
        } else {
            // Lifted, the leaf has fallen away by the lift, as in the main
            // view: along the normal by the lift, give or take what the tip's
            // own turn (angle θ, about its centre, one tip radius above the
            // touch) moves the touching point — at most tipRadius·θ.
            let drop: SIMD3<Float> = f.leaf.toLeaf(apexLow) - apexLow
            let rot: simd_float3x3 = f.leaf.rotation
            let trace: Float = rot.columns.0.x + rot.columns.1.y + rot.columns.2.z
            let theta: Float = acos(min(max((trace - 1) / 2, -1), 1))
            let turn: Float = 1000 * funiculusTipRadius * theta
            let off: Float = simd_length(drop - s.contact.normal * (1000 * f.lift))
            worstTurn = max(worstTurn, turn)
            expect(off <= turn + 0.1, "lifted \(f.lift) mm, the inset leaf is \(off) µm from where the lift puts it (turn allows \(turn))")
            worstDrop = max(worstDrop, abs(drop.y - 1000 * f.lift * s.contact.normal.y))
            if f.lift > 0.01 { expect(r[0].insetGround > 5, "lifted \(f.lift) mm, the wax is \(r[0].insetGround) µm from the apex") }
        }
    }
    print(String(format: "        touch-down apex to wax, worst %.4f µm; deepest hair point in the leaf %.4f µm; inset drop vs main-view lift, worst %.2f µm vertically (the tip's turn moves the touching point up to %.2f µm)",
                 worstDown, -min(deepest, 0), worstDrop, worstTurn))
    expect(worstDown < 0.01, "at a touch-down the apex is \(worstDown) µm off the wax")
    expect(deepest > -0.01, "the hair enters the leaf by \(-deepest) µm")
    expect(worstDrop < 3, "the inset leaf and the main-view lift disagree by \(worstDrop) µm")
}

section("the loop")

test("the loop is forward: the last frame runs on into the first, and the taps only go forward") {
    guard let s = scene, let c = contactFrame else { expect(false); return }
    // Time L is time 0.
    let end: FrameState = s.frame(at: loopSeconds)
    expect(simd_distance(end.ant.antennae[0].tipCentre, c.ant.antennae[0].tipCentre) < 1e-5)
    expect(simd_distance(end.ant.antennae[1].tipCentre, c.ant.antennae[1].tipCentre) < 1e-5)
    expect(simd_length(end.leaf.translation - c.leaf.translation) < 1e-3)
    // The step from the last frame into the first is an ordinary step.
    func step(_ a: FrameState, _ b: FrameState) -> Float {
        let right: Float = simd_distance(a.ant.antennae[0].tipCentre, b.ant.antennae[0].tipCentre)
        let left: Float = simd_distance(a.ant.antennae[1].tipCentre, b.ant.antennae[1].tipCentre)
        return max(right, left)
    }
    var biggest: Float = 0
    for i in 1..<frames.count { biggest = max(biggest, step(frames[i - 1], frames[i])) }
    let seam: Float = step(frames[frames.count - 1], frames[0])
    print(String(format: "        seam step %.5f mm, largest step inside the loop %.5f mm", seam, biggest))
    expect(seam <= biggest * 1.01 + 1e-6, "the seam jumps")
    // Forward: progress through the taps never falls, and a loop is exactly
    // tapsPerLoop taps.
    var last: Float = frames[0].progress
    for i in 1...frames.count {
        let t: Float = frameTime(i, of: frames.count) - (i == frames.count ? 1e-4 : 0)
        let u: Float = s.frame(at: t).progress
        expect(u >= last - 1e-5, "the loop goes back at frame \(i): \(last) → \(u)")
        last = u
    }
    expect(abs(last - Float(tapsPerLoop)) < 0.01, "the loop is \(last) taps, not \(tapsPerLoop)")
    // Each tap is the same, played forward: the lift is periodic in the tap,
    // and within a tap it rests first, then rises and falls.
    let perTap: Int = frames.count / tapsPerLoop
    for i in 0..<(frames.count - perTap) {
        expect(abs(frames[i].lift - frames[i + perTap].lift) < 1e-5, "tap \(i / perTap + 1) differs from the next at frame \(i)")
    }
    for i in 0..<perTap {
        let want: Float = liftAt(phase: Float(i) / Float(perTap))
        expect(abs(frames[i].lift - want) < 1e-5, "frame \(i) lift \(frames[i].lift), a forward tap has \(want)")
    }
}

test("nothing pops: 1 ms apart, the tips and the inset's leaf move only a little, even at the seam and each touch and lift-off") {
    guard let s = scene else { expect(false); return }
    var times: [Float] = (0..<frames.count).map { frameTime($0, of: frames.count) }
    for k in 0..<tapsPerLoop {
        let down: Float = Float(k) * tapSeconds
        let off: Float = (Float(k) + contactFraction) * tapSeconds
        times += [down - 1e-3, down, off - 1e-3, off]
    }
    times.append(loopSeconds - 1e-3)
    var worstTip: Float = 0
    var worstLeaf: Float = 0
    for t in times {
        let a: FrameState = s.frame(at: t)
        let b: FrameState = s.frame(at: t + 1e-3)
        for i in 0..<2 { worstTip = max(worstTip, simd_distance(a.ant.antennae[i].tipCentre, b.ant.antennae[i].tipCentre)) }
        worstLeaf = max(worstLeaf, simd_length(a.leaf.translation - b.leaf.translation))
    }
    // The fastest the tip moves: liftHeight·π over the lifted part of a tap.
    let fastest: Float = liftHeight * Float.pi / ((1 - contactFraction) * tapSeconds) * 1e-3
    print(String(format: "        %d moments: largest tip move in 1 ms %.2e mm (bound %.2e), leaf in the inset %.3f µm",
                 times.count, worstTip, fastest, worstLeaf))
    expect(worstTip < fastest * 1.1 + 1e-6, "a tip jumps \(worstTip) mm in 1 ms")
    expect(worstLeaf < fastest * 1000 * 1.2 + 1e-3, "the inset's leaf jumps \(worstLeaf) µm in 1 ms")
}

section("what does not move")

test("the wax molecule holds still, whole and rigid, at every frame: a solid wax (hentriacontane melts at 67.9 °C)") {
    guard let s = scene else { expect(false); return }
    expect(abs(waxAlkaneMeltingPoint - 67.9) < 1e-4)
    expect(waxAlkaneMeltingPoint > 40, "the wax would be liquid on a warm day")
    let m0: Molecule = s.molecule
    var worst: Float = 0
    for f in frames {
        expectEqual(f.molecule.atoms.count, m0.atoms.count)
        expectEqual(f.molecule.bonds.count, m0.bonds.count)
        for (a, b) in zip(f.molecule.atoms, m0.atoms) {
            expect(a.element == b.element)
            worst = max(worst, simd_distance(a.position, b.position))
        }
        // Rigid, whatever it did: every bond keeps its length.
        for (x, y) in f.molecule.bonds {
            let l0: Float = simd_distance(m0.atoms[x].position, m0.atoms[y].position)
            worst = max(worst, abs(simd_distance(f.molecule.atoms[x].position, f.molecule.atoms[y].position) - l0))
        }
    }
    print(String(format: "        %d frames: largest atom move %.1e Å", frames.count, worst))
    expect(worst < 1e-6, "an atom moves \(worst) Å")
}

test("the smell state is step 40's at every frame: nothing reaches the smell hair, and no odorant inset") {
    guard let s = scene else { expect(false); return }
    expect(!objectHasOdour)
    expect(s.volatiles.isEmpty, "\(s.volatiles.count) odour molecules in the inset")
    expect(s.source.contains("constant int VOLS = 0;"), "the kernel draws odour dots")
    expect(s.source.contains("constant bool ODOUR_INSET = false;"), "the kernel draws an odorant inset")
    var worst: Int = 0
    for f in frames { worst = max(worst, f.volatiles.count) }
    expectEqual(worst, 0)
}

section("the loop's pictures")

test("only the moving region changes: outside it, whole frames differ from the first by float noise at most") {
    guard let d = gpu.device, let s = scene, let first = contactFrame else { expect(false); return }
    let w: Int = 640
    let h: Int = 360
    let region: MovingRegion = movingRegion(scene: s, frames: frames, width: w, height: h)
    guard let r = try? Renderer(device: d, scene: s, width: w, height: h) else { expect(false, "renderer"); return }
    _ = try? r.render(first, samples: 1)
    let n: Int = w * h * 4
    let base: [UInt8] = Array(UnsafeBufferPointer(start: r.pixels.contents().assumingMemoryBound(to: UInt8.self), count: n))
    var outside: Int = 0
    var inside: Int = 0
    var worstOutside: Int = 0
    let perTap: Int = frames.count / tapsPerLoop
    for f in [perTap / 5, perTap / 2, perTap * 7 / 10, perTap * 9 / 10, perTap + 3] {
        _ = try? r.render(frames[f], samples: 1)
        let full: [UInt8] = Array(UnsafeBufferPointer(start: r.pixels.contents().assumingMemoryBound(to: UInt8.self), count: n))
        for y in 0..<h {
            for x in 0..<w {
                let i: Int = (y * w + x) * 4
                if full[i] != base[i] || full[i + 1] != base[i + 1] || full[i + 2] != base[i + 2] {
                    if region.contains(x, y) { inside += 1 } else {
                        outside += 1
                        for c in 0..<3 { worstOutside = max(worstOutside, abs(Int(full[i + c]) - Int(base[i + c]))) }
                    }
                }
            }
        }
        base.withUnsafeBytes { raw in r.pixels.contents().copyMemory(from: raw.baseAddress!, byteCount: n) }
        for rect in region.rects { _ = try? r.render(frames[f], samples: 1, region: rect) }
        let composite: [UInt8] = Array(UnsafeBufferPointer(start: r.pixels.contents().assumingMemoryBound(to: UInt8.self), count: n))
        var worstComposite: Int = 0
        for i in 0..<n where i % 4 != 3 { worstComposite = max(worstComposite, abs(Int(composite[i]) - Int(full[i]))) }
        expect(worstComposite <= 1, "frame \(f): the region render differs from the whole render by \(worstComposite) levels")
    }
    print("        region \(region.rects.count) rects, \(100 * region.area / (w * h))% of the frame; changed pixels inside \(inside), outside \(outside), worst outside \(worstOutside) levels")
    expect(worstOutside <= 1, "a pixel outside the region changes by \(worstOutside) levels")
    expect(outside < w * h / 1000, "\(outside) pixels outside the region change")
    expect(inside > 0, "nothing moved")
}

test("lifted, the main view shows the tip raised, the inset no wax at the apex, and still no odour anywhere") {
    guard let d = gpu.device, let s = scene, let up = frames.max(by: { $0.lift < $1.lift }) else { expect(false); return }
    guard let img = try? renderAnt(width: 1920, height: 1080, samples: 1, scene: s, frame: up, on: d).image else {
        expect(false, "render failed"); return
    }
    let a: Antenna = up.ant.antennae[0]
    let low: SIMD2<Float> = projectMain(a.tipCentre - SIMD3<Float>(0, a.tipRadius, 0), width: 1920, height: 1080)
    let touch: SIMD2<Float> = projectMain(s.contact.point, width: 1920, height: 1080)
    print(String(format: "        tip raised %.1f px on screen", simd_distance(low, touch)))
    expect(simd_distance(low, touch) > 10, "the lift is invisible: \(simd_distance(low, touch)) px")
    let apex: SIMD2<Float> = projectInset(s.hairs[0].tip, height: 1080)
    var leaf: Int = 0
    for dy in -12...12 {
        for dx in -12...12 {
            let v: SIMD4<Float> = img.seen(Int(apex.x) + dx, Int(apex.y) + dy)
            if v.x == 2 && (v.y == 1 || v.y == 8) { leaf += 1 }
        }
    }
    expectEqual(leaf, 0)
    var odour: Int = 0
    for y in 0..<img.height {
        for x in 0..<img.width {
            let v: SIMD4<Float> = img.seen(x, y)
            if v.x == 4 || (v.x == 2 && v.y == 6) { odour += 1 }
        }
    }
    expectEqual(odour, 0)
}

finish()
