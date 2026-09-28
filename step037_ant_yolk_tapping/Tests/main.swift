// Tests for step 37: step 36's tests, copied (run on the contact frame, where
// the scene is step 36's), then the motion as step 30 tests it — the touch at
// every touch-down, no penetration at any frame, a clear lift, a forward loop —
// and the odour: one molecule in per tap, every frame, never going back.
//
// Step 36's tests. The anatomy is checked against the literature it cites,
// the contact and the pores against the GPU's own distance functions — the
// same source that draws the picture — and the finished frame for what it
// shows where.
//
// ANT_MUTANT=segments13|hover|tastePores|volatiles|formula|press|rewind breaks the scene
// on purpose; `make mutants` requires the suite to fail for each.

import Foundation
import Metal
import simd

let mutant: Mutant = {
    switch ProcessInfo.processInfo.environment["ANT_MUTANT"] {
    case "segments13": return .segments13
    case "hover": return .hover
    case "tastePores": return .tastePores
    case "volatiles": return .volatiles
    case "formula": return .formula
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

/// Every frame of the GIF's loop, and the first — the touch-down, step 36's pose.
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

section("the yolk")

test("the crumb is built of yolk spheres: every one within 150 µm, mean near 100 µm") {
    guard let s = scene else { expect(false); return }
    let d: [Float] = s.spheres.map { 2 * $0.radius }
    let mean: Float = d.reduce(0, +) / Float(d.count)
    print(String(format: "        %d yolk spheres, %.0f–%.0f µm, mean %.0f µm", d.count, (d.min() ?? 0) * 1000, (d.max() ?? 0) * 1000, mean * 1000))
    expect(d.count >= 50, "only \(d.count) spheres")
    for x in d { expect(x <= yolkSphereMaxDiameter, "a yolk sphere \(x * 1000) µm across") }
    expect(abs(mean - yolkSphereMeanDiameter) / yolkSphereMeanDiameter < 0.25, "mean \(mean * 1000) µm")
}

test("the crumb is crumb-sized (0.4–1.2 mm), rests on the card, and nothing floats") {
    guard let s = scene else { expect(false); return }
    let crumb: [YolkSphere] = buildCrumb().spheres
    let lo: SIMD3<Float> = crumb.reduce(SIMD3<Float>(repeating: 9)) { simd_min($0, $1.centre - $1.radius) }
    let hi: SIMD3<Float> = crumb.reduce(SIMD3<Float>(repeating: -9)) { simd_max($0, $1.centre + $1.radius) }
    let size: SIMD3<Float> = hi - lo
    print(String(format: "        crumb %.2f × %.2f × %.2f mm", size.x, size.z, size.y))
    for e in [size.x, size.y, size.z] { expect(crumbSizeRange.contains(e) || e < crumbSizeRange.upperBound, "extent \(e)") }
    expect(crumbSizeRange.contains(max(size.x, size.z)), "longest extent \(max(size.x, size.z)) mm")
    expect(abs(lo.y) < 1e-5, "lowest point \(lo.y) mm off the card")
    // Every sphere touches or overlaps another (or the core): no floaters.
    for (i, a) in s.spheres.enumerated() {
        let touching: Bool = s.spheres.enumerated().contains { j, b in
            j != i && simd_distance(a.centre, b.centre) < a.radius + b.radius + 1e-4
        }
        expect(touching || a.centre.y - a.radius < 1e-4, "sphere \(i) floats")
    }
}

test("the yolk is cited: half water, a quarter fat, a sixth protein; pale yellow from its CIELAB colour") {
    expect(abs(yolkWater - 52.31) < 1e-3 && abs(yolkFat - 26.54) < 1e-3 && abs(yolkProtein - 15.86) < 1e-3)
    // CIELAB → sRGB round trip on a known point: white (L* 100) is (1, 1, 1).
    let w: SIMD3<Float> = labToLinearSRGB(SIMD3<Float>(100, 0, 0))
    expect(simd_length(w - SIMD3<Float>(1, 1, 1)) < 0.01, "white is \(w)")
    // Yolk: red > green > blue, and light.
    let y: SIMD3<Float> = yolkAlbedo
    expect(y.x > y.y && y.y > y.z, "yolk colour \(y) is not yellow-orange")
    expect(yolkLab.x >= 69, "cooked yolk should be at least as light as the lightest raw yolk")
    expect(foodHasVapour)
}

section("the touch")

test("the right antenna's tip touches the crumb: distance 0 within 2 µm, and no penetration") {
    guard let s = scene, let f = contactFrame else { expect(false); return }
    let a: Antenna = f.ant.antennae[0]
    let (point, normal) = contactPoint()
    // On the CPU: the contact point lies on the crumb's surface, above the card.
    let dc: Float = crumbSDF(point, spheres: s.spheres, core: s.core)
    expect(abs(dc) < 1e-4, "contact point is off the surface: \(dc)")
    expect(point.y > 0.1, "contact too low on the crumb: \(point.y)")
    // On the GPU: the lowest point of the tip is on the sugar.
    let lowest: SIMD3<Float> = a.tipCentre - normal * a.tipRadius
    guard let r = probe([lowest]) else { expect(false, "probe failed"); return }
    print(String(format: "        tip to yolk %.4f mm, tip to its own surface %.4f mm", r[0].food, r[0].ant))
    expect(abs(r[0].food) < 0.002, "the tip's nearest point is \(r[0].food) mm from the yolk")
    expect(abs(r[0].ant) < 0.002, "that point is not on the antenna: \(r[0].ant)")
    // No part of the antenna inside the sugar: sample a box round the tip.
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
    expect(worst > -0.002, "antenna reaches \(-worst) mm into the yolk")
}

test("in the inset the taste hair's apex rests on a yolk granule top at the origin") {
    guard let s = scene else { expect(false); return }
    let h: Sensillum = s.hairs[0]
    expect(abs(h.tip.y - h.tipRadius) < 1e-6 && abs(h.tip.x) < 1e-6 && abs(h.tip.z) < 1e-6, "apex not over the origin")
    // The GPU's yolk surface at the origin, and just below the apex.
    guard let r = probe([SIMD3<Float>(0, 0.0005, 0)]) else { expect(false, "probe failed"); return }
    print(String(format: "        drawn distance just above the contact: %.4f µm", r[0].insetDistance))
    expect(abs(r[0].insetDistance) < 0.01, "the surface at the contact is \(r[0].insetDistance) µm off")
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

section("smell: odour in the air")

/// Step 36's odour test, for one frame: what the smell hair has in the air.
/// Returns the problems found, and how many dots are drawn and resting at a pore.
func odourCheck(_ vols: [Volatile], hairs: [Sensillum]) -> (problems: [String], drawn: Int, arriving: Int) {
    let smell: Sensillum = hairs[1]
    let taste: Sensillum = hairs[0]
    var problems: [String] = []
    var arriving: Int = 0
    var drawn: Int = 0
    for v in vols where v.radius > 0 {
        drawn += 1
        if !(v.pore.row >= 0 && v.pore.row < smell.poreRows && v.pore.column >= 0 && v.pore.column < smell.poreColumns) {
            problems.append("pore \(v.pore) is not on the lattice")
        }
        let (mouth, outward) = smell.wallPore(row: v.pore.row, column: v.pore.column)
        let gap: Float = simd_distance(v.position, mouth)
        if v.age < odourEnter {
            // Outside the hair, on the outer side of its pore, and clear of both hairs.
            if simd_dot(v.position - mouth, outward) <= 0 { problems.append("an odour molecule inside the hair") }
            let ds: Float = roundConeDistance(v.position, smell.base, smell.tip, smell.baseRadius, smell.tipRadius)
            if ds < v.radius - 1e-4 { problems.append("an odour molecule in the smell hair's wall: \(ds)") }
            if gap < 2 * volatileDotRadius + smell.wallPoreRadius && v.radius >= volatileDotRadius - 1e-5 { arriving += 1 }
        } else if gap > odourRest + 1e-3 {
            // Going in: only ever between its resting place and the pore.
            problems.append("a molecule going in is \(gap) µm from its pore")
        }
        // None sits inside the taste hair or the yolk.
        let dt: Float = roundConeDistance(v.position, taste.base, taste.tip, taste.baseRadius, taste.tipRadius)
        if dt <= v.radius { problems.append("an odour molecule in the taste hair") }
        if v.position.y <= 0.6 + v.radius { problems.append("an odour molecule in the yolk") }
    }
    return (problems, drawn, arriving)
}

test("cooked yolk gives the smell hair something: odour molecules, each at or heading for a wall pore") {
    guard let s = scene, let f = contactFrame else { expect(false); return }
    let smell: Sensillum = s.hairs[1]
    let c = odourCheck(f.volatiles, hairs: s.hairs)
    for p in c.problems { expect(false, p) }
    expect(c.drawn >= 3, "only \(c.drawn) odour molecules")
    expect(c.arriving >= 1, "only \(c.arriving) arriving at a pore")
    // Every pore a molecule goes to is really carved: the drawn distance at
    // the pore mouth, just inside the wall, says "outside the hair".
    for t in 0..<odourPores.count {
        let p = odourPore(t, smell: smell, facing: insetDirection)
        guard let r = probe([p.mouth - p.outward * 0.02]) else { expect(false, "probe failed"); return }
        expect(r[0].smellHair > 0, "no pore at the mouth molecule \(t) is heading for: \(r[0].smellHair)")
    }
}

section("the molecules")

test("oleic acid is C18H34O2: 54 atoms, 53 bonds, no ring, one cis C=C and one C=O") {
    guard let s = scene else { expect(false); return }
    let m: Molecule = s.chemistry.oleic
    let f: [String: Int] = m.formula
    expect(f["C"] == 18 && f["H"] == 34 && f["O"] == 2, "oleic acid is \(f)")
    expectEqual(m.atoms.count, 54)
    expectEqual(m.bonds.count, 53)
    expectEqual(m.rings, 0)
    let doubles: [Int] = m.bonds.indices.filter { m.orders[$0] == 2 }
    let cc: [Int] = doubles.filter { m.atoms[m.bonds[$0].0].element == "C" && m.atoms[m.bonds[$0].1].element == "C" }
    expectEqual(doubles.count, 2)
    expectEqual(cc.count, 1)
    // cis: across the C=C, the two chain carbons are on the same side —
    // dihedral near 0°, not 180°.
    if let k = cc.first {
        let (a, b) = m.bonds[k]
        func carbonNeighbour(_ i: Int, not j: Int) -> Int? {
            m.bonds.compactMap { ($0.0 == i && $0.1 != j && m.atoms[$0.1].element == "C") ? $0.1 : (($0.1 == i && $0.0 != j && m.atoms[$0.0].element == "C") ? $0.0 : nil) }.first
        }
        if let c1 = carbonNeighbour(a, not: b), let c2 = carbonNeighbour(b, not: a) {
            let p0 = m.atoms[c1].position, p1 = m.atoms[a].position, p2 = m.atoms[b].position, p3 = m.atoms[c2].position
            let b1 = p1 - p0, b2 = p2 - p1, b3 = p3 - p2
            let n1 = simd_cross(b1, b2), n2 = simd_cross(b2, b3)
            let dih: Float = deg(atan2(simd_dot(simd_cross(n1, n2), simd_normalize(b2)), simd_dot(n1, n2)))
            print(String(format: "        C=C dihedral %.1f°", dih))
            expect(abs(dih) < 30, "the double bond is trans: \(dih)°")
        } else { expect(false, "no chain carbons on the double bond") }
    }
}

test("hexanal is C6H12O with a C=O at the chain end; every atom has its valence") {
    guard let s = scene else { expect(false); return }
    let m: Molecule = s.chemistry.odorant
    let f: [String: Int] = m.formula
    expect(f["C"] == 6 && f["H"] == 12 && f["O"] == 1, "hexanal is \(f)")
    expectEqual(m.rings, 0)
    for a in m.atoms { expectEqual(a.view, 1) }
    let all: Molecule = s.molecule
    let v: [Int] = all.valences
    for (i, a) in all.atoms.enumerated() {
        let want: Int = a.element == "C" ? 4 : (a.element == "O" ? 2 : 1)
        expect(v[i] == want, "\(a.element) \(i) has valence \(v[i])")
    }
}

test("the taste inset's waters are whole and clear of the acid") {
    guard let s = scene else { expect(false); return }
    expectEqual(s.chemistry.waters.count, 3)
    for w in s.chemistry.waters {
        expectEqual(w.formula["O"] ?? 0, 1)
        expectEqual(w.formula["H"] ?? 0, 2)
        for a in w.atoms { for b in s.chemistry.oleic.atoms {
            expect(simd_distance(a.position, b.position) > 2.0, "a water collides with the acid")
        } }
    }
}

section("the tap (step 30's, on yolk)")

test("the rate is the literature's, and the slow-down is stated: 4 strokes/s, 2.5 s a tap, ×10") {
    expect(realStrokeRange.contains(realStrokesPerSecond))
    expect(abs(slowdown - 10) < 1e-5, "slowed ×\(slowdown)")
    expect(loopSeconds >= 8 && loopSeconds <= 12, "loop \(loopSeconds) s")
    // The GIF plays the loop at its real length: frames × delay = loop.
    expect(abs(Float(defaultFrames * defaultDelayCentiseconds) / 100 - loopSeconds) < 1e-4)
    expectEqual(defaultFrames % tapsPerLoop, 0)
    // The odour is schematic, and the comment's arithmetic holds: n-hexane
    // (Tang et al. 2015) crosses 8 µm of air in microseconds.
    let seconds: Float = (8e-4 * 8e-4) / (2 * hexaneDiffusivity)
    expect(seconds > 1e-6 && seconds < 1e-5, "\(seconds) s")
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

/// The crumb's distance from the right antenna at one frame, on the GPU: the
/// tip's lowest point (along the contact normal), and the deepest any antenna
/// point reaches into the yolk among points packed round the tip.
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

test("at every touch-down the tip meets the yolk: 0 within 2 µm, on the GPU") {
    let downs: [FrameState] = frames.filter { $0.lift == 0 }
    expect(downs.count >= tapsPerLoop, "only \(downs.count) touching frames")
    var worst: Float = 0
    for f in downs {
        guard let r = tipSurvey(f, points: 0) else { expect(false, "probe failed"); return }
        worst = max(worst, abs(r.lowest))
    }
    print(String(format: "        %d touching frames; worst tip-to-yolk %.4f mm", downs.count, worst))
    expect(worst < 0.002, "a touching tip is \(worst) mm from the yolk")
}

test("at no frame does any part of the antenna go into the yolk") {
    var worst: Float = 1
    for f in frames {
        guard let r = tipSurvey(f, points: 12_000) else { expect(false, "probe failed"); return }
        worst = min(worst, r.deepest)
    }
    print(String(format: "        %d frames × 12,000 points: nearest antenna point to the inside of the yolk %.4f mm", frames.count, worst))
    expect(worst > -0.002, "the antenna reaches \(-worst) mm into the yolk")
}

test("lifted, the tip clears the yolk — by about the lift, and never less than 0.03 mm near the top") {
    let ups: [FrameState] = frames.filter { $0.lift > 0.5 * liftHeight }
    expect(!ups.isEmpty, "no lifted frames")
    var least: Float = 1
    var worstShort: Float = 0
    for f in ups {
        guard let r = tipSurvey(f, points: 4000) else { expect(false, "probe failed"); return }
        least = min(least, r.lowest)
        // The crumb is lumpy, not a face: the nearest yolk may be a neighbouring
        // sphere, so the clearance may fall a little short of the lift — never over it.
        expect(r.lowest <= f.lift + 0.002, "lift \(f.lift) mm but the tip is \(r.lowest) mm up")
        worstShort = max(worstShort, f.lift - r.lowest)
        expect(r.deepest > 0, "a lifted antenna point is in the yolk")
    }
    print(String(format: "        %d lifted frames, least clearance %.3f mm (lift %.3f mm at the top); clearance at most %.3f mm short of the lift",
                 ups.count, least, liftHeight, worstShort))
    expect(worstShort < 0.2 * liftHeight, "the clearance falls \(worstShort) mm short of the lift")
    expect(least > 0.03)
}

test("the taste hair meets the yolk's film at touch-down, and leaves it when lifted (inset, GPU)") {
    guard let s = scene, let c = contactFrame else { expect(false); return }
    let h: Sensillum = s.hairs[0]
    let apexLow = SIMD3<Float>(h.tip.x, h.tip.y - h.tipRadius, h.tip.z)
    let beside: SIMD3<Float> = apexLow + SIMD3<Float>(0.7, 0.05, 0)
    expect(abs(c.crystal.toCrystal(apexLow).y) < 0.002, "at touch-down the apex is \(c.crystal.toCrystal(apexLow).y) µm off the yolk")
    guard let r0 = probe([beside, SIMD3<Float>(0, 0.0005, 0)], frame: c) else { expect(false, "probe failed"); return }
    expect(r0[0].insetMaterial == 5 && r0[0].insetDistance < 0, "beside the apex at touch-down: material \(r0[0].insetMaterial)")
    var lowestLifted: Float = 1e9
    var sideways: Float = 0
    for f in frames where f.lift > 0.001 {
        let y: Float = f.crystal.toCrystal(apexLow).y
        lowestLifted = min(lowestLifted, y - 1000 * f.lift)
        let q: SIMD3<Float> = f.crystal.toCrystal(apexLow)
        sideways = max(sideways, SIMD2<Float>(q.x, q.z).squaredLength().squareRoot())
        expect(y > filmThickness + meniscusHeight, "lifted \(f.lift) mm, the apex is only \(y) µm up")
        guard let r = probe([beside], frame: f) else { expect(false, "probe failed"); return }
        expect(r[0].insetMaterial != 5, "lifted, the film is still at the apex")
    }
    print(String(format: "        yolk drop vs lift, worst difference %.2f µm; the yolk slides at most %.2f µm sideways", abs(lowestLifted), sideways))
    // The inset's y is the contact normal, the direction the tip lifts along:
    // so the yolk drops by the lift, straight down in the inset.
    expect(abs(lowestLifted) < 3, "the inset yolk and the main-view lift disagree by \(lowestLifted) µm")
    expect(sideways < 10, "the yolk slides \(sideways) µm sideways")
}

test("in the inset no hair or cuticle ever reaches into the yolk or its film (GPU, every frame)") {
    guard let s = scene else { expect(false); return }
    // Points on and just inside the taste hair round its apex, and on the
    // dome: at each, the drawn inset surface must be the hair (or dome), never the yolk.
    let h: Sensillum = s.hairs[0]
    var pts: [SIMD3<Float>] = []
    var rng = SystemRandomNumberGenerator()
    for _ in 0..<3000 {
        let d: SIMD3<Float> = simd_normalize(SIMD3<Float>(Float.random(in: -1...1, using: &rng), Float.random(in: -1...1, using: &rng),
                                                          Float.random(in: -1...1, using: &rng)))
        pts.append(h.tip + d * (h.tipRadius * Float.random(in: 0.5...0.999, using: &rng)))
    }
    var worst: Float = 1e9
    for f in frames {
        for p in pts { worst = min(worst, f.crystal.toCrystal(p).y + 0.3) }   // the granule floor is y = −0.3
        guard let r = probe(pts, frame: f) else { expect(false, "probe failed"); return }
        for x in r where x.insetMaterial == 1 { expect(false, "at t \(f.time) yolk drawn inside the taste hair"); break }
    }
    print(String(format: "        inside of the apex never below the yolk's floor: least height %.3f µm", worst))
    expect(worst > 0)
}

section("the odour in motion")

/// A dot visible in the inset circle.
func inInsetCircle(_ p: SIMD3<Float>) -> Bool {
    let q: SIMD2<Float> = projectInset(p, height: 1080)
    let c = SIMD2<Float>(insetCentre.x * 1080, insetCentre.y * 1080)
    return simd_distance(q, c) < insetRadius * 1080
}

extension SIMD2 where Scalar == Float {
    func squaredLength() -> Float { x * x + y * y }
}

test("the loop is forward: the last frame runs on into the first, and molecules only come in") {
    guard let s = scene, let c = contactFrame else { expect(false); return }
    // Time L is time 0.
    let end: FrameState = s.frame(at: loopSeconds)
    expect(simd_distance(end.ant.antennae[0].tipCentre, c.ant.antennae[0].tipCentre) < 1e-5)
    expectEqual(end.volatiles.count, c.volatiles.count)
    var endWorst: Float = 0
    for (a, b) in zip(end.volatiles, c.volatiles) { endWorst = max(endWorst, simd_distance(a.position, b.position), abs(a.radius - b.radius)) }
    expect(endWorst < 1e-3, "time L differs from time 0 by \(endWorst) µm")
    // The step from the last frame into the first is an ordinary step:
    // dots matched by nearest, since slots relabel once a tap.
    func step(_ a: FrameState, _ b: FrameState) -> Float {
        var m: Float = simd_distance(a.ant.antennae[0].tipCentre, b.ant.antennae[0].tipCentre) * 1000
        for v in b.volatiles where v.radius > 0 {
            let d: Float = a.volatiles.map { simd_distance($0.position, v.position) }.min() ?? 1e9
            m = max(m, d)
        }
        return m
    }
    var biggest: Float = 0
    for i in 1..<frames.count { biggest = max(biggest, step(frames[i - 1], frames[i])) }
    let seam: Float = step(frames[frames.count - 1], frames[0])
    print(String(format: "        seam step %.3f, largest step inside the loop %.3f", seam, biggest))
    expect(seam <= biggest * 1.01 + 1e-5, "the seam jumps")
    // Forward: progress never falls, and a loop brings in exactly one molecule a tap.
    var last: Float = odourProgress(0, mutant: s.mutant)
    for i in 1...frames.count {
        let u: Float = odourProgress(frameTime(i, of: frames.count) - (i == frames.count ? 1e-4 : 0), mutant: s.mutant)
        expect(u >= last - 1e-5, "the odour goes back at frame \(i): \(last) → \(u)")
        last = u
    }
    expect(abs(last - Float(tapsPerLoop)) < 0.01, "the loop brings in \(last) molecules, not \(tapsPerLoop)")
    // Every molecule only gets older, and never further from its pore once there.
    for i in 1..<frames.count {
        for (a, b) in zip(frames[i - 1].volatiles, frames[i].volatiles) where b.age >= a.age {
            expect(b.age - a.age < 0.5, "a molecule ages \(b.age - a.age) taps in one frame")
        }
    }
    // Each tap is the same: the lift is periodic in the tap.
    for i in 0..<(frames.count - frames.count / tapsPerLoop) {
        expect(abs(frames[i].lift - frames[i + frames.count / tapsPerLoop].lift) < 1e-5)
    }
}

test("each molecule goes in forward: along its way it only gets nearer its pore, then goes in") {
    guard let s = scene else { expect(false); return }
    let smell: Sensillum = s.hairs[1]
    var worstBack: Float = 0
    for t in 0..<odourPores.count {
        let p = odourPore(t, smell: smell, facing: insetDirection)
        var lastDepth: Float = -1e9
        var last: Float = 1e9
        for k in 0...400 {
            let a: Float = odourLife * Float(k) / 400
            let d = odourDot(age: a, target: t, smell: smell, facing: insetDirection)
            // Past the bend, straight distance to the pore falls; going in, depth grows.
            if a > 0.6 * odourArrive && a <= odourArrive {
                let gap: Float = simd_distance(d.position, p.mouth)
                worstBack = max(worstBack, gap - last)
                last = gap
            }
            if a >= odourEnter {
                let depth: Float = -simd_dot(d.position - p.mouth, p.outward)
                expect(depth >= lastDepth - 1e-5, "a molecule comes back out of its pore")
                lastDepth = depth
            }
        }
        let gone = odourDot(age: odourLife, target: t, smell: smell, facing: insetDirection)
        expect(gone.radius < 1e-6, "a molecule is still drawn when it has gone in")
    }
    expect(worstBack < 1e-4, "a molecule backs away from its pore by \(worstBack) µm")
}

test("nothing pops: inside the circle every odour dot moves continuously, even as slots relabel") {
    guard let s = scene else { expect(false); return }
    // A millisecond apart, a moving dot goes a few thousandths of a µm; one
    // that appears or vanishes has no partner near it at all. Check at every
    // frame, and either side of each instant the slots relabel — the start
    // of each tap — and the loop's seam. A dot shrinking into its pore may
    // vanish, but only once it has shrunk to nothing.
    var times: [Float] = (0..<frames.count).map { frameTime($0, of: frames.count) }
    for k in 0..<tapsPerLoop {
        let edge: Float = Float(k) * tapSeconds
        times += [edge - 2e-3, edge - 1e-3, edge, edge + 1e-3]
    }
    times.append(loopSeconds - 1e-3)
    var pops: Int = 0
    var worst: Float = 0
    var worstSize: Float = 0
    func visible(_ v: Volatile) -> Bool { v.radius > 0.005 && inInsetCircle(v.position) }
    for t in times {
        let a: FrameState = s.frame(at: t)
        let b: FrameState = s.frame(at: t + 1e-3)
        for (x, y) in [(a, b), (b, a)] {
            for v in y.volatiles where visible(v) {
                let near: Volatile? = x.volatiles.min { simd_distance($0.position, v.position) < simd_distance($1.position, v.position) }
                guard let n = near else { pops += 1; continue }
                let d: Float = simd_distance(n.position, v.position)
                if d > 0.05 || abs(n.radius - v.radius) > 0.01 { pops += 1 } else {
                    worst = max(worst, d)
                    worstSize = max(worstSize, abs(n.radius - v.radius))
                }
            }
        }
    }
    print(String(format: "        %d moments: largest move in 1 ms %.4f µm, size change %.4f µm; %d dots appear or vanish in the circle",
                 times.count, worst, worstSize, pops))
    expectEqual(pops, 0)
    // And a dot not yet full size is never inside the circle unless it is going in.
    for f in frames {
        for v in f.volatiles where v.radius > 0 && v.radius < volatileDotRadius - 1e-4 && v.age < odourEnter {
            expect(!inInsetCircle(v.position), "a dot grows inside the circle at t \(f.time)")
        }
    }
}

test("the smell state at every frame: yolk smells — dots in the air, one at a pore, none in a hair or the yolk") {
    guard let s = scene else { expect(false); return }
    expect(foodHasVapour)
    var fewest: Int = 99
    var leastArriving: Int = 99
    var problems: Int = 0
    var closest: Float = 1e9
    for f in frames {
        let c = odourCheck(f.volatiles, hairs: s.hairs)
        problems += c.problems.count
        if let p = c.problems.first { expect(false, "t \(f.time): \(p)") }
        fewest = min(fewest, c.drawn)
        leastArriving = min(leastArriving, c.arriving)
        // Dots never overlap one another.
        let d: [Volatile] = f.volatiles.filter { $0.radius > 0 }
        for i in 0..<d.count { for j in (i + 1)..<d.count {
            closest = min(closest, simd_distance(d[i].position, d[j].position) - d[i].radius - d[j].radius)
        } }
    }
    print("        \(frames.count) frames: at least \(fewest) dots drawn and \(leastArriving) resting at a pore in every frame; closest two dots \(closest) µm apart")
    expectEqual(problems, 0)
    expect(fewest >= 3, "a frame with only \(fewest) odour dots")
    expect(leastArriving >= 1, "a frame with no molecule at a pore")
    expect(closest > 0, "two odour dots overlap")
    // And the GPU draws them: at every frame, each drawn dot's centre probes
    // as odour (material 6), inside its dot.
    for f in frames {
        let pts: [SIMD3<Float>] = f.volatiles.filter { $0.radius > 0.02 && $0.age < odourEnter }.map { $0.position }
        guard let r = probe(pts, frame: f) else { expect(false, "probe failed"); return }
        for x in r { expect(x.insetMaterial == 6 && x.insetDistance < 0, "a dot not drawn at t \(f.time): \(x.insetMaterial)") }
    }
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
    // As step 30: round the tip and the crumb with the antenna at its highest.
    if let up = frames.max(by: { $0.lift < $1.lift }) {
        for (m, v) in worstOverReport(box: SIMD3(2.0, 0, 0.3), SIMD3(3.2, 1.2, 1.6), step: 0.004, inset: false, frame: up) {
            w[m] = max(w[m] ?? 0, v)
        }
    }
    print(String(format: "        worst over-report: table %.2f, ant %.2f, yolk %.2f; the ray allows %.2f",
                 w[1] ?? 0, w[2] ?? 0, w[3] ?? 0, 1 / stepScale))
    expect(w.count == 3, "not every material was sampled: \(w)")
    for (m, v) in w { expect(v * stepScale <= 1.0, "material \(m) oversteps: \(v)") }
}

test("inset: the hairs, the dome, the yolk granules and the film are honest outside too") {
    let w: [Int: Float] = worstOverReport(box: SIMD3(-4, -0.5, -12), SIMD3(8, 12, 4), step: 0.02, inset: true)
    // A second pass concentrated round the tip, where the film and pore are.
    let t: [Int: Float] = worstOverReport(box: SIMD3(-1.5, -0.2, -1.5), SIMD3(1.5, 1.5, 1.5), step: 0.005, inset: true)
    // And a third in a thin shell round each hair's wall, down among the pores.
    let pores: [Int: Float] = worstOverReport(box: .zero, .zero, step: 0.004, inset: true, points: hairShellPoints(count: 80_000))
    var all: [Int: Float] = w
    for (m, v) in t { all[m] = max(all[m] ?? 0, v) }
    for (m, v) in pores { all[m] = max(all[m] ?? 0, v) }
    // As step 30: and with the antenna lifted, when the yolk is moved.
    if let up = frames.max(by: { $0.lift < $1.lift }), up.lift > 0 {
        let yolkBox: SIMD3<Float> = up.crystal.translation
        for (m, v) in worstOverReport(box: SIMD3(-4, -0.5, -12) - yolkBox, SIMD3(8, 12, 4) - yolkBox, step: 0.02,
                                      inset: true, frame: up) {
            all[m] = max(all[m] ?? 0, v)
        }
        for (m, v) in worstOverReport(box: SIMD3(-4, -0.5, -8), SIMD3(8, 12, 4), step: 0.01, inset: true, frame: up) {
            all[m] = max(all[m] ?? 0, v)
        }
    }
    // And round the odour dots, at a frame where one is going into its pore.
    if let going = frames.first(where: { f in f.volatiles.contains { $0.age > odourEnter + 0.1 && $0.radius > 0.02 } }) {
        var near: [SIMD3<Float>] = []
        var rng = SystemRandomNumberGenerator()
        for v in going.volatiles where v.radius > 0 {
            for _ in 0..<20_000 {
                near.append(v.position + SIMD3<Float>(Float.random(in: -0.4...0.4, using: &rng), Float.random(in: -0.4...0.4, using: &rng),
                                                      Float.random(in: -0.4...0.4, using: &rng)))
            }
        }
        for (m, v) in worstOverReport(box: .zero, .zero, step: 0.004, inset: true, points: near, frame: going) {
            all[m] = max(all[m] ?? 0, v)
        }
    }
    print(String(format: "        worst over-report: yolk %.2f, cuticle %.2f, taste hair %.2f, smell hair %.2f, film %.2f, odour %.2f",
                 all[1] ?? 0, all[2] ?? 0, all[3] ?? 0, all[4] ?? 0, all[5] ?? 0, all[6] ?? 0))
    expect(all.count >= 5, "not every material was sampled: \(all)")
    for (m, v) in all { expect(v * stepScale <= 1.0, "material \(m) oversteps: \(v)") }
}

section("the picture")

let render: AntImage? = {
    guard let d = gpu.device, let s = scene, let f = contactFrame else { return nil }
    return try? renderAnt(width: 960, height: 540, samples: 1, scene: s, frame: f, on: d).image
}()

test("the touch is visible: antenna and yolk both within 3 px of the projected contact") {
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
    expect(sawAnt && sawSugar, "near the contact: antenna \(sawAnt), yolk \(sawSugar)")
}

test("the inset shows both hairs, the film and odour dots; both molecule insets show C, O and H") {
    guard let img = render else { expect(false, "render failed"); return }
    var seen: Set<Int> = []
    var taste: Set<Int> = []
    var smell: Set<Int> = []
    for y in 0..<img.height {
        for x in 0..<img.width {
            let v: SIMD4<Float> = img.seen(x, y)
            if v.x == 2 { seen.insert(Int(v.y)) }
            if v.x == 3 && v.y >= 1 && v.y <= 3 { taste.insert(Int(v.y)) }
            if v.x == 4 && v.y >= 1 && v.y <= 3 { smell.insert(Int(v.y)) }
        }
    }
    expect(seen.isSuperset(of: [3, 4, 5, 6]), "inset materials seen: \(seen.sorted())")
    expectEqual(taste, [1, 2, 3])
    expectEqual(smell, [1, 2, 3])
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

test("only the moving region changes: outside it — the molecule insets too — frames differ from the first by float noise at most") {
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
    var molecule: Int = 0
    let hf: Float = Float(h)
    func inMoleculeInset(_ x: Int, _ y: Int) -> Bool {
        let p = SIMD2<Float>(Float(x) + 0.5, Float(y) + 0.5) / hf
        return simd_distance(p, moleculeCentre) < moleculeRadius || simd_distance(p, odourCentre) < odourRadius
    }
    let perTap: Int = frames.count / tapsPerLoop
    for f in [perTap / 5, perTap / 2, perTap * 7 / 10, perTap * 9 / 10, perTap + 3] {
        _ = try? r.render(frames[f], samples: 1)
        let full: [UInt8] = Array(UnsafeBufferPointer(start: r.pixels.contents().assumingMemoryBound(to: UInt8.self), count: n))
        for y in 0..<h {
            for x in 0..<w {
                let i: Int = (y * w + x) * 4
                if full[i] != base[i] || full[i + 1] != base[i + 1] || full[i + 2] != base[i + 2] {
                    if inMoleculeInset(x, y) { molecule += 1 }
                    if region.contains(x, y) { inside += 1 } else {
                        outside += 1
                        if ProcessInfo.processInfo.environment["ANT_DEBUG"] != nil && outside % 20 == 1 { print("        outside change at \(x), \(y) frame \(f)") }
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
    print("        region \(region.rects.count) rects, \(100 * region.area / (w * h))% of the frame; changed pixels inside \(inside), outside \(outside), worst outside \(worstOutside) levels; in the molecule insets \(molecule)")
    expect(worstOutside <= 1, "a pixel outside the region changes by \(worstOutside) levels")
    expect(outside < w * h / 1000, "\(outside) pixels outside the region change")
    expectEqual(molecule, 0)
    expect(inside > 0, "nothing moved")
}

test("lifted, the main view shows the tip raised off the yolk, and the inset no film at the tip") {
    guard let d = gpu.device, let s = scene, let up = frames.max(by: { $0.lift < $1.lift }) else { expect(false); return }
    guard let img = try? renderAnt(width: 1920, height: 1080, samples: 1, scene: s, frame: up, on: d).image else {
        expect(false, "render failed"); return
    }
    let a: Antenna = up.ant.antennae[0]
    let low: SIMD2<Float> = projectMain(a.tipCentre - s.contact.normal * a.tipRadius, width: 1920, height: 1080)
    let touch: SIMD2<Float> = projectMain(s.contact.point, width: 1920, height: 1080)
    print(String(format: "        tip raised %.1f px on screen", simd_distance(low, touch)))
    expect(simd_distance(low, touch) > 10, "the lift is invisible: \(simd_distance(low, touch)) px")
    let apex: SIMD2<Float> = projectInset(s.hairs[0].tip, height: 1080)
    var film: Int = 0
    for dy in -12...12 { for dx in -12...12 where img.seen(Int(apex.x) + dx, Int(apex.y) + dy).y == 5 { film += 1 } }
    expectEqual(film, 0)
}

finish()
