// Tests for step 35: step 34's tests, copied (run on the contact frame, where
// the scene is step 34's), then the motion as step 30 tests it — the touch at
// every touch-down, no penetration at any frame, a clear lift, a forward loop —
// the sugars going into the taste pore one per touch, and the odour dots
// arriving at the smell hair's pores at every frame.
//
// Step 34's tests. The anatomy is checked against the literature it cites,
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

/// Every frame of the GIF's loop, and the first — the touch-down, step 34's pose.
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

section("the honey")

test("the drop is a spherical cap meeting the card at its contact angle, 0° < θ < 90°") {
    expect(dropContactAngle > 0 && dropContactAngle < Float.pi / 2, "contact angle \(deg(dropContactAngle))°")
    // The rim: where the sphere meets the card, radius a.
    let rimR: Float = (dropSphereRadius * dropSphereRadius - dropCentre.y * dropCentre.y).squareRoot()
    expect(abs(rimR - dropBaseRadius) < 1e-5, "rim radius \(rimR) mm")
    // The angle between the card and the surface tangent at the rim is θ.
    let rim = SIMD3<Float>(dropAt.x + dropBaseRadius, 0, dropAt.y)
    let n: SIMD3<Float> = simd_normalize(rim - dropCentre)
    let theta: Float = acos(simd_dot(n, SIMD3<Float>(0, 1, 0)))
    expect(abs(theta - dropContactAngle) < 1e-4, "angle at the rim \(deg(theta))°")
    print(String(format: "        drop %.2f mm wide, %.2f mm high, %.0f° at the rim", 2 * dropBaseRadius, dropHeight, deg(theta)))
}

test("gravity cannot flatten it: Bond number under 0.1 even at a low surface tension") {
    let bond: Float = honeyDensity * 9.81 * pow(dropBaseRadius / 1000, 2) / honeySurfaceTensionFloor
    print(String(format: "        Bond number %.3f", bond))
    expect(bond < 0.1, "Bond number \(bond): a cap is the wrong shape")
}

test("the drop is ant-sized: 0.5–1.5 mm across, and lower than the ant's head") {
    expect(2 * dropBaseRadius >= 0.5 && 2 * dropBaseRadius <= 1.5)
    expect(dropHeight < headCentre.y)
}

test("honey's optics and make-up: n 1.4935 at 17.2% water, amber within USDA's amber grade") {
    expect(abs(honeyIndex - 1.4935) < 1e-6)
    expect(honeyIndex > waterIndex && honeyIndex < 1.56)
    expect(amberOpticalDensity > 1.389 && amberOpticalDensity <= 3.008)
    expect(honeyAbsorptionRGB.z > honeyAbsorptionRGB.y && honeyAbsorptionRGB.y > honeyAbsorptionRGB.x, "not amber: \(honeyAbsorptionRGB)")
    expect(abs(honeyFructose + honeyGlucose + honeyWater - 86.67) < 0.01)
    expect(foodHasVapour)
}

section("the touch")

test("the right antenna's tip touches the drop: distance 0 within 2 µm, and no penetration") {
    guard let s = contactFrame else { expect(false); return }
    let a: Antenna = s.ant.antennae[0]
    let (point, normal) = contactPoint()
    // On the CPU: the contact point lies on the drop's surface, above the card.
    expect(abs(dropSDF(point)) < 1e-5, "contact point is off the surface: \(dropSDF(point))")
    expect(point.y > 0.05, "contact too low on the drop: \(point.y)")
    // On the GPU: the lowest point of the tip is on the sugar.
    let lowest: SIMD3<Float> = a.tipCentre - normal * a.tipRadius
    guard let r = probe([lowest]) else { expect(false, "probe failed"); return }
    print(String(format: "        tip to honey %.4f mm, tip to its own surface %.4f mm", r[0].food, r[0].ant))
    expect(abs(r[0].food) < 0.002, "the tip's nearest point is \(r[0].food) mm from the honey")
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
    expect(worst > -0.002, "antenna reaches \(-worst) mm into the honey")
}

test("in the inset the taste hair's apex just meets the honey surface — the meniscus, not a dip") {
    guard let s = scene else { expect(false); return }
    let h: Sensillum = s.hairs[0]
    let R: Float = dropSphereRadius * 1000
    func honey(_ p: SIMD3<Float>) -> Float { simd_length(p - SIMD3<Float>(0, -R, 0)) - R }
    var lowest: Float = 1e9
    var rng = SystemRandomNumberGenerator()
    for _ in 0..<20_000 {
        let d: SIMD3<Float> = simd_normalize(SIMD3<Float>(Float.random(in: -1...1, using: &rng), Float.random(in: -1...1, using: &rng),
                                                          Float.random(in: -1...1, using: &rng)))
        let t: Float = Float.random(in: 0...1, using: &rng)
        let p: SIMD3<Float> = h.base + (h.tip - h.base) * t + d * (h.baseRadius + (h.tipRadius - h.baseRadius) * t)
        if roundConeDistance(p, h.base, h.tip, h.baseRadius, h.tipRadius) <= 1e-4 { lowest = min(lowest, honey(p)) }
    }
    // And the apex cap's lowest point, exactly.
    lowest = min(lowest, honey(h.tip - SIMD3<Float>(0, h.tipRadius, 0)))
    print(String(format: "        hair's deepest point relative to the honey surface: %.4f µm", lowest))
    expect(lowest > -0.005, "the hair dips \(-lowest) µm into the honey")
    expect(lowest < 0.02, "the hair hovers \(lowest) µm above the honey")
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

section("smell: odour in the air, at every frame")

/// Odour dots that are whole (not going into their pore), and the pore check.
func odourCheck(_ f: FrameState, hairs: [Sensillum]) -> (drawn: Int, atPore: Int, faults: [String]) {
    let smell: Sensillum = hairs[1]
    let taste: Sensillum = hairs[0]
    var faults: [String] = []
    var drawn: Int = 0
    var atPore: Int = 0
    for v in f.volatiles where v.radius > 0 {
        drawn += 1
        if !(v.pore.row >= 0 && v.pore.row < smell.poreRows && v.pore.column >= 0 && v.pore.column < smell.poreColumns) {
            faults.append("pore \(v.pore) is not on the lattice")
        }
        let (mouth, outward) = smell.wallPore(row: v.pore.row, column: v.pore.column)
        let gap: Float = simd_distance(v.position, mouth)
        if gap < 2 * volatileDotRadius + smell.wallPoreRadius { atPore += 1 }
        let toSmell: Float = roundConeDistance(v.position, smell.base, smell.tip, smell.baseRadius, smell.tipRadius)
        if v.along < odourEnter {
            // In the air: outside the smell hair, clear of it by its own radius
            // except where it rests on its pore's outer side.
            if v.along >= odourArrive && simd_dot(v.position - mouth, outward) <= 0 { faults.append("a resting dot on the inner side of its pore") }
            if toSmell < v.radius - 0.1 * volatileDotRadius && gap > 0.2 { faults.append("a dot inside the smell hair, \(toSmell)") }
        } else {
            // Going in: on its pore's own axis, within a pore radius of it.
            let off: SIMD3<Float> = (v.position - mouth) - outward * simd_dot(v.position - mouth, outward)
            if simd_length(off) > smell.wallPoreRadius { faults.append("a dot going in beside its pore, \(simd_length(off)) µm off") }
        }
        if roundConeDistance(v.position, taste.base, taste.tip, taste.baseRadius, taste.tipRadius) < v.radius {
            faults.append("a dot inside the taste hair")
        }
        // Never in the honey, wherever the honey is at that frame.
        let y: Float = f.crystal.toCrystal(v.position).y
        if y < v.radius { faults.append("a dot in the honey, y \(y)") }
    }
    return (drawn, atPore, faults)
}

test("honey gives the smell hair something at every frame: dots in the air, arriving at wall pores") {
    guard let s = scene else { expect(false); return }
    expect(foodHasVapour)
    var fewest: Int = 99
    var fewestAtPore: Int = 99
    for f in frames {
        let (drawn, atPore, faults) = odourCheck(f, hairs: s.hairs)
        for x in faults.prefix(3) { expect(false, "t \(f.time): \(x)") }
        fewest = min(fewest, drawn)
        fewestAtPore = min(fewestAtPore, atPore)
    }
    print("        every frame: at least \(fewest) dots drawn, at least \(fewestAtPore) at a pore")
    expect(fewest >= 3, "a frame with only \(fewest) odour dots")
    expect(fewestAtPore >= 1, "a frame with no odour dot at a pore")
    // The pores the dots go to are really carved: the drawn distance at each
    // pore mouth, just inside the wall, says "outside the hair".
    for lane in s.lanes {
        guard let r = probe([lane.mouth - lane.outward * 0.02]) else { expect(false, "probe failed"); return }
        expect(r[0].smellHair > 0, "no pore at the mouth the dots are heading for: \(r[0].smellHair)")
    }
}

test("the odour inset shows the dots arriving, and the drawn dots match the dots sent (GPU)") {
    guard let f = contactFrame else { expect(false); return }
    // At each whole dot's centre the inset's distance is minus its radius, and
    // the material is odour (6).
    let whole: [Volatile] = f.volatiles.filter { $0.radius > 0.5 * volatileDotRadius }
    expect(!whole.isEmpty)
    guard let r = probe(whole.map { $0.position }, frame: f) else { expect(false, "probe failed"); return }
    for (v, p) in zip(whole, r) {
        expectEqual(p.insetMaterial, 6)
        expect(abs(p.insetDistance + v.radius) < 1e-3, "dot drawn at distance \(p.insetDistance), radius \(v.radius)")
    }
}

section("the molecules")

test("fructose and glucose are both C6H12O6: 24 atoms, 24 bonds, one ring each, straight from their files") {
    guard let s = scene else { expect(false); return }
    for (name, m) in [("fructose", s.chemistry.fructose), ("glucose", s.chemistry.glucose)] {
        let f: [String: Int] = m.formula
        expect(f["C"] == 6 && f["H"] == 12 && f["O"] == 6, "\(name) is \(f)")
        expectEqual(m.atoms.count, 24)
        expectEqual(m.bonds.count, 24)
        expectEqual(m.rings, 1)
    }
    // Same formula, different molecules: fructose's ring holds five carbons
    // and an oxygen with the anomeric carbon bonded to two carbons (a
    // ketose); glucose's anomeric carbon carries one carbon (an aldose).
    func anomericCarbons(_ m: Molecule) -> [Int] {
        // A carbon bonded to two oxygens is the anomeric one.
        (0..<m.atoms.count).filter { i in
            m.atoms[i].element == "C" && m.bonds.filter { ($0.0 == i && m.atoms[$0.1].element == "O") || ($0.1 == i && m.atoms[$0.0].element == "O") }.count == 2
        }
    }
    func carbonNeighbours(_ m: Molecule, _ i: Int) -> Int {
        m.bonds.filter { ($0.0 == i && m.atoms[$0.1].element == "C") || ($0.1 == i && m.atoms[$0.0].element == "C") }.count
    }
    let fa: [Int] = anomericCarbons(s.chemistry.fructose)
    let ga: [Int] = anomericCarbons(s.chemistry.glucose)
    expectEqual(fa.count, 1)
    expectEqual(ga.count, 1)
    if let f0 = fa.first { expectEqual(carbonNeighbours(s.chemistry.fructose, f0), 2) }
    if let g0 = ga.first { expectEqual(carbonNeighbours(s.chemistry.glucose, g0), 1) }
}

test("every atom has its valence (bond orders summed): C 4, O 2, H 1 — sugars, waters and odorant, every frame") {
    var bad: Int = 0
    for f in frames {
        let all: Molecule = f.molecule
        let v: [Int] = all.valences
        for (i, a) in all.atoms.enumerated() {
            let want: Int = a.element == "C" ? 4 : (a.element == "O" ? 2 : 1)
            if v[i] != want { bad += 1; if bad < 4 { expect(false, "t \(f.time): \(a.element) \(i) has valence \(v[i])") } }
        }
    }
    expectEqual(bad, 0)
}

test("phenylacetaldehyde is C8H8O with one aromatic ring and a C=O") {
    guard let s = scene else { expect(false); return }
    let m: Molecule = s.chemistry.odorant
    let f: [String: Int] = m.formula
    expect(f["C"] == 8 && f["H"] == 8 && f["O"] == 1, "odorant is \(f)")
    expectEqual(m.rings, 1)
    let carbonyl: Int = m.bonds.indices.filter { k in
        m.orders[k] == 2 && (m.atoms[m.bonds[k].0].element == "O" || m.atoms[m.bonds[k].1].element == "O")
    }.count
    expectEqual(carbonyl, 1)
    for a in m.atoms { expectEqual(a.view, 1) }
}

test("every frame, the taste inset holds honey's own ratio: 2.5 waters per sugar, as many fructose as glucose") {
    print(String(format: "        honey: %.2f waters per hexose", watersPerHexose))
    expect(abs(watersPerHexose - 2.5) < 0.1)
    expectEqual(fructoseWaterSpots.count + glucoseWaterSpots.count, Int((2 * watersPerHexose).rounded()))
    for f in frames {
        let fru: Int = f.honey.units.filter { $0.kind == .fructose }.count
        let glc: Int = f.honey.units.filter { $0.kind == .glucose }.count
        expectEqual(fru, glc)
        expectEqual(Float(f.honey.waters.count) / Float(f.honey.sugars.count), 2.5)
    }
}

section("the tap (step 30's, on honey)")

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

/// The honey's distance from the right antenna at one frame, on the GPU: the
/// tip's lowest point (towards the drop), and the deepest any antenna point
/// reaches into the honey among points packed round the tip.
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

test("at every touch-down the tip meets the honey: 0 within 2 µm, on the GPU") {
    let downs: [FrameState] = frames.filter { $0.lift == 0 }
    expect(downs.count >= tapsPerLoop, "only \(downs.count) touching frames")
    var worst: Float = 0
    for f in downs {
        guard let r = tipSurvey(f, points: 0) else { expect(false, "probe failed"); return }
        worst = max(worst, abs(r.lowest))
    }
    print(String(format: "        %d touching frames; worst tip-to-honey %.4f mm", downs.count, worst))
    expect(worst < 0.002, "a touching tip is \(worst) mm from the honey")
}

test("at no frame does any part of the antenna go into the honey") {
    var worst: Float = 1
    for f in frames {
        guard let r = tipSurvey(f, points: 12_000) else { expect(false, "probe failed"); return }
        worst = min(worst, r.deepest)
    }
    print(String(format: "        %d frames × 12,000 points: nearest antenna point to the inside of the honey %.4f mm", frames.count, worst))
    expect(worst > -0.002, "the antenna reaches \(-worst) mm into the honey")
}

test("lifted, the tip clears the honey — by the lift, and never less than 0.03 mm near the top") {
    let ups: [FrameState] = frames.filter { $0.lift > 0.5 * liftHeight }
    expect(!ups.isEmpty, "no lifted frames")
    var least: Float = 1
    for f in ups {
        guard let r = tipSurvey(f, points: 4000) else { expect(false, "probe failed"); return }
        least = min(least, r.lowest)
        expect(abs(r.lowest - f.lift) < 0.002, "lift \(f.lift) mm but the tip is \(r.lowest) mm up")
        expect(r.deepest > 0, "a lifted antenna point is in the honey")
    }
    print(String(format: "        %d lifted frames, least clearance %.3f mm (lift %.3f mm at the top)", ups.count, least, liftHeight))
    expect(least > 0.03)
}

test("the taste hair meets the honey surface at touch-down, and leaves it when lifted (inset, GPU)") {
    guard let s = scene, let c = contactFrame else { expect(false); return }
    let h: Sensillum = s.hairs[0]
    let R: Float = dropSphereRadius * 1000
    func honey(_ p: SIMD3<Float>) -> Float { simd_length(p - SIMD3<Float>(0, -R, 0)) - R }
    let apexLow = SIMD3<Float>(h.tip.x, h.tip.y - h.tipRadius, h.tip.z)
    let beside: SIMD3<Float> = apexLow + SIMD3<Float>(0.7, 0.05, 0)
    let d0: Float = honey(c.crystal.toCrystal(apexLow))
    expect(abs(d0) < 0.005, "at touch-down the apex is \(d0) µm off the honey")
    guard let r0 = probe([beside, apexLow + SIMD3<Float>(0, -0.02, 0)], frame: c) else { expect(false, "probe failed"); return }
    expect(r0[0].insetMaterial == 5 && r0[0].insetDistance < 0, "beside the apex at touch-down: material \(r0[0].insetMaterial)")
    expect(r0[1].insetDistance < 0, "just under the apex at touch-down is not honey: \(r0[1].insetDistance)")
    var worst: Float = 0
    for f in frames where f.lift > 0.001 {
        let y: Float = honey(f.crystal.toCrystal(apexLow))
        worst = max(worst, abs(y - 1000 * f.lift))
        expect(y > filmThickness + meniscusHeight, "lifted \(f.lift) mm, the apex is only \(y) µm up")
        guard let r = probe([beside], frame: f) else { expect(false, "probe failed"); return }
        expect(r[0].insetMaterial != 5, "lifted, the meniscus is still at the apex")
    }
    print(String(format: "        honey drop vs lift, worst difference %.2f µm", worst))
    expect(worst < 3, "the inset honey and the main-view lift disagree by \(worst) µm")
}

section("sugars and odour in motion")

/// Each atom of `b` matched to the nearest atom of the same element and view in `a`.
func matchAtoms(_ a: Molecule, _ b: Molecule) -> [(from: Atom, to: Atom, d: Float)] {
    var byElement: [String: [Atom]] = [:]
    for x in a.atoms { byElement[x.element + "\(x.view)", default: []].append(x) }
    return b.atoms.map { q in
        let cands: [Atom] = byElement[q.element + "\(q.view)"] ?? []
        var best: Atom = q
        var bd: Float = 1e9
        for c in cands {
            let d: Float = simd_distance_squared(c.position, q.position)
            if d < bd { bd = d; best = c }
        }
        return (best, q, bd.squareRoot())
    }
}

/// Is an atom inside its molecule circle, allowing for its own drawn size?
func inCircle(_ a: Atom) -> Bool {
    let field: Float = a.view == 0 ? moleculeField : odourField
    return (a.position.x * a.position.x + a.position.y * a.position.y).squareRoot() < field / 2 + 1.2
}

/// Is an odour dot inside the micrometre inset's circle, allowing for its size?
func dotInCircle(_ p: SIMD3<Float>) -> Bool {
    let q: SIMD3<Float> = p - insetCamera.centre
    let x: Float = simd_dot(q, insetCamera.right)
    let y: Float = simd_dot(q, insetCamera.up)
    return (x * x + y * y).squareRoot() < insetField / 2 + 2 * volatileDotRadius
}

/// Largest move of any visible odour dot to its nearest partner, and how many
/// visible dots have no partner near at all.
func dotSteps(_ a: FrameState, _ b: FrameState, far: Float) -> (worst: Float, pops: Int) {
    var worst: Float = 0
    var pops: Int = 0
    for (x, y) in [(a, b), (b, a)] {
        for v in y.volatiles where v.radius > 0.01 && dotInCircle(v.position) {
            var best: Float = 1e9
            for w in x.volatiles where w.lane == v.lane {
                let d: Float = simd_distance(w.position, v.position) + abs(w.radius - v.radius)
                best = min(best, d)
            }
            if best > far { pops += 1 } else { worst = max(worst, best) }
        }
    }
    return (worst, pops)
}

test("the loop is forward: the last frame runs on into the first, and sugars and odour only go in") {
    guard let s = scene, let c = contactFrame else { expect(false); return }
    // Time L is time 0.
    let end: FrameState = s.frame(at: loopSeconds)
    expect(simd_distance(end.ant.antennae[0].tipCentre, c.ant.antennae[0].tipCentre) < 1e-5)
    expectEqual(end.molecule.atoms.count, c.molecule.atoms.count)
    var endWorst: Float = 0
    for (a, b) in zip(end.molecule.atoms, c.molecule.atoms) { endWorst = max(endWorst, simd_distance(a.position, b.position)) }
    for (a, b) in zip(end.volatiles, c.volatiles) { endWorst = max(endWorst, simd_distance(a.position, b.position)) }
    expect(endWorst < 1e-3, "time L differs from time 0 by \(endWorst)")
    // The step from the last frame into the first is an ordinary step: atoms
    // and dots matched by nearest, since slots relabel.
    func step(_ a: FrameState, _ b: FrameState) -> (tip: Float, atoms: Float, dots: Float) {
        var m: Float = 0
        for (_, to, d) in matchAtoms(a.molecule, b.molecule) where inCircle(to) { m = max(m, d) }
        let tip: Float = simd_distance(a.ant.antennae[0].tipCentre, b.ant.antennae[0].tipCentre)
        return (tip, m, dotSteps(a, b, far: 1e9).worst)
    }
    var biggest: (tip: Float, atoms: Float, dots: Float) = (0, 0, 0)
    for i in 1..<frames.count {
        let x = step(frames[i - 1], frames[i])
        biggest = (max(biggest.tip, x.tip), max(biggest.atoms, x.atoms), max(biggest.dots, x.dots))
    }
    let seam = step(frames[frames.count - 1], frames[0])
    print(String(format: "        seam step: tip %.4f mm, atoms %.3f Å, dots %.3f µm; largest inside the loop %.4f, %.3f, %.3f",
                 seam.tip, seam.atoms, seam.dots, biggest.tip, biggest.atoms, biggest.dots))
    expect(seam.tip <= biggest.tip * 1.01 + 1e-6, "the tip jumps at the seam")
    expect(seam.atoms <= biggest.atoms * 1.01 + 1e-5, "the sugars jump at the seam")
    expect(seam.dots <= biggest.dots * 1.01 + 1e-5, "the odour jumps at the seam")
    // Forward: progress never falls, and a loop takes exactly one sugar per tap
    // and brings each lane one dot per tap.
    var lastSugar: Float = sugarProgress(0, mutant: s.mutant)
    var lastOdour: Float = odourProgress(0, mutant: s.mutant)
    for i in 1...frames.count {
        let t: Float = frameTime(i, of: frames.count) - (i == frames.count ? 1e-4 : 0)
        let u: Float = sugarProgress(t, mutant: s.mutant)
        let o: Float = odourProgress(t, mutant: s.mutant)
        expect(u >= lastSugar - 1e-5, "sugars go back at frame \(i): \(lastSugar) → \(u)")
        expect(o >= lastOdour - 1e-5, "odour goes back at frame \(i): \(lastOdour) → \(o)")
        lastSugar = u
        lastOdour = o
    }
    expect(abs(lastSugar - Float(tapsPerLoop)) < 0.01, "the loop takes \(lastSugar) sugars in, not \(tapsPerLoop)")
    expect(abs(lastOdour - Float(tapsPerLoop)) < 0.01, "the loop brings \(lastOdour) dots a lane, not \(tapsPerLoop)")
    // Each dot only goes forward along its lane, frame to frame, and each
    // sugar only goes further along the drift path, into the pore.
    for i in 1..<frames.count {
        for (a, b) in zip(frames[i - 1].volatiles, frames[i].volatiles) where b.along >= a.along {
            expect(b.along - a.along < 0.5, "a dot leaps at frame \(i)")
        }
    }
    // Each tap is the same: the lift is periodic in the tap.
    for i in 0..<(frames.count - frames.count / tapsPerLoop) {
        expect(abs(frames[i].lift - frames[i + frames.count / tapsPerLoop].lift) < 1e-5)
    }
}

test("nothing pops: inside the circles every atom and odour dot moves continuously, even as slots relabel") {
    guard let s = scene else { expect(false); return }
    // A millisecond apart, a moving atom goes a few hundredths of an ångström;
    // one that appears or vanishes has no partner near it at all. Check at
    // every frame, and either side of each instant the slots relabel — the
    // end of each touch for the sugars, every half tap for the odour lanes —
    // and the loop's seam.
    var times: [Float] = (0..<frames.count).map { frameTime($0, of: frames.count) }
    for k in 0..<tapsPerLoop {
        for edge in [(Float(k) + contactFraction) * tapSeconds, Float(k) * tapSeconds, (Float(k) + 0.5) * tapSeconds] {
            times += [edge - 2e-3, edge - 1e-3, edge, edge + 1e-3]
        }
    }
    times.append(loopSeconds - 1e-3)
    var pops: Int = 0
    var worst: Float = 0
    var dotPops: Int = 0
    var dotWorst: Float = 0
    for t in times {
        let a: FrameState = s.frame(at: t)
        let b: FrameState = s.frame(at: t + 1e-3)
        for (_, to, d) in matchAtoms(a.molecule, b.molecule) where inCircle(to) {
            if d > 0.2 { pops += 1 } else { worst = max(worst, d) }
        }
        for (_, to, d) in matchAtoms(b.molecule, a.molecule) where inCircle(to) && d > 0.2 { pops += 1 }
        let (w, p) = dotSteps(a, b, far: 0.05)
        dotPops += p
        dotWorst = max(dotWorst, w)
    }
    print(String(format: "        %d moments: largest move in 1 ms %.3f Å, %.4f µm; %d atoms and %d dots appear or vanish in view",
                 times.count, worst, dotWorst, pops, dotPops))
    expectEqual(pops, 0)
    expectEqual(dotPops, 0)
}

test("each touch takes one sugar a place into the pore; lifted, the sugars are still") {
    guard let s = scene else { expect(false); return }
    // One place per tap, all of it while touching.
    for k in 0..<tapsPerLoop {
        let t0: Float = Float(k) * tapSeconds
        expect(abs(sugarProgress(t0 + contactFraction * tapSeconds, mutant: s.mutant) - sugarProgress(t0, mutant: s.mutant) - 1) < 1e-4)
    }
    // The direction is into the pore: along the taste hair, from its tip towards its base.
    let h: Sensillum = s.hairs[0]
    let into: SIMD3<Float> = simd_normalize(h.base - h.tip)
    let screen = SIMD2<Float>(simd_dot(into, insetCamera.right), simd_dot(into, insetCamera.up))
    expect(simd_dot(simd_normalize(screen), s.drift) > 0.999, "the sugars do not drift towards the pore")
    guard let first = frames.first(where: { $0.lift > 0 }) else { expect(false); return }
    for f in frames where f.lift > 0 && Int(f.time / tapSeconds) == Int(first.time / tapSeconds) {
        for (a, b) in zip(f.honey.units, first.honey.units) {
            for (x, y) in zip(a.sugar.atoms, b.sugar.atoms) {
                // Only the small thermal wobble turns them; their centres hold.
                expect(simd_distance(x.position, y.position) < 2.0)
            }
            let ca: SIMD3<Float> = a.sugar.atoms.reduce(SIMD3<Float>(0, 0, 0)) { $0 + $1.position } / 24
            let cb: SIMD3<Float> = b.sugar.atoms.reduce(SIMD3<Float>(0, 0, 0)) { $0 + $1.position } / 24
            expect(simd_distance(ca, cb) < 1e-4, "a sugar moves while the hair is lifted")
        }
    }
}

test("every frame: each sugar and its waters move rigidly, keep their geometry, and never collide") {
    guard let s = scene else { expect(false); return }
    func distances(_ atoms: [Atom]) -> [Float] {
        var out: [Float] = []
        for i in 0..<atoms.count { for j in (i + 1)..<atoms.count { out.append(simd_distance(atoms[i].position, atoms[j].position)) } }
        return out
    }
    let reference: [Sugar: [Float]] = [
        .fructose: distances(sugarUnit(.fructose, chemistry: s.chemistry, turn: simd_quatf(angle: 0, axis: SIMD3(0, 1, 0)), centre: .zero).molecules.flatMap { $0.atoms }),
        .glucose: distances(sugarUnit(.glucose, chemistry: s.chemistry, turn: simd_quatf(angle: 0, axis: SIMD3(0, 1, 0)), centre: .zero).molecules.flatMap { $0.atoms }),
    ]
    var worstRigid: Float = 0
    var worstWater: Float = 0
    var closest: Float = 1e9
    for f in frames {
        for u in f.honey.units {
            let d: [Float] = distances(u.molecules.flatMap { $0.atoms })
            if let r = reference[u.kind] { for (x, y) in zip(d, r) { worstRigid = max(worstRigid, abs(x - y)) } }
            for w in u.waters {
                worstWater = max(worstWater, abs(simd_distance(w.atoms[0].position, w.atoms[1].position) - waterOH),
                                 abs(simd_distance(w.atoms[0].position, w.atoms[2].position) - waterOH))
            }
        }
        // No two atoms of different molecules (sugars, waters) closer than 2 Å.
        let parts: [Molecule] = f.honey.units.flatMap { $0.molecules }
        for i in 0..<parts.count {
            for j in (i + 1)..<parts.count {
                for a in parts[i].atoms { for b in parts[j].atoms { closest = min(closest, simd_distance(a.position, b.position)) } }
            }
        }
    }
    print(String(format: "        %d frames: worst rigid-body error %.1e Å, water O–H %.1e Å; closest atoms of two molecules %.2f Å",
                 frames.count, worstRigid, worstWater, closest))
    expect(worstRigid < 1e-3, "a sugar or its waters deform by \(worstRigid) Å")
    expect(worstWater < 1e-3)
    expect(closest > 2.0, "two molecules come within \(closest) Å")
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
    // As step 30: round the tip and the drop with the antenna at its highest.
    if let up = frames.max(by: { $0.lift < $1.lift }) {
        for (m, v) in worstOverReport(box: SIMD3(1.9, 0, 0.3), SIMD3(3.1, 1.2, 1.4), step: 0.004, inset: false, frame: up) {
            w[m] = max(w[m] ?? 0, v)
        }
    }
    print(String(format: "        worst over-report: table %.2f, ant %.2f, honey %.2f; the ray allows %.2f",
                 w[1] ?? 0, w[2] ?? 0, w[3] ?? 0, 1 / stepScale))
    expect(w.count == 3, "not every material was sampled: \(w)")
    for (m, v) in w { expect(v * stepScale <= 1.0, "material \(m) oversteps: \(v)") }
}

test("inset: the hairs, the dome, the honey and its meniscus are honest outside too") {
    let w: [Int: Float] = worstOverReport(box: SIMD3(-4, -0.5, -12), SIMD3(8, 12, 4), step: 0.02, inset: true)
    // A second pass concentrated round the tip, where the film and pore are.
    let t: [Int: Float] = worstOverReport(box: SIMD3(-1.5, -0.2, -1.5), SIMD3(1.5, 1.5, 1.5), step: 0.005, inset: true)
    // And a third in a thin shell round each hair's wall, down among the pores.
    let pores: [Int: Float] = worstOverReport(box: .zero, .zero, step: 0.004, inset: true, points: hairShellPoints(count: 80_000))
    var all: [Int: Float] = w
    for (m, v) in t { all[m] = max(all[m] ?? 0, v) }
    for (m, v) in pores { all[m] = max(all[m] ?? 0, v) }
    // As step 30: and with the antenna lifted, when the honey is moved.
    if let up = frames.max(by: { $0.lift < $1.lift }), up.lift > 0 {
        let crystalBox: SIMD3<Float> = up.crystal.translation
        for (m, v) in worstOverReport(box: SIMD3(-4, -0.5, -12) - crystalBox, SIMD3(8, 12, 4) - crystalBox, step: 0.02,
                                      inset: true, frame: up) {
            all[m] = max(all[m] ?? 0, v)
        }
        for (m, v) in worstOverReport(box: SIMD3(-4, -0.5, -8), SIMD3(8, 12, 4), step: 0.01, inset: true, frame: up) {
            all[m] = max(all[m] ?? 0, v)
        }
    }
    // Step 35: and packed round the odour dots, which now move.
    if let f = contactFrame {
        var rng = SystemRandomNumberGenerator()
        var dots: [SIMD3<Float>] = []
        for v in f.volatiles where v.radius > 0 {
            for _ in 0..<4000 {
                dots.append(v.position + SIMD3<Float>(Float.random(in: -0.4...0.4, using: &rng), Float.random(in: -0.4...0.4, using: &rng),
                                                      Float.random(in: -0.4...0.4, using: &rng)))
            }
        }
        for (m, v) in worstOverReport(box: .zero, .zero, step: 0.004, inset: true, points: dots, frame: f) { all[m] = max(all[m] ?? 0, v) }
    }
    print(String(format: "        worst over-report: honey %.2f, cuticle %.2f, taste hair %.2f, smell hair %.2f, meniscus %.2f, odour %.2f",
                 all[1] ?? 0, all[2] ?? 0, all[3] ?? 0, all[4] ?? 0, all[5] ?? 0, all[6] ?? 0))
    expect(all.count >= 5, "not every material was sampled: \(all)")
    for (m, v) in all { expect(v * stepScale <= 1.0, "material \(m) oversteps: \(v)") }
}

section("the picture")

let render: AntImage? = {
    guard let d = gpu.device, let s = scene, let f = contactFrame else { return nil }
    return try? renderAnt(width: 960, height: 540, samples: 1, scene: s, frame: f, on: d).image
}()

test("the touch is visible: antenna and honey both within 3 px of the projected contact") {
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
    expect(sawAnt && sawSugar, "near the contact: antenna \(sawAnt), honey \(sawSugar)")
}

test("the inset shows both hairs, the meniscus and odour dots; both molecule insets show C, O and H") {
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
                        outside += 1; if ProcessInfo.processInfo.environment["ANT_DEBUG"] != nil && outside % 20 == 0 { print("        out", x, y, region.rects) }
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

test("lifted, the main view shows card or air between the tip and the honey, and the inset no meniscus at the tip") {
    guard let d = gpu.device, let s = scene, let up = frames.max(by: { $0.lift < $1.lift }) else { expect(false); return }
    guard let img = try? renderAnt(width: 1920, height: 1080, samples: 1, scene: s, frame: up, on: d).image else {
        expect(false, "render failed"); return
    }
    let a: Antenna = up.ant.antennae[0]
    let low: SIMD2<Float> = projectMain(a.tipCentre - SIMD3<Float>(0, a.tipRadius, 0), width: 1920, height: 1080)
    let touch: SIMD2<Float> = projectMain(s.contact.point, width: 1920, height: 1080)
    print(String(format: "        tip raised %.1f px on screen", simd_distance(low, touch)))
    expect(simd_distance(low, touch) > 10, "the lift is invisible: \(simd_distance(low, touch)) px")
    // What shows just under the lifted tip is not the antenna touching honey:
    // halfway between the tip and the contact, the ray finds no antenna.
    let mid: SIMD2<Float> = (low + touch) / 2
    expect(img.seen(Int(mid.x), Int(mid.y)).y != 2 || simd_distance(low, touch) < 4, "the antenna still reaches the honey")
    let apex: SIMD2<Float> = projectInset(s.hairs[0].tip, height: 1080)
    var film: Int = 0
    for dy in -12...12 { for dx in -12...12 where img.seen(Int(apex.x) + dx, Int(apex.y) + dy).y == 5 { film += 1 } }
    expectEqual(film, 0)
}

finish()
