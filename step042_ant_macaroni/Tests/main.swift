// Tests for step 42. The anatomy is checked against the literature it cites,
// the contact and the pores against the GPU's own distance functions — the
// same source that draws the picture — and the finished frame for what it
// shows where.
//
// ANT_MUTANT=segments13|hover|tastePores|noOdour|antScale breaks the scene
// on purpose; `make mutants` requires the suite to fail for each.

import Foundation
import Metal
import simd

let mutant: Mutant = {
    switch ProcessInfo.processInfo.environment["ANT_MUTANT"] {
    case "segments13": return .segments13
    case "hover": return .hover
    case "tastePores": return .tastePores
    case "noOdour": return .noOdour
    case "antScale": return .antScale
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

func probe(_ pts: [SIMD3<Float>]) -> [Probe]? {
    guard let d = gpu.device, let s = scene, let l = library else { return nil }
    return try? probeScene(pts, scene: s, library: l, on: d)
}

func deg(_ r: Float) -> Float { r * 180 / Float.pi }

section("the ant, against the literature")

test("the scene loads") {
    expect(scene != nil, "could not build the scene (are the Resources/ structure files there?)")
}

test("each antenna has 12 segments — scape plus an 11-segment funiculus — as a worker's does") {
    guard let s = scene else { expect(false); return }
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
    guard let s = scene else { expect(false); return }
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
    guard let s = scene else { expect(false); return }
    let head: Shape = s.ant.body[0]
    expect(abs(head.b.z * 2 - headWidth) < 1e-4, "drawn head width \(head.b.z * 2)")
}

test("six legs, and every one joins the mesosoma — none the head, petiole or gaster") {
    guard let s = scene else { expect(false); return }
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
    guard let s = scene else { expect(false); return }
    let pet: [Shape] = s.ant.body.filter { $0.part == .petiole }
    expectEqual(pet.count, 1)
    let mesoBack: Float = s.ant.body.filter { $0.part == .mesosoma }.map { $0.extent(along: SIMD3(1, 0, 0)).lo }.min() ?? 0
    let gasterFront: Float = s.ant.body.filter { $0.part == .gaster }.map { $0.extent(along: SIMD3(1, 0, 0)).hi }.max() ?? 0
    expect(pet[0].a.x < mesoBack + 0.05 && pet[0].a.x > gasterFront - 0.05,
           "petiole at x \(pet[0].a.x), mesosoma ends \(mesoBack), gaster starts \(gasterFront)")
}

test("the worker is 3.4–5.0 mm long, mandibles to gaster tip") {
    guard let s = scene else { expect(false); return }
    let l: Float = s.ant.length
    print(String(format: "        length %.2f mm", l))
    expect(workerLengthRange.contains(l), "length \(l) mm")
}

section("the macaroni")

test("the macaroni is macaroni-sized: dry diameter inside 21 CFR 139.110's 0.11–0.27 inch, cooked 1.3× that, as drawn") {
    let lo: Float = 0.11 * 25.4, hi: Float = 0.27 * 25.4
    expect(abs(dryDiameterRange.lowerBound - lo) < 1e-4 && abs(dryDiameterRange.upperBound - hi) < 1e-4)
    expect(dryDiameterRange.contains(dryOuterDiameter), "dry diameter \(dryOuterDiameter) mm is not macaroni")
    expect(abs(2 * outerRadius - dryOuterDiameter * cookedSwelling) < 1e-5)
    expect(innerRadius > 0 && wall > 0.5, "wall \(wall)")
    guard let s = scene else { expect(false); return }
    let m: Macaroni = s.macaroni
    print(String(format: "        drawn: %.2f mm across, wall %.2f mm, bore %.2f mm, outer curve %.1f mm", 2 * m.ro, m.ro - m.ri, 2 * m.ri,
                 (m.bend + m.ro) * m.angle))
    // What is drawn is what was measured and modelled — not rescaled.
    expect(abs(m.ro - outerRadius) < 1e-5 && abs(m.ri - innerRadius) < 1e-5, "drawn radii \(m.ro), \(m.ri)")
    expect(abs(m.bend - bendRadius) < 1e-5, "drawn bend \(m.bend)")
}

test("true scale: the macaroni dwarfs a 4 mm ant — wider than the ant is tall, its curve five ants long") {
    guard let s = scene else { expect(false); return }
    let m: Macaroni = s.macaroni
    let antTop: Float = s.ant.body.map { $0.extent(along: SIMD3(0, 1, 0)).hi }.max() ?? 0
    print(String(format: "        macaroni %.1f mm high against the ant's %.2f mm; curve %.1f mm against its %.2f mm length",
                 2 * m.ro, antTop, (m.bend + m.ro) * m.angle, s.ant.length))
    expect(2 * m.ro > 4 * antTop, "the macaroni is only \(2 * m.ro / antTop)× the ant's height")
    expect((m.bend + m.ro) * m.angle > 4.5 * s.ant.length, "the curve is only \((m.bend + m.ro) * m.angle / s.ant.length) ant lengths")
}

test("it rests on the card, sauce thicker below than on top, and its end is a clean cut through the wall") {
    guard let s = scene else { expect(false); return }
    let m: Macaroni = s.macaroni
    expect(abs(m.tubeCentreHeight - m.ro) < 1e-6)
    // The lowest pasta point of the tube is on the card: probe straight
    // below the centre line, mid-elbow.
    let mid: SIMD2<Float> = m.centre + m.facing * m.bend
    expect(macaroniSDF(SIMD3<Float>(mid.x, 0.001, mid.y), m) < 0, "the tube is not down on the card")
    expect(sauceThickness(m, dy: -m.ro) > sauceThickness(m, dy: m.ro), "sauce does not run down")
    expect(abs(sauceThickness(m, dy: m.ro) - sauceTop) < 1e-5 && abs(sauceThickness(m, dy: -m.ro) - sauceBottom) < 1e-5)
    // The bore is open: the tube's centre line, mid-elbow, is outside.
    expect(macaroniSDF(SIMD3<Float>(mid.x, m.ro, mid.y), m) > 0.5, "the macaroni is not hollow")
}

test("its colours: orange sauce, pale cream pasta; and it smells") {
    let o: SIMD3<Float> = sauceAlbedo
    expect(o.x > o.y && o.y > o.z && o.z < 0.3 * o.x, "sauce colour \(o) is not orange")
    let pa: SIMD3<Float> = pastaAlbedo
    expect(pa.x > pa.z && pa.y > pa.z && (pa.x + pa.y + pa.z) > (o.x + o.y + o.z), "pasta colour \(pa)")
    expect(objectHasOdour)
}

section("the touch")

test("the right antenna's tip touches the sauce: distance 0 within 2 µm, and no penetration") {
    guard let s = scene else { expect(false); return }
    let a: Antenna = s.ant.antennae[0]
    let (point, normal) = contactPoint()
    // On the CPU: the contact point lies on the sauce, above the card, and on
    // the outer curve (facing the ant, −x).
    expect(abs(macaroniSDF(point, s.macaroni)) < 1e-4, "contact point is off the surface: \(macaroniSDF(point, s.macaroni))")
    expect(point.y > 0.05, "contact too low: \(point.y)")
    expect(normal.x < -0.5, "the contact is not on the outer curve facing the ant")
    // On the GPU: the lowest point of the tip is on the sauce.
    let lowest: SIMD3<Float> = a.tipCentre - normal * a.tipRadius
    guard let r = probe([lowest]) else { expect(false, "probe failed"); return }
    print(String(format: "        tip to sauce %.4f mm, tip to its own surface %.4f mm", r[0].food, r[0].ant))
    expect(abs(r[0].food) < 0.002, "the tip's nearest point is \(r[0].food) mm from the sauce")
    expect(abs(r[0].ant) < 0.002, "that point is not on the antenna: \(r[0].ant)")
    // No part of the antenna inside the sauce: sample a box round the tip.
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
    expect(worst > -0.002, "antenna reaches \(-worst) mm into the sauce")
}

test("no part of the ant is inside the macaroni") {
    guard let s = scene else { expect(false); return }
    var worst: Float = 9
    for shape in s.ant.shapes where shape.kind == .roundCone {
        for k in 0...20 {
            let t: Float = Float(k) / 20
            let c: SIMD3<Float> = shape.a + (shape.b - shape.a) * t
            worst = min(worst, macaroniSDF(c, s.macaroni) - (shape.ra + (shape.rb - shape.ra) * t))
        }
    }
    print(String(format: "        nearest ant surface to the macaroni: %.4f mm", worst))
    // The touching tip reads a hair below zero here: the distance is a
    // bound, scaled down 3% for the sauce's changing thickness, so at one
    // tip radius it under-reads by ~0.0016 mm. Anything deeper is real.
    expect(worst > -0.003, "an ant part reaches into the macaroni: \(worst)")
}

test("in the inset the taste hair's apex just meets the sauce surface — the meniscus climbs, the hair does not dip") {
    guard let s = scene else { expect(false); return }
    let h: Sensillum = s.hairs[0]
    let R: Float = sauceSurfaceRadiusMicrometres
    func sauce(_ p: SIMD3<Float>) -> Float { simd_length(p - SIMD3<Float>(0, -R, 0)) - R }
    var lowest: Float = 1e9
    var rng = SystemRandomNumberGenerator()
    for _ in 0..<20_000 {
        let d: SIMD3<Float> = simd_normalize(SIMD3<Float>(Float.random(in: -1...1, using: &rng), Float.random(in: -1...1, using: &rng),
                                                          Float.random(in: -1...1, using: &rng)))
        let t: Float = Float.random(in: 0...1, using: &rng)
        let p: SIMD3<Float> = h.base + (h.tip - h.base) * t + d * (h.baseRadius + (h.tipRadius - h.baseRadius) * t)
        if roundConeDistance(p, h.base, h.tip, h.baseRadius, h.tipRadius) <= 1e-4 { lowest = min(lowest, sauce(p)) }
    }
    // And the apex cap's lowest point, exactly.
    lowest = min(lowest, sauce(h.tip - SIMD3<Float>(0, h.tipRadius, 0)))
    print(String(format: "        hair's deepest point relative to the sauce surface: %.4f µm", lowest))
    expect(lowest > -0.005, "the hair dips \(-lowest) µm into the sauce")
    expect(lowest < 0.02, "the hair hovers \(lowest) µm above the sauce")
    expect(meniscusHeight > 0, "a fatty sauce should climb the waxy hair")
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

test("cheese sauce gives the smell hair something: odour molecules, each at or heading for a wall pore") {
    guard let s = scene else { expect(false); return }
    let smell: Sensillum = s.hairs[1]
    expect(s.volatiles.count >= 3, "only \(s.volatiles.count) odour molecules")
    var arriving: Int = 0
    for v in s.volatiles {
        expect(v.pore.row >= 0 && v.pore.row < smell.poreRows && v.pore.column >= 0 && v.pore.column < smell.poreColumns,
               "pore \(v.pore) is not on the lattice")
        let (mouth, outward) = smell.wallPore(row: v.pore.row, column: v.pore.column)
        let gap: Float = simd_distance(v.position, mouth)
        // Outside the hair, on the outer side of its pore.
        expect(simd_dot(v.position - mouth, outward) > 0, "an odour molecule inside the hair")
        if gap < 2 * volatileDotRadius + smell.wallPoreRadius { arriving += 1 }
        // None sits inside the taste hair or the sauce.
        let taste: Sensillum = s.hairs[0]
        expect(roundConeDistance(v.position, taste.base, taste.tip, taste.baseRadius, taste.tipRadius) > volatileDotRadius)
        expect(v.position.y > volatileDotRadius, "an odour molecule in the sauce")
    }
    expect(arriving >= 2, "only \(arriving) arriving at a pore")
    // The pore the dot is at is really carved: the drawn distance at the pore
    // mouth, just inside the wall, says "outside the hair".
    guard let v0 = s.volatiles.first else { return }
    let (mouth, outward) = smell.wallPore(row: v0.pore.row, column: v0.pore.column)
    guard let r = probe([mouth - outward * 0.02]) else { expect(false, "probe failed"); return }
    expect(r[0].smellHair > 0, "no pore at the mouth the molecule is heading for: \(r[0].smellHair)")
}

section("the molecules")

func dihedral(_ p0: SIMD3<Float>, _ p1: SIMD3<Float>, _ p2: SIMD3<Float>, _ p3: SIMD3<Float>) -> Float {
    let b1: SIMD3<Float> = p1 - p0, b2: SIMD3<Float> = p2 - p1, b3: SIMD3<Float> = p3 - p2
    let n1: SIMD3<Float> = simd_cross(b1, b2), n2: SIMD3<Float> = simd_cross(b2, b3)
    return deg(atan2(simd_dot(simd_cross(n1, n2), simd_normalize(b2)), simd_dot(n1, n2)))
}

test("norbixin is C24H28O4: 56 atoms, no ring, nine C=C and two C=O, one C=C cis — the 9'-cis colour molecule") {
    guard let s = scene else { expect(false); return }
    let m: Molecule = s.chemistry.norbixin
    let f: [String: Int] = m.formula
    expect(f["C"] == 24 && f["H"] == 28 && f["O"] == 4 && f.count == 3, "norbixin is \(f)")
    expectEqual(m.atoms.count, 56)
    expectEqual(m.rings, 0)
    let doubles: [Int] = m.bonds.indices.filter { m.orders[$0] == 2 }
    let cc: [Int] = doubles.filter { m.atoms[m.bonds[$0].0].element == "C" && m.atoms[m.bonds[$0].1].element == "C" }
    expectEqual(cc.count, 9)
    expectEqual(doubles.count - cc.count, 2)
    // Each C=C: cis or trans by the dihedral between the chain carbons on
    // either side (the ones not bearing the methyl — the longest path).
    var neighbours: [[Int]] = Array(repeating: [], count: m.atoms.count)
    for (a, b) in m.bonds { neighbours[a].append(b); neighbours[b].append(a) }
    func chainNeighbour(_ i: Int, not j: Int) -> Int? {
        // The carbon neighbour that itself has carbon neighbours beyond (not
        // a methyl), else nil.
        neighbours[i].first { k in k != j && m.atoms[k].element == "C" && neighbours[k].filter { m.atoms[$0].element == "C" }.count >= 2 }
            ?? neighbours[i].first { k in k != j && m.atoms[k].element == "C" && neighbours[k].contains { m.atoms[$0].element == "O" } }
    }
    var cis: Int = 0
    for k in cc {
        let (a, b) = m.bonds[k]
        if let c1 = chainNeighbour(a, not: b), let c2 = chainNeighbour(b, not: a) {
            let d: Float = dihedral(m.atoms[c1].position, m.atoms[a].position, m.atoms[b].position, m.atoms[c2].position)
            if abs(d) < 30 { cis += 1 }
        }
    }
    print("        \(cc.count) C=C, \(cis) of them cis")
    expectEqual(cis, 1)
    // Conjugated: the C=C bonds alternate with single bonds along one chain
    // — every C=C carbon is bonded to a carbon of another double bond (or to
    // the carboxyl carbon).
    let inDouble: Set<Int> = Set(doubles.flatMap { [m.bonds[$0].0, m.bonds[$0].1] })
    for k in cc {
        for a in [m.bonds[k].0, m.bonds[k].1] {
            let partner: Int = m.bonds[k].0 == a ? m.bonds[k].1 : m.bonds[k].0
            expect(neighbours[a].contains { $0 != partner && inDouble.contains($0) }, "C=C carbon \(a) breaks the conjugation")
        }
    }
}

test("butanoic acid is C4H8O2 with one C=O; every atom has its valence; the ions carry none") {
    guard let s = scene else { expect(false); return }
    let o: Molecule = s.chemistry.odorant
    let f: [String: Int] = o.formula
    expect(f["C"] == 4 && f["H"] == 8 && f["O"] == 2, "butanoic acid is \(f)")
    expectEqual(o.rings, 0)
    expectEqual(o.bonds.indices.filter { o.orders[$0] == 2 }.count, 1)
    for a in o.atoms { expectEqual(a.view, 1) }
    let all: Molecule = s.molecule
    let v: [Int] = all.valences
    for (i, a) in all.atoms.enumerated() {
        let want: Int
        switch a.element {
        case "C": want = 4
        case "O": want = 2
        case "H": want = 1
        default: want = 0
        }
        expect(v[i] == want, "\(a.element) \(i) has valence \(v[i])")
    }
}

test("the salt: one Na⁺ and one Cl⁻, each with four whole waters at Ohtaki & Radnai's distances, clear of norbixin") {
    guard let s = scene else { expect(false); return }
    expectEqual(s.chemistry.ions.count, 2)
    expectEqual(s.chemistry.ions[0].atoms[0].element, "Na")
    expectEqual(s.chemistry.ions[1].atoms[0].element, "Cl")
    expectEqual(s.chemistry.waters.count, 8)
    for (k, w) in s.chemistry.waters.enumerated() {
        expectEqual(w.formula["O"] ?? 0, 1)
        expectEqual(w.formula["H"] ?? 0, 2)
        let ion: Atom = s.chemistry.ions[k < 4 ? 0 : 1].atoms[0]
        let dO: Float = simd_distance(w.atoms[0].position, ion.position)
        let want: Float = k < 4 ? 2.43 : 3.20
        expect(abs(dO - want) < 1e-3, "ion–O \(dO) Å")
        // Na⁺: oxygen towards the ion (both H further out). Cl⁻: one H
        // pointing at it (nearer than the O).
        let dH: Float = min(simd_distance(w.atoms[1].position, ion.position), simd_distance(w.atoms[2].position, ion.position))
        expect(k < 4 ? dH > dO : dH < dO, "water \(k) faces its ion the wrong way")
        for a in w.atoms { for b in s.chemistry.norbixin.atoms {
            expect(simd_distance(a.position, b.position) > 2.0, "a water collides with norbixin")
        } }
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
                     points given: [SIMD3<Float>] = []) -> [Int: Float] {
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
    guard let pa = probe(a), let pb = probe(b) else { return [:] }
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
    print(String(format: "        worst over-report: table %.2f, ant %.2f, macaroni %.2f; the ray allows %.2f",
                 w[1] ?? 0, w[2] ?? 0, w[3] ?? 0, 1 / stepScale))
    expect(w.count == 3, "not every material was sampled: \(w)")
    for (m, v) in w { expect(v * stepScale <= 1.0, "material \(m) oversteps: \(v)") }
}

test("inset: the hairs, the dome, the sauce and its meniscus are honest outside too") {
    let w: [Int: Float] = worstOverReport(box: SIMD3(-4, -0.5, -12), SIMD3(8, 12, 4), step: 0.02, inset: true)
    // A second pass concentrated round the tip, where the film and pore are.
    let t: [Int: Float] = worstOverReport(box: SIMD3(-1.5, -0.2, -1.5), SIMD3(1.5, 1.5, 1.5), step: 0.005, inset: true)
    // And a third in a thin shell round each hair's wall, down among the pores.
    let pores: [Int: Float] = worstOverReport(box: .zero, .zero, step: 0.004, inset: true, points: hairShellPoints(count: 80_000))
    var all: [Int: Float] = w
    for (m, v) in t { all[m] = max(all[m] ?? 0, v) }
    for (m, v) in pores { all[m] = max(all[m] ?? 0, v) }
    print(String(format: "        worst over-report: sauce %.2f, cuticle %.2f, taste hair %.2f, smell hair %.2f, meniscus %.2f, odour %.2f",
                 all[1] ?? 0, all[2] ?? 0, all[3] ?? 0, all[4] ?? 0, all[5] ?? 0, all[6] ?? 0))
    expect(all.count >= 5, "not every material was sampled: \(all)")
    for (m, v) in all { expect(v * stepScale <= 1.0, "material \(m) oversteps: \(v)") }
}

section("the picture")

let render: AntImage? = {
    guard let d = gpu.device, let s = scene else { return nil }
    return try? renderAnt(width: 960, height: 540, samples: 1, scene: s, on: d).image
}()

test("the touch is visible: antenna and macaroni both within 3 px of the projected contact") {
    guard let img = render, let s = scene else { expect(false, "render failed"); return }
    let a: Antenna = s.ant.antennae[0]
    // The contact is on the tube's flank, not under the tip: project it.
    let p: SIMD2<Float> = projectMain(contactPoint().point, width: img.width, height: img.height)
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
    expect(sawAnt && sawSugar, "near the contact: antenna \(sawAnt), macaroni \(sawSugar)")
}

test("the inset shows both hairs, the meniscus and odour dots; the taste inset shows C, O, H, Na⁺ and Cl⁻, the odour inset C, O and H") {
    guard let img = render else { expect(false, "render failed"); return }
    var seen: Set<Int> = []
    var taste: Set<Int> = []
    var smell: Set<Int> = []
    for y in 0..<img.height {
        for x in 0..<img.width {
            let v: SIMD4<Float> = img.seen(x, y)
            if v.x == 2 { seen.insert(Int(v.y)) }
            if v.x == 3 && v.y >= 1 && v.y <= 5 { taste.insert(Int(v.y)) }
            if v.x == 4 && v.y >= 1 && v.y <= 3 { smell.insert(Int(v.y)) }
        }
    }
    expect(seen.isSuperset(of: [3, 4, 5, 6]), "inset materials seen: \(seen.sorted())")
    expectEqual(taste, [1, 2, 3, 4, 5])
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

finish()
