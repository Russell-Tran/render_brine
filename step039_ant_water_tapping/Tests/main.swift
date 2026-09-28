// Tests for step 39: step 38's tests, copied (run on the contact frame, where
// the scene is step 38's), then the motion as step 30 tests it — the touch at
// every touch-down, no penetration at any frame, a clear lift, a forward loop —
// and the water's side of it: the water-repellent tip meets the surface and is
// never under it, at any frame or any moment of lift-off; no odour at the smell
// hair at any frame; the humidity sensillum and the waters unchanged throughout.
//
// Step 38's tests. The anatomy is checked against the literature it cites,
// the contact and the pores against the GPU's own distance functions — the
// same source that draws the picture — and the finished frame for what it
// shows where.
//
// ANT_MUTANT=segments13|hover|tastePores|odour|waterAngle|press|rewind breaks the scene
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
    case "waterAngle": return .waterAngle
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

/// Every frame of the GIF's loop, and the first — the touch-down, step 38's pose.
let frames: [FrameState] = scene.map { s in (0..<defaultFrames).map { s.frame(at: frameTime($0, of: defaultFrames)) } } ?? []
let contactFrame: FrameState? = frames.first

func probe(_ pts: [SIMD3<Float>], frame given: FrameState? = nil) -> [Probe]? {
    guard let d = gpu.device, let s = scene, let l = library, let f = given ?? contactFrame else { return nil }
    return try? probeScene(pts, scene: s, frame: f, library: l, on: d)
}

func deg(_ r: Float) -> Float { r * 180 / Float.pi }

/// The moment, just after the touch ends (or, `landing`, just before the
/// next one begins), when the dimple is `dimple` µm deep: found by bisection,
/// since the dimple only shrinks as the tip rises and grows as it lands.
func liftOffMoment(dimple want: Float, landing: Bool = false) -> FrameState? {
    guard let s = scene else { return nil }
    let off: Float = contactFraction * tapSeconds
    var lo: Float = landing ? tapSeconds - 0.2 : off
    var hi: Float = landing ? tapSeconds - 1e-6 : off + 0.2
    for _ in 0..<60 {
        let mid: Float = (lo + hi) / 2
        let d: Float = s.frame(at: mid).dimple
        // Lifting off, deeper means earlier; landing, deeper means later.
        if (d > want) != landing { lo = mid } else { hi = mid }
    }
    return s.frame(at: (lo + hi) / 2)
}

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

section("the water drop")

test("the drop is a spherical cap meeting the card at its contact angle, 0° < θ < 90°") {
    expect(dropContactAngle > 0 && dropContactAngle < Float.pi / 2, "contact angle \(deg(dropContactAngle))°")
    let rimR: Float = (dropSphereRadius * dropSphereRadius - dropCentre.y * dropCentre.y).squareRoot()
    expect(abs(rimR - dropBaseRadius) < 1e-5, "rim radius \(rimR) mm")
    let rim = SIMD3<Float>(dropAt.x + dropBaseRadius, 0, dropAt.y)
    let n: SIMD3<Float> = simd_normalize(rim - dropCentre)
    let theta: Float = acos(simd_dot(n, SIMD3<Float>(0, 1, 0)))
    expect(abs(theta - dropContactAngle) < 1e-4, "angle at the rim \(deg(theta))°")
    print(String(format: "        drop %.2f mm wide, %.2f mm high, %.0f° at the rim", 2 * dropBaseRadius, dropHeight, deg(theta)))
}

test("gravity cannot flatten it: water's own Bond number, ρga²/γ, is under 0.1") {
    expect(abs(waterDensity - 997.05) < 0.01 && abs(waterSurfaceTension - 0.07197) < 1e-6)
    let bond: Float = waterDensity * 9.81 * pow(dropBaseRadius / 1000, 2) / waterSurfaceTension
    let capillary: Float = (waterSurfaceTension / (waterDensity * 9.81)).squareRoot() * 1000
    print(String(format: "        Bond number %.3f; capillary length %.2f mm", bond, capillary))
    expect(bond < 0.1, "Bond number \(bond): a cap is the wrong shape")
    expect(abs(capillary - 2.7) < 0.05, "capillary length \(capillary) mm")
}

test("the drop is ant-sized: 0.5–1.5 mm across, and lower than the ant's head") {
    expect(2 * dropBaseRadius >= 0.5 && 2 * dropBaseRadius <= 1.5)
    expect(dropHeight < headCentre.y)
}

test("water's optics: n 1.333, and clear — a millimetre of it passes over 99.9% of every colour") {
    expect(abs(waterIndex - 1.333) < 1e-6)
    let through: SIMD3<Float> = SIMD3<Float>(exp(-waterAbsorptionRGB.x), exp(-waterAbsorptionRGB.y), exp(-waterAbsorptionRGB.z))
    print(String(format: "        through 1 mm: R %.5f G %.5f B %.5f", through.x, through.y, through.z))
    expect(min(through.x, min(through.y, through.z)) > 0.999, "water is tinted: \(through)")
    // Pope & Fry's order: water absorbs red most, blue least.
    expect(waterAbsorptionRGB.x > waterAbsorptionRGB.y && waterAbsorptionRGB.y > waterAbsorptionRGB.z)
}

section("the touch")

test("the right antenna's tip touches the drop: distance 0 within 2 µm, and no penetration") {
    guard let f = contactFrame else { expect(false); return }
    let a: Antenna = f.ant.antennae[0]
    let (point, normal) = contactPoint()
    // On the CPU: the contact point lies on the drop's surface, above the card.
    expect(abs(dropSDF(point)) < 1e-5, "contact point is off the surface: \(dropSDF(point))")
    expect(point.y > 0.05, "contact too low on the drop: \(point.y)")
    // On the GPU: the lowest point of the tip is on the water.
    let lowest: SIMD3<Float> = a.tipCentre - normal * a.tipRadius
    guard let r = probe([lowest]) else { expect(false, "probe failed"); return }
    print(String(format: "        tip to water %.4f mm, tip to its own surface %.4f mm", r[0].food, r[0].ant))
    expect(abs(r[0].food) < 0.002, "the tip's nearest point is \(r[0].food) mm from the water")
    expect(abs(r[0].ant) < 0.002, "that point is not on the antenna: \(r[0].ant)")
    // No part of the antenna inside the water: sample a box round the tip.
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
    expect(worst > -0.002, "antenna reaches \(-worst) mm into the water")
}

/// The water surface in the inset at one moment, as the kernel draws it
/// (before its Lipschitz scaling): the drop's sphere, moved by the frame, plus
/// that moment's dimple under the hair (Motion.swift's waterSurface).
func waterSurface(_ p: SIMD3<Float>, hair h: Sensillum, frame f: FrameState) -> Float {
    let dHair: Float = roundConeDistance(p, h.base, h.tip, h.baseRadius, h.tipRadius)
    return waterSurface(f.water.toWater(p), hairDistance: dHair, depth: f.dimple)
}

/// Points on the taste hair's outer surface, `n` of them, and a few on the apex cap.
func hairSurfacePoints(_ h: Sensillum, count n: Int) -> [SIMD3<Float>] {
    var rng = SystemRandomNumberGenerator()
    var out: [SIMD3<Float>] = []
    for _ in 0..<n {
        let d: SIMD3<Float> = simd_normalize(SIMD3<Float>(Float.random(in: -1...1, using: &rng), Float.random(in: -1...1, using: &rng),
                                                          Float.random(in: -1...1, using: &rng)))
        let t: Float = Float.random(in: 0...1, using: &rng)
        let p0: SIMD3<Float> = h.base + (h.tip - h.base) * t + d * (h.baseRadius + (h.tipRadius - h.baseRadius) * t)
        let q: SIMD3<Float> = p0 - d * roundConeDistance(p0, h.base, h.tip, h.baseRadius, h.tipRadius)
        if abs(roundConeDistance(q, h.base, h.tip, h.baseRadius, h.tipRadius)) < 1e-3 { out.append(q) }
    }
    return out
}

test("in the inset the water-repellent tip touches the water at its apex only — a dimple, not immersion") {
    guard let s = scene, let f = contactFrame else { expect(false); return }
    let h: Sensillum = s.hairs[0]
    let apex: SIMD3<Float> = h.tip - SIMD3<Float>(0, h.tipRadius, 0)
    let atApex: Float = waterSurface(apex, hair: h, frame: f)
    print(String(format: "        apex to the water surface %.4f µm; apex %.2f µm below the undisturbed level", atApex, -apex.y))
    expect(abs(atApex) < 0.002, "the apex is \(atApex) µm from the water")
    // Every other point of the hair's surface is above the water: the water
    // neither wraps the tip nor climbs the hair.
    var lowest: Float = 1e9
    var wet: Int = 0
    var rng = SystemRandomNumberGenerator()
    for _ in 0..<40_000 {
        let d: SIMD3<Float> = simd_normalize(SIMD3<Float>(Float.random(in: -1...1, using: &rng), Float.random(in: -1...1, using: &rng),
                                                          Float.random(in: -1...1, using: &rng)))
        let t: Float = Float.random(in: 0...1, using: &rng)
        let p0: SIMD3<Float> = h.base + (h.tip - h.base) * t + d * (h.baseRadius + (h.tipRadius - h.baseRadius) * t)
        // Pull onto the hair's surface.
        let q: SIMD3<Float> = p0 - d * roundConeDistance(p0, h.base, h.tip, h.baseRadius, h.tipRadius)
        guard abs(roundConeDistance(q, h.base, h.tip, h.baseRadius, h.tipRadius)) < 1e-3 else { continue }
        let w: Float = waterSurface(q, hair: h, frame: f)
        lowest = min(lowest, w)
        if w < -0.005 { wet += 1 }
    }
    expect(lowest > -0.005, "part of the hair is \(-lowest) µm under water")
    expectEqual(wet, 0)
    // The dimple is shallow — the apex is less than 0.3 µm below the level.
    expect(-apex.y > 0 && -apex.y < 0.3, "apex depth \(-apex.y) µm")
    // And the GPU draws the water where this says it is.
    guard let r = probe([SIMD3<Float>(6, 0.3, 2), SIMD3<Float>(6, -0.3, 2)]) else { expect(false, "probe failed"); return }
    expect(r[0].insetDistance > 0, "0.3 µm above the surface is not air")
    expect(r[1].insetDistance < 0 && r[1].insetMaterial == 1, "0.3 µm under the surface is not water")
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

section("smell and humidity: water has no odour")

test("no odour molecules reach the smell hair: water has no smell") {
    guard let s = scene else { expect(false); return }
    expect(!objectHasOdour)
    expectEqual(s.volatiles.count, 0)
}

test("water is volatile, and a humidity sensillum — not the smell hair — is drawn: a knob in a pit, no pores") {
    guard let s = scene else { expect(false); return }
    expect(objectIsVolatile)
    let hy: Hygro = s.hygro
    let dome = s.dome
    // The pit is cut into the dome's surface, and the knob sits in it: its
    // top below the rim, its base on the pit's floor.
    let rimDepth: Float = simd_distance(hy.pitCentre, dome.centre) - dome.radius
    expect(rimDepth < 0 && -rimDepth < hy.pitRadius, "the pit does not open at the surface")
    let knobTop: SIMD3<Float> = hy.knobCentre + hy.outward * hy.knobRadius
    let topBelow: Float = dome.radius - simd_distance(knobTop, dome.centre)
    print(String(format: "        knob top %.2f µm below the rim; pit %.1f µm across", topBelow,
                 2 * (hy.pitRadius * hy.pitRadius - rimDepth * rimDepth).squareRoot()))
    expect(topBelow > 0 && topBelow < 0.6, "knob top \(topBelow) µm below the surface")
    let gap: Float = hy.pitRadius - (simd_distance(hy.knobCentre, hy.pitCentre) + hy.knobRadius)
    expect(abs(gap) < 1e-4, "knob is not resting on the pit floor: gap \(gap)")
    // It is its own sensillum: away from both hairs.
    for hair in s.hairs {
        expect(roundConeDistance(hy.knobCentre, hair.base, hair.tip, hair.baseRadius, hair.tipRadius) > 1.0)
    }
    // The GPU draws the knob, with no pores in it: just inside its surface,
    // all round its upper part, is solid.
    let side: SIMD3<Float> = simd_normalize(simd_cross(hy.outward, SIMD3<Float>(0, 0, 1)))
    let side2: SIMD3<Float> = simd_cross(hy.outward, side)
    var pts: [SIMD3<Float>] = []
    for k in 0..<200 {
        let a: Float = Float(k) * 2.399
        let z: Float = 1 - Float(k) / 200 * 0.6
        let dir: SIMD3<Float> = hy.outward * z + (side * cos(a) + side2 * sin(a)) * (1 - z * z).squareRoot()
        pts.append(hy.knobCentre + dir * (hy.knobRadius - 0.02))
    }
    pts.append(knobTop + hy.outward * 0.01)
    guard let r = probe(pts) else { expect(false, "probe failed"); return }
    var inside: Int = 0
    for q in r.dropLast() where q.insetDistance < 0 { inside += 1 }
    expectEqual(inside, 200)
    expect(r[r.count - 1].insetMaterial == 7, "the knob is not what the GPU finds there: \(r[r.count - 1].insetMaterial)")
}

section("the molecules")

test("five whole waters, each O–H 0.9572 Å and H–O–H 104.52°") {
    guard let s = scene else { expect(false); return }
    expectEqual(s.chemistry.waters.count, 5)
    for w in s.chemistry.waters {
        expectEqual(w.formula["O"] ?? 0, 1)
        expectEqual(w.formula["H"] ?? 0, 2)
        let o: SIMD3<Float> = w.atoms[0].position
        let a: SIMD3<Float> = w.atoms[1].position - o
        let b: SIMD3<Float> = w.atoms[2].position - o
        expect(abs(simd_length(a) - 0.9572) < 1e-3 && abs(simd_length(b) - 0.9572) < 1e-3, "O–H \(simd_length(a)), \(simd_length(b))")
        let angle: Float = deg(acos(simd_dot(simd_normalize(a), simd_normalize(b))))
        expect(abs(angle - 104.52) < 0.05, "H–O–H \(angle)°")
    }
    let v: [Int] = s.molecule.valences
    for (i, a) in s.molecule.atoms.enumerated() {
        expectEqual(v[i], a.element == "O" ? 2 : 1)
    }
}

test("four hydrogen bonds, each H···O 1.97 Å and straight, from an H to another water's O") {
    guard let s = scene else { expect(false); return }
    let m: Molecule = s.molecule
    expectEqual(m.hbonds.count, 4)
    for (h, o) in m.hbonds {
        expectEqual(m.atoms[h].element, "H")
        expectEqual(m.atoms[o].element, "O")
        expect(h / 3 != o / 3, "a hydrogen bond inside one molecule")
        let d: Float = simd_distance(m.atoms[h].position, m.atoms[o].position)
        expect(abs(d - hydrogenBondLength) < 0.005, "H···O \(d) Å")
        let donor: SIMD3<Float> = m.atoms[(h / 3) * 3].position
        let c: Float = simd_dot(simd_normalize(donor - m.atoms[h].position), simd_normalize(m.atoms[o].position - m.atoms[h].position))
        let ang: Float = deg(acos(min(max(c, -1), 1)))
        expect(ang > 175, "O–H···O \(ang)°")
    }
    expect(abs(hydrogenBondLength - 1.97) < 1e-6)
    // Nothing else crowds: atoms of different waters, not hydrogen-bonded,
    // stay at least 2 Å apart.
    for i in 0..<m.atoms.count {
        for j in (i + 1)..<m.atoms.count where i / 3 != j / 3 {
            let bonded: Bool = m.hbonds.contains { ($0.0 == i && $0.1 == j) || ($0.0 == j && $0.1 == i) }
            if !bonded {
                expect(simd_distance(m.atoms[i].position, m.atoms[j].position) > 2.0,
                       "atoms \(i) and \(j) of different waters \(simd_distance(m.atoms[i].position, m.atoms[j].position)) Å apart")
            }
        }
    }
}

section("the tap (step 30's, on water)")

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

/// The drop's distance from the right antenna at one frame, on the GPU: the
/// tip's lowest point, and the deepest any antenna point reaches into the
/// water among points packed round the tip.
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

test("at every touch-down the tip meets the drop: 0 within 2 µm, on the GPU") {
    let downs: [FrameState] = frames.filter { $0.lift == 0 }
    expect(downs.count >= tapsPerLoop, "only \(downs.count) touching frames")
    var worst: Float = 0
    for f in downs {
        guard let r = tipSurvey(f, points: 0) else { expect(false, "probe failed"); return }
        worst = max(worst, abs(r.lowest))
    }
    print(String(format: "        %d touching frames; worst tip-to-water %.4f mm", downs.count, worst))
    expect(worst < 0.002, "a touching tip is \(worst) mm from the water")
}

test("at no frame does any part of the antenna go into the drop (main view, 12,000 points a frame)") {
    var worst: Float = 1
    for f in frames {
        guard let r = tipSurvey(f, points: 12_000) else { expect(false, "probe failed"); return }
        worst = min(worst, r.deepest)
    }
    print(String(format: "        %d frames × 12,000 points: nearest antenna point to the inside of the drop %.4f mm", frames.count, worst))
    expect(worst > -0.002, "the antenna reaches \(-worst) mm into the water")
}

test("lifted, the tip clears the drop — by the lift, and never less than 0.03 mm near the top") {
    let ups: [FrameState] = frames.filter { $0.lift > 0.5 * liftHeight }
    expect(!ups.isEmpty, "no lifted frames")
    var least: Float = 1
    for f in ups {
        guard let r = tipSurvey(f, points: 4000) else { expect(false, "probe failed"); return }
        least = min(least, r.lowest)
        expect(abs(r.lowest - f.lift) < 0.002, "lift \(f.lift) mm but the tip is \(r.lowest) mm up")
        expect(r.deepest > 0, "a lifted antenna point is in the water")
    }
    print(String(format: "        %d lifted frames, least clearance %.3f mm (lift %.3f mm at the top)", ups.count, least, liftHeight))
    expect(least > 0.03)
}

section("the touch in the inset: a water-repellent tip, never wetted")

/// Points in the inset (µm) to test the antenna against the water: packed
/// round the apex, on the taste hair's surface, and over the whole field.
func insetTestPoints() -> [SIMD3<Float>] {
    guard let s = scene else { return [] }
    var rng = SystemRandomNumberGenerator()
    var pts: [SIMD3<Float>] = hairSurfacePoints(s.hairs[0], count: 3000)
    for _ in 0..<5000 {
        pts.append(SIMD3<Float>(Float.random(in: -1.2...1.2, using: &rng), Float.random(in: -0.3...1.5, using: &rng),
                                Float.random(in: -1.2...1.2, using: &rng)))
    }
    for _ in 0..<4000 {
        pts.append(SIMD3<Float>(Float.random(in: -8...9, using: &rng), Float.random(in: -1...14, using: &rng),
                                Float.random(in: -14...5, using: &rng)))
    }
    return pts
}

test("at every touch-down the taste hair's apex meets the water's surface: 0 within 2 nm (GPU)") {
    guard let s = scene else { expect(false); return }
    let h: Sensillum = s.hairs[0]
    let apexLow = SIMD3<Float>(h.tip.x, h.tip.y - h.tipRadius, h.tip.z)
    let downs: [FrameState] = frames.filter { $0.lift == 0 }
    var worst: Float = 0
    for f in downs {
        expect(abs(f.dimple - dimpleDepth) < 1e-4, "touching, the dimple is \(f.dimple) µm deep, not step 38's \(dimpleDepth)")
        guard let r = probe([apexLow], frame: f) else { expect(false, "probe failed"); return }
        worst = max(worst, abs(r[0].water))
    }
    print(String(format: "        %d touching frames; worst apex-to-water %.5f µm", downs.count, worst))
    expect(worst < 0.002, "a touching apex is \(worst) µm from the water")
}

test("at no frame is any point of the antenna — hairs, dome, socket, knob — under the water (GPU)") {
    let pts: [SIMD3<Float>] = insetTestPoints()
    var worst: Float = 1e9
    var inside: Int = 0
    for f in frames {
        guard let q = probe(pts, frame: f) else { expect(false, "probe failed"); return }
        for p in q where p.antenna < 0 { inside += 1; worst = min(worst, p.water) }
    }
    print(String(format: "        %d frames × %d points, %d inside the antenna: lowest water distance among them %.4f µm", frames.count, pts.count, inside, worst))
    expect(inside > 1000, "too few antenna points sampled")
    expect(worst > -0.002, "an antenna point is \(-worst) µm under the water")
}

test("lift-off and landing, moment by moment: the dimple follows the apex, touching it and nothing more") {
    guard let s = scene else { expect(false); return }
    let h: Sensillum = s.hairs[0]
    let surface: [SIMD3<Float>] = hairSurfacePoints(h, count: 4000)
    let apexLow = SIMD3<Float>(h.tip.x, h.tip.y - h.tipRadius, h.tip.z)
    let off: Float = contactFraction * tapSeconds
    var lowest: Float = 1e9
    var worstTouch: Float = 0
    var moments: Int = 0
    var lastOff: Float = dimpleDepth
    var gpuChecked: Int = 0
    for k in 0...200 {
        for landing in [false, true] {
            let t: Float = landing ? tapSeconds - 0.06 + 0.06 * Float(k) / 200 : off + 0.06 * Float(k) / 200
            let f: FrameState = s.frame(at: t)
            moments += 1
            expect(f.dimple >= 0 && f.dimple <= dimpleDepth + 1e-6, "dimple \(f.dimple) µm")
            for p in surface { lowest = min(lowest, waterSurface(p, hair: h, frame: f)) }
            let apex: Float = waterSurface(apexLow, hair: h, frame: f)
            lowest = min(lowest, apex)
            // While there is a dimple, the apex is in it: touching, exactly.
            if f.dimple > 1e-4 { worstTouch = max(worstTouch, abs(apex)) }
            if !landing {
                expect(f.dimple <= lastOff + 1e-6, "lifting, the dimple deepens at t \(t)")
                lastOff = f.dimple
            }
            // And the GPU draws the same, at every tenth moment.
            if k % 10 == 0 {
                guard let q = probe(surface.prefix(600) + [apexLow], frame: f) else { expect(false, "probe failed"); return }
                for r in q { lowest = min(lowest, r.water * (1 + dimpleDepth / dimpleWidth)) }
                gpuChecked += 1
            }
        }
    }
    print(String(format: "        %d moments (%d on the GPU): lowest water value on the hair %.5f µm; touching apex off the surface by %.5f µm at worst",
                 moments, gpuChecked, lowest, worstTouch))
    expect(lowest > -0.002, "part of the hair is \(-lowest) µm under water")
    expect(worstTouch < 0.002, "while dimpled, the apex is \(worstTouch) µm off the water")
    expect(lastOff == 0, "the dimple never relaxes: \(lastOff) µm")
    // The dimple lets go smoothly: half-depth is reached, and the moment is found.
    guard let half = liftOffMoment(dimple: 0.5 * dimpleDepth) else { expect(false); return }
    expect(abs(half.dimple - 0.5 * dimpleDepth) < 1e-3, "half-depth moment has \(half.dimple)")
}

test("lifted, the dimple is gone, the water is out of reach, and the drop moves by the lift (inset)") {
    guard let s = scene else { expect(false); return }
    let h: Sensillum = s.hairs[0]
    let apexLow = SIMD3<Float>(h.tip.x, h.tip.y - h.tipRadius, h.tip.z)
    let beside: SIMD3<Float> = apexLow + SIMD3<Float>(0.7, -0.05, 0)
    var worstRise: Float = 0
    for f in frames where f.lift > 0.001 {
        expectEqual(f.dimple, 0)
        let above: Float = waterSurface(apexLow, hair: h, frame: f)
        worstRise = max(worstRise, abs(above - 1000 * f.lift))
        expect(above > 0.9, "lifted \(f.lift) mm, the apex is only \(above) µm over the water")
        guard let r = probe([beside], frame: f) else { expect(false, "probe failed"); return }
        expect(r[0].insetMaterial != 1 && r[0].water > 0, "lifted, water beside the apex")
    }
    guard let c = contactFrame, let r0 = probe([beside], frame: c) else { expect(false); return }
    expect(r0[0].insetMaterial == 1 && r0[0].water < 0, "touching, there is no water just beside the apex")
    print(String(format: "        apex height over the water vs the main view's lift: worst difference %.2f µm", worstRise))
    expect(worstRise < 5, "the inset and the main view disagree by \(worstRise) µm")
}

section("the loop")

test("the loop is forward: the last frame runs on into the first, and the clock only advances") {
    guard let s = scene, let c = contactFrame else { expect(false); return }
    let end: FrameState = s.frame(at: loopSeconds)
    expect(simd_distance(end.ant.antennae[0].tipCentre, c.ant.antennae[0].tipCentre) < 1e-5)
    expect(simd_distance(end.ant.antennae[1].tipCentre, c.ant.antennae[1].tipCentre) < 1e-5)
    expect(abs(end.dimple - c.dimple) < 1e-6)
    func step(_ a: FrameState, _ b: FrameState) -> Float {
        max(simd_distance(a.ant.antennae[0].tipCentre, b.ant.antennae[0].tipCentre),
            simd_distance(a.ant.antennae[1].tipCentre, b.ant.antennae[1].tipCentre))
    }
    var biggest: Float = 0
    for i in 1..<frames.count { biggest = max(biggest, step(frames[i - 1], frames[i])) }
    let seam: Float = step(frames[frames.count - 1], frames[0])
    print(String(format: "        seam step %.5f mm, largest step inside the loop %.5f mm", seam, biggest))
    expect(seam <= biggest * 1.01 + 1e-6, "the seam jumps")
    // Forward: the loop's clock never runs back, and it ends one loop on.
    var last: Float = frames[0].progress
    for i in 1...frames.count {
        let u: Float = i == frames.count ? loopProgress(loopSeconds - 1e-4, mutant: s.mutant) : frames[i].progress
        expect(u >= last - 1e-5, "the loop runs back at frame \(i): \(last) → \(u)")
        last = u
    }
    expect(abs(last - Float(tapsPerLoop)) < 0.01, "the loop is \(last) taps, not \(tapsPerLoop)")
    // Each tap is the same: the lift and the left antenna's sway are periodic in the tap.
    let perTap: Int = frames.count / tapsPerLoop
    for i in 0..<(frames.count - perTap) {
        expect(abs(frames[i].lift - frames[i + perTap].lift) < 1e-5, "tap \(i / perTap) differs at frame \(i)")
        expect(simd_distance(frames[i].ant.antennae[1].tipCentre, frames[i + perTap].ant.antennae[1].tipCentre) < 1e-5)
    }
}

test("nothing pops: 1 ms apart the tip, the dimple and the drop move by only a little, and the waters not at all") {
    guard let s = scene, let c = contactFrame else { expect(false); return }
    var times: [Float] = (0..<frames.count).map { frameTime($0, of: frames.count) }
    for k in 0..<tapsPerLoop {
        let off: Float = (Float(k) + contactFraction) * tapSeconds
        times += [off - 1e-3, off, off + 1e-3, off + 0.02, Float(k + 1) * tapSeconds - 2e-3]
    }
    times.append(loopSeconds - 1e-3)
    var worstTip: Float = 0
    var worstDip: Float = 0
    var worstShift: Float = 0
    for t in times {
        let a: FrameState = s.frame(at: t)
        let b: FrameState = s.frame(at: t + 1e-3)
        worstTip = max(worstTip, simd_distance(a.ant.antennae[0].tipCentre, b.ant.antennae[0].tipCentre))
        worstDip = max(worstDip, abs(a.dimple - b.dimple))
        worstShift = max(worstShift, simd_distance(a.water.translation, b.water.translation))
    }
    print(String(format: "        %d moments: in 1 ms the tip moves %.2e mm, the dimple %.4f µm, the inset's drop %.3f µm at most",
                 times.count, worstTip, worstDip, worstShift))
    // The fastest the sin² lift can go: π · lift / (lifted share of a tap), per ms.
    let fastest: Float = Float.pi * liftHeight / ((1 - contactFraction) * tapSeconds) / 1000
    expect(worstTip <= fastest * 1.05 + 1e-7, "the tip jumps \(worstTip) mm in 1 ms")
    expect(worstShift <= fastest * 1000 * 1.2 + 1e-4, "the inset's drop jumps \(worstShift) µm in 1 ms")
    expect(worstDip < 0.02, "the dimple jumps \(worstDip) µm in 1 ms")
    // The five waters are the same at every frame: nothing drifts, nothing pops.
    for f in frames {
        expectEqual(f.molecule.atoms.count, c.molecule.atoms.count)
        for (a, b) in zip(f.molecule.atoms, c.molecule.atoms) { expect(simd_distance(a.position, b.position) < 1e-6) }
    }
}

section("smell and humidity, every frame")

test("no odour at the smell hair at any frame: none in the scene, and none where odour would arrive (GPU)") {
    guard let s = scene else { expect(false); return }
    expect(!objectHasOdour)
    expectEqual(s.volatiles.count, 0)
    // Where step 34's odour molecules would sit, beside the smell hair's pores,
    // the GPU finds no odour dot at any frame.
    let wouldBe: [Volatile] = buildVolatiles(hairs: s.hairs, mutant: .odour, facing: insetDirection)
    expect(!wouldBe.isEmpty)
    let pts: [SIMD3<Float>] = wouldBe.map { $0.position }
    var dots: Int = 0
    for f in frames {
        guard let q = probe(pts, frame: f) else { expect(false, "probe failed"); return }
        for r in q where r.insetMaterial == 6 { dots += 1 }
    }
    print("        \(frames.count) frames × \(pts.count) arrival points: \(dots) odour dots found")
    expectEqual(dots, 0)
}

test("the humidity sensillum is there at every frame, the same knob in the same pit, riding with the hairs") {
    guard let s = scene else { expect(false); return }
    expect(objectIsVolatile)
    let top: SIMD3<Float> = s.hygro.knobCentre + s.hygro.outward * (s.hygro.knobRadius - 0.02)
    var missing: Int = 0
    for f in frames {
        guard let q = probe([top], frame: f) else { expect(false, "probe failed"); return }
        if q[0].insetMaterial != 7 || q[0].insetDistance >= 0 { missing += 1 }
    }
    expectEqual(missing, 0)
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
        for (m, v) in worstOverReport(box: SIMD3(2.0, 0, 0.3), SIMD3(3.2, 1.2, 1.6), step: 0.004, inset: false, frame: up) {
            w[m] = max(w[m] ?? 0, v)
        }
    }
    print(String(format: "        worst over-report: table %.2f, ant %.2f, water %.2f; the ray allows %.2f",
                 w[1] ?? 0, w[2] ?? 0, w[3] ?? 0, 1 / stepScale))
    expect(w.count == 3, "not every material was sampled: \(w)")
    for (m, v) in w { expect(v * stepScale <= 1.0, "material \(m) oversteps: \(v)") }
}

test("inset: the hairs, the dome and its pit, the knob and the dimpled water are honest outside too") {
    let w: [Int: Float] = worstOverReport(box: SIMD3(-4, -0.5, -12), SIMD3(8, 12, 4), step: 0.02, inset: true)
    // A second pass concentrated round the tip, where the film and pore are.
    let t: [Int: Float] = worstOverReport(box: SIMD3(-1.5, -0.2, -1.5), SIMD3(1.5, 1.5, 1.5), step: 0.005, inset: true)
    // And a third in a thin shell round each hair's wall, down among the pores.
    let pores: [Int: Float] = worstOverReport(box: .zero, .zero, step: 0.004, inset: true, points: hairShellPoints(count: 80_000))
    var all: [Int: Float] = w
    for (m, v) in t { all[m] = max(all[m] ?? 0, v) }
    for (m, v) in pores { all[m] = max(all[m] ?? 0, v) }
    // Step 39: with the antenna lifted, when the water is moved away; and at
    // a moment of lift-off, when the dimple is half as deep.
    if let up = frames.max(by: { $0.lift < $1.lift }), up.lift > 0 {
        let shift: SIMD3<Float> = up.water.translation
        for (m, v) in worstOverReport(box: SIMD3(-4, -0.5, -12) - shift, SIMD3(8, 12, 4) - shift, step: 0.02,
                                      inset: true, frame: up) {
            all[m] = max(all[m] ?? 0, v)
        }
        for (m, v) in worstOverReport(box: SIMD3(-4, -0.5, -8), SIMD3(8, 12, 4), step: 0.01, inset: true, frame: up) {
            all[m] = max(all[m] ?? 0, v)
        }
    }
    if let half = liftOffMoment(dimple: 0.5 * dimpleDepth) {
        for (m, v) in worstOverReport(box: SIMD3(-1.5, -0.2, -1.5), SIMD3(1.5, 1.5, 1.5), step: 0.005, inset: true, frame: half) {
            all[m] = max(all[m] ?? 0, v)
        }
    } else { expect(false, "no lift-off moment found") }
    print(String(format: "        worst over-report: water %.2f, cuticle %.2f, taste hair %.2f, smell hair %.2f, knob %.2f",
                 all[1] ?? 0, all[2] ?? 0, all[3] ?? 0, all[4] ?? 0, all[7] ?? 0))
    expect(all.count >= 5, "not every material was sampled: \(all)")
    for (m, v) in all { expect(v * stepScale <= 1.0, "material \(m) oversteps: \(v)") }
}

section("the picture")

let render: AntImage? = {
    guard let d = gpu.device, let s = scene, let f = contactFrame else { return nil }
    return try? renderAnt(width: 960, height: 540, samples: 1, scene: s, frame: f, on: d).image
}()

test("the touch is visible: antenna and water both within 3 px of the projected contact") {
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
    expect(sawAnt && sawSugar, "near the contact: antenna \(sawAnt), water \(sawSugar)")
}

test("the inset shows both hairs, the water and the humidity knob, and no odour; the molecule inset shows O, H and hydrogen bonds") {
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
    expect(seen.isSuperset(of: [1, 2, 3, 4, 7]), "inset materials seen: \(seen.sorted())")
    expect(!seen.contains(6), "odour dots drawn in the inset")
    expectEqual(odourInset, 0)
    // Oxygen (code 1 → 2), hydrogen (2 → 3), covalent bonds (9), hydrogen bonds (10); no carbon.
    expect(taste.isSuperset(of: [2, 3, 9, 10]), "molecule inset shows \(taste.sorted())")
    expect(!taste.contains(1), "carbon in a picture of water")
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
    var outLo = SIMD2<Int>(w, h)
    var outHi = SIMD2<Int>(-1, -1)
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
                        outLo = SIMD2<Int>(min(outLo.x, x), min(outLo.y, y))
                        outHi = SIMD2<Int>(max(outHi.x, x), max(outHi.y, y))
                        if ProcessInfo.processInfo.environment["ANT_DEBUG"] != nil && outside % 25 == 0 { print("        outside \(x),\(y) frame \(f)") }
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
    print("        region \(region.rects.count) rects, \(100 * region.area / (w * h))% of the frame; changed pixels inside \(inside), outside \(outside), worst outside \(worstOutside) levels, within x \(outLo.x)–\(outHi.x), y \(outLo.y)–\(outHi.y)")
    expect(worstOutside <= 1, "a pixel outside the region changes by \(worstOutside) levels")
    expect(outside < w * h / 1000, "\(outside) pixels outside the region change")
    expect(inside > 0, "nothing moved")
}

test("lifted, the main view shows the tip well off the drop, and the inset no water at all") {
    guard let d = gpu.device, let s = scene, let up = frames.max(by: { $0.lift < $1.lift }) else { expect(false); return }
    guard let img = try? renderAnt(width: 1920, height: 1080, samples: 1, scene: s, frame: up, on: d).image else {
        expect(false, "render failed"); return
    }
    let a: Antenna = up.ant.antennae[0]
    let low: SIMD2<Float> = projectMain(a.tipCentre - SIMD3<Float>(0, a.tipRadius, 0), width: 1920, height: 1080)
    let touch: SIMD2<Float> = projectMain(s.contact.point, width: 1920, height: 1080)
    print(String(format: "        tip raised %.1f px on screen", simd_distance(low, touch)))
    expect(simd_distance(low, touch) > 10, "the lift is invisible: \(simd_distance(low, touch)) px")
    var water: Int = 0
    var hairs: Int = 0
    for y in 0..<1080 {
        for x in 0..<1920 {
            let v: SIMD4<Float> = img.seen(x, y)
            if v.x == 2 && v.y == 1 { water += 1 }
            if v.x == 2 && (v.y == 3 || v.y == 4) { hairs += 1 }
        }
    }
    expectEqual(water, 0)
    expect(hairs > 1000, "the hairs are gone from the inset")
}

finish()
