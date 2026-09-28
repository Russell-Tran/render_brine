// Tests for step 49: an ant taps one raw grain of rice. Step 43's tests,
// copied where they apply — the ant against the literature, the hairs and
// their pores, the motion as step 30 tests it (the touch at every touch-down,
// no penetration at any frame, a clear lift, a forward loop), the odour (dots
// that never pop, a rigid odorant, odour at every frame) — and new ones for
// the grain: its size against LaKast's measurements, dry with no film, the
// exact distance to it, the amylose chain's formula and α-1,4 links, and
// nothing ever reaching the taste pore.
//
// The anatomy is checked against the literature it cites, the contact and
// the pores against the GPU's own distance functions — the same source that
// draws the picture — and the finished frame for what it shows where.
//
// ANT_MUTANT=segments13|hover|tastePores|noOdour|grainSize|press|rewind breaks
// the scene on purpose; `make mutants` requires the suite to fail for each.

import Foundation
import Metal
import simd

let mutant: Mutant = {
    switch ProcessInfo.processInfo.environment["ANT_MUTANT"] {
    case "segments13": return .segments13
    case "hover": return .hover
    case "tastePores": return .tastePores
    case "noOdour": return .noOdour
    case "grainSize": return .grainSize
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

/// Every frame of the GIF's loop, and the first — the touch-down.
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
    guard let f = contactFrame else { expect(false); return }
    expectEqual(f.ant.antennae.count, 2)
    for (i, a) in f.ant.antennae.enumerated() {
        expectEqual(a.segments.count, workerAntennaSegments)
        // Count what goes to the GPU, too: shapes tagged as this antenna.
        var drawn: Int = 0
        for x in f.ant.shapes where x.part == AntPart.antenna && x.index == i { drawn += 1 }
        expectEqual(drawn, 12)
    }
}

test("the antennae are elbowed, and the funiculus has no club") {
    guard let f = contactFrame else { expect(false); return }
    for a in f.ant.antennae {
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
    guard let f = contactFrame else { expect(false); return }
    let head: Shape = f.ant.body[0]
    expect(abs(head.b.z * 2 - headWidth) < 1e-4, "drawn head width \(head.b.z * 2)")
}

test("six legs, and every one joins the mesosoma — none the head, petiole or gaster") {
    guard let f = contactFrame else { expect(false); return }
    expectEqual(f.ant.legs.count, 6)
    let meso: [Shape] = f.ant.body.filter { $0.part == .mesosoma }
    let others: [Shape] = f.ant.body.filter { $0.part == .head || $0.part == .petiole || $0.part == .gaster }
    for leg in f.ant.legs {
        expect(meso.contains { $0.contains(leg.root) }, "\(leg.name) root \(leg.root) is not inside the mesosoma")
        expect(!others.contains { $0.contains(leg.root) }, "\(leg.name) root is inside another tagma")
        // The first segment starts at that root: the leg is attached, not floating.
        expect(simd_distance(leg.shapes[0].a, leg.root) < 1e-6, "\(leg.name) coxa does not start at its root")
        // And the tarsus tip stands on the table.
        let last: Shape = leg.shapes[leg.shapes.count - 1]
        expect(abs(last.b.y - last.rb) < 0.004, "\(leg.name) tarsus tip at \(last.b.y - last.rb) mm above the table")
    }
    let legShapes: Int = f.ant.shapes.filter { $0.part == .leg }.count
    expectEqual(legShapes, 6 * 9)
}

test("one petiole, a single node between mesosoma and gaster, as in all Formicinae") {
    guard let f = contactFrame else { expect(false); return }
    let pet: [Shape] = f.ant.body.filter { $0.part == .petiole }
    expectEqual(pet.count, 1)
    let mesoBack: Float = f.ant.body.filter { $0.part == .mesosoma }.map { $0.extent(along: SIMD3(1, 0, 0)).lo }.min() ?? 0
    let gasterFront: Float = f.ant.body.filter { $0.part == .gaster }.map { $0.extent(along: SIMD3(1, 0, 0)).hi }.max() ?? 0
    expect(pet[0].a.x < mesoBack + 0.05 && pet[0].a.x > gasterFront - 0.05,
           "petiole at x \(pet[0].a.x), mesosoma ends \(mesoBack), gaster starts \(gasterFront)")
}

test("the worker is 3.4–5.0 mm long, mandibles to gaster tip") {
    guard let f = contactFrame else { expect(false); return }
    let l: Float = f.ant.length
    print(String(format: "        length %.2f mm", l))
    expect(workerLengthRange.contains(l), "length \(l) mm")
}

section("the grain")

test("the grain is LaKast's long grain as measured: 7.60 × 2.08 × 1.73 mm, drawn at that size") {
    // US 9,398,750 B2, Table 10 (Food.swift).
    expect(abs(grainLength - 7.60) < 1e-5 && abs(grainWidth - 2.08) < 1e-5 && abs(grainThickness - 1.73) < 1e-5)
    // A long grain by FAO's grading of milled rice: 6.0 mm or more.
    expect(grainLength >= 6.0, "\(grainLength) mm is not a long grain")
    guard let s = scene else { expect(false); return }
    let g: RiceGrain = s.grain
    let drawn: SIMD3<Float> = g.semi * 2
    print(String(format: "        drawn: %.2f long, %.2f wide, %.2f thick (mm)", drawn.x, drawn.z, drawn.y))
    expect(abs(drawn.x - grainLength) < 1e-4, "drawn length \(drawn.x) mm")
    expect(abs(drawn.z - grainWidth) < 1e-4, "drawn width \(drawn.z) mm")
    expect(abs(drawn.y - grainThickness) < 1e-4, "drawn thickness \(drawn.y) mm")
    // And the GPU draws that grain: probe the three extremes of its axes.
    let ends: [SIMD3<Float>] = [g.world(SIMD3<Float>(g.semi.x, 0, 0)), g.world(SIMD3<Float>(0, g.semi.y, 0)),
                                g.world(SIMD3<Float>(0, 0, -g.semi.z))]
    guard let r = probe(ends) else { expect(false, "probe failed"); return }
    for (k, p) in r.enumerated() { expect(abs(p.food) < 1e-4, "axis end \(k) is \(p.food) mm off the drawn grain") }
}

test("true scale: the grain is longer than the ant and taller than it") {
    guard let s = scene, let f = contactFrame else { expect(false); return }
    let g: RiceGrain = s.grain
    let antTop: Float = f.ant.body.map { $0.extent(along: SIMD3(0, 1, 0)).hi }.max() ?? 0
    print(String(format: "        grain %.2f mm long against the ant's %.2f mm; %.2f mm high against its %.2f mm",
                 2 * g.semi.x, f.ant.length, 2 * g.semi.y, antTop))
    expect(2 * g.semi.x > 1.5 * f.ant.length, "the grain is only \(2 * g.semi.x / f.ant.length)× the ant's length")
    expect(2 * g.semi.y > 1.4 * antTop, "the grain is only \(2 * g.semi.y / antTop)× the ant's height")
}

test("it lies flat on the card, long axis level, its lowest point on the table") {
    guard let s = scene else { expect(false); return }
    let g: RiceGrain = s.grain
    expect(abs(g.axis.y) < 1e-6 && abs(simd_length(g.axis) - 1) < 1e-6, "axis \(g.axis)")
    expect(abs(g.centre.y - g.semi.y) < 1e-6, "centre height \(g.centre.y) for half-thickness \(g.semi.y)")
    let foot = SIMD3<Float>(g.centre.x, 0, g.centre.z)
    expect(abs(grainSDF(foot, g)) < 1e-5, "the grain's underside is \(grainSDF(foot, g)) mm off the card")
}

test("raw rice is dry: no film, no meniscus — and pale, and with a faint smell") {
    expect(!grainHasFilm)
    expectEqual(filmThickness, 0)
    expectEqual(meniscusHeight, 0)
    // USDA fdcId 169756: 11.62 g water, 0.12 g sugars per 100 g.
    expect(abs(riceWaterGramsPer100g - 11.62) < 1e-4 && abs(riceSugarsGramsPer100g - 0.12) < 1e-4)
    let c: SIMD3<Float> = riceAlbedo
    expect(c.x > 0.55 && c.y > 0.55 && c.z > 0.45, "rice colour \(c) is not pale")
    expect(c.x >= c.y && c.y >= c.z && c.x - c.z < 0.15, "rice colour \(c) is not a warm white")
    expect(objectHasOdour)
}

test("the grain's distance is the true distance: Newton against a dense sampling of the surface, and the GPU against the CPU") {
    guard let s = scene else { expect(false); return }
    let g: RiceGrain = s.grain
    let e: SIMD3<Float> = g.semi
    // The surface, sampled densely by angle.
    var surface: [SIMD3<Float>] = []
    let nu: Int = 720, nv: Int = 360
    for i in 0...nv {
        let v: Float = Float.pi * Float(i) / Float(nv)
        for j in 0..<nu {
            let u: Float = 2 * Float.pi * Float(j) / Float(nu)
            surface.append(SIMD3<Float>(e.x * cos(v), e.y * sin(v) * cos(u), e.z * sin(v) * sin(u)))
        }
    }
    var rng = SystemRandomNumberGenerator()
    var worstLong: Float = 0      // Newton reporting more than the sampled nearest point: must not happen
    var worstShort: Float = 0     // Newton reporting less: allowed only by the sampling's coarseness
    var pts: [SIMD3<Float>] = []
    for _ in 0..<300 {
        let q = SIMD3<Float>(Float.random(in: -5.5...5.5, using: &rng), Float.random(in: -2...2, using: &rng),
                             Float.random(in: -2.5...2.5, using: &rng))
        let d: Float = ellipsoidDistance(q, e)
        guard d > 0 else { continue }
        var near: Float = 1e9
        for p in surface { near = min(near, simd_distance(p, q)) }
        worstLong = max(worstLong, d - near)
        worstShort = max(worstShort, near - d)
        pts.append(g.world(q))
    }
    print(String(format: "        %d points: Newton exceeds the sampled nearest by %.1e mm at most, falls short by %.1e mm at most",
                 pts.count, worstLong, worstShort))
    expect(worstLong < 1e-4, "the distance claims \(worstLong) mm more room than there is")
    expect(worstShort < 0.01, "the distance is \(worstShort) mm short")
    guard let r = probe(pts) else { expect(false, "probe failed"); return }
    var worstGPU: Float = 0
    for (k, p) in pts.enumerated() { worstGPU = max(worstGPU, abs(r[k].food - grainSDF(p, g))) }
    print(String(format: "        GPU against CPU: %.1e mm", worstGPU))
    expect(worstGPU < 1e-4, "the kernel's grain differs from Food.swift's by \(worstGPU) mm")
}

section("the touch")

test("the right antenna's tip touches the grain: distance 0 within 2 µm, and no penetration") {
    guard let s = scene, let f = contactFrame else { expect(false); return }
    let a: Antenna = f.ant.antennae[0]
    let (point, normal) = contactPoint()
    // On the CPU: the contact point lies on the grain, and the normal there
    // is the grain's own (the gradient of its implicit function).
    expect(abs(grainSDF(point, s.grain)) < 1e-4, "contact point is off the surface: \(grainSDF(point, s.grain))")
    let l: SIMD3<Float> = s.grain.local(point)
    let grad: SIMD3<Float> = simd_normalize(s.grain.worldDirection(l / (s.grain.semi * s.grain.semi)))
    expect(simd_distance(grad, normal) < 1e-3, "normal \(normal) against the gradient \(grad)")
    expect(point.y > 1.0, "contact too low: \(point.y)")
    // Facing the camera, so the touch can be seen.
    expect(simd_dot(normal, -mainDirection) > 0.3, "the contact faces away from the camera")
    // On the GPU: the lowest point of the tip is on the grain.
    let lowest: SIMD3<Float> = a.tipCentre - normal * a.tipRadius
    guard let r = probe([lowest]) else { expect(false, "probe failed"); return }
    print(String(format: "        tip to grain %.4f mm, tip to its own surface %.4f mm", r[0].food, r[0].ant))
    expect(abs(r[0].food) < 0.002, "the tip's nearest point is \(r[0].food) mm from the grain")
    expect(abs(r[0].ant) < 0.002, "that point is not on the antenna: \(r[0].ant)")
    // No part of the antenna inside the grain: sample a box round the tip.
    var pts: [SIMD3<Float>] = []
    var rng = SystemRandomNumberGenerator()
    for _ in 0..<60_000 {
        pts.append(point + SIMD3<Float>(Float.random(in: -0.15...0.15, using: &rng),
                                        Float.random(in: -0.15...0.15, using: &rng),
                                        Float.random(in: -0.15...0.15, using: &rng)))
    }
    guard let q = probe(pts) else { expect(false, "probe failed"); return }
    var worst: Float = 0
    for p in q where p.ant < 0 { worst = min(worst, p.food) }
    expect(worst > -0.002, "antenna reaches \(-worst) mm into the grain")
}

test("no part of the ant is inside the grain") {
    guard let s = scene, let f = contactFrame else { expect(false); return }
    var worst: Float = 9
    for shape in f.ant.shapes where shape.kind == .roundCone {
        for k in 0...20 {
            let t: Float = Float(k) / 20
            let c: SIMD3<Float> = shape.a + (shape.b - shape.a) * t
            worst = min(worst, grainSDF(c, s.grain) - (shape.ra + (shape.rb - shape.ra) * t))
        }
    }
    print(String(format: "        nearest ant surface to the grain: %.4f mm", worst))
    expect(worst > -0.001, "an ant part reaches into the grain: \(worst)")
}

test("in the inset the taste hair's apex just meets the grain's surface — dry, nothing climbs it") {
    guard let s = scene else { expect(false); return }
    let h: Sensillum = s.hairs[0]
    let R: Float = grainSurfaceRadiusMicrometres
    func grain(_ p: SIMD3<Float>) -> Float { simd_length(p - SIMD3<Float>(0, -R, 0)) - R }
    var lowest: Float = 1e9
    var rng = SystemRandomNumberGenerator()
    for _ in 0..<20_000 {
        let d: SIMD3<Float> = simd_normalize(SIMD3<Float>(Float.random(in: -1...1, using: &rng), Float.random(in: -1...1, using: &rng),
                                                          Float.random(in: -1...1, using: &rng)))
        let t: Float = Float.random(in: 0...1, using: &rng)
        let p: SIMD3<Float> = h.base + (h.tip - h.base) * t + d * (h.baseRadius + (h.tipRadius - h.baseRadius) * t)
        if roundConeDistance(p, h.base, h.tip, h.baseRadius, h.tipRadius) <= 1e-4 { lowest = min(lowest, grain(p)) }
    }
    lowest = min(lowest, grain(h.tip - SIMD3<Float>(0, h.tipRadius, 0)))
    print(String(format: "        grain radius at the contact %.0f µm; hair's deepest point relative to the grain %.4f µm", R, lowest))
    expect(lowest > -0.005, "the hair dips \(-lowest) µm into the grain")
    expect(lowest < 0.02, "the hair hovers \(lowest) µm above the grain")
    // The inset's sphere is the grain's own curvature there: between its
    // smallest and largest possible radii.
    expect(R > 1000 * s.grain.semi.y * s.grain.semi.y / s.grain.semi.x && R < 1000 * s.grain.semi.x * s.grain.semi.x / s.grain.semi.y,
           "radius \(R) µm")
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

section("smell: a faint trace of hexanal")

/// The odour dots drawn at one frame: those more than half grown.
func shownDots(_ f: FrameState) -> [Volatile] { f.volatiles.filter { $0.drawn > 0.5 } }

test("raw rice gives the smell hair a faint trace at every frame: one or two hexanal dots, each at or heading for a wall pore") {
    guard let s = scene else { expect(false); return }
    let smell: Sensillum = s.hairs[1]
    let taste: Sensillum = s.hairs[0]
    expect(objectHasOdour)
    expectEqual(s.odourPaths.count, odourSlots)
    // Faint: fewer than the four step 43's cheese sauce kept on the way.
    expectEqual(odourSlots, 2)
    var fewest: Int = 99
    var most: Int = 0
    var worstWall: Float = 9
    var worstTaste: Float = 9
    var worstGrain: Float = 9
    for f in frames {
        fewest = min(fewest, shownDots(f).count)
        most = max(most, shownDots(f).count)
        for v in f.volatiles where v.drawn > 0.01 {
            expect(v.pore.row >= 0 && v.pore.row < smell.poreRows && v.pore.column >= 0 && v.pore.column < smell.poreColumns,
                   "pore \(v.pore) is not on the lattice")
            let (mouth, outward) = smell.wallPore(row: v.pore.row, column: v.pore.column)
            expect(simd_dot(v.position - mouth, outward) >= 0, "an odour molecule inside the hair")
            let rr: Float = v.radius
            worstWall = min(worstWall, roundConeDistance(v.position, smell.base, smell.tip, smell.baseRadius, smell.tipRadius) - rr)
            worstTaste = min(worstTaste, roundConeDistance(v.position, taste.base, taste.tip, taste.baseRadius, taste.tipRadius) - rr)
            worstGrain = min(worstGrain, f.crystal.toCrystal(v.position).y - rr)
        }
    }
    print(String(format: "        %d frames: %d–%d dots shown; nearest dot surface to the smell hair's wall %.3f µm, taste hair %.2f µm, grain %.2f µm",
                 frames.count, fewest, most, worstWall, worstTaste, worstGrain))
    expect(fewest >= 1, "a frame shows no odour at all")
    expect(most <= 2, "a frame shows \(most) dots: not a faint trace")
    expect(worstWall > -0.002, "a dot sinks \(-worstWall) µm into the smell hair's wall")
    expect(worstTaste > 0, "a dot touches the taste hair")
    expect(worstGrain > 0, "a dot is in the grain")
    guard let c = contactFrame else { expect(false); return }
    var arriving: Int = 0
    for v in c.volatiles {
        let (mouth, _) = smell.wallPore(row: v.pore.row, column: v.pore.column)
        if simd_distance(v.position, mouth) < 2 * volatileDotRadius + smell.wallPoreRadius && v.drawn > 0.9 { arriving += 1 }
    }
    expect(arriving >= 1, "no odour molecule at a pore at touch-down")
    for p in s.odourPaths {
        guard let r = probe([p.mouth - p.outward * 0.02]) else { expect(false, "probe failed"); return }
        expect(r[0].smellHair > 0, "no pore at the mouth a molecule is heading for: \(r[0].smellHair)")
    }
}

section("the molecules")

func dihedral(_ p0: SIMD3<Float>, _ p1: SIMD3<Float>, _ p2: SIMD3<Float>, _ p3: SIMD3<Float>) -> Float {
    let b1: SIMD3<Float> = p1 - p0, b2: SIMD3<Float> = p2 - p1, b3: SIMD3<Float> = p3 - p2
    let n1: SIMD3<Float> = simd_cross(b1, b2), n2: SIMD3<Float> = simd_cross(b2, b3)
    return deg(atan2(simd_dot(simd_cross(n1, n2), simd_normalize(b2)), simd_dot(n1, n2)))
}

/// The pyranose rings of a sugar: each ring O and its five carbons, in order
/// round the ring from the anomeric carbon C1 to C5 (C5 is the one carrying
/// the exocyclic C6).
struct Pyranose {
    var ringO: Int
    var carbons: [Int]    // C1 … C5
    var c6: Int
}

func pyranoses(_ m: Molecule) -> [Pyranose] {
    var nb: [[Int]] = Array(repeating: [], count: m.atoms.count)
    for (a, b) in m.bonds { nb[a].append(b); nb[b].append(a) }
    func isC(_ i: Int) -> Bool { m.atoms[i].element == "C" }
    var out: [Pyranose] = []
    for o in m.atoms.indices where m.atoms[o].element == "O" {
        let cs: [Int] = nb[o].filter(isC)
        guard cs.count == 2 else { continue }
        // A path of five carbons from one neighbour to the other, not through o.
        func paths(_ from: Int, _ to: Int, _ seen: [Int]) -> [Int]? {
            if seen.count == 5 { return from == to ? seen : nil }
            for k in nb[from] where isC(k) && !seen.contains(k) {
                if let p = paths(k, to, seen + [k]) { return p }
            }
            return nil
        }
        guard let ring = paths(cs[0], cs[1], [cs[0]]) else { continue }
        // C5 carries a carbon outside the ring; C1 does not.
        func exoC(_ c: Int) -> Int? { nb[c].first { isC($0) && !ring.contains($0) } }
        if let c6 = exoC(ring[4]), exoC(ring[0]) == nil {
            out.append(Pyranose(ringO: o, carbons: ring, c6: c6))
        } else if let c6 = exoC(ring[0]), exoC(ring[4]) == nil {
            out.append(Pyranose(ringO: o, carbons: Array(ring.reversed()), c6: c6))
        }
    }
    return out
}

test("the amylose piece is C24H42O21: four glucose rings, each joined to the next C1–O–C4, every link α") {
    guard let s = scene else { expect(false); return }
    let m: Molecule = s.chemistry.amylose
    let f: [String: Int] = m.formula
    expect(f["C"] == 24 && f["H"] == 42 && f["O"] == 21 && f.count == 3, "amylose piece is \(f)")
    expectEqual(m.atoms.count, 87)
    expectEqual(m.rings, 4)
    expect(m.orders.allSatisfy { $0 == 1 }, "a sugar has no double bonds")
    let rings: [Pyranose] = pyranoses(m)
    expectEqual(rings.count, 4)
    var nb: [[Int]] = Array(repeating: [], count: m.atoms.count)
    for (a, b) in m.bonds { nb[a].append(b); nb[b].append(a) }
    var links14: Int = 0
    var alpha: Int = 0
    for o in m.atoms.indices where m.atoms[o].element == "O" {
        let cs: [Int] = nb[o].filter { m.atoms[$0].element == "C" }
        guard cs.count == 2, !rings.contains(where: { $0.ringO == o }) else { continue }
        // A glycosidic oxygen: C1 of one ring to C4 of another.
        guard let r1 = rings.first(where: { cs.contains($0.carbons[0]) }),
              let r4 = rings.first(where: { cs.contains($0.carbons[3]) }), r1.ringO != r4.ringO else { continue }
        links14 += 1
        // α: the link leaves C1 on the far face of the ring from C6.
        let order: [Int] = [r1.ringO] + r1.carbons
        var normal = SIMD3<Float>(0, 0, 0)
        for k in 0..<order.count {
            normal += simd_cross(m.atoms[order[k]].position, m.atoms[order[(k + 1) % order.count]].position)
        }
        let c1: SIMD3<Float> = m.atoms[r1.carbons[0]].position
        let c5: SIMD3<Float> = m.atoms[r1.carbons[4]].position
        let sideO: Float = simd_dot(m.atoms[o].position - c1, normal)
        let sideC6: Float = simd_dot(m.atoms[r1.c6].position - c5, normal)
        if sideO * sideC6 < 0 { alpha += 1 }
    }
    print("        \(rings.count) pyranose rings, \(links14) C1–O–C4 links, \(alpha) of them α")
    expectEqual(links14, 3)
    expectEqual(alpha, 3)
    // It fits its inset.
    for a in m.atoms { expect(simd_length(SIMD2<Float>(a.position.x, a.position.y)) + 0.4 < moleculeField / 2, "an atom leaves the inset") }
}

test("hexanal is C6H12O with one C=O; every atom in both insets has its valence") {
    guard let s = scene else { expect(false); return }
    let o: Molecule = s.chemistry.odorant
    let f: [String: Int] = o.formula
    expect(f["C"] == 6 && f["H"] == 12 && f["O"] == 1 && f.count == 3, "hexanal is \(f)")
    expectEqual(o.rings, 0)
    expectEqual(o.bonds.indices.filter { o.orders[$0] == 2 }.count, 1)
    for a in o.atoms { expectEqual(a.view, 1) }
    for a in s.chemistry.amylose.atoms { expectEqual(a.view, 0) }
    guard let f0 = contactFrame else { expect(false); return }
    let all: Molecule = f0.molecule
    let v: [Int] = all.valences
    for (i, a) in all.atoms.enumerated() {
        let want: Int
        switch a.element {
        case "C": want = 4
        case "O": want = 2
        case "H": want = 1
        default: want = -1
        }
        expect(v[i] == want, "\(a.element) \(i) has valence \(v[i])")
    }
}

section("the tap (step 30's, on a grain of rice)")

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

/// The grain's distance from the right antenna at one frame, on the GPU: the
/// tip's nearest point (towards the contact, along the grain's normal), and
/// the deepest any antenna point reaches into the grain among points packed
/// round the tip.
func tipSurvey(_ f: FrameState, points n: Int) -> (lowest: Float, deepest: Float)? {
    guard let s = scene else { return nil }
    let a: Antenna = f.ant.antennae[0]
    let lowest: SIMD3<Float> = a.tipCentre - s.contact.normal * a.tipRadius
    var pts: [SIMD3<Float>] = [lowest]
    var rng = SystemRandomNumberGenerator()
    for _ in 0..<n {
        pts.append(a.tipCentre + SIMD3<Float>(Float.random(in: -0.16...0.16, using: &rng),
                                              Float.random(in: -0.16...0.16, using: &rng),
                                              Float.random(in: -0.16...0.16, using: &rng)))
    }
    guard let q = probe(pts, frame: f) else { return nil }
    var deepest: Float = 1
    for p in q.dropFirst() where p.ant < 0 { deepest = min(deepest, p.food) }
    return (q[0].food, deepest)
}

test("at every touch-down the tip meets the grain: 0 within 2 µm, on the GPU") {
    let downs: [FrameState] = frames.filter { $0.lift == 0 }
    expect(downs.count >= tapsPerLoop, "only \(downs.count) touching frames")
    var worst: Float = 0
    for f in downs {
        guard let r = tipSurvey(f, points: 0) else { expect(false, "probe failed"); return }
        worst = max(worst, abs(r.lowest))
    }
    print(String(format: "        %d touching frames; worst tip-to-grain %.4f mm", downs.count, worst))
    expect(worst < 0.002, "a touching tip is \(worst) mm from the grain")
}

test("at no frame does any part of the antenna go into the grain") {
    var worst: Float = 1
    for f in frames {
        guard let r = tipSurvey(f, points: 12_000) else { expect(false, "probe failed"); return }
        worst = min(worst, r.deepest)
    }
    print(String(format: "        %d frames × 12,000 points: nearest antenna point to the inside of the grain %.4f mm", frames.count, worst))
    expect(worst > -0.002, "the antenna reaches \(-worst) mm into the grain")
}

test("at no frame is any part of the ant inside the grain (every segment, on the CPU)") {
    guard let s = scene else { expect(false); return }
    var worst: Float = 9
    for f in frames {
        for shape in f.ant.shapes where shape.kind == .roundCone {
            for k in 0...20 {
                let t: Float = Float(k) / 20
                let c: SIMD3<Float> = shape.a + (shape.b - shape.a) * t
                worst = min(worst, grainSDF(c, s.grain) - (shape.ra + (shape.rb - shape.ra) * t))
            }
        }
    }
    print(String(format: "        %d frames: nearest ant surface to the grain %.4f mm", frames.count, worst))
    expect(worst > -0.001, "an ant part reaches into the grain: \(worst)")
}

test("lifted, the tip clears the grain — by the lift, and never less than 0.03 mm near the top") {
    let ups: [FrameState] = frames.filter { $0.lift > 0.5 * liftHeight }
    expect(!ups.isEmpty, "no lifted frames")
    var least: Float = 1
    for f in ups {
        guard let r = tipSurvey(f, points: 4000) else { expect(false, "probe failed"); return }
        least = min(least, r.lowest)
        // The grain's distance is exact, so the tip's lowest point is off it
        // by the lift, less the little the grain curves away under it.
        expect(abs(r.lowest - f.lift) < 0.002 + 0.03 * f.lift, "lift \(f.lift) mm but the tip is \(r.lowest) mm off")
        expect(r.deepest > 0, "a lifted antenna point is in the grain")
    }
    print(String(format: "        %d lifted frames, least clearance %.3f mm (lift %.3f mm at the top)", ups.count, least, liftHeight))
    expect(least > 0.03)
}

test("the taste hair meets the grain at touch-down, and leaves it when lifted — never dipping in (inset, GPU)") {
    guard let s = scene, let c = contactFrame else { expect(false); return }
    let h: Sensillum = s.hairs[0]
    let apexLow = SIMD3<Float>(h.tip.x, h.tip.y - h.tipRadius, h.tip.z)
    // Just under the surface beside the apex: grain at touch-down, air when lifted.
    let beside: SIMD3<Float> = apexLow + SIMD3<Float>(0.7, -0.03, 0)
    expect(abs(c.crystal.toCrystal(apexLow).y) < 0.002, "at touch-down the apex is \(c.crystal.toCrystal(apexLow).y) µm off the grain")
    guard let r0 = probe([beside, apexLow], frame: c) else { expect(false, "probe failed"); return }
    expect(r0[0].insetMaterial == 1 && r0[0].insetDistance < 0, "beside the apex at touch-down: material \(r0[0].insetMaterial), \(r0[0].insetDistance) µm")
    expect(abs(r0[1].insetDistance) < 0.005, "the apex's lowest point is \(r0[1].insetDistance) µm from any surface")
    var surface: [SIMD3<Float>] = [apexLow]
    var rng = SystemRandomNumberGenerator()
    while surface.count < 6000 {
        let d: SIMD3<Float> = simd_normalize(SIMD3<Float>(Float.random(in: -1...1, using: &rng), Float.random(in: -1...1, using: &rng),
                                                          Float.random(in: -1...1, using: &rng)))
        let t: Float = Float.random(in: 0.8...1, using: &rng)
        let p: SIMD3<Float> = h.base + (h.tip - h.base) * t + d * (h.baseRadius + (h.tipRadius - h.baseRadius) * t)
        if abs(roundConeDistance(p, h.base, h.tip, h.baseRadius, h.tipRadius)) < 1e-3 { surface.append(p) }
    }
    let R: Float = grainSurfaceRadiusMicrometres
    var deepest: Float = 9
    var lowestLifted: Float = 1e9
    for f in frames {
        for p in surface {
            let q: SIMD3<Float> = f.crystal.toCrystal(p)
            deepest = min(deepest, simd_length(q - SIMD3<Float>(0, -R, 0)) - R)
        }
        guard f.lift > 0.001 else { continue }
        let y: Float = f.crystal.toCrystal(apexLow).y
        lowestLifted = min(lowestLifted, y - 1000 * f.lift)
        expect(y > 0.5, "lifted \(f.lift) mm, the apex is only \(y) µm up")
        guard let r = probe([beside], frame: f) else { expect(false, "probe failed"); return }
        expect(r[0].insetMaterial != 1 || r[0].insetDistance > 0, "lifted, the grain is still at the apex")
    }
    print(String(format: "        hair's deepest point into the grain over all frames %.4f µm; grain drop vs lift, worst difference %.2f µm",
                 -deepest, abs(lowestLifted)))
    expect(deepest > -0.005, "the hair dips \(-deepest) µm into the grain")
    expect(abs(lowestLifted) < 3, "the inset grain and the main-view lift disagree by \(lowestLifted) µm")
}

section("the odour in motion, and nothing at the taste pore")

/// Each atom of `b` matched to the nearest atom of the same element and inset in `a`.
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

/// The biggest move of any shown odour dot between two moments, slot by slot
/// (a slot that is not drawn at either moment has nothing to show moving).
func dotStep(_ a: FrameState, _ b: FrameState) -> (move: Float, grow: Float) {
    var move: Float = 0
    var grow: Float = 0
    for (x, y) in zip(a.volatiles, b.volatiles) {
        grow = max(grow, abs(x.radius - y.radius))
        if x.drawn > 0.01 && y.drawn > 0.01 { move = max(move, simd_distance(x.position, y.position)) }
    }
    return (move, grow)
}

test("the loop is forward: the last frame runs on into the first, and the odour only arrives") {
    guard let s = scene, let c = contactFrame else { expect(false); return }
    let end: FrameState = s.frame(at: loopSeconds)
    expect(simd_distance(end.ant.antennae[0].tipCentre, c.ant.antennae[0].tipCentre) < 1e-5)
    expectEqual(end.molecule.atoms.count, c.molecule.atoms.count)
    var endWorst: Float = 0
    for (a, b) in zip(end.molecule.atoms, c.molecule.atoms) { endWorst = max(endWorst, simd_distance(a.position, b.position)) }
    for (a, b) in zip(end.volatiles, c.volatiles) { endWorst = max(endWorst, simd_distance(a.position, b.position), abs(a.drawn - b.drawn)) }
    expect(endWorst < 1e-3, "time L differs from time 0 by \(endWorst)")
    func step(_ a: FrameState, _ b: FrameState) -> Float {
        var m: Float = simd_distance(a.ant.antennae[0].tipCentre, b.ant.antennae[0].tipCentre) * 100
        let (move, grow) = dotStep(a, b)
        m = max(m, move, grow)
        for (_, _, d) in matchAtoms(a.molecule, b.molecule) { m = max(m, d / 10) }
        return m
    }
    var biggest: Float = 0
    for i in 1..<frames.count { biggest = max(biggest, step(frames[i - 1], frames[i])) }
    let seam: Float = step(frames[frames.count - 1], frames[0])
    print(String(format: "        seam step %.4f, largest step inside the loop %.4f", seam, biggest))
    expect(seam <= biggest * 1.01 + 1e-5, "the seam jumps")
    var last: Float = odourProgress(0, mutant: s.mutant)
    for i in 1...frames.count {
        let u: Float = odourProgress(frameTime(i, of: frames.count) - (i == frames.count ? 1e-4 : 0), mutant: s.mutant)
        expect(u >= last - 1e-5, "the odour goes back at frame \(i): \(last) → \(u)")
        last = u
    }
    expect(abs(last - Float(tapsPerLoop)) < 0.01, "the loop is \(last) taps of odour, not \(tapsPerLoop)")
    // Every shown dot only comes nearer its pore, frame to frame; and one
    // reaches its pore every other tap: two arrivals a loop, two taps apart.
    var arrivals: [Int] = Array(repeating: 0, count: tapsPerLoop)
    for i in 0..<frames.count {
        let a: FrameState = frames[i]
        let b: FrameState = i + 1 < frames.count ? frames[i + 1] : frames[0]
        for (k, (x, y)) in zip(a.volatiles, b.volatiles).enumerated() {
            let p: OdourPath = s.odourPaths[k]
            if x.drawn > 0.01 && y.drawn > 0.01 && y.life > x.life {
                expect(simd_distance(y.position, p.mouth) <= simd_distance(x.position, p.mouth) + 1e-5, "a dot backs away at frame \(i)")
            }
            if x.life < odourArrive && y.life >= odourArrive { arrivals[(i + 1) % frames.count * tapsPerLoop / frames.count] += 1 }
            expect(y.life >= x.life || x.life - y.life > 0.9, "a dot's trip runs backwards at frame \(i)")
        }
    }
    print("        arrivals per tap: \(arrivals)")
    expectEqual(arrivals, [1, 0, 1, 0])
    for i in 0..<(frames.count - frames.count / tapsPerLoop) {
        expect(abs(frames[i].lift - frames[i + frames.count / tapsPerLoop].lift) < 1e-5)
    }
}

test("nothing pops: every odour dot and every atom moves or grows continuously, even as a trip starts again") {
    guard let s = scene else { expect(false); return }
    // A millisecond apart, a dot drifts a thousandth of a µm and grows or
    // shrinks by less; one that jumped or appeared whole would be far more.
    // Check at every frame, either side of each instant a slot's trip starts
    // again, and the loop's seam.
    var times: [Float] = (0..<frames.count).map { frameTime($0, of: frames.count) }
    for p in s.odourPaths {
        for n in 0...2 {
            let u: Float = Float(tripTaps) * (Float(n) - p.offset)
            let t: Float = u * tapSeconds
            if t > 0 && t < loopSeconds { times += [t - 2e-3, t - 1e-3, t, t + 1e-3] }
        }
    }
    times.append(loopSeconds - 1e-3)
    var pops: Int = 0
    var worstMove: Float = 0
    var worstGrow: Float = 0
    for t in times {
        let a: FrameState = s.frame(at: t)
        let b: FrameState = s.frame(at: t + 1e-3)
        let (move, grow) = dotStep(a, b)
        if move > 0.02 || grow > 0.005 { pops += 1 }
        worstMove = max(worstMove, move)
        worstGrow = max(worstGrow, grow)
        for (_, _, d) in matchAtoms(a.molecule, b.molecule) where d > 0.2 { pops += 1 }
    }
    print(String(format: "        %d moments: largest dot move in 1 ms %.4f µm, largest radius change %.5f µm; %d pops", times.count, worstMove, worstGrow, pops))
    expectEqual(pops, 0)
}

test("every frame: the odorant moves rigidly, the amylose not at all, and no molecule leaves its circle") {
    guard let s = scene, let c = contactFrame else { expect(false); return }
    let o0: [Atom] = c.odorant.atoms
    var worstRigid: Float = 0
    var worstStill: Float = 0
    var worstOut: Float = -9
    for f in frames {
        let o: [Atom] = f.odorant.atoms
        expectEqual(o.count, o0.count)
        for i in 0..<o.count {
            for j in (i + 1)..<o.count {
                let d0: Float = simd_distance(o0[i].position, o0[j].position)
                worstRigid = max(worstRigid, abs(simd_distance(o[i].position, o[j].position) - d0))
            }
            worstOut = max(worstOut, simd_length(SIMD2<Float>(o[i].position.x, o[i].position.y)) + 0.4 - odourField / 2)
        }
        let det0: Float = simd_dot(simd_cross(o0[1].position - o0[0].position, o0[2].position - o0[0].position), o0[3].position - o0[0].position)
        let det1: Float = simd_dot(simd_cross(o[1].position - o[0].position, o[2].position - o[0].position), o[3].position - o[0].position)
        expect(det0 * det1 > 0, "the odorant is mirrored")
        let still: Molecule = s.chemistry.amylose
        for (a, b) in zip(f.molecule.atoms.prefix(still.atoms.count), still.atoms) {
            worstStill = max(worstStill, simd_distance(a.position, b.position))
        }
    }
    print(String(format: "        %d frames: odorant distances kept to %.1e Å, amylose still to %.1e Å; odorant's reach past its circle %.2f Å",
                 frames.count, worstRigid, worstStill, worstOut))
    expect(worstRigid < 1e-3, "the odorant deforms by \(worstRigid) Å")
    expect(worstStill < 1e-6, "the taste inset moves by \(worstStill) Å")
    expect(worstOut < 0, "the odorant leaves its circle by \(worstOut) Å")
}

test("nothing enters the taste pore: no odour dot ever comes near it, and the taste inset holds only the starch") {
    guard let s = scene else { expect(false); return }
    let h: Sensillum = s.hairs[0]
    let pore: SIMD3<Float> = h.tip + simd_normalize(h.tip - h.base) * h.tipRadius
    var nearest: Float = 1e9
    for f in frames {
        for v in f.volatiles where v.drawn > 0.001 { nearest = min(nearest, simd_distance(v.position, pore) - v.radius) }
    }
    print(String(format: "        nearest odour dot to the taste pore over all frames: %.2f µm", nearest))
    expect(nearest > 1.0, "an odour dot comes within \(nearest) µm of the taste pore")
    guard let c = contactFrame else { expect(false); return }
    let taste: [Atom] = c.molecule.atoms.filter { $0.view == 0 }
    expectEqual(taste.count, s.chemistry.amylose.atoms.count)
    expect(taste.allSatisfy { ["C", "H", "O"].contains($0.element) }, "something besides starch in the taste inset")
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
    var w: [Int: Float] = worstOverReport(box: SIMD3(-2.3, 0, -3.5), SIMD3(9, 2.2, 2.4), step: 0.01, inset: false)
    for (m, v) in worstOverReport(box: SIMD3(-2.1, 0, -0.7), SIMD3(2.2, 1.3, 0.7), step: 0.005, inset: false) {
        w[m] = max(w[m] ?? 0, v)
    }
    // As step 30: round the tip and the grain with the antenna at its highest.
    if let up = frames.max(by: { $0.lift < $1.lift }), let s = scene {
        let c: SIMD3<Float> = s.contact.point
        for (m, v) in worstOverReport(box: c - SIMD3<Float>(0.6, 0.6, 0.6), c + SIMD3<Float>(0.6, 0.6, 0.6), step: 0.004,
                                      inset: false, frame: up) {
            w[m] = max(w[m] ?? 0, v)
        }
    }
    print(String(format: "        worst over-report: table %.2f, ant %.2f, grain %.2f; the ray allows %.2f",
                 w[1] ?? 0, w[2] ?? 0, w[3] ?? 0, 1 / stepScale))
    expect(w.count == 3, "not every material was sampled: \(w)")
    for (m, v) in w { expect(v * stepScale <= 1.0, "material \(m) oversteps: \(v)") }
    // The grain's own distance is exact: it never over-reports at all.
    expect((w[3] ?? 9) < 1.01, "the grain over-reports: \(w[3] ?? 9)")
}

test("inset: the hairs, the dome and the grain are honest outside too") {
    let w: [Int: Float] = worstOverReport(box: SIMD3(-4, -0.5, -12), SIMD3(8, 12, 4), step: 0.02, inset: true)
    let t: [Int: Float] = worstOverReport(box: SIMD3(-1.5, -0.2, -1.5), SIMD3(1.5, 1.5, 1.5), step: 0.005, inset: true)
    let pores: [Int: Float] = worstOverReport(box: .zero, .zero, step: 0.004, inset: true, points: hairShellPoints(count: 80_000))
    var all: [Int: Float] = w
    for (m, v) in t { all[m] = max(all[m] ?? 0, v) }
    for (m, v) in pores { all[m] = max(all[m] ?? 0, v) }
    if let up = frames.max(by: { $0.lift < $1.lift }), up.lift > 0 {
        let away: SIMD3<Float> = up.crystal.rotation.transpose * up.crystal.translation
        for (m, v) in worstOverReport(box: SIMD3(-4, -1.5, -4) - away, SIMD3(4, 2, 4) - away, step: 0.02,
                                      inset: true, frame: up) {
            all[m] = max(all[m] ?? 0, v)
        }
        for (m, v) in worstOverReport(box: SIMD3(-4, -0.5, -8), SIMD3(8, 12, 4), step: 0.01, inset: true, frame: up) {
            all[m] = max(all[m] ?? 0, v)
        }
    }
    if frames.count > 37 {
        let f: FrameState = frames[37]
        var pts: [SIMD3<Float>] = []
        var rng = SystemRandomNumberGenerator()
        for v in f.volatiles where v.drawn > 0.01 {
            for _ in 0..<20_000 {
                pts.append(v.position + SIMD3<Float>(Float.random(in: -0.4...0.4, using: &rng), Float.random(in: -0.4...0.4, using: &rng),
                                                     Float.random(in: -0.4...0.4, using: &rng)))
            }
        }
        for (m, v) in worstOverReport(box: .zero, .zero, step: 0.004, inset: true, points: pts, frame: f) { all[m] = max(all[m] ?? 0, v) }
    }
    print(String(format: "        worst over-report: grain %.2f, cuticle %.2f, taste hair %.2f, smell hair %.2f, odour %.2f",
                 all[1] ?? 0, all[2] ?? 0, all[3] ?? 0, all[4] ?? 0, all[6] ?? 0))
    expect(all.count >= 5, "not every material was sampled: \(all)")
    expect(all[5] == nil, "a film was found in the inset")
    for (m, v) in all { expect(v * stepScale <= 1.0, "material \(m) oversteps: \(v)") }
}

section("the picture")

let render: AntImage? = {
    guard let d = gpu.device, let s = scene else { return nil }
    guard let f = contactFrame else { return nil }
    return try? renderAnt(width: 960, height: 540, samples: 1, scene: s, frame: f, on: d).image
}()

test("the touch is visible: antenna and grain both within 3 px of the projected contact") {
    guard let img = render else { expect(false, "render failed"); return }
    let p: SIMD2<Float> = projectMain(contactPoint().point, width: img.width, height: img.height)
    var sawAnt = false
    var sawGrain = false
    for dy in -3...3 {
        for dx in -3...3 {
            let v: SIMD4<Float> = img.seen(Int(p.x) + dx, Int(p.y) + dy)
            if v.x == 1 && v.y == 2 { sawAnt = true }
            if v.x == 1 && v.y == 3 { sawGrain = true }
        }
    }
    expect(sawAnt && sawGrain, "near the contact: antenna \(sawAnt), grain \(sawGrain)")
}

test("the inset shows grain, both hairs and odour dots, and no film; the taste inset shows only C, O and H, the odour inset C, O and H") {
    guard let img = render else { expect(false, "render failed"); return }
    var seen: Set<Int> = []
    var taste: Set<Int> = []
    var smell: Set<Int> = []
    for y in 0..<img.height {
        for x in 0..<img.width {
            let v: SIMD4<Float> = img.seen(x, y)
            if v.x == 2 { seen.insert(Int(v.y)) }
            if v.x == 3 && v.y >= 1 && v.y <= 5 { taste.insert(Int(v.y)) }
            if v.x == 4 && v.y >= 1 && v.y <= 5 { smell.insert(Int(v.y)) }
        }
    }
    expect(seen.isSuperset(of: [1, 2, 3, 4, 6]), "inset materials seen: \(seen.sorted())")
    expect(!seen.contains(5), "the inset shows a film")
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
    for f in [perTap / 5, perTap / 2, perTap * 7 / 10, perTap * 9 / 10, perTap + 3, 3 * perTap + 11] {
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

test("lifted, the tip visibly draws back in the main view (about 10 px), and the inset shows no grain at all") {
    guard let d = gpu.device, let s = scene, let up = frames.max(by: { $0.lift < $1.lift }) else { expect(false); return }
    guard let img = try? renderAnt(width: 1920, height: 1080, samples: 1, scene: s, frame: up, on: d).image else {
        expect(false, "render failed"); return
    }
    let a: Antenna = up.ant.antennae[0]
    let low: SIMD2<Float> = projectMain(a.tipCentre - s.contact.normal * a.tipRadius, width: 1920, height: 1080)
    let touch: SIMD2<Float> = projectMain(s.contact.point, width: 1920, height: 1080)
    print(String(format: "        tip moved %.1f px on screen (the main view is %.0f mm wide)", simd_distance(low, touch), mainViewWidth))
    expect(simd_distance(low, touch) > 9.5, "the lift is hard to see: \(simd_distance(low, touch)) px")
    var grain: Int = 0
    let c = SIMD2<Float>(insetCentre.x * 1080, insetCentre.y * 1080)
    let r: Float = insetRadius * 1080
    for y in Int(c.y - r)..<Int(c.y + r) {
        for x in Int(c.x - r)..<Int(c.x + r) where img.seen(x, y).x == 2 {
            let m: Int = Int(img.seen(x, y).y)
            if m == 1 || m == 5 { grain += 1 }
        }
    }
    expectEqual(grain, 0)
}

finish()
