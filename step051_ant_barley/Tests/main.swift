// Tests for step 51: one grain of cooked pearled barley, tapped. Step 43's
// tests, copied, with the food's own swapped in: the grain's size against
// Wang et al. and USDA, its film, the exact distance to it, and the β-glucan
// piece built from PubChem conformers. Then the motion as step 30 tests it —
// the touch at every touch-down, no penetration at any frame, a clear lift, a
// forward loop — and the odour: one hexanal into a pore per tap, dots that
// never pop, a rigid odorant, and odour at every frame.
//
// ANT_MUTANT=segments13|hover|tastePores|noOdour|grainSize|press|rewind breaks the scene
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

test("the grain is a cooked pearled barley grain's size: Wang et al.'s half-axes × 1.4, as drawn — about 8 × 4.5 × 3 mm") {
    // Wang et al. 2018: a = 2.89, b = 1.59, c = 1.06 mm (half-axes).
    expect(abs(rawHalfAxes.x - 2.89) < 1e-5 && abs(rawHalfAxes.y - 1.59) < 1e-5 && abs(rawHalfAxes.z - 1.06) < 1e-5)
    // USDA: water 10.09 → 68.8 g/100 g, so mass × 2.88; its cube root is
    // the swelling, and Wang's 2.62 volume ratio agrees within a few percent.
    let fromMass: Float = pow(massGain, 1.0 / 3.0)
    let fromVolume: Float = pow(2.62, 1.0 / 3.0)
    print(String(format: "        mass gain %.2f× → linear %.3f×; Wang's 2.62 volume → %.3f×; used %.2f×", massGain, fromMass, fromVolume, linearSwelling))
    expect(abs(massGain - 2.88) < 0.01, "mass gain \(massGain)")
    expect(abs(linearSwelling - fromMass) < 0.03 && abs(linearSwelling - fromVolume) < 0.03)
    guard let s = scene else { expect(false); return }
    let g: Grain = s.grain
    let size: SIMD3<Float> = 2 * g.axes
    print(String(format: "        drawn: %.2f × %.2f × %.2f mm (long × wide × thick; drawn half-axes are x, z, y)", size.x, size.z, size.y))
    // The grain's axes are (long, up, across): up is Wang's c (thickness),
    // across is b (width).
    let want = SIMD3<Float>(2.89 * 1.4, 1.06 * 1.4, 1.59 * 1.4)
    expect(simd_length(g.axes - want) < 1e-4, "drawn half-axes \(g.axes), want \(want)")
    expect(size.x > 7.5 && size.x < 8.5 && size.z > 4.0 && size.z < 5.0 && size.y > 2.6 && size.y < 3.3, "not ~8 × 4.5 × 3 mm")
}

test("true scale: the grain is about twice the ant's length and three times its height") {
    guard let s = scene, let f = contactFrame else { expect(false); return }
    let g: Grain = s.grain
    let antTop: Float = f.ant.body.map { $0.extent(along: SIMD3(0, 1, 0)).hi }.max() ?? 0
    let ratio: Float = 2 * g.axes.x / f.ant.length
    print(String(format: "        grain %.1f mm long against the ant's %.2f mm (%.2f×); %.1f mm high against its %.2f mm", 2 * g.axes.x, f.ant.length, ratio, 2 * g.axes.y, antTop))
    expect(ratio > 1.7 && ratio < 3.3, "the grain is \(ratio) ant lengths")
    expect(2 * g.axes.y > 2.5 * antTop, "the grain is only \(2 * g.axes.y / antTop)× the ant's height")
}

test("it lies on the card on its flattest face, long axis level, wet all over with a thin film the ant does not see") {
    guard let s = scene else { expect(false); return }
    let g: Grain = s.grain
    expect(abs(g.centre.y - g.axes.y) < 1e-5, "the grain does not rest on the card")
    expect(g.axes.y < g.axes.z && g.axes.z < g.axes.x, "not lying on its flattest face")
    expect(abs(g.long.y) < 1e-6)
    expect(bareGrainSDF(SIMD3<Float>(g.centre.x, 0.0005, g.centre.z), g) < 0, "the grain is not down on the card")
    // The film: its thickness everywhere, measured along the normal.
    expect(filmThicknessMM > 0 && filmThicknessMM < 0.02, "film \(filmThicknessMM) mm")
    var rng = SystemRandomNumberGenerator()
    var worst: Float = 0
    for _ in 0..<2000 {
        let d: SIMD3<Float> = simd_normalize(SIMD3<Float>(Float.random(in: -1...1, using: &rng), Float.random(in: 0.1...1, using: &rng),
                                                          Float.random(in: -1...1, using: &rng)))
        let q: SIMD3<Float> = d * g.axes                      // not on the surface; project it
        let onSurface: SIMD3<Float> = q / simd_length(q / g.axes)
        let n: SIMD3<Float> = simd_normalize(g.worldDirection(onSurface / (g.axes * g.axes)))
        let p: SIMD3<Float> = g.world(onSurface) + n * g.film
        worst = max(worst, abs(grainSDF(p, g)), abs(bareGrainSDF(p, g) - g.film))
    }
    print(String(format: "        film %.0f µm; 2000 points one film-thickness out along the normal read %.1e mm off the film surface", g.film * 1000, worst))
    expect(worst < 2e-5, "the film is not a constant offset: \(worst)")
    // In the main view 5 µm is under a pixel.
    expect(filmThicknessMM / mainMillimetresPerPixel(width: 1920) < 1, "the film would show as a layer")
}

test("the exact ellipsoid distance agrees with a brute-force search of the surface, on the CPU and on the GPU") {
    guard let s = scene else { expect(false); return }
    let g: Grain = s.grain
    // A dense surface net, then the nearest net point refined.
    var net: [SIMD3<Float>] = []
    let nu: Int = 400, nv: Int = 200
    for i in 0..<nu {
        for j in 0...nv {
            let u: Float = 2 * Float.pi * Float(i) / Float(nu)
            let v: Float = Float.pi * Float(j) / Float(nv)
            net.append(SIMD3<Float>(g.axes.x * sin(v) * cos(u), g.axes.y * cos(v), g.axes.z * sin(v) * sin(u)))
        }
    }
    var rng = SystemRandomNumberGenerator()
    var pts: [SIMD3<Float>] = []
    var worst: Float = 0
    for k in 0..<300 {
        let dir: SIMD3<Float> = simd_normalize(SIMD3<Float>(Float.random(in: -1...1, using: &rng), Float.random(in: -1...1, using: &rng),
                                                            Float.random(in: -1...1, using: &rng)))
        let far: Float = k < 150 ? Float.random(in: 0.001...0.3, using: &rng) : Float.random(in: 0.3...8, using: &rng)
        let q: SIMD3<Float> = dir * g.axes / simd_length(dir) * (1 + far / 4)   // outside
        let exact: Float = ellipsoidDistance(q, g.axes)
        var best: Float = 1e9
        for x in net { best = min(best, simd_distance(x, q)) }
        // The net's spacing is ≤ 0.07 mm: it can only over-read, by at most
        // half a spacing squared over the distance, or the spacing itself.
        expect(exact <= best + 1e-4, "exact \(exact) exceeds the net's \(best) at \(q)")
        worst = max(worst, best - exact)
        let pw: SIMD3<Float> = g.world(q)
        if pw.y > 0.01 { pts.append(pw) }
        if best - exact > 0.07 { expect(false, "exact \(exact), net \(best)") }
    }
    guard let r = probe(pts) else { expect(false, "probe failed"); return }
    var gpuWorst: Float = 0
    for (i, p) in pts.enumerated() { gpuWorst = max(gpuWorst, abs(r[i].food - grainSDF(p, g))) }
    print(String(format: "        300 points: the net reads at most %.4f mm more than the exact distance; GPU vs CPU %.1e mm", worst, gpuWorst))
    expect(gpuWorst < 2e-4, "the kernel's grain distance differs from Food.swift's by \(gpuWorst)")
}

test("its colour: pale cream; and it smells, of hexanal") {
    let c: SIMD3<Float> = grainAlbedo
    expect(c.x >= c.y && c.y >= c.z && c.z > 0.6 * c.x && c.x > 0.35, "grain colour \(c) is not pale cream")
    expect(objectHasOdour)
    expect(!fibreIsATaste)
}

section("the touch")

test("the right antenna's tip touches the film: distance 0 within 2 µm, 5 µm above the grain, and no penetration") {
    guard let s = scene, let f = contactFrame else { expect(false); return }
    let a: Antenna = f.ant.antennae[0]
    let (point, normal) = contactPoint()
    expect(abs(grainSDF(point, s.grain)) < 2e-5, "contact point is off the film: \(grainSDF(point, s.grain))")
    expect(abs(bareGrainSDF(point, s.grain) - filmThicknessMM) < 2e-5, "the film is not \(filmThicknessMM) mm thick at the contact")
    expect(point.y > 0.05, "contact too low: \(point.y)")
    expect(normal.x < -0.5, "the contact does not face the ant")
    let lowest: SIMD3<Float> = a.tipCentre - normal * a.tipRadius
    guard let r = probe([lowest]) else { expect(false, "probe failed"); return }
    print(String(format: "        tip to film %.4f mm, tip to its own surface %.4f mm, tip to the grain beneath %.4f mm", r[0].food, r[0].ant,
                 bareGrainSDF(lowest, s.grain)))
    expect(abs(r[0].food) < 0.002, "the tip's nearest point is \(r[0].food) mm from the film")
    expect(abs(r[0].ant) < 0.002, "that point is not on the antenna: \(r[0].ant)")
    var pts: [SIMD3<Float>] = []
    var rng = SystemRandomNumberGenerator()
    for _ in 0..<60_000 {
        pts.append(point + SIMD3<Float>(Float.random(in: -0.15...0.15, using: &rng),
                                        Float.random(in: -0.15...0.25, using: &rng),
                                        Float.random(in: -0.15...0.15, using: &rng)))
    }
    guard let q = probe(pts) else { expect(false, "probe failed"); return }
    var worst: Float = 0
    for p in q where p.ant < 0 { worst = min(worst, p.food) }
    expect(worst > -0.002, "antenna reaches \(-worst) mm into the film")
}

test("no part of the ant is inside the grain or its film") {
    guard let s = scene, let f = contactFrame else { expect(false); return }
    var worst: Float = 9
    for shape in f.ant.shapes where shape.kind == .roundCone {
        for k in 0...20 {
            let t: Float = Float(k) / 20
            let c: SIMD3<Float> = shape.a + (shape.b - shape.a) * t
            worst = min(worst, grainSDF(c, s.grain) - (shape.ra + (shape.rb - shape.ra) * t))
        }
    }
    print(String(format: "        nearest ant surface to the film: %.4f mm", worst))
    expect(worst > -0.001, "an ant part reaches into the grain: \(worst)")
}

test("in the inset the taste hair's apex just meets the film's surface, 5 µm above the grain — the meniscus climbs, the hair does not dip") {
    guard let s = scene else { expect(false); return }
    let h: Sensillum = s.hairs[0]
    let R: Float = filmSurfaceRadiusMicrometres
    func film(_ p: SIMD3<Float>) -> Float { simd_length(p - SIMD3<Float>(0, -R, 0)) - R }
    var lowest: Float = 1e9
    var rng = SystemRandomNumberGenerator()
    for _ in 0..<20_000 {
        let d: SIMD3<Float> = simd_normalize(SIMD3<Float>(Float.random(in: -1...1, using: &rng), Float.random(in: -1...1, using: &rng),
                                                          Float.random(in: -1...1, using: &rng)))
        let t: Float = Float.random(in: 0...1, using: &rng)
        let p: SIMD3<Float> = h.base + (h.tip - h.base) * t + d * (h.baseRadius + (h.tipRadius - h.baseRadius) * t)
        if roundConeDistance(p, h.base, h.tip, h.baseRadius, h.tipRadius) <= 1e-4 { lowest = min(lowest, film(p)) }
    }
    lowest = min(lowest, film(h.tip - SIMD3<Float>(0, h.tipRadius, 0)))
    print(String(format: "        hair's deepest point relative to the film surface: %.4f µm; the grain lies %.1f µm below; film curve radius %.2f mm",
                 lowest, filmThickness, R / 1000))
    expect(lowest > -0.005, "the hair dips \(-lowest) µm into the film")
    expect(lowest < 0.02, "the hair hovers \(lowest) µm above the film")
    expect(abs(filmThickness - 1000 * filmThicknessMM) < 1e-4, "the inset's film is not the main view's")
    expect(meniscusHeight > 0 && meniscusHeight < filmThickness, "the meniscus")
    expect(R > 500, "the film curves too tightly for the inset: \(R) µm")
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

/// The odour dots drawn at one frame: those more than half grown.
func shownDots(_ f: FrameState) -> [Volatile] { f.volatiles.filter { $0.drawn > 0.5 } }

test("cooked barley gives the smell hair something at every frame: odour molecules, each at or heading for a wall pore") {
    guard let s = scene else { expect(false); return }
    let smell: Sensillum = s.hairs[1]
    let taste: Sensillum = s.hairs[0]
    expect(objectHasOdour)
    expectEqual(s.odourPaths.count, odourSlots)
    var fewest: Int = 99
    var worstWall: Float = 9
    var worstTaste: Float = 9
    var worstSauce: Float = 9
    for f in frames {
        fewest = min(fewest, shownDots(f).count)
        for v in f.volatiles where v.drawn > 0.01 {
            expect(v.pore.row >= 0 && v.pore.row < smell.poreRows && v.pore.column >= 0 && v.pore.column < smell.poreColumns,
                   "pore \(v.pore) is not on the lattice")
            let (mouth, outward) = smell.wallPore(row: v.pore.row, column: v.pore.column)
            // Outside the hair, on the outer side of its pore.
            expect(simd_dot(v.position - mouth, outward) >= 0, "an odour molecule inside the hair")
            // Its drawn surface never enters the smell hair's wall or the
            // taste hair, and never the film, wherever the film is now.
            let rr: Float = v.radius
            worstWall = min(worstWall, roundConeDistance(v.position, smell.base, smell.tip, smell.baseRadius, smell.tipRadius) - rr)
            worstTaste = min(worstTaste, roundConeDistance(v.position, taste.base, taste.tip, taste.baseRadius, taste.tipRadius) - rr)
            worstSauce = min(worstSauce, f.crystal.toCrystal(v.position).y - rr)
        }
    }
    print(String(format: "        %d frames: at least %d dots shown; nearest dot surface to the smell hair's wall %.3f µm, taste hair %.2f µm, film %.2f µm",
                 frames.count, fewest, worstWall, worstTaste, worstSauce))
    expect(fewest >= 3, "a frame shows only \(fewest) odour molecules")
    // A dot resting at its pore sits 0.03 µm off the wall, and shrinking in
    // keeps that gap in proportion: never into the cuticle (2 nm allowed for
    // the pore lattice's rounding). Step 42's slanted approach failed this.
    expect(worstWall > -0.002, "a dot sinks \(-worstWall) µm into the smell hair's wall")
    expect(worstTaste > 0, "a dot touches the taste hair")
    expect(worstSauce > 0, "a dot is in the film")
    // At the touch-down, as in step 42, one rests at its pore.
    guard let c = contactFrame else { expect(false); return }
    var arriving: Int = 0
    for v in c.volatiles {
        let (mouth, _) = smell.wallPore(row: v.pore.row, column: v.pore.column)
        if simd_distance(v.position, mouth) < 2 * volatileDotRadius + smell.wallPoreRadius && v.drawn > 0.9 { arriving += 1 }
    }
    expect(arriving >= 1, "no odour molecule at a pore at touch-down")
    // Each pore a dot heads for is really carved: the drawn distance at the
    // pore mouth, just inside the wall, says "outside the hair".
    for p in s.odourPaths {
        guard let r = probe([p.mouth - p.outward * 0.02]) else { expect(false, "probe failed"); return }
        expect(r[0].smellHair > 0, "no pore at the mouth a molecule is heading for: \(r[0].smellHair)")
    }
}

section("the molecules")

/// Rings of exactly six atoms, from the bond graph: each C1–C5 + O5 ring.
func sixRings(_ m: Molecule) -> [[Int]] {
    var nb: [[Int]] = Array(repeating: [], count: m.atoms.count)
    for (a, b) in m.bonds { nb[a].append(b); nb[b].append(a) }
    var found: Set<[Int]> = []
    func walk(_ path: [Int]) {
        if path.count == 6 {
            if nb[path[5]].contains(path[0]) { found.insert(path.sorted()) }
            return
        }
        for k in nb[path[path.count - 1]] where !path.contains(k) && k > path[0] && m.atoms[k].element != "H" { walk(path + [k]) }
    }
    for i in 0..<m.atoms.count where m.atoms[i].element != "H" { walk([i]) }
    return Array(found)
}

/// Every glucose residue named: ring O5, C1 … C5, C6, from the graph alone.
struct Residue { var o5: Int; var c: [Int] }   // c[0] = C1 … c[5] = C6

func residues(_ m: Molecule) -> [Residue] {
    var nb: [[Int]] = Array(repeating: [], count: m.atoms.count)
    for (a, b) in m.bonds { nb[a].append(b); nb[b].append(a) }
    func el(_ i: Int) -> String { m.atoms[i].element }
    var out: [Residue] = []
    for ring in sixRings(m) {
        guard let o5 = ring.first(where: { el($0) == "O" }), ring.filter({ el($0) == "O" }).count == 1 else { continue }
        let cs: [Int] = nb[o5].filter { ring.contains($0) }
        guard cs.count == 2 else { continue }
        // C5 carries the exocyclic CH2 (C6); C1 is the other.
        func exoC(_ c: Int) -> Int? { nb[c].first { el($0) == "C" && !ring.contains($0) } }
        let c5: Int = exoC(cs[0]) != nil ? cs[0] : cs[1]
        let c1: Int = c5 == cs[0] ? cs[1] : cs[0]
        guard let c6 = exoC(c5) else { continue }
        var chain: [Int] = [c1]
        while chain.count < 5 {
            guard let next = nb[chain[chain.count - 1]].first(where: { ring.contains($0) && el($0) == "C" && !chain.contains($0) }) else { break }
            chain.append(next)
        }
        guard chain.count == 5, chain[4] == c5 else { continue }
        out.append(Residue(o5: o5, c: chain + [c6]))
    }
    return out
}

/// The glycosidic links: (donor residue, acceptor residue, acceptor carbon 3 or 4, bridging O).
func glycosidicLinks(_ m: Molecule, _ rs: [Residue]) -> [(donor: Int, acceptor: Int, position: Int, oxygen: Int)] {
    var nb: [[Int]] = Array(repeating: [], count: m.atoms.count)
    for (a, b) in m.bonds { nb[a].append(b); nb[b].append(a) }
    var out: [(Int, Int, Int, Int)] = []
    for (i, r) in rs.enumerated() {
        let c1: Int = r.c[0]
        for o in nb[c1] where m.atoms[o].element == "O" && o != r.o5 {
            for c in nb[o] where c != c1 {
                for (j, s) in rs.enumerated() where j != i {
                    if let pos = s.c.firstIndex(of: c) { out.append((i, j, pos + 1, o)) }
                }
            }
        }
    }
    return out.map { (donor: $0.0, acceptor: $0.1, position: $0.2, oxygen: $0.3) }
}

/// β or α at a residue's C1: β-D-glucose has its anomeric O on the same face
/// of the ring as C6 (Haworth: both up).
func isBeta(_ m: Molecule, _ r: Residue) -> Bool? {
    var nb: [[Int]] = Array(repeating: [], count: m.atoms.count)
    for (a, b) in m.bonds { nb[a].append(b); nb[b].append(a) }
    let ring: [Int] = Array(r.c.prefix(5)) + [r.o5]
    let p: [SIMD3<Float>] = ring.map { m.atoms[$0].position }
    var centre = SIMD3<Float>(0, 0, 0)
    for q in p { centre += q }
    centre /= 6
    var normal = SIMD3<Float>(0, 0, 0)
    for k in 0..<6 { normal += simd_cross(p[k] - centre, p[(k + 1) % 6] - centre) }
    guard let o1 = nb[r.c[0]].first(where: { m.atoms[$0].element == "O" && $0 != r.o5 }) else { return nil }
    let a: Float = simd_dot(m.atoms[o1].position - m.atoms[r.c[0]].position, normal)
    let b: Float = simd_dot(m.atoms[r.c[5]].position - m.atoms[r.c[4]].position, normal)
    return a * b > 0
}

func angleAt(_ m: Molecule, _ a: Int, _ o: Int, _ b: Int) -> Float {
    let u: SIMD3<Float> = simd_normalize(m.atoms[a].position - m.atoms[o].position)
    let v: SIMD3<Float> = simd_normalize(m.atoms[b].position - m.atoms[o].position)
    return deg(acos(min(max(simd_dot(u, v), -1), 1)))
}

let templates: [Molecule] = ["cellobiose_beta_CID10712_3d.sdf", "laminaribiose_beta_CID5287770_3d.sdf"].compactMap {
    try? loadMolfile("Resources/" + $0)
}

test("the β-glucan piece is C₄₂H₇₂O₃₆: seven glucose rings, every atom with its valence") {
    guard let s = scene else { expect(false); return }
    let m: Molecule = s.chemistry.glucan
    let f: [String: Int] = m.formula
    print("        β-glucan piece: \(f)")
    expect(f["C"] == 42 && f["H"] == 72 && f["O"] == 36 && f.count == 3, "the piece is \(f)")
    expectEqual(m.atoms.count, 150)
    expectEqual(m.rings, 7)
    expectEqual(residues(m).count, 7)
    let v: [Int] = m.valences
    for (i, a) in m.atoms.enumerated() {
        let want: Int = a.element == "C" ? 4 : (a.element == "O" ? 2 : 1)
        expect(v[i] == want, "\(a.element) \(i) has valence \(v[i])")
    }
    expect(m.orders.allSatisfy { $0 == 1 }, "a sugar has no double bonds")
    for a in m.atoms { expectEqual(a.view, 0) }
}

test("its links: a cellotriosyl and a cellotetraosyl run joined by ONE β-(1→3) link — five β-(1→4), all β") {
    guard let s = scene else { expect(false); return }
    let m: Molecule = s.chemistry.glucan
    let rs: [Residue] = residues(m)
    let links = glycosidicLinks(m, rs)
    expectEqual(links.count, 6)
    expectEqual(links.filter { $0.position == 4 }.count, 5)
    expectEqual(links.filter { $0.position == 3 }.count, 1)
    // An unbranched chain: every residue donates at most once and accepts at most once.
    for i in 0..<rs.count {
        expect(links.filter { $0.donor == i }.count <= 1 && links.filter { $0.acceptor == i }.count <= 1, "residue \(i) branches")
    }
    // Walk it from the non-reducing end: the runs of 1→4 links either side
    // of the 1→3 one are 2 and 3 links long — three and four glucoses.
    guard let start = (0..<rs.count).first(where: { i in !links.contains { $0.acceptor == i } }) else { expect(false, "no end"); return }
    var order: [Int] = []
    var at: Int = start
    while let l = links.first(where: { $0.donor == at }) { order.append(l.position); at = l.acceptor }
    print("        from the non-reducing end: \(order.map { "β1→\($0)" }.joined(separator: ", "))")
    expectEqual(order, [4, 4, 3, 4, 4, 4])
    // Every anomeric centre β, the reducing end's too.
    var betas: Int = 0
    for r in rs where isBeta(m, r) == true { betas += 1 }
    expectEqual(betas, 7)
    // And the test can tell: in the templates, each residue is β as well.
    for t in templates { for r in residues(t) { expect(isBeta(t, r) == true, "a template residue reads α") } }
}

test("its geometry is PubChem's: bond lengths and link angles within the templates' own, no atoms crowding") {
    guard let s = scene else { expect(false); return }
    expectEqual(templates.count, 2)
    let m: Molecule = s.chemistry.glucan
    func key(_ x: Molecule, _ b: (Int, Int)) -> String { [x.atoms[b.0].element, x.atoms[b.1].element].sorted().joined() }
    var range: [String: (Float, Float)] = [:]
    var linkAngles: [Float] = []
    var closest: Float = 9
    func nonBondedClosest(_ x: Molecule) -> Float {
        var nb: [Set<Int>] = Array(repeating: [], count: x.atoms.count)
        for (a, b) in x.bonds { nb[a].insert(b); nb[b].insert(a) }
        var best: Float = 9
        for i in 0..<x.atoms.count {
            for j in (i + 1)..<x.atoms.count where !nb[i].contains(j) && nb[i].isDisjoint(with: nb[j]) {
                best = min(best, simd_distance(x.atoms[i].position, x.atoms[j].position))
            }
        }
        return best
    }
    for t in templates {
        for b in t.bonds {
            let d: Float = simd_distance(t.atoms[b.0].position, t.atoms[b.1].position)
            let k: String = key(t, b)
            let r: (Float, Float) = range[k] ?? (9, 0)
            range[k] = (min(r.0, d), max(r.1, d))
        }
        let rs: [Residue] = residues(t)
        for l in glycosidicLinks(t, rs) { linkAngles.append(angleAt(t, rs[l.donor].c[0], l.oxygen, rs[l.acceptor].c[l.position - 1])) }
        closest = min(closest, nonBondedClosest(t))
    }
    var worstBond: Float = 0
    for b in m.bonds {
        let d: Float = simd_distance(m.atoms[b.0].position, m.atoms[b.1].position)
        guard let r = range[key(m, b)] else { expect(false, "a \(key(m, b)) bond the templates lack"); continue }
        worstBond = max(worstBond, r.0 - d, d - r.1)
    }
    let rs: [Residue] = residues(m)
    var worstAngle: Float = 0
    for l in glycosidicLinks(m, rs) {
        let a: Float = angleAt(m, rs[l.donor].c[0], l.oxygen, rs[l.acceptor].c[l.position - 1])
        worstAngle = max(worstAngle, linkAngles.map { abs($0 - a) }.min() ?? 99)
    }
    let crowd: Float = nonBondedClosest(m)
    print(String(format: "        bonds outside the templates' ranges by at most %.4f Å; link angles within %.2f° of a template's; closest non-bonded pair %.2f Å (templates %.2f Å)",
                 max(worstBond, 0), worstAngle, crowd, closest))
    expect(worstBond < 0.005, "a bond is \(worstBond) Å outside the templates' range")
    expect(worstAngle < 0.5, "a glycosidic angle differs from the templates' by \(worstAngle)°")
    expect(crowd > closest - 0.15, "two atoms crowd to \(crowd) Å")
}

test("hexanal is C₆H₁₂O with one C=O, every atom with its valence; the β-glucan fits its circle") {
    guard let s = scene else { expect(false); return }
    let o: Molecule = s.chemistry.odorant
    let f: [String: Int] = o.formula
    expect(f["C"] == 6 && f["H"] == 12 && f["O"] == 1, "hexanal is \(f)")
    expectEqual(o.rings, 0)
    expectEqual(o.bonds.indices.filter { o.orders[$0] == 2 }.count, 1)
    for a in o.atoms { expectEqual(a.view, 1) }
    let v: [Int] = o.valences
    for (i, a) in o.atoms.enumerated() {
        let want: Int = a.element == "C" ? 4 : (a.element == "O" ? 2 : 1)
        expect(v[i] == want, "\(a.element) \(i) has valence \(v[i])")
    }
    var out: Float = -9
    for a in s.chemistry.glucan.atoms {
        out = max(out, simd_length(SIMD2<Float>(a.position.x, a.position.y)) + 0.4 - moleculeField / 2)
    }
    print(String(format: "        the β-glucan's reach past its circle %.2f Å", out))
    expect(out < 0, "the β-glucan leaves its circle by \(out) Å")
}

section("the tap (step 30's, on the wet grain)")

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

/// The film's distance from the right antenna at one frame, on the GPU: the
/// tip's nearest point (towards the contact, along the film's normal), and
/// the deepest any antenna point reaches into the film among points packed
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

test("at every touch-down the tip meets the film: 0 within 2 µm, on the GPU") {
    let downs: [FrameState] = frames.filter { $0.lift == 0 }
    expect(downs.count >= tapsPerLoop, "only \(downs.count) touching frames")
    var worst: Float = 0
    for f in downs {
        guard let r = tipSurvey(f, points: 0) else { expect(false, "probe failed"); return }
        worst = max(worst, abs(r.lowest))
    }
    print(String(format: "        %d touching frames; worst tip-to-film %.4f mm", downs.count, worst))
    expect(worst < 0.002, "a touching tip is \(worst) mm from the film")
}

test("at no frame does any part of the antenna go into the film") {
    var worst: Float = 1
    for f in frames {
        guard let r = tipSurvey(f, points: 12_000) else { expect(false, "probe failed"); return }
        worst = min(worst, r.deepest)
    }
    print(String(format: "        %d frames × 12,000 points: nearest antenna point to the inside of the film %.4f mm", frames.count, worst))
    expect(worst > -0.002, "the antenna reaches \(-worst) mm into the film")
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
    print(String(format: "        %d frames: nearest ant surface to the film %.4f mm", frames.count, worst))
    // The grain's distance is exact outside; allow float noise and the cone sampling.
    expect(worst > -0.003, "an ant part reaches into the grain: \(worst)")
}

test("lifted, the tip clears the film — by the lift, and never less than 0.03 mm near the top") {
    let ups: [FrameState] = frames.filter { $0.lift > 0.5 * liftHeight }
    expect(!ups.isEmpty, "no lifted frames")
    var least: Float = 1
    for f in ups {
        guard let r = tipSurvey(f, points: 4000) else { expect(false, "probe failed"); return }
        least = min(least, r.lowest)
        // The film's distance is exact outside (Food.swift); allow 2 µm, and
        // the step-43 margin of 4% for the tip's own curve.
        expect(abs(r.lowest - f.lift) < 0.002 + 0.04 * f.lift, "lift \(f.lift) mm but the tip is \(r.lowest) mm off")
        expect(r.deepest > 0, "a lifted antenna point is in the film")
    }
    print(String(format: "        %d lifted frames, least clearance %.3f mm (lift %.3f mm at the top)", ups.count, least, liftHeight))
    expect(least > 0.03)
}

test("the taste hair meets the film at touch-down, and leaves it when lifted — never dipping in (inset, GPU)") {
    guard let s = scene, let c = contactFrame else { expect(false); return }
    let h: Sensillum = s.hairs[0]
    let apexLow = SIMD3<Float>(h.tip.x, h.tip.y - h.tipRadius, h.tip.z)
    let beside: SIMD3<Float> = apexLow + SIMD3<Float>(0.7, 0.05, 0)
    expect(abs(c.crystal.toCrystal(apexLow).y) < 0.002, "at touch-down the apex is \(c.crystal.toCrystal(apexLow).y) µm off the film")
    guard let r0 = probe([beside], frame: c) else { expect(false, "probe failed"); return }
    expect(r0[0].insetMaterial == 5 && r0[0].insetDistance < 0, "beside the apex at touch-down: material \(r0[0].insetMaterial)")
    // The hair's surface, sampled, against the film as it is at each frame.
    var surface: [SIMD3<Float>] = [apexLow]
    var rng = SystemRandomNumberGenerator()
    while surface.count < 6000 {
        let d: SIMD3<Float> = simd_normalize(SIMD3<Float>(Float.random(in: -1...1, using: &rng), Float.random(in: -1...1, using: &rng),
                                                          Float.random(in: -1...1, using: &rng)))
        let t: Float = Float.random(in: 0.8...1, using: &rng)
        let p: SIMD3<Float> = h.base + (h.tip - h.base) * t + d * (h.baseRadius + (h.tipRadius - h.baseRadius) * t)
        if abs(roundConeDistance(p, h.base, h.tip, h.baseRadius, h.tipRadius)) < 1e-3 { surface.append(p) }
    }
    let R: Float = filmSurfaceRadiusMicrometres
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
        expect(y > meniscusHeight, "lifted \(f.lift) mm, the apex is only \(y) µm up")
        guard let r = probe([beside], frame: f) else { expect(false, "probe failed"); return }
        expect(r[0].insetMaterial != 5, "lifted, the meniscus is still at the apex")
    }
    print(String(format: "        hair's deepest point into the film over all frames %.4f µm; film drop vs lift, worst difference %.2f µm",
                 -deepest, abs(lowestLifted)))
    expect(deepest > -0.005, "the hair dips \(-deepest) µm into the film")
    expect(abs(lowestLifted) < 3, "the inset film and the main-view lift disagree by \(lowestLifted) µm")
}

section("the odour in motion")

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
    // Time L is time 0.
    let end: FrameState = s.frame(at: loopSeconds)
    expect(simd_distance(end.ant.antennae[0].tipCentre, c.ant.antennae[0].tipCentre) < 1e-5)
    expectEqual(end.molecule.atoms.count, c.molecule.atoms.count)
    var endWorst: Float = 0
    for (a, b) in zip(end.molecule.atoms, c.molecule.atoms) { endWorst = max(endWorst, simd_distance(a.position, b.position)) }
    for (a, b) in zip(end.volatiles, c.volatiles) { endWorst = max(endWorst, simd_distance(a.position, b.position), abs(a.drawn - b.drawn)) }
    expect(endWorst < 1e-3, "time L differs from time 0 by \(endWorst)")
    // The step from the last frame into the first is an ordinary step.
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
    // Forward: progress never falls, and a loop is exactly one tap's worth per tap.
    var last: Float = odourProgress(0, mutant: s.mutant)
    for i in 1...frames.count {
        let u: Float = odourProgress(frameTime(i, of: frames.count) - (i == frames.count ? 1e-4 : 0), mutant: s.mutant)
        expect(u >= last - 1e-5, "the odour goes back at frame \(i): \(last) → \(u)")
        last = u
    }
    expect(abs(last - Float(tapsPerLoop)) < 0.01, "the loop is \(last) taps of odour, not \(tapsPerLoop)")
    // Every shown dot only comes nearer its pore, frame to frame; and one
    // reaches its pore (life passes the arrival) in every tap, exactly one.
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
    expectEqual(arrivals, Array(repeating: 1, count: tapsPerLoop))
    // Each tap is the same: the lift is periodic in the tap.
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
            let u: Float = Float(odourSlots) * (Float(n) - p.offset)
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

test("every frame: the odorant moves rigidly, the taste inset not at all, and no molecule leaves its circle") {
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
        // A mirror image keeps every distance: rule it out by the handedness
        // of three atoms' frame.
        let det0: Float = simd_dot(simd_cross(o0[1].position - o0[0].position, o0[2].position - o0[0].position), o0[3].position - o0[0].position)
        let det1: Float = simd_dot(simd_cross(o[1].position - o[0].position, o[2].position - o[0].position), o[3].position - o[0].position)
        expect(det0 * det1 > 0, "the odorant is mirrored")
        let still: Molecule = merge([s.chemistry.glucan])
        for (a, b) in zip(f.molecule.atoms.prefix(still.atoms.count), still.atoms) {
            worstStill = max(worstStill, simd_distance(a.position, b.position))
        }
    }
    print(String(format: "        %d frames: odorant distances kept to %.1e Å, taste inset still to %.1e Å; odorant's reach past its circle %.2f Å",
                 frames.count, worstRigid, worstStill, worstOut))
    expect(worstRigid < 1e-3, "the odorant deforms by \(worstRigid) Å")
    expect(worstStill < 1e-6, "the taste inset moves by \(worstStill) Å")
    expect(worstOut < 0, "the odorant leaves its circle by \(worstOut) Å")
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
    // As step 30: round the tip and the film with the antenna at its highest.
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
}

test("inset: the hairs, the dome, the film and its meniscus are honest outside too") {
    let w: [Int: Float] = worstOverReport(box: SIMD3(-4, -0.5, -12), SIMD3(8, 12, 4), step: 0.02, inset: true)
    // A second pass concentrated round the tip, where the film and pore are.
    let t: [Int: Float] = worstOverReport(box: SIMD3(-1.5, -0.2, -1.5), SIMD3(1.5, 1.5, 1.5), step: 0.005, inset: true)
    // And a third in a thin shell round each hair's wall, down among the pores.
    let pores: [Int: Float] = worstOverReport(box: .zero, .zero, step: 0.004, inset: true, points: hairShellPoints(count: 80_000))
    var all: [Int: Float] = w
    for (m, v) in t { all[m] = max(all[m] ?? 0, v) }
    for (m, v) in pores { all[m] = max(all[m] ?? 0, v) }
    // As step 30: and with the antenna lifted, when the film is moved — round
    // the moved film, and round the hairs, dots and all.
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
    // And round the odour dots, where they are at a mid-loop frame.
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
    print(String(format: "        worst over-report: film %.2f, cuticle %.2f, taste hair %.2f, smell hair %.2f, meniscus %.2f, odour %.2f",
                 all[1] ?? 0, all[2] ?? 0, all[3] ?? 0, all[4] ?? 0, all[5] ?? 0, all[6] ?? 0))
    expect(all.count >= 5, "not every material was sampled: \(all)")
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
    expect(sawAnt && sawSugar, "near the contact: antenna \(sawAnt), grain \(sawSugar)")
}

test("the inset shows both hairs, the meniscus and odour dots; the taste inset shows C, O and H, the odour inset too") {
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
    for f in [perTap / 5, perTap / 2, perTap * 7 / 10, perTap * 9 / 10, perTap + 3, 3 * perTap + 11] {
        _ = try? r.render(frames[f], samples: 1)
        let full: [UInt8] = Array(UnsafeBufferPointer(start: r.pixels.contents().assumingMemoryBound(to: UInt8.self), count: n))
        for y in 0..<h {
            for x in 0..<w {
                let i: Int = (y * w + x) * 4
                if full[i] != base[i] || full[i + 1] != base[i + 1] || full[i + 2] != base[i + 2] {
                    if region.contains(x, y) { inside += 1 } else {
                        outside += 1; if ProcessInfo.processInfo.environment["ANT_DEBUG"] != nil && outside % 5 == 0 { print("        outside x \(x) y \(y) frame \(f)") }
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

test("lifted, the tip visibly draws back in the main view, and the inset shows no film or grain at all") {
    guard let d = gpu.device, let s = scene, let up = frames.max(by: { $0.lift < $1.lift }) else { expect(false); return }
    guard let img = try? renderAnt(width: 1920, height: 1080, samples: 1, scene: s, frame: up, on: d).image else {
        expect(false, "render failed"); return
    }
    let a: Antenna = up.ant.antennae[0]
    let low: SIMD2<Float> = projectMain(a.tipCentre - s.contact.normal * a.tipRadius, width: 1920, height: 1080)
    let touch: SIMD2<Float> = projectMain(s.contact.point, width: 1920, height: 1080)
    print(String(format: "        tip moved %.1f px on screen (0.07 mm in the main view)", simd_distance(low, touch)))
    expect(simd_distance(low, touch) > 4, "the lift is invisible: \(simd_distance(low, touch)) px")
    // Between the lifted tip and the film, on screen, is neither: the gap shows.
    let mid: SIMD2<Float> = projectMain(s.contact.point + s.contact.normal * (up.lift / 2), width: 1920, height: 1080)
    let seenMid: SIMD4<Float> = img.seen(Int(mid.x.rounded()), Int(mid.y.rounded()))
    print("        at the gap's middle the main view sees material \(Int(seenMid.y))")
    var film: Int = 0
    let c = SIMD2<Float>(insetCentre.x * 1080, insetCentre.y * 1080)
    let r: Float = insetRadius * 1080
    for y in Int(c.y - r)..<Int(c.y + r) {
        for x in Int(c.x - r)..<Int(c.x + r) where img.seen(x, y).x == 2 {
            let m: Int = Int(img.seen(x, y).y)
            if m == 1 || m == 5 { film += 1 }
        }
    }
    expectEqual(film, 0)
}

finish()
