// Tests for step 30: step 26's tests, copied (run on the contact frame, where
// the scene is step 26's), then the motion: the touch at every touch-down, no
// penetration at any frame, a clear lift, a forward loop, rigid molecules.
//
// Tests for step 26. The anatomy is checked against the literature it cites,
// the contact and the pores against the GPU's own distance functions — the
// same source that draws the picture — and the finished frame for what it
// shows where.
//
// ANT_MUTANT=segments13|hover|tastePores|press|rewind breaks the scene on purpose;
// `make mutants` requires the suite to fail for each.

import Foundation
import Metal
import simd

let mutant: Mutant = {
    switch ProcessInfo.processInfo.environment["ANT_MUTANT"] {
    case "segments13": return .segments13
    case "hover": return .hover
    case "tastePores": return .tastePores
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

let scene: Scene? = try? Scene(mutant: mutant, sucrosePath: "Resources/sucrose_6S1T.pdb")
let library: MTLLibrary? = {
    guard let d = gpu.device, let s = scene else { return nil }
    return try? makeLibrary(d, s)
}()

/// Every frame of the GIF's loop, and the first — the touch-down, step 26's pose.
let frames: [FrameState] = scene.map { s in (0..<defaultFrames).map { s.frame(at: frameTime($0, of: defaultFrames)) } } ?? []
let contactFrame: FrameState? = frames.first

func probe(_ pts: [SIMD3<Float>], frame given: FrameState? = nil) -> [Probe]? {
    guard let d = gpu.device, let s = scene, let l = library, let f = given ?? contactFrame else { return nil }
    return try? probeScene(pts, scene: s, frame: f, library: l, on: d)
}

func deg(_ r: Float) -> Float { r * 180 / Float.pi }

section("the ant, against the literature")

test("the scene loads") {
    expect(scene != nil, "could not build the scene (is Resources/sucrose_6S1T.pdb there?)")
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

section("the sugar")

test("the faces come from the real unit cell: (100) and (001) meet at 180° − β") {
    let a: SIMD3<Float> = reciprocalNormal(1, 0, 0)
    let c: SIMD3<Float> = reciprocalNormal(0, 0, 1)
    let angle: Float = deg(acos(simd_dot(a, c)))
    expect(abs(angle - (180 - 102.938)) < 0.01, "angle between the normals \(angle)°")
    // b is the monoclinic unique axis: (010) is square to both.
    let b: SIMD3<Float> = reciprocalNormal(0, 1, 0)
    expect(abs(simd_dot(a, b)) < 1e-5 && abs(simd_dot(c, b)) < 1e-5)
}

test("grains are granulated-sugar size: sieve (middle) dimension within 0.30–0.67 mm") {
    for (g, frame) in buildGrains() {
        let e: [Float] = g.extents(frame: frame).sorted()
        print(String(format: "        grain %.2f × %.2f × %.2f mm", e[0], e[1], e[2]))
        expect(grainSieveRange.contains(e[1]), "middle dimension \(e[1]) mm")
    }
}

test("every grain rests on the table, and none overlaps another") {
    let grains: [Grain] = buildGrains().map { $0.grain }
    for (i, g) in grains.enumerated() {
        let low: Float = g.vertices().map { $0.y }.min() ?? 1
        expect(abs(low) < 1e-4, "grain \(i) lowest corner at \(low) mm")
        for (j, h) in grains.enumerated() where j != i {
            for v in g.vertices() { expect(h.sdf(v) > 0, "grain \(i) corner inside grain \(j)") }
        }
    }
}

section("the touch")

test("the right antenna's tip touches the grain: distance 0 within 2 µm, and no penetration") {
    guard let s = scene, let f = contactFrame else { expect(false); return }
    let a: Antenna = f.ant.antennae[0]
    let (point, normal) = contactPoint(s.grains)
    // On the CPU: the contact point lies on the grain's top face.
    expect(abs(s.grains[0].sdf(point)) < 1e-4, "contact point is off the face: \(s.grains[0].sdf(point))")
    expect(normal.y > 0.999, "the top face is not level: \(normal)")
    // On the GPU: the lowest point of the tip is on the sugar.
    let lowest: SIMD3<Float> = a.tipCentre - normal * a.tipRadius
    guard let r = probe([lowest]) else { expect(false, "probe failed"); return }
    print(String(format: "        tip to sugar %.4f mm, tip to its own surface %.4f mm", r[0].tastedGrain, r[0].ant))
    expect(abs(r[0].tastedGrain) < 0.002, "the tip's lowest point is \(r[0].tastedGrain) mm from the sugar")
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
    for p in q where p.ant < 0 { worst = min(worst, p.tastedGrain) }
    expect(worst > -0.002, "antenna reaches \(-worst) mm into the sugar")
}

section("the tap (step 30)")

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
            // The scape starts at the socket; each segment starts where the last ended.
            worstGap = max(worstGap, simd_distance(a.segments[0].a, a.socket))
            for k in 1..<a.segments.count {
                worstGap = max(worstGap, simd_distance(a.segments[k].a, a.segments[k - 1].b))
            }
            // Segments are rigid: each one's length within 1% of its length at contact.
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

/// The tasted grain's distance from the right antenna at one frame, on the
/// GPU: the tip's lowest point, and the deepest any antenna point reaches
/// into the sugar among points packed round the tip.
func tipSurvey(_ f: FrameState, points n: Int) -> (lowest: Float, deepest: Float)? {
    guard let s = scene else { return nil }
    let a: Antenna = f.ant.antennae[0]
    let normal: SIMD3<Float> = s.contact.normal
    let lowest: SIMD3<Float> = a.tipCentre - normal * a.tipRadius
    var pts: [SIMD3<Float>] = [lowest]
    var rng = SystemRandomNumberGenerator()
    for _ in 0..<n {
        pts.append(a.tipCentre + SIMD3<Float>(Float.random(in: -0.16...0.16, using: &rng),
                                              Float.random(in: -0.12...0.16, using: &rng),
                                              Float.random(in: -0.16...0.16, using: &rng)))
    }
    guard let q = probe(pts, frame: f) else { return nil }
    var deepest: Float = 1
    for p in q.dropFirst() where p.ant < 0 { deepest = min(deepest, p.tastedGrain) }
    return (q[0].tastedGrain, deepest)
}

test("at every touch-down the tip meets the sugar: 0 within 2 µm, on the GPU") {
    let downs: [FrameState] = frames.filter { $0.lift == 0 }
    expect(downs.count >= tapsPerLoop, "only \(downs.count) touching frames")
    var worst: Float = 0
    for f in downs {
        guard let r = tipSurvey(f, points: 0) else { expect(false, "probe failed"); return }
        worst = max(worst, abs(r.lowest))
    }
    print(String(format: "        %d touching frames; worst tip-to-sugar %.4f mm", downs.count, worst))
    expect(worst < 0.002, "a touching tip is \(worst) mm from the sugar")
}

test("at no frame does any part of the antenna go into the sugar") {
    var worst: Float = 1
    for f in frames {
        guard let r = tipSurvey(f, points: 12_000) else { expect(false, "probe failed"); return }
        worst = min(worst, r.deepest)
    }
    print(String(format: "        %d frames × 12,000 points: nearest antenna point to the inside of the sugar %.4f mm", frames.count, worst))
    expect(worst > -0.002, "the antenna reaches \(-worst) mm into the sugar")
}

test("lifted, the tip clears the sugar — by the lift, and never less than 0.03 mm near the top") {
    let ups: [FrameState] = frames.filter { $0.lift > 0.5 * liftHeight }
    expect(!ups.isEmpty, "no lifted frames")
    var least: Float = 1
    for f in ups {
        guard let r = tipSurvey(f, points: 4000) else { expect(false, "probe failed"); return }
        least = min(least, r.lowest)
        expect(abs(r.lowest - f.lift) < 0.002, "lift \(f.lift) mm but the tip is \(r.lowest) mm up")
        expect(r.deepest > 0, "a lifted antenna point is in the sugar")
    }
    print(String(format: "        %d lifted frames, least clearance %.3f mm (lift %.3f mm at the top)", ups.count, least, liftHeight))
    expect(least > 0.03)
}

test("the taste hair meets the film at touch-down, and leaves it when lifted (inset, GPU)") {
    guard let s = scene, let c = contactFrame else { expect(false); return }
    let h: Sensillum = s.hairs[0]
    // The apex's lowest point, and a point just beside it in the film.
    let apexLow = SIMD3<Float>(h.tip.x, h.tip.y - h.tipRadius, h.tip.z)
    let beside: SIMD3<Float> = apexLow + SIMD3<Float>(0.7, 0.05, 0)
    expect(abs(c.crystal.toCrystal(apexLow).y) < 0.002, "at touch-down the apex is \(c.crystal.toCrystal(apexLow).y) µm off the crystal")
    guard let r0 = probe([beside], frame: c) else { expect(false, "probe failed"); return }
    expect(r0[0].insetMaterial == 5 && r0[0].insetDistance < 0, "beside the apex at touch-down: material \(r0[0].insetMaterial)")
    var lowestLifted: Float = 1e9
    for f in frames where f.lift > 0.001 {
        let y: Float = f.crystal.toCrystal(apexLow).y
        lowestLifted = min(lowestLifted, y - 1000 * f.lift)
        expect(y > filmThickness + meniscusHeight, "lifted \(f.lift) mm, the apex is only \(y) µm up")
        guard let r = probe([beside], frame: f) else { expect(false, "probe failed"); return }
        expect(r[0].insetMaterial != 5, "lifted, the film is still at the apex")
    }
    // The inset's crystal falls away by the lift itself, give or take the
    // antenna tip's small turn.
    print(String(format: "        crystal drop vs lift, worst difference %.2f µm", abs(lowestLifted)))
    expect(abs(lowestLifted) < 3, "the inset crystal and the main-view lift disagree by \(lowestLifted) µm")
}

test("the loop is forward: the last frame runs on into the first, and the molecules only go in") {
    guard let s = scene, let c = contactFrame else { expect(false); return }
    // Time L is time 0.
    let end: FrameState = s.frame(at: loopSeconds)
    expect(simd_distance(end.ant.antennae[0].tipCentre, c.ant.antennae[0].tipCentre) < 1e-5)
    for (a, b) in zip(end.molecules, c.molecules) { expect(simd_distance(a.offset, b.offset) < 1e-3) }
    // The step from the last frame into the first is an ordinary step.
    // Molecules are matched by position — the slots relabel once a tap, when
    // each molecule reaches the place the one ahead of it started from.
    func nearest(_ q: MoleculePose, in earlier: [MoleculePose]) -> MoleculePose {
        earlier.min { simd_distance($0.offset, q.offset) < simd_distance($1.offset, q.offset) } ?? q
    }
    func step(_ a: FrameState, _ b: FrameState) -> Float {
        var m: Float = simd_distance(a.ant.antennae[0].tipCentre, b.ant.antennae[0].tipCentre) * 100
        for q in b.molecules {
            let d: Float = simd_distance(nearest(q, in: a.molecules).offset, q.offset)
            if d < moleculeSpacing / 2 { m = max(m, d) }
        }
        return m
    }
    var biggest: Float = 0
    for i in 1..<frames.count { biggest = max(biggest, step(frames[i - 1], frames[i])) }
    let seam: Float = step(frames[frames.count - 1], frames[0])
    print(String(format: "        seam step %.3f, largest step inside the loop %.3f", seam, biggest))
    expect(seam <= biggest * 1.01 + 1e-5, "the seam jumps")
    // Forward: molecule progress never falls, and a loop is exactly one place per tap.
    var last: Float = moleculeProgress(0, mutant: s.mutant)
    for i in 1...frames.count {
        let u: Float = moleculeProgress(frameTime(i, of: frames.count) - (i == frames.count ? 1e-4 : 0), mutant: s.mutant)
        expect(u >= last - 1e-5, "molecules move backwards at frame \(i): \(last) → \(u)")
        last = u
    }
    expect(abs(last - Float(tapsPerLoop)) < 0.01, "the loop advances \(last) places, not \(tapsPerLoop)")
    // Every molecule moves up the path, into the pore, never out.
    for i in 1..<frames.count {
        for q in frames[i].molecules {
            let p: MoleculePose = nearest(q, in: frames[i - 1].molecules)
            // A molecule with none near it a frame ago is a new arrival.
            if simd_distance(p.offset, q.offset) > moleculeSpacing / 2 { continue }
            let along: Float = simd_dot(q.offset - p.offset, SIMD3<Float>(s.drift, 0))
            expect(along > -0.05, "a molecule moves out of the pore at frame \(i)")
        }
    }
    // Each tap is the same: the lift is periodic in the tap.
    for i in 0..<(frames.count - frames.count / tapsPerLoop) {
        expect(abs(frames[i].lift - frames[i + frames.count / tapsPerLoop].lift) < 1e-5)
    }
}

test("the molecules move as rigid bodies: C12H22O11, every bond and every distance unchanged") {
    guard let s = scene else { expect(false); return }
    let atoms: [SIMD3<Float>] = s.sucrose.atoms.map { $0.position }
    var worst: Float = 0
    for f in frames {
        expectEqual(f.molecules.count, moleculeSlots)
        for m in f.molecules {
            let placed: [SIMD3<Float>] = atoms.map { m.place($0) }
            for (a, b) in s.sucrose.bonds {
                worst = max(worst, abs(simd_distance(placed[a], placed[b]) - simd_distance(atoms[a], atoms[b])))
            }
            for i in stride(from: 0, to: atoms.count, by: 4) {
                for j in stride(from: i + 1, to: atoms.count, by: 3) {
                    worst = max(worst, abs(simd_distance(placed[i], placed[j]) - simd_distance(atoms[i], atoms[j])))
                }
            }
        }
    }
    print(String(format: "        worst change in any bond or distance: %.1e Å", worst))
    expect(worst < 1e-3, "a molecule deforms by \(worst) Å")
    let f: [String: Int] = s.sucrose.formula
    expect(f["C"] == 12 && f["H"] == 22 && f["O"] == 11)
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

section("the molecule")

test("the sucrose drawn is C12H22O11 — 45 atoms, 23 straight from PDB 6S1T") {
    guard let s = scene else { expect(false); return }
    let f: [String: Int] = s.sucrose.formula
    expectEqual(f["C"] ?? 0, 12)
    expectEqual(f["H"] ?? 0, 22)
    expectEqual(f["O"] ?? 0, 11)
    expectEqual(s.sucrose.atoms.count, 45)
    expectEqual(s.sucrose.atoms.filter { $0.fromCrystal }.count, 23)
    // One molecule with two rings: bonds = atoms − 1 + 2.
    expectEqual(s.sucrose.bonds.count, 46)
}

test("every atom has its valence: C four bonds, O two, H one") {
    guard let s = scene else { expect(false); return }
    var count: [Int] = Array(repeating: 0, count: s.sucrose.atoms.count)
    for (a, b) in s.sucrose.bonds { count[a] += 1; count[b] += 1 }
    for (i, atom) in s.sucrose.atoms.enumerated() {
        let want: Int = atom.element == "C" ? 4 : (atom.element == "O" ? 2 : 1)
        expect(count[i] == want, "\(atom.element) \(i) has \(count[i]) bonds")
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
    // Step 30: round the tip and the grain with the antenna at its highest.
    if let up = frames.max(by: { $0.lift < $1.lift }) {
        for (m, v) in worstOverReport(box: SIMD3(2.0, 0, 0.3), SIMD3(3.2, 1.2, 1.6), step: 0.004, inset: false, frame: up) {
            w[m] = max(w[m] ?? 0, v)
        }
    }
    print(String(format: "        worst over-report: table %.2f, ant %.2f, sugar %.2f; the ray allows %.2f",
                 w[1] ?? 0, w[2] ?? 0, w[3] ?? 0, 1 / stepScale))
    expect(w.count == 3, "not every material was sampled: \(w)")
    for (m, v) in w { expect(v * stepScale <= 1.0, "material \(m) oversteps: \(v)") }
}

test("inset: the hairs, the dome, the crystal and the film are honest outside too") {
    let w: [Int: Float] = worstOverReport(box: SIMD3(-4, -0.5, -12), SIMD3(8, 12, 4), step: 0.02, inset: true)
    // A second pass concentrated round the tip, where the film and pore are.
    let t: [Int: Float] = worstOverReport(box: SIMD3(-1.5, -0.2, -1.5), SIMD3(1.5, 1.5, 1.5), step: 0.005, inset: true)
    // And a third in a thin shell round each hair's wall, down among the pores.
    let pores: [Int: Float] = worstOverReport(box: .zero, .zero, step: 0.004, inset: true, points: hairShellPoints(count: 80_000))
    var all: [Int: Float] = w
    for (m, v) in t { all[m] = max(all[m] ?? 0, v) }
    for (m, v) in pores { all[m] = max(all[m] ?? 0, v) }
    // Step 30: and with the antenna lifted, when the crystal is moved.
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
    print(String(format: "        worst over-report: crystal %.2f, cuticle %.2f, taste hair %.2f, smell hair %.2f, film %.2f",
                 all[1] ?? 0, all[2] ?? 0, all[3] ?? 0, all[4] ?? 0, all[5] ?? 0))
    expect(all.count == 5, "not every material was sampled: \(all)")
    for (m, v) in all { expect(v * stepScale <= 1.0, "material \(m) oversteps: \(v)") }
}

section("the picture")

let render: AntImage? = {
    guard let d = gpu.device, let s = scene, let f = contactFrame else { return nil }
    return try? renderAnt(width: 960, height: 540, samples: 1, scene: s, frame: f, on: d).image
}()

test("the touch is visible: antenna and sugar both within 3 px of the projected contact") {
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
    expect(sawAnt && sawSugar, "near the contact: antenna \(sawAnt), sugar \(sawSugar)")
}

test("the inset shows both hairs and the film; the molecule inset shows C, O and H") {
    guard let img = render else { expect(false, "render failed"); return }
    var seen: Set<Int> = []
    var atoms: Set<Int> = []
    for y in 0..<img.height {
        for x in 0..<img.width {
            let v: SIMD4<Float> = img.seen(x, y)
            if v.x == 2 { seen.insert(Int(v.y)) }
            if v.x == 3 && v.y >= 1 && v.y <= 3 { atoms.insert(Int(v.y)) }
        }
    }
    expect(seen.isSuperset(of: [3, 4, 5]), "inset materials seen: \(seen.sorted())")
    expectEqual(atoms, [1, 2, 3])
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
    for f in [perTap / 2, perTap * 7 / 10, perTap * 8 / 10, perTap * 9 / 10, perTap + 3] {
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
        // And the loop's way of drawing it — the first frame, then the region
        // alone on top — gives the whole frame exactly.
        base.withUnsafeBytes { raw in r.pixels.contents().copyMemory(from: raw.baseAddress!, byteCount: n) }
        for rect in region.rects { _ = try? r.render(frames[f], samples: 1, region: rect) }
        let composite: [UInt8] = Array(UnsafeBufferPointer(start: r.pixels.contents().assumingMemoryBound(to: UInt8.self), count: n))
        var worstComposite: Int = 0
        for i in 0..<n where i % 4 != 3 { worstComposite = max(worstComposite, abs(Int(composite[i]) - Int(full[i]))) }
        expect(worstComposite <= 1, "frame \(f): the region render differs from the whole render by \(worstComposite) levels")
    }
    print("        region \(region.rects.count) rects, \(100 * region.area / (w * h))% of the frame; changed pixels inside \(inside), outside \(outside), worst outside \(worstOutside) levels")
    // Outside the region only float noise may differ: a soft-shadow ray that
    // passes the moving antenna takes differently sized steps, and lands on
    // a slightly different minimum — one level in 255, at a few pixels.
    expect(worstOutside <= 1, "a pixel outside the region changes by \(worstOutside) levels")
    expect(outside < w * h / 1000, "\(outside) pixels outside the region change")
    expect(inside > 0, "nothing moved")
}

test("lifted, the main view shows table between the tip and the sugar, and the inset no film at the tip") {
    guard let d = gpu.device, let s = scene, let up = frames.max(by: { $0.lift < $1.lift }) else { expect(false); return }
    guard let img = try? renderAnt(width: 1920, height: 1080, samples: 1, scene: s, frame: up, on: d).image else {
        expect(false, "render failed"); return
    }
    let a: Antenna = up.ant.antennae[0]
    let low: SIMD2<Float> = projectMain(a.tipCentre - SIMD3<Float>(0, a.tipRadius, 0), width: 1920, height: 1080)
    let touch: SIMD2<Float> = projectMain(s.contact.point, width: 1920, height: 1080)
    print(String(format: "        tip raised %.1f px on screen", simd_distance(low, touch)))
    expect(simd_distance(low, touch) > 10, "the lift is invisible: \(simd_distance(low, touch)) px")
    // The inset: the pixels round the apex see no film.
    let apex: SIMD2<Float> = projectInset(s.hairs[0].tip, height: 1080)
    var film: Int = 0
    for dy in -12...12 { for dx in -12...12 where img.seen(Int(apex.x) + dx, Int(apex.y) + dy).y == 5 { film += 1 } }
    expectEqual(film, 0)
}

finish()
