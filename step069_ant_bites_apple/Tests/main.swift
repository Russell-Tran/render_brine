// Tests for step 69: one worker biting the edge of a piece of peeled Golden
// Delicious apple. The ant against the literature (step 44's and 50's tests,
// kept); the bite from the geometry the GPU draws — at every grip each jaw
// meets the flesh, at no moment does any part go into it, open it clears it;
// the antennae's taps (step 30's tests); the inset's roads — juice to the
// crop, bits of flesh to the infrabuccal pocket and never down the gut; the
// loop forward and seamless; and the new rule — the bite shows on screen: at
// least 10 px of jaw movement in the main view and many changed pixels in the
// magnified one.
//
// ANT_MUTANT=segments13|rewind|biteThrough|biteShort|stillBite|solidsToCrop|
// press|pieceSize breaks the scene on purpose; `make mutants` requires the
// suite to fail for each.

import Foundation
import Metal
import simd

let mutant: Mutant = {
    switch ProcessInfo.processInfo.environment["ANT_MUTANT"] {
    case "segments13": return .segments13
    case "rewind": return .rewind
    case "biteThrough": return .biteThrough
    case "biteShort": return .biteShort
    case "stillBite": return .stillBite
    case "solidsToCrop": return .solidsToCrop
    case "press": return .press
    case "pieceSize": return .pieceSize
    default: return .none
    }
}()

final class DeviceBox {
    let device: MTLDevice? = try? findDevice()
}
let gpu = DeviceBox()

let scene = Scene(mutant: mutant)
let library: MTLLibrary? = {
    guard let d = gpu.device else { return nil }
    return try? makeLibrary(d, scene)
}()

let gifFrames: Int = Int((loopSeconds * 100).rounded()) / defaultDelayCentiseconds
let frames: [FrameState] = (0..<gifFrames).map { scene.frame(at: frameTime($0, of: gifFrames)) }
let fineTimes: [Float] = (0..<Int(loopSeconds / 0.01)).map { Float($0) * 0.01 }
let fine: [FrameState] = fineTimes.map { scene.frame(at: $0) }

func probe(_ pts: [SIMD3<Float>], frame f: FrameState) -> [Probe]? {
    guard let d = gpu.device, let l = library else { return nil }
    return try? probeScene(pts, scene: scene, frame: f, library: l, on: d)
}

func deg(_ r: Float) -> Float { r * 180 / Float.pi }

/// Points on a shape's surface.
func surfacePoints(_ s: Shape, _ n: Int) -> [SIMD3<Float>] {
    var out: [SIMD3<Float>] = []
    switch s.kind {
    case .roundCone:
        let ax: SIMD3<Float> = simd_normalize(s.b - s.a)
        let helper: SIMD3<Float> = abs(ax.y) < 0.9 ? SIMD3(0, 1, 0) : SIMD3(1, 0, 0)
        let e1: SIMD3<Float> = simd_normalize(simd_cross(ax, helper))
        let e2: SIMD3<Float> = simd_cross(ax, e1)
        for i in 0...n {
            let t: Float = Float(i) / Float(n)
            let c: SIMD3<Float> = s.a + (s.b - s.a) * t
            let r: Float = s.ra + (s.rb - s.ra) * t
            for k in 0..<16 {
                let ang: Float = Float(k) * Float.pi / 8
                out.append(c + (e1 * cos(ang) + e2 * sin(ang)) * r)
            }
        }
        out.append(s.a - ax * s.ra)
        out.append(s.b + ax * s.rb)
    case .ellipsoid, .roundBox:
        let z: SIMD3<Float> = simd_cross(s.xAxis, s.yAxis)
        let e: SIMD3<Float> = s.kind == .ellipsoid ? s.b : s.b + SIMD3<Float>(repeating: s.ra)
        for i in 0...n {
            let th: Float = Float.pi * Float(i) / Float(n)
            for k in 0..<(2 * n) {
                let ph: Float = Float.pi * Float(k) / Float(n)
                out.append(s.a + s.xAxis * (sin(th) * cos(ph) * e.x) + s.yAxis * (cos(th) * e.y) + z * (sin(th) * sin(ph) * e.z))
            }
        }
    }
    return out
}

/// A jaw's nearest point to the flesh (CPU, dense), and its distance.
func nearest(_ shapes: [Shape]) -> (point: SIMD3<Float>, distance: Float) {
    var best: (SIMD3<Float>, Float) = (.zero, 1e9)
    for s in shapes {
        guard s.kind == .roundCone else { continue }
        let ax: SIMD3<Float> = simd_normalize(s.b - s.a)
        for i in 0...400 {
            let t: Float = Float(i) / 400
            let c: SIMD3<Float> = s.a + (s.b - s.a) * t
            let r: Float = s.ra + (s.rb - s.ra) * t
            // The point of this cross-section nearest the flesh: towards the
            // flesh along the local gradient, one radius out.
            let e: Float = 1e-4
            let g = SIMD3<Float>(pieceSDF(c + SIMD3(e, 0, 0), scene.piece) - pieceSDF(c - SIMD3(e, 0, 0), scene.piece),
                                 pieceSDF(c + SIMD3(0, e, 0), scene.piece) - pieceSDF(c - SIMD3(0, e, 0), scene.piece),
                                 pieceSDF(c + SIMD3(0, 0, e), scene.piece) - pieceSDF(c - SIMD3(0, 0, e), scene.piece))
            var dir: SIMD3<Float> = -simd_normalize(g)
            dir -= ax * simd_dot(dir, ax) * 0      // the sphere at c: any direction
            let p: SIMD3<Float> = c + dir * r
            let d: Float = pieceSDF(p, scene.piece)
            if d < best.1 { best = (p, d) }
        }
        let tipEnd: SIMD3<Float> = s.b + ax * s.rb
        let dt: Float = pieceSDF(tipEnd, scene.piece)
        if dt < best.1 { best = (tipEnd, dt) }
    }
    return best
}

func jaw(_ a: BitingAnt, _ side: Int) -> [Shape] { a.mandibles.filter { $0.index == side } }

section("the ant, against the literature")

test("each antenna has 12 segments — scape plus an 11-segment funiculus — at every frame") {
    var bad: Int = 0
    for f in frames {
        for i in 0..<2 where f.ant.shapes.filter({ $0.part == .antenna && $0.index == i }).count != workerAntennaSegments { bad += 1 }
    }
    expectEqual(bad, 0)
}

test("the head is Seifert's: CS 976 µm, CL/CW 1.074, scape SL/CS 0.979") {
    expect(abs(headWidth - 0.941) < 0.002 && abs(headLength - 1.011) < 0.002)
    expect(abs(scapeLength / cephalicSize - 0.979) < 1e-4)
}

test("six legs joined to the mesosoma, all six feet on the card; one petiole; 3.4–5.0 mm long") {
    let a: BitingAnt = frames[0].ant
    expectEqual(a.model.legs.count, 6)
    let meso: [Shape] = a.model.body.filter { $0.part == .mesosoma }
    for leg in a.model.legs {
        expect(meso.contains { $0.contains(leg.root) }, "\(leg.name) is not joined to the mesosoma")
        let tip: Shape = leg.shapes[leg.shapes.count - 1]
        expect(abs(tip.extent(along: SIMD3(0, 1, 0)).lo) < 1e-4, "\(leg.name) is off the card")
    }
    expectEqual(a.model.body.filter { $0.part == .petiole }.count, 1)
    let l: Float = AntModel(body: bodyShapes(gasterBend: 0), legs: [], antennae: []).length
    expect(workerLengthRange.contains(l), "length \(l)")
}

test("two mandibles, each a blade with four teeth on its inner edge, pale, and rigid: every frame the blade keeps its length and root") {
    let r0: [Shape] = jaw(frames[0].ant, 0)
    expectEqual(r0.count, 1 + mandibleTeeth.count)
    expectEqual(jaw(frames[0].ant, 1).count, 1 + mandibleTeeth.count)
    for s in frames[0].ant.mandibles { expect(s.light, "a mandible is not the pale cuticle") }
    var worst: Float = 0
    for f in frames {
        for side in 0..<2 {
            let b0: Shape = jaw(frames[0].ant, side)[0]
            let b: Shape = jaw(f.ant, side)[0]
            worst = max(worst, simd_distance(b.a, b0.a), abs(simd_distance(b.a, b.b) - simd_distance(b0.a, b0.b)))
        }
    }
    expect(worst < 1e-5, "a jaw's root moves or its blade stretches by \(worst) mm")
    // Teeth point inwards: towards the other jaw.
    let right: [Shape] = jaw(frames[0].ant, 0)
    for t in right.dropFirst() { expect(t.b.z < t.a.z, "a right-jaw tooth points outwards") }
}

section("the apple")

test("the piece is 14 × 9 × 7 mm, from a Golden Delicious over 76 mm across, on the card, one upright edge at the ant") {
    let m: ApplePiece = scene.piece
    func reach(_ d: SIMD3<Float>) -> Float {
        var lo: Float = 0
        var hi: Float = 40
        for _ in 0..<60 { let mid: Float = (lo + hi) / 2; if pieceSDF(m.centre + d * mid, m) < 0 { lo = mid } else { hi = mid } }
        return 2 * lo
    }
    let dims = (reach(pieceV), reach(pieceU), reach(SIMD3(0, 1, 0)))
    print(String(format: "        drawn %.2f × %.2f × %.2f mm", dims.0, dims.1, dims.2))
    expect(abs(dims.0 - pieceLength) < 0.01 && abs(dims.1 - pieceDepth) < 0.01 && abs(dims.2 - pieceHeight) < 0.01)
    let diagonal: Float = (pieceLength * pieceLength + pieceDepth * pieceDepth + pieceHeight * pieceHeight).squareRoot()
    expect(diagonal < fruitWidthAtLeast / 2)
    expect(abs(m.centre.y - m.half.z) < 1e-5, "the piece is not on the card")
    // The edge: the two faces meet at cornerBase, square to each other, 45° either side of the ant's heading.
    expect(abs(simd_dot(rightFaceNormal, leftFaceNormal)) < 1e-6)
    expect(abs(simd_dot(rightFaceNormal, SIMD3<Float>(-1, 0, 0)) - simd_dot(leftFaceNormal, SIMD3<Float>(-1, 0, 0))) < 1e-6)
    let root2: Float = Float(2).squareRoot()
    let inset: Float = m.rounding * (root2 - 1) / root2
    let diagonalIn: SIMD3<Float> = (pieceU + pieceV) * inset
    let onEdge: SIMD3<Float> = cornerBase + SIMD3<Float>(0, 2, 0) + diagonalIn
    expect(abs(pieceSDF(onEdge, m)) < 1e-3, "the edge is not at cornerBase: \(pieceSDF(onEdge, m))")
}

test("the same apple as step 68: Golden Delicious's sugars, fructose first in both sources; its measured flesh colour") {
    expect(gdFructose > gdSucrose && gdSucrose > gdGlucose)
    expect(usdaFructose > usdaSucrose && usdaFructose > usdaGlucose)
    expect(fleshAlbedo.x >= fleshAlbedo.y && fleshAlbedo.y > fleshAlbedo.z)
}

section("the bite")

test("the grip is solved, not typed: the jaws close from wide open until they meet the flesh, and both meet it together") {
    print(String(format: "        ant at x %.3f mm; grip: each jaw %.2f° out from rest; widest gape %.0f°", bite.x0, deg(bite.grip),
                 deg(gapeMax)))
    expect(bite.grip > 5 * Float.pi / 180 && bite.grip < gapeMax - 10 * Float.pi / 180, "grip \(deg(bite.grip))°")
    let right = nearest(mandiblesInWorld(x0: bite.x0, open: bite.grip).filter { $0.index == 0 })
    let left = nearest(mandiblesInWorld(x0: bite.x0, open: bite.grip).filter { $0.index == 1 })
    print(String(format: "        at the grip: right jaw %.5f mm from the flesh, left %.5f", right.distance, left.distance))
    expect(abs(right.distance) < 1e-3 && abs(left.distance) < 1e-3)
    // Each on its own face.
    expect(right.point.z > 0 && left.point.z < 0, "a jaw meets the wrong face")
}

test("at every grip each jaw meets the flesh: 0 within 2 µm, on the GPU") {
    var n: Int = 0
    var worst: Float = 0
    for f in fine where f.ant.bitePhase > 0.72 || f.ant.bitePhase < 0.001 {
        guard abs(f.ant.opening - bite.grip) < 1e-6 || mutant != .none else { continue }
        if n > 40 { break }
        let pts: [SIMD3<Float>] = [nearest(jaw(f.ant, 0)).point, nearest(jaw(f.ant, 1)).point]
        guard let q = probe(pts, frame: f) else { expect(false, "probe failed"); return }
        for p in q { worst = max(worst, abs(p.food)); n += 1 }
    }
    print(String(format: "        %d jaw contacts; worst jaw-to-flesh %.4f mm", n, worst))
    expect(n >= 20, "only \(n) gripping samples")
    expect(worst < 0.002, "a gripping jaw is \(worst) mm from the flesh")
}

test("never through the flesh: at no moment does any part of a jaw — blade or tooth — go into it (CPU, every 10 ms; GPU, every frame)") {
    var worst: Float = 9
    for f in fine {
        for s in f.ant.mandibles { for p in surfacePoints(s, 40) { worst = min(worst, pieceSDF(p, scene.piece)) } }
    }
    var gpuWorst: Float = 9
    for f in frames {
        var pts: [SIMD3<Float>] = []
        for s in f.ant.mandibles { pts += surfacePoints(s, 12) }
        guard let q = probe(pts.map { $0 }, frame: f) else { expect(false, "probe failed"); return }
        for (p, r) in zip(pts, q) where r.ant <= 1e-4 { gpuWorst = min(gpuWorst, r.food); _ = p }
    }
    print(String(format: "        nearest jaw surface to the inside of the flesh: CPU %.4f mm, GPU %.4f mm", worst, gpuWorst))
    expect(worst > -0.002, "a jaw goes \(-worst) mm into the flesh")
    expect(gpuWorst > -0.002, "the drawn jaw goes \(-gpuWorst) mm into the flesh")
}

test("open, the jaws clear the flesh; no other part of the ant touches the apple, ever") {
    var openClear: Float = 9
    for f in fine where f.ant.opening > bite.grip + 0.3 {
        for s in f.ant.mandibles { for p in surfacePoints(s, 20) { openClear = min(openClear, pieceSDF(p, scene.piece)) } }
    }
    var body: Float = 9
    for f in stride(from: 0, to: fine.count, by: 10).map({ fine[$0] }) {
        for s in f.ant.shapes where s.part != .mandible && s.part != .antenna {
            let (c, r) = s.bound
            if pieceSDF(c, scene.piece) > r + 0.1 { continue }
            for p in surfacePoints(s, 12) { body = min(body, pieceSDF(p, scene.piece)) }
        }
    }
    print(String(format: "        jaws well open clear the flesh by %.3f mm; the rest of the ant by %.3f mm", openClear, body))
    expect(openClear > 0.01, "open jaws still touch: \(openClear)")
    expect(body > 0.02, "the ant's body touches the apple: \(body)")
}

test("the bite's rhythm: two bites a loop, each opening, holding open, closing and gripping") {
    var grips: Int = 0
    var was: Bool = fine[fine.count - 1].ant.gripping
    for f in fine { if f.ant.gripping && !was { grips += 1 }; was = f.ant.gripping }
    expectEqual(grips, bitesPerLoop)
    let widest: Float = fine.map { $0.ant.opening }.max() ?? 0
    expect(abs(widest - gapeMax) < 1e-3, "widest \(deg(widest))°")
    expect(abs(slowdown - 10) < 1e-5 && abs(Float(gifFrames * defaultDelayCentiseconds) / 100 - loopSeconds) < 1e-4)
}

section("the antennae (step 30's tap)")

test("at every touch-down both tips meet the juice (0 within 2 µm, GPU), never go in, clear it by the lift, and reach") {
    var worstTouch: Float = 0
    var deepest: Float = 1
    var n: Int = 0
    var rng = SystemRandomNumberGenerator()
    for f in frames {
        var pts: [SIMD3<Float>] = []
        for i in 0..<2 {
            let (_, nrm) = antennaContact(i)
            pts.append(f.ant.antennaTips[i] - nrm * funiculusTipRadius)
            for _ in 0..<1500 {
                pts.append(f.ant.antennaTips[i] + SIMD3<Float>(Float.random(in: -0.15...0.15, using: &rng),
                                                               Float.random(in: -0.15...0.15, using: &rng),
                                                               Float.random(in: -0.15...0.15, using: &rng)))
            }
        }
        guard let q = probe(pts, frame: f) else { expect(false, "probe failed"); return }
        for i in 0..<2 {
            let base: Int = i * 1501
            if f.ant.lift == 0 { worstTouch = max(worstTouch, abs(q[base].food)); n += 1 } else {
                expect(abs(q[base].food - f.ant.lift) < 0.002, "lift \(f.ant.lift) but the tip is \(q[base].food) off")
            }
            for k in 1...1500 where q[base + k].ant < 0 { deepest = min(deepest, q[base + k].food) }
        }
        for a in f.ant.model.antennae { expect(a.reached, "an antenna cannot reach its tip") }
    }
    print(String(format: "        %d touching tips, worst %.4f mm; deepest antenna point %.4f mm", n, worstTouch, deepest))
    expect(n >= 8 && worstTouch < 0.002)
    expect(deepest > -0.002, "an antenna goes \(-deepest) mm into the juice")
}

section("the inset: where the juice goes, where the bits go")

func insideCrop(_ p: GutPoint) -> Bool {
    let d: GutPoint = (p - cropCentre) / cropRadii
    return simd_length(d) <= 1
}
func insidePocket(_ p: GutPoint) -> Bool { simd_distance(p, pocketCentre) <= pocketRadius }

test("every drop of juice ends in the crop, in the gaster; every bit of flesh ends in the infrabuccal pocket and never goes down the gut") {
    var juiceEnds: Int = 0
    var solidEnds: Int = 0
    var solidAstray: Int = 0
    for f in fine {
        for d in f.gut.dots where d.radius > 0 {
            if d.kind == .juice && d.arrived { expect(insideCrop(d.position), "a drop ends outside the crop"); juiceEnds += 1 }
            if d.kind == .solid {
                if d.arrived { if insidePocket(d.position) { solidEnds += 1 } else { solidAstray += 1 } }
                // Past the pharynx, back into the neck and beyond: the gut.
                if d.position.x < pharynxPoint.x - 0.05 || insideCrop(d.position) { solidAstray += 1 }
            }
        }
    }
    // The crop is in the gaster; the pocket is in the head, below the mouth.
    let g: Shape = gasterShape(angle: restingGasterLift)
    expect(g.contains(SIMD3<Float>(cropCentre.x, cropCentre.y, 0)), "the crop is not in the gaster")
    let head: Shape = sideViewBody()[0]
    expect(head.contains(SIMD3<Float>(pocketCentre.x, pocketCentre.y, 0)), "the pocket is not in the head")
    expect(pocketCentre.y < pharynxPoint.y, "the pocket is not below the pharynx")
    print("        juice arriving in the crop: \(juiceEnds) samples; bits arriving in the pocket: \(solidEnds); astray: \(solidAstray)")
    expect(juiceEnds > 10 && solidEnds > 10)
    expectEqual(solidAstray, 0)
}

test("the bits and the juice set off only when the jaws have closed on the flesh") {
    for f in fine {
        for d in f.gut.dots where d.radius > 0 && d.life < 0.02 {
            let p: Float = f.ant.bitePhase
            expect(p >= biteOpening + biteOpen + biteClosing - 0.01 || p < 0.1, "a dot sets off at bite phase \(p)")
        }
    }
}

section("the loop")

test("forward and seamless: time L is time 0; dots only move on; the step across the seam is an ordinary step") {
    let end: FrameState = scene.frame(at: loopSeconds)
    var worst: Float = 0
    for (a, b) in zip(end.ant.shapes, frames[0].ant.shapes) { worst = max(worst, simd_distance(a.a, b.a), simd_distance(a.b, b.b)) }
    for (a, b) in zip(end.gut.dots, frames[0].gut.dots) { worst = max(worst, simd_distance(a.position, b.position), abs(a.radius - b.radius)) }
    expect(worst < 1e-4, "time L differs from time 0 by \(worst)")
    // Every dot's life only increases, but for the moment it starts again.
    var back: Int = 0
    for i in 1..<fine.count {
        for (a, b) in zip(fine[i - 1].gut.dots, fine[i].gut.dots) where b.life < a.life && !(a.life > 0.9 && b.life < 0.1) && a.radius > 0 {
            back += 1
        }
    }
    expectEqual(back, 0)
    // The jaws' largest step between frames, and across the seam.
    func step(_ a: FrameState, _ b: FrameState) -> Float { abs(a.ant.opening - b.ant.opening) }
    var biggest: Float = 0
    for i in 1..<frames.count { biggest = max(biggest, step(frames[i - 1], frames[i])) }
    expect(step(frames[frames.count - 1], end) <= biggest + 1e-6)
}

test("nothing pops: 1 ms apart, the jaws, antennae and every dot move only a little") {
    var pops: Int = 0
    var times: [Float] = (0..<frames.count).map { frameTime($0, of: frames.count) }
    for k in 0..<bitesPerLoop {
        for x in [Float(0), biteOpening, biteOpening + biteOpen, biteOpening + biteOpen + biteClosing, dotStart] {
            let t: Float = (Float(k) + x) * biteSeconds
            times += [t - 1e-3, t, t + 1e-3]
        }
    }
    for t in times {
        let a: FrameState = scene.frame(at: t)
        let b: FrameState = scene.frame(at: t + 1e-3)
        var m: Float = 0
        for (x, y) in zip(a.ant.shapes, b.ant.shapes) { m = max(m, simd_distance(x.a, y.a), simd_distance(x.b, y.b)) }
        if m > 0.01 { pops += 1 }
        for (x, y) in zip(a.gut.dots, b.gut.dots) where x.radius > 0 && y.radius > 0 {
            if simd_distance(x.position, y.position) > 0.02 || abs(x.radius - y.radius) > 0.005 { pops += 1 }
        }
        for (x, y) in zip(a.gut.dots, b.gut.dots) where (x.radius > 0) != (y.radius > 0) {
            if max(x.radius, y.radius) > 0.004 { pops += 1 }
        }
    }
    expectEqual(pops, 0)
}

section("the distance function")

test("outside every surface, no distance claims more room than the ray allows (main view, round the jaws too)") {
    var worst: [Int: Float] = [:]
    var rng = SystemRandomNumberGenerator()
    for f in [frames[0], frames[frames.count / 4], frames[frames.count / 2]] {
        for (lo, hi, step) in [(SIMD3<Float>(-4, 0, -2), SIMD3<Float>(1.5, 2, 2), Float(0.01)),
                               (SIMD3<Float>(-0.5, 0.1, -0.5), SIMD3<Float>(0.4, 0.7, 0.5), Float(0.003))] {
            var a: [SIMD3<Float>] = []
            var b: [SIMD3<Float>] = []
            for _ in 0..<80_000 {
                let p = SIMD3<Float>(Float.random(in: lo.x...hi.x, using: &rng), Float.random(in: lo.y...hi.y, using: &rng),
                                     Float.random(in: lo.z...hi.z, using: &rng))
                let d: SIMD3<Float> = simd_normalize(SIMD3<Float>(Float.random(in: -1...1, using: &rng), Float.random(in: -1...1, using: &rng),
                                                                  Float.random(in: -1...1, using: &rng)))
                a.append(p)
                b.append(p + d * step)
            }
            guard let pa = probe(a, frame: f), let pb = probe(b, frame: f) else { expect(false, "probe failed"); return }
            for i in 0..<a.count where pa[i].mainMaterial == pb[i].mainMaterial && pa[i].mainDistance > 0 && pb[i].mainDistance > 0 {
                worst[pa[i].mainMaterial] = max(worst[pa[i].mainMaterial] ?? 0, abs(pa[i].mainDistance - pb[i].mainDistance) / step)
            }
        }
    }
    print(String(format: "        worst over-report: card %.2f, ant %.2f, apple %.2f; the ray allows %.2f", worst[1] ?? 0, worst[2] ?? 0,
                 worst[3] ?? 0, 1 / stepScale))
    expect(worst.count == 3)
    for (m, v) in worst { expect(v * stepScale <= 1.0, "material \(m) oversteps: \(v)") }
}

section("the picture")

test("the scale bar is true: 1 mm in the main view; the magnified view says its own magnification") {
    let w: Int = 1600
    let h: Int = 900
    let p = SIMD3<Float>(-1, 0.3, 0.2)
    let dx: Float = simd_distance(projectMain(p, width: w, height: h), projectMain(p + mainCamera.right, width: w, height: h))
    expect(abs(dx - 1 / mainMillimetresPerPixel(width: w)) < 0.01, "1 mm is \(dx) px")
    let dz: Float = simd_distance(projectZoom(p, height: h), projectZoom(p + zoomCamera.right * 0.1, height: h))
    let zoomPx: Float = 2 * zoomRadius * Float(h) / zoomField * 0.1
    expect(abs(dz - zoomPx) < 0.01, "0.1 mm in the magnified view is \(dz) px, not \(zoomPx)")
    print(String(format: "        magnification ×%.2f", zoomMagnification(width: w, height: h)))
}

// The new rule: the bite must show. Render the grip and the widest gape at
// the GIF's size; measure how far the right jaw's tip moves in the main view,
// and count changed pixels in the magnified view.
test("VISIBLE: the bite moves a jaw tip at least 10 px in the main view and changes thousands of pixels in the magnified one") {
    guard let d = gpu.device else { expect(false); return }
    let w: Int = 1600
    let h: Int = 900
    let grip: FrameState = scene.frame(at: 0)
    let openTime: Float = (biteOpening + biteOpen / 2) * biteSeconds
    let open: FrameState = scene.frame(at: openTime)
    guard let r = try? Renderer(device: d, scene: scene, width: w, height: h) else { expect(false, "renderer"); return }
    _ = try? r.render(grip, samples: 1)
    let n: Int = w * h * 4
    let a: [UInt8] = Array(UnsafeBufferPointer(start: r.pixels.contents().assumingMemoryBound(to: UInt8.self), count: n))
    _ = try? r.render(open, samples: 1)
    let b: [UInt8] = Array(UnsafeBufferPointer(start: r.pixels.contents().assumingMemoryBound(to: UInt8.self), count: n))
    let tip0: SIMD3<Float> = jaw(grip.ant, 0)[0].b
    let tip1: SIMD3<Float> = jaw(open.ant, 0)[0].b
    let moved: Float = simd_distance(projectMain(tip0, width: w, height: h), projectMain(tip1, width: w, height: h))
    let movedZoom: Float = simd_distance(projectZoom(tip0, height: h), projectZoom(tip1, height: h))
    let zc: SIMD2<Float> = zoomCentre * Float(h)
    let zr: Float = zoomRadius * Float(h)
    var changed: Int = 0
    for y in Int(zc.y - zr)..<Int(zc.y + zr) {
        for x in Int(zc.x - zr)..<Int(zc.x + zr) where simd_distance(SIMD2<Float>(Float(x), Float(y)), zc) < zr {
            let i: Int = (y * w + x) * 4
            var m: Int = 0
            for k in 0..<3 { m = max(m, abs(Int(a[i + k]) - Int(b[i + k]))) }
            if m > 40 { changed += 1 }
        }
    }
    print(String(format: "        jaw tip moved %.1f px in the main view, %.1f px magnified; %d pixels changed by > 40 levels there",
                 moved, movedZoom, changed))
    expect(moved >= 10, "the bite moves the jaw only \(moved) px: invisible")
    expect(changed >= 2000, "only \(changed) pixels change in the magnified view")
}

test("text and plates stay off the ant, and the inset panel and magnified view never hide it") {
    let w: Int = 1600
    let h: Int = 900
    let rects: [PixelRect] = [titleBlock(height: h), insetBlock(height: h), scaleBarBlock(height: h)]
    let zc: SIMD2<Float> = zoomCentre * Float(h)
    let zr: Float = zoomRadius * Float(h) + 8
    let pxPerMM: Float = Float(w) / mainViewWidth
    var hits: [String: Int] = [:]
    var nearest: Float = 1e9
    for f in frames {
        for s in f.ant.shapes {
            let (c, r) = s.bound
            let shadow: SIMD3<Float> = c - keyDirection * (c.y / keyDirection.y)
            for q in [c, shadow] {
                let p: SIMD2<Float> = projectMain(q, width: w, height: h)
                let rp: Float = r * pxPerMM + (q == c ? 0 : 6)
                for b in rects {
                    let dx: Float = max(Float(b.x0) - p.x, 0, p.x - Float(b.x1))
                    let dy: Float = max(Float(b.y0) - p.y, 0, p.y - Float(b.y1))
                    let d: Float = (dx * dx + dy * dy).squareRoot() - rp
                    nearest = min(nearest, d)
                    if d < 0 { hits["label \(b.x0),\(b.y0) \(s.part)", default: 0] += 1 }
                }
                let dz: Float = simd_distance(p, zc) - zr - rp
                nearest = min(nearest, dz)
                if dz < 0 && q == c { hits["magnified view \(s.part)", default: 0] += 1 }
            }
        }
    }
    print(String(format: "        nearest the ant or its shadow comes to a label, the panel or the magnified view: %.1f px", nearest))
    for (k, v) in hits.sorted(by: { $0.key < $1.key }) { print("        \(k): \(v)") }
    expect(hits.isEmpty)
}

finish()
