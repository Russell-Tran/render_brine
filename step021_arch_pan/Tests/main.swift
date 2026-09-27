// Tests for step 21. The shot is checked where a camera move goes wrong: that
// it starts and ends where it was told to, that it does not jolt where the
// arch's circle meets its straight line, that it never runs backwards, and
// that it never puts the lens inside the gum. The one piece of new geometry,
// the retromolar pad, is checked as step 20 checked its own: that its
// distance is a distance.
//
// SHOT_MUTANT=raw|rewind breaks the shot on purpose; `make mutants` requires
// the suite to fail for each.

import Foundation
import Metal
import simd

let mutant: ShotMutant = {
    switch ProcessInfo.processInfo.environment["SHOT_MUTANT"] {
    case "raw": return .raw
    case "rewind": return .rewind
    default: return .none
    }
}()

let device: MTLDevice? = try? findDevice()
let placed: [PlacedTooth] = placeTeeth()

/// Every frame's camera, as the loop will render it.
let cameras: [Camera] = (0..<frameCount).map { railCamera(plan($0, mutant: mutant).arc, mutant: mutant) }
let travel: [Camera] = Array(cameras[0...travelFrames])

section("where the shot starts and ends")

test("it starts facing the midpoint between #25 and #24, from straight in front") {
    let first: Camera = cameras[0]
    expectEqual(plan(0, mutant: mutant).arc, 0)
    expectEqual(first.target, SIMD3<Float>(0, aimHeight, 0))
    expect(first.position.x == 0 && first.position.z < 0, "camera at \(first.position)")
}

test("it ends facing the centre of #18, and #18 is the second molar on the viewer's right at the start") {
    let t: PlacedTooth = placed[tooth18]
    expect(t.spec.kind == .secondMolar, "tooth \(tooth18) is a \(t.spec.name)")
    let end: Camera = cameras[travelFrames]
    expectEqual(plan(travelFrames, mutant: mutant).arc, endArc)
    expectEqual(SIMD2<Float>(end.target.x, end.target.z), t.centre)
    // The patient's left is the viewer's right, facing them. This is what
    // makes the tooth #18 and not #31, its mirror image.
    let start: Camera = cameras[0]
    let across: Float = simd_dot(SIMD3<Float>(t.centre.x, 0, t.centre.y) - start.position, start.right)
    expect(across > 0, "#18 should be to the right of the first frame's centre: \(across)")
    // The camera ends on the cheek side of the molar, looking in.
    let fromTooth = SIMD2<Float>(end.position.x, end.position.z) - t.centre
    expect(simd_dot(simd_normalize(fromTooth), t.outward) > 0.95, "not square to the buccal face")
}

section("no jolt at the canine")

test("the rail bends smoothly everywhere, including where the circle meets the line at 17.5 mm") {
    // The second difference of the camera's position along the arc, over a
    // step h, divided by h². On a smooth rail it is the rail's curvature times
    // its speed squared and stays put as h shrinks. Where the rail has a
    // corner — the camera's speed jumping from 3× the look-at point's round
    // the circle to 1× along the line — it grows as 1/h: 40 per mm here.
    let h: Float = 0.05
    var worst: Float = 0
    var worstAt: Float = 0
    var s: Float = -5
    while s <= endArc + 5 {
        let a: SIMD3<Float> = railCamera(s - h, mutant: mutant).position
        let b: SIMD3<Float> = railCamera(s, mutant: mutant).position
        let c: SIMD3<Float> = railCamera(s + h, mutant: mutant).position
        let bend: Float = simd_length(a - 2 * b + c) / (h * h)
        if bend > worst { worst = bend; worstAt = s }
        s += h / 3
    }
    print(String(format: "        sharpest bend %.2f per mm, at arc %.2f mm", worst, worstAt))
    expect(worst < 1.0, "the rail bends at \(worst) per mm at arc \(worstAt) mm")
}

test("frame to frame, the camera's swing and its speed change gradually") {
    var worstSwing: Float = 0
    var worstLurch: Float = 0
    let heading: [Float] = travel.map { atan2($0.forward.x, $0.forward.z) }
    for f in 1..<(travel.count - 1) {
        let swingBefore: Float = heading[f] - heading[f - 1]
        let swingAfter: Float = heading[f + 1] - heading[f]
        worstSwing = max(worstSwing, abs(swingAfter - swingBefore))
        let lurch: SIMD3<Float> = travel[f + 1].position - 2 * travel[f].position + travel[f - 1].position
        worstLurch = max(worstLurch, simd_length(lurch))
    }
    print(String(format: "        worst change in swing %.5f rad/frame, in step %.3f mm/frame", worstSwing, worstLurch))
    expect(worstSwing < 0.002, "the heading's rate jumps by \(worstSwing) rad in one frame")
    expect(worstLurch < 0.1, "the camera's step jumps by \(worstLurch) mm in one frame")
}

section("forward only, then a dissolve")

test("the look-at point never moves back along the arch, and eases in and out of rest") {
    var arcs: [Float] = []
    for f in 0..<frameCount { arcs.append(plan(f, mutant: mutant).arc) }
    for f in 1..<frameCount {
        expect(arcs[f] >= arcs[f - 1], "frame \(f) runs backwards: \(arcs[f - 1]) → \(arcs[f])")
    }
    let glide: Float = arcs[travelFrames / 2 + 1] - arcs[travelFrames / 2]
    let off: Float = arcs[1] - arcs[0]
    let into: Float = arcs[travelFrames] - arcs[travelFrames - 1]
    expect(off < 0.05 * glide && into < 0.05 * glide, "start \(off), end \(into), glide \(glide) mm/frame")
}

test("after a hold on #18 the loop closes by dissolving to frame 0, never by rewinding") {
    let fadeFrom: Int = travelFrames + holdFrames + 1
    for f in 0..<fadeFrom { expectEqual(plan(f, mutant: mutant).dissolve, 0) }
    for f in (travelFrames + 1)..<frameCount {
        expectEqual(plan(f, mutant: mutant).arc, endArc)
    }
    var last: Float = 0
    for f in fadeFrom..<frameCount {
        let w: Float = plan(f, mutant: mutant).dissolve
        expect(w > last && w < 1, "dissolve at frame \(f) is \(w) after \(last)")
        last = w
    }
    // The last frame is one even step short of frame 0, so the wrap completes it.
    let step: Float = 1 / Float(dissolveFrames + 1)
    expect(abs(last - (1 - step)) < 1e-5, "last dissolve \(last)")
    print(String(format: "        %d frames, %.2f s: %d travel, %d hold, %d dissolve",
                 frameCount, loopSeconds, travelFrames, holdFrames, dissolveFrames))
}

section("the lens stays out of the mouth")

test("every camera position is at least 5 mm from every surface") {
    // The scene's distance is a lower bound, not the exact distance — the
    // gum's is divided by its steepest slope so a ray can trust it — so the
    // real clearance is larger than this reads. 5 mm by this measure is safe.
    guard let dev = device, let d = try? probeScene(cameras.map { $0.position }, retromolarPad: true, on: dev)
    else { expect(false, "probe failed"); return }
    var nearest: Float = .infinity
    var at: Int = 0
    for (f, r) in d.enumerated() where r.x < nearest { nearest = r.x; at = f }
    print(String(format: "        nearest surface to the lens: at least %.1f mm (frame %d, material %d)",
                 nearest, at, Int(d[at].y)))
    expect(nearest > 5, "the camera comes within \(nearest) mm of the scene at frame \(at)")
}

section("the retromolar pad")

test("the pad's distance never claims more room than there is, beyond what the ray allows") {
    guard let dev = device else { expect(false, "no GPU"); return }
    // Step 20's test, pointed at the new geometry: pairs of nearby points
    // round the back of both second molars, outside every surface.
    var a: [SIMD3<Float>] = []
    var b: [SIMD3<Float>] = []
    var rng = SystemRandomNumberGenerator()
    for t in [placed[6], placed[13]] {
        let behind: SIMD2<Float> = t.centre + t.tangent * (t.spec.width / 2 + padBehind)
        for _ in 0..<40_000 {
            let p = SIMD3<Float>(behind.x + Float.random(in: -14...14, using: &rng),
                                 Float.random(in: -26...4, using: &rng),
                                 behind.y + Float.random(in: -14...14, using: &rng))
            let dir: SIMD3<Float> = simd_normalize(SIMD3<Float>(Float.random(in: -1...1, using: &rng),
                                                                Float.random(in: -1...1, using: &rng),
                                                                Float.random(in: -1...1, using: &rng)))
            a.append(p)
            b.append(p + dir * 0.05)
        }
    }
    guard let da = try? probeScene(a, retromolarPad: true, on: dev),
          let db = try? probeScene(b, retromolarPad: true, on: dev)
    else { expect(false, "probe failed"); return }
    var worst: Float = 0
    for i in 0..<a.count where da[i].y == 2 && db[i].y == 2 && da[i].x > 0 && db[i].x > 0 {
        worst = max(worst, abs(da[i].x - db[i].x) / 0.05)
    }
    print(String(format: "        worst over-report round the pads %.2f; the ray allows %.2f", worst, 1 / stepScale))
    expect(worst * stepScale <= 1.0, "the pad oversteps: \(worst)")
}

test("the pad closes the ridge behind #18 without covering the tooth") {
    guard let dev = device else { expect(false, "no GPU"); return }
    let t: PlacedTooth = placed[tooth18]
    let top: Float = -t.spec.crownHeight + gumMarginAboveCEJ           // the gum margin
    let behind: SIMD2<Float> = t.centre + t.tangent * (t.spec.width / 2 + padBehind)
    let distal: SIMD2<Float> = t.centre + t.tangent * (t.spec.width / 2 - 1.5)
    let pts: [SIMD3<Float>] = [
        SIMD3<Float>(behind.x, top + 1, behind.y),                     // just above the old gum line
        SIMD3<Float>(t.centre.x, -3, t.centre.y),                      // the middle of the crown
        SIMD3<Float>(distal.x, -2.5, distal.y),                        // the crown's distal end
    ]
    guard let with = try? probeScene(pts, retromolarPad: true, on: dev),
          let without = try? probeScene(pts, on: dev)
    else { expect(false, "probe failed"); return }
    expect(without[0].x > 0 && with[0].x < 0 && with[0].y == 2,
           "behind the molar: \(without[0]) without the pad, \(with[0]) with it")
    expect(with[1].x < 0 && with[1].y == 1, "the crown should still be enamel: \(with[1])")
    expect(with[2].x < 0 && with[2].y == 1, "the distal crown should still be enamel: \(with[2])")
}

section("the frames")

test("the first frame faces the midline and the last faces a second molar") {
    guard let dev = device,
          let first = try? renderMouth(width: 320, height: 180, samples: 1, camera: cameras[0],
                                       retromolarPad: true, on: dev).image,
          let last = try? renderMouth(width: 320, height: 180, samples: 1, camera: cameras[travelFrames],
                                      retromolarPad: true, on: dev).image
    else { expect(false, "render failed"); return }
    // The arch is its own mirror image and the first camera sits on the
    // midline, so what each pixel sees must mirror left to right.
    var same: Int = 0
    for y in 0..<180 {
        for x in 0..<160 where first.seen(x, y).x == first.seen(319 - x, y).x { same += 1 }
    }
    let mirrored: Double = Double(same) / Double(160 * 180)
    // And straight ahead in the last frame is #18's buccal face.
    let centre: SIMD4<Float> = last.seen(160, 90)
    print(String(format: "        first frame %.1f%% mirror-symmetric; last frame's centre sees material %d, kind %d",
                 mirrored * 100, Int(centre.x), Int(centre.z)))
    expect(mirrored > 0.99, "the first frame is only \(mirrored) symmetric")
    expect(centre.x == 1 && Int(centre.z) == CrownKind.secondMolar.rawValue, "centre sees \(centre)")
}

finish()
