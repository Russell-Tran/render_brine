// Tests for step 67. The shot is checked where a camera move goes wrong, as
// step 21's was: that it starts and ends where it was told to — the midpoint
// of #8 and #9, and #16 on the patient's left — that it does not jolt at the
// canine, that it never runs backwards, that it never puts the lens inside the
// gum, and (new, from step 29) that the motion clearly reads: frames a second
// apart must differ substantially. The teeth are step 66's; the table they are
// sized from is checked again, since a camera ending on the right number on
// the wrong teeth would pass everything else.
//
// SHOT_MUTANT=raw|rewind|frozenCamera|mirrored and UPPER_MUTANT=mandibularSizes
// break it on purpose; `make mutants` requires the suite to fail for each.

import Foundation
import Metal
import simd

/// Holds the scene so a failure to build one is a test failure, not a crash.
final class SceneBox {
    var renderer: MouthRenderer?
    var error: String = ""
    init() {
        do { renderer = try MouthRenderer(on: try findDevice()) } catch { self.error = "\(error)" }
    }
}
let box = SceneBox()

func probe(_ pts: [SIMD3<Float>]) -> [SIMD2<Float>]? {
    guard let r = box.renderer else { return nil }
    return try? r.probe(pts)
}

let placed: [PlacedTooth] = placeTeeth()
let cameras: [Camera] = (0..<frameCount).map { frameCamera($0) }
let travel: [Camera] = Array(cameras[0...travelFrames])

section("the teeth the camera travels past")

test("they are the maxillary teeth of Wheeler's table, not step 20's mandibular ones") {
    expectEqual(toothTable.map { $0.width }, [8.5, 6.5, 7.5, 7.0, 7.0, 10.0, 9.0, 8.5])
    expectEqual(toothTable.map { $0.crownHeight }, [10.5, 9.0, 10.0, 8.5, 8.5, 7.5, 7.0, 6.5])
    expectEqual(toothTable.map { $0.depth }, [7.0, 6.0, 8.0, 9.0, 9.0, 11.0, 11.0, 10.0])
    expectEqual(placed.count, 16)
    expectEqual(placed.map { $0.number }.sorted(), Array(1...16))
}

test("#16 sits on the arch straight behind #15, touching it, on the patient's left; #1 mirrors it") {
    let t16: PlacedTooth = placed[tooth16]
    let t15: PlacedTooth = placed[tooth15]
    guard let i1 = toothIndex(1, in: placed) else { expect(false, "no #1"); return }
    let t1: PlacedTooth = placed[i1]
    expect(t16.number == 16 && t16.spec.name == "third molar" && t16.centre.x < 0, "#16: \(t16.spec.name) at \(t16.centre)")
    expect(t15.number == 15 && t15.spec.name == "second molar", "#15: \(t15.spec.name)")
    let gap: Float = simd_distance(t16.centre, t15.centre)
    let touching: Float = (t16.spec.width + t15.spec.width) / 2
    expect(abs(gap - touching) < 1e-3, "#16 is \(gap) mm from #15; touching is \(touching)")
    expect(simd_dot(simd_normalize(t16.centre - t15.centre), t15.tangent) > 0.99999, "#16 is not straight behind #15")
    expect(abs(t16.centre.x + t1.centre.x) < 1e-4 && abs(t16.centre.y - t1.centre.y) < 1e-4, "#1 does not mirror #16")
}

section("where the shot starts and ends")

test("it starts facing the midpoint between #8 and #9, from straight in front and below") {
    let first: Camera = cameras[0]
    expectEqual(plan(0).arc, 0)
    expectEqual(first.target, SIMD3<Float>(0, aimHeight, 0))
    expect(first.position.x == 0 && first.position.z < 0 && first.position.y < 0, "camera at \(first.position)")
    guard let i8 = toothIndex(8, in: placed), let i9 = toothIndex(9, in: placed) else { expect(false, "missing"); return }
    let mid: SIMD2<Float> = (placed[i8].centre + placed[i9].centre) / 2
    expect(abs(mid.x) < 1e-5 && abs(first.target.z - mid.y) < 0.5, "the midpoint of #8 and #9 is \(mid)")
}

test("it ends facing the centre of #16, which was on the viewer's right at the start") {
    let t: PlacedTooth = placed[tooth16]
    let end: Camera = cameras[travelFrames]
    expectEqual(plan(travelFrames).arc, endArc)
    let aimed = SIMD2<Float>(end.target.x, end.target.z)
    expect(simd_distance(aimed, t.centre) < 1e-4, "the last frame aims at \(aimed); #16 is at \(t.centre)")
    // The patient's left is the viewer's right, facing them: what makes it
    // #16 and not #1, its mirror image.
    let start: Camera = cameras[0]
    let across: Float = simd_dot(SIMD3<Float>(t.centre.x, 0, t.centre.y) - start.position, start.right)
    expect(across > 0, "#16 should be to the right of the first frame's centre: \(across)")
    // On the cheek side of the molar, looking in and up.
    let fromTooth = SIMD2<Float>(end.position.x, end.position.z) - t.centre
    expect(simd_dot(simd_normalize(fromTooth), t.outward) > 0.95, "not square to #16's buccal face")
    expect(end.position.y < 0 && end.forward.y > 0, "the camera should look up at the upper arch")
}

test("#16 is directly above step 21's #17: the same arch position, one jaw up") {
    // Step 21 ended on the lower third molar at the back of its arch, on the
    // patient's left, from the cheek side, 15 mm on the other side of the
    // occlusal plane. The same rail, turned over, ends on #16.
    expect(railHeight == -15 && railOffset == 35, "the rail should be step 21's, turned over")
    expect(placed[tooth16].side < 0, "#16 should be on the same side as step 21's #17 (−x)")
}

section("no jolt at the canine")

test("the raw arch really has a kink at the canine's distal end, about 15°") {
    let r: Float = hawleyRadius
    let before: SIMD2<Float> = archPoint(r - 1e-3).tangent
    let after: SIMD2<Float> = archPoint(r + 1e-3).tangent
    let kink: Float = acos(min(simd_dot(before, after), 1)) * 180 / Float.pi
    print(String(format: "        kink %.1f° at %.1f mm from the midline", kink, r))
    expect(kink > 10 && kink < 20, "the kink is \(kink)°")
}

test("the rail bends smoothly everywhere, including across the canine") {
    // Step 21's measure: the second difference of the camera's position along
    // the arc over h², which stays put on a smooth rail and grows as 1/h at a
    // corner.
    let h: Float = 0.05
    var worst: Float = 0
    var worstAt: Float = 0
    var s: Float = -5
    while s <= endArc + 5 {
        let a: SIMD3<Float> = railCamera(s - h).position
        let b: SIMD3<Float> = railCamera(s).position
        let c: SIMD3<Float> = railCamera(s + h).position
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
    let arcs: [Float] = (0..<frameCount).map { plan($0).arc }
    for f in 1..<frameCount {
        expect(arcs[f] >= arcs[f - 1], "frame \(f) runs backwards: \(arcs[f - 1]) → \(arcs[f])")
    }
    let glide: Float = arcs[travelFrames / 2 + 1] - arcs[travelFrames / 2]
    let off: Float = arcs[1] - arcs[0]
    let into: Float = arcs[travelFrames] - arcs[travelFrames - 1]
    expect(off < 0.05 * glide && into < 0.05 * glide, "start \(off), end \(into), glide \(glide) mm/frame")
}

test("after a hold on #16 the loop closes by dissolving forward to frame 0, never by rewinding") {
    let fadeFrom: Int = travelFrames + holdFrames + 1
    for f in 0..<fadeFrom { expectEqual(plan(f).dissolve, 0) }
    for f in (travelFrames + 1)..<frameCount { expectEqual(plan(f).arc, endArc) }
    var last: Float = 0
    for f in fadeFrom..<frameCount {
        let w: Float = plan(f).dissolve
        expect(w > last && w < 1, "dissolve at frame \(f) is \(w) after \(last)")
        last = w
    }
    let step: Float = 1 / Float(dissolveFrames + 1)
    expect(abs(last - (1 - step)) < 1e-5, "last dissolve \(last)")
    print(String(format: "        %d frames, %.2f s: %d travel, %d hold, %d dissolve",
                 frameCount, loopSeconds, travelFrames, holdFrames, dissolveFrames))
}

section("the lens stays out of the mouth")

test("every camera position is at least 5 mm from every surface") {
    guard let d = probe(cameras.map { $0.position }) else { expect(false, "probe failed: \(box.error)"); return }
    var nearest: Float = .infinity
    var at: Int = 0
    for (f, r) in d.enumerated() where r.x < nearest { nearest = r.x; at = f }
    print(String(format: "        nearest surface to the lens: at least %.1f mm (frame %d, material %d)",
                 nearest, at, Int(d[at].y)))
    expect(nearest > 5, "the camera comes within \(nearest) mm of the scene at frame \(at)")
}

test("round #16 and its tuberosity, where the shot ends, the distance never over-reports") {
    let t: PlacedTooth = placed[tooth16]
    var rng = SystemRandomNumberGenerator()
    var a: [SIMD3<Float>] = []
    var b: [SIMD3<Float>] = []
    for _ in 0..<80_000 {
        let p = SIMD3<Float>(t.centre.x + Float.random(in: -16...16, using: &rng),
                             Float.random(in: -8...28, using: &rng),
                             t.centre.y + Float.random(in: -14...20, using: &rng))
        let dir: SIMD3<Float> = simd_normalize(SIMD3<Float>(Float.random(in: -1...1, using: &rng),
                                                            Float.random(in: -1...1, using: &rng),
                                                            Float.random(in: -1...1, using: &rng)))
        a.append(p)
        b.append(p + dir * 0.05)
    }
    guard let da = probe(a), let db = probe(b) else { expect(false, "probe failed"); return }
    var worst: [Float] = [0, 0, 0, 0]
    for i in 0..<a.count where da[i].y == db[i].y && da[i].x > 0 && db[i].x > 0 {
        let m: Int = Int(da[i].y)
        worst[m] = max(worst[m], abs(da[i].x - db[i].x) / 0.05)
    }
    print(String(format: "        worst over-report round #16: enamel %.2f, gum %.2f, palate %.2f; the ray allows %.2f",
                 worst[1], worst[2], worst[3], 1 / stepScale))
    for m in 1...3 { expect(worst[m] * stepScale <= 1.0, "material \(m) oversteps: \(worst[m])") }
}

section("the frames")

let testWidth: Int = 320
let testHeight: Int = 180

func renderFrame(_ f: Int) -> MouthImage? {
    guard let r = box.renderer else { return nil }
    return try? r.render(width: testWidth, height: testHeight, samples: 1, camera: cameras[f]).image
}

test("the first frame faces the midline and the last looks straight at #16") {
    guard let first = renderFrame(0), let last = renderFrame(travelFrames) else { expect(false, "render failed: \(box.error)"); return }
    // The arch is its own mirror image and the first camera is on the midline.
    var same: Int = 0
    let half: Int = testWidth / 2
    for y in 0..<testHeight {
        for x in 0..<half where first.seen(x, y).x == first.seen(testWidth - 1 - x, y).x { same += 1 }
    }
    let mirrored: Double = Double(same) / Double(half * testHeight)
    let centre: SIMD4<Float> = last.seen(half, testHeight / 2)
    print(String(format: "        first frame %.1f%% mirror-symmetric; last frame's centre sees material %d, tooth #%d",
                 mirrored * 100, Int(centre.x), Int(centre.w)))
    expect(mirrored > 0.99, "the first frame is only \(mirrored) symmetric")
    expect(centre.x == 1 && Int(centre.w) == 16, "the last frame's centre sees \(centre), not #16")
    // And #16 fills a good part of the middle of the last frame.
    var sixteen: Int = 0
    for y in (testHeight / 4)..<(3 * testHeight / 4) {
        for x in (testWidth / 4)..<(3 * testWidth / 4) where last.seen(x, y).x == 1 && Int(last.seen(x, y).w) == 16 { sixteen += 1 }
    }
    let share: Double = Double(sixteen) / Double(testWidth * testHeight / 4)
    print(String(format: "        #16 covers %.0f%% of the middle quarter of the last frame", share * 100))
    expect(share > 0.15, "#16 covers only \(share) of the middle of the last frame")
}

test("the motion reads: every pair of frames a second apart during the travel differs substantially") {
    // Step 29's rule. A pixel counts as changed when its colour moves by more
    // than 40 (sum over R, G, B, of 765). Frames one second (20 frames) apart,
    // across the whole travel, the slow start included.
    let step: Int = Int(framesPerSecond)
    var shares: [Double] = []
    var f: Int = 0
    while f + step <= travelFrames {
        guard let a = renderFrame(f), let b = renderFrame(f + step) else { expect(false, "render failed"); return }
        var changed: Int = 0
        for y in 0..<testHeight {
            for x in 0..<testWidth {
                let p: SIMD4<UInt8> = a.rgba(x, y)
                let q: SIMD4<UInt8> = b.rgba(x, y)
                let d: Int = abs(Int(p.x) - Int(q.x)) + abs(Int(p.y) - Int(q.y)) + abs(Int(p.z) - Int(q.z))
                if d > 40 { changed += 1 }
            }
        }
        shares.append(Double(changed) / Double(testWidth * testHeight))
        f += step
    }
    print("        changed each second: " + shares.map { String(format: "%.0f%%", $0 * 100) }.joined(separator: " "))
    let least: Double = shares.min() ?? 0
    expect(least > 0.2, "a second of travel changes only \(least) of the frame")
}

finish()
