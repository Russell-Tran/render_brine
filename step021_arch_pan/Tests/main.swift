// Tests for step 21. The shot is checked where a camera move goes wrong: that
// it starts and ends where it was told to, that it does not jolt where the
// arch's circle meets its straight line, that it never runs backwards, and
// that it never puts the lens inside the gum. It ends on the wisdom tooth, #17,
// which step 20 adds when asked; the tests check that tooth is where the arch
// says and that the last frame really looks at it. The retromolar pad behind it
// is checked as step 20 checked its own geometry — that its distance is a
// distance — and for the thing it is there to fix: no wall at the back of the
// gum.
//
// SHOT_MUTANT=raw|rewind|noWisdom breaks the shot on purpose; `make mutants`
// requires the suite to fail for each.

import Foundation
import Metal
import simd

let mutant: ShotMutant = {
    switch ProcessInfo.processInfo.environment["SHOT_MUTANT"] {
    case "raw": return .raw
    case "rewind": return .rewind
    case "noWisdom": return .noWisdom
    default: return .none
    }
}()

let device: MTLDevice? = try? findDevice()
let thirdMolars: Bool = withThirdMolars(mutant)
let placed: [PlacedTooth] = placeTeeth(thirdMolars: thirdMolars)
/// #17 and #32 as placed, if the scene has them.
let wisdom: [PlacedTooth] = placed.count > tooth32Index ? [placed[tooth17], placed[tooth32Index]] : []

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

test("it ends facing the centre of #17, the wisdom tooth, on the viewer's right at the start") {
    guard let t: PlacedTooth = wisdom.first else { expect(false, "the scene has no #17"); return }
    expect(t.spec.name == "third molar" && t.side == travelSide, "tooth \(tooth17) is a \(t.spec.name), side \(t.side)")
    let end: Camera = cameras[travelFrames]
    expectEqual(plan(travelFrames, mutant: mutant).arc, endArc)
    expectEqual(SIMD2<Float>(end.target.x, end.target.z), t.centre)
    expectEqual(end.target.y, aimHeight)
    // The patient's left is the viewer's right, facing them. This is what
    // makes the tooth #17 and not #32, its mirror image.
    let start: Camera = cameras[0]
    let across: Float = simd_dot(SIMD3<Float>(t.centre.x, 0, t.centre.y) - start.position, start.right)
    expect(across > 0, "#17 should be to the right of the first frame's centre: \(across)")
    // The camera ends on the cheek side of the molar, looking in.
    let fromTooth = SIMD2<Float>(end.position.x, end.position.z) - t.centre
    expect(simd_dot(simd_normalize(fromTooth), t.outward) > 0.95, "not square to the buccal face")
}

test("#17 sits on the arch straight behind #18, touching it, and #32 mirrors it") {
    guard wisdom.count == 2 else { expect(false, "the scene has no wisdom teeth"); return }
    let t17: PlacedTooth = wisdom[0]
    let t32: PlacedTooth = wisdom[1]
    let t18: PlacedTooth = placed[tooth18]
    expect(t18.spec.name == "second molar" && t18.side == travelSide, "tooth \(tooth18) is a \(t18.spec.name)")
    // On the straight part of the arch, so the chord is the arc exactly.
    let gap: Float = simd_distance(t17.centre, t18.centre)
    let touching: Float = (t17.spec.width + t18.spec.width) / 2
    expect(abs(gap - touching) < 1e-3, "#17 is \(gap) mm from #18; touching is \(touching)")
    expect(simd_dot(t17.tangent, t18.tangent) > 0.99999, "#17 is turned off the line of #18")
    expect(simd_dot(simd_normalize(t17.centre - t18.centre), t18.tangent) > 0.99999, "#17 is not straight behind #18")
    expect(abs(t17.centre.x + t32.centre.x) < 1e-4 && abs(t17.centre.y - t32.centre.y) < 1e-4,
           "#32 does not mirror #17: \(t17.centre) vs \(t32.centre)")
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

test("after a hold on #17 the loop closes by dissolving to frame 0, never by rewinding") {
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
    guard let dev = device,
          let d = try? probeScene(cameras.map { $0.position }, retromolarPad: true, thirdMolars: thirdMolars, on: dev)
    else { expect(false, "probe failed"); return }
    var nearest: Float = .infinity
    var at: Int = 0
    for (f, r) in d.enumerated() where r.x < nearest { nearest = r.x; at = f }
    print(String(format: "        nearest surface to the lens: at least %.1f mm (frame %d, material %d)",
                 nearest, at, Int(d[at].y)))
    expect(nearest > 5, "the camera comes within \(nearest) mm of the scene at frame \(at)")
}

section("the wisdom teeth and the retromolar pad")

test("the wisdom teeth are solid enamel, clear of the tongue, with gum round their necks") {
    guard let dev = device else { expect(false, "no GPU"); return }
    guard wisdom.count == 2 else { expect(false, "the scene has no wisdom teeth"); return }
    var pts: [SIMD3<Float>] = []
    for t in wisdom {
        let h: Float = t.spec.crownHeight
        let lingual: SIMD2<Float> = t.centre - t.outward * (t.spec.depth / 2 - 1.0)
        pts.append(SIMD3<Float>(t.centre.x, -h / 2, t.centre.y))              // mid-crown
        pts.append(SIMD3<Float>(lingual.x, -h / 2, lingual.y))                // just inside the tongue side
        let cheek: SIMD2<Float> = t.centre + t.outward * (t.spec.cervicalDepth / 2 + 0.5)
        pts.append(SIMD3<Float>(cheek.x, -h - 3, cheek.y))                    // over the root, cheek side: gum
    }
    guard let r = try? probeScene(pts, retromolarPad: true, thirdMolars: true, on: dev)
    else { expect(false, "probe failed"); return }
    for k in 0..<2 {
        expect(r[3 * k].x < 0 && r[3 * k].y == 1, "wisdom tooth \(k) mid-crown: \(r[3 * k])")
        expect(r[3 * k + 1].x < 0 && r[3 * k + 1].y == 1, "wisdom tooth \(k) lingual crown: \(r[3 * k + 1])")
        expect(r[3 * k + 2].x < 0 && r[3 * k + 2].y == 2, "wisdom tooth \(k) over the root: \(r[3 * k + 2])")
    }
}

test("the pad's distance never claims more room than there is, beyond what the ray allows") {
    guard let dev = device else { expect(false, "no GPU"); return }
    guard wisdom.count == 2 else { expect(false, "the scene has no wisdom teeth"); return }
    // Step 20's test, pointed at the new geometry: pairs of nearby points
    // round the back of both wisdom teeth, outside every surface.
    var a: [SIMD3<Float>] = []
    var b: [SIMD3<Float>] = []
    var rng = SystemRandomNumberGenerator()
    for t in wisdom {
        let behind: SIMD2<Float> = t.centre + t.tangent * (t.spec.width / 2 + padBehind)
        for _ in 0..<40_000 {
            let p = SIMD3<Float>(behind.x + Float.random(in: -20...20, using: &rng),
                                 Float.random(in: -26...4, using: &rng),
                                 behind.y + Float.random(in: -20...20, using: &rng))
            let dir: SIMD3<Float> = simd_normalize(SIMD3<Float>(Float.random(in: -1...1, using: &rng),
                                                                Float.random(in: -1...1, using: &rng),
                                                                Float.random(in: -1...1, using: &rng)))
            a.append(p)
            b.append(p + dir * 0.05)
        }
    }
    guard let da = try? probeScene(a, retromolarPad: true, thirdMolars: true, on: dev),
          let db = try? probeScene(b, retromolarPad: true, thirdMolars: true, on: dev)
    else { expect(false, "probe failed"); return }
    var worst: Float = 0
    for i in 0..<a.count where da[i].y == 2 && db[i].y == 2 && da[i].x > 0 && db[i].x > 0 {
        worst = max(worst, abs(da[i].x - db[i].x) / 0.05)
    }
    print(String(format: "        worst over-report round the pads %.2f; the ray allows %.2f", worst, 1 / stepScale))
    expect(worst * stepScale <= 1.0, "the pad oversteps: \(worst)")
}

test("the pad closes the ridge behind #17 without covering the tooth") {
    guard let dev = device else { expect(false, "no GPU"); return }
    guard let t: PlacedTooth = wisdom.first else { expect(false, "the scene has no #17"); return }
    let top: Float = -t.spec.crownHeight + gumMarginAboveCEJ           // the gum margin
    let behind: SIMD2<Float> = t.centre + t.tangent * (t.spec.width / 2 + padBehind)
    let distal: SIMD2<Float> = t.centre + t.tangent * (t.spec.width / 2 - 1.5)
    let pts: [SIMD3<Float>] = [
        SIMD3<Float>(behind.x, top + 1, behind.y),                     // just above the old gum line
        SIMD3<Float>(t.centre.x, -3, t.centre.y),                      // the middle of the crown
        SIMD3<Float>(distal.x, -2.5, distal.y),                        // the crown's distal end
    ]
    guard let with = try? probeScene(pts, retromolarPad: true, thirdMolars: true, on: dev),
          let without = try? probeScene(pts, thirdMolars: true, on: dev)
    else { expect(false, "probe failed"); return }
    expect(without[0].x > 0 && with[0].x < 0 && with[0].y == 2,
           "behind the molar: \(without[0]) without the pad, \(with[0]) with it")
    expect(with[1].x < 0 && with[1].y == 1, "the crown should still be enamel: \(with[1])")
    expect(with[2].x < 0 && with[2].y == 1, "the distal crown should still be enamel: \(with[2])")
}

test("behind #17 the gum slopes away: no wall, no post, down to below the last frame") {
    guard let dev = device else { expect(false, "no GPU"); return }
    guard let t: PlacedTooth = wisdom.first else { expect(false, "the scene has no #17"); return }
    // The height of the gum's top along the line of the arch, every 0.5 mm
    // from the pad's crest back, read by probing each column top-down. The
    // first pad was a column with straight sides and the gum without one ends
    // in a wall: both drop tens of millimetres in one step. The last frame
    // sees down to about y = −20 here (the camera is 15 mm up and looks down
    // 28° with a ±17° lens, 35 mm out), so that is how far the slope must hold.
    let step: Float = 0.5
    let dy: Float = 0.05
    let columns: Int = 50
    let rows: Int = 600
    var pts: [SIMD3<Float>] = []
    for c in 0..<columns {
        let u: Float = t.spec.width / 2 + padBehind + Float(c) * step
        let xz: SIMD2<Float> = t.centre + t.tangent * u
        for r in 0..<rows { pts.append(SIMD3<Float>(xz.x, 2 - Float(r) * dy, xz.y)) }
    }
    guard let d = try? probeScene(pts, retromolarPad: true, thirdMolars: true, on: dev)
    else { expect(false, "probe failed"); return }
    var heights: [Float] = []
    for c in 0..<columns {
        var h: Float = -30
        for r in 0..<rows where d[c * rows + r].x < 0 && d[c * rows + r].y == 2 {
            h = 2 - Float(r) * dy
            break
        }
        heights.append(h)
    }
    var worst: Float = 0
    var rises: Int = 0
    for c in 1..<columns where heights[c - 1] > -20 {
        worst = max(worst, heights[c - 1] - heights[c])
        if heights[c] > heights[c - 1] + dy { rises += 1 }
    }
    print(String(format: "        gum top behind #17: %.1f mm at the crest, %.1f mm 1 cm back; steepest fall %.2f mm per %.1f mm",
                 heights[0], heights[20], worst, step))
    expect(heights[0] > top(t) , "no pad behind #17: the gum's top there is \(heights[0])")
    expect(rises == 0, "the gum's top rises again \(rises) times behind the crest")
    // 2.5 mm down per 0.5 mm along is 79°: steep, as the pad's end rolls into
    // the floor, but not a wall.
    expect(worst <= 2.5, "the gum drops \(worst) mm in \(step) mm: a wall")
}

/// The gum margin's height on a tooth.
func top(_ t: PlacedTooth) -> Float { -t.spec.crownHeight + gumMarginAboveCEJ }

section("the frames")

test("the first frame faces the midline and the last looks straight at #17") {
    guard let dev = device,
          let first = try? renderMouth(width: 320, height: 180, samples: 1, camera: cameras[0],
                                       retromolarPad: true, thirdMolars: thirdMolars, on: dev).image,
          let last = try? renderMouth(width: 320, height: 180, samples: 1, camera: cameras[travelFrames],
                                      retromolarPad: true, thirdMolars: thirdMolars, on: dev).image
    else { expect(false, "render failed"); return }
    // The arch is its own mirror image and the first camera sits on the
    // midline, so what each pixel sees must mirror left to right.
    var same: Int = 0
    for y in 0..<180 {
        for x in 0..<160 where first.seen(x, y).x == first.seen(319 - x, y).x { same += 1 }
    }
    let mirrored: Double = Double(same) / Double(160 * 180)
    // And straight ahead in the last frame is enamel with a molar's crown.
    let centre: SIMD4<Float> = last.seen(160, 90)
    print(String(format: "        first frame %.1f%% mirror-symmetric; last frame's centre sees material %d, kind %d",
                 mirrored * 100, Int(centre.x), Int(centre.z)))
    expect(mirrored > 0.99, "the first frame is only \(mirrored) symmetric")
    expect(centre.x == 1 && Int(centre.z) == CrownKind.secondMolar.rawValue, "centre sees \(centre)")

    // A third molar is drawn with a second molar's crown, so the kind alone
    // cannot tell #17 from #18. Follow the centre ray through the same scene
    // to the first enamel it meets, and check that point is on #17's crown.
    guard let t: PlacedTooth = wisdom.first else { expect(false, "the scene has no #17"); return }
    let cam: Camera = cameras[travelFrames]
    let along: [Float] = (0..<6000).map { Float($0) * 0.01 }
    let ray: [SIMD3<Float>] = along.map { cam.position + cam.forward * $0 }
    guard let d = try? probeScene(ray, retromolarPad: true, thirdMolars: thirdMolars, on: dev),
          let hit: Int = d.firstIndex(where: { $0.x < 0 })
    else { expect(false, "the centre ray hits nothing"); return }
    let p: SIMD3<Float> = ray[hit]
    let rel = SIMD2<Float>(p.x, p.z) - t.centre
    let u: Float = simd_dot(rel, t.tangent)
    let v: Float = simd_dot(rel, t.outward)
    print(String(format: "        centre ray meets material %d at %.1f mm, %.2f mm along #17 and %.2f mm out from its centre",
                 Int(d[hit].y), along[hit], u, v))
    expect(d[hit].y == 1 && abs(u) < t.spec.width / 2 && abs(v) <= t.spec.depth / 2 + 0.5,
           "the centre ray meets material \(d[hit].y) at \(p), not #17's crown")
}

finish()
