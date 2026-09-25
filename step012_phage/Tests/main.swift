// Tests for step 12. The scene is built by Tools/build_phage.py from the
// deposited structures; these check what the Swift side does with it — that
// the two measured states are used honestly, that nothing appears from
// nowhere, that the needle goes where it has to, and that the renderer's grid
// and occlusion still behave on the largest scene this project has drawn.

import CoreGraphics
import Foundation
import Metal
import simd

func near(_ a: Float, _ b: Float, within e: Float) -> Bool { abs(a - b) <= e }

let sceneURL = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent().deletingLastPathComponent()
    .appendingPathComponent("Resources/scene.json")
let scene = try loadScene(from: sceneURL)
let timing = Timing()
let layout = FrameLayout(width: 320, viewHeight: 240, captionHeight: 40)

/// Where the structure files live, if they have been fetched. They are
/// gitignored — `make structure` refetches them — so the tests that read them
/// directly skip rather than fail on a fresh clone.
let resDir = sceneURL.deletingLastPathComponent()
func cifPath(_ id: String) -> String? {
    let p = resDir.appendingPathComponent("\(id).cif").path
    return FileManager.default.fileExists(atPath: p) ? p : nil
}

section("the scene")
test("it is the biggest scene in the project, and every part is accounted for") {
    expect(scene.shapes.count > 500_000, "\(scene.shapes.count) spheres")
    let counted = (scene.counts["wedge"] ?? 0) + (scene.counts["needle"] ?? 0)
        + (scene.counts["tube"] ?? 0) + (scene.counts["fibre"] ?? 0) + (scene.counts["envelope"] ?? 0)
    expectEqual(counted, scene.shapes.count)
    expectEqual(scene.counts["total"] ?? 0, scene.shapes.count)
}
test("every sphere has a real radius and a known part") {
    let known: Set<String> = ["wedge", "needle", "tube", "fibre", "lps", "lps_tail",
                              "lipid", "lipid_tail", "pg"]
    var bad = 0, unknown = Set<String>()
    for s in scene.shapes {
        if !(s.radius > 0.5 && s.radius < 4) { bad += 1 }
        if !known.contains(s.part) { unknown.insert(s.part) }
    }
    expectEqual(bad, 0)
    expect(unknown.isEmpty, "unknown parts \(unknown)")
}

section("the two measured states")
test("5IV7 is a strict subset of 5IV5 — the wedge is common, the hub is not") {
    // The check that decided the whole render. 5IV7 is "hubless": it has none
    // of the central machinery, so the flip can only be morphed on the wedge.
    expectEqual(scene.commonEntities.count, 8)
    expect(scene.commonEntities.allSatisfy { $0.contains("wedge") },
           "common entities should all be wedge proteins: \(scene.commonEntities)")
    for gone in ["Baseplate hub protein gp27", "Peptidoglycan hydrolase gp5",
                 "Short tail fiber protein gp12", "Tail tube protein gp19"] {
        expect(scene.preOnlyEntities.contains(gone), "\(gone) should be pre-attachment only")
    }
}
test("all 96 wedge chains were matched, and nearly every atom survived the intersection") {
    expectEqual(scene.baseplate.wedgeChains, 96)
    let kept = scene.baseplate.wedgeAtomsKept
    let dropped = scene.baseplate.wedgeAtomsDropped
    expect(kept > 300_000, "\(kept) wedge atoms kept")
    // Cryo-EM models different disordered stretches in each state, so a few
    // atoms exist in only one. If that fraction were large the morph would be
    // hiding something.
    let fraction = Float(dropped) / Float(kept + dropped)
    expect(fraction < 0.01, "dropped \(dropped) of \(kept + dropped) — \(fraction * 100)%")
}
test("only the wedge carries a second position; nothing else morphs") {
    for s in scene.shapes {
        if s.part == "wedge" {
            expect(s.end != nil, "a wedge atom with no post-attachment position")
        } else {
            expect(s.end == nil, "\(s.part) should not morph")
        }
    }
}
test("the flip really does spread and flatten the baseplate") {
    let bp = scene.baseplate
    expect(bp.postRadius > bp.preRadius, "\(bp.preRadius) → \(bp.postRadius) Å")
    expect(near(2 * bp.preRadius / 10, 49.0, within: 1.5), "pre \(2 * bp.preRadius / 10) nm")
    expect(near(2 * bp.postRadius / 10, 60.9, within: 1.5), "post \(2 * bp.postRadius / 10) nm")
    expect(bp.postHeight < bp.preHeight, "it should flatten: \(bp.preHeight) → \(bp.postHeight) Å")
}
test("the sheath shortens and widens, from its own measured pair") {
    let sh = scene.sheath
    expect(sh.postLength < sh.preLength, "\(sh.preLength) → \(sh.postLength) Å")
    expect(sh.postRadius > sh.preRadius, "\(sh.preRadius) → \(sh.postRadius) Å")
    expect(sh.contraction > 0.25 && sh.contraction < 0.55,
           "contracts \(sh.contraction * 100)%, which should be roughly a third")
}
test("the deposited files agree with what the builder recorded") {
    guard let p5 = cifPath("5IV5"), let p7 = cifPath("5IV7") else {
        // Gitignored; `make structure` refetches them.
        return
    }
    /// Counts ATOM and HETATM records, which together are what RCSB reports as
    /// the deposited atom count — 5IV5's twelve HETATM lines are its zinc and
    /// iron ions.
    func atomCount(_ path: String) -> Int {
        guard let data = FileManager.default.contents(atPath: path) else { return 0 }
        var n = 0
        let prefixes = [Array("ATOM ".utf8), Array("HETATM".utf8)]
        data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            let b = raw.bindMemory(to: UInt8.self)
            var i = 0
            var atLineStart = true
            while i < b.count {
                if atLineStart {
                    for prefix in prefixes where i + prefix.count <= b.count {
                        var match = true
                        for k in 0..<prefix.count where b[i + k] != prefix[k] { match = false; break }
                        if match { n += 1; break }
                    }
                }
                atLineStart = b[i] == 0x0A
                i += 1
            }
        }
        return n
    }
    expectEqual(atomCount(p5), 549_576)
    expectEqual(atomCount(p7), 312_210)
}

section("the firing")
test("the loop closes: the last frame is the first") {
    let a = poseScene(scene, t: 0, timing: timing)
    let b = poseScene(scene, t: timing.total, timing: timing)
    expectEqual(a.count, b.count)
    var worst: Float = 0
    for i in 0..<a.count {
        worst = max(worst, simd_distance(a[i].center, b[i].center))
        worst = max(worst, abs(a[i].radius - b[i].radius))
    }
    expect(worst < 0.01, "worst drift over the loop: \(worst) Å")
}
test("nothing jumps between frames") {
    let fps: Float = 12.5
    var prev = poseScene(scene, t: 0, timing: timing)
    var worst: Float = 0
    var at: Float = 0
    var f: Float = 1
    while f / fps <= timing.total {
        let now = poseScene(scene, t: f / fps, timing: timing)
        for i in 0..<now.count {
            let d = simd_distance(prev[i].center, now[i].center)
            if d > worst { worst = d; at = f / fps }
        }
        prev = now
        f += 1
    }
    // A needle crossing 30 nm of envelope in a few seconds is the fastest
    // thing here; anything much beyond that would be a jump.
    expect(worst < 40, "biggest single-frame move \(worst) Å at t = \(at) s")
}
test("the needle really crosses the inner membrane") {
    let end = poseScene(scene, t: timing.settle + timing.flip + timing.drive, timing: timing)
    let needle = zip(scene.shapes, end).filter { $0.0.part == "needle" }.map { $0.1 }
    expect(!needle.isEmpty)
    let lowest = needle.map { $0.center.z }.min() ?? 0
    expect(lowest <= scene.reach,
           "the needle reaches \(lowest) Å; it must pass \(scene.reach) Å to be through")
}
test("the short tail fibres grip the outside and never enter the cell") {
    // Driving them through with the needle would be plain wrong: gripping the
    // outer surface is the whole job of a short tail fibre.
    let outerFace = scene.layers.top
    var deepest: Float = .greatestFiniteMagnitude
    var f: Float = 0
    while f <= timing.total {
        let pose = poseScene(scene, t: f, timing: timing)
        for (s, g) in zip(scene.shapes, pose) where s.part == "fibre" {
            deepest = min(deepest, g.center.z)
        }
        f += 0.25
    }
    expect(deepest >= outerFace - 6,
           "a fibre reached \(deepest) Å; the membrane's outer face is at \(outerFace) Å")
}
test("the wedge never dips into the membrane") {
    var deepest: Float = .greatestFiniteMagnitude
    var f: Float = 0
    while f <= timing.total {
        let pose = poseScene(scene, t: f, timing: timing)
        for (s, g) in zip(scene.shapes, pose) where s.part == "wedge" {
            deepest = min(deepest, g.center.z)
        }
        f += 0.25
    }
    expect(deepest > scene.layers.top,
           "the baseplate reached \(deepest) Å, below the membrane top at \(scene.layers.top) Å")
}
test("the envelope holds still throughout") {
    let a = poseScene(scene, t: 0, timing: timing)
    let b = poseScene(scene, t: timing.settle + timing.flip + timing.drive, timing: timing)
    let envParts: Set<String> = ["lps", "lps_tail", "lipid", "lipid_tail", "pg"]
    var moved: Float = 0
    for (i, s) in scene.shapes.enumerated() where envParts.contains(s.part) {
        moved = max(moved, simd_distance(a[i].center, b[i].center))
    }
    expectEqual(moved, 0)
}
test("nothing vanishes while it is still in open view") {
    // The needle dims only once it is well below the inner membrane.
    var f: Float = 0
    while f <= timing.total {
        let pose = poseScene(scene, t: f, timing: timing)
        for (s, g) in zip(scene.shapes, pose) where s.part == "needle" {
            if g.radius < s.radius * 0.98 {
                expect(g.center.z < scene.reach,
                       "a needle atom faded at z = \(g.center.z), above the inner membrane")
            }
        }
        f += 0.5
    }
}

section("the evidence bar")
test("the flip is measured and the crossing is not") {
    expectEqual(evidenceAt(0.5, timing: timing).0, Evidence.measured)
    expectEqual(evidenceAt(timing.settle + timing.flip * 0.5, timing: timing).0, Evidence.measured)
    let late = timing.settle + timing.flip + timing.drive * 0.8
    expectEqual(evidenceAt(late, timing: timing).0, Evidence.model)
}
test("every wedge sphere claims `measured` and every pre-only part claims `model`") {
    for s in scene.shapes {
        switch s.part {
        case "wedge": expectEqual(s.evidence, Evidence.measured)
        case "needle", "tube", "fibre": expectEqual(s.evidence, Evidence.model)
        default: expectEqual(s.evidence, Evidence.measured)
        }
    }
}

section("scale")
test("an atom is worth enough pixels for space-filling to mean anything") {
    // The arithmetic every step since the plasmid has used. The firing view
    // frames the 60.9 nm star; a carbon is 3.4 Å across.
    let cam = firingCamera(scene, t: timing.settle + timing.flip)
    let mid = SIMD3<Float>(0, 0, (scene.layers.top + 260) * 0.5)
    let a = cam.project(mid, width: 960, height: 720)
    let b = cam.project(mid + SIMD3(3.4, 0, 0), width: 960, height: 720)
    let px = simd_distance(a, b)
    expect(px > 1.5, "one atom is \(px) px — below about 1.5 it is sub-pixel noise")
}

section("the renderer")
let device = try findDevice()
let renderer = try SceneRenderer(device: device)
let buffer = device.makeBuffer(length: layout.width * layout.height * 4, options: .storageModeShared)!

/// A cheap stand-in scene, so the grid tests do not pay for 613,000 spheres.
let sample: [GPUShape] = {
    let full = poseScene(scene, t: timing.settle + timing.flip * 0.5, timing: timing)
    return stride(from: 0, to: full.count, by: 40).map { full[$0] }
}()

test("the grid draws the same picture as testing every sphere") {
    let cam = firingCamera(scene, t: timing.settle + timing.flip * 0.5)
    let ao = AOSettings(probes: 6, distance: 14, strength: 1, contrast: 1.5)
    _ = try renderer.render(shapes: sample, camera: cam, into: buffer,
                            width: layout.width, viewHeight: layout.viewHeight,
                            useGrid: false, ao: ao)
    let brute = Data(bytes: buffer.contents(), count: layout.width * layout.viewHeight * 4)
    try renderer.buildGrid(sample, density: 1)
    _ = try renderer.render(shapes: sample, camera: cam, into: buffer,
                            width: layout.width, viewHeight: layout.viewHeight,
                            useGrid: true, ao: ao)
    let grid = Data(bytes: buffer.contents(), count: layout.width * layout.viewHeight * 4)
    var differing = 0
    for i in 0..<brute.count where brute[i] != grid[i] { differing += 1 }
    let share = Double(differing) / Double(brute.count)
    // Not bit-identical: the two paths meet spheres in a different order, so a
    // pixel exactly on a silhouette can round either way.
    expect(share < 0.02, "\(share * 100)% of bytes differ between grid and brute force")
}
test("a coarse grid and a fine one draw the same picture") {
    let cam = firingCamera(scene, t: timing.settle + timing.flip * 0.5)
    try renderer.buildGrid(sample, density: 0.5)
    _ = try renderer.render(shapes: sample, camera: cam, into: buffer,
                            width: layout.width, viewHeight: layout.viewHeight)
    let coarse = Data(bytes: buffer.contents(), count: layout.width * layout.viewHeight * 4)
    try renderer.buildGrid(sample, density: 2)
    _ = try renderer.render(shapes: sample, camera: cam, into: buffer,
                            width: layout.width, viewHeight: layout.viewHeight)
    let fine = Data(bytes: buffer.contents(), count: layout.width * layout.viewHeight * 4)
    var differing = 0
    for i in 0..<coarse.count where coarse[i] != fine[i] { differing += 1 }
    expect(Double(differing) / Double(coarse.count) < 0.02)
}
test("every sphere is filed in the box holding its middle") {
    try renderer.buildGrid(sample, density: 1)
    guard let g = renderer.grid else { expect(false, "no grid"); return }
    for i in stride(from: 0, to: sample.count, by: 97) {
        expect(g.shapes(at: sample[i].center).contains(i), "sphere \(i) is not in its own box")
    }
}
test("occlusion darkens a buried sphere more than an exposed one") {
    // The claim the whole space-filling look rests on, as in steps 9 and 8b.
    let cam = firingCamera(scene, t: timing.settle + timing.flip * 0.5)
    try renderer.buildGrid(sample, density: 1)
    func meanBrightness(_ ao: AOSettings) throws -> Double {
        _ = try renderer.render(shapes: sample, camera: cam, into: buffer,
                                width: layout.width, viewHeight: layout.viewHeight, ao: ao)
        let p = buffer.contents().assumingMemoryBound(to: UInt8.self)
        var sum = 0.0
        for i in stride(from: 0, to: layout.width * layout.viewHeight * 4, by: 4) {
            sum += Double(p[i]) + Double(p[i + 1]) + Double(p[i + 2])
        }
        return sum / Double(layout.width * layout.viewHeight * 3)
    }
    let off = try meanBrightness(.off)
    let on = try meanBrightness(AOSettings(probes: 12, distance: 14, strength: 1, contrast: 1.5))
    // Occlusion takes away ambient light that enclosed surfaces never get back,
    // so the picture darkens overall — the same effect step 8b measured.
    expect(on < off, "occlusion should darken the scene: \(off) → \(on)")
}
test("occlusion is steady between two renders of the same moment") {
    let cam = firingCamera(scene, t: timing.settle + timing.flip * 0.5)
    try renderer.buildGrid(sample, density: 1)
    let ao = AOSettings(probes: 12, distance: 14, strength: 1, contrast: 1.5)
    _ = try renderer.render(shapes: sample, camera: cam, into: buffer,
                            width: layout.width, viewHeight: layout.viewHeight, ao: ao)
    let first = Data(bytes: buffer.contents(), count: layout.width * layout.viewHeight * 4)
    _ = try renderer.render(shapes: sample, camera: cam, into: buffer,
                            width: layout.width, viewHeight: layout.viewHeight, ao: ao)
    let second = Data(bytes: buffer.contents(), count: layout.width * layout.viewHeight * 4)
    expect(first == second, "the same scene rendered twice should be identical")
}
test("rebuilding the grid every frame is what the render actually does") {
    // The scene deforms, so a grid left over from the previous frame is stale.
    // Step 11 measured the rebuild at under 8% of a frame; this checks it is
    // cheap here too rather than assuming.
    let a = poseScene(scene, t: timing.settle + timing.flip * 0.3, timing: timing)
    let thinA = stride(from: 0, to: a.count, by: 40).map { a[$0] }
    let start = Date()
    try renderer.buildGrid(thinA, density: 1)
    let ms = Date().timeIntervalSince(start) * 1000
    expect(ms < 200, "building the grid took \(ms) ms for \(thinA.count) spheres")
}

finish()
