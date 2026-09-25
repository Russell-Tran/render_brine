// Tests for step 8a. Sequence tests check that the plasmid really is pGLO;
// geometry tests check that the circle really is closed, right-handed B-DNA;
// grid tests check that the acceleration structure finds exactly what brute
// force finds, which is the only thing that makes a 4,000× speedup worth
// having; the rest check the level of detail, the loops and the encoder.

import CoreGraphics
import Foundation
import ImageIO
import Metal
import simd

let pglo = try loadPlasmid(from: URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent().deletingLastPathComponent()
    .appendingPathComponent("Resources/pglo.json"))
let gfp = pglo.features.first { $0.name == "GFP" }!
let diveTarget = pglo.index(ofBasePair: gfp.start)
let stops: Set<String> = ["TAA", "TAG", "TGA"]

func codons(of f: Feature) -> [String] {
    var s = pglo.bases(from: f.start, count: f.length)
    if f.strand < 0 {
        s = String(s.reversed().map { c -> Character in
            switch c { case "A": return "T"; case "T": return "A"; case "G": return "C"; default: return "G" }
        })
    }
    return stride(from: 0, to: s.count, by: 3).map { i in
        String(Array(s)[i..<min(i + 3, s.count)])
    }
}

section("the sequence really is pGLO")
test("5,371 base pairs of ACGT") {
    expectEqual(pglo.length, 5371)
    expectEqual(pglo.sequence.count, 5371)
    expect(pglo.sequence.allSatisfy { "ACGT".utf8.contains($0) })
}
test("every annotated feature lies inside the plasmid and none overlap") {
    var owner = [Int](repeating: -1, count: pglo.length)
    for (i, f) in pglo.features.enumerated() {
        expect(f.start >= 1 && f.end <= pglo.length, "\(f.name) runs \(f.start)–\(f.end)")
        expect(f.start <= f.end, "\(f.name) is backwards")
        for bp in (f.start - 1)..<f.end {
            expect(owner[bp] == -1, "\(f.name) overlaps \(owner[bp] < 0 ? "" : pglo.features[owner[bp]].name)")
            owner[bp] = i
        }
    }
}
test("araC, GFP and bla are real open reading frames") {
    for name in ["araC", "GFP", "bla"] {
        let f = pglo.features.first { $0.name == name }!
        let c = codons(of: f)
        expectEqual(f.length % 3, 0)
        expectEqual(c.first, "ATG")
        expect(!c.contains { stops.contains($0) }, "\(name) has an internal stop codon")
    }
}
test("GFP starts at 1,342 and its first codons spell the protein's start") {
    expectEqual(gfp.start, 1342)
    expectEqual(gfp.length, 717)
    // M A S K G E E L — the canonical N-terminus of green fluorescent protein.
    expectEqual(Array(codons(of: gfp)[0..<8]), ["ATG", "GCT", "AGC", "AAA", "GGA", "GAA", "GAA", "CTT"])
}
test("about two thirds of the plasmid is doing the work, and the rest is grey") {
    let lit = pglo.features.filter { $0.color != "grey" }.reduce(0) { $0 + $1.length }
    expect(lit > 3000 && lit < 3800, "\(lit) bp lit")
    expectEqual(Set(pglo.features.map { $0.color }), ["gfp", "switch", "bla", "ori", "grey"])
}

section("the circle: real B-DNA, closed")
test("the ring's size comes from the plasmid's own length") {
    expect(abs(pglo.circumference - 5371 * 3.4) < 0.01)
    expect(abs(pglo.ringRadius - 2906.4) < 1, "radius \(pglo.ringRadius) Å")
    expect(abs(pglo.circumference / 10_000 - 1.826) < 0.01, "\(pglo.circumference / 10_000) µm around")
}
test("a closed circle needs a whole number of turns, and 512 is the nearest") {
    expectEqual(pglo.helicalTurns, 512)
    // 5371 / 10.5 = 511.5: the molecule cannot be both closed and relaxed,
    // which is exactly why real plasmids are supercoiled.
    let ideal = Float(pglo.length) / 10.5
    expect(abs(ideal - 511.5) < 0.05, "\(ideal) ideal turns")
    expect(abs(pglo.basePairsPerTurnDrawn - 10.49) < 0.01, "\(pglo.basePairsPerTurnDrawn) bp per turn")
}
test("consecutive base pairs are one rise apart, all the way round including the seam") {
    var worst: Float = 0
    for i in 0..<pglo.length {
        let a = pglo.frame(at: i).origin, b = pglo.frame(at: (i + 1) % pglo.length).origin
        worst = max(worst, abs(simd_distance(a, b) - 3.4))
    }
    // Straight-line distance, so it is a chord of the ring rather than the arc,
    // and single precision at a 2,906 Å radius is worth a thousandth either way.
    expect(worst < 0.01, "worst step is off by \(worst) Å")
}
test("the helix is right-handed, and twists 34.3° per step") {
    var total: Float = 0
    for i in 0..<12 {
        let a = pglo.frame(at: i), b = pglo.frame(at: i + 1)
        // The turn from one base pair's y axis to the next, about the helix axis.
        let angle = atan2(simd_dot(simd_cross(a.y, b.y), a.tangent), simd_dot(a.y, b.y))
        expect(angle > 0, "step \(i) twists \(angle * 180 / .pi)°, so the helix is left-handed")
        total += angle
    }
    let perStep = total / 12 * 180 / .pi
    expect(abs(perStep - 34.3) < 0.2, "\(perStep)° per step")
}
test("every base pair's frame is orthonormal and right-handed") {
    for i in stride(from: 0, to: pglo.length, by: 97) {
        let f = pglo.frame(at: i)
        for v in [f.tangent, f.y, f.z] { expect(abs(simd_length(v) - 1) < 1e-4) }
        expect(abs(simd_dot(f.tangent, f.y)) < 1e-4)
        expect(abs(simd_dot(f.y, f.z)) < 1e-4)
        expect(simd_distance(simd_cross(f.tangent, f.y), f.z) < 1e-4, "not right-handed at \(i)")
    }
}
test("the backbone sits 9 Å out, as the crystal templates say, with two strands apart") {
    expect(pglo.backboneRadius > 8.5 && pglo.backboneRadius < 9.9, "\(pglo.backboneRadius) Å")
    let apart = abs(pglo.strandPhase[0] - pglo.strandPhase[1]) * 180 / .pi
    // The two strands are not opposite each other: that asymmetry is what
    // makes one groove major and the other minor.
    expect(apart > 90 && apart < 170, "\(apart)° apart")
    for i in stride(from: 0, to: pglo.length, by: 313) {
        let f = pglo.frame(at: i)
        for s in 0..<2 {
            expect(abs(simd_distance(pglo.backbonePoint(at: i, strand: s), f.origin) - pglo.backboneRadius) < 1e-3)
        }
    }
}

section("three levels of detail")
let ringScene = buildScene(pglo, atomWindow: 0..<0)
let window = (diveTarget - 40)..<(diveTarget + 41)
var diveScene = buildScene(pglo, atomWindow: window)

test("the ring alone is one tube segment per base pair") {
    expectEqual(ringScene.tubeCount, pglo.length)
    expectEqual(ringScene.atomCount, 0)
}
test("atoms are built only for the window, at ~63 per base pair") {
    expectEqual(diveScene.tubeCount, pglo.length)
    let spheres = diveScene.shapes[diveScene.tubeCount...].filter { $0.isSphere }.count
    expect(spheres > 60 * window.count && spheres < 66 * window.count, "\(spheres) atoms for \(window.count) bp")
}
test("each tier switches on where its detail stops being sub-pixel") {
    // An atom is 3.4 Å across; a frame 960 px wide sees 0.8578 × distance Å.
    func atomPixels(_ d: Float) -> Float { 3.4 / (0.8578 * d / 960) }
    expectEqual(detailLevel(cameraDistance: 9000), 0)
    expect(atomPixels(spaceFillStart) > 1.5 && atomPixels(spaceFillStart) < 2.5,
           "space-filling starts when an atom is \(atomPixels(spaceFillStart)) px")
    expect(atomPixels(spaceFillFull) > 3.5, "and is complete at \(atomPixels(spaceFillFull)) px")
    expect(detailLevel(cameraDistance: spaceFillFull) >= 0.999)
    expect(detailLevel(cameraDistance: ballFull) >= 1.999)
    // Monotonic: coming closer never takes detail away.
    var previous: Float = 0
    for d in stride(from: Float(9000), through: 50, by: -50) {
        let level = detailLevel(cameraDistance: d)
        expect(level >= previous - 1e-5, "detail fell from \(previous) to \(level) at \(d) Å")
        previous = level
    }
}
test("the tube is drawn over life size far out and at life size by the time atoms appear") {
    diveScene.setDetail(0)
    let far = diveScene.shapes[0].radius
    diveScene.setDetail(1)
    let near = diveScene.shapes[0].radius
    expect(abs(far / near - Double(tubeExaggeration).float) < 0.01, "\(far) Å then \(near) Å")
    expect(abs(near - duplexRadius) < 1e-4, "life size is \(duplexRadius) Å, drew \(near)")
}
test("inside the window the tube gets out of the way; outside it never does") {
    diveScene.setDetail(1)
    let inside = diveScene.shapes[window.lowerBound + 5].radius
    let outside = diveScene.shapes[10].radius
    expectEqual(inside, 0)
    expect(outside > 0, "the rest of the ring vanished")
}
test("space-filling uses van der Waals radii and draws no bonds; sticks are the other way round") {
    let atoms = diveScene.tubeCount..<diveScene.shapes.count
    diveScene.setDetail(1)
    let spaceSpheres = atoms.filter { diveScene.shapes[$0].isSphere && diveScene.shapes[$0].radius > 1 }.count
    let spaceSticks = atoms.filter { !diveScene.shapes[$0].isSphere && diveScene.shapes[$0].radius > 0 }.count
    diveScene.setDetail(2)
    let ballSpheres = atoms.filter { diveScene.shapes[$0].isSphere && diveScene.shapes[$0].radius > 0 }.count
    let ballSticks = atoms.filter { !diveScene.shapes[$0].isSphere && diveScene.shapes[$0].radius > 0 }.count
    expect(spaceSpheres > 0 && spaceSticks == 0, "space-filling drew \(spaceSticks) sticks")
    expect(ballSpheres == spaceSpheres && ballSticks > 0, "ball-and-stick drew \(ballSticks) sticks")
    // Every sphere shrinks from its van der Waals size to a ball.
    for i in atoms where diveScene.shapes[i].isSphere {
        expect(diveScene.ballRadius[i] < diveScene.spaceRadius[i])
    }
}
test("the atoms are real: bond lengths and Watson–Crick pairs from the crystal") {
    for (kind, t) in pglo.templates {
        expectEqual(t.atoms.filter { $0.name == "P" }.count, 2)
        let pairs = kind == "AT" || kind == "TA" ? 2 : 3
        expectEqual(t.hbonds.count, pairs)
        for b in t.bonds {
            let d = simd_distance(t.atoms[b.a].position, t.atoms[b.b].position)
            expect(d > 0.9 && d < 1.8, "\(kind): a bond is \(d) Å")
        }
        for h in t.hbonds {
            let d = simd_distance(t.atoms[h.hydrogen].position, t.atoms[h.acceptor].position)
            expect(d > 1.4 && d < 2.6, "\(kind): H to acceptor is \(d) Å")
            expectEqual(t.atoms[h.hydrogen].element, "H")
        }
    }
}
test("neighbouring nucleotides join up: O3'–P across every step of the window") {
    var worst: Float = 0
    for i in window.dropLast() {
        guard let a = diveScene.atomAnchor[i], let b = diveScene.atomAnchor[i + 1] else { continue }
        let ta = pglo.templates[pglo.templateKey(at: i)]!, tb = pglo.templates[pglo.templateKey(at: i + 1)]!
        func find(_ t: BasePairTemplate, _ base: Int, _ strand: Int, _ name: String) -> SIMD3<Float>? {
            guard let k = t.atoms.firstIndex(where: { $0.strand == strand && $0.name == name }) else { return nil }
            let s = diveScene.shapes[base + k]
            return SIMD3(s.a.x, s.a.y, s.a.z)
        }
        if let o = find(ta, a, 0, "O3'"), let p = find(tb, b, 0, "P") {
            worst = max(worst, abs(simd_distance(o, p) - 1.6))
        }
    }
    // The templates come from a 3.33 Å crystal placed on an ideal 3.4 Å ring,
    // so the joints stretch a little; this is how much.
    expect(worst < 0.4, "an O3'–P join is off by \(worst) Å")
}

section("the uniform grid finds what brute force finds")
let device = try findDevice()
let renderer = try SceneRenderer(device: device)
let view = (width: 320, height: 200)
let bytes = view.width * view.height * 4
let bufferA = device.makeBuffer(length: bytes, options: .storageModeShared)!
let bufferB = device.makeBuffer(length: bytes, options: .storageModeShared)!
func pixels(_ b: MTLBuffer) -> [UInt8] {
    Array(UnsafeBufferPointer(start: b.contents().assumingMemoryBound(to: UInt8.self), count: bytes))
}

test("every shape is filed in the boxes it overlaps") {
    let grid = UniformGrid(shapes: diveScene.widest(), density: 1)
    expect(grid.cellCount > 1 && grid.items.count >= diveScene.shapes.count)
    for (i, s) in diveScene.widest().enumerated() where i % 997 == 0 {
        let centre = s.isSphere ? SIMD3(s.a.x, s.a.y, s.a.z)
                                : (SIMD3(s.a.x, s.a.y, s.a.z) + SIMD3(s.b.x, s.b.y, s.b.z)) / 2
        expect(grid.shapes(at: centre).contains(i), "shape \(i) is not in the box holding its middle")
    }
}
test("a box outside everything is empty, and a ray that misses gets the background") {
    var scene = buildScene(pglo, atomWindow: 0..<0)
    scene.setDetail(0)
    try renderer.buildGrid(scene.widest(), density: 1)
    let grid = renderer.grid!
    expect(grid.shapes(at: SIMD3(0, 0, 0)).isEmpty, "the middle of the ring should be empty space")
    // Pointed away from the plasmid entirely.
    let away = Camera(target: SIMD3(0, 50_000, 0), direction: SIMD3(0, 1, 0), distance: 1000, fov: 30)
    try renderer.render(shapes: scene.shapes, camera: away, into: bufferA,
                        width: view.width, viewHeight: view.height, samplesPerSide: 1, useGrid: true)
    let top = pixels(bufferA)
    expect(abs(Int(top[0]) - 140) <= 3 && abs(Int(top[1]) - 200) <= 3 && abs(Int(top[2]) - 235) <= 3,
           "the top of the gradient is \(top[0]), \(top[1]), \(top[2])")
}
test("grid and brute force draw the same picture, from three viewpoints") {
    var scene = buildScene(pglo, atomWindow: (diveTarget - 12)..<(diveTarget + 13))
    try renderer.buildGrid(scene.widest(), density: 2)
    let target = pglo.frame(at: diveTarget).origin
    let views: [(String, Camera, Float)] = [
        ("the whole ring", Camera.orbit(target: .zero, distance: 9193, yaw: 0, pitch: 35, fov: 30), 0),
        ("space-filling", Camera(target: target, direction: SIMD3(1, 0.4, 0), distance: 600, fov: 30), 1),
        ("ball-and-stick", Camera(target: target, direction: SIMD3(1, 0.4, 0), distance: 90, fov: 30), 2),
    ]
    for (name, camera, level) in views {
        scene.setDetail(level)
        try renderer.render(shapes: scene.shapes, camera: camera, into: bufferA,
                            width: view.width, viewHeight: view.height, samplesPerSide: 2, useGrid: false)
        let brute = pixels(bufferA)
        try renderer.render(shapes: scene.shapes, camera: camera, into: bufferB,
                            width: view.width, viewHeight: view.height, samplesPerSide: 2, useGrid: true)
        let grid = pixels(bufferB)
        var differing = 0, worst = 0
        for i in 0..<bytes where brute[i] != grid[i] {
            differing += 1
            worst = max(worst, abs(Int(brute[i]) - Int(grid[i])))
        }
        // Not bit-identical: the two paths meet shapes in a different order, so
        // a pixel exactly on a silhouette can round either way. Anything more
        // than a handful of edge pixels would mean the grid is losing shapes.
        let share = Double(differing) / Double(bytes) * 100
        expect(share < 0.2, "\(name): \(differing) bytes differ (\(share)%), worst by \(worst)")
        expect(worst < 60, "\(name): a pixel differs by \(worst), too much for an edge")
    }
}
test("a finer grid gives the same picture as a coarse one") {
    var scene = buildScene(pglo, atomWindow: 0..<0)
    scene.setDetail(0)
    let camera = Camera.orbit(target: .zero, distance: 9193, yaw: 20, pitch: 35, fov: 30)
    try renderer.buildGrid(scene.widest(), density: 0.5)
    try renderer.render(shapes: scene.shapes, camera: camera, into: bufferA,
                        width: view.width, viewHeight: view.height, samplesPerSide: 2)
    let coarse = pixels(bufferA)
    try renderer.buildGrid(scene.widest(), density: 4)
    try renderer.render(shapes: scene.shapes, camera: camera, into: bufferB,
                        width: view.width, viewHeight: view.height, samplesPerSide: 2)
    let fine = pixels(bufferB)
    let differing = (0..<bytes).filter { coarse[$0] != fine[$0] }.count
    expect(Double(differing) / Double(bytes) < 0.002, "\(differing) bytes differ between grid resolutions")
}

section("the loops")
test("the ring comes back exactly where it started after one turn") {
    let a = Camera.orbit(target: .zero, distance: 9193, yaw: 0, pitch: 35, fov: 30, lightSpin: 0)
    let b = Camera.orbit(target: .zero, distance: 9193, yaw: 360, pitch: 35, fov: 30, lightSpin: radians(360))
    expect(simd_distance(a.origin, b.origin) < 0.01, "\(a.origin) then \(b.origin)")
    expect(simd_distance(a.forward, b.forward) < 1e-5)
}
test("the whole ring stays inside the frame all the way round") {
    for yaw in stride(from: Float(0), to: 360, by: 15) {
        let camera = Camera.orbit(target: .zero, distance: 9193, yaw: yaw, pitch: 35, fov: 30)
        for i in stride(from: 0, to: pglo.length, by: 37) {
            let p = camera.project(pglo.frame(at: i).origin, width: 960, height: 600)
            expect(p.x > 10 && p.x < 950 && p.y > 10 && p.y < 590,
                   "at yaw \(yaw), base pair \(i) projects to \(p)")
        }
    }
}

section("renderer and encoder")
test("the palette keeps a shade of every colour that means something") {
    // A spread of background blues, as a real frame would give it: median cut
    // cannot split a box of identical colors, so one flat color is no test.
    var blues: [RGB] = []
    for i in 0..<2000 {
        let r: Int = 8 + i % 130
        let g: Int = 40 + i % 160
        let b: Int = 90 + i % 145
        blues.append(RGB(UInt8(r), UInt8(g), UInt8(b)))
    }
    let palette = plasmidPalette(samples: blues, count: 96)
    expectEqual(palette.count, 96)
    for key in ["gfp", "switch", "bla", "ori"] {
        let want = featureColor(key) * 255
        let nearest = palette.map { p -> Float in
            simd_distance(SIMD3(Float(p.x), Float(p.y), Float(p.z)), want)
        }.min()!
        expect(nearest < 40, "\(key) is \(nearest) away from its nearest palette entry")
    }
}
test("the scene outgrows what setBytes could ever have carried") {
    let scene = buildScene(pglo, atomWindow: (diveTarget - 300)..<(diveTarget + 301))
    let bytes = scene.shapes.count * MemoryLayout<GPUShape>.stride
    expect(bytes > 4096 * 1000, "only \(bytes) bytes")
}
test("the GIF writer makes a looping file Apple can read back") {
    let url = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("step8a-test.gif")
    let palette = plasmidPalette(samples: (0..<600).map { _ in RGB(20, 50, 90) }, count: 64)
    let gif = GIFWriter(url: url, width: 8, height: 4, palette: palette, delayCentiseconds: 10)
    gif.add([UInt8](repeating: 3, count: 32))
    gif.add([UInt8](repeating: 7, count: 32))
    try gif.finish()
    let source = CGImageSourceCreateWithURL(url as CFURL, nil)!
    expectEqual(CGImageSourceGetCount(source), 2)
    let props = CGImageSourceCopyProperties(source, nil) as? [String: Any]
    let loop = (props?[kCGImagePropertyGIFDictionary as String] as? [String: Any])?[kCGImagePropertyGIFLoopCount as String]
    expectEqual((loop as? Int) ?? -1, 0)
    try? FileManager.default.removeItem(at: url)
}

finish()

extension Double { var float: Float { Float(self) } }
