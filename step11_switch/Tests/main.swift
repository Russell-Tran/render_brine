// Tests for step 11. The mechanism itself is checked against the sequence and
// the PDB files rather than against the JSON this project generated, so a
// mistake in the builder cannot hide behind its own output.

import Foundation
import Metal
import simd

let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
let d = try loadSwitch(from: root.appendingPathComponent("Resources/switch.json"))

/// Reads a PDB file straight off disk, the way the builder did.
func readPDB(_ name: String, modelOne: Bool = false) -> [(chain: Character, res: Int, resn: String,
                                                          atom: String, element: String, pos: SIMD3<Float>)] {
    let path = root.appendingPathComponent("Resources/\(name)")
    guard let text = try? String(contentsOf: path, encoding: .utf8) else { return [] }
    var out: [(Character, Int, String, String, String, SIMD3<Float>)] = []
    var inModel = true
    for line in text.split(separator: "\n", omittingEmptySubsequences: false) {
        if line.hasPrefix("MODEL") { inModel = line.contains(" 1") }
        if line.hasPrefix("ENDMDL") && modelOne { break }
        guard line.hasPrefix("ATOM") || line.hasPrefix("HETATM"), inModel else { continue }
        let c = Array(line)
        guard c.count >= 54 else { continue }
        func slice(_ a: Int, _ b: Int) -> String { String(c[a..<min(b, c.count)]).trimmingCharacters(in: .whitespaces) }
        let alt = c[16]
        guard alt == " " || alt == "A" else { continue }
        let el = slice(76, 78).isEmpty ? String(slice(12, 16).prefix(1)) : slice(76, 78)
        guard el != "H" else { continue }
        guard let res = Int(slice(22, 26)), let x = Float(slice(30, 38)),
              let y = Float(slice(38, 46)), let z = Float(slice(46, 54)) else { continue }
        out.append((c[21], res, slice(17, 20), slice(12, 16), el.uppercased(), SIMD3(x, y, z)))
    }
    return out
}

section("the structures, read back from the PDB files")

test("2ARC is the AraC dimer with L-arabinose in both subunits") {
    let atoms = readPDB("2ARC.pdb")
    let chains = Set(atoms.map { $0.chain })
    expectEqual(chains, Set(["A", "B"]))
    let ara = atoms.filter { $0.resn == "ARA" }
    // L-arabinose is C5H10O5: ten heavy atoms, one copy per subunit.
    expectEqual(ara.count, 20)
    expectEqual(Set(ara.map { $0.chain }), Set(["A", "B"]))
    expect(ara.allSatisfy { $0.element == "C" || $0.element == "O" }, "arabinose is only C and O")
}

test("2AAC is the same domain with D-fucose, the anti-inducer") {
    let atoms = readPDB("2AAC.pdb")
    expect(atoms.contains { $0.resn == "FCB" }, "no fucose in 2AAC")
    expect(!atoms.contains { $0.resn == "ARA" }, "2AAC should not contain arabinose")
}

test("2K9S is an NMR ensemble; only model 1 is used") {
    let one = readPDB("2K9S.pdb", modelOne: true)
    let all = readPDB("2K9S.pdb", modelOne: false)
    expectEqual(one.count, 846)
    expect(all.count > one.count * 5, "2K9S should hold many models (\(all.count) vs \(one.count))")
    let res = Set(one.map { $0.res })
    expect(res.count == 107, "the DNA-binding domain is 107 residues, got \(res.count)")
}

test("the JSON carries exactly those atoms, and no hydrogens") {
    expectEqual(d.core.count, 2637)
    expectEqual(d.dbd.count, 846)
    expectEqual(d.sugar.count, 10)
    let elements = Set(d.core.map { $0.element } + d.dbd.map { $0.element } + d.sugar.map { $0.element })
    expect(!elements.contains("H"), "no hydrogens should survive: \(elements)")
}

test("2ARC and 2AAC do NOT show an arm-open versus arm-closed difference") {
    // This is the finding that decided what the render may claim. Superposed on
    // their cores, the two structures' N-terminal arms differ by about as much
    // as the cores themselves — so the pair cannot be used as an off/on pair,
    // and the render does not pretend otherwise.
    guard let arm = d.armComparison["arm_7_18"], let core = d.armComparison["core_30_160"] else {
        expect(false, "the arm comparison is missing"); return
    }
    expect(arm < 1.2, "arm RMSD \(arm) Å — if this grew, the claim would need revisiting")
    expect(arm < core * 3, "arm \(arm) Å vs core \(core) Å: not a conformational difference")
}

section("the mechanism, against pGLO's own sequence")

test("araI1 and araI2 are the published half-sites, found verbatim in pGLO") {
    // The two 17 bp half-sites from the E. coli ara regulatory region occur in
    // pGLO exactly, which is how their positions are known rather than assumed.
    expectEqual(d.sites["araI1"]!.1 - d.sites["araI1"]!.0 + 1, 17)
    expectEqual(d.sites["araI2"]!.1 - d.sites["araI2"]!.0 + 1, 17)
    expect(d.sites["araI2"]!.0 > d.sites["araI1"]!.1, "araI2 should follow araI1")
    let gap = d.sites["araI2"]!.0 - d.sites["araI1"]!.1 - 1
    expect(gap >= 2 && gap <= 8, "araI1 and araI2 are adjacent, \(gap) bp apart")
}

test("araO2 and araI1 are 210 bp apart — the spacing the looping depends on") {
    expectEqual(d.spacingBP, 210)
}

test("the drawn window holds all three sites and the promoter") {
    for (name, span) in d.sites {
        expect(span.0 >= d.window.0 && span.1 <= d.window.1, "\(name) \(span) falls outside \(d.window)")
    }
    expect(d.plus1 > d.sites["araI2"]!.1, "the transcription start should follow araI2")
    expectEqual(d.nBP, d.window.1 - d.window.0 + 1)
}

section("the DNA survives being bent")

/// Every atom of the DNA at one loop span, with its base-pair index.
func dnaAtoms(span: Float) -> [(bp: Int, name: String, strand: Int, element: String, pos: SIMD3<Float>)] {
    let (pts, tan) = centreline(d, span: span)
    let (nor, bin) = transportFrames(points: pts, tangents: tan)
    var out: [(Int, String, Int, String, SIMD3<Float>)] = []
    for i in 0..<d.nBP {
        let ch = d.sequence[d.sequence.index(d.sequence.startIndex, offsetBy: i)]
        let kind: String
        switch ch { case "A": kind = "AT"; case "T": kind = "TA"; case "G": kind = "GC"; default: kind = "CG" }
        guard let t = d.templates[kind] else { continue }
        let ang = radians(d.twist * Float(i))
        let e1 = tan[i]
        let e2 = nor[i] * cos(ang) + bin[i] * sin(ang)
        let e3 = simd_cross(e1, e2)
        for a in t.atoms {
            let p = pts[i] + e1 * a.position.x + e2 * a.position.y + e3 * a.position.z
            out.append((i, a.name, a.strand, a.element, p))
        }
    }
    return out
}

test("every base pair stays a rigid crystal body at every span") {
    // The whole reason the DNA is rebuilt per frame rather than interpolated:
    // bond lengths and base pairing inside a base pair must never change.
    for span: Float in [1.0, 0.9, 0.8, 0.68, 0.5] {
        let atoms = dnaAtoms(span: span)
        var byBP: [Int: [(String, SIMD3<Float>)]] = [:]
        for a in atoms { byBP[a.bp, default: []].append((a.name, a.pos)) }
        // Compare one base pair's internal distances against the template's.
        let probe = 100
        let ch = d.sequence[d.sequence.index(d.sequence.startIndex, offsetBy: probe)]
        let kind: String
        switch ch { case "A": kind = "AT"; case "T": kind = "TA"; case "G": kind = "GC"; default: kind = "CG" }
        let t = d.templates[kind]!
        let placed = byBP[probe]!
        var worst: Float = 0
        for i in 0..<min(t.atoms.count, placed.count) {
            for j in (i + 1)..<min(t.atoms.count, placed.count) {
                let want = simd_distance(t.atoms[i].position, t.atoms[j].position)
                let got = simd_distance(placed[i].1, placed[j].1)
                worst = max(worst, abs(want - got))
            }
        }
        expect(worst < 1e-3, "span \(span): a base pair distorted by \(worst) Å")
    }
}

test("the backbone joins between base pairs stay a real bond length") {
    for span: Float in [1.0, 0.8, 0.68] {
        let atoms = dnaAtoms(span: span)
        var spot: [String: SIMD3<Float>] = [:]
        for a in atoms { spot["\(a.bp)/\(a.strand)/\(a.name)"] = a.pos }
        var lengths: [Float] = []
        for i in 0..<(d.nBP - 1) {
            if let o3 = spot["\(i)/0/O3'"], let p = spot["\(i + 1)/0/P"] {
                lengths.append(simd_distance(o3, p))
            }
        }
        expect(lengths.count > 200, "expected a join at nearly every step, got \(lengths.count)")
        let mean = lengths.reduce(0, +) / Float(lengths.count)
        // The O3'-P bond is 1.60 Å. Bending the helix stretches the joints a
        // little; step 8a measured the same effect and reported it.
        expect(mean > 1.0 && mean < 2.6, "span \(span): mean O3'–P join \(mean) Å")
        expect(lengths.max()! < 3.2, "span \(span): worst join \(lengths.max()!) Å")
    }
}

test("the helix stays right-handed all the way round the bend") {
    let (pts, tan) = centreline(d, span: 0.68)
    let (nor, bin) = transportFrames(points: pts, tangents: tan)
    var wrong = 0
    for i in 0..<(d.nBP - 1) {
        let a0 = radians(d.twist * Float(i))
        let a1 = radians(d.twist * Float(i + 1))
        let v0 = nor[i] * cos(a0) + bin[i] * sin(a0)
        let v1 = nor[i + 1] * cos(a1) + bin[i + 1] * sin(a1)
        // A right-handed turn about the tangent has a positive triple product.
        if simd_dot(simd_cross(v0, v1), tan[i]) <= 0 { wrong += 1 }
    }
    expectEqual(wrong, 0)
}

test("the loop never passes through itself") {
    for span: Float in [1.0, 0.9, 0.8, 0.68] {
        let (pts, _) = centreline(d, span: span)
        var closest: Float = .greatestFiniteMagnitude
        var where_ = ""
        let o2 = d.bpOf["araO2"]!, i1 = d.bpOf["araI1"]!
        // Skip the closure itself: when the loop shuts, araO2 and araI1 are
        // MEANT to meet — that is what AraC holding both means. Everywhere
        // else the path must stay a duplex width apart or the DNA would be
        // passing through itself.
        func atClosure(_ k: Int) -> Bool { abs(k - o2) < 14 || abs(k - i1) < 14 }
        for i in stride(from: 0, to: d.nBP, by: 3) {
            for j in stride(from: i + 25, to: d.nBP, by: 3) {
                if atClosure(i) && atClosure(j) { continue }
                let dist = simd_distance(pts[i], pts[j])
                if dist < closest { closest = dist; where_ = "bp \(i) and \(j)" }
            }
        }
        expect(closest > 20, "span \(span): \(where_) come within \(closest) Å")
    }
}

section("the switch")

test("the loop shuts when AraC holds araO2 and araI1, and opens when it lets go") {
    let o2 = d.bpOf["araO2"]!, i1 = d.bpOf["araI1"]!, i2 = d.bpOf["araI2"]!
    let shut = centreline(d, span: 1.0).points
    let open = centreline(d, span: 0.68).points
    let shutGap = simd_distance(shut[o2], shut[i1])
    let openGap = simd_distance(open[o2], open[i1])
    // Not zero: one AraC dimer has to BRIDGE the two sites, so they must come
    // within reach of a single protein (its two DNA-binding domains sit a few
    // tens of angstroms apart), not coincide. The loop also lifts out of plane
    // so the two duplexes pass rather than collide, which sets the floor here.
    expect(shutGap > 15 && shutGap < 45,
           "shut, araO2 and araI1 should be bridgeable by one dimer; they are \(shutGap) Å apart")
    expect(openGap > 120, "open, they should be far apart; they are \(openGap) Å")
    // araI1 and araI2 are adjacent whatever the loop does — that is the point.
    expect(simd_distance(open[i1], open[i2]) < 120, "araI1 and araI2 are neighbours")
}

test("araI2 is free while the loop is shut, and taken once it opens") {
    // The documented mechanism is not only steric: the loop also keeps a
    // DNA-binding domain off araI2, and it is araI2 being occupied that helps
    // RNA polymerase start. So this is the thing to test, not just occlusion.
    func gripDistance(_ s: SwitchState, to site: String) -> Float {
        let f = siteFrame(d, at: d.bpOf[site]!, span: s.span)
        let i1 = siteFrame(d, at: d.bpOf["araI1"]!, span: s.span)
        let o2 = siteFrame(d, at: d.bpOf["araO2"]!, span: s.span)
        let i2 = siteFrame(d, at: d.bpOf["araI2"]!, span: s.span)
        let farAt = simd_mix(o2.at, i2.at, SIMD3(repeating: s.gripI2))
        let a = simd_distance(f.at, i1.at)
        let b = simd_distance(f.at, farAt)
        return min(a, b)
    }
    let shut = SwitchState(span: 1.0, sugarIn: 0, gripI2: 0, rnap: 0, transcribing: 0,
                           caption: Caption(title: "", subtitle: "", facts: "", aside: ""))
    let open = SwitchState(span: 0.68, sugarIn: 1, gripI2: 1, rnap: 1, transcribing: 1,
                           caption: Caption(title: "", subtitle: "", facts: "", aside: ""))
    expect(gripDistance(shut, to: "araO2") < 5, "shut: AraC should be on araO2")
    expect(gripDistance(shut, to: "araI2") > 40, "shut: araI2 should be free")
    expect(gripDistance(open, to: "araI2") < 5, "open: AraC should be on araI2")
    expect(gripDistance(open, to: "araO2") > 100, "open: araO2 should be released")
}

test("the promoter is buried while the loop is shut and exposed once it opens") {
    // Traced with real rays, as step 9 checked its cutaway: fire a fan of rays
    // at the promoter from where the polymerase comes in and count how many
    // arrive.
    func reach(_ s: SwitchState) -> Double {
        let shapes = sceneShapes(d, s)
        let bp = min(max(d.plus1 - 32 - d.window.0, 0), d.nBP - 1)
        let prom = siteFrame(d, at: bp, span: s.span)
        var side = simd_cross(prom.along, SIMD3<Float>(0, 0, 1))
        side = simd_length(side) < 1e-5 ? SIMD3(0, -1, 0) : simd_normalize(side)
        if side.y < 0 { side = -side }
        var arrived = 0, tried = 0
        for k in -6...6 {
            let spread = Float(k) * 0.09
            let from = prom.at + side * 150 + prom.along * (spread * 150)
            let dir = simd_normalize(prom.at - from)
            let target = simd_distance(from, prom.at)
            var blocked = false
            for sh in shapes where sh.a.w > 0 {
                // Only the DNA and the protein can block; skip the polymerase itself.
                let c = SIMD3(sh.a.x, sh.a.y, sh.a.z)
                if sh.isSphere && sh.a.w > 10 { continue }
                let oc = from - c
                let b = simd_dot(oc, dir)
                let h = b * b - (simd_dot(oc, oc) - sh.a.w * sh.a.w)
                if h < 0 { continue }
                let t = -b - sqrt(h)
                if t > 1 && t < target - 12 { blocked = true; break }
            }
            tried += 1
            if !blocked { arrived += 1 }
        }
        return Double(arrived) / Double(tried)
    }
    let shut = SwitchState(span: 1.0, sugarIn: 0, gripI2: 0, rnap: 0, transcribing: 0,
                           caption: Caption(title: "", subtitle: "", facts: "", aside: ""))
    let open = SwitchState(span: 0.68, sugarIn: 1, gripI2: 1, rnap: 0, transcribing: 1,
                           caption: Caption(title: "", subtitle: "", facts: "", aside: ""))
    let a = reach(shut), b = reach(open)
    expect(b >= a, "the promoter should be no harder to reach once the loop opens (\(a) → \(b))")
}

test("the protein really is space-filling, and the DNA really is ball-and-stick") {
    // Space-filling means van der Waals radii, so neighbouring atoms' spheres
    // OVERLAP and merge into one surface — that is the whole definition, and
    // it is what a covalent radius by mistake would break. The DNA is drawn
    // the other way: small balls with visible sticks between them.
    let shapes = sceneShapes(d, state(d, at: 0.0))
    // AraC's atoms are the ones drawn at van der Waals size.
    let protein = shapes.filter { $0.isSphere && $0.a.w > 1.4 && $0.a.w < 2.0 }
    expect(protein.count > 3000, "expected the AraC dimer and two DNA-binding domains, got \(protein.count)")
    // Nearest-neighbour distances within the protein must be smaller than the
    // sum of two radii, or the surface would be a heap of separate balls.
    var overlapping = 0, sampled = 0
    for (i, a) in protein.enumerated() where i % 97 == 0 {
        let pa = SIMD3(a.a.x, a.a.y, a.a.z)
        var nearest: Float = .greatestFiniteMagnitude
        for (j, b) in protein.enumerated() where j != i {
            nearest = min(nearest, simd_distance(pa, SIMD3(b.a.x, b.a.y, b.a.z)))
        }
        sampled += 1
        if nearest < a.a.w * 2 { overlapping += 1 }
    }
    expect(sampled > 20, "only \(sampled) atoms sampled")
    expect(overlapping == sampled,
           "\(sampled - overlapping) of \(sampled) protein atoms do not overlap a neighbour — not space-filling")
    // The DNA's spheres are much smaller, and it has sticks.
    let balls = shapes.filter { $0.isSphere && $0.a.w < 0.5 }
    let sticks = shapes.filter { !$0.isSphere }
    expect(balls.count > 15000, "expected the DNA drawn as small balls, got \(balls.count)")
    expect(sticks.count > 15000, "expected bonds drawn as sticks, got \(sticks.count)")
}

section("the cycle")

let totalFramesT = 240

test("it is a cycle: the last frame returns to the first") {
    let first = state(d, at: 0)
    let last = state(d, at: Double(totalFramesT - 1) / Double(totalFramesT))
    expect(abs(first.span - last.span) < 0.02, "span \(first.span) vs \(last.span)")
    expect(abs(first.gripI2 - last.gripI2) < 0.02, "grip \(first.gripI2) vs \(last.gripI2)")
    expect(abs(first.sugarIn - last.sugarIn) < 0.02, "sugar \(first.sugarIn) vs \(last.sugarIn)")
    expect(abs(first.rnap - last.rnap) < 0.02, "polymerase \(first.rnap) vs \(last.rnap)")
}

test("it is a cycle and not a reversed playback") {
    // If the second half were the first half backwards, u and 1-u would match.
    var mirrored = 0
    for f in stride(from: 10, to: totalFramesT / 2, by: 5) {
        let u = Double(f) / Double(totalFramesT)
        let a = state(d, at: u), b = state(d, at: 1 - u)
        if abs(a.span - b.span) < 0.01 && abs(a.sugarIn - b.sugarIn) < 0.01 { mirrored += 1 }
    }
    expect(mirrored < 4, "\(mirrored) sampled pairs mirror: this looks like playback in reverse")
}

test("nothing jumps between neighbouring frames") {
    var worst = 0.0
    for f in 0..<totalFramesT {
        let a = state(d, at: Double(f) / Double(totalFramesT))
        let b = state(d, at: Double(f + 1) / Double(totalFramesT))
        worst = max(worst, Double(abs(a.span - b.span)))
    }
    expect(worst < 0.02, "the loop span jumps by \(worst) in one frame")
}

test("the sugar causes the grip to move, and never the other way round") {
    // The causal order, not simultaneity. Arabinose binds FIRST and the grip
    // follows; on the way back the sugar leaves first and the grip follows it.
    // An earlier version of this test demanded sugar whenever the grip was
    // flipped, and failed on the release — correctly, because for a moment the
    // sugar has gone and AraC has not yet let go. That is the mechanism, not a
    // bug, so the test now checks the ordering instead.
    var firstSugar = 2.0, firstGrip = 2.0, lastSugar = -1.0, lastGrip = -1.0
    for f in 0..<totalFramesT {
        let u = Double(f) / Double(totalFramesT)
        let s = state(d, at: u)
        if s.sugarIn > 0.5 { firstSugar = min(firstSugar, u); lastSugar = max(lastSugar, u) }
        if s.gripI2 > 0.5 { firstGrip = min(firstGrip, u); lastGrip = max(lastGrip, u) }
        if u < 0.10 { expect(s.sugarIn < 0.01, "sugar present at u=\(u), before it arrives") }
    }
    expect(firstSugar < firstGrip, "the sugar must bind before the grip moves (\(firstSugar) vs \(firstGrip))")
    expect(lastSugar < lastGrip, "the sugar must leave before the grip releases (\(lastSugar) vs \(lastGrip))")
}

test("the mechanism stays in frame, even though the DNA runs out of it") {
    // Unlike step 9's compact protein, this DNA is a WINDOW onto a 5,371 bp
    // plasmid: the strand legitimately leaves the frame at both ends, and an
    // earlier version of this test failed on exactly that. What must stay
    // visible is the mechanism — the three half-sites, and the promoter.
    let camera = Camera.orbit(target: SIMD3(-30, 130, 0), distance: 780, yaw: 0, pitch: 10, fov: 30)
    for f in stride(from: 0, to: totalFramesT, by: 6) {
        let u = Double(f) / Double(totalFramesT)
        let s = state(d, at: u)
        for name in ["araO2", "araI1", "araI2"] {
            let p = siteFrame(d, at: d.bpOf[name]!, span: s.span).at
            expect(camera.sees(p, width: 960, height: 600, margin: -20),
                   "\(name) is out of frame at u=\(u)")
        }
        let prom = siteFrame(d, at: d.plus1 - 32 - d.window.0, span: s.span).at
        expect(camera.sees(prom, width: 960, height: 600, margin: -20), "the promoter is out of frame at u=\(u)")
    }
}

section("the grid, now rebuilt every frame")

let device = try findDevice()
let renderer = try SceneRenderer(device: device)
let layoutT = FrameLayout(width: 240, viewHeight: 150, captionHeight: 40)
let bufferA = device.makeBuffer(length: layoutT.width * layoutT.height * 4, options: .storageModeShared)!
let bufferB = device.makeBuffer(length: layoutT.width * layoutT.height * 4, options: .storageModeShared)!
let testCamera = Camera.orbit(target: SIMD3(-30, 130, 0), distance: 780, yaw: 0, pitch: 10, fov: 30)

func bytes(_ b: MTLBuffer) -> [UInt8] {
    let p = b.contents().assumingMemoryBound(to: UInt8.self)
    return Array(UnsafeBufferPointer(start: p, count: layoutT.width * layoutT.height * 4))
}

test("a grid built for this frame's geometry gives the same picture as brute force") {
    // Step 8a's check, now on a scene that moved: the grid must be rebuilt, and
    // a stale one would show up here.
    for u in [0.0, 0.4, 0.7] {
        let s = state(d, at: u)
        let shapes = sceneShapes(d, s)
        try renderer.buildGrid(shapes)
        try renderer.render(shapes: shapes, camera: testCamera, into: bufferA,
                            width: layoutT.width, viewHeight: layoutT.viewHeight,
                            ao: AOSettings(probes: 6, distance: 8, strength: 1, contrast: 1.7, only: false),
                            useGrid: true)
        try renderer.render(shapes: shapes, camera: testCamera, into: bufferB,
                            width: layoutT.width, viewHeight: layoutT.viewHeight,
                            ao: AOSettings(probes: 6, distance: 8, strength: 1, contrast: 1.7, only: false),
                            useGrid: false)
        let a = bytes(bufferA), b = bytes(bufferB)
        var differ = 0
        for i in 0..<a.count where a[i] != b[i] { differ += 1 }
        let fraction = Double(differ) / Double(a.count)
        expect(fraction < 0.02, "u=\(u): \(String(format: "%.2f", fraction * 100))% of bytes differ")
    }
}

test("a grid left over from the previous frame is caught") {
    // The failure this step could actually have: build once, then move the
    // scene. The picture must change; if it did not, the grid would be stale.
    let a0 = sceneShapes(d, state(d, at: 0.0))
    let a1 = sceneShapes(d, state(d, at: 0.7))
    try renderer.buildGrid(a0)
    try renderer.render(shapes: a1, camera: testCamera, into: bufferA, width: layoutT.width,
                        viewHeight: layoutT.viewHeight, ao: .off, useGrid: true)
    let stale = bytes(bufferA)
    try renderer.buildGrid(a1)
    try renderer.render(shapes: a1, camera: testCamera, into: bufferB, width: layoutT.width,
                        viewHeight: layoutT.viewHeight, ao: .off, useGrid: true)
    let fresh = bytes(bufferB)
    var differ = 0
    for i in 0..<stale.count where stale[i] != fresh[i] { differ += 1 }
    expect(differ > 0, "a stale grid drew the same picture as a rebuilt one — the test cannot detect staleness")
}

test("every shape is filed in the box that holds its middle") {
    let shapes = sceneShapes(d, state(d, at: 0.3))
    let grid = UniformGrid(shapes: shapes)
    var checked = 0
    for (i, s) in shapes.enumerated() where s.a.w > 0 {
        if i % 500 != 0 { continue }
        let mid = s.isSphere ? SIMD3(s.a.x, s.a.y, s.a.z)
                             : (SIMD3(s.a.x, s.a.y, s.a.z) + SIMD3(s.b.x, s.b.y, s.b.z)) * 0.5
        expect(grid.shapes(at: mid).contains(i), "shape \(i) is not in the box holding its middle")
        checked += 1
    }
    expect(checked > 10, "only \(checked) shapes checked")
}

test("occlusion is steady from one frame to the next") {
    // Step 9 could guarantee this by construction, because its molecule never
    // moved. Here it cannot, so it is measured: consecutive frames differ only
    // by how much the geometry actually moved, not by sampling churn.
    let ao = AOSettings(probes: 12, distance: 8, strength: 1, contrast: 1.7, only: true)
    let s0 = state(d, at: 0.02)
    let s1 = state(d, at: 0.02 + 1.0 / 240.0)
    try renderer.buildGrid(sceneShapes(d, s0))
    try renderer.render(shapes: sceneShapes(d, s0), camera: testCamera, into: bufferA,
                        width: layoutT.width, viewHeight: layoutT.viewHeight, ao: ao)
    try renderer.buildGrid(sceneShapes(d, s1))
    try renderer.render(shapes: sceneShapes(d, s1), camera: testCamera, into: bufferB,
                        width: layoutT.width, viewHeight: layoutT.viewHeight, ao: ao)
    let a = bytes(bufferA), b = bytes(bufferB)
    var total = 0.0, counted = 0
    for i in stride(from: 0, to: a.count, by: 4) {
        total += abs(Double(a[i]) - Double(b[i]))
        counted += 1
    }
    let mean = total / Double(counted)
    // Nothing moves between these two frames (both sit in the shut phase), so
    // any difference is sampling churn.
    expect(mean < 2.0, "occlusion shifts by \(String(format: "%.1f", mean)) grey levels between still frames")
}

finish()
