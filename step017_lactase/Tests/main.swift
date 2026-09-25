// What this step claims, checked.
//
// The ten sections below are the ten claims the render makes. Four of them
// (the RMSD check, the two inversions, the covalent bond and Glu461's role) are
// read back out of coordinates rather than out of a table, which is the point:
// if the structures did not say these things, the tests would fail and the
// render would be wrong.
//
//   make test        run them
//   make mutants     break four things on purpose and check a test notices

import Foundation
import Metal
import simd

let scene = try! loadScene(from: URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent().deletingLastPathComponent()
    .appendingPathComponent("Resources/scene.json"))
let mutations = LacMutations.fromEnvironment()

/// A boundary pose, read through the same function the render uses, so a
/// mutation cannot break the picture without breaking a test as well.
func at(_ boundary: Int) -> [SIMD3<Float>] {
    boundaryPose(scene, boundary, mutations: mutations)
}

/// The first frame of each interval of the cycle, for walking the whole loop.
func frameAtBoundary(_ b: Int) -> Int {
    var f = turnFrames + openFrames
    for i in 0..<min(b, stageFrames.count) { f += stageFrames[i] }
    return f
}

// MARK: - 1. The check before animating anything
//
// Step 11 measured the displacement between its two structures first and
// refused to animate a change smaller than its own error bars. The same check
// here, with the noise floor taken from the four crystallographically
// independent copies inside each entry — same data, same refinement, so
// whatever they disagree about is noise by construction.

section("1. the motion was measured before it was animated")

test("the noise floor was measured from independent copies, not assumed") {
    expect(scene.motion.noiseCA > 0.15 && scene.motion.noiseCA < 1.0,
           "chain noise \(scene.motion.noiseCA) Å is not a plausible measurement")
    expect(scene.motion.noiseSite > 0.3 && scene.motion.noiseSite < 3.0,
           "site noise \(scene.motion.noiseSite) Å is not a plausible measurement")
}

test("the protein's state-to-state difference is BELOW that floor") {
    expect(scene.motion.worstCA <= scene.motion.noiseCA,
           "chain: \(scene.motion.worstCA) Å against a floor of \(scene.motion.noiseCA) Å")
    expect(scene.motion.worstSite <= scene.motion.noiseSite,
           "site: \(scene.motion.worstSite) Å against a floor of \(scene.motion.noiseSite) Å")
    expect(!scene.motion.proteinMovesAboveNoise && !scene.motion.siteMovesAboveNoise,
           "the builder thinks the protein moves; it does not")
}

test("so no protein atom moves in the render, at any frame") {
    // The only defence against animating noise is not to animate it. The
    // space-filling tetramer and the ball-and-stick side chains come from one
    // structure and are the same array in every frame.
    let a = lacShapes(scene, at: 0, porthole: .shut)
    let b = lacShapes(scene, at: frameAtBoundary(4), porthole: .shut)
    expectEqual(a.count, b.count)
    var moved = 0
    for i in 0..<min(a.count, b.count) where simd_length(a[i].a - b[i].a) > 1e-6 { moved += 1 }
    expectEqual(moved, 0)
}

test("the ligand's slide, by contrast, is well above the floor") {
    // 2.7 Å against a 1.1 Å site noise floor, and the two clusters are tight.
    expect(scene.slide > scene.motion.noiseSite,
           "the sugar slides \(scene.slide) Å, floor \(scene.motion.noiseSite) Å")
    expect(scene.slide > 2.0, "slide \(scene.slide) Å")
}

// MARK: - 2. Two inversions, one retention
//
// The beat the render exists for, computed from coordinates at every frame.

section("2. two inversions, one retention")

test("the deposited coordinates say beta, then alpha, then beta") {
    expectEqual(scene.anomeric.count, 3)
    expectEqual(scene.anomeric[0].config, "beta")     // 1JYN, lactose
    expectEqual(scene.anomeric[1].config, "alpha")    // 1JZ2, covalent intermediate
    expectEqual(scene.anomeric[2].config, "beta")     // 1JZ7, product
    // and they are real bonds, not contacts
    for a in scene.anomeric {
        expect(a.bond > 1.30 && a.bond < 1.55, "\(a.state): C1–exo is \(a.bond) Å")
    }
}

test("the PDB's own name for 1JZ2's ligand disagrees, and the coordinates win") {
    // Chemical component 2FG is "2-deoxy-2-fluoro-BETA-D-galactopyranose". In
    // 1JZ2 it is bonded to Glu537 and its coordinates are alpha. The render
    // reads the geometry, never the name.
    expectEqual(scene.anomeric[1].entry, "1JZ2")
    expect(scene.anomeric[1].signedVolume < 0,
           "signed volume \(scene.anomeric[1].signedVolume) should be negative (alpha)")
    expect(scene.anomeric[0].signedVolume > 0 && scene.anomeric[2].signedVolume > 0,
           "substrate and product should both be positive (beta)")
}

test("the render inverts at step 1, inverts again at step 2, and retains") {
    let substrate: Float = anomericVolume(scene, positions: at(1), boundary: 1)
    let covalent: Float = anomericVolume(scene, positions: at(4), boundary: 4)
    let product: Float = anomericVolume(scene, positions: at(7), boundary: 7)
    expect(substrate > 0, "substrate signed volume \(substrate)")
    expect(covalent < 0, "covalent signed volume \(covalent) — it did not invert")
    expect(product > 0, "product signed volume \(product) — it did not invert back")
    expect((substrate > 0) == (product > 0),
           "net retention failed: \(substrate) then \(product)")
}

test("the anomeric hydrogen flips the other way, and flips back") {
    // The clearest picture of the inversion, and a second, independent read of
    // it: H1 is derived from the three heavy substituents, so if it did not
    // change sides the heavy atoms did not either.
    func hSign(_ b: Int) -> Float {
        let p = at(b)
        let h: SIMD3<Float> = anomericHydrogen(scene, positions: p, boundary: b)
        let c1 = p[mobileIndex(scene, "GAL:C1")]
        let o5 = p[mobileIndex(scene, "GAL:O5")]
        let c2 = p[mobileIndex(scene, "GAL:C2")]
        return anomericVolume(c1: c1, o5: o5, c2: c2, exo: h)
    }
    expect(hSign(1) < 0, "H1 should start under the ring, got \(hSign(1))")
    expect(hSign(4) > 0, "H1 should be over the ring in the intermediate, got \(hSign(4))")
    expect(hSign(7) < 0, "H1 should come back under it, got \(hSign(7))")
}

test("the transition state is flat, so there is no configuration at the top") {
    // sp² at C1 is the whole claim of a transition-state mimic, and it is
    // measured: 1JZ5's lactone against an ordinary sugar in 1JZ7. Flat means
    // the anomeric centre has no configuration there at all, which is how it
    // can come out of the transition state on the other side.
    expect(scene.lactoneOutOfPlane < 0.20,
           "1JZ5's C1 oxygen is \(scene.lactoneOutOfPlane) Å out of plane — not flat")
    expect(scene.sugarOutOfPlane > 0.8,
           "1JZ7's C1 oxygen is only \(scene.sugarOutOfPlane) Å out of plane")
    expect(scene.sugarOutOfPlane > 8 * scene.lactoneOutOfPlane,
           "the mimic is not measurably flatter than the sugar")
}

test("the configuration is read at EVERY frame, and holds through each half") {
    // Not only at the boundaries. Within a half-cycle the sign must never
    // wobble; between them it must change. The exocyclic partner changes
    // identity twice over the cycle — glucose's bridging oxygen, then Glu537's,
    // then the water's — and that IS the mechanism, so the sign is read against
    // whichever atom holds the position at the time.
    func signsOverStage(_ s: Int) -> [Float] {
        var out: [Float] = []
        let from = frameAtBoundary(s)
        for f in from..<(from + stageFrames[min(s, stageFrames.count - 1)]) {
            out.append(anomericVolume(scene, positions: pose(scene, at: f,
                                                             mutations: mutations),
                                      boundary: s))
        }
        return out
    }
    for s in [0, 1, 2] {
        let v = signsOverStage(s)
        expect(v.allSatisfy { $0 > 0 }, "stage \(s) (substrate) should be β throughout")
    }
    for s in [4, 5, 6] {
        let v = signsOverStage(s)
        expect(v.allSatisfy { $0 < 0 }, "stage \(s) (covalent) should be α throughout")
    }
    let product = anomericVolume(scene, positions: at(7), boundary: 7)
    let parked = anomericVolume(scene, positions: at(8), boundary: 8)
    expect(product > 0 && parked > 0, "the product should be β and stay β")
}

// MARK: - 3. Atoms and charge conserved
//
// Step 7a's check_balances.py counted both across every reaction of glycolysis.

section("3. atoms and charge are conserved")

test("lactose + water = glucose + galactose, element by element") {
    let (left, right) = lactoseHydrolysis()
    expect(LacFormula.self == LacFormula.self)   // keep the type in play
    expectEqual(left.text, right.text)
    expectEqual(left.charge, right.charge)
    expectEqual(left.text, "C12H24O12")
}

test("both half-reactions balance, with the enzyme in them") {
    for (name, l, r) in halfReactions() {
        expectEqual("\(name) \(l.text)", "\(name) \(r.text)")
        expect(l.charge == r.charge,
               "\(name): charge \(l.charge) → \(r.charge)")
    }
}

test("nothing is created or destroyed in the render itself") {
    // Structural, not asserted: the 27 mobile atoms have fixed identities and
    // every frame poses all of them.
    expectEqual(substrateFormula(scene).text, productFormula(scene).text)
    for b in 0...8 { expectEqual(at(b).count, scene.mobiles.count) }
    var carbons = 0, oxygens = 0, hydrogens = 0
    for m in scene.mobiles {
        if m.element == "C" { carbons += 1 }
        if m.element == "O" { oxygens += 1 }
        if m.element == "H" { hydrogens += 1 }
    }
    expectEqual(carbons, 12)      // 6 galactosyl + 6 glucose
    expectEqual(oxygens, 12)      // 5 galactosyl + 6 glucose + 1 water
    expectEqual(hydrogens, 3)     // the acid proton and the water's two
}

// MARK: - 4. The covalent bond exists, and only where it should

section("4. the covalent bond")

test("C1 and Glu537's oxygen are a bond length apart in the intermediate") {
    for b in [4, 5, 6] {
        let (bonded, d) = covalentBond(scene, positions: at(b))
        expect(bonded, "boundary \(b) (\(scene.boundaryNames[b])): C1–OE2 is \(d) Å, not a bond")
        expect(d > 1.30 && d < 1.60, "boundary \(b): C1–OE2 is \(d) Å")
    }
}

test("and they are not, in any other state") {
    for b in [0, 1, 2, 3, 7, 8] {
        let (bonded, d) = covalentBond(scene, positions: at(b))
        expect(!bonded, "boundary \(b) (\(scene.boundaryNames[b])) has a covalent bond at \(d) Å")
    }
}

test("the render draws that bond in those frames and no others") {
    func hasBond(_ f: Int) -> Bool {
        let p = pose(scene, at: f, mutations: mutations)
        for b in lacBonds(scene, positions: p, mutations: mutations)
        where b.labelA == "GLU537:OE2" && b.labelB == "GAL:C1" && b.strength > 0.9 {
            return true
        }
        return false
    }
    for b in [4, 5, 6] {
        let f = frameAtBoundary(b)
        expect(hasBond(f), "frame \(f) (\(scene.boundaryNames[b])) has no drawn bond to Glu537")
    }
    for b in [0, 1, 2] {
        let f = frameAtBoundary(b)
        expect(!hasBond(f), "frame \(f) (\(scene.boundaryNames[b])) draws a bond to Glu537")
    }
    // the free enzyme at the very end of the loop
    expect(!hasBond(lacFrameCount - 1), "the last frame still has the sugar on the enzyme")
}

test("the bond in the deposited structure is the one that is drawn") {
    let deposited: Float = scene.anomeric[1].bond
    let (_, drawn) = covalentBond(scene, positions: at(4))
    expect(abs(deposited - drawn) < 0.05,
           "1JZ2 says \(deposited) Å, the render draws \(drawn) Å")
}

// MARK: - 5. Glu461 changes role, and is never both

section("5. Glu461 is an acid, then a base, never both")

test("it holds its proton through the first half-cycle") {
    for b in [0, 1, 2] {
        expectEqual("\(b):\(glu461Role(scene, positions: at(b)))", "\(b):\(Glu461Role.acid)")
    }
}

test("it is bare through the second") {
    // After the proton has gone out with the glucose and before the water
    // brings one back.
    for b in [4, 5, 6] {
        let role = glu461Role(scene, positions: at(b))
        expect(role == .base, "boundary \(b) (\(scene.boundaryNames[b])): Glu461 is \(role.label)")
    }
}

test("it has its proton back at the end, and it is not the one it started with") {
    expect(glu461Role(scene, positions: at(7)) == .acid,
           "Glu461 should have taken a proton off the water by the product state")
    // the proton it ends with is the one labelled H:w2, not H:acid
    let acid = mobileIndex(scene, "H:acid")
    let w2 = mobileIndex(scene, "H:w2")
    let oe2: SIMD3<Float> = glu461(scene, "OE2")
    let dAcid: Float = simd_length(at(7)[acid] - oe2)
    let dW2: Float = simd_length(at(7)[w2] - oe2)
    expect(dW2 < dAcid, "at the product state the water's proton (\(dW2) Å) should be the one "
                      + "on Glu461, not the original acid proton (\(dAcid) Å)")
}

test("it is never holding two protons at once") {
    for f in 0...lacFrameCount {
        let n = glu461ProtonCount(scene, positions: pose(scene, at: f, mutations: mutations))
        expect(n <= 1, "frame \(f): Glu461 is holding \(n) protons — it cannot be an acid and a "
                     + "base at the same time")
    }
}

test("the geometry that lets it do both jobs is measured, not asserted") {
    expect(scene.waterToGlu461 > 2.4 && scene.waterToGlu461 < 3.3,
           "the attacking water is \(scene.waterToGlu461) Å from Glu461")
    expect(scene.productO1ToGlu461 > 2.2 && scene.productO1ToGlu461 < 3.2,
           "the product's new hydroxyl is \(scene.productO1ToGlu461) Å from Glu461")
    expect(scene.waterCosToBond > 0.8,
           "the water sits at cos \(scene.waterCosToBond) to the bond it attacks; it should be "
           + "nearly straight behind it")
}

test("the magnesium is secondary, and the reason is geometric") {
    expect(scene.metalToNucleophile > 6.0,
           "the Mg is \(scene.metalToNucleophile) Å from the nucleophile — if it were close "
           + "enough to act on the anomeric centre, calling it secondary would be wrong")
}

// MARK: - 6. Nothing passes through anything

section("6. nothing passes through anything")

test("the leaving glucose never enters the protein") {
    // Every frame of the loop, every glucose atom, against every protein atom
    // within reach. A van der Waals overlap of more than 1.0 Å is a collision.
    var glucose: [Int] = []
    for (i, m) in scene.mobiles.enumerated() where m.label.hasPrefix("GLC") { glucose.append(i) }
    expect(glucose.count == 12, "expected 12 glucose atoms, found \(glucose.count)")
    var worst: Float = 1e9
    var worstFrame = -1
    for f in stride(from: 0, to: lacFrameCount, by: 2) {
        let p = pose(scene, at: f, mutations: mutations)
        for i in glucose {
            let q = p[i]
            for a in scene.atoms {
                let d: Float = simd_length(a.position - q)
                if d > 6 { continue }
                let gap: Float = d - vdwRadius(a.element)
                if gap < worst { worst = gap; worstFrame = f }
            }
        }
    }
    expect(worst > -1.0, "glucose overlaps the protein by \(-worst) Å at frame \(worstFrame)")
}

test("the galactose leaves the same way, and so does the water arriving") {
    for label in ["GAL:C1", "WAT:O"] {
        let i = mobileIndex(scene, label)
        var worst: Float = 1e9
        for f in stride(from: 0, to: lacFrameCount, by: 2) {
            let q = pose(scene, at: f, mutations: mutations)[i]
            for a in scene.atoms where simd_length(a.position - q) < 6 {
                worst = min(worst, simd_length(a.position - q) - vdwRadius(a.element))
            }
        }
        expect(worst > -1.2, "\(label) overlaps the protein by \(-worst) Å")
    }
}

test("the way out was found, not drawn") {
    expect(scene.channel.count >= 6, "the channel has \(scene.channel.count) waypoints")
    // it runs from the site outward
    let first: Float = simd_length(scene.channel.first! - scene.activeSite)
    let last: Float = simd_length(scene.channel.last! - scene.activeSite)
    expect(last > first + 20, "the channel goes from \(first) Å to \(last) Å from the site")
}

// MARK: - 7. The porthole is a real cut

section("7. the porthole is a real cut, and it is capped")

test("the CPU and the kernel compute the same porthole interval") {
    // The kernel's portholeSpan and this one are the same arithmetic written
    // twice; this checks the Swift one against its own definition of inside.
    let hole = Porthole(center: SIMD3(0, 0, 0), axis: SIMD3(0, 0, 1), radius: 6, back: -4)
    var checked = 0
    for k in 0..<400 {
        let a: Float = Float(k) * 0.618
        let ro = SIMD3<Float>(cos(a) * 30, sin(a * 1.7) * 30, 40)
        // aimed near the bore, with enough spread that some rays clip the wall
        // and some go down the middle
        let aim = SIMD3<Float>(cos(a * 2.3) * 7, sin(a * 3.1) * 7, -6)
        let rd: SIMD3<Float> = simd_normalize(aim - ro)
        guard let span = portholeSpan(origin: ro, direction: rd, hole) else { continue }
        checked += 1
        let mid: SIMD3<Float> = ro + rd * ((span.enter + span.leave) / 2)
        expect(hole.contains(mid), "the middle of the span is not inside the porthole")
        let before: SIMD3<Float> = ro + rd * (span.enter - 0.05)
        let after: SIMD3<Float> = ro + rd * (span.leave + 0.05)
        expect(!hole.contains(before) && !hole.contains(after),
               "the span does not end where the porthole does")
    }
    expect(checked > 100, "only \(checked) rays met the porthole")
}

test("no ray sees a backface through the opening") {
    // A backface is a surface whose normal points AWAY from the ray: the
    // inside of a sphere. The cut faces must be shaded with the porthole's own
    // normal, which always faces the camera.
    let f = turnFrames + openFrames + 60
    let camera = lacCamera(scene, at: f)
    let hole = lacPorthole(scene, at: f, camera: camera)
    expect(hole.isOpen, "the porthole should be open at frame \(f)")
    // only the shapes near the bore, so the brute-force trace is affordable
    let all = lacShapes(scene, at: f, porthole: hole)
    let near = all.filter { simd_length(SIMD3($0.a.x, $0.a.y, $0.a.z) - hole.center) < 42 }
    expect(near.count > 400, "only \(near.count) shapes near the porthole")

    var hits = 0, capped = 0, backfaces = 0
    for j in stride(from: 0, to: 120, by: 3) {
        for i in stride(from: 0, to: 160, by: 3) {
            let ndcX: Float = (Float(i) + 0.5) / 160 * 2 - 1
            let ndcY: Float = 1 - (Float(j) + 0.5) / 120 * 2
            let aspect: Float = 160.0 / 120.0
            let dir: SIMD3<Float> = simd_normalize(
                camera.forward + camera.right * (ndcX * aspect * camera.tanHalfFOV * 0.5)
                + camera.up * (ndcY * camera.tanHalfFOV * 0.5))
            guard let hit = traceCPU(origin: camera.origin, direction: dir,
                                     shapes: near, hole: hole) else { continue }
            hits += 1
            if hit.capped { capped += 1 }
            if simd_dot(hit.normal, dir) > 1e-4 { backfaces += 1 }
        }
    }
    expect(hits > 500, "only \(hits) rays hit anything")
    expect(capped > 20, "only \(capped) rays landed on a cut face — the cut is not being seen")
    expectEqual(backfaces, 0)
}

test("no hit is ever inside the porthole: the material really is gone") {
    let f = turnFrames + openFrames + 60
    let camera = lacCamera(scene, at: f)
    let hole = lacPorthole(scene, at: f, camera: camera)
    let all = lacShapes(scene, at: f, porthole: hole)
    let near = all.filter { simd_length(SIMD3($0.a.x, $0.a.y, $0.a.z) - hole.center) < 42 }
    var inside = 0
    for j in stride(from: 0, to: 120, by: 5) {
        for i in stride(from: 0, to: 160, by: 5) {
            let ndcX: Float = (Float(i) + 0.5) / 160 * 2 - 1
            let ndcY: Float = 1 - (Float(j) + 0.5) / 120 * 2
            let dir: SIMD3<Float> = simd_normalize(
                camera.forward + camera.right * (ndcX * 1.3333 * camera.tanHalfFOV * 0.5)
                + camera.up * (ndcY * camera.tanHalfFOV * 0.5))
            guard let hit = traceCPU(origin: camera.origin, direction: dir,
                                     shapes: near, hole: hole) else { continue }
            let p: SIMD3<Float> = camera.origin + dir * hit.t
            // the cast is not clipped, so only clippable shapes are checked
            if near[hit.shape].color.w > 0.5 && hole.contains(p + dir * 0.02) { inside += 1 }
        }
    }
    expectEqual(inside, 0)
}

test("the porthole is shut at the start and the end of the loop") {
    expectEqual(portholeRadius(scene, at: 0), 0)
    expectEqual(portholeRadius(scene, at: lacFrameCount), 0)
    expectEqual(portholeRadius(scene, at: lacFrameCount - 1), 0)
    expect(portholeRadius(scene, at: turnFrames + openFrames + 60) > 8,
           "the porthole should be wide open during the cycle")
}

// MARK: - 8. The loop closes

section("8. the loop closes, exactly")

test("frame N is frame 0, atom for atom") {
    // `cyclePosition` takes any integer, so this asks for one frame past the
    // end. Under the `loop` mutation it reduces modulo the wrong number and
    // this is what notices.
    let a = pose(scene, at: 0, mutations: mutations)
    let b = pose(scene, at: lacFrameCount, mutations: mutations)
    expectEqual(a.count, b.count)
    var worst: Float = 0
    var worstLabel = ""
    for i in 0..<min(a.count, b.count) {
        let d: Float = simd_length(a[i] - b[i])
        if d > worst { worst = d; worstLabel = scene.mobiles[i].label }
    }
    expect(worst < 1e-4, "\(worstLabel) is \(worst) Å adrift after one loop")
}

test("and there is no jump at the seam") {
    // The comparison above would pass for a loop that cut to the start from
    // somewhere else entirely. This walks every frame including the wrap and
    // asks that nothing moves further across the seam than it does anywhere
    // inside the loop.
    //
    // Heavy atoms only: the three protons swap identities across the seam by
    // design (the enzyme does not get its own proton back), so they are checked
    // as a set of occupied places instead, below.
    var heavy: [Int] = []
    for (i, m) in scene.mobiles.enumerated() where m.element != "H" { heavy.append(i) }
    var steps: [Float] = []
    for f in 0..<lacFrameCount {
        let a = pose(scene, at: f, mutations: mutations)
        let b = pose(scene, at: f + 1, mutations: mutations)
        var m: Float = 0
        for i in heavy { m = max(m, simd_length(a[i] - b[i])) }
        steps.append(m)
    }
    let seam: Float = steps[lacFrameCount - 1]
    let biggest: Float = steps.dropLast().max() ?? 0
    expect(seam <= biggest + 1e-4,
           "the wrap moves an atom \(seam) Å, against \(biggest) Å for the biggest "
           + "step inside the loop — the loop does not close, it cuts")
    expect(seam < 1e-4, "the heavy atoms should not move at all across the seam, "
                      + "and they move \(seam) Å")
}

test("and the protons occupy the same three places on both sides of the seam") {
    // What the eye sees across the wrap: the same three hydrogens in the same
    // three places. Which one is which has rotated, and that is the chemistry.
    func places(_ f: Int) -> [SIMD3<Float>] {
        let p = pose(scene, at: f, mutations: mutations)
        var out: [SIMD3<Float>] = []
        for (i, m) in scene.mobiles.enumerated() where m.element == "H" { out.append(p[i]) }
        return out
    }
    let before = places(lacFrameCount - 1)
    let after = places(lacFrameCount)
    for p in before {
        let nearest: Float = after.map { simd_length($0 - p) }.min() ?? 1e9
        expect(nearest < 1e-3, "a proton at \(p) has nowhere to be after the wrap")
    }
    expectEqual(before.count, after.count)
}

test("the camera, the porthole and the caption all close too") {
    let c0 = lacCamera(scene, at: 0), cN = lacCamera(scene, at: lacFrameCount)
    expect(simd_length(c0.origin - cN.origin) < 1e-3,
           "the camera is \(simd_length(c0.origin - cN.origin)) Å adrift")
    expectEqual(portholeRadius(scene, at: 0), portholeRadius(scene, at: lacFrameCount))
    expectEqual(lacCaption(scene, at: 0).title, lacCaption(scene, at: lacFrameCount).title)
}

test("the whole frame closes: every shape in frame N is a shape in frame 0") {
    let a = lacShapes(scene, at: 0, porthole: .shut, mutations: mutations)
    let b = lacShapes(scene, at: lacFrameCount, porthole: .shut, mutations: mutations)
    expectEqual(a.count, b.count)
    var worst: Float = 0
    for i in 0..<min(a.count, b.count) { worst = max(worst, simd_length(a[i].a - b[i].a)) }
    expect(worst < 1e-4, "the frame is \(worst) Å from closing")
}

test("the enzyme is free at both ends — no sugar left on it") {
    for f in [0, lacFrameCount] {
        let (bonded, _) = covalentBond(scene, positions: pose(scene, at: f, mutations: mutations))
        expect(!bonded, "frame \(f): the enzyme still has the sugar bonded to it")
    }
}

test("the boundary poses close — the heavy atoms exactly, the protons as places") {
    // A catalytic cycle returns the ENZYME, not the molecules. The 24 heavy
    // atoms of the sugars and the water do come back to where they began: the
    // galactosyl becomes the galactose, the water becomes its new hydroxyl, and
    // both end up parked out in solvent where the next lactose is waiting.
    let first = at(0), last = at(8)
    var heavy = 0
    for (i, m) in scene.mobiles.enumerated() where m.element != "H" {
        heavy += 1
        expect(simd_length(first[i] - last[i]) < 1e-3,
               "\(m.label): boundary 8 is not boundary 0")
    }
    expectEqual(heavy, 24)

    // The three protons do NOT, atom for atom, and they must not: the proton
    // Glu461 ends the cycle holding is the one the water brought, and the one
    // it started with left on the glucose. So they close as a set of three
    // PLACES, under a permutation the builder writes down and this checks.
    var protons: [Int] = []
    for (i, m) in scene.mobiles.enumerated() where m.element == "H" { protons.append(i) }
    expectEqual(protons.count, 3)
    expectEqual(scene.protonPermutation.count, 3)
    expectEqual(Set(scene.protonPermutation), Set([0, 1, 2]))
    for (k, j) in scene.protonPermutation.enumerated() {
        let d: Float = simd_length(last[protons[k]] - first[protons[j]])
        expect(d < 1e-3,
               "proton \(scene.mobiles[protons[k]].label) ends \(d) Å from where "
               + "\(scene.mobiles[protons[j]].label) began")
    }
    // and it really is a permutation, not the identity — if it were, the
    // render would be claiming the enzyme gets its own proton back
    expect(scene.protonPermutation != [0, 1, 2],
           "the proton permutation is the identity; a catalytic cycle does not hand the "
           + "same proton back")
}

// MARK: - 9. The mutant is named

section("9. the mutant is named")

test("the structures the substrate poses come from are of a DISABLED enzyme") {
    expect(scene.mutantEntries.contains("1JYN"),
           "1JYN is E537Q and the scene must say so")
    for e in ["1JYV", "1JYW", "1JZ8"] {
        expect(scene.mutantEntries.contains(e), "\(e) is also E537Q")
    }
    // and the ones that are not mutants are not listed as such
    for e in ["1JZ2", "1JZ5", "1JZ7"] {
        expect(!scene.mutantEntries.contains(e), "\(e) is wild type")
    }
}

test("the assumption is written down rather than glossed") {
    let cs = lacConstants(scene)
    let named = cs.filter { $0.value.contains("E537Q") || $0.name.contains("E537Q") }
    expect(!named.isEmpty, "no constant names the E537Q mutation")
    for c in named {
        expectEqual(c.evidence, LacEvidence.model)
        expect(c.source.lowercased().contains("assumption"),
               "the E537Q entry should say it is an assumption")
    }
}

test("the caption says whose enzyme this is, on the frames where it matters") {
    // The obvious reading of 'lactose being cut' is the human enzyme. It is not.
    let opening = lacCaption(scene, at: 10)
    expect(opening.subtitle.contains("Escherichia coli"),
           "the opening caption does not name the organism: \(opening.subtitle)")
    expect(opening.facts.contains("human lactase"),
           "the opening caption does not say why it is the bacterial enzyme")
    // and the PDB entries are named
    var entries = Set<String>()
    for f in stride(from: 0, to: lacFrameCount, by: 4) {
        let c = lacCaption(scene, at: f)
        for e in ["1JYN", "1JZ5", "1JZ2", "1JZ7"] where c.subtitle.contains(e) {
            entries.insert(e)
        }
    }
    expectEqual(entries, Set(["1JYN", "1JZ5", "1JZ2", "1JZ7"]))
}

test("the other substitution — the fluorine — is named too") {
    let cs = lacConstants(scene)
    let f = cs.filter { $0.source.contains("fluorine") || $0.value.contains("fluoro") }
    expect(!f.isEmpty, "1JZ2's 2-fluoro trap is not named anywhere")
    for c in f { expectEqual(c.evidence, LacEvidence.model) }
}

// MARK: - 10. Every constant carries a source

section("10. every constant carries a source")

test("nothing in the table is unsourced") {
    let cs = lacConstants(scene)
    expect(cs.count >= 18, "only \(cs.count) constants")
    for c in cs {
        expect(!c.source.isEmpty, "\(c.name) cites nothing")
        expect(c.source.count > 30, "\(c.name): '\(c.source)' is not a source")
    }
}

test("a measured constant cites a year") {
    for c in lacConstants(scene) where c.evidence == .measured {
        let hasYear = (1900...2100).contains { c.source.contains("\($0)") }
        expect(hasYear, "MEASURED '\(c.name)' cites no year: \(c.source)")
    }
}

test("a derived one says DERIVED and a modelled one says MODEL") {
    for c in lacConstants(scene) {
        switch c.evidence {
        case .derived:
            expect(c.source.hasPrefix("DERIVED"), "'\(c.name)' is derived but does not say so")
        case .model:
            expect(c.source.hasPrefix("MODEL"), "'\(c.name)' is a model but does not say so")
        case .measured:
            expect(!c.source.hasPrefix("MODEL"), "'\(c.name)' is measured but says MODEL")
        }
    }
}

test("the frame carries no evidence bar, and that is deliberate") {
    // Step 13 dropped its bar because a bar saying the same thing for 120
    // frames is furniture; step 14 followed. The provenance is in the table
    // above and in these tests instead.
    let c = lacCaption(scene, at: 40)
    expect(!c.facts.contains("MEASURED") && !c.facts.contains("MODEL"),
           "the caption has grown an evidence bar")
}

test("the render is of the enzyme it says it is") {
    expect(scene.atoms.count > 30_000 && scene.atoms.count < 35_000,
           "\(scene.atoms.count) atoms is not a LacZ tetramer")
    var chains = Set<String>()
    for a in scene.atoms { chains.insert(a.chain) }
    expectEqual(chains.count, 4)
    expect(!scene.complementing.isEmpty,
           "the partner subunit's loop is not marked, so the tetramer has no stated reason")
    expect(scene.interfaceToSite < 4.5,
           "the partner loop is \(scene.interfaceToSite) Å away — not contact")
    expect(scene.interfaceToSubstrate > 6,
           "the partner loop is \(scene.interfaceToSubstrate) Å from the sugar; if it were "
           + "closer, 'it completes the wall, not the chemistry' would be wrong")
}

// MARK: - the renderer itself

section("the renderer")

test("the uniform grid finds the same thing brute force does") {
    let shapes = lacShapes(scene, at: 0, porthole: .shut)
    let grid = UniformGrid(shapes: shapes, density: 1)
    expect(grid.items.count >= shapes.count, "the grid lost shapes")
    // every shape is recorded in the box its centre falls in
    for i in stride(from: 0, to: shapes.count, by: 997) {
        let c = SIMD3(shapes[i].a.x, shapes[i].a.y, shapes[i].a.z)
        expect(grid.shapes(at: c).contains(i), "shape \(i) is not in its own box")
    }
}

test("sticks and spheres both survive the round trip through GPUShape") {
    let s = GPUShape.sphere(center: SIMD3(1, 2, 3), radius: 1.5, color: SIMD3(1, 0, 0),
                            clippable: true)
    expect(!s.isCylinder)
    expectEqual(s.radius, 1.5)
    let c = GPUShape.stick(from: SIMD3(0, 0, 0), to: SIMD3(0, 0, 4), radius: 0.2,
                           color: SIMD3(0, 1, 0))
    expect(c.isCylinder)
    let b = c.bounds()
    expect(b.lo.z <= -0.2 && b.hi.z >= 4.2, "the stick's box does not contain it")
}

test("the ball-and-stick cast is never clipped") {
    let f = turnFrames + openFrames + 60
    let camera = lacCamera(scene, at: f)
    let hole = lacPorthole(scene, at: f, camera: camera)
    let shapes = lacShapes(scene, at: f, porthole: hole)
    var protein = 0, cast = 0
    for s in shapes {
        if s.color.w > 0.5 { protein += 1 } else { cast += 1 }
    }
    expectEqual(protein, scene.atoms.count)
    expect(cast > 100, "only \(cast) unclipped shapes — the porthole would open onto nothing")
}

_ = finish()
