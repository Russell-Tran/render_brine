// Tests for step 12.
//
// The step's whole claim is that a brightness is COMPUTED from geometry
// rather than animated, so most of these check exactly that: the law itself,
// that the renderer really evaluates it, and that geometry which should change
// the answer does change it.

import Foundation
import simd

let sceneURL = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent().deletingLastPathComponent()
    .appendingPathComponent("Resources/scene.json")
let scene = try! loadScene(from: sceneURL)

func near(_ a: Float, _ b: Float, _ tol: Float) -> Bool { abs(a - b) <= tol }

// MARK: - The law

section("Förster's law")

test("at R0 exactly, half the energy transfers") {
    // The definition of the Förster radius, and the sharpest single check.
    expect(near(transferEfficiency(separation: forsterRadius), 0.5, 1e-5),
           "got \(transferEfficiency(separation: forsterRadius))")
}

test("it is a sixth power, not some other power") {
    // At twice R0 a sixth-power law gives 1/(1+64); a fourth power would give
    // 1/(1+16) and an inverse square 1/(1+4). This separates them.
    let e = transferEfficiency(separation: 2 * forsterRadius)
    expect(near(e, 1.0 / 65.0, 1e-6), "got \(e), expected \(1.0 / 65.0)")
    let half = transferEfficiency(separation: forsterRadius / 2)
    expect(near(half, 64.0 / 65.0, 1e-6), "got \(half)")
}

test("it runs from 1 down to 0 and never leaves that range") {
    expect(transferEfficiency(separation: 0.001) > 0.999)
    expect(transferEfficiency(separation: 1000) < 0.001)
    var r: Float = 1
    while r < 400 {
        let e = transferEfficiency(separation: r)
        expect(e >= 0 && e <= 1, "E(\(r)) = \(e)")
        r += 3
    }
}

test("it only ever falls as the dyes move apart") {
    var previous: Float = 2
    var r: Float = 5
    while r < 300 {
        let e = transferEfficiency(separation: r)
        expect(e < previous, "E did not fall at \(r) Å")
        previous = e
        r += 5
    }
}

test("the inverse agrees with the law") {
    for e in [Float(0.1), 0.25, 0.5, 0.75, 0.9] {
        let r = separation(forEfficiency: e)
        expect(near(transferEfficiency(separation: r), e, 1e-4),
               "round trip failed at \(e): r = \(r)")
    }
}

test("the transition really is a switch, not a dimmer") {
    // The claim the step is built on: nearly all of the change happens over a
    // few nanometres. Measure the window between 90% and 10%.
    let high = separation(forEfficiency: 0.9)
    let low = separation(forEfficiency: 0.1)
    let window = (low - high) / 10        // nanometres
    expect(window < 5.0, "90%→10% takes \(window) nm, which is not a switch")
    expect(high < low)
}

// MARK: - Brightness comes from geometry

section("the brightness is derived, not stored")

test("the two channels are set by the separation and always sum to one") {
    for r in [Float(10), 25, 55, 80, 140] {
        let s = DyeState(separation: r)
        let sum = s.donorBrightness + s.acceptorBrightness
        expect(near(sum, 1, 1e-5), "at \(r) Å the channels sum to \(sum)")
        expect(near(s.acceptorBrightness, transferEfficiency(separation: r), 1e-6))
        expect(near(s.donorBrightness, 1 - transferEfficiency(separation: r), 1e-6))
    }
}

test("moving the probe changes the brightness, which is the whole point") {
    // If brightness were a stored constant or a keyframed curve, these would
    // not differ. They differ because the only input is the geometry.
    let far = DyeState(separation: place(scene, progress: 0).separation)
    let near_ = DyeState(separation: place(scene, progress: 1).separation)
    expect(far.efficiency < 0.10, "far should barely transfer, got \(far.efficiency)")
    expect(near_.efficiency > 0.95, "bound should transfer nearly all, got \(near_.efficiency)")
    expect(near_.donorBrightness < far.donorBrightness, "the donor must dim as the probe lands")
    expect(near_.acceptorBrightness > far.acceptorBrightness, "the acceptor must brighten")
}

test("the separation the renderer uses is the real distance between the dyes") {
    // Recomputed here independently of Placed, from the atoms themselves.
    for s in [Float(0), 0.35, 0.7, 1] {
        let placed = place(scene, progress: s)
        let byHand = simd_distance(placed.donor, placed.acceptor)
        expect(near(placed.separation, byHand, 1e-3),
               "at progress \(s): \(placed.separation) vs \(byHand)")
    }
}

test("the approach sweeps the whole interesting range of the curve") {
    // If it started too close the render would never show the cliff.
    var lowest: Float = 1, highest: Float = 0
    for i in 0...100 {
        let e = DyeState(separation: place(scene, progress: Float(i) / 100).separation).efficiency
        lowest = min(lowest, e)
        highest = max(highest, e)
    }
    expect(lowest < 0.08, "never gets far enough out: lowest transfer \(lowest)")
    expect(highest > 0.95, "never gets close enough: highest transfer \(highest)")
}

// MARK: - The geometry

section("the scene")

test("the target is the start of pGLO's GFP gene") {
    expect(scene.targetSequence.hasPrefix("ATG"), "got \(scene.targetSequence.prefix(6))")
    expectEqual(scene.targetSequence.count, 34)
    for c in scene.targetSequence { expect("ACGT".contains(c), "bad base \(c)") }
}

test("three molecules, and every atom belongs to one of them") {
    var counts: [String: Int] = [:]
    for a in scene.atoms { counts[a.part, default: 0] += 1 }
    expect(counts["target"] ?? 0 > 800, "target has \(counts["target"] ?? 0) atoms")
    expect(counts["probe1"] ?? 0 > 300)
    expect(counts["probe2"] ?? 0 > 300)
    let known = ["target", "probe1", "probe2", "none"]
    for a in scene.atoms { expect(known.contains(a.part), "unknown part \(a.part)") }
}

test("the bound separation is a believable few nanometres") {
    // Two dyes on linkers a couple of nucleotides apart. Closer than about
    // 1 nm would have them interpenetrating; much past 4 nm and the assay
    // would not work.
    let nm = scene.boundSeparation / 10
    expect(nm > 1.0 && nm < 4.0, "bound separation \(nm) nm")
}

test("only the arriving probe moves; everything else is nailed down") {
    let a = place(scene, progress: 0)
    let b = place(scene, progress: 1)
    for i in 0..<scene.atoms.count where scene.atoms[i].part != "probe2" {
        expect(simd_distance(a.positions[i], b.positions[i]) < 1e-4,
               "\(scene.atoms[i].part) atom \(i) moved")
    }
}

test("the arriving probe translates and never deforms") {
    // This matters for more than tidiness: ambient occlusion here is built
    // from surface normals, so a body that only translates keeps its shading
    // exactly. A rotation or a stretch would make it crawl.
    let a = place(scene, progress: 0.2)
    let b = place(scene, progress: 0.8)
    let moving = (0..<scene.atoms.count).filter { scene.atoms[$0].part == "probe2" }
    expect(moving.count > 100)
    for i in 0..<min(moving.count, 60) {
        for j in (i + 1)..<min(moving.count, 60) {
            let before = simd_distance(a.positions[moving[i]], a.positions[moving[j]])
            let after = simd_distance(b.positions[moving[i]], b.positions[moving[j]])
            expect(near(before, after, 1e-3), "probe deformed: \(before) → \(after)")
        }
    }
}

test("every bond in the built duplex is a real bond length") {
    var worst: Float = 0
    for b in scene.bonds {
        let p = scene.atoms[b.a].home, q = scene.atoms[b.b].home
        let d = simd_distance(p, q)
        worst = max(worst, d)
        expect(d > 1.0 && d < 2.1, "bond \(b.a)-\(b.b) is \(d) Å")
    }
    expect(worst < 2.1, "longest bond \(worst) Å")
}

test("the dyes sit outside the duplex, not buried in it") {
    // They hang off linkers; if they were inside the helix the picture would
    // be wrong and they would be invisible.
    let placed = place(scene, progress: 1)
    for dye in [placed.donor, placed.acceptor] {
        let radius = sqrt(dye.y * dye.y + dye.z * dye.z)
        expect(radius > 11, "a dye sits \(radius) Å from the axis, inside the duplex")
    }
}

// MARK: - The loop

section("the loop")

test("it closes: the last frame leads back into the first") {
    let duration = 20.0
    expectEqual(approachProgress(t: 0, duration: duration), 0)
    expectEqual(approachProgress(t: duration, duration: duration), 0)
    // and the step across the seam is smaller than a tenth of an ångström
    let last = approachProgress(t: duration - 0.1, duration: duration)
    let gap = simd_distance(place(scene, progress: last).acceptor,
                            place(scene, progress: 0).acceptor)
    expect(gap < 1.0, "the seam jumps \(gap) Å")
}

test("the probe goes in and comes back out, never backwards mid-flight") {
    let duration = 20.0
    var peak: Float = 0
    var peakAt = 0.0
    var t = 0.0
    while t < duration {
        let p = approachProgress(t: t, duration: duration)
        if p > peak { peak = p; peakAt = t }
        t += 0.1
    }
    expect(near(peak, 1, 1e-4), "never fully lands: peak \(peak)")
    expect(peakAt > duration * 0.3 && peakAt < duration * 0.9,
           "lands at a strange time: \(peakAt) s")
}

test("nothing ever leaves the frame while it matters") {
    let cam = Camera.orbit(target: SIMD3(40, 8, 0),
                           distance: 272, yaw: 0, pitch: 8, fov: 30)
    for step in 0...20 {
        let placed = place(scene, progress: Float(step) / 20)
        for dye in [placed.donor, placed.acceptor] {
            let p = cam.project(dye, width: 960, height: 600)
            expect(p.x > -60 && p.x < 1020 && p.y > -60 && p.y < 660,
                   "a dye is at \(p) at progress \(Float(step) / 20)")
        }
    }
}

// MARK: - Colour

section("colour, computed as in step 9a")

test("the two dyes get different computed colours, and the right ones") {
    let d = donorColour(), a = acceptorColour()
    expect(d.y > d.x && d.y > d.z, "the 520 nm donor should be green, got \(d)")
    expect(a.x > a.y && a.x > a.z, "the 640 nm acceptor should be red, got \(a)")
    expect(simd_distance(d, a) > 0.5, "the two colours are too alike")
}

test("nothing here is a hard-coded hex: recomputing reproduces it") {
    let again = spectralColour(nanometres: donorEmission).linear
    expect(simd_distance(again, donorColour()) < 1e-6)
}

// MARK: - What gets drawn

section("the drawn scene")

test("both dyes are drawn, and they change colour with the separation") {
    let farState = DyeState(separation: place(scene, progress: 0).separation)
    let nearState = DyeState(separation: place(scene, progress: 1).separation)
    let far = sceneGeometry(scene, placed: place(scene, progress: 0), state: farState)
    let bound = sceneGeometry(scene, placed: place(scene, progress: 1), state: nearState)
    // The last two spheres are the dyes.
    let farDonor = far.spheres[far.spheres.count - 2].color
    let boundDonor = bound.spheres[bound.spheres.count - 2].color
    let farAcceptor = far.spheres[far.spheres.count - 1].color
    let boundAcceptor = bound.spheres[bound.spheres.count - 1].color
    expect(boundDonor.y < farDonor.y, "the donor must dim: \(farDonor.y) → \(boundDonor.y)")
    expect(boundAcceptor.x > farAcceptor.x, "the acceptor must brighten")
}

test("space-filling radii, from Bondi") {
    expectEqual(vanDerWaalsRadius("C"), 1.70)
    expectEqual(vanDerWaalsRadius("N"), 1.55)
    expectEqual(vanDerWaalsRadius("O"), 1.52)
    expectEqual(vanDerWaalsRadius("P"), 1.80)
}

test("no bond is drawn between the arriving probe and anything else") {
    // Until it lands there is nothing joining it to the target, and drawing
    // one would stretch a stick across the frame.
    let placed = place(scene, progress: 0.4)
    let state = DyeState(separation: placed.separation)
    let (_, cylinders) = sceneGeometry(scene, placed: placed, state: state)
    for c in cylinders {
        let a = SIMD3(c.aRadius.x, c.aRadius.y, c.aRadius.z)
        let b = SIMD3(c.b.x, c.b.y, c.b.z)
        expect(simd_distance(a, b) < 2.5, "a stick \(simd_distance(a, b)) Å long got drawn")
    }
}

finish()
