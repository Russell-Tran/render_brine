// The shape of the loop, in one place, so the tests can ask what any frame is
// doing without rendering it.
//
// WHY THE CELL IS EMPTIED AND REFILLED
//
// Hydrolysis does not run backwards. A loop that showed the reaction going and
// then simply cut back to fresh sucrose would be a lie told by the encoder. So
// the loop begins at a bench: the cell drains, and is refilled with fresh
// solution. That is a real thing a polarimetrist does between runs, it takes a
// tenth of the loop, and it makes the GIF close honestly in BOTH panels — the
// camera comes back to where it started, the site comes back to free enzyme,
// and the cell comes back to sucrose by being refilled rather than by running
// the chemistry in reverse.
//
// It also costs something, and the tests are written around it: the reading
// passes through zero once during the reaction, and once more while the cell is
// empty. "Crosses zero exactly once" is therefore asserted over the reaction,
// and a separate test asserts the reading only ever climbs while the cell is
// filling.

import Foundation
import simd

enum Stage: String {
    case drain = "draining the cell"
    case refill = "refilling: fresh sucrose"
    case reaction = "hydrolysis"
}

/// Where the loop is at `u`, 0 to 1.
struct Beat {
    var u: Double
    var stage: Stage
    var fill: Double            // how much of the polarimeter cell holds solution
    var progress: Double        // how far the catalytic cycle has got, 0 to 1
    var turnover: Int           // which turnover, counting from 0
    var cycleU: Double          // position within that turnover
    var zoom: Double            // 0 shows the octamer, 1 shows one dimer
    var opening: Float          // how far the porthole has irised open
    var octamerYaw: Float
    var rockDegrees: Float
}

// The beats of the loop, as fractions of it.
let drainEnds = 0.060
let refillEnds = 0.140
let zoomInEnds = 0.240
let portholeOpensFrom = 0.200
let portholeOpenBy = 0.270
let portholeClosesFrom = 0.840
let portholeShutBy = 0.900
let zoomOutFrom = 0.860
let zoomOutEnds = 0.940
let fillSettles = 0.018

// The loop's length, shared with the tests so they cannot be walking a
// different film from the one that gets encoded. 96 frames at 10 fps is 9.6
// seconds; at 128 palette entries and 120 frames the same loop came out 17.0 MB
// and at these settings 11.9, which is the trade this step makes. The left
// panel's camera moves every frame, so step 8's changed-pixels-only encoding
// has very little to hold on to — the number `make run` prints says how little.
let defaultFrameCount = 96
let defaultDelayCentiseconds = 10
let defaultPaletteSize = 72
let defaultCellMolecules = 96

/// How many complete turnovers the site does over the loop. The cycle is at
/// free enzyme at both ends of every one of them, which is what makes the loop
/// close at the site as well as at the camera.
let turnoversPerLoop = 4

func smoothstep(_ x: Double) -> Double {
    let c: Double = min(max(x, 0), 1)
    return c * c * (3 - 2 * c)
}

private func ramp(_ u: Double, _ from: Double, _ to: Double) -> Double {
    if u <= from { return 0 }
    if u >= to { return 1 }
    return smoothstep((u - from) / (to - from))
}

func beat(at u: Double) -> Beat {
    let t: Double = min(max(u, 0), 1)
    let stage: Stage
    var fill: Double
    var progress: Double
    if t < drainEnds {
        stage = .drain
        fill = 1 - smoothstep(t / drainEnds)
        progress = 1
    } else if t < refillEnds {
        stage = .refill
        // Full a little before the stage ends, so the last frame of the fill is
        // a full cell and not a 98% one. Otherwise the reading is still on its
        // way up when hydrolysis is supposed to have started, and the test that
        // the reading only ever climbs while filling catches it — correctly.
        fill = smoothstep((t - drainEnds) / (refillEnds - drainEnds - fillSettles))
        progress = 0
    } else {
        stage = .reaction
        fill = 1
        progress = (t - refillEnds) / (1.0 - refillEnds)
    }
    progress = min(max(progress, 0), 1)

    let scaled: Double = progress * Double(turnoversPerLoop)
    let turnover: Int = min(Int(scaled), turnoversPerLoop - 1)
    let cycleU: Double = progress >= 1.0 ? 1.0 : scaled - Double(turnover)

    let zoomIn: Double = ramp(t, refillEnds, zoomInEnds)
    let zoomOut: Double = ramp(t, zoomOutFrom, zoomOutEnds)
    let zoom: Double = zoomIn * (1 - zoomOut)

    let openUp: Double = ramp(t, portholeOpensFrom, portholeOpenBy)
    let shut: Double = ramp(t, portholeClosesFrom, portholeShutBy)
    let opening: Double = openUp * (1 - shut)

    // The close-up rocks rather than orbits: the porthole has to keep facing
    // the camera, and a full turn would carry the site round the back.
    var rock: Double = 0
    if t > zoomInEnds && t < zoomOutFrom {
        let span: Double = zoomOutFrom - zoomInEnds
        rock = sin(2 * Double.pi * (t - zoomInEnds) / span) * 13.0
    }

    return Beat(u: t, stage: stage, fill: fill, progress: progress,
                turnover: turnover, cycleU: cycleU, zoom: zoom,
                opening: Float(opening), octamerYaw: Float(360.0 * t),
                rockDegrees: Float(rock))
}

// MARK: - Cameras

let octamerDistance: Float = 372
let dimerDistance: Float = 112
let fieldOfView: Float = 30

/// The molecule panel's camera: an orbit of the whole octamer blended with a
/// fixed look straight down the funnel at one active site. The blend is the
/// zoom, so the move in is a dolly and there is nothing to cut.
func moleculeCamera(_ b: Beat, scene: InvertaseScene) -> Camera {
    let wide = Camera.orbit(target: .zero, distance: octamerDistance,
                            yaw: b.octamerYaw, pitch: 14, fov: fieldOfView)
    // A rock about whichever world axis is least parallel to the funnel.
    let axis = scene.siteAxis
    let helper: SIMD3<Float> = abs(axis.y) > 0.9 ? SIMD3(1, 0, 0) : SIMD3(0, 1, 0)
    let pivot: SIMD3<Float> = simd_normalize(simd_cross(axis, helper))
    let looking: SIMD3<Float> = rotate(axis, about: pivot, degrees: b.rockDegrees)
    let close = Camera(origin: scene.siteCentre + looking * dimerDistance,
                       target: scene.siteCentre, fov: fieldOfView)
    let k = Float(smoothstep(b.zoom))
    let origin: SIMD3<Float> = wide.origin + (close.origin - wide.origin) * k
    let target: SIMD3<Float> = wide.target + (close.target - wide.target) * k
    return Camera(origin: origin, target: target, fov: fieldOfView)
}

/// The bench's camera never moves. The polarimeter is an instrument on a table;
/// swinging round it would say something about the instrument rather than about
/// the number, and a still camera is also the cheapest thing a GIF can encode.
func polarimeterCamera() -> Camera {
    let direction = simd_normalize(SIMD3<Float>(0.46, 0.36, 1.00))
    let target = SIMD3<Float>(-6, 1, 0)
    return Camera(origin: target + direction * 352, target: target, fov: fieldOfView)
}

// MARK: - The molecule panel's geometry

/// Every atom's survival factor: 1 for the dimer, fading for the six chains of
/// the other three dimers as the camera moves in, then the porthole on top.
///
/// Nothing is clipped. An atom on its way out shrinks, so every surface in the
/// frame is a whole sphere with its normal pointing outwards, and there is no
/// cut face to cap and no backface to see through.
func survivalFactors(_ scene: InvertaseScene, zoom: Double, opening: Float) -> [Float] {
    let dimer = Set(scene.dimerChains)
    let fade: Float = 1 - Float(smoothstep(zoom))
    var keep = [Float](repeating: 1, count: scene.atoms.count)
    for (i, a) in scene.atoms.enumerated() where !dimer.contains(a.chain) {
        keep[i] = fade
    }
    if opening <= 0 { return keep }
    return portholeFactors(scene.atoms, keep: keep, opening: opening,
                           centre: scene.siteCentre, axis: scene.siteAxis,
                           radius: portholeRadius)
}

let portholeRadius: Float = 15.0

/// The whole molecule panel, ready for the grid.
func moleculeScene(_ scene: InvertaseScene, beat b: Beat) -> [GPUPrim] {
    var prims: [GPUPrim] = []
    prims.reserveCapacity(scene.atoms.count + 400)
    let keep = survivalFactors(scene, zoom: b.zoom, opening: b.opening)
    let partner: String = scene.dimerChains.count > 1 ? scene.dimerChains[1] : "B"
    let catalyticSeqs: Set<Int> = [scene.catalytic.nucleophilePDB, scene.catalytic.acidBasePDB,
                                   scene.catalytic.stabiliserPDB]
    for (i, a) in scene.atoms.enumerated() {
        let survives: Float = keep[i]
        if survives <= 0.02 { continue }
        // The catalytic side chains are drawn ball-and-stick below, so their
        // space-filling copies would only bury them.
        if a.chain == "A" && catalyticSeqs.contains(a.seq) && b.opening > 0.3 { continue }
        let colour: SIMD3<Float>
        if a.chain == "A" {
            colour = spectrumColour(a.t)
        } else if a.chain == partner {
            colour = partnerColour
        } else {
            colour = otherChainColour
        }
        let r: Float = vdwRadius(a.element) * survives
        prims.append(.sphere(a.position, r, colour))
    }

    // Inside the porthole: the three catalytic residues, and the chemistry.
    let strength: Float = b.opening
    if strength > 0.02 {
        let (spheres, sticks) = catalyticSticks(scene, chain: "A", scale: 1, strength: strength)
        for s in spheres { prims.append(.sphere(s.0, s.1, s.2)) }
        for s in sticks { prims.append(.stick(s.0, s.1, siteBondRadius * strength, s.2)) }
        let site = siteGeometry(scene, at: b.cycleU, scale: 1)
        for s in site.spheres { prims.append(.sphere(s.position, s.radius * strength, s.colour)) }
        for s in site.sticks { prims.append(.stick(s.a, s.b, siteBondRadius * strength, s.colour)) }
    }
    return prims
}

// MARK: - What the bar says
//
// The caption lives here rather than in main.swift so the tests can read it.
// They insist on two things: that the resolution is on it, because 3.40 Å is
// coarse for a space-filling render and the picture will not look it; and that
// every frame with sucrose in the site says MODEL, because no crystal of this
// enzyme has ever had sucrose in it.

func stageName(_ b: Beat) -> String {
    if b.stage != .reaction { return b.stage.rawValue }
    let position = cyclePosition(b.cycleU)
    switch position.from {
    case "free": return position.blend > 0.5 ? "sucrose binding" : "free enzyme"
    case "michaelis": return "Asp23 attacks C2 · Glu204 protonates"
    case "covalent": return "β-fructosyl–enzyme · glucose leaves"
    case "hydrolysis": return "water breaks the ester"
    case "product": return "β-D-fructose leaves"
    default: return "hydrolysis"
    }
}

func caption(_ scene: InvertaseScene, _ b: Beat, readout: PolarimeterReadout,
             invertRotation: Double) -> Caption {
    let invert = String(format: "%.1f", invertRotation)
    let subtitle = String(format:
        "sucrose [α]D +%.1f° splits into glucose +%.1f° and fructose %.1f°, mixing to %@° — "
        + "the solution changes hand, the sugar does not",
        sucrose.specificRotation, glucose.specificRotation, fructose.specificRotation, invert)
    let facts = String(format:
        "%@ invertase (%@) · PDB %@ at %.2f Å · the octamer is the biological unit "
        + "· GH32: RETAINING, two inversions at C2, net retention",
        scene.organism, scene.gene, scene.pdb, scene.resolution)
    let aside: String
    switch b.stage {
    case .drain, .refill:
        aside = String(format: "cell %.0f%% full · sodium D line %.2f nm (%@)",
                       readout.fill * 100, sodiumDLineNanometres,
                       hexString(nanometres: Float(sodiumDLineNanometres)))
    case .reaction:
        aside = String(format: "%d of %d molecules split · turnover %d of %d · pose MODEL from PDB %@",
                       readout.splitGlyphs, readout.splitGlyphs + readout.sucroseGlyphs,
                       b.turnover + 1, turnoversPerLoop, scene.template.pdb)
    }
    return Caption(title: "Invert sugar", subtitle: subtitle, facts: facts, aside: aside)
}

