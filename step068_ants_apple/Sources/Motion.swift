// Step 68's clock, and step 30's tap on it.
//
// ONE CLOCK. Everything in the loop — walking, tapping, drinking, turning,
// the molecules — runs slowed ×10, the slow-down step 30 chose for the tap:
//
//   * Rate. Ants' antennal movements run at 3–6 strokes per second: O'Fallon,
//     Suarez & Smith 2016, *Insectes Sociaux* 63, write "(3–6 strokes/s;
//     Lenoir 1982)", citing Lenoir, *Behav Process* 7: 27–35 (checked in the
//     paper itself for step 33, 2026-09-27). 4 per second, a real tap every
//     quarter second, played over 2.5 s: slowed ×10. Step 30's.
//   * The walk is step 44's 30 mm/s at 25 °C (Walk.swift), slowed by the
//     same ×10 — so the ants walk 3 mm a second on screen, not step 44's 2.
//   * Share of each tap spent touching: step 30's 40%, MODEL.
//   * Lift: MODEL, 0.15 mm — twice step 30's 0.07 mm. The main view here is
//     wider (three ants and a piece of apple), and 0.07 mm would be 4 pixels;
//     0.15 mm is 10. An antenna drumming on food lifts at least this much;
//     no measured amplitude was found.
//   * The fructose and the odour: MODEL, schematic, as steps 30 and 50 — the
//     order of events, not their speed. One fructose enters ant 0's taste
//     pore per touch of its right antenna, and only while the hair is in the
//     juice; the odour keeps arriving at the smell hair, one molecule a tap.
//
// Browning is the only thing a cut apple does on a clock of minutes, and it
// is left out (Food.swift): nothing here runs on a second clock.
//
// The loop is FORWARD: each ant's trip repeats every loop, off-screen between
// trips; the fructose advances a place per touch of ant 0, 3 a loop; the odour
// makes whole trips; the odorants turn once. The last frame runs on into the
// first. Never a rewind.

import Foundation
import simd

/// Real antennal strokes per second (Lenoir 1982, via O'Fallon et al. 2016:
/// 3–6 strokes/s); 4 is inside that range. Step 30's.
let realStrokesPerSecond: Float = 4
let realStrokeRange: ClosedRange<Float> = 3...6

/// Seconds per tap on screen. MODEL: slow enough to read. Step 30's.
let tapSeconds: Float = 2.5
/// How much slower than life, for everything: a real tap takes 1/4 s, this one 2.5 s.
let slowdown: Float = tapSeconds * realStrokesPerSecond
/// The loop: ten taps' time. Each ant's trip — in, three touches and a
/// drink, turn, out — takes a little under it (Walk.swift).
let tapsPerLoop: Int = 10
let loopSeconds: Float = tapSeconds * Float(tapsPerLoop)

/// Share of each tap with the tip resting on the juice. MODEL, step 30's.
let contactFraction: Float = 0.40
/// How far the tip moves off the juice, mm (see above). MODEL.
func liftHeight(_ m: Mutant) -> Float { m == .stillTap ? 0.003 : 0.15 }

/// Where a time falls in the loop. The rewind mutant plays the first half
/// forward and then backward — continuous at the seam, but a rewind.
func effectiveTime(_ t: Float, mutant: Mutant) -> Float {
    let w: Float = t.truncatingRemainder(dividingBy: loopSeconds)
    let u: Float = w < 0 ? w + loopSeconds : w
    if mutant == .rewind { return u < loopSeconds / 2 ? 2 * u : 2 * (loopSeconds - u) }
    return u
}

/// Tip height off the juice at a tap phase: zero while touching, then up
/// and back down on a sin² curve, so it leaves and lands at zero speed.
/// Step 30's.
func liftAt(phase: Float, height: Float) -> Float {
    var p: Float = phase - phase.rounded(.down)
    if p < 0 { p += 1 }
    if p < contactFraction { return 0 }
    let x: Float = (p - contactFraction) / (1 - contactFraction)
    let s: Float = sin(Float.pi * x)
    return height * s * s
}

func smoothstep01(_ x: Float) -> Float {
    let c: Float = min(max(x, 0), 1)
    let rise: Float = 3 - 2 * c
    return c * c * rise
}

/// Smoother still (zero first and second derivative at both ends).
func smootherstep01(_ x: Float) -> Float {
    let c: Float = min(max(x, 0), 1)
    let sixC: Float = c * 6
    let inner: Float = c * (sixC - 15) + 10
    let cube: Float = c * c * c
    return cube * inner
}

// MARK: - the taste inset: fructose, one per touch of ant 0

/// Fructose spacing along the drift path, Å — one per touch. MODEL: step 30's
/// 13 Å was for sucrose, 12 heavy atoms longer; fructose is half its size,
/// so 10 Å keeps neighbours apart (a test checks never within 2 Å).
let moleculeSpacing: Float = 10.0
/// Fructose drawn at once, in the 24 Å field. Step 30's four.
let moleculeSlots: Int = 4

/// One fructose in the molecule inset: its rotation and where its centroid is.
struct MoleculePose {
    var rotation: simd_quatf
    var offset: SIMD3<Float>

    func place(_ p: SIMD3<Float>) -> SIMD3<Float> { rotation.act(p) + offset }
}

/// The drift path, in the molecule inset's screen plane: the direction that
/// points up the taste hair into its pore, as the micrometre inset shows it.
/// Step 30's.
func driftDirection(hairAxis: SIMD3<Float>, insetRight: SIMD3<Float>, insetUp: SIMD3<Float>) -> SIMD2<Float> {
    let into: SIMD3<Float> = -hairAxis
    return simd_normalize(SIMD2<Float>(simd_dot(into, insetRight), simd_dot(into, insetUp)))
}

/// Every pose depends only on how far along the path that fructose is (and,
/// for a small thermal wobble, on time) — so after one touch each stands
/// exactly where the one ahead of it stood. Step 35's, as step 50 split it.
func moleculePoses(progress u: Float, wobblePhase: Float, direction d: SIMD2<Float>) -> [MoleculePose] {
    let frac: Float = u - u.rounded(.down)
    let along = SIMD3<Float>(d.x, d.y, 0)
    let across = SIMD3<Float>(-d.y, d.x, 0)
    let wobbleTurn: Float = 2 * Float.pi
    let wobbleAngle: Float = 0.10 * sin(wobbleTurn * wobblePhase)
    let wobble = simd_quatf(angle: wobbleAngle, axis: simd_normalize(SIMD3<Float>(1, 0.4, -0.3)))
    let twoPi: Float = 2 * Float.pi
    var out: [MoleculePose] = []
    for k in 0..<moleculeSlots {
        let place: Float = Float(k) - 2 + frac
        let s: Float = place * moleculeSpacing
        let tumbleAngle: Float = 0.16 * s
        let tumble = simd_quatf(angle: tumbleAngle, axis: simd_normalize(SIMD3<Float>(0.5, -0.2, 1)))
        let sideAngle: Float = s * twoPi / 29
        let depthAngle: Float = s * twoPi / 37
        let side: Float = 1.6 * sin(sideAngle)
        let depth: Float = -2.5 * cos(depthAngle)
        let shift: SIMD3<Float> = along * s + across * side
        out.append(MoleculePose(rotation: wobble * tumble, offset: shift + SIMD3<Float>(0, 0, depth)))
    }
    return out
}

/// How many taps' worth of odour has arrived: time in taps, so the dots
/// drift steadily, touching or not. The rewind mutant runs it back.
func odourProgress(_ t: Float, mutant: Mutant) -> Float {
    effectiveTime(t, mutant: mutant) / tapSeconds
}

/// The odorants in their inset each turn once a loop about a fixed axis —
/// rigidly, forward. MODEL, schematic.
let odorantSpinAxis: SIMD3<Float> = simd_normalize(SIMD3<Float>(0.3, 1, 0.2))
func odorantSpin(_ t: Float, mutant: Mutant) -> simd_quatf {
    let a: Float = 2 * Float.pi * effectiveTime(t, mutant: mutant) / loopSeconds
    return simd_quatf(angle: a, axis: odorantSpinAxis)
}

// MARK: - the micrometre inset rides on ant 0's right antenna tip

/// Step 30's, as step 50 generalised it: a point `q` in the inset (µm,
/// attached to the antenna as it was at contact) lies over juice point
/// `R q + T`. R is the apical segment's turn since contact, T the rest, both
/// turned into the inset's axes by the rotation that takes the juice's
/// normal to +y. Here the ant also walks and turns, so R and T carry the
/// whole ant's motion since its touch — the same formula, larger numbers.
struct CrystalFrame {
    var rotation: simd_float3x3
    var translation: SIMD3<Float>   // µm

    func toCrystal(_ q: SIMD3<Float>) -> SIMD3<Float> { rotation * q + translation }
}

func insetBasis(normal n: SIMD3<Float>) -> simd_float3x3 {
    let up = SIMD3<Float>(0, 1, 0)
    return simd_distance(n, up) < 1e-7 ? matrix_identity_float3x3 : simd_float3x3(simd_quatf(from: simd_normalize(n), to: up))
}

/// `contact` and `now` are the antenna's apical segment in the world (end a,
/// end b = the tip's centre), at the touch and now. A point attached to the
/// antenna that was at world w at the touch is now at c1 + r (w − c0); the
/// inset's point q sat at the touch at p + Mᵀq/1000 (M: the rotation taking
/// the juice's normal to +y). So its juice coordinates now are
/// M r Mᵀ q + 1000 M (c1 − p + r (p − c0)). Step 50's formula.
func crystalFrame(contact: (a: SIMD3<Float>, b: SIMD3<Float>), now: (a: SIMD3<Float>, b: SIMD3<Float>),
                  contactPoint p: SIMD3<Float>, normal n: SIMD3<Float>) -> CrystalFrame {
    let d0: SIMD3<Float> = simd_normalize(contact.b - contact.a)
    let d1: SIMD3<Float> = simd_normalize(now.b - now.a)
    let r: simd_float3x3 = simd_distance(d0, d1) < 1e-7 ? matrix_identity_float3x3 : simd_float3x3(simd_quatf(from: d0, to: d1))
    let c0: SIMD3<Float> = contact.b
    let c1: SIMD3<Float> = now.b
    let turned: SIMD3<Float> = r * (p - c0)
    let tMM: SIMD3<Float> = c1 + turned - p
    let m: simd_float3x3 = insetBasis(normal: n)
    let rInset: simd_float3x3 = m * r * m.transpose
    let tInset: SIMD3<Float> = m * tMM
    return CrystalFrame(rotation: rInset, translation: tInset * 1000)
}

/// The GIF's delay per frame, hundredths of a second: 10 fps, 200 frames a
/// loop. Chosen after measuring (main.swift).
let defaultDelayCentiseconds: Int = 10
/// Frame time for frame `f` of `n`.
func frameTime(_ f: Int, of n: Int) -> Float { loopSeconds * Float(f) / Float(n) }
