// Step 50: a new scene, animated from the start (there is no still). Step
// 43's staging of a food bigger than the ant, with step 35's sugars: the
// right antenna taps the juice on the cut face of a sweet-corn kernel, again
// and again; one glucose goes into the taste pore at each touch, and the
// smell of raw sweet corn keeps arriving at the smell hair. Everything else
// holds still, apart from a much smaller sway of the left antenna.
//
// The tap is step 30's, unchanged — its rate, slow-down, lift and timing:
//
//   * Rate. Ants' antennal movements run at 3–6 strokes per second: O'Fallon,
//     Suarez & Smith 2016, *Insectes Sociaux* 63, write "(3–6 strokes/s;
//     Lenoir 1982)", citing Lenoir, *Behav Process* 7: 27–35, on antennal
//     communication during trophallaxis in Myrmica rubra (checked in the
//     paper itself for step 33, 2026-09-27). No measured rate for an ant
//     tapping food turned up, so that rate is used: 4 per second, a real tap
//     every quarter second. The render plays each tap over 2.5 s: slowed ×10,
//     and labelled so.
//   * Amplitude and the share of each tap spent touching: MODEL, step 30's —
//     the tip moves 0.07 mm off the juice, about three-quarters of its own
//     0.096 mm thickness, and rests on it for 40% of each tap. "Off" is along
//     the juice's outward normal, which on the upright cut face points
//     straight back towards the ant: the tip draws back from the face.
//   * The glucose: MODEL, schematic, as step 30's sucrose and step 35's
//     sugars. Small molecules cross a few nanometres of water in nanoseconds;
//     the inset makes no claim about speed. It shows the direction of travel
//     — out of the juice and into the pore, one glucose per touch — and only
//     while the hair is in the juice, because only then is there a liquid path.
//   * The odour: MODEL, schematic, as step 43's. A molecule crosses the few
//     micrometres drawn here in about a microsecond; no watchable slow-down
//     could show that. The inset shows the order of events instead: one
//     molecule reaches a wall pore of the smell hair each tap and goes in,
//     while the next ones are on their way, touching or not.
//
// The loop is FORWARD: four whole taps; each touch takes each glucose a place
// further in, each odour molecule makes its trip once per loop, the four a
// tap apart, and the odorant in its own inset turns once — so the last frame
// runs straight on into the first. Never a rewind.

import Foundation
import simd

/// Real antennal strokes per second (Lenoir 1982, via O'Fallon et al. 2016:
/// 3–6 strokes/s); 4 is inside that range. Step 30's.
let realStrokesPerSecond: Float = 4
let realStrokeRange: ClosedRange<Float> = 3...6

/// Seconds per tap on screen. MODEL: slow enough to read. Step 30's.
let tapSeconds: Float = 2.5
/// How much slower than life: a real tap takes 1/4 s, this one 2.5 s.
let slowdown: Float = tapSeconds * realStrokesPerSecond
/// Taps in one loop.
let tapsPerLoop: Int = 4
let loopSeconds: Float = tapSeconds * Float(tapsPerLoop)

/// Share of each tap with the tip resting on the juice. MODEL, step 30's.
let contactFraction: Float = 0.40
/// How far the tip moves off the juice, mm. MODEL: slight — ¾ of the tip's own diameter.
let liftHeight: Float = 0.07
/// The left antenna's sway, mm: a much smaller, slower-looking drift. MODEL.
let swayAmplitude: Float = 0.018

/// Where a time falls in the loop. The rewind mutant plays the first half
/// forward and then backward — continuous at the seam, but a rewind.
func effectiveTime(_ t: Float, mutant: Mutant) -> Float {
    let w: Float = t.truncatingRemainder(dividingBy: loopSeconds)
    let u: Float = w < 0 ? w + loopSeconds : w
    if mutant == .rewind { return u < loopSeconds / 2 ? 2 * u : 2 * (loopSeconds - u) }
    return u
}

/// Phase of the current tap, 0 at touch-down, and which tap it is.
func tapPhase(_ t: Float, mutant: Mutant) -> (phase: Float, tap: Int) {
    let e: Float = effectiveTime(t, mutant: mutant) / tapSeconds
    let k: Float = e.rounded(.down)
    return (e - k, Int(k))
}

/// Tip height off the juice at a phase: zero while touching, then up and
/// back down on a sin² curve, so it leaves and lands at zero speed.
func liftAt(phase: Float) -> Float {
    if phase < contactFraction { return 0 }
    let x: Float = (phase - contactFraction) / (1 - contactFraction)
    let s: Float = sin(Float.pi * x)
    return liftHeight * s * s
}

func smoothstep01(_ x: Float) -> Float {
    let c: Float = min(max(x, 0), 1)
    let rise: Float = 3 - 2 * c
    return c * c * rise
}

/// How many places the glucose has advanced: one per tap, moving only
/// while the hair is in the juice. Step 30's moleculeProgress.
func sugarProgress(_ t: Float, mutant: Mutant) -> Float {
    let (phase, tap) = tapPhase(t, mutant: mutant)
    return Float(tap) + smoothstep01(phase / contactFraction)
}

/// Glucose spacing along the drift path, Å — one per tap. MODEL: step 35's
/// 14 Å alternated fructose and glucose; with glucose after glucose it
/// brought one's water within 1.85 Å of the next (a test caught it), so 15.
let moleculeSpacing: Float = 15.0
/// Glucose drawn at once: between touches one sits in the middle of the
/// 24 Å field with its neighbours cut by the edge; the fourth is the one
/// arriving next. Step 35's.
let moleculeSlots: Int = 4

/// One glucose (and its waters) in the molecule inset: its rotation and
/// where its centroid is. Step 30's.
struct MoleculePose {
    var rotation: simd_quatf
    var offset: SIMD3<Float>

    func place(_ p: SIMD3<Float>) -> SIMD3<Float> { rotation.act(p) + offset }
}

/// The drift path, in the molecule inset's screen plane (x right, y up):
/// the direction that points up the taste hair, into its pore, as the
/// micrometre inset shows it. Step 30's.
func driftDirection(hairAxis: SIMD3<Float>, insetRight: SIMD3<Float>, insetUp: SIMD3<Float>) -> SIMD2<Float> {
    let into: SIMD3<Float> = -hairAxis
    return simd_normalize(SIMD2<Float>(simd_dot(into, insetRight), simd_dot(into, insetUp)))
}

/// Every pose depends only on how far along the path that glucose is (and,
/// for a small thermal wobble, on the tap phase) — so after one tap each
/// stands exactly where the one ahead of it stood. Step 35's, with its
/// inline arithmetic split into typed steps.
func moleculePoses(_ t: Float, mutant: Mutant, direction d: SIMD2<Float>) -> [MoleculePose] {
    let u: Float = sugarProgress(t, mutant: mutant)
    let frac: Float = u - u.rounded(.down)
    let phase: Float = tapPhase(t, mutant: mutant).phase
    let along = SIMD3<Float>(d.x, d.y, 0)
    let across = SIMD3<Float>(-d.y, d.x, 0)
    let wobbleAngle: Float = 0.10 * sin(2 * Float.pi * phase)
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
/// drift steadily, touching or not (the air does not care about the tap).
/// The rewind mutant runs it back — the odour would leave the pores.
func odourProgress(_ t: Float, mutant: Mutant) -> Float {
    effectiveTime(t, mutant: mutant) / tapSeconds
}

/// The odorant in its inset turns once a loop about a fixed axis — rigidly,
/// forward. MODEL, schematic: a free molecule in air tumbles far faster.
let odorantSpinAxis: SIMD3<Float> = simd_normalize(SIMD3<Float>(0.3, 1, 0.2))
func odorantSpin(_ t: Float, mutant: Mutant) -> simd_quatf {
    let a: Float = 2 * Float.pi * effectiveTime(t, mutant: mutant) / loopSeconds
    return simd_quatf(angle: a, axis: odorantSpinAxis)
}

/// The GIF: frames and delay, chosen by measuring the file (see main.swift).
let defaultFrames: Int = 100
let defaultDelayCentiseconds: Int = 10

/// Frame time for frame `f` of `n`.
func frameTime(_ f: Int, of n: Int) -> Float { loopSeconds * Float(f) / Float(n) }

/// The left antenna's small drift: a slow loop in the vertical plane, one
/// turn per tap, so it closes with the loop.
func swayAt(_ t: Float, mutant: Mutant) -> SIMD3<Float> {
    let a: Float = 2 * Float.pi * effectiveTime(t, mutant: mutant) / tapSeconds
    return SIMD3<Float>(0.4 * cos(a), sin(a), 0) * swayAmplitude
}

/// The inset camera rides on the antenna tip. So as the tip lifts, the
/// juice moves away from the hairs: a point `q` in the inset (µm, attached to
/// the antenna as it was at contact) lies over juice point `R q + T`. R is
/// the apical segment's turn since contact, T the rest. Step 30's, with one
/// change: step 30's inset axes were the world's (its sugar face was level);
/// here the inset's y is the juice's normal at the contact, so R and T are
/// turned into the inset's axes by the rotation `insetBasis` that takes the
/// normal to +y (the shortest one; the inset's other two axes are MODEL, as
/// the hairs' layout in it always was).
struct CrystalFrame {
    var rotation: simd_float3x3
    var translation: SIMD3<Float>   // µm

    func toCrystal(_ q: SIMD3<Float>) -> SIMD3<Float> { rotation * q + translation }
}

func insetBasis(normal n: SIMD3<Float>) -> simd_float3x3 {
    let up = SIMD3<Float>(0, 1, 0)
    return simd_distance(n, up) < 1e-7 ? matrix_identity_float3x3 : simd_float3x3(simd_quatf(from: simd_normalize(n), to: up))
}

func crystalFrame(contact: Antenna, now: Antenna, contactPoint p: SIMD3<Float>, normal n: SIMD3<Float>) -> CrystalFrame {
    let s0: Shape = contact.segments[contact.segments.count - 1]
    let s1: Shape = now.segments[now.segments.count - 1]
    let d0: SIMD3<Float> = simd_normalize(s0.b - s0.a)
    let d1: SIMD3<Float> = simd_normalize(s1.b - s1.a)
    let r: simd_float3x3 = simd_distance(d0, d1) < 1e-7 ? matrix_identity_float3x3 : simd_float3x3(simd_quatf(from: d0, to: d1))
    let c0: SIMD3<Float> = contact.tipCentre
    let c1: SIMD3<Float> = now.tipCentre
    let turned: SIMD3<Float> = r * (p - c0)
    let tMM: SIMD3<Float> = c1 + turned - p
    let m: simd_float3x3 = insetBasis(normal: n)
    let rInset: simd_float3x3 = m * r * m.transpose
    let tInset: SIMD3<Float> = m * tMM
    return CrystalFrame(rotation: rInset, translation: tInset * 1000)
}
