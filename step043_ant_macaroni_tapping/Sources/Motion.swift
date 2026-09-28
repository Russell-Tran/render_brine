// Step 43: step 42's still, slightly and slowly animated, as step 30 animated
// step 26's. The right antenna taps the cheese sauce on the macaroni, again
// and again; everything else holds still, apart from a much smaller sway of
// the left antenna, and the odour, which keeps arriving.
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
//     the tip moves 0.07 mm off the sauce, about three-quarters of its own
//     0.096 mm thickness, and rests on it for 40% of each tap. Here "off" is
//     along the sauce's outward normal at the contact, which on the tube's
//     lower flank points back towards the ant and a little down: the tip
//     draws back from an overhanging wall, as it would.
//   * The odour: MODEL, schematic. Butanoic acid diffuses in air at
//     59 ± 8 Torr cm² s⁻¹ at 298 K (Tang et al. 2015, *Atmos Chem Phys* 15:
//     5585, Table 3's preferred value, from Lugg 1968; checked there) — 0.078
//     cm²/s at one atmosphere — so a molecule crosses the 3.6 µm drawn here in
//     about (3.6 µm)² / 2D ≈ 0.8 µs. No slow-down that keeps the tap
//     watchable could show that. The inset shows the order of events instead:
//     one molecule reaches a wall pore of the smell hair each tap and goes in,
//     and while it does the next ones are on their way.
//
// The loop is FORWARD: four whole taps; each odour molecule makes its trip
// once per loop, the four a tap apart, so the last frame runs straight on
// into the first. Never a rewind. The taste inset (norbixin and the salt in
// the sauce) is the still's and does not move; the odorant in its own inset
// turns slowly, rigidly, once a loop.

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

/// Share of each tap with the tip resting on the sauce. MODEL, step 30's.
let contactFraction: Float = 0.40
/// How far the tip moves off the sauce, mm. MODEL: slight — ¾ of the tip's own diameter.
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

/// Tip height off the sauce at a phase: zero while touching, then up and
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

/// How many taps' worth of odour has arrived: time in taps, so the dots
/// drift steadily, touching or not (the air does not care about the tap).
/// The rewind mutant runs it back — the odour would leave the pores.
func odourProgress(_ t: Float, mutant: Mutant) -> Float {
    effectiveTime(t, mutant: mutant) / tapSeconds
}

/// Butanoic acid in air, Tang et al. 2015 (above): 59 Torr cm²/s at 298 K.
let butanoicDiffusionTorrCm2PerS: Float = 59
/// At one atmosphere, m²/s.
var butanoicDiffusion: Float { butanoicDiffusionTorrCm2PerS / 760 * 1e-4 }

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
/// sauce moves away from the hairs: a point `q` in the inset (µm, attached to
/// the antenna as it was at contact) lies over sauce point `R q + T`. R is
/// the apical segment's turn since contact, T the rest. Step 30's, with one
/// change: step 30's inset axes were the world's (its sugar face was level);
/// here the inset's y is the sauce's normal at the contact, so R and T are
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
