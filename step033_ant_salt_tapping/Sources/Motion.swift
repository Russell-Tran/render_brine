// Step 33: step 32's still, slightly and slowly animated, as step 30 animated
// step 26's. The right antenna taps the salt, again and again; everything
// else holds still, apart from a much smaller sway of the left antenna.
//
// The tap is step 30's, unchanged — its rate, slow-down, lift and timing:
//
//   * Rate. Ants' antennal movements run at 3–6 strokes per second: O'Fallon,
//     Suarez & Smith 2016, *Insectes Sociaux* 63, write "(3–6 strokes/s;
//     Lenoir 1982)", citing Lenoir, *Behav Process* 7: 27–35, on antennal
//     communication during trophallaxis in Myrmica rubra, and set it against
//     the 19.5–41.5 strikes/s "rapid antennation" of trap-jaw ants (checked
//     in the paper itself, 2026-09-27). No measured rate for an ant tapping
//     salt turned up, so that rate is used: 4 per second, a real tap every
//     quarter second. The render plays each tap over 2.5 s: slowed ×10, and
//     labelled so.
//   * Amplitude and the share of each tap spent touching: MODEL, chosen to be
//     "slight" — the tip rises 0.07 mm, about three-quarters of its own
//     0.096 mm thickness, and rests on the salt for 40% of each tap.
//   * The ions: MODEL, schematic. Small ions in water diffuse at about
//     10⁻⁹ m²/s — infinitely dilute KCl at 1.77 × 10⁻⁹ m²/s at 20 °C
//     (Harned & Nuttall, as quoted by Wang et al. 2014, *Biomicrofluidics*
//     8: 024118; checked there, for KCl, not NaCl) — so an ion crosses the
//     molecule inset's 3 nm in about (3 nm)² / 2D ≈ 3 ns; no slow-down that
//     keeps the tap watchable could show that. The inset instead shows the order of events — a pair leaves the
//     kink, takes on water, and rises through the film, one pair per touch —
//     and only while the hair is in the film, as step 30 moved its sugar.
//
// The loop is FORWARD: four whole taps; each touch takes one pair off the
// kink, and the kink (and the view following it) moves one cell edge along
// the step, so the last frame runs straight on into the first. Never a rewind.

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

/// Share of each tap with the tip resting on the salt. MODEL, step 30's.
let contactFraction: Float = 0.40
/// How far the tip rises, mm. MODEL: slight — ¾ of the tip's own diameter.
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

/// Tip height above the salt at a phase: zero while touching, then up and
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

/// How many pairs have left the kink: one per tap, each leaving only while
/// the hair is in the film. Its fraction is how far the current one has got.
func ionProgress(_ t: Float, mutant: Mutant) -> Float {
    let (phase, tap) = tapPhase(t, mutant: mutant)
    return Float(tap) + smoothstep01(phase / contactFraction)
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
/// crystal and its film move away from the hairs: a point `q` in the inset
/// (µm, attached to the antenna as it was at contact) lies over crystal point
/// `R q + T`. R is the apical segment's turn since contact, T the rest.
/// Step 30's, unchanged.
struct CrystalFrame {
    var rotation: simd_float3x3
    var translation: SIMD3<Float>   // µm

    func toCrystal(_ q: SIMD3<Float>) -> SIMD3<Float> { rotation * q + translation }
}

func crystalFrame(contact: Antenna, now: Antenna, contactPoint p: SIMD3<Float>) -> CrystalFrame {
    let s0: Shape = contact.segments[contact.segments.count - 1]
    let s1: Shape = now.segments[now.segments.count - 1]
    let d0: SIMD3<Float> = simd_normalize(s0.b - s0.a)
    let d1: SIMD3<Float> = simd_normalize(s1.b - s1.a)
    let r: simd_float3x3 = simd_distance(d0, d1) < 1e-7 ? matrix_identity_float3x3 : simd_float3x3(simd_quatf(from: d0, to: d1))
    let c0: SIMD3<Float> = contact.tipCentre
    let c1: SIMD3<Float> = now.tipCentre
    let turned: SIMD3<Float> = r * (p - c0)
    let tMM: SIMD3<Float> = c1 + turned - p
    return CrystalFrame(rotation: r, translation: tMM * 1000)
}
