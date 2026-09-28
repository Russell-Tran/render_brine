// Step 39: step 38's still, slightly and slowly animated, as step 30 animated
// step 26's. The right antenna taps the water drop, again and again;
// everything else holds still, apart from a much smaller sway of the left
// antenna.
//
// The tap is step 30's, unchanged — its rate, slow-down, lift and timing:
//
//   * Rate. Ants' antennal movements run at 3–6 strokes per second: O'Fallon,
//     Suarez & Smith 2016, *Insectes Sociaux* 63, write "(3–6 strokes/s;
//     Lenoir 1982)", citing Lenoir, *Behav Process* 7: 27–35, on antennal
//     communication during trophallaxis in Myrmica rubra, and set it against
//     the 19.5–41.5 strikes/s "rapid antennation" of trap-jaw ants (checked
//     in the paper itself, 2026-09-27, for step 33). No measured rate for an
//     ant tapping water turned up, so that rate is used: 4 per second, a real tap every
//     quarter second. The render plays each tap over 2.5 s: slowed ×10, and
//     labelled so.
//   * Amplitude and the share of each tap spent touching: MODEL, chosen to be
//     "slight" — the tip rises 0.07 mm, about three-quarters of its own
//     0.096 mm thickness, and rests on the water for 40% of each tap.
//   * The touch: step 38's. The tip is water-repellent (Food.swift), so it
//     presses a shallow dimple into the surface and is never wetted. As the
//     tip rises the dimple rises with it — the water stays in contact with
//     the apex, none of it above — until the surface is level again, and
//     then the tip leaves it (dimpleNow, below). Nothing adheres: the tip
//     takes no water away. MODEL, the simplest honest reading of step 38's
//     dimple; the tests check the water never covers any point of the hair.
//   * Humidity: step 38's coelocapitular sensillum, unchanged. It sits on the
//     same antennal segment, so it rides up and down with the tip; the air a
//     tenth of a millimetre over a drop is still humid air. No odour, ever.
//   * The molecule inset holds still: five hydrogen-bonded waters, as in
//     step 38. Their bonds break and re-form within picoseconds, which no
//     slow-down that keeps the tap watchable could show.
//
// The loop is FORWARD: four whole taps, each the same, so the last frame
// runs straight on into the first. Nothing drifts. Never a rewind.

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

/// Share of each tap with the tip resting on the water. MODEL, step 30's.
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

/// Tip height above the water at a phase: zero while touching, then up and
/// back down on a sin² curve, so it leaves and lands at zero speed.
func liftAt(phase: Float) -> Float {
    if phase < contactFraction { return 0 }
    let x: Float = (phase - contactFraction) / (1 - contactFraction)
    let s: Float = sin(Float.pi * x)
    return liftHeight * s * s
}

/// Taps since the loop began, with the fraction of the current one: the
/// loop's clock as the picture plays it. Forward, it only rises.
func loopProgress(_ t: Float, mutant: Mutant) -> Float {
    let (phase, tap) = tapPhase(t, mutant: mutant)
    return Float(tap) + phase
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
/// water moves away from the hairs: a point `q` in the inset (µm, attached to
/// the antenna as it was at contact) lies over water point `R q + T`. R is
/// the apical segment's turn since contact, T the rest. Step 30's crystal
/// frame, unchanged but for its name.
struct WaterFrame {
    var rotation: simd_float3x3
    var translation: SIMD3<Float>   // µm

    func toWater(_ q: SIMD3<Float>) -> SIMD3<Float> { rotation * q + translation }
}

func waterFrame(contact: Antenna, now: Antenna, contactPoint p: SIMD3<Float>) -> WaterFrame {
    let s0: Shape = contact.segments[contact.segments.count - 1]
    let s1: Shape = now.segments[now.segments.count - 1]
    let d0: SIMD3<Float> = simd_normalize(s0.b - s0.a)
    let d1: SIMD3<Float> = simd_normalize(s1.b - s1.a)
    let r: simd_float3x3 = simd_distance(d0, d1) < 1e-7 ? matrix_identity_float3x3 : simd_float3x3(simd_quatf(from: d0, to: d1))
    let c0: SIMD3<Float> = contact.tipCentre
    let c1: SIMD3<Float> = now.tipCentre
    let turned: SIMD3<Float> = r * (p - c0)
    let tMM: SIMD3<Float> = c1 + turned - p
    return WaterFrame(rotation: r, translation: tMM * 1000)
}

/// The water surface in the inset's water frame, before the kernel's
/// Lipschitz scaling: the drop's sphere, minus a dimple `depth` deep under
/// the hair, falling off with distance from it (step 38's shape). Negative
/// is in the water. The kernel's waterSDF is this over (1 + D/W).
func waterSurface(_ w: SIMD3<Float>, hairDistance dHair: Float, depth: Float) -> Float {
    let R: Float = dropSphereRadius * 1000
    let sphere: Float = simd_length(w - SIMD3<Float>(0, -R, 0)) - R
    let fall: Float = exp(-max(dHair, 0) / dimpleWidth)
    return sphere + depth * fall
}

/// How deep the dimple is now, µm. Step 38's depth at touch-down; as the tip
/// rises, just as deep as the apex still is below the level surface, so the
/// water keeps touching the apex and nothing more; zero once the apex is
/// above the level. Never more than step 38's (the kernel's bound needs that).
/// At the apex the surface is then sphere + depth = 0 exactly, and everywhere
/// else on the hair — further from the drop than the apex — it is positive.
func dimpleNow(_ wf: WaterFrame, taste h: Sensillum) -> Float {
    let R: Float = dropSphereRadius * 1000
    let tip: SIMD3<Float> = wf.toWater(h.tip)
    let below: Float = simd_length(tip - SIMD3<Float>(0, -R, 0)) - R - h.tipRadius
    return min(max(-below, 0), dimpleDepth)
}
