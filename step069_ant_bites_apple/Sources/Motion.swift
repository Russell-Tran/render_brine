// Step 69's clock: step 68's one clock, slowed ×10, with a bite on it.
//
//   * The tap: step 30's, as step 68 ran it — 4 antennal strokes a second
//     (Lenoir 1982, via O'Fallon, Suarez & Smith 2016, *Insectes Sociaux* 63:
//     "3–6 strokes/s"), each shown over 2.5 s: slowed ×10. The tips touch the
//     juice for 40% of each tap and lift 0.15 mm (step 68's MODEL numbers).
//   * The bite: MODEL. No measured rate of an ant biting soft fruit was
//     found; two bites a second is used, so each bite takes two taps on
//     screen, 5 s. Of each bite the jaws spend 30% opening, 15% open, 25%
//     closing, and 30% gripping — resting against the flesh, closed exactly as
//     far as the flesh lets them (Pose.swift solves where that is).
//   * The inset's juice and bits of flesh: MODEL, schematic — the order of
//     events and where each goes, not their speed or size.
//
// The loop is FORWARD: two whole bites and four whole taps; the juice and
// the bits move on and are replaced by the next, one set per bite, so the
// last frame runs on into the first. Browning, on a clock of minutes, is not
// drawn (Food.swift).

import Foundation
import simd

/// Real antennal strokes per second (Lenoir 1982, via O'Fallon et al. 2016:
/// 3–6 strokes/s); 4 is inside that range. Step 30's.
let realStrokesPerSecond: Float = 4
let realStrokeRange: ClosedRange<Float> = 3...6

/// Seconds per tap on screen. Step 30's.
let tapSeconds: Float = 2.5
/// How much slower than life, for everything.
let slowdown: Float = tapSeconds * realStrokesPerSecond
/// Taps per bite, and bites per loop.
let tapsPerBite: Int = 2
let bitesPerLoop: Int = 2
let biteSeconds: Float = tapSeconds * Float(tapsPerBite)
let loopSeconds: Float = biteSeconds * Float(bitesPerLoop)
/// Real bites per second: MODEL (see above).
var realBitesPerSecond: Float { slowdown / biteSeconds }

/// Share of each tap with the tip resting on the juice. MODEL, step 30's.
let contactFraction: Float = 0.40
/// How far the antenna tips move off the juice, mm. MODEL, step 68's.
let liftHeight: Float = 0.15

/// The bite's parts, as fractions of one bite. MODEL.
let biteOpening: Float = 0.30
let biteOpen: Float = 0.15
let biteClosing: Float = 0.25
/// The widest gape, each mandible turned out this far from its closed rest.
/// MODEL: wide enough to straddle the edge with room to spare.
let gapeMax: Float = 75 * Float.pi / 180

/// Where a time falls in the loop. The rewind mutant plays the first half
/// forward and then backward — continuous at the seam, but a rewind.
func effectiveTime(_ t: Float, mutant: Mutant) -> Float {
    let w: Float = t.truncatingRemainder(dividingBy: loopSeconds)
    let u: Float = w < 0 ? w + loopSeconds : w
    if mutant == .rewind { return u < loopSeconds / 2 ? 2 * u : 2 * (loopSeconds - u) }
    return u
}

func smoothstep01(_ x: Float) -> Float {
    let c: Float = min(max(x, 0), 1)
    let rise: Float = 3 - 2 * c
    return c * c * rise
}

/// Tip height off the juice at a tap phase: zero while touching, then up and
/// back on a sin² curve. Step 30's.
func liftAt(phase: Float, height: Float) -> Float {
    var p: Float = phase - phase.rounded(.down)
    if p < 0 { p += 1 }
    if p < contactFraction { return 0 }
    let x: Float = (p - contactFraction) / (1 - contactFraction)
    let s: Float = sin(Float.pi * x)
    return height * s * s
}

/// The tap's phase at time t: 0 at a touch-down.
func tapPhase(_ t: Float, mutant: Mutant) -> Float { effectiveTime(t, mutant: mutant) / tapSeconds }

/// The bite's phase at time t, 0 to 1, and which bite: 0 is the moment the
/// jaws, gripping, begin to open.
func bitePhase(_ t: Float, mutant: Mutant) -> (phase: Float, bite: Int) {
    let e: Float = effectiveTime(t, mutant: mutant) / biteSeconds
    let k: Float = e.rounded(.down)
    return (e - k, Int(k))
}

/// How open the jaws are at a bite phase: 0 at the grip (closed on the
/// flesh), 1 at the widest gape.
func gapeAt(phase x: Float) -> Float {
    if x < biteOpening { return smoothstep01(x / biteOpening) }
    let heldUntil: Float = biteOpening + biteOpen
    if x < heldUntil { return 1 }
    let closedAt: Float = heldUntil + biteClosing
    if x < closedAt { return 1 - smoothstep01((x - heldUntil) / biteClosing) }
    return 0
}

/// The GIF: 10 cs a frame, 100 frames a loop.
let defaultDelayCentiseconds: Int = 10
func frameTime(_ f: Int, of n: Int) -> Float { loopSeconds * Float(f) / Float(n) }
