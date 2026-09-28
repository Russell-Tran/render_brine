// Step 35: step 34's still, slightly and slowly animated, as step 30 animated
// step 26's. The right antenna taps the honey, again and again; everything
// else holds still, apart from a much smaller sway of the left antenna.
//
// The tap is step 30's, unchanged — its rate, slow-down, lift and timing:
//
//   * Rate. Ants' antennal movements run at 3–6 strokes per second: O'Fallon,
//     Suarez & Smith 2016, *Insectes Sociaux* 63, write "(3–6 strokes/s;
//     Lenoir 1982)", citing Lenoir, *Behav Process* 7: 27–35, on antennal
//     communication during trophallaxis in Myrmica rubra, and set it against
//     the 19.5–41.5 strikes/s "rapid antennation" of trap-jaw ants (checked
//     in the paper itself, 2026-09-27, for step 33). No measured rate for an
//     ant tapping honey turned up, so that rate is used: 4 per second, a real
//     tap every quarter second. The render plays each tap over 2.5 s: slowed
//     ×10, and labelled so.
//   * Amplitude and the share of each tap spent touching: MODEL, chosen to be
//     "slight" — the tip rises 0.07 mm, about three-quarters of its own
//     0.096 mm thickness, and rests on the honey for 40% of each tap.
//   * The sugars: MODEL, schematic, as step 30's sucrose. In water, small
//     molecules cross a few nanometres in nanoseconds (step 33's check: ions
//     at about 10⁻⁹ m²/s). Honey is more viscous than water, which slows
//     that — by how much is not checked here, so the inset makes no claim
//     about speed. It shows the direction of travel — out of the honey and
//     into the pore, one sugar per touch — and only while the hair is in the
//     honey, because only then is there a liquid path.
//   * The odour: MODEL, schematic. Dots drift in to the smell hair's wall
//     pores at a pace chosen to be seen — a real odour molecule's path through
//     air is a random walk, and far quicker. They keep coming whether the
//     antenna is up or down: the air is everywhere; only the taste hair needs
//     to touch.
//   * The honey's surface stays rigid. A liquid surface dimples under a load,
//     and how deep depends on that load and on honey's surface tension. The
//     tip here only touches — contact from geometry, no push — so there is no
//     load to model, and step 34 found no checked surface tension for honey
//     (its value is an UNVERIFIED lower bound, used only for the Bond
//     number). With nothing to bound it by, no dimple is drawn; the
//     meniscus in the micrometre inset is the only deformation, as in step 34.
//     Nor is a thread of honey drawn out as the hair lifts: its apex only
//     just met the surface, and how a bridge that small would stretch and
//     break is not something measured here.
//
// The loop is FORWARD: four whole taps; each touch takes one sugar a place
// further in, and each tap brings each odour lane a new dot, so the last frame
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

/// Share of each tap with the tip resting on the honey. MODEL, step 30's.
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

/// Tip height above the honey at a phase: zero while touching, then up and
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

/// How many places the sugars have advanced: one per tap, moving only
/// while the hair is in the honey. Step 30's moleculeProgress.
func sugarProgress(_ t: Float, mutant: Mutant) -> Float {
    let (phase, tap) = tapPhase(t, mutant: mutant)
    return Float(tap) + smoothstep01(phase / contactFraction)
}

/// How far the odour has come, in taps: steadily, touching or not.
func odourProgress(_ t: Float, mutant: Mutant) -> Float {
    effectiveTime(t, mutant: mutant) / tapSeconds
}

/// Sugar spacing along the drift path, Å — one sugar per tap. MODEL: wide
/// enough that neighbours' waters never meet (step 30's 13 Å was for sucrose
/// alone).
let moleculeSpacing: Float = 14.0
/// Sugars drawn at once: between touches one sits in the middle of the
/// 24 Å field with its neighbours cut by the edge; the fourth is the one
/// arriving next. An even number, so the inset always holds as many fructose
/// as glucose.
let moleculeSlots: Int = 4

/// One sugar (and its waters) in the molecule inset: its rotation and where
/// its centroid is. Step 30's.
struct MoleculePose {
    var rotation: simd_quatf
    var offset: SIMD3<Float>

    func place(_ p: SIMD3<Float>) -> SIMD3<Float> { rotation.act(p) + offset }
}

/// The drift path, in the molecule inset's screen plane (x right, y up):
/// the direction that points up the taste hair, into its pore, as the
/// micrometre inset shows it. `insetRight` and `insetUp` are that camera's
/// axes. Step 30's.
func driftDirection(hairAxis: SIMD3<Float>, insetRight: SIMD3<Float>, insetUp: SIMD3<Float>) -> SIMD2<Float> {
    let into: SIMD3<Float> = -hairAxis
    return simd_normalize(SIMD2<Float>(simd_dot(into, insetRight), simd_dot(into, insetUp)))
}

/// Every pose depends only on how far along the path that sugar is (and,
/// for a small thermal wobble, on the tap phase) — so after one tap each
/// sugar stands exactly where the one ahead of it stood. Step 30's, with
/// this step's spacing.
func moleculePoses(_ t: Float, mutant: Mutant, direction d: SIMD2<Float>) -> [MoleculePose] {
    let u: Float = sugarProgress(t, mutant: mutant)
    let frac: Float = u - u.rounded(.down)
    let phase: Float = tapPhase(t, mutant: mutant).phase
    let along = SIMD3<Float>(d.x, d.y, 0)
    let across = SIMD3<Float>(-d.y, d.x, 0)
    let wobble = simd_quatf(angle: 0.10 * sin(2 * Float.pi * phase), axis: simd_normalize(SIMD3<Float>(1, 0.4, -0.3)))
    var out: [MoleculePose] = []
    for k in 0..<moleculeSlots {
        let s: Float = (Float(k) - 2 + frac) * moleculeSpacing
        let tumble = simd_quatf(angle: 0.16 * s, axis: simd_normalize(SIMD3<Float>(0.5, -0.2, 1)))
        let side: Float = 1.6 * sin(s * 2 * Float.pi / 29)
        let depth: Float = -2.5 * cos(s * 2 * Float.pi / 37)
        let shift: SIMD3<Float> = along * s + across * side
        out.append(MoleculePose(rotation: wobble * tumble, offset: shift + SIMD3<Float>(0, 0, depth)))
    }
    return out
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
/// honey and its meniscus move away from the hairs (the struct keeps step
/// 30's name): a point `q` in the inset
/// (µm, attached to the antenna as it was at contact) lies over honey point
/// `R q + T`. R is the apical segment's turn since contact, T the rest.
/// Step 30's, unchanged.
struct CrystalFrame {
    var rotation: simd_float3x3
    var translation: SIMD3<Float>   // µm

    func toCrystal(_ q: SIMD3<Float>) -> SIMD3<Float> { rotation * q + translation }
}

/// Step 35: the honey's surface normal at the contact is tilted (Food.swift),
/// but the inset's y is that normal, so the main view's turn and shift are
/// first turned into the inset's axes — by the least turn that takes the
/// normal to +y. On salt, whose face is level, that turn is none, and this is
/// step 30's exactly.
func crystalFrame(contact: Antenna, now: Antenna, contactPoint p: SIMD3<Float>,
                  normal n: SIMD3<Float> = SIMD3<Float>(0, 1, 0)) -> CrystalFrame {
    let s0: Shape = contact.segments[contact.segments.count - 1]
    let s1: Shape = now.segments[now.segments.count - 1]
    let d0: SIMD3<Float> = simd_normalize(s0.b - s0.a)
    let d1: SIMD3<Float> = simd_normalize(s1.b - s1.a)
    let r: simd_float3x3 = simd_distance(d0, d1) < 1e-7 ? matrix_identity_float3x3 : simd_float3x3(simd_quatf(from: d0, to: d1))
    let c0: SIMD3<Float> = contact.tipCentre
    let c1: SIMD3<Float> = now.tipCentre
    let turned: SIMD3<Float> = r * (p - c0)
    let tMM: SIMD3<Float> = c1 + turned - p
    let up = SIMD3<Float>(0, 1, 0)
    let unit: SIMD3<Float> = simd_normalize(n)
    let b: simd_float3x3 = simd_distance(unit, up) < 1e-7 ? matrix_identity_float3x3 : simd_float3x3(simd_quatf(from: unit, to: up))
    let inInset: simd_float3x3 = b * r * b.transpose
    return CrystalFrame(rotation: inInset, translation: (b * tMM) * 1000)
}
