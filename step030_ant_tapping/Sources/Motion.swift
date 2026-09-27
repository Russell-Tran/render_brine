// Step 30: step 26's still, slightly and slowly animated. The right antenna
// taps the sugar grain, again and again; everything else holds still, apart
// from a much smaller sway of the left antenna.
//
// What the literature gives, and what is MODEL:
//
//   * Rate. Ants' ordinary antennal movements run at 3–6 strokes per second
//     (Lenoir 1982, *Behav Processes* 7: 27–35, Myrmica rubra, as summarised
//     by O'Fallon, Suarez & Smith 2016, *Insectes Sociaux* 63: 265–270, who
//     set them against the much faster 19.5–41.5 strikes/s "rapid
//     antennation" of trap-jaw ants; a honeybee's antennal strokes during food
//     transfer run at 13 per second, Goyret & Farina 2003). No measured rate
//     for an ant tapping a sugar grain turned up, so the ordinary ant rate is
//     used: 4 per second, a real tap every quarter second. The render plays
//     each tap over 2.5 s: slowed ×10, and labelled so.
//   * Amplitude and the share of each tap spent touching: MODEL, chosen to be
//     "slight" — the tip rises 0.07 mm, about three-quarters of its own
//     0.096 mm thickness (16 px in the 1920-wide frame), and rests on the
//     sugar for 40% of each tap.
//   * The molecules: MODEL, schematic. A real sucrose molecule in water
//     diffuses at about 5 × 10⁻¹⁰ m²/s, so it crosses the molecule inset's
//     2 nm in nanoseconds; no slow-down that keeps the tap watchable could
//     show that. The inset instead shows the direction of travel — out of the
//     film and into the pore, one molecule per touch — and only while the
//     hair is in the film, because only then is there a liquid path.
//
// The loop is FORWARD: four whole taps, and the molecules advance one place
// per tap, so the last frame runs straight on into the first. Never a rewind.

import Foundation
import simd

/// Real antennal strokes per second (Lenoir 1982, via O'Fallon et al. 2016:
/// 3–6 strokes/s); 4 is inside that range.
let realStrokesPerSecond: Float = 4
let realStrokeRange: ClosedRange<Float> = 3...6

/// Seconds per tap on screen. MODEL: slow enough to read.
let tapSeconds: Float = 2.5
/// How much slower than life: a real tap takes 1/4 s, this one 2.5 s.
let slowdown: Float = tapSeconds * realStrokesPerSecond
/// Taps in one loop.
let tapsPerLoop: Int = 4
let loopSeconds: Float = tapSeconds * Float(tapsPerLoop)

/// Share of each tap with the tip resting on the sugar. MODEL.
let contactFraction: Float = 0.40
/// How far the tip rises, mm. MODEL: slight — ¾ of the tip's own diameter.
let liftHeight: Float = 0.07
/// The left antenna's sway, mm: a much smaller, slower-looking drift. MODEL.
let swayAmplitude: Float = 0.018

/// Molecule spacing along the drift path, Å — one molecule per tap. MODEL.
let moleculeSpacing: Float = 13.0
/// Molecules drawn at once: between touches one sits in the middle of the
/// 21 Å field with its neighbours cut by the edge; the fourth is the one
/// arriving next.
let moleculeSlots: Int = 4

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

/// Tip height above the sugar at a phase: zero while touching, then up and
/// back down on a sin² curve, so it leaves and lands at zero speed.
func liftAt(phase: Float) -> Float {
    if phase < contactFraction { return 0 }
    let x: Float = (phase - contactFraction) / (1 - contactFraction)
    let s: Float = sin(Float.pi * x)
    return liftHeight * s * s
}

func smoothstep01(_ x: Float) -> Float {
    let c: Float = min(max(x, 0), 1)
    return c * c * (3 - 2 * c)
}

/// How many places the molecules have advanced: one per tap, moving only
/// while the hair is in the film.
func moleculeProgress(_ t: Float, mutant: Mutant) -> Float {
    let (phase, tap) = tapPhase(t, mutant: mutant)
    return Float(tap) + smoothstep01(phase / contactFraction)
}

/// The GIF: 100 frames at 10 cs is the 10 s loop at 10 frames a second —
/// 25 frames a tap. Chosen by measuring the file (see main.swift).
let defaultFrames: Int = 100
let defaultDelayCentiseconds: Int = 10

/// Frame time for frame `f` of `n`.
func frameTime(_ f: Int, of n: Int) -> Float { loopSeconds * Float(f) / Float(n) }

/// One molecule in the molecule inset: its rotation and where its centroid is.
struct MoleculePose {
    var rotation: simd_quatf
    var offset: SIMD3<Float>

    func place(_ p: SIMD3<Float>) -> SIMD3<Float> { rotation.act(p) + offset }
}

/// The drift path, in the molecule inset's screen plane (x right, y up):
/// the direction that points up the taste hair, into its pore, as the
/// micrometre inset shows it. `insetRight` and `insetUp` are that camera's axes.
func driftDirection(hairAxis: SIMD3<Float>, insetRight: SIMD3<Float>, insetUp: SIMD3<Float>) -> SIMD2<Float> {
    let into: SIMD3<Float> = -hairAxis
    return simd_normalize(SIMD2<Float>(simd_dot(into, insetRight), simd_dot(into, insetUp)))
}

/// Every pose depends only on how far along the path that molecule is (and,
/// for a small thermal wobble, on the tap phase) — so after one tap each
/// molecule stands exactly where the one ahead of it stood.
func moleculePoses(_ t: Float, mutant: Mutant, direction d: SIMD2<Float>) -> [MoleculePose] {
    let u: Float = moleculeProgress(t, mutant: mutant)
    let frac: Float = u - u.rounded(.down)
    let phase: Float = tapPhase(t, mutant: mutant).phase
    let along = SIMD3<Float>(d.x, d.y, 0)
    let across = SIMD3<Float>(-d.y, d.x, 0)
    let base = simd_quatf(angle: 0.9, axis: simd_normalize(SIMD3<Float>(0.3, 1, 0.2)))
    let wobble = simd_quatf(angle: 0.10 * sin(2 * Float.pi * phase), axis: simd_normalize(SIMD3<Float>(1, 0.4, -0.3)))
    var out: [MoleculePose] = []
    for k in 0..<moleculeSlots {
        let s: Float = (Float(k) - 2 + frac) * moleculeSpacing
        let tumble = simd_quatf(angle: 0.16 * s, axis: simd_normalize(SIMD3<Float>(0.5, -0.2, 1)))
        let side: Float = 1.6 * sin(s * 2 * Float.pi / 29)
        let depth: Float = -2.5 * cos(s * 2 * Float.pi / 37)
        out.append(MoleculePose(rotation: wobble * tumble * base,
                                offset: along * s + across * side + SIMD3<Float>(0, 0, depth)))
    }
    return out
}

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
    let tMM: SIMD3<Float> = c1 + r * (p - c0) - p
    return CrystalFrame(rotation: r, translation: tMM * 1000)
}
