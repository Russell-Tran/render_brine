// Step 37: step 36's still, slightly and slowly animated, as step 30 animated
// step 26's. The right antenna taps the crumb of yolk, again and again; the
// hexanal keeps coming to the smell hair; everything else holds still, apart
// from a much smaller sway of the left antenna.
//
// The tap is step 30's, unchanged — its rate, slow-down, lift and timing:
//
//   * Rate. Ants' antennal movements run at 3–6 strokes per second: O'Fallon,
//     Suarez & Smith 2016, *Insectes Sociaux* 63, write "(3–6 strokes/s;
//     Lenoir 1982)", citing Lenoir, *Behav Process* 7: 27–35, on antennal
//     communication during trophallaxis in Myrmica rubra, and set it against
//     the 19.5–41.5 strikes/s "rapid antennation" of trap-jaw ants (checked
//     in the paper itself, 2026-09-27). No measured rate for an ant tapping
//     yolk turned up, so that rate is used: 4 per second, a real tap every
//     quarter second. The render plays each tap over 2.5 s: slowed ×10, and
//     labelled so.
//   * Amplitude and the share of each tap spent touching: MODEL, chosen to be
//     "slight" — the tip rises 0.07 mm, about three-quarters of its own
//     0.096 mm thickness, and rests on the yolk for 40% of each tap. The crumb
//     is not a flat face, so the tip rises along the yolk's own normal at the
//     contact, the direction step 36 set it down along.
//   * The odour: MODEL, schematic. Hexanal's own diffusivity in air was not
//     found; n-hexane, the same six-carbon chain without the C=O, diffuses in
//     air at (59 ± 3) Torr cm²/s at 298 K — 0.078 cm²/s at one atmosphere
//     (Tang et al., *Atmos Chem Phys* 15: 5585, 2015, supplement §1.4, the
//     preferred value; checked there). So a molecule crosses the inset's
//     8 µm in about (8 µm)² / 2D ≈ 4 µs: no slow-down that keeps the tap
//     watchable could show that. The inset instead shows the order of events
//     — a molecule drifts in, settles at a wall pore and goes in — one
//     molecule per tap, and says it is schematic.
//
// The loop is FORWARD: four whole taps, and four molecules on the way at any
// moment, one a tap apart; after each tap every molecule stands where the one
// ahead of it stood, so the last frame runs straight on into the first.
// Never a rewind.

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

/// Share of each tap with the tip resting on the yolk. MODEL, step 30's.
let contactFraction: Float = 0.40
/// How far the tip rises, mm. MODEL: slight — ¾ of the tip's own diameter.
let liftHeight: Float = 0.07
/// The left antenna's sway, mm: a much smaller, slower-looking drift. MODEL.
let swayAmplitude: Float = 0.018

/// n-hexane's diffusivity in air at 298 K and one atmosphere, cm²/s (Tang et
/// al. 2015: 59 Torr cm²/s ÷ 760 Torr), standing in for hexanal's.
let hexaneDiffusivity: Float = 59.0 / 760.0

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

/// Tip height above the yolk at a phase: zero while touching, then up and
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

/// How many molecules have gone in since the loop began: one a tap, steadily
/// — the air does not wait for the antenna. Its fraction is how far along
/// the current tap is.
func odourProgress(_ t: Float, mutant: Mutant) -> Float {
    effectiveTime(t, mutant: mutant) / tapSeconds
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

// MARK: - the inset rides on the tip

/// The inset camera rides on the antenna tip. So as the tip lifts, the yolk
/// and its film move away from the hairs: a point `q` in the inset (µm,
/// attached to the antenna as it was at contact) lies over yolk point
/// `R q + T`. R is the apical segment's turn since contact, T the rest.
/// Step 30's, with one change: step 36's inset has y along the yolk's normal
/// at the contact (its surface is y = 0 there), and that normal is not the
/// world's up — so R and T are carried into the inset's own axes.
/// (The kernel's name for the yolk surface is still step 32's `crystal`.)
struct CrystalFrame {
    var rotation: simd_float3x3
    var translation: SIMD3<Float>   // µm

    func toCrystal(_ q: SIMD3<Float>) -> SIMD3<Float> { rotation * q + translation }
}

/// The inset's axes in the world: y is the contact normal, x and z the
/// world's, turned by the least rotation that takes up onto that normal.
func insetAxes(normal n: SIMD3<Float>) -> simd_float3x3 {
    let up = SIMD3<Float>(0, 1, 0)
    if simd_distance(simd_normalize(n), up) < 1e-7 { return matrix_identity_float3x3 }
    return simd_float3x3(simd_quatf(from: up, to: simd_normalize(n)))
}

func crystalFrame(contact: Antenna, now: Antenna, contactPoint p: SIMD3<Float>, normal: SIMD3<Float>) -> CrystalFrame {
    let s0: Shape = contact.segments[contact.segments.count - 1]
    let s1: Shape = now.segments[now.segments.count - 1]
    let d0: SIMD3<Float> = simd_normalize(s0.b - s0.a)
    let d1: SIMD3<Float> = simd_normalize(s1.b - s1.a)
    let r: simd_float3x3 = simd_distance(d0, d1) < 1e-7 ? matrix_identity_float3x3 : simd_float3x3(simd_quatf(from: d0, to: d1))
    let c0: SIMD3<Float> = contact.tipCentre
    let c1: SIMD3<Float> = now.tipCentre
    let turned: SIMD3<Float> = r * (p - c0)
    let tMM: SIMD3<Float> = c1 + turned - p
    let q: simd_float3x3 = insetAxes(normal: normal)
    let qt: simd_float3x3 = q.transpose
    let rInset: simd_float3x3 = qt * r * q
    let tInset: SIMD3<Float> = qt * (tMM * 1000)
    return CrystalFrame(rotation: rInset, translation: tInset)
}

// MARK: - the hexanal, coming to the smell hair

// Each molecule lives four taps in the inset. It comes in from the lower
// left, from beyond the circle, drifting up past the front of the taste hair
// (MODEL path, chosen so it clears both hairs by over a micrometre); settles
// just outside one wall pore on the side facing the camera, as step 36 drew
// two of them; rests there a tap; and goes in, getting smaller as it sinks
// into the pore, until it is gone. Molecule n (one a tap) goes to pore
// n mod 4, so the four pores take one molecule each per loop.

/// Where every molecule enters, µm, inset coordinates: beyond the circle's
/// lower-left edge (it projects 8.9 µm from the centre; the circle is 7.25).
let odourEntry = SIMD3<Float>(-6.2, 2.0, 2.5)
/// How far out from the pore cluster the paths bend in, µm. MODEL.
let odourBend: Float = 1.4
/// The pores they go to: step 36's rows, in the camera-facing column or the
/// next one round.
let odourPores: [(row: Int, dj: Int)] = [(5, 0), (8, 0), (11, 1), (13, 1)]
/// Ages (in taps) where a molecule reaches its pore, and where it starts to go in.
let odourArrive: Float = 2.5
let odourEnter: Float = 3.5
let odourLife: Float = 4.0
/// Molecules drawn at once: one per tap of life.
let odourSlots: Int = 4
/// How far out from the pore mouth a molecule rests, µm (step 36's 0.13).
let odourRest: Float = 0.13
/// A slight wander off the straight path, µm, so the drift does not look
/// like a conveyor. MODEL.
let odourWander: Float = 0.3

/// The column whose pores look most nearly at the camera and down to the
/// food — step 36's choice.
func facingColumn(_ smell: Sensillum, facing view: SIMD3<Float>) -> Int {
    let want: SIMD3<Float> = simd_normalize(-view + SIMD3<Float>(-0.4, -0.5, 0))
    var bestColumn: Int = 0
    var bestDot: Float = -2
    for j in 0..<smell.poreColumns {
        let d: Float = simd_dot(smell.wallPore(row: 6, column: j).outward, want)
        if d > bestDot { bestDot = d; bestColumn = j }
    }
    return bestColumn
}

/// The pore molecule `target` goes to: lattice row, column, mouth and outward.
func odourPore(_ target: Int, smell: Sensillum, facing view: SIMD3<Float>)
    -> (row: Int, column: Int, mouth: SIMD3<Float>, outward: SIMD3<Float>) {
    let p = odourPores[target]
    let j: Int = (facingColumn(smell, facing: view) + p.dj + smell.poreColumns) % smell.poreColumns
    let (mouth, outward) = smell.wallPore(row: p.row, column: j)
    return (p.row, j, mouth, outward)
}

/// Where the paths bend in: out in front of the four pores.
func odourBendPoint(smell: Sensillum, facing view: SIMD3<Float>) -> SIMD3<Float> {
    var rest = SIMD3<Float>(0, 0, 0)
    var out = SIMD3<Float>(0, 0, 0)
    for k in 0..<odourPores.count {
        let p = odourPore(k, smell: smell, facing: view)
        rest += p.mouth + p.outward * odourRest
        out += p.outward
    }
    rest /= Float(odourPores.count)
    return rest + simd_normalize(out) * odourBend
}

/// One molecule at `age` taps, going to pore `target`: centre (µm) and the
/// radius its dot is drawn at (0 = not drawn).
func odourDot(age a: Float, target: Int, smell: Sensillum, facing view: SIMD3<Float>) -> (position: SIMD3<Float>, radius: Float) {
    let pore = odourPore(target, smell: smell, facing: view)
    let rest: SIMD3<Float> = pore.mouth + pore.outward * odourRest
    if a >= odourEnter {
        // Going in: from its resting place down into the pore, shrinking to nothing.
        let s: Float = smoothstep01((a - odourEnter) / (odourLife - odourEnter))
        let deep: SIMD3<Float> = pore.mouth - pore.outward * 0.05
        let sunk: SIMD3<Float> = (deep - rest) * s
        let left: Float = 1 - s
        return (rest + sunk, volatileDotRadius * left)
    }
    if a >= odourArrive { return (rest, volatileDotRadius) }
    // Drifting in: a quadratic curve from the entry, bent at the bend point,
    // easing to a stop at the pore; plus a small wander that is zero at both ends.
    // Nearly steady along the way (speed between 1 and 4/3 of the mean),
    // easing to a stop at the pore: s = x + x² − x³.
    let x: Float = a / odourArrive
    let x2: Float = x * x
    let s: Float = x + x2 - x2 * x
    let bend: SIMD3<Float> = odourBendPoint(smell: smell, facing: view)
    let w0: Float = (1 - s) * (1 - s)
    let w1: Float = 2 * s * (1 - s)
    let w2: Float = s * s
    let fromEntry: SIMD3<Float> = odourEntry * w0
    let fromBend: SIMD3<Float> = bend * w1
    let fromRest: SIMD3<Float> = rest * w2
    let curve: SIMD3<Float> = fromEntry + fromBend + fromRest
    let ph: Float = 1.7 * Float(target)
    let env: Float = sin(Float.pi * s)
    let side1 = SIMD3<Float>(0, 1, 0)
    let side2 = SIMD3<Float>(0, 0, 1)
    let angle: Float = 3 * Float.pi * s + ph
    let upDown: SIMD3<Float> = side1 * sin(angle)
    let nearFar: SIMD3<Float> = side2 * cos(angle)
    let across: SIMD3<Float> = upDown + nearFar
    let wanderSize: Float = odourWander * env
    let wander: SIMD3<Float> = across * wanderSize
    // Born beyond the circle at nothing, grown to full size before it can be seen.
    let grow: Float = smoothstep01(a / 0.15)
    return (curve + wander, volatileDotRadius * grow)
}

/// The odour at `progress` taps into the loop: slot k holds the molecule that
/// entered k taps before the current one; molecule n goes to pore n mod 4.
/// The volatiles mutant takes them all away.
func buildVolatiles(hairs: [Sensillum], progress u: Float, mutant: Mutant, facing view: SIMD3<Float>) -> [Volatile] {
    guard foodHasVapour && mutant != .volatiles else { return [] }
    let smell: Sensillum = hairs[1]
    let n: Float = u.rounded(.down)
    let f: Float = u - n
    let count: Int = odourPores.count
    return (0..<odourSlots).map { k -> Volatile in
        let id: Int = Int(n) - k
        let target: Int = ((id % count) + count) % count
        let age: Float = f + Float(k)
        let dot = odourDot(age: age, target: target, smell: smell, facing: view)
        let pore = odourPore(target, smell: smell, facing: view)
        return Volatile(position: dot.position, pore: (pore.row, pore.column), radius: dot.radius, age: age)
    }
}
