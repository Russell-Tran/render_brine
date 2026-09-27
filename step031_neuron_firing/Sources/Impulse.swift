// One nerve impulse, as numbers: where it starts, how fast it really runs,
// how much the picture slows it, and when it reaches each point of the axon.
// Nothing here touches the GPU. The kernel in Render.swift mirrors `pulse`
// and `arrival` line for line, and a test holds the two together.
//
// The cell itself is step 27's, untouched (Neuron.swift is a byte-for-byte
// copy). Only the shading changes from frame to frame.

import Foundation
import simd

// MARK: - the real numbers

/// Where an action potential starts: the axon initial segment, the first
/// stretch of axon just past the hillock. Shown in this very cell type by
/// Coombs, Curtis & Eccles, *J Physiol* 1957 ("The interpretation of spike
/// potentials of motoneurones"): in cat spinal motoneurons the initial-segment
/// ("IS") spike fires first and the soma-dendritic ("SD") spike follows it.
/// The general case: Stuart, Spruston, Sakmann & Häusser, *Trends Neurosci*
/// 1997, and Kole & Stuart, *Neuron* 2012, on the initial segment's dense
/// sodium channels.
///
/// Here the impulse starts where the hillock's cone meets the trunk. MODEL,
/// the exact point: the initial segment runs some tens of µm, and in some
/// neurons the spike first fires near its far end rather than its start.
let initiationArcLength: Float = 0

/// Conduction velocity, m/s. Hursh (*Am J Physiol* 1939, "Conduction velocity
/// and diameter of nerve fibers", cat nerve) found a myelinated fibre conducts
/// at about 6 m/s per µm of its whole diameter, axon plus myelin. The axon's
/// share of that diameter, the g-ratio, is about 0.6 (Rushton, *J Physiol*
/// 1951, the optimum; measured values sit near it). So step 27's 7 µm axon is
/// a fibre of about 11.7 µm, conducting at 70 m/s — inside the 50–100 m/s
/// usually quoted for alpha motor axons. Derived, not typed.
let hurshFactor: Float = 6        // m/s per µm of fibre diameter
let gRatio: Float = 0.6
let fibreDiameter: Float = axonDiameter / gRatio
let conductionVelocity: Float = hurshFactor * fibreDiameter

/// How long one spike lasts at one point, s: about a millisecond, the
/// textbook figure (e.g. Kandel et al., *Principles of Neural Science*); cat
/// motoneuron spikes are of this order.
let spikeDuration: Float = 0.001

// HONESTY. Real motor axons are MYELINATED, and conduct by saltatory jumps
// from one node of Ranvier to the next. Step 27 chose to leave the myelin
// out, so here the glow slides continuously along a bare axon instead. The
// caption says so.
//
// And two things are drawn narrower than life. At 70 m/s a 1 ms spike is
// about 7 cm long along the axon — hundreds of times longer than this whole
// picture — so a true-width glow would light every visible micrometre at
// once. And at this slow-down each point would stay lit for minutes. The
// glow is shortened to a comet so the direction can be seen.
//
// The spike also spreads back from the initial segment into the soma and
// part way into the dendrites (the SD spike above). This render shows only
// the impulse going OUT, the one that reaches the muscle.

let realSpikeLength: Float = conductionVelocity * spikeDuration   // m: 0.07

// MARK: - the picture's clock

/// One loop of the animation, s. The impulse, then rest, then the next.
let loopSeconds: Float = 8

/// How fast the glow runs in the picture, µm per animation second. MODEL:
/// slow enough to follow, fast enough to finish with rest to spare.
let pictureSpeed: Float = 100

/// The slow-down, derived: 70 m/s against 100 µm/s.
let slowdown: Float = conductionVelocity * 1_000_000 / pictureSpeed   // 700,000

/// When the impulse leaves the initial segment, s into the loop. MODEL.
let impulseStart: Float = 0.5

/// Time the glow spends hidden in the break, s. MODEL, and a cut: the break
/// stands for about a metre of axon (step 27's `axonRealLength`), which takes
/// the real impulse ~14 ms — at this slow-down, nearly three hours.
let breakSeconds: Float = 0.8
let realBreakSeconds: Float = axonRealLength / 1_000_000 / conductionVelocity

/// The glow at one point, against time since the impulse arrived there, s.
/// A quick rise, an exponential fall, and a fade to exactly zero, so the rest
/// of the loop is perfectly still. MODEL — shaped like a spike, timed to be
/// seen.
let glowRise: Float = 0.08
let glowDecay: Float = 0.35
let glowFadeStart: Float = 0.9
let glowFadeEnd: Float = 1.5

func smoothstep(_ e0: Float, _ e1: Float, _ x: Float) -> Float {
    let t: Float = min(max((x - e0) / (e1 - e0), 0), 1)
    return t * t * (3 - 2 * t)
}

/// Brightness of the glow `d` seconds after the impulse arrives, 0…1,
/// repeating every loop. Mirrored in the kernel.
func pulse(_ since: Float) -> Float {
    var d: Float = since - loopSeconds * floor(since / loopSeconds)
    if d > loopSeconds - glowRise { d -= loopSeconds }
    let up: Float = smoothstep(-glowRise, 0, d)
    let down: Float = d < 0 ? 1 : exp(-d / glowDecay)
    return up * down * (1 - smoothstep(glowFadeStart, glowFadeEnd, d))
}

// MARK: - when the impulse reaches each piece

/// What to break, for `make mutants`. Each must make the suite fail.
enum ImpulseMutant: String {
    case none
    case startsInDendrite   // the first dendrite tree lights first
    case backwards          // terminals first, running back to the hillock
}

/// When the impulse reaches a piece of process: at its `a` end, at its `b`
/// end, and a jump added to every point past the break (the metre of axon it
/// stands for). A piece the impulse never enters has none.
struct Arrival {
    var ta: Float
    var tb: Float
    var breakShift: Float
}

struct Impulse {
    var dendrites: [Arrival?]
    var axon: [Arrival?]
}

/// The arrival time at a point of one piece: straight along it at the
/// picture's speed, plus the break's jump past the break. Mirrored in the
/// kernel.
func arrival(at p: SIMD3<Float>, on s: Neurite, _ a: Arrival, _ n: Neuron) -> Float {
    let ba: SIMD3<Float> = s.b - s.a
    let u: Float = min(max(simd_dot(p - s.a, ba) / simd_dot(ba, ba), 0), 1)
    var t: Float = a.ta + (a.tb - a.ta) * u
    if a.breakShift != 0 && simd_dot(p - n.breakCentre, n.breakNormal) > 0 { t += a.breakShift }
    return t
}

func buildImpulse(_ n: Neuron, _ mutant: ImpulseMutant = .none) -> Impulse {
    var axon: [Arrival?] = Array(repeating: nil, count: n.axon.count)
    var dendrites: [Arrival?] = Array(repeating: nil, count: n.dendrites.count)
    // The trunk is the axon's second piece, after the hillock (step 27). The
    // hillock itself never lights: the impulse starts just past it.
    let trunk: Int = 1
    let t0: Float = impulseStart + (mutant == .startsInDendrite ? 0.6 : 0)
    let trunkLength: Float = simd_distance(n.axon[trunk].a, n.axon[trunk].b)
    let tStart: Float = t0 + initiationArcLength / pictureSpeed
    axon[trunk] = Arrival(ta: tStart, tb: tStart + trunkLength / pictureSpeed, breakShift: breakSeconds)
    // Terminals and boutons, each from where its parent ends.
    for i in (trunk + 1)..<n.axon.count {
        let s: Neurite = n.axon[i]
        guard s.kind == .terminal || s.kind == .bouton, let p = s.parent, let pa = axon[p] else { continue }
        let start: Float = pa.tb + pa.breakShift
        axon[i] = Arrival(ta: start, tb: start + simd_distance(s.a, s.b) / pictureSpeed, breakShift: 0)
    }
    switch mutant {
    case .none:
        break
    case .startsInDendrite:
        for i in 0..<segmentsPerTree { dendrites[i] = Arrival(ta: impulseStart, tb: impulseStart, breakShift: 0) }
    case .backwards:
        // Time run in reverse over the same span: t → start + end − t.
        var end: Float = 0
        for case let a? in axon { end = max(end, a.tb + a.breakShift) }
        let mirror: Float = impulseStart + end
        for i in 0..<axon.count {
            guard let a = axon[i] else { continue }
            axon[i] = Arrival(ta: mirror - a.ta, tb: mirror - a.tb, breakShift: -a.breakShift)
        }
    }
    return Impulse(dendrites: dendrites, axon: axon)
}

/// The span of the loop in which anything glows at all, s. Outside it every
/// frame is the resting still, which the renderer draws once.
func activeWindow(_ imp: Impulse) -> (from: Float, to: Float) {
    var lo: Float = .greatestFiniteMagnitude
    var hi: Float = -.greatestFiniteMagnitude
    for case let a? in imp.dendrites + imp.axon {
        let ends: [Float] = [a.ta, a.tb, a.ta + a.breakShift, a.tb + a.breakShift]
        lo = min(lo, ends.min()!)
        hi = max(hi, ends.max()!)
    }
    return (lo - glowRise, hi + glowFadeEnd)
}

func isResting(_ t: Float, _ imp: Impulse) -> Bool {
    let w = activeWindow(imp)
    let phase: Float = t - loopSeconds * floor(t / loopSeconds)
    return !(phase >= w.from && phase <= w.to)
}
