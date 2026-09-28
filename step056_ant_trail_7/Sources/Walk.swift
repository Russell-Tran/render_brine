// A file of workers walking a pheromone trail: where each one is, how its
// legs step, when it dabs the trail — as functions of time. Nothing here
// touches the GPU. Step 44's walk, generalised from three ants to a file of
// `fileCount`.
//
// The world: MILLIMETRES, y up, the ground is y = 0, the trail runs along
// the x axis (z = 0) and the ants walk towards +x.
//
// Four rules shape everything below.
//
//   1. THE FEET ARE DRIVEN BY DISTANCE, NOT TIME. Each leg's phase is where
//      its ant is along the trail divided by the stride length. A foot in
//      stance is put down at a point fixed by where the ant was at
//      touchdown, so it cannot slide however the ant speeds up or stops; and
//      when the ant stops, every leg stops with it.
//
//   2. EVERY ANT WALKS THE SAME TIMETABLE, ONE HEADWAY APART. There is one
//      timetable, X(τ): where an ant is at its own clock τ. Ant k's clock is
//      the picture's clock plus k headways, so ant k+1 is exactly one headway
//      ahead of ant k — the gap measured in time, as Dussutour et al.
//      measured it. Where an ant is along the trail decides everything else
//      about it: its legs (rule 1), how the trail wanders under it, and
//      where it stops to dab — each ant stops at the same DAB SITES, and the
//      ant behind closes up and falls back, as a following ant does.
//
//   3. THE LOOP IS ONE HEADWAY: A CONVEYOR ONE ANT LONG. After one loop each
//      ant has walked exactly one spacing and stands, in every detail, where
//      the ant ahead of it stood at the start: frame N flows into frame 0,
//      and every position only ever advances. Nothing needs a whole number
//      of strides for this — step 44 needed that because its ants were
//      different individuals, each coming back round to itself; here each
//      ant turns into the one ahead, and the legs of the one ahead are, by
//      rule 1, exactly where its own will be. The price: the workers must be
//      identical (one size), since each becomes the next. Their antennae
//      still sweep out of step.
//
//   4. THE CAMERA DOES NOT MOVE. The ants walk in on the left and out on the
//      right.
//
// Every number carries where it came from, or says MODEL and why.

import Foundation
import simd

// MARK: - the file

/// How many workers the frame holds — Russell's number for this step.
let fileCount: Int = 7
/// Where the dab sites are, as nose-to-nose spacings from the frame centre
/// (each rounded to a whole number of strides, below). MODEL: step 44 had
/// one mark per ~22 mm walked; three sites 2.2 spacings (28 mm) apart give
/// each ant a mark every 28 mm (step 55's rate), and spread the three dabs
/// through the loop (1.4, 2.3 and 2.3 s apart).
let dabSiteSpacings: [Float] = [-2.2, 0, 2.2]

// MARK: - speed

// Walking speed, step 44's. Dussutour, Deneubourg & Fourcassié (2005, *J Exp
// Biol* 208(15): 2903, doi 10.1242/jeb.01711, "Temporal organization of
// bi-directional traffic in the ant Lasius niger") timed L. niger workers
// crossing a 10 mm-wide bridge "between the two bottlenecks ... in the
// absence of interactions with other ants": 2.96 ± 0.61 s. Between the
// bottlenecks lie an entrance (15 mm), the central part (60 mm) and the other
// entrance (15 mm): 90 mm. Colonies at 25 ± 1 °C. So the speed is DERIVED —
// 90 mm / 2.96 s = 30 mm/s — and rests on our reading of their bridge layout
// (had they timed only the central 60 mm, it would be 20 mm/s).
let dussutourDistance: Float = 90.0
let dussutourTime: Float = 2.96
let realSpeed: Float = dussutourDistance / dussutourTime    // mm/s, ≈ 30.4
let speedTemperature: Int = 25

// MARK: - how close they follow

// The headway: the time between one ant and the next passing the same point.
// The same paper, on the same bridges, in its most crowded trial (about 120
// ants a minute, both directions together): the delay "between two ants
// following each other and crossing the line between the bottleneck and the
// entrance ... was less than 0.4 s (0.5 s) for 31% (51%) of the total number
// of pairs of successive ants (N = 242 pairs)". So half of all following
// pairs were within about 0.5 s. We take 0.4 s: a close file, but one that
// almost a third of real pairs matched or beat. It was measured at a
// bottleneck, where ants bunch, and the spacing below is DERIVED from it with
// the uncrowded speed above.
let headwayReal: Float = 0.4
/// Nose-to-nose spacing while cruising, mm: ≈ 12.2 — three ant lengths.
let spacing: Float = realSpeed * headwayReal

// MARK: - the gait

// Ants walk with the alternating tripod: the fore and hind legs of one side
// step together with the middle leg of the other, and the two tripods
// alternate. Zollikofer (1994, *J Exp Biol* 192: 95–106, "Stepping patterns in
// ants I") filmed workers of twelve species in Cataglyphis, Formica, LASIUS
// and Myrmica: "the alternating tripod gait prevails over a wide range of
// speeds", with "temporal rigidity of tripod coordination".
let tripodA: Set<Int> = [0, 4, 2]    // right fore, left mid, right hind
let tripodB: Set<Int> = [3, 1, 5]    // left fore, right mid, left hind

// Duty factor — the fraction of each stride a foot is down. Reinhardt &
// Blickhan (2014, *J Exp Biol*, doi 10.1242/jeb.098426, wood ant Formica
// polyctena, "Level locomotion in wood ants"): 0.62 for the middle legs at
// 9.5–12.5 cm/s. 0.6 here: MODEL, Formica's figure borrowed for Lasius.
let dutyFactor: Float = 0.6

/// The ant's length, measured from the model.
let baseLength: Float = AntModel(body: bodyShapes(gasterBend: 0), legs: [], antennae: []).length

/// Stride length, mm: step 44's, kept — it is the stride these legs were
/// tested to reach (a test checks no leg is ever stretched). Step 44 derived
/// it as its conveyor (3 × (ant + 3.5 mm gap)) over 9 strides: 2.53 mm, 0.62
/// body lengths, which at 30 mm/s is 12 strides a second, close to the
/// 11.7 ± 0.4 Hz Reinhardt & Blickhan measured in Formica.
let strideLength: Float = 3 * (baseLength + 3.5) / 9
let stepFrequency: Float = realSpeed / strideLength   // real strides per second

/// How high a swinging foot lifts, mm. MODEL (step 44): low, skimming steps.
let footLift: Float = 0.18

/// Where each foot sits at mid-stance, in the ant's frame (right side; the
/// left mirrors). MODEL, step 44's.
let stanceCentres: [SIMD3<Float>] = [
    SIMD3(1.75, 0, 1.35),
    SIMD3(0.26, 0, 1.60),
    SIMD3(-1.60, 0, 1.50),
]

/// Every worker's size relative to step 32's model. One size for all: the
/// conveyor turns each ant into the one ahead (rule 3). Step 32's ant is
/// 4.1 mm, inside the cited worker range of 3.4–5.0 mm.
let antScale: Float = 1.0

/// The trail's own wander: every ant follows the same line, so the lateral
/// offset is a function of where along the trail an ant is. MODEL, as step
/// 44: no measured lateral wander was found; L. niger follow trails closely —
/// 74–83% correct choices at a fork (Czaczkes, Castorena, Schürch & Heinze
/// 2017, *Physiol Entomol*, doi 10.1111/phen.12174) — so the line stays
/// within a tenth of a millimetre of the drawn trail. (amplitude mm,
/// wavelength mm, phase)
let wander: [(amp: Float, wavelength: Float, phase: Float)] = [(0.06, 19.0, 0.3), (0.035, 7.3, 2.1)]

/// Antennal sweeps per second of picture. MODEL (step 44's 5–6 per 11.25 s,
/// 7–8 a second in real time). Neighbours are 0.4 s apart, so they sweep
/// about 0.7 of a sweep out of step.
let sweepRate: Float = 0.45

// MARK: - the dab

// Trail laying. L. niger lays its trail from the hindgut: "In Lasius niger
// the trail pheromone was identified as 3,4-dihydro-8-hydroxy-3,5,7-
// trimethylisocoumarin (Bestmann et al. 1992)" — Bestmann, Kern, Schäfer &
// Witschel, *Angew Chem Int Ed* 31: 795–796 — and "in all cases where a
// formicine species' trail pheromone has been identified, it has been
// located in the hindgut" (both as restated by Butterfield, Bacon & Hill,
// *J Chem Ecol* 2025). It is laid in dots, not a stripe: "Pheromone
// deposition is a very stereotyped behaviour in L. niger ... It involves the
// ant pausing for ca. 0.2 seconds, backing up, and firmly pressing the tip of
// their abdomen onto the substrate" (Czaczkes et al. 2016, *PLoS ONE*, "The
// effect of trail pheromone and path confinement on learning of complex
// routes in the ant Lasius niger", PMC4784821). Here: a 0.2 s stop, the
// gaster bent down until its tip meets the ground, held while the ant stands
// still. The "backing up" is left out — this loop only ever moves forward.
let dabPauseReal: Float = 0.2      // s, the whole slow-down and stop
let dabHoldReal: Float = 0.06      // s of it standing still, gaster down. MODEL
let dabBendReal: Float = 0.05      // s for the gaster to bend down or lift. MODEL
/// Between dabs the gaster is carried a little raised. MODEL (step 44).
let walkingGasterLift: Float = -6 * Float.pi / 180

/// How long a fresh mark stays highlighted, picture seconds. The real mark
/// lasts far longer — "a single dot of L. niger trail pheromone has been
/// estimated to become undetectable in 47 min at room temperature" (Forster,
/// Czaczkes, Warner et al. 2014, *Ethology*, doi 10.1111/eth.12248) — so
/// this is a highlight, not a lifetime, and the caption says so.
let markFade: Float = 3.5
let markLifetimeMinutes: Int = 47

// MARK: - the loop

/// How much the picture is slowed. MODEL, step 44's ×15: at 30 mm/s and 12
/// strides a second, ×15 shows a stride in about 1.2 s.
let slowdown: Float = 15
/// The picture's loop: one headway, slowed. 6.0 s.
let loopSeconds: Float = headwayReal * slowdown
/// The GIF's delay per frame, hundredths of a second (20 fps: 120 frames).
let gifDelayCentiseconds: Int = 5
/// The GIF: its width in pixels, and its name.
let gifWidth: Int = 2770
let gifName: String = "ant_trail_7.gif"
/// Cruising speed in the picture, mm per picture second.
let cruise: Float = realSpeed / slowdown

/// The dab's timings in picture seconds.
let dabWindow: Float = dabPauseReal * slowdown
let dabHold: Float = dabHoldReal * slowdown
let dabRamp: Float = (dabWindow - dabHold) / 2
let dabBend: Float = dabBendReal * slowdown
/// Walking time a stop costs, picture seconds.
let dabLoss: Float = dabHold + dabRamp
/// The gaster's bend and tip flex at the moment its tip meets the ground.
let dabFull: (bend: Float, flex: Float) = dabPose()

/// A flat-topped bump: 1 across the hold, easing to 0 over `ramp` each side.
func plateau(_ tau: Float, hold: Float, ramp: Float) -> Float {
    let a: Float = abs(tau)
    if a <= hold / 2 { return 1 }
    if a >= hold / 2 + ramp { return 0 }
    return 0.5 * (1 + cos(Float.pi * (a - hold / 2) / ramp))
}

/// ∫ plateau from −∞ to tau, exactly.
func plateauIntegral(_ tau: Float, hold: Float, ramp: Float) -> Float {
    let h: Float = hold / 2
    if tau <= -h - ramp { return 0 }
    if tau < -h {
        let u: Float = tau + h + ramp
        return 0.5 * (u - ramp / Float.pi * sin(Float.pi * u / ramp))
    }
    if tau <= h { return ramp / 2 + (tau + h) }
    if tau < h + ramp {
        let u: Float = tau - h
        return ramp / 2 + hold + 0.5 * (u + ramp / Float.pi * sin(Float.pi * u / ramp))
    }
    return ramp + hold
}

// MARK: - the timetable

/// Where the dab sites are, mm along the trail, and the gait's phase offset.
/// The first site is placed, then the offset chosen so an ant stopping there
/// stands in the middle of a changeover (both tripods down: phase 0 to
/// duty − ½ for tripod A, so (duty − ½)/2); the other sites sit a whole
/// number of strides from it, so the same holds at each.
let gaitOffset: Float = {
    let x0: Float = dabSiteSpacings[0] * spacing
    let want: Float = (dutyFactor - 0.5) / 2
    var o: Float = (want - x0 / strideLength).truncatingRemainder(dividingBy: 1)
    if o < 0 { o += 1 }
    return o
}()
let dabSites: [Float] = dabSiteSpacings.map { s in
    let x0: Float = dabSiteSpacings[0] * spacing
    let strides: Float = ((s * spacing - x0) / strideLength).rounded()
    return x0 + strides * strideLength
}

/// When on the timetable (τ, picture seconds) the ant is stopped at each
/// site. X(0) = 0; before the first stop, X(τ) = cruise · τ, and every stop
/// costs `dabLoss` of walking.
let dabTaus: [Float] = dabSites.enumerated().map { (i, x) in
    let lostBefore: Float = dabLoss * (Float(i) + 0.5)
    return x / cruise + lostBefore
}

/// Walking time lost to stops by timetable time τ.
func lostTime(_ tau: Float) -> Float {
    var sum: Float = 0
    for c in dabTaus { sum += plateauIntegral(tau - c, hold: dabHold, ramp: dabRamp) }
    return sum
}

/// The timetable: where an ant is along the trail at its own clock τ.
func trailX(_ tau: Float) -> Float { cruise * (tau - lostTime(tau)) }

/// Speed on the timetable, mm per picture second.
func timetableSpeed(_ tau: Float) -> Float {
    var stop: Float = 0
    for c in dabTaus { stop += plateau(tau - c, hold: dabHold, ramp: dabRamp) }
    return cruise * (1 - stop)
}

/// How far into a dab the gaster is, 0–1: full (tip on the ground) only
/// while the ant stands still — 80% of the standing time, as step 44.
func dabAmount(_ tau: Float) -> Float {
    var k: Float = 0
    for c in dabTaus { k = max(k, plateau(tau - c, hold: dabHold * 0.8, ramp: dabBend)) }
    return k
}

/// The gaster's bend and its tip's flex at timetable time τ.
func gasterPose(_ tau: Float) -> (bend: Float, flex: Float) {
    let k: Float = dabAmount(tau)
    return (walkingGasterLift + (dabFull.bend - walkingGasterLift) * k, dabFull.flex * k)
}

/// The picture's clock as the ants see it. With the rewind mutant, the second
/// half of each loop plays the first half backwards.
func antClock(_ t: Float, mutant: Mutant) -> Float {
    if mutant != .rewind { return t }
    let lap: Float = (t / loopSeconds).rounded(.down)
    let u: Float = t - lap * loopSeconds
    return lap * loopSeconds + (u <= loopSeconds / 2 ? u : loopSeconds - u)
}

/// Ant k's own clock at picture time t: k headways ahead.
func antTau(_ k: Int, time t: Float, mutant: Mutant) -> Float {
    antClock(t, mutant: mutant) + Float(k) * loopSeconds
}

/// Where an ant at `x` along the trail stands, facing where.
func pose(_ x: Float) -> (at: SIMD3<Float>, yaw: Float) {
    var z: Float = 0
    var slope: Float = 0
    for w in wander {
        let k: Float = 2 * Float.pi / w.wavelength
        let arg: Float = k * x + w.phase
        z += w.amp * sin(arg)
        slope += w.amp * k * cos(arg)
    }
    return (SIMD3<Float>(x, 0, z), atan(slope))
}

/// Leg `j`'s phase offset: tripod B half a stride behind tripod A.
func legOffset(_ j: Int, mutant: Mutant) -> Float {
    if mutant == .nonTripod { return gaitOffset }
    return gaitOffset + (tripodA.contains(j) ? 0 : 0.5)
}

/// Leg j's foot at mid-stance, in the ant's frame, shifted so the stance
/// straddles it.
func stanceLocal(_ j: Int) -> SIMD3<Float> {
    let side: Float = j < 3 ? 1 : -1
    var local: SIMD3<Float> = stanceCentres[j % 3] * SIMD3<Float>(1, 1, side)
    let halfStance: Float = dutyFactor * strideLength / 2
    local.x += halfStance / antScale
    return local
}

/// Where leg `j`'s foot lands for the stance that begins with the ant at `x`.
func touchdown(_ j: Int, at x: Float) -> SIMD3<Float> {
    let (at, yaw) = pose(x)
    return at + turn(stanceLocal(j), yaw) * antScale
}

/// Leg `j`'s foot in the world with the ant at `x`, and whether it is down.
func foot(_ j: Int, at x: Float, mutant: Mutant) -> (point: SIMD3<Float>, stance: Bool) {
    let off: Float = legOffset(j, mutant: mutant)
    let cycles: Float = x / strideLength + off
    let k: Float = cycles.rounded(.down)
    let phase: Float = cycles - k
    let start: Float = (k - off) * strideLength       // where the ant was at this stride's touchdown
    if phase < dutyFactor {
        if mutant == .sliding {
            // Planted in the ANT's frame, so it rides along with the body.
            let (at, yaw) = pose(x)
            return (at + turn(stanceLocal(j), yaw) * antScale, true)
        }
        return (touchdown(j, at: start), true)
    }
    let u: Float = (phase - dutyFactor) / (1 - dutyFactor)
    let from: SIMD3<Float> = touchdown(j, at: start)
    let to: SIMD3<Float> = touchdown(j, at: start + strideLength)
    let ease: Float = 3 - 2 * u
    let e: Float = u * u * ease
    var p: SIMD3<Float> = from + (to - from) * e
    p.y = footLift * antScale * sin(Float.pi * u)
    return (p, false)
}

/// Body +x → (cos, 0, sin), body +z → (−sin, 0, cos).
func turn(_ v: SIMD3<Float>, _ yaw: Float) -> SIMD3<Float> {
    let c: Float = cos(yaw)
    let s: Float = sin(yaw)
    return SIMD3<Float>(v.x * c - v.z * s, v.y, v.x * s + v.z * c)
}

func unturn(_ v: SIMD3<Float>, _ yaw: Float) -> SIMD3<Float> { turn(v, -yaw) }

// MARK: - one ant at one moment

/// One ant, posed and placed in the world.
struct WorldAnt {
    let index: Int                   // place in the endless file; larger is further ahead
    let tau: Float                   // its own clock
    let distance: Float              // where it is along the trail, mm
    let at: SIMD3<Float>
    let yaw: Float
    let model: AntModel              // in the ant's own frame
    let shapes: [Shape]              // in the world
    let feet: [SIMD3<Float>]         // lowest point of each foot, world
    let stance: [Bool]               // what the timetable says, for comparison

    /// Every shape's lowest point.
    var lowest: Float { shapes.map { $0.extent(along: SIMD3(0, 1, 0)).lo }.min() ?? 0 }
    var bound: (centre: SIMD3<Float>, radius: Float) {
        let c: SIMD3<Float> = at + SIMD3<Float>(0, 0.6, 0)
        let r: Float = shapes.map { simd_distance($0.bound.centre, c) + $0.bound.radius }.max() ?? 0
        return (c, r)
    }
}

/// The antenna tips sweep side to side over the trail just above the ground
/// ahead of the head, the two moving together — one crossing inward as the
/// other swings out. Returns scape direction and tip, in the ant's frame.
func antennaPose(_ index: Int, tau: Float) -> (scape: SIMD3<Float>, tip: SIMD3<Float>) {
    let sgn: Float = index == 0 ? 1 : -1
    let w: Float = 2 * Float.pi * sweepRate
    let sweep: Float = sin(w * tau)
    let rest: Float = sgn * 0.28
    let tipZ: Float = rest + 0.26 * sweep
    // The scape swings forward and out, turning with the sweep. MODEL pose.
    let scape: SIMD3<Float> = simd_normalize(SIMD3<Float>(0.75, 0.02, sgn * 0.66 + 0.8 * (tipZ - rest)))
    let elbow: SIMD3<Float> = antennaSocket(side: sgn) + scape * scapeLength
    // The tip just above the ground (its round end 0.03 mm clear), as far
    // ahead as leaves the funiculus a gentle bow: chord 94% of its length.
    let funiculus: Float = funiculusLengths(count: workerAntennaSegments - 1).reduce(0, +)
    let y: Float = funiculusTipRadius + 0.03
    let chord: Float = 0.94 * funiculus
    let dy: Float = elbow.y - y
    let dz: Float = elbow.z - tipZ
    let dx: Float = max(chord * chord - dy * dy - dz * dz, 0).squareRoot()
    return (scape, SIMD3<Float>(elbow.x + dx, y, tipZ))
}

/// The ant whose own clock reads τ.
func worldAnt(tau: Float, index: Int, mutant: Mutant) -> WorldAnt {
    let x: Float = trailX(tau)
    let (at, yaw) = pose(x)
    var legs: [Leg] = []
    var feet: [SIMD3<Float>] = []
    var stance: [Bool] = []
    for j in 0..<6 {
        let (p, down) = foot(j, at: x, mutant: mutant)
        let local: SIMD3<Float> = unturn(p - at, yaw) / antScale
        legs.append(buildLeg(index: j, foot: local))
        feet.append(p)
        stance.append(down)
    }
    var antennae: [Antenna] = []
    for i in 0..<2 {
        let (scape, tip) = antennaPose(i, tau: tau)
        antennae.append(buildAntenna(index: i, scapeDir: scape, tip: tip))
    }
    let (bend, flex) = gasterPose(tau)
    let model = AntModel(body: bodyShapes(gasterBend: bend, tipFlex: flex), legs: legs, antennae: antennae)
    let shapes: [Shape] = model.shapes.map { $0.placed(scale: antScale, yaw: yaw, at: at) }
    return WorldAnt(index: index, tau: tau, distance: x, at: at, yaw: yaw, model: model, shapes: shapes,
                    feet: feet, stance: stance)
}

/// Ant k at picture time t.
func worldAnt(_ k: Int, time t: Float, mutant: Mutant) -> WorldAnt {
    worldAnt(tau: antTau(k, time: t, mutant: mutant), index: k, mutant: mutant)
}

/// Every ant whose body lies within `range` of x at time t.
func antsInView(time t: Float, range: ClosedRange<Float>, mutant: Mutant) -> [WorldAnt] {
    // X(τ) ≈ cruise · τ give or take the stops, so ant k is near
    // cruise · (t + k · loop); search a generous band of k and keep those in range.
    let t0: Float = antClock(t, mutant: mutant)
    let lo: Int = Int(((range.lowerBound - 12) / cruise - t0) / loopSeconds) - 2
    let hi: Int = Int(((range.upperBound + 12) / cruise - t0) / loopSeconds) + 2
    var out: [WorldAnt] = []
    for k in lo...hi {
        let x: Float = trailX(antTau(k, time: t, mutant: mutant))
        if x > range.lowerBound - 4.5 && x < range.upperBound + 4.5 {
            out.append(worldAnt(k, time: t, mutant: mutant))
        }
    }
    return out
}

// MARK: - the marks on the trail

/// Where the dab at each site touches the ground: the bottom of the gaster
/// tip's round end, with the ant stopped there.
func dabPoint(_ tau: Float) -> SIMD3<Float> {
    let a: WorldAnt = worldAnt(tau: tau, index: 0, mutant: .none)
    var cones: [Shape] = []
    for s in a.shapes where s.part == AntPart.gaster {
        if s.kind == Shape.Kind.roundCone { cones.append(s) }
    }
    let tip: Shape = cones[0]
    return tip.b - SIMD3<Float>(0, tip.rb, 0)
}

/// A fresh mark: where, and how bright (0–1+; the kernel clamps).
struct Mark {
    var point: SIMD3<Float>
    var strength: Float
}

let dabPoints: [SIMD3<Float>] = dabTaus.map { dabPoint($0) }

/// How bright a mark is `age` picture seconds after the gaster touched.
func markGlow(_ age: Float) -> Float { (1 - exp(-age / 0.12)) * exp(-age / markFade) }

/// The fresh marks at time t. At each site a new ant dabs every loop, on the
/// same spot, so each site's glow is the sum over every dab so far — which
/// is periodic in the loop, and never jumps when a new dab lands.
func marks(time t: Float, range: ClosedRange<Float>) -> [Mark] {
    var out: [Mark] = []
    for (i, c) in dabTaus.enumerated() {
        let p: SIMD3<Float> = dabPoints[i]
        guard p.x > range.lowerBound - 1 && p.x < range.upperBound + 1 else { continue }
        var age: Float = (t - (c - dabHold / 2)).truncatingRemainder(dividingBy: loopSeconds)
        if age < 0 { age += loopSeconds }
        var s: Float = 0
        for n in 0..<12 { s += markGlow(age + Float(n) * loopSeconds) }
        out.append(Mark(point: SIMD3<Float>(p.x, 0, p.z), strength: s))
    }
    return out
}
