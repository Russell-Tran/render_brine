// Three workers walking a pheromone trail: where each one is, how its legs
// step, when it dabs the trail — as functions of time. Nothing here touches
// the GPU.
//
// The world: MILLIMETRES, y up, the ground is y = 0, the trail runs along
// the x axis (z = 0) and the ants walk towards +x.
//
// Three rules shape everything below.
//
//   1. THE FEET ARE DRIVEN BY DISTANCE, NOT TIME. Each leg's phase is the
//      distance its ant has walked divided by the stride length. A foot in
//      stance is put down at a point fixed by the distance at touchdown, so
//      it cannot slide however the ant speeds up or stops; and when the ant
//      stops, every leg stops with it.
//
//   2. THE LOOP IS A CONVEYOR. The ants are an endless file with period
//      G = 3 × spacing — ant 0, 1, 2, ant 0, 1, 2, … — and in one loop each
//      one walks exactly G. So at the end of the loop every ant stands where
//      the same-numbered ant ahead of it stood at the start: frame N flows
//      into frame 0 with positions only ever advancing. The stride divides G
//      a whole number of times, so the legs come round too.
//
//   3. THE CAMERA DOES NOT MOVE. The ants walk in on the left and out on the
//      right.
//
// Every number carries where it came from, or says MODEL and why.

import Foundation
import simd

// MARK: - speed

// Walking speed. Dussutour, Deneubourg & Fourcassié (2005, *J Exp Biol*
// 208(15): 2903, doi 10.1242/jeb.01711, "Temporal organization of bi-directional traffic in the ant
// Lasius niger") timed L. niger workers crossing a 10 mm-wide bridge "between
// the two bottlenecks ... in the absence of interactions with other ants":
// 2.96 ± 0.61 s. Between the bottlenecks lie an entrance (15 mm), the
// central part (60 mm) and the other entrance (15 mm): 90 mm. Colonies at
// 25 ± 1 °C. So the speed is DERIVED — 90 mm / 2.96 s = 30 mm/s — and rests on
// our reading of their bridge layout (had they timed only the central 60 mm,
// it would be 20 mm/s). Ant walking speed rises steeply with temperature, so
// it is labelled with its 25 °C.
let dussutourDistance: Float = 90.0
let dussutourTime: Float = 2.96
let realSpeed: Float = dussutourDistance / dussutourTime    // mm/s, ≈ 30.4
let speedTemperature: Int = 25

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
// polyctena, "Level locomotion in wood ants"):
// 0.62 for the middle legs at 9.5–12.5 cm/s, falling below 0.5 only near
// 18.8 cm/s. Above 0.5 both tripods share the ground briefly at each
// changeover. 0.6 here: MODEL, Formica's figure borrowed for Lasius.
let dutyFactor: Float = 0.6

// Stride length and frequency. Reinhardt & Blickhan found wood ants step at
// 11.7 ± 0.4 Hz and lengthen the stride and quicken the step together to go
// faster. For Lasius no stride table was found. The stride here is DERIVED
// from the loop (it must divide it; see `stridesPerGroup`) and comes out at
// 2.5 mm — 0.62 body lengths, about the most these legs reach — which at
// 30 mm/s is 12 strides a second, close to Formica's 11.7 Hz.
let stridesPerGroup: Int = 9

/// How high a swinging foot lifts, mm. MODEL: small insects walk with low,
/// skimming steps; about a fifth of a millimetre reads without looking like
/// a march.
let footLift: Float = 0.18

/// Where each foot sits at mid-stance, in the ant's frame (right side; the
/// left mirrors). MODEL: step 26's relaxed standing feet, the fore foot drawn
/// back and the hind foot out behind a little so a full stride stays within
/// what femur and tibia can reach (a test checks no leg is ever stretched).
let stanceCentres: [SIMD3<Float>] = [
    SIMD3(1.75, 0, 1.35),
    SIMD3(0.26, 0, 1.60),
    SIMD3(-1.60, 0, 1.50),
]

// MARK: - the file of ants

/// Nose-to-tail gap between one ant's gaster tip and the next one's
/// mandibles, mm. MODEL: no measured following distance for L. niger was
/// found; a little under one body length keeps them clearly a file, and
/// leaves room for the antennae (which reach about 1 mm ahead) and for a
/// dabbing ant's stop (a test checks no two ants ever touch).
let noseToTailGap: Float = 3.5

/// Small differences between the three, so they are not copies. MODEL,
/// all inside the cited worker range of 3.4–5.0 mm.
let antScales: [Float] = [1.00, 0.96, 1.03]

/// Heading wobble about the trail: lateral offset amplitude (mm), whole
/// wobbles per loop, and phase. MODEL: no measured lateral wander was found.
/// L. niger follow trails well — 74–83% correct choices at a fork, "more
/// accurate than previously reported" (Czaczkes, Castorena, Schürch &
/// Heinze 2017, *Physiol Entomol*, doi 10.1111/phen.12174) — and these ants
/// stay within a tenth of a millimetre of the line.
let wobbleAmplitude: [Float] = [0.07, 0.09, 0.06]
let wobblesPerLoop: [Int] = [2, 3, 2]
let wobblePhase: [Float] = [0.3, 2.1, 4.0]

/// Antennal sweeps per loop, and phase. MODEL: no measured sweep rate was
/// found. Five or six per loop is 7–8 a second in real time.
let sweepsPerLoop: [Int] = [5, 6, 5]
let sweepPhase: [Float] = [0.0, 1.3, 2.6]

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
// routes in the ant Lasius niger", PMC4784821). Here: a 0.2 s stop, the gaster bent down until its tip
// meets the ground, held while the ant stands still. The "backing up" is
// left out — this loop only ever moves forward.
let dabPauseReal: Float = 0.2      // s, the whole slow-down and stop
let dabHoldReal: Float = 0.06      // s of it standing still, gaster down. MODEL
let dabBendReal: Float = 0.05      // s for the gaster to bend down or lift. MODEL
/// Between dabs the gaster is carried a little raised, clear of the ground.
/// MODEL: step 26's resting gaster, lifted 6° about the petiole.
let walkingGasterLift: Float = -6 * Float.pi / 180

/// When in the loop each ant dabs, as a fraction of the loop. Each ant dabs
/// once a loop (MODEL rate: one mark every 22 mm or so). Staggered by a
/// third, the ant BEHIND stopping a third of a loop before the ant ahead, so
/// a stop only ever opens a gap first; a little jitter keeps it from looking
/// clockwork.
let dabFraction: [Float] = [0.14, 0.14 - 1.0 / 3.0 + 0.03, 0.14 + 1.0 / 3.0 - 0.02]

/// How long a fresh mark stays highlighted, picture seconds. The real mark
/// lasts far longer — "a single dot of L. niger trail pheromone has been
/// estimated to become undetectable in 47 min at room temperature" (Forster,
/// Czaczkes, Warner et al. 2014, *Ethology*, doi 10.1111/eth.12248) — so
/// this is a highlight, not a lifetime, and the caption says so.
let markFade: Float = 3.5
let markLifetimeMinutes: Int = 47

// MARK: - the loop

/// The picture's loop, seconds, and the GIF's delay per frame, hundredths:
/// 225 frames of 5 hundredths (20 fps). Chosen after measuring ten frames
/// (main.swift): the three moving ants span the frame, so every frame stores
/// a strip the full width, about 40 KB at 1100 px wide.
let loopSeconds: Float = 11.25
let gifDelayCentiseconds: Int = 5

/// The ant's length at scale 1, measured from the model.
let baseLength: Float = AntModel(body: bodyShapes(gasterBend: 0), legs: [], antennae: []).length
/// Nose-to-nose spacing, and the conveyor's period: three ants.
let spacing: Float = baseLength + noseToTailGap
let groupPeriod: Float = 3 * spacing
let strideLength: Float = groupPeriod / Float(stridesPerGroup)
/// Real time the loop stands for, and so how much the picture is slowed.
let realLoopSeconds: Float = groupPeriod / realSpeed
let slowdown: Float = loopSeconds / realLoopSeconds
let stepFrequency: Float = realSpeed / strideLength   // real strides per second

/// The dab's timings in picture seconds.
let dabWindow: Float = dabPauseReal * slowdown
let dabHold: Float = dabHoldReal * slowdown
let dabRamp: Float = (dabWindow - dabHold) / 2
let dabBend: Float = dabBendReal * slowdown
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

/// The same time wrapped into [−T/2, T/2) around an event.
func wrapped(_ t: Float) -> Float {
    var x: Float = t.truncatingRemainder(dividingBy: loopSeconds)
    if x < -loopSeconds / 2 { x += loopSeconds }
    if x >= loopSeconds / 2 { x -= loopSeconds }
    return x
}

/// One ant's timetable, and the things that make it itself.
struct AntPlan {
    private(set) var id: Int
    private(set) var scale: Float
    private(set) var dabTime: Float   // picture seconds, centre of the stop
    private(set) var cruise: Float    // mm/s in the picture when not stopping
    private(set) var lead: Float      // mm subtracted so the ant keeps its place on average
    private(set) var gaitOffset: Float  // tripod A's phase at distance 0

    /// Build ant `id`'s plan: its dab time and cruising speed from the tables
    /// above, then two derived numbers. `lead` centres the ant on its place
    /// in the file: it runs a little ahead of its mean place before a stop
    /// and a little behind after, and averaged over the loop it is exactly
    /// where the spacing puts it (without this, every ant would start the
    /// loop level and the one that stops first would be run into). The gait
    /// offset makes it stop with all six feet down.
    init(id: Int) {
        self.id = id
        scale = antScales[id]
        var f: Float = dabFraction[id].truncatingRemainder(dividingBy: 1)
        if f < 0 { f += 1 }
        dabTime = f * loopSeconds
        // Walk G in one loop although the stop costs `hold + ramp` seconds
        // of walking: cruise a little faster the rest of the time.
        cruise = groupPeriod / (loopSeconds - (dabHold + dabRamp))
        lead = 0
        gaitOffset = 0
        let mean: Float = groupPeriod / loopSeconds
        var sum: Float = 0
        let n: Int = 4000
        for k in 0..<n {
            let t: Float = (Float(k) + 0.5) / Float(n) * loopSeconds
            sum += rawDistance(t) - mean * t
        }
        let centred = AntPlan(copy: self, lead: sum / Float(n), gaitOffset: 0)
        // Stop in the middle of a changeover, when both tripods stand: phase
        // 0 to (duty − ½) for tripod A, so (duty − ½)/2 is its middle.
        let d: Float = centred.rawDistance(dabTime)
        let want: Float = (dutyFactor - 0.5) / 2
        var o: Float = (want - d / strideLength).truncatingRemainder(dividingBy: 1)
        if o < 0 { o += 1 }
        self = AntPlan(copy: centred, lead: centred.lead, gaitOffset: o)
    }

    private init(copy p: AntPlan, lead: Float, gaitOffset: Float) {
        id = p.id; scale = p.scale; dabTime = p.dabTime; cruise = p.cruise
        self.lead = lead
        self.gaitOffset = gaitOffset
    }

    private func stopped(_ t: Float) -> Float {
        var sum: Float = 0
        for k in -2...2 {
            sum += plateauIntegral(t - dabTime - Float(k) * loopSeconds, hold: dabHold, ramp: dabRamp)
        }
        return sum
    }

    /// Distance walked, mm, on the forward-only timetable: cruise, less the
    /// time lost stopping, less the lead.
    func rawDistance(_ t: Float) -> Float {
        let lap: Float = (t / loopSeconds).rounded(.down)
        let u: Float = t - lap * loopSeconds
        return lap * groupPeriod + cruise * (u - (stopped(u) - stopped(0))) - lead
    }

    /// Distance walked since t = 0 — with the rewind mutant, the second half
    /// of each loop plays the first half backwards.
    func distance(_ t: Float, mutant: Mutant) -> Float {
        if mutant == .rewind {
            var u: Float = t.truncatingRemainder(dividingBy: loopSeconds)
            if u < 0 { u += loopSeconds }
            return rawDistance(u <= loopSeconds / 2 ? u : loopSeconds - u)
        }
        return rawDistance(t)
    }

    /// Speed in the picture, mm/s: cruise, easing to a stop around the dab.
    func speed(_ t: Float) -> Float {
        cruise * (1 - plateau(wrapped(t - dabTime), hold: dabHold, ramp: dabRamp))
    }

    /// How far into the dab the gaster is, 0–1: down during the approach,
    /// full (tip on the ground) only while the ant stands still — its hold
    /// is 80% of the standing time, so it is already lifting as the ant
    /// moves off.
    func dabAmount(_ t: Float) -> Float {
        plateau(wrapped(t - dabTime), hold: dabHold * 0.8, ramp: dabBend)
    }

    /// The gaster's bend and its tip's flex at time t.
    func gaster(_ t: Float) -> (bend: Float, flex: Float) {
        let k: Float = dabAmount(t)
        return (walkingGasterLift + (dabFull.bend - walkingGasterLift) * k, dabFull.flex * k)
    }

    /// Where the ant stands, facing where, after walking `d`. Copy `m` is the
    /// same ant m group periods further along the file.
    func pose(_ d: Float, copy m: Int) -> (at: SIMD3<Float>, yaw: Float) {
        let k: Float = 2 * Float.pi * Float(wobblesPerLoop[id]) / groupPeriod
        let a: Float = wobbleAmplitude[id]
        let z: Float = a * sin(k * d + wobblePhase[id])
        let slope: Float = a * k * cos(k * d + wobblePhase[id])
        let x: Float = -Float(id) * spacing + Float(m) * groupPeriod + d
        return (SIMD3<Float>(x, 0, z), atan(slope))
    }

    /// Leg `j`'s phase offset: tripod B half a stride behind tripod A.
    func legOffset(_ j: Int, mutant: Mutant) -> Float {
        if mutant == .nonTripod { return gaitOffset }
        return gaitOffset + (tripodA.contains(j) ? 0 : 0.5)
    }

    /// Where leg `j`'s foot lands for the stance that begins at distance `d`.
    func touchdown(_ j: Int, at d: Float, copy m: Int) -> SIMD3<Float> {
        let side: Float = j < 3 ? 1 : -1
        var local: SIMD3<Float> = stanceCentres[j % 3] * SIMD3<Float>(1, 1, side)
        local.x += dutyFactor * strideLength / 2 / scale
        let (at, yaw) = pose(d, copy: m)
        return at + turn(local, yaw) * scale
    }

    /// Leg `j`'s foot in the world at distance `d`, and whether it is down.
    func foot(_ j: Int, at d: Float, copy m: Int, mutant: Mutant) -> (point: SIMD3<Float>, stance: Bool) {
        let off: Float = legOffset(j, mutant: mutant)
        let cycles: Float = d / strideLength + off
        let k: Float = cycles.rounded(.down)
        let phase: Float = cycles - k
        let start: Float = (k - off) * strideLength       // distance at this stride's touchdown
        if phase < dutyFactor {
            if mutant == .sliding {
                // Planted in the ANT's frame, so it rides along with the body.
                let side: Float = j < 3 ? 1 : -1
                var local: SIMD3<Float> = stanceCentres[j % 3] * SIMD3<Float>(1, 1, side)
                local.x += dutyFactor * strideLength / 2 / scale
                let (at, yaw) = pose(d, copy: m)
                return (at + turn(local, yaw) * scale, true)
            }
            return (touchdown(j, at: start, copy: m), true)
        }
        let u: Float = (phase - dutyFactor) / (1 - dutyFactor)
        let from: SIMD3<Float> = touchdown(j, at: start, copy: m)
        let to: SIMD3<Float> = touchdown(j, at: start + strideLength, copy: m)
        let e: Float = u * u * (3 - 2 * u)
        var p: SIMD3<Float> = from + (to - from) * e
        p.y = footLift * scale * sin(Float.pi * u)
        return (p, false)
    }
}

/// Body +x → (cos, 0, sin), body +z → (−sin, 0, cos).
func turn(_ v: SIMD3<Float>, _ yaw: Float) -> SIMD3<Float> {
    let c: Float = cos(yaw)
    let s: Float = sin(yaw)
    return SIMD3<Float>(v.x * c - v.z * s, v.y, v.x * s + v.z * c)
}

func unturn(_ v: SIMD3<Float>, _ yaw: Float) -> SIMD3<Float> { turn(v, -yaw) }

let antPlans: [AntPlan] = (0..<3).map { AntPlan(id: $0) }

// MARK: - one ant at one moment

/// One ant, posed and placed in the world.
struct WorldAnt {
    let id: Int
    let copy: Int
    let distance: Float
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
func antennaPose(_ index: Int, plan: AntPlan, time t: Float) -> (scape: SIMD3<Float>, tip: SIMD3<Float>) {
    let sgn: Float = index == 0 ? 1 : -1
    let w: Float = 2 * Float.pi * Float(sweepsPerLoop[plan.id]) / loopSeconds
    let sweep: Float = sin(w * t + sweepPhase[plan.id])
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

func worldAnt(_ plan: AntPlan, copy m: Int, time t: Float, mutant: Mutant) -> WorldAnt {
    let d: Float = plan.distance(t, mutant: mutant)
    let (at, yaw) = plan.pose(d, copy: m)
    var legs: [Leg] = []
    var feet: [SIMD3<Float>] = []
    var stance: [Bool] = []
    for j in 0..<6 {
        let (p, down) = plan.foot(j, at: d, copy: m, mutant: mutant)
        let local: SIMD3<Float> = unturn(p - at, yaw) / plan.scale
        legs.append(buildLeg(index: j, foot: local))
        feet.append(p)
        stance.append(down)
    }
    var antennae: [Antenna] = []
    for i in 0..<2 {
        let (scape, tip) = antennaPose(i, plan: plan, time: t)
        antennae.append(buildAntenna(index: i, scapeDir: scape, tip: tip))
    }
    let (bend, flex) = plan.gaster(t)
    let model = AntModel(body: bodyShapes(gasterBend: bend, tipFlex: flex), legs: legs, antennae: antennae)
    let shapes: [Shape] = model.shapes.map { $0.placed(scale: plan.scale, yaw: yaw, at: at) }
    return WorldAnt(id: plan.id, copy: m, distance: d, at: at, yaw: yaw, model: model, shapes: shapes,
                    feet: feet, stance: stance)
}

/// Every ant whose body lies within `range` of x at time t.
func antsInView(time t: Float, range: ClosedRange<Float>, mutant: Mutant) -> [WorldAnt] {
    var out: [WorldAnt] = []
    for plan in antPlans {
        for m in -3...3 {
            let d: Float = plan.distance(t, mutant: mutant)
            let x: Float = plan.pose(d, copy: m).at.x
            if x > range.lowerBound - 4.5 && x < range.upperBound + 4.5 {
                out.append(worldAnt(plan, copy: m, time: t, mutant: mutant))
            }
        }
    }
    return out
}

// MARK: - the marks on the trail

/// Where ant `plan`'s dab touches the ground, for copy 0: the bottom of the
/// gaster tip's round end.
func dabPoint(_ plan: AntPlan) -> SIMD3<Float> {
    let a: WorldAnt = worldAnt(plan, copy: 0, time: plan.dabTime, mutant: .none)
    let tip: Shape = a.shapes.filter { $0.part == .gaster && $0.kind == .roundCone }[0]
    return tip.b - SIMD3<Float>(0, tip.rb, 0)
}

/// A fresh mark: where, and how bright (0–1).
struct Mark {
    var point: SIMD3<Float>
    var strength: Float
}

let dabPoints: [SIMD3<Float>] = antPlans.map { dabPoint($0) }

/// The fresh marks at time t in the x range: each appears the moment the
/// gaster touches and fades. Periodic in the loop because the file is.
func marks(time t: Float, range: ClosedRange<Float>) -> [Mark] {
    var out: [Mark] = []
    for plan in antPlans {
        var age: Float = (t - (plan.dabTime - dabHold / 2)).truncatingRemainder(dividingBy: loopSeconds)
        if age < 0 { age += loopSeconds }
        let s: Float = (1 - exp(-age / 0.12)) * exp(-age / markFade)
        for m in -3...3 {
            let p: SIMD3<Float> = dabPoints[plan.id] + SIMD3<Float>(Float(m) * groupPeriod, 0, 0)
            if p.x > range.lowerBound - 1 && p.x < range.upperBound + 1 && s > 1e-4 {
                out.append(Mark(point: SIMD3<Float>(p.x, 0, p.z), strength: s))
            }
        }
    }
    return out
}
