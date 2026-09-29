// lib/ant/v1 — the pose API: where a SIMULATION says the ant is, and how far
// it has walked, in; the ant's shapes out. Nothing here touches the GPU.
//
// The gait is step 55's (step055_ant_trail_5/Sources/Walk.swift), which is
// step 44's; what changes is who drives it. In the steps the ants walked one
// scripted trail, `pose(x)`. Here the caller supplies the path: a function
// from distance walked to where the ant stood (and faced) when it had walked
// that far — a `Track` a simulation records as it goes, or a straight line.
// Everything else follows the lineage's first rule unchanged:
//
//   THE FEET ARE DRIVEN BY DISTANCE, NOT TIME (step 44). Each leg's phase is
//   the distance walked divided by the stride length. A foot in stance is put
//   down at a point fixed by where the ant was at touchdown, so it cannot
//   slide however the ant speeds up, stops or turns; and when the ant stops,
//   every leg stops with it.
//
// Time drives only the antennae's sweep. The dab (gaster pressed to the
// ground) is a state the caller sets; `dabStop` gives the lineage's timing.
//
// Units: MILLIMETRES and SECONDS OF REAL TIME. The world is y up with the
// ground at y = 0 (v1 is flat ground only). A ground position is (x, z). A
// heading is the angle that turns the ant's forward (+x) to (cos h, 0, sin h)
// and its right side (+z) to (−sin h, 0, cos h): h = atan2(dz, dx).
//
// Every number carries where it came from, or says MODEL and why.

import Foundation
import simd

extension AntV1 {

    // MARK: - speed (step 44, via 55)

    // Walking speed. Dussutour, Deneubourg & Fourcassié (2005, *J Exp Biol*
    // 208(15): 2903, doi 10.1242/jeb.01711, "Temporal organization of
    // bi-directional traffic in the ant Lasius niger") timed L. niger workers
    // crossing a 10 mm-wide bridge "between the two bottlenecks ... in the
    // absence of interactions with other ants": 2.96 ± 0.61 s. Between the
    // bottlenecks lie an entrance (15 mm), the central part (60 mm) and the
    // other entrance (15 mm): 90 mm. Colonies at 25 ± 1 °C. So the speed is
    // DERIVED — 90 mm / 2.96 s = 30 mm/s — and rests on our reading of their
    // bridge layout (had they timed only the central 60 mm, it would be
    // 20 mm/s). Ant walking speed rises steeply with temperature, so it is
    // labelled with its 25 °C (step 44).
    static let dussutourDistance: Float = 90.0
    static let dussutourTime: Float = 2.96
    /// mm per real second, ≈ 30.4.
    static let walkingSpeed: Float = dussutourDistance / dussutourTime
    static let speedTemperature: Int = 25

    // MARK: - the gait (step 44, via 55)

    // Ants walk with the alternating tripod: the fore and hind legs of one
    // side step together with the middle leg of the other, and the two
    // tripods alternate. Zollikofer (1994, *J Exp Biol* 192: 95–106,
    // "Stepping patterns in ants I") filmed workers of twelve species in
    // Cataglyphis, Formica, LASIUS and Myrmica: "the alternating tripod gait
    // prevails over a wide range of speeds", with "temporal rigidity of
    // tripod coordination".
    static let tripodA: Set<Int> = [0, 4, 2]    // right fore, left mid, right hind
    static let tripodB: Set<Int> = [3, 1, 5]    // left fore, right mid, left hind

    // Duty factor — the fraction of each stride a foot is down. Reinhardt &
    // Blickhan (2014, *J Exp Biol*, doi 10.1242/jeb.098426, wood ant Formica
    // polyctena, "Level locomotion in wood ants"): 0.62 for the middle legs
    // at 9.5–12.5 cm/s, falling below 0.5 only near 18.8 cm/s. Above 0.5 both
    // tripods share the ground briefly at each changeover. 0.6 here: MODEL,
    // Formica's figure borrowed for Lasius.
    static let dutyFactor: Float = 0.6

    /// The ant's length at scale 1, measured from the model (4.1 mm, inside
    /// the cited worker range).
    static let baseLength: Float = Model(body: bodyShapes(gasterBend: 0), legs: [], antennae: []).length

    /// Stride length, mm: step 44's, kept by steps 55–57 — the stride these
    /// legs were tested to reach. Step 44 derived it as its conveyor
    /// (3 × (ant + 3.5 mm gap)) over 9 strides: 2.53 mm, 0.62 body lengths,
    /// which at 30 mm/s is 12 strides a second, close to the 11.7 ± 0.4 Hz
    /// Reinhardt & Blickhan measured in Formica. For Lasius no stride table
    /// was found (step 44).
    static let strideConveyor: Float = 3 * (baseLength + 3.5)   // step 44's loop, mm
    static let strideLength: Float = strideConveyor / 9
    /// Real strides per second at `walkingSpeed`.
    static let stepFrequency: Float = walkingSpeed / strideLength

    /// How high a swinging foot lifts, mm. MODEL (step 44): small insects
    /// walk with low, skimming steps; about a fifth of a millimetre reads
    /// without looking like a march.
    static let footLift: Float = 0.18

    /// Where each foot sits at mid-stance, in the ant's frame (right side;
    /// the left mirrors). MODEL (step 44): step 26's relaxed standing feet,
    /// the fore foot drawn back and the hind foot out behind a little so a
    /// full stride stays within what femur and tibia can reach.
    static let stanceCentres: [SIMD3<Float>] = [
        SIMD3<Float>(1.75, 0, 1.35),
        SIMD3<Float>(0.26, 0, 1.60),
        SIMD3<Float>(-1.60, 0, 1.50),
    ]

    /// The tightest turn, mm of radius at the ant's origin, this gait holds.
    /// MEASURED on this module (lib/ant/v1, not biology): the lineage's feet
    /// were only ever asked to walk a nearly straight trail. At 5 mm, left or
    /// right, every leg keeps its lengths, nothing goes below the ground and
    /// the tripods stay whole (a test checks it); at 4 mm a knee dips
    /// 0.13 mm underground, at 3 mm a tibia stretches 0.02 mm. A simulation
    /// should keep an ant's path at least this straight.
    static let minimumTurnRadius: Float = 5

    // MARK: - the antennae's sweep (step 55)

    /// Antennal sweeps per REAL second. MODEL: no measured sweep rate was
    /// found (step 44). Step 44 drew 5–6 sweeps per 11.25 s loop, "7–8 a
    /// second in real time"; step 55 made it 0.45 per picture second at its
    /// ×15 slow-down, which is this: 0.45 × 15 = 6.75 a real second.
    static let sweepRate: Float = 0.45 * 15

    // MARK: - the dab (step 44, via 55)

    // Trail laying. L. niger lays its trail from the hindgut: "In Lasius
    // niger the trail pheromone was identified as 3,4-dihydro-8-hydroxy-
    // 3,5,7-trimethylisocoumarin (Bestmann et al. 1992)" — Bestmann, Kern,
    // Schäfer & Witschel, *Angew Chem Int Ed* 31: 795–796 — and "in all cases
    // where a formicine species' trail pheromone has been identified, it has
    // been located in the hindgut" (both as restated by Butterfield, Bacon &
    // Hill, *J Chem Ecol* 2025). It is laid in dots, not a stripe:
    // "Pheromone deposition is a very stereotyped behaviour in L. niger ...
    // It involves the ant pausing for ca. 0.2 seconds, backing up, and firmly
    // pressing the tip of their abdomen onto the substrate" (Czaczkes et al.
    // 2016, *PLoS ONE*, "The effect of trail pheromone and path confinement
    // on learning of complex routes in the ant Lasius niger", PMC4784821).
    // Here: a 0.2 s stop, the gaster bent down until its tip meets the
    // ground, held while the ant stands still. The "backing up" is left out
    // (the steps' loops only moved forward; a simulation may add it).
    static let dabPauseReal: Float = 0.2      // s, the whole slow-down and stop
    static let dabHoldReal: Float = 0.06      // s of it standing still, gaster down. MODEL
    static let dabBendReal: Float = 0.05      // s for the gaster to bend down or lift. MODEL
    /// Each side's ease from walking to standing, s.
    static let dabRampReal: Float = (dabPauseReal - dabHoldReal) / 2
    /// Walking time one dab costs, s: at `walkingSpeed` the ant ends this
    /// much × speed behind where it would have been.
    static let dabWalkingLost: Float = dabHoldReal + dabRampReal
    /// Between dabs the gaster is carried a little raised, clear of the
    /// ground. MODEL (step 44): step 26's resting gaster, lifted 6° about the
    /// petiole.
    static let walkingGasterLift: Float = -6 * Float.pi / 180
    /// The gaster's bend and tip flex at the moment its tip meets the ground.
    static let dabFull: (bend: Float, flex: Float) = dabPose()

    /// A flat-topped bump: 1 across the hold, easing to 0 over `ramp` each
    /// side (step 44).
    static func plateau(_ tau: Float, hold: Float, ramp: Float) -> Float {
        let a: Float = abs(tau)
        if a <= hold / 2 { return 1 }
        if a >= hold / 2 + ramp { return 0 }
        return 0.5 * (1 + cos(Float.pi * (a - hold / 2) / ramp))
    }

    /// ∫ plateau from −∞ to tau, exactly (step 44) — how much walking a
    /// stop has cost so far, for a caller that wants distance in closed form.
    static func plateauIntegral(_ tau: Float, hold: Float, ramp: Float) -> Float {
        let h: Float = hold / 2
        let rp: Float = ramp / Float.pi
        if tau <= -h - ramp { return 0 }
        if tau < -h {
            let u: Float = tau + h + ramp
            let arg: Float = Float.pi * u / ramp
            let wave: Float = rp * sin(arg)
            return 0.5 * (u - wave)
        }
        if tau <= h { return ramp / 2 + (tau + h) }
        if tau < h + ramp {
            let u: Float = tau - h
            let arg: Float = Float.pi * u / ramp
            let wave: Float = rp * sin(arg)
            let before: Float = ramp / 2 + hold
            return before + 0.5 * (u + wave)
        }
        return ramp + hold
    }

    /// One dab, as the lineage times it, `t` real seconds from the middle of
    /// the stop: the speed as a fraction of walking speed (0 while standing),
    /// and the dab amount to pass as `State.dab` (1 = tip on the ground, only
    /// while the ant stands still: 80% of the standing time, step 44).
    static func dabStop(_ t: Float) -> (speed: Float, dab: Float) {
        let stop: Float = plateau(t, hold: dabHoldReal, ramp: dabRampReal)
        let dab: Float = plateau(t, hold: dabHoldReal * 0.8, ramp: dabBendReal)
        return (1 - stop, dab)
    }

    /// Where to stop for a dab: the first distance at or after `s` at which
    /// the ant stands in the middle of a changeover, all six feet down (tripod
    /// A at phase (duty − ½)/2). Step 55 placed its dab sites so; an ant that
    /// stops elsewhere freezes its swinging feet in the air for the stop.
    static func changeoverDistance(atOrAfter s: Float, gaitOffset: Float = 0) -> Float {
        let want: Float = (dutyFactor - 0.5) / 2
        let cycles: Float = s / strideLength + gaitOffset - want
        let k: Float = cycles.rounded(.up)
        return (k + want - gaitOffset) * strideLength
    }

    // MARK: - the pose API

    /// Where an ant stands on the ground: (x, z) in mm, and its heading.
    struct Ground: Equatable {
        /// (x, z) on the ground, mm.
        var position: SIMD2<Float>
        var heading: Float
        init(_ position: SIMD2<Float>, heading: Float) {
            self.position = position
            self.heading = heading
        }
    }

    /// The optional state a simulation may set. Defaults are the lineage's
    /// walking worker.
    struct State: Equatable {
        /// 0 walking (gaster carried raised), 1 gaster tip pressed to the
        /// ground; in between, part-way (step 44's bend). See `dabStop`.
        var dab: Float = 0
        /// Antenna sweep offset, in sweeps — so neighbours need not sweep in
        /// step (step 55's ants were a fixed fraction of a sweep apart).
        var sweepPhase: Float = 0
        /// Leg phase offset, in strides, for both tripods together.
        var gaitOffset: Float = 0
        /// Size relative to step 32's 4.1 mm worker (step 44 used 0.96–1.04).
        /// Uniform, so distances stay honest; the stride stays step 44's mm.
        var scale: Float = 1
        init(dab: Float = 0, sweepPhase: Float = 0, gaitOffset: Float = 0, scale: Float = 1) {
            self.dab = dab
            self.sweepPhase = sweepPhase
            self.gaitOffset = gaitOffset
            self.scale = scale
        }
    }

    /// One ant, posed and placed in the world.
    struct Posed {
        let distance: Float              // distance walked, mm
        let time: Float                  // real seconds
        let at: SIMD3<Float>             // the ant's origin, on the ground
        let yaw: Float                   // its heading
        let scale: Float
        let model: Model                 // in the ant's own frame
        let shapes: [Shape]              // in the world: what the renderer draws
        let feet: [SIMD3<Float>]         // lowest point of each foot, world, legs 0–5
        let stance: [Bool]               // which feet the gait has down

        /// Every shape's lowest point.
        var lowest: Float { shapes.map { $0.extent(along: SIMD3(0, 1, 0)).lo }.min() ?? 0 }
        /// A sphere holding the whole ant (step 44's: centred 0.6 mm up).
        var bound: (centre: SIMD3<Float>, radius: Float) {
            let c: SIMD3<Float> = at + SIMD3<Float>(0, 0.6 * scale, 0)
            let r: Float = shapes.map { simd_distance($0.bound.centre, c) + $0.bound.radius }.max() ?? 0
            return (c, r)
        }
    }

    /// Body +x → (cos, 0, sin), body +z → (−sin, 0, cos).
    static func turn(_ v: SIMD3<Float>, _ yaw: Float) -> SIMD3<Float> {
        let c: Float = cos(yaw)
        let s: Float = sin(yaw)
        return SIMD3<Float>(v.x * c - v.z * s, v.y, v.x * s + v.z * c)
    }

    static func unturn(_ v: SIMD3<Float>, _ yaw: Float) -> SIMD3<Float> { turn(v, -yaw) }

    /// Leg `j`'s phase offset in strides: tripod B half a stride behind
    /// tripod A (step 44).
    static func legOffset(_ j: Int, gaitOffset: Float, mutant: Mutant) -> Float {
        if mutant == .nonTripod { return gaitOffset }
        return gaitOffset + (tripodA.contains(j) ? 0 : 0.5)
    }

    /// Leg j's foot at mid-stance, in the ant's frame, shifted so the stance
    /// straddles it (step 44).
    static func stanceLocal(_ j: Int, scale: Float) -> SIMD3<Float> {
        let side: Float = j < 3 ? 1 : -1
        var local: SIMD3<Float> = stanceCentres[j % 3] * SIMD3<Float>(1, 1, side)
        let halfStance: Float = dutyFactor * strideLength / 2
        local.x += halfStance / scale
        return local
    }

    static func place(_ g: Ground) -> SIMD3<Float> { SIMD3<Float>(g.position.x, 0, g.position.y) }

    /// Where leg `j`'s foot lands for the stance that begins when the ant has
    /// walked `s`.
    static func touchdown(_ j: Int, at s: Float, path: (Float) -> Ground, scale: Float) -> SIMD3<Float> {
        let g: Ground = path(s)
        return place(g) + turn(stanceLocal(j, scale: scale), g.heading) * scale
    }

    /// Leg `j`'s foot in the world when the ant has walked `x`, and whether
    /// it is down (step 44's `foot`, the trail replaced by `path`).
    static func foot(_ j: Int, distance x: Float, path: (Float) -> Ground, state: State,
                     mutant: Mutant) -> (point: SIMD3<Float>, stance: Bool) {
        let off: Float = legOffset(j, gaitOffset: state.gaitOffset, mutant: mutant)
        let cycles: Float = x / strideLength + off
        let k: Float = cycles.rounded(.down)
        let phase: Float = cycles - k
        let start: Float = (k - off) * strideLength       // distance walked at this stride's touchdown
        if phase < dutyFactor {
            if mutant == .sliding {
                // Planted in the ANT's frame, so it rides along with the body.
                let g: Ground = path(x)
                return (place(g) + turn(stanceLocal(j, scale: state.scale), g.heading) * state.scale, true)
            }
            return (touchdown(j, at: start, path: path, scale: state.scale), true)
        }
        let u: Float = (phase - dutyFactor) / (1 - dutyFactor)
        let from: SIMD3<Float> = touchdown(j, at: start, path: path, scale: state.scale)
        let to: SIMD3<Float> = touchdown(j, at: start + strideLength, path: path, scale: state.scale)
        let ease: Float = 3 - 2 * u
        let e: Float = u * u * ease
        let travel: SIMD3<Float> = to - from
        var p: SIMD3<Float> = from + travel * e
        let lift: Float = footLift * state.scale
        p.y = lift * sin(Float.pi * u)
        return (p, false)
    }

    /// The antenna tips sweep side to side just above the ground ahead of the
    /// head, the two moving together — one crossing inward as the other
    /// swings out (step 44). Returns scape direction and tip, ant's frame.
    static func antennaPose(_ index: Int, time t: Float, sweepPhase: Float) -> (scape: SIMD3<Float>, tip: SIMD3<Float>) {
        let sgn: Float = index == 0 ? 1 : -1
        let turns: Float = sweepRate * t + sweepPhase
        let sweep: Float = sin(2 * Float.pi * turns)
        let rest: Float = sgn * 0.28
        let tipZ: Float = rest + 0.26 * sweep
        // The scape swings forward and out, turning with the sweep. MODEL pose.
        let swing: Float = 0.8 * (tipZ - rest)
        let scape: SIMD3<Float> = simd_normalize(SIMD3<Float>(0.75, 0.02, sgn * 0.66 + swing))
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

    /// The gaster's bend and its tip's flex for a dab amount 0–1 (step 44).
    static func gasterPose(dab: Float) -> (bend: Float, flex: Float) {
        let k: Float = min(max(dab, 0), 1)
        let bend: Float = walkingGasterLift + (dabFull.bend - walkingGasterLift) * k
        return (bend, dabFull.flex * k)
    }

    /// THE POSE. The ant that has walked `distance` mm along `path` (distance
    /// walked → where it stood then), at `time` real seconds, in `state`.
    /// Deterministic: the same arguments give the same shapes, bit for bit.
    /// Stance feet are fixed by `path` at their touchdown distance, so they
    /// are planted exactly as long as `path` does not change its past.
    static func pose(distance: Float, time: Float, path: (Float) -> Ground,
                     state: State = State(), mutant: Mutant = .none) -> Posed {
        let g: Ground = path(distance)
        let at: SIMD3<Float> = place(g)
        let yaw: Float = g.heading
        var legs: [Leg] = []
        var feet: [SIMD3<Float>] = []
        var stance: [Bool] = []
        for j in 0..<6 {
            let (p, down) = foot(j, distance: distance, path: path, state: state, mutant: mutant)
            let local: SIMD3<Float> = unturn(p - at, yaw) / state.scale
            legs.append(buildLeg(index: j, foot: local))
            feet.append(p)
            stance.append(down)
        }
        var antennae: [Antenna] = []
        for i in 0..<2 {
            let (scape, tip) = antennaPose(i, time: time, sweepPhase: state.sweepPhase)
            antennae.append(buildAntenna(index: i, scapeDir: scape, tip: tip, mutant: mutant))
        }
        let (bend, flex) = gasterPose(dab: state.dab)
        let model = Model(body: bodyShapes(gasterBend: bend, tipFlex: flex), legs: legs, antennae: antennae)
        let shapes: [Shape] = model.shapes.map { $0.placed(scale: state.scale, yaw: yaw, at: at) }
        return Posed(distance: distance, time: time, at: at, yaw: yaw, scale: state.scale, model: model,
                     shapes: shapes, feet: feet, stance: stance)
    }

    /// The pose for an ant that has walked in a STRAIGHT line along its
    /// heading to reach `position` — the simplest call. For an ant that
    /// turns, record a `Track` and use `pose(distance:time:path:)`, or its
    /// stance feet will swing round with the heading.
    static func pose(at position: SIMD2<Float>, heading: Float, distance: Float, time: Float,
                     state: State = State(), mutant: Mutant = .none) -> Posed {
        let line = straightPath(through: position, heading: heading, atDistance: distance)
        return pose(distance: distance, time: time, path: line, state: state, mutant: mutant)
    }

    /// A straight path: at distance `d0` the ant is at `p0`, facing `heading`.
    static func straightPath(through p0: SIMD2<Float>, heading: Float, atDistance d0: Float) -> (Float) -> Ground {
        let dir = SIMD2<Float>(cos(heading), sin(heading))
        return { s in Ground(p0 + dir * (s - d0), heading: heading) }
    }

    // MARK: - a recorded path

    /// Where an ant has been, recorded by distance walked, for `pose(path:)`.
    /// Append as the simulation runs (distances must not decrease). Between
    /// samples, position and heading are interpolated linearly (heading the
    /// short way round); before the first and after the last, the path runs
    /// straight on along the end heading.
    ///
    /// A swinging foot aims at where it will land one stride on, so a frame
    /// is final once the track reaches `distance + strideLength`
    /// (`isSettled`); rendered earlier, a swing may still change course,
    /// though no stance foot ever moves.
    struct Track {
        private(set) var distances: [Float] = []
        private(set) var grounds: [Ground] = []

        init() {}

        /// Record that at `distance` walked the ant stands at `ground`.
        /// Returns false, recording nothing, unless `distance` is beyond the
        /// last one: the past is fixed, since planted feet hang on it. (So an
        /// ant standing still — a dab — just stops appending; and v1 cannot
        /// turn on the spot, because distance, not time, drives the body.)
        @discardableResult
        mutating func append(distance: Float, _ ground: Ground) -> Bool {
            if let last = distances.last, distance <= last { return false }
            distances.append(distance)
            grounds.append(ground)
            return true
        }

        /// The last distance recorded.
        var reach: Float { distances.last ?? 0 }

        /// Is every foot's course final for an ant at `distance`?
        func isSettled(_ distance: Float) -> Bool { reach >= distance + AntV1.strideLength }

        /// Where the ant stood when it had walked `s`.
        func ground(_ s: Float) -> Ground {
            guard let first = grounds.first, let last = grounds.last else {
                return Ground(SIMD2<Float>(0, 0), heading: 0)
            }
            if s <= distances[0] {
                let dir = SIMD2<Float>(cos(first.heading), sin(first.heading))
                return Ground(first.position + dir * (s - distances[0]), heading: first.heading)
            }
            let n: Int = distances.count
            if s >= distances[n - 1] {
                let dir = SIMD2<Float>(cos(last.heading), sin(last.heading))
                return Ground(last.position + dir * (s - distances[n - 1]), heading: last.heading)
            }
            // Bisection for the sample interval holding s.
            var lo: Int = 0
            var hi: Int = n - 1
            while hi - lo > 1 {
                let mid: Int = (lo + hi) / 2
                if distances[mid] <= s { lo = mid } else { hi = mid }
            }
            let span: Float = distances[hi] - distances[lo]
            let f: Float = span > 0 ? (s - distances[lo]) / span : 0
            let a: Ground = grounds[lo]
            let b: Ground = grounds[hi]
            let step: SIMD2<Float> = b.position - a.position
            let p: SIMD2<Float> = a.position + step * f
            var dh: Float = (b.heading - a.heading).truncatingRemainder(dividingBy: 2 * Float.pi)
            if dh > Float.pi { dh -= 2 * Float.pi }
            if dh < -Float.pi { dh += 2 * Float.pi }
            return Ground(p, heading: a.heading + dh * f)
        }
    }

    /// The pose of an ant along a recorded track.
    static func pose(distance: Float, time: Float, track: Track,
                     state: State = State(), mutant: Mutant = .none) -> Posed {
        pose(distance: distance, time: time, path: { track.ground($0) }, state: state, mutant: mutant)
    }
}
