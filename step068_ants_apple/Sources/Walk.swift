// Three workers visit the apple: each walks in along its own lane, stops
// with its mandibles just off the wet face, taps the juice with both
// antennae (tasting), pushes its glossa out to the juice (drinking), turns
// round on the spot and walks off the way it came. Where each one is, how its
// legs step, where its antennae and glossa are — as functions of time.
// Nothing here touches the GPU.
//
// The world: MILLIMETRES, y up, the card is y = 0. The drinking face of the
// apple stands on the card along `faceTangent`, facing `faceNormal`
// (Food.swift).
//
// Step 44's rules, copied:
//
//   1. THE FEET ARE DRIVEN BY DISTANCE, NOT TIME. Each leg's phase is the
//      "gait distance" its ant has gone divided by the stride. Walking, that
//      is the distance walked; turning on the spot, it is the arc the turn
//      would walk at `turnRadius` — so a turning ant steps as a walking one
//      does. A foot in stance is put down at a point fixed at touchdown, so
//      it cannot slide, and when the ant stops every leg stops with it.
//   2. THE LOOP IS FORWARD. Each ant makes one trip a loop, starting and
//      ending out of the frame; between trips it is not drawn — it is off
//      the card we see — so nothing appears or vanishes in view.
//   3. THE CAMERA DOES NOT MOVE.
//
// Every number carries where it came from, or says MODEL and why.

import Foundation
import simd

// MARK: - speed

// Walking speed: step 44's, derived from Dussutour, Deneubourg & Fourcassié
// (2005, *J Exp Biol* 208: 2903, doi 10.1242/jeb.01711): L. niger workers
// crossed a 90 mm bridge "in the absence of interactions with other ants" in
// 2.96 ± 0.61 s at 25 ± 1 °C: 30 mm/s. That was on a trail; near food ants may
// well slow down, but no measured approach speed was found, so the trail
// speed is used and the caption gives its temperature.
let dussutourDistance: Float = 90.0
let dussutourTime: Float = 2.96
let realSpeed: Float = dussutourDistance / dussutourTime    // mm/s, ≈ 30.4
let speedTemperature: Int = 25
/// On screen, slowed by the loop's one slow-down (Motion.swift).
let pictureSpeed: Float = realSpeed / slowdown              // mm/s, ≈ 3.04

// MARK: - the gait (step 44's)

// Alternating tripod: Zollikofer (1994, *J Exp Biol* 192: 95–106) filmed
// Lasius among twelve species: "the alternating tripod gait prevails over a
// wide range of speeds". Step 44's.
let tripodA: Set<Int> = [0, 4, 2]    // right fore, left mid, right hind
let tripodB: Set<Int> = [3, 1, 5]    // left fore, right mid, left hind

/// Duty factor: step 44's 0.6, MODEL (Formica's 0.62, Reinhardt & Blickhan 2014).
let dutyFactor: Float = 0.6

/// Stride, mm. MODEL: 2.2 mm, a little under step 44's 2.5 (which was
/// derived from that step's loop, not measured). Here the fore feet must stay
/// clear of the apple while the ant stands with its mandibles 0.06 mm from
/// it, and a shorter stride keeps a planted fore foot back. At 30 mm/s it is
/// 14 strides a second — step 44 had 12, Formica 11.7 (Reinhardt & Blickhan).
let strideLength: Float = 2.2
let stepFrequency: Float = realSpeed / strideLength   // real strides per second

/// How high a swinging foot lifts, mm. Step 44's MODEL.
let footLift: Float = 0.18

/// Where each foot sits at mid-stance, in the ant's frame (right side; the
/// left mirrors). Step 44's, with the fore foot drawn 0.5 mm further back:
/// MODEL, so that a fore foot just put down, half a stance ahead, is still
/// behind the mandibles (a test checks every foot against the apple).
let stanceCentres: [SIMD3<Float>] = [
    SIMD3(1.25, 0, 1.40),
    SIMD3(0.26, 0, 1.60),
    SIMD3(-1.60, 0, 1.50),
]

/// Between steps the gaster is carried a little raised. Step 44's MODEL.
let walkingGasterLift: Float = -6 * Float.pi / 180

/// Small differences between the three, so they are not copies. MODEL, all
/// inside the cited worker range of 3.4–5.0 mm. Step 44's.
let antScales: [Float] = [1.00, 0.96, 1.03]

// MARK: - the visit

/// Where along the face each ant drinks, mm from the face's middle along
/// `faceTangent` (towards the camera and right). MODEL: three places 4.6 mm
/// apart, room for an ant to turn round beside a standing neighbour (a test
/// checks no two ants ever touch). Ant 0 — the one the micrometre inset rides
/// on — takes the front place, nearest the camera, where no other ant stands
/// between it and the view.
let laneOffsets: [Float] = [4.6, 0.0, -4.6]
/// Which way each turns round: +1 turns its head towards +tangent. MODEL:
/// each turns so that it swings clear of any neighbour standing then (a test checks).
let turnSigns: [Float] = [-1, 1, -1]

/// How far each walks in from its start to its place, and out again, mm:
/// far enough to start and end out of the frame (a test checks). MODEL.
let walkIn: Float = 20.0
/// The mandibles' gap to the juice while the ant stands there, mm. MODEL:
/// close, not touching — the glossa reaches the rest of the way.
let mandibleGap: Float = 0.06
/// Turning, the ant pivots about a point this far behind its origin, mm
/// (scale 1). MODEL: far enough back that the gaster's tip swings on a
/// smaller circle than the gap from the pivot to the apple, and the head
/// only draws away from it (a test checks no part enters the apple).
let turnPivotBack: Float = 0.6
/// Turning on the spot, the gait distance is the arc at this radius. MODEL:
/// chosen so the body turns no more than about 20° while a foot is planted,
/// which the legs can follow (a test checks no leg is ever stretched).
let turnRadius: Float = 3.0

/// Touches per visit: both antennae tap together, three times; the glossa
/// goes out after the first touch and stays out until the last lifts. MODEL:
/// taste first, then drink. How long a real Lasius niger drinks is set by
/// what it wants to carry — scouts at a big drop drink until their own
/// "desired volume" (Mailleux, Deneubourg & Detrain, *Anim Behav* 59:
/// 1061–1069, 2000) — far longer than the 0.4 s real time shown: the drink
/// is cut short, and the caption says so.
let touchesPerVisit: Int = 3
/// Tap phases (Motion.swift) of the glossa: out between these, full between
/// the middle two, in by the last. The stand begins at phase 0.7, the lift's
/// top, and ends at 0.7 + touches.
let glossaOut: (start: Float, full: Float, back: Float, stowed: Float) = (1.45, 1.80, 3.20, 3.55)
/// The stand's first and last phase.
let standStartPhase: Float = 0.7
var standPhases: Float { Float(touchesPerVisit) }
/// How long the ant takes to slow to its stand and to set off again,
/// picture seconds. MODEL.
let rampSeconds: Float = 0.8

/// Antennal sweeps while walking, picture seconds per sweep. MODEL: step
/// 44's rate (5–6 a loop of 11.25 s, one every 2 s) slowed a little to one
/// every 2.5 s — 4 a second in real time, the tap's rate — so ten fit the loop.
let sweepSeconds: Float = 2.5

// MARK: - one ant's trip

/// Body +x → (cos, 0, sin), body +z → (−sin, 0, cos).
func turn(_ v: SIMD3<Float>, _ yaw: Float) -> SIMD3<Float> {
    let c: Float = cos(yaw)
    let s: Float = sin(yaw)
    return SIMD3<Float>(v.x * c - v.z * s, v.y, v.x * s + v.z * c)
}

func unturn(_ v: SIMD3<Float>, _ yaw: Float) -> SIMD3<Float> { turn(v, -yaw) }

/// A flat-topped bump: 1 across the hold, easing to 0 over `ramp` each side
/// (step 44's).
func plateau(_ tau: Float, hold: Float, ramp: Float) -> Float {
    let a: Float = abs(tau)
    if a <= hold / 2 { return 1 }
    if a >= hold / 2 + ramp { return 0 }
    return 0.5 * (1 + cos(Float.pi * (a - hold / 2) / ramp))
}

/// ∫ plateau from −∞ to tau, exactly (step 44's).
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

/// The mandibles' reach ahead of the ant's origin at scale 1, mm, from the
/// model's own shapes.
let mandibleReach: Float = {
    var best: Float = 0
    for m in bodyShapes(gasterBend: 0) where m.part == .mandible {
        let reach: Float = m.extent(along: SIMD3<Float>(1, 0, 0)).hi
        best = max(best, reach)
    }
    return best
}()

/// Which part of its trip an ant is in.
enum Stage: Int { case walkingIn = 0, standing = 1, turning = 2, walkingOut = 3, away = 4 }

/// One ant's trip, and the things that make it itself.
struct Trip {
    let id: Int
    let scale: Float
    let heading: SIMD3<Float>       // unit, walking in: straight at the face
    let yaw0: Float
    let spot: SIMD3<Float>          // the ant's origin while it stands
    let start: SIMD3<Float>
    let turnSign: Float
    let turnLength: Float           // gait distance of the half turn
    let holdSeconds: Float          // standing still
    let holdStart: Float            // trip time the stand begins
    let endTime: Float              // trip time it is out of view again
    let offset: Float               // loop time its trip starts
    let gaitOffset: Float           // tripod A's phase at gait distance 0

    init(id: Int, offset: Float) {
        self.id = id
        self.offset = offset
        scale = antScales[id]
        heading = -faceNormal
        yaw0 = atan2(heading.z, heading.x)
        let facePoint: SIMD3<Float> = faceBase + faceTangent * laneOffsets[id]
        spot = facePoint + faceNormal * ((mandibleReach + mandibleGap) * antScales[id])
        start = spot - heading * walkIn
        turnSign = turnSigns[id]
        turnLength = Float.pi * turnRadius
        holdSeconds = Float(touchesPerVisit) * tapSeconds
        // Cruise at pictureSpeed; the stand is a plateau of stopping whose
        // hold starts exactly when the ant has walked `walkIn`.
        holdStart = walkIn / pictureSpeed + rampSeconds / 2
        let total: Float = walkIn + turnLength + walkIn
        endTime = total / pictureSpeed + rampSeconds + holdSeconds
        // Stand in the middle of a changeover, when both tripods are down:
        // phase 0 to (duty − ½) for tripod A, so (duty − ½)/2 is its middle.
        let want: Float = (dutyFactor - 0.5) / 2
        var o: Float = (want - walkIn / strideLength).truncatingRemainder(dividingBy: 1)
        if o < 0 { o += 1 }
        gaitOffset = o
    }

    /// Trip time at loop time t (the rewind mutant through effectiveTime).
    func tripTime(_ t: Float, mutant: Mutant) -> Float {
        var x: Float = (effectiveTime(t, mutant: mutant) - offset).truncatingRemainder(dividingBy: loopSeconds)
        if x < 0 { x += loopSeconds }
        return x
    }

    /// Gait distance at trip time τ: cruise, less the time lost standing.
    func gaitDistance(_ tau: Float) -> Float {
        let centre: Float = holdStart + holdSeconds / 2
        let lost: Float = plateauIntegral(tau - centre, hold: holdSeconds, ramp: rampSeconds)
        return pictureSpeed * (tau - lost)
    }

    func stage(_ tau: Float) -> Stage {
        if tau > endTime { return .away }
        let s: Float = gaitDistance(tau)
        if tau >= holdStart && tau <= holdStart + holdSeconds { return .standing }
        if s < walkIn { return .walkingIn }
        if s < walkIn + turnLength { return .turning }
        return .walkingOut
    }

    /// Tap phase at trip time τ, while standing (Motion.swift's phases:
    /// touching from each whole number to +0.4).
    func tapPhase(_ tau: Float) -> Float {
        standStartPhase + (tau - holdStart) / tapSeconds
    }

    /// Where the ant is and which way it faces after gait distance s.
    /// Walking in: straight. Turning: about a point `turnPivotBack` behind
    /// its origin, its heading easing round 180° (so it turns from rest and
    /// ends turning slowly). Walking out: back along its lane, easing into
    /// its stride over the first 1.2 mm.
    func pose(_ s: Float) -> (at: SIMD3<Float>, yaw: Float) {
        if s <= walkIn { return (start + heading * s, yaw0) }
        let pivot: SIMD3<Float> = spot - heading * (turnPivotBack * scale)
        if s <= walkIn + turnLength {
            let u: Float = (s - walkIn) / turnLength
            let a: Float = turnSign * Float.pi * smoothstep01(u)
            return (pivot + turn(spot - pivot, a), yaw0 + a)
        }
        let x: Float = s - walkIn - turnLength
        let ease: Float = 1.2
        let easeLength: Float = 2 * ease
        let easeScale: Float = 4 * ease
        let easing: Float = x * x / easeScale
        let along: Float = x < easeLength ? easing : x - ease
        let turned: SIMD3<Float> = pivot - (spot - pivot)
        return (turned - heading * along, yaw0 + turnSign * Float.pi)
    }

    /// Leg `j`'s phase offset: tripod B half a stride behind tripod A.
    func legOffset(_ j: Int) -> Float { gaitOffset + (tripodA.contains(j) ? 0 : 0.5) }

    /// Where leg `j`'s foot is put down for the stance that begins at gait
    /// distance `d`: under its stance centre as the body will stand at
    /// mid-stance. (Walking straight, this is step 44's formula exactly.)
    func touchdown(_ j: Int, at d: Float) -> SIMD3<Float> {
        let side: Float = j < 3 ? 1 : -1
        let local: SIMD3<Float> = stanceCentres[j % 3] * SIMD3<Float>(1, 1, side)
        let halfStance: Float = dutyFactor * strideLength / 2
        let (at, yaw) = pose(d + halfStance)
        return at + turn(local, yaw) * scale
    }

    /// Leg `j`'s foot in the world at gait distance `d`, and whether it is down.
    func foot(_ j: Int, at d: Float, mutant: Mutant) -> (point: SIMD3<Float>, stance: Bool) {
        let off: Float = legOffset(j)
        let cycles: Float = d / strideLength + off
        let k: Float = cycles.rounded(.down)
        let phase: Float = cycles - k
        let start: Float = (k - off) * strideLength
        if phase < dutyFactor {
            if mutant == .sliding {
                // Planted in the ANT's frame, so it rides along with the body.
                let side: Float = j < 3 ? 1 : -1
                var local: SIMD3<Float> = stanceCentres[j % 3] * SIMD3<Float>(1, 1, side)
                local.x += (dutyFactor / 2 - phase) * strideLength / scale
                let (at, yaw) = pose(d)
                return (at + turn(local, yaw) * scale, true)
            }
            return (touchdown(j, at: start), true)
        }
        let u: Float = (phase - dutyFactor) / (1 - dutyFactor)
        let from: SIMD3<Float> = touchdown(j, at: start)
        let to: SIMD3<Float> = touchdown(j, at: start + strideLength)
        let rise: Float = 3 - 2 * u
        let e: Float = u * u * rise
        let step: SIMD3<Float> = (to - from) * e
        var p: SIMD3<Float> = from + step
        p.y = footLift * scale * sin(Float.pi * u)
        return (p, false)
    }

    /// Where on the face each antenna touches, world: the juice's surface at
    /// the antenna's side of the head. MODEL: 1.0 mm either side of the
    /// ant's midline, 0.5 mm up — where the raised scapes let the funiculus
    /// reach down with a gentle bow.
    func antennaContact(_ index: Int) -> SIMD3<Float> {
        let sgn: Float = index == 0 ? 1 : -1
        let right: SIMD3<Float> = turn(SIMD3<Float>(0, 0, 1), yaw0)
        let onFace: SIMD3<Float> = faceBase + faceTangent * laneOffsets[id]
        let side: Float = sgn * scale
        let up: Float = 0.5 * scale
        return onFace + right * side + SIMD3<Float>(0, up, 0)
    }

    /// How far the glossa must reach from the mouth for its tip to rest on
    /// the juice, mm (in the ant's own scale-1 units), solved from the ant's
    /// place and the face — not typed.
    var glossaReach: Float {
        let mouth: SIMD3<Float> = spot + turn(glossaMouth, yaw0) * scale
        let dir: SIMD3<Float> = turn(glossaDirection, yaw0)
        let gap: Float = simd_dot(mouth - faceBase, faceNormal)            // mouth to the juice's plane
        let closing: Float = -simd_dot(dir, faceNormal)
        return (gap - glossaRadii.tip * scale) / (closing * scale)
    }
}

/// The three trips, a third of a loop apart. Ant 0's first touch-down is at
/// loop time 0 — the frame the stills and the first tests look at.
let trips: [Trip] = {
    let probe = Trip(id: 0, offset: 0)
    // First touch-down: phase 1.0, 0.3 taps after the stand begins.
    let firstTouch: Float = probe.holdStart + (1 - standStartPhase) * tapSeconds
    let third: Float = loopSeconds / 3
    return (0..<3).map { (k: Int) -> Trip in
        let after: Float = Float(k) * third
        return Trip(id: k, offset: after - firstTouch)
    }
}()

// MARK: - the antennae

/// Step 44's sweep, walking: the tips sweep side to side just above the
/// card ahead of the head. Returns scape direction, tip and bow, ant frame.
func sweepPose(_ index: Int, time t: Float) -> (scape: SIMD3<Float>, tip: SIMD3<Float>, bow: SIMD3<Float>) {
    let sgn: Float = index == 0 ? 1 : -1
    let w: Float = 2 * Float.pi / sweepSeconds
    let sweep: Float = sin(w * t)
    let rest: Float = sgn * 0.28
    let tipZ: Float = rest + 0.26 * sweep
    let scape: SIMD3<Float> = simd_normalize(SIMD3<Float>(0.75, 0.02, sgn * 0.66 + 0.8 * (tipZ - rest)))
    let elbow: SIMD3<Float> = antennaSocket(side: sgn) + scape * scapeLength
    let funiculus: Float = funiculusLengths(count: workerAntennaSegments - 1).reduce(0, +)
    let y: Float = funiculusTipRadius + 0.03
    let chord: Float = 0.94 * funiculus
    let dy: Float = elbow.y - y
    let dz: Float = elbow.z - tipZ
    let dx: Float = max(chord * chord - dy * dy - dz * dz, 0).squareRoot()
    return (scape, SIMD3<Float>(elbow.x + dx, y, tipZ), SIMD3<Float>(0.4, 1, 0))
}

/// Standing at the face: the scapes raised and a little forward and out,
/// the funiculus bowing back and out, away from the face. MODEL pose.
func faceScape(_ sgn: Float) -> SIMD3<Float> { simd_normalize(SIMD3<Float>(0.10, 0.80, sgn * 0.60)) }
func faceBow(_ sgn: Float) -> SIMD3<Float> { SIMD3<Float>(-0.8, 0.4, sgn * 0.45) }

/// Turning away: both antennae held up and back, clear of the face. MODEL.
func raisedPose(_ index: Int) -> (scape: SIMD3<Float>, tip: SIMD3<Float>, bow: SIMD3<Float>) {
    let sgn: Float = index == 0 ? 1 : -1
    let scape: SIMD3<Float> = faceScape(sgn)
    let elbow: SIMD3<Float> = antennaSocket(side: sgn) + scape * scapeLength
    return (scape, elbow + SIMD3<Float>(-0.45, 0.45, sgn * 0.50), SIMD3<Float>(-0.6, 1, sgn * 0.4))
}

func blendPose(_ a: (scape: SIMD3<Float>, tip: SIMD3<Float>, bow: SIMD3<Float>),
               _ b: (scape: SIMD3<Float>, tip: SIMD3<Float>, bow: SIMD3<Float>),
               _ w: Float) -> (scape: SIMD3<Float>, tip: SIMD3<Float>, bow: SIMD3<Float>) {
    let ds: SIMD3<Float> = (b.scape - a.scape) * w
    let s: SIMD3<Float> = simd_normalize(a.scape + ds)
    let dt: SIMD3<Float> = (b.tip - a.tip) * w
    let db: SIMD3<Float> = (b.bow - a.bow) * w
    return (s, a.tip + dt, a.bow + db)
}

// MARK: - one ant at one moment

/// One ant, posed and placed in the world.
struct WorldAnt {
    let id: Int
    let tau: Float                   // trip time
    let stage: Stage
    let distance: Float              // gait distance
    let at: SIMD3<Float>
    let yaw: Float
    let model: AntModel              // in the ant's own frame
    let shapes: [Shape]              // in the world
    let feet: [SIMD3<Float>]         // lowest point of each foot, world
    let stance: [Bool]               // what the timetable says, for comparison
    let tapPhase: Float              // standing: the tap's phase; otherwise 0
    let lift: Float                  // standing: the tips' height off the juice; otherwise −1
    let glossa: Float                // how far the glossa is out, mm (scale 1)
    let antennaTips: [SIMD3<Float>]  // world centres of the two tip segments' round ends
    /// The right antenna's apical segment in the world (a, b = tip centre).
    let apical: (a: SIMD3<Float>, b: SIMD3<Float>)

    var bound: (centre: SIMD3<Float>, radius: Float) {
        let c: SIMD3<Float> = at + SIMD3<Float>(0, 0.6, 0)
        let r: Float = shapes.map { simd_distance($0.bound.centre, c) + $0.bound.radius }.max() ?? 0
        return (c, r)
    }
    var touching: Bool { stage == .standing && lift == 0 }
}

func worldAnt(_ trip: Trip, time t: Float, mutant: Mutant) -> WorldAnt {
    let tau: Float = trip.tripTime(t, mutant: mutant)
    let stage: Stage = trip.stage(tau)
    let d: Float = trip.gaitDistance(min(tau, trip.endTime))
    let (at, yaw) = trip.pose(d)
    var legs: [Leg] = []
    var feet: [SIMD3<Float>] = []
    var stance: [Bool] = []
    for j in 0..<6 {
        let (p, down) = trip.foot(j, at: d, mutant: mutant)
        let local: SIMD3<Float> = unturn(p - at, yaw) / trip.scale
        legs.append(buildLeg(index: j, foot: local))
        feet.append(p)
        stance.append(down)
    }
    // The antennae: sweeping while walking in; blending to the face as the
    // ant closes the last 2.5 mm; tapping while it stands; raised as it turns
    // away; sweeping again once turned.
    let lift0: Float = liftHeight(mutant)
    var phase: Float = 0
    var lift: Float = -1
    var glossa: Float = 0
    var antennae: [Antenna] = []
    var tips: [SIMD3<Float>] = []
    var apical: (a: SIMD3<Float>, b: SIMD3<Float>) = (.zero, .zero)
    if stage == .standing {
        phase = trip.tapPhase(tau)
        lift = liftAt(phase: phase, height: lift0)
        glossa = glossaAt(phase: phase, trip: trip, mutant: mutant)
    }
    for i in 0..<2 {
        let sgn: Float = i == 0 ? 1 : -1
        let sweep = sweepPose(i, time: effectiveTime(t, mutant: mutant) + Float(trip.id) * 0.7)
        // The face pose, its tip placed from the world contact along the
        // juice's normal: touching, lifted, or held at the lift's top.
        let hover: Float = mutant == .hover ? 0.1 : 0
        let standingLift: Float = stage == .standing ? lift : lift0
        let press: Float = (mutant == .press && standingLift == 0) ? -0.02 : 0
        let tipWorld: SIMD3<Float> = trip.antennaContact(i) + faceNormal * ((funiculusTipRadius + standingLift + hover + press) * trip.scale)
        let tipLocal: SIMD3<Float> = unturn(tipWorld - at, yaw) / trip.scale
        let face = (scape: faceScape(sgn), tip: tipLocal, bow: faceBow(sgn))
        var pose = sweep
        switch stage {
        case .walkingIn:
            let w: Float = smootherstep01((d - (walkIn - 2.5)) / 2.2)
            pose = blendPose(sweep, face, w)
        case .standing:
            pose = face
        case .turning:
            let u: Float = (d - walkIn) / trip.turnLength
            let toRaised: Float = smootherstep01(u / 0.12)
            let toSweep: Float = smootherstep01((u - 0.5) / 0.45)
            pose = blendPose(blendPose(face, raisedPose(i), toRaised), sweep, toSweep)
        case .walkingOut, .away:
            pose = sweep
        }
        let a: Antenna = buildAntenna(index: i, scapeDir: pose.scape, tip: pose.tip, bow: pose.bow, mutant: mutant)
        antennae.append(a)
        let tipShape: Shape = a.segments[a.segments.count - 1].placed(scale: trip.scale, yaw: yaw, at: at)
        tips.append(tipShape.b)
        if i == 0 { apical = (tipShape.a, tipShape.b) }
    }
    let body: [Shape] = bodyShapes(gasterBend: walkingGasterLift, glossa: glossa)
    let model = AntModel(body: body, legs: legs, antennae: antennae)
    let shapes: [Shape] = model.shapes.map { $0.placed(scale: trip.scale, yaw: yaw, at: at) }
    return WorldAnt(id: trip.id, tau: tau, stage: stage, distance: d, at: at, yaw: yaw, model: model, shapes: shapes,
                    feet: feet, stance: stance, tapPhase: phase, lift: lift, glossa: glossa, antennaTips: tips,
                    apical: apical)
}

/// How far the glossa is out at a tap phase: out after the first touch,
/// resting on the juice through the drink, back in before the ant turns.
/// The dryGlossa mutant stops it 0.05 mm short.
func glossaAt(phase: Float, trip: Trip, mutant: Mutant) -> Float {
    let full: Float = trip.glossaReach - (mutant == .dryGlossa ? 0.05 / trip.scale : 0)
    let out: Float = smoothstep01((phase - glossaOut.start) / (glossaOut.full - glossaOut.start))
    let back: Float = 1 - smoothstep01((phase - glossaOut.back) / (glossaOut.stowed - glossaOut.back))
    return full * min(out, back)
}

/// Every ant on its trip at time t (those away from the card are left out).
func antsAt(time t: Float, mutant: Mutant) -> [WorldAnt] {
    trips.compactMap { trip in
        let tau: Float = trip.tripTime(t, mutant: mutant)
        return trip.stage(tau) == .away ? nil : worldAnt(trip, time: t, mutant: mutant)
    }
}
