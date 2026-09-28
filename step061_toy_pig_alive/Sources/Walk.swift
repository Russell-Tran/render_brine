// A toy come to life: it stands frozen in the pose it was moulded in, wakes,
// looks round, walks once round a closed circle, settles back into exactly
// that pose, and freezes again. Nothing here touches the GPU.
//
// FICTION, BY DESIGN. A one-piece toy has no joints. Everything else stays
// true: the pieces are rigid (they turn about joints, never bend or
// stretch), nothing passes through anything, and the gait is the real
// animal's — its footfall order and how long each foot stays down.
//
// Three rules, as in step 44:
//
//   1. PLANTED FEET DO NOT MOVE. Every step is an explicit event: a foot
//      lifts from one foothold at one time and lands on the next at a later
//      time; between steps it stands still on the table, exactly. Where each
//      foothold is comes from the distance walked (a foot lands where it
//      will be under its hip halfway through its stance), so the feet keep
//      pace with the body at any speed.
//
//   2. THE LOOP CLOSES GOING FORWARD. The path is a circle; the toy walks
//      it once, so at the end it stands where it started, facing the way it
//      started — having turned a full 360°, never backed up. Distance along
//      the path never decreases, and the stride divides the circle a whole
//      number of times, so the legs come round too.
//
//   3. THE CAMERA DOES NOT MOVE.
//
// Millimetres and picture seconds. The world: y up, the table is y = 0.

import Foundation
import simd

// MARK: - the walk's parameters

/// What an animal's walk needs: the gait (from the literature) and the
/// choreography (MODEL, for the picture).
struct WalkSpec {
    /// Fraction of each stride a foot is on the ground.
    var dutyFactor: Float
    /// When each leg touches down, as a fraction of the stride after the
    /// left hind, in the design's leg order (LF, RF, LH, RH).
    var offsets: [Float]
    /// The order the feet land in, by name, for the test.
    var footfallOrder: [String]
    /// Stride length (one full cycle of every leg), mm, and how many strides
    /// round the circle. The circle's circumference is strides × stride.
    var stride: Float
    var strides: Int
    /// Picture seconds per stride while cruising.
    var strideSeconds: Float
    /// How far the body drops while walking, mm, so the legs can reach a full
    /// stride; it rises back to the moulded pose when it settles. MODEL.
    var crouch: Float
    /// How high a swinging foot lifts, mm. MODEL.
    var footLift: Float
    /// How far the head turns when it looks round, radians. MODEL.
    var lookYaw: Float
    /// How far the tail swings each stride, radians. MODEL.
    var tailSway: Float
    /// The left hind's phase when the walk starts (0 = just landed). MODEL,
    /// chosen so no foot is caught at the very end of its swing.
    var startPhase: Float = 0.1
    /// Seconds: frozen, waking, the walk's speed ramps, settling, frozen.
    var holdStart: Float = 1.5
    var wake: Float = 3.2
    var ramp: Float = 0.8
    var settle: Float = 2.6
    var holdEnd: Float = 1.5
    /// A step taken while the body stands still (waking, settling). MODEL.
    var standingStepSeconds: Float = 0.42

    var circumference: Float { stride * Float(strides) }
    var radius: Float { circumference / (2 * .pi) }
    var walkSeconds: Float { Float(strides) * strideSeconds + ramp }
    var loopSeconds: Float { holdStart + wake + walkSeconds + settle + holdEnd }
    var walkStart: Float { holdStart + wake }
    var walkEnd: Float { walkStart + walkSeconds }
    var settleEnd: Float { walkEnd + settle }
}

/// The lateral-sequence walk's offsets in the design's leg order (LF, RF,
/// LH, RH): left hind, left fore, right hind, right fore, a quarter-stride
/// apart. The `wrongGait` mutant swaps in a trot, the diagonal pairs landing
/// together.
func gaitOffsets(_ mutant: Mutant) -> [Float] {
    if mutant == .wrongGait { return [0.0, 0.5, 0.5, 0.0] }
    return [0.25, 0.75, 0.0, 0.5]
}

// MARK: - the path

/// The circle the toy walks: it starts at the origin facing +x and turns
/// left (anticlockwise seen from above), so the centre is on its left, at −z.
struct Path {
    let radius: Float
    var centre: SIMD3<Float> { SIMD3<Float>(0, 0, -radius) }
    /// Where the body's origin is (on the table) and which way it faces,
    /// after walking `s`.
    func at(_ s: Float) -> (point: SIMD3<Float>, yaw: Float) {
        let phi: Float = s / radius
        let p: SIMD3<Float> = centre + SIMD3<Float>(sin(phi), 0, cos(phi)) * radius
        return (p, phi)
    }
}

// MARK: - one step

/// One step of one foot: it lifts from `from` at `lift` and lands on `to`
/// at `land` (picture seconds). Between steps a foot stands still.
struct Step {
    var from: Frame
    var to: Frame
    var lift: Float
    var land: Float
}

/// A smooth ease 0→1.
func ease(_ x: Float) -> Float {
    let u: Float = min(max(x, 0), 1)
    return u * u * (3 - 2 * u)
}

func wrapAngle(_ a: Float) -> Float {
    var x: Float = a.truncatingRemainder(dividingBy: 2 * .pi)
    if x > .pi { x -= 2 * .pi }
    if x < -.pi { x += 2 * .pi }
    return x
}

// MARK: - the whole performance

struct Performance {
    let design: ToyDesign
    let spec: WalkSpec
    let path: Path
    let mutant: Mutant
    /// Each leg's steps, in time order.
    private(set) var steps: [[Step]] = []
    /// Each leg's foothold at rest (the moulded pose).
    let rest: [Frame]

    init(design: ToyDesign, spec: WalkSpec, mutant: Mutant = activeMutant) {
        self.design = design
        self.spec = spec
        self.path = Path(radius: spec.radius)
        self.mutant = mutant
        let pose0: Pose = design.restPose()
        rest = pose0.feet
        steps = (0..<design.legs.count).map { planSteps(leg: $0) }
    }

    // MARK: distance and time

    /// Distance walked at picture time t (before any mutant): nothing until
    /// the walk starts, then easing up to cruising speed, cruising, easing to
    /// a stop at exactly once round.
    func rawDistance(_ t: Float) -> Float {
        let L: Float = spec.circumference
        let te: Float = spec.ramp
        let v: Float = L / (spec.walkSeconds - te)
        let t0: Float = spec.walkStart
        let t1: Float = spec.walkEnd
        if t <= t0 { return 0 }
        if t >= t1 { return L }
        if t < t0 + te {
            let x: Float = (t - t0) / te
            return v * te * (x * x * x - x * x * x * x / 2)
        }
        if t > t1 - te {
            let y: Float = (t1 - t) / te
            return L - v * te * (y * y * y - y * y * y * y / 2)
        }
        return v * (te / 2 + (t - t0 - te))
    }

    /// The time the walk reaches distance s (inverse of rawDistance).
    func time(atDistance s: Float) -> Float {
        var lo: Float = spec.walkStart
        var hi: Float = spec.walkEnd
        if s <= 0 { return lo }
        if s >= spec.circumference { return hi }
        for _ in 0..<60 {
            let m: Float = (lo + hi) / 2
            if rawDistance(m) < s { lo = m } else { hi = m }
        }
        return (lo + hi) / 2
    }

    /// Picture time as the performance plays it. The `rewind` mutant closes
    /// the loop by playing the first half backwards in the second.
    func playTime(_ t: Float) -> Float {
        let T: Float = spec.loopSeconds
        var u: Float = t.truncatingRemainder(dividingBy: T)
        if u < 0 { u += T }
        if mutant == .rewind && u > T / 2 { return T - u }
        if mutant == .frozen { return 0 }
        return u
    }

    func distance(_ t: Float) -> Float { rawDistance(playTime(t)) }

    // MARK: the body

    /// How far the body is lowered at time u: down while waking, back up at
    /// the very end of settling.
    func crouch(_ u: Float) -> Float {
        let down: Float = ease((u - spec.holdStart) / 0.8)
        let up: Float = 1 - ease((u - (spec.settleEnd - 0.7)) / 0.7)
        return spec.crouch * min(down, up)
    }

    /// The body's frame at picture time u (already mapped by playTime).
    func bodyFrame(_ u: Float) -> Frame {
        let s: Float = rawDistance(u)
        let (p, yaw) = path.at(s)
        return Frame.level(origin: p + SIMD3<Float>(0, design.bodyHeight - crouch(u), 0), yaw: yaw)
    }

    /// The body's frame, standing uncrouched, after walking s: what the
    /// footholds are measured from.
    func standingFrame(_ s: Float) -> Frame {
        let (p, yaw) = path.at(s)
        return Frame.level(origin: p + SIMD3<Float>(0, design.bodyHeight, 0), yaw: yaw)
    }

    // MARK: the feet

    /// Leg j's phase after walking s: 0 at touchdown, stance below the duty
    /// factor.
    func phase(_ j: Int, _ s: Float) -> Float {
        let x: Float = s / spec.stride + spec.startPhase - spec.offsets[j]
        return x - x.rounded(.down)
    }

    /// Where leg j lands for the stance that starts at distance tau: under
    /// its resting place halfway through that stance, facing the way the
    /// body will face then.
    func foothold(_ j: Int, touchdown tau: Float) -> Frame {
        let mid: Float = tau + spec.dutyFactor * spec.stride / 2
        let f: Frame = standingFrame(mid)
        var p: SIMD3<Float> = f.toWorld(design.legs[j].restFoot)
        p.y = 0
        return Frame.level(origin: p, yaw: path.at(mid).yaw)
    }

    /// The steps leg j takes, from the moulded pose round the circle and back.
    func planSteps(leg j: Int) -> [Step] {
        let S: Float = spec.stride
        let L: Float = spec.circumference
        let df: Float = spec.dutyFactor
        var out: [Step] = []
        var at: Frame = rest[j]
        let p0: Float = phase(j, 0)
        // Waking: a foot whose stance at the start would plant it ahead of
        // where it rests steps forward to it first, while the body stands
        // still, so no foot is left trailing further than a stride allows.
        let wakeOrder: [Int] = legOrder()
        let wakeSlot: Float = spec.walkStart - 0.35 - Float(wakeOrder.count - (wakeOrder.firstIndex(of: j) ?? 0)) * spec.standingStepSeconds * 0.8
        var liftDistance: Float
        if p0 < df {
            if p0 < df / 2 {
                let target: Frame = foothold(j, touchdown: -p0 * S)
                out.append(Step(from: at, to: target, lift: wakeSlot, land: wakeSlot + spec.standingStepSeconds))
                at = target
            }
            liftDistance = (df - p0) * S
        } else {
            liftDistance = 0
        }
        // The walk: each later touchdown at a whole number of strides.
        var tau: Float = (1 - p0) * S
        if p0 >= df { tau = (1 - p0) * S }
        var redirected: Bool = false
        while tau < L - 1e-3 || (liftDistance < L - 1e-3 && !redirected) {
            let liftT: Float = time(atDistance: liftDistance)
            if tau < L - 1e-3 {
                let target: Frame = foothold(j, touchdown: tau)
                out.append(Step(from: at, to: target, lift: liftT, land: time(atDistance: tau)))
                at = target
                liftDistance = tau + df * S
                tau += S
            } else {
                // Still in the air when the walk ends: land at rest instead,
                // taking the gait's own swing time (the body is all but
                // stopped by then; a slower swing would linger by the leg
                // behind it).
                let swing: Float = (1 - df) * spec.strideSeconds
                out.append(Step(from: at, to: rest[j], lift: liftT, land: max(liftT + swing, spec.walkEnd - spec.ramp * 0.5)))
                at = rest[j]
                redirected = true
            }
        }
        // Settling: every other foot steps back to where it rests, one at a
        // time, in the gait's own order.
        if !redirected {
            let order: [Int] = legOrder()
            let k: Int = order.firstIndex(of: j) ?? 0
            let start: Float = spec.walkEnd + 0.45 + Float(k) * spec.standingStepSeconds * 1.05
            out.append(Step(from: at, to: rest[j], lift: start, land: start + spec.standingStepSeconds))
        }
        return out
    }

    /// Legs in footfall order.
    func legOrder() -> [Int] {
        spec.footfallOrder.compactMap { n in design.legs.firstIndex { $0.name == n } }
    }

    /// Leg j's foot at picture time u (already mapped), and whether it is down.
    func foot(_ j: Int, _ u: Float) -> (frame: Frame, down: Bool) {
        var planted: Frame = rest[j]
        var landedAt: Float = -1
        for st in steps[j] {
            if u < st.lift { break }
            if u < st.land {
                let x: Float = (u - st.lift) / (st.land - st.lift)
                let e: Float = ease(x)
                var p: SIMD3<Float> = st.from.o + (st.to.o - st.from.o) * e
                p.y = spec.footLift * sin(Float.pi * x)
                let ya: Float = atan2(-st.from.x.z, st.from.x.x)
                let yb: Float = atan2(-st.to.x.z, st.to.x.x)
                let yaw: Float = ya + wrapAngle(yb - ya) * e
                return (Frame.level(origin: p, yaw: yaw), false)
            }
            planted = st.to
            landedAt = st.land
        }
        if mutant == .slidingFeet && landedAt >= 0 {
            // Wrong on purpose: the foot rides along with the body as if
            // glued to it, from where it landed.
            let b0: Frame = bodyFrame(landedAt)
            let b1: Frame = bodyFrame(u)
            let local: SIMD3<Float> = b0.toLocal(planted.o)
            var p: SIMD3<Float> = b1.toWorld(local)
            p.y = 0
            let yawNow: Float = atan2(-b1.x.z, b1.x.x) - atan2(-b0.x.z, b0.x.x) + atan2(-planted.x.z, planted.x.x)
            return (Frame.level(origin: p, yaw: yawNow), true)
        }
        return (planted, true)
    }

    // MARK: head and tail

    /// The head's turn: looking left, then right, while waking; into the turn
    /// while walking; back to straight as it settles.
    func head(_ u: Float) -> (yaw: Float, pitch: Float) {
        let w0: Float = spec.holdStart + 0.7
        let look: Float = spec.lookYaw
        var yaw: Float = 0
        let a: Float = ease((u - w0) / 0.6)
        let b: Float = ease((u - (w0 + 1.1)) / 0.8)
        let c: Float = ease((u - (w0 + 2.1)) / 0.5)
        yaw = look * a - 2 * look * b + look * c
        // Into the turn while walking.
        let walking: Float = min(ease((u - spec.walkStart) / spec.ramp), 1 - ease((u - spec.walkEnd + 0.2) / 0.6))
        yaw += 0.12 * walking
        let pitch: Float = 0.10 * (ease((u - w0) / 0.5) - ease((u - (w0 + 2.4)) / 0.5))
        return (yaw, pitch)
    }

    /// The tail's swing: a wag as it wakes, then in time with the stride.
    func tail(_ u: Float) -> Float {
        let s: Float = rawDistance(u)
        let walking: Float = min(ease((u - spec.walkStart) / spec.ramp), 1 - ease((u - spec.walkEnd + 0.3) / 0.5))
        let w0: Float = spec.holdStart + 0.3
        let wagWindow: Float = min(ease((u - w0) / 0.3), 1 - ease((u - (w0 + 1.3)) / 0.3))
        let wag: Float = 2.2 * spec.tailSway * sin(2 * Float.pi * (u - w0) / 0.8) * wagWindow
        return spec.tailSway * sin(2 * Float.pi * s / spec.stride) * walking + wag
    }

    // MARK: the pose

    func pose(_ t: Float) -> Pose {
        let u: Float = playTime(t)
        let feet: [Frame] = (0..<design.legs.count).map { foot($0, u).frame }
        let h = head(u)
        return Pose(body: bodyFrame(u), headYaw: h.yaw, headPitch: h.pitch, tailYaw: tail(u), feet: feet)
    }

    func posed(_ t: Float) -> PosedToy { design.posed(pose(t), seams: mutant != .noSeam) }

    /// Which feet are down at time t.
    func down(_ t: Float) -> [Bool] {
        let u: Float = playTime(t)
        return (0..<design.legs.count).map { foot($0, u).down }
    }
}
