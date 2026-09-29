// Step 79: the world and its ants — a nest entrance, a sugar pile, and a
// dozen and more Lasius niger workers following simple rules, each one
// sourced or MODEL (SimConstants.swift). Nothing about any route is written
// down anywhere: the trail, if one forms, forms from these rules.
//
// The rules, per ant, per tick:
//   * IN THE NEST it is not drawn. It comes out when its time comes (the first
//     scouts at the start; recruits a little after a successful forager gets
//     home; a forager after unloading), once the entrance is clear. Each time
//     out is a new OUTING (a new track for the film).
//   * SEARCHING: a correlated random walk. If its antennae meet scent it
//     starts FOLLOWING (bearing round, on a walking arc, to head away from the
//     nest if it was facing it). If it searches too long it goes home without
//     laying.
//   * FOLLOWING: at each tick it samples the scent over its antennae's sweep
//     and bears towards the directions the Deneubourg/Beckers choice function
//     weights most. If it loses the scent for a moment it searches in tight
//     loops, then at large.
//   * On touching a sugar grain with its antennae or head it stops: TASTES
//     (three taps) and FEEDS, then RETURNS: it steers home (path integration,
//     MODEL), following a trail that leads home if it meets one, and laying
//     trail if it is a layer. Only laden ants lay (MODEL: see `layingNow`).
//   * LAYING is a Poisson stream of dabs; each dab is a 0.2 s slow-down and
//     stop with the gaster tip on the ground, and puts one mark into the
//     field there, at the middle of the stop.
//   * No ant walks into another: if its next step would bring its body
//     within reach of another's (ahead of it or abreast) it tries bearing
//     hard right or left instead, else waits; after waiting a moment it may
//     squeeze past. At the food itself they crowd without keeping apart.
//   * The arena's edge turns it back, on a walking arc.
//   * EVERY heading change happens while walking and is capped at step length
//     ÷ 5 mm (the shared ant's tightest turn); nothing turns on the spot.

import Foundation
import simd


/// One ant's full state.
struct SimAnt {
    let id: Int
    var pos: SIMD2<Float>
    var yaw: Float
    var distance: Float = 0
    var state: SimAntState = .inNest
    var carrying: Bool = false
    var fed: Bool = false
    let layer: Bool
    /// Times out of the nest so far; the current outing is `outing − 1`.
    var outing: Int = 0
    /// Seconds into the current dab's stop (negative: not dabbing).
    var dabClock: Float = -1
    var dabDeposited: Bool = false
    /// Seconds until the next dab while laying.
    var nextDab: Float = 0
    /// Seconds left tasting and feeding.
    var feedLeft: Float = 0
    /// Seconds since the scent was lost (following).
    var lost: Float = 0
    /// Seconds out of the nest on this outbound trip.
    var outbound: Float = 0
    /// Seconds of tight local search left after losing a trail.
    var localSearch: Float = 0
    var loopSign: Float = 1
    /// Turning to face away from the nest after finding scent.
    var turningOut: Bool = false
    /// Seconds spent waiting for the way to clear.
    var waited: Float = 0
    /// Millimetres left to walk squeezing past others (see `waitPatience`),
    /// and whether it is pushing through all of them.
    var squeezeLeft: Float = 0
    var forcing: Bool = false
    /// Simulated second it leaves the nest (infinity: waiting to be recruited).
    var releaseAt: Double = .infinity
    var rng: SimRandom

    var forward: SIMD2<Float> { SIMD2<Float>(cos(yaw), sin(yaw)) }
    /// The ant's right, in the ground plane (+z when facing +x, as step 44).
    /// Turning right is +yaw.
    var right: SIMD2<Float> { SIMD2<Float>(-sin(yaw), cos(yaw)) }
    var visible: Bool { state != .inNest }
    var dabbing: Bool { dabClock >= 0 }

    /// The two antenna tips: (left, right).
    var antennae: (SIMD2<Float>, SIMD2<Float>) {
        let ahead: SIMD2<Float> = pos + forward * SimConst.antennaAhead
        let side: SIMD2<Float> = right * SimConst.antennaSide
        return (ahead - side, ahead + side)
    }

    var gasterTip: SIMD2<Float> { pos - forward * SimConst.gasterTipBehind }

    /// Footprint capsule's two end points.
    func capsule(at p: SIMD2<Float>, yaw y: Float) -> (SIMD2<Float>, SIMD2<Float>) {
        let f: SIMD2<Float> = SIMD2<Float>(cos(y), sin(y))
        return (p + f * SimConst.bodyFront, p + f * SimConst.bodyRear)
    }

    /// Speed as a fraction of walking speed, and the dab amount (0–1, 1 =
    /// gaster tip on the ground), `dabClock` seconds into a dab — the timing
    /// of lib/ant/v1 and step 44 (a flat-topped bump centred on the stop).
    static func dabProfile(_ clock: Float) -> (speed: Float, dab: Float) {
        if clock < 0 { return (1, 0) }
        let tau: Float = clock - SimConst.dabPause / 2
        let stop: Float = simPlateau(tau, hold: SimConst.dabHold, ramp: SimConst.dabRamp)
        let dab: Float = simPlateau(tau, hold: SimConst.dabHold * 0.8, ramp: SimConst.dabBend)
        return (1 - stop, dab)
    }
}

/// A flat-topped bump: 1 across the hold, easing to 0 over `ramp` each side
/// (step 44's `plateau`).
func simPlateau(_ tau: Float, hold: Float, ramp: Float) -> Float {
    let a: Float = abs(tau)
    if a <= hold / 2 { return 1 }
    if a >= hold / 2 + ramp { return 0 }
    let x: Float = Float.pi * (a - hold / 2) / ramp
    return 0.5 * (1 + cos(x))
}


/// Distance between segments ab and cd in the plane.
func simSegmentDistance(_ a: SIMD2<Float>, _ b: SIMD2<Float>, _ c: SIMD2<Float>, _ d: SIMD2<Float>) -> Float {
    func pointSeg(_ p: SIMD2<Float>, _ s0: SIMD2<Float>, _ s1: SIMD2<Float>) -> Float {
        let v: SIMD2<Float> = s1 - s0
        let len2: Float = simd_dot(v, v)
        var t: Float = len2 > 0 ? simd_dot(p - s0, v) / len2 : 0
        t = min(max(t, 0), 1)
        return simd_distance(p, s0 + v * t)
    }
    // Do they cross?
    func cross(_ u: SIMD2<Float>, _ v: SIMD2<Float>) -> Float { u.x * v.y - u.y * v.x }
    let r: SIMD2<Float> = b - a
    let s: SIMD2<Float> = d - c
    let den: Float = cross(r, s)
    if den != 0 {
        let t: Float = cross(c - a, s) / den
        let u: Float = cross(c - a, r) / den
        if t >= 0 && t <= 1 && u >= 0 && u <= 1 { return 0 }
    }
    let d1: Float = min(pointSeg(a, c, d), pointSeg(b, c, d))
    let d2: Float = min(pointSeg(c, a, b), pointSeg(d, a, b))
    return min(d1, d2)
}

/// Wrap an angle into (−π, π].
func simWrap(_ a: Float) -> Float {
    var x: Float = a.truncatingRemainder(dividingBy: 2 * Float.pi)
    if x > Float.pi { x -= 2 * Float.pi }
    if x <= -Float.pi { x += 2 * Float.pi }
    return x
}

/// The sugar pile: grains dropped at random into the pile's disc without
/// overlapping (random sequential placement), seeded.
func simSugarPile(_ c: SimWorldConfig) -> [SimGrain] {
    var rng = SimRandom(seed: c.seed, stream: 0xC0FFEE)
    var grains: [SimGrain] = []
    var tries: Int = 0
    while grains.count < c.grainCount && tries < 200_000 {
        tries += 1
        let size: Float = rng.float(0.30, 0.67)
        let root: Float = Float(rng.uniform().squareRoot())
        let room: Float = c.sugarRadius - size / 2
        let r: Float = room * root
        let a: Float = rng.float(0, 2 * Float.pi)
        let x: Float = c.sugar.x + r * cos(a)
        let z: Float = c.sugar.y + r * sin(a)
        let yaw: Float = rng.float(0, 2 * Float.pi)
        let face: UInt8 = UInt8(rng.next() % 4)
        var clear: Bool = true
        for g in grains {
            let dx: Float = g.x - x
            let dz: Float = g.z - z
            let gap: Float = (g.size + size) / 2
            let d2: Float = dx * dx + dz * dz
            if d2 < gap * gap { clear = false; break }
        }
        if clear { grains.append(SimGrain(x: x, z: z, size: size, yaw: yaw, restFace: face)) }
    }
    return grains
}

/// The simulation: ants + field, advanced one tick at a time.
final class SimWorld {
    let config: SimWorldConfig
    let grains: [SimGrain]
    let field: SimField
    private(set) var ants: [SimAnt] = []
    private(set) var tick: Int = 0
    private(set) var dabs: [SimDabEvent] = []
    /// Successful returns (a fed ant reaching the nest).
    private(set) var returns: Int = 0
    /// Times an ant was let through after waiting (see `waitPatience`).
    private(set) var squeezes: Int = 0
    /// Simulated second of the first contact with sugar (nil: not yet).
    private(set) var firstContact: Double?

    var time: Double { Double(tick) * SimConst.tick }

    init(config: SimWorldConfig) throws {
        self.config = config
        grains = simSugarPile(config)
        field = try SimField.forSimulation(width: config.gridWidth, height: config.gridHeight)
        var worldRNG = SimRandom(seed: config.seed, stream: 0xA11)
        for i in 0..<config.antCount {
            let layer: Bool = worldRNG.uniform() >= SimConst.neverLayFraction
            var a = SimAnt(id: i, pos: config.nest, yaw: 0, layer: layer,
                           rng: SimRandom(seed: config.seed, stream: UInt64(i + 1)))
            if i < config.scouts { a.releaseAt = Double(i) * Double(config.scoutSpacing) }
            ants.append(a)
        }
    }

    // MARK: - sensing

    func touchesSugar(_ p: SIMD2<Float>) -> Bool {
        let d0: Float = simd_distance(p, config.sugar)
        if d0 > config.sugarRadius + 0.5 { return false }
        for g in grains {
            let dx: Float = g.x - p.x
            let dz: Float = g.z - p.y
            let r: Float = g.size / 2 + 0.05
            let d2: Float = dx * dx + dz * dz
            if d2 <= r * r { return true }
        }
        return false
    }

    /// Does the ant touch a grain now: with an antenna tip, anywhere on its
    /// antennae's sweep (`sweepPoints`), or with its head?
    func touchesSugarNow(_ a: SimAnt, _ tipL: SIMD2<Float>, _ tipR: SIMD2<Float>) -> Bool {
        let head: SIMD2<Float> = a.pos + a.forward * SimConst.bodyFront
        if simd_distance(head, config.sugar) > config.sugarRadius + SimConst.sweepRadius + 1 { return false }
        if touchesSugar(tipL) || touchesSugar(tipR) || touchesSugar(head) { return true }
        for p in sweepPoints(a) where touchesSugar(p) { return true }
        return false
    }

    /// Where the antennal sweep reaches: the arc `sweep` samples.
    func sweepPoints(_ a: SimAnt) -> [SIMD2<Float>] {
        let head: SIMD2<Float> = a.pos + a.forward * SimConst.bodyFront
        let n: Int = SimConst.sweepSamples
        var out: [SIMD2<Float>] = []
        for k in -n...n {
            let frac: Float = Float(k) / Float(n)
            let ang: Float = a.yaw + frac * SimConst.sweepHalfAngle
            out.append(head + SIMD2<Float>(cos(ang), sin(ang)) * SimConst.sweepRadius)
        }
        return out
    }

    // MARK: - one tick

    /// Does ant i's footprint at (p, y) close on the footprint of another
    /// visible ant that is ahead of it or abreast (gap below `limit` and
    /// smaller than it is now)? Ants behind don't block: they give way.
    func blocked(_ i: Int, at p: SIMD2<Float>, yaw y: Float, limit: Float, squeezing: Bool = false) -> Bool {
        let me: (SIMD2<Float>, SIMD2<Float>) = ants[i].capsule(at: p, yaw: y)
        let here: (SIMD2<Float>, SIMD2<Float>) = ants[i].capsule(at: ants[i].pos, yaw: ants[i].yaw)
        let f: SIMD2<Float> = ants[i].forward
        // At the food ants crowd and clamber over one another: no spacing
        // is kept within `foodCrowdMargin` of the pile's edge. MODEL.
        let crowd: Float = config.sugarRadius + SimConst.foodCrowdMargin
        if simd_distance(p, config.sugar) < crowd { return false }
        for j in 0..<ants.count where j != i && ants[j].visible {
            let o: SIMD2<Float> = ants[j].pos
            if simd_distance(o, p) > 12 { continue }
            if simd_distance(o, config.sugar) < crowd { continue }
            if simd_dot(o - ants[i].pos, f) < -1 { continue }
            // Squeezing past: an ant coming the other way is ignored, and so
            // is one stuck waiting too if it gives way to this one (the lower
            // id goes first, so two stuck ants never both push through).
            // Never one at the food.
            if squeezing && ants[j].state != .tasting {
                if simd_dot(ants[j].forward, f) < -0.3 { continue }
                if ants[j].waited > 0 && j > i { continue }
            }
            let other: (SIMD2<Float>, SIMD2<Float>) = ants[j].capsule(at: o, yaw: ants[j].yaw)
            let now: Float = simSegmentDistance(me.0, me.1, other.0, other.1)
            if now < limit {
                let before: Float = simSegmentDistance(here.0, here.1, other.0, other.1)
                if now <= before { return true }
            }
        }
        return false
    }

    /// Which way an ant leaves the nest: one of `exitDirections` directions,
    /// drawn with probability ∝ (k + C)ⁿ — the choice function again, over
    /// the scent `exitSniffRadius` out along each. With no scent anywhere
    /// every direction is equally likely (up to a random turn within its
    /// sector). MODEL: ants leaving by a marked path pick it up at the door.
    func exitHeading(_ rng: inout SimRandom) -> Float {
        let n: Int = SimConst.exitDirections
        var weights: [Float] = []
        var total: Float = 0
        for k in 0..<n {
            let sector: Float = 2 * Float.pi / Float(n)
            let ang: Float = sector * Float(k)
            let p: SIMD2<Float> = config.nest + SIMD2<Float>(cos(ang), sin(ang)) * SimConst.exitSniffRadius
            let c: Float = max(field.sample(x: p.x, z: p.y), 0)
            let w: Float = pow(SimConst.choiceK + c, SimConst.choiceN)
            weights.append(w)
            total += w
        }
        let draw: Float = Float(rng.uniform())
        let u: Float = draw * total
        var acc: Float = 0
        var pick: Int = n - 1
        for k in 0..<n {
            acc += weights[k]
            if u < acc { pick = k; break }
        }
        let wobble: Float = rng.float(-0.5, 0.5)
        let jitter: Float = wobble * 2 * Float.pi / Float(n)
        let sector: Float = 2 * Float.pi / Float(n)
        let heading: Float = sector * Float(pick)
        return simWrap(heading + jitter)
    }

    /// The antennal sweep: the field sampled on an arc of radius
    /// `SimConst.sweepRadius` about the head, from straight ahead out to
    /// `SimConst.sweepHalfAngle` either side. A sample whose direction points
    /// the wrong way — back towards the nest for an outbound ant, away from it
    /// for one going home — counts `SimConst.backwardWeight` as much. MODEL
    /// (see SimConstants).
    ///
    /// Returns the strongest sample on each side (the straight-ahead one
    /// counts for both), and the direction to bear: the sample directions
    /// averaged with the choice function's weights (k + C)ⁿ — the Deneubourg
    /// form, applied to the several "paths" the antennae find ahead instead of
    /// to two arms of a fork. So an ant crossing a trail at an angle turns
    /// along it (the arc meets the trail ahead on one side), and an ant on a
    /// trail keeps to its middle.
    func sweep(_ a: SimAnt, outbound: Bool) -> (left: Float, right: Float, bear: Float) {
        let head: SIMD2<Float> = a.pos + a.forward * SimConst.bodyFront
        let fromNest: SIMD2<Float> = a.pos - config.nest
        // The way this ant wants to go: out, or home.
        let away: SIMD2<Float> = outbound ? fromNest : -fromNest
        let n: Int = SimConst.sweepSamples
        var left: Float = 0
        var right: Float = 0
        var sumW: Float = 0
        var sumA: Float = 0
        for k in -n...n {
            let frac: Float = Float(k) / Float(n)
            let off: Float = frac * SimConst.sweepHalfAngle
            let ang: Float = a.yaw + off
            let dir = SIMD2<Float>(cos(ang), sin(ang))
            let p: SIMD2<Float> = head + dir * SimConst.sweepRadius
            var c: Float = max(field.sample(x: p.x, z: p.y), 0)
            if simd_dot(dir, away) < 0 { c *= SimConst.backwardWeight }
            if k <= 0 { left = max(left, c) }
            if k >= 0 { right = max(right, c) }
            let w: Float = pow(SimConst.choiceK + c, SimConst.choiceN)
            sumW += w
            sumA += w * off
        }
        return (left, right, sumW > 0 ? sumA / sumW : 0)
    }

    /// The tightest turn an ant makes here, mm: the shared ant's 5 mm with
    /// the rounding margin (3 mm for the `tightTurn` mutant).
    var turnCapRadius: Float {
        config.mutant == .tightTurn ? 3 : SimConst.minTurnRadius * SimConst.turnCapMargin
    }

    /// Can an ant at (p, y) always stay in the arena? Its slack is how far
    /// the better of its two tightest turning circles (radius `minTurnRadius`
    /// either side), widened by the antennae's reach, lies inside the arena
    /// (negative: pokes out). Walking round that circle keeps the slack, so
    /// an ant that only takes steps keeping it ≥ 0 can never be trapped
    /// against an edge — and never has to turn on the spot.
    func escapeSlack(_ p: SIMD2<Float>, yaw y: Float) -> Float {
        let r = SIMD2<Float>(-sin(y), cos(y))
        let radius: Float = turnCapRadius
        let reach: Float = radius + SimConst.antennaAhead + SimConst.antennaSide
        var best: Float = -.infinity
        for side in [Float(1), Float(-1)] {
            let c: SIMD2<Float> = p + r * (side * radius)
            let sx: Float = min(c.x - reach, config.width - reach - c.x)
            let sz: Float = min(c.y - reach, config.depth - reach - c.y)
            best = max(best, min(sx, sz))
        }
        return best
    }

    /// A step to (p, y) keeps the escape: its slack is not below zero — or,
    /// for an ant already short of it (rounding), not below its slack now.
    func keepsEscape(_ a: SimAnt, _ p: SIMD2<Float>, yaw y: Float) -> Bool {
        let now: Float = escapeSlack(a.pos, yaw: a.yaw)
        return escapeSlack(p, yaw: y) >= min(0, now) - 1e-3
    }

    /// Inside the arena with the ant's antennae and gaster.
    func inBounds(_ p: SIMD2<Float>, yaw y: Float) -> Bool {
        let f = SIMD2<Float>(cos(y), sin(y))
        let pts: [SIMD2<Float>] = [p, p + f * (SimConst.antennaAhead + SimConst.antennaSide),
                                   p - f * SimConst.gasterTipBehind]
        let m: Float = 0.0
        for q in pts where q.x < m || q.y < m || q.x > config.width - m || q.y > config.depth - m { return false }
        return true
    }

    private func release(_ n: Int, at t: Double) {
        var left: Int = n
        for j in 0..<ants.count where left > 0 {
            if ants[j].state == .inNest && ants[j].releaseAt == .infinity {
                ants[j].releaseAt = t + Double(SimConst.recruitDelay) + Double(n - left) * 0.5
                left -= 1
            }
        }
    }

    /// Only laden ants lay. MODEL simplification: Beckers et al. (1992) saw
    /// trail laid "more or less equally both to and from the nest"; but these
    /// ants carry no memory of the route, so an outbound ant that lays wherever
    /// it wanders would lay false trails. Laying only on the way home keeps
    /// every mark on a real nest–food path.
    private func layingNow(_ a: SimAnt) -> Bool {
        a.fed && a.layer && a.state == .returning && a.carrying
    }

    private(set) var track: [[SimTrackPoint]] = []

    func step() throws {
        if track.isEmpty { track = Array(repeating: [], count: ants.count) }
        let dt: Float = Float(SimConst.tick)
        let t: Double = time
        var newDabs: [SimDabGPU] = []
        for i in 0..<ants.count {
            var a: SimAnt = ants[i]
            defer { ants[i] = a }

            if a.state == .inNest {
                if t >= a.releaseAt {
                    let y: Float = exitHeading(&a.rng)
                    let savedYaw: Float = ants[i].yaw
                    ants[i].yaw = y
                    let clear: Bool = !blocked(i, at: config.nest, yaw: y, limit: 2 * SimConst.bodyRadius)
                    ants[i].yaw = savedYaw
                    if clear {
                        a.state = .searching
                        a.pos = config.nest
                        a.yaw = y
                        a.outbound = 0
                        a.lost = 0
                        a.turningOut = false
                        a.releaseAt = .infinity
                        a.outing += 1
                        track[i].append(SimTrackPoint(tick: UInt32(tick), outing: UInt32(a.outing - 1),
                                                      distance: a.distance, x: a.pos.x, z: a.pos.y, yaw: a.yaw))
                    }
                }
                continue
            }

            // The turn the rules want this tick (radians, + = right).
            var want: Float = 0
            if a.state == .tasting {
                a.feedLeft -= dt
                if a.feedLeft <= 0 {
                    a.state = .returning
                    a.carrying = true
                    a.fed = true
                    a.nextDab = a.rng.exponential(rate: SimConst.dabsPerSecond)
                }
                continue
            }

            do {
            // Sense: what each antenna finds over its sweep (the strongest
            // sample on its side), outbound ants discounting a side that
            // points back towards the nest.
            let (tipL, tipR) = a.antennae
            let sensed: (left: Float, right: Float, bear: Float) = sweep(a, outbound: a.state != .returning)
            let cl: Float = sensed.left
            let cr: Float = sensed.right
            let scent: Float = cl + cr

            if a.state == .searching || a.state == .following {
                a.outbound += dt
                if touchesSugarNow(a, tipL, tipR) {
                    a.state = .tasting
                    a.feedLeft = SimConst.tasteSeconds + SimConst.feedSeconds
                    a.dabClock = -1
                    if firstContact == nil { firstContact = t }
                    continue
                }
                if a.outbound > SimConst.searchGiveUp {
                    a.state = .returning
                    a.carrying = false
                }
            }

            let noise: Float
            switch a.state {
            case .searching:
                if scent > SimConst.detectThreshold {
                    a.state = .following
                    a.lost = 0
                    // Head away from the nest along the scent. MODEL.
                    let away: SIMD2<Float> = a.pos - config.nest
                    a.turningOut = simd_dot(a.forward, away) < 0
                    noise = SimConst.walkTurnDiffusion
                } else if a.localSearch > 0 {
                    // Search loops: circle as tightly as it can, now one way,
                    // now the other.
                    a.localSearch -= dt
                    let draw: Float = Float(a.rng.uniform())
                    let chance: Float = SimConst.loopSwitchRate * dt
                    if draw < chance { a.loopSign = -a.loopSign }
                    want = a.loopSign
                    noise = 0
                } else {
                    noise = SimConst.searchTurnDiffusion
                }
            case .following:
                if scent > SimConst.detectThreshold {
                    a.lost = 0
                    // Bear the way the sweep says (capped below by the
                    // turning radius).
                    want = sensed.bear
                } else {
                    a.lost += dt
                    if a.lost > SimConst.lostPatience {
                        a.state = .searching
                        a.turningOut = false
                        a.localSearch = SimConst.localSearchSeconds
                        a.loopSign = a.rng.uniform() < 0.5 ? -1 : 1
                    }
                }
                noise = SimConst.walkTurnDiffusion
            case .returning:
                let home: SIMD2<Float> = config.nest - a.pos
                let dist: Float = simd_length(home)
                if dist <= config.nestRadius {
                    a.state = .inNest
                    a.pos = config.nest
                    a.dabClock = -1
                    if a.carrying {
                        a.carrying = false
                        returns += 1
                        release(SimConst.recruitsPerReturn, at: t)
                    }
                    a.releaseAt = t + Double(SimConst.unloadSeconds)
                    continue
                }
                let bearing: Float = simWrap(atan2(home.y, home.x) - a.yaw)
                // Too close to reach the entrance on a 5 mm arc: carry on
                // straight to come round again (MODEL).
                let tooTight: Bool = dist < 2 * SimConst.minTurnRadius && abs(bearing) > 1.0
                if !tooTight {
                    let most: Float = SimConst.homingTurnRate * dt
                    let homing: Float = min(max(bearing, -most), most)
                    // On a trail that leads roughly home, follow it (and so
                    // reinforce it), with a pull towards home. MODEL.
                    let onTrail: Bool = scent > SimConst.detectThreshold && abs(bearing) < Float.pi / 2
                    if onTrail && dist > SimMetric.nestExclusion {
                        want = sensed.bear + SimConst.homePullOnTrail * homing
                    } else {
                        want = homing
                    }
                }
                noise = SimConst.walkTurnDiffusion
            default:
                noise = 0
            }
            if a.turningOut {
                let away: SIMD2<Float> = a.pos - config.nest
                if simd_dot(a.forward, away) >= 0 {
                    a.turningOut = false
                } else {
                    let cross: Float = a.forward.x * away.y - a.forward.y * away.x
                    want = cross >= 0 ? 1 : -1
                }
            }
            if noise > 0 {
                let s: Float = (noise * dt).squareRoot()
                want += s * a.rng.normal()
            }
            // The edge: near a wall and not already heading away from it,
            // bear away from it on a walking arc.
            let walls: [(Float, SIMD2<Float>)] = [
                (a.pos.x, SIMD2<Float>(-1, 0)), (config.width - a.pos.x, SIMD2<Float>(1, 0)),
                (a.pos.y, SIMD2<Float>(0, -1)), (config.depth - a.pos.y, SIMD2<Float>(0, 1)),
            ]
            let centre = SIMD2<Float>(config.width / 2, config.depth / 2)
            for (gap, out) in walls where gap < SimConst.edgeLookahead && simd_dot(a.forward, out) > -0.2 {
                let inward: SIMD2<Float> = centre - a.pos
                let cross: Float = a.forward.x * inward.y - a.forward.y * inward.x
                want = cross >= 0 ? 1 : -1
            }
            }

            // Lay: a dab is due — start its stop.
            if layingNow(a) && !a.dabbing {
                // Beckers et al. (1992): ants returning to the nest "lay up
                // to five times more on the segment closest to the source
                // than that closest to the nest" — the rate falls linearly
                // from the full rate at the sugar to a fifth at the nest.
                let toNest: Float = simd_distance(a.pos, config.nest)
                let span: Float = simd_distance(config.sugar, config.nest)
                let share: Float = min(max(toNest / span, 0), 1)
                let nearNest: Float = SimConst.nestEndLayingFraction
                let factor: Float = nearNest + (1 - nearNest) * share
                a.nextDab -= dt * factor * (a.waited > 0 ? 0 : 1)
                if a.nextDab <= 0 {
                    a.nextDab = a.rng.exponential(rate: SimConst.dabsPerSecond)
                    a.dabClock = 0
                    a.dabDeposited = false
                }
            }
            var speedFraction: Float = 1
            if a.dabbing {
                speedFraction = SimAnt.dabProfile(a.dabClock).speed
                if !a.dabDeposited && a.dabClock >= SimConst.dabPause / 2 {
                    a.dabDeposited = true
                    if config.mutant != .noPheromone {
                        let g: SIMD2<Float> = a.gasterTip
                        let gx: Float = min(max(g.x, SimConst.cell), config.width - SimConst.cell)
                        let gz: Float = min(max(g.y, SimConst.cell), config.depth - SimConst.cell)
                        dabs.append(SimDabEvent(tick: UInt32(tick), x: gx, z: gz, marks: SimConst.marksPerDab))
                        newDabs.append(field.gpuDab(x: gx, z: gz, marks: SimConst.marksPerDab))
                    }
                }
                a.dabClock += dt
                if a.dabClock >= SimConst.dabPause { a.dabClock = -1 }
            }

            // Move on an arc no tighter than the shared ant can walk.
            let stepLength: Float = SimConst.walkSpeed * dt * speedFraction
            if stepLength <= 0 { continue }
            let most: Float = stepLength / turnCapRadius
            let turn: Float = min(max(want, -most), most)
            var chosen: Float? = nil
            // Bodies are kept apart (legs may brush). An ant that has waited
            // `waitPatience` may squeeze past for `squeezeDistance` mm: past
            // an ant coming the other way, or one stuck waiting that gives way
            // to it; stuck again while squeezing, it pushes through. Without
            // backing up or turning on the spot (which the shared ant can't
            // draw), ants that meet head to head could otherwise wait on each
            // other for ever. MODEL; `squeezes` counts them, and the tests
            // measure how often bodies overlap.
            if a.waited > SimConst.waitPatience {
                // Still stuck while already squeezing: push through regardless.
                a.forcing = a.squeezeLeft > 0
                a.squeezeLeft = SimConst.squeezeDistance
                a.waited = 0
                squeezes += 1
            }
            let limit: Float = 2 * SimConst.bodyRadius
            ants[i].yaw = a.yaw
            for candidate in [turn, most, -most] {
                let y: Float = a.yaw + candidate
                let mid: Float = a.yaw + candidate / 2
                let next: SIMD2<Float> = a.pos + SIMD2<Float>(cos(mid), sin(mid)) * stepLength
                if !inBounds(next, yaw: y) || !keepsEscape(a, next, yaw: y) { continue }
                if !a.forcing && blocked(i, at: next, yaw: y, limit: limit, squeezing: a.squeezeLeft > 0) { continue }
                chosen = candidate
                break
            }
            guard let c = chosen else {
                a.waited += dt      // wait: nothing turns on the spot
                continue
            }
            a.waited = 0
            a.squeezeLeft = max(a.squeezeLeft - stepLength, 0)
            if a.squeezeLeft == 0 { a.forcing = false }
            let mid: Float = a.yaw + c / 2
            a.pos += SIMD2<Float>(cos(mid), sin(mid)) * stepLength
            a.yaw = simWrap(a.yaw + c)
            a.distance += stepLength
            track[i].append(SimTrackPoint(tick: UInt32(tick + 1), outing: UInt32(a.outing - 1),
                                          distance: a.distance, x: a.pos.x, z: a.pos.y, yaw: a.yaw))
        }
        try field.step(dabs: newDabs)
        tick += 1
    }
}
