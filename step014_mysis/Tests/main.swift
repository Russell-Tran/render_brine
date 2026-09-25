// Tests for step 14.
//
// Three things are on trial that step 13 did not have to prove.
//
// The OPERATOR. Darkfield is not brightfield with the sign flipped. It deposits
// at interfaces and nowhere else, it weights each interface by how edge-on it
// is, and its background is exactly zero rather than nearly zero. Each of those
// is easy to write in a way that looks right and is quietly wrong, so each is
// checked against a closed form rather than against a picture.
//
// The FROZEN CAMERA. The whole shot is one camera that never moves, and the
// swept-bound mask is an optimisation that is only correct because of it. So
// the camera is checked bit for bit, and the mask is checked by rendering a
// frame without it and demanding the same pixels.
//
// The ANIMAL. Two statocysts, one in each uropod endopod, is the diagnostic
// character of the order Mysida — if this render has one, or three, or has them
// in the exopods, it is not a mysid. Eight pairs of thoracopods is a claim
// about a real animal and may not drift.
//
// Five of these tests exist to catch a specific mistake, and `make mutants`
// makes each mistake on purpose to prove they do:
//
//   MYSIS_MUTATE=grazing   every interface scatters the same, rims and all
//   MYSIS_MUTATE=mask      the swept bound is built from frame 0 alone
//   MYSIS_MUTATE=phase     the beat is reduced modulo the wrong period
//   MYSIS_MUTATE=wave      the metachronal wave runs head to tail
//   MYSIS_MUTATE=union     intervals summed instead of merged

import CoreGraphics
import Foundation
import Metal
import simd

let mutations = MMutations.fromEnvironment()
let posture = MysisPosture()
let sigma: [SIMD3<Float>] = sigmaTableM()
let reynolds = MReynolds()

func near(_ a: Float, _ b: Float, within e: Float) -> Bool { abs(a - b) <= e }
func near(_ a: Double, _ b: Double, within e: Double) -> Bool { abs(a - b) <= e }

/// A ray straight down −Z through `p`, which is what the closed-form tests use.
func rayThrough(_ p: SIMD3<Float>) -> (SIMD3<Float>, SIMD3<Float>) {
    (SIMD3(p.x, p.y, p.z + 100_000), SIMD3(0, 0, -1))
}

func scatterThrough(_ point: SIMD3<Float>, _ prims: [GPUPrim],
                    _ table: [SIMD3<Float>] = sigma) -> CPUScatter {
    let (o, d) = rayThrough(point)
    return cpuScatter(origin: o, direction: d, prims: prims, sigma: table, mutations: mutations)
}

let oneSigma: [SIMD3<Float>] = {
    var t = [SIMD3<Float>](repeating: .zero, count: mTissueCount)
    t[MTissue.cuticle.rawValue] = SIMD3(1.0, 2.0, 3.0)
    t[MTissue.innerWall.rawValue] = SIMD3(0.5, 0.25, 0.125)
    return t
}()

// MARK: - the loop

section("the loop, which has to close exactly")

test("frame 120 is frame 0, primitive for primitive") {
    let a = poseMysis(frame: 0, posture: posture, mutations: mutations)
    let b = poseMysis(frame: mysisFrameCount, posture: posture, mutations: mutations)
    expectEqual(a.prims.count, b.prims.count)
    var worst: Float = 0
    for i in 0..<min(a.prims.count, b.prims.count) {
        let p = a.prims[i]
        let q = b.prims[i]
        worst = max(worst, simd_reduce_max(simd_abs(p.r0 - q.r0)))
        worst = max(worst, simd_reduce_max(simd_abs(p.r1 - q.r1)))
        worst = max(worst, simd_reduce_max(simd_abs(p.r2 - q.r2)))
        worst = max(worst, simd_reduce_max(simd_abs(p.c - q.c)))
    }
    // The phase is reduced to a whole number of frames BEFORE it becomes an
    // angle, so this is not "close", it is identical.
    expect(worst < 1e-6, "worst drift around the loop: \(worst)")
    expectEqual(worst, 0)
}

test("every snow particle is back where it started at frame 120") {
    var worst: Float = 0
    for k in 0..<(snowParticleCount + flocParticleCount) {
        let a: SIMD3<Float> = snowPosition(k, frame: 0, mutations: mutations)
        let b: SIMD3<Float> = snowPosition(k, frame: mysisFrameCount, mutations: mutations)
        worst = max(worst, simd_distance(a, b))
    }
    expect(worst < 1e-3, "worst particle drift around the loop: \(worst) µm")
    // ...and they really do move in between.
    let mid: SIMD3<Float> = snowPosition(3, frame: 0, mutations: mutations)
    let later: SIMD3<Float> = snowPosition(3, frame: 17, mutations: mutations)
    expect(simd_distance(mid, later) > 100, "the snow did not move at all")
}

test("six consecutive lags, each 2π/6, summing to 2π") {
    // This, and not "the phases span 2π", is what one metachronal wave on six
    // limbs means. Six evenly spaced limbs span 5/6 of a turn from the first to
    // the last; it is the SIXTH step, from the hindmost round to the foremost,
    // that closes the circle.
    let lags: [Float] = (0..<beatingPairs).map { b -> Float in
        let here: Float = exopodPhase(b, frame: 3, mutations: mutations)
        let ahead: Float = exopodPhase((b + 1) % beatingPairs, frame: 3, mutations: mutations)
        var d: Float = ahead - here
        while d < 0 { d += 2 * Float.pi }
        while d >= 2 * Float.pi { d -= 2 * Float.pi }
        return d
    }
    let total: Float = lags.reduce(0, +)
    expect(near(total, 2 * Float.pi, within: 1e-4), "the six lags sum to \(total), not 2π")
    for (b, d) in lags.enumerated() {
        expect(near(d, 2 * Float.pi / Float(beatingPairs), within: 1e-5),
               "lag \(b) is \(d), not 2π/6")
    }
}

test("the wave runs posterior → anterior, which is adlocomotory") {
    for b in 0..<(beatingPairs - 1) {
        let front: Float = exopodPhase(b, frame: 5, mutations: mutations)
        let back: Float = exopodPhase(b + 1, frame: 5, mutations: mutations)
        expect(back > front, "exopod \(b + 1) should lead exopod \(b): \(back) vs \(front)")
    }
    let spread: Float = exopodPhase(beatingPairs - 1, frame: 5, mutations: mutations)
        - exopodPhase(0, frame: 5, mutations: mutations)
    let expected: Float = Float(beatingPairs - 1) * 2 * Float.pi / Float(beatingPairs)
    expect(near(spread, expected, within: 1e-5), "front-to-back spread \(spread)")
}

test("no two exopods are ever in the same phase") {
    for f in 0..<framesPerCycle {
        var phases: [Float] = (0..<beatingPairs).map { b -> Float in
            var p: Float = exopodPhase(b, frame: f, mutations: mutations)
            while p < 0 { p += 2 * Float.pi }
            return p.truncatingRemainder(dividingBy: 2 * Float.pi)
        }
        phases.sort()
        for i in 1..<phases.count {
            expect(phases[i] - phases[i - 1] > 0.5,
                   "frame \(f): two exopods at \(phases[i - 1]) and \(phases[i])")
        }
    }
}

test("nothing on the animal jumps between frames") {
    func animalOnly(_ p: MysisPose) -> Int { p.snowPrims.first ?? p.prims.count }
    var previous = poseMysis(frame: 0, posture: posture, mutations: mutations)
    var worst: Float = 0
    for f in 1...mysisFrameCount {
        let now = poseMysis(frame: f, posture: posture, mutations: mutations)
        let n: Int = min(animalOnly(previous), animalOnly(now))
        for i in 0..<n {
            let a = SIMD3<Float>(previous.prims[i].c.x, previous.prims[i].c.y,
                                 previous.prims[i].c.z)
            let b = SIMD3<Float>(now.prims[i].c.x, now.prims[i].c.y, now.prims[i].c.z)
            worst = max(worst, simd_distance(a, b))
        }
        previous = now
    }
    // An exopod tip sweeps about 2.6 mm per half stroke in 10 frames: roughly
    // 260 µm a frame. Past 600 µm would be a jump, not a beat.
    expect(worst < 600, "biggest single-frame move \(worst) µm")
}

// MARK: - the camera

section("the camera, which is the whole premise")

test("the camera basis at frame 0 is bit-identical to frame 119") {
    let a: Camera = mysisCamera(frame: 0)
    let b: Camera = mysisCamera(frame: mysisFrameCount - 1)
    expectEqual(a.origin, b.origin)
    expectEqual(a.forward, b.forward)
    expectEqual(a.right, b.right)
    expectEqual(a.up, b.up)
    expectEqual(a.halfWidth, b.halfWidth)
    expectEqual(a.halfHeight, b.halfHeight)
    // ...and at every frame in between, because "the camera never moves" is a
    // claim about all 120 of them and not about the two ends.
    for f in 0..<mysisFrameCount {
        let c: Camera = mysisCamera(frame: f)
        expectEqual(c.origin, a.origin)
        expectEqual(c.forward, a.forward)
        expectEqual(c.right, a.right)
        expectEqual(c.up, a.up)
    }
}

test("the camera is orthographic, so one number is the pixel scale everywhere") {
    let cam: Camera = mysisCamera()
    let bar: Float = 5000 / micronsPerPixelM
    // Measured off the camera itself, at the centre of the field and at a
    // corner of it: an orthographic camera gives the same answer at both.
    let a = cam.project(SIMD3(0, 0, 0), width: mFrameWidth, height: mFrameViewHeight)
    let b = cam.project(SIMD3(5000, 0, 0), width: mFrameWidth, height: mFrameViewHeight)
    expect(near(simd_distance(a, b), bar, within: 0.01),
           "5 mm measures \(simd_distance(a, b)) px, not \(bar)")
    let c = cam.project(SIMD3(-9000, 6000, -12_000), width: mFrameWidth, height: mFrameViewHeight)
    let d = cam.project(SIMD3(-4000, 6000, -12_000), width: mFrameWidth, height: mFrameViewHeight)
    expect(near(simd_distance(c, d), bar, within: 0.01),
           "5 mm at the edge and 12 mm deep measures \(simd_distance(c, d)) px")
}

test("the posture's three axes are orthonormal and right-handed") {
    let (a, d, l) = posture.axes()
    for v in [a, d, l] { expect(near(simd_length(v), 1, within: 1e-5), "not unit: \(v)") }
    expect(abs(simd_dot(a, d)) < 1e-5, "anterior·dorsal = \(simd_dot(a, d))")
    expect(abs(simd_dot(a, l)) < 1e-5, "anterior·left = \(simd_dot(a, l))")
    expect(abs(simd_dot(d, l)) < 1e-5, "dorsal·left = \(simd_dot(d, l))")
    let cross: SIMD3<Float> = simd_cross(a, d)
    expect(simd_distance(cross, l) < 1e-5, "left is not anterior × dorsal")
    // The composition the shot asked for: head to the upper left, tail to the
    // lower right, and 35° below the horizontal.
    expect(a.x < 0 && a.y > 0, "the head does not point up and to the left: \(a)")
    let slope: Float = atan2(a.y, -a.x) * 180 / .pi
    expect(near(slope, 35, within: 0.5), "the body runs at \(slope)°, not 35°")
    // And the consequence, stated rather than hidden: with the dorsum up and
    // the head to the left, it is the RIGHT flank that faces the camera.
    expect(l.z < 0, "the animal's left side faces the camera, which cannot be")
    expect(d.y > 0, "the dorsum does not point up")
}

// MARK: - the animal

section("the animal, which is a claim about a real species")

let pose = poseMysis(frame: 7, posture: posture, mutations: mutations)

test("the body plan is the published one") {
    expectEqual(pose.counts["thoracopod pair"] ?? 0, 8)
    expectEqual(pose.counts["maxilliped pair"] ?? 0, 2)
    expectEqual(pose.counts["beating exopod"] ?? 0, 12)      // 6 pairs
    expectEqual(pose.counts["pleonite"] ?? 0, 6)
    expectEqual(pose.counts["compound eye"] ?? 0, 2)
    expectEqual(pose.counts["oostegite pair"] ?? 0, 3)
    expectEqual(pose.counts["uropod"] ?? 0, 4)
    expectEqual(pose.counts["uropod endopod"] ?? 0, 2)
    expectEqual(pose.counts["telson"] ?? 0, 2)               // the two halves of a cleft telson
    // Six beating pairs is eight thoracopod pairs less the two maxilliped
    // pairs, and the metachrony has to be built on that same six.
    expectEqual(beatingPairs, thoracopodPairs - maxillipedPairs)
    expectEqual(pose.exopodLobes.count, 2 * beatingPairs)
}

test("exactly two statocysts, one in each uropod endopod") {
    // The diagnostic character of the order. Not one, not three, and not in the
    // exopod.
    let beads: [Int] = pose.prims.indices.filter {
        pose.prims[$0].tissue == MTissue.statolith.rawValue
    }
    // Three nested shells apiece — the concentric growth layers — so nine
    // primitives would be wrong and six is right.
    expectEqual(beads.count, 3 * uropodStatocysts)
    expectEqual(pose.statolith.count, uropodStatocysts)
    expectEqual(pose.uropodEndopod.count, 2)

    for (side, bead) in pose.statolith.enumerated() {
        let centre = SIMD3<Float>(pose.prims[bead].c.x, pose.prims[bead].c.y,
                                  pose.prims[bead].c.z)
        let endopod: GPUPrim = pose.prims[pose.uropodEndopod[side]]
        expect(cpuContains(endopod, point: centre),
               "statocyst \(side) is not inside its uropod endopod")
        // And it is inside ITS OWN endopod, not the other one.
        let other: GPUPrim = pose.prims[pose.uropodEndopod[1 - side]]
        expect(!cpuContains(other, point: centre),
               "statocyst \(side) is inside both endopods at once")
    }
    // A statolith is a bead, not a plate: all three semi-axes within a percent.
    for bead in beads {
        let b = pose.prims[bead].bounds()
        let e: SIMD3<Float> = b.hi - b.lo
        expect(near(e.x, e.y, within: 1) && near(e.y, e.z, within: 1),
               "a statolith measures \(e), which is not a bead")
    }
}

test("the animal is 25 mm from the rostrum to the telson") {
    var lo: Float = .greatestFiniteMagnitude
    var hi: Float = -.greatestFiniteMagnitude
    let anterior: SIMD3<Float> = posture.axes().anterior
    let snowStart: Int = pose.snowPrims.first ?? pose.prims.count
    for i in 0..<snowStart {
        let b = pose.prims[i].bounds()
        for k in 0..<8 {
            let corner = SIMD3<Float>(k & 1 == 0 ? b.lo.x : b.hi.x,
                                      k & 2 == 0 ? b.lo.y : b.hi.y,
                                      k & 4 == 0 ? b.lo.z : b.hi.z)
            let along: Float = simd_dot(corner, anterior)
            lo = min(lo, along)
            hi = max(hi, along)
        }
    }
    // Tip to tip includes the antennal flagella and the uropods, which a
    // measured "body length" does not. The body proper is the constant.
    let plan: Float = (rostrumTipX - telsonApexX) / 1000
    expect(near(plan, 25, within: 0.001), "the body plan is \(plan) mm, not 25")
    let tipToTip: Float = (hi - lo) / 1000
    expect(tipToTip > 25 && tipToTip < 48,
           "tip to tip, antennae and all, measures \(tipToTip) mm")
}

test("every primitive has a known tissue and a real size") {
    var unknown = 0
    var degenerate = 0
    for p in pose.prims {
        if p.tissue < 0 || p.tissue >= mTissueCount { unknown += 1 }
        let b = p.bounds()
        let e: SIMD3<Float> = b.hi - b.lo
        if !(e.x > 0 && e.y > 0 && e.z > 0) { degenerate += 1 }
        if e.x.isNaN || e.y.isNaN || e.z.isNaN { degenerate += 1 }
    }
    expectEqual(unknown, 0)
    expectEqual(degenerate, 0)
    expect(pose.prims.count <= mailboxCapacity,
           "\(pose.prims.count) primitives against a \(mailboxCapacity)-bit mailbox")
    expect(pose.prims.count > 350, "only \(pose.prims.count) primitives")
}

test("the marsupium is under the thorax, and it is carrying something") {
    expect(pose.counts["embryo"] ?? 0 > 6, "an empty brood pouch")
    let (_, dorsal, _) = posture.axes()
    let pouch: GPUPrim = pose.prims[pose.marsupium[0]]
    let pouchCentre = SIMD3<Float>(pouch.c.x, pouch.c.y, pouch.c.z)
    // Ventral of the body axis: the pouch hangs BELOW, which is the whole
    // reason it is visible from this angle at all.
    expect(simd_dot(pouchCentre, dorsal) < -600,
           "the pouch sits \(simd_dot(pouchCentre, dorsal)) µm dorsally, so it is on top")
    var inside = 0
    for i in pose.prims.indices where pose.prims[i].tissue == MTissue.embryo.rawValue {
        let c = SIMD3<Float>(pose.prims[i].c.x, pose.prims[i].c.y, pose.prims[i].c.z)
        if cpuContains(pouch, point: c) { inside += 1 }
    }
    expect(inside >= (pose.counts["embryo"] ?? 0) - 2,
           "only \(inside) of \(pose.counts["embryo"] ?? 0) embryos are in the pouch")
}

test("neighbouring exopods never pass through each other, at any of the 120 frames") {
    var worst: Float = .greatestFiniteMagnitude
    var worstFrame = -1
    for f in 0..<mysisFrameCount {
        let p = poseMysis(frame: f, posture: posture, mutations: mutations)
        // The twelve exopods alternate left, right, left, right down the body,
        // so a limb's neighbour on its own side is two along.
        for i in 0..<(p.exopodLobes.count - 2) {
            let a: GPUPrim = p.prims[p.exopodLobes[i][0]]
            let b: GPUPrim = p.prims[p.exopodLobes[i + 2][0]]
            let ca = SIMD3<Float>(a.c.x, a.c.y, a.c.z)
            let cb = SIMD3<Float>(b.c.x, b.c.y, b.c.z)
            let gap: Float = simd_distance(ca, cb)
            if gap < worst { worst = gap; worstFrame = f }
        }
    }
    expect(worst > 300, "two exopod paddles came within \(worst) µm at frame \(worstFrame)")
}

// MARK: - the snow

section("the snow, which only ever falls")

test("no particle's vertical velocity is ever positive") {
    // The prescribed field, not a difference of positions: the conveyor is
    // arithmetic, and this is the arithmetic.
    for k in 0..<(snowParticleCount + flocParticleCount) {
        for f in 0..<mysisFrameCount {
            let v: Float = snowVelocityY(particle: k, frame: f)
            expect(v < 0, "particle \(k) at frame \(f) has vertical velocity \(v)")
        }
    }
    expect(snowSpeedMicronsPerSecond > 0, "the field is not moving")
}

test("a particle either falls or respawns at the top — never rises a little") {
    var respawns = 0
    var falls = 0
    for k in 0..<(snowParticleCount + flocParticleCount) {
        for f in 0..<mysisFrameCount {
            let a: SIMD3<Float> = snowPosition(k, frame: f)
            let b: SIMD3<Float> = snowPosition(k, frame: f + 1)
            let dy: Float = b.y - a.y
            if dy < 0 {
                falls += 1
                // One frame of fall, exactly.
                let step: Float = snowSpanMicrons / Float(mysisFrameCount)
                expect(near(-dy, step, within: 0.5), "particle \(k) fell \(-dy), not \(step)")
            } else {
                respawns += 1
                // A respawn is a whole span, not a nudge: the particle leaving
                // the bottom is REPLACED at the top, it does not drift back.
                expect(dy > snowSpanMicrons * 0.9,
                       "particle \(k) rose \(dy) µm at frame \(f), which is neither")
            }
            // Nothing moves sideways: the field is vertical.
            expectEqual(a.x, b.x)
            expectEqual(a.z, b.z)
        }
    }
    expect(falls > respawns * 20, "\(respawns) respawns against \(falls) falls")
}

test("the snow does not turn with the animal") {
    // It is in world coordinates, so changing the posture must not move it.
    var other = MysisPosture()
    other.roll = radians(40)
    let a = poseMysis(frame: 9, posture: posture)
    let b = poseMysis(frame: 9, posture: other)
    expectEqual(a.snowPrims.count, b.snowPrims.count)
    for k in 0..<a.snowPrims.count {
        let p = a.prims[a.snowPrims[k]]
        let q = b.prims[b.snowPrims[k]]
        expectEqual(p.c, q.c)
    }
}

// MARK: - darkfield, against closed forms

section("darkfield, against closed forms")

test("black stays black: a ray that hits nothing is exactly zero") {
    let out = cpuScatter(origin: SIMD3(0, 0, 1000), direction: SIMD3(0, 0, -1),
                         prims: [], sigma: oneSigma, mutations: mutations)
    expectEqual(out.intensity, SIMD3<Float>(0, 0, 0))
    expectEqual(out.interfaces, 0)
    // And the tone curve keeps it there: 1 − exp(−0) is exactly 1 − 1.
    expectEqual(darkfieldTone(.zero), SIMD3<Float>(0, 0, 0))
    // There is no background term anywhere to leak: I_bg is not small, it is
    // absent.
    let farAway = GPUPrim.sph(SIMD3(0, 0, -900_000), 10, .cuticle)
    let out2 = cpuScatter(origin: SIMD3(50_000, 50_000, 1000), direction: SIMD3(0, 0, -1),
                          prims: [farAway], sigma: oneSigma, mutations: mutations)
    expectEqual(out2.intensity, SIMD3<Float>(0, 0, 0))
}

test("the deposit is at the interfaces, so a sphere is worth exactly 2σg") {
    let r: Float = 400
    let s = GPUPrim.sph(SIMD3(11, -23, 5), r, .cuticle)
    // Straight through the middle the ray meets both surfaces head on, so the
    // grazing weight is zero and the sphere is invisible. That is not a bug,
    // it is what darkfield does to a smooth sphere's middle.
    let middle = scatterThrough(SIMD3(11, -23, 5), [s], oneSigma)
    expectEqual(middle.interfaces, 2)
    if !mutations.contains(.flatGrazing) {
        for k in 0..<3 {
            expect(middle.intensity[k] < 1e-6,
                   "the middle of a sphere scattered \(middle.intensity[k])")
        }
    }
    // Off to the side, the closed form: |n·v| = √(1 − b²/r²) at both surfaces.
    let b: Float = 0.8 * r
    let off = scatterThrough(SIMD3(11 + b, -23, 5), [s], oneSigma)
    let cosine: Float = (1 - (b * b) / (r * r)).squareRoot()
    let g: Float = mutations.contains(.flatGrazing)
        ? 1 : pow(1 - cosine, grazingExponent)
    for k in 0..<3 {
        let expected: Float = 2 * oneSigma[MTissue.cuticle.rawValue][k] * g
        expect(near(off.intensity[k], expected, within: 1e-4),
               "channel \(k): \(off.intensity[k]) against \(expected)")
    }
}

test("doubling a body's thickness does not change what it scatters") {
    // The claim the interface model makes: what a ray collects is the SURFACES
    // it crossed, not how far it travelled between them. Two equal spheres
    // pushed apart along the ray make a thicker body with the same two
    // surfaces at the same angles — so the answer must not move.
    let r: Float = 500
    let b: Float = 0.7 * r
    let one = [GPUPrim.sph(.zero, r, .cuticle)]
    let thin = scatterThrough(SIMD3(b, 0, 0), one, oneSigma)
    for gap in [Float(0), 300, 700] {
        let fat = [GPUPrim.sph(SIMD3(0, 0, gap * 0.5), r, .cuticle),
                   GPUPrim.sph(SIMD3(0, 0, -gap * 0.5), r, .cuticle)]
        let out = scatterThrough(SIMD3(b, 0, 0), fat, oneSigma)
        expectEqual(out.interfaces, thin.interfaces)
        for k in 0..<3 {
            expect(near(out.intensity[k], thin.intensity[k], within: 1e-4),
                   "gap \(gap), channel \(k): \(out.intensity[k]) vs \(thin.intensity[k])")
        }
    }
}

test("doubling the number of shells does change it") {
    let r: Float = 500
    let b: Float = 0.7 * r
    let outer = GPUPrim.sph(.zero, r, .cuticle)
    let inner = GPUPrim.sph(.zero, r * 0.9, .innerWall)
    let alone = scatterThrough(SIMD3(b, 0, 0), [outer], oneSigma)
    let nested = scatterThrough(SIMD3(b, 0, 0), [outer, inner], oneSigma)
    expectEqual(alone.interfaces, 2)
    expectEqual(nested.interfaces, 4)
    for k in 0..<3 {
        expect(nested.intensity[k] > alone.intensity[k] + 1e-5,
               "channel \(k): a second shell added \(nested.intensity[k] - alone.intensity[k])")
    }
    // And a shell of the SAME tissue nested inside adds nothing at all,
    // because it is not a surface — it is inside a solid.
    let sameTissue = GPUPrim.sph(.zero, r * 0.9, .cuticle)
    let swallowed = scatterThrough(SIMD3(b, 0, 0), [outer, sameTissue], oneSigma)
    if !mutations.contains(.noUnion) {
        expectEqual(swallowed.interfaces, 2)
        for k in 0..<3 {
            expect(near(swallowed.intensity[k], alone.intensity[k], within: 1e-5),
                   "a swallowed shell changed channel \(k) by "
                   + "\(swallowed.intensity[k] - alone.intensity[k])")
        }
    }
}

test("union, not sum: two coincident shells scatter as one") {
    let r: Float = 450
    let b: Float = 0.6 * r
    let a = GPUPrim.sph(.zero, r, .cuticle)
    let c = GPUPrim.sph(.zero, r, .cuticle)
    let one = scatterThrough(SIMD3(b, 0, 0), [a], oneSigma)
    let two = scatterThrough(SIMD3(b, 0, 0), [a, c], oneSigma)
    expectEqual(two.interfaces, one.interfaces)
    for k in 0..<3 {
        expect(near(two.intensity[k], one.intensity[k], within: 1e-5),
               "two coincident shells gave \(two.intensity[k]) where one gives \(one.intensity[k])")
    }
    // Different tissues in the same place do NOT merge: σ_s is a property of
    // the tissue and two tissues really are two boundaries.
    let d = GPUPrim.sph(.zero, r, .innerWall)
    let mixed = scatterThrough(SIMD3(b, 0, 0), [a, d], oneSigma)
    expectEqual(mixed.interfaces, 4)
}

test("grazing: the brightest part of a sphere is its silhouette") {
    let r: Float = 600
    let s = [GPUPrim.sph(.zero, r, .cuticle)]
    var previous: Float = -1
    var profile: [Float] = []
    for i in 0...40 {
        let b: Float = r * Float(i) / 41
        let out = scatterThrough(SIMD3(b, 0, 0), s, oneSigma)
        profile.append(out.intensity.y)
        if mutations.contains(.flatGrazing) {
            // The mutation makes every interface scatter the same, and this is
            // exactly where that shows: a disc instead of a ring.
            expect(out.intensity.y > previous,
                   "the radial profile is flat at b = \(b): \(out.intensity.y)")
        } else {
            expect(out.intensity.y > previous,
                   "the profile is not monotone outward at b = \(b): "
                   + "\(out.intensity.y) after \(previous)")
        }
        previous = out.intensity.y
    }
    // The outermost sample is the brightest, by a long way.
    let brightest: Float = profile.max() ?? 0
    expectEqual(profile.last!, brightest)
    expect(profile.last! > profile[profile.count / 2] * 3,
           "the rim is only \(profile.last! / max(profile[profile.count / 2], 1e-9))× the middle")
}

test("blue scatters more than red at every cuticle interface") {
    // Rayleigh: σ_s ∝ λ⁻⁴, so this is not a palette choice, it is the
    // exponent. Every unpigmented surface on the animal has to obey it.
    for t in MTissue.allCases where t.isCuticle {
        guard let s = scatterers[t] else { expect(false, "no σ_s for \(t.name)"); continue }
        let v: SIMD3<Float> = s.sigma
        expect(v.z > v.y && v.y > v.x,
               "\(t.name) scatters \(v), which is not blue > green > red")
        let ratio: Float = v.z / v.x
        expect(near(ratio, pow(612.0 / 465.0, 4), within: 1e-3),
               "\(t.name) has blue/red = \(ratio), not (612/465)⁴")
    }
    // The pigmented tissues are allowed to break it, and the orange one must.
    let orange: SIMD3<Float> = scatterers[.hepatopancreas]!.sigma
    expect(orange.x > orange.y && orange.y > orange.z,
           "the hepatopancreas scatters \(orange), which is not orange")
    // The statolith is a crystal far bigger than the wavelength, so it is
    // nearly neutral — that is what makes it read as a hard white point.
    let bead: SIMD3<Float> = scatterers[.statolith]!.sigma
    expect(bead.z / bead.x < 1.15, "the statolith is tinted: \(bead)")
    expect(bead.y > orange.y * 2, "the statolith is not the brightest thing here")
}

test("the wavelength law is a law and not a lookup") {
    expectEqual(rayleighWeights(exponent: 0), SIMD3<Float>(1, 1, 1))
    let w: SIMD3<Float> = rayleighWeights(exponent: 4)
    expect(near(w.y, 1, within: 1e-6), "green is not the reference: \(w.y)")
    expect(near(w.x, pow(549.0 / 612.0, 4), within: 1e-5), "red weight \(w.x)")
    expect(near(w.z, pow(549.0 / 465.0, 4), within: 1e-5), "blue weight \(w.z)")
    // Doubling the exponent squares the ratio, which is what a power law means.
    let w8: SIMD3<Float> = rayleighWeights(exponent: 8)
    expect(near(w8.z, w.z * w.z, within: 1e-4), "λ⁻⁸ is not λ⁻⁴ squared")
}

test("the tone curve cannot reorder anything the physics said") {
    var previous: Float = -1
    for i in 0...60 {
        let x: Float = Float(i) / 12
        let out: Float = darkfieldTone(SIMD3(x, x, x)).y
        expect(out > previous, "the tone curve went down at I = \(x)")
        expect(out >= 0 && out < 1, "the tone curve left [0, 1) at I = \(x): \(out)")
        previous = out
    }
}

// MARK: - the GPU

section("the GPU, the grid, the mailbox and the swept bound")

let device = try findDevice()
let renderer = try DarkfieldRenderer(device: device)
let probeSize = 65
let probeBuffer = device.makeBuffer(length: probeSize * probeSize * 4,
                                    options: .storageModeShared)!

func probeIntensity(_ prims: [GPUPrim], at centre: SIMD3<Float>, sigma table: [SIMD3<Float>],
                    useGrid: Bool, micronsPerPixel: Float = 20,
                    density: Float = 1) throws -> SIMD3<Float> {
    let cam = Camera(centre: centre, micronsPerPixel: micronsPerPixel,
                     width: probeSize, height: probeSize, standOff: 100_000)
    if useGrid { try renderer.buildGrid(prims, density: density) }
    _ = try renderer.render(prims: prims, sigma: table, camera: cam, into: probeBuffer,
                            width: probeSize, viewHeight: probeSize, samplesPerSide: 1,
                            useGrid: useGrid, mask: nil, mutations: mutations)
    return renderer.intensity(atX: probeSize / 2, y: probeSize / 2, width: probeSize)
}

test("the GPU agrees with the closed form for a sphere") {
    let r: Float = 500
    let s = GPUPrim.sph(.zero, r, .cuticle)
    // Aim 0.6 r off the centre so the grazing weight is not zero.
    let b: Float = 0.6 * r
    let gpu = try probeIntensity([s], at: SIMD3(b, 0, 0), sigma: oneSigma, useGrid: true)
    let cosine: Float = (1 - (b * b) / (r * r)).squareRoot()
    let g: Float = mutations.contains(.flatGrazing) ? 1 : pow(1 - cosine, grazingExponent)
    for k in 0..<3 {
        let expected: Float = 2 * oneSigma[MTissue.cuticle.rawValue][k] * g
        expect(near(gpu[k], expected, within: 2e-3),
               "channel \(k): GPU \(gpu[k]) against \(expected)")
    }
}

test("the GPU agrees with the CPU reference on the whole animal") {
    let p = poseMysis(frame: 7, posture: posture, mutations: mutations)
    let centre: SIMD3<Float> = p.exopodHinge[6]
    let gpu = try probeIntensity(p.prims, at: centre, sigma: sigma, useGrid: true)
    let cam = Camera(centre: centre, micronsPerPixel: 20, width: probeSize, height: probeSize,
                     standOff: 100_000)
    let (o, d) = cam.ray(sx: Float(probeSize / 2) + 0.5, sy: Float(probeSize / 2) + 0.5,
                         width: probeSize, height: probeSize)
    let cpu = cpuScatter(origin: o, direction: d, prims: p.prims, sigma: sigma,
                         mutations: mutations)
    expect(cpu.intensity.y > 0.02, "the probe ray missed the animal (I = \(cpu.intensity))")
    for k in 0..<3 {
        expect(near(gpu[k], cpu.intensity[k], within: 2e-3),
               "channel \(k): GPU \(gpu[k]) against CPU \(cpu.intensity[k])")
    }
}

test("the grid draws the same picture as testing every primitive") {
    let p = poseMysis(frame: 7, posture: posture, mutations: mutations)
    let centre: SIMD3<Float> = p.exopodHinge[4]
    let withGrid = try probeIntensity(p.prims, at: centre, sigma: sigma, useGrid: true)
    let brute = try probeIntensity(p.prims, at: centre, sigma: sigma, useGrid: false)
    for k in 0..<3 {
        expect(near(withGrid[k], brute[k], within: 1e-4),
               "channel \(k): grid \(withGrid[k]), brute force \(brute[k])")
    }
}

test("intervals per ray match the CPU reference, primitive by primitive") {
    // What the mailbox guarantees is a COUNT. The per-tissue union merges a
    // duplicated interval back into itself, so a primitive met twice does not
    // change the answer — turning the mailbox off would leave the picture
    // identical. What it changes is how fast the list fills, and a list that
    // overflows drops a surface, which in darkfield is a missing outline.
    let p = poseMysis(frame: 7, posture: posture, mutations: mutations)
    let centre: SIMD3<Float> = p.exopodHinge[6]
    _ = try probeIntensity(p.prims, at: centre, sigma: sigma, useGrid: true)
    var bx = 0
    var by = 0
    var met = 0
    for y in 0..<probeSize {
        for x in 0..<probeSize {
            let n: Int = renderer.intervalsMet(atX: x, y: y, width: probeSize)
            if n > met { met = n; bx = x; by = y }
        }
    }
    let cam = Camera(centre: centre, micronsPerPixel: 20, width: probeSize, height: probeSize,
                     standOff: 100_000)
    let (o, d) = cam.ray(sx: Float(bx) + 0.5, sy: Float(by) + 0.5,
                         width: probeSize, height: probeSize)
    let truth = cpuScatter(origin: o, direction: d, prims: p.prims, sigma: sigma,
                           mutations: mutations)
    expect(truth.intervals > 5, "the busiest probe ray crossed only \(truth.intervals) primitives")
    guard let g = renderer.grid else { expect(false, "no grid"); return }
    let boxesPer: Double = Double(g.items.count) / Double(p.prims.count)
    expect(boxesPer > 1.5,
           "the grid files each primitive in only \(boxesPer) boxes, so there is nothing to mail")
    expectEqual(met, truth.intervals)
}

test("no ray in the real render runs out of room for intervals") {
    let w = mFrameWidth
    let h = mFrameViewHeight
    guard let b = device.makeBuffer(length: w * h * 4, options: .storageModeShared) else {
        expect(false, "no buffer"); return
    }
    let cam: Camera = mysisCamera()
    var worst: UInt32 = 0
    var busiest = 0
    for f in [0, 5, 10, 15] {
        let p = poseMysis(frame: f, posture: posture, mutations: mutations)
        try renderer.buildGrid(p.prims, density: 1)
        _ = try renderer.render(prims: p.prims, sigma: sigma, camera: cam, into: b,
                                width: w, viewHeight: h, samplesPerSide: 2,
                                useGrid: true, mask: nil, mutations: mutations)
        worst = max(worst, renderer.overflowCount)
        for y in stride(from: 0, to: h, by: 3) {
            for x in stride(from: 0, to: w, by: 3) {
                busiest = max(busiest, renderer.intervalsMet(atX: x, y: y, width: w) / 4)
            }
        }
    }
    expectEqual(worst, UInt32(0))
    expect(busiest < maxSurfacesPerRay,
           "the busiest ray carried \(busiest) intervals of \(maxSurfacesPerRay)")
    expect(busiest > 8, "the busiest ray carried only \(busiest) — is the scene there?")
}

test("a pixel nothing reaches is black, byte for byte") {
    let p = poseMysis(frame: 7, posture: posture, mutations: mutations)
    let cam = Camera(centre: SIMD3(400, 600, 0), micronsPerPixel: micronsPerPixelM * 4,
                     width: 320, height: 240, standOff: cameraStandOff)
    guard let b = device.makeBuffer(length: 320 * 240 * 4, options: .storageModeShared) else {
        expect(false, "no buffer"); return
    }
    try renderer.buildGrid(p.prims, density: 1)
    _ = try renderer.render(prims: p.prims, sigma: sigma, camera: cam, into: b,
                            width: 320, viewHeight: 240, useGrid: true, mask: nil,
                            mutations: mutations)
    let bytes = Data(bytes: b.contents(), count: 320 * 240 * 4)
    // A field four times the render's is 100 mm across; the corners are a long
    // way from a 25 mm animal, and there is no lamp to leak in.
    for (x, y) in [(0, 0), (319, 0), (0, 239), (319, 239), (1, 120), (318, 120)] {
        let i: Int = (y * 320 + x) * 4
        for c in 0..<3 {
            expectEqual(bytes[i + c], UInt8(0))
        }
    }
}

// MARK: - the swept bound

section("the swept bound, which is only correct because the camera is bolted down")

let testCamera: Camera = mysisCamera()
let testSweptFrames: [Int] = mutations.contains(.frameZeroMask)
    ? [0] : Array(0..<mysisFrameCount)
let testSwept: SweptBound = buildSweptBound(frames: testSweptFrames, camera: testCamera,
                                            width: mFrameWidth, height: mFrameViewHeight) { f in
    poseMysis(frame: f, posture: posture, mutations: mutations).prims
}

test("the mask is conservative: masked and unmasked render the same pixels") {
    let w = mFrameWidth
    let h = mFrameViewHeight
    guard let a = device.makeBuffer(length: w * h * 4, options: .storageModeShared),
          let b = device.makeBuffer(length: w * h * 4, options: .storageModeShared) else {
        expect(false, "no buffer"); return
    }
    // A frame in the middle of the loop, where a mask built from frame 0 alone
    // has had the most time to stop being a bound.
    let f = 60
    let p = poseMysis(frame: f, posture: posture, mutations: mutations)
    try renderer.buildGrid(p.prims, density: 1)
    _ = try renderer.render(prims: p.prims, sigma: sigma, camera: testCamera, into: a,
                            width: w, viewHeight: h, useGrid: true, mask: nil,
                            mutations: mutations)
    let withoutMask = Data(bytes: a.contents(), count: w * h * 4)
    _ = try renderer.render(prims: p.prims, sigma: sigma, camera: testCamera, into: b,
                            width: w, viewHeight: h, useGrid: true, mask: testSwept,
                            mutations: mutations)
    let withMask = Data(bytes: b.contents(), count: w * h * 4)
    var differing = 0
    for i in 0..<(w * h) where withoutMask[i * 4] != withMask[i * 4]
        || withoutMask[i * 4 + 1] != withMask[i * 4 + 1]
        || withoutMask[i * 4 + 2] != withMask[i * 4 + 2] {
        differing += 1
    }
    expectEqual(differing, 0)
}

test("the bound really bounds, and the conveyor is what it costs") {
    // The finding, written down as a test so it cannot quietly change. A swept
    // bound over the ANIMAL keeps only a quarter of the frame alive, because
    // the animal is confined and the camera is bolted down. A swept bound over
    // the whole scene keeps nearly all of it, because a snow particle falls the
    // height of the frame and therefore marks its whole column in every one of
    // the 120 frames. The bound cannot tell the two apart, so the conveyor
    // spends the optimisation.
    let animalSwept: SweptBound = buildSweptBound(frames: Array(0..<mysisFrameCount),
                                                  camera: testCamera, width: mFrameWidth,
                                                  height: mFrameViewHeight) { f in
        animalPrims(poseMysis(frame: f, posture: posture))
    }
    expect(animalSwept.liveFraction < 0.45,
           "the animal alone keeps \(animalSwept.liveFraction * 100)% of the tiles")
    expect(animalSwept.liveFraction > 0.05,
           "the animal alone keeps only \(animalSwept.liveFraction * 100)%, which cannot be")
    if !mutations.contains(.frameZeroMask) {
        expect(testSwept.liveFraction > animalSwept.liveFraction + 0.3,
               "the snow costs only \(testSwept.liveFraction - animalSwept.liveFraction) "
               + "of the frame, so this note is out of date")
    }
    expect(testSwept.liveFraction <= 1.0, "more than every tile is alive")
    // Every primitive of every frame is inside the world box, which is what
    // makes it a bound rather than a guess.
    var outside = 0
    for f in stride(from: 0, to: mysisFrameCount, by: 7) {
        for p in poseMysis(frame: f, posture: posture).prims {
            let bb = p.bounds()
            if simd_reduce_min(bb.lo - testSwept.lo) < -1e-3 { outside += 1 }
            if simd_reduce_max(bb.hi - testSwept.hi) > 1e-3 { outside += 1 }
        }
    }
    if !mutations.contains(.frameZeroMask) { expectEqual(outside, 0) }
}

test("the changed-pixel fraction is small, which is what the GIF encoder wants") {
    // Measured, not assumed: the frozen camera and the black background are
    // supposed to leave most of the frame untouched between frames.
    let w = 320
    let h = 240
    guard let a = device.makeBuffer(length: w * h * 4, options: .storageModeShared),
          let b = device.makeBuffer(length: w * h * 4, options: .storageModeShared) else {
        expect(false, "no buffer"); return
    }
    let cam = Camera(centre: SIMD3(400, 600, 0),
                     micronsPerPixel: micronsPerPixelM * Float(mFrameWidth) / Float(w),
                     width: w, height: h, standOff: cameraStandOff)
    func shoot(_ f: Int, _ into: MTLBuffer) throws {
        let p = poseMysis(frame: f, posture: posture, mutations: mutations)
        try renderer.buildGrid(p.prims, density: 1)
        _ = try renderer.render(prims: p.prims, sigma: sigma, camera: cam, into: into,
                                width: w, viewHeight: h, useGrid: true, mask: nil,
                                mutations: mutations)
    }
    try shoot(30, a)
    let first = Data(bytes: a.contents(), count: w * h * 4)
    try shoot(31, b)
    let second = Data(bytes: b.contents(), count: w * h * 4)
    var changed = 0
    for i in 0..<(w * h) where first[i * 4] != second[i * 4]
        || first[i * 4 + 1] != second[i * 4 + 1] || first[i * 4 + 2] != second[i * 4 + 2] {
        changed += 1
    }
    let fraction: Double = Double(changed) / Double(w * h)
    expect(fraction > 0, "nothing changed between two frames, so nothing is moving")
    expect(fraction < 0.40,
           String(format: "%.1f%% of pixels changed between frames", fraction * 100))
}

test("the two statocysts are the brightest things anywhere near them") {
    let p = poseMysis(frame: 7, posture: posture, mutations: mutations)
    expectEqual(p.statolith.count, 2)
    let beads: [SIMD3<Float>] = p.statolith.map { i -> SIMD3<Float> in
        SIMD3(p.prims[i].c.x, p.prims[i].c.y, p.prims[i].c.z)
    }
    let anchor: SIMD3<Float> = (beads[0] + beads[1]) * 0.5
    let size = 161
    guard let b = device.makeBuffer(length: size * size * 4, options: .storageModeShared) else {
        expect(false, "no buffer"); return
    }
    // A 3.2 mm field centred between the two beads: both of them are in it, and
    // so are the uropod blades, the telson setae and the cuticle around them.
    let cam = Camera(centre: anchor, micronsPerPixel: 20, width: size, height: size,
                     standOff: cameraStandOff)
    try renderer.buildGrid(p.prims, density: 1)
    _ = try renderer.render(prims: p.prims, sigma: sigma, camera: cam, into: b,
                            width: size, viewHeight: size, samplesPerSide: 1,
                            useGrid: true, mask: nil, mutations: mutations)
    // Both beads land in this field — the animal is nearly side on, so the far
    // uropod's statocyst projects about 20 px from the near one's — so "near
    // the statocysts" is the disc that holds the pair.
    var onTheBeads: Float = -1
    var around: Float = -1
    var best: Float = -1
    var bx = 0
    var by = 0
    for y in 0..<size {
        for x in 0..<size {
            let v: SIMD3<Float> = renderer.intensity(atX: x, y: y, width: size)
            let lum: Float = v.x + v.y + v.z
            let dx: Float = Float(x) - Float(size / 2)
            let dy: Float = Float(y) - Float(size / 2)
            let d: Float = (dx * dx + dy * dy).squareRoot()
            if d < 32 { onTheBeads = max(onTheBeads, lum) }
            if d > 42 && d < 74 { around = max(around, lum) }
            if lum > best { best = lum; bx = x; by = y }
        }
    }
    let dx: Float = Float(bx) - Float(size / 2)
    let dy: Float = Float(by) - Float(size / 2)
    let r: Float = (dx * dx + dy * dy).squareRoot()
    expect(r < 32, "the brightest pixel is \(r) px from the statocysts, not on them")
    // The claim on the bar: two hard points brighter than anything near them.
    expect(onTheBeads > around * 2,
           "the beads reach \(onTheBeads) and their surroundings \(around)")
    expect(best > 5, "the statocysts only reached \(best)")
}

// MARK: - the numbers on the bar

section("what the caption bar claims")

test("every constant carries a source") {
    expect(mysisConstants.count >= 30, "only \(mysisConstants.count) constants recorded")
    var missing: [String] = []
    for c in mysisConstants {
        if c.source.trimmingCharacters(in: .whitespaces).isEmpty { missing.append(c.name) }
        if c.name.trimmingCharacters(in: .whitespaces).isEmpty { missing.append("(unnamed)") }
        if !c.value.isFinite { missing.append("\(c.name): not a number") }
    }
    expect(missing.isEmpty, "no source for: \(missing)")
    for c in mysisConstants where c.evidence == .model {
        expect(c.source.hasPrefix("MODEL"), "\(c.name) is a model but does not say so")
    }
    for c in mysisConstants where c.evidence == .derived {
        expect(c.source.hasPrefix("DERIVED"), "\(c.name) is derived but does not say so")
    }
    // A MEASURED line has to name a year or a taxon — something a reader could
    // go and look up. "It is well known" is not a source.
    func hasYear(_ text: String) -> Bool {
        let digits: [Character] = Array(text)
        for i in 0..<max(digits.count - 3, 0) {
            let four = digits[i..<(i + 4)]
            if four.allSatisfy({ $0.isNumber }) && (four.first == "1" || four.first == "2") {
                return true
            }
        }
        return false
    }
    for c in mysisConstants where c.evidence == .measured {
        let cites: Bool = hasYear(c.source) || c.source.contains("Mysida")
            || c.source.contains("fresh water")
        expect(cites, "\(c.name) claims to be measured but cites nothing: \(c.source)")
    }
}

test("the table really is the table the render uses") {
    func value(_ name: String) -> Double? {
        mysisConstants.first { $0.name == name }?.value
    }
    expectEqual(value("pairs of thoracopods"), Double(thoracopodPairs))
    expectEqual(value("beating exopod pairs"), Double(beatingPairs))
    expectEqual(value("statocysts"), Double(pose.statolith.count))
    expectEqual(value("exopod beat frequency"), Double(beatFrequencyHz))
    expectEqual(value("grazing exponent k"), Double(grazingExponent))
    expectEqual(value("Reynolds number, body"), reynolds.body)
    expectEqual(value("kinematic viscosity of the medium"), kinematicViscosity)
}

test("the medium is fresh water, not step 13's seawater") {
    // Mysis diluviana is a Great Lakes animal. Copying step 13's 1.05e-6 m²/s
    // would be quoting the viscosity of the sea for a lake.
    expect(near(kinematicViscosity, 1.57e-6, within: 1e-8),
           "ν = \(kinematicViscosity) m²/s")
    expect(kinematicViscosity > 1.3e-6,
           "ν = \(kinematicViscosity) is a warm-water number, not a 4 °C one")
    let expected: Double = (Double(snowSpeedMicronsPerSecond) * 1e-6 * 25e-3) / 1.57e-6
    expect(near(reynolds.body, expected, within: 0.5), "Re = \(reynolds.body)")
    expect(reynolds.body > 150 && reynolds.body < 260, "Re ≈ 200 means \(reynolds.body)")
    expect(reynolds.exopod < reynolds.body, "the exopod should be gentler than the body")
    expect(reynolds.seta < 1, "the setae live at Re = \(reynolds.seta), which must be below 1")
}

test("the timing is 120 frames, 6 beats, 9.6 s shown for 1.67 s lived") {
    expectEqual(mysisFrameCount, 120)
    expectEqual(framesPerCycle * cyclesPerLoop, mysisFrameCount)
    expect(near(loopDisplayedSeconds, 9.6, within: 1e-9), "\(loopDisplayedSeconds) s displayed")
    let beats: Double = Double(beatFrequencyHz) * loopRealSeconds
    expect(near(beats, Double(cyclesPerLoop), within: 1e-9), "\(beats) beats in the loop")
    expect(near(loopDisplayedSeconds / loopRealSeconds, slowMotionFactor, within: 1e-9))
    expect(frameDelayCentiseconds == 8, "\(frameDelayCentiseconds) cs a frame")
}

test("the beat frequency is what the size trend says, not what feels right") {
    // Two anchors: 10 mm Artemia at 5 Hz, 40 mm krill hovering at 3 Hz. A power
    // law f = A·L^−α through them, evaluated at 25 mm.
    let alpha: Double = log(5.0 / 3.0) / log(40.0 / 10.0)
    let predicted: Double = 3.0 * pow(40.0 / 25.0, alpha)
    expect(near(Double(beatFrequencyHz), predicted, within: 0.05),
           "the trend gives \(predicted) Hz and the render uses \(beatFrequencyHz)")
    // And the thing worth saying out loud: it is SLOWER than the brine shrimp,
    // not faster.
    expect(Double(beatFrequencyHz) < 5.0,
           "the trend does not make a 25 mm mysid beat faster than a 10 mm Artemia")
}

test("the caption bar fits inside the frame") {
    let layout = FrameLayout(width: mFrameWidth, viewHeight: mFrameViewHeight,
                             captionHeight: mFrameCaptionHeight)
    let k = layout.scale
    let caption = mysisCaption()
    // No evidence bar on the right any more, so the three lines have the whole
    // width — which the title needs, because "Rendered model of…" is a good
    // deal longer than what it replaced.
    let left: CGFloat = 18 * k
    let right: CGFloat = CGFloat(layout.width) - 18 * k
    for (s, size, bold) in [(caption.title, 16 * k, true), (caption.subtitle, 12 * k, false),
                            (caption.facts, 11.5 * k, false)] {
        let w = textWidth(s, size: size, bold: bold)
        expect(left + w < right, "\"\(s)\" runs to \(left + w) in a \(layout.width) px frame")
    }
    // drawCaption puts the facts line at barTop + 54k at 11.5k.
    let bottom: CGFloat = 54 * k + 11.5 * k
    expect(bottom < CGFloat(layout.captionHeight), "the last line reaches \(bottom)")
}

test("the labels point at things that are actually in the picture") {
    let cam: Camera = mysisCamera()
    for (name, anchor) in pose.labelAnchors {
        let p: SIMD2<Float> = cam.project(anchor, width: mFrameWidth, height: mFrameViewHeight)
        expect(p.x > 0 && p.x < Float(mFrameWidth),
               "the \(name) anchor is at x = \(p.x), off a \(mFrameWidth) px frame")
        expect(p.y > 0 && p.y < Float(mFrameViewHeight),
               "the \(name) anchor is at y = \(p.y), off a \(mFrameViewHeight) px view")
    }
    expect(pose.labelAnchors["statocyst"] != nil, "nothing points at the statocyst")
    expect(pose.labelAnchors["marsupium"] != nil, "nothing points at the marsupium")
}

test("five millimetres of scale bar is five millimetres of animal") {
    let bar: Float = 5000 / micronsPerPixelM
    expect(near(bar, 256.41, within: 0.01), "the 5 mm bar is \(bar) px")
    expect(bar < Float(mFrameWidth) / 3, "the bar takes up a third of the frame")
}

finish()
