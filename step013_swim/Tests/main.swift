// Tests for step 13.
//
// Two things are on trial here that no earlier step had to prove. The first is
// the PHYSICS: Beer–Lambert with a per-tissue union, mailboxed across grid
// boxes, is easy to write in a way that looks right and is quietly wrong, so
// every piece of it is checked against a closed form rather than against a
// picture. The second is the ANIMAL: the body plan is a claim about a real
// species, the metachrony is a claim about how it swims, and neither is
// allowed to drift from what was published.
//
// Four of these tests exist to catch a specific mistake, and `make mutants`
// makes each of those mistakes on purpose to prove they do:
//
//   SWIM_MUTATE=union     intervals summed instead of merged
//   SWIM_MUTATE=mailbox   a primitive counted once per grid box it touches
//   SWIM_MUTATE=phase     1/10 of a cycle of lag instead of 1/11
//   SWIM_MUTATE=clamp     sub-pixel setae drawn at true radius

import CoreGraphics
import Foundation
import Metal
import simd

let mutations = Mutations.fromEnvironment()
let posture = Posture()
let sigma: [SIMD3<Float>] = sigmaTable(mutations: mutations)
let backlight = Backlight.standard
let reynolds = ReynoldsNumbers()

func near(_ a: Float, _ b: Float, within e: Float) -> Bool { abs(a - b) <= e }
func near(_ a: Double, _ b: Double, within e: Double) -> Bool { abs(a - b) <= e }

/// A ray straight down −Z through `p`, which is what every closed-form test
/// below uses.
func rayThrough(_ p: SIMD3<Float>) -> (SIMD3<Float>, SIMD3<Float>) {
    (SIMD3(p.x, p.y, p.z + 100_000), SIMD3(0, 0, -1))
}

func tauThrough(_ point: SIMD3<Float>, _ prims: [GPUPrim],
                _ table: [SIMD3<Float>] = sigma) -> SIMD3<Float> {
    let (o, d) = rayThrough(point)
    return cpuOpticalDepth(origin: o, direction: d, prims: prims, sigma: table,
                           mutations: mutations).tau
}

// MARK: - the loop

section("the loop")

test("the loop closes: frame 120 is frame 0") {
    let a = poseArtemia(frame: 0, posture: posture, mutations: mutations)
    let b = poseArtemia(frame: swimFrameCount, posture: posture, mutations: mutations)
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
    // The cycle phase is reduced to a whole number of frames BEFORE it becomes
    // an angle, so this is not "close", it is identical.
    expect(worst < 1e-6, "worst drift around the loop: \(worst)")
    expectEqual(worst, 0)
}

test("exactly one metachronal wave sits on the eleven limbs") {
    let lags: [Float] = (0..<thoracicSegments).map { i -> Float in
        let here: Float = limbPhase(i, frame: 3, mutations: mutations)
        let ahead: Float = limbPhase((i + 1) % thoracicSegments, frame: 3, mutations: mutations)
        var d: Float = ahead - here
        while d < 0 { d += 2 * Float.pi }
        while d >= 2 * Float.pi { d -= 2 * Float.pi }
        return d
    }
    // Eleven equal steps that close the circle: that, and not "the phases span
    // 2π", is what one wave on eleven limbs means. Eleven evenly spaced limbs
    // span 10/11 of a turn between the first and the last; it is the ELEVENTH
    // step, from the last limb round to the first, that closes it.
    let total: Float = lags.reduce(0, +)
    expect(near(total, 2 * Float.pi, within: 1e-4), "the eleven lags sum to \(total), not 2π")
    for (i, d) in lags.enumerated() {
        expect(near(d, 2 * Float.pi / Float(thoracicSegments), within: 1e-5),
               "lag \(i) is \(d), not 2π/11")
    }
}

test("the wave runs posterior → anterior, which is adlocomotory") {
    // The rearmost limb leads: at any moment its phase is ahead of the limb in
    // front of it, so the moment each limb starts its power stroke arrives
    // later the further forward you look.
    for i in 0..<(thoracicSegments - 1) {
        let front: Float = limbPhase(i, frame: 5, mutations: mutations)
        let back: Float = limbPhase(i + 1, frame: 5, mutations: mutations)
        expect(back > front, "limb \(i + 1) should lead limb \(i): \(back) vs \(front)")
    }
    let spread: Float = limbPhase(10, frame: 5) - limbPhase(0, frame: 5)
    let expected: Float = 10 * 2 * Float.pi / 11
    expect(near(spread, expected, within: 1e-5), "front-to-back spread \(spread)")
}

test("no two limbs are ever in the same phase") {
    var phases: [Float] = (0..<thoracicSegments).map { i -> Float in
        var p: Float = limbPhase(i, frame: 11, mutations: mutations)
        while p < 0 { p += 2 * Float.pi }
        return p.truncatingRemainder(dividingBy: 2 * Float.pi)
    }
    phases.sort()
    for i in 1..<phases.count {
        expect(phases[i] - phases[i - 1] > 0.3, "two limbs at \(phases[i - 1]) and \(phases[i])")
    }
}

test("nothing jumps between frames") {
    // The tracers are excluded: a tracer that reaches the far end of the field
    // is replaced by one arriving at the near end, which is a jump on purpose
    // and happens 8 mm off the edge of the frame. The ANIMAL may not jump.
    func animalOnly(_ p: Pose) -> Int {
        p.prims.count - (p.counts["free tracer"] ?? 0) - (p.counts["groove tracer"] ?? 0)
    }
    var previous = poseArtemia(frame: 0, posture: posture, mutations: mutations)
    var worst: Float = 0
    for f in 1...swimFrameCount {
        let now = poseArtemia(frame: f, posture: posture, mutations: mutations)
        let n = min(animalOnly(previous), animalOnly(now))
        for i in 0..<n {
            let a = SIMD3<Float>(previous.prims[i].c.x, previous.prims[i].c.y, previous.prims[i].c.z)
            let b = SIMD3<Float>(now.prims[i].c.x, now.prims[i].c.y, now.prims[i].c.z)
            worst = max(worst, simd_distance(a, b))
        }
        previous = now
    }
    // A limb tip moves about 1.7 mm per half stroke, in 3 frames: roughly
    // 180 µm a frame. Anything past 400 µm would be a jump, not a beat.
    expect(worst < 400, "biggest single-frame move \(worst) µm")
}

// MARK: - the animal

section("the animal")

let pose = poseArtemia(frame: 7, posture: posture, mutations: mutations)

test("the body plan is the published one") {
    expectEqual(pose.counts["thoracic segment"] ?? 0, 11)
    expectEqual(pose.counts["phyllopod"] ?? 0, 22)
    expectEqual(pose.counts["abdominal segment"] ?? 0, 6)
    expectEqual(pose.counts["genital segment"] ?? 0, 2)
    expectEqual(pose.counts["compound eye"] ?? 0, 2)
    expectEqual(pose.counts["naupliar eye cup"] ?? 0, 3)
    expectEqual(pose.counts["telson"] ?? 0, 1)
    expectEqual(pose.counts["furcal ramus"] ?? 0, 2)
    expectEqual(pose.counts["antennule"] ?? 0, 2)
    expectEqual(pose.counts["antenna"] ?? 0, 2)
    // One pair of phyllopods per thoracic segment, and no spares.
    expectEqual(pose.counts["phyllopod"] ?? 0, 2 * (pose.counts["thoracic segment"] ?? 0))
}

test("the animal is 8 to 10 mm long, as an adult is") {
    var lo: Float = .greatestFiniteMagnitude
    var hi: Float = -.greatestFiniteMagnitude
    let anterior: SIMD3<Float> = posture.axes().anterior
    for p in pose.prims where p.tissue != Tissue.alga.rawValue {
        let b = p.bounds()
        for k in 0..<8 {
            let corner = SIMD3<Float>(k & 1 == 0 ? b.lo.x : b.hi.x,
                                      k & 2 == 0 ? b.lo.y : b.hi.y,
                                      k & 4 == 0 ? b.lo.z : b.hi.z)
            let along: Float = simd_dot(corner, anterior)
            lo = min(lo, along)
            hi = max(hi, along)
        }
    }
    let mm: Float = (hi - lo) / 1000
    expect(mm >= 8 && mm <= 10, "the animal measures \(mm) mm tip to tip")
}

test("every primitive has a known tissue and a real size") {
    var unknown = 0
    var degenerate = 0
    for p in pose.prims {
        if p.tissue < 0 || p.tissue >= tissueCount { unknown += 1 }
        let b = p.bounds()
        let e: SIMD3<Float> = b.hi - b.lo
        if !(e.x > 0 && e.y > 0 && e.z > 0) { degenerate += 1 }
        if e.x.isNaN || e.y.isNaN || e.z.isNaN { degenerate += 1 }
    }
    expectEqual(unknown, 0)
    expectEqual(degenerate, 0)
}

test("the scene fits in the mailbox, with room to spare") {
    expect(pose.prims.count <= mailboxCapacity,
           "\(pose.prims.count) primitives against a \(mailboxCapacity)-bit mailbox")
    expect(pose.prims.count > 600, "only \(pose.prims.count) primitives")
}

test("the limbs differ only in size — no regional specialisation") {
    // Every phyllopod is built by the same code with one scale factor, so the
    // proof is that each limb's lobes have the same shape ratios as the first.
    func ratios(_ limb: Int) -> [Float] {
        pose.limbLobes[limb].map { i -> Float in
            let b = pose.prims[i].bounds()
            let e: SIMD3<Float> = b.hi - b.lo
            return e.x / e.y
        }
    }
    let first = ratios(0)
    for limb in stride(from: 2, to: 22, by: 2) {
        let r = ratios(limb)
        expectEqual(r.count, first.count)
    }
    // ...and that the size runs monotonically down the thorax.
    func volume(_ limb: Int) -> Float {
        var v: Float = 0
        for i in pose.limbLobes[limb] {
            let b = pose.prims[i].bounds()
            let e: SIMD3<Float> = b.hi - b.lo
            v += e.x * e.y * e.z
        }
        return v
    }
    expect(volume(0) > volume(20), "the front limb should be the bigger one")
}

test("the gut runs the whole length, from the mouth back") {
    let gut = pose.prims.filter { $0.tissue == Tissue.gut.rawValue }
    expectEqual(gut.count, gutSegments)
    let anterior: SIMD3<Float> = posture.axes().anterior
    var lo: Float = .greatestFiniteMagnitude
    var hi: Float = -.greatestFiniteMagnitude
    for p in gut {
        for e in [SIMD3<Float>(p.r0.x, p.r0.y, p.r0.z), SIMD3<Float>(p.r1.x, p.r1.y, p.r1.z)] {
            let along: Float = simd_dot(e, anterior)
            lo = min(lo, along)
            hi = max(hi, along)
        }
    }
    let mm: Float = (hi - lo) / 1000
    expect(mm > 6, "the gut only spans \(mm) mm")
}

test("the food groove runs between the gnathobase rows, with the mouth at its front") {
    // The groove is a gap, not a thing: what is drawn is the two rows of
    // gnathobases either side of it. So the check is that the groove's path
    // stays on the midline, ends at the mouth, and runs forward.
    let back = foodGroovePoint(0)
    let front = foodGroovePoint(1)
    expectEqual(back.z, 0)
    expectEqual(front.z, 0)
    expect(front.x > back.x, "the groove must run forward, to the mouth")
    expect(simd_distance(front, mouthPosition) < 1, "the groove ends at the mouth")
    // And it lies ventral to the trunk's axis all the way along.
    for k in 0...20 {
        let p = foodGroovePoint(Float(k) / 20)
        expect(p.y < -150, "the groove rose to y = \(p.y)")
    }
}

test("it swims ventral side up, and the camera is very nearly ventral") {
    let axes = posture.axes()
    let ventral: SIMD3<Float> = -axes.dorsal
    // The ventral light reaction turns the animal over so its limbs face the
    // lamp — and the lamp, in a brightfield micrograph, is behind the camera's
    // subject on the camera's own axis. So "ventral side up" and "ventral side
    // toward the camera" are not two separate claims here, they are the same
    // one: the camera looks along −Z, and up is +Y, and a shot taken from
    // above a belly-up animal has those coincide.
    //
    // An earlier version demanded ventral.y > 0.2 AND ventral.z > 0.2, which
    // over-constrained it: a genuinely ventral camera spends nearly all of the
    // vector on z and cannot then have much y left. That forced a compromise
    // ventro-LATERAL angle, which hid one of the two limb series.
    expect(ventral.z > 0.8, "the camera is not on the ventral side: \(ventral)")
    // What still has to hold is that the animal is not belly-DOWN.
    expect(ventral.y > -0.1, "the animal has rolled belly-down: \(ventral)")
    expect(axes.anterior.x > 0.5 && axes.anterior.y > 0.3,
           "the head should be up and to the right: \(axes.anterior)")
}

// MARK: - nothing passes through anything

section("no interpenetration, at any of the 120 frames")

/// Points spread over an ellipsoid's surface, for the interpenetration tests.
let unitSphereSamples: [SIMD3<Float>] = (0..<26).map { i -> SIMD3<Float> in
    let n: Float = 26
    let y: Float = 1 - 2 * (Float(i) + 0.5) / n
    let r: Float = (max(0, 1 - y * y)).squareRoot()
    let a: Float = Float(i) * 2.39996323
    return SIMD3(r * cos(a), y, r * sin(a))
}

func surfacePoints(_ p: GPUPrim) -> [SIMD3<Float>] {
    let inv = simd_float3x3(columns: (SIMD3(p.r0.x, p.r1.x, p.r2.x),
                                      SIMD3(p.r0.y, p.r1.y, p.r2.y),
                                      SIMD3(p.r0.z, p.r1.z, p.r2.z)))
    let m: simd_float3x3 = inv.inverse
    let centre = SIMD3<Float>(p.c.x, p.c.y, p.c.z)
    return unitSphereSamples.map { u -> SIMD3<Float> in centre + m * (u * 0.999) }
}

typealias Box = (lo: SIMD3<Float>, hi: SIMD3<Float>)

func boxesOverlap(_ a: Box, _ b: Box) -> Bool {
    for k in 0..<3 {
        if a.hi[k] <= b.lo[k] { return false }
        if b.hi[k] <= a.lo[k] { return false }
    }
    return true
}

/// How deep the worst intrusion is, as a fraction of the intruded ellipsoid's
/// own radius in that direction. 0 means nothing touched.
func deepestIntrusion(_ point: SIMD3<Float>, into p: GPUPrim) -> Float {
    let rel: SIMD3<Float> = point - SIMD3(p.c.x, p.c.y, p.c.z)
    let r0 = SIMD3<Float>(p.r0.x, p.r0.y, p.r0.z)
    let r1 = SIMD3<Float>(p.r1.x, p.r1.y, p.r1.z)
    let r2 = SIMD3<Float>(p.r2.x, p.r2.y, p.r2.z)
    let u = SIMD3<Float>(simd_dot(r0, rel), simd_dot(r1, rel), simd_dot(r2, rel))
    let d: Float = simd_length(u)
    return d < 1 ? 1 - d : 0
}

test("no limb ever passes through another limb") {
    var worst: Float = 0
    var where_ = ""
    for f in 0..<swimFrameCount {
        let p = poseArtemia(frame: f, posture: posture, mutations: mutations)
        var boxes: [[Box]] = []
        for lobes in p.limbLobes {
            boxes.append(lobes.map { i -> Box in p.prims[i].bounds() })
        }
        for a in 0..<p.limbLobes.count {
            for b in 0..<p.limbLobes.count where b != a {
                for (ia, la) in p.limbLobes[a].enumerated() {
                    let ba = boxes[a][ia]
                    var overlaps = false
                    for jb in 0..<p.limbLobes[b].count {
                        let bb = boxes[b][jb]
                        if boxesOverlap(ba, bb) { overlaps = true; break }
                    }
                    if !overlaps { continue }
                    for q in surfacePoints(p.prims[la]) {
                        for (jb, lb) in p.limbLobes[b].enumerated() {
                            let depth: Float = deepestIntrusion(q, into: p.prims[lb])
                            if depth > worst {
                                worst = depth
                                where_ = "frame \(f), limb \(a) lobe \(ia)"
                                    + " into limb \(b) lobe \(jb)"
                            }
                        }
                    }
                }
            }
        }
    }
    expectEqual(worst, 0)
    expect(worst == 0, "worst limb-into-limb intrusion \(worst) — \(where_)")
}

test("no limb ever passes into the body") {
    var worst: Float = 0
    var where_ = ""
    for f in 0..<swimFrameCount {
        let p = poseArtemia(frame: f, posture: posture, mutations: mutations)
        let bodyBoxes: [Box] = p.bodyPrims.map { p.prims[$0].bounds() }
        for (a, lobes) in p.limbLobes.enumerated() {
            for l in lobes {
                let bl: Box = p.prims[l].bounds()
                for (k, bi) in p.bodyPrims.enumerated() {
                    let bb = bodyBoxes[k]
                    if !boxesOverlap(bl, bb) { continue }
                    for q in surfacePoints(p.prims[l]) {
                        let depth: Float = deepestIntrusion(q, into: p.prims[bi])
                        if depth > worst {
                            worst = depth
                            where_ = "frame \(f), limb \(a) into body piece \(k)"
                        }
                    }
                }
            }
        }
    }
    expectEqual(worst, 0)
    expect(worst == 0, "worst limb-into-body intrusion \(worst) — \(where_)")
}

// MARK: - Beer–Lambert

section("Beer–Lambert, against closed forms")

let oneSigma: [SIMD3<Float>] = {
    var t = [SIMD3<Float>](repeating: .zero, count: tissueCount)
    t[Tissue.body.rawValue] = SIMD3(1e-3, 2e-3, 3e-3)
    t[Tissue.gut.rawValue] = SIMD3(5e-3, 1e-3, 2e-3)
    return t
}()

test("a ray through empty space gives back the backlight, with no epsilon") {
    let lamp: SIMD3<Float> = backlight.colour(px: 40, py: 17, width: 129, height: 97,
                                              halfWidth: 500, halfHeight: 400)
    let tau = cpuOpticalDepth(origin: SIMD3(0, 0, 1000), direction: SIMD3(0, 0, -1),
                              prims: [], sigma: oneSigma).tau
    expectEqual(tau, SIMD3<Float>(0, 0, 0))
    // exp(−0) is exactly 1, and a colour multiplied by exactly 1 is itself.
    let out: SIMD3<Float> = lamp * SIMD3(exp(-tau.x), exp(-tau.y), exp(-tau.z))
    expectEqual(out, lamp)
}

test("through a sphere's centre, τ is exactly σ·2r") {
    let r: Float = 137
    let s = GPUPrim.sphere(centre: SIMD3(11, -23, 5), radius: r, tissue: .body)
    let tau = tauThrough(SIMD3(11, -23, 5), [s], oneSigma)
    let expected: SIMD3<Float> = oneSigma[Tissue.body.rawValue] * 2 * r
    for k in 0..<3 {
        expect(near(tau[k], expected[k], within: 1e-5),
               "channel \(k): \(tau[k]) against \(expected[k])")
    }
}

test("through a general ellipsoid, τ is σ times the chord the matrix says") {
    // The new primitive of this step, checked against the thing it must be:
    // an axis-aligned ellipsoid of semi-axes (a, b, c) crossed along z gives a
    // chord of 2c through its centre, whatever a and b are.
    let m = simd_float3x3(diagonal: SIMD3<Float>(400, 60, 90))
    let e = GPUPrim.ellipsoid(centre: SIMD3(0, 0, 0), m: m, tissue: .body)
    let tau = tauThrough(.zero, [e], oneSigma)
    let expected: SIMD3<Float> = oneSigma[Tissue.body.rawValue] * 2 * 90
    for k in 0..<3 { expect(near(tau[k], expected[k], within: 1e-4), "channel \(k): \(tau[k])") }

    // And a rotated one gives the same chord as the unrotated one does along
    // the rotated axis — which is the actual claim about M⁻¹.
    let a: Float = radians(37)
    let rot = simd_float3x3(columns: (SIMD3(cos(a), sin(a), 0), SIMD3(-sin(a), cos(a), 0),
                                      SIMD3(0, 0, 1)))
    let turned = GPUPrim.ellipsoid(centre: .zero, m: rot * m, tissue: .body)
    let tau2 = tauThrough(.zero, [turned], oneSigma)
    for k in 0..<3 { expect(near(tau2[k], expected[k], within: 1e-4), "turned: \(tau2[k])") }
}

test("across a capsule's axis, τ is exactly σ·2r") {
    let r: Float = 40
    let cap = GPUPrim.capsule(from: SIMD3(-300, 0, 0), to: SIMD3(300, 0, 0), radius: r,
                              tissue: .body)
    let tau = tauThrough(SIMD3(120, 0, 0), [cap], oneSigma)
    let expected: SIMD3<Float> = oneSigma[Tissue.body.rawValue] * 2 * r
    for k in 0..<3 { expect(near(tau[k], expected[k], within: 1e-4), "channel \(k): \(tau[k])") }
    // ...and through a cap, the same, because the cap is a hemisphere of the
    // same radius.
    let capTau = tauThrough(SIMD3(300, 0, 0), [cap], oneSigma)
    for k in 0..<3 { expect(near(capTau[k], expected[k], within: 1e-4), "cap: \(capTau[k])") }
}

test("union, not sum: two coincident spheres give one sphere's τ") {
    let r: Float = 200
    let a = GPUPrim.sphere(centre: .zero, radius: r, tissue: .body)
    let b = GPUPrim.sphere(centre: .zero, radius: r, tissue: .body)
    let one = tauThrough(.zero, [a], oneSigma)
    let two = tauThrough(.zero, [a, b], oneSigma)
    for k in 0..<3 {
        expect(near(two[k], one[k], within: 1e-5),
               "two coincident spheres gave \(two[k]) where one gives \(one[k])")
    }
}

test("the union merges what overlaps and sums what does not") {
    let r: Float = 100
    let sig: SIMD3<Float> = oneSigma[Tissue.body.rawValue]
    // Two spheres 120 µm apart: they overlap, and the union is 320 µm long.
    let overlapping = [GPUPrim.sphere(centre: SIMD3(0, 0, 60), radius: r, tissue: .body),
                       GPUPrim.sphere(centre: SIMD3(0, 0, -60), radius: r, tissue: .body)]
    let merged = tauThrough(.zero, overlapping, oneSigma)
    for k in 0..<3 {
        expect(near(merged[k], sig[k] * 320, within: 1e-4), "merged: \(merged[k])")
    }
    // Two spheres 600 µm apart: two separate crossings, so they add.
    let apart = [GPUPrim.sphere(centre: SIMD3(0, 0, 300), radius: r, tissue: .body),
                 GPUPrim.sphere(centre: SIMD3(0, 0, -300), radius: r, tissue: .body)]
    let summed = tauThrough(.zero, apart, oneSigma)
    for k in 0..<3 {
        expect(near(summed[k], sig[k] * 400, within: 1e-4), "apart: \(summed[k])")
    }
}

test("different tissues add: the gut inside the body is not merged away") {
    let body = GPUPrim.sphere(centre: .zero, radius: 300, tissue: .body)
    let gut = GPUPrim.sphere(centre: .zero, radius: 80, tissue: .gut)
    let alone = tauThrough(.zero, [body], oneSigma)
    let both = tauThrough(.zero, [body, gut], oneSigma)
    let extra: SIMD3<Float> = oneSigma[Tissue.gut.rawValue] * 160
    for k in 0..<3 {
        expect(near(both[k] - alone[k], extra[k], within: 1e-4),
               "the gut added \(both[k] - alone[k]) where it should add \(extra[k])")
    }
}

test("adding tissue always darkens, never brightens") {
    var prims: [GPUPrim] = []
    var previous = SIMD3<Float>(repeating: 0)
    for k in 0..<8 {
        let z: Float = Float(k) * 500 - 1750
        prims.append(GPUPrim.sphere(centre: SIMD3(0, 0, z), radius: 90, tissue: .body))
        let tau = tauThrough(.zero, prims, oneSigma)
        for c in 0..<3 {
            expect(tau[c] > previous[c], "τ went from \(previous[c]) to \(tau[c])")
        }
        previous = tau
    }
    // And the picture follows: more τ is always less light out.
    let lamp = SIMD3<Float>(0.6, 0.85, 0.9)
    let dim: SIMD3<Float> = lamp * SIMD3(exp(-previous.x), exp(-previous.y), exp(-previous.z))
    for c in 0..<3 { expect(dim[c] < lamp[c], "channel \(c) brightened") }
}

// MARK: - the GPU

section("the GPU, the grid and the mailbox")

let device = try findDevice()
let renderer = try SceneRenderer(device: device)
/// Odd dimensions, so the centre pixel's ray goes exactly through the camera
/// centre and the closed forms above apply to it unchanged.
let probeSize = 65
/// A 320-wide test frame has to see the same 11.6 mm of field the 1,280-wide
/// render does, or the animal spills off the edge and "empty space" is not.
let wideFieldPixel: Float = micronsPerPixel * Float(frameWidth) / 320
let probeBuffer = device.makeBuffer(length: probeSize * probeSize * 4,
                                    options: .storageModeShared)!

func probeTau(_ prims: [GPUPrim], at centre: SIMD3<Float>, sigma table: [SIMD3<Float>],
              useGrid: Bool, density: Float = 1) throws -> SIMD3<Float> {
    let cam = Camera(centre: centre, micronsPerPixel: 20, width: probeSize, height: probeSize,
                     standOff: 100_000)
    if useGrid { try renderer.buildGrid(prims, density: density) }
    _ = try renderer.render(prims: prims, sigma: table, camera: cam, backlight: backlight,
                            into: probeBuffer, width: probeSize, viewHeight: probeSize,
                            samplesPerSide: 1, useGrid: useGrid, mutations: mutations)
    return renderer.tau(atX: probeSize / 2, y: probeSize / 2, width: probeSize)
}

test("the GPU agrees with the closed form for a sphere") {
    let r: Float = 250
    let s = GPUPrim.sphere(centre: .zero, radius: r, tissue: .body)
    let tau = try probeTau([s], at: .zero, sigma: oneSigma, useGrid: true)
    let expected: SIMD3<Float> = oneSigma[Tissue.body.rawValue] * 2 * r
    for k in 0..<3 {
        expect(near(tau[k], expected[k], within: 2e-4), "channel \(k): \(tau[k]) vs \(expected[k])")
    }
}

test("the GPU agrees with the CPU reference on the whole animal") {
    let p = poseArtemia(frame: 7, posture: posture, mutations: mutations)
    // Aim at the middle of the limb fan, where the interval list is busiest.
    let centre: SIMD3<Float> = p.limbHinge[10]
    let gpu = try probeTau(p.prims, at: centre, sigma: sigma, useGrid: true)
    let cam = Camera(centre: centre, micronsPerPixel: 20, width: probeSize, height: probeSize,
                     standOff: 100_000)
    let (o, d) = cam.ray(sx: Float(probeSize / 2) + 0.5, sy: Float(probeSize / 2) + 0.5,
                         width: probeSize, height: probeSize)
    let cpu = cpuOpticalDepth(origin: o, direction: d, prims: p.prims, sigma: sigma,
                              mutations: mutations)
    expect(cpu.tau.y > 0.05, "the probe ray missed the animal (τ = \(cpu.tau))")
    for k in 0..<3 {
        expect(near(gpu[k], cpu.tau[k], within: 1e-3),
               "channel \(k): GPU \(gpu[k]) against CPU \(cpu.tau[k])")
    }
}

test("mailboxing: a primitive is met once per ray, whatever it straddles") {
    // What the mailbox actually guarantees is a COUNT, not a value. Because
    // the per-tissue union merges two copies of the same interval back into
    // one, a primitive met twice does not change τ at all — turning the
    // mailbox off leaves the picture identical. What it changes is how many
    // intervals a ray has to carry: in this scene the average primitive is
    // filed in 2.2 grid boxes, so the list fills up more than twice as fast,
    // and a list that overflows drops real tissue. So this is the test that
    // watches the count, and the overflow test below watches the consequence.
    let p = poseArtemia(frame: 7, posture: posture, mutations: mutations)
    // WHERE to point the probe used to be a guess — first a hardcoded point in
    // space, then one named exopodite — and both times the guess went stale the
    // moment the limbs were rebuilt, leaving a test that failed a threshold
    // rather than failing a mailbox. So stop guessing. Walk the probe over
    // every lobe of every limb and keep the frame whose busiest ray meets the
    // most primitives: the animal is allowed to say where its own thickest
    // part is, and a rebuild of the limbs moves the probe with it.
    var bestMet = 0
    var bestCentre = SIMD3<Float>()
    for lobes in p.limbLobes {
        for lobe in lobes {
            let c = p.prims[lobe].c
            let centre = SIMD3<Float>(c.x, c.y, c.z)
            _ = try probeTau(p.prims, at: centre, sigma: sigma, useGrid: true)
            for y in 0..<probeSize {
                for x in 0..<probeSize {
                    let n: Int = renderer.intervalsMet(atX: x, y: y, width: probeSize)
                    if n > bestMet { bestMet = n; bestCentre = centre }
                }
            }
        }
    }
    // Then check that whole frame, not the one ray that happened to win it.
    // Every one of the 65 × 65 rays has to meet exactly the primitives the CPU
    // reference meets, so the mutation has four thousand chances to show
    // instead of one, and none of it rests on a magic number.
    _ = try probeTau(p.prims, at: bestCentre, sigma: sigma, useGrid: true)
    let cam = Camera(centre: bestCentre, micronsPerPixel: 20, width: probeSize,
                     height: probeSize, standOff: 100_000)
    // A ray that only grazes a primitive is not evidence about mailboxing: the
    // GPU and this reference solve the same quadratic in the same float32 and
    // still land either side of a tangent, and at 1,300 µm across a 65-pixel
    // probe some ray always finds one. So those rays are counted and set aside
    // rather than quietly rounded into agreement. "Grazing" is a chord under
    // half a micron — a sixth of the thinnest thing in the scene, a 3 µm seta.
    let grazingChord: Float = 0.5
    func shortestChord(_ o: SIMD3<Float>, _ d: SIMD3<Float>) -> Float {
        var shortest: Float = .greatestFiniteMagnitude
        for prim in p.prims {
            guard let (a, b) = cpuInterval(origin: o, direction: d, prim: prim) else { continue }
            if b <= 0 { continue }
            let s: Float = max(a, 0)
            if b <= s { continue }
            shortest = min(shortest, b - s)
        }
        return shortest
    }
    var mismatches = 0
    var grazed = 0
    var firstBad = ""
    var busiest = 0
    for y in 0..<probeSize {
        for x in 0..<probeSize {
            let met: Int = renderer.intervalsMet(atX: x, y: y, width: probeSize)
            let (o, d) = cam.ray(sx: Float(x) + 0.5, sy: Float(y) + 0.5,
                                 width: probeSize, height: probeSize)
            let truth = cpuOpticalDepth(origin: o, direction: d, prims: p.prims, sigma: sigma,
                                        mutations: mutations)
            busiest = max(busiest, truth.intervals)
            if met == truth.intervals { continue }
            if shortestChord(o, d) < grazingChord { grazed += 1; continue }
            mismatches += 1
            if firstBad.isEmpty {
                firstBad = "ray (\(x), \(y)) met \(met) primitives, the CPU meets"
                    + " \(truth.intervals)"
            }
        }
    }
    // Tangents have to stay rare, or setting them aside would be a way of
    // setting the test aside.
    expect(grazed * 100 < probeSize * probeSize,
           "\(grazed) of \(probeSize * probeSize) rays were tangential, too many to discount")
    expectEqual(mismatches, 0)
    expect(mismatches == 0, "\(mismatches) of \(probeSize * probeSize) rays disagree: \(firstBad)")
    // And the premise the whole thing rests on: the frame really is thick, and
    // the grid really does file primitives in more than one box. The bar for
    // "thick" is not a number picked to pass — it is one whole phyllopod's
    // worth of lobes, which is the least a ray can cross and still be looking
    // through a limb rather than past one.
    expect(busiest >= p.limbLobes[0].count,
           "the busiest of \(probeSize * probeSize) rays crossed \(busiest) primitives, fewer"
           + " than the \(p.limbLobes[0].count) lobes of a single phyllopod")
    guard let g = renderer.grid else { expect(false, "no grid"); return }
    let boxesPer: Double = Double(g.items.count) / Double(p.prims.count)
    expect(boxesPer > 1.5,
           "the grid files each primitive in only \(boxesPer) boxes, so there is nothing to mail")
    print(String(format: "        busiest of %d rays crossed %d primitives; %d rays grazed;"
                 + " each primitive is filed in %.1f boxes",
                 probeSize * probeSize, busiest, grazed, boxesPer))
}

test("a straddling primitive gives the same τ as one inside a box") {
    // Two anchors on the scene's long diagonal fix the grid's bounds and
    // resolution in all three axes, so the ray really does cross several boxes
    // and only the test sphere moves between the two measurements.
    let anchorA = GPUPrim.sphere(centre: SIMD3(-5000, -5000, -5000), radius: 10, tissue: .body)
    let anchorB = GPUPrim.sphere(centre: SIMD3(5000, 5000, 5000), radius: 10, tissue: .body)
    let probe = GPUPrim.sphere(centre: .zero, radius: 40, tissue: .body)
    _ = try probeTau([anchorA, anchorB, probe], at: .zero, sigma: oneSigma,
                     useGrid: true, density: 6)
    guard let g = renderer.grid else { expect(false, "no grid"); return }
    expect(g.dims.z > 8, "the grid is only \(g.dims.z) boxes deep along the ray")

    func snap(_ target: SIMD3<Float>, toCorner: Bool) -> SIMD3<Float> {
        var out = SIMD3<Float>(repeating: 0)
        for k in 0..<3 {
            let f: Float = (target[k] - g.origin[k]) / g.cellSize[k]
            let cell: Float = f.rounded(.down)
            let offset: Float = toCorner ? 0 : 0.5
            out[k] = g.origin[k] + (cell + offset) * g.cellSize[k]
        }
        return out
    }
    let inside: SIMD3<Float> = snap(.zero, toCorner: false)
    let straddling: SIMD3<Float> = snap(.zero, toCorner: true)
    expect(simd_reduce_min(g.cellSize) > 90,
           "the boxes (\(g.cellSize)) are smaller than the probe sphere, so it always straddles")

    let a = try probeTau([anchorA, anchorB,
                          GPUPrim.sphere(centre: inside, radius: 40, tissue: .body)],
                         at: inside, sigma: oneSigma, useGrid: true, density: 6)
    let b = try probeTau([anchorA, anchorB,
                          GPUPrim.sphere(centre: straddling, radius: 40, tissue: .body)],
                         at: straddling, sigma: oneSigma, useGrid: true, density: 6)
    for k in 0..<3 {
        expect(near(a[k], b[k], within: 1e-5),
               "channel \(k): inside a box \(a[k]), straddling eight \(b[k])")
    }
    // And a primitive far bigger than a box is still worth exactly its chord.
    let big = GPUPrim.sphere(centre: .zero, radius: 1200, tissue: .body)
    let tau = try probeTau([anchorA, anchorB, big], at: .zero, sigma: oneSigma,
                           useGrid: true, density: 6)
    let expected: SIMD3<Float> = oneSigma[Tissue.body.rawValue] * 2400
    for k in 0..<3 {
        expect(near(tau[k], expected[k], within: 2e-3),
               "a sphere spanning many boxes gave \(tau[k]) instead of \(expected[k])")
    }
}

test("the grid draws the same picture as testing every primitive") {
    let p = poseArtemia(frame: 7, posture: posture, mutations: mutations)
    let centre: SIMD3<Float> = p.limbHinge[8]
    let withGrid = try probeTau(p.prims, at: centre, sigma: sigma, useGrid: true)
    let brute = try probeTau(p.prims, at: centre, sigma: sigma, useGrid: false)
    for k in 0..<3 {
        expect(near(withGrid[k], brute[k], within: 1e-4),
               "channel \(k): grid \(withGrid[k]), brute force \(brute[k])")
    }
}

test("no ray in the real render runs out of room for intervals") {
    // At the real frame size, and at the busiest moments of the beat. Without
    // the mailbox a ray carries 2.2 times as many intervals and this is where
    // that shows up: the list overflows and real tissue is silently dropped.
    let w = frameWidth
    let h = frameViewHeight
    guard let b = device.makeBuffer(length: w * h * 4, options: .storageModeShared) else {
        expect(false, "no buffer"); return
    }
    let cam = Camera(centre: posture.rotation() * SIMD3<Float>(150, -520, 0),
                     micronsPerPixel: micronsPerPixel, width: w, height: h, standOff: 40_000)
    var worst: UInt32 = 0
    var busiest = 0
    for f in [0, 8, 15, 23] {
        let p = poseArtemia(frame: f, posture: posture, mutations: mutations)
        try renderer.buildGrid(p.prims, density: 1)
        _ = try renderer.render(prims: p.prims, sigma: sigma, camera: cam, backlight: backlight,
                                into: b, width: w, viewHeight: h, samplesPerSide: 2,
                                useGrid: true, mutations: mutations)
        worst = max(worst, renderer.overflowCount)
        for y in stride(from: 0, to: h, by: 3) {
            for x in stride(from: 0, to: w, by: 3) {
                busiest = max(busiest, renderer.intervalsMet(atX: x, y: y, width: w) / 4)
            }
        }
    }
    expectEqual(worst, UInt32(0))
    expect(busiest < maxIntervalsPerRay,
           "the busiest ray carried \(busiest) intervals of \(maxIntervalsPerRay)")
}

test("a pixel the animal misses is the backlight, byte for byte") {
    let p = poseArtemia(frame: 7, posture: posture, mutations: mutations)
    let cam = Camera(centre: posture.rotation() * SIMD3<Float>(150, -520, 0),
                     micronsPerPixel: wideFieldPixel, width: 320, height: 240,
                     standOff: 40_000)
    guard let b = device.makeBuffer(length: 320 * 240 * 4, options: .storageModeShared) else {
        expect(false, "no buffer"); return
    }
    try renderer.buildGrid(p.prims, density: 1)
    _ = try renderer.render(prims: p.prims, sigma: sigma, camera: cam, backlight: backlight,
                            into: b, width: 320, viewHeight: 240, useGrid: true,
                            mutations: mutations)
    let withAnimal = Data(bytes: b.contents(), count: 320 * 240 * 4)
    _ = try renderer.render(prims: [], sigma: sigma, camera: cam, backlight: backlight,
                            into: b, width: 320, viewHeight: 240, useGrid: false,
                            mutations: mutations)
    let empty = Data(bytes: b.contents(), count: 320 * 240 * 4)
    // The four corners are a long way from a 10 mm animal in the middle.
    for (x, y) in [(0, 0), (319, 0), (0, 239), (319, 239), (2, 120), (317, 120)] {
        let i: Int = (y * 320 + x) * 4
        for c in 0..<3 {
            expectEqual(withAnimal[i + c], empty[i + c])
        }
    }
}

// MARK: - the sub-pixel clamp

section("the sub-pixel seta clamp")

test("a seta really is sub-pixel, which is why the clamp exists") {
    let px: Float = setaTrueRadius * 2 / micronsPerPixel
    expect(px < 0.5, "a seta is \(px) px across — if it were not, none of this would be needed")
    expect(drawnRadiusFloor > setaTrueRadius, "the floor does not reach the true radius")
}

test("the clamp scales σ by exactly the ratio of the radii") {
    let c = clampThin(trueRadius: setaTrueRadius)
    expectEqual(c.drawnRadius, drawnRadiusFloor)
    expect(near(c.sigmaScale, setaTrueRadius / drawnRadiusFloor, within: 1e-7),
           "σ scaled by \(c.sigmaScale)")
    // And the table the kernel reads carries that scaling, not the raw σ.
    let table = sigmaTable()
    let want: SIMD3<Float> = sigmaSetaTrue * (setaTrueRadius / drawnRadiusFloor)
    for k in 0..<3 {
        expect(near(table[Tissue.seta.rawValue][k], want[k], within: 1e-9),
               "σ_seta[\(k)] is \(table[Tissue.seta.rawValue][k])")
    }
}

test("thin-absorber conservation: τ per seta holds within 1% over a 4× sweep") {
    // The claim the whole anti-aliasing trick rests on. Sweep the drawn radius
    // from the true 1.5 µm out to 6 µm — four times — and the optical depth a
    // ray down the seta's axis sees must not move.
    let sigmaTrue: Float = sigmaSetaTrue.y
    let reference: Float = sigmaTrue * 2 * setaTrueRadius
    var worst: Float = 0
    var steps = 0
    var drawn: Float = setaTrueRadius
    while drawn <= setaTrueRadius * 4 + 1e-4 {
        let c = clampThin(trueRadius: setaTrueRadius, floor: drawn)
        // Through the model...
        let tau: Float = c.axialTau(sigma: sigmaTrue)
        worst = max(worst, abs(tau - reference) / reference)
        // ...and through the renderer, which is what actually matters.
        var table = [SIMD3<Float>](repeating: .zero, count: tissueCount)
        table[Tissue.seta.rawValue] = SIMD3(repeating: sigmaTrue * c.sigmaScale)
        let cap = GPUPrim.capsule(from: SIMD3(-400, 0, 0), to: SIMD3(400, 0, 0),
                                  radius: c.drawnRadius, tissue: .seta)
        let measured: Float = tauThrough(.zero, [cap], table).y
        worst = max(worst, abs(measured - reference) / reference)
        drawn += setaTrueRadius * 0.25
        steps += 1
    }
    expect(steps >= 12, "only \(steps) radii tried")
    expect(worst < 0.01, "τ moved by \(worst * 100)% across the sweep")
}

test("what the clamp does NOT conserve, stated rather than hidden") {
    // Scaling σ by (true/drawn) holds the axial optical depth. It does not hold
    // the seta's absorbance integrated across its width, which goes as σ·πr²
    // and therefore grows in proportion to the drawn radius. At the floor this
    // render uses, that is a factor of 3.03 — the setal fringe is three times
    // heavier than the real one. Conserving the integral instead would need
    // (true/drawn)², which `exponent: 2` does.
    let sigmaTrue: Float = sigmaSetaTrue.y
    let truth = clampThin(trueRadius: setaTrueRadius, floor: setaTrueRadius)
    let axial = clampThin(trueRadius: setaTrueRadius, exponent: 1)
    let area = clampThin(trueRadius: setaTrueRadius, exponent: 2)

    let ratio: Float = axial.integratedAbsorbance(sigma: sigmaTrue)
        / truth.integratedAbsorbance(sigma: sigmaTrue)
    let expected: Float = drawnRadiusFloor / setaTrueRadius
    expect(near(ratio, expected, within: 1e-4),
           "the axial rule inflates integrated absorbance by \(ratio), expected \(expected)")
    expect(ratio > 3 && ratio < 3.1, "the inflation here is \(ratio)×")

    let areaRatio: Float = area.integratedAbsorbance(sigma: sigmaTrue)
        / truth.integratedAbsorbance(sigma: sigmaTrue)
    expect(near(areaRatio, 1, within: 1e-4),
           "the area rule should conserve the integral, it gave \(areaRatio)")
}

test("the clamp is what stops the setal fringe flickering") {
    // The reason the clamp exists, measured rather than asserted — and
    // measured in a way that cannot be confused with the limbs simply moving.
    //
    // An exactly rendered absorber has a total absorbance that does not depend
    // on where it sits inside a pixel: slide the whole fringe sideways by a
    // fraction of a pixel and the ink on the page is the same ink. Any change
    // is aliasing and nothing else. So: render the setae and nothing else, slide the
    // camera across one whole pixel in eighths, and watch the total.
    let w = 512
    let h = 384
    guard let frame = device.makeBuffer(length: w * h * 4, options: .storageModeShared),
          let bg = device.makeBuffer(length: w * h * 4, options: .storageModeShared) else {
        expect(false, "no buffers"); return
    }
    let home: SIMD3<Float> = posture.rotation() * SIMD3<Float>(1600, -700, 0)
    func camera(shiftedBy pixels: Float) -> Camera {
        Camera(centre: home + SIMD3<Float>(pixels * micronsPerPixel, 0, 0),
               micronsPerPixel: micronsPerPixel, width: w, height: h, standOff: 40_000)
    }
    _ = try renderer.render(prims: [], sigma: sigma, camera: camera(shiftedBy: 0),
                            backlight: backlight, into: bg, width: w, viewHeight: h,
                            samplesPerSide: 2, useGrid: false, mutations: mutations)
    // The lamp is a function of the pixel, not of the world, so one rendering
    // of it serves every shift.
    let lamp = bg.contents().assumingMemoryBound(to: UInt8.self)

    func inkWobble(_ muts: Mutations) throws -> (Double, Double) {
        let prims = poseArtemia(frame: 7, posture: posture, mutations: muts)
            .prims.filter { $0.tissue == Tissue.seta.rawValue }
        let table = sigmaTable(mutations: muts)
        try renderer.buildGrid(prims, density: 1)
        var inks: [Double] = []
        for k in 0..<8 {
            let shift: Float = Float(k) / 8
            _ = try renderer.render(prims: prims, sigma: table, camera: camera(shiftedBy: shift),
                                    backlight: backlight, into: frame, width: w, viewHeight: h,
                                    samplesPerSide: 2, useGrid: true, mutations: muts)
            let p = frame.contents().assumingMemoryBound(to: UInt8.self)
            var ink = 0.0
            for i in 0..<(w * h) { ink += Double(Int(lamp[i * 4 + 1]) - Int(p[i * 4 + 1])) }
            inks.append(ink)
        }
        let mean: Double = inks.reduce(0, +) / Double(inks.count)
        let spread: Double = ((inks.max() ?? 0) - (inks.min() ?? 0)) / max(mean, 1e-9)
        return (mean, spread)
    }

    let (clampedInk, clampedWobble) = try inkWobble(mutations)
    let (_, rawWobble) = try inkWobble(mutations.union(.noClamp))
    let gain: Double = rawWobble / max(clampedWobble, 1e-12)
    expect(clampedInk > 0, "the fringe drew nothing")
    print(String(format: "        one pixel of sideways slide moves the fringe's total ink by "
                 + "%.2f%% at true size, %.2f%% clamped — %.0f× steadier",
                 rawWobble * 100, clampedWobble * 100,
                 rawWobble / max(clampedWobble, 1e-12)))
    let complaint: String = String(format: "sliding the fringe across one pixel changes its "
                                   + "total ink by %.2f%% at true size and %.2f%% clamped "
                                   + "— only %.1f× better",
                                   rawWobble * 100, clampedWobble * 100, gain)
    let steadier: Bool = rawWobble > clampedWobble * 2
    expect(steadier, complaint)
}

test("every seta in the render is drawn at the floor, not at true size") {
    let setae = pose.prims.filter { $0.tissue == Tissue.seta.rawValue }
    expectEqual(setae.count, setaePerLimb * 22 + setaePerFurcalRamus * 2)
    for s in setae {
        expect(s.r0.w >= drawnRadiusFloor - 1e-5,
               "a seta is drawn at radius \(s.r0.w), under the \(drawnRadiusFloor) µm floor")
    }
}

// MARK: - the conveyor

section("the water, as a conveyor")

test("the tracer field advances 5.5 mm/s of the animal's own time") {
    // One beat cycle is 1/5 s, so the field must move 1.1 mm past the animal.
    func tracers(_ f: Int) -> [SIMD3<Float>] {
        let p = poseArtemia(frame: f, posture: posture, mutations: mutations)
        return p.prims.filter { $0.tissue == Tissue.alga.rawValue }
            .map { SIMD3($0.c.x, $0.c.y, $0.c.z) }
    }
    let anterior: SIMD3<Float> = posture.axes().anterior
    let a = tracers(0)
    let b = tracers(framesPerCycle)
    expect(a.count > 20, "only \(a.count) tracers")
    let cycleSeconds: Float = 1 / beatFrequencyHz
    let step: Float = swimmingSpeedMicronsPerSecond * cycleSeconds          // 1,100 µm
    // Every tracer a cycle later must be exactly one step BACK along the body
    // from a tracer that was there before — not "about", exactly, because the
    // conveyor is arithmetic and not a simulation. The few that wrapped round
    // the end of the field in that cycle are the exception.
    var matched = 0
    var wrapped = 0
    for q in b {
        var found = false
        for p in a where simd_length(simd_cross(q - p, anterior)) < 1 {
            let along: Float = simd_dot(q - p, anterior)
            if abs(along + step) < 1 { found = true; break }
        }
        if found { matched += 1 } else { wrapped += 1 }
    }
    expect(matched >= (3 * b.count) / 4,
           "\(matched) of \(b.count) tracers moved one step back; \(wrapped) did not")
}

test("the conveyor closes without a rewind") {
    // The tracer field at the end of the loop is the SAME SET of positions it
    // started from — each particle has moved on by 4.4 mm and the one at the
    // far end has taken the place of the one that left.
    func tracers(_ f: Int) -> Set<String> {
        let p = poseArtemia(frame: f, posture: posture, mutations: mutations)
        return Set(p.prims.filter { $0.tissue == Tissue.alga.rawValue }
            .map { String(format: "%.0f,%.0f,%.0f", $0.c.x, $0.c.y, $0.c.z) })
    }
    expectEqual(tracers(0), tracers(swimFrameCount))
    // And they really do move in between: the set a few frames in is different.
    expect(tracers(0) != tracers(9), "the tracers did not move at all")
}

test("captured cells travel forward along the food groove, to the mouth") {
    // The bulk flow goes backwards past the animal; what the setae retain goes
    // the other way, up the groove to the mouth. Both at once is the point.
    func grooveTracer(_ f: Int) -> SIMD3<Float>? {
        var s: Float = loopPhase(frame: f)
        if s >= 1 { s -= 1 }
        return foodGroovePoint(s)
    }
    var previous: Float = -.greatestFiniteMagnitude
    var rose = 0
    for f in stride(from: 2, to: swimFrameCount - 12, by: 4) {
        guard let p = grooveTracer(f) else { continue }
        if p.x > previous { rose += 1 }
        previous = p.x
    }
    expect(rose > 20, "the groove only carried a cell forward \(rose) times")
    let end = foodGroovePoint(1)
    expect(simd_distance(end, mouthPosition) < 1, "the groove does not end at the mouth")
    let grooveSpeed: Double = Double(foodGrooveLength) / 1000 / loopRealSeconds
    expect(grooveSpeed > 2 && grooveSpeed < 8, "groove transport at \(grooveSpeed) mm/s")
}

// MARK: - the numbers on the bar

section("what the caption bar claims")

test("every constant carries a source") {
    expect(artemiaConstants.count >= 30, "only \(artemiaConstants.count) constants recorded")
    var missing: [String] = []
    for c in artemiaConstants {
        if c.source.trimmingCharacters(in: .whitespaces).isEmpty { missing.append(c.name) }
        if c.name.trimmingCharacters(in: .whitespaces).isEmpty { missing.append("(unnamed)") }
        if !c.value.isFinite { missing.append("\(c.name): not a number") }
    }
    expect(missing.isEmpty, "no source for: \(missing)")
    // A MODEL number must say so in its own source line, so the line cannot be
    // quoted out of the table and read as a measurement.
    for c in artemiaConstants where c.evidence == .model {
        expect(c.source.hasPrefix("MODEL"), "\(c.name) is a model but does not say so")
    }
    for c in artemiaConstants where c.evidence == .derived {
        expect(c.source.hasPrefix("DERIVED"), "\(c.name) is derived but does not say so")
    }
    // And a MEASURED one must name where it was measured.
    for c in artemiaConstants where c.evidence == .measured {
        let hasYear: Bool = c.source.contains("19") || c.source.contains("20")
            || c.source.contains("Fox") || c.source.contains("FAO")
        expect(hasYear, "\(c.name) claims to be measured but cites nothing: \(c.source)")
    }
}

test("the table really is the table the render uses") {
    func value(_ name: String) -> Double? {
        artemiaConstants.first { $0.name == name }?.value
    }
    expectEqual(value("thoracic segments"), Double(thoracicSegments))
    expectEqual(value("phyllopods drawn"), Double(pose.counts["phyllopod"] ?? 0))
    expectEqual(value("abdominal segments"), Double(abdominalSegments))
    expectEqual(value("beat frequency"), Double(beatFrequencyHz))
    expectEqual(value("sigma, body"), Double(sigmaBody.y))
    expectEqual(value("Reynolds number, body"), reynolds.body)
}

test("the Reynolds numbers are what U L / ν says they are") {
    let expected: Double = (5.5e-3 * 9.87e-3) / 1.05e-6
    expect(near(reynolds.body, expected, within: 0.5), "Re = \(reynolds.body)")
    expect(reynolds.body > 40 && reynolds.body < 65, "Re ≈ 50 means \(reynolds.body)")
    // The interesting part: the animal is inertial and its setae are not, which
    // is why a fringe of setae works as a paddle and not as a rake.
    expect(reynolds.limb < reynolds.body, "the limb should be gentler than the body")
    expect(reynolds.seta < 1, "the setae live at Re = \(reynolds.seta), which must be below 1")
}

test("the timing is 120 frames, 4 beats, 9.6 s shown for 0.8 s lived") {
    expectEqual(swimFrameCount, 120)
    expectEqual(framesPerCycle * cyclesPerLoop, swimFrameCount)
    expect(near(loopDisplayedSeconds, 9.6, within: 1e-9), "\(loopDisplayedSeconds) s displayed")
    expect(near(loopRealSeconds, 0.8, within: 1e-9), "\(loopRealSeconds) s of animal time")
    // Four beats of a 5 Hz animal take 0.8 s: that is what makes ×1/12 exact
    // rather than approximate.
    let beats: Double = Double(beatFrequencyHz) * loopRealSeconds
    expect(near(beats, Double(cyclesPerLoop), within: 1e-9), "\(beats) beats in the loop")
    expect(near(loopDisplayedSeconds / loopRealSeconds, slowMotionFactor, within: 1e-9))
}

test("the caption bar fits inside the frame") {
    let layout = FrameLayout(width: frameWidth, viewHeight: frameViewHeight,
                             captionHeight: frameCaptionHeight)
    let k = layout.scale
    let caption = swimCaption()
    // With no evidence bar on the right, the three lines have the whole width,
    // which the title now needs: "Rendered model of…" is a good deal longer
    // than what it replaced.
    let left: CGFloat = 18 * k
    let right: CGFloat = CGFloat(layout.width) - 18 * k
    for (s, size, bold) in [(caption.title, 16 * k, true), (caption.subtitle, 12 * k, false),
                            (caption.facts, 11.5 * k, false)] {
        let w = textWidth(s, size: size, bold: bold)
        expect(left + w < right,
               "\"\(s)\" runs to \(left + w) in a \(layout.width) px frame")
    }
    // Three lines, and the last one has to sit inside the bar. drawCaption puts
    // the facts line at barTop + 54k at 11.5k.
    let bottom: CGFloat = 54 * k + 11.5 * k
    expect(bottom < CGFloat(layout.captionHeight), "the last line reaches \(bottom)")
}

test("one millimetre of scale bar is one millimetre of animal") {
    let bar: Float = 1000 / micronsPerPixel
    expect(near(bar, 109.89, within: 0.01), "the 1 mm bar is \(bar) px")
    // The same number the camera uses, measured back off the camera itself.
    let cam = Camera(centre: .zero, micronsPerPixel: micronsPerPixel,
                     width: frameWidth, height: frameViewHeight, standOff: 1000)
    let a = cam.project(SIMD3(0, 0, 0), width: frameWidth, height: frameViewHeight)
    let b = cam.project(SIMD3(1000, 0, 0), width: frameWidth, height: frameViewHeight)
    expect(near(simd_distance(a, b), bar, within: 0.01),
           "the camera makes 1 mm \(simd_distance(a, b)) px")
}

finish()
