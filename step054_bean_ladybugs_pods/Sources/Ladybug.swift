// The seven-spot ladybird on the bean, as numbers: its body, its spots, its
// six legs, where it walks and how its feet step — as functions of time.
// Nothing here touches the GPU; it is the part a test reads against the
// sources.
//
// Millimetres. The beetle's own frame: x forward (the head end), y to its
// left, z up out of its back, the origin under the middle of its belly.
//
// Sources, checked for this step:
//
//   Animal Diversity Web, Coccinella septempunctata (T. Bauer, 2013;
//     animaldiversity.org): "This species typically has seven black spots on
//     its elytra (although it can range from 0 to 9). There is one spot next
//     to the scutellum that bridges the junction between the two elytra;
//     there are two white patches on either side of the scutellum, just above
//     this black scutellar spot. The three spots on each elytra are variable
//     in placement"; "two characteristic pale white spots along the anterior
//     side of the pronotum"; "Range length 6.50 to 7.8 mm"; "it mainly preys
//     on aphids".
//   UC IPM, Sevenspotted Lady Beetle (ipm.ucanr.edu): "The hard, shiny body
//     is relatively large, almost 1/3 inch (8 mm) long and about 1/6 inch
//     (4 mm) wide. The head and thorax are black; both have 2 well-separated,
//     white blotches, one on each side. The wing covers are orange or reddish
//     with 7 black spots and white along the front margin, adjacent to the
//     front, central, black spot."
//   UK Beetle Recording, Coccinella septempunctata (coleoptera.org.uk):
//     "Length : 5 - 8mm ... Number of spots : 0-9 (7) ... Pronotum : black
//     with anterior-lateral white marks. Leg colour : black."
//   Heepe, Wolff & Gorb, Beilstein J. Nanotechnol. 7:1322–1329 (2016),
//     C. septempunctata: tarsi "of forelegs ... midlegs ... and hindlegs";
//     "The tarsus is composed of three tarsomeres and two ventrally curved
//     claws" — three pairs of legs, as in every insect.
//   Zurek, Gorb & Voigt, Interface Focus 5:20140055 (2015), "Locomotion and
//     attachment of leaf beetle larvae Gastrophysa viridula": larvae walk
//     "Instead of the tripod gait of adults" — adult beetles walk in the
//     alternating tripod. That is a leaf beetle (Chrysomelidae), not a
//     ladybird: the gait is borrowed from another beetle family, flagged.
//   Shannag & Obeidat, Ann. Appl. Biol. 152:331–337 (2008): C.
//     septempunctata released on faba bean "significantly reduced aphid
//     density" of the black bean aphid, Aphis fabae. (Faba bean, not the
//     common bean; the caption says only that ladybirds hunt aphids such as
//     this one. No aphids are drawn.)
//
// NOT FOUND, and so MODEL: a measured walking speed for C. septempunctata
// (none in the papers I could read — Europe PMC and publishers' pages); its
// stride and duty factor; its leg segment lengths; its width beyond UC IPM's
// rounded "about 1/6 inch".
//
// THE TWO CLOCKS. The bean is a time-lapse: ten hours in ten seconds, about
// 3600 times faster than life. A ladybird walking at its own pace in that
// film would cross the frame in a hundredth of a frame. So the ladybirds run
// on their own clock, real time, one film second a second, and the frame
// says so. The two clocks put a hard limit on where a ladybird can be. The
// camera rises 30 mm a second with the growing tip; a ladybird that stayed on
// the plant for ever would have to climb that fast (40 mm/s along the wound
// stem) or, run backwards in time, it would once have been above the tip, on
// no plant at all. Walking at 10 mm/s it cannot keep up — so each one
// ARRIVES: it walks up the pole, above the plant, and the plant grows up past
// it. It steps from the pole onto a coil of the stem, along the stem to a
// node, out along the leaf stalk and onto a leaflet, drifting down the frame
// the while, and leaves at the bottom. The pole is where it meets the bean.
//
// THE LOOP. The ladybirds come in two files (two routes), one ladybird per
// file per loop: the ladybird of copy m is copy 0 a loop (ten seconds) later
// in its walk, placed one loop's rise (300 mm, three coils) higher — the same
// coil, node and leaf of the plant one loop up. So at the end of a loop every
// ladybird stands where the one ahead of it stood at the start, one loop
// higher, and the film joins going forward (as step 44's ants did).
//
// THE FEET are driven by distance walked (step 44): a planted foot is fixed
// to the surface under it — a point of the pole, or of the stem, leaf stalk
// or leaflet, which move as the plant grows — so it cannot slide. A foothold
// must be in the leg's reach through its whole stance, with nothing of the
// plant between hip and foot; the knee bends whichever way keeps the leg off
// the plant; and where the path bridges a corner (the step from the pole up
// onto a coil) the beetle pitches nose-up and its body is lifted just clear.

import Foundation
import simd

// MARK: - the body

struct Ellipsoid {
    var centre: V3
    var radii: V3
}

/// The body's parts in the beetle's frame at scale 1. MODEL shapes, sized to
/// the sources: 7.0 mm from the tail of the elytra to the front of the head
/// (ADW 6.5–7.8 mm), 4.8 mm across the elytra (UC IPM "about 4 mm", at "almost
/// 8 mm" long — a ratio narrower than the species' photographs; flagged), and
/// 2.9 mm high, a dome ("convex", MODEL height). The elytra are one dome, cut
/// flat underneath at their rim; the pronotum, head and underside are black.
let elytraPart = Ellipsoid(centre: V3(-0.5, 0, 0.5), radii: V3(2.8, 2.4, 2.4))
let elytraCut: Float = 0.5
let pronotumPart = Ellipsoid(centre: V3(2.2, 0, 0.8), radii: V3(0.95, 1.85, 1.05))
let headPart = Ellipsoid(centre: V3(3.2, 0, 0.6), radii: V3(0.5, 0.85, 0.5))
let bellyPart = Ellipsoid(centre: V3(-0.3, 0, 0.55), radii: V3(2.6, 2.05, 0.55))
let bodyParts: [Ellipsoid] = [elytraPart, pronotumPart, headPart, bellyPart]

let ladybirdLengthRange: ClosedRange<Float> = 6.5...7.8   // ADW
let ladybirdLengthCited: Float = 7.0
/// Width: UC IPM's "about 4 mm" is rounded (1/6 inch); MODEL band to 5.5.
let ladybirdWidthRange: ClosedRange<Float> = 4.0...5.5

/// The spots, (x, y, radius) seen from above in the beetle's frame. ADW: one
/// spot bridging the two elytra next to the scutellum, and three on each
/// elytron, "variable in placement". The places and sizes are MODEL — the
/// usual look of the species: one near the front edge, a larger one in the
/// middle, one near the tail. The scutellar spot is first.
let scutellarSpot = SIMD3<Float>(1.25, 0, 0.45)
let sideSpots: [SIMD3<Float>] = [
    SIMD3(0.7, 1.5, 0.45),
    SIMD3(-0.65, 1.3, 0.6),
    SIMD3(-2.15, 0.95, 0.45),
]
/// Every spot, the three on the right mirrored from the left.
let spots: [SIMD3<Float>] = [scutellarSpot] + sideSpots + sideSpots.map { SIMD3($0.x, -$0.y, $0.z) }
let spotsPerElytron: Int = 3
/// ADW / UC IPM: the white patches either side of the scutellum, at the front
/// edge of the elytra just ahead of the scutellar spot. Places MODEL.
let elytraWhite: [SIMD3<Float>] = [SIMD3(1.62, 0.72, 0.3), SIMD3(1.62, -0.72, 0.3)]
/// UK Beetle Recording: pronotum "black with anterior-lateral white marks";
/// ADW: "along the anterior side". UC IPM: the head has two white blotches
/// too. Places MODEL.
let pronotumWhite: [SIMD3<Float>] = [SIMD3(2.72, 1.2, 0.42), SIMD3(2.72, -1.2, 0.42)]
let headWhite: [SIMD3<Float>] = [SIMD3(3.45, 0.42, 0.2), SIMD3(3.45, -0.42, 0.2)]

func beetleKernelConstants() -> String {
    func f3(_ v: V3) -> String { "float3(\(v.x), \(v.y), \(v.z))" }
    func list(_ name: String, _ a: [SIMD3<Float>]) -> String {
        "constant float3 \(name)[\(a.count)] = { " + a.map { f3($0) }.joined(separator: ", ") + " };"
    }
    return [
        "constant float3 ELYTRA_C = \(f3(elytraPart.centre));",
        "constant float3 ELYTRA_R = \(f3(elytraPart.radii));",
        "constant float ELYTRA_CUT = \(elytraCut);",
        "constant float3 PRONOTUM_C = \(f3(pronotumPart.centre));",
        "constant float3 PRONOTUM_R = \(f3(pronotumPart.radii));",
        "constant float3 HEAD_C = \(f3(headPart.centre));",
        "constant float3 HEAD_R = \(f3(headPart.radii));",
        "constant float3 BELLY_C = \(f3(bellyPart.centre));",
        "constant float3 BELLY_R = \(f3(bellyPart.radii));",
        "constant int SPOT_N = \(spots.count);",
        list("SPOTS", spots),
        list("ELYTRA_WHITE", elytraWhite),
        list("PRONOTUM_WHITE", pronotumWhite),
        list("HEAD_WHITE", headWhite),
    ].joined(separator: "\n")
}

/// A sphere round a beetle's body, for the kernel to skip it by.
func beetleBound(_ b: Beetle) -> (centre: V3, radius: Float) {
    (b.origin + b.forward * (0.2 * b.scale) + b.up * (1.2 * b.scale), 4.0 * b.scale)
}

// MARK: - the legs

/// Six legs: 0–2 the right fore, mid and hind, 3–5 the left (step 44's order).
/// The eight_legs mutant adds a fourth pair between mid and hind.
func legCount(_ m: Mutant) -> Int { m == .eightLegs ? 8 : 6 }
let insectLegCount: Int = 6

/// Where each leg joins the body (the coxa), under the thorax, right side;
/// the left mirrors. MODEL.
let hips: [V3] = [V3(1.5, -0.55, 0.3), V3(0.35, -0.8, 0.25), V3(-0.8, -0.9, 0.25), V3(-0.25, -0.85, 0.25)]
/// Where each foot sits at mid-stance, before it is set down on the surface,
/// right side. MODEL: a ladybird stands with its feet just out beyond the rim
/// of its elytra, the fore feet forward, the hind feet back.
let stanceCentres: [V3] = [V3(2.3, -2.3, -0.6), V3(0.3, -2.75, -0.6), V3(-1.7, -2.5, -0.6), V3(-0.7, -2.7, -0.6)]
/// Femur, and tibia with tarsus. MODEL (no measured leg lengths found):
/// short legs, mostly hidden under the dome.
let femurLength: Float = 1.5
let tibiaLength: Float = 1.9
let femurRadius: Float = 0.16
let tibiaRadius: Float = 0.12
/// The foot's round end, and how high a swinging foot lifts. MODEL.
let footRadius: Float = 0.1
let footLift: Float = 0.35

/// Leg j's side (−1 right, +1 left) and its row (0 fore, 1 mid, 2 hind, 3 the
/// mutant's extra).
func legSide(_ j: Int) -> Float { (j < 3 || j == 6) ? -1 : 1 }
func legRow(_ j: Int) -> Int { j >= 6 ? 3 : j % 3 }

/// The alternating tripod: right fore, left mid, right hind step together,
/// then the other three (Zurek et al. 2015, adult beetles; borrowed).
let tripodA: Set<Int> = [0, 4, 2]
let tripodB: Set<Int> = [3, 1, 5]

/// Stride and duty factor. MODEL: 3.0 mm a stride (under half the body), and
/// each foot down 60 % of the time, so both tripods stand together briefly at
/// each changeover (step 44's figure for ants, borrowed again).
let strideLength: Float = 3.0
let dutyFactor: Float = 0.6

// MARK: - pace and clock

/// Walking speed, mm per second of the ladybird's own (real) time. MODEL: no
/// measured speed found; 10 mm/s, about a body and a half a second — an
/// unhurried walk. At 20 frames a second a stride takes six frames.
let walkingSpeed: Float = 10
/// The ladybirds' clock runs at real time: one film second is one second.
let ladybirdClockFactor: Float = 1

/// Seconds of film for hours of the plant.
func filmSeconds(_ hours: Float) -> Float { hours * loopSeconds / loopHours }
func plantHours(_ seconds: Float) -> Float { seconds * loopHours / loopSeconds }
/// How much faster the bean runs than the ladybirds.
func plantSpeedUp() -> Float { loopHours * 3600 / loopSeconds / ladybirdClockFactor }

let clockCaption: String = "ladybirds in real time (walking pace a model); the bean"

func clockCaptionLine() -> String {
    String(format: "%@ about %.0f× faster", clockCaption, (plantSpeedUp() / 100).rounded() * 100)
}

// MARK: - where it walks

/// The ladybird is this far below the point where the new coil is laid, at
/// the moment it is laid (body centre to coil axis). MODEL, chosen by the
/// clearance test: nearer, and the young shoot sweeping round above the new
/// coil brushes it.
let approachGap: Float = 10

/// A ladybird's route: the node whose leaf it walks out on, the angle round
/// the pole at which it climbs (as the stem's φ there: it meets the coil a
/// radian before the node), and which leaflet it takes. MODEL routes.
struct Route {
    let node: Int
    let azimuth: Float
    let blade: Int
}

/// Two files of ladybirds. The first climbs at the right front of the pole,
/// steps onto the coil a radian before node 0 (azimuth 0, the viewer's
/// right), and walks out onto that leaf's lateral leaflet on the camera's
/// side. The second climbs up the back of the pole, hidden behind it, steps
/// onto the coil there and walks half a turn round the stem — into view round
/// the left of the pole — to node 1, which faces the camera, and out along its
/// leaf. (Hidden while it climbs, so that two or three are in view at once.)
let routes: [Route] = [
    Route(node: 0, azimuth: -1.0, blade: 2),
    Route(node: 1, azimuth: 1.5 * Float.pi - 1.5, blade: 1),
]

/// How far the body's underside rides above the surface. MODEL: just clear.
let bellyClearance: Float = 0.35

/// The heights the walk is laid out by: the stem's once it has opened.
func matureStemHeight(_ phi: Float) -> Float { maturePitch * phi / (2 * Float.pi) + riseOffset }
/// The wound stem's height at φ once the spire has opened: the front's height
/// less the full opened rise (see riseBehindFront).
let riseOffset: Float = (maturePitch - maturePitch * closeSpireFraction) * 2 * Float.pi * openingTurns / (4 * Float.pi)

/// A body pose: origin, forward, up (left follows).
struct Pose {
    var origin: V3
    var forward: V3
    var up: V3
    var left: V3 { simd_cross(up, forward) }

    func world(_ local: V3, scale s: Float) -> V3 {
        origin + forward * (local.x * s) + left * (local.y * s) + up * (local.z * s)
    }

    static func blend(_ a: Pose, _ b: Pose, _ w: Float) -> Pose {
        let o: V3 = a.origin + (b.origin - a.origin) * w
        let f: V3 = simd_normalize(a.forward + (b.forward - a.forward) * w)
        let u0: V3 = a.up + (b.up - a.up) * w
        let u: V3 = simd_normalize(u0 - f * simd_dot(u0, f))
        return Pose(origin: o, forward: f, up: u)
    }
}

// MARK: - the surfaces, at any hour

/// Stem sample i of the drawn chain at hour t — the same lattice and the same
/// corner-cutting correction as `stem` (Bean.swift).
func stemSample(_ i: Int, _ t: Float) -> V3 {
    let dphi: Float = 2 * Float.pi / Float(samplesPerTurn)
    let circumscribe: Float = 1 / cos(dphi / 2)
    let p: V3 = woundPoint(Float(i) * dphi, t, .none)
    return V3(p.x * circumscribe, p.y, p.z * circumscribe)
}

let stemLatticeStep: Float = 2 * Float.pi / Float(samplesPerTurn)

/// A capsule segment's frame: along it, and two directions round it. `hint`
/// is the direction e1 leans towards.
func segmentFrame(_ a: V3, _ b: V3, hint: V3) -> (dir: V3, e1: V3, e2: V3) {
    let dir: V3 = simd_normalize(b - a)
    var e: V3 = hint - dir * simd_dot(hint, dir)
    if simd_length(e) < 1e-4 { e = perpendicular(to: dir) }
    let e1: V3 = simd_normalize(e)
    return (dir, e1, simd_cross(dir, e1))
}

func radialHint(_ a: V3, _ b: V3) -> V3 {
    let m: V3 = (a + b) * 0.5
    let h = V3(m.x, 0, m.z)
    return simd_length(h) > 1e-4 ? simd_normalize(h) : V3(1, 0, 0)
}

/// Where on the plant a foot is planted. Fixed to the material: a point of the
/// pole; of a stem segment (by its lattice index, so it moves with the stem);
/// of a segment of one of the leaf's chains; or of a leaflet's surface.
struct Anchor: Equatable {
    enum Kind: Int { case pole = 0, stem, leafChain, blade }
    var kind: Kind
    /// Stem: lattice index. Leaf chain: chain number × 100 + segment (chains
    /// 0 petiole, 1 rachis, 2–4 petiolules). Blade: blade index + 10 × face
    /// (0 upper, 1 lower).
    var index: Int
    /// Pole: the point. Stem and chain: (h along the segment, and the unit
    /// direction round it in the segment's frame). Blade: (x / length,
    /// s / half-width, which half).
    var a: Float
    var b: Float
    var c: Float
}

/// The plant a ladybird can stand on, at one hour, for copy 0: the pole, the
/// wound stem, and its leaf.
struct Footing {
    var t: Float
    var leaf: LeafGeometry?
    var node: Node?
    /// Lattice range of stem samples that are wound (both ends of a segment).
    var stemFirst: Int
    var stemLast: Int
    /// The node's raceme, if it has one: not stood on, but kept clear of.
    var obstacleChains: [Chain] = []
    var obstacleBlades: [Leaflet] = []

    init(_ t: Float, _ route: Route) {
        self.t = t
        node = nodeAt(route.node, t, .none)
        leaf = node.map { leafGeometry(at: $0) }
        if let n = node, let f = inflorescence(n, t, .none) {
            obstacleChains.append(f.stalk)
            for fl in f.flowers {
                obstacleChains += [fl.pedicel, fl.calyx]
                if let k = fl.keel { obstacleChains.append(k) }
                if let p = fl.pod { obstacleChains.append(p) }
                obstacleBlades += fl.petals
            }
        }
        let phiF: Float = frontAngle(t)
        stemLast = Int(floor(phiF / stemLatticeStep - 0.06))
        // Only the stem near the walk: a turn and a half below the node.
        stemFirst = Int(floor((Float(route.node) * nodeAngleStep - 3 * Float.pi) / stemLatticeStep))
    }

    func leafChain(_ k: Int) -> Chain? {
        guard let l = leaf else { return nil }
        if k == 0 { return l.petiole }
        if k == 1 { return l.rachis }
        let j: Int = k - 2
        return j < l.petiolules.count ? l.petiolules[j] : nil
    }

    /// Point and outward normal of an anchor, now.
    func point(_ an: Anchor) -> (p: V3, n: V3)? {
        switch an.kind {
        case .pole:
            let p = V3(an.a, an.b, an.c)
            return (p, simd_normalize(V3(p.x, 0, p.z)))
        case .stem:
            let i: Int = an.index
            guard i >= stemFirst - 40, i + 1 <= stemLast else { return nil }
            let a: V3 = stemSample(i, t)
            let b: V3 = stemSample(i + 1, t)
            let fr = segmentFrame(a, b, hint: radialHint(a, b))
            let n: V3 = simd_normalize(fr.e1 * an.b + fr.e2 * an.c)
            return (a + (b - a) * an.a + n * stemRadius, n)
        case .leafChain:
            guard let ch = leafChain(an.index / 100) else { return nil }
            let j: Int = an.index % 100
            guard j + 1 < ch.points.count else { return nil }
            let a: V3 = ch.points[j]
            let b: V3 = ch.points[j + 1]
            let fr = segmentFrame(a, b, hint: V3(0, 1, 0))
            let n: V3 = simd_normalize(fr.e1 * an.b + fr.e2 * an.c)
            let r: Float = ch.radii[j] + (ch.radii[j + 1] - ch.radii[j]) * an.a
            return (a + (b - a) * an.a + n * r, n)
        case .blade:
            guard let l = leaf else { return nil }
            let k: Int = an.index % 10
            guard k < l.blades.count else { return nil }
            return bladeSurface(l.blades[k], xi: an.a, sFrac: an.b, half: an.c, lower: an.index >= 10)
        }
    }

    /// The ground's distance at p — the kernel's own functions, on the CPU:
    /// exact for the pole and the capsules, the kernel's estimate for blades.
    func distance(_ p: V3) -> Float {
        var d: Float = simd_length(V3(p.x, 0, p.z)) - poleRadius
        if stemLast > stemFirst {
            for i in stemFirst..<stemLast {
                let a: V3 = stemSample(i, t)
                let b: V3 = stemSample(i + 1, t)
                if simd_distance(p, a) > d + 10 { continue }
                d = min(d, capsuleDistance(p, a, b, stemRadius, stemRadius))
            }
        }
        for k in 0..<5 {
            guard let ch = leafChain(k) else { continue }
            for j in 0..<(ch.points.count - 1) {
                d = min(d, capsuleDistance(p, ch.points[j], ch.points[j + 1], ch.radii[j], ch.radii[j + 1]))
            }
        }
        if let l = leaf {
            for bl in l.blades { d = min(d, cpuLeafletDistance(p, bl)) }
        }
        for ch in obstacleChains {
            for j in 0..<(ch.points.count - 1) {
                d = min(d, capsuleDistance(p, ch.points[j], ch.points[j + 1], ch.radii[j], ch.radii[j + 1]))
            }
        }
        for bl in obstacleBlades { d = min(d, cpuLeafletDistance(p, bl)) }
        return d
    }

    /// The nearest point of the ground to p, as an anchor.
    func project(_ p: V3) -> (anchor: Anchor, point: V3, normal: V3, distance: Float) {
        // The pole.
        let flat = V3(p.x, 0, p.z)
        let rn: V3 = simd_length(flat) > 1e-4 ? simd_normalize(flat) : V3(1, 0, 0)
        var bestD: Float = simd_length(flat) - poleRadius
        var bestP: V3 = V3(rn.x * poleRadius, p.y, rn.z * poleRadius)
        var best = Anchor(kind: .pole, index: 0, a: bestP.x, b: bestP.y, c: bestP.z)
        var bestN: V3 = rn
        // The stem's wound segments.
        if stemLast > stemFirst {
            for i in stemFirst..<stemLast {
                let a: V3 = stemSample(i, t)
                let b: V3 = stemSample(i + 1, t)
                let q = capsuleNearest(p, a, b, stemRadius, stemRadius, hint: radialHint(a, b))
                if q.d < bestD {
                    bestD = q.d; bestP = q.p; bestN = q.n
                    best = Anchor(kind: .stem, index: i, a: q.h, b: q.u, c: q.v)
                }
            }
        }
        // The leaf's chains and blades.
        if leaf != nil {
            for k in 0..<5 {
                guard let ch = leafChain(k) else { continue }
                for j in 0..<(ch.points.count - 1) {
                    let q = capsuleNearest(p, ch.points[j], ch.points[j + 1], ch.radii[j], ch.radii[j + 1],
                                           hint: V3(0, 1, 0))
                    if q.d < bestD {
                        bestD = q.d; bestP = q.p; bestN = q.n
                        best = Anchor(kind: .leafChain, index: k * 100 + j, a: q.h, b: q.u, c: q.v)
                    }
                }
            }
            if let l = leaf {
                for (k, bl) in l.blades.enumerated() {
                    guard let q = bladeNearest(p, bl) else { continue }
                    if q.d < bestD {
                        bestD = q.d; bestP = q.p; bestN = q.n
                        best = Anchor(kind: .blade, index: k + (q.lower ? 10 : 0), a: q.xi, b: q.sFrac, c: q.half)
                    }
                }
            }
        }
        return (best, bestP, bestN, bestD)
    }
}

func capsuleDistance(_ p: V3, _ a: V3, _ b: V3, _ ra: Float, _ rb: Float) -> Float {
    let ab: V3 = b - a
    let h: Float = min(max(simd_dot(p - a, ab) / max(simd_dot(ab, ab), 1e-12), 0), 1)
    return simd_length(p - a - ab * h) - (ra + (rb - ra) * h)
}

let outlineCache: [[SIMD2<Float>]] = (0..<6).map { outlinePolyline(LeafShape(rawValue: $0) ?? .beanOvate) }

/// The kernel's leaflet distance (Kernel.swift, leafletD), line for line.
func cpuLeafletDistance(_ p: V3, _ l: Leaflet) -> Float {
    let q: V3 = p - l.origin
    let L: Float = l.length
    let W: Float = l.halfWidth
    let x: Float = simd_dot(q, l.u)
    let y: Float = simd_dot(q, l.v)
    let z: Float = simd_dot(q, l.w)
    let xc: Float = min(max(x / L, 0), 1)
    let zs: Float = z + l.droop * L * xc * xc
    let dropSlope: Float = 2 * l.droop * xc
    let ya: Float = abs(y)
    let cf: Float = cos(l.fold)
    let sf: Float = sin(l.fold)
    let s: Float = ya * cf + zs * sf
    let h: Float = -ya * sf + zs * cf
    let poly: [SIMD2<Float>] = outlineCache[l.shape.rawValue]
    var best: Float = 1e9
    var halfWidth: Float = 0
    let pt = SIMD2<Float>(x, s)
    for i in 0..<(poly.count - 1) {
        let a: SIMD2<Float> = poly[i] * SIMD2<Float>(L, W)
        let b: SIMD2<Float> = poly[i + 1] * SIMD2<Float>(L, W)
        let ab: SIMD2<Float> = b - a
        let hh: Float = min(max(simd_dot(pt - a, ab) / max(simd_dot(ab, ab), 1e-12), 0), 1)
        best = min(best, simd_length(pt - a - ab * hh))
        if x >= a.x && x <= b.x && b.x > a.x { halfWidth = a.y + (b.y - a.y) * (x - a.x) / (b.x - a.x) }
    }
    let inside: Bool = x > 0 && x < L && s < halfWidth
    var edge: Float = inside ? -best : best
    edge = max(edge, -s)
    let hd: Float = (abs(h) - bladeThickness) / (1 + dropSlope * dropSlope).squareRoot()
    let e = SIMD2<Float>(edge, hd)
    return simd_length(simd_max(e, SIMD2<Float>(0, 0))) + min(max(e.x, e.y), 0)
}

/// The nearest point on a tapered capsule's surface, with h and the unit
/// direction round it in the segment's frame. The same distance the kernel
/// uses, so the point returned lies on the drawn surface.
func capsuleNearest(_ p: V3, _ a: V3, _ b: V3, _ ra: Float, _ rb: Float, hint: V3)
    -> (d: Float, p: V3, n: V3, h: Float, u: Float, v: Float) {
    let ab: V3 = b - a
    let h: Float = min(max(simd_dot(p - a, ab) / max(simd_dot(ab, ab), 1e-12), 0), 1)
    let c: V3 = a + ab * h
    let r: Float = ra + (rb - ra) * h
    let fr = segmentFrame(a, b, hint: hint)
    var dv: V3 = p - c
    dv -= fr.dir * simd_dot(dv, fr.dir)
    if simd_length(dv) < 1e-6 { dv = fr.e1 }
    let n: V3 = simd_normalize(dv)
    return (simd_distance(p, c) - r, c + n * r, n, h, simd_dot(n, fr.e1), simd_dot(n, fr.e2))
}

/// A leaflet's surface at (ξ, s/W, half) — the same blade the kernel draws:
/// the outline in the blade's plane, folded up about the midrib and drooped.
/// On the face h = ±0.15 mm the kernel's distance is exactly zero.
let bladeThickness: Float = 0.15

func bladeSurface(_ l: Leaflet, xi: Float, sFrac: Float, half: Float, lower: Bool) -> (p: V3, n: V3) {
    let L: Float = l.length
    let W: Float = l.halfWidth
    let xc: Float = min(max(xi, 0), 1)
    let s: Float = sFrac * W
    let h: Float = lower ? -bladeThickness : bladeThickness
    let cf: Float = cos(l.fold)
    let sf: Float = sin(l.fold)
    let ya: Float = s * cf - h * sf
    let zs: Float = s * sf + h * cf
    let z: Float = zs - l.droop * L * xc * xc
    let p: V3 = l.origin + l.u * (xi * L) + l.v * (half * ya) + l.w * z
    // The face's normal: the half-blade's, tilted by the droop's slope.
    let slope: Float = 2 * l.droop * xc
    let nl: V3 = simd_normalize(V3(slope * cf, -half * sf, cf))
    var n: V3 = l.u * nl.x + l.v * nl.y + l.w * nl.z
    if lower { n = -n }
    return (p, simd_normalize(n))
}

/// The nearest point of a leaflet's face to p (an estimate, as the kernel's
/// distance is; refined by re-measuring). Only on the blade's inner part —
/// feet are not put on its very edge or on the fold.
func bladeNearest(_ p: V3, _ l: Leaflet) -> (d: Float, p: V3, n: V3, xi: Float, sFrac: Float, half: Float, lower: Bool)? {
    let q: V3 = p - l.origin
    let L: Float = l.length
    let W: Float = l.halfWidth
    if L < 1 || W < 0.5 { return nil }
    let x: Float = simd_dot(q, l.u)
    let y: Float = simd_dot(q, l.v)
    let z: Float = simd_dot(q, l.w)
    let xi: Float = x / L
    guard xi > 0.02, xi < 0.97 else { return nil }
    let xc: Float = xi
    let zs: Float = z + l.droop * L * xc * xc
    let ya: Float = abs(y)
    let cf: Float = cos(l.fold)
    let sf: Float = sin(l.fold)
    let s: Float = ya * cf + zs * sf
    let hh: Float = -ya * sf + zs * cf
    let halfW: Float = W * outlineProfile(l.shape, xi)
    let sMin: Float = bladeThickness * sf / max(cf, 0.2) + 0.05
    guard s > sMin, s < halfW - 0.2 else { return nil }
    let half: Float = y >= 0 ? 1 : -1
    // Always the upper face: the ladybirds walk on the tops of their leaflets,
    // and a foot put on the underside from above would push its leg through
    // the blade. From below, the upper face is the far one, so its distance
    // is only ever larger than the true one — never nearer than it is.
    _ = hh
    let surf = bladeSurface(l, xi: xi, sFrac: s / W, half: half, lower: false)
    return (simd_distance(p, surf.p), surf.p, surf.n, xi, s / W, half, false)
}

// MARK: - the path, segment by segment

/// Where the body rides on each surface, as a pose, from a parameter:
/// the pole (height), the stem (φ), the leaf stalk (h, 0…1 along it), and the
/// leaflet's midrib (ξ, 0…1 along it).
enum Segment: Int { case pole = 0, stem, petiole, blade }

/// Pose on the pole at height y.
func polePose(_ y: Float, _ route: Route) -> Pose {
    let n = V3(cos(route.azimuth), 0, -sin(route.azimuth))
    let g: V3 = n * poleRadius + V3(0, y, 0)
    return Pose(origin: g + n * bellyClearance, forward: V3(0, 1, 0), up: n)
}

/// Pose on top of the stem's crest (the side away from the pole) at φ, hour t.
func stemPose(_ phi: Float, _ t: Float) -> Pose {
    let u: Float = phi / stemLatticeStep
    let i: Int = Int(floor(u))
    let f: Float = u - Float(i)
    let a: V3 = stemSample(i, t)
    let b: V3 = stemSample(i + 1, t)
    let fr = segmentFrame(a, b, hint: radialHint(a, b))
    let c: V3 = a + (b - a) * f
    let g: V3 = c + fr.e1 * stemRadius
    return Pose(origin: g + fr.e1 * bellyClearance, forward: fr.dir, up: fr.e1)
}

/// How far round the leaf stalk the body rides at h: it starts on the side
/// that continues the stem's crest and rolls up onto the stalk's top by 60 %
/// of the way out (MODEL — an insect on a thin stalk walks round it).
func petioleRoll(_ h: Float, start: Float) -> Float { start * (1 - smoothstep(0.04, 0.6, h)) }

func chainAxis(_ ch: Chain, _ h: Float) -> (c: V3, dir: V3, r: Float, e1: V3) {
    let n: Int = ch.points.count - 1
    let u: Float = min(max(h, 0), 1) * Float(n)
    let j: Int = min(Int(floor(u)), n - 1)
    let f: Float = u - Float(j)
    let a: V3 = ch.points[j]
    let b: V3 = ch.points[j + 1]
    let fr = segmentFrame(a, b, hint: V3(0, 1, 0))
    var c: V3 = a + (b - a) * f
    // Beyond either end, carry straight on.
    if h < 0 { c = ch.points[0] + fr.dir * (h * polylineLength(ch.points)) }
    if h > 1 { c = ch.points[n] + fr.dir * ((h - 1) * polylineLength(ch.points)) }
    let r: Float = ch.radii[j] + (ch.radii[j + 1] - ch.radii[j]) * min(max(f, 0), 1)
    return (c, fr.dir, r, fr.e1)
}

/// The roll at the stalk's base that lines its crest up with the stem's.
func petioleStartRoll(_ t: Float, _ route: Route) -> Float {
    guard let n = nodeAt(route.node, t, .none) else { return 0 }
    let pet: Chain = leafGeometry(at: n).petiole
    let ax = chainAxis(pet, 0.05)
    let want: V3 = n.outward - ax.dir * simd_dot(n.outward, ax.dir)
    let e2: V3 = simd_cross(ax.dir, ax.e1)
    return atan2(simd_dot(want, e2), simd_dot(want, ax.e1))
}

func petiolePose(_ h: Float, _ t: Float, roll: Float, _ route: Route) -> Pose? {
    guard let n = nodeAt(route.node, t, .none) else { return nil }
    let pet: Chain = leafGeometry(at: n).petiole
    let ax = chainAxis(pet, h)
    let e2: V3 = simd_cross(ax.dir, ax.e1)
    let th: Float = petioleRoll(h, start: roll)
    let up: V3 = ax.e1 * cos(th) + e2 * sin(th)
    return Pose(origin: ax.c + up * (ax.r + bellyClearance), forward: ax.dir, up: up)
}

/// Pose on the leaflet's midrib at ξ: over the fold's crease, raised so the
/// belly clears the two halves rising either side.
func bladePose(_ xi: Float, _ t: Float, _ route: Route) -> Pose? {
    guard let n = nodeAt(route.node, t, .none) else { return nil }
    let l: Leaflet = leafGeometry(at: n).blades[route.blade]
    let L: Float = l.length
    let xc: Float = min(max(xi, 0), 1)
    let zc: Float = bladeThickness / cos(l.fold) - l.droop * L * xc * xc
    let g: V3 = l.origin + l.u * (xi * L) + l.w * zc
    let slope: Float = 2 * l.droop * xc
    let up: V3 = simd_normalize(l.w + l.u * slope)
    let fwd: V3 = simd_normalize(l.u - l.w * slope)
    let rise: Float = 2.4 * tan(max(l.fold, 0))
    return Pose(origin: g + up * (bellyClearance + rise), forward: fwd, up: up)
}

// MARK: - keeping the body clear

/// Points over the body's skin, in the beetle's frame: what the clearance
/// lift checks against the plant.
let bodySkin: [V3] = {
    var out: [V3] = []
    var rng = Lcg(state: 17)
    for part in bodyParts {
        for _ in 0..<110 {
            var q: V3 = part.centre + rng.unit() * part.radii
            if part.centre == elytraPart.centre { q.z = max(q.z, elytraCut) }
            out.append(q)
        }
    }
    return out
}()

/// How close the body may come to the plant. MODEL.
let bodyMargin: Float = 0.15

/// Where the path bridges a corner — the step from the pole up onto a coil,
/// or from the stem onto the leaf stalk — a body laid on the blended path
/// would sink into one surface or the other. So the body is lifted along its
/// own up until every point of its skin clears the plant by the margin: a
/// continuous function of the pose, so the lift never jumps.
func clearanceLift(_ pose: Pose, _ ground: Footing) -> Pose {
    var p: Pose = pose
    for _ in 0..<6 {
        var worst: Float = .infinity
        for q in bodySkin { worst = min(worst, ground.distance(p.world(q, scale: ladybirdScale))) }
        if worst >= bodyMargin - 1e-4 { break }
        p.origin += p.up * (bodyMargin - worst)
    }
    return p
}

// MARK: - the walk, laid out in time

/// Copy 0's walk: when it steps from each surface to the next, and its
/// parameter on each surface at every moment, integrated at its walking pace
/// relative to the surface (which may be growing under it). Film seconds.
struct WalkPlan {
    let route: Route
    /// Where on the pole it steps onto the stem, and the φ it steps onto.
    let yStep: Float
    let phiStart: Float
    let phiEnd: Float
    let hStart: Float
    let hEnd: Float
    let xiStart: Float
    let xiEnd: Float
    /// Distance walked at each switch (0 at the step onto the stem), and the
    /// blend widths there.
    var switchD: [Float] = [0, 0, 0]
    let blendWidth: [Float] = [14, 10, 10]
    /// How far the body pitches nose-up at the middle of each switch, radians,
    /// pivoting about its tail: a beetle lifts its front to climb a step.
    /// MODEL.
    let switchPitch: [Float] = [0.55, 0.2, 0.15]
    /// Distance walked when the walk ends (it stands still after, far below
    /// the frame).
    var endD: Float = 0
    /// The film time at d = 0.
    let tau0: Float
    /// Tables of the stem, stalk and leaflet parameters against time.
    var table0: Float = 0
    let tableStep: Float = 0.01
    var phiTable: [Float] = []
    var hTable: [Float] = []
    var xiTable: [Float] = []
    var rollTable: [Float] = []

    init(_ route: Route) {
        self.route = route
        // The coil at the climbing azimuth, once opened, crosses the climbing
        // line at matureStemHeight(azimuth). The body steps onto it when its
        // head is under the coil: 6 mm below the coil's axis.
        yStep = matureStemHeight(route.azimuth) - 6.0
        phiStart = route.azimuth + 0.12
        // Off the stem just before the node, onto the stalk near its base.
        phiEnd = Float(route.node) * nodeAngleStep - 0.14
        hStart = 0.06
        hEnd = 0.97
        xiStart = 0.04
        xiEnd = 0.9
        // It reaches the stepping height when? Chosen so the coil it steps
        // onto has been laid long enough to be tight on the pole: the front
        // passes the climbing azimuth at hour azimuth/(2π)·gyre, and the
        // ladybird is then `approachGap` lower than where the coil is laid.
        let tLay: Float = route.azimuth / (2 * Float.pi) * gyreHours
        let yLay: Float = frontHeight(tLay)
        let yAtLay: Float = yLay - approachGap
        tau0 = filmSeconds(tLay) + (yStep - yAtLay) / walkingSpeed
        integrate()
    }

    private mutating func integrate() {
        table0 = tau0 - 2
        let n: Int = 3000
        phiTable = [Float](repeating: 0, count: n)
        hTable = [Float](repeating: 0, count: n)
        xiTable = [Float](repeating: 0, count: n)
        rollTable = [Float](repeating: 0, count: n)
        let i0: Int = Int(((tau0 - table0) / tableStep).rounded())
        // Stem: from phiStart at tau0, forward and (for the blend) back.
        func stemMetric(_ phi: Float, _ tau: Float) -> Float {
            let t: Float = plantHours(tau)
            return simd_distance(stemPose(phi + 0.01, t).origin, stemPose(phi - 0.01, t).origin) / 0.02
        }
        func petMetric(_ h: Float, _ tau: Float) -> Float {
            let t: Float = plantHours(tau)
            guard let n = nodeAt(route.node, t, .none) else { return 60 }
            return max(polylineLength(leafGeometry(at: n).petiole.points), 5)
        }
        func bladeMetric(_ xi: Float, _ tau: Float) -> Float {
            let t: Float = plantHours(tau)
            guard let n = nodeAt(route.node, t, .none) else { return 60 }
            let l: Leaflet = leafGeometry(at: n).blades[route.blade]
            let s: Float = 2 * l.droop * min(max(xi, 0), 1)
            return max(l.length * (1 + s * s).squareRoot(), 5)
        }
        let dt: Float = tableStep
        // Stem.
        phiTable[i0] = phiStart
        for i in (i0 + 1)..<n { phiTable[i] = phiTable[i - 1] + walkingSpeed * dt / stemMetric(phiTable[i - 1], table0 + Float(i - 1) * dt) }
        for i in stride(from: i0 - 1, through: 0, by: -1) { phiTable[i] = phiTable[i + 1] - walkingSpeed * dt / stemMetric(phiTable[i + 1], table0 + Float(i + 1) * dt) }
        var i1: Int = i0
        while i1 < n - 1 && phiTable[i1] < phiEnd { i1 += 1 }
        switchD[0] = 0
        switchD[1] = walkingSpeed * Float(i1 - i0) * dt
        // Stalk.
        hTable[i1] = hStart
        for i in (i1 + 1)..<n { hTable[i] = hTable[i - 1] + walkingSpeed * dt / petMetric(hTable[i - 1], table0 + Float(i - 1) * dt) }
        for i in stride(from: i1 - 1, through: 0, by: -1) { hTable[i] = hTable[i + 1] - walkingSpeed * dt / petMetric(hTable[i + 1], table0 + Float(i + 1) * dt) }
        var i2: Int = i1
        while i2 < n - 1 && hTable[i2] < hEnd { i2 += 1 }
        switchD[2] = walkingSpeed * Float(i2 - i0) * dt
        // Leaflet.
        xiTable[i2] = xiStart
        for i in (i2 + 1)..<n { xiTable[i] = xiTable[i - 1] + walkingSpeed * dt / bladeMetric(xiTable[i - 1], table0 + Float(i - 1) * dt) }
        for i in stride(from: i2 - 1, through: 0, by: -1) { xiTable[i] = xiTable[i + 1] - walkingSpeed * dt / bladeMetric(xiTable[i + 1], table0 + Float(i + 1) * dt) }
        var i3: Int = i2
        while i3 < n - 1 && xiTable[i3] < xiEnd { i3 += 1 }
        endD = walkingSpeed * Float(i3 - i0) * dt
        for i in 0..<n { rollTable[i] = petioleStartRoll(plantHours(table0 + Float(i) * dt), route) }
    }

    func lookup(_ table: [Float], _ tau: Float) -> Float {
        let u: Float = (tau - table0) / tableStep
        if u <= 0 { return table[0] + (table[1] - table[0]) * u }
        let n: Int = table.count
        if u >= Float(n - 1) { return table[n - 1] + (table[n - 1] - table[n - 2]) * (u - Float(n - 1)) }
        let i: Int = Int(floor(u))
        let f: Float = u - Float(i)
        return table[i] + (table[i + 1] - table[i]) * f
    }

    /// Distance walked by film time tau (copy 0): steady, and it stops at the
    /// end of the walk.
    func distance(_ tau: Float) -> Float { min(walkingSpeed * (tau - tau0), endD) }

    /// The film time at which distance d is reached.
    func time(atDistance d: Float) -> Float { tau0 + d / walkingSpeed }

    /// The body's pose at distance d, with the plant as it is at film time tau.
    func pose(distance d: Float, tau: Float) -> Pose {
        let t: Float = plantHours(tau)
        // The parameters on each surface belong to the moment d was reached
        // (for d past the end, the end).
        let td: Float = time(atDistance: min(d, endD))
        func segPose(_ k: Int) -> Pose {
            switch k {
            case 0:
                return polePose(yStep + d, route)
            case 1:
                return stemPose(lookup(phiTable, td), t)
            case 2:
                return petiolePose(lookup(hTable, td), t, roll: lookup(rollTable, td), route) ?? stemPose(phiEnd, t)
            default:
                return bladePose(min(lookup(xiTable, td), xiEnd), t, route) ?? stemPose(phiEnd, t)
            }
        }
        // Which segment, and blending across each switch.
        var k: Int = 0
        for s in 0..<3 where d >= switchD[s] { k = s + 1 }
        var p: Pose = segPose(k)
        for s in 0..<3 {
            let w: Float = blendWidth[s]
            if abs(d - switchD[s]) < w / 2 {
                let a: Pose = segPose(s)
                let b: Pose = segPose(s + 1)
                let u: Float = smoothstep(switchD[s] - w / 2, switchD[s] + w / 2, d)
                p = Pose.blend(a, b, u)
                // Nose up, about the tail, most at the middle of the switch.
                let th: Float = switchPitch[s] * sin(Float.pi * min(max((d - switchD[s] + w / 2) / w, 0), 1))
                let pivot: V3 = p.world(V3(-3.0, 0, 0), scale: ladybirdScale)
                let axis: V3 = p.left
                p = Pose(origin: pivot + rotate(p.origin - pivot, about: axis, by: -th),
                         forward: rotate(p.forward, about: axis, by: -th), up: rotate(p.up, about: axis, by: -th))
            }
        }
        return clearanceLift(p, Footing(t, route))
    }

    /// The body's place without the clearance lift, cheaply, for deciding
    /// whether it is near the picture.
    func roughOrigin(distance d: Float, tau: Float) -> V3 {
        let t: Float = plantHours(tau)
        let td: Float = time(atDistance: min(d, endD))
        switch segment(atDistance: d) {
        case .pole: return polePose(yStep + d, route).origin
        case .stem: return stemPose(lookup(phiTable, td), t).origin
        case .petiole: return petiolePose(lookup(hTable, td), t, roll: 0, route)?.origin ?? V3(0, 0, 0)
        case .blade: return bladePose(min(lookup(xiTable, td), xiEnd), t, route)?.origin ?? V3(0, 0, 0)
        }
    }

    func segment(atDistance d: Float) -> Segment {
        var k: Int = 0
        for s in 0..<3 where d >= switchD[s] { k = s + 1 }
        return Segment(rawValue: k) ?? .blade
    }
}

let walkPlans: [WalkPlan] = routes.map { WalkPlan($0) }

// MARK: - one ladybird at one moment

/// The ladybirds differ a little in size (MODEL, inside ADW's 6.5–7.8 mm).
let ladybirdScale: Float = 1.0

struct LegState {
    var hip: V3
    var knee: V3
    var foot: V3
    var stance: Bool
    /// Where the foot is planted, and the stance it belongs to.
    var anchor: Anchor?
    var stride: Int
    /// The leg could not reach its foot (a test requires this never happens).
    var stretched: Bool
}

struct Ladybird {
    var lineage: Int
    var copy: Int
    var distance: Float
    var segment: Segment
    var pose: Pose
    var beetle: Beetle
    var legs: [LegState]
}

/// Leg j's phase offset: tripod B half a stride behind tripod A.
func legOffset(_ j: Int) -> Float {
    if j >= 6 { return j == 6 ? 0.25 : 0.75 }
    return tripodA.contains(j) ? 0 : 0.5
}

func stanceLocal(_ j: Int) -> V3 {
    let c: V3 = stanceCentres[legRow(j)]
    return V3(c.x, c.y * -legSide(j), c.z)
}

func hipLocal(_ j: Int) -> V3 {
    let c: V3 = hips[legRow(j)]
    return V3(c.x, c.y * -legSide(j), c.z)
}

/// Where leg j is set down for the stance that begins at distance d (copy 0):
/// the body's pose then, the foot's place in it, and the nearest surface.
func touchdown(_ plan: WalkPlan, _ j: Int, atDistance d: Float) -> Anchor {
    let tau: Float = plan.time(atDistance: d)
    let pose: Pose = plan.pose(distance: d, tau: tau)
    var local: V3 = stanceLocal(j)
    local.x += dutyFactor * strideLength / 2 / ladybirdScale
    let want: V3 = pose.world(local, scale: ladybirdScale)
    let hip: V3 = pose.world(hipLocal(j), scale: ladybirdScale)
    // Where the hip will be when this foot lifts again: it must still reach.
    let dEnd: Float = d + dutyFactor * strideLength
    let poseEnd: Pose = plan.pose(distance: dEnd, tau: plan.time(atDistance: dEnd))
    let hipEnd: V3 = poseEnd.world(hipLocal(j), scale: ladybirdScale)
    let dMid: Float = d + 0.5 * dutyFactor * strideLength
    let hipMid: V3 = plan.pose(distance: dMid, tau: plan.time(atDistance: dMid)).world(hipLocal(j), scale: ladybirdScale)
    let ground = Footing(plantHours(tau), plan.route)
    let reach: Float = 0.9 * (femurLength + tibiaLength)
    func worstReach(_ q: (anchor: Anchor, point: V3, normal: V3, distance: Float)) -> Float {
        let f: V3 = q.point + q.normal * footRadius
        var r: Float = max(simd_distance(f, hip), simd_distance(f, hipEnd), simd_distance(f, hipMid))
        // A foothold on the far side of a stalk, behind the plant as seen
        // from the hip, is no foothold: the leg would have to pass through.
        for h in [hip, hipMid, hipEnd] {
            for k in 1...9 {
                let x: V3 = h + (f - h) * (Float(k) / 10)
                if ground.distance(x) < 0.05 { r += 10; break }
            }
        }
        return r
    }
    // The nearest surface to where the foot would go; if that would be out of
    // the leg's reach at either end of the stance (over a step, beside a thin
    // stalk), the nearest surface to points drawn in towards the hips and
    // under the body instead — the first that stays in reach.
    let under: V3 = (hip + hipEnd) * 0.5 - (pose.up + poseEnd.up) * 0.75
    var best = ground.project(want)
    var bestReach: Float = worstReach(best)
    if bestReach > reach {
        // Candidates under the body, along it and to this leg's side.
        let side: V3 = pose.left * legSide(j)
        var tries: [V3] = []
        for k in 1...10 { tries.append(want + (under - want) * (Float(k) / 10)) }
        for fwd in [Float(-1.5), -0.75, 0, 0.75, 1.5, 2.25, 3.0] {
            for lat in [Float(0.4), 1.2, 2.0] {
                for down in [Float(0.8), 1.8] {
                    tries.append(hip + pose.forward * fwd + side * lat - pose.up * down)
                }
            }
        }
        for c in tries {
            let q = ground.project(c)
            let r: Float = worstReach(q) + 0.05 * simd_distance(q.point, want)
            if r < bestReach { best = q; bestReach = r }
        }
    }
    return best.anchor
}

/// Footholds, once worked out, kept: each depends only on its lineage, leg
/// and stride.
final class AnchorCache {
    var store: [Int: Anchor] = [:]
    func anchor(_ lineage: Int, _ j: Int, _ k: Int, _ d: Float) -> Anchor {
        let key: Int = (k * 16 + j) * 8 + lineage
        if let a = store[key] { return a }
        let a: Anchor = touchdown(walkPlans[lineage], j, atDistance: d)
        store[key] = a
        return a
    }
}
let anchorCache = AnchorCache()

/// How clear a leg is of the ground: the least of its segments' distances
/// less their radii.
func legClearance(_ hip: V3, _ knee: V3, _ foot: V3, _ ground: Footing) -> Float {
    var worst: Float = .infinity
    for k in 0...6 {
        let f: Float = Float(k) / 6
        worst = min(worst, ground.distance(hip + (knee - hip) * f) - femurRadius)
    }
    // The tibia down to just short of the foot's round end, whose own
    // distance is set by where it is planted.
    for k in 0...7 {
        let f: Float = Float(k) / 8
        let r: Float = tibiaRadius + (footRadius - tibiaRadius) * f
        worst = min(worst, ground.distance(knee + (foot - knee) * f) - r)
    }
    return worst
}

/// The leg with its knee bent whichever way keeps it off the plant: up and
/// out first (as an insect holds it), then turned about the hip–foot line
/// until femur and tibia both clear the surface.
func solveLegClear(hip h: V3, foot f: V3, up: V3, out: V3, ground: Footing) -> (knee: V3, foot: V3, stretched: Bool, clear: Float) {
    var best = solveLeg(hip: h, foot: f, up: up, out: out)
    var bestClear: Float = legClearance(h, best.knee, best.foot, ground)
    if bestClear >= 0.02 { return (best.knee, best.foot, best.stretched, bestClear) }
    let axis: V3 = simd_normalize(f - h)
    for step in 1...8 {
        for sgn in [Float(1), -1] {
            let turn: Float = sgn * Float(step) * 0.25
            let u2: V3 = rotate(up, about: axis, by: turn)
            let o2: V3 = rotate(out, about: axis, by: turn)
            let c = solveLeg(hip: h, foot: f, up: u2, out: o2)
            let cl: Float = legClearance(h, c.knee, c.foot, ground)
            if cl > bestClear { best = c; bestClear = cl }
            if bestClear >= 0.02 { return (best.knee, best.foot, best.stretched, bestClear) }
        }
    }
    return (best.knee, best.foot, best.stretched, bestClear)
}

/// Two-bone leg: hip to knee to foot, the knee bent up and out.
func solveLeg(hip h: V3, foot f: V3, up: V3, out: V3) -> (knee: V3, foot: V3, stretched: Bool) {
    var d: V3 = f - h
    var len: Float = simd_length(d)
    let reach: Float = femurLength + tibiaLength
    var stretched: Bool = false
    if len > reach * 0.999 {
        stretched = true
        d = d / len * (reach * 0.999)
        len = reach * 0.999
    }
    let e: V3 = d / max(len, 1e-5)
    let a: Float = (femurLength * femurLength - tibiaLength * tibiaLength + len * len) / (2 * len)
    let hk: Float = max(femurLength * femurLength - a * a, 0).squareRoot()
    let b0: V3 = simd_normalize(up + out * 0.6)
    var b: V3 = b0 - e * simd_dot(b0, e)
    if simd_length(b) < 1e-4 { b = perpendicular(to: e) }
    b = simd_normalize(b)
    return (h + e * a + b * hk, h + d, stretched)
}

/// Copy 0 of a lineage at film time tau, placed as copy m (a loop later in
/// its walk, a loop's rise higher). `tau` is the frame's time; the walk's own
/// time is tau − m loops.
func ladybird(lineage li: Int = 0, copy m: Int, tau: Float, mutant: Mutant) -> Ladybird {
    let plan: WalkPlan = walkPlans[li]
    let local: Float = tau - Float(m) * loopSeconds
    let d: Float = plan.distance(local)
    let pose: Pose = plan.pose(distance: d, tau: local)
    let ground = Footing(plantHours(local), plan.route)
    let shift = V3(0, Float(m) * loopRise, 0)
    let s: Float = ladybirdScale
    var legs: [LegState] = []
    for j in 0..<legCount(mutant) {
        let off: Float = legOffset(j)
        let cycles: Float = d / strideLength + off
        let k: Int = Int(floor(cycles))
        let phase: Float = cycles - Float(k)
        let dStart: Float = (Float(k) - off) * strideLength
        let hip: V3 = pose.world(hipLocal(j), scale: s)
        let out: V3 = pose.left * legSide(j)
        var foot: V3
        var stance: Bool
        var anchor: Anchor? = nil
        if phase < dutyFactor {
            stance = true
            if mutant == .slidingFeet {
                // Held in the beetle's frame, set down afresh each frame: it
                // rides along with the body instead of staying put.
                var lc: V3 = stanceLocal(j)
                lc.x += (dutyFactor / 2 - phase) * strideLength / s
                let q = ground.project(pose.world(lc, scale: s))
                foot = q.point + q.normal * footRadius
            } else {
                let an: Anchor = anchorCache.anchor(li, j, k, dStart)
                anchor = an
                if let q = ground.point(an) {
                    foot = q.p + q.n * footRadius
                } else {
                    foot = pose.world(stanceLocal(j), scale: s)
                }
            }
        } else {
            stance = false
            let u: Float = (phase - dutyFactor) / (1 - dutyFactor)
            let a0: Anchor = anchorCache.anchor(li, j, k, dStart)
            let a1: Anchor = anchorCache.anchor(li, j, k + 1, dStart + strideLength)
            let from: V3 = ground.point(a0)?.p ?? pose.world(stanceLocal(j), scale: s)
            let to: V3 = ground.point(a1)?.p ?? pose.world(stanceLocal(j), scale: s)
            let e: Float = u * u * (3 - 2 * u)
            let lin: V3 = from + (to - from) * e
            let q = ground.project(lin)
            let reach: Float = 0.97 * (femurLength + tibiaLength)
            // A foot in the air, lifted off the surface, and held within the
            // leg's reach: over a step it swings closer in to the body than
            // the straight line between its footholds. Round a thin stalk it
            // lifts higher, as far as it must for the leg to clear it.
            var f0: V3 = q.point
            for extra in [Float(0), 0.3, 0.7, 1.2] {
                var p: V3 = q.point + q.normal * (footRadius + (footLift + extra) * sin(Float.pi * u))
                let fromHip: Float = simd_distance(p, hip)
                if fromHip > reach { p = hip + (p - hip) * (reach / fromHip) }
                f0 = p
                let c = solveLegClear(hip: hip, foot: p, up: pose.up, out: out, ground: ground)
                if c.clear >= 0.02 && ground.distance(p) >= footRadius { break }
            }
            foot = f0
        }
        let leg = solveLegClear(hip: hip, foot: foot, up: pose.up, out: out, ground: ground)
        legs.append(LegState(hip: hip + shift, knee: leg.knee + shift, foot: leg.foot + shift, stance: stance,
                             anchor: anchor, stride: k, stretched: leg.stretched))
    }
    let b = Beetle(origin: pose.origin + shift, forward: pose.forward, left: pose.left, up: pose.up, scale: s)
    return Ladybird(lineage: li, copy: m, distance: d, segment: plan.segment(atDistance: d),
                    pose: Pose(origin: pose.origin + shift, forward: pose.forward, up: pose.up), beetle: b, legs: legs)
}

/// The copies of the file in or near the frame at hour t: those whose body
/// projects within `margin` of the picture (as a fraction of its height).
/// The rest are off the picture; they are not drawn.
func ladybirdsInView(_ t: Float, _ m: Mutant, margin: Float = 0.12) -> [Ladybird] {
    let tau: Float = filmSeconds(t)
    let cam: Camera = camera(t)
    var out: [Ladybird] = []
    for li in 0..<walkPlans.count {
        for c in -3...3 {
            let local: Float = tau - Float(c) * loopSeconds
            let d: Float = walkPlans[li].distance(local)
            // Where the body is, roughly (no lift): enough to decide.
            let o: V3 = walkPlans[li].roughOrigin(distance: d, tau: local) + V3(0, Float(c) * loopRise, 0)
            let p: SIMD2<Float> = cam.project(o, width: 1000, height: 1300)
            if p.y > -1300 * margin && p.y < 1300 * (1 + margin) && p.x > -1000 * margin && p.x < 1000 * (1 + margin) {
                out.append(ladybird(lineage: li, copy: c, tau: tau, mutant: m))
            }
        }
    }
    return out
}

func addLadybirds(_ t: Float, _ m: Mutant, into s: inout PlantScene) {
    for lb in ladybirdsInView(t, m) {
        s.beetles.append(lb.beetle)
        for leg in lb.legs {
            s.chains.append(Chain(points: [leg.hip, leg.knee, leg.foot],
                                  radii: [femurRadius, tibiaRadius, footRadius], material: .leg))
        }
    }
}
