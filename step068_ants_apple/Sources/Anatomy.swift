// Step 68: step 44's poseable Lasius niger worker, copied, for three ants
// that walk up to a piece of peeled apple, tap it, taste it and drink. Three
// additions, nothing else changed in the body, legs or antennae:
//
//   * the glossa (the tongue at the tip of the labium) can be pushed out of
//     the mouth, to drink;
//   * an antenna's funiculus can bow in any direction asked for (step 44's
//     always bowed forward and up), so its tip can rest on the apple's wet face
//     without the arc bulging into it;
//   * step 50's two sense hairs in micrometres, for the round inset that
//     rides on one ant's right antenna tip.
//
// Units: MILLIMETRES, in the ant's own frame — y up, the ground is y = 0,
// the ant faces +x and its right side is +z. Walk.swift places that frame in
// the world. Every number carries where it came from, or says MODEL and why.

import Foundation
import simd

// MARK: - what to break, for the mutation check

/// Each one must make the test suite fail.
enum Mutant: Int {
    case none = 0
    case segments13 = 1   // an antenna with a 12-segment funiculus: 13 in all, a male's count
    case sliding = 2      // stance feet dragged along with the body instead of planted
    case rewind = 3       // the loop played forward then backward
    case press = 4        // an antenna tip pushed 0.02 mm into the juice at every touch-down
    case hover = 5        // the antenna tips held 0.1 mm off the juice at touch-down
    case tastePores = 6   // the taste hair given many wall pores instead of one tip pore
    case noOdour = 7      // the odour molecules taken away — but a cut apple smells
    case pieceSize = 8    // the apple piece drawn at half its size
    case formula = 9      // fructose one hydrogen short: C6H11O6
    case dryGlossa = 10   // the glossa stops 0.05 mm short of the juice: "drinking" nothing
    case stillTap = 11    // the tap's lift cut to 3 µm: moving, but invisibly
}

// MARK: - the ant: Lasius niger, worker

// Size. Workers are 3.4–5.0 mm long (AntWiki, "Lasius niger", and Wikipedia,
// "Black garden ant", both 3.5–5 mm; NatureSpot 3.4–5 mm).
let workerLengthRange: ClosedRange<Float> = 3.4...5.0

// The head, from Seifert's revision of Lasius s. str. (Seifert 2020, *Soil
// Organisms* 92(1): 15–86, the L. niger worker description): CS 976 µm, where
// CS is the mean of head length CL and head width CW; CL/CW 1.074; SL/CS
// 0.979; EYE/CS 0.245. The three lengths below are DERIVED from those, not
// typed: CW = 2·CS/(1 + CL/CW), CL = CS·2 − CW.
let cephalicSize: Float = 0.976
let headLengthOverWidth: Float = 1.074
let headWidth: Float = 2 * cephalicSize / (1 + headLengthOverWidth)   // 0.941 mm
let headLength: Float = 2 * cephalicSize - headWidth                   // 1.011 mm
let scapeLength: Float = 0.979 * cephalicSize                          // 0.955 mm
let eyeLength: Float = 0.245 * cephalicSize                            // 0.239 mm

// The antenna. Worker and queen Lasius antennae have 12 segments, males 13
// (Hölldobler & Wilson, *The Ants*, 1990; the Lasius genus diagnosis on
// AntWiki and in the Zenodo-deposited Lasius treatment: "antennae 12
// segmented in the worker and female, 13 segmented in the male"). Elbowed:
// a long scape, then the funiculus — here 11 segments, widening "gradually
// toward the apex but without a distinct club" (same genus diagnosis).
let workerAntennaSegments: Int = 12

/// Funiculus segment lengths, base to apex, in mm. MODEL (step 26): the
/// pedicel longer than the next few, the apical one about twice the
/// preapical, and the total 1.25× the scape, as drawings of Lasius show.
func funiculusLengths(count: Int) -> [Float] {
    let total: Float = 1.25 * scapeLength
    var weights: [Float] = [1.5]
    for _ in 0..<(count - 2) { weights.append(1.0) }
    weights.append(2.0)
    let sum: Float = weights.reduce(0, +)
    return weights.map { $0 / sum * total }
}

/// Antenna radii in mm: scape 0.04, funiculus widening from 0.032 to 0.048.
/// MODEL, sized from the same drawings (step 26).
let scapeRadius: Float = 0.040
let funiculusBaseRadius: Float = 0.032
let funiculusTipRadius: Float = 0.048

// MARK: - shapes, as the GPU draws them

/// Which part of the ant a primitive belongs to. The raw values go to the
/// kernel, and the tests count by them.
enum AntPart: Int {
    case head = 1, mesosoma = 2, petiole = 3, gaster = 4
    case leg = 5, antenna = 6, eye = 7, mandible = 8, glossa = 9
}

/// A round cone (a capsule whose two ends may differ in radius), an
/// ellipsoid with its own axes, or a rounded box.
struct Shape {
    enum Kind: Int { case roundCone = 0, ellipsoid = 1, roundBox = 2 }
    var kind: Kind
    var part: AntPart
    var a: SIMD3<Float>        // round cone: first end; ellipsoid, box: centre
    var b: SIMD3<Float>        // round cone: second end; ellipsoid, box: semi-axes
    var ra: Float = 0
    var rb: Float = 0
    var xAxis: SIMD3<Float> = SIMD3(1, 0, 0)   // ellipsoid and box only
    var yAxis: SIMD3<Float> = SIMD3(0, 1, 0)   // ellipsoid and box only
    var light: Bool = false    // the paler, reddish-brown cuticle
    var index: Int = 0         // which leg or antenna (0-based), for the tests

    static func cone(_ part: AntPart, _ a: SIMD3<Float>, _ b: SIMD3<Float>, _ ra: Float, _ rb: Float,
                     light: Bool = false, index: Int = 0) -> Shape {
        Shape(kind: .roundCone, part: part, a: a, b: b, ra: ra, rb: rb, light: light, index: index)
    }

    static func slab(_ part: AntPart, _ c: SIMD3<Float>, _ semi: SIMD3<Float>, round: Float,
                     x: SIMD3<Float>, y: SIMD3<Float>) -> Shape {
        Shape(kind: .roundBox, part: part, a: c, b: semi, ra: round, xAxis: simd_normalize(x), yAxis: simd_normalize(y))
    }

    static func blob(_ part: AntPart, _ c: SIMD3<Float>, _ semi: SIMD3<Float>,
                     x: SIMD3<Float> = SIMD3(1, 0, 0), y: SIMD3<Float> = SIMD3(0, 1, 0)) -> Shape {
        Shape(kind: .ellipsoid, part: part, a: c, b: semi, xAxis: simd_normalize(x), yAxis: simd_normalize(y))
    }

    /// Is a point inside, by the shape's own implicit function?
    func contains(_ p: SIMD3<Float>) -> Bool {
        switch kind {
        case .ellipsoid:
            let z: SIMD3<Float> = simd_cross(xAxis, yAxis)
            let q: SIMD3<Float> = p - a
            let l = SIMD3<Float>(simd_dot(q, xAxis), simd_dot(q, yAxis), simd_dot(q, z)) / b
            return simd_length_squared(l) <= 1
        case .roundCone:
            return roundConeDistance(p, a, b, ra, rb) <= 0
        case .roundBox:
            let z: SIMD3<Float> = simd_cross(xAxis, yAxis)
            let q: SIMD3<Float> = p - a
            let l = SIMD3<Float>(abs(simd_dot(q, xAxis)), abs(simd_dot(q, yAxis)), abs(simd_dot(q, z))) - b
            return simd_length(simd_max(l, SIMD3<Float>(repeating: 0))) + min(max(l.x, max(l.y, l.z)), 0) - ra <= 0
        }
    }

    /// Furthest reach along a direction.
    func extent(along d: SIMD3<Float>) -> (lo: Float, hi: Float) {
        switch kind {
        case .roundCone:
            let pa: Float = simd_dot(a, d)
            let pb: Float = simd_dot(b, d)
            return (min(pa - ra, pb - rb), max(pa + ra, pb + rb))
        case .ellipsoid:
            let z: SIMD3<Float> = simd_cross(xAxis, yAxis)
            let ex: Float = simd_dot(xAxis, d) * b.x
            let ey: Float = simd_dot(yAxis, d) * b.y
            let ez: Float = simd_dot(z, d) * b.z
            let r: Float = (ex * ex + ey * ey + ez * ez).squareRoot()
            let c: Float = simd_dot(a, d)
            return (c - r, c + r)
        case .roundBox:
            let z: SIMD3<Float> = simd_cross(xAxis, yAxis)
            let r: Float = abs(simd_dot(xAxis, d)) * b.x + abs(simd_dot(yAxis, d)) * b.y + abs(simd_dot(z, d)) * b.z + ra
            let c: Float = simd_dot(a, d)
            return (c - r, c + r)
        }
    }

    /// A sphere that holds the whole shape.
    var bound: (centre: SIMD3<Float>, radius: Float) {
        switch kind {
        case .roundCone: return ((a + b) / 2, simd_distance(a, b) / 2 + max(ra, rb))
        case .ellipsoid: return (a, max(b.x, max(b.y, b.z)))
        case .roundBox: return (a, simd_length(b) + ra)
        }
    }

    /// This shape moved into the world: scaled by `s` about the ant's own
    /// origin (a point on the ground), turned by `yaw` about the vertical,
    /// and put down at `at`. Uniform scale keeps every distance honest.
    func placed(scale s: Float, yaw: Float, at: SIMD3<Float>) -> Shape {
        let c: Float = cos(yaw)
        let n: Float = sin(yaw)
        // Body +x goes to (cos, 0, sin); body +z (the right side) to (−sin, 0, cos).
        func turn(_ v: SIMD3<Float>) -> SIMD3<Float> { SIMD3<Float>(v.x * c - v.z * n, v.y, v.x * n + v.z * c) }
        var out: Shape = self
        switch kind {
        case .roundCone:
            out.a = at + turn(a) * s
            out.b = at + turn(b) * s
        case .ellipsoid, .roundBox:
            out.a = at + turn(a) * s
            out.b = b * s
            out.xAxis = turn(xAxis)
            out.yAxis = turn(yAxis)
        }
        out.ra = ra * s
        out.rb = rb * s
        return out
    }
}

/// Exact distance to a round cone (Quílez's formulation), in Swift so the
/// tests can check contact without the GPU as well as with it.
func roundConeDistance(_ p: SIMD3<Float>, _ a: SIMD3<Float>, _ b: SIMD3<Float>, _ r1: Float, _ r2: Float) -> Float {
    let ba: SIMD3<Float> = b - a
    let l2: Float = simd_dot(ba, ba)
    let rr: Float = r1 - r2
    let a2: Float = l2 - rr * rr
    let il2: Float = 1.0 / l2
    let pa: SIMD3<Float> = p - a
    let y: Float = simd_dot(pa, ba)
    let z: Float = y - l2
    let xv: SIMD3<Float> = pa * l2 - ba * y
    let x2: Float = simd_dot(xv, xv)
    let y2: Float = y * y * l2
    let z2: Float = z * z * l2
    let k: Float = (rr >= 0 ? 1 : -1) * rr * rr * x2
    if (z >= 0 ? 1 : -1) * a2 * z2 > k { return (x2 + z2).squareRoot() * il2 - r2 }
    if (y >= 0 ? 1 : -1) * a2 * y2 < k { return (x2 + y2).squareRoot() * il2 - r1 }
    return ((x2 * a2 * il2).squareRoot() + y * rr) * il2 - r1
}

// MARK: - the body

// The body plan of an ant (Hölldobler & Wilson 1990; Bolton's glossary): head;
// mesosoma, carrying all six legs; the petiole — in Formicinae, and so in
// Lasius, a single waist segment, a vertical scale (AntWiki Lasius genus page:
// "petiole ... vertical and scale-like"); then the gaster. Ellipsoid sizes are
// step 26's MODEL, sized to Seifert's head and a length in the cited range.

let headCentre = SIMD3<Float>(1.36, 0.70, 0)
let gasterCentre = SIMD3<Float>(-1.30, 0.58, 0)
let gasterSemiAxes = SIMD3<Float>(0.74, 0.48, 0.52)
let gasterAxis: SIMD3<Float> = simd_normalize(SIMD3<Float>(0.99, 0.12, 0))
/// The head is prognathous and pitched 15° nose-down. MODEL (step 26). A
/// trail follower carries its head low, antennae forward and down.
let headPitch: Float = 15 * Float.pi / 180
let headForward = SIMD3<Float>(cos(headPitch), -sin(headPitch), 0)
let headUp = SIMD3<Float>(sin(headPitch), cos(headPitch), 0)

/// Where the gaster hinges when it bends down to dab: the back of the
/// petiole scale. MODEL — the gaster articulates on the petiole.
let gasterPivot = SIMD3<Float>(-0.54, 0.62, 0)

/// Rotate a point about the pivot by `angle` about +z: positive swings the
/// gaster's tail (−x) down.
func bendGaster(_ p: SIMD3<Float>, _ angle: Float) -> SIMD3<Float> {
    let q: SIMD3<Float> = p - gasterPivot
    let c: Float = cos(angle)
    let s: Float = sin(angle)
    return gasterPivot + SIMD3<Float>(q.x * c - q.y * s, q.x * s + q.y * c, q.z)
}

/// The gaster ellipsoid, bent down by `angle`.
func gasterShape(angle: Float) -> Shape {
    let x: SIMD3<Float> = bendGaster(gasterPivot + gasterAxis, angle) - gasterPivot
    let y = SIMD3<Float>(-x.y, x.x, 0)
    return .blob(.gaster, bendGaster(gasterCentre, angle), gasterSemiAxes, x: x, y: y)
}

// The gaster's tip. A formicine gaster ends in the acidopore, a small nozzle
// at the tip of the last sternite (Hölldobler & Wilson 1990; Bolton's
// glossary), and L. niger lays trail by "firmly pressing the tip of their
// abdomen onto the substrate" (Czaczkes et al. 2016, cited in Walk.swift).
// An ellipsoid is blunt: bent down, it would touch with its belly. So the
// last segments are drawn as a tapering cone from inside the rear of the
// gaster to a narrow tip, which flexes down on its own joint — the way the
// telescoped end segments of a gaster curl. Sizes MODEL.
let gasterTipJoint: Float = 0.72        // fraction of the long semi-axis behind the centre
let gasterTipDrop: Float = 0.20         // mm below the long axis: the ventral side
let gasterTipLength: Float = 0.28       // mm, joint to the tip's end
let gasterTipRadii: (Float, Float) = (0.22, 0.045)

/// The tip cone, for a gaster bent by `angle` and a tip flexed by `flex`.
func gasterTipShape(angle: Float, flex: Float) -> Shape {
    let g: Shape = gasterShape(angle: angle)
    let joint: SIMD3<Float> = g.a - g.xAxis * (gasterSemiAxes.x * gasterTipJoint) - g.yAxis * gasterTipDrop
    // Back along the gaster's axis, turned down by `flex` about +z.
    let back: SIMD3<Float> = -g.xAxis
    let c: Float = cos(flex)
    let sn: Float = sin(flex)
    let dir = SIMD3<Float>(back.x * c - back.y * sn, back.x * sn + back.y * c, 0)
    return .cone(.gaster, joint, joint + simd_normalize(dir) * gasterTipLength, gasterTipRadii.0, gasterTipRadii.1)
}

/// The dab, solved rather than typed: the gaster bends down until its belly
/// is 0.04 mm off the ground (MODEL clearance), then the tip flexes until its
/// lowest point just meets the ground. Returns (bend, flex), by bisection on
/// exact lowest points.
func dabPose() -> (bend: Float, flex: Float) {
    var lo: Float = 0
    var hi: Float = 1.2
    for _ in 0..<60 {
        let mid: Float = (lo + hi) / 2
        if gasterShape(angle: mid).extent(along: SIMD3(0, 1, 0)).lo > 0.04 { lo = mid } else { hi = mid }
    }
    let bend: Float = lo
    var flo: Float = 0
    var fhi: Float = 1.5
    for _ in 0..<60 {
        let mid: Float = (flo + fhi) / 2
        if gasterTipShape(angle: bend, flex: mid).extent(along: SIMD3(0, 1, 0)).lo > 0 { flo = mid } else { fhi = mid }
    }
    return (bend, flo)
}

// The glossa. Ants take up liquid with the glossa, the tongue-like tip of
// the labium: "Ants were observed to employ two different techniques for
// liquid food intake, in which the glossa works either as a passive
// duct-like structure (sucking), or as an up- and downwards moving shovel
// (licking)"; species "with a crop capable of storing liquids, sucked the
// fluid food, such as formicine ants of the genus Camponotus" (Paul & Roces,
// *J Insect Physiol* 49: 347–357, 2003, abstract via Europe PMC). Lasius is a
// formicine with a crop, but was not among the species filmed: that it sucks
// is inferred from Camponotus, not observed. So here the glossa is pushed out
// to the juice and held there, still, while the ant drinks — no lapping.
// Its shape, root and reach are MODEL: a soft pale cone from inside the head
// out under the clypeus, between the mandibles, angled 35° below the body's
// forward axis, 0.045 mm thick at the root and 0.028 at the tip.
let glossaRootAhead: SIMD3<Float> = headForward * 0.30
let glossaRootDown: SIMD3<Float> = headUp * 0.12
let glossaRoot: SIMD3<Float> = headCentre + glossaRootAhead - glossaRootDown
let glossaMouthAhead: SIMD3<Float> = headForward * 0.50
let glossaMouthDown: SIMD3<Float> = headUp * 0.20
let glossaMouth: SIMD3<Float> = headCentre + glossaMouthAhead - glossaMouthDown
let glossaDirection: SIMD3<Float> = simd_normalize(SIMD3<Float>(0.82, -0.57, 0))
let glossaRadii: (root: Float, tip: Float) = (0.045, 0.028)
/// How far the glossa can reach out of the mouth, mm. MODEL: a few tenths of
/// a millimetre, about a third of the head's length; a test holds the drink
/// to it.
let glossaMaxReach: Float = 0.32

/// The glossa, pushed `reach` mm out of the mouth (0: its tip at the mouth).
func glossaShape(reach: Float) -> Shape {
    let tip: SIMD3<Float> = glossaMouth + glossaDirection * reach
    return .cone(.glossa, glossaRoot, tip, glossaRadii.root, glossaRadii.tip, light: true)
}

/// The body without its appendages, the gaster bent by `gasterBend` and its
/// tip flexed by `tipFlex`, the glossa out by `glossa` mm.
func bodyShapes(gasterBend: Float, tipFlex: Float = 0, glossa: Float = 0) -> [Shape] {
    let side = SIMD3<Float>(0, 0, 1)
    var s: [Shape] = []
    let hx: Float = headLength / 2 - 0.02
    s.append(.blob(.head, headCentre - headForward * 0.02, SIMD3(hx, 0.30, headWidth / 2), x: headForward, y: headUp))
    s.append(.blob(.head, headCentre + headForward * 0.40 - headUp * 0.05, SIMD3(0.14, 0.13, 0.20), x: headForward, y: headUp))
    for sgn in [Float(1), Float(-1)] {
        let c: SIMD3<Float> = headCentre - headForward * 0.03 + headUp * 0.06 + side * (sgn * (headWidth / 2 - 0.035))
        s.append(.blob(.eye, c, SIMD3(eyeLength / 2, eyeLength * 0.36, 0.085), x: headForward, y: headUp))
    }
    s.append(.cone(.head, headCentre - headForward * 0.45, SIMD3(0.80, 0.66, 0), 0.10, 0.09))
    s.append(.blob(.mesosoma, SIMD3(0.54, 0.70, 0), SIMD3(0.35, 0.25, 0.23)))
    s.append(.blob(.mesosoma, SIMD3(0.16, 0.69, 0), SIMD3(0.28, 0.23, 0.19)))
    s.append(.blob(.mesosoma, SIMD3(-0.20, 0.62, 0), SIMD3(0.23, 0.19, 0.17)))
    s.append(.slab(.petiole, SIMD3(-0.48, 0.64, 0), SIMD3(0.004, 0.15, 0.06), round: 0.07,
                   x: SIMD3(0.97, 0.24, 0), y: SIMD3(-0.24, 0.97, 0)))
    s.append(gasterShape(angle: gasterBend))
    s.append(gasterTipShape(angle: gasterBend, flex: tipFlex))
    for sgn in [Float(1), Float(-1)] {
        let root: SIMD3<Float> = headCentre + headForward * 0.43 - headUp * 0.12 + side * (sgn * 0.17)
        let tip: SIMD3<Float> = headCentre + headForward * 0.66 - headUp * 0.17 + side * (sgn * 0.03)
        s.append(.cone(.mandible, root, tip, 0.06, 0.022, light: true))
    }
    s.append(glossaShape(reach: glossa))
    return s
}

// MARK: - the legs

/// One leg's plan (step 26). Segment lengths in mm are MODEL: formicine
/// proportions, hind femur about the head length.
struct LegSpec {
    var name: String
    var coxaRoot: SIMD3<Float>   // where it joins the body — right side; mirrored for left
    var coxa: Float
    var femur: Float
    var tibia: Float
    var tarsus: Float
}

let legSpecs: [LegSpec] = [
    LegSpec(name: "fore", coxaRoot: SIMD3(0.62, 0.52, 0.10), coxa: 0.28, femur: 0.78, tibia: 0.72, tarsus: 0.78),
    LegSpec(name: "mid", coxaRoot: SIMD3(0.18, 0.53, 0.11), coxa: 0.22, femur: 0.84, tibia: 0.80, tarsus: 0.84),
    LegSpec(name: "hind", coxaRoot: SIMD3(-0.10, 0.50, 0.10), coxa: 0.26, femur: 0.98, tibia: 0.94, tarsus: 1.02),
]

/// The tarsus radius (step 26), and the radius of its last bead's round end.
let tarsusRadius: Float = 0.022
let tarsusTipRadius: Float = tarsusRadius * 0.85

/// A built leg: its segment shapes, the point where it meets the body, and
/// the joints, so a test can check no segment was stretched.
struct Leg {
    var name: String
    var side: Float
    var root: SIMD3<Float>
    var hip: SIMD3<Float>
    var knee: SIMD3<Float>
    var ankle: SIMD3<Float>
    var shapes: [Shape]
}

/// Two-link reach: the knee, given hip, ankle, the two lengths, and which way
/// the knee should bow (up and out, as an ant's does).
func knee(hip: SIMD3<Float>, ankle: SIMD3<Float>, femur: Float, tibia: Float, bow: SIMD3<Float>) -> SIMD3<Float> {
    let d: SIMD3<Float> = ankle - hip
    let len: Float = min(simd_length(d), femur + tibia - 1e-4)
    let u: SIMD3<Float> = simd_normalize(d)
    let along: Float = (femur * femur - tibia * tibia + len * len) / (2 * len)
    let h: Float = max(femur * femur - along * along, 0).squareRoot()
    let perp: SIMD3<Float> = simd_normalize(bow - u * simd_dot(bow, u))
    return hip + u * along + perp * h
}

/// Leg `index` 0–5: right fore, mid, hind, then left fore, mid, hind. `foot`
/// is where the lowest point of the last tarsomere is, in the ant's frame —
/// on the ground (y = 0) in stance, lifted in swing. Step 26's construction:
/// the coxa hangs down and a little out, the tarsus lies along the ground
/// pointing away from the body, femur and tibia reach between.
func buildLeg(index: Int, foot: SIMD3<Float>) -> Leg {
    let side: Float = index < 3 ? 1 : -1
    let spec: LegSpec = legSpecs[index % 3]
    let flip = SIMD3<Float>(1, 1, side)
    let root: SIMD3<Float> = spec.coxaRoot * flip
    let coxaEnd: SIMD3<Float> = root + simd_normalize(SIMD3<Float>(0, -1, 0.35 * side)) * spec.coxa
    let hip: SIMD3<Float> = coxaEnd + SIMD3<Float>(0, -0.03, 0.05 * side)      // trochanter
    var outward: SIMD3<Float> = foot - SIMD3<Float>(hip.x, 0, hip.z)
    outward = simd_normalize(SIMD3<Float>(outward.x, 0, outward.z))
    let tipPoint = SIMD3<Float>(foot.x, foot.y + tarsusTipRadius, foot.z)
    let ankle: SIMD3<Float> = SIMD3<Float>(foot.x, foot.y, foot.z) - outward * (spec.tarsus * 0.97) + SIMD3<Float>(0, 0.07, 0)
    let bow: SIMD3<Float> = SIMD3<Float>(0, 1.2, 0) + outward
    let k: SIMD3<Float> = knee(hip: hip, ankle: ankle, femur: spec.femur, tibia: spec.tibia, bow: bow)
    var s: [Shape] = []
    s.append(.cone(.leg, root, coxaEnd, 0.075, 0.055, index: index))
    s.append(.cone(.leg, coxaEnd, hip, 0.050, 0.045, index: index))
    s.append(.cone(.leg, hip, k, 0.058, 0.044, index: index))
    s.append(.cone(.leg, k, ankle, 0.040, 0.028, index: index))
    // Five tarsomeres, the basitarsus the longest, the last ending on the ground.
    let fractions: [Float] = [0.40, 0.16, 0.13, 0.12, 0.19]
    var at: SIMD3<Float> = ankle
    let run: SIMD3<Float> = tipPoint - ankle
    for (i, f) in fractions.enumerated() {
        let next: SIMD3<Float> = at + run * f
        let last: Bool = i == fractions.count - 1
        s.append(.cone(.leg, at, next, tarsusRadius * 1.05, last ? tarsusTipRadius : tarsusRadius * 0.85, light: true, index: index))
        at = next
    }
    return Leg(name: "\(side > 0 ? "right" : "left") \(spec.name)", side: side, root: root, hip: hip, knee: k,
               ankle: ankle, shapes: s)
}

// MARK: - the antennae

/// A built antenna: its segments and where its tip is.
struct Antenna {
    var side: Float
    var socket: SIMD3<Float>
    var elbow: SIMD3<Float>
    var segments: [Shape]        // scape first, then the funiculus
    var tipCentre: SIMD3<Float>
    var reached: Bool            // could the funiculus bend to reach the tip asked for?
    var tipRadius: Float { funiculusTipRadius }
}

/// Points along a circular arc of length `length` from `p0` to `p1`, bowing
/// towards `bow` (step 26).
func arcPoints(from p0: SIMD3<Float>, to p1: SIMD3<Float>, length: Float, bow: SIMD3<Float>,
               at fractions: [Float]) -> [SIMD3<Float>] {
    let chordV: SIMD3<Float> = p1 - p0
    let chord: Float = simd_length(chordV)
    let u: SIMD3<Float> = chordV / chord
    let w: SIMD3<Float> = simd_normalize(bow - u * simd_dot(bow, u))
    if chord >= length * 0.9999 { return fractions.map { p0 + u * (length * $0) } }
    var lo: Float = 1e-4
    var hi: Float = 2 * Float.pi - 1e-3
    for _ in 0..<60 {
        let mid: Float = (lo + hi) / 2
        let c: Float = 2 * (length / mid) * sin(mid / 2)
        if c > chord { lo = mid } else { hi = mid }
    }
    let theta: Float = (lo + hi) / 2
    let r: Float = length / theta
    let mid: SIMD3<Float> = (p0 + p1) / 2
    let sag: Float = r * cos(theta / 2)
    let centre: SIMD3<Float> = mid - w * sag
    let e1: SIMD3<Float> = simd_normalize(p0 - centre)
    let e2raw: SIMD3<Float> = (p1 - centre) - e1 * simd_dot(p1 - centre, e1)
    let e2: SIMD3<Float> = simd_normalize(e2raw)
    return fractions.map { f in
        let a: Float = theta * f
        return centre + (e1 * cos(a) + e2 * sin(a)) * r
    }
}

/// Where each antenna rises from the head: the torulus, on the frons behind
/// the clypeus, one either side of the midline (step 26).
func antennaSocket(side sgn: Float) -> SIMD3<Float> {
    headCentre + headForward * 0.26 + headUp * 0.13 + SIMD3<Float>(0, 0, sgn * 0.15)
}

/// Antenna `index` 0 right, 1 left, with its scape along `scapeDir` and its
/// tip centred at `tip`, both in the ant's frame. Step 68: the funiculus bows
/// towards `bow` (step 44's was always forward and up), and the segments13
/// mutant gives it one segment too many.
func buildAntenna(index: Int, scapeDir: SIMD3<Float>, tip: SIMD3<Float>, bow: SIMD3<Float> = SIMD3<Float>(0.4, 1, 0),
                  mutant: Mutant = .none) -> Antenna {
    let sgn: Float = index == 0 ? 1 : -1
    let funiculusCount: Int = mutant == .segments13 ? workerAntennaSegments : workerAntennaSegments - 1
    let lengths: [Float] = funiculusLengths(count: funiculusCount)
    let total: Float = lengths.reduce(0, +)
    var fractions: [Float] = [0]
    var run: Float = 0
    for l in lengths { run += l; fractions.append(run / total) }
    let socket: SIMD3<Float> = antennaSocket(side: sgn)
    let elbow: SIMD3<Float> = socket + simd_normalize(scapeDir) * scapeLength
    var segs: [Shape] = [.cone(.antenna, socket, elbow, scapeRadius * 0.8, scapeRadius, light: true, index: index)]
    let pts: [SIMD3<Float>] = arcPoints(from: elbow, to: tip, length: total, bow: bow, at: fractions)
    for i in 0..<funiculusCount {
        let f0: Float = fractions[i]
        let f1: Float = fractions[i + 1]
        let r0: Float = funiculusBaseRadius + (funiculusTipRadius - funiculusBaseRadius) * f0
        let r1: Float = funiculusBaseRadius + (funiculusTipRadius - funiculusBaseRadius) * f1
        segs.append(.cone(.antenna, pts[i], pts[i + 1], r0 * 0.78, r1, index: index))
    }
    let reached: Bool = simd_distance(elbow, tip) < total * 0.9999
    return Antenna(side: sgn, socket: socket, elbow: elbow, segments: segs, tipCentre: pts[pts.count - 1], reached: reached)
}

// MARK: - the whole ant

/// One ant in its own frame.
struct AntModel {
    var body: [Shape]
    var legs: [Leg]
    var antennae: [Antenna]
    var shapes: [Shape] { body + legs.flatMap { $0.shapes } + antennae.flatMap { $0.segments } }

    /// Body length from mandible tips to the tip of the gaster, along the
    /// body axis — how the field guides measure it.
    var length: Float {
        let parts: [Shape] = body.filter { $0.part != .eye }
        let lo: Float = parts.map { $0.extent(along: SIMD3(1, 0, 0)).lo }.min() ?? 0
        let hi: Float = parts.map { $0.extent(along: SIMD3(1, 0, 0)).hi }.max() ?? 0
        return hi - lo
    }
}

// MARK: - the two sense hairs, in micrometres

// Insect chemosensory hairs are classed by their pores (Altner & Prillinger,
// *Int Rev Cytol* 67: 69–139, 1980; Zacharuk, *Annu Rev Entomol* 25: 27–47,
// 1980): no pores — mechanoreceptors; ONE pore, at the tip — uniporous,
// contact chemoreceptors, taste; MANY pores in the wall — multiporous,
// olfactory, smell. (As restated by Nowińska & Brożek, *Zoomorphology* 2017,
// and, with the caveat that the match is "rough but imperfect", in the
// Journal of Insect Science gustation review of 2023.) Ant antennae carry
// both: gustatory sensilla chaetica with a terminal pore and a flexible
// socket, and multiporous basiconic sensilla (Gellert et al., *Sci Rep* 12:
// 19328, 2022, eight ant species by SEM).
//
// Gellert et al. measured the wall pores of ant basiconic sensilla at
// 0.07 ± 0.009 µm (Camponotus pennsylvanicus) and 0.09 ± 0.013 µm
// (Harpegnathos saltator). The rest of the dimensions here are MODEL, inside
// the ranges SEM studies of formicine antennae show: chaetica tens of µm long
// and 1–2 µm thick at the base; basiconica shorter and blunt.

enum HairKind: Int {
    case taste = 0   // sensillum chaeticum, uniporous
    case smell = 1   // sensillum basiconicum, multiporous
}

struct Sensillum {
    var kind: HairKind
    var base: SIMD3<Float>
    var tip: SIMD3<Float>       // centre of the rounded apex
    var baseRadius: Float
    var tipRadius: Float
    var lumenFraction: Float    // inner radius / outer radius: the wall is hollow
    var tipPoreRadius: Float    // 0 = no tip pore
    var wallPoreRadius: Float   // 0 = no wall pores
    var poreSpacing: Float      // centre-to-centre, along the hair and around it
    var poreStart: Float        // fraction of the length where the pore field starts
    var poreEnd: Float          // … and ends, short of the apex cap
    var socket: Bool            // a flexible socket ring at the base

    var length: Float { simd_distance(base, tip) }

    /// Wall pores, as rows × columns on the pore lattice. Zero when none.
    var poreRows: Int {
        wallPoreRadius > 0 ? Int(((poreEnd - poreStart) * length / poreSpacing).rounded(.down)) + 1 : 0
    }
    var poreColumns: Int {
        let meanR: Float = (baseRadius + tipRadius) / 2
        return wallPoreRadius > 0 ? max(Int((2 * Float.pi * meanR / poreSpacing).rounded()), 1) : 0
    }
    var wallPoreCount: Int { poreRows * poreColumns }
}

/// The film where the tip touches, µm (step 50's, copied: the apple's juice
/// is, like the corn's, a watery sugar solution). The juice needs no film:
/// it is the liquid. What the inset shows is the juice's meniscus climbing
/// the hair where the apex meets it — step 35's choice for honey, another
/// sugar solution: a contact sensillum wets what it touches, its terminal pore
/// filled with sensillum lymph. (Step 39 drew pure water dimpling under a
/// water-repellent tip instead; no contact angle has been measured on a
/// Lasius antenna, for either.) Its height is chosen to be visible, not
/// measured. MODEL.
let filmThickness: Float = 0.0     // no separate film: the surface is juice
let filmRadius: Float = 2.8
let meniscusHeight: Float = 0.45

/// The two hairs in the inset. The taste hair comes down towards the viewer
/// and rests its apex in the juice; the smell hair stands beside it in the air.
/// Its uniporous tip pore is 0.4 µm across (MODEL; terminal pores of
/// gustatory sensilla are sub-micron).
func buildHairs(mutant: Mutant) -> [Sensillum] {
    let tasteTipR: Float = 0.55
    let tasteDir: SIMD3<Float> = simd_normalize(SIMD3<Float>(-0.18, -0.42, 0.89))
    // The apex cap's lowest point sits exactly on the crystal (y = 0).
    let tasteTip = SIMD3<Float>(0, tasteTipR, 0)
    let tasteBase: SIMD3<Float> = tasteTip - tasteDir * 13.0
    var taste = Sensillum(kind: .taste, base: tasteBase, tip: tasteTip, baseRadius: 1.0, tipRadius: tasteTipR,
                          lumenFraction: 0.55, tipPoreRadius: 0.25, wallPoreRadius: 0, poreSpacing: 0.3,
                          poreStart: 0.25, poreEnd: 0.93, socket: true)
    if mutant == .tastePores {
        taste.tipPoreRadius = 0
        taste.wallPoreRadius = 0.04
    }
    let smellDir: SIMD3<Float> = simd_normalize(SIMD3<Float>(-0.05, -0.80, 0.60))
    let smellTip = SIMD3<Float>(3.6, 2.2, 1.0)
    let smellBase: SIMD3<Float> = smellTip - smellDir * 8.5
    let smell = Sensillum(kind: .smell, base: smellBase, tip: smellTip, baseRadius: 0.95, tipRadius: 0.6,
                          lumenFraction: 0.72, tipPoreRadius: 0, wallPoreRadius: 0.04, poreSpacing: 0.3,
                          poreStart: 0.30, poreEnd: 0.95, socket: false)
    return [taste, smell]
}

/// The antenna's apical segment as the inset sees it: a sphere whose surface
/// carries both hair bases. At 48 µm radius (the main view's funiculus tip
/// radius) it is nearly flat at this scale.
func antennaDome(_ hairs: [Sensillum]) -> (centre: SIMD3<Float>, radius: Float) {
    let r: Float = funiculusTipRadius * 1000
    // Put the centre above and behind both bases so each base sits on the
    // surface to within the dome's curvature; the hair bases are buried a
    // little way in, so they join the cuticle cleanly.
    // The surface normal there: as near "up and back" as it can be while
    // lying square to the line between the two bases, so both sit on it.
    let mid: SIMD3<Float> = (hairs[0].base + hairs[1].base) / 2
    let between: SIMD3<Float> = simd_normalize(hairs[1].base - hairs[0].base)
    let want: SIMD3<Float> = simd_normalize(SIMD3<Float>(0.1, 1.0, -0.5))
    let up: SIMD3<Float> = simd_normalize(want - between * simd_dot(want, between))
    return (mid + up * (r - 1.7), r)
}


extension Sensillum {
    /// The frame the pore lattice is laid out in — the same one gpuHairs
    /// sends to the kernel: the axis, and two directions square to it.
    var frame: (u: SIMD3<Float>, e1: SIMD3<Float>, e2: SIMD3<Float>) {
        let u: SIMD3<Float> = simd_normalize(tip - base)
        let helper: SIMD3<Float> = abs(u.y) < 0.9 ? SIMD3(0, 1, 0) : SIMD3(1, 0, 0)
        let e1: SIMD3<Float> = simd_normalize(simd_cross(u, helper))
        let e2: SIMD3<Float> = simd_cross(u, e1)
        return (u, e1, e2)
    }

    /// The mouth of one wall pore on the outer surface, and the outward
    /// direction there: row `i` along the hair, column `j` round it, on the
    /// same staggered lattice the kernel carves.
    func wallPore(row i: Int, column j: Int) -> (mouth: SIMD3<Float>, outward: SIMD3<Float>) {
        let (u, e1, e2) = frame
        let tc: Float = poreStart * length + Float(i) * poreSpacing
        let stagger: Float = Float(i % 2) * 0.5
        let phi: Float = (Float(j) + stagger) * 2 * Float.pi / Float(max(poreColumns, 1))
        let out: SIMD3<Float> = e1 * cos(phi) + e2 * sin(phi)
        let r: Float = baseRadius + (tipRadius - baseRadius) * tc / length
        return (base + u * tc + out * r, out)
    }
}
