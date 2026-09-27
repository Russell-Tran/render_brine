// The ant, the sugar and the two sense hairs as numbers. Nothing here touches
// the GPU; it is the part a test can read against the literature.
//
// Three scales, three unit systems, and they are never mixed:
//
//   * the main view in MILLIMETRES — y up, the table top is y = 0, the ant
//     faces +x and its right side is +z, which is the side the camera is on;
//   * the round inset in MICROMETRES — y up, the sugar crystal's face is y = 0,
//     the point where the taste hair touches it is the origin;
//   * the molecule inset in ÅNGSTRÖMS, straight from the PDB file.
//
// Every number carries where it came from, or says MODEL and why.

import Foundation
import simd

// MARK: - what to break, for the mutation check

/// Each one must make the test suite fail.
enum Mutant: Int {
    case none = 0
    case segments13 = 1   // an antenna with a 12-segment funiculus: 13 in all, a male's count
    case hover = 2        // the antenna tip held 0.1 mm off the sugar
    case tastePores = 3   // the taste hair given many wall pores instead of one tip pore
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
// a long scape, then the funiculus — here 11 segments. Lasius antennae
// broaden "gradually toward the apex but without a distinct club" (same
// genus diagnosis), so the funiculus widens smoothly and no segment jumps.
let workerAntennaSegments: Int = 12

/// Funiculus segment lengths, base to apex, in mm. MODEL: no per-segment
/// table to hand for L. niger. Proportions follow the usual formicine
/// pattern — the first funicular segment (pedicel) longer than the next few,
/// the apical one about twice the preapical — and the total, 1.25× the scape,
/// is what drawings of Lasius workers show (the funiculus is longer than the
/// scape but not twice it).
func funiculusLengths(count: Int) -> [Float] {
    let total: Float = 1.25 * scapeLength
    var weights: [Float] = [1.5]
    for _ in 0..<(count - 2) { weights.append(1.0) }
    weights.append(2.0)
    let sum: Float = weights.reduce(0, +)
    return weights.map { $0 / sum * total }
}

/// Antenna radii in mm: scape 0.04, funiculus widening from 0.032 to 0.048
/// at the apex. MODEL, sized from the same drawings (a Lasius scape is about
/// a twelfth as thick as it is long).
let scapeRadius: Float = 0.040
let funiculusBaseRadius: Float = 0.032
let funiculusTipRadius: Float = 0.048

// MARK: - shapes, as the GPU draws them

/// Which part of the ant a primitive belongs to. The raw values go to the
/// kernel, and the tests count by them.
enum AntPart: Int {
    case head = 1, mesosoma = 2, petiole = 3, gaster = 4
    case leg = 5, antenna = 6, eye = 7, mandible = 8
}

/// A round cone (a capsule whose two ends may differ in radius), or an
/// ellipsoid with its own axes. Both have exact or near-exact distances.
struct Shape {
    enum Kind: Int { case roundCone = 0, ellipsoid = 1, roundBox = 2 }
    var kind: Kind
    var part: AntPart
    var a: SIMD3<Float>        // round cone: first end; ellipsoid: centre
    var b: SIMD3<Float>        // round cone: second end; ellipsoid: semi-axes
    var ra: Float = 0
    var rb: Float = 0
    var xAxis: SIMD3<Float> = SIMD3(1, 0, 0)   // ellipsoid only
    var yAxis: SIMD3<Float> = SIMD3(0, 1, 0)   // ellipsoid only
    var light: Bool = false    // the paler, reddish-brown cuticle
    var index: Int = 0         // which leg or antenna (0-based), for the tests

    static func cone(_ part: AntPart, _ a: SIMD3<Float>, _ b: SIMD3<Float>, _ ra: Float, _ rb: Float,
                     light: Bool = false, index: Int = 0) -> Shape {
        Shape(kind: .roundCone, part: part, a: a, b: b, ra: ra, rb: rb, light: light, index: index)
    }

    /// A box with rounded edges: `semi` is the half-size of the inner box,
    /// `ra` the rounding. Its distance is exact outside.
    static func slab(_ part: AntPart, _ c: SIMD3<Float>, _ semi: SIMD3<Float>, round: Float,
                     x: SIMD3<Float>, y: SIMD3<Float>) -> Shape {
        Shape(kind: .roundBox, part: part, a: c, b: semi, ra: round, xAxis: simd_normalize(x), yAxis: simd_normalize(y))
    }

    static func blob(_ part: AntPart, _ c: SIMD3<Float>, _ semi: SIMD3<Float>,
                     x: SIMD3<Float> = SIMD3(1, 0, 0), y: SIMD3<Float> = SIMD3(0, 1, 0)) -> Shape {
        Shape(kind: .ellipsoid, part: part, a: c, b: semi, xAxis: simd_normalize(x), yAxis: simd_normalize(y))
    }

    /// Is a point inside, by the shape's own implicit function? For the
    /// ellipsoid this is the exact test (x/a)² + (y/b)² + (z/c)² ≤ 1.
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

    /// Furthest reach along a direction: the ant's extent, for its length.
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
// "petiole ... vertical and scale-like"; one node, never the two of the
// Myrmicinae); then the gaster.
//
// The ellipsoid sizes below are MODEL, sized to the head from Seifert and to a
// total length in the cited 3.4–5.0 mm range. Weber's length and gaster size
// vary with how full the crop is; this worker is an ordinary one.

let headCentre = SIMD3<Float>(1.36, 0.70, 0)
let gasterCentre = SIMD3<Float>(-1.30, 0.58, 0)
let gasterSemiAxes = SIMD3<Float>(0.74, 0.48, 0.52)
let gasterAxis: SIMD3<Float> = simd_normalize(SIMD3<Float>(0.99, 0.12, 0))
/// The head is prognathous and pitched 15° nose-down towards the sugar. MODEL.
let headPitch: Float = 15 * Float.pi / 180
let headForward = SIMD3<Float>(cos(headPitch), -sin(headPitch), 0)
let headUp = SIMD3<Float>(sin(headPitch), cos(headPitch), 0)

/// The body without its appendages.
func bodyShapes() -> [Shape] {
    let side = SIMD3<Float>(0, 0, 1)
    var s: [Shape] = []
    // Head: CL along the head's axis, CW across. The head is a touch
    // flattened top to bottom, as in all Lasius. The last 0.08 mm of CL is
    // clypeus, drawn as a smaller blob so the front is not a ball.
    let hx: Float = headLength / 2 - 0.02
    s.append(.blob(.head, headCentre - headForward * 0.02, SIMD3(hx, 0.30, headWidth / 2), x: headForward, y: headUp))
    s.append(.blob(.head, headCentre + headForward * 0.40 - headUp * 0.05, SIMD3(0.14, 0.13, 0.20), x: headForward, y: headUp))
    // Compound eyes on the sides, a little behind the middle, EYE/CS from Seifert.
    for sgn in [Float(1), Float(-1)] {
        let c: SIMD3<Float> = headCentre - headForward * 0.03 + headUp * 0.06 + side * (sgn * (headWidth / 2 - 0.035))
        s.append(.blob(.eye, c, SIMD3(eyeLength / 2, eyeLength * 0.36, 0.085), x: headForward, y: headUp))
    }
    // Neck.
    s.append(.cone(.head, headCentre - headForward * 0.45, SIMD3(0.80, 0.66, 0), 0.10, 0.09))
    // Mesosoma: pronotum, mesonotum, propodeum, melted together.
    s.append(.blob(.mesosoma, SIMD3(0.54, 0.70, 0), SIMD3(0.35, 0.25, 0.23)))
    s.append(.blob(.mesosoma, SIMD3(0.16, 0.69, 0), SIMD3(0.28, 0.23, 0.19)))
    s.append(.blob(.mesosoma, SIMD3(-0.20, 0.62, 0), SIMD3(0.23, 0.19, 0.17)))
    // Petiole: one vertical scale, thin fore-and-aft, tipped slightly forward.
    s.append(.slab(.petiole, SIMD3(-0.48, 0.64, 0), SIMD3(0.004, 0.15, 0.06), round: 0.07,
                   x: SIMD3(0.97, 0.24, 0), y: SIMD3(-0.24, 0.97, 0)))
    // Gaster. Its four visible tergites overlap like roof tiles; the shader
    // draws their hind margins (Render.swift), the shape is one ellipsoid.
    s.append(.blob(.gaster, gasterCentre, gasterSemiAxes, x: gasterAxis, y: SIMD3(-gasterAxis.y, gasterAxis.x, 0)))
    // Mandibles: two blades closing in front of the clypeus. Seifert's
    // "mandibular dentation" is 8 teeth; at this scale each blade is a
    // tapering cone and the teeth are left out.
    for sgn in [Float(1), Float(-1)] {
        let root: SIMD3<Float> = headCentre + headForward * 0.43 - headUp * 0.12 + side * (sgn * 0.17)
        let tip: SIMD3<Float> = headCentre + headForward * 0.66 - headUp * 0.17 + side * (sgn * 0.03)
        s.append(.cone(.mandible, root, tip, 0.06, 0.022, light: true))
    }
    return s
}

// MARK: - the legs

/// One leg's plan. Segment lengths in mm are MODEL: formicine proportions
/// (hind leg longest, femur ≈ tibia, tarsus a little longer), scaled so the
/// hind femur is about the head length, as in Lasius workers.
struct LegSpec {
    var name: String
    var coxaRoot: SIMD3<Float>   // where it joins the body — right side; mirrored for left
    var foot: SIMD3<Float>       // where the tarsus tip rests on the table
    var coxa: Float
    var femur: Float
    var tibia: Float
    var tarsus: Float
}

/// Where the three coxae join the mesosoma — the pro-, meso- and metacoxae
/// on the underside of pronotum, mesonotum and propodeum — and where the
/// feet stand in a relaxed tripod. MODEL placements.
let legSpecs: [LegSpec] = [
    LegSpec(name: "fore", coxaRoot: SIMD3(0.62, 0.52, 0.10), foot: SIMD3(1.80, 0, 1.20),
            coxa: 0.28, femur: 0.78, tibia: 0.72, tarsus: 0.78),
    LegSpec(name: "mid", coxaRoot: SIMD3(0.18, 0.53, 0.11), foot: SIMD3(0.30, 0, 1.62),
            coxa: 0.22, femur: 0.84, tibia: 0.80, tarsus: 0.84),
    LegSpec(name: "hind", coxaRoot: SIMD3(-0.10, 0.50, 0.10), foot: SIMD3(-1.62, 0, 1.55),
            coxa: 0.26, femur: 0.98, tibia: 0.94, tarsus: 1.02),
]

/// A built leg: its segment shapes and the point where it meets the body.
struct Leg {
    var name: String
    var side: Float
    var root: SIMD3<Float>
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

func buildLegs() -> [Leg] {
    var legs: [Leg] = []
    var index: Int = 0
    for side in [Float(1), Float(-1)] {
        for spec in legSpecs {
            let flip = SIMD3<Float>(1, 1, side)
            let root: SIMD3<Float> = spec.coxaRoot * flip
            let foot: SIMD3<Float> = spec.foot * flip
            // The coxa hangs down and a little out.
            let coxaEnd: SIMD3<Float> = root + simd_normalize(SIMD3<Float>(0, -1, 0.35 * side)) * spec.coxa
            let hip: SIMD3<Float> = coxaEnd + SIMD3<Float>(0, -0.03, 0.05 * side)      // trochanter
            // The tarsus lies along the table, pointing outward from the body.
            var outward: SIMD3<Float> = foot - SIMD3<Float>(hip.x, 0, hip.z)
            outward = simd_normalize(SIMD3<Float>(outward.x, 0, outward.z))
            let tarsusR: Float = 0.022
            let ankle: SIMD3<Float> = foot - outward * (spec.tarsus * 0.97) + SIMD3<Float>(0, 0.07, 0)
            let bow: SIMD3<Float> = SIMD3<Float>(0, 0.7, 0) + outward
            let k: SIMD3<Float> = knee(hip: hip, ankle: ankle, femur: spec.femur, tibia: spec.tibia, bow: bow)
            var s: [Shape] = []
            s.append(.cone(.leg, root, coxaEnd, 0.075, 0.055, index: index))
            s.append(.cone(.leg, coxaEnd, hip, 0.050, 0.045, index: index))
            s.append(.cone(.leg, hip, k, 0.058, 0.044, index: index))
            s.append(.cone(.leg, k, ankle, 0.040, 0.028, light: false, index: index))
            // Five tarsomeres, the first the longest (the basitarsus), each a
            // little bead so the joints show, ending on the table.
            let fractions: [Float] = [0.40, 0.16, 0.13, 0.12, 0.19]
            var at: SIMD3<Float> = ankle
            let tipPoint = SIMD3<Float>(foot.x, tarsusR, foot.z)
            let run: SIMD3<Float> = tipPoint - ankle
            for f in fractions {
                let next: SIMD3<Float> = at + run * f
                s.append(.cone(.leg, at, next, tarsusR * 1.05, tarsusR * 0.85, light: true, index: index))
                at = next
            }
            legs.append(Leg(name: "\(side > 0 ? "right" : "left") \(spec.name)", side: side, root: root, shapes: s))
            index += 1
        }
    }
    return legs
}

// MARK: - the sugar

// Sucrose crystallises monoclinic, space group P2₁, a = 10.8631 Å,
// b = 8.7044 Å, c = 7.7624 Å, β = 102.938° (Wikipedia "Sucrose", from the
// neutron structure of Brown & Levy, *Acta Cryst B* 1973). Its normal habit is
// "simple or normal of stout-prismatic form", bounded by a(100), c(001),
// d(101), r(1̄01) and p(110) faces, with the cleavage parallel to a
// (VanHook, "Habit modification of sucrose crystals", *J Sugar Beet Res*
// 22(1), 1983, which follows Vavrinecz's *Atlas of Sugar Crystals*).
//
// So every face normal here is COMPUTED from that cell: the normal of the
// plane (hkl) is h·a* + k·b* + l·c*, the reciprocal lattice vector. The
// crystal's tilts and the 102.9° corner come out of the lattice, not an eye.
let cellA: Float = 10.8631
let cellB: Float = 8.7044
let cellC: Float = 7.7624
let cellBeta: Float = 102.938 * Float.pi / 180

func reciprocalNormal(_ h: Float, _ k: Float, _ l: Float) -> SIMD3<Float> {
    // Direct basis: a along x, b along y, c in the xz plane at β from a.
    let a = SIMD3<Float>(cellA, 0, 0)
    let b = SIMD3<Float>(0, cellB, 0)
    let c = SIMD3<Float>(cellC * cos(cellBeta), 0, cellC * sin(cellBeta))
    let v: Float = simd_dot(a, simd_cross(b, c))
    let aS: SIMD3<Float> = simd_cross(b, c) / v
    let bS: SIMD3<Float> = simd_cross(c, a) / v
    let cS: SIMD3<Float> = simd_cross(a, b) / v
    let n: SIMD3<Float> = aS * h + bS * k + cS * l
    return simd_normalize(n)
}

/// One face form and how far its planes sit from the crystal's centre, in mm
/// for a grain of scale 1. MODEL distances, chosen for the stout prism VanHook
/// calls normal, with the a-faces closest (the tabular direction).
struct FaceForm {
    var hkl: SIMD3<Float>
    var distance: Float
}

let sugarForms: [FaceForm] = [
    FaceForm(hkl: SIMD3(1, 0, 0), distance: 0.20), FaceForm(hkl: SIMD3(-1, 0, 0), distance: 0.20),
    FaceForm(hkl: SIMD3(0, 0, 1), distance: 0.24), FaceForm(hkl: SIMD3(0, 0, -1), distance: 0.24),
    FaceForm(hkl: SIMD3(1, 1, 0), distance: 0.30), FaceForm(hkl: SIMD3(1, -1, 0), distance: 0.30),
    FaceForm(hkl: SIMD3(-1, 1, 0), distance: 0.30), FaceForm(hkl: SIMD3(-1, -1, 0), distance: 0.30),
    FaceForm(hkl: SIMD3(1, 0, 1), distance: 0.27), FaceForm(hkl: SIMD3(-1, 0, -1), distance: 0.27),
    FaceForm(hkl: SIMD3(-1, 0, 1), distance: 0.29), FaceForm(hkl: SIMD3(1, 0, -1), distance: 0.29),
    FaceForm(hkl: SIMD3(0, 1, 1), distance: 0.36), FaceForm(hkl: SIMD3(0, -1, -1), distance: 0.36),
    FaceForm(hkl: SIMD3(0, 1, -1), distance: 0.36), FaceForm(hkl: SIMD3(0, -1, 1), distance: 0.36),
]

// Granulated sugar: "average crystal size ranging from 0.3 to 0.55 mm", and
// mean apertures (the sieve size that passes half the sample) "of up to
// 670 µm and as low as 475 µm" between refineries (ScienceDirect Topics,
// "Sugar Crystals", quoting the sugar-technology literature; one refinery
// specification targets 0.60 ± 0.05 mm). A grain passes a sieve by its
// intermediate dimension, so that is what the tests compare.
let grainSieveRange: ClosedRange<Float> = 0.30...0.67

/// Edge rounding, mm. Real grains are chipped and worn at the edges by
/// handling. MODEL.
let grainRounding: Float = 0.012

/// One grain in the world: its faces as world-space planes.
struct Grain {
    var centre: SIMD3<Float>
    var normals: [SIMD3<Float>]
    var distances: [Float]

    /// The largest signed plane distance — never more than the true distance
    /// outside a convex solid, exact inside — with the edges worn by a smooth
    /// maximum of radius `grainRounding`. The same arithmetic as the kernel's.
    func sdf(_ p: SIMD3<Float>) -> Float {
        let q: SIMD3<Float> = p - centre
        var d: Float = simd_dot(q, normals[0]) - distances[0]
        for i in 1..<normals.count {
            d = smoothMax(d, simd_dot(q, normals[i]) - distances[i], grainRounding)
        }
        return d
    }

    /// Corners of the polyhedron, by intersecting every three planes.
    func vertices() -> [SIMD3<Float>] {
        var out: [SIMD3<Float>] = []
        let n: Int = normals.count
        for i in 0..<n {
            for j in (i + 1)..<n {
                for k in (j + 1)..<n {
                    let m = simd_float3x3(rows: [normals[i], normals[j], normals[k]])
                    if abs(m.determinant) < 1e-4 { continue }
                    let rhs = SIMD3<Float>(distances[i], distances[j], distances[k])
                    let q: SIMD3<Float> = m.inverse * rhs
                    var inside = true
                    for f in 0..<n where simd_dot(q, normals[f]) > distances[f] + 1e-4 { inside = false; break }
                    if inside { out.append(q + centre) }
                }
            }
        }
        return out
    }

    /// The three extents along the crystal's own a*, b, c directions — the
    /// sieve compares the middle one.
    func extents(frame: [SIMD3<Float>]) -> [Float] {
        let v: [SIMD3<Float>] = vertices()
        return frame.map { d in
            let proj: [Float] = v.map { simd_dot($0, d) }
            return (proj.max() ?? 0) - (proj.min() ?? 0)
        }
    }
}

/// How a grain lies: which face it rests on, its turn about the vertical,
/// its size relative to the forms above, and where on the table it sits.
struct GrainPlacement {
    var restFace: SIMD3<Float>   // (hkl) of the face on the table
    var yaw: Float               // radians
    var scale: Float
    var at: SIMD2<Float>         // (x, z) on the table
}

/// Polynomial smooth maximum, as the kernel's smax.
func smoothMax(_ a: Float, _ b: Float, _ k: Float) -> Float {
    let h: Float = min(max(0.5 + 0.5 * (a - b) / k, 0), 1)
    return b + (a - b) * h + k * h * (1 - h)
}

/// A rotation taking unit vector `from` onto unit vector `to`.
func rotation(from: SIMD3<Float>, to: SIMD3<Float>) -> simd_quatf {
    simd_quatf(from: simd_normalize(from), to: simd_normalize(to))
}

/// The world grain: rotate so the rest face points straight down, turn it,
/// and lift it until that face lies exactly on the table.
func placeGrain(_ g: GrainPlacement) -> (grain: Grain, frame: [SIMD3<Float>]) {
    let restN: SIMD3<Float> = reciprocalNormal(g.restFace.x, g.restFace.y, g.restFace.z)
    let q1: simd_quatf = rotation(from: restN, to: SIMD3<Float>(0, -1, 0))
    let q2 = simd_quatf(angle: g.yaw, axis: SIMD3<Float>(0, 1, 0))
    let q: simd_quatf = q2 * q1
    var normals: [SIMD3<Float>] = []
    var dists: [Float] = []
    var restDistance: Float = 0
    for f in sugarForms {
        let n: SIMD3<Float> = q.act(reciprocalNormal(f.hkl.x, f.hkl.y, f.hkl.z))
        normals.append(n)
        dists.append(f.distance * g.scale)
        if simd_length(f.hkl - g.restFace) < 1e-3 { restDistance = f.distance * g.scale }
    }
    let centre = SIMD3<Float>(g.at.x, restDistance, g.at.y)
    // The crystal frame: a* (normal to the cleavage), b, and a* × b.
    let aStar: SIMD3<Float> = q.act(reciprocalNormal(1, 0, 0))
    let bAxis: SIMD3<Float> = q.act(SIMD3<Float>(0, 1, 0))
    let third: SIMD3<Float> = simd_normalize(simd_cross(aStar, bAxis))
    return (Grain(centre: centre, normals: normals, distances: dists), [aStar, bAxis, third])
}

/// The grains on the table. The first is the one being tasted: it lies on
/// its cleavage face (100), so its top is the other a-face, flat and level.
let grainPlacements: [GrainPlacement] = [
    GrainPlacement(restFace: SIMD3(-1, 0, 0), yaw: 0.35, scale: 1.00, at: SIMD2(2.62, 1.05)),
    GrainPlacement(restFace: SIMD3(0, 0, -1), yaw: 1.3, scale: 0.92, at: SIMD2(3.45, 0.55)),
    GrainPlacement(restFace: SIMD3(-1, 0, 0), yaw: 2.3, scale: 1.08, at: SIMD2(3.20, 1.70)),
    GrainPlacement(restFace: SIMD3(1, 0, 0), yaw: -0.6, scale: 0.88, at: SIMD2(2.35, 2.05)),
    GrainPlacement(restFace: SIMD3(0, 0, 1), yaw: 0.9, scale: 0.95, at: SIMD2(3.95, 1.35)),
]

// MARK: - the antennae, and the touch

/// A built antenna: its segments and where its tip ends up.
struct Antenna {
    var side: Float
    var socket: SIMD3<Float>
    var elbow: SIMD3<Float>
    var segments: [Shape]        // scape first, then the funiculus
    var tipCentre: SIMD3<Float>  // centre of the apical segment's end cap
    var tipRadius: Float
}

/// Points along a circular arc of length `length` from `p0` to `p1`, bowing
/// towards `bow`. The funiculus is a gentle curve, not a straight rod.
func arcPoints(from p0: SIMD3<Float>, to p1: SIMD3<Float>, length: Float, bow: SIMD3<Float>,
               at fractions: [Float]) -> [SIMD3<Float>] {
    let chordV: SIMD3<Float> = p1 - p0
    let chord: Float = simd_length(chordV)
    let u: SIMD3<Float> = chordV / chord
    let w: SIMD3<Float> = simd_normalize(bow - u * simd_dot(bow, u))
    if chord >= length * 0.9999 { return fractions.map { p0 + chordV * $0 } }
    // Solve chord = 2R sin(θ/2), length = Rθ for θ by bisection.
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

/// Where the right antenna's tip touches the tasted grain: a point on its top
/// face, chosen towards the ant. The tip's end cap rests on the face there.
func contactPoint(_ grains: [Grain]) -> (point: SIMD3<Float>, normal: SIMD3<Float>) {
    let g: Grain = grains[0]
    // The top face is the one whose normal points most nearly up.
    var best: Int = 0
    for i in 0..<g.normals.count where g.normals[i].y > g.normals[best].y { best = i }
    let n: SIMD3<Float> = g.normals[best]
    let top: SIMD3<Float> = g.centre + n * g.distances[best]
    return (top + SIMD3<Float>(-0.10, 0, -0.06), n)
}

func buildAntennae(grains: [Grain], mutant: Mutant) -> [Antenna] {
    var out: [Antenna] = []
    let side = SIMD3<Float>(0, 0, 1)
    let funiculusCount: Int = mutant == .segments13 ? workerAntennaSegments : workerAntennaSegments - 1
    let lengths: [Float] = funiculusLengths(count: funiculusCount)
    let total: Float = lengths.reduce(0, +)
    var fractions: [Float] = [0]
    var run: Float = 0
    for l in lengths { run += l; fractions.append(run / total) }
    for (index, sgn) in [Float(1), Float(-1)].enumerated() {
        // The torulus, where the antenna rises from the head: on the frons
        // behind the clypeus, one either side of the midline.
        let socket: SIMD3<Float> = headCentre + headForward * 0.26 + headUp * 0.13 + side * (sgn * 0.15)
        // The scape swings forward and out, raised. MODEL pose.
        let scapeDir: SIMD3<Float> = sgn > 0 ? simd_normalize(SIMD3<Float>(0.45, 0.55, 0.70))
                                             : simd_normalize(SIMD3<Float>(0.35, 0.62, -0.70))
        let elbow: SIMD3<Float> = socket + scapeDir * scapeLength
        var segs: [Shape] = [.cone(.antenna, socket, elbow, scapeRadius * 0.8, scapeRadius, light: true, index: index)]
        let tipR: Float = funiculusTipRadius
        let tipCentre: SIMD3<Float>
        if sgn > 0 {
            let (p, n) = contactPoint(grains)
            let lift: Float = mutant == .hover ? 0.1 : 0
            tipCentre = p + n * (tipR + lift)
        } else {
            tipCentre = SIMD3<Float>(2.30, 0.35, -0.60)
        }
        let bow = SIMD3<Float>(1, 0.6, 0)
        let pts: [SIMD3<Float>] = arcPoints(from: elbow, to: tipCentre, length: total, bow: bow, at: fractions)
        for i in 0..<funiculusCount {
            let f0: Float = fractions[i]
            let f1: Float = fractions[i + 1]
            let r0: Float = funiculusBaseRadius + (funiculusTipRadius - funiculusBaseRadius) * f0
            let r1: Float = funiculusBaseRadius + (funiculusTipRadius - funiculusBaseRadius) * f1
            // Each segment starts narrow and swells towards its far end, so
            // the joints read as joints — the apical one ends round.
            let last: Bool = i == funiculusCount - 1
            segs.append(.cone(.antenna, pts[i], pts[i + 1], r0 * 0.78, last ? tipR : r1, index: index))
        }
        out.append(Antenna(side: sgn, socket: socket, elbow: elbow, segments: segs,
                           tipCentre: tipCentre, tipRadius: tipR))
    }
    return out
}

/// Everything the main view draws of the ant.
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

func buildGrains() -> [(grain: Grain, frame: [SIMD3<Float>])] {
    grainPlacements.map { placeGrain($0) }
}

func buildAnt(mutant: Mutant) -> AntModel {
    let grains: [Grain] = buildGrains().map { $0.grain }
    return AntModel(body: bodyShapes(), legs: buildLegs(), antennae: buildAntennae(grains: grains, mutant: mutant))
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

/// The film of water on the crystal where the tip touches, µm. MODEL: a
/// contact sensillum wets what it touches — the terminal pore is filled with
/// sensillum lymph and a viscous pore fluid — and sugar dissolves into that
/// film (sucrose: 2.01 g per mL of water at 20 °C, Wikipedia "Sucrose").
/// Thickness and spread are chosen to be visible, not measured.
let filmThickness: Float = 0.12
let filmRadius: Float = 2.8
let meniscusHeight: Float = 0.45

/// The two hairs in the inset. The taste hair comes down towards the viewer
/// and rests its apex in the film; the smell hair stands beside it in the air.
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

// MARK: - sucrose, from the crystal

struct Atom {
    var element: String      // "C", "O", "H"
    var position: SIMD3<Float>   // Å
    var fromCrystal: Bool
}

struct Molecule {
    var atoms: [Atom]
    var bonds: [(Int, Int)]

    var formula: [String: Int] {
        var f: [String: Int] = [:]
        for a in atoms { f[a.element, default: 0] += 1 }
        return f
    }
}

enum LoadError: Error { case missing(String) }

/// Sucrose as it sits in PDB 6S1T (β-fructofuranosidase from Schwanniomyces
/// occidentalis with sucrose, 2.09 Å; chain F = GLC 1 + FRU 2 joined C1–O2),
/// copied verbatim into Resources/. X-ray structures at this resolution have
/// no hydrogens, so the 22 are added here in idealised geometry — C–H
/// 1.09 Å, O–H 0.96 Å, tetrahedral angles; the hydroxyl rotations are MODEL.
func loadSucrose(path: String) throws -> Molecule {
    guard let text = try? String(contentsOfFile: path, encoding: .utf8) else { throw LoadError.missing(path) }
    var heavy: [Atom] = []
    for line in text.split(separator: "\n") where line.hasPrefix("HETATM") {
        let s = Array(line)
        func field(_ a: Int, _ b: Int) -> String { String(s[a..<min(b, s.count)]).trimmingCharacters(in: .whitespaces) }
        let x: Float = Float(field(30, 38)) ?? 0
        let y: Float = Float(field(38, 46)) ?? 0
        let z: Float = Float(field(46, 54)) ?? 0
        heavy.append(Atom(element: field(76, 78), position: SIMD3(x, y, z), fromCrystal: true))
    }
    var bonds: [(Int, Int)] = []
    for i in 0..<heavy.count {
        for j in (i + 1)..<heavy.count where simd_distance(heavy[i].position, heavy[j].position) < 1.75 {
            bonds.append((i, j))
        }
    }
    var atoms: [Atom] = heavy
    func neighbours(_ i: Int) -> [Int] {
        bonds.compactMap { $0.0 == i ? $0.1 : ($0.1 == i ? $0.0 : nil) }
    }
    let tet: Float = 109.47 * Float.pi / 180
    for i in 0..<heavy.count {
        let nb: [Int] = neighbours(i)
        let p: SIMD3<Float> = heavy[i].position
        let us: [SIMD3<Float>] = nb.map { simd_normalize(heavy[$0].position - p) }
        var dirs: [SIMD3<Float>] = []
        if heavy[i].element == "C" {
            if nb.count == 3 {
                dirs = [simd_normalize(-(us[0] + us[1] + us[2]))]
            } else if nb.count == 2 {
                let bis: SIMD3<Float> = simd_normalize(-(us[0] + us[1]))
                let perp: SIMD3<Float> = simd_normalize(simd_cross(us[0], us[1]))
                let half: Float = tet / 2
                dirs = [bis * cos(half) + perp * sin(half), bis * cos(half) - perp * sin(half)]
            }
            for d in dirs {
                atoms.append(Atom(element: "H", position: p + d * 1.09, fromCrystal: false))
                bonds.append((i, atoms.count - 1))
            }
        } else if heavy[i].element == "O" && nb.count == 1 {
            // Hydroxyl: C–O–H at the tetrahedral angle, anti to one of the
            // carbon's other neighbours.
            let c: Int = nb[0]
            let u: SIMD3<Float> = simd_normalize(p - heavy[c].position)
            let others: [Int] = neighbours(c).filter { $0 != i }
            let w: SIMD3<Float> = simd_normalize(heavy[others[0]].position - heavy[c].position)
            let v: SIMD3<Float> = simd_normalize(-(w - u * simd_dot(w, u)))
            let d: SIMD3<Float> = u * (-cos(tet)) + v * sin(tet)
            atoms.append(Atom(element: "H", position: p + d * 0.96, fromCrystal: false))
            bonds.append((i, atoms.count - 1))
        }
    }
    // Centre on the heavy atoms' centroid.
    var c = SIMD3<Float>(0, 0, 0)
    for a in heavy { c += a.position }
    c /= Float(heavy.count)
    for i in 0..<atoms.count { atoms[i].position -= c }
    return Molecule(atoms: atoms, bonds: bonds)
}

// MARK: - optics, from refractive indices

/// Normal-incidence Fresnel reflectance between two media.
func fresnelF0(_ n1: Double, _ n2: Double) -> Float {
    let r: Double = (n1 - n2) / (n1 + n2)
    return Float(r * r)
}

/// Insect cuticle, n ≈ 1.56: Leertouwer, Wilts & Stavenga, *Opt Express* 19:
/// 24061 (2011), measured butterfly-scale chitin at 1.56 in the visible and
/// that value is the one used across insect optics. No measurement of ant
/// cuticle itself was found; the ant is taken to be chitin like the rest.
let cuticleIndex: Double = 1.56
/// Sucrose crystal: biaxial, nα 1.540, nβ 1.567, nγ 1.572 (McCrone Particle
/// Atlas, "Sucrose"). The birefringence (0.032) is ignored and the mean used.
let sucroseIndex: Double = (1.540 + 1.567 + 1.572) / 3
/// Water, for the film.
let waterIndex: Double = 1.333

let cuticleF0: Float = fresnelF0(1.0, cuticleIndex)
let sucroseF0: Float = fresnelF0(1.0, sucroseIndex)
let waterF0: Float = fresnelF0(1.0, waterIndex)

// MARK: - why it is taste, not smell

// Sucrose has no boiling point — it decomposes (NOAA CAMEO Chemicals / USCG
// CHRIS sheet "Sucrose": "Boiling point at 1 atm: Not pertinent (decomposes)";
// melting 160–186 °C with decomposition) — and its vapour pressure is
// "0 mmHg (approx)" (NIOSH Pocket Guide, "Sucrose"). With no sucrose in the
// air there is nothing for a multiporous smell hair to catch; an ant finds
// sugar by touching it, with uniporous taste hairs. That is the picture.
