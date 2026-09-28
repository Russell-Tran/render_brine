// One cotton fibre, and the yarn it is spun into.
//
// A cotton fibre is a single cell: a seed hair of the cotton plant, which
// grows, lays down a thick wall of nearly pure cellulose, dies, and dries.
// Drying collapses the hollow tube into a flat ribbon with a kidney-shaped
// cross-section and a slit of a lumen, and the ribbon twists (the
// "convolutions"). Along its length the helix of cellulose in the wall
// changes hand at "reversals".
//
// Lengths here are in MICROMETRES unless a name says otherwise.

import Foundation
import simd

enum Mutant: Int {
    case none = 0
    case roundFibre = 1   // the fibre drawn round and untwisted, as a synthetic filament would be
    case alphaLinks = 2   // the glucose units joined α(1→4), starch's link, instead of β(1→4)
    case noReversal = 3   // every fibre twisted one way along its whole length
    case gappyKnit = 4    // the courses pulled apart, so a row's loops no longer hold the row below

    static var fromEnvironment: Mutant {
        switch ProcessInfo.processInfo.environment["COTTON_MUTANT"] {
        case "roundFibre": return .roundFibre
        case "alphaLinks": return .alphaLinks
        case "noReversal": return .noReversal
        case "gappyKnit": return .gappyKnit
        default: return .none
        }
    }
}

// MARK: - the cross-section, measured

// LaFave, Etukuri, Courtney, Kothari, Rife & Saski 2023, "A simplified
// microscopy technique to rapidly characterize individual fiber traits in
// cotton", Methods Protoc 6: 92 (doi 10.3390/mps6050092; Europe PMC full
// text, PMC10609321), Table 3b: mean cross-sections of five mature upland
// lines, from 6 µm resin sections. Their five values of each, averaged here:
//   outer perimeter   55.45 64.22 61.33 62.02 52.23 µm → 59.05
//   wall ("true") area 115.49 159.11 160.73 136.73 104.92 µm² → 135.40
//   lumen area        13.59 14.83 12.02 15.04 11.80 µm² → 13.456
//   lumen perimeter   32.62 33.91 26.94 36.12 28.02 µm → 31.522
// Their lumen circularity is 0.17–0.24: a slit, not a round hole. They
// describe the fibres as "mostly kidney-shaped" with lumens "not tightly
// closed". Density of the wall 1.52 g/cm³, the value they use.
let measuredPerimeter: Float = (55.45 + 64.22 + 61.33 + 62.02 + 52.23) / 5
let measuredWallArea: Float = (115.49 + 159.11 + 160.73 + 136.73 + 104.92) / 5
let measuredLumenArea: Float = (13.59 + 14.83 + 12.02 + 15.04 + 11.80) / 5
let measuredLumenPerimeter: Float = (32.62 + 33.91 + 26.94 + 36.12 + 28.02) / 5
let wallDensity: Float = 1.52           // g/cm³

/// The shape drawn is a BENT CAPSULE: a band of constant thickness 2B whose
/// centre line is an arc of length L, with round ends — a kidney bean. For
/// that shape perimeter = 2L + 2πB and area = 2BL + πB², so the two measured
/// numbers fix B and L exactly (the smaller root of πB² − P·B + A = 0). The
/// lumen is a second, thinner bent capsule on the same arc. Only the bend is
/// not measured: MODEL, a 16 µm radius, enough to read as a kidney; LaFave
/// et al. give no curvature.
struct CrossSection {
    var bendRadius: Float      // Rb: the arc's radius; its centre is on the concave side
    var halfArc: Float         // Φ: half the angle the outer centre arc spans
    var halfThickness: Float   // B
    var lumenHalfArc: Float    // Φ of the lumen's arc
    var lumenHalfThickness: Float

    static func bentCapsule(perimeter p: Float, area a: Float) -> (b: Float, l: Float) {
        let disc: Float = p * p - 4 * Float.pi * a
        let b: Float = (p - disc.squareRoot()) / (2 * Float.pi)
        return (b, p / 2 - Float.pi * b)
    }

    static let modelBendRadius: Float = 16.0

    static func cotton() -> CrossSection {
        let outer = bentCapsule(perimeter: measuredPerimeter, area: measuredWallArea + measuredLumenArea)
        let lumen = bentCapsule(perimeter: measuredLumenPerimeter, area: measuredLumenArea)
        let rb: Float = modelBendRadius
        return CrossSection(bendRadius: rb, halfArc: outer.l / (2 * rb), halfThickness: outer.b,
                            lumenHalfArc: lumen.l / (2 * rb), lumenHalfThickness: lumen.b)
    }

    /// The `roundFibre` mutant: a round fibre of the same perimeter with a
    /// round lumen of the same area — a filament, not a collapsed cell.
    static func round() -> CrossSection {
        CrossSection(bendRadius: modelBendRadius, halfArc: 0, halfThickness: measuredPerimeter / (2 * Float.pi),
                     lumenHalfArc: 0, lumenHalfThickness: (measuredLumenArea / Float.pi).squareRoot())
    }

    /// The farthest the section reaches from the fibre's axis, which runs
    /// through the middle of the arc: a bound for culling and for tests.
    var reach: Float {
        let end = SIMD2<Float>(bendRadius * sin(halfArc), bendRadius * (1 - cos(halfArc)))
        return simd_length(end) + halfThickness
    }
}

/// Exact distance to a bent capsule: an arc of radius `ra` spanning ±Φ about
/// +y, thickened by `rb` (Quílez's sdArc). `p` is relative to the arc's
/// circle centre, with x already folded to ≥ 0.
func arcDistance(_ p: SIMD2<Float>, halfArc phi: Float, ra: Float, rb: Float) -> Float {
    let sc = SIMD2<Float>(sin(phi), cos(phi))
    if sc.y * p.x > sc.x * p.y { return simd_length(p - sc * ra) - rb }
    return abs(simd_length(p) - ra) - rb
}

/// Signed distance to the cell wall in its own cross-section plane: u along
/// the ribbon's width, v across its thickness, the bend's centre at v = +Rb.
/// Inside the lumen counts as outside the wall.
func wallDistance(u: Float, v: Float, _ s: CrossSection) -> Float {
    let q = SIMD2<Float>(abs(u), s.bendRadius - v)
    let outer: Float = arcDistance(q, halfArc: s.halfArc, ra: s.bendRadius, rb: s.halfThickness)
    let lumen: Float = arcDistance(q, halfArc: s.lumenHalfArc, ra: s.bendRadius, rb: s.lumenHalfThickness)
    return max(outer, -lumen)
}

// MARK: - convolutions and reversals

// Convolutions: "about 60 twists or convolutions per centimeter" — Jaekel,
// Torres, Antonietti, Rojas & Filonenko 2024, Sci Rep 14: 18406 (PMC11310312),
// citing Dochia et al. 2012 (Handbook of Natural Fibres, not seen); the same
// figure in Kurpińska et al. 2022, Sci Rep 12: 20565 (PMC9709079). MODEL
// reading: one convolution = one HALF turn of the ribbon (each place it
// turns edge-on), so 3 full turns per mm. UNVERIFIED which reading the
// primary source intends; the other would double the twist.
let convolutionsPerCentimetre: Float = 60
let twistRate: Float = convolutionsPerCentimetre / 10_000 * Float.pi     // rad per µm, 0.0188

// Reversals: Gould & Seagull 2002, J Cotton Sci 6: 52–59, Table 1 (read in
// the journal's PDF): mature (50 days after flowering) untreated MD51 fibres
// have 17.9 reversals per cm, counted in polarized light — a mean spacing of
// 0.56 mm. What reverses there is the helix of cellulose microfibrils in the
// wall ("microfibrils routinely reverse their gyre"), and they note, citing
// Hsieh 1999, that reversals "may induce changes in the convolution of
// mature dried fibers". Zang et al. 2021 (iScience 24: 102930, PMC8361218)
// see both right- and left-handed helices in mature fibres.
// MODEL, UNVERIFIED: that the ribbon's own twist changes hand exactly at a
// reversal — the sources above support the wall helix reversing, and both
// hands of twist occurring, but none read here says the convolutions flip
// at the same place. Drawn so, and captioned for the wall helix only.
let reversalsPerCentimetre: Float = 17.938
let meanReversalSpacing: Float = 10_000 / reversalsPerCentimetre          // 557 µm
/// How gradually the twist turns over at a reversal. Gould & Seagull's
/// mature-fibre reversals are nearly all the gradual "S" kind; the width of
/// the change is MODEL.
let reversalHalfWidth: Float = 18                                        // µm

/// Twist rate at arc length s, signed: + one hand, − the other. Each
/// reversal flips the sign through a tanh ramp.
func twistAt(_ s: Float, reversals: [Float], hand: Float, rate: Float) -> Float {
    var sigma: Float = hand
    for r in reversals { sigma *= tanh((s - r) / reversalHalfWidth) }
    return rate * sigma
}

// MARK: - fibre length (for the caption)

// "growing to a length of up to 6cm" — Jareczek, Grover & Wendel 2023,
// Front Plant Sci 14: 1146802 (PMC10017751), citing Kim & Triplett 2001.
// The same review: the mature cell "collapses into a bean shape in cross
// section", and domesticated fibre is "nearly pure (~98%) cellulose"
// (citing Haigler et al.). The brief's lead of 90–95% is lower than this.
let maxFibreLengthCentimetres: Float = 6
let cellulosePercent: Float = 98

// MARK: - a ribbon: any fibre, drawn

/// A fibre as the GPU draws it: a centre-line polyline and, at each vertex,
/// the direction of the ribbon's width (already turned by its twist) and the
/// mitre plane shared with the next segment. Units are the scene's: µm in
/// the insets, mm in the main view (`scale` converts the µm cross-section).
struct Ribbon {
    var points: [SIMD3<Float>]
    var widths: [SIMD3<Float>]        // unit, ⟂ the local tangent
    var twist: [Float]                // accumulated twist angle at each vertex, rad (for tests)
    var arc: [Float]                  // arc length at each vertex, in scene units
    var reversals: [Float]            // arc lengths of reversals, scene units
    var scale: Float                  // scene units per µm
    var highlight: Bool = false       // the fibre the inset is about
    var cutEnd: Bool = true           // flat end at the last vertex (else a flat end too, inside the yarn)

    /// Mitre normals: the bisector of the segments meeting at each vertex.
    var mitres: [SIMD3<Float>] {
        var out: [SIMD3<Float>] = []
        let n: Int = points.count
        for k in 0..<n {
            let tin: SIMD3<Float> = k > 0 ? simd_normalize(points[k] - points[k - 1]) : simd_normalize(points[1] - points[0])
            let tout: SIMD3<Float> = k < n - 1 ? simd_normalize(points[k + 1] - points[k]) : tin
            out.append(simd_normalize(tin + tout))
        }
        return out
    }

    var length: Float { arc.last ?? 0 }
}

/// A vector ⟂ to t.
func perpendicular(to t: SIMD3<Float>) -> SIMD3<Float> {
    let helper: SIMD3<Float> = abs(t.y) < 0.9 ? SIMD3<Float>(0, 1, 0) : SIMD3<Float>(1, 0, 0)
    return simd_normalize(simd_cross(t, helper))
}

/// Builds a ribbon along a centre line: the width direction is carried
/// along without spin (parallel transport, the rotation-minimising frame)
/// and then turned about the tangent by the accumulated twist.
func makeRibbon(_ centre: [SIMD3<Float>], startWidth: SIMD3<Float>, startTwist: Float,
                reversals: [Float], hand: Float, rate: Float, scale: Float) -> Ribbon {
    var widths: [SIMD3<Float>] = []
    var twists: [Float] = []
    var arcs: [Float] = [0]
    for k in 1..<centre.count { arcs.append(arcs[k - 1] + simd_distance(centre[k], centre[k - 1])) }
    var t0: SIMD3<Float> = simd_normalize(centre[1] - centre[0])
    var carried: SIMD3<Float> = simd_normalize(startWidth - t0 * simd_dot(startWidth, t0))
    var beta: Float = startTwist
    for k in 0..<centre.count {
        let tk: SIMD3<Float> = k < centre.count - 1 ? simd_normalize(centre[k + 1] - centre[k]) : t0
        if k > 0 {
            // Parallel transport: rotate `carried` by the turn from t0 to tk.
            let axis: SIMD3<Float> = simd_cross(t0, tk)
            let s: Float = simd_length(axis)
            if s > 1e-7 {
                let angle: Float = atan2(s, simd_dot(t0, tk))
                carried = simd_quatf(angle: angle, axis: axis / s).act(carried)
            }
            carried = simd_normalize(carried - tk * simd_dot(carried, tk))
            // Twist: integrate the signed rate over the step (midpoint).
            let mid: Float = (arcs[k] + arcs[k - 1]) / 2
            beta += twistAt(mid, reversals: reversals, hand: hand, rate: rate) * (arcs[k] - arcs[k - 1])
        }
        t0 = tk
        widths.append(simd_quatf(angle: beta, axis: tk).act(carried))
        twists.append(beta)
    }
    return Ribbon(points: centre, widths: widths, twist: twists, arc: arcs, reversals: reversals, scale: scale)
}

/// A small deterministic random source, so every run draws the same fibres.
struct Seeded: RandomNumberGenerator {
    var state: UInt64
    init(_ seed: UInt64) { state = seed &* 0x9E3779B97F4A7C15 | 1 }
    mutating func next() -> UInt64 {
        state ^= state << 13
        state ^= state >> 7
        state ^= state << 17
        return state
    }
    mutating func uniform(_ lo: Float, _ hi: Float) -> Float {
        lo + (hi - lo) * Float(next() >> 40) / Float(1 << 24)
    }
}

/// Reversal positions along a fibre of length `length` (scene units): a
/// Poisson process with Gould & Seagull's mean spacing. The spacing law is
/// MODEL; Raes et al. 1968 (Text Res J 38: 182, abstract via Crossref) found
/// the distribution has modes every 200–250 µm, not a pure exponential.
func randomReversals(length: Float, spacing: Float, rng: inout Seeded) -> [Float] {
    var out: [Float] = []
    var s: Float = -spacing * rng.uniform(0, 1)
    while true {
        s += -spacing * log(max(rng.uniform(0, 1), 1e-6))
        if s >= length { break }
        if s > 0 { out.append(s) }
    }
    return out
}
