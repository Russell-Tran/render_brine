// The yarn: cotton fibres twisted together. And the piece of it the big
// inset looks at — a stretch of one loop's leg, its surface fibres, and one
// fibre that has worked loose and stands up as fuzz.

import Foundation
import simd

// MARK: - the yarn, measured

// Hossain, Hossain, Al Mamun, Fahim, Rafa, Pappu, Darda, Arafat & Islam
// 2025, "Influence of sinker timing on loop shape, width and areal density
// of weft-knitted cotton plain jersey fabric", PLoS One 20: e0323572
// (PMC12080846), Table 1: "100% cotton combed, single ply", nominal count
// 30/1 Ne = 19.68 tex (actual 19.78), 19.6 twists per inch actual (20
// nominal), twist direction Z. Their fabric, not a sock: MODEL that a
// cotton sock's foot is knitted like it (Tomljenović et al. 2023 say the
// foot of a sock is plain single jersey).
let yarnTex: Float = 19.68
let yarnTwistsPerInch: Float = 19.6
let yarnTwistPerMillimetre: Float = yarnTwistsPerInch / 25.4       // 0.772 turns/mm
/// Z twist is a right-handed helix: +1.
let yarnHand: Float = 1

/// One fibre's linear density, from the measured wall area and density
/// (LaFave et al. 2023): 135.4 µm² × 1.52 g/cm³ = 0.206 tex.
let wallAreaSquareMetres: Float = measuredWallArea * 1e-12
let wallDensityKgPerCubicMetre: Float = wallDensity * 1000
let fibreKgPerMetre: Float = wallAreaSquareMetres * wallDensityKgPerCubicMetre
let fibreTex: Float = fibreKgPerMetre * 1e6                                  // kg/m → g/km
/// Fibres in a cross-section of the yarn: 19.68 / 0.206 ≈ 96.
let fibresPerYarnSection: Float = yarnTex / fibreTex

/// How much of the yarn's cross-section is fibre (lumens counted as fibre).
/// MODEL, 0.6: no measured packing fraction for this yarn was found.
let yarnPacking: Float = 0.6
/// The yarn's radius, from ~96 fibres of 148.9 µm² outer area at that
/// packing: 86.9 µm.
let yarnRadius: Float = (fibresPerYarnSection * (measuredWallArea + measuredLumenArea) / (yarnPacking * Float.pi)).squareRoot()

/// A surface fibre's centre line sits one flat fibre's half-thickness under
/// the yarn's surface.
func surfaceRadius(_ s: CrossSection) -> Float { yarnRadius - s.halfThickness }

/// The surface helix angle: tan α = 2π R T. About 22°.
func helixAngle(radius: Float) -> Float { atan(2 * Float.pi * radius * yarnTwistPerMillimetre / 1000) }

/// Fibres in the drawn surface layer: as many as fit round it without
/// touching when they lie flat, measured square to their own direction.
func surfaceFibreCount(_ s: CrossSection) -> Int {
    let r: Float = surfaceRadius(s)
    let across: Float = 2 * Float.pi * r * cos(helixAngle(radius: r))
    return Int(floor(across / (2 * s.reach + 1.0)))
}

/// The hidden fibres beneath the surface layer are drawn as one solid core,
/// kept clear of a surface fibre turned edge-on. MODEL simplification.
func coreRadius(_ s: CrossSection) -> Float { surfaceRadius(s) - s.reach - 0.5 }

// MARK: - the inset's piece of yarn

/// Where the yarn lies in the inset's world (µm): its axis through
/// `yarnOrigin` along `yarnAxis`. Straight across the inset — a loop's leg
/// is its straightest part, and at 520 µm the loop's curve (radius ~0.3 mm)
/// is left out. MODEL framing.
let yarnOrigin = SIMD3<Float>(0, -150, 0)
let yarnAxis: SIMD3<Float> = simd_normalize(SIMD3<Float>(1, 0.12, 0))
/// Half the drawn length of the yarn piece along its axis.
let yarnHalfLength: Float = 430

/// The inset's contents.
struct YarnPiece {
    var section: CrossSection
    var surface: [Ribbon]           // the surface layer, fibre 0 excluded
    var fuzz: Ribbon                // fibre 0: follows the surface, then leaves it, then ends in a cut
    var peelArc: Float              // arc length along `fuzz` where it leaves the yarn
    var core: Float                 // core radius
    var fibreCount: Int             // surface fibres including the fuzz fibre
}

/// A right-handed (Z) helix round the yarn axis: the point at axial
/// distance x, radius r, phase φ.
func yarnFrame() -> (e1: SIMD3<Float>, e2: SIMD3<Float>) {
    let e1: SIMD3<Float> = perpendicular(to: yarnAxis)
    let e2: SIMD3<Float> = simd_cross(yarnAxis, e1)
    return (e1, e2)
}

func helixPoint(x: Float, radius r: Float, phase: Float) -> SIMD3<Float> {
    let (e1, e2) = yarnFrame()
    let theta: Float = phase + yarnHand * 2 * Float.pi * yarnTwistPerMillimetre * x / 1000
    return yarnOrigin + yarnAxis * x + (e1 * cos(theta) + e2 * sin(theta)) * r
}

/// Arc-length-spaced points along a helix from x0 to x1.
func helixPoints(x0: Float, x1: Float, radius r: Float, phase: Float, step: Float) -> [SIMD3<Float>] {
    let slope: Float = 1 / cos(helixAngle(radius: r))       // arc per unit x
    let n: Int = max(Int(ceil((x1 - x0) * slope / step)), 2)
    return (0...n).map { helixPoint(x: x0 + (x1 - x0) * Float($0) / Float(n), radius: r, phase: phase) }
}

/// Points along a cubic Bézier, resampled to even arc length.
func bezierPoints(_ p0: SIMD3<Float>, _ p1: SIMD3<Float>, _ p2: SIMD3<Float>, _ p3: SIMD3<Float>, step: Float) -> [SIMD3<Float>] {
    func at(_ t: Float) -> SIMD3<Float> {
        let u: Float = 1 - t
        let a: SIMD3<Float> = p0 * (u * u * u) + p1 * (3 * u * u * t)
        let b: SIMD3<Float> = p2 * (3 * u * t * t) + p3 * (t * t * t)
        return a + b
    }
    var dense: [SIMD3<Float>] = []
    for k in 0...2000 { dense.append(at(Float(k) / 2000)) }
    var cum: [Float] = [0]
    for k in 1..<dense.count { cum.append(cum[k - 1] + simd_distance(dense[k], dense[k - 1])) }
    let total: Float = cum.last!
    let n: Int = max(Int(ceil(total / step)), 2)
    var out: [SIMD3<Float>] = []
    var j: Int = 0
    for k in 0...n {
        let want: Float = total * Float(k) / Float(n)
        while j < cum.count - 2 && cum[j + 1] < want { j += 1 }
        let f: Float = (want - cum[j]) / max(cum[j + 1] - cum[j], 1e-9)
        let span: SIMD3<Float> = dense[j + 1] - dense[j]
        out.append(dense[j] + span * f)
    }
    return out
}

/// The fuzz fibre's free part: MODEL path. It leaves the yarn's top surface,
/// arcs up and to the right across the inset, and ends where it was cut —
/// the cut end faces the viewer, for the cross-section inset.
let fuzzLeaveX: Float = -150                 // axial position where it leaves (µm)
let fuzzEnd = SIMD3<Float>(150, 128, 70)     // its cut end
let fuzzEndDirection: SIMD3<Float> = simd_normalize(SIMD3<Float>(0.55, -0.10, 0.83))
/// Where along its free part the reversal is drawn (µm past leaving the
/// yarn). MODEL placement inside the observed spacing distribution.
let fuzzReversalAfter: Float = 250
/// Twist phase at the start. MODEL, set so the ribbon lies flat to the
/// viewer where it leaves the yarn.
let fuzzStartTwist: Float = 0.3

/// Vertex spacing along fibres: short enough that the twist turns at most
/// ~0.15 rad between vertices.
let fibreStep: Float = 8

func buildYarn(mutant: Mutant) -> YarnPiece {
    let section: CrossSection = mutant == .roundFibre ? .round() : .cotton()
    let rate: Float = mutant == .roundFibre ? 0 : twistRate
    let r: Float = surfaceRadius(section)
    let n: Int = surfaceFibreCount(section)
    var rng = Seeded(52)
    let (e1, e2) = yarnFrame()
    var surface: [Ribbon] = []
    // Fibre 0's phase puts it on top of the yarn (towards +y) where it leaves.
    let top: Float = atan2(simd_dot(SIMD3<Float>(0, 1, 0.35), e2), simd_dot(SIMD3<Float>(0, 1, 0.35), e1))
    let phase0: Float = top - yarnHand * 2 * Float.pi * yarnTwistPerMillimetre * fuzzLeaveX / 1000
    for k in 1..<n {
        let phase: Float = phase0 + 2 * Float.pi * Float(k) / Float(n)
        let pts: [SIMD3<Float>] = helixPoints(x0: -yarnHalfLength, x1: yarnHalfLength, radius: r, phase: phase, step: fibreStep)
        let len: Float = Float(pts.count - 1) * fibreStep
        let revs: [Float] = mutant == .noReversal ? [] : randomReversals(length: len * 1.2, spacing: meanReversalSpacing, rng: &rng)
        let hand: Float = rng.uniform(0, 1) < 0.5 ? 1 : -1
        let radial: SIMD3<Float> = simd_normalize(pts[0] - (yarnOrigin + yarnAxis * simd_dot(pts[0] - yarnOrigin, yarnAxis)))
        let t0: SIMD3<Float> = simd_normalize(pts[1] - pts[0])
        // Width starts tangent to the yarn's surface (lying flat), then twists.
        let flat: SIMD3<Float> = simd_normalize(simd_cross(radial, t0))
        var rib: Ribbon = makeRibbon(pts, startWidth: flat, startTwist: rng.uniform(0, 2 * Float.pi),
                                     reversals: revs, hand: hand, rate: rate, scale: 1)
        rib.cutEnd = true
        surface.append(rib)
    }
    // Fibre 0: along the surface from the far end to where it leaves, then free.
    let attached: [SIMD3<Float>] = helixPoints(x0: -yarnHalfLength, x1: fuzzLeaveX, radius: r, phase: phase0, step: fibreStep)
    let pLeave: SIMD3<Float> = attached[attached.count - 1]
    let tLeave: SIMD3<Float> = simd_normalize(attached[attached.count - 1] - attached[attached.count - 2])
    let radialLeave: SIMD3<Float> = simd_normalize(pLeave - (yarnOrigin + yarnAxis * simd_dot(pLeave - yarnOrigin, yarnAxis)))
    let lift: SIMD3<Float> = simd_normalize(tLeave + radialLeave * 0.9)
    let free: [SIMD3<Float>] = bezierPoints(pLeave, pLeave + lift * 200, fuzzEnd - fuzzEndDirection * 230, fuzzEnd, step: fibreStep)
    let centre: [SIMD3<Float>] = attached + free.dropFirst()
    var peel: Float = 0
    for k in 1..<attached.count { peel += simd_distance(attached[k], attached[k - 1]) }
    let revs: [Float] = mutant == .noReversal ? [] : [peel + fuzzReversalAfter]
    let t0: SIMD3<Float> = simd_normalize(centre[1] - centre[0])
    let radial0: SIMD3<Float> = simd_normalize(centre[0] - (yarnOrigin + yarnAxis * simd_dot(centre[0] - yarnOrigin, yarnAxis)))
    var fuzz: Ribbon = makeRibbon(centre, startWidth: simd_normalize(simd_cross(radial0, t0)), startTwist: fuzzStartTwist,
                                  reversals: revs, hand: 1, rate: rate, scale: 1)
    fuzz.highlight = true
    fuzz.cutEnd = true
    return YarnPiece(section: section, surface: surface, fuzz: fuzz, peelArc: peel, core: coreRadius(section), fibreCount: n)
}
