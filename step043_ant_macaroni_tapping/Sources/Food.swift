// The object: one elbow macaroni in orange cheese sauce, lying on the card,
// at true scale beside a 4 mm ant — in millimetres — and its colours.

import Foundation
import simd

// MARK: - how big an elbow macaroni is

// The legal definition: in the US standard of identity, "macaroni" is
// "tube-shaped and more than 0.11 inch but not more than 0.27 inch in
// diameter" (21 CFR 139.110(a), via the Cornell LII text) — 2.79 to 6.86 mm,
// DRY. That is the only dimension found in a source that could be read;
// everything below builds on it and is marked.
let dryDiameterRange: ClosedRange<Float> = (0.11 * 25.4)...(0.27 * 25.4)    // mm

/// Dry outer diameter: 0.20 inch, inside the legal range. MODEL — elbow
/// macaroni sits in the upper half of it; no packet measurement was found.
let dryOuterDiameter: Float = 0.20 * 25.4          // 5.08 mm
/// Dry wall. MODEL: about a fifth of the diameter. UNVERIFIED.
let dryWall: Float = 1.0                           // mm
/// Linear swelling on cooking. MODEL (UNVERIFIED): pasta roughly doubles
/// its mass in water as it cooks; if that went into volume evenly, sizes
/// would grow by the cube root of about 2.2, 1.3×. No measured swelling of
/// elbow macaroni's diameter was found.
let cookedSwelling: Float = 1.3

let outerRadius: Float = dryOuterDiameter * cookedSwelling / 2     // 3.30 mm
let wall: Float = dryWall * cookedSwelling                          // 1.30 mm
let innerRadius: Float = outerRadius - wall                         // 2.00 mm

/// The elbow: the tube's centre line bends round a circle of this radius
/// through this angle. MODEL: an elbow's outer curve about 2 cm long, as a
/// cooked elbow is to the eye. UNVERIFIED.
let bendRadius: Float = 5.0                        // mm, to the tube's centre line
let bendAngle: Float = 140 * Float.pi / 180
var outerArcLength: Float { (bendRadius + outerRadius) * bendAngle }

/// The sauce over it: thicker low down, where it has run, thinner on top,
/// where it has drained and the pasta shows through. MODEL thicknesses.
let sauceTop: Float = 0.08                         // mm
let sauceBottom: Float = 0.30                      // mm

// The science mutant draws the macaroni at the ant's scale instead of its
// own: every length above shrunk by this factor. The tests must object.
func macaroniScale(_ m: Mutant) -> Float { m == .antScale ? 0.2 : 1.0 }

/// Where it lies: resting on the card, its outer curve towards the ant.
/// The torus's axis is vertical, through `bendCentre` on the card; the
/// middle of the elbow points at the ant (direction `facing`, from the
/// centre). MODEL placement, set so the right antenna meets the outer curve.
struct Macaroni {
    var centre: SIMD2<Float>      // (x, z) of the bend's axis
    var facing: SIMD2<Float>      // unit, from the axis towards the middle of the elbow
    var bend: Float
    var ro: Float                 // pasta outer radius
    var ri: Float                 // pasta inner radius
    var sTop: Float
    var sBottom: Float
    var angle: Float

    var tubeCentreHeight: Float { ro }   // the pasta rests on the card
}

/// The contact's height on the tube. MODEL: the lower flank of the tube,
/// level with the ant's antennal elbow, which the funiculus reaches
/// almost straight — a tube 6.6 mm tall overhangs an ant 1 mm tall, and a
/// funiculus bent up and over would pass through it.
let contactHeight: Float = 1.25          // mm above the card
/// The outer curve at the contact faces straight back at the ant (−x). The
/// elbow's middle is turned 68° from there, away from the camera, so the
/// macaroni lies behind and beside the ant instead of between it and the
/// viewer; the near cut end is then 2° past the contact. MODEL.
let contactBearing: Float = Float.pi                   // direction from the bend axis to the contact
let middleOffset: Float = 69.5 * Float.pi / 180
var facingAnt: SIMD2<Float> { SIMD2<Float>(cos(contactBearing + middleOffset), sin(contactBearing + middleOffset)) }
var contactDirection: SIMD2<Float> { SIMD2<Float>(cos(contactBearing), sin(contactBearing)) }
/// The antenna reaches from its elbow forward and out to the right,
/// the tip's centre 0.97 of the funiculus's length away — so the funiculus
/// is nearly straight and stays clear of the overhang. MODEL.
let reachDirection: SIMD2<Float> = simd_normalize(SIMD2<Float>(0.80, 0.60))
/// Which way the right funiculus bows: back and up, away from the tube.
let rightFuniculusBow = SIMD3<Float>(-1, 0.5, 0)
let reachFraction: Float = 0.97

/// The contact point in x, z: out along reachDirection from the elbow until
/// the tip's centre (the contact plus a tip radius along the surface normal)
/// is reachFraction of the funiculus's length from the elbow. Bisection.
func solveContactXZ(_ m: Macaroni) -> SIMD2<Float> {
    let elbow: SIMD3<Float> = rightElbow()
    let total: Float = funiculusLengths(count: workerAntennaSegments - 1).reduce(0, +)
    let y: Float = contactHeight - m.tubeCentreHeight
    let sr: Float = m.ro + sauceThickness(m, dy: y)
    let out: Float = max(sr * sr - y * y, 0).squareRoot()
    let n = SIMD3<Float>(contactDirection.x * out / sr, y / sr, contactDirection.y * out / sr)
    var lo: Float = 0
    var hi: Float = 3
    for _ in 0..<60 {
        let mid: Float = (lo + hi) / 2
        let xz: SIMD2<Float> = SIMD2<Float>(elbow.x, elbow.z) + reachDirection * mid
        let tip: SIMD3<Float> = SIMD3<Float>(xz.x, contactHeight, xz.y) + n * funiculusTipRadius
        if simd_distance(tip, elbow) < reachFraction * total { lo = mid } else { hi = mid }
    }
    return SIMD2<Float>(elbow.x, elbow.z) + reachDirection * lo
}

func buildMacaroni(mutant: Mutant) -> Macaroni {
    let k: Float = macaroniScale(mutant)
    var m = Macaroni(centre: .zero, facing: facingAnt, bend: bendRadius * k, ro: outerRadius * k, ri: innerRadius * k,
                     sTop: sauceTop * k, sBottom: sauceBottom * k, angle: bendAngle)
    // Put the axis so the sauce's outer surface passes through the contact
    // point at contactHeight, straight out along `contactDirection`.
    let y: Float = contactHeight - m.tubeCentreHeight
    let sr: Float = m.ro + sauceThickness(m, dy: y)
    let reach: Float = m.bend + max(sr * sr - y * y, 0).squareRoot()
    // The contact is placed for the full-size macaroni; the antScale
    // mutant keeps that point and shrinks the macaroni about it.
    let full = Macaroni(centre: .zero, facing: facingAnt, bend: bendRadius, ro: outerRadius, ri: innerRadius,
                        sTop: sauceTop, sBottom: sauceBottom, angle: bendAngle)
    m.centre = solveContactXZ(full) - contactDirection * reach
    return m
}

/// Sauce thickness at height dy above the tube's centre line: sTop at the
/// top, sBottom at the bottom, linear between.
func sauceThickness(_ m: Macaroni, dy: Float) -> Float {
    let t: Float = min(max((m.ro - dy) / (2 * m.ro), 0), 1)
    return m.sTop + (m.sBottom - m.sTop) * t
}

/// The macaroni, sauce and all, as a distance (mm). In the tube's cross-
/// section (ρ from the bend axis, y) the solid is the ring between the
/// inner and outer sauce surfaces; the elbow's two straight cut ends are the
/// planes through the axis at ±angle/2 from `facing`; the card cuts it
/// below. Each piece is a safe bound, the thickness's slow change with
/// height is allowed for, and the max of safe bounds is a safe bound.
func macaroniSDF(_ p: SIMD3<Float>, _ m: Macaroni) -> Float {
    let rel = SIMD2<Float>(p.x, p.z) - m.centre
    let rho: Float = simd_length(rel)
    let dy: Float = p.y - m.tubeCentreHeight
    let q = SIMD2<Float>(rho - m.bend, dy)
    let r: Float = simd_length(q)
    let s: Float = sauceThickness(m, dy: dy)
    let mid: Float = (m.ro + m.ri) / 2
    let half: Float = (m.ro - m.ri) / 2 + s
    let slope: Float = (m.sBottom - m.sTop) / (2 * m.ro)
    let ring: Float = (abs(r - mid) - half) / (1 + slope)
    let (n1, n2) = wedgeNormals(m)
    let wedge: Float = max(simd_dot(rel, n1), simd_dot(rel, n2))
    return max(max(ring, wedge), -p.y)
}

/// The outward normals of the two cut-end planes (in x, z).
func wedgeNormals(_ m: Macaroni) -> (SIMD2<Float>, SIMD2<Float>) {
    let h: Float = m.angle / 2
    func turn(_ v: SIMD2<Float>, _ a: Float) -> SIMD2<Float> {
        SIMD2<Float>(v.x * cos(a) - v.y * sin(a), v.x * sin(a) + v.y * cos(a))
    }
    // The end directions, and the normals pointing away from the wedge.
    let e1: SIMD2<Float> = turn(m.facing, h)
    let e2: SIMD2<Float> = turn(m.facing, -h)
    return (turn(e1, Float.pi / 2), turn(e2, -Float.pi / 2))
}

/// Where the right antenna's tip touches the sauce: the contact point, and
/// the exact surface normal there (the tube's own, straight out of its
/// cross-section).
func contactPoint(_ m: Macaroni) -> (point: SIMD3<Float>, normal: SIMD3<Float>) {
    let y: Float = contactHeight - m.tubeCentreHeight
    let sr: Float = m.ro + sauceThickness(m, dy: y)
    let out: Float = max(sr * sr - y * y, 0).squareRoot()
    let d: SIMD2<Float> = contactDirection
    let xz: SIMD2<Float> = m.centre + d * (m.bend + out)
    let n2 = SIMD2<Float>(out, y) / sr                    // (radial, up) in the cross-section
    let n = SIMD3<Float>(d.x * n2.x, n2.y, d.y * n2.x)
    return (SIMD3<Float>(xz.x, contactHeight, xz.y), simd_normalize(n))
}

func contactPoint() -> (point: SIMD3<Float>, normal: SIMD3<Float>) { contactPoint(buildMacaroni(mutant: .none)) }

// MARK: - colour

/// CIELAB (D65) to linear sRGB, the standard formulas.
func labToLinearSRGB(_ lab: SIMD3<Float>) -> SIMD3<Float> {
    let fy: Float = (lab.x + 16) / 116
    let fx: Float = fy + lab.y / 500
    let fz: Float = fy - lab.z / 200
    func finv(_ t: Float) -> Float { t > 6.0 / 29.0 ? t * t * t : 3 * pow(6.0 / 29.0, 2) * (t - 4.0 / 29.0) }
    let X: Float = 0.95047 * finv(fx)
    let Y: Float = 1.0 * finv(fy)
    let Z: Float = 1.08883 * finv(fz)
    let r: Float = 3.2406 * X - 1.5372 * Y - 0.4986 * Z
    let g: Float = -0.9689 * X + 1.8758 * Y + 0.0415 * Z
    let b: Float = 0.0557 * X - 0.2040 * Y + 1.0570 * Z
    return simd_clamp(SIMD3<Float>(r, g, b), SIMD3<Float>(repeating: 0), SIMD3<Float>(repeating: 1))
}

/// The sauce: boxed mac-and-cheese orange. MODEL colour — no measured
/// L*a*b* for the sauce was found.
let sauceLab = SIMD3<Float>(64, 30, 66)
/// Cooked durum pasta: pale cream-yellow. MODEL.
let pastaLab = SIMD3<Float>(78, 2, 30)
let sauceAlbedo: SIMD3<Float> = labToLinearSRGB(sauceLab)
let pastaAlbedo: SIMD3<Float> = labToLinearSRGB(pastaLab)

// MARK: - optics

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
/// The sauce's surface is water-based (a dairy emulsion): its gloss is
/// water's, n = 1.333.
let sauceIndex: Double = 1.333

let cuticleF0: Float = fresnelF0(1.0, cuticleIndex)
let sauceF0: Float = fresnelF0(1.0, sauceIndex)

// MARK: - why it is orange

// Boxed macaroni and cheese is coloured, not orange by nature: Kraft's
// sauce was coloured with Yellow 5 and Yellow 6 until 2016 and since then
// with "paprika, annatto, and turmeric" (Wikipedia, "Kraft Dinner"). Annatto
// — from the seeds of Bixa orellana — colours with its carotenoids bixin and
// norbixin (EFSA, *EFSA J* 17: e05626, 2019), and hard and semi-hard
// cheeses are the foods most often labelled with it (same opinion). The
// taste inset shows norbixin, the water-soluble one — as the molecule that
// makes the sauce orange, NOT as something the ant tastes.

// MARK: - why it is smell as well as taste

// Cheese sauce smells. Among Cheddar's key odorants, confirmed by aroma
// recombination, are acetic and butanoic acid (Wang et al., "Key aroma
// compounds identified in Cheddar cheese with different ripening times...",
// *J Dairy Sci* 104: 1576, 2021). Butanoic acid — the smell of butter and
// cheese — is the odorant drawn reaching the smell hair. (Processed cheese
// sauce is not aged Cheddar; its odorants are assumed to include the same
// short fatty acid, which dairy fat releases. MODEL.)
let objectHasOdour: Bool = true

// And it has tastes an insect can detect. In Drosophila, low salt is
// sensed through the ionotropic receptor IR76b (Zhang, Ni & Montell,
// *Science* 340: 1334, 2013) and fatty acids through IR25a, IR76b and IR56d
// (Ahn, Chen & Amrein, *eLife* 6: e30115, 2017). None of this has been
// shown for Lasius, and the picture does not claim it: the taste inset
// shows the salt that is there, not a receptor.

/// The sauce's surface where the hair touches, as the micrometre inset
/// sees it: a sphere with the tube's own radius there (the sauce's outer
/// radius at the contact height), in µm. At 3.6 mm it is flat across the
/// inset's 14.5 µm.
var sauceSurfaceRadiusMicrometres: Float {
    let m: Macaroni = buildMacaroni(mutant: .none)
    return (m.ro + sauceThickness(m, dy: contactHeight - m.tubeCentreHeight)) * 1000
}
