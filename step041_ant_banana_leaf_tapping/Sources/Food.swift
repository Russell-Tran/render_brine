// The object: a piece of banana leaf (Musa) lying on the card, in
// millimetres — its thickness, its pleats and vein ridges, its cut edge —
// and its colour.

import Foundation
import simd

// MARK: - the leaf's surface, from measurements

// The leaf is a Cavendish banana's, and its surface is the one Alayan et al.
// measured ("Sustainable treatment of banana leaves for phytosanitary
// applications: impact, spreading, and impregnation of mineral oil", *Pest
// Manag Sci* 82: 5555, 2026, Musa acuminata Cavendish, abaxial surface, by
// profilometry and AFM). They found four textures, the first three
// "perpendicular to the midrib" — that is, running along the secondary
// veins:
//
//   * pleats 7 ± 1 mm apart, 250 ± 80 µm high;
//   * ridges 200 ± 50 µm apart, 15 ± 5 µm high;
//   * ridges 20 ± 5 µm apart, 2 ± 0.5 µm high;
//   * an even roughness 1 ± 0.5 µm apart, 0.2 ± 0.05 µm high (by AFM).
//
// The first two are drawn in the main view and the last two in the
// micrometre inset, each at the measured spacing and height. Where the
// paper's "peak height" is read as crest-to-trough, that is how it is used.
let pleatSpacing: Float = 7.0            // mm
let pleatHeight: Float = 0.250           // mm, crest to trough
let veinRidgeSpacing: Float = 0.200      // mm
let veinRidgeHeight: Float = 0.015       // mm
let cellRidgeSpacing: Float = 20.0       // µm (inset)
let cellRidgeHeight: Float = 2.0         // µm (inset)
let waxBumpSpacing: Float = 1.0          // µm (inset)
let waxBumpHeight: Float = 0.2           // µm (inset)

// And the same paper: a water drop on a banana leaf stands at a contact
// angle of 117 ± 4° — water-repellent, "like most plants". (The inset does
// not need it; the tests hold the constant to the cited value, since the
// caption's "waxy, water-repellent" rests on it.)
let leafWaterContactAngle: Float = 117

/// Lamina thickness. MODEL: Alayan et al. put the middle of the palisade
/// parenchyma 98.5 µm below the surface, which makes the whole lamina a few
/// hundred µm thick; 0.35 mm is a choice consistent with that, not a
/// measured thickness. UNVERIFIED.
let leafThickness: Float = 0.35          // mm

/// The veins' direction on the card (x, z), and the direction square to it
/// across which the pleats and ridges rise and fall. MODEL: the piece lies
/// with its veins running away from the ant, a little to its right.
let veinDirection: SIMD2<Float> = simd_normalize(SIMD2<Float>(1, 0.45))
let acrossVeins = SIMD2<Float>(-veinDirection.y, veinDirection.x)

/// The cut edge nearest the ant: a straight line through two points on the
/// card; the leaf lies on the side away from the ant. MODEL: placed so the
/// right antenna reaches the surface just inside the edge and the left
/// antenna's tip stays over bare card.
let cutEdgeA = SIMD2<Float>(2.28, 1.60)
let cutEdgeB = SIMD2<Float>(2.52, -0.80)
var cutEdgeInward: SIMD2<Float> {
    let d: SIMD2<Float> = simd_normalize(cutEdgeB - cutEdgeA)
    return SIMD2<Float>(-d.y, d.x)          // points +x, away from the ant
}

/// Phase of the pleats, mm across the veins. MODEL: a trough lies near the
/// contact, so the leaf there rests on the card and its top is within the
/// antenna's reach; the crests rise elsewhere.
let pleatPhase: Float = 0.4

/// Bottom of the leaf above the card: a pleat, 0 in the troughs where it
/// rests on the card, pleatHeight at the crests.
func leafBottom(_ xz: SIMD2<Float>) -> Float {
    let u: Float = simd_dot(xz, acrossVeins) - pleatPhase
    return pleatHeight * 0.5 * (1 - cos(2 * Float.pi * u / pleatSpacing))
}

/// Top of the leaf: bottom + thickness + the vein ridges.
func leafTop(_ xz: SIMD2<Float>) -> Float {
    let u: Float = simd_dot(xz, acrossVeins)
    let ridge: Float = veinRidgeHeight * 0.5 * (1 + cos(2 * Float.pi * u / veinRidgeSpacing))
    return leafBottom(xz) + leafThickness + ridge
}

/// The steepest slope the top can have, for the distance's safety factor:
/// the pleat's plus the ridges'.
let pleatMaxSlope: Float = pleatHeight * Float.pi / pleatSpacing
let veinRidgeMaxSlope: Float = veinRidgeHeight * Float.pi / veinRidgeSpacing
let leafMaxSlope: Float = pleatMaxSlope + veinRidgeMaxSlope

/// The leaf as a distance, mm: a sheet between two height fields, cut by
/// the edge. A height field's distance is its vertical gap divided by at
/// most √(1 + slope²), so dividing by that is safe; the max of safe bounds
/// is a safe bound of the intersection.
func leafSDF(_ p: SIMD3<Float>) -> Float {
    let xz = SIMD2<Float>(p.x, p.z)
    let k: Float = 1 / (1 + leafMaxSlope * leafMaxSlope).squareRoot()
    let above: Float = (p.y - leafTop(xz)) * k
    let below: Float = (leafBottom(xz) - p.y) * k
    let edge: Float = -simd_dot(xz - cutEdgeA, cutEdgeInward)
    return max(max(above, below), edge)
}

/// Where the right antenna's tip touches the leaf: the point on the top
/// surface straight below an aim point just inside the cut edge, and the
/// top's exact normal there, (−∂h/∂x, 1, −∂h/∂z) normalised. The ridges'
/// tightest curvature (radius 1 / (½·height·(2π/spacing)²) ≈ 0.14 mm) is
/// wider than the tip (0.048 mm), so a tip placed along that normal touches
/// at this one point.
let contactAim = SIMD2<Float>(2.50, 0.80)

func contactPoint() -> (point: SIMD3<Float>, normal: SIMD3<Float>) {
    let e: Float = 1e-4
    let h: Float = leafTop(contactAim)
    let dx: Float = (leafTop(contactAim + SIMD2(e, 0)) - leafTop(contactAim - SIMD2(e, 0))) / (2 * e)
    let dz: Float = (leafTop(contactAim + SIMD2(0, e)) - leafTop(contactAim - SIMD2(0, e))) / (2 * e)
    return (SIMD3<Float>(contactAim.x, h, contactAim.y), simd_normalize(SIMD3<Float>(-dx, 1, -dz)))
}

// MARK: - colour

// The lower (abaxial) side of a banana leaf is paler than the upper and
// carries a whitish wax bloom. Colour MODEL: no measured CIELAB value for
// the banana abaxial surface was found; this is a pale green, lightened
// further by the wax in the shader.
let leafLab = SIMD3<Float>(58, -22, 26)
/// The cut face shows the mesophyll: a deeper green. MODEL.
let leafCutLab = SIMD3<Float>(42, -30, 32)

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

let leafAlbedo: SIMD3<Float> = labToLinearSRGB(leafLab)
let leafCutAlbedo: SIMD3<Float> = labToLinearSRGB(leafCutLab)

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
let cuticleF0: Float = fresnelF0(1.0, cuticleIndex)

// MARK: - what the leaf is made of at its surface: wax

// Banana leaves are heavily waxed: field-grown leaves carry 80–90 µg of
// epicuticular wax per cm², and "the predominant chemical groups in leaf
// waxes were paraffins, primary alcohols and fatty acids, irrespective of
// genome" (Freeman & Turner, "The epicuticular waxes on the organs of
// different varieties of banana (Musa spp.) differ in form, chemistry and
// concentration", *Aust J Bot* 33: 393, 1985). The same paper found the
// wax's FORM varies with variety and leaf age; its crystal habit on the
// Cavendish abaxial surface was not found in a source that could be read,
// so the inset draws what was measured — Alayan et al.'s 1 µm, 0.2 µm
// roughness — and does not claim a crystal shape.
let leafWaxLoad: ClosedRange<Float> = 80...90      // µg/cm², field-grown

// Stomata: banana "has elliptical-shaped guard cells surrounded by four to
// six subsidiary cells" (Eyland et al., *Plant Physiol* 186: 998, 2021,
// citing Rudall et al. 2017), and they are counted on the abaxial side —
// 70 per mm² in Grand Naine (Cavendish) in Soares et al., *Plants* 11: 1953
// (2022), "Gene expression, histology and histochemistry in the interaction
// between Musa sp. and Pseudocercospora fijiensis". The ant stands on a
// piece lying lower side up, so this is the side it touches. At 70 per mm²
// a stoma is about 120 µm from the next, so one in the inset's 24 µm field
// is a choice of where to look, not the average view. Stoma size MODEL
// (UNVERIFIED): 20 µm long.
let stomataPerSquareMillimetre: Float = 70
let stomaLength: Float = 20.0            // µm, MODEL
let stomaWidth: Float = 13.0             // µm across both guard cells, MODEL

/// In the inset (µm): the direction across the 20 µm ridges, and where the
/// stoma sits, its long axis along the ridges (in parallel-veined leaves
/// stomata lie in rows along the veins; the alignment here is MODEL). Placement MODEL: in front of and left of the hairs, where it
/// can be seen; its centre height is the ridge surface there.
let insetAcrossRidges: SIMD2<Float> = simd_normalize(SIMD2<Float>(1, -0.35))
var stomaAxis: SIMD2<Float> { SIMD2<Float>(insetAcrossRidges.y, -insetAcrossRidges.x) * -1 }
func insetRidgeHeight(_ xz: SIMD2<Float>) -> Float {
    let u: Float = simd_dot(xz, insetAcrossRidges)
    return cellRidgeHeight * 0.5 * (cos(2 * Float.pi * u / cellRidgeSpacing) - 1)
}
let stomaAt = SIMD2<Float>(-6.0, 9.5)
var stomaCentre: SIMD3<Float> { SIMD3<Float>(stomaAt.x, insetRidgeHeight(stomaAt) - 0.3, stomaAt.y) }

// MARK: - smell: an intact leaf gives off little

// Green-leaf volatiles — the six-carbon "cut grass" smell — are "hardly
// detectable in undamaged plant tissues yet are rapidly synthesized from
// damaged cells within seconds of injury" (Matsui, *J Exp Bot* 77: 3267,
// 2026; and Matsui & Engelberth, *Plant Cell Physiol* 63: 1378, 2022:
// "resting levels in intact plant tissues are low"). The ant touches intact
// surface, a little way from the cut, and the piece was cut long enough ago
// (MODEL) that no burst is drawn: no odour molecules reach the smell hair.
let objectHasOdour: Bool = false

// Why a taste hair on wax is not nonsense: plant-surface waxes are the
// first chemistry an insect meets on a leaf, and plant-feeding insects do
// respond to them (Müller & Riederer, "Plant surface properties in chemical
// ecology", *J Chem Ecol* 31: 2621, 2005). Whether Lasius tastes leaf wax,
// and with what receptor, is not known, and the picture does not claim it.
