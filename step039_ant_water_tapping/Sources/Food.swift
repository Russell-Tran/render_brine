// The object: one small droplet of water on the card, in millimetres, and
// the optics that make it clear and glassy.

import Foundation
import simd

// MARK: - the droplet

// A droplet this small is a spherical cap: gravity cannot flatten it. The
// Bond number ρ g a² / γ compares weight with surface tension. Water at
// 25 °C: density 997.05 kg/m³ (the standard table value) and surface tension
// 71.97 mN/m (IAPWS, "Revised Release on Surface Tension of Ordinary Water
// Substance", 2014). The capillary length √(γ/ρg) is 2.7 mm; this drop is a
// sixth of that across, so its Bond number is about 0.02.
let waterDensity: Float = 997.05          // kg/m³
let waterSurfaceTension: Float = 0.07197  // N/m

/// Base radius and contact angle. MODEL: an ant-sized drop, about as wide as
/// the ant's head is long (step 34's honey drop, the same size), and a
/// contact angle for water on a plain sized card. Paper and card wet
/// partially; no measured angle for this card was found, so 75° is a choice
/// inside 0°–90°, and the tests only hold it there. UNVERIFIED.
let dropBaseRadius: Float = 0.42            // mm
let dropContactAngle: Float = 75 * Float.pi / 180

/// The sphere the cap is cut from: radius a / sin θ, centre below the card
/// by R cos θ, so the surface meets the card at exactly θ.
let dropSphereRadius: Float = dropBaseRadius / sin(dropContactAngle)
let dropAt = SIMD2<Float>(2.55, 0.85)       // (x, z) on the card. MODEL.
let dropCentre = SIMD3<Float>(dropAt.x, -dropSphereRadius * cos(dropContactAngle), dropAt.y)
var dropHeight: Float { dropSphereRadius * (1 - cos(dropContactAngle)) }

/// The drop as a distance: the sphere, cut off by the card. Exact outside
/// wherever the sphere is nearer than the cut; never more than the truth.
func dropSDF(_ p: SIMD3<Float>) -> Float {
    max(simd_length(p - dropCentre) - dropSphereRadius, -p.y)
}

/// Where the right antenna's tip touches the drop: a point on the cap near
/// its crown, on the side towards the ant, and the surface normal there.
func contactPoint() -> (point: SIMD3<Float>, normal: SIMD3<Float>) {
    let n: SIMD3<Float> = simd_normalize(SIMD3<Float>(-0.30, 0.95, -0.10))
    return (dropCentre + n * dropSphereRadius, n)
}

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
/// Water: n = 1.333 at 20 °C for yellow light, the textbook value.
let waterIndex: Double = 1.333

let cuticleF0: Float = fresnelF0(1.0, cuticleIndex)
let waterF0: Float = fresnelF0(1.0, waterIndex)

// Colour: none worth the name. Pure water's absorption coefficient at 610,
// 550 and 465 nm is 0.002644, 0.000565 and 0.0001011 per cm (Pope & Fry,
// "Absorption spectrum (380–700 nm) of pure water. II. Integrating cavity
// measurements", *Appl Opt* 36: 8710, 1997; the OMLC table of their data).
// Through a millimetre of drop that takes 0.03% of the red and less of the
// rest: the drop is colourless, and the kernel uses these numbers as they
// are, so it is colourless because water is, not because it was painted so.
let waterAbsorptionRGB = SIMD3<Float>(0.002644, 0.000565, 0.0001011) / 10   // per mm

// MARK: - why it is taste and not smell

// Water has no smell — but insects can TASTE it. Drosophila has dedicated
// water-taste neurons, and the channel that makes them fire on water is
// PPK28, an osmosensitive member of the DEG/ENaC family: without it the
// neurons ignore water and flies drink less (Cameron, Hiroi, Ngai & Scott,
// *Nature* 465: 91, 2010; Chen, Wang & Wang, *J Neurosci* 30: 6247, 2010).
// No one has shown the same channel in Lasius, and the picture does not
// claim it: it shows that a taste hair's one pore is where water is
// touched, not which receptor answers.
//
// Water is volatile, though, and the air over a drop is humid. Insects sense
// humidity with hygroreceptors, in their own sensillum type — not the
// many-pored smell hairs. In the formicine ant Camponotus japonicus (the
// same subfamily as Lasius) Nakanishi et al. (*Cell Tissue Res* 338: 79,
// 2009) found "coelocapitular sensilla (putatively hygro- and
// thermoreceptive)", three neurons each; the name says what they look like,
// a knob (capitulum) in a hollow (coelo-). Their size is not given in the
// abstract, so the one in the inset is sized MODEL. It is labelled, and no
// odour molecules reach the smell hair.
let objectIsVolatile: Bool = true
let objectHasOdour: Bool = false

// MARK: - the water-repellent tip

// Insect cuticle is waxed: a layer of cuticular hydrocarbons — straight and
// branched alkanes and alkenes — restricts water loss (Blomquist & Ginzel,
// *Annu Rev Entomol* 66: 45, 2021), and ant cuticle repels water (Mlot,
// Tovey & Hu, *PNAS* 108: 7669, 2011, on fire ants). No contact angle has
// been measured on a Lasius antenna. So the inset draws a hydrophobic tip
// touching the surface: the water is pressed into a shallow dimple under the
// apex and does not climb the hair. Dimple depth and width MODEL.
let dimpleDepth: Float = 0.15      // µm
let dimpleWidth: Float = 0.5       // µm, e-folding distance from the hair
