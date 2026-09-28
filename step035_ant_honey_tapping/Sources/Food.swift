// The food: one small droplet of honey on the card, in millimetres, and the
// optics that make it glossy and amber.

import Foundation
import simd

// MARK: - what honey is

// Average composition of American honey, g per 100 g: fructose 38.19,
// glucose 31.28, water 17.2, sucrose 1.31, "maltose" (reducing
// disaccharides) 7.31 (White et al., "Composition of American Honeys", USDA
// Technical Bulletin 1261, 1962 — the figures the National Honey Board's
// reference guide also rests on). Fructose and glucose are 69% of it.
let honeyFructose: Float = 38.19
let honeyGlucose: Float = 31.28
let honeyWater: Float = 17.2

/// Molar masses, g/mol: C6H12O6 180.16, H2O 18.015.
let hexoseMolarMass: Float = 180.16
let waterMolarMass: Float = 18.015

/// Waters per fructose-or-glucose molecule in average honey: about 2.5. The
/// molecule inset draws its sugars and waters in that ratio.
let watersPerHexose: Float = (honeyWater / waterMolarMass) / ((honeyFructose + honeyGlucose) / hexoseMolarMass)

// MARK: - the droplet

// A droplet this small is a spherical cap: gravity cannot flatten it. The
// Bond number ρ g a² / γ compares weight with surface tension. Honey at 17%
// water has specific gravity 1.4237 at 20 °C (Bogdanov, *Book of Honey*
// ch. 4, "Physical properties of honey", Bee Product Science 2011, table
// after White). Its surface tension is not pinned down here — no measured
// value was checked — so the test uses a deliberately low 40 mN/m (water is
// 72); even so the Bond number is under 0.1.
let honeyDensity: Float = 1423.7          // kg/m³
let honeySurfaceTensionFloor: Float = 0.040   // N/m, UNVERIFIED lower bound, MODEL

/// Base radius and contact angle. MODEL: an ant-sized drop, about as wide as
/// the ant's head is long, and a contact angle for honey on a sized card —
/// honey wets such surfaces partially, so between 0° and 90°. No measured
/// angle for this pairing was found.
let dropBaseRadius: Float = 0.42            // mm
let dropContactAngle: Float = 65 * Float.pi / 180

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
/// Honey: refractive index 1.4935 at 17.2 g water per 100 g, 20 °C
/// (Bogdanov 2011, ch. 4, table "after Chataway": 17.2 → 1.4935). This is
/// how honey's water content is measured — by refractometer.
let honeyIndex: Double = 1.4935
/// Water, for comparison and for the tests.
let waterIndex: Double = 1.333

let cuticleF0: Float = fresnelF0(1.0, cuticleIndex)
let honeyF0: Float = fresnelF0(1.0, honeyIndex)

// Colour. The USDA grades honey colour by the optical density at 560 nm of
// a matching caramel–glycerin solution 3.15 cm thick: "amber" runs from
// 1.389 to 3.008 (USDA AMS, United States Standards for Grades of Extracted
// Honey; the same table in the National Honey Board's reference guide:
// light amber 51–85 mm Pfund, amber 86–114). This drop is a deep amber, OD
// 2.9, so at 560 nm it absorbs α = 2.9 · ln 10 / 31.5 mm = 0.21 per mm.
// Even so, a drop under a millimetre across is only pale gold: there is too
// little honey in the light's path to look like honey in a jar.
// How that absorption rises into the blue is MODEL: an exponential in
// wavelength, e-folding every 35 nm, evaluated at 610, 550 and 465 nm for
// the red, green and blue channels.
let amberOpticalDensity: Float = 2.9
let amberAlpha560: Float = amberOpticalDensity * Float(log(10.0)) / 31.5
func honeyAbsorption(atNanometres l: Float) -> Float { amberAlpha560 * exp((560 - l) / 35) }
let honeyAbsorptionRGB = SIMD3<Float>(honeyAbsorption(atNanometres: 610), honeyAbsorption(atNanometres: 550),
                                      honeyAbsorption(atNanometres: 465))   // per mm

// MARK: - why it is smell as well as taste

// Unlike sugar and salt, honey has a smell: it gives off volatile aroma
// compounds. Among the most odour-active in honey is phenylacetaldehyde
// (honey-like, floral), with one of the highest odour activity values in
// rape honey (Ruisinger & Schieberle, *J Agric Food Chem* 60: 4186, 2012,
// "Characterization of the key aroma compounds in rape honey by means of the
// molecular sensory science concept") and identified by aroma extract
// dilution in linden and other honeys. So here the smell hair has something
// to catch: a few phenylacetaldehyde molecules in the air, reaching its wall
// pores, while the taste hair touches the drop and tastes its sugars.
let foodHasVapour: Bool = true
