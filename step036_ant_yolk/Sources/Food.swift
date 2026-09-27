// The food: a crumb of hard-boiled egg yolk on the card, in millimetres —
// built from the yolk's own grains, the yolk spheres — and its colour.

import Foundation
import simd

// MARK: - what cooked yolk is

// Raw yolk, g per 100 g: water 52.31, fat 26.54, protein 15.86 (USDA
// FoodData Central, "Egg, yolk, raw, fresh", FDC 172184). Boiling drives off
// little water, so a crumb of hard-boiled yolk is about half water, a quarter
// fat and a sixth protein.
let yolkWater: Float = 52.31
let yolkFat: Float = 26.54
let yolkProtein: Float = 15.86

// Why cooked yolk crumbles: its original histological units, the yolk
// spheres, set as separate microgels. Boiled yolk gel is an accumulation of
// yolk-sphere microgels "with an average diameter of approximately 100 µm",
// irregular polyhedra "with diameters within 150 µm" (Food Chemistry 2023,
// "Effect of yolk spheres as a key histological structure on the
// morphology, character, and oral sensation of boiled egg yolk gel"). At
// this view's 4.4 µm per pixel a 100 µm sphere is 23 pixels across — so the
// crumb is drawn as what it is made of: yolk spheres, pressed together.
let yolkSphereMeanDiameter: Float = 0.100     // mm
let yolkSphereMaxDiameter: Float = 0.150      // mm

/// A crumb's size. MODEL: a crumb has no standard size; this one is under a
/// millimetre, a mouthful for an ant, and the tests hold it to 0.4–1.2 mm.
let crumbSizeRange: ClosedRange<Float> = 0.4...1.2

/// One yolk sphere: centre and radius, mm.
struct YolkSphere {
    var centre: SIMD3<Float>
    var radius: Float
}

/// A small deterministic random sequence, so the crumb is the same every run
/// and on the CPU and GPU alike.
struct Lcg {
    var state: UInt32
    mutating func next() -> Float {
        state = state &* 1_664_525 &+ 1_013_904_223
        return Float(state >> 8) / Float(1 << 24)
    }
}

/// The crumb: yolk spheres packed on a jittered grid inside an irregular
/// envelope — two overlapping ellipsoids, so it is lumpy, not an egg — with
/// the lowest resting on the card. Spheres buried deeper than one layer
/// inside are left out: nothing could see them, and a solid core ellipsoid
/// fills the middle so no gap opens. Sizes vary 100–136 µm across, mean
/// about 118, overlapping their 80 µm grid so they press into one another
/// as the polyhedral yolk spheres do. Placement MODEL.
func buildCrumb() -> (spheres: [YolkSphere], core: (centre: SIMD3<Float>, semi: SIMD3<Float>)) {
    var rng = Lcg(state: 20260927)
    let at = SIMD3<Float>(2.60, 0, 0.95)
    let lobes: [(c: SIMD3<Float>, r: SIMD3<Float>)] = [
        (at + SIMD3(0.0, 0.20, 0.0), SIMD3(0.36, 0.22, 0.28)),
        (at + SIMD3(0.20, 0.17, 0.14), SIMD3(0.22, 0.18, 0.20)),
    ]
    func inside(_ p: SIMD3<Float>) -> Float {
        // < 1 inside the envelope; the smaller the deeper.
        lobes.map { simd_length((p - $0.c) / $0.r) }.min() ?? 9
    }
    let step: Float = 0.080
    var out: [YolkSphere] = []
    for i in -7...8 {
        for j in 0...6 {
            for k in -6...7 {
                let base: SIMD3<Float> = at + SIMD3<Float>(Float(i), Float(j), Float(k)) * step
                let jitter = SIMD3<Float>(rng.next() - 0.5, rng.next() - 0.5, rng.next() - 0.5) * (step * 0.55)
                let r: Float = 0.050 + 0.018 * rng.next()
                var c: SIMD3<Float> = base + jitter
                let e: Float = inside(c)
                if e > 1.0 || e < 0.62 { continue }
                if c.y < r { c.y = r }          // resting on the card
                out.append(YolkSphere(centre: c, radius: r))
            }
        }
    }
    let core = (centre: at + SIMD3<Float>(0.05, 0.16, 0.03), semi: SIMD3<Float>(0.28, 0.14, 0.22))
    return (out, core)
}

/// Two loose bits beside the crumb: a few spheres each, broken off.
func buildBits() -> [YolkSphere] {
    [YolkSphere(centre: SIMD3(3.30, 0.050, 0.62), radius: 0.050),
     YolkSphere(centre: SIMD3(3.37, 0.046, 0.69), radius: 0.046),
     YolkSphere(centre: SIMD3(3.31, 0.105, 0.67), radius: 0.045),
     YolkSphere(centre: SIMD3(3.05, 0.055, 1.62), radius: 0.055),
     YolkSphere(centre: SIMD3(3.12, 0.048, 1.66), radius: 0.048)]
}

/// Blend radius where neighbouring spheres meet, mm. MODEL: pressed yolk
/// spheres meet in soft creases, not knife-edges.
let crumbBlend: Float = 0.026

/// Polynomial smooth minimum, as the kernel's smin. It never exceeds the
/// plain minimum, so the blended distance is still a safe lower bound.
func smoothMin(_ a: Float, _ b: Float, _ k: Float) -> Float {
    let h: Float = min(max(0.5 + 0.5 * (b - a) / k, 0), 1)
    return b + (a - b) * h - k * h * (1 - h)
}

/// The crumb and bits as a distance, the same arithmetic as the kernel's.
/// The core ellipsoid is only bounded (scaled by its smallest semi-axis), so
/// it can never over-report.
func crumbSDF(_ p: SIMD3<Float>, spheres: [YolkSphere], core: (centre: SIMD3<Float>, semi: SIMD3<Float>)) -> Float {
    let q: SIMD3<Float> = (p - core.centre) / core.semi
    var d: Float = (simd_length(q) - 1) * min(core.semi.x, min(core.semi.y, core.semi.z))
    for s in spheres {
        d = smoothMin(d, simd_distance(p, s.centre) - s.radius, crumbBlend)
    }
    return max(d, -p.y)
}

/// Where the right antenna's tip touches the crumb. Aim at a point on the
/// crumb's top, towards the ant; put the tip's centre on the line from there
/// straight up, exactly one tip radius from the crumb by the drawn distance
/// (bisection); the contact is the nearest crumb point to that centre.
func contactPoint() -> (point: SIMD3<Float>, normal: SIMD3<Float>) {
    let (spheres, core) = buildCrumb()
    let all: [YolkSphere] = spheres + buildBits()
    func d(_ p: SIMD3<Float>) -> Float { crumbSDF(p, spheres: all, core: core) }
    let tipR: Float = funiculusTipRadius
    let aim = SIMD2<Float>(2.44, 0.86)
    var lo: Float = 0.0
    var hi: Float = 2.0
    for _ in 0..<60 {
        let mid: Float = (lo + hi) / 2
        if d(SIMD3<Float>(aim.x, mid, aim.y)) > tipR { hi = mid } else { lo = mid }
    }
    let centre = SIMD3<Float>(aim.x, hi, aim.y)
    let e: Float = 1e-4
    let g = SIMD3<Float>(d(centre + SIMD3(e, 0, 0)) - d(centre - SIMD3(e, 0, 0)),
                         d(centre + SIMD3(0, e, 0)) - d(centre - SIMD3(0, e, 0)),
                         d(centre + SIMD3(0, 0, e)) - d(centre - SIMD3(0, 0, e)))
    // The blended field is not an exact distance, so walk the first guess
    // onto the surface itself (a few Newton steps along the gradient).
    func grad(_ p: SIMD3<Float>) -> SIMD3<Float> {
        simd_normalize(SIMD3<Float>(d(p + SIMD3(e, 0, 0)) - d(p - SIMD3(e, 0, 0)),
                                    d(p + SIMD3(0, e, 0)) - d(p - SIMD3(0, e, 0)),
                                    d(p + SIMD3(0, 0, e)) - d(p - SIMD3(0, 0, e))))
    }
    var p: SIMD3<Float> = centre - simd_normalize(g) * tipR
    for _ in 0..<8 { p -= grad(p) * d(p) }
    return (p, grad(p))
}

// MARK: - colour, from CIELAB

// Raw yolk colour spans L* 43–69, a* 0–13, b* 22–48 (Dvořák et al.,
// "Photocolorimetric determination of yolk colour in relation to selected
// quality parameters of eggs", *J Sci Food Agric* 2009: CIELAB after gloss
// removal, Isa Brown hens, compared with the Roche Yolk Colour Fan). Cooking turns it from deep orange to a paler,
// brighter yellow (the literature on thermal colour change of yolk says so
// qualitatively); no measured L*a*b* for hard-boiled yolk was found, so the
// value here is MODEL: just past the light, yellow end of the raw range.
let yolkLab = SIMD3<Float>(79, 1, 44)

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

let yolkAlbedo: SIMD3<Float> = labToLinearSRGB(yolkLab)

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
/// Water, for the moisture film where the taste hair touches.
let waterIndex: Double = 1.333

let cuticleF0: Float = fresnelF0(1.0, cuticleIndex)
let waterF0: Float = fresnelF0(1.0, waterIndex)

// MARK: - why it is smell as well as taste

// Lasius niger foragers bring in protein as well as sugar. Portha,
// Deneubourg & Detrain (*Anim Behav* 68: 115, 2004) offered L. niger scouts
// sucrose and protein droplets: nearly all drank sugar, a "substantial
// fraction" passed protein by, and the brood — which protein feeds — changed
// how many laid a trail to it. So an ant stopping at yolk is ordinary, and
// not guaranteed to take it. Yolk is protein and fat.
//
// And cooked yolk smells. The boiled-yolk aroma is mainly lipid-oxidation
// products: "boiled eggs mainly derived their flavor from hexanal, 2-pentyl-
// furan, 2-butanone, 3-methyl-butanal and heptane" (Zhou et al., *Foods*
// 13, 2024, "Unraveling the formation mechanism of egg's unique flavor via
// flavoromics and lipidomics", measured on yolk). So the smell hair here
// catches hexanal. (The sulphur smell of a boiled egg is hydrogen sulphide,
// but that comes mostly from the WHITE's proteins; it is not drawn as the
// yolk's.)
let foodHasVapour: Bool = true
