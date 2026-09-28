// The object: one raw grain of milled long-grain white rice, lying on the
// card at true scale beside a 4 mm ant — in millimetres — and its colours.

import Foundation
import simd

// MARK: - how big a grain of long-grain rice is

// A named variety, measured: LaKast, a US long-grain rice ("nonglutinous,
// nonaromatic", "Scent: Nonscented"). Table 10 of its patent gives the
// 2009–12 means: length 7.60 mm, width 2.08 mm, thickness 1.73 mm, 21.9 mg a
// grain (Moldenhauer, US 9,398,750 B2, 2016, patents.google.com/patent/
// US9398750B2). That the table is of MILLED grains is inferred from the
// patent's 21.9 g per 1000 milled grains elsewhere; it does not say so beside
// the table. FAO grades milled rice of 6.0–7.0 mm as long and 7.0 mm or more
// as extra long (fao.org/4/x5048e/x5048e02.htm): a long grain either way.
let grainLength: Float = 7.60        // mm
let grainWidth: Float = 2.08         // mm, across, lying flat
let grainThickness: Float = 1.73     // mm, top to bottom, lying flat
let grainMassMilligrams: Float = 21.9

/// The shape: an ellipsoid through those three measured dimensions. MODEL —
/// a milled grain is blunter at the ends and slightly flattened on its
/// sides; an ellipsoid has the right size in every direction and an exact
/// distance, which keeps the touch testable.
let grainSemiAxes = SIMD3<Float>(grainLength / 2, grainThickness / 2, grainWidth / 2)   // along, up, across

// The science mutant draws the grain at half its size. The tests must object.
func grainScale(_ m: Mutant) -> Float { m == .grainSize ? 0.5 : 1.0 }

/// Where it lies: flat on the card, its long axis turned 50° from the ant's
/// heading, away from the camera, so its near end lies in front of the ant's
/// head and the grain runs back behind it. MODEL placement (chosen so the
/// tip lifts across the camera's view, ~10 px at 1920 wide, not towards it).
let grainHeading: Float = -50 * Float.pi / 180
var grainAxis: SIMD3<Float> { SIMD3<Float>(cos(grainHeading), 0, sin(grainHeading)) }

/// Where on the grain the tip touches, in the grain's own frame: 84% of the
/// way from the middle to the near end, and on the upper side, turned 20°
/// from the top towards the far side. MODEL: the upper shoulder near the end,
/// the part of a grain an ant's antenna meets first.
let contactAlong: Float = -0.84
let contactRound: Float = -20 * Float.pi / 180
/// The antenna reaches straight ahead from its elbow; the tip's centre is
/// 0.97 of the funiculus's length away, so it is nearly straight. MODEL.
let reachDirection = SIMD2<Float>(1, 0)
let reachFraction: Float = 0.97
/// Which way the right funiculus bows: inwards, towards the ant's midline,
/// and a little down. MODEL: bowed up, as step 26's was, it would continue
/// the scape almost straight (a 17° elbow); bowed in, the elbow is ~50°.
let rightFuniculusBow = SIMD3<Float>(0, -0.4, -1)

struct RiceGrain {
    var centre: SIMD3<Float>
    var axis: SIMD3<Float>        // unit, along the grain, horizontal
    var semi: SIMD3<Float>        // along, up, across

    /// The grain's third axis: across it, horizontal.
    var across: SIMD3<Float> { simd_cross(axis, SIMD3<Float>(0, 1, 0)) }

    /// World point to the grain's own frame (along, up, across).
    func local(_ p: SIMD3<Float>) -> SIMD3<Float> {
        let q: SIMD3<Float> = p - centre
        return SIMD3<Float>(simd_dot(q, axis), q.y, simd_dot(q, across))
    }

    func world(_ l: SIMD3<Float>) -> SIMD3<Float> {
        let a: SIMD3<Float> = axis * l.x
        let c: SIMD3<Float> = across * l.z
        return centre + a + SIMD3<Float>(0, l.y, 0) + c
    }

    func worldDirection(_ l: SIMD3<Float>) -> SIMD3<Float> {
        let a: SIMD3<Float> = axis * l.x
        let c: SIMD3<Float> = across * l.z
        return a + SIMD3<Float>(0, l.y, 0) + c
    }
}

/// The contact point in the grain's frame, and the outward normal there.
func localContact(_ semi: SIMD3<Float>) -> (point: SIMD3<Float>, normal: SIMD3<Float>) {
    let along2: Float = contactAlong * contactAlong
    let left: Float = max(1 - along2, 0)
    let rest: Float = left.squareRoot()
    let y: Float = rest * semi.y * cos(contactRound)
    let z: Float = rest * semi.z * sin(contactRound)
    let p = SIMD3<Float>(contactAlong * semi.x, y, z)
    let n: SIMD3<Float> = p / (semi * semi)
    return (p, simd_normalize(n))
}

/// Where the contact must be in the world: at the contact's height above the
/// card, out along reachDirection from the elbow until the tip's centre (the
/// contact plus a tip radius along the normal) is reachFraction of the
/// funiculus's length from the elbow. Bisection, as step 42 placed its macaroni.
func solveContact(_ semi: SIMD3<Float>, axis: SIMD3<Float>) -> SIMD3<Float> {
    let elbow: SIMD3<Float> = rightElbow()
    let total: Float = funiculusLengths(count: workerAntennaSegments - 1).reduce(0, +)
    let probe = RiceGrain(centre: .zero, axis: axis, semi: semi)
    let (lp, ln) = localContact(semi)
    let n: SIMD3<Float> = probe.worldDirection(ln)
    let height: Float = semi.y + lp.y
    var lo: Float = 0
    var hi: Float = 3
    for _ in 0..<60 {
        let mid: Float = (lo + hi) / 2
        let xz: SIMD2<Float> = SIMD2<Float>(elbow.x, elbow.z) + reachDirection * mid
        let tip: SIMD3<Float> = SIMD3<Float>(xz.x, height, xz.y) + n * funiculusTipRadius
        if simd_distance(tip, elbow) < reachFraction * total { lo = mid } else { hi = mid }
    }
    let xz: SIMD2<Float> = SIMD2<Float>(elbow.x, elbow.z) + reachDirection * lo
    return SIMD3<Float>(xz.x, height, xz.y)
}

func buildGrain(mutant: Mutant) -> RiceGrain {
    let full: SIMD3<Float> = grainSemiAxes
    let contact: SIMD3<Float> = solveContact(full, axis: grainAxis)
    let lp: SIMD3<Float> = localContact(full).point
    let placed = RiceGrain(centre: .zero, axis: grainAxis, semi: full)
    // The full-size grain rests on the card with its contact at `contact`.
    let centre: SIMD3<Float> = contact - placed.worldDirection(lp)
    let k: Float = grainScale(mutant)
    if k == 1 { return RiceGrain(centre: centre, axis: grainAxis, semi: full) }
    // The grainSize mutant: half size, still on the card, centred where the
    // full grain was — so the tip, which stays put, no longer touches it.
    let small: SIMD3<Float> = full * k
    return RiceGrain(centre: SIMD3<Float>(centre.x, small.y, centre.z), axis: grainAxis, semi: small)
}

/// Exact distance to an ellipsoid with semi-axes `e` (the smallest is e.y,
/// the largest e.x), from a point `q` in its own frame. Eberly's reduction:
/// the nearest surface point is x_i = e_i² y_i / (t + e_i²), where t is the
/// root of F(t) = Σ (e_i y_i / (t + e_i²))² − 1. F falls and is convex, so
/// Newton's method started LEFT of the root climbs to it without ever
/// passing it — and short of the root, the distance it gives is short too,
/// never long: a ray can always trust it. Fourteen steps (tests measure it).
func ellipsoidDistance(_ q: SIMD3<Float>, _ e: SIMD3<Float>) -> Float {
    var y: SIMD3<Float> = simd_abs(q)
    y.y = max(y.y, 1e-4)
    let e2: SIMD3<Float> = e * e
    let g: Float = simd_length(y / e)
    let inside: Bool = g < 1
    let reach: Float = e.y * simd_length(y) - e2.x
    var t: Float = inside ? (e.y * y.y - e2.y) : max(0, reach)
    for _ in 0..<14 {
        let den: SIMD3<Float> = e2 + SIMD3<Float>(repeating: t)
        let a: SIMD3<Float> = e * y / den
        let a2: SIMD3<Float> = a * a
        let f: Float = a2.x + a2.y + a2.z - 1
        let s: SIMD3<Float> = a2 / den
        let df: Float = -2 * (s.x + s.y + s.z)
        t -= f / df
    }
    let den: SIMD3<Float> = e2 + SIMD3<Float>(repeating: t)
    let x: SIMD3<Float> = e2 * y / den
    let d: Float = simd_distance(x, y)
    return inside ? -d : d
}

/// The grain as a distance, mm.
func grainSDF(_ p: SIMD3<Float>, _ g: RiceGrain) -> Float {
    ellipsoidDistance(g.local(p), g.semi)
}

/// Where the right antenna's tip touches the grain: the contact point and
/// the exact surface normal there.
func contactPoint(_ g: RiceGrain) -> (point: SIMD3<Float>, normal: SIMD3<Float>) {
    let (lp, ln) = localContact(g.semi)
    return (g.world(lp), simd_normalize(g.worldDirection(ln)))
}

func contactPoint() -> (point: SIMD3<Float>, normal: SIMD3<Float>) { contactPoint(buildGrain(mutant: .none)) }

/// The grain's surface where the hair touches, as the micrometre inset sees
/// it: a sphere of the grain's Gaussian radius of curvature there, 1/√K, in
/// µm. K for an ellipsoid: 1 / (a²b²c² (x²/a⁴ + y²/b⁴ + z²/c⁴)²).
var grainSurfaceRadiusMicrometres: Float {
    let e: SIMD3<Float> = grainSemiAxes
    let p: SIMD3<Float> = localContact(e).point
    let e2: SIMD3<Float> = e * e
    let p2: SIMD3<Float> = p * p
    let e4: SIMD3<Float> = e2 * e2
    let w: SIMD3<Float> = p2 / e4
    let s: Float = w.x + w.y + w.z
    let abc: Float = e.x * e.y * e.z
    let k: Float = 1 / (abc * abc * s * s)
    return 1000 / k.squareRoot()
}

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

/// Milled white rice: a translucent, faintly warm white. MODEL colour — no
/// measured L*a*b* was opened for this step.
let riceLab = SIMD3<Float>(85, 0.5, 7)
let riceAlbedo: SIMD3<Float> = labToLinearSRGB(riceLab)

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
/// The grain's surface: a dry solid of starch and protein. Its index is
/// MODEL, taken as the cuticle's 1.56 for a satin sheen — no measured index
/// of the rice endosperm was opened for this step.
let riceIndex: Double = 1.56

let cuticleF0: Float = fresnelF0(1.0, cuticleIndex)
let riceF0: Float = fresnelF0(1.0, riceIndex)

// MARK: - dry: no film

/// Raw rice is dry: 11.62 g water per 100 g (USDA FoodData Central, SR
/// Legacy fdcId 169756, raw long-grain white rice). There is no liquid at its
/// surface for the tip to meet, and none is drawn: the hair touches the solid.
let riceWaterGramsPer100g: Float = 11.62
let grainHasFilm: Bool = false

// MARK: - why nothing is tasted

// A grain of rice is mostly starch, a glucose polymer; it is not sweet until
// it is broken down to sugars. Nothing measured shows an ant tasting starch
// (none was found). In insects: Drosophila needs Gr64a for glucose, sucrose
// and maltose (Jiao, Moon & Montell, *PNAS* 104: 14110, 2007); a fly's
// response to maltodextrin was put down to "the 10% contaminating simple
// sugars" (Burke & Waddell, *Curr Biol* 21: 746, 2011); in the blowfly
// starch is among "the inhibitors specific to the P site" (Ahamed et al.,
// *Chem Senses* 26: 507, 2001). Raw white rice carries 0.12 g of free sugars
// per 100 g (USDA fdcId 169756), and a dry solid gives the tip pore no
// solution to draw it from. So the taste inset shows starch, labelled as no
// taste, and nothing enters the taste pore.
let riceSugarsGramsPer100g: Float = 0.12

// MARK: - why there is a faint smell

// Plain white rice is not an aromatic rice: in non-aromatic IR-64, the
// aroma compound 2-acetyl-1-pyrroline was "below the level of detection" raw
// and cooked (Kasote et al., *Foods* 10: 1917, 2021). But raw rice is not
// odourless: "only eicosane and hexanal were detected in uncooked rice"
// (Dong et al., *Foods* 15: 2205, 2026). Hexanal, the green, grassy aldehyde
// of oxidised fat, is the one that flies; the inset shows a faint trace of
// it — fewer molecules than the cheese sauce of step 43 sent.
let objectHasOdour: Bool = true

// MARK: - the starch the inset shows

// Rice starch grains are compound, 10–20 µm across, and each granule in them
// is "a sharp-edged polyhedron with a typical diameter of 3 to 8 μm"
// (Matsushima et al., *Plant Physiol* 164: 623, 2014). The micrometre inset
// shows the grain's milled surface with a faint relief of facets that size —
// schematic: shading only, the surface itself kept smooth.
let starchGranuleRange: ClosedRange<Float> = 3...8     // µm
/// The facet size drawn, µm: the middle of that range.
let starchFacet: Float = 5.5
