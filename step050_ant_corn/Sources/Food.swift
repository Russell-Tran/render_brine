// The food: one kernel of fresh, raw sweet corn, cut from the cob, lying on
// the card beside a 4 mm ant — in millimetres — with a film of its own juice
// on the cut face, and its colours.

import Foundation
import simd

// MARK: - which corn, and how big a kernel is

// Sweet corn, not field corn: the sh2 ("supersweet") hybrid HMX59YS718.
// Its patent gives, for "average kernels from 10 ears", a "fresh kernel
// width 8.4 (mm)" and a "fresh kernel depth 12.9 (mm)", an ear diameter at
// mid-point of 5 cm, a cob diameter of 2.5 cm and 19.4 kernel rows; Table 2
// defines Kernel Width as "the width of a kernel at the mid-point in mm"
// (Snyder / HM Clause, US 11,234,394 B2, 2022, Table 1 and Table 2 —
// patents.google.com/patent/US11234394B2, read 2026-09-27).
let freshKernelWidth: Float = 8.4          // mm, across the row (round the ear)
let freshKernelDepth: Float = 12.9         // mm, from the crown in to the cob
let earDiameter: Float = 50                // mm
let cobDiameter: Float = 25                // mm
let kernelRows: Float = 19.4

// The two agree with each other: depth ≈ (ear − cob)/2 = 12.5 mm, and 19.4
// rows of 8.4 mm kernels go 163 mm round an ear whose circumference is 157
// mm. So the 8.4 mm is read as the kernel's width at the ear's surface — its
// crown, where it is widest. (The patent says "at the mid-point" without
// saying of what; the reading is checked by that sum, in a test.)
//
// Kernels are wedges: each fills its share of the ear's circle, so the width
// shrinks towards the cob in proportion to the distance from the ear's axis
// — at 12.9 mm in, (25 − 12.9)/25 of the crown's width. DERIVED from the
// ear diameter above; the kernel's own taper was not measured.
func kernelWidth(atDepth d: Float) -> Float { freshKernelWidth * (earDiameter / 2 - d) / (earDiameter / 2) }

/// Thickness, along the ear (the row's direction). MODEL, UNVERIFIED: the
/// patent does not give it. A dried dent kernel is 4.7 mm thick (inbred
/// LH176Bt810, US 6,469,233 B1 — a different corn, used only as a check on
/// size); a fresh sweet one is taken as a little plumper.
let kernelThickness: Float = 5.0
/// How rounded its long edges and crown are. MODEL: to the eye.
let kernelRounding: Float = 1.0

// The cut. Sweet corn is tasted where it is cut: its sugar is in the juice
// inside, and "Brix" in the patent is "the percentage of soluble solids in
// the juice of the kernel" (13.2 for this hybrid). Whether an intact
// kernel's skin is sugary outside was not found (UNVERIFIED), so the kernel
// here is cut — as a knife takes kernels off the cob, across the kernel,
// leaving its base on the cob. Where the knife goes is MODEL: 80% of the way
// in from the crown.
let cutFraction: Float = 0.80
var cutDepth: Float { cutFraction * freshKernelDepth }                   // 10.3 mm: the piece's length
/// The juice on the cut face: a thin film, the tip touches its surface.
/// MODEL, UNVERIFIED: no thickness of the juice on a cut kernel was found;
/// 20 µm, less than half the antenna tip's radius.
let juiceFilm: Float = 0.020

// The science mutant draws the kernel at half its measured size: the tests
// must object.
func kernelScale(_ m: Mutant) -> Float { m == .kernelSize ? 0.5 : 1.0 }

/// The kernel as placed. Its own frame: `origin` is on the card, at the
/// middle of the cut face; `axis` (x, z) runs from the cut face to the
/// crown, along the card; across it is `side`; up is y. In that frame the
/// kernel is u from 0 (cut) to `length` (crown), |v| ≤ half its width at
/// that u, and y from 0 to `thickness`, its edges rounded by `rounding`.
struct Kernel {
    var origin: SIMD2<Float>      // (x, z)
    var axis: SIMD2<Float>        // unit
    var length: Float             // cut face to crown
    var crownWidth: Float
    var taper: Float              // half-width lost per mm towards the cob
    var thickness: Float
    var rounding: Float
    var film: Float
    var fullDepth: Float          // crown to where the kernel met the cob

    var side: SIMD2<Float> { SIMD2<Float>(-axis.y, axis.x) }
    /// Outer half-width at u.
    func halfWidth(_ u: Float) -> Float { crownWidth / 2 - taper * (length - u) }
    /// Local coordinates of a world point.
    func local(_ p: SIMD3<Float>) -> SIMD3<Float> {
        let d = SIMD2<Float>(p.x, p.z) - origin
        return SIMD3<Float>(simd_dot(d, axis), simd_dot(d, side), p.y)
    }
    /// The rounded kernel's core: the wedge shrunk by `rounding` on every
    /// side (Kernel's surface is this core grown by `rounding`). The side
    /// lines move in by rounding × √(1 + taper²), square to themselves.
    var core: (u0: Float, u1: Float, r0: Float, r1: Float) {
        let u0: Float = length - fullDepth + rounding
        let u1: Float = length - rounding
        let slope2: Float = 1 + taper * taper
        let inward: Float = rounding * slope2.squareRoot()
        return (u0, u1, halfWidth(u0) - inward, halfWidth(u1) - inward)
    }
}

/// The contact's height on the cut face. MODEL: level with the ant's
/// antennal elbow (1.28 mm up), so the funiculus reaches almost straight, as
/// in step 42; the face is flat from `rounding` up to thickness − rounding.
let contactHeight: Float = 1.25
/// The kernel's axis, turned 50° from the ant's heading away from the
/// camera, so the juicy cut face half-faces the viewer. MODEL.
let kernelYaw: Float = 50 * Float.pi / 180
var kernelAxis: SIMD2<Float> { SIMD2<Float>(cos(kernelYaw), -sin(kernelYaw)) }
/// The cut face's outward normal in (x, z): straight back along the axis.
var contactDirection: SIMD2<Float> { -kernelAxis }
/// The antenna reaches from its elbow forward and a little in, almost
/// obliquely onto the face (38° off its normal); the tip's centre 0.97 of the
/// funiculus's length away, so the funiculus is nearly straight. MODEL.
let reachDirection: SIMD2<Float> = simd_normalize(SIMD2<Float>(0.95, -0.20))
/// Which way the right funiculus bows: back and up, away from the face.
let rightFuniculusBow = SIMD3<Float>(-1, 0.5, 0)
let reachFraction: Float = 0.97

/// The contact point in x, z: out along reachDirection from the elbow until
/// the tip's centre (the contact plus a tip radius along the face's normal)
/// is reachFraction of the funiculus's length from the elbow. Bisection.
func solveContactXZ() -> SIMD2<Float> {
    let elbow: SIMD3<Float> = rightElbow()
    let total: Float = funiculusLengths(count: workerAntennaSegments - 1).reduce(0, +)
    let n = SIMD3<Float>(contactDirection.x, 0, contactDirection.y)
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

func buildKernel(mutant: Mutant) -> Kernel {
    let k: Float = kernelScale(mutant)
    let taper: Float = (freshKernelWidth / 2) / (earDiameter / 2)      // 0.168: the same wedge at any size
    var kernel = Kernel(origin: .zero, axis: kernelAxis, length: cutDepth * k, crownWidth: freshKernelWidth * k,
                        taper: taper, thickness: kernelThickness * k, rounding: kernelRounding * k,
                        film: juiceFilm * k, fullDepth: freshKernelDepth * k)
    // The juice's surface, u = −film, passes through the contact point, on
    // the kernel's axis. The kernelSize mutant keeps that point and shrinks
    // the kernel about it.
    kernel.origin = solveContactXZ() + kernelAxis * kernel.film
    return kernel
}

/// Exact distance to an isosceles trapezoid in 2-D (Quílez): half-widths r1
/// at y = −he and r2 at y = +he, symmetric in x.
func trapezoidDistance(_ q: SIMD2<Float>, r1: Float, r2: Float, he: Float) -> Float {
    let p = SIMD2<Float>(abs(q.x), q.y)
    let k1 = SIMD2<Float>(r2, he)
    let k2 = SIMD2<Float>(r2 - r1, 2 * he)
    let cap: Float = p.y < 0 ? r1 : r2
    let ca = SIMD2<Float>(p.x - min(p.x, cap), abs(p.y) - he)
    let s0: Float = simd_dot(k1 - p, k2) / simd_dot(k2, k2)
    let s1: Float = min(max(s0, 0), 1)
    let cb: SIMD2<Float> = p - k1 + k2 * s1
    let sign: Float = (cb.x < 0 && ca.y < 0) ? -1 : 1
    return sign * min(simd_dot(ca, ca), simd_dot(cb, cb)).squareRoot()
}

/// The whole kernel, uncut, as a distance: the core wedge in (u, v), made a
/// slab in y, grown by the rounding — exact outside. Then cut: only u ≥ −film
/// is kept, the juice film being the slice from −film to 0 of the same
/// section. A max of lower bounds is a lower bound: never more than the truth.
func kernelSDF(_ p: SIMD3<Float>, _ m: Kernel) -> Float {
    let q: SIMD3<Float> = m.local(p)
    let c = m.core
    let mid: Float = (c.u0 + c.u1) / 2
    let d2: Float = trapezoidDistance(SIMD2<Float>(q.y, q.x - mid), r1: c.r0, r2: c.r1, he: (c.u1 - c.u0) / 2)
    let dw: Float = abs(q.z - m.thickness / 2) - (m.thickness / 2 - m.rounding)
    let outside: Float = simd_length(simd_max(SIMD2<Float>(d2, dw), SIMD2<Float>(0, 0)))
    let solid: Float = outside + min(max(d2, dw), 0) - m.rounding
    return max(solid, -q.x - m.film)
}

/// Where the right antenna's tip touches the juice: the contact point, and
/// the juice's normal there — the cut face's, straight back along the axis.
func contactPoint(_ m: Kernel) -> (point: SIMD3<Float>, normal: SIMD3<Float>) {
    let xz: SIMD2<Float> = m.origin - m.axis * m.film
    return (SIMD3<Float>(xz.x, contactHeight, xz.y), SIMD3<Float>(-m.axis.x, 0, -m.axis.y))
}

func contactPoint() -> (point: SIMD3<Float>, normal: SIMD3<Float>) { contactPoint(buildKernel(mutant: .none)) }

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

/// The skin (pericarp) over the yellow endosperm: the patent calls the
/// ear's "intensity of yellow color" medium. MODEL colour — no measured
/// L*a*b* for this kernel was found.
let pericarpLab = SIMD3<Float>(76, 6, 62)
/// The cut endosperm under its juice: paler and creamier than the skin. MODEL.
let endospermLab = SIMD3<Float>(84, 1, 42)
let pericarpAlbedo: SIMD3<Float> = labToLinearSRGB(pericarpLab)
let endospermAlbedo: SIMD3<Float> = labToLinearSRGB(endospermLab)

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
/// The juice is mostly water (76 g per 100 g, below): its gloss is taken as
/// water's, n = 1.333. Its 13.2 °Brix of dissolved solids raise that a
/// little — by how much was not checked, so it is not drawn. MODEL.
let juiceIndex: Double = 1.333

let cuticleF0: Float = fresnelF0(1.0, cuticleIndex)
let juiceF0: Float = fresnelF0(1.0, juiceIndex)

// MARK: - what the tip tastes: glucose

// Raw yellow sweet corn, per 100 g: water 76.05 g, total sugars 6.26 g —
// glucose 3.43, fructose 1.94, sucrose 0.89 — and starch 5.7 g (USDA
// FoodData Central, SR Legacy, fdcId 169998, "Corn, sweet, yellow, raw").
// Glucose is the largest of its sugars, so glucose is what the taste inset
// shows entering the pore. Whether an ant tastes starch is not known (no ant
// data found); sugars it does taste: Lasius niger workers prefer
// disaccharides to monosaccharides but take both (Madsen, Sørensen &
// Offenberg, *J Insect Physiol* 100: 140, 2017).
let cornWater: Float = 76.05
let cornGlucose: Float = 3.43
let cornFructose: Float = 1.94
let cornSucrose: Float = 0.89
let cornStarch: Float = 5.7

/// Molar masses, g/mol: C6H12O6 180.16, H2O 18.015.
let hexoseMolarMass: Float = 180.16
let waterMolarMass: Float = 18.015
/// Waters per glucose in the kernel: about 220. Far too many to draw; the
/// inset shows each glucose with three, and says so.
let watersPerGlucose: Float = (cornWater / waterMolarMass) / (cornGlucose / hexoseMolarMass)

// MARK: - why it is smell as well as taste

// Raw sweet corn smells. In native (untreated) sweet corn "alcohols were the
// main volatile components", and the common characteristic flavour
// compounds of native and washed corn "were screened as decylaldehyde and
// 1-octene-3-ol" (Zhang, Gao, Zhang, Feng & Zhuang, *Front Chem* 10: 725208,
// 2022 — sweet corn from the Henan Academy of Agricultural Sciences).
// 1-Octen-3-ol is the odorant drawn reaching the smell hair. Not dimethyl
// sulfide, the sulfur smell people know from corn: the same paper, citing
// Wiley 1985, puts "sulfur-containing compounds" as the largest share in
// BLANCHED corn, alcohols in native corn — and this kernel is raw.
let objectHasOdour: Bool = true
