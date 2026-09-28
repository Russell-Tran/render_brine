// Why gold is gold, as numbers: the measured optical constants of the metal,
// the Fresnel equations for a conductor, and the colour science that turns a
// reflectance spectrum into a pixel. Nothing here touches the GPU; it is the
// part a test can read against the papers.
//
// Every earlier step reflected light off DIELECTRICS — water, enamel, glass —
// whose refractive index is a real number near 1.3–1.6 and whose reflectance
// at normal incidence is a few per cent, the same at every wavelength. A metal
// is different in kind. Its refractive index is complex, n + ik, and k — the
// extinction coefficient — is large: light entering the metal dies within a
// few tens of nanometres, and nearly all of it is reflected instead. How much
// is reflected depends on n and k, and n and k depend on wavelength. For gold
// they change sharply in the green, and that is the whole of its colour.
//
// WHY THEY CHANGE THERE. Gold absorbs blue and violet light because photons
// above about 2.4 eV can lift an electron from the filled 5d band to the Fermi
// level in the 6s–6p band. Below that energy only the free electrons respond,
// and a free-electron metal reflects almost everything, as silver does across
// the whole visible range. In silver the same d→s edge sits near 3.9 eV, in
// the ultraviolet, so silver reflects violet as well as red and looks white.
// The difference is relativistic: gold's 6s electrons move fast enough near
// its nucleus (Z = 79) that their orbital contracts and drops in energy, while
// the 5d orbitals expand and rise, and the gap between them closes (Pyykkö &
// Desclaux, "Relativity and the periodic system of elements", *Acc Chem Res*
// 12:276–281, 1979 — citation checked, text not reachable online). The size
// of the effect was calculated directly by Romaniello & de Boeij, "The role of
// relativity in the optical response of gold within the time-dependent
// current-density-functional theory", *J Chem Phys* 122:164303 (2005): the
// onset of interband absorption moves "from around 3.5 eV, obtained in a
// nonrelativistic calculation, to around 1.9 eV when relativity is included"
// (their abstract, checked). Without relativity gold's edge would be at
// 354 nm, in the ultraviolet — gold would be silvery. (They note their 1.9 eV
// is low because the local-density approximation underestimates the gap; the
// measured edge is the ~2.4 eV the tables below show.) The test suite finds
// both edges in the measured data rather than taking them on trust.

import Foundation
import simd

// MARK: - the measured optical constants

/// One row of an optical-constants table: vacuum wavelength in micrometres,
/// then n and k.
typealias NKRow = (micrometres: Double, n: Double, k: Double)

/// Gold, measured on vacuum-evaporated films by reflection and transmission at
/// room temperature: P. B. Johnson & R. W. Christy, "Optical constants of the
/// noble metals", *Phys Rev B* 6:4370–4379 (1972), Table I. Rows copied from
/// the refractiveindex.info database file `data/main/Au/nk/Johnson.yml`
/// (public domain), which lists the same paper as its source; J&C tabulate
/// at fixed photon energies, which is why the wavelengths are irregular.
/// 0.29–0.89 µm: the visible range and far enough into the ultraviolet to
/// find silver's edge.
let goldJC: [NKRow] = [
    (0.2924, 1.49, 1.878), (0.3009, 1.53, 1.889), (0.3107, 1.53, 1.893), (0.3204, 1.54, 1.898),
    (0.3315, 1.48, 1.883), (0.3425, 1.48, 1.871), (0.3542, 1.50, 1.866), (0.3679, 1.48, 1.895),
    (0.3815, 1.46, 1.933), (0.3974, 1.47, 1.952), (0.4133, 1.46, 1.958), (0.4305, 1.45, 1.948),
    (0.4509, 1.38, 1.914), (0.4714, 1.31, 1.849), (0.4959, 1.04, 1.833), (0.5209, 0.62, 2.081),
    (0.5486, 0.43, 2.455), (0.5821, 0.29, 2.863), (0.6168, 0.21, 3.272), (0.6595, 0.14, 3.697),
    (0.7045, 0.13, 4.103), (0.7560, 0.14, 4.542), (0.8211, 0.16, 5.083), (0.8920, 0.17, 5.663),
]

/// Silver, same paper, same table, same photon energies
/// (refractiveindex.info `data/main/Ag/nk/Johnson.yml`). Used for the
/// comparison the render's argument rests on, and by the `silver` mutant.
let silverJC: [NKRow] = [
    (0.2924, 1.39, 1.161), (0.3009, 1.34, 0.964), (0.3107, 1.13, 0.616), (0.3204, 0.81, 0.392),
    (0.3315, 0.17, 0.829), (0.3425, 0.14, 1.142), (0.3542, 0.10, 1.419), (0.3679, 0.07, 1.657),
    (0.3815, 0.05, 1.864), (0.3974, 0.05, 2.070), (0.4133, 0.05, 2.275), (0.4305, 0.04, 2.462),
    (0.4509, 0.04, 2.657), (0.4714, 0.05, 2.869), (0.4959, 0.05, 3.093), (0.5209, 0.05, 3.324),
    (0.5486, 0.06, 3.586), (0.5821, 0.05, 3.858), (0.6168, 0.06, 4.152), (0.6595, 0.05, 4.483),
    (0.7045, 0.04, 4.838), (0.7560, 0.03, 5.242), (0.8211, 0.04, 5.727), (0.8920, 0.04, 6.312),
]

/// Nickel, the barrier layer, for its colour in the cross-section inset:
/// P. B. Johnson & R. W. Christy, "Optical constants of transition metals: Ti,
/// V, Cr, Mn, Fe, Co, Ni, and Pd", *Phys Rev B* 9:5056–5070 (1974)
/// (refractiveindex.info `data/main/Ni/nk/Johnson.yml`).
let nickelJC: [NKRow] = [
    (0.354, 1.74, 2.32), (0.368, 1.70, 2.40), (0.381, 1.72, 2.48), (0.397, 1.72, 2.57),
    (0.413, 1.70, 2.69), (0.431, 1.71, 2.82), (0.451, 1.73, 2.95), (0.471, 1.78, 3.09),
    (0.496, 1.82, 3.25), (0.521, 1.85, 3.42), (0.549, 1.92, 3.61), (0.582, 1.96, 3.80),
    (0.617, 1.99, 4.02), (0.659, 1.99, 4.26), (0.704, 2.06, 4.50), (0.756, 2.13, 4.73),
    (0.821, 2.26, 4.97),
]

/// Brass, 70% copper and 30% zinc — cartridge brass, the lock's body — for its
/// colour in the inset: M. R. Querry, "Optical constants", US Army CRDEC
/// Contractor Report CRDC-CR-85034 (1985), DTIC ADA158623
/// (refractiveindex.info `data/other/alloys/Cu-Zn/nk/Querry-Cu70Zn30.yml`).
/// Which brass a padlock body is cast or machined from is not published for
/// the product whose dimensions are used; 70/30 is the table there is — MODEL.
let brassQuerry: [NKRow] = [
    (0.370, 1.497, 1.818), (0.380, 1.487, 1.818), (0.390, 1.471, 1.813), (0.400, 1.445, 1.805),
    (0.410, 1.405, 1.794), (0.420, 1.350, 1.786), (0.430, 1.278, 1.784), (0.440, 1.191, 1.797),
    (0.450, 1.094, 1.829), (0.460, 0.994, 1.883), (0.470, 0.900, 1.957), (0.480, 0.816, 2.046),
    (0.490, 0.745, 2.145), (0.500, 0.686, 2.250), (0.510, 0.639, 2.358), (0.520, 0.602, 2.464),
    (0.530, 0.573, 2.568), (0.540, 0.549, 2.668), (0.550, 0.527, 2.765), (0.560, 0.505, 2.860),
    (0.570, 0.484, 2.958), (0.580, 0.468, 3.059), (0.590, 0.460, 3.159), (0.600, 0.450, 3.253),
    (0.610, 0.452, 3.345), (0.620, 0.449, 3.434), (0.630, 0.445, 3.522), (0.640, 0.444, 3.609),
    (0.650, 0.444, 3.695), (0.660, 0.445, 3.778), (0.670, 0.444, 3.860), (0.680, 0.444, 3.943),
    (0.690, 0.445, 4.025), (0.700, 0.446, 4.106), (0.710, 0.448, 4.186), (0.720, 0.450, 4.266),
    (0.730, 0.452, 4.346), (0.740, 0.455, 4.424), (0.750, 0.457, 4.501), (0.760, 0.458, 4.579),
    (0.770, 0.460, 4.657), (0.780, 0.464, 4.737), (0.790, 0.469, 4.814), (0.800, 0.473, 4.890),
]

/// n and k at a wavelength, linearly interpolated between the table's rows.
/// J&C's rows are ~15–30 nm apart in the visible. Linear interpolation adds no
/// physics of its own — MODEL.
func nk(_ table: [NKRow], nanometres: Double) -> (n: Double, k: Double) {
    let um: Double = nanometres / 1000
    if um <= table[0].micrometres { return (table[0].n, table[0].k) }
    for i in 1..<table.count where table[i].micrometres >= um {
        let a: NKRow = table[i - 1]
        let b: NKRow = table[i]
        let t: Double = (um - a.micrometres) / (b.micrometres - a.micrometres)
        let n: Double = a.n + (b.n - a.n) * t
        let k: Double = a.k + (b.k - a.k) * t
        return (n, k)
    }
    let last: NKRow = table[table.count - 1]
    return (last.n, last.k)
}

// MARK: - the Fresnel equations for a conductor

/// A complex number, just enough of one for Snell's law with an absorbing
/// medium. Swift's standard library has none.
struct Complex {
    var re: Double
    var im: Double
    init(_ re: Double, _ im: Double = 0) { self.re = re; self.im = im }
    static func + (a: Complex, b: Complex) -> Complex { Complex(a.re + b.re, a.im + b.im) }
    static func - (a: Complex, b: Complex) -> Complex { Complex(a.re - b.re, a.im - b.im) }
    static func * (a: Complex, b: Complex) -> Complex {
        let re: Double = a.re * b.re - a.im * b.im
        let im: Double = a.re * b.im + a.im * b.re
        return Complex(re, im)
    }
    static func / (a: Complex, b: Complex) -> Complex {
        let d: Double = b.re * b.re + b.im * b.im
        let re: Double = (a.re * b.re + a.im * b.im) / d
        let im: Double = (a.im * b.re - a.re * b.im) / d
        return Complex(re, im)
    }
    var normSquared: Double { re * re + im * im }
    /// The principal square root, which for a medium with k ≥ 0 picks the wave
    /// that decays into the metal.
    var squareRoot: Complex {
        let r: Double = normSquared.squareRoot()
        let a: Double = ((r + re) / 2).squareRoot()
        let b: Double = ((r - re) / 2).squareRoot()
        return Complex(a, im < 0 ? -b : b)
    }
}

/// What to break, for the mutation check. Each one must make the suite fail.
enum Mutant: Int {
    case none = 0
    case silver = 1       // silver's n,k fed to the render instead of gold's
    case dielectric = 2   // Fresnel with k = 0: a transparent material with gold's n
    case noNickel = 3     // the barrier layer missing from the plating
    case thickGold = 4    // the gold 100× too thick in the inset
}

/// The mutant under test, read once from the environment, so the same binary
/// can be run broken on purpose.
let activeMutant: Mutant = {
    switch ProcessInfo.processInfo.environment["LOCK_MUTANT"] {
    case "silver": return .silver
    case "dielectric": return .dielectric
    case "noNickel": return .noNickel
    case "thickGold": return .thickGold
    default: return .none
    }
}()

/// Unpolarised reflectance of a flat interface from air (n = 1) into a medium
/// of complex index n + ik, for light arriving at angle θ from the normal
/// (`cosTheta` = cos θ). The Fresnel amplitude coefficients with a complex
/// index, written the way pbrt-v4's `FrComplex` writes them (Pharr, Jakob &
/// Humphreys, *Physically Based Rendering*, 4th ed., 2023; section number not
/// checked): Snell's law gives a complex cos θt, and
///     r⊥ = (cos θi − η cos θt) / (cos θi + η cos θt)
///     r∥ = (η cos θi − cos θt) / (η cos θi + cos θt)
/// The reflectance is the mean of |r⊥|² and |r∥|²: the lights are unpolarised.
/// The test suite checks this against the independent textbook closed form in
/// real arithmetic (the a², b² form) at every angle, so a slip in either shows.
func fresnelConductor(n: Double, k: Double, cosTheta: Double, mutant: Mutant = activeMutant) -> Double {
    let kk: Double = mutant == .dielectric ? 0 : k
    let eta = Complex(n, kk)
    let ci: Double = min(max(cosTheta, 0), 1)
    let sin2i: Double = 1 - ci * ci
    let eta2: Complex = eta * eta
    let sin2t: Complex = Complex(sin2i) / eta2
    let cost: Complex = (Complex(1) - sin2t).squareRoot
    let etaCi: Complex = eta * Complex(ci)
    let etaCt: Complex = eta * cost
    let rPerp: Complex = (Complex(ci) - etaCt) / (Complex(ci) + etaCt)
    let rPar: Complex = (etaCi - cost) / (etaCi + cost)
    let r: Double = (rPerp.normSquared + rPar.normSquared) / 2
    return min(r, 1)
}

// MARK: - colour: from a spectrum to a pixel

/// The wavelengths the colour integral runs over: 380–780 nm every 10 nm.
/// MODEL: finer than J&C's own row spacing (15–30 nm in the visible), so the
/// grid adds no detail the data lacks and loses none it has.
let spectralGrid: [Double] = (0...40).map { (i: Int) -> Double in
    let step: Double = 10 * Double(i)
    return 380 + step
}

/// The CIE 1931 2° standard observer, x̄ ȳ z̄, at 380, 390 … 780 nm. From the
/// CIE table as distributed by the Colour & Vision Research Laboratory, UCL
/// (cvrl.org, file `ciexyz31_1.csv`, 1 nm), sampled every 10 nm.
let cieXYZ1931: [SIMD3<Double>] = [
    SIMD3(0.001368, 0.000039, 0.006450), SIMD3(0.004243, 0.000120, 0.020050), SIMD3(0.014310, 0.000396, 0.067850),
    SIMD3(0.043510, 0.001210, 0.207400), SIMD3(0.134380, 0.004000, 0.645600), SIMD3(0.283900, 0.011600, 1.385600),
    SIMD3(0.348280, 0.023000, 1.747060), SIMD3(0.336200, 0.038000, 1.772110), SIMD3(0.290800, 0.060000, 1.669200),
    SIMD3(0.195360, 0.090980, 1.287640), SIMD3(0.095640, 0.139020, 0.812950), SIMD3(0.032010, 0.208020, 0.465180),
    SIMD3(0.004900, 0.323000, 0.272000), SIMD3(0.009300, 0.503000, 0.158200), SIMD3(0.063270, 0.710000, 0.078250),
    SIMD3(0.165500, 0.862000, 0.042160), SIMD3(0.290400, 0.954000, 0.020300), SIMD3(0.433450, 0.994950, 0.008750),
    SIMD3(0.594500, 0.995000, 0.003900), SIMD3(0.762100, 0.952000, 0.002100), SIMD3(0.916300, 0.870000, 0.001650),
    SIMD3(1.026300, 0.757000, 0.001100), SIMD3(1.062200, 0.631000, 0.000800), SIMD3(1.002600, 0.503000, 0.000340),
    SIMD3(0.854450, 0.381000, 0.000190), SIMD3(0.642400, 0.265000, 0.000050), SIMD3(0.447900, 0.175000, 0.000020),
    SIMD3(0.283500, 0.107000, 0.0), SIMD3(0.164900, 0.061000, 0.0), SIMD3(0.087400, 0.032000, 0.0),
    SIMD3(0.046770, 0.017000, 0.0), SIMD3(0.022700, 0.008210, 0.0), SIMD3(0.011359, 0.004102, 0.0),
    SIMD3(0.005790, 0.002091, 0.0), SIMD3(0.002899, 0.001047, 0.0), SIMD3(0.001440, 0.000520, 0.0),
    SIMD3(0.000690, 0.000249, 0.0), SIMD3(0.000332, 0.000120, 0.0), SIMD3(0.000166, 0.000060, 0.0),
    SIMD3(0.000083, 0.000030, 0.0), SIMD3(0.000042, 0.000015, 0.0),
]

/// CIE standard illuminant D65, relative spectral power, at the same grid
/// (ISO 11664-2 / CIE S 014-2; values as tabulated in the colour-science
/// package's `colour/colorimetry/datasets/illuminants/sds.py`, which copies the
/// CIE table). The studio lights are taken to be D65 — "white" in sRGB's own
/// sense, so a perfect mirror shows them as (1, 1, 1). MODEL: real studio
/// LEDs are near D65 but not on it.
let d65: [Double] = [
    49.9755, 54.6482, 82.7549, 91.4860, 93.4318, 86.6823, 104.865, 117.008, 117.812, 114.861,
    115.923, 108.811, 109.354, 107.802, 104.790, 107.689, 104.405, 104.046, 100.000, 96.3342,
    95.7880, 88.6856, 90.0062, 89.5991, 87.6987, 83.2886, 83.6992, 80.0268, 80.2146, 82.2778,
    78.2842, 69.7213, 71.6091, 74.3490, 61.6040, 69.8856, 75.0870, 63.5927, 46.4182, 66.8054,
    63.3828,
]

/// CIE XYZ of a surface with reflectance `r(λ)` lit by D65, normalised so a
/// perfect white reflector has Y = 1: X = Σ r S x̄ / Σ S ȳ, and so on.
func xyz(reflectance r: (Double) -> Double) -> SIMD3<Double> {
    var sum = SIMD3<Double>(0, 0, 0)
    var norm: Double = 0
    for (i, l) in spectralGrid.enumerated() {
        let s: Double = d65[i]
        let rs: Double = r(l) * s
        sum += cieXYZ1931[i] * rs
        norm += s * cieXYZ1931[i].y
    }
    return sum / norm
}

/// CIE XYZ to linear sRGB, IEC 61966-2-1's matrix (the same one step 20 uses).
func linearSRGB(xyz v: SIMD3<Double>) -> SIMD3<Double> {
    let r: Double = 3.2404542 * v.x - 1.5371385 * v.y - 0.4985314 * v.z
    let g: Double = -0.9692660 * v.x + 1.8760108 * v.y + 0.0415560 * v.z
    let b: Double = 0.0556434 * v.x - 0.2040259 * v.y + 1.0572252 * v.z
    return SIMD3(r, g, b)
}

/// A perfect reflector through the same 10 nm sums. D65 summed at 10 nm lands
/// a hair off sRGB's exact white point, so every reflectance colour is divided
/// by this (a von Kries correction of well under 1% per channel) and a perfect
/// mirror is exactly (1, 1, 1).
let whiteThroughGrid: SIMD3<Double> = linearSRGB(xyz: xyz { _ in 1.0 })

/// The linear-sRGB colour of white D65 light reflected off a flat surface of
/// this material at angle θ: the reflectance spectrum from the conductor
/// Fresnel equations, integrated against the eye's response. This is the
/// whole colour pipeline — nothing about "gold" is typed anywhere but the
/// n,k table.
func reflectanceRGB(_ table: [NKRow], cosTheta: Double, mutant: Mutant = activeMutant) -> SIMD3<Double> {
    let c: SIMD3<Double> = linearSRGB(xyz: xyz { l in
        let v = nk(table, nanometres: l)
        return fresnelConductor(n: v.n, k: v.k, cosTheta: cosTheta, mutant: mutant)
    })
    return c / whiteThroughGrid
}

/// Which metal the render's lock is made of. Gold — unless the `silver`
/// mutant swaps the table.
func lockSurfaceTable(_ mutant: Mutant) -> [NKRow] { mutant == .silver ? silverJC : goldJC }

/// How many entries the kernel's reflectance table has, evenly spaced in
/// cos θ from 0 (grazing) to 1 (normal). 129 steps of 1/128: the curve is
/// smooth everywhere except a knee near grazing, and at this spacing linear
/// interpolation is within 0.1% of the direct computation (tested).
let fresnelTableSize: Int = 129

/// The table the kernel reads: linear-sRGB reflectance of the lock's surface
/// against cos θ, computed spectrally, angle by angle, here on the CPU.
func fresnelTable(_ mutant: Mutant) -> [SIMD3<Float>] {
    let table: [NKRow] = lockSurfaceTable(mutant)
    return (0..<fresnelTableSize).map { i in
        let c: Double = Double(i) / Double(fresnelTableSize - 1)
        let v: SIMD3<Double> = reflectanceRGB(table, cosTheta: c, mutant: mutant)
        return SIMD3<Float>(Float(v.x), Float(v.y), Float(v.z))
    }
}

// MARK: - CIELAB, as in step 20

/// Linear sRGB to CIELAB (D65 white), the standard CIE formulas via XYZ —
/// copied from step 20's `Anatomy.swift`, extended to Double input.
func linearSRGBToLab(_ c: SIMD3<Double>) -> SIMD3<Double> {
    let x: Double = 0.4124564 * c.x + 0.3575761 * c.y + 0.1804375 * c.z
    let y: Double = 0.2126729 * c.x + 0.7151522 * c.y + 0.0721750 * c.z
    let z: Double = 0.0193339 * c.x + 0.1191920 * c.y + 0.9503041 * c.z
    func f(_ t: Double) -> Double {
        let delta: Double = 6.0 / 29.0
        return t > delta * delta * delta ? cbrt(t) : t / (3 * delta * delta) + 4.0 / 29.0
    }
    let fx: Double = f(x / 0.95047)
    let fy: Double = f(y / 1.0)
    let fz: Double = f(z / 1.08883)
    return SIMD3<Double>(116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz))
}

/// sRGB bytes to linear, the standard transfer curve.
func srgbByteToLinear(_ v: UInt8) -> Double {
    let c: Double = Double(v) / 255
    return c <= 0.04045 ? c / 12.92 : pow((c + 0.055) / 1.055, 2.4)
}

/// Linear to the sRGB transfer curve, 0…1.
func srgbEncode(_ v: Double) -> Double {
    let c: Double = min(max(v, 0), 1)
    return c <= 0.0031308 ? 12.92 * c : 1.055 * pow(c, 1.0 / 2.4) - 0.055
}

/// CIE76 colour difference.
func deltaE(_ a: SIMD3<Double>, _ b: SIMD3<Double>) -> Double { simd_distance(a, b) }

/// The kernel's tone curve, the same one step 20 uses, here on the CPU so a
/// test can predict a pixel from the constants alone: a filmic shoulder on
/// LUMINANCE, then the colour scaled to match, so a hue survives; only above
/// white does it bleed towards white. Returns display values 0…1.
func toneMap(_ x: SIMD3<Double>) -> SIMD3<Double> {
    let l: Double = 0.2126 * x.x + 0.7152 * x.y + 0.0722 * x.z
    let lm: Double = l * (1 + l / 16) / (1 + l)
    var c: SIMD3<Double> = x * (lm / max(l, 1e-5)) * toneGain
    let over: Double = max(c.max() - 1, 0)
    c = c + (SIMD3<Double>(1, 1, 1) - c) * min(over, 1)
    return SIMD3(srgbEncode(c.x), srgbEncode(c.y), srgbEncode(c.z))
}

/// The tone curve's gain, step 20's 1.12.
let toneGain: Double = 1.12

// MARK: - what the data says about the edge

/// The imaginary part of the dielectric function, ε₂ = 2nk: the absorption.
/// Its rise marks where interband transitions switch on.
func epsilon2(_ row: NKRow) -> Double { 2 * row.n * row.k }

/// The photon energy, eV, of a table's interband absorption edge: where ε₂
/// climbs most steeply between neighbouring rows, above ε₂'s minimum. The
/// minimum sits between the free-electron tail (rising towards the infrared)
/// and the interband edge (rising towards the ultraviolet), so the steepest
/// climb above it is the edge itself. The midpoint of the steepest segment is
/// taken — MODEL, a convention of this test; the edge is not a sharp line.
func interbandOnsetEV(_ table: [NKRow]) -> Double {
    let byEnergy: [NKRow] = table.sorted { $0.micrometres > $1.micrometres }   // low energy first
    var minIndex: Int = 0
    for i in byEnergy.indices where epsilon2(byEnergy[i]) < epsilon2(byEnergy[minIndex]) { minIndex = i }
    var best: Double = -.infinity
    var edge: Double = .infinity
    for i in (minIndex + 1)..<byEnergy.count {
        let ea: Double = photonEV / byEnergy[i - 1].micrometres
        let eb: Double = photonEV / byEnergy[i].micrometres
        let slope: Double = (epsilon2(byEnergy[i]) - epsilon2(byEnergy[i - 1])) / (eb - ea)
        if slope > best { best = slope; edge = (ea + eb) / 2 }
    }
    return edge
}

/// hc in eV·µm: a photon of wavelength λ µm carries 1.23984/λ eV (CODATA).
let photonEV: Double = 1.239842

/// How far light gets into the metal before its intensity falls by 1/e:
/// δ = λ / (4πk). Nanometres in, nanometres out.
func skinDepthNM(_ table: [NKRow], nanometres: Double) -> Double {
    let v = nk(table, nanometres: nanometres)
    return nanometres / (4 * Double.pi * v.k)
}
