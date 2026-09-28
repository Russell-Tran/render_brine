// What a plastic toy is made of, as numbers: the refractive index of PVC,
// what that index says about how much light the surface mirrors, the paints
// over it, and the tone curve that turns light into a pixel. Nothing here
// touches the GPU; it is the part a test can read against the sources.
//
// THE OBJECT. Small, one-piece, solid animal figures, a few centimetres long,
// sold loose or in sets in a clear plastic tube or box. Russell's spec for
// these steps: "The plastic toys should be the kind that are like 2-3 inches
// lengthwise" — 50.8 to 76.2 mm nose to tail tip, which is what `Toy.swift`
// checks. How they are made, from the sources that could be reached:
//   * the material: figures of this kind "are made of different plastics,
//     like polyvinyl chloride", with a non-phthalate plasticiser (Wikipedia,
//     article on a German figurine maker, "Schleich", checked 2026-09-28; the
//     maker is not named anywhere in the picture). Hard and soft plastic
//     figures are "generally painted" (Wikipedia, "Toy soldier", checked).
//     PVC is taken as the plastic here.
//   * the making: "a mould is made for the injection moulding machine. The
//     figurines are then cast and hand painted" (same article); and on any
//     injection moulding, "a parting line, sprue, gate marks, and ejector pin
//     marks are usually present on the final part" (Wikipedia, "Injection
//     moulding", checked). The render keeps the parting line and leaves out
//     the gate and ejector marks, which sit underneath.
//   * the finish: painted, matte-ish; modern figures use "matte-finished
//     acrylic paint" rather than gloss enamel (Wikipedia, "Toy soldier").

import Foundation
import simd

// MARK: - the refractive index of PVC

/// PVC's refractive index across the visible, measured on a commercial PVC
/// ("Manufacturer: Jubang plastic material, China") at room temperature:
/// X. Zhang, J. Qiu, X. Li, J. Zhao & L. Liu, "Complex refractive indices
/// measurements of polymers in visible and near-infrared bands", *Appl Opt*
/// 59:2337–2344 (2020), doi 10.1364/AO.383831. Rows copied from the
/// refractiveindex.info database file
/// `data/organic/(C2H3Cl)n - polyvinyl chloride/nk/Zhang.yml` (CC0), which
/// cites that paper; every 20 nm of its 10 nm table from 400 to 780 nm. The
/// extinction coefficient k is 1e-6 there, so PVC is a clear dielectric: its
/// colour in a toy comes from pigment, its gloss from n alone.
let pvcZhang: [(micrometres: Double, n: Double)] = [
    (0.40, 1.56135), (0.42, 1.55812), (0.44, 1.55491), (0.46, 1.55236), (0.48, 1.55010),
    (0.50, 1.54850), (0.52, 1.54692), (0.54, 1.54533), (0.55, 1.54493), (0.56, 1.54389),
    (0.58, 1.54275), (0.60, 1.54137), (0.62, 1.54073), (0.64, 1.53987), (0.66, 1.53946),
    (0.68, 1.53812), (0.70, 1.53754), (0.72, 1.53732), (0.74, 1.53593), (0.76, 1.53569),
    (0.78, 1.53526),
]

/// Two other clear plastics from the same paper and database, for
/// comparison only: PMMA — acrylic, the binder of acrylic paint — at 550 nm
/// (`poly(methyl methacrylate)/nk/Zhang-Mitsubishi.yml`) and polystyrene
/// (`polystyrene/nk/Zhang.yml`).
let pmmaIndex550: Double = 1.49463
let polystyreneIndex550: Double = 1.59338

/// n at a wavelength, linear between rows. MODEL: the rows are 10–20 nm
/// apart and n changes by under 0.002 between them.
func pvcIndex(nanometres: Double) -> Double {
    let um: Double = nanometres / 1000
    if um <= pvcZhang[0].micrometres { return pvcZhang[0].n }
    for i in 1..<pvcZhang.count where pvcZhang[i].micrometres >= um {
        let a = pvcZhang[i - 1]
        let b = pvcZhang[i]
        let t: Double = (um - a.micrometres) / (b.micrometres - a.micrometres)
        return a.n + (b.n - a.n) * t
    }
    return pvcZhang[pvcZhang.count - 1].n
}

/// The three wavelengths the render's red, green and blue channels stand for
/// when it asks for n. MODEL: roughly the dominant wavelengths of sRGB's
/// primaries; PVC's dispersion across them changes F0 by only 0.3 points.
let channelNanometres = SIMD3<Double>(610, 550, 465)

// MARK: - Fresnel for a dielectric

/// Reflectance at normal incidence from air into a medium of index n:
/// ((n − 1)/(n + 1))², as step 20's `fresnelF0` computes it for enamel.
func fresnelF0(_ n1: Double, _ n2: Double) -> Double {
    let r: Double = (n1 - n2) / (n1 + n2)
    return r * r
}

/// Unpolarised reflectance of a smooth interface from air into index n at an
/// angle whose cosine is `c`: the mean of the s and p Fresnel reflectances.
/// At c = 1 it is `fresnelF0(1, n)`; at c = 0, 1.
func fresnelDielectric(n: Double, cosTheta c0: Double) -> Double {
    let c: Double = min(max(c0, 0), 1)
    let s2: Double = 1 - c * c
    let t2: Double = 1 - s2 / (n * n)
    if t2 <= 0 { return 1 }
    let ct: Double = t2.squareRoot()
    let rs: Double = (c - n * ct) / (c + n * ct)
    let rp: Double = (n * c - ct) / (n * c + ct)
    return (rs * rs + rp * rp) / 2
}

// MARK: - what to break

/// Ways to break the step on purpose; each must make the suite fail. The
/// shared file lists every toy step's mutants; each Makefile runs its own.
enum Mutant: String {
    case none
    case pairedPlates      // Stegosaurus: plates side by side in pairs, not alternating
    case wholeHoof         // pig, cow: one toe per foot instead of two
    case dielectricZero    // F0 = 0: a surface that mirrors nothing
    case noSeam            // the mould's parting line missing
    case slidingFeet = "sliding_feet"   // a planted foot rides along with the body
    case rewind            // the loop closes by playing backwards
    case frozen            // the toy never comes to life
    case wrongGait         // a trot: diagonal pairs together, where a walk is sourced
}

let activeMutant: Mutant = Mutant(rawValue: ProcessInfo.processInfo.environment["TOY_MUTANT"] ?? "") ?? .none

/// The index the render's surface reflects with: PVC's, per channel.
///
/// The surface seen is the paint film's top, and its reflectance is set by
/// the paint's binder. Paint for PVC toys is taken to be vinyl-based, with
/// the plastic's own index — MODEL, no source reached for what binder toy
/// paints use. If it were acrylic instead, n = 1.495 (PMMA, Zhang 2020)
/// would give F0 = 3.9% against PVC's 4.6%: the same gloss to the eye.
func surfaceIndex(_ mutant: Mutant = activeMutant) -> SIMD3<Double> {
    SIMD3<Double>(pvcIndex(nanometres: channelNanometres.x), pvcIndex(nanometres: channelNanometres.y),
                  pvcIndex(nanometres: channelNanometres.z))
}

/// F0 per channel, as the kernel uses it. The `dielectricZero` mutant sets it
/// to nothing.
func surfaceF0(_ mutant: Mutant = activeMutant) -> SIMD3<Double> {
    if mutant == .dielectricZero { return SIMD3<Double>(0, 0, 0) }
    let n: SIMD3<Double> = surfaceIndex(mutant)
    return SIMD3<Double>(fresnelF0(1, n.x), fresnelF0(1, n.y), fresnelF0(1, n.z))
}

/// How many entries the kernel's Fresnel table has, evenly spaced in cos θ.
let fresnelTableSize: Int = 65

/// The table the kernel reads: the exact dielectric Fresnel reflectance of
/// the surface against cos θ, per channel.
func fresnelTable(_ mutant: Mutant = activeMutant) -> [SIMD3<Float>] {
    let n: SIMD3<Double> = surfaceIndex(mutant)
    return (0..<fresnelTableSize).map { i in
        let c: Double = Double(i) / Double(fresnelTableSize - 1)
        if mutant == .dielectricZero { return SIMD3<Float>(0, 0, 0) }
        let r = Float(fresnelDielectric(n: n.x, cosTheta: c))
        let g = Float(fresnelDielectric(n: n.y, cosTheta: c))
        let b = Float(fresnelDielectric(n: n.z, cosTheta: c))
        return SIMD3<Float>(r, g, b)
    }
}

// MARK: - paint

/// One paint: its diffuse colour (linear sRGB albedo) and how rough its
/// surface is (GGX α). Every colour here is MODEL: plausible for mass-made
/// painted figures of this class, not measured off any product. Roughness
/// is MODEL too: "matte-ish" paint is a broad, dim sheen (α ≈ 0.3–0.45), the
/// glossy black of painted eyes a tighter one.
struct Paint {
    var name: String
    var albedo: SIMD3<Float>
    var roughness: Float
}

/// A paint from sRGB bytes, so the table below reads like a paint chart.
func paint(_ name: String, _ r: Int, _ g: Int, _ b: Int, rough: Float) -> Paint {
    func lin(_ v: Int) -> Float { Float(srgbByteToLinear(UInt8(v))) }
    return Paint(name: name, albedo: SIMD3<Float>(lin(r), lin(g), lin(b)), roughness: rough)
}

/// The bare plastic as moulded, for the inset's cut face: PVC coloured in the
/// mass, a little paler than the paint over it. MODEL.
let bareRoughness: Float = 0.25

/// Paint film thickness. MODEL: no measured figure for toy paint was
/// reached; 25 µm is a typical single brushed or sprayed coat of paint.
/// UNVERIFIED — flagged in the caption as a model value.
let paintMicrometres: Float = 25

// MARK: - tone and colour, as in steps 20 and 53

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

/// Linear sRGB to CIELAB (D65 white), copied from step 53.
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

/// The tone curve's gain, step 20's 1.12.
let toneGain: Double = 1.12
