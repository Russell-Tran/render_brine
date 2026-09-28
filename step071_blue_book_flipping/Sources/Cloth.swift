// The cloth's look. Book cloth is woven cotton [Bailey], and woven fibre
// does not shine like plastic: it has no mirror. What it has is sheen —
// light caught by the fibres that stand up from the weave, strongest when
// the cloth is seen or lit at a grazing angle, and a soft diffuse colour
// from the dyed fibres underneath.
//
// The sheen is Estevez & Kulla's microfacet sheen ("Production Friendly
// Microfacet Sheen BRDF", Sony Pictures Imageworks, SIGGRAPH 2017 course),
// read for this step: fibres modelled as microfacets whose normals crowd
// towards the grazing direction,
//     D(m) = (2 + 1/r) sin^(1/r) θm / (2π),
// with the shadowing Λ fitted as they give (Table 1), combined as
// f = F G D / (4 |ωo·N| |ωi·N|), and layered over the diffuse by scaling the
// diffuse by one minus the sheen's albedo (their §5). We leave out their
// non-physical terminator softening (§4). The roughness r is MODEL.
//
// F, the sheen's strength and colour, Estevez & Kulla leave to the artist
// (their "spec tint"): it stands for light scattered by fibres standing up
// from the weave, which passes through dyed fibre and comes out tinted. It
// is MODEL — no measurement of book cloth's sheen was reached — set pale
// blue, strong enough to read at grazing angles and no more.
//
// The weave also has grain: cotton yarn is uneven, so each thread is a
// little lighter or darker than its neighbours and varies slowly along its
// length. MODEL (±10% in albedo), but it is what lets cloth read as cloth
// at a distance where single threads are smaller than a pixel.
//
// (Step 70's plastic mutant and its GGX lobe are kept below so the kernel
// is step 70's; nothing in this step turns them on.)

import Foundation
import simd

/// Sheen roughness r ∈ (0, 1]. MODEL: a medium cotton cloth.
let sheenRoughness: Double = 0.40
/// The sheen's colour, F in the formula, linear sRGB. MODEL.
let sheenColour = SIMD3<Float>(0.07, 0.08, 0.13)
/// How much each thread's albedo differs from the next, ±. MODEL.
let threadGrain: Float = 0.10
/// The plastic mutant's GGX roughness.
let plasticAlpha: Double = 0.08

/// The dyed cloth's diffuse colour, linear sRGB. MODEL: a dark blue, one of
/// the standard shades [Bailey]; no dye measured.
let clothAlbedo = SIMD3<Float>(0.030, 0.052, 0.150)
/// The text block's paper, the edges seen at the head and fore-edge. MODEL.
let paperAlbedo = SIMD3<Float>(0.78, 0.76, 0.70)
/// The headband's two silk colours. MODEL.
let headbandAlbedoA = SIMD3<Float>(0.62, 0.58, 0.48)
let headbandAlbedoB = SIMD3<Float>(0.05, 0.07, 0.20)

/// Dielectric Fresnel reflectance, unpolarised, from air into index n.
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

// MARK: - Estevez & Kulla's sheen

func sheenD(cosThetaM c: Double, r: Double) -> Double {
    let s: Double = max(1 - c * c, 0).squareRoot()
    let inv: Double = 1 / r
    return (2 + inv) * pow(s, inv) / (2 * Double.pi)
}

/// Their fitted L(x) = a / (1 + b x^c) + d x + e, parameters interpolated
/// between r = 0 and r = 1 by (1 − r)² (their Table 1).
func sheenL(_ x: Double, r: Double) -> Double {
    let p0: [Double] = [25.3245, 3.32435, 0.16801, -1.27393, -4.85967]
    let p1: [Double] = [21.5473, 3.82987, 0.19823, -1.97760, -4.32054]
    let w: Double = (1 - r) * (1 - r)
    var p: [Double] = [0, 0, 0, 0, 0]
    for i in 0..<5 { p[i] = w * p0[i] + (1 - w) * p1[i] }
    return p[0] / (1 + p[1] * pow(x, p[2])) + p[3] * x + p[4]
}

/// Λ(θ): e^L(cos θ) below cos θ = 0.5, e^(2L(0.5) − L(1 − cos θ)) above.
func sheenLambda(cosTheta c: Double, r: Double) -> Double {
    if c < 0.5 { return exp(sheenL(c, r: r)) }
    let l5: Double = sheenL(0.5, r: r)
    return exp(2 * l5 - sheenL(1 - c, r: r))
}

/// The sheen BRDF for view v and light l about normal +z, per unit F (the
/// kernel multiplies by the sheen's colour).
func sheenBRDF(v: SIMD3<Double>, l: SIMD3<Double>, r: Double = sheenRoughness) -> Double {
    let cv: Double = v.z
    let cl: Double = l.z
    if cv <= 0 || cl <= 0 { return 0 }
    let h: SIMD3<Double> = simd_normalize(v + l)
    let d: Double = sheenD(cosThetaM: h.z, r: r)
    let g: Double = 1 / (1 + sheenLambda(cosTheta: cv, r: r) + sheenLambda(cosTheta: cl, r: r))
    return g * d / (4 * cv * cl)
}

// MARK: - the mutant's plastic

func ggxD(cosThetaM c: Double, alpha a: Double) -> Double {
    let a2: Double = a * a
    let t: Double = c * c * (a2 - 1) + 1
    return a2 / (Double.pi * t * t)
}

func ggxG1(_ c: Double, alpha a: Double) -> Double {
    let a2: Double = a * a
    return 2 * c / (c + (a2 + (1 - a2) * c * c).squareRoot())
}

func plasticBRDF(v: SIMD3<Double>, l: SIMD3<Double>) -> Double {
    let cv: Double = v.z
    let cl: Double = l.z
    if cv <= 0 || cl <= 0 { return 0 }
    let h: SIMD3<Double> = simd_normalize(v + l)
    let d: Double = ggxD(cosThetaM: h.z, alpha: plasticAlpha)
    let g: Double = ggxG1(cv, alpha: plasticAlpha) * ggxG1(cl, alpha: plasticAlpha)
    let f: Double = fresnelDielectric(n: 1.5, cosTheta: simd_dot(v, h))
    return f * g * d / (4 * cv * cl)
}

/// The top lobe the kernel uses: sheen, or the mutant's plastic.
func topBRDF(v: SIMD3<Double>, l: SIMD3<Double>, mutant: Mutant = activeMutant) -> Double {
    sheenBRDF(v: v, l: l)
}

/// The top lobe's directional albedo at cos θv — per unit F for the sheen —
/// the fraction of light it takes, so the diffuse underneath gets the rest
/// (Kelemen-style scaling, as Estevez & Kulla layer it). A 96 × 192
/// midpoint quadrature.
func topAlbedo(cosView cv: Double, mutant: Mutant = activeMutant) -> Double {
    let sv: Double = max(1 - cv * cv, 0).squareRoot()
    let v = SIMD3<Double>(sv, 0, cv)
    let nt: Int = 96
    let np: Int = 192
    var sum: Double = 0
    for i in 0..<nt {
        let th: Double = (Double(i) + 0.5) / Double(nt) * Double.pi / 2
        let ct: Double = cos(th)
        let st: Double = sin(th)
        let dth: Double = Double.pi / 2 / Double(nt)
        for j in 0..<np {
            let ph: Double = (Double(j) + 0.5) / Double(np) * 2 * Double.pi
            let l = SIMD3<Double>(st * cos(ph), st * sin(ph), ct)
            let dph: Double = 2 * Double.pi / Double(np)
            let w: Double = ct * st * dth * dph
            sum += topBRDF(v: v, l: l, mutant: mutant) * w
        }
    }
    return sum
}

/// F for the top lobe: the sheen's colour, or 1 for the plastic (whose
/// Fresnel is inside its BRDF).
func topColour(_ mutant: Mutant = activeMutant) -> SIMD3<Float> {
    sheenColour
}

/// The albedo table the kernel reads, evenly spaced in cos θv.
let albedoTableSize: Int = 33
func topAlbedoTable(_ mutant: Mutant = activeMutant) -> [Float] {
    (0..<albedoTableSize).map { i in
        let c: Double = max(Double(i) / Double(albedoTableSize - 1), 0.02)
        return Float(topAlbedo(cosView: c, mutant: mutant))
    }
}

// MARK: - tone and colour, as in steps 20, 53 and 60

func srgbByteToLinear(_ v: UInt8) -> Double {
    let c: Double = Double(v) / 255
    return c <= 0.04045 ? c / 12.92 : pow((c + 0.055) / 1.055, 2.4)
}

func srgbEncode(_ v: Double) -> Double {
    let c: Double = min(max(v, 0), 1)
    return c <= 0.0031308 ? 12.92 * c : 1.055 * pow(c, 1.0 / 2.4) - 0.055
}

/// The tone curve's gain, step 20's 1.12.
let toneGain: Double = 1.12
