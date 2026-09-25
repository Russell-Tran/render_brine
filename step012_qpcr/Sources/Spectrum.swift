// Turning a wavelength into a colour.
//
// This is the new thing in step 9a. Every colour in this project so far was
// CHOSEN: CPK conventions for elements, palettes picked by eye, a green for
// GFP's chromophore that looked about right. Here the two proteins' colours
// are DERIVED, from the light they actually emit:
//
//     wavelength -> CIE XYZ -> linear sRGB -> gamma -> the pixel
//
// The first step uses the CIE 1931 colour-matching functions, which are the
// measured average response of the three kinds of cone in the human eye. They
// are normally a table; the analytic fits used here are from Wyman, Sloan &
// Shirley, "Simple Analytic Approximations to the CIE XYZ Color Matching
// Functions", Journal of Computer Graphics Techniques 2(2):1 (2013) - a few
// piecewise Gaussians, accurate through the middle of the visible range.
//
// A CAVEAT WORTH STATING, because it changes what the render can claim: no
// screen can show a single wavelength. The sRGB gamut is a triangle inside the
// horseshoe of real colours, and every pure spectral colour lies outside it.
// The conversion below therefore produces NEGATIVE values for the primaries
// that would have to be subtracted, and those get clipped to zero. Both of
// these proteins clip. That is the honest reason a photograph of a fluorescent
// protein never looks as vivid as the eye reports.

import Foundation
import simd

/// Planck's constant times the speed of light, in electron-volt nanometres, so
/// that photon energy in eV is just HC / (wavelength in nm).
let planckTimesC: Double = 1239.84

/// The energy of one photon of this wavelength, in electron volts.
func photonEnergy(nanometres: Double) -> Double {
    return planckTimesC / nanometres
}

/// A lopsided Gaussian: different widths below and above the peak. This is the
/// shape Wyman et al. fit the colour-matching functions with.
private func lobe(_ x: Float, _ peak: Float, _ widthBelow: Float, _ widthAbove: Float) -> Float {
    let width: Float = x < peak ? widthBelow : widthAbove
    let t: Float = (x - peak) / width
    let e: Float = -0.5 * t * t
    return exp(e)
}

/// The CIE 1931 colour-matching functions at one wavelength: how strongly this
/// light drives each of the eye's three responses.
func colourMatching(nanometres: Float) -> SIMD3<Float> {
    let l: Float = nanometres
    // Written as separate typed terms rather than one long expression: the
    // laptop's older Swift type checker is slow on chained Float arithmetic.
    let x1: Float = 1.056 * lobe(l, 599.8, 37.9, 31.0)
    let x2: Float = 0.362 * lobe(l, 442.0, 16.0, 26.7)
    let x3: Float = 0.065 * lobe(l, 501.1, 20.4, 26.2)
    let y1: Float = 0.821 * lobe(l, 568.8, 46.9, 40.5)
    let y2: Float = 0.286 * lobe(l, 530.9, 16.3, 31.1)
    let z1: Float = 1.217 * lobe(l, 437.0, 11.8, 36.0)
    let z2: Float = 0.681 * lobe(l, 459.0, 26.0, 13.8)
    return SIMD3(x1 + x2 - x3, y1 + y2, z1 + z2)
}

/// CIE XYZ to linear sRGB, the standard D65 matrix. Values outside 0...1 mean
/// the colour is outside what the display can make.
func linearSRGB(fromXYZ xyz: SIMD3<Float>) -> SIMD3<Float> {
    let x: Float = xyz.x, y: Float = xyz.y, z: Float = xyz.z
    let r: Float = 3.2406 * x - 1.5372 * y - 0.4986 * z
    let g: Float = -0.9689 * x + 1.8758 * y + 0.0415 * z
    let b: Float = 0.0557 * x - 0.2040 * y + 1.0570 * z
    return SIMD3(r, g, b)
}

/// The sRGB transfer curve, linear light to stored value.
func gammaEncode(_ c: Float) -> Float {
    let v: Float = min(max(c, 0), 1)
    if v <= 0.0031308 { return 12.92 * v }
    let p: Float = pow(v, 1.0 / 2.4)
    return 1.055 * p - 0.055
}

/// What a single wavelength looks like, as a colour the renderer can use.
///
/// Returns linear values, because the renderer shades in linear light. The
/// result is normalised so its brightest channel is 1: a monochromatic colour
/// carries no meaningful brightness of its own, and what is wanted here is the
/// most saturated version of that hue the display can manage.
///
/// `clipped` reports whether any primary had to be clamped up from negative,
/// which is to say whether this colour is outside the sRGB gamut. For pure
/// spectral light it always is.
func spectralColour(nanometres: Float) -> (linear: SIMD3<Float>, clipped: Bool) {
    let xyz: SIMD3<Float> = colourMatching(nanometres: nanometres)
    let raw: SIMD3<Float> = linearSRGB(fromXYZ: xyz)
    let clipped: Bool = raw.min() < -1e-4
    var v: SIMD3<Float> = simd_max(raw, SIMD3<Float>(repeating: 0))
    let peak: Float = v.max()
    if peak > 1e-9 { v /= peak }
    return (v, clipped)
}

/// The same colour as 8-bit sRGB, for captions and for the GIF palette.
func spectralRGB8(nanometres: Float) -> (r: Int, g: Int, b: Int) {
    let c: SIMD3<Float> = spectralColour(nanometres: nanometres).linear
    let r: Float = gammaEncode(c.x) * 255
    let g: Float = gammaEncode(c.y) * 255
    let b: Float = gammaEncode(c.z) * 255
    return (Int(r.rounded()), Int(g.rounded()), Int(b.rounded()))
}

/// `#rrggbb`, for the caption.
func hexString(nanometres: Float) -> String {
    let c = spectralRGB8(nanometres: nanometres)
    return String(format: "#%02x%02x%02x", c.r, c.g, c.b)
}
