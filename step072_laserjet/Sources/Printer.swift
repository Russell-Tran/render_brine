// The printer: an HP LaserJet Professional P1102w, built to HP's published
// size, with the parts HP's own user guide names — the input tray (open, as
// HP draws it), the priority input slot, the output bin with its foldable
// extension, the print-cartridge door and its lift-tab, the power button,
// and the control panel's two buttons and three lights — in the wireless
// model's black. A neutral product render: no HP artwork, the badge plain.
//
// Sources, every one read for this step:
//   [HP-UG]  HP, "HP LaserJet Professional P1100 Printer series — User
//            Guide" (ENWW), read as mirrored page by page at manualsdir.com
//            (manual 398415; HP's own hosting would not load):
//            • Table C-1, Physical specifications: "Product weight 5.3 kg
//              (11.6 lb); Product height 194 mm (7.6 in); Product depth
//              224 mm (8.8 in); Product width 347 mm (13.7 in)", with HP's
//              footnote "Values are based on preliminary data".
//            • "Product views", wireless model: output bin, foldable output
//              tray extension, priority input slot, main input tray, power
//              button, print-cartridge door lift-tab, control panel.
//            • "Control-panel layout", wireless model: wireless button,
//              wireless light, attention light, ready light, cancel button.
//            • "Product features": "FastRes 600 setting provides 600 dots
//              per inch (dpi) effective print quality. FastRes 1200 setting
//              provides 1,200 dpi effective print quality"; a 150-sheet
//              input tray and a 125-sheet output bin of 75 g/m² paper.
//            • "Tray and bin capacity": paper 60–163 g/m².
//   [icecat] manua.ls / Icecat product sheet for the P1102w: 349 × 238 ×
//            196 mm, 5300 g — a third-party figure, 2–14 mm off HP's; noted,
//            not used.
//   [Zhang]  X. Zhang et al., Appl. Opt. 59, 2337 (2020), polystyrene,
//            refractiveindex.info `polystyrene/nk/Zhang.yml`: n = 1.59338
//            at 550 nm (read for this step).
//
// Everything else about the shape — where each part sits, how big, how
// round — is MODEL, drawn to the proportions of HP's product-view figure.
// Millimetres; x across the printer, y up from the table, z towards the
// front.

import Foundation
import simd

enum Mutant: String {
    case none
    case wrongDPI           // the inset's dots at 300 dpi
    case wrongSize          // the printer 20% too big
    case tonerTooBig        // toner particles as big as a dot
}

let activeMutant: Mutant = Mutant(rawValue: ProcessInfo.processInfo.environment["PRINTER_MUTANT"] ?? "") ?? .none

// MARK: - HP's numbers [HP-UG]

let hpWidth: Float = 347
let hpDepth: Float = 224
let hpHeight: Float = 194
let hpWeightKilograms: Float = 5.3
/// FastRes 600: the resolution the inset shows. FastRes 1200 is HP's
/// enhancement to 1,200 dpi "effective"; not what is drawn.
let dotsPerInch: Float = 600
let inputTraySheets: Int = 150
let outputBinSheets: Int = 125
let paperGrams: Float = 75

/// The printer's size as drawn: HP's, or the mutant's 20% more.
func printerScale(_ m: Mutant = activeMutant) -> Float { m == .wrongSize ? 1.2 : 1 }
func drawnWidth(_ m: Mutant = activeMutant) -> Float { hpWidth * printerScale(m) }
func drawnDepth(_ m: Mutant = activeMutant) -> Float { hpDepth * printerScale(m) }
func drawnHeight(_ m: Mutant = activeMutant) -> Float { hpHeight * printerScale(m) }

// MARK: - the plastic

/// The housing's refractive index: polystyrene's [Zhang]. That the housing
/// is a styrenic plastic (HIPS or ABS, common for printer cases) is MODEL;
/// HP's material was not found.
let housingIndex: Double = 1.59338
/// Matte black: diffuse albedo and GGX roughness, both MODEL.
let housingAlbedo = SIMD3<Float>(0.030, 0.030, 0.032)
let housingRoughness: Float = 0.42
/// The control panel's buttons, a slightly lighter moulding. MODEL.
let buttonAlbedo = SIMD3<Float>(0.055, 0.055, 0.06)
let buttonRoughness: Float = 0.30
/// The plain badge: a satin grey disc where the logo would be. MODEL.
let badgeAlbedo = SIMD3<Float>(0.20, 0.20, 0.21)
/// Paper in the input tray. MODEL.
let paperAlbedo = SIMD3<Float>(0.80, 0.80, 0.78)
/// The lights: radiance of the lit wireless (blue) and ready (green)
/// lights; the attention light (amber) is off, as on a printer that is
/// ready. MODEL.
let wirelessLight = SIMD3<Float>(0.4, 1.6, 6.0)
let readyLight = SIMD3<Float>(0.8, 6.0, 0.8)
let attentionOff = SIMD3<Float>(0.10, 0.05, 0.01)

/// Dielectric Fresnel reflectance, unpolarised.
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

// MARK: - tone and colour, as in steps 20, 53 and 60

func srgbByteToLinear(_ v: UInt8) -> Double {
    let c: Double = Double(v) / 255
    return c <= 0.04045 ? c / 12.92 : pow((c + 0.055) / 1.055, 2.4)
}

func srgbEncode(_ v: Double) -> Double {
    let c: Double = min(max(v, 0), 1)
    return c <= 0.0031308 ? 12.92 * c : 1.055 * pow(c, 1.0 / 2.4) - 0.055
}

let toneGain: Double = 1.12
