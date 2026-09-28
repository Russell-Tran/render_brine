// The food: one piece of peeled, cut Golden Delicious apple lying on the
// card, its flesh wet with its own juice — in millimetres — and what is in
// that juice and that air, from sources on this cultivar.
//
// Every number carries where it came from, or says MODEL and why. Where a
// number is generic (not measured on Golden Delicious), it says so.

import Foundation
import simd

// MARK: - which apple

// One named cultivar for everything below: 'Golden Delicious' (Malus
// domestica Borkh.). Its sugars, its smell, its flesh colour and the fruit it
// is cut from are all from studies of Golden Delicious itself.

// The fruit. "each apple weighing approximately 150 g" — the Golden and Red
// Delicious apples whose headspace Ferreira, Ribeiro & Nunes sampled whole
// (*Molecules* 29: 2954, 2024, doi 10.3390/molecules29132954, section 4.10;
// read in full via Europe PMC, PMC11243091). And the North Carolina Extension
// Gardener Plant Toolbox lists 'Golden Delicious' as "Fruit Width: > 3
// inches" (plants.ces.ncsu.edu, read 2026-09-28): over 76 mm across.
let fruitMass: Float = 150            // g, approximately
let fruitWidthAtLeast: Float = 76.2   // mm (3 inches)

// The piece. MODEL: a chunk as a knife cuts one from a peeled apple, 14 mm
// along the face the ants drink at, 9 mm deep and 7 mm high — cut from the
// flesh between the skin and the core, which it fits inside easily (a test
// checks its diagonal against the fruit's 38 mm radius). No source sizes
// pieces of apple; the fruit bounds it.
let pieceLength: Float = 14
let pieceDepth: Float = 9
let pieceHeight: Float = 7
/// How rounded the knife-cut edges are. MODEL: a sharp cut, softened a
/// little so the edges catch the light.
let pieceRounding: Float = 0.15
/// The juice on the cut flesh: a thin film, the tip and the glossa touch its
/// surface. MODEL, UNVERIFIED, step 50's 20 µm: no thickness of the juice on
/// a cut apple face was found. It is part of the drawn surface: the surface
/// drawn IS the juice's.
let juiceFilm: Float = 0.020

// Peeled: no skin anywhere, so every face is cut flesh under juice. What
// happens to cut apple flesh over MINUTES — browning, as polyphenol oxidase
// meets the air — is left out: this loop shows two and a half seconds of
// real time (Motion.swift), and the flesh is drawn at one moment's colour.

/// The pieceSize mutant draws the piece at half its size.
func pieceScale(_ m: Mutant) -> Float { m == .pieceSize ? 0.5 : 1.0 }

// MARK: - where it lies

/// The face the ants drink at faces left and towards the camera: its outward
/// normal, in the card's plane, 28° round from −x towards +z. MODEL, for the
/// view: the ants come at it head-on and the camera sees them from the side.
let faceYaw: Float = 28 * Float.pi / 180
let faceNormal = SIMD3<Float>(-cos(faceYaw), 0, sin(faceYaw))
/// Along the face, in the card's plane (square to the normal).
let faceTangent = SIMD3<Float>(sin(faceYaw), 0, cos(faceYaw))
/// The middle of the face's bottom edge, on the card. MODEL: the origin.
let faceBase = SIMD3<Float>(0, 0, 0)

/// The piece as placed: a box with rounded edges. Its frame: `u` into the
/// piece from the drinking face (−normal), `v` along the face (tangent), y up.
struct ApplePiece {
    var centre: SIMD3<Float>     // world
    var half: SIMD3<Float>       // half extents along (u, v, y), juice included
    var rounding: Float
    var film: Float

    /// Local coordinates of a world point: (u, v, y) from the centre.
    func local(_ p: SIMD3<Float>) -> SIMD3<Float> {
        let d: SIMD3<Float> = p - centre
        return SIMD3<Float>(simd_dot(d, -faceNormal), simd_dot(d, faceTangent), d.y)
    }
}

func buildPiece(mutant: Mutant) -> ApplePiece {
    let k: Float = pieceScale(mutant)
    // The juice's surface on the drinking face is the plane through faceBase:
    // the pieceSize mutant keeps that plane and shrinks the piece behind it.
    let half = SIMD3<Float>(pieceDepth / 2, pieceLength / 2, pieceHeight / 2) * k
    let centre: SIMD3<Float> = faceBase - faceNormal * half.x + SIMD3<Float>(0, half.z, 0)
    return ApplePiece(centre: centre, half: half, rounding: pieceRounding * k, film: juiceFilm)
}

/// The piece as a distance: a rounded box, exact outside (Quílez). The
/// surface is the juice's.
func pieceSDF(_ p: SIMD3<Float>, _ m: ApplePiece) -> Float {
    let q: SIMD3<Float> = simd_abs(m.local(p)) - (m.half - SIMD3<Float>(repeating: m.rounding))
    let outside: Float = simd_length(simd_max(q, SIMD3<Float>(repeating: 0)))
    let inside: Float = min(max(q.x, max(q.y, q.z)), 0)
    return outside + inside - m.rounding
}

// MARK: - colour

/// CIELAB (D65) to linear sRGB, the standard formulas (step 50's).
func labToLinearSRGB(_ lab: SIMD3<Float>) -> SIMD3<Float> {
    let fy: Float = (lab.x + 16) / 116
    let fx: Float = fy + lab.y / 500
    let fz: Float = fy - lab.z / 200
    func finv(_ t: Float) -> Float {
        let delta: Float = 6.0 / 29.0
        if t > delta { return t * t * t }
        let slope: Float = 3 * delta * delta
        let shifted: Float = t - 4.0 / 29.0
        return slope * shifted
    }
    let X: Float = 0.95047 * finv(fx)
    let Y: Float = 1.0 * finv(fy)
    let Z: Float = 1.08883 * finv(fz)
    let r: Float = 3.2406 * X - 1.5372 * Y - 0.4986 * Z
    let g: Float = -0.9689 * X + 1.8758 * Y + 0.0415 * Z
    let b: Float = 0.0557 * X - 0.2040 * Y + 1.0570 * Z
    return simd_clamp(SIMD3<Float>(r, g, b), SIMD3<Float>(repeating: 0), SIMD3<Float>(repeating: 1))
}

// The flesh. Fresh-cut 'Golden Delicious' slices cut with a sharp knife
// (30 N) measured L* 81.5, a* 1.6, b* 25.4 (Incardona, Amodio, Derossi,
// Colelli & Giacalone, *Foods* 14: 636, 2025, doi 10.3390/foods14040636,
// table of treatment effects, read via PMC11854140). Those are means over
// their 15 days at 5 °C; a day-0 value was not tabulated, and that sharp-cut
// flesh browned least. Used as the flesh's colour at one moment; how it
// browns is not drawn (see above).
let fleshLab = SIMD3<Float>(81.5, 1.6, 25.4)
let fleshAlbedo: SIMD3<Float> = labToLinearSRGB(fleshLab)

// The flesh's texture. The same paper's micro-CT of the sharp-cut Golden
// Delicious gives a structure separation index of 280 ± 8.75 µm (and 14%
// porosity): the spacing of its air spaces between the cells. The flesh is
// drawn with a faint cellular mottle at that spacing. MODEL as drawn — a
// texture, not a cell-by-cell model.
let fleshCellSpacing: Float = 0.280   // mm

// MARK: - optics

/// Normal-incidence Fresnel reflectance between two media.
func fresnelF0(_ n1: Double, _ n2: Double) -> Float {
    let r: Double = (n1 - n2) / (n1 + n2)
    return Float(r * r)
}

/// Insect cuticle, n ≈ 1.56 (Leertouwer, Wilts & Stavenga, *Opt Express* 19:
/// 24061, 2011; step 50's).
let cuticleIndex: Double = 1.56
/// The juice's gloss is taken as water's, n = 1.333 — step 50's choice. Its
/// 13.9% soluble solids (below) raise that a little; not drawn. MODEL.
let juiceIndex: Double = 1.333

let cuticleF0: Float = fresnelF0(1.0, cuticleIndex)
let juiceF0: Float = fresnelF0(1.0, juiceIndex)

// MARK: - what the juice tastes of: the cultivar's own sugars

// Golden Delicious pulp at harvest, g/kg (Ferreira, Ribeiro & Nunes 2024,
// Table 2, freeze-dried pulp by HPAEC-PAD, n = 5): fructose 43.6, sucrose
// 17.1, glucose 9.1, sorbitol 0.9, total 70.7. Fructose is the most abundant
// sugar, 62% of the total. Total soluble solids 13.9% (their Table 1, by
// refractometer).
let gdFructose: Float = 43.6
let gdSucrose: Float = 17.1
let gdGlucose: Float = 9.1
let gdSorbitol: Float = 0.9
let gdTotalSugars: Float = 70.7
let gdSolubleSolids: Float = 13.9

// A second source agrees on the order: USDA FoodData Central SR Legacy
// 168202, "Apples, raw, golden delicious, with skin", g per 100 g: fructose
// 6.1, sucrose 2.07, glucose 1.87, total sugars 10.0 (retrieved from the FDC
// API, 2026-09-28). With skin, not peeled — the peel is a small part of the
// fruit — so only its order is used.
let usdaFructose: Float = 6.1
let usdaSucrose: Float = 2.07
let usdaGlucose: Float = 1.87

// So the taste inset shows FRUCTOSE entering the pore. Dissolved, fructose
// is mostly β-D-fructopyranose: at equilibrium at 20 °C in D2O, "the
// distribution of the β-pyranose, β-furanose, α-furanose, α-pyranose and the
// keto tautomers was found to be 68.23%, 22.35%, 6.24%, 2.67% and 0.50%"
// (Barclay, Ginic-Markovic, Johnston, Cooper & Petrovsky, *Carbohydr Res*
// 347: 136–141, 2012, abstract via Europe PMC). The inset draws that form.
// (Generic: a property of fructose, not of the apple.) Lasius niger workers
// accept fructose as they do sucrose and glucose: "Sucrose, fructose and
// glucose share a same potential to act as phagostimulants" (Detrain &
// Prieur, *J Insect Physiol* 64: 74–80, 2014, abstract via Europe PMC).
let fructoseBetaPyranoseShare: Float = 68.23

// MARK: - what the air smells of: the cut, and the cultivar

// Cutting makes the smell. "Six-carbon (C6-) volatiles, including the
// aldehydes cis-3-hexenal, trans-2-hexenal, and hexanal, as well as their
// corresponding alcohols, are produced from action of the lipoxygenase (LOX)
// pathway on substrates released by tissue disruption" (Contreras, "The
// lipoxygenase pathway in apple peel", PhD dissertation, Michigan State
// University, 2014, abstract; her fruit was 'Jonagold' — the mechanism is
// generic to apple).
//
// In Golden Delicious itself, juiced (its tissue disrupted, as a cut
// disrupts it), "The main VC found in Golden Delicious apples at harvest time
// were (in decreasing order) 2-hexenal, 2-methyl 1-butanol, hexanal, butyl
// acetate, 2-methyl 1-propanol, 2-methyl butyl acetate, cis 3-hexenal, and
// hexyl acetate" (Salas, González-Aguilar, Jacobo-Cuéllar, Espino, Sepúlveda,
// Guerrero & Olivas, *Rev Fitotec Mex* 39(2), 2016, SciELO). The whole fruit's
// headspace carries the esters: "2-Methylbutyl acetate, butyl acetate and
// hexyl acetate are high-impact esters regarding apple flavour" (Ferreira et
// al. 2024, of Golden and Red Delicious).
//
// The smell inset draws the cultivar's first two classes: 2-hexenal, the
// green note the cut releases (drawn as (E)-hex-2-enal, PubChem CID 5281168:
// Salas et al. name "2-hexenal" without its isomer; Contreras names the
// trans), and butyl acetate (CID 31272), its most abundant ester there.
let objectHasOdour: Bool = true
