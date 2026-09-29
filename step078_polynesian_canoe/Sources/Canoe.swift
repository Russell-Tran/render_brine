// The canoe: a waʻa kaulua, the Hawaiian double-hulled voyaging canoe, built
// extremely simply to the proportions the Polynesian Voyaging Society
// published for its replica Hōkūleʻa (1975). A simplified depiction informed
// by that replica and by its designer's notes, not a portrait of any
// particular canoe.
//
// Sources, every one read for this step:
//   [PVS-plan] Polynesian Voyaging Society, "Hokule'a Plans" (drawing,
//            pvs.kcc.hawaii.edu/pics/hokuplan.gif, read as archived by the
//            Wayback Machine, 2009-09-18). Printed on it: "Length overall
//            62'4"; Length LWL 54'0"; Beam 17'6"; Draft 2'6"; Total sail area
//            540 sq. ft.; Displacement 25,000 lbs (fully loaded)". Its plan,
//            profile and midship section are what the shapes below are
//            measured from (MODEL where so marked).
//   [PVS-build] PVS, "The Building of the Hokule'a – 1973–75" (archived
//            2009): "two 62-foot hulls; eight ʻiako, or crossbeams, joining
//            the two hulls; pola, or decking ...; rails along the decking;
//            and two masts"; "The 8-ton Hokule'a can be loaded with about
//            11,000 pounds"; hulls "of plywood, fiberglass, and resin, and
//            the sails were made from canvas"; the rig: "a sail attached to
//            spar and boom plus a shorter mast on which the spar, boom and
//            sail are raised and lowered".
//   [Kane]   Herb Kawainui Kāne, "In Search of the Ancient Polynesian
//            Voyaging Canoe" (archive.hokulea.com): hulls "a rounded 'V' ...
//            with the sides swelling outward in convex curvature"; "Below the
//            waterline the curvature of all Polynesian hulls is convex, both
//            in length and in section"; two sails, "The foresail should be
//            the larger"; the arched crossbeam as a classical Hawaiian
//            feature (Hōkūleʻa has them); end pieces (manu) "rise higher at
//            the stern than at the bow"; "the simple triangular sail carried
//            on straight spars is no less efficient".
//   [PVS-parts] PVS, "Parts of the Hawaiian Canoe": waʻa kaulua, ʻiako,
//            pola, manu, kia, peʻa ihu / peʻa hope, hoe uli.
//   [Finney] Ben Finney, "Founding of PVS; Building Hōkūleʻa"
//            (archive.hokulea.com): "two hulls each 62 feet in length, eight
//            crossbeams, decking, rails and two masts".
//   [Nayar]  K. G. Nayar et al., Desalination 390, 1 (2016), as quoted by
//            Wikipedia, "Seawater": "At a temperature of 25 °C, the salinity
//            of 35 g/kg and 1 atm pressure, the density of seawater is
//            1023.6 kg/m3" (surface seawater 1020–1029).
//
// Metres; x along the canoe (bow +x), y up from the keel, z across
// (starboard +z). The hulls' keel at midships is y = 0.

import Foundation
import simd

enum Mutant: String {
    case none
    case floatsWrong        // the waterline placed by eye, 30% too deep
    case wrongSize          // the whole canoe 20% too big
    case singleHull         // one hull only
}

let activeMutant: Mutant = Mutant(rawValue: ProcessInfo.processInfo.environment["CANOE_MUTANT"] ?? "") ?? .none

// MARK: - units

let metresPerFoot: Double = 0.3048
let kilogramsPerPound: Double = 0.45359237

func feet(_ ft: Double, _ inches: Double = 0) -> Double { (ft + inches / 12) * metresPerFoot }

// MARK: - PVS's published numbers [PVS-plan]

let pvsLengthOverall: Double = feet(62, 4)        // 18.999 m
let pvsLengthWaterline: Double = feet(54, 0)      // 16.459 m
let pvsBeam: Double = feet(17, 6)                 // 5.334 m
let pvsDraft: Double = feet(2, 6)                 // 0.762 m
let pvsSailAreaSquareFeet: Double = 540
let pvsSailArea: Double = 540 * metresPerFoot * metresPerFoot   // 50.17 m²
let pvsDisplacementPounds: Double = 25_000
/// The mass the water must carry: PVS's fully loaded displacement.
let canoeMass: Double = pvsDisplacementPounds * kilogramsPerPound  // 11 339.8 kg
/// [PVS-build]: eight crossbeams, two hulls.
let crossbeamCount: Int = 8

/// [Nayar]: sea water at 25 °C, 35 g/kg — Hawaiian surface water is warm
/// and of ordinary salinity, so the tropical figure, not fresh water.
let seaWaterDensity: Double = 1023.6

// MARK: - the shapes, measured off PVS's drawing (MODEL)
//
// The midship section on [PVS-plan] is drawn to the same scale as the
// profile: its crossbeams span 136 px for PVS's 17 ft 6 in (7.77 px/ft), and
// at that scale its drawn draft is 19–20 px, 2 ft 6 in, as printed. Measured
// at that scale: each hull 31 px wide (1.22 m), their centres 88 px apart
// (3.45 m, so 4.67 m, 15.3 ft, over both hulls), 34 px from keel to gunwale
// (1.33 m), widest 22 px above the keel (0.86 m).

let hullHalfWidth: Float = 0.604
let hullCentreZ: Float = 1.726
let gunwaleHeight: Float = 1.334
let widestAboveKeel: Float = 0.935
/// Kāne's "rounded V": the section is where two circles overlap (a vesica),
/// widest 0.86 m up and pointed at the keel, its sides convex. The two
/// circles that give the measured width and height:
///   R − c = half-width, R² − c² = (height of the widest point)².
let sectionRadius: Float = {
    let sum: Float = widestAboveKeel * widestAboveKeel / hullHalfWidth
    return (sum + hullHalfWidth) / 2
}()
let sectionOffset: Float = sectionRadius - hullHalfWidth
/// Each hull's point, bow and stern: the manu carry the length out to
/// PVS's 62 ft 4 in overall. MODEL, from the plan view.
let hullHalfLength: Float = 9.30
/// In plan each hull is a long lens (two big circles), 1.22 m at midships
/// and pointed at ±9.30 m.
let planRadius: Float = {
    let sum: Float = hullHalfLength * hullHalfLength / hullHalfWidth
    return (sum + hullHalfWidth) / 2
}()
let planOffset: Float = planRadius - hullHalfWidth
/// Toward the ends the section narrows and sharpens — Kāne's hulls are
/// "deeper or had a greater amount of 'V' shape along the keel". The two
/// circles' axes splay outward from midships, so each section is the same
/// vesica with its circles further apart: narrower, shallower, sharper.
/// The splay is set by where the section would vanish, `sectionVanishes`
/// metres from midships.
///
/// Two numbers here are fitted, not measured: that splay and the keel's
/// rocker (below). Together they make the simplified hull reproduce the
/// two hydrostatic numbers PVS printed with its displacement — 54 ft on
/// the waterline and 2 ft 6 in of draft at 25,000 lb. A hull of constant
/// section, rocker alone setting the 54 ft, is 28% fuller than that (it
/// floats at 0.655 m). The draft the picture shows is not placed: it is
/// solved for by buoyancy every run (Buoyancy.swift), and the tests hold
/// ρV = M, the 54 ft and the 2 ft 6 in each to a stated tolerance.
let sectionVanishes: Float = 15.3
var sectionSplay: Float { hullHalfWidth / sectionVanishes }
/// In profile the keel is an arc — Kāne's "convex ... in length" — of
/// this radius (fitted, above).
let rockerRadius: Float = 44.8

// The manu: end pieces rising out of the hull ends, higher at the stern
// [Kane]. Each is a curved blade: an arc leaving the gunwale level with the
// deck and sweeping out and up to its tip. Tips measured off the profile
// (MODEL): the bow's 1.25 m, the stern's 2.4 m above the gunwale, both 2.2 m
// beyond where they rise, their tips making the length overall.
struct Manu {
    var tipOut: Float       // how far out along the canoe, from the start
    var tipUp: Float        // how far up
    var radius: Float {
        let out2: Float = tipOut * tipOut
        let up2: Float = tipUp * tipUp
        return (out2 + up2) / (2 * tipUp)
    }
    var sweep: Float { atan2(tipOut, radius - tipUp) }
    /// How far out the blade's outer edge reaches from where it starts: at
    /// its tip's outer corner, or, if it curls back past upright, at the
    /// arc's outermost point.
    var reach: Float {
        let outer: Float = radius + manuHalfThickness
        return sweep <= Float.pi / 2 ? outer * sin(sweep) : outer
    }
}
let bowManu = Manu(tipOut: 2.2, tipUp: 1.25)
let sternManu = Manu(tipOut: 2.2, tipUp: 2.4)
/// Half the blade's thickness in the profile, and half its width across. MODEL.
let manuHalfThickness: Float = 0.07
let manuHalfWidth: Float = 0.10
/// Where each manu starts: its outer edge reaches ± length overall / 2.
func manuStart(_ m: Manu) -> Float { Float(pvsLengthOverall / 2) - m.reach }

/// The eight ʻiako, measured off the plan view (MODEL positions), bow to
/// stern. Drawn straight, as round poles resting on the gunwales, reaching
/// PVS's 17 ft 6 in beam; Hōkūleʻa's are arched (Kāne), a detail left out.
let crossbeamX: [Float] = [5.48, 3.66, 1.84, 0.02, -1.74, -3.62, -5.35, -7.26]
let crossbeamRadius: Float = 0.10
let crossbeamHalfSpan: Float = Float(feet(17, 6) / 2)
/// The pola: a deck on the crossbeams, hull centre to hull centre. MODEL.
let deckTop: Float = gunwaleHeight + 2 * crossbeamRadius + 0.05
let deckHalfThickness: Float = 0.025

// MARK: - the rig

/// The masts (kia) stand on the pola over the first and fifth ʻiako, where
/// PVS's profile has them. MODEL.
let foremastX: Float = 5.48
let aftmastX: Float = -1.74
let mastRadius: Float = 0.10
let sparRadius: Float = 0.065
/// Each mast is shorter than its sail's luff spar ([PVS-build]'s "shorter
/// mast"): this fraction of the peak's height. MODEL.
let mastHeightFraction: Float = 0.62
/// How far each sail is swung off the centreline about its mast. MODEL.
let sailYaw: Float = 0.30

/// A crab-claw sail: the tack at the foot of the mast, a straight luff spar
/// up to the peak, a straight boom out to the clew [Kane: straight spars],
/// and the head between them cut concave — the crab claw. Peak and clew as
/// PVS's profile draws them, relative to the tack (MODEL); x forward, y up.
struct Sail {
    var mastX: Float
    var peak: SIMD2<Float>
    var clew: SIMD2<Float>
}
/// The head's concavity: its depth as a fraction of the head's chord. MODEL.
let headSag: Float = 0.22

let drawnForesail = Sail(mastX: foremastX, peak: SIMD2<Float>(-0.33, 11.59), clew: SIMD2<Float>(-5.36, 10.25))
let drawnAftsail = Sail(mastX: aftmastX, peak: SIMD2<Float>(-0.29, 9.29), clew: SIMD2<Float>(-5.08, 8.14))

/// Area of a sail: the triangle less the circular segment cut from its head.
func sailArea(_ s: Sail) -> Double {
    let p = SIMD2<Double>(Double(s.peak.x), Double(s.peak.y))
    let q = SIMD2<Double>(Double(s.clew.x), Double(s.clew.y))
    let cross: Double = p.x * q.y - p.y * q.x
    let triangle: Double = abs(cross) / 2
    let chord: Double = simd_distance(p, q)
    let sag: Double = Double(headSag) * chord
    let half: Double = chord / 2
    let top: Double = half * half + sag * sag
    let r: Double = top / (2 * sag)
    let theta: Double = 2 * asin(chord / (2 * r))
    let segment: Double = r * r * (theta - sin(theta)) / 2
    return triangle - segment
}

/// Both sails scaled together about their tacks so that they make PVS's
/// 540 sq ft; the foresail stays the larger [Kane].
let sailScale: Float = {
    let drawn: Double = sailArea(drawnForesail) + sailArea(drawnAftsail)
    return Float((pvsSailArea / drawn).squareRoot())
}()
let foresail = Sail(mastX: foremastX, peak: drawnForesail.peak * sailScale, clew: drawnForesail.clew * sailScale)
let aftsail = Sail(mastX: aftmastX, peak: drawnAftsail.peak * sailScale, clew: drawnAftsail.clew * sailScale)

/// The circle whose inside is cut from the sail's head: through peak and
/// clew, dipping `headSag` × chord below the chord. (centre, radius)
func headCircle(_ s: Sail) -> (SIMD2<Float>, Float) {
    let mid: SIMD2<Float> = (s.peak + s.clew) / 2
    let chord: Float = simd_distance(s.peak, s.clew)
    let sag: Float = headSag * chord
    let half: Float = chord / 2
    let top: Float = half * half + sag * sag
    let r: Float = top / (2 * sag)
    let along: SIMD2<Float> = simd_normalize(s.clew - s.peak)
    var up = SIMD2<Float>(-along.y, along.x)
    if up.y < 0 { up = -up }
    return (mid + up * (r - sag), r)
}

/// The hoe uli, the steering paddle, over the stern between the hulls,
/// its blade in the water. MODEL, after the profile.
let paddleTop = SIMD3<Float>(-7.6, deckTop + 0.9, 0)
let paddleBottom = SIMD3<Float>(-11.0, 0.30, 0)
let paddleShaftRadius: Float = 0.05
let paddleBladeLength: Float = 1.6
let paddleBladeHalfWidth: Float = 0.20
let paddleBladeHalfThickness: Float = 0.025

// MARK: - size, as drawn

/// The canoe as drawn: PVS's size, or the mutant's 20% more.
func canoeScale(_ m: Mutant = activeMutant) -> Float { m == .wrongSize ? 1.2 : 1 }

// MARK: - materials (MODEL)

/// The hulls: painted plywood and fibreglass [PVS-build]; a dark brown.
let hullAlbedo = SIMD3<Float>(0.055, 0.035, 0.025)
/// The manu, ʻiako, pola and spars: wood.
let woodAlbedo = SIMD3<Float>(0.36, 0.22, 0.12)
let deckAlbedo = SIMD3<Float>(0.45, 0.33, 0.20)
/// Canvas sails [PVS-build], weathered off-white.
let sailAlbedo = SIMD3<Float>(0.78, 0.72, 0.60)
/// How much of the sun reaches through the canvas to its shaded side.
let sailTranslucency: Float = 0.30

/// The water: a flat, calm sea, drawn see-through and without refraction
/// so that the hulls below the waterline show true to scale. Its
/// extinction and deep colour are MODEL, chosen so the keel is seen.
let waterIndex: Float = 1.34
let waterExtinction: Float = 0.30
let deepWater = SIMD3<Float>(0.008, 0.070, 0.110)

// MARK: - tone and colour, as in steps 20, 53, 60 and 72

func srgbEncode(_ v: Double) -> Double {
    let c: Double = min(max(v, 0), 1)
    return c <= 0.0031308 ? 12.92 * c : 1.055 * pow(c, 1.0 / 2.4) - 0.055
}

let toneGain: Double = 1.0
