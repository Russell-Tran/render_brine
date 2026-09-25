// What "invert sugar" actually means, done as arithmetic.
//
// Invertase does not invert anything at the anomeric centre. It is a RETAINING
// glycosidase: it breaks the bond to the fructose and remakes it, on opposite
// faces, twice, and the sugar comes out the same way round it went in. The
// mechanism module (Invertase.swift) measures that from coordinates.
//
// The inversion in the name is macroscopic, and it is this file. Sucrose turns
// the plane of polarised light to the right. Its two halves, separated, turn it
// to the LEFT, because fructose is nearly twice as strongly laevorotatory as
// glucose is dextrorotatory, so the sum comes out negative. That sign flip is
// what was actually observed, long before anyone knew there was an aspartate.
//
//     sucrose      [α]D  +66.5
//     D-glucose    [α]D  +52.7
//     D-fructose   [α]D  -92.4
//     the mixture  [α]D  -19.9   <- DERIVED BELOW, never written down
//
// Two things this file is careful about that a textbook usually is not.
//
// FIRST, specific rotation is defined per GRAM, not per mole, so a mixture's
// specific rotation is a MASS-weighted mean of its parts'. Equimolar glucose
// and fructose happens to also be equal-mass, since hydrolysing sucrose gives
// one of each and they have the same formula, so the mean is the plain average
// — but the code weights by mass anyway, because that is the definition and
// because the sucrose that has not reacted yet does not have the same mass.
//
// SECOND, and this is the part that gets dropped: hydrolysis ADDS A WATER.
// Sucrose is 342.30 g/mol and its two products are 180.16 each, so a gram of
// sucrose becomes 1.0526 g of invert sugar. The observed rotation is [α] times
// concentration, and the concentration goes up. So the fully inverted solution
// does not read -19.9/66.5 = -0.299 of what it started at; it reads -0.314.
// That is a 5% effect on the final number and it falls straight out of the
// molar masses if you do not shortcut past them.
//
// Specific rotations are quoted at the sodium D line, 589 nm, 20 °C, in water,
// which is why the polarimeter's lamp in the render is that colour — computed
// through the CIE 1931 machinery step 9a built, not picked off a swatch.

import Foundation

/// The mutation switch. `make mutants` sets INV_MUTATE to break exactly one
/// thing, and the suite has to go red for each. It lives in the production code
/// rather than in the tests on purpose: a mutation that only the tests can see
/// proves nothing about the tests.
///
///   hardcode  quote invert sugar's −19.9° instead of deriving it
///   decouple  let the beam's twist drift away from the molecules counted
///   invert    let deglycosylation keep the configuration instead of inverting
///   ester     leave the fructosyl where it was and still draw the bond
enum Mutation: String {
    case none, hardcode, decouple, invert, ester
}

let mutation: Mutation = {
    guard let name = ProcessInfo.processInfo.environment["INV_MUTATE"] else { return .none }
    return Mutation(rawValue: name) ?? .none
}()

/// What kind of claim a number is, the scale steps 10 and 13 settled on.
/// There is no evidence bar drawn on the frame here — step 13 dropped it,
/// because a bar saying the same thing for a hundred frames is furniture — so
/// this lives in the constants table and in the tests instead.
enum Evidence: String {
    case measured = "MEASURED"    // somebody measured it and published it
    case derived = "DERIVED"      // arithmetic this code does on measured numbers
    case model = "MODEL"          // a choice made here, to make the picture
}

/// Every number this step leans on, with where it came from. The tests walk
/// this table: nothing may be here without a source, and anything claiming to
/// be MEASURED must cite a year.
struct Constant {
    var name: String
    var value: Double
    var unit: String
    var evidence: Evidence
    var source: String
}

/// One dissolved sugar.
struct Sugar {
    var name: String
    var carbon: Int
    var hydrogen: Int
    var oxygen: Int
    /// g/mol, from the formula and the 2021 IUPAC atomic weights.
    var molarMass: Double
    /// [α]D²⁰ in degrees per (decimetre × g/mL).
    var specificRotation: Double
    var source: String
}

// Atomic weights, conventional values, IUPAC Commission on Isotopic
// Abundances and Atomic Weights, Pure Appl. Chem. 94:573 (2022).
let atomicMassCarbon = 12.011
let atomicMassHydrogen = 1.008
let atomicMassOxygen = 15.999

/// Molar mass straight from the formula, so the three sugars cannot disagree
/// with each other about what they are made of.
func molarMass(carbon: Int, hydrogen: Int, oxygen: Int) -> Double {
    let c: Double = Double(carbon) * atomicMassCarbon
    let h: Double = Double(hydrogen) * atomicMassHydrogen
    let o: Double = Double(oxygen) * atomicMassOxygen
    return c + h + o
}

// The sources: specific rotations as tabulated for the sodium D line at 20 °C
// in water. Haynes (ed.), CRC Handbook of Chemistry and Physics, 97th edition
// (2016), "Physical Constants of Organic Compounds"; the same three numbers
// appear in Lide's earlier editions and in Tucker & Schieberle unchanged.
let sucrose = Sugar(name: "sucrose", carbon: 12, hydrogen: 22, oxygen: 11,
                    molarMass: molarMass(carbon: 12, hydrogen: 22, oxygen: 11),
                    specificRotation: 66.5,
                    source: "CRC Handbook of Chemistry and Physics, 97th ed. (2016), [α]D20 in water")
let glucose = Sugar(name: "D-glucose", carbon: 6, hydrogen: 12, oxygen: 6,
                    molarMass: molarMass(carbon: 6, hydrogen: 12, oxygen: 6),
                    specificRotation: 52.7,
                    source: "CRC Handbook of Chemistry and Physics, 97th ed. (2016), equilibrium [α]D20")
let fructose = Sugar(name: "D-fructose", carbon: 6, hydrogen: 12, oxygen: 6,
                     molarMass: molarMass(carbon: 6, hydrogen: 12, oxygen: 6),
                     specificRotation: -92.4,
                     source: "CRC Handbook of Chemistry and Physics, 97th ed. (2016), equilibrium [α]D20")
let waterMolarMass = molarMass(carbon: 0, hydrogen: 2, oxygen: 1)

/// The sodium D line. Everything about specific rotation is defined at it, and
/// the polarimeter's beam in the render is the colour this wavelength makes.
/// (The D line is really a close doublet, D₂ 588.995 nm and D₁ 589.592 nm; the
/// single figure below is the usual mean and is what "the D line" means in the
/// definition of [α]D.)
let sodiumDLineNanometres: Double = 589.29

/// The saccharimetrist's "normal weight": 26.000 g of sucrose made up to
/// 100 mL reads exactly 100 on the International Sugar Scale in a 200 mm tube.
/// It is used here because it is the standard sucrose solution, so the numbers
/// the polarimeter shows are the numbers a real bench would show.
let normalWeightGramsPer100mL: Double = 26.000
let cellLengthDecimetres: Double = 2.000

// Aliases, so that inside `Composition` — where `sucrose` is a COUNT — the
// three sugars can still be named. The shadowing is not a nuisance to route
// around; it is the compiler noticing that "sucrose" means two things here.
private let sucrose_ = sucrose
private let glucose_ = glucose
private let fructose_ = fructose

/// How many molecules of each are dissolved. These are COUNTED off the
/// molecules the render actually draws in the cell — never set by hand — so the
/// number on the dial cannot drift away from the picture beside it.
struct Composition {
    var sucrose: Int
    var glucose: Int
    var fructose: Int

    var originalSucrose: Int { sucrose + glucose }
    var converted: Int { glucose }

    /// The fraction of the sucrose that has been hydrolysed.
    var fractionConverted: Double {
        let total: Int = originalSucrose
        if total == 0 { return 0 }
        return Double(converted) / Double(total)
    }

    /// Grams of each species per mole of ORIGINAL sucrose, which is what turns
    /// a count of molecules into a concentration below.
    func massPerOriginalMole(_ sugar: Sugar, count: Int) -> Double {
        let total: Int = originalSucrose
        if total == 0 { return 0 }
        let share: Double = Double(count) / Double(total)
        return share * sugar.molarMass
    }

    var totalMassPerOriginalMole: Double {
        let s: Double = massPerOriginalMole(sucrose_, count: sucrose)
        let g: Double = massPerOriginalMole(glucose_, count: glucose)
        let f: Double = massPerOriginalMole(fructose_, count: fructose)
        return s + g + f
    }
}

/// The specific rotation of the dissolved solute as a whole: the mass-weighted
/// mean of its parts'. For an equimolar glucose/fructose mixture with no
/// sucrose left this returns −19.85, which is where "invert sugar" gets its
/// tabulated number — and it is arrived at here, not typed in.
func specificRotation(of c: Composition) -> Double {
    // The mutation the whole first test exists for: quote the textbook number
    // instead of working it out. It looks right for the finished mixture and is
    // wrong for every moment before it, starting with the first frame.
    if mutation == .hardcode { return -19.9 }
    let mass: Double = c.totalMassPerOriginalMole
    if mass <= 0 { return 0 }
    let s: Double = c.massPerOriginalMole(sucrose_, count: c.sucrose) * sucrose_.specificRotation
    let g: Double = c.massPerOriginalMole(glucose_, count: c.glucose) * glucose_.specificRotation
    let f: Double = c.massPerOriginalMole(fructose_, count: c.fructose) * fructose_.specificRotation
    return (s + g + f) / mass
}

/// Grams of solute per millilitre, for a solution made up from
/// `normalWeightGramsPer100mL` of sucrose and then partly hydrolysed. This is
/// where the added water shows up: as the sucrose splits, the dissolved mass
/// rises by the mass of the water taken in.
func concentrationGramsPerMillilitre(_ c: Composition) -> Double {
    let startingConcentration: Double = normalWeightGramsPer100mL / 100.0
    let molarity: Double = startingConcentration / sucrose_.molarMass
    return molarity * c.totalMassPerOriginalMole
}

/// What the polarimeter reads, in degrees: α = [α] · l · c, with `l` in
/// decimetres and `c` in g/mL. `fill` is how much of the cell holds solution,
/// which matters only while it is being emptied and refilled.
func observedRotation(_ c: Composition, fill: Double = 1.0) -> Double {
    let clamped: Double = min(max(fill, 0), 1)
    let path: Double = cellLengthDecimetres * clamped
    return specificRotation(of: c) * path * concentrationGramsPerMillilitre(c)
}

/// The reading for a full cell of untouched sucrose, and for one fully inverted.
/// Both derived; neither is written down anywhere in this file.
func rotationAtRest(molecules: Int) -> (start: Double, end: Double) {
    let fresh = Composition(sucrose: molecules, glucose: 0, fructose: 0)
    let inverted = Composition(sucrose: 0, glucose: molecules, fructose: molecules)
    return (observedRotation(fresh), observedRotation(inverted))
}

/// Going the other way: the fraction converted that a reading implies. The
/// reading is linear in that fraction, so this is an exact inverse — which is
/// what lets the two panels be checked against each other rather than merely
/// drawn next to each other.
func fractionImplied(byRotation alpha: Double, molecules: Int) -> Double {
    let ends = rotationAtRest(molecules: molecules)
    let span: Double = ends.start - ends.end
    if abs(span) < 1e-12 { return 0 }
    return (ends.start - alpha) / span
}

/// The conversion at which the reading passes through zero — the moment the
/// solution stops being dextrorotatory. Derived, and it is NOT one half: it
/// takes about three quarters of the sucrose, because sucrose starts a long way
/// positive and the products only end a little way negative.
func zeroCrossingFraction(molecules: Int) -> Double {
    return fractionImplied(byRotation: 0, molecules: molecules)
}

// MARK: - Atom bookkeeping

/// Element counts, so "sucrose + water = glucose + fructose" can be checked
/// rather than asserted.
struct Formula: Equatable {
    var carbon: Int
    var hydrogen: Int
    var oxygen: Int

    static func + (a: Formula, b: Formula) -> Formula {
        Formula(carbon: a.carbon + b.carbon, hydrogen: a.hydrogen + b.hydrogen,
                oxygen: a.oxygen + b.oxygen)
    }

    var text: String {
        "C\(carbon)H\(hydrogen)O\(oxygen)"
    }

    /// Heavy atoms only — what a crystal structure can see, and what the render
    /// draws. Hydrogens are invisible at any X-ray resolution here.
    var heavyAtoms: Int { carbon + oxygen }
}

func formula(_ s: Sugar) -> Formula {
    Formula(carbon: s.carbon, hydrogen: s.hydrogen, oxygen: s.oxygen)
}

let waterFormula = Formula(carbon: 0, hydrogen: 2, oxygen: 1)

// MARK: - The table

func chemistryConstants(molecules: Int) -> [Constant] {
    let ends = rotationAtRest(molecules: molecules)
    let inverted = Composition(sucrose: 0, glucose: molecules, fructose: molecules)
    return [
        Constant(name: "sucrose [α]D20", value: sucrose_.specificRotation, unit: "deg·mL/(g·dm)",
                 evidence: .measured, source: sucrose_.source),
        Constant(name: "D-glucose [α]D20", value: glucose.specificRotation, unit: "deg·mL/(g·dm)",
                 evidence: .measured, source: glucose.source),
        Constant(name: "D-fructose [α]D20", value: fructose.specificRotation, unit: "deg·mL/(g·dm)",
                 evidence: .measured, source: fructose.source),
        Constant(name: "invert sugar [α]D20", value: specificRotation(of: inverted),
                 unit: "deg·mL/(g·dm)", evidence: .derived,
                 source: "mass-weighted mean of D-glucose and D-fructose, computed in Chemistry.swift"),
        Constant(name: "sucrose molar mass", value: sucrose_.molarMass, unit: "g/mol",
                 evidence: .derived,
                 source: "C12H22O11 with IUPAC 2021 atomic weights, Pure Appl. Chem. 94:573 (2022)"),
        Constant(name: "hexose molar mass", value: glucose.molarMass, unit: "g/mol",
                 evidence: .derived,
                 source: "C6H12O6 with IUPAC 2021 atomic weights, Pure Appl. Chem. 94:573 (2022)"),
        Constant(name: "sodium D line", value: sodiumDLineNanometres, unit: "nm",
                 evidence: .measured,
                 source: "the sodium doublet 588.995/589.592 nm, NIST Atomic Spectra Database (2024)"),
        Constant(name: "normal weight", value: normalWeightGramsPer100mL, unit: "g/100 mL",
                 evidence: .measured,
                 source: "International Sugar Scale normal weight, ICUMSA Methods Book (2015)"),
        Constant(name: "cell length", value: cellLengthDecimetres, unit: "dm",
                 evidence: .model,
                 source: "the standard 200 mm polarimeter tube, chosen here for the render"),
        Constant(name: "reading, fresh sucrose", value: ends.start, unit: "deg",
                 evidence: .derived,
                 source: "[α]·l·c for the normal weight, computed in Chemistry.swift"),
        Constant(name: "reading, fully inverted", value: ends.end, unit: "deg",
                 evidence: .derived,
                 source: "[α]·l·c after hydrolysis, including the mass of the water taken up"),
        Constant(name: "zero crossing", value: zeroCrossingFraction(molecules: molecules),
                 unit: "fraction converted", evidence: .derived,
                 source: "where the derived reading changes sign, computed in Chemistry.swift"),
    ]
}
