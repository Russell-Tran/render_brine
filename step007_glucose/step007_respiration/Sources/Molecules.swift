// Glucose burning in oxygen, one molecule of each:
//
//     C₆H₁₂O₆ + 6 O₂  →  6 CO₂ + 6 H₂O + energy
//
// Every one of the 36 atoms (6 C, 12 H, 18 O) is tracked from where it sits in
// the reactants to where it ends up in the products. This shows the overall
// accounting: which atoms end up where and where the electrons go. In a cell
// the same change happens through about 30 enzyme steps (glycolysis, the Krebs
// cycle, the electron transport chain), which aren't shown.
//
// Geometry sources:
//   β-D-glucose: PubChem CID 64689, 3D conformer (computed, MMFF94s).
//   O₂: O–O 1.21 Å (Wikipedia, "Triplet oxygen", experimental).
//   CO₂: C=O 1.163 Å, linear (Wikipedia, "Carbon dioxide").
//   H₂O: O–H 0.9572 Å, H–O–H 104.52° (gas-phase values).
//
// Distances are in ångströms (1 Å = 0.1 nm).

import Foundation
import simd

enum Element: Equatable {
    case carbon, oxygen, hydrogen

    /// Ball radius for a ball-and-stick picture (Å), not the atom's true size.
    var ballRadius: Float {
        switch self {
        case .carbon: return 0.34
        case .oxygen: return 0.32
        case .hydrogen: return 0.22
        }
    }
    /// Standard CPK colors: carbon dark grey, oxygen red, hydrogen white.
    var color: SIMD3<Float> {
        switch self {
        case .carbon: return SIMD3(0.28, 0.29, 0.31)
        case .oxygen: return SIMD3(0.86, 0.16, 0.14)
        case .hydrogen: return SIMD3(0.93, 0.93, 0.93)
        }
    }
    /// Pauling electronegativity, used to assign oxidation states.
    var electronegativity: Float {
        switch self {
        case .carbon: return 2.55
        case .oxygen: return 3.44
        case .hydrogen: return 2.20
        }
    }
}

struct Atom {
    var element: Element
    var position: SIMD3<Float>
    /// A soft glow behind the atom: positive = gold (losing electrons),
    /// negative = cyan (gaining electrons), 0 = none.
    var glow: Float = 0
}

struct Bond: Equatable {
    var a: Int
    var b: Int
    var order: Float
}

struct MoleculeState {
    var atoms: [Atom]
    var bonds: [Bond]
}

func radians(_ degrees: Float) -> Float { degrees * .pi / 180 }
func mix(_ a: Float, _ b: Float, _ t: Float) -> Float { a + (b - a) * t }
func smoothstep01(_ t: Float) -> Float {
    let x = min(max(t, 0), 1)
    return x * x * (3 - 2 * x)
}

// MARK: - Measured sizes

let oxygenBondLength: Float = 1.21        // O=O
let carbonDioxideBondLength: Float = 1.163 // C=O in CO₂
let waterBondLength: Float = 0.9572       // O–H in H₂O
let waterAngle: Float = 104.52            // H–O–H

// MARK: - β-D-glucose from PubChem

/// PubChem CID 64689, 3D conformer. Atoms in PubChem's order: 6 O, 6 C, 12 H.
let glucoseAtoms: [(Element, SIMD3<Float>)] = [
    (.oxygen, SIMD3(-0.6679, 1.1587, 0.2570)),    // 0 ring O
    (.oxygen, SIMD3(-0.8870, -2.4483, -0.3388)),  // 1
    (.oxygen, SIMD3(1.8623, -2.0693, 0.4696)),    // 2
    (.oxygen, SIMD3(2.8609, 0.5414, -0.4619)),    // 3
    (.oxygen, SIMD3(1.1222, 2.6552, 0.2574)),     // 4 on the anomeric carbon
    (.oxygen, SIMD3(-3.3742, 0.9717, -0.1865)),   // 5 on CH₂OH
    (.carbon, SIMD3(-0.3727, -1.2470, 0.2300)),   // 6
    (.carbon, SIMD3(1.0856, -1.0709, -0.1940)),   // 7
    (.carbon, SIMD3(-1.2211, -0.0621, -0.2375)),  // 8 ring carbon next to the ring O
    (.carbon, SIMD3(1.6082, 0.3151, 0.1839)),     // 9
    (.carbon, SIMD3(0.6388, 1.4132, -0.2534)),    // 10 anomeric carbon (C1)
    (.carbon, SIMD3(-2.6550, -0.1577, 0.2740)),   // 11 CH₂OH carbon (C6)
    (.hydrogen, SIMD3(-0.4248, -1.3522, 1.3206)),
    (.hydrogen, SIMD3(1.2066, -1.2487, -1.2697)),
    (.hydrogen, SIMD3(-1.2548, -0.0098, -1.3343)),
    (.hydrogen, SIMD3(1.7952, 0.3598, 1.2636)),
    (.hydrogen, SIMD3(0.5967, 1.5141, -1.3440)),
    (.hydrogen, SIMD3(-2.6916, -0.1535, 1.3685)),
    (.hydrogen, SIMD3(-3.1564, -1.0581, -0.0922)),
    (.hydrogen, SIMD3(-0.8514, -2.3615, -1.3066)),
    (.hydrogen, SIMD3(1.4973, -2.9356, 0.2200)),
    (.hydrogen, SIMD3(2.7165, 0.4989, -1.4227)),
    (.hydrogen, SIMD3(1.4876, 2.5033, 1.1448)),
    (.hydrogen, SIMD3(-2.9192, 1.7652, 0.1440)),
]

/// PubChem's bond table (all single bonds), converted to 0-based indices.
let glucoseBondPairs: [(Int, Int)] = [
    (0, 8), (0, 10), (1, 6), (1, 19), (2, 7), (2, 20), (3, 9), (3, 21), (4, 10), (4, 22),
    (5, 11), (5, 23), (6, 7), (6, 8), (6, 12), (7, 9), (7, 13), (8, 11), (8, 14), (9, 10),
    (9, 15), (10, 16), (11, 17), (11, 18),
]

/// The six-membered ring: O, then five carbons.
let glucoseRing = [0, 8, 6, 7, 9, 10]

// MARK: - Building the reaction

/// A reaction: every atom's start and end, plus the bonds of each side.
struct Reaction {
    var elements: [Element]
    var start: [SIMD3<Float>]
    var end: [SIMD3<Float>]
    var reactantBonds: [Bond]
    var productBonds: [Bond]
    var carbons: [Int]           // glucose carbons, in glucose order
    var o2Oxygens: [Int]         // the 12 oxygens that arrive as O₂
}

/// Points around an ellipse in the picture plane.
func ellipsePoint(_ degrees: Float, rx: Float, ry: Float) -> SIMD3<Float> {
    let a = radians(degrees)
    return SIMD3(rx * cos(a), ry * sin(a), 0)
}

/// Rotates `v` about `axis` (unit) by `degrees`.
func rotate(_ v: SIMD3<Float>, about axis: SIMD3<Float>, by degrees: Float) -> SIMD3<Float> {
    let a = radians(degrees)
    let c = cos(a), s = sin(a)
    let term1 = v * c
    let term2 = simd_cross(axis, v) * s
    let term3 = axis * (simd_dot(axis, v) * (1 - c))
    return term1 + term2 + term3
}

func buildReaction() -> Reaction {
    var elements: [Element] = []
    var start: [SIMD3<Float>] = []

    // Glucose, centered on its heavy atoms.
    var center = SIMD3<Float>(0, 0, 0)
    for (e, p) in glucoseAtoms where e != .hydrogen { center += p }
    center /= 12
    for (e, p) in glucoseAtoms {
        elements.append(e)
        start.append(p - center)
    }
    var reactantBonds = glucoseBondPairs.map { Bond(a: $0.0, b: $0.1, order: 1) }

    // Six O₂ around it (atoms 24...35), each tilted a little out of the plane.
    var o2Oxygens: [Int] = []
    for j in 0..<6 {
        let angle = Float(60 * j + 30)
        let c = ellipsePoint(angle, rx: 9.0, ry: 5.4)
        let tangent = simd_normalize(SIMD3<Float>(-sin(radians(angle)), cos(radians(angle)), 0))
        let tilt: Float = j % 2 == 0 ? 0.5 : -0.5
        let axis = simd_normalize(tangent + SIMD3<Float>(0, 0, tilt))
        let half: Float = oxygenBondLength / 2
        let first = elements.count
        elements.append(.oxygen); start.append(c - axis * half)
        elements.append(.oxygen); start.append(c + axis * half)
        reactantBonds.append(Bond(a: first, b: first + 1, order: 2))
        o2Oxygens.append(contentsOf: [first, first + 1])
    }

    var end = start
    var productBonds: [Bond] = []

    // Six CO₂ on an outer ring. Each glucose carbon keeps one of its own
    // oxygens and takes one oxygen from an O₂. Carbons are matched to CO₂
    // slots in order of angle, so each travels the short way out.
    let carbons = [6, 7, 8, 9, 10, 11]
    let ownOxygen: [Int: Int] = [6: 1, 7: 2, 8: 0, 9: 3, 10: 4, 11: 5]
    let carbonsByAngle = carbons.sorted { atan2(start[$0].y, start[$0].x) < atan2(start[$1].y, start[$1].x) }
    // Rotate the slot numbering so the first carbon lands on the nearest slot.
    let firstAngle: Float = atan2(start[carbonsByAngle[0]].y, start[carbonsByAngle[0]].x) * 180 / .pi
    let slotOffset = Int(((firstAngle + 360).truncatingRemainder(dividingBy: 360) / 60).rounded()) % 6
    for (k, c) in carbonsByAngle.enumerated() {
        let slot = (k + slotOffset) % 6
        let angle = Float(60 * slot)
        let site = ellipsePoint(angle, rx: 11.0, ry: 6.3)
        let tangent = simd_normalize(SIMD3<Float>(-sin(radians(angle)), cos(radians(angle)), 0))
        // The O₂ whose first oxygen joins this CO₂: the one just ahead of it.
        let fromO2 = o2Oxygens[2 * slot]
        end[c] = site
        end[ownOxygen[c]!] = site - tangent * carbonDioxideBondLength
        end[fromO2] = site + tangent * carbonDioxideBondLength
        productBonds.append(Bond(a: c, b: ownOxygen[c]!, order: 2))
        productBonds.append(Bond(a: c, b: fromO2, order: 2))
    }

    // Six H₂O on an inner ring, each made from the other oxygen of an O₂ and
    // two of glucose's hydrogens (the nearest ones not yet used).
    var unusedHydrogens = Array(12..<24)
    for j in 0..<6 {
        let angle = Float(60 * j + 30)
        let site = ellipsePoint(angle, rx: 5.6, ry: 3.3)
        let outward = simd_normalize(SIMD3<Float>(cos(radians(angle)), sin(radians(angle)), 0))
        let oxygen = o2Oxygens[2 * j + 1]
        end[oxygen] = site
        unusedHydrogens.sort { simd_distance(start[$0], site) < simd_distance(start[$1], site) }
        let h1 = unusedHydrogens.removeFirst()
        let h2 = unusedHydrogens.removeFirst()
        let half: Float = waterAngle / 2
        end[h1] = site + rotate(outward, about: SIMD3(0, 0, 1), by: half) * waterBondLength
        end[h2] = site + rotate(outward, about: SIMD3(0, 0, 1), by: -half) * waterBondLength
        productBonds.append(Bond(a: oxygen, b: h1, order: 1))
        productBonds.append(Bond(a: oxygen, b: h2, order: 1))
    }

    return Reaction(elements: elements, start: start, end: end, reactantBonds: reactantBonds,
                    productBonds: productBonds, carbons: carbons, o2Oxygens: o2Oxygens)
}

// MARK: - Oxidation states

/// Oxidation state of every atom, from the bonds: in each bond the more
/// electronegative atom counts the shared electrons as its own (−1 per bond
/// order) and the other atom loses them (+1). Bonds between like atoms count
/// for neither.
func oxidationStates(elements: [Element], bonds: [Bond]) -> [Float] {
    var states = [Float](repeating: 0, count: elements.count)
    for b in bonds {
        let ea = elements[b.a].electronegativity, eb = elements[b.b].electronegativity
        if ea > eb {
            states[b.a] -= b.order
            states[b.b] += b.order
        } else if eb > ea {
            states[b.b] -= b.order
            states[b.a] += b.order
        }
    }
    return states
}

/// Electrons that move from carbon to oxygen: how much the carbons' oxidation
/// states rise in total.
func electronsTransferred(_ r: Reaction) -> Float {
    let before = oxidationStates(elements: r.elements, bonds: r.reactantBonds)
    let after = oxidationStates(elements: r.elements, bonds: r.productBonds)
    return r.carbons.reduce(0) { $0 + after[$1] - before[$1] }
}

// MARK: - The animation

struct Timeline {
    let calm: Double = 3.0        // glucose and O₂, intact
    let react: Double = 5.5       // bonds break, atoms move, new bonds form
    let after: Double = 3.0       // CO₂ and H₂O, settled
    var total: Double { calm + react + after }

    /// 0 → 1 across the reaction.
    func progress(at t: Double) -> Float { Float(min(max((t - calm) / react, 0), 1)) }

    func stage(at t: Double) -> String {
        let p = progress(at: t)
        if p <= 0 { return "Glucose + 6 O₂" }
        if p < 1 { return "Carbon hands its electrons to oxygen" }
        return "6 CO₂ + 6 H₂O + energy"
    }
}

/// Phases within the reaction (fractions of `progress`): old bonds fade, atoms
/// travel, new bonds form.
let breakEnd: Float = 0.3
let moveStart: Float = 0.15
let moveEnd: Float = 0.85
let formStart: Float = 0.7

/// How far atom `i` has traveled, 0 → 1. Atoms set off at slightly different
/// times, so the motion doesn't look mechanical.
func travel(_ progress: Float, atom i: Int) -> Float {
    let stagger: Float = Float(i % 5) * 0.03
    let span: Float = moveEnd - moveStart - 0.12
    return smoothstep01((progress - moveStart - stagger) / span)
}

/// The scene at `progress` (0 = reactants, 1 = products).
func reactionState(_ r: Reaction, progress: Float) -> MoleculeState {
    var atoms: [Atom] = []
    // Electrons on the move: a soft glow behind the atoms giving (gold) and
    // taking (cyan) them, strongest mid-journey.
    let journey: Float = min(max((progress - moveStart) / (moveEnd - moveStart), 0), 1)
    // sin(π) isn't exactly 0 in floating point, so clamp the ends to exactly 0.
    let moving: Float = (journey <= 0 || journey >= 1) ? 0 : sin(Float.pi * journey)
    for i in 0..<r.elements.count {
        let t = travel(progress, atom: i)
        var p = r.start[i] + (r.end[i] - r.start[i]) * t
        // Lift each path out of the plane a little (alternating sides), so
        // atoms pass by each other instead of through each other.
        let lift: Float = (i % 2 == 0 ? 1.4 : -1.4) * sin(Float.pi * t)
        p.z += lift
        var glow: Float = 0
        if r.carbons.contains(i) { glow = 0.45 * moving }
        if r.o2Oxygens.contains(i) { glow = -0.45 * moving }
        atoms.append(Atom(element: r.elements[i], position: p, glow: glow))
    }
    let fadeOut: Float = 1 - smoothstep01(progress / breakEnd)
    let fadeIn: Float = smoothstep01((progress - formStart) / (1 - formStart))
    var bonds: [Bond] = []
    for b in r.reactantBonds where fadeOut > 0 { bonds.append(Bond(a: b.a, b: b.b, order: b.order * fadeOut)) }
    for b in r.productBonds where fadeIn > 0 { bonds.append(Bond(a: b.a, b: b.b, order: b.order * fadeIn)) }
    return MoleculeState(atoms: atoms, bonds: bonds)
}

func distance(_ state: MoleculeState, _ i: Int, _ j: Int) -> Float {
    simd_distance(state.atoms[i].position, state.atoms[j].position)
}

func angle(_ state: MoleculeState, _ i: Int, _ j: Int, _ k: Int) -> Float {
    let u = simd_normalize(state.atoms[i].position - state.atoms[j].position)
    let v = simd_normalize(state.atoms[k].position - state.atoms[j].position)
    return acos(min(max(simd_dot(u, v), -1), 1)) * 180 / .pi
}

/// "+4", "0", "−1".
func signedLabel(_ x: Float) -> String {
    let n = Int(x.rounded())
    if n > 0 { return "+\(n)" }
    if n < 0 { return "−\(-n)" }
    return "0"
}
