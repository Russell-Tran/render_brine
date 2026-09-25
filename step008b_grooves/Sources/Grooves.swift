// The duplex from Resources/grooves.json, drawn two ways.
//
// SPACE-FILLING is the point of this step: every atom a sphere at its full van
// der Waals radius (Bondi 1964), so neighbours overlap and merge into one
// solid. That is the size an atom's electrons actually keep other atoms out
// of, and drawing it reveals two things balls-on-sticks hides — that the core
// of the duplex is packed solid, and that the space outside it is not empty
// filler but the major and minor grooves.
//
// BALL-AND-STICK is kept only for the comparison still: the same twelve
// crystal base pairs, nothing changed but the representation.
//
// Colour is by which part of the molecule an atom belongs to, not by element.
// At this scale the question is not what an atom is but which groove it lines,
// and CPK colouring would answer the wrong one.

import Foundation
import simd

func radians(_ degrees: Float) -> Float { degrees * .pi / 180 }

/// Which part of the duplex an atom belongs to, which is what it is coloured by.
enum Part: String {
    case backbone, major, minor, interior, hydrogen
}

struct GAtom {
    var element: String
    var position: SIMD3<Float>
    var part: Part
    var name: String
    var pair: Int          // which base pair along the helix; -1 in the crystal
    var strand: Int        // 0 or 1; -1 in the crystal
}

struct Duplex {
    var sequence: String
    var atoms: [GAtom]
    var bonds: [(a: Int, b: Int, order: Int)]
    /// Fraction of each 1 Å cylindrical shell about the axis that is inside an
    /// atom, measured in the builder from the real crystal.
    var occupancy: [Float]
    var grooveWidth: [String: Float]
    var groovePP: [String: Float]
}

enum SceneError: Error, CustomStringConvertible {
    case badFile(String)
    var description: String {
        switch self {
        case .badFile(let d): return "couldn't read the duplex: \(d)"
        }
    }
}

/// Van der Waals radii, Bondi, J. Phys. Chem. 68:441 (1964). Read from the
/// file rather than hard-coded here, so the render and the measurement that
/// justifies it cannot drift apart.
var vdwRadii: [String: Float] = [:]

func vdw(_ element: String) -> Float {
    vdwRadii[element] ?? 1.7
}

private func vector(_ a: [NSNumber]) -> SIMD3<Float> {
    SIMD3(a[0].floatValue, a[1].floatValue, a[2].floatValue)
}

/// Both structures: the 40 bp idealised helix that turns, and the twelve
/// measured crystal base pairs it is checked against.
func loadDuplexes(from url: URL) throws -> (helix: Duplex, crystal: Duplex) {
    let data = try Data(contentsOf: url)
    guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
          let radii = root["vdw"] as? [String: NSNumber],
          let h = root["helix"] as? [String: Any],
          let c = root["crystal"] as? [String: Any] else { throw SceneError.badFile("missing fields") }
    vdwRadii = radii.mapValues { $0.floatValue }

    // The 40-mer, whose atoms the builder has already classified by part.
    guard let rawHelix = h["atoms"] as? [[String: Any]],
          let helixBonds = h["bonds"] as? [[Int]] else { throw SceneError.badFile("bad helix") }
    var helixAtoms: [GAtom] = []
    helixAtoms.reserveCapacity(rawHelix.count)
    for a in rawHelix {
        guard let el = a["el"] as? String, let p = a["pos"] as? [NSNumber], p.count == 3 else {
            throw SceneError.badFile("bad helix atom")
        }
        let part = Part(rawValue: a["part"] as? String ?? "interior") ?? .interior
        helixAtoms.append(GAtom(element: el, position: vector(p), part: part,
                                name: a["name"] as? String ?? "", pair: a["pair"] as? Int ?? -1,
                                strand: a["strand"] as? Int ?? -1))
    }

    // The crystal, whose atoms carry a PDB label instead; classified here by
    // the same rules so the comparison still is coloured identically.
    guard let rawCrystal = c["atoms"] as? [[String: Any]],
          let crystalBonds = c["bonds"] as? [[Int]] else { throw SceneError.badFile("bad crystal") }
    let crystalSeq = c["sequence"] as? String ?? ""
    var crystalAtoms: [GAtom] = []
    for a in rawCrystal {
        guard let el = a["el"] as? String, let p = a["pos"] as? [NSNumber], p.count == 3 else {
            throw SceneError.badFile("bad crystal atom")
        }
        let label = a["label"] as? String ?? ""
        let (name, base) = splitLabel(label, sequence: crystalSeq)
        crystalAtoms.append(GAtom(element: el, position: vector(p),
                                  part: classify(name: name, base: base), name: name,
                                  pair: -1, strand: -1))
    }

    func widths(_ d: [String: Any], _ key: String) -> [String: Float] {
        var out: [String: Float] = [:]
        for (name, v) in (d["grooves"] as? [String: [String: NSNumber]]) ?? [:] {
            out[name] = v[key]?.floatValue ?? 0
        }
        return out
    }
    let occupancy = (c["occupancy"] as? [NSNumber] ?? []).map { $0.floatValue }

    let helix = Duplex(sequence: h["sequence"] as? String ?? "", atoms: helixAtoms,
                       bonds: helixBonds.map { (a: $0[0], b: $0[1], order: $0[2]) },
                       occupancy: occupancy,
                       grooveWidth: widths(h, "width"), groovePP: widths(h, "pp"))
    let crystal = Duplex(sequence: crystalSeq, atoms: crystalAtoms,
                         bonds: crystalBonds.map { (a: $0[0], b: $0[1], order: $0[2]) },
                         occupancy: occupancy,
                         grooveWidth: widths(c, "width"), groovePP: widths(c, "pp"))
    return (helix, crystal)
}

/// "A5.N6" → atom name N6 on base A (via the dodecamer's sequence).
func splitLabel(_ label: String, sequence: String) -> (name: String, base: String) {
    let parts = label.split(separator: ".", maxSplits: 1)
    guard parts.count == 2 else { return ("", "A") }
    let name = String(parts[1])
    let chainSeq = parts[0]
    let chain = chainSeq.first.map(String.init) ?? "A"
    let number = Int(chainSeq.dropFirst()) ?? 1
    let letters = Array(sequence)
    guard !letters.isEmpty else { return (name, "A") }
    // Residue n on chain A pairs with 25 − n on chain B.
    let index = chain == "A" ? number - 1 : (25 - number) - 1
    guard index >= 0 && index < letters.count else { return (name, "A") }
    let base = String(letters[index])
    return (name, chain == "A" ? base : complement(base))
}

func complement(_ base: String) -> String {
    switch base {
    case "A": return "T"
    case "T": return "A"
    case "C": return "G"
    case "G": return "C"
    default: return base
    }
}

let backboneNames: Set<String> = ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'"]

/// Which groove a base atom's edge looks into. Seeman, Rosenberg & Rich,
/// PNAS 73:804 (1976).
let majorEdge: [String: Set<String>] = [
    "A": ["N7", "C5", "C6", "N6"],
    "G": ["N7", "C5", "C6", "O6"],
    "T": ["C4", "O4", "C5", "C7", "C5M", "C6"],
    "C": ["C4", "N4", "C5", "C6"],
]
let minorEdge: [String: Set<String>] = [
    "A": ["C2", "N3"],
    "G": ["C2", "N2", "N3"],
    "T": ["C2", "O2"],
    "C": ["C2", "O2"],
]

func classify(name: String, base: String) -> Part {
    if backboneNames.contains(name) { return .backbone }
    if name.hasPrefix("H") { return .hydrogen }
    if majorEdge[base]?.contains(name) == true { return .major }
    if minorEdge[base]?.contains(name) == true { return .minor }
    return .interior
}

/// Colour by part, not by element.
///
/// The backbone is the pale ridge that runs along the outside between the two
/// grooves. The two sets of base edges are the walls the grooves are cut
/// between, and they are the thing to look at: the major groove's edge is
/// where a protein reads the sequence.
func partColor(_ part: Part) -> SIMD3<Float> {
    switch part {
    case .backbone: return SIMD3(0.78, 0.74, 0.62)     // warm pale sand
    case .major:    return SIMD3(0.16, 0.62, 0.78)     // the wide groove, teal
    case .minor:    return SIMD3(0.80, 0.42, 0.24)     // the narrow one, rust
    case .interior: return SIMD3(0.42, 0.44, 0.48)     // ring atoms facing neither
    case .hydrogen: return SIMD3(0.88, 0.88, 0.86)
    }
}

/// Every heavy atom as a sphere at its full van der Waals radius. No sticks:
/// the spheres overlap, which is the entire point.
///
/// Hydrogens are drawn nowhere in this step, for the same reason step 9 left
/// them out of GFP: X-rays at this resolution cannot see them, so 1BNA's were
/// added by a model rather than measured, and a space-filling surface is
/// conventionally heavy-atom anyway. Drawn, their 902 white spheres speckle
/// the surface and drown the colouring that carries the whole argument.
///
/// They are still counted in the OCCUPANCY measurement, where leaving them out
/// would be wrong: a hydrogen occupies volume whether or not it was measured.
func spaceFilling(_ duplex: Duplex) -> [GPUSphere] {
    var spheres: [GPUSphere] = []
    spheres.reserveCapacity(duplex.atoms.count)
    for atom in duplex.atoms where atom.element != "H" {
        let p = atom.position
        let c = partColor(atom.part)
        spheres.append(GPUSphere(centerRadius: SIMD4(p.x, p.y, p.z, vdw(atom.element)),
                                 color: SIMD4(c.x, c.y, c.z, 1), glow: .zero))
    }
    return spheres
}

/// The step 6–8 representation, for the comparison still only: small balls at
/// a fraction of the van der Waals radius, joined by sticks.
let ballScale: Float = 0.25

func ballAndStick(_ duplex: Duplex, camera: Camera) -> (spheres: [GPUSphere], cylinders: [GPUCylinder]) {
    var spheres: [GPUSphere] = []
    for atom in duplex.atoms where atom.element != "H" {
        let p = atom.position
        let c = partColor(atom.part)
        spheres.append(GPUSphere(centerRadius: SIMD4(p.x, p.y, p.z, vdw(atom.element) * ballScale),
                                 color: SIMD4(c.x, c.y, c.z, 1), glow: .zero))
    }
    var cylinders: [GPUCylinder] = []
    let grey = SIMD4<Float>(0.62, 0.64, 0.67, 1)
    for bond in duplex.bonds {
        guard bond.a < duplex.atoms.count, bond.b < duplex.atoms.count else { continue }
        guard duplex.atoms[bond.a].element != "H", duplex.atoms[bond.b].element != "H" else { continue }
        let a = duplex.atoms[bond.a].position, b = duplex.atoms[bond.b].position
        cylinders.append(GPUCylinder(aRadius: SIMD4(a.x, a.y, a.z, bondRadius),
                                     b: SIMD4(b.x, b.y, b.z, 0), color: grey))
    }
    return (spheres, cylinders)
}

/// The palette keeps a few shades of every part's colour whatever else it
/// chooses. Median cut alone gives colours to whatever covers the most pixels,
/// so the narrow minor-groove edge — which is exactly what the render is
/// about — would lose its rust to the backbone's sand (step 8 hit this with
/// phosphorus turning red).
func groovePalette(samples: [RGB], count: Int) -> [RGB] {
    var reserved: [RGB] = []
    for part in [Part.backbone, .major, .minor, .interior, .hydrogen] {
        let c = partColor(part)
        for light: Float in [0.3, 0.5, 0.7, 0.9, 1.1] {
            let v = simd_clamp(c * light, SIMD3<Float>(repeating: 0), SIMD3<Float>(repeating: 1)) * 255
            reserved.append(RGB(UInt8(v.x.rounded()), UInt8(v.y.rounded()), UInt8(v.z.rounded())))
        }
    }
    return reserved + medianCutPalette(samples, count: count - reserved.count)
}

/// How far the camera sits from the duplex. The 40-mer is 40 x 3.4 = 136 Å
/// long. The frame is short and wide to suit it — an empty sky above a thin
/// duplex is wasted pixels — which widens the aspect and lets the camera come
/// closer than a square-ish frame would allow.
/// A test walks the whole turn and checks every atom's own sphere stays in
/// frame. That test is also what caught the duplex being stamped from x = 0
/// upwards rather than centred, which no distance would have fixed.
let cameraDistance: Float = 138

/// Mean of a list, 0 when empty.
func mean(_ xs: [Float]) -> Float { xs.isEmpty ? 0 : xs.reduce(0, +) / Float(xs.count) }
