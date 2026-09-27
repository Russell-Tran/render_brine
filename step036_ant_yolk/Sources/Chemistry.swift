// The chemistry in the insets: oleic acid, yolk fat's commonest fatty acid,
// at the taste pore, and in the air by the smell hair, hexanal — coordinates
// from real structure files, not drawn.

import Foundation
import simd

// MARK: - atoms and molecules

struct Atom {
    var element: String          // "C", "O", "H"
    var position: SIMD3<Float>   // Å
    var view: Int = 0            // which molecule inset: 0 taste, 1 smell
}

struct Molecule {
    var atoms: [Atom]
    var bonds: [(Int, Int)]
    var orders: [Int] = []       // bond orders, parallel to bonds (1 if absent)

    var formula: [String: Int] {
        var f: [String: Int] = [:]
        for a in atoms { f[a.element, default: 0] += 1 }
        return f
    }

    /// Sum of bond orders at each atom: C 4, O 2, H 1 for a proper molecule.
    var valences: [Int] {
        var v: [Int] = Array(repeating: 0, count: atoms.count)
        for (k, b) in bonds.enumerated() {
            let o: Int = k < orders.count ? orders[k] : 1
            v[b.0] += o
            v[b.1] += o
        }
        return v
    }

    /// Rings, from the graph: bonds − atoms + connected pieces.
    var rings: Int {
        var parent: [Int] = Array(0..<atoms.count)
        func find(_ x: Int) -> Int { var x = x; while parent[x] != x { x = parent[x] }; return x }
        for b in bonds { let a = find(b.0), c = find(b.1); if a != c { parent[a] = c } }
        let pieces: Int = Set((0..<atoms.count).map { find($0) }).count
        return bonds.count - atoms.count + pieces
    }

    /// Centred on its own centroid, turned, moved, and tagged for an inset.
    func placed(turn: simd_quatf, at offset: SIMD3<Float>, view: Int) -> Molecule {
        var c = SIMD3<Float>(0, 0, 0)
        for a in atoms { c += a.position }
        c /= Float(max(atoms.count, 1))
        let moved: [Atom] = atoms.map { Atom(element: $0.element, position: turn.act($0.position - c) + offset, view: view) }
        return Molecule(atoms: moved, bonds: bonds, orders: orders)
    }
}

enum LoadError: Error { case missing(String), malformed(String) }

/// Reads an MDL molfile (V2000): the counts line, then atoms, then bonds with
/// their orders. Both files here carry their own hydrogens, so nothing is
/// added or guessed.
func loadMolfile(_ path: String) throws -> Molecule {
    guard let text = try? String(contentsOfFile: path, encoding: .utf8) else { throw LoadError.missing(path) }
    let lines: [String] = text.components(separatedBy: "\n")
    guard lines.count > 4 else { throw LoadError.malformed(path) }
    let counts = Array(lines[3])
    func int(_ s: ArraySlice<Character>) -> Int { Int(String(s).trimmingCharacters(in: .whitespaces)) ?? 0 }
    let nAtoms: Int = int(counts[0..<3])
    let nBonds: Int = int(counts[3..<6])
    guard lines.count >= 4 + nAtoms + nBonds else { throw LoadError.malformed(path) }
    var atoms: [Atom] = []
    for i in 0..<nAtoms {
        let f: [Substring] = lines[4 + i].split(separator: " ", omittingEmptySubsequences: true)
        guard f.count >= 4, let x = Float(f[0]), let y = Float(f[1]), let z = Float(f[2]) else { throw LoadError.malformed(path) }
        atoms.append(Atom(element: String(f[3]), position: SIMD3(x, y, z)))
    }
    var bonds: [(Int, Int)] = []
    var orders: [Int] = []
    for i in 0..<nBonds {
        let l = Array(lines[4 + nAtoms + i])
        bonds.append((int(l[0..<3]) - 1, int(l[3..<6]) - 1))
        orders.append(int(l[6..<9]))
    }
    return Molecule(atoms: atoms, bonds: bonds, orders: orders)
}

// Water: O–H 0.9572 Å, H–O–H 104.52° (Hoy & Bunker 1979, as given in
// Wikipedia "Properties of water").
let waterOH: Float = 0.9572
let waterAngle: Float = 104.52 * Float.pi / 180

func waterMolecule(at o: SIMD3<Float>, turn: simd_quatf, view: Int) -> Molecule {
    let half: Float = waterAngle / 2
    let h1: SIMD3<Float> = o + turn.act(SIMD3<Float>(sin(half), cos(half), 0)) * waterOH
    let h2: SIMD3<Float> = o + turn.act(SIMD3<Float>(-sin(half), cos(half), 0)) * waterOH
    return Molecule(atoms: [Atom(element: "O", position: o, view: view), Atom(element: "H", position: h1, view: view),
                            Atom(element: "H", position: h2, view: view)],
                    bonds: [(0, 1), (0, 2)], orders: [1, 1])
}

/// Joins molecules into one list, renumbering the bonds.
func merge(_ ms: [Molecule]) -> Molecule {
    var atoms: [Atom] = []
    var bonds: [(Int, Int)] = []
    var orders: [Int] = []
    for m in ms {
        let base: Int = atoms.count
        atoms += m.atoms
        bonds += m.bonds.map { ($0.0 + base, $0.1 + base) }
        orders += m.bonds.indices.map { $0 < m.orders.count ? m.orders[$0] : 1 }
    }
    return Molecule(atoms: atoms, bonds: bonds, orders: orders)
}

// MARK: - the two insets' contents

// The taste molecule: oleic acid, cis-9-octadecenoic acid, the commonest
// fatty acid in yolk fat — 30.6–43.9% of hen-yolk fatty acids across the
// breeds and species surveyed in *Philippine Journal of Science* (2023),
// "Fat content, fatty acid composition, and fatty acid-based nutritional
// indices/ratios of egg yolks from different poultry species and breeds".
// Insects can taste fatty acids: in Drosophila, sweet-sensing taste neurons
// respond to them through the ionotropic receptors IR25a, IR76b and IR56d
// (Ahn, Chen & Amrein, *eLife* 6: e30115, 2017). Whether Lasius niger does
// the same has not been shown, and the picture does not claim it: it shows
// the molecule, not a receptor. Most yolk fat is bound in triglycerides and
// phospholipids; free oleic acid is a small share.
//
// The odour molecule: hexanal (Food.swift says why). Both are PubChem 3-D
// conformers with their hydrogens — CID 445639 (oleic acid) and CID 6184
// (hexanal) — copied verbatim into Resources/. Turns and positions MODEL.

struct YolkChemistry {
    var oleic: Molecule
    var waters: [Molecule]
    var odorant: Molecule

    /// Everything the two molecule insets draw, in their own ångströms.
    var all: Molecule { merge([oleic] + waters + [odorant]) }
}

func buildChemistry(resources: String, mutant: Mutant) throws -> YolkChemistry {
    var ole: Molecule = try loadMolfile(resources + "/oleic_acid_CID445639_3d.sdf")
    let hex: Molecule = try loadMolfile(resources + "/hexanal_CID6184_3d.sdf")
    if mutant == .formula {
        // Drop the acid hydrogen: C18H33O2 — oleate without its charge, not
        // a molecule at all.
        if let k = ole.atoms.lastIndex(where: { $0.element == "H" }) {
            ole.atoms.remove(at: k)
            let keep: [Int] = ole.bonds.indices.filter { ole.bonds[$0].0 != k && ole.bonds[$0].1 != k }
            ole.orders = keep.map { ole.orders[$0] }
            ole.bonds = keep.map { b -> (Int, Int) in
                let (x, y) = ole.bonds[b]
                return (x > k ? x - 1 : x, y > k ? y - 1 : y)
            }
        }
    }
    // Lay the chain across the inset: turn its longest axis onto x.
    let o: Molecule = alignLongAxis(ole).placed(turn: simd_quatf(angle: 0.35, axis: SIMD3<Float>(1, 0, 0)),
                                                at: SIMD3(0, 0.6, 0), view: 0)
    // A few waters of the moisture film, near the acid head (yolk is half
    // water). MODEL positions.
    let head: SIMD3<Float> = carboxylCarbon(o).map { o.atoms[$0].position } ?? SIMD3(0, 0, 0)
    let spots: [SIMD3<Float>] = [SIMD3(1.2, 3.4, 1.0), SIMD3(-2.8, -2.0, 1.2), SIMD3(-0.4, -3.4, -1.0)]
    // Each water is pushed straight out from the head until no atom of it is
    // within 2.2 Å of the acid: waters beside the molecule, never inside it.
    let ws: [Molecule] = spots.enumerated().map { k, d in
        let turn = simd_quatf(angle: 0.9 + 1.3 * Float(k), axis: simd_normalize(SIMD3<Float>(1, Float(k) - 1, 0.4)))
        var reach: Float = 1
        var w: Molecule = waterMolecule(at: head + d, turn: turn, view: 0)
        func clash(_ m: Molecule) -> Bool {
            m.atoms.contains { a in o.atoms.contains { simd_distance(a.position, $0.position) < 2.2 } }
        }
        while clash(w) && reach < 4 {
            reach += 0.05
            w = waterMolecule(at: head + d * reach, turn: turn, view: 0)
        }
        return w
    }
    let h: Molecule = alignLongAxis(hex).placed(turn: simd_quatf(angle: 0.5, axis: simd_normalize(SIMD3<Float>(1, 0.3, 0.2))),
                                                at: SIMD3(0, -0.3, 0), view: 1)
    return YolkChemistry(oleic: o, waters: ws, odorant: h)
}

/// The carbon of the –COOH group: the carbon bonded to two oxygens.
func carboxylCarbon(_ m: Molecule) -> Int? {
    for i in 0..<m.atoms.count where m.atoms[i].element == "C" {
        var oxygens: Int = 0
        for b in m.bonds {
            let other: Int = b.0 == i ? b.1 : (b.1 == i ? b.0 : -1)
            if other >= 0 && m.atoms[other].element == "O" { oxygens += 1 }
        }
        if oxygens == 2 { return i }
    }
    return nil
}

/// Turns a molecule so the line through its two most distant heavy atoms
/// runs along x.
func alignLongAxis(_ m: Molecule) -> Molecule {
    var a: Int = 0, b: Int = 0
    var far: Float = 0
    for i in 0..<m.atoms.count where m.atoms[i].element != "H" {
        for j in (i + 1)..<m.atoms.count where m.atoms[j].element != "H" {
            let d: Float = simd_distance(m.atoms[i].position, m.atoms[j].position)
            if d > far { far = d; a = i; b = j }
        }
    }
    let q: simd_quatf = simd_quatf(from: simd_normalize(m.atoms[b].position - m.atoms[a].position), to: SIMD3<Float>(1, 0, 0))
    var out: Molecule = m
    out.atoms = m.atoms.map { Atom(element: $0.element, position: q.act($0.position), view: $0.view) }
    return out
}

// MARK: - odour molecules for the smell hair

/// One odour molecule in the micrometre inset, drawn as a dot: where it is,
/// and the wall pore it is heading for.
struct Volatile {
    var position: SIMD3<Float>   // µm
    var pore: (row: Int, column: Int)
}

/// Cooked yolk has a smell (Food.swift): a few hexanal molecules drift up
/// from the crumb to the smell hair. Two are arriving at wall pores on the
/// side facing the camera; three are still on their way up. The `volatiles`
/// mutant removes them, and the tests must notice.
func buildVolatiles(hairs: [Sensillum], mutant: Mutant, facing view: SIMD3<Float>) -> [Volatile] {
    guard foodHasVapour && mutant != .volatiles else { return [] }
    let smell: Sensillum = hairs[1]
    // The column whose pores look most nearly at the camera and down to the
    // food — the pores we can see molecules arrive at.
    let want: SIMD3<Float> = simd_normalize(-view + SIMD3<Float>(-0.4, -0.5, 0))
    var bestColumn: Int = 0
    var bestDot: Float = -2
    for j in 0..<smell.poreColumns {
        let d: Float = simd_dot(smell.wallPore(row: 6, column: j).outward, want)
        if d > bestDot { bestDot = d; bestColumn = j }
    }
    var out: [Volatile] = []
    let plan: [(row: Int, dj: Int, away: Float)] = [(5, 0, 0.13), (11, 1, 0.13), (8, 0, 1.3), (3, -1, 2.6), (13, 1, 3.4)]
    for p in plan {
        let j: Int = (bestColumn + p.dj + smell.poreColumns) % smell.poreColumns
        let (mouth, outward) = smell.wallPore(row: p.row, column: j)
        // Those on the way are below and out from their pore, rising from the yolk.
        let path: SIMD3<Float> = simd_normalize(outward + SIMD3<Float>(0, -0.9, 0))
        out.append(Volatile(position: mouth + path * p.away, pore: (p.row, j)))
    }
    return out
}

/// Dot radius for an odour molecule in the inset, µm. A hexanal molecule is under a nanometre across; at this scale it would be
/// invisible, so it is drawn several hundred times too big, and the label
/// says so.
let volatileDotRadius: Float = 0.10
