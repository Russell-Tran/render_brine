// The chemistry in the insets: fructose and glucose with honey's own share of
// water at the taste pore, and in the air by the smell hair, the odorant
// phenylacetaldehyde — coordinates from real structure files, not drawn.

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

// In solution the sugars are mostly rings, and mostly one form of ring:
// fructose is chiefly β-D-fructopyranose (about 70%) and glucose chiefly
// β-D-glucopyranose (about 64%, the rest α) — Wikipedia, "Fructose" and
// "Glucose". So those are the forms drawn, from the RCSB Chemical Component
// Dictionary's ideal coordinates, hydrogens included: BDF (β-D-
// fructopyranose) and BGC (β-D-glucose), copied verbatim into Resources/.
// Phenylacetaldehyde is PubChem CID 998's computed 3-D conformer, also
// verbatim. The turns and positions in the insets are MODEL — the molecules
// tumble.

struct HoneyChemistry {
    var fructose: Molecule
    var glucose: Molecule
    var waters: [Molecule]
    var odorant: Molecule

    /// Everything the two molecule insets draw, in their own ångströms.
    var all: Molecule { merge([fructose, glucose] + waters + [odorant]) }
}

func buildChemistry(resources: String, mutant: Mutant) throws -> HoneyChemistry {
    var fru: Molecule = try loadMolfile(resources + "/fructose_BDF_ideal.sdf")
    let glc: Molecule = try loadMolfile(resources + "/glucose_BGC_ideal.sdf")
    let pa: Molecule = try loadMolfile(resources + "/phenylacetaldehyde_CID998_3d.sdf")
    if mutant == .formula {
        // Drop one hydroxyl hydrogen: C6H11O6 — not a sugar, not neutral.
        if let k = fru.atoms.lastIndex(where: { $0.element == "H" }) {
            fru.atoms.remove(at: k)
            let keep: [Int] = fru.bonds.indices.filter { fru.bonds[$0].0 != k && fru.bonds[$0].1 != k }
            fru.orders = keep.map { fru.orders[$0] }
            fru.bonds = keep.map { b -> (Int, Int) in
                let (x, y) = fru.bonds[b]
                return (x > k ? x - 1 : x, y > k ? y - 1 : y)
            }
        }
    }
    let f: Molecule = fru.placed(turn: simd_quatf(angle: 0.9, axis: simd_normalize(SIMD3<Float>(0.3, 1, 0.2))),
                                 at: SIMD3(-3.4, 1.6, 0), view: 0)
    let g: Molecule = glc.placed(turn: simd_quatf(angle: 2.1, axis: simd_normalize(SIMD3<Float>(1, 0.3, -0.4))),
                                 at: SIMD3(3.6, -2.6, -1.5), view: 0)
    // Two sugars, so five waters: honey's 2.5 per hexose (Food.swift).
    let wSpots: [SIMD3<Float>] = [SIMD3(0.6, 5.2, 1.5), SIMD3(-6.6, -3.2, 1.0), SIMD3(0.2, -5.6, 2.5),
                                  SIMD3(6.8, 3.0, 0.5), SIMD3(-1.0, -1.2, 3.8)]
    let ws: [Molecule] = wSpots.enumerated().map { k, p in
        waterMolecule(at: p, turn: simd_quatf(angle: 0.8 + 1.1 * Float(k), axis: simd_normalize(SIMD3<Float>(1, Float(k) - 2, 0.5))), view: 0)
    }
    let o: Molecule = pa.placed(turn: simd_quatf(angle: 0.7, axis: simd_normalize(SIMD3<Float>(1, 0.4, 0.2))),
                                at: SIMD3(0, -0.3, 0), view: 1)
    return HoneyChemistry(fructose: f, glucose: g, waters: ws, odorant: o)
}

// MARK: - odour molecules for the smell hair

/// One odour molecule in the micrometre inset, drawn as a dot: where it is,
/// and the wall pore it is heading for.
struct Volatile {
    var position: SIMD3<Float>   // µm
    var pore: (row: Int, column: Int)
}

/// Honey has a smell (Food.swift): a few odorant molecules drift up from the
/// honey surface to the smell hair. Two are arriving at wall pores on the
/// side facing the camera; three are still on their way up. The `volatiles`
/// mutant removes them, and the tests must notice.
func buildVolatiles(hairs: [Sensillum], mutant: Mutant, facing view: SIMD3<Float>) -> [Volatile] {
    guard foodHasVapour && mutant != .volatiles else { return [] }
    let smell: Sensillum = hairs[1]
    // The column whose pores look most nearly at the camera and down to the
    // honey — the pores we can see molecules arrive at.
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
        // Those on the way are below and out from their pore, rising from the honey.
        let path: SIMD3<Float> = simd_normalize(outward + SIMD3<Float>(0, -0.9, 0))
        out.append(Volatile(position: mouth + path * p.away, pore: (p.row, j)))
    }
    return out
}

/// Dot radius for an odour molecule in the inset, µm. A phenylacetaldehyde
/// molecule is under a nanometre across; at this scale it would be
/// invisible, so it is drawn several hundred times too big, and the label
/// says so.
let volatileDotRadius: Float = 0.10
