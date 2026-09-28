// The chemistry in the inset at the taste pore: one molecule of leaf wax, a
// long straight alkane, built atom by atom from measured bond geometry.
// There is no odorant inset: an intact leaf gives off almost nothing.

import Foundation
import simd

// MARK: - atoms and molecules

struct Atom {
    var element: String          // "C", "H"
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

// MARK: - the wax molecule

// Which molecule. Banana leaf wax is mostly "paraffins, primary alcohols and
// fatty acids" (Freeman & Turner 1985, Food.swift). A paraffin is drawn:
// hentriacontane, n-C31H64. Its chain length is MODEL (UNVERIFIED for
// Musa): plant leaf-wax alkanes are overwhelmingly odd-numbered, with C29
// and C31 the commonest across plants (the pattern Eglinton & Hamilton,
// "Leaf epicuticular waxes", *Science* 156: 1322, 1967, made famous); no
// chain-length profile for Cavendish leaf wax was found.
let waxCarbons: Int = 31

// Its geometry. PubChem computes no 3-D conformer for a chain this long (it
// has too many rotatable bonds), so the molecule is built, all-trans — the
// shape alkanes take in a wax crystal — from bond lengths and angles
// measured on PubChem's 3-D conformer of n-hexadecane (CID 11006, the
// longest straight alkane PubChem gives in 3-D), averaged here over all its
// bonds: C–C 1.530 Å, C–H 1.096 Å, C–C–C 113.1°, H–C–H 107.2°, H–C–C
// (methyl) 109.3°.
let alkaneCC: Float = 1.530
let alkaneCH: Float = 1.096
let alkaneCCC: Float = 113.1 * Float.pi / 180
let alkaneHCH: Float = 107.2 * Float.pi / 180
let alkaneHCC: Float = 109.3 * Float.pi / 180

/// n-CₙH₂ₙ₊₂, all-trans: the carbons zig-zag in the xy plane along x; each
/// CH2's two hydrogens straddle that plane, bisecting away from its two
/// carbon neighbours; each end CH3 is staggered, one H in the plane, anti
/// to the chain.
func buildAlkane(carbons n: Int, dropHydrogen: Bool = false) -> Molecule {
    var atoms: [Atom] = []
    var bonds: [(Int, Int)] = []
    let half: Float = alkaneCCC / 2
    let dx: Float = alkaneCC * sin(half)
    let dy: Float = alkaneCC * cos(half)
    for i in 0..<n {
        let y: Float = i % 2 == 0 ? 0 : dy
        atoms.append(Atom(element: "C", position: SIMD3<Float>(Float(i) * dx, y, 0)))
        if i > 0 { bonds.append((i - 1, i)) }
    }
    func addH(_ c: Int, _ dir: SIMD3<Float>) {
        atoms.append(Atom(element: "H", position: atoms[c].position + simd_normalize(dir) * alkaneCH))
        bonds.append((c, atoms.count - 1))
    }
    let zAxis = SIMD3<Float>(0, 0, 1)
    for i in 0..<n {
        let c: SIMD3<Float> = atoms[i].position
        if i > 0 && i < n - 1 {
            // Away from the neighbours' midpoint, tilted ± out of plane.
            let away: SIMD3<Float> = simd_normalize(c - (atoms[i - 1].position + atoms[i + 1].position) / 2)
            let a: Float = alkaneHCH / 2
            addH(i, away * cos(a) + zAxis * sin(a))
            addH(i, away * cos(a) - zAxis * sin(a))
        } else {
            // A methyl: three H at H–C–C to the one carbon neighbour,
            // 120° apart about that bond, the first in the plane and anti.
            let nb: Int = i == 0 ? 1 : n - 2
            let u: SIMD3<Float> = simd_normalize(atoms[nb].position - c)
            let inPlane: SIMD3<Float> = simd_normalize(simd_cross(zAxis, u))
            // Pick the in-plane perpendicular that points away from the
            // next carbon along (anti).
            let next: Int = i == 0 ? 2 : n - 3
            let side: SIMD3<Float> = simd_dot(inPlane, atoms[next].position - atoms[nb].position) > 0 ? -inPlane : inPlane
            let w: SIMD3<Float> = simd_cross(u, side)
            for k in 0..<3 {
                let phi: Float = Float(k) * 2 * Float.pi / 3
                let perp: SIMD3<Float> = side * cos(phi) + w * sin(phi)
                addH(i, u * cos(alkaneHCC) + perp * sin(alkaneHCC))
            }
        }
    }
    var m = Molecule(atoms: atoms, bonds: bonds, orders: Array(repeating: 1, count: bonds.count))
    if dropHydrogen, let k = m.atoms.lastIndex(where: { $0.element == "H" }) {
        m.atoms.remove(at: k)
        let keep: [Int] = m.bonds.indices.filter { m.bonds[$0].0 != k && m.bonds[$0].1 != k }
        m.orders = keep.map { m.orders[$0] }
        m.bonds = keep.map { b -> (Int, Int) in
            let (x, y) = m.bonds[b]
            return (x > k ? x - 1 : x, y > k ? y - 1 : y)
        }
    }
    return m
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

struct LeafChemistry {
    var alkane: Molecule

    /// Everything the molecule inset draws, in its own ångströms.
    var all: Molecule { alkane }
}

/// The alkane laid across the inset's diameter, tipped a little so its
/// zig-zag and hydrogens read. Turn MODEL.
func buildChemistry(resources: String, mutant: Mutant) throws -> LeafChemistry {
    let raw: Molecule = buildAlkane(carbons: waxCarbons, dropHydrogen: mutant == .alkaneFormula)
    let turn: simd_quatf = simd_quatf(angle: -0.42, axis: SIMD3<Float>(0, 0, 1)) * simd_quatf(angle: 0.55, axis: SIMD3<Float>(1, 0, 0))
    return LeafChemistry(alkane: raw.placed(turn: turn, at: SIMD3(0, -0.8, 0), view: 0))
}

// MARK: - odour molecules for the smell hair

/// One odour molecule in the micrometre inset, drawn as a dot: where it is,
/// and the wall pore it is heading for.
struct Volatile {
    var position: SIMD3<Float>   // µm
    var pore: (row: Int, column: Int)
}

/// An intact leaf gives off almost nothing (Food.swift), so no odour
/// molecules come to the smell hair. The code that would place them is kept
/// so the `odour` mutant can switch them on and the tests must object.
func buildVolatiles(hairs: [Sensillum], mutant: Mutant, facing view: SIMD3<Float>) -> [Volatile] {
    guard objectHasOdour || mutant == .odour else { return [] }
    let smell: Sensillum = hairs[1]
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
        let path: SIMD3<Float> = simd_normalize(outward + SIMD3<Float>(0, -0.9, 0))
        out.append(Volatile(position: mouth + path * p.away, pore: (p.row, j)))
    }
    return out
}

/// Dot radius for an odour molecule in the inset, µm (used only by the
/// mutant: there are none).
let volatileDotRadius: Float = 0.10
