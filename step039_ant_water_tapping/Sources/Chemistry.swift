// The chemistry in the inset at the taste pore: water, five molecules of it
// held together by hydrogen bonds. Built from water's measured geometry, not
// drawn by eye. There is no odorant inset: water has no smell.

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
    var hbonds: [(Int, Int)] = [] // hydrogen bonds (H, acceptor O): drawn dashed, not counted as valence

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
        return Molecule(atoms: moved, bonds: bonds, orders: orders, hbonds: hbonds)
    }
}

// Water: O–H 0.9572 Å, H–O–H 104.52° (Hoy & Bunker 1979, as given in
// Wikipedia "Properties of water").
let waterOH: Float = 0.9572
let waterAngle: Float = 104.52 * Float.pi / 180

/// A hydrogen bond in water, H···O: "The typical length of a hydrogen bond in
/// water is 197 pm" (Wikipedia, "Hydrogen bond"; the range there for X–H···Y
/// is 160–200 pm). Drawn straight — O–H···O at 180° — the usual idealisation;
/// in the liquid they bend and break within picoseconds, so this is one
/// instant, tidied.
let hydrogenBondLength: Float = 1.97

/// The tetrahedral angle, for where water's two lone pairs point. MODEL: the
/// textbook sp3 picture — the lone pairs are not atoms and have no measured
/// position; it is the standard way to aim an accepted hydrogen bond.
let tetrahedral: Float = acos(-1.0 / 3.0)

/// A water given by its oxygen and one direction it must present: either one
/// O–H bond (a donor, `lonePair` false) or one lone pair (an acceptor), with
/// `roll` turning the rest of it about that direction.
func water(at o: SIMD3<Float>, presenting d: SIMD3<Float>, lonePair: Bool, roll: Float,
           angle: Float = waterAngle, view: Int = 0) -> Molecule {
    let dn: SIMD3<Float> = simd_normalize(d)
    let helper: SIMD3<Float> = abs(dn.y) < 0.9 ? SIMD3(0, 1, 0) : SIMD3(1, 0, 0)
    let w0: SIMD3<Float> = simd_normalize(simd_cross(dn, helper))
    let w: SIMD3<Float> = simd_quatf(angle: roll, axis: dn).act(w0)
    let a: Float = angle / 2
    let bis: SIMD3<Float>
    let e: SIMD3<Float>
    if lonePair {
        // Lone pair L = −b cos β + n sin β, β half the tetrahedral angle.
        let beta: Float = tetrahedral / 2
        bis = -dn * cos(beta) + w * sin(beta)
        let n: SIMD3<Float> = dn * sin(beta) + w * cos(beta)
        e = simd_normalize(simd_cross(n, bis))
    } else {
        // First O–H along d: H = b cos α + e sin α.
        bis = dn * cos(a) - w * sin(a)
        e = dn * sin(a) + w * cos(a)
    }
    let h1: SIMD3<Float> = o + (bis * cos(a) + e * sin(a)) * waterOH
    let h2: SIMD3<Float> = o + (bis * cos(a) - e * sin(a)) * waterOH
    return Molecule(atoms: [Atom(element: "O", position: o, view: view), Atom(element: "H", position: h1, view: view),
                            Atom(element: "H", position: h2, view: view)],
                    bonds: [(0, 1), (0, 2)], orders: [1, 1])
}

/// The two lone-pair directions of a water: −b cos β ± n sin β.
func lonePairs(_ m: Molecule) -> [SIMD3<Float>] {
    let o: SIMD3<Float> = m.atoms[0].position
    let u1: SIMD3<Float> = simd_normalize(m.atoms[1].position - o)
    let u2: SIMD3<Float> = simd_normalize(m.atoms[2].position - o)
    let b: SIMD3<Float> = simd_normalize(u1 + u2)
    let n: SIMD3<Float> = simd_normalize(simd_cross(u1, u2))
    let beta: Float = tetrahedral / 2
    return [-b * cos(beta) + n * sin(beta), -b * cos(beta) - n * sin(beta)]
}

/// Joins molecules into one list, renumbering the bonds.
func merge(_ ms: [Molecule]) -> Molecule {
    var atoms: [Atom] = []
    var bonds: [(Int, Int)] = []
    var orders: [Int] = []
    var hb: [(Int, Int)] = []
    for m in ms {
        let base: Int = atoms.count
        atoms += m.atoms
        bonds += m.bonds.map { ($0.0 + base, $0.1 + base) }
        orders += m.bonds.indices.map { $0 < m.orders.count ? m.orders[$0] : 1 }
        hb += m.hbonds.map { ($0.0 + base, $0.1 + base) }
    }
    return Molecule(atoms: atoms, bonds: bonds, orders: orders, hbonds: hb)
}

// MARK: - the taste inset: five waters, hydrogen-bonded

// The arrangement: one water in the middle, as in ice and in much of the
// liquid, with four neighbours — two it gives its hydrogens to, two that
// give theirs to it. Every H···O is hydrogenBondLength, every O–H···O
// straight, every lone pair aimed tetrahedrally. The rolls about each bond
// are MODEL, chosen so no two molecules crowd each other.

/// How the cluster is turned for the camera. MODEL.
let clusterTurn = simd_quatf(angle: 1.6, axis: simd_normalize(SIMD3<Float>(0.6, 0.3, 1)))

struct WaterChemistry {
    var waters: [Molecule]
    var cluster: Molecule          // the five, merged, with the hydrogen bonds
    var odorant: Molecule?         // none: water has no smell

    /// Everything the molecule inset draws, in its own ångströms.
    var all: Molecule { cluster }
}

func buildChemistry(resources: String, mutant: Mutant) throws -> WaterChemistry {
    let angle: Float = mutant == .waterAngle ? 90 * Float.pi / 180 : waterAngle
    let reach: Float = waterOH + hydrogenBondLength      // O···O along a straight bond
    let centre: Molecule = water(at: .zero, presenting: SIMD3(1, 0.25, 0.1), lonePair: false, roll: 0.6, angle: angle)
    let o0: SIMD3<Float> = centre.atoms[0].position
    var ws: [Molecule] = [centre]
    var pairs: [(donorH: (Int, Int), acceptor: Int)] = []   // ((water, atom), water)
    // Two acceptors, one on each of the centre's hydrogens.
    for (k, hi) in [1, 2].enumerated() {
        let u: SIMD3<Float> = simd_normalize(centre.atoms[hi].position - o0)
        let w: Molecule = water(at: o0 + u * reach, presenting: -u, lonePair: true, roll: 1.1 + 2.0 * Float(k), angle: angle)
        ws.append(w)
        pairs.append(((0, hi), ws.count - 1))
    }
    // Two donors, one on each of the centre's lone pairs.
    for (k, l) in lonePairs(centre).enumerated() {
        let w: Molecule = water(at: o0 + l * reach, presenting: -l, lonePair: false, roll: 0.4 + 2.3 * Float(k), angle: angle)
        ws.append(w)
        pairs.append(((ws.count - 1, 1), 0))
    }
    var cluster: Molecule = merge(ws)
    cluster.hbonds = pairs.map { p in (p.donorH.0 * 3 + p.donorH.1, p.acceptor * 3) }
    // Turn it to face the camera nicely, and centre it a little high in the
    // inset, under its label. Turn MODEL.
    let placedCluster: Molecule = cluster.placed(turn: clusterTurn, at: SIMD3(0, -0.9, 0), view: 0)
    var placedWaters: [Molecule] = []
    for k in 0..<ws.count {
        let lo: Int = 3 * k
        let three: [Atom] = Array(placedCluster.atoms[lo..<(lo + 3)])
        placedWaters.append(Molecule(atoms: three, bonds: [(0, 1), (0, 2)], orders: [1, 1]))
    }
    return WaterChemistry(waters: placedWaters, cluster: placedCluster, odorant: nil)
}

// MARK: - odour molecules for the smell hair

/// One odour molecule in the micrometre inset, drawn as a dot: where it is,
/// and the wall pore it is heading for.
struct Volatile {
    var position: SIMD3<Float>   // µm
    var pore: (row: Int, column: Int)
}

/// Water has no smell (Food.swift), so no odour molecules come to the smell
/// hair. The code that would place them is kept — step 34's, unchanged — so
/// that the `odour` mutant can switch them on and the tests must object.
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
