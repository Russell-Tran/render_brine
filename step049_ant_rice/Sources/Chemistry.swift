// Step 49: the chemistry in the insets. At the taste pore, what a grain of
// rice is made of — starch, drawn as a short piece of amylose, its straight
// chain of α-1,4-linked glucose — labelled as no taste: nothing from a dry
// grain enters the pore. In the air by the smell hair, a faint trace of
// hexanal. Both from real structure files, not drawn: PubChem 3-D conformers
// with their hydrogens, copied verbatim into Resources/.

import Foundation
import simd

// MARK: - atoms and molecules

struct Atom {
    var element: String          // "C", "O", "H", "Na", "Cl"
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

// Amylose, starch's unbranched chain: glucose units joined α-1,4. A whole
// chain is far too long to draw; the inset shows four of its units, as
// α-maltotetraose — PubChem CID 446495, "Alpha-maltotetraose", C24H42O21,
// the 3-D conformer with its hydrogens. It stands in for a piece of the
// chain, not as a free sugar: free sugars are 0.12 g per 100 g of raw white
// rice (Food.swift). The conformer is PubChem's computed one, not amylose's
// helix. Its turn and place are MODEL.
//
// Hexanal: PubChem CID 6184's 3-D conformer, as in step 37. Turns and
// positions MODEL — the molecules tumble.

struct RiceChemistry {
    var amylose: Molecule
    var odorant: Molecule

    /// Everything the two molecule insets draw, in their own ångströms.
    var all: Molecule { merge([amylose, odorant]) }

    /// The odorant turned by `spin` about its own centre — a rigid turn, so
    /// every bond and angle is kept.
    func odorantTurned(_ spin: simd_quatf) -> Molecule {
        var c = SIMD3<Float>(0, 0, 0)
        for a in odorant.atoms { c += a.position }
        c /= Float(max(odorant.atoms.count, 1))
        var out: Molecule = odorant
        out.atoms = odorant.atoms.map { Atom(element: $0.element, position: spin.act($0.position - c) + c, view: $0.view) }
        return out
    }

    /// Both insets at one moment: the taste inset still, the odorant turned.
    func all(spin: simd_quatf) -> Molecule { merge([amylose, odorantTurned(spin)]) }
}

func buildChemistry(resources: String, mutant: Mutant) throws -> RiceChemistry {
    let g4: Molecule = try loadMolfile(resources + "/alpha_maltotetraose_CID446495_3d.sdf")
    let hx: Molecule = try loadMolfile(resources + "/hexanal_CID6184_3d.sdf")
    // The chain across the inset, its long axis along x, a little below the
    // middle so the labels above it stay clear.
    let a: Molecule = alignLongAxis(g4).placed(turn: simd_quatf(angle: 0.35, axis: SIMD3<Float>(1, 0, 0)),
                                               at: SIMD3(0, -2.0, 0), view: 0)
    let o: Molecule = alignLongAxis(hx).placed(turn: simd_quatf(angle: 0.6, axis: simd_normalize(SIMD3<Float>(1, 0.3, 0.2))),
                                               at: SIMD3(0, -0.3, 0), view: 1)
    return RiceChemistry(amylose: a, odorant: o)
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
/// how much of it is drawn (0 to 1), and the wall pore it is heading for.
struct Volatile {
    var position: SIMD3<Float>   // µm
    var drawn: Float
    var pore: (row: Int, column: Int)
    var life: Float              // 0 to 1 through its trip
    var radius: Float { volatileDotRadius * drawn }
}

/// Where one odour molecule travels: in to its pore's mouth along step 42's
/// approach — out from the pore and down, towards the grain it came from
/// (see odourPoint for the last stretch).
struct OdourPath {
    var pore: (row: Int, column: Int)
    var mouth: SIMD3<Float>
    var outward: SIMD3<Float>
    var direction: SIMD3<Float>  // unit, from the mouth out along the path
    var offset: Float            // where in its life this one is at progress 0
}

// The odour keeps arriving, faintly. Two molecules, each making one trip per
// four-tap loop to its own pore, half a loop apart — so one arrives every
// other tap (over cheese sauce, step 43 sent one every tap), and at the loop's
// end each is back where it started. A trip (its `life`, 0 to 1): it appears
// at the far end of its path, growing in from nothing over the first 6% (a
// drawing choice: odorant is all through the air, and the inset marks the
// ones about to arrive); it drifts in, slowing, until 80% of the way through
// its life it rests at the pore's mouth; then it shrinks into the pore, its
// centre following it in so the dot never enters the wall. All MODEL,
// schematic (Motion.swift).
let odourSlots: Int = 2
/// Taps one trip lasts: the whole loop.
let tripTaps: Int = tapsPerLoop
/// How far out a trip starts, and where it rests at the pore, µm. The rest
/// is step 42's 0.13 µm: dot surface 0.03 µm off the wall.
let odourPathLength: Float = 3.6
let odourRest: Float = 0.13
let odourArrive: Float = 0.80
let odourGrowIn: Float = 0.06

/// Two of step 42's odour pores, same approach. The `noOdour` mutant
/// removes them, and the tests must notice.
func buildOdourPaths(hairs: [Sensillum], mutant: Mutant, facing view: SIMD3<Float>) -> [OdourPath] {
    guard objectHasOdour && mutant != .noOdour else { return [] }
    let smell: Sensillum = hairs[1]
    let want: SIMD3<Float> = simd_normalize(-view + SIMD3<Float>(-0.4, -0.5, 0))
    var bestColumn: Int = 0
    var bestDot: Float = -2
    for j in 0..<smell.poreColumns {
        let d: Float = simd_dot(smell.wallPore(row: 6, column: j).outward, want)
        if d > bestDot { bestDot = d; bestColumn = j }
    }
    var out: [OdourPath] = []
    let plan: [(row: Int, dj: Int)] = [(5, 0), (11, 1)]
    for (k, p) in plan.enumerated() {
        let j: Int = (bestColumn + p.dj + smell.poreColumns) % smell.poreColumns
        let (mouth, outward) = smell.wallPore(row: p.row, column: j)
        let path: SIMD3<Float> = simd_normalize(outward + SIMD3<Float>(0, -0.9, 0))
        // At progress 0 the lives are 0.80 and 0.30.
        let offset: Float = odourArrive - 0.5 * Float(k)
        out.append(OdourPath(pore: (p.row, j), mouth: mouth, outward: outward, direction: path, offset: offset))
    }
    return out
}

/// Distance out from the mouth, and how much is drawn, at `life`.
func odourTrip(life x: Float) -> (away: Float, drawn: Float) {
    if x < odourArrive {
        let s: Float = x / odourArrive
        let left: Float = 1 - s
        let ease: Float = 1 - left * left.squareRoot()          // 1 − (1 − s)^1.5: in steadily, slowing to rest
        let away: Float = odourPathLength + (odourRest - odourPathLength) * ease
        return (away, smoothstep01(x / odourGrowIn))
    }
    let drawn: Float = 1 - smoothstep01((x - odourArrive) / (1 - odourArrive))
    return (odourRest * drawn, drawn)
}

/// Where a dot is when it is `away` µm out along its path. Far out, the path
/// is step 42's line; over the last 1.2 µm it swings round to come straight
/// out of the pore, so a dot at rest (0.13 µm out, 0.10 µm in radius) sits
/// 0.03 µm clear of the wall, and one shrinking into the pore never touches
/// it — step 42's slanted line brought its resting dots 0.03 µm into the
/// curved wall beside the pore.
func odourPoint(_ p: OdourPath, away: Float) -> SIMD3<Float> {
    let bend: Float = smoothstep01(away / 1.2)
    let swing: SIMD3<Float> = (p.direction - p.outward) * bend
    let d: SIMD3<Float> = simd_normalize(p.outward + swing)
    return p.mouth + d * away
}

/// The odour dots at `progress` taps (Motion.swift).
func volatiles(_ paths: [OdourPath], progress u: Float) -> [Volatile] {
    paths.map { p in
        let raw: Float = u / Float(tripTaps) + p.offset
        let life: Float = raw - raw.rounded(.down)
        let (away, drawn) = odourTrip(life: life)
        return Volatile(position: odourPoint(p, away: away), drawn: drawn, pore: p.pore, life: life)
    }
}

/// Dot radius for an odour molecule in the inset, µm. A hexanal
/// molecule is under a nanometre across; at this scale it would be
/// invisible, so it is drawn several hundred times too big, and the label
/// says so.
let volatileDotRadius: Float = 0.10
