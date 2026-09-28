// Step 43: step 42's chemistry, copied. The odorant in its inset now turns,
// rigidly, and the odour dots at the smell hair travel: one reaches its pore
// and goes in every tap (see the odour molecules, below).
//
// The chemistry in the insets: at the taste pore, the salt that is there to
// taste — Na⁺ and Cl⁻ in water — and norbixin, the annatto molecule that
// makes the sauce orange; in the air by the smell hair, the cheese odorant
// butanoic acid. Organic molecules from real structure files, not drawn.

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

// Water: O–H 0.9572 Å, H–O–H 104.52° (Hoy & Bunker 1979, as given in
// Wikipedia "Properties of water").
let waterOH: Float = 0.9572
let waterAngle: Float = 104.52 * Float.pi / 180

/// One water: O at `o`, the H–O–H bisector along `bisector`, in the plane
/// containing `inPlane`.
func water(o: SIMD3<Float>, bisector: SIMD3<Float>, inPlane: SIMD3<Float>, view: Int) -> Molecule {
    let b: SIMD3<Float> = simd_normalize(bisector)
    let side: SIMD3<Float> = simd_normalize(inPlane - b * simd_dot(inPlane, b))
    let half: Float = waterAngle / 2
    let h1: SIMD3<Float> = o + (b * cos(half) + side * sin(half)) * waterOH
    let h2: SIMD3<Float> = o + (b * cos(half) - side * sin(half)) * waterOH
    return Molecule(atoms: [Atom(element: "O", position: o, view: view), Atom(element: "H", position: h1, view: view),
                            Atom(element: "H", position: h2, view: view)],
                    bonds: [(0, 1), (0, 2)], orders: [1, 1])
}

// The ions, as in step 32: Shannon's six-fold radii, Na⁺ 1.02 Å and Cl⁻
// 1.81 Å (*Acta Cryst A* 32: 751, 1976), drawn at 0.62 of those so the
// waters stay readable (a drawing choice); and each ion's first shell from
// Ohtaki & Radnai (*Chem Rev* 93: 1157, 1993): six waters, Na–O 2.43 Å with
// oxygen in, Cl–O 3.20 Å with one hydrogen pointing at the ion. Four of the
// six are drawn — the four square to the line of sight — so the ion is not
// hidden behind its own shell; the label says so.
let sodiumRadius: Float = 1.02
let chlorideRadius: Float = 1.81
let ionDrawScale: Float = 0.62
let sodiumWaterDistance: Float = 2.43
let chlorideWaterDistance: Float = 3.20

func hydratedIon(_ element: String, at c: SIMD3<Float>, spin: Float, view: Int) -> [Molecule] {
    let sodium: Bool = element == "Na"
    var out: [Molecule] = [Molecule(atoms: [Atom(element: element, position: c, view: view)], bonds: [])]
    let square: [SIMD3<Float>] = [SIMD3(1, 0, 0), SIMD3(0, 1, 0), SIMD3(-1, 0, 0), SIMD3(0, -1, 0)]
    let turn = simd_quatf(angle: spin, axis: SIMD3<Float>(0, 0, 1))
    for (k, d0) in square.enumerated() {
        let d: SIMD3<Float> = turn.act(d0)
        let helper: SIMD3<Float> = k % 2 == 0 ? SIMD3<Float>(0, 0, 1) : turn.act(square[(k + 1) % 4])
        if sodium {
            out.append(water(o: c + d * sodiumWaterDistance, bisector: d, inPlane: helper, view: view))
        } else {
            let side: SIMD3<Float> = simd_normalize(helper - d * simd_dot(helper, d))
            let half: Float = waterAngle / 2
            let bis: SIMD3<Float> = -d * cos(half) + side * sin(half)
            out.append(water(o: c + d * chlorideWaterDistance, bisector: bis, inPlane: side, view: view))
        }
    }
    return out
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

// Norbixin: annatto's water-soluble colouring principle (EFSA 2019,
// Food.swift), drawn as its 9'-cis form — PubChem CID 6537492, "(2E,4E,6E,
// 8E,10E,12E,14E,16Z,18E)-4,8,13,17-tetramethylicosa-2,4,6,8,10,12,14,16,18-
// nonaenedioic acid", the 3-D conformer with its hydrogens, copied verbatim
// into Resources/. The cis bond is annatto's natural form (UNVERIFIED here:
// the EFSA abstract names bixin and norbixin without isomers); cooking turns
// some of it trans. Its nine conjugated C=C bonds are why it is coloured: a
// chain of alternating double bonds that long absorbs blue light.
//
// Butanoic acid: PubChem CID 264's 3-D conformer, also verbatim. Turns and
// positions MODEL — the molecules tumble.

struct MacChemistry {
    var norbixin: Molecule
    var ions: [Molecule]           // Na⁺ then Cl⁻, one atom each
    var waters: [Molecule]
    var odorant: Molecule

    /// Everything the two molecule insets draw, in their own ångströms.
    var all: Molecule { merge([norbixin] + ions + waters + [odorant]) }

    /// Step 43: the odorant turned by `spin` about its own centre — a rigid
    /// turn, so every bond and angle is kept.
    func odorantTurned(_ spin: simd_quatf) -> Molecule {
        var c = SIMD3<Float>(0, 0, 0)
        for a in odorant.atoms { c += a.position }
        c /= Float(max(odorant.atoms.count, 1))
        var out: Molecule = odorant
        out.atoms = odorant.atoms.map { Atom(element: $0.element, position: spin.act($0.position - c) + c, view: $0.view) }
        return out
    }

    /// Both insets at one moment: the taste inset still, the odorant turned.
    func all(spin: simd_quatf) -> Molecule {
        var parts: [Molecule] = [norbixin]
        parts.append(contentsOf: ions)
        parts.append(contentsOf: waters)
        parts.append(odorantTurned(spin))
        return merge(parts)
    }
}

func buildChemistry(resources: String, mutant: Mutant) throws -> MacChemistry {
    let nb: Molecule = try loadMolfile(resources + "/norbixin_9cis_CID6537492_3d.sdf")
    let ba: Molecule = try loadMolfile(resources + "/butanoic_acid_CID264_3d.sdf")
    // Norbixin across the upper half of the inset, its long axis along x.
    let n: Molecule = alignLongAxis(nb).placed(turn: simd_quatf(angle: 0.25, axis: SIMD3<Float>(1, 0, 0)),
                                               at: SIMD3(0, 2.4, 0), view: 0)
    // The salt below it, dissolved: each ion with four of its waters.
    let na: [Molecule] = hydratedIon("Na", at: SIMD3(-5.2, -6.2, 0), spin: 0.3, view: 0)
    let cl: [Molecule] = hydratedIon("Cl", at: SIMD3(5.6, -6.4, 0), spin: 0.5, view: 0)
    let o: Molecule = alignLongAxis(ba).placed(turn: simd_quatf(angle: 0.6, axis: simd_normalize(SIMD3<Float>(1, 0.3, 0.2))),
                                               at: SIMD3(0, -0.3, 0), view: 1)
    return MacChemistry(norbixin: n, ions: [na[0], cl[0]], waters: Array(na.dropFirst()) + Array(cl.dropFirst()), odorant: o)
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
/// approach — out from the pore and down, towards the sauce it came from
/// (see odourPoint for the last stretch).
struct OdourPath {
    var pore: (row: Int, column: Int)
    var mouth: SIMD3<Float>
    var outward: SIMD3<Float>
    var direction: SIMD3<Float>  // unit, from the mouth out along the path
    var offset: Float            // where in its life this one is at progress 0
}

// Step 43: the odour keeps arriving. Four molecules, each making one trip per
// four-tap loop to its own pore, a tap apart — so one arrives every tap, and
// at the loop's end each is back where it started. A trip (its `life`, 0 to
// 1): it appears at the far end of its path, growing in from nothing over the
// first 6% (a drawing choice, as step 33's waters grew in: odorant is all
// through the air, and the inset marks the ones about to arrive); it drifts
// in, slowing, until 80% of the way through its life it rests at the pore's
// mouth; then it shrinks into the pore, its centre following it in so the
// dot never enters the wall. At progress 0 — the touch-down, the still's
// moment — they stand close to where step 42 put its dots: one at its pore,
// the others 0.7, 1.8 and 3.3 µm out. All MODEL, schematic (Motion.swift).
let odourSlots: Int = 4
/// How far out a trip starts, and where it rests at the pore, µm. The rest
/// is step 42's 0.13 µm: dot surface 0.03 µm off the wall.
let odourPathLength: Float = 3.6
let odourRest: Float = 0.13
let odourArrive: Float = 0.80
let odourGrowIn: Float = 0.06

/// Step 42's odour, as paths: four of its five pores (not row 8), same
/// approach; the last is the one step 42 drew farthest out. The
/// `noOdour` mutant removes them, and the tests must notice.
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
    let plan: [(row: Int, dj: Int)] = [(5, 0), (11, 1), (3, -1), (13, 1)]
    for (k, p) in plan.enumerated() {
        let j: Int = (bestColumn + p.dj + smell.poreColumns) % smell.poreColumns
        let (mouth, outward) = smell.wallPore(row: p.row, column: j)
        let path: SIMD3<Float> = simd_normalize(outward + SIMD3<Float>(0, -0.9, 0))
        // At progress 0 the lives are 0.80, 0.55, 0.30, 0.05.
        let offset: Float = odourArrive - 0.25 * Float(k)
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
        let raw: Float = u / Float(odourSlots) + p.offset
        let life: Float = raw - raw.rounded(.down)
        let (away, drawn) = odourTrip(life: life)
        return Volatile(position: odourPoint(p, away: away), drawn: drawn, pore: p.pore, life: life)
    }
}

/// Dot radius for an odour molecule in the inset, µm. A butanoic acid
/// molecule is under a nanometre across; at this scale it would be
/// invisible, so it is drawn several hundred times too big, and the label
/// says so.
let volatileDotRadius: Float = 0.10
