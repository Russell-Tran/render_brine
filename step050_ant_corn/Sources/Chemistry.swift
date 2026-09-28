// Step 50: step 43's chemistry, copied, with the taste inset changed from
// step 43's still salt and norbixin to step 35's moving sugars: glucose, the
// largest of sweet corn's sugars (Food.swift), drifting into the taste pore
// one per touch. The odorant is 1-octen-3-ol, turning rigidly once a loop,
// and the odour dots at the smell hair travel as step 43's do.
//
// Organic molecules from real structure files, not drawn.

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

/// One water, O at `o`, turned by `turn` (step 35's).
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

// Glucose in solution is mostly a ring, and mostly β-D-glucopyranose (about
// 64%, the rest α — Wikipedia, "Glucose", as step 34 took it): drawn from the
// RCSB Chemical Component Dictionary's ideal coordinates for BGC (β-D-
// glucose), hydrogens included — step 35's file, copied verbatim into
// Resources/.
//
// 1-Octen-3-ol: PubChem CID 18827 ("oct-1-en-3-ol", C8H16O), its computed
// 3-D conformer with hydrogens, downloaded 2026-09-27 and copied verbatim
// into Resources/. The CID is the compound without its stereocentre fixed;
// the conformer is one hand of it (which hand sweet corn makes was not
// checked, and the label does not say). Turns and positions MODEL — the
// molecules tumble.

/// The molecules as loaded: glucose (whole, as its file has it) and the
/// odorant, placed in the smell inset.
struct CornChemistry {
    var glucose: Molecule
    var odorant: Molecule

    /// Step 43's: the odorant turned by `spin` about its own centre — a
    /// rigid turn, so every bond and angle is kept.
    func odorantTurned(_ spin: simd_quatf) -> Molecule {
        var c = SIMD3<Float>(0, 0, 0)
        for a in odorant.atoms { c += a.position }
        c /= Float(max(odorant.atoms.count, 1))
        var out: Molecule = odorant
        out.atoms = odorant.atoms.map { Atom(element: $0.element, position: spin.act($0.position - c) + c, view: $0.view) }
        return out
    }
}

func buildChemistry(resources: String, mutant: Mutant) throws -> CornChemistry {
    var glc: Molecule = try loadMolfile(resources + "/glucose_BGC_ideal.sdf")
    let oct: Molecule = try loadMolfile(resources + "/1-octen-3-ol_CID18827_3d.sdf")
    if mutant == .formula {
        // Step 35's formula mutant: drop one hydroxyl hydrogen — C6H11O6,
        // not a sugar, not neutral.
        if let k = glc.atoms.lastIndex(where: { $0.element == "H" }) {
            glc.atoms.remove(at: k)
            let keep: [Int] = glc.bonds.indices.filter { glc.bonds[$0].0 != k && glc.bonds[$0].1 != k }
            glc.orders = keep.map { glc.orders[$0] }
            glc.bonds = keep.map { b -> (Int, Int) in
                let (x, y) = glc.bonds[b]
                return (x > k ? x - 1 : x, y > k ? y - 1 : y)
            }
        }
    }
    let o: Molecule = alignLongAxis(oct).placed(turn: simd_quatf(angle: 0.6, axis: simd_normalize(SIMD3<Float>(1, 0.3, 0.2))),
                                                at: SIMD3(0, -0.3, 0), view: 1)
    return CornChemistry(glucose: glc, odorant: o)
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

// MARK: - glucose on its way into the taste pore

/// The juice has about 220 waters per glucose (Food.swift); each glucose is
/// drawn with three of them, at step 35's three spots round its glucose,
/// and the label says so. MODEL: the waters are not bound to one sugar; they
/// travel with it here so that none appears or vanishes.
let glucoseWaterSpots: [SIMD3<Float>] = [SIMD3(-3.4, -3.0, 4.0), SIMD3(3.2, 5.6, 2.0), SIMD3(-4.6, 1.4, 5.3)]
/// The glucose's own turn, step 34's.
let glucoseTurn = simd_quatf(angle: 2.1, axis: simd_normalize(SIMD3<Float>(1, 0.3, -0.4)))

/// One glucose and its waters, moving as one rigid body.
struct SugarUnit {
    var sugar: Molecule
    var waters: [Molecule]
    var molecules: [Molecule] { [sugar] + waters }
}

/// A glucose and its waters, all turned by `turn` about the glucose's
/// centroid and moved to `centre`. Step 35's sugarUnit, glucose only.
func sugarUnit(chemistry: CornChemistry, turn: simd_quatf, centre: SIMD3<Float>) -> SugarUnit {
    let sugar: Molecule = chemistry.glucose.placed(turn: turn * glucoseTurn, at: centre, view: 0)
    let waters: [Molecule] = glucoseWaterSpots.enumerated().map { k, p in
        let n: Float = Float(2 + k)
        let axisY: Float = n - 2
        let wTurn = simd_quatf(angle: 0.8 + 1.1 * n, axis: simd_normalize(SIMD3<Float>(1, axisY, 0.5)))
        return waterMolecule(at: centre + turn.act(p), turn: turn * wTurn, view: 0)
    }
    return SugarUnit(sugar: sugar, waters: waters)
}

/// Everything both molecule insets draw at one moment: the glucose drifting
/// into the taste pore, and the odorant, turned.
struct CornScene {
    var units: [SugarUnit]
    var odorant: Molecule

    var all: Molecule { merge(units.flatMap { $0.molecules } + [odorant]) }
    var sugars: [Molecule] { units.map { $0.sugar } }
    var waters: [Molecule] { units.flatMap { $0.waters } }
}

/// The corn scene at `t` seconds: each glucose at its pose along the drift
/// path (Motion.swift) — all glucose, so a slot's relabelling after each
/// touch changes nothing drawn — and the odorant at its turn.
func buildCornScene(_ t: Float, chemistry: CornChemistry, direction d: SIMD2<Float>, mutant: Mutant) -> CornScene {
    let poses: [MoleculePose] = moleculePoses(t, mutant: mutant, direction: d)
    let units: [SugarUnit] = poses.map { sugarUnit(chemistry: chemistry, turn: $0.rotation, centre: $0.offset) }
    return CornScene(units: units, odorant: chemistry.odorantTurned(odorantSpin(t, mutant: mutant)))
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
/// approach — out from the pore and down, towards the food it came from
/// (see odourPoint for the last stretch).
struct OdourPath {
    var pore: (row: Int, column: Int)
    var mouth: SIMD3<Float>
    var outward: SIMD3<Float>
    var direction: SIMD3<Float>  // unit, from the mouth out along the path
    var offset: Float            // where in its life this one is at progress 0
}

// Step 43's odour paths, unchanged: the odour keeps arriving. Four molecules, each making one trip per
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

/// Dot radius for an odour molecule in the inset, µm. A 1-octen-3-ol
/// molecule is under a nanometre across; at this scale it would be
/// invisible, so it is drawn several hundred times too big, and the label
/// says so.
let volatileDotRadius: Float = 0.10
