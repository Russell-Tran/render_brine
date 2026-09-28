// Step 68: step 50's chemistry, copied, with the apple's molecules: the
// taste inset shows FRUCTOSE — Golden Delicious's most abundant sugar
// (Food.swift) — drifting into ant 0's taste pore one per touch; the smell
// inset shows the cut's green note, 2-hexenal, and the cultivar's ester,
// butyl acetate, each turning rigidly once a loop; the odour dots at the
// smell hair travel as step 50's do.
//
// Molecules from real structure files, not drawn.

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
/// their orders. The odorant's file carries its own hydrogens, so nothing is
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

// β-D-Fructopyranose: PubChem CID 24310 ("Beta-D-Fructopyranose", C6H12O6,
// (2R,3S,4R,5R)-2-(hydroxymethyl)oxane-2,3,4,5-tetrol), its computed 3-D
// conformer with hydrogens, downloaded 2026-09-28 and copied verbatim into
// Resources/. The ring form dissolved fructose mostly takes (Food.swift).
// (E)-Hex-2-enal: CID 5281168 ("trans-2-Hexenal", C6H10O). Butyl acetate:
// CID 31272 (C6H12O2). Both computed conformers with hydrogens, downloaded
// the same day. Turns and positions MODEL — the molecules tumble.

/// The molecules as loaded: fructose, and the two odorants placed side by
/// side in the smell inset.
struct AppleChemistry {
    var fructose: Molecule
    var hexenal: Molecule
    var butylAcetate: Molecule

    /// Both odorants, each turned by `spin` about its own centre — rigid
    /// turns, so every bond and angle is kept.
    func odorantsTurned(_ spin: simd_quatf) -> Molecule {
        func turned(_ m: Molecule) -> Molecule {
            var c = SIMD3<Float>(0, 0, 0)
            for a in m.atoms { c += a.position }
            c /= Float(max(m.atoms.count, 1))
            var out: Molecule = m
            out.atoms = m.atoms.map { Atom(element: $0.element, position: spin.act($0.position - c) + c, view: $0.view) }
            return out
        }
        return merge([turned(hexenal), turned(butylAcetate)])
    }
}

/// Where each odorant sits in its inset, Å: the green note above, the ester
/// below. MODEL, for the layout.
let hexenalCentre = SIMD3<Float>(0, 4.0, 0)
let butylAcetateCentre = SIMD3<Float>(0, -2.6, 0)

func buildChemistry(resources: String, mutant: Mutant) throws -> AppleChemistry {
    var fru: Molecule = try loadMolfile(resources + "/beta-D-fructopyranose_CID24310_3d.sdf")
    let hex: Molecule = try loadMolfile(resources + "/E-2-hexenal_CID5281168_3d.sdf")
    let but: Molecule = try loadMolfile(resources + "/butyl_acetate_CID31272_3d.sdf")
    if mutant == .formula {
        // The formula mutant: drop one hydroxyl hydrogen — C6H11O6, not
        // fructose, not neutral.
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
    let h: Molecule = alignLongAxis(hex).placed(turn: simd_quatf(angle: 0.5, axis: simd_normalize(SIMD3<Float>(1, 0.3, 0.2))),
                                                at: hexenalCentre, view: 1)
    let b: Molecule = alignLongAxis(but).placed(turn: simd_quatf(angle: -0.4, axis: simd_normalize(SIMD3<Float>(0.2, 1, 0.3))),
                                                at: butylAcetateCentre, view: 1)
    return AppleChemistry(fructose: fru, hexenal: h, butylAcetate: b)
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

// MARK: - fructose on its way into the taste pore

/// Each fructose's own turn in the inset. MODEL: chosen so the ring shows.
let fructoseTurn = simd_quatf(angle: 1.2, axis: simd_normalize(SIMD3<Float>(1, 0.3, -0.4)))

/// One fructose, moving as a rigid body. No waters are drawn with it: the
/// inset's pale ground stands for the juice's water, as steps 30 and 50.
struct SugarUnit {
    var sugar: Molecule
    var molecules: [Molecule] { [sugar] }
}

/// A fructose turned by `turn` about its centroid and moved to `centre`.
func sugarUnit(chemistry: AppleChemistry, turn: simd_quatf, centre: SIMD3<Float>) -> SugarUnit {
    SugarUnit(sugar: chemistry.fructose.placed(turn: turn * fructoseTurn, at: centre, view: 0))
}

/// Everything both molecule insets draw at one moment.
struct AppleScene {
    var units: [SugarUnit]
    var odorants: Molecule

    var all: Molecule { merge(units.flatMap { $0.molecules } + [odorants]) }
    var sugars: [Molecule] { units.map { $0.sugar } }
}

/// The molecules at one moment: each fructose at its pose along the drift
/// path (Motion.swift) — all alike, so a slot's relabelling after each touch
/// changes nothing drawn — and the odorants at their turn.
func buildAppleScene(progress u: Float, wobblePhase: Float, spin: simd_quatf, chemistry: AppleChemistry,
                     direction d: SIMD2<Float>) -> AppleScene {
    let poses: [MoleculePose] = moleculePoses(progress: u, wobblePhase: wobblePhase, direction: d)
    let units: [SugarUnit] = poses.map { sugarUnit(chemistry: chemistry, turn: $0.rotation, centre: $0.offset) }
    return AppleScene(units: units, odorants: chemistry.odorantsTurned(spin))
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

// Step 43's odour paths, as step 50 kept them, with step 42's fifth pore
// back: the odour keeps arriving. Five molecules, each making one trip every
// five taps to its own pore, a tap apart — so one arrives every tap, and a
// ten-tap loop takes each round twice, back where it started. A trip (its `life`, 0 to
// 1): it appears at the far end of its path, growing in from nothing over the
// first 6% (a drawing choice, as step 33's waters grew in: odorant is all
// through the air, and the inset marks the ones about to arrive); it drifts
// in, slowing, until 80% of the way through its life it rests at the pore's
// mouth; then it shrinks into the pore, its centre following it in so the
// dot never enters the wall. At progress 0 — the touch-down, the still's
// moment — one rests at its pore, the others on their way. All MODEL,
// schematic (Motion.swift).
let odourSlots: Int = 5
/// How far out a trip starts, and where it rests at the pore, µm. The rest
/// is step 42's 0.13 µm: dot surface 0.03 µm off the wall.
let odourPathLength: Float = 3.6
let odourRest: Float = 0.13
let odourArrive: Float = 0.80
let odourGrowIn: Float = 0.06

/// Step 42's odour, as paths: its five pores, same approach. The
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
    let plan: [(row: Int, dj: Int)] = [(5, 0), (11, 1), (3, -1), (13, 1), (8, -1)]
    for (k, p) in plan.enumerated() {
        let j: Int = (bestColumn + p.dj + smell.poreColumns) % smell.poreColumns
        let (mouth, outward) = smell.wallPore(row: p.row, column: j)
        let path: SIMD3<Float> = simd_normalize(outward + SIMD3<Float>(0, -0.9, 0))
        // At progress 0 the lives are 0.80, 0.60, 0.40, 0.20, 0.00.
        let spacing: Float = 1 / Float(odourSlots)
        let offset: Float = odourArrive - spacing * Float(k)
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

/// Dot radius for an odour molecule in the inset, µm. A 2-hexenal or butyl
/// acetate molecule is under a nanometre across; at this scale it would be
/// invisible, so it is drawn several hundred times too big, and the label
/// says so.
let volatileDotRadius: Float = 0.10
