// Step 35: step 34's chemistry, copied. The sugars now drift into the taste
// pore, one per touch, each carrying its share of honey's water; the odour
// dots drift in to the smell hair's wall pores (see Motion.swift for when).
//
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

/// The molecules as loaded: the two sugars (each whole, as its file has it)
/// and the odorant, placed in the smell inset, where it holds still.
struct HoneyChemistry {
    var fructose: Molecule
    var glucose: Molecule
    var odorant: Molecule
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
    let o: Molecule = pa.placed(turn: simd_quatf(angle: 0.7, axis: simd_normalize(SIMD3<Float>(1, 0.4, 0.2))),
                                at: SIMD3(0, -0.3, 0), view: 1)
    return HoneyChemistry(fructose: fru, glucose: glc, odorant: o)
}

// MARK: - sugars on their way into the taste pore

enum Sugar: Int {
    case fructose = 0
    case glucose = 1
}

/// Honey has about 2.5 waters per fructose or glucose (Food.swift). Each
/// fructose here carries two waters and each glucose three, so every
/// fructose–glucose pair carries step 34's five and the inset always holds
/// honey's own ratio. Where the waters sit round their sugar is step 34's five
/// water spots, measured from the sugar each was nearer to. MODEL, as there:
/// in honey the waters are not bound to one sugar; they travel with it here
/// only so the ratio holds in every frame.
let fructoseWaterSpots: [SIMD3<Float>] = [SIMD3(4.0, 3.6, 1.5), SIMD3(-3.2, -4.8, 1.0)]
let glucoseWaterSpots: [SIMD3<Float>] = [SIMD3(-3.4, -3.0, 4.0), SIMD3(3.2, 5.6, 2.0), SIMD3(-4.6, 1.4, 5.3)]
/// Each sugar's own turn, step 34's.
let fructoseTurn = simd_quatf(angle: 0.9, axis: simd_normalize(SIMD3<Float>(0.3, 1, 0.2)))
let glucoseTurn = simd_quatf(angle: 2.1, axis: simd_normalize(SIMD3<Float>(1, 0.3, -0.4)))

/// One sugar and its waters, moving as one rigid body.
struct SugarUnit {
    var kind: Sugar
    var sugar: Molecule
    var waters: [Molecule]
    var molecules: [Molecule] { [sugar] + waters }
}

/// A sugar and its waters, all turned by `turn` about the sugar's centroid
/// and moved to `centre`.
func sugarUnit(_ kind: Sugar, chemistry: HoneyChemistry, turn: simd_quatf, centre: SIMD3<Float>) -> SugarUnit {
    let own: simd_quatf = kind == .fructose ? fructoseTurn : glucoseTurn
    let file: Molecule = kind == .fructose ? chemistry.fructose : chemistry.glucose
    let sugar: Molecule = file.placed(turn: turn * own, at: centre, view: 0)
    let spots: [SIMD3<Float>] = kind == .fructose ? fructoseWaterSpots : glucoseWaterSpots
    let first: Int = kind == .fructose ? 0 : fructoseWaterSpots.count
    let waters: [Molecule] = spots.enumerated().map { k, p in
        let n: Float = Float(first + k)
        let wTurn = simd_quatf(angle: 0.8 + 1.1 * n, axis: simd_normalize(SIMD3<Float>(1, n - 2, 0.5)))
        return waterMolecule(at: centre + turn.act(p), turn: turn * wTurn, view: 0)
    }
    return SugarUnit(kind: kind, sugar: sugar, waters: waters)
}

/// Everything both molecule insets draw at one moment: the sugars drifting
/// into the taste pore, and the odorant.
struct HoneyScene {
    var units: [SugarUnit]
    var odorant: Molecule

    var all: Molecule { merge(units.flatMap { $0.molecules } + [odorant]) }
    var sugars: [Molecule] { units.map { $0.sugar } }
    var waters: [Molecule] { units.flatMap { $0.waters } }
}

/// The honey scene at `t` seconds: the sugars at their poses along the drift
/// path (Motion.swift). A sugar's kind goes with it, not with its slot: the
/// one in slot k during tap n is sugar number k − n, and even numbers are
/// fructose, odd glucose — so after one touch each unit stands where the one
/// ahead of it stood, kind and all.
func buildHoneyScene(_ t: Float, chemistry: HoneyChemistry, direction d: SIMD2<Float>, mutant: Mutant) -> HoneyScene {
    let poses: [MoleculePose] = moleculePoses(t, mutant: mutant, direction: d)
    let tap: Int = Int(sugarProgress(t, mutant: mutant).rounded(.down))
    let units: [SugarUnit] = poses.enumerated().map { k, pose in
        let n: Int = k - tap
        let kind: Sugar = ((n % 2) + 2) % 2 == 0 ? .fructose : .glucose
        return sugarUnit(kind, chemistry: chemistry, turn: pose.rotation, centre: pose.offset)
    }
    return HoneyScene(units: units, odorant: chemistry.odorant)
}

// MARK: - odour molecules for the smell hair

/// One odour molecule in the micrometre inset, drawn as a dot: where it is,
/// how big it is drawn (it shrinks away as it goes into its pore), and the
/// wall pore it is heading for.
struct Volatile {
    var position: SIMD3<Float>   // µm
    var radius: Float            // µm; 0 once it is in
    var pore: (row: Int, column: Int)
    var lane: Int
    var along: Float             // how far along its lane, 0 … odourSlots
}

/// Dot radius for an odour molecule in the inset, µm. A phenylacetaldehyde
/// molecule is under a nanometre across; at this scale it would be
/// invisible, so it is drawn several hundred times too big, and the label
/// says so.
let volatileDotRadius: Float = 0.10

/// Step 35: the odour keeps coming. Two of step 34's pores — rows 5 and 11 of
/// the columns facing the camera — each take a stream of dots, one a tap,
/// half a tap apart. A dot drifts in through the air from beyond the right of
/// the circle, comes to its pore along the pore's own outward line, rests at
/// the mouth, and goes in. Honey's vapour is all round the hair, so where the
/// dots come from is a drawing choice: from the side, so none appears out of
/// nothing and none has to rise out of honey that the lifted hair has left
/// far below. Paths and pace MODEL, schematic (see Motion.swift).
struct OdourLane {
    var pore: (row: Int, column: Int)
    var mouth: SIMD3<Float>
    var outward: SIMD3<Float>
    var start: SIMD3<Float>
    var bend1: SIMD3<Float>
    var bend2: SIMD3<Float>
    var end: SIMD3<Float>
}

/// Dots drawn in each lane at once: in the air, at the pore, going in.
let odourSlots: Int = 3
/// Along a lane (in taps): drifting in until 2.2, at the mouth until 2.6,
/// then into the pore until 3, shrinking to nothing as it goes.
let odourArrive: Float = 2.2
let odourEnter: Float = 2.6
/// How far out of the pore mouth a dot rests, and how deep it goes, µm.
let odourRest: Float = 0.13
let odourDepth: Float = 0.08

func odourLanes(hairs: [Sensillum], camera: OrthoCamera, field: Float) -> [OdourLane] {
    let smell: Sensillum = hairs[1]
    // The column whose pores look most nearly at the camera and down to the
    // honey — the pores we can see molecules arrive at. Step 34's.
    let want: SIMD3<Float> = simd_normalize(-camera.forward + SIMD3<Float>(-0.4, -0.5, 0))
    var bestColumn: Int = 0
    var bestDot: Float = -2
    for j in 0..<smell.poreColumns {
        let d: Float = simd_dot(smell.wallPore(row: 6, column: j).outward, want)
        if d > bestDot { bestDot = d; bestColumn = j }
    }
    let half: Float = field / 2
    // Each lane: its pore (row, column step), where it enters the circle's
    // right edge (screen height, in half-fields), and how far in front.
    let plan: [(row: Int, dj: Int, height: Float, front: Float)] = [(5, 0, 0.42, 3.0), (11, 1, -0.02, 2.0)]
    return plan.map { p in
        let j: Int = (bestColumn + p.dj + smell.poreColumns) % smell.poreColumns
        let (mouth, outward) = smell.wallPore(row: p.row, column: j)
        let across: SIMD3<Float> = camera.right * (1.12 * half)
        let upward: SIMD3<Float> = camera.up * (p.height * half)
        let edge: SIMD3<Float> = across + upward
        let start: SIMD3<Float> = camera.centre + edge - camera.forward * p.front
        let bend1: SIMD3<Float> = start - camera.right * 2.2
        let bend2: SIMD3<Float> = mouth + outward * 2.0
        return OdourLane(pore: (p.row, j), mouth: mouth, outward: outward, start: start, bend1: bend1, bend2: bend2,
                         end: mouth + outward * odourRest)
    }
}

/// A dot `along` taps into its lane: where it is and how big it is drawn.
/// Depends on nothing else, so each dot stands where the one ahead stood a
/// tap before.
func odourPose(_ lane: OdourLane, along s: Float) -> (position: SIMD3<Float>, radius: Float) {
    if s < odourArrive {
        // A cubic Bézier, eased so the dot slows as it comes in.
        let b: Float = s / odourArrive
        let c: Float = 1 - (1 - b) * (1 - b)
        let d: Float = 1 - c
        let p0: SIMD3<Float> = lane.start * (d * d * d)
        let w1: Float = 3 * d * d * c   // typed apart: the mini's compiler took 8 ms inline
        let w2: Float = 3 * d * c * c
        let p1: SIMD3<Float> = lane.bend1 * w1
        let p2: SIMD3<Float> = lane.bend2 * w2
        let p3: SIMD3<Float> = lane.end * (c * c * c)
        return (p0 + p1 + p2 + p3, volatileDotRadius)
    }
    if s < odourEnter { return (lane.end, volatileDotRadius) }
    let x: Float = smoothstep01((s - odourEnter) / (Float(odourSlots) - odourEnter))
    let out: Float = odourRest - (odourRest + odourDepth) * x
    return (lane.mouth + lane.outward * out, volatileDotRadius * (1 - x))
}

/// Honey has a smell (Food.swift): the dots in every lane at progress `u`
/// (taps, Motion.swift). The `volatiles` mutant removes them all, and the
/// tests must notice.
func buildVolatiles(lanes: [OdourLane], progress u: Float, mutant: Mutant) -> [Volatile] {
    guard foodHasVapour && mutant != .volatiles else { return [] }
    var out: [Volatile] = []
    for (l, lane) in lanes.enumerated() {
        let ul: Float = u + Float(l) / Float(lanes.count)
        let f: Float = ul - ul.rounded(.down)
        for k in 0..<odourSlots {
            let s: Float = Float(k) + f
            let (p, r) = odourPose(lane, along: s)
            out.append(Volatile(position: p, radius: r, pore: lane.pore, lane: l, along: s))
        }
    }
    return out
}
