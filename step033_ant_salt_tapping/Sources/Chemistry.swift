// Step 33: step 32's chemistry, copied; the face now has a step and a kink,
// and pairs leave it one per touch (see the crystal face, below).
//
// The chemistry in the insets: the rock-salt lattice at the crystal face, the
// ions leaving it into the film with their shells of water, and — for the
// smell hair — the list of odour molecules in the air, which for salt is empty.

import Foundation
import simd

// MARK: - atoms and ions

struct Atom {
    var element: String          // "Na", "Cl", "O", "H"
    var position: SIMD3<Float>   // Å
    var charge: Int = 0
    /// Step 33: how much of the atom is drawn, 0 to 1 — waters gathering
    /// round a leaving ion grow in. Where it is never changes with this.
    var drawn: Float = 1
}

struct Molecule {
    var atoms: [Atom]
    var bonds: [(Int, Int)]

    var formula: [String: Int] {
        var f: [String: Int] = [:]
        for a in atoms { f[a.element, default: 0] += 1 }
        return f
    }
}

// Shannon's effective ionic radii for six-fold coordination, the coordination
// both ions have in rock salt: Na⁺ 102 pm, Cl⁻ 181 pm (Shannon, *Acta Cryst
// A* 32: 751, 1976; the Wikipedia "Ionic radius" table). Their sum, 283 pm,
// is half the cell edge to within 0.4% — the hard-sphere picture of NaCl.
let sodiumRadius: Float = 1.02    // Å
let chlorideRadius: Float = 1.81  // Å

/// Ions are drawn at this fraction of their ionic radius, both the same, so
/// their ORDER and ratio stay Shannon's while the lattice and the water
/// around the loose ions stay readable. At full size the lattice is a wall
/// of touching spheres. MODEL (a drawing choice), and the label says so.
let ionDrawScale: Float = 0.62

// Water: O–H 0.9572 Å, H–O–H 104.52° (Hoy & Bunker 1979, as given in
// Wikipedia "Properties of water").
let waterOH: Float = 0.9572
let waterAngle: Float = 104.52 * Float.pi / 180

// A dissolved ion's first shell (Ohtaki & Radnai, *Chem Rev* 93: 1157, 1993,
// reviewing diffraction and simulation): Na⁺ holds about six waters with the
// oxygen towards it at Na–O ≈ 2.4–2.5 Å; Cl⁻ about six, each pointing one
// hydrogen at it, Cl–O ≈ 3.1–3.3 Å. Six, octahedral, and the mid-range
// distances are used; the exact number fluctuates in real water.
let sodiumWaterDistance: Float = 2.43
let chlorideWaterDistance: Float = 3.20
let hydrationNumber: Int = 6

/// One water molecule: O at `o`, its two hydrogens placed so the bisector of
/// H–O–H points along `bisector`, in the plane containing `inPlane`.
func water(o: SIMD3<Float>, bisector: SIMD3<Float>, inPlane: SIMD3<Float>) -> [Atom] {
    let b: SIMD3<Float> = simd_normalize(bisector)
    let side: SIMD3<Float> = simd_normalize(inPlane - b * simd_dot(inPlane, b))
    let half: Float = waterAngle / 2
    let h1: SIMD3<Float> = o + (b * cos(half) + side * sin(half)) * waterOH
    let h2: SIMD3<Float> = o + (b * cos(half) - side * sin(half)) * waterOH
    return [Atom(element: "O", position: o), Atom(element: "H", position: h1), Atom(element: "H", position: h2)]
}

let octahedron: [SIMD3<Float>] = [SIMD3(1, 0, 0), SIMD3(-1, 0, 0), SIMD3(0, 1, 0), SIMD3(0, -1, 0), SIMD3(0, 0, 1), SIMD3(0, 0, -1)]

/// A hydrated ion: the ion and six waters at the octahedron's corners,
/// turned by `turn`. Na⁺: oxygen in, hydrogens out. Cl⁻: one O–H pointing
/// straight at the ion.
func hydratedIon(_ element: String, at c: SIMD3<Float>, turn: simd_quatf) -> Molecule {
    let sodium: Bool = element == "Na"
    var atoms: [Atom] = [Atom(element: element, position: c, charge: sodium ? 1 : -1)]
    var bonds: [(Int, Int)] = []
    for (k, d0) in octahedron.enumerated() {
        let d: SIMD3<Float> = turn.act(d0)
        let helper: SIMD3<Float> = turn.act(octahedron[(k + 2) % 6])
        let base: Int = atoms.count
        if sodium {
            let o: SIMD3<Float> = c + d * sodiumWaterDistance
            atoms += water(o: o, bisector: d, inPlane: helper)
        } else {
            // One hydrogen on the Cl···O line: rotate the bisector off that
            // line by half the H–O–H angle.
            let o: SIMD3<Float> = c + d * chlorideWaterDistance
            let side: SIMD3<Float> = simd_normalize(helper - d * simd_dot(helper, d))
            let half: Float = waterAngle / 2
            let bis: SIMD3<Float> = -d * cos(half) + side * sin(half)
            atoms += water(o: o, bisector: bis, inPlane: side)
        }
        bonds += [(base, base + 1), (base, base + 2)]
    }
    return Molecule(atoms: atoms, bonds: bonds)
}

// MARK: - the crystal face

// Step 33: the face now dissolves while we watch, one Na⁺/Cl⁻ pair per touch.
// A crystal dissolves from its kinks: an ion at a kink on a surface step is
// held by the fewest neighbours, so that is where ions leave, and each pair
// that leaves moves the kink one cell edge along the step, leaving the step
// exactly as it was, shifted. So the face here carries one step (the top
// layer complete behind it, one layer lower in front) and one kink, and the
// view follows the kink as it retreats: after each touch the picture is the
// one before, which is what lets the loop run forward. The step, the kink and
// the view following it are MODEL — the textbook picture of dissolution, not
// a measured NaCl surface.

/// Rows of the face (along z), layers deep, and ions along each row. Rows
/// reach far past the circle on both sides, so the ions that the view gains
/// and loses as it follows the kink come and go well outside it.
let faceRows: Int = 8
let faceLayers: Int = 3
let rowReach: Int = 12            // ions r = −12 … 11 along every row
/// Top-layer rows j < stepRow are whole; row stepRow ends at the kink; rows
/// beyond it are the lower terrace, one layer down.
let stepRow: Int = 4
/// Where the kink sits along the row, in the face's own ångströms. The pair
/// that leaves next is at r = −1 and −2: Cl⁻ at +a/4, Na⁺ at −a/4.
let kinkOffset: Float = 1.5 * haliteCell / 2

/// A face ion (i along the row relative to the kink, j across, k down) at
/// `slide`, how far the view has followed the kink since the last pair left
/// (0 to 1: one cell edge, two ions).
func faceIon(_ r: Int, _ j: Int, _ k: Int, slide: Float) -> Atom {
    let s: Float = haliteCell / 2
    let na: Bool = (r + j + k) % 2 == 0
    let p = SIMD3<Float>((Float(r) + 2 * slide) * s + kinkOffset, -Float(k) * s, (Float(j) - Float(faceRows - 1) / 2) * s)
    return Atom(element: na ? "Na" : "Cl", position: p, charge: na ? 1 : -1)
}

/// The face at `slide`: everything but the pair now leaving. The ratio
/// mutant keeps a copy of the leaving chloride in the lattice.
func kinkedFace(slide: Float, mutant: Mutant) -> Molecule {
    var atoms: [Atom] = []
    for k in 0..<faceLayers {
        for j in 0..<faceRows {
            if k == 0 && j > stepRow { continue }
            for r in -rowReach..<rowReach {
                if k == 0 && j == stepRow && r >= -2 {
                    let chloride: Bool = (r + j + k) % 2 != 0
                    if !(mutant == .ionRatio && chloride) { continue }
                }
                atoms.append(faceIon(r, j, k, slide: slide))
            }
        }
    }
    return Molecule(atoms: atoms, bonds: [])
}

/// Where the next pair sits before it leaves: the last two ions of the step row.
func kinkSite(_ element: String) -> SIMD3<Float> {
    let r: Int = element == "Na" ? -2 : -1
    return faceIon(r, stepRow, 0, slide: 0).position
}

// MARK: - the pairs in the film

/// One Na⁺/Cl⁻ pair on its way out: `age` counts touches since it began to
/// leave (0 at the kink, 1 where step 32 showed its pair, then up and away).
struct LoosePair {
    var age: Float
    var sodium: Molecule
    var chloride: Molecule
}

/// Step 32's loose ions, where they stood in its still: now where each pair
/// has got to one touch after leaving. Positions MODEL, as there.
let settledSodium = SIMD3<Float>(-4.2, 7.6, 2.0)
let settledChloride = SIMD3<Float>(4.6, 8.8, -1.0)
let sodiumTurn = simd_quatf(angle: 0.5, axis: simd_normalize(SIMD3<Float>(1, 1, 0)))
let chlorideTurn = simd_quatf(angle: 0.8, axis: simd_normalize(SIMD3<Float>(0.3, 1, 0.5)))

/// After the first touch a pair keeps rising through the film, towards the
/// hair, this many ångströms a touch, turning slowly. MODEL: schematic —
/// a real ion diffuses across the inset's 3 nm in nanoseconds.
let pairRise: Float = 13.0
let pairTumble: Float = 0.3       // radians a touch
/// Pairs drawn at once: leaving, settled, and the one on its way out of the circle.
let pairSlots: Int = 3

/// How far through its hydration shell's forming a pair is at `age`: the
/// film's waters are drawn gathering round each ion as it rises clear of
/// the face, growing in from nothing. MODEL, a drawing choice: the water is
/// there all along (the film is water); only the shell is new.
func shellDrawn(age: Float) -> Float { smoothstep01((age - 0.35) / 0.5) }

/// Where a pair's ion is at `age`: straight up off its kink first, then out
/// to step 32's place, then up towards the hair.
func ionPosition(_ element: String, age: Float) -> SIMD3<Float> {
    let site: SIMD3<Float> = kinkSite(element)
    let settled: SIMD3<Float> = element == "Na" ? settledSodium : settledChloride
    if age >= 1 { return settled + SIMD3<Float>(0, pairRise * (age - 1), 0) }
    let up: Float = smoothstep01(age / 0.7)
    let out: Float = smoothstep01((age - 0.1) / 0.9)
    let travel: SIMD3<Float> = settled - site
    let xz: SIMD3<Float> = site + travel * out
    return SIMD3<Float>(xz.x, settled.y * up, xz.z)
}

func loosePair(age: Float) -> LoosePair {
    let turn = simd_quatf(angle: pairTumble * max(age - 1, 0), axis: simd_normalize(SIMD3<Float>(0.2, 1, -0.3)))
    let drawn: Float = shellDrawn(age: age)
    func shell(_ element: String, _ base: simd_quatf) -> Molecule {
        var m: Molecule = hydratedIon(element, at: ionPosition(element, age: age), turn: turn * base)
        for i in 1..<m.atoms.count { m.atoms[i].drawn = drawn }
        return m
    }
    return LoosePair(age: age, sodium: shell("Na", sodiumTurn), chloride: shell("Cl", chlorideTurn))
}

/// Everything in the molecule inset at one moment, in the inset's own
/// ångströms: the face with its kink, and the pairs in the film above.
struct SaltScene {
    var slab: Molecule
    var pairs: [LoosePair]

    /// The pair one touch out — fully hydrated, where step 32 drew its pair.
    var settled: LoosePair {
        var best: LoosePair = pairs[0]
        for p in pairs {
            let miss: Float = abs(p.age - 1)
            let bestMiss: Float = abs(best.age - 1)
            if miss < bestMiss { best = p }
        }
        return best
    }
    var sodium: Molecule { settled.sodium }
    var chloride: Molecule { settled.chloride }

    /// All of it as one list for the GPU, in inset coordinates.
    var all: Molecule {
        var atoms: [Atom] = []
        var bonds: [(Int, Int)] = []
        for m in [slab] + pairs.flatMap({ [$0.sodium, $0.chloride] }) {
            let base: Int = atoms.count
            atoms += m.atoms.map { var a = $0; a.position = saltView.act(a.position) + saltViewOffset; return a }
            bonds += m.bonds.map { ($0.0 + base, $0.1 + base) }
        }
        return Molecule(atoms: atoms, bonds: bonds)
    }
}

/// How the salt scene is turned for the camera (which looks along −z): 16°
/// down onto the face, 30° round. MODEL, for a view that shows the lattice
/// in depth and leaves the film above it clear.
let saltView: simd_quatf = simd_quatf(angle: 0.28, axis: SIMD3<Float>(1, 0, 0)) * simd_quatf(angle: 0.52, axis: SIMD3<Float>(0, 1, 0))
let saltViewOffset = SIMD3<Float>(0, -3.2, 0)

/// The salt scene at `progress` touches into the loop (see Motion.swift):
/// the whole part is how many pairs have left, the fraction how far the one
/// now leaving has got.
func buildSaltScene(progress u: Float, mutant: Mutant) -> SaltScene {
    let f: Float = u - u.rounded(.down)
    let slab: Molecule = kinkedFace(slide: f, mutant: mutant)
    let pairs: [LoosePair] = (0..<pairSlots).map { loosePair(age: f + Float($0)) }
    return SaltScene(slab: slab, pairs: pairs)
}

// MARK: - odour molecules for the smell hair

/// One odour molecule in the micrometre inset, drawn as a dot: where it is,
/// and the wall pore it is heading for.
struct Volatile {
    var position: SIMD3<Float>   // µm
    var pore: (row: Int, column: Int)
}

/// Salt has no vapour (Food.swift), so the air round the smell hair holds no
/// salt at all. The `volatiles` mutant puts some there anyway, and the tests
/// must notice.
func buildVolatiles(hairs: [Sensillum], mutant: Mutant) -> [Volatile] {
    guard foodHasVapour || mutant == .volatiles else { return [] }
    let smell: Sensillum = hairs[1]
    var out: [Volatile] = []
    let targets: [(Int, Int)] = [(4, 2), (9, 4), (14, 3), (7, 1)]
    for (k, t) in targets.enumerated() {
        let (mouth, outward) = smell.wallPore(row: t.0, column: t.1)
        // The first two are arriving at their pores; the rest are still on the way.
        let away: Float = k < 2 ? 0.14 : 1.2 + 0.8 * Float(k)
        out.append(Volatile(position: mouth + outward * away, pore: t))
    }
    return out
}

/// Dot radius for an odour molecule in the inset, µm. A molecule is well
/// under a nanometre; at this scale it would be invisible, so it is drawn
/// hundreds of times too big, and the label says so.
let volatileDotRadius: Float = 0.10
