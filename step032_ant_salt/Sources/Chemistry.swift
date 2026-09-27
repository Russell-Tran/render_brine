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

/// A slab of the rock-salt lattice, `n` × `n` ions across and `layers` deep,
/// its top (100) face at y = 0; ion (i, j, k) is Na⁺ when i + j + k is even.
/// Ions listed in `leave` are gone from the top layer — they are the ones in
/// the film above. Even `n` gives equal numbers of each.
func rockSaltSlab(n: Int, layers: Int, leave: [SIMD3<Int32>], mutant: Mutant) -> Molecule {
    let s: Float = haliteCell / 2
    var atoms: [Atom] = []
    let off: Float = Float(n - 1) / 2
    for k in 0..<layers {
        for i in 0..<n {
            for j in 0..<n {
                if leave.contains(SIMD3<Int32>(Int32(i), Int32(j), Int32(k))) {
                    // The ratio mutant keeps the chloride that left, too.
                    if !(mutant == .ionRatio && (i + j + k) % 2 == 1) { continue }
                }
                let na: Bool = (i + j + k) % 2 == 0
                let p = SIMD3<Float>((Float(i) - off) * s, -Float(k) * s, (Float(j) - off) * s)
                atoms.append(Atom(element: na ? "Na" : "Cl", position: p, charge: na ? 1 : -1))
            }
        }
    }
    return Molecule(atoms: atoms, bonds: [])
}

/// The top-layer sites the two loose ions came from: a neighbouring Na⁺ and
/// Cl⁻ near the middle of the face.
let leavingSites: [SIMD3<Int32>] = [SIMD3(4, 4, 0), SIMD3(4, 3, 0)]

/// Everything in the molecule inset, in the inset's own ångströms: the slab
/// tilted so the view looks a little down onto the face, and above it the
/// two ions that left, each in its water shell.
struct SaltScene {
    var slab: Molecule
    var sodium: Molecule
    var chloride: Molecule

    /// All of it as one list for the GPU, in inset coordinates.
    var all: Molecule {
        var atoms: [Atom] = []
        var bonds: [(Int, Int)] = []
        for m in [slab, sodium, chloride] {
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

func buildSaltScene(mutant: Mutant) -> SaltScene {
    let slab: Molecule = rockSaltSlab(n: 8, layers: 3, leave: leavingSites, mutant: mutant)
    // The loose ions: in the film above the face, a few ångströms apart,
    // each still near the gap it left. Positions MODEL.
    let na: Molecule = hydratedIon("Na", at: SIMD3(-4.2, 7.6, 2.0), turn: simd_quatf(angle: 0.5, axis: simd_normalize(SIMD3<Float>(1, 1, 0))))
    let cl: Molecule = hydratedIon("Cl", at: SIMD3(4.6, 8.8, -1.0), turn: simd_quatf(angle: 0.8, axis: simd_normalize(SIMD3<Float>(0.3, 1, 0.5))))
    return SaltScene(slab: slab, sodium: na, chloride: cl)
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
