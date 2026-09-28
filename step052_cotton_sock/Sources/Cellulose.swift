// The molecule: cellulose, a chain of glucose units joined β(1→4), each
// unit turned 180° from its neighbour. Built from the crystal structure of
// cellulose Iβ — the form in cotton and other higher plants — not drawn:
//
// Nishiyama, Langan & Chanzy 2002, "Crystal structure and hydrogen-bonding
// system in cellulose Iβ from synchrotron X-ray and neutron fiber
// diffraction", J Am Chem Soc 124: 9074–9082 (doi 10.1021/ja0257319). Their
// deposited CIF (ja0257319_s1_1.cif) is in the Crystallography Open
// Database as entry 4114994, public domain; copied verbatim to Resources/.
// Space group P2₁, the 2₁ screw axis along the chain (c = 10.380 Å, two
// glucose units per repeat): applying (x, y, z) → (−x, −y, z + ½) to one
// glucose unit makes the next, turned 180° about the chain and moved 5.19 Å
// along it. That screw IS the alternating flip.

import Foundation
import simd

struct Atom {
    var element: String
    var position: SIMD3<Float>      // Å
    var name: String = ""           // e.g. "C1", "O4", "H61A"
    var residue: Int = 0
}

struct Molecule {
    var atoms: [Atom]
    var bonds: [(Int, Int)]

    var formula: [String: Int] {
        var f: [String: Int] = [:]
        for a in atoms { f[a.element, default: 0] += 1 }
        return f
    }

    var valences: [Int] {
        var v: [Int] = Array(repeating: 0, count: atoms.count)
        for b in bonds { v[b.0] += 1; v[b.1] += 1 }
        return v
    }

    func index(_ name: String, _ residue: Int) -> Int? {
        atoms.firstIndex { $0.name == name && $0.residue == residue }
    }
}

enum LoadError: Error { case missing(String), malformed(String) }

/// A CIF's cell and its atom sites (label, element, fractional x y z), with
/// the standard uncertainties in brackets dropped.
struct CIF {
    var a: Float, b: Float, c: Float, gamma: Float     // Å, degrees (monoclinic, unique axis c)
    var sites: [(label: String, element: String, frac: SIMD3<Float>)]

    static func load(_ path: String) throws -> CIF {
        guard let text = try? String(contentsOfFile: path, encoding: .utf8) else { throw LoadError.missing(path) }
        let lines: [String] = text.components(separatedBy: "\n")
        func number(_ s: Substring) -> Float? {
            let clean: String = String(s.split(separator: "(").first ?? s)
            return Float(clean)
        }
        var cell: [String: Float] = [:]
        for l in lines {
            let f: [Substring] = l.split(separator: " ", omittingEmptySubsequences: true)
            if f.count >= 2, f[0].hasPrefix("_cell_length_") || f[0].hasPrefix("_cell_angle_") {
                cell[String(f[0])] = number(f[1])
            }
        }
        guard let a = cell["_cell_length_a"], let b = cell["_cell_length_b"], let c = cell["_cell_length_c"],
              let g = cell["_cell_angle_gamma"] else { throw LoadError.malformed("\(path): no cell") }
        // The atom_site loop: rows after its header names, until the next loop.
        var sites: [(String, String, SIMD3<Float>)] = []
        var inSites = false
        var header: [String] = []
        for l in lines {
            let t: String = l.trimmingCharacters(in: .whitespaces)
            if t == "loop_" { if inSites && !sites.isEmpty { break }; header = []; inSites = false; continue }
            if t.hasPrefix("_atom_site_") { header.append(t); inSites = true; continue }
            guard inSites, !t.isEmpty, !t.hasPrefix("_") else { continue }
            let f: [Substring] = t.split(separator: " ", omittingEmptySubsequences: true)
            guard let il = header.firstIndex(of: "_atom_site_label"), let ie = header.firstIndex(of: "_atom_site_type_symbol"),
                  let ix = header.firstIndex(of: "_atom_site_fract_x"), f.count > max(il, ie, ix + 2),
                  let x = number(f[ix]), let y = number(f[ix + 1]), let z = number(f[ix + 2]) else { continue }
            sites.append((String(f[il]), String(f[ie]), SIMD3<Float>(x, y, z)))
        }
        guard !sites.isEmpty else { throw LoadError.malformed("\(path): no atom sites") }
        return CIF(a: a, b: b, c: c, gamma: g, sites: sites.map { (label: $0.0, element: $0.1, frac: $0.2) })
    }

    /// Fractional → Cartesian Å: a along x, b in the xy plane at γ, c along z.
    func cartesian(_ f: SIMD3<Float>) -> SIMD3<Float> {
        let gr: Float = gamma * Float.pi / 180
        return SIMD3<Float>(a * f.x + b * cos(gr) * f.y, b * sin(gr) * f.y, c * f.z)
    }
}

/// Glucose units in the drawn chain.
let chainUnits: Int = 4

// O–H length for the hydroxyl hydrogens, which the CIF does not list (its
// hydrogens are the C–H ones, placed geometrically). MODEL, 0.97 Å, a
// typical neutron O–H; UNVERIFIED against this structure.
let hydroxylLength: Float = 0.97

/// The chain: `chainUnits` glucose units of the origin chain ("…1" labels),
/// made by the 2₁ screw, with the hydroxyl hydrogens added.
///
/// Nishiyama et al. (abstract, via Europe PMC): the O3–H···O5 intrachain
/// hydrogen bond has a well-defined hydrogen; the O2 and O6 hydrogens are
/// disordered. So O3–H is drawn pointing at the next unit's ring oxygen O5,
/// and the O2, O6 and chain-end hydrogens are MODEL: anti to a ring carbon,
/// at the tetrahedral angle — one position out of several the crystal
/// holds.
func buildCellulose(resources: String, mutant: Mutant) throws -> Molecule {
    let cif: CIF = try CIF.load(resources + "/cellulose_Ibeta_COD4114994.cif")
    // One glucose unit of chain 1: labels like C11 (C1 of chain 1), O51,
    // H61A. The second digit is the chain.
    var unit: [(name: String, element: String, pos: SIMD3<Float>)] = []
    for s in cif.sites {
        let chars = Array(s.label)
        guard chars.count >= 3, chars[2] == "1" else { continue }
        let name: String = String(chars[0...1]) + String(chars.dropFirst(3))
        unit.append((name, s.element, cif.cartesian(s.frac)))
    }
    func screw(_ p: SIMD3<Float>, _ n: Int) -> SIMD3<Float> {
        let s: Float = n % 2 == 0 ? 1 : -1
        return SIMD3<Float>(s * p.x, s * p.y, p.z + Float(n) * cif.c / 2)
    }
    var atoms: [Atom] = []
    for n in 0..<chainUnits {
        for u in unit { atoms.append(Atom(element: u.element, position: screw(u.pos, n), name: u.name, residue: n)) }
    }
    // The last unit's anomeric oxygen O1: where the next unit's O4 would be.
    // It keeps the crystal's β position, and is the reducing end's OH.
    if let o4 = unit.first(where: { $0.name == "O1" }) {
        atoms.append(Atom(element: "O", position: screw(o4.pos, chainUnits), name: "O1r", residue: chainUnits - 1))
    }
    var mol = Molecule(atoms: atoms, bonds: [])
    // In the CIF's labels the glycosidic oxygen "O1x" is the one on C4 (the
    // bond list has C41–O11 and C11–O11 by the screw): rename it O4 here.
    for k in mol.atoms.indices where mol.atoms[k].name == "O1" { mol.atoms[k].name = "O4" }
    for k in mol.atoms.indices where mol.atoms[k].name == "O1r" { mol.atoms[k].name = "O1" }
    var bonds: [(Int, Int)] = []
    func bond(_ a: String, _ ra: Int, _ b: String, _ rb: Int) {
        if let i = mol.index(a, ra), let j = mol.index(b, rb) { bonds.append((i, j)) }
    }
    for n in 0..<chainUnits {
        for (a, b) in [("C1", "C2"), ("C2", "C3"), ("C3", "C4"), ("C4", "C5"), ("C5", "O5"), ("O5", "C1"),
                       ("C5", "C6"), ("C6", "O6"), ("C2", "O2"), ("C3", "O3"), ("C4", "O4"),
                       ("C1", "H1"), ("C2", "H2"), ("C3", "H3"), ("C4", "H4"), ("C5", "H5"), ("C6", "H6A"), ("C6", "H6B")] {
            bond(a, n, b, n)
        }
        // The β(1→4) link: this unit's C1 to the next unit's O4 (the last
        // unit's C1 to its own O1).
        if n + 1 < chainUnits { bond("C1", n, "O4", n + 1) } else { bond("C1", n, "O1", n) }
    }
    // (The CIF's carbon hydrogens H11…H61B became H1…H6B above: C-number,
    // chain digit dropped.)
    mol.bonds = bonds
    if mutant == .alphaLinks { makeAlpha(&mol) }
    addHydroxylHydrogens(&mol, cif: cif, screw: screw, unit: unit)
    return mol
}

/// The α mutant: at every anomeric carbon, swap where the glycosidic oxygen
/// and the hydrogen point — O1 axial instead of equatorial, as in starch —
/// keeping the bond lengths. (The next unit is not rebuilt to follow it:
/// the mutant only has to be wrong in the way the tests look for.)
func makeAlpha(_ m: inout Molecule) {
    for n in 0..<chainUnits {
        guard let c1 = m.index("C1", n), let h1 = m.index("H1", n) else { continue }
        let oName: String = n + 1 < chainUnits ? "O4" : "O1"
        guard let o = m.index(oName, n + 1 < chainUnits ? n + 1 : n) else { continue }
        let c: SIMD3<Float> = m.atoms[c1].position
        let dO: SIMD3<Float> = m.atoms[o].position - c
        let dH: SIMD3<Float> = m.atoms[h1].position - c
        m.atoms[o].position = c + simd_normalize(dH) * simd_length(dO)
        m.atoms[h1].position = c + simd_normalize(dO) * simd_length(dH)
    }
}

func addHydroxylHydrogens(_ m: inout Molecule, cif: CIF, screw: (SIMD3<Float>, Int) -> SIMD3<Float>,
                          unit: [(name: String, element: String, pos: SIMD3<Float>)]) {
    func add(_ oIndex: Int, towards dir: SIMD3<Float>) {
        let h = Atom(element: "H", position: m.atoms[oIndex].position + simd_normalize(dir) * hydroxylLength,
                     name: "H" + m.atoms[oIndex].name, residue: m.atoms[oIndex].residue)
        m.atoms.append(h)
        m.bonds.append((oIndex, m.atoms.count - 1))
    }
    /// A hydrogen on oxygen O bonded to carbon C, anti to carbon X across
    /// the C–O bond, C–O–H 109.5°.
    func anti(_ o: Int, _ c: Int, _ x: Int) {
        let po: SIMD3<Float> = m.atoms[o].position, pc: SIMD3<Float> = m.atoms[c].position, px: SIMD3<Float> = m.atoms[x].position
        let axis: SIMD3<Float> = simd_normalize(po - pc)
        let away: SIMD3<Float> = simd_normalize((pc - px) - axis * simd_dot(pc - px, axis))
        let theta: Float = (180 - 109.5) * Float.pi / 180
        add(o, towards: axis * cos(theta) + away * sin(theta))
    }
    let o5unit: SIMD3<Float>? = unit.first(where: { $0.name == "O5" })?.pos
    for n in 0..<chainUnits {
        // O3–H → the O5 of the neighbouring unit nearest it (the well-defined
        // intrachain hydrogen bond), computed for units n±1 by the screw.
        if let o3 = m.index("O3", n), let o5 = o5unit {
            let p: SIMD3<Float> = m.atoms[o3].position
            let candidates: [SIMD3<Float>] = [screw(o5, n - 1), screw(o5, n + 1)]
            let target: SIMD3<Float> = candidates.min { simd_distance($0, p) < simd_distance($1, p) }!
            add(o3, towards: target - p)
        }
        if let o2 = m.index("O2", n), let c2 = m.index("C2", n), let c1 = m.index("C1", n) { anti(o2, c2, c1) }
        if let o6 = m.index("O6", n), let c6 = m.index("C6", n), let c5 = m.index("C5", n) { anti(o6, c6, c5) }
    }
    // Chain ends: O4–H on the first (non-reducing) unit, O1–H on the last.
    if let o4 = m.index("O4", 0), let c4 = m.index("C4", 0), let c3 = m.index("C3", 0) { anti(o4, c4, c3) }
    if let o1 = m.index("O1", chainUnits - 1), let c1 = m.index("C1", chainUnits - 1), let o5 = m.index("O5", chainUnits - 1) {
        anti(o1, c1, o5)
    }
}

/// Turn and centre the chain for the inset: the chain axis (crystal c)
/// along x, and the direction the CH₂OH groups alternate in along y, so
/// the 180° flips read as up, down, up, down.
func placeForInset(_ m: Molecule) -> Molecule {
    var centre = SIMD3<Float>(0, 0, 0)
    for a in m.atoms { centre += a.position }
    centre /= Float(m.atoms.count)
    let axis = SIMD3<Float>(0, 0, 1)
    var flip = SIMD3<Float>(0, 0, 0)
    for n in 0..<chainUnits {
        guard let c6 = m.index("C6", n), let c1 = m.index("C1", n), let c4 = m.index("C4", n) else { continue }
        let ring: SIMD3<Float> = (m.atoms[c1].position + m.atoms[c4].position) / 2
        var d: SIMD3<Float> = m.atoms[c6].position - ring
        d -= axis * simd_dot(d, axis)
        flip += n % 2 == 0 ? d : -d
    }
    let up: SIMD3<Float> = simd_normalize(flip)
    let toward: SIMD3<Float> = simd_cross(axis, up)
    // Rows of the rotation: new x = axis, new y = up, new z = toward viewer.
    var out: Molecule = m
    out.atoms = m.atoms.map { a in
        let q: SIMD3<Float> = a.position - centre
        var b: Atom = a
        b.position = SIMD3<Float>(simd_dot(q, axis), simd_dot(q, up), simd_dot(q, toward))
        return b
    }
    return out
}
