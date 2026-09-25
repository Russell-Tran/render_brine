// The GFP molecule from Resources/gfp.json (made by Tools/build_gfp.py from
// the 1EMA crystal structure), drawn space-filling: every atom a sphere at
// its van der Waals radius, with no sticks, so neighbours overlap and merge
// into one solid shape.
//
// The exception is the chromophore, which is drawn ball-and-stick when the
// cutaway opens. Two styles in one frame, each doing what it is good at:
// space-filling shows the barrel's shape, ball-and-stick shows how the
// chromophore is put together.

import Foundation
import simd

func radians(_ degrees: Float) -> Float { degrees * .pi / 180 }

struct ProteinAtom {
    var element: String
    var position: SIMD3<Float>   // barrel axis along y, centred on the origin
    var seq: Int                 // residue number in the crystal
    var res: String              // residue name, e.g. "VAL" or "CRO"
    var name: String             // atom name, e.g. "CA"
    var t: Float                 // 0 at the N terminus, 1 at the C terminus
    var isChromophore: Bool
}

struct Protein {
    var atoms: [ProteinAtom]
    var bonds: [(a: Int, b: Int)]
    var chromophoreAtoms: [Int]
    var chromophoreCenter: SIMD3<Float>
    var residueCount: Int
    var strandCount: Int
    var wallRadius: Float
    var wallRadiusSpread: Float
    var wallLength: Float
    var selenomethionines: [Int]
}

enum SceneError: Error, CustomStringConvertible {
    case badFile(String)
    var description: String {
        switch self {
        case .badFile(let detail): return "couldn't read the protein: \(detail)"
        }
    }
}

func loadProtein(from url: URL) throws -> Protein {
    let data = try Data(contentsOf: url)
    guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
          let rawAtoms = root["atoms"] as? [[String: Any]],
          let rawBonds = root["bonds"] as? [[Int]] else { throw SceneError.badFile("missing fields") }
    var atoms: [ProteinAtom] = []
    atoms.reserveCapacity(rawAtoms.count)
    for a in rawAtoms {
        guard let el = a["el"] as? String, let p = a["pos"] as? [NSNumber], p.count == 3 else {
            throw SceneError.badFile("bad atom")
        }
        atoms.append(ProteinAtom(element: el,
                                 position: SIMD3(p[0].floatValue, p[1].floatValue, p[2].floatValue),
                                 seq: a["seq"] as? Int ?? 0,
                                 res: a["res"] as? String ?? "",
                                 name: a["name"] as? String ?? "",
                                 t: (a["t"] as? NSNumber)?.floatValue ?? 0,
                                 isChromophore: a["chromophore"] as? Bool ?? false))
    }
    let c = (root["chromophoreCenter"] as? [NSNumber] ?? []).map { $0.floatValue }
    return Protein(atoms: atoms,
                   bonds: rawBonds.map { (a: $0[0], b: $0[1]) },
                   chromophoreAtoms: root["chromophoreAtoms"] as? [Int] ?? [],
                   chromophoreCenter: c.count == 3 ? SIMD3(c[0], c[1], c[2]) : .zero,
                   residueCount: root["residueCount"] as? Int ?? 0,
                   strandCount: root["strandCount"] as? Int ?? 0,
                   wallRadius: (root["wallRadius"] as? NSNumber)?.floatValue ?? 0,
                   wallRadiusSpread: (root["wallRadiusSpread"] as? NSNumber)?.floatValue ?? 0,
                   wallLength: (root["wallLength"] as? NSNumber)?.floatValue ?? 0,
                   selenomethionines: root["selenomethionines"] as? [Int] ?? [])
}

// MARK: - Space filling

/// Van der Waals radii in ångströms, Bondi, J. Phys. Chem. 68:441 (1964).
/// These are what make a space-filling model: each atom takes up the room its
/// electron cloud actually occupies, which is about twice the covalent radius,
/// so bonded neighbours overlap instead of needing a stick between them.
func vdwRadius(_ element: String) -> Float {
    switch element {
    case "C": return 1.70
    case "N": return 1.55
    case "O": return 1.52
    case "S": return 1.80
    case "SE": return 1.90
    case "P": return 1.80
    case "H": return 1.20
    default: return 1.70
    }
}

/// Radii for the chromophore's ball-and-stick, where the point is connectivity
/// rather than bulk. Same proportions as steps 6 to 8.
func ballRadius(_ element: String) -> Float {
    switch element {
    case "C": return 0.55
    case "N": return 0.54
    case "O": return 0.52
    case "S": return 0.62
    case "SE": return 0.64
    default: return 0.36
    }
}

/// Where a residue sits along the chain, as a colour: a muted spectrum from
/// deep blue at the N terminus through teal and olive to warm red at the C
/// terminus. This is the usual way to show a fold's topology — you can count
/// the barrel's eleven strands by the order their colours run.
///
/// It is deliberately desaturated, so the one vivid colour in the frame is the
/// chromophore. Colouring by element would be a grey and red mush at this size.
func spectrumColor(_ t: Float) -> SIMD3<Float> {
    let stops: [SIMD3<Float>] = [
        SIMD3(0.22, 0.31, 0.60),   // N terminus, deep blue
        SIMD3(0.24, 0.50, 0.62),
        SIMD3(0.33, 0.58, 0.52),
        SIMD3(0.55, 0.60, 0.41),
        SIMD3(0.70, 0.52, 0.36),
        SIMD3(0.68, 0.36, 0.34),   // C terminus, warm red
    ]
    let x: Float = min(max(t, 0), 1) * Float(stops.count - 1)
    let i = min(Int(x), stops.count - 2)
    let f: Float = x - Float(i)
    return stops[i] + (stops[i + 1] - stops[i]) * f
}

/// The chromophore: the only saturated colour in the frame, and green because
/// that is the light this protein makes.
let chromophoreColor = SIMD3<Float>(0.32, 0.92, 0.38)

// MARK: - The cutaway

/// How much of each atom survives the cutaway, 0 (gone) to 1 (whole).
///
/// `opening` runs 0 to 1, and irises a round window open in whichever face is
/// towards the camera, centred on the chromophore. Only atoms inside that
/// window and in front of the chromophore are removed.
///
/// The first attempt simply deleted everything nearer the camera than the
/// chromophore. That works, but a half-space cut takes away half the molecule,
/// and what is left no longer reads as a barrel. A round window keeps the rim
/// intact, so you can still see what you are looking into.
func cutFactors(_ protein: Protein, opening: Float, camera: Camera) -> [Float] {
    var out = [Float](repeating: 1, count: protein.atoms.count)
    if opening <= 0 { return out }
    let forward = camera.forward
    let center = protein.chromophoreCenter
    // Just past the chromophore, so it is never hidden by what remains.
    let backPlane: Float = simd_dot(center, forward) + 1.5
    // The barrel wall has a radius of about 11.6 Å, so this is a porthole in
    // its face rather than the removal of the whole face.
    let windowRadius: Float = 8.5 * opening
    let softness: Float = 3.0
    for (i, atom) in protein.atoms.enumerated() {
        if atom.isChromophore { continue }          // never cut the chromophore itself
        let offset = atom.position - center
        let depth: Float = simd_dot(offset, forward)
        if depth >= backPlane - simd_dot(center, forward) { continue }   // behind it: keep
        let lateral: Float = simd_length(offset - forward * depth)
        // Gone well inside the window, whole well outside, faded across the rim.
        let x: Float = (lateral - windowRadius) / softness
        out[i] = min(max(x, 0), 1)
    }
    return out
}

/// How far the cutaway has opened at time `u` through the loop, 0 to 1.
/// Closed, opens, holds open, closes, closed — so the loop joins itself
/// without ever playing backwards.
func opening(at u: Double) -> Float {
    func smooth(_ x: Double) -> Float {
        let c = min(max(x, 0), 1)
        return Float(c * c * (3 - 2 * c))
    }
    if u < 0.30 { return 0 }
    if u < 0.42 { return smooth((u - 0.30) / 0.12) }
    if u < 0.62 { return 1 }
    if u < 0.74 { return 1 - smooth((u - 0.62) / 0.12) }
    return 0
}

// MARK: - Geometry for the GPU

func sceneGeometry(_ protein: Protein, cut: [Float], showChromophore: Bool)
    -> (spheres: [GPUSphere], cylinders: [GPUCylinder]) {
    var spheres: [GPUSphere] = []
    spheres.reserveCapacity(protein.atoms.count)
    for (i, atom) in protein.atoms.enumerated() {
        if atom.isChromophore { continue }
        let keep: Float = cut[i]
        guard keep > 0.01 else { continue }
        let p = atom.position
        let c = spectrumColor(atom.t)
        // Shrinking the radius, rather than fading, keeps every surface opaque:
        // a ray tracer of solid spheres has no transparency to blend.
        let r: Float = vdwRadius(atom.element) * keep
        spheres.append(GPUSphere(centerRadius: SIMD4(p.x, p.y, p.z, r),
                                 color: SIMD4(c.x, c.y, c.z, 1), glow: .zero))
    }
    var cylinders: [GPUCylinder] = []
    if showChromophore {
        for i in protein.chromophoreAtoms {
            let atom = protein.atoms[i]
            let p = atom.position
            let r: Float = ballRadius(atom.element)
            spheres.append(GPUSphere(centerRadius: SIMD4(p.x, p.y, p.z, r),
                                     color: SIMD4(chromophoreColor.x, chromophoreColor.y, chromophoreColor.z, 1),
                                     glow: .zero))
        }
        let inChromophore = Set(protein.chromophoreAtoms)
        let stick = SIMD4<Float>(0.62, 0.78, 0.60, 1)
        for bond in protein.bonds where inChromophore.contains(bond.a) && inChromophore.contains(bond.b) {
            let a = protein.atoms[bond.a].position, b = protein.atoms[bond.b].position
            cylinders.append(GPUCylinder(aRadius: SIMD4(a.x, a.y, a.z, bondRadius),
                                         b: SIMD4(b.x, b.y, b.z, 0), color: stick))
        }
    }
    return (spheres, cylinders)
}

/// Mean of a list, 0 when empty.
func mean(_ xs: [Float]) -> Float { xs.isEmpty ? 0 : xs.reduce(0, +) / Float(xs.count) }

/// The palette keeps shades of the chromophore's green whatever else it
/// chooses. Median cut hands colours to whatever covers the most pixels, and
/// the chromophore is small and only on screen for part of the loop, so left
/// to itself it would lose its green — exactly the trap step 8 hit with
/// phosphorus.
func gfpPalette(samples: [RGB], count: Int) -> [RGB] {
    var reserved: [RGB] = []
    for light: Float in [0.25, 0.45, 0.65, 0.85, 1.0, 1.15] {
        let v = simd_clamp(chromophoreColor * light, SIMD3<Float>(repeating: 0),
                           SIMD3<Float>(repeating: 1)) * 255
        reserved.append(RGB(UInt8(v.x.rounded()), UInt8(v.y.rounded()), UInt8(v.z.rounded())))
    }
    return reserved + medianCutPalette(samples, count: count - reserved.count)
}
