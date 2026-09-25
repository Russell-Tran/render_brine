// The two proteins from Resources/pair.json (made by Tools/build_pair.py),
// drawn space-filling exactly as step 9 drew GFP alone — but two of them,
// side by side, each in its own half of the frame.
//
// WHY TWO HALF-FRAMES RATHER THAN ONE SCENE. Step 9's ambient occlusion is
// steady from frame to frame because the molecule never moves: the camera
// orbits it and the lights are rotated to match, so every surface point keeps
// its shading. Putting two barrels in one scene and spinning them would throw
// that away — the probe pattern is built from a frame that references the
// world axes, so a rotating surface would draw a slightly different spray each
// frame and shimmer.
//
// So each protein stays fixed in its own coordinates and gets its own orbiting
// camera, rendered into its own half of the frame. Step 9's guarantee holds
// unchanged for both. It is also cheaper: occlusion probes only ever test the
// ~1,750 spheres of one protein, not both.

import Foundation
import simd

func radians(_ degrees: Float) -> Float { degrees * .pi / 180 }

struct ProteinAtom {
    var element: String
    var position: SIMD3<Float>   // barrel axis along y, chromophore facing +z
    var seq: Int
    var res: String
    var name: String
    var t: Float                 // 0 at the N terminus, 1 at the C terminus
    var isChromophore: Bool
    var isPi: Bool               // part of the conjugated system
}

struct FluorescentProtein {
    var key: String              // "GFP" or "mCherry"
    var label: String            // as it appears in the caption
    var pdb: String
    var atoms: [ProteinAtom]
    var bonds: [(a: Int, b: Int)]
    var chromophoreAtoms: [Int]
    var chromophoreCenter: SIMD3<Float>
    var chromophoreName: String
    var chromophoreAtomCount: Int
    var formedFrom: [String]
    var missingResidues: [Int]
    var piAtoms: [Int]
    var piBonds: [(a: Int, b: Int)]
    var extraPi: [String]        // atoms this one has in its pi system and the other doesn't
    var acylimineLength: Float   // N1–CA1, the bond that decides the colour
    var excitation: Float        // nm
    var emission: Float          // nm
    var residueCount: Int
    var strandCount: Int
    var wallRadius: Float
    var selenomethionines: [Int]

    /// The colour of the light this protein emits, computed rather than chosen.
    var emissionColour: SIMD3<Float> { spectralColour(nanometres: emission).linear }
    var emissionClipped: Bool { spectralColour(nanometres: emission).clipped }
    /// The energy of one emitted photon, in electron volts.
    var photonEnergyEV: Double { photonEnergy(nanometres: Double(emission)) }
}

enum SceneError: Error, CustomStringConvertible {
    case badFile(String)
    var description: String {
        switch self {
        case .badFile(let detail): return "couldn't read the pair: \(detail)"
        }
    }
}

struct ProteinPair {
    var gfp: FluorescentProtein
    var mCherry: FluorescentProtein
    var foldMedian: Float        // median distance from an mCherry CA to the nearest GFP CA
    var foldWithin3A: Float      // fraction within 3 Å
    var both: [FluorescentProtein] { [gfp, mCherry] }
}

private func protein(from raw: [String: Any]) throws -> FluorescentProtein {
    guard let rawAtoms = raw["atoms"] as? [[String: Any]],
          let rawBonds = raw["bonds"] as? [[Int]] else { throw SceneError.badFile("missing atoms") }
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
                                 isChromophore: a["chromophore"] as? Bool ?? false,
                                 isPi: a["pi"] as? Bool ?? false))
    }
    let c = (raw["chromophoreCenter"] as? [NSNumber] ?? []).map { $0.floatValue }
    let piBonds = (raw["piBonds"] as? [[Int]] ?? []).map { (a: $0[0], b: $0[1]) }
    return FluorescentProtein(
        key: raw["key"] as? String ?? "",
        label: raw["label"] as? String ?? "",
        pdb: raw["pdb"] as? String ?? "",
        atoms: atoms,
        bonds: rawBonds.map { (a: $0[0], b: $0[1]) },
        chromophoreAtoms: raw["chromophoreAtoms"] as? [Int] ?? [],
        chromophoreCenter: c.count == 3 ? SIMD3(c[0], c[1], c[2]) : .zero,
        chromophoreName: raw["chromophoreName"] as? String ?? "",
        chromophoreAtomCount: raw["chromophoreAtomCount"] as? Int ?? 0,
        formedFrom: raw["formedFrom"] as? [String] ?? [],
        missingResidues: raw["missingResidues"] as? [Int] ?? [],
        piAtoms: raw["piAtoms"] as? [Int] ?? [],
        piBonds: piBonds,
        extraPi: raw["extraPi"] as? [String] ?? [],
        acylimineLength: (raw["acylimineLength"] as? NSNumber)?.floatValue ?? 0,
        excitation: (raw["excitation"] as? NSNumber)?.floatValue ?? 0,
        emission: (raw["emission"] as? NSNumber)?.floatValue ?? 0,
        residueCount: raw["residueCount"] as? Int ?? 0,
        strandCount: raw["strandCount"] as? Int ?? 0,
        wallRadius: (raw["wallRadius"] as? NSNumber)?.floatValue ?? 0,
        selenomethionines: raw["selenomethionines"] as? [Int] ?? [])
}

func loadPair(from url: URL) throws -> ProteinPair {
    let data = try Data(contentsOf: url)
    guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
          let list = root["proteins"] as? [[String: Any]], list.count == 2 else {
        throw SceneError.badFile("expected two proteins")
    }
    let a = try protein(from: list[0])
    let b = try protein(from: list[1])
    let fold = root["foldAgreement"] as? [String: Any] ?? [:]
    return ProteinPair(gfp: a.key == "GFP" ? a : b,
                       mCherry: a.key == "GFP" ? b : a,
                       foldMedian: (fold["median"] as? NSNumber)?.floatValue ?? 0,
                       foldWithin3A: (fold["within3A"] as? NSNumber)?.floatValue ?? 0)
}

// MARK: - Space filling

/// Van der Waals radii in ångströms, Bondi, J. Phys. Chem. 68:441 (1964).
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

/// Radii for the chromophore's ball-and-stick, where connectivity is the point.
/// Deliberately larger than a true ball-and-stick proportion, as step 9's was:
/// the chromophore is 22 atoms seen down a porthole from 134 Å away, and at
/// honest proportions it is a few pixels of squiggle.
func ballRadius(_ element: String) -> Float {
    switch element {
    case "C": return 0.72
    case "N": return 0.70
    case "O": return 0.68
    case "S": return 0.78
    case "SE": return 0.80
    default: return 0.46
    }
}

/// The barrel's colour by position along the chain, N terminus to C terminus.
/// Deliberately muted, so the only vivid colours in the frame are the two
/// emission colours — which are the subject.
func spectrumColor(_ t: Float) -> SIMD3<Float> {
    let stops: [SIMD3<Float>] = [
        SIMD3(0.20, 0.27, 0.46),
        SIMD3(0.22, 0.42, 0.50),
        SIMD3(0.30, 0.47, 0.43),
        SIMD3(0.46, 0.48, 0.35),
        SIMD3(0.55, 0.42, 0.31),
        SIMD3(0.54, 0.30, 0.29),
    ]
    let x: Float = min(max(t, 0), 1) * Float(stops.count - 1)
    let i = min(Int(x), stops.count - 2)
    let f: Float = x - Float(i)
    return stops[i] + (stops[i + 1] - stops[i]) * f
}

// MARK: - The cutaway

/// How much of each atom survives the porthole, 0 (gone) to 1 (whole).
/// Step 9's round window, unchanged: a half-space cut removes half the molecule
/// and what is left stops reading as a barrel.
func cutFactors(_ p: FluorescentProtein, opening: Float, camera: Camera) -> [Float] {
    var out = [Float](repeating: 1, count: p.atoms.count)
    if opening <= 0 { return out }
    let forward = camera.forward
    let centre = p.chromophoreCenter
    let backDepth: Float = 1.5
    let windowRadius: Float = 8.5 * opening
    let softness: Float = 3.0
    for (i, atom) in p.atoms.enumerated() {
        if atom.isChromophore { continue }
        let offset = atom.position - centre
        let depth: Float = simd_dot(offset, forward)
        if depth >= backDepth { continue }
        let lateral: Float = simd_length(offset - forward * depth)
        let x: Float = (lateral - windowRadius) / softness
        out[i] = min(max(x, 0), 1)
    }
    return out
}

/// How far the porthole has opened at time `u` through the loop. Both proteins
/// are given the same value, so the two open and shut together and the
/// comparison is always like for like.
func opening(at u: Double) -> Float {
    func smooth(_ x: Double) -> Float {
        let c = min(max(x, 0), 1)
        return Float(c * c * (3 - 2 * c))
    }
    if u < 0.28 { return 0 }
    if u < 0.40 { return smooth((u - 0.28) / 0.12) }
    if u < 0.64 { return 1 }
    if u < 0.76 { return 1 - smooth((u - 0.64) / 0.12) }
    return 0
}

// MARK: - Geometry for the GPU

/// The barrel plus, when the porthole is open, the chromophore in ball-and-stick.
///
/// The chromophore is drawn in the protein's own emission colour — computed
/// from its wavelength, not picked. Its conjugated atoms are drawn in that
/// colour at full strength; the few atoms outside the pi system (the methionine
/// or threonine side chain, which has nothing to do with the colour) are drawn
/// muted, so the eye lands on the part that matters.
func sceneGeometry(_ p: FluorescentProtein, cut: [Float], showChromophore: Bool)
    -> (spheres: [GPUSphere], cylinders: [GPUCylinder]) {
    var spheres: [GPUSphere] = []
    spheres.reserveCapacity(p.atoms.count)
    for (i, atom) in p.atoms.enumerated() {
        if atom.isChromophore { continue }
        let keep: Float = cut[i]
        guard keep > 0.01 else { continue }
        let q = atom.position
        let c = spectrumColor(atom.t)
        let r: Float = vdwRadius(atom.element) * keep
        spheres.append(GPUSphere(centerRadius: SIMD4(q.x, q.y, q.z, r),
                                 color: SIMD4(c.x, c.y, c.z, 1), glow: .zero))
    }
    var cylinders: [GPUCylinder] = []
    guard showChromophore else { return (spheres, cylinders) }

    let emit: SIMD3<Float> = p.emissionColour
    let offPi: SIMD3<Float> = SIMD3(0.46, 0.46, 0.44)   // not conjugated, so not the story
    for i in p.chromophoreAtoms {
        let atom = p.atoms[i]
        let q = atom.position
        let c: SIMD3<Float> = atom.isPi ? emit : offPi
        let r: Float = ballRadius(atom.element)
        spheres.append(GPUSphere(centerRadius: SIMD4(q.x, q.y, q.z, r),
                                 color: SIMD4(c.x, c.y, c.z, 1), glow: .zero))
    }
    let inPi = Set(p.piAtoms)
    let inChromophore = Set(p.chromophoreAtoms)
    // Conjugated bonds take the emission colour and are drawn fatter; the rest
    // of the chromophore's bonds stay thin and grey.
    let dullStick = SIMD4<Float>(0.50, 0.50, 0.48, 1)
    let liveStick = SIMD4<Float>(emit.x * 0.85, emit.y * 0.85, emit.z * 0.85, 1)
    for bond in p.bonds where inChromophore.contains(bond.a) && inChromophore.contains(bond.b) {
        let a = p.atoms[bond.a].position, b = p.atoms[bond.b].position
        let conjugated: Bool = inPi.contains(bond.a) && inPi.contains(bond.b)
        let r: Float = conjugated ? bondRadius * 1.35 : bondRadius * 0.8
        cylinders.append(GPUCylinder(aRadius: SIMD4(a.x, a.y, a.z, r),
                                     b: SIMD4(b.x, b.y, b.z, 0),
                                     color: conjugated ? liveStick : dullStick))
    }
    return (spheres, cylinders)
}

/// The GIF palette keeps shades of both emission colours whatever else it picks.
/// Median cut hands colours to whatever covers the most pixels, and the two
/// chromophores are small and only on screen for part of the loop — exactly the
/// trap step 8 hit with phosphorus and step 9 with its green.
func pairPalette(_ pair: ProteinPair, samples: [RGB], count: Int) -> [RGB] {
    var reserved: [RGB] = []
    for p in pair.both {
        let c: SIMD3<Float> = p.emissionColour
        for light: Float in [0.3, 0.55, 0.8, 1.0] {
            let v = simd_clamp(c * light, SIMD3<Float>(repeating: 0), SIMD3<Float>(repeating: 1))
            let r: Float = gammaEncode(v.x) * 255
            let g: Float = gammaEncode(v.y) * 255
            let b: Float = gammaEncode(v.z) * 255
            reserved.append(RGB(UInt8(r.rounded()), UInt8(g.rounded()), UInt8(b.rounded())))
        }
    }
    return reserved + medianCutPalette(samples, count: count - reserved.count)
}
