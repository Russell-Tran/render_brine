// The DNA duplex from Resources/dna.json (made by Tools/build_dna.py from
// the 1BNA crystal structure), turned about its own helix axis and laid
// across the frame, as balls, sticks and dashed hydrogen bonds.

import Foundation
import simd

func radians(_ degrees: Float) -> Float { degrees * .pi / 180 }

struct Atom {
    var element: String
    var position: SIMD3<Float>   // helix axis along x, centered on the origin
    var charge: Int
    var label: String            // e.g. "A5.N6": strand A, residue 5, atom N6
}

struct HydrogenBond {
    var donor: Int
    var hydrogen: Int
    var acceptor: Int
    var pair: String             // e.g. "A5–T20"
}

struct DNA {
    var sequence: String
    var atoms: [Atom]
    var bonds: [(a: Int, b: Int, order: Int)]
    var hbonds: [HydrogenBond]
    var rise: [Float]            // Å per base-pair step, measured
    var twist: [Float]           // degrees per step, measured (+ = right-handed)
}

enum SceneError: Error, CustomStringConvertible {
    case badFile(String)
    var description: String {
        switch self {
        case .badFile(let detail): return "couldn't read the DNA: \(detail)"
        }
    }
}

func loadDNA(from url: URL) throws -> DNA {
    let data = try Data(contentsOf: url)
    guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
          let rawAtoms = root["atoms"] as? [[String: Any]],
          let rawBonds = root["bonds"] as? [[Int]],
          let rawH = root["hbonds"] as? [[String: Any]] else { throw SceneError.badFile("missing fields") }
    var atoms: [Atom] = []
    for a in rawAtoms {
        guard let el = a["el"] as? String, let p = a["pos"] as? [NSNumber], p.count == 3 else {
            throw SceneError.badFile("bad atom")
        }
        atoms.append(Atom(element: el, position: SIMD3(p[0].floatValue, p[1].floatValue, p[2].floatValue),
                          charge: a["charge"] as? Int ?? 0, label: a["label"] as? String ?? ""))
    }
    let bonds = rawBonds.map { (a: $0[0], b: $0[1], order: $0[2]) }
    let hbonds = rawH.map { HydrogenBond(donor: $0["donor"] as? Int ?? 0, hydrogen: $0["h"] as? Int ?? 0,
                                         acceptor: $0["acceptor"] as? Int ?? 0, pair: $0["pair"] as? String ?? "") }
    let rise = (root["rise"] as? [NSNumber] ?? []).map { $0.floatValue }
    let twist = (root["twist"] as? [NSNumber] ?? []).map { $0.floatValue }
    return DNA(sequence: root["sequence"] as? String ?? "", atoms: atoms, bonds: bonds, hbonds: hbonds,
               rise: rise, twist: twist)
}

/// How the helix sits in the frame: its axis tilted this far from horizontal,
/// rising to the right.
let axisTilt: Float = 15

/// Where atom position `p` is after turning the helix `angle` degrees about its
/// own axis (x) and tilting the axis across the frame.
func place(_ p: SIMD3<Float>, turn angle: Float) -> SIMD3<Float> {
    let a = radians(angle)
    let spun = SIMD3<Float>(p.x, p.y * cos(a) - p.z * sin(a), p.y * sin(a) + p.z * cos(a))
    let t = radians(axisTilt)
    return SIMD3(spun.x * cos(t) - spun.y * sin(t), spun.x * sin(t) + spun.y * cos(t), spun.z)
}

/// Fixed camera straight on; the helix does the turning, under a fixed light.
func dnaCamera() -> Camera {
    Camera.orbit(target: SIMD3(0, 0, 0), distance: 68, yaw: 0, pitch: 0, fov: 30)
}

func ballRadius(_ element: String) -> Float {
    switch element {
    case "C": return 0.34
    case "N": return 0.33
    case "O": return 0.32
    case "P": return 0.40
    default: return 0.22     // H
    }
}

/// CPK colors: carbon dark grey, nitrogen blue, oxygen red, phosphorus orange,
/// hydrogen white.
func elementColor(_ element: String) -> SIMD3<Float> {
    switch element {
    case "C": return SIMD3(0.28, 0.29, 0.31)
    case "N": return SIMD3(0.20, 0.36, 0.92)
    case "O": return SIMD3(0.86, 0.16, 0.14)
    case "P": return SIMD3(1.0, 0.55, 0.10)
    default: return SIMD3(0.93, 0.93, 0.93)
    }
}

/// The GIF's palette: a few shades of every element's color, always kept, plus
/// the rest chosen by median cut from sample frames. Median cut alone gives
/// colors to what covers the most pixels, so rare colors lose out: the 22
/// phosphorus atoms' shaded sides came out red, the wrong element.
func dnaPalette(samples: [RGB], count: Int) -> [RGB] {
    var reserved: [RGB] = []
    for element in ["C", "N", "O", "P", "H"] {
        let c = elementColor(element)
        for light: Float in [0.35, 0.6, 0.85, 1.1] {
            let v = simd_clamp(c * light, SIMD3<Float>(repeating: 0), SIMD3<Float>(repeating: 1)) * 255
            reserved.append(RGB(UInt8(v.x.rounded()), UInt8(v.y.rounded()), UInt8(v.z.rounded())))
        }
    }
    return reserved + medianCutPalette(samples, count: count - reserved.count)
}

/// Hydrogen bonds are dashed: this long a dash, this long a gap (Å).
let dashLength: Float = 0.22
let dashGap: Float = 0.16
let hbondRadius: Float = 0.05
let hbondColor = SIMD4<Float>(0.80, 0.88, 0.97, 1)

/// The dashes along a line from `a` to `b`, centered so both ends match.
func dashes(from a: SIMD3<Float>, to b: SIMD3<Float>) -> [(SIMD3<Float>, SIMD3<Float>)] {
    let length = simd_distance(a, b)
    let period = dashLength + dashGap
    let count = max(Int((length + dashGap) / period), 1)
    let used = Float(count) * period - dashGap
    let dir = (b - a) / length
    var start = a + dir * ((length - used) / 2)
    var out: [(SIMD3<Float>, SIMD3<Float>)] = []
    for _ in 0..<count {
        out.append((start, start + dir * dashLength))
        start += dir * period
    }
    return out
}

func sceneGeometry(_ dna: DNA, turn angle: Float, camera: Camera) -> (spheres: [GPUSphere], cylinders: [GPUCylinder]) {
    let placed = dna.atoms.map { place($0.position, turn: angle) }
    var spheres: [GPUSphere] = []
    spheres.reserveCapacity(placed.count)
    for (i, atom) in dna.atoms.enumerated() {
        let p = placed[i], c = elementColor(atom.element)
        spheres.append(GPUSphere(centerRadius: SIMD4(p.x, p.y, p.z, ballRadius(atom.element)),
                                 color: SIMD4(c.x, c.y, c.z, 1), glow: .zero))
    }
    var cylinders: [GPUCylinder] = []
    let grey = SIMD4<Float>(0.62, 0.64, 0.67, 1)
    for bond in dna.bonds {
        let a = placed[bond.a], b = placed[bond.b]
        if bond.order == 1 {
            cylinders.append(GPUCylinder(aRadius: SIMD4(a.x, a.y, a.z, bondRadius), b: SIMD4(b.x, b.y, b.z, 0), color: grey))
        } else {
            // Two sticks side by side, offset across the line of sight.
            var side = simd_cross(simd_normalize(b - a), camera.forward)
            if simd_length(side) < 1e-4 { side = simd_cross(simd_normalize(b - a), SIMD3(0, 1, 0)) }
            side = simd_normalize(side) * doubleBondOffset
            for s in [side, -side] {
                cylinders.append(GPUCylinder(aRadius: SIMD4(a.x + s.x, a.y + s.y, a.z + s.z, bondRadius),
                                             b: SIMD4(b.x + s.x, b.y + s.y, b.z + s.z, 0), color: grey))
            }
        }
    }
    for h in dna.hbonds {
        // From the hydrogen's surface to the acceptor's, so dashes don't hide in the balls.
        let hp = placed[h.hydrogen], ap = placed[h.acceptor]
        let dir = simd_normalize(ap - hp)
        let from = hp + dir * ballRadius("H")
        let to = ap - dir * ballRadius(dna.atoms[h.acceptor].element)
        for (s, e) in dashes(from: from, to: to) {
            cylinders.append(GPUCylinder(aRadius: SIMD4(s.x, s.y, s.z, hbondRadius), b: SIMD4(e.x, e.y, e.z, 0),
                                         color: hbondColor))
        }
    }
    return (spheres, cylinders)
}

/// Mean of a list, 0 when empty.
func mean(_ xs: [Float]) -> Float { xs.isEmpty ? 0 : xs.reduce(0, +) / Float(xs.count) }
