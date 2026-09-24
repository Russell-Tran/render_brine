// Reads the keyframes made by Tools/build_timeline.py and blends between them.
//
// Every atom in a movie has a permanent label (glucose's C1..C6 keep their
// numbers all the way to pyruvate). A keyframe gives each atom's position,
// visibility, charge and whether it glows (a free H⁺), plus the bonds, the
// ATP/ADP/NAD⁺ tokens and the caption.

import Foundation
import simd

func radians(_ degrees: Float) -> Float { degrees * .pi / 180 }

struct AtomKey {
    var position: SIMD3<Float>
    var visible: Float
    var charge: Float
    var glow: Float
}

struct TokenKey {
    var text: String
    var position: SIMD3<Float>
    var alpha: Float
}

struct BondKey: Hashable {
    let a: String
    let b: String
    init(_ a: String, _ b: String) {
        if a < b { self.a = a; self.b = b } else { self.a = b; self.b = a }
    }
}

struct Keyframe {
    var time: Double
    var atoms: [String: AtomKey]
    var bonds: [BondKey: Float]
    var tokens: [String: TokenKey]
    var title: String
    var enzyme: String
    var equation: String
    var ledger: [Int]           // ATP spent, ATP made, NADH made
}

struct Movie {
    var name: String
    var duration: Double
    var cameraDistance: Float
    var cameraSwing: Float
    var elements: [String: String]
    var keys: [Keyframe]
}

enum MovieError: Error, CustomStringConvertible {
    case badFile(String)
    var description: String {
        switch self {
        case .badFile(let detail): return "couldn't read the timeline: \(detail)"
        }
    }
}

private func vector(_ a: [Any], _ from: Int) -> SIMD3<Float> {
    let x = (a[from] as? NSNumber)?.floatValue ?? 0
    let y = (a[from + 1] as? NSNumber)?.floatValue ?? 0
    let z = (a[from + 2] as? NSNumber)?.floatValue ?? 0
    return SIMD3(x, y, z)
}

func loadMovies(from url: URL) throws -> [Movie] {
    let data = try Data(contentsOf: url)
    guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
          let gifs = root["gifs"] as? [[String: Any]] else { throw MovieError.badFile("no gifs") }
    var movies: [Movie] = []
    for gif in gifs {
        guard let name = gif["name"] as? String, let duration = gif["duration"] as? Double,
              let elements = gif["elements"] as? [String: String], let rawKeys = gif["keys"] as? [[String: Any]]
        else { throw MovieError.badFile("bad gif entry") }
        var keys: [Keyframe] = []
        for k in rawKeys {
            var atoms: [String: AtomKey] = [:]
            for (lbl, value) in (k["atoms"] as? [String: [Any]]) ?? [:] {
                let vis = (value[3] as? NSNumber)?.floatValue ?? 1
                let charge = (value[4] as? NSNumber)?.floatValue ?? 0
                let glow = (value[5] as? NSNumber)?.floatValue ?? 0
                atoms[lbl] = AtomKey(position: vector(value, 0), visible: vis, charge: charge, glow: glow)
            }
            var bonds: [BondKey: Float] = [:]
            for b in (k["bonds"] as? [[Any]]) ?? [] {
                if let a = b[0] as? String, let c = b[1] as? String, let o = b[2] as? NSNumber {
                    bonds[BondKey(a, c)] = o.floatValue
                }
            }
            var tokens: [String: TokenKey] = [:]
            for (id, value) in (k["tokens"] as? [String: [Any]]) ?? [:] {
                let text = value[0] as? String ?? ""
                let alpha = (value[4] as? NSNumber)?.floatValue ?? 0
                tokens[id] = TokenKey(text: text, position: vector(value, 1), alpha: alpha)
            }
            keys.append(Keyframe(time: (k["t"] as? NSNumber)?.doubleValue ?? 0, atoms: atoms, bonds: bonds,
                                 tokens: tokens, title: k["title"] as? String ?? "", enzyme: k["enzyme"] as? String ?? "",
                                 equation: k["equation"] as? String ?? "", ledger: k["ledger"] as? [Int] ?? [0, 0, 0]))
        }
        let distance = (gif["camera_distance"] as? NSNumber)?.floatValue ?? 25
        let swing = (gif["camera_swing"] as? NSNumber)?.floatValue ?? 12
        movies.append(Movie(name: name, duration: duration, cameraDistance: distance, cameraSwing: swing,
                            elements: elements, keys: keys))
    }
    return movies
}

/// Smooth 0→1 easing, so motion starts and stops gently.
func smoothstep01(_ t: Float) -> Float {
    let x = min(max(t, 0), 1)
    return x * x * (3 - 2 * x)
}

/// Everything on screen at one instant.
struct Frame {
    var atoms: [(label: String, element: String, key: AtomKey)]
    var bonds: [(a: String, b: String, order: Float)]
    var tokens: [TokenKey]
    var title: String
    var enzyme: String
    var equation: String
    var ledger: [Int]
}

extension Movie {
    /// A camera that sways gently side to side and comes back exactly at the
    /// end, so the loop is seamless.
    func camera(at t: Double) -> Camera {
        let yaw = cameraSwing * Float(sin(2 * Double.pi * t / duration))
        return Camera.orbit(target: SIMD3(0, 0, 0), distance: cameraDistance, yaw: yaw, pitch: 8, fov: 30)
    }

    /// The movie at time `t` seconds (wrapping around, so it loops).
    func frame(at time: Double) -> Frame {
        var t = time.truncatingRemainder(dividingBy: duration)
        if t < 0 { t += duration }
        var i = 0
        while i + 2 < keys.count && keys[i + 1].time <= t { i += 1 }
        let k0 = keys[i], k1 = keys[i + 1]
        let span: Double = max(k1.time - k0.time, 1e-6)
        let s: Float = smoothstep01(Float((t - k0.time) / span))

        var atoms: [(label: String, element: String, key: AtomKey)] = []
        for label in elements.keys.sorted() {
            let a0 = k0.atoms[label], a1 = k1.atoms[label]
            guard let from = a0 ?? a1, let to = a1 ?? a0 else { continue }
            let v0: Float = a0 == nil ? 0 : from.visible
            let v1: Float = a1 == nil ? 0 : to.visible
            let pos = from.position + (to.position - from.position) * s
            let key = AtomKey(position: pos, visible: v0 + (v1 - v0) * s,
                              charge: s < 0.5 ? from.charge : to.charge,
                              glow: from.glow + (to.glow - from.glow) * s)
            atoms.append((label, elements[label] ?? "C", key))
        }
        var bonds: [(a: String, b: String, order: Float)] = []
        for key in Set(k0.bonds.keys).union(k1.bonds.keys) {
            let o0: Float = k0.bonds[key] ?? 0
            let o1: Float = k1.bonds[key] ?? 0
            let order: Float = o0 + (o1 - o0) * s
            if order > 0.01 { bonds.append((key.a, key.b, order)) }
        }
        bonds.sort { ($0.a, $0.b) < ($1.a, $1.b) }
        var tokens: [TokenKey] = []
        for id in Set(k0.tokens.keys).union(k1.tokens.keys).sorted() {
            let a = k0.tokens[id], b = k1.tokens[id]
            guard let from = a ?? b, let to = b ?? a else { continue }
            let alpha0: Float = a == nil ? 0 : from.alpha
            let alpha1: Float = b == nil ? 0 : to.alpha
            tokens.append(TokenKey(text: s < 0.5 ? from.text : to.text,
                                   position: from.position + (to.position - from.position) * s,
                                   alpha: alpha0 + (alpha1 - alpha0) * s))
        }
        let ledger = s < 0.85 ? k0.ledger : k1.ledger
        return Frame(atoms: atoms, bonds: bonds, tokens: tokens, title: k1.title, enzyme: k1.enzyme,
                     equation: k1.equation, ledger: ledger)
    }
}

// MARK: - Turning a frame into spheres and cylinders

func ballRadius(_ element: String) -> Float {
    switch element {
    case "C": return 0.34
    case "O": return 0.32
    case "P": return 0.40
    default: return 0.22     // H
    }
}

/// CPK colors: carbon dark grey, oxygen red, hydrogen white, phosphorus orange.
func elementColor(_ element: String) -> SIMD3<Float> {
    switch element {
    case "C": return SIMD3(0.28, 0.29, 0.31)
    case "O": return SIMD3(0.86, 0.16, 0.14)
    case "P": return SIMD3(1.0, 0.55, 0.10)
    default: return SIMD3(0.93, 0.93, 0.93)
    }
}

/// Bonds fade out as their atoms separate, so a bond forming between atoms
/// that are still far apart doesn't draw a long stick across the screen.
func bondFade(length: Float) -> Float {
    min(max((2.3 - length) / 0.5, 0), 1)
}

func sceneGeometry(_ frame: Frame, camera: Camera) -> (spheres: [GPUSphere], cylinders: [GPUCylinder]) {
    var index: [String: Int] = [:]
    var spheres: [GPUSphere] = []
    for (i, atom) in frame.atoms.enumerated() {
        index[atom.label] = i
        let v = atom.key.visible
        guard v > 0.01 else { continue }
        let p = atom.key.position
        let c = elementColor(atom.element)
        var glow = SIMD4<Float>(0, 0, 0, 0)
        if atom.key.glow > 0.01 {
            let g = positiveGlow * atom.key.glow * v * 0.8
            glow = SIMD4(g.x, g.y, g.z, 0.32)
        }
        spheres.append(GPUSphere(centerRadius: SIMD4(p.x, p.y, p.z, ballRadius(atom.element) * v),
                                 color: SIMD4(c.x, c.y, c.z, 1), glow: glow))
    }
    var cylinders: [GPUCylinder] = []
    let grey = SIMD4<Float>(0.62, 0.64, 0.67, 1)
    for bond in frame.bonds {
        guard let i = index[bond.a], let j = index[bond.b] else { continue }
        let atomA = frame.atoms[i].key, atomB = frame.atoms[j].key
        let vis: Float = min(atomA.visible, atomB.visible)
        let a = atomA.position, b = atomB.position
        let length: Float = simd_distance(a, b)
        let strength: Float = vis * bondFade(length: length)
        guard strength > 0.01, length > 1e-4 else { continue }
        let along = (b - a) / length
        var side = simd_cross(along, camera.forward)
        if simd_length(side) < 1e-4 { side = simd_cross(along, SIMD3(0, 1, 0)) }
        side = simd_normalize(side)
        if bond.order <= 1 {
            let r: Float = bondRadius * bond.order * strength
            cylinders.append(GPUCylinder(aRadius: SIMD4(a.x, a.y, a.z, r), b: SIMD4(b.x, b.y, b.z, 0), color: grey))
        } else {
            let extra: Float = bond.order - 1
            let shiftMain = side * (-doubleBondOffset * extra)
            let shiftExtra = side * doubleBondOffset * extra
            let a1 = a + shiftMain, b1 = b + shiftMain, a2 = a + shiftExtra, b2 = b + shiftExtra
            cylinders.append(GPUCylinder(aRadius: SIMD4(a1.x, a1.y, a1.z, bondRadius * strength),
                                         b: SIMD4(b1.x, b1.y, b1.z, 0), color: grey))
            cylinders.append(GPUCylinder(aRadius: SIMD4(a2.x, a2.y, a2.z, bondRadius * extra * strength),
                                         b: SIMD4(b2.x, b2.y, b2.z, 0), color: grey))
        }
    }
    return (spheres, cylinders)
}
