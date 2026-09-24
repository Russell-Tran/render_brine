// One carbonic acid molecule giving up a proton:
//
//     H₂CO₃  →  H⁺ + HCO₃⁻
//
// The molecule is flat, so its atoms are placed in the x–y plane, in ångströms
// (1 Å = 0.1 nm), with carbon at the origin.
//
// Carbonic acid (the lowest-energy "cis-cis" form: both H's lean toward the
// C=O oxygen): C=O 1.222 Å, C–OH 1.357 Å, O–H 0.980 Å (computed gas-phase
// geometry). Angles O=C–O 125°, C–O–H 106°.
//
// Bicarbonate: the oxygen that lost its H and the old C=O oxygen become
// equivalent. They share the double bond and the −1 charge (resonance), so
// both bonds settle near 1.25 Å (carbonate-like, as in crystal structures),
// with C–OH near 1.36 Å and O–H 0.97 Å. Angles O–C–O 126° between the
// equivalent pair and 117° to the OH.
//
// The angles are approximate; the point is the real change: two different
// C–O bonds become two identical one-and-a-half bonds.

import Foundation
import simd

enum Element {
    case carbon, oxygen, hydrogen

    /// Ball radius for a ball-and-stick picture (Å), not the atom's true size.
    var ballRadius: Float {
        switch self {
        case .carbon: return 0.34
        case .oxygen: return 0.32
        case .hydrogen: return 0.22
        }
    }
    /// Standard CPK colors: carbon dark grey, oxygen red, hydrogen white.
    var color: SIMD3<Float> {
        switch self {
        case .carbon: return SIMD3(0.28, 0.29, 0.31)
        case .oxygen: return SIMD3(0.86, 0.16, 0.14)
        case .hydrogen: return SIMD3(0.93, 0.93, 0.93)
        }
    }
}

struct Atom {
    var element: Element
    var position: SIMD3<Float>
    var charge: Float = 0        // formal charge, shown as a glow and a label
}

struct Bond {
    var a: Int
    var b: Int
    var order: Float             // 1 single, 2 double, 1.5 shared by resonance
}

/// A snapshot of the reaction. Atom order never changes:
/// 0 C, 1 O (the C=O oxygen), 2 O (keeps its H), 3 O (loses its H),
/// 4 H (on O2), 5 H (the proton that leaves).
struct MoleculeState {
    var atoms: [Atom]
    var bonds: [Bond]
}

// Geometry of each end state, as polar coordinates around carbon.
struct PlanarGeometry {
    var angles: [Float]          // degrees clockwise from +y, for O1, O2, O3
    var lengths: [Float]         // C–O bond lengths for O1, O2, O3 (Å)
    var hydrogenBond: Float      // O–H length (Å)
    var cOH: Float               // C–O–H angle (degrees)
}

let carbonicAcidGeometry = PlanarGeometry(angles: [0, 125, -125], lengths: [1.222, 1.357, 1.357],
                                          hydrogenBond: 0.980, cOH: 106)
let bicarbonateGeometry = PlanarGeometry(angles: [8, 125, -118], lengths: [1.25, 1.36, 1.25],
                                         hydrogenBond: 0.970, cOH: 106)

func radians(_ degrees: Float) -> Float { degrees * .pi / 180 }

/// A unit vector in the x–y plane, `degrees` clockwise from +y.
func planarDirection(_ degrees: Float) -> SIMD3<Float> {
    let a = radians(degrees)
    return SIMD3(sin(a), cos(a), 0)
}

/// Rotates a vector in the x–y plane by `degrees` (counterclockwise).
func rotateInPlane(_ v: SIMD3<Float>, by degrees: Float) -> SIMD3<Float> {
    let a = radians(degrees)
    let c = cos(a), s = sin(a)
    return SIMD3(c * v.x - s * v.y, s * v.x + c * v.y, 0)
}

/// Places a hydrogen on `oxygen` with the C–O–H angle given, on the same side
/// of the C–O line as `sameSideAs` ("cis": it leans toward that atom).
func placeHydrogen(oxygen: SIMD3<Float>, carbon: SIMD3<Float>, sameSideAs ref: SIMD3<Float>,
                   angle: Float, length: Float) -> SIMD3<Float> {
    let back = simd_normalize(carbon - oxygen)
    let bondLine = oxygen - carbon
    let refSide: Float = simd_cross(bondLine, ref - carbon).z
    for turn in [angle, -angle] {
        let candidate = oxygen + rotateInPlane(back, by: turn) * length
        let side: Float = simd_cross(bondLine, candidate - carbon).z
        if (side > 0) == (refSide > 0) { return candidate }
    }
    return oxygen + rotateInPlane(back, by: angle) * length
}

func mix(_ a: Float, _ b: Float, _ t: Float) -> Float { a + (b - a) * t }

/// Smooth 0→1 easing, so the shape change starts and ends gently.
func smoothstep01(_ t: Float) -> Float {
    let x = min(max(t, 0), 1)
    return x * x * (3 - 2 * x)
}

/// The molecule partway through losing its proton.
/// - `progress`: 0 = carbonic acid, 1 = bicarbonate (the bonds rearranging).
/// - `protonDistance`: how far the leaving H is from its oxygen (Å).
func moleculeState(progress: Float, protonDistance: Float) -> MoleculeState {
    let s = smoothstep01(progress)
    let a = carbonicAcidGeometry, b = bicarbonateGeometry
    let carbon = SIMD3<Float>(0, 0, 0)
    var oxygens: [SIMD3<Float>] = []
    for i in 0..<3 {
        let angle: Float = mix(a.angles[i], b.angles[i], s)
        let length: Float = mix(a.lengths[i], b.lengths[i], s)
        oxygens.append(planarDirection(angle) * length)
    }
    let ohLength: Float = mix(a.hydrogenBond, b.hydrogenBond, s)
    let h4 = placeHydrogen(oxygen: oxygens[1], carbon: carbon, sameSideAs: oxygens[0],
                           angle: a.cOH, length: ohLength)
    // The leaving proton moves straight out along its old O–H direction.
    let h5Home = placeHydrogen(oxygen: oxygens[2], carbon: carbon, sameSideAs: oxygens[0],
                               angle: a.cOH, length: 1)
    let h5Direction = simd_normalize(h5Home - oxygens[2])
    let h5 = oxygens[2] + h5Direction * protonDistance

    let atoms = [
        Atom(element: .carbon, position: carbon),
        Atom(element: .oxygen, position: oxygens[0], charge: -0.5 * s),
        Atom(element: .oxygen, position: oxygens[1]),
        Atom(element: .oxygen, position: oxygens[2], charge: -0.5 * s),
        Atom(element: .hydrogen, position: h4),
        Atom(element: .hydrogen, position: h5, charge: s),
    ]
    // The O–H bond to the leaving proton fades as it pulls away.
    let bondStretch: Float = (protonDistance - a.hydrogenBond) / 0.6
    let leavingOrder: Float = 1 - min(max(bondStretch, 0), 1)
    let bonds = [
        Bond(a: 0, b: 1, order: mix(2, 1.5, s)),
        Bond(a: 0, b: 2, order: 1),
        Bond(a: 0, b: 3, order: mix(1, 1.5, s)),
        Bond(a: 2, b: 4, order: 1),
        Bond(a: 3, b: 5, order: leavingOrder),
    ]
    return MoleculeState(atoms: atoms, bonds: bonds)
}

func distance(_ state: MoleculeState, _ i: Int, _ j: Int) -> Float {
    simd_distance(state.atoms[i].position, state.atoms[j].position)
}

/// Angle at atom `j` between atoms `i` and `k`, in degrees.
func angle(_ state: MoleculeState, _ i: Int, _ j: Int, _ k: Int) -> Float {
    let u = simd_normalize(state.atoms[i].position - state.atoms[j].position)
    let v = simd_normalize(state.atoms[k].position - state.atoms[j].position)
    return acos(min(max(simd_dot(u, v), -1), 1)) * 180 / .pi
}

/// Sum of the bond orders touching atom `i`.
func valence(_ state: MoleculeState, of i: Int) -> Float {
    state.bonds.filter { $0.a == i || $0.b == i }.reduce(0) { $0 + $1.order }
}

// MARK: - The animation timeline

/// Where the story is at `seconds` into the animation.
struct Timeline {
    let calm: Double = 1.6         // carbonic acid, intact
    let split: Double = 2.0        // proton leaves, bonds rearrange
    let after: Double = 2.4        // bicarbonate settles, H⁺ drifts off
    var total: Double { calm + split + after }

    /// 0 → 1 across the split.
    func progress(at t: Double) -> Float {
        Float(min(max((t - calm) / split, 0), 1))
    }

    /// The leaving proton's distance from its oxygen (Å): at rest, then
    /// pulling away, then drifting a little further and slowing to a stop,
    /// so both products stay in view.
    func protonDistance(at t: Double) -> Float {
        let rest: Float = carbonicAcidGeometry.hydrogenBond
        if t < calm { return rest }
        let p: Float = progress(at: t)
        let pulled: Float = rest + 1.4 * p * p
        let afterSplit: Double = max(t - calm - split, 0)
        let drift: Float = 1.3 * Float(1 - exp(-afterSplit / 0.8))
        return pulled + drift
    }

    func stage(at t: Double) -> String {
        if t < calm { return "Carbonic acid, H₂CO₃" }
        if t < calm + split { return "H₂CO₃ → H⁺ + HCO₃⁻" }
        return "Bicarbonate, HCO₃⁻, and a free H⁺"
    }
}
