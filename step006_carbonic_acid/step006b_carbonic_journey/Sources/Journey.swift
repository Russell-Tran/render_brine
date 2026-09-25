// The whole carbonic acid chain, one molecule of each, as a looping journey:
//
//     CO₂ + H₂O  →  H₂CO₃  →  HCO₃⁻ + H₃O⁺
//
// Each cycle uses one CO₂, one attacking water and one "helper" water:
//   act 1  CO₂, the water and the helper drift together
//   act 2  the water's O bonds to carbon and CO₂ bends (180° → 125°), while a
//          proton is relayed through the helper water to one of CO₂'s oxygens
//          (a concerted, cyclic relay: the route computational studies suggest,
//          not a settled fact)
//   act 3  carbonic acid; its new OH turns to the most stable (cis-cis) shape
//          and the helper moves around to its other side
//   act 4  the OH proton moves to the helper, making H₃O⁺ + HCO₃⁻
//   act 5  the products fall away while the next CO₂ and waters float in
//
// The in-between motion is drawn smoothly between known structures. It is not
// a quantum-chemistry simulation.
//
// Geometry sources (Å = 0.1 nm):
//   CO₂: C=O 1.163 Å, linear (Wikipedia, "Carbon dioxide").
//   H₂O: O–H 0.9572 Å, H–O–H 104.52° (gas-phase values).
//   H₂CO₃ (cis-cis): C=O 1.222 Å, C–OH 1.357 Å, O–H 0.980 Å, O=C–O 125°,
//     C–O–H 106° (computed gas-phase geometry, as in step 6a).
//   HCO₃⁻: two equal C–O 1.25 Å, C–OH 1.36 Å, O–H 0.97 Å, angles 126° / 117°
//     (typical of carbonate-type bonds in crystal structures, as in step 6a).
//   H₃O⁺: O–H 0.974 Å, H–O–H 113.6°, pyramidal (infrared spectroscopy, via
//     Wikipedia, "Hydronium").

import Foundation
import simd

enum Element: Equatable {
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
    /// Formal charge (shown as a label).
    var charge: Float = 0
    /// Glow behind the atom: positive = gold, negative = cyan (the renderer reads this).
    var glow: Float = 0
}

struct Bond: Equatable {
    var a: Int
    var b: Int
    var order: Float
}

struct MoleculeState {
    var atoms: [Atom]
    var bonds: [Bond]
}

func radians(_ degrees: Float) -> Float { degrees * .pi / 180 }
func mix(_ a: Float, _ b: Float, _ t: Float) -> Float { a + (b - a) * t }
func mix3(_ a: SIMD3<Float>, _ b: SIMD3<Float>, _ t: Float) -> SIMD3<Float> { a + (b - a) * t }
func smoothstep01(_ t: Float) -> Float {
    let x = min(max(t, 0), 1)
    return x * x * (3 - 2 * x)
}

// MARK: - Measured sizes

let carbonDioxideBond: Float = 1.163
let waterBond: Float = 0.9572
let waterAngle: Float = 104.52
let hydroniumBond: Float = 0.974
let hydroniumAngle: Float = 113.6

struct PlanarShape {
    var angles: [Float]     // degrees clockwise from +y, for O1, O2 (keeps its H), O3
    var lengths: [Float]    // C–O lengths for O1, O2, O3
    var hydrogenBond: Float // O–H
    var cOH: Float          // C–O–H angle
}
let carbonicAcidShape = PlanarShape(angles: [0, 125, -125], lengths: [1.222, 1.357, 1.357],
                                    hydrogenBond: 0.980, cOH: 106)
let bicarbonateShape = PlanarShape(angles: [8, 125, -118], lengths: [1.25, 1.36, 1.25],
                                   hydrogenBond: 0.970, cOH: 106)

// MARK: - Atoms of one cycle

/// Atom indices within one cycle's group of nine atoms.
enum Slot {
    static let carbon = 0
    static let o1 = 1          // CO₂ oxygen that stays double-bonded, then shares the charge
    static let o3 = 2          // CO₂ oxygen that picks up the relayed proton, then loses it
    static let o2 = 3          // the attacking water's oxygen; keeps one H
    static let hKept = 4       // stays on o2 throughout
    static let hRelayed = 5    // from the attacking water to the helper
    static let helperO = 6     // the helper water's oxygen; ends as H₃O⁺
    static let proton = 7      // helper → o3 (act 2) → helper (act 4)
    static let hHelper = 8     // stays on the helper
    static let count = 9
}

let groupElements: [Element] = [.carbon, .oxygen, .oxygen, .oxygen, .hydrogen, .hydrogen, .oxygen, .hydrogen, .hydrogen]

/// Bonds in a fixed order; only their orders change.
let groupBondPairs: [(Int, Int)] = [
    (Slot.carbon, Slot.o1), (Slot.carbon, Slot.o3), (Slot.carbon, Slot.o2),
    (Slot.o2, Slot.hKept), (Slot.o2, Slot.hRelayed), (Slot.helperO, Slot.hRelayed),
    (Slot.helperO, Slot.proton), (Slot.o3, Slot.proton), (Slot.helperO, Slot.hHelper),
]

/// Bond orders for hydration progress `s` (act 2) and proton-transfer progress
/// `u` (act 4). Every atom's bonds stay consistent with its charge throughout.
func bondOrders(hydration s: Float, transfer u: Float) -> [Float] {
    [
        2 - 0.5 * u,        // C–O1: double, then one-and-a-half
        2 - s + 0.5 * u,    // C–O3: double → single → one-and-a-half
        s,                  // C–O2: forms
        1,                  // O2–H kept
        1 - s,              // O2–H relayed: breaks
        s,                  // helper–H relayed: forms
        1 - s + u,          // helper–proton: breaks in act 2, re-forms in act 4
        s - u,              // O3–proton: forms in act 2, breaks in act 4
        1,                  // helper–H kept
    ]
}

/// Formal charges: bicarbonate's two free oxygens share −1, and H₃O⁺'s oxygen carries +1.
func charges(transfer u: Float) -> [Float] {
    var q = [Float](repeating: 0, count: Slot.count)
    q[Slot.o1] = -0.5 * u
    q[Slot.o3] = -0.5 * u
    q[Slot.helperO] = u
    return q
}

// MARK: - Geometry helpers

/// A unit vector in the x–y plane, `degrees` clockwise from +y.
func planar(_ degrees: Float) -> SIMD3<Float> {
    let a = radians(degrees)
    return SIMD3(sin(a), cos(a), 0)
}

/// Rotates a vector in the x–y plane by `degrees` counterclockwise.
func rotateInPlane(_ v: SIMD3<Float>, by degrees: Float) -> SIMD3<Float> {
    let a = radians(degrees)
    let c = cos(a), s = sin(a)
    return SIMD3(c * v.x - s * v.y, s * v.x + c * v.y, v.z)
}

/// Direction of an H on `oxygen` making angle `cOH` at the oxygen, on the same
/// side of the C–O line as `ref` (cis) or the opposite side (trans).
func hydrogenDirection(oxygen: SIMD3<Float>, carbon: SIMD3<Float>, ref: SIMD3<Float>, cis: Bool,
                       cOH: Float) -> SIMD3<Float> {
    let back = simd_normalize(carbon - oxygen)
    let line = oxygen - carbon
    let refSide: Float = simd_cross(line, ref - carbon).z
    for turn in [cOH, -cOH] {
        let d = rotateInPlane(back, by: turn)
        let side: Float = simd_cross(line, oxygen + d - carbon).z
        if ((side > 0) == (refSide > 0)) == cis { return d }
    }
    return rotateInPlane(back, by: cOH)
}

/// The two H directions of a water whose H–O–H angle bisector is `bisector`,
/// spread in the plane containing `bisector` and `spread`.
func waterDirections(bisector: SIMD3<Float>, spread: SIMD3<Float>, angle: Float = waterAngle)
    -> (SIMD3<Float>, SIMD3<Float>) {
    let m = simd_normalize(bisector)
    let n = simd_normalize(spread - m * simd_dot(spread, m))
    let half = radians(angle / 2)
    return (m * cos(half) + n * sin(half), m * cos(half) - n * sin(half))
}

/// H₃O⁺'s three H directions: the first is `first`; all three make `angle`
/// with each other (pyramidal), tilted toward +z.
func hydroniumDirections(first: SIMD3<Float>, angle: Float = hydroniumAngle) -> [SIMD3<Float>] {
    let u = simd_normalize(first)
    // Each O–H makes angle θ with the pyramid's axis; three evenly spaced
    // around it make the H–O–H angle: cos(HOH) = 1.5 cos²θ − 0.5.
    let cos2: Float = (cos(radians(angle)) + 0.5) / 1.5
    let ct: Float = cos2.squareRoot()
    let st: Float = (1 - cos2).squareRoot()
    var w = SIMD3<Float>(0, 0, 1)
    w = simd_normalize(w - u * simd_dot(w, u))
    let axis = u * ct - w * st
    let b = u * st + w * ct
    let c = simd_cross(axis, b)
    var dirs: [SIMD3<Float>] = []
    for degrees in [Float(0), 120, 240] {
        let p = radians(degrees)
        let around = b * cos(p) + c * sin(p)
        dirs.append(axis * ct + around * st)
    }
    return dirs
}

/// Quadratic Bézier point, for curved paths that steer around atoms.
func bezier(_ p0: SIMD3<Float>, _ control: SIMD3<Float>, _ p1: SIMD3<Float>, _ t: Float) -> SIMD3<Float> {
    let a: Float = (1 - t) * (1 - t)
    let b: Float = 2 * (1 - t) * t
    let c: Float = t * t
    return p0 * a + control * b + p1 * c
}

/// Blends two unit directions (not antiparallel) and renormalizes.
func blendDirection(_ a: SIMD3<Float>, _ b: SIMD3<Float>, _ t: Float) -> SIMD3<Float> {
    simd_normalize(mix3(a, b, t))
}

// MARK: - The key poses (positions of the nine atoms, carbon at the origin)

struct Poses {
    var start: [SIMD3<Float>]        // act 1 begins: molecules a little apart
    var gathered: [SIMD3<Float>]     // act 1 ends: CO₂, water and helper side by side
    var acid: [SIMD3<Float>]         // act 2 ends: H₂CO₃ (new OH trans), helper below
    var cisAcid: [SIMD3<Float>]      // act 3 ends: H₂CO₃ cis-cis, helper ready to accept
    var products: [SIMD3<Float>]     // act 4 ends: HCO₃⁻ + H₃O⁺
    var protonTrans: SIMD3<Float>    // proton direction on O3 at the end of act 2
    var protonCis: SIMD3<Float>      // proton direction on O3 at the end of act 3
    var helperControl: SIMD3<Float>  // bends the helper's path in act 3 around O3
}

func buildPoses() -> Poses {
    let c = SIMD3<Float>(0, 0, 0)
    var p = [SIMD3<Float>](repeating: .zero, count: Slot.count)

    // Gathered: CO₂ upright, the attacking water lower right, the helper below.
    p[Slot.carbon] = c
    p[Slot.o1] = SIMD3(0, carbonDioxideBond, 0)
    p[Slot.o3] = SIMD3(0, -carbonDioxideBond, 0)
    p[Slot.o2] = planar(125) * 2.75
    p[Slot.helperO] = SIMD3(0.35, -3.35, 0)
    let toHelper = simd_normalize(p[Slot.helperO] - p[Slot.o2])
    let up1 = rotateInPlane(toHelper, by: waterAngle), up2 = rotateInPlane(toHelper, by: -waterAngle)
    let keptDir = up1.y > up2.y ? up1 : up2
    p[Slot.hRelayed] = p[Slot.o2] + toHelper * waterBond
    p[Slot.hKept] = p[Slot.o2] + keptDir * waterBond
    let toO3 = simd_normalize(p[Slot.o3] - p[Slot.helperO])
    let h1 = rotateInPlane(toO3, by: waterAngle), h2 = rotateInPlane(toO3, by: -waterAngle)
    let helperDir = simd_distance(p[Slot.helperO] + h1, p[Slot.o2]) > simd_distance(p[Slot.helperO] + h2, p[Slot.o2]) ? h1 : h2
    p[Slot.proton] = p[Slot.helperO] + toO3 * waterBond
    p[Slot.hHelper] = p[Slot.helperO] + helperDir * waterBond
    let gathered = p

    // Start: the three molecules a little apart.
    var start = gathered
    let co2Shift = SIMD3<Float>(-0.6, 1.6, 0)
    let waterShift = SIMD3<Float>(1.9, 0.3, 0)
    let helperShift = SIMD3<Float>(-1.2, -0.3, 0)
    for i in [Slot.carbon, Slot.o1, Slot.o3] { start[i] += co2Shift }
    for i in [Slot.o2, Slot.hKept, Slot.hRelayed] { start[i] += waterShift }
    for i in [Slot.helperO, Slot.proton, Slot.hHelper] { start[i] += helperShift }

    // Acid: H₂CO₃ with the new OH trans (the proton arrives from below), helper beneath it.
    var a = gathered
    let ac = carbonicAcidShape
    a[Slot.o1] = planar(ac.angles[0]) * ac.lengths[0]
    a[Slot.o2] = planar(ac.angles[1]) * ac.lengths[1]
    a[Slot.o3] = planar(ac.angles[2]) * ac.lengths[2]
    a[Slot.hKept] = a[Slot.o2] + hydrogenDirection(oxygen: a[Slot.o2], carbon: c, ref: a[Slot.o1], cis: true,
                                                   cOH: ac.cOH) * ac.hydrogenBond
    let protonTrans = hydrogenDirection(oxygen: a[Slot.o3], carbon: c, ref: a[Slot.o1], cis: false, cOH: ac.cOH)
    a[Slot.proton] = a[Slot.o3] + protonTrans * ac.hydrogenBond
    a[Slot.helperO] = a[Slot.proton] + protonTrans * 1.75
    let toO2 = simd_normalize(a[Slot.o2] - a[Slot.helperO])
    let r1 = rotateInPlane(toO2, by: waterAngle), r2 = rotateInPlane(toO2, by: -waterAngle)
    let otherDir = simd_distance(a[Slot.helperO] + r1, a[Slot.o3]) > simd_distance(a[Slot.helperO] + r2, a[Slot.o3]) ? r1 : r2
    a[Slot.hRelayed] = a[Slot.helperO] + toO2 * waterBond
    a[Slot.hHelper] = a[Slot.helperO] + otherDir * waterBond
    let acid = a

    // Products: bicarbonate, and H₃O⁺ where the helper waits in act 3.
    var d = acid
    let bc = bicarbonateShape
    d[Slot.o1] = planar(bc.angles[0]) * bc.lengths[0]
    d[Slot.o2] = planar(bc.angles[1]) * bc.lengths[1]
    d[Slot.o3] = planar(bc.angles[2]) * bc.lengths[2]
    d[Slot.hKept] = d[Slot.o2] + hydrogenDirection(oxygen: d[Slot.o2], carbon: c, ref: d[Slot.o1], cis: true,
                                                   cOH: bc.cOH) * bc.hydrogenBond
    let protonCis = hydrogenDirection(oxygen: acid[Slot.o3], carbon: c, ref: acid[Slot.o1], cis: true, cOH: ac.cOH)
    let cisProton = acid[Slot.o3] + protonCis * ac.hydrogenBond
    let waiting = cisProton + protonCis * 1.75      // helper O, hydrogen-bonded to the proton
    d[Slot.helperO] = waiting + protonCis * 0.3      // backs off a little as it takes the proton
    let hydronium = hydroniumDirections(first: simd_normalize(d[Slot.o3] - d[Slot.helperO]))
    d[Slot.proton] = d[Slot.helperO] + hydronium[0] * hydroniumBond
    d[Slot.hRelayed] = d[Slot.helperO] + hydronium[1] * hydroniumBond
    d[Slot.hHelper] = d[Slot.helperO] + hydronium[2] * hydroniumBond
    let products = d

    // Cis acid: H₂CO₃ cis-cis, the helper (still water) waiting to accept.
    var ca = acid
    ca[Slot.proton] = cisProton
    ca[Slot.helperO] = waiting
    let (w1, w2) = waterDirections(bisector: hydronium[1] + hydronium[2], spread: hydronium[1] - hydronium[2])
    ca[Slot.hRelayed] = waiting + w1 * waterBond
    ca[Slot.hHelper] = waiting + w2 * waterBond
    let cisAcid = ca

    return Poses(start: start, gathered: gathered, acid: acid, cisAcid: cisAcid, products: products,
                 protonTrans: protonTrans, protonCis: protonCis,
                 helperControl: SIMD3(-4.6, -3.8, 0))
}

// MARK: - The timeline

/// One ~13.5 s cycle. The last frame flows straight into the first, so the GIF
/// loops as an endless stream of CO₂ turning into bicarbonate.
struct Timeline {
    let act1: Double = 2.5      // drift together
    let act2: Double = 4.0      // hydration with the relayed proton
    let act3: Double = 1.5      // carbonic acid; OH turns cis; helper moves round
    let act4: Double = 3.0      // proton to the helper: HCO₃⁻ + H₃O⁺
    let act5: Double = 2.5      // products fall away
    var total: Double { act1 + act2 + act3 + act4 + act5 }

    var act2Start: Double { act1 }
    var act3Start: Double { act1 + act2 }
    var act4Start: Double { act1 + act2 + act3 }
    var act5Start: Double { act1 + act2 + act3 + act4 }

    func fraction(_ t: Double, from start: Double, length: Double) -> Float {
        Float(min(max((t - start) / length, 0), 1))
    }

    func stage(at t: Double) -> String {
        if t < act2Start { return "CO₂ meets water" }
        if t < act3Start { return "CO₂ + H₂O → H₂CO₃  (a helper water relays the proton)" }
        if t < act4Start { return "Carbonic acid, H₂CO₃" }
        if t < act5Start { return "H₂CO₃ + H₂O → HCO₃⁻ + H₃O⁺" }
        return "Bicarbonate and hydronium drift off"
    }
}

/// How far products have fallen away (0 → 1) during act 5, speeding up.
func exitAmount(_ timeline: Timeline, _ t: Double) -> Float {
    let x = timeline.fraction(t, from: timeline.act5Start, length: timeline.act5 - 0.5)
    return x * x
}

/// How far the next group has floated in (0 → 1) during act 5, slowing down.
func entryAmount(_ timeline: Timeline, _ t: Double) -> Float {
    let x = timeline.fraction(t, from: timeline.act5Start + 0.3, length: timeline.act5 - 0.3)
    return 1 - (1 - x) * (1 - x)
}

let exitDown: Float = 13        // bicarbonate sinks this far (Å)
let exitLeft: Float = 15        // H₃O⁺ drifts this far left (Å)
let entryHeight: Float = 12     // the next group starts this far above its start pose (Å)

/// The group's nine atom positions at time `t` (before any exit or entry offsets).
func groupPositions(_ poses: Poses, _ timeline: Timeline, _ t: Double) -> [SIMD3<Float>] {
    if t < timeline.act2Start {
        let f = smoothstep01(timeline.fraction(t, from: 0, length: timeline.act1))
        return (0..<Slot.count).map { mix3(poses.start[$0], poses.gathered[$0], f) }
    }
    if t < timeline.act3Start {
        let f = smoothstep01(timeline.fraction(t, from: timeline.act2Start, length: timeline.act2))
        return (0..<Slot.count).map { mix3(poses.gathered[$0], poses.acid[$0], f) }
    }
    if t < timeline.act4Start {
        let x = timeline.fraction(t, from: timeline.act3Start, length: timeline.act3)
        // The OH turns first; the helper sets off a moment later so they never collide.
        let turnF = smoothstep01(x / 0.6)
        let f = smoothstep01((x - 0.25) / 0.75)
        var p = poses.acid
        // The OH turns from trans to cis by swinging away from carbon, not through it.
        let o3 = p[Slot.o3]
        let turn = protonTurnAngle(from: poses.protonTrans, to: poses.protonCis, awayFrom: -o3)
        p[Slot.proton] = o3 + rotateInPlane(poses.protonTrans, by: turn * turnF) * carbonicAcidShape.hydrogenBond
        // The helper water travels around O3 on a curve, turning as it goes.
        let from = poses.acid[Slot.helperO], to = poses.cisAcid[Slot.helperO]
        let helper = bezier(from, poses.helperControl, to, f)
        p[Slot.helperO] = helper
        for h in [Slot.hRelayed, Slot.hHelper] {
            let d0 = simd_normalize(poses.acid[h] - from)
            let d1 = simd_normalize(poses.cisAcid[h] - to)
            p[h] = helper + blendDirection(d0, d1, f) * waterBond
        }
        return p
    }
    if t < timeline.act5Start {
        let f = smoothstep01(timeline.fraction(t, from: timeline.act4Start, length: timeline.act4))
        return (0..<Slot.count).map { mix3(poses.cisAcid[$0], poses.products[$0], f) }
    }
    return poses.products
}

/// The signed angle (degrees, counterclockwise) to turn `from` into `to`,
/// choosing the way round that doesn't pass through direction `awayFrom`.
func protonTurnAngle(from: SIMD3<Float>, to: SIMD3<Float>, awayFrom avoid: SIMD3<Float>) -> Float {
    func heading(_ v: SIMD3<Float>) -> Float { atan2(v.y, v.x) * 180 / .pi }
    var delta: Float = heading(to) - heading(from)
    while delta > 180 { delta -= 360 }
    while delta <= -180 { delta += 360 }
    // Does the short way pass the direction to avoid?
    var toAvoid: Float = heading(avoid) - heading(from)
    while toAvoid > 180 { toAvoid -= 360 }
    while toAvoid <= -180 { toAvoid += 360 }
    let passes = delta > 0 ? (toAvoid > 0 && toAvoid < delta) : (toAvoid < 0 && toAvoid > delta)
    if passes { delta += delta > 0 ? -360 : 360 }
    return delta
}

/// Everything on screen at time `t` in the cycle: the current group, and in
/// act 5 also the next group floating in. Atom and bond indices are grouped
/// nine by nine.
func journeyState(_ poses: Poses, _ timeline: Timeline, at time: Double) -> MoleculeState {
    let t = time.truncatingRemainder(dividingBy: timeline.total)
    let s = timeline.fraction(t, from: timeline.act2Start, length: timeline.act2)
    let u = timeline.fraction(t, from: timeline.act4Start, length: timeline.act4)
    let hydration = smoothstep01(s)
    let transfer = smoothstep01(u)

    var positions = groupPositions(poses, timeline, t)
    if t >= timeline.act5Start {
        let e = exitAmount(timeline, t)
        for i in [Slot.carbon, Slot.o1, Slot.o2, Slot.o3, Slot.hKept] { positions[i].y -= exitDown * e }
        for i in [Slot.helperO, Slot.hRelayed, Slot.proton, Slot.hHelper] {
            positions[i].x -= exitLeft * e
            positions[i].y -= 1.0 * e
        }
    }
    var atoms: [Atom] = []
    var bonds: [Bond] = []
    let q = charges(transfer: transfer)
    for i in 0..<Slot.count {
        atoms.append(Atom(element: groupElements[i], position: positions[i], charge: q[i], glow: q[i]))
    }
    let orders = bondOrders(hydration: hydration, transfer: transfer)
    for (k, pair) in groupBondPairs.enumerated() {
        bonds.append(Bond(a: pair.0, b: pair.1, order: orders[k]))
    }

    if t >= timeline.act5Start {
        // The next CO₂, water and helper float down into their start pose.
        let g = entryAmount(timeline, t)
        let offset = SIMD3<Float>(0, entryHeight * (1 - g), 0)
        let base = atoms.count
        let q0 = charges(transfer: 0)
        for i in 0..<Slot.count {
            atoms.append(Atom(element: groupElements[i], position: poses.start[i] + offset, charge: q0[i], glow: 0))
        }
        let orders0 = bondOrders(hydration: 0, transfer: 0)
        for (k, pair) in groupBondPairs.enumerated() {
            bonds.append(Bond(a: base + pair.0, b: base + pair.1, order: orders0[k]))
        }
    }
    return MoleculeState(atoms: atoms, bonds: bonds)
}

// MARK: - Measuring

func distance(_ state: MoleculeState, _ i: Int, _ j: Int) -> Float {
    simd_distance(state.atoms[i].position, state.atoms[j].position)
}

/// Angle at atom `j` between atoms `i` and `k`, in degrees.
func angle(_ state: MoleculeState, _ i: Int, _ j: Int, _ k: Int) -> Float {
    let a = simd_normalize(state.atoms[i].position - state.atoms[j].position)
    let b = simd_normalize(state.atoms[k].position - state.atoms[j].position)
    return acos(min(max(simd_dot(a, b), -1), 1)) * 180 / .pi
}

/// Sum of the bond orders touching atom `i`.
func valence(_ state: MoleculeState, of i: Int) -> Float {
    state.bonds.filter { $0.a == i || $0.b == i }.reduce(0) { $0 + $1.order }
}

/// A state made from just a set of positions (with the group's bonds), for tests.
func poseState(_ positions: [SIMD3<Float>], hydration: Float, transfer: Float) -> MoleculeState {
    let q = charges(transfer: transfer)
    let atoms = (0..<Slot.count).map { Atom(element: groupElements[$0], position: positions[$0], charge: q[$0]) }
    let orders = bondOrders(hydration: hydration, transfer: transfer)
    let bonds = groupBondPairs.enumerated().map { Bond(a: $0.element.0, b: $0.element.1, order: orders[$0.offset]) }
    return MoleculeState(atoms: atoms, bonds: bonds)
}
