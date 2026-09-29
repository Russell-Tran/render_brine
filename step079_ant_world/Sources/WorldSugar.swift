// Step 79: the sugar pile as the film draws it — step 26's sucrose crystals,
// placed where the simulation dropped them (SimGrain: centre, sieve size,
// turn, rest face).
//
// The crystal is step 26's (step026_ant/Sources/Anatomy.swift), copied, not
// re-derived: face normals computed from the sucrose cell, the stout-prism
// habit, the worn edges. Only the placement is new: step 26 placed five
// grains by hand at a scale; here each of the simulation's 200 grains is
// scaled so its SIEVE size (the middle of its three extents, as step 26's
// test compares) is the size the simulation drew from step 26's sieve range.

import Foundation
import simd

// Sucrose crystallises monoclinic, space group P2₁, a = 10.8631 Å,
// b = 8.7044 Å, c = 7.7624 Å, β = 102.938° (Wikipedia "Sucrose", from the
// neutron structure of Brown & Levy, *Acta Cryst B* 1973). Its normal habit is
// "simple or normal of stout-prismatic form", bounded by a(100), c(001),
// d(101), r(1̄01) and p(110) faces, with the cleavage parallel to a
// (VanHook, "Habit modification of sucrose crystals", *J Sugar Beet Res*
// 22(1), 1983, which follows Vavrinecz's *Atlas of Sugar Crystals*). All
// step 26's.
let sucroseCellA: Float = 10.8631
let sucroseCellB: Float = 8.7044
let sucroseCellC: Float = 7.7624
let sucroseCellBeta: Float = 102.938 * Float.pi / 180

/// The normal of the plane (hkl): h·a* + k·b* + l·c* (step 26).
func sucroseNormal(_ h: Float, _ k: Float, _ l: Float) -> SIMD3<Float> {
    let a = SIMD3<Float>(sucroseCellA, 0, 0)
    let b = SIMD3<Float>(0, sucroseCellB, 0)
    let cx: Float = sucroseCellC * cos(sucroseCellBeta)
    let cz: Float = sucroseCellC * sin(sucroseCellBeta)
    let c = SIMD3<Float>(cx, 0, cz)
    let v: Float = simd_dot(a, simd_cross(b, c))
    let aS: SIMD3<Float> = simd_cross(b, c) / v
    let bS: SIMD3<Float> = simd_cross(c, a) / v
    let cS: SIMD3<Float> = simd_cross(a, b) / v
    let na: SIMD3<Float> = aS * h
    let nb: SIMD3<Float> = bS * k
    let nc: SIMD3<Float> = cS * l
    let n: SIMD3<Float> = na + nb + nc
    return simd_normalize(n)
}

/// One face form and its plane's distance from the centre, mm at scale 1.
/// MODEL distances (step 26), for the stout prism VanHook calls normal.
struct SugarForm {
    var hkl: SIMD3<Float>
    var distance: Float
}

let sugarForms: [SugarForm] = [
    SugarForm(hkl: SIMD3<Float>(1, 0, 0), distance: 0.20), SugarForm(hkl: SIMD3<Float>(-1, 0, 0), distance: 0.20),
    SugarForm(hkl: SIMD3<Float>(0, 0, 1), distance: 0.24), SugarForm(hkl: SIMD3<Float>(0, 0, -1), distance: 0.24),
    SugarForm(hkl: SIMD3<Float>(1, 1, 0), distance: 0.30), SugarForm(hkl: SIMD3<Float>(1, -1, 0), distance: 0.30),
    SugarForm(hkl: SIMD3<Float>(-1, 1, 0), distance: 0.30), SugarForm(hkl: SIMD3<Float>(-1, -1, 0), distance: 0.30),
]
let sugarForms2: [SugarForm] = [
    SugarForm(hkl: SIMD3<Float>(1, 0, 1), distance: 0.27), SugarForm(hkl: SIMD3<Float>(-1, 0, -1), distance: 0.27),
    SugarForm(hkl: SIMD3<Float>(-1, 0, 1), distance: 0.29), SugarForm(hkl: SIMD3<Float>(1, 0, -1), distance: 0.29),
    SugarForm(hkl: SIMD3<Float>(0, 1, 1), distance: 0.36), SugarForm(hkl: SIMD3<Float>(0, -1, -1), distance: 0.36),
    SugarForm(hkl: SIMD3<Float>(0, 1, -1), distance: 0.36), SugarForm(hkl: SIMD3<Float>(0, -1, 1), distance: 0.36),
]
/// All sixteen planes (split in two literals for the compiler).
let sugarAllForms: [SugarForm] = sugarForms + sugarForms2
let sugarFaces: Int = 16

/// Edge rounding, mm (step 26): real grains are chipped and worn at the
/// edges by handling. MODEL.
let sugarRounding: Float = 0.012

/// Sucrose crystal: biaxial, nα 1.540, nβ 1.567, nγ 1.572 (McCrone Particle
/// Atlas, "Sucrose", via step 26). The birefringence is ignored and the mean
/// used.
private let sucroseAlpha: Float = 1.540
private let sucroseBeta: Float = 1.567
private let sucroseGamma: Float = 1.572
let sucroseIndex: Float = (sucroseAlpha + sucroseBeta + sucroseGamma) / 3
private let sucroseR: Float = (sucroseIndex - 1) / (sucroseIndex + 1)
let sucroseF0: Float = sucroseR * sucroseR

/// Polynomial smooth maximum, as the kernel's smax (step 26).
func sugarSmoothMax(_ a: Float, _ b: Float, _ k: Float) -> Float {
    let h: Float = min(max(0.5 + 0.5 * (a - b) / k, 0), 1)
    let lift: Float = k * h * (1 - h)
    return b + (a - b) * h + lift
}

/// One grain in the world: its faces as world-space planes.
struct SugarGrain {
    var centre: SIMD3<Float>
    var normals: [SIMD3<Float>]
    var distances: [Float]
    /// The crystal's own a*, b and a* × b directions (for the sieve size).
    var frame: [SIMD3<Float>]

    /// Step 26's grain distance: the largest signed plane distance, edges
    /// worn by a smooth maximum. Exact inside, a lower bound outside.
    func sdf(_ p: SIMD3<Float>) -> Float {
        let q: SIMD3<Float> = p - centre
        var d: Float = simd_dot(q, normals[0]) - distances[0]
        for i in 1..<normals.count {
            let e: Float = simd_dot(q, normals[i]) - distances[i]
            d = sugarSmoothMax(d, e, sugarRounding)
        }
        return d
    }

    /// Corners, by intersecting every three planes (step 26).
    func vertices() -> [SIMD3<Float>] {
        var out: [SIMD3<Float>] = []
        let n: Int = normals.count
        for i in 0..<n {
            for j in (i + 1)..<n {
                for k in (j + 1)..<n {
                    let m = simd_float3x3(rows: [normals[i], normals[j], normals[k]])
                    if abs(m.determinant) < 1e-4 { continue }
                    let rhs = SIMD3<Float>(distances[i], distances[j], distances[k])
                    let q: SIMD3<Float> = m.inverse * rhs
                    var inside: Bool = true
                    for f in 0..<n where simd_dot(q, normals[f]) > distances[f] + 1e-4 { inside = false; break }
                    if inside { out.append(q + centre) }
                }
            }
        }
        return out
    }

    /// Extents along the crystal's own axes; the sieve passes the middle one.
    func extents() -> [Float] {
        let v: [SIMD3<Float>] = vertices()
        return frame.map { d in
            let proj: [Float] = v.map { simd_dot($0, d) }
            let hi: Float = proj.max() ?? 0
            let lo: Float = proj.min() ?? 0
            return hi - lo
        }
    }

    var sieveSize: Float { extents().sorted()[1] }

    /// A sphere round every corner.
    var boundRadius: Float { vertices().map { simd_distance($0, centre) }.max() ?? 0 }
}

/// Step 26's rest faces by the simulation's code 0–3.
func sugarRestFace(_ code: UInt8) -> SIMD3<Float> {
    switch code {
    case 0: return SIMD3<Float>(-1, 0, 0)
    case 1: return SIMD3<Float>(0, 0, -1)
    case 2: return SIMD3<Float>(1, 0, 0)
    default: return SIMD3<Float>(0, 0, 1)
    }
}

/// Step 26's placement: rotate so the rest face points straight down, turn
/// about the vertical, lift until that face lies on the ground.
func sugarPlace(restFace: SIMD3<Float>, yaw: Float, scale: Float, at: SIMD2<Float>) -> SugarGrain {
    let restN: SIMD3<Float> = sucroseNormal(restFace.x, restFace.y, restFace.z)
    let q1 = simd_quatf(from: restN, to: SIMD3<Float>(0, -1, 0))
    let q2 = simd_quatf(angle: yaw, axis: SIMD3<Float>(0, 1, 0))
    let q: simd_quatf = q2 * q1
    var normals: [SIMD3<Float>] = []
    var dists: [Float] = []
    var restDistance: Float = 0
    for f in sugarAllForms {
        normals.append(q.act(sucroseNormal(f.hkl.x, f.hkl.y, f.hkl.z)))
        dists.append(f.distance * scale)
        if simd_length(f.hkl - restFace) < 1e-3 { restDistance = f.distance * scale }
    }
    let centre = SIMD3<Float>(at.x, restDistance, at.y)
    let aStar: SIMD3<Float> = q.act(sucroseNormal(1, 0, 0))
    let bAxis: SIMD3<Float> = q.act(SIMD3<Float>(0, 1, 0))
    let third: SIMD3<Float> = simd_normalize(simd_cross(aStar, bAxis))
    return SugarGrain(centre: centre, normals: normals, distances: dists, frame: [aStar, bAxis, third])
}

/// The sieve size of a scale-1 grain (the same for every rest face and turn).
let sugarUnitSieve: Float = sugarPlace(restFace: SIMD3<Float>(1, 0, 0), yaw: 0, scale: 1,
                                       at: SIMD2<Float>(0, 0)).sieveSize

/// The simulation's grain as step 26's crystal, scaled to its sieve size.
func sugarGrain(_ g: SimGrain) -> SugarGrain {
    let scale: Float = g.size / sugarUnitSieve
    return sugarPlace(restFace: sugarRestFace(g.restFace), yaw: g.yaw, scale: scale, at: SIMD2<Float>(g.x, g.z))
}

/// Grains flattened for the GPU: per grain its centre and bounding radius,
/// then its 16 planes (normal, distance).
func sugarGPU(_ grains: [SugarGrain]) -> [SIMD4<Float>] {
    var out: [SIMD4<Float>] = []
    for g in grains {
        out.append(SIMD4<Float>(g.centre, g.boundRadius))
        for i in 0..<sugarFaces { out.append(SIMD4<Float>(g.normals[i], g.distances[i])) }
    }
    return out
}
