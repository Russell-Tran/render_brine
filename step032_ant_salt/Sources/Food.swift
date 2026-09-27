// The food: a few grains of table salt, in millimetres, and the optics that
// make them look like salt rather than ice.

import Foundation
import simd

// MARK: - halite

// Table salt is sodium chloride, the mineral halite: cubic, space group
// Fm3̄m, a = 5.6402 Å (Wikipedia, "Ionic radius", quoting the NaCl cell edge
// 564.02 pm; Mindat and the Wikipedia "Halite" page give 5.6404 Å). It
// cleaves perfectly on {100}, so its grains are cubes: every face is square
// to its neighbours, and nothing needs computing from a cell as sucrose's did.
let haliteCell: Float = 5.6402   // Å

/// Vacuum-evaporated table salt: "the size of crystals is in the size range
/// of 200–500 µm" (patent WO 2009/087645 A1, CSMCRI, describing the cubic
/// table salt it sets out to improve). A grain passes a sieve by its middle
/// dimension — for a cube, simply its edge.
let saltSieveRange: ClosedRange<Float> = 0.20...0.50

/// Edge rounding, mm. MODEL: table-salt cubes are not knife-edged; handling
/// and humid air round and chip them.
let grainRounding: Float = 0.010

/// Faces per grain on the GPU: the six cube faces, then two corner chips.
let facesPerGrain: Int = 8

/// One grain in the world: its faces as world-space planes. Planes 0…5 are
/// the cube's +x, −x, +y, −y, +z, −z faces, in that order — the kernel reads
/// the cube's own axes from planes 0, 2 and 4 to draw the growth steps.
struct Grain {
    var centre: SIMD3<Float>
    var normals: [SIMD3<Float>]
    var distances: [Float]

    /// The largest signed plane distance — never more than the true distance
    /// outside a convex solid, exact inside — with the edges worn by a smooth
    /// maximum of radius `grainRounding`. The same arithmetic as the kernel's.
    func sdf(_ p: SIMD3<Float>) -> Float {
        let q: SIMD3<Float> = p - centre
        var d: Float = simd_dot(q, normals[0]) - distances[0]
        for i in 1..<normals.count {
            d = smoothMax(d, simd_dot(q, normals[i]) - distances[i], grainRounding)
        }
        return d
    }

    /// Corners of the polyhedron, by intersecting every three planes.
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
                    var inside = true
                    for f in 0..<n where simd_dot(q, normals[f]) > distances[f] + 1e-4 { inside = false; break }
                    if inside { out.append(q + centre) }
                }
            }
        }
        return out
    }

    /// Extents along the cube's own three axes.
    func extents() -> [Float] {
        let v: [SIMD3<Float>] = vertices()
        return [normals[0], normals[2], normals[4]].map { d in
            let proj: [Float] = v.map { simd_dot($0, d) }
            return (proj.max() ?? 0) - (proj.min() ?? 0)
        }
    }
}

/// How a grain lies: its edge, its turn about the vertical, where it sits,
/// and which corners are chipped (as ±1 signs along the cube axes, with the
/// depth of the chip in mm). Salt grains lie on a cube face — they cleave and
/// grow that way — so the only freedom is the turn. MODEL placements.
struct GrainPlacement {
    var edge: Float
    var yaw: Float
    var at: SIMD2<Float>
    var chips: [(corner: SIMD3<Float>, depth: Float)]
}

let grainPlacements: [GrainPlacement] = [
    GrainPlacement(edge: 0.38, yaw: 0.35, at: SIMD2(2.62, 1.05), chips: [(SIMD3(1, 1, -1), 0.05)]),
    GrainPlacement(edge: 0.33, yaw: 0.95, at: SIMD2(3.40, 0.52), chips: [(SIMD3(-1, 1, 1), 0.04), (SIMD3(1, -1, 1), 0.03)]),
    GrainPlacement(edge: 0.42, yaw: 0.15, at: SIMD2(3.22, 1.72), chips: [(SIMD3(1, 1, 1), 0.07)]),
    GrainPlacement(edge: 0.30, yaw: -0.50, at: SIMD2(2.30, 2.05), chips: []),
    GrainPlacement(edge: 0.36, yaw: 0.62, at: SIMD2(3.95, 1.32), chips: [(SIMD3(-1, 1, -1), 0.04)]),
]

/// Polynomial smooth maximum, as the kernel's smax.
func smoothMax(_ a: Float, _ b: Float, _ k: Float) -> Float {
    let h: Float = min(max(0.5 + 0.5 * (a - b) / k, 0), 1)
    return b + (a - b) * h + k * h * (1 - h)
}

/// A grain in the world: a cube of the given edge resting on its −y face,
/// turned about the vertical; each chip is a plane cutting off a corner.
/// Unused chip slots are parked far outside, where they never bind.
func placeGrain(_ g: GrainPlacement) -> Grain {
    let q = simd_quatf(angle: g.yaw, axis: SIMD3<Float>(0, 1, 0))
    let axes: [SIMD3<Float>] = [SIMD3(1, 0, 0), SIMD3(-1, 0, 0), SIMD3(0, 1, 0), SIMD3(0, -1, 0), SIMD3(0, 0, 1), SIMD3(0, 0, -1)]
    let half: Float = g.edge / 2
    var normals: [SIMD3<Float>] = axes.map { q.act($0) }
    var dists: [Float] = Array(repeating: half, count: 6)
    for k in 0..<(facesPerGrain - 6) {
        if k < g.chips.count {
            let c = g.chips[k]
            normals.append(q.act(simd_normalize(c.corner)))
            dists.append(half * Float(3).squareRoot() - c.depth)
        } else {
            normals.append(SIMD3<Float>(0, 1, 0))
            dists.append(10)
        }
    }
    return Grain(centre: SIMD3<Float>(g.at.x, half, g.at.y), normals: normals, distances: dists)
}

func buildGrains() -> [Grain] { grainPlacements.map { placeGrain($0) } }

/// Where the right antenna's tip touches the tasted grain: a point on its top
/// face, chosen towards the ant. The tip's end cap rests on the face there.
func contactPoint() -> (point: SIMD3<Float>, normal: SIMD3<Float>) {
    let g: Grain = buildGrains()[0]
    var best: Int = 0
    for i in 0..<g.normals.count where g.normals[i].y > g.normals[best].y && g.distances[i] < 5 { best = i }
    let n: SIMD3<Float> = g.normals[best]
    let top: SIMD3<Float> = g.centre + n * g.distances[best]
    return (top + SIMD3<Float>(-0.08, 0, -0.05), n)
}

// MARK: - optics, from refractive indices

/// Normal-incidence Fresnel reflectance between two media.
func fresnelF0(_ n1: Double, _ n2: Double) -> Float {
    let r: Double = (n1 - n2) / (n1 + n2)
    return Float(r * r)
}

/// Insect cuticle, n ≈ 1.56: Leertouwer, Wilts & Stavenga, *Opt Express* 19:
/// 24061 (2011), measured butterfly-scale chitin at 1.56 in the visible and
/// that value is the one used across insect optics. No measurement of ant
/// cuticle itself was found; the ant is taken to be chitin like the rest.
let cuticleIndex: Double = 1.56
/// Halite, n = 1.5443 (Mindat, via Wikidata "halite"; Wikipedia "Halite").
/// Cubic, so isotropic: no birefringence to ignore, unlike sucrose.
let haliteIndex: Double = 1.5443
/// Water, for the film.
let waterIndex: Double = 1.333

let cuticleF0: Float = fresnelF0(1.0, cuticleIndex)
let haliteF0: Float = fresnelF0(1.0, haliteIndex)
let waterF0: Float = fresnelF0(1.0, waterIndex)

// What keeps salt from looking like ice, as the shader draws it. All MODEL,
// qualitative, and said so:
//   * edges worn round (grainRounding above) and corners chipped;
//   * shallow square growth terraces on the faces — cubic salt grows layer
//     by layer from the edges in, and hopper growth leaves stepped, slightly
//     sunken faces;
//   * a faint frosting of the faces and a milky haze inside, standing in for
//     the tiny brine inclusions and surface damage that make a heap of table
//     salt white rather than clear.
let saltFrost: Float = 0.22
let saltHaze: Float = 1.8     // per mm of path: fraction scattered ≈ 1 − e^(−haze · path)

// MARK: - why it is taste, not smell

// Sodium chloride has a vapour pressure of 1 mm Hg only at 865 °C and boils
// at 1461 °C (Fisher Scientific MSDS "Sodium chloride"). At room temperature
// no salt reaches an ant through the air: there is nothing for a multiporous
// smell hair to catch. Ants do want salt — sodium is scarce inland, and
// inland ant communities take NaCl baits more readily (Kaspari, Yanoviak &
// Dudley, *PNAS* 105: 17848, 2008, 17 New World ant communities 4–2757 km
// from the sea) — and they find it the way they find sugar: by touch.
let foodHasVapour: Bool = false
