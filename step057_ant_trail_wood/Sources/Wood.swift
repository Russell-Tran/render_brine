// The board the ants walk on: a planed, quarter-sawn Scots pine board, as
// numbers. Nothing here touches the GPU; Render.swift writes the same
// functions into the kernel, and a test checks the two agree.
//
// World: MILLIMETRES, y up. The board's planed face is the plane y = 0 before
// its roughness is added; the grain (the board's length, the tracheids) runs
// along z, towards and away from the camera, so the annual rings show as
// stripes across the ants' path (x) and every ant walks over them.
//
// Every number carries where it came from, or says MODEL and why.

import Foundation
import simd

// MARK: - the wood, from the sources

// The species and the rings. Pinkowski, Krauss, Piernik & Szymański (2016,
// *BioResources* 11(2): 5181–5189, "Effect of thermal treatment on the surface
// roughness of Scots pine (Pinus sylvestris L.) wood after plane milling")
// measured a planed Scots pine board: "Scots pine (Pinus sylvestris L.)
// sapwood ... with annual increment widths of 2.3 mm and 31.6% latewood",
// and "the radial surfaces ... were subjected to longitudinal milling". So
// this board is theirs: sapwood, the RADIAL face (quarter-sawn, the rings
// cut square, so the stripes are one ring width apart), planed along the grain.
let woodSpecies: String = "Scots pine"
let ringWidth: Float = 2.3              // mm, mean annual increment
let latewoodFraction: Float = 0.316     // of each ring

// The roughness the planing leaves, from the same paper: measured "separately
// for earlywood and latewood", with a stylus "parallel to the feed direction"
// (along the grain), sampling length 12.5 mm, cut-off 2.5 mm. Unmodified
// wood (their "Control", Table 6): Ra 7.16 µm on earlywood, 3.76 µm on
// latewood — "latewood is characterised by a greater hardness and a more
// compact structure". Their Table 5 gives Rz and Rt only pooled over the
// control and the two heat treatments (earlywood Ra 6.39, Rz 19.03, Rt 25.78;
// latewood Ra 3.22, Rz 10.80, Rt 16.13 µm), so the SHAPE of the profile is
// taken from those pooled ratios — Rz/Ra 2.98 and 3.35, Rt/Ra 4.03 and 5.01 —
// and scaled by the control's Ra. Flagged: the ratios are pooled.
let raEarlywood: Float = 7.16e-3        // mm
let raLatewood: Float = 3.76e-3         // mm
let rzOverRaEarly: Float = 19.03 / 6.39
let rzOverRaLate: Float = 10.80 / 3.22
let rtOverRaEarly: Float = 25.78 / 6.39
let rtOverRaLate: Float = 16.13 / 3.22

// No measured height STEP between earlywood and latewood after planing was
// found (only their roughness), so the two zones sit level and differ in
// roughness, colour and nothing else. MODEL, by omission.

// The transition. "Other woods undergo an abrupt transition from earlywood
// to latewood, such as southern yellow pine (Pinus) ..." (Wiedenhoeft 2010,
// Wood Handbook FPL-GTR-190, ch. 3 "Structure and Function of Wood"); for
// Scots pine, "earlywood to latewood transition fairly abrupt, color contrast
// medium" (The Wood Database, "Scots Pine"). The widths of the two blends are
// MODEL: earlywood to latewood over 6% of a ring (0.14 mm), and the ring
// boundary — latewood to next year's earlywood, always sharp — over 1% (23 µm,
// about one pixel).
let earlyToLateBlend: Float = 0.06
let ringBoundaryBlend: Float = 0.01

// Rays are NOT drawn. Softwood rays are "only one cell in width, called a
// uniseriate ray", of ray cells "approximately 15 µm high by 10 µm wide by
// 150 to 250 µm long" (Wiedenhoeft 2010). On this radial face a ray is a
// ribbon across the grain whose height along the grain is a stack of those
// 15 µm cells; no measured ray height or colour contrast for Scots pine was
// found, and one cell is under a pixel here (21 µm), so there is no honest
// way to say they would show. Left out, and said so in the code, not drawn.
let rayCellHeightMicrometres: Float = 15

// MARK: - MODEL choices for the drawn board

/// Year-to-year variation of ring width, ± fraction. MODEL: real rings vary
/// (Fabisiak & Fabisiak 2021, *BioResources* 16(4): 7492, measured Scots pine
/// rings from 1.00 to 4.53 mm in one class of tree); ±12% keeps the mean at
/// 2.3 mm across the frame (a test checks it) while no two stripes match.
let ringWidthJitter: Float = 0.12
/// Grain run-out: the rings are not quite square to the ants' path. MODEL:
/// 2°, and a gentle waviness of 0.05 mm over 9 mm along the grain.
let grainRunOut: Float = 2 * Float.pi / 180
let grainWaveAmplitude: Float = 0.05
let grainWaveLength: Float = 9.0
/// Correlation lengths of the planed roughness: across the grain about one
/// tracheid (MODEL: 0.05 mm cells), along it much longer (MODEL: 0.7 mm cells —
/// cut tracheids leave grooves that run with the grain).
let roughCellAcross: Float = 0.05
let roughCellAlong: Float = 0.7
/// How peaked each zone's profile is: tanh(k·n) of a value noise n, with k
/// chosen so the drawn Rz/Ra and Rt/Ra land on the pooled ratios above (the
/// tests measure them). MODEL shaping, cited targets.
let roughShapeEarly: Float = 2.8
let roughShapeLate: Float = 0.8

/// The first ring boundary to the left of everything the camera or the
/// ants ever reach, and how many rings are laid out from it. MODEL.
let ringOrigin: Float = -20.0
let ringCount: Int = 18

// MARK: - the noise, written so Metal computes the same numbers

/// A 32-bit integer hash of a lattice point, to a value in [0, 1). The same
/// wrapping multiplications and shifts are in the kernel.
@inline(__always) func latticeValue(_ ix: Int32, _ iz: Int32) -> Float {
    let hx: UInt32 = UInt32(bitPattern: ix) &* 0x8da6_b343
    let hz: UInt32 = UInt32(bitPattern: iz) &* 0xd816_3841
    var h: UInt32 = hx ^ hz
    h ^= h >> 13
    h = h &* 0x5bd1_e995
    h ^= h >> 15
    return Float(h & 0x00ff_ffff) / 16_777_216
}

/// Smooth value noise, zero-centred, in (−0.5, 0.5).
@inline(__always) func valueNoise(_ u: Float, _ v: Float) -> Float {
    let fu: Float = floor(u)
    let fv: Float = floor(v)
    let a: Float = u - fu
    let b: Float = v - fv
    let ea: Float = 3 - 2 * a
    let eb: Float = 3 - 2 * b
    let su: Float = a * a * ea
    let sv: Float = b * b * eb
    let ix = Int32(fu)
    let iz = Int32(fv)
    let v00: Float = latticeValue(ix, iz)
    let v10: Float = latticeValue(ix &+ 1, iz)
    let v01: Float = latticeValue(ix, iz &+ 1)
    let v11: Float = latticeValue(ix &+ 1, iz &+ 1)
    let lo: Float = v00 + (v10 - v00) * su
    let hi: Float = v01 + (v11 - v01) * su
    return lo + (hi - lo) * sv - 0.5
}

/// The raw roughness field: two octaves, stretched along the grain.
@inline(__always) func roughNoise(_ x: Float, _ z: Float) -> Float {
    let n1: Float = valueNoise(x / roughCellAcross, z / roughCellAlong)
    let u2: Float = x / (roughCellAcross * 0.5) + 31.7
    let v2: Float = z / (roughCellAlong * 0.5) + 11.3
    return n1 + 0.5 * valueNoise(u2, v2)
}

/// The Ra of tanh(k·n), measured the way the paper measured the wood: along
/// the grain (z), 12.5 mm profiles, each about its own mean line — so the
/// shaped noise, scaled by the cited Ra over this, has exactly that Ra.
/// Measured here, once, on fixed profiles; the kernel is handed the numbers.
func profileRaShaped(_ k: Float) -> Float {
    var sum: Double = 0
    var profiles: Int = 0
    for i in 0..<48 {
        let fi: Float = Float(i)
        let x: Float = fi * 0.4173 - 10.0
        var values: [Float] = []
        values.reserveCapacity(6250)
        for j in 0..<6250 {
            let fj: Float = Float(j)
            let z: Float = fj * 0.002 - 6.25
            values.append(tanh(k * roughNoise(x, z)))
        }
        let mean: Float = values.reduce(0, +) / Float(values.count)
        var dev: Float = 0
        for v in values { dev += abs(v - mean) }
        sum += Double(dev / Float(values.count))
        profiles += 1
    }
    return Float(sum / Double(profiles))
}
let roughUnitEarly: Float = profileRaShaped(roughShapeEarly)
let roughUnitLate: Float = profileRaShaped(roughShapeLate)

// MARK: - the rings

/// The ring boundaries' positions across the grain (the radial coordinate,
/// pith towards −x), each ring its own width.
let ringStarts: [Float] = {
    var out: [Float] = []
    var at: Float = ringOrigin
    for k in 0..<(ringCount + 1) {
        out.append(at)
        let u: Float = latticeValue(Int32(k), 977) * 2 - 1
        let factor: Float = 1 + ringWidthJitter * u
        at += ringWidth * factor
    }
    return out
}()

/// Where a point on the face sits across the rings: x, less the run-out and
/// the waviness.
@inline(__always) func radialCoordinate(_ x: Float, _ z: Float) -> Float {
    let turns: Float = z / grainWaveLength
    let angle: Float = 2 * Float.pi * turns
    let wave: Float = grainWaveAmplitude * sin(angle)
    return x - z * tan(grainRunOut) - wave
}

@inline(__always) func smoothstep(_ a: Float, _ b: Float, _ x: Float) -> Float {
    let t: Float = min(max((x - a) / (b - a), 0), 1)
    let e: Float = 3 - 2 * t
    return t * t * e
}

/// How much latewood a point on the face is, 0 (earlywood) to 1: each ring
/// begins in earlywood, turns to latewood for its last 31.6%, and ends sharp.
func latewood(_ x: Float, _ z: Float) -> Float {
    let r: Float = radialCoordinate(x, z)
    var k: Int = 0
    while k < ringCount - 1 && r >= ringStarts[k + 1] { k += 1 }
    let w: Float = ringStarts[k + 1] - ringStarts[k]
    let s: Float = (r - ringStarts[k]) / w
    // Latewood from here to the ring's end: the sharp boundary blend takes
    // half its width back, so the onset moves out by the same half.
    let onset: Float = 1 - latewoodFraction - ringBoundaryBlend / 2
    let rise: Float = smoothstep(onset - earlyToLateBlend / 2, onset + earlyToLateBlend / 2, s)
    let fall: Float = 1 - smoothstep(1 - ringBoundaryBlend, 1, s)
    return rise * fall
}

// MARK: - the surface

/// Which way to break the wood, for the mutation check (set by the tests).
var woodMutant: Mutant = .none

/// The planed face's height above y = 0 at (x, z), mm: each zone's shaped
/// roughness at that zone's cited Ra, blended where the zones meet.
func woodHeight(_ x: Float, _ z: Float) -> Float {
    let n: Float = roughNoise(x, z)
    var lw: Float = latewood(x, z)
    if woodMutant == .swapRoughness { lw = 1 - lw }
    let scaleEarly: Float = raEarlywood / roughUnitEarly
    let scaleLate: Float = raLatewood / roughUnitLate
    let early: Float = scaleEarly * tanh(roughShapeEarly * n)
    let late: Float = scaleLate * tanh(roughShapeLate * n)
    return early + (late - early) * lw
}

/// The highest (and, negated, the lowest) the face can ever reach: the
/// noise's two octaves are each within ±0.5, so |n| < 0.75, and tanh is
/// monotone. A strict bound, not a sample.
let woodHeightMax: Float = {
    let scaleEarly: Float = raEarlywood / roughUnitEarly
    let scaleLate: Float = raLatewood / roughUnitLate
    let argEarly: Float = roughShapeEarly * 0.75
    let argLate: Float = roughShapeLate * 0.75
    let early: Float = scaleEarly * tanh(argEarly)
    let late: Float = scaleLate * tanh(argLate)
    return max(early, late)
}()

/// The face's steepest slope, measured on a fine grid over the whole area
/// the ants and camera use, with a quarter more for safety. The kernel
/// divides its height gap by √(1 + L²), so no ray is ever told the surface is
/// further away than it is (a test checks this against the drawn surface).
let woodSlopeBound: Float = {
    var worst: Float = 0
    let e: Float = 0.0005
    for i in 0..<1400 {
        let fi: Float = Float(i)
        let x: Float = fi * 0.0173 - 12.0
        for j in 0..<220 {
            let fj: Float = Float(j)
            let z: Float = fj * 0.0419 - 5.0
            let h: Float = woodHeight(x, z)
            let gx: Float = (woodHeight(x + e, z) - h) / e
            let gz: Float = (woodHeight(x, z + e) - h) / e
            let gx2: Float = gx * gx
            let gz2: Float = gz * gz
            let slope: Float = (gx2 + gz2).squareRoot()
            worst = max(worst, slope)
        }
    }
    return worst * 1.25
}()

// MARK: - resting things on it

/// Where a sphere of radius r, centred over (x, z), comes to rest on the
/// face: the height of its LOWEST point. The highest bump under its disc
/// decides it — a grid of heights under the disc, each lifted by the
/// sphere's curvature, then refined round the best. Exact to well under a
/// tenth of a micrometre (a test measures the gap and the overlap).
func sphereRest(x: Float, z: Float, radius r: Float) -> Float {
    if woodMutant == .sink { return 0 }
    func lift(_ dx: Float, _ dz: Float) -> Float {
        let dx2: Float = dx * dx
        let dz2: Float = dz * dz
        let rho2: Float = dx2 + dz2
        if rho2 >= r * r { return -1 }
        let cap: Float = (r * r - rho2).squareRoot()
        return woodHeight(x + dx, z + dz) + cap - r
    }
    var best: Float = -1
    var bx: Float = 0
    var bz: Float = 0
    let n: Int = 14
    let step: Float = r / Float(n)
    for i in -n...n {
        for j in -n...n {
            let fi: Float = Float(i)
            let fj: Float = Float(j)
            let dx: Float = fi * step
            let dz: Float = fj * step
            let v: Float = lift(dx, dz)
            if v > best { best = v; bx = dx; bz = dz }
        }
    }
    var s: Float = step
    for _ in 0..<4 {
        s /= 4
        let cx: Float = bx
        let cz: Float = bz
        for i in -4...4 {
            for j in -4...4 {
                let fi: Float = Float(i)
                let fj: Float = Float(j)
                let dx: Float = cx + fi * s
                let dz: Float = cz + fj * s
                let v: Float = lift(dx, dz)
                if v > best { best = v; bx = dx; bz = dz }
            }
        }
    }
    // A hair of clearance so rounding never puts the sphere into the wood.
    return best + 5e-5
}

/// Points on the underside of a round cone, for checking it against the
/// face: rings round the axis every `spacing` mm, lower half only.
func undersidePoints(_ s: Shape, spacing: Float) -> [(point: SIMD3<Float>, fraction: Float)] {
    let axis: SIMD3<Float> = s.b - s.a
    let len: Float = simd_length(axis)
    let u: SIMD3<Float> = axis / max(len, 1e-6)
    var side: SIMD3<Float> = simd_cross(u, SIMD3<Float>(0, 1, 0))
    if simd_length(side) < 1e-4 { side = SIMD3<Float>(1, 0, 0) }
    side = simd_normalize(side)
    let down: SIMD3<Float> = simd_normalize(simd_cross(u, side) * (simd_cross(u, side).y > 0 ? -1 : 1))
    let steps: Int = max(Int(len / spacing), 1)
    var out: [(point: SIMD3<Float>, fraction: Float)] = []
    for k in 0...steps {
        let fk: Float = Float(k)
        let fsteps: Float = Float(steps)
        let f: Float = fk / fsteps
        let c: SIMD3<Float> = s.a + axis * f
        let r: Float = s.ra + (s.rb - s.ra) * f
        for a in -4...4 {
            let fa: Float = Float(a)
            let phi: Float = fa * (Float.pi / 10)
            let across: SIMD3<Float> = down * cos(phi)
            let sideways: SIMD3<Float> = side * sin(phi)
            let dir: SIMD3<Float> = across + sideways
            out.append((c + dir * r, f))
        }
    }
    // The two round ends: their lower hemispheres, which reach past the axis.
    for (c, r, f) in [(s.a, s.ra, Float(0)), (s.b, s.rb, Float(1))] {
        for i in 0...6 {
            let fi: Float = Float(i)
            let rise: Float = Float.pi / 12 * 0.95
            let theta: Float = fi * rise
            for j in 0..<12 {
                let slice: Float = 2 * Float.pi / 12
                let fj: Float = Float(j)
                let phi: Float = fj * slice
                let st: Float = sin(theta)
                let dx: Float = st * cos(phi)
                let dz: Float = st * sin(phi)
                let d = SIMD3<Float>(dx, -cos(theta), dz)
                out.append((c + d * r, f))
            }
        }
    }
    return out
}

/// The gap between a round cone and the face near a point, mm, measured from
/// the face's side: the least exact distance from points ON the face (a grid
/// round `near`, `reach` each way, refined round the nearest) to the cone.
/// Negative when the face is inside it. For a steep, strongly tapered cone —
/// the gaster's tip at a dab — where rings round the axis miss the true
/// lowest point.
func coneGap(_ s: Shape, near: SIMD2<Float>, reach: Float) -> Float {
    func dist(_ x: Float, _ z: Float) -> Float {
        let q = SIMD3<Float>(x, woodHeight(x, z), z)
        return roundConeDistance(q, s.a, s.b, s.ra, s.rb)
    }
    var best: Float = 1e9
    var bx: Float = near.x
    var bz: Float = near.y
    let n: Int = 16
    let step: Float = reach / Float(n)
    for i in -n...n {
        for j in -n...n {
            let fi: Float = Float(i)
            let fj: Float = Float(j)
            let x: Float = near.x + fi * step
            let z: Float = near.y + fj * step
            let v: Float = dist(x, z)
            if v < best { best = v; bx = x; bz = z }
        }
    }
    var st: Float = step
    for _ in 0..<4 {
        st /= 4
        let cx: Float = bx
        let cz: Float = bz
        for i in -4...4 {
            for j in -4...4 {
                let fi: Float = Float(i)
                let fj: Float = Float(j)
                let x: Float = cx + fi * st
                let z: Float = cz + fj * st
                let v: Float = dist(x, z)
                if v < best { best = v; bx = x; bz = z }
            }
        }
    }
    return best
}
