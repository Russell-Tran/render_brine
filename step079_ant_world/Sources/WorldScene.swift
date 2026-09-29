// Step 79: what the film shows, as numbers — the camera, the ground, the nest
// entrance, how the pheromone is drawn, and the ants posed from the
// simulation's record with the shared ant (lib/ant/v1).
//
// Units: MILLIMETRES, y up, the ground at y = 0; a ground point is (x, z) as
// in the simulation (SimWorldConfig: the arena is 240 × 135 mm).

import Foundation
import simd

// MARK: - the camera

/// An ORTHOGRAPHIC camera, as in steps 26 and 44–57: a millimetre is the same
/// number of pixels anywhere at one depth, and a macro lens is close to it.
/// It looks at `target` on the ground from `azimuth` (radians, the direction
/// it looks along, projected on the ground: 0 = looking towards +x) and
/// `elevation` (radians above the horizon), taking in `width` mm across.
struct WorldCamera: Equatable {
    var target: SIMD3<Float>
    var azimuth: Float
    var elevation: Float
    var width: Float

    var forward: SIMD3<Float> {
        let ce: Float = cos(elevation)
        let fx: Float = ce * cos(azimuth)
        let fz: Float = ce * sin(azimuth)
        return SIMD3<Float>(fx, -sin(elevation), fz)
    }
    var right: SIMD3<Float> { simd_normalize(simd_cross(forward, SIMD3<Float>(0, 1, 0))) }
    var up: SIMD3<Float> { simd_cross(right, forward) }
    var halfWidth: Float { width / 2 }

    /// World point to pixel (x right, y down) in a frame `w` × `h`.
    func project(_ p: SIMD3<Float>, width w: Int, height h: Int) -> SIMD2<Float> {
        let q: SIMD3<Float> = p - target
        let half: Float = Float(w) / 2
        let sx: Float = simd_dot(q, right) / halfWidth
        let sy: Float = simd_dot(q, up) / halfWidth
        let px: Float = half + sx * half
        let py: Float = Float(h) / 2 - sy * half
        return SIMD2<Float>(px, py)
    }

    /// The ground point seen at pixel (px, py) of a `w` × `h` frame.
    func groundPoint(pixel: SIMD2<Float>, width w: Int, height h: Int) -> SIMD2<Float> {
        let half: Float = Float(w) / 2
        let sx: Float = (pixel.x - half) / half
        let sy: Float = (Float(h) / 2 - pixel.y) / half
        let across: SIMD3<Float> = right * (sx * halfWidth)
        let upward: SIMD3<Float> = up * (sy * halfWidth)
        let o: SIMD3<Float> = target + across + upward
        let f: SIMD3<Float> = forward
        let t: Float = -o.y / f.y
        return SIMD2<Float>(o.x + f.x * t, o.z + f.z * t)
    }

    /// The wide shot: the whole arena, nest on the left, sugar on the right.
    /// MODEL framing: 250 mm across, looking towards −z from 60° up (so +x is to the right).
    static let wide = WorldCamera(target: SIMD3<Float>(120, 0, 67.5), azimuth: lookingBack,
                                  elevation: degrees(60), width: 250)
    /// The close shot: 50 mm across, at the trail's sugar end, 40° up. MODEL.
    static let close = WorldCamera(target: SIMD3<Float>(172, 0, 60), azimuth: lookingBack,
                                   elevation: degrees(40), width: 50)

    /// Looking towards −z, so +x is to the right of the frame.
    static let lookingBack: Float = -Float.pi / 2
    static func degrees(_ d: Float) -> Float { d * Float.pi / 180 }
}

// MARK: - the ground and the nest

/// The ground: a plain pale card, step 26/44's table (albedo 0.80, MODEL).
/// PLAIN on purpose — no fibre, no grain: part 2 (P1) found that at the
/// film's 2.2 Mbit/s fine texture goes soft, so the film's ground has none.
let worldCardAlbedo: Float = 0.80

/// The nest entrance: a round hole in the card at the simulation's nest, as
/// wide as its `nestRadius`, dropping into dark soil below. MODEL: a lab
/// arena's entrance, the depth only needs to be deeper than light reaches.
let worldNestDepth: Float = 6.0
/// Soil in the hole, linear albedo. MODEL: a dark loam.
let worldSoilAlbedo = SIMD3<Float>(0.060, 0.045, 0.034)
/// Excavated soil round the entrance: Lasius niger nests in soil and carries
/// the diggings out of the entrance; here a flat scatter of crumbs on the
/// card, densest at the rim and gone `worldNestRingWidth` mm out. MODEL (no
/// measured crater size for a lab colony's entrance; drawn flat, so it adds
/// no geometry the ants would have to walk over). Crumbs about
/// `worldCrumbSize` mm, coarse enough to survive the film's bit rate.
let worldNestRingWidth: Float = 4.0
let worldCrumbSize: Float = 0.35

// MARK: - the pheromone, drawn

/// Step 44's convention: the pheromone is DRAWN, though it is invisible in
/// life, as a soft blue on the card. Step 44's colour, linear sRGB.
let worldTrailColour = SIMD3<Float>(0.30, 0.52, 0.95)
/// How strongly the densest pheromone tints the card (the rest is card).
/// MODEL, step 44's 0.85 cap made gentler so the ants stay the subject.
let worldTrailMax: Float = 0.70
/// The tint rises as 1 − e^(−c / scale), c in marks/mm². MODEL: the scale is
/// five times the simulation's detection threshold (SimConst.detectThreshold,
/// 0.02 marks/mm²), so a just-detectable trace shows faintly (18% of full)
/// and the formed trail (median 0.23 marks/mm² on it, part 3) nearly full.
let worldTrailScale: Float = 5 * SimConst.detectThreshold

/// How far the card is tinted towards the trail colour at `c` marks/mm²
/// (the kernel's formula).
func worldTrailTint(_ c: Float) -> Float {
    let e: Float = exp(-max(c, 0) / worldTrailScale)
    return worldTrailMax * (1 - e)
}

// MARK: - posing the ants from the record

/// Per-ant constants that never change along a track, so its feet stay
/// planted: the antennae sweep out of step with the neighbours, and the gait
/// starts at a different phase. MODEL (step 55 staggered its ants' sweeps by
/// a fraction of a sweep); the golden ratio spreads them evenly.
func worldAntState(id: Int, dab: Float) -> AntV1.State {
    let golden: Float = 0.618034
    let k: Float = Float(id)
    let sweep: Float = (k * golden).truncatingRemainder(dividingBy: 1)
    let gait: Float = (k * 0.37).truncatingRemainder(dividingBy: 1)
    return AntV1.State(dab: dab, sweepPhase: sweep, gaitOffset: gait, scale: 1)
}

/// Turns the simulation's record into posed ants, frame by frame, keeping
/// one lib/ant/v1 Track per (ant, outing).
final class WorldCast {
    let record: SimFilmRecord
    private var tracks: [Int: AntV1.Track] = [:]

    init(record: SimFilmRecord) {
        self.record = record
    }

    func track(ant i: Int, outing k: Int) -> AntV1.Track {
        let key: Int = i * 100_000 + k
        if let t = tracks[key] { return t }
        var t = AntV1.Track()
        for p in record.trackPoints(ant: i, outing: k) {
            t.append(distance: p.distance, AntV1.Ground(SIMD2<Float>(p.x, p.z), heading: p.yaw))
        }
        tracks[key] = t
        return t
    }

    /// The visible ants at film frame `f`, posed, with their ids.
    func ants(atFrame f: Int, mutant: AntV1.Mutant = .none) -> [(id: Int, posed: AntV1.Posed)] {
        let time: Float = Float(record.simSeconds(atFrame: f))
        var out: [(id: Int, posed: AntV1.Posed)] = []
        for i in 0..<record.antCount {
            let a: SimAntFrame = record.ant(i, atFrame: f)
            if !a.visible { continue }
            let t: AntV1.Track = track(ant: i, outing: Int(a.outing))
            let st: AntV1.State = worldAntState(id: i, dab: a.dab)
            let p: AntV1.Posed = AntV1.pose(distance: a.distance, time: time, track: t, state: st, mutant: mutant)
            out.append((i, p))
        }
        return out
    }
}

// MARK: - ants for the scaling measurement (P3)

/// `count` ants standing in the camera's view, for timing one frame: one
/// to a cell of a 7 × 7 grid over the frame (cells on the sugar pile or the
/// nest left out), a little jittered, each walking straight on its own
/// heading, so no two touch. Seeded, and filled in one shuffled order, so
/// every count is timed on the same arrangement's first `count` ants.
func worldScalingAnts(count: Int, camera: WorldCamera, config: SimWorldConfig) -> [AntV1.Posed] {
    var rng = SimRandom(seed: 79, stream: 0x5CA1E)
    let w: Int = 1920
    let h: Int = 1080
    let side: Int = 7
    var cells: [SIMD2<Float>] = []
    for r in 0..<side {
        for c in 0..<side {
            let u: Float = 0.04 + 0.92 * (Float(c) + 0.5) / Float(side)
            let v: Float = 0.08 + 0.84 * (Float(r) + 0.5) / Float(side)
            let px = SIMD2<Float>(u * Float(w), v * Float(h))
            let g: SIMD2<Float> = camera.groundPoint(pixel: px, width: w, height: h)
            let clearSugar: Float = config.sugarRadius + 4
            let clearNest: Float = config.nestRadius + 4
            if simd_distance(g, config.sugar) < clearSugar || simd_distance(g, config.nest) < clearNest { continue }
            cells.append(g)
        }
    }
    for k in stride(from: cells.count - 1, to: 0, by: -1) {
        let bound: UInt64 = UInt64(k + 1)
        let draw: UInt64 = rng.next() % bound
        let j: Int = Int(draw)
        cells.swapAt(k, j)
    }
    // Cell spacing on the ground, to scale the jitter.
    let p0: SIMD2<Float> = camera.groundPoint(pixel: SIMD2<Float>(0, 0), width: w, height: h)
    let p1: SIMD2<Float> = camera.groundPoint(pixel: SIMD2<Float>(Float(w), 0), width: w, height: h)
    let spacing: Float = simd_distance(p0, p1) * 0.92 / Float(side)
    var out: [AntV1.Posed] = []
    for n in 0..<min(count, cells.count) {
        let jx: Float = rng.float(-0.08, 0.08) * spacing
        let jz: Float = rng.float(-0.05, 0.05) * spacing
        let g: SIMD2<Float> = cells[n] + SIMD2<Float>(jx, jz)
        let flip: Bool = rng.next() % 2 == 0
        let heading: Float = rng.float(-0.2, 0.2) + (flip ? Float.pi : 0)
        let distance: Float = rng.float(0, 50)
        let time: Float = rng.float(0, 10)
        let st: AntV1.State = worldAntState(id: n, dab: 0)
        out.append(AntV1.pose(at: g, heading: heading, distance: distance, time: time, state: st))
    }
    return out
}
