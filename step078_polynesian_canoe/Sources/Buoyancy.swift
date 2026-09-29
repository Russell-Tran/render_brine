// Where the canoe floats: Archimedes. The hulls sink until the sea water
// they push aside weighs what the canoe weighs, fully loaded:
//
//     ρ_sea · V(draft) = M,   M = 25,000 lb [PVS-plan], ρ_sea = 1023.6 kg/m³ [Nayar]
//
// V(w), the volume of the canoe below a waterline w above its keel, comes
// from the kernel's own distance function: on a 1 cm grid of vertical lines
// through the hulls, the `columns` kernel finds where each line enters the
// wet parts (the two hulls and the steering paddle's blade) from below and
// leaves them at the top, and each line holds min(w, top) − bottom of water.
// Bisection on w then finds the draft. Nothing is placed by eye.

import Foundation
import simd

struct ColumnGrid {
    let dx: Float
    let dz: Float
    /// (bottom, top) of the wet parts on each line; misses dropped.
    let spans: [SIMD2<Float>]

    /// Cubic metres under a waterline `w` metres above the keel.
    func volume(below w: Float) -> Double {
        var v: Double = 0
        for s in spans where s.x < w {
            v += Double(min(w, s.y) - s.x)
        }
        let area: Double = Double(dx) * Double(dz)
        return v * area
    }
}

/// The wet parts' spans on a grid of spacing `step`, shifted by `offset`
/// (a fraction of a cell), over the drawn canoe.
func columnGrid(_ r: CanoeRenderer, step: Float, offset: Float = 0.5, mutant: Mutant = activeMutant) throws -> ColumnGrid {
    let s: Float = canoeScale(mutant)
    let x0: Float = -11.8 * s
    let x1: Float = 9.8 * s
    let z0: Float = -2.5 * s
    let z1: Float = 2.5 * s
    let nx: Int = Int(((x1 - x0) / step).rounded(.up))
    let nz: Int = Int(((z1 - z0) / step).rounded(.up))
    var xz: [SIMD2<Float>] = []
    xz.reserveCapacity(nx * nz)
    for i in 0..<nx {
        let x: Float = x0 + (Float(i) + offset) * step
        for j in 0..<nz {
            let z: Float = z0 + (Float(j) + offset) * step
            xz.append(SIMD2<Float>(x, z))
        }
    }
    let raw: [SIMD2<Float>] = try r.columns(xz)
    let spans: [SIMD2<Float>] = raw.filter { $0.x < 1e8 && $0.y > -1e8 && $0.y > $0.x }
    return ColumnGrid(dx: step, dz: step, spans: spans)
}

struct Flotation {
    /// Height of the waterline above the keel, metres: the draft.
    let draft: Float
    /// Cubic metres displaced at that draft.
    let displaced: Double
    /// Cubic metres this hull would displace at PVS's printed draft.
    let atPVSDraft: Double
    /// ρ · V − M, kilograms.
    var imbalance: Double { seaWaterDensity * displaced - canoeMass }
}

/// Bisect for the waterline at which the displaced sea water weighs M.
func solveDraft(_ g: ColumnGrid, mutant: Mutant = activeMutant) -> Flotation {
    let want: Double = canoeMass / seaWaterDensity
    var lo: Float = 0
    var hi: Float = gunwaleHeight * canoeScale(mutant)
    for _ in 0..<60 {
        let mid: Float = (lo + hi) / 2
        if g.volume(below: mid) < want { lo = mid } else { hi = mid }
    }
    let w: Float = (lo + hi) / 2
    let pvs: Float = Float(pvsDraft) * canoeScale(mutant)
    return Flotation(draft: w, displaced: g.volume(below: w), atPVSDraft: g.volume(below: pvs))
}

/// Where the canoe is drawn: the computed draft, or, for the mutant, a
/// waterline put where it looked right — 30% deeper.
func placedDraft(_ f: Flotation, mutant: Mutant = activeMutant) -> Float {
    mutant == .floatsWrong ? f.draft * 1.3 : f.draft
}

/// The waterline's length: along the starboard hull's centre, on the sea's
/// surface, how far the hull reaches (1 mm steps, off the kernel's own
/// distance function).
func waterlineLength(_ r: CanoeRenderer, sink: Float, mutant: Mutant = activeMutant) throws -> Float {
    let s: Float = canoeScale(mutant)
    let n: Int = 24_000
    var pts: [SIMD3<Float>] = []
    pts.reserveCapacity(n)
    for i in 0..<n {
        let fi: Float = Float(i)
        let mm: Float = fi * 0.001
        let u: Float = mm - 12
        let x: Float = u * s
        pts.append(SIMD3<Float>(x, 0, hullCentreZ * s))
    }
    let pr: [SIMD4<Float>] = try r.probe(pts, sink: sink)
    var lo: Float = .infinity, hi: Float = -.infinity
    for i in 0..<n where pr[i].z < 0 {
        lo = min(lo, pts[i].x)
        hi = max(hi, pts[i].x)
    }
    return hi > lo ? hi - lo : 0
}

/// The near (port) hull's keel under the sea, in world coordinates, found
/// by the `columns` kernel on 1 cm-apart lines down the hull's centre:
/// for the dashed keel line in the picture.
func keelLine(_ r: CanoeRenderer, sink: Float, mutant: Mutant = activeMutant) throws -> [SIMD3<Float>] {
    let s: Float = canoeScale(mutant)
    let z: Float = -hullCentreZ * s
    var xz: [SIMD2<Float>] = []
    for i in 0..<2000 {
        let fi: Float = Float(i)
        let u: Float = fi * 0.01 - 10
        xz.append(SIMD2<Float>(u * s, z))
    }
    let spans: [SIMD2<Float>] = try r.columns(xz)
    var out: [SIMD3<Float>] = []
    for (i, c) in xz.enumerated() where spans[i].x < sink && spans[i].y > spans[i].x {
        out.append(SIMD3<Float>(c.x, spans[i].x - sink, z))
    }
    return out
}
