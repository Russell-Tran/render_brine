// The inset: the canoe cut across at midships, sampled straight from the
// kernel's distance function (a probe per inset pixel), with the sea where
// the buoyancy put it. Like the midship section on PVS's plan.

import Foundation
import simd

/// The section's window, metres: across (z) and up (y, the sea at 0).
let sectionX: Float = 0.0
let sectionZRange: ClosedRange<Float> = -3.3...3.3
let sectionYRange: ClosedRange<Float> = -1.35...2.15

struct SectionInset {
    let columns: Int
    let rows: Int
    /// Material at each cell's centre, row 0 at the top; 0 where empty.
    let mats: [Int]

    func mat(_ i: Int, _ j: Int) -> Int { mats[j * columns + i] }
}

/// Probe a `columns` × `rows` grid over the section's window.
func sampleSection(_ r: CanoeRenderer, sink: Float, columns: Int = 660, rows: Int = 350) throws -> SectionInset {
    let zw: Float = sectionZRange.upperBound - sectionZRange.lowerBound
    let yh: Float = sectionYRange.upperBound - sectionYRange.lowerBound
    var pts: [SIMD3<Float>] = []
    pts.reserveCapacity(columns * rows)
    for j in 0..<rows {
        let fy: Float = (Float(j) + 0.5) / Float(rows)
        let y: Float = sectionYRange.upperBound - fy * yh
        for i in 0..<columns {
            let fz: Float = (Float(i) + 0.5) / Float(columns)
            let z: Float = sectionZRange.lowerBound + fz * zw
            pts.append(SIMD3<Float>(sectionX, y, z))
        }
    }
    let pr: [SIMD4<Float>] = try r.probe(pts, sink: sink)
    let mats: [Int] = pr.map { $0.x < 0 ? Int($0.y) : 0 }
    return SectionInset(columns: columns, rows: rows, mats: mats)
}
