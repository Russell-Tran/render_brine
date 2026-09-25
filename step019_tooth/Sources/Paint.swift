// Turning the solved lattice into the two numbers the kernel samples on the cut
// face: a stress, and a crack.
//
// The lattice is triangular and the field is a plain grid, so this is a
// resample. It is done on the CPU once per distinct state rather than per frame,
// because a state is a solve and there are twenty-odd of them in a loop of a
// hundred and forty-four frames.
//
// The crack is the part with a subtlety in it. A broken bond is not a crack
// along the bond — it is a crack ACROSS it. The two nodes have come apart along
// the bond's own direction, so the free surface that has opened runs
// perpendicular to it, through its midpoint. Drawing the bonds themselves would
// draw the crack rotated by ninety degrees, which looks plausible and is wrong.

import Foundation
import simd

extension StressField {
    /// Clear the field and fill the stress channel from a solved model.
    /// Millimetres in, megapascals out.
    mutating func paint(_ model: ToothModel, gain: Float = 1) {
        for i in 0..<texels.count { texels[i] = .zero }
        let lattice = model.lattice
        let stride: Float = 1 / perMM
        // Walk the lattice, not the grid: every node stamps the grid cells it
        // owns. Going the other way would need a search per cell.
        let reach: Int = max(Int((Float(model.spacingMM) * 0.62 * perMM).rounded(.up)), 1)
        for n in 0..<lattice.tissue.count {
            guard lattice.tissue[n].isTooth else { continue }
            let value: Float = lattice.maxPrincipalStress(n) * gain / 1e6
            let p: SIMD2<Float> = model.mm[n]
            let (ci, cj) = cell(x: p.x, y: p.y)
            for dj in -reach...reach {
                for di in -reach...reach {
                    let i: Int = ci + di
                    let j: Int = cj + dj
                    guard inside(i, j) else { continue }
                    let qx: Float = originX + Float(i) * stride
                    let qy: Float = originY + Float(j) * stride
                    let d: SIMD2<Float> = SIMD2(qx, qy) - p
                    guard simd_length(d) < Float(model.spacingMM) * 0.62 else { continue }
                    var t: SIMD2<Float> = self[i, j]
                    t.x = t.x == 0 ? value : (t.x + value) * 0.5
                    self[i, j] = t
                }
            }
        }
    }

    /// Stamp the crack surfaces of every broken bond into the y channel.
    /// `fade` lets a newly broken bond come in over a frame or two.
    mutating func paintCracks(_ model: ToothModel, upTo limit: Int = .max,
                              widthMM: Float = 0.055) {
        let lattice = model.lattice
        let order: [Int32] = lattice.breakOrder
        let count: Int = min(limit, order.count)
        var seen = Set<Int32>()
        for k in 0..<count {
            let id: Int32 = order[k]
            guard !seen.contains(id) else { continue }
            seen.insert(id)
            let bond = lattice.bonds[Int(id)]
            let a = Int(bond.a)
            let b = Int(bond.b)
            guard lattice.tissue[a].isTooth || lattice.tissue[b].isTooth else { continue }
            let pa: SIMD2<Float> = model.mm[a]
            let pb: SIMD2<Float> = model.mm[b]
            let mid: SIMD2<Float> = (pa + pb) * 0.5
            let along: SIMD2<Float> = simd_normalize(pb - pa)
            let across = SIMD2<Float>(-along.y, along.x)
            let half: Float = Float(model.spacingMM) * 0.5
            stamp(from: mid - across * half, to: mid + across * half, width: widthMM)
        }
    }

    /// A thick segment into the crack channel, by distance to the segment.
    private mutating func stamp(from a: SIMD2<Float>, to b: SIMD2<Float>, width: Float) {
        let stride: Float = 1 / perMM
        let lo = simd_min(a, b) - SIMD2<Float>(repeating: width + stride)
        let hi = simd_max(a, b) + SIMD2<Float>(repeating: width + stride)
        let (i0, j0) = cell(x: lo.x, y: lo.y)
        let (i1, j1) = cell(x: hi.x, y: hi.y)
        let ab: SIMD2<Float> = b - a
        let denom: Float = max(simd_dot(ab, ab), 1e-12)
        for j in j0...j1 {
            for i in i0...i1 {
                guard inside(i, j) else { continue }
                let q = SIMD2<Float>(originX + Float(i) * stride, originY + Float(j) * stride)
                var t: Float = simd_dot(q - a, ab) / denom
                t = min(max(t, 0), 1)
                let d: Float = simd_length(q - (a + ab * t))
                guard d < width + stride else { continue }
                let cover: Float = min(max((width + stride - d) / stride, 0), 1)
                var texel: SIMD2<Float> = self[i, j]
                texel.y = max(texel.y, cover)
                self[i, j] = texel
            }
        }
    }
}
