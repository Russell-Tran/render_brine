// The cloven hoof, read off the drawn toy: two claws on every foot, with the
// cleft between them open where the claws meet the table.

import Foundation
import Metal
import simd

func testHooves(_ renderer: ToyRenderer?, toy: PosedToy) {
    test("cloven hooves: every foot has two claws, digits III and IV, and the cleft between them is open (GPU)") {
        guard let r = renderer else { expect(false, "no GPU"); return }
        var report: [String] = []
        for (i, s) in toy.segments.enumerated() where s.part == .foot {
            let claws: [Prim] = s.prims.filter { $0.tag == .toe }
            expectEqual(claws.count, 2)
            // A line straight across the foot, a little behind the claw tips
            // and just above the table, probed on the kernel that draws.
            let tipX: Float = claws.map { $0.b.x }.max() ?? 0
            let reach: Float = claws.map { abs($0.b.z) + $0.rb }.max() ?? 1
            let n: Int = 400
            var pts: [SIMD3<Float>] = []
            for k in 0..<n {
                let z: Float = -reach * 1.3 + 2.6 * reach * Float(k) / Float(n - 1)
                pts.append(toy.frames[i].toWorld(SIMD3<Float>(tipX - 0.05, 0.12, z)))
            }
            guard let d = try? r.probe(pts, toy: toy) else { expect(false, "probe failed"); return }
            var runs: Int = 0
            var gap: Float = 0
            var inside: Bool = false
            var gapStart: Int = -1
            let dz: Float = 2.6 * reach / Float(n - 1)
            for k in 0..<n {
                let solid: Bool = d[k].z < 0
                if solid && !inside { runs += 1; if gapStart >= 0 { gap = max(gap, Float(k - gapStart) * dz) } }
                if !solid && inside { gapStart = k }
                inside = solid
            }
            report.append(String(format: "%@: %d claw%@, cleft %.2f mm", s.name, runs, runs == 1 ? "" : "s", gap))
            expectEqual(runs, 2)
            expect(gap > 0.2, "\(s.name): the cleft is \(gap) mm — closed")
        }
        print("        " + report.joined(separator: "; "))
    }
}
