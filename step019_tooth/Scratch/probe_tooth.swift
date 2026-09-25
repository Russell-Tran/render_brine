import Foundation
import simd

let spacing: Double = 1.0e-4          // 0.1 mm
let thickness: Double = 10.0e-3       // 10 mm out of plane
let ox: Float = -5.4, oy: Float = -13.2
let cols = Int(11.0e-3 / spacing) + 1
let rows = Int(21.2e-3 / (spacing * 3.0.squareRoot() / 2)) + 1
func toMM(_ p: SIMD2<Float>) -> SIMD2<Float> { SIMD2(p.x * 1000 + ox, p.y * 1000 + oy) }

let lat = Lattice(cols: cols, rows: rows, spacing: spacing, thickness: thickness,
                  properties: toothProperties,
                  tissueAt: { tissueAt(toMM($0)) },
                  kindAt: { p, t in kindAt(toMM(p), t) })

var counts: [Tissue: Int] = [:]
for t in lat.tissue { counts[t, default: 0] += 1 }
print("nodes \(cols)x\(rows) = \(cols*rows), bonds \(lat.bonds.count)")
for t in Tissue.allCases { print("  \(t): \(counts[t] ?? 0)") }

func applyLoad(_ c: LoadCase) {
    var nodes: [Int] = []
    for n in 0..<lat.tissue.count {
        guard lat.tissue[n] == .enamel else { continue }
        let q = toMM(lat.rest[n])
        guard abs(q.x - c.contactX) < c.contactHalfWidth else { continue }
        guard q.y > occlusalHeight(x: q.x) - 0.25 else { continue }
        nodes.append(n)
    }
    let each = Float(c.newtons) / Float(max(nodes.count, 1))
    for n in nodes { lat.setLoad(c.direction * each, at: n) }
    print("  \(c.name): \(nodes.count) contact nodes")
}
applyLoad(lateralBruxing)

let out = lat.solve(tolerance: 1e-7, maxIterations: 30000)
print(String(format: "solve: residual %.2e in %d iterations", out.residual, out.iterations))

// Where is the peak tensile stress on the OUTER surface?
var best: Float = -1e30; var bestY: Float = 0; var bestX: Float = 0
for n in 0..<lat.tissue.count {
    guard lat.tissue[n].isSolid else { continue }
    let q = toMM(lat.rest[n])
    let outer = outerHalfWidth(y: q.y)
    guard abs(abs(q.x) - outer) < 0.15 else { continue }     // surface nodes only
    guard q.y > -6, q.y < 7 else { continue }
    let s = lat.maxPrincipalStress(n)
    if s > best { best = s; bestY = q.y; bestX = q.x }
}
print(String(format: "peak surface tension %.1f MPa at x=%.2f y=%.2f mm (CEJ is y=0)", best/1e6, bestX, bestY))

// Crack propagation.
var rounds = 0; var total = 0
while rounds < 60 {
    let broke = lat.breakOverstrained()
    if broke == 0 { break }
    total += broke
    rounds += 1
    _ = lat.solve(tolerance: 1e-7, maxIterations: 30000)
}
print("cracking: \(total) bonds broken over \(rounds) rounds")
var byTissue: [String: Int] = [:]
var deepestY: Float = 99
for (i, b) in lat.bonds.enumerated() where b.broken {
    _ = i
    let ta = lat.tissue[Int(b.a)], tb = lat.tissue[Int(b.b)]
    let key = ta == tb ? "\(ta)" : "\(ta)|\(tb)"
    byTissue[key, default: 0] += 1
    let q = toMM(lat.rest[Int(b.a)])
    let outer = outerHalfWidth(y: q.y)
    let depth = outer - abs(q.x)
    if q.y > 0 && depth < deepestY { }
}
for (k, v) in byTissue.sorted(by: { $0.value > $1.value }) { print("  broken in \(k): \(v)") }
