// Analytic validation of the solver. Tension: delta = FL/EA. Bending:
// delta = FL^3/3EI. Both ratios climb toward 1 as the lattice refines; the
// bending error is exactly 3/rows.
import Foundation
import simd
let a: Double = 1e-4, t: Double = 1e-3, E: Double = 20e9
let root3 = 3.0.squareRoot()
func beam(rows: Int, cols: Int, mode: String) -> (Double, Float, Int) {
    let props: [Tissue: TissueProperties] = [
        .dentin: TissueProperties(youngsModulus: E, criticalStrain: 1e9, name: "beam")]
    let lat = Lattice(cols: cols, rows: rows, spacing: a, thickness: t, properties: props,
                      tissueAt: { _ in .dentin },
                      kindAt: { p, _ in p.x < Float(1.6 * a) ? .fixed : .free })
    let h = Double(rows - 1) * a * root3 / 2
    let L = Double(cols - 1) * a
    let F: Double = mode == "bend" ? 0.02 : 20.0
    for j in 0..<rows {
        let f = mode == "bend" ? SIMD2<Float>(0, Float(-F / Double(rows)))
                               : SIMD2<Float>(Float(F / Double(rows)), 0)
        lat.setLoad(f, at: lat.index(col: cols - 1, row: j))
    }
    let out = lat.solve(tolerance: 1e-8, maxIterations: 20000)
    let mid = lat.index(col: cols - 1, row: rows / 2)
    let measured = mode == "bend" ? Double(-lat.displacement(mid).y) : Double(lat.displacement(mid).x)
    let analytic = mode == "bend"
        ? F * L * L * L / (3 * E * (t * h * h * h / 12))
        : F * L / (E * (h * t))
    return (measured / analytic, out.residual, out.iterations)
}
for r in [11, 21, 41] {
    let (ratio, res, its) = beam(rows: r, cols: 60, mode: "pull")
    print(String(format: "tension rows %3d  ratio %.4f  residual %.1e  %d its", r, ratio, res, its))
}
for r in [11, 21, 31, 41] {
    let (ratio, res, its) = beam(rows: r, cols: 12 * r, mode: "bend")
    print(String(format: "bend    rows %3d  ratio %.4f  residual %.1e  %d its", r, ratio, res, its))
}
