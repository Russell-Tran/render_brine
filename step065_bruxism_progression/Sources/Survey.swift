// Measuring the worn teeth as a dentist's index would: drop straight down onto
// every tooth on a fine grid, from the same distance functions that drew the
// picture, and read back what each point is — unworn enamel, facet, or exposed
// dentine. The tests score the Tooth Wear Index from this; the labels find
// their features with it; step 65 times its captions with it.

import Foundation
import Metal
import simd

/// One point of the survey, in the tooth's own frame.
struct SurveyPoint {
    var uv: SIMD2<Float>
    var height: Float
    var surface: Int            // 0 unworn enamel, 1 facet enamel, 2 exposed dentine
}

/// One tooth's survey.
struct ToothSurvey {
    var index: Int
    var tooth: PlacedTooth
    var points: [SurveyPoint]

    var spec: ToothSpec { tooth.spec }
    var worn: [SurveyPoint] { points.filter { $0.surface > 0 } }
    var dentine: [SurveyPoint] { points.filter { $0.surface == 2 } }

    /// Tooth-local (u, v, y) to world.
    func world(_ uv: SIMD2<Float>, _ y: Float) -> SIMD3<Float> {
        let c = SIMD3<Float>(tooth.centre.x, 0, tooth.centre.y)
        let t = SIMD3<Float>(tooth.tangent.x, 0, tooth.tangent.y)
        let o = SIMD3<Float>(tooth.outward.x, 0, tooth.outward.y)
        return c + t * uv.x + o * uv.y + SIMD3<Float>(0, y, 0)
    }

    /// The occlusal table, the surface the occlusal criteria score: in plan,
    /// the crown's top outline in step 20's own shape — as wide as the contacts
    /// mesiodistally and `topHalfDepth` across, which Wheeler's puts at 55–65%
    /// of the crown's depth — with the same rounded corners as its walls.
    func onOcclusalTable(_ uv: SIMD2<Float>) -> Bool {
        let a: Float = spec.width / 2
        let b: Float = topHalfDepth(spec)
        let r: Float = min(a, b) * 0.7
        let q: SIMD2<Float> = simd_abs(uv) - SIMD2<Float>(a, b) + SIMD2<Float>(r, r)
        let outside: Float = simd_length(simd_max(q, SIMD2<Float>(0, 0)))
        let inside: Float = min(max(q.x, q.y), 0)
        return outside + inside - r <= 0
    }

    /// The fraction of the occlusal table worn through to dentine.
    var occlusalDentineFraction: Float {
        let table: [SurveyPoint] = points.filter { onOcclusalTable($0.uv) }
        guard !table.isEmpty else { return 0 }
        let d: Int = table.filter { $0.surface == 2 }.count
        return Float(d) / Float(table.count)
    }

    /// How deep the facet has gone into dentine, at its deepest: mm below the
    /// bottom of the enamel cap (the unworn top less the cap's thickness).
    var dentineLoss: Float {
        var worst: Float = 0
        for p in dentine {
            let capBottom: Float = crownTopCPU(spec, p.uv) - topEnamel(spec.kind)
            worst = max(worst, capBottom - p.height)
        }
        return worst
    }

    /// The Tooth Wear Index score this tooth's surface has, measured.
    func stage(depth: Float) -> Int {
        switch twiSurface(spec.kind) {
        case .incisal:
            return incisalStage(worn: !worn.isEmpty, dentineExposed: !dentine.isEmpty,
                                dentineLoss: dentineLoss, depth: depth)
        case .occlusal:
            return occlusalStage(worn: !worn.isEmpty, dentineFraction: occlusalDentineFraction)
        }
    }

    /// The least-squares plane y = c0 + c1·u + c2·v through every worn point,
    /// and how far the points stray from it: (coefficients, rms, worst).
    func facetFit() -> (c: SIMD3<Double>, rms: Double, worst: Double)? {
        let w: [SurveyPoint] = worn
        guard w.count >= 6 else { return nil }
        var m = simd_double3x3(0)
        var rhs = SIMD3<Double>(0, 0, 0)
        for p in w {
            let row = SIMD3<Double>(1, Double(p.uv.x), Double(p.uv.y))
            m += simd_double3x3(columns: (row * row.x, row * row.y, row * row.z))
            rhs += row * Double(p.height)
        }
        let c: SIMD3<Double> = m.inverse * rhs
        var sq: Double = 0
        var worst: Double = 0
        for p in w {
            let fit: Double = c.x + c.y * Double(p.uv.x) + c.z * Double(p.uv.y)
            let e: Double = abs(fit - Double(p.height))
            sq += e * e
            worst = max(worst, e)
        }
        return (c, (sq / Double(w.count)).squareRoot(), worst)
    }
}

/// Survey every tooth at wear `depth`, on a grid `step` mm apart.
func surveyTeeth(depth: Float, step: Float = 0.1, mutant: Mutant = .none,
                 on device: MTLDevice) throws -> [ToothSurvey] {
    let placed: [PlacedTooth] = placeTeeth()
    var starts: [SIMD3<Float>] = []
    var owners: [(tooth: Int, uv: SIMD2<Float>)] = []
    for (i, t) in placed.enumerated() {
        let a: Float = t.spec.width / 2
        let b: Float = t.spec.depth / 2
        let nu: Int = Int((2 * a / step).rounded())
        let nv: Int = Int((2 * b / step).rounded())
        for iu in 0...nu {
            for iv in 0...nv {
                let uv = SIMD2<Float>(-a + Float(iu) * step, -b + Float(iv) * step)
                let s = ToothSurvey(index: i, tooth: t, points: [])
                starts.append(s.world(uv, 3.0))
                owners.append((i, uv))
            }
        }
    }
    let hits: [SIMD4<Float>] = try dropOntoTeeth(starts, depth: depth, mutant: mutant, on: device)
    var out: [ToothSurvey] = placed.enumerated().map { ToothSurvey(index: $0.offset, tooth: $0.element, points: []) }
    for (k, h) in hits.enumerated() where h.w > 0.5 && Int(h.z) == owners[k].tooth {
        out[owners[k].tooth].points.append(SurveyPoint(uv: owners[k].uv, height: h.x, surface: Int(h.y)))
    }
    return out
}

/// Straight down at each tooth's facet anchor: the surface height there.
func anchorHeights(depth: Float, mutant: Mutant = .none, on device: MTLDevice) throws -> [Float] {
    let placed: [PlacedTooth] = placeTeeth()
    let starts: [SIMD3<Float>] = placed.enumerated().map { i, t in
        let a: SIMD3<Float> = facetAnchorLocal(t.spec)
        return ToothSurvey(index: i, tooth: t, points: []).world(SIMD2<Float>(a.x, a.y), 3.0)
    }
    return try dropOntoTeeth(starts, depth: depth, mutant: mutant, on: device).map { $0.x }
}
