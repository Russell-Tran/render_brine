// An analytic yardstick for the cervical stress, so the lattice's number at the
// neck can be checked against something that is not another lattice.
//
// Above the alveolar crest the tooth is a short cantilever. Euler–Bernoulli on
// the CEJ section (width 2·cejHalfWidth, depth sectionThickness) gives the
// bending stress on each face from the moment of the contact force about the
// section centroid, plus the axial term. That is the DENTIN-face stress. The
// enamel skin on top of it is strain-compatible with the dentin it sits on, so
// it carries E_enamel/E_dentin = 4.7× that stress — which is the actual
// mechanism by which a 60 µm knife edge of enamel is the thing that fails: the
// stiff skin takes the load, the thickness only decides how little energy it
// costs to break it.
//
// Compiled with the step's Sources, as a probe, so every constant is the
// model's own. Run from step019_tooth:
//
//   SDK=/Library/Developer/CommandLineTools/SDKs/MacOSX26.sdk
//   swiftc -O -sdk $SDK Sources/Lattice.swift Sources/Tooth.swift \
//          Sources/Model.swift Scratch/yardstick_cervical.swift -o .build/yardstick
//
// This file is the program's main.swift by construction of the command line
// (the last file listed is treated as main), so top-level code is legal.

import Foundation
import simd

func say(_ s: String) { FileHandle.standardError.write((s + "\n").data(using: .utf8)!) }

struct Yardstick {
    var momentNm: Double
    var dentinPlusX: Double     // Pa, tension positive, on the +x (buccal) face
    var dentinMinusX: Double
    var enamelPlusX: Double
    var enamelMinusX: Double
    var tensionFace: String
}

/// Beam theory at y = 0 for one load case. Moment is taken about the section
/// centroid at the CEJ; positive is anticlockwise, which puts tension on −x.
func yardstick(_ lc: LoadCase) -> Yardstick {
    let w: Double = Double(2 * outerHalfWidth(y: 0)) * 1e-3
    let t: Double = sectionThickness
    let inertia: Double = t * w * w * w / 12
    let area: Double = t * w
    let fx: Double = lc.newtons * Double(lc.direction.x)
    let fy: Double = lc.newtons * Double(lc.direction.y)
    let xc: Double = Double(lc.contactX) * 1e-3
    let yc: Double = Double(occlusalHeight(x: lc.contactX)) * 1e-3
    let moment: Double = fx * yc - fy * xc
    let axial: Double = fy / area
    let half: Double = w / 2
    let plus: Double = -moment * half / inertia + axial
    let minus: Double = moment * half / inertia + axial
    let skin: Double = enamelModulus / dentinModulus
    return Yardstick(momentNm: moment, dentinPlusX: plus, dentinMinusX: minus,
                     enamelPlusX: plus * skin, enamelMinusX: minus * skin,
                     tensionFace: plus > minus ? "+x (buccal)" : "−x (lingual)")
}

say(String(format: "CEJ section %.1f mm wide × %.0f mm deep; enamel/dentin modulus ratio %.2f",
           Double(2 * outerHalfWidth(y: 0)), sectionThickness * 1e3,
           enamelModulus / dentinModulus))
say("")

for lc in [axialBruxing, lateralBruxing, outerInclineBruxing] {
    let y = yardstick(lc)
    say("=== \(lc.name)   F = (\(Int(lc.newtons * Double(lc.direction.x))), \(Int(lc.newtons * Double(lc.direction.y)))) N at x = \(lc.contactX) mm")
    say(String(format: "  beam:    M = %+.2f N·m   dentin face  +x %6.1f  −x %6.1f MPa   enamel skin  +x %6.1f  −x %6.1f MPa   tension on %@",
               y.momentNm, y.dentinPlusX / 1e6, y.dentinMinusX / 1e6,
               y.enamelPlusX / 1e6, y.enamelMinusX / 1e6, y.tensionFace))

    let m = ToothModel(spacingMM: 0.10, loadCase: lc, seed: 1)
    let out = m.solve()
    // The lattice's own reads of the same place: raw bond stress on the
    // cervical surface, each side separately, and the smoothed value.
    var bestPlus: Float = 0
    var bestMinus: Float = 0
    for n in m.surfaceNodes {
        let q: SIMD2<Float> = m.mm[n]
        guard q.y >= -0.5, q.y <= 0.5 else { continue }
        let s: Float = m.lattice.peakBondStress(n)
        if q.x > 0 { bestPlus = max(bestPlus, s) } else { bestMinus = max(bestMinus, s) }
    }
    let smoothed = m.peakSurfaceTension(smoothedOver: 0.33)
    say(String(format: "  lattice: raw cervical bond stress  +x %6.1f  −x %6.1f MPa   (res %.1e, %d its)   smoothed peak %5.1f MPa at (%.2f, %.2f)",
               bestPlus / 1e6, bestMinus / 1e6, out.residual, out.iterations,
               smoothed.stressPa / 1e6, smoothed.at.x, smoothed.at.y))
    say("")
}
