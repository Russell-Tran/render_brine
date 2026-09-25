// GIF 2: three ways a plasmid gets in, side by side, at one scale and on one
// clock, each labelled with how well it is actually known.
//
// This is the point of step 10. The three routes are not three variations on a
// theme — they sit at three completely different levels of evidence, and a
// render that showed them identically would be lying by omission:
//
//   CHEMICAL (CaCl2 + heat shock)   MODEL. Used since 1970, run in thousands of
//        classrooms, and its molecular mechanism has never been established.
//        Drawn here in full anyway — a model that has guided fifty years of
//        successful protocol has earned a depiction — with the caption saying
//        "our best model" rather than "what happens".
//   ELECTROPORATION                 SIMULATED. The physics is understood and
//        molecular-dynamics studies do produce hydrophilic pores under field,
//        so there is a real shape to draw, though a simulated one.
//   NATURAL COMPETENCE              MEASURED. A solved protein structure:
//        Bacillus subtilis ComEA, PDB 8DFK. E. coli K-12 has largely lost this
//        machinery, which is exactly why the lab reaches for calcium and heat.

import Foundation
import simd

enum Route: Int, CaseIterable {
    case chemical = 0, electroporation, competence

    var title: String {
        switch self {
        case .chemical: return "CaCl₂ + heat shock"
        case .electroporation: return "Electroporation"
        case .competence: return "Natural competence"
        }
    }
    var evidence: Evidence {
        switch self {
        case .chemical: return .model
        case .electroporation: return .simulated
        case .competence: return .measured
        }
    }
    var note: String {
        switch self {
        case .chemical: return "mechanism never established · our best model"
        case .electroporation: return "pores seen in molecular-dynamics simulation"
        case .competence: return "solved structure · ComEA, PDB 8DFK"
        }
    }
    /// What the kit on a classroom bench actually does.
    var isTheLabMethod: Bool { self == .chemical }
}

/// A protein read from a PDB file, one bead per alpha carbon — a standard
/// coarse graining, and the right grain next to lipids drawn at 2.6 Å beads.
struct Protein {
    var beads: [SIMD3<Float>]
    var source: String

    init(pdb url: URL, source: String) throws {
        var out: [SIMD3<Float>] = []
        let text = try String(contentsOf: url, encoding: .utf8)
        for line in text.split(separator: "\n", omittingEmptySubsequences: false) {
            guard line.hasPrefix("ATOM") , line.count >= 54 else { continue }
            let chars = Array(line)
            let name = String(chars[12..<16]).trimmingCharacters(in: .whitespaces)
            guard name == "CA" else { continue }
            // Only the first alternate location, so a residue is not doubled.
            let alt = chars[16]
            guard alt == " " || alt == "A" else { continue }
            guard let x = Float(String(chars[30..<38]).trimmingCharacters(in: .whitespaces)),
                  let y = Float(String(chars[38..<46]).trimmingCharacters(in: .whitespaces)),
                  let z = Float(String(chars[46..<54]).trimmingCharacters(in: .whitespaces))
            else { continue }
            out.append(SIMD3(x, y, z))
        }
        beads = out
        self.source = source
    }

    /// Centred on the origin, with its longest axis stood up along y so it can
    /// span a membrane.
    func upright(scale: Float = 1) -> [SIMD3<Float>] {
        guard !beads.isEmpty else { return [] }
        var c = SIMD3<Float>.zero
        for b in beads { c += b }
        c /= Float(beads.count)
        let centred = beads.map { $0 - c }
        // The longest principal direction, found by power iteration on the
        // covariance — enough to decide which way is "long".
        var v = SIMD3<Float>(0, 1, 0)
        for _ in 0..<24 {
            var next = SIMD3<Float>.zero
            for p in centred { next += p * simd_dot(p, v) }
            let n = simd_length(next)
            if n < 1e-6 { break }
            v = next / n
        }
        // Rotation taking v to +y.
        let target = SIMD3<Float>(0, 1, 0)
        let axis = simd_cross(v, target)
        let s = simd_length(axis)
        if s < 1e-6 { return centred.map { $0 * scale } }
        let k = axis / s
        let cosA = simd_dot(v, target)
        let sinA = s
        func rot(_ p: SIMD3<Float>) -> SIMD3<Float> {
            p * cosA + simd_cross(k, p) * sinA + k * simd_dot(k, p) * (1 - cosA)
        }
        return centred.map { rot($0) * scale }
    }
}

// MARK: - Opening a way through the membrane

/// Shapes for one route at time `t` (0 shut, 1 fully open).
///
/// All three start from exactly the same membrane, so the only difference on
/// screen is what each route does to it.
func routeShapes(_ scene: Scene, base: [GPUShape], route: Route, t: Float,
                 omIndices: [Int], dnaIndices: [Int], ionIndices: [Int],
                 protein: [SIMD3<Float>], reach: Float) -> [GPUShape] {
    var out = base
    let open = smoothstep01(t)
    let face = scene.layers.omOuterFace
    let centreX: Float = 0, centreZ: Float = 0

    // The plasmid presses against the membrane in every panel.
    let drop = SIMD3<Float>(0, face + reach + 4, 0)
    for i in dnaIndices { out[i].a = SIMD4(scene.beads[i].position + drop, out[i].a.w) }
    for i in ionIndices where route == .chemical {
        // Calcium is the whole point of the chemical route, so it is kept there
        // and left out of the other two, where it plays no part.
        out[i].a = SIMD4(scene.beads[i].position + drop, out[i].a.w)
    }
    if route != .chemical {
        for i in ionIndices { out[i].radius = 0 }
    }

    switch route {
    case .chemical:
        // The model: charge screened, the membrane loosened by the cold and
        // then jolted at 42 °C, and a transient gap appearing. Drawn as a
        // shallow, ragged opening — deliberately less crisp than the
        // electroporation pore beside it, because far less is known about it.
        let radius: Float = 34 * open
        for i in omIndices {
            let p = scene.beads[i].position
            let d = sqrt((p.x - centreX) * (p.x - centreX) + (p.z - centreZ) * (p.z - centreZ))
            if d < radius {
                // A soft edge rather than a clean rim: the gap has no agreed shape.
                let fade = min(max((radius - d) / max(radius, 1e-3), 0), 1)
                out[i].radius = scene.beads[i].radius * (1 - fade)
            }
        }
    case .electroporation:
        // A hydrophilic pore: lipids clear from the middle and their heads
        // curl round to line the walls, which is what simulations show.
        let radius: Float = 30 * open
        for i in omIndices {
            let p = scene.beads[i].position
            let d = sqrt((p.x - centreX) * (p.x - centreX) + (p.z - centreZ) * (p.z - centreZ))
            if d < radius {
                out[i].radius = 0
            } else if d < radius + 12 && scene.beads[i].part == "head" {
                // Rim heads pulled inward and downward, lining the pore.
                let pull = (radius + 12 - d) / 12
                let inward = simd_normalize(SIMD3<Float>(centreX - p.x, 0, centreZ - p.z) + SIMD3(1e-6, 0, 0))
                let moved = p + inward * (pull * 7) + SIMD3(0, -pull * 6 * (p.y > scene.layers.omCenter ? 1 : -1), 0)
                out[i].a = SIMD4(moved, out[i].a.w)
            }
        }
    case .competence:
        // A real machine. ComEA is placed spanning the membrane and the lipids
        // it would displace are cleared for it.
        let radius: Float = 26
        for i in omIndices {
            let p = scene.beads[i].position
            let d = sqrt((p.x - centreX) * (p.x - centreX) + (p.z - centreZ) * (p.z - centreZ))
            if d < radius * open { out[i].radius = 0 }
        }
        let colour = SIMD3<Float>(0.62, 0.45, 0.80)
        for b in protein {
            let p = SIMD3(b.x + centreX, b.y + scene.layers.omCenter + 10, b.z + centreZ)
            out.append(GPUShape.sphere(center: p, radius: 3.6 * open, color: colour))
        }
    }
    return out
}
