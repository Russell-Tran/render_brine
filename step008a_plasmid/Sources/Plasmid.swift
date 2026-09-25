// The pGLO plasmid from Resources/pglo.json: its sequence, its feature map,
// and the geometry of the whole 5,371 bp circle.
//
// Two levels of detail, because no single one works across the whole dive:
//
//   * At ring scale the plasmid is 5,371 base pairs and about 339,000 atoms,
//     and one atom covers roughly a twentieth of a pixel. So each base pair is
//     drawn as three sticks — two backbone segments and one rung — which is
//     ~16,000 shapes instead of ~339,000, and still shows the real helix.
//   * Close in, real atoms: a base-pair template taken from the 1BNA crystal
//     structure (step 8) is stamped down at each position with the plasmid's
//     own sequence. The stick model's backbone is derived from where the
//     phosphorus atoms actually sit in those templates, so the two levels line
//     up and can be cross-faded.

import Foundation
import simd

func radians(_ degrees: Float) -> Float { degrees * .pi / 180 }

struct Feature {
    var name: String
    var start: Int          // 1-based, inclusive, as GenBank counts
    var end: Int
    var strand: Int
    var note: String
    var color: String       // "gfp", "switch", "bla", "ori" or "grey"
    var role: String
    var length: Int { end - start + 1 }
}

struct TemplateAtom {
    var element: String
    var strand: Int         // 0 or 1
    var name: String        // the PDB atom name, e.g. "P" or "C1'"
    var position: SIMD3<Float>   // in base-pair-local coordinates
}

struct BasePairTemplate {
    var atoms: [TemplateAtom]
    var bonds: [(a: Int, b: Int, order: Int)]
    /// The Watson–Crick hydrogen bonds holding the pair together: the donor's
    /// hydrogen and the acceptor it points at. Three for G:C, two for A:T.
    var hbonds: [(hydrogen: Int, acceptor: Int)]
}

enum PlasmidError: Error, CustomStringConvertible {
    case badFile(String)
    var description: String {
        switch self {
        case .badFile(let detail): return "couldn't read the plasmid: \(detail)"
        }
    }
}

/// Ideal B-DNA, from Wang, PNAS 76:200 (1979) and Watson–Crick geometry.
let riseAngstrom: Float = 3.4
let basePairsPerTurn: Float = 10.5

struct Plasmid {
    var name: String
    var length: Int
    var sequence: [UInt8]                  // ASCII A, C, G, T
    var features: [Feature]
    var templates: [String: BasePairTemplate]

    /// Laid out as a flat circle, the backbone's own length sets the radius.
    var circumference: Float { Float(length) * riseAngstrom }
    var ringRadius: Float { circumference / (2 * .pi) }

    /// A closed circle must contain a whole number of helical turns, but
    /// 5,371 / 10.5 = 511.5. So the drawn molecule has to be over- or
    /// under-wound by at least half a turn — which is the topological reason
    /// real plasmids are supercoiled rather than relaxed. 512 is the nearer
    /// choice (10.490 bp per turn, against 10.511 for 511).
    var helicalTurns: Int {
        let ideal = Float(length) / basePairsPerTurn
        let down = Int(ideal.rounded(.down)), up = down + 1
        return abs(Float(length) / Float(down) - basePairsPerTurn)
             < abs(Float(length) / Float(up) - basePairsPerTurn) ? down : up
    }
    var basePairsPerTurnDrawn: Float { Float(length) / Float(helicalTurns) }

    /// Where each backbone strand sits: how far the phosphorus is from the
    /// helix axis, and each strand's angle around it. Measured from the
    /// crystal templates rather than assumed, so sticks and atoms coincide.
    var backboneRadius: Float = 0
    var strandPhase: [Float] = [0, 0]

    /// The feature covering each base pair, by index, or nil.
    var featureAt: [Int]  = []             // index into `features`, or −1
}

func loadPlasmid(from url: URL) throws -> Plasmid {
    let data = try Data(contentsOf: url)
    guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
          let name = root["name"] as? String,
          let length = root["length"] as? Int,
          let sequence = root["sequence"] as? String,
          let rawFeatures = root["features"] as? [[String: Any]],
          let rawTemplates = root["templates"] as? [String: [String: Any]]
    else { throw PlasmidError.badFile("missing fields") }
    guard sequence.count == length else { throw PlasmidError.badFile("sequence is not \(length) bp") }

    let features = rawFeatures.map {
        Feature(name: $0["name"] as? String ?? "", start: $0["start"] as? Int ?? 0,
                end: $0["end"] as? Int ?? 0, strand: $0["strand"] as? Int ?? 1,
                note: $0["note"] as? String ?? "", color: $0["color"] as? String ?? "grey",
                role: $0["role"] as? String ?? "")
    }
    var templates: [String: BasePairTemplate] = [:]
    for (kind, raw) in rawTemplates {
        guard let rawAtoms = raw["atoms"] as? [[String: Any]],
              let rawBonds = raw["bonds"] as? [[Int]],
              let rawH = raw["hbonds"] as? [[Int]] else { throw PlasmidError.badFile("bad template \(kind)") }
        let atoms = rawAtoms.map { a -> TemplateAtom in
            let p = (a["pos"] as? [NSNumber]) ?? [0, 0, 0]
            return TemplateAtom(element: a["el"] as? String ?? "C", strand: a["strand"] as? Int ?? 0,
                                name: a["name"] as? String ?? "",
                                position: SIMD3(p[0].floatValue, p[1].floatValue, p[2].floatValue))
        }
        templates[kind] = BasePairTemplate(atoms: atoms,
                                           bonds: rawBonds.map { (a: $0[0], b: $0[1], order: $0[2]) },
                                           hbonds: rawH.map { (hydrogen: $0[0], acceptor: $0[1]) })
    }

    var plasmid = Plasmid(name: name, length: length, sequence: Array(sequence.utf8),
                          features: features, templates: templates)

    // Take the backbone's radius and each strand's angle from where the
    // phosphorus atoms really are, averaged over the four templates.
    var radii: [Float] = []
    var phases: [[Float]] = [[], []]
    for (_, t) in templates {
        for strand in 0..<2 {
            guard let p = t.atoms.first(where: { $0.strand == strand && $0.name == "P" }) else {
                throw PlasmidError.badFile("a template is missing a phosphorus")
            }
            radii.append(simd_length(SIMD2(p.position.y, p.position.z)))
            phases[strand].append(atan2(p.position.z, p.position.y))
        }
    }
    plasmid.backboneRadius = radii.reduce(0, +) / Float(radii.count)
    // Angles average through their unit vectors, so the wrap at ±π is harmless.
    plasmid.strandPhase = phases.map { list in
        let s = list.reduce(SIMD2<Float>(0, 0)) { $0 + SIMD2(cos($1), sin($1)) }
        return atan2(s.y, s.x)
    }

    var owner = [Int](repeating: -1, count: length)
    for (i, f) in features.enumerated() where f.start >= 1 && f.end <= length {
        for bp in (f.start - 1)..<f.end { owner[bp] = i }
    }
    plasmid.featureAt = owner
    return plasmid
}

// MARK: - Where each base pair sits on the ring

/// A base pair's own coordinate frame on the ring: the origin is its center,
/// `tangent` runs along the local helix axis, and `y`/`z` turn with the twist.
struct RingFrame {
    var origin: SIMD3<Float>
    var tangent: SIMD3<Float>
    var y: SIMD3<Float>
    var z: SIMD3<Float>
}

extension Plasmid {
    /// Base pair `i`, with the whole plasmid turned `spin` radians about the
    /// axis through the middle of the ring.
    func frame(at i: Int, spin: Float = 0) -> RingFrame {
        let phi = 2 * Float.pi * Float(i) / Float(length) + spin
        let cphi = cos(phi), sphi = sin(phi)
        let origin = SIMD3<Float>(ringRadius * cphi, 0, ringRadius * sphi)
        let tangent = SIMD3<Float>(-sphi, 0, cphi)
        let outward = SIMD3<Float>(cphi, 0, sphi)
        let up = SIMD3<Float>(0, 1, 0)
        // Right-handed: (tangent, outward, up) is a right-handed set, so
        // turning from `outward` toward `up` as i grows winds the strands the
        // way B-DNA winds.
        let theta = 2 * Float.pi * Float(helicalTurns) * Float(i) / Float(length)
        let y = outward * cos(theta) + up * sin(theta)
        return RingFrame(origin: origin, tangent: tangent, y: y, z: simd_cross(tangent, y))
    }

    /// Where strand `s`'s phosphorus sits at base pair `i`.
    func backbonePoint(at i: Int, strand s: Int, spin: Float = 0) -> SIMD3<Float> {
        let f = frame(at: i, spin: spin)
        let a = strandPhase[s]
        return f.origin + (f.y * cos(a) + f.z * sin(a)) * backboneRadius
    }

    /// The base pair nearest a position along the plasmid, as an index.
    func index(ofBasePair bp: Int) -> Int { max(0, min(length - 1, bp - 1)) }

    var sequenceString: String { String(decoding: sequence, as: UTF8.self) }

    /// The stretch of sequence from `start` (1-based) of `count` bases.
    func bases(from start: Int, count: Int) -> String {
        let a = max(0, start - 1), b = min(length, a + count)
        return String(decoding: sequence[a..<b], as: UTF8.self)
    }
}

// MARK: - Colors

/// Only the parts that earn their place are colored; the rest of the circle is
/// neutral. GFP green, the arabinose switch violet, bla amber, the origin blue.
func featureColor(_ key: String) -> SIMD3<Float> {
    switch key {
    case "gfp":    return SIMD3(0.20, 0.74, 0.35)
    case "switch": return SIMD3(0.60, 0.40, 0.88)
    case "bla":    return SIMD3(0.90, 0.58, 0.20)
    case "ori":    return SIMD3(0.25, 0.55, 0.90)
    default:       return SIMD3(0.42, 0.46, 0.50)
    }
}

func ballRadius(_ element: String) -> Float {
    switch element {
    case "C": return 0.34
    case "N": return 0.33
    case "O": return 0.32
    case "P": return 0.40
    default: return 0.22     // H
    }
}

/// CPK colors, as in steps 6 to 8.
func elementColor(_ element: String) -> SIMD3<Float> {
    switch element {
    case "C": return SIMD3(0.28, 0.29, 0.31)
    case "N": return SIMD3(0.20, 0.36, 0.92)
    case "O": return SIMD3(0.86, 0.16, 0.14)
    case "P": return SIMD3(1.0, 0.55, 0.10)
    default: return SIMD3(0.93, 0.93, 0.93)
    }
}
