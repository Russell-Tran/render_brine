// Reads Resources/scene.json (built by Tools/build_scene.py) and turns it into
// spheres for the ray tracer, plus the motion for both GIFs.

import Foundation
import simd

struct Bead {
    var position: SIMD3<Float>
    var radius: Float
    var color: SIMD3<Float>
    var part: String          // head, tail, lps, pg, dna, calcium
    var group: String         // outer_membrane, peptidoglycan, inner_membrane, plasmid, calcium
    var evidence: String      // measured, simulated, model
}

struct Layers {
    var omCenter: Float
    var pgCenter: Float
    var imCenter: Float
    var bilayer: Float
    var periplasm: Float
    var omToPG: Float
    var omOuterFace: Float
}

struct Plasmid {
    var bp: Int
    var sigma: Float
    var lk0: Float
    var deltaLk: Float
    var targetWrithe: Float
    var measuredWrithe: Float
    var superRadius: Float
    var extent: Float
    var contourNm: Float
}

struct Scene {
    var patchX: Float
    var patchZ: Float
    var layers: Layers
    var plasmid: Plasmid
    var beads: [Bead]
    var measurements: [String: String]

    /// Indices of the beads belonging to a group, worked out once.
    func indices(of group: String) -> [Int] {
        beads.enumerated().compactMap { $0.element.group == group ? $0.offset : nil }
    }
}

enum SceneError: Error, CustomStringConvertible {
    case badFile(String)
    var description: String {
        switch self {
        case .badFile(let d): return "couldn't read the scene: \(d)"
        }
    }
}

func loadScene(from url: URL) throws -> Scene {
    let data = try Data(contentsOf: url)
    guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
          let rawBeads = root["beads"] as? [[String: Any]],
          let rawLayers = root["layers"] as? [String: NSNumber],
          let rawPlasmid = root["plasmid"] as? [String: NSNumber]
    else { throw SceneError.badFile("missing fields") }

    var beads: [Bead] = []
    beads.reserveCapacity(rawBeads.count)
    for b in rawBeads {
        guard let p = b["p"] as? [NSNumber], p.count == 3,
              let r = b["r"] as? NSNumber, let c = b["c"] as? [NSNumber], c.count == 3
        else { throw SceneError.badFile("bad bead") }
        beads.append(Bead(position: SIMD3(p[0].floatValue, p[1].floatValue, p[2].floatValue),
                          radius: r.floatValue,
                          color: SIMD3(c[0].floatValue, c[1].floatValue, c[2].floatValue),
                          part: b["part"] as? String ?? "",
                          group: b["group"] as? String ?? "",
                          evidence: b["evidence"] as? String ?? "model"))
    }
    func f(_ d: [String: NSNumber], _ k: String) -> Float { d[k]?.floatValue ?? 0 }
    let layers = Layers(omCenter: f(rawLayers, "om_center"), pgCenter: f(rawLayers, "pg_center"),
                        imCenter: f(rawLayers, "im_center"), bilayer: f(rawLayers, "bilayer"),
                        periplasm: f(rawLayers, "periplasm"), omToPG: f(rawLayers, "om_to_pg"),
                        omOuterFace: f(rawLayers, "om_outer_face"))
    let plasmid = Plasmid(bp: Int(f(rawPlasmid, "bp")), sigma: f(rawPlasmid, "sigma"),
                          lk0: f(rawPlasmid, "lk0"), deltaLk: f(rawPlasmid, "delta_lk"),
                          targetWrithe: f(rawPlasmid, "target_writhe"),
                          measuredWrithe: f(rawPlasmid, "measured_writhe"),
                          superRadius: f(rawPlasmid, "super_radius"), extent: f(rawPlasmid, "extent"),
                          contourNm: f(rawPlasmid, "contour_nm"))
    var measurements: [String: String] = [:]
    for (k, v) in (root["measurements"] as? [String: Any]) ?? [:] {
        measurements[k] = String(describing: v)
    }
    return Scene(patchX: (root["patch_x"] as? NSNumber)?.floatValue ?? 2000,
                 patchZ: (root["patch_z"] as? NSNumber)?.floatValue ?? 240,
                 layers: layers, plasmid: plasmid, beads: beads, measurements: measurements)
}

// MARK: - Turning the scene into spheres

func gpuShapes(_ scene: Scene) -> [GPUShape] {
    scene.beads.map { GPUShape.sphere(center: $0.position, radius: $0.radius, color: $0.color) }
}

/// Smooth 0→1 easing, so motion starts and stops gently. Same as every step
/// since 7a.
func smoothstep01(_ t: Float) -> Float {
    let x = min(max(t, 0), 1)
    return x * x * (3 - 2 * x)
}

// MARK: - GIF 1: the approach

/// Where the plasmid sits at time `t` through the approach, as an offset from
/// its resting position in the JSON.
///
/// It only ever TRANSLATES. That is a deliberate constraint, not an oversight:
/// ambient occlusion here is a function of the surface normal and the local
/// geometry, so a body that translates carries its shading with it unchanged,
/// while one that tumbles would have its normals turn and its shading crawl
/// from frame to frame. See the note at the top of Render.swift.
func plasmidOffset(_ scene: Scene, t: Float, reach: Float = 305) -> SIMD3<Float> {
    let start = SIMD3<Float>(0, scene.layers.omOuterFace + reach + 330, 0)
    // It stops with its nearest bead pressed against the outer face, which is
    // where the honest part of the story ends.
    let end = SIMD3<Float>(0, scene.layers.omOuterFace + reach + 4, 0)
    let s = smoothstep01(t)
    return start + (end - start) * s
}

/// Calcium crowds in as the plasmid closes: the ions around the DNA travel
/// with it, and those on the membrane drift very slightly toward the contact.
func approachShapes(_ scene: Scene, base: [GPUShape], dnaIndices: [Int],
                    ionIndices: [Int], ionAttachedToDNA: [Bool], t: Float, reach: Float) -> [GPUShape] {
    var out = base
    let offset = plasmidOffset(scene, t: t, reach: reach)
    for i in dnaIndices {
        out[i].a = SIMD4(scene.beads[i].position + offset, out[i].a.w)
    }
    for (k, i) in ionIndices.enumerated() where ionAttachedToDNA[k] {
        out[i].a = SIMD4(scene.beads[i].position + offset, out[i].a.w)
    }
    return out
}

/// How far the plasmid's lowest bead sits below its own centre, so the caption
/// can report a real gap rather than one worked out from a stale constant.
func plasmidReach(_ scene: Scene, dnaIndices: [Int]) -> Float {
    var lowest: Float = .greatestFiniteMagnitude
    for i in dnaIndices { lowest = min(lowest, scene.beads[i].position.y - scene.beads[i].radius) }
    return -lowest
}

// MARK: - Which ions ride with the DNA

/// An ion counts as belonging to the plasmid if it started within a shell of
/// some DNA bead. The builder placed 60% of them that way; this recovers the
/// split from the geometry rather than trusting a count.
func ionsOnDNA(_ scene: Scene, dnaIndices: [Int], ionIndices: [Int]) -> [Bool] {
    let dna = dnaIndices.map { scene.beads[$0].position }
    return ionIndices.map { i in
        let p = scene.beads[i].position
        for d in dna where simd_distance(p, d) < 24 { return true }
        return false
    }
}
