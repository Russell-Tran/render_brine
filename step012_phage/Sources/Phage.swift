// Reads Resources/scene.json (built by Tools/build_phage.py) and turns it into
// spheres for the ray tracer, plus the motion of the firing.
//
// What moves, and on what evidence
// --------------------------------
// The 96 wedge chains carry BOTH their positions: where each atom sits in
// 5IV5 (hexagonal, before firing) and where the same atom — matched by
// residue number and atom name — sits in 5IV7 (star-shaped, after). Those two
// ends are measured; only the path between them is interpolated, which is the
// same bargain every reaction render in this project has made since step 6.
//
// The hub, the needle and the short tail fibres exist only in the pre-firing
// structure, because after firing they are gone from the baseplate. They are
// drawn before the flip and driven downward through the envelope, and that
// motion is labelled `model`: it is what the machine must do, not what anyone
// has photographed.

import Foundation
import simd

struct PhageShape {
    var start: SIMD3<Float>       // where it sits before firing
    var end: SIMD3<Float>?        // where 5IV7 puts it, if both states resolved it
    var radius: Float
    var part: String              // wedge, needle, tube, fibre, lps, lipid, lipid_tail, pg
    var group: String             // the entity or envelope layer
    var evidence: Evidence
}

struct Baseplate {
    var preRadius: Float
    var postRadius: Float
    var preHeight: Float
    var postHeight: Float
    var wedgeChains: Int
    var wedgeAtomsKept: Int
    var wedgeAtomsDropped: Int
}

struct Sheath {
    var preLength: Float
    var postLength: Float
    var preRadius: Float
    var postRadius: Float
    /// How far the sheath shortens, as a fraction — what drives the tube down.
    var contraction: Float { 1 - postLength / preLength }
}

struct EnvelopeLayers {
    var om: Float
    var pg: Float
    var im: Float
    var top: Float
    var bilayer: Float
    var periplasm: Float
}

struct PhageScene {
    var shapes: [PhageShape]
    var baseplate: Baseplate
    var sheath: Sheath
    var layers: EnvelopeLayers
    var reach: Float              // z the tube must pass to cross the inner membrane
    var counts: [String: Int]
    var commonEntities: [String]
    var preOnlyEntities: [String]
    /// The lowest point any short tail fibre reaches before firing, measured
    /// from the structure so the fibres can be set down on the membrane.
    var fibreLow: Float

    func indices(part: String) -> [Int] {
        shapes.enumerated().compactMap { $0.element.part == part ? $0.offset : nil }
    }
}

enum PhageError: Error, CustomStringConvertible {
    case badFile(String)
    var description: String {
        switch self {
        case .badFile(let d): return "couldn't read the scene: \(d)"
        }
    }
}

private func vec(_ a: [Any]) -> SIMD3<Float> {
    SIMD3(((a[0] as? NSNumber)?.floatValue ?? 0),
          ((a[1] as? NSNumber)?.floatValue ?? 0),
          ((a[2] as? NSNumber)?.floatValue ?? 0))
}

func loadScene(from url: URL) throws -> PhageScene {
    let data = try Data(contentsOf: url)
    guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
          let raw = root["shapes"] as? [[String: Any]] else { throw PhageError.badFile("no shapes") }

    var shapes: [PhageShape] = []
    shapes.reserveCapacity(raw.count)
    for s in raw {
        guard let p = s["p"] as? [Any], let r = (s["r"] as? NSNumber)?.floatValue else {
            throw PhageError.badFile("bad shape")
        }
        let ev: Evidence
        switch s["ev"] as? String ?? "model" {
        case "measured": ev = .measured
        case "simulated": ev = .simulated
        default: ev = .model
        }
        shapes.append(PhageShape(start: vec(p),
                                 end: (s["q"] as? [Any]).map(vec),
                                 radius: r,
                                 part: s["part"] as? String ?? "?",
                                 group: s["group"] as? String ?? "?",
                                 evidence: ev))
    }

    func f(_ d: [String: Any]?, _ k: String) -> Float { ((d?[k] as? NSNumber)?.floatValue ?? 0) }
    func i(_ d: [String: Any]?, _ k: String) -> Int { ((d?[k] as? NSNumber)?.intValue ?? 0) }
    let bp = root["baseplate"] as? [String: Any]
    let sh = root["sheath"] as? [String: Any]
    let ly = root["layers"] as? [String: Any]
    let ents = root["entities"] as? [String: Any]

    return PhageScene(
        shapes: shapes,
        baseplate: Baseplate(preRadius: f(bp, "preRadius"), postRadius: f(bp, "postRadius"),
                             preHeight: f(bp, "preHeight"), postHeight: f(bp, "postHeight"),
                             wedgeChains: i(bp, "wedgeChains"), wedgeAtomsKept: i(bp, "wedgeAtomsKept"),
                             wedgeAtomsDropped: i(bp, "wedgeAtomsDropped")),
        sheath: Sheath(preLength: f(sh, "preLength"), postLength: f(sh, "postLength"),
                       preRadius: f(sh, "preRadius"), postRadius: f(sh, "postRadius")),
        layers: EnvelopeLayers(om: f(ly, "om"), pg: f(ly, "pg"), im: f(ly, "im"),
                               top: f(ly, "top"), bilayer: f(ly, "bilayer"), periplasm: f(ly, "periplasm")),
        reach: (root["reach"] as? NSNumber)?.floatValue ?? 0,
        counts: (root["counts"] as? [String: NSNumber])?.mapValues { $0.intValue } ?? [:],
        commonEntities: (ents?["common"] as? [String]) ?? [],
        preOnlyEntities: (ents?["preOnly"] as? [String]) ?? [],
        fibreLow: shapes.lazy.filter { $0.part == "fibre" }.map { $0.start.z }.min() ?? 0)
}

// MARK: - Colours

/// Coloured by what a part is, not by element. At this size CPK would be an
/// undifferentiated grey-and-red mush, as step 10 found.
func partColor(_ part: String, group: String) -> SIMD3<Float> {
    switch part {
    case "wedge":
        // The six wedge proteins get related blues so the star shape reads as
        // one object while its components stay separable.
        switch group {
        case "Baseplate wedge protein gp6": return SIMD3(0.36, 0.47, 0.72)
        case "Baseplate wedge protein gp7": return SIMD3(0.44, 0.56, 0.80)
        case "Baseplate wedge protein gp8": return SIMD3(0.30, 0.41, 0.66)
        case "Baseplate wedge protein gp9": return SIMD3(0.52, 0.63, 0.84)
        case "Baseplate wedge protein gp10": return SIMD3(0.40, 0.52, 0.76)
        case "Baseplate wedge protein gp11": return SIMD3(0.48, 0.60, 0.82)
        case "Baseplate wedge protein gp25": return SIMD3(0.33, 0.44, 0.70)
        default: return SIMD3(0.42, 0.54, 0.78)
        }
    case "needle":
        // The needle and hub in warm violet, so the thing that goes through the
        // wall is the thing the eye follows.
        return group.contains("gp5") ? SIMD3(0.82, 0.40, 0.66) : SIMD3(0.70, 0.42, 0.74)
    case "tube": return SIMD3(0.58, 0.46, 0.72)
    case "fibre": return SIMD3(0.86, 0.52, 0.42)
    case "lps": return SIMD3(0.86, 0.68, 0.30)
    case "lipid": return SIMD3(0.80, 0.62, 0.28)
    case "lps_tail", "lipid_tail": return SIMD3(0.62, 0.50, 0.26)
    case "pg": return SIMD3(0.42, 0.66, 0.40)
    default: return SIMD3(0.6, 0.6, 0.6)
    }
}

// MARK: - The firing

/// Smooth 0→1 easing, as every step since 7a.
func smoothstep01(_ t: Float) -> Float {
    let x = min(max(t, 0), 1)
    return x * x * (3 - 2 * x)
}

/// The phases of one loop, in seconds. The flip and the contraction are the
/// subject; everything else is there to frame them.
struct Timing {
    var settle: Float = 2.0       // loaded, sitting above the cell
    var flip: Float = 5.0         // hexagon -> star, the measured pair
    var drive: Float = 4.5        // hub and needle driven down through the wall
    var hold: Float = 2.5         // through, DNA starting down the tube
    var reset: Float = 4.0        // back to loaded, so the loop closes
    var total: Float { settle + flip + drive + hold + reset }

    /// 0 before the flip starts, 1 when it is complete.
    func flipProgress(_ t: Float) -> Float {
        smoothstep01((t - settle) / flip)
    }
    /// How far the hub has travelled, 0 to 1.
    func driveProgress(_ t: Float) -> Float {
        smoothstep01((t - settle - flip) / drive)
    }
    /// The reset, which runs everything backwards over a shorter time so the
    /// loop closes. It reads as the next phage arriving, not as a rewind,
    /// because the caption says so and the camera pulls back through it.
    func resetProgress(_ t: Float) -> Float {
        smoothstep01((t - settle - flip - drive - hold) / reset)
    }
}

/// Where every shape sits at time `t`, and how big it is.
///
/// Three pre-firing parts move in three different ways, because they do three
/// different things. Driving them all downward together — which is what a
/// single "hub" group would do — would push the short tail fibres through the
/// cell wall, and gripping the outside is their whole job.
func poseScene(_ scene: PhageScene, t: Float, timing: Timing) -> [GPUShape] {
    let flip = timing.flipProgress(t)
    let drive = timing.driveProgress(t)
    let reset = timing.resetProgress(t)

    // During the reset the flip unwinds and the parts come back, so the last
    // frame equals the first.
    let f: Float = reset > 0 ? flip * (1 - reset) : flip
    let d: Float = reset > 0 ? drive * (1 - reset) : drive

    // Far enough for the needle's tip to clear the inner membrane, plus its
    // own length so the whole thing is inside rather than straddling.
    let travel: Float = scene.layers.top - scene.reach + 180
    // The fibres come to rest ON the membrane's outer face. The drop is
    // measured from the lowest fibre atom, so they touch down rather than
    // sinking through — gripping the outside is the whole job of a short
    // tail fibre, and driving it into the cell would be plain wrong.
    let fibreDrop: Float = max(0, scene.fibreLow - (scene.layers.top + 4))

    var out: [GPUShape] = []
    out.reserveCapacity(scene.shapes.count)
    for s in scene.shapes {
        var p = s.start
        var radius = s.radius
        switch s.part {
        case "wedge":
            if let e = s.end { p = s.start + (e - s.start) * f }
        case "needle":
            // Straight down and through. It does not fade: once inside, the
            // needle is simply in the cytoplasm, and the membranes in front of
            // it do the hiding. An earlier version dimmed it on a timer and a
            // test caught that doing so while it was still in open view.
            p.z -= travel * d
        case "tube":
            // The sheath contracts and drives the tube down; the tube is what
            // the DNA then travels through. Left in place it reads as a lump
            // floating in the sky, because the sheath itself is out of frame.
            p.z -= travel * d
        case "fibre":
            // Down onto the membrane and outward, which is how a short tail
            // fibre grips: it swings from folded up to splayed against the LPS.
            let spread: Float = 1 + 0.18 * f
            p.x *= spread
            p.y *= spread
            p.z -= fibreDrop * f
        default:
            break     // the envelope holds still
        }
        out.append(GPUShape.sphere(center: p, radius: radius,
                                   color: partColor(s.part, group: s.group)))
    }
    return out
}

/// The firing camera. It looks at the baseplate from slightly above the cell
/// surface, far enough out to hold the whole star once it spreads.
///
/// Scale arithmetic, the same as every step since the plasmid: the baseplate
/// is 60.9 nm across after the flip, so framing ~90 nm over `width` pixels
/// puts one carbon atom (3.4 Å across) at about 3.6 px at 960 wide — resolved
/// enough for space-filling to mean something. The whole-virion view would be
/// 250 nm and 1.7 px an atom, which is the granular look and honest at that
/// distance; this render stays close.
func firingCamera(_ scene: PhageScene, t: Float, timing: Timing = Timing()) -> Camera {
    let mid = (scene.layers.top + 260) * 0.5
    // A slow drift in, then back out through the reset, so the loop closes.
    let reset = timing.resetProgress(t)
    let closeness = smoothstep01(t / (timing.settle + timing.flip)) * (1 - reset)
    let dist: Float = 1750 - 210 * closeness
    let pitch: Float = radians(16)
    return Camera(origin: SIMD3(0, -dist * cos(pitch), mid + dist * sin(pitch)),
                  target: SIMD3(0, 0, mid - 40),
                  fov: 30)
}

/// The evidence level and note for the moment at `t` — what the picture is
/// claiming right now.
func evidenceAt(_ t: Float, timing: Timing) -> (Evidence, String) {
    if t < timing.settle + timing.flip * 0.05 {
        return (.measured, "PDB 5IV5, the baseplate before firing")
    }
    if t < timing.settle + timing.flip {
        return (.measured, "morphing between 5IV5 and 5IV7, both solved")
    }
    if t < timing.settle + timing.flip + timing.drive * 0.15 {
        return (.measured, "PDB 5IV7, the baseplate after firing")
    }
    return (.model, "the needle's path through the wall is inferred, never solved")
}

// MARK: - Stills

import CoreGraphics
import ImageIO
import Metal
import UniformTypeIdentifiers

/// Writes the frame buffer out as a JPEG, for the results page.
func writeJPEG(_ buffer: MTLBuffer, layout: FrameLayout, to path: String) throws {
    let bytes = layout.width * layout.height * 4
    let data = Data(bytes: buffer.contents(), count: bytes)
    guard let provider = CGDataProvider(data: data as CFData),
          let image = CGImage(width: layout.width, height: layout.height, bitsPerComponent: 8,
                              bitsPerPixel: 32, bytesPerRow: layout.width * 4,
                              space: CGColorSpace(name: CGColorSpace.sRGB)!,
                              bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                              provider: provider, decode: nil, shouldInterpolate: false,
                              intent: .defaultIntent) else {
        throw RenderError.gpu("could not make an image")
    }
    let url = URL(fileURLWithPath: path)
    try? FileManager.default.createDirectory(at: url.deletingLastPathComponent(),
                                             withIntermediateDirectories: true)
    guard let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.jpeg.identifier as CFString, 1, nil) else {
        throw RenderError.gpu("could not create \(path)")
    }
    CGImageDestinationAddImage(dest, image, [kCGImageDestinationLossyCompressionQuality as String: 0.9] as CFDictionary)
    guard CGImageDestinationFinalize(dest) else { throw RenderError.gpu("could not write \(path)") }
}
