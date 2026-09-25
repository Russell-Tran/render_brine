// The scene: a DNA target with two hybridisation probes, one already bound
// and one arriving, built by Tools/build_scene.py.
//
// The animation is one probe approaching along the target until it lands in
// its place. Everything the viewer sees about brightness follows from the
// donor-acceptor separation that approach produces - nothing is keyframed.

import Foundation
import simd

func radians(_ degrees: Float) -> Float { degrees * .pi / 180 }

struct SceneAtom {
    var element: String
    var home: SIMD3<Float>      // where it sits when the probe is bound
    var part: String            // "target", "probe1", "probe2", "none"
    var basePair: Int
}

struct Scene {
    var targetSequence: String
    var atoms: [SceneAtom]
    var bonds: [(a: Int, b: Int, order: Int)]
    var donorHome: SIMD3<Float>
    var acceptorHome: SIMD3<Float>
    var boundSeparation: Float  // ångströms, with both probes in place
    var probe2Range: (Int, Int)
    var gapNucleotides: Int
}

enum SceneError: Error, CustomStringConvertible {
    case badFile(String)
    var description: String {
        switch self {
        case .badFile(let d): return "couldn't read the scene: \(d)"
        }
    }
}

private func vector(_ a: [NSNumber]) -> SIMD3<Float> {
    SIMD3(a[0].floatValue, a[1].floatValue, a[2].floatValue)
}

func loadScene(from url: URL) throws -> Scene {
    let data = try Data(contentsOf: url)
    guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
          let rawAtoms = root["atoms"] as? [[String: Any]],
          let rawBonds = root["bonds"] as? [[Int]],
          let donor = root["donor_home"] as? [NSNumber],
          let acceptor = root["acceptor_home"] as? [NSNumber]
    else { throw SceneError.badFile("missing fields") }

    var atoms: [SceneAtom] = []
    atoms.reserveCapacity(rawAtoms.count)
    for a in rawAtoms {
        guard let el = a["el"] as? String, let p = a["pos"] as? [NSNumber], p.count == 3
        else { throw SceneError.badFile("bad atom") }
        atoms.append(SceneAtom(element: el, home: vector(p),
                               part: a["part"] as? String ?? "none",
                               basePair: a["bp"] as? Int ?? 0))
    }
    let bonds = rawBonds.map { (a: $0[0], b: $0[1], order: $0[2]) }
    let p2 = (root["probe2"] as? [Int]) ?? [0, 0]
    return Scene(targetSequence: root["target_sequence"] as? String ?? "",
                 atoms: atoms, bonds: bonds,
                 donorHome: vector(donor), acceptorHome: vector(acceptor),
                 boundSeparation: (root["bound_distance_A"] as? NSNumber)?.floatValue ?? 0,
                 probe2Range: (p2[0], p2[1]),
                 gapNucleotides: root["gap_nt"] as? Int ?? 0)
}

// MARK: - The approach

/// How far along its approach the arriving probe is, at time `t` in a loop of
/// `duration` seconds. 0 is far out in solution, 1 is landed.
///
/// The probe spends most of the loop bound, because that is the state the
/// instrument reads; the approach is the short part. Eased at both ends so
/// nothing jerks, and returning exactly to 0 so the loop closes.
func approachProgress(t: Double, duration: Double) -> Float {
    let u = t / duration
    let arrive = 0.10, settle = 0.45, leave = 0.88
    var raw: Double
    if u < arrive {
        raw = 0
    } else if u < settle {
        raw = (u - arrive) / (settle - arrive)
    } else if u < leave {
        raw = 1
    } else {
        raw = 1 - (u - leave) / (1 - leave)
    }
    let x = min(max(raw, 0), 1)
    return Float(x * x * (3 - 2 * x))   // smoothstep
}

/// Where the arriving probe sits at a given progress.
///
/// It comes in from ABOVE rather than from along the helix axis. Sliding it in
/// end-on would carry it out of frame - the target is already 116 Å long and
/// fills the view - and descending out of solution is closer to what actually
/// happens anyway. A small sideways component keeps it from sitting directly
/// on top of its own landing site on the way down.
/// How far out it starts is set by the physics, not by taste. R0 for this pair
/// is 56 Å, so transfer is still under 6% at about 90 Å of separation and only
/// reaches half at 56. For the curve to sweep its whole range on screen the
/// probe has to begin roughly 90 Å from the donor - which is most of the
/// length of the target it is landing on. That is worth noticing rather than
/// hiding: Förster transfer reaches about as far as these molecules are big,
/// which is exactly why it works as a molecular ruler.
func probeOffset(progress s: Float) -> SIMD3<Float> {
    let away: Float = 1 - s
    let above: Float = away * 85.0         // ångströms up, out of solution
    let alongAxis: Float = away * 60.0     // drifting in along the target
    let towardCamera: Float = away * 16.0
    return SIMD3(alongAxis, above, towardCamera)
}

/// Every atom's position at this progress, and where the two dyes are.
struct Placed {
    var positions: [SIMD3<Float>]
    var donor: SIMD3<Float>
    var acceptor: SIMD3<Float>
    var separation: Float
}

func place(_ scene: Scene, progress s: Float) -> Placed {
    let offset = probeOffset(progress: s)
    var out = [SIMD3<Float>](repeating: .zero, count: scene.atoms.count)
    for i in 0..<scene.atoms.count {
        let a = scene.atoms[i]
        out[i] = a.part == "probe2" ? a.home + offset : a.home
    }
    let acceptor = scene.acceptorHome + offset
    let separation = simd_distance(scene.donorHome, acceptor)
    return Placed(positions: out, donor: scene.donorHome, acceptor: acceptor,
                  separation: separation)
}

// MARK: - Drawing

/// Van der Waals radii, Bondi (1964), as steps 9 and 9a used.
func vanDerWaalsRadius(_ element: String) -> Float {
    switch element {
    case "C": return 1.70
    case "N": return 1.55
    case "O": return 1.52
    case "P": return 1.80
    case "S": return 1.80
    default: return 1.20
    }
}

/// The target is grey; the two probes are tinted so they read as separate
/// molecules. The dyes get their computed emission colours.
func partColour(_ part: String) -> SIMD3<Float> {
    switch part {
    case "target": return SIMD3(0.42, 0.45, 0.50)
    case "probe1": return SIMD3(0.30, 0.52, 0.46)
    case "probe2": return SIMD3(0.52, 0.36, 0.42)
    default: return SIMD3(0.35, 0.37, 0.40)
    }
}

/// The dyes are drawn larger than an atom because they stand for a whole dye
/// molecule on a linker, not a single atom - the same licence steps 9 and 11
/// took for GFP's chromophore and for arabinose.
let dyeRadius: Float = 4.4

func sceneGeometry(_ scene: Scene, placed: Placed, state: DyeState)
    -> (spheres: [GPUSphere], cylinders: [GPUCylinder]) {
    var spheres: [GPUSphere] = []
    spheres.reserveCapacity(scene.atoms.count + 2)
    for i in 0..<scene.atoms.count {
        let a = scene.atoms[i]
        guard a.part != "none" else { continue }
        let p = placed.positions[i]
        let c = partColour(a.part)
        spheres.append(GPUSphere(centerRadius: SIMD4(p.x, p.y, p.z, vanDerWaalsRadius(a.element)),
                                 color: SIMD4(c.x, c.y, c.z, 1), glow: .zero))
    }

    // The two dyes, each carrying its own brightness. This is where the law
    // reaches the picture: the emission hues come from the wavelengths through
    // the CIE functions, and the amounts come from the separation.
    //
    // Each dye keeps a dim BODY of its own colour underneath the emission. A
    // dye that is transferring all its energy away really is dark, but drawn
    // as pure black it reads as a hole punched in the picture rather than as a
    // molecule sitting quiet. The body is 34% and the emission rides on top.
    let body: Float = 0.34
    let dLit: Float = body + (1 - body) * state.donorBrightness
    let aLit: Float = body + (1 - body) * state.acceptorBrightness
    let dc = donorColour() * dLit
    let ac = acceptorColour() * aLit
    let d = placed.donor, acc = placed.acceptor
    spheres.append(GPUSphere(centerRadius: SIMD4(d.x, d.y, d.z, dyeRadius),
                             color: SIMD4(dc.x, dc.y, dc.z, 1), glow: .zero))
    spheres.append(GPUSphere(centerRadius: SIMD4(acc.x, acc.y, acc.z, dyeRadius),
                             color: SIMD4(ac.x, ac.y, ac.z, 1), glow: .zero))

    var cylinders: [GPUCylinder] = []
    let grey = SIMD4<Float>(0.52, 0.54, 0.58, 1)
    for b in scene.bonds {
        let pa = scene.atoms[b.a], pb = scene.atoms[b.b]
        guard pa.part != "none", pb.part != "none" else { continue }
        // A bond between the arriving probe and anything else does not exist
        // until it lands; skip those while it is away.
        if (pa.part == "probe2") != (pb.part == "probe2") { continue }
        let p = placed.positions[b.a], q = placed.positions[b.b]
        cylinders.append(GPUCylinder(aRadius: SIMD4(p.x, p.y, p.z, bondRadius),
                                     b: SIMD4(q.x, q.y, q.z, 0), color: grey))
    }
    return (spheres, cylinders)
}
