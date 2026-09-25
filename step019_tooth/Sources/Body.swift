// The three-dimensional body, built out of step 13's two primitives.
//
// The section in Tooth.swift is a profile in the (x, y) half-plane. This file
// revolves it about the tooth's long axis by stacking oblate ellipsoids on the
// axis, one stack per tissue, and lets the priority rule in the kernel do the
// nesting — the same trick step 15 used for the coconut's six shells, except
// that here the shells' radii come from a function rather than from one number
// each.
//
// TWO THINGS THE REVOLUTION COSTS, both stated on the page:
//
//   * A molar has four or five cusps. The section has two, and revolving two
//     cusps gives a RING. The body is therefore a ring-cusped tooth with a
//     central pit, which is what a solid of revolution of this section is, and
//     not what a lower first molar is.
//   * The fissure is drawn as a material rather than cut out as a void. On the
//     cut face that is exactly right — a sectioned tooth shows the fissure as a
//     dark slot — and it costs the kernel nothing.
//
// WHY A STACK AND NOT ONE ELLIPSOID EACH. Step 15 could use one ellipsoid per
// layer because a coconut is an ellipsoid. A tooth is not: the crown bulges to a
// height of contour and the root tapers to an apex, and the layers inside it are
// offsets of that. The cost of a stack is intervals: step 13's ray holds forty,
// and a ray down the axis of this tooth crosses five stacks at once, so each
// stack's overlap is chosen to keep the total under that. `stackOverlap` is
// where that budget is spent, and a test counts the worst ray in the frame.

import Foundation
import simd

// MARK: - materials, and their order IS their priority

enum Material: Int, CaseIterable {
    case bone = 0
    case pdl                 // periodontal ligament
    case cementum
    case enamel
    case dej
    case dentin
    case pulp
    case fissure             // the central pit, drawn as a dark slot

    var name: String {
        switch self {
        case .bone: return "alveolar bone"
        case .pdl: return "periodontal ligament"
        case .cementum: return "cementum"
        case .enamel: return "enamel"
        case .dej: return "dentino-enamel junction"
        case .dentin: return "dentin"
        case .pulp: return "pulp"
        case .fissure: return "central fissure"
        }
    }

    /// Which lattice tissue this material's cut face should be coloured by.
    /// `nil` means the stress field does not apply there.
    var tissue: ToothTissue? {
        switch self {
        case .bone: return .bone
        case .pdl: return .pdl
        case .cementum: return .cementum
        case .enamel: return .enamel
        case .dej: return .dej
        case .dentin: return .dentin
        case .pulp: return nil
        case .fissure: return nil
        }
    }
}

let materialCount = Material.allCases.count

/// Flat colours, in the series' key: mineral is pale and warm, living tissue is
/// red, and the two things that are not tooth — bone and ligament — are held
/// back so the tooth reads first.
let toothAlbedo: [SIMD3<Float>] = [
    SIMD3(0.775, 0.745, 0.660),   // bone
    SIMD3(0.800, 0.545, 0.500),   // pdl
    SIMD3(0.790, 0.725, 0.595),   // cementum
    SIMD3(0.955, 0.950, 0.925),   // enamel
    SIMD3(0.895, 0.820, 0.650),   // dej
    SIMD3(0.885, 0.775, 0.550),   // dentin
    SIMD3(0.700, 0.360, 0.375),   // pulp
    SIMD3(0.400, 0.372, 0.345),   // fissure — the far wall of the groove, in shadow
]

// MARK: - packing a primitive with one of THIS step's materials
//
// The packing belongs to step 13 and is not rewritten: build the primitive step
// 13's way and overwrite the one field that says which material it is, exactly
// as step 15 did.

extension GPUPrim {
    static func ellipsoid(centre: SIMD3<Float>, m: simd_float3x3, material: Material) -> GPUPrim {
        var p: GPUPrim = GPUPrim.ellipsoid(centre: centre, m: m, tissue: .body)
        p.r2.w = Float(material.rawValue)
        return p
    }

    /// A disc of revolution: radius `r` about the y axis, half-height `b`.
    static func disc(y: Float, radius r: Float, half b: Float, material: Material) -> GPUPrim {
        let m = simd_float3x3(diagonal: SIMD3<Float>(r, b, r))
        return GPUPrim.ellipsoid(centre: SIMD3(0, y, 0), m: m, material: material)
    }

    var material: Material { Material(rawValue: Int(r2.w.rounded())) ?? .bone }
}

// MARK: - stacking

/// How many half-heights of overlap a stack carries. A disc of half-height
/// b = f·step dips to r·√(1 − (1/2f)²) midway between two centres; inflating
/// every radius by that same factor puts the error on both sides of the true
/// profile instead of all on one, which halves it. So the worst radial error is
/// about r/(16 f²), and the number of discs a ray meets at once is 2f.
func stackedRadius(_ r: Float, overlap f: Float) -> Float {
    let half: Float = 1 / (2 * f)
    let shrink: Float = (1 - half * half).squareRoot()
    let mid: Float = (1 + shrink) / 2
    return r / mid
}

/// One stack. `radius` is the true profile; the discs are placed every `step`
/// from `from` to `to` inclusive, and their half-heights are clipped at the ends
/// so the stack does not spill past the range it was asked for — which is what
/// keeps enamel out of the root.
func stack(_ material: Material, from lo: Float, to hi: Float, step: Float,
           overlap f: Float, radius: (Float) -> Float) -> [GPUPrim] {
    var out: [GPUPrim] = []
    let b: Float = f * step
    let count: Int = max(Int(((hi - lo) / step).rounded(.up)), 1)
    for i in 0...count {
        let t: Float = Float(i) / Float(count)
        let y: Float = lo + (hi - lo) * t
        let r: Float = radius(y)
        guard r > 1e-4 else { continue }
        let room: Float = min(y - lo, hi - y) + 0.5 * step
        let half: Float = max(min(b, room), 0.25 * step)
        out.append(GPUPrim.disc(y: y, radius: stackedRadius(r, overlap: f), half: half,
                                material: material))
    }
    return out
}

// MARK: - the tooth

/// Where the bone block ends. Wider than the frame, so the only bone boundaries
/// the picture ever shows are the crest and the socket.
let boneBlockRadius: Float = 12.0
let boneBlockBottom: Float = -16.5
let boneCrestDome: Float = 1.2

func toothPrimitives() -> [GPUPrim] {
    var prims: [GPUPrim] = []

    // The jaw. A stack of wide discs whose upper pole is the alveolar crest, so
    // the crest domes gently down and away from the tooth, which is what a
    // healthy ridge does.
    prims += stack(.bone, from: boneBlockBottom, to: alveolarCrestY - boneCrestDome,
                   step: 0.5, overlap: 2.3) { _ in boneBlockRadius }

    prims += stack(.pdl, from: apexY - pdlThickness - 0.05, to: alveolarCrestY,
                   step: 0.22, overlap: 3.0) { pdlOuterRadius(y: $0) }

    prims += stack(.cementum, from: apexY, to: 0, step: 0.22, overlap: 3.6) {
        outerHalfWidth(y: $0)
    }

    prims += stack(.enamel, from: 0, to: crownHeight, step: 0.17, overlap: 4.0) {
        outerHalfWidth(y: $0)
    }

    prims += stack(.dej, from: -0.1, to: dejTop, step: 0.2, overlap: 3.0) { dejRadius(y: $0) }

    prims += stack(.dentin, from: apexY, to: dentinHornTop, step: 0.22, overlap: 3.6) {
        dentinRadius(y: $0)
    }

    prims += stack(.pulp, from: apexY + 0.3, to: pulpRoofY, step: 0.35, overlap: 1.9) {
        pulpRadius(y: $0)
    }

    prims += stack(.fissure, from: fissureFloorY, to: crownHeight + 0.05,
                   step: 0.13, overlap: 3.2) { fissureRadius(y: $0) }

    return prims
}

/// The material the primitives resolve to at a point, by the same priority rule
/// the kernel uses: the highest index that covers the point. The test that
/// compares this with `tissueAt` is what makes "the body is that section
/// revolved" a checked claim rather than a caption.
func materialAt(_ p: SIMD3<Float>, prims: [GPUPrim]) -> Material? {
    var best = -1
    for prim in prims where cpuContains(prim, point: p) {
        let m: Int = prim.tissue
        if m > best { best = m }
    }
    guard best >= 0 else { return nil }
    return Material(rawValue: best)
}

/// What `tissueAt` would call a material — the mapping the comparison test uses.
func tissueOf(_ m: Material?) -> ToothTissue {
    guard let m = m else { return .outside }
    switch m {
    case .bone: return .bone
    case .pdl: return .pdl
    case .cementum: return .cementum
    case .enamel: return .enamel
    case .dej: return .dej
    case .dentin: return .dentin
    case .pulp: return .pulp
    case .fissure: return .outside
    }
}
