// The scene: AraC, the DNA it grips, the sugar that changes its mind, and the
// RNA polymerase waiting for the loop to open.
//
// The DNA is rebuilt from scratch every frame rather than interpolated between
// two end states. That matters: each base pair is stamped from a real crystal
// template as a RIGID BODY, so every bond length and hydrogen bond inside it is
// exactly what the crystallographers measured, at every instant of the
// animation and not only at the ends. Only the path the base pairs sit on is
// modelled, and only the stretch between the two gripped sites bends.
//
// Scale: 290 base pairs is 986 A of DNA. Looped it spans ~330 A, open ~390 A.
// At 1280 px that puts one atom's van der Waals sphere at about 11 px, so for
// the first time in this project the whole mechanism fits in atoms at once,
// with no level-of-detail switching anywhere.

import Foundation
import simd

struct TemplateAtom {
    var element: String
    var strand: Int
    var name: String
    var position: SIMD3<Float>
}

struct BasePairTemplate {
    var atoms: [TemplateAtom]
    var bonds: [(Int, Int)]
    var hbonds: [(Int, Int)]
}

struct PlainAtom {
    var element: String
    var position: SIMD3<Float>
}

struct SwitchData {
    var sequence: String
    var nBP: Int
    var rise: Float
    var twist: Float               // degrees per base pair
    var loopFrom: Int
    var loopTo: Int
    var bpOf: [String: Int]        // half-site centre, as a base-pair index
    var sites: [String: (Int, Int)]
    var spacingBP: Int
    var plus1: Int
    var window: (Int, Int)
    var armComparison: [String: Double]
    var templates: [String: BasePairTemplate]
    var core: [PlainAtom]          // AraC dimerisation / sugar-binding domain (2ARC)
    var dbd: [PlainAtom]           // AraC DNA-binding domain (2K9S)
    var sugar: [PlainAtom]         // L-arabinose, as 2ARC has it
}

enum SwitchError: Error, CustomStringConvertible {
    case badFile(String)
    var description: String {
        switch self {
        case .badFile(let d): return "couldn't read the switch: \(d)"
        }
    }
}

func loadSwitch(from url: URL) throws -> SwitchData {
    let raw = try Data(contentsOf: url)
    guard let root = try JSONSerialization.jsonObject(with: raw) as? [String: Any] else {
        throw SwitchError.badFile("not an object")
    }
    func num(_ k: String) -> Double { (root[k] as? NSNumber)?.doubleValue ?? 0 }
    func vec(_ a: Any?) -> SIMD3<Float> {
        guard let v = a as? [NSNumber], v.count == 3 else { return .zero }
        return SIMD3(v[0].floatValue, v[1].floatValue, v[2].floatValue)
    }
    func plain(_ key: String) -> [PlainAtom] {
        guard let p = root["protein"] as? [String: Any], let list = p[key] as? [[String: Any]] else { return [] }
        return list.map { PlainAtom(element: $0["el"] as? String ?? "C", position: vec($0["pos"])) }
    }
    var templates: [String: BasePairTemplate] = [:]
    for (kind, any) in (root["templates"] as? [String: Any]) ?? [:] {
        guard let t = any as? [String: Any] else { continue }
        let atoms = ((t["atoms"] as? [[String: Any]]) ?? []).map {
            TemplateAtom(element: $0["el"] as? String ?? "C",
                         strand: $0["strand"] as? Int ?? 0,
                         name: $0["name"] as? String ?? "",
                         position: vec($0["pos"]))
        }
        let bonds = ((t["bonds"] as? [[Int]]) ?? []).map { ($0[0], $0[1]) }
        let hbonds = ((t["hbonds"] as? [[Int]]) ?? []).map { ($0[0], $0[1]) }
        templates[kind] = BasePairTemplate(atoms: atoms, bonds: bonds, hbonds: hbonds)
    }
    var sites: [String: (Int, Int)] = [:]
    for (k, v) in (root["sites"] as? [String: [Int]]) ?? [:] where v.count == 2 {
        sites[k] = (v[0], v[1])
    }
    let win = (root["window"] as? [Int]) ?? [0, 0]
    return SwitchData(
        sequence: root["sequence"] as? String ?? "",
        nBP: Int(num("n_bp")), rise: Float(num("rise")), twist: Float(num("twist")),
        loopFrom: Int(num("loop_from")), loopTo: Int(num("loop_to")),
        bpOf: (root["bp_of"] as? [String: Int]) ?? [:],
        sites: sites, spacingBP: Int(num("spacing_bp")), plus1: Int(num("plus1")),
        window: (win[0], win.count > 1 ? win[1] : 0),
        armComparison: (root["arm_comparison"] as? [String: Double]) ?? [:],
        templates: templates, core: plain("core"), dbd: plain("dbd"), sugar: plain("sugar"))
}

// MARK: - the path the DNA takes

/// A planar path: straight lead-in, then the loop region bent through
/// `span` × 2π, then straight lead-out. Arc length never changes, so the same
/// DNA is present however open the loop is.
func centreline(_ d: SwitchData, span: Float) -> (points: [SIMD3<Float>], tangents: [SIMD3<Float>]) {
    let inside = max(d.loopTo - d.loopFrom, 1)
    let kappa: Float = span * 2 * .pi / Float(inside)
    var points: [SIMD3<Float>] = []
    var tangents: [SIMD3<Float>] = []
    points.reserveCapacity(d.nBP)
    tangents.reserveCapacity(d.nBP)
    // The loop rises slightly out of plane as it goes round, so that where it
    // closes the two ends PASS rather than meet. A perfectly planar loop puts
    // the incoming and outgoing duplexes in the same place, which is not
    // something DNA can do — real looped DNA crosses over itself. The rise is
    // 26 A across the loop, just over one duplex width (20 A). A test measures
    // the closest approach and would fail if this were removed.
    let lift: Float = span * 26.0 / (Float(inside) * d.rise)
    var angle: Float = 0
    var p = SIMD3<Float>.zero
    for i in 0..<d.nBP {
        let inLoop = i >= d.loopFrom && i < d.loopTo
        let raw = SIMD3<Float>(cos(angle), sin(angle), inLoop ? lift : 0)
        let t = simd_normalize(raw)
        tangents.append(t)
        points.append(p)
        p += t * d.rise
        if inLoop { angle += kappa }
    }
    // Anchor on araI1, the end of the loop. Without this the whole downstream
    // half — araI2, the promoter, everything RNA polymerase cares about —
    // swings across the frame as the loop opens, because the path's direction
    // is integrated from the far end. Anchored, the promoter stays put and the
    // LOOP is what visibly moves, which is both easier to read and closer to
    // what the DNA actually does.
    let anchor = min(max(d.loopTo, 0), d.nBP - 1)
    let a0 = points[anchor]
    let dir = tangents[anchor]
    let c: Float = dir.x
    let sn: Float = dir.y
    // Rotate by -angle(dir) about z, then translate the anchor to the origin.
    func turn(_ v: SIMD3<Float>) -> SIMD3<Float> {
        let x: Float = v.x * c + v.y * sn
        let y: Float = -v.x * sn + v.y * c
        return SIMD3(x, y, v.z)
    }
    for i in 0..<d.nBP {
        points[i] = turn(points[i] - a0)
        tangents[i] = turn(tangents[i])
    }
    return (points, tangents)
}

/// Parallel-transport frames, so the helix does not spin spuriously where the
/// path bends.
func transportFrames(points: [SIMD3<Float>], tangents: [SIMD3<Float>])
    -> (normals: [SIMD3<Float>], binormals: [SIMD3<Float>]) {
    var normals: [SIMD3<Float>] = []
    var binormals: [SIMD3<Float>] = []
    var n = SIMD3<Float>(0, 0, 1)
    n = simd_normalize(n - tangents[0] * simd_dot(n, tangents[0]))
    for t in tangents {
        var v = n - t * simd_dot(n, t)
        let len = simd_length(v)
        v = len > 1e-9 ? v / len : n
        n = v
        normals.append(v)
        binormals.append(simd_cross(t, v))
    }
    return (normals, binormals)
}

/// The frame at one base pair: where it sits and which way is "out" of the helix.
func siteFrame(_ d: SwitchData, at bp: Int, span: Float)
    -> (at: SIMD3<Float>, out: SIMD3<Float>, along: SIMD3<Float>) {
    let (pts, tan) = centreline(d, span: span)
    let (nor, bin) = transportFrames(points: pts, tangents: tan)
    let i = min(max(bp, 0), d.nBP - 1)
    let ang = radians(d.twist * Float(i))
    let out = nor[i] * cos(ang) + bin[i] * sin(ang)
    return (pts[i], out, tan[i])
}

// MARK: - colours and radii

/// Bondi's van der Waals radii (J. Phys. Chem. 68:441, 1964), as step 9 used.
func vdwRadius(_ element: String) -> Float {
    switch element {
    case "C": return 1.70
    case "N": return 1.55
    case "O": return 1.52
    case "P": return 1.80
    case "S": return 1.80
    case "SE": return 1.90
    default: return 1.60
    }
}

/// Ball-and-stick radii, as steps 6 to 8.
func ballRadius(_ element: String) -> Float {
    switch element {
    case "C": return 0.34
    case "N": return 0.33
    case "O": return 0.32
    case "P": return 0.40
    default: return 0.30
    }
}

let dnaColor = SIMD3<Float>(0.62, 0.66, 0.72)
let dnaBackbone = SIMD3<Float>(0.78, 0.72, 0.52)
let siteColor = SIMD3<Float>(0.55, 0.38, 0.82)       // the three AraC half-sites
let promoterColor = SIMD3<Float>(0.20, 0.62, 0.42)   // pBAD, once it can be read
let coreColor = SIMD3<Float>(0.49, 0.33, 0.75)       // AraC sugar-binding domain
let dbdColor = SIMD3<Float>(0.62, 0.48, 0.85)        // AraC DNA-binding domain
let sugarColor = SIMD3<Float>(0.22, 0.78, 0.36)      // L-arabinose
let rnapColor = SIMD3<Float>(0.27, 0.36, 0.42)       // RNA polymerase, as a shape

// MARK: - building a frame of the scene

/// Everything the animation needs to know at one instant.
struct SwitchState {
    var span: Float           // 1 = loop closed, less = open
    var sugarIn: Float        // 0 = absent, 1 = in the pocket
    var gripI2: Float         // 0 = holding araO2, 1 = holding araI2
    var rnap: Float           // 0 = away, 1 = on the promoter
    var transcribing: Float   // 0 = not, 1 = yes (colours the promoter)
    var caption: Caption
}

/// Places a rigid body by a frame built from `out` and `along`.
func place(_ atoms: [PlainAtom], at origin: SIMD3<Float>, out: SIMD3<Float>, along: SIMD3<Float>,
           scale: Float = 1) -> [SIMD3<Float>] {
    let e1 = simd_normalize(along)
    var e2 = out - e1 * simd_dot(out, e1)
    e2 = simd_length(e2) > 1e-6 ? simd_normalize(e2) : simd_normalize(simd_cross(e1, SIMD3(0, 0, 1)))
    let e3 = simd_cross(e1, e2)
    return atoms.map { a in
        let p = a.position * scale
        return origin + e1 * p.x + e2 * p.y + e3 * p.z
    }
}

/// The whole scene at one instant, as shapes for the ray tracer.
func sceneShapes(_ d: SwitchData, _ s: SwitchState) -> [GPUShape] {
    var shapes: [GPUShape] = []
    shapes.reserveCapacity(26_000)

    let (pts, tan) = centreline(d, span: s.span)
    let (nor, bin) = transportFrames(points: pts, tangents: tan)

    // Which base pairs belong to which landmark, so they can be coloured.
    var mark = [Int](repeating: 0, count: d.nBP)     // 0 plain, 1 half-site, 2 promoter
    let lo = d.window.0
    for (name, span) in d.sites {
        _ = name
        for b in span.0...span.1 {
            let i = b - lo
            if i >= 0 && i < d.nBP { mark[i] = 1 }
        }
    }
    // pBAD's -35 to +1, the stretch RNA polymerase needs.
    for b in (d.plus1 - 45)...(d.plus1 - 1) {
        let i = b - lo
        if i >= 0 && i < d.nBP && mark[i] == 0 { mark[i] = 2 }
    }

    // The DNA: every base pair stamped as a rigid crystal body.
    var atomIndex: [SIMD3<Float>] = []
    var atomElement: [String] = []
    var bpStart: [Int] = []
    for i in 0..<d.nBP {
        let ch = d.sequence[d.sequence.index(d.sequence.startIndex, offsetBy: i)]
        let kind: String
        switch ch {
        case "A": kind = "AT"
        case "T": kind = "TA"
        case "G": kind = "GC"
        default: kind = "CG"
        }
        guard let t = d.templates[kind] else { continue }
        let ang = radians(d.twist * Float(i))
        let e1 = tan[i]
        let e2 = nor[i] * cos(ang) + bin[i] * sin(ang)
        let e3 = simd_cross(e1, e2)
        bpStart.append(atomIndex.count)
        for a in t.atoms {
            let p = pts[i] + e1 * a.position.x + e2 * a.position.y + e3 * a.position.z
            atomIndex.append(p)
            atomElement.append(a.element)
        }
    }
    // Draw the DNA ball-and-stick, coloured by landmark.
    var base = 0
    for i in 0..<bpStart.count {
        let ch = d.sequence[d.sequence.index(d.sequence.startIndex, offsetBy: i)]
        let kind: String
        switch ch {
        case "A": kind = "AT"
        case "T": kind = "TA"
        case "G": kind = "GC"
        default: kind = "CG"
        }
        guard let t = d.templates[kind] else { continue }
        var color = dnaColor
        if mark[i] == 1 { color = siteColor }
        if mark[i] == 2 { color = simd_mix(dnaColor, promoterColor, SIMD3(repeating: s.transcribing)) }
        for (j, a) in t.atoms.enumerated() {
            let isBackbone = a.name == "P" || a.name.hasPrefix("O1P") || a.name.hasPrefix("O2P")
                || a.name.hasPrefix("OP") || a.name.hasSuffix("'")
            let c = isBackbone && mark[i] == 0 ? dnaBackbone : color
            shapes.append(.sphere(center: atomIndex[base + j], radius: ballRadius(a.element), color: c))
        }
        for (u, v) in t.bonds {
            shapes.append(.cylinder(from: atomIndex[base + u], to: atomIndex[base + v],
                                    radius: bondRadius, color: color))
        }
        base += t.atoms.count
    }
    // Join consecutive base pairs along each strand.
    for i in 0..<(bpStart.count - 1) {
        let kindOf: (Int) -> String = { k in
            let ch = d.sequence[d.sequence.index(d.sequence.startIndex, offsetBy: k)]
            switch ch { case "A": return "AT"; case "T": return "TA"; case "G": return "GC"; default: return "CG" }
        }
        guard let t0 = d.templates[kindOf(i)], let t1 = d.templates[kindOf(i + 1)] else { continue }
        func find(_ t: BasePairTemplate, _ start: Int, _ strand: Int, _ name: String) -> Int? {
            for (j, a) in t.atoms.enumerated() where a.strand == strand && a.name == name { return start + j }
            return nil
        }
        for (strand, pair) in [(0, ("O3'", "P")), (1, ("P", "O3'"))] {
            if let u = find(t0, bpStart[i], strand, pair.0), let v = find(t1, bpStart[i + 1], strand, pair.1) {
                let len = simd_distance(atomIndex[u], atomIndex[v])
                if len < 3.0 {
                    shapes.append(.cylinder(from: atomIndex[u], to: atomIndex[v],
                                            radius: bondRadius, color: dnaBackbone))
                }
            }
        }
    }

    // AraC. The two DNA-binding domains sit over the two gripped half-sites;
    // the sugar-binding dimer sits between them. MODEL: the domains are real
    // structures, but no one has solved the whole protein on looped DNA.
    let o2 = siteFrame(d, at: d.bpOf["araO2"] ?? 20, span: s.span)
    let i1 = siteFrame(d, at: d.bpOf["araI1"] ?? 230, span: s.span)
    let i2 = siteFrame(d, at: d.bpOf["araI2"] ?? 255, span: s.span)
    // The far grip slides from araO2 to araI2 as the switch flips.
    let farAt = simd_mix(o2.at, i2.at, SIMD3(repeating: s.gripI2))
    let farOut = simd_normalize(simd_mix(o2.out, i2.out, SIMD3(repeating: s.gripI2)))
    let farAlong = simd_normalize(simd_mix(o2.along, i2.along, SIMD3(repeating: s.gripI2)))

    let dockI1: SIMD3<Float> = i1.at + i1.out * 14
    let dockFar: SIMD3<Float> = farAt + farOut * 14
    for (at, out, along) in [(dockI1, i1.out, i1.along), (dockFar, farOut, farAlong)] {
        for p in place(d.dbd, at: at, out: out, along: along) {
            shapes.append(.sphere(center: p, radius: vdwRadius("C"), color: dbdColor))
        }
    }
    // Typed steps rather than one long expression: the laptop's older Swift
    // type checker gives up on the chained form.
    let sum: SIMD3<Float> = dockI1 + dockFar
    let mid: SIMD3<Float> = sum * 0.5
    let outward: SIMD3<Float> = simd_normalize(simd_mix(i1.out, farOut, SIMD3(repeating: 0.5)))
    let coreAt: SIMD3<Float> = mid + outward * 22
    let corePos = place(d.core, at: coreAt, out: outward, along: simd_normalize(farAt - i1.at))
    for (j, p) in corePos.enumerated() {
        shapes.append(.sphere(center: p, radius: vdwRadius(d.core[j].element), color: coreColor))
    }

    // L-arabinose: drifts in from outside, then sits in the pocket.
    if s.sugarIn > 0.01 {
        let pocket = coreAt + outward * 6
        let approach = coreAt + outward * 90
        let at = simd_mix(approach, pocket, SIMD3(repeating: s.sugarIn))
        let sugarPos = place(d.sugar, at: at, out: outward, along: simd_normalize(farAt - i1.at), scale: 1.6)
        for (j, p) in sugarPos.enumerated() {
            shapes.append(.sphere(center: p, radius: ballRadius(d.sugar[j].element) * 2.2, color: sugarColor))
        }
        for u in 0..<sugarPos.count {
            for v in (u + 1)..<sugarPos.count where simd_distance(sugarPos[u], sugarPos[v]) < 2.6 {
                shapes.append(.cylinder(from: sugarPos[u], to: sugarPos[v], radius: bondRadius * 2,
                                        color: sugarColor))
            }
        }
    }

    // RNA polymerase, drawn as a SHAPE, not as atoms. The real holoenzyme is
    // ~26,000 atoms and would dominate the frame and the render; the story only
    // needs something that is shut out and then let in. Sized to the real
    // enzyme, roughly 100 x 100 x 160 A.
    let promoterBP = d.plus1 - 32 - lo
    let prom = siteFrame(d, at: max(0, min(promoterBP, d.nBP - 1)), span: s.span)
    // Approach the promoter from a fixed side rather than along `out`, which
    // spins with the helical twist and would drop the polymerase somewhere
    // different for every base pair. `side` is perpendicular to the DNA and in
    // the plane the loop lies in, pointing away from the loop.
    var side: SIMD3<Float> = simd_cross(prom.along, SIMD3<Float>(0, 0, 1))
    if simd_length(side) < 1e-5 { side = SIMD3<Float>(0, -1, 0) }
    side = simd_normalize(side)
    // Above the promoter, where the frame is clear: below it the caption bar
    // cuts the polymerase in half, and the loop occupies everything to the left.
    if side.y < 0 { side = -side }
    let parked: SIMD3<Float> = prom.at + side * 120 + prom.along * 230
    let landed: SIMD3<Float> = prom.at + side * 46
    let rnapAt = simd_mix(parked, landed, SIMD3(repeating: s.rnap))
    // A crab-claw silhouette, roughly the real holoenzyme's 100 x 100 x 160 A.
    let lobes: [(SIMD3<Float>, Float)] = [
        (SIMD3(0, 0, 0), 26), (SIMD3(20, 14, 4), 22), (SIMD3(20, -16, -4), 21),
        (SIMD3(-22, 5, 0), 20), (SIMD3(38, 0, 0), 17), (SIMD3(-40, -2, 6), 15)]
    let e1: SIMD3<Float> = simd_normalize(prom.along)
    let e2: SIMD3<Float> = side
    let e3: SIMD3<Float> = simd_cross(e1, e2)
    for (offset, r) in lobes {
        let p: SIMD3<Float> = rnapAt + e1 * offset.x + e2 * offset.y + e3 * offset.z
        shapes.append(.sphere(center: p, radius: r, color: rnapColor))
    }
    return shapes
}

// MARK: - the cycle

/// Smooth 0→1 between `a` and `b`.
func ramp(_ u: Double, _ a: Double, _ b: Double) -> Float {
    if u <= a { return 0 }
    if u >= b { return 1 }
    let x = Float((u - a) / (b - a))
    return x * x * (3 - 2 * x)
}

/// The cycle. Each stage runs forwards; nothing is played backwards.
func state(_ d: SwitchData, at u: Double, openSpan: Float = 0.68) -> SwitchState {
    let sugarArrives = ramp(u, 0.14, 0.27)
    let sugarLeaves = ramp(u, 0.78, 0.86)
    let flips = ramp(u, 0.27, 0.48)
    let unflips = ramp(u, 0.86, 0.99)
    let lands = ramp(u, 0.48, 0.62)
    let leaves = ramp(u, 0.76, 0.84)

    let sugarIn = sugarArrives * (1 - sugarLeaves)
    let gripI2 = flips * (1 - unflips)
    let rnap = lands * (1 - leaves)
    let span = 1 - (1 - openSpan) * gripI2
    let transcribing = rnap

    let title: String
    let subtitle: String
    let facts: String
    if u < 0.14 {
        title = "Switched off"
        subtitle = "AraC holds araO2 and araI1 at once, and the DNA between them has nowhere to go but round"
        facts = "the loop keeps RNA polymerase off pBAD · araO2 and araI1 are \(d.spacingBP) bp apart"
    } else if u < 0.27 {
        title = "Arabinose arrives"
        subtitle = "the sugar binds a pocket in AraC's sugar-binding domain (PDB 2ARC, 1.5 Å)"
        facts = "affinity for the adjacent pair araI1+araI2 rises about fiftyfold"
    } else if u < 0.48 {
        title = "The grip moves"
        subtitle = "AraC lets go of the distant araO2 and takes araI2, next door to araI1"
        facts = "Schleif's light switch mechanism · the loop has nothing holding it shut"
    } else if u < 0.84 {
        title = "pBAD can be read"
        subtitle = "RNA polymerase reaches the promoter and transcribes — GFP, the protein of step 9"
        facts = "polymerase drawn as a shape, not atoms · the real holoenzyme is ~26,000 atoms"
    } else {
        title = "And back"
        subtitle = "the sugar leaves, AraC takes araO2 again, and the loop re-forms"
        facts = "a real cycle: every stage runs forwards, nothing is played in reverse"
    }
    let caption = Caption(title: title, subtitle: subtitle, facts: facts,
                          aside: "pGLO \(d.window.0)–\(d.window.1) · \(d.nBP) bp")
    return SwitchState(span: span, sugarIn: sugarIn, gripI2: gripI2, rnap: rnap,
                       transcribing: transcribing, caption: caption)
}
