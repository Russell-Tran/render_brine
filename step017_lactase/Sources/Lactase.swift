// E. coli (lacZ) beta-galactosidase cutting lactose into glucose and galactose:
// the scene, the catalytic cycle, and the chemistry the tests check.
//
// The subject, and a substitution carried in the open
// ---------------------------------------------------
// The enzyme that cuts lactose in a human being is lactase-phlorizin hydrolase
// (UniProt P09848). Nobody has solved it — no crystal structure, no cryo-EM
// map, nothing. Tools/build_lactase.py asks the PDB every time it runs and
// stops the build if that ever stops being true.
//
// So the enzyme on screen is the E. coli one, and the caption bar says so on
// every frame. Same reaction, same two-step mechanism, same family (GH2),
// different organism and a different protein.
//
// What is measured and what is not
// --------------------------------
// Juers et al. (Biochemistry 40:14781, 2001) solved this cycle deliberately,
// state by state, and four of this render's nine boundary poses are a deposited
// structure — including 1JZ2, which IS the covalent intermediate, trapped with
// a 2-fluoro sugar and refined at 2.1 A. Step 12's high-water mark was both
// ENDS of a motion solved with the path between them drawn; here the middle is
// solved too.
//
// Step 11's discipline, run first
// -------------------------------
// Step 11 measured the difference between its two structures before animating
// it, found it smaller than the noise, and refused to animate it. The same
// check runs here, in the builder, before a frame is drawn — and it comes back
// the same way. Every state-to-state difference in the PROTEIN (0.18-0.27 A over
// the chain, 0.63-0.77 A over the active site) is SMALLER than the difference
// between the four copies inside a single entry (0.34 A and 1.13 A), which were
// refined against one set of data and therefore cannot differ for any reason
// but noise.
//
// So the protein does not move in this render. Not as a simplification — as a
// refusal. `scene.motion.verdict` carries the finding and a test fails the
// build if the numbers ever stop supporting it.
//
// What DOES move, far above that noise: the ligand. The four substrate
// complexes sit within 0.10 A of each other in a SHALLOW site; the three
// wild-type complexes sit within 0.42 A of each other 2.7 A deeper. The sugar's
// slide between them is 6x the spread within either group.

import Foundation
import simd

// MARK: - Loading

struct LacAtom {
    var element: String
    var position: SIMD3<Float>
    var chain: String
    var seq: Int
    var res: String
}

struct LacCastAtom {
    var element: String
    var name: String
    var res: String
    var seq: Int
    var position: SIMD3<Float>
}

/// One of the 27 atoms whose identity is fixed across the whole cycle.
struct LacMobile {
    var element: String
    var label: String            // "GAL:C1", "GLC:O4", "WAT:O", "H:acid"
    var keys: [SIMD3<Float>]     // one position per boundary; keys.first == keys.last
}

struct LacMotion {
    var noiseCA: Float
    var noiseSite: Float
    var worstCA: Float
    var worstSite: Float
    var proteinMovesAboveNoise: Bool
    var siteMovesAboveNoise: Bool
    var verdict: String
}

struct LacAnomeric {
    var state: String
    var entry: String
    var signedVolume: Float
    var config: String
    var bond: Float
}

struct LacScene {
    var atoms: [LacAtom]
    var complementing: Set<Int>          // atoms of the partner subunit's 272-288 loop
    var cast: [LacCastAtom]              // the ball-and-stick side chains, from 1JZ7
    var mobiles: [LacMobile]
    var boundaryNames: [String]
    var boundarySource: [String]
    var channel: [SIMD3<Float>]
    var activeSite: SIMD3<Float>
    var nucleophile: SIMD3<Float>
    var metal: SIMD3<Float>
    var motion: LacMotion
    var anomeric: [LacAnomeric]
    var slide: Float
    var lactoneOutOfPlane: Float
    var sugarOutOfPlane: Float
    var waterToC1: Float
    var waterCosToBond: Float
    var waterToGlu461: Float
    var productO1ToGlu461: Float
    var metalToNucleophile: Float
    var interfaceToSite: Float
    var interfaceToSubstrate: Float
    var mutantEntries: [String]
    var protonPermutation: [Int]
    var provenance: [String: String]

    /// The centre of the ball-and-stick cast: where the porthole is bored.
    var castCenter: SIMD3<Float> {
        var sum = SIMD3<Float>.zero
        for a in cast { sum += a.position }
        return cast.isEmpty ? activeSite : sum / Float(cast.count)
    }

    /// How far the cast reaches from its own centre, which sets how wide the
    /// porthole has to open and how far behind the site it has to stop.
    var castRadius: Float {
        let c: SIMD3<Float> = castCenter
        var r: Float = 0
        for a in cast { r = max(r, simd_length(a.position - c)) }
        // Only the molecules that are IN the site. The departing glucose and
        // the parked ones are tens of angstroms away down the channel, and
        // sizing the porthole to reach them would bore a hole the camera
        // could not get far enough back to see as a hole.
        for m in mobiles {
            for k in m.keys {
                let d: Float = simd_length(k - c)
                if d < 11 { r = max(r, d) }
            }
        }
        return r
    }

    /// The porthole's radius when fully open, and how far back the camera has
    /// to sit for it to read as an opening in a surface rather than as a
    /// tunnel the camera is already inside. Both from the cast's own size.
    var portholeFullRadius: Float { castRadius + 2.0 }

    var nearDistance: Float {
        let tanV: Float = tan(radians(26.0 / 2))
        let aspect: Float = Float(lacFrameWidth) / Float(lacFrameViewHeight)
        // the bore fills a little over half the frame's width, which leaves a
        // rim of intact surface all the way round it — without that rim the
        // shot stops being a hole in something and becomes a tunnel
        let halfWidth: Float = portholeFullRadius / 0.55
        return halfWidth / (tanV * aspect)
    }

    /// Down inside the bore, where the chemistry fills the frame. Sized from
    /// the cast rather than chosen: the side chains and the sugar span about
    /// `castRadius`, and this puts that across nine tenths of the frame's
    /// height.
    var insideDistance: Float {
        let tanV: Float = tan(radians(26.0 / 2))
        return castRadius / (0.90 * tanV)
    }
}

enum LacError: Error, CustomStringConvertible {
    case badFile(String)
    var description: String {
        switch self {
        case .badFile(let d): return "couldn't read the scene: \(d)"
        }
    }
}

private func vec(_ any: Any?) -> SIMD3<Float> {
    guard let n = any as? [NSNumber], n.count == 3 else { return .zero }
    return SIMD3(n[0].floatValue, n[1].floatValue, n[2].floatValue)
}

private func f(_ any: Any?) -> Float { (any as? NSNumber)?.floatValue ?? 0 }

func loadScene(from url: URL) throws -> LacScene {
    let data = try Data(contentsOf: url)
    guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
        throw LacError.badFile("not an object")
    }
    guard let rawAtoms = root["atoms"] as? [[String: Any]] else {
        throw LacError.badFile("no atoms")
    }
    var atoms: [LacAtom] = []
    atoms.reserveCapacity(rawAtoms.count)
    for a in rawAtoms {
        atoms.append(LacAtom(element: a["el"] as? String ?? "C",
                             position: vec(a["pos"]),
                             chain: a["chain"] as? String ?? "A",
                             seq: a["seq"] as? Int ?? 0,
                             res: a["res"] as? String ?? ""))
    }

    let interface = root["interface"] as? [String: Any] ?? [:]
    let complementing = Set((interface["complementingAtoms"] as? [Int]) ?? [])
    let partner = interface["partner"] as? String ?? "D"
    let loops = interface["loops"] as? [String: Any] ?? [:]
    let partnerLoop = loops[partner] as? [String: Any] ?? [:]

    let castRaw = (root["cast"] as? [String: Any])?["1JZ7"] as? [[String: Any]] ?? []
    var cast: [LacCastAtom] = []
    for a in castRaw {
        cast.append(LacCastAtom(element: a["el"] as? String ?? "C",
                                name: a["name"] as? String ?? "",
                                res: a["res"] as? String ?? "",
                                seq: a["seq"] as? Int ?? 0,
                                position: vec(a["pos"])))
    }

    guard let bs = root["boundaries"] as? [[[String: Any]]], bs.count == 9 else {
        throw LacError.badFile("expected nine boundary poses")
    }
    let n = bs[0].count
    var mobiles: [LacMobile] = []
    for i in 0..<n {
        var keys: [SIMD3<Float>] = []
        for k in 0..<9 { keys.append(vec(bs[k][i]["pos"])) }
        mobiles.append(LacMobile(element: bs[0][i]["el"] as? String ?? "C",
                                 label: bs[0][i]["label"] as? String ?? "?",
                                 keys: keys))
    }

    let m = root["motion"] as? [String: Any] ?? [:]
    let motion = LacMotion(noiseCA: f(m["noiseCA"]), noiseSite: f(m["noiseSite"]),
                           worstCA: f(m["worstCA"]), worstSite: f(m["worstSite"]),
                           proteinMovesAboveNoise: m["proteinMovesAboveNoise"] as? Bool ?? true,
                           siteMovesAboveNoise: m["siteMovesAboveNoise"] as? Bool ?? true,
                           verdict: m["verdict"] as? String ?? "")

    var anomeric: [LacAnomeric] = []
    for a in (root["anomeric"] as? [[String: Any]] ?? []) {
        anomeric.append(LacAnomeric(state: a["state"] as? String ?? "",
                                    entry: a["entry"] as? String ?? "",
                                    signedVolume: f(a["signedVolume"]),
                                    config: a["config"] as? String ?? "",
                                    bond: f(a["bond"])))
    }

    let sites = root["sites"] as? [String: Any] ?? [:]
    let lactone = root["lactone"] as? [String: Any] ?? [:]
    let water = root["water"] as? [String: Any] ?? [:]
    let metal = root["metal"] as? [String: Any] ?? [:]

    return LacScene(
        atoms: atoms,
        complementing: complementing,
        cast: cast,
        mobiles: mobiles,
        boundaryNames: root["boundaryNames"] as? [String] ?? [],
        boundarySource: root["boundarySource"] as? [String] ?? [],
        channel: (root["channel"] as? [[NSNumber]] ?? []).map {
            SIMD3($0[0].floatValue, $0[1].floatValue, $0[2].floatValue)
        },
        activeSite: vec(root["activeSite"]),
        nucleophile: vec(root["nucleophile"]),
        metal: vec(metal["pos"]),
        motion: motion,
        anomeric: anomeric,
        slide: f(sites["slide"]),
        lactoneOutOfPlane: f(lactone["outOfPlane"]),
        sugarOutOfPlane: f(lactone["sugarOutOfPlane"]),
        waterToC1: f(water["toC1"]),
        waterCosToBond: f(water["cosToBond"]),
        waterToGlu461: f(water["toGlu461"]),
        productO1ToGlu461: f(water["productO1ToGlu461"]),
        metalToNucleophile: f(metal["toNucleophile"]),
        interfaceToSite: f(partnerLoop["toSiteResidues"]),
        interfaceToSubstrate: f(partnerLoop["toSubstrate"]),
        mutantEntries: root["mutantEntries"] as? [String] ?? [],
        protonPermutation: root["protonPermutation"] as? [Int] ?? [0, 1, 2],
        provenance: (root["provenance"] as? [String: String]) ?? [:])
}

// MARK: - Radii and colour

/// Van der Waals radii in ångströms, Bondi, J. Phys. Chem. 68:441 (1964).
func vdwRadius(_ element: String) -> Float {
    switch element {
    case "C": return 1.70
    case "N": return 1.55
    case "O": return 1.52
    case "S": return 1.80
    case "P": return 1.80
    case "SE": return 1.90
    case "F": return 1.47
    case "MG": return 1.73
    case "NA": return 2.27
    case "H": return 1.20
    default: return 1.70
    }
}

/// Covalent radii in ångströms, Cordero et al., Dalton Trans. (2008) 2832.
/// These decide what counts as a bond, which is how the covalent intermediate
/// is detected rather than declared.
func covalentRadius(_ element: String) -> Float {
    switch element {
    case "C": return 0.76
    case "N": return 0.71
    case "O": return 0.66
    case "S": return 1.05
    case "F": return 0.57
    case "MG": return 1.41
    case "H": return 0.31
    default: return 0.76
    }
}

/// Radii for ball-and-stick, where the point is connectivity rather than bulk.
/// The same proportions as steps 6 to 9.
func ballRadius(_ element: String) -> Float {
    switch element {
    case "C": return 0.50
    case "N": return 0.49
    case "O": return 0.47
    case "S": return 0.58
    case "MG": return 0.62
    case "H": return 0.26
    default: return 0.34
    }
}

let bondRadius: Float = 0.145

/// One muted colour per subunit, so the tetramer reads as four things. The
/// catalytic one is the lightest; the partner whose loop completes the wall of
/// its site is next; the two that have nothing to do with this active site are
/// darker still. Desaturated on purpose — the only saturated colours in the
/// frame are inside the porthole.
func chainColor(_ chain: String, complementing: Bool) -> SIMD3<Float> {
    if complementing { return SIMD3(0.78, 0.55, 0.86) }
    switch chain {
    case "A": return SIMD3(0.86, 0.87, 0.84)     // the one whose site is opened
    case "D": return SIMD3(0.56, 0.62, 0.74)     // the partner across the interface
    case "B": return SIMD3(0.50, 0.58, 0.55)
    default: return SIMD3(0.60, 0.55, 0.50)
    }
}

/// Element colours for the ball-and-stick cast: the usual convention, but the
/// two catalytic glutamates are given their own colours because the whole story
/// is which of them is doing what.
let nucleophileColor = SIMD3<Float>(0.98, 0.52, 0.22)     // Glu537, the nucleophile
let acidBaseColor = SIMD3<Float>(0.40, 0.76, 0.98)        // Glu461, acid then base
let sugarColor = SIMD3<Float>(0.62, 0.90, 0.52)           // the galactosyl
let glucoseColor = SIMD3<Float>(0.96, 0.86, 0.42)         // the leaving glucose
let waterColor = SIMD3<Float>(0.52, 0.72, 0.96)
let protonColor = SIMD3<Float>(0.98, 0.95, 0.88)
let metalColor = SIMD3<Float>(0.72, 0.80, 0.86)

func castColor(_ a: LacCastAtom) -> SIMD3<Float> {
    if a.seq == 537 { return nucleophileColor }
    if a.seq == 461 { return acidBaseColor }
    switch a.element {
    case "O": return SIMD3(0.88, 0.42, 0.38)
    case "N": return SIMD3(0.42, 0.55, 0.88)
    case "S": return SIMD3(0.88, 0.80, 0.36)
    default: return SIMD3(0.60, 0.64, 0.62)
    }
}

func mobileColor(_ label: String) -> SIMD3<Float> {
    if label.hasPrefix("GAL") { return sugarColor }
    if label.hasPrefix("GLC") { return glucoseColor }
    if label.hasPrefix("WAT") { return waterColor }
    return protonColor
}

// MARK: - The cycle
//
// 240 frames at 12.5 fps: 19.2 seconds. The frame budget is spent where the
// beat is — nearly half of it on the two half-cycles themselves, because
// inverting an anomeric centre twice is the thing this render exists to show
// and it cannot be shown in a blink.

// 880 across, as step 12 was: this is a protein render, not one of the
// single-molecule close-ups. The frame size is also what the GIF costs — every
// frame in which the CAMERA moves changes nearly every pixel. At 1280 with a long
// establishing spin the loop came to 56 MB, against the 13-15 MB the rest of
// this project's renders weigh. So the establishing turn is short and the chemistry is long, which is
// the right way round anyway.
let lacFrameWidth = 880
let lacFrameViewHeight = 660
let lacFrameCaptionHeight = 110
let lacDelayCentiseconds = 8              // 12.5 fps

let turnFrames = 26                       // one full turn of the closed tetramer
let openFrames = 10                       // the porthole irises open, camera moves in
let cycleFrames = 140                     // the chemistry: 70% of the loop
let closeFrames = 10
let holdFrames = 6
let lacFrameCount = turnFrames + openFrames + cycleFrames + closeFrames + holdFrames

/// How the 140 cycle frames are divided between the eight intervals. The two
/// half-cycles (3 and 6 below) get 28 frames each, more than any other beat.
let stageFrames: [Int] = [
    20,    // 0->1  the substrate arrives and binds in the shallow site
    12,    // 1->2  it slides 2.7 A into the deep site
    28,    // 2->3  GLYCOSYLATION begins: Glu537 comes in, the bond stretches
    18,    // 3->4  ... and breaks. The anomeric centre inverts; glucose leaves
    14,    // 4->5  the glucose goes down the channel
    12,    // 5->6  the water comes in behind, where the glucose was
    28,    // 6->7  DEGLYCOSYLATION: the centre inverts back. Retention.
     8,    // 7->8  the galactose goes down the channel too
]

func smoothstep(_ x: Float) -> Float {
    let c: Float = min(max(x, 0), 1)
    return c * c * (3 - 2 * c)
}

/// Where in the cycle frame `f` is: the interval index and how far through it,
/// with the whole thing periodic in `lacFrameCount` so the loop closes.
///
/// Written to take ANY integer, including `lacFrameCount` itself, which is what
/// makes the loop-closure test non-vacuous: it asks for frame N and frame 0 and
/// compares the poses.
func cyclePosition(frame: Int, mutations: LacMutations = []) -> (stage: Int, u: Float) {
    // The mutation reduces modulo the wrong number of frames, which leaves
    // every frame looking perfectly fine and quietly stops the loop closing —
    // step 14's `badLoopPhase`, the same trick.
    let period = mutations.contains(.openLoop) ? lacFrameCount + 7 : lacFrameCount
    let f = ((frame % period) + period) % period
    let start = turnFrames + openFrames
    if f < start { return (0, 0) }
    var k = f - start
    if k >= cycleFrames { return (stageFrames.count, 0) }
    for (i, n) in stageFrames.enumerated() {
        if k < n { return (i, Float(k) / Float(n)) }
        k -= n
    }
    return (stageFrames.count, 0)
}

/// Every mobile atom's position at frame `f`.
///
/// Boundary poses come from the structures; the paths between them are
/// smoothstep interpolation and nothing more. Because `keys.first == keys.last`
/// and `cyclePosition` is periodic, `pose(at: lacFrameCount) == pose(at: 0)`.
func pose(_ scene: LacScene, at frame: Int, mutations: LacMutations = []) -> [SIMD3<Float>] {
    let (stage, u) = cyclePosition(frame: frame, mutations: mutations)
    // The mutation runs the second half-cycle to the wrong pose, so the centre
    // inverts once and stays inverted: a net INVERSION, which is what the other
    // kind of glycosidase does.
    let last = stageFrames.count
    let i: Int = min(stage, last)
    let a = boundaryPose(scene, i, mutations: mutations)
    let b = boundaryPose(scene, min(i + 1, last), mutations: mutations)
    var out: [SIMD3<Float>] = []
    out.reserveCapacity(scene.mobiles.count)
    let t: Float = smoothstep(u)
    for k in 0..<scene.mobiles.count {
        out.append(a[k] + (b[k] - a[k]) * t)
    }
    if mutations.contains(.bothRoles) {
        // Park a second proton on Glu461's other carboxylate oxygen, so it is
        // an acid and a base in the same frame.
        let i = mobileIndex(scene, "H:w1")
        let oe1: SIMD3<Float> = glu461(scene, "OE1")
        let cd: SIMD3<Float> = glu461(scene, "CD")
        let dir: SIMD3<Float> = simd_normalize(oe1 - cd)
        out[i] = oe1 + dir * 0.98
    }
    return out
}

/// One of the nine boundary poses, which is what every frame is interpolated
/// between and what the tests read. Going through one function means a mutation
/// cannot be applied to the render and miss the test, or the other way round.
func boundaryPose(_ scene: LacScene, _ boundary: Int,
                  mutations: LacMutations = []) -> [SIMD3<Float>] {
    var out = scene.mobiles.map { $0.keys[boundary] }
    if mutations.contains(.invertOnce) && boundary == 7 {
        // Put the oxygen the water brought on the SAME face the enzyme's
        // oxygen was on, so the second half-cycle does not invert: one
        // inversion, net INVERSION, which is what the other kind of
        // glycosidase does and what this enzyme does not.
        let i = mobileIndex(scene, "WAT:O")
        let ring = ringFrame(scene, boundary: boundary)
        let d: Float = simd_dot(out[i] - ring.c1, ring.normal)
        out[i] = out[i] - ring.normal * (2 * d)
    }
    return out
}

/// C1, and the normal of the O5-C1-C2 plane, at a boundary pose.
func ringFrame(_ scene: LacScene, boundary: Int) -> (c1: SIMD3<Float>, normal: SIMD3<Float>) {
    var c1 = SIMD3<Float>.zero, o5 = SIMD3<Float>.zero, c2 = SIMD3<Float>.zero
    for m in scene.mobiles {
        if m.label == "GAL:C1" { c1 = m.keys[boundary] }
        if m.label == "GAL:O5" { o5 = m.keys[boundary] }
        if m.label == "GAL:C2" { c2 = m.keys[boundary] }
    }
    return (c1, simd_normalize(simd_cross(o5 - c1, c2 - c1)))
}

// MARK: - The chemistry, read back out of the coordinates

/// The index of a mobile atom by label.
func mobileIndex(_ scene: LacScene, _ label: String) -> Int {
    for (i, m) in scene.mobiles.enumerated() where m.label == label { return i }
    return -1
}

/// Which heavy atom sits in the exocyclic position at C1 at a given boundary.
/// Three different atoms hold it over the cycle, and that is the mechanism:
/// the glucose's bridging oxygen, then Glu537's carboxylate oxygen, then the
/// oxygen the water brought.
func exocyclicAt(boundary: Int) -> String {
    if boundary <= 3 { return "GLC:O4" }
    if boundary <= 6 { return "GLU537:OE2" }
    return "WAT:O"
}

/// Glu537's nucleophilic oxygen. It never moves — the protein does not move in
/// this render, because the measurement said it must not.
func glu537OE2(_ scene: LacScene) -> SIMD3<Float> {
    for a in scene.cast where a.seq == 537 && a.name == "OE2" { return a.position }
    return scene.nucleophile
}

func glu461(_ scene: LacScene, _ name: String) -> SIMD3<Float> {
    for a in scene.cast where a.seq == 461 && a.name == name { return a.position }
    return .zero
}

/// The signed volume at the anomeric carbon:
///
///     (exo - C1) . [ (O5 - C1) x (C2 - C1) ]
///
/// Positive and negative are the two faces of the ring. Nothing in it depends
/// on an atom's NAME — which matters, because the PDB calls 1JZ2's ligand
/// "2-deoxy-2-fluoro-BETA-D-galactopyranose" while its coordinates, bonded to
/// Glu537, are alpha. The coordinates decide.
func anomericVolume(c1: SIMD3<Float>, o5: SIMD3<Float>, c2: SIMD3<Float>,
                    exo: SIMD3<Float>) -> Float {
    let a: SIMD3<Float> = o5 - c1
    let b: SIMD3<Float> = c2 - c1
    let n: SIMD3<Float> = simd_cross(a, b)
    return simd_dot(exo - c1, n)
}

/// The same thing at a whole posed frame, using whichever atom is in the
/// exocyclic position then.
func anomericVolume(_ scene: LacScene, positions: [SIMD3<Float>], boundary: Int) -> Float {
    let c1 = positions[mobileIndex(scene, "GAL:C1")]
    let o5 = positions[mobileIndex(scene, "GAL:O5")]
    let c2 = positions[mobileIndex(scene, "GAL:C2")]
    let label = exocyclicAt(boundary: boundary)
    let exo: SIMD3<Float> = label == "GLU537:OE2" ? glu537OE2(scene)
                                                 : positions[mobileIndex(scene, label)]
    return anomericVolume(c1: c1, o5: o5, c2: c2, exo: exo)
}

/// The anomeric hydrogen: the fourth substituent at C1, derived from the three
/// heavy ones by putting it where a tetrahedral carbon has no choice but to put
/// it. DERIVED, not measured — X-rays at these resolutions see no hydrogen —
/// but derived from measured positions rather than chosen.
///
/// It is drawn because it is the clearest possible picture of the inversion:
/// it starts under the ring, ends up over it, and comes back.
func anomericHydrogen(_ scene: LacScene, positions: [SIMD3<Float>],
                      boundary: Int) -> SIMD3<Float> {
    let c1 = positions[mobileIndex(scene, "GAL:C1")]
    let o5 = positions[mobileIndex(scene, "GAL:O5")]
    let c2 = positions[mobileIndex(scene, "GAL:C2")]
    let label = exocyclicAt(boundary: boundary)
    let exo: SIMD3<Float> = label == "GLU537:OE2" ? glu537OE2(scene)
                                                 : positions[mobileIndex(scene, label)]
    let u1: SIMD3<Float> = simd_normalize(o5 - c1)
    let u2: SIMD3<Float> = simd_normalize(c2 - c1)
    let u3: SIMD3<Float> = simd_normalize(exo - c1)
    let sum: SIMD3<Float> = u1 + u2 + u3
    let n: Float = simd_length(sum)
    // At the flat transition state the three sum to nearly nothing, so fall
    // back on the ring normal, which is where an sp2 hydrogen lies anyway.
    if n < 0.12 {
        let ring: SIMD3<Float> = simd_normalize(simd_cross(o5 - c1, c2 - c1))
        let side: Float = simd_dot(u3, ring) > 0 ? -1 : 1
        return c1 + simd_normalize(u1 + u2) * (-1.09) + ring * (0.02 * side)
    }
    return c1 - (sum / n) * 1.09
}

/// A bond, as the distance test sees it: `strength` falls from 1 to 0 as the
/// two atoms pull apart, so a breaking bond thins out instead of popping.
struct LacBond {
    var a: SIMD3<Float>
    var b: SIMD3<Float>
    var color: SIMD3<Float>
    var strength: Float
    var labelA: String
    var labelB: String
}

/// Are these two atoms bonded, and how strongly? Covalent radii plus a margin,
/// exactly as steps 8 and 9 decided their bonds.
func bondStrength(_ ea: String, _ eb: String, distance d: Float) -> Float {
    let limit: Float = covalentRadius(ea) + covalentRadius(eb)
    let full: Float = limit + 0.20
    let gone: Float = limit + 0.95
    if d <= full { return 1 }
    if d >= gone { return 0 }
    return 1 - smoothstep((d - full) / (gone - full))
}

/// Is there a real covalent bond between Glu537's carboxylate oxygen and the
/// anomeric carbon at this frame, and how long is it?
///
/// This is the test the covalent intermediate has to pass: not "the code says
/// it is bonded here" but "the two atoms are a bond length apart".
func covalentBond(_ scene: LacScene, positions: [SIMD3<Float>]) -> (bonded: Bool, length: Float) {
    let c1 = positions[mobileIndex(scene, "GAL:C1")]
    let d: Float = simd_length(c1 - glu537OE2(scene))
    let limit: Float = covalentRadius("C") + covalentRadius("O") + 0.20
    return (d <= limit, d)
}

/// What Glu461 is doing. The proton it holds is the whole of its role: with one
/// it is an acid and can protonate the oxygen that leaves; without one it is a
/// base and can take a proton off the attacking water. It is never both, and a
/// test checks exactly that.
enum Glu461Role {
    case acid            // holding its proton, ready to give it to the leaving oxygen
    case inFlight        // the proton is between it and the leaving oxygen
    case base            // bare carboxylate, ready to take one off the water
    var label: String {
        switch self {
        case .acid: return "acid"
        case .inFlight: return "handing it over"
        case .base: return "base"
        }
    }
}

/// Read off the coordinates, not from a table: how far the acidic proton is
/// from Glu461's carboxylate oxygen.
func glu461Role(_ scene: LacScene, positions: [SIMD3<Float>]) -> Glu461Role {
    let oe2: SIMD3<Float> = glu461(scene, "OE2")
    var nearest: Float = .greatestFiniteMagnitude
    for (i, m) in scene.mobiles.enumerated() where m.element == "H" {
        nearest = min(nearest, simd_length(positions[i] - oe2))
    }
    // An acid is holding a proton COVALENTLY, at about 0.98 Å. A hydrogen bond
    // at 1.9 Å is not holding it — that is the water still holding it, with
    // Glu461 poised as a base to take it. The band between is the transfer
    // itself, and it is narrow on purpose.
    let limit: Float = covalentRadius("O") + covalentRadius("H") + 0.20
    if nearest <= limit { return .acid }
    if nearest <= limit + 0.30 { return .inFlight }
    return .base
}

/// How many protons Glu461 is holding. Must never be two — an acid and a base
/// at the same time is not a thing.
func glu461ProtonCount(_ scene: LacScene, positions: [SIMD3<Float>]) -> Int {
    let oe2: SIMD3<Float> = glu461(scene, "OE2")
    let oe1: SIMD3<Float> = glu461(scene, "OE1")
    let limit: Float = covalentRadius("O") + covalentRadius("H") + 0.20
    var n = 0
    for (i, m) in scene.mobiles.enumerated() where m.element == "H" {
        let d: Float = min(simd_length(positions[i] - oe2), simd_length(positions[i] - oe1))
        if d <= limit { n += 1 }
    }
    return n
}

// MARK: - Atom and charge bookkeeping
//
// Step 7a's check_balances.py counted atoms and charge across every reaction of
// glycolysis. The same accounting here, and it is structural rather than
// written down: the 27 mobile atoms have fixed identities and never appear or
// disappear, so the balance is a property of the scene, not a claim about it.

struct LacFormula {
    var counts: [String: Int] = [:]
    var charge: Int = 0

    mutating func add(_ element: String, _ n: Int = 1) {
        counts[element, default: 0] += n
    }

    static func == (a: LacFormula, b: LacFormula) -> Bool {
        a.counts.filter { $0.value != 0 } == b.counts.filter { $0.value != 0 }
            && a.charge == b.charge
    }

    var text: String {
        counts.filter { $0.value != 0 }.sorted { $0.key < $1.key }
            .map { "\($0.key)\($0.value)" }.joined()
    }
}

/// lactose + water, counted from the atom identities the render actually uses.
func substrateFormula(_ scene: LacScene) -> LacFormula {
    var out = LacFormula()
    for m in scene.mobiles {
        out.add(m.element)
    }
    out.add("H", 1)            // the anomeric hydrogen, drawn but not in the mobile list
    // Every hydroxyl hydrogen the heavy-atom model leaves out. Lactose has 8
    // OH; with the water and the acid proton that is the full H count.
    out.add("H", 14)
    return out
}

/// glucose + galactose, the same way.
func productFormula(_ scene: LacScene) -> LacFormula {
    // Identical by construction: the mobile atoms are the same atoms, so this
    // is the same sum. Counted separately anyway, because the point of a
    // balance check is that the two sides are computed and then compared.
    var out = LacFormula()
    for m in scene.mobiles { out.add(m.element) }
    out.add("H", 1)
    out.add("H", 14)
    return out
}

/// The textbook equation, written out independently of the model, so the two
/// can be checked against each other.
func lactoseHydrolysis() -> (left: LacFormula, right: LacFormula) {
    var left = LacFormula()
    left.add("C", 12); left.add("H", 22); left.add("O", 11)      // lactose
    left.add("H", 2); left.add("O", 1)                           // water
    var right = LacFormula()
    right.add("C", 6); right.add("H", 12); right.add("O", 6)     // glucose
    right.add("C", 6); right.add("H", 12); right.add("O", 6)     // galactose
    return (left, right)
}

/// The two half-reactions, with the enzyme's two glutamates in them, so the
/// bookkeeping covers the intermediate and not just the ends.
///
///   1  lactose + E537-COO(-) + E461-COOH  ->  glucose + E537-COO-Gal + E461-COO(-)
///   2  E537-COO-Gal + H2O + E461-COO(-)   ->  galactose + E537-COO(-) + E461-COOH
func halfReactions() -> [(String, LacFormula, LacFormula)] {
    func make(_ c: Int, _ h: Int, _ o: Int, charge: Int = 0) -> LacFormula {
        var f = LacFormula()
        f.add("C", c); f.add("H", h); f.add("O", o)
        f.charge = charge
        return f
    }
    func sum(_ fs: [LacFormula]) -> LacFormula {
        var out = LacFormula()
        for f in fs {
            for (k, v) in f.counts { out.add(k, v) }
            out.charge += f.charge
        }
        return out
    }
    let lactose = make(12, 22, 11)
    let glucose = make(6, 12, 6)
    let galactose = make(6, 12, 6)
    let water = make(0, 2, 1)
    let nucleophileFree = make(0, 0, 2, charge: -1)          // Glu537 carboxylate
    let nucleophileBound = make(6, 11, 7)                    // Glu537-O-galactosyl
    let acid = make(0, 1, 2)                                 // Glu461 COOH
    let base = make(0, 0, 2, charge: -1)                     // Glu461 COO(-)
    return [
        ("glycosylation",
         sum([lactose, nucleophileFree, acid]),
         sum([glucose, nucleophileBound, base])),
        ("deglycosylation",
         sum([nucleophileBound, water, base]),
         sum([galactose, nucleophileFree, acid])),
    ]
}

// MARK: - Mutations
//
// Four deliberate breakages, each of exactly one thing, so the suite can be
// shown to notice. `make mutants` runs them.

struct LacMutations: OptionSet {
    let rawValue: Int
    static let invertOnce = LacMutations(rawValue: 1 << 0)     // break the retention
    static let noCovalentBond = LacMutations(rawValue: 1 << 1) // break the intermediate
    static let openLoop = LacMutations(rawValue: 1 << 2)       // break the loop closure
    static let bothRoles = LacMutations(rawValue: 1 << 3)      // Glu461 acid AND base

    static func fromEnvironment() -> LacMutations {
        let text: String = ProcessInfo.processInfo.environment["LAC_MUTATE"] ?? ""
        var out: LacMutations = []
        for name in text.split(separator: ",") {
            switch name {
            case "invert": out.insert(.invertOnce)
            case "bond": out.insert(.noCovalentBond)
            case "loop": out.insert(.openLoop)
            case "roles": out.insert(.bothRoles)
            default: break
            }
        }
        return out
    }
}

// MARK: - The porthole and the camera

/// How wide the porthole is at frame `f`, in ångströms. Shut, irises open,
/// stays open for the whole cycle, irises shut, shut — so the loop joins itself
/// without ever running backwards.
func portholeRadius(_ scene: LacScene, at frame: Int) -> Float {
    let period = lacFrameCount
    let f = ((frame % period) + period) % period
    let full: Float = scene.portholeFullRadius
    if f < turnFrames { return 0 }
    if f < turnFrames + openFrames {
        return full * smoothstep(Float(f - turnFrames) / Float(openFrames))
    }
    let closeAt = turnFrames + openFrames + cycleFrames
    if f < closeAt { return full }
    if f < closeAt + closeFrames {
        return full * (1 - smoothstep(Float(f - closeAt) / Float(closeFrames)))
    }
    return 0
}

/// How far back the camera has to be for the whole tetramer to fit, whatever
/// angle it is seen from. Measured from the atoms rather than guessed: the
/// molecule turns about y, so the horizontal silhouette is set by the largest
/// radius in the xz plane and the vertical one by the largest |y|.
func lacFarDistance(_ scene: LacScene, fov: Float) -> Float {
    var radial: Float = 0
    var height: Float = 0
    for a in scene.atoms {
        let p: SIMD3<Float> = a.position
        let r: Float = sqrt(p.x * p.x + p.z * p.z) + vdwRadius(a.element)
        if r > radial { radial = r }
        let h: Float = abs(p.y) + vdwRadius(a.element)
        if h > height { height = h }
    }
    let tanV: Float = tan(radians(fov / 2))
    let aspect: Float = Float(lacFrameWidth) / Float(lacFrameViewHeight)
    let tanH: Float = tanV * aspect
    let need: Float = max(radial / tanH, height / tanV)
    return need * 1.10
}

/// How far into the porthole the camera has gone at frame `f`: 0 out at the
/// whole tetramer, 1 at the mouth of the bore, 2 down inside it on the
/// chemistry. Two stages rather than one because they fight: at the mouth the
/// bore reads as a hole in a surface but the molecules are small, and close in
/// the molecules fill the frame but the rim has gone. So the shot does both,
/// in that order, and comes back out the same way.
func lacZoom(at frame: Int) -> Float {
    let period = lacFrameCount
    let f = ((frame % period) + period) % period
    let start = turnFrames + openFrames
    let pushIn = 14                          // frames spent going from mouth to close
    let pullOut = 12
    if f < turnFrames { return 0 }
    if f < start { return smoothstep(Float(f - turnFrames) / Float(openFrames)) }
    let k = f - start
    if k < pushIn { return 1 + smoothstep(Float(k) / Float(pushIn)) }
    if k < cycleFrames - pullOut { return 2 }
    if k < cycleFrames {
        let u: Float = Float(k - (cycleFrames - pullOut)) / Float(pullOut)
        return 1 + (1 - smoothstep(u))
    }
    let closeAt = start + cycleFrames
    if f < closeAt + closeFrames {
        return 1 - smoothstep(Float(f - closeAt) / Float(closeFrames))
    }
    return 0
}

/// The camera. It orbits the whole tetramer once with the porthole shut, then
/// stops turning and moves in — and from there to the end of the cycle it only
/// ever moves along its own view axis, never around. That matters: ambient
/// occlusion is computed from the surface normal alone, so a scene that does
/// not rotate keeps exactly the shading it had, frame after frame.
func lacCamera(_ scene: LacScene, at frame: Int) -> Camera {
    let period = lacFrameCount
    let f = ((frame % period) + period) % period
    let far: Float = lacFarDistance(scene, fov: 30)
    let mouth: Float = scene.nearDistance
    let inside: Float = scene.insideDistance
    let target: SIMD3<Float> = scene.castCenter

    if f < turnFrames {
        let yaw: Float = 360 * Float(f) / Float(turnFrames)
        return Camera.orbit(target: .zero, distance: far, yaw: yaw, pitch: 8, fov: 30)
    }
    let z: Float = lacZoom(at: f)
    let t: Float = min(z, 1)                 // 0 out wide, 1 at the mouth
    let s: Float = max(z - 1, 0)             // 0 at the mouth, 1 inside
    let centre: SIMD3<Float> = target * t
    let atMouth: Float = far + (mouth - far) * t
    let d: Float = atMouth + (inside - atMouth) * s
    let fov: Float = 30 + (26 - 30) * t
    // Oblique to the channel, not down it. The way out of this site points
    // straight at the viewer, so a head-on shot puts the departing glucose
    // exactly in front of the galactosyl it is leaving — the two atoms whose
    // parting is the whole beat, one hidden behind the other. Swinging the
    // camera off the axis separates them. The porthole is bored along whatever
    // axis the camera ends up on, so it is still a hole pointed at the viewer.
    let yaw: Float = 24 * t
    let pitch: Float = 8 + 16 * t
    return Camera.orbit(target: centre, distance: d, yaw: yaw, pitch: pitch, fov: fov)
}

/// The porthole itself: bored straight at the camera, through the cast's centre,
/// and stopped just behind the cast so the ball-and-stick is never buried in
/// what is left of the protein.
func lacPorthole(_ scene: LacScene, at frame: Int, camera: Camera) -> Porthole {
    let r: Float = portholeRadius(scene, at: frame)
    if r <= 0.001 { return .shut }
    let centre: SIMD3<Float> = scene.castCenter
    let axis: SIMD3<Float> = simd_normalize(camera.origin - centre)
    return Porthole(center: centre, axis: axis, radius: r,
                    back: -(scene.castRadius + 2.0))
}

// MARK: - Geometry for the GPU

/// Everything in the frame, as shapes. The protein is space-filling and
/// clippable; the cast and the reacting molecules are ball-and-stick and are
/// never clipped, because they are what the porthole is for.
func lacShapes(_ scene: LacScene, at frame: Int, porthole: Porthole,
               mutations: LacMutations = []) -> [GPUShape] {
    var out: [GPUShape] = []
    out.reserveCapacity(scene.atoms.count + 400)

    for (i, a) in scene.atoms.enumerated() {
        let c: SIMD3<Float> = chainColor(a.chain, complementing: scene.complementing.contains(i))
        out.append(GPUShape.sphere(center: a.position, radius: vdwRadius(a.element),
                                   color: c, clippable: true))
    }
    guard porthole.isOpen else { return out }

    // --- the cast --------------------------------------------------------
    for a in scene.cast {
        out.append(GPUShape.sphere(center: a.position, radius: ballRadius(a.element),
                                   color: castColor(a), clippable: false))
    }
    for i in 0..<scene.cast.count {
        for j in (i + 1)..<scene.cast.count {
            let a = scene.cast[i], b = scene.cast[j]
            if a.seq != b.seq { continue }
            let d: Float = simd_length(a.position - b.position)
            if bondStrength(a.element, b.element, distance: d) < 0.99 { continue }
            let mid: SIMD3<Float> = (a.position + b.position) / 2
            out.append(GPUShape.stick(from: a.position, to: mid, radius: bondRadius,
                                      color: castColor(a)))
            out.append(GPUShape.stick(from: mid, to: b.position, radius: bondRadius,
                                      color: castColor(b)))
        }
    }
    // the magnesium and the three things that hold it
    out.append(GPUShape.sphere(center: scene.metal, radius: ballRadius("MG"),
                               color: metalColor, clippable: false))
    for a in scene.cast where a.name == "OE1" || a.name == "ND1" || a.name == "OE2" {
        let d: Float = simd_length(a.position - scene.metal)
        if d > 2.6 { continue }
        out.append(GPUShape.stick(from: scene.metal, to: a.position,
                                  radius: bondRadius * 0.6, color: metalColor))
    }

    // --- the molecules ---------------------------------------------------
    let p: [SIMD3<Float>] = pose(scene, at: frame, mutations: mutations)
    let (stage, _) = cyclePosition(frame: frame, mutations: mutations)
    let boundary = min(stage, 8)
    for (i, m) in scene.mobiles.enumerated() {
        out.append(GPUShape.sphere(center: p[i], radius: ballRadius(m.element),
                                   color: mobileColor(m.label), clippable: false))
    }
    let h1: SIMD3<Float> = anomericHydrogen(scene, positions: p, boundary: boundary)
    out.append(GPUShape.sphere(center: h1, radius: ballRadius("H"),
                               color: protonColor, clippable: false))
    let c1 = p[mobileIndex(scene, "GAL:C1")]
    out.append(GPUShape.stick(from: c1, to: h1, radius: bondRadius * 0.7, color: protonColor))

    for b in lacBonds(scene, positions: p, mutations: mutations) {
        let r: Float = bondRadius * (0.35 + 0.65 * b.strength)
        out.append(GPUShape.stick(from: b.a, to: b.b, radius: r, color: b.color))
    }
    return out
}

/// Every bond among the reacting atoms, found by distance. Nothing is looked up
/// — the covalent intermediate exists in these frames because C1 and Glu537's
/// oxygen are 1.45 Å apart in them, and for no other reason.
func lacBonds(_ scene: LacScene, positions p: [SIMD3<Float>],
              mutations: LacMutations = []) -> [LacBond] {
    var out: [LacBond] = []
    let n = scene.mobiles.count
    for i in 0..<n {
        for j in (i + 1)..<n {
            let a = scene.mobiles[i], b = scene.mobiles[j]
            let d: Float = simd_length(p[i] - p[j])
            let s: Float = bondStrength(a.element, b.element, distance: d)
            if s <= 0.001 { continue }
            let colour: SIMD3<Float> = a.element == "H" ? protonColor
                : (b.element == "H" ? protonColor : mobileColor(a.label))
            out.append(LacBond(a: p[i], b: p[j], color: colour, strength: s,
                               labelA: a.label, labelB: b.label))
        }
    }
    // the bond to the enzyme itself
    let c1 = p[mobileIndex(scene, "GAL:C1")]
    let oe2: SIMD3<Float> = glu537OE2(scene)
    let d: Float = simd_length(c1 - oe2)
    var s: Float = bondStrength("C", "O", distance: d)
    if mutations.contains(.noCovalentBond) { s = 0 }
    if s > 0.001 {
        out.append(LacBond(a: oe2, b: c1, color: nucleophileColor, strength: s,
                           labelA: "GLU537:OE2", labelB: "GAL:C1"))
    }
    // the proton's bond to Glu461, when it has one
    let o461: SIMD3<Float> = glu461(scene, "OE2")
    for (i, m) in scene.mobiles.enumerated() where m.element == "H" {
        let dh: Float = simd_length(p[i] - o461)
        let sh: Float = bondStrength("O", "H", distance: dh)
        if sh > 0.001 {
            out.append(LacBond(a: o461, b: p[i], color: acidBaseColor, strength: sh,
                               labelA: "GLU461:OE2", labelB: m.label))
        }
    }
    return out
}

// MARK: - The caption bar
//
// House style, and NO evidence bar, for step 13's reason: a bar that says the
// same thing for 240 frames is furniture. Provenance lives in `lacConstants`
// and in the tests, which fail the build if a measured constant cites nothing.
//
// The bar names the ORGANISM on every frame. That is not decoration: the
// obvious reading of "lactose being cut" is the human enzyme, and this is not
// the human enzyme.

func lacCaption(_ scene: LacScene, at frame: Int) -> Caption {
    let period = lacFrameCount
    let f = ((frame % period) + period) % period
    if f < turnFrames {
        return Caption(
            title: "\u{03B2}-galactosidase, the enzyme that cuts lactose",
            subtitle: "Escherichia coli lacZ \u{00B7} tetramer \u{00B7} PDB 1JZ7, 1.50 \u{00C5}",
            facts: "\(scene.atoms.count) atoms at van der Waals radii \u{00B7} "
                 + "no human lactase has ever been solved, so this is the bacterial enzyme",
            aside: "one subunit per colour")
    }
    let (stage, _) = cyclePosition(frame: frame)
    switch stage {
    case 0:
        return Caption(title: "Lactose binds — but not where it reacts",
                       subtitle: "PDB 1JYN \u{00B7} E537Q mutant \u{00B7} the shallow site",
                       facts: "the nucleophile was mutated away so the substrate would sit "
                            + "there uncut \u{00B7} four such complexes agree to 0.10 \u{00C5}",
                       aside: "MEASURED")
    case 1:
        let s = String(format: "%.1f", scene.slide)
        return Caption(title: "It slides \(s) \u{00C5} deeper",
                       subtitle: "shallow site \u{2192} deep site \u{00B7} both ends solved",
                       facts: "only now is Glu461 close enough to the oxygen that has to "
                            + "leave \u{00B7} the complex in between is the one state nobody has caught",
                       aside: "MODEL: the path")
    case 2, 3:
        return Caption(title: "Glycosylation: Glu537 attacks, glucose leaves",
                       subtitle: "PDB 1JZ5 (flat, sp\u{00B2}) \u{2192} 1JZ2, the covalent "
                               + "intermediate itself",
                       facts: "Glu461 gives its proton to the bridging oxygen \u{00B7} the "
                            + "anomeric carbon INVERTS: \u{03B2} \u{2192} \u{03B1}",
                       aside: "inversion 1 of 2")
    case 4:
        return Caption(title: "The galactose is bonded to the protein",
                       subtitle: "\u{03B1}-galactosyl\u{2013}enzyme \u{00B7} PDB 1JZ2, trapped "
                               + "with a 2-fluoro sugar, 2.10 \u{00C5}",
                       facts: "C1\u{2013}OE2 is 1.45 \u{00C5}: a real bond, refined, not drawn in "
                            + "\u{00B7} glucose goes out the way it came",
                       aside: "MEASURED")
    case 5:
        let a = String(format: "%.1f", scene.waterToC1)
        let b = String(format: "%.1f", scene.waterToGlu461)
        return Caption(title: "A water comes in behind",
                       subtitle: "1JZ2's own water \u{00B7} \(a) \u{00C5} from C1, in line with "
                               + "the bond it will attack",
                       facts: "\(b) \u{00C5} from Glu461, which is now bare and can take its "
                            + "proton \u{00B7} same residue, opposite job",
                       aside: "MEASURED")
    case 6, 7:
        return Caption(title: "Deglycosylation: and it inverts back",
                       subtitle: "PDB 1JZ7, \u{03B2}-galactose \u{00B7} 1.50 \u{00C5}",
                       facts: "two inversions make a retention \u{00B7} the product has the same "
                            + "configuration as the substrate \u{00B7} Koshland, 1953",
                       aside: "inversion 2 of 2")
    default:
        return Caption(title: "Free enzyme again",
                       subtitle: "the loop closes on itself \u{00B7} atom for atom",
                       facts: "a catalytic cycle returns the enzyme, not the sugar \u{2014} the "
                            + "next lactose is the next one, not a rewind",
                       aside: "")
    }
}

// MARK: - The constants table
//
// Every number this render rests on, with what kind of claim it is and where it
// came from. A test fails the build if a measured one cites no year, or a
// modelled one does not say so.

enum LacEvidence: String {
    case measured = "MEASURED"
    case derived = "DERIVED"
    case model = "MODEL"
}

struct LacConstant {
    var name: String
    var value: String
    var evidence: LacEvidence
    var source: String
}

func lacConstants(_ scene: LacScene) -> [LacConstant] {
    func n(_ v: Float, _ d: Int = 2) -> String { String(format: "%.\(d)f", v) }
    let juers = "Juers, Heightman, Vasella, McCarter, Mackenzie, Withers & Matthews, "
              + "Biochemistry 40:14781 (2001)"
    var out: [LacConstant] = []

    // --- what the structures say -----------------------------------------
    out.append(LacConstant(
        name: "no structure of human lactase",
        value: "0 entries",
        evidence: .measured,
        source: "RCSB search for UniProt P09848 (lactase-phlorizin hydrolase), run by "
              + "Tools/build_lactase.py on every build, 2026; returns nothing. This is "
              + "why the render substitutes the E. coli enzyme."))
    out.append(LacConstant(
        name: "the enzyme drawn",
        value: "E. coli lacZ \u{03B2}-galactosidase, 1023 residues per subunit, tetramer",
        evidence: .measured,
        source: "PDB 1JZ7 at 1.50 \u{00C5}, " + juers))
    out.append(LacConstant(
        name: "heavy atoms in the tetramer",
        value: "\(scene.atoms.count)",
        evidence: .derived,
        source: "DERIVED: counted from PDB 1JZ7 with waters, DMSO, buffer and the bound "
              + "sugars removed (the sugars are animated instead)"))
    out.append(LacConstant(
        name: "van der Waals radii",
        value: "C 1.70, N 1.55, O 1.52, S 1.80 \u{00C5}",
        evidence: .measured,
        source: "Bondi, J. Phys. Chem. 68:441 (1964)"))
    out.append(LacConstant(
        name: "covalent radii, which decide what counts as a bond",
        value: "C 0.76, N 0.71, O 0.66, H 0.31 \u{00C5}",
        evidence: .measured,
        source: "Cordero et al., Dalton Trans. 2832 (2008)"))

    // --- the mechanism ----------------------------------------------------
    out.append(LacConstant(
        name: "Glu537 is the nucleophile",
        value: "C1\u{2013}OE2 = " + n(scene.anomeric.count > 1 ? scene.anomeric[1].bond : 1.45)
             + " \u{00C5} in the trapped intermediate",
        evidence: .measured,
        source: "PDB 1JZ2, the 2-F-galactosyl-enzyme, 2.10 \u{00C5}; nucleophile identified by "
              + "trapping in Gebler, Aebersold & Withers, J. Biol. Chem. 267:11126 (1992)"))
    out.append(LacConstant(
        name: "Glu461 is the acid and then the base",
        value: "holds the attacking water at " + n(scene.waterToGlu461) + " \u{00C5}, then the "
             + "product's new hydroxyl at " + n(scene.productO1ToGlu461) + " \u{00C5}",
        evidence: .measured,
        source: "PDB 1JZ2 and 1JZ7; " + juers))
    out.append(LacConstant(
        name: "the attacking water is in line with the bond it attacks",
        value: "cos = " + n(scene.waterCosToBond) + " to the C1\u{2013}OE2 axis, "
             + n(scene.waterToC1) + " \u{00C5} away",
        evidence: .derived,
        source: "DERIVED: measured from PDB 1JZ2's own water by build_lactase.py (1 would be "
              + "exactly behind the bond)"))
    out.append(LacConstant(
        name: "the magnesium is secondary",
        value: n(scene.metalToNucleophile) + " \u{00C5} from the nucleophile; held by Glu461, "
             + "His418 and Glu416",
        evidence: .measured,
        source: "PDB 1JZ7, coordination measured by build_lactase.py. It binds the ACID/BASE "
              + "rather than the substrate, which is why " + juers + " concluded Glu461 is far "
              + "better placed to help the leaving group than the metal is."))
    out.append(LacConstant(
        name: "two inversions, one retention",
        value: scene.anomeric.map { "\($0.state) \($0.config)" }.joined(separator: " \u{2192} "),
        evidence: .measured,
        source: "computed from the deposited coordinates of PDB 1JYN, 1JZ2 and 1JZ7 by "
              + "build_lactase.py; Koshland, Biol. Rev. 28:416 (1953) for the double "
              + "displacement"))
    out.append(LacConstant(
        name: "the transition-state mimic really is flat",
        value: "C1's oxygen lies " + n(scene.lactoneOutOfPlane) + " \u{00C5} out of the "
             + "O5\u{2013}C1\u{2013}C2 plane, against " + n(scene.sugarOutOfPlane)
             + " \u{00C5} for an ordinary sugar",
        evidence: .derived,
        source: "DERIVED: measured from PDB 1JZ5 (D-galactonolactone) and 1JZ7 by "
              + "build_lactase.py. sp\u{00B2} at C1 is the whole claim of a mimic."))

    // --- what moves and what does not -------------------------------------
    out.append(LacConstant(
        name: "the protein does not measurably move",
        value: "state to state " + n(scene.motion.worstCA, 2) + " \u{00C5} (chain) and "
             + n(scene.motion.worstSite, 2) + " \u{00C5} (site), against a noise floor of "
             + n(scene.motion.noiseCA, 2) + " and " + n(scene.motion.noiseSite, 2),
        evidence: .derived,
        source: "DERIVED: the floor is the spread among the four crystallographically "
              + "independent copies inside each entry, which were refined against one set of "
              + "data and so cannot differ for any reason but noise. Step 11's discipline."))
    out.append(LacConstant(
        name: "the sugar's slide from the shallow site to the deep one",
        value: n(scene.slide) + " \u{00C5}",
        evidence: .measured,
        source: "PDB 1JYN/1JYV/1JYW/1JZ8 against 1JZ5/1JZ2/1JZ7; the four shallow complexes "
              + "agree to 0.10 \u{00C5} and the three deep ones to 0.42 \u{00C5}, so the "
              + "separation is real. " + juers))
    out.append(LacConstant(
        name: "the tetramer, not a monomer",
        value: "the partner subunit's 272\u{2013}288 loop reaches "
             + n(scene.interfaceToSite, 1) + " \u{00C5} from the site's residues",
        evidence: .measured,
        source: "PDB 1JZ7 (1.50 \u{00C5}), " + juers + "; the contact distances measured by "
              + "build_lactase.py. The loop completes the WALL of the site at "
              + "van der Waals contact, but it comes no nearer than "
              + n(scene.interfaceToSubstrate, 1) + " \u{00C5} to the sugar and donates no "
              + "catalytic residue \u{2014} so 'the neighbour completes the active site' is "
              + "true of the pocket, not of the chemistry."))

    // --- the assumptions, named -------------------------------------------
    out.append(LacConstant(
        name: "1JYN, 1JYV, 1JYW and 1JZ8 are of a DISABLED enzyme",
        value: "E537Q",
        evidence: .model,
        source: "MODEL: the nucleophile was mutated to glutamine precisely so the substrate "
              + "would sit in the site uncut. Treating that pose as the working enzyme's "
              + "Michaelis complex is the standard assumption and it is still an assumption. "
              + "Worse: every shallow complex is the mutant and every deep one is wild type, "
              + "so 'substrate binds shallow first' cannot be separated from 'taking the "
              + "nucleophile away leaves it shallow'."))
    out.append(LacConstant(
        name: "1JZ2's sugar is fluorinated",
        value: "2-deoxy-2-fluoro",
        evidence: .model,
        source: "MODEL: the fluorine is the trap that made the intermediate live long enough "
              + "to crystallise. The 2-hydroxyl is drawn back at the fluorine's position "
              + "(C\u{2013}F 1.39 \u{00C5} against C\u{2013}O 1.43 \u{00C5}), which moves one "
              + "atom about 0.04 \u{00C5} and changes nothing at C1."))
    for (k, v) in scene.provenance.sorted(by: { $0.key < $1.key }) {
        let level: LacEvidence = v.hasPrefix("DERIVED") ? .derived : .model
        out.append(LacConstant(name: k, value: "\u{2014}", evidence: level, source: v))
    }
    out.append(LacConstant(
        name: "the frame rate and the loop",
        value: "\(lacFrameCount) frames at 12.5 fps = "
             + String(format: "%.1f", Double(lacFrameCount) / 12.5) + " s",
        evidence: .model,
        source: "MODEL: a choice. Real turnover is a few hundred per second, so this is "
              + "slowed by roughly seven orders of magnitude."))
    return out
}
