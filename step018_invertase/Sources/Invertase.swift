// The left panel: yeast invertase, space-filling, with a porthole cut over one
// active site and ball-and-stick chemistry going on inside it.
//
// The molecule comes from Resources/invertase.json, which Tools/build_invertase.py
// makes out of two crystal structures. Everything geometric here — the pose of
// the sucrose, the two inversions at the anomeric carbon, the length of the
// covalent bond to Asp23 — was computed there from coordinates. This file draws
// it and measures it back.

import Foundation
import simd

// MARK: - Loading

struct SceneAtom {
    var element: String
    var position: SIMD3<Float>
    var chain: String
    var seq: Int
    var res: String
    var name: String
    var t: Float              // 0 at the chain's N end, 1 at its C end
}

struct StateAtom {
    var element: String
    var name: String
    var position: SIMD3<Float>
}

/// One of the five states the catalytic cycle passes through. Each is a set of
/// coordinates, not a drawing.
struct CatalyticState {
    var name: String
    var fructosyl: [StateAtom]
    var glucosyl: [StateAtom]
    var fructosylBonds: [(String, String)]
    var glucosylBonds: [(String, String)]
    var esterBond: Bool               // is the fructosyl covalently on the enzyme?
    var water: SIMD3<Float>?

    func atom(_ name: String, in group: [StateAtom]) -> SIMD3<Float>? {
        for a in group where a.name == name { return a.position }
        return nil
    }
    var fructosylC2: SIMD3<Float>? { atom("C2", in: fructosyl) }
}

struct Catalytic {
    var nucleophilePDB: Int
    var nucleophilePaper: Int
    var nucleophileOxygen: String
    var acidBasePDB: Int
    var acidBasePaper: Int
    var acidBaseOxygen: String
    var stabiliserPDB: Int
    var stabiliserPaper: Int
}

struct Template {
    var pdb: String
    var resolution: Double
    var organism: String
    var identityPercent: Double
    var alignedResidues: Int
    var siteResidues: Int
    var siteRMSD: Double
    var wholeChainRMSD: Double
    var source: String
}

struct InvertaseScene {
    var pdb: String
    var resolution: Double
    var organism: String
    var gene: String
    var family: String
    var source: String
    var biologicalUnit: String
    var chains: [String]
    var dimerChains: [String]
    var atoms: [SceneAtom]
    var dimerAtomIndices: [Int]
    var siteCentre: SIMD3<Float>
    var siteAxis: SIMD3<Float>
    var dimerCentre: SIMD3<Float>
    var nucleophileOxygen: SIMD3<Float>
    var acidOxygen: SIMD3<Float>
    var catalytic: Catalytic
    var template: Template
    var states: [String: CatalyticState]
    var attackAngle: Double
    var attackDistance: Double
    var acidDistance: Double
    var esterLength: Double
    var anomericVolumes: [String: Double]
    var extent: SIMD3<Float>
}

enum SceneError: Error, CustomStringConvertible {
    case badFile(String)
    var description: String {
        switch self {
        case .badFile(let detail):
            return "couldn't read the structure: \(detail) — run `make structure` first"
        }
    }
}

private func vec(_ any: Any?) -> SIMD3<Float>? {
    guard let n = any as? [NSNumber], n.count == 3 else { return nil }
    return SIMD3(n[0].floatValue, n[1].floatValue, n[2].floatValue)
}

private func stateAtoms(_ any: Any?) -> [StateAtom] {
    guard let list = any as? [[String: Any]] else { return [] }
    var out: [StateAtom] = []
    for a in list {
        guard let name = a["name"] as? String, let p = vec(a["pos"]) else { continue }
        out.append(StateAtom(element: a["el"] as? String ?? "C", name: name, position: p))
    }
    return out
}

private func pairs(_ any: Any?) -> [(String, String)] {
    guard let list = any as? [[String]] else { return [] }
    return list.compactMap { $0.count == 2 ? ($0[0], $0[1]) : nil }
}

func loadScene(from url: URL) throws -> InvertaseScene {
    let data = try Data(contentsOf: url)
    guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
          let enzyme = root["enzyme"] as? [String: Any],
          let numbering = root["numbering"] as? [String: Any],
          let templateInfo = root["template"] as? [String: Any],
          let catalysis = root["catalysis"] as? [String: Any],
          let rawAtoms = root["atoms"] as? [[String: Any]],
          let rawStates = root["states"] as? [String: Any] else {
        throw SceneError.badFile("missing sections")
    }

    // Residues per chain, so each chain can be coloured from its own N end.
    var chainResidues: [String: Set<Int>] = [:]
    for a in rawAtoms {
        guard let ch = a["ch"] as? String, let seq = a["seq"] as? Int else { continue }
        chainResidues[ch, default: []].insert(seq)
    }
    var order: [String: [Int: Float]] = [:]
    for (ch, set) in chainResidues {
        let sorted = set.sorted()
        var map: [Int: Float] = [:]
        let span: Float = Float(max(sorted.count - 1, 1))
        for (i, s) in sorted.enumerated() { map[s] = Float(i) / span }
        order[ch] = map
    }

    var atoms: [SceneAtom] = []
    atoms.reserveCapacity(rawAtoms.count)
    var lo = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
    var hi = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)
    for a in rawAtoms {
        guard let p = vec(a["pos"]), let ch = a["ch"] as? String,
              let seq = a["seq"] as? Int else { throw SceneError.badFile("bad atom") }
        lo = simd_min(lo, p)
        hi = simd_max(hi, p)
        atoms.append(SceneAtom(element: a["el"] as? String ?? "C", position: p, chain: ch,
                               seq: seq, res: a["res"] as? String ?? "",
                               name: a["name"] as? String ?? "",
                               t: order[ch]?[seq] ?? 0))
    }

    let dimerChains = (enzyme["dimerChains"] as? [String]) ?? ["A", "B"]
    let dimerSet = Set(dimerChains)
    var dimerIndices: [Int] = []
    for (i, a) in atoms.enumerated() where dimerSet.contains(a.chain) { dimerIndices.append(i) }

    func group(_ key: String) -> [String: Any] { (numbering[key] as? [String: Any]) ?? [:] }
    let nuc = group("nucleophile"), acid = group("acidBase"), stab = group("stabiliser")
    let catalytic = Catalytic(
        nucleophilePDB: nuc["pdbSeq"] as? Int ?? 0,
        nucleophilePaper: nuc["paperSeq"] as? Int ?? 0,
        nucleophileOxygen: nuc["oxygen"] as? String ?? "OD2",
        acidBasePDB: acid["pdbSeq"] as? Int ?? 0,
        acidBasePaper: acid["paperSeq"] as? Int ?? 0,
        acidBaseOxygen: acid["oxygen"] as? String ?? "OE1",
        stabiliserPDB: stab["pdbSeq"] as? Int ?? 0,
        stabiliserPaper: stab["paperSeq"] as? Int ?? 0)

    let template = Template(
        pdb: templateInfo["pdb"] as? String ?? "",
        resolution: (templateInfo["resolutionAngstroms"] as? NSNumber)?.doubleValue ?? 0,
        organism: templateInfo["organism"] as? String ?? "",
        identityPercent: (templateInfo["identityPercent"] as? NSNumber)?.doubleValue ?? 0,
        alignedResidues: templateInfo["alignedResidues"] as? Int ?? 0,
        siteResidues: templateInfo["siteResidues"] as? Int ?? 0,
        siteRMSD: (templateInfo["siteRmsdAngstroms"] as? NSNumber)?.doubleValue ?? 0,
        wholeChainRMSD: (templateInfo["wholeChainRmsdAngstroms"] as? NSNumber)?.doubleValue ?? 0,
        source: templateInfo["source"] as? String ?? "")

    var states: [String: CatalyticState] = [:]
    for (name, raw) in rawStates {
        guard let s = raw as? [String: Any] else { continue }
        states[name] = CatalyticState(
            name: name,
            fructosyl: stateAtoms(s["fructosyl"]),
            glucosyl: stateAtoms(s["glucosyl"]),
            fructosylBonds: pairs(s["fructosylBonds"]),
            glucosylBonds: pairs(s["glucosylBonds"]),
            esterBond: s["esterBond"] as? Bool ?? false,
            water: vec(s["water"]))
    }

    // The two mechanism mutations. `invert` lets the water attack the face the
    // nucleophile is still on, so the product comes out α and the enzyme really
    // would be an inverting one — the mistake its name invites. `ester` leaves
    // the fructosyl unit where the Michaelis complex put it while still calling
    // it covalent, so the bond is drawn across 2.75 Å of nothing.
    if mutation == .invert, let covalent = states["covalent"], var product = states["product"] {
        product.fructosyl = covalent.fructosyl
        if let c2 = covalent.atom("C2", in: covalent.fructosyl) {
            // The water comes in on the face the nucleophile is already on
            // instead of the far one, so the carbon never flips back and the
            // product is α. That is a single displacement — an INVERTING
            // enzyme, which is what the name invertase invites you to expect
            // and what GH32 is not.
            let toward: SIMD3<Float> =
                simd_normalize((vec(root["nucleophileOxygen"]) ?? .zero) - c2)
            product.fructosyl.append(StateAtom(element: "O", name: "O2",
                                               position: c2 + toward * 1.41))
        }
        states["product"] = product
    }
    if mutation == .ester, let michaelis = states["michaelis"] {
        for name in ["covalent", "hydrolysis"] {
            guard var s = states[name] else { continue }
            s.fructosyl = michaelis.fructosyl.filter { $0.name != "O2" }
            states[name] = s
        }
    }

    var volumes: [String: Double] = [:]
    if let raw = catalysis["anomericVolumes"] as? [String: Any] {
        for (k, v) in raw { volumes[k] = (v as? NSNumber)?.doubleValue ?? 0 }
    }

    return InvertaseScene(
        pdb: enzyme["pdb"] as? String ?? "",
        resolution: (enzyme["resolutionAngstroms"] as? NSNumber)?.doubleValue ?? 0,
        organism: enzyme["organism"] as? String ?? "",
        gene: enzyme["gene"] as? String ?? "",
        family: enzyme["family"] as? String ?? "",
        source: enzyme["source"] as? String ?? "",
        biologicalUnit: enzyme["biologicalUnit"] as? String ?? "",
        chains: (enzyme["chains"] as? [String]) ?? [],
        dimerChains: dimerChains,
        atoms: atoms,
        dimerAtomIndices: dimerIndices,
        siteCentre: vec(root["siteCentre"]) ?? .zero,
        siteAxis: vec(root["siteAxis"]) ?? SIMD3(0, 0, 1),
        dimerCentre: vec(root["dimerCentre"]) ?? .zero,
        nucleophileOxygen: vec(root["nucleophileOxygen"]) ?? .zero,
        acidOxygen: vec(root["acidOxygen"]) ?? .zero,
        catalytic: catalytic,
        template: template,
        states: states,
        attackAngle: (catalysis["attackAngleDegrees"] as? NSNumber)?.doubleValue ?? 0,
        attackDistance: (catalysis["attackDistanceAngstroms"] as? NSNumber)?.doubleValue ?? 0,
        acidDistance: (catalysis["acidToLeavingOxygenAngstroms"] as? NSNumber)?.doubleValue ?? 0,
        esterLength: (catalysis["esterBondAngstroms"] as? NSNumber)?.doubleValue ?? 0,
        anomericVolumes: volumes,
        extent: hi - lo)
}

// MARK: - Radii and colour

/// Van der Waals radii in ångströms, Bondi, J. Phys. Chem. 68:441 (1964).
/// These are what makes a space-filling model: each atom takes up the room its
/// electron cloud occupies, so bonded neighbours overlap and merge into a solid.
func vdwRadius(_ element: String) -> Float {
    switch element {
    case "C": return 1.70
    case "N": return 1.55
    case "O": return 1.52
    case "S": return 1.80
    case "SE": return 1.90
    case "P": return 1.80
    case "H": return 1.20
    default: return 1.70
    }
}

/// Radii for ball-and-stick, where the point is connectivity rather than bulk.
/// Same proportions as steps 6 to 9.
func ballRadius(_ element: String) -> Float {
    switch element {
    case "C": return 0.55
    case "N": return 0.54
    case "O": return 0.52
    case "S": return 0.62
    default: return 0.36
    }
}

let bondRadius: Float = 0.24

/// The site is drawn at a third of the scale the wall around it is: a van der
/// Waals sphere is 1.70 Å and a ball-and-stick one 0.55, and through a porthole
/// 30 Å across that is a few pixels. Ball-and-stick radii are a convention
/// rather than a measurement — nothing about a carbon atom is 0.55 Å — so they
/// are scaled up here until the chemistry is legible. The POSITIONS are not
/// touched, and every distance the tests measure is between positions.
let siteBallScale: Float = 1.5
let siteBondRadius: Float = 0.34

/// Position along a chain as a colour: the muted spectrum step 9 used, deep
/// blue at the N end through teal and olive to warm red at the C end. Here it
/// lets you count the five blades of the β-propeller by the order the colours
/// come round.
func spectrumColour(_ t: Float) -> SIMD3<Float> {
    let stops: [SIMD3<Float>] = [
        SIMD3(0.22, 0.31, 0.60),
        SIMD3(0.24, 0.50, 0.62),
        SIMD3(0.33, 0.58, 0.52),
        SIMD3(0.55, 0.60, 0.41),
        SIMD3(0.70, 0.52, 0.36),
        SIMD3(0.68, 0.36, 0.34),
    ]
    let x: Float = min(max(t, 0), 1) * Float(stops.count - 1)
    let i: Int = min(Int(x), stops.count - 2)
    let f: Float = x - Float(i)
    return stops[i] + (stops[i + 1] - stops[i]) * f
}

let partnerColour = SIMD3<Float>(0.40, 0.44, 0.52)     // the other half of the dimer
let otherChainColour = SIMD3<Float>(0.35, 0.39, 0.47)  // the six chains of the other dimers
let nucleophileColour = SIMD3<Float>(0.96, 0.44, 0.36) // Asp23, the one that attacks
let acidBaseColour = SIMD3<Float>(0.47, 0.78, 0.98)    // Glu204, the one that protonates
let stabiliserColour = SIMD3<Float>(0.72, 0.62, 0.92)  // Asp152, which holds the transition state
let glucoseColour = SIMD3<Float>(0.62, 0.83, 0.52)
let fructoseColour = SIMD3<Float>(0.98, 0.76, 0.34)
let waterColour = SIMD3<Float>(0.55, 0.83, 0.92)
let stickColour = SIMD3<Float>(0.72, 0.74, 0.70)

// MARK: - The porthole

/// How much of each atom survives the cut, 0 (gone) to 1 (whole).
///
/// Step 9's technique, and the reason it is worth repeating: a half-space cut
/// takes away half the molecule and what is left stops reading as a fold. A
/// round window keeps the rim, so you can still see what you are looking into.
///
/// Nothing is CLIPPED. Atoms inside the window shrink to nothing instead, so
/// every surface in the frame is still a whole sphere with its normal facing
/// outwards. That is why there is no cut face to cap and no backface to see
/// through — a property the tests check by casting rays down the hole.
func portholeFactors(_ atoms: [SceneAtom], keep: [Float], opening: Float,
                     centre: SIMD3<Float>, axis: SIMD3<Float>, radius: Float) -> [Float] {
    var out = keep
    if opening <= 0 { return out }
    let windowRadius: Float = radius * opening
    let softness: Float = 2.6
    // Just past the site, so nothing behind it is taken away with the wall.
    let backPlane: Float = 1.2
    for (i, atom) in atoms.enumerated() {
        if out[i] <= 0 { continue }
        let offset: SIMD3<Float> = atom.position - centre
        let along: Float = simd_dot(offset, axis)
        if along <= backPlane { continue }                 // behind the site: keep
        let lateralVector: SIMD3<Float> = offset - axis * along
        let lateral: Float = simd_length(lateralVector)
        let x: Float = (lateral - windowRadius) / softness
        out[i] = min(out[i], min(max(x, 0), 1))
    }
    return out
}

// MARK: - The catalytic cycle

/// The five states, and the order they happen in. `free` appears at both ends
/// because that is the whole point of a catalyst: the cycle closes on it.
let cycleOrder = ["free", "michaelis", "covalent", "hydrolysis", "product", "free"]

/// Which two states a point in the cycle lies between, and how far.
/// `u` runs 0 to 1 over one complete turnover.
func cyclePosition(_ u: Double) -> (from: String, to: String, blend: Double, phase: Int) {
    let steps: Int = cycleOrder.count - 1
    let scaled: Double = min(max(u, 0), 0.999999) * Double(steps)
    let i: Int = Int(scaled)
    let f: Double = scaled - Double(i)
    // Hold each state still for the first part of its span and move in the
    // second, so the eye gets to read each one before it changes.
    let hold: Double = 0.45
    let moving: Double = f <= hold ? 0 : (f - hold) / (1 - hold)
    let eased: Double = moving * moving * (3 - 2 * moving)
    return (cycleOrder[i], cycleOrder[i + 1], eased, i)
}

/// The covalent bond to Asp23 is drawn only while the fructosyl unit is
/// actually esterified — that is, in the two states between the glucose leaving
/// and the water arriving, and in neither of the two on either side. Drawn
/// during the morph *into* the covalent state it would be a bond forming out of
/// nothing across 1.3 Å of empty space.
func esterBonded(_ position: (from: String, to: String, blend: Double, phase: Int)) -> Bool {
    // Moving: the only leg where the fructosyl is esterified for the whole of
    // it is covalent -> hydrolysis, where the sugar does not move at all and
    // only the water comes in.
    if position.blend > 1e-6 { return position.from == "covalent" && position.to == "hydrolysis" }
    // Standing still at a state that is esterified.
    return position.from == "covalent" || position.from == "hydrolysis"
}

/// The atoms of the site at a point in the cycle, interpolated between the two
/// states it lies between. Atoms present in only one of them fade in or out by
/// radius, never by transparency — there is none in a ray tracer of solids.
struct SiteGeometry {
    var spheres: [(position: SIMD3<Float>, radius: Float, colour: SIMD3<Float>)] = []
    var sticks: [(a: SIMD3<Float>, b: SIMD3<Float>, colour: SIMD3<Float>)] = []
    var fructosylC2: SIMD3<Float>?
    var esterLength: Float?
    var glucoseFraction: Float = 0      // how much glucose is in the site
    var fructoseFraction: Float = 0
}

private func lerp(_ a: SIMD3<Float>, _ b: SIMD3<Float>, _ t: Float) -> SIMD3<Float> {
    a + (b - a) * t
}

private func named(_ atoms: [StateAtom]) -> [String: SIMD3<Float>] {
    var out: [String: SIMD3<Float>] = [:]
    for a in atoms { out[a.name] = a.position }
    return out
}

func siteGeometry(_ scene: InvertaseScene, at u: Double, scale: Float) -> SiteGeometry {
    let position = cyclePosition(u)
    guard let from = scene.states[position.from], let to = scene.states[position.to] else {
        return SiteGeometry()
    }
    let t = Float(position.blend)
    var out = SiteGeometry()

    func blendGroup(_ a: [StateAtom], _ b: [StateAtom], colour: SIMD3<Float>,
                    bondsA: [(String, String)], bondsB: [(String, String)]) -> Float {
        let mapA = named(a), mapB = named(b)
        var here: [String: SIMD3<Float>] = [:]
        var weight: [String: Float] = [:]
        for (name, pa) in mapA {
            if let pb = mapB[name] {
                here[name] = lerp(pa, pb, t)
                weight[name] = 1
            } else {
                here[name] = pa
                weight[name] = 1 - t          // leaving: shrink away
            }
        }
        for (name, pb) in mapB where mapA[name] == nil {
            here[name] = pb
            weight[name] = t                   // arriving: grow in
        }
        for (name, p) in here {
            let w: Float = weight[name] ?? 1
            if w <= 0.02 { continue }
            let element: String = name.hasPrefix("O") ? "O" : "C"
            let r: Float = ballRadius(element) * w * siteBallScale * scale
            out.spheres.append((p * scale, r, colour))
        }
        let bonds = t < 0.5 ? bondsA : bondsB
        for (x, y) in bonds {
            guard let pa = here[x], let pb = here[y] else { continue }
            let w: Float = min(weight[x] ?? 1, weight[y] ?? 1)
            if w <= 0.05 { continue }
            out.sticks.append((pa * scale, pb * scale, stickColour))
        }
        var present: Float = 0
        for (_, w) in weight { present = max(present, w) }
        return present
    }

    out.fructoseFraction = blendGroup(from.fructosyl, to.fructosyl, colour: fructoseColour,
                                      bondsA: from.fructosylBonds, bondsB: to.fructosylBonds)
    out.glucoseFraction = blendGroup(from.glucosyl, to.glucosyl, colour: glucoseColour,
                                     bondsA: from.glucosylBonds, bondsB: to.glucosylBonds)

    // The bridging bond to the glucose, while there still is one.
    if let fo = named(from.fructosyl)["O2"], let gc = named(from.glucosyl)["C1"],
       position.from == "michaelis" {
        let toF = named(to.fructosyl)["O2"] ?? fo
        let toG = named(to.glucosyl)["C1"] ?? gc
        if position.blend < 0.35 {
            out.sticks.append((lerp(fo, toF, t) * scale, lerp(gc, toG, t) * scale, stickColour))
        }
    }

    // The water that breaks the intermediate.
    let waterFrom = from.water, waterTo = to.water
    if waterFrom != nil || waterTo != nil {
        let p = lerp(waterFrom ?? waterTo!, waterTo ?? waterFrom!, t)
        var w: Float = 1
        if waterFrom == nil { w = t }
        if waterTo == nil { w = 1 - t }
        if w > 0.02 {
            let r: Float = ballRadius("O") * w * 1.1 * siteBallScale * scale
            out.spheres.append((p * scale, r, waterColour))
        }
    }

    // The covalent bond to Asp23, and its length, measured not assumed.
    let c2From = named(from.fructosyl)["C2"]
    let c2To = named(to.fructosyl)["C2"]
    if let a = c2From, let b = c2To {
        let c2 = lerp(a, b, t)
        out.fructosylC2 = c2
        if esterBonded(position) {
            out.esterLength = simd_length(c2 - scene.nucleophileOxygen)
            out.sticks.append((c2 * scale, scene.nucleophileOxygen * scale, nucleophileColour))
        }
    } else if let a = c2From ?? c2To {
        out.fructosylC2 = a
    }
    return out
}

/// The three catalytic side chains, drawn ball-and-stick so they can be told
/// apart from the wall they sit in.
func catalyticSticks(_ scene: InvertaseScene, chain: String, scale: Float,
                     strength: Float) -> ([(SIMD3<Float>, Float, SIMD3<Float>)],
                                          [(SIMD3<Float>, SIMD3<Float>, SIMD3<Float>)]) {
    var spheres: [(SIMD3<Float>, Float, SIMD3<Float>)] = []
    var sticks: [(SIMD3<Float>, SIMD3<Float>, SIMD3<Float>)] = []
    if strength <= 0.02 { return (spheres, sticks) }
    let wanted: [Int: SIMD3<Float>] = [
        scene.catalytic.nucleophilePDB: nucleophileColour,
        scene.catalytic.acidBasePDB: acidBaseColour,
        scene.catalytic.stabiliserPDB: stabiliserColour,
    ]
    var byResidue: [Int: [SceneAtom]] = [:]
    for a in scene.atoms where a.chain == chain {
        if wanted[a.seq] != nil && a.name != "N" && a.name != "C" && a.name != "O" {
            byResidue[a.seq, default: []].append(a)
        }
    }
    for (seq, group) in byResidue {
        let colour = wanted[seq] ?? stickColour
        for a in group {
            let r: Float = ballRadius(a.element) * strength * siteBallScale * scale
            spheres.append((a.position * scale, r, colour))
        }
        for i in 0..<group.count {
            for j in (i + 1)..<group.count {
                let d: Float = simd_distance(group[i].position, group[j].position)
                if d < 1.9 {
                    sticks.append((group[i].position * scale, group[j].position * scale, colour))
                }
            }
        }
    }
    return (spheres, sticks)
}

// MARK: - Measuring the chemistry back out of the drawing

/// The signed volume whose sign says which face of the furanose ring the
/// exocyclic substituent sits on. Positive is β, calibrated in the builder
/// against 6S1T's own ligand, which the PDB annotates β-D-fructofuranose.
func anomericVolume(ringOxygen: SIMD3<Float>, ringCarbon: SIMD3<Float>,
                    exocyclic: SIMD3<Float>, anomeric: SIMD3<Float>) -> Float {
    let a: SIMD3<Float> = ringOxygen - anomeric
    let b: SIMD3<Float> = ringCarbon - anomeric
    let d: SIMD3<Float> = exocyclic - anomeric
    return simd_dot(simd_cross(a, b), d)
}

/// The configuration at C2 in one state, measured from that state's own
/// coordinates. In `covalent` and `hydrolysis` the exocyclic substituent is the
/// enzyme's own carboxylate oxygen; everywhere else it is the sugar's O2.
func anomericConfiguration(_ scene: InvertaseScene, state name: String) -> Float? {
    guard let state = scene.states[name] else { return nil }
    let map = named(state.fructosyl)
    guard let c2 = map["C2"], let c3 = map["C3"], let o5 = map["O5"] else { return nil }
    let exocyclic: SIMD3<Float>
    if let o2 = map["O2"] {
        exocyclic = o2
    } else if state.esterBond {
        exocyclic = scene.nucleophileOxygen
    } else {
        return nil
    }
    return anomericVolume(ringOxygen: o5, ringCarbon: c3, exocyclic: exocyclic, anomeric: c2)
}

/// The structural half of the constants table: what came out of the two
/// crystal structures, and what was built on top of them. The tests walk this
/// with the chemistry's table — nothing without a source, nothing claiming to
/// be MEASURED without a year, and the substrate's pose marked MODEL.
func structureConstants(_ scene: InvertaseScene) -> [Constant] {
    let t = scene.template
    return [
        Constant(name: "resolution", value: scene.resolution, unit: "Å", evidence: .measured,
                 source: scene.source),
        Constant(name: "chains in the file", value: Double(scene.chains.count), unit: "chains",
                 evidence: .measured, source: scene.source + "; " + scene.biologicalUnit),
        Constant(name: "heavy atoms", value: Double(scene.atoms.count), unit: "atoms",
                 evidence: .measured, source: scene.source),
        Constant(name: "nucleophile", value: Double(scene.catalytic.nucleophilePaper), unit: "Asp",
                 evidence: .measured,
                 source: "the NDPNG-motif aspartate, Sainz-Polo et al., J. Biol. Chem. 288:9755 (2013); "
                       + "found in the coordinates by its motif, not by its number"),
        Constant(name: "acid/base", value: Double(scene.catalytic.acidBasePaper), unit: "Glu",
                 evidence: .measured,
                 source: "the EC-motif glutamate, Sainz-Polo et al., J. Biol. Chem. 288:9755 (2013)"),
        Constant(name: "transition-state stabiliser", value: Double(scene.catalytic.stabiliserPaper),
                 unit: "Asp", evidence: .measured,
                 source: "the RDP-motif aspartate, Sainz-Polo et al., J. Biol. Chem. 288:9755 (2013)"),
        Constant(name: "crystal/paper offset", value: Double(scene.catalytic.nucleophilePaper
                                                             - scene.catalytic.nucleophilePDB),
                 unit: "residues", evidence: .derived,
                 source: "the paper numbers the mature chain from its serine, the crystal from the "
                       + "methionine after it; checked against all three motifs in build_invertase.py"),
        Constant(name: "template identity", value: t.identityPercent, unit: "%", evidence: .derived,
                 source: "Needleman-Wunsch over \(t.alignedResidues) residues against " + t.source),
        Constant(name: "site superposition", value: t.siteRMSD, unit: "Å", evidence: .derived,
                 source: "Kabsch on \(t.siteResidues) active-site Cα; Kabsch, Acta Cryst. A32:922 (1976)"),
        Constant(name: "sucrose pose", value: scene.attackDistance, unit: "Å nucleophile–C2",
                 evidence: .model,
                 source: "MODEL: no structure of sucrose bound to invertase exists, so the pose is "
                       + "carried over from " + t.source + " by that superposition"),
        Constant(name: "attack angle", value: scene.attackAngle, unit: "deg", evidence: .derived,
                 source: "O2–C2–Oδ measured off the transferred pose; nothing was aimed at it"),
        Constant(name: "acid to leaving oxygen", value: scene.acidDistance, unit: "Å",
                 evidence: .derived, source: "measured off the transferred pose"),
        Constant(name: "fructosyl–enzyme ester", value: scene.esterLength, unit: "Å",
                 evidence: .model,
                 source: "MODEL: built at the small-molecule C(sp3)–O mean, Allen et al., "
                       + "J. Chem. Soc. Perkin Trans. II S1 (1987)"),
        Constant(name: "turnovers per loop", value: Double(4), unit: "cycles", evidence: .model,
                 source: "MODEL: a display mapping. One real site does not empty a flask in four "
                       + "turnovers; what is enforced is that the reading always matches the "
                       + "molecules drawn"),
    ]
}

/// The heavy atoms in a state, counted off the coordinates.
func heavyAtomCount(_ scene: InvertaseScene, state name: String) -> Int {
    guard let s = scene.states[name] else { return 0 }
    return s.fructosyl.count + s.glucosyl.count + (s.water == nil ? 0 : 1)
}
