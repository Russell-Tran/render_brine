// The upper arch as numbers: sixteen maxillary teeth, the wisdom teeth
// included, where each one sits, the palate they ring, and what colour each
// tissue is. Nothing here touches the GPU; it is the part a test can read
// against the textbooks.
//
// Step 20's Anatomy.swift, copied and turned into the upper jaw. Its layout,
// its arch construction and its colour science are kept; the teeth are the
// maxillary ones, the tongue gives way to the hard palate, and every tooth now
// carries its Universal number.
//
// Millimetres throughout, in the ARCH'S OWN FRAME, which is step 20's: y = 0 is
// the occlusal plane and the crowns stand on the +y side of their gum, x runs
// across the mouth (+x is the patient's RIGHT), z runs back towards the
// throat, and the midline between the two central incisors is x = 0, z = 0.
// The upper arch hangs from the skull, so the renderer turns this frame over
// (y → −y) when it draws it: the crowns point down in the picture. Turning it
// over about the horizontal leaves x and z alone, so +x stays the patient's
// right. The renderer is right-handed (the camera's right is forward × up), so
// facing the patient from the front, +x lands on the VIEWER'S LEFT — step 21
// caught step 20's comments with this backwards (d3d4b89).

import Foundation
import simd

// MARK: - what to break, for the mutation check

/// Anatomy mutants, read from UPPER_MUTANT. Each must make a test fail.
enum ArchMutant: String {
    case none
    case mandibularSizes    // step 20's lower-tooth table in place of the upper one
    case mirrored           // the Universal numbers put on the wrong side
    case noRugae            // a smooth palate
}

let archMutant: ArchMutant = {
    let raw: String = ProcessInfo.processInfo.environment["UPPER_MUTANT"] ?? ""
    return ArchMutant(rawValue: raw) ?? .none
}()

// MARK: - the teeth

/// What a crown's top looks like. The raw value goes straight into the kernel.
enum CrownKind: Int, CaseIterable {
    case incisor = 0        // a straight cutting edge
    case canine = 1         // one pointed cusp
    case firstPremolar = 2  // buccal cusp about 1 mm longer than the lingual
    case secondPremolar = 3 // two cusps of nearly equal length
    case firstMolar = 4     // rhomboid, four cusps, the mesiolingual the largest
    case secondMolar = 5    // the same with a smaller distolingual cusp
    case thirdMolar = 6     // heart-shaped, three cusps
}

/// One maxillary tooth, in the numbers the dental anatomy tables give.
///
/// Every value is from the "Measurement Table" at the head of each tooth's
/// section in Nelson & Ash, *Wheeler's Dental Anatomy, Physiology and
/// Occlusion*, 9th edition (Saunders 2010): the "dimensions suggested for
/// carving technique", in millimetres. CHECKED against the table images as
/// published online with the book's chapters (pocketdentistry.com, chapters
/// 6 "The Permanent Maxillary Incisors", 8 "The Permanent Canines", 9 "The
/// Permanent Maxillary Premolars" and 11 "The Permanent Maxillary Molars",
/// files B9781416062097000064_t0010/_t0020, …000088_t0010, …00009X_t0010/
/// _t0020 and …000118_t0010/_t0020/_t0030), read 2026-09-28.
///
/// "Width" is mesiodistal — along the arch; "depth" is labiolingual or
/// buccolingual — across it. "cejCurve" is the table's curvature of the
/// cervical line on the MESIAL side, as step 20 used it: how far the
/// cementoenamel junction rises between teeth, which is what scallops the gum.
struct ToothSpec: Equatable {
    var name: String
    var kind: CrownKind
    var crownHeight: Float      // cervico-incisal / cervico-occlusal
    var width: Float            // mesiodistal, at the contact areas
    var cervicalWidth: Float    // mesiodistal, at the cervical line
    var depth: Float            // labiolingual, at the height of contour
    var cervicalDepth: Float    // labiolingual, at the cervical line
    var cejCurve: Float         // mesial curvature of the cervical line
}

/// Central incisor to third molar, one side. Wheeler's rows, 9th ed.:
///
///   tooth            crown  MD    MD cerv  LL    LL cerv  CEJ mesial
///   central incisor  10.5   8.5   7.0      7.0   6.0      3.5
///   lateral incisor   9.0   6.5   5.0      6.0   5.0      3.0
///   canine           10.0   7.5   5.5      8.0   7.0      2.5
///   first premolar    8.5   7.0   5.0      9.0   8.0      1.0
///   second premolar   8.5   7.0   5.0      9.0   8.0      1.0
///   first molar       7.5  10.0   8.0     11.0  10.0      1.0
///   second molar      7.0   9.0   7.0     11.0  10.0      1.0
///   third molar       6.5   8.5   6.5     10.0   9.5      1.0
///
/// Beside step 20's mandibular table the difference is plain: the upper
/// central is 8.5 mm wide against the lower's 5.0, the lateral 6.5 against 5.5,
/// and the upper molars are broader buccolingually (11.0) than long.
let maxillaryTeeth: [ToothSpec] = [
    ToothSpec(name: "central incisor", kind: .incisor, crownHeight: 10.5, width: 8.5,
              cervicalWidth: 7.0, depth: 7.0, cervicalDepth: 6.0, cejCurve: 3.5),
    ToothSpec(name: "lateral incisor", kind: .incisor, crownHeight: 9.0, width: 6.5,
              cervicalWidth: 5.0, depth: 6.0, cervicalDepth: 5.0, cejCurve: 3.0),
    ToothSpec(name: "canine", kind: .canine, crownHeight: 10.0, width: 7.5,
              cervicalWidth: 5.5, depth: 8.0, cervicalDepth: 7.0, cejCurve: 2.5),
    ToothSpec(name: "first premolar", kind: .firstPremolar, crownHeight: 8.5, width: 7.0,
              cervicalWidth: 5.0, depth: 9.0, cervicalDepth: 8.0, cejCurve: 1.0),
    ToothSpec(name: "second premolar", kind: .secondPremolar, crownHeight: 8.5, width: 7.0,
              cervicalWidth: 5.0, depth: 9.0, cervicalDepth: 8.0, cejCurve: 1.0),
    ToothSpec(name: "first molar", kind: .firstMolar, crownHeight: 7.5, width: 10.0,
              cervicalWidth: 8.0, depth: 11.0, cervicalDepth: 10.0, cejCurve: 1.0),
    ToothSpec(name: "second molar", kind: .secondMolar, crownHeight: 7.0, width: 9.0,
              cervicalWidth: 7.0, depth: 11.0, cervicalDepth: 10.0, cejCurve: 1.0),
    ToothSpec(name: "third molar", kind: .thirdMolar, crownHeight: 6.5, width: 8.5,
              cervicalWidth: 6.5, depth: 10.0, cervicalDepth: 9.5, cejCurve: 1.0),
]

/// Step 20's mandibular table, with step 21's third molar, for the
/// `mandibularSizes` mutant only: the arch built from the wrong jaw's teeth.
/// (The third-molar row step 20 marked UNVERIFIED is on the same site,
/// chapter 12's …00012X_t0030, and reads 7.0, 10.0, 7.5, 9.5, 9.0, 1.0 — as
/// step 20 quoted it.)
let mandibularTeethForMutant: [ToothSpec] = [
    ToothSpec(name: "central incisor", kind: .incisor, crownHeight: 9.0, width: 5.0,
              cervicalWidth: 3.5, depth: 6.0, cervicalDepth: 5.3, cejCurve: 3.0),
    ToothSpec(name: "lateral incisor", kind: .incisor, crownHeight: 9.5, width: 5.5,
              cervicalWidth: 4.0, depth: 6.5, cervicalDepth: 5.8, cejCurve: 3.0),
    ToothSpec(name: "canine", kind: .canine, crownHeight: 11.0, width: 7.0,
              cervicalWidth: 5.5, depth: 7.5, cervicalDepth: 7.0, cejCurve: 2.5),
    ToothSpec(name: "first premolar", kind: .firstPremolar, crownHeight: 8.5, width: 7.0,
              cervicalWidth: 5.0, depth: 7.5, cervicalDepth: 6.5, cejCurve: 1.0),
    ToothSpec(name: "second premolar", kind: .secondPremolar, crownHeight: 8.0, width: 7.0,
              cervicalWidth: 5.0, depth: 8.0, cervicalDepth: 7.0, cejCurve: 1.0),
    ToothSpec(name: "first molar", kind: .firstMolar, crownHeight: 7.5, width: 11.0,
              cervicalWidth: 9.0, depth: 10.5, cervicalDepth: 9.0, cejCurve: 1.0),
    ToothSpec(name: "second molar", kind: .secondMolar, crownHeight: 7.0, width: 10.5,
              cervicalWidth: 8.0, depth: 10.0, cervicalDepth: 9.0, cejCurve: 1.0),
    ToothSpec(name: "third molar", kind: .thirdMolar, crownHeight: 7.0, width: 10.0,
              cervicalWidth: 7.5, depth: 9.5, cervicalDepth: 9.0, cejCurve: 1.0),
]

/// The table the arch is built from: the maxillary one, except under the mutant.
let toothTable: [ToothSpec] = archMutant == .mandibularSizes ? mandibularTeethForMutant : maxillaryTeeth

// MARK: - the arch

// Bonwill–Hawley, as step 20 built the lower arch: the six anterior teeth on a
// circle whose radius is the combined width of one central incisor, one
// lateral and one canine (Hawley, *Dental Cosmos* 1905), and the premolars and
// molars running back from it in straight lines. The radius is DERIVED from
// the table: 8.5 + 6.5 + 7.5 = 22.5 mm, against step 20's 17.5.
let hawleyRadius: Float = toothTable[0].width + toothTable[1].width + toothTable[2].width

/// Centre-to-centre distance between the two first molars: the maxillary
/// intermolar width measured "between the central fossae of right and left
/// first molars" in 50 men and 50 women aged 20–30, 50.43 ± 2.25 mm and
/// 47.90 ± 2.06 mm (Wankhede et al., *J Oral Maxillofac Pathol* 2023,
/// 27:121–129, PMC10207221; abstract checked). The mean of the two, 49.17 mm.
/// A central fossa is the middle of the occlusal table, so it is the tooth's
/// centre here. (Step 20's lower arch aims its first molars 38 mm apart.)
let firstMolarSpan: Float = 49.17

/// The same study's maxillary intercanine width, cusp tip to cusp tip, is
/// 36.08 ± 2.04 mm in men and 34.31 ± 1.75 mm in women; the mean of the two is
/// what the Hawley circle should land near without being told. Not used to
/// build anything: the tests read the arch against it.
let publishedIntercanine: (mean: Float, sd: Float) = (35.195, 2.04)

/// And its interpremolar width between the distal pits of the first
/// premolars: 38.97 ± 2.10 (men), 36.92 ± 1.87 (women). Checked, not built.
let publishedInterpremolar: (mean: Float, sd: Float) = (37.945, 2.10)

/// One tooth, placed. `tangent` points distally along the arch, `outward`
/// points labially or buccally — away from the palate.
struct PlacedTooth {
    var spec: ToothSpec
    var number: Int             // Universal: #1 upper right third molar … #16 upper left
    var centre: SIMD2<Float>    // (x, z)
    var tangent: SIMD2<Float>
    var outward: SIMD2<Float>
    var side: Float             // +1 patient's right, −1 patient's left
}

/// Arc length from the midline to each tooth's centre on one side: each tooth
/// starts where the one in front of it ends, so neighbours touch.
let archCentres: [Float] = {
    var out: [Float] = []
    var edge: Float = 0
    for t in toothTable {
        out.append(edge + t.width / 2)
        edge += t.width
    }
    return out
}()

/// Where along one half of the arch a distance `s` from the midline lands, and
/// which way the arch runs there. Circle up to the distal end of the canine,
/// straight line after it, aimed so the first molars land `firstMolarSpan`
/// apart. The line leaves the circle about 15° off its direction: the "canine
/// kink" a travelling camera must be smoothed over (step 67; step 21's was 20°).
func archPoint(_ s: Float) -> (point: SIMD2<Float>, tangent: SIMD2<Float>) {
    let r: Float = hawleyRadius
    if s <= r {
        let angle: Float = s / r
        let p = SIMD2<Float>(r * sin(angle), r * (1 - cos(angle)))
        return (p, SIMD2<Float>(cos(angle), sin(angle)))
    }
    let end = SIMD2<Float>(r * sin(1), r * (1 - cos(1)))
    let molarS: Float = archCentres[5]
    let run: Float = molarS - r
    let dx: Float = firstMolarSpan / 2 - end.x
    let slope: Float = dx / run
    let dz: Float = (1 - slope * slope).squareRoot()
    let dir = SIMD2<Float>(slope, dz)
    return (end + dir * (s - r), dir)
}

/// One tooth set on the arch at arc length `arc` from the midline, on `side`.
func placeTooth(_ spec: ToothSpec, number: Int, at arc: Float, side: Float) -> PlacedTooth {
    let (p, t) = archPoint(arc)
    let centre = SIMD2<Float>(p.x * side, p.y)
    let tangent = simd_normalize(SIMD2<Float>(t.x * side, t.y))
    var outward = SIMD2<Float>(tangent.y, -tangent.x)
    let towardMidline = SIMD2<Float>(-centre.x, 25 - centre.y)
    if simd_dot(outward, towardMidline) > 0 { outward = -outward }
    return PlacedTooth(spec: spec, number: number, centre: centre, tangent: tangent, outward: outward, side: side)
}

/// The Universal number of the `i`th tooth from the midline (0 central … 7
/// third molar) on `side`. Upper right: #8 central back to #1 third molar;
/// upper left: #9 central back to #16. The patient's right is +x.
func universalNumber(index i: Int, side: Float) -> Int {
    let right: Bool = archMutant == .mirrored ? side < 0 : side > 0
    return right ? 8 - i : 9 + i
}

/// All sixteen: the patient's left side (−x, #9 … #16) first, then the right
/// (+x, #8 … #1), as step 20 ordered its sides.
func placeTeeth() -> [PlacedTooth] {
    var out: [PlacedTooth] = []
    for side in [Float(-1), Float(1)] {
        for (i, spec) in toothTable.enumerated() {
            out.append(placeTooth(spec, number: universalNumber(index: i, side: side), at: archCentres[i], side: side))
        }
    }
    return out
}

/// Index in `placeTeeth()` of a Universal number.
func toothIndex(_ number: Int, in placed: [PlacedTooth]) -> Int? {
    placed.firstIndex { $0.number == number }
}

// MARK: - the crowns' own shapes, beyond the table

/// The maxillary first molar's occlusal outline is a rhomboid, "the
/// mesiobuccal and distolingual angles acute, the mesiolingual and distobuccal
/// obtuse" (Wheeler's, maxillary first molar, occlusal aspect). Drawn as the
/// rectangle of the table sheared along the arch: the buccal side slid mesially
/// by this much per millimetre across. MODEL: 0.18 is a 10° lean, which puts
/// the acute angles near 80°; the book gives the shape, not an angle. The
/// second molar is "less rhomboidal", the third often heart-shaped: half and a
/// quarter of it, MODEL.
func occlusalShear(_ kind: CrownKind) -> Float {
    switch kind {
    case .firstMolar: return 0.18
    case .secondMolar: return 0.09
    case .thirdMolar: return 0.045
    default: return 0
    }
}

// MARK: - the gum

/// The free gingival margin sits about 1 mm coronal to the cementoenamel
/// junction in a healthy adult (Gargiulo, Wentz & Orban, *J Periodontol* 1961).
/// Step 20's value.
let gumMarginAboveCEJ: Float = 1.0

/// Step 20's papilla fill, MODEL bounded by Tarnow et al. 1992.
let papillaExtra: Float = 1.0

/// Step 20's gum thickness over the root past the collar. MODEL.
let gumWidth: Float = 2.5

/// Step 20's widening of the ridge with depth, mm per mm. MODEL.
let gumSlope: Float = 0.12

// MARK: - the palate

/// Palatal vault height: "a vertical line … from the midpoint of the IM
/// [intermolar] horizontal line to the mid-palatine raphe", the IM line
/// joining the palatal gingival margins of the first molars. 15.61 ± 2.7 mm
/// over 109 adult casts (Alaqeely et al., *Medicina* 2025, "Evaluation of the
/// Palatal Features in Relation to Graft Harvesting in the Saudi Population",
/// PMC11766974; full text checked).
let palatalVaultHeight: Float = 15.61

/// The same study: the palatal rugae most often reach back to "the middle of
/// the second premolar". Their hindmost lateral end is put there.
let rugaeReachTooth: Int = 4     // the second premolar, from the midline

/// Primary rugae per side: 4.35 ± 0.98 right, 4.33 ± 0.92 left (Armstrong et
/// al., *Sci Rep* 2020, "Palatal rugae morphology is associated with variation
/// in tooth number", PMC7645628; checked). Four each side, rounded.
let rugaePerSide: Int = 4

/// Their lengths. Primary rugae are 5–10 mm by the Lysell (1955) classification
/// Armstrong et al. use, and their first primary ruga measures 8.78 ± 1.56 mm.
/// MODEL within that: the first 8.8 mm, each one behind a little shorter.
let rugaLengths: [Float] = [8.8, 9.0, 8.0, 7.0]

/// A ruga's cross-section: a rounded ridge. MODEL: rugae are low folds of
/// the mucosa about a millimetre high and two wide; no measurement to hand.
let rugaRadius: Float = 1.1
let rugaHeight: Float = 1.0

/// The incisive papilla: "12.62 ± 1.60 mm" from the maxillary central incisors
/// to the incisive papilla in 100 dentate adults (Karthik et al., *J Family
/// Med Prim Care* 2020, PMC7014900; checked). The paper's landmark on the
/// incisors is not stated clearly; taken here as the labial face at the
/// midline, to the papilla's centre. UNVERIFIED which landmark they meant.
let incisivePapillaDistance: Float = 12.62

/// The papilla is "a small pear or oval-shaped mucosal prominence" (ibid.).
/// Its size is MODEL: 4 mm across, 7 long, standing a millimetre proud.
let papillaHalfWidth: Float = 2.0
let papillaHalfLength: Float = 3.5
let papillaHeight: Float = 1.0

/// How far the palate's rim — where the flat shelf under the gum turns up into
/// the vault — sits below the first molars' gum margin, in the arch's frame.
/// MODEL: one gum collar (step 20's GUM_COLLAR, 4.5 mm), where the gum has
/// reached full thickness, so the rim is buried in gum all round the arch.
let palateRimBelowMargin: Float = 4.5

/// The vault: an ellipsoid of air carved into the palate, its widest section at
/// the rim. Everything is derived from the teeth and the measurements above:
/// its front at the palatal face of the central incisors' necks, its sides
/// through the palatal faces of the first molars' necks, its back behind the
/// last molars, and its depth set so the raphe stands `palatalVaultHeight`
/// above the first molars' palatal margins.
struct Vault {
    var rimY: Float             // the arch frame's y at the rim
    var centreZ: Float
    var radii: SIMD3<Float>     // across, deep, along

    /// The vault's surface height (arch frame y) over (x, z), or nil outside it.
    func roof(_ x: Float, _ z: Float) -> Float? {
        let qx: Float = x / radii.x
        let qz: Float = (z - centreZ) / radii.z
        let k: Float = 1 - qx * qx - qz * qz
        guard k > 0 else { return nil }
        return rimY - radii.y * k.squareRoot()
    }
}

/// How far behind the last molar's distal face the vault ends. MODEL: about
/// where the hard palate ends behind the tuberosity.
let vaultBehindLastMolar: Float = 6.0

let vault: Vault = {
    let placed: [PlacedTooth] = placeTeeth()
    let central: PlacedTooth = placed[0]
    let molar: PlacedTooth = placed[5]
    let last: PlacedTooth = placed[7]
    let front: Float = central.centre.y + central.spec.cervicalDepth / 2
    let back: Float = last.centre.y + last.spec.width / 2 + vaultBehindLastMolar
    let centreZ: Float = (front + back) / 2
    let rz: Float = (back - front) / 2
    // Through the palatal face of the first molar's neck, at its z.
    let neck: SIMD2<Float> = molar.centre - molar.outward * (molar.spec.cervicalDepth / 2)
    let along: Float = (neck.y - centreZ) / rz
    let rx: Float = abs(neck.x) / (1 - along * along).squareRoot()
    let margin: Float = -molar.spec.crownHeight + gumMarginAboveCEJ
    let rimY: Float = margin - palateRimBelowMargin
    // The raphe at the first molars' z must sit the vault height below the
    // margin line; the ellipsoid there is sqrt(1 − along²) of its full depth.
    let wanted: Float = palatalVaultHeight - palateRimBelowMargin
    let atMolar: Float = (1 - ((molar.centre.y - centreZ) / rz) * ((molar.centre.y - centreZ) / rz)).squareRoot()
    let ry: Float = wanted / atMolar
    return Vault(rimY: rimY, centreZ: centreZ, radii: SIMD3<Float>(rx, ry, rz))
}()

/// One ruga as three points on the palate (arch frame): medial, a wave, lateral.
/// Built on the patient's right (+x) and mirrored for the left.
///
/// Layout, MODEL, following the descriptions in the studies above: the first
/// primary ruga starts beside the incisive papilla, a millimetre off its side,
/// and runs out and a little forward (the "forward" and "curved" forms are the
/// commonest in the rugoscopy literature); the ones behind it start closer to
/// the midline raphe and run more nearly straight across, then a little back;
/// and the last one's lateral end reaches the middle of the second premolar,
/// as Alaqeely et al. found.
struct Ruga {
    var points: [SIMD3<Float>]
}

let rugae: [Ruga] = {
    guard archMutant != .noRugae else { return [] }
    let placed: [PlacedTooth] = placeTeeth()
    let papillaZ: Float = incisivePapillaZ
    let premolar: PlacedTooth = placed[8 + rugaeReachTooth]   // patient's right second premolar
    let reachZ: Float = premolar.centre.y
    var out: [Ruga] = []
    for i in 0..<rugaePerSide {
        let f: Float = Float(i) / Float(rugaePerSide - 1)
        // Angle from straight across, positive = lateral end further back.
        let lean: Float = -0.30 + 0.55 * f
        let len: Float = rugaLengths[i]
        let dir = SIMD2<Float>(cos(lean), sin(lean))
        let medialX: Float = papillaHalfWidth + 1.0 - 1.0 * f
        // The last ruga's lateral end lands on `reachZ`; the others are spaced
        // evenly from the first, which starts level with the papilla's centre.
        let lastMedialZ: Float = reachZ - len * sin(0.25)
        let medialZ: Float = papillaZ + (lastMedialZ - papillaZ) * f
        let a = SIMD2<Float>(medialX, medialZ)
        let c: SIMD2<Float> = a + dir * len
        let mid: SIMD2<Float> = (a + c) / 2 + SIMD2<Float>(-dir.y, dir.x) * 0.7
        var pts: [SIMD3<Float>] = []
        for q in [a, mid, c] {
            let y: Float = vault.roof(q.x, q.y) ?? vault.rimY
            // The ridge's centre is sunk so it stands `rugaHeight` proud.
            pts.append(SIMD3<Float>(q.x, y + rugaHeight - rugaRadius, q.y))
        }
        out.append(Ruga(points: pts))
        out.append(Ruga(points: pts.map { SIMD3<Float>(-$0.x, $0.y, $0.z) }))
    }
    return out
}()

/// The incisive papilla's centre, z: the labial face of the central incisors at
/// the midline, plus the published distance.
let incisivePapillaZ: Float = {
    let placed: [PlacedTooth] = placeTeeth()
    let central: PlacedTooth = placed[0]
    let labial: Float = central.centre.y - central.spec.depth / 2
    return labial + incisivePapillaDistance
}()

// MARK: - colour
//
// Colours are measured CIELAB, converted, as in step 20.

/// CIELAB (D65) to linear sRGB, the standard CIE formulas via XYZ.
func labToLinearSRGB(_ lab: SIMD3<Double>) -> SIMD3<Float> {
    let fy: Double = (lab.x + 16) / 116
    let fx: Double = fy + lab.y / 500
    let fz: Double = fy - lab.z / 200
    func finv(_ t: Double) -> Double {
        let delta: Double = 6.0 / 29.0
        return t > delta ? t * t * t : 3 * delta * delta * (t - 4.0 / 29.0)
    }
    let xn: Double = 0.95047
    let yn: Double = 1.0
    let zn: Double = 1.08883
    let x: Double = xn * finv(fx)
    let y: Double = yn * finv(fy)
    let z: Double = zn * finv(fz)
    let r: Double = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z
    let g: Double = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z
    let b: Double = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z
    return SIMD3<Float>(Float(max(r, 0)), Float(max(g, 0)), Float(max(b, 0)))
}

/// And back, for the tests that read a colour off the finished picture.
func linearSRGBToLab(_ c: SIMD3<Float>) -> SIMD3<Double> {
    let r: Double = Double(c.x)
    let g: Double = Double(c.y)
    let b: Double = Double(c.z)
    let x: Double = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    let y: Double = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    let z: Double = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b
    func f(_ t: Double) -> Double {
        let delta: Double = 6.0 / 29.0
        return t > delta * delta * delta ? cbrt(t) : t / (3 * delta * delta) + 4.0 / 29.0
    }
    let fx: Double = f(x / 0.95047)
    let fy: Double = f(y / 1.0)
    let fz: Double = f(z / 1.08883)
    return SIMD3<Double>(116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz))
}

/// sRGB bytes to linear, the standard transfer curve.
func srgbByteToLinear(_ v: UInt8) -> Float {
    let c: Float = Float(v) / 255
    return c <= 0.04045 ? c / 12.92 : pow((c + 0.055) / 1.055, 2.4)
}

/// Which measured colour a tooth takes. Step 20's source measured MAXILLARY
/// anterior teeth, so here each gets its own row rather than step 20's mean of
/// the two incisors.
enum ShadeClass: Int, CaseIterable {
    case central = 0, lateral = 1, canine = 2, posterior = 3
}

func shadeClass(index i: Int) -> ShadeClass {
    switch i {
    case 0: return .central
    case 1: return .lateral
    case 2: return .canine
    default: return .posterior
    }
}

// Middle third of the crown, measured in vivo with a VITA Easyshade on 160
// adults: maxillary central incisor L* 73.0, a* −0.5, b* 14.5; lateral 70.0,
// 0.2, 18.6; canine 65.1, 1.4, 23.6 (Measurement and analysis of maxillary
// anterior teeth color in the Chinese population, 2023, PMC10155942 — step
// 20's source and numbers). Premolars and molars were not measured there and
// get the lateral incisor's value — MODEL, as in step 20.
let middleThirdLab: [ShadeClass: SIMD3<Double>] = [
    .central: SIMD3(73.0, -0.5, 14.5),
    .lateral: SIMD3(70.0, 0.2, 18.6),
    .canine: SIMD3(65.1, 1.4, 23.6),
    .posterior: SIMD3(70.0, 0.2, 18.6),
]

/// Step 20's cervical shift: direction from Hasegawa, Ikeda & Kawaguchi,
/// *J Prosthet Dent* 2000; size MODEL.
let cervicalShift: SIMD3<Double> = SIMD3(-4.0, 1.5, 5.0)

/// Step 20's incisal enamel: L* 73.5, a* 2.2, b* 11.9 (Wee et al., *J Prosthet
/// Dent* 2022), measured on vital MAXILLARY central incisors.
let incisalLab: SIMD3<Double> = SIMD3(73.5, 2.2, 11.9)

/// Keratinized gingiva, all subjects: L* 52.9, a* 23.3, b* 14.9 (Ho, Ghinea,
/// Herrera, Angelov & Paravina, *Sci Rep* 2015). Step 20's value.
let gingivaLab: SIMD3<Double> = SIMD3(52.9, 23.3, 14.9)

/// The hard palate's mucosa. MODEL: no CIELAB measurement of it could be found
/// (Europe PMC searched for palatal mucosa colour with CIELAB, spectrophotometer
/// and colorimeter). It is masticatory mucosa — keratinized, bound to bone —
/// the same class of tissue as the attached gingiva, so it takes Ho et al.'s
/// gingiva value unchanged.
let palateLab: SIMD3<Double> = gingivaLab

// MARK: - reflectance, from refractive indices (step 20's)

/// Normal-incidence Fresnel reflectance between two media.
func fresnelF0(_ n1: Double, _ n2: Double) -> Float {
    let r: Double = (n1 - n2) / (n1 + n2)
    return Float(r * r)
}

/// Enamel 1.62 (Spitzer & ten Bosch, *Calcif Tissue Res* 1975); saliva as
/// water, 1.333. The wet tooth's highlight is the film's, 2.0%; the enamel
/// under it reflects 0.94%.
let enamelIndex: Double = 1.62
let salivaIndex: Double = 1.333
let filmF0: Float = fresnelF0(1.0, salivaIndex)
let enamelUnderFilmF0: Float = fresnelF0(salivaIndex, enamelIndex)
let dryEnamelF0: Float = fresnelF0(1.0, enamelIndex)
