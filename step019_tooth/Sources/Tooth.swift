// One molar, in section, with every number's provenance beside it.
//
// The geometry is a mesiodistal section through a lower first molar, built from
// published sectional anatomy rather than scanned from a real tooth — so the
// shape is MODEL and says so, while the material properties and the loads are
// measured and cite where from.
//
// What this step is actually about is the dentino-enamel junction. Cracks form
// in enamel throughout an ordinary life and are routinely arrested there, and
// the reason is a modulus mismatch: a very stiff, brittle shell bonded through
// a graded layer to a much more compliant, much tougher core. Dental ceramics
// now copy the arrangement deliberately.
//
// THE TOOTH IS NOT CLAMPED. That is the correction this file exists to make.
// The first version of this step pinned every node below y = −6 mm rigidly, and
// got a stress field whose maximum sat one node row above the pin, 5.9 mm below
// the neck, with 25,444 bonds broken and the whole crown in rubble. Both
// symptoms were the same artefact. A tooth hangs in a PERIODONTAL LIGAMENT
// roughly 0.2 mm thick, and the alveolar bone's crest reaches to within about
// 1.5 mm of the neck — so the support starts near the neck and it is soft.
// Putting both of those in is what moved the peak to the cervical region and
// turned the demolition into a crack. See `pdlModulus` and `alveolarCrestY`.
//
// The abfraction story — that grinding carves the wedge-shaped notches seen at
// the gumline — is CONTESTED, and the render must not settle it. See
// `evidenceFor` at the bottom.

import Foundation
import simd

// MARK: - constants, each with where it came from

struct Fact {
    var name: String
    var value: String
    var standing: Standing
    var source: String
}

enum Standing: String, CaseIterable {
    case measured = "MEASURED"
    case simulated = "SIMULATED"
    case model = "MODEL"
    case contested = "CONTESTED"
}

// -- materials ---------------------------------------------------------------
//
// Enamel's modulus is quoted anywhere from 3× to 7× dentin's depending on the
// method and the site (Int. J. Oral Sci. review, 2014); 84 GPa against 18 GPa
// sits inside that band at 4.7×.
//
// The critical strains are the load-bearing modelling choice in this whole
// step. They are derived as σ/E from tensile strengths, and the RATIO is what
// decides whether cracks arrest — dentin is both more compliant and stronger in
// tension, so its failure strain is nearly an order of magnitude larger than
// enamel's. That ratio, not either number alone, is the mechanism.

let enamelModulus: Double = 84.0e9
let dentinModulus: Double = 18.0e9
let dejModulus: Double = 38.0e9        // graded: geometric mean of its neighbours
let cementumModulus: Double = 15.0e9

// The periodontal ligament. Published moduli for it span FIVE orders of
// magnitude — 0.01 MPa to 1750 MPa — because it is a viscoelastic, fluid-filled,
// fibre-reinforced tissue and every measurement method sees a different part of
// it. Two values dominate dental finite element work: 0.0689 MPa (Rees &
// Jacobsen, *Biomaterials* 1997, from load–displacement of intruding teeth) and
// 50–68.9 MPa (the same authors' figure read three orders up, which is what most
// later papers cite). 50 MPa is used here because it is the common FEA value,
// and `pdlSensitivity` in the tests sweeps it across four decades to show what
// the answer does and does not depend on. It turns out the peak's LOCATION is
// robust and its MAGNITUDE is not.
let pdlModulus: Double = 50.0e6
let boneModulus: Double = 13.7e9       // cortical bone, the standard dental FEA value

let enamelStrength: Double = 30.0e6    // Pa, tensile
let dentinStrength: Double = 60.0e6
let dejStrength: Double = 45.0e6
let cementumStrength: Double = 35.0e6

func criticalStrain(strength: Double, modulus: Double) -> Double { strength / modulus }

/// The ligament is not a brittle solid and does not fail at these loads; it is
/// given a strength it can never reach rather than a number pretending to be a
/// measurement. Its bonds to cementum therefore inherit cementum's strength,
/// which is what "only as strong as its weaker end" means. Bone is held rigid,
/// so its bonds carry no strain at all.
let unbreakableStrength: Double = 1.0e12

let toothProperties: [ToothTissue: TissueProperties] = [
    .enamel: TissueProperties(youngsModulus: enamelModulus,
                              tensileStrength: enamelStrength, name: "enamel"),
    .dej: TissueProperties(youngsModulus: dejModulus,
                           tensileStrength: dejStrength, name: "dentino-enamel junction"),
    .dentin: TissueProperties(youngsModulus: dentinModulus,
                              tensileStrength: dentinStrength, name: "dentin"),
    .cementum: TissueProperties(youngsModulus: cementumModulus,
                                tensileStrength: cementumStrength, name: "cementum"),
    .pdl: TissueProperties(youngsModulus: pdlModulus,
                           tensileStrength: unbreakableStrength, name: "periodontal ligament"),
    .bone: TissueProperties(youngsModulus: boneModulus,
                            tensileStrength: unbreakableStrength, name: "alveolar bone"),
]

// -- loads -------------------------------------------------------------------

let chewingForce: Double = 100        // N, mid of the measured 20–120 N range
let maximumBiteForce: Double = 450    // N, mid of 300–600 N for adult molars
let bruxingForce: Double = 1000       // N, the reported upper end in sleep bruxism

// -- geometry, in millimetres, y up, origin at the cementoenamel junction -----

let crownHeight: Float = 7.5          // CEJ to cusp tip
let rootLength: Float = 13.0
let cejHalfWidth: Float = 4.2
let heightOfContour: Float = 2.0      // where the crown is widest
let maxHalfWidth: Float = 5.0
let cuspTipX: Float = 2.6
let fissureDepth: Float = 1.3         // cusp tip down to the central fissure
let enamelAtCusp: Float = 2.2         // thickest, measured over the cusp
let enamelAtCEJ: Float = 0.06         // a knife edge — this is the whole point
let dejBand: Float = 0.18             // thickness of the graded junction
let cementumThickness: Float = 0.12
let pulpRoofY: Float = 3.4
let pulpHalfWidthMax: Float = 1.25
let apexRadius: Float = 1.60          // the root ends rounded, not chopped off

/// How far below the CEJ the alveolar bone crest sits. 1–2 mm is the healthy
/// range; below that is what periodontal charting calls bone loss. This one
/// number is most of the fix to the clamping artefact: the old model held the
/// tooth 6 mm below the neck and it behaved like a cantilever.
let alveolarCrestY: Float = -1.5

/// Periodontal ligament thickness. 0.15–0.38 mm, narrowest at mid-root; 0.2 mm
/// is the usual single figure.
let pdlThickness: Float = 0.2

/// How much bone the LATTICE carries. Bone is held rigid, so a thicker shell
/// would change nothing in the solve; the render draws the whole jaw block.
let boneShell: Float = 0.5

let apexY: Float = -rootLength

// -- the occlusal surface -----------------------------------------------------

/// The occlusal span: the fissure sits at x = 0 and the cusp tips at ±cuspTipX,
/// so one half-period of sin² over twice the cusp offset puts the peaks exactly
/// on the tips. (The first version of this function had five terms, three of
/// which cancelled algebraically to nothing. It reduced to a plain sine — and a
/// plain sine of |x| has a CORNER at x = 0, which is a V-notch with a tip one
/// lattice spacing wide. The first solve of the fixed model duly put its
/// maximum there, 105 MPa at the bottom of a notch whose sharpness was a typo.
/// sin² is flat at both ends, so the fissure has a floor with a 1.1 mm radius
/// and the cusps have rounded tips, which is what they have.)
let occlusalSpan: Float = 2 * cuspTipX
let fissureFloorY: Float = crownHeight - fissureDepth

/// The height of the top surface above the CEJ at a given horizontal offset.
func occlusalHeight(x: Float) -> Float {
    let ax: Float = abs(x)
    guard ax < occlusalSpan else { return fissureFloorY }
    let phase: Float = Float.pi * ax / occlusalSpan
    let hump: Float = sin(phase)
    return fissureFloorY + fissureDepth * hump * hump
}

/// The two branches of that sine, inverted. Inside the cusp tips it is the
/// fissure's wall; outside them it is the crown's own shoulder. Both are needed
/// because the solid between them is an ANNULUS at any height in the fissure,
/// and the render builds the body as a solid of revolution.
func fissureRadius(y: Float) -> Float {
    guard y > fissureFloorY else { return 0 }
    guard y < crownHeight else { return cuspTipX }
    let t: Float = (y - fissureFloorY) / fissureDepth
    let clamped: Float = min(max(t, 0), 1)
    let angle: Float = asin(clamped.squareRoot())
    return occlusalSpan * angle / Float.pi
}

func occlusalOuterBranch(y: Float) -> Float {
    guard y > fissureFloorY else { return maxHalfWidth * 2 }
    guard y < crownHeight else { return 0 }
    return occlusalSpan - fissureRadius(y: y)
}

// -- the outer profile --------------------------------------------------------

/// Half-width of the crown's side wall at height `y`, before the occlusal
/// surface cuts it back.
func crownSideWidth(y: Float) -> Float {
    if y <= heightOfContour {
        let f: Float = y / heightOfContour
        return cejHalfWidth + (maxHalfWidth - cejHalfWidth) * f
    }
    let span: Float = crownHeight - heightOfContour
    let f: Float = (y - heightOfContour) / span
    let clamped: Float = min(max(f, 0), 1)
    let shoulder: Float = maxHalfWidth - cuspTipX - 0.7
    return maxHalfWidth - shoulder * clamped * clamped
}

/// The root, tapering to a rounded apex.
func rootWidth(y: Float) -> Float {
    let f: Float = min(max(-y / rootLength, 0), 1)
    let taper: Float = 1 - 0.78 * f * f
    var w: Float = cejHalfWidth * taper
    let intoApex: Float = (-y) - (rootLength - apexRadius)
    if intoApex > 0 {
        let u: Float = min(intoApex / apexRadius, 1)
        let round: Float = (1 - u * u)
        w *= max(round, 0).squareRoot()
    }
    return max(w, 0)
}

/// Half-width of the whole tooth at height `y`.
func outerHalfWidth(y: Float) -> Float {
    guard y > apexY - 1e-4 else { return 0 }
    if y < 0 { return rootWidth(y: y) }
    guard y <= crownHeight else { return 0 }
    return min(crownSideWidth(y: y), occlusalOuterBranch(y: y))
}

// -- the layers ---------------------------------------------------------------

/// Enamel thickness measured horizontally in from the side wall. Thick over the
/// cusps, a knife edge at the CEJ, nothing below it.
func enamelThickness(y: Float) -> Float {
    guard y > 0 else { return 0 }
    let f: Float = min(max(y / crownHeight, 0), 1)
    let curve: Float = f * f
    return enamelAtCEJ + (enamelAtCusp - enamelAtCEJ) * curve
}

/// Cementum thickness, blended up into the enamel's knife edge at the neck so
/// the two coverings meet instead of stepping past each other. That overlap is
/// real: cementum overlaps enamel at the CEJ in roughly 60% of teeth.
func cementumThicknessAt(y: Float) -> Float {
    guard y < 0 else { return enamelAtCEJ + dejBand }
    let at0: Float = enamelAtCEJ + dejBand
    let decay: Float = exp(y / 0.35)
    return cementumThickness + (at0 - cementumThickness) * decay
}

/// Where the dentin horn under the cusp stops, and where the junction over it
/// stops. Deriving these from the enamel's thickness over the cusp is what puts
/// the horn in the right place: the junction sits one enamel thickness below the
/// tip, not at the tip.
let dentinHornTop: Float = crownHeight - enamelAtCusp - dejBand
let dejTop: Float = crownHeight - enamelAtCusp
let hornSlope: Float = 1.6

/// The dentin core's radius. Two constraints, whichever is tighter: the
/// horizontal offset in from the side wall, and a cone descending from the tip
/// of the dentin horn. The cone is what keeps the enamel a CAP rather than a
/// tube — without it the core runs all the way to the cusp tip and the fissure
/// would be floored with exposed dentin.
func dentinRadius(y: Float) -> Float {
    if y >= dentinHornTop { return 0 }
    if y < 0 {
        return max(outerHalfWidth(y: y) - cementumThicknessAt(y: y), 0)
    }
    let offset: Float = outerHalfWidth(y: y) - enamelThickness(y: y) - dejBand
    let cone: Float = hornSlope * (dentinHornTop - y)
    return max(min(offset, cone), 0)
}

func dejRadius(y: Float) -> Float {
    if y < 0 { return dentinRadius(y: y) }
    if y >= dejTop { return 0 }
    let offset: Float = outerHalfWidth(y: y) - enamelThickness(y: y)
    let cone: Float = hornSlope * (dejTop - y)
    return max(min(offset, cone), 0)
}

func pulpRadius(y: Float) -> Float {
    if y >= pulpRoofY { return 0 }
    if y >= 0 {
        let f: Float = y / pulpRoofY
        let curve: Float = f * f * f.squareRoot()      // f^2.5: a domed roof
        return pulpHalfWidthMax * (1 - curve)
    }
    // The chamber narrows into a canal within a millimetre or so of the neck: a
    // molar's canal is well under a millimetre across, and carrying the chamber's
    // width down the root would have hollowed out the part that has to bend.
    let f: Float = min(max(-y / rootLength, 0), 1)
    let narrow: Float = 0.36 + 0.64 * exp(y / 0.8)
    return max(pulpHalfWidthMax * narrow * (1 - 0.9 * f), 0)
}

/// How thick the ligament is at a height. It THINS to nothing over the half
/// millimetre below the crest rather than stopping square.
///
/// That is not cosmetic. A ligament with a square end puts a step in the
/// support exactly where the tooth is being asked about, and the stress there
/// is a property of the step: the peak moved with the mesh by 60% and sat at
/// y = −1.55 mm, on the crest, not on the neck. The real alveolar crest is a
/// rounded margin and the ligament narrows into it.
func pdlThicknessAt(y: Float) -> Float { pdlThickness }

/// How thick the rigid bone shell is at a height. It WEDGES OUT to nothing over
/// the millimetre below the crest rather than stopping square.
///
/// This is the fix that matters and it took two goes. Tapering the LIGAMENT to
/// nothing at the crest was the first attempt, and it was worse: with no
/// ligament left there, rigid bone touched the root directly and the model had a
/// point clamp exactly where it was being asked a question. What has to fade is
/// the SUPPORT, not the cushion, so the bone's inner edge keeps its 0.2 mm of
/// ligament all the way up and the bone itself wedges away.
func boneShellAt(y: Float) -> Float {
    let fade: Float = 0.9
    guard y > alveolarCrestY - fade else { return boneShell }
    let t: Float = (alveolarCrestY - y) / fade
    let clamped: Float = min(max(t, 0), 1)
    let smooth: Float = clamped * clamped * (3 - 2 * clamped)
    return boneShell * smooth
}

/// Outer radius of the ligament: the root's surface plus its thickness, wrapping
/// round the apex as a cap. Zero above the alveolar crest, because above the
/// crest the tooth is held by nothing.
func pdlOuterRadius(y: Float) -> Float {
    if y > alveolarCrestY { return 0 }
    if y >= apexY { return rootWidth(y: y) + pdlThicknessAt(y: y) }
    let d: Float = apexY - y
    if d >= pdlThickness { return 0 }
    let r: Float = pdlThickness * pdlThickness - d * d
    return max(r, 0).squareRoot()
}

/// The lattice's bone: a thin rigid shell just outside the ligament.
func boneOuterRadius(y: Float) -> Float {
    if y > alveolarCrestY { return 0 }
    if y < apexY - pdlThickness - boneShell { return 0 }
    let inner: Float = pdlOuterRadius(y: y)
    if inner > 0 { return inner + boneShellAt(y: y) }
    let d: Float = apexY - y
    let r: Float = (pdlThickness + boneShell) * (pdlThickness + boneShell) - d * d
    return max(r, 0).squareRoot()
}

// MARK: - what tissue sits where

/// What tissue sits at a point in the section. Coordinates in millimetres. The
/// order of these tests IS the nesting, innermost last, and the render's
/// primitive priorities are built from the same order — a test checks the two
/// agree point by point.
func tissueAt(_ p: SIMD2<Float>) -> ToothTissue {
    let x: Float = abs(p.x)
    let y: Float = p.y
    let outer: Float = outerHalfWidth(y: y)
    if x > outer {
        if x <= pdlOuterRadius(y: y) { return .pdl }
        if x <= boneOuterRadius(y: y) { return .bone }
        return .outside
    }
    if outer <= 0 { return .outside }
    if x < fissureRadius(y: y) { return .outside }
    if x < pulpRadius(y: y) { return .pulp }
    if x < dentinRadius(y: y) { return .dentin }
    if x < dejRadius(y: y) { return .dej }
    return y >= 0 ? .enamel : .cementum
}

/// How a node is constrained. The bone is rigid — that is the only pinning in
/// this model, and every node of the tooth itself is free. `rigidRoot` is the
/// mutation: it puts the old clamp back.
func kindAt(_ p: SIMD2<Float>, _ t: ToothTissue, rigidRoot: Bool = false) -> NodeKind {
    guard t.isSolid else { return .free }
    if rigidRoot { return p.y < -6.0 ? .fixed : .free }
    return t == .bone ? .fixed : .free
}

// MARK: - where the opposing tooth presses

/// A load case. `direction` is a unit vector; axial is straight down the long
/// axis, and the oblique cases are the NORMAL TO THE CUSP FACET at the contact
/// point, plus friction — not a direction anybody typed.
///
/// That distinction matters more than it sounds. An oblique resultant on a flat
/// cusp tip would need a coefficient of friction of about 1.0 to exist at all,
/// and wet enamel on enamel is 0.1–0.4. Oblique occlusal forces are oblique
/// because the CONTACT IS ON AN INCLINE and the force is normal to it. So the
/// contact point is what picks the direction here, and the two inclines of one
/// cusp pick opposite ones.
struct LoadCase {
    var name: String
    var newtons: Double
    var direction: SIMD2<Float>
    var contactX: Float
    var contactHalfWidth: Float

    /// The contact pressure this case puts on the occlusal surface, in MPa,
    /// averaged over a patch `2·contactHalfWidth` long and `sectionThickness`
    /// wide. The distribution is Hertzian rather than flat, so the peak is 4/π
    /// times this.
    func contactPressureMPa(thickness: Double) -> Double {
        let area: Double = 2 * Double(contactHalfWidth) * 1e-3 * thickness
        return newtons / area / 1e6
    }
}

/// The slope of the occlusal surface, dy/dx. Differentiated by hand from
/// `occlusalHeight`, which is floor + D·sin²(πx/S):
///
///     dy/dx = D·(π/S)·sin(2πx/S)
///
/// so the steepest facets are at x = S/4 and x = 3S/4 and the slope there is
/// D·π/S = 38°. Real molar cusp inclines are quoted at roughly 25–45°, so the
/// shape is in range without anybody having chosen an angle.
func occlusalSlope(x: Float) -> Float {
    let ax: Float = abs(x)
    guard ax < occlusalSpan else { return 0 }
    let k: Float = Float.pi / occlusalSpan
    let slope: Float = fissureDepth * k * sin(2 * k * ax)
    return x >= 0 ? slope : -slope
}

/// Sliding friction between wet enamel surfaces. 0.1–0.4 is the usual range;
/// 0.2 is taken. It rotates the resultant by atan(0.2) = 11°, which is small
/// and is the difference between a direction with a reason and a direction
/// somebody liked.
let enamelFriction: Float = 0.2

/// The force a contact on the facet at `x` puts on this tooth: the inward facet
/// normal, plus friction acting in −x.
///
/// −x is where the friction goes because on the working side of a lateral
/// excursion the opposing tooth slides lingually across this one, and friction
/// on this tooth follows that motion. Flipping it would flip the answer by 22°,
/// which is why it is written down rather than assumed.
func facetForceDirection(x: Float) -> SIMD2<Float> {
    let m: Float = occlusalSlope(x: x)
    let outward: SIMD2<Float> = simd_normalize(SIMD2<Float>(-m, 1))
    var tangent: SIMD2<Float> = simd_normalize(SIMD2<Float>(1, m))
    if tangent.x > 0 { tangent = -tangent }
    let pressed: SIMD2<Float> = -outward + tangent * enamelFriction
    return simd_normalize(pressed)
}

func facetLoad(name: String, newtons: Double, contactX: Float,
               halfWidth: Float = 0.30) -> LoadCase {
    LoadCase(name: name, newtons: newtons, direction: facetForceDirection(x: contactX),
             contactX: contactX, contactHalfWidth: halfWidth)
}

/// The two inclines of the buccal cusp, and they are not the same load case.
///
/// The OUTER (buccal) incline's normal points down and lingually. Its horizontal
/// component bends the crown lingually, which puts the buccal neck in tension —
/// but its vertical component is 3.3 mm off the axis and bends it the other way,
/// so the two partly cancel.
///
/// The INNER (lingual) incline's normal points down and BUCCALLY, and there the
/// two moments ADD. Same tooth, same 1000 N, same cusp: the neck sees more than
/// twice as much from one incline as from the other. That is measured in the
/// tests and it is one of the reasons abfraction is contested rather than
/// settled — the answer depends on where in the grinding cycle you look.
let outerInclineX: Float = 3.25
let innerInclineX: Float = 1.30

let axialChewing = LoadCase(name: "chewing, axial", newtons: chewingForce,
                            direction: SIMD2<Float>(0, -1),
                            contactX: cuspTipX, contactHalfWidth: 0.9)

/// The same 1000 N, straight down the axis. Used by the tests: it is how
/// "oblique exceeds axial" is measured at equal force rather than at equal name.
let axialBruxing = LoadCase(name: "bruxing, axial", newtons: bruxingForce,
                            direction: SIMD2<Float>(0, -1),
                            contactX: cuspTipX, contactHalfWidth: 0.9)

let lateralBruxing = facetLoad(name: "bruxing, lingual incline",
                               newtons: bruxingForce, contactX: innerInclineX)

let outerInclineBruxing = facetLoad(name: "bruxing, buccal incline",
                                    newtons: bruxingForce, contactX: outerInclineX)

/// Out-of-plane thickness of the section, in metres. The section is
/// BUCCOLINGUAL — through the mesial root of a lower first molar, so the two
/// cusps in it are the buccal and the lingual one, +x is buccal, and a lateral
/// excursive load lies in the plane. (The first draft of this file called the
/// section mesiodistal and then loaded it buccolingually, which is a load
/// applied out of the plane it was solved in.) A molar is about 10 mm across
/// mesiodistally, which is the depth this section stands for.
let sectionThickness: Double = 10.0e-3

// MARK: - the evidence bar, which changes as the render plays
//
// Step 10 invented this because its three transformation routes each carried a
// different standard of evidence and the bar flipped between them. Steps 13 and
// 14 dropped it, correctly, because it said the same three lines for 120 frames
// and had become furniture. Here it changes again — and a test fails if it
// stops changing.

enum Beat: Int, CaseIterable {
    case anatomy = 0        // the section, unloaded
    case chewing            // ordinary load
    case bruxing            // heavy off-axis load, the stress field
    case cracking           // enamel cracks initiate and run
    case arrest             // they stop at the junction
    case lesion             // and the contested part
}

func evidenceFor(_ beat: Beat) -> Standing {
    switch beat {
    case .anatomy: return .measured
    case .chewing: return .measured
    case .bruxing: return .simulated
    case .cracking: return .simulated
    case .arrest: return .measured
    case .lesion: return .contested
    }
}

func captionFor(_ beat: Beat) -> String {
    switch beat {
    case .anatomy: return "enamel thins to a knife edge at the neck"
    case .chewing: return "ordinary chewing, 100 N down the long axis"
    case .bruxing: return "bruxing, 1000 N off-axis — tension gathers at the neck"
    case .cracking: return "enamel cracks, because enamel is brittle"
    case .arrest: return "and the junction stops them"
    case .lesion: return "that grinding carves the notch is disputed"
    }
}

func noteFor(_ beat: Beat) -> String {
    switch beat {
    case .anatomy: return "section from published anatomy, not a scan"
    case .chewing: return "20–120 N measured for mastication"
    case .bruxing: return "solved here: 2D lattice, linear elastic, held in ligament"
    case .cracking: return "brittle tensile failure, critical strain σ/E"
    case .arrest: return "crack arrest at the DEJ is measured, Biomaterials 2010"
    case .lesion: return "abrasion and erosion are established; abfraction is not"
    }
}

/// The three rows of the evidence bar for a beat. The bar must CHANGE across
/// the loop — that is the whole reason steps 13 and 14 were right to drop it and
/// this step is right to bring it back — so a test compares these row lists
/// across beats and fails if they are all the same.
func evidenceRowsFor(_ beat: Beat) -> [(Standing, String)] {
    switch beat {
    case .anatomy:
        return [(.measured, "enamel 84 GPa, dentin 18 GPa"),
                (.model, "section from published anatomy"),
                (.model, "body is that section revolved")]
    case .chewing:
        return [(.measured, "20–120 N for mastication"),
                (.measured, "ligament 0.2 mm, crest 1.5 mm"),
                (.simulated, "displacement, this lattice")]
    case .bruxing:
        return [(.measured, "up to ~1000 N in sleep bruxism"),
                (.simulated, "stress field, 2D linear elastic"),
                (.model, "Poisson 1/3, forced by the lattice")]
    case .cracking:
        return [(.simulated, "failure at critical strain σ/E"),
                (.model, "10% scatter in critical strain"),
                (.measured, "enamel cracks in ordinary life")]
    case .arrest:
        return [(.measured, "arrest at the DEJ, Biomaterials 2010"),
                (.simulated, "arrest rate over seeded runs"),
                (.model, "DEJ as a 0.18 mm graded band")]
    case .lesion:
        return [(.contested, "abfraction: no consensus"),
                (.measured, "abrasion and erosion are established"),
                (.simulated, "cervical stress concentration")]
    }
}

let toothFacts: [Fact] = [
    Fact(name: "enamel mineral fraction", value: "~96%", standing: .measured,
         source: "carbonated hydroxyapatite, 4% water and organic"),
    Fact(name: "enamel modulus", value: "84 GPa", standing: .measured,
         source: "within the 3–7× dentin band, Int. J. Oral Sci. review 2014"),
    Fact(name: "dentin modulus", value: "18 GPa", standing: .measured,
         source: "Int. J. Oral Sci. review 2014"),
    Fact(name: "dentin fracture toughness", value: "3.08 MPa·√m ±10%", standing: .measured,
         source: "normal coronal dentin"),
    Fact(name: "enamel toughness anisotropy", value: "varies ×3 with rod angle", standing: .measured,
         source: "Acta Biomaterialia — not a single value"),
    Fact(name: "periodontal ligament thickness", value: "0.15–0.38 mm", standing: .measured,
         source: "narrowest at mid-root; 0.2 mm modelled"),
    Fact(name: "periodontal ligament modulus", value: "0.01–1750 MPa", standing: .contested,
         source: "five orders of magnitude across methods; 50 MPa modelled here"),
    Fact(name: "alveolar crest position", value: "1–2 mm below the CEJ", standing: .measured,
         source: "healthy periodontium; 1.5 mm modelled"),
    Fact(name: "cortical bone modulus", value: "13.7 GPa", standing: .measured,
         source: "standard value in dental finite element work"),
    Fact(name: "chewing force", value: "20–120 N", standing: .measured, source: "mastication"),
    Fact(name: "maximum bite, molars", value: "300–600 N", standing: .measured,
         source: "healthy adults"),
    Fact(name: "bruxing load", value: "up to ~1000 N", standing: .measured,
         source: "can exceed the daytime maximum in sleep bruxism"),
    Fact(name: "normal wear, molars", value: "~29 µm/year", standing: .measured,
         source: "~15 µm/yr premolars, non-bruxers"),
    Fact(name: "cracks arrest at the DEJ", value: "—", standing: .measured,
         source: "modulus-mismatch mechanism, Biomaterials 2010"),
    Fact(name: "stress concentrates at the CEJ", value: "—", standing: .simulated,
         source: "robust across published FEA; lateral load exceeds axial"),
    Fact(name: "this stress field", value: "—", standing: .simulated,
         source: "solved here: triangular spring lattice, 2D, linear elastic"),
    Fact(name: "lattice Poisson ratio", value: "1/3, forced", standing: .model,
         source: "central-force triangular lattice; enamel ~0.30, dentin ~0.31"),
    Fact(name: "ligament Poisson ratio", value: "1/3, forced; real ~0.45", standing: .model,
         source: "a central-force lattice cannot be near-incompressible"),
    Fact(name: "critical strains", value: "σ/E per tissue", standing: .model,
         source: "the ratio between them is what decides arrest"),
    Fact(name: "critical-strain scatter", value: "±10%, lognormal", standing: .model,
         source: "a perfect lattice cracks along its own symmetry axes"),
    Fact(name: "tooth geometry", value: "—", standing: .model,
         source: "published sectional anatomy, not a scan of a real tooth"),
    Fact(name: "the three-dimensional body", value: "—", standing: .model,
         source: "that section revolved: two cusps become one ring"),
    Fact(name: "abfraction as a cause of cervical lesions", value: "—", standing: .contested,
         source: "limited evidence, no consensus, confounded with abrasion and erosion"),
]

// MARK: - an analytic yardstick for the neck
//
// Above the alveolar crest the tooth is a short cantilever. Euler–Bernoulli on
// the CEJ section gives the bending stress on each face from the moment of the
// contact force about the section centroid, plus the axial term. That is the
// DENTIN-face stress; the enamel skin on top of it is strain-compatible with the
// dentin it sits on, so it carries E_enamel/E_dentin = 4.7 times as much.
//
// This exists so the lattice's cervical number can be checked against something
// that is not another lattice, and a test does exactly that. Written by the
// coordinating session; kept because it caught a real thing — see
// `enamelSkinStress`.

struct CervicalBeam {
    var momentNm: Double
    var dentinBuccal: Double     // Pa, tension positive, on the +x face
    var dentinLingual: Double
    var enamelBuccal: Double
    var enamelLingual: Double
    var tensionOnBuccal: Bool
}

func cervicalBeamStress(_ lc: LoadCase) -> CervicalBeam {
    let halfMM: Float = outerHalfWidth(y: 0)
    let widthMM: Double = Double(halfMM) * 2
    let w: Double = widthMM * 1e-3
    let t: Double = sectionThickness
    let inertia: Double = t * w * w * w / 12
    let area: Double = t * w
    let fx: Double = lc.newtons * Double(lc.direction.x)
    let fy: Double = lc.newtons * Double(lc.direction.y)
    let xc: Double = Double(lc.contactX) * 1e-3
    let yc: Double = Double(occlusalHeight(x: lc.contactX)) * 1e-3
    let moment: Double = fx * yc - fy * xc
    let axial: Double = fy / area
    let half: Double = w / 2
    let plus: Double = -moment * half / inertia + axial
    let minus: Double = moment * half / inertia + axial
    let skin: Double = enamelModulus / dentinModulus
    return CervicalBeam(momentNm: moment, dentinBuccal: plus, dentinLingual: minus,
                        enamelBuccal: plus * skin, enamelLingual: minus * skin,
                        tensionOnBuccal: plus > minus)
}
