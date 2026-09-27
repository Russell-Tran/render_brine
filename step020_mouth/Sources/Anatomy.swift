// The lower arch as numbers: fourteen teeth (sixteen with the wisdom teeth),
// where each one sits, and what colour each tissue is. Nothing here touches
// the GPU; it is the part a test can read against the textbooks.
//
// Millimetres throughout. y is up, and y = 0 is the occlusal plane, where the
// incisal edges and cusp tips meet their opponents. x runs across the mouth
// (+x is the patient's RIGHT: the renderer is right-handed, so facing the
// patient from the front +x lands on the viewer's left, and the patient's
// right is on the viewer's left — step 21 caught this written backwards),
// z runs back towards the throat, and the midline
// between the two central incisors is x = 0, z = 0.

import Foundation
import simd

// MARK: - the teeth

/// What a crown's top looks like. The raw value goes straight into the kernel.
enum CrownKind: Int, CaseIterable {
    case incisor = 0        // a straight cutting edge
    case canine = 1         // one pointed cusp
    case firstPremolar = 2  // a big buccal cusp and a small lingual one
    case secondPremolar = 3 // one buccal cusp, two lingual
    case firstMolar = 4     // five cusps
    case secondMolar = 5    // four cusps
}

/// One mandibular tooth, in the numbers the dental anatomy tables give.
///
/// Every value is from the table of mandibular crown measurements in Nelson,
/// *Wheeler's Dental Anatomy, Physiology and Occlusion* (the "Measurement of
/// the teeth" table at the head of each tooth's chapter). "Width" is
/// mesiodistal — along the arch; "depth" is labiolingual — across it.
/// "cejCurve" is how far the cementoenamel junction rises between teeth,
/// relative to its lowest point mid-facial, and it is what makes gums scallop:
/// the gingiva follows the junction, so a 3 mm curve under an incisor is a
/// 3 mm peak of gum between two incisors.
struct ToothSpec {
    var name: String
    var kind: CrownKind
    var crownHeight: Float      // cervico-incisal / cervico-occlusal
    var width: Float            // mesiodistal, at the contact areas
    var cervicalWidth: Float    // mesiodistal, at the cervical line
    var depth: Float            // labiolingual, at the height of contour
    var cervicalDepth: Float    // labiolingual, at the cervical line
    var cejCurve: Float         // mesial curvature of the cervical line
}

let mandibularTeeth: [ToothSpec] = [
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
]

/// The mandibular third molar — the wisdom tooth, #17 on the patient's left
/// and #32 on the right — for a camera that travels to the back of the arch
/// (step 21). Not in `mandibularTeeth`, so step 20's fourteen are untouched;
/// `placeTeeth(thirdMolars: true)` adds one per side.
///
/// Sizes are the mandibular third molar row of the same Wheeler's table (Nelson,
/// *Wheeler's Dental Anatomy*, chapter "The Permanent Mandibular Molars"):
/// crown 7.0 tall, 10.0 mesiodistally (7.5 at the cervix), 9.5 buccolingually
/// (9.0 at the cervix), cervical line curving 1.0. UNVERIFIED: these are the
/// values as quoted from the book, and no copy of the table could be reached
/// online to check them against. They are plausible beside the rows that are
/// checked: as tall as the second molar and a little narrower each way.
///
/// Its crown is drawn as a second molar's, four cusps: a mandibular third
/// molar "could resemble a four-cusped mandibular second molar or a five-cusped
/// mandibular first molar" (Scheid & Weiss, *Woelfel's Dental Anatomy*, "Maxillary
/// and mandibular third molar type traits"). MODEL: the four-cusp form of the
/// two. Erupted and in line with its neighbours, as in a mouth that had room
/// for it — MODEL, since many are impacted or tipped.
let mandibularThirdMolar = ToothSpec(name: "third molar", kind: .secondMolar, crownHeight: 7.0, width: 10.0,
                                     cervicalWidth: 7.5, depth: 9.5, cervicalDepth: 9.0, cejCurve: 1.0)

// MARK: - the arch

// The arch form is Bonwill–Hawley, the oldest one in orthodontics and still the
// simplest: the six anterior teeth sit on a circle whose radius is the combined
// width of one central incisor, one lateral and one canine (Hawley, *Dental
// Cosmos* 1905), and the premolars and molars run back from it in straight
// lines. The radius is DERIVED from the table above, not typed.
let hawleyRadius: Float = mandibularTeeth[0].width + mandibularTeeth[1].width + mandibularTeeth[2].width

/// Centre-to-centre distance between the two first molars. MODEL: adult
/// mandibular intermolar widths measured across the mesiobuccal cusps are in
/// the low forties (Bishara et al., *Am J Orthod Dentofacial Orthop* 1997,
/// "Arch width changes from 6 weeks to 45 years of age"); a molar's centre sits
/// about 2.5 mm lingual of its buccal cusps, so the centres are ~5 mm closer.
let firstMolarSpan: Float = 38.0

/// One tooth, placed. `tangent` points distally along the arch, `outward`
/// points labially or buccally — away from the tongue.
struct PlacedTooth {
    var spec: ToothSpec
    var centre: SIMD2<Float>    // (x, z)
    var tangent: SIMD2<Float>
    var outward: SIMD2<Float>
    var side: Float             // +1 patient's right, −1 patient's left
}

/// Where along one half of the arch a distance `s` from the midline lands, and
/// which way the arch runs there. Circle up to the end of the canine, straight
/// line after it, aimed so the first molar lands at `firstMolarSpan`.
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

/// Arc length from the midline to each tooth's centre on one side: each tooth
/// starts where the one in front of it ends, so neighbours touch.
let archCentres: [Float] = {
    var out: [Float] = []
    var edge: Float = 0
    for t in mandibularTeeth {
        out.append(edge + t.width / 2)
        edge += t.width
    }
    return out
}()

/// Arc length from the midline to a third molar's centre: it starts where the
/// second molar ends, so they touch, like every other pair of neighbours. The
/// straight line behind the canine simply continues.
let thirdMolarArc: Float = archCentres[6] + mandibularTeeth[6].width / 2 + mandibularThirdMolar.width / 2

/// One tooth set on the arch at arc length `arc` from the midline, on `side`.
func placeTooth(_ spec: ToothSpec, at arc: Float, side: Float) -> PlacedTooth {
    let (p, t) = archPoint(arc)
    let centre = SIMD2<Float>(p.x * side, p.y)
    let tangent = simd_normalize(SIMD2<Float>(t.x * side, t.y))
    // Perpendicular to the arch, on the side away from its centre.
    var outward = SIMD2<Float>(tangent.y, -tangent.x)
    let towardMidline = SIMD2<Float>(-centre.x, 20 - centre.y)
    if simd_dot(outward, towardMidline) > 0 { outward = -outward }
    return PlacedTooth(spec: spec, centre: centre, tangent: tangent, outward: outward, side: side)
}

/// All fourteen: the patient's left side (−x) first, then the right.
///
/// With `thirdMolars` on, the two wisdom teeth come after them, so no other
/// tooth's index moves: index 14 is #17, the patient's LEFT third molar (−x),
/// and index 15 is #32, the right. Off — the default, and what step 20 and
/// step 22 draw — the list is the same fourteen, in the same order, from the
/// same arithmetic, as before the option existed.
func placeTeeth(thirdMolars: Bool = false) -> [PlacedTooth] {
    var out: [PlacedTooth] = []
    for side in [Float(-1), Float(1)] {
        for (i, spec) in mandibularTeeth.enumerated() {
            out.append(placeTooth(spec, at: archCentres[i], side: side))
        }
    }
    if thirdMolars {
        for side in [Float(-1), Float(1)] {
            out.append(placeTooth(mandibularThirdMolar, at: thirdMolarArc, side: side))
        }
    }
    return out
}

/// Where the wisdom teeth sit in `placeTeeth(thirdMolars: true)`.
let tooth17Index: Int = 14
let tooth32Index: Int = 15

// MARK: - the gum

/// The free gingival margin sits about 1 mm coronal to the cementoenamel
/// junction in a healthy adult: the gingival sulcus (0.69 mm) above the
/// junctional epithelium's upper edge (Gargiulo, Wentz & Orban, *J Periodontol*
/// 1961).
let gumMarginAboveCEJ: Float = 1.0

/// How far the papilla climbs above the junction's own curve between teeth.
/// MODEL, bounded by Tarnow et al. (*J Periodontol* 1992): the papilla fills
/// the embrasure when the contact point is within 5 mm of the bone crest, so a
/// healthy young arch shows no dark triangles. Chosen to fill without reaching
/// the contacts.
let papillaExtra: Float = 1.0

/// How thick the gum is over the root once past its collar. MODEL: a
/// mandibular alveolar process is roughly 2–3 mm wider than the root on each
/// side at the premolars, soft tissue included.
let gumWidth: Float = 2.5

/// How fast the ridge keeps widening below the collar, mm per mm of depth, so
/// the tapering roots sit in a solid jaw rather than a comb. MODEL.
let gumSlope: Float = 0.12

// MARK: - colour
//
// Colours are specified as measured CIELAB and converted, so the numbers a
// dentist would recognise are the numbers in the file.

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

// Middle third of the crown, by tooth type. Measured in vivo with a
// spectrophotometer (VITA Easyshade) on 160 adults: central incisor L* 73.0,
// a* −0.5, b* 14.5; lateral 70.0, 0.2, 18.6; canine 65.1, 1.4, 23.6 —
// canines really are darker and yellower than incisors (Measurement and
// analysis of maxillary anterior teeth color in the Chinese population, 2023,
// PMC10155942). Those are maxillary teeth; the mandibular ones are taken as
// the same. Premolars and molars were not measured there and get the lateral
// incisor's value — MODEL.
let middleThirdLab: [CrownKind: SIMD3<Double>] = [
    .incisor: SIMD3(71.5, -0.15, 16.5),   // mean of the central and lateral
    .canine: SIMD3(65.1, 1.4, 23.6),
    .firstPremolar: SIMD3(70.0, 0.2, 18.6),
    .secondPremolar: SIMD3(70.0, 0.2, 18.6),
    .firstMolar: SIMD3(70.0, 0.2, 18.6),
    .secondMolar: SIMD3(70.0, 0.2, 18.6),
]

/// Towards the neck, a* and b* rise and translucency falls (Hasegawa, Ikeda &
/// Kawaguchi, *J Prosthet Dent* 2000, "Color and translucency of in vivo
/// natural central incisors"): the enamel thins and the dentin shows. The
/// DIRECTION is theirs; the size of the shift is MODEL.
let cervicalShift: SIMD3<Double> = SIMD3(-4.0, 1.5, 5.0)

/// Incisal enamel, measured at the mid-incisal third of central incisors:
/// L* 73.5, a* 2.2, b* 11.9 (Wee et al., *J Prosthet Dent* 2022, "Color and
/// translucency of enamel in vital maxillary central incisors"). That is at
/// infinite thickness; in the mouth the edge is thin and the dark oral cavity
/// shows through it, which the shader adds as translucency.
let incisalLab: SIMD3<Double> = SIMD3(73.5, 2.2, 11.9)

/// Keratinized gingiva, all subjects: L* 52.9, a* 23.3, b* 14.9 (Ho, Ghinea,
/// Herrera, Angelov & Paravina, *Sci Rep* 2015, "Color Range and Color
/// Distribution of Healthy Human Gingiva").
let gingivaLab: SIMD3<Double> = SIMD3(52.9, 23.3, 14.9)

/// The tongue's dorsum. MODEL: no measurement to hand, so the gingiva's value
/// darkened and made redder, which is what the non-keratinized, more vascular
/// tissue looks like beside it.
let tongueLab: SIMD3<Double> = SIMD3(46.0, 27.0, 15.0)

// MARK: - reflectance, from refractive indices

/// Normal-incidence Fresnel reflectance between two media.
func fresnelF0(_ n1: Double, _ n2: Double) -> Float {
    let r: Double = (n1 - n2) / (n1 + n2)
    return Float(r * r)
}

/// Enamel's refractive index, 1.62 (Spitzer & ten Bosch, *Calcif Tissue Res*
/// 1975, on human and bovine enamel). Saliva is taken as water, 1.333.
let enamelIndex: Double = 1.62
let salivaIndex: Double = 1.333

/// A tooth in a mouth is wet. So its sharp highlight comes from the saliva
/// film's top surface — air to water, 2.0% — and the enamel beneath reflects
/// only against water, not air: 0.94%, a sixth of the 5.6% a dry tooth would
/// show. That is why wet teeth have crisp small highlights over a soft body.
let filmF0: Float = fresnelF0(1.0, salivaIndex)
let enamelUnderFilmF0: Float = fresnelF0(salivaIndex, enamelIndex)
let dryEnamelF0: Float = fresnelF0(1.0, enamelIndex)
