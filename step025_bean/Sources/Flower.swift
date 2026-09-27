// The bean flower as numbers: every length, where each part sits, and the
// paths the thin things follow. Nothing here touches the GPU; it is the part a
// test can read against the floras.
//
// Millimetres throughout. The flower lies on its side with its axis along +x,
// from the pedicel on the left to the coiled keel on the right. +y is up — the
// dorsal side, where the banner (standard) is — and +z points at the camera,
// so the plane z = 0 is the flower's plane of symmetry and the cutaway removes
// everything in front of it.
//
// Sources, checked for this step:
//
//   FTEA — Flora of Tropical East Africa, via Plants of the World Online and
//     World Flora Online (wfo-0000183282, wfo-4000029154). Genus: "keel linear,
//     beak long, spirally coiled through 1-5 turns ... Style spirally coiled
//     through at least 360°; stigma oblique." Species: "Standard oblate-oblong,
//     1(–1·9) cm long, 1·1 cm wide ... keel spirally incurved, ± 2·2 cm long";
//     calyx "tube 2–3 mm long; lobes ± 1 mm long"; "bracteoles conspicuous,
//     ovate-lanceolate, 5–6 mm long, 2·5–3·5 mm wide"; "pedicels 3–10 mm
//     long"; pods "11–12·5(–20) cm long, 1–1·3 cm wide, ... (5–)10–12-seeded".
//   Flora of China (same WFO page): "keel ca. 1 cm, apex spirally twisted.
//     Ovary pubescent ... Seeds 4-10."
//   Gleason's Northeastern Flora (WFO genus page): "stamens 10, diadelphous;
//     style bearded along the upper side, spirally coiled and thickened beyond
//     the middle".
//   PalDat, Phaseolus vulgaris (paldat.org, 2021): monad, "size of hydrated
//     pollen (LM) 41-50 µm", "shortest polar axis in equatorial view 36-40 µm",
//     triporate, spheroidal, circular in polar view, reticulate.
//   Hoc, Amela García & others, Rev. Biol. Trop. 1999, "Flower biology and
//     reproductive system of Phaseolus vulgaris var. aborigineus" (SciELO):
//     nectar collects "en la base del tubo formado por 9 de los 10 estambres, el
//     restante (vexilar) no está soldado"; the stigma is receptive in the bud;
//     pollen is "presented on stylar subapical trichomes"; 5.4 ovules per fruit.
//   Lord & Kohorn, Am. J. Bot. 73:70–78 (1986), tepary bean: wet non-papillate
//     stigma; "Anthers dehisce and the tricolporate pollen is released onto the
//     receptive stigma one day before anthesis"; pollen tubes run along "the
//     ventral side of the stylar canal and upper ovary".
//   McGregor, Insect Pollination of Cultivated Crop Plants (USDA Handbook 496,
//     1976), P. vulgaris: "The anthers dehisce the evening of the day before
//     the flower opens"; natural crossing "0 to 1.4 percent" (Kristofferson),
//     "0 to 10 percent" (Emerson).

import Foundation
import simd

// MARK: - what can be broken on purpose

/// Each mutant must make the test suite fail; `make mutants` checks that.
enum Mutant: String {
    case none
    case tubeLeaves = "tube_leaves"   // the pollen tube takes a short cut out of the style
    case open                         // the flower drawn fully open, not a bud
    case allFree = "all_free"         // ten free stamens, no 9 + 1 sheath

    static var fromEnvironment: Mutant {
        Mutant(rawValue: ProcessInfo.processInfo.environment["BEAN_MUTANT"] ?? "none") ?? .none
    }
}

// MARK: - cited sizes

/// FTEA: standard 1(–1.9) cm long.
let standardLengthRange: ClosedRange<Float> = 10.0...19.0
/// FTEA: keel "± 2·2 cm long". Flora of China says "ca. 1 cm" — a factor of two
/// apart, most likely one measured along the coil and one across it. The keel
/// here is measured along its centreline and held to FTEA's figure.
let keelLengthCited: Float = 22.0
/// FTEA: the keel's beak coils "through 1-5 turns"; the style "through at
/// least 360°".
let keelTurnsCitedMinimum: Float = 1.0
let styleCoilCitedMinimumDegrees: Float = 360.0
/// MODEL within FTEA's 1–5: enough to make the coil read as a coil at this
/// bud size without the turns touching.
let keelTurns: Float = 1.75

/// PalDat: hydrated grains 41–50 µm across the equator, 36–40 µm pole to pole —
/// slightly oblate. The grains drawn are the middle of each range.
let pollenEquatorialRange: ClosedRange<Float> = 0.041...0.050
let pollenPolarRange: ClosedRange<Float> = 0.036...0.040
let pollenEquatorialDiameter: Float = 0.045
let pollenPolarDiameter: Float = 0.038
/// PalDat: triporate. (Lord & Kohorn call tepary pollen tricolporate; an older
/// description of P. vulgaris 'Red Mexican' calls it tricolpate with round
/// pores. Three equatorial pores in every account; that is what is drawn.)
let pollenPores: Int = 3

/// Six ovules. Within Flora of China's 4–10 seeds and FTEA's (5–)10–12, and
/// next to the 5.4 ovules per fruit Hoc et al. counted in the wild variety.
let ovuleCount: Int = 6

/// Pollen tube radius. MODEL: angiosperm pollen tubes are 5 µm (Arabidopsis)
/// to 20 µm (Camellia) across (Geitmann lab, Front. Plant Sci. 2020, "Mechanics
/// of Pollen Tube Elongation: A Perspective"); 10 µm is taken for the bean.
let pollenTubeRadius: Float = 0.005

/// The pod: the same ovary about two weeks after the flower opened. Pods of
/// common bean elongate rapidly until 15 days after anthesis, and rapid seed
/// growth begins only after that (Plants 2020, "Transcriptional Dynamics and
/// Candidate Genes Involved in Pod Maturation of Common Bean", PMC7238275) — so
/// at 14 days the pod is near full length with small seeds: the snap-bean
/// stage. Size: FTEA's 11–12.5 × 1–1.3 cm.
let podDaysAfterAnthesis: Int = 14
let podLength: Float = 120.0
let podWidth: Float = 11.0

// MARK: - the flower's own layout (MODEL unless marked)

/// The keel and style share one centreline, the spine: straight along the
/// ovary, then a nearly flat spiral. Its shape is MODEL, fixed by what has to
/// fit: the turns may not touch, the whole must sit inside a bud of about the
/// standard's length.
let spineStartX: Float = 2.0
let coilStartX: Float = 9.0
let coilOuterRadius: Float = 2.5
/// Each turn comes 1.0 mm closer to the centre, which is exactly enough room
/// for two keel tubes side by side (outer radius ≤ 0.5 mm each).
let coilInnerRadius: Float = coilOuterRadius - keelTurns * 1.0
/// A tenth of a millimetre per turn out of plane: nearly flat, so one cut plane
/// opens every turn.
let coilPitch: Float = 0.1
/// The keel ends a little beyond the stigma, closed.
let keelBeyondStigma: Float = 0.28

/// The ovary: from its base in the receptacle to where the style begins. MODEL:
/// no measured length found; set so the ovary, its sheath of filaments and the
/// keel's broad base fit inside the bud.
let ovaryStartX: Float = 0.6
let ovaryEndX: Float = 7.2
let ovaryHalfHeight: Float = 0.5
let ovaryHalfWidth: Float = 0.38
let ovaryWall: Float = 0.1
let ovaryTaperStartX: Float = 6.6

/// Ovules in a row, hanging from the upper (ventral) side of the locule.
/// Sizes MODEL.
let ovuleFirstX: Float = 1.4
let ovuleSpacing: Float = 0.92
let ovuleSemiAxes = SIMD3<Float>(0.30, 0.20, 0.17)
let ovuleCentreY: Float = 0.08
/// The ovule the pollen tube reaches: the one nearest the style.
let fertilisedOvule: Int = ovuleCount - 1

/// The staminal sheath: nine filaments fused into a tube round the ovary, open
/// along the top where the tenth, vexillary, stamen lies free (Gleason:
/// diadelphous; Hoc et al.: nine fused, the vexillary not). Sizes MODEL.
let sheathRadius: Float = 0.8
let sheathStartX: Float = 1.2
let sheathEndX: Float = 6.4
/// Half-angle of the slit along the sheath's top, degrees.
let sheathSlitHalfAngle: Float = 22.0
let filamentRadius: Float = 0.017
/// Where the filaments run inside the keel beak, measured from the style's
/// axis: close round the style along the coil, then out to where the anthers
/// sit, round the brush. MODEL.
let filamentOffsetAlongStyle: Float = 0.15
let filamentOffsetInBeak: Float = 0.27
/// Anthers: small, pale, dehisced. Half-length along the filament, half-width,
/// half-thickness. MODEL.
let antherSemiAxes = SIMD3<Float>(0.19, 0.10, 0.07)

/// Style: slender at the base and "thickened beyond the middle" (Gleason).
/// Radii MODEL.
let styleBaseRadius: Float = 0.07
let styleThickRadius: Float = 0.11
/// Stigma: an oblique knob at the tip (FTEA "stigma oblique"), wet and
/// non-papillate (Lord & Kohorn). Size MODEL.
let stigmaRadius: Float = 0.13

/// The brush of stylar hairs just below the stigma, on the style's upper side
/// (Gleason "bearded along the upper side"; Hoc et al. "stylar subapical
/// trichomes"). On the coiled style the upper side faces the coil's centre.
/// Hair length, spacing and extent MODEL.
let brushLength: Float = 1.5
let brushGapBelowStigma: Float = 0.15
let hairLength: Float = 0.13
let hairRowSpacing: Float = 0.03
let hairColumns: Int = 7
let hairColumnStepDegrees: Float = 20.0
let hairLeanDegrees: Float = 25.0

/// The keel's inner radius: broad round the ovary, narrowing into the beak.
/// Petal thickness MODEL.
let keelBaseRadius: Float = 1.05
let keelBeakRadius: Float = 0.45
let keelTipRadius: Float = 0.36
let keelTaperStartX: Float = 6.9
let keelTaperEndX: Float = 8.2
let petalThickness: Float = 0.06

// MARK: - the spine

struct SpinePoint {
    var p: SIMD3<Float>
    var t: SIMD3<Float>
    var n: SIMD3<Float>      // toward the coil's centre (the style's upper side)
    var b: SIMD3<Float>      // t × n: toward the camera
    var s: Float             // arc length from the spine's start
    var keelRadius: Float    // inner radius of the keel's cavity
    var styleRadius: Float   // −1 where there is no style
}

func smoothstep(_ a: Float, _ b: Float, _ x: Float) -> Float {
    let t: Float = min(max((x - a) / (b - a), 0), 1)
    return t * t * (3 - 2 * t)
}

/// The spiral's angle where the style ends at the stigma.
let stigmaAngle: Float = keelTurns * 2 * Float.pi

func coilRadius(_ phi: Float) -> Float {
    coilOuterRadius - (coilOuterRadius - coilInnerRadius) * phi / stigmaAngle
}

let coilCentre = SIMD3<Float>(coilStartX, coilOuterRadius, 0)

func coilPoint(_ phi: Float) -> SIMD3<Float> {
    let r: Float = coilRadius(phi)
    let z: Float = -coilPitch * phi / (2 * Float.pi)
    return coilCentre + SIMD3<Float>(r * sin(phi), -r * cos(phi), z)
}

/// The spine, sampled every ~0.08 mm, with a rotation-minimising frame.
let spine: [SpinePoint] = buildSpine()

/// Arc length at the ovary's tip, where the style begins, and at the stigma.
let styleStartS: Float = spine.first(where: { $0.p.x >= ovaryEndX - 0.1 })!.s
let stigmaS: Float = { _ = spine.count; return spineStigmaS }()

private var spineStigmaS: Float = 0

private func buildSpine() -> [SpinePoint] {
    var pts: [SIMD3<Float>] = []
    var x: Float = spineStartX
    while x < coilStartX - 1e-4 {
        pts.append(SIMD3<Float>(x, 0, 0))
        x += 0.08
    }
    let endPhi: Float = stigmaAngle + keelBeyondStigma / coilInnerRadius
    var phi: Float = 0
    var stigmaIndex: Int = 0
    while phi < endPhi {
        if phi <= stigmaAngle { stigmaIndex = pts.count }
        pts.append(coilPoint(phi))
        phi += 0.08 / coilRadius(min(phi, stigmaAngle))
    }
    // Land a vertex exactly on the stigma so the style ends there.
    pts.insert(coilPoint(stigmaAngle), at: stigmaIndex + 1)
    pts.append(coilPoint(endPhi))

    var out: [SpinePoint] = []
    var s: Float = 0
    var n = SIMD3<Float>(0, 1, 0)
    for i in 0..<pts.count {
        if i > 0 { s += simd_distance(pts[i], pts[i - 1]) }
        let a: SIMD3<Float> = pts[max(i - 1, 0)]
        let b: SIMD3<Float> = pts[min(i + 1, pts.count - 1)]
        let t: SIMD3<Float> = simd_normalize(b - a)
        n = simd_normalize(n - t * simd_dot(n, t))
        let bn: SIMD3<Float> = simd_cross(t, n)
        let taper: Float = smoothstep(keelTaperStartX, keelTaperEndX, pts[i].x)
        let keelR: Float = keelBaseRadius + (keelBeakRadius - keelBaseRadius) * taper
        out.append(SpinePoint(p: pts[i], t: t, n: n, b: bn, s: s, keelRadius: keelR, styleRadius: -1))
    }
    // In the coil the beak narrows from its beak radius to its tip radius.
    let coilStart: Int = out.firstIndex(where: { $0.p.x >= coilStartX - 1e-4 })!
    let stigmaVertex: Int = stigmaIndex + 1
    spineStigmaS = out[stigmaVertex].s
    let startS: Float = out[coilStart].s
    let endS: Float = out[out.count - 1].s
    for i in coilStart..<out.count {
        let u: Float = (out[i].s - startS) / (endS - startS)
        out[i].keelRadius = keelBeakRadius + (keelTipRadius - keelBeakRadius) * u
    }
    // Style radius from the ovary's tip to the stigma.
    // The style starts a little inside the ovary's tip, so the two join.
    let s0: Float = out.first(where: { $0.p.x >= ovaryEndX - 0.1 })!.s
    let s1: Float = spineStigmaS
    for i in 0..<out.count where out[i].s >= s0 - 1e-5 && i <= stigmaVertex {
        let u: Float = (out[i].s - s0) / (s1 - s0)
        let thick: Float = smoothstep(0.3, 0.6, u)
        out[i].styleRadius = styleBaseRadius + (styleThickRadius - styleBaseRadius) * thick
            - 0.012 * smoothstep(0.9, 1.0, u)
    }
    return out
}

/// Position and frame on the spine at arc length s, by linear interpolation.
func spineAt(_ s: Float) -> SpinePoint {
    var lo: Int = 0
    var hi: Int = spine.count - 1
    while hi - lo > 1 {
        let mid: Int = (lo + hi) / 2
        if spine[mid].s <= s { lo = mid } else { hi = mid }
    }
    let a: SpinePoint = spine[lo]
    let b: SpinePoint = spine[hi]
    let h: Float = min(max((s - a.s) / max(b.s - a.s, 1e-6), 0), 1)
    var out: SpinePoint = a
    out.p = a.p + (b.p - a.p) * h
    out.t = simd_normalize(a.t + (b.t - a.t) * h)
    out.n = simd_normalize(a.n + (b.n - a.n) * h)
    out.b = simd_cross(out.t, out.n)
    out.s = s
    out.keelRadius = a.keelRadius + (b.keelRadius - a.keelRadius) * h
    out.styleRadius = a.styleRadius + (b.styleRadius - a.styleRadius) * h
    return out
}

/// How far the spine's tangent turns, in degrees, between two arc lengths,
/// measured in the plane of the coil. A full turn of a spiral is 360°.
func turning(from s0: Float, to s1: Float) -> Float {
    var total: Float = 0
    var prev: SIMD3<Float>? = nil
    for p in spine where p.s >= s0 && p.s <= s1 {
        if let q = prev {
            let a: Float = atan2(q.x * p.t.y - q.y * p.t.x, q.x * p.t.x + q.y * p.t.y)
            total += a
        }
        prev = p.t
    }
    return abs(total) * 180 / Float.pi
}

/// The keel's centreline length: along the spine from where the keel leaves
/// the calyx to its closed tip.
let keelStartX: Float = 2.6
var keelCentrelineLength: Float {
    let start: Float = spine.first(where: { $0.p.x >= keelStartX })!.s
    return spine[spine.count - 1].s - start
}

// MARK: - the stigma, and where the parts round it sit

let stigmaFrame: SpinePoint = spineAt(stigmaS)
/// Oblique: the knob sits a little forward and to the upper side of the tip.
let stigmaCentre: SIMD3<Float> = stigmaFrame.p + stigmaFrame.t * 0.03 + stigmaFrame.n * 0.02

// MARK: - stamens

struct Stamen {
    var angleDegrees: Float    // round the ovary; 0 is the top, the vexillary side
    var fused: Bool            // part of the sheath, or free
    var antherS: Float         // arc length along the spine of the anther's centre
}

/// Ten stamens, 36° apart round the ovary. Nine fused into the sheath and one
/// free — unless the mutant makes all ten free.
func makeStamens(_ mutant: Mutant) -> [Stamen] {
    var out: [Stamen] = []
    for i in 0..<10 {
        let angle: Float = Float(i) * 36
        // Staggered along the style so ten anthers fit round it; the two that
        // would stand between the camera and the brush sit further back. MODEL.
        var back: Float = i % 2 == 0 ? 0.34 : 0.72
        if angle > 50 && angle < 130 { back = 1.05 }
        let fused: Bool = mutant == .allFree ? false : i != 0
        out.append(Stamen(angleDegrees: angle, fused: fused, antherS: stigmaS - back))
    }
    return out
}

/// Whether the geometry draws a sheath: only if nine stamens are fused.
func hasSheath(_ st: [Stamen]) -> Bool { st.filter { $0.fused }.count == 9 }

/// A thin tube as a polyline with a radius at each vertex.
struct Chain {
    var points: [SIMD3<Float>]
    var radii: [Float]
    var material: UInt32
}

let materialOvule: UInt32 = 6
let materialStamen: UInt32 = 7
let materialTube: UInt32 = 12

/// Where a filament runs at arc length s: on the sheath round the ovary, then
/// closing in on the style inside the beak.
func filamentOffset(_ s: Float) -> Float {
    let x: Float = spineStartX + s        // valid on the straight part
    let beakS: Float = spine.first(where: { $0.p.x >= 7.5 })!.s
    if s < beakS {
        let u: Float = smoothstep(sheathEndX, 7.5, x)
        return sheathRadius + (filamentOffsetAlongStyle - sheathRadius) * u
    }
    let out: Float = smoothstep(stigmaS - 2.4, stigmaS - 1.6, s)
    return filamentOffsetAlongStyle + (filamentOffsetInBeak - filamentOffsetAlongStyle) * out
}

func offsetPoint(_ f: SpinePoint, radius: Float, angleDegrees: Float) -> SIMD3<Float> {
    let a: Float = angleDegrees * Float.pi / 180
    return f.p + (f.n * cos(a) + f.b * sin(a)) * radius
}

struct Anther {
    var centre: SIMD3<Float>
    var axis: SIMD3<Float>
    var side: SIMD3<Float>
}

func makeAnthers(_ st: [Stamen]) -> [Anther] {
    st.map { m in
        let f: SpinePoint = spineAt(m.antherS)
        let c: SIMD3<Float> = offsetPoint(f, radius: filamentOffsetInBeak, angleDegrees: m.angleDegrees)
        let a: Float = m.angleDegrees * Float.pi / 180
        let radial: SIMD3<Float> = f.n * cos(a) + f.b * sin(a)
        let side: SIMD3<Float> = simd_normalize(simd_cross(f.t, radial))
        return Anther(centre: c, axis: f.t, side: side)
    }
}

/// Each filament from its base to its anther.
func filamentChains(_ st: [Stamen]) -> [Chain] {
    st.map { m in
        var pts: [SIMD3<Float>] = []
        let a: Float = m.angleDegrees * Float.pi / 180
        // Before the spine starts, the filament runs straight along the sheath.
        var x: Float = sheathStartX
        while x < spineStartX {
            pts.append(SIMD3<Float>(x, sheathRadius * cos(a), sheathRadius * sin(a)))
            x += 0.2
        }
        let end: Float = m.antherS - antherSemiAxes.x * 0.9
        var s: Float = 0
        while s < end {
            let f: SpinePoint = spineAt(s)
            pts.append(offsetPoint(f, radius: filamentOffset(s), angleDegrees: m.angleDegrees))
            s += f.p.x < coilStartX ? 0.2 : 0.12
        }
        pts.append(offsetPoint(spineAt(end), radius: filamentOffset(end), angleDegrees: m.angleDegrees))
        return Chain(points: pts, radii: Array(repeating: filamentRadius, count: pts.count),
                     material: materialStamen)
    }
}

// MARK: - ovules

func ovuleCentres() -> [SIMD3<Float>] {
    (0..<ovuleCount).map { SIMD3<Float>(ovuleFirstX + Float($0) * ovuleSpacing, ovuleCentreY, 0) }
}

/// Where the pollen tube enters the ovule: MODEL, on its upper end facing the
/// style.
func micropyle(_ i: Int) -> SIMD3<Float> {
    let c: SIMD3<Float> = ovuleCentres()[i]
    let a: Float = 35 * Float.pi / 180
    return c + SIMD3<Float>(ovuleSemiAxes.x * cos(a), ovuleSemiAxes.y * sin(a), 0) * 0.97
}

/// Each ovule hangs from the locule's roof on a short stalk.
func funiculi() -> [Chain] {
    ovuleCentres().map { c in
        Chain(points: [c + SIMD3<Float>(-0.05, ovuleSemiAxes.y * 0.8, 0), SIMD3<Float>(c.x - 0.12, 0.36, 0)],
              radii: [0.045, 0.04], material: materialOvule)
    }
}

// MARK: - pollen

struct PollenGrain {
    var centre: SIMD3<Float>
    var pole: SIMD3<Float>
}

/// A small deterministic random source, so every render places the same grains.
struct Lcg {
    var state: UInt64
    mutating func next() -> Float {
        state = state &* 6364136223846793005 &+ 1442695040888963407
        return Float((state >> 33) & 0xFFFFFF) / Float(0xFFFFFF)
    }
    mutating func unit() -> SIMD3<Float> {
        while true {
            let v = SIMD3<Float>(next() * 2 - 1, next() * 2 - 1, next() * 2 - 1)
            let l: Float = simd_length(v)
            if l > 0.1 && l <= 1 { return v / l }
        }
    }
}

let grainEquatorialRadius: Float = pollenEquatorialDiameter / 2
let grainPolarRadius: Float = pollenPolarDiameter / 2

/// Where the camera looks from: in front of the flower's right side, above
/// and a little forward, so the spiral is seen nearly face-on.
let viewDirection = SIMD3<Float>(0.30, 0.52, 1.0)

/// The grain that germinates: on the stigma, on the side facing the camera.
let germinatingPole: SIMD3<Float> = simd_normalize(simd_normalize(viewDirection) + stigmaFrame.t * 0.55)
let germinatingGrain = PollenGrain(centre: stigmaCentre + germinatingPole * (stigmaRadius + grainPolarRadius * 0.9),
                                   pole: germinatingPole)
/// The pore its tube leaves by: on the equator, facing back up the style.
let germinationPore: SIMD3<Float> = {
    let back: SIMD3<Float> = -stigmaFrame.t
    let e: SIMD3<Float> = simd_normalize(back - germinatingPole * simd_dot(back, germinatingPole))
    return germinatingGrain.centre + e * grainEquatorialRadius
}()

func pollenGrains(_ ant: [Anther]) -> [PollenGrain] {
    var rng = Lcg(state: 25)
    var out: [PollenGrain] = [germinatingGrain]
    func free(_ c: SIMD3<Float>) -> Bool {
        out.allSatisfy { simd_distance($0.centre, c) > pollenEquatorialDiameter * 1.02 }
    }
    // On the stigma.
    var tries: Int = 0
    while out.count < 10 && tries < 4000 {
        tries += 1
        let d: SIMD3<Float> = rng.unit()
        if simd_dot(d, -stigmaFrame.t) > 0.35 { continue }       // not where the style joins
        let c: SIMD3<Float> = stigmaCentre + d * (stigmaRadius + grainPolarRadius * 0.9)
        // Keep clear of the germinating grain and the path its tube takes.
        if simd_distance(c, germinatingGrain.centre) < 0.085 { continue }
        if simd_distance(c, tubeEntry) < 0.07 { continue }
        if free(c) { out.append(PollenGrain(centre: c, pole: d)) }
    }
    // Held in the brush.
    tries = 0
    while out.count < 58 && tries < 8000 {
        tries += 1
        let s: Float = stigmaS - brushGapBelowStigma - rng.next() * (brushLength - 0.1)
        let a: Float = (rng.next() * 2 - 1) * 68
        let f: SpinePoint = spineAt(s)
        let r: Float = f.styleRadius + 0.025 + rng.next() * (hairLength - 0.01)
        let c: SIMD3<Float> = offsetPoint(f, radius: r, angleDegrees: a)
        if free(c) { out.append(PollenGrain(centre: c, pole: rng.unit())) }
    }
    // Spilling from the dehisced anthers.
    for an in ant {
        for _ in 0..<2 {
            let u: Float = rng.next() * 2 - 1
            let c: SIMD3<Float> = an.centre + an.axis * (u * antherSemiAxes.x * 0.7)
                + an.side * ((rng.next() < 0.5 ? -1 : 1) * (antherSemiAxes.y + grainEquatorialRadius * 0.8))
            if free(c) { out.append(PollenGrain(centre: c, pole: rng.unit())) }
        }
    }
    return out
}

// MARK: - the brush

struct Hair {
    var base: SIMD3<Float>
    var tip: SIMD3<Float>
}

let hairRows: Int = Int((brushLength - brushGapBelowStigma) / hairRowSpacing)
let brushStartS: Float = stigmaS - brushLength

/// Row k, column j (j = 0 is the middle, on the upper side). Odd rows are
/// offset by half a column, so the hairs stand staggered.
func hair(row k: Int, column j: Int) -> Hair {
    let s: Float = brushStartS + Float(k) * hairRowSpacing
    let f: SpinePoint = spineAt(s)
    let stagger: Float = k % 2 == 1 ? 0.5 : 0
    let angle: Float = (Float(j - hairColumns / 2) + stagger - 0.25) * hairColumnStepDegrees
    let a: Float = angle * Float.pi / 180
    let dir: SIMD3<Float> = f.n * cos(a) + f.b * sin(a)
    let base: SIMD3<Float> = f.p + dir * (f.styleRadius - 0.01)
    let lean: Float = hairLeanDegrees * Float.pi / 180
    // Longest mid-brush, shorter at its two ends. MODEL.
    let u: Float = Float(k) / Float(max(hairRows - 1, 1))
    let length: Float = hairLength * (0.55 + 0.45 * sin(Float.pi * min(u * 1.3, 1)))
    let tip: SIMD3<Float> = base + (dir * cos(lean) + f.t * sin(lean)) * length
    return Hair(base: base, tip: tip)
}

func makeHairs() -> [Hair] {
    var out: [Hair] = []
    for k in 0..<hairRows { for j in 0..<hairColumns { out.append(hair(row: k, column: j)) } }
    return out
}

// MARK: - the pollen tube

/// Where the tube goes into the stigma, just under its surface.
let tubeEntry: SIMD3<Float> = {
    let pole: SIMD3<Float> = germinatingPole
    let up: SIMD3<Float> = simd_normalize(-stigmaFrame.t - pole * simd_dot(-stigmaFrame.t, pole))
    let c: Float = cos(1.35)
    let s: Float = sin(1.35)
    let dir: SIMD3<Float> = pole * c + up * s
    return stigmaCentre + dir * (stigmaRadius - 0.015)
}()

/// From the germinating grain's pore, over the stigma, into it, down the
/// ventral side of the stylar canal (Lord & Kohorn), along the upper side of
/// the ovary's cavity, and into one ovule.
func pollenTube(_ mutant: Mutant) -> Chain {
    var ctrl: [SIMD3<Float>] = []
    let f: SpinePoint = stigmaFrame
    let pole: SIMD3<Float> = germinatingPole
    ctrl.append(germinationPore)
    // Out of the pore, down onto the stigma, over its wet surface towards the
    // style for about a tenth of a millimetre, then in.
    let up: SIMD3<Float> = simd_normalize(-f.t - pole * simd_dot(-f.t, pole))
    let out1: SIMD3<Float> = germinationPore + up * 0.008 - pole * 0.014
    ctrl.append(out1)
    for i in 0...10 {
        let a: Float = 0.3 + 0.9 * Float(i) / 10
        let dir: SIMD3<Float> = pole * cos(a) + up * sin(a)
        ctrl.append(stigmaCentre + dir * (stigmaRadius + pollenTubeRadius * 0.5))
    }
    ctrl.append(tubeEntry)
    ctrl.append(stigmaCentre + simd_normalize(pole * 0.2 - f.t * 0.98) * 0.06)
    let ventral: Float = 0.03
    if mutant == .tubeLeaves {
        // The short cut: straight from the stigma to the ovary's tip, across
        // the open coil.
        let a: SIMD3<Float> = spineAt(stigmaS - 0.12).p
        let b = SIMD3<Float>(ovaryEndX, ventral, 0)
        for i in 0...40 { ctrl.append(a + (b - a) * (Float(i) / 40)) }
    } else {
        var s: Float = stigmaS - 0.12
        while s > styleStartS + 0.02 {
            let g: SpinePoint = spineAt(s)
            ctrl.append(g.p + g.n * ventral)
            s -= 0.1
        }
        ctrl.append(SIMD3<Float>(ovaryEndX, ventral, 0))
    }
    ctrl.append(SIMD3<Float>(7.0, 0.05, 0))
    ctrl.append(SIMD3<Float>(6.8, 0.12, 0))
    ctrl.append(SIMD3<Float>(6.6, 0.22, 0))
    ctrl.append(SIMD3<Float>(6.42, 0.27, 0))
    ctrl.append(micropyle(fertilisedOvule))
    // Resample so no segment is longer than 0.05 mm.
    var pts: [SIMD3<Float>] = [ctrl[0]]
    for i in 1..<ctrl.count {
        let a: SIMD3<Float> = ctrl[i - 1]
        let b: SIMD3<Float> = ctrl[i]
        let n: Int = max(Int(ceil(simd_distance(a, b) / 0.05)), 1)
        for k in 1...n { pts.append(a + (b - a) * (Float(k) / Float(n))) }
    }
    return Chain(points: pts, radii: Array(repeating: pollenTubeRadius, count: pts.count), material: materialTube)
}

// MARK: - everything, for a given mutant

struct FlowerModel {
    var mutant: Mutant
    var stamens: [Stamen]
    var anthers: [Anther]
    var grains: [PollenGrain]
    var hairs: [Hair]
    var tube: Chain
    var chains: [Chain]        // filaments, funiculi, tube — in that order
    var ovules: [SIMD3<Float>]

    init(_ mutant: Mutant) {
        self.mutant = mutant
        stamens = makeStamens(mutant)
        anthers = makeAnthers(stamens)
        grains = pollenGrains(anthers)
        hairs = makeHairs()
        tube = pollenTube(mutant)
        chains = filamentChains(stamens) + funiculi() + [tube]
        ovules = ovuleCentres()
    }
}

// MARK: - the bud

/// The bud's petals are shells round two blended ellipsoids: one round the
/// coil, one round the ovary and the keel's broad base. The banner is the outer
/// shell over the top and sides; the wings, just inside it, cover the sides and
/// underneath, and their margins overlap the banner's and each other's — a
/// closed bud. Sizes MODEL, chosen to hold the coil and to match the standard's
/// length (tested).
let envelopeFrontCentre = SIMD3<Float>(9.0, 2.45, -0.05)
let envelopeFrontRadii = SIMD3<Float>(3.25, 3.3, 1.35)
let envelopeBackCentre = SIMD3<Float>(4.6, 0.2, 0)
let envelopeBackRadii = SIMD3<Float>(4.3, 1.75, 1.5)
/// The line round which the banner and wings divide the bud between them.
let petalSplitY: Float = 1.2
/// The banner covers all but the bottom 110°; each wing covers 40°–185° of one
/// side, so the wings tuck under the banner and cross each other underneath.
let bannerBottomGapHalfDegrees: Float = 55
let wingFromDegrees: Float = 40
let wingToDegrees: Float = 185
/// The banner lies outside the wings by this much.
let bannerOffset: Float = 0.07

/// For the open mutant: the banner swung back and the wings spread, as at
/// anthesis. Radians.
func bannerOpenAngle(_ m: Mutant) -> Float { m == .open ? 1.7 : 0 }
func wingOpenAngle(_ m: Mutant) -> Float { m == .open ? 1.1 : 0 }
