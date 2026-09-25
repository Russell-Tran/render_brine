// One opossum shrimp, built from numbers rather than from a structure file.
//
// Mysis diluviana Audzijonyte & Väinölä 2005 — the Great Lakes mysid, split out
// of the Mysis relicta complex in 2005, and the only mysid native to the
// Laurentian Great Lakes (the other one there, Hemimysis anomala, arrived in
// 2006 in ballast water). A glacial relict that has been in the cold deep water
// since the ice left.
//
// What is counted or measured, with where:
//
//   8 pairs of thoracopods; the first two are maxillipeds, the other six are
//   biramous pereopods whose exopods beat        (Mysida, general morphology)
//   a statocyst in the ENDOPOD OF EACH UROPOD — exactly two. This is the
//   diagnostic character of the order: a bead of dense mineral in the tail
//   fan, a balance organ you can see through the cuticle. Fluorite (CaF₂) in
//   86% of mysid species, vaterite in 9%, and vaterite peaks in fresh water
//   (Ariani, Wittmann & Franco, Biological Bulletin 185:393, 1993)
//   a marsupium — the brood pouch slung under the thorax, roofed by the
//   sternum and floored by the oostegites, which is what "opossum shrimp"
//   means. Most Mysidae carry two or three pairs of oostegites
//   adults up to 25 mm                          (Mysis diluviana; the order
//                                                runs 5–25 mm)
//   diel vertical migration every day, tracking a light threshold: mysids
//   prefer 10⁻⁶ to 10⁻⁵ lux and avoid 3.4×10⁻⁷ to 2.1×10⁻⁶ mylux
//   (Gal, Loew, Rudstam & Mohammadian, Can. J. Fish. Aquat. Sci. 56:311, 1999;
//   Boscarino, Rudstam, Loew & Mills, CJFAS 66:101, 2009)
//   one visual pigment, an A1 rhodopsin peaking at 520 nm
//
// What is a model, and labelled as one everywhere it appears:
//
//   the exopod beat frequency. I could find no published beat frequency for a
//   mysid. It is scaled here from the only two points I could anchor on: a
//   10 mm Artemia at 5 Hz (step 13's own model) and a 40 mm Antarctic krill
//   HOVERING at 3 Hz, measured (Murphy, Webster & Yen, Marine Biology
//   158:2541, 2011). A power law through those two gives 3.6 Hz at 25 mm. Note
//   what that means: the trend says a mysid hovers SLOWER than a brine shrimp,
//   not faster.
//   every σ_s amplitude, and the grazing weight (1 − |n·v|)^k, which stands in
//   for a real angular scattering distribution
//   astaxanthin as the identity of the orange in the hepatopancreas
//   the snow: there is no fluid solver, only tracers on a prescribed conveyor
//
// Ford, Bailey & Santhanakrishnan, Integrative and Comparative Biology
// 62(3):791 (2022) — the same paper step 13 used — for the metachrony being
// ADLOCOMOTORY, and for mysids sitting on that trend alongside euphausiids.

import CoreGraphics
import Foundation
import simd

// MARK: - Constants, each with where it came from

struct MConstant {
    var name: String
    var value: Double
    var unit: String
    var evidence: Evidence
    var source: String
}

// -- geometry of the picture ------------------------------------------------

/// 19.5 µm to the pixel: a 25 mm animal laid across the 4:3 diagonal spans
/// 1,282 px of a 1,280 px frame along its own axis, which is why the tail fan
/// leaves through the corner.
let micronsPerPixelM: Float = 19.5
let mFrameWidth = 1280
let mFrameViewHeight = 960
let mFrameCaptionHeight = 160
let cameraStandOff: Float = 40_000

// -- the animal -------------------------------------------------------------

/// Eight pairs of thoracopods. The first two pairs are maxillipeds and are
/// held forward at the mouth; the other six are the biramous pereopods whose
/// exopods beat.
let thoracopodPairs = 8
let maxillipedPairs = 2
let beatingPairs = thoracopodPairs - maxillipedPairs        // 6
let pleonites = 6
let uropodStatocysts = 2
let oostegitePairs = 3
let compoundEyeCount = 2
let setaePerExopod = 8
let setaePerUropod = 8
let setaePerTelson = 6

let mysisLengthMicrons: Float = 25_000      // rostrum tip to telson apex
let statolithRadius: Float = 150
let setaRadius: Float = 26

// -- the swimming -----------------------------------------------------------

/// 3.6 Hz. See the note at the top: derived from two anchor points, one of
/// them measured, and it points the other way from what "smaller animal,
/// faster beat" would suggest.
let beatFrequencyHz: Float = 3.6
/// Fresh water at 4 °C, which is what the deep Great Lakes are all year.
/// NOT step 13's 1.05×10⁻⁶ m²/s — that is seawater at 20 °C, and this animal
/// has never been in the sea.
let kinematicViscosity: Double = 1.57e-6
let framesPerCycle = 20
let cyclesPerLoop = 6
let mysisFrameCount = framesPerCycle * cyclesPerLoop         // 120
let frameDelayCentiseconds = 8                               // 80 ms

let displayedSecondsPerFrame: Double = Double(frameDelayCentiseconds) / 100
let loopDisplayedSeconds: Double = displayedSecondsPerFrame * Double(mysisFrameCount)
let loopRealSeconds: Double = Double(cyclesPerLoop) / Double(beatFrequencyHz)
let slowMotionFactor: Double = loopDisplayedSeconds / loopRealSeconds

/// Six consecutive lags of 2π/6 each, which together close the circle.
///
/// Step 13's brief claimed "the phase across the N limbs spans exactly 2π" and
/// that is arithmetically wrong: N limbs at 1/N lag span (N−1)/N of a turn from
/// the first to the last. It is the Nth step — rearmost round to foremost —
/// that closes it. The test here checks the real invariant.
let phaseLagFraction: Float = 1.0 / Float(beatingPairs)

let exopodSweepAmplitude: Float = radians(31)
let exopodSweepMean: Float = radians(-6)
/// The power stroke is quicker than the recovery. A periodic warp of the phase
/// does that without breaking the loop.
let strokeAsymmetry: Float = 0.20

// -- the snow ---------------------------------------------------------------

/// The conveyor wraps every 21 mm — a little more than the 18.7 mm the frame is
/// tall, so the band covers the picture with a margin at both ends — and it
/// advances exactly one wrap per loop, so the particle leaving the bottom is
/// replaced by the one arriving at the top at the matching phase. A conveyor,
/// never a rewind.
///
/// One wrap per 1.67 s is 12.6 mm/s of water going past a hovering animal,
/// which is the animal rising at 12.6 mm/s. That is the right order for a diel
/// vertical migration — tens of metres in an hour is about 14 mm/s — but the
/// number here comes from making the loop close, not from a measurement.
let snowSpanMicrons: Float = 21_000
let snowSpeedMicronsPerSecond: Float = Float(Double(snowSpanMicrons) / loopRealSeconds)
let snowParticleCount = 64
let flocParticleCount = 20

// MARK: - Reynolds numbers, derived here and nowhere else

struct MReynolds {
    var body: Double
    var exopod: Double
    var seta: Double

    init() {
        // The animal hovers, so the speed that matters for the body is the
        // speed of the water going past it — which in this shot is the snow's,
        // because that is what "the animal is rising" means.
        let u: Double = Double(snowSpeedMicronsPerSecond) * 1e-6
        let l: Double = Double(mysisLengthMicrons) * 1e-6
        body = u * l / kinematicViscosity

        let limbLength: Double = Double(exopodReach) * 1e-6
        let arc: Double = limbLength * Double(2 * exopodSweepAmplitude)
        let tipSpeed: Double = arc * 2 * Double(beatFrequencyHz)
        exopod = tipSpeed * limbLength / kinematicViscosity
        seta = tipSpeed * (Double(setaRadius) * 2 * 1e-6) / kinematicViscosity
    }
}

// MARK: - The body plan, in microns, in the animal's own frame
//
// x is anterior (+) to posterior (−), y is dorsal (+) to ventral (−), z is the
// animal's left (+). Right-handed.

let rostrumTipX: Float = 13_000
let carapaceFrontX: Float = 11_900
let carapaceBackX: Float = 2_600
let abdomenBackX: Float = -9_200
let telsonApexX: Float = -12_000
let uropodTipX: Float = -15_200
let pleonitePitch: Float = (carapaceBackX - abdomenBackX) / Float(pleonites)
let exopodReach: Float = 2_400

let thoracopodFrontX: Float = 10_600
let thoracopodBackX: Float = 3_300
let thoracopodPitch: Float = (thoracopodFrontX - thoracopodBackX) / Float(thoracopodPairs - 1)

func thoracopodX(_ i: Int) -> Float {
    let n: Float = Float(i)
    return thoracopodFrontX - n * thoracopodPitch
}

func pleoniteX(_ j: Int) -> Float {
    let n: Float = Float(j) + 0.5
    return carapaceBackX - n * pleonitePitch
}

func pleoniteSemi(_ j: Int) -> SIMD3<Float> {
    let n: Float = Float(j)
    return SIMD3(1560, 760 - 74 * n, 640 - 58 * n)
}

// MARK: - σ_s, the whole colour model
//
// An amplitude, a wavelength exponent and a pigment, per tissue. Nothing here
// is three hand-picked RGB numbers: the blue-white is the exponent, and the
// only places a colour is written down at all are the pigments, which really
// do have spectra.

let scatterers: [MTissue: Scatterer] = [
    // Chitin lamellae at every index step: small against 550 nm, so p = 4.
    .cuticle: Scatterer(amplitude: 0.52, exponent: 4, tint: SIMD3(1, 1, 1)),
    .innerWall: Scatterer(amplitude: 0.30, exponent: 4, tint: SIMD3(1, 1, 1)),
    .pereopod: Scatterer(amplitude: 0.40, exponent: 4, tint: SIMD3(1, 1, 1)),
    .exopod: Scatterer(amplitude: 0.46, exponent: 4, tint: SIMD3(1, 1, 1)),
    .seta: Scatterer(amplitude: 0.70, exponent: 4, tint: SIMD3(1, 1, 1)),
    .antenna: Scatterer(amplitude: 0.44, exponent: 4, tint: SIMD3(1, 1, 1)),
    .marsupium: Scatterer(amplitude: 0.34, exponent: 4, tint: SIMD3(1, 1, 1)),
    // Screening pigment: it absorbs what it would otherwise throw sideways,
    // which is the whole job of an eye. Dark, and warm where it is not.
    .eye: Scatterer(amplitude: 0.11, exponent: 2, tint: SIMD3(1.00, 0.56, 0.40)),
    // The reflective layer behind the rhabdoms. A mirror is nearly spectrally
    // flat, so the exponent goes down and the ring reads white.
    .tapetum: Scatterer(amplitude: 1.05, exponent: 0.6, tint: SIMD3(1, 1, 1)),
    // Astaxanthin: the carotenoid crustaceans carry, absorbing hard through
    // the blue and green and passing the red. MODEL — the pigment is named
    // from what crustaceans generally carry, not measured in this animal.
    .hepatopancreas: Scatterer(amplitude: 0.62, exponent: 2, tint: SIMD3(1.00, 0.40, 0.11)),
    .gut: Scatterer(amplitude: 0.26, exponent: 2, tint: SIMD3(0.92, 0.84, 0.58)),
    .embryo: Scatterer(amplitude: 0.40, exponent: 2, tint: SIMD3(1.00, 0.78, 0.58)),
    // Fluorite, or vaterite: either way a dense crystal bead far larger than
    // the wavelength, so it scatters geometrically — flat spectrum, and hard.
    // Two of these are the brightest things in the frame.
    .statolith: Scatterer(amplitude: 2.60, exponent: 0.3, tint: SIMD3(1, 1, 1)),
    .snow: Scatterer(amplitude: 0.55, exponent: 2, tint: SIMD3(1.00, 0.94, 0.84)),
    .floc: Scatterer(amplitude: 0.80, exponent: 2, tint: SIMD3(1.00, 0.97, 0.92)),
]

/// The σ table the kernel reads, in tissue order.
func sigmaTableM() -> [SIMD3<Float>] {
    var out = [SIMD3<Float>](repeating: .zero, count: mTissueCount)
    for t in MTissue.allCases {
        out[t.rawValue] = scatterers[t]?.sigma ?? .zero
    }
    return out
}

// MARK: - Time

/// Where in the beat cycle frame `f` is, in cycles, reduced BEFORE it becomes
/// an angle — so frame 120 reproduces frame 0 bit for bit rather than to
/// within a float's worth of 12π.
func cyclePhaseM(frame f: Int, mutations: MMutations = []) -> Float {
    // The mutation reduces modulo the wrong number of frames, which leaves the
    // beat looking perfectly fine and quietly stops the loop closing.
    let period: Int = mutations.contains(.badLoopPhase) ? framesPerCycle + 1 : framesPerCycle
    let m: Int = ((f % period) + period) % period
    return Float(m) / Float(framesPerCycle)
}

/// Where in the whole loop frame `f` is, in [0, 1).
func loopPhaseM(frame f: Int, mutations: MMutations = []) -> Float {
    let period: Int = mutations.contains(.badLoopPhase) ? mysisFrameCount + 7 : mysisFrameCount
    let m: Int = ((f % period) + period) % period
    return Float(m) / Float(mysisFrameCount)
}

/// The phase of beating pair `b` (0 is the foremost of the six, 5 the
/// hindmost), in radians. The hindmost leads and each pair in front of it lags
/// by 2π/6, so the wave runs posterior → anterior: adlocomotory metachrony,
/// the same direction step 13 found for Artemia.
func exopodPhase(_ b: Int, frame f: Int, mutations: MMutations = []) -> Float {
    let behind: Float = mutations.contains(.reversedWave)
        ? Float(b) : Float(beatingPairs - 1 - b)
    let cycles: Float = cyclePhaseM(frame: f, mutations: mutations) - behind * phaseLagFraction
    return 2 * Float.pi * cycles
}

func exopodAngle(_ b: Int, frame f: Int, mutations: MMutations = []) -> Float {
    let phase: Float = exopodPhase(b, frame: f, mutations: mutations)
    let warped: Float = phase + strokeAsymmetry * sin(phase)
    return exopodSweepMean + exopodSweepAmplitude * cos(warped)
}

/// How far the exopod paddle is feathered open. It spreads on the power stroke
/// and folds on the recovery, which is why a limb that pushes hard one way
/// slips back the other.
func exopodSpread(_ b: Int, frame f: Int, mutations: MMutations = []) -> Float {
    let phase: Float = exopodPhase(b, frame: f, mutations: mutations)
    return 0.66 + 0.34 * sin(phase)
}

// MARK: - The snow, as a conveyor
//
// No fluid solver. Particles fall at one prescribed speed in world −y and wrap
// after exactly one span, and the wrap lands at the phase the particle above it
// has just left. So the field at frame 120 is the field at frame 0, particle
// for particle, and nothing ever runs backwards.

/// The prescribed velocity field, in µm/s. It has no arguments it depends on
/// because it is constant — and the test that no particle ever moves up is a
/// test on this, not on a difference of positions.
func snowVelocityY(particle: Int, frame f: Int) -> Float {
    -snowSpeedMicronsPerSecond
}

private let snowSeeds: [SIMD3<Float>] = {
    // A fixed lattice with a golden-angle twist: deterministic, reproducible,
    // and with no two particles in the same screen column.
    var out: [SIMD3<Float>] = []
    let total: Int = snowParticleCount + flocParticleCount
    for k in 0..<total {
        let n: Float = Float(k)
        let a: Float = n * 2.39996323                       // the golden angle
        let x: Float = -14_000 + 28_500 * ((a / (2 * Float.pi)).truncatingRemainder(dividingBy: 1))
        let y: Float = snowSpanMicrons * n / Float(total)
        let z: Float = -7_000 + 14_000 * (((a * 1.618) / (2 * Float.pi))
            .truncatingRemainder(dividingBy: 1))
        out.append(SIMD3(x, y, z))
    }
    return out
}()

/// Where particle `k` is at frame `f`, in WORLD coordinates — the snow does not
/// turn with the animal, it falls.
func snowPosition(_ k: Int, frame f: Int, mutations: MMutations = []) -> SIMD3<Float> {
    let seed: SIMD3<Float> = snowSeeds[k]
    let p: Float = loopPhaseM(frame: f, mutations: mutations)
    let raw: Float = seed.y - p * snowSpanMicrons + 2 * snowSpanMicrons
    let wrapped: Float = raw.truncatingRemainder(dividingBy: snowSpanMicrons)
    let y: Float = snowFieldBottom + wrapped
    return SIMD3(seed.x, y, seed.z)
}

/// The bottom of the wrap band. The frame spans y from −8.8 to +10.0 mm, so the
/// band from −11 to +10 mm holds it.
let snowFieldBottom: Float = -11_000

func snowRadius(_ k: Int) -> Float {
    if k < snowParticleCount {
        // 25 to 70 µm: one to four pixels across, which is what a fleck of
        // detritus looks like through a real objective — a speck, not a shape.
        let n: Float = Float(k % 7)
        return 25 + 7.5 * n
    }
    // The aggregates are big enough to resolve, so they come out as rings:
    // a bright rim and a dark middle, which is why a darkfield photograph of
    // lake water looks like it is full of bubbles.
    let n: Float = Float((k - snowParticleCount) % 5)
    return 150 + 42 * n
}

func snowTissue(_ k: Int) -> MTissue { k < snowParticleCount ? .snow : .floc }

// MARK: - Posing the animal in the world
//
// The camera never moves, never rolls and never dollies, so the animal is aimed
// instead. Screen x is world x and screen y is world y throughout.
//
// A NOTE ON WHICH FLANK IS SHOWING. The brief asked for the animal running
// upper-left to lower-right with the rostrum at the top, the tail fan leaving
// through the bottom-right corner, AND for a view of its LEFT dorsolateral
// side. Those two cannot both be had. With (anterior, dorsal, left) right-
// handed, an animal whose head points to the viewer's left and whose dorsum
// points up necessarily turns its RIGHT flank to the camera — the same reason a
// fish facing left in a photograph is showing you its right side. The
// composition was the more specific instruction and the more visible one, so
// the composition is what the render keeps. This is its right dorsolateral
// side, and the caption says so.

struct MysisPosture {
    /// How far below the horizontal the body axis runs, going right.
    var slope: Float = radians(35)
    /// Rotation of the anterior out of the screen plane, toward the viewer.
    var pitch: Float = radians(6)
    /// Roll about the body axis. 180° puts the right flank square to the
    /// camera; short of that tips the dorsum toward it, which is the
    /// dorsolateral three-quarter the shot wants.
    var roll: Float = radians(158)

    /// Anterior, dorsal and left, in world coordinates. Right-handed:
    /// left = anterior × dorsal, exactly as in step 13.
    func axes() -> (anterior: SIMD3<Float>, dorsal: SIMD3<Float>, left: SIMD3<Float>) {
        let cp: Float = cos(pitch)
        let sp: Float = sin(pitch)
        let cs: Float = cos(slope)
        let ss: Float = sin(slope)
        // Head to the upper left, tail to the lower right.
        let anterior = SIMD3<Float>(-cp * cs, cp * ss, sp)
        // Two vectors perpendicular to it: one in the screen plane, one out of
        // it toward the viewer.
        let inPlane: SIMD3<Float> = simd_normalize(SIMD3(anterior.y, -anterior.x, 0))
        let outward: SIMD3<Float> = simd_normalize(simd_cross(inPlane, anterior))
        let cr: Float = cos(roll)
        let sr: Float = sin(roll)
        let left: SIMD3<Float> = simd_normalize(outward * cr + inPlane * sr)
        let dorsal: SIMD3<Float> = simd_normalize(simd_cross(left, anterior))
        return (anterior, dorsal, left)
    }

    func rotation() -> simd_float3x3 {
        let a = axes()
        return simd_float3x3(columns: (a.anterior, a.dorsal, a.left))
    }
}

/// Everything the renderer and the tests need about one frame.
struct MysisPose {
    var prims: [GPUPrim]
    var exopodLobes: [[Int]]          // 12 beating exopods, indices into prims
    var exopodHinge: [SIMD3<Float>]
    var uropodEndopod: [Int]          // 2, one per side
    var statolith: [Int]              // 2, one per uropod endopod
    var marsupium: [Int]
    var snowPrims: [Int]
    var counts: [String: Int]
    var rotation: simd_float3x3
    var labelAnchors: [String: SIMD3<Float>]
}

/// A limb's own frame: out along the limb, fore-aft (the direction it beats in)
/// and laterally.
private func limbFrame(angle: Float, splay: Float, side: Float)
    -> (u: SIMD3<Float>, fore: SIMD3<Float>, lat: SIMD3<Float>) {
    let s: Float = sin(angle)
    let c: Float = cos(angle)
    let cs: Float = cos(splay)
    let ss: Float = sin(splay)
    let u: SIMD3<Float> = simd_normalize(SIMD3(s, -c * cs, side * c * ss))
    let xhat = SIMD3<Float>(1, 0, 0)
    let along: Float = simd_dot(xhat, u)
    var fore: SIMD3<Float> = xhat - u * along
    if simd_length(fore) < 1e-4 { fore = SIMD3(0, 0, 1) }
    fore = simd_normalize(fore)
    var lat: SIMD3<Float> = simd_normalize(simd_cross(fore, u))
    if lat.z * side < 0 { lat = -lat }
    return (u, fore, lat)
}

/// Builds the whole animal for one frame, already in world coordinates.
func poseMysis(frame f: Int, posture: MysisPosture = MysisPosture(),
               mutations: MMutations = []) -> MysisPose {
    let R: simd_float3x3 = posture.rotation()
    func world(_ p: SIMD3<Float>) -> SIMD3<Float> { R * p }
    func worldMatrix(_ m: simd_float3x3) -> simd_float3x3 { R * m }

    var prims: [GPUPrim] = []
    var exopodLobes: [[Int]] = []
    var hinges: [SIMD3<Float>] = []
    var uropodEndopod: [Int] = []
    var statolith: [Int] = []
    var marsupium: [Int] = []
    var snowPrims: [Int] = []
    var counts: [String: Int] = [:]
    var anchors: [String: SIMD3<Float>] = [:]

    func addAxisAligned(_ centre: SIMD3<Float>, _ semi: SIMD3<Float>, _ t: MTissue) -> Int {
        let m = simd_float3x3(diagonal: semi)
        prims.append(.ell(world(centre), worldMatrix(m), t))
        return prims.count - 1
    }
    func addOriented(_ centre: SIMD3<Float>, _ m: simd_float3x3, _ t: MTissue) -> Int {
        prims.append(.ell(world(centre), worldMatrix(m), t))
        return prims.count - 1
    }
    func addCapsule(_ a: SIMD3<Float>, _ b: SIMD3<Float>, _ r: Float, _ t: MTissue) -> Int {
        prims.append(.cap(world(a), world(b), r, t))
        return prims.count - 1
    }

    // -- the hover ----------------------------------------------------------
    // The animal holds station. What it does is breathe with the beat: a small
    // surge and heave, and a slow flex down the abdomen. Model numbers, and
    // deliberately undramatic ones — the caridoid escape reaction, the thing a
    // mysid is famous for, cannot be looped and is out of scope.
    let beat: Float = 2 * Float.pi * cyclePhaseM(frame: f, mutations: mutations)
    let surge: Float = 46 * sin(beat)
    let heave: Float = 30 * cos(beat)
    func flex(_ x: Float) -> Float {
        if x >= carapaceBackX { return 0 }
        let u: Float = min((carapaceBackX - x) / 9000, 1)
        return 165 * u * u * sin(beat - Float.pi / 3)
    }
    func place(_ p: SIMD3<Float>) -> SIMD3<Float> {
        SIMD3(p.x + surge, p.y + heave + flex(p.x), p.z)
    }

    // -- rostrum and carapace -----------------------------------------------
    // The carapace is one shield over the whole thorax, fused to the front
    // somites and free at the back — so it is drawn as one union of five lobes,
    // not five segments, and in darkfield it reads as one long rim.
    _ = addAxisAligned(place(SIMD3(12_380, 720, 0)), SIMD3(720, 215, 330), .cuticle)
    let carapaceX: [Float] = [11_000, 9_100, 7_100, 5_100, 3_200]
    let carapaceSemi: [SIMD3<Float>] = [SIMD3(1_620, 950, 760), SIMD3(1_600, 1_240, 1_000),
                                        SIMD3(1_580, 1_330, 1_090), SIMD3(1_560, 1_280, 1_060),
                                        SIMD3(1_500, 1_140, 940)]
    for k in 0..<carapaceX.count {
        _ = addAxisAligned(place(SIMD3(carapaceX[k], 60, 0)), carapaceSemi[k], .cuticle)
    }
    counts["carapace lobe"] = carapaceX.count
    // The second shell. Darkfield sees surfaces, so a body with an inner wall
    // reads as a body and a body without one reads as a balloon.
    for k in 1..<carapaceX.count {
        let s: SIMD3<Float> = carapaceSemi[k] * 0.58
        _ = addAxisAligned(place(SIMD3(carapaceX[k], 40, 0)), s, .innerWall)
    }
    counts["inner wall"] = carapaceX.count - 1

    // -- the stalked eyes ----------------------------------------------------
    // Big, dark, on mobile stalks, turned three-quarters on so both of them
    // catch the light. The bright ring is the tapetum, the reflective layer
    // behind the rhabdoms — the same thing that makes a cat's eye shine.
    for side in [Float(1), -1] {
        _ = addCapsule(place(SIMD3(11_350, 300, side * 560)),
                       place(SIMD3(12_050, 700, side * 1_020)), 270, .cuticle)
        let eyeCentre: SIMD3<Float> = place(SIMD3(12_330, 820, side * 1_230))
        _ = addAxisAligned(eyeCentre, SIMD3(530, 515, 515), .tapetum)
        _ = addAxisAligned(eyeCentre, SIMD3(470, 455, 455), .eye)
        if side > 0 { anchors["eye"] = world(eyeCentre) }
    }
    counts["compound eye"] = compoundEyeCount

    // -- antennal scales, antennules, antennal flagella ----------------------
    // The antennal scale (scaphocerite) is a flat blade and one of the things
    // that says "mysid" at a glance. Edge-on to the camera it is a hard bright
    // line, which is the grazing weight doing its job.
    for side in [Float(1), -1] {
        let dir: SIMD3<Float> = simd_normalize(SIMD3(0.93, 0.16, side * 0.33))
        let lat: SIMD3<Float> = simd_normalize(simd_cross(dir, SIMD3(0, 1, 0)))
        let nrm: SIMD3<Float> = simd_normalize(simd_cross(dir, lat))
        let m = simd_float3x3(columns: (dir * 1_250, nrm * 95, lat * 430))
        _ = addOriented(place(SIMD3(12_300, 120, side * 1_180)), m, .antenna)
    }
    counts["antennal scale"] = 2

    var antennuleSegments = 0
    for side in [Float(1), -1] {
        var base: SIMD3<Float> = SIMD3(11_800, 420, side * 700)
        var dir: SIMD3<Float> = simd_normalize(SIMD3(0.90, 0.30, side * 0.32))
        for k in 0..<9 {
            let step: Float = 420 - 18 * Float(k)
            let next: SIMD3<Float> = base + dir * step
            let r: Float = 135 - 9 * Float(k)
            _ = addCapsule(place(base), place(next), max(r, 40), .antenna)
            base = next
            let bend = SIMD3<Float>(-0.06, -0.10, side * 0.02)
            dir = simd_normalize(dir + bend)
            antennuleSegments += 1
        }
    }
    counts["antennule segment"] = antennuleSegments

    // The antennal flagella. They start at 35° ventral of the body axis, which
    // is exactly horizontal on screen, and curve down as they go — so they
    // sweep out to the left of the frame the way they do in the photograph.
    var antennaSegments = 0
    for side in [Float(1), -1] {
        var base: SIMD3<Float> = SIMD3(11_600, -260, side * 880)
        var dir: SIMD3<Float> = simd_normalize(SIMD3(cos(radians(35)), -sin(radians(35)),
                                                     side * 0.10))
        for k in 0..<14 {
            let step: Float = 1_150 - 22 * Float(k)
            let next: SIMD3<Float> = base + dir * step
            let r: Float = 128 - 6.2 * Float(k)
            _ = addCapsule(place(base), place(next), max(r, 34), .antenna)
            base = next
            let bend = SIMD3<Float>(-0.028, -0.052, side * -0.004)
            dir = simd_normalize(dir + bend)
            antennaSegments += 1
        }
    }
    counts["antennal flagellum segment"] = antennaSegments

    // -- stomach, hepatopancreas, gut ---------------------------------------
    // The orange mass immediately behind the eyes: the stomach and the paired
    // lobes of the digestive gland. In a live mysid this is the one strongly
    // coloured thing in an otherwise glassy animal.
    let stomach: SIMD3<Float> = place(SIMD3(10_650, 520, 0))
    _ = addAxisAligned(stomach, SIMD3(720, 510, 490), .hepatopancreas)
    for side in [Float(1), -1] {
        _ = addAxisAligned(place(SIMD3(9_250, 90, side * 560)),
                           SIMD3(1_420, 700, 500), .hepatopancreas)
    }
    anchors["hepatopancreas"] = world(place(SIMD3(9_600, 300, 560)))
    counts["hepatopancreas lobe"] = 3

    var gutSegments = 0
    for k in 0..<12 {
        let u0: Float = Float(k) / 12
        let u1: Float = Float(k + 1) / 12
        func gutPoint(_ u: Float) -> SIMD3<Float> {
            let x: Float = 9_800 + (abdomenBackX - 9_800) * u
            let y: Float = 620 - 300 * u * u
            return place(SIMD3(x, y, 0))
        }
        _ = addCapsule(gutPoint(u0), gutPoint(u1), 215, .gut)
        gutSegments += 1
    }
    counts["gut segment"] = gutSegments

    // -- the eight pairs of thoracopods -------------------------------------
    // Pairs 0 and 1 are the maxillipeds: gnathal, held forward at the mouth,
    // and they do not beat. Pairs 2 through 7 are the biramous pereopods. Each
    // of those carries the natatory exopod that does.
    var protopods = 0
    var endopodSegments = 0
    var exopodLobeCount = 0
    var setaCount = 0
    for i in 0..<thoracopodPairs {
        let isMaxilliped: Bool = i < maxillipedPairs
        let b: Int = i - maxillipedPairs
        let x: Float = thoracopodX(i)
        let angle: Float = isMaxilliped
            ? radians(52) : exopodAngle(b, frame: f, mutations: mutations)
        let spread: Float = isMaxilliped
            ? 0.8 : exopodSpread(b, frame: f, mutations: mutations)
        let size: Float = 1.0 - 0.14 * Float(i) / Float(thoracopodPairs - 1)
        for side in [Float(1), -1] {
            let hinge: SIMD3<Float> = place(SIMD3(x, -1_020, side * 560))
            let splay: Float = radians(24 + 7 * Float(i) / Float(thoracopodPairs - 1))
            let (u, fore, lat) = limbFrame(angle: angle, splay: splay, side: side)

            // protopod
            let protoTip: SIMD3<Float> = hinge + u * (620 * size)
            _ = addCapsule(hinge, protoTip, 190 * size, .pereopod)
            protopods += 1

            // endopod — four segments, tapering, swinging only a third as far
            // as the exopod because in a mysid it is a feeding and walking leg,
            // not a paddle.
            var base: SIMD3<Float> = protoTip
            var dir: SIMD3<Float> = simd_normalize(u * 0.94 + fore * 0.34)
            for k in 0..<4 {
                let step: Float = 560 * size - 40 * Float(k)
                let next: SIMD3<Float> = base + dir * step
                _ = addCapsule(base, next, (150 - 20 * Float(k)) * size, .pereopod)
                base = next
                dir = simd_normalize(dir + fore * -0.13 + u * 0.05)
                endopodSegments += 1
            }

            if isMaxilliped {
                continue
            }

            // exopod — the paddle, and the flagellum on the end of it
            var lobes: [Int] = []
            hinges.append(world(hinge))
            let a: Float = radians(-9)
            let normal: SIMD3<Float> = fore * cos(a) + lat * sin(a)
            let broad: SIMD3<Float> = lat * cos(a) - fore * sin(a)
            let padA: Float = 900 * size * (0.74 + 0.26 * spread)
            let padC: Float = 300 * size * (0.55 + 0.45 * spread)
            let padCentre: SIMD3<Float> = hinge + u * (1_180 * size) + lat * (330 * size)
            let padM = simd_float3x3(columns: (u * padA, normal * (72 * size), broad * padC))
            let padIndex: Int = addOriented(padCentre, padM, .exopod)
            lobes.append(padIndex)
            exopodLobeCount += 1

            var fBase: SIMD3<Float> = padCentre + u * (padA * 0.92)
            var fDir: SIMD3<Float> = simd_normalize(u * 0.9 + broad * 0.3)
            for k in 0..<3 {
                let step: Float = 380 * size
                let next: SIMD3<Float> = fBase + fDir * step
                let idx: Int = addCapsule(fBase, next, (105 - 22 * Float(k)) * size, .exopod)
                lobes.append(idx)
                fBase = next
                fDir = simd_normalize(fDir + broad * 0.14)
                exopodLobeCount += 1
            }

            // the setal fringe on the paddle's margin
            for k in 0..<setaePerExopod {
                let t: Float = Float(k) / Float(setaePerExopod - 1)
                let alpha: Float = radians(-80 + 160 * t)
                let ca: Float = cos(alpha)
                let sa: Float = sin(alpha)
                let rim: SIMD3<Float> = padCentre + u * (padA * ca * 0.93)
                    + broad * (padC * sa * 0.93)
                let dir2: SIMD3<Float> = simd_normalize(u * (ca * padC) + broad * (sa * padA))
                let length: Float = 420 * size * (0.7 + 0.3 * spread)
                _ = addCapsule(rim, rim + dir2 * length, setaRadius, .seta)
                setaCount += 1
            }
            exopodLobes.append(lobes)
        }
    }
    counts["thoracopod pair"] = thoracopodPairs
    counts["maxilliped pair"] = maxillipedPairs
    counts["beating exopod"] = exopodLobes.count
    counts["protopod"] = protopods
    counts["endopod segment"] = endopodSegments
    counts["exopod lobe"] = exopodLobeCount

    // -- the marsupium ------------------------------------------------------
    // The brood pouch. It is slung under the thorax, roofed by the sternum and
    // floored by three pairs of oostegites, and the animal is named for it. No
    // other render in this project can show one, because no other animal in
    // this project has one.
    let pouchCentre: SIMD3<Float> = place(SIMD3(4_250, -1_900, 0))
    marsupium.append(addAxisAligned(pouchCentre, SIMD3(2_050, 830, 940), .marsupium))
    anchors["marsupium"] = world(pouchCentre)
    for k in 0..<oostegitePairs {
        let x: Float = 5_300 - Float(k) * 1_100
        for side in [Float(1), -1] {
            let dir: SIMD3<Float> = simd_normalize(SIMD3(0.12, -0.75, side * 0.65))
            let lat2: SIMD3<Float> = simd_normalize(simd_cross(dir, SIMD3(1, 0, 0)))
            let nrm: SIMD3<Float> = simd_normalize(simd_cross(dir, lat2))
            let m = simd_float3x3(columns: (dir * 980, nrm * 110, lat2 * 700))
            marsupium.append(addOriented(place(SIMD3(x, -1_680, side * 540)), m, .marsupium))
        }
    }
    counts["oostegite pair"] = oostegitePairs

    var embryos = 0
    for k in 0..<11 {
        let a: Float = Float(k) * 2.39996323
        let r: Float = 1_250 * sqrt(Float(k) / 11)
        let ex: Float = 4_250 + r * cos(a)
        let ey: Float = -1_900 + r * sin(a) * 0.40
        let ez: Float = Float((k % 3) - 1) * 360
        _ = addAxisAligned(place(SIMD3(ex, ey, ez)), SIMD3(360, 345, 335), .embryo)
        embryos += 1
    }
    counts["embryo"] = embryos

    // -- the abdomen --------------------------------------------------------
    // Six pleonites. They barely overlap, which is the point: a joint that
    // merges away entirely would take the crease with it, and in darkfield the
    // creases between somites are the brightest lines on the animal.
    for j in 0..<pleonites {
        _ = addAxisAligned(place(SIMD3(pleoniteX(j), 0, 0)), pleoniteSemi(j), .cuticle)
    }
    counts["pleonite"] = pleonites

    // Female Mysis have reduced, unsegmented pleopods — a sex character, and
    // the reason this abdomen does not look like a caridean shrimp's.
    var pleopods = 0
    for j in 0..<5 {
        for side in [Float(1), -1] {
            let x: Float = pleoniteX(j)
            let semi: SIMD3<Float> = pleoniteSemi(j)
            let root: SIMD3<Float> = place(SIMD3(x, -semi.y * 0.78, side * semi.z * 0.40))
            let tip: SIMD3<Float> = place(SIMD3(x - 620, -semi.y * 1.30, side * semi.z * 0.80))
            _ = addCapsule(root, tip, 105, .pereopod)
            pleopods += 1
        }
    }
    counts["pleopod"] = pleopods

    // -- the telson ---------------------------------------------------------
    // Cleft at the apex, with a fan of spines. Mysis telsons are notched, and
    // the notch is one of the characters the species key turns on.
    for side in [Float(1), -1] {
        let root: SIMD3<Float> = place(SIMD3(abdomenBackX, 0, side * 90))
        let apex: SIMD3<Float> = place(SIMD3(telsonApexX, -120, side * 230))
        let dir: SIMD3<Float> = simd_normalize(apex - root)
        let lat2: SIMD3<Float> = simd_normalize(simd_cross(dir, SIMD3(0, 1, 0)))
        let nrm: SIMD3<Float> = simd_normalize(simd_cross(dir, lat2))
        let half: SIMD3<Float> = (root + apex) * 0.5
        let m = simd_float3x3(columns: (dir * (simd_distance(root, apex) * 0.5),
                                        nrm * 340, lat2 * 210))
        _ = addOriented(half, m, .cuticle)
    }
    var telsonSetae = 0
    for k in 0..<setaePerTelson {
        let t: Float = Float(k) / Float(setaePerTelson - 1)
        let side: Float = t < 0.5 ? 1 : -1
        let base: SIMD3<Float> = place(SIMD3(telsonApexX + 120, -110, side * (120 + 180 * t)))
        let dir: SIMD3<Float> = simd_normalize(SIMD3(-1, -0.12, side * (0.15 + 0.5 * t)))
        _ = addCapsule(base, base + dir * 900, setaRadius * 1.4, .seta)
        telsonSetae += 1
    }
    counts["telson"] = 2
    counts["telson seta"] = telsonSetae

    // -- the uropods, and the two statocysts --------------------------------
    // THE diagnostic mysid character. Each uropod's inner branch, the endopod,
    // carries a statocyst near its base: a vesicle lined with sensory hairs
    // with a dense mineral statolith resting in it. Two of them, one per side,
    // and in darkfield they are the two hardest, brightest points in the frame
    // because a dense crystal a third of a millimetre across scatters
    // geometrically while everything around it is scattering as λ⁻⁴.
    var uropodSetae = 0
    for side in [Float(1), -1] {
        let exoRoot: SIMD3<Float> = place(SIMD3(-9_000, -200, side * 860))
        let exoTip: SIMD3<Float> = place(SIMD3(uropodTipX, -560, side * 2_050))
        let exoDir: SIMD3<Float> = simd_normalize(exoTip - exoRoot)
        let exoLat: SIMD3<Float> = simd_normalize(simd_cross(exoDir, SIMD3(0, 1, 0)))
        let exoNrm: SIMD3<Float> = simd_normalize(simd_cross(exoDir, exoLat))
        let exoM = simd_float3x3(columns: (exoDir * (simd_distance(exoRoot, exoTip) * 0.5),
                                           exoNrm * 130, exoLat * 380))
        _ = addOriented((exoRoot + exoTip) * 0.5, exoM, .cuticle)

        let endRoot: SIMD3<Float> = place(SIMD3(-8_950, -260, side * 470))
        let endTip: SIMD3<Float> = place(SIMD3(-14_200, -560, side * 1_180))
        let endDir: SIMD3<Float> = simd_normalize(endTip - endRoot)
        let endLat: SIMD3<Float> = simd_normalize(simd_cross(endDir, SIMD3(0, 1, 0)))
        let endNrm: SIMD3<Float> = simd_normalize(simd_cross(endDir, endLat))
        let endM = simd_float3x3(columns: (endDir * (simd_distance(endRoot, endTip) * 0.5),
                                           endNrm * 120, endLat * 310))
        let endIndex: Int = addOriented((endRoot + endTip) * 0.5, endM, .cuticle)
        uropodEndopod.append(endIndex)

        // The statocyst sits a fifth of the way out along the endopod.
        let cyst: SIMD3<Float> = endRoot + (endTip - endRoot) * 0.20
        _ = addAxisAligned(cyst, SIMD3(330, 310, 310), .innerWall)
        // Mysid statoliths grow in concentric layers — the rings are what the
        // ageing literature counts — so this is three nested shells and not one
        // ball. It is also the only way a body drawn at its interfaces can read
        // as a solid bright point rather than as a ring.
        for layer in 0..<3 {
            let r: Float = statolithRadius * (1.0 - 0.30 * Float(layer))
            let idx: Int = addAxisAligned(cyst, SIMD3(r, r, r), .statolith)
            if layer == 0 { statolith.append(idx) }
        }
        if side > 0 { anchors["statocyst"] = world(cyst) }

        for k in 0..<setaePerUropod {
            let t: Float = Float(k) / Float(setaePerUropod - 1)
            let base: SIMD3<Float> = exoRoot + (exoTip - exoRoot) * (0.25 + 0.72 * t)
            let dir: SIMD3<Float> = simd_normalize(exoLat * side + exoDir * 0.25)
            _ = addCapsule(base, base + dir * (520 + 180 * t), setaRadius, .seta)
            uropodSetae += 1
        }
    }
    counts["uropod"] = 4
    counts["uropod endopod"] = uropodEndopod.count
    counts["statocyst"] = statolith.count
    counts["uropod seta"] = uropodSetae
    counts["seta"] = setaCount + telsonSetae + uropodSetae

    // -- the snow -----------------------------------------------------------
    // Not attached to the animal and not rotated with it: it falls in world
    // coordinates, straight down, at one speed, forever.
    for k in 0..<snowSeeds.count {
        let centre: SIMD3<Float> = snowPosition(k, frame: f, mutations: mutations)
        let r: Float = snowRadius(k)
        let m = simd_float3x3(diagonal: SIMD3(r, r * 0.8, r))
        prims.append(.ell(centre, m, snowTissue(k)))
        snowPrims.append(prims.count - 1)
    }
    counts["snow particle"] = snowParticleCount
    counts["floc"] = flocParticleCount
    counts["total"] = prims.count

    return MysisPose(prims: prims, exopodLobes: exopodLobes, exopodHinge: hinges,
                     uropodEndopod: uropodEndopod, statolith: statolith,
                     marsupium: marsupium, snowPrims: snowPrims, counts: counts,
                     rotation: R, labelAnchors: anchors)
}

/// The animal without the conveyor: everything but the drifting particles.
/// The swept bound over THIS is the bound the optimisation was hoping for; the
/// bound over the whole scene is dominated by the snow, because a particle that
/// falls the height of the frame marks its whole column for all 120 frames.
func animalPrims(_ p: MysisPose) -> [GPUPrim] {
    let first: Int = p.snowPrims.first ?? p.prims.count
    return Array(p.prims[0..<first])
}

// MARK: - The camera, built once and never again
//
// This is the whole premise of the shot, so it is one function with no
// dependence on the frame at all — not a function that happens to return the
// same thing. A test calls it at frame 0 and frame 119 and compares the two
// bases bit for bit.

func mysisCamera(frame: Int = 0, width: Int = mFrameWidth,
                 viewHeight: Int = mFrameViewHeight) -> Camera {
    let centre = SIMD3<Float>(400, 600, 0)
    return Camera(centre: centre, micronsPerPixel: micronsPerPixelM,
                  width: width, height: viewHeight, standOff: cameraStandOff)
}

// MARK: - The constants table

private let measuredConstants: [MConstant] = [
    MConstant(name: "pairs of thoracopods", value: Double(thoracopodPairs), unit: "",
              evidence: .measured,
              source: "Mysida: eight pairs of thoracic appendages; Meland & Willassen, "
                    + "Mol. Phylogenet. Evol. 44:1083 (2007), and general Mysida morphology"),
    MConstant(name: "pairs of maxillipeds", value: Double(maxillipedPairs), unit: "",
              evidence: .measured,
              source: "Mysida: the first two thoracopods are maxillipeds, leaving six "
                    + "biramous pereopods with natatory exopods"),
    MConstant(name: "beating exopod pairs", value: Double(beatingPairs), unit: "",
              evidence: .derived,
              source: "DERIVED: 8 thoracopod pairs less the 2 maxilliped pairs"),
    MConstant(name: "statocysts", value: Double(uropodStatocysts), unit: "",
              evidence: .measured,
              source: "the diagnostic character of Mysida: one statocyst in the endopod "
                    + "of each uropod; Ariani, Wittmann & Franco, Biol. Bull. 185:393 (1993)"),
    MConstant(name: "pleonites", value: Double(pleonites), unit: "",
              evidence: .measured, source: "six abdominal somites, as in all Mysida"),
    MConstant(name: "oostegite pairs forming the marsupium", value: Double(oostegitePairs),
              unit: "", evidence: .measured,
              source: "most Mysidae carry two or three pairs of oostegites; the marsupium "
                    + "is what 'opossum shrimp' names"),
    MConstant(name: "adult body length", value: Double(mysisLengthMicrons) / 1000, unit: "mm",
              evidence: .measured,
              source: "Mysis diluviana adults to 25 mm; NOAA GLANSIS species account (2020)"),
    MConstant(name: "native Great Lakes mysid species", value: 1, unit: "",
              evidence: .measured,
              source: "Mysis diluviana is the only mysid native to the Laurentian Great "
                    + "Lakes; Audzijonyte & Vainola, Hydrobiologia 544:89 (2005)"),
    MConstant(name: "light avoidance threshold, lower", value: 1e-6, unit: "lux",
              evidence: .measured,
              source: "Boscarino, Rudstam, Loew & Mills, Can. J. Fish. Aquat. Sci. 66:101 "
                    + "(2009) — mysids prefer 1e-6 to 1e-5 lux"),
    MConstant(name: "light avoidance threshold, upper", value: 1e-5, unit: "lux",
              evidence: .measured,
              source: "Gal, Loew, Rudstam & Mohammadian, CJFAS 56:311 (1999) — avoidance "
                    + "from 3.4e-7 to 2.1e-6 mylux"),
    MConstant(name: "visual pigment peak", value: 520, unit: "nm",
              evidence: .measured,
              source: "one A1 rhodopsin peaking at 520 nm; Jokela-Maatta et al., "
                    + "J. Comp. Physiol. A 191:1087 (2005)"),
    MConstant(name: "darkfield removes the zeroth order", value: 0, unit: "",
              evidence: .measured,
              source: "Abbe 1873, the theory of image formation: a dark stop deletes "
                    + "the undiffracted order at the back focal plane, so the image is "
                    + "built from the higher orders alone"),
    MConstant(name: "krill hovering beat frequency", value: 3.0, unit: "Hz",
              evidence: .measured,
              source: "Murphy, Webster & Yen, Marine Biology 158:2541 (2011) — "
                    + "Euphausia superba hovering, 40 mm"),
    MConstant(name: "metachrony is adlocomotory", value: 1, unit: "",
              evidence: .measured,
              source: "Ford, Bailey & Santhanakrishnan, Integr. Comp. Biol. 62(3):791 (2022)"),
    MConstant(name: "kinematic viscosity of the medium", value: kinematicViscosity,
              unit: "m^2/s", evidence: .measured,
              source: "fresh water at 4 C — this is a Great Lakes animal, not a marine one, "
                    + "so step 13's seawater value does not apply here"),
]

private let modelConstants: [MConstant] = [
    MConstant(name: "exopod beat frequency", value: Double(beatFrequencyHz), unit: "Hz",
              evidence: .model,
              source: "MODEL: no published mysid value found. A power law through 10 mm "
                    + "Artemia at 5 Hz (step 13's own model) and 40 mm krill hovering at "
                    + "3 Hz (Murphy 2011) gives 3.6 Hz at 25 mm"),
    MConstant(name: "phase lag between neighbouring exopods", value: Double(phaseLagFraction),
              unit: "cycles", evidence: .model,
              source: "MODEL: 1/6, so six consecutive lags close the circle on six limbs"),
    MConstant(name: "exopod sweep amplitude",
              value: Double(exopodSweepAmplitude * 180 / .pi), unit: "deg",
              evidence: .model, source: "MODEL: chosen so neighbouring exopods do not "
                    + "interpenetrate at any frame"),
    MConstant(name: "stroke asymmetry", value: Double(strokeAsymmetry), unit: "",
              evidence: .model,
              source: "MODEL: a periodic phase warp, power stroke quicker than recovery"),
    MConstant(name: "grazing exponent k", value: Double(grazingExponent), unit: "",
              evidence: .model,
              source: "MODEL: g = (1 - |n.v|)^k stands in for a real angular scattering "
                    + "distribution; k tuned so rims read without blowing out"),
    MConstant(name: "astaxanthin as the orange pigment", value: 0, unit: "",
              evidence: .model,
              source: "MODEL: the carotenoid crustaceans generally carry, named here from "
                    + "that generality and not measured in this animal"),
    MConstant(name: "snow drift speed", value: Double(snowSpeedMicronsPerSecond) / 1000,
              unit: "mm/s", evidence: .model,
              source: "MODEL: the flow field is prescribed, not solved — one 12 mm wrap "
                    + "per loop, which is also about the ascent rate a diel migration needs"),
    MConstant(name: "sigma_s, cuticle", value: Double(scatterers[.cuticle]?.amplitude ?? 0),
              unit: "", evidence: .model,
              source: "MODEL: amplitude tuned so the carapace rim reads without clipping"),
    MConstant(name: "sigma_s, statolith", value: Double(scatterers[.statolith]?.amplitude ?? 0),
              unit: "", evidence: .model,
              source: "MODEL: tuned so the two statoliths are the brightest points"),
    MConstant(name: "sigma_s, hepatopancreas",
              value: Double(scatterers[.hepatopancreas]?.amplitude ?? 0), unit: "",
              evidence: .model, source: "MODEL: tuned to a strong orange behind the eyes"),
    MConstant(name: "the water", value: 0, unit: "", evidence: .model,
              source: "MODEL: no fluid solver — tracers on a prescribed conveyor"),
    MConstant(name: "this is the RIGHT dorsolateral side", value: 1, unit: "",
              evidence: .model,
              source: "MODEL: the brief asked for the left flank AND for the head at the "
                    + "upper left with the dorsum up; those are mutually exclusive in a "
                    + "right-handed animal, and the composition won"),
]

private let derivedConstants: [MConstant] = [
    MConstant(name: "Rayleigh exponent", value: 4, unit: "",
              evidence: .derived,
              source: "DERIVED: sigma_s goes as lambda^-4 for scatterers small against "
                    + "the wavelength (Strutt 1871); the blue-white is that, not a choice"),
    MConstant(name: "blue over red at a cuticle interface",
              value: Double(rayleighWeights(exponent: 4).z / rayleighWeights(exponent: 4).x),
              unit: "", evidence: .derived,
              source: "DERIVED: (612/465)^4 from the two band centres"),
    MConstant(name: "Reynolds number, body", value: MReynolds().body, unit: "",
              evidence: .derived, source: "DERIVED here: U L / nu, U the drift past a "
                    + "hovering animal, nu fresh water at 4 C"),
    MConstant(name: "Reynolds number, exopod", value: MReynolds().exopod, unit: "",
              evidence: .derived, source: "DERIVED here: exopod tip speed x exopod length / nu"),
    MConstant(name: "Reynolds number, seta", value: MReynolds().seta, unit: "",
              evidence: .derived, source: "DERIVED here: exopod tip speed x seta diameter / nu"),
    MConstant(name: "pixel scale", value: Double(micronsPerPixelM), unit: "um/px",
              evidence: .derived,
              source: "DERIVED: the orthographic field divided by the frame width"),
    MConstant(name: "body length across the frame",
              value: Double(mysisLengthMicrons / micronsPerPixelM), unit: "px",
              evidence: .derived, source: "DERIVED: 25 mm at 19.5 um/px"),
    MConstant(name: "loop, real time", value: loopRealSeconds, unit: "s",
              evidence: .derived, source: "DERIVED: 6 beats at 3.6 Hz"),
    MConstant(name: "loop, displayed time", value: loopDisplayedSeconds, unit: "s",
              evidence: .derived, source: "DERIVED: 120 frames x 80 ms"),
    MConstant(name: "slow motion", value: slowMotionFactor, unit: "x",
              evidence: .derived, source: "DERIVED: displayed seconds over real seconds"),
    MConstant(name: "snow travel per loop", value: Double(snowSpanMicrons) / 1000, unit: "mm",
              evidence: .derived,
              source: "DERIVED: one wrap span, which is what makes the conveyor close"),
]

let mysisConstants: [MConstant] = measuredConstants + modelConstants + derivedConstants

// MARK: - What the caption bar says

func mysisCaption() -> Caption {
    Caption(title: "Rendered model of one mysis shrimp swimming",
            subtitle: "Mysis diluviana, adult female \u{00B7} darkfield "
                    + "\u{00B7} right dorsolateral",
            facts: "8 thoracopod pairs \u{00B7} 2 statocysts in the uropod endopods "
                 + "\u{00B7} the camera never moves",
            aside: "")
}

// No evidence bar on this frame, for step 13's reason. Step 10 earned one
// because the thing being rated CHANGED as the render played. Here it would
// have said the same three lines for all 120 frames, which makes it furniture
// rather than information.
//
// The provenance has not gone anywhere: every constant in `mysisConstants`
// carries its evidence level and its source, and a test fails the build if a
// measured one cites nothing, a derived one does not say DERIVED, or a model
// one does not say MODEL. For this step that table holds, among the rest, the
// 25 mm body, the two statocysts in the uropod endopods, the only-native-mysid
// claim, the 10⁻⁶ to 10⁻⁵ lux light threshold and Abbe's zeroth order as
// MEASURED; the λ⁻⁴ wavelength bias and all three Reynolds numbers as DERIVED;
// and the beat frequency, the grazing weight, astaxanthin and the snow's flow
// field as MODEL.
