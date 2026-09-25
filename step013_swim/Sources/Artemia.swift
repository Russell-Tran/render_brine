// One brine shrimp, built from numbers rather than from a structure file.
//
// Every step of this project since step 6 has started from a deposited
// structure: a PDB entry, a cryo-EM map, something somebody solved. There is
// no PDB entry for an animal. So step 13 does the other thing biology does —
// it builds the body from published counts and measurements, and says out
// loud, on the caption bar, which parts of the picture are counted, which are
// arithmetic, and which are a modelling choice.
//
// What is counted (Fox, *Invertebrate Anatomy OnLine*, Artemia franciscana;
// Criel & Macrae in *Artemia: Basic and Applied Biology*, ch. 1):
//
//   11 thoracic segments, each with a pair of phyllopods      → 22 limbs
//   no regional specialisation: the limbs differ only in size
//   6 abdominal segments, limbless
//   a genital region between thorax and abdomen
//   a telson bearing the caudal furca and its plumose setae
//   two stalked compound eyes and one median naupliar eye of three pigment cups
//   antennae and antennules
//   a midventral food groove between the gnathobases, mouth at its front end
//   a gut running the whole length
//   adult length 8–10 mm (FAO, *Manual on the production and use of live food
//   for aquaculture*, FAO Fisheries Technical Paper 361, 1996)
//   it swims VENTRAL SIDE UP — the ventral light reaction (Fox)
//
// What is measured elsewhere:
//
//   adult swimming speed 5.5 mm/s (Larsen, Madsen & Riisgård, *Aquatic
//   Biology* 4:47–54, 2008) — about 0.6 body lengths a second, which is why
//   this animal must not look like it is sprinting
//
// What is a model, and labelled as one, everywhere it appears:
//
//   the beat frequency, 5 Hz, extrapolated from Williams' larval measurements
//   (*Biological Bulletin* 187:164, 1994, doi:10.2307/1542239 — 9.5 Hz at
//   0.4 mm falling to 6.7 Hz at 4 mm). It is an extrapolation past the end of
//   the data, not a measurement of an adult.
//   the phase lag between neighbouring limbs, 1/11 of a cycle
//   every σ in the picture
//   the water: there is no fluid solver here at all, only tracers on a
//   prescribed conveyor
//
// Ford, Bailey & Santhanakrishnan, *Integrative and Comparative Biology*
// 62(3):791 (2022), for the metachrony: the wave is ADLOCOMOTORY, travelling
// in the same direction the animal swims. Artemia swims head first and the
// wave runs tail to head, so the rearmost limb starts its power stroke and
// each limb in front of it follows.

import CoreGraphics
import Foundation
import simd

// MARK: - Constants, each with where it came from

struct Constant {
    var name: String
    var value: Double
    var unit: String
    var evidence: Evidence
    var source: String
}

// -- geometry of the picture ------------------------------------------------

/// 9.1 µm to the pixel. Chosen so a 10 mm animal spans about 1,100 px of a
/// 1,280 px frame — and, because the camera is orthographic, it is ONE number
/// over the whole field, which is what the sub-pixel seta clamp below needs.
let micronsPerPixel: Float = 9.1
let frameWidth = 1280
let frameViewHeight = 960
let frameCaptionHeight = 160

// -- the animal -------------------------------------------------------------

let thoracicSegments = 11
let phyllopodPairs = 11
let abdominalSegments = 6
let genitalSegments = 2
let compoundEyes = 2
let naupliarEyeCups = 3
let setaePerLimb = 24
let setaePerFurcalRamus = 9
let gutSegments = 14

/// A seta is 2–5 µm across. This uses 3 µm, the middle of that range.
let setaTrueRadius: Float = 1.5
let furcalSetaTrueRadius: Float = 2.5
/// Artemia is a non-selective filter feeder taking particles from about 1 to
/// 50 µm (FAO Fisheries Technical Paper 361). The tracers here are 24 µm
/// algal cells and aggregates — the coarse end, because the fine end is
/// smaller than a pixel.
let algaRadius: Float = 12

// -- the swimming -----------------------------------------------------------

let beatFrequencyHz: Float = 5.0
let swimmingSpeedMicronsPerSecond: Float = 5500     // 5.5 mm/s, measured
let kinematicViscosity: Double = 1.05e-6            // m²/s, seawater at 20 °C
let framesPerCycle = 30
let cyclesPerLoop = 4
let swimFrameCount = framesPerCycle * cyclesPerLoop          // 120
let frameDelayCentiseconds = 8                               // 80 ms
let slowMotionFactor: Double = 12

/// Displayed seconds per frame, and real seconds per frame. The loop shows
/// 9.6 s of screen time for 0.8 s of animal time.
let displayedSecondsPerFrame: Double = Double(frameDelayCentiseconds) / 100
let realSecondsPerFrame: Double = displayedSecondsPerFrame / slowMotionFactor
let loopRealSeconds: Double = realSecondsPerFrame * Double(swimFrameCount)
let loopDisplayedSeconds: Double = displayedSecondsPerFrame * Double(swimFrameCount)

/// Exactly one metachronal wave sits on the body: neighbouring limbs are 1/11
/// of a cycle apart, so the eleven lags close the circle.
let phaseLagFraction: Float = 1.0 / 11.0
let brokenPhaseLagFraction: Float = 1.0 / 10.0      // the mutation

let limbSweepAmplitude: Float = radians(26)
let limbSweepMean: Float = radians(-4)
/// The power stroke is quicker than the recovery. A periodic warp of the phase
/// does that without breaking the loop.
let strokeAsymmetry: Float = 0.18

// MARK: - Reynolds numbers, derived here and nowhere else

/// Re = U L / ν. Everything in this struct is arithmetic on numbers that came
/// from somewhere else, which is exactly what DERIVED means on the bar.
struct ReynoldsNumbers {
    var body: Double
    var limb: Double
    var seta: Double

    init() {
        let u: Double = Double(swimmingSpeedMicronsPerSecond) * 1e-6      // m/s
        let l: Double = Double(adultLengthMicrons) * 1e-6                 // m
        body = u * l / kinematicViscosity

        // The limb's own Reynolds number uses its tip speed, not the animal's:
        // a limb 1.21 mm long sweeping ±34° twice a cycle at 5 Hz.
        let limbLength: Double = Double(limbReach) * 1e-6
        let arc: Double = limbLength * Double(2 * limbSweepAmplitude)
        let tipSpeed: Double = arc * 2 * Double(beatFrequencyHz)
        limb = tipSpeed * limbLength / kinematicViscosity
        seta = tipSpeed * (Double(setaTrueRadius) * 2 * 1e-6) / kinematicViscosity
    }
}

// MARK: - The body plan, in microns, in the animal's own frame
//
// x is anterior (+) to posterior (−), y is dorsal (+) to ventral (−), z is the
// animal's left (+). Right-handed.

let headCentreX: Float = 3900
let thoraxFrontX: Float = 3250
let thoraxPitch: Float = 350
let genitalX: [Float] = [-480, -820]
let abdomenFrontX: Float = -1180
let abdomenPitch: Float = 375
let telsonX: Float = -3390
let limbReach: Float = 1210

/// Where thoracic segment `i` sits (i = 0 is the front one).
func thoracicSegmentX(_ i: Int) -> Float {
    let n: Float = Float(i)
    return thoraxFrontX - n * thoraxPitch
}

func abdominalSegmentX(_ j: Int) -> Float {
    let n: Float = Float(j)
    return abdomenFrontX - n * abdomenPitch
}

/// Semi-axes of thoracic segment `i`: it tapers toward the back.
func thoracicSemi(_ i: Int) -> SIMD3<Float> {
    let n: Float = Float(i)
    return SIMD3(195, 300 - 8 * n, 255 - 6 * n)
}

func abdominalSemi(_ j: Int) -> SIMD3<Float> {
    let n: Float = Float(j)
    return SIMD3(215, 175 - 8 * n, 160 - 7 * n)
}

/// Head to furca, the body proper; the antennules and caudal setae add more.
let adultLengthMicrons: Float = 9870

/// The anterior end of the midventral food groove — the mouth, just behind the
/// labrum.
let mouthPosition = SIMD3<Float>(3340, -330, 0)

/// The food groove, a midventral channel walled by the two rows of
/// gnathobases, running from behind the last thoracopod forward to the mouth.
/// `s` = 0 at the back, 1 at the mouth.
func foodGroovePoint(_ s: Float) -> SIMD3<Float> {
    let t: Float = min(max(s, 0), 1)
    let x: Float = -120 + (mouthPosition.x + 120) * t
    let dip: Float = 40 * sin(Float.pi * t)
    let y: Float = -235 - 95 * t - dip
    return SIMD3(x, y, 0)
}

let foodGrooveLength: Float = {
    var total: Float = 0
    var previous = foodGroovePoint(0)
    for k in 1...64 {
        let p = foodGroovePoint(Float(k) / 64)
        total += simd_distance(previous, p)
        previous = p
    }
    return total
}()

// MARK: - σ, the whole colour model
//
// Three independent extinction coefficients per tissue, in 1/µm. Every one of
// them is a MODEL number: chosen so the render looks like a brightfield
// micrograph of an Artemia, not measured from one. The comment on each line
// says what transmission it produces through a typical chord, which is the
// form in which it was actually chosen.
//
// Where one tissue sits inside another — the gut inside the trunk, an eye
// inside the head — its σ is the EXCESS over what surrounds it, because the
// per-tissue unions add and the surrounding tissue's own absorption is already
// counted along that same stretch of ray.

let sigmaBody = SIMD3<Float>(1.85e-3, 2.36e-3, 1.95e-3)     // 500 µm → T ≈ 0.40/0.31/0.38, blue-grey
let sigmaLimb = SIMD3<Float>(9.7e-4, 1.38e-3, 1.28e-3)      // 250 µm → T ≈ 0.78/0.71/0.73
let sigmaSetaTrue = SIMD3<Float>(1.35e-2, 1.90e-2, 1.80e-2) // 3 µm across → τ ≈ 0.04/0.06/0.05
let sigmaCompoundEye = SIMD3<Float>(9.2e-3, 1.30e-2, 1.40e-2)  // 230 µm → T ≈ 0.12/0.05/0.04
let sigmaNaupliarEye = SIMD3<Float>(1.25e-2, 2.04e-2, 2.26e-2) // 84 µm → T ≈ 0.35/0.18/0.15
let sigmaGut = SIMD3<Float>(8.0e-4, 1.6e-3, 2.9e-3)         // 156 µm → T ≈ 0.88/0.78/0.64, olive
let sigmaAppendage = SIMD3<Float>(1.2e-3, 1.7e-3, 1.6e-3)   // 88 µm → T ≈ 0.90/0.86/0.87
let sigmaAlga = SIMD3<Float>(5.0e-2, 2.2e-2, 5.5e-2)        // 24 µm → T ≈ 0.30/0.59/0.27, green
let sigmaEgg = SIMD3<Float>(4.0e-3, 5.5e-3, 6.6e-3)         // 230 µm → T ≈ 0.40/0.28/0.22

// MARK: - The sub-pixel seta clamp
//
// At 9.1 µm to the pixel a 3 µm seta is a third of a pixel wide. Drawn at true
// size it would be caught by some samples and missed by others, and the miss
// pattern would change every frame: a fringe of 528 setae flickering like
// static. So each thin absorber is drawn at a floor of half a pixel in radius,
// and its σ is scaled down by (true radius / drawn radius) so that
//
//     τ = σ · 2r
//
// through the axis is exactly what it was. The seta is three times wider than
// life and three times less absorbing, and a ray down its middle cannot tell.
//
// What this conserves is the optical depth ALONG THE AXIS. It does not
// conserve the seta's integrated contribution to the image, which goes as
// σ·πr² and therefore grows in proportion to the drawn radius — a widened seta
// darkens its pixel neighbourhood by (drawn/true) more than the real one would.
// Conserving that instead would need σ scaled by (true/drawn)². Both rules are
// implemented and tested below; the axial rule is the one this render uses,
// and `sigmaSetaTrue` was tuned for the look under it.

let subPixelRadiusFloorPixels: Float = 0.5
let drawnRadiusFloor: Float = subPixelRadiusFloorPixels * micronsPerPixel   // 4.55 µm

struct ThinAbsorber {
    var trueRadius: Float
    var drawnRadius: Float
    var sigmaScale: Float

    /// Optical depth through the axis: the quantity the clamp conserves.
    func axialTau(sigma: Float) -> Float {
        let scaled: Float = sigma * sigmaScale
        return scaled * 2 * drawnRadius
    }

    /// Absorbance integrated across the whole width, in the optically thin
    /// limit: the quantity the clamp does NOT conserve.
    func integratedAbsorbance(sigma: Float) -> Float {
        let scaled: Float = sigma * sigmaScale
        let area: Float = Float.pi * drawnRadius * drawnRadius
        return scaled * area
    }
}

/// `exponent` 1 is the axial rule this render uses; 2 is the area-conserving
/// rule, kept so the tests can show the difference rather than assert it.
func clampThin(trueRadius r: Float, floor: Float = drawnRadiusFloor,
               exponent: Int = 1, enabled: Bool = true) -> ThinAbsorber {
    if !enabled || r >= floor {
        return ThinAbsorber(trueRadius: r, drawnRadius: r, sigmaScale: 1)
    }
    let ratio: Float = r / floor
    let scale: Float = exponent == 2 ? ratio * ratio : ratio
    return ThinAbsorber(trueRadius: r, drawnRadius: floor, sigmaScale: scale)
}

/// The σ table the kernel reads, in tissue order. The seta entry is the only
/// one the clamp touches.
func sigmaTable(mutations: Mutations = []) -> [SIMD3<Float>] {
    let clamped = clampThin(trueRadius: setaTrueRadius, enabled: !mutations.contains(.noClamp))
    var out = [SIMD3<Float>](repeating: .zero, count: tissueCount)
    out[Tissue.body.rawValue] = sigmaBody
    out[Tissue.limb.rawValue] = sigmaLimb
    out[Tissue.seta.rawValue] = sigmaSetaTrue * clamped.sigmaScale
    out[Tissue.compoundEye.rawValue] = sigmaCompoundEye
    out[Tissue.naupliarEye.rawValue] = sigmaNaupliarEye
    out[Tissue.gut.rawValue] = sigmaGut
    out[Tissue.appendage.rawValue] = sigmaAppendage
    out[Tissue.alga.rawValue] = sigmaAlga
    out[Tissue.egg.rawValue] = sigmaEgg
    return out
}

// MARK: - Time

/// Where in the beat cycle frame `f` is, in cycles, reduced BEFORE it is
/// turned into an angle. Frame 120 therefore reproduces frame 0 bit for bit
/// rather than to within a float's worth of 8π.
func cyclePhase(frame f: Int) -> Float {
    let m: Int = ((f % framesPerCycle) + framesPerCycle) % framesPerCycle
    return Float(m) / Float(framesPerCycle)
}

/// Where in the whole loop frame `f` is, in [0, 1), reduced the same way.
func loopPhase(frame f: Int) -> Float {
    let m: Int = ((f % swimFrameCount) + swimFrameCount) % swimFrameCount
    return Float(m) / Float(swimFrameCount)
}

/// The phase of limb pair `i` (0 is the front pair, 10 the back one), in
/// radians. The back pair leads; each pair in front of it lags by 1/11 of a
/// cycle, so the wave runs posterior → anterior while the animal swims
/// anteriorly — adlocomotory metachrony.
func limbPhase(_ i: Int, frame f: Int, mutations: Mutations = []) -> Float {
    let lag: Float = mutations.contains(.badPhase) ? brokenPhaseLagFraction : phaseLagFraction
    let behind: Float = Float(thoracicSegments - 1 - i)
    let cycles: Float = cyclePhase(frame: f) - behind * lag
    return 2 * Float.pi * cycles
}

/// The stroke angle of limb pair `i`: positive is forward (anterior). Phase 0
/// is the top of the power stroke, with the limb thrown fully forward and
/// about to sweep back.
func limbAngle(_ i: Int, frame f: Int, mutations: Mutations = []) -> Float {
    let phase: Float = limbPhase(i, frame: f, mutations: mutations)
    let warped: Float = phase + strokeAsymmetry * sin(phase)
    return limbSweepMean + limbSweepAmplitude * cos(warped)
}

/// How far the exopodite is spread. It opens on the power stroke and folds on
/// the recovery — the same feathering an oar gets, and the reason a limb that
/// pushes hard one way slips back the other.
func limbSpread(_ i: Int, frame f: Int, mutations: Mutations = []) -> Float {
    let phase: Float = limbPhase(i, frame: f, mutations: mutations)
    return 0.68 + 0.32 * sin(phase)
}

// MARK: - Posing the animal in the world

/// The animal is built in its own frame and put into the world by one rotation.
/// The camera never moves and never rolls: the animal is aimed instead, which
/// keeps screen x as world x and makes the pixel scale trivially exact.
struct Posture {
    var azimuth: Float = radians(36.87)   // the 3-4-5 diagonal of a 4:3 frame
    var tilt: Float = radians(15)         // head toward the viewer
    var ventralRoll: Float = radians(32)  // ventral surface turned toward the viewer

    /// Anterior, dorsal and left, in world coordinates.
    func axes() -> (anterior: SIMD3<Float>, dorsal: SIMD3<Float>, left: SIMD3<Float>) {
        let ct: Float = cos(tilt)
        let st: Float = sin(tilt)
        let ca: Float = cos(azimuth)
        let sa: Float = sin(azimuth)
        let anterior = SIMD3<Float>(ct * ca, ct * sa, st)
        let inPlane = SIMD3<Float>(-sa, ca, 0)              // upper-left, ⟂ anterior
        let outward: SIMD3<Float> = simd_cross(anterior, inPlane)   // toward the viewer
        let cr: Float = cos(ventralRoll)
        let sr: Float = sin(ventralRoll)
        let ventral: SIMD3<Float> = simd_normalize(inPlane * cr + outward * sr)
        let dorsal: SIMD3<Float> = -ventral
        let left: SIMD3<Float> = simd_normalize(simd_cross(anterior, dorsal))
        return (anterior, dorsal, left)
    }

    func rotation() -> simd_float3x3 {
        let a = axes()
        return simd_float3x3(columns: (a.anterior, a.dorsal, a.left))
    }
}

/// Everything the renderer and the tests need about one frame.
struct Pose {
    var prims: [GPUPrim]
    var limbLobes: [[Int]]        // 22 limbs, each a list of indices into `prims`
    var limbHinge: [SIMD3<Float>] // 22 hinges, in world space
    var bodyPrims: [Int]
    var counts: [String: Int]
    var rotation: simd_float3x3
}

/// A phyllopod's own frame: along the limb, fore-aft (its thin axis, which is
/// also the direction it beats in), and laterally.
private func limbFrame(angle: Float, splay: Float, side: Float)
    -> (u: SIMD3<Float>, fore: SIMD3<Float>, lat: SIMD3<Float>) {
    let s: Float = sin(angle)
    let c: Float = cos(angle)
    let cs: Float = cos(splay)
    let ss: Float = sin(splay)
    // Straight ventral, swung fore-aft by `angle`, then splayed out sideways.
    let u = simd_normalize(SIMD3<Float>(s, -c * cs, side * c * ss))
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
func poseArtemia(frame f: Int, posture: Posture = Posture(), mutations: Mutations = []) -> Pose {
    let R: simd_float3x3 = posture.rotation()
    func world(_ p: SIMD3<Float>) -> SIMD3<Float> { R * p }
    func worldMatrix(_ m: simd_float3x3) -> simd_float3x3 { R * m }

    var prims: [GPUPrim] = []
    var limbLobes: [[Int]] = []
    var hinges: [SIMD3<Float>] = []
    var bodyPrims: [Int] = []
    var counts: [String: Int] = [:]

    /// An ellipsoid whose semi-axes are along the body frame's own axes.
    func addAxisAligned(_ centre: SIMD3<Float>, _ semi: SIMD3<Float>, _ tissue: Tissue) -> Int {
        let m = simd_float3x3(diagonal: semi)
        prims.append(.ellipsoid(centre: world(centre), m: worldMatrix(m), tissue: tissue))
        return prims.count - 1
    }
    func addOriented(_ centre: SIMD3<Float>, _ m: simd_float3x3, _ tissue: Tissue) -> Int {
        prims.append(.ellipsoid(centre: world(centre), m: worldMatrix(m), tissue: tissue))
        return prims.count - 1
    }
    func addCapsule(_ a: SIMD3<Float>, _ b: SIMD3<Float>, _ r: Float, _ tissue: Tissue) -> Int {
        prims.append(.capsule(from: world(a), to: world(b), radius: r, tissue: tissue))
        return prims.count - 1
    }

    // -- the swimming itself, as the body feels it --------------------------
    // Metachrony exists to make thrust smooth, so the surge is small: 25 µm,
    // about a four-hundredth of the body. A model number, and a deliberately
    // undramatic one.
    let beat: Float = 2 * Float.pi * cyclePhase(frame: f)
    let surge: Float = 25 * sin(beat)
    let heave: Float = 14 * cos(beat)
    /// The abdomen flexes a little with each beat. Also a model number.
    func flex(_ x: Float) -> Float {
        if x >= 0 { return 0 }
        let u: Float = min(-x / 3000, 1)
        return 62 * u * u * sin(beat - Float.pi / 3)
    }
    func place(_ p: SIMD3<Float>) -> SIMD3<Float> {
        SIMD3(p.x + surge, p.y + heave + flex(p.x), p.z)
    }

    // -- head ---------------------------------------------------------------
    bodyPrims.append(addAxisAligned(place(SIMD3(headCentreX, 0, 0)), SIMD3(520, 330, 300), .body))
    // The labrum: the big fleshy upper lip that covers the mouth. Prominent in
    // every side view of an Artemia and the front wall of the food groove.
    bodyPrims.append(addAxisAligned(place(SIMD3(3620, -330, 0)), SIMD3(250, 210, 165), .body))
    counts["head"] = 2

    // -- stalked compound eyes ----------------------------------------------
    for side in [Float(1), -1] {
        _ = addCapsule(place(SIMD3(4150, 250, side * 150)),
                       place(SIMD3(4290, 330, side * 300)), 55, .appendage)
        _ = addAxisAligned(place(SIMD3(4340, 355, side * 330)), SIMD3(120, 115, 110), .compoundEye)
    }
    counts["compound eye"] = compoundEyes

    // -- the median naupliar eye: three pigment cups ------------------------
    // The larval eye, kept through life. Three cups: one ventral median and a
    // dorsolateral pair.
    _ = addAxisAligned(place(SIMD3(4180, 60, 0)), SIMD3(44, 42, 42), .naupliarEye)
    for side in [Float(1), -1] {
        _ = addAxisAligned(place(SIMD3(4210, 175, side * 65)), SIMD3(42, 40, 40), .naupliarEye)
    }
    counts["naupliar eye cup"] = naupliarEyeCups

    // -- antennules and antennae --------------------------------------------
    // Drawn as an adult FEMALE: the antennae are reduced lobes rather than the
    // male's claspers, and there is an ovisac behind the thorax.
    for side in [Float(1), -1] {
        _ = addCapsule(place(SIMD3(4280, 60, side * 90)),
                       place(SIMD3(4700, 150, side * 140)), 24, .appendage)
        _ = addCapsule(place(SIMD3(4700, 150, side * 140)),
                       place(SIMD3(5040, 210, side * 175)), 18, .appendage)
        _ = addCapsule(place(SIMD3(4150, -140, side * 180)),
                       place(SIMD3(4420, -300, side * 260)), 55, .appendage)
    }
    counts["antennule"] = 2
    counts["antenna"] = 2

    // -- the trunk ----------------------------------------------------------
    for i in 0..<thoracicSegments {
        let x: Float = thoracicSegmentX(i)
        bodyPrims.append(addAxisAligned(place(SIMD3(x, 0, 0)), thoracicSemi(i), .body))
    }
    counts["thoracic segment"] = thoracicSegments

    for x in genitalX {
        bodyPrims.append(addAxisAligned(place(SIMD3(x, -10, 0)), SIMD3(215, 250, 225), .body))
    }
    counts["genital segment"] = genitalSegments

    // The ovisac, and the cysts in it.
    bodyPrims.append(addAxisAligned(place(SIMD3(-820, -230, 0)), SIMD3(400, 270, 250), .body))
    var eggCount = 0
    for k in 0..<10 {
        let a: Float = Float(k) * 2.39996            // the golden angle, so they scatter
        let r: Float = 170 * sqrt(Float(k) / 10)
        let ex: Float = -820 + r * cos(a)
        let ey: Float = -240 + r * sin(a) * 0.62
        let ez: Float = Float((k % 3) - 1) * 95
        _ = addAxisAligned(place(SIMD3(ex, ey, ez)), SIMD3(112, 108, 105), .egg)
        eggCount += 1
    }
    counts["egg"] = eggCount

    for j in 0..<abdominalSegments {
        let x: Float = abdominalSegmentX(j)
        bodyPrims.append(addAxisAligned(place(SIMD3(x, 0, 0)), abdominalSemi(j), .body))
    }
    counts["abdominal segment"] = abdominalSegments

    bodyPrims.append(addAxisAligned(place(SIMD3(telsonX, 0, 0)), SIMD3(200, 125, 115), .body))
    counts["telson"] = 1

    // -- caudal furca, and its plumose setae --------------------------------
    var furcalSetae = 0
    for side in [Float(1), -1] {
        let root = place(SIMD3(-3560, -10, side * 70))
        let tip = place(SIMD3(-4180, -60, side * 150))
        _ = addCapsule(root, tip, 62, .appendage)
        let axis: SIMD3<Float> = simd_normalize(tip - root)
        for k in 0..<setaePerFurcalRamus {
            let u: Float = Float(k) / Float(setaePerFurcalRamus - 1)
            let base: SIMD3<Float> = root + (tip - root) * (0.42 + 0.58 * u)
            let fan: Float = radians(-46 + 92 * u)
            let outward = SIMD3<Float>(-cos(fan), -sin(fan) * 0.55, side * sin(fan) * 0.85)
            let dir: SIMD3<Float> = simd_normalize(outward + axis * 0.55)
            let clamped = clampThin(trueRadius: furcalSetaTrueRadius,
                                    enabled: !mutations.contains(.noClamp))
            prims.append(.capsule(from: world(base), to: world(base + dir * 620),
                                  radius: clamped.drawnRadius, tissue: .seta))
            furcalSetae += 1
        }
    }
    counts["furcal ramus"] = 2
    counts["furcal seta"] = furcalSetae

    // -- the gut ------------------------------------------------------------
    // A tube of higher σ running the whole length, from the mouth to the anus
    // on the telson. Overlapping capsules, which the per-tissue union merges
    // into one tube instead of a string of dark beads.
    var gutPrims = 0
    for k in 0..<gutSegments {
        let u0: Float = Float(k) / Float(gutSegments)
        let u1: Float = Float(k + 1) / Float(gutSegments)
        func gutPoint(_ u: Float) -> SIMD3<Float> {
            let x: Float = mouthPosition.x + (telsonX - mouthPosition.x) * u
            let y: Float = -90 + 100 * u * u
            return place(SIMD3(x, y, 0))
        }
        _ = addCapsule(gutPoint(u0), gutPoint(u1), 78, .gut)
        gutPrims += 1
    }
    counts["gut segment"] = gutPrims

    // -- the 22 phyllopods --------------------------------------------------
    // No regional specialisation: every one is the same limb, differing only
    // in size. Each is three lobes — endopodite, exopodite (the flabellum) and
    // epipodite (the gill) — plus the gnathobase, the medial endite that walls
    // the food groove and hands food forward along it.
    var lobeCount = 0
    var setaCount = 0
    for i in 0..<thoracicSegments {
        let semi: SIMD3<Float> = thoracicSemi(i)
        let x: Float = thoracicSegmentX(i)
        // The limbs shorten toward the back, which is the only way they differ.
        let size: Float = 1.0 - 0.22 * Float(i) / Float(thoracicSegments - 1)
        let angle: Float = limbAngle(i, frame: f, mutations: mutations)
        let spread: Float = limbSpread(i, frame: f, mutations: mutations)
        for side in [Float(1), -1] {
            let hinge: SIMD3<Float> = place(SIMD3(x, -semi.y * 0.80, side * semi.z * 0.62))
            let splay: Float = radians(26 + 6 * Float(i) / Float(thoracicSegments - 1))
            let (u, fore, lat) = limbFrame(angle: angle, splay: splay, side: side)
            hinges.append(world(hinge))
            var lobes: [Int] = []

            // A phyllopod is not a flat plate: each lobe is cupped, so its flat
            // face is tilted out of the pure fore-aft plane. Without that every
            // lobe on one side of the animal is exactly edge-on to the camera
            // at once, and a fan of soft blades turns into a row of hard bars.
            //
            // The cup is a roll ABOUT the limb's own axis, so a lobe never
            // leaves the ray that runs out from its hinge. Tilting the axis
            // instead — the obvious way to write this — moves the outer lobes
            // fore and aft by ±90 µm, which is most of the 155 µm the beat
            // leaves between one limb and the next, and neighbouring limbs then
            // pass through each other. The interpenetration test found that.
            func cupped(_ degrees: Float) -> (SIMD3<Float>, SIMD3<Float>) {
                let a: Float = radians(degrees)
                let ca: Float = cos(a)
                let sa: Float = sin(a)
                let normal: SIMD3<Float> = fore * ca + lat * sa
                let broad: SIMD3<Float> = lat * ca - fore * sa
                return (normal, broad)
            }
            func lobe(_ along: Float, _ sideways: Float, _ a: Float, _ b: Float, _ c: Float,
                      _ cup: Float) -> Int {
                let (normal, broad) = cupped(cup)
                let centre: SIMD3<Float> = hinge + u * (along * size) + lat * (sideways * size)
                let m = simd_float3x3(columns: (u * (a * size), normal * b, broad * (c * size)))
                return addOriented(centre, m, .limb)
            }
            // endopodite, then the broad exopodite that carries the setal
            // fringe, then the epipodite behind it, then the gnathobase.
            lobes.append(lobe(440, -70, 330, 30, 225, 9))
            let exoA: Float = 360 * (0.72 + 0.28 * spread)
            let exoC: Float = 295 * (0.58 + 0.42 * spread)
            let (exoNormal, exoBroad) = cupped(-7)
            let exoCentre: SIMD3<Float> = hinge + u * (850 * size) + lat * (130 * size)
            let exoM = simd_float3x3(columns: (u * (exoA * size), exoNormal * 26,
                                               exoBroad * (exoC * size)))
            let exoIndex: Int = addOriented(exoCentre, exoM, .limb)
            lobes.append(exoIndex)
            lobes.append(lobe(430, 290, 250, 34, 175, 15))
            lobes.append(lobe(200, -60, 140, 34, 70, 4))
            lobeCount += 4

            // The setal fringe on the exopodite's margin.
            for k in 0..<setaePerLimb {
                let t: Float = Float(k) / Float(setaePerLimb - 1)
                let alpha: Float = radians(-84 + 168 * t)
                let ca: Float = cos(alpha)
                let sa: Float = sin(alpha)
                let rim: SIMD3<Float> = exoCentre + u * (exoA * size * ca * 0.94)
                    + exoBroad * (exoC * size * sa * 0.94)
                let dir: SIMD3<Float> = simd_normalize(u * (ca * exoC) + exoBroad * (sa * exoA))
                let length: Float = 210 * size * (0.74 + 0.26 * spread)
                let clamped = clampThin(trueRadius: setaTrueRadius,
                                        enabled: !mutations.contains(.noClamp))
                prims.append(.capsule(from: world(rim), to: world(rim + dir * length),
                                      radius: clamped.drawnRadius, tissue: .seta))
                setaCount += 1
            }
            limbLobes.append(lobes)
        }
    }
    counts["phyllopod"] = limbLobes.count
    counts["limb lobe"] = lobeCount
    counts["limb seta"] = setaCount

    // -- the water, as tracers ----------------------------------------------
    // There is no fluid solver here. These are algal cells on a prescribed
    // conveyor, and the conveyor is what carries the claim: 5.5 mm/s past the
    // animal, which over one 0.8 s loop is exactly 4.4 mm.
    let p: Float = loopPhase(frame: f)
    let step: Float = swimmingSpeedMicronsPerSecond * Float(loopRealSeconds)   // 4400 µm
    let lanes = 4
    let span: Float = step * Float(lanes)
    let fieldMin: Float = -8000
    var freeTracers = 0
    let seedY: [Float] = [-2350, -1750, -1150, -620, 180, 640, -1450, 900, 1420, -2050, -260]
    let seedZ: [Float] = [700, -980, 520, -340, 1080, -720, 260, -1180, -560, 340, 980]
    for seed in 0..<seedY.count {
        let base: Float = Float(seed) * (step / Float(seedY.count)) + 130
        for lane in 0..<lanes {
            let raw: Float = base + Float(lane) * step - p * step + span
            let x: Float = fieldMin + raw.truncatingRemainder(dividingBy: span)
            let centre = SIMD3<Float>(x, seedY[seed], seedZ[seed])
            _ = addAxisAligned(centre, SIMD3(algaRadius, algaRadius, algaRadius), .alga)
            freeTracers += 1
        }
    }
    counts["free tracer"] = freeTracers

    // Captured cells, riding the food groove FORWARD to the mouth. The limbs
    // sweep water back; the setae retain what is in it; the gnathobases pass
    // it medially into the groove, and the groove runs it the other way, to
    // the mouth. The same beat swims, breathes and feeds.
    var grooveTracers = 0
    for k in 0..<8 {
        let offset: Float = Float(k) / 8
        var s: Float = p + offset
        if s >= 1 { s -= 1 }
        let centre: SIMD3<Float> = place(foodGroovePoint(s))
        // It appears where a limb hands it in and is gone when it is eaten.
        let fadeIn: Float = min(s / 0.07, 1)
        let fadeOut: Float = min((1 - s) / 0.10, 1)
        let fade: Float = min(fadeIn, fadeOut)
        let r: Float = algaRadius * fade * fade
        if r > 0.8 {
            _ = addAxisAligned(centre, SIMD3(r, r, r), .alga)
            grooveTracers += 1
        }
    }
    counts["groove tracer"] = grooveTracers
    counts["total"] = prims.count

    return Pose(prims: prims, limbLobes: limbLobes, limbHinge: hinges,
                bodyPrims: bodyPrims, counts: counts, rotation: R)
}

// MARK: - The constants table
//
// Every number above that a viewer might take for a fact appears here with
// where it came from. A test refuses the build if any entry has no source.

private let countedConstants: [Constant] = [
    Constant(name: "thoracic segments", value: Double(thoracicSegments), unit: "",
             evidence: .measured,
             source: "Fox, Invertebrate Anatomy OnLine — Artemia franciscana"),
    Constant(name: "pairs of functional thoracopods", value: Double(phyllopodPairs), unit: "",
             evidence: .measured,
             source: "Criel & Macrae, Artemia Morphology and Structure (2002), ch. 1"),
    Constant(name: "phyllopods drawn", value: Double(phyllopodPairs * 2), unit: "",
             evidence: .measured,
             source: "Criel & Macrae (2002) — 11 pairs, no regional specialisation"),
    Constant(name: "abdominal segments", value: Double(abdominalSegments), unit: "",
             evidence: .measured,
             source: "Fox, Invertebrate Anatomy OnLine — Artemia franciscana"),
    Constant(name: "genital segments", value: Double(genitalSegments), unit: "",
             evidence: .measured,
             source: "Fox, Invertebrate Anatomy OnLine — the genital region"),
    Constant(name: "compound eyes", value: Double(compoundEyes), unit: "",
             evidence: .measured,
             source: "Fox, Invertebrate Anatomy OnLine — paired stalked compound eyes"),
    Constant(name: "naupliar eye pigment cups", value: Double(naupliarEyeCups), unit: "",
             evidence: .measured,
             source: "Fox, Invertebrate Anatomy OnLine — median naupliar eye, three cups"),
    Constant(name: "adult body length", value: Double(adultLengthMicrons) / 1000, unit: "mm",
             evidence: .measured,
             source: "FAO Fisheries Technical Paper 361 (1996) — adults 8–10 mm"),
    Constant(name: "adult swimming speed", value: 5.5, unit: "mm/s",
             evidence: .measured,
             source: "Larsen, Madsen & Riisgård, Aquatic Biology 4:47–54 (2008)"),
    Constant(name: "swims ventral side up", value: 1, unit: "",
             evidence: .measured,
             source: "Fox, Invertebrate Anatomy OnLine — the ventral light reaction"),
    Constant(name: "metachrony is adlocomotory", value: 1, unit: "",
             evidence: .measured,
             source: "Ford et al., Integr. Comp. Biol. 62(3):791 (2022)"),

]

private let modelConstants: [Constant] = [
    Constant(name: "beat frequency", value: Double(beatFrequencyHz), unit: "Hz",
             evidence: .model,
             source: "MODEL: extrapolated past Williams 1994 (Biol. Bull. 187, "
                   + "doi:10.2307/1542239), 9.5 Hz at 0.4 mm → 6.7 Hz at 4 mm larvae"),
    Constant(name: "phase lag between neighbouring limbs", value: Double(phaseLagFraction),
             unit: "cycles", evidence: .model,
             source: "MODEL: 1/11 so exactly one wave sits on eleven limbs and the loop closes"),
    Constant(name: "limb sweep amplitude", value: Double(limbSweepAmplitude * 180 / .pi),
             unit: "deg", evidence: .model,
             source: "MODEL: chosen so neighbouring limbs never interpenetrate"),
    Constant(name: "stroke asymmetry", value: Double(strokeAsymmetry), unit: "",
             evidence: .model,
             source: "MODEL: a periodic phase warp, power stroke quicker than recovery"),
    Constant(name: "food groove transport", value: Double(foodGrooveLength) / 1000 / loopRealSeconds,
             unit: "mm/s", evidence: .model,
             source: "MODEL: one groove length per loop, chosen so the conveyor closes"),
    Constant(name: "sigma, body", value: Double(sigmaBody.y), unit: "1/µm", evidence: .model,
             source: "MODEL: tuned so the trunk reads blue-grey against the cyan field"),
    Constant(name: "sigma, limb", value: Double(sigmaLimb.y), unit: "1/µm", evidence: .model,
             source: "MODEL: tuned so overlapping limbs read darker"),
    Constant(name: "sigma, seta (true)", value: Double(sigmaSetaTrue.y), unit: "1/µm",
             evidence: .model, source: "MODEL: tuned for the setal fringe"),
    Constant(name: "sigma, compound eye", value: Double(sigmaCompoundEye.y), unit: "1/µm",
             evidence: .model, source: "MODEL: tuned to near-black with a red-brown cast"),
    Constant(name: "sigma, naupliar eye", value: Double(sigmaNaupliarEye.y), unit: "1/µm",
             evidence: .model, source: "MODEL: tuned to a dark brown speck"),
    Constant(name: "sigma, gut", value: Double(sigmaGut.y), unit: "1/µm", evidence: .model,
             source: "MODEL: excess over the trunk, tuned to olive-tan"),
    Constant(name: "sigma, appendage", value: Double(sigmaAppendage.y), unit: "1/µm",
             evidence: .model, source: "MODEL: tuned so antennules read as thin lines"),
    Constant(name: "sigma, alga", value: Double(sigmaAlga.y), unit: "1/µm", evidence: .model,
             source: "MODEL: a green cell — takes red and blue, passes green"),
    Constant(name: "sigma, egg", value: Double(sigmaEgg.y), unit: "1/µm", evidence: .model,
             source: "MODEL: tuned to brown cysts in the ovisac"),
    Constant(name: "the water", value: 0, unit: "", evidence: .model,
             source: "MODEL: no fluid solver — tracers on a prescribed conveyor"),
    Constant(name: "seta diameter", value: Double(setaTrueRadius * 2), unit: "µm",
             evidence: .model,
             source: "MODEL: mid-range of the 2–5 µm setae described for branchiopod limbs"),

]

private let derivedConstants: [Constant] = [
    Constant(name: "kinematic viscosity of the medium", value: kinematicViscosity, unit: "m²/s",
             evidence: .measured,
             source: "seawater at 20 °C; see the note — real Artemia brine is more viscous"),
    Constant(name: "Reynolds number, body", value: ReynoldsNumbers().body, unit: "",
             evidence: .derived, source: "DERIVED here: U L / ν from the three above"),
    Constant(name: "Reynolds number, limb", value: ReynoldsNumbers().limb, unit: "",
             evidence: .derived, source: "DERIVED here: limb tip speed × limb length / ν"),
    Constant(name: "Reynolds number, seta", value: ReynoldsNumbers().seta, unit: "",
             evidence: .derived, source: "DERIVED here: limb tip speed × seta diameter / ν"),
    Constant(name: "pixel scale", value: Double(micronsPerPixel), unit: "µm/px",
             evidence: .derived,
             source: "DERIVED: the orthographic field divided by the frame width"),
    Constant(name: "sub-pixel seta width", value: Double(setaTrueRadius * 2 / micronsPerPixel),
             unit: "px", evidence: .derived,
             source: "DERIVED: seta diameter ÷ pixel scale — this is why the clamp exists"),
    Constant(name: "slow motion", value: slowMotionFactor, unit: "×", evidence: .derived,
             source: "DERIVED: 30 frames a cycle at 80 ms against a 5 Hz beat"),
    Constant(name: "loop, real time", value: loopRealSeconds, unit: "s", evidence: .derived,
             source: "DERIVED: 120 frames ÷ (5 Hz × 30 frames a cycle) × 4 cycles"),
    Constant(name: "loop, displayed time", value: loopDisplayedSeconds, unit: "s",
             evidence: .derived, source: "DERIVED: 120 frames × 80 ms"),
    Constant(name: "tracer travel per loop", value: Double(swimmingSpeedMicronsPerSecond)
                * loopRealSeconds / 1000, unit: "mm", evidence: .derived,
             source: "DERIVED: 5.5 mm/s × the loop's real duration"),
]

/// The three groups, in one table. Kept as three literals because the Swift
/// type checker on the laptop's toolchain gives up on a forty-element array
/// literal of structs; `make typecheck` is what says so.
let artemiaConstants: [Constant] = countedConstants + modelConstants + derivedConstants

// MARK: - What the caption bar says
//
// Here rather than in main.swift so that the tests can measure it: a caption
// that overflows the frame is a bug like any other.

func swimCaption() -> Caption {
    Caption(title: "Rendered model of a brine shrimp swimming",
            subtitle: "Artemia franciscana, adult female · transmitted light",
            facts: "11 thoracic segments · 22 phyllopods · wave runs tail → head",
            aside: "")
}

// No evidence bar on this frame. Step 10 earned one because the thing being
// rated CHANGED as the render played — the three transformation routes each
// carried a different standard of evidence, and the bar flipped between them.
// Here it would have said the same three lines for all 120 frames, which makes
// it furniture rather than information.
//
// The provenance has not gone anywhere: every constant in `artemiaConstants`
// still carries its evidence level and its source, and the tests below still
// fail if a measured constant cites nothing or a derived one does not say so.
// That is where the claim belongs — in the data and in the results page, not
// stamped on a picture that never changes its mind.
