// A germinating coconut, built from published stages rather than from a film.
//
// There is no usable time-lapse of a coconut germinating. It takes months, and
// the half of it worth watching happens inside an opaque shell. What is
// published is a set of STAGED OBSERVATIONS — nuts opened at intervals and
// described — so this render interpolates between documented stages. It is not
// a replay of a measured time series, and the constants table says so on every
// line that is a rate.
//
// What the render exists to show is not the sprout. It is the HAUSTORIUM.
//
// The white ball in a sprouted coconut is not coconut meat. It is a new organ:
// after germination begins, the single cotyledon stays inside the shell and
// swells into a spongy absorbing body — the coconut apple — which fills the
// water cavity and digests the solid endosperm from the inside, feeding the
// seedling for months. It fills the cavity in roughly four months. That is the
// whole reason this is a cutaway: outside, a shoot and a fan of roots; inside,
// in the dark, a ball slowly eating the nut it lives in, and the two are the
// same budget.
//
// Three things a bean-germination mental model gets wrong, and they are the
// content of this render rather than footnotes to it:
//
//   1. A COCONUT HAS NO TAPROOT. It grows an adventitious fibrous root system:
//      a fan of roughly equal roots from the base of the sprout. `Tests` fails
//      if any root exceeds 1.4× the mean.
//   2. THE SHELL STAYS WHOLE. The shoot leaves through one of the three eyes —
//      the germination pore. Only one of the three is soft; the cotyledonary
//      petiole extends through it and displaces it. Nothing cracks.
//   3. THE COTYLEDON NEVER LEAVES THE SHELL. It stays inside and becomes the
//      haustorium.
//
// And a fourth, which the geometry here says out loud and which is the one
// correction this file makes to the usual telling: the haustorium does NOT get
// most of its bulk from the solid endosperm. It gets most of it from the
// coconut water, which it absorbs as it swells. See `Budget` below — by the
// end of this loop roughly three quarters of the haustorium's volume came from
// the liquid endosperm and one quarter from the solid.
//
// Sources
//   Fruit Biology of Coconut (*Cocos nucifera* L.), *Plants* 11(23):3293 (2022)
//     — embryo structure, the cotyledonary petiole through the germination
//       pore, displacement of the soft eye, haustorium formation.
//   Comprehensive biochemical profiling of coconut haustorium (2024)
//     — the cavity-filling timescale, ~4 months.
//   Germination rate is the significant characteristic determining coconut palm
//     diversity, *AoB PLANTS* — germination timing and its spread.
//   Darwin & Darwin, *The Power of Movement in Plants* (1880) — circumnutation,
//     borrowed here for the CHARACTER of growth, not its rate.

import CoreGraphics
import Foundation
import simd

// MARK: - The frame
//
// Portrait, and the first portrait render in the series. Nothing below assumes
// landscape: the camera's field is built from the pixel counts and one scale.

let frameWidth = 960
let frameViewHeight = 1160
let frameCaptionHeight = 120
let frameHeight = frameViewHeight + frameCaptionHeight      // 1280

/// Millimetres to the pixel. One number over the whole field, because the
/// camera is orthographic — which is also what makes the cut plane a plane on
/// screen and not a hyperbola.
let mmPerPixel: Float = 0.50

// MARK: - The loop
//
// One shot, held, dissolved. The growth runs once; the finished seedling is
// held for about two seconds; then the picture cross-dissolves back to the
// dormant nut. NOTHING EVER RUNS BACKWARDS. A dissolve is not a rewind, and
// the monotone-growth test exists to keep it that way.

let dormantFrames = 8
let growFrames = 84
let holdFrames = 28
let dissolveFrames = 24
let coconutFrameCount = dormantFrames + growFrames + holdFrames + dissolveFrames   // 144
let frameDelayCentiseconds = 7
let dissolveStartFrame = dormantFrames + growFrames + holdFrames                   // 120
let loopDisplayedSeconds: Double = Double(frameDelayCentiseconds) * Double(coconutFrameCount) / 100

/// What the growth stands for in real time. MODEL: interpolated between staged
/// observations, not a measured rate.
let modelledMonths: Double = 4.0

/// How far through the growth frame `f` is, in [0, 1]. Flat through the dormant
/// frames, flat again through the hold and the dissolve.
func growth(frame f: Int) -> Float {
    let a: Int = dormantFrames
    let b: Int = dormantFrames + growFrames
    if f <= a { return 0 }
    if f >= b { return 1 }
    let num: Float = Float(f - a)
    let den: Float = Float(b - a)
    return num / den
}

/// How much of the dissolve has happened by frame `f`, in [0, 1]. 1 means the
/// frame IS the dormant nut again, pixel for pixel, which is what closes the
/// loop without ever playing anything in reverse.
func dissolve(frame f: Int) -> Float {
    if f < dissolveStartFrame { return 0 }
    let num: Float = Float(f - (dissolveStartFrame - 1))
    let den: Float = Float(dissolveFrames)
    return min(num / den, 1)
}

func smoothstep(_ a: Float, _ b: Float, _ x: Float) -> Float {
    if b <= a { return x < a ? 0 : 1 }
    let t: Float = min(max((x - a) / (b - a), 0), 1)
    let u: Float = t * t
    let v: Float = 3 - 2 * t
    return u * v
}

// MARK: - Materials
//
// Fifteen of them, and their ORDER IS THEIR PRIORITY: where two materials
// occupy the same point, the one with the higher index is what you see. That
// single rule is what makes a stack of nested solid ellipsoids behave like a
// stack of shells without any of them having to be hollow, and it is also what
// lets a root run through soil and a shoot run through husk.

enum Material: Int, CaseIterable {
    case topsoil = 0
    case subsoil
    case exocarp            // the thin smooth skin
    case mesocarp           // the fibrous husk, coir
    case endocarp           // the hard shell, carrying the three germination pores
    case testa              // the thin brown seed coat
    case endosperm          // the solid white meat — this is what gets eaten, on camera
    case water              // the liquid endosperm filling the cavity
    case embryo             // a few millimetres, under the soft eye
    case haustorium         // the coconut apple: the swollen cotyledon
    case petiole            // the cotyledonary petiole, through the pore
    case shoot
    case blade
    case midrib
    case root

    var name: String {
        switch self {
        case .topsoil: return "topsoil"
        case .subsoil: return "subsoil"
        case .exocarp: return "exocarp"
        case .mesocarp: return "mesocarp (husk)"
        case .endocarp: return "endocarp (shell)"
        case .testa: return "testa"
        case .endosperm: return "solid endosperm"
        case .water: return "liquid endosperm"
        case .embryo: return "embryo"
        case .haustorium: return "haustorium"
        case .petiole: return "cotyledonary petiole"
        case .shoot: return "shoot"
        case .blade: return "leaf blade"
        case .midrib: return "midrib"
        case .root: return "adventitious root"
        }
    }
}

let materialCount = Material.allCases.count

/// The layer stack, outside in. The nesting test walks exactly this list.
let layerStack: [Material] = [.exocarp, .mesocarp, .endocarp, .testa, .endosperm, .water]

// MARK: - Packing a primitive with one of THIS step's materials
//
// The packing itself belongs to step 13 and is not rewritten here: these build
// the primitive step 13's way and then overwrite the one field that says which
// material it is, because step 13's `Tissue` names brine-shrimp tissues and
// there are fifteen of these.

extension GPUPrim {
    static func ellipsoid(centre: SIMD3<Float>, m: simd_float3x3, material: Material) -> GPUPrim {
        var p: GPUPrim = GPUPrim.ellipsoid(centre: centre, m: m, tissue: .body)
        p.r2.w = Float(material.rawValue)
        return p
    }

    static func sphere(centre: SIMD3<Float>, radius: Float, material: Material) -> GPUPrim {
        let m = simd_float3x3(diagonal: SIMD3<Float>(radius, radius, radius))
        return GPUPrim.ellipsoid(centre: centre, m: m, material: material)
    }

    /// An axis-aligned ellipsoid: semi-axes along x, y, z.
    static func spheroid(centre: SIMD3<Float>, semi: SIMD3<Float>, material: Material) -> GPUPrim {
        GPUPrim.ellipsoid(centre: centre, m: simd_float3x3(diagonal: semi), material: material)
    }

    static func capsule(from a: SIMD3<Float>, to b: SIMD3<Float>, radius: Float,
                        material: Material) -> GPUPrim {
        var p: GPUPrim = GPUPrim.capsule(from: a, to: b, radius: radius, tissue: .body)
        p.r2.w = Float(material.rawValue)
        return p
    }

    var material: Material { Material(rawValue: self.tissue) ?? .topsoil }
}

// MARK: - The nut, in millimetres
//
// One nut, lying on its side and half buried, which is how they are actually
// planted. The long axis is world x and the stem end — the end that carried
// the three eyes — points toward +x, where the sprout will come out.

let nutCentre = SIMD3<Float>(-55, 0, 0)

/// A whole husked fruit, 300 mm by 190 mm.
let exocarpSemi = SIMD3<Float>(150, 95, 95)
/// The skin is under a millimetre; drawn at 1.5 mm so it is more than one pixel
/// of hairline in the cut face. MODEL, for legibility.
let exocarpThickness: Float = 1.5
/// Coir. Thicker toward the ends than at the equator, which is what you get for
/// free from two concentric ellipsoids and is also what a coconut does.
let mesocarpSemi: SIMD3<Float> = exocarpSemi - exocarpThickness
let endocarpSemi = SIMD3<Float>(78, 56, 56)
let endocarpThickness: Float = 4
let testaSemi: SIMD3<Float> = endocarpSemi - endocarpThickness
let testaThickness: Float = 1.2
let endospermSemi: SIMD3<Float> = testaSemi - testaThickness      // (72.8, 50.8, 50.8)

/// The meat, at the start and at the end of the loop. The cavity is whatever is
/// left inside it, so these two numbers drive the whole interior budget.
let meatThicknessStart: Float = 13.0
let meatThicknessEnd: Float = 9.0

func ellipsoidVolume(_ s: SIMD3<Float>) -> Double {
    let a = Double(s.x)
    let b = Double(s.y)
    let c = Double(s.z)
    let k: Double = 4.0 / 3.0 * Double.pi
    return k * a * b * c
}

// MARK: - The three eyes
//
// Three germination pores on the stem end, 120° apart. Exactly one of them is
// soft, and it is the only one anything ever goes through. The other two stay
// sealed, and a test walks the whole endocarp surface every frame to prove it.

let poreCount = 3
let functionalPore = 0
/// How far off the stem-end pole the ring of pores sits.
let poreRingAngle: Float = radians(34)
/// Which way up the nut was laid. The soft eye ends up above the axis and
/// toward the viewer, and the clip plane is then set through its channel — see
/// `cutPlaneZ`.
let poreBearing0: Float = radians(35)
let poreRadius: Float = 7.5
/// How far the ring of pores sits off the nut's long axis.
let poreRingRadius: Float = endocarpSemi.y * sin(poreRingAngle)

/// Pore `k`'s centre, on the endocarp's outer surface, in world coordinates.
func poreCentre(_ k: Int) -> SIMD3<Float> {
    let bearing: Float = poreBearing0 + Float(k) * radians(120)
    let ca: Float = cos(poreRingAngle)
    let sa: Float = sin(poreRingAngle)
    let x: Float = endocarpSemi.x * ca
    let ring: Float = endocarpSemi.y * sa
    let y: Float = ring * cos(bearing)
    let z: Float = ring * sin(bearing)
    return nutCentre + SIMD3(x, y, z)
}

/// Where the petiole leaves the husk, and therefore where the sprout begins:
/// on the exocarp, thirty millimetres above the soil line, just past the stem
/// end of the fruit and on the soft eye's own side of it.
let sproutBase: SIMD3<Float> = {
    let y: Float = 14
    let z: Float = 16
    let ry: Float = y / exocarpSemi.y
    let rz: Float = z / exocarpSemi.z
    let left: Float = 1 - ry * ry - rz * rz
    let x: Float = exocarpSemi.x * left.squareRoot()
    return nutCentre + SIMD3(x, y, z)
}()

// MARK: - The cut
//
// The clip plane is z = `cutPlaneZ`; everything in front of it is discarded.
// It is OFF-CENTRE on purpose. Through the middle it would slice the embryo and
// the soft eye in half, which is the conventional diagram and a good deal
// harder to read; set forward of them, the pore and the embryo survive whole
// and are seen from inside the opened cavity.

/// Through the soft eye's channel, which is 18 mm off the nut's own long axis.
/// The plane therefore passes 1.9 mm in front of the embryo's centre rather
/// than through it: the embryo and the pore come out as near-full sections in
/// the cut face instead of being halved, which is the whole point of putting
/// the cut off-centre. It also means the bore the petiole makes through the
/// shell and the husk is exposed along its length — with the plane anywhere
/// else, that bore is buried under 50 mm of coir and the second beat of the
/// render is invisible.
let cutPlaneZ: Float = poreRingRadius * sin(poreBearing0)
/// How far the camera stands off the cut plane's normal. Straight down the
/// normal you would see only a flat disc — "a cutaway" and "a cross-section"
/// are not the same picture. This angle is what puts the remaining half's own
/// curved surface beside its cut face, which is the whole trick.
let cameraYaw: Float = radians(28)
let cameraStandOff: Float = 1200

// MARK: - Soil
//
// A very shallow dome rather than a plane, so the soil line has a little life
// in it: sag across the visible width is about eight millimetres. MODEL, like
// the horizon depth and the grain sizes, and chosen for legibility.
//
// It is an ellipsoid stretched along the fruit's own axis rather than a huge
// sphere, and that is a rendering decision as much as a drawing one. A sphere
// flat enough to read as ground has to be metres across, and the uniform grid
// takes ITS bounding box as the whole scene's — so one enormous primitive turns
// every grid cell into a box the size of a room and the acceleration structure
// into a slower way of doing brute force. Bounding the ground is what makes the
// grid worth building.

let soilSemi = SIMD3<Float>(1400, 340, 300)

/// How finely the grid is cut, in boxes per primitive. Two, not one: the scene
/// is only a few hundred primitives but two of them are the ground, and a
/// finer grid is what keeps the ground from being met by every ray in the
/// frame. `make bench` prints the whole curve.
let gridDensity: Float = 2
let topsoilDepth: Float = 60
let soilSurfaceY: Float = 0

// MARK: - The growth, stage by stage
//
// Every window below is a MODEL: the ORDER is what the staged observations
// give, the timing between them is interpolation. The haustorium's window is
// the whole loop, because it is the thing the render exists for.

struct Stages {
    var petiole: Float      // the petiole extends through the pore
    var shoot: Float        // the spear outside
    var green: Float        // the spear greens
    var leaf: Float         // the first true leaf
    var root: Float         // the fan of adventitious roots
    var haustorium: Float   // the coconut apple

    init(growth g: Float) {
        petiole = smoothstep(0.00, 0.16, g)
        shoot = smoothstep(0.10, 0.86, g)
        green = smoothstep(0.24, 0.62, g)
        leaf = smoothstep(0.60, 1.00, g)
        root = smoothstep(0.16, 0.94, g)
        haustorium = smoothstep(0.03, 1.00, g)
    }
}

// MARK: - The interior budget
//
// The nut is a closed box. Solid endosperm, liquid endosperm and haustorium
// together fill exactly the volume inside the testa, at every frame, and the
// three quantities are derived from two numbers so that the closure is exact
// rather than approximate:
//
//   m   the meat's thickness, 13 mm falling to 9 mm
//   f   how much of the cavity the haustorium fills, 0 rising to 0.93
//
// The cavity is the endosperm ellipsoid inset by m; the haustorium is f of the
// cavity by volume; the water is the rest. So
//
//   E + W + H = V(endosperm outer)      exactly, always
//
// and the interesting number falls out rather than being asserted: by the end
// the haustorium has gained 434 mL, of which 109 mL is volume vacated by
// digested solid endosperm and 325 mL is coconut water it absorbed. The usual
// telling — "the haustorium is made of the meat it digests" — is about three
// quarters wrong BY VOLUME. It is right about where the food comes from and
// wrong about where the bulk comes from.

let haustoriumFinalFill: Float = 0.93

struct Budget {
    var meatThickness: Float
    var cavitySemi: SIMD3<Float>
    var cavityVolume: Double
    var endospermVolume: Double      // the solid meat still there
    var haustoriumVolume: Double
    var waterVolume: Double
    var fill: Float                  // haustorium / cavity, by volume

    init(stages: Stages) {
        let h: Float = stages.haustorium
        let span: Float = meatThicknessStart - meatThicknessEnd
        meatThickness = meatThicknessStart - span * h
        cavitySemi = endospermSemi - meatThickness
        cavityVolume = ellipsoidVolume(cavitySemi)
        endospermVolume = ellipsoidVolume(endospermSemi) - cavityVolume
        fill = haustoriumFinalFill * h
        haustoriumVolume = cavityVolume * Double(fill)
        waterVolume = cavityVolume - haustoriumVolume
    }

    /// The whole interior, which never changes.
    var interiorVolume: Double { endospermVolume + waterVolume + haustoriumVolume }
}

let budgetAtStart = Budget(stages: Stages(growth: 0))
let budgetAtEnd = Budget(stages: Stages(growth: 1))

/// Solid endosperm digested by the end, in mm³ — the cavity's growth.
let endospermLost: Double = budgetAtEnd.cavityVolume - budgetAtStart.cavityVolume
/// Coconut water absorbed by the end, in mm³.
let waterAbsorbed: Double = budgetAtStart.waterVolume - budgetAtEnd.waterVolume
/// How many times its own weight of vacated meat volume the haustorium gains.
/// DERIVED, and the number that says the bulk is water.
let haustoriumBulkRatio: Double = budgetAtEnd.haustoriumVolume / endospermLost

// MARK: - Circumnutation
//
// Growing tips do not rise like a piston; they trace a slow helix as they
// extend. Darwin wrote a book about it. Borrowed here for the CHARACTER of the
// growth and labelled MODEL, because nobody has published a coconut's nutation
// period and because the timescale here is months compressed to seconds, so no
// real period would survive the compression anyway.
//
// It displaces the tip in x and z and never in y, which is deliberate: the
// monotone-growth test measures the shoot by its tip height, and a wobble that
// touched y would make "never decreases" a lie on some frames for a reason that
// has nothing to do with growing backwards.

let nutationFrames = 24
let nutationAmplitude: Float = 9        // mm at the tip
let nutationTwistsPerShoot: Float = 0.9

func nutationOffset(arc u: Float, frame f: Int) -> SIMD3<Float> {
    let turns: Float = Float(f % nutationFrames) / Float(nutationFrames)
    let along: Float = nutationTwistsPerShoot * u
    let phase: Float = 2 * Float.pi * (turns - along)
    let amp: Float = nutationAmplitude * u * u
    return SIMD3(amp * cos(phase), 0, amp * sin(phase))
}

// MARK: - The shoot
//
// A spear that leaves the husk horizontally and turns up within the first third
// of its length. Radius tapers from 22 mm at the collar to 8 mm at the tip.

let shootSegments = 16
let shootArcLength: Float = 230
let shootBaseRadius: Float = 18
let shootTipRadius: Float = 9
let shootTurnFraction: Float = 0.30

/// The tangent angle above horizontal at arc fraction `u`.
func shootTangentAngle(_ u: Float) -> Float {
    let t: Float = smoothstep(0, shootTurnFraction, u)
    return Float.pi / 2 * t
}

/// The shoot's centreline at arc fraction `u`, for a shoot of arc length `L`.
func shootPoint(_ u: Float, length L: Float, frame f: Int) -> SIMD3<Float> {
    let steps = 48
    var p: SIMD3<Float> = sproutBase
    let n: Int = max(1, Int((u * Float(steps)).rounded(.up)))
    let du: Float = u / Float(n)
    for i in 0..<n {
        let mid: Float = (Float(i) + 0.5) * du
        let a: Float = shootTangentAngle(mid)
        let dx: Float = cos(a) * L * du
        let dy: Float = sin(a) * L * du
        p.x += dx
        p.y += dy
    }
    // The spear leaves the husk in the plane of the cut — so its collar is
    // seen in section, like everything else the plane passes through — and
    // swings clear of it within the first third, after which it is a solid
    // lit surface. That transition is not a seam: it is the cut plane crossing
    // a curved body, which is what a cutaway looks like.
    p.z += shootLeanZ(u)
    let wobble: SIMD3<Float> = nutationOffset(arc: u, frame: f)
    return p + wobble
}

let shootClearance: Float = 44

func shootLeanZ(_ u: Float) -> Float {
    let t: Float = smoothstep(0, 0.32, u)
    return -shootClearance * t
}

// MARK: - The first true leaf
//
// ENTIRE and lance-shaped. Not pinnate, not split, not feathery. The split
// frond is months past the end of this loop, and drawing one would be the
// second most visible error available here, after a taproot. The test below
// checks the half-width profile has one hump and no interior zeros, which is
// what "entire" means as a number.
//
// The primitive is a BENT RIBBON: a curved strip with a midrib. A palm blade is
// not an ellipsoid. It is built as a chain of flattened ellipsoids in two
// shallow-tilted halves, which gives the strip its keel, plus capsules down the
// middle for the midrib. It is composed from step 13's primitives rather than
// added to its kernel, on purpose: this step reuses step 13's traversal
// verbatim and does not get to change what a primitive is.

let bladeStations = 26
let bladeLengthFinal: Float = 125
let bladeMaxHalfWidth: Float = 21
let bladeThickness: Float = 1.6
let bladeKeelAngle: Float = radians(15)

/// The blade is turned to face the viewer, and arches in the plane of the
/// frame. A leaf presenting its edge would be a dark strip four pixels wide and
/// would say nothing at all about whether it is entire or pinnate — which is
/// the one thing this leaf is here to say. A first leaf orients to the light in
/// any case; here the light and the camera are on the same side.
let bladeLateral: SIMD3<Float> = SIMD3(cos(cameraYaw), 0, -sin(cameraYaw))
let bladeVertical = SIMD3<Float>(0, 1, 0)
let leafletCountEntire = 1

/// The blade's half-width at arc fraction `s`. One hump, peaking at about 42%
/// of the way along: lance-shaped.
func bladeHalfWidth(_ s: Float, leaflets: Int = leafletCountEntire) -> Float {
    let clamped: Float = min(max(s, 0), 1)
    let e: Float = pow(clamped, 0.8)
    let envelope: Float = sin(Float.pi * e)
    if leaflets <= 1 {
        let shaped: Float = pow(max(envelope, 0), 0.9)
        return bladeMaxHalfWidth * shaped
    }
    // The mutation: a pinnate leaf, months too early.
    let n: Float = Float(leaflets)
    let u: Float = clamped * n
    let frac: Float = u - u.rounded(.down)
    let pinna: Float = sin(Float.pi * frac)
    return bladeMaxHalfWidth * envelope * pinna
}

// MARK: - The roots
//
// Adventitious and fibrous: a fan of roughly equal roots from the base of the
// sprout, spreading sideways and down. NO TAPROOT, no dominant root. The
// lengths below vary by ±14% about their mean, and the test allows 1.4×.

let primaryRoots = 13
let lateralsPerRoot = 2
let rootSegments = 11
let lateralSegments = 6
let rootMeanLength: Float = 185
let rootBaseRadius: Float = 6.0
let rootTipRadius: Float = 1.6

/// A deterministic spread in [0, 1) — the golden angle, so thirteen roots
/// scatter rather than pair up.
func rootJitter(_ k: Int) -> Float {
    let g: Float = Float(k) * 0.6180339887
    return g - g.rounded(.down)
}

func rootLength(_ k: Int, taproot: Bool) -> Float {
    let j: Float = rootJitter(k)
    let spread: Float = 0.86 + 0.28 * j
    let base: Float = rootMeanLength * spread
    if taproot && k == 0 { return base * 3.2 }
    return base
}

/// Root `k`'s direction, as a horizontal bearing and a dip below horizontal.
///
/// The bearing matters more here than it looks. A SECTION SHOWS WHAT THE PLANE
/// CUTS, and soil is opaque: a root that dives away behind the clip plane is
/// buried under two hundred millimetres of subsoil and cannot be seen, and one
/// that leans in front of it has been cut away entirely. So the fan is built in
/// two lobes centred on the bearings that run PARALLEL to the clip plane —
/// dir.z ≈ 0 — with a real spread either side of them. The roots near the
/// middle of a lobe run the length of the section and read as roots; the ones
/// out at the edges give the short cut ends you actually see in a sectioned
/// root ball. Both are honest, and neither is a plane of roots pretending to be
/// a fan.
func rootBearing(_ k: Int, taproot: Bool) -> (SIMD3<Float>, SIMD3<Float>) {
    if taproot && k == 0 {
        return (SIMD3(0, -1, 0), SIMD3(0, -1, 0))
    }
    let lobe: Float = (k % 2 == 0) ? 0 : Float.pi
    let scatter: Float = rootJitter(k * 3 + 1) - 0.5
    let azimuth: Float = lobe + radians(64) * scatter
    let n: Float = Float(primaryRoots - 1)
    let u: Float = Float(k) / n
    // The backward lobe has to dip harder: it leaves straight into the stem end
    // of the fruit and has to get under it.
    let shallow: Float = (k % 2 == 0) ? radians(17) : radians(34)
    let dip: Float = shallow + radians(52) * u
    func aim(_ d: Float) -> SIMD3<Float> {
        let cd: Float = cos(d)
        let sd: Float = sin(d)
        let x: Float = cd * cos(azimuth)
        let z: Float = cd * sin(azimuth)
        return simd_normalize(SIMD3(x, -sd, z))
    }
    // They steepen a little as they run, and no one of them leads: that is what
    // an adventitious system looks like and what a taproot does not.
    return (aim(dip), aim(dip + radians(16)))
}

/// Where root `k` leaves the sprout's collar: clustered on the clip plane, so
/// the crown itself is always in section.
func rootOrigin(_ k: Int) -> SIMD3<Float> {
    let n: Float = Float(primaryRoots - 1)
    let u: Float = Float(k) / n
    let a: Float = 2 * Float.pi * u
    let dx: Float = 7 * cos(a)
    let dz: Float = 5 * sin(a)
    // Clear of the husk, which is what keeps the fan a fan: a collar ON the
    // fruit's own surface would have every root ejected along it by the
    // push-out below, and thirteen roots would leave as a starburst.
    let x: Float = sproutBase.x + 13 + dx
    let y: Float = sproutBase.y - 8 + 5 * rootJitter(k * 5 + 2)
    return SIMD3(x, y, cutPlaneZ + dz)
}

/// Roots grow AROUND the husk, not through it. The fan leaves the collar right
/// against the fruit, so any point that lands inside the exocarp is pushed back
/// out to its surface plus a margin. Without this the priority rule would
/// happily draw a root through forty millimetres of coir, because a root
/// outranks a husk — and it would be a lie about how the plant is put together.
func pushOutOfNut(_ p: SIMD3<Float>, margin: Float) -> SIMD3<Float> {
    let rel: SIMD3<Float> = p - nutCentre
    let u: SIMD3<Float> = rel / exocarpSemi
    let q: Float = simd_length(u)
    if q >= 1 || q < 0.25 { return p }
    let onSurface: SIMD3<Float> = rel / q
    let grad: SIMD3<Float> = onSurface / (exocarpSemi * exocarpSemi)
    let outward: SIMD3<Float> = simd_normalize(grad)
    return nutCentre + onSurface + outward * margin
}

// MARK: - Breakages, for the mutation check
//
// Switched on with COCO_MUTATE. Nothing here is ever on in a real render; each
// one exists so the suite can be shown to go red for it.

struct Breakage: OptionSet {
    let rawValue: UInt32
    static let taproot = Breakage(rawValue: 1 << 0)    // one root three times the rest
    static let invert = Breakage(rawValue: 1 << 1)     // testa outside the endocarp
    static let budget = Breakage(rawValue: 1 << 2)     // haustorium grows, endosperm does not
    static let pinnate = Breakage(rawValue: 1 << 3)    // a split frond, months too early
    static let mailbox = Breakage(rawValue: 1 << 4)    // a primitive met once per grid box

    static func fromEnvironment() -> Breakage {
        let text: String = ProcessInfo.processInfo.environment["COCO_MUTATE"] ?? ""
        var out = Breakage([])
        for word in text.split(separator: ",") {
            switch word.trimmingCharacters(in: .whitespaces) {
            case "taproot": out.insert(.taproot)
            case "invert": out.insert(.invert)
            case "budget": out.insert(.budget)
            case "pinnate": out.insert(.pinnate)
            case "mailbox": out.insert(.mailbox)
            default: break
            }
        }
        return out
    }

    /// The bit the kernel reads. Step 13's gather takes bit 1 as "no mailbox",
    /// and this step inherits that gather unchanged, so the bit must match.
    var kernelBits: UInt32 { contains(.mailbox) ? 2 : 0 }
}

/// How many segments to spend on a part that is only `fraction` grown.
///
/// This is not tidiness, it is the interval budget. A shoot drawn with all
/// sixteen of its capsules while it is still thirty millimetres long puts
/// sixteen balls of radius eighteen inside one another, and a ray through that
/// blob collects sixteen intervals for one bud — which is how a list that holds
/// forty overflows silently, with the dropped intervals hidden inside something
/// else. The fix is to draw a short thing with few segments, which is also what
/// it looks like.
func segments(_ full: Int, grown fraction: Float, atLeast floor: Int = 3) -> Int {
    let scaled: Float = Float(full) * min(max(fraction, 0), 1)
    let rounded: Int = Int(scaled.rounded())
    return max(floor, min(full, rounded))
}

// MARK: - One frame of the plant

struct Plant {
    var prims: [GPUPrim] = []
    var counts: [String: Int] = [:]
    var budget: Budget
    var stages: Stages
    var growth: Float
    var shootTipY: Float = 0
    var shootArc: Float = 0
    var rootLengths: [Float] = []
    var bladeLength: Float = 0
    var leafletCount: Int = leafletCountEntire
    var haustoriumCentre = SIMD3<Float>(repeating: 0)
    var haustoriumSemi = SIMD3<Float>(repeating: 0)
    var cavityCentre: SIMD3<Float> { nutCentre }
    var petioleThrough: Int = functionalPore

    var maxRootLength: Float { rootLengths.max() ?? 0 }
    var meanRootLength: Float {
        if rootLengths.isEmpty { return 0 }
        var total: Float = 0
        for l in rootLengths { total += l }
        return total / Float(rootLengths.count)
    }
}

/// Semi-axes of layer `m` for this frame. Only the cavity moves; the mutation
/// `invert` swaps the shell and the seed coat, which is the inversion the
/// nesting test exists to catch.
func layerSemi(_ m: Material, budget b: Budget, breakage: Breakage) -> SIMD3<Float> {
    switch m {
    case .exocarp: return exocarpSemi
    case .mesocarp: return mesocarpSemi
    case .endocarp: return breakage.contains(.invert) ? testaSemi : endocarpSemi
    case .testa: return breakage.contains(.invert) ? endocarpSemi : testaSemi
    case .endosperm: return endospermSemi
    case .water: return b.cavitySemi
    default: return SIMD3(repeating: 0)
    }
}

/// Builds the whole plant for one frame, already in world coordinates.
func poseCoconut(frame f: Int, breakage: Breakage = []) -> Plant {
    let g: Float = growth(frame: f)
    let stages = Stages(growth: g)
    var budget = Budget(stages: stages)
    if breakage.contains(.budget) {
        // The haustorium swells and the meat never thins: the coconut eats
        // nothing and grows anyway.
        budget.meatThickness = meatThicknessStart
        budget.cavitySemi = endospermSemi - meatThicknessStart
        budget.cavityVolume = ellipsoidVolume(budget.cavitySemi)
        budget.endospermVolume = ellipsoidVolume(endospermSemi) - budget.cavityVolume
        budget.haustoriumVolume = budget.cavityVolume * Double(budget.fill)
        budget.waterVolume = budget.cavityVolume - budget.haustoriumVolume
    }

    var plant = Plant(budget: budget, stages: stages, growth: g)
    var prims: [GPUPrim] = []
    prims.reserveCapacity(340)

    // -- soil ---------------------------------------------------------------
    let soilCentreY: Float = soilSurfaceY - soilSemi.y
    prims.append(.spheroid(centre: SIMD3(nutCentre.x, soilCentreY, 0),
                           semi: soilSemi, material: .topsoil))
    prims.append(.spheroid(centre: SIMD3(nutCentre.x, soilCentreY - topsoilDepth, 0),
                           semi: soilSemi, material: .subsoil))
    plant.counts["soil horizon"] = 2

    // -- the nut, outside in ------------------------------------------------
    for m in layerStack {
        let semi: SIMD3<Float> = layerSemi(m, budget: budget, breakage: breakage)
        prims.append(.spheroid(centre: nutCentre, semi: semi, material: m))
    }
    plant.counts["layer"] = layerStack.count

    // -- the three eyes -----------------------------------------------------
    // Sunk into the endocarp's outer surface. Two of them are sealed for good;
    // the third is soft, and once the petiole has gone through it, it has been
    // DISPLACED rather than broken.
    for k in 0..<poreCount {
        let c: SIMD3<Float> = poreCentre(k)
        // Set flush into the shell rather than stuck on it: the plug's short
        // axis runs along the shell's own outward normal.
        let grad: SIMD3<Float> = (c - nutCentre) / (endocarpSemi * endocarpSemi)
        let outward: SIMD3<Float> = simd_normalize(grad)
        var across = SIMD3<Float>(0, 1, 0) - outward * outward.y
        if simd_length(across) < 1e-3 { across = SIMD3(1, 0, 0) - outward * outward.x }
        let a: SIMD3<Float> = simd_normalize(across)
        let b: SIMD3<Float> = simd_normalize(simd_cross(outward, a))
        let m = simd_float3x3(columns: (outward * (endocarpThickness * 0.62),
                                        a * poreRadius, b * poreRadius))
        prims.append(.ellipsoid(centre: c, m: m, material: .testa))
    }
    plant.counts["germination pore"] = poreCount

    // -- the embryo ---------------------------------------------------------
    // A few millimetres, lying in the endosperm directly under the soft eye.
    let poreDir: SIMD3<Float> = simd_normalize(poreCentre(functionalPore) - nutCentre)
    let embryoCentre: SIMD3<Float> = nutCentre + poreDir * 64
    prims.append(.spheroid(centre: embryoCentre, semi: SIMD3(4.2, 3.4, 3.4), material: .embryo))
    plant.counts["embryo"] = 1

    // -- the haustorium -----------------------------------------------------
    // It swells out of the embryo, at the cotyledon's distal end, into the
    // water cavity: a sphere at first, taking the cavity's own shape as it
    // fills it. Its VOLUME is exactly `budget.fill` of the cavity — the scale
    // is solved for rather than eyeballed, which is what makes the conservation
    // test an equality.
    if budget.fill > 1e-5 {
        let ac: Float = budget.cavitySemi.x
        let bc: Float = budget.cavitySemi.y
        let w: Float = smoothstep(0.05, 0.75, budget.fill)
        let rhoA: Float = bc + (ac - bc) * w
        let rhoB: Float = bc
        let ratio: Float = budget.fill * ac / rhoA
        let k: Float = pow(max(ratio, 0), 1.0 / 3.0)
        let semi = SIMD3<Float>(k * rhoA, k * rhoB, k * rhoB)
        // Offset toward the embryo while it is small, shrinking to none as it
        // takes over the cavity. The 0.82 is slack: it keeps the ball strictly
        // inside the cavity wall, which a test checks by sampling the surface.
        let slackX: Float = (ac - semi.x) * 0.72
        let slackY: Float = (bc - semi.y) * 0.72
        let slackZ: Float = (bc - semi.z) * 0.72
        let lean: Float = 1 - w
        let offset = SIMD3<Float>(slackX * lean * poreDir.x,
                                  slackY * lean * poreDir.y,
                                  slackZ * lean * poreDir.z)
        let centre: SIMD3<Float> = nutCentre + offset
        prims.append(.spheroid(centre: centre, semi: semi, material: .haustorium))
        plant.haustoriumCentre = centre
        plant.haustoriumSemi = semi
        plant.counts["haustorium"] = 1
    }

    // -- the cotyledonary petiole -------------------------------------------
    // It extends from the embryo, through the soft eye, out through the husk.
    // The shell does not crack; the eye is displaced. Everything else stays
    // sealed, and the one-pore test walks the whole endocarp to prove it.
    let petioleT: Float = stages.petiole
    if petioleT > 1e-4 {
        let a: SIMD3<Float> = embryoCentre
        let b: SIMD3<Float> = poreCentre(functionalPore)
        let c: SIMD3<Float> = sproutBase
        let steps: Int = segments(6, grown: petioleT, atLeast: 2)
        var previous: SIMD3<Float> = a
        for i in 1...steps {
            let t: Float = Float(i) / Float(steps) * petioleT
            // Quadratic through embryo → pore → husk exit.
            let u: Float = min(t, 1)
            let ab: SIMD3<Float> = a + (b - a) * min(u * 2.2, 1)
            let bc: SIMD3<Float> = b + (c - b) * max(u * 1.45 - 0.45, 0)
            let blend: Float = smoothstep(0.30, 0.62, u)
            let p: SIMD3<Float> = ab + (bc - ab) * blend
            // Narrow where it goes through the eye and swelling only once it is
            // outside: the aperture in the shell is the eye's own size, which
            // is what "displaces the soft eye" means and what the one-pore test
            // measures.
            let swell: Float = smoothstep(0.55, 1.0, u)
            let radius: Float = 5.2 + 13.5 * swell
            prims.append(.capsule(from: previous, to: p, radius: radius, material: .petiole))
            previous = p
        }
        plant.counts["petiole"] = steps
    }

    // -- the shoot ----------------------------------------------------------
    let shootT: Float = stages.shoot
    var tipY: Float = sproutBase.y
    if shootT > 1e-4 {
        let arc: Float = shootArcLength * shootT
        plant.shootArc = arc
        let segs: Int = segments(shootSegments, grown: shootT)
        var previous: SIMD3<Float> = shootPoint(0, length: arc, frame: f)
        for i in 1...segs {
            let u: Float = Float(i) / Float(segs)
            let p: SIMD3<Float> = shootPoint(u, length: arc, frame: f)
            let taper: Float = shootBaseRadius + (shootTipRadius - shootBaseRadius) * u
            prims.append(.capsule(from: previous, to: p, radius: taper, material: .shoot))
            previous = p
            tipY = max(tipY, p.y)
        }
        plant.counts["shoot segment"] = segs

        // -- the first true leaf ---------------------------------------------
        let leafT: Float = stages.leaf
        if leafT > 1e-4 {
            let leaflets: Int = breakage.contains(.pinnate) ? 11 : leafletCountEntire
            plant.leafletCount = leaflets
            let length: Float = bladeLengthFinal * leafT
            plant.bladeLength = length
            let tip: SIMD3<Float> = shootPoint(1, length: arc, frame: f)
            // The blade leaves the spear leaning forward and arches over. It
            // lives in the x–y plane, so a view ray crosses it and never runs
            // down it.
            let lateral: SIMD3<Float> = bladeLateral
            let stations: Int = segments(bladeStations, grown: leafT, atLeast: 4)
            var previous: SIMD3<Float> = tip
            var bladePrims = 0
            for i in 1...stations {
                let s: Float = Float(i) / Float(stations)
                let arch: Float = radians(58) * pow(s, 1.35)
                let ca: Float = cos(arch)
                let sa: Float = sin(arch)
                let step: Float = length / Float(stations)
                let along: SIMD3<Float> = bladeVertical * ca + lateral * (sa * 0.85)
                let p: SIMD3<Float> = previous + simd_normalize(along) * step
                let mid: Float = (Float(i) - 0.5) / Float(stations)
                // The blade widens as it lengthens. Without this it would come
                // out of the spear at full width and zero length — a bar, not a
                // leaf — for the first few frames it exists.
                let opening: Float = pow(leafT, 0.6)
                let half: Float = bladeHalfWidth(mid, leaflets: leaflets) * opening
                if half > 0.4 {
                    let tangent: SIMD3<Float> = simd_normalize(p - previous)
                    let normal: SIMD3<Float> = simd_normalize(simd_cross(tangent, lateral))
                    let centre: SIMD3<Float> = (previous + p) * 0.5
                    let ck: Float = cos(bladeKeelAngle)
                    let sk: Float = sin(bladeKeelAngle)
                    for side in [Float(1), -1] {
                        // Each half of the strip is tilted up a little, which
                        // is the keel a palm blade folds along.
                        let out: SIMD3<Float> = lateral * (side * ck) + normal * sk
                        let axis: SIMD3<Float> = simd_normalize(out)
                        let m = simd_float3x3(columns: (tangent * (step * 1.25),
                                                        normal * bladeThickness,
                                                        axis * (half * 0.55)))
                        let seat: SIMD3<Float> = centre + axis * (half * 0.5)
                        prims.append(.ellipsoid(centre: seat, m: m, material: .blade))
                        bladePrims += 1
                    }
                    prims.append(.capsule(from: previous, to: p,
                                          radius: 2.4 - 1.7 * mid, material: .midrib))
                    bladePrims += 1
                }
                previous = p
                tipY = max(tipY, p.y)
            }
            plant.counts["blade"] = bladePrims
        }
    }
    plant.shootTipY = tipY

    // -- the roots ----------------------------------------------------------
    // A fan of roughly equal roots. The mutation puts a taproot in.
    let rootT: Float = stages.root
    let taproot: Bool = breakage.contains(.taproot)
    if rootT > 1e-4 {
        var rootPrims = 0
        for k in 0..<primaryRoots {
            let full: Float = rootLength(k, taproot: taproot)
            let length: Float = full * rootT
            plant.rootLengths.append(length)
            let (d0, d1) = rootBearing(k, taproot: taproot)
            let origin: SIMD3<Float> = rootOrigin(k)
            let segs: Int = segments(rootSegments, grown: rootT)
            var previous: SIMD3<Float> = origin
            var path: [SIMD3<Float>] = [origin]
            for i in 1...segs {
                let u: Float = Float(i) / Float(segs)
                let dir: SIMD3<Float> = simd_normalize(d0 + (d1 - d0) * u)
                let step: Float = length / Float(segs)
                let wander: Float = 2.6 * sin(Float(k) * 1.7 + u * 5.1)
                let taper: Float = rootBaseRadius + (rootTipRadius - rootBaseRadius) * u
                let raw: SIMD3<Float> = previous + dir * step + SIMD3(0, 0, wander * 0.4)
                let p: SIMD3<Float> = pushOutOfNut(raw, margin: taper + 2)
                prims.append(.capsule(from: previous, to: p, radius: taper, material: .root))
                rootPrims += 1
                previous = p
                path.append(p)
            }
            // Laterals, which is what makes a fibrous system fibrous. They are
            // not counted as roots by the taproot test: that test is about the
            // primary fan, where a dominant root would have to appear.
            let lateralT: Float = smoothstep(0.45, 1.0, rootT)
            if lateralT > 1e-3 {
                for j in 0..<lateralsPerRoot {
                    let at: Int = (path.count * (5 + 3 * j)) / 11
                    if at >= path.count - 1 { continue }
                    let root: SIMD3<Float> = path[at]
                    let ahead: SIMD3<Float> = path[at + 1]
                    let axis: SIMD3<Float> = simd_normalize(ahead - root)
                    let seed: Float = rootJitter(k * 7 + j * 3 + 1)
                    let a: Float = 2 * Float.pi * seed
                    var side = SIMD3<Float>(cos(a), -0.35, sin(a))
                    side = simd_normalize(side - axis * simd_dot(side, axis))
                    let dir: SIMD3<Float> = simd_normalize(axis * 0.45 + side)
                    let len: Float = 46 * lateralT * (0.7 + 0.6 * seed)
                    let ribs: Int = segments(lateralSegments, grown: lateralT, atLeast: 2)
                    var from: SIMD3<Float> = root
                    for i in 1...ribs {
                        let u: Float = Float(i) / Float(ribs)
                        let reach: SIMD3<Float> = root + dir * (len * u)
                        let to: SIMD3<Float> = pushOutOfNut(reach, margin: 3)
                        prims.append(.capsule(from: from, to: to,
                                              radius: 1.5 - 0.9 * u, material: .root))
                        rootPrims += 1
                        from = to
                    }
                }
            }
        }
        plant.counts["root"] = primaryRoots
        plant.counts["root segment"] = rootPrims
    }

    plant.prims = prims
    plant.counts["primitive"] = prims.count
    return plant
}

// MARK: - Colour
//
// Albedo per material. The cut-face shading desaturates and flattens these in
// the kernel; these are the values an uncut, lit surface uses.

func albedoTable(stages: Stages) -> [SIMD3<Float>] {
    var out = [SIMD3<Float>](repeating: SIMD3(repeating: 0.5), count: 16)
    out[Material.topsoil.rawValue] = SIMD3(0.255, 0.190, 0.140)
    out[Material.subsoil.rawValue] = SIMD3(0.392, 0.308, 0.216)
    out[Material.exocarp.rawValue] = SIMD3(0.420, 0.300, 0.185)
    out[Material.mesocarp.rawValue] = SIMD3(0.660, 0.475, 0.280)
    out[Material.endocarp.rawValue] = SIMD3(0.180, 0.118, 0.082)
    out[Material.testa.rawValue] = SIMD3(0.400, 0.235, 0.155)
    out[Material.endosperm.rawValue] = SIMD3(0.955, 0.955, 0.940)
    out[Material.water.rawValue] = SIMD3(0.570, 0.735, 0.760)
    out[Material.embryo.rawValue] = SIMD3(0.870, 0.800, 0.505)
    // Warmer and creamier than the meat on purpose: at this scale the two
    // whites are ten pixels apart and the whole render turns on telling them
    // apart. A real coconut apple is a shade yellower than the endosperm and a
    // great deal spongier, and the cut shading gives it the sponge.
    out[Material.haustorium.rawValue] = SIMD3(0.965, 0.885, 0.700)
    out[Material.petiole.rawValue] = SIMD3(0.830, 0.810, 0.600)
    // The spear leaves the husk the colour of the inside of the nut and greens
    // in the light. That is not decoration: it is the moment the seedling stops
    // living on the endosperm alone.
    let pale = SIMD3<Float>(0.855, 0.820, 0.605)
    let green = SIMD3<Float>(0.300, 0.495, 0.230)
    out[Material.shoot.rawValue] = pale + (green - pale) * stages.green
    out[Material.blade.rawValue] = SIMD3(0.235, 0.480, 0.195)
    out[Material.midrib.rawValue] = SIMD3(0.520, 0.635, 0.320)
    out[Material.root.rawValue] = SIMD3(0.660, 0.580, 0.440)
    return out
}

// MARK: - The constants table
//
// Every line carries an evidence level and a source string, and a test fails if
// a MEASURED line cites nothing, if a MODEL line does not say MODEL, or if a
// DERIVED line does not say DERIVED. Provenance lives here and in the tests,
// which is why there is no evidence bar on the caption: a bar that said the
// same thing on all 144 frames would be decoration.

struct CoconutConstant {
    var name: String
    var value: Double
    var unit: String
    var evidence: Evidence
    var source: String
}

let coconutConstants: [CoconutConstant] = [
    CoconutConstant(name: "haustorium fills the cavity in", value: modelledMonths,
                    unit: "months", evidence: .measured,
                    source: "Comprehensive biochemical profiling of coconut haustorium, 2024"),
    CoconutConstant(name: "haustorium forms from", value: 1, unit: "cotyledon, distal end",
                    evidence: .measured,
                    source: "Fruit Biology of Coconut, Plants 11(23):3293, 2022"),
    CoconutConstant(name: "germination pores", value: Double(poreCount), unit: "pores",
                    evidence: .measured,
                    source: "Fruit Biology of Coconut, Plants 11(23):3293, 2022"),
    CoconutConstant(name: "functional pores", value: 1, unit: "soft eye",
                    evidence: .measured,
                    source: "Fruit Biology of Coconut, Plants 11(23):3293, 2022"),
    CoconutConstant(name: "root system", value: Double(primaryRoots),
                    unit: "adventitious roots, no taproot", evidence: .measured,
                    source: "Germination rate … coconut palm diversity, AoB PLANTS, 2018"),
    CoconutConstant(name: "first true leaf", value: Double(leafletCountEntire),
                    unit: "entire, lance-shaped", evidence: .measured,
                    source: "Fruit Biology of Coconut, Plants 11(23):3293, 2022"),
    CoconutConstant(name: "meat thickness, mature nut", value: Double(meatThicknessStart),
                    unit: "mm", evidence: .measured,
                    source: "Fruit Biology of Coconut, Plants 11(23):3293, 2022"),
    CoconutConstant(name: "shell thickness", value: Double(endocarpThickness), unit: "mm",
                    evidence: .measured,
                    source: "Fruit Biology of Coconut, Plants 11(23):3293, 2022"),
    CoconutConstant(name: "growth rate between stages", value: Double(growFrames),
                    unit: "frames for 4 months", evidence: .model,
                    source: "MODEL — interpolated between staged observations, not a time series"),
    CoconutConstant(name: "circumnutation period", value: Double(nutationFrames),
                    unit: "frames", evidence: .model,
                    source: "MODEL — borrowed character, Darwin 1880; no coconut measurement"),
    CoconutConstant(name: "circumnutation amplitude", value: Double(nutationAmplitude),
                    unit: "mm at the tip", evidence: .model,
                    source: "MODEL — borrowed character, Darwin 1880; no coconut measurement"),
    CoconutConstant(name: "topsoil depth", value: Double(topsoilDepth), unit: "mm",
                    evidence: .model, source: "MODEL — soil profile chosen for legibility"),
    CoconutConstant(name: "soil grain size, topsoil", value: 2.6, unit: "mm",
                    evidence: .model, source: "MODEL — grain size chosen for legibility"),
    CoconutConstant(name: "cavity volume, dormant", value: budgetAtStart.cavityVolume / 1000,
                    unit: "mL", evidence: .derived,
                    source: "DERIVED — ellipsoid volume of the meat's inner surface"),
    CoconutConstant(name: "solid endosperm, dormant", value: budgetAtStart.endospermVolume / 1000,
                    unit: "mL", evidence: .derived,
                    source: "DERIVED — testa ellipsoid less the cavity"),
    CoconutConstant(name: "solid endosperm digested", value: endospermLost / 1000,
                    unit: "mL", evidence: .derived,
                    source: "DERIVED — growth of the cavity over the loop"),
    CoconutConstant(name: "coconut water absorbed", value: waterAbsorbed / 1000,
                    unit: "mL", evidence: .derived,
                    source: "DERIVED — cavity less haustorium, start to end"),
    CoconutConstant(name: "haustorium bulk per mL of meat", value: haustoriumBulkRatio,
                    unit: "×", evidence: .derived,
                    source: "DERIVED — haustorium volume over endosperm volume lost"),
]

// MARK: - The caption
//
// House style, straight from step 13, and no evidence bar. Step 13 dropped its
// because it said the same thing on every frame; the same applies here, and
// harder — every constant in this step already carries its evidence level and
// its source in `coconutConstants`, and a test fails if a measured one cites
// nothing. Provenance belongs where it can be checked, not where it can only be
// looked at.

let coconutTitle = "A coconut sprouting — Cocos nucifera in section"
let coconutFacts = "no taproot · the shell stays whole · the cotyledon never leaves it"

func coconutCaption(frame f: Int, plant: Plant) -> Caption {
    let g: Float = plant.growth
    let s: Stages = plant.stages
    var subtitle: String
    if g <= 0 {
        subtitle = "dormant: a full cavity of water, 13 mm of meat, and an 8 mm embryo under one eye"
    } else if s.shoot < 0.05 {
        subtitle = "the cotyledonary petiole extends through the soft eye — the shell does not crack"
    } else if s.leaf < 0.05 {
        subtitle = "outside, a spear and a fan of equal roots; inside, the cotyledon swells"
    } else if g < 1 {
        subtitle = "the haustorium fills the cavity as the endosperm thins — one budget, two halves"
    } else {
        subtitle = "four months on: the first true leaf is entire, and the cavity is a white ball"
    }
    let months: Double = modelledMonths * Double(g)
    let aside = String(format: "month %.1f of ~%.0f  ·  MODEL timeline", months, modelledMonths)
    return Caption(title: coconutTitle, subtitle: subtitle, facts: coconutFacts, aside: aside)
}
