// The toy sailboat: a single-handed racing dinghy with one mast and one
// triangular sail, hull, mast, boom and sail moulded as one piece.
//
// THE BOAT. Its proportions are the most-built single-handed dinghy class,
// cat-rigged (one sail, no jib): length overall 4.23 m, beam 1.37 m,
// mainsail 7.06 m² (Wikipedia, "Laser (dinghy)", checked 2026-09-28; the
// class's name is not in the picture — the caption calls it a dinghy).
// The sail's sides: luff 5.13 m, leech 5.57 m, foot 2.74 m; the mast two
// tubes, 6160 mm together (lasersailingtips.com, "Intro & Specs", checked;
// its beam, 1.42 m, differs from Wikipedia's — Wikipedia's is used). A flat
// triangle with those sides is 6.99 m²; the other 0.07 m² is the sail's
// curved roach, not drawn. At 1:60 the toy is 70.5 mm long, inside
// Russell's 2–3 inch spec.
//
// THE TOY. One piece, from a mould that splits along the boat's
// CENTREPLANE — the only way a hull with a mast and a sail standing on it
// can come out: each half forms one side of hull, mast and sail, and pulls
// sideways. So:
//   * the parting line runs along the keel, up the stem, over the deck's
//     crown and round the mast, boom and sail's edges;
//   * the sail lies IN the parting plane, so its faces need no draft; it is
//     1.5 mm thick (9 cm at full size), inside a moulder's recommended
//     0.89–3.81 mm (Protolabs, Toy.swift) — a real sail's cloth would be a
//     hundredth of a millimetre at this scale, too thin to fill;
//   * faces the halves slide along need draft (Protolabs: 0.5° least, 2°
//     usual): the deck's camber, the bottom's V and a shallow V in the
//     transom give it;
//   * no cockpit well: a pocket open upwards would lock both halves in (an
//     undercut). The toy's deck is solid, and so is its hull;
//   * no centreboard or rudder: the toy stands on its hull.

import CoreGraphics
import Foundation
import simd

/// The class's numbers, cm (see the header).
let dinghyLengthCM: Float = 423
let dinghyBeamCM: Float = 137
let sailAreaM2: Float = 7.06
let sailLuffCM: Float = 513
let sailLeechCM: Float = 557
let sailFootCM: Float = 274
let mastTubesCM: Float = 616
/// The part of the mast inside its step, below the deck. MODEL, UNVERIFIED
/// (a forum figure, ~0.3 m): it leaves 5.86 m above the deck.
let mastInStepCM: Float = 30
/// The tack's height above the deck. MODEL: sets the boom's height.
let tackAboveDeckCM: Float = 70
/// Where the mast stands, from the bow. MODEL.
let mastFromBowCM: Float = 115
/// The hull's depth, keel to the deck's crown amidships. MODEL (no figure
/// reached).
let dinghyDepthCM: Float = 40

/// The toy's scale. MODEL: chosen to bring the toy inside 2–3 inches. The
/// `wrongSize` mutant makes it 25% too big.
func dinghyScale(_ mutant: Mutant = activeMutant) -> Float { mutant == .wrongSize ? 1.25 / 60.0 : 1.0 / 60.0 }

/// Moulded thicknesses, toy mm. MODEL, inside Protolabs' 0.89–3.81 mm.
let sailThickness: Float = 1.5
let mastDiameterFoot: Float = 2.0
let mastDiameterHead: Float = 1.4
let boomDiameter: Float = 1.6

/// Paints. MODEL: a toy's primary colours — a blue hull, white deck, a red
/// sail, aluminium-grey spars.
enum BoatPaint: Int {
    case hull = 0, deck, sail, spar
}
let boatPaints: [Paint] = [
    paint("blue hull", 34, 70, 146, rough: 0.28),
    paint("white deck", 236, 234, 226, rough: 0.30),
    paint("red sail", 204, 50, 38, rough: 0.40),
    paint("spar grey", 150, 154, 160, rough: 0.22),
]

/// The PVC's own colour in the mass, seen on the inset's cut face. MODEL.
let massColour = SIMD3<Float>(0.80, 0.72, 0.52)

/// The camera: from the port bow, low, as a sailboat is photographed.
/// MODEL. The frame: the tall toy's height sets the distance.
let stillCameraAzimuthDegrees: Float = 60
let stillCameraElevationDegrees: Float = 19
let stillFraming: (length: Float, height: Float, shift: Float, drop: Float) = (0.55, 0.74, 0.07, -0.028)
let tableFalloffMillimetres: Float = 1500

struct DinghyGeometry {
    let s: Float                // cm → toy mm
    let k: Float                // 1 at true scale
    let length: Float
    let halfBeam: Float
    let depth: Float
    /// Deck: crowned `depth` above the keel at x = 0, rising 1.5% forward,
    /// camber radius 50 mm (the deck edge 1.3 mm, 8 cm, below the crown).
    /// Plan: a vesica of half-length 48 widest 12.75 aft of the middle, cut
    /// by a wide transom; 8° flare; 9° deadrise; rocker radius 300. MODEL.
    let deckSlope: Float = 0.015
    var camberRadius: Float { 50 * k }
    var planHalfLength: Float { 48 * k }
    var planCentre: Float { -12.75 * k }
    var transom: Float { -35.05 * k }
    var keelLowX: Float { -6 * k }
    var keelRadius: Float { 300 * k }
    var gunwale: Float { depth - 1.32 * k }
    /// The rounded deck edge takes 0.25 mm off each side of the plan's
    /// widest line; the plan is that much wider so the drawn beam is the
    /// class's (measured off the drawn toy by a test).
    var beamAllowance: Float { 0.255 * k }

    init(scale: Float) {
        s = scale * 10
        k = scale * 60
        length = dinghyLengthCM * s
        halfBeam = dinghyBeamCM * s / 2
        depth = dinghyDepthCM * s
    }

    var bow: Float { planCentre + planHalfLength }
    var mastX: Float { bow - mastFromBowCM * s }
    func deckY(_ x: Float) -> Float { depth + deckSlope * x }

    var spec: HullSpec {
        let rad: Float = .pi / 180
        return HullSpec(planHalfLength: planHalfLength, beam: halfBeam + beamAllowance, planCentre: planCentre, transom: transom,
                        transomVee: draftMostDegrees * rad, planHeight: gunwale, flare: 8 * rad, keelLowX: keelLowX,
                        keelRadius: keelRadius, deadrise: 9 * rad, bilge: 3.5, decked: true, deckCrown: depth,
                        deckSlope: deckSlope, deckCamberRadius: camberRadius, topRound: 1.2, cap: depth + 3)
    }

    /// The sail, in its plane (x forward, y up from the deck at the mast):
    /// tack, head and clew from the three sides. The luff runs up the mast's
    /// after edge; the angle at the tack comes from the law of cosines.
    var sailCorners: (tack: SIMD2<Float>, head: SIMD2<Float>, clew: SIMD2<Float>) {
        let luff: Float = sailLuffCM * s
        let foot: Float = sailFootCM * s
        let leech: Float = sailLeechCM * s
        let cosT: Float = (luff * luff + foot * foot - leech * leech) / (2 * luff * foot)
        let sinT: Float = (1 - cosT * cosT).squareRoot()
        let tack = SIMD2<Float>(-0.3 * k, tackAboveDeckCM * s)
        let head = SIMD2<Float>(tack.x, tack.y + luff)
        let clew = SIMD2<Float>(tack.x - foot * sinT, tack.y + foot * cosT)
        return (tack, head, clew)
    }
}

func boatDesign(_ mutant: Mutant = activeMutant) -> BoatDesign {
    let g = DinghyGeometry(scale: dinghyScale(mutant))
    let hull = Segment(name: "hull", part: .hull,
                       prims: [g.spec.prim(paint: BoatPaint.hull.rawValue, inside: BoatPaint.deck.rawValue, seam: false)],
                       blend: 0, seam: true)

    // The rig: mast, boom and sail, stood on the deck at the mast.
    let foot = SIMD3<Float>(g.mastX, g.deckY(g.mastX) - 0.4, 0)
    let above: Float = (mastTubesCM - mastInStepCM) * g.s
    let spar: Int = BoatPaint.spar.rawValue
    var rig: [Prim] = []
    rig.append(Prim.cone(foot, mastDiameterFoot / 2, foot + SIMD3<Float>(0, above + 0.4 - mastDiameterHead / 2, 0),
                         mastDiameterHead / 2, paint: spar).tagged(.mast))
    let c = g.sailCorners
    let origin = SIMD3<Float>(g.mastX, g.deckY(g.mastX), 0)
    let boomR: Float = boomDiameter / 2
    let tackW: SIMD3<Float> = origin + SIMD3<Float>(0.6, c.tack.y - boomR, 0)
    let clewW: SIMD3<Float> = origin + SIMD3<Float>(c.clew.x - 0.4, c.clew.y - boomR, 0)
    rig.append(Prim.cone(tackW, boomR, clewW, boomR * 0.9, paint: spar).tagged(.boom))
    if mutant != .noSail {
        let sail = ProfileSpec(origin: origin, xAxis: SIMD3<Float>(1, 0, 0), normal: SIMD3<Float>(0, 0, 1),
                               a: c.tack, b: c.clew, c: c.head, halfThickness: sailThickness / 2, edge: 0.3)
        rig.append(sail.prim(paint: BoatPaint.sail.rawValue, seam: false).tagged(.sail))
    }
    let rigSeg = Segment(name: "rig", part: .rig, prims: rig, blend: 0.5, seam: true)
    return BoatDesign(name: "sailboat", paints: boatPaints, segments: [hull, rigSeg], frames: [Frame.identity, Frame.identity])
}

// MARK: - the inset

/// The inset's cut: square across the boat 14 mm aft of the mast, through
/// the hull, the boom and the sail, seen from astern (right = starboard).
/// How many millimetres the inset's height spans:
let insetFieldMillimetres: Float = 30

func cutStation() -> Float { DinghyGeometry(scale: dinghyScale()).mastX - 14 }

/// A star-shaped section's outline on the cut: for each angle round a
/// centre on the plane, where the surface is, found by bisection.
func sectionLoop(_ toy: PosedToy, cut: Cut, centre: SIMD2<Float>, radius: Float, steps: Int) -> [SIMD3<Float>] {
    var out: [SIMD3<Float>] = []
    for i in 0...steps {
        let a: Float = Float(i) / Float(steps) * 2 * .pi
        var lo: Float = 0
        var hi: Float = radius
        for _ in 0..<40 {
            let m: Float = (lo + hi) / 2
            let u: Float = centre.x + cos(a) * m
            let v: Float = centre.y + sin(a) * m
            let p: SIMD3<Float> = cut.centre + cut.right * u + cut.down * v
            if toy.sdf(p).d <= 0 { lo = m } else { hi = m }
        }
        let u: Float = centre.x + cos(a) * lo
        let v: Float = centre.y + sin(a) * lo
        out.append(cut.centre + cut.right * u + cut.down * v)
    }
    return out
}

func boatCut(_ toy: PosedToy) -> Cut {
    let x: Float = cutStation()
    var c = Cut(centre: SIMD3<Float>(x, 12.5, 0), right: SIMD3<Float>(0, 0, 1), down: SIMD3<Float>(0, -1, 0))
    // The hull's section, round its middle (in the cut's own coordinates:
    // right, and down from the centre).
    c.marks = sectionLoop(toy, cut: c, centre: SIMD2<Float>(0, 12.5 - 3.2), radius: 13, steps: 96)
    return c
}

/// How thick the drawn toy is, along z, at a point of the centreplane.
func thicknessAcross(_ toy: PosedToy, x: Float, y: Float) -> Float {
    var z: Float = 0
    while z < 5 && toy.sdf(SIMD3<Float>(x, y, z)).d <= 0 { z += 0.0005 }
    return 2 * z
}

func drawInsetNotes(_ ctx: CGContext, setup s: StillSetup, rect r: CGRect, k: CGFloat, height h: CGFloat) {
    let cut: Cut = s.cut
    let hI: Int = Int(h)
    let g = DinghyGeometry(scale: dinghyScale())
    func px(_ y: Float, _ z: Float) -> CGPoint { insetPixel(SIMD3<Float>(cut.centre.x, y, z), cut: cut, height: hI) }
    ctx.setStrokeColor(ink)
    ctx.setLineWidth(1.4 * k)

    // The hull.
    let hullP: CGPoint = px(3.2, 0)
    let t1: String = "solid PVC: no hollow inside"
    put(t1, ctx, x: hullP.x - textWidth(t1, size: 15 * k, bold: true) / 2, top: hullP.y - 9 * k, size: 15 * k, bold: true, height: h)

    // The sail: its thickness, measured off the drawn toy.
    let sailY: Float = 23
    let th: Float = thicknessAcross(s.toy, x: cut.centre.x, y: sailY)
    let sp: CGPoint = px(sailY, 0.8)
    let sx: CGFloat = sp.x + 40 * k
    line(ctx, CGPoint(x: sx - 4 * k, y: sp.y + 8 * k), CGPoint(x: sp.x + 3 * k, y: sp.y), height: h)
    put(String(format: "sail %.1f mm thick", th), ctx, x: sx, top: sp.y - 4 * k, size: 15 * k, bold: true, height: h)
    put("9 cm at full size: thick enough for the", ctx, x: sx, top: sp.y + 16 * k, size: 12 * k, height: h)
    put("plastic to fill it (moulders: 0.9–3.8 mm)", ctx, x: sx, top: sp.y + 31 * k, size: 12 * k, height: h)

    // The boom.
    let bp: CGPoint = px(g.deckY(cut.centre.x) + tackAboveDeckCM * g.s - boomDiameter / 2, -0.9)
    put("boom", ctx, x: bp.x - 70 * k, top: bp.y - 8 * k, size: 13 * k, bold: true, height: h)
    line(ctx, CGPoint(x: bp.x - 30 * k, y: bp.y), CGPoint(x: bp.x - 3 * k, y: bp.y), height: h)

    // The parting plane and the draft.
    let crown: CGPoint = px(g.deckY(cut.centre.x), 0)
    let keel: CGPoint = px(0, 0)
    ctx.setStrokeColor(softInk)
    ctx.setLineDash(phase: 0, lengths: [5 * k, 5 * k])
    line(ctx, CGPoint(x: keel.x, y: keel.y + 20 * k), CGPoint(x: crown.x, y: crown.y - 24 * k), height: h)
    ctx.setLineDash(phase: 0, lengths: [])
    let note: CGPoint = px(14.2, 0)
    put("parting plane (dashed): each mould half", ctx, x: r.minX + 16 * k, top: note.y, size: 13 * k, bold: true, height: h)
    put("forms one side and pulls sideways. The deck's", ctx, x: r.minX + 16 * k, top: note.y + 18 * k, size: 12 * k, height: h)
    put("camber and the bottom's V lean away from it:", ctx, x: r.minX + 16 * k, top: note.y + 33 * k, size: 12 * k, height: h)
    put("the draft that lets the halves go", ctx, x: r.minX + 16 * k, top: note.y + 48 * k, size: 12 * k, height: h)

    let pxPerMM: Float = 1 / insetMillimetresPerPixel(height: hI)
    let paintPx: Float = paintMicrometres / 1000 * pxPerMM
    let um: Int = Int(paintMicrometres)
    put(String(format: "paint, %d µm (model): %.1f px at this scale", um, paintPx), ctx, x: r.minX + 16 * k, top: r.maxY - 30 * k,
        size: 12 * k, color: softInk, height: h)
}

// MARK: - the caption

func boatCaption(lengthMM: Float) -> Caption {
    let g = DinghyGeometry(scale: dinghyScale())
    return Caption(
        title: "A plastic toy sailboat",
        lines: [
            "One piece of PVC, injection-moulded and painted; 2–3 inches long (this one \(Int(lengthMM.rounded())) mm): a single-handed dinghy at 1:60.",
            "Its gloss is PVC's refractive index: n = 1.545 (Zhang et al. 2020) mirrors 4.6% of the light at normal incidence, more at grazing.",
            "One mast and one triangular sail, its sides the class's own (luff 5.13, leech 5.57, foot 2.74 m); a solid deck, no centreboard.",
            "The mould splits down the centreplane, so the sail lies in it, 1.5 mm thick. Unbranded; paint colours are a model.",
        ],
        cutName: "hull and sail, looking forward",
        seamStart: SIMD3<Float>(g.bow - 12, g.deckY(g.bow - 12) + 4, 0),
        seamDirection: SIMD3<Float>(0, -1, 0),
        seamLabel: CGPoint(x: -230, y: -90))
}
