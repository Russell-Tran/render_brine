// The toy voyaging canoe: a toy depiction of the traditional Hawaiian
// double-hulled voyaging canoe, the waʻa kaulua, in the proportions of the
// Polynesian Voyaging Society's replica Hōkūleʻa. It is a toy of a
// traditional design, not a model of any particular canoe.
//
// THE CANOE, from the sources that could be read:
//   [PVS-diagram] Polynesian Voyaging Society, "Hōkūleʻa" diagram (graphic by
//     Dave Swann / Star-Advertiser, "Source: Polynesian Voyaging Society";
//     hokulea.com, linked from its "Voyaging Canoes" page, read 2026-09-28):
//     "Length: 64ft, 9in", "Beam: 19ft, 8in", "Mast Height: 31ft, 2in",
//     "Spar Height: 41ft, 5in", "Draft: 2 Feet, 6 Inches"; its parts named:
//     kia (masts), ʻopeʻa (spars), paepae (booms), manu ihu and manu hope
//     (bow and stern end pieces), ʻiako (crossbeams), pola (deck), hoe uli
//     (steering paddle). Its side and end views are what the proportions
//     below are read off (at 15.1 and 30.5 px per foot; ±5%, MODEL where so
//     marked).
//   [PVS-voyaging] hokulea.com, "Voyaging Canoes": Hōkūleʻa is a "deep sea
//     voyaging canoe built in the tradition of ancient Hawaiian wa'a kaulua
//     (double-hulled voyaging canoe)", designed by Herb Kawainui Kāne,
//     launched March 8, 1975.
//   [PVS-plan] PVS, "Hokule'a Plans" (pvs.kcc.hawaii.edu/pics/hokuplan.gif,
//     read as the Wayback Machine archived it in 2009): the 1975 lines —
//     62'4" overall, beam 17'6" (the figures Wikipedia repeats); its plan
//     view shows the eight ʻiako, counted here, and the pola between the
//     hulls from the first ʻiako to the last. The later diagram's figures
//     are used for size: the canoe has been rebuilt since.
//   Wikipedia, "Hōkūleʻa" (read): "a performance-accurate waʻa kaulua", with
//     "crabclaw sails".
//
// THE TOY is at 1:300 — 65.8 mm, inside Russell's 2–3 inch spec — and made
// in TWO mouldings, because no one mould could release both parts:
//   * the PLATFORM — two hulls, the ʻiako, the pola, the manu — from a mould
//     that opens up and down, split at the gunwales. The hulls' V is their
//     draft; the manu thin a little towards their tips (0.5°, Protolabs'
//     least for vertical faces). The real hulls swell outward below the
//     gunwale (the diagram's end view): that tumblehome would lock the
//     lower half in, so the toy's sides run straight up;
//   * each RIG — kia, ʻopeʻa, paepae and the crab-claw sail between them —
//     moulded flat in a mould split along the sail's plane, as step 74's
//     sail is, and plugged into the deck. A sail standing over the deck
//     cannot come out of the platform's up-and-down mould: the air under the
//     paepae would trap the steel.

import CoreGraphics
import Foundation
import simd

/// PVS's diagram, feet and inches, as metres.
let canoeLengthM: Float = (64 * 12 + 9) * 0.0254
let canoeBeamM: Float = (19 * 12 + 8) * 0.0254
let mastHeightM: Float = (31 * 12 + 2) * 0.0254
let sparHeightM: Float = (41 * 12 + 5) * 0.0254
let canoeDraftM: Float = 30 * 0.0254

/// Read off the diagram's views (MODEL at ±5%): the masts 13.4 and 33.9 ft
/// from the bow; each hull 4.9 ft wide and 6.6 ft deep, keel to gunwale,
/// their centres 14.2 ft apart; the manu rising 4.0 ft (bow) and 6.0 ft
/// (stern) above the gunwale; each paepae's end 18.5 ft aft of and 25.8 ft
/// above the tack; each sail's concave edge sagging 2.8 ft. Feet.
let foremastFromBowFt: Float = 13.4
let aftmastFromBowFt: Float = 33.9
let hullWidthFt: Float = 4.9
let hullDepthFt: Float = 6.6
let hullSpacingFt: Float = 14.2
let manuIhuRiseFt: Float = 4.0
let manuHopeRiseFt: Float = 6.0
let boomReachAftFt: Float = 18.5
let boomReachUpFt: Float = 25.8
let sparAftFt: Float = 0.66
let leechSagFt: Float = 2.8
/// The eight ʻiako, as fractions of the length from the bow, read off
/// PVS-plan's plan view. MODEL (±0.01).
let iakoFromBow: [Float] = [0.201, 0.302, 0.402, 0.503, 0.598, 0.698, 0.799, 0.894]

let metresPerFoot: Float = 0.3048

/// The toy's scale. MODEL: chosen to bring the toy inside 2–3 inches. The
/// `wrongSize` mutant makes it 25% too big.
func canoeScale(_ mutant: Mutant = activeMutant) -> Float { mutant == .wrongSize ? 1.25 / 300.0 : 1.0 / 300.0 }

/// Moulded sizes, toy mm. MODEL, inside Protolabs' 0.89–3.81 mm (Toy.swift):
/// ʻiako 1.0 mm round, pola 1.0 mm thick, sails 1.2 mm, manu 1.6 mm, kia
/// 1.3 mm tapering to 1.0, ʻopeʻa and paepae 0.9 mm.
let iakoRadius: Float = 0.5
let polaThickness: Float = 1.0
let sailThickness75: Float = 1.2
let manuThickness: Float = 1.6

/// Paints. The hulls as PVS's diagram draws them — red-brown below the
/// waterline, cream above — with the line at the published 2 ft 6 in draft;
/// the pola, ʻiako and spars wood; the sails canvas (the diagram's tan).
/// Colours MODEL.
enum BoatPaint: Int {
    case topside = 0, bottom, wood, sail, spar
}
let boatPaints: [Paint] = [
    paint("cream topsides", 232, 224, 204, rough: 0.30),
    paint("red-brown bottom", 150, 62, 44, rough: 0.30),
    paint("deck wood", 176, 134, 86, rough: 0.34),
    paint("canvas sail", 214, 184, 136, rough: 0.42),
    paint("spar wood", 120, 82, 50, rough: 0.28),
]

/// The PVC's own colour in the mass, seen on the inset's cut face. MODEL.
let massColour = SIMD3<Float>(0.80, 0.72, 0.52)

/// The camera: from the port bow, a little above. MODEL.
let stillCameraAzimuthDegrees: Float = 62
let stillCameraElevationDegrees: Float = 23
let stillFraming: (length: Float, height: Float, shift: Float, drop: Float) = (0.50, 0.70, 0.11, -0.01)
let tableFalloffMillimetres: Float = 1200

struct CanoeGeometry {
    let s: Float                // metres → toy mm
    let length: Float
    let halfBeam: Float
    let hullHalfWidth: Float
    let hullDepth: Float
    let hullOffset: Float       // each hull's centre from the canoe's centreplane
    var bow: Float { length / 2 }

    init(scale: Float) {
        s = scale * 1000
        length = canoeLengthM * s
        halfBeam = canoeBeamM * s / 2
        hullHalfWidth = hullWidthFt * metresPerFoot * s / 2
        hullDepth = hullDepthFt * metresPerFoot * s
        hullOffset = hullSpacingFt * metresPerFoot * s / 2
    }

    func ft(_ f: Float) -> Float { f * metresPerFoot * s }
    func fromBow(_ f: Float) -> Float { bow - f * length }
    var iakoY: Float { hullDepth + iakoRadius - 0.4 }
    /// The pola's underside at the ʻiako's middles: their upper halves are
    /// in it, so no crevice between a round beam and a flat deck traps the
    /// mould (a test walks vertical lines through the platform).
    var polaTop: Float { iakoY + polaThickness }
    var waterline: Float { canoeDraftM * s }

    /// One hull: the plan a vesica, its sides flared 4° (their draft above
    /// the V), a V bottom of 50° deadrise rounded by a smooth-max, the keel
    /// rising to the ends (rocker radius 150 mm), the gunwale nearly level
    /// (sheer radius 1200 mm). Shapes MODEL on the diagram's views.
    func hull(side: Float) -> HullSpec {
        let rad: Float = .pi / 180
        let k: Float = s / (1000.0 / 300.0)
        return HullSpec(planHalfLength: bow - 1.1 * k, beam: hullHalfWidth, planCentre: 0, planHeight: hullDepth,
                        flare: 4 * rad, keelLowX: 0, keelRadius: 150 * k, deadrise: 50 * rad, bilge: 1.2,
                        sheerLowX: 0, sheerLowY: hullDepth, sheerRadius: 1200 * k, topRound: 0.2, cap: hullDepth + 3,
                        seamDrop: 0.3, offsetZ: side * hullOffset)
    }

    var masts: [Float] { [bow - ft(foremastFromBowFt), bow - ft(aftmastFromBowFt)] }
}

/// A manu: a curved end piece standing on a hull's end, in the hull's
/// centreplane, its tip `rise` above the gunwale and a little beyond the
/// hull's end; its inboard edge hollowed by a circle. `dir` is +1 at the
/// bow, −1 at the stern.
func manu(_ g: CanoeGeometry, side: Float, dir: Float, rise: Float, tipX: Float) -> Prim {
    let z: Float = side * g.hullOffset
    let origin = SIMD3<Float>(0, g.hullDepth, z)
    let baseIn = SIMD2<Float>(dir * (tipX - 6.0), -0.6)
    let baseOut = SIMD2<Float>(dir * (tipX - 1.2), -0.6)
    let tip = SIMD2<Float>(dir * tipX, rise)
    // The hollow of the inboard edge: a circle through near the inner base
    // and the tip, sagging 1.1 mm.
    let mid: SIMD2<Float> = (baseIn + tip) / 2
    let chord: Float = simd_distance(baseIn, tip)
    let sag: Float = 1.1
    let r: Float = (chord * chord / 4 + sag * sag) / (2 * sag)
    let along: SIMD2<Float> = simd_normalize(tip - baseIn)
    var normal2 = SIMD2<Float>(-along.y, along.x)
    if simd_dot(normal2, SIMD2<Float>(-dir, 1)) < 0 { normal2 = -normal2 }
    let centre: SIMD2<Float> = mid + normal2 * (r - sag)
    let spec = ProfileSpec(origin: origin, xAxis: SIMD3<Float>(1, 0, 0), normal: SIMD3<Float>(0, 0, 1),
                           a: baseIn, b: baseOut, c: tip, cut: SIMD3<Float>(centre.x, centre.y, r),
                           halfThickness: manuThickness / 2, taper: tan(draftLeastDegrees * .pi / 180), edge: 0.35)
    return spec.prim(paint: BoatPaint.topside.rawValue, seam: false).tagged(.manu)
}

/// One rig, in the centreplane, its tack at the deck just aft of its kia:
/// kia, ʻopeʻa (the spar, nearly upright), paepae (the boom, reaching aft
/// and up) and the crab-claw sail between them, its edge between the two
/// ends hollowed.
func rig(_ g: CanoeGeometry, mastX: Float, mutant: Mutant) -> [Prim] {
    let deck: Float = g.polaTop
    let spar: Int = BoatPaint.spar.rawValue
    var out: [Prim] = []
    out.append(Prim.cone(SIMD3<Float>(mastX, deck - 0.8, 0), 0.65, SIMD3<Float>(mastX, deck + mastHeightM * g.s - 0.5, 0), 0.5,
                         paint: spar).tagged(.mast))
    let tack = SIMD2<Float>(-0.45, 0.3)
    let sparTip = SIMD2<Float>(tack.x - g.ft(sparAftFt), sparHeightM * g.s - 0.4)
    let boomTip = SIMD2<Float>(tack.x - g.ft(boomReachAftFt), tack.y + g.ft(boomReachUpFt))
    let o = SIMD3<Float>(mastX, deck, 0)
    func w(_ p: SIMD2<Float>) -> SIMD3<Float> { o + SIMD3<Float>(p.x, p.y, 0) }
    out.append(Prim.cone(w(tack), 0.5, w(sparTip), 0.4, paint: spar).tagged(.spar))
    out.append(Prim.cone(w(tack), 0.5, w(boomTip), 0.4, paint: spar).tagged(.boom))
    if mutant != .noSail {
        let mid: SIMD2<Float> = (sparTip + boomTip) / 2
        let chord: Float = simd_distance(sparTip, boomTip)
        let sag: Float = g.ft(leechSagFt)
        let r: Float = (chord * chord / 4 + sag * sag) / (2 * sag)
        let along: SIMD2<Float> = simd_normalize(boomTip - sparTip)
        var away = SIMD2<Float>(-along.y, along.x)
        if simd_dot(away, mid - tack) < 0 { away = -away }
        let centre: SIMD2<Float> = mid + away * (r - sag)
        let sail = ProfileSpec(origin: o, xAxis: SIMD3<Float>(1, 0, 0), normal: SIMD3<Float>(0, 0, 1),
                               a: tack, b: boomTip, c: sparTip, cut: SIMD3<Float>(centre.x, centre.y, r),
                               halfThickness: sailThickness75 / 2, edge: 0.3)
        out.append(sail.prim(paint: BoatPaint.sail.rawValue, seam: false).tagged(.sail))
    }
    return out
}

func boatDesign(_ mutant: Mutant = activeMutant) -> BoatDesign {
    let g = CanoeGeometry(scale: canoeScale(mutant))
    let k: Float = g.s / (1000.0 / 300.0)
    var platform: [Prim] = []
    let sides: [Float] = mutant == .singleHull ? [1] : [-1, 1]
    for side in sides {
        platform.append(g.hull(side: side).prim(paint: BoatPaint.topside.rawValue, inside: -1, seam: true))
        platform.append(manu(g, side: side, dir: 1, rise: g.ft(manuIhuRiseFt), tipX: g.bow))
        platform.append(manu(g, side: side, dir: -1, rise: g.ft(manuHopeRiseFt), tipX: g.bow))
    }
    // The eight ʻiako, across both hulls and a little beyond.
    let wood: Int = BoatPaint.wood.rawValue
    for f in iakoFromBow {
        let x: Float = g.fromBow(f)
        platform.append(Prim.cone(SIMD3<Float>(x, g.iakoY, -g.halfBeam + iakoRadius), iakoRadius,
                                  SIMD3<Float>(x, g.iakoY, g.halfBeam - iakoRadius), iakoRadius, paint: wood).tagged(.crossbeam))
    }
    // The pola: the deck between the hulls, first ʻiako to last, on them.
    let x0: Float = g.fromBow(iakoFromBow[0]) + 0.8 * k
    let x1: Float = g.fromBow(iakoFromBow[7]) - 0.8 * k
    let inner: Float = g.hullOffset - g.hullHalfWidth - 0.1
    let polaY: Float = g.polaTop - polaThickness / 2
    platform.append(Prim.box(SIMD3<Float>((x0 + x1) / 2, polaY, 0), half: SIMD3<Float>((x0 - x1) / 2, polaThickness / 2, inner),
                             edge: 0.3, paint: wood).tagged(.deck))
    // The hoe uli, the steering paddle: from the pola's after end, down and
    // aft past the sterns, its blade upright. MODEL placing.
    let pStart = SIMD3<Float>(x1 + 1.5, polaY, 0)
    let pEnd = SIMD3<Float>(-g.bow - 2.4 * k, 3.2 * k, 0)
    platform.append(Prim.cone(pStart, 0.45, pEnd, 0.4, paint: BoatPaint.spar.rawValue).tagged(.sweep))
    let dir: SIMD3<Float> = simd_normalize(pEnd - pStart)
    let bladeC: SIMD3<Float> = pEnd + dir * 1.6
    platform.append(Prim.box(bladeC, half: SIMD3<Float>(2.0, 0.85, 0.45), edge: 0.4, xAxis: dir, zAxis: SIMD3<Float>(0, 0, 1),
                             paint: BoatPaint.spar.rawValue).tagged(.sweep))
    let waterline = Patch(c: SIMD3<Float>(0, g.waterline, 0), r: 0, paint: BoatPaint.bottom.rawValue, kind: 1)
    let plat = Segment(name: "platform", part: .hull, prims: platform, blend: 0.45, seam: false, patches: [waterline])

    var segs: [Segment] = [plat]
    var frames: [Frame] = [Frame.identity]
    for (i, m) in g.masts.enumerated() {
        segs.append(Segment(name: i == 0 ? "fore rig" : "after rig", part: .rig, prims: rig(g, mastX: m, mutant: mutant),
                            blend: 0.4, seam: true))
        frames.append(Frame.identity)
    }
    return BoatDesign(name: "voyaging canoe", paints: boatPaints, segments: segs, frames: frames)
}

// MARK: - the inset

/// The inset's cut: square across the canoe along the fourth ʻiako, through
/// both hulls, the ʻiako and the pola, seen from astern (right = starboard).
let insetFieldMillimetres: Float = 25

func cutStation() -> Float { CanoeGeometry(scale: canoeScale()).fromBow(iakoFromBow[3]) }

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
    let g = CanoeGeometry(scale: canoeScale())
    let x: Float = cutStation()
    var c = Cut(centre: SIMD3<Float>(x + 0.12, 5.0, 0), right: SIMD3<Float>(0, 0, 1), down: SIMD3<Float>(0, -1, 0))
    // The outline over the top: along the ʻiako and pola from the port end
    // to the starboard end, each hull's outline below.
    var marks: [SIMD3<Float>] = []
    for side in [Float(-1), 1] {
        let loop = sectionLoop(toy, cut: c, centre: SIMD2<Float>(side * g.hullOffset, 5.0 - g.hullDepth * 0.5), radius: 5.0, steps: 48)
        marks += loop.filter { $0.y < g.hullDepth - 0.05 }
    }
    c.marks = marks
    return c
}

func drawInsetNotes(_ ctx: CGContext, setup s: StillSetup, rect r: CGRect, k: CGFloat, height h: CGFloat) {
    let cut: Cut = s.cut
    let hI: Int = Int(h)
    let g = CanoeGeometry(scale: canoeScale())
    func px(_ y: Float, _ z: Float) -> CGPoint { insetPixel(SIMD3<Float>(cut.centre.x, y, z), cut: cut, height: hI) }
    ctx.setStrokeColor(ink)
    ctx.setLineWidth(1.4 * k)

    put("two hulls, solid PVC,", ctx, x: r.minX + 16 * k, top: r.minY + 14 * k, size: 15 * k, bold: true, height: h)
    put("joined by the ʻiako (crossbeams) and the pola (deck)", ctx, x: r.minX + 16 * k, top: r.minY + 34 * k, size: 13 * k, height: h)

    // The ʻiako and pola, labelled from above.
    let beam: CGPoint = px(g.iakoY, -g.halfBeam + 0.8)
    line(ctx, CGPoint(x: beam.x + 10 * k, y: beam.y - 58 * k), CGPoint(x: beam.x + 2 * k, y: beam.y - 4 * k), height: h)
    put("ʻiako", ctx, x: beam.x - 4 * k, top: beam.y - 80 * k, size: 13 * k, bold: true, height: h)
    let deck: CGPoint = px(g.polaTop, 1.5)
    line(ctx, CGPoint(x: deck.x + 20 * k, y: deck.y - 50 * k), CGPoint(x: deck.x + 4 * k, y: deck.y - 3 * k), height: h)
    put("pola", ctx, x: deck.x + 14 * k, top: deck.y - 72 * k, size: 13 * k, bold: true, height: h)

    // The draft: the V of each hull, and the straight sides.
    let keel: CGPoint = px(0, g.hullOffset)
    let note = CGPoint(x: r.minX + 16 * k, y: keel.y + 40 * k)
    put("draft: each hull's V and sides lean away", ctx, x: note.x, top: note.y, size: 13 * k, bold: true, height: h)
    put("from the lower mould half, so it lets go. The real hulls", ctx, x: note.x, top: note.y + 18 * k, size: 12 * k, height: h)
    put("swell outward below the gunwale; that would lock the", ctx, x: note.x, top: note.y + 33 * k, size: 12 * k, height: h)
    put("mould, so the toy's sides run straight. The rigs are", ctx, x: note.x, top: note.y + 48 * k, size: 12 * k, height: h)
    put("moulded apart, flat, and plugged into the pola.", ctx, x: note.x, top: note.y + 63 * k, size: 12 * k, height: h)

    // The parting line at the port hull's gunwale, outside.
    var zOut: Float = -g.hullOffset - g.hullHalfWidth - 1
    let yS: Float = g.hullDepth - 0.3
    while s.toy.sdf(SIMD3<Float>(cut.centre.x, yS, zOut)).d > 0 { zOut += 0.002 }
    let sp: CGPoint = px(yS, zOut)
    line(ctx, CGPoint(x: sp.x + 22 * k, y: sp.y + 44 * k), CGPoint(x: sp.x + 1 * k, y: sp.y + 3 * k), height: h)
    put("parting line", ctx, x: sp.x + 10 * k, top: sp.y + 48 * k, size: 13 * k, bold: true, height: h)

    let pxPerMM: Float = 1 / insetMillimetresPerPixel(height: hI)
    let paintPx: Float = paintMicrometres / 1000 * pxPerMM
    let um: Int = Int(paintMicrometres)
    put(String(format: "paint, %d µm (model): %.1f px at this scale", um, paintPx), ctx, x: r.minX + 16 * k, top: r.maxY - 30 * k,
        size: 12 * k, color: softInk, height: h)
}

// MARK: - the caption

func boatCaption(lengthMM: Float) -> Caption {
    let g = CanoeGeometry(scale: canoeScale())
    return Caption(
        title: "A plastic toy voyaging canoe (waʻa kaulua)",
        lines: [
            "A toy depiction of the traditional Hawaiian double-hulled voyaging canoe, in the proportions of the Polynesian Voyaging Society's Hōkūleʻa.",
            "PVC, injection-moulded and painted; 2–3 inches long (this one \(Int(lengthMM.rounded())) mm, at 1:300). Gloss from PVC's n = 1.545 (Zhang et al. 2020): F0 4.6%.",
            "Two hulls with their manu, eight ʻiako and the pola; two kia, each with a crab-claw sail between its ʻopeʻa and paepae; the hoe uli astern.",
            "Hull colours as PVS's diagram draws them, the line at the published 2 ft 6 in draft. Unbranded; colours and small shapes are a model.",
        ],
        cutName: "hulls along an ʻiako, looking forward",
        seamStart: SIMD3<Float>(g.bow - 20, g.hullDepth - 0.3, -g.halfBeam - 3),
        seamDirection: SIMD3<Float>(0, 0, 1),
        seamLabel: CGPoint(x: -120, y: 110))
}
