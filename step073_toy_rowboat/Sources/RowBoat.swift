// The toy rowing boat: a small traditional rowboat, as a good toy makes one.
//
// THE BOAT. The Whitehall is the classic American pulling boat: "a nearly
// straight stem, and slight flare to the bow, rounded sides, with a keel
// running the entire length of the bottom and a distinctive wine glass
// transom with a full skeg", lapstrake-built, 14 to 22 ft long, rowed from
// fixed thwarts (Wikipedia, "Whitehall rowboat", checked 2026-09-28).
// Rowing boats are "fitted with thwarts - seats that go from one side of the
// hull to the other", and a rowlock holds each oar at the gunwale
// (Wikipedia, "Rowing boat", checked). Its proportions here are one
// builder's current traditional fixed-seat 14 ft Whitehall-type rowboat:
// length 14 ft 2 in (4.3 m), beam 51 in (130 cm), depth 18 in (46 cm) (the
// maker's spec table, checked 2026-09-28; the maker is not named in the
// picture). The toy keeps those three at 1:65 — 66.4 × 20.0 × 7.1 mm,
// inside Russell's 2–3 inch spec.
//
// THE TOY. One piece, moulded in a mould that opens UP AND DOWN: an open
// boat can only come out of a mould that way, because the hollow inside it
// has to be formed by a core that lifts straight out. So:
//   * the parting line runs round the gunwale, the widest line of the hull;
//   * every face the mould slides along leans a little — draft. Protolabs'
//     guidelines ask for 2° in most situations (Toy.swift). The hull's flare
//     is that draft outside; the hollow's walls lean the same way inside, so
//     the wall is one even thickness;
//   * the thwarts are solid blocks down to the floor, their faces drafted:
//     a real thwart is a plank with air under it, which a core lifting out
//     could not leave behind;
//   * the walls are 1.4 mm, inside Protolabs' recommended range for
//     moulded plastic (0.89–3.81 mm, Toy.swift). At full size that is 9 cm.
//   * the oars lie level in their rowlocks at gunwale height, blades flat —
//     at rest, as oars are left — so they lie across the parting plane and
//     come out with the hull. Many toys mould their oars on in this way.

import CoreGraphics
import Foundation
import simd

/// The builder's spec, cm (see the header).
let rowboatLengthCM: Float = 431.8      // 14 ft 2 in
let rowboatBeamCM: Float = 130          // 51 in
let rowboatDepthCM: Float = 46          // 18 in

/// The toy's scale. MODEL: chosen to bring the toy inside 2–3 inches. The
/// `wrongSize` mutant makes it 25% too big.
func rowboatScale(_ mutant: Mutant = activeMutant) -> Float { mutant == .wrongSize ? 1.25 / 65.0 : 1.0 / 65.0 }

/// Oars. MODEL, UNVERIFIED: 7½ ft (2.29 m) is a usual oar for a 14 ft
/// rowboat; no sizing rule could be reached. Wikipedia ("Oar", checked) gives
/// dinghy oars as "less than 2 metres" and racing oars 250–300 cm — this
/// sits between. The blade's length, "about 50 cm", is the same article's;
/// its width, 14 cm, and the inboard share, 28%, are MODEL.
let oarLengthCM: Float = 229
let oarBladeLengthCM: Float = 50
let oarBladeWidthCM: Float = 14
let oarInboardShare: Float = 0.28
/// The oars rest angled 30° aft of square. MODEL.
let oarRestAngleDegrees: Float = 30

/// Paints. MODEL: a toy's — a dark green hull, a buff inside like
/// varnished wood, wooden oars, brass-coloured rowlocks.
enum BoatPaint: Int {
    case hull = 0, inside, oar, rowlock
}
let boatPaints: [Paint] = [
    paint("green hull", 40, 84, 60, rough: 0.30),
    paint("buff inside", 212, 186, 140, rough: 0.34),
    paint("oar wood", 190, 142, 84, rough: 0.28),
    paint("rowlock", 176, 146, 74, rough: 0.22),
]

/// The PVC's own colour in the mass, seen on the inset's cut face. MODEL.
let massColour = SIMD3<Float>(0.62, 0.60, 0.52)

/// The camera: from the port bow, higher than for the animals so the eye
/// sees down into the boat. MODEL.
let stillCameraAzimuthDegrees: Float = 74
let stillCameraElevationDegrees: Float = 25

/// The toy's hull, in toy mm. Everything here is the builder's three numbers
/// at scale, or MODEL shape where no number was reached (flagged).
struct RowboatHull {
    let s: Float                // cm → toy mm
    let depth: Float
    let halfBeam: Float
    let length: Float
    // Plan: a vesica of half-length 38 (the fine, nearly straight bow and
    // the transom a little under half the beam, as a Whitehall's wine-glass
    // transom is), widest 7% of the length aft of the middle. MODEL.
    var planHalfLength: Float { 38 * s * 65 / 10 }
    var planCentre: Float { -4.8 * s * 65 / 10 }
    var transom: Float { -32.3 * s * 65 / 10 }
    /// The topsides' flare: 12°. MODEL — "slight flare" (Wikipedia), and well
    /// over the 2° a mould needs.
    let flareDegrees: Float = 12
    /// Deadrise, the bottom's V: 14°. MODEL.
    let deadriseDegrees: Float = 14
    /// Sheer: lowest 2 mm aft of the middle, rising 2.4 mm to the bow and
    /// 1.5 mm to the transom (radius 285 mm). Keel: lowest 4 mm aft of the
    /// middle, rocker rising 3.5 mm to the forefoot (radius 195 mm). MODEL.
    var sheerLowX: Float { -2 * s * 65 / 10 }
    var sheerRadius: Float { 285 * s * 65 / 10 }
    var keelLowX: Float { -4 * s * 65 / 10 }
    var keelRadius: Float { 195 * s * 65 / 10 }
    /// Wall and floor. MODEL, inside the moulder's range (Toy.swift).
    let wall: Float = 1.4
    let floor: Float = 1.8
    /// Thwarts: bow, rowing (a little aft of the middle) and stern, each
    /// 3.4 mm (22 cm) fore and aft, tops 2.7 mm (18 cm) under the sheer. MODEL.
    var thwartX: [Float] { [15.5 * s * 65 / 10, -3.5 * s * 65 / 10, -20 * s * 65 / 10] }
    let thwartHalfWidth: Float = 1.7
    let thwartBelowSheer: Float = 2.7

    init(scale: Float) {
        s = scale * 10
        depth = rowboatDepthCM * s
        halfBeam = rowboatBeamCM * s / 2
        length = rowboatLengthCM * s
    }

    func sheerY(_ x: Float) -> Float {
        let dx: Float = x - sheerLowX
        return depth + sheerRadius - (sheerRadius * sheerRadius - dx * dx).squareRoot()
    }

    var spec: HullSpec {
        let rad: Float = .pi / 180
        return HullSpec(planHalfLength: planHalfLength, beam: halfBeam, planCentre: planCentre, transom: transom,
                        planHeight: depth, flare: flareDegrees * rad, keelLowX: keelLowX, keelRadius: keelRadius,
                        deadrise: deadriseDegrees * rad, bilge: 3.0, sheerLowX: sheerLowX, sheerLowY: depth,
                        sheerRadius: sheerRadius, topRound: 0.25, cap: depth + 6, wall: wall, floor: floor,
                        innerRound: 1.5, thwartX: thwartX, thwartTop: thwartX.map { sheerY($0) - thwartBelowSheer },
                        thwartHalfWidth: thwartHalfWidth, thwartDraft: draftMostDegrees * rad, seamDrop: 0.5)
    }

    /// The hull's half-width at the sheer at station x (the plan's vesica,
    /// widened by the flare above the plan's height).
    func halfWidthAtSheer(_ x: Float) -> Float {
        let lh: Float = planHalfLength
        let r: Float = (halfBeam + lh * lh / halfBeam) / 2
        let sx: Float = x - planCentre
        let w: Float = (r * r - sx * sx).squareRoot() - (r - halfBeam)
        return w + (sheerY(x) - depth) * tan(flareDegrees * .pi / 180)
    }

    /// Where the rowlocks are: 4.6 mm (30 cm) aft of the rowing thwart's
    /// middle. MODEL.
    var rowlockX: Float { thwartX[1] - 4.6 * s * 65 / 10 }
}

func boatDesign(_ mutant: Mutant = activeMutant) -> BoatDesign {
    let h = RowboatHull(scale: rowboatScale(mutant))
    let hullPaint: Int = BoatPaint.hull.rawValue
    var hullPrims: [Prim] = [h.spec.prim(paint: hullPaint, inside: BoatPaint.inside.rawValue, seam: true)]

    // The skeg: the keel carried straight aft to the transom, as a Whitehall
    // has ("a full skeg"); the toy stands on it and the keel.
    let k: Float = h.s * 65 / 10
    let skeg = ProfileSpec(origin: .zero, xAxis: SIMD3<Float>(1, 0, 0), normal: SIMD3<Float>(0, 0, 1),
                           a: SIMD2<Float>(-8 * k, 0), b: SIMD2<Float>(-30.6 * k, 0), c: SIMD2<Float>(-31.0 * k, 2.6 * k),
                           halfThickness: 0.6, edge: 0.25)
    hullPrims.append(skeg.prim(paint: hullPaint, seam: false))

    // Rowlocks, one each side on the gunwale.
    let rx: Float = h.rowlockX
    let ry: Float = h.sheerY(rx)
    let rz: Float = h.halfWidthAtSheer(rx) - 0.55
    for side in [Float(-1), 1] {
        hullPrims.append(Prim.box(SIMD3<Float>(rx, ry + 0.25, side * rz), half: SIMD3<Float>(0.95, 0.6, 0.6), edge: 0.3,
                                  paint: BoatPaint.rowlock.rawValue).tagged(.rowlock))
    }
    let hull = Segment(name: "hull", part: .hull, prims: hullPrims, blend: 0.4, seam: false)

    // The oars: level at the rowlocks, looms over the gunwale, blades flat.
    let oarY: Float = ry + 0.75
    let frame: Frame = Frame.horizontalParting(at: oarY)
    let len: Float = oarLengthCM * h.s
    let inboard: Float = len * oarInboardShare
    let outboard: Float = len - inboard
    let bladeLen: Float = oarBladeLengthCM * h.s
    let bladeHalfW: Float = oarBladeWidthCM * h.s / 2
    let ang: Float = oarRestAngleDegrees * .pi / 180
    var oarPrims: [Prim] = []
    let wood: Int = BoatPaint.oar.rawValue
    for side in [Float(-1), 1] {
        let pivot = SIMD3<Float>(rx, oarY, side * rz)
        let dir = SIMD3<Float>(-sin(ang), 0, side * cos(ang))
        let handleEnd: SIMD3<Float> = pivot - dir * inboard
        let bladeStart: SIMD3<Float> = pivot + dir * (outboard - bladeLen)
        let grip: SIMD3<Float> = handleEnd + dir * 1.8
        oarPrims.append(Prim.cone(frame.toLocal(handleEnd), 0.45, frame.toLocal(grip), 0.45, paint: wood).tagged(.oar))
        oarPrims.append(Prim.cone(frame.toLocal(grip), 0.55, frame.toLocal(bladeStart + dir * 0.6), 0.48, paint: wood).tagged(.oar))
        let bladeCentre: SIMD3<Float> = pivot + dir * (outboard - bladeLen / 2)
        oarPrims.append(Prim.box(frame.toLocal(bladeCentre), half: SIMD3<Float>(bladeLen / 2, bladeHalfW, 0.5), edge: 0.45,
                                 xAxis: frame.dirToLocal(dir), zAxis: frame.dirToLocal(SIMD3<Float>(0, 1, 0)),
                                 paint: wood).tagged(.blade))
    }
    var segs: [Segment] = [hull]
    var frames: [Frame] = [Frame.identity]
    if mutant != .noOars {
        segs.append(Segment(name: "oars", part: .oars, prims: oarPrims, blend: 0.3, seam: true))
        frames.append(frame)
    }
    return BoatDesign(name: "rowboat", paints: boatPaints, segments: segs, frames: frames)
}

// MARK: - the inset

/// The inset's cut: square across the hull 5 mm forward of the middle,
/// between the rowing and bow thwarts, seen from astern (right = starboard).
/// How many millimetres the inset's height spans:
let insetFieldMillimetres: Float = 22

func boatCut(_ toy: PosedToy) -> Cut {
    var c = Cut(centre: SIMD3<Float>(5.0, 5.2, 0), right: SIMD3<Float>(0, 0, 1), down: SIMD3<Float>(0, -1, 0))
    // The outline of the section's outside, gunwale to gunwale under the keel.
    let sheer: Float = RowboatHull(scale: rowboatScale()).sheerY(c.centre.x)
    var portOut: [SIMD3<Float>] = []
    var starOut: [SIMD3<Float>] = []
    var portIn: [SIMD3<Float>] = []
    var starIn: [SIMD3<Float>] = []
    for i in 0...24 {
        let y: Float = sheer - 0.3 - Float(i) * (sheer - 1.1) / 24
        let a = wallAt(toy, cut: c, y: y)
        starOut.append(SIMD3<Float>(c.centre.x, y, a.outer))
        portOut.append(SIMD3<Float>(c.centre.x, y, -a.outer))
        if a.inner > 0.3 {
            starIn.append(SIMD3<Float>(c.centre.x, y, a.inner))
            portIn.append(SIMD3<Float>(c.centre.x, y, -a.inner))
        }
    }
    var keelY: Float = 1.0
    while keelY > -1 && toy.sdf(SIMD3<Float>(c.centre.x, keelY, 0)).d <= 0 { keelY -= 0.002 }
    // The hollow's floor, across from the starboard wall to the port.
    var floorPts: [SIMD3<Float>] = []
    let zIn: Float = (starIn.first?.z ?? 1) - 0.1
    for j in 0...16 {
        let z: Float = zIn - 2 * zIn * Float(j) / 16
        var fy: Float = sheer + 0.5
        while fy > 0 && toy.sdf(SIMD3<Float>(c.centre.x, fy, z)).d > 0 { fy -= 0.002 }
        floorPts.append(SIMD3<Float>(c.centre.x, fy, z))
    }
    // Round the section: port outside down, under the keel, starboard
    // outside up, then the inside of the hollow back to port.
    c.marks = portOut + [SIMD3<Float>(c.centre.x, keelY, 0)] + starOut.reversed()
        + starIn + floorPts + portIn.reversed()
    return c
}

/// Where the outside and inside of the starboard wall are at height y, on
/// the cut: marched in from beyond the hull along −z, then on through the
/// wall into the hollow.
func wallAt(_ toy: PosedToy, cut: Cut, y: Float) -> (outer: Float, inner: Float) {
    var z: Float = 20
    let x: Float = cut.centre.x
    while z > -20 && toy.sdf(SIMD3<Float>(x, y, z)).d > 0 { z -= 0.002 }
    let outer: Float = z
    while z > -20 && toy.sdf(SIMD3<Float>(x, y, z)).d <= 0 { z -= 0.002 }
    return (outer, z)
}

/// The wall's lean from vertical on the cut, degrees, outside and inside:
/// from where the wall is at two heights.
func draftOnCut(_ toy: PosedToy, cut: Cut) -> (outer: Float, inner: Float, thickness: Float) {
    let y1: Float = 4.6
    let y2: Float = 6.4
    let a = wallAt(toy, cut: cut, y: y1)
    let b = wallAt(toy, cut: cut, y: y2)
    let dy: Float = y2 - y1
    let outer: Float = atan((b.outer - a.outer) / dy) * 180 / .pi
    let inner: Float = atan((b.inner - a.inner) / dy) * 180 / .pi
    let mid = wallAt(toy, cut: cut, y: (y1 + y2) / 2)
    return (outer, inner, mid.outer - mid.inner)
}

/// The notes on the inset: the wall and its draft, the hollow, the parting
/// line, the paint.
func drawInsetNotes(_ ctx: CGContext, setup s: StillSetup, rect r: CGRect, k: CGFloat, height h: CGFloat) {
    let cut: Cut = boatCut(s.toy)
    let hI: Int = Int(h)
    let dr = draftOnCut(s.toy, cut: cut)
    func px(_ p: SIMD3<Float>) -> CGPoint { insetPixel(p, cut: cut, height: hI) }
    // The starboard wall: its outside, drawn as a line, against a plumb line.
    let x: Float = cut.centre.x
    let y1: Float = 3.8
    let y2: Float = 7.6
    let wA = wallAt(s.toy, cut: cut, y: 4.6)
    let wB = wallAt(s.toy, cut: cut, y: 6.4)
    let slope: Float = (wB.outer - wA.outer) / 1.8
    let zAt: (Float) -> Float = { y in wA.outer + (y - 4.6) * slope }
    let top: CGPoint = px(SIMD3<Float>(x, y2, zAt(y2) + 0.25))
    let bottom: CGPoint = px(SIMD3<Float>(x, y1, zAt(y1) + 0.25))
    let plumbTop: CGPoint = px(SIMD3<Float>(x, y2, zAt(y1) + 0.25))
    ctx.setStrokeColor(ink)
    ctx.setLineWidth(2 * k)
    line(ctx, bottom, top, height: h)
    ctx.setLineDash(phase: 0, lengths: [6 * k, 5 * k])
    line(ctx, bottom, plumbTop, height: h)
    ctx.setLineDash(phase: 0, lengths: [])
    let deg: String = String(format: "%.0f°", dr.outer)
    let nx: CGFloat = top.x - 190 * k
    let ny: CGFloat = top.y - 92 * k
    put("draft \(deg) from the plumb line", ctx, x: nx, top: ny, size: 15 * k, bold: true, height: h)
    put("the sides lean so the hull lifts out of", ctx, x: nx, top: ny + 20 * k, size: 12 * k, height: h)
    put("the mould; 2° is enough (Protolabs)", ctx, x: nx, top: ny + 35 * k, size: 12 * k, height: h)
    line(ctx, CGPoint(x: top.x - 4 * k, y: ny + 52 * k), CGPoint(x: top.x - 1 * k, y: top.y + 6 * k), height: h)

    // The wall's thickness, and the hollow.
    let t: String = String(format: "%.1f", dr.thickness)
    put("wall \(t) mm of solid PVC", ctx, x: r.minX + 16 * k, top: r.minY + 14 * k, size: 15 * k, bold: true, height: h)
    put("(9 cm at full size; a moulder keeps walls 0.9–3.8 mm)", ctx, x: r.minX + 16 * k, top: r.minY + 34 * k, size: 12 * k,
        color: softInk, height: h)
    let hollowP: CGPoint = px(SIMD3<Float>(x, 4.2, 0))
    let w1: CGFloat = textWidth("hollow: the mould's", size: 13 * k, bold: false)
    put("hollow: the mould's", ctx, x: hollowP.x - w1 / 2, top: hollowP.y - 18 * k, size: 13 * k, height: h)
    let w2: CGFloat = textWidth("core lifted out of here", size: 13 * k, bold: false)
    put("core lifted out of here", ctx, x: hollowP.x - w2 / 2, top: hollowP.y - 2 * k, size: 13 * k, height: h)

    // The parting line: the bump just under the gunwale, both sides.
    let sheer: Float = RowboatHull(scale: rowboatScale()).sheerY(x)
    let seamY: Float = sheer - 0.5
    let port = wallAt(s.toy, cut: cut, y: seamY)
    let sp: CGPoint = px(SIMD3<Float>(x, seamY, -port.outer))
    ctx.setStrokeColor(ink)
    ctx.setLineWidth(1.4 * k)
    let lab = CGPoint(x: sp.x - 20 * k, y: sp.y - 80 * k)
    line(ctx, CGPoint(x: lab.x + 30 * k, y: lab.y + 36 * k), CGPoint(x: sp.x - 3 * k, y: sp.y - 3 * k), height: h)
    put("parting line", ctx, x: lab.x, top: lab.y, size: 13 * k, bold: true, height: h)
    put("the mould's halves met here", ctx, x: lab.x, top: lab.y + 17 * k, size: 12 * k, height: h)

    let pxPerMM: Float = 1 / insetMillimetresPerPixel(height: hI)
    let paintPx: Float = paintMicrometres / 1000 * pxPerMM
    let um: Int = Int(paintMicrometres)
    put(String(format: "paint, %d µm (model): %.1f px at this scale", um, paintPx), ctx, x: r.minX + 16 * k, top: r.maxY - 30 * k,
        size: 12 * k, height: h)
}

// MARK: - the caption

func boatCaption(lengthMM: Float) -> Caption {
    let seamY: Float = RowboatHull(scale: rowboatScale()).sheerY(20) - 0.5
    return Caption(
        title: "A plastic toy rowing boat",
        lines: [
            "One piece of PVC, injection-moulded and painted; 2–3 inches long (this one \(Int(lengthMM.rounded())) mm): a 14 ft Whitehall-type rowboat at 1:65.",
            "Its gloss is PVC's refractive index: n = 1.545 (Zhang et al. 2020) mirrors 4.6% of the light at normal incidence, more at grazing.",
            "Fine bow, flared sides, a wine-glass transom and a full skeg; three thwarts, and a pair of oars resting level in their rowlocks.",
            "An open boat leaves its mould upwards, so its sides lean (draft) and its thwarts are solid blocks. Unbranded; paint colours are a model.",
        ],
        cutName: "hull, looking forward",
        seamStart: SIMD3<Float>(20, seamY, -25),
        seamDirection: SIMD3<Float>(0, 0, 1),
        seamLabel: CGPoint(x: -150, y: 120))
}
