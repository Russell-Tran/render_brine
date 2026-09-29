// What the CPU draws over the GPU's picture: the title and caption, the
// computed waterline and the draft, a few labels, the buoyancy sums, a true
// midship section, and a true scale bar. Text is step 42's CoreText
// approach, as steps 53, 60, 70 and 72 copied it.

import CoreGraphics
import CoreText
import Foundation
import Metal
import simd

let ink = CGColor(srgbRed: 0.10, green: 0.11, blue: 0.12, alpha: 1)
let softInk = CGColor(srgbRed: 0.10, green: 0.11, blue: 0.12, alpha: 0.70)
let paper = CGColor(srgbRed: 1, green: 1, blue: 1, alpha: 0.88)
let waterlineInk = CGColor(srgbRed: 1.0, green: 0.93, blue: 0.35, alpha: 1)
let seaInk = CGColor(srgbRed: 0.35, green: 0.62, blue: 0.72, alpha: 0.55)

func ctLine(_ s: String, size: CGFloat, bold: Bool, italic: Bool = false, color: CGColor) -> CTLine {
    let name: String = bold ? "HelveticaNeue-Bold" : (italic ? "HelveticaNeue-Italic" : "HelveticaNeue")
    let font = CTFontCreateWithName(name as CFString, size, nil)
    let attributes: [NSAttributedString.Key: Any] = [
        NSAttributedString.Key(kCTFontAttributeName as String): font,
        NSAttributedString.Key(kCTForegroundColorAttributeName as String): color,
    ]
    return CTLineCreateWithAttributedString(NSAttributedString(string: s, attributes: attributes))
}

func textWidth(_ s: String, size: CGFloat, bold: Bool, italic: Bool = false) -> CGFloat {
    CGFloat(CTLineGetTypographicBounds(ctLine(s, size: size, bold: bold, italic: italic, color: ink), nil, nil, nil))
}

func put(_ s: String, _ ctx: CGContext, x: CGFloat, top: CGFloat, size: CGFloat, bold: Bool = false,
         italic: Bool = false, color: CGColor = ink, height: CGFloat) {
    ctx.textPosition = CGPoint(x: x, y: height - top - size)
    CTLineDraw(ctLine(s, size: size, bold: bold, italic: italic, color: color), ctx)
}

func line(_ ctx: CGContext, _ a: CGPoint, _ b: CGPoint, height h: CGFloat) {
    ctx.beginPath()
    ctx.move(to: CGPoint(x: a.x, y: h - a.y))
    ctx.addLine(to: CGPoint(x: b.x, y: h - b.y))
    ctx.strokePath()
}

func flip(_ r: CGRect, _ h: CGFloat) -> CGRect {
    CGRect(x: r.minX, y: h - r.maxY, width: r.width, height: r.height)
}

func cg(_ v: SIMD2<Float>) -> CGPoint { CGPoint(x: CGFloat(v.x), y: CGFloat(v.y)) }

// MARK: - geometry the tests read

/// Scale bar: 1 m at amidships, on the water.
let mainBarMillimetres: Float = 1000

struct ScaleBar {
    var label: String
    var x: CGFloat
    var y: CGFloat
    var pixels: CGFloat
}

func mainScaleBar(_ s: StillSetup, width: Int, height: Int) -> ScaleBar {
    let k: CGFloat = CGFloat(height) / 1080
    let px: CGFloat = CGFloat(mainBarMillimetres * pixelsPerMillimetre(at: s.centre, height: height, cam: s.camera))
    return ScaleBar(label: "1 m", x: 64 * k, y: 236 * k, pixels: px)
}

/// The midship section inset, in units of image height (x, y from the
/// top-left), and its scale.
let insetOrigin = SIMD2<Float>(1.17, 0.585)
let insetSize = SIMD2<Float>(0.56, 0.385)
/// Millimetres of boat per inset pixel at 1080 rows.
let insetMillimetresAt1080: Float = 3.2

func insetRect(height: Int) -> CGRect {
    let h: CGFloat = CGFloat(height)
    return CGRect(x: CGFloat(insetOrigin.x) * h, y: CGFloat(insetOrigin.y) * h,
                  width: CGFloat(insetSize.x) * h, height: CGFloat(insetSize.y) * h)
}

func insetMillimetresPerPixel(height: Int) -> Float { insetMillimetresAt1080 * 1080 / Float(height) }

/// Where the inset puts the waterline (y) and the centreline (x), in pixels.
func insetFrame(height: Int) -> (centreX: CGFloat, waterY: CGFloat) {
    let r: CGRect = insetRect(height: height)
    return (r.minX + r.width * 0.47, r.minY + r.height * 0.60)
}

/// The draft's dimension on the picture: from the water's surface down to
/// the keel's depth, standing in the water 300 mm off the near side, well
/// forward (world, mm); `foot` is where its extension line meets the hull.
let draftDimensionX: Float = 1250
func draftDimensionPoints(_ layout: Layout, draft: Float) -> (top: SIMD3<Float>, bottom: SIMD3<Float>, foot: SIMD3<Float>) {
    let x: Float = draftDimensionX * layout.hull.scale
    let side: Float = layout.hull.halfWidth(x: x, y: draft)
    let z: Float = side + 300
    return (SIMD3<Float>(x, 0, z), SIMD3<Float>(x, -draft, z), SIMD3<Float>(x, -draft, 0))
}

/// The computed waterline on the near (starboard) side, world points.
func waterlinePoints(_ layout: Layout, draft: Float) -> [SIMD3<Float>] {
    var pts: [SIMD3<Float>] = []
    let h: HullShape = layout.hull
    let n: Int = 240
    for i in 0...n {
        let x: Float = -h.halfLength + h.length * Float(i) / Float(n)
        let w: Float = h.halfWidth(x: x, y: draft)
        if w > 0 { pts.append(SIMD3<Float>(x, 0, w)) }
    }
    return pts
}

func drawBar(_ ctx: CGContext, _ b: ScaleBar, k: CGFloat, height h: CGFloat, color: CGColor = ink) {
    ctx.setFillColor(color)
    ctx.fill(flip(CGRect(x: b.x, y: b.y, width: b.pixels, height: 3 * k), h))
    ctx.fill(flip(CGRect(x: b.x, y: b.y - 6 * k, width: 2 * k, height: 9 * k), h))
    ctx.fill(flip(CGRect(x: b.x + b.pixels - 2 * k, y: b.y - 6 * k, width: 2 * k, height: 9 * k), h))
    let w: CGFloat = textWidth(b.label, size: 15 * k, bold: true)
    let color2: CGColor = color
    put(b.label, ctx, x: b.x + b.pixels / 2 - w / 2, top: b.y - 26 * k, size: 15 * k, bold: true, color: color2, height: h)
}

// MARK: - the words

let captionTitle: String = "A Whitehall pulling boat, floating where Archimedes puts it"

func captionLines(_ l: Layout, _ b: Buoyancy) -> [String] {
    let oarFeet: Float = l.oars[0].length / inch / 12
    return [
        "Newfound Woodworks' 16 ft 8 in Whitehall, cedar strip: 5080 mm long, 1067 mm beam, 16.2 in deep amidships, 28 in at the bow, 113 lb.",
        String(format: "A few clean shapes to those numbers: the hull with its keel and full skeg, two rowing thwarts and a stern seat, two pairs of oarlocks ([NW] \"double oarlocks\"),"),
        String(format: "and one pair of %.0f-ft spruce oars at rest, blades trailing 1 cm clear of the water (length by Shaw & Tenney's rule from the %.0f-mm span).", oarFeet, l.span),
        "The lines are a model fitted to Newfound's waterline length, waterline beam and full-load draft; the transom is simplified (a U, not a wineglass).",
    ]
}

func buoyancyLines(_ b: Buoyancy, drawn: Float) -> [String] {
    let each: Double = b.oarKilograms / 2
    return [
        String(format: "boat 113 lb = %.2f kg (Newfound)  +  oars 2 × %.2f kg (their drawn volume × spruce's 425 kg/m³)  =  %.2f kg",
               b.hullKilograms, each, b.totalKilograms),
        String(format: "in sea water, 1025 kg/m³ (the average at the surface; a Whitehall is a harbour boat), it must displace %.2f litres:",
               b.totalKilograms / b.density * 1000),
        String(format: "the hull's own shape, summed slice by slice, displaces that when it sinks %.1f mm — its draft, keel to waterline.", b.draft),
        String(format: "With one rower of 80.7 kg (North America's adult average; not drawn): %.0f mm.  At Newfound's full 725 lb:", b.draftWithRower),
        String(format: "%.0f mm in fresh water, %.0f mm in sea water — Newfound give 8.13 in, %.1f mm.",
               b.draftAtCapacityFresh, b.draftAtCapacitySea, nwDraftAtCapacity),
    ]
}

// MARK: - drawing

func annotate(_ image: BoatImage, setup s: StillSetup, layout: Layout, buoyancy b: Buoyancy, draft: Float) {
    let w: Int = image.width
    let hI: Int = image.height
    guard let ctx = CGContext(data: image.pixels.contents(), width: w, height: hI, bitsPerComponent: 8,
                              bytesPerRow: w * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                              bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return }
    let h: CGFloat = CGFloat(hI)
    let k: CGFloat = h / 1080
    let cam: Camera = s.camera
    func at(_ p: SIMD3<Float>) -> CGPoint { cg(project(p, width: w, height: hI, cam: cam)) }
    ctx.setShouldAntialias(true)
    ctx.setLineCap(.round)

    // Title and caption, on the pale tent above the horizon.
    put(captionTitle, ctx, x: 64 * k, top: 34 * k, size: 30 * k, bold: true, height: h)
    for (i, ln) in captionLines(layout, b).enumerated() {
        put(ln, ctx, x: 64 * k, top: (78 + 21 * CGFloat(i)) * k, size: 15.5 * k, color: softInk, height: h)
    }

    // The computed waterline along the near side.
    let wl: [SIMD3<Float>] = waterlinePoints(layout, draft: draft)
    ctx.setStrokeColor(waterlineInk)
    ctx.setLineWidth(2.2 * k)
    ctx.setLineDash(phase: 0, lengths: [10 * k, 6 * k])
    ctx.beginPath()
    for (i, p) in wl.enumerated() {
        let q: CGPoint = at(p)
        if i == 0 { ctx.move(to: CGPoint(x: q.x, y: h - q.y)) } else { ctx.addLine(to: CGPoint(x: q.x, y: h - q.y)) }
    }
    ctx.strokePath()
    ctx.setLineDash(phase: 0, lengths: [])

    // Labels in white boxes with a leader to a point on the boat.
    func tag(_ lines: [String], x: CGFloat, top: CGFloat, to target: CGPoint?, bold: Bool = false) {
        let size: CGFloat = 14 * k
        var wmax: CGFloat = 0
        for l in lines { wmax = max(wmax, textWidth(l, size: size, bold: bold)) }
        let rows: CGFloat = CGFloat(lines.count)
        let rowH: CGFloat = 18 * k
        let boxH: CGFloat = rows * rowH + 8 * k
        let box = CGRect(x: x - 7 * k, y: top - 5 * k, width: wmax + 14 * k, height: boxH)
        if let t = target {
            ctx.setStrokeColor(CGColor(srgbRed: 1, green: 1, blue: 1, alpha: 0.9))
            ctx.setLineWidth(1.6 * k)
            let from = CGPoint(x: min(max(t.x, box.minX), box.maxX), y: t.y < box.minY ? box.minY : box.maxY)
            line(ctx, from, t, height: h)
            ctx.setFillColor(CGColor(srgbRed: 1, green: 1, blue: 1, alpha: 0.95))
            ctx.fillEllipse(in: flip(CGRect(x: t.x - 3 * k, y: t.y - 3 * k, width: 6 * k, height: 6 * k), h))
        }
        ctx.setFillColor(paper)
        ctx.fill(flip(box, h))
        for (i, l) in lines.enumerated() {
            let row: CGFloat = CGFloat(i) * rowH
            put(l, ctx, x: x, top: top + row, size: size, bold: bold, height: h)
        }
    }

    // The draft: surface to the skeg's foot, just aft of the transom.
    let dd = draftDimensionPoints(layout, draft: draft)
    let pt: CGPoint = at(dd.top), pb: CGPoint = at(dd.bottom), pf: CGPoint = at(dd.foot)
    ctx.setStrokeColor(waterlineInk)
    ctx.setLineWidth(2 * k)
    line(ctx, pt, pb, height: h)
    line(ctx, CGPoint(x: pt.x - 9 * k, y: pt.y), CGPoint(x: pt.x + 9 * k, y: pt.y), height: h)
    line(ctx, CGPoint(x: pb.x - 9 * k, y: pb.y), CGPoint(x: pb.x + 9 * k, y: pb.y), height: h)
    ctx.setLineWidth(1.2 * k)
    ctx.setLineDash(phase: 0, lengths: [4 * k, 4 * k])
    line(ctx, pb, pf, height: h)
    ctx.setLineDash(phase: 0, lengths: [])
    let tagX: CGFloat = pt.x - 330 * k
    let tagTop: CGFloat = pb.y + 34 * k
    let midY: CGFloat = (pt.y + pb.y) / 2
    tag([String(format: "draft %.1f mm, from buoyancy", draft), "waterline (dashed) to the keel's foot"],
        x: tagX, top: tagTop, to: CGPoint(x: pt.x, y: midY))

    // Parts.
    let aft: Station = layout.stations[0]
    let lockN: Oarlock = layout.oarlocks[2]
    let dw = SIMD3<Float>(0, -draft, 0)
    tag(["oarlocks, two pairs,", "on oar blocks"], x: at(lockN.pivot + dw).x - 40 * k, top: 196 * k,
        to: at(lockN.pivot + dw))
    // On the aft thwart's top, on the far side of the centreline, where it
    // shows over the near gunwale.
    let seatP = SIMD3<Float>(aft.seatCentreX, aft.seatTop, -200 * layout.hull.scale) + dw
    tag(["thwarts (seats): 13 in forward", "of their oarlocks (Angus)"], x: 600 * k, top: 196 * k, to: at(seatP))
    let o: Oar = layout.oars[0]
    let bladeMid: SIMD3<Float> = o.pivot + o.axis * (o.outboard - bladeLength / 2) + dw
    tag(["oars at rest, blades trailing"], x: at(bladeMid).x - 120 * k, top: at(bladeMid).y + 60 * k, to: at(bladeMid))

    // The buoyancy sums, on the water at the lower left.
    let bl: [String] = buoyancyLines(b, drawn: draft)
    let bx: CGFloat = 64 * k
    let btop: CGFloat = 886 * k
    var bw: CGFloat = textWidth("Why it floats here", size: 17 * k, bold: true)
    for l in bl { bw = max(bw, textWidth(l, size: 14.5 * k, bold: false)) }
    ctx.setFillColor(paper)
    let lineH: CGFloat = 20 * k
    let panelRows: CGFloat = CGFloat(bl.count) * lineH
    let panelH: CGFloat = 48 * k + panelRows
    ctx.fill(flip(CGRect(x: bx - 12 * k, y: btop - 12 * k, width: bw + 24 * k, height: panelH), h))
    put("Why it floats here", ctx, x: bx, top: btop, size: 17 * k, bold: true, height: h)
    for (i, l) in bl.enumerated() {
        let row: CGFloat = CGFloat(i) * lineH
        put(l, ctx, x: bx, top: btop + 28 * k + row, size: 14.5 * k, height: h)
    }

    drawSection(ctx, layout: layout, draft: draft, buoyancy: b, imageHeight: hI)
    drawProfile(ctx, layout: layout, draft: draft, buoyancy: b, imageHeight: hI)

    let bar: ScaleBar = mainScaleBar(s, width: w, height: hI)
    ctx.setFillColor(paper)
    let barBoxW: CGFloat = bar.pixels + 174 * k
    ctx.fill(flip(CGRect(x: bar.x - 12 * k, y: bar.y - 36 * k, width: barBoxW, height: 50 * k), h))
    drawBar(ctx, bar, k: k, height: h)
    put("amidships, at the water", ctx, x: bar.x + bar.pixels + 12 * k, top: bar.y - 7 * k, size: 13 * k, color: softInk, height: h)
}

/// The side view inset (image-height units) and its scale.
let profileOrigin = SIMD2<Float>(0.048, 0.60)
let profileSize = SIMD2<Float>(1.08, 0.19)
let profileMillimetresAt1080: Float = 5.0

func profileRect(height: Int) -> CGRect {
    let h: CGFloat = CGFloat(height)
    return CGRect(x: CGFloat(profileOrigin.x) * h, y: CGFloat(profileOrigin.y) * h,
                  width: CGFloat(profileSize.x) * h, height: CGFloat(profileSize.y) * h)
}

func profileMillimetresPerPixel(height: Int) -> Float { profileMillimetresAt1080 * 1080 / Float(height) }

/// The side view: the hull's profile from the same lines (sheer, bottom,
/// keel, stem, transom), the sea below the computed waterline, the wetted
/// length, the draft at the skeg, and Newfound's full load for comparison.
func drawProfile(_ ctx: CGContext, layout: Layout, draft: Float, buoyancy b: Buoyancy, imageHeight hI: Int) {
    let h: CGFloat = CGFloat(hI)
    let k: CGFloat = h / 1080
    let r: CGRect = profileRect(height: hI)
    let mm: CGFloat = CGFloat(profileMillimetresPerPixel(height: hI))
    let hull: HullShape = layout.hull
    // Bow to the right, as in the picture... the picture has the bow to the
    // right too.
    let cx: CGFloat = r.minX + r.width * 0.47
    let wy: CGFloat = r.minY + r.height * 0.74
    func P(_ x: Float, _ y: Float) -> CGPoint {
        CGPoint(x: cx + CGFloat(x) / mm, y: wy - CGFloat(y - draft) / mm)
    }
    ctx.setFillColor(CGColor(srgbRed: 1, green: 1, blue: 1, alpha: 0.93))
    ctx.fill(flip(r, h))
    ctx.saveGState()
    ctx.clip(to: flip(r, h))
    ctx.setFillColor(seaInk)
    ctx.fill(flip(CGRect(x: r.minX, y: wy, width: r.width, height: r.maxY - wy), h))
    // Outline: sheer from transom to stem, down the stem, along the keel's
    // bottom back to the transom, up the transom.
    let n: Int = 200
    let L: Float = hull.length
    let x0: Float = -hull.halfLength
    let path = CGMutablePath()
    for i in 0...n {
        let x: Float = x0 + L * Float(i) / Float(n)
        let q: CGPoint = P(x, hull.sheer(x))
        if i == 0 { path.move(to: CGPoint(x: q.x, y: h - q.y)) } else { path.addLine(to: CGPoint(x: q.x, y: h - q.y)) }
    }
    for i in stride(from: n, through: 0, by: -1) {
        let x: Float = x0 + L * Float(i) / Float(n)
        let q: CGPoint = P(x, hull.keelBottom(x))
        path.addLine(to: CGPoint(x: q.x, y: h - q.y))
    }
    path.closeSubpath()
    ctx.setFillColor(CGColor(srgbRed: 0.66, green: 0.36, blue: 0.16, alpha: 1))
    ctx.addPath(path)
    ctx.fillPath()
    // The keel and skeg, darker: between the keel's bottom and the hull's.
    let keel = CGMutablePath()
    for i in 0...n {
        let x: Float = x0 + L * Float(i) / Float(n)
        let q: CGPoint = P(x, hull.bottom(x))
        if i == 0 { keel.move(to: CGPoint(x: q.x, y: h - q.y)) } else { keel.addLine(to: CGPoint(x: q.x, y: h - q.y)) }
    }
    for i in stride(from: n, through: 0, by: -1) {
        let x: Float = x0 + L * Float(i) / Float(n)
        let q: CGPoint = P(x, hull.keelBottom(x))
        keel.addLine(to: CGPoint(x: q.x, y: h - q.y))
    }
    keel.closeSubpath()
    ctx.setFillColor(CGColor(srgbRed: 0.36, green: 0.14, blue: 0.07, alpha: 1))
    ctx.addPath(keel)
    ctx.fillPath()
    // The sea's surface, and Newfound's full-load waterline (fresh water).
    ctx.setStrokeColor(CGColor(srgbRed: 0.10, green: 0.35, blue: 0.50, alpha: 1))
    ctx.setLineWidth(2 * k)
    line(ctx, CGPoint(x: r.minX, y: wy), CGPoint(x: r.maxX, y: wy), height: h)
    let full: CGFloat = P(0, b.draftAtCapacityFresh).y
    ctx.setLineDash(phase: 0, lengths: [6 * k, 4 * k])
    ctx.setLineWidth(1.3 * k)
    ctx.setStrokeColor(CGColor(srgbRed: 1, green: 1, blue: 1, alpha: 0.9))
    line(ctx, CGPoint(x: P(x0 - 120, 0).x, y: full), CGPoint(x: P(-x0 + 120, 0).x, y: full), height: h)
    ctx.setLineDash(phase: 0, lengths: [])
    ctx.restoreGState()

    // The wetted length at this draft, and the draft at the skeg.
    let wl = hull.waterline(at: draft)
    ctx.setStrokeColor(ink)
    ctx.setLineWidth(1.4 * k)
    let skegX: CGFloat = P(x0, 0).x - 26 * k
    let yKeel: CGFloat = P(0, 0).y
    line(ctx, CGPoint(x: skegX, y: wy), CGPoint(x: skegX, y: yKeel), height: h)
    line(ctx, CGPoint(x: skegX - 6 * k, y: wy), CGPoint(x: skegX + 6 * k, y: wy), height: h)
    line(ctx, CGPoint(x: skegX - 6 * k, y: yKeel), CGPoint(x: skegX + 6 * k, y: yKeel), height: h)
    let dl: String = String(format: "draft %.1f mm", draft)
    put(dl, ctx, x: skegX - 8 * k, top: yKeel + 6 * k, size: 13.5 * k, bold: true, height: h)
    let lwlLabel: String = String(format: "waterline %.2f m long, empty (Newfound's %.2f m is at full load)", wl.length / 1000, nwLWL / 1000)
    let lwlW: CGFloat = textWidth(lwlLabel, size: 13 * k, bold: false)
    put(lwlLabel, ctx, x: cx - lwlW / 2, top: wy + 24 * k, size: 13 * k, height: h)
    let fl: String = String(format: "full load, 725 lb, fresh water: %.0f mm (Newfound: 8.13 in = %.1f mm)",
                            b.draftAtCapacityFresh, nwDraftAtCapacity)
    let flW: CGFloat = textWidth(fl, size: 12.5 * k, bold: false)
    put(fl, ctx, x: cx - flW / 2, top: full - 19 * k, size: 12.5 * k,
        color: CGColor(srgbRed: 1, green: 1, blue: 1, alpha: 0.95), height: h)

    ctx.setStrokeColor(ink)
    ctx.setLineWidth(2.5 * k)
    ctx.stroke(flip(r.insetBy(dx: -1.25 * k, dy: -1.25 * k), h))
    put("From the side, true to scale: the same lines, the computed waterline", ctx, x: r.minX + 12 * k, top: r.minY + 10 * k,
        size: 15 * k, bold: true, height: h)
    let px: CGFloat = 1000 / mm
    let sb = ScaleBar(label: "1 m", x: r.maxX - px - 20 * k, y: r.maxY - 12 * k, pixels: px)
    drawBar(ctx, sb, k: k, height: h)
}

/// The midship section (x = 0), true to scale: the hull's outline from the
/// same function the kernel mirrors, the sea below the computed waterline,
/// the draft, the depth and the beams.
func drawSection(_ ctx: CGContext, layout: Layout, draft: Float, buoyancy b: Buoyancy, imageHeight hI: Int) {
    let h: CGFloat = CGFloat(hI)
    let k: CGFloat = h / 1080
    let r: CGRect = insetRect(height: hI)
    let mm: CGFloat = CGFloat(insetMillimetresPerPixel(height: hI))
    let f = insetFrame(height: hI)
    let hull: HullShape = layout.hull
    // Hull y (mm) → inset pixel y; z (mm) → x.
    func P(_ z: Float, _ y: Float) -> CGPoint {
        CGPoint(x: f.centreX + CGFloat(z) / mm, y: f.waterY - CGFloat(y - draft) / mm)
    }
    ctx.setFillColor(CGColor(srgbRed: 1, green: 1, blue: 1, alpha: 0.93))
    ctx.fill(flip(r, h))
    ctx.saveGState()
    ctx.clip(to: flip(r, h))
    // The sea.
    ctx.setFillColor(seaInk)
    ctx.fill(flip(CGRect(x: r.minX, y: f.waterY, width: r.width, height: r.maxY - f.waterY), h))
    // The section: body outline from the keel up each side, and the keel.
    let x0: Float = 0
    let kb: Float = hull.bottom(x0)
    let sh: Float = hull.sheer(x0)
    var right: [CGPoint] = []
    let n: Int = 160
    for i in 0...n {
        let y: Float = kb + (sh - kb) * Float(i) / Float(n)
        right.append(P(hull.halfWidth(x: x0, y: y), y))
    }
    let path = CGMutablePath()
    path.move(to: CGPoint(x: right[0].x, y: h - right[0].y))
    for p in right { path.addLine(to: CGPoint(x: p.x, y: h - p.y)) }
    for p in right.reversed() {
        let m = CGPoint(x: 2 * f.centreX - p.x, y: p.y)
        path.addLine(to: CGPoint(x: m.x, y: h - m.y))
    }
    path.closeSubpath()
    ctx.setFillColor(CGColor(srgbRed: 0.66, green: 0.36, blue: 0.16, alpha: 1))
    ctx.addPath(path)
    ctx.fillPath()
    let kw: Float = hull.keelHalfWidth
    let keelTop: CGPoint = P(-kw, kb)
    let keelBot: CGPoint = P(kw, hull.keelBottom(x0))
    ctx.setFillColor(CGColor(srgbRed: 0.36, green: 0.14, blue: 0.07, alpha: 1))
    ctx.fill(flip(CGRect(x: keelTop.x, y: keelTop.y, width: keelBot.x - keelTop.x, height: keelBot.y - keelTop.y), h))
    // The inside: the skin's thickness in from the outline (drawn as a
    // lighter fill: the dry cavity).
    let inner = CGMutablePath()
    var started: Bool = false
    var innerPts: [CGPoint] = []
    for i in 0...n {
        let y: Float = kb + hull.skin + (sh - kb - hull.skin) * Float(i) / Float(n)
        let wv: Float = hull.halfWidth(x: x0, y: y) - hull.skin
        if wv > 0 { innerPts.append(P(wv, y)) }
    }
    for p in innerPts {
        if !started { inner.move(to: CGPoint(x: p.x, y: h - p.y)); started = true } else { inner.addLine(to: CGPoint(x: p.x, y: h - p.y)) }
    }
    for p in innerPts.reversed() { inner.addLine(to: CGPoint(x: 2 * f.centreX - p.x, y: h - p.y)) }
    inner.closeSubpath()
    ctx.setFillColor(CGColor(srgbRed: 0.97, green: 0.95, blue: 0.91, alpha: 1))
    ctx.addPath(inner)
    ctx.fillPath()
    // The waterline.
    ctx.setStrokeColor(CGColor(srgbRed: 0.10, green: 0.35, blue: 0.50, alpha: 1))
    ctx.setLineWidth(2 * k)
    line(ctx, CGPoint(x: r.minX, y: f.waterY), CGPoint(x: r.maxX, y: f.waterY), height: h)
    ctx.restoreGState()

    // Dimensions: draft (keel to water) at the right, depth (keel to sheer)
    // at the left, the waterline beam across.
    ctx.setStrokeColor(ink)
    ctx.setLineWidth(1.4 * k)
    let halfB: Float = hull.beam(x0)
    let dxR: CGFloat = P(halfB, 0).x + 34 * k
    let yKeel: CGFloat = P(0, 0).y
    let ySheer: CGFloat = P(0, sh).y
    func vdim(_ x: CGFloat, _ y0: CGFloat, _ y1: CGFloat) {
        line(ctx, CGPoint(x: x, y: y0), CGPoint(x: x, y: y1), height: h)
        line(ctx, CGPoint(x: x - 6 * k, y: y0), CGPoint(x: x + 6 * k, y: y0), height: h)
        line(ctx, CGPoint(x: x - 6 * k, y: y1), CGPoint(x: x + 6 * k, y: y1), height: h)
    }
    vdim(dxR, f.waterY, yKeel)
    ctx.setLineDash(phase: 0, lengths: [3 * k, 3 * k])
    line(ctx, CGPoint(x: keelBot.x, y: yKeel), CGPoint(x: dxR, y: yKeel), height: h)
    ctx.setLineDash(phase: 0, lengths: [])
    let midWater: CGFloat = (f.waterY + yKeel) / 2
    let draftTop: CGFloat = midWater - 8 * k
    put(String(format: "draft %.1f mm", draft), ctx, x: dxR + 10 * k, top: draftTop, size: 14 * k, bold: true, height: h)
    let dxL: CGFloat = P(-halfB, 0).x - 30 * k
    vdim(dxL, ySheer, yKeel)
    let depthLabel: String = String(format: "%.1f mm", sh)
    let midDepth: CGFloat = (ySheer + yKeel) / 2
    let depthW: CGFloat = textWidth(depthLabel, size: 13 * k, bold: false)
    let inchW: CGFloat = textWidth("(16.2 in)", size: 13 * k, bold: false)
    put(depthLabel, ctx, x: dxL - depthW - 8 * k, top: midDepth - 16 * k, size: 13 * k, height: h)
    put("(16.2 in)", ctx, x: dxL - inchW - 8 * k, top: midDepth + 1 * k, size: 13 * k, color: softInk, height: h)
    let beamY: CGFloat = ySheer - 16 * k
    let bl: CGPoint = P(-halfB, sh), br: CGPoint = P(halfB, sh)
    line(ctx, CGPoint(x: bl.x, y: beamY), CGPoint(x: br.x, y: beamY), height: h)
    line(ctx, CGPoint(x: bl.x, y: beamY - 5 * k), CGPoint(x: bl.x, y: beamY + 5 * k), height: h)
    line(ctx, CGPoint(x: br.x, y: beamY - 5 * k), CGPoint(x: br.x, y: beamY + 5 * k), height: h)
    let beamLabel: String = String(format: "beam here %.0f mm", 2 * halfB)
    let beamW: CGFloat = textWidth(beamLabel, size: 13 * k, bold: false)
    let beamMid: CGFloat = (bl.x + br.x) / 2
    put(beamLabel, ctx, x: beamMid - beamW / 2, top: beamY - 20 * k, size: 13 * k,
        height: h)

    // Frame, title, scale bar.
    ctx.setStrokeColor(ink)
    ctx.setLineWidth(2.5 * k)
    ctx.stroke(flip(r.insetBy(dx: -1.25 * k, dy: -1.25 * k), h))
    put("Amidships, cut across, true to scale", ctx, x: r.minX + 12 * k, top: r.minY + 10 * k, size: 15 * k, bold: true, height: h)
    put("sea below the computed waterline; the inside is dry", ctx, x: r.minX + 12 * k, top: r.minY + 30 * k, size: 12.5 * k,
        color: softInk, height: h)
    let px: CGFloat = 200 / mm
    let sb = ScaleBar(label: "200 mm", x: r.maxX - px - 20 * k, y: r.maxY - 16 * k, pixels: px)
    drawBar(ctx, sb, k: k, height: h)
}
