// What the CPU draws over the GPU's picture: the title and the argument, the
// inset — a cut straight across the front board's fore-edge and the top
// leaves at mid-height, drawn to scale from the same distance function the
// GPU drew (read back by the probe kernel), with the cloth's threads drawn at
// their true pitch — and two scale bars. Text is step 42's CoreText
// approach, as steps 53 and 60 copied it.

import CoreGraphics
import CoreText
import Foundation
import Metal
import simd

let ink = CGColor(srgbRed: 0.10, green: 0.11, blue: 0.12, alpha: 1)
let softInk = CGColor(srgbRed: 0.10, green: 0.11, blue: 0.12, alpha: 0.66)

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

/// Text with its top-left at (x, top) in image coordinates (y down).
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

// MARK: - the inset's geometry, which the tests read

/// The inset, in units of image height (x, y from the top-left).
let insetOrigin = SIMD2<Float>(1.17, 0.27)
let insetSize = SIMD2<Float>(0.52, 0.52)
/// How many millimetres the inset's height spans.
let insetFieldMillimetres: Float = 9.0
func insetMillimetresPerPixel(height: Int) -> Float { insetFieldMillimetres / (insetSize.y * Float(height)) }

/// Scale bars: 50 mm in the main view, 1 mm in the inset.
let mainBarMillimetres: Float = 50
let insetBarMillimetres: Float = 1

func insetRect(height: Int) -> CGRect {
    let h: CGFloat = CGFloat(height)
    return CGRect(x: CGFloat(insetOrigin.x) * h, y: CGFloat(insetOrigin.y) * h,
                  width: CGFloat(insetSize.x) * h, height: CGFloat(insetSize.y) * h)
}

/// The world point at the inset's centre: on the mid-height cut (z = 0),
/// just inside the front board's fore-edge, level with the top leaves.
func insetCentre(_ L: BookLayout) -> SIMD2<Float> {
    SIMD2<Float>(L.boardX1 - 2.3, L.y1 + 0.3)
}

/// The threads are drawn only left of this x; the tests read the layers
/// down the column at `insetTestColumnX`, which nothing is drawn over.
func insetThreadsEndX(_ L: BookLayout) -> Float { L.boardX1 - 4.6 }
func insetTestColumnX(_ L: BookLayout) -> Float { leafWidth - 0.8 }

/// A world point (x, y, z = 0) for an inset pixel: right is +x, down is −y.
func insetPoint(_ px: CGFloat, _ py: CGFloat, layout L: BookLayout, height: Int) -> SIMD3<Float> {
    let r: CGRect = insetRect(height: height)
    let mm: Float = insetMillimetresPerPixel(height: height)
    let c: SIMD2<Float> = insetCentre(L)
    let u: Float = Float(px - r.midX) * mm
    let v: Float = Float(py - r.midY) * mm
    return SIMD3<Float>(c.x + u, c.y - v, 0)
}

/// The inset pixel for a world point on the cut.
func insetPixel(_ p: SIMD2<Float>, layout L: BookLayout, height: Int) -> CGPoint {
    let r: CGRect = insetRect(height: height)
    let mm: Float = insetMillimetresPerPixel(height: height)
    let c: SIMD2<Float> = insetCentre(L)
    return CGPoint(x: r.midX + CGFloat((p.x - c.x) / mm), y: r.midY - CGFloat((p.y - c.y) / mm))
}

/// The inset's colours, linear sRGB. MODEL, chosen to tell the layers apart.
let insetAir = SIMD3<Float>(0.86, 0.86, 0.84)
let insetCloth = SIMD3<Float>(0.10, 0.17, 0.46)
let insetGlue = SIMD3<Float>(0.80, 0.55, 0.12)
let insetBoard = SIMD3<Float>(0.36, 0.34, 0.31)
let insetPastedown = SIMD3<Float>(0.93, 0.88, 0.72)
let insetLeafA = SIMD3<Float>(0.97, 0.96, 0.92)
let insetLeafB = SIMD3<Float>(0.78, 0.77, 0.73)

func insetColour(_ material: Int, _ p: SIMD3<Float>, _ L: BookLayout) -> SIMD3<Float> {
    switch material {
    case Int(Material.cloth.rawValue): return insetCloth
    case Int(Material.adhesive.rawValue): return insetGlue
    case Int(Material.board.rawValue): return insetBoard
    case Int(Material.pastedown.rawValue): return insetPastedown
    case Int(Material.paper.rawValue):
        // Alternate shades leaf by leaf, so each 0.10 mm leaf is seen.
        let k: Int = Int(floor((p.y - L.y0) / paperCaliper))
        return k % 2 == 0 ? insetLeafA : insetLeafB
    default: return insetAir
    }
}

struct ScaleBar {
    var label: String
    var x: CGFloat
    var y: CGFloat
    var pixels: CGFloat
}

func mainScaleBar(_ s: StillSetup, width: Int, height: Int) -> ScaleBar {
    let k: CGFloat = CGFloat(height) / 1080
    let px: CGFloat = CGFloat(mainBarMillimetres * pixelsPerMillimetre(at: s.centre, height: height, cam: s.camera))
    return ScaleBar(label: String(format: "%g mm", mainBarMillimetres), x: 64 * k, y: CGFloat(height) - 70 * k, pixels: px)
}

func insetScaleBar(height: Int) -> ScaleBar {
    let r: CGRect = insetRect(height: height)
    let k: CGFloat = CGFloat(height) / 1080
    let px: CGFloat = CGFloat(insetBarMillimetres / insetMillimetresPerPixel(height: height))
    return ScaleBar(label: String(format: "%g mm", insetBarMillimetres), x: r.maxX - px - 24 * k, y: r.maxY - 26 * k, pixels: px)
}

func drawBar(_ ctx: CGContext, _ b: ScaleBar, k: CGFloat, height h: CGFloat) {
    ctx.setFillColor(ink)
    ctx.fill(flip(CGRect(x: b.x, y: b.y, width: b.pixels, height: 3 * k), h))
    ctx.fill(flip(CGRect(x: b.x, y: b.y - 6 * k, width: 2 * k, height: 9 * k), h))
    ctx.fill(flip(CGRect(x: b.x + b.pixels - 2 * k, y: b.y - 6 * k, width: 2 * k, height: 9 * k), h))
    let w: CGFloat = textWidth(b.label, size: 15 * k, bold: true)
    put(b.label, ctx, x: b.x + b.pixels / 2 - w / 2, top: b.y - 26 * k, size: 15 * k, bold: true, height: h)
}

// MARK: - the caption

func captionLines(_ L: BookLayout) -> (title: String, lines: [String]) {
    let t: String = String(format: "%.2f", L.T)
    let c: String = String(format: "%.4f", paperCaliper)
    return ("A clothbound hardback, closed",
            ["Dyed cotton book cloth glued over 2.5 mm greyboard. The case overhangs the text block by ⅛ in all round",
             "(the squares); each board swings in a French joint, a groove ⅛ in wide; the spine is rounded to a third of a circle.",
             "The text block: \(pageCount) pages = \(leafCount) leaves × \(c) mm (50 lb offset) = \(t) mm, thinner than it looks.",
             "Cloth has no mirror — its standing fibres give a soft sheen at grazing angles (Estevez & Kulla 2017), not a gloss.",
             "Sources: Etherington & Roberts' dictionary; Bailey 1916; Bean & Brodhead 1914; a paper caliper chart."])
}

// MARK: - drawing

func annotate(_ image: BookImage, setup s: StillSetup, renderer: BookRenderer) {
    let w: Int = image.width
    let hI: Int = image.height
    let L: BookLayout = s.layout
    let buf = image.pixels.contents().assumingMemoryBound(to: UInt8.self)

    // The inset's pixels first, each pixel's centre classified by the probe.
    let r: CGRect = insetRect(height: hI)
    let x0: Int = Int(r.minX)
    let y0: Int = Int(r.minY)
    let x1: Int = min(Int(r.maxX), w)
    let y1: Int = min(Int(r.maxY), hI)
    var pts: [SIMD3<Float>] = []
    for y in y0..<y1 {
        for x in x0..<x1 { pts.append(insetPoint(CGFloat(x) + 0.5, CGFloat(y) + 0.5, layout: L, height: hI)) }
    }
    let probed: [SIMD4<Float>] = (try? renderer.probe(pts, scene: s.scene)) ?? []
    func bytes(_ c: SIMD3<Float>) -> SIMD3<UInt8> {
        SIMD3<UInt8>(UInt8((srgbEncode(Double(c.x)) * 255).rounded()), UInt8((srgbEncode(Double(c.y)) * 255).rounded()),
                     UInt8((srgbEncode(Double(c.z)) * 255).rounded()))
    }
    if probed.count == pts.count {
        var i: Int = 0
        for y in y0..<y1 {
            for x in x0..<x1 {
                let c: SIMD3<UInt8> = bytes(insetColour(Int(probed[i].w), pts[i], L))
                let o: Int = (y * w + x) * 4
                buf[o] = c.x
                buf[o + 1] = c.y
                buf[o + 2] = c.z
                buf[o + 3] = 255
                i += 1
            }
        }
    }

    guard let ctx = CGContext(data: image.pixels.contents(), width: w, height: hI, bitsPerComponent: 8,
                              bytesPerRow: w * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                              bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return }
    let h: CGFloat = CGFloat(hI)
    let k: CGFloat = h / 1080
    ctx.setLineCap(.round)
    ctx.setShouldAntialias(true)

    // The threads, on the cover's flat top face inside the inset: filling
    // threads cut across (ellipses, one every 1/77 in) with a warp thread
    // passing over and under them. Drawn at true pitch; the cloth band is
    // the distance function's.
    let mm: CGFloat = CGFloat(insetMillimetresPerPixel(height: hI))
    let clothTop: Float = L.top
    let clothMid: Float = clothTop - clothThickness / 2
    let fp: Float = fillingPitch
    ctx.saveGState()
    ctx.clip(to: flip(r, h))
    let flatEnd: Float = insetThreadsEndX(L)
    var xs: Float = (insetPoint(r.minX, r.midY, layout: L, height: hI).x / fp).rounded(.down) * fp
    var odd: Bool = false
    ctx.setFillColor(CGColor(srgbRed: 0.22, green: 0.32, blue: 0.70, alpha: 1))
    ctx.setStrokeColor(CGColor(srgbRed: 0.05, green: 0.08, blue: 0.25, alpha: 1))
    ctx.setLineWidth(1.2 * k)
    while xs < flatEnd {
        let c: CGPoint = insetPixel(SIMD2<Float>(xs, clothMid), layout: L, height: hI)
        let ew: CGFloat = CGFloat(fp * 0.40) / mm
        let eh: CGFloat = CGFloat(clothThickness * 0.36) / mm
        let dy: CGFloat = (odd ? -1 : 1) * CGFloat(clothThickness * 0.14) / mm
        let e = CGRect(x: c.x - ew, y: c.y + dy - eh, width: 2 * ew, height: 2 * eh)
        ctx.fillEllipse(in: flip(e, h))
        ctx.strokeEllipse(in: flip(e, h))
        xs += fp
        odd.toggle()
    }
    // The warp: a wave over one filling thread and under the next.
    ctx.setStrokeColor(CGColor(srgbRed: 0.55, green: 0.66, blue: 0.95, alpha: 1))
    ctx.setLineWidth(CGFloat(clothThickness * 0.22) / mm)
    ctx.beginPath()
    var first: Bool = true
    var xw: Float = insetPoint(r.minX, r.midY, layout: L, height: hI).x
    while xw < flatEnd {
        let base: Float = cos(Float.pi * (xw / fp))
        let yv: Float = clothMid - base * clothThickness * 0.22
        let pp: CGPoint = insetPixel(SIMD2<Float>(xw, yv), layout: L, height: hI)
        if first { ctx.move(to: CGPoint(x: pp.x, y: h - pp.y)); first = false } else { ctx.addLine(to: CGPoint(x: pp.x, y: h - pp.y)) }
        xw += fp / 24
    }
    ctx.strokePath()
    ctx.restoreGState()

    // Title and argument.
    let cap = captionLines(L)
    put(cap.title, ctx, x: 64 * k, top: 40 * k, size: 30 * k, bold: true, height: h)
    for (i, ln) in cap.lines.enumerated() {
        put(ln, ctx, x: 64 * k, top: (84 + 22 * CGFloat(i)) * k, size: 16 * k, color: softInk, height: h)
    }

    // The ring on the book where the cut is, and lines to the inset.
    let cutWorld = SIMD3<Float>(L.boardX1 - 1.0, L.y1 + 1.5, 0)
    let cp: SIMD2<Float> = project(cutWorld, width: w, height: hI, cam: s.camera)
    let c = CGPoint(x: CGFloat(cp.x), y: CGFloat(cp.y))
    let ringR: CGFloat = 16 * k
    ctx.setStrokeColor(softInk)
    ctx.setLineWidth(1.5 * k)
    for corner in [CGPoint(x: r.minX, y: r.minY), CGPoint(x: r.minX, y: r.maxY)] {
        let dx: CGFloat = corner.x - c.x
        let dy: CGFloat = corner.y - c.y
        let d: CGFloat = (dx * dx + dy * dy).squareRoot()
        line(ctx, CGPoint(x: c.x + dx / d * ringR, y: c.y + dy / d * ringR), corner, height: h)
    }
    ctx.setStrokeColor(ink)
    ctx.setLineWidth(2.5 * k)
    ctx.strokeEllipse(in: flip(CGRect(x: c.x - ringR, y: c.y - ringR, width: 2 * ringR, height: 2 * ringR), h))

    // The inset's frame, heading, labels and bar.
    ctx.setStrokeColor(ink)
    ctx.setLineWidth(3 * k)
    ctx.stroke(flip(r.insetBy(dx: -1.5 * k, dy: -1.5 * k), h))
    put("Cut across the fore-edge at mid-height, true to scale", ctx, x: r.minX, top: r.minY - 30 * k, size: 17 * k, bold: true, height: h)
    // A label: its text's top-left at a world point on the cut, and a
    // leader from the nearer end of the text to the thing it names.
    func label(_ s: String, _ target: SIMD2<Float>, text at: SIMD2<Float>, bold: Bool = true, size: CGFloat = 14,
               sub: String? = nil) {
        let q: CGPoint = insetPixel(target, layout: L, height: hI)
        let t: CGPoint = insetPixel(at, layout: L, height: hI)
        let tw: CGFloat = textWidth(s, size: size * k, bold: bold)
        let midY: CGFloat = t.y + size * k * 0.55
        let from: CGPoint = q.x > t.x + tw ? CGPoint(x: t.x + tw + 4 * k, y: midY)
            : (q.x < t.x ? CGPoint(x: t.x - 4 * k, y: midY) : CGPoint(x: t.x + tw / 2, y: q.y > t.y ? t.y + size * k * 1.2 : t.y - 2 * k))
        ctx.setStrokeColor(ink)
        ctx.setLineWidth(1.2 * k)
        line(ctx, from, q, height: h)
        put(s, ctx, x: t.x, top: t.y, size: size * k, bold: bold, height: h)
        if let sub = sub { put(sub, ctx, x: t.x, top: t.y + size * k * 1.2, size: 13 * k, color: softInk, height: h) }
    }
    let c0: SIMD2<Float> = insetCentre(L)
    let left: Float = c0.x - insetFieldMillimetres / 2 + 0.15
    let rightX: Float = L.boardX1 + 0.25
    label("cloth: woven cotton, 0.30 mm (model)", SIMD2<Float>(left + 1.0, L.top - 0.02), text: SIMD2<Float>(left, L.top + 1.3))
    label(String(format: "threads cut across: one every %.2f mm (%.0f per inch)", fillingPitch, fillingPerInch),
          SIMD2<Float>(insetThreadsEndX(L) - 0.25, L.top - clothThickness / 2),
          text: SIMD2<Float>(insetThreadsEndX(L) + 0.25, L.top + 0.55), bold: false, size: 13)
    label("glue", SIMD2<Float>(L.boardX1 - 1.3, L.top - clothThickness - adhesiveThickness / 2), text: SIMD2<Float>(rightX, L.top - 0.15))
    label("greyboard", SIMD2<Float>(L.boardX1 - 1.5, L.top - shellThickness - boardThickness * 0.45),
          text: SIMD2<Float>(rightX, L.top - 1.0), sub: "2.5 mm")
    label("turn-in", SIMD2<Float>(L.boardX1 - 1.2, L.y1 + pastedownThickness + clothThickness / 2),
          text: SIMD2<Float>(rightX, L.y1 + 0.35), sub: "cloth folded")
    put("round the edge", ctx, x: insetPixel(SIMD2<Float>(rightX, 0), layout: L, height: hI).x,
        top: insetPixel(SIMD2<Float>(0, L.y1 + 0.35), layout: L, height: hI).y + 33 * k, size: 13 * k, color: softInk, height: h)
    label("pastedown", SIMD2<Float>(insetThreadsEndX(L) - 1.2, L.y1 + pastedownThickness / 2), text: SIMD2<Float>(left + 0.9, L.y1 - 0.65))
    label(String(format: "leaves, %.2f mm each", paperCaliper), SIMD2<Float>(leafWidth - 3.1, L.y1 - 1.5),
          text: SIMD2<Float>(left + 0.9, L.y1 - 2.7))
    label("fore-edge: concave, from the round", SIMD2<Float>(leafWidth - L.bulge(L.y1 - 3.1) + 0.02, L.y1 - 3.1),
          text: SIMD2<Float>(leafWidth - 0.6, L.y1 - 3.3), bold: false, size: 13)

    // The square, dimensioned: from the leaves' edge to the case's.
    let yDim: Float = L.y1 - 1.0
    let pa: CGPoint = insetPixel(SIMD2<Float>(leafWidth, yDim), layout: L, height: hI)
    let pb: CGPoint = insetPixel(SIMD2<Float>(L.boardX1, yDim), layout: L, height: hI)
    ctx.setStrokeColor(ink)
    ctx.setLineWidth(1.6 * k)
    line(ctx, pa, pb, height: h)
    line(ctx, CGPoint(x: pa.x, y: pa.y - 6 * k), CGPoint(x: pa.x, y: pa.y + 6 * k), height: h)
    line(ctx, CGPoint(x: pb.x, y: pb.y - 6 * k), CGPoint(x: pb.x, y: pb.y + 6 * k), height: h)
    let sq: String = String(format: "square, %.2f mm (⅛ in)", L.s)
    put(sq, ctx, x: (pa.x + pb.x) / 2 - textWidth(sq, size: 14 * k, bold: true) / 2, top: pa.y + 10 * k, size: 14 * k, bold: true, height: h)
    drawBar(ctx, insetScaleBar(height: hI), k: k, height: h)

    let bar: ScaleBar = mainScaleBar(s, width: w, height: hI)
    drawBar(ctx, bar, k: k, height: h)
    put("at the middle of the cover", ctx, x: bar.x + bar.pixels + 12 * k, top: bar.y - 7 * k, size: 13 * k, color: softInk, height: h)
}
