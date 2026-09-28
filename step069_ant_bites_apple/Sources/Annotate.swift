// Step 69's labels: the title and caption, the 1 mm bar, a leader to the
// jaws, and the inset — a diagram of where a bite goes inside the ant, drawn
// by the CPU from the ant's own side view (Gut.swift). Text drawing is step
// 19's CoreText approach, as in every step. All text sits on the card or on
// plates, off the ant and off the apple (a test checks).

import CoreGraphics
import CoreText
import Foundation
import Metal
import simd

private let ink = CGColor(srgbRed: 0.13, green: 0.14, blue: 0.16, alpha: 1)
private let softInk = CGColor(srgbRed: 0.13, green: 0.14, blue: 0.16, alpha: 0.62)
private let plate = CGColor(srgbRed: 1, green: 1, blue: 1, alpha: 0.90)
/// The juice and the bits, as the inset colours them.
let juiceRGB = SIMD3<Float>(0.18, 0.62, 0.86)
let solidRGB = SIMD3<Float>(0.80, 0.52, 0.16)
private let juiceInk = CGColor(srgbRed: 0.10, green: 0.48, blue: 0.72, alpha: 1)
private let solidInk = CGColor(srgbRed: 0.66, green: 0.38, blue: 0.08, alpha: 1)

private func ctLine(_ s: String, size: CGFloat, bold: Bool, italic: Bool = false, color: CGColor) -> CTLine {
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

private func put(_ s: String, _ ctx: CGContext, x: CGFloat, top: CGFloat, size: CGFloat, bold: Bool = false,
                 italic: Bool = false, color: CGColor = ink, height: CGFloat) {
    ctx.textPosition = CGPoint(x: x, y: height - top - size)
    CTLineDraw(ctLine(s, size: size, bold: bold, italic: italic, color: color), ctx)
}

private func line(_ ctx: CGContext, _ a: CGPoint, _ b: CGPoint, height h: CGFloat) {
    ctx.beginPath()
    ctx.move(to: CGPoint(x: a.x, y: h - a.y))
    ctx.addLine(to: CGPoint(x: b.x, y: h - b.y))
    ctx.strokePath()
}

func mainMillimetresPerPixel(width: Int) -> Float { mainViewWidth / Float(width) }

/// The two outer tangent lines between two circles: the magnifier callout.
private func tangents(_ c1: CGPoint, _ r1: CGFloat, _ c2: CGPoint, _ r2: CGFloat) -> [(CGPoint, CGPoint)] {
    let dx: CGFloat = c2.x - c1.x
    let dy: CGFloat = c2.y - c1.y
    let d: CGFloat = (dx * dx + dy * dy).squareRoot()
    let base: CGFloat = atan2(dy, dx)
    let off: CGFloat = acos((r1 - r2) / d)
    var out: [(CGPoint, CGPoint)] = []
    for a in [base + off, base - off] {
        let ca: CGFloat = cos(a)
        let sa: CGFloat = sin(a)
        out.append((CGPoint(x: c1.x + r1 * ca, y: c1.y + r1 * sa), CGPoint(x: c2.x + r2 * ca, y: c2.y + r2 * sa)))
    }
    return out
}

/// The title, and the caption, from the numbers.
let titleText: String = " worker biting a piece of peeled Golden Delicious apple"
func captionLines() -> [String] {
    [
        "Its mandibles swing open and close on the cut edge, stopping where they meet the flesh.",
        String(format: "Slowed ×%.0f: two bites a second (not measured) while the antennae tap %.0f times a second.",
               slowdown, realStrokesPerSecond),
        "Ants drink their food: the juice — mostly fructose — goes to the crop. Bits of flesh are",
        "strained out at the mouth into the infrabuccal pocket and packed into a pellet, spat out later.",
        "The bite is a fresh cut: it gives off 2-hexenal. Browning, minutes away, is not shown.",
    ]
}

// MARK: - where the words and the inset sit

/// The title and caption block, top left.
func titleBlock(height hI: Int) -> PixelRect {
    let k: Float = Float(hI) / 900
    let rows: Float = Float(captionLines().count)
    let depth: Float = 19 * rows
    return PixelRect(x0: Int(40 * k), y0: Int(20 * k), x1: Int(820 * k), y1: Int((78 + depth) * k))
}

/// The inset panel: a white plate, left, under the caption.
func insetBlock(height hI: Int) -> PixelRect {
    let k: Float = Float(hI) / 900
    return PixelRect(x0: Int(40 * k), y0: Int(250 * k), x1: Int(565 * k), y1: Int(530 * k))
}

/// The 1 mm bar, under the caption.
func scaleBarBlock(height hI: Int) -> PixelRect {
    let k: Float = Float(hI) / 900
    let mm: Float = 1 / mainMillimetresPerPixel(width: hI * 16 / 9)
    return PixelRect(x0: Int(40 * k), y0: Int(190 * k), x1: Int(60 * k + mm), y1: Int(232 * k))
}

/// The magnified view's words: a plate inside the top of its circle.
func zoomLabelBlock(height hI: Int) -> PixelRect {
    let h: Float = Float(hI)
    let k: Float = h / 900
    let cx: Float = zoomCentre.x * h
    let top: Float = (zoomCentre.y - zoomRadius) * h
    return PixelRect(x0: Int(cx - 165 * k), y0: Int(top + 26 * k), x1: Int(cx + 165 * k), y1: Int(top + 74 * k))
}

// MARK: - the diagram

/// The side-view millimetres the diagram shows, and how they map into the panel.
struct InsetMap {
    let panel: PixelRect
    let lo: SIMD2<Float>
    let scale: Float        // px per mm

    init(height hI: Int) {
        panel = insetBlock(height: hI)
        let k: Float = Float(hI) / 900
        // The ant spans x −2.1 … 2.2 mm, y 0 … 1.2 mm in its own side view.
        lo = SIMD2<Float>(-2.15, -0.05)
        let usableW: Float = Float(panel.x1 - panel.x0) - 40 * k
        scale = usableW / 4.4
    }

    func point(_ p: SIMD2<Float>) -> CGPoint {
        let k: Float = scale
        let x: Float = Float(panel.x0) + 20 + (p.x - lo.x) * k
        let baseY: Float = Float(panel.y1) - 70
        let y: Float = baseY - (p.y - lo.y) * k
        return CGPoint(x: CGFloat(x), y: CGFloat(y))
    }
}

private func drawInset(_ ctx: CGContext, frame: FrameState, height hI: Int) {
    let h: CGFloat = CGFloat(hI)
    let k: CGFloat = h / 900
    let map = InsetMap(height: hI)
    let pr: PixelRect = map.panel
    ctx.setFillColor(plate)
    ctx.fill(CGRect(x: CGFloat(pr.x0), y: h - CGFloat(pr.y1), width: CGFloat(pr.x1 - pr.x0), height: CGFloat(pr.y1 - pr.y0)))
    ctx.setStrokeColor(ink)
    ctx.setLineWidth(1.2 * k)
    ctx.stroke(CGRect(x: CGFloat(pr.x0), y: h - CGFloat(pr.y1), width: CGFloat(pr.x1 - pr.x0), height: CGFloat(pr.y1 - pr.y0)))
    put("Where a bite goes", ctx, x: CGFloat(pr.x0) + 14 * k, top: CGFloat(pr.y0) + 10 * k, size: 15 * k, bold: true, height: h)
    put("the ant seen from its side, cut open — schematic, not to scale", ctx, x: CGFloat(pr.x0) + 14 * k,
        top: CGFloat(pr.y0) + 30 * k, size: 11.5 * k, color: softInk, height: h)

    // The body outline: each ellipsoid of the body as its side-view ellipse,
    // the neck as a band — from the same shapes the GPU draws.
    let pxPerMM: CGFloat = CGFloat(map.scale)
    ctx.setFillColor(CGColor(srgbRed: 0.90, green: 0.89, blue: 0.87, alpha: 1))
    ctx.setStrokeColor(CGColor(srgbRed: 0.35, green: 0.33, blue: 0.31, alpha: 1))
    ctx.setLineWidth(1.4 * k)
    for s in sideViewBody() where s.part != .leg && s.part != .antenna {
        switch s.kind {
        case .ellipsoid, .roundBox:
            let c: CGPoint = map.point(GutPoint(s.a.x, s.a.y))
            let ang: CGFloat = CGFloat(atan2(s.xAxis.y, s.xAxis.x))
            let rx: CGFloat = CGFloat(s.kind == .ellipsoid ? s.b.x : s.b.x + s.ra) * pxPerMM
            let ry: CGFloat = CGFloat(s.kind == .ellipsoid ? s.b.y : s.b.y + s.ra) * pxPerMM
            ctx.saveGState()
            ctx.translateBy(x: c.x, y: h - c.y)
            ctx.rotate(by: ang)
            let r = CGRect(x: -rx, y: -ry, width: 2 * rx, height: 2 * ry)
            ctx.fillEllipse(in: r)
            ctx.strokeEllipse(in: r)
            ctx.restoreGState()
        case .roundCone:
            let a: CGPoint = map.point(GutPoint(s.a.x, s.a.y))
            let b: CGPoint = map.point(GutPoint(s.b.x, s.b.y))
            ctx.setLineWidth(CGFloat(s.ra + s.rb) * pxPerMM)
            ctx.setLineCap(.round)
            ctx.setStrokeColor(CGColor(srgbRed: 0.62, green: 0.58, blue: 0.54, alpha: 1))
            line(ctx, a, b, height: h)
            ctx.setStrokeColor(CGColor(srgbRed: 0.35, green: 0.33, blue: 0.31, alpha: 1))
            ctx.setLineWidth(1.4 * k)
        }
    }
    // The gut: the oesophagus as a thin tube along the juice's road, the
    // crop, the pocket and its pellet.
    ctx.setLineCap(.round)
    ctx.setLineJoin(.round)
    ctx.setStrokeColor(CGColor(srgbRed: 0.55, green: 0.72, blue: 0.82, alpha: 1))
    ctx.setLineWidth(5 * k)
    ctx.beginPath()
    for (i, p) in juicePath.enumerated() {
        let q: CGPoint = map.point(p)
        if i == 0 { ctx.move(to: CGPoint(x: q.x, y: h - q.y)) } else { ctx.addLine(to: CGPoint(x: q.x, y: h - q.y)) }
    }
    ctx.strokePath()
    let crop: CGPoint = map.point(cropCentre)
    let crx: CGFloat = CGFloat(cropRadii.x) * pxPerMM
    let cry: CGFloat = CGFloat(cropRadii.y) * pxPerMM
    ctx.setFillColor(CGColor(srgbRed: 0.74, green: 0.87, blue: 0.95, alpha: 1))
    ctx.fillEllipse(in: CGRect(x: crop.x - crx, y: h - crop.y - cry, width: 2 * crx, height: 2 * cry))
    ctx.setStrokeColor(CGColor(srgbRed: 0.35, green: 0.55, blue: 0.70, alpha: 1))
    ctx.setLineWidth(1.4 * k)
    ctx.strokeEllipse(in: CGRect(x: crop.x - crx, y: h - crop.y - cry, width: 2 * crx, height: 2 * cry))
    let pocket: CGPoint = map.point(pocketCentre)
    let prad: CGFloat = CGFloat(pocketRadius) * pxPerMM
    ctx.setFillColor(CGColor(srgbRed: 0.97, green: 0.90, blue: 0.80, alpha: 1))
    ctx.fillEllipse(in: CGRect(x: pocket.x - prad, y: h - pocket.y - prad, width: 2 * prad, height: 2 * prad))
    ctx.setStrokeColor(CGColor(srgbRed: 0.66, green: 0.45, blue: 0.20, alpha: 1))
    ctx.strokeEllipse(in: CGRect(x: pocket.x - prad, y: h - pocket.y - prad, width: 2 * prad, height: 2 * prad))
    let pel: CGFloat = CGFloat(pelletRadius) * pxPerMM
    ctx.setFillColor(CGColor(srgbRed: 0.55, green: 0.36, blue: 0.14, alpha: 1))
    ctx.fillEllipse(in: CGRect(x: pocket.x - pel, y: h - pocket.y - pel, width: 2 * pel, height: 2 * pel))

    // The dots: juice blue, bits of flesh amber.
    for d in frame.gut.dots where d.radius > 1e-4 {
        let c: CGPoint = map.point(d.position)
        let r: CGFloat = CGFloat(d.radius) * pxPerMM
        let rgb: SIMD3<Float> = d.kind == .juice ? juiceRGB : solidRGB
        ctx.setFillColor(CGColor(srgbRed: CGFloat(rgb.x), green: CGFloat(rgb.y), blue: CGFloat(rgb.z), alpha: 1))
        ctx.fillEllipse(in: CGRect(x: c.x - r, y: h - c.y - r, width: 2 * r, height: 2 * r))
    }

    // The words, along the plate's foot, each with a leader to what it names.
    let footY: CGFloat = CGFloat(pr.y1) - 50 * k
    let juiceX: CGFloat = CGFloat(pr.x0) + 14 * k
    let solidX: CGFloat = CGFloat(pr.x0) + 300 * k
    put("juice → the crop, in the gaster", ctx, x: juiceX, top: footY, size: 12.5 * k, bold: true, color: juiceInk, height: h)
    put("drunk: pharynx, then oesophagus", ctx, x: juiceX, top: footY + 17 * k, size: 11 * k, color: softInk, height: h)
    put("bits of flesh → infrabuccal pocket", ctx, x: solidX, top: footY, size: 12.5 * k, bold: true, color: solidInk, height: h)
    put("strained out, packed into a pellet, spat out", ctx, x: solidX, top: footY + 17 * k, size: 11 * k, color: softInk, height: h)
    ctx.setLineWidth(1 * k)
    ctx.setStrokeColor(juiceInk)
    line(ctx, CGPoint(x: juiceX + 60 * k, y: footY - 3 * k), CGPoint(x: crop.x, y: crop.y + cry * 0.8), height: h)
    ctx.setStrokeColor(solidInk)
    line(ctx, CGPoint(x: solidX + 150 * k, y: footY - 3 * k), CGPoint(x: pocket.x, y: pocket.y + prad), height: h)
}

func annotate(_ image: AntImage, scene: Scene, frame: FrameState) {
    let w: Int = image.width
    let hI: Int = image.height
    guard let ctx = CGContext(data: image.pixels.contents(), width: w, height: hI, bitsPerComponent: 8,
                              bytesPerRow: w * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                              bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return }
    let h: CGFloat = CGFloat(hI)
    let k: CGFloat = h / 900

    put("Lasius niger", ctx, x: 50 * k, top: 30 * k, size: 24 * k, italic: true, height: h)
    let lw: CGFloat = textWidth("Lasius niger", size: 24 * k, bold: false, italic: true)
    put(titleText, ctx, x: 50 * k + lw, top: 30 * k, size: 24 * k, height: h)
    for (i, l) in captionLines().enumerated() {
        put(l, ctx, x: 50 * k, top: (66 + 19 * CGFloat(i)) * k, size: 13.5 * k, color: softInk, height: h)
    }

    // 1 mm bar.
    let mmPx: CGFloat = CGFloat(1.0 / mainMillimetresPerPixel(width: w))
    let bar: PixelRect = scaleBarBlock(height: hI)
    let barY: CGFloat = CGFloat(bar.y1) - 8 * k
    ctx.setFillColor(ink)
    ctx.fill(CGRect(x: 50 * k, y: h - barY, width: mmPx, height: 3 * k))
    ctx.fill(CGRect(x: 50 * k, y: h - barY - 3 * k, width: 2 * k, height: 9 * k))
    ctx.fill(CGRect(x: 50 * k + mmPx - 2 * k, y: h - barY - 3 * k, width: 2 * k, height: 9 * k))
    let bw: CGFloat = textWidth("1 mm", size: 14 * k, bold: true)
    put("1 mm", ctx, x: 50 * k + mmPx / 2 - bw / 2, top: barY - 22 * k, size: 14 * k, bold: true, height: h)

    // The magnified view of the jaws: a callout from the jaws in the main
    // view, a ring, and the words under it.
    let zc = CGPoint(x: CGFloat(zoomCentre.x) * h, y: CGFloat(zoomCentre.y) * h)
    let zr: CGFloat = CGFloat(zoomRadius) * h
    let jaw: SIMD2<Float> = projectMain(zoomLookAt, width: w, height: hI)
    let jp = CGPoint(x: CGFloat(jaw.x), y: CGFloat(jaw.y))
    let spot: CGFloat = 24 * k
    ctx.setStrokeColor(softInk)
    ctx.setLineWidth(1.5 * k)
    for (a, b) in tangents(jp, spot, zc, zr) { line(ctx, a, b, height: h) }
    ctx.setStrokeColor(ink)
    ctx.setLineWidth(2 * k)
    ctx.strokeEllipse(in: CGRect(x: jp.x - spot, y: h - jp.y - spot, width: 2 * spot, height: 2 * spot))
    ctx.setStrokeColor(CGColor(srgbRed: 1, green: 1, blue: 1, alpha: 1))
    ctx.setLineWidth(6 * k)
    ctx.strokeEllipse(in: CGRect(x: zc.x - zr, y: h - zc.y - zr, width: 2 * zr, height: 2 * zr))
    ctx.setStrokeColor(ink)
    ctx.setLineWidth(1.5 * k)
    ctx.strokeEllipse(in: CGRect(x: zc.x - zr - 3.5 * k, y: h - zc.y - zr - 3.5 * k, width: 2 * zr + 7 * k, height: 2 * zr + 7 * k))
    // Its own 0.1 mm bar, inside the circle's foot.
    let zPxPerMM: CGFloat = 2 * zr / CGFloat(zoomField)
    let zb: CGFloat = 0.1 * zPxPerMM
    let zby: CGFloat = zc.y + zr - 30 * k
    ctx.setFillColor(plate)
    ctx.fill(CGRect(x: zc.x - zb / 2 - 10 * k, y: h - zby - 10 * k, width: zb + 20 * k, height: 36 * k))
    ctx.setFillColor(ink)
    ctx.fill(CGRect(x: zc.x - zb / 2, y: h - zby, width: zb, height: 3 * k))
    let zbl: CGFloat = textWidth("0.1 mm", size: 13 * k, bold: true)
    put("0.1 mm", ctx, x: zc.x - zbl / 2, top: zby - 20 * k, size: 13 * k, bold: true, height: h)
    let zl: PixelRect = zoomLabelBlock(height: hI)
    ctx.setFillColor(plate)
    ctx.fill(CGRect(x: CGFloat(zl.x0), y: h - CGFloat(zl.y1), width: CGFloat(zl.x1 - zl.x0), height: CGFloat(zl.y1 - zl.y0)))
    let mag: Float = zoomMagnification(width: w, height: hI)
    let zTitle: String = String(format: "the mandibles, from above (×%.1f)", mag)
    put(zTitle, ctx, x: zc.x - textWidth(zTitle, size: 14 * k, bold: true) / 2, top: CGFloat(zl.y0) + 6 * k, size: 14 * k,
        bold: true, height: h)
    let deg: Float = frame.ant.opening * 180 / Float.pi
    let sub: String = frame.ant.gripping ? "closed on the flesh, as far as it lets them"
        : String(format: "each turned %.0f° out from rest", deg)
    put(sub, ctx, x: zc.x - textWidth(sub, size: 12 * k, bold: false) / 2, top: CGFloat(zl.y0) + 26 * k, size: 12 * k,
        color: softInk, height: h)
    drawInset(ctx, frame: frame, height: hI)
}
