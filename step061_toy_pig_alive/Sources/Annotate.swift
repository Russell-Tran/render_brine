// What the CPU draws over the GPU's picture: the title and the argument, a
// ring on the leg where the cut is taken, the inset — that cut, drawn to
// scale from the same distance function the GPU drew — a pointer to the
// parting line, and two scale bars. Text is step 42's CoreText approach, as
// step 53 copied it.

import CoreGraphics
import CoreText
import Foundation
import Metal
import simd

let ink = CGColor(srgbRed: 0.10, green: 0.11, blue: 0.12, alpha: 1)
let softInk = CGColor(srgbRed: 0.10, green: 0.11, blue: 0.12, alpha: 0.66)
let paper = CGColor(srgbRed: 1, green: 1, blue: 1, alpha: 1)

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

func cgLinear(_ c: SIMD3<Float>) -> CGColor {
    CGColor(srgbRed: CGFloat(srgbEncode(Double(c.x))), green: CGFloat(srgbEncode(Double(c.y))),
            blue: CGFloat(srgbEncode(Double(c.z))), alpha: 1)
}

// MARK: - geometry the tests read

func insetRect(height: Int) -> CGRect {
    let h: CGFloat = CGFloat(height)
    return CGRect(x: CGFloat(insetOrigin.x) * h, y: CGFloat(insetOrigin.y) * h,
                  width: CGFloat(insetSize.x) * h, height: CGFloat(insetSize.y) * h)
}

/// The cut: the centre of the leg's section and the axes the inset maps
/// onto (right → the toy's forward, down → its right side).
struct Cut {
    var centre: SIMD3<Float>
    var right: SIMD3<Float>
    var down: SIMD3<Float>
}

func cutFor(_ s: StillSetup) -> Cut {
    let lowerIndex: Int = s.toy.segments.firstIndex { $0.part == .lowerLeg && $0.limb == s.cutLimb } ?? 0
    let f: Frame = s.toy.frames[lowerIndex]
    let along: Float = (f.o.y - s.cutHeight) / max(f.y.y, 1e-4)
    let c: SIMD3<Float> = f.o - f.y * along
    let fwd: SIMD3<Float> = simd_normalize(SIMD3<Float>(s.pose.body.x.x, 0, s.pose.body.x.z))
    return Cut(centre: c, right: fwd, down: simd_cross(fwd, SIMD3<Float>(0, 1, 0)))
}

/// A world point for an inset pixel (fractional pixel coordinates).
func insetPoint(_ px: CGFloat, _ py: CGFloat, cut: Cut, height: Int) -> SIMD3<Float> {
    let r: CGRect = insetRect(height: height)
    let mm: Float = insetMillimetresPerPixel(height: height)
    let u: Float = Float(px - r.midX) * mm
    let v: Float = Float(py - r.midY) * mm
    return cut.centre + cut.right * u + cut.down * v
}

/// The colour the inset gives each region. The paint's own colour; the PVC
/// under it, coloured in the mass, paler (MODEL); air, the inset's paper.
enum Region { case air, paint, plastic }

func insetRegion(_ toy: PosedToy, _ p: SIMD3<Float>) -> Region {
    let d: Float = toy.sdf(p).d
    if d > 0 { return .air }
    if d > -paintMicrometres / 1000 { return .paint }
    return .plastic
}

let insetAir = SIMD3<Float>(0.86, 0.86, 0.84)

func plasticColour(_ paint: Paint) -> SIMD3<Float> { paint.albedo * 0.55 + SIMD3<Float>(0.30, 0.30, 0.28) }

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

/// A point on the parting line to point at: straight down from a point
/// above the named piece's mid-plane, onto its surface.
func seamPointer(_ s: StillSetup, caption: Caption) -> SIMD3<Float> {
    guard let i = s.toy.segments.firstIndex(where: { $0.part == caption.seamSegment }) else { return s.centre }
    let f: Frame = s.toy.frames[i]
    var p: SIMD3<Float> = f.toWorld(caption.seamStart)
    let down: SIMD3<Float> = -f.y
    for _ in 0..<300 {
        let d: Float = s.toy.sdf(p).d
        if d < 0.002 { break }
        p += down * (d * 0.9)
    }
    return p
}

// MARK: - drawing

/// The title and the paragraph under it, per toy.
struct Caption {
    var title: String
    var lines: [String]
    var cutName: String
    /// Where the pointer to the parting line lands: the piece, and a point
    /// above its mid-plane (in its frame) to drop onto the surface from.
    var seamSegment: Part
    var seamStart: SIMD3<Float>
}

func annotate(_ image: ToyImage, setup s: StillSetup, caption: Caption) {
    let w: Int = image.width
    let hI: Int = image.height
    let buf = image.pixels.contents().assumingMemoryBound(to: UInt8.self)

    // The inset's pixels first, straight into the image: each pixel's centre
    // classified by the toy's own distance function.
    let r: CGRect = insetRect(height: hI)
    let cut: Cut = cutFor(s)
    let paintIndex: Int = s.design.legs[s.cutLimb].lowerPrims[0].paint
    let paintCol: SIMD3<Float> = s.design.paints[paintIndex].albedo
    let plasticCol: SIMD3<Float> = plasticColour(s.design.paints[paintIndex])
    func bytes(_ c: SIMD3<Float>) -> SIMD3<UInt8> {
        SIMD3<UInt8>(UInt8((srgbEncode(Double(c.x)) * 255).rounded()), UInt8((srgbEncode(Double(c.y)) * 255).rounded()),
                     UInt8((srgbEncode(Double(c.z)) * 255).rounded()))
    }
    let air: SIMD3<UInt8> = bytes(insetAir)
    let pnt: SIMD3<UInt8> = bytes(paintCol)
    let pvc: SIMD3<UInt8> = bytes(plasticCol)
    for y in Int(r.minY)..<min(Int(r.maxY), hI) {
        for x in Int(r.minX)..<min(Int(r.maxX), w) {
            let p: SIMD3<Float> = insetPoint(CGFloat(x) + 0.5, CGFloat(y) + 0.5, cut: cut, height: hI)
            let c: SIMD3<UInt8>
            switch insetRegion(s.toy, p) {
            case .air: c = air
            case .paint: c = pnt
            case .plastic: c = pvc
            }
            let i: Int = (y * w + x) * 4
            buf[i] = c.x
            buf[i + 1] = c.y
            buf[i + 2] = c.z
            buf[i + 3] = 255
        }
    }

    guard let ctx = CGContext(data: image.pixels.contents(), width: w, height: hI, bitsPerComponent: 8,
                              bytesPerRow: w * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                              bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return }
    let h: CGFloat = CGFloat(hI)
    let k: CGFloat = h / 1080
    ctx.setLineCap(.round)
    ctx.setShouldAntialias(true)

    put(caption.title, ctx, x: 64 * k, top: 40 * k, size: 30 * k, bold: true, height: h)
    for (i, line) in caption.lines.enumerated() {
        put(line, ctx, x: 64 * k, top: (84 + 22 * CGFloat(i)) * k, size: 16 * k, color: softInk, height: h)
    }

    // The ring on the leg, and lines to the inset.
    let cp: SIMD2<Float> = project(cut.centre, width: w, height: hI, cam: s.camera)
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
    put("Cut across the \(caption.cutName), true to scale", ctx, x: r.minX, top: r.minY - 30 * k, size: 17 * k, bold: true, height: h)
    let um: Int = Int(paintMicrometres)
    put("paint, \(um) µm (model)", ctx, x: r.minX + 16 * k, top: r.minY + 14 * k, size: 15 * k, bold: true, height: h)
    put("solid PVC — no hollow inside", ctx, x: r.minX + 16 * k, top: r.minY + 36 * k, size: 15 * k, bold: true, height: h)
    put("the parting line, where the mould's halves met", ctx, x: r.minX + 16 * k, top: r.maxY - 64 * k, size: 13 * k,
        color: softInk, height: h)
    put("(the bump at the front and back of the cut)", ctx, x: r.minX + 16 * k, top: r.maxY - 46 * k, size: 13 * k,
        color: softInk, height: h)
    drawBar(ctx, insetScaleBar(height: hI), k: k, height: h)

    // The parting line on the toy itself.
    let sp: SIMD2<Float> = project(seamPointer(s, caption: caption), width: w, height: hI, cam: s.camera)
    let spc = CGPoint(x: CGFloat(sp.x), y: CGFloat(sp.y))
    // The label up and to the right of the point, clear of the caption.
    let label = CGPoint(x: spc.x + 60 * k, y: max(spc.y - 80 * k, 220 * k))
    ctx.setStrokeColor(softInk)
    ctx.setLineWidth(1.4 * k)
    line(ctx, CGPoint(x: label.x - 4 * k, y: label.y + 10 * k), CGPoint(x: spc.x + 2 * k, y: spc.y - 4 * k), height: h)
    put("parting line", ctx, x: label.x, top: label.y, size: 15 * k, bold: true, height: h)

    let bar: ScaleBar = mainScaleBar(s, width: w, height: hI)
    drawBar(ctx, bar, k: k, height: h)
    put("at the toy", ctx, x: bar.x + bar.pixels + 12 * k, top: bar.y - 7 * k, size: 13 * k, color: softInk, height: h)
}
