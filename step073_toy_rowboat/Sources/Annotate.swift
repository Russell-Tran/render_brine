// What the CPU draws over the GPU's picture: the title and the argument, a
// box on the boat where the cut is taken, the inset — that cut, drawn to
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

/// The cut: a point on the plane of the section and the axes the inset maps
/// onto (right, and down), set by the boat's file.
struct Cut {
    var centre: SIMD3<Float>
    var right: SIMD3<Float>
    var down: SIMD3<Float>
    /// The section's outline on the toy, for marking where the cut is.
    var marks: [SIMD3<Float>] = []
}

/// A world point for an inset pixel (fractional pixel coordinates).
func insetPoint(_ px: CGFloat, _ py: CGFloat, cut: Cut, height: Int) -> SIMD3<Float> {
    let r: CGRect = insetRect(height: height)
    let mm: Float = insetMillimetresPerPixel(height: height)
    let u: Float = Float(px - r.midX) * mm
    let v: Float = Float(py - r.midY) * mm
    return cut.centre + cut.right * u + cut.down * v
}

/// Where a world point on the cut's plane lands in the inset, in pixels.
func insetPixel(_ p: SIMD3<Float>, cut: Cut, height: Int) -> CGPoint {
    let r: CGRect = insetRect(height: height)
    let mm: Float = insetMillimetresPerPixel(height: height)
    let d: SIMD3<Float> = p - cut.centre
    let u: Float = simd_dot(d, cut.right) / mm
    let v: Float = simd_dot(d, cut.down) / mm
    return CGPoint(x: r.midX + CGFloat(u), y: r.midY + CGFloat(v))
}

/// What the inset shows at a point: air, or the PVC. The paint over it, 25
/// µm, is under a pixel at the inset's scale and is said so in words.
enum Region { case air, plastic }

func insetRegion(_ toy: PosedToy, _ p: SIMD3<Float>) -> Region {
    toy.sdf(p).d > 0 ? .air : .plastic
}

/// The inset's background, darker than white paint so a pale toy's cut
/// shows (step 62). MODEL.
let insetAir = SIMD3<Float>(0.50, 0.53, 0.58)

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

func drawBar(_ ctx: CGContext, _ b: ScaleBar, k: CGFloat, height h: CGFloat, color: CGColor = ink) {
    ctx.setFillColor(color)
    ctx.fill(flip(CGRect(x: b.x, y: b.y, width: b.pixels, height: 3 * k), h))
    ctx.fill(flip(CGRect(x: b.x, y: b.y - 6 * k, width: 2 * k, height: 9 * k), h))
    ctx.fill(flip(CGRect(x: b.x + b.pixels - 2 * k, y: b.y - 6 * k, width: 2 * k, height: 9 * k), h))
    let w: CGFloat = textWidth(b.label, size: 15 * k, bold: true)
    put(b.label, ctx, x: b.x + b.pixels / 2 - w / 2, top: b.y - 26 * k, size: 15 * k, bold: true, color: color, height: h)
}

/// A point on the parting line to point at: marched from the caption's
/// start point along its direction onto the surface.
func seamPointer(_ s: StillSetup, caption: Caption) -> SIMD3<Float> {
    var p: SIMD3<Float> = caption.seamStart
    let dir: SIMD3<Float> = simd_normalize(caption.seamDirection)
    for _ in 0..<400 {
        let d: Float = s.toy.sdf(p).d
        if d < 0.002 { break }
        p += dir * (d * 0.9)
    }
    return p
}

// MARK: - drawing

/// The title and the paragraph under it, per toy.
struct Caption {
    var title: String
    var lines: [String]
    var cutName: String
    /// Where the pointer to the parting line lands: marched from a world
    /// point along a direction onto the surface; and where its label sits,
    /// in 1080p pixels from that point.
    var seamStart: SIMD3<Float>
    var seamDirection: SIMD3<Float>
    var seamLabel: CGPoint
}

func annotate(_ image: ToyImage, setup s: StillSetup, caption: Caption) {
    let w: Int = image.width
    let hI: Int = image.height
    let buf = image.pixels.contents().assumingMemoryBound(to: UInt8.self)

    // The inset's pixels first, straight into the image: each pixel's centre
    // classified by the toy's own distance function.
    let r: CGRect = insetRect(height: hI)
    let cut: Cut = s.cut
    func bytes(_ c: SIMD3<Float>) -> SIMD3<UInt8> {
        SIMD3<UInt8>(UInt8((srgbEncode(Double(c.x)) * 255).rounded()), UInt8((srgbEncode(Double(c.y)) * 255).rounded()),
                     UInt8((srgbEncode(Double(c.z)) * 255).rounded()))
    }
    let air: SIMD3<UInt8> = bytes(insetAir)
    let pvc: SIMD3<UInt8> = bytes(massColour)
    for y in Int(r.minY)..<min(Int(r.maxY), hI) {
        for x in Int(r.minX)..<min(Int(r.maxX), w) {
            let p: SIMD3<Float> = insetPoint(CGFloat(x) + 0.5, CGFloat(y) + 0.5, cut: cut, height: hI)
            let c: SIMD3<UInt8> = insetRegion(s.toy, p) == .air ? air : pvc
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

    // Where the cut is taken: the section's outline drawn on the toy, and
    // lines from its highest and lowest points to the inset's corners.
    let marks: [CGPoint] = cut.marks.map { m in
        let q: SIMD2<Float> = project(m, width: w, height: hI, cam: s.camera)
        return CGPoint(x: CGFloat(q.x), y: CGFloat(q.y))
    }
    if let topPt = marks.min(by: { $0.y < $1.y }), let lowPt = marks.max(by: { $0.y < $1.y }) {
        ctx.setStrokeColor(softInk)
        ctx.setLineWidth(1.5 * k)
        line(ctx, topPt, CGPoint(x: r.minX, y: r.minY), height: h)
        line(ctx, lowPt, CGPoint(x: r.minX, y: r.maxY), height: h)
        // Only the part of the outline the camera sees: a mark is seen when
        // the ray through its pixel stops at it (the kernel's ray length).
        let seen: [Bool] = cut.marks.enumerated().map { (i, m) in
            let px: Int = min(max(Int(marks[i].x), 0), w - 1)
            let py: Int = min(max(Int(marks[i].y), 0), hI - 1)
            let depth: Float = simd_distance(m, s.camera.position)
            return abs(image.light(px, py).w - depth) < 0.35
        }
        ctx.setStrokeColor(ink)
        ctx.setLineWidth(2.5 * k)
        ctx.setLineDash(phase: 0, lengths: [7 * k, 5 * k])
        for i in 1..<marks.count where seen[i] && seen[i - 1] {
            line(ctx, marks[i - 1], marks[i], height: h)
        }
        ctx.setLineDash(phase: 0, lengths: [])
    }

    // The inset's frame, heading, the boat's own notes, and the bar.
    ctx.setStrokeColor(ink)
    ctx.setLineWidth(3 * k)
    ctx.stroke(flip(r.insetBy(dx: -1.5 * k, dy: -1.5 * k), h))
    put("Cut across the \(caption.cutName), true to scale", ctx, x: r.minX, top: r.minY - 30 * k, size: 17 * k, bold: true, height: h)
    drawInsetNotes(ctx, setup: s, rect: r, k: k, height: h)
    drawBar(ctx, insetScaleBar(height: hI), k: k, height: h)

    // The parting line on the toy itself.
    let sp: SIMD2<Float> = project(seamPointer(s, caption: caption), width: w, height: hI, cam: s.camera)
    let spc = CGPoint(x: CGFloat(sp.x), y: CGFloat(sp.y))
    let label = CGPoint(x: spc.x + caption.seamLabel.x * k, y: spc.y + caption.seamLabel.y * k)
    ctx.setStrokeColor(softInk)
    ctx.setLineWidth(1.4 * k)
    let lw: CGFloat = textWidth("parting line", size: 15 * k, bold: true)
    let fromX: CGFloat = caption.seamLabel.x < 0 ? label.x + lw + 6 * k : label.x - 6 * k
    line(ctx, CGPoint(x: fromX, y: label.y + 10 * k), CGPoint(x: spc.x, y: spc.y), height: h)
    put("parting line", ctx, x: label.x, top: label.y, size: 15 * k, bold: true, height: h)

    let bar: ScaleBar = mainScaleBar(s, width: w, height: hI)
    drawBar(ctx, bar, k: k, height: h)
    put("at the toy", ctx, x: bar.x + bar.pixels + 12 * k, top: bar.y - 7 * k, size: 13 * k, color: softInk, height: h)
}
