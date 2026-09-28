// What the CPU draws over the GPU's picture: the title and the caption, the
// inset (Inset.swift) with its labels, and two true scale bars. Text is step
// 42's CoreText approach, as steps 53, 60 and 70 copied it.

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

// MARK: - geometry the tests read

/// The inset, in units of image height (x, y from the top-left).
let insetOrigin = SIMD2<Float>(1.12, 0.27)
let insetSize = SIMD2<Float>(0.54, 0.54)

func insetRect(height: Int) -> CGRect {
    let h: CGFloat = CGFloat(height)
    return CGRect(x: CGFloat(insetOrigin.x) * h, y: CGFloat(insetOrigin.y) * h,
                  width: CGFloat(insetSize.x) * h, height: CGFloat(insetSize.y) * h)
}

/// Micrometres per inset pixel.
func insetMicrometresPerPixel(height: Int) -> Float { insetFieldMicrometres / (insetSize.y * Float(height)) }

/// Scale bars: 100 mm at the printer, 100 µm in the inset.
let mainBarMillimetres: Float = 100
let insetBarMicrometres: Float = 100

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
    let px: CGFloat = CGFloat(insetBarMicrometres / insetMicrometresPerPixel(height: height))
    return ScaleBar(label: String(format: "%g µm", insetBarMicrometres), x: r.maxX - px - 24 * k, y: r.maxY - 26 * k, pixels: px)
}

func drawBar(_ ctx: CGContext, _ b: ScaleBar, k: CGFloat, height h: CGFloat, backed: Bool = false) {
    if backed {
        let w: CGFloat = max(b.pixels, textWidth(b.label, size: 15 * k, bold: true)) + 16 * k
        ctx.setFillColor(CGColor(srgbRed: 1, green: 1, blue: 1, alpha: 0.85))
        ctx.fill(flip(CGRect(x: b.x + b.pixels / 2 - w / 2, y: b.y - 32 * k, width: w, height: 42 * k), h))
    }
    ctx.setFillColor(ink)
    ctx.fill(flip(CGRect(x: b.x, y: b.y, width: b.pixels, height: 3 * k), h))
    ctx.fill(flip(CGRect(x: b.x, y: b.y - 6 * k, width: 2 * k, height: 9 * k), h))
    ctx.fill(flip(CGRect(x: b.x + b.pixels - 2 * k, y: b.y - 6 * k, width: 2 * k, height: 9 * k), h))
    let w: CGFloat = textWidth(b.label, size: 15 * k, bold: true)
    put(b.label, ctx, x: b.x + b.pixels / 2 - w / 2, top: b.y - 26 * k, size: 15 * k, bold: true, height: h)
}

// MARK: - the caption

let captionTitle: String = "HP LaserJet P1102w, drawn to HP's published size"
func captionLines(_ m: Mutant = activeMutant) -> [String] {
    [String(format: "%.0f × %.0f × %.0f mm (width × depth × height), %.1f kg: HP's user guide, Table C-1 (HP marks the values preliminary).",
            hpWidth, hpDepth, hpHeight, hpWeightKilograms),
     "Input tray open as HP draws it; output bin with its extension up; control panel: two buttons, three lights (ready and wireless lit).",
     "Matte black plastic: a dark body under a rough dielectric skin (polystyrene's index). A neutral render — not HP's image; badge left plain.",
     "How it prints: a laser writes the page as charge on a drum; toner sticks where it wrote, is pressed onto the paper,",
     "and fused there by heat and pressure (Wikipedia, \"Laser printing\"). The inset shows the result at 600 dpi."]
}

// MARK: - drawing

func annotate(_ image: PrinterImage, setup s: StillSetup, inset c: InsetContent) {
    let w: Int = image.width
    let hI: Int = image.height
    guard let ctx = CGContext(data: image.pixels.contents(), width: w, height: hI, bitsPerComponent: 8,
                              bytesPerRow: w * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                              bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return }
    let h: CGFloat = CGFloat(hI)
    let k: CGFloat = h / 1080
    ctx.setShouldAntialias(true)
    let r: CGRect = insetRect(height: hI)
    drawInset(ctx, c, rect: r, imageHeight: h)

    put(captionTitle, ctx, x: 64 * k, top: 40 * k, size: 30 * k, bold: true, height: h)
    for (i, ln) in captionLines().enumerated() {
        put(ln, ctx, x: 64 * k, top: (84 + 22 * CGFloat(i)) * k, size: 16 * k, color: softInk, height: h)
    }

    ctx.setStrokeColor(ink)
    ctx.setLineWidth(3 * k)
    ctx.stroke(flip(r.insetBy(dx: -1.5 * k, dy: -1.5 * k), h))
    put("How it prints: the edge of a 12-point \"A\", true to scale", ctx, x: r.minX, top: r.minY - 30 * k, size: 17 * k,
        bold: true, height: h)
    // Labels in white boxes on the inset.
    func tag(_ lines: [String], x: CGFloat, top: CGFloat, bold: Bool = false) {
        let size: CGFloat = 13.5 * k
        var wmax: CGFloat = 0
        for l in lines { wmax = max(wmax, textWidth(l, size: size, bold: bold)) }
        let box = CGRect(x: x - 6 * k, y: top - 5 * k, width: wmax + 12 * k, height: CGFloat(lines.count) * 17 * k + 8 * k)
        ctx.setFillColor(CGColor(srgbRed: 1, green: 1, blue: 1, alpha: 0.86))
        ctx.fill(flip(box, h))
        for (i, l) in lines.enumerated() { put(l, ctx, x: x, top: top + CGFloat(i) * 17 * k, size: size, bold: bold, height: h) }
    }
    let pitch: Float = c.pitch
    tag([String(format: "blue crosses: the %.0f-dpi grid, a dot every %.1f µm", micrometresPerInch / pitch, pitch),
         "(FastRes 600; HP's FastRes 1200 is an enhancement, not shown)"], x: r.minX + 16 * k, top: r.minY + 14 * k)
    tag(["toner: particles 5–9 µm across (model; ~5 µm is what", "600 dpi needs, Wikipedia), heaped a dot at a time,",
         "then melted together by the fuser's heat and pressure"], x: r.minX + 16 * k, top: r.maxY - 118 * k)
    tag(["paper, 75 g/m²: a mat of cellulose", "fibres, the molecule of step 52's cotton"], x: r.minX + 16 * k, top: r.minY + 250 * k)
    drawBar(ctx, insetScaleBar(height: hI), k: k, height: h, backed: true)

    let bar: ScaleBar = mainScaleBar(s, width: w, height: hI)
    drawBar(ctx, bar, k: k, height: h)
    put("at the printer's middle", ctx, x: bar.x + bar.pixels + 12 * k, top: bar.y - 7 * k, size: 13 * k, color: softInk, height: h)
}
