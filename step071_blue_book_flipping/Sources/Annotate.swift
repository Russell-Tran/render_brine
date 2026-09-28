// What the CPU draws over each frame: the title, the caption that says the
// motion is fiction and the bending is not, and a scale bar. Text is step
// 42's CoreText approach, as steps 53, 60 and 70 copied it.

import CoreGraphics
import CoreText
import Foundation
import Metal
import simd

let ink = CGColor(srgbRed: 0.10, green: 0.11, blue: 0.12, alpha: 1)
let softInk = CGColor(srgbRed: 0.10, green: 0.11, blue: 0.12, alpha: 0.70)

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

func flip(_ r: CGRect, _ h: CGFloat) -> CGRect {
    CGRect(x: r.minX, y: h - r.maxY, width: r.width, height: r.height)
}

/// The caption: the fiction said plainly, and what is not fiction.
let captionTitle: String = "The blue book turns its own pages"
let captionLines: [String] = [
    "The pages turn by themselves — fiction.",
    "The paper bends as paper does: every page is a",
    "developable surface, the flat leaf, never stretched.",
]

/// The box the caption's text covers, in pixels (x, y from the top-left),
/// for a frame `height` tall.
func captionBox(height: Int) -> CGRect {
    let k: CGFloat = CGFloat(height) / 360
    var w: CGFloat = textWidth(captionTitle, size: 15 * k, bold: true)
    for ln in captionLines { w = max(w, textWidth(ln, size: 10.5 * k, bold: false)) }
    let lines: CGFloat = CGFloat(captionLines.count - 1)
    let spaced: CGFloat = 13 * lines
    let lastLine: CGFloat = 10.5 * 1.3
    let bottom: CGFloat = (30 + spaced + lastLine) * k
    return CGRect(x: 14 * k, y: 10 * k, width: w, height: bottom - 10 * k)
}

/// The scale bar: 50 mm, square to the camera at the spine's head.
let barMillimetres: Float = 50

struct ScaleBar {
    var x: CGFloat
    var y: CGFloat
    var pixels: CGFloat
}

func scaleBar(_ s: Shot, width: Int, height: Int) -> ScaleBar {
    let k: CGFloat = CGFloat(height) / 360
    let px: CGFloat = CGFloat(barMillimetres * pixelsPerMillimetre(at: s.centre, height: height, cam: s.camera))
    return ScaleBar(x: 14 * k, y: CGFloat(height) - 16 * k, pixels: px)
}

func annotate(_ pixels: MTLBuffer, width w: Int, height hI: Int, shot s: Shot) {
    guard let ctx = CGContext(data: pixels.contents(), width: w, height: hI, bitsPerComponent: 8,
                              bytesPerRow: w * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                              bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return }
    let h: CGFloat = CGFloat(hI)
    let k: CGFloat = h / 360
    ctx.setShouldAntialias(true)
    put(captionTitle, ctx, x: 14 * k, top: 10 * k, size: 15 * k, bold: true, height: h)
    for (i, ln) in captionLines.enumerated() {
        put(ln, ctx, x: 14 * k, top: (30 + 13 * CGFloat(i)) * k, size: 10.5 * k, color: softInk, height: h)
    }
    let b: ScaleBar = scaleBar(s, width: w, height: hI)
    ctx.setFillColor(ink)
    ctx.fill(flip(CGRect(x: b.x, y: b.y, width: b.pixels, height: 2 * k), h))
    ctx.fill(flip(CGRect(x: b.x, y: b.y - 4 * k, width: 1.5 * k, height: 6 * k), h))
    ctx.fill(flip(CGRect(x: b.x + b.pixels - 1.5 * k, y: b.y - 4 * k, width: 1.5 * k, height: 6 * k), h))
    let label: String = String(format: "%g mm", barMillimetres)
    put(label, ctx, x: b.x + b.pixels + 6 * k, top: b.y - 8 * k, size: 10 * k, bold: true, height: h)
}
