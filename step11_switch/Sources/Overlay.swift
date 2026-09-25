// What the CPU draws under the GPU's picture, straight into the same memory:
// the caption bar.

import CoreGraphics
import CoreText
import Foundation
import Metal

struct FrameLayout {
    let width: Int
    let viewHeight: Int      // the ray-traced view
    let captionHeight: Int   // the text bar under it
    var height: Int { viewHeight + captionHeight }
    var scale: CGFloat { CGFloat(width) / 640 }   // laid out for a 640-wide frame
}

struct Caption {
    var title: String
    var subtitle: String
    var facts: String
    var aside: String        // right-aligned, next to the title
}

private let ink = CGColor(srgbRed: 0.92, green: 0.95, blue: 0.96, alpha: 1)
private let muted = CGColor(srgbRed: 0.60, green: 0.69, blue: 0.73, alpha: 1)
private let accent = CGColor(srgbRed: 0.45, green: 0.83, blue: 0.86, alpha: 1)
private let captionBackground = CGColor(srgbRed: 0.05, green: 0.08, blue: 0.11, alpha: 1)

private func line(_ s: String, size: CGFloat, bold: Bool, color: CGColor) -> CTLine {
    let font = CTFontCreateWithName((bold ? "HelveticaNeue-Bold" : "HelveticaNeue") as CFString, size, nil)
    let attributes: [NSAttributedString.Key: Any] = [
        NSAttributedString.Key(kCTFontAttributeName as String): font,
        NSAttributedString.Key(kCTForegroundColorAttributeName as String): color,
    ]
    return CTLineCreateWithAttributedString(NSAttributedString(string: s, attributes: attributes))
}

/// Draws text whose top-left corner is at (x, top) in top-down pixels.
private func text(_ s: String, _ ctx: CGContext, x: CGFloat, top: CGFloat, size: CGFloat, bold: Bool = false,
                  color: CGColor = ink, height: CGFloat) {
    ctx.textPosition = CGPoint(x: x, y: height - top - size)
    CTLineDraw(line(s, size: size, bold: bold, color: color), ctx)
}

func textWidth(_ s: String, size: CGFloat, bold: Bool) -> CGFloat {
    CGFloat(CTLineGetTypographicBounds(line(s, size: size, bold: bold, color: ink), nil, nil, nil))
}

func drawCaption(_ caption: Caption, into buffer: MTLBuffer, layout: FrameLayout) {
    guard let ctx = CGContext(data: buffer.contents(), width: layout.width, height: layout.height,
                              bitsPerComponent: 8, bytesPerRow: layout.width * 4,
                              space: CGColorSpace(name: CGColorSpace.sRGB)!,
                              bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return }
    let k = layout.scale
    let h = CGFloat(layout.height)
    let barTop = CGFloat(layout.viewHeight)
    ctx.setFillColor(captionBackground)
    ctx.fill(CGRect(x: 0, y: 0, width: CGFloat(layout.width), height: CGFloat(layout.captionHeight)))
    text(caption.title, ctx, x: 18 * k, top: barTop + 10 * k, size: 16 * k, bold: true, height: h)
    text(caption.subtitle, ctx, x: 18 * k, top: barTop + 34 * k, size: 12 * k, color: accent, height: h)
    text(caption.facts, ctx, x: 18 * k, top: barTop + 54 * k, size: 11.5 * k, color: muted, height: h)
    let w = textWidth(caption.aside, size: 11.5 * k, bold: false)
    text(caption.aside, ctx, x: CGFloat(layout.width) - 18 * k - w, top: barTop + 13 * k, size: 11.5 * k,
         color: muted, height: h)
}
