// What the CPU draws under the GPU's picture, straight into the same memory:
// the caption bar, and this step's EVIDENCE BAR.
//
// The evidence bar is the point of step 10. Every earlier render stood on
// measured structure, so it needed no such thing. Here the envelope's
// dimensions are measured, the electroporation pore is what simulations
// produce, and the calcium mechanism has never been observed at all — three
// very different things in one picture. The bar says which is which, on screen,
// while it is being shown: a filled square for measured, a half square for
// simulated, a hollow one for a model.

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

/// How well the thing on screen is actually known.
enum Evidence: Int {
    case measured = 2        // dimensions from published measurements of real cells
    case simulated = 1       // a shape molecular dynamics produces
    case model = 0           // a mechanism inferred from bulk behaviour, never seen

    var label: String {
        switch self {
        case .measured: return "MEASURED"
        case .simulated: return "SIMULATED"
        case .model: return "MODEL"
        }
    }
    var color: CGColor {
        switch self {
        case .measured: return CGColor(srgbRed: 0.50, green: 0.84, blue: 0.62, alpha: 1)
        case .simulated: return CGColor(srgbRed: 0.90, green: 0.76, blue: 0.40, alpha: 1)
        case .model: return CGColor(srgbRed: 0.92, green: 0.55, blue: 0.48, alpha: 1)
        }
    }
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

/// The evidence bar, drawn over the picture in a corner. Three marks, one per
/// level, with the active one filled and named. Small on purpose: it should be
/// readable without competing with the render.
func drawEvidence(_ level: Evidence, note: String, into buffer: MTLBuffer, layout: FrameLayout,
                  atTop: Bool = true, x originX: CGFloat? = nil, top topOverride: CGFloat? = nil) {
    guard let ctx = CGContext(data: buffer.contents(), width: layout.width, height: layout.height,
                              bitsPerComponent: 8, bytesPerRow: layout.width * 4,
                              space: CGColorSpace(name: CGColorSpace.sRGB)!,
                              bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return }
    let k = layout.scale
    let h = CGFloat(layout.height)
    let x0 = originX ?? 18 * k
    let top = topOverride ?? (atTop ? 16 * k : CGFloat(layout.viewHeight) - 44 * k)
    let box: CGFloat = 8 * k
    let gap: CGFloat = 4 * k

    // Three marks, lowest evidence on the left, so the bar reads as a scale.
    for (i, lvl) in [Evidence.model, .simulated, .measured].enumerated() {
        let rect = CGRect(x: x0 + CGFloat(i) * (box + gap), y: h - top - box, width: box, height: box)
        ctx.setLineWidth(1.2 * k)
        if lvl.rawValue <= level.rawValue {
            ctx.setFillColor(level.color)
            ctx.fill(rect)
        } else {
            ctx.setStrokeColor(CGColor(srgbRed: 0.78, green: 0.84, blue: 0.87, alpha: 0.55))
            ctx.stroke(rect.insetBy(dx: 0.6 * k, dy: 0.6 * k))
        }
    }
    let labelX = x0 + 3 * (box + gap) + 4 * k
    text(level.label, ctx, x: labelX, top: top - 1 * k, size: 9.5 * k, bold: true,
         color: level.color, height: h)
    if !note.isEmpty {
        text(note, ctx, x: x0, top: top + box + 11 * k, size: 9 * k,
             color: CGColor(srgbRed: 0.84, green: 0.89, blue: 0.92, alpha: 0.92), height: h)
    }
}

/// A short label placed over the picture, for the three panels of GIF 2.
func drawPanelLabel(_ s: String, into buffer: MTLBuffer, layout: FrameLayout,
                    centerX: CGFloat, top: CGFloat, size: CGFloat, bold: Bool = true,
                    color: CGColor? = nil) {
    guard let ctx = CGContext(data: buffer.contents(), width: layout.width, height: layout.height,
                              bitsPerComponent: 8, bytesPerRow: layout.width * 4,
                              space: CGColorSpace(name: CGColorSpace.sRGB)!,
                              bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return }
    let h = CGFloat(layout.height)
    let w = textWidth(s, size: size, bold: bold)
    text(s, ctx, x: centerX - w / 2, top: top, size: size, bold: bold, color: color ?? ink, height: h)
}

/// A vertical rule between panels.
func drawDivider(into buffer: MTLBuffer, layout: FrameLayout, x: CGFloat) {
    guard let ctx = CGContext(data: buffer.contents(), width: layout.width, height: layout.height,
                              bitsPerComponent: 8, bytesPerRow: layout.width * 4,
                              space: CGColorSpace(name: CGColorSpace.sRGB)!,
                              bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return }
    let h = CGFloat(layout.height)
    ctx.setFillColor(CGColor(srgbRed: 0.05, green: 0.08, blue: 0.11, alpha: 0.55))
    ctx.fill(CGRect(x: x - 1, y: h - CGFloat(layout.viewHeight), width: 2, height: CGFloat(layout.viewHeight)))
}
