// What the CPU draws under the GPU's picture, straight into the same memory:
// the caption bar, the parallax ladder, and the evidence bar.
//
// The evidence bar's middle mark is DERIVED here rather than SIMULATED, which
// is the honest label for this step: the bus's dimensions are published or
// regulated numbers, the parallax and the blur-against-distance relation are
// arithmetic done from them, and the roadside and the sun are inventions.
//
// The caption does not change for the whole loop, and that is deliberate. A
// counter ticking over would be a patch of pixels changing every frame in a
// render whose entire claim is that almost nothing changes.

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
    case measured = 2        // a published or regulated dimension
    case derived = 1         // arithmetic done from one of those
    case model = 0           // invented because the picture needs something there

    var label: String {
        switch self {
        case .measured: return "MEASURED"
        case .derived: return "DERIVED"
        case .model: return "MODEL"
        }
    }
    var color: CGColor {
        switch self {
        case .measured: return CGColor(srgbRed: 0.50, green: 0.84, blue: 0.62, alpha: 1)
        case .derived: return CGColor(srgbRed: 0.90, green: 0.76, blue: 0.40, alpha: 1)
        case .model: return CGColor(srgbRed: 0.92, green: 0.55, blue: 0.48, alpha: 1)
        }
    }
}

private let ink = CGColor(srgbRed: 0.92, green: 0.95, blue: 0.96, alpha: 1)
private let muted = CGColor(srgbRed: 0.60, green: 0.69, blue: 0.73, alpha: 1)
private let accent = CGColor(srgbRed: 0.45, green: 0.83, blue: 0.86, alpha: 1)
private let captionBackground = CGColor(srgbRed: 0.05, green: 0.08, blue: 0.11, alpha: 1)

private func line(_ s: String, size: CGFloat, bold: Bool, color: CGColor) -> CTLine {
    let font = CTFontCreateWithName((bold ? "HelveticaNeue-Bold" : "HelveticaNeue") as CFString,
                                    size, nil)
    let attributes: [NSAttributedString.Key: Any] = [
        NSAttributedString.Key(kCTFontAttributeName as String): font,
        NSAttributedString.Key(kCTForegroundColorAttributeName as String): color,
    ]
    return CTLineCreateWithAttributedString(NSAttributedString(string: s, attributes: attributes))
}

/// Draws text whose top-left corner is at (x, top) in top-down pixels.
private func text(_ s: String, _ ctx: CGContext, x: CGFloat, top: CGFloat, size: CGFloat,
                  bold: Bool = false, color: CGColor = ink, height: CGFloat) {
    ctx.textPosition = CGPoint(x: x, y: height - top - size)
    CTLineDraw(line(s, size: size, bold: bold, color: color), ctx)
}

func textWidth(_ s: String, size: CGFloat, bold: Bool) -> CGFloat {
    CGFloat(CTLineGetTypographicBounds(line(s, size: size, bold: bold, color: ink), nil, nil, nil))
}

private func context(_ buffer: MTLBuffer, _ layout: FrameLayout) -> CGContext? {
    CGContext(data: buffer.contents(), width: layout.width, height: layout.height,
              bitsPerComponent: 8, bytesPerRow: layout.width * 4,
              space: CGColorSpace(name: CGColorSpace.sRGB)!,
              bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
}

func drawCaption(_ caption: Caption, into buffer: MTLBuffer, layout: FrameLayout) {
    guard let ctx = context(buffer, layout) else { return }
    let k = layout.scale
    let h = CGFloat(layout.height)
    let barTop = CGFloat(layout.viewHeight)
    ctx.setFillColor(captionBackground)
    ctx.fill(CGRect(x: 0, y: 0, width: CGFloat(layout.width), height: CGFloat(layout.captionHeight)))
    text(caption.title, ctx, x: 18 * k, top: barTop + 10 * k, size: 16 * k, bold: true, height: h)
    text(caption.subtitle, ctx, x: 18 * k, top: barTop + 34 * k, size: 12 * k, color: accent, height: h)
    text(caption.facts, ctx, x: 18 * k, top: barTop + 54 * k, size: 11.5 * k, color: muted, height: h)
    let w = textWidth(caption.aside, size: 11.5 * k, bold: false)
    text(caption.aside, ctx, x: CGFloat(layout.width) - 18 * k - w, top: barTop + 13 * k,
         size: 11.5 * k, color: muted, height: h)
}

/// One rung of the parallax ladder: a distance and the angular rate that goes
/// with it. Drawn as a row so the four-hundred-to-one spread is legible as a
/// spread rather than as five numbers.
struct LadderRung {
    var label: String
    var distance: String
    var rate: String
}

func drawLadder(_ rungs: [LadderRung], into buffer: MTLBuffer, layout: FrameLayout,
                top: CGFloat) {
    guard let ctx = context(buffer, layout) else { return }
    let k = layout.scale
    let h = CGFloat(layout.height)
    let barTop = CGFloat(layout.viewHeight)
    let left: CGFloat = 18 * k
    let usable: CGFloat = CGFloat(layout.width) - 36 * k
    let columnWidth: CGFloat = usable / CGFloat(max(rungs.count, 1))
    text("PARALLAX, ABEAM   ω = v / d", ctx, x: left, top: barTop + top - 14 * k, size: 8.5 * k,
         bold: true, color: muted, height: h)
    for (i, rung) in rungs.enumerated() {
        let x = left + CGFloat(i) * columnWidth
        text(rung.label, ctx, x: x, top: barTop + top, size: 9 * k, color: muted, height: h)
        text(rung.distance, ctx, x: x, top: barTop + top + 11 * k, size: 11 * k, bold: true,
             color: ink, height: h)
        let dw = textWidth(rung.distance, size: 11 * k, bold: true)
        text(rung.rate, ctx, x: x + dw + 8 * k, top: barTop + top + 11 * k, size: 11 * k,
             bold: true, color: accent, height: h)
    }
}

/// The evidence bar. Three marks, one per level, with the active one filled and
/// named. Small on purpose: it should be readable without competing with the
/// render.
func drawEvidence(_ level: Evidence, note: String, into buffer: MTLBuffer, layout: FrameLayout,
                  x originX: CGFloat? = nil, top topOverride: CGFloat? = nil) {
    guard let ctx = context(buffer, layout) else { return }
    let k = layout.scale
    let h = CGFloat(layout.height)
    let x0 = originX ?? 18 * k
    let top = topOverride ?? 16 * k
    let box: CGFloat = 8 * k
    let gap: CGFloat = 4 * k

    // Three marks, least evidence on the left, so the bar reads as a scale.
    for (i, lvl) in [Evidence.model, .derived, .measured].enumerated() {
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

/// A soft dark plate, so the evidence bar stays readable against a bright
/// window.
func drawPlate(into buffer: MTLBuffer, layout: FrameLayout, rect: CGRect) {
    guard let ctx = context(buffer, layout) else { return }
    let h = CGFloat(layout.height)
    let flipped = CGRect(x: rect.origin.x, y: h - rect.origin.y - rect.height,
                         width: rect.width, height: rect.height)
    ctx.setFillColor(CGColor(srgbRed: 0.04, green: 0.08, blue: 0.13, alpha: 0.46))
    ctx.addPath(CGPath(roundedRect: flipped, cornerWidth: 5, cornerHeight: 5, transform: nil))
    ctx.fillPath()
}

/// A short label placed over the picture.
func drawPanelLabel(_ s: String, into buffer: MTLBuffer, layout: FrameLayout,
                    centerX: CGFloat, top: CGFloat, size: CGFloat, bold: Bool = true,
                    color: CGColor? = nil) {
    guard let ctx = context(buffer, layout) else { return }
    let h = CGFloat(layout.height)
    let w = textWidth(s, size: size, bold: bold)
    text(s, ctx, x: centerX - w / 2, top: top, size: size, bold: bold, color: color ?? ink, height: h)
}
