// What the CPU draws under the GPU's picture, straight into the same memory:
// the caption bar in house style, and this step's EVIDENCE BAR.
//
// Step 10's evidence bar rated ONE thing at a time on a scale from measured to
// model. That will not do here. A swimming brine shrimp is a mixture: the body
// plan is counted from published anatomy, the swimming speed was measured with
// a stopwatch and a ruler, the beat frequency is extrapolated from larvae, and
// every σ in the picture is a number chosen to make it look like a micrograph.
// Those are not three points on one scale, they are three different KINDS of
// claim, so the bar here shows all three at once with what belongs to each.
//
// Nothing in the render is simulated — there is no fluid solver — so this step
// replaces step 10's SIMULATED with DERIVED: numbers that follow from the other
// two by arithmetic done in this code, with nothing added.

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

/// What kind of claim a number on screen is.
enum Evidence: Int {
    case measured = 2        // counted or measured and published by somebody
    case derived = 1         // arithmetic this code does on the measured numbers
    case model = 0           // a choice, made here, to make the picture

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

/// One row of the evidence bar: a filled square, the level's name, and what in
/// this picture belongs to it.
struct EvidenceRow {
    var level: Evidence
    var note: String
}

/// The evidence bar, three rows down the right-hand side of the caption bar.
/// Same marks, same colours and same type sizes as step 10's, laid out as a
/// legend rather than a scale because these three are not ranked.
func drawEvidenceBar(_ rows: [EvidenceRow], into buffer: MTLBuffer, layout: FrameLayout,
                     leftEdge: CGFloat) {
    guard let ctx = context(buffer, layout) else { return }
    let k = layout.scale
    let h = CGFloat(layout.height)
    let barTop = CGFloat(layout.viewHeight)
    let box: CGFloat = 8 * k
    let rowStep: CGFloat = 17 * k
    for (i, row) in rows.enumerated() {
        let top: CGFloat = barTop + 14 * k + CGFloat(i) * rowStep
        let rect = CGRect(x: leftEdge, y: h - top - box, width: box, height: box)
        ctx.setFillColor(row.level.color)
        ctx.fill(rect)
        text(row.level.label, ctx, x: leftEdge + box + 6 * k, top: top - 0.5 * k, size: 9.5 * k,
             bold: true, color: row.level.color, height: h)
        text(row.note, ctx, x: leftEdge + box + 6 * k + 58 * k, top: top - 0.5 * k, size: 9.5 * k,
             color: muted, height: h)
    }
}

/// A short label over the picture, for the scale bar's caption and the
/// anatomical pointers.
func drawLabel(_ s: String, into buffer: MTLBuffer, layout: FrameLayout,
               x: CGFloat, top: CGFloat, size: CGFloat, bold: Bool = false,
               color: CGColor? = nil, centred: Bool = false) {
    guard let ctx = context(buffer, layout) else { return }
    let h = CGFloat(layout.height)
    let w = centred ? textWidth(s, size: size, bold: bold) : 0
    text(s, ctx, x: x - w / 2, top: top, size: size, bold: bold,
         color: color ?? CGColor(srgbRed: 0.10, green: 0.22, blue: 0.28, alpha: 1), height: h)
}

/// A scale bar drawn on the picture in dark ink, because on a bright field
/// white rules vanish. `lengthPixels` is worked out from the same microns-per-
/// pixel the camera uses, so the bar cannot drift from the render.
func drawScaleBar(into buffer: MTLBuffer, layout: FrameLayout, lengthPixels: CGFloat,
                  label: String, x: CGFloat, bottom: CGFloat) {
    guard let ctx = context(buffer, layout) else { return }
    let k = layout.scale
    let h = CGFloat(layout.height)
    let ink = CGColor(srgbRed: 0.08, green: 0.20, blue: 0.26, alpha: 0.92)
    ctx.setFillColor(ink)
    ctx.fill(CGRect(x: x, y: h - bottom, width: lengthPixels, height: 3 * k))
    ctx.fill(CGRect(x: x, y: h - bottom - 4 * k, width: 2 * k, height: 11 * k))
    ctx.fill(CGRect(x: x + lengthPixels - 2 * k, y: h - bottom - 4 * k, width: 2 * k, height: 11 * k))
    let w = textWidth(label, size: 11 * k, bold: true)
    text(label, ctx, x: x + lengthPixels / 2 - w / 2, top: bottom - 20 * k, size: 11 * k,
         bold: true, color: ink, height: h)
}
