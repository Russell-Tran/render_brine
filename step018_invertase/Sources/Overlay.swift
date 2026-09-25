// What the CPU draws over the GPU's picture, straight into the same memory:
// the caption bar in house style, the two panel labels, and the polarimeter's
// dial.
//
// There is NO evidence bar on the frame. Step 13 dropped it and was right to:
// a bar saying the same three things for a hundred frames is furniture. The
// provenance lives in the constants table `make run` prints, and in the tests.
//
// The dial is not decoration. It shows the number the render computed from the
// molecules drawn in the cell, and the scale it sits on is drawn from the two
// derived end points, so the zero mark lands where the arithmetic puts it —
// about three quarters of the way along, not halfway.

import CoreGraphics
import CoreText
import Foundation
import Metal
import simd

struct FrameLayout {
    let width: Int
    let viewHeight: Int      // the ray-traced view
    let captionHeight: Int   // the text bar under it
    var height: Int { viewHeight + captionHeight }
    var scale: CGFloat { CGFloat(width) / 840 }   // laid out for an 840-wide frame
}

struct Caption {
    var title: String
    var subtitle: String
    var facts: String
    var aside: String        // right-aligned, next to the title

    /// Everything the bar says, for the tests that insist the resolution and
    /// the two rotations are on it.
    var text: String { [title, subtitle, facts, aside].joined(separator: " · ") }
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

func cgColour(_ c: SIMD3<Float>) -> CGColor {
    CGColor(srgbRed: CGFloat(gammaEncode(c.x)), green: CGFloat(gammaEncode(c.y)),
            blue: CGFloat(gammaEncode(c.z)), alpha: 1)
}

/// A label over one panel.
struct PanelLabel {
    var name: String
    var detail: String
    var colour: CGColor
    var centreX: CGFloat
}

func drawPanelLabels(_ labels: [PanelLabel], into buffer: MTLBuffer, layout: FrameLayout) {
    guard let ctx = context(buffer, layout) else { return }
    let k = layout.scale
    let h = CGFloat(layout.height)
    let shadow = CGColor(srgbRed: 0, green: 0.02, blue: 0.05, alpha: 0.95)
    for label in labels {
        ctx.saveGState()
        ctx.setShadow(offset: .zero, blur: 7 * k, color: shadow)
        let nameSize: CGFloat = 15 * k
        let nw = textWidth(label.name, size: nameSize, bold: true)
        text(label.name, ctx, x: label.centreX - nw / 2, top: 13 * k, size: nameSize, bold: true,
             color: ink, height: h)
        let detailSize: CGFloat = 11.5 * k
        let dw = textWidth(label.detail, size: detailSize, bold: false)
        text(label.detail, ctx, x: label.centreX - dw / 2, top: 34 * k, size: detailSize,
             color: label.colour, height: h)
        ctx.restoreGState()
    }
}

/// A hairline between the two panels, so they read as two instruments rather
/// than one wide picture.
func drawDivider(at x: Int, into buffer: MTLBuffer, layout: FrameLayout) {
    guard let ctx = context(buffer, layout) else { return }
    let k = layout.scale
    let h = CGFloat(layout.height)
    ctx.setFillColor(CGColor(srgbRed: 0.04, green: 0.06, blue: 0.09, alpha: 0.85))
    ctx.fill(CGRect(x: CGFloat(x) - k, y: h - CGFloat(layout.viewHeight),
                    width: 2 * k, height: CGFloat(layout.viewHeight)))
}

/// The polarimeter's reading, with the scale it sits on.
struct Dial {
    var degrees: Double
    var start: Double            // the reading for untouched sucrose
    var end: Double              // the reading when fully inverted
    var fractionConverted: Double
    var zeroCrossing: Double
    var note: String
    var centreX: CGFloat
    var lightColour: CGColor
}

func drawDial(_ dial: Dial, into buffer: MTLBuffer, layout: FrameLayout) {
    guard let ctx = context(buffer, layout) else { return }
    let k = layout.scale
    let h = CGFloat(layout.height)
    let shadow = CGColor(srgbRed: 0, green: 0.02, blue: 0.05, alpha: 0.95)

    ctx.saveGState()
    ctx.setShadow(offset: .zero, blur: 8 * k, color: shadow)
    let sign = dial.degrees < 0 ? "−" : "+"
    let reading = String(format: "%@%.2f°", sign, abs(dial.degrees))
    let readingSize: CGFloat = 30 * k
    let rw = textWidth(reading, size: readingSize, bold: true)
    let positive = CGColor(srgbRed: 0.98, green: 0.82, blue: 0.45, alpha: 1)
    let negative = CGColor(srgbRed: 0.53, green: 0.85, blue: 0.98, alpha: 1)
    text(reading, ctx, x: dial.centreX - rw / 2, top: CGFloat(layout.viewHeight) - 96 * k,
         size: readingSize, bold: true, color: dial.degrees < 0 ? negative : positive, height: h)
    let noteSize: CGFloat = 11 * k
    let nw = textWidth(dial.note, size: noteSize, bold: false)
    text(dial.note, ctx, x: dial.centreX - nw / 2, top: CGFloat(layout.viewHeight) - 60 * k,
         size: noteSize, color: muted, height: h)
    ctx.restoreGState()

    // The scale. Its zero mark is placed from the two derived end points, so it
    // sits where the arithmetic says: about three quarters of the way across.
    let scaleWidth: CGFloat = 250 * k
    let left = dial.centreX - scaleWidth / 2
    let top = CGFloat(layout.viewHeight) - 44 * k
    let trackHeight: CGFloat = 5 * k
    ctx.setFillColor(CGColor(srgbRed: 0.16, green: 0.20, blue: 0.25, alpha: 0.95))
    ctx.fill(CGRect(x: left, y: h - top - trackHeight, width: scaleWidth, height: trackHeight))

    let zeroX = left + scaleWidth * CGFloat(min(max(dial.zeroCrossing, 0), 1))
    ctx.setFillColor(CGColor(srgbRed: 0.55, green: 0.62, blue: 0.68, alpha: 1))
    ctx.fill(CGRect(x: zeroX - 0.5 * k, y: h - top - trackHeight - 4 * k,
                    width: 1.5 * k, height: trackHeight + 8 * k))
    let zeroLabel = "0°"
    let zw = textWidth(zeroLabel, size: 9.5 * k, bold: false)
    text(zeroLabel, ctx, x: zeroX - zw / 2, top: top + 12 * k, size: 9.5 * k, color: muted, height: h)

    let markX = left + scaleWidth * CGFloat(min(max(dial.fractionConverted, 0), 1))
    ctx.setFillColor(dial.lightColour)
    ctx.fillEllipse(in: CGRect(x: markX - 5 * k, y: h - top - trackHeight / 2 - 5 * k,
                               width: 10 * k, height: 10 * k))

    let startLabel = String(format: "%+.1f°", dial.start)
    let endLabel = String(format: "%+.1f°", dial.end)
    text(startLabel, ctx, x: left, top: top + 12 * k, size: 9.5 * k, color: muted, height: h)
    let ew = textWidth(endLabel, size: 9.5 * k, bold: false)
    text(endLabel, ctx, x: left + scaleWidth - ew, top: top + 12 * k, size: 9.5 * k,
         color: muted, height: h)
}

/// A short line naming what the active site is doing, under the left panel's
/// label. The state's name and nothing else — the caption bar carries the rest.
func drawStageLabel(_ stage: String, colour: CGColor, centreX: CGFloat,
                    into buffer: MTLBuffer, layout: FrameLayout) {
    guard let ctx = context(buffer, layout) else { return }
    let k = layout.scale
    let h = CGFloat(layout.height)
    ctx.saveGState()
    ctx.setShadow(offset: .zero, blur: 8 * k, color: CGColor(srgbRed: 0, green: 0.02, blue: 0.05,
                                                             alpha: 0.95))
    let size: CGFloat = 12.5 * k
    let w = textWidth(stage, size: size, bold: true)
    text(stage, ctx, x: centreX - w / 2, top: CGFloat(layout.viewHeight) - 44 * k, size: size,
         bold: true, color: colour, height: h)
    ctx.restoreGState()
}
