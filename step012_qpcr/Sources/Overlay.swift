// What the CPU draws on top of the GPU's picture: the caption bar, and the
// FRET curve with a marker tracking the live separation.
//
// The curve is here because the molecular half of this render is honest but
// quiet - two dyes changing colour. The curve is what makes the sixth power
// visible: the marker barely moves for most of the approach, then falls off
// a cliff. That cliff is the whole physical point of the step.

import CoreGraphics
import CoreText
import Foundation
import Metal
import simd

struct FrameLayout {
    let width: Int
    let viewHeight: Int
    let captionHeight: Int
    var height: Int { viewHeight + captionHeight }
    var scale: CGFloat { CGFloat(width) / 640 }
}

struct Caption {
    var title: String
    var subtitle: String
    var facts: String
    var aside: String
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

private func text(_ s: String, _ ctx: CGContext, x: CGFloat, top: CGFloat, size: CGFloat, bold: Bool = false,
                  color: CGColor = ink, height: CGFloat) {
    ctx.textPosition = CGPoint(x: x, y: height - top - size)
    CTLineDraw(line(s, size: size, bold: bold, color: color), ctx)
}

func textWidth(_ s: String, size: CGFloat, bold: Bool) -> CGFloat {
    CGFloat(CTLineGetTypographicBounds(line(s, size: size, bold: bold, color: ink), nil, nil, nil))
}

/// A colour for the overlay, from a linear-light colour the renderer uses.
private func cg(_ c: SIMD3<Float>, _ alpha: CGFloat = 1) -> CGColor {
    CGColor(srgbRed: CGFloat(gammaEncode(c.x)), green: CGFloat(gammaEncode(c.y)),
            blue: CGFloat(gammaEncode(c.z)), alpha: alpha)
}

/// The panel showing E against separation, with the live point marked.
/// Drawn in the top-right of the ray-traced view.
private func drawCurve(_ state: DyeState, _ ctx: CGContext, layout: FrameLayout) {
    let k = layout.scale
    let h = CGFloat(layout.height)
    let w: CGFloat = 214 * k, ph: CGFloat = 116 * k
    // Left column, under the channel bars. The arriving probe descends through
    // the top-right, and the two dyes end up low and centre-right when the
    // probe lands - earlier versions of this panel sat on top of one or the
    // other, hiding the moment the render exists to show.
    let x0 = 20 * k
    let yTop = 96 * k
    let plotL = x0 + 34 * k, plotR = x0 + w - 12 * k
    let plotT = yTop + 22 * k, plotB = yTop + ph - 20 * k

    // Panel
    ctx.setFillColor(CGColor(srgbRed: 0.04, green: 0.07, blue: 0.10, alpha: 0.78))
    ctx.addPath(CGPath(roundedRect: CGRect(x: x0, y: h - yTop - ph, width: w, height: ph),
                       cornerWidth: 6 * k, cornerHeight: 6 * k, transform: nil))
    ctx.fillPath()

    let maxSep: Float = 140      // ångströms across the plot
    func px(_ r: Float) -> CGFloat { plotL + CGFloat(min(r, maxSep) / maxSep) * (plotR - plotL) }
    func py(_ e: Float) -> CGFloat { h - (plotB - CGFloat(e) * (plotB - plotT)) }

    // Axes
    ctx.setStrokeColor(CGColor(srgbRed: 0.30, green: 0.36, blue: 0.40, alpha: 1))
    ctx.setLineWidth(1 * k)
    ctx.beginPath()
    ctx.move(to: CGPoint(x: plotL, y: h - plotT)); ctx.addLine(to: CGPoint(x: plotL, y: h - plotB))
    ctx.addLine(to: CGPoint(x: plotR, y: h - plotB))
    ctx.strokePath()

    // R0 marker: the half-way separation.
    ctx.setStrokeColor(CGColor(srgbRed: 0.45, green: 0.50, blue: 0.55, alpha: 0.8))
    ctx.setLineDash(phase: 0, lengths: [2.5 * k, 2.5 * k])
    ctx.beginPath()
    ctx.move(to: CGPoint(x: px(forsterRadius), y: h - plotT))
    ctx.addLine(to: CGPoint(x: px(forsterRadius), y: h - plotB))
    ctx.strokePath()
    ctx.setLineDash(phase: 0, lengths: [])

    // The curve itself, drawn from the same function the renderer uses.
    ctx.setStrokeColor(cg(acceptorColour(), 0.95))
    ctx.setLineWidth(2 * k)
    ctx.beginPath()
    var first = true
    var r: Float = 2
    while r <= maxSep {
        let p = CGPoint(x: px(r), y: py(transferEfficiency(separation: r)))
        if first { ctx.move(to: p); first = false } else { ctx.addLine(to: p) }
        r += 1
    }
    ctx.strokePath()

    // The live point.
    let mx = px(state.separation), my = py(state.efficiency)
    ctx.setStrokeColor(CGColor(srgbRed: 1, green: 1, blue: 1, alpha: 0.35))
    ctx.setLineWidth(1 * k)
    ctx.beginPath()
    ctx.move(to: CGPoint(x: mx, y: h - plotB)); ctx.addLine(to: CGPoint(x: mx, y: my))
    ctx.strokePath()
    ctx.setFillColor(CGColor(srgbRed: 1, green: 1, blue: 1, alpha: 1))
    ctx.fillEllipse(in: CGRect(x: mx - 3.6 * k, y: my - 3.6 * k, width: 7.2 * k, height: 7.2 * k))

    // Labels
    text("transfer", ctx, x: x0 + 12 * k, top: yTop + 7 * k, size: 10 * k, bold: true,
         color: ink, height: h)
    text("R\u{2080}", ctx, x: px(forsterRadius) - 5 * k, top: plotT - 11 * k, size: 9 * k,
         color: muted, height: h)
    text("100%", ctx, x: x0 + 6 * k, top: plotT - 4 * k, size: 8.5 * k, color: muted, height: h)
    text("0", ctx, x: x0 + 24 * k, top: plotB - 5 * k, size: 8.5 * k, color: muted, height: h)
    text("14 nm", ctx, x: plotR - 22 * k, top: plotB + 3 * k, size: 8.5 * k, color: muted, height: h)

    let readout = String(format: "%.1f nm  ·  %.0f%%", state.separation / 10, state.efficiency * 100)
    text(readout, ctx, x: x0 + 12 * k, top: yTop + ph - 15 * k, size: 10.5 * k, bold: true,
         color: cg(acceptorColour()), height: h)
}

/// Two small bars showing what each dye is emitting right now.
private func drawChannels(_ state: DyeState, _ ctx: CGContext, layout: FrameLayout) {
    let k = layout.scale
    let h = CGFloat(layout.height)
    let x0 = 20 * k
    let barW: CGFloat = 132 * k, barH: CGFloat = 9 * k
    let rows: [(String, Float, SIMD3<Float>)] = [
        ("donor 520 nm", state.donorBrightness, donorColour()),
        ("acceptor 640 nm", state.acceptorBrightness, acceptorColour()),
    ]
    // A panel behind them: the sky at the top of the frame is pale, and muted
    // text on it was unreadable in the first render.
    ctx.setFillColor(CGColor(srgbRed: 0.04, green: 0.07, blue: 0.10, alpha: 0.78))
    ctx.addPath(CGPath(roundedRect: CGRect(x: x0 - 10 * k, y: h - 78 * k,
                                           width: barW + 20 * k, height: 68 * k),
                       cornerWidth: 6 * k, cornerHeight: 6 * k, transform: nil))
    ctx.fillPath()
    var top = 22 * k
    for (name, value, colour) in rows {
        text(name, ctx, x: x0, top: top, size: 10 * k, color: ink, height: h)
        let y = h - top - 24 * k
        ctx.setFillColor(CGColor(srgbRed: 1, green: 1, blue: 1, alpha: 0.12))
        ctx.fill(CGRect(x: x0, y: y, width: barW, height: barH))
        ctx.setFillColor(cg(colour))
        ctx.fill(CGRect(x: x0, y: y, width: barW * CGFloat(max(value, 0)), height: barH))
        top += 34 * k
    }
}

func drawOverlay(_ caption: Caption, state: DyeState, into buffer: MTLBuffer, layout: FrameLayout) {
    guard let ctx = CGContext(data: buffer.contents(), width: layout.width, height: layout.height,
                              bitsPerComponent: 8, bytesPerRow: layout.width * 4,
                              space: CGColorSpace(name: CGColorSpace.sRGB)!,
                              bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return }
    let k = layout.scale
    let h = CGFloat(layout.height)
    let barTop = CGFloat(layout.viewHeight)

    drawCurve(state, ctx, layout: layout)
    drawChannels(state, ctx, layout: layout)

    ctx.setFillColor(captionBackground)
    ctx.fill(CGRect(x: 0, y: 0, width: CGFloat(layout.width), height: CGFloat(layout.captionHeight)))
    text(caption.title, ctx, x: 18 * k, top: barTop + 10 * k, size: 16 * k, bold: true, height: h)
    text(caption.subtitle, ctx, x: 18 * k, top: barTop + 34 * k, size: 12 * k, color: accent, height: h)
    text(caption.facts, ctx, x: 18 * k, top: barTop + 54 * k, size: 11.5 * k, color: muted, height: h)
    let w = textWidth(caption.aside, size: 11.5 * k, bold: false)
    text(caption.aside, ctx, x: CGFloat(layout.width) - 18 * k - w, top: barTop + 13 * k,
         size: 11.5 * k, color: muted, height: h)
}
