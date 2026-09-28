// What the CPU draws over each frame: the title and caption, the 1 mm scale
// bar, the trail's label, and a footfall diagram of the tripod gait. All of
// it is the same in every frame, so it costs the GIF nothing after frame 0.
// Text drawing is step 19's CoreText approach, copied rather than shared.

import CoreGraphics
import CoreText
import Foundation
import Metal
import simd

private let ink = CGColor(srgbRed: 0.13, green: 0.14, blue: 0.16, alpha: 1)
private let softInk = CGColor(srgbRed: 0.13, green: 0.14, blue: 0.16, alpha: 0.62)
private let trailInk = CGColor(srgbRed: 0.16, green: 0.34, blue: 0.78, alpha: 1)
private let tripodAInk = CGColor(srgbRed: 0.13, green: 0.14, blue: 0.16, alpha: 1)
private let tripodBInk = CGColor(srgbRed: 0.72, green: 0.40, blue: 0.16, alpha: 1)

private func ctLine(_ s: String, size: CGFloat, bold: Bool, italic: Bool = false, color: CGColor) -> CTLine {
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
private func put(_ s: String, _ ctx: CGContext, x: CGFloat, top: CGFloat, size: CGFloat, bold: Bool = false,
                 italic: Bool = false, color: CGColor = ink, height: CGFloat) {
    ctx.textPosition = CGPoint(x: x, y: height - top - size)
    CTLineDraw(ctLine(s, size: size, bold: bold, italic: italic, color: color), ctx)
}

private func fillRect(_ ctx: CGContext, _ x: CGFloat, _ top: CGFloat, _ w: CGFloat, _ hgt: CGFloat, height h: CGFloat) {
    ctx.fill(CGRect(x: x, y: h - top - hgt, width: w, height: hgt))
}

/// The caption lines, from the numbers — so the words cannot drift from the model.
func captionLines() -> [String] {
    let slow: Int = Int(slowdown.rounded())
    return [
        String(format: "Walking about %.0f cm/s at %d °C, shown slowed ×%d. Alternating tripod gait, about %.0f strides a second.",
               realSpeed / 10, speedTemperature, slow, stepFrequency),
        String(format: "They follow %.1f s apart, about %.0f mm nose to nose: a close file. Each ant stops and presses its gaster tip to the trail, laying pheromone from the hindgut.",
               headwayReal, spacing),
    ]
}

func numberWord(_ n: Int) -> String {
    let words: [String] = ["no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    return n < words.count ? words[n] : "\(n)"
}

/// The label for the trail itself.
let trailLabel: String = "trail pheromone — invisible in life"
func markLabel() -> String {
    "bright spots: fresh marks, highlighted briefly (a real one lasts about \(markLifetimeMinutes) min)"
}

/// The rows the text occupies, image coordinates (y down): the caption block
/// ends at the first number, the legends along the bottom begin at the
/// second. A test keeps every ant between them.
func textBands(width w: Int, height hI: Int) -> (captionBottom: Float, legendTop: Float) {
    let k: Float = Float(hI) / 300
    let h: Float = Float(hI)
    let captionBottom: Float = (52 + 19 * Float(captionLines().count - 1) + 14.5 + 4) * k
    let rowH: Float = 9 * k
    let top0: Float = h - 26 * k - 6 * rowH - 2 * k
    let legendTop: Float = min(top0 - 20 * k, h - 50 * k, h - 26 * k - 22 * k)
    return (captionBottom, legendTop)
}

func annotate(_ image: TrailImage) {
    let w: Int = image.width
    let hI: Int = image.height
    guard let ctx = CGContext(data: image.pixels.contents(), width: w, height: hI, bitsPerComponent: 8,
                              bytesPerRow: w * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                              bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return }
    let h: CGFloat = CGFloat(hI)
    // Text is sized to the frame's height, so a wider file keeps the same type.
    let k: CGFloat = CGFloat(hI) / 300
    let margin: CGFloat = 26 * k

    // Title and caption.
    put("Lasius niger", ctx, x: margin, top: 18 * k, size: 25 * k, italic: true, height: h)
    let lw: CGFloat = textWidth("Lasius niger", size: 25 * k, bold: false, italic: true)
    put(" workers follow a pheromone trail — \(numberWord(fileCount)) in view", ctx, x: margin + lw, top: 18 * k, size: 25 * k, height: h)
    for (i, line) in captionLines().enumerated() {
        put(line, ctx, x: margin, top: (52 + 19 * CGFloat(i)) * k, size: 14.5 * k, color: softInk, height: h)
    }

    // Scale bar, bottom left: 1 mm along the trail, true anywhere in frame.
    let mmPx: CGFloat = CGFloat(1.0 / millimetresPerPixel(width: w))
    let barY: CGFloat = h - 26 * k
    ctx.setFillColor(ink)
    fillRect(ctx, margin, barY, mmPx, 3 * k, height: h)
    fillRect(ctx, margin, barY - 3 * k, 2 * k, 9 * k, height: h)
    fillRect(ctx, margin + mmPx - 2 * k, barY - 3 * k, 2 * k, 9 * k, height: h)
    let bw: CGFloat = textWidth("1 mm", size: 14 * k, bold: true)
    put("1 mm", ctx, x: margin + mmPx / 2 - bw / 2, top: barY - 22 * k, size: 14 * k, bold: true, height: h)

    // The trail's label: a swatch of the drawn colour and the words, beside
    // the scale bar — no leader line, which the ants would walk through.
    let labelX: CGFloat = margin + mmPx + 30 * k
    let labelTop: CGFloat = h - 50 * k
    ctx.setFillColor(trailInk)
    fillRect(ctx, labelX, labelTop + 4 * k, 22 * k, 8 * k, height: h)
    put(trailLabel, ctx, x: labelX + 30 * k, top: labelTop, size: 14 * k, bold: true, color: trailInk, height: h)
    put(markLabel(), ctx, x: labelX + 30 * k, top: labelTop + 19 * k, size: 12.5 * k, color: softInk, height: h)

    // Footfall diagram, bottom right: two strides of the six legs, a bar
    // where the foot is down. Drawn from the same duty factor and tripods
    // the ants walk with.
    let rows: [(String, Int)] = [("L1", 3), ("L2", 4), ("L3", 5), ("R1", 0), ("R2", 1), ("R3", 2)]
    let rowH: CGFloat = 9 * k
    let cycleW: CGFloat = 70 * k
    let dx: CGFloat = CGFloat(w) - margin - 2 * cycleW
    let top0: CGFloat = h - 26 * k - CGFloat(rows.count) * rowH - 2 * k
    put("footfalls: bar = foot down", ctx, x: dx - 26 * k, top: top0 - 20 * k, size: 12 * k, color: softInk, height: h)
    for (r, (name, leg)) in rows.enumerated() {
        let top: CGFloat = top0 + CGFloat(r) * rowH
        put(name, ctx, x: dx - 24 * k, top: top - 1 * k, size: 9.5 * k, color: softInk, height: h)
        let inA: Bool = tripodA.contains(leg)
        ctx.setFillColor(inA ? tripodAInk : tripodBInk)
        let offset: Float = inA ? 0 : 0.5
        for c in -1...2 {
            let start: CGFloat = (CGFloat(c) + CGFloat(offset)) * cycleW
            let a: CGFloat = max(start, 0)
            let b: CGFloat = min(start + CGFloat(dutyFactor) * cycleW, 2 * cycleW)
            if b > a { fillRect(ctx, dx + a, top + 1.5 * k, b - a, rowH - 3 * k, height: h) }
        }
    }
    put("tripod 1: R1 L2 R3", ctx, x: dx - 26 * k - 132 * k, top: top0 + 8 * k, size: 11.5 * k, bold: true, color: tripodAInk, height: h)
    put("tripod 2: L1 R2 L3", ctx, x: dx - 26 * k - 132 * k, top: top0 + 26 * k, size: 11.5 * k, bold: true, color: tripodBInk, height: h)
}
