// What the CPU draws over each frame: the title and caption, the 1 mm scale
// bar, the trail's label, the wood's labels, and a footfall diagram of the
// tripod gait. All of it is the same in every frame, so it costs the GIF
// nothing after frame 0. Text drawing is step 19's CoreText approach, copied
// rather than shared. Step 57 adds the wood's lines and labels.

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
private let woodInk = CGColor(srgbRed: 0.33, green: 0.18, blue: 0.06, alpha: 1)

/// The depth (z) whose ground point lands on a given image row: the
/// camera looks square along x, so a row is one z on the ground.
func woodLabelZ(width w: Int, height h: Int, row: CGFloat) -> Float {
    var lo: Float = -20
    var hi: Float = 20
    for _ in 0..<50 {
        let mid: Float = (lo + hi) / 2
        // Larger z is nearer the camera, lower in the frame.
        let probe: SIMD3<Float> = SIMD3<Float>(0, 0, mid)
        let probeRow: CGFloat = CGFloat(project(probe, width: w, height: h).y)   // typed apart: 5 ms inline on the mini
        if probeRow < row { lo = mid } else { hi = mid }
    }
    return (lo + hi) / 2
}

/// Where to name the zones, at depth z: the middle of the latewood band
/// nearest x = 5.5 mm, and the middle of the earlywood just before it.
func woodLabelPositions(z: Float) -> [(String, Float)] {
    var bestLate: Float = 5.5
    var bestDist: Float = 1e9
    var x: Float = 0
    var inLate: Bool = false
    var start: Float = 0
    var bands: [(Float, Float)] = []
    while x < 10 {
        let l: Bool = latewood(x, z) > 0.5
        if l && !inLate { start = x }
        if !l && inLate { bands.append((start, x)) }
        inLate = l
        x += 0.002
    }
    for (a, b) in bands {
        let c: Float = (a + b) / 2
        if abs(c - 5.5) < bestDist { bestDist = abs(c - 5.5); bestLate = c }
    }
    // The earlywood: from the band before this latewood's end to its start.
    var earlyEnd: Float = bestLate
    var earlyStart: Float = bestLate
    var e: Float = bestLate
    while e > bestLate - 3 && latewood(e, z) > 0.5 { e -= 0.002 }
    earlyEnd = e
    while e > bestLate - 4 && latewood(e, z) <= 0.5 { e -= 0.002 }
    earlyStart = e
    return [("latewood", bestLate), ("earlywood", (earlyStart + earlyEnd) / 2)]
}

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
    let latePercent: Int = Int((latewoodFraction * 100).rounded())
    let raE: Float = raEarlywood * 1000
    let raL: Float = raLatewood * 1000
    return [
        String(format: "Walking about %.0f cm/s at %d °C, shown slowed ×%d. Alternating tripod gait, about %.0f strides a second.",
               realSpeed / 10, speedTemperature, slow, stepFrequency),
        "Each ant stops and presses its gaster tip to the trail, laying pheromone from the hindgut (a dihydroisocoumarin).",
        String(format: "The board: planed %@ sapwood, quarter-sawn — annual rings %.1f mm apart, %d%% latewood (Pinkowski et al. 2016).",
               woodSpecies, ringWidth, latePercent),
        String(format: "Planing leaves earlywood rougher (Ra %.1f µm) than latewood (%.1f µm). The feet stand on that relief, drawn true to scale: a pixel or so.",
               raE, raL),
    ]
}

/// The label for the trail itself.
let trailLabel: String = "trail pheromone — invisible in life"
func markLabel() -> String {
    "bright spots: fresh marks, highlighted briefly (one was estimated to last about \(markLifetimeMinutes) min at room temperature)"
}
/// What the surface does to a trail: the only comparison of porous and
/// non-porous surfaces found. Jeanson, Ratnieks & Deneubourg (2003,
/// *Physiol Entomol* 28: 192–198, "Pheromone trail decay rates on different
/// substrates in the Pharaoh's ant, Monomorium pharaonis"), abstract: "the
/// half-life times of the pheromone are estimated as approximately 9 min and
/// 3 min on plastic and paper [newspaper], respectively". A different ant,
/// newspaper not wood: so the line says exactly that, and no more.
func substrateLabel() -> String {
    "the surface matters: a Pharaoh's ant trail's half-life was ~3 min on newspaper, ~9 on plastic (Jeanson et al. 2003); none found for wood"
}

func annotate(_ image: TrailImage) {
    let w: Int = image.width
    let hI: Int = image.height
    guard let ctx = CGContext(data: image.pixels.contents(), width: w, height: hI, bitsPerComponent: 8,
                              bytesPerRow: w * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                              bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return }
    let h: CGFloat = CGFloat(hI)
    let k: CGFloat = CGFloat(w) / 1200
    let margin: CGFloat = 26 * k

    // Title and caption.
    put("Lasius niger", ctx, x: margin, top: 18 * k, size: 25 * k, italic: true, height: h)
    let lw: CGFloat = textWidth("Lasius niger", size: 25 * k, bold: false, italic: true)
    put(" workers follow a pheromone trail across a pine board", ctx, x: margin + lw, top: 18 * k, size: 25 * k, height: h)
    for (i, line) in captionLines().enumerated() {
        let row: CGFloat = CGFloat(i)
        let rowTop: CGFloat = 50 + 18 * row
        put(line, ctx, x: margin, top: rowTop * k, size: 14 * k, color: softInk, height: h)
    }

    // The wood's two zones, named once each: a small label over one latewood
    // band and the earlywood beside it, with a tick down to the band, in the
    // strip between the caption and the ants (a test keeps the ants out of it).
    let labelRowTop: CGFloat = 132 * k
    let tickZ: Float = woodLabelZ(width: w, height: hI, row: labelRowTop + 22 * k)
    for (text, x) in woodLabelPositions(z: tickZ) {
        let p: SIMD2<Float> = project(SIMD3<Float>(x, 0, tickZ), width: w, height: hI)
        let tw: CGFloat = textWidth(text, size: 12 * k, bold: true)
        let cx: CGFloat = CGFloat(p.x)
        put(text, ctx, x: cx - tw / 2, top: labelRowTop, size: 12 * k, bold: true, color: woodInk, height: h)
        ctx.setFillColor(woodInk)
        fillRect(ctx, cx - 0.75 * k, labelRowTop + 16 * k, 1.5 * k, 8 * k, height: h)
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
    let labelTop: CGFloat = h - 66 * k
    ctx.setFillColor(trailInk)
    fillRect(ctx, labelX, labelTop + 4 * k, 22 * k, 8 * k, height: h)
    put(trailLabel, ctx, x: labelX + 30 * k, top: labelTop, size: 14 * k, bold: true, color: trailInk, height: h)
    put(markLabel(), ctx, x: labelX + 30 * k, top: labelTop + 19 * k, size: 12 * k, color: softInk, height: h)
    put(substrateLabel(), ctx, x: labelX + 30 * k, top: labelTop + 35 * k, size: 12 * k, color: softInk, height: h)

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
