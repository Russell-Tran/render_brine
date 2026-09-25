// What the CPU draws over the GPU's picture: the labels, the stress key, the
// scale bar, and the evidence bar.
//
// The caption bar itself is step 13's `drawCaption`, unchanged. The evidence bar
// is NOT step 13's, and it is the one thing here that had to be rewritten rather
// than reused: step 13's bar has three levels — measured, derived, model — and
// this step needs a fourth, CONTESTED, for the abfraction claim. Adding a level
// to step 13's file would change step 13's picture, so the drawing is repeated
// here at the same sizes and positions and with the same marks.
//
// Steps 13 and 14 removed the evidence bar for being furniture: it said the same
// three lines for a hundred and twenty frames. It comes back in this step
// because here it CHANGES — each beat carries different rows, and one of the
// tests fails if it stops changing.

import CoreGraphics
import CoreText
import Foundation
import Metal
import simd

private let plate = CGColor(srgbRed: 0.04, green: 0.07, blue: 0.10, alpha: 0.72)
private let leaderInk = CGColor(srgbRed: 0.86, green: 0.90, blue: 0.92, alpha: 0.85)
private let labelInk = CGColor(srgbRed: 0.94, green: 0.96, blue: 0.97, alpha: 1)
private let labelKey = CGColor(srgbRed: 0.56, green: 0.88, blue: 0.90, alpha: 1)
private let mutedInk = CGColor(srgbRed: 0.60, green: 0.69, blue: 0.73, alpha: 1)

/// The four standings, in the series' colours. MEASURED, SIMULATED and MODEL
/// keep step 10's; CONTESTED is new and is deliberately the loudest.
extension Standing {
    var colour: CGColor {
        switch self {
        case .measured: return CGColor(srgbRed: 0.50, green: 0.84, blue: 0.62, alpha: 1)
        case .simulated: return CGColor(srgbRed: 0.55, green: 0.72, blue: 0.92, alpha: 1)
        case .model: return CGColor(srgbRed: 0.90, green: 0.76, blue: 0.40, alpha: 1)
        case .contested: return CGColor(srgbRed: 0.95, green: 0.45, blue: 0.42, alpha: 1)
        }
    }
}

private func ctLine(_ s: String, size: CGFloat, bold: Bool, color: CGColor) -> CTLine {
    let font = CTFontCreateWithName((bold ? "HelveticaNeue-Bold" : "HelveticaNeue") as CFString,
                                    size, nil)
    let attributes: [NSAttributedString.Key: Any] = [
        NSAttributedString.Key(kCTFontAttributeName as String): font,
        NSAttributedString.Key(kCTForegroundColorAttributeName as String): color,
    ]
    return CTLineCreateWithAttributedString(NSAttributedString(string: s, attributes: attributes))
}

private func put(_ s: String, _ ctx: CGContext, x: CGFloat, top: CGFloat, size: CGFloat,
                 bold: Bool = false, color: CGColor, height: CGFloat) {
    ctx.textPosition = CGPoint(x: x, y: height - top - size)
    CTLineDraw(ctLine(s, size: size, bold: bold, color: color), ctx)
}

private func frameContext(_ buffer: MTLBuffer, _ layout: FrameLayout) -> CGContext? {
    CGContext(data: buffer.contents(), width: layout.width, height: layout.height,
              bitsPerComponent: 8, bytesPerRow: layout.width * 4,
              space: CGColorSpace(name: CGColorSpace.sRGB)!,
              bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
}

// MARK: - the evidence bar

struct StandingRow {
    var standing: Standing
    var note: String
}

func standingRows(_ beat: Beat) -> [StandingRow] {
    evidenceRowsFor(beat).map { StandingRow(standing: $0.0, note: $0.1) }
}

func drawStandingBar(_ rows: [StandingRow], into buffer: MTLBuffer, layout: FrameLayout,
                     leftEdge: CGFloat) {
    guard let ctx = frameContext(buffer, layout) else { return }
    let k = layout.scale
    let h = CGFloat(layout.height)
    let barTop = CGFloat(layout.viewHeight)
    let box: CGFloat = 8 * k
    let rowStep: CGFloat = 17 * k
    for (i, row) in rows.enumerated() {
        let top: CGFloat = barTop + 14 * k + CGFloat(i) * rowStep
        let rect = CGRect(x: leftEdge, y: h - top - box, width: box, height: box)
        ctx.setFillColor(row.standing.colour)
        ctx.fill(rect)
        put(row.standing.rawValue, ctx, x: leftEdge + box + 6 * k, top: top - 0.5 * k,
            size: 9.5 * k, bold: true, color: row.standing.colour, height: h)
        put(row.note, ctx, x: leftEdge + box + 6 * k + 62 * k, top: top - 0.5 * k,
            size: 9.5 * k, color: mutedInk, height: h)
    }
}

// MARK: - labels with leaders

struct ToothLabel {
    var text: String
    var anchor: SIMD3<Float>      // the world point it names
    var at: CGPoint               // the plate's corner, in view pixels
    var leftOfAnchor: Bool
    var key: Bool = false
}

private func drawLeader(_ ctx: CGContext, from: CGPoint, to: CGPoint, height: CGFloat,
                        scale k: CGFloat) {
    let stub: CGFloat = from.x < to.x ? 7 * k : -7 * k
    ctx.setStrokeColor(leaderInk)
    ctx.setLineWidth(1.0 * k)
    ctx.beginPath()
    ctx.move(to: CGPoint(x: from.x, y: height - from.y))
    ctx.addLine(to: CGPoint(x: from.x + stub, y: height - from.y))
    ctx.addLine(to: CGPoint(x: to.x, y: height - to.y))
    ctx.strokePath()
    ctx.setFillColor(leaderInk)
    ctx.fillEllipse(in: CGRect(x: to.x - 2 * k, y: height - to.y - 2 * k,
                               width: 4 * k, height: 4 * k))
}

func drawLabels(_ items: [ToothLabel], into buffer: MTLBuffer, layout: FrameLayout,
                camera: Camera, fade: Float) {
    guard fade > 0.01, let ctx = frameContext(buffer, layout) else { return }
    let k = layout.scale
    let h = CGFloat(layout.height)
    let size: CGFloat = 11 * k
    for item in items {
        let p: SIMD2<Float> = camera.project(item.anchor, width: layout.width,
                                             height: layout.viewHeight)
        let anchor = CGPoint(x: CGFloat(p.x), y: CGFloat(p.y))
        let w: CGFloat = textWidth(item.text, size: size, bold: item.key)
        let padX: CGFloat = 6 * k
        let padY: CGFloat = 4 * k
        let originX: CGFloat = item.leftOfAnchor ? item.at.x - w - 2 * padX : item.at.x
        let rect = CGRect(x: originX, y: h - item.at.y - size - 2 * padY,
                          width: w + 2 * padX, height: size + 2 * padY)
        let tail = CGPoint(x: item.leftOfAnchor ? originX + w + 2 * padX : originX,
                           y: item.at.y + size / 2 + padY)
        ctx.setAlpha(CGFloat(fade))
        drawLeader(ctx, from: tail, to: anchor, height: h, scale: k)
        ctx.setFillColor(plate)
        ctx.fill(rect)
        put(item.text, ctx, x: originX + padX, top: item.at.y + padY, size: size,
            bold: item.key, color: item.key ? labelKey : labelInk, height: h)
        ctx.setAlpha(1)
    }
}

// MARK: - the stress key
//
// A colour map with no key is decoration. This is the same ramp the kernel
// evaluates — it calls the same arithmetic, written once in Swift and once in
// Metal, and a test compares the two at a dozen points so they cannot drift.

func stressColour(_ mpa: Float, range: Float) -> SIMD3<Float> {
    let t: Float = min(max(mpa / range, -1), 1)
    let neutral = SIMD3<Float>(0.90, 0.89, 0.86)
    if t >= 0 {
        let warm = SIMD3<Float>(0.97, 0.80, 0.30)
        let hot = SIMD3<Float>(0.85, 0.22, 0.14)
        if t < 0.5 { return neutral + (warm - neutral) * (t * 2) }
        return warm + (hot - warm) * ((t - 0.5) * 2)
    }
    let s: Float = -t
    let pale = SIMD3<Float>(0.66, 0.78, 0.86)
    let deep = SIMD3<Float>(0.16, 0.36, 0.60)
    if s < 0.5 { return neutral + (pale - neutral) * (s * 2) }
    return pale + (deep - pale) * ((s - 0.5) * 2)
}

func drawStressKey(into buffer: MTLBuffer, layout: FrameLayout, range: Float,
                   fade: Float, threshold: Float) {
    guard fade > 0.01, let ctx = frameContext(buffer, layout) else { return }
    let k = layout.scale
    let h = CGFloat(layout.height)
    let barWidth: CGFloat = 168 * k
    let barHeight: CGFloat = 9 * k
    let left: CGFloat = CGFloat(layout.width) - barWidth - 34 * k
    let top: CGFloat = 40 * k
    ctx.setAlpha(CGFloat(fade))
    let steps = 84
    for i in 0..<steps {
        let f: Float = Float(i) / Float(steps - 1)
        let mpa: Float = (f * 2 - 1) * range
        let c: SIMD3<Float> = stressColour(mpa, range: range)
        ctx.setFillColor(CGColor(srgbRed: CGFloat(c.x), green: CGFloat(c.y),
                                 blue: CGFloat(c.z), alpha: 1))
        let x: CGFloat = left + barWidth * CGFloat(i) / CGFloat(steps)
        ctx.fill(CGRect(x: x, y: h - top - barHeight,
                        width: barWidth / CGFloat(steps) + 1, height: barHeight))
    }
    put("maximum principal stress, MPa", ctx, x: left, top: top - 13 * k, size: 9.5 * k,
        bold: true, color: labelInk, height: h)
    put("−\(Int(range))", ctx, x: left, top: top + barHeight + 3 * k, size: 9 * k,
        color: labelInk, height: h)
    let zeroW = textWidth("0", size: 9 * k, bold: false)
    put("0", ctx, x: left + barWidth / 2 - zeroW / 2, top: top + barHeight + 3 * k,
        size: 9 * k, color: labelInk, height: h)
    let hiW = textWidth("+\(Int(range))", size: 9 * k, bold: false)
    put("+\(Int(range))", ctx, x: left + barWidth - hiW, top: top + barHeight + 3 * k,
        size: 9 * k, color: labelInk, height: h)
    // Where enamel gives way, marked on the ramp itself. A key that shows the
    // colours but not the number that matters is half a key.
    let f: CGFloat = CGFloat((threshold / range + 1) / 2)
    let tick: CGFloat = left + barWidth * f
    ctx.setFillColor(CGColor(srgbRed: 0.08, green: 0.09, blue: 0.10, alpha: 1))
    ctx.fill(CGRect(x: tick - 1 * k, y: h - top - barHeight - 4 * k,
                    width: 2 * k, height: barHeight + 8 * k))
    put("enamel fails", ctx, x: tick - 24 * k, top: top + barHeight + 15 * k, size: 9 * k,
        bold: true, color: labelInk, height: h)
    ctx.setAlpha(1)
}

// MARK: - the scale bar

func drawScaleRule(into buffer: MTLBuffer, layout: FrameLayout, millimetres: Float) {
    guard let ctx = frameContext(buffer, layout) else { return }
    let k = layout.scale
    let h = CGFloat(layout.height)
    let length: CGFloat = CGFloat(millimetres / mmPerPixel) * CGFloat(layout.width)
        / CGFloat(frameWidth)
    let x: CGFloat = 22 * k
    let bottom: CGFloat = CGFloat(layout.viewHeight) - 26 * k
    let ink = CGColor(srgbRed: 0.93, green: 0.95, blue: 0.96, alpha: 0.93)
    ctx.setFillColor(ink)
    ctx.fill(CGRect(x: x, y: h - bottom, width: length, height: 2.5 * k))
    ctx.fill(CGRect(x: x, y: h - bottom - 3 * k, width: 2 * k, height: 9 * k))
    ctx.fill(CGRect(x: x + length - 2 * k, y: h - bottom - 3 * k, width: 2 * k, height: 9 * k))
    put("\(Int(millimetres)) mm", ctx, x: x, top: bottom - 20 * k, size: 10.5 * k,
        bold: true, color: ink, height: h)
}

/// The anatomical labels, at world points taken from the anatomy functions —
/// none of them is a hand-placed pixel, so none of them can drift away from what
/// is drawn underneath it. The PLATES are placed in fractions of the frame
/// rather than in pixels, so a half-width render puts them in the same places.
func toothLabels(layout: FrameLayout) -> [ToothLabel] {
    let w: CGFloat = CGFloat(layout.width)
    let h: CGFloat = CGFloat(layout.viewHeight)
    func mid(_ a: Float, _ b: Float, y: Float) -> SIMD3<Float> {
        SIMD3<Float>((a + b) * 0.5, y, 0)
    }
    let enamelY: Float = 5.6
    let enamelAnchor = mid(dejRadius(y: enamelY), outerHalfWidth(y: enamelY), y: enamelY)
    let dejY: Float = 3.4
    let dejAnchor = mid(dentinRadius(y: dejY), dejRadius(y: dejY), y: dejY)
    let dentinY: Float = -1.0
    let dentinAnchor = mid(pulpRadius(y: dentinY), dentinRadius(y: dentinY), y: dentinY)
    return [
        ToothLabel(text: "enamel, 84 GPa", anchor: enamelAnchor,
                   at: CGPoint(x: w * 0.745, y: h * 0.135), leftOfAnchor: false),
        ToothLabel(text: "dentino-enamel junction", anchor: dejAnchor,
                   at: CGPoint(x: w * 0.720, y: h * 0.225), leftOfAnchor: false, key: true),
        ToothLabel(text: "dentin, 18 GPa", anchor: dentinAnchor,
                   at: CGPoint(x: w * 0.700, y: h * 0.560), leftOfAnchor: false),
        ToothLabel(text: "pulp", anchor: SIMD3<Float>(0, 1.6, 0),
                   at: CGPoint(x: w * 0.255, y: h * 0.235), leftOfAnchor: true),
        ToothLabel(text: "the neck — CEJ", anchor: SIMD3<Float>(-outerHalfWidth(y: 0), 0, 0),
                   at: CGPoint(x: w * 0.255, y: h * 0.395), leftOfAnchor: true, key: true),
        ToothLabel(text: "cementum", anchor: SIMD3<Float>(-rootWidth(y: -5.5) + 0.06, -5.5, 0),
                   at: CGPoint(x: w * 0.255, y: h * 0.640), leftOfAnchor: true),
        ToothLabel(text: "ligament, 0.2 mm",
                   anchor: SIMD3<Float>(rootWidth(y: -7.5) + pdlThickness * 0.5, -7.5, 0),
                   at: CGPoint(x: w * 0.700, y: h * 0.730), leftOfAnchor: false, key: true),
        ToothLabel(text: "alveolar bone", anchor: SIMD3<Float>(7.2, -6.5, 0),
                   at: CGPoint(x: w * 0.760, y: h * 0.870), leftOfAnchor: false),
    ]
}
