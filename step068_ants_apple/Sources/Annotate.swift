// Step 68: step 50's labels, copied and reworded for the apple: the rings
// round the insets, the callout from ant 0's right antenna tip (it rides on
// the tip, and fades in only as the ant reaches the apple and out as it turns
// away, so it never jumps), a note of how far the juice has dropped when the
// tip is up, three scale bars, the hair labels, the molecule captions, and
// the title and caption. All text sits on the card or on plates, clear of
// the walking ants and the apple's drinking face.
//
// Text drawing is step 19's CoreText approach, copied rather than shared.

import CoreGraphics
import CoreText
import Foundation
import Metal
import simd

private let ink = CGColor(srgbRed: 0.13, green: 0.14, blue: 0.16, alpha: 1)
private let softInk = CGColor(srgbRed: 0.13, green: 0.14, blue: 0.16, alpha: 0.62)
private let ringInk = CGColor(srgbRed: 1, green: 1, blue: 1, alpha: 1)
private let plate = CGColor(srgbRed: 1, green: 1, blue: 1, alpha: 0.84)

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

private func scaleBar(_ ctx: CGContext, label: String, pixels: CGFloat, x: CGFloat, y: CGFloat, k: CGFloat,
                      height h: CGFloat, color: CGColor = ink, backed: Bool = false) {
    if backed {
        ctx.setFillColor(plate)
        ctx.fill(CGRect(x: x - 10 * k, y: h - y - 10 * k, width: pixels + 20 * k, height: 40 * k))
    }
    ctx.setFillColor(color)
    ctx.fill(CGRect(x: x, y: h - y, width: pixels, height: 3 * k))
    ctx.fill(CGRect(x: x, y: h - y - 3 * k, width: 2 * k, height: 9 * k))
    ctx.fill(CGRect(x: x + pixels - 2 * k, y: h - y - 3 * k, width: 2 * k, height: 9 * k))
    let w: CGFloat = textWidth(label, size: 15 * k, bold: true)
    put(label, ctx, x: x + pixels / 2 - w / 2, top: y - 24 * k, size: 15 * k, bold: true, color: color, height: h)
}

private func line(_ ctx: CGContext, _ a: CGPoint, _ b: CGPoint, height h: CGFloat) {
    ctx.beginPath()
    ctx.move(to: CGPoint(x: a.x, y: h - a.y))
    ctx.addLine(to: CGPoint(x: b.x, y: h - b.y))
    ctx.strokePath()
}

/// A line from `a` to `b`, drawn only where it lies outside the circle.
private func lineOutside(_ ctx: CGContext, _ a: CGPoint, _ b: CGPoint, circle c: CGPoint, radius r: CGFloat, height h: CGFloat) {
    let d = CGPoint(x: b.x - a.x, y: b.y - a.y)
    let f = CGPoint(x: a.x - c.x, y: a.y - c.y)
    let A: CGFloat = d.x * d.x + d.y * d.y
    let B: CGFloat = 2 * (f.x * d.x + f.y * d.y)
    let C: CGFloat = f.x * f.x + f.y * f.y - r * r
    let disc: CGFloat = B * B - 4 * A * C
    var start: CGFloat = 0
    if disc > 0 && C < 0 { start = (-B + disc.squareRoot()) / (2 * A) }
    if start >= 1 { return }
    line(ctx, CGPoint(x: a.x + d.x * start, y: a.y + d.y * start), b, height: h)
}

private func circle(_ ctx: CGContext, centre c: CGPoint, radius r: CGFloat, height h: CGFloat) {
    ctx.strokeEllipse(in: CGRect(x: c.x - r, y: h - c.y - r, width: 2 * r, height: 2 * r))
}

/// The two outer tangent lines between two circles: the magnifier callout.
private func tangents(_ c1: CGPoint, _ r1: CGFloat, _ c2: CGPoint, _ r2: CGFloat) -> [(CGPoint, CGPoint)] {
    let dx: CGFloat = c2.x - c1.x
    let dy: CGFloat = c2.y - c1.y
    let d: CGFloat = (dx * dx + dy * dy).squareRoot()
    let base: CGFloat = atan2(dy, dx)
    let off: CGFloat = acos((r1 - r2) / d)
    var out: [(CGPoint, CGPoint)] = []
    for a in [base + off, base - off] {
        let ca: CGFloat = cos(a)
        let sa: CGFloat = sin(a)
        let p1 = CGPoint(x: c1.x + r1 * ca, y: c1.y + r1 * sa)
        let p2 = CGPoint(x: c2.x + r2 * ca, y: c2.y + r2 * sa)
        out.append((p1, p2))
    }
    return out
}

/// Pixel sizes of each view, for the scale bars and the tests.
func mainMillimetresPerPixel(width: Int) -> Float { mainViewWidth / Float(width) }
func insetMicrometresPerPixel(height: Int) -> Float { insetField / (2 * insetRadius * Float(height)) }
func moleculeAngstromsPerPixel(height: Int) -> Float { moleculeField / (2 * moleculeRadius * Float(height)) }

/// How much of the callout to ant 0's tip is drawn: fully while its tip is
/// within 0.4 mm of where it touches, fading out by 2 mm.
func calloutStrength(_ frame: FrameState, scene: Scene) -> Float {
    let d: Float = simd_distance(frame.focus.antennaTips[0], scene.contact.point)
    return 1 - smoothstep01((d - 0.4) / 1.6)
}

/// The title, in two lines, and the caption, from the numbers — so the
/// words cannot drift from the model. Short lines: the block sits on the
/// card left of the apple.
let titleLines: [String] = ["workers at a piece of peeled", "Golden Delicious apple"]
func captionLines() -> [String] {
    [
        "Peeled, the cut flesh is wet with juice. Each ant taps",
        "it with both antennae — taste hairs at the tips touch",
        "the juice — then pushes out its glossa (tongue) and",
        "sucks the juice up. Golden Delicious juice is mostly",
        "fructose; the cut gives off 2-hexenal, the fruit esters.",
        String(format: "Slowed ×%.0f: a tap takes ¼ s (ants antennate %.0f–%.0f times", slowdown,
               realStrokeRange.lowerBound, realStrokeRange.upperBound),
        String(format: "a second); they walk %.0f cm/s (at %d °C). Drinks are cut", realSpeed / 10, speedTemperature),
        "short: a real one lasts until the ant holds all it wants.",
        "Browning, which takes minutes, is not shown.",
    ]
}

/// Where the fixed words sit, in pixels of a `width` × `height` frame: the
/// title block, the fructose caption, the scale bar. The tests keep every
/// ant out of these at every frame.
/// The two odorants' names, notes and heights in their inset (Å).
func odourNames() -> [(String, String, Float)] {
    [("2-hexenal, C₆H₁₀O", "green: the cut releases it", hexenalCentre.y),
     ("butyl acetate, C₆H₁₂O₂", "an ester of the fruit", butylAcetateCentre.y)]
}
func odourNameBlock(height hI: Int) -> PixelRect {
    let h: Float = Float(hI)
    let k: Float = h / 900
    let right: Float = (odourCentre.x - odourRadius) * h - 6 * k
    let top: Float = odourCentre.y * h - odourRadius * h
    return PixelRect(x0: Int(right - 190 * k), y0: Int(top), x1: Int(right), y1: Int(top + 2 * odourRadius * h))
}
func titleBlock(height hI: Int) -> PixelRect {
    let k: Float = Float(hI) / 900
    let lines: Int = captionLines().count
    let lineCount: Float = Float(lines)
    let depth: Float = 18 * lineCount
    let bottom: Float = (96 + depth) * k
    return PixelRect(x0: Int(40 * k), y0: Int(20 * k), x1: Int(560 * k), y1: Int(bottom))
}
func fructoseBlock(height hI: Int) -> PixelRect {
    let h: Float = Float(hI)
    let k: Float = h / 900
    let right: Float = (moleculeCentre.x + moleculeRadius) * h
    let top: Float = (moleculeCentre.y + moleculeRadius) * h + 14 * k
    return PixelRect(x0: Int(right - 250 * k), y0: Int(top), x1: Int(right), y1: Int(top + 94 * k))
}
func scaleBarBlock(height hI: Int) -> PixelRect {
    let k: Float = Float(hI) / 900
    let mm: Float = 1 / mainMillimetresPerPixel(width: hI * 16 / 9)
    return PixelRect(x0: Int(40 * k), y0: Int(268 * k), x1: Int(60 * k + mm), y1: Int(306 * k))
}

func annotate(_ image: AntImage, scene: Scene, frame: FrameState) {
    let w: Int = image.width
    let hI: Int = image.height
    guard let ctx = CGContext(data: image.pixels.contents(), width: w, height: hI, bitsPerComponent: 8,
                              bytesPerRow: w * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                              bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return }
    let h: CGFloat = CGFloat(hI)
    let k: CGFloat = h / 900
    ctx.setLineCap(.round)

    let insetC = CGPoint(x: CGFloat(insetCentre.x) * h, y: CGFloat(insetCentre.y) * h)
    let insetR: CGFloat = CGFloat(insetRadius) * h
    let molC = CGPoint(x: CGFloat(moleculeCentre.x) * h, y: CGFloat(moleculeCentre.y) * h)
    let molR: CGFloat = CGFloat(moleculeRadius) * h
    let odC = CGPoint(x: CGFloat(odourCentre.x) * h, y: CGFloat(odourCentre.y) * h)
    let odR: CGFloat = CGFloat(odourRadius) * h

    // The callout from ant 0's right antenna tip to the inset, riding on the
    // tip, fading with its distance from the juice.
    let strength: CGFloat = CGFloat(calloutStrength(frame, scene: scene))
    if strength > 0.004 {
        let focus: WorldAnt = frame.focus
        let touch: SIMD3<Float> = focus.antennaTips[0] - scene.contact.normal * (funiculusTipRadius * trips[0].scale)
        let tp: SIMD2<Float> = projectMain(touch, width: w, height: hI)
        let touchPt = CGPoint(x: CGFloat(tp.x), y: CGFloat(tp.y))
        let spotR: CGFloat = 20 * k
        ctx.setStrokeColor(CGColor(srgbRed: 0.13, green: 0.14, blue: 0.16, alpha: 0.62 * strength))
        ctx.setLineWidth(1.5 * k)
        for (a, b) in tangents(touchPt, spotR, insetC, insetR) { line(ctx, a, b, height: h) }
        ctx.setStrokeColor(CGColor(srgbRed: 0.13, green: 0.14, blue: 0.16, alpha: strength))
        ctx.setLineWidth(2 * k)
        circle(ctx, centre: touchPt, radius: spotR, height: h)
    }

    // From the tip pore in the inset to the fructose inset.
    let apex: SIMD3<Float> = scene.hairs[0].tip + simd_normalize(scene.hairs[0].tip - scene.hairs[0].base) * scene.hairs[0].tipRadius
    let ap: SIMD2<Float> = projectInset(apex, height: hI)
    let poreR: CGFloat = 26 * k
    let porePt = CGPoint(x: CGFloat(ap.x), y: CGFloat(ap.y))
    ctx.setStrokeColor(softInk)
    ctx.setLineWidth(1.5 * k)
    for (a, b) in tangents(porePt, poreR, molC, molR) { lineOutside(ctx, a, b, circle: insetC, radius: insetR + 3.5 * k, height: h) }
    ctx.setStrokeColor(ringInk)
    ctx.setLineWidth(2 * k)
    circle(ctx, centre: porePt, radius: poreR, height: h)

    // Lifted, or away, the inset has left the juice behind: say how far.
    let drop: Float = frame.crystal.translation.y
    if drop > 6 {
        let gap1 = "↓ juice"
        let gap2: String = drop < 2000 ? String(format: "%.0f µm away", drop) : "far away"
        let right: CGFloat = porePt.x - poreR - 8 * k
        put(gap1, ctx, x: right - textWidth(gap1, size: 13 * k, bold: false), top: porePt.y - 40 * k, size: 13 * k,
            color: softInk, height: h)
        put(gap2, ctx, x: right - textWidth(gap2, size: 13 * k, bold: false), top: porePt.y - 23 * k, size: 13 * k,
            color: softInk, height: h)
    }

    // Rings round the insets.
    for (c, r) in [(insetC, insetR), (molC, molR), (odC, odR)] {
        ctx.setStrokeColor(ringInk)
        ctx.setLineWidth(6 * k)
        circle(ctx, centre: c, radius: r, height: h)
        ctx.setStrokeColor(ink)
        ctx.setLineWidth(1.5 * k)
        circle(ctx, centre: c, radius: r + 3.5 * k, height: h)
    }

    // Scale bars.
    let mmPx: CGFloat = CGFloat(1.0 / mainMillimetresPerPixel(width: w))
    scaleBar(ctx, label: "1 mm", pixels: mmPx, x: 50 * k, y: 300 * k, k: k, height: h)
    let umPx: CGFloat = CGFloat(5.0 / insetMicrometresPerPixel(height: hI))
    scaleBar(ctx, label: "5 µm", pixels: umPx, x: insetC.x - umPx / 2, y: insetC.y + insetR - 28 * k, k: k, height: h, backed: true)
    let nmPx: CGFloat = CGFloat(moleculeBarNanometres * 10 / moleculeAngstromsPerPixel(height: hI))
    scaleBar(ctx, label: String(format: "%g nm", moleculeBarNanometres), pixels: nmPx, x: molC.x - nmPx / 2,
             y: molC.y + molR - 22 * k, k: k, height: h, backed: true)
    let odPx: CGFloat = CGFloat(odourBarNanometres * 10 / (odourField / (2 * odourRadius * Float(hI))))
    scaleBar(ctx, label: String(format: "%g nm", odourBarNanometres), pixels: odPx, x: odC.x - odPx / 2,
             y: odC.y + odR - 18 * k, k: k, height: h, backed: true)

    // The two hairs.
    let taste: Sensillum = scene.hairs[0]
    let smell: Sensillum = scene.hairs[1]
    let tAlong: SIMD3<Float> = (taste.tip - taste.base) * 0.62
    let sAlong: SIMD3<Float> = (smell.tip - smell.base) * 0.55
    let tLabel: SIMD2<Float> = projectInset(taste.base + tAlong, height: hI)
    let sLabel: SIMD2<Float> = projectInset(smell.base + sAlong, height: hI)
    func tag(_ title: String, _ sub: String, at p: SIMD2<Float>, dx: CGFloat, dy: CGFloat, leftOf: Bool) {
        let size: CGFloat = 14 * k
        let tw: CGFloat = max(textWidth(title, size: size, bold: true), textWidth(sub, size: 12 * k, bold: false))
        let anchor = CGPoint(x: CGFloat(p.x), y: CGFloat(p.y))
        let x0: CGFloat = leftOf ? anchor.x + dx - tw : anchor.x + dx
        let y0: CGFloat = anchor.y + dy
        ctx.setFillColor(plate)
        ctx.fill(CGRect(x: x0 - 6 * k, y: h - y0 - 34 * k, width: tw + 12 * k, height: 38 * k))
        ctx.setStrokeColor(ink)
        ctx.setLineWidth(1.2 * k)
        line(ctx, CGPoint(x: leftOf ? x0 + tw + 6 * k : x0 - 6 * k, y: y0 + 15 * k), anchor, height: h)
        put(title, ctx, x: x0, top: y0, size: size, bold: true, height: h)
        put(sub, ctx, x: x0, top: y0 + 17 * k, size: 12 * k, height: h)
    }
    tag("taste hair", "one pore, at the tip: sugar", at: tLabel, dx: -60 * k, dy: -26 * k, leftOf: true)
    tag("smell hair", "wall pores, odour arriving", at: sLabel, dx: -150 * k, dy: -120 * k, leftOf: true)
    if let far = scene.odourPaths.last {
        let fp: SIMD2<Float> = projectInset(odourPoint(far, away: odourPathLength - 0.3), height: hI)
        tag("odour molecules", "dots, far too big", at: fp, dx: 24 * k, dy: 40 * k, leftOf: false)
    }
    // Where the inset looks: its title, on a plate inside its top.
    let insetTitle = "the front ant's right antenna tip"
    let itw: CGFloat = textWidth(insetTitle, size: 13 * k, bold: true)
    ctx.setFillColor(plate)
    ctx.fill(CGRect(x: insetC.x - itw / 2 - 8 * k, y: h - (insetC.y - insetR + 44 * k), width: itw + 16 * k, height: 22 * k))
    put(insetTitle, ctx, x: insetC.x - itw / 2, top: insetC.y - insetR + 26 * k, size: 13 * k, bold: true, height: h)

    func centred(_ text: String, _ c: CGPoint, top: CGFloat, size: CGFloat, bold: Bool = false, color: CGColor = ink) {
        let tw: CGFloat = textWidth(text, size: size, bold: bold)
        put(text, ctx, x: c.x - tw / 2, top: top, size: size, bold: bold, color: color, height: h)
    }
    // The fructose: its words on a plate left of its inset, on the card
    // below the apple, where no ant walks (a test checks).
    let mLines: [(String, CGFloat, Bool, Bool)] = [
        ("fructose, C₆H₁₂O₆", 14, true, false),
        ("Golden Delicious's main sugar:", 11.5, false, false),
        ("one enters the pore per touch", 11.5, false, false),
        ("drawn as β-pyranose (68% in water)", 11, false, false),
        ("schematic: the order, not the speed", 11, false, true),
    ]
    let fb: PixelRect = fructoseBlock(height: hI)
    ctx.setFillColor(plate)
    ctx.fill(CGRect(x: CGFloat(fb.x0), y: h - CGFloat(fb.y1), width: CGFloat(fb.x1 - fb.x0), height: CGFloat(fb.y1 - fb.y0)))
    var yy: CGFloat = CGFloat(fb.y0) + 6 * k
    for (t, sz, b, it) in mLines {
        put(t, ctx, x: CGFloat(fb.x1) - 8 * k - textWidth(t, size: sz * k, bold: b, italic: it), top: yy, size: sz * k, bold: b,
            italic: it, color: b ? ink : softInk, height: h)
        yy += (sz + 5) * k
    }
    // The odorants: each one's name on a plate left of the circle, level
    // with it, on the card between the caption and the apple.
    for (name, sub, yA) in odourNames() {
        let cy: CGFloat = odC.y - CGFloat(yA) * odR / CGFloat(odourField / 2)
        let tw: CGFloat = max(textWidth(name, size: 12.5 * k, bold: true), textWidth(sub, size: 11 * k, bold: false))
        let right: CGFloat = odC.x - odR - 12 * k
        ctx.setFillColor(plate)
        ctx.fill(CGRect(x: right - tw - 6 * k, y: h - cy - 18 * k, width: tw + 12 * k, height: 36 * k))
        put(name, ctx, x: right - textWidth(name, size: 12.5 * k, bold: true), top: cy - 15 * k, size: 12.5 * k, bold: true, height: h)
        put(sub, ctx, x: right - textWidth(sub, size: 11 * k, bold: false), top: cy + 2 * k, size: 11 * k, color: softInk, height: h)
    }

    // Title and caption, top left, on the card clear of the apple.
    put("Lasius niger", ctx, x: 50 * k, top: 30 * k, size: 24 * k, italic: true, height: h)
    let lw: CGFloat = textWidth("Lasius niger ", size: 24 * k, bold: false, italic: true)
    put(titleLines[0], ctx, x: 50 * k + lw, top: 30 * k, size: 24 * k, height: h)
    put(titleLines[1], ctx, x: 50 * k, top: 58 * k, size: 24 * k, height: h)
    for (i, l) in captionLines().enumerated() {
        put(l, ctx, x: 50 * k, top: (98 + 18 * CGFloat(i)) * k, size: 13 * k, color: softInk, height: h)
    }
}
