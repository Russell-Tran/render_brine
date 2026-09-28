// Step 41: step 40's labels, copied; as in step 30 the contact ring rides on
// the moving tip, the leaf's own labels in the inset ride on the leaf (and
// go when it has dropped out of view), and a line says what is slowed.
//
// What the CPU draws over the GPU's picture: the rings round the two insets,
// the lines that say where each one looks, three scale bars and a few labels.
// Text drawing is step 19's CoreText approach, copied rather than shared.
//
// Every anchor is a projected world point — the contact, the tip pore — so no
// line can drift away from what it points at.

import CoreGraphics
import CoreText
import Foundation
import Metal
import simd

private let ink = CGColor(srgbRed: 0.13, green: 0.14, blue: 0.16, alpha: 1)
private let softInk = CGColor(srgbRed: 0.13, green: 0.14, blue: 0.16, alpha: 0.55)
private let ringInk = CGColor(srgbRed: 1, green: 1, blue: 1, alpha: 1)

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

/// A scale bar: a rule of `pixels` length with end ticks and a label above.
private func scaleBar(_ ctx: CGContext, label: String, pixels: CGFloat, x: CGFloat, y: CGFloat, k: CGFloat,
                      height h: CGFloat, color: CGColor = ink, backed: Bool = false) {
    if backed {
        // A pale plate behind bar and label, where the view behind is busy.
        ctx.setFillColor(CGColor(srgbRed: 1, green: 1, blue: 1, alpha: 0.82))
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

/// A line from `a` to `b`, drawn only where it lies outside the circle
/// (c, r): a callout that starts inside one inset must not cross its picture.
private func lineOutside(_ ctx: CGContext, _ a: CGPoint, _ b: CGPoint, circle c: CGPoint, radius r: CGFloat, height h: CGFloat) {
    let d = CGPoint(x: b.x - a.x, y: b.y - a.y)
    let f = CGPoint(x: a.x - c.x, y: a.y - c.y)
    let A: CGFloat = d.x * d.x + d.y * d.y
    let B: CGFloat = 2 * (f.x * d.x + f.y * d.y)
    let C: CGFloat = f.x * f.x + f.y * f.y - r * r
    let disc: CGFloat = B * B - 4 * A * C
    var start: CGFloat = 0
    if disc > 0 && C < 0 { start = (-B + disc.squareRoot()) / (2 * A) }   // a is inside: leave from the exit
    if start >= 1 { return }
    line(ctx, CGPoint(x: a.x + d.x * start, y: a.y + d.y * start), b, height: h)
}

private func circle(_ ctx: CGContext, centre c: CGPoint, radius r: CGFloat, height h: CGFloat) {
    ctx.strokeEllipse(in: CGRect(x: c.x - r, y: h - c.y - r, width: 2 * r, height: 2 * r))
}

/// The two outer tangent lines between two circles, as pairs of points: the
/// classic magnifier callout.
private func tangents(_ c1: CGPoint, _ r1: CGFloat, _ c2: CGPoint, _ r2: CGFloat) -> [(CGPoint, CGPoint)] {
    let dx: CGFloat = c2.x - c1.x
    let dy: CGFloat = c2.y - c1.y
    let d: CGFloat = (dx * dx + dy * dy).squareRoot()
    let base: CGFloat = atan2(dy, dx)
    let off: CGFloat = acos((r1 - r2) / d)
    return [base + off, base - off].map { a in
        (CGPoint(x: c1.x + r1 * cos(a), y: c1.y + r1 * sin(a)), CGPoint(x: c2.x + r2 * cos(a), y: c2.y + r2 * sin(a)))
    }
}

/// Pixel sizes of each view, for the scale bars and the tests.
func mainMillimetresPerPixel(width: Int) -> Float { mainViewWidth / Float(width) }
func insetMicrometresPerPixel(height: Int) -> Float { insetField / (2 * insetRadius * Float(height)) }
func moleculeAngstromsPerPixel(height: Int) -> Float { moleculeField / (2 * moleculeRadius * Float(height)) }

func annotate(_ image: AntImage, scene: Scene, frame: FrameState) {
    let w: Int = image.width
    let hI: Int = image.height
    guard let ctx = CGContext(data: image.pixels.contents(), width: w, height: hI, bitsPerComponent: 8,
                              bytesPerRow: w * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                              bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return }
    let h: CGFloat = CGFloat(hI)
    let k: CGFloat = h / 1080
    ctx.setLineCap(.round)

    // Where the antenna touches, in the main view — or, lifted, the point
    // of the tip that will touch: the ring rides on the tip, as the inset does.
    let tip: Antenna = frame.ant.antennae[0]
    let touch: SIMD3<Float> = tip.tipCentre - SIMD3<Float>(0, tip.tipRadius, 0)
    let tp: SIMD2<Float> = projectMain(touch, width: w, height: hI)
    let touchPt = CGPoint(x: CGFloat(tp.x), y: CGFloat(tp.y))
    let spotR: CGFloat = 26 * k
    let insetC = CGPoint(x: CGFloat(insetCentre.x) * h, y: CGFloat(insetCentre.y) * h)
    let insetR: CGFloat = CGFloat(insetRadius) * h
    let molC = CGPoint(x: CGFloat(moleculeCentre.x) * h, y: CGFloat(moleculeCentre.y) * h)
    let molR: CGFloat = CGFloat(moleculeRadius) * h

    // Callout from the contact to the inset.
    ctx.setStrokeColor(softInk)
    ctx.setLineWidth(1.5 * k)
    for (a, b) in tangents(touchPt, spotR, insetC, insetR) { line(ctx, a, b, height: h) }
    ctx.setStrokeColor(ink)
    ctx.setLineWidth(2 * k)
    circle(ctx, centre: touchPt, radius: spotR, height: h)

    // From the tip pore in the inset to the molecule inset.
    let apex: SIMD3<Float> = scene.hairs[0].tip + simd_normalize(scene.hairs[0].tip - scene.hairs[0].base) * scene.hairs[0].tipRadius
    let ap: SIMD2<Float> = projectInset(apex, height: hI)
    let poreR: CGFloat = 30 * k
    let porePt = CGPoint(x: CGFloat(ap.x), y: CGFloat(ap.y))
    ctx.setStrokeColor(softInk)
    ctx.setLineWidth(1.5 * k)
    for (a, b) in tangents(porePt, poreR, molC, molR) { line(ctx, a, b, height: h) }
    ctx.setStrokeColor(ringInk)
    ctx.setLineWidth(2 * k)
    circle(ctx, centre: porePt, radius: poreR, height: h)

    // Lifted, the inset (which rides on the tip) has left the leaf behind:
    // say how far below it is, from the same drop that moves it there.
    let drop: Float = frame.leaf.translation.y
    if drop > 6 {
        let gap = String(format: "↓ leaf %.0f µm below", drop)
        put(gap, ctx, x: porePt.x + poreR + 12 * k, top: porePt.y + 6 * k, size: 14 * k, color: softInk, height: h)
    }

    // Rings round the insets: white inside a thin dark line.
    for (c, r) in [(insetC, insetR), (molC, molR)] {
        ctx.setStrokeColor(ringInk)
        ctx.setLineWidth(6 * k)
        circle(ctx, centre: c, radius: r, height: h)
        ctx.setStrokeColor(ink)
        ctx.setLineWidth(1.5 * k)
        circle(ctx, centre: c, radius: r + 3.5 * k, height: h)
    }

    // Scale bars, one per view, each from that view's own pixel size.
    let mmPx: CGFloat = CGFloat(1.0 / mainMillimetresPerPixel(width: w))
    scaleBar(ctx, label: "1 mm", pixels: mmPx, x: 60 * k, y: h - 60 * k, k: k, height: h)
    let umPx: CGFloat = CGFloat(5.0 / insetMicrometresPerPixel(height: hI))
    scaleBar(ctx, label: "5 µm", pixels: umPx, x: insetC.x - umPx / 2, y: insetC.y + insetR - 34 * k, k: k, height: h)
    let nmPx: CGFloat = CGFloat(moleculeBarNanometres * 10 / moleculeAngstromsPerPixel(height: hI))
    scaleBar(ctx, label: String(format: "%g nm", moleculeBarNanometres), pixels: nmPx, x: molC.x - nmPx / 2, y: molC.y + molR - 26 * k, k: k, height: h, backed: true)

    // Labels: the two hair types, the molecule, a one-line caption.
    let taste: Sensillum = scene.hairs[0]
    let smell: Sensillum = scene.hairs[1]
    let tLabel: SIMD2<Float> = projectInset(taste.base + (taste.tip - taste.base) * 0.62, height: hI)
    let sLabel: SIMD2<Float> = projectInset(smell.base + (smell.tip - smell.base) * 0.55, height: hI)
    func tag(_ title: String, _ sub: String, at p: SIMD2<Float>, dx: CGFloat, dy: CGFloat, leftOf: Bool) {
        let size: CGFloat = 16 * k
        let tw: CGFloat = max(textWidth(title, size: size, bold: true), textWidth(sub, size: 13.5 * k, bold: false))
        let anchor = CGPoint(x: CGFloat(p.x), y: CGFloat(p.y))
        let x0: CGFloat = leftOf ? anchor.x + dx - tw : anchor.x + dx
        let y0: CGFloat = anchor.y + dy
        ctx.setFillColor(CGColor(srgbRed: 1, green: 1, blue: 1, alpha: 0.82))
        ctx.fill(CGRect(x: x0 - 6 * k, y: h - y0 - 38 * k, width: tw + 12 * k, height: 42 * k))
        ctx.setStrokeColor(ink)
        ctx.setLineWidth(1.2 * k)
        line(ctx, CGPoint(x: leftOf ? x0 + tw + 6 * k : x0 - 6 * k, y: y0 + 17 * k), anchor, height: h)
        put(title, ctx, x: x0, top: y0, size: size, bold: true, height: h)
        put(sub, ctx, x: x0, top: y0 + 19 * k, size: 13.5 * k, height: h)
    }
    tag("taste hair", "one pore, at the tip: tastes the wax", at: tLabel, dx: -70 * k, dy: -30 * k, leftOf: true)
    tag("smell hair", "wall pores: little to smell", at: sLabel, dx: 40 * k, dy: -60 * k, leftOf: false)
    // The stoma and the wax. Step 41: their anchors are points on the leaf,
    // carried into the inset by this frame's transform (the inverse of
    // inset → leaf), so each label points at its own thing as the leaf falls
    // away; a label whose anchor has left the inset is not drawn. And once
    // the leaf has dropped more than 2 µm the labels go, so they never sweep
    // across the scale bar and the hair labels as the leaf falls and returns.
    func onLeaf(_ l: SIMD3<Float>) -> SIMD2<Float>? {
        if simd_length(frame.leaf.translation) > 2 { return nil }
        let q: SIMD3<Float> = frame.leaf.rotation.transpose * (l - frame.leaf.translation)
        let px: SIMD2<Float> = projectInset(q, height: hI)
        let off: SIMD2<Float> = px - SIMD2<Float>(Float(insetC.x), Float(insetC.y))
        return simd_length(off) < Float(insetR) * 0.92 ? px : nil
    }
    if let st = onLeaf(stomaCentre + SIMD3<Float>(0, 1.2, 0)) {
        tag("stoma", "guard cells round a pore into the leaf", at: st, dx: 10 * k, dy: -95 * k, leftOf: true)
    }
    if let wx = onLeaf(SIMD3<Float>(5.0, insetRidgeHeight(SIMD2<Float>(5.0, 3.0)) + 0.1, 3.0)) {
        tag("epicuticular wax", "0.2 µm roughness, measured by AFM", at: wx, dx: -20 * k, dy: 16 * k, leftOf: true)
    }

    func centred(_ text: String, _ c: CGPoint, top: CGFloat, size: CGFloat, bold: Bool = false, color: CGColor = ink) {
        let tw: CGFloat = textWidth(text, size: size, bold: bold)
        put(text, ctx, x: c.x - tw / 2, top: top, size: size, bold: bold, color: color, height: h)
    }
    centred("hentriacontane, C₃₁H₆₄", molC, top: molC.y - molR + 16 * k, size: 15 * k, bold: true)
    centred("a leaf-wax alkane: banana wax is mostly", molC, top: molC.y - molR + 36 * k, size: 12.5 * k, color: softInk)
    centred("alkanes, primary alcohols and fatty acids", molC, top: molC.y - molR + 52 * k, size: 12.5 * k, color: softInk)

    // The smell side: a ring on the smell hair where the odour dots arrive,
    // a callout to the odour inset (drawn only outside the micrometre inset),
    // and a note that the dots are far larger than molecules.
    if !frame.volatiles.isEmpty {
        let odC = CGPoint(x: CGFloat(odourCentre.x) * h, y: CGFloat(odourCentre.y) * h)
        let odR: CGFloat = CGFloat(odourRadius) * h
        var mid = SIMD3<Float>(0, 0, 0)
        for v in frame.volatiles.prefix(2) { mid += v.position }
        let arrive: SIMD2<Float> = projectInset(mid / 2, height: hI)
        let arrivePt = CGPoint(x: CGFloat(arrive.x), y: CGFloat(arrive.y))
        let ringR: CGFloat = 46 * k
        ctx.setStrokeColor(softInk)
        ctx.setLineWidth(1.5 * k)
        for (a, b) in tangents(arrivePt, ringR, odC, odR) {
            lineOutside(ctx, a, b, circle: insetC, radius: insetR + 3.5 * k, height: h)
        }
        ctx.setStrokeColor(ringInk)
        ctx.setLineWidth(2 * k)
        circle(ctx, centre: arrivePt, radius: ringR, height: h)
        ctx.setStrokeColor(ringInk)
        ctx.setLineWidth(6 * k)
        circle(ctx, centre: odC, radius: odR, height: h)
        ctx.setStrokeColor(ink)
        ctx.setLineWidth(1.5 * k)
        circle(ctx, centre: odC, radius: odR + 3.5 * k, height: h)
        let odPx: CGFloat = CGFloat(odourBarNanometres * 10 / (odourField / (2 * odourRadius * Float(hI))))
        scaleBar(ctx, label: String(format: "%g nm", odourBarNanometres), pixels: odPx, x: odC.x - odPx / 2,
                 y: odC.y + odR - 22 * k, k: k, height: h)
        centred("odorant (a mutant: none here)", odC, top: odC.y - odR + 26 * k, size: 13.5 * k, bold: true)
        centred("this inset should not exist", odC, top: odC.y - odR + 44 * k, size: 11.5 * k, color: softInk)
        // Which dot is which: a leader to the lowest one still on its way.
        // Step 41: in typed steps, for the compiler.
        func toPore(_ v: Volatile) -> Float {
            let mouth: SIMD3<Float> = smell.wallPore(row: v.pore.row, column: v.pore.column).mouth
            return simd_length(v.position - mouth)
        }
        if let far = frame.volatiles.max(by: { (a: Volatile, b: Volatile) -> Bool in toPore(a) < toPore(b) }) {
            let fp: SIMD2<Float> = projectInset(far.position, height: hI)
            tag("odour molecules", "dots, hundreds of times too big", at: fp, dx: 40 * k, dy: 30 * k, leftOf: false)
        }
    }

    put("Lasius niger", ctx, x: 60 * k, top: 46 * k, size: 26 * k, bold: false, italic: true, height: h)
    let lw: CGFloat = textWidth("Lasius niger", size: 26 * k, bold: false, italic: true)
    put(" worker making contact with the surface of a banana leaf", ctx, x: 60 * k + lw, top: 46 * k, size: 26 * k, height: h)
    put("A leaf is wax, not food. The ant feels and tastes its surface.", ctx, x: 60 * k, top: 82 * k, size: 16 * k,
        color: softInk, height: h)
    let rate = String(format: "Tapping, slowed ×%.0f: a real tap takes about ¼ s (ants antennate at %.0f–%.0f strokes a second).",
                      slowdown, realStrokeRange.lowerBound, realStrokeRange.upperBound)
    put(rate, ctx, x: 60 * k, top: 106 * k, size: 16 * k, color: softInk, height: h)
}
