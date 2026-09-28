// What the CPU draws over the GPU's picture: the title, the ring on the face
// where the cut is taken, the inset — a cross-section through the plating,
// drawn to scale — two scale bars, and a small chart of the reflectance
// spectra the colour came from. Text is step 42's CoreText approach, copied.
//
// The inset is a metallographic section as it would look under a reflected-
// light microscope: each polished layer shows the colour of its own normal-
// incidence reflectance, computed from its own measured n and k, so even the
// inset's colours are derived, not picked. Only the mounting resin above the
// surface is a chosen colour.

import CoreGraphics
import CoreText
import Foundation
import Metal
import simd

private let ink = CGColor(srgbRed: 0.10, green: 0.11, blue: 0.12, alpha: 1)
private let softInk = CGColor(srgbRed: 0.10, green: 0.11, blue: 0.12, alpha: 0.62)
private let paper = CGColor(srgbRed: 1, green: 1, blue: 1, alpha: 1)
private let paleInk = CGColor(srgbRed: 1, green: 1, blue: 1, alpha: 0.88)

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

private func line(_ ctx: CGContext, _ a: CGPoint, _ b: CGPoint, height h: CGFloat) {
    ctx.beginPath()
    ctx.move(to: CGPoint(x: a.x, y: h - a.y))
    ctx.addLine(to: CGPoint(x: b.x, y: h - b.y))
    ctx.strokePath()
}

/// A rectangle given in image coordinates (y down), converted for CoreGraphics.
private func flip(_ r: CGRect, _ h: CGFloat) -> CGRect {
    CGRect(x: r.minX, y: h - r.maxY, width: r.width, height: r.height)
}

/// A colour from a linear-sRGB value, sRGB-encoded.
private func cg(_ c: SIMD3<Double>, alpha: CGFloat = 1) -> CGColor {
    CGColor(srgbRed: CGFloat(srgbEncode(c.x)), green: CGFloat(srgbEncode(c.y)), blue: CGFloat(srgbEncode(c.z)), alpha: alpha)
}

// MARK: - geometry the tests read

/// The inset's frame in pixels (y down).
func insetRect(height: Int) -> CGRect {
    let h: CGFloat = CGFloat(height)
    return CGRect(x: CGFloat(insetOrigin.x) * h, y: CGFloat(insetOrigin.y) * h,
                  width: CGFloat(insetSize.x) * h, height: CGFloat(insetSize.y) * h)
}

/// One layer of the inset as drawn: its name, its colour, and the rows it
/// spans in pixels (y down, fractional).
struct DrawnLayer {
    var name: String
    var linear: SIMD3<Double>
    var top: CGFloat
    var bottom: CGFloat
}

/// The micrograph's exposure: a polished section photographed in reflected
/// light, with white set a little above the brightest layer so none clips
/// (gold's red channel is just over 1 — it is outside sRGB). MODEL.
let micrographExposure: Double = 0.9

/// The mounting resin's colour, linear. Chosen: a dark neutral, as epoxy
/// mounts look under the microscope. MODEL.
let resinLinear = SIMD3<Double>(0.018, 0.018, 0.020)

/// The colour of a polished metal section under the microscope: its normal-
/// incidence reflectance, from its own n and k, times the exposure. Always the
/// true physics — the mutants change the lock, not how a section is lit.
func sectionColour(_ name: String) -> SIMD3<Double> {
    let table: [NKRow]
    switch name {
    case "gold": table = goldJC
    case "nickel": table = nickelJC
    default: table = brassQuerry
    }
    return reflectanceRGB(table, cosTheta: 1, mutant: .none) * micrographExposure
}

/// The layers of the inset in pixels, from the plating stack: resin down to
/// the surface, then each metal at its own thickness, the last to the bottom.
func insetLayers(height: Int, mutant: Mutant = activeMutant) -> [DrawnLayer] {
    let r: CGRect = insetRect(height: height)
    let pxPerUm: CGFloat = CGFloat(1 / insetMicrometresPerPixel(height: height))
    var y: CGFloat = r.minY + CGFloat(insetSurfaceMicrometres) * pxPerUm
    var out: [DrawnLayer] = [DrawnLayer(name: "resin", linear: resinLinear, top: r.minY, bottom: y)]
    for layer in platingStack(mutant) {
        let bottom: CGFloat = layer.micrometres.isFinite ? min(y + CGFloat(layer.micrometres) * pxPerUm, r.maxY) : r.maxY
        out.append(DrawnLayer(name: layer.name, linear: sectionColour(layer.name), top: y, bottom: bottom))
        y = bottom
        if y >= r.maxY { break }
    }
    return out
}

/// The column of the inset kept clear of the fibre and the labels, where the
/// tests read the layers back off the picture.
func insetMeasureColumn(height: Int) -> Int {
    let r: CGRect = insetRect(height: height)
    return Int(r.minX + 0.62 * r.width)
}

/// Where each scale bar was drawn, and how long it is, in pixels.
struct ScaleBar {
    var label: String
    var x: CGFloat
    var y: CGFloat         // the top row of the bar
    var pixels: CGFloat
}

func mainScaleBar(height: Int) -> ScaleBar {
    let h: CGFloat = CGFloat(height)
    let k: CGFloat = h / 1080
    let px: CGFloat = CGFloat(mainBarMillimetres * mainPixelsPerMillimetre(height: height))
    return ScaleBar(label: String(format: "%g mm", mainBarMillimetres), x: 64 * k, y: h - 58 * k, pixels: px)
}

func insetScaleBar(height: Int) -> ScaleBar {
    let r: CGRect = insetRect(height: height)
    let k: CGFloat = CGFloat(height) / 1080
    let px: CGFloat = CGFloat(insetBarMicrometres / insetMicrometresPerPixel(height: height))
    return ScaleBar(label: String(format: "%g µm", insetBarMicrometres), x: r.maxX - px - 26 * k,
                    y: r.maxY - 28 * k, pixels: px)
}

/// A scale bar: a rule with end ticks and a label above.
private func drawBar(_ ctx: CGContext, _ b: ScaleBar, k: CGFloat, height h: CGFloat, color: CGColor = ink) {
    ctx.setFillColor(color)
    ctx.fill(flip(CGRect(x: b.x, y: b.y, width: b.pixels, height: 3 * k), h))
    ctx.fill(flip(CGRect(x: b.x, y: b.y - 6 * k, width: 2 * k, height: 9 * k), h))
    ctx.fill(flip(CGRect(x: b.x + b.pixels - 2 * k, y: b.y - 6 * k, width: 2 * k, height: 9 * k), h))
    let w: CGFloat = textWidth(b.label, size: 15 * k, bold: true)
    put(b.label, ctx, x: b.x + b.pixels / 2 - w / 2, top: b.y - 26 * k, size: 15 * k, bold: true, color: color, height: h)
}

/// The chart of reflectance against wavelength, in pixels (y down).
func chartRect(height: Int) -> CGRect {
    let r: CGRect = insetRect(height: height)
    let k: CGFloat = CGFloat(height) / 1080
    return CGRect(x: r.minX, y: r.maxY + 34 * k, width: r.width, height: CGFloat(height) - r.maxY - 52 * k)
}

// MARK: - drawing

func annotate(_ image: LockImage, mutant: Mutant = activeMutant) {
    let w: Int = image.width
    let hI: Int = image.height
    guard let ctx = CGContext(data: image.pixels.contents(), width: w, height: hI, bitsPerComponent: 8,
                              bytesPerRow: w * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                              bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return }
    let h: CGFloat = CGFloat(hI)
    let k: CGFloat = h / 1080
    ctx.setLineCap(.round)
    ctx.setShouldAntialias(true)

    // Title and the one-paragraph argument.
    put("A gold-plated padlock", ctx, x: 64 * k, top: 40 * k, size: 30 * k, bold: true, height: h)
    let lines: [String] = [
        "No colour here was picked. Every reflection is white light times gold's reflectance, computed from its",
        "measured n + ik (Johnson & Christy 1972) through the Fresnel equations for a metal, wavelength by wavelength.",
        "Gold absorbs blue above ~2.4 eV; relativity pulls that edge down from the ultraviolet, where silver's stays.",
    ]
    for (i, s) in lines.enumerated() {
        put(s, ctx, x: 64 * k, top: (84 + 22 * CGFloat(i)) * k, size: 16 * k, color: softInk, height: h)
    }

    // The ring on the face where the cut is taken, and lines to the inset.
    let cp: SIMD2<Float> = project(lockToWorld(cutPointLocal), width: w, height: hI)
    let c = CGPoint(x: CGFloat(cp.x), y: CGFloat(cp.y))
    let ringR: CGFloat = 13 * k
    let r: CGRect = insetRect(height: hI)
    ctx.setStrokeColor(paleInk)
    ctx.setLineWidth(1.5 * k)
    for corner in [CGPoint(x: r.minX, y: r.minY), CGPoint(x: r.minX, y: r.maxY)] {
        let dx: CGFloat = corner.x - c.x
        let dy: CGFloat = corner.y - c.y
        let d: CGFloat = (dx * dx + dy * dy).squareRoot()
        line(ctx, CGPoint(x: c.x + dx / d * ringR, y: c.y + dy / d * ringR), corner, height: h)
    }
    ctx.setStrokeColor(paper)
    ctx.setLineWidth(2.5 * k)
    ctx.strokeEllipse(in: flip(CGRect(x: c.x - ringR, y: c.y - ringR, width: 2 * ringR, height: 2 * ringR), h))

    // The inset: the layers, edge to edge, at their true thicknesses.
    let layers: [DrawnLayer] = insetLayers(height: hI, mutant: mutant)
    ctx.saveGState()
    ctx.clip(to: flip(r, h))
    ctx.setShouldAntialias(false)
    for l in layers {
        ctx.setFillColor(cg(l.linear))
        ctx.fill(flip(CGRect(x: r.minX, y: l.top, width: r.width, height: l.bottom - l.top), h))
    }
    ctx.setShouldAntialias(true)

    // A cotton fibre in the resin, to the same scale.
    let pxPerUm: CGFloat = CGFloat(1 / insetMicrometresPerPixel(height: hI))
    let surfaceY: CGFloat = layers[0].bottom
    let fibreR: CGFloat = CGFloat(cottonFibreMicrometres) / 2 * pxPerUm
    let fibreC = CGPoint(x: r.minX + 0.30 * r.width, y: surfaceY - fibreR - 0.9 * pxPerUm)
    ctx.setStrokeColor(CGColor(srgbRed: 0.92, green: 0.92, blue: 0.88, alpha: 0.9))
    ctx.setLineWidth(2 * k)
    ctx.setLineDash(phase: 0, lengths: [7 * k, 5 * k])
    ctx.strokeEllipse(in: flip(CGRect(x: fibreC.x - fibreR, y: fibreC.y - fibreR, width: 2 * fibreR, height: 2 * fibreR), h))
    ctx.setLineDash(phase: 0, lengths: [])
    let f1: String = "a cotton fibre, 15 µm across"
    let f2: String = "(Xiong et al. 2026)"
    put(f1, ctx, x: fibreC.x - textWidth(f1, size: 15 * k, bold: true) / 2, top: fibreC.y - 18 * k, size: 15 * k,
        bold: true, color: paleInk, height: h)
    put(f2, ctx, x: fibreC.x - textWidth(f2, size: 13 * k, bold: false) / 2, top: fibreC.y + 2 * k, size: 13 * k,
        color: paleInk, height: h)

    // Labels: the gold's from the resin above it, the others on their layers.
    let labelX: CGFloat = r.minX + 0.665 * r.width
    if let gold = layers.first(where: { $0.name == "gold" }) {
        let y: CGFloat = min(gold.top, r.maxY) - 58 * k
        put("gold, \(format(platingStack(mutant)[0].micrometres)) µm", ctx, x: labelX, top: y, size: 17 * k, bold: true,
            color: paper, height: h)
        put("thinner than the fibre is wide", ctx, x: labelX, top: y + 21 * k, size: 13 * k, color: paleInk, height: h)
        ctx.setStrokeColor(paper)
        ctx.setLineWidth(1.4 * k)
        line(ctx, CGPoint(x: labelX - 6 * k, y: y + 12 * k), CGPoint(x: labelX - 12 * k, y: gold.top - 2 * k), height: h)
    }
    if let ni = layers.first(where: { $0.name == "nickel" }) {
        let mid: CGFloat = (ni.top + ni.bottom) / 2
        put("nickel, \(format(nickelMicrometres)) µm", ctx, x: labelX, top: mid - 30 * k, size: 17 * k, bold: true, height: h)
        put("a barrier: keeps the brass's", ctx, x: labelX, top: mid - 8 * k, size: 13 * k, height: h)
        put("copper out of the gold", ctx, x: labelX, top: mid + 8 * k, size: 13 * k, height: h)
    }
    if let brass = layers.first(where: { $0.name == "brass" }), brass.bottom - brass.top > 40 * k {
        put("brass, the lock's body", ctx, x: r.minX + 18 * k, top: brass.top + 18 * k, size: 17 * k, bold: true, height: h)
    }
    drawBar(ctx, insetScaleBar(height: hI), k: k, height: h)
    ctx.restoreGState()

    // Frame and heading.
    ctx.setStrokeColor(paper)
    ctx.setLineWidth(4 * k)
    ctx.stroke(flip(r.insetBy(dx: -2 * k, dy: -2 * k), h))
    put("Cut through the surface at the ring, true to scale", ctx, x: r.minX, top: r.minY - 30 * k, size: 17 * k,
        bold: true, height: h)

    // The main view's scale bar, for lengths square to the camera at the lock.
    let bar: ScaleBar = mainScaleBar(height: hI)
    drawBar(ctx, bar, k: k, height: h)
    put("at the lock", ctx, x: bar.x + bar.pixels + 12 * k, top: bar.y - 7 * k, size: 13 * k, color: softInk, height: h)

    drawChart(ctx, chartRect(height: hI), k: k, height: h)
}

private func format(_ v: Float) -> String { String(format: "%g", v) }

/// Reflectance at normal incidence, 400–700 nm, for gold and silver, from
/// their n and k: the colour's cause in one picture.
private func drawChart(_ ctx: CGContext, _ box: CGRect, k: CGFloat, height h: CGFloat) {
    ctx.setFillColor(CGColor(srgbRed: 1, green: 1, blue: 1, alpha: 0.86))
    ctx.fill(flip(box, h))
    let plot = CGRect(x: box.minX + 56 * k, y: box.minY + 30 * k, width: box.width - 130 * k, height: box.height - 62 * k)
    put("Reflectance at normal incidence, from n and k", ctx, x: box.minX + 12 * k, top: box.minY + 8 * k, size: 14 * k,
        bold: true, height: h)
    func px(_ nm: Double, _ r: Double) -> CGPoint {
        let x: CGFloat = plot.minX + CGFloat((nm - 400) / 300) * plot.width
        let y: CGFloat = plot.maxY - CGFloat(r) * plot.height
        return CGPoint(x: x, y: y)
    }
    // Axes.
    ctx.setStrokeColor(softInk)
    ctx.setLineWidth(1 * k)
    line(ctx, px(400, 0), px(700, 0), height: h)
    line(ctx, px(400, 0), px(400, 1), height: h)
    for f in [0.0, 0.5, 1.0] {
        put(String(format: "%.0f%%", f * 100), ctx, x: plot.minX - 44 * k, top: px(400, f).y - 7 * k, size: 11 * k,
            color: softInk, height: h)
    }
    for nm in [400.0, 500.0, 600.0, 700.0] {
        let p: CGPoint = px(nm, 0)
        put(String(format: "%.0f nm", nm), ctx, x: p.x - 18 * k, top: p.y + 4 * k, size: 11 * k, color: softInk, height: h)
    }
    // The visible spectrum under the axis, as a thin strip.
    for i in 0..<60 {
        let nm: Double = 400 + Double(i) * 5
        let v = linearSRGB(xyz: cieXYZ1931[min(Int((nm - 380) / 10), 40)])
        let m: Double = max(v.max(), 1e-6)
        let c = SIMD3<Double>(max(v.x, 0) / m, max(v.y, 0) / m, max(v.z, 0) / m)
        ctx.setFillColor(cg(c * 0.8))
        let a: CGPoint = px(nm, 0)
        let b: CGPoint = px(nm + 5, 0)
        ctx.fill(flip(CGRect(x: a.x, y: a.y - 4 * k, width: b.x - a.x + 0.5, height: 3 * k), h))
    }
    // The curves.
    let series: [(String, [NKRow], CGColor)] = [
        ("gold", goldJC, CGColor(srgbRed: 0.80, green: 0.56, blue: 0.10, alpha: 1)),
        ("silver", silverJC, CGColor(srgbRed: 0.45, green: 0.47, blue: 0.52, alpha: 1)),
    ]
    for (name, table, colour) in series {
        ctx.setStrokeColor(colour)
        ctx.setLineWidth(3 * k)
        ctx.beginPath()
        for i in 0...60 {
            let nm: Double = 400 + Double(i) * 5
            let v = nk(table, nanometres: nm)
            let rr: Double = fresnelConductor(n: v.n, k: v.k, cosTheta: 1, mutant: .none)
            let p: CGPoint = px(nm, rr)
            if i == 0 { ctx.move(to: CGPoint(x: p.x, y: h - p.y)) } else { ctx.addLine(to: CGPoint(x: p.x, y: h - p.y)) }
        }
        ctx.strokePath()
        let v = nk(table, nanometres: 700)
        let end: CGPoint = px(700, fresnelConductor(n: v.n, k: v.k, cosTheta: 1, mutant: .none))
        put(name, ctx, x: end.x + 8 * k, top: end.y - (name == "gold" ? 2 : 16) * k, size: 14 * k, bold: true,
            color: colour, height: h)
    }
    // The edge.
    let edgeNM: Double = photonEV / interbandOnsetEV(goldJC) * 1000
    ctx.setStrokeColor(softInk)
    ctx.setLineDash(phase: 0, lengths: [4 * k, 4 * k])
    line(ctx, px(edgeNM, 0.02), px(edgeNM, 0.98), height: h)
    ctx.setLineDash(phase: 0, lengths: [])
    let e: Double = interbandOnsetEV(goldJC)
    let tp: CGPoint = px(edgeNM, 0.36)
    put(String(format: "gold's interband edge, %.1f eV (%.0f nm):", e, edgeNM), ctx, x: tp.x + 10 * k, top: tp.y - 8 * k,
        size: 12 * k, color: softInk, height: h)
    put("at shorter wavelengths, blue is absorbed", ctx, x: tp.x + 10 * k, top: tp.y + 8 * k, size: 12 * k,
        color: softInk, height: h)
}
