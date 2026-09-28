// What the CPU draws over the GPU's picture: rings round the three insets,
// the lines that say where each one looks, four scale bars, and labels.
// Text is step 19's CoreText approach, copied (by way of step 42).
//
// Every anchor is a projected world point — a hair's root, the fibre's
// reversal, its cut end — so no line can drift away from what it points at.

import CoreGraphics
import CoreText
import Foundation
import Metal
import simd

private let ink = CGColor(srgbRed: 0.12, green: 0.13, blue: 0.15, alpha: 1)
private let softInk = CGColor(srgbRed: 0.12, green: 0.13, blue: 0.15, alpha: 0.62)
private let paper = CGColor(srgbRed: 1, green: 1, blue: 1, alpha: 0.86)
private let white = CGColor(srgbRed: 1, green: 1, blue: 1, alpha: 1)

/// One drawn label: its text and the box it occupies (image coordinates,
/// y down) — returned so the tests can check they fit and don't collide.
struct Label {
    var text: String
    var box: CGRect
    var view: Int          // 1 main, 2 yarn inset, 3 cut inset, 4 molecule, 0 page
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

/// Pixel sizes of each view, for the scale bars and the tests.
func mainMillimetresPerPixel(width: Int) -> Float { mainViewWidth / Float(width) }
func insetMicrometresPerPixel(height: Int) -> Float { insetField / (2 * insetRadius * Float(height)) }
func cutMicrometresPerPixel(height: Int) -> Float { cutField / (2 * cutRadius * Float(height)) }
func moleculeAngstromsPerPixel(height: Int) -> Float { moleculeField / (2 * moleculeRadius * Float(height)) }

/// The scale bars' lengths, in each view's own unit.
let mainBarMillimetres: Float = 1
let insetBarMicrometres: Float = 100
let cutBarMicrometres: Float = 10

/// A point along a ribbon at arc length s.
func ribbonPoint(_ r: Ribbon, at s: Float) -> SIMD3<Float> {
    for k in 1..<r.points.count where r.arc[k] >= s {
        let f: Float = (s - r.arc[k - 1]) / max(r.arc[k] - r.arc[k - 1], 1e-6)
        return r.points[k - 1] + (r.points[k] - r.points[k - 1]) * f
    }
    return r.points[r.points.count - 1]
}

/// The hair the main view's ring picks out: the one whose root lands
/// nearest a spot just left of the yarn inset.
func calloutHair(_ scene: Scene, width: Int, height: Int) -> Int {
    let h: Float = Float(height)
    let want = SIMD2<Float>(0.80 * h, 0.44 * h)
    var best: Int = 0
    var bestD: Float = .infinity
    for (i, hair) in scene.hairs.enumerated() {
        let p: SIMD2<Float> = projectMain(hair.ribbon.points[0], width: width, height: height)
        let d: Float = simd_distance(p, want)
        if d < bestD { bestD = d; best = i }
    }
    return best
}

@discardableResult
func annotate(_ image: SockImage, scene: Scene) -> [Label] {
    let w: Int = image.width
    let hI: Int = image.height
    var labels: [Label] = []
    guard let ctx = CGContext(data: image.pixels.contents(), width: w, height: hI, bitsPerComponent: 8,
                              bytesPerRow: w * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                              bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return [] }
    let h: CGFloat = CGFloat(hI)
    let k: CGFloat = h / 1080
    ctx.setLineCap(.round)

    func pt(_ v: SIMD2<Float>) -> CGPoint { CGPoint(x: CGFloat(v.x), y: CGFloat(v.y)) }
    func put(_ s: String, x: CGFloat, top: CGFloat, size: CGFloat, bold: Bool = false, italic: Bool = false, color: CGColor = ink) {
        ctx.textPosition = CGPoint(x: x, y: h - top - size)
        CTLineDraw(ctLine(s, size: size, bold: bold, italic: italic, color: color), ctx)
    }
    func line(_ a: CGPoint, _ b: CGPoint) {
        ctx.beginPath()
        ctx.move(to: CGPoint(x: a.x, y: h - a.y))
        ctx.addLine(to: CGPoint(x: b.x, y: h - b.y))
        ctx.strokePath()
    }
    /// A line from a to b drawn only where it lies outside the circle
    /// (c, r): a callout that starts inside an inset must not cross its picture.
    func lineOutside(_ a: CGPoint, _ b: CGPoint, circle c: CGPoint, radius r: CGFloat) {
        let d = CGPoint(x: b.x - a.x, y: b.y - a.y)
        let f = CGPoint(x: a.x - c.x, y: a.y - c.y)
        let A: CGFloat = d.x * d.x + d.y * d.y
        let B: CGFloat = 2 * (f.x * d.x + f.y * d.y)
        let C: CGFloat = f.x * f.x + f.y * f.y - r * r
        let disc: CGFloat = B * B - 4 * A * C
        var start: CGFloat = 0
        if disc > 0 && C < 0 { start = (-B + disc.squareRoot()) / (2 * A) }
        if start >= 1 { return }
        line(CGPoint(x: a.x + d.x * start, y: a.y + d.y * start), b)
    }
    func circle(_ c: CGPoint, _ r: CGFloat) {
        ctx.strokeEllipse(in: CGRect(x: c.x - r, y: h - c.y - r, width: 2 * r, height: 2 * r))
    }
    func tangents(_ c1: CGPoint, _ r1: CGFloat, _ c2: CGPoint, _ r2: CGFloat) -> [(CGPoint, CGPoint)] {
        let dx: CGFloat = c2.x - c1.x, dy: CGFloat = c2.y - c1.y
        let d: CGFloat = (dx * dx + dy * dy).squareRoot()
        let base: CGFloat = atan2(dy, dx)
        let off: CGFloat = acos(max(min((r1 - r2) / d, 1), -1))
        return [base + off, base - off].map { a in
            (CGPoint(x: c1.x + r1 * cos(a), y: c1.y + r1 * sin(a)), CGPoint(x: c2.x + r2 * cos(a), y: c2.y + r2 * sin(a)))
        }
    }
    /// A two-line tag on a pale plate, with a leader to `anchor`. The plate
    /// sits at (x0, y0) top-left; the leader leaves from its nearer side.
    func tag(_ title: String, _ sub: String, anchor: CGPoint, x0: CGFloat, y0: CGFloat, view: Int, size: CGFloat = 15.5) {
        let ts: CGFloat = size * k, ss: CGFloat = (size - 3) * k
        let lines: [String] = sub.isEmpty ? [] : sub.components(separatedBy: "\n")
        var tw: CGFloat = textWidth(title, size: ts, bold: true)
        for l in lines { tw = max(tw, textWidth(l, size: ss, bold: false)) }
        let hgt: CGFloat = ts + 8 * k + CGFloat(lines.count) * (ss + 4 * k)
        let box = CGRect(x: x0 - 6 * k, y: y0 - 4 * k, width: tw + 12 * k, height: hgt)
        ctx.setFillColor(paper)
        ctx.fill(CGRect(x: box.minX, y: h - box.maxY, width: box.width, height: box.height))
        ctx.setStrokeColor(ink)
        ctx.setLineWidth(1.3 * k)
        let fromX: CGFloat = anchor.x < box.midX ? box.minX : box.maxX
        let fromY: CGFloat = min(max(anchor.y, box.minY + 6 * k), box.maxY - 6 * k)
        if !box.contains(anchor) { line(CGPoint(x: fromX, y: fromY), anchor) }
        ctx.setFillColor(ink)
        ctx.fillEllipse(in: CGRect(x: anchor.x - 2.5 * k, y: h - anchor.y - 2.5 * k, width: 5 * k, height: 5 * k))
        put(title, x: x0, top: y0, size: ts, bold: true)
        for (i, l) in lines.enumerated() {
            put(l, x: x0, top: y0 + ts + 6 * k + CGFloat(i) * (ss + 4 * k), size: ss, color: softInk)
        }
        labels.append(Label(text: title, box: box, view: view))
    }
    func scaleBar(_ label: String, pixels: CGFloat, x: CGFloat, y: CGFloat, backed: Bool, view: Int) {
        let lw: CGFloat = textWidth(label, size: 14 * k, bold: true)
        let box = CGRect(x: x - 8 * k, y: y - 26 * k, width: max(pixels, lw) + 16 * k, height: 38 * k)
        if backed {
            ctx.setFillColor(paper)
            ctx.fill(CGRect(x: box.minX, y: h - box.maxY, width: box.width, height: box.height))
        }
        ctx.setFillColor(ink)
        ctx.fill(CGRect(x: x, y: h - y, width: pixels, height: 3 * k))
        ctx.fill(CGRect(x: x, y: h - y - 3 * k, width: 2 * k, height: 9 * k))
        ctx.fill(CGRect(x: x + pixels - 2 * k, y: h - y - 3 * k, width: 2 * k, height: 9 * k))
        put(label, x: x + pixels / 2 - lw / 2, top: y - 22 * k, size: 14 * k, bold: true)
        labels.append(Label(text: label, box: box, view: view))
    }

    let insetC = CGPoint(x: CGFloat(insetCentre.x) * h, y: CGFloat(insetCentre.y) * h)
    let insetR: CGFloat = CGFloat(insetRadius) * h
    let cutC = CGPoint(x: CGFloat(cutCentre.x) * h, y: CGFloat(cutCentre.y) * h)
    let cutR: CGFloat = CGFloat(cutRadius) * h
    let molC = CGPoint(x: CGFloat(moleculeCentre.x) * h, y: CGFloat(moleculeCentre.y) * h)
    let molR: CGFloat = CGFloat(moleculeRadius) * h
    let cam: OrthoCamera = cutCamera(scene.yarn)

    // --- callouts: main → yarn inset → cut inset → molecule
    let hairIndex: Int = calloutHair(scene, width: w, height: hI)
    let hair: Hair = scene.hairs[hairIndex]
    let root: CGPoint = pt(projectMain(hair.ribbon.points[0], width: w, height: hI))
    let spotR: CGFloat = 30 * k
    ctx.setStrokeColor(CGColor(srgbRed: 1, green: 1, blue: 1, alpha: 0.75))
    ctx.setLineWidth(1.5 * k)
    for (a, b) in tangents(root, spotR, insetC, insetR) { line(a, b) }
    ctx.setStrokeColor(white)
    ctx.setLineWidth(2.2 * k)
    circle(root, spotR)

    let fuzz: Ribbon = scene.yarn.fuzz
    let endW: SIMD3<Float> = fuzz.points[fuzz.points.count - 1]
    let endPt: CGPoint = pt(projectInset(endW, height: hI))
    let endR: CGFloat = 22 * k
    ctx.setStrokeColor(softInk)
    ctx.setLineWidth(1.5 * k)
    for (a, b) in tangents(endPt, endR, cutC, cutR) { lineOutside(a, b, circle: insetC, radius: insetR + 3.5 * k) }
    ctx.setStrokeColor(ink)
    ctx.setLineWidth(2 * k)
    circle(endPt, endR)

    // From the cut face's wall to the molecule: cellulose is what the wall is made of.
    let wallW: SIMD3<Float> = endW + fuzz.widths[fuzz.widths.count - 1] * 6
    let wallPt: CGPoint = pt(projectCut(wallW, camera: cam, height: hI))
    let wallR: CGFloat = 16 * k
    ctx.setStrokeColor(softInk)
    ctx.setLineWidth(1.5 * k)
    for (a, b) in tangents(wallPt, wallR, molC, molR) { lineOutside(a, b, circle: cutC, radius: cutR + 3.5 * k) }
    ctx.setStrokeColor(ink)
    ctx.setLineWidth(2 * k)
    circle(wallPt, wallR)

    // Rings round the insets: white inside a thin dark line.
    for (c, r) in [(insetC, insetR), (cutC, cutR), (molC, molR)] {
        ctx.setStrokeColor(white)
        ctx.setLineWidth(6 * k)
        circle(c, r)
        ctx.setStrokeColor(ink)
        ctx.setLineWidth(1.5 * k)
        circle(c, r + 3.5 * k)
    }

    // --- scale bars, each from its own view's pixel size
    let mmPx: CGFloat = CGFloat(mainBarMillimetres / mainMillimetresPerPixel(width: w))
    scaleBar("1 mm", pixels: mmPx, x: 56 * k, y: h - 50 * k, backed: true, view: 1)
    let umPx: CGFloat = CGFloat(insetBarMicrometres / insetMicrometresPerPixel(height: hI))
    scaleBar("100 µm", pixels: umPx, x: insetC.x - umPx / 2, y: insetC.y + insetR - 30 * k, backed: true, view: 2)
    let cutPx: CGFloat = CGFloat(cutBarMicrometres / cutMicrometresPerPixel(height: hI))
    scaleBar("10 µm", pixels: cutPx, x: cutC.x - cutPx / 2, y: cutC.y + cutR - 22 * k, backed: true, view: 3)
    let nmPx: CGFloat = CGFloat(moleculeBarNanometres * 10 / moleculeAngstromsPerPixel(height: hI))
    scaleBar(String(format: "%g nm", moleculeBarNanometres), pixels: nmPx, x: molC.x - nmPx / 2, y: molC.y + molR - 20 * k,
             backed: true, view: 4)

    // --- main view labels
    let leg: CGPoint = pt(projectMain(scene.knit.world(scene.knit.point(-5, 3, Float.pi / 2)), width: w, height: hI))
    tag("knit loops", "plain jersey: each row of Vs is one yarn,\npulled through the row below",
        anchor: leg, x0: leg.x - 120 * k, y0: leg.y + 70 * k, view: 1)
    // A long hair near the left for the fuzz label.
    var fuzzTip: CGPoint = root
    var bestScore: CGFloat = .infinity
    let wantTip = CGPoint(x: 0.38 * h, y: 0.36 * h)
    for (i, hr) in scene.hairs.enumerated() where i != hairIndex {
        let tip: CGPoint = pt(projectMain(hr.ribbon.points[hr.ribbon.points.count - 1], width: w, height: hI))
        let score: CGFloat = hypot(tip.x - wantTip.x, tip.y - wantTip.y) - CGFloat(hr.ribbon.length) * 60 * k
        if score < bestScore { bestScore = score; fuzzTip = tip }
    }
    tag("fuzz", "loose fibre ends standing up\nfrom the yarn", anchor: fuzzTip, x0: fuzzTip.x - 40 * k, y0: fuzzTip.y - 92 * k, view: 1)

    // --- yarn inset labels
    let revW: SIMD3<Float> = ribbonPoint(fuzz, at: fuzz.reversals.first ?? (scene.yarn.peelArc + fuzzReversalAfter))
    let rev: CGPoint = pt(projectInset(revW, height: hI))
    tag("reversal", "the helix of cellulose in the wall\nchanges hand here, as step 46's\ntendrils do at a perversion",
        anchor: rev, x0: insetC.x - insetR * 0.86, y0: insetC.y - insetR * 0.62, view: 2)
    let convW: SIMD3<Float> = ribbonPoint(fuzz, at: scene.yarn.peelArc + fuzzReversalAfter + 90)
    let conv: CGPoint = pt(projectInset(convW, height: hI))
    tag("convolutions", "the dried ribbon twists,\n~6 half-turns per mm",
        anchor: conv, x0: insetC.x + insetR * 0.18, y0: insetC.y - insetR * 0.92, view: 2)
    let oneW: SIMD3<Float> = ribbonPoint(fuzz, at: scene.yarn.peelArc + 60)
    let one: CGPoint = pt(projectInset(oneW, height: hI))
    tag("one fibre = one cell", "a seed hair, tinted to follow it",
        anchor: one, x0: insetC.x - insetR * 0.92, y0: insetC.y + insetR * 0.02, view: 2)
    let yarnW: SIMD3<Float> = helixPoint(x: 150, radius: yarnRadius, phase: 2.0)
    let yarnPt: CGPoint = pt(projectInset(yarnW, height: hI))
    tag("yarn", String(format: "~%.0f fibres, twisted %.1f turns/cm (Z)", fibresPerYarnSection, yarnTwistPerMillimetre * 10),
        anchor: yarnPt, x0: insetC.x - insetR * 0.30, y0: insetC.y + insetR * 0.50, view: 2)

    // --- cut inset labels (outside the circle, left of it)
    func centred(_ text: String, _ c: CGPoint, top: CGFloat, size: CGFloat, bold: Bool = false, color: CGColor = ink, view: Int) {
        let tw: CGFloat = textWidth(text, size: size, bold: bold)
        put(text, x: c.x - tw / 2, top: top, size: size, bold: bold, color: color)
        let box = CGRect(x: c.x - tw / 2, y: top, width: tw, height: size * 1.2)
        labels.append(Label(text: text, box: box, view: view))
    }
    let lastStep: SIMD3<Float> = fuzz.points[fuzz.points.count - 1] - fuzz.points[fuzz.points.count - 2]
    let lumenW: SIMD3<Float> = endW - lastStep * 0.01
    let lumenPt: CGPoint = pt(projectCut(lumenW, camera: cam, height: hI))
    tag("cut across", "kidney-shaped wall; the lumen,\ndried to a slit",
        anchor: lumenPt, x0: cutC.x - cutR - 250 * k, y0: cutC.y - cutR + 6 * k, view: 3, size: 14.5)

    // --- molecule labels (inside the top of the circle)
    centred("cellulose", molC, top: molC.y - molR + 18 * k, size: 15 * k, bold: true, view: 4)
    centred("glucose, joined β(1→4); each unit", molC, top: molC.y - molR + 37 * k, size: 11.5 * k, color: softInk, view: 4)
    centred("turned 180° from the next", molC, top: molC.y - molR + 52 * k, size: 11.5 * k, color: softInk, view: 4)

    // --- title and caption, on a pale plate so the knit behind stays out of the way
    let capLines: Int = 5
    let plateW: CGFloat = 760 * k
    ctx.setFillColor(CGColor(srgbRed: 1, green: 1, blue: 1, alpha: 0.84))
    let capHeight: CGFloat = 21 * k * CGFloat(capLines)
    let plateH: CGFloat = 58 * k + capHeight + 12 * k
    let plateBottom: CGFloat = 40 * k + 58 * k + capHeight
    ctx.fill(CGRect(x: 40 * k, y: h - plateBottom, width: plateW, height: plateH))
    put("A cotton sock, close up", x: 56 * k, top: 40 * k, size: 28 * k)
    let cap: [String] = [
        "Knit loops of yarn; the yarn is fibres twisted together; each fibre is a single plant cell,",
        String(format: "a cotton seed hair up to %.0f cm long whose dead wall is ~%.0f%% cellulose.", maxFibreLengthCentimetres, cellulosePercent),
        "Cellulose is a chain of glucose, the sugar of steps 7, 17 and 18. Its β links are why we",
        "can't digest cotton: starch joins the same glucose α(1→4), the link turned the other way.",
        "Drawn as 100% cotton. Most everyday socks blend cotton with nylon and spandex for fit and wear.",
    ]
    for (i, c) in cap.enumerated() { put(c, x: 56 * k, top: 80 * k + CGFloat(i) * 21 * k, size: 15 * k, color: softInk) }
    let capW: CGFloat = cap.map { textWidth($0, size: 15 * k, bold: false) }.max() ?? 0
    labels.append(Label(text: "caption", box: CGRect(x: 56 * k, y: 40 * k, width: capW, height: 40 * k + 21 * k * CGFloat(cap.count) + 10 * k), view: 0))
    return labels
}
