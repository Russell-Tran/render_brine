// What the CPU draws over the GPU's picture, straight into the same memory:
// the panel titles, the anatomy labels, the break marks on the shortened
// colon, an arrow on every water symbol crossing the wall (so the direction of
// each crossing reads at a glance and the two panels' traffic can be counted),
// the close-up's captions, and the key.
//
// Text is Core Text into a CGContext over the frame buffer, as in steps 13
// and 19.

import CoreGraphics
import CoreText
import Foundation
import Metal
import simd

private let ink = CGColor(srgbRed: 0.93, green: 0.95, blue: 0.96, alpha: 1)
private let muted = CGColor(srgbRed: 0.62, green: 0.69, blue: 0.73, alpha: 1)
private let accent = CGColor(srgbRed: 0.50, green: 0.84, blue: 0.88, alpha: 1)
private let warn = CGColor(srgbRed: 0.98, green: 0.62, blue: 0.45, alpha: 1)
private let leader = CGColor(srgbRed: 0.85, green: 0.89, blue: 0.92, alpha: 0.75)
private let plate = CGColor(srgbRed: 0.03, green: 0.05, blue: 0.07, alpha: 0.72)

private func cgColour(_ c: SIMD4<Float>, alpha: Float = 1) -> CGColor {
    let v: SIMD4<Double> = SIMD4<Double>(c)
    return CGColor(srgbRed: v.x, green: v.y, blue: v.z, alpha: Double(alpha))
}

private func ctLine(_ s: String, size: CGFloat, bold: Bool, color: CGColor) -> CTLine {
    let font = CTFontCreateWithName((bold ? "HelveticaNeue-Bold" : "HelveticaNeue") as CFString, size, nil)
    let attributes: [NSAttributedString.Key: Any] = [
        NSAttributedString.Key(kCTFontAttributeName as String): font,
        NSAttributedString.Key(kCTForegroundColorAttributeName as String): color,
    ]
    return CTLineCreateWithAttributedString(NSAttributedString(string: s, attributes: attributes))
}

func textWidth(_ s: String, size: CGFloat, bold: Bool = false) -> CGFloat {
    CGFloat(CTLineGetTypographicBounds(ctLine(s, size: size, bold: bold, color: ink), nil, nil, nil))
}

/// Everything below is in top-down pixels; this does the flip.
private struct Canvas {
    let ctx: CGContext
    let height: CGFloat
    let k: CGFloat

    func text(_ s: String, x: CGFloat, top: CGFloat, size: CGFloat, bold: Bool = false,
              color: CGColor = ink, centred: Bool = false, backed: Bool = false) {
        let w: CGFloat = textWidth(s, size: size * k, bold: bold)
        let x0: CGFloat = centred ? x - w / 2 : x
        if backed {
            ctx.setFillColor(plate)
            let em: CGFloat = size * k
            let y0: CGFloat = height - top - em * 1.25 - k
            let r = CGRect(x: x0 - 4 * k, y: y0, width: w + 8 * k, height: em * 1.45)
            ctx.addPath(CGPath(roundedRect: r, cornerWidth: 3 * k, cornerHeight: 3 * k, transform: nil))
            ctx.fillPath()
        }
        ctx.textPosition = CGPoint(x: x0, y: height - top - size * k)
        CTLineDraw(ctLine(s, size: size * k, bold: bold, color: color), ctx)
    }

    func line(_ a: CGPoint, _ b: CGPoint, color: CGColor = leader, width: CGFloat = 1) {
        ctx.setStrokeColor(color)
        ctx.setLineWidth(width * k)
        ctx.move(to: CGPoint(x: a.x, y: height - a.y))
        ctx.addLine(to: CGPoint(x: b.x, y: height - b.y))
        ctx.strokePath()
    }

    func arrow(_ a: CGPoint, _ b: CGPoint, color: CGColor, width: CGFloat = 1.4, head: CGFloat = 4) {
        line(a, b, color: color, width: width)
        let d = CGPoint(x: b.x - a.x, y: b.y - a.y)
        let len: CGFloat = max(sqrt(d.x * d.x + d.y * d.y), 1e-3)
        let u = CGPoint(x: d.x / len, y: d.y / len)
        let h: CGFloat = head * k
        let p1 = CGPoint(x: b.x - u.x * h - u.y * h * 0.6, y: b.y - u.y * h + u.x * h * 0.6)
        let p2 = CGPoint(x: b.x - u.x * h + u.y * h * 0.6, y: b.y - u.y * h - u.x * h * 0.6)
        ctx.setFillColor(color)
        ctx.move(to: CGPoint(x: b.x, y: height - b.y))
        ctx.addLine(to: CGPoint(x: p1.x, y: height - p1.y))
        ctx.addLine(to: CGPoint(x: p2.x, y: height - p2.y))
        ctx.closePath()
        ctx.fillPath()
    }

    func disc(_ c: CGPoint, radius: CGFloat, fill: CGColor, stroke: CGColor? = nil) {
        let r = CGRect(x: c.x - radius, y: height - c.y - radius, width: 2 * radius, height: 2 * radius)
        ctx.setFillColor(fill)
        ctx.fillEllipse(in: r)
        if let s = stroke {
            ctx.setStrokeColor(s)
            ctx.setLineWidth(1 * k)
            ctx.strokeEllipse(in: r)
        }
    }
}

private func canvas(_ frame: Frame) -> Canvas? {
    let l: Layout = frame.layout
    guard let ctx = CGContext(data: frame.pixels.contents(), width: l.width, height: l.height,
                              bitsPerComponent: 8, bytesPerRow: l.width * 4,
                              space: CGColorSpace(name: CGColorSpace.sRGB)!,
                              bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
    ctx.setShouldAntialias(true)
    return Canvas(ctx: ctx, height: CGFloat(l.height), k: CGFloat(l.k))
}

private func pt(_ p: SIMD3<Float>, _ panel: Int, _ l: Layout) -> CGPoint {
    let s: SIMD2<Float> = project(p, panel: panel, layout: l)
    return CGPoint(x: CGFloat(s.x), y: CGFloat(s.y))
}

/// Where the ring of contraction is in a panel at time t, if it is on a part
/// of the gut where the wave runs; and the relaxed segment ahead of it.
func ringPositions(panel: Int, t: Float) -> [Float] {
    let w: Wave = wave(panel)
    var out: [Float] = []
    // waveOffset(x) = 0 at a ring; step through every ring on screen.
    var x: Float = gutStartX - waveOffset(gutStartX, panel: panel, t: t)
    while x < gutEndX {
        let active: Bool = (x < -36 || x > 20) && (x < breakStartX - 6 || x > breakEndX + 4) && x > gutStartX + 8
        if active { out.append(x) }
        x += w.wavelength
    }
    return out
}

// MARK: - the whole overlay

func drawOverlay(into frame: Frame, time t: Float, mutant: Mutant = .none) {
    guard let c = canvas(frame) else { return }
    let l: Layout = frame.layout
    let k: CGFloat = CGFloat(l.k)
    let ph: CGFloat = CGFloat(l.panelHeight)

    // Panel separators and titles.
    c.line(CGPoint(x: 0, y: ph), CGPoint(x: CGFloat(l.width), y: ph),
           color: CGColor(srgbRed: 0.25, green: 0.30, blue: 0.34, alpha: 1), width: 1)
    c.text("normal", x: 16 * k, top: 12 * k, size: 22, bold: true)
    c.text("secretory diarrhea (cholera-type)", x: 16 * k, top: ph + 12 * k, size: 22, bold: true, color: warn)
    c.text("the same stretch of gut, cut open lengthwise · particles are oversized symbols, not to scale",
           x: 16 * k + textWidth("normal", size: 22 * k, bold: true) + 14 * k, top: 20 * k, size: 11.5, color: muted)

    // Break marks: the colon is shortened between breakStartX and breakEndX.
    for panel in 0..<2 {
        for edge: Float in [breakStartX + 1.2, breakEndX - 1.2] {
            var pts: [CGPoint] = []
            var y: Float = colonAxisY - colonRadius - colonWall - 4
            var flip: Float = -1
            while y <= colonAxisY + colonRadius + colonWall + 4 {
                pts.append(pt(SIMD3(edge + flip * 1.0, y, 0), panel, l))
                y += 3.0
                flip = -flip
            }
            for i in 1..<pts.count { c.line(pts[i - 1], pts[i], color: ink, width: 1.6) }
        }
    }
    for panel in 0..<2 {
        let mid: CGPoint = pt(SIMD3((breakStartX + breakEndX) / 2, colonAxisY, 0), panel, l)
        c.text("colon", x: mid.x, top: mid.y - 14 * k, size: 10.5, color: muted, centred: true)
        c.text("shortened", x: mid.x, top: mid.y, size: 10.5, color: muted, centred: true)
    }

    // Anatomy, upper panel: the ileum's labels sit under it, where there is room.
    let under: Float = ileumAxisY - ileumRadius - ileumWall - 1
    let ileLabel: CGPoint = pt(SIMD3(-80, under, 0), 0, l)
    c.text("terminal ileum: villi, with crypts between them", x: ileLabel.x, top: ileLabel.y + 6 * k, size: 12.5, centred: true)
    let valve: CGPoint = pt(SIMD3(lipX - 0.5, ileumAxisY - ileumRadius - ileumWall * 0.5, 0), 0, l)
    let valveLabel = CGPoint(x: ileLabel.x + 30 * k, y: ileLabel.y + 44 * k)
    c.line(CGPoint(x: valveLabel.x + 200 * k, y: valveLabel.y - 2 * k), valve)
    c.text("ileocecal valve: the villi stop here", x: valveLabel.x, top: valveLabel.y - 8 * k, size: 12.5)
    let cecum: CGPoint = pt(SIMD3(-19, colonAxisY - colonRadius + 6, -1), 0, l)
    let cecumLabel: CGPoint = pt(SIMD3(-38, colonAxisY - colonRadius - colonWall - 2, 0), 0, l)
    c.line(CGPoint(x: cecumLabel.x + 4 * k, y: cecumLabel.y + 4 * k), cecum)
    c.text("cecum", x: cecumLabel.x - 40 * k, top: cecumLabel.y, size: 12.5)
    let colonLabel: CGPoint = pt(SIMD3(52, colonAxisY + colonRadius + colonWall + 2, 0), 0, l)
    c.text("colon: flat lining with crypts, no villi", x: colonLabel.x, top: colonLabel.y - 17 * k, size: 12.5, centred: true)
    let rectum: CGPoint = pt(SIMD3(126, colonAxisY + colonRadius + colonWall + 2, 0), 0, l)
    c.text("rectum →", x: rectum.x, top: rectum.y - 17 * k, size: 12.5, centred: true)

    // Contents, upper panel.
    let chyme: CGPoint = pt(SIMD3(12, colonAxisY - 9, -2), 0, l)
    c.text("watery chyme", x: chyme.x, top: chyme.y, size: 11.5, centred: true, backed: true)
    let firming: CGPoint = pt(SIMD3(52, colonAxisY - colonRadius + 2.5, -2), 0, l)
    c.text("water absorbed, stool firming", x: firming.x, top: firming.y, size: 11.5, centred: true, backed: true)
    let formed: CGPoint = pt(SIMD3(125, colonAxisY - colonRadius + 2.5, -2), 0, l)
    c.text("formed stool, ~75% water", x: formed.x, top: formed.y, size: 11.5, centred: true, backed: true)

    // Lower panel labels.
    let crypts: CGPoint = pt(SIMD3(-70, ileumAxisY - ileumRadius - ileumWall - 1, 0), 1, l)
    c.text("crypts pour out Cl⁻, then Na⁺, then water", x: crypts.x, top: crypts.y + 4 * k, size: 12.5, color: warn, centred: true)
    c.text("villus tips still absorb (why oral rehydration works)", x: crypts.x, top: crypts.y + 20 * k, size: 11.5, color: muted, centred: true)
    let colonB: CGPoint = pt(SIMD3(48, colonAxisY + colonRadius + colonWall + 2, 0), 1, l)
    c.text("colon still absorbing at its ~5 L/day maximum — overwhelmed, not idle", x: colonB.x, top: colonB.y - 22 * k, size: 12.5, centred: true)
    let rice: CGPoint = pt(SIMD3(126, colonAxisY - colonRadius - colonWall - 4, 0), 1, l)
    c.text("rice-water stool", x: rice.x, top: rice.y + 4 * k, size: 12.5, color: warn, centred: true)
    let fast: CGPoint = pt(SIMD3(48, colonAxisY - colonRadius - colonWall - 1, 0), 1, l)
    c.text("thin, fast, and far more than the colon can take back (>1 L/hour in severe cholera)",
           x: fast.x, top: fast.y + 9 * k, size: 11.5, color: muted, centred: true)

    // The wave, labelled where it is, in the upper panel: above the ileum,
    // below the colon, wherever there is room.
    for x in ringPositions(panel: 0, t: t) {
        let inIleum: Bool = x < lipX
        let ahead: Float = x + relaxationLead
        let base: CGFloat
        let ring: CGPoint
        let front: CGPoint
        if inIleum {
            ring = pt(SIMD3(x, ileumAxisY + ileumRadius + ileumWall + 1, 0), 0, l)
            front = pt(SIMD3(ahead, ileumAxisY + ileumRadius + ileumWall + 1, 0), 0, l)
            base = ring.y - 26 * k
        } else {
            ring = pt(SIMD3(x, colonAxisY - colonRadius - colonWall - 0.5, 0), 0, l)
            front = pt(SIMD3(ahead, colonAxisY - colonRadius - colonWall - 0.5, 0), 0, l)
            base = ring.y + 22 * k
        }
        let right: CGFloat = front.x + 14 * k + textWidth("relaxes ahead", size: 11 * k) / 2
        guard right < CGFloat(l.width) - 4 * k else { continue }
        c.text("squeeze", x: ring.x, top: base, size: 11, color: accent, centred: true)
        c.text("relaxes ahead", x: front.x + 14 * k, top: base, size: 11, color: accent, centred: true)
        c.arrow(CGPoint(x: ring.x + 22 * k, y: base + 7 * k), CGPoint(x: front.x - 26 * k, y: base + 7 * k),
                color: accent, width: 1.2)
    }

    // An arrow on every water symbol crossing the wall: its last stretch of path.
    for panel in 0..<2 {
        for e in emitters(panel: panel, mutant: mutant) where e.species == .water {
            for kf in 0..<e.perLoop {
                let age: Float = fract(t / loopSeconds - e.phase - Float(kf) / Float(e.perLoop))
                let norm: Float = age / e.life
                // Only while it crosses the wall: that is the traffic being counted.
                let w = arrowWindow(e.route)
                guard norm > w.from, norm < w.to else { continue }
                let a: Float = smoothstep(w.from, w.from + 0.05, norm) * (1 - smoothstep(w.to - 0.06, w.to, norm))
                guard a > 0.1 else { continue }
                let p1: CGPoint = pt(wallPath(e, age: norm, t: t, mutant: mutant), panel, l)
                let p0: CGPoint = pt(wallPath(e, age: max(norm - 0.14, 0), t: t, mutant: mutant), panel, l)
                let dx: CGFloat = p1.x - p0.x
                let dy: CGFloat = p1.y - p0.y
                let len: CGFloat = sqrt(dx * dx + dy * dy)
                guard len > 0.5 else { continue }
                // A fixed-length arrow ending at the symbol, along its motion.
                let tail = CGPoint(x: p1.x - dx / len * 15 * k, y: p1.y - dy / len * 15 * k)
                let colour = CGColor(srgbRed: 0.66, green: 0.87, blue: 1.0, alpha: CGFloat(a) * 0.95)
                c.arrow(tail, p1, color: colour, width: 1.3, head: 4.2)
            }
        }
    }

    drawStrip(c, l, t: t)
}

// MARK: - the strip: close-up captions and the key

private func drawStrip(_ c: Canvas, _ l: Layout, t: Float) {
    let k: CGFloat = CGFloat(l.k)
    let ix: CGFloat = CGFloat(l.insetX)
    let iy: CGFloat = CGFloat(l.insetY)
    let iw: CGFloat = CGFloat(l.insetW)
    let ih: CGFloat = CGFloat(l.insetH)

    // Close-up frame and its labels.
    c.ctx.setStrokeColor(CGColor(srgbRed: 0.35, green: 0.40, blue: 0.44, alpha: 1))
    c.ctx.setLineWidth(1 * k)
    c.ctx.stroke(CGRect(x: ix, y: c.height - iy - ih, width: iw, height: ih))
    c.text("close-up: the wall of one crypt, in cholera", x: ix + 8 * k, top: iy + 5 * k, size: 12, bold: true, backed: true)
    c.text("crypt lumen", x: ix + iw - 8 * k - textWidth("crypt lumen", size: 11 * k), top: iy + 6 * k, size: 11, color: accent)
    c.text("tissue / blood side", x: ix + 8 * k, top: iy + ih - 18 * k, size: 11, color: muted, backed: true)

    // CFTR and the tight junction, pointed out on the rightmost cell and gap.
    let cell: Int = insetCellCount - 1
    let cftr: CGPoint = pt(SIMD3(insetCellX(cell), insetTop - 0.6, 0.1), 2, l)
    c.text("CFTR", x: cftr.x + 10 * k, top: cftr.y - 22 * k, size: 11, bold: true,
           color: CGColor(srgbRed: 0.30, green: 0.90, blue: 0.75, alpha: 1), backed: true)
    c.line(CGPoint(x: cftr.x + 12 * k, y: cftr.y - 9 * k), cftr)
    let tj: CGPoint = pt(SIMD3(insetGapX(cell - 1), insetTop - 1.3, 0.1), 2, l)
    c.text("tight junction", x: tj.x - 90 * k, top: tj.y - 26 * k, size: 11, backed: true)
    c.line(CGPoint(x: tj.x - 40 * k, y: tj.y - 12 * k), tj)
    let nk: CGPoint = pt(SIMD3(insetCellX(cell), insetBottom + 0.5, 0.1), 2, l)
    c.text("NKCC1", x: nk.x + 12 * k, top: nk.y + 6 * k, size: 11, bold: true,
           color: CGColor(srgbRed: 0.55, green: 0.68, blue: 1.0, alpha: 1), backed: true)
    c.line(CGPoint(x: nk.x + 12 * k, y: nk.y + 10 * k), nk)

    // The lumen goes negative when chloride arrives, until sodium follows.
    let charge: Float = insetLumenCharge(at: t)
    if charge > 0.02 {
        let pos: CGPoint = pt(SIMD3(-4, insetTop + 3.4, 0.5), 2, l)
        let col = CGColor(srgbRed: 0.55, green: 0.95, blue: 0.60, alpha: CGFloat(charge))
        c.text("−   −   −", x: pos.x, top: pos.y - 14 * k, size: 20, bold: true, color: col, centred: true)
        c.text("lumen negative", x: pos.x + 90 * k, top: pos.y - 9 * k, size: 11, color: col)
    }

    // The steps, beside the close-up, the current one lit.
    let x0: CGFloat = ix + iw + 22 * k
    var y: CGFloat = iy + 2 * k
    let now: Int = insetStep(at: t)
    let steps: [String] = [
        "Cl⁻ leaves THROUGH the cells: toxin → cAMP → CFTR open",
        "lumen goes negative; Na⁺ follows BETWEEN the cells",
        "water follows both, by osmosis",
    ]
    for (i, s) in steps.enumerated() {
        let n: String = "\(i + 1)"
        let lit: Bool = i == now
        let col: CGColor = lit ? ink : muted
        c.text(n, x: x0, top: y, size: 13, bold: true, color: lit ? warn : muted)
        c.text(s, x: x0 + 16 * k, top: y, size: 13, bold: lit, color: col)
        y += 19 * k
    }

    // The key: every species at its drawn size, in proportion to its radius.
    y += 8 * k
    var x: CGFloat = x0
    let keys: [(Species, String)] = [(.chloride, "Cl⁻"), (.sodium, "Na⁺"), (.potassium, "K⁺ (sparse)"),
                                     (.bicarbonate, "HCO₃⁻"), (.water, "water"), (.fleck, "mucus")]
    for (s, name) in keys {
        let r: CGFloat = CGFloat(radiusPm(s) * panelMMPerPm / l.scale) * 1.35
        let colour: SIMD4<Float> = speciesColour(s)
        c.disc(CGPoint(x: x + r, y: y + 8 * k), radius: r, fill: cgColour(colour, alpha: max(colour.w, 0.8)),
               stroke: CGColor(srgbRed: 0, green: 0, blue: 0, alpha: 0.6))
        c.text(name, x: x + 2 * r + 4 * k, top: y + 1 * k, size: 12)
        x += 2 * r + 12 * k + textWidth(name, size: 12 * k)
    }
    y += 22 * k
    let s: StoolElectrolytes = choleraStool
    c.text("cholera stool (mmol/L): Na⁺ \(s.sodium) · Cl⁻ \(s.chloride) · K⁺ \(s.potassium) · HCO₃⁻ \(s.bicarbonate) — the lower lumen holds that many of each",
           x: x0, top: y, size: 11, color: muted)
    y += 15 * k
    let gap: Int = Int(osmoticGap(sodium: Float(s.sodium), potassium: Float(s.potassium)))
    c.text("osmotic gap 290 − 2(\(s.sodium) + \(s.potassium)) = \(gap) mOsm/kg: under 50, so secretory, not osmotic",
           x: x0, top: y, size: 11, color: muted)
    y += 15 * k
    let n: (into: Int, out: Int, colonOut: Int) = waterBudget(panel: 0)
    let d: (into: Int, out: Int, colonOut: Int) = waterBudget(panel: 1)
    c.text("water arrows per loop — normal: \(n.into) in, \(n.out) back · cholera: \(d.into) in, \(d.out) back (\(d.colonOut) by the colon)",
           x: x0, top: y, size: 11, color: muted)
    y += 15 * k
    c.text("colon's ceiling ~5 L/day (Debongnie & Phillips 1978); severe cholera can exceed 1 L/hour",
           x: x0, top: y, size: 11, color: muted)
    y += 15 * k
    c.text("not to scale: a water molecule is ~a millionth of a villus; mucosa drawn ~2–7× enlarged",
           x: x0, top: y, size: 11, color: muted)
}
