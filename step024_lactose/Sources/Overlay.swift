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
    c.text("lactase working", x: 16 * k, top: 12 * k, size: 22, bold: true)
    c.text("lactase missing: osmotic diarrhea", x: 16 * k, top: ph + 12 * k, size: 22, bold: true, color: warn)
    let grams: Int = Int(lactoseGramsPerCup) * cups
    c.text("both after \(cups) glasses of milk (~\(grams) g lactose) · symbols oversized, not to scale",
           x: 16 * k + textWidth("lactase working", size: 22 * k, bold: true) + 14 * k, top: 20 * k, size: 11.5, color: muted)

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
    let gold = CGColor(srgbRed: 0.98, green: 0.74, blue: 0.30, alpha: 1)
    let under: Float = ileumAxisY - ileumRadius - ileumWall - 1
    let ileLabel: CGPoint = pt(SIMD3(-64, under, 0), 0, l)
    c.text("small intestine: lactase (gold) on the villus tips splits lactose", x: ileLabel.x, top: ileLabel.y + 4 * k, size: 12.5, color: gold, centred: true)
    c.text("glucose + galactose go in through SGLT1", x: ileLabel.x, top: ileLabel.y + 20 * k, size: 11.5, color: muted, centred: true)
    let valve: CGPoint = pt(SIMD3(lipX - 0.5, ileumAxisY - ileumRadius - ileumWall * 0.5, 0), 0, l)
    let valveLabel = CGPoint(x: ileLabel.x + 10 * k, y: ileLabel.y + 50 * k)
    let valveText: String = "ileocecal valve: the villi stop here"
    c.line(CGPoint(x: valveLabel.x + textWidth(valveText, size: 12.5 * k) + 4 * k, y: valveLabel.y - 2 * k), valve)
    c.text(valveText, x: valveLabel.x, top: valveLabel.y - 8 * k, size: 12.5)
    let cecum: CGPoint = pt(SIMD3(-19, colonAxisY - colonRadius + 6, -1), 0, l)
    let cecumLabel: CGPoint = pt(SIMD3(-38, colonAxisY - colonRadius - colonWall - 2, 0), 0, l)
    c.line(CGPoint(x: cecumLabel.x + 4 * k, y: cecumLabel.y + 4 * k), cecum)
    c.text("cecum", x: cecumLabel.x - 40 * k, top: cecumLabel.y, size: 12.5)
    let colonLabel: CGPoint = pt(SIMD3(60, colonAxisY + colonRadius + colonWall + 2, 0), 0, l)
    c.text("colon: bacteria here too — no lactose reaches them", x: colonLabel.x, top: colonLabel.y - 17 * k, size: 12.5, centred: true)
    let rectum: CGPoint = pt(SIMD3(126, colonAxisY + colonRadius + colonWall + 2, 0), 0, l)
    c.text("rectum →", x: rectum.x, top: rectum.y - 17 * k, size: 12.5, centred: true)

    // Contents, upper panel.
    let chyme: CGPoint = pt(SIMD3(12, colonAxisY - 9, -2), 0, l)
    c.text("watery chyme, no lactose left", x: chyme.x, top: chyme.y, size: 11.5, centred: true, backed: true)
    let firming: CGPoint = pt(SIMD3(52, colonAxisY - colonRadius + 2.5, -2), 0, l)
    c.text("water absorbed, stool firming", x: firming.x, top: firming.y, size: 11.5, centred: true, backed: true)
    // Formed stool is ~75% water (median 74.6%: Rose, Parker, Jefferson &
    // Cartmell, Crit Rev Environ Sci Technol 45:1827, 2015) — as in step 23.
    let formed: CGPoint = pt(SIMD3(125, colonAxisY - colonRadius + 2.5, -2), 0, l)
    c.text("formed stool, ~75% water", x: formed.x, top: formed.y, size: 11.5, centred: true, backed: true)

    // Lower panel labels.
    let passes: CGPoint = pt(SIMD3(-70, ileumAxisY - ileumRadius - ileumWall - 1, 0), 1, l)
    c.text("no lactase: lactose passes the villi untouched and pulls water IN", x: passes.x, top: passes.y + 4 * k, size: 12.5, color: warn, centred: true)
    c.text("osmosis alone — no chloride secretion, no ion leading the way", x: passes.x, top: passes.y + 20 * k, size: 11.5, color: muted, centred: true)
    let colonB: CGPoint = pt(SIMD3(56, colonAxisY + colonRadius + colonWall + 2, 0), 1, l)
    c.text("bacteria ferment the lactose → gas (H₂, CO₂) + short-chain fatty acids", x: colonB.x, top: colonB.y - 22 * k, size: 12.5, centred: true)
    let salvage: CGPoint = pt(SIMD3(44, colonAxisY - colonRadius - colonWall - 1, 0), 1, l)
    c.text("the colon takes some SCFAs back, and water with them — not enough",
           x: salvage.x, top: salvage.y + 9 * k, size: 11.5, color: muted, centred: true)
    let loose: CGPoint = pt(SIMD3(126, colonAxisY - colonRadius - colonWall - 4, 0), 1, l)
    c.text("loose, bubbly, acidic stool", x: loose.x, top: loose.y + 2 * k, size: 12.5, color: warn, centred: true)
    c.text(String(format: "pH ~%.1f", lowerStoolPH), x: loose.x, top: loose.y + 17 * k, size: 11.5, color: muted, centred: true)

    // The wave, labelled where it is, in the upper panel.
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
                let tail = CGPoint(x: p1.x - dx / len * 15 * k, y: p1.y - dy / len * 15 * k)
                let colour = CGColor(srgbRed: 0.66, green: 0.87, blue: 1.0, alpha: CGFloat(a) * 0.95)
                c.arrow(tail, p1, color: colour, width: 1.3, head: 4.2)
            }
        }
    }

    drawStrip(c, l, t: t, mutant: mutant)
}

// MARK: - the strip: close-up captions and the key

private func drawStrip(_ c: Canvas, _ l: Layout, t: Float, mutant: Mutant) {
    let k: CGFloat = CGFloat(l.k)
    let ix: CGFloat = CGFloat(l.insetX)
    let iy: CGFloat = CGFloat(l.insetY)
    let iw: CGFloat = CGFloat(l.insetW)
    let ih: CGFloat = CGFloat(l.insetH)
    let gold = CGColor(srgbRed: 0.98, green: 0.74, blue: 0.30, alpha: 1)

    // Close-up frame, its divide, and its labels.
    c.ctx.setStrokeColor(CGColor(srgbRed: 0.35, green: 0.40, blue: 0.44, alpha: 1))
    c.ctx.setLineWidth(1 * k)
    c.ctx.stroke(CGRect(x: ix, y: c.height - iy - ih, width: iw, height: ih))
    let divide: CGPoint = pt(SIMD3(insetGapX(insetLactaseCells - 1), 0, 0.6), 2, l)
    c.line(CGPoint(x: divide.x, y: iy + 22 * k), CGPoint(x: divide.x, y: iy + ih - 4 * k),
           color: CGColor(srgbRed: 0.85, green: 0.89, blue: 0.92, alpha: 0.5), width: 1)
    c.text("close-up: the brush border", x: ix + 8 * k, top: iy + 5 * k, size: 12, bold: true, backed: true)
    c.text("lumen", x: ix + iw - 8 * k - textWidth("lumen", size: 11 * k), top: iy + 6 * k, size: 11, color: accent)
    c.text("tissue / blood side", x: ix + 8 * k, top: iy + ih - 18 * k, size: 11, color: muted, backed: true)
    for (i, caption) in ["lactase (LPH, gold brush border): split,", "each sugar in with 2 Na⁺"].enumerated() {
        c.text(caption, x: divide.x - 8 * k - textWidth(caption, size: 11 * k, bold: true), top: iy + ih - (34 - 16 * CGFloat(i)) * k,
               size: 11, bold: true, color: gold, backed: true)
    }
    c.text("without: lactose stays out,", x: divide.x + 8 * k, top: iy + ih - 34 * k, size: 11, bold: true, color: warn, backed: true)
    c.text("water comes in — no ions first", x: divide.x + 8 * k, top: iy + ih - 18 * k, size: 11, bold: true, color: warn, backed: true)

    // The markers, pointed out on one cell.
    let sglt: CGPoint = pt(SIMD3(insetCellX(insetCellCount - 1), insetTop - 1.0, 0.1), 2, l)
    c.text("SGLT1", x: sglt.x + 12 * k, top: sglt.y - 40 * k, size: 11, bold: true,
           color: CGColor(srgbRed: 0.30, green: 0.90, blue: 0.75, alpha: 1), backed: true)
    c.line(CGPoint(x: sglt.x + 14 * k, y: sglt.y - 24 * k), sglt)
    let glut: CGPoint = pt(SIMD3(insetCellX(insetCellCount - 1), insetBottom + 0.5, 0.1), 2, l)
    c.text("GLUT2", x: glut.x + 12 * k, top: glut.y + 6 * k, size: 11, bold: true,
           color: CGColor(srgbRed: 0.55, green: 0.68, blue: 1.0, alpha: 1), backed: true)
    c.line(CGPoint(x: glut.x + 12 * k, y: glut.y + 10 * k), glut)

    // The key: every symbol at its drawn size.
    let x0: CGFloat = ix + iw + 22 * k
    var y: CGFloat = iy + 2 * k
    func swatch(_ s: Species, at x: CGFloat, radius r: CGFloat) {
        let colour: SIMD4<Float> = speciesColour(s)
        c.disc(CGPoint(x: x + r, y: y + 8 * k), radius: r, fill: cgColour(colour, alpha: max(colour.w, 0.8)),
               stroke: CGColor(srgbRed: 0, green: 0, blue: 0, alpha: 0.6))
    }
    let px: CGFloat = CGFloat(1 / l.scale) * 1.35
    var x: CGFloat = x0
    let rows: [[(Species, String)]] = [
        [(.glucose, "glucose"), (.galactose, "galactose"), (.water, "water"), (.sodium, "Na⁺"), (.potassium, "K⁺")],
        [(.scfa, "SCFA (acetate, propionate, butyrate)"), (.gas, "gas"), (.bacterium, "bacterium")],
    ]
    for (ri, row) in rows.enumerated() {
        x = x0
        if ri == 0 {
            // Lactose: the two sugars, bonded.
            let r: CGFloat = CGFloat(drawnRadius(.glucose)) * px
            swatch(.galactose, at: x, radius: r)
            swatch(.glucose, at: x + r * 1.84, radius: r)
            c.text("lactose", x: x + r * 3.84 + 4 * k, top: y + 1 * k, size: 12)
            x += r * 3.84 + 14 * k + textWidth("lactose", size: 12 * k)
        }
        for (s, name) in row {
            let r: CGFloat = CGFloat(s == .gas ? bubbleMaxMM * 0.7 : drawnRadius(s)) * px
            if s == .bacterium {
                for b in 0..<3 { swatch(s, at: x + CGFloat(b) * r * 1.0, radius: r) }
                x += r * 2
            } else {
                swatch(s, at: x, radius: r)
            }
            c.text(name, x: x + 2 * r + 4 * k, top: y + 1 * k, size: 12)
            x += 2 * r + 12 * k + textWidth(name, size: 12 * k)
        }
        y += 21 * k
    }
    y += 2 * k
    let s: StoolMix = stool(mutant: mutant)
    let na: Int = Int(s.concentration(s.sodium).rounded())
    let kk: Int = Int(s.concentration(s.potassium).rounded())
    let lines: [String] = [
        "~24 g lactose (2 cups): where symptoms become appreciable; ~12 g is usually tolerated (NIH Consensus 2010)",
        "lower stool Na⁺ \(na), K⁺ \(kk) mmol/L → osmotic gap 290 − 2(\(na) + \(kk)) = \(Int(s.gap.rounded())) mOsm/kg: over 125, osmotic",
        "  (Na⁺, K⁺ measured in lactulose diarrhea, Hammer et al. 1989) · the gap is sugar and SCFAs · normal stool's gap: \(Int(osmoticGap(sodium: normalStoolSodium, potassium: normalStoolPotassium)))",
        String(format: "pH ~%.1f: carbohydrate malabsorption gives under %.1f (Eherer & Fordtran 1992)", lowerStoolPH, carbohydrateMalabsorptionPH),
        "SCFA made \(scfaProduced(mutant: mutant)), taken back \(scfaAbsorbed(mutant: mutant)), left in stool \(s.scfa) — drawn; measured salvage is higher, ~88%",
        "CH₄ too in ~1 in 3 adults (made by archaea; not drawn) · ~14% of the H₂ is breathed out: the breath test",
        "not an allergy: no immune reaction, just sugar left undigested · human lactase is LPH, not step 17's E. coli enzyme",
        "not to scale: sugars and SCFAs drawn at 0.6 of the ions' scale; a bacterium is ~10,000× a water molecule",
    ]
    for line in lines {
        c.text(line, x: x0, top: y, size: 10.5, color: muted)
        y += 14 * k
    }
}
