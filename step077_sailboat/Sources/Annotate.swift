// What the CPU draws over the GPU's picture: the title, the caption with
// the buoyancy sum, labels on the boat, the draught with the board down on
// the cut face, a true scale bar, and an inset: the hull's cross-section at
// its deepest station, drawn from the same distance function, true to scale,
// with the waterline Archimedes gives and its draught measured. Text is step
// 42's CoreText approach, as steps 53, 60 and 72 copied it.

import CoreGraphics
import CoreText
import Foundation
import Metal
import simd

let ink = CGColor(srgbRed: 0.10, green: 0.11, blue: 0.12, alpha: 1)
let softInk = CGColor(srgbRed: 0.10, green: 0.11, blue: 0.12, alpha: 0.70)
let waterInk = CGColor(srgbRed: 0.05, green: 0.22, blue: 0.50, alpha: 1)

func ctLine(_ s: String, size: CGFloat, bold: Bool, italic: Bool = false, color: CGColor) -> CTLine {
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

func put(_ s: String, _ ctx: CGContext, x: CGFloat, top: CGFloat, size: CGFloat, bold: Bool = false,
         italic: Bool = false, color: CGColor = ink, height: CGFloat) {
    ctx.textPosition = CGPoint(x: x, y: height - top - size)
    CTLineDraw(ctLine(s, size: size, bold: bold, italic: italic, color: color), ctx)
}

func line(_ ctx: CGContext, _ a: CGPoint, _ b: CGPoint, height h: CGFloat) {
    ctx.beginPath()
    ctx.move(to: CGPoint(x: a.x, y: h - a.y))
    ctx.addLine(to: CGPoint(x: b.x, y: h - b.y))
    ctx.strokePath()
}

func flip(_ r: CGRect, _ h: CGFloat) -> CGRect {
    CGRect(x: r.minX, y: h - r.maxY, width: r.width, height: r.height)
}

/// A dimension line between two image points with end ticks.
func dimension(_ ctx: CGContext, _ a: CGPoint, _ b: CGPoint, tick: CGFloat, height h: CGFloat) {
    line(ctx, a, b, height: h)
    let d = CGPoint(x: b.x - a.x, y: b.y - a.y)
    let len: CGFloat = max((d.x * d.x + d.y * d.y).squareRoot(), 1e-6)
    let n = CGPoint(x: -d.y / len * tick, y: d.x / len * tick)
    line(ctx, CGPoint(x: a.x - n.x, y: a.y - n.y), CGPoint(x: a.x + n.x, y: a.y + n.y), height: h)
    line(ctx, CGPoint(x: b.x - n.x, y: b.y - n.y), CGPoint(x: b.x + n.x, y: b.y + n.y), height: h)
}

// MARK: - geometry the tests read

/// The inset, in units of image height (x, y from the top-left).
let insetOrigin = SIMD2<Float>(1.00, 0.36)
let insetSize = SIMD2<Float>(0.72, 0.31)
/// The inset's field: metres across and the height of its lowest edge
/// below the waterline.
let insetFieldWidth: Float = 2.1
let insetBelow: Float = 0.30

func insetRect(height: Int) -> CGRect {
    let h: CGFloat = CGFloat(height)
    return CGRect(x: CGFloat(insetOrigin.x) * h, y: CGFloat(insetOrigin.y) * h,
                  width: CGFloat(insetSize.x) * h, height: CGFloat(insetSize.y) * h)
}

/// Pixels per metre in the inset.
func insetPixelsPerMetre(height: Int) -> Float { insetSize.x * Float(height) / insetFieldWidth }

/// The inset's station: where the keel line is lowest.
let insetStation: Double = hullShape.keelLowX

/// Scale bars: 1 m at the boat's centre plane, 100 mm in the inset.
let mainBarMetres: Float = 1
let insetBarMetres: Float = 0.1

struct ScaleBar {
    var label: String
    var x: CGFloat
    var y: CGFloat
    var pixels: CGFloat
}

func mainScaleBar(_ s: StillSetup, width: Int, height: Int) -> ScaleBar {
    let k: CGFloat = CGFloat(height) / 1080
    let px: CGFloat = CGFloat(mainBarMetres * pixelsPerMetre(at: s.centre, height: height, cam: s.camera))
    return ScaleBar(label: String(format: "%g m", mainBarMetres), x: 64 * k, y: CGFloat(height) - 60 * k, pixels: px)
}

func insetScaleBar(height: Int) -> ScaleBar {
    let r: CGRect = insetRect(height: height)
    let k: CGFloat = CGFloat(height) / 1080
    let px: CGFloat = CGFloat(insetBarMetres * insetPixelsPerMetre(height: height))
    return ScaleBar(label: String(format: "%g mm", insetBarMetres * 1000), x: r.minX + 24 * k, y: r.maxY - 18 * k, pixels: px)
}

func drawBar(_ ctx: CGContext, _ b: ScaleBar, k: CGFloat, height h: CGFloat) {
    ctx.setFillColor(ink)
    ctx.fill(flip(CGRect(x: b.x, y: b.y, width: b.pixels, height: 3 * k), h))
    ctx.fill(flip(CGRect(x: b.x, y: b.y - 6 * k, width: 2 * k, height: 9 * k), h))
    let rightTick: CGFloat = b.x + b.pixels - 2 * k
    ctx.fill(flip(CGRect(x: rightTick, y: b.y - 6 * k, width: 2 * k, height: 9 * k), h))
    let w: CGFloat = textWidth(b.label, size: 15 * k, bold: true)
    let tx: CGFloat = b.x + b.pixels / 2 - w / 2
    put(b.label, ctx, x: tx, top: b.y - 26 * k, size: 15 * k, bold: true, height: h)
}

// MARK: - what the picture says

struct Report {
    var flotation: Flotation
    var drawn: Double
}

let captionTitle: String = "An ILCA 7 (Laser) dinghy, floating where Archimedes puts it"
func captionLines(_ r: Report) -> [String] {
    let v: Double = r.flotation.volume
    return [
        String(format: "Class size: %.2f m long, %.2f m beam (Wikipedia, \"Laser (dinghy)\"); sail %.2f m² — luff %.2f, foot %.2f, leech %.2f m (lasersailingtips).",
               classLength, classBeam, sailArea, sailLuff, sailFoot, sailLeech),
        String(format: "Mast %.3f m assembled from its two sections; centreboard %.0f mm chord, %.0f mm below the hull (Laser sailors' forums: unverified).",
               mastLength, boardChord * 1000, boardDepthBelowHull * 1000),
        String(format: "Load (Day & Nixon 2014): hull %.0f kg + rig and foils %.0f kg + an %.0f kg sailor (counted, not drawn) = %.0f kg.",
               hullMass, rigAndFoilsMass, sailorMass, totalMass),
        String(format: "Fresh water at 15 °C, %.1f kg/m³: it sinks until it displaces %.4f m³ — %.1f kg of water — hull, board and rudder together.",
               waterDensity, v, v * waterDensity),
        "The hull is a few simple shapes fitted to Day & Nixon's tank-test hydrostatics (their 160 kg case), not the builder's lines.",
        "Cut-away: the near side of the water is sliced off outside the hull; refraction is left out so the underwater body reads true to scale.",
    ]
}

// MARK: - the inset's section, from the distance function

/// The hull's section at the inset station in world coordinates: inside
/// where the hull's own distance is negative.
func sectionInside(z: Double, yWorld: Double, draught: Double, mutant m: Mutant = activeMutant) -> Bool {
    let s: Double = boatScale(m)
    let q = SIMD3<Double>(insetStation, yWorld + draught, z) / s
    return hullDistance(q) < 0
}

/// Lowest point of the section, at the centreline, in world metres.
func sectionBottom(draught: Double, mutant m: Mutant = activeMutant) -> Double {
    // Between well below the hull and the waterline (inside the hull at the
    // centreline); above it, the cockpit well opens the section.
    var lo: Double = -1, hi: Double = 0
    for _ in 0..<60 {
        let mid: Double = (lo + hi) / 2
        if sectionInside(z: 0, yWorld: mid, draught: draught, mutant: m) { hi = mid } else { lo = mid }
    }
    return (lo + hi) / 2
}

/// The section's half-width at a world height: scanning in from outside
/// to the first point inside, then refined by bisection.
func sectionHalfWidth(yWorld: Double, draught: Double, mutant m: Mutant = activeMutant) -> Double {
    var z: Double = 1.2
    while z > 0 && !sectionInside(z: z, yWorld: yWorld, draught: draught, mutant: m) { z -= 0.002 }
    if z <= 0 { return 0 }
    var a: Double = z, b: Double = z + 0.002
    for _ in 0..<40 {
        let mid: Double = (a + b) / 2
        if sectionInside(z: mid, yWorld: yWorld, draught: draught, mutant: m) { a = mid } else { b = mid }
    }
    return (a + b) / 2
}

func drawInset(_ ctx: CGContext, rect r: CGRect, draught: Double, imageHeight h: CGFloat, k: CGFloat) {
    let ppm: CGFloat = r.width / CGFloat(insetFieldWidth)
    let waterY: CGFloat = r.maxY - CGFloat(insetBelow) * ppm
    let cx: CGFloat = r.midX
    ctx.setFillColor(CGColor(srgbRed: 0.97, green: 0.97, blue: 0.96, alpha: 1))
    ctx.fill(flip(r, h))
    ctx.setFillColor(CGColor(srgbRed: 0.72, green: 0.84, blue: 0.90, alpha: 1))
    ctx.fill(flip(CGRect(x: r.minX, y: waterY, width: r.width, height: r.maxY - waterY), h))
    // The section, pixel by pixel (4 × 4 samples each), from the distance function.
    let x0: Int = Int(r.minX), x1: Int = Int(r.maxX), y0: Int = Int(r.minY), y1: Int = Int(r.maxY)
    for py in y0..<y1 {
        for px in x0..<x1 {
            var n: Int = 0
            for sy in 0..<4 {
                for sx in 0..<4 {
                    let fx: CGFloat = CGFloat(px) + (CGFloat(sx) + 0.5) / 4
                    let fy: CGFloat = CGFloat(py) + (CGFloat(sy) + 0.5) / 4
                    let z: Double = Double((fx - cx) / ppm)
                    let yw: Double = Double((waterY - fy) / ppm)
                    if sectionInside(z: z, yWorld: yw, draught: draught) { n += 1 }
                }
            }
            if n > 0 {
                let a: CGFloat = CGFloat(n) / 16
                let under: Bool = CGFloat(py) + 0.5 > waterY
                let c: CGColor = under ? CGColor(srgbRed: 0.93, green: 0.94, blue: 0.93, alpha: a)
                                       : CGColor(srgbRed: 1.0, green: 1.0, blue: 1.0, alpha: a)
                ctx.setFillColor(CGColor(srgbRed: 0.25, green: 0.27, blue: 0.30, alpha: a))
                ctx.fill(flip(CGRect(x: CGFloat(px), y: CGFloat(py), width: 1, height: 1), h))
                // A 2-px outline: fill the interior lighter where all neighbours are inside.
                let zc: Double = Double((CGFloat(px) + 0.5 - cx) / ppm)
                let yc: Double = Double((waterY - CGFloat(py) - 0.5) / ppm)
                let e: Double = Double(2.2 / ppm)
                let deep: Bool = sectionInside(z: zc - e, yWorld: yc, draught: draught)
                    && sectionInside(z: zc + e, yWorld: yc, draught: draught)
                    && sectionInside(z: zc, yWorld: yc - e, draught: draught)
                    && sectionInside(z: zc, yWorld: yc + e, draught: draught)
                if deep && n == 16 {
                    ctx.setFillColor(c)
                    ctx.fill(flip(CGRect(x: CGFloat(px), y: CGFloat(py), width: 1, height: 1), h))
                }
            }
        }
    }
    // The waterline.
    ctx.setStrokeColor(waterInk)
    ctx.setLineWidth(2 * k)
    line(ctx, CGPoint(x: r.minX, y: waterY), CGPoint(x: r.maxX, y: waterY), height: h)
    put("waterline, from buoyancy", ctx, x: r.minX + 14 * k, top: waterY + 8 * k, size: 13 * k, bold: true, color: waterInk, height: h)

    // Draught at the centreline.
    let bottom: Double = sectionBottom(draught: draught)
    let by: CGFloat = waterY - CGFloat(bottom) * ppm
    ctx.setStrokeColor(ink)
    ctx.setLineWidth(1.5 * k)
    let dx: CGFloat = cx
    dimension(ctx, CGPoint(x: dx, y: waterY), CGPoint(x: dx, y: by), tick: 7 * k, height: h)
    let tLabel: String = String(format: "hull draught %.0f mm", -bottom * 1000)
    put(tLabel, ctx, x: dx + 12 * k, top: by + 8 * k, size: 15 * k, bold: true, height: h)
    put("(Day & Nixon measured 94 mm, board out, at this load)", ctx, x: dx + 12 * k, top: by + 27 * k, size: 12.5 * k,
        color: softInk, height: h)

    // Waterline beam and deck beam.
    let wl: Double = sectionHalfWidth(yWorld: -0.0005, draught: draught)
    let wlPx: CGFloat = CGFloat(wl) * ppm
    let wly: CGFloat = waterY + 14 * k
    ctx.setStrokeColor(waterInk)
    dimension(ctx, CGPoint(x: cx - wlPx, y: wly), CGPoint(x: cx + wlPx, y: wly), tick: 5 * k, height: h)
    let deckY: Double = hullShape.depth * boatScale() - draught
    let deckHalf: Double = sectionHalfWidth(yWorld: deckY - 0.004, draught: draught)
    let dkPx: CGFloat = CGFloat(deckHalf) * ppm
    let deckPx: CGFloat = CGFloat(deckY) * ppm
    let dky: CGFloat = waterY - deckPx - 16 * k
    ctx.setStrokeColor(ink)
    dimension(ctx, CGPoint(x: cx - dkPx, y: dky), CGPoint(x: cx + dkPx, y: dky), tick: 5 * k, height: h)
    let deckLabel: String = String(format: "deck %.2f m wide here (beam %.2f m at the widest)", 2 * deckHalf, classBeam * boatScale())
    put(deckLabel, ctx, x: cx - textWidth(deckLabel, size: 14 * k, bold: true) / 2, top: dky - 24 * k, size: 14 * k, bold: true, height: h)
    let wlLabel: String = String(format: "waterline beam %.2f m here", 2 * wl)
    put(wlLabel, ctx, x: r.minX + 14 * k, top: waterY + 27 * k, size: 13 * k, color: waterInk, height: h)

    drawBar(ctx, insetScaleBar(height: Int(h)), k: k, height: h)
}

// MARK: - drawing

func annotate(_ image: BoatImage, setup s: StillSetup, report rep: Report, mutant m: Mutant = activeMutant) {
    let w: Int = image.width
    let hI: Int = image.height
    guard let ctx = CGContext(data: image.pixels.contents(), width: w, height: hI, bitsPerComponent: 8,
                              bytesPerRow: w * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                              bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return }
    let h: CGFloat = CGFloat(hI)
    let k: CGFloat = h / 1080
    ctx.setShouldAntialias(true)
    let draught: Double = rep.drawn
    let sc: Double = boatScale(m)
    let cam: Camera = s.camera
    func at(_ p: SIMD3<Double>) -> CGPoint {
        let q: SIMD3<Double> = p * sc - SIMD3<Double>(0, draught, 0)
        let v: SIMD2<Float> = project(SIMD3<Float>(Float(q.x), Float(q.y), Float(q.z)), width: w, height: hI, cam: cam)
        return CGPoint(x: CGFloat(v.x), y: CGFloat(v.y))
    }

    // Title and caption, top right.
    let colX: CGFloat = 1000 * k
    let shownVolume: Double = rep.flotation.body.volume(below: draught)
    put(captionTitle, ctx, x: colX, top: 40 * k, size: 28 * k, bold: true, height: h)
    let eq: String = String(format: "ρ V = M:  %.1f kg/m³ × %.4f m³ = %.1f kg  →  hull draught %.0f mm",
                            waterDensity, shownVolume, waterDensity * shownVolume, draught * 1000)
    put(eq, ctx, x: colX, top: 82 * k, size: 19 * k, bold: true, color: waterInk, height: h)
    for (i, ln) in captionLines(rep).enumerated() {
        put(ln, ctx, x: colX, top: (122 + 21 * CGFloat(i)) * k, size: 13.2 * k, color: softInk, height: h)
    }

    // The inset.
    let r: CGRect = insetRect(height: hI)
    drawInset(ctx, rect: r, draught: draught, imageHeight: h, k: k)
    ctx.setStrokeColor(ink)
    ctx.setLineWidth(2.5 * k)
    ctx.stroke(flip(r.insetBy(dx: -1.25 * k, dy: -1.25 * k), h))
    let stationFromBow: Double = hullShape.bow - insetStation
    put(String(format: "Cross-section at the deepest station, %.2f m aft of the bow, true to scale", stationFromBow),
        ctx, x: r.minX, top: r.minY - 27 * k, size: 16 * k, bold: true, height: h)

    // Labels on the boat.
    let sail: Sail = sailGeometry()
    ctx.setStrokeColor(softInk)
    ctx.setLineWidth(1.2 * k)
    func label(_ lines: [String], anchor: SIMD3<Double>, dx: CGFloat, dy: CGFloat, bold: Bool = false, color: CGColor = ink) {
        let a: CGPoint = at(anchor)
        let tx: CGFloat = a.x + dx * k
        let ty: CGFloat = a.y + dy * k
        // The leader meets the text at its near end.
        var wmax: CGFloat = 0
        for (i, l) in lines.enumerated() { wmax = max(wmax, textWidth(l, size: 14 * k, bold: bold && i == 0)) }
        let end: CGFloat = dx < 0 ? tx + wmax + 5 * k : tx - 4 * k
        let lineStep: CGFloat = 17 * k
        line(ctx, a, CGPoint(x: end, y: ty + 9 * k), height: h)
        for (i, l) in lines.enumerated() {
            let row: CGFloat = CGFloat(i)
            let rowTop: CGFloat = ty + row * lineStep
            put(l, ctx, x: tx, top: rowTop, size: 14 * k, bold: bold && i == 0, color: color, height: h)
        }
    }
    let headLabel: SIMD3<Double> = sail.head + SIMD3<Double>(0, 0.02, 0)
    label([String(format: "mast %.2f m", mastLength), "(2 aluminium sections)"], anchor: headLabel, dx: 40, dy: -8, bold: true)
    let corners: SIMD3<Double> = sail.tack + sail.head + sail.clew
    let sailMid: SIMD3<Double> = corners / 3
    label([String(format: "mainsail %.2f m²", sail.area), String(format: "(class %.2f m² with its roach)", sailArea)],
          anchor: sailMid, dx: -390, dy: -150, bold: true)
    let boomMid: SIMD3<Double> = (sail.tack + sail.clew) / 2
    label(["boom"], anchor: boomMid, dx: -40, dy: 40, bold: true)
    if m != .noKeel {
        let tip: Double = keelHeight(boardX) - boardDepthBelowHull
        let boardLow = SIMD3<Double>(boardX + boardChord / 2, tip + 0.15, 0)
        label(["centreboard", String(format: "%.0f mm chord, %.0f mm below the hull", boardChord * 1000, boardDepthBelowHull * 1000)],
              anchor: boardLow, dx: 120, dy: 30, bold: true)
    }
    let rud = SIMD3<Double>(rudderX - rudderChord / 2, rudderBottomY + 0.2, 0)
    label(["rudder", "(size: model)"], anchor: rud, dx: -150, dy: 40, bold: true)

    // The draught with the board down, on the centre plane beside the board.
    let tipW: Double = (keelHeight(boardX) - boardDepthBelowHull) * sc - draught
    let dimX: Double = (boardX + boardChord / 2 + 0.35) * sc
    let pa: CGPoint = at(SIMD3<Double>(dimX / sc, draught / sc, 0))
    let pb: CGPoint = at(SIMD3<Double>(dimX / sc, (tipW + draught) / sc, 0))
    ctx.setStrokeColor(ink)
    ctx.setLineWidth(1.8 * k)
    if m != .noKeel {
        let tipGuide: CGPoint = at(SIMD3<Double>(boardX + boardChord / 2, (tipW + draught) / sc, 0))
        ctx.setLineDash(phase: 0, lengths: [4 * k, 4 * k])
        line(ctx, tipGuide, pb, height: h)
        ctx.setLineDash(phase: 0, lengths: [])
        dimension(ctx, pa, pb, tick: 6 * k, height: h)
        let midY: CGFloat = (pa.y + pb.y) / 2
        put(String(format: "%.2f m draught, board down", -tipW), ctx, x: pa.x + 12 * k, top: midY - 12 * k,
            size: 15 * k, bold: true, height: h)
        put(String(format: "(class figure %.3f m)", classDraft), ctx, x: pa.x + 12 * k, top: midY + 7 * k,
            size: 13 * k, color: softInk, height: h)
    }
    // The waterline on the cut face.
    let wlPoint: CGPoint = at(SIMD3<Double>(-3.3 / sc, draught / sc, Double(s.water.hi.z) / sc))
    put("water level", ctx, x: wlPoint.x + 6 * k, top: wlPoint.y - 22 * k, size: 14 * k, bold: true, color: waterInk, height: h)
    put("fresh water, cut away", ctx, x: wlPoint.x + 6 * k, top: wlPoint.y + 8 * k, size: 13 * k, color: waterInk, height: h)

    let bar: ScaleBar = mainScaleBar(s, width: w, height: hI)
    drawBar(ctx, bar, k: k, height: h)
    put("at the boat's centre plane", ctx, x: bar.x + bar.pixels + 12 * k, top: bar.y - 7 * k, size: 13 * k, color: softInk, height: h)
}
