// What the CPU draws over the GPU's picture: the title and the caption, the
// names of the parts, the waterline and the computed draft, the midship
// section (Section.swift) and two true scale bars. Text is step 42's
// CoreText approach, as steps 53, 60, 70 and 72 copied it.

import CoreGraphics
import CoreText
import Foundation
import Metal
import simd

let ink = CGColor(srgbRed: 0.08, green: 0.09, blue: 0.11, alpha: 1)
let softInk = CGColor(srgbRed: 0.08, green: 0.09, blue: 0.11, alpha: 0.72)
let seaInk = CGColor(srgbRed: 0.02, green: 0.33, blue: 0.62, alpha: 1)

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

func cg(_ p: SIMD2<Float>) -> CGPoint { CGPoint(x: CGFloat(p.x), y: CGFloat(p.y)) }

// MARK: - geometry the tests read

/// The section inset, in units of image height (x, y from the top-left).
let insetOrigin = SIMD2<Float>(1.175, 0.515)
let insetWidth: Float = 0.54
func insetRect(height: Int) -> CGRect {
    let h: CGFloat = CGFloat(height)
    let zw: Float = sectionZRange.upperBound - sectionZRange.lowerBound
    let yh: Float = sectionYRange.upperBound - sectionYRange.lowerBound
    let w: CGFloat = CGFloat(insetWidth) * h
    let tall: CGFloat = w * CGFloat(yh / zw)
    return CGRect(x: CGFloat(insetOrigin.x) * h, y: CGFloat(insetOrigin.y) * h, width: w, height: tall)
}

/// Inset pixels per metre.
func insetPixelsPerMetre(height: Int) -> Float {
    let zw: Float = sectionZRange.upperBound - sectionZRange.lowerBound
    let w: CGFloat = insetRect(height: height).width
    return Float(w) / zw
}

/// Where a section point (z across, y up from the sea) lands in the image.
func insetPoint(z: Float, y: Float, height: Int) -> CGPoint {
    let r: CGRect = insetRect(height: height)
    let k: CGFloat = CGFloat(insetPixelsPerMetre(height: height))
    let dz: CGFloat = CGFloat(z - sectionZRange.lowerBound)
    let dy: CGFloat = CGFloat(sectionYRange.upperBound - y)
    return CGPoint(x: r.minX + dz * k, y: r.minY + dy * k)
}

/// Scale bars: 5 m at the canoe, 1 m in the inset.
let mainBarMetres: Float = 5
let insetBarMetres: Float = 1

struct ScaleBar {
    var label: String
    var x: CGFloat
    var y: CGFloat
    var pixels: CGFloat
}

func mainScaleBar(_ s: StillSetup, width: Int, height: Int) -> ScaleBar {
    let k: CGFloat = CGFloat(height) / 1080
    let px: CGFloat = CGFloat(mainBarMetres * pixelsPerMetre(at: s.centre, height: height, cam: s.camera))
    return ScaleBar(label: String(format: "%g m", mainBarMetres), x: 64 * k, y: CGFloat(height) - 64 * k, pixels: px)
}

func insetScaleBar(height: Int) -> ScaleBar {
    let r: CGRect = insetRect(height: height)
    let k: CGFloat = CGFloat(height) / 1080
    let px: CGFloat = CGFloat(insetBarMetres * insetPixelsPerMetre(height: height))
    return ScaleBar(label: String(format: "%g m", insetBarMetres), x: r.maxX - px - 22 * k, y: r.maxY - 18 * k, pixels: px)
}

func drawBar(_ ctx: CGContext, _ b: ScaleBar, k: CGFloat, height h: CGFloat, color: CGColor = ink) {
    ctx.setFillColor(color)
    ctx.fill(flip(CGRect(x: b.x, y: b.y, width: b.pixels, height: 3 * k), h))
    ctx.fill(flip(CGRect(x: b.x, y: b.y - 6 * k, width: 2 * k, height: 9 * k), h))
    ctx.fill(flip(CGRect(x: b.x + b.pixels - 2 * k, y: b.y - 6 * k, width: 2 * k, height: 9 * k), h))
    let w: CGFloat = textWidth(b.label, size: 15 * k, bold: true)
    put(b.label, ctx, x: b.x + b.pixels / 2 - w / 2, top: b.y - 26 * k, size: 15 * k, bold: true, color: color, height: h)
}

// MARK: - the caption

let captionTitle: String = "Waʻa kaulua: a Hawaiian double-hulled voyaging canoe, floating where the sea puts it"
func captionLines(_ f: Flotation) -> [String] {
    let draftFeet: Double = Double(f.draft) / metresPerFoot
    return [
        "A simplified depiction informed by the Polynesian Voyaging Society's published plans for its replica Hōkūleʻa (1975) and Herb Kawainui Kāne's design notes — not any particular canoe.",
        "PVS's plan: 62 ft 4 in overall, 54 ft on the waterline, beam 17 ft 6 in, 540 sq ft of sail, 25,000 lb fully loaded; two hulls, eight ʻiako (crossbeams), a pola (deck), two masts.",
        String(format: "In sea water of 1023.6 kg/m³ (25 °C, 35 g/kg; Nayar et al. 2016) the hulls sink until they push aside %.2f m³, which weighs the canoe's %.0f kg: draft %.3f m (%.2f ft),",
               f.displaced, canoeMass, f.draft, draftFeet),
        "solved for by buoyancy. The hulls' section is measured off PVS's midship section; their taper to the ends and keel rocker are fitted so they float as PVS's plan says.",
        "Left out: crew and stores (their weight is in the 25,000 lb), lashings, rails and rigging lines; the ʻiako drawn straight (Hōkūleʻa's are arched). The sea is drawn see-through, without refraction.",
    ]
}

// MARK: - drawing

/// The section's colours, by material.
func sectionColour(_ m: Int) -> SIMD3<UInt8> {
    switch m {
    case 2: return SIMD3<UInt8>(58, 40, 30)
    case 3: return SIMD3<UInt8>(120, 80, 48)
    case 4: return SIMD3<UInt8>(150, 100, 60)
    case 5: return SIMD3<UInt8>(178, 138, 92)
    case 6: return SIMD3<UInt8>(140, 96, 58)
    case 7: return SIMD3<UInt8>(220, 208, 180)
    case 8: return SIMD3<UInt8>(140, 96, 58)
    default: return SIMD3<UInt8>(0, 0, 0)
    }
}

func drawSection(_ ctx: CGContext, _ s: SectionInset, rect r: CGRect, imageHeight h: CGFloat) {
    // Build the section as a bitmap: canoe by material, sea below y = 0.
    var bytes: [UInt8] = Array(repeating: 255, count: s.columns * s.rows * 4)
    let yh: Float = sectionYRange.upperBound - sectionYRange.lowerBound
    for j in 0..<s.rows {
        let fy: Float = (Float(j) + 0.5) / Float(s.rows)
        let y: Float = sectionYRange.upperBound - fy * yh
        for i in 0..<s.columns {
            let m: Int = s.mat(i, j)
            var c: SIMD3<UInt8>
            if m != 0 {
                c = sectionColour(m)
                if y < 0 {
                    // The canoe under water, seen through it.
                    let blue = SIMD3<Float>(Float(c.x) * 0.75, Float(c.y) * 0.8, Float(c.z) * 0.9 + 30)
                    c = SIMD3<UInt8>(UInt8(min(blue.x, 255)), UInt8(min(blue.y, 255)), UInt8(min(blue.z, 255)))
                }
            } else {
                c = y < 0 ? SIMD3<UInt8>(178, 212, 236) : SIMD3<UInt8>(248, 249, 251)
            }
            let o: Int = (j * s.columns + i) * 4
            bytes[o] = c.x
            bytes[o + 1] = c.y
            bytes[o + 2] = c.z
            bytes[o + 3] = 255
        }
    }
    let space = CGColorSpace(name: CGColorSpace.sRGB)!
    let data = Data(bytes) as CFData
    guard let provider = CGDataProvider(data: data),
          let img = CGImage(width: s.columns, height: s.rows, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: s.columns * 4,
                            space: space, bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                            provider: provider, decode: nil, shouldInterpolate: true, intent: .defaultIntent)
    else { return }
    ctx.interpolationQuality = .high
    ctx.draw(img, in: flip(r, h))
}

func annotate(_ image: CanoeImage, setup s: StillSetup, flotation f: Flotation, sink: Float, section: SectionInset,
              keel keelWorld: [SIMD3<Float>], mutant: Mutant = activeMutant) {
    let w: Int = image.width
    let hI: Int = image.height
    guard let ctx = CGContext(data: image.pixels.contents(), width: w, height: hI, bitsPerComponent: 8,
                              bytesPerRow: w * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                              bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return }
    let h: CGFloat = CGFloat(hI)
    let k: CGFloat = h / 1080
    ctx.setShouldAntialias(true)
    let sc: Float = canoeScale(mutant)
    func at(_ p: SIMD3<Float>) -> CGPoint {
        // A point of the drawn canoe (keel frame, unit scale) in the image.
        let q: SIMD3<Float> = p * sc - SIMD3<Float>(0, sink, 0)
        return cg(project(q, width: w, height: hI, cam: s.camera))
    }

    // Title and caption, on a pale band so they read over the sky.
    put(captionTitle, ctx, x: 64 * k, top: 34 * k, size: 30 * k, bold: true, height: h)
    for (i, ln) in captionLines(f).enumerated() {
        put(ln, ctx, x: 64 * k, top: (78 + 21 * CGFloat(i)) * k, size: 15.5 * k, color: softInk, height: h)
    }

    // Part names, each with a leader to its part.
    func tag(_ text: String, _ target: CGPoint, dx: CGFloat, dy: CGFloat, italic: Bool = false) {
        let size: CGFloat = 16 * k
        let tw: CGFloat = textWidth(text, size: size, bold: false, italic: italic)
        let anchor = CGPoint(x: target.x + dx * k, y: target.y + dy * k)
        ctx.setStrokeColor(ink)
        ctx.setLineWidth(1.4 * k)
        line(ctx, target, anchor, height: h)
        ctx.setFillColor(ink)
        ctx.fillEllipse(in: flip(CGRect(x: target.x - 2.5 * k, y: target.y - 2.5 * k, width: 5 * k, height: 5 * k), h))
        let left: Bool = dx < 0
        let x: CGFloat = left ? anchor.x - tw - 6 * k : anchor.x + 6 * k
        let box = CGRect(x: x - 5 * k, y: anchor.y - 12 * k, width: tw + 10 * k, height: 24 * k)
        ctx.setFillColor(CGColor(srgbRed: 1, green: 1, blue: 1, alpha: 0.82))
        ctx.fill(flip(box, h))
        put(text, ctx, x: x, top: anchor.y - 9 * k, size: size, italic: italic, height: h)
    }
    let near: Float = -hullCentreZ
    let foreMid: SIMD2<Float> = (foresail.peak + foresail.clew) * 0.36
    let foreP = SIMD3<Float>(foresail.mastX + foreMid.x * cos(sailYaw), deckTop + foreMid.y, -foreMid.x * sin(sailYaw))
    let aftMid: SIMD2<Float> = (aftsail.peak + aftsail.clew) * 0.36
    let aftP = SIMD3<Float>(aftsail.mastX + aftMid.x * cos(sailYaw), deckTop + aftMid.y, -aftMid.x * sin(sailYaw))
    tag("peʻa ihu, the foresail: a crab-claw sail", at(foreP), dx: -150, dy: -120)
    tag("peʻa hope, the aft sail", at(aftP), dx: 120, dy: -150)
    let bowTipX: Float = manuStart(bowManu) + bowManu.tipOut * 0.8
    tag("manu (end pieces)", at(SIMD3<Float>(bowTipX, gunwaleHeight + 0.55, near)), dx: -60, dy: -60)
    let beamEnd = SIMD3<Float>(crossbeamX[1], gunwaleHeight + crossbeamRadius, -crossbeamHalfSpan)
    tag("ʻiako: eight crossbeams", at(beamEnd), dx: -150, dy: -25)
    tag("pola (deck)", at(SIMD3<Float>(crossbeamX[2] - 0.9, deckTop, -0.6)), dx: -30, dy: -70)
    tag("kia (mast)", at(SIMD3<Float>(aftmastX, deckTop + 3.5, 0)), dx: -70, dy: 20)
    tag("hoe uli (steering paddle)", at(SIMD3<Float>(-9.9, 0.95, 0)), dx: -20, dy: 110)

    // The waterline and the draft on the near hull: the keel, under the
    // sea, dashed (found by the columns kernel); a dimension from it up to the waterline at midships.
    ctx.setStrokeColor(seaInk)
    ctx.setLineWidth(2 * k)
    ctx.setLineDash(phase: 0, lengths: [7 * k, 5 * k])
    let wu: Float = sink / sc
    let keel: [CGPoint] = keelWorld.map { cg(project($0, width: w, height: hI, cam: s.camera)) }
    if let first = keel.first {
        ctx.beginPath()
        ctx.move(to: CGPoint(x: first.x, y: h - first.y))
        for p in keel.dropFirst() { ctx.addLine(to: CGPoint(x: p.x, y: h - p.y)) }
        ctx.strokePath()
    }
    ctx.setLineDash(phase: 0, lengths: [])
    let dimX: Float = 0.0
    let top: CGPoint = at(SIMD3<Float>(dimX, wu, near - hullHalfWidth - 0.35))
    let bottom: CGPoint = at(SIMD3<Float>(dimX, 0, near - hullHalfWidth - 0.35))
    let keelMid: CGPoint = at(SIMD3<Float>(dimX, 0, near))
    ctx.setLineWidth(1.6 * k)
    line(ctx, top, bottom, height: h)
    line(ctx, CGPoint(x: top.x - 7 * k, y: top.y), CGPoint(x: top.x + 7 * k, y: top.y), height: h)
    line(ctx, CGPoint(x: bottom.x - 7 * k, y: bottom.y), CGPoint(x: keelMid.x + 4 * k, y: keelMid.y), height: h)
    let draftText = String(format: "draft %.2f m, from buoyancy", sink)
    let dtw: CGFloat = textWidth(draftText, size: 16 * k, bold: true)
    ctx.setFillColor(CGColor(srgbRed: 1, green: 1, blue: 1, alpha: 0.85))
    let boxX: CGFloat = bottom.x - dtw / 2 - 6 * k
    ctx.fill(flip(CGRect(x: boxX, y: bottom.y + 10 * k, width: dtw + 12 * k, height: 24 * k), h))
    put(draftText, ctx, x: bottom.x - dtw / 2, top: bottom.y + 13 * k, size: 16 * k, bold: true, color: seaInk, height: h)
    tag("waterline", at(SIMD3<Float>(6.6, wu, near - 0.3)), dx: -40, dy: 45, italic: true)

    // The midship section.
    let r: CGRect = insetRect(height: hI)
    ctx.setFillColor(CGColor(srgbRed: 1, green: 1, blue: 1, alpha: 1))
    ctx.fill(flip(r.insetBy(dx: -1, dy: -1), h))
    drawSection(ctx, section, rect: r, imageHeight: h)
    ctx.setStrokeColor(ink)
    ctx.setLineWidth(2.5 * k)
    ctx.stroke(flip(r.insetBy(dx: -1.25 * k, dy: -1.25 * k), h))
    put("Cut across at midships, sampled from the same shapes", ctx, x: r.minX, top: r.minY - 50 * k, size: 17 * k, bold: true, height: h)
    put("the sea where the load puts it: both hulls carry it together", ctx, x: r.minX, top: r.minY - 28 * k, size: 15 * k,
        color: softInk, height: h)
    // Waterline across the inset.
    let wl0: CGPoint = insetPoint(z: sectionZRange.lowerBound, y: 0, height: hI)
    let wl1: CGPoint = insetPoint(z: sectionZRange.upperBound, y: 0, height: hI)
    ctx.setStrokeColor(seaInk)
    ctx.setLineWidth(2 * k)
    line(ctx, wl0, wl1, height: h)
    put("waterline", ctx, x: wl1.x - 80 * k, top: wl1.y - 22 * k, size: 14 * k, italic: true, color: seaInk, height: h)
    // Draft on the port hull (the one on the left), on its inner side.
    let hz: Float = -hullCentreZ * sc
    let keelY: Float = -sink
    let dz: Float = hz + hullHalfWidth * sc + 0.25
    let p0: CGPoint = insetPoint(z: dz, y: 0, height: hI)
    let p1: CGPoint = insetPoint(z: dz, y: keelY, height: hI)
    let pk: CGPoint = insetPoint(z: hz, y: keelY, height: hI)
    ctx.setLineWidth(1.6 * k)
    line(ctx, p0, p1, height: h)
    line(ctx, CGPoint(x: p1.x + 6 * k, y: p1.y), CGPoint(x: pk.x, y: pk.y), height: h)
    line(ctx, CGPoint(x: p0.x - 6 * k, y: p0.y), CGPoint(x: p0.x + 6 * k, y: p0.y), height: h)
    let d1 = String(format: "draft %.3f m", sink)
    let d2 = String(format: "(PVS: %.3f m)", pvsDraft)
    let mid: CGFloat = (p0.y + p1.y) / 2
    put(d1, ctx, x: p0.x + 10 * k, top: mid - 16 * k, size: 15 * k, bold: true, color: seaInk, height: h)
    put(d2, ctx, x: p0.x + 10 * k, top: mid + 2 * k, size: 13 * k, color: seaInk, height: h)
    // The beam, over the crossbeam ends.
    let beamY: Float = (gunwaleHeight + 2 * crossbeamRadius) * sc - sink + 0.45
    let b0: CGPoint = insetPoint(z: -crossbeamHalfSpan * sc, y: beamY, height: hI)
    let b1: CGPoint = insetPoint(z: crossbeamHalfSpan * sc, y: beamY, height: hI)
    ctx.setStrokeColor(ink)
    ctx.setLineWidth(1.4 * k)
    line(ctx, b0, b1, height: h)
    line(ctx, CGPoint(x: b0.x, y: b0.y - 6 * k), CGPoint(x: b0.x, y: b0.y + 6 * k), height: h)
    line(ctx, CGPoint(x: b1.x, y: b1.y - 6 * k), CGPoint(x: b1.x, y: b1.y + 6 * k), height: h)
    let beamText = String(format: "beam %.2f m (PVS: 17 ft 6 in)", crossbeamHalfSpan * 2 * sc)
    let btw: CGFloat = textWidth(beamText, size: 14 * k, bold: false)
    let beamMid: CGFloat = (b0.x + b1.x) / 2
    let beamLeft: CGFloat = beamMid - btw / 2
    let beamTop: CGFloat = b0.y - 22 * k
    put(beamText, ctx, x: beamLeft, top: beamTop, size: 14 * k, height: h)
    let mass = String(format: "%.0f kg ÷ %.1f kg/m³ = %.2f m³ below the waterline", canoeMass, seaWaterDensity, canoeMass / seaWaterDensity)
    put(mass, ctx, x: r.minX + 12 * k, top: r.maxY - 30 * k, size: 13.5 * k, color: ink, height: h)
    drawBar(ctx, insetScaleBar(height: hI), k: k, height: h)

    let bar: ScaleBar = mainScaleBar(s, width: w, height: hI)
    drawBar(ctx, bar, k: k, height: h, color: CGColor(srgbRed: 1, green: 1, blue: 1, alpha: 1))
    put("at the canoe's middle, on the waterline", ctx, x: bar.x + bar.pixels + 12 * k, top: bar.y - 7 * k, size: 13.5 * k,
        color: CGColor(srgbRed: 1, green: 1, blue: 1, alpha: 0.9), height: h)
}
