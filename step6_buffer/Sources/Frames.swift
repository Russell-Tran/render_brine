// Frames: the GPU draws the molecules on the left; the CPU draws the labels,
// pH gauge and pH-over-time plot on the right, straight into the same memory
// (unified memory again). Frames are collected into an animated GIF.

import CoreGraphics
import CoreText
import Foundation
import ImageIO
import Metal
import UniformTypeIdentifiers

struct FrameLayout {
    let boxPixels: Int
    let panelWidth: Int
    var width: Int { boxPixels + panelWidth }
    var height: Int { boxPixels }
}

/// What the side panel shows for one frame.
struct PanelInfo {
    var time: Double
    var pH: Double
    var concentrations: [Species: Double]   // mM
    var history: [(time: Double, pH: Double)]
    var doseTime: Double
    var doseMM: Double
    var endTime: Double
    var slowdown: Double
}

let pHAxis: ClosedRange<Double> = 2...8

/// Where a pH value sits along the gauge, 0 (pH 2) to 1 (pH 8).
func gaugeFraction(_ pH: Double) -> Double {
    let f: Double = (pH - pHAxis.lowerBound) / (pHAxis.upperBound - pHAxis.lowerBound)
    return min(max(f, 0), 1)
}

private let ink = CGColor(srgbRed: 0.90, green: 0.94, blue: 0.95, alpha: 1)
private let muted = CGColor(srgbRed: 0.58, green: 0.68, blue: 0.72, alpha: 1)
private let panelBackground = CGColor(srgbRed: 0.06, green: 0.10, blue: 0.13, alpha: 1)
private let accent = CGColor(srgbRed: 0.95, green: 0.25, blue: 0.30, alpha: 1)

private func color(_ c: SIMD3<Float>) -> CGColor {
    CGColor(srgbRed: CGFloat(c.x), green: CGFloat(c.y), blue: CGFloat(c.z), alpha: 1)
}

/// Draws text with its top-left corner at (x, top), in top-down coordinates.
private func text(_ s: String, _ ctx: CGContext, x: CGFloat, top: CGFloat, size: CGFloat,
                  bold: Bool = false, color: CGColor = ink, height: CGFloat) {
    let fontName = bold ? "HelveticaNeue-Bold" : "HelveticaNeue"
    let font = CTFontCreateWithName(fontName as CFString, size, nil)
    let attributes: [NSAttributedString.Key: Any] = [
        NSAttributedString.Key(kCTFontAttributeName as String): font,
        NSAttributedString.Key(kCTForegroundColorAttributeName as String): color,
    ]
    let line = CTLineCreateWithAttributedString(NSAttributedString(string: s, attributes: attributes))
    ctx.textPosition = CGPoint(x: x, y: height - top - size)
    CTLineDraw(line, ctx)
}

/// Draws the side panel into the right-hand part of `frame`.
func drawPanel(_ info: PanelInfo, into frame: MTLBuffer, layout: FrameLayout) {
    guard let ctx = CGContext(data: frame.contents(), width: layout.width, height: layout.height,
                              bitsPerComponent: 8, bytesPerRow: layout.width * 4,
                              space: CGColorSpace(name: CGColorSpace.sRGB)!,
                              bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return }
    let h = CGFloat(layout.height)
    let x0 = CGFloat(layout.boxPixels)
    let w = CGFloat(layout.panelWidth)
    let pad: CGFloat = 16
    let left = x0 + pad
    let inner = w - 2 * pad
    func top(_ t: CGFloat) -> CGFloat { h - t }   // top-down y to CoreGraphics y

    ctx.setFillColor(panelBackground)
    ctx.fill(CGRect(x: x0, y: 0, width: w, height: h))

    text("Bicarbonate buffer", ctx, x: left, top: 14, size: 15, bold: true, height: h)
    text("CO₂ + H₂O ⇌ H₂CO₃ ⇌ H⁺ + HCO₃⁻", ctx, x: left, top: 34, size: 11.5, color: muted, height: h)
    text(String(format: "t = %.2f s   (slowed down %.0f×)", info.time, info.slowdown), ctx,
         x: left, top: 54, size: 11.5, color: muted, height: h)

    // Big pH readout.
    text(String(format: "pH %.2f", info.pH), ctx, x: left, top: 72, size: 30, bold: true, height: h)

    // Gauge from pH 2 to 8.
    let gaugeTop: CGFloat = 112
    let gaugeHeight: CGFloat = 8
    ctx.setFillColor(CGColor(srgbRed: 0.16, green: 0.24, blue: 0.28, alpha: 1))
    ctx.fill(CGRect(x: left, y: top(gaugeTop + gaugeHeight), width: inner, height: gaugeHeight))
    let marker = left + inner * CGFloat(gaugeFraction(info.pH))
    ctx.setFillColor(ink)
    ctx.fill(CGRect(x: marker - 1.5, y: top(gaugeTop + gaugeHeight + 4), width: 3, height: gaugeHeight + 8))
    for p in stride(from: 2, through: 8, by: 2) {
        let x = left + inner * CGFloat(gaugeFraction(Double(p)))
        text("\(p)", ctx, x: x - 3, top: gaugeTop + 13, size: 10, color: muted, height: h)
    }

    // pH over time.
    let plotTop: CGFloat = 146
    let plotHeight: CGFloat = 70
    ctx.setStrokeColor(CGColor(srgbRed: 0.16, green: 0.24, blue: 0.28, alpha: 1))
    ctx.setLineWidth(1)
    for p in stride(from: 2.0, through: 8.0, by: 2.0) {
        let y = top(plotTop + plotHeight * CGFloat(1 - gaugeFraction(p)))
        ctx.move(to: CGPoint(x: left, y: y))
        ctx.addLine(to: CGPoint(x: left + inner, y: y))
    }
    ctx.strokePath()
    func plotX(_ t: Double) -> CGFloat { left + inner * CGFloat(t / info.endTime) }
    func plotY(_ p: Double) -> CGFloat { top(plotTop + plotHeight * CGFloat(1 - gaugeFraction(p))) }
    // Dose marker.
    ctx.setStrokeColor(accent)
    ctx.move(to: CGPoint(x: plotX(info.doseTime), y: top(plotTop)))
    ctx.addLine(to: CGPoint(x: plotX(info.doseTime), y: top(plotTop + plotHeight)))
    ctx.strokePath()
    if info.history.count > 1 {
        ctx.setStrokeColor(ink)
        ctx.setLineWidth(2)
        ctx.setLineJoin(.round)
        ctx.move(to: CGPoint(x: plotX(info.history[0].time), y: plotY(info.history[0].pH)))
        for point in info.history.dropFirst() { ctx.addLine(to: CGPoint(x: plotX(point.time), y: plotY(point.pH))) }
        ctx.strokePath()
    }
    text("pH over time", ctx, x: left, top: plotTop + plotHeight + 4, size: 10, color: muted, height: h)

    // Legend with live concentrations.
    var y: CGFloat = 240
    for s in [Species.bicarbonate, .co2, .carbonicAcid, .hydrogen, .sodium, .chloride] {
        ctx.setFillColor(color(s.color))
        ctx.fillEllipse(in: CGRect(x: left, y: top(y + 11), width: 9, height: 9))
        text(s.name, ctx, x: left + 16, top: y, size: 12, height: h)
        text(concentrationLabel(info.concentrations[s] ?? 0), ctx, x: left + 100, top: y, size: 12,
             color: muted, height: h)
        y += 17
    }

    // What's happening.
    let status: String
    let detail: String
    if info.time < info.doseTime {
        status = String(format: "Adding %.0f mM HCl at t = %.2f s", info.doseMM, info.doseTime)
        detail = "HCO₃⁻ and CO₂ at equilibrium"
    } else {
        status = String(format: "Same acid in plain water: pH %.1f", plainWaterPH(afterAcid: info.doseMM))
        detail = "H⁺ + HCO₃⁻ → H₂CO₃ at once; → CO₂ in ~0.5 s"
    }
    text(status, ctx, x: left, top: 346, size: 12, bold: true, height: h)
    text(detail, ctx, x: left, top: 363, size: 11, color: muted, height: h)
    text(String(format: "1 dot = %.0f µM · 25 °C · H⁺ glows", micromolarPerDot), ctx,
         x: left, top: h - 18, size: 10, color: muted, height: h)
}

/// Collects frames into an animated GIF that loops forever.
final class GIFWriter {
    private let destination: CGImageDestination
    private let delay: Double

    init(url: URL, frameCount: Int, delay: Double) throws {
        try? FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        guard let d = CGImageDestinationCreateWithURL(url as CFURL, UTType.gif.identifier as CFString, frameCount, nil) else {
            throw SimulationError.gpu("could not create \(url.path)")
        }
        let loop = [kCGImagePropertyGIFDictionary as String: [kCGImagePropertyGIFLoopCount as String: 0]]
        CGImageDestinationSetProperties(d, loop as CFDictionary)
        destination = d
        self.delay = delay
    }

    /// Adds the current contents of `frame` (copied, so the buffer can be reused).
    func add(_ frame: MTLBuffer, layout: FrameLayout) {
        let data = Data(bytes: frame.contents(), count: layout.width * layout.height * 4)
        guard let provider = CGDataProvider(data: data as CFData),
              let image = CGImage(width: layout.width, height: layout.height, bitsPerComponent: 8, bitsPerPixel: 32,
                                  bytesPerRow: layout.width * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                  bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                                  provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)
        else { return }
        let props = [kCGImagePropertyGIFDictionary as String: [kCGImagePropertyGIFDelayTime as String: delay]]
        CGImageDestinationAddImage(destination, image, props as CFDictionary)
    }

    func finish() throws {
        guard CGImageDestinationFinalize(destination) else { throw SimulationError.gpu("could not write the GIF") }
    }
}
