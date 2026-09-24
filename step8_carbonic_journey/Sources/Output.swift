// Labels and the caption bar, drawn by the CPU straight into the frame the GPU
// just rendered (unified memory), and the GIF writer.

import CoreGraphics
import CoreText
import Foundation
import ImageIO
import Metal
import UniformTypeIdentifiers
import simd

struct FrameLayout {
    let width: Int
    let viewHeight: Int      // the ray-traced molecule view
    let captionHeight: Int   // the text bar under it
    var height: Int { viewHeight + captionHeight }
}

private let ink = CGColor(srgbRed: 0.92, green: 0.95, blue: 0.96, alpha: 1)
private let muted = CGColor(srgbRed: 0.60, green: 0.69, blue: 0.73, alpha: 1)
private let captionBackground = CGColor(srgbRed: 0.05, green: 0.08, blue: 0.11, alpha: 1)
private let nameColor = SIMD3<Float>(0.95, 0.97, 1.0)

private func cg(_ c: SIMD3<Float>, alpha: Float = 1) -> CGColor {
    CGColor(srgbRed: CGFloat(c.x), green: CGFloat(c.y), blue: CGFloat(c.z), alpha: CGFloat(alpha))
}

/// Draws text whose top-left corner is at (x, top), in top-down pixel coordinates.
private func text(_ s: String, _ ctx: CGContext, x: CGFloat, top: CGFloat, size: CGFloat, bold: Bool = false,
                  color: CGColor = ink, height: CGFloat) {
    let font = CTFontCreateWithName((bold ? "HelveticaNeue-Bold" : "HelveticaNeue") as CFString, size, nil)
    let attributes: [NSAttributedString.Key: Any] = [
        NSAttributedString.Key(kCTFontAttributeName as String): font,
        NSAttributedString.Key(kCTForegroundColorAttributeName as String): color,
    ]
    let line = CTLineCreateWithAttributedString(NSAttributedString(string: s, attributes: attributes))
    ctx.textPosition = CGPoint(x: x, y: height - top - size)
    CTLineDraw(line, ctx)
}

func angstroms(_ x: Float) -> String { String(format: "%.2f Å", x) }

/// The carbon's three C–O distances: the two that end up equal (O1, O3), then the OH (O2).
func bondReadout(_ state: MoleculeState) -> String {
    let a = distance(state, Slot.carbon, Slot.o1)
    let b = distance(state, Slot.carbon, Slot.o3)
    let c = distance(state, Slot.carbon, Slot.o2)
    return "C–O bonds:  \(angstroms(a))   \(angstroms(b))   \(angstroms(c)) (to the OH)"
}

/// The caption's middle line: a live readout and the real timescale of each step.
func detailLine(_ state: MoleculeState, _ timeline: Timeline, at t: Double) -> String {
    if t < timeline.act2Start {
        return "Only about 0.17% of dissolved CO₂ is carbonic acid at any moment"
    }
    if t < timeline.act3Start {
        let bend = angle(state, Slot.o1, Slot.carbon, Slot.o3)
        return String(format: "O–C–O angle %.0f° (180° → 125°) · real time: ~25 s per CO₂ without an enzyme (rate 0.039 s⁻¹)", bend)
    }
    if t < timeline.act4Start {
        return bondReadout(state) + " · the new OH turns to the most stable (cis-cis) shape"
    }
    if t < timeline.act5Start {
        return bondReadout(state) + " · real time: tens of nanoseconds"
    }
    return "In water this repeats, one CO₂ after another, all the time"
}

let captionLegend = "Ball-and-stick · gold = + charge, cyan = − charge · motion drawn between known shapes, not simulated · enormously slowed"

/// A label near atom `atom`, pushed away from `center` so it doesn't sit on bonds.
private func label(_ s: String, atom p3: SIMD3<Float>, center c3: SIMD3<Float>, color: SIMD3<Float>, alpha: Float,
                   size: CGFloat, camera: Camera, layout: FrameLayout, ctx: CGContext) {
    if alpha < 0.02 { return }
    let k = CGFloat(layout.width) / 640
    let h = CGFloat(layout.height)
    let p = camera.project(p3, width: layout.width, height: layout.viewHeight)
    let c = camera.project(c3, width: layout.width, height: layout.viewHeight)
    var away = p - c
    if simd_length(away) < 1 { away = SIMD2(1, -1) }
    let dir = simd_normalize(away)
    let spot = p + dir * Float(30 * k)
    // Labels pointing left end at the spot instead of starting there.
    let width = CGFloat(s.count) * size * k * 0.55
    let x = dir.x < -0.3 ? CGFloat(spot.x) - width : CGFloat(spot.x) - 10 * k
    let top = CGFloat(spot.y) - 10 * k
    if x < 0 || x + width > CGFloat(layout.width) || top < 0 || top > CGFloat(layout.viewHeight) - 20 * k { return }
    ctx.saveGState()
    // A soft dark shadow keeps labels readable on the gradient.
    ctx.setShadow(offset: .zero, blur: 5 * k, color: CGColor(srgbRed: 0, green: 0.05, blue: 0.12, alpha: 0.9 * CGFloat(alpha)))
    text(s, ctx, x: x, top: top, size: size * k, bold: true, color: cg(color, alpha: alpha), height: h)
    ctx.restoreGState()
}

/// Draws the molecule and charge labels and the caption bar.
func drawOverlay(state: MoleculeState, timeline: Timeline, at t: Double, camera: Camera,
                 into frame: MTLBuffer, layout: FrameLayout) {
    guard let ctx = CGContext(data: frame.contents(), width: layout.width, height: layout.height,
                              bitsPerComponent: 8, bytesPerRow: layout.width * 4,
                              space: CGColorSpace(name: CGColorSpace.sRGB)!,
                              bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return }
    let h = CGFloat(layout.height)
    let k = CGFloat(layout.width) / 640
    let a = state.atoms

    // Molecule names while the reactants gather; they fade as the reaction starts.
    let s = timeline.fraction(t, from: timeline.act2Start, length: timeline.act2)
    let names: Float = t < timeline.act3Start ? 1 - smoothstep01(s * 2.5) : 0
    let groups = a.count / Slot.count
    for g in 0..<groups {
        let base = g * Slot.count
        let isNext = g > 0
        let alpha: Float = isNext ? entryAmount(timeline, t) : names
        let p1: SIMD3<Float> = a[base + Slot.carbon].position
        let p2: SIMD3<Float> = a[base + Slot.o2].position
        let p3: SIMD3<Float> = a[base + Slot.helperO].position
        let middle: SIMD3<Float> = (p1 + p2 + p3) / 3
        label("CO₂", atom: a[base + Slot.o1].position, center: middle, color: nameColor, alpha: alpha, size: 13,
              camera: camera, layout: layout, ctx: ctx)
        label("H₂O", atom: a[base + Slot.o2].position, center: middle, color: nameColor, alpha: alpha, size: 13,
              camera: camera, layout: layout, ctx: ctx)
        label("helper H₂O", atom: a[base + Slot.helperO].position, center: middle, color: nameColor, alpha: alpha,
              size: 13, camera: camera, layout: layout, ctx: ctx)
    }

    // Charge labels on the current group's products.
    let transfer = -a[Slot.o1].charge * 2
    let carbon = a[Slot.carbon].position
    label("−½", atom: a[Slot.o1].position, center: carbon, color: negativeGlow, alpha: transfer, size: 18,
          camera: camera, layout: layout, ctx: ctx)
    label("−½", atom: a[Slot.o3].position, center: carbon, color: negativeGlow, alpha: transfer, size: 18,
          camera: camera, layout: layout, ctx: ctx)
    label("H₃O⁺", atom: a[Slot.helperO].position, center: a[Slot.o3].position, color: positiveGlow, alpha: transfer,
          size: 18, camera: camera, layout: layout, ctx: ctx)

    // Caption bar.
    let barTop = CGFloat(layout.viewHeight)
    ctx.setFillColor(captionBackground)
    ctx.fill(CGRect(x: 0, y: 0, width: CGFloat(layout.width), height: CGFloat(layout.captionHeight)))
    text(timeline.stage(at: t), ctx, x: 18 * k, top: barTop + 12 * k, size: 17 * k, bold: true, height: h)
    text(detailLine(state, timeline, at: t), ctx, x: 18 * k, top: barTop + 38 * k, size: 11.5 * k, color: muted, height: h)
    text(captionLegend, ctx, x: 18 * k, top: barTop + 58 * k, size: 10 * k, color: muted, height: h)
}

/// Collects frames into an animated GIF that loops forever.
final class GIFWriter {
    private let destination: CGImageDestination
    private let delay: Double

    init(url: URL, frameCount: Int, delay: Double) throws {
        try? FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        guard let d = CGImageDestinationCreateWithURL(url as CFURL, UTType.gif.identifier as CFString, frameCount, nil) else {
            throw RenderError.gpu("could not create \(url.path)")
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
        guard CGImageDestinationFinalize(destination) else { throw RenderError.gpu("could not write the GIF") }
    }
}
