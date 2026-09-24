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

/// Carbon oxidation-state labels: the reactant values while the old bonds
/// hold, the product values once the new bonds form, and nothing in between.
func carbonLabelAlpha(progress: Float) -> (before: Float, after: Float) {
    let before: Float = 1 - smoothstep01(progress / breakEnd)
    let after: Float = smoothstep01((progress - formStart) / (1 - formStart))
    return (before, after)
}

/// The caption's middle line.
func accountingLine(_ reaction: Reaction, progress: Float) -> String {
    let before = oxidationStates(elements: reaction.elements, bonds: reaction.reactantBonds)
    let after = oxidationStates(elements: reaction.elements, bonds: reaction.productBonds)
    if progress <= 0 {
        let labels = reaction.carbons.map { signedLabel(before[$0]) }.joined(separator: ", ")
        return "Carbon oxidation states: \(labels) (average 0) · oxygen in O₂: 0"
    }
    if progress < 1 {
        return "Bonds break and re-form; 24 electrons move from carbon to oxygen"
    }
    let carbon = signedLabel(after[reaction.carbons[0]])
    let oxygen = signedLabel(after[reaction.o2Oxygens[0]])
    return "Carbon now \(carbon) in every CO₂ · oxygen now \(oxygen) · 24 electrons moved · ≈2,880 kJ/mol released"
}

let captionLegend = "Numbers = carbon oxidation state · gold/cyan glow = electrons leaving/arriving · overall accounting, not the ~30 enzyme steps · not to time scale"

/// Draws the carbon labels and the caption bar.
func drawOverlay(reaction: Reaction, state: MoleculeState, camera: Camera, stage: String, progress: Float,
                 into frame: MTLBuffer, layout: FrameLayout) {
    guard let ctx = CGContext(data: frame.contents(), width: layout.width, height: layout.height,
                              bitsPerComponent: 8, bytesPerRow: layout.width * 4,
                              space: CGColorSpace(name: CGColorSpace.sRGB)!,
                              bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return }
    let h = CGFloat(layout.height)
    let k = CGFloat(layout.width) / 640   // laid out for a 640-pixel-wide frame, then scaled

    // Oxidation-state numbers next to each carbon.
    let (beforeAlpha, afterAlpha) = carbonLabelAlpha(progress: progress)
    let before = oxidationStates(elements: reaction.elements, bonds: reaction.reactantBonds)
    let after = oxidationStates(elements: reaction.elements, bonds: reaction.productBonds)
    let middle = camera.project(.zero, width: layout.width, height: layout.viewHeight)
    for c in reaction.carbons {
        let alpha: Float = max(beforeAlpha, afterAlpha)
        if alpha < 0.02 { continue }
        let label = beforeAlpha >= afterAlpha ? signedLabel(before[c]) : signedLabel(after[c])
        let p = camera.project(state.atoms[c].position, width: layout.width, height: layout.viewHeight)
        // Put the label on the side facing away from the middle of the scene,
        // which is clear of bonds for both glucose and the CO₂ ring.
        var away = p - middle
        if simd_length(away) < 1 { away = SIMD2(1, -1) }
        let spot = p + simd_normalize(away) * Float(24 * k)
        let x = CGFloat(spot.x) - 9 * k
        let top = CGFloat(spot.y) - 9 * k
        if x < 0 || x > CGFloat(layout.width) - 30 * k || top < 0 || top > CGFloat(layout.viewHeight) - 20 * k { continue }
        ctx.saveGState()
        // A soft dark shadow keeps labels readable on the gradient.
        ctx.setShadow(offset: .zero, blur: 5 * k, color: CGColor(srgbRed: 0, green: 0.05, blue: 0.12, alpha: 0.9 * CGFloat(alpha)))
        text(label, ctx, x: x, top: top, size: 13 * k, bold: true, color: cg(SIMD3(1.0, 0.86, 0.45), alpha: alpha), height: h)
        ctx.restoreGState()
    }

    // Caption bar.
    let barTop = CGFloat(layout.viewHeight)
    ctx.setFillColor(captionBackground)
    ctx.fill(CGRect(x: 0, y: 0, width: CGFloat(layout.width), height: CGFloat(layout.captionHeight)))
    text(stage, ctx, x: 18 * k, top: barTop + 12 * k, size: 17 * k, bold: true, height: h)
    text(accountingLine(reaction, progress: progress), ctx, x: 18 * k, top: barTop + 38 * k, size: 12 * k,
         color: muted, height: h)
    text(captionLegend, ctx, x: 18 * k, top: barTop + 58 * k, size: 9.5 * k, color: muted, height: h)
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
