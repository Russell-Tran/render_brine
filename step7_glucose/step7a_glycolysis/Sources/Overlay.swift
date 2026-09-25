// What the CPU draws on top of the GPU's picture, straight into the same
// memory: ATP/ADP/NAD⁺ tokens, glucose's carbon numbers, the caption and the
// ATP ledger. Plus the GIF writer.

import CoreGraphics
import CoreText
import Foundation
import ImageIO
import Metal
import UniformTypeIdentifiers
import simd

struct FrameLayout {
    let width: Int
    let viewHeight: Int      // the ray-traced view
    let captionHeight: Int   // the text bar under it
    var height: Int { viewHeight + captionHeight }
    var scale: CGFloat { CGFloat(width) / 640 }   // laid out for a 640-wide frame
}

private let ink = CGColor(srgbRed: 0.92, green: 0.95, blue: 0.96, alpha: 1)
private let muted = CGColor(srgbRed: 0.60, green: 0.69, blue: 0.73, alpha: 1)
private let accent = CGColor(srgbRed: 0.45, green: 0.83, blue: 0.86, alpha: 1)
private let gold = CGColor(srgbRed: 1.0, green: 0.80, blue: 0.35, alpha: 1)
private let captionBackground = CGColor(srgbRed: 0.05, green: 0.08, blue: 0.11, alpha: 1)

/// Draws text whose top-left corner is at (x, top) in top-down pixels. Returns its width.
@discardableResult
private func text(_ s: String, _ ctx: CGContext, x: CGFloat, top: CGFloat, size: CGFloat, bold: Bool = false,
                  color: CGColor = ink, height: CGFloat) -> CGFloat {
    let font = CTFontCreateWithName((bold ? "HelveticaNeue-Bold" : "HelveticaNeue") as CFString, size, nil)
    let attributes: [NSAttributedString.Key: Any] = [
        NSAttributedString.Key(kCTFontAttributeName as String): font,
        NSAttributedString.Key(kCTForegroundColorAttributeName as String): color,
    ]
    let line = CTLineCreateWithAttributedString(NSAttributedString(string: s, attributes: attributes))
    ctx.textPosition = CGPoint(x: x, y: height - top - size)
    CTLineDraw(line, ctx)
    return CGFloat(CTLineGetTypographicBounds(line, nil, nil, nil))
}

private func textWidth(_ s: String, size: CGFloat, bold: Bool) -> CGFloat {
    let font = CTFontCreateWithName((bold ? "HelveticaNeue-Bold" : "HelveticaNeue") as CFString, size, nil)
    let attributes: [NSAttributedString.Key: Any] = [NSAttributedString.Key(kCTFontAttributeName as String): font]
    let line = CTLineCreateWithAttributedString(NSAttributedString(string: s, attributes: attributes))
    return CGFloat(CTLineGetTypographicBounds(line, nil, nil, nil))
}

/// "net +2", "net −2", "net 0"
func netLabel(spent: Int, made: Int) -> String {
    let net = made - spent
    if net > 0 { return "net +\(net)" }
    if net < 0 { return "net −\(-net)" }
    return "net 0"
}

/// Carbon labels from glucose's numbering: "C3" and "n_C3" both → "3".
func carbonNumber(_ label: String) -> String? {
    let bare = label.hasPrefix("n_") ? String(label.dropFirst(2)) : label
    guard bare.count == 2, bare.first == "C", let d = bare.last, d.isNumber else { return nil }
    return String(d)
}

func drawOverlay(_ frame: Frame, camera: Camera, into buffer: MTLBuffer, layout: FrameLayout) {
    guard let ctx = CGContext(data: buffer.contents(), width: layout.width, height: layout.height,
                              bitsPerComponent: 8, bytesPerRow: layout.width * 4,
                              space: CGColorSpace(name: CGColorSpace.sRGB)!,
                              bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return }
    let k = layout.scale
    let h = CGFloat(layout.height)
    let viewH = CGFloat(layout.viewHeight)
    let shadow = CGColor(srgbRed: 0, green: 0.05, blue: 0.12, alpha: 0.9)

    // Carbon numbers, just off each carbon.
    ctx.saveGState()
    ctx.setShadow(offset: .zero, blur: 4 * k, color: shadow)
    for atom in frame.atoms where atom.key.visible > 0.6 {
        guard let n = carbonNumber(atom.label) else { continue }
        let p = camera.project(atom.key.position, width: layout.width, height: layout.viewHeight)
        let x = CGFloat(p.x) + 9 * k, top = CGFloat(p.y) - 20 * k
        if x > 0 && x < CGFloat(layout.width) - 20 * k && top > 0 && top < viewH - 16 * k {
            text(n, ctx, x: x, top: top, size: 11 * k, bold: true, color: ink, height: h)
        }
    }
    ctx.restoreGState()

    // Tokens: ATP / ADP / NAD⁺ / NADH as labeled capsules.
    for token in frame.tokens where token.alpha > 0.02 {
        let p = camera.project(token.position, width: layout.width, height: layout.viewHeight)
        let size: CGFloat = 13 * k
        let w = textWidth(token.text, size: size, bold: true) + 18 * k
        let capH: CGFloat = 24 * k
        let cx = CGFloat(p.x), cy = CGFloat(p.y)
        let rect = CGRect(x: cx - w / 2, y: h - cy - capH / 2, width: w, height: capH)
        let a = CGFloat(token.alpha)
        let isATP = token.text == "ATP"
        let isNAD = token.text.hasPrefix("NAD")
        let fill: CGColor = isATP ? CGColor(srgbRed: 0.98, green: 0.72, blue: 0.25, alpha: 0.95 * a)
            : isNAD ? CGColor(srgbRed: 0.35, green: 0.78, blue: 0.82, alpha: 0.95 * a)
            : CGColor(srgbRed: 0.80, green: 0.82, blue: 0.85, alpha: 0.95 * a)
        ctx.saveGState()
        ctx.setShadow(offset: CGSize(width: 0, height: -2 * k), blur: 6 * k,
                      color: CGColor(srgbRed: 0, green: 0.05, blue: 0.12, alpha: 0.5 * a))
        ctx.addPath(CGPath(roundedRect: rect, cornerWidth: capH / 2, cornerHeight: capH / 2, transform: nil))
        ctx.setFillColor(fill)
        ctx.fillPath()
        ctx.restoreGState()
        text(token.text, ctx, x: cx - w / 2 + 9 * k, top: cy - size / 2 - 2 * k, size: size, bold: true,
             color: CGColor(srgbRed: 0.06, green: 0.10, blue: 0.14, alpha: a), height: h)
    }

    // Caption bar.
    let barTop = viewH
    ctx.setFillColor(captionBackground)
    ctx.fill(CGRect(x: 0, y: 0, width: CGFloat(layout.width), height: CGFloat(layout.captionHeight)))
    text(frame.title, ctx, x: 18 * k, top: barTop + 10 * k, size: 16 * k, bold: true, height: h)
    if !frame.enzyme.isEmpty {
        text(frame.enzyme, ctx, x: 18 * k, top: barTop + 34 * k, size: 12 * k, color: accent, height: h)
    }
    text(frame.equation, ctx, x: 18 * k, top: barTop + 54 * k, size: 11.5 * k, color: muted, height: h)

    // ATP ledger, right-aligned.
    let spent = frame.ledger.count > 0 ? frame.ledger[0] : 0
    let made = frame.ledger.count > 1 ? frame.ledger[1] : 0
    let nadh = frame.ledger.count > 2 ? frame.ledger[2] : 0
    let right = CGFloat(layout.width) - 18 * k
    let line1 = "ATP  spent \(spent) · made \(made)"
    let net = netLabel(spent: spent, made: made)
    let line2 = "NADH  \(nadh)"
    let w1 = textWidth(line1, size: 12 * k, bold: false)
    let wNet = textWidth(net, size: 16 * k, bold: true)
    text(net, ctx, x: right - wNet, top: barTop + 10 * k, size: 16 * k, bold: true, color: gold, height: h)
    text(line1, ctx, x: right - w1, top: barTop + 34 * k, size: 12 * k, color: ink, height: h)
    text(line2, ctx, x: right - textWidth(line2, size: 11.5 * k, bold: false), top: barTop + 54 * k,
         size: 11.5 * k, color: muted, height: h)
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
