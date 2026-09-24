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

let ballAndStickLegend = "Stick thickness = bond order · gold = + charge, cyan = − charge · not to time scale"

/// "1.22 Å"
func angstroms(_ x: Float) -> String { String(format: "%.2f Å", x) }

/// What the caption says about the three carbon–oxygen bonds: the two that
/// become equal first (C–O1, C–O3), then the one to the OH that stays (C–O2).
func bondReadout(_ state: MoleculeState) -> String {
    let a = distance(state, 0, 1), b = distance(state, 0, 3), c = distance(state, 0, 2)
    return "Carbon–oxygen bonds:  \(angstroms(a))   \(angstroms(b))   \(angstroms(c)) (to the OH)"
}

/// Draws the charge labels next to atoms and the caption bar.
func drawOverlay(state: MoleculeState, camera: Camera, stage: String, legend: String, progress: Float,
                 into frame: MTLBuffer, layout: FrameLayout) {
    guard let ctx = CGContext(data: frame.contents(), width: layout.width, height: layout.height,
                              bitsPerComponent: 8, bytesPerRow: layout.width * 4,
                              space: CGColorSpace(name: CGColorSpace.sRGB)!,
                              bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return }
    let h = CGFloat(layout.height)
    let s = smoothstep01(progress)
    let k = CGFloat(layout.width) / 640   // everything was laid out for a 640-pixel-wide frame

    // Charge labels, fading in as the proton leaves.
    if s > 0.02 {
        let labels: [(atom: Int, text: String, color: SIMD3<Float>)] = [
            (5, "H⁺", positiveGlow), (1, "−½", negativeGlow), (3, "−½", negativeGlow),
        ]
        let carbon = camera.project(state.atoms[0].position, width: layout.width, height: layout.viewHeight)
        for label in labels {
            let p = camera.project(state.atoms[label.atom].position, width: layout.width, height: layout.viewHeight)
            // Put the label just beyond the atom, on the side away from carbon,
            // so it never sits on a bond.
            let away = simd_normalize(p - carbon)
            let spot = p + away * Float(34 * k)
            let x = CGFloat(spot.x) - 11 * k
            let top = CGFloat(spot.y) - 11 * k
            if x > 0 && x < CGFloat(layout.width) - 30 * k && top > 0 && top < CGFloat(layout.viewHeight) - 20 * k {
                // A soft dark shadow keeps the label readable on the light blue gradient.
                ctx.saveGState()
                ctx.setShadow(offset: .zero, blur: 5 * k, color: CGColor(srgbRed: 0, green: 0.05, blue: 0.12, alpha: 0.9 * CGFloat(s)))
                text(label.text, ctx, x: x, top: top, size: 18 * k, bold: true, color: cg(label.color, alpha: s), height: h)
                ctx.restoreGState()
            }
        }
    }

    // Caption bar.
    let barTop = CGFloat(layout.viewHeight)
    ctx.setFillColor(captionBackground)
    ctx.fill(CGRect(x: 0, y: 0, width: CGFloat(layout.width), height: CGFloat(layout.captionHeight)))
    text(stage, ctx, x: 18 * k, top: barTop + 12 * k, size: 17 * k, bold: true, height: h)
    text(bondReadout(state), ctx, x: 18 * k, top: barTop + 38 * k, size: 12.5 * k, color: muted, height: h)
    text(legend, ctx, x: 18 * k, top: barTop + 58 * k, size: 11 * k, color: muted, height: h)
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
