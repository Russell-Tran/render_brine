// Putting a frame together on the CPU: the plain ground behind, the plant over
// it, and the few words drawn on top; then a whole loop into one GIF with step
// 8's encoder.
//
// This file is the same in steps 45 and 46.

import CoreGraphics
import CoreText
import Foundation
import ImageIO
import Metal
import UniformTypeIdentifiers
import simd

/// A line of text, or a label with a thin leader line to what it names.
/// Positions are pixels from the top left.
struct Caption {
    var text: String
    var x: Float
    var baseline: Float
    var size: Float
    var italic: Bool = false
    var alpha: Float = 1
    var centred: Bool = false
    /// Draw a leader from just beside the text to this point.
    var pointAt: SIMD2<Float>? = nil
}

/// The ground, in display values: plain, a faint gradient top to bottom. It is
/// fixed to the frame, not the world, so it cannot give the loop away.
struct Ground {
    var top: V3
    var bottom: V3
}

private func ctLine(_ s: String, size: CGFloat, font: String, color: CGColor) -> CTLine {
    let f = CTFontCreateWithName(font as CFString, size, nil)
    let attributes: [NSAttributedString.Key: Any] = [
        NSAttributedString.Key(kCTFontAttributeName as String): f,
        NSAttributedString.Key(kCTForegroundColorAttributeName as String): color,
    ]
    return CTLineCreateWithAttributedString(NSAttributedString(string: s, attributes: attributes))
}

/// The finished frame as sRGB bytes (RGBA, alpha 255), captions drawn.
func finishFrame(_ f: Frame, ground: Ground, captions: [Caption]) -> [UInt8] {
    let w: Int = f.width
    let h: Int = f.height
    var bytes = [UInt8](repeating: 255, count: w * h * 4)
    for y in 0..<h {
        let g: V3 = ground.top + (ground.bottom - ground.top) * (Float(y) / Float(max(h - 1, 1)))
        for x in 0..<w {
            let i: Int = y * w + x
            let px: SIMD4<Float> = f.rgba[i]
            let c: V3 = V3(px.x, px.y, px.z) + g * (1 - px.w)
            let cl: V3 = simd_clamp(c, V3(0, 0, 0), V3(1, 1, 1))
            bytes[i * 4] = UInt8((cl.x * 255).rounded())
            bytes[i * 4 + 1] = UInt8((cl.y * 255).rounded())
            bytes[i * 4 + 2] = UInt8((cl.z * 255).rounded())
        }
    }
    let hh: CGFloat = CGFloat(h)
    bytes.withUnsafeMutableBytes { raw in
        guard let ctx = CGContext(data: raw.baseAddress, width: w, height: h, bitsPerComponent: 8,
                                  bytesPerRow: w * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return }
        ctx.setLineCap(.round)
        for c in captions {
            let ink = CGColor(srgbRed: 0.16, green: 0.20, blue: 0.18, alpha: CGFloat(0.9 * c.alpha))
            let font: String = c.italic ? "HelveticaNeue-Italic" : "HelveticaNeue"
            let line: CTLine = ctLine(c.text, size: CGFloat(c.size), font: font, color: ink)
            let width: CGFloat = CGFloat(CTLineGetTypographicBounds(line, nil, nil, nil))
            let x0: CGFloat = CGFloat(c.x) - (c.centred ? width / 2 : 0)
            if let target = c.pointAt {
                let faint = CGColor(srgbRed: 0.16, green: 0.20, blue: 0.18, alpha: CGFloat(0.55 * c.alpha))
                ctx.setStrokeColor(faint)
                ctx.setLineWidth(1.0)
                // From the end of the text nearer the target.
                let startX: CGFloat = CGFloat(target.x) < x0 ? x0 - 3 : x0 + width + 3
                let startY: CGFloat = hh - CGFloat(c.baseline) + CGFloat(c.size) * 0.3
                ctx.move(to: CGPoint(x: startX, y: startY))
                ctx.addLine(to: CGPoint(x: CGFloat(target.x), y: hh - CGFloat(target.y)))
                ctx.strokePath()
            }
            ctx.textPosition = CGPoint(x: x0, y: hh - CGFloat(c.baseline))
            CTLineDraw(line, ctx)
        }
    }
    return bytes
}

func savePNG(_ bytes: [UInt8], width: Int, height: Int, to url: URL) throws {
    try? FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    let data = Data(bytes)
    guard let provider = CGDataProvider(data: data as CFData),
          let space = CGColorSpace(name: CGColorSpace.sRGB),
          let cg = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
                           bytesPerRow: width * 4, space: space,
                           bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                           provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent),
          let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)
    else { throw RenderError.png("could not set up the encoder for \(url.path)") }
    CGImageDestinationAddImage(dest, cg, nil)
    guard CGImageDestinationFinalize(dest) else { throw RenderError.png("could not write \(url.path)") }
}

/// A whole loop into one GIF. One palette for all of it, drawn from frames
/// spread across the loop so no stretch of it is posterised; then every frame
/// quantised on the GPU and handed to step 8's writer, which stores only the
/// box of pixels that changed — which here, with the camera rising, is the
/// whole frame every time. `frame(i)` returns frame i's finished bytes.
func writeLoopGIF(path: String, width: Int, height: Int, frames: Int, delayCentiseconds: Int,
                  paletteFrames: [Int], device: MTLDevice, frame: (Int) throws -> [UInt8]) throws -> Int {
    let pixelCount: Int = width * height
    guard let work = device.makeBuffer(length: pixelCount * 4, options: .storageModeShared) else {
        throw RenderError.gpu("could not allocate the frame buffer")
    }
    var cache: [Int: [UInt8]] = [:]
    var sampled: [RGB] = []
    for f in paletteFrames {
        let b: [UInt8] = try frame(f)
        cache[f] = b
        b.withUnsafeBytes { work.contents().copyMemory(from: $0.baseAddress!, byteCount: pixelCount * 4) }
        sampled += samplePixels(work, pixels: pixelCount, step: 5)
    }
    let palette: [RGB] = medianCutPalette(sampled, count: 250)
    let quantizer = try Quantizer(device: device, palette: palette, pixels: pixelCount)
    let gif = GIFWriter(url: URL(fileURLWithPath: path), width: width, height: height,
                        palette: palette, delayCentiseconds: delayCentiseconds)
    for f in 0..<frames {
        let b: [UInt8] = try cache[f] ?? frame(f)
        cache[f] = nil
        b.withUnsafeBytes { work.contents().copyMemory(from: $0.baseAddress!, byteCount: pixelCount * 4) }
        gif.add(try quantizer.indices(of: work))
        if f % 20 == 0 { print("  frame \(f)/\(frames)") }
    }
    try gif.finish()
    return ((try? FileManager.default.attributesOfItem(atPath: path))?[.size] as? Int) ?? 0
}
