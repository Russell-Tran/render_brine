// Getting a frame out: a PNG for looking at one moment, and the palette the
// GIF encoder uses.

import CoreGraphics
import Foundation
import ImageIO
import Metal
import simd

func writePNG(_ buffer: MTLBuffer, layout: FrameLayout, to path: String) {
    guard let ctx = CGContext(data: buffer.contents(), width: layout.width,
                              height: layout.height, bitsPerComponent: 8,
                              bytesPerRow: layout.width * 4,
                              space: CGColorSpace(name: CGColorSpace.sRGB)!,
                              bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue),
          let image = ctx.makeImage() else { return }
    try? FileManager.default.createDirectory(atPath: "renders", withIntermediateDirectories: true)
    let url = URL(fileURLWithPath: path) as CFURL
    guard let dest = CGImageDestinationCreateWithURL(url, "public.png" as CFString, 1, nil) else {
        return
    }
    CGImageDestinationAddImage(dest, image, nil)
    CGImageDestinationFinalize(dest)
}

private func rgb8(_ linear: SIMD3<Float>, _ scale: Float) -> RGB {
    let v: SIMD3<Float> = simd_clamp(linear * scale, SIMD3<Float>(repeating: 0),
                                     SIMD3<Float>(repeating: 1))
    let r: Float = gammaEncode(v.x) * 255
    let g: Float = gammaEncode(v.y) * 255
    let b: Float = gammaEncode(v.z) * 255
    return RGB(UInt8(r.rounded()), UInt8(g.rounded()), UInt8(b.rounded()))
}

/// The palette keeps the colours that carry the meaning whatever median cut
/// would otherwise choose. Median cut hands colours to whatever covers the most
/// pixels, and the things that matter here — the sodium beam, the two sugars,
/// the three catalytic side chains — are small and on screen for part of the
/// loop only. Step 8 lost phosphorus this way and step 9 nearly lost its green.
func invertasePalette(samples: [RGB], count: Int) -> [RGB] {
    let keepers: [SIMD3<Float>] = [
        spectralColour(nanometres: Float(sodiumDLineNanometres)).linear,
        glucoseColour, fructoseColour, waterColour,
        nucleophileColour, acidBaseColour, stabiliserColour,
    ]
    var reserved: [RGB] = []
    for colour in keepers {
        for scale: Float in [0.30, 0.55, 0.80, 1.0] {
            reserved.append(rgb8(colour, scale))
        }
    }
    // Drop duplicates, keeping the first of each.
    var seen = Set<UInt32>()
    var unique: [RGB] = []
    for c in reserved {
        let key: UInt32 = UInt32(c.x) << 16 | UInt32(c.y) << 8 | UInt32(c.z)
        if seen.insert(key).inserted { unique.append(c) }
    }
    let room: Int = max(count - unique.count, 8)
    return unique + medianCutPalette(samples, count: room)
}
