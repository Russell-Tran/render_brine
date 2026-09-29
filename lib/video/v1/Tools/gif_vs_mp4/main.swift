// Prediction P1 (experiments/world/PREREGISTRATION.md): "At the same frames and
// resolution, the MP4 is ≥ 10× smaller than a GIF."
//
// Decodes a showcase GIF with ImageIO (every frame, composited, as RGBA8),
// writes those exact frames to MP4 with MP4Writer at the GIF's own size and
// frame rate, reads the MP4 back and measures PSNR against the GIF's frames.
// Done at several bit rates, starting from the film's own setting scaled to
// this size and rate (same bits per pixel per frame), so the size ratio is
// always reported next to the quality it buys.
//
//   .build/gif_vs_mp4 ../../../showcase/ant_trail_wood.gif

import Foundation
import ImageIO
import CoreGraphics

var gifPath: String = "../../../showcase/ant_trail_wood.gif"
if CommandLine.arguments.count > 1 { gifPath = CommandLine.arguments[1] }
let gifURL: URL = URL(fileURLWithPath: gifPath)

guard let source = CGImageSourceCreateWithURL(gifURL as CFURL, nil) else {
    print("cannot open \(gifPath)"); exit(1)
}
let frameCount: Int = CGImageSourceGetCount(source)
var delays: [Double] = []
var frames: [[UInt8]] = []
var width: Int = 0
var height: Int = 0
let space: CGColorSpace = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
for i in 0..<frameCount {
    guard let image = CGImageSourceCreateImageAtIndex(source, i, nil) else { print("frame \(i) unreadable"); exit(1) }
    width = image.width; height = image.height
    var px = [UInt8](repeating: 0, count: width * height * 4)
    let drew: Bool = px.withUnsafeMutableBytes { (raw: UnsafeMutableRawBufferPointer) -> Bool in
        guard let ctx = CGContext(data: raw.baseAddress, width: width, height: height, bitsPerComponent: 8,
                                  bytesPerRow: width * 4, space: space,
                                  bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { return false }
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        return true
    }
    if !drew { print("cannot draw frame \(i)"); exit(1) }
    for k in stride(from: 3, to: px.count, by: 4) { px[k] = 255 }
    frames.append(px)
    let props = CGImageSourceCopyPropertiesAtIndex(source, i, nil) as? [String: Any]
    let gif = props?[kCGImagePropertyGIFDictionary as String] as? [String: Any]
    let d: Double = (gif?[kCGImagePropertyGIFUnclampedDelayTime as String] as? NSNumber)?.doubleValue
        ?? (gif?[kCGImagePropertyGIFDelayTime as String] as? NSNumber)?.doubleValue ?? 0.1
    delays.append(d)
}
let gifBytes: Int = ((try? FileManager.default.attributesOfItem(atPath: gifPath))?[.size] as? NSNumber)?.intValue ?? 0
let delay: Double = delays.first ?? 0.1
let allSame: Bool = delays.allSatisfy { abs($0 - delay) < 1e-6 }
let fpsDouble: Double = 1 / delay
let fps: Int32 = Int32(fpsDouble.rounded())
let seconds: Double = Double(frameCount) * delay
// Are consecutive frames really different (ImageIO composites; a frame that is
// only a GIF sub-rectangle would show as a mostly-black frame here)?
var blackFrames: Int = 0
for f in frames {
    var dark: Int = 0
    for k in stride(from: 0, to: f.count, by: 4 * 97) where f[k] < 8 && f[k + 1] < 8 && f[k + 2] < 8 { dark += 1 }
    let darkPixels: Int = dark * 97
    let allPixels: Int = width * height
    if darkPixels * 2 > allPixels { blackFrames += 1 }
}
print(String(format: "GIF %@: %d frames, %d × %d, delay %.3f s (%@) → %d fps, %.2f s, %d bytes (%.2f MB); frames >50%% black: %d",
             gifURL.lastPathComponent, frameCount, width, height, delay, allSame ? "all frames" : "NOT all frames",
             fps, seconds, gifBytes, Double(gifBytes) / 1e6, blackFrames))
if !allSame || abs(fpsDouble - Double(fps)) > 1e-6 { print("warning: the GIF's timing is not a whole constant frame rate") }

// The film's setting (MP4Settings' defaults) at 1920×1080×30 → bits per pixel per frame.
let film = MP4Settings()
let filmPixels: Int = film.width * film.height
let filmPixelRate: Double = Double(filmPixels) * Double(film.fps)
let filmBitsPerPixel: Double = Double(film.averageBitRate) / filmPixelRate
let gifPixels: Int = width * height
let pixelRate: Double = Double(gifPixels) * Double(fps)
print(String(format: "film setting: %d bit/s at %d×%d×%d = %.4f bits per pixel per frame",
             film.averageBitRate, film.width, film.height, film.fps, filmBitsPerPixel))
print("scale  bit/s(target)  MP4 bytes  GIF/MP4  PSNR mean  PSNR min  maxErr  encode s")

let hw = probeH264Hardware(width: width, height: height)
print("hardware H.264 at \(width)×\(height): session \(hw.sessionCreated), using hardware \(hw.sessionUsesHardware)")

for scale in [0.5, 1.0, 2.0, 4.0, 8.0] {
    let rate: Double = filmBitsPerPixel * pixelRate * scale
    var s = MP4Settings(width: width, height: height, fps: fps)
    s.averageBitRate = Int(rate.rounded())
    s.maxKeyFrameInterval = 2 * Int(fps)          // 2 s, as the film
    let out = URL(fileURLWithPath: ".build/p1_\(gifURL.deletingPathExtension().lastPathComponent)_x\(scale).mp4")
    do {
        let t0 = Date()
        let w = try MP4Writer(url: out, settings: s)
        for f in frames { try w.append(rgba: f) }
        let bytes: Int = try w.finish()
        let dt: Double = Date().timeIntervalSince(t0)
        var sumMSE: Double = 0
        var minP: Double = .infinity
        var maxErr: Int = 0
        var decodedCount: Int = 0
        _ = try readMP4(url: out) { n, rgba in
            decodedCount += 1
            guard n < frames.count else { return }
            let (p, e) = rgbPSNR(frames[n], rgba)
            minP = min(minP, p); maxErr = max(maxErr, e)
            let peak: Double = 255 * 255
            let tenth: Double = p / 10
            let mse: Double = peak / pow(10, tenth)
            sumMSE += mse
        }
        // Mean PSNR here is PSNR of the mean MSE over all frames (the whole clip as one signal).
        let framesMeasured: Int = max(decodedCount, 1)
        let meanMSE: Double = sumMSE / Double(framesMeasured)
        let peakSquared: Double = 255 * 255
        let meanP: Double = 10 * log10(peakSquared / meanMSE)
        let ratio: Double = Double(gifBytes) / Double(bytes)
        print(String(format: "%5.1f  %13d  %9d  %6.1f×  %7.2f dB  %6.2f dB  %6d  %8.2f%@",
                     scale, s.averageBitRate, bytes, ratio, meanP, minP, maxErr, dt,
                     decodedCount == frames.count ? "" : "  (decoded \(decodedCount) frames!)"))
    } catch {
        print("scale \(scale): \(error)")
    }
}
