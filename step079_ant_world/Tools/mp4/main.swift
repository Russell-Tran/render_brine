// Step 79: the film's MP4, with lib/video/v1 (AVAssetWriter, H.264, the
// hardware encoder, 2.2 Mbit/s). A tool of its own so that AVFoundation is
// compiled into nothing else (see Sources/Film.swift).
//
//   .build/mp4 encode DIR COUNT OUT.mp4   frames DIR/0000.png … in order
//   .build/mp4 check FILE COUNT [MB]      H.264, 1920×1080, COUNT frames at 30 fps,
//                                         timestamps strictly increasing (the film
//                                         never goes back), duration COUNT/30 s,
//                                         at most MB megabytes (default 20)
//   .build/mp4 synth COUNT                encode COUNT small synthetic frames and
//                                         check them the same way (a quick test)

import AVFoundation
import CoreGraphics
import Foundation
import ImageIO

func fail(_ s: String) -> Never {
    print("FAIL  \(s)")
    exit(1)
}

func readPNG(_ url: URL) -> (Int, Int, [UInt8]) {
    guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
          let img = CGImageSourceCreateImageAtIndex(src, 0, nil),
          let space = CGColorSpace(name: CGColorSpace.sRGB) else { fail("cannot read \(url.path)") }
    let w: Int = img.width
    let h: Int = img.height
    var out = [UInt8](repeating: 255, count: w * h * 4)
    let drawn: Bool = out.withUnsafeMutableBytes { raw -> Bool in
        guard let ctx = CGContext(data: raw.baseAddress, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                                  space: space, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { return false }
        ctx.draw(img, in: CGRect(x: 0, y: 0, width: w, height: h))
        return true
    }
    if !drawn { fail("cannot decode \(url.path)") }
    return (w, h, out)
}

/// Check an MP4: returns a one-line summary, or fails.
func check(_ url: URL, count: Int, width: Int, height: Int, maxMB: Double) -> String {
    let info: MP4TrackInfo
    // Decoded (the callback asks for pixels), so frames come out in
    // presentation order; read raw, H.264's reordered frames would not.
    do { info = try readMP4(url: url) { _, _ in } } catch { fail("cannot read \(url.path): \(error)") }
    let times: [CMTime] = info.presentationTimes
    // Matched, not compared with ==: AVFoundation's operators make every ==
    // slow to type-check in this module.
    let codec: FourCharCode = info.codec
    let h264: FourCharCode = kCMVideoCodecType_H264
    switch codec { case h264: break; default: fail("codec is not H.264") }
    switch (info.width, info.height) { case (width, height): break; default: fail("size \(info.width)×\(info.height), want \(width)×\(height)") }
    let frames: Int = times.count
    switch frames { case count: break; default: fail("\(frames) frames, want \(count)") }
    for k in 1..<max(times.count, 1) {
        let a: Double = CMTimeGetSeconds(times[k - 1])
        let b: Double = CMTimeGetSeconds(times[k])
        if !(b > a) { fail("frame \(k) at \(b) s is not after frame \(k - 1) at \(a) s") }
        let want: Double = Double(k) / 30
        if abs(b - want) > 1e-9 { fail("frame \(k) at \(b) s, want \(want) s") }
    }
    let seconds: Double = CMTimeGetSeconds(info.duration)
    let want: Double = Double(count) / 30
    if abs(seconds - want) > 1e-6 { fail("duration \(seconds) s, want \(want) s") }
    let bytes: Int = (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? Int) ?? 0
    let mb: Double = Double(bytes) / 1_048_576
    if mb > maxMB { fail(String(format: "%.2f MB, over %.0f MB", mb, maxMB)) }
    return String(format: "ok    %@: H.264 %d×%d, %d frames, %.3f s, %.1f fps, timestamps increasing, %.2f MB",
                  url.lastPathComponent, info.width, info.height, times.count, seconds, info.nominalFrameRate, mb)
}

var args: [String] = CommandLine.arguments
if !args.isEmpty { args.removeFirst() }
switch args.first ?? "" {
case "encode":
    guard args.count >= 4, let n = Int(args[2]) else { fail("usage: encode DIR COUNT OUT") }
    let dir = URL(fileURLWithPath: args[1])
    let out = URL(fileURLWithPath: args[3])
    let t0: Date = Date()
    let first = readPNG(dir.appendingPathComponent("0000.png"))
    var settings = MP4Settings()
    settings.width = first.0
    settings.height = first.1
    settings.fps = 30
    try? FileManager.default.removeItem(at: out)
    do {
        let writer = try MP4Writer(url: out, settings: settings)
        for f in 0..<n {
            let frame = readPNG(dir.appendingPathComponent(String(format: "%04d.png", f)))
            try writer.append(rgba: frame.2)
        }
        let bytes: Int = try writer.finish()
        let mb: Double = Double(bytes) / 1_048_576
        print(String(format: "encoded %d frames → %@ (%.2f MB) in %.1f s", n, out.path, mb, Date().timeIntervalSince(t0)))
    } catch { fail("encoding: \(error)") }

case "check":
    guard args.count >= 3, let n = Int(args[2]) else { fail("usage: check FILE COUNT [MB]") }
    let mb: Double = args.count > 3 ? (Double(args[3]) ?? 20) : 20
    print(check(URL(fileURLWithPath: args[1]), count: n, width: 1920, height: 1080, maxMB: mb))

case "synth":
    guard args.count >= 2, let n = Int(args[1]) else { fail("usage: synth COUNT") }
    let url = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("step079_synth.mp4")
    var settings = MP4Settings()
    settings.width = 320
    settings.height = 180
    settings.fps = 30
    do {
        let writer = try MP4Writer(url: url, settings: settings)
        for f in 0..<n {
            var px = [UInt8](repeating: 255, count: 320 * 180 * 4)
            let shade: UInt8 = UInt8(f % 256)
            for k in stride(from: 0, to: px.count, by: 4) { px[k] = shade }
            try writer.append(rgba: px)
        }
        try writer.finish()
    } catch { fail("encoding: \(error)") }
    print(check(url, count: n, width: 320, height: 180, maxMB: 20))

default:
    fail("usage: mp4 encode DIR COUNT OUT | check FILE COUNT [MB] | synth COUNT")
}
