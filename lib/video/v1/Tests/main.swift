// lib/video/v1 tests: write a short synthetic 1080p30 clip, read it back two
// ways (AVFoundation and a byte walk of the MP4 boxes), and check it is the
// film we meant: H.264, 1920×1080, every frame present once, in order, at
// exactly n/30 s, and close to the pixels we put in.
//
// Run with VIDEO_MUTANT=<name> to break the writer; the suite must then FAIL.

import Foundation
import CoreMedia
import AVFoundation

let clipFrames: Int = 75     // 2.5 s at 30 fps: crosses one 60-frame key-frame interval
let W: Int = 1920
let H: Int = 1080
let fps: Int32 = 30
let buildDir = URL(fileURLWithPath: ".build", isDirectory: true)
try? FileManager.default.createDirectory(at: buildDir, withIntermediateDirectories: true)
let clipURL: URL = buildDir.appendingPathComponent("test_clip.mp4")

// MARK: - the synthetic frames

// Frame index as 8 bits, most significant first, in 8 squares across the top:
// white = 1, black = 0. Squares are 96 px, far larger than H.264's 16 px
// macroblocks, so compression cannot flip a bit.
let bitCount: Int = 8
let bitSide: Int = 96
let bitGap: Int = 24
let bitTop: Int = 48
func bitLeft(_ b: Int) -> Int { 48 + b * (bitSide + bitGap) }

// Three pure colour patches (red, green, blue), 192 px, lower left: they catch
// a red/blue swap.
let patchSide: Int = 192
let patchTop: Int = 800
let patchColors: [(UInt8, UInt8, UInt8)] = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
func patchLeft(_ k: Int) -> Int { 48 + k * (patchSide + 48) }

func syntheticFrame(_ n: Int) -> [UInt8] {
    var px = [UInt8](repeating: 255, count: W * H * 4)
    // Background: a smooth diagonal gradient that drifts 8 px per frame, so every
    // frame differs everywhere (a real inter-frame prediction problem).
    let shift: Int = n * 8
    // A disc of radius 120 moving right 12 px per frame.
    let cx: Int = 700 + n * 12
    let cy: Int = 540
    let r2: Int = 120 * 120
    for y in 0..<H {
        for x in 0..<W {
            let i: Int = (y * W + x) * 4
            let g: Int = (x + y + shift) % 512
            let v: Int = g < 256 ? g : 511 - g
            let dx: Int = x - cx
            let dy: Int = y - cy
            if dx * dx + dy * dy <= r2 {
                px[i] = 250; px[i + 1] = 200; px[i + 2] = 40
            } else {
                px[i] = UInt8(v / 2 + 40); px[i + 1] = UInt8(v / 3 + 60); px[i + 2] = UInt8(200 - v / 2)
            }
        }
    }
    func fill(_ x0: Int, _ y0: Int, _ side: Int, _ c: (UInt8, UInt8, UInt8)) {
        for y in y0..<(y0 + side) {
            for x in x0..<(x0 + side) {
                let i: Int = (y * W + x) * 4
                px[i] = c.0; px[i + 1] = c.1; px[i + 2] = c.2
            }
        }
    }
    for b in 0..<bitCount {
        let on: Bool = (n >> (bitCount - 1 - b)) & 1 == 1
        let c: UInt8 = on ? 255 : 0
        fill(bitLeft(b), bitTop, bitSide, (c, c, c))
    }
    for k in 0..<3 { fill(patchLeft(k), patchTop, patchSide, patchColors[k]) }
    return px
}

func meanRGB(_ px: [UInt8], _ x0: Int, _ y0: Int, _ side: Int) -> (Double, Double, Double) {
    // The inner half of the square, away from edges where chroma blurs.
    let m: Int = side / 4
    var s: (Double, Double, Double) = (0, 0, 0)
    var n: Double = 0
    for y in (y0 + m)..<(y0 + side - m) {
        for x in (x0 + m)..<(x0 + side - m) {
            let i: Int = (y * W + x) * 4
            s.0 += Double(px[i]); s.1 += Double(px[i + 1]); s.2 += Double(px[i + 2])
            n += 1
        }
    }
    return (s.0 / n, s.1 / n, s.2 / n)
}

func decodeIndex(_ px: [UInt8]) -> Int {
    var n: Int = 0
    for b in 0..<bitCount {
        let m = meanRGB(px, bitLeft(b), bitTop, bitSide)
        n = n << 1 | (m.1 > 128 ? 1 : 0)
    }
    return n
}

// MARK: - write it once

print("lib/video/v1 tests" + (videoMutant == .none ? "" : "  (MUTANT \(videoMutant.rawValue))"))
let inputs: [[UInt8]] = (0..<clipFrames).map(syntheticFrame)
var clipBytes: Int = 0
var writeSeconds: Double = 0
var writeError: Error? = nil
do {
    let t0 = Date()
    let writer = try MP4Writer(url: clipURL, settings: MP4Settings())
    for f in inputs { try writer.append(rgba: f) }
    clipBytes = try writer.finish()
    writeSeconds = Date().timeIntervalSince(t0)
} catch {
    writeError = error
}

var info: MP4TrackInfo? = nil
var decoded: [[UInt8]] = []
var readError: Error? = nil
do {
    info = try readMP4(url: clipURL) { _, rgba in decoded.append(rgba) }
} catch {
    readError = error
}
let boxes: MP4BoxFacts? = try? mp4Boxes(url: clipURL)

// MARK: - tests

section("the encoder is hardware")
test("VideoToolbox lists a hardware H.264 encoder, and a require-hardware 1920×1080 session opens on it") {
    let report = probeH264Hardware(width: W, height: H)
    print("        hardware H.264 encoders: \(report.hardwareEncoderIDs); session created \(report.sessionCreated), using hardware \(report.sessionUsesHardware)")
    expect(!report.hardwareEncoderIDs.isEmpty, "no hardware H.264 encoder listed")
    expect(report.sessionCreated, "a require-hardware session could not be created")
    expect(report.sessionUsesHardware, "the session does not report UsingHardwareAcceleratedVideoEncoder")
}
test("the writer's default settings require the hardware encoder") {
    expect(MP4Settings().requireHardwareEncoder)
}
// AVAssetWriter hides the session it makes, so show the requirement reaches it:
// 8192×4320 is beyond this hardware H.264 encoder (a VideoToolbox session that
// requires hardware fails to open at that size), but VideoToolbox's software
// encoder takes it. A writer that honours "require hardware" must refuse it;
// one that doesn't would quietly encode in software.
test("the writer honours 'require hardware': it refuses 8192×4320, which only software can encode") {
    let bigW: Int = 8192
    let bigH: Int = 4320
    let probe = probeH264Hardware(width: bigW, height: bigH)
    expect(!probe.sessionCreated, "hardware accepted 8192×4320 on this machine; the test needs a size it refuses")
    let bigURL: URL = buildDir.appendingPathComponent("too_big.mp4")
    let frame = [UInt8](repeating: 100, count: bigW * bigH * 4)
    var hardwareRefused: Bool = false
    do {
        let w = try MP4Writer(url: bigURL, settings: MP4Settings(width: bigW, height: bigH, fps: 30))
        try w.append(rgba: frame)
        _ = try w.finish()
    } catch {
        hardwareRefused = true
    }
    expect(hardwareRefused, "a require-hardware writer encoded 8192×4320 (so it fell back to software)")
    var software = MP4Settings(width: bigW, height: bigH, fps: 30)
    software.requireHardwareEncoder = false
    do {
        let w = try MP4Writer(url: bigURL, settings: software)
        try w.append(rgba: frame)
        _ = try w.finish()
    } catch {
        expect(false, "without the requirement the software encoder should take it: \(error)")
    }
    try? FileManager.default.removeItem(at: bigURL)
}

section("the file")
test("the clip was written without error, with the hardware encoder required") {
    expect(writeError == nil, "\(String(describing: writeError))")
    expect(clipBytes > 0)
    let seconds: Double = Double(clipFrames) / Double(fps)
    print(String(format: "        %d frames, %d bytes, %.0f bytes/s of film; wrote in %.2f s (%.0f frames/s incl. drawing-free append)",
                 clipFrames, clipBytes, Double(clipBytes) / seconds, writeSeconds, Double(clipFrames) / writeSeconds))
}
test("AVFoundation opens and decodes it to the end") {
    expect(readError == nil, "\(String(describing: readError))")
}
test("codec H.264, 1920×1080, nominal 30 fps") {
    guard let t = info else { expect(false, "no track info"); return }
    expectEqual(t.codec, kCMVideoCodecType_H264)
    expectEqual(t.width, W)
    expectEqual(t.height, H)
    expect(abs(t.nominalFrameRate - 30) < 0.001, "nominal frame rate \(t.nominalFrameRate)")
    expectEqual(boxes?.sampleEntry ?? "", "avc1")
}
test("moov before mdat (fast start)") {
    expect(boxes?.moovBeforeMdat == true)
}

section("frames and time")
test("exact frame count: \(clipFrames) decoded, \(clipFrames) in the sample table") {
    expectEqual(info?.presentationTimes.count ?? -1, clipFrames)
    expectEqual(Int(boxes?.sampleCount ?? 0), clipFrames)
}
test("exact duration: \(clipFrames)/30 s, in the asset, the track and the media header") {
    guard let t = info, let b = boxes else { expect(false, "unreadable"); return }
    let want = CMTime(value: CMTimeValue(clipFrames), timescale: fps)
    expect(CMTimeCompare(t.duration, want) == 0, "asset duration \(CMTimeGetSeconds(t.duration)) s")
    expect(CMTimeCompare(t.trackTimeRange.duration, want) == 0, "track duration \(CMTimeGetSeconds(t.trackTimeRange.duration)) s")
    expect(CMTimeCompare(t.trackTimeRange.start, .zero) == 0, "track starts at \(CMTimeGetSeconds(t.trackTimeRange.start))")
    expectEqual(b.mediaTimescale, UInt32(fps))
    expectEqual(b.mediaDuration, UInt64(clipFrames))
    let movieSeconds: Double = Double(b.movieDuration) / Double(b.movieTimescale)
    expect(abs(movieSeconds - Double(clipFrames) / 30) < 1e-9, "mvhd duration \(movieSeconds) s")
}
test("constant frame rate: every sample lasts exactly 1/30 s (one stts run, delta 1 at timescale 30)") {
    guard let b = boxes else { expect(false, "unreadable"); return }
    expectEqual(b.timeToSample.count, 1)
    if let run = b.timeToSample.first {
        expectEqual(Int(run.count), clipFrames)
        expectEqual(run.delta, 1)
    }
}
test("frame n is presented at exactly n/30 s") {
    guard let t = info else { expect(false, "unreadable"); return }
    for (n, pts) in t.presentationTimes.enumerated() {
        let want = CMTime(value: CMTimeValue(n), timescale: fps)
        if CMTimeCompare(pts, want) != 0 {
            expect(false, "frame \(n) at \(CMTimeGetSeconds(pts)) s, want \(CMTimeGetSeconds(want)) s"); return
        }
    }
}
test("forward only: the index drawn in decoded frame n is n, for every n") {
    let got: [Int] = decoded.map(decodeIndex)
    expectEqual(got, Array(0..<clipFrames))
}

section("pixels")
var psnrs: [Double] = []
var worstError: Int = 0
if decoded.count == inputs.count {
    for (a, b) in zip(inputs, decoded) {
        let (p, e) = rgbPSNR(a, b)
        psnrs.append(p); worstError = max(worstError, e)
    }
}
// Tolerance. MODEL: 35 dB is inside the 30–50 dB "typical values for the PSNR
// in lossy image and video compression" (Wikipedia, "Peak signal-to-noise
// ratio") and 3 dB below the 38.0 dB this clip's worst frame measures at the
// default settings on the M3 Max (see the printout), so a real loss of
// fidelity fails it and ordinary encoder variation does not. The largest
// single-channel errors (~200/255) sit on the disc's yellow-on-blue edge:
// 4:2:0 chroma is stored at half resolution, so a one-pixel colour edge
// cannot survive; that is the format, not the writer.
let minPSNR: Double = 35
test("every decoded frame is within PSNR ≥ \(Int(minPSNR)) dB of its input") {
    expectEqual(psnrs.count, clipFrames)
    let worst: Double = psnrs.min() ?? 0
    let mean: Double = psnrs.isEmpty ? 0 : psnrs.reduce(0, +) / Double(psnrs.count)
    print(String(format: "        PSNR min %.2f dB, mean %.2f dB; largest single-channel error %d / 255 (at sharp edges)",
                 worst, mean, worstError))
    expect(worst >= minPSNR, "worst frame \(worst) dB")
}
test("red, green and blue stay red, green and blue (within 24 / 255 of each channel)") {
    guard let first = decoded.first else { expect(false, "no frames"); return }
    for k in 0..<3 {
        let m = meanRGB(first, patchLeft(k), patchTop, patchSide)
        let c = patchColors[k]
        let er: Double = abs(m.0 - Double(c.0))
        let eg: Double = abs(m.1 - Double(c.1))
        let eb: Double = abs(m.2 - Double(c.2))
        let err: Double = max(er, eg, eb)
        print(String(format: "        patch %d: in (%d,%d,%d) out (%.0f,%.0f,%.0f)", k, c.0, c.1, c.2, m.0, m.1, m.2))
        expect(err <= 24, "patch \(k) off by \(err)")
    }
}

section("the API refuses what it can't do")
test("odd sizes are refused (4:2:0 needs even width and height)") {
    expectThrows { _ = try MP4Writer(url: buildDir.appendingPathComponent("odd.mp4"), settings: MP4Settings(width: 1921, height: 1080)) }
}
test("a frame of the wrong size is refused; append after finish is refused") {
    let small = MP4Settings(width: 64, height: 64, fps: 30)
    do {
        let w = try MP4Writer(url: buildDir.appendingPathComponent("small.mp4"), settings: small)
        expectThrows { try w.append(rgba: [UInt8](repeating: 0, count: 10)) }
        try w.append(rgba: [UInt8](repeating: 128, count: 64 * 64 * 4))
        _ = try w.finish()
        expectThrows { try w.append(rgba: [UInt8](repeating: 128, count: 64 * 64 * 4)) }
    } catch {
        expect(false, "\(error)")
    }
}

finish()
