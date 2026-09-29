// lib/video/v1 — reading an MP4 back, to test what MP4Writer wrote.
//
// Two independent readers, so a test does not only trust AVFoundation to
// agree with itself:
//   * readMP4(url:) — AVFoundation: AVURLAsset for the track's format and
//     timing, AVAssetReader to decode every frame back to RGBA8 in
//     presentation order;
//   * mp4Boxes(url:) — a plain byte walk of the ISO base media file boxes
//     (ISO/IEC 14496-12): mdhd's timescale and duration, stts's
//     (sample count, sample delta) runs, stsz's sample count, the stsd entry's
//     four-character code ('avc1' for H.264), and whether moov precedes mdat.
//
// APIs: AVAsset.load(_:) and loadTracks(withMediaType:) are macOS 12+/13+;
// nothing here is macOS 15+.

import Foundation
import AVFoundation
import CoreMedia
import CoreVideo

// MARK: - AVFoundation's view

struct MP4TrackInfo {
    var codec: FourCharCode            // kCMVideoCodecType_H264 = 'avc1'
    var width: Int
    var height: Int
    var nominalFrameRate: Float
    var duration: CMTime               // the asset's duration
    var trackTimeRange: CMTimeRange
    var presentationTimes: [CMTime]    // one per decoded frame, in the order decoded
}

private final class VideoBox<T>: @unchecked Sendable { var value: T? = nil; var error: Error? = nil }

/// Runs an async body to completion from synchronous code (tests and tools
/// only — never call it from inside an async context).
func videoBlocking<T>(_ body: @escaping () async throws -> T) throws -> T {
    let box = VideoBox<T>()
    let done = DispatchSemaphore(value: 0)
    Task.detached {
        do { box.value = try await body() } catch { box.error = error }
        done.signal()
    }
    done.wait()
    if let e = box.error { throw e }
    guard let v = box.value else { throw MP4WriterError.writerFailed("no value") }
    return v
}

private struct VideoTrackBasics: @unchecked Sendable {
    var track: AVAssetTrack
    var codec: FourCharCode
    var width: Int
    var height: Int
    var fps: Float
    var duration: CMTime
    var range: CMTimeRange
}

private func loadBasics(_ asset: AVURLAsset) throws -> VideoTrackBasics {
    return try videoBlocking { () async throws -> VideoTrackBasics in
        let tracks: [AVAssetTrack] = try await asset.loadTracks(withMediaType: .video)
        let trackCount: Int = tracks.count
        guard trackCount == 1, let track = tracks.first else {
            throw MP4WriterError.writerFailed("expected 1 video track, found \(trackCount)")
        }
        let formats: [CMFormatDescription] = try await track.load(.formatDescriptions)
        guard let format = formats.first else { throw MP4WriterError.writerFailed("no format description") }
        let dims: CMVideoDimensions = CMVideoFormatDescriptionGetDimensions(format)
        let fps: Float = try await track.load(.nominalFrameRate)
        let duration: CMTime = try await asset.load(.duration)
        let range: CMTimeRange = try await track.load(.timeRange)
        return VideoTrackBasics(track: track, codec: CMFormatDescriptionGetMediaSubType(format),
                                width: Int(dims.width), height: Int(dims.height), fps: fps,
                                duration: duration, range: range)
    }
}

/// Reads the file's single video track and decodes every frame. `frame` is
/// called with each decoded frame as tightly packed RGBA8 (alpha 255), in the
/// order AVAssetReader delivers them (presentation order).
func readMP4(url: URL, frame: ((Int, [UInt8]) -> Void)? = nil) throws -> MP4TrackInfo {
    let asset = AVURLAsset(url: url)
    let basics: VideoTrackBasics = try loadBasics(asset)
    let reader: AVAssetReader = try AVAssetReader(asset: asset)
    let outSettings: [String: Any] = [kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA)]
    let output = AVAssetReaderTrackOutput(track: basics.track, outputSettings: frame == nil ? nil : outSettings)
    output.alwaysCopiesSampleData = false
    guard reader.canAdd(output) else { throw MP4WriterError.writerFailed("reader cannot add output") }
    reader.add(output)
    guard reader.startReading() else {
        throw MP4WriterError.writerFailed(reader.error.map { "\($0)" } ?? "startReading failed")
    }
    var times: [CMTime] = []
    while let sample = output.copyNextSampleBuffer() {
        if CMSampleBufferGetNumSamples(sample) == 0 { continue }   // markers carry no frame
        times.append(CMSampleBufferGetPresentationTimeStamp(sample))
        if let callback = frame, let image = CMSampleBufferGetImageBuffer(sample) {
            callback(times.count - 1, rgbaFromBGRA(image))
        }
    }
    guard reader.status == .completed else {
        throw MP4WriterError.writerFailed(reader.error.map { "\($0)" } ?? "reader status \(reader.status.rawValue)")
    }
    return MP4TrackInfo(codec: basics.codec, width: basics.width, height: basics.height,
                        nominalFrameRate: basics.fps, duration: basics.duration,
                        trackTimeRange: basics.range, presentationTimes: times)
}

private func rgbaFromBGRA(_ image: CVImageBuffer) -> [UInt8] {
    CVPixelBufferLockBaseAddress(image, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(image, .readOnly) }
    let w: Int = CVPixelBufferGetWidth(image)
    let h: Int = CVPixelBufferGetHeight(image)
    let rowBytes: Int = CVPixelBufferGetBytesPerRow(image)
    var out = [UInt8](repeating: 255, count: w * h * 4)
    guard let base = CVPixelBufferGetBaseAddress(image) else { return out }
    let src = base.assumingMemoryBound(to: UInt8.self)
    for y in 0..<h {
        let row: Int = y * rowBytes
        let outRow: Int = y * w * 4
        for x in 0..<w {
            let s: Int = row + x * 4
            let d: Int = outRow + x * 4
            out[d] = src[s + 2]
            out[d + 1] = src[s + 1]
            out[d + 2] = src[s]
        }
    }
    return out
}

// MARK: - the boxes themselves

struct MP4BoxFacts {
    var sampleEntry: String = ""                 // stsd's first entry type, "avc1" for H.264
    var mediaTimescale: UInt32 = 0               // mdhd
    var mediaDuration: UInt64 = 0                // mdhd, in media timescale units
    var movieTimescale: UInt32 = 0               // mvhd
    var movieDuration: UInt64 = 0                // mvhd
    var timeToSample: [(count: UInt32, delta: UInt32)] = []   // stts runs
    var sampleCount: UInt32 = 0                  // stsz
    var moovBeforeMdat: Bool = false
}

/// Walks the file's boxes. Assumes one track (what MP4Writer writes).
func mp4Boxes(url: URL) throws -> MP4BoxFacts {
    let data: Data = try Data(contentsOf: url)
    let bytes: [UInt8] = [UInt8](data)
    var facts = MP4BoxFacts()
    var sawMdat: Bool = false
    func u32(_ i: Int) -> UInt32 {
        let a: UInt32 = UInt32(bytes[i]) << 24
        let b: UInt32 = UInt32(bytes[i + 1]) << 16
        let c: UInt32 = UInt32(bytes[i + 2]) << 8
        return a | b | c | UInt32(bytes[i + 3])
    }
    func u64(_ i: Int) -> UInt64 { (UInt64(u32(i)) << 32) | UInt64(u32(i + 4)) }
    func name(_ i: Int) -> String { String(bytes: bytes[i..<(i + 4)], encoding: .ascii) ?? "????" }
    let containers: Set<String> = ["moov", "trak", "mdia", "minf", "stbl", "edts"]

    func walk(_ start: Int, _ end: Int) {
        var i: Int = start
        while i + 8 <= end {
            var size: Int = Int(u32(i))
            let type: String = name(i + 4)
            var header: Int = 8
            if size == 1 {
                let large: UInt64 = u64(i + 8)
                size = Int(large); header = 16
            }
            if size == 0 { size = end - i }
            if size < header || i + size > end { return }
            let body: Int = i + header
            switch type {
            case "mdat":
                sawMdat = true
            case "moov":
                if !sawMdat { facts.moovBeforeMdat = true }
                walk(body, i + size)
            case "mvhd":
                let version: UInt8 = bytes[body]
                if version == 1 {
                    facts.movieTimescale = u32(body + 20); facts.movieDuration = u64(body + 24)
                } else {
                    facts.movieTimescale = u32(body + 12); facts.movieDuration = UInt64(u32(body + 16))
                }
            case "mdhd":
                let version: UInt8 = bytes[body]
                if version == 1 {
                    facts.mediaTimescale = u32(body + 20); facts.mediaDuration = u64(body + 24)
                } else {
                    facts.mediaTimescale = u32(body + 12); facts.mediaDuration = UInt64(u32(body + 16))
                }
            case "stsd":
                if body + 16 <= i + size { facts.sampleEntry = name(body + 12) }
            case "stts":
                let entries: UInt32 = u32(body + 4)
                let n: Int = Int(entries)
                for k in 0..<n {
                    let at: Int = body + 8 + k * 8
                    facts.timeToSample.append((count: u32(at), delta: u32(at + 4)))
                }
            case "stsz":
                facts.sampleCount = u32(body + 8)
            default:
                if containers.contains(type) { walk(body, i + size) }
            }
            i += size
        }
    }
    walk(0, bytes.count)
    return facts
}

// MARK: - comparing pictures

/// PSNR over R, G and B of two RGBA8 images of the same size (alpha ignored):
/// 10·log10(255² / MSE), MSE averaged over all pixels and the three channels
/// (Wikipedia, "Peak signal-to-noise ratio", colour images). Also the largest
/// single-channel difference. Returns +infinity PSNR for identical images.
func rgbPSNR(_ a: [UInt8], _ b: [UInt8]) -> (psnr: Double, maxError: Int) {
    precondition(a.count == b.count && a.count % 4 == 0, "images differ in size")
    var sum: Double = 0
    var worst: Int = 0
    var i: Int = 0
    while i < a.count {
        for c in 0..<3 {
            let d: Int = Int(a[i + c]) - Int(b[i + c])
            sum += Double(d * d)
            if abs(d) > worst { worst = abs(d) }
        }
        i += 4
    }
    let mse: Double = sum / Double(a.count / 4 * 3)
    if mse == 0 { return (Double.infinity, 0) }
    let peak: Double = 255 * 255
    return (10 * log10(peak / mse), worst)
}
