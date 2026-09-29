// lib/video/v1 — an H.264 MP4 writer for rendered frames.
//
// VERSION-PINNED. A step that uses this compiles this file into its own binary
// (its Makefile lists ../lib/video/v1/Sources/MP4Writer.swift). Never change
// behaviour here once a step depends on it: a change makes lib/video/v2.
//
// What it does: takes RGBA8 frames (the layout every step's Renderer writes
// into its MTLBuffer: 4 bytes per pixel, R G B A, rows top to bottom) and
// appends them to an MP4 through AVAssetWriter, H.264, at a constant frame
// rate. Frame n is presented at exactly n / fps seconds (timescale = fps, so
// the times are integers, never rounded), and the session ends at exactly
// frameCount / fps, so the file's duration is exact too.
//
// Only APIs available on macOS 10.10–10.15 are used here (nothing from macOS
// 15+): AVAssetWriter, AVAssetWriterInputPixelBufferAdaptor, VideoToolbox's
// encoder specification keys (macOS 10.9) and encoder list (10.14), vImage.

import Foundation
import AVFoundation
import CoreMedia
import CoreVideo
import VideoToolbox
import Accelerate

// MARK: - what to break, for the mutation check

/// Each one must make lib/video/v1's test suite fail. Read once, from the
/// environment variable VIDEO_MUTANT; unset means no mutant (the normal case).
enum VideoMutant: String {
    case none
    case wrongFrameCount   // the writer silently drops one frame (the 3rd appended)
    case wrongFps          // frames stamped at 1/25 s apart while the file claims 30 fps
    case swapChannels      // RGBA read as BGRA: red and blue swapped
    case softwareAllowed   // the "require hardware" specification not passed to the writer
}

let videoMutant: VideoMutant = {
    let raw: String = ProcessInfo.processInfo.environment["VIDEO_MUTANT"] ?? "none"
    return VideoMutant(rawValue: raw) ?? .none
}()

// MARK: - settings

struct MP4Settings {
    var width: Int = 1920
    var height: Int = 1080
    var fps: Int32 = 30

    // Average bit rate, bits per second. 2.2 Mbit/s = 275,000 bytes/s, 16.5 MB
    // for a 60 s film, under the handoff's "aiming for ≤ 20 MB"
    // (handoffs/2026-09-28_laptop_ant_world.md, part 5) with room for the
    // encoder's overshoot: re-encoding three showcase GIFs at this setting's
    // bits per pixel (`make p1`), the files came out 2–8 % above target (the
    // target is an AVERAGE, not a cap), so 16.5 MB → ~17.9 MB at worst seen.
    // (At half this rate the encoder hits a floor and overshoots by up to a
    // third; don't go lower expecting proportionally smaller files.)
    // MODEL: that is 28 % of YouTube's recommended 8 Mbps for a 1080p SDR
    // 24–30 fps upload (support.google.com/youtube/answer/1722171) and 37 % of
    // the lowest 1080p rung (6000 kb/s) of Apple's HLS Authoring Specification
    // bit-rate ladder: the 20 MB git budget decides it, and our frames — a
    // still ground with small moving ants — are far easier than the camera
    // footage those figures are for. `make p1` shows what it costs in quality.
    var averageBitRate: Int = 2_200_000

    // A key frame at least every 60 frames = 2 s at 30 fps: Apple's HLS
    // Authoring Specification for Apple Devices, item 1.13, "Key frames (IDRs)
    // SHOULD be present every two seconds." (YouTube's page asks for half a
    // second, but that is for an upload YouTube re-encodes; here the MP4 is
    // the delivered file, and key frames are the expensive ones: on
    // ant_trail_wood.gif at the film's bits per pixel, a 2 s interval scored
    // +2.9 dB PSNR over a 0.5 s one at the same size.)
    var maxKeyFrameInterval: Int = 60

    // Refuse to run on a software encoder. VideoToolbox's
    // kVTVideoEncoderSpecification_RequireHardwareAcceleratedVideoEncoder:
    // "only use hardware encode and return an error if this isn't possible"
    // (VTCompressionProperties.h, macOS 14.5 SDK). Passed to AVAssetWriter
    // through AVVideoEncoderSpecificationKey (AVVideoSettings.h, macOS 10.10+).
    var requireHardwareEncoder: Bool = true

    // "moov atom at the front of the file (Fast Start)" — YouTube's page
    // (support.google.com/youtube/answer/1722171); AVAssetWriter.shouldOptimizeForNetworkUse does that ("enables a
    // player to begin playing the media after downloading only a small
    // portion of it", Apple developer documentation).
    var fastStart: Bool = true
}

enum MP4WriterError: Error, CustomStringConvertible {
    case cannotCreate(String)
    case badFrame(String)
    case writerFailed(String)

    var description: String {
        switch self {
        case .cannotCreate(let s): return "MP4Writer: cannot create: \(s)"
        case .badFrame(let s): return "MP4Writer: bad frame: \(s)"
        case .writerFailed(let s): return "MP4Writer: writer failed: \(s)"
        }
    }
}

// MARK: - the writer

/// Append frames one at a time, then `finish()`. Not thread-safe: call it from
/// one thread. Blocks (briefly) when the encoder is busy, so a renderer can
/// simply loop render → append.
final class MP4Writer {
    let url: URL
    let settings: MP4Settings
    private let writer: AVAssetWriter
    private let input: AVAssetWriterInput
    private let adaptor: AVAssetWriterInputPixelBufferAdaptor
    private(set) var framesAppended: Int = 0    // frames the caller handed in
    private var framesWritten: Int = 0          // frames given to the encoder
    private var finished: Bool = false

    /// Creates the file (replacing any file already at `url`).
    init(url: URL, settings: MP4Settings = MP4Settings()) throws {
        guard settings.width > 0, settings.height > 0, settings.width % 2 == 0, settings.height % 2 == 0 else {
            // 4:2:0 chroma needs even sizes.
            throw MP4WriterError.cannotCreate("width and height must be positive and even, got \(settings.width)×\(settings.height)")
        }
        guard settings.fps > 0 else { throw MP4WriterError.cannotCreate("fps must be positive") }
        self.url = url
        self.settings = settings
        try? FileManager.default.removeItem(at: url)
        do {
            writer = try AVAssetWriter(outputURL: url, fileType: .mp4)
        } catch {
            throw MP4WriterError.cannotCreate("\(error)")
        }
        writer.shouldOptimizeForNetworkUse = settings.fastStart

        let compression: [String: Any] = [
            AVVideoAverageBitRateKey: settings.averageBitRate,
            AVVideoMaxKeyFrameIntervalKey: settings.maxKeyFrameInterval,
            AVVideoExpectedSourceFrameRateKey: Int(settings.fps),
            AVVideoProfileLevelKey: AVVideoProfileLevelH264HighAutoLevel,
        ]
        // Tag the output BT.709, the HD standard: without a tag, players guess.
        let color: [String: Any] = [
            AVVideoColorPrimariesKey: AVVideoColorPrimaries_ITU_R_709_2,
            AVVideoTransferFunctionKey: AVVideoTransferFunction_ITU_R_709_2,
            AVVideoYCbCrMatrixKey: AVVideoYCbCrMatrix_ITU_R_709_2,
        ]
        var output: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: settings.width,
            AVVideoHeightKey: settings.height,
            AVVideoCompressionPropertiesKey: compression,
            AVVideoColorPropertiesKey: color,
        ]
        if settings.requireHardwareEncoder && videoMutant != .softwareAllowed {
            let requireKey: String = kVTVideoEncoderSpecification_RequireHardwareAcceleratedVideoEncoder as String
            output[AVVideoEncoderSpecificationKey] = [requireKey: true]
        }
        guard writer.canApply(outputSettings: output, forMediaType: .video) else {
            throw MP4WriterError.cannotCreate("the writer refuses these output settings")
        }
        input = AVAssetWriterInput(mediaType: .video, outputSettings: output)
        input.expectsMediaDataInRealTime = false
        // Track timescale = fps, so every presentation time n/fps is an integer
        // count of track ticks.
        input.mediaTimeScale = CMTimeScale(settings.fps)
        let pixelAttributes: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA),
            kCVPixelBufferWidthKey as String: settings.width,
            kCVPixelBufferHeightKey as String: settings.height,
        ]
        adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input,
                                                       sourcePixelBufferAttributes: pixelAttributes)
        guard writer.canAdd(input) else { throw MP4WriterError.cannotCreate("cannot add the video input") }
        writer.add(input)
        guard writer.startWriting() else {
            throw MP4WriterError.cannotCreate(writer.error.map { "\($0)" } ?? "startWriting failed")
        }
        writer.startSession(atSourceTime: .zero)
    }

    /// The presentation time of frame `index`: exactly index / fps.
    func presentationTime(ofFrame index: Int) -> CMTime {
        let stampFps: Int32 = videoMutant == .wrongFps ? 25 : settings.fps
        return CMTime(value: CMTimeValue(index), timescale: stampFps)
    }

    /// Appends one frame. `rgba` points at width × height pixels, R G B A bytes,
    /// `bytesPerRow` apart (default: tightly packed, width × 4). Alpha is
    /// ignored (the video is opaque).
    func append(rgba: UnsafeRawPointer, bytesPerRow: Int? = nil) throws {
        guard !finished else { throw MP4WriterError.badFrame("append after finish") }
        let rowBytes: Int = bytesPerRow ?? settings.width * 4
        guard rowBytes >= settings.width * 4 else { throw MP4WriterError.badFrame("bytesPerRow \(rowBytes) < width × 4") }
        let index: Int = framesAppended
        framesAppended += 1
        if videoMutant == .wrongFrameCount && index == 2 { return }

        guard let pool = adaptor.pixelBufferPool else {
            throw MP4WriterError.writerFailed(writer.error.map { "\($0)" } ?? "no pixel buffer pool")
        }
        var made: CVPixelBuffer? = nil
        let status: CVReturn = CVPixelBufferPoolCreatePixelBuffer(nil, pool, &made)
        guard status == kCVReturnSuccess, let pixelBuffer = made else {
            throw MP4WriterError.writerFailed("CVPixelBufferPoolCreatePixelBuffer \(status)")
        }
        try copyRGBAIntoBGRA(rgba, rowBytes, pixelBuffer)
        tagBT709(pixelBuffer)

        while !input.isReadyForMoreMediaData {
            if writer.status == .failed {
                throw MP4WriterError.writerFailed(writer.error.map { "\($0)" } ?? "failed")
            }
            Thread.sleep(forTimeInterval: 0.001)
        }
        let time: CMTime = presentationTime(ofFrame: framesWritten)
        guard adaptor.append(pixelBuffer, withPresentationTime: time) else {
            throw MP4WriterError.writerFailed(writer.error.map { "\($0)" } ?? "append failed at frame \(index)")
        }
        framesWritten += 1
    }

    /// Appends one tightly packed frame: `rgba.count` must be width × height × 4.
    func append(rgba: [UInt8]) throws {
        let expected: Int = settings.width * settings.height * 4
        guard rgba.count == expected else {
            throw MP4WriterError.badFrame("\(rgba.count) bytes, expected \(expected)")
        }
        try rgba.withUnsafeBytes { (raw: UnsafeRawBufferPointer) throws -> Void in
            guard let base = raw.baseAddress else { throw MP4WriterError.badFrame("empty") }
            try append(rgba: base)
        }
    }

    /// Ends the session at exactly frameCount / fps, closes the file, and
    /// returns its size in bytes.
    @discardableResult
    func finish() throws -> Int {
        guard !finished else { throw MP4WriterError.writerFailed("finish called twice") }
        finished = true
        input.markAsFinished()
        writer.endSession(atSourceTime: CMTime(value: CMTimeValue(framesAppended), timescale: settings.fps))
        let done = DispatchSemaphore(value: 0)
        writer.finishWriting { done.signal() }
        done.wait()
        guard writer.status == .completed else {
            throw MP4WriterError.writerFailed(writer.error.map { "\($0)" } ?? "status \(writer.status.rawValue)")
        }
        let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
        let size: Int = (attributes[.size] as? NSNumber)?.intValue ?? 0
        return size
    }

    // Tag the source buffer with the same colour as the output, so
    // AVFoundation does not convert colour on the way in (the output tag
    // alone, on untagged sources, made every channel come back ~9/255 darker).
    private func tagBT709(_ pixelBuffer: CVPixelBuffer) {
        CVBufferSetAttachment(pixelBuffer, kCVImageBufferColorPrimariesKey, kCVImageBufferColorPrimaries_ITU_R_709_2, .shouldPropagate)
        CVBufferSetAttachment(pixelBuffer, kCVImageBufferTransferFunctionKey, kCVImageBufferTransferFunction_ITU_R_709_2, .shouldPropagate)
        CVBufferSetAttachment(pixelBuffer, kCVImageBufferYCbCrMatrixKey, kCVImageBufferYCbCrMatrix_ITU_R_709_2, .shouldPropagate)
    }

    // RGBA → BGRA (CoreVideo's 32BGRA, the format AVFoundation's encoders take
    // natively), one vImage channel permute per frame.
    private func copyRGBAIntoBGRA(_ rgba: UnsafeRawPointer, _ rowBytes: Int, _ pixelBuffer: CVPixelBuffer) throws {
        CVPixelBufferLockBaseAddress(pixelBuffer, [])
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, []) }
        guard let destination = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            throw MP4WriterError.writerFailed("pixel buffer has no base address")
        }
        let w: vImagePixelCount = vImagePixelCount(settings.width)
        let h: vImagePixelCount = vImagePixelCount(settings.height)
        var src = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: rgba), height: h, width: w, rowBytes: rowBytes)
        var dst = vImage_Buffer(data: destination, height: h, width: w,
                                rowBytes: CVPixelBufferGetBytesPerRow(pixelBuffer))
        // map[i] = which source channel goes to destination channel i.
        let map: [UInt8] = videoMutant == .swapChannels ? [0, 1, 2, 3] : [2, 1, 0, 3]
        let error: vImage_Error = vImagePermuteChannels_ARGB8888(&src, &dst, map, vImage_Flags(kvImageNoFlags))
        guard error == kvImageNoError else { throw MP4WriterError.writerFailed("vImage permute \(error)") }
    }
}

// MARK: - is the encoder hardware?

/// What VideoToolbox says about H.264 hardware encoding on this machine.
struct H264HardwareReport {
    /// Encoders VideoToolbox lists for H.264 that it flags hardware accelerated
    /// (kVTVideoEncoderList_IsHardwareAccelerated, VTVideoEncoderList.h, macOS 10.14+).
    var hardwareEncoderIDs: [String]
    /// A compression session made with the same "require hardware" specification
    /// the writer passes, at the given size: did it open, and does it report
    /// kVTCompressionPropertyKey_UsingHardwareAcceleratedVideoEncoder = true?
    var sessionCreated: Bool
    var sessionUsesHardware: Bool
}

/// Probes VideoToolbox directly. AVAssetWriter does not expose the session it
/// makes, so this cannot look inside the writer; it shows that a hardware H.264
/// encoder exists and accepts this size, and the writer's own "require
/// hardware" specification makes it fail rather than fall back to software.
func probeH264Hardware(width: Int, height: Int) -> H264HardwareReport {
    var ids: [String] = []
    var list: CFArray? = nil
    if VTCopyVideoEncoderList(nil, &list) == noErr, let encoders = list as? [[String: Any]] {
        let codecKey: String = kVTVideoEncoderList_CodecType as String
        let hwKey: String = kVTVideoEncoderList_IsHardwareAccelerated as String
        let idKey: String = kVTVideoEncoderList_EncoderID as String
        for e in encoders {
            let codec: UInt32 = (e[codecKey] as? NSNumber)?.uint32Value ?? 0
            let hw: Bool = (e[hwKey] as? NSNumber)?.boolValue ?? false
            if codec == kCMVideoCodecType_H264 && hw { ids.append((e[idKey] as? String) ?? "?") }
        }
    }
    let requireKey: String = kVTVideoEncoderSpecification_RequireHardwareAcceleratedVideoEncoder as String
    let spec: [String: Any] = [requireKey: true]
    var session: VTCompressionSession? = nil
    let status: OSStatus = VTCompressionSessionCreate(allocator: nil, width: Int32(width), height: Int32(height),
                                                      codecType: kCMVideoCodecType_H264,
                                                      encoderSpecification: spec as CFDictionary,
                                                      imageBufferAttributes: nil, compressedDataAllocator: nil,
                                                      outputCallback: nil, refcon: nil, compressionSessionOut: &session)
    var usesHardware: Bool = false
    if status == noErr, let s = session {
        // VTSessionCopyProperty returns a +1 reference through a raw pointer.
        let slot = UnsafeMutablePointer<Unmanaged<CFTypeRef>?>.allocate(capacity: 1)
        slot.initialize(to: nil)
        let key: CFString = kVTCompressionPropertyKey_UsingHardwareAcceleratedVideoEncoder
        if VTSessionCopyProperty(s, key: key, allocator: nil, valueOut: UnsafeMutableRawPointer(slot)) == noErr,
           let copied = slot.pointee {
            let value: CFTypeRef = copied.takeRetainedValue()
            usesHardware = (value as? Bool) ?? false
        }
        slot.deinitialize(count: 1)
        slot.deallocate()
        VTCompressionSessionInvalidate(s)
    }
    return H264HardwareReport(hardwareEncoderIDs: ids, sessionCreated: status == noErr && session != nil,
                              sessionUsesHardware: usesHardware)
}
