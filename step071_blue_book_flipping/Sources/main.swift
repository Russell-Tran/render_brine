// Step 71: step 70's blue hardback opens, turns its pages on its own, and
// closes on its other side; then dissolves forward to the start.
//
//   .build/flip [width height samples-per-side [frames [from]]]  → renders/blue_book_flipping.gif
//   .build/flip stills [width height samples-per-side]           → a few frames as PNGs
//
// `frames` and `from` render a stretch of the loop only, to measure what a
// frame costs in the GIF before committing to a size. The GIF encoder is
// step 8's.

import Foundation
import Metal

/// Step 8's GIF encoder names its errors this way.
enum RenderError: Error {
    case kernelCompile(String)
    case gpu(String)
}

let args: [String] = Array(CommandLine.arguments.dropFirst())
let firstArgument: String = args.first ?? ""
let stillsWord: String = "stills"
let stills: Bool = firstArgument.elementsEqual(stillsWord)
let numbers: [String] = stills ? Array(args.dropFirst()) : args

func argument(_ n: Int, _ fallback: Int) -> Int {
    guard n < numbers.count, let v = Int(numbers[n]) else { return fallback }
    return v
}

let width: Int = argument(0, 640)
let height: Int = argument(1, 360)
let samples: Int = argument(2, 3)
let total: Int = frameCount
let from: Int = min(max(argument(4, 0), 0), total - 1)
let limit: Int = min(argument(3, total), total - from)
let pixelCount: Int = width * height

/// Blend a frame toward frame 0, in display values, as step 21 does.
func dissolve(_ buffer: MTLBuffer, toward first: [UInt8], weight: Float) {
    let p = buffer.contents().assumingMemoryBound(to: UInt8.self)
    for i in 0..<(pixelCount * 4) {
        let a: Float = Float(p[i])
        let b: Float = Float(first[i])
        let v: Float = a + (b - a) * weight
        p[i] = UInt8(max(min(v.rounded(), 255), 0))
    }
}

func snapshot(_ buffer: MTLBuffer) -> [UInt8] {
    let p = buffer.contents().assumingMemoryBound(to: UInt8.self)
    return Array(UnsafeBufferPointer(start: p, count: pixelCount * 4))
}

func time(ofFrame f: Int) -> Float { Float(f) / framesPerSecond }

if activeMutant != .none {
    FileHandle.standardError.write("flip: a mutant is set; this loop is broken on purpose\n".data(using: .utf8)!)
}

do {
    let device = try findDevice()
    let s: Shot = shot()
    let renderer = try BookRenderer(device: device, width: width, height: height, studio: s.studio)
    func frame(_ f: Int) throws -> Double {
        let st: BookState = bookState(at: time(ofFrame: f))
        let g: Double = try renderer.render(buildScene(st), camera: s.camera, samples: samples)
        annotate(renderer.image.pixels, width: width, height: height, shot: s)
        return g
    }
    let start = Date()
    try? FileManager.default.createDirectory(atPath: "renders", withIntermediateDirectories: true)
    if stills {
        func at(_ t: Float) -> Int { min(Int((t * framesPerSecond).rounded()), total - 1) }
        let riffleSpan: Float = riffleEnd - riffleStart
        let picks: [Int] = [0, at(holdStart + openDuration / 2), at(riffleStart), at(riffleStart + riffleSpan * 0.25),
                            at(riffleStart + riffleSpan * 0.5), at(riffleStart + riffleSpan * 0.75), at(riffleEnd),
                            at(closeStart + closeDuration * 0.45), at(closeEnd), at(fadeStart + fadeDuration / 2), total - 1]
        var first: [UInt8] = []
        for f in picks where f < total {
            _ = try frame(f)
            if f == 0 { first = snapshot(renderer.image.pixels) }
            let w: Float = bookState(at: time(ofFrame: f)).fade
            if w > 0 { dissolve(renderer.image.pixels, toward: first, weight: w) }
            let path: String = String(format: "renders/frame%03d.png", f)
            try savePNG(renderer.image, to: URL(fileURLWithPath: path))
            print(String(format: "frame %3d  t %.2f s  fade %.2f → %@", f, time(ofFrame: f), w, path))
        }
        exit(0)
    }

    guard let work = device.makeBuffer(length: pixelCount * 4, options: .storageModeShared) else {
        throw RenderError.gpu("could not allocate the frame buffer")
    }
    // One palette for the whole loop, from frames spread across it (closed,
    // opening, riffling, closing) and the dissolve's halfway blend.
    var sampled: [RGB] = []
    var firstFrame: [UInt8] = []
    let paletteFrames: [Int] = (0..<10).map { $0 * (total - 1) / 9 }
    for (i, f) in paletteFrames.enumerated() {
        _ = try frame(f)
        sampled += samplePixels(renderer.image.pixels, pixels: pixelCount, step: 5)
        if i == 0 { firstFrame = snapshot(renderer.image.pixels) }
    }
    // The dissolve blends the last frame's blue cover into the table and
    // frame 0's cover into the table: sample the blends it passes through.
    _ = try frame(Int(fadeStart * framesPerSecond))
    let held: [UInt8] = snapshot(renderer.image.pixels)
    for w in [Float(0.12), 0.3, 0.5, 0.7, 0.88] {
        held.withUnsafeBytes { renderer.image.pixels.contents().copyMemory(from: $0.baseAddress!, byteCount: pixelCount * 4) }
        dissolve(renderer.image.pixels, toward: firstFrame, weight: w)
        sampled += samplePixels(renderer.image.pixels, pixels: pixelCount, step: 3)
    }
    let palette: [RGB] = medianCutPalette(sampled, count: 250)
    let quantizer = try Quantizer(device: device, palette: palette, pixels: pixelCount)
    let path = "renders/blue_book_flipping.gif"
    let gif = GIFWriter(url: URL(fileURLWithPath: path), width: width, height: height,
                        palette: palette, delayCentiseconds: frameDelayCentiseconds)
    var gpu: Double = 0
    var last: [UInt8] = []
    let fadeFrame: Int = Int((fadeStart * framesPerSecond).rounded(.up))
    for f in from..<(from + limit) {
        let st: BookState = bookState(at: time(ofFrame: f))
        // In the hold and the dissolve the book is still: render once.
        if f <= fadeFrame || last.isEmpty {
            gpu += try frame(f)
            last = snapshot(renderer.image.pixels)
        }
        last.withUnsafeBytes { work.contents().copyMemory(from: $0.baseAddress!, byteCount: pixelCount * 4) }
        if st.fade > 0 { dissolve(work, toward: firstFrame, weight: st.fade) }
        gif.add(try quantizer.indices(of: work))
        if f % 20 == 0 { print("  frame \(f)/\(total)") }
    }
    try gif.finish()
    let bytes: Int = ((try? FileManager.default.attributesOfItem(atPath: path))?[.size] as? Int) ?? 0
    let wall: Double = Date().timeIntervalSince(start)
    print("GPU: \(device.name)")
    print(String(format: "%d frames of %d, %d × %d, %d samples per pixel, %d cs a frame, %.2f s loop",
                 limit, total, width, height, samples * samples, frameDelayCentiseconds, loopSeconds))
    print(String(format: "%.1f s on the GPU, %.0f s wall → %@ (%d bytes, %.2f MB)",
                 gpu, wall, path, bytes, Double(bytes) / 1_048_576))
} catch {
    FileHandle.standardError.write("flip: \(error)\n".data(using: .utf8)!)
    exit(1)
}
