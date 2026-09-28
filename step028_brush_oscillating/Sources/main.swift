// Step 28: step 22's still — same mouth, camera, light and press — with the
// round head oscillating about its own axis as a real one does, slowed ~290×,
// as a GIF that loops forward forever.
//
//   .build/brush_oscillating [width height samples-per-side [frames [from]]]
//        → renders/brush_oscillating.gif
//   .build/brush_oscillating stills [width height samples-per-side]
//        → a few frames as PNGs, to look at
//
// `frames` and `from` encode a stretch of the loop only, to measure what a
// frame costs in the GIF before committing to a size.

import Foundation
import Metal

/// Step 8's GIF encoder names its errors this way.
enum RenderError: Error {
    case kernelCompile(String)
    case gpu(String)
}

let args: [String] = Array(CommandLine.arguments.dropFirst())
let firstArgument: String = args.first ?? ""
let stills: Bool = firstArgument.elementsEqual("stills")
let numbers: [String] = stills ? Array(args.dropFirst()) : args

func argument(_ n: Int, _ fallback: Int) -> Int {
    guard n < numbers.count, let v = Int(numbers[n]) else { return fallback }
    return v
}

let width: Int = argument(0, 1920)
let height: Int = argument(1, 1080)
let samples: Int = argument(2, 4)
let from: Int = min(max(argument(4, 0), 0), frameCount - 1)
let limit: Int = min(argument(3, frameCount), frameCount - from)
let pixelCount: Int = width * height

// Why 1920 × 1080, 80 frames at 5 cs, and no dithering. Step 8's encoder
// stores only the box round the pixels that changed since the last frame, and
// here only the disc, its tufts and their shadows move — 120,000–150,000
// pixels a frame, about 7% of it. Measured first on ten frames at full size
// and the still's own 16 samples a pixel: 0.80 MB, most of it the first,
// whole frame — so a whole loop would be ~4 MB, well inside 10 MB, and there
// was no reason to shrink the frame or thin the frames. It came to 4.22 MB.
// 20 frames a second keeps the turn to 1.8° a frame, smooth at this pace.
// The still's banding on the enamel is the shared palette's (step 21 made
// the same trade); dithering would have cost size for little visible gain.
//
// The cost is the rendering: ~35 s a frame on the M4 mini for the window,
// 49 minutes for the loop — the brush's 25 tufts are marched at every step
// inside its bound.

func copyPixels(_ from: MTLBuffer, _ to: MTLBuffer) {
    to.contents().copyMemory(from: from.contents(), byteCount: pixelCount * 4)
}

do {
    let device = try findDevice()
    let start = Date()
    let probe = try SceneProbe(device: device)
    let oscillating = try OscillatingBrush(probe: probe)

    // Step 22's still, rendered whole once: every frame starts from it.
    let (stillImage, stillGPU) = try renderMouth(width: width, height: height, samples: samples,
                                                 extra: brushExtra(oscillating.rest), on: device)
    let window: PixelWindow = try movingWindow(brush: oscillating, width: width, height: height, on: device)
    let windowPixels: Int = window.x1 * (window.y1 - window.y0)
    let windowShare: Double = Double(windowPixels) / Double(pixelCount) * 100
    print(String(format: "moving window: columns 0–%d, rows %d–%d (%.0f%% of the frame)",
                 window.x1 - 1, window.y0, window.y1 - 1, windowShare))
    guard let work = device.makeBuffer(length: pixelCount * 4, options: .storageModeShared),
          let aux = device.makeBuffer(length: pixelCount * 16, options: .storageModeShared)
    else { throw RenderError.gpu("could not allocate the frame buffer") }
    var gpu: Double = stillGPU
    var freeFrames: Int = 0

    func frame(_ f: Int) throws {
        let b: Brush = try oscillating.brush(frame: f)
        if b.tufts.contains(where: { $0.free }) { freeFrames += 1 }
        copyPixels(stillImage.pixels, work)
        gpu += try renderWindow(window, width: width, height: height, samples: samples,
                                extra: brushExtra(b), into: work, aux: aux, on: device)
        drawCaption(work, width: width, height: height)
    }

    if stills {
        for f in [0, frameCount / 8, frameCount / 4, 3 * frameCount / 8, frameCount / 2, 5 * frameCount / 8,
                  3 * frameCount / 4] {
            try frame(f)
            let path: String = String(format: "renders/frame%03d.png", f)
            try savePNG(MouthImage(width: width, height: height, pixels: work, aux: aux),
                        to: URL(fileURLWithPath: path))
            print(String(format: "frame %3d  turn %+6.2f°  → %@", f, spinAngle(frame: f) * 180 / Float.pi, path))
        }
        exit(0)
    }

    // Render every frame once, keeping only the window's rows.
    let rowBytes: Int = width * 4
    var windows: [[UInt8]] = []
    for f in from..<(from + limit) {
        try frame(f)
        let p = work.contents().assumingMemoryBound(to: UInt8.self)
        windows.append(Array(UnsafeBufferPointer(start: p + window.y0 * rowBytes,
                                                 count: (window.y1 - window.y0) * rowBytes)))
        if f % 10 == 0 { print(String(format: "  frame %d/%d  turn %+.2f°", f, frameCount, spinAngle(frame: f) * 180 / Float.pi)) }
    }
    func load(_ k: Int) {
        copyPixels(stillImage.pixels, work)
        drawCaption(work, width: width, height: height)
        windows[k].withUnsafeBytes {
            work.contents().advanced(by: window.y0 * rowBytes).copyMemory(from: $0.baseAddress!, byteCount: $0.count)
        }
    }

    // One palette for the whole loop: the still whole, and the window from
    // frames spread across the swing — the tufts show different faces, and
    // their shadows fall differently, at each end.
    load(0)
    var sampled: [RGB] = samplePixels(work, pixels: pixelCount, step: 7)
    for k in stride(from: 0, to: windows.count, by: max(windows.count / 10, 1)) {
        load(k)
        let slice: Int = window.y0 * width
        guard let tmp = device.makeBuffer(bytes: work.contents().advanced(by: slice * 4),
                                          length: (window.y1 - window.y0) * rowBytes, options: .storageModeShared)
        else { throw RenderError.gpu("could not allocate a sample buffer") }
        sampled += samplePixels(tmp, pixels: (window.y1 - window.y0) * width, step: 5)
    }
    let palette: [RGB] = medianCutPalette(sampled, count: 250)
    let quantizer = try Quantizer(device: device, palette: palette, pixels: pixelCount)
    let path = "renders/brush_oscillating.gif"
    let gif = GIFWriter(url: URL(fileURLWithPath: path), width: width, height: height,
                        palette: palette, delayCentiseconds: frameDelayCentiseconds)
    for k in 0..<windows.count {
        load(k)
        gif.add(try quantizer.indices(of: work))
    }
    try gif.finish()
    let bytes: Int = ((try? FileManager.default.attributesOfItem(atPath: path))?[.size] as? Int) ?? 0
    let wall: Double = Date().timeIntervalSince(start)
    print("GPU: \(device.name)")
    print(String(format: "%d frames of %d, %d × %d, %d samples per pixel, %d cs a frame, %.1f s loop, slowed %.0f×",
                 limit, frameCount, width, height, samples * samples, frameDelayCentiseconds, loopSeconds, slowdown))
    print("\(freeFrames) of \(limit) frames have a tuft standing free over the #21/#22 gap")
    print(String(format: "%.1f s on the GPU, %.0f s wall → %@ (%.2f MB)",
                 gpu, wall, path, Double(bytes) / 1_048_576))
} catch {
    FileHandle.standardError.write("brush_oscillating: \(error)\n".data(using: .utf8)!)
    exit(1)
}
