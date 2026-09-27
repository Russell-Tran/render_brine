// Step 21: step 20's lower arch, with the camera travelling along it.
//
//   .build/arch_pan [width height samples-per-side [frames [from]]]  → renders/arch_pan.gif
//   .build/arch_pan stills [width height samples-per-side]           → a few frames as PNGs
//
// `frames` and `from` render a stretch of the loop only, to measure what a
// frame costs in the GIF before committing to a size.
//
// Every frame is step 20's renderer with a different camera; nothing else
// changes. The GIF encoder is step 8's.

import Foundation
import Metal

/// Step 8's GIF encoder names its errors this way.
enum RenderError: Error {
    case kernelCompile(String)
    case gpu(String)
}

let args: [String] = Array(CommandLine.arguments.dropFirst())
let firstArgument: String = args.first ?? ""
let stills: Bool = firstArgument == "stills"
let numbers: [String] = stills ? Array(args.dropFirst()) : args

func argument(_ n: Int, _ fallback: Int) -> Int {
    guard n < numbers.count, let v = Int(numbers[n]) else { return fallback }
    return v
}

let width: Int = argument(0, 544)
let height: Int = argument(1, 306)
let samples: Int = argument(2, 3)
let from: Int = min(max(argument(4, 0), 0), frameCount - 1)
let limit: Int = min(argument(3, frameCount), frameCount - from)
let pixelCount: Int = width * height

// Why 544 × 306, and no dithering. Step 8's encoder saves space by storing
// only the box around the pixels that changed since the last frame. With the
// camera moving, that box is the whole frame, every frame, so the GIF costs
// roughly (pixels per frame) × (moving frames). Measured on this loop when it
// ended on #18: 640 × 360 costs ~77 kB per moving frame, which would make the
// loop ~15 MB; at 576 × 324, 20 frames a second, a 4 × 4 ordered dither
// against banding made it 11.0 MB and dropping the dither gave back 14%. Side
// by side the dither's gain on the gum's gradient was barely visible, so it
// went, and 576 × 324 came to 9.5 MB.
//
// Travelling on to #17 adds 10 mm of arch. Measured again, ten frames at a
// time: at 576 × 324 a moving frame averages ~58 kB, so 8.5 s of travel (191
// changing frames with the dissolve) would be ~11 MB; at 544 × 306 ten frames
// cost 0.91× as much. 8 s of travel at 544 × 306 — 181 changing frames —
// comes to under 10 MB with the frame only 6% narrower.

/// Blend a frame toward frame 0, in display values, as step 19 does.
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

func frame(_ f: Int, on device: MTLDevice) throws -> (image: MouthImage, gpu: Double) {
    let cam: Camera = railCamera(plan(f).arc)
    let r = try renderMouth(width: width, height: height, samples: samples, camera: cam,
                             retromolarPad: true, thirdMolars: withThirdMolars(), on: device)
    return (r.image, r.gpuSeconds)
}

do {
    let device = try findDevice()
    let start = Date()
    if stills {
        let picks: [Int] = [0, travelFrames / 4, travelFrames / 2, 3 * travelFrames / 4, travelFrames,
                            travelFrames + holdFrames + dissolveFrames / 2]
        var first: [UInt8] = []
        for f in picks {
            let (image, _) = try frame(f, on: device)
            if f == 0 { first = snapshot(image.pixels) }
            let w: Float = plan(f).dissolve
            if w > 0 { dissolve(image.pixels, toward: first, weight: w) }
            let path: String = String(format: "renders/frame%03d.png", f)
            try savePNG(image, to: URL(fileURLWithPath: path))
            print(String(format: "frame %3d  arc %5.2f mm  dissolve %.2f → %@", f, plan(f).arc, w, path))
        }
        exit(0)
    }

    guard let work = device.makeBuffer(length: pixelCount * 4, options: .storageModeShared) else {
        throw RenderError.gpu("could not allocate the frame buffer")
    }

    // One palette for the whole loop, from frames spread across the travel —
    // and from the dissolve's halfway frame. The dissolve lays gum over
    // background and teeth over gum, colours no single frame contains; a
    // palette drawn from the travel alone posterised it into blotches.
    var sampled: [RGB] = []
    var firstFrame: [UInt8] = []
    for k in 0..<10 {
        let (image, _) = try frame(k * travelFrames / 9, on: device)
        sampled += samplePixels(image.pixels, pixels: pixelCount, step: 7)
        if k == 0 { firstFrame = snapshot(image.pixels) }
        if k == 9 {
            dissolve(image.pixels, toward: firstFrame, weight: 0.5)
            sampled += samplePixels(image.pixels, pixels: pixelCount, step: 3)
        }
    }
    let palette: [RGB] = medianCutPalette(sampled, count: 250)
    let quantizer = try Quantizer(device: device, palette: palette, pixels: pixelCount)
    let path = "renders/arch_pan.gif"
    let gif = GIFWriter(url: URL(fileURLWithPath: path), width: width, height: height,
                        palette: palette, delayCentiseconds: frameDelayCentiseconds)
    var gpu: Double = 0
    var pixels: [UInt8] = []
    for f in from..<(from + limit) {
        let p: FramePlan = plan(f)
        // After the travel the camera is still, so the frame on #17 is
        // rendered once and every later frame starts from it.
        if f <= travelFrames || pixels.isEmpty {
            let (image, seconds) = try frame(f, on: device)
            gpu += seconds
            pixels = snapshot(image.pixels)
        }
        pixels.withUnsafeBytes { work.contents().copyMemory(from: $0.baseAddress!, byteCount: pixelCount * 4) }
        if p.dissolve > 0 { dissolve(work, toward: firstFrame, weight: p.dissolve) }
        gif.add(try quantizer.indices(of: work))
        if f % 25 == 0 { print("  frame \(f)/\(limit)") }
    }
    try gif.finish()
    let bytes: Int = ((try? FileManager.default.attributesOfItem(atPath: path))?[.size] as? Int) ?? 0
    let wall: Double = Date().timeIntervalSince(start)
    print("GPU: \(device.name)")
    print(String(format: "%d frames of %d, %d × %d, %d samples per pixel, %d cs a frame, %.1f s loop",
                 limit, frameCount, width, height, samples * samples, frameDelayCentiseconds, loopSeconds))
    print(String(format: "%.1f s on the GPU, %.0f s wall → %@ (%.2f MB)",
                 gpu, wall, path, Double(bytes) / 1_048_576))
} catch {
    FileHandle.standardError.write("arch_pan: \(error)\n".data(using: .utf8)!)
    exit(1)
}
