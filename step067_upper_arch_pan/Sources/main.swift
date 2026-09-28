// Step 67: step 66's upper arch, with the camera travelling along it.
//
//   .build/upper_arch_pan [width height samples-per-side [frames [from]]]  → renders/upper_arch_pan.gif
//   .build/upper_arch_pan stills [width height samples-per-side]           → a few frames as PNGs
//
// `frames` and `from` render a stretch of the loop only, to measure what a
// frame costs in the GIF before committing to a size (step 21's practice).
//
// Every frame is step 66's renderer with a different camera, compiled once.
// The GIF encoder is step 8's.

import Foundation
import Metal

/// Step 8's GIF encoder names its errors this way.
enum RenderError: Error {
    case kernelCompile(String)
    case gpu(String)
}

let args: [String] = Array(CommandLine.arguments.dropFirst())
let firstArgument: String = args.first ?? ""
// A method, not `==`: the operator's many overloads were measured at the 5 ms
// type-check limit on this line, with a literal and without.
let stillsWord: String = "stills"
let stills: Bool = firstArgument.elementsEqual(stillsWord)
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

// Why 544 × 306, 20 frames a second, no dithering: step 21's measurements on
// the lower arch, where every pixel changes every frame and the GIF costs
// about (bytes per frame) × (moving frames). Measured again here, ten frames
// at a time, before rendering the loop: see the step's commit message.

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

if shotMutant != .none || archMutant != .none {
    FileHandle.standardError.write("upper_arch_pan: a mutant is set; this loop is broken on purpose\n".data(using: .utf8)!)
}

do {
    let device = try findDevice()
    let renderer = try MouthRenderer(on: device)
    func frame(_ f: Int) throws -> (image: MouthImage, gpu: Double) {
        let r = try renderer.render(width: width, height: height, samples: samples, camera: frameCamera(f))
        return (r.image, r.gpuSeconds)
    }
    let start = Date()
    if stills {
        let picks: [Int] = [0, travelFrames / 4, travelFrames / 2, 3 * travelFrames / 4, travelFrames,
                            travelFrames + holdFrames + dissolveFrames / 2]
        var first: [UInt8] = []
        for f in picks {
            let (image, _) = try frame(f)
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

    // One palette for the whole loop, from ten frames spread across the travel
    // — the first and the last, on #16, among them — and from the dissolve's
    // halfway frame, whose blends of gum over background no single frame has.
    var sampled: [RGB] = []
    var firstFrame: [UInt8] = []
    for k in 0..<10 {
        let (image, _) = try frame(k * travelFrames / 9)
        sampled += samplePixels(image.pixels, pixels: pixelCount, step: 7)
        if k == 0 { firstFrame = snapshot(image.pixels) }
        if k == 9 {
            dissolve(image.pixels, toward: firstFrame, weight: 0.5)
            sampled += samplePixels(image.pixels, pixels: pixelCount, step: 3)
        }
    }
    let palette: [RGB] = medianCutPalette(sampled, count: 250)
    let quantizer = try Quantizer(device: device, palette: palette, pixels: pixelCount)
    let path = "renders/upper_arch_pan.gif"
    try? FileManager.default.createDirectory(atPath: "renders", withIntermediateDirectories: true)
    let gif = GIFWriter(url: URL(fileURLWithPath: path), width: width, height: height,
                        palette: palette, delayCentiseconds: frameDelayCentiseconds)
    var gpu: Double = 0
    var pixels: [UInt8] = []
    for f in from..<(from + limit) {
        let p: FramePlan = plan(f)
        // After the travel the camera is still, so the frame on #16 is
        // rendered once and every later frame starts from it.
        if f <= travelFrames || pixels.isEmpty {
            let (image, seconds) = try frame(f)
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
    print(String(format: "%d frames of %d, %d × %d, %d samples per pixel, %d cs a frame, %.2f s loop",
                 limit, frameCount, width, height, samples * samples, frameDelayCentiseconds, loopSeconds))
    print(String(format: "%.1f s on the GPU, %.0f s wall → %@ (%d bytes, %.2f MB)",
                 gpu, wall, path, bytes, Double(bytes) / 1_048_576))
} catch {
    FileHandle.standardError.write("upper_arch_pan: \(error)\n".data(using: .utf8)!)
    exit(1)
}
