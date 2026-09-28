// Step 29: step 25's bean flower, with the pollen tube growing — rendered on
// the GPU and saved as a looping GIF.
//
//   .build/bean_growing                 render renders/bean_growing.gif
//   .build/bean_growing still F out.png one frame of the loop, as a PNG
//   .build/bean_growing full out.png [mm]  step 25's still (tube fully grown, or to mm), no caption
//   .build/bean_growing rectcheck [mm]     re-rendering only rectangles vs rendering whole
//
// Environment overrides, so a short run needs no recompile:
//   BEAN_WIDTH, BEAN_HEIGHT  frame size (default 1920 × 1080)
//   BEAN_FRAMES              frames per loop (default 240)
//   BEAN_DELAY               hundredths of a second per frame (default 5)
//   BEAN_SAMPLES             samples per pixel side (default 3, as step 25)
//   BEAN_LIMIT               encode only this many frames (for measuring)
//   BEAN_OUT                 where the GIF goes

import Foundation
import Metal

/// Step 8's GIF encoder names its errors RenderError; here they are BeanError.
typealias RenderError = BeanError

let env: [String: String] = ProcessInfo.processInfo.environment
func envInt(_ name: String, _ fallback: Int) -> Int {
    guard let text: String = env[name], let value = Int(text) else { return fallback }
    return value
}
func note(_ s: String) { FileHandle.standardError.write((s + "\n").data(using: .utf8)!) }

let args: [String] = Array(CommandLine.arguments.dropFirst())
let width: Int = envInt("BEAN_WIDTH", 1920)
let height: Int = envInt("BEAN_HEIGHT", 1080)
let samples: Int = envInt("BEAN_SAMPLES", 3)
let frameCount: Int = envInt("BEAN_FRAMES", 240)
let delayCentiseconds: Int = envInt("BEAN_DELAY", 5)

/// An 8 × 8 Bayer matrix, 0…63.
let bayer: [Int] = [0, 32, 8, 40, 2, 34, 10, 42, 48, 16, 56, 24, 50, 18, 58, 26,
                    12, 44, 4, 36, 14, 46, 6, 38, 60, 28, 52, 20, 62, 30, 54, 22,
                    3, 35, 11, 43, 1, 33, 9, 41, 51, 19, 59, 27, 49, 17, 57, 25,
                    15, 47, 7, 39, 13, 45, 5, 37, 63, 31, 55, 23, 61, 29, 53, 21]

/// How far, in 8-bit levels, the ordered dither nudges each pixel before it
/// is matched to the palette. The first full-size render showed contour bands
/// across the blurred pod and the ground's gradient; a fixed pattern breaks
/// them up, and because it is fixed to the pixel grid a pixel that does not
/// change between frames still does not change — the GIF stays small. MODEL.
let ditherLevels: Float = 6

func dithered(_ bytes: [UInt8]) -> [UInt8] {
    var out: [UInt8] = bytes
    for y in 0..<height {
        for x in 0..<width {
            let d: Float = (Float(bayer[(y & 7) * 8 + (x & 7)]) + 0.5) / 64 - 0.5
            let i: Int = (y * width + x) * 4
            for c in 0..<3 {
                let v: Float = Float(bytes[i + c]) + d * ditherLevels
                out[i + c] = UInt8(min(max(v.rounded(), 0), 255))
            }
        }
    }
    return out
}

func bufferOf(_ bytes: [UInt8], _ device: MTLDevice) throws -> MTLBuffer {
    let made: MTLBuffer? = bytes.withUnsafeBytes { raw in
        device.makeBuffer(bytes: raw.baseAddress!, length: raw.count, options: .storageModeShared)
    }
    guard let b = made else { throw BeanError.gpu("could not allocate a frame buffer") }
    return b
}

func renderGIF(_ loop: Loop, device: MTLDevice, path: String) throws {
    let pixels: Int = width * height
    let limit: Int = envInt("BEAN_LIMIT", 0)
    let count: Int = limit > 0 ? min(limit, frameCount) : frameCount
    let start = Date()

    // One palette for the whole loop, from frames spread across it — the
    // bare grain, the tube part-grown, fully grown, and mid-dissolve — with
    // extra weight on the inset, where the tube's and the pollen's gold is.
    var sampleRGB: [RGB] = []
    let picks: [Int] = [0, frameCount / 6, frameCount / 3, frameCount / 2, loop.timeline.arrivalFrame,
                        (loop.timeline.arrivalFrame + frameCount) / 2, frameCount - 3]
    for f in picks {
        let bytes: [UInt8] = try loop.frame(f)
        let b: MTLBuffer = try bufferOf(bytes, device)
        sampleRGB += samplePixels(b, pixels: pixels, step: 13)
        let place: InsetPlacement = insetPlacement(width: width, height: height)
        let r: Int = Int(place.radius)
        for y in stride(from: max(Int(place.centre.y) - r, 0), to: min(Int(place.centre.y) + r, height), by: 6) {
            for x in stride(from: max(Int(place.centre.x) - r, 0), to: min(Int(place.centre.x) + r, width), by: 6) {
                let i: Int = (y * width + x) * 4
                sampleRGB.append(RGB(bytes[i], bytes[i + 1], bytes[i + 2]))
            }
        }
    }
    let palette: [RGB] = medianCutPalette(sampleRGB, count: 250)
    let quantizer = try Quantizer(device: device, palette: palette, pixels: pixels)
    let gif = GIFWriter(url: URL(fileURLWithPath: path), width: width, height: height,
                        palette: palette, delayCentiseconds: delayCentiseconds)
    for f in 0..<count {
        let b: MTLBuffer = try bufferOf(dithered(try loop.frame(f)), device)
        gif.add(try quantizer.indices(of: b))
        if f % 10 == 0 { note("  frame \(f)/\(count)") }
    }
    try gif.finish()
    let attributes = try? FileManager.default.attributesOfItem(atPath: path)
    let bytes: Int = (attributes?[.size] as? Int) ?? 0
    let mb: Double = Double(bytes) / 1_048_576
    print(String(format: "bean_growing: %d of %d frames, %d × %d, %d spp, %d cs/frame (%.1f s loop) → %@ (%.2f MB, %.0f KB/frame)",
                 count, frameCount, width, height, samples * samples, delayCentiseconds,
                 Double(frameCount * delayCentiseconds) / 100, path, mb, Double(bytes) / 1024 / Double(count)))
    print(String(format: "  tube %.2f mm, arrives on frame %d; growth %.1f s of playback for %.0f h real (%.2f mm/h)",
                 loop.timeline.tubeLength, loop.timeline.arrivalFrame, loop.growthSeconds, realGrowthHours,
                 growthRate(tubeLength: loop.timeline.tubeLength)))
    print(String(format: "  %.1f s GPU, %.1f s wall", loop.frames.gpuSeconds, Date().timeIntervalSince(start)))
}

do {
    let device = try findDevice()
    let scene = try BeanScene(FlowerModel(.none), on: device)
    let frames = try BeanFrames(scene, width: width, height: height, samples: samples)
    let loop = Loop(frames, frameCount: frameCount, delayCentiseconds: delayCentiseconds, mutant: .none)
    let mode: String = args.first ?? "gif"
    switch mode {
    case "still":
        let f: Int = args.count > 1 ? (Int(args[1]) ?? 0) : 0
        let out: String = args.count > 2 ? args[2] : "renders/still.png"
        try savePNG(try loop.frame(f), width: width, height: height, to: URL(fileURLWithPath: out))
        let s: FrameState = loop.timeline.state(f)
        print(String(format: "frame %d: tube %.3f of %.3f mm, fade %.2f → %@", f, s.grown,
                     loop.timeline.tubeLength, s.fade, out))
    case "rectcheck":
        // Grow the tube frame by frame, re-rendering only the rectangles, and
        // compare with rendering the last frame whole.
        var grown: Float = 5
        if args.count > 1, let g = Float(args[1]) { grown = g }
        // In steps, as the loop does, then back down once, to show the order
        // does not matter.
        for k in 0...4 { _ = try frames.render(grown: grown * Float(k) / 4 * 1.2) }
        let a: [UInt8] = try frames.render(grown: grown).bytes
        let whole = try BeanFrames(scene, width: width, height: height, samples: samples)
        let b: [UInt8] = try whole.render(grown: grown).bytes
        var differ: Int = 0
        for i in stride(from: 0, to: a.count, by: 4) where a[i] != b[i] || a[i + 1] != b[i + 1] || a[i + 2] != b[i + 2] {
            differ += 1
            if differ < 20 { print("  differs at x \(i / 4 % width), y \(i / 4 / width): \(a[i]) vs \(b[i])") }
        }
        print("\(differ) pixels differ from a whole render")
    case "full":
        let out: String = args.count > 1 ? args[1] : "renders/full.png"
        var grown: Float = scene.layout.tubeLength
        if args.count > 2, let g = Float(args[2]) { grown = g }
        let start = Date()
        try savePNG(try frames.render(grown: grown).bytes, width: width, height: height,
                    to: URL(fileURLWithPath: out))
        print(String(format: "%.1f s GPU, %.1f s wall → %@", frames.gpuSeconds, Date().timeIntervalSince(start), out))
    default:
        try FileManager.default.createDirectory(atPath: "renders", withIntermediateDirectories: true)
        try renderGIF(loop, device: device, path: env["BEAN_OUT"] ?? "renders/bean_growing.gif")
    }
} catch {
    note("bean_growing: \(error)")
    exit(1)
}
