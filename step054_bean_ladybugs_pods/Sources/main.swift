// Step 54: step 45's pole bean in flower, with seven-spot ladybirds walking
// on it. The climb is a time-lapse; the flowers and pods are a gradient along
// the stem by node age, not a time-lapse of a pod growing; the ladybirds run
// in real time on their own clock.
//
//   .build/bean_ladybugs_pods [width height samples-per-side [frames [colours]]] → renders/bean_ladybugs_pods.gif
//   .build/bean_ladybugs_pods stills [width height samples-per-side]             → renders/frameNNN.png
//
// `frames` renders only the first so many frames, to measure what a frame
// costs in the GIF before committing to a size. `make run` copies the finished
// loop to ../showcase/bean_ladybugs_pods.gif.
//
// Why 480 × 624 at 20 frames a second: step 45's size and rate, so the two
// films match. The camera rises every frame, so every pixel moves and step 8's
// encoder stores the whole frame every time; the GIF costs about (bytes per
// frame) × (frames). See the measurements in the colour note below.

import Foundation
import Metal

let args: [String] = Array(CommandLine.arguments.dropFirst())
let firstArgument: String = args.first ?? ""
let stillsCommand: String = "stills"
/// Colours in the GIF's one palette. Ten frames measured first, as in steps
/// 47 and 48: 23.7 kB a frame at 80 colours, 30.5 kB at 160, 35.7 kB at 250 —
/// about 7 MB for the 200-frame loop even at the full 250, inside the 10 MB
/// the showcase allows (this film has less fine detail than step 48's pea).
/// So the full palette, drawn from ten frames spread across the loop, for the
/// white petals' and the red elytra's gradients.
let gifColours: Int = 250
let stills: Bool = firstArgument.elementsEqual(stillsCommand)
var numbers: [String] = args
if stills { numbers.removeFirst() }

func argument(_ n: Int, _ fallback: Int) -> Int {
    guard n < numbers.count, let v = Int(numbers[n]) else { return fallback }
    return v
}

let width: Int = argument(0, 480)
let height: Int = argument(1, 624)
let samples: Int = argument(2, 2)
let limit: Int = min(argument(3, frameCount), frameCount)
let paletteColours: Int = min(max(argument(4, gifColours), 16), 250)

func captions(width: Int, height: Int) -> [Caption] {
    let k: Float = Float(height) / 520
    let h: Float = Float(height)
    return [
        Caption(text: nameCaption, x: 14 * k, baseline: 24 * k, size: 15 * k, italic: true),
        Caption(text: ladybirdNameCaption, x: 14 * k, baseline: 39 * k, size: 10.5 * k),
        Caption(text: selfCaption, x: 14 * k, baseline: 52 * k, size: 10.5 * k),
        Caption(text: aphidCaption, x: 14 * k, baseline: 65 * k, size: 10.5 * k),
        Caption(text: gradientCaption, x: 14 * k, baseline: h - 55 * k, size: 10.5 * k),
        Caption(text: compressedCaption, x: 14 * k, baseline: h - 42 * k, size: 10.5 * k),
        Caption(text: timeCaption(), x: 14 * k, baseline: h - 27 * k, size: 10.5 * k, alpha: 0.85),
        Caption(text: clockCaptionLine(), x: 14 * k, baseline: h - 13 * k, size: 10.5 * k, alpha: 0.85),
    ]
}

/// The plant faded behind the caption lines at the top and bottom, as in step
/// 48 — the pods and leaves hang where the words are. Fixed to the frame, so
/// it cannot give the loop away.
func veiled(_ f: Frame) -> Frame {
    let k: Float = Float(f.height) / 520
    let h: Float = Float(f.height)
    var px: [SIMD4<Float>] = f.rgba
    for y in 0..<f.height {
        let fy: Float = Float(y)
        let top: Float = 1 - smoothstep(68 * k, 80 * k, fy)
        let bottom: Float = smoothstep(h - 70 * k, h - 58 * k, fy)
        let keep: Float = 1 - 0.62 * max(top, bottom)
        if keep >= 1 { continue }
        for x in 0..<f.width { px[y * f.width + x] *= keep }
    }
    return Frame(width: f.width, height: f.height, rgba: px, seen: f.seen)
}

func render(frame f: Int, with r: PlantRenderer) throws -> (bytes: [UInt8], gpu: Double) {
    let t: Float = hourOf(frame: f)
    let cam: Camera = camera(t)
    let scene: PlantScene = fullScene(t, .none)
    // Legs are 0.24–0.32 mm across, under half a pixel here, so they are
    // held to at least half a pixel's radius (as step 48 held its tendrils)
    // and do not flicker in and out.
    let mmPerPixel: Float = cam.millimetresPerPixel(at: cam.position + cam.forward * cameraDistance, height: height)
    let (image, gpu) = try r.render(scene, camera: cam, width: width, height: height, samples: samples,
                                    tubeMin: 0.5 * mmPerPixel)
    // The magnified inset (Inset.swift), drawn over the veiled picture.
    let (framed, insetCaptions) = try withInset(veiled(image), t, scene: scene, renderer: r, samples: samples + 1)
    return (finishFrame(framed, ground: ground, captions: captions(width: width, height: height) + insetCaptions), gpu)
}

/// Picture.swift's loop writer with the palette's size as a parameter (step
/// 48's variant).
func writeLoopGIF(path: String, width: Int, height: Int, frames: Int, delayCentiseconds: Int,
                  paletteFrames: [Int], colours: Int, device: MTLDevice, frame: (Int) throws -> [UInt8]) throws -> Int {
    let pixelCount: Int = width * height
    guard let work = device.makeBuffer(length: pixelCount * 4, options: .storageModeShared) else {
        throw RenderError.gpu("could not allocate the frame buffer")
    }
    var cache: [Int: [UInt8]] = [:]
    var sampled: [RGB] = []
    for f in paletteFrames {
        let b: [UInt8] = try frame(f)
        cache[f] = b
        b.withUnsafeBytes { work.contents().copyMemory(from: $0.baseAddress!, byteCount: pixelCount * 4) }
        sampled += samplePixels(work, pixels: pixelCount, step: 5)
    }
    let palette: [RGB] = medianCutPalette(sampled, count: colours)
    let quantizer = try Quantizer(device: device, palette: palette, pixels: pixelCount)
    let gif = GIFWriter(url: URL(fileURLWithPath: path), width: width, height: height,
                        palette: palette, delayCentiseconds: delayCentiseconds)
    for f in 0..<frames {
        let b: [UInt8] = try cache[f] ?? frame(f)
        cache[f] = nil
        b.withUnsafeBytes { work.contents().copyMemory(from: $0.baseAddress!, byteCount: pixelCount * 4) }
        gif.add(try quantizer.indices(of: work))
        if f % 20 == 0 { print("  frame \(f)/\(frames)") }
    }
    try gif.finish()
    return ((try? FileManager.default.attributesOfItem(atPath: path))?[.size] as? Int) ?? 0
}

do {
    let device = try findDevice()
    let start = Date()
    let renderer = try PlantRenderer(look: lookFor(.none), on: device)
    if stills {
        for f in [0, frameCount / 3, frameCount / 2, 2 * frameCount / 3, frameCount - 1] {
            let (bytes, gpu) = try render(frame: f, with: renderer)
            let path: String = String(format: "renders/frame%03d.png", f)
            try savePNG(bytes, width: width, height: height, to: URL(fileURLWithPath: path))
            print(String(format: "frame %3d  hour %5.2f  %.2f s GPU → %@", f, hourOf(frame: f), gpu, path))
        }
        exit(0)
    }
    var gpu: Double = 0
    let path: String = limit == frameCount ? "renders/bean_ladybugs_pods.gif" : "renders/measure.gif"
    // Palette: frames spread evenly across the loop.
    let picks: [Int] = (0..<10).map { $0 * limit / 10 }
    let bytes: Int = try writeLoopGIF(path: path, width: width, height: height, frames: limit,
                                      delayCentiseconds: frameDelayCentiseconds, paletteFrames: picks,
                                      colours: paletteColours, device: device) { f in
        let (b, g) = try render(frame: f, with: renderer)
        gpu += g
        return b
    }
    let wall: Double = Date().timeIntervalSince(start)
    print("GPU: \(device.name)")
    print("\(paletteColours) colours")
    print(String(format: "%d frames of %d, %d × %d, %d samples per pixel, %d cs a frame, %.1f s loop = %.2f h of growth",
                 limit, frameCount, width, height, samples * samples, frameDelayCentiseconds, loopSeconds, loopHours))
    print(String(format: "%.1f s on the GPU, %.0f s wall → %@ (%.2f MB, %.1f kB a frame)",
                 gpu, wall, path, Double(bytes) / 1_048_576, Double(bytes) / 1024 / Double(limit)))
} catch {
    FileHandle.standardError.write("bean_ladybugs_pods: \(error)\n".data(using: .utf8)!)
    exit(1)
}
