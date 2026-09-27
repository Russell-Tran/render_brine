// Step 45: a pole bean climbing its pole by twining — a time-lapse loop with
// the camera rising with the growing tip.
//
//   .build/bean_twining [width height samples-per-side [frames]]  → ../showcase/bean_twining.gif
//   .build/bean_twining stills [width height samples-per-side]    → renders/frameNNN.png
//
// `frames` renders only the first so many frames, to measure what a frame
// costs in the GIF before committing to a size.
//
// Why 480 × 624 at 20 frames a second. The camera rises every frame, so every
// pixel of the picture moves and step 8's encoder, which stores the box round
// the pixels that changed, stores the whole frame every time (the step 14
// lesson). The GIF costs about (bytes per frame) × (frames), so ten frames
// were measured first: 23.3 kB a frame at 400 × 520, 30.5 kB at 480 × 624.
// At 480 × 624 a 10 s loop of 200 frames comes to about 6 MB, inside the
// 10 MB the showcase allows, so the loop can have both the larger frame and
// 20 frames a second rather than 15, for a smoother rise.

import Foundation
import Metal

let args: [String] = Array(CommandLine.arguments.dropFirst())
let firstArgument: String = args.first ?? ""
let stillsCommand: String = "stills"
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

func captions(width: Int, height: Int) -> [Caption] {
    let k: Float = Float(height) / 520
    let h: Float = Float(height)
    return [
        Caption(text: "Phaseolus vulgaris", x: 14 * k, baseline: 26 * k, size: 16 * k, italic: true),
        Caption(text: "pole bean, climbing by twining", x: 14 * k, baseline: 43 * k, size: 12 * k),
        Caption(text: "the stem winds anticlockwise, seen from above", x: 14 * k, baseline: h - 30 * k, size: 11.5 * k),
        Caption(text: timeCaption(), x: 14 * k, baseline: h - 13 * k, size: 11.5 * k, alpha: 0.8),
    ]
}

func render(frame f: Int, with r: PlantRenderer) throws -> (bytes: [UInt8], gpu: Double) {
    let t: Float = hourOf(frame: f)
    let cam: Camera = camera(t)
    let scene: PlantScene = beanScene(t, .none)
    let (image, gpu) = try r.render(scene, camera: cam, width: width, height: height, samples: samples, tubeMin: 0)
    return (finishFrame(image, ground: ground, captions: captions(width: width, height: height)), gpu)
}

do {
    let device = try findDevice()
    let start = Date()
    let renderer = try PlantRenderer(look: look, on: device)
    if stills {
        for f in [0, frameCount / 3, 2 * frameCount / 3, frameCount - 1] {
            let (bytes, gpu) = try render(frame: f, with: renderer)
            let path: String = String(format: "renders/frame%03d.png", f)
            try savePNG(bytes, width: width, height: height, to: URL(fileURLWithPath: path))
            print(String(format: "frame %3d  hour %5.2f  %.2f s GPU → %@", f, hourOf(frame: f), gpu, path))
        }
        exit(0)
    }
    var gpu: Double = 0
    let path: String = limit == frameCount ? "../showcase/bean_twining.gif" : "renders/measure.gif"
    // Palette: frames spread evenly across the loop.
    let picks: [Int] = (0..<10).map { $0 * limit / 10 }
    let bytes: Int = try writeLoopGIF(path: path, width: width, height: height, frames: limit,
                                      delayCentiseconds: frameDelayCentiseconds, paletteFrames: picks,
                                      device: device) { f in
        let (b, g) = try render(frame: f, with: renderer)
        gpu += g
        return b
    }
    let wall: Double = Date().timeIntervalSince(start)
    print("GPU: \(device.name)")
    print(String(format: "%d frames of %d, %d × %d, %d samples per pixel, %d cs a frame, %.1f s loop = %.2f h of growth",
                 limit, frameCount, width, height, samples * samples, frameDelayCentiseconds, loopSeconds, loopHours))
    print(String(format: "%.1f s on the GPU, %.0f s wall → %@ (%.2f MB, %.1f kB a frame)",
                 gpu, wall, path, Double(bytes) / 1_048_576, Double(bytes) / 1024 / Double(limit)))
} catch {
    FileHandle.standardError.write("bean_twining: \(error)\n".data(using: .utf8)!)
    exit(1)
}
