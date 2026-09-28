// Step 46: a sweet pea climbing netting with its tendrils — a time-lapse loop,
// the camera rising with the growing tip, as in step 45.
//
//   .build/sweetpea_tendrils [width height samples-per-side [frames]]  → ../showcase/sweetpea_tendrils.gif
//   .build/sweetpea_tendrils stills [width height samples-per-side]    → renders/frameNNN.png
//
// `frames` renders only the first so many frames, to measure what a frame
// costs in the GIF before committing to a size.
//
// Why 480 × 624 at 20 frames a second — the same as step 45, so the pair
// match. The camera rises every frame, so every pixel moves and step 8's
// encoder stores the whole frame every time (the step 14 lesson); the GIF
// costs about (bytes per frame) × (frames). Ten frames measured first: 44.1
// kB a frame here, against step 45's 30.5 — the net's strings and the thin
// tendrils compress worse than the bean's broad leaves — so 200 frames come
// to about 8.8 MB, still inside the 10 MB the showcase allows.

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

func captions(_ t: Float, _ cam: Camera, width: Int, height: Int) -> [Caption] {
    let k: Float = Float(height) / 520
    let h: Float = Float(height)
    var out: [Caption] = [
        Caption(text: "Lathyrus odoratus, sweet pea", x: 14 * k, baseline: 26 * k, size: 16 * k, italic: true),
        Caption(text: relationCaption, x: 14 * k,
                baseline: 43 * k, size: 11 * k),
        Caption(text: toxicCaption, x: 14 * k, baseline: 57 * k, size: 11 * k),
        Caption(text: tendrilCaption, x: 14 * k,
                baseline: h - 30 * k, size: 11 * k),
        Caption(text: timeCaption(), x: 14 * k, baseline: h - 13 * k, size: 11 * k, alpha: 0.8),
    ]
    // Name the perversion on each spring that has formed, where it is.
    for n in nodes(t) {
        let td: TendrilGeometry = tendril(n, t, .none)
        guard td.contraction > 0.3, td.free.count > 2 else { continue }
        let mid: V3 = td.free[td.free.count / 2]
        let p: SIMD2<Float> = cam.project(mid, width: width, height: height)
        guard p.y > 80 * k && p.y < h - 60 * k else { continue }
        let x: Float = p.x + n.side * 46 * k
        out.append(Caption(text: "perversion", x: x, baseline: p.y - 18 * k, size: 11 * k, italic: true,
                           alpha: smoothstep(0.3, 0.7, td.contraction), centred: true,
                           pointAt: p + SIMD2<Float>(-n.side * 3 * k, -3 * k)))
    }
    return out
}

func render(frame f: Int, with r: PlantRenderer) throws -> (bytes: [UInt8], gpu: Double) {
    let t: Float = hourOf(frame: f)
    let cam: Camera = camera(t)
    let scene: PlantScene = sweetPeaScene(t, .none)
    // A tendril is 0.8 mm across — about two pixels at 480 × 624, fewer in a
    // smaller test render — so it is held to at least a pixel's width (as step
    // 25 held its pollen tube) and cannot flicker in and out.
    let mmPerPixel: Float = cam.millimetresPerPixel(at: cam.position + cam.forward * cameraDistance, height: height)
    let (image, gpu) = try r.render(scene, camera: cam, width: width, height: height, samples: samples,
                                    tubeMin: 0.5 * mmPerPixel)
    return (finishFrame(image, ground: ground, captions: captions(t, cam, width: width, height: height)), gpu)
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
    let path: String = limit == frameCount ? "../showcase/sweetpea_tendrils.gif" : "renders/measure.gif"
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
    FileHandle.standardError.write("sweetpea_tendrils: \(error)\n".data(using: .utf8)!)
    exit(1)
}
