// Step 48: step 47's garden pea, now in flower and setting pods — the same
// vine, net, camera and loop, with Mendel's white flowers above and pods
// below. The climb is a time-lapse; the flowers and pods are a gradient along
// the stem by node age, not a time-lapse of a pod growing.
//
//   .build/pea_flowers_pods [width height samples-per-side [frames [colours]]] → renders/pea_flowers_pods.gif
//   .build/pea_flowers_pods stills [width height samples-per-side]             → renders/frameNNN.png
//
// `frames` renders only the first so many frames, to measure what a frame
// costs in the GIF before committing to a size. `make run` copies the finished
// loop to ../showcase/pea_flowers_pods.gif.
//
// Why 480 × 624 at 20 frames a second: step 46's size and rate. The camera
// rises every frame, so every pixel moves and step 8's encoder stores the
// whole frame every time; the GIF costs about (bytes per frame) × (frames).
// Ten frames measured first, as in step 47: 62.7 kB a frame with a
// 250-colour palette (12.5 MB for the loop, over the 10 MB the showcase
// allows), 46.7 kB at 96 colours, 44.2 kB at 80 — about 8.8 MB, with no
// banding seen on the white petals, the tan withering ones or the pods. 80
// colours, as in step 47, which leaves room for a palette drawn from the whole
// loop. Frame size, rate and count are step 47's, so the films match. The
// writer is Picture.swift's, copied here with the palette size as a parameter.

import Foundation
import Metal

let args: [String] = Array(CommandLine.arguments.dropFirst())
let firstArgument: String = args.first ?? ""
let stillsCommand: String = "stills"
/// Colours in the GIF's one palette; see the note at the top.
let gifColours: Int = 80
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
let paletteColours: Int = min(max(argument(4, gifColours), 16), 250)   // a 5th argument overrides, to measure

func captions(_ t: Float, _ cam: Camera, width: Int, height: Int) -> [Caption] {
    let k: Float = Float(height) / 520
    let h: Float = Float(height)
    var out: [Caption] = [
        Caption(text: nameCaption, x: 14 * k, baseline: 26 * k, size: 16 * k, italic: true),
        Caption(text: whiteCaption, x: 14 * k, baseline: 43 * k, size: 11 * k),
        Caption(text: selfCaption, x: 14 * k, baseline: 57 * k, size: 11 * k),
        Caption(text: gradientCaption, x: 14 * k, baseline: h - 44 * k, size: 11 * k),
        Caption(text: compressedCaption, x: 14 * k, baseline: h - 30 * k, size: 11 * k),
        Caption(text: timeCaption(), x: 14 * k, baseline: h - 13 * k, size: 11 * k, alpha: 0.8),
    ]
    // Name a closed bud, where the pollen has already reached the stigma.
    for n in nodes(t) {
        guard let f = inflorescence(n, t, .none), f.stage == .bud || f.stage == .opening,
              let fl = f.flowers.first, let keel = fl.petals.last else { continue }
        let at: V3 = keel.origin + keel.u * (0.5 * keel.length)
        let p: SIMD2<Float> = cam.project(at, width: width, height: height)
        guard p.y > 62 * k && p.y < h - 80 * k && p.x > 20 * k && p.x < Float(width) - 20 * k else { continue }
        let fade: Float = 1 - smoothstep(flowerOpens + 1, flowerOpen, n.age)
        let x: Float = min(max(p.x - n.side * 70 * k, 70 * k), Float(width) - 70 * k)
        out.append(Caption(text: budLabel, x: x, baseline: p.y + 26 * k, size: 11 * k, italic: true,
                           alpha: fade, centred: true, pointAt: p))
    }
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

/// The plant faded behind the caption lines at the top and bottom — this step
/// says more on the frame than step 47, and the pods hang where the words
/// are. The veil is fixed to the frame, so it cannot give the loop away.
func veiled(_ f: Frame) -> Frame {
    let k: Float = Float(f.height) / 520
    let h: Float = Float(f.height)
    var px: [SIMD4<Float>] = f.rgba
    for y in 0..<f.height {
        let fy: Float = Float(y)
        let top: Float = 1 - smoothstep(58 * k, 72 * k, fy)
        let bottom: Float = smoothstep(h - 64 * k, h - 52 * k, fy)
        let keep: Float = 1 - 0.62 * max(top, bottom)
        if keep >= 1 { continue }
        for x in 0..<f.width { px[y * f.width + x] *= keep }
    }
    return Frame(width: f.width, height: f.height, rgba: px, seen: f.seen)
}

func render(frame f: Int, with r: PlantRenderer) throws -> (bytes: [UInt8], gpu: Double) {
    let t: Float = hourOf(frame: f)
    let cam: Camera = camera(t)
    let scene: PlantScene = peaScene(t, .none)
    // A tendril is 0.8 mm across — about two pixels at 480 × 624, fewer in a
    // smaller test render — so it is held to at least a pixel's width (as step
    // 25 held its pollen tube) and cannot flicker in and out.
    let mmPerPixel: Float = cam.millimetresPerPixel(at: cam.position + cam.forward * cameraDistance, height: height)
    let (image, gpu) = try r.render(scene, camera: cam, width: width, height: height, samples: samples,
                                    tubeMin: 0.5 * mmPerPixel)
    return (finishFrame(veiled(image), ground: ground, captions: captions(t, cam, width: width, height: height)), gpu)
}

/// Picture.swift's loop writer (shared, unchanged, 250 colours) with the
/// palette's size as a parameter. See the note at the top for why.
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
        for f in [0, frameCount / 3, 2 * frameCount / 3, frameCount - 1] {
            let (bytes, gpu) = try render(frame: f, with: renderer)
            let path: String = String(format: "renders/frame%03d.png", f)
            try savePNG(bytes, width: width, height: height, to: URL(fileURLWithPath: path))
            print(String(format: "frame %3d  hour %5.2f  %.2f s GPU → %@", f, hourOf(frame: f), gpu, path))
        }
        exit(0)
    }
    var gpu: Double = 0
    let path: String = limit == frameCount ? "renders/pea_flowers_pods.gif" : "renders/measure.gif"
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
    FileHandle.standardError.write("pea_flowers_pods: \(error)\n".data(using: .utf8)!)
    exit(1)
}
