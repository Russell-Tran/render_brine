// Step 65: step 20's lower arch wearing down under bruxism, year by year, the
// camera held exactly still — rendered on the GPU, marked, and saved as a GIF.
//
//   .build/progression [width height samples-per-side]          → renders/bruxism_progression.gif
//   .build/progression measure N [width height samples-per-side] → the first N wear frames only, to size the GIF
//   .build/progression stills [width height samples-per-side]    → a few frames as PNGs
//
// Every frame is step 64's renderer at that frame's depth, with the newly
// worn surface tinted; the GIF encoder is step 8's.

import Foundation
import Metal

/// Step 8's GIF encoder names its errors this way.
enum RenderError: Error {
    case kernelCompile(String)
    case gpu(String)
}

let args: [String] = Array(CommandLine.arguments.dropFirst())
/// The first argument if it is a word (measure, stills), else nothing.
func modeWord() -> String {
    guard let first = args.first else { return "" }
    let isNumber: Bool = Int(first) != nil
    return isNumber ? "" : first
}
let mode: String = modeWord()
let numbers: [String] = mode.isEmpty ? args : Array(args.dropFirst())
/// How many wear frames `measure` renders: its number, or 10.
func measureWord() -> Int {
    guard mode == "measure", let first = numbers.first, let n = Int(first) else { return mode == "measure" ? 10 : 0 }
    return n
}
let measureCount: Int = measureWord()
let sizeArgs: [String] = mode == "measure" ? Array(numbers.dropFirst()) : numbers

func argument(_ n: Int, _ fallback: Int) -> Int {
    guard n < sizeArgs.count, let v = Int(sizeArgs[n]) else { return fallback }
    return v
}

// 1280 × 720: at step 21's 544 × 306 the labels' small type would not read.
// Only the teeth, the labels and the inset change between frames and the
// camera never moves, so step 8's encoder stores only the changed box of each
// frame; `measure` sized this before the full run (see the commit).
let width: Int = argument(0, 1280)
let height: Int = argument(1, 720)
let samples: Int = argument(2, 3)
let pixelCount: Int = width * height

let planMutant: PlanMutant = PlanMutant(rawValue: ProcessInfo.processInfo.environment["BRUX_MUTANT"] ?? "") ?? .none

func snapshot(_ buffer: MTLBuffer) -> [UInt8] {
    let p = buffer.contents().assumingMemoryBound(to: UInt8.self)
    return Array(UnsafeBufferPointer(start: p, count: pixelCount * 4))
}

/// Blend toward frame 0, in display values, as steps 19 and 21 do.
func dissolve(_ a: [UInt8], toward b: [UInt8], weight: Float) -> [UInt8] {
    var out = [UInt8](repeating: 255, count: a.count)
    for i in 0..<a.count {
        let x: Float = Float(a[i])
        let y: Float = Float(b[i])
        let v: Float = x + (y - x) * weight
        out[i] = UInt8(max(min(v.rounded(), 255), 0))
    }
    return out
}

/// One frame, rendered and marked, before any dissolve.
func picture(_ p: FramePlan, times: FeatureTimes, aims: TargetAims, on device: MTLDevice) throws -> (MouthImage, Double) {
    let (image, gpu) = try renderMouth(width: width, height: height, samples: samples, camera: p.camera,
                                       depth: p.depth, glow: SIMD2<Float>(glowBand, glowStrength), on: device)
    let survey: [ToothSurvey] = try surveyTeeth(depth: p.depth, step: 0.1, on: device)
    let targets: FrameTargets = frameTargets(survey, aims: aims)
    let k: Float = Float(height) / 1080
    let section: SectionGrid = try sampleSection(depth: p.depth, perMillimetre: 30 * k, on: device)
    drawFrameOverlay(image, plan: p, times: times, targets: targets, section: section)
    return (image, gpu)
}

do {
    let device = try findDevice()
    let start = Date()
    let times: FeatureTimes = try featureTimes(on: device)
    guard let aims = targetAims(try surveyTeeth(depth: finalWearDepth, step: 0.1, on: device)) else {
        throw RenderError.gpu("nothing to label")
    }
    print(String(format: "labels at years: facets %.1f, incisor dentine %.1f, canine flat %.1f, premolar dentine %.1f",
                 times.facets, times.incisorDentine, times.canineFlat, times.premolarDentine))

    if mode == "stills" {
        for f in [0, healthyFrames + wearFrames / 4, healthyFrames + wearFrames / 2,
                  healthyFrames + 3 * wearFrames / 4, healthyFrames + wearFrames - 1] {
            let p: FramePlan = plan(f, mutant: planMutant)
            let (image, _) = try picture(p, times: times, aims: aims, on: device)
            let path: String = String(format: "renders/frame%03d.png", f)
            try savePNG(image, to: URL(fileURLWithPath: path))
            print(String(format: "frame %3d  year %4.1f  depth %.2f mm → %@", f, p.years, p.depth, path))
        }
        exit(0)
    }

    // Render every distinct frame once and keep it: the holds reuse theirs,
    // and the dissolve blends the last worn frame toward the first.
    let first: Int = mode == "measure" ? healthyFrames : 0
    let last: Int = mode == "measure" ? min(healthyFrames + measureCount, frameCount) : frameCount
    var rendered: [Int: [UInt8]] = [:]
    var byDepth: [Float: [UInt8]] = [:]
    var gpu: Double = 0
    var firstFrame: [UInt8] = []
    for f in first..<last {
        let p: FramePlan = plan(f, mutant: planMutant)
        if let same = byDepth[p.depth], p.camera.position == stillCamera.position {
            rendered[f] = same
        } else {
            let (image, seconds) = try picture(p, times: times, aims: aims, on: device)
            gpu += seconds
            let px: [UInt8] = snapshot(image.pixels)
            rendered[f] = px
            if p.camera.position == stillCamera.position { byDepth[p.depth] = px }
        }
        if f == 0 { firstFrame = rendered[f]! }
        if p.dissolve > 0, let worn = rendered[f] {
            // A MouthImage-shaped buffer for the restart note, drawn crisply on top.
            let mixed: [UInt8] = dissolve(worn, toward: firstFrame, weight: p.dissolve)
            guard let buf = device.makeBuffer(bytes: mixed, length: mixed.count, options: .storageModeShared),
                  let aux = device.makeBuffer(length: 16, options: .storageModeShared) else {
                throw RenderError.gpu("no buffer")
            }
            let img = MouthImage(width: width, height: height, pixels: buf, aux: aux)
            drawRestart(img, strength: 1)
            rendered[f] = snapshot(buf)
        }
        if f % 10 == 0 { print(String(format: "  frame %3d/%d  year %4.1f", f, frameCount, p.years)) }
    }

    // One palette for the loop, from frames spread across it — the dissolve's
    // middle included, whose blended colours no single frame holds.
    var sampled: [RGB] = []
    guard let work = device.makeBuffer(length: pixelCount * 4, options: .storageModeShared) else {
        throw RenderError.gpu("could not allocate the frame buffer")
    }
    let keys: [Int] = rendered.keys.sorted()
    for i in 0..<10 {
        let f: Int = keys[i * (keys.count - 1) / 9]
        rendered[f]!.withUnsafeBytes { work.contents().copyMemory(from: $0.baseAddress!, byteCount: pixelCount * 4) }
        sampled += samplePixels(work, pixels: pixelCount, step: 7)
    }
    if let mid = keys.first(where: { plan($0, mutant: planMutant).dissolve >= 0.5 }) {
        rendered[mid]!.withUnsafeBytes { work.contents().copyMemory(from: $0.baseAddress!, byteCount: pixelCount * 4) }
        sampled += samplePixels(work, pixels: pixelCount, step: 3)
    }
    let palette: [RGB] = medianCutPalette(sampled, count: 250)
    let quantizer = try Quantizer(device: device, palette: palette, pixels: pixelCount)
    let path: String = mode == "measure" ? "renders/measure.gif" : "renders/bruxism_progression.gif"
    let gif = GIFWriter(url: URL(fileURLWithPath: path), width: width, height: height,
                        palette: palette, delayCentiseconds: frameDelayCentiseconds)
    for f in keys {
        rendered[f]!.withUnsafeBytes { work.contents().copyMemory(from: $0.baseAddress!, byteCount: pixelCount * 4) }
        gif.add(try quantizer.indices(of: work))
    }
    try gif.finish()
    let bytes: Int = ((try? FileManager.default.attributesOfItem(atPath: path))?[.size] as? Int) ?? 0
    let wall: Double = Date().timeIntervalSince(start)
    print("GPU: \(device.name)")
    print(String(format: "%d frames of %d, %d × %d, %d samples per pixel, %d cs a frame, %.1f s loop",
                 keys.count, frameCount, width, height, samples * samples, frameDelayCentiseconds, loopSeconds))
    print(String(format: "%.1f s on the GPU, %.0f s wall → %@ (%.2f MB, %.0f kB a frame)",
                 gpu, wall, path, Double(bytes) / 1_048_576, Double(bytes) / 1024 / Double(keys.count)))
} catch {
    FileHandle.standardError.write("progression: \(error)\n".data(using: .utf8)!)
    exit(1)
}
