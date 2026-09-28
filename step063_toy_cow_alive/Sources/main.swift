// Step 63: the plastic toy cow comes to life, as a looping GIF.
//
//   .build/toy_alive                  render renders/toy_cow_alive.gif
//   .build/toy_alive stills           write frames across the loop as PNGs
//   .build/toy_alive still T out      one frame at T picture seconds
//
// Environment overrides:
//   TOY_WIDTH    frame width (default 800; height is width / 1.5)
//   TOY_SAMPLES  samples per pixel side (default 2)
//   TOY_DELAY    hundredths of a second per frame (default 6)
//   TOY_LIMIT    encode only this many frames (to measure a frame's size)
//   TOY_FIRST    with TOY_LIMIT, the frame to start from
//   TOY_OUT      where the GIF goes

import Foundation
import Metal

let env: [String: String] = ProcessInfo.processInfo.environment
func envInt(_ name: String, _ fallback: Int) -> Int {
    guard let text: String = env[name], let value = Int(text) else { return fallback }
    return value
}
func note(_ s: String) { FileHandle.standardError.write((s + "\n").data(using: .utf8)!) }

let args: [String] = Array(CommandLine.arguments.dropFirst())
let width: Int = envInt("TOY_WIDTH", animWidth)
let height: Int = Int((Float(width) / animAspect).rounded())
let samples: Int = envInt("TOY_SAMPLES", 2)
let delay: Int = envInt("TOY_DELAY", gifDelayCentiseconds)

let perf = Performance(design: cowAnimDesign(), spec: cowWalk)
let setup: AnimSetup = animSetup(perf)
let loopCentiseconds: Int = Int((perf.spec.loopSeconds * 100).rounded())
let frameCount: Int = loopCentiseconds / delay
func frameTime(_ f: Int) -> Float { Float(f * delay) / 100 }

func renderFrame(_ r: ToyRenderer, _ t: Float) throws -> Double {
    let g: Double = try r.render(perf.posed(t), camera: setup.camera, samples: samples)
    annotateFrame(r.image, setup: setup, caption: cowAnimCaption)
    return g
}

func renderGIF(_ r: ToyRenderer, path: String) throws {
    let pixelCount: Int = width * height
    let limit: Int = envInt("TOY_LIMIT", 0)
    let count: Int = limit > 0 ? min(limit, frameCount) : frameCount
    var gpu: Double = 0
    let start = Date()
    // One palette for the whole loop, from frames spread across it, with the
    // toy's own pixels weighted in so its paint and sheen are in it.
    var sample: [RGB] = []
    for f in stride(from: 0, to: frameCount, by: max(frameCount / 14, 1)) {
        gpu += try renderFrame(r, frameTime(f))
        sample += samplePixels(r.image.pixels, pixels: pixelCount, step: 13)
        for y in stride(from: 0, to: height, by: 2) {
            for x in stride(from: 0, to: width, by: 2) where r.image.seen(x, y).x == 2 {
                let p: SIMD4<UInt8> = r.image.rgba(x, y)
                sample.append(RGB(p.x, p.y, p.z))
            }
        }
    }
    let palette: [RGB] = medianCutPalette(sample, count: 250)
    let quantizer = try Quantizer(device: r.device, palette: palette, pixels: pixelCount)
    let gif = GIFWriter(url: URL(fileURLWithPath: path), width: width, height: height, palette: palette,
                        delayCentiseconds: delay)
    // TOY_FIRST starts the measurement part-way in (the loop opens frozen,
    // which costs next to nothing and says nothing about the walk).
    let first: Int = limit > 0 ? envInt("TOY_FIRST", 0) : 0
    for f in first..<(first + count) {
        gpu += try renderFrame(r, frameTime(f))
        gif.add(try quantizer.indices(of: r.image.pixels))
        if f % 20 == 0 { note("  frame \(f)/\(count)") }
    }
    try gif.finish()
    let attributes = try? FileManager.default.attributesOfItem(atPath: path)
    let bytes: Int = (attributes?[.size] as? Int) ?? 0
    print(String(format: "toy_alive: %d of %d frames, %d × %d, %d spp, %d cs/frame (%.2f s loop) → %@ (%.2f MB, %.0f KB/frame)",
                 count, frameCount, width, height, samples * samples, delay, Double(perf.spec.loopSeconds), path,
                 Double(bytes) / 1_048_576, Double(bytes) / 1024 / Double(count)))
    print(String(format: "  %.1f s GPU, %.1f s wall", gpu, Date().timeIntervalSince(start)))
}

do {
    guard loopCentiseconds % delay == 0 else {
        note("toy_alive: a \(delay) cs delay does not divide the \(loopCentiseconds) cs loop")
        exit(1)
    }
    let device = try findDevice()
    let r = try ToyRenderer(device: device, width: width, height: height, paints: perf.design.paints, studio: setup.studio)
    print(String(format: "circle radius %.1f mm (%d strides of %.1f mm); loop %.2f s: hold %.1f, wake %.1f, walk %.1f, settle %.1f, hold %.1f",
                 perf.spec.radius, perf.spec.strides, perf.spec.stride, perf.spec.loopSeconds, perf.spec.holdStart,
                 perf.spec.wake, perf.spec.walkSeconds, perf.spec.settle, perf.spec.holdEnd))
    let mode: String = args.first ?? "gif"
    if mode == "still" {
        let t: Float = args.count > 1 ? (Float(args[1]) ?? 0) : 0
        let out: String = args.count > 2 ? args[2] : "renders/still.png"
        let g: Double = try renderFrame(r, t)
        try savePNG(r.image, to: URL(fileURLWithPath: out))
        print(String(format: "t = %.2f s → %@ (%.2f s GPU)", t, out, g))
    } else if mode == "stills" {
        let T: Float = perf.spec.loopSeconds
        for t in [Float(0), perf.spec.walkStart - 1.4, perf.spec.walkStart + 2.5, perf.spec.walkStart + perf.spec.walkSeconds * 0.4,
                  perf.spec.walkStart + perf.spec.walkSeconds * 0.75, perf.spec.walkEnd + 1.2, T - 0.5] {
            _ = try renderFrame(r, t)
            let out = "renders/frame_" + String(format: "%05.2f", t) + ".png"
            try savePNG(r.image, to: URL(fileURLWithPath: out))
            print("wrote \(out)")
        }
    } else {
        try renderGIF(r, path: env["TOY_OUT"] ?? "renders/toy_cow_alive.gif")
    }
} catch {
    note("toy_alive: \(error)")
    exit(1)
}
