// Step 24: lactose intolerance — osmotic diarrhoea. Lactase working above,
// lactase missing below, both after two glasses of milk.
//
//   .build/lactose                 render renders/lactose.gif
//   .build/lactose stills          write a few frames as PNGs
//   .build/lactose still T out.png one frame at T seconds
//
// Environment overrides, so a short run needs no recompile:
//   GUT_WIDTH    frame width (default 1280)
//   GUT_FRAMES   frames per loop (default 80, at 12 cs: 9.6 s)
//   GUT_SAMPLES  samples per pixel side (default 2)
//   GUT_LIMIT    render only this many frames of the loop (for measuring)

import Foundation
import Metal

let env: [String: String] = ProcessInfo.processInfo.environment
func envInt(_ name: String, _ fallback: Int) -> Int {
    guard let text: String = env[name], let value = Int(text) else { return fallback }
    return value
}
func note(_ s: String) { FileHandle.standardError.write((s + "\n").data(using: .utf8)!) }

let args: [String] = Array(CommandLine.arguments.dropFirst())
let layout = Layout(width: envInt("GUT_WIDTH", 1280))
let samples: Int = envInt("GUT_SAMPLES", 2)
let frameCount: Int = envInt("GUT_FRAMES", 80)

/// Frame delay in hundredths of a second: the loop is frameCount × delay.
let delayCentiseconds: Int = envInt("GUT_DELAY", 12)

func renderGIF(_ renderer: Renderer, path: String) throws {
    let pixelCount: Int = layout.width * layout.height
    let limit: Int = envInt("GUT_LIMIT", 0)
    let count: Int = limit > 0 ? min(limit, frameCount) : frameCount

    // One palette for the whole loop, from frames spread across it.
    var palette: [RGB] = []
    var samplesRGB: [RGB] = []
    for f in stride(from: 0, to: frameCount, by: max(frameCount / 8, 1)) {
        let t: Float = frameTime(f, of: frameCount)
        try renderer.render(time: t, samples: samples)
        drawOverlay(into: renderer.frame, time: t)
        samplesRGB += samplePixels(renderer.pixels, pixels: pixelCount, step: 7)
        samplesRGB += samplePixels(renderer.pixels, pixels: pixelCount, step: 401)
    }
    palette = medianCutPalette(samplesRGB, count: 250)
    let quantizer = try Quantizer(device: renderer.device, palette: palette, pixels: pixelCount)
    try FileManager.default.createDirectory(atPath: "renders", withIntermediateDirectories: true)
    let gif = GIFWriter(url: URL(fileURLWithPath: path), width: layout.width, height: layout.height,
                        palette: palette, delayCentiseconds: delayCentiseconds)
    var gpuSeconds: Double = 0
    let start = Date()
    for f in 0..<count {
        let t: Float = frameTime(f, of: frameCount)
        gpuSeconds += try renderer.render(time: t, samples: samples)
        drawOverlay(into: renderer.frame, time: t)
        gif.add(try quantizer.indices(of: renderer.pixels))
        if f % 12 == 0 { note("  frame \(f)/\(count)") }
    }
    try gif.finish()
    let attributes = try? FileManager.default.attributesOfItem(atPath: path)
    let bytes: Int = (attributes?[.size] as? Int) ?? 0
    let n: Double = Double(count)
    let mb: Double = Double(bytes) / 1_048_576
    print(String(format: "lactose: %d of %d frames, %d × %d, %d spp, %d cs/frame (%.1f s loop) → %@ (%.2f MB, %.0f KB/frame)",
                 count, frameCount, layout.width, layout.height, samples * samples, delayCentiseconds,
                 Double(frameCount * delayCentiseconds) / 100, path, mb, Double(bytes) / 1024 / n))
    print(String(format: "  %.0f ms GPU per frame, %.1f s wall", gpuSeconds * 1000 / n, Date().timeIntervalSince(start)))
}

do {
    let device = try findDevice()
    let renderer = try Renderer(device: device, layout: layout)
    let mode: String = args.first ?? "gif"
    if mode == "still" {
        var t: Float = 0
        if args.count > 1, let v = Float(args[1]) { t = v }
        let out: String = args.count > 2 ? args[2] : "renders/still.png"
        let start = Date()
        let gpu: Double = try renderer.render(time: t, samples: samples)
        drawOverlay(into: renderer.frame, time: t)
        try savePNG(renderer.frame, to: URL(fileURLWithPath: out))
        print(String(format: "%d × %d, t = %.2f s: %.3f s GPU, %.2f s wall → %@",
                     layout.width, layout.height, t, gpu, Date().timeIntervalSince(start), out))
    } else if mode == "stills" {
        for f in stride(from: 0, to: frameCount, by: max(frameCount / 4, 1)) {
            let t: Float = frameTime(f, of: frameCount)
            try renderer.render(time: t, samples: samples)
            drawOverlay(into: renderer.frame, time: t)
            let out = "renders/frame_\(String(format: "%03d", f)).png"
            try savePNG(renderer.frame, to: URL(fileURLWithPath: out))
            print("wrote \(out)")
        }
    } else {
        try renderGIF(renderer, path: env["GUT_OUT"] ?? "renders/lactose.gif")
    }
} catch {
    note("lactose: \(error)")
    exit(1)
}
