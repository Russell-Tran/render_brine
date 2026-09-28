// Step 69: one ant biting a piece of peeled apple, rendered on the GPU and
// saved as a looping GIF.
//
//   .build/ant_bites_apple                  render renders/ant_bites_apple.gif
//   .build/ant_bites_apple still T out.png  one frame at T seconds, as a PNG
//   .build/ant_bites_apple stills           a few frames across the loop, as PNGs
//
// Environment overrides, so a short run needs no recompile:
//   ANT_WIDTH    frame width (default 1600; height is 9/16 of it)
//   ANT_DELAY    centiseconds per frame (default in main below)
//   ANT_SAMPLES  samples per pixel side (default 3)
//   ANT_LIMIT    encode only this many frames (for measuring the GIF size)
//   ANT_OUT      output path

import Foundation
import Metal

let env: [String: String] = ProcessInfo.processInfo.environment
func envInt(_ name: String, _ fallback: Int) -> Int {
    guard let text: String = env[name], let value = Int(text) else { return fallback }
    return value
}
func note(_ s: String) { FileHandle.standardError.write((s + "\n").data(using: .utf8)!) }

let args: [String] = Array(CommandLine.arguments.dropFirst())
let width: Int = envInt("ANT_WIDTH", 1600)
let height: Int = width * 9 / 16
let samples: Int = envInt("ANT_SAMPLES", 3)
let delayCentiseconds: Int = envInt("ANT_DELAY", defaultDelayCentiseconds)
let loopCentiseconds: Int = Int((loopSeconds * 100).rounded())
let frameCount: Int = loopCentiseconds / delayCentiseconds

func renderGIF(_ r: Renderer, path: String) throws {
    let pixelCount: Int = r.width * r.height
    let limit: Int = envInt("ANT_LIMIT", 0)
    let count: Int = limit > 0 ? min(limit, frameCount) : frameCount
    var gpu: Double = 0
    let start = Date()
    // One palette for the whole loop, from frames spread across it.
    var sampled: [RGB] = []
    for f in stride(from: 0, to: frameCount, by: max(frameCount / 12, 1)) {
        let fs: FrameState = r.scene.frame(at: frameTime(f, of: frameCount))
        gpu += try r.render(fs, samples: samples)
        annotate(r.image, scene: r.scene, frame: fs)
        sampled += samplePixels(r.pixels, pixels: pixelCount, step: 13)
        sampled += samplePixels(r.pixels, pixels: pixelCount, step: 397)
    }
    let palette: [RGB] = medianCutPalette(sampled, count: 250)
    let quantizer = try Quantizer(device: r.device, palette: palette, pixels: pixelCount)
    let gif = GIFWriter(url: URL(fileURLWithPath: path), width: r.width, height: r.height,
                        palette: palette, delayCentiseconds: delayCentiseconds)
    // The GIF stores only what changed since the last frame. Float noise in
    // the soft shadows leaves specks a level or two apart across the card
    // from frame to frame; a pixel within `still` levels of what the last
    // frame showed keeps what it showed, so those specks cost nothing. No
    // pixel is ever more than `still` levels from its own render.
    let still: Int = envInt("ANT_STILL", 3)
    var shown: [UInt8] = []
    for f in 0..<count {
        let fs: FrameState = r.scene.frame(at: frameTime(f, of: frameCount))
        gpu += try r.render(fs, samples: samples)
        annotate(r.image, scene: r.scene, frame: fs)
        let p = r.pixels.contents().assumingMemoryBound(to: UInt8.self)
        if shown.isEmpty {
            shown = Array(UnsafeBufferPointer(start: p, count: pixelCount * 4))
        } else {
            for i in 0..<pixelCount {
                let o: Int = i * 4
                let d0: Int = abs(Int(p[o]) - Int(shown[o]))
                let d1: Int = abs(Int(p[o + 1]) - Int(shown[o + 1]))
                let d2: Int = abs(Int(p[o + 2]) - Int(shown[o + 2]))
                if max(d0, max(d1, d2)) <= still {
                    p[o] = shown[o]; p[o + 1] = shown[o + 1]; p[o + 2] = shown[o + 2]
                } else {
                    shown[o] = p[o]; shown[o + 1] = p[o + 1]; shown[o + 2] = p[o + 2]
                }
            }
        }
        gif.add(try quantizer.indices(of: r.pixels))
        if f % 10 == 0 { note("  frame \(f)/\(count)") }
    }
    try gif.finish()
    let attributes = try? FileManager.default.attributesOfItem(atPath: path)
    let bytes: Int = (attributes?[.size] as? Int) ?? 0
    print(String(format: "ant_bites_apple: %d of %d frames, %d × %d, %d spp, %d cs/frame (%.1f s loop) → %@ (%.2f MB, %.0f KB/frame)",
                 count, frameCount, r.width, r.height, samples * samples, delayCentiseconds, Double(loopSeconds), path,
                 Double(bytes) / 1_048_576, Double(bytes) / 1024 / Double(count)))
    print(String(format: "  %.1f s GPU, %.1f s wall", gpu, Date().timeIntervalSince(start)))
}

do {
    guard loopCentiseconds % delayCentiseconds == 0 else {
        note("ant_bites_apple: a \(delayCentiseconds) cs delay does not divide the \(loopCentiseconds) cs loop")
        exit(1)
    }
    let device = try findDevice()
    let scene = Scene(mutant: .none)
    let renderer = try Renderer(device: device, scene: scene, width: width, height: height)
    let mode: String = args.first ?? "gif"
    print(String(format: "GPU: %@; slowed ×%.0f; loop %.1f s; ant at x %.3f mm; grip %.2f°, gape %.1f°",
                 device.name, slowdown, loopSeconds, bite.x0, bite.grip * 180 / Float.pi, gapeMax * 180 / Float.pi))
    if mode == "still" {
        let t: Float = args.count > 1 ? (Float(args[1]) ?? 0) : 0
        let out: String = args.count > 2 ? args[2] : "renders/still.png"
        let start = Date()
        let frame: FrameState = scene.frame(at: t)
        let gpu: Double = try renderer.render(frame, samples: samples)
        annotate(renderer.image, scene: scene, frame: frame)
        try savePNG(renderer.image, to: URL(fileURLWithPath: out))
        print(String(format: "%d × %d, t = %.2f s: %.2f s GPU, %.2f s wall → %@",
                     width, height, t, gpu, Date().timeIntervalSince(start), out))
    } else if mode == "stills" {
        for t in [Float(0), 0.75, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0] {
            let frame: FrameState = scene.frame(at: t)
            _ = try renderer.render(frame, samples: samples)
            annotate(renderer.image, scene: scene, frame: frame)
            let out = "renders/frame_" + String(format: "%05.2f", t) + ".png"
            try savePNG(renderer.image, to: URL(fileURLWithPath: out))
            print("wrote \(out)")
        }
    } else {
        try FileManager.default.createDirectory(atPath: "renders", withIntermediateDirectories: true)
        try renderGIF(renderer, path: env["ANT_OUT"] ?? "renders/ant_bites_apple.gif")
    }
} catch {
    note("ant_bites_apple: \(error)")
    exit(1)
}
