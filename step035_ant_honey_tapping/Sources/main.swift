// Step 35: an ant taps honey — step 34's still, slightly and slowly animated,
// rendered on the GPU and saved as a looping GIF.
//
//   .build/ant_honey_tapping                  render renders/ant_honey_tapping.gif
//   .build/ant_honey_tapping still T out.png  one frame at T seconds, as a PNG
//   .build/ant_honey_tapping stills           a few frames across one tap, as PNGs
//
// Environment overrides, so a short run needs no recompile:
//   ANT_WIDTH    frame width (default 1920; height is 9/16 of it)
//   ANT_FRAMES   frames per loop (default in Motion.swift)
//   ANT_DELAY    centiseconds per frame (default in Motion.swift)
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
let width: Int = envInt("ANT_WIDTH", 1920)
let height: Int = width * 9 / 16
let samples: Int = envInt("ANT_SAMPLES", 3)
let frameCount: Int = envInt("ANT_FRAMES", defaultFrames)
let delayCentiseconds: Int = envInt("ANT_DELAY", defaultDelayCentiseconds)

/// Renders the loop: the first frame whole, then only the part that moves,
/// laid over a clean (unlabelled) copy of that first frame; labels on top.
final class Loop {
    let renderer: Renderer
    let frames: [FrameState]
    let region: MovingRegion
    private var base: [UInt8] = []
    var gpuSeconds: Double = 0

    init(renderer: Renderer, frames: [FrameState]) throws {
        self.renderer = renderer
        self.frames = frames
        region = movingRegion(scene: renderer.scene, frames: frames, width: renderer.width, height: renderer.height)
        gpuSeconds += try renderer.render(frames[0], samples: samples)
        let n: Int = renderer.width * renderer.height * 4
        base = Array(UnsafeBufferPointer(start: renderer.pixels.contents().assumingMemoryBound(to: UInt8.self), count: n))
    }

    func draw(_ f: Int) throws {
        base.withUnsafeBytes { raw in
            if let src = raw.baseAddress { renderer.pixels.contents().copyMemory(from: src, byteCount: raw.count) }
        }
        for rect in region.rects { gpuSeconds += try renderer.render(frames[f], samples: samples, region: rect) }
        annotate(renderer.image, scene: renderer.scene, frame: frames[f])
    }
}

func renderGIF(_ loop: Loop, path: String) throws {
    let r: Renderer = loop.renderer
    let pixelCount: Int = r.width * r.height
    let limit: Int = envInt("ANT_LIMIT", 0)
    let count: Int = limit > 0 ? min(limit, frameCount) : frameCount
    // One palette for the whole loop, from frames spread across one tap and
    // across the loop — touching, rising, highest, landing.
    var sampled: [RGB] = []
    for f in stride(from: 0, to: frameCount, by: max(frameCount / 10, 1)) {
        try loop.draw(f)
        sampled += samplePixels(r.pixels, pixels: pixelCount, step: 13)
        sampled += samplePixels(r.pixels, pixels: pixelCount, step: 397)
        // Step 35: and the moving parts again, more densely — the drop's
        // smooth ambers and the insets band visibly on a palette sampled
        // only from the whole frame, where plain card dominates.
        let p = r.pixels.contents().assumingMemoryBound(to: UInt8.self)
        for rect in loop.region.rects {
            for y in stride(from: rect.y0, to: rect.y1, by: 5) {
                for x in stride(from: rect.x0 + (y / 5) % 5, to: rect.x1, by: 5) {
                    let i: Int = (y * r.width + x) * 4
                    sampled.append(RGB(p[i], p[i + 1], p[i + 2]))
                }
            }
        }
    }
    let palette: [RGB] = medianCutPalette(sampled, count: 250)
    let quantizer = try Quantizer(device: r.device, palette: palette, pixels: pixelCount)
    let gif = GIFWriter(url: URL(fileURLWithPath: path), width: r.width, height: r.height,
                        palette: palette, delayCentiseconds: delayCentiseconds)
    let start = Date()
    for f in 0..<count {
        try loop.draw(f)
        gif.add(try quantizer.indices(of: r.pixels))
        if f % 10 == 0 { note("  frame \(f)/\(count)") }
    }
    try gif.finish()
    let attributes = try? FileManager.default.attributesOfItem(atPath: path)
    let bytes: Int = (attributes?[.size] as? Int) ?? 0
    let mb: Double = Double(bytes) / 1_048_576
    let reg: PixelRect = loop.region.bounds
    print(String(format: "ant_honey_tapping: %d of %d frames, %d × %d, %d spp, %d cs/frame (%.1f s loop) → %@ (%.2f MB, %.0f KB/frame)",
                 count, frameCount, r.width, r.height, samples * samples, delayCentiseconds,
                 Double(frameCount * delayCentiseconds) / 100, path, mb, Double(bytes) / 1024 / Double(count)))
    print("  moving region within x \(reg.x0)–\(reg.x1), y \(reg.y0)–\(reg.y1); rendered each frame: \(100 * loop.region.area / pixelCount)% of the frame")
    for r in loop.region.rects { print("    rect x \(r.x0)–\(r.x1), y \(r.y0)–\(r.y1)") }
    print(String(format: "  %.1f s GPU in all, %.1f s wall for the encode pass", loop.gpuSeconds, Date().timeIntervalSince(start)))
}

do {
    let device = try findDevice()
    let scene = try Scene(mutant: .none)
    let renderer = try Renderer(device: device, scene: scene, width: width, height: height)
    let mode: String = args.first ?? "gif"
    print("GPU: \(device.name); tap every \(tapSeconds) s on screen, slowed ×\(Int(slowdown)); loop \(loopSeconds) s")
    if mode == "still" {
        let t: Float = args.count > 1 ? (Float(args[1]) ?? 0) : 0
        let out: String = args.count > 2 ? args[2] : "renders/still.png"
        let start = Date()
        let frame: FrameState = scene.frame(at: t)
        let gpu: Double = try renderer.render(frame, samples: samples)
        annotate(renderer.image, scene: scene, frame: frame)
        try savePNG(renderer.image, to: URL(fileURLWithPath: out))
        print(String(format: "%d × %d, t = %.2f s (phase %.2f, lift %.3f mm): %.2f s GPU, %.2f s wall → %@",
                     width, height, t, frame.phase, frame.lift, gpu, Date().timeIntervalSince(start), out))
    } else if mode == "stills" {
        let frames: [FrameState] = (0..<frameCount).map { scene.frame(at: frameTime($0, of: frameCount)) }
        let loop = try Loop(renderer: renderer, frames: frames)
        let perTap: Int = frameCount / tapsPerLoop
        for f in stride(from: 0, to: perTap, by: max(perTap / 5, 1)) {
            try loop.draw(f)
            let out = "renders/frame_\(String(format: "%03d", f)).png"
            try savePNG(renderer.image, to: URL(fileURLWithPath: out))
            print("wrote \(out): phase \(frames[f].phase), lift \(frames[f].lift) mm")
        }
    } else {
        let start = Date()
        let frames: [FrameState] = (0..<frameCount).map { scene.frame(at: frameTime($0, of: frameCount)) }
        let loop = try Loop(renderer: renderer, frames: frames)
        try FileManager.default.createDirectory(atPath: "renders", withIntermediateDirectories: true)
        try renderGIF(loop, path: env["ANT_OUT"] ?? "renders/ant_honey_tapping.gif")
        print(String(format: "  %.1f s wall in all", Date().timeIntervalSince(start)))
    }
} catch {
    note("ant_honey_tapping: \(error)")
    exit(1)
}
