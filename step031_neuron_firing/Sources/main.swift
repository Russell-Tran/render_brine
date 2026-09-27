// Step 31: step 27's neuron with one nerve impulse, slowed, as a looping GIF.
//
//   .build/neuron_firing                   render renders/neuron_firing.gif
//   .build/neuron_firing stills            write a few frames as PNGs
//   .build/neuron_firing still T out.png   one frame at T seconds
//
// Environment overrides, so a short run needs no recompile:
//   NEURON_WIDTH    frame width (default 1920; height is 9/16 of it)
//   NEURON_SAMPLES  samples per pixel side (default 3, step 27's)
//   NEURON_DELAY    hundredths of a second per frame (default 5: 20 fps)
//   NEURON_LIMIT    render only this many frames, from the impulse's start
//                   (for measuring the size of a moving frame)
//   NEURON_OUT      where the GIF goes

import Foundation
import Metal

let env: [String: String] = ProcessInfo.processInfo.environment
func envInt(_ name: String, _ fallback: Int) -> Int {
    guard let text: String = env[name], let value = Int(text) else { return fallback }
    return value
}
func note(_ s: String) { FileHandle.standardError.write((s + "\n").data(using: .utf8)!) }

let args: [String] = Array(CommandLine.arguments.dropFirst())
let width: Int = envInt("NEURON_WIDTH", 1920)
let height: Int = width * 9 / 16
let samples: Int = envInt("NEURON_SAMPLES", 3)
let delayCentiseconds: Int = envInt("NEURON_DELAY", 5)
/// Frames in one loop: the loop's length in hundredths, over the delay.
let frameCount: Int = Int((loopSeconds * 100).rounded()) / delayCentiseconds

func frameTime(_ f: Int) -> Float { Float(f) * loopSeconds / Float(frameCount) }

/// Renders one frame and adds the scale bar and caption. A frame with no glow
/// in it is the resting still, drawn once and copied after that.
final class Frames {
    let renderer: NeuronRenderer
    let impulse: Impulse
    private var rest: [UInt8]?
    private(set) var gpuSeconds: Double = 0
    private(set) var rendered: Int = 0

    init(_ renderer: NeuronRenderer, _ impulse: Impulse) {
        self.renderer = renderer
        self.impulse = impulse
    }

    func draw(_ t: Float) throws {
        let bytes: Int = renderer.image.width * renderer.image.height * 4
        let raw = renderer.image.pixels.contents()
        if isResting(t, impulse), let r = rest {
            r.withUnsafeBytes { raw.copyMemory(from: $0.baseAddress!, byteCount: bytes) }
            return
        }
        gpuSeconds += try renderer.render(time: t, samples: samples)
        rendered += 1
        drawScaleBar(renderer.image)
        drawCaption(renderer.image)
        if isResting(t, impulse) {
            rest = Array(UnsafeBufferPointer(start: raw.assumingMemoryBound(to: UInt8.self), count: bytes))
        }
    }
}

func renderGIF(_ frames: Frames, path: String) throws {
    let image: NeuronImage = frames.renderer.image
    let pixelCount: Int = image.width * image.height
    let limit: Int = envInt("NEURON_LIMIT", 0)
    // A measuring run starts where the impulse does, so its frames all move.
    let first: Int = limit > 0 ? Int(impulseStart / loopSeconds * Float(frameCount)) : 0
    let count: Int = limit > 0 ? min(limit, frameCount) : frameCount

    // One palette for the whole loop, from frames spread across it — the rest,
    // and the glow on the near side, far side and boutons.
    var sample: [RGB] = []
    for f in stride(from: 0, to: frameCount, by: max(frameCount / 16, 1)) {
        try frames.draw(frameTime(f))
        sample += samplePixels(image.pixels, pixels: pixelCount, step: 7)
        sample += samplePixels(image.pixels, pixels: pixelCount, step: 401)
        // The glow covers few pixels, so weight the band the axon crosses
        // more in moving frames: otherwise its fading tail shows bands.
        if !isResting(frameTime(f), frames.impulse) {
            let p = image.pixels.contents().assumingMemoryBound(to: UInt8.self)
            let bg = UInt8((backgroundDisplay * 255).rounded())
            let rows: Range<Int> = (image.height * 45 / 100)..<(image.height * 72 / 100)
            for y in rows {
                for x in stride(from: image.width * 30 / 100, to: image.width, by: 2) {
                    let i: Int = (y * image.width + x) * 4
                    let c = RGB(p[i], p[i + 1], p[i + 2])
                    if c != RGB(bg, bg, bg) { sample.append(c) }
                }
            }
        }
    }
    let palette: [RGB] = medianCutPalette(sample, count: 250)
    let quantizer = try Quantizer(device: frames.renderer.device, palette: palette, pixels: pixelCount)
    let gif = GIFWriter(url: URL(fileURLWithPath: path), width: image.width, height: image.height,
                        palette: palette, delayCentiseconds: delayCentiseconds)
    let start = Date()
    for k in 0..<count {
        let f: Int = (first + k) % frameCount
        try frames.draw(frameTime(f))
        gif.add(try quantizer.indices(of: image.pixels))
        if k % 20 == 0 { note("  frame \(k)/\(count)") }
    }
    try gif.finish()
    let attributes = try? FileManager.default.attributesOfItem(atPath: path)
    let bytes: Int = (attributes?[.size] as? Int) ?? 0
    let mb: Double = Double(bytes) / 1_048_576
    print(String(format: "neuron_firing: %d of %d frames, %d × %d, %d spp, %d cs/frame (%.1f s loop) → %@ (%.2f MB, %.0f KB/frame)",
                 count, frameCount, image.width, image.height, samples * samples, delayCentiseconds,
                 Double(loopSeconds), path, mb, Double(bytes) / 1024 / Double(count)))
    print(String(format: "  %d frames rendered, %.0f ms GPU each; %.1f s wall", frames.rendered,
                 frames.gpuSeconds * 1000 / Double(max(frames.rendered, 1)), Date().timeIntervalSince(start)))
}

do {
    let device = try findDevice()
    let cell: Neuron = buildNeuron()
    let impulse: Impulse = buildImpulse(cell)
    let renderer = try NeuronRenderer(cell, impulse, width: width, height: height, on: device)
    let frames = Frames(renderer, impulse)
    let mode: String = args.first ?? "gif"
    print(String(format: "conduction %.0f m/s, picture %.0f µm/s: slowed %.0f×; break %.1f ms real, %.1f s shown",
                 conductionVelocity, pictureSpeed, slowdown, realBreakSeconds * 1000, breakSeconds))
    if mode == "still" {
        var t: Float = 0
        if args.count > 1, let v = Float(args[1]) { t = v }
        let out: String = args.count > 2 ? args[2] : "renders/still.png"
        try frames.draw(t)
        try savePNG(renderer.image, to: URL(fileURLWithPath: out))
        print("t = \(t) s → \(out)")
    } else if mode == "stills" {
        for t in [Float(0), 0.9, 1.4, 2.0, 2.6, 3.4, 4.3, 4.6, 5.2] {
            try frames.draw(t)
            let out = "renders/frame_" + String(format: "%.1f", t) + ".png"
            try savePNG(renderer.image, to: URL(fileURLWithPath: out))
            print("wrote \(out)")
        }
    } else {
        try renderGIF(frames, path: env["NEURON_OUT"] ?? "renders/neuron_firing.gif")
    }
} catch {
    note("neuron_firing: \(error)")
    exit(1)
}
