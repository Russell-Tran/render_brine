// Step 55: five black garden ants walking a pheromone trail, as a looping GIF.
//
//   .build/ant_trail                 render renders/ant_trail_5.gif
//   .build/ant_trail stills          write a few frames as PNGs to renders/
//   .build/ant_trail still T out     one frame at T picture seconds
//
// Environment overrides, so a short run needs no recompile:
//   TRAIL_WIDTH    frame width (default below; height is frameAspect of it)
//   TRAIL_SAMPLES  samples per pixel side (default 2)
//   TRAIL_DELAY    hundredths of a second per frame (default 5); must divide
//                  the loop into whole frames
//   TRAIL_LIMIT    encode only this many frames (to measure a frame's size)
//   TRAIL_OUT      where the GIF goes

import Foundation
import Metal

let env: [String: String] = ProcessInfo.processInfo.environment
func envInt(_ name: String, _ fallback: Int) -> Int {
    guard let text: String = env[name], let value = Int(text) else { return fallback }
    return value
}
func note(_ s: String) { FileHandle.standardError.write((s + "\n").data(using: .utf8)!) }

let args: [String] = Array(CommandLine.arguments.dropFirst())
let width: Int = envInt("TRAIL_WIDTH", gifWidth)
let height: Int = Int((Float(width) * frameAspect).rounded())
let samples: Int = envInt("TRAIL_SAMPLES", 2)
let delayCentiseconds: Int = envInt("TRAIL_DELAY", gifDelayCentiseconds)
let loopCentiseconds: Int = Int((loopSeconds * 100).rounded())
let frameCount: Int = loopCentiseconds / delayCentiseconds

func frameTime(_ f: Int) -> Float { Float(f * delayCentiseconds) / 100 }

func renderGIF(_ r: TrailRenderer, path: String) throws {
    let pixelCount: Int = width * height
    let limit: Int = envInt("TRAIL_LIMIT", 0)
    let count: Int = limit > 0 ? min(limit, frameCount) : frameCount
    var gpu: Double = 0
    let start = Date()
    // One palette for the whole loop, from frames spread across it — so the
    // ants at every place in the frame, and the fresh marks, are in it.
    var sample: [RGB] = []
    for f in stride(from: 0, to: frameCount, by: max(frameCount / 12, 1)) {
        gpu += try r.render(time: frameTime(f), samples: samples)
        annotate(r.image)
        sample += samplePixels(r.image.pixels, pixels: pixelCount, step: 11)
        // The ants are few pixels among much table: weight the band they walk in.
        let p = r.image.pixels.contents().assumingMemoryBound(to: UInt8.self)
        for y in stride(from: height * 30 / 100, to: height * 80 / 100, by: 2) {
            for x in stride(from: 0, to: width, by: 2) {
                let i: Int = (y * width + x) * 4
                if p[i] < 150 { sample.append(RGB(p[i], p[i + 1], p[i + 2])) }
            }
        }
    }
    let palette: [RGB] = medianCutPalette(sample, count: 250)
    let quantizer = try Quantizer(device: r.device, palette: palette, pixels: pixelCount)
    let gif = GIFWriter(url: URL(fileURLWithPath: path), width: width, height: height,
                        palette: palette, delayCentiseconds: delayCentiseconds)
    for f in 0..<count {
        gpu += try r.render(time: frameTime(f), samples: samples)
        annotate(r.image)
        gif.add(try quantizer.indices(of: r.image.pixels))
        if f % 20 == 0 { note("  frame \(f)/\(count)") }
    }
    try gif.finish()
    let attributes = try? FileManager.default.attributesOfItem(atPath: path)
    let bytes: Int = (attributes?[.size] as? Int) ?? 0
    print(String(format: "ant_trail: %d of %d frames, %d × %d, %d spp, %d cs/frame (%.2f s loop) → %@ (%.2f MB, %.0f KB/frame)",
                 count, frameCount, width, height, samples * samples, delayCentiseconds, Double(loopSeconds), path,
                 Double(bytes) / 1_048_576, Double(bytes) / 1024 / Double(max(count, 1))))
    print(String(format: "  %.1f s GPU, %.1f s wall", gpu, Date().timeIntervalSince(start)))
}

do {
    guard loopCentiseconds % delayCentiseconds == 0 else {
        note("ant_trail: a \(delayCentiseconds) cs delay does not divide the \(loopCentiseconds) cs loop")
        exit(1)
    }
    let device = try findDevice()
    let r = try TrailRenderer(width: width, height: height, mutant: .none, on: device)
    print(String(format: "real speed %.1f mm/s at %d °C; headway %.2f s → spacing %.2f mm (ant %.2f); loop %.2f s shown: slowed %.1f×; %d ants in %.1f mm",
                 realSpeed, speedTemperature, headwayReal, spacing, baseLength, loopSeconds, slowdown, fileCount, frameWorldWidth))
    print("dab sites at \(dabSites) mm; stop centres τ = \(dabTaus) s")
    print(String(format: "stride %.3f mm, %.1f strides/s real; dab: gaster bent %.1f°, tip flexed %.1f°; %d frames of %d cs",
                 strideLength, stepFrequency, dabFull.bend * 180 / Float.pi, dabFull.flex * 180 / Float.pi, frameCount,
                 delayCentiseconds))
    let mode: String = args.first ?? "gif"
    if mode == "still" {
        let t: Float = args.count > 1 ? (Float(args[1]) ?? 0) : 0
        let out: String = args.count > 2 ? args[2] : "renders/still.png"
        let g: Double = try r.render(time: t, samples: samples)
        annotate(r.image)
        try savePNG(r.image, to: URL(fileURLWithPath: out))
        print(String(format: "t = %.2f s → %@ (%.2f s GPU)", t, out, g))
    } else if mode == "stills" {
        for t in stride(from: Float(0), to: loopSeconds, by: 0.75) {
            _ = try r.render(time: t, samples: samples)
            annotate(r.image)
            let out = "renders/frame_" + String(format: "%.1f", t) + ".png"
            try savePNG(r.image, to: URL(fileURLWithPath: out))
            print("wrote \(out)")
        }
    } else {
        try renderGIF(r, path: env["TRAIL_OUT"] ?? "renders/\(gifName)")
    }
} catch {
    note("ant_trail: \(error)")
    exit(1)
}
