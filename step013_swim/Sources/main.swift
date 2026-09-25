// Step 13: one brine shrimp swimming, as a brightfield micrograph.
//
//   .build/swim            render renders/swim.gif
//   .build/swim stills     write a few frames as PNGs, to look at them
//   .build/swim bench      time brute force against the uniform grid
//
// Environment overrides, so a short run needs no recompile:
//   SWIM_WIDTH    frame width in pixels (default 1280)
//   SWIM_FRAMES   render only this many frames
//   SWIM_AT       render one frame, to look at a moment
//   SWIM_GRID=0   brute force instead of the grid
//   SWIM_MUTATE   deliberately break something; see Mutations

import CoreGraphics
import Foundation
import ImageIO
import Metal
import simd

let env: [String: String] = ProcessInfo.processInfo.environment
func envInt(_ name: String, _ fallback: Int) -> Int {
    guard let text: String = env[name] else { return fallback }
    guard let value: Int = Int(text) else { return fallback }
    return value
}
func envFlag(_ name: String, _ fallback: Bool) -> Bool {
    guard let text: String = env[name] else { return fallback }
    return text != "0"
}

let width: Int = envInt("SWIM_WIDTH", frameWidth)
let viewHeight: Int = width * frameViewHeight / frameWidth
let layout = FrameLayout(width: width, viewHeight: viewHeight,
                         captionHeight: width * frameCaptionHeight / frameWidth)
let useGrid: Bool = envFlag("SWIM_GRID", true)
let mutations = Mutations.fromEnvironment()
let backlight = Backlight.standard
let posture = Posture()
let reynolds = ReynoldsNumbers()

/// The pixel scale is fixed at 9.1 µm, so a narrower frame simply sees less of
/// the field rather than shrinking the animal — which is what a microscope
/// does, and what keeps the seta clamp honest at any width.
let camera: Camera = {
    let centreBody = SIMD3<Float>(150, -520, 0)
    let rotated: SIMD3<Float> = posture.rotation() * centreBody
    // A last nudge in screen space, so the animal sits in the middle of the
    // field rather than in the middle of its own bounding box.
    let centreWorld: SIMD3<Float> = rotated + SIMD3<Float>(190, -120, 0)
    return Camera(centre: centreWorld, micronsPerPixel: micronsPerPixel,
                  width: width, height: viewHeight, standOff: 40_000)
}()

// MARK: - The caption


func drawOverlay(into buffer: MTLBuffer) {
    drawCaption(swimCaption(), into: buffer, layout: layout)
    let k = layout.scale
    let bar: CGFloat = CGFloat(1000 / micronsPerPixel)      // 1 mm
    drawScaleBar(into: buffer, layout: layout, lengthPixels: bar, label: "1 mm",
                 x: 26 * k, bottom: CGFloat(layout.captionHeight) + 26 * k)
}

// MARK: - The palette
//
// Most of this frame is lamp: a smooth radial ramp that median cut would
// happily spend half its palette on and still band. So the ramp gets 64
// entries sampled straight off the backlight curve — one step every level or
// less, which is smoother than 8 bits can show — and median cut is fed only
// the pixels the animal actually touched.

func backlightRamp(_ bl: Backlight, count: Int) -> [RGB] {
    (0..<count).map { i -> RGB in
        let r: Float = Float(i) / Float(count - 1)
        let f: Float = pow(r, bl.falloff)
        let c: SIMD3<Float> = (bl.centre + (bl.edge - bl.centre) * f) * 255
        return RGB(UInt8(min(max(c.x.rounded(), 0), 255)),
                   UInt8(min(max(c.y.rounded(), 0), 255)),
                   UInt8(min(max(c.z.rounded(), 0), 255)))
    }
}

/// Every `step`th pixel the animal changed. The lamp is recomputed on the CPU
/// with the same formula the kernel uses, so "changed" means "some τ landed
/// here" rather than "is not exactly the average colour".
func animalSamples(_ frame: MTLBuffer, step: Int) -> [RGB] {
    let p = frame.contents().assumingMemoryBound(to: UInt8.self)
    var out: [RGB] = []
    out.reserveCapacity(layout.width * layout.viewHeight / step)
    var i = 0
    while i < layout.width * layout.viewHeight {
        let x: Int = i % layout.width
        let y: Int = i / layout.width
        let lamp: SIMD3<Float> = backlight.colour(px: x, py: y, width: layout.width,
                                                  height: layout.viewHeight,
                                                  halfWidth: camera.halfWidth,
                                                  halfHeight: camera.halfHeight)
        let lr: Int = Int((lamp.x * 255).rounded())
        let lg: Int = Int((lamp.y * 255).rounded())
        let lb: Int = Int((lamp.z * 255).rounded())
        let dr: Int = abs(Int(p[i * 4]) - lr)
        let dg: Int = abs(Int(p[i * 4 + 1]) - lg)
        let db: Int = abs(Int(p[i * 4 + 2]) - lb)
        if dr + dg + db > 3 {
            out.append(RGB(p[i * 4], p[i * 4 + 1], p[i * 4 + 2]))
        }
        i += step
    }
    return out
}

// MARK: - PNG, for looking at single frames

func writePNG(_ buffer: MTLBuffer, to path: String) throws {
    guard let ctx = CGContext(data: buffer.contents(), width: layout.width, height: layout.height,
                              bitsPerComponent: 8, bytesPerRow: layout.width * 4,
                              space: CGColorSpace(name: CGColorSpace.sRGB)!,
                              bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue),
          let image = ctx.makeImage() else {
        throw RenderError.gpu("could not make an image")
    }
    try FileManager.default.createDirectory(atPath: "renders", withIntermediateDirectories: true)
    let url = URL(fileURLWithPath: path) as CFURL
    guard let dest = CGImageDestinationCreateWithURL(url, "public.png" as CFString, 1, nil) else {
        throw RenderError.gpu("could not open \(path)")
    }
    CGImageDestinationAddImage(dest, image, nil)
    CGImageDestinationFinalize(dest)
}

// MARK: - Rendering

func renderFrame(_ f: Int, renderer: SceneRenderer, buffer: MTLBuffer) throws -> (Double, Int) {
    let pose = poseArtemia(frame: f, posture: posture, mutations: mutations)
    let sigma = sigmaTable(mutations: mutations)
    if useGrid { try renderer.buildGrid(pose.prims, density: 1) }
    let ms = try renderer.render(prims: pose.prims, sigma: sigma, camera: camera,
                                 backlight: backlight, into: buffer,
                                 width: layout.width, viewHeight: layout.viewHeight,
                                 useGrid: useGrid, mutations: mutations)
    drawOverlay(into: buffer)
    return (ms, pose.prims.count)
}

func renderSwim(_ renderer: SceneRenderer, _ buffer: MTLBuffer) throws {
    var frameCount: Int = swimFrameCount
    let limit: Int = envInt("SWIM_FRAMES", 0)
    if limit > 0 { frameCount = min(limit, swimFrameCount) }
    let pixels = layout.width * layout.height

    var samples: [RGB] = []
    for f in stride(from: 0, to: swimFrameCount, by: max(swimFrameCount / 6, 1)) {
        _ = try renderFrame(f, renderer: renderer, buffer: buffer)
        samples += animalSamples(buffer, step: 5)
        samples += samplePixels(buffer, pixels: pixels, step: 601)   // the caption bar too
    }
    let ramp = backlightRamp(backlight, count: 64)
    let palette = ramp + medianCutPalette(samples, count: 160)

    let quantizer = try Quantizer(device: renderer.device, palette: palette, pixels: pixels)
    let path = "renders/swim.gif"
    let gif = GIFWriter(url: URL(fileURLWithPath: path), width: layout.width,
                        height: layout.height, palette: palette,
                        delayCentiseconds: frameDelayCentiseconds)

    var gpuSeconds = 0.0
    var buildSeconds = 0.0
    var overflow: UInt32 = 0
    var prims = 0
    let start = Date()
    for f in 0..<frameCount {
        let (ms, n) = try renderFrame(f, renderer: renderer, buffer: buffer)
        gpuSeconds += ms
        buildSeconds += renderer.lastBuildSeconds
        overflow += renderer.overflowCount
        prims = n
        gif.add(try quantizer.indices(of: buffer))
        if f % 10 == 0 {
            FileHandle.standardError.write("  frame \(f)/\(frameCount)\r".data(using: .utf8)!)
        }
    }
    try gif.finish()
    let attributes = try? FileManager.default.attributesOfItem(atPath: path)
    let bytes: Int = (attributes?[.size] as? Int) ?? 0
    let mb: Double = Double(bytes) / 1_048_576
    let n: Double = Double(frameCount)
    let wall: Double = Date().timeIntervalSince(start)
    print(String(format: "swim: %d frames, %d × %d, %d primitives → %@ (%.1f MB)",
                 frameCount, layout.width, layout.height, prims, path, mb))
    print(String(format: "  %.1f ms GPU + %.1f ms grid per frame, %.0f s wall",
                 gpuSeconds * 1000 / n, buildSeconds * 1000 / n, wall))
    print(String(format: "  %.1f s displayed for %.2f s of animal time (×1/%.0f), %.0f beats",
                 loopDisplayedSeconds, loopRealSeconds, slowMotionFactor,
                 Double(cyclesPerLoop)))
    print("  interval overflows: \(overflow)")
}

func renderStills(_ renderer: SceneRenderer, _ buffer: MTLBuffer) throws {
    try FileManager.default.createDirectory(atPath: "renders", withIntermediateDirectories: true)
    for f in [0, 7, 15, 22] {
        _ = try renderFrame(f, renderer: renderer, buffer: buffer)
        let path = String(format: "renders/frame%03d.png", f)
        try writePNG(buffer, to: path)
        print("  \(path)")
    }
}

func bench(_ renderer: SceneRenderer, _ buffer: MTLBuffer) throws {
    let pose = poseArtemia(frame: 7, posture: posture)
    let sigma = sigmaTable()
    print("scene: \(pose.prims.count) primitives")
    var brute = Double.greatestFiniteMagnitude
    for _ in 0..<3 {
        let ms = try renderer.render(prims: pose.prims, sigma: sigma, camera: camera,
                                     backlight: backlight, into: buffer,
                                     width: layout.width, viewHeight: layout.viewHeight,
                                     useGrid: false)
        brute = min(brute, ms)
    }
    print(String(format: "brute force   %7.1f ms", brute * 1000))
    for density in [Float(0.5), 1, 2] {
        try renderer.buildGrid(pose.prims, density: density)
        var best = Double.greatestFiniteMagnitude
        for _ in 0..<3 {
            let ms = try renderer.render(prims: pose.prims, sigma: sigma, camera: camera,
                                         backlight: backlight, into: buffer,
                                         width: layout.width, viewHeight: layout.viewHeight,
                                         useGrid: true)
            best = min(best, ms)
        }
        let g = renderer.grid!
        print(String(format: "grid ×%-4.1f    %7.1f ms   %3d × %3d × %3d   %5.1f per used box   "
                     + "%4.1f boxes per primitive   %4.1f× faster",
                     density, best * 1000, g.dims.x, g.dims.y, g.dims.z, g.averageOccupancy,
                     Double(g.items.count) / Double(pose.prims.count), brute / best))
    }
}

// MARK: - main

do {
    let device = try findDevice()
    let renderer = try SceneRenderer(device: device)
    guard let buffer = device.makeBuffer(length: layout.width * layout.height * 4,
                                         options: .storageModeShared) else {
        throw RenderError.gpu("could not allocate the frame")
    }
    let pose = poseArtemia(frame: 0, posture: posture, mutations: mutations)
    print("GPU: \(device.name)")
    print("Artemia franciscana, adult female: "
          + "\(pose.counts["thoracic segment"] ?? 0) thoracic segments, "
          + "\(pose.counts["phyllopod"] ?? 0) phyllopods, "
          + "\(pose.counts["abdominal segment"] ?? 0) abdominal segments, "
          + "\(pose.counts["limb seta"] ?? 0) + \(pose.counts["furcal seta"] ?? 0) setae")
    print(String(format: "Re: %.0f body, %.0f limb, %.3f seta — "
                 + "the animal is inertial, its setae are not",
                 reynolds.body, reynolds.limb, reynolds.seta))
    print("\(pose.prims.count) primitives, "
          + (useGrid ? "uniform grid" : "brute force")
          + (mutations.isEmpty ? "" : "  MUTATED: \(mutations.rawValue)"))

    let single: Int = envInt("SWIM_AT", -1)
    if single >= 0 {
        _ = try renderFrame(single, renderer: renderer, buffer: buffer)
        try writePNG(buffer, to: String(format: "renders/frame%03d.png", single))
        print("  renders/frame\(single).png")
    } else {
        switch CommandLine.arguments.dropFirst().first {
        case "stills": try renderStills(renderer, buffer)
        case "bench": try bench(renderer, buffer)
        default: try renderSwim(renderer, buffer)
        }
    }
} catch {
    FileHandle.standardError.write("swim: \(error)\n".data(using: .utf8)!)
    exit(1)
}
