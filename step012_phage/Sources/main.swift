// Step 12: a T4 phage baseplate firing into E. coli.
//
//   .build/phage           render renders/firing.gif
//   .build/phage bench     time brute force against the uniform grid
//
// Environment overrides, for the benchmark and short test renders:
//   P_WIDTH   frame width in pixels (default 960)
//   P_PROBES  occlusion probes per hit (default 12; 0 turns it off)
//   P_FRAMES  render only this many frames

import CoreGraphics
import Foundation
import Metal
import simd

let env = ProcessInfo.processInfo.environment
let widthText: String = env["P_WIDTH"] ?? ""
let width: Int = Int(widthText) ?? 860
let layout = FrameLayout(width: width, viewHeight: width * 3 / 4, captionHeight: width / 8)
let probesText: String = env["P_PROBES"] ?? ""
let probes: Int = Int(probesText) ?? 12
let delay = 8                                   // hundredths of a second: 12.5 fps
let sceneURL = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent().deletingLastPathComponent()
    .appendingPathComponent("Resources/scene.json")

let ao = AOSettings(probes: probes, distance: 14, strength: 1, contrast: 1.5)
let timing = Timing()

func firingCaption(_ scene: PhageScene, t: Float) -> Caption {
    let bp = scene.baseplate
    let preNm: Float = 2 * bp.preRadius / 10
    let postNm: Float = 2 * bp.postRadius / 10
    let across: String = String(format: "%.1f → %.1f nm across", preNm, postNm)
    if t < timing.settle {
        return Caption(title: "Loaded",
                       subtitle: "T4 baseplate, hexagonal · PDB 5IV5",
                       facts: "96 wedge chains · the hub and needle sit in the middle, ready",
                       aside: "\(scene.counts["total"] ?? 0) spheres")
    }
    if t < timing.settle + timing.flip {
        return Caption(title: "The baseplate flips",
                       subtitle: "hexagon → star · both states solved",
                       facts: "\(across) · every atom moving between two measured positions",
                       aside: String(format: "%.0f%%", timing.flipProgress(t) * 100))
    }
    if t < timing.settle + timing.flip + timing.drive {
        return Caption(title: "Fired",
                       subtitle: "the hub and needle driven through the wall",
                       facts: "outer membrane, peptidoglycan, inner membrane — about 30 nm of it",
                       aside: "PDB 5IV7")
    }
    if t < timing.settle + timing.flip + timing.drive + timing.hold {
        return Caption(title: "Through",
                       subtitle: "169,000 base pairs now follow, down the tube",
                       facts: "31× the whole pGLO plasmid of step 8a · about a minute in real time",
                       aside: "not drawn")
    }
    return Caption(title: "Reset",
                   subtitle: "back to loaded, so the loop closes",
                   facts: "a phage fires once — this is the next one, not a rewind",
                   aside: "")
}

/// The evidence bar sits against the bright top of the gradient, where a green
/// MEASURED is almost invisible. A soft dark plate behind it fixes that without
/// touching the shared overlay code.
func evidencePlate(into buffer: MTLBuffer, layout: FrameLayout) {
    guard let ctx = CGContext(data: buffer.contents(), width: layout.width, height: layout.height,
                              bitsPerComponent: 8, bytesPerRow: layout.width * 4,
                              space: CGColorSpace(name: CGColorSpace.sRGB)!,
                              bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return }
    let k = layout.scale
    let h = CGFloat(layout.height)
    let rect = CGRect(x: 10 * k, y: h - 58 * k, width: CGFloat(layout.width) - 20 * k, height: 48 * k)
    ctx.setFillColor(CGColor(srgbRed: 0.04, green: 0.08, blue: 0.13, alpha: 0.42))
    ctx.addPath(CGPath(roundedRect: rect, cornerWidth: 5 * k, cornerHeight: 5 * k, transform: nil))
    ctx.fillPath()
}

func makeFrameBuffer(_ device: MTLDevice) throws -> MTLBuffer {
    guard let b = device.makeBuffer(length: layout.width * layout.height * 4, options: .storageModeShared)
    else { throw RenderError.gpu("could not allocate the frame") }
    return b
}

// MARK: - benchmark

func bench(_ scene: PhageScene, _ renderer: SceneRenderer, _ buffer: MTLBuffer) throws {
    let t: Float = timing.settle + timing.flip * 0.5      // mid-flip, the busiest moment
    let shapes = poseScene(scene, t: t, timing: timing)
    let cam = firingCamera(scene, t: t)
    print("scene: \(shapes.count) spheres\n")

    // Brute force is unusable at this size, so it is timed small and scaled.
    let small = FrameLayout(width: 160, viewHeight: 120, captionHeight: 0)
    guard let sb = renderer.device.makeBuffer(length: small.width * small.height * 4,
                                              options: .storageModeShared) else { return }
    var brute = 0.0
    for _ in 0..<2 {
        brute = try renderer.render(shapes: shapes, camera: cam, into: sb,
                                    width: small.width, viewHeight: small.viewHeight,
                                    useGrid: false, ao: ao)
    }
    let pixelRatio = Double(layout.width * layout.viewHeight) / Double(small.width * small.height)
    let bruteFull = brute * pixelRatio
    print(String(format: "brute force  %8.1f ms at 160×120  →  %8.0f ms at %d×%d (scaled)",
                 brute * 1000, bruteFull * 1000, layout.width, layout.viewHeight))

    for density in [Float(0.5), 1, 2] {
        let t0 = Date()
        try renderer.buildGrid(shapes, density: density)
        let buildMs = Date().timeIntervalSince(t0) * 1000
        var best = Double.greatestFiniteMagnitude
        for _ in 0..<3 {
            let ms = try renderer.render(shapes: shapes, camera: cam, into: buffer,
                                         width: layout.width, viewHeight: layout.viewHeight,
                                         useGrid: true, ao: ao)
            best = min(best, ms)
        }
        let g = renderer.grid!
        print(String(format: "grid ×%-4.1f   %4d × %4d × %4d   %6.1f per used box   build %5.1f ms   trace %6.1f ms   %5.0f× faster",
                     density, g.dims.x, g.dims.y, g.dims.z, g.averageOccupancy,
                     buildMs, best * 1000, bruteFull / best))
    }

    // What occlusion costs here, against step 9's 8× and step 10's 4×.
    try renderer.buildGrid(shapes, density: 1)
    for p in [0, 6, 12, 24] {
        var best = Double.greatestFiniteMagnitude
        for _ in 0..<3 {
            let ms = try renderer.render(shapes: shapes, camera: cam, into: buffer,
                                         width: layout.width, viewHeight: layout.viewHeight,
                                         useGrid: true,
                                         ao: AOSettings(probes: p, distance: 14, strength: 1, contrast: 1.5))
            best = min(best, ms)
        }
        print(String(format: "occlusion %2d probes   %6.1f ms", p, best * 1000))
    }
}

// MARK: - the render

func renderFiring(_ scene: PhageScene, _ renderer: SceneRenderer, _ buffer: MTLBuffer) throws {
    let fps = 100.0 / Double(delay)
    var frameCount = Int((Double(timing.total) * fps).rounded())
    if let n = Int(env["P_FRAMES"] ?? ""), n > 0 { frameCount = n }
    let pixels = layout.width * layout.height

    func renderFrame(_ f: Int) throws -> Double {
        let t = Float(Double(f) / fps)
        let shapes = poseScene(scene, t: t, timing: timing)
        let cam = firingCamera(scene, t: t)
        // The scene deforms every frame, so the grid is rebuilt every frame —
        // as step 11 established, that costs a small share of the total.
        try renderer.buildGrid(shapes, density: 1)
        let ms = try renderer.render(shapes: shapes, camera: cam, into: buffer,
                                     width: layout.width, viewHeight: layout.viewHeight, ao: ao)
        drawCaption(firingCaption(scene, t: t), into: buffer, layout: layout)
        let (ev, note) = evidenceAt(t, timing: timing)
        evidencePlate(into: buffer, layout: layout)
        drawEvidence(ev, note: note, into: buffer, layout: layout)
        return ms
    }

    // One palette for the whole loop, sampled a few frames apart.
    var samples: [RGB] = []
    for f in stride(from: 0, to: frameCount, by: max(frameCount / 5, 1)) {
        _ = try renderFrame(f)
        samples += samplePixels(buffer, pixels: pixels, step: 7)
    }
    let palette = medianCutPalette(samples, count: 96)
    let quantizer = try Quantizer(device: renderer.device, palette: palette, pixels: pixels)

    let path = "renders/firing.gif"
    let gif = GIFWriter(url: URL(fileURLWithPath: path), width: layout.width, height: layout.height,
                        palette: palette, delayCentiseconds: delay)
    var gpuSeconds = 0.0
    let start = Date()
    for f in 0..<frameCount {
        gpuSeconds += try renderFrame(f)
        gif.add(try quantizer.indices(of: buffer))
        if f % 25 == 0 { FileHandle.standardError.write("  frame \(f)/\(frameCount)\r".data(using: .utf8)!) }
    }
    try gif.finish()
    let kb = ((try? FileManager.default.attributesOfItem(atPath: path)[.size] as? Int) ?? 0) / 1024
    print(String(format: "firing: %d frames, %.1f s at %.1f fps, %d × %d, %.0f ms GPU per frame, %.0f s total → %@ (%d KB)",
                 frameCount, Double(timing.total), fps, layout.width, layout.height,
                 gpuSeconds * 1000 / Double(frameCount), Date().timeIntervalSince(start), path, kb))
}

// MARK: - stills

func renderStills(_ scene: PhageScene, _ renderer: SceneRenderer, _ buffer: MTLBuffer) throws {
    let moments: [(String, Float)] = [
        ("loaded", 1.0),
        ("mid_flip", timing.settle + timing.flip * 0.5),
        ("star", timing.settle + timing.flip + 0.2),
        ("driving", timing.settle + timing.flip + timing.drive * 0.6),
        ("through", timing.settle + timing.flip + timing.drive + 1.0),
    ]
    try FileManager.default.createDirectory(atPath: "renders", withIntermediateDirectories: true)
    for (name, t) in moments {
        let shapes = poseScene(scene, t: t, timing: timing)
        try renderer.buildGrid(shapes, density: 1)
        _ = try renderer.render(shapes: shapes, camera: firingCamera(scene, t: t), into: buffer,
                                width: layout.width, viewHeight: layout.viewHeight, ao: ao)
        drawCaption(firingCaption(scene, t: t), into: buffer, layout: layout)
        let (ev, note) = evidenceAt(t, timing: timing)
        evidencePlate(into: buffer, layout: layout)
        drawEvidence(ev, note: note, into: buffer, layout: layout)
        try writeJPEG(buffer, layout: layout, to: "renders/\(name).jpg")
        print("  renders/\(name).jpg")
    }
    // One with occlusion off, for the comparison the last four steps have run.
    let t = timing.settle + timing.flip * 0.5
    let shapes = poseScene(scene, t: t, timing: timing)
    try renderer.buildGrid(shapes, density: 1)
    _ = try renderer.render(shapes: shapes, camera: firingCamera(scene, t: t), into: buffer,
                            width: layout.width, viewHeight: layout.viewHeight, ao: AOSettings.off)
    drawCaption(firingCaption(scene, t: t), into: buffer, layout: layout)
    try writeJPEG(buffer, layout: layout, to: "renders/mid_flip_noao.jpg")
    print("  renders/mid_flip_noao.jpg")
}

// MARK: - main

do {
    let scene = try loadScene(from: sceneURL)
    let device = try findDevice()
    let renderer = try SceneRenderer(device: device)
    let buffer = try makeFrameBuffer(device)
    print("GPU: \(device.name)")
    print("scene: \(scene.shapes.count) spheres — "
          + "\(scene.counts["wedge"] ?? 0) wedge (morphing), "
          + "\(scene.counts["needle"] ?? 0) needle (through the wall), "
          + "\(scene.counts["fibre"] ?? 0) fibres (gripping), "
          + "\(scene.counts["envelope"] ?? 0) envelope")

    switch CommandLine.arguments.dropFirst().first {
    case "bench": try bench(scene, renderer, buffer)
    case "stills": try renderStills(scene, renderer, buffer)
    default: try renderFiring(scene, renderer, buffer)
    }
} catch {
    FileHandle.standardError.write("phage: \(error)\n".data(using: .utf8)!)
    exit(1)
}
