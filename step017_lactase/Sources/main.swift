// Step 17: lactose cut into glucose and galactose by beta-galactosidase.
//
//   .build/lactase            render renders/lactase.gif
//   .build/lactase stills     write a few frames as JPEGs, to look at them
//   .build/lactase bench      time the grid against brute force, and occlusion
//   .build/lactase constants  print the constants table with its sources
//
// Environment overrides, so a short run needs no recompile:
//   LAC_WIDTH    frame width in pixels (default 1280)
//   LAC_FRAMES   render only this many frames
//   LAC_AT       render one frame, to look at a moment
//   LAC_PROBES   occlusion probes per hit (0 turns it off)
//   LAC_GRID=0   brute force instead of the uniform grid
//   LAC_MUTATE   deliberately break something; see LacMutations

import CoreGraphics
import Foundation
import ImageIO
import Metal
import UniformTypeIdentifiers
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

let width: Int = envInt("LAC_WIDTH", lacFrameWidth)
let viewHeight: Int = width * lacFrameViewHeight / lacFrameWidth
let captionHeight: Int = width * lacFrameCaptionHeight / lacFrameWidth
let layout = FrameLayout(width: width, viewHeight: viewHeight, captionHeight: captionHeight)
let probes: Int = envInt("LAC_PROBES", 10)
let useGrid: Bool = envFlag("LAC_GRID", true)
let mutations = LacMutations.fromEnvironment()
let ao = AOSettings(probes: probes, distance: 9, strength: 1, contrast: 1.6)

let sceneURL = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent().deletingLastPathComponent()
    .appendingPathComponent("Resources/scene.json")

func makeFrameBuffer(_ device: MTLDevice) throws -> MTLBuffer {
    guard let b = device.makeBuffer(length: layout.width * layout.height * 4,
                                    options: .storageModeShared) else {
        throw RenderError.gpu("could not allocate the frame")
    }
    return b
}

/// Writes the frame buffer out as a JPEG, as step 12 does for its stills.
func writeJPEG(_ buffer: MTLBuffer, to path: String) throws {
    let bytes = layout.width * layout.height * 4
    let data = Data(bytes: buffer.contents(), count: bytes)
    let info = CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue)
    guard let provider = CGDataProvider(data: data as CFData),
          let image = CGImage(width: layout.width, height: layout.height, bitsPerComponent: 8,
                              bitsPerPixel: 32, bytesPerRow: layout.width * 4,
                              space: CGColorSpace(name: CGColorSpace.sRGB)!,
                              bitmapInfo: info, provider: provider, decode: nil,
                              shouldInterpolate: false, intent: .defaultIntent) else {
        throw RenderError.gpu("could not make an image")
    }
    let url = URL(fileURLWithPath: path)
    try? FileManager.default.createDirectory(at: url.deletingLastPathComponent(),
                                             withIntermediateDirectories: true)
    guard let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.jpeg.identifier as CFString,
                                                     1, nil) else {
        throw RenderError.gpu("could not open \(path)")
    }
    CGImageDestinationAddImage(dest, image, [kCGImageDestinationLossyCompressionQuality: 0.88] as CFDictionary)
    if !CGImageDestinationFinalize(dest) { throw RenderError.gpu("could not write \(path)") }
}

// MARK: - One frame

/// Renders frame `f`. The tetramer never moves — the camera goes round it and
/// stops — so the only thing rebuilt every frame is the small part of the scene
/// that reacts.
func renderFrame(_ f: Int, scene: LacScene, renderer: SceneRenderer,
                 buffer: MTLBuffer) throws -> Double {
    let camera = lacCamera(scene, at: f)
    let hole = lacPorthole(scene, at: f, camera: camera)
    let shapes = lacShapes(scene, at: f, porthole: hole, mutations: mutations)
    try renderer.buildGrid(shapes, density: 1)
    let ms = try renderer.render(shapes: shapes, camera: camera, into: buffer,
                                 width: layout.width, viewHeight: layout.viewHeight,
                                 useGrid: useGrid, ao: ao, porthole: hole)
    drawCaption(lacCaption(scene, at: f), into: buffer, layout: layout)
    return ms
}

// MARK: - The render

func renderLoop(_ scene: LacScene, _ renderer: SceneRenderer, _ buffer: MTLBuffer) throws {
    var frameCount = lacFrameCount
    if let n = Int(env["LAC_FRAMES"] ?? ""), n > 0 { frameCount = min(n, lacFrameCount) }
    let pixels = layout.width * layout.height

    // One palette for the whole loop, from frames spread across it so the
    // porthole's saturated interior is represented as well as the grey outside.
    var samples: [RGB] = []
    for f in stride(from: 0, to: lacFrameCount, by: max(lacFrameCount / 8, 1)) {
        _ = try renderFrame(f, scene: scene, renderer: renderer, buffer: buffer)
        samples += samplePixels(buffer, pixels: pixels, step: 7)
    }
    let palette = medianCutPalette(samples, count: 96)
    let quantizer = try Quantizer(device: renderer.device, palette: palette, pixels: pixels)

    let path = "renders/lactase.gif"
    try? FileManager.default.createDirectory(atPath: "renders", withIntermediateDirectories: true)
    let gif = GIFWriter(url: URL(fileURLWithPath: path), width: layout.width,
                        height: layout.height, palette: palette,
                        delayCentiseconds: lacDelayCentiseconds)
    var gpuSeconds = 0.0
    let start = Date()
    for f in 0..<frameCount {
        gpuSeconds += try renderFrame(f, scene: scene, renderer: renderer, buffer: buffer)
        gif.add(try quantizer.indices(of: buffer))
        if f % 10 == 0 {
            FileHandle.standardError.write("  frame \(f)/\(frameCount)\r".data(using: .utf8)!)
        }
    }
    try gif.finish()
    let size = (try? FileManager.default.attributesOfItem(atPath: path)[.size] as? Int) ?? 0
    let mb = Double(size) / 1_048_576
    let msPerFrame = gpuSeconds * 1000 / Double(frameCount)
    print(String(format: "\nlactase: %d frames, %.1f s at 12.5 fps, %d \u{00D7} %d, "
                       + "%.0f ms GPU per frame, %.0f s wall \u{2192} %@ (%.1f MB)",
                 frameCount, Double(frameCount) / 12.5, layout.width, layout.height,
                 msPerFrame, Date().timeIntervalSince(start), path, mb))
}

// MARK: - Stills

func renderStills(_ scene: LacScene, _ renderer: SceneRenderer, _ buffer: MTLBuffer) throws {
    let start = turnFrames + openFrames
    var at: [(String, Int)] = [("tetramer", 22), ("opening", turnFrames + openFrames / 2)]
    var k = start
    for (i, n) in stageFrames.enumerated() {
        let name = scene.boundaryNames.indices.contains(i) ? scene.boundaryNames[i] : "stage\(i)"
        at.append((String(format: "%d_%@", i, name.replacingOccurrences(of: " ", with: "_")),
                   k + n / 2))
        k += n
    }
    for (name, f) in at {
        _ = try renderFrame(f, scene: scene, renderer: renderer, buffer: buffer)
        try writeJPEG(buffer, to: "renders/\(name).jpg")
        print("  renders/\(name).jpg  (frame \(f))")
    }
    // One with the occlusion off, for the comparison every step since 9 has run.
    let f = start + 60
    let camera = lacCamera(scene, at: f)
    let hole = lacPorthole(scene, at: f, camera: camera)
    let shapes = lacShapes(scene, at: f, porthole: hole)
    try renderer.buildGrid(shapes, density: 1)
    _ = try renderer.render(shapes: shapes, camera: camera, into: buffer,
                            width: layout.width, viewHeight: layout.viewHeight,
                            ao: AOSettings.off, porthole: hole)
    drawCaption(lacCaption(scene, at: f), into: buffer, layout: layout)
    try writeJPEG(buffer, to: "renders/no_occlusion.jpg")
    print("  renders/no_occlusion.jpg")
}

// MARK: - Benchmark

func bench(_ scene: LacScene, _ renderer: SceneRenderer, _ buffer: MTLBuffer) throws {
    let f = turnFrames + openFrames + 60          // porthole open, cast on screen
    let camera = lacCamera(scene, at: f)
    let hole = lacPorthole(scene, at: f, camera: camera)
    let shapes = lacShapes(scene, at: f, porthole: hole)
    print("scene: \(shapes.count) shapes\n")

    let small = FrameLayout(width: 160, viewHeight: 120, captionHeight: 0)
    guard let sb = renderer.device.makeBuffer(length: small.width * small.height * 4,
                                              options: .storageModeShared) else { return }
    var brute = 0.0
    for _ in 0..<2 {
        brute = try renderer.render(shapes: shapes, camera: camera, into: sb,
                                    width: small.width, viewHeight: small.viewHeight,
                                    useGrid: false, ao: ao, porthole: hole)
    }
    let ratio = Double(layout.width * layout.viewHeight) / Double(small.width * small.height)
    let bruteFull = brute * ratio
    print(String(format: "brute force  %8.1f ms at 160\u{00D7}120  \u{2192}  %8.0f ms at "
                       + "%d\u{00D7}%d (scaled)",
                 brute * 1000, bruteFull * 1000, layout.width, layout.viewHeight))

    for density in [Float(0.5), 1, 2] {
        let t0 = Date()
        try renderer.buildGrid(shapes, density: density)
        let buildMs = Date().timeIntervalSince(t0) * 1000
        var best = Double.greatestFiniteMagnitude
        for _ in 0..<3 {
            let ms = try renderer.render(shapes: shapes, camera: camera, into: buffer,
                                         width: layout.width, viewHeight: layout.viewHeight,
                                         useGrid: true, ao: ao, porthole: hole)
            best = min(best, ms)
        }
        let g = renderer.grid!
        print(String(format: "grid \u{00D7}%-4.1f  %4d \u{00D7} %4d \u{00D7} %4d  %6.1f per used "
                           + "box  build %5.1f ms  trace %6.1f ms  %5.0f\u{00D7} faster",
                     density, g.dims.x, g.dims.y, g.dims.z, g.averageOccupancy,
                     buildMs, best * 1000, bruteFull / best))
    }

    try renderer.buildGrid(shapes, density: 1)
    for p in [0, 6, 10, 16, 24] {
        var best = Double.greatestFiniteMagnitude
        for _ in 0..<3 {
            let ms = try renderer.render(shapes: shapes, camera: camera, into: buffer,
                                         width: layout.width, viewHeight: layout.viewHeight,
                                         useGrid: true,
                                         ao: AOSettings(probes: p, distance: 9, strength: 1,
                                                        contrast: 1.6),
                                         porthole: hole)
            best = min(best, ms)
        }
        print(String(format: "occlusion %2d probes   %6.1f ms", p, best * 1000))
    }

    // What the porthole costs: the same frame with the cut off.
    var shut = Double.greatestFiniteMagnitude
    for _ in 0..<3 {
        let ms = try renderer.render(shapes: shapes, camera: camera, into: buffer,
                                     width: layout.width, viewHeight: layout.viewHeight,
                                     useGrid: true, ao: ao, porthole: .shut)
        shut = min(shut, ms)
    }
    print(String(format: "\nporthole shut (no CSG)  %6.1f ms", shut * 1000))
}

// MARK: - main

do {
    let scene = try loadScene(from: sceneURL)
    switch CommandLine.arguments.dropFirst().first {
    case "constants":
        print("Step 17 constants \u{2014} \(lacConstants(scene).count) of them\n")
        for c in lacConstants(scene) {
            print("[\(c.evidence.rawValue)] \(c.name)")
            print("    \(c.value)")
            print("    \(c.source)\n")
        }
        exit(0)
    default:
        break
    }

    let device = try findDevice()
    let renderer = try SceneRenderer(device: device)
    let buffer = try makeFrameBuffer(device)
    print("GPU: \(device.name)")
    print("\(scene.atoms.count) protein atoms, space-filling; "
          + "\(scene.cast.count) side-chain atoms and \(scene.mobiles.count) reacting atoms, "
          + "ball-and-stick")
    print("motion check: \(scene.motion.verdict)")
    if !mutations.isEmpty { print("MUTATED: \(env["LAC_MUTATE"] ?? "")") }

    switch CommandLine.arguments.dropFirst().first {
    case "bench": try bench(scene, renderer, buffer)
    case "stills": try renderStills(scene, renderer, buffer)
    case "at":
        let f = envInt("LAC_AT", 0)
        _ = try renderFrame(f, scene: scene, renderer: renderer, buffer: buffer)
        try writeJPEG(buffer, to: "renders/at\(f).jpg")
        print("  renders/at\(f).jpg")
    default: try renderLoop(scene, renderer, buffer)
    }
} catch {
    FileHandle.standardError.write("lactase: \(error)\n".data(using: .utf8)!)
    exit(1)
}
