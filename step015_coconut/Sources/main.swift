// Step 15: a coconut sprouting, as a vertical cutaway.
//
//   .build/coconut          render renders/coconut.gif
//   .build/coconut stills   write a few frames as PNGs, to look at them
//   .build/coconut bench    time brute force against the uniform grid
//   .build/coconut sweep    sweep the clip plane through the nut, as PNGs
//
// Environment overrides, so a short run needs no recompile:
//   COCO_WIDTH    frame width in pixels (default 960 — this is a PORTRAIT step)
//   COCO_FRAMES   render only this many frames
//   COCO_AT       render one frame, to look at a moment
//   COCO_CUT      move the clip plane, in mm
//   COCO_GRID=0   brute force instead of the grid
//   COCO_AO       ambient-occlusion rays per sample
//   COCO_MUTATE   deliberately break something; see Breakage

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
func envFloat(_ name: String, _ fallback: Float) -> Float {
    guard let text: String = env[name] else { return fallback }
    guard let value: Float = Float(text) else { return fallback }
    return value
}
func envFlag(_ name: String, _ fallback: Bool) -> Bool {
    guard let text: String = env[name] else { return fallback }
    return text != "0"
}

let width: Int = envInt("COCO_WIDTH", frameWidth)
let viewHeight: Int = width * frameViewHeight / frameWidth
let layout = FrameLayout(width: width, viewHeight: viewHeight,
                         captionHeight: width * frameCaptionHeight / frameWidth)
let useGrid: Bool = envFlag("COCO_GRID", true)
let breakage = Breakage.fromEnvironment()
let cutZ: Float = envFloat("COCO_CUT", cutPlaneZ)
var look = Look()
let camera: Camera = cutawayCamera(width: width, height: viewHeight,
                                   target: cutawayTarget, yaw: cameraYaw)

func drawOverlay(into buffer: MTLBuffer, plant: Plant, frame f: Int) {
    drawCutBoundaries(frame: buffer, aux: renderer.auxPointer!,
                      width: layout.width, viewHeight: layout.viewHeight)
    drawSoilLine(into: buffer, layout: layout, camera: camera)
    drawAnnotations(coconutAnnotations(plant: plant, layout: layout), into: buffer,
                    layout: layout, camera: camera)
    drawCaption(coconutCaption(frame: f, plant: plant), into: buffer, layout: layout)
    drawScaleRule(into: buffer, layout: layout, millimetres: 100)
}

// MARK: - The palette
//
// The sky is a smooth vertical ramp that median cut would happily band, so it
// gets 48 entries sampled straight off the ramp; the rest of the palette is
// median cut over the things that actually change — soil, husk, meat, roots,
// leaf — sampled from frames spread across the whole loop, because the scene at
// frame 0 and the scene at frame 120 barely share a colour.

func skyRamp(_ l: Look, count: Int) -> [RGB] {
    (0..<count).map { i -> RGB in
        let v: Float = Float(i) / Float(count - 1)
        let c: SIMD3<Float> = (l.skyBottom + (l.skyTop - l.skyBottom) * v) * 255
        return RGB(UInt8(min(max(c.x.rounded(), 0), 255)),
                   UInt8(min(max(c.y.rounded(), 0), 255)),
                   UInt8(min(max(c.z.rounded(), 0), 255)))
    }
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

/// Top-level code cannot propagate, so the two things that can fail before
/// there is anything to draw fail here, with a message rather than a trap.
func orDie<T>(_ what: String, _ body: () throws -> T) -> T {
    do {
        return try body()
    } catch {
        FileHandle.standardError.write("coconut: \(what): \(error)\n".data(using: .utf8)!)
        exit(1)
    }
}

let device: MTLDevice = orDie("no GPU") { try findDevice() }
let renderer: CutawayRenderer = orDie("kernel") { try CutawayRenderer(device: device) }
let frameBuffer: MTLBuffer = orDie("frame buffer") { () -> MTLBuffer in
    guard let b = device.makeBuffer(length: layout.width * layout.height * 4,
                                    options: .storageModeShared) else {
        throw RenderError.gpu("could not allocate the frame")
    }
    return b
}
let pixelCount: Int = layout.width * layout.height
if envInt("COCO_AO", -1) >= 0 { look.aoRays = envInt("COCO_AO", look.aoRays) }

@discardableResult
func renderFrame(_ f: Int, into buffer: MTLBuffer, plane: Float = cutZ,
                 grid: Bool = useGrid) throws -> (seconds: Double, plant: Plant) {
    let plant = poseCoconut(frame: f, breakage: breakage)
    let albedo = albedoTable(stages: plant.stages)
    if grid { try renderer.buildGrid(plant.prims, density: gridDensity) }
    let seconds = try renderer.render(prims: plant.prims, albedo: albedo, camera: camera,
                                      look: look, cutZ: plane, into: buffer,
                                      width: layout.width, viewHeight: layout.viewHeight,
                                      useGrid: grid, breakage: breakage)
    drawOverlay(into: buffer, plant: plant, frame: f)
    return (seconds, plant)
}

func snapshot(_ buffer: MTLBuffer) -> [UInt8] {
    let p = buffer.contents().assumingMemoryBound(to: UInt8.self)
    return Array(UnsafeBufferPointer(start: p, count: pixelCount * 4))
}

func renderLoop() throws {
    var frameCount: Int = coconutFrameCount
    let limit: Int = envInt("COCO_FRAMES", 0)
    if limit > 0 { frameCount = min(limit, coconutFrameCount) }

    // Frame 0 is rendered first and kept: it is what the dissolve goes back to,
    // and the loop-closure test insists the last dissolved frame is this array
    // byte for byte.
    _ = try renderFrame(0, into: frameBuffer)
    let dormant: [UInt8] = snapshot(frameBuffer)

    var samples: [RGB] = []
    for f in stride(from: 0, to: coconutFrameCount, by: max(coconutFrameCount / 8, 1)) {
        _ = try renderFrame(f, into: frameBuffer)
        samples += samplePixels(frameBuffer, pixels: pixelCount, step: 7)
        samples += samplePixels(frameBuffer, pixels: pixelCount, step: 397)
    }
    let ramp = skyRamp(look, count: 48)
    let palette = ramp + medianCutPalette(samples, count: 200)

    let quantizer = try Quantizer(device: device, palette: palette, pixels: pixelCount)
    let path = "renders/coconut.gif"
    try FileManager.default.createDirectory(atPath: "renders", withIntermediateDirectories: true)
    let gif = GIFWriter(url: URL(fileURLWithPath: path), width: layout.width,
                        height: layout.height, palette: palette,
                        delayCentiseconds: frameDelayCentiseconds)

    var gpuSeconds = 0.0
    var buildSeconds = 0.0
    var overflow: UInt32 = 0
    var primitives = 0
    var peakPrimitives = 0
    let start = Date()
    for f in 0..<frameCount {
        let (seconds, plant) = try renderFrame(f, into: frameBuffer)
        gpuSeconds += seconds
        buildSeconds += renderer.lastBuildSeconds
        overflow += renderer.overflowCount
        primitives = plant.prims.count
        peakPrimitives = max(peakPrimitives, primitives)
        let alpha: Float = dissolve(frame: f)
        if alpha > 0 {
            crossDissolve(frame: frameBuffer, toward: dormant, alpha: alpha, pixels: pixelCount)
        }
        gif.add(try quantizer.indices(of: frameBuffer))
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
    print(String(format: "coconut: %d frames, %d × %d, %d primitives → %@ (%.1f MB)",
                 frameCount, layout.width, layout.height, peakPrimitives, path, mb))
    print(String(format: "  %.1f ms GPU + %.1f ms grid per frame, %.0f s wall",
                 gpuSeconds * 1000 / n, buildSeconds * 1000 / n, wall))
    print(String(format: "  %d dormant + %d growing + %d held + %d dissolving = %.1f s",
                 dormantFrames, growFrames, holdFrames, dissolveFrames, loopDisplayedSeconds))
    print("  interval overflows: \(overflow)")
}

func renderStills() throws {
    try FileManager.default.createDirectory(atPath: "renders", withIntermediateDirectories: true)
    for f in [0, 30, 60, 92, 130] {
        _ = try renderFrame(f, into: frameBuffer)
        let path = String(format: "renders/frame%03d.png", f)
        try writePNG(frameBuffer, to: path)
        print("  \(path)")
    }
}

/// The clip plane swept through the whole nut. Nothing here ever ships; it is
/// the picture form of the capping test, which is exhaustive and on the CPU.
func renderSweep() throws {
    try FileManager.default.createDirectory(atPath: "renders", withIntermediateDirectories: true)
    for (i, z) in [Float(-70), -30, 0, 18, 45, 80].enumerated() {
        _ = try renderFrame(92, into: frameBuffer, plane: z)
        let path = String(format: "renders/sweep%02d.png", i)
        try writePNG(frameBuffer, to: path)
        print("  \(path)   z = \(z) mm")
    }
}

/// Where the interval list gets crowded. Overflow is silent in the picture —
/// the dropped intervals are usually inside something else — so it has to be
/// hunted with a number rather than with an eye.
func probe() throws {
    var globalWorst = 0
    var globalOver: UInt32 = 0
    for f in 0..<coconutFrameCount {
        _ = try renderFrame(f, into: frameBuffer)
        guard let aux = renderer.auxPointer else { return }
        var worst = 0
        var at = (0, 0)
        for y in 0..<layout.viewHeight {
            for x in 0..<layout.width {
                let met = Int(aux[y * layout.width + x].w.rounded())
                if met > worst { worst = met; at = (x, y) }
            }
        }
        globalOver += renderer.overflowCount
        if worst > globalWorst {
            globalWorst = worst
            print(String(format: "frame %3d  worst %4d intervals over %d samples at (%d, %d)",
                         f, worst, 4, at.0, at.1))
        }
    }
    print(String(format: "worst %d intervals per pixel (%.1f per ray of %d), %d overflows",
                 globalWorst, Double(globalWorst) / 4, maxIntervalsPerRay, Int(globalOver)))
}

func bench() throws {
    let plant = poseCoconut(frame: 92, breakage: breakage)
    let albedo = albedoTable(stages: plant.stages)
    print("scene: \(plant.prims.count) primitives")
    var brute = Double.greatestFiniteMagnitude
    for _ in 0..<3 {
        let s = try renderer.render(prims: plant.prims, albedo: albedo, camera: camera,
                                    look: look, cutZ: cutZ, into: frameBuffer,
                                    width: layout.width, viewHeight: layout.viewHeight,
                                    useGrid: false)
        brute = min(brute, s)
    }
    print(String(format: "brute force   %7.1f ms", brute * 1000))
    for density in [Float(0.5), 1, 2] {
        try renderer.buildGrid(plant.prims, density: density)
        var best = Double.greatestFiniteMagnitude
        for _ in 0..<3 {
            let s = try renderer.render(prims: plant.prims, albedo: albedo, camera: camera,
                                        look: look, cutZ: cutZ, into: frameBuffer,
                                        width: layout.width, viewHeight: layout.viewHeight,
                                        useGrid: true)
            best = min(best, s)
        }
        let g = renderer.grid!
        let spans: Double = Double(g.items.count) / Double(plant.prims.count)
        print(String(format: "grid ×%-4.1f    %7.1f ms   %3d × %3d × %3d   %5.1f per used box   "
                     + "%4.1f boxes per primitive   %4.1f× faster",
                     density, best * 1000, g.dims.x, g.dims.y, g.dims.z, g.averageOccupancy,
                     spans, brute / best))
    }
}

// MARK: - main

do {
    let plant = poseCoconut(frame: coconutFrameCount - 1, breakage: breakage)
    print("GPU: \(device.name)")
    print("Cocos nucifera, germinating: "
          + "\(plant.counts["layer"] ?? 0) pericarp and seed layers, "
          + "\(poreCount) germination pores (1 functional), "
          + "\(plant.counts["root"] ?? 0) adventitious roots, "
          + "\(plant.leafletCount) entire leaf")
    print(String(format: "interior: %.0f mL meat + %.0f mL water at rest; by the end the "
                 + "haustorium is %.0f mL", budgetAtStart.endospermVolume / 1000,
                 budgetAtStart.waterVolume / 1000, budgetAtEnd.haustoriumVolume / 1000))
    print(String(format: "  of which %.0f mL came from digested meat and %.0f mL from "
                 + "absorbed water (×%.1f)",
                 endospermLost / 1000, waterAbsorbed / 1000, haustoriumBulkRatio))
    print("\(plant.prims.count) primitives, "
          + (useGrid ? "uniform grid" : "brute force")
          + (breakage.isEmpty ? "" : "  MUTATED: \(breakage.rawValue)"))

    let single: Int = envInt("COCO_AT", -1)
    if single >= 0 {
        _ = try renderFrame(single, into: frameBuffer)
        try writePNG(frameBuffer, to: String(format: "renders/frame%03d.png", single))
        print("  renders/frame\(single).png")
    } else {
        switch CommandLine.arguments.dropFirst().first {
        case "stills": try renderStills()
        case "sweep": try renderSweep()
        case "bench": try bench()
        case "probe": try probe()
        default: try renderLoop()
        }
    }
} catch {
    FileHandle.standardError.write("coconut: \(error)\n".data(using: .utf8)!)
    exit(1)
}
