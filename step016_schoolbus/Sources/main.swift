// Step 16: inside a moving schoolbus.
//
//   .build/schoolbus           render renders/schoolbus.gif
//   .build/schoolbus control   render renders/orbit_control.gif — a moving camera
//   .build/schoolbus bench     render both and print what the mask bought
//   .build/schoolbus stills    write the still frames for the results page
//
// Environment overrides, for short test renders:
//   S_WIDTH   frame width in pixels (default 960)
//   S_FRAMES  render only this many frames
//   S_PROBES  occlusion probes per interior hit (default 6; 0 turns it off)

import CoreGraphics
import Foundation
import ImageIO
import Metal
import UniformTypeIdentifiers
import simd

let env = ProcessInfo.processInfo.environment
let widthText: String = env["S_WIDTH"] ?? ""
let width: Int = Int(widthText) ?? 960
let layout = FrameLayout(width: width, viewHeight: width * 3 / 4, captionHeight: width * 3 / 16)
let probesText: String = env["S_PROBES"] ?? ""
let probes: Int = Int(probesText) ?? 6

let specsURL = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent().deletingLastPathComponent()
    .appendingPathComponent("Resources/bus.json")

let loop = Loop()
let road = Road()
let sunModel = Sun()

func settings(for bus: Bus, frame: Int) -> RenderSettings {
    var s = RenderSettings()
    s.samplesPerSide = 2
    s.aoProbes = probes
    s.aoDistance = 0.85
    s.exposure = loop.exposure
    s.speed = loop.speed
    s.travelWrapped = loop.travelWrapped(frame: frame)
    s.groundY = bus.groundY
    s.roadCentre = road.centre
    s.roadHalfWidth = road.laneWidth
    s.dashPeriod = road.dashPeriod
    s.dashLength = road.dashLength
    s.sun = sunModel.direction
    return s
}

// MARK: - the caption

func ladderRungs(_ props: [Prop]) -> [LadderRung] {
    var out: [LadderRung] = []
    for prop in props {
        guard let label = prop.ladderLabel else { continue }
        let rate: Float = degreesPerSecond(abeamRate(speed: loop.speed, lateral: prop.lateral))
        let distance: String = prop.lateral >= 1000
            ? String(format: "%.0f m", prop.lateral)
            : String(format: "%.0f m", prop.lateral)
        let rateText: String = rate >= 10
            ? String(format: "%.0f °/s", rate)
            : String(format: "%.1f °/s", rate)
        out.append(LadderRung(label: label, distance: distance, rate: rateText))
    }
    return out
}

func mainCaption(traced: Double) -> Caption {
    let kmh: Float = loop.speed * 3.6
    let subtitle = String(format: "the camera is bolted to the bus — everything moving is the world · %.1f m/s (%.0f km/h)",
                          loop.speed, kmh)
    let facts = String(format: "%d frames, %.1f s, %.1f m travelled · three %.1f m roadside tiles · one velocity vector, three flow fields",
                       loop.frames, loop.duration, loop.distance, loop.tile)
    let aside = String(format: "%.0f%% of pixels traced", traced * 100)
    return Caption(title: "Inside a moving schoolbus", subtitle: subtitle, facts: facts, aside: aside)
}

func decorate(_ buffer: MTLBuffer, caption: Caption, rungs: [LadderRung]) {
    drawCaption(caption, into: buffer, layout: layout)
    drawLadder(rungs, into: buffer, layout: layout, top: 84 * layout.scale)
    let k = layout.scale
    drawPlate(into: buffer, layout: layout,
              rect: CGRect(x: 10 * k, y: 10 * k, width: 360 * k, height: 44 * k))
    drawEvidence(.measured,
                 note: "bus MEASURED · parallax and blur DERIVED · roadside and sun MODEL",
                 into: buffer, layout: layout)
}

// MARK: - plumbing

func makeFrameBuffer(_ device: MTLDevice) throws -> MTLBuffer {
    guard let b = device.makeBuffer(length: layout.width * layout.height * 4,
                                    options: .storageModeShared)
    else { throw RenderError.gpu("could not allocate the frame") }
    return b
}

func writeJPEG(_ buffer: MTLBuffer, layout: FrameLayout, to path: String) throws {
    let bytes = layout.width * layout.height * 4
    let data = Data(bytes: buffer.contents(), count: bytes)
    guard let provider = CGDataProvider(data: data as CFData),
          let image = CGImage(width: layout.width, height: layout.height, bitsPerComponent: 8,
                              bitsPerPixel: 32, bytesPerRow: layout.width * 4,
                              space: CGColorSpace(name: CGColorSpace.sRGB)!,
                              bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                              provider: provider, decode: nil, shouldInterpolate: false,
                              intent: .defaultIntent) else {
        throw RenderError.gpu("could not make an image")
    }
    let url = URL(fileURLWithPath: path)
    try? FileManager.default.createDirectory(at: url.deletingLastPathComponent(),
                                             withIntermediateDirectories: true)
    guard let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.jpeg.identifier as CFString,
                                                    1, nil) else {
        throw RenderError.gpu("could not create \(path)")
    }
    CGImageDestinationAddImage(dest, image,
                               [kCGImageDestinationLossyCompressionQuality as String: 0.92] as CFDictionary)
    guard CGImageDestinationFinalize(dest) else { throw RenderError.gpu("could not write \(path)") }
}

struct LoopStats {
    var frames: Int = 0
    var gpuSeconds: Double = 0
    var wallSeconds: Double = 0
    var gridSeconds: Double = 0
    var bytes: Int = 0
    var changedFraction: Double = 0
    /// The area of the box around everything that changed, as a share of the
    /// frame. Step 14 found this out the hard way: the step-8 encoder stores ONE
    /// rectangle per frame, so a small number of changed pixels scattered over
    /// the whole picture buys nothing at all. The two numbers are measured
    /// separately here because they are not the same claim.
    var boxFraction: Double = 0
    var tracedFraction: Double = 0
    var path: String = ""
}

func frameCount() -> Int {
    if let n = Int(env["S_FRAMES"] ?? ""), n > 0 { return n }
    return loop.frames
}

// MARK: - the render

/// The whole animation system. `masked` decides whether later frames retrace
/// only the pixels the mask says can change.
func renderLoop(bus: Bus, props: [Prop], renderer: SceneRenderer, buffer: MTLBuffer,
                masked: Bool, orbiting: Bool, path: String) throws -> LoopStats {
    let frames = frameCount()
    let pixels = layout.width * layout.height
    var stats = LoopStats()
    stats.frames = frames
    stats.path = path

    func camera(_ f: Int) -> Camera {
        orbiting ? bus.orbitCamera(frame: f, loop: loop) : bus.camera()
    }

    // The mask, from the bus alone. It does not depend on the frame, so it is
    // computed once — and for the orbiting control it cannot be used at all,
    // because a camera that moves invalidates it on the first frame.
    var indices: [UInt32] = []
    if masked {
        let mask = try renderer.computeMask(camera: camera(0), settings: settings(for: bus, frame: 0),
                                            width: layout.width, viewHeight: layout.viewHeight)
        indices = maskedIndices(mask)
        stats.tracedFraction = Double(indices.count) / Double(layout.width * layout.viewHeight)
    } else {
        stats.tracedFraction = 1
    }
    let rungs = ladderRungs(props)
    let caption = mainCaption(traced: masked ? stats.tracedFraction : 1)

    func renderFrame(_ f: Int, full: Bool) throws -> Double {
        let travel: Float = loop.travelWrapped(frame: f)
        let world = buildWorld(props, travel: travel)
        try renderer.setWorld(world)
        stats.gridSeconds += renderer.lastGridBuildSeconds
        let s = settings(for: bus, frame: f)
        let gpu: Double
        if full {
            gpu = try renderer.renderFull(camera: camera(f), settings: s, into: buffer,
                                          width: layout.width, viewHeight: layout.viewHeight)
        } else {
            gpu = try renderer.renderMasked(camera: camera(f), settings: s, into: buffer,
                                            width: layout.width, viewHeight: layout.viewHeight,
                                            indices: indices)
        }
        decorate(buffer, caption: caption, rungs: rungs)
        return gpu
    }

    // One palette for the whole loop, sampled a few frames apart.
    var samples: [RGB] = []
    _ = try renderFrame(0, full: true)
    samples += samplePixels(buffer, pixels: pixels, step: 7)
    let step = max(frames / 5, 1)
    for f in stride(from: step, to: frames, by: step) {
        _ = try renderFrame(f, full: !masked)
        samples += samplePixels(buffer, pixels: pixels, step: 7)
    }
    let palette = medianCutPalette(samples, count: 112)
    let quantizer = try Quantizer(device: renderer.device, palette: palette, pixels: pixels)

    let gif = GIFWriter(url: URL(fileURLWithPath: path), width: layout.width,
                        height: layout.height, palette: palette,
                        delayCentiseconds: loop.delayCentiseconds)
    var previous: [UInt8]?
    var changed = 0
    var compared = 0
    var boxArea = 0.0
    var boxFrames = 0
    stats.gridSeconds = 0
    let start = Date()
    for f in 0..<frames {
        // Frame 0 is always traced whole: it is the frame every later one
        // copies its still pixels from.
        stats.gpuSeconds += try renderFrame(f, full: !masked || f == 0)
        let indicesOut = try quantizer.indices(of: buffer)
        if let prev = previous {
            var x0 = layout.width, y0 = layout.height, x1 = -1, y1 = -1
            for y in 0..<layout.height {
                let row = y * layout.width
                for x in 0..<layout.width where indicesOut[row + x] != prev[row + x] {
                    changed += 1
                    x0 = min(x0, x); x1 = max(x1, x); y0 = min(y0, y); y1 = max(y1, y)
                }
            }
            compared += indicesOut.count
            if x1 >= 0 {
                boxArea += Double((x1 - x0 + 1) * (y1 - y0 + 1))
                boxFrames += 1
            }
        }
        previous = indicesOut
        gif.add(indicesOut)
        if f % 20 == 0 {
            FileHandle.standardError.write("  frame \(f)/\(frames)\r".data(using: .utf8)!)
        }
    }
    try gif.finish()
    stats.wallSeconds = Date().timeIntervalSince(start)
    stats.changedFraction = compared == 0 ? 0 : Double(changed) / Double(compared)
    stats.boxFraction = boxFrames == 0 ? 0
        : boxArea / Double(boxFrames) / Double(layout.width * layout.height)
    let attributes = try? FileManager.default.attributesOfItem(atPath: path)
    stats.bytes = (attributes?[.size] as? Int) ?? 0
    return stats
}

func report(_ stats: LoopStats, _ name: String) {
    let ms: Double = stats.gpuSeconds * 1000 / Double(max(stats.frames, 1))
    print(String(format: "%-16@ %3d frames  %6.1f ms/frame GPU  %6.1f s wall  traced %5.1f%%  changed %5.1f%%  change box %5.1f%%  %7.2f MB  %@",
                 name as NSString, stats.frames, ms, stats.wallSeconds,
                 stats.tracedFraction * 100, stats.changedFraction * 100,
                 stats.boxFraction * 100,
                 Double(stats.bytes) / 1_048_576, stats.path as NSString))
}

// MARK: - stills

func renderStills(bus: Bus, props: [Prop], renderer: SceneRenderer, buffer: MTLBuffer) throws {
    try FileManager.default.createDirectory(atPath: "renders", withIntermediateDirectories: true)
    let rungs = ladderRungs(props)
    let mask = try renderer.computeMask(camera: bus.camera(), settings: settings(for: bus, frame: 0),
                                        width: layout.width, viewHeight: layout.viewHeight)
    let indices = maskedIndices(mask)
    let traced = Double(indices.count) / Double(layout.width * layout.viewHeight)
    let caption = mainCaption(traced: traced)

    for f in [0, 30, 60] {
        let world = buildWorld(props, travel: loop.travelWrapped(frame: f))
        try renderer.setWorld(world)
        _ = try renderer.renderFull(camera: bus.camera(), settings: settings(for: bus, frame: f),
                                    into: buffer, width: layout.width, viewHeight: layout.viewHeight)
        decorate(buffer, caption: caption, rungs: rungs)
        try writeJPEG(buffer, layout: layout, to: "renders/frame\(f).jpg")
        print("  renders/frame\(f).jpg")
    }

    // The mask itself, painted over the picture: window pixels and the region
    // any sun stripe can touch.
    let world = buildWorld(props, travel: 0)
    try renderer.setWorld(world)
    _ = try renderer.renderFull(camera: bus.camera(), settings: settings(for: bus, frame: 0),
                                into: buffer, width: layout.width, viewHeight: layout.viewHeight)
    let p = buffer.contents().assumingMemoryBound(to: UInt8.self)
    for i in 0..<(layout.width * layout.viewHeight) where mask[i] == 0 {
        p[i * 4 + 0] = UInt8(Int(p[i * 4 + 0]) / 3)
        p[i * 4 + 1] = UInt8(Int(p[i * 4 + 1]) / 3)
        p[i * 4 + 2] = UInt8(Int(p[i * 4 + 2]) / 3)
    }
    decorate(buffer, caption: caption, rungs: rungs)
    drawPanelLabel("the \(Int((traced * 100).rounded()))% that can change", into: buffer,
                   layout: layout, centerX: CGFloat(layout.width) / 2,
                   top: CGFloat(layout.viewHeight) - 34 * layout.scale, size: 13 * layout.scale)
    try writeJPEG(buffer, layout: layout, to: "renders/mask.jpg")
    print("  renders/mask.jpg")

    // Sunlight alone, so the stripe region is visible as a thing rather than
    // as a slight brightening.
    var lit = settings(for: bus, frame: 0)
    lit.ambient = 0.02
    try renderer.setWorld(buildWorld(props, travel: 0))
    _ = try renderer.renderFull(camera: bus.camera(), settings: lit, into: buffer,
                                width: layout.width, viewHeight: layout.viewHeight)
    decorate(buffer, caption: caption, rungs: rungs)
    try writeJPEG(buffer, layout: layout, to: "renders/sunonly.jpg")
    print("  renders/sunonly.jpg")

    // No blur, for the comparison.
    var still = settings(for: bus, frame: 42)
    still.exposure = 0
    try renderer.setWorld(buildWorld(props, travel: loop.travelWrapped(frame: 42)))
    _ = try renderer.renderFull(camera: bus.camera(), settings: still, into: buffer,
                                width: layout.width, viewHeight: layout.viewHeight)
    decorate(buffer, caption: caption, rungs: rungs)
    try writeJPEG(buffer, layout: layout, to: "renders/noblur.jpg")
    print("  renders/noblur.jpg")
}

// MARK: - the measurements

func bench(bus: Bus, props: [Prop], renderer: SceneRenderer, buffer: MTLBuffer) throws {
    let world = buildWorld(props, travel: 0)
    try renderer.setWorld(world)
    print("bus:   \(bus.shapes.count) shapes, grid \(renderer.busGrid!.dims)")
    print("world: \(world.count) shapes, grid \(renderer.worldGrid!.dims), "
          + String(format: "%.1f per used box", renderer.worldGrid!.averageOccupancy))

    let mask = try renderer.computeMask(camera: bus.camera(), settings: settings(for: bus, frame: 0),
                                        width: layout.width, viewHeight: layout.viewHeight)
    let indices = maskedIndices(mask)
    let view = layout.width * layout.viewHeight
    print(String(format: "mask:  %d of %d pixels can change — %.1f%%",
                 indices.count, view, Double(indices.count) / Double(view) * 100))

    var fullBest = Double.greatestFiniteMagnitude
    var maskedBest = Double.greatestFiniteMagnitude
    for f in [10, 40, 70] {
        try renderer.setWorld(buildWorld(props, travel: loop.travelWrapped(frame: f)))
        let s = settings(for: bus, frame: f)
        let a = try renderer.renderFull(camera: bus.camera(), settings: s, into: buffer,
                                        width: layout.width, viewHeight: layout.viewHeight)
        let b = try renderer.renderMasked(camera: bus.camera(), settings: s, into: buffer,
                                          width: layout.width, viewHeight: layout.viewHeight,
                                          indices: indices)
        fullBest = min(fullBest, a)
        maskedBest = min(maskedBest, b)
    }
    print(String(format: "trace: every pixel %6.1f ms   masked %6.1f ms   %.2f× faster",
                 fullBest * 1000, maskedBest * 1000, fullBest / maskedBest))

    let masked = try renderLoop(bus: bus, props: props, renderer: renderer, buffer: buffer,
                                masked: true, orbiting: false, path: "renders/schoolbus.gif")
    report(masked, "bolted camera")
    let control = try renderLoop(bus: bus, props: props, renderer: renderer, buffer: buffer,
                                 masked: false, orbiting: true, path: "renders/orbit_control.gif")
    report(control, "moving camera")
    print(String(format: "\npayoff: %.2f× on GPU time, %.2f× on file size, changed pixels %.1f%% against %.1f%%",
                 control.gpuSeconds / masked.gpuSeconds,
                 Double(control.bytes) / Double(max(masked.bytes, 1)),
                 masked.changedFraction * 100, control.changedFraction * 100))
}

// MARK: - main

do {
    let specs = try loadSpecs(from: specsURL)
    let bus = buildBus(specs)
    let props = buildProps(loop, bus: bus)
    let device = try findDevice()
    let renderer = try SceneRenderer(device: device)
    let buffer = try makeFrameBuffer(device)
    try renderer.setBus(bus.shapes)
    print("GPU: \(device.name)")
    print("bus: \(specs.vehicle) — \(bus.shapes.count) shapes, "
          + String(format: "%.3f m wide inside, %.3f m of headroom, %d rows",
                   bus.halfWidth * 2, bus.ceiling, bus.rows))

    switch CommandLine.arguments.dropFirst().first {
    case "bench": try bench(bus: bus, props: props, renderer: renderer, buffer: buffer)
    case "stills": try renderStills(bus: bus, props: props, renderer: renderer, buffer: buffer)
    case "control":
        let s = try renderLoop(bus: bus, props: props, renderer: renderer, buffer: buffer,
                               masked: false, orbiting: true, path: "renders/orbit_control.gif")
        report(s, "moving camera")
    default:
        let s = try renderLoop(bus: bus, props: props, renderer: renderer, buffer: buffer,
                               masked: true, orbiting: false, path: "renders/schoolbus.gif")
        report(s, "bolted camera")
    }
} catch {
    FileHandle.standardError.write("schoolbus: \(error)\n".data(using: .utf8)!)
    exit(1)
}
