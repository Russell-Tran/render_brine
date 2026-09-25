// Step 14: one mysis shrimp swimming, as a darkfield micrograph, from a camera
// that never moves.
//
//   .build/mysis            render renders/mysis.gif
//   .build/mysis stills     write a few frames as PNGs, to look at them
//   .build/mysis bench      time the swept-bound mask, and the grid
//
// Environment overrides, so a short run needs no recompile:
//   MYSIS_WIDTH    frame width in pixels (default 1280)
//   MYSIS_FRAMES   render only this many frames
//   MYSIS_AT       render one frame, to look at a moment
//   MYSIS_GRID=0   brute force instead of the uniform grid
//   MYSIS_MASK=0   trace every pixel, swept bound or not
//   MYSIS_MUTATE   deliberately break something; see MMutations

import CoreGraphics
import CoreText
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

let width: Int = envInt("MYSIS_WIDTH", mFrameWidth)
let viewHeight: Int = width * mFrameViewHeight / mFrameWidth
let layout = FrameLayout(width: width, viewHeight: viewHeight,
                         captionHeight: width * mFrameCaptionHeight / mFrameWidth)
let useGrid: Bool = envFlag("MYSIS_GRID", true)
let useMask: Bool = envFlag("MYSIS_MASK", true)
let mutations = MMutations.fromEnvironment()
let posture = MysisPosture()
let reynolds = MReynolds()
let camera: Camera = mysisCamera(width: width, viewHeight: viewHeight)
let sigma: [SIMD3<Float>] = sigmaTableM()

// MARK: - The swept bound
//
// Built once, from every frame of the loop, because the camera is bolted down
// for every frame of the loop. If the mutation is on it is built from frame 0
// alone, which is what "not a bound any more" looks like.

let sweptFrames: [Int] = mutations.contains(.frameZeroMask)
    ? [0] : Array(0..<mysisFrameCount)

let swept: SweptBound = buildSweptBound(frames: sweptFrames, camera: camera,
                                        width: layout.width, height: layout.viewHeight) { f in
    poseMysis(frame: f, posture: posture, mutations: mutations).prims
}

// MARK: - The overlay
//
// Step 13's caption bar and evidence bar, unchanged. The scale bar and the
// anatomical labels are drawn here instead of through step 13's, because step
// 13's are dark ink for a bright field and this field is black.

private let labelInk = CGColor(srgbRed: 0.78, green: 0.86, blue: 0.92, alpha: 0.95)

func lightContext(_ buffer: MTLBuffer) -> CGContext? {
    CGContext(data: buffer.contents(), width: layout.width, height: layout.height,
              bitsPerComponent: 8, bytesPerRow: layout.width * 4,
              space: CGColorSpace(name: CGColorSpace.sRGB)!,
              bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
}

func drawLightText(_ s: String, _ ctx: CGContext, x: CGFloat, top: CGFloat, size: CGFloat,
                   bold: Bool = false, colour: CGColor = labelInk) {
    let name: String = bold ? "HelveticaNeue-Bold" : "HelveticaNeue"
    let font = CTFontCreateWithName(name as CFString, size, nil)
    let attributes: [NSAttributedString.Key: Any] = [
        NSAttributedString.Key(kCTFontAttributeName as String): font,
        NSAttributedString.Key(kCTForegroundColorAttributeName as String): colour,
    ]
    let line = CTLineCreateWithAttributedString(
        NSAttributedString(string: s, attributes: attributes))
    ctx.textPosition = CGPoint(x: x, y: CGFloat(layout.height) - top - size)
    CTLineDraw(line, ctx)
}

/// Where each label's text sits, and where its leader ends: worked out from the
/// pose's own anchors, so a label cannot drift from the thing it names.
func labelPlacements(_ pose: MysisPose) -> [(String, CGPoint, CGPoint)] {
    let order: [(String, String, CGFloat, CGFloat)] = [
        ("statocyst", "statocyst (1 of 2)", -150, 44),
        ("marsupium", "marsupium", -96, 96),
        ("hepatopancreas", "hepatopancreas", 92, -40),
    ]
    var out: [(String, CGPoint, CGPoint)] = []
    let k: CGFloat = layout.scale
    for (key, text, dx, dy) in order {
        guard let anchor: SIMD3<Float> = pose.labelAnchors[key] else { continue }
        let p: SIMD2<Float> = camera.project(anchor, width: layout.width, height: layout.viewHeight)
        let at = CGPoint(x: CGFloat(p.x), y: CGFloat(p.y))
        let to = CGPoint(x: at.x + dx * k, y: at.y + dy * k)
        out.append((text, at, to))
    }
    return out
}

func drawOverlay(into buffer: MTLBuffer, pose: MysisPose) {
    drawCaption(mysisCaption(), into: buffer, layout: layout)
    guard let ctx = lightContext(buffer) else { return }
    let k: CGFloat = layout.scale
    let h: CGFloat = CGFloat(layout.height)

    // The scale bar, in light ink because the field is black, and at the foot
    // of the picture rather than at its head: on a black field it has to sit
    // where the animal is not.
    let bar: CGFloat = CGFloat(5000 / micronsPerPixelM)        // 5 mm
    let x: CGFloat = 26 * k
    let baseline: CGFloat = CGFloat(layout.captionHeight) + 30 * k
    let pale = CGColor(srgbRed: 0.85, green: 0.90, blue: 0.94, alpha: 0.9)
    ctx.setFillColor(pale)
    ctx.fill(CGRect(x: x, y: baseline, width: bar, height: 3 * k))
    ctx.fill(CGRect(x: x, y: baseline - 4 * k, width: 2 * k, height: 11 * k))
    ctx.fill(CGRect(x: x + bar - 2 * k, y: baseline - 4 * k, width: 2 * k, height: 11 * k))
    let w: CGFloat = textWidth("5 mm", size: 11 * k, bold: true)
    drawLightText("5 mm", ctx, x: x + bar / 2 - w / 2, top: h - baseline - 18 * k,
                  size: 11 * k, bold: true, colour: pale)

    // the three labels, each with a leader back to the structure it names
    ctx.setStrokeColor(CGColor(srgbRed: 0.62, green: 0.74, blue: 0.84, alpha: 0.75))
    ctx.setLineWidth(1.0 * k)
    for (text, at, to) in labelPlacements(pose) {
        ctx.beginPath()
        ctx.move(to: CGPoint(x: at.x, y: h - at.y))
        ctx.addLine(to: CGPoint(x: to.x, y: h - to.y))
        ctx.strokePath()
        let tw: CGFloat = textWidth(text, size: 10.5 * k, bold: false)
        let tx: CGFloat = to.x < at.x ? to.x - tw : to.x + 4 * k
        drawLightText(text, ctx, x: tx, top: to.y - 7 * k, size: 10.5 * k)
    }
}

// MARK: - The palette
//
// Nearly every pixel of this frame is exactly black, which is the easiest thing
// a palette ever has to hold and the reason the step 8 encoder should do well
// here: an unchanged black pixel costs nothing at all. Index 0 is pure black so
// the background quantizes to itself with no error, a ramp along the cuticle's
// own colour keeps the soft edges from banding, and median cut is fed only the
// pixels that are not black.

func scatterRamp(count: Int) -> [RGB] {
    let base: SIMD3<Float> = sigma[MTissue.cuticle.rawValue]
    return (0..<count).map { i -> RGB in
        let t: Float = 3.2 * Float(i) / Float(count - 1)
        let c: SIMD3<Float> = darkfieldTone(base * t) * 255
        return RGB(UInt8(min(max(c.x.rounded(), 0), 255)),
                   UInt8(min(max(c.y.rounded(), 0), 255)),
                   UInt8(min(max(c.z.rounded(), 0), 255)))
    }
}

/// Every `step`th pixel that is not black.
func litSamples(_ frame: MTLBuffer, step: Int) -> [RGB] {
    let p = frame.contents().assumingMemoryBound(to: UInt8.self)
    var out: [RGB] = []
    out.reserveCapacity(layout.width * layout.viewHeight / step)
    var i = 0
    while i < layout.width * layout.viewHeight {
        let r: UInt8 = p[i * 4]
        let g: UInt8 = p[i * 4 + 1]
        let b: UInt8 = p[i * 4 + 2]
        if Int(r) + Int(g) + Int(b) > 6 { out.append(RGB(r, g, b)) }
        i += step
    }
    return out
}

// MARK: - PNG, for looking at single frames

func writePNG(_ buffer: MTLBuffer, to path: String) throws {
    guard let ctx = CGContext(data: buffer.contents(), width: layout.width,
                              height: layout.height, bitsPerComponent: 8,
                              bytesPerRow: layout.width * 4,
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

func renderFrame(_ f: Int, renderer: DarkfieldRenderer, buffer: MTLBuffer,
                 mask: Bool = true) throws -> (Double, Int) {
    let pose = poseMysis(frame: f, posture: posture, mutations: mutations)
    if useGrid { try renderer.buildGrid(pose.prims, density: 1) }
    let seconds = try renderer.render(prims: pose.prims, sigma: sigma, camera: camera,
                                      into: buffer, width: layout.width,
                                      viewHeight: layout.viewHeight,
                                      useGrid: useGrid,
                                      mask: (mask && useMask) ? swept : nil,
                                      mutations: mutations)
    drawOverlay(into: buffer, pose: pose)
    return (seconds, pose.prims.count)
}

func renderMysis(_ renderer: DarkfieldRenderer, _ buffer: MTLBuffer) throws {
    var frameCount: Int = mysisFrameCount
    let limit: Int = envInt("MYSIS_FRAMES", 0)
    if limit > 0 { frameCount = min(limit, mysisFrameCount) }
    let pixels: Int = layout.width * layout.height

    var samples: [RGB] = []
    for f in stride(from: 0, to: mysisFrameCount, by: max(mysisFrameCount / 6, 1)) {
        _ = try renderFrame(f, renderer: renderer, buffer: buffer)
        samples += litSamples(buffer, step: 3)
        samples += samplePixels(buffer, pixels: pixels, step: 601)   // the caption bar too
    }
    let palette: [RGB] = [RGB(0, 0, 0)] + scatterRamp(count: 40)
        + medianCutPalette(samples, count: 200)

    let quantizer = try Quantizer(device: renderer.device, palette: palette, pixels: pixels)
    let path = "renders/mysis.gif"
    let gif = GIFWriter(url: URL(fileURLWithPath: path), width: layout.width,
                        height: layout.height, palette: palette,
                        delayCentiseconds: frameDelayCentiseconds)

    var gpuSeconds = 0.0
    var buildSeconds = 0.0
    var overflow: UInt32 = 0
    var prims = 0
    var previous: [UInt8]? = nil
    var changedTotal = 0
    var boxTotal = 0
    var changedFrames = 0
    let viewPixels: Int = layout.width * layout.viewHeight
    let start = Date()
    for f in 0..<frameCount {
        let (seconds, n) = try renderFrame(f, renderer: renderer, buffer: buffer)
        gpuSeconds += seconds
        buildSeconds += renderer.lastBuildSeconds
        overflow += renderer.overflowCount
        prims = n
        let indices: [UInt8] = try quantizer.indices(of: buffer)
        if let prev = previous {
            // Two numbers, not one. The CHANGED FRACTION is what the frozen
            // camera buys. The BOUNDING BOX is what the encoder can actually
            // use, because it stores one rectangle per frame — and falling snow
            // scatters its changes over the whole frame, so the box stays big
            // however few pixels moved inside it.
            var changed = 0
            var x0 = layout.width, y0 = layout.viewHeight, x1 = -1, y1 = -1
            for y in 0..<layout.viewHeight {
                let row: Int = y * layout.width
                for x in 0..<layout.width where indices[row + x] != prev[row + x] {
                    changed += 1
                    if x < x0 { x0 = x }
                    if x > x1 { x1 = x }
                    if y < y0 { y0 = y }
                    if y > y1 { y1 = y }
                }
            }
            changedTotal += changed
            if x1 >= x0 && y1 >= y0 { boxTotal += (x1 - x0 + 1) * (y1 - y0 + 1) }
            changedFrames += 1
        }
        previous = indices
        gif.add(indices)
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
    let changedFraction: Double = changedFrames == 0 ? 0
        : Double(changedTotal) / Double(changedFrames) / Double(viewPixels)
    print(String(format: "mysis: %d frames, %d x %d, %d primitives -> %@ (%.2f MB)",
                 frameCount, layout.width, layout.height, prims, path, mb))
    print(String(format: "  %.1f ms GPU + %.1f ms grid per frame, %.0f s wall",
                 gpuSeconds * 1000 / n, buildSeconds * 1000 / n, wall))
    let boxFraction: Double = changedFrames == 0 ? 0
        : Double(boxTotal) / Double(changedFrames) / Double(viewPixels)
    print(String(format: "  changed pixels: %.2f%% of the view per frame, "
                 + "in a box covering %.1f%%", changedFraction * 100, boxFraction * 100))
    print(String(format: "  swept bound keeps %.1f%% of the %d tiles alive",
                 swept.liveFraction * 100, swept.tileCount))
    print(String(format: "  %.1f s displayed for %.2f s of animal time (x1/%.2f), %d beats",
                 loopDisplayedSeconds, loopRealSeconds, slowMotionFactor, cyclesPerLoop))
    print("  interval overflows: \(overflow)")
}

func renderStills(_ renderer: DarkfieldRenderer, _ buffer: MTLBuffer) throws {
    try FileManager.default.createDirectory(atPath: "renders", withIntermediateDirectories: true)
    for f in [0, 5, 10, 15, 60] {
        _ = try renderFrame(f, renderer: renderer, buffer: buffer)
        let path = String(format: "renders/frame%03d.png", f)
        try writePNG(buffer, to: path)
        print("  \(path)")
    }
}

func bench(_ renderer: DarkfieldRenderer, _ buffer: MTLBuffer) throws {
    let pose = poseMysis(frame: 30, posture: posture)
    print("scene: \(pose.prims.count) primitives")
    print(String(format: "swept world box: %.1f x %.1f x %.1f mm",
                 Double(swept.hi.x - swept.lo.x) / 1000,
                 Double(swept.hi.y - swept.lo.y) / 1000,
                 Double(swept.hi.z - swept.lo.z) / 1000))
    print(String(format: "swept screen mask: %d x %d tiles of %d px, %.1f%% alive",
                 swept.tilesX, swept.tilesY, maskTileSize, swept.liveFraction * 100))
    // The same bound over the animal alone. The gap between these two numbers
    // is the whole story of this optimisation: the animal is confined and the
    // snow is not, and a swept bound cannot tell them apart.
    let animalSwept: SweptBound = buildSweptBound(frames: Array(0..<mysisFrameCount),
                                                  camera: camera, width: layout.width,
                                                  height: layout.viewHeight) { f in
        animalPrims(poseMysis(frame: f, posture: posture))
    }
    print(String(format: "  the same bound over the animal alone: %.1f%% alive",
                 animalSwept.liveFraction * 100))

    func time(_ label: String, grid: Bool, mask: SweptBound?) throws -> Double {
        if grid { try renderer.buildGrid(pose.prims, density: 1) }
        var best = Double.greatestFiniteMagnitude
        for _ in 0..<5 {
            let s = try renderer.render(prims: pose.prims, sigma: sigma, camera: camera,
                                        into: buffer, width: layout.width,
                                        viewHeight: layout.viewHeight,
                                        useGrid: grid, mask: mask)
            best = min(best, s)
        }
        print(String(format: "  %-28s %7.2f ms", (label as NSString).utf8String!, best * 1000))
        return best
    }

    let bruteNoMask = try time("brute force, no mask", grid: false, mask: nil)
    let bruteMask = try time("brute force, swept mask", grid: false, mask: swept)
    let gridNoMask = try time("grid, no mask", grid: true, mask: nil)
    let gridMask = try time("grid, swept mask", grid: true, mask: swept)
    print(String(format: "the grid is %.1fx brute force; the swept mask is %.2fx on top of it, "
                 + "%.2fx on brute force",
                 bruteNoMask / gridNoMask, gridNoMask / gridMask, bruteNoMask / bruteMask))

    // And the same measurement on the scene the bound was designed for.
    print("the animal alone, no conveyor:")
    let animalOnly: [GPUPrim] = animalPrims(pose)
    func timeAnimal(_ label: String, mask: SweptBound?) throws -> Double {
        try renderer.buildGrid(animalOnly, density: 1)
        var best = Double.greatestFiniteMagnitude
        for _ in 0..<5 {
            let s = try renderer.render(prims: animalOnly, sigma: sigma, camera: camera,
                                        into: buffer, width: layout.width,
                                        viewHeight: layout.viewHeight,
                                        useGrid: true, mask: mask)
            best = min(best, s)
        }
        print(String(format: "  %-28s %7.2f ms", (label as NSString).utf8String!, best * 1000))
        return best
    }
    let aNoMask = try timeAnimal("grid, no mask", mask: nil)
    let aMask = try timeAnimal("grid, swept mask", mask: animalSwept)
    print(String(format: "  the swept mask is %.2fx when nothing crosses the frame", aNoMask / aMask))
}

// MARK: - main

do {
    let device = try findDevice()
    let renderer = try DarkfieldRenderer(device: device)
    guard let buffer = device.makeBuffer(length: layout.width * layout.height * 4,
                                         options: .storageModeShared) else {
        throw RenderError.gpu("could not allocate the frame")
    }
    let pose = poseMysis(frame: 0, posture: posture, mutations: mutations)
    print("GPU: \(device.name)")
    print("Mysis diluviana, adult female: "
          + "\(pose.counts["thoracopod pair"] ?? 0) thoracopod pairs "
          + "(\(pose.counts["maxilliped pair"] ?? 0) maxilliped, "
          + "\(beatingPairs) beating), "
          + "\(pose.counts["statocyst"] ?? 0) statocysts, "
          + "\(pose.counts["oostegite pair"] ?? 0) oostegite pairs, "
          + "\(pose.counts["seta"] ?? 0) setae")
    print(String(format: "Re: %.0f body, %.1f exopod, %.3f seta \u{2014} "
                 + "the animal is inertial, its setae are not",
                 reynolds.body, reynolds.exopod, reynolds.seta))
    print(String(format: "darkfield: I_bg = 0, deposit at interfaces, "
                 + "blue scatters %.1fx red",
                 Double(rayleighWeights(exponent: 4).z / rayleighWeights(exponent: 4).x)))
    print("\(pose.prims.count) primitives, "
          + (useGrid ? "uniform grid" : "brute force")
          + (useMask ? ", swept-bound mask" : ", no mask")
          + (mutations.isEmpty ? "" : "  MUTATED: \(mutations.rawValue)"))

    let single: Int = envInt("MYSIS_AT", -1)
    if single >= 0 {
        _ = try renderFrame(single, renderer: renderer, buffer: buffer)
        try writePNG(buffer, to: String(format: "renders/frame%03d.png", single))
        print("  renders/frame\(single).png")
    } else {
        switch CommandLine.arguments.dropFirst().first {
        case "stills": try renderStills(renderer, buffer)
        case "bench": try bench(renderer, buffer)
        default: try renderMysis(renderer, buffer)
        }
    }
} catch {
    FileHandle.standardError.write("mysis: \(error)\n".data(using: .utf8)!)
    exit(1)
}
