// Step 11: AraC's light switch. The protein ties pGLO's DNA in a loop to keep
// GFP switched off, and unties it when arabinose arrives.
//
// This is a genuine cycle, not a rewind: the sugar arrives, the grip moves from
// araO2 to araI2, the loop opens, RNA polymerase lands, then the sugar leaves
// and the loop re-forms. Every stage runs forwards.

import Foundation
import simd

let layout = FrameLayout(width: 960, viewHeight: 600, captionHeight: 120)
let delay = 10                     // hundredths of a second: 10 fps
let paletteSize = 96
let seconds = 24.0
let switchURL = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent().deletingLastPathComponent()
    .appendingPathComponent("Resources/switch.json")

/// Environment overrides, so a short run needs no recompile:
///   SW_FRAMES=8 make run      render only the first few frames
///   SW_AT=120 make run        render one frame, to look at a moment
///   SW_AO=0 make run          turn ambient occlusion off
///   SW_GRID=0 make run        brute force instead of the grid
// Spelled out in typed steps: the laptop's older Swift type checker is slower
// and chained `??` sits near the -warn-long-expression-type-checking limit.
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

let probes: Int = envInt("SW_PROBES", 12)
let aoOn: Bool = envFlag("SW_AO", true)
let useGrid: Bool = envFlag("SW_GRID", true)
let frameLimit: Int = envInt("SW_FRAMES", 0)
let openSpan: Float = envFloat("SW_OPEN", 0.68)
let cameraDistance: Float = envFloat("SW_DIST", 780)

/// A palette that always keeps a few shades of each scene colour. Median cut
/// alone gives colours to whatever covers the most pixels, so the small bright
/// things — the sugar, the half-sites — lose out and come back the wrong hue.
/// Step 8 hit this with phosphorus turning red.
func switchPalette(samples: [RGB], count: Int) -> [RGB] {
    var reserved: [RGB] = []
    for c in [dnaColor, dnaBackbone, siteColor, promoterColor, coreColor, dbdColor, sugarColor, rnapColor] {
        for light: Float in [0.35, 0.6, 0.85, 1.1] {
            let v = simd_clamp(c * light, SIMD3<Float>(repeating: 0), SIMD3<Float>(repeating: 1)) * 255
            reserved.append(RGB(UInt8(v.x.rounded()), UInt8(v.y.rounded()), UInt8(v.z.rounded())))
        }
    }
    return reserved + medianCutPalette(samples, count: max(count - reserved.count, 8))
}

do {
    let d = try loadSwitch(from: switchURL)
    let device = try findDevice()
    let renderer = try SceneRenderer(device: device)
    guard let buffer = device.makeBuffer(length: layout.width * layout.height * 4,
                                         options: .storageModeShared) else {
        throw RenderError.gpu("could not allocate the frame")
    }
    let ao = aoOn ? AOSettings(probes: probes, distance: envFloat("SW_AO_DIST", 8.0),
                               strength: 1.0, contrast: envFloat("SW_AO_POW", 1.7), only: false)
                  : AOSettings.off
    print("GPU: \(device.name)")
    print("araO2 \(d.sites["araO2"]!) · araI1 \(d.sites["araI1"]!) · araI2 \(d.sites["araI2"]!)"
          + " · spacing \(d.spacingBP) bp")
    print("\(d.nBP) bp of DNA, ambient occlusion "
          + (aoOn ? "\(probes) probes" : "off") + ", " + (useGrid ? "uniform grid" : "brute force"))

    let totalFrames = Int((seconds * 100 / Double(delay)).rounded())
    let pixels = layout.width * layout.height
    var totalBuild = 0.0
    var totalTrace = 0.0
    var lastCost = FrameCost()

    /// Renders frame `f`. The camera is fixed: unlike every earlier step, the
    /// SCENE is what moves, so there is nothing to be gained by orbiting.
    func renderFrame(_ f: Int) throws {
        let u = Double(f) / Double(totalFrames)
        let s = state(d, at: u, openSpan: openSpan)
        let shapes = sceneShapes(d, s)
        // Fixed camera: unlike every earlier step the SCENE is what moves, so
        // there is nothing to gain by orbiting. Framed to hold both the shut
        // loop and the open one without a cut.
        let camera = Camera.orbit(target: SIMD3(-30, 130, 0), distance: cameraDistance,
                                  yaw: 0, pitch: 10, fov: 30)
        if useGrid {
            let cost = try renderer.frame(shapes: shapes, camera: camera, into: buffer,
                                          width: layout.width, viewHeight: layout.viewHeight, ao: ao)
            totalBuild += cost.build
            totalTrace += cost.trace
            lastCost = cost
        } else {
            totalTrace += try renderer.render(shapes: shapes, camera: camera, into: buffer,
                                              width: layout.width, viewHeight: layout.viewHeight,
                                              ao: ao, useGrid: false)
            lastCost.shapes = shapes.count
        }
        drawCaption(s.caption, into: buffer, layout: layout)
    }

    let frameCount = frameLimit > 0 ? min(frameLimit, totalFrames) : totalFrames
    let single: Int = envInt("SW_AT", -1)

    var samples: [RGB] = []
    for f in stride(from: 0, to: totalFrames, by: max(totalFrames / 6, 1)) {
        try renderFrame(f)
        samples += samplePixels(buffer, pixels: pixels, step: 7)
    }
    let palette = switchPalette(samples: samples, count: paletteSize)
    let quantizer = try Quantizer(device: device, palette: palette, pixels: pixels)

    let path = "renders/switch.gif"
    let gif = GIFWriter(url: URL(fileURLWithPath: path), width: layout.width, height: layout.height,
                        palette: palette, delayCentiseconds: delay)
    totalBuild = 0; totalTrace = 0
    let start = Date()
    let frames = single >= 0 ? [single] : Array(0..<frameCount)
    for f in frames {
        try renderFrame(f)
        gif.add(try quantizer.indices(of: buffer))
    }
    try gif.finish()
    let kb = ((try? FileManager.default.attributesOfItem(atPath: path)[.size] as? Int) ?? 0) / 1024
    let n = Double(frames.count)
    let buildMS = totalBuild * 1000 / n
    let traceMS = totalTrace * 1000 / n
    print(String(format: "switch: %d of %d frames, %d × %d, %d shapes → %@ (%d KB)",
                 frames.count, totalFrames, layout.width, layout.height, lastCost.shapes, path, kb))
    print(String(format: "  grid rebuild %.1f ms + trace %.0f ms = %.0f ms per frame  (rebuild is %.1f%%)",
                 buildMS, traceMS, buildMS + traceMS, 100 * buildMS / max(buildMS + traceMS, 1e-9)))
    print(String(format: "  grid: %d boxes, %.1f shapes per used box", lastCost.cells, lastCost.occupancy))
    // Typed steps: the laptop's older type checker is slow on the chained form.
    let perFrame: Double = buildMS + traceMS
    let projected: Double = perFrame * Double(totalFrames) / 1000
    let wall: Double = Date().timeIntervalSince(start)
    print(String(format: "  wall %.0f s; full loop would be %.0f s of GPU time", wall, projected))
} catch {
    FileHandle.standardError.write("switch: \(error)\n".data(using: .utf8)!)
    exit(1)
}
