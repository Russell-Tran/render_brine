// Step 19: a molar in section, with the stress field it is actually carrying
// painted on the cut face, and the cracks drawn as they run.
//
//   .build/tooth          render renders/tooth.gif
//   .build/tooth stills   write a few frames as PNGs
//   .build/tooth physics  print the numbers and nothing else
//   .build/tooth bench    time brute force against the uniform grid
//
// Environment overrides, so a short run needs no recompile:
//   TOOTH_WIDTH    frame width in pixels (default 960 — this is a PORTRAIT step)
//   TOOTH_FRAMES   render only this many frames
//   TOOTH_AT       render one frame, to look at a moment
//   TOOTH_MESH     lattice spacing in mm (default 0.10)
//   TOOTH_GRID=0   brute force instead of the grid

import CoreGraphics
import Foundation
import ImageIO
import Metal
import simd

let env: [String: String] = ProcessInfo.processInfo.environment
func envInt(_ name: String, _ fallback: Int) -> Int {
    guard let text: String = env[name], let value = Int(text) else { return fallback }
    return value
}
func envDouble(_ name: String, _ fallback: Double) -> Double {
    guard let text: String = env[name], let value = Double(text) else { return fallback }
    return value
}
func envFlag(_ name: String, _ fallback: Bool) -> Bool {
    guard let text: String = env[name] else { return fallback }
    return text != "0"
}

let width: Int = envInt("TOOTH_WIDTH", frameWidth)
let viewHeight: Int = width * frameViewHeight / frameWidth
let layout = FrameLayout(width: width, viewHeight: viewHeight,
                         captionHeight: width * frameCaptionHeight / frameWidth)
let useGrid: Bool = envFlag("TOOTH_GRID", true)
let meshMM: Double = envDouble("TOOTH_MESH", 0.10)
let camera: Camera = cutawayCamera(width: width, height: viewHeight,
                                   target: cameraTarget, yaw: cameraYaw)
let pixelCount: Int = layout.width * layout.height

func orDie<T>(_ what: String, _ body: () throws -> T) -> T {
    do { return try body() } catch {
        FileHandle.standardError.write("tooth: \(what): \(error)\n".data(using: .utf8)!)
        exit(1)
    }
}

func note(_ s: String) {
    FileHandle.standardError.write((s + "\n").data(using: .utf8)!)
}

// MARK: - the physics, solved once
//
// Three models and one ramp, and then nothing is solved again for the rest of
// the run. Two of the beats are pure linear elasticity, so ONE solve covers the
// whole of their load ramp by multiplication — that is not a shortcut, it is
// what linear means, and the test that checks it is the round trip to zero.

struct PhysicsState {
    var field: StressField
    var newtons: Float
    var broken: Int
    var label: String
}

func solvedStates() -> ([PhysicsState], ToothModel.RampResult, ToothModel, ToothModel) {
    var states: [PhysicsState] = []

    let chewing = ToothModel(spacingMM: meshMM, loadCase: axialChewing, scatter: 0)
    let a = chewing.solve()
    note(String(format: "  chewing  %d nodes, %d bonds, residual %.1e in %d iterations",
                chewing.cols * chewing.rows, chewing.lattice.bonds.count, a.residual, a.iterations))
    var chewField = StressField.standard()
    chewField.paint(chewing)
    states.append(PhysicsState(field: chewField, newtons: Float(axialChewing.newtons),
                               broken: 0, label: "chewing"))

    let brux = ToothModel(spacingMM: meshMM, loadCase: lateralBruxing, scatter: 0)
    let b = brux.solve()
    note(String(format: "  bruxing  residual %.1e in %d iterations, %.0f µm of movement",
                b.residual, b.iterations, brux.lattice.maxDisplacement * 1e6))
    var bruxField = StressField.standard()
    bruxField.paint(brux)
    states.append(PhysicsState(field: bruxField, newtons: Float(lateralBruxing.newtons),
                               broken: 0, label: "bruxing, elastic"))

    // The ramp. Every solve along it is kept, because the crack growing is what
    // beats four and five are.
    let cracking = ToothModel(spacingMM: meshMM, loadCase: lateralBruxing, seed: crackSeed)
    cracking.solve()
    var snapshots: [PhysicsState] = []
    var lastBroken = -1
    // The ramp runs all the way to the load that pushes the crack into the
    // dentin, because that load IS the arrest result. Only the states before
    // that are kept as frames: past it the tooth splits in one step and there is
    // nothing to watch.
    let result = cracking.ramp(from: 0.5, to: 1.4, factor: 1.006) { model in
        let broken: Int = model.lattice.breakOrder.count
        guard broken > 0, broken != lastBroken else { return }
        guard broken < 120, model.dentinCoreBreaks() == 0 else { return }
        lastBroken = broken
        var f = StressField.standard()
        f.paint(model)
        f.paintCracks(model)
        snapshots.append(PhysicsState(field: f, newtons: model.currentNewtons,
                                      broken: broken, label: "cracking"))
    }
    note(String(format: "  ramp     first bond at %.0f N, junction at %.0f N, dentin at %.0f N"
                + "  →  arrest margin %.2f×",
                result.initiationNewtons, result.junctionNewtons,
                result.dentinNewtons, result.arrestMargin))
    states += snapshots
    return (states, result, cracking, brux)
}

/// Which seed the render's crack uses. The arrest TEST is a rate over many
/// seeds; the render has to pick one, and picking it in a named constant is the
/// difference between choosing a seed and hiding one.
let crackSeed: UInt64 = 1

// MARK: - the shot list

func shot(frame f: Int, states: [PhysicsState]) -> Shot {
    let b: Beat = beat(frame: f)
    let t: Float = beatProgress(frame: f)
    let crackStates: Int = max(states.count - 2, 1)
    switch b {
    case .anatomy:
        return Shot(beat: b, state: 0, gain: 0, stressMix: 0,
                    labelFade: smoothstep(0.05, 0.3, t), keyFade: 0, newtons: 0)
    case .chewing:
        let ramp: Float = smoothstep(0, 0.45, t)
        return Shot(beat: b, state: 0, gain: ramp, stressMix: ramp * 0.85,
                    labelFade: 1 - smoothstep(0.1, 0.5, t), keyFade: ramp,
                    newtons: Float(axialChewing.newtons) * ramp)
    case .bruxing:
        let ramp: Float = smoothstep(0, 0.6, t)
        return Shot(beat: b, state: 1, gain: ramp, stressMix: 0.85,
                    labelFade: 0, keyFade: 1, newtons: Float(lateralBruxing.newtons) * ramp)
    case .cracking:
        let k: Int = min(Int(t * Float(crackStates)), crackStates - 1)
        return Shot(beat: b, state: 2 + k, gain: 1, stressMix: 0.85,
                    labelFade: 0, keyFade: 1, newtons: states[2 + k].newtons)
    case .arrest:
        return Shot(beat: b, state: states.count - 1, gain: 1, stressMix: 0.85,
                    labelFade: smoothstep(0.15, 0.4, t), keyFade: 1,
                    newtons: states[states.count - 1].newtons)
    case .lesion:
        let fade: Float = 1 - smoothstep(0.55, 0.95, t)
        return Shot(beat: b, state: states.count - 1, gain: 1, stressMix: 0.85 * fade,
                    labelFade: fade, keyFade: fade,
                    newtons: states[states.count - 1].newtons * fade)
    }
}

func caption(for s: Shot, ramp: ToothModel.RampResult) -> Caption {
    var facts: String
    switch s.beat {
    case .anatomy:
        facts = "84 GPa shell · 0.18 mm graded junction · 18 GPa core · 0.2 mm of ligament"
    case .chewing:
        facts = String(format: "%.0f N down the long axis — nothing here is near failing",
                       s.newtons)
    case .bruxing:
        facts = String(format: "%.0f N at 27° on the cusp's inner incline · enamel fails at 30 MPa",
                       s.newtons)
    case .cracking:
        facts = String(format: "first bond at %.0f N · now at %.0f N",
                       ramp.initiationNewtons, s.newtons)
    case .arrest:
        facts = String(format: "it takes %.2f× the load that started the crack to push it "
                       + "out of the junction and into the dentin", ramp.arrestMargin)
    case .lesion:
        facts = "the same 1000 N on the other incline gives half the cervical tension"
    }
    return Caption(title: "The neck of a tooth",
                   subtitle: captionFor(s.beat),
                   facts: facts,
                   aside: "")
}

// MARK: - rendering

let device: MTLDevice = orDie("no GPU") { try findDevice() }
let renderer: CutawayRenderer = orDie("kernel") { try CutawayRenderer(device: device) }
let frameBuffer: MTLBuffer = orDie("frame buffer") { () -> MTLBuffer in
    guard let b = device.makeBuffer(length: pixelCount * 4, options: .storageModeShared) else {
        throw RenderError.gpu("could not allocate the frame")
    }
    return b
}
let prims: [GPUPrim] = toothPrimitives()

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
    guard let dest = CGImageDestinationCreateWithURL(URL(fileURLWithPath: path) as CFURL,
                                                     "public.png" as CFString, 1, nil) else {
        throw RenderError.gpu("could not open \(path)")
    }
    CGImageDestinationAddImage(dest, image, nil)
    CGImageDestinationFinalize(dest)
}

var scratchField = StressField.standard()

@discardableResult
func renderFrame(_ f: Int, states: [PhysicsState], ramp: ToothModel.RampResult,
                 into buffer: MTLBuffer, grid: Bool = useGrid) throws -> Double {
    let s: Shot = shot(frame: f, states: states)
    var look = Look()
    look.stressMix = s.stressMix
    let source: StressField = states[min(s.state, states.count - 1)].field
    if s.gain == 1 {
        scratchField = source
    } else {
        scratchField = source
        for i in 0..<scratchField.texels.count { scratchField.texels[i].x *= s.gain }
    }
    let seconds = try renderer.render(prims: prims, albedo: toothAlbedo, camera: camera,
                                      look: look, field: scratchField, into: buffer,
                                      width: layout.width, viewHeight: layout.viewHeight,
                                      useGrid: grid)
    drawCutBoundaries(frame: buffer, aux: renderer.auxPointer!,
                      width: layout.width, viewHeight: layout.viewHeight)
    drawLabels(toothLabels(layout: layout), into: buffer, layout: layout,
               camera: camera, fade: s.labelFade)
    drawStressKey(into: buffer, layout: layout, range: look.stressRangeMPa,
                  fade: s.keyFade, threshold: Float(enamelStrength / 1e6))
    drawScaleRule(into: buffer, layout: layout, millimetres: 5)
    drawCaption(caption(for: s, ramp: ramp), into: buffer, layout: layout)
    drawStandingBar(standingRows(s.beat), into: buffer, layout: layout,
                    leftEdge: CGFloat(layout.width) * 0.50)
    return seconds
}

func snapshot(_ buffer: MTLBuffer) -> [UInt8] {
    let p = buffer.contents().assumingMemoryBound(to: UInt8.self)
    return Array(UnsafeBufferPointer(start: p, count: pixelCount * 4))
}

func renderLoop(states: [PhysicsState], ramp: ToothModel.RampResult) throws {
    var frameCount: Int = toothFrameCount
    let limit: Int = envInt("TOOTH_FRAMES", 0)
    if limit > 0 { frameCount = min(limit, toothFrameCount) }

    try renderFrame(0, states: states, ramp: ramp, into: frameBuffer)
    let fresh: [UInt8] = snapshot(frameBuffer)

    var samples: [RGB] = []
    for f in stride(from: 0, to: toothFrameCount, by: max(toothFrameCount / 10, 1)) {
        try renderFrame(f, states: states, ramp: ramp, into: frameBuffer)
        samples += samplePixels(frameBuffer, pixels: pixelCount, step: 7)
        samples += samplePixels(frameBuffer, pixels: pixelCount, step: 397)
    }
    let palette = medianCutPalette(samples, count: 250)
    let quantizer = try Quantizer(device: device, palette: palette, pixels: pixelCount)
    let path = "renders/tooth.gif"
    try FileManager.default.createDirectory(atPath: "renders", withIntermediateDirectories: true)
    let gif = GIFWriter(url: URL(fileURLWithPath: path), width: layout.width,
                        height: layout.height, palette: palette,
                        delayCentiseconds: frameDelayCentiseconds)

    var gpuSeconds = 0.0
    var overflow: UInt32 = 0
    let start = Date()
    for f in 0..<frameCount {
        gpuSeconds += try renderFrame(f, states: states, ramp: ramp, into: frameBuffer)
        overflow += renderer.overflowCount
        let alpha: Float = dissolve(frame: f)
        if alpha > 0 {
            crossDissolve(frame: frameBuffer, toward: fresh, alpha: alpha, pixels: pixelCount)
        }
        gif.add(try quantizer.indices(of: frameBuffer))
        if f % 12 == 0 { note("  frame \(f)/\(frameCount)") }
    }
    try gif.finish()
    let attributes = try? FileManager.default.attributesOfItem(atPath: path)
    let bytes: Int = (attributes?[.size] as? Int) ?? 0
    let n: Double = Double(frameCount)
    let wall: Double = Date().timeIntervalSince(start)
    print(String(format: "tooth: %d frames, %d × %d, %d primitives → %@ (%.1f MB)",
                 frameCount, layout.width, layout.height, prims.count, path,
                 Double(bytes) / 1_048_576))
    print(String(format: "  %.1f ms GPU per frame, %.0f s wall, %.1f s of loop",
                 gpuSeconds * 1000 / n, wall, loopSeconds))
    print("  interval overflows: \(overflow)")
}

// MARK: - main

do {
    print("GPU: \(device.name)")
    print("solving the section …")
    let (states, ramp, cracked, elastic) = solvedStates()
    _ = cracked
    if useGrid { try renderer.buildGrid(prims, density: 1) }
    let peakCervical: Float = elastic.peakSurfaceTension(between: -2, and: 1)
    let skin = elastic.enamelSkinPeak(between: -0.3, and: 1.5)
    print(String(format: "%d primitives, %d physics states, %d bonds broken",
                 prims.count, states.count, ramp.totalBroken))
    print(String(format: "cervical surface tension %.0f MPa at 1000 N (junction band; the 60 µm "
                 + "enamel skin on it is below mesh, and carries %.0f MPa)",
                 peakCervical / 1e6, skin.stressPa / 1e6))
    print(String(format: "crack starts at %.0f N, reaches the junction at %.0f N, and needs "
                 + "%.2f× the starting load to leave it for the dentin",
                 ramp.initiationNewtons, ramp.junctionNewtons, ramp.arrestMargin))

    let single: Int = envInt("TOOTH_AT", -1)
    if single >= 0 {
        try renderFrame(single, states: states, ramp: ramp, into: frameBuffer)
        try writePNG(frameBuffer, to: String(format: "renders/frame%03d.png", single))
        print("  renders/frame\(single).png")
    } else {
        switch CommandLine.arguments.dropFirst().first {
        case "stills":
            for f in [0, 30, 54, 78, 102, 126] {
                try renderFrame(f, states: states, ramp: ramp, into: frameBuffer)
                let path = String(format: "renders/frame%03d.png", f)
                try writePNG(frameBuffer, to: path)
                print("  \(path)")
            }
        case "physics":
            break
        case "bench":
            var brute = Double.greatestFiniteMagnitude
            for _ in 0..<3 {
                let s = try renderer.render(prims: prims, albedo: toothAlbedo, camera: camera,
                                            look: Look(), field: states[0].field,
                                            into: frameBuffer, width: layout.width,
                                            viewHeight: layout.viewHeight, useGrid: false)
                brute = min(brute, s)
            }
            print(String(format: "brute force   %7.1f ms", brute * 1000))
            for density in [Float(0.5), 1, 2] {
                try renderer.buildGrid(prims, density: density)
                var best = Double.greatestFiniteMagnitude
                for _ in 0..<3 {
                    let s = try renderer.render(prims: prims, albedo: toothAlbedo,
                                                camera: camera, look: Look(),
                                                field: states[0].field, into: frameBuffer,
                                                width: layout.width,
                                                viewHeight: layout.viewHeight, useGrid: true)
                    best = min(best, s)
                }
                let g = renderer.grid!
                print(String(format: "grid ×%-4.1f    %7.1f ms   %3d × %3d × %3d   %4.1f× faster",
                             density, best * 1000, g.dims.x, g.dims.y, g.dims.z, brute / best))
            }
        default:
            try renderLoop(states: states, ramp: ramp)
        }
    }
} catch {
    FileHandle.standardError.write("tooth: \(error)\n".data(using: .utf8)!)
    exit(1)
}
