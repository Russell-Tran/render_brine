// Step 9: green fluorescent protein (PDB 1EMA), space-filling, turning once
// about the barrel's own axis, with the front dissolving away part-way round
// to show the chromophore inside. A seamless looping GIF.

import Foundation
import simd

// Size and speed follow step 8, which measured them: a turning molecule
// changes several percent of the pixels every frame and those pixels cost
// about a byte each, even with one shared palette and only-what-changed
// frames. 960 px is about the width GitHub shows a README image at.
let layout = FrameLayout(width: 960, viewHeight: 600, captionHeight: 120)
let delay = 10                           // hundredths of a second: 10 fps
let paletteSize = 96
let seconds = 20.0
let cameraDistance: Float = 118
let proteinURL = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent().deletingLastPathComponent()
    .appendingPathComponent("Resources/gfp.json")

/// Environment overrides, so a short test run needs no recompile:
///   GFP_FRAMES=8 make run     render only the first few frames
///   GFP_AO=0 make run         turn ambient occlusion off
///   GFP_PROBES=16 make run    change how many probe rays each hit casts
// Spelled out in typed steps rather than chained `??`: the laptop's older
// Swift type checker is slower, and the chained form sat right on the
// -warn-long-expression-type-checking limit even here.
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

let probes: Int = envInt("GFP_PROBES", 12)
let aoOn: Bool = envFlag("GFP_AO", true)
let frameLimit: Int = envInt("GFP_FRAMES", 0)
let aoOnly: Bool = envFlag("GFP_AO_ONLY", false)    // show the occlusion term by itself
let aoDistance: Float = envFloat("GFP_AO_DIST", 8.0)

func caption(for protein: Protein, opening: Float) -> Caption {
    let subtitle = opening > 0.5
        ? "inside: the chromophore, which GFP builds from three of its own amino acids"
        : "an eleven-stranded barrel · X-ray crystal structure (PDB 1EMA, S65T)"
    let facts = opening > 0.5
        ? "Thr65–Tyr66–Gly67 fuse into one residue · no enzyme, no cofactor · the barrel keeps water away from it"
        : String(format: "%d residues · %d strands · wall %.0f Å across · coloured from the N end (blue) to the C end (red)",
                 protein.residueCount, protein.strandCount, 2 * protein.wallRadius)
    return Caption(title: "Green fluorescent protein",
                   subtitle: subtitle,
                   facts: facts,
                   aside: "1,771 atoms, space-filling")
}

do {
    let protein = try loadProtein(from: proteinURL)
    let device = try findDevice()
    let renderer = try MoleculeRenderer(device: device)
    guard let buffer = device.makeBuffer(length: layout.width * layout.height * 4,
                                         options: .storageModeShared) else {
        throw RenderError.gpu("could not allocate the frame")
    }
    let ao = aoOn ? AOSettings(probes: probes, distance: aoDistance, strength: 1.0, only: aoOnly,
                               contrast: envFloat("GFP_AO_POW", 1.7)) : AOSettings.off
    print("GPU: \(device.name)")
    print("\(protein.atoms.count) atoms, ambient occlusion \(aoOn ? "\(probes) probes, \(ao.distance) Å" : "off")")

    let totalFrames = Int((seconds * 100 / Double(delay)).rounded())
    let pixels = layout.width * layout.height

    /// Renders frame `f` of `totalFrames` into `buffer`. The molecule never
    /// moves: the camera goes round it and the lights follow, which is what
    /// keeps the occlusion steady from frame to frame.
    func renderFrame(_ f: Int) throws -> Double {
        let u = Double(f) / Double(totalFrames)
        let yaw = Float(360.0 * u)
        let camera = Camera.orbit(target: .zero, distance: cameraDistance, yaw: yaw, pitch: 6, fov: 30)
        let open = opening(at: u)
        let cut = cutFactors(protein, opening: open, camera: camera)
        let (spheres, cylinders) = sceneGeometry(protein, cut: cut, showChromophore: open > 0.01)
        let t = try renderer.render(spheres: spheres, cylinders: cylinders, camera: camera, into: buffer,
                                    width: layout.width, viewHeight: layout.viewHeight,
                                    ao: ao, lightYaw: yaw)
        drawCaption(caption(for: protein, opening: open), into: buffer, layout: layout)
        return t
    }

    let frameCount = frameLimit > 0 ? min(frameLimit, totalFrames) : totalFrames
    // GFP_AT=125 renders just that one frame, for looking at a moment quickly.
    let single: Int = envInt("GFP_AT", -1)

    // One palette for the whole loop, from frames spread across it so the
    // open phase (and its green) is represented as well as the closed one.
    var samples: [RGB] = []
    for f in stride(from: 0, to: totalFrames, by: max(totalFrames / 6, 1)) {
        _ = try renderFrame(f)
        samples += samplePixels(buffer, pixels: pixels, step: 7)
    }
    let palette = gfpPalette(samples: samples, count: paletteSize)
    let quantizer = try Quantizer(device: device, palette: palette, pixels: pixels)

    let path = "renders/gfp.gif"
    let gif = GIFWriter(url: URL(fileURLWithPath: path), width: layout.width, height: layout.height,
                        palette: palette, delayCentiseconds: delay)
    var gpuSeconds = 0.0
    let start = Date()
    let frames = single >= 0 ? [single] : Array(0..<frameCount)
    for f in frames {
        gpuSeconds += try renderFrame(f)
        gif.add(try quantizer.indices(of: buffer))
    }
    try gif.finish()
    let kb = ((try? FileManager.default.attributesOfItem(atPath: path)[.size] as? Int) ?? 0) / 1024
    let msPerFrame = gpuSeconds * 1000 / Double(frames.count)
    print(String(format: "gfp: %d of %d frames, %d × %d, %.0f ms GPU per frame, %.0f s wall → %@ (%d KB)",
                 frameCount, totalFrames, layout.width, layout.height, msPerFrame,
                 Date().timeIntervalSince(start), path, kb))
    if frameCount < totalFrames {
        print(String(format: "  full loop would be %.0f s of GPU time (%d frames)",
                     msPerFrame * Double(totalFrames) / 1000, totalFrames))
    }
} catch {
    FileHandle.standardError.write("gfp: \(error)\n".data(using: .utf8)!)
    exit(1)
}
