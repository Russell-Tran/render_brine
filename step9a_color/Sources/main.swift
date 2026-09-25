// Step 9a: GFP beside mCherry, space-filling with ambient occlusion, turning
// together, both portholes opening at the same moment to show the two
// chromophores — and the extra conjugation that makes one of them red.
//
// Each protein is rendered into its own half of the frame by its own camera,
// so each stays fixed in its own coordinates and step 9's steady occlusion
// holds for both. See Pair.swift for why.

import CoreGraphics
import Foundation
import simd

// Size, speed and palette follow the measurement step 8 made and step 9
// repeated: a turning molecule changes several percent of the pixels every
// frame, and those pixels cost about a byte each whatever the encoding. Two
// turning molecules change twice as many. The numbers here were chosen by
// rendering the loop at several settings and comparing; see the results page.
let envEarly: [String: String] = ProcessInfo.processInfo.environment
let panelWidth = Int(envEarly["COLOR_PANEL"] ?? "") ?? 430
let viewHeight = Int(envEarly["COLOR_HEIGHT"] ?? "") ?? 470
let layout = FrameLayout(width: panelWidth * 2, viewHeight: viewHeight, captionHeight: 124)
let delay = Int(envEarly["COLOR_DELAY"] ?? "") ?? 10        // hundredths of a second: 10 fps
let paletteSize = Int(envEarly["COLOR_PALETTE"] ?? "") ?? 96
let seconds = 20.0
let cameraDistance: Float = 134
// The camera looks a little above the molecule's middle, so each barrel sits
// low enough in its panel to leave headroom for the label above it.
let cameraTarget = SIMD3<Float>(0, 5.5, 0)
let pairURL = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent().deletingLastPathComponent()
    .appendingPathComponent("Resources/pair.json")

/// Environment overrides, so a short run needs no recompile:
///   COLOR_FRAMES=8 make run     render only the first few frames
///   COLOR_AO=0 make run         turn ambient occlusion off
///   COLOR_AT=90 make run        render one frame only
let env: [String: String] = envEarly

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

let probes: Int = envInt("COLOR_PROBES", 12)
let aoOn: Bool = envFlag("COLOR_AO", true)
let frameLimit: Int = envInt("COLOR_FRAMES", 0)
let aoDistance: Float = envFloat("COLOR_AO_DIST", 8.0)

func caption(for pair: ProteinPair, opening: Float) -> Caption {
    let g = pair.gfp, m = pair.mCherry
    let gap: Double = g.photonEnergyEV - m.photonEnergyEV
    if opening > 0.5 {
        let facts = String(format: "%@'s π system runs %d atoms; %@'s runs %d — the acylimine N1=CA1 carries it two further",
                           g.label, g.piAtoms.count, m.label, m.piAtoms.count)
        return Caption(title: "One bond further",
                       subtitle: String(format: "N1–CA1 measures %.2f Å in GFP (a single bond) and %.2f Å in mCherry (a double one)",
                                        g.acylimineLength, m.acylimineLength),
                       facts: facts,
                       aside: String(format: "%.3f eV apart", gap))
    }
    let facts = String(format: "%d and %d residues · the same barrel fold · colours computed from the emitted wavelength, not chosen",
                       g.residueCount, m.residueCount)
    return Caption(title: "Why one is green and one is red",
                   subtitle: String(format: "%@ emits at %.0f nm (%.3f eV) · %@ at %.0f nm (%.3f eV)",
                                    g.label, g.emission, g.photonEnergyEV,
                                    m.label, m.emission, m.photonEnergyEV),
                   facts: facts,
                   aside: "space-filling · PDB 1EMA, 2H5Q")
}

func cgColour(_ c: SIMD3<Float>) -> CGColor {
    return CGColor(srgbRed: CGFloat(gammaEncode(c.x)), green: CGFloat(gammaEncode(c.y)),
                   blue: CGFloat(gammaEncode(c.z)), alpha: 1)
}

do {
    let pair = try loadPair(from: pairURL)
    let device = try findDevice()
    let renderer = try MoleculeRenderer(device: device)
    let framePixels = layout.width * layout.height
    guard let frame = device.makeBuffer(length: framePixels * 4, options: .storageModeShared),
          let panel = device.makeBuffer(length: panelWidth * viewHeight * 4, options: .storageModeShared) else {
        throw RenderError.gpu("could not allocate the frame")
    }
    let ao = aoOn ? AOSettings(probes: probes, distance: aoDistance, strength: 1.0, only: false,
                               contrast: envFloat("COLOR_AO_POW", 1.7)) : AOSettings.off
    print("GPU: \(device.name)")
    for p in pair.both {
        let c = spectralRGB8(nanometres: p.emission)
        print(String(format: "%-11s %@  %d atoms · emits %.0f nm → computed %@ (%d,%d,%d)%@",
                     (p.label as NSString).utf8String!, p.pdb, p.atoms.count, p.emission,
                     hexString(nanometres: p.emission), c.r, c.g, c.b,
                     p.emissionClipped ? " · outside sRGB, clipped" : ""))
    }
    print("ambient occlusion \(aoOn ? "\(probes) probes, \(ao.distance) Å" : "off")")

    let totalFrames = Int((seconds * 100 / Double(delay)).rounded())

    /// Copies the panel buffer into one half of the frame, row by row.
    func blit(_ column: Int) {
        let src = panel.contents().assumingMemoryBound(to: UInt8.self)
        let dst = frame.contents().assumingMemoryBound(to: UInt8.self)
        let rowBytes = panelWidth * 4
        for y in 0..<viewHeight {
            let from = src + y * rowBytes
            let to = dst + (y * layout.width + column * panelWidth) * 4
            to.update(from: from, count: rowBytes)
        }
    }

    /// One frame: each protein rendered by its own camera into its own half.
    /// The molecules never move — the cameras orbit and the lights follow —
    /// which is what keeps the occlusion steady, exactly as in step 9.
    func renderFrame(_ f: Int) throws -> Double {
        let u = Double(f) / Double(totalFrames)
        let yaw = Float(360.0 * u)
        let open = opening(at: u)
        var seconds = 0.0
        for (column, p) in pair.both.enumerated() {
            let camera = Camera.orbit(target: cameraTarget, distance: cameraDistance, yaw: yaw, pitch: 6, fov: 30)
            let cut = cutFactors(p, opening: open, camera: camera)
            let (spheres, cylinders) = sceneGeometry(p, cut: cut, showChromophore: open > 0.01)
            seconds += try renderer.render(spheres: spheres, cylinders: cylinders, camera: camera,
                                           into: panel, width: panelWidth, viewHeight: viewHeight,
                                           ao: ao, lightYaw: yaw)
            blit(column)
        }
        let labels = pair.both.enumerated().map { column, p in
            PanelLabel(name: p.label,
                       detail: String(format: "emits %.0f nm", p.emission),
                       colour: cgColour(p.emissionColour),
                       centreX: CGFloat(column * panelWidth + panelWidth / 2))
        }
        drawPanelLabels(labels, into: frame, layout: layout)
        drawCaption(caption(for: pair, opening: open), into: frame, layout: layout)
        return seconds
    }

    let frameCount = frameLimit > 0 ? min(frameLimit, totalFrames) : totalFrames
    let single: Int = envInt("COLOR_AT", -1)

    // One palette for the whole loop, sampled across it so the open phase and
    // its two chromophore colours are represented as well as the closed one.
    var samples: [RGB] = []
    for f in stride(from: 0, to: totalFrames, by: max(totalFrames / 6, 1)) {
        _ = try renderFrame(f)
        samples += samplePixels(frame, pixels: framePixels, step: 7)
    }
    let palette = pairPalette(pair, samples: samples, count: paletteSize)
    let quantizer = try Quantizer(device: device, palette: palette, pixels: framePixels)

    let path = "renders/color.gif"
    let gif = GIFWriter(url: URL(fileURLWithPath: path), width: layout.width, height: layout.height,
                        palette: palette, delayCentiseconds: delay)
    var gpuSeconds = 0.0
    let start = Date()
    let frames = single >= 0 ? [single] : Array(0..<frameCount)
    for f in frames {
        gpuSeconds += try renderFrame(f)
        gif.add(try quantizer.indices(of: frame))
    }
    try gif.finish()
    let kb = ((try? FileManager.default.attributesOfItem(atPath: path)[.size] as? Int) ?? 0) / 1024
    let msPerFrame = gpuSeconds * 1000 / Double(frames.count)
    print(String(format: "color: %d of %d frames, %d × %d, %.0f ms GPU per frame, %.0f s wall → %@ (%d KB)",
                 frameCount, totalFrames, layout.width, layout.height, msPerFrame,
                 Date().timeIntervalSince(start), path, kb))
    if frameCount < totalFrames {
        print(String(format: "  full loop would be %.0f s of GPU time (%d frames)",
                     msPerFrame * Double(totalFrames) / 1000, totalFrames))
    }
} catch {
    FileHandle.standardError.write("color: \(error)\n".data(using: .utf8)!)
    exit(1)
}
