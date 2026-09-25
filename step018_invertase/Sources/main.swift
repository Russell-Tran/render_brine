// Step 18: sucrose split into glucose and fructose by yeast invertase, beside
// the polarimeter reading that gave invert sugar its name.
//
// Two panels, and they are not allowed to disagree. On the left, one active
// site of the octamer runs its catalytic cycle: sucrose binds, Asp23 attacks
// the fructose's anomeric carbon, Glu204 protonates the bridging oxygen, the
// glucose goes, a covalent fructosyl-enzyme forms, water breaks it, the
// fructose goes, the enzyme is back where it started. On the right, a
// polarimeter. The reading is computed from the molecules DRAWN in the cell —
// counted, put through the specific rotations, and drawn as the twist of the
// beam itself.
//
// Nothing here has shown a molecular event beside its bulk observable before.
// It is worth doing because that is the order it was found in: the polarimeter
// reading went negative in the 1830s; the aspartate was not named until 2013.

import CoreGraphics
import Foundation
import simd

let env: [String: String] = ProcessInfo.processInfo.environment

func envInt(_ name: String, _ fallback: Int) -> Int {
    guard let s: String = env[name] else { return fallback }
    guard let v: Int = Int(s) else { return fallback }
    return v
}
func envFloat(_ name: String, _ fallback: Float) -> Float {
    guard let s: String = env[name] else { return fallback }
    guard let v: Float = Float(s) else { return fallback }
    return v
}
func envFlag(_ name: String, _ fallback: Bool) -> Bool {
    guard let s: String = env[name] else { return fallback }
    return s != "0"
}

let panelWidth = envInt("INV_PANEL", 640)
let viewHeight = envInt("INV_HEIGHT", 620)
let layout = FrameLayout(width: panelWidth * 2, viewHeight: viewHeight, captionHeight: 120)
let delay = envInt("INV_DELAY", defaultDelayCentiseconds)     // hundredths of a second
let totalFrames = envInt("INV_FRAMES_TOTAL", defaultFrameCount)
let paletteSize = envInt("INV_PALETTE", defaultPaletteSize)
let moleculesInCell = envInt("INV_MOLECULES", defaultCellMolecules)
let probes = envInt("INV_PROBES", 10)
let aoOn = envFlag("INV_AO", true)
let samples = envInt("INV_SAMPLES", 2)

let sceneURL = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent().deletingLastPathComponent()
    .appendingPathComponent("Resources/invertase.json")

/// The sodium D line, turned into a screen colour by the CIE 1931 machinery
/// step 9a built. The lamp in the render is this colour because that is the
/// wavelength specific rotation is defined at — not because orange looked right.
let sodium = spectralColour(nanometres: Float(sodiumDLineNanometres))
let sodiumLinear = sodium.linear
let sodiumHex = hexString(nanometres: Float(sodiumDLineNanometres))

do {
    let scene = try loadScene(from: sceneURL)
    let device = try findDevice()
    let renderer = try SceneRenderer(device: device)
    let framePixels = layout.width * layout.height
    guard let frame = device.makeBuffer(length: framePixels * 4, options: .storageModeShared),
          let panel = device.makeBuffer(length: panelWidth * viewHeight * 4,
                                        options: .storageModeShared) else {
        throw RenderError.gpu("could not allocate the frame")
    }
    let ao = aoOn ? AOSettings(probes: probes, distance: 7.0, strength: 1.0,
                               contrast: envFloat("INV_AO_POW", 1.7)) : AOSettings.off

    let ends = rotationAtRest(molecules: moleculesInCell)
    let invertedMixture = Composition(sucrose: 0, glucose: moleculesInCell,
                                      fructose: moleculesInCell)
    let invertRotation = specificRotation(of: invertedMixture)
    let crossing = zeroCrossingFraction(molecules: moleculesInCell)

    print("GPU: \(device.name)")
    print("""
          \(scene.organism) invertase (\(scene.gene)), PDB \(scene.pdb) at \
          \(String(format: "%.2f", scene.resolution)) Å
            \(scene.atoms.count) heavy atoms in \(scene.chains.count) chains — \(scene.biologicalUnit)
            dimer \(scene.dimerChains.joined()) carries the site; \
          \(scene.dimerAtomIndices.count) atoms
            Asp\(scene.catalytic.nucleophilePaper) nucleophile, \
          Glu\(scene.catalytic.acidBasePaper) acid/base, \
          Asp\(scene.catalytic.stabiliserPaper) stabiliser \
          (crystal numbering Asp\(scene.catalytic.nucleophilePDB), \
          Glu\(scene.catalytic.acidBasePDB), Asp\(scene.catalytic.stabiliserPDB))
          """)
    print("""
          substrate pose: transferred from PDB \(scene.template.pdb) \
          (\(scene.template.organism), \(String(format: "%.2f", scene.template.resolution)) Å)
            \(String(format: "%.1f", scene.template.identityPercent))% identical over \
          \(scene.template.alignedResidues) residues; \(scene.template.siteResidues) active-site Cα \
          at \(String(format: "%.2f", scene.template.siteRMSD)) Å
            what fell out of it: nucleophile–C2 \
          \(String(format: "%.2f", scene.attackDistance)) Å, attack angle \
          \(String(format: "%.1f", scene.attackAngle))°, acid–O(glycosidic) \
          \(String(format: "%.2f", scene.acidDistance)) Å
          """)
    print("""
          the chemistry, derived:
            invert sugar [α]D = \(String(format: "%+.2f", invertRotation))° — \
          the mass-weighted mean of \(String(format: "%+.1f", glucose.specificRotation)) and \
          \(String(format: "%.1f", fructose.specificRotation)), never written down
            \(String(format: "%.3f", normalWeightGramsPer100mL)) g/100 mL in a \
          \(String(format: "%.3f", cellLengthDecimetres)) dm cell reads \
          \(String(format: "%+.2f", ends.start))° fresh, \(String(format: "%+.2f", ends.end))° inverted
            a gram of sucrose becomes \
          \(String(format: "%.4f", 2 * glucose.molarMass / sucrose.molarMass)) g of invert sugar, \
          which is why the end is not \(String(format: "%.3f", invertRotation / sucrose.specificRotation)) \
          of the start but \(String(format: "%.3f", ends.end / ends.start))
            the reading passes zero at \(String(format: "%.1f", crossing * 100))% converted, \
          not at half
            sodium D line \(String(format: "%.2f", sodiumDLineNanometres)) nm → \(sodiumHex)\
          \(sodium.clipped ? " (outside sRGB, clipped)" : "")
          """)
    // Padding is done in Swift rather than with printf's %-24s. Handing
    // String(format:) a `utf8String` pointer works right up until the NSString
    // it came from is released out from under it, which on this toolchain is
    // immediately, and the crash is a strlen off the end of freed memory.
    func pad(_ s: String, _ width: Int) -> String {
        s.count >= width ? s : s + String(repeating: " ", count: width - s.count)
    }
    print("\nconstants")
    let table = chemistryConstants(molecules: moleculesInCell) + structureConstants(scene)
    for c in table {
        let value = String(format: "%12.4f", c.value)
        print("  \(pad(c.name, 28))\(value)  \(pad(c.unit, 22))\(pad(c.evidence.rawValue, 9))\(c.source)")
    }
    print("")

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

    var lastReadout: PolarimeterReadout?
    var lastPrimCount = 0

    func renderFrame(_ f: Int) throws -> Double {
        let u = Double(f) / Double(totalFrames)
        let b = beat(at: u)
        var seconds = 0.0

        // Left: the molecule.
        let molecule = moleculeScene(scene, beat: b)
        try renderer.buildGrid(molecule)
        let camera = moleculeCamera(b, scene: scene)
        let lightYaw: Float = b.octamerYaw * Float(1 - smoothstep(b.zoom))
        seconds += try renderer.render(prims: molecule, camera: camera, into: panel,
                                       width: panelWidth, height: viewHeight,
                                       samplesPerSide: samples, ao: ao, backdrop: .sky,
                                       lightYaw: lightYaw)
        blit(0)

        // Right: the bench.
        let (bench, readout) = polarimeterScene(progress: b.progress, fill: b.fill,
                                                molecules: moleculesInCell,
                                                lightColour: sodiumLinear)
        try renderer.buildGrid(bench)
        seconds += try renderer.render(prims: bench, camera: polarimeterCamera(), into: panel,
                                       width: panelWidth, height: viewHeight,
                                       samplesPerSide: samples, ao: ao, backdrop: .bench)
        blit(1)
        lastReadout = readout
        lastPrimCount = molecule.count + bench.count

        drawDivider(at: panelWidth, into: frame, layout: layout)
        let labels = [
            PanelLabel(name: b.zoom > 0.5 ? "one dimer, one active site" : "invertase octamer",
                       detail: b.zoom > 0.5
                         // Not "a porthole onto the site": the GH32 site is at
                         // the bottom of the propeller's funnel and is already
                         // open to solvent. The cut widens a hole rather than
                         // making one, and the test measures how much.
                         ? "space-filling · the porthole widens the funnel over the site"
                         : "\(scene.chains.count) chains, D4 — a tetramer of dimers",
                       colour: cgColour(SIMD3(0.55, 0.78, 0.82)),
                       centreX: CGFloat(panelWidth / 2)),
            PanelLabel(name: "polarimeter",
                       detail: "sodium D line · \(String(format: "%.3f", cellLengthDecimetres)) dm cell",
                       colour: cgColour(sodiumLinear),
                       centreX: CGFloat(panelWidth + panelWidth / 2)),
        ]
        drawPanelLabels(labels, into: frame, layout: layout)
        drawStageLabel(stageName(b), colour: cgColour(SIMD3(0.92, 0.80, 0.55)),
                       centreX: CGFloat(panelWidth / 2), into: frame, layout: layout)
        drawDial(Dial(degrees: readout.rotationDegrees, start: ends.start, end: ends.end,
                      fractionConverted: readout.composition.fractionConverted,
                      zeroCrossing: crossing,
                      note: readout.fill < 0.999
                        ? String(format: "cell %.0f%% full", readout.fill * 100)
                        : String(format: "%.0f%% of the sucrose split",
                                 readout.composition.fractionConverted * 100),
                      centreX: CGFloat(panelWidth + panelWidth / 2),
                      lightColour: cgColour(sodiumLinear)),
                 into: frame, layout: layout)
        drawCaption(caption(scene, b, readout: readout, invertRotation: invertRotation),
                    into: frame, layout: layout)
        return seconds
    }

    let arguments = CommandLine.arguments
    let stillsOnly = arguments.contains("stills")
    let single: Int = envInt("INV_AT", -1)
    let frameLimit: Int = envInt("INV_FRAMES", 0)
    let frameCount = frameLimit > 0 ? min(frameLimit, totalFrames) : totalFrames

    if stillsOnly {
        try FileManager.default.createDirectory(atPath: "renders", withIntermediateDirectories: true)
        // One frame from each beat of the loop: the drain, the refill, the way
        // in, and then a turnover under way, the zero crossing, and the way out.
        for f in [3, 11, 22, 40, 58, 76, 90] {
            _ = try renderFrame(f)
            writePNG(frame, layout: layout, to: "renders/frame\(f).png")
            if let r = lastReadout {
                print(String(format: "frame %3d  α %+7.2f°  %3d split  %d prims",
                             f, r.rotationDegrees, r.splitGlyphs, lastPrimCount))
            }
        }
        print("wrote stills to renders/")
        exit(0)
    }

    // One palette for the whole loop, sampled across it so the close-up's
    // chemistry colours are represented as well as the wide shot's.
    var swatches: [RGB] = []
    for f in stride(from: 0, to: totalFrames, by: max(totalFrames / 8, 1)) {
        _ = try renderFrame(f)
        swatches += samplePixels(frame, pixels: framePixels, step: 9)
    }
    let palette = invertasePalette(samples: swatches, count: paletteSize)
    let quantizer = try Quantizer(device: device, palette: palette, pixels: framePixels)

    try FileManager.default.createDirectory(atPath: "renders", withIntermediateDirectories: true)
    let path = "renders/invertase.gif"
    let gif = GIFWriter(url: URL(fileURLWithPath: path), width: layout.width, height: layout.height,
                        palette: palette, delayCentiseconds: delay)
    var gpuSeconds = 0.0
    let start = Date()
    let frames = single >= 0 ? [single] : Array(0..<frameCount)
    // Steps 13 and 14 measured what the changed-pixels-only encoder actually
    // gets, rather than assuming it gets anything. Here the left panel's camera
    // moves every frame, so the answer should be bad; it is printed either way.
    var previous: [UInt8]?
    var changedTotal = 0.0
    var boxTotal = 0.0
    for f in frames {
        gpuSeconds += try renderFrame(f)
        let indices = try quantizer.indices(of: frame)
        if let old = previous {
            var changed = 0
            var minX = layout.width, maxX = -1, minY = layout.height, maxY = -1
            for y in 0..<layout.height {
                let row = y * layout.width
                for x in 0..<layout.width where indices[row + x] != old[row + x] {
                    changed += 1
                    if x < minX { minX = x }
                    if x > maxX { maxX = x }
                    if y < minY { minY = y }
                    if y > maxY { maxY = y }
                }
            }
            changedTotal += Double(changed) / Double(framePixels)
            if maxX >= minX {
                let boxWidth: Double = Double(maxX - minX + 1)
                let boxHeight: Double = Double(maxY - minY + 1)
                let box: Double = boxWidth * boxHeight
                boxTotal += box / Double(framePixels)
            }
        }
        previous = indices
        gif.add(indices)
    }
    try gif.finish()
    let bytes = ((try? FileManager.default.attributesOfItem(atPath: path)[.size] as? Int) ?? 0)
    let msPerFrame = gpuSeconds * 1000 / Double(frames.count)
    let wall = Date().timeIntervalSince(start)
    print(String(format: "invertase: %d of %d frames, %d × %d, %.0f ms GPU per frame, %.0f s wall → %@ (%.1f MB)",
                 frames.count, totalFrames, layout.width, layout.height, msPerFrame, wall, path,
                 Double(bytes) / 1024 / 1024))
    if frames.count > 1 {
        let n = Double(frames.count - 1)
        print(String(format: "  %.2f%% of pixels change per frame, but the box around them covers "
                     + "%.1f%% — the camera moves, so there is little for the encoder to hold on to",
                     changedTotal / n * 100, boxTotal / n * 100))
    }
    if frames.count < totalFrames {
        print(String(format: "  the full loop would be %.0f s of GPU time (%d frames)",
                     msPerFrame * Double(totalFrames) / 1000, totalFrames))
    }
} catch {
    FileHandle.standardError.write("invertase: \(error)\n".data(using: .utf8)!)
    exit(1)
}
