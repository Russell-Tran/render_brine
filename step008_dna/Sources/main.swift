// Step 8: the DNA double helix (PDB 1BNA) turning once about its own axis,
// as a seamless looping GIF.

import Foundation
import simd

// Size and speed, chosen by measurement (see GIF.swift). A turning 758-atom
// molecule changes ~7% of the pixels every frame, and those pixels cost about
// a byte each even with a shared palette and only-what-changed frames:
//   1280 × 960, 12.5 fps, 255 colors: 32 MB     960 × 720, 12.5 fps: 20 MB
//   960 × 720, 10 fps, 96 colors: 12.9 MB  ← this one
//   64 colors: 11.5 MB, but the orange phosphorus turns red, so no.
// 960 px is about the width GitHub shows a README image at anyway.
let layout = FrameLayout(width: 960, viewHeight: 600, captionHeight: 120)
let delay = 10                           // hundredths of a second per frame: 10 fps
let paletteSize = 96
let seconds = 24.0                       // one full turn, 15° a second
let dnaURL = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent().deletingLastPathComponent()
    .appendingPathComponent("Resources/dna.json")

let camera = dnaCamera()

func caption(for dna: DNA) -> Caption {
    let perTurn = 360 / mean(dna.twist)
    return Caption(
        title: "DNA, the B form",
        subtitle: "the Dickerson dodecamer · \(dna.sequence) · X-ray crystal structure (PDB 1BNA)",
        facts: String(format: "right-handed · %.1f base pairs per turn here (10.5 in solution) · %.1f Å per base pair · 20 Å wide",
                      perTurn, mean(dna.rise)),
        aside: "C₂₃₂H₂₇₂N₉₂O₁₄₀P₂₂²²⁻")
}

do {
    let dna = try loadDNA(from: dnaURL)
    let device = try findDevice()
    let renderer = try MoleculeRenderer(device: device)
    guard let buffer = device.makeBuffer(length: layout.width * layout.height * 4, options: .storageModeShared) else {
        throw RenderError.gpu("could not allocate the frame")
    }
    print("GPU: \(device.name)")
    let frameCount = Int((seconds * 100 / Double(delay)).rounded())
    let pixels = layout.width * layout.height
    let text = caption(for: dna)

    func renderFrame(_ f: Int) throws -> Double {
        let angle = Float(360.0 * Double(f) / Double(frameCount))
        let (spheres, cylinders) = sceneGeometry(dna, turn: angle, camera: camera)
        let t = try renderer.render(spheres: spheres, cylinders: cylinders, camera: camera, into: buffer,
                                    width: layout.width, viewHeight: layout.viewHeight)
        drawCaption(text, into: buffer, layout: layout)
        return t
    }

    // One palette for the whole loop, from four frames a quarter-turn apart.
    var samples: [RGB] = []
    for f in stride(from: 0, to: frameCount, by: frameCount / 4) {
        _ = try renderFrame(f)
        samples += samplePixels(buffer, pixels: pixels, step: 7)
    }
    let palette = dnaPalette(samples: samples, count: paletteSize)
    let quantizer = try Quantizer(device: device, palette: palette, pixels: pixels)

    let path = "renders/dna.gif"
    let gif = GIFWriter(url: URL(fileURLWithPath: path), width: layout.width, height: layout.height,
                        palette: palette, delayCentiseconds: delay)
    var gpuSeconds = 0.0
    let start = Date()
    for f in 0..<frameCount {
        gpuSeconds += try renderFrame(f)
        gif.add(try quantizer.indices(of: buffer))
    }
    try gif.finish()
    let kb = ((try? FileManager.default.attributesOfItem(atPath: path)[.size] as? Int) ?? 0) / 1024
    print(String(format: "dna: %d frames, %.1f s at %.1f fps, %d × %d, %.1f ms GPU per frame, %.0f s total → %@ (%d KB)",
                 frameCount, seconds, 100 / Double(delay), layout.width, layout.height,
                 gpuSeconds * 1000 / Double(frameCount), Date().timeIntervalSince(start), path, kb))
} catch {
    FileHandle.standardError.write("dna: \(error)\n".data(using: .utf8)!)
    exit(1)
}
