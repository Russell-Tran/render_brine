// Step 8b: DNA space-filling, so the grooves read as the channels they are.
// Forty base pairs turning once about their own axis, as a seamless looping
// GIF — plus a still that puts the same twelve crystal base pairs side by side
// in both representations, which is the argument of the step in one image.

import CoreGraphics
import Foundation
import ImageIO
import Metal
import UniformTypeIdentifiers
import simd

let layout = FrameLayout(width: 960, viewHeight: 400, captionHeight: 120)
let delay = 10                     // hundredths of a second: 10 fps
let paletteSize = 96
let seconds = 20.0
let duplexURL = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent().deletingLastPathComponent()
    .appendingPathComponent("Resources/grooves.json")

/// Environment overrides, so a short run needs no recompile:
///   GR_FRAMES=8 make run     render only the first few frames
///   GR_AO=0 make run         turn ambient occlusion off
///   GR_AT=90 make run        render one frame and stop
///   GR_STILL=1 make run      write the comparison still and stop
// Spelled out in typed steps: the laptop's older Swift type checker is slower,
// and the chained `??` form sits near the -warn-long-expression-type-checking
// limit.
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

let probes: Int = envInt("GR_PROBES", 12)
let aoOn: Bool = envFlag("GR_AO", true)
let frameLimit: Int = envInt("GR_FRAMES", 0)
let aoDistance: Float = envFloat("GR_AO_DIST", 8.0)

func caption(for helix: Duplex) -> Caption {
    let major = helix.grooveWidth["major"] ?? 0
    let minor = helix.grooveWidth["minor"] ?? 0
    let facts = String(format: "major groove %.1f Å across (teal edge) · minor %.1f Å (rust) · "
                       + "backbone in sand · every atom at its van der Waals radius",
                       major, minor)
    return Caption(title: "The grooves are real",
                   subtitle: "40 base pairs of pGLO on an ideal B-DNA helix · base pairs measured, helix idealised",
                   facts: facts,
                   aside: "\(helix.atoms.filter { $0.element != "H" }.count) heavy atoms, space-filling")
}

/// Writes one buffer out as a JPEG, for the stills the results page uses.
func writeStill(_ buffer: MTLBuffer, to path: String, layout: FrameLayout) throws {
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
    guard let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.jpeg.identifier as CFString, 1, nil) else {
        throw RenderError.gpu("could not create \(path)")
    }
    CGImageDestinationAddImage(dest, image, [kCGImageDestinationLossyCompressionQuality as String: 0.9] as CFDictionary)
    guard CGImageDestinationFinalize(dest) else { throw RenderError.gpu("could not write \(path)") }
}

do {
    let (helix, crystal) = try loadDuplexes(from: duplexURL)
    let device = try findDevice()
    let renderer = try MoleculeRenderer(device: device)
    guard let buffer = device.makeBuffer(length: layout.width * layout.height * 4,
                                         options: .storageModeShared) else {
        throw RenderError.gpu("could not allocate the frame")
    }
    let ao = aoOn ? AOSettings(probes: probes, distance: aoDistance, strength: 1.0, only: false,
                               contrast: envFloat("GR_AO_POW", 1.7)) : AOSettings.off
    print("GPU: \(device.name)")
    print("helix: \(helix.atoms.count) atoms over \(helix.sequence.count) bp; "
          + "crystal: \(crystal.atoms.count) atoms")
    print("ambient occlusion \(aoOn ? "\(probes) probes, \(ao.distance) Å" : "off")")

    let totalFrames = Int((seconds * 100 / Double(delay)).rounded())
    let pixels = layout.width * layout.height
    let text = caption(for: helix)

    /// The duplex never moves. The camera circles its axis and the lights are
    /// unwound to match, so a given point on the surface is shaded the same in
    /// every frame and the occlusion cannot crawl.
    func renderFrame(_ f: Int) throws -> Double {
        let u = Double(f) / Double(totalFrames)
        let roll = Float(360.0 * u)
        let camera = Camera.orbitAboutX(target: .zero, distance: cameraDistance,
                                        roll: roll, lean: 5, fov: 30)
        let spheres = spaceFilling(helix)
        let t = try renderer.render(spheres: spheres, cylinders: [], camera: camera, into: buffer,
                                    width: layout.width, viewHeight: layout.viewHeight,
                                    ao: ao, lightRoll: roll)
        drawCaption(text, into: buffer, layout: layout)
        return t
    }

    // The comparison still: the same twelve crystal base pairs, once as balls
    // on sticks and once space-filling. Nothing changes but the representation.
    func renderComparison() throws {
        let camera = Camera.orbitAboutX(target: .zero, distance: 62, roll: 0, lean: 5, fov: 30)
        let stick = ballAndStick(crystal, camera: camera)
        _ = try renderer.render(spheres: stick.spheres, cylinders: stick.cylinders, camera: camera,
                                into: buffer, width: layout.width, viewHeight: layout.viewHeight,
                                ao: ao, lightRoll: 0)
        drawCaption(Caption(title: "Ball-and-stick",
                            subtitle: "the same twelve base pairs · PDB 1BNA, measured atoms",
                            facts: "the grooves are not visible as anything: only gaps between sticks",
                            aside: "steps 6–8's representation"),
                    into: buffer, layout: layout)
        try writeStill(buffer, to: "renders/compare_stick.jpg", layout: layout)

        let spheres = spaceFilling(crystal)
        _ = try renderer.render(spheres: spheres, cylinders: [], camera: camera, into: buffer,
                                width: layout.width, viewHeight: layout.viewHeight,
                                ao: ao, lightRoll: 0)
        let major = crystal.grooveWidth["major"] ?? 0
        let minor = crystal.grooveWidth["minor"] ?? 0
        drawCaption(Caption(title: "Space-filling",
                            subtitle: "the same twelve base pairs · the only change is the representation",
                            facts: String(format: "the grooves are channels: major %.1f Å across, minor %.1f Å",
                                          major, minor),
                            aside: "van der Waals radii, Bondi 1964"),
                    into: buffer, layout: layout)
        try writeStill(buffer, to: "renders/compare_space.jpg", layout: layout)
        print("wrote renders/compare_stick.jpg and renders/compare_space.jpg")
    }

    if envFlag("GR_STILL", false) {
        try renderComparison()
        exit(0)
    }

    let frameCount = frameLimit > 0 ? min(frameLimit, totalFrames) : totalFrames
    let single: Int = envInt("GR_AT", -1)

    // One palette for the whole loop, sampled across it.
    var samples: [RGB] = []
    for f in stride(from: 0, to: totalFrames, by: max(totalFrames / 6, 1)) {
        _ = try renderFrame(f)
        samples += samplePixels(buffer, pixels: pixels, step: 7)
    }
    let palette = groovePalette(samples: samples, count: paletteSize)
    let quantizer = try Quantizer(device: device, palette: palette, pixels: pixels)

    let path = "renders/grooves.gif"
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
    print(String(format: "grooves: %d of %d frames, %d × %d, %.0f ms GPU per frame, %.0f s wall → %@ (%d KB)",
                 frames.count, totalFrames, layout.width, layout.height, msPerFrame,
                 Date().timeIntervalSince(start), path, kb))
    if frames.count < totalFrames {
        print(String(format: "  full loop would be %.0f s of GPU time (%d frames)",
                     msPerFrame * Double(totalFrames) / 1000, totalFrames))
    }
    if single < 0 && frameCount == totalFrames { try renderComparison() }
} catch {
    FileHandle.standardError.write("grooves: \(error)\n".data(using: .utf8)!)
    exit(1)
}
