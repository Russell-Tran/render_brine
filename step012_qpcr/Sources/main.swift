// Step 12: FRET in a real-time PCR probe. Two hybridisation probes land on a
// DNA target a couple of nucleotides apart; the donor hands its energy to the
// acceptor, and how much it hands over is computed on every frame from the
// separation the geometry actually produces.

import Foundation
import simd

let layout = FrameLayout(width: 960, viewHeight: 600, captionHeight: 120)
let delay = 10                           // hundredths of a second: 10 fps
let paletteSize = 96
let seconds = 20.0
let cameraDistance: Float = 272
let sceneURL = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent().deletingLastPathComponent()
    .appendingPathComponent("Resources/scene.json")

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

let probes: Int = envInt("QPCR_PROBES", 12)
let aoOn: Bool = envFlag("QPCR_AO", true)
let frameLimit: Int = envInt("QPCR_FRAMES", 0)

func caption(for scene: Scene, state: DyeState, progress: Float) -> Caption {
    let bound = progress > 0.92
    let subtitle = bound
        ? "both probes bound · the donor's energy crosses to the acceptor without a photon"
        : "the second probe is still arriving · too far for transfer"
    let facts = String(
        format: "E = 1/(1+(r/R\u{2080})\u{2076}) · R\u{2080} = %.0f Å · separation %.1f nm · transfer %.0f%%",
        forsterRadius, state.separation / 10, state.efficiency * 100)
    return Caption(title: "Förster transfer, computed from the geometry",
                   subtitle: subtitle,
                   facts: facts,
                   aside: "LightCycler HybProbe chemistry")
}

do {
    let scene = try loadScene(from: sceneURL)
    let device = try findDevice()
    let renderer = try MoleculeRenderer(device: device)
    guard let buffer = device.makeBuffer(length: layout.width * layout.height * 4,
                                         options: .storageModeShared) else {
        throw RenderError.gpu("could not allocate the frame")
    }
    print("GPU: \(device.name)")
    print(String(format: "target %@ · %d atoms · bound separation %.2f nm",
                 scene.targetSequence, scene.atoms.count, scene.boundSeparation / 10))

    let ao = aoOn ? AOSettings(probes: probes, distance: 8.0, strength: 1.0, only: false, contrast: 1.7)
                  : AOSettings.off
    var frameCount = Int((seconds * 100 / Double(delay)).rounded())
    if frameLimit > 0 { frameCount = min(frameCount, frameLimit) }
    let pixels = layout.width * layout.height

    // The scene sits still and the camera orbits, so ambient occlusion is
    // stable frame to frame - the arrangement steps 9, 9a and 11 arrived at.
    // Here the arriving probe DOES move, so that guarantee is weaker than in
    // step 9: occlusion on the probe's own surface is recomputed as it comes
    // in. It does not crawl because the probe only translates, which is the
    // same reasoning step 10 used.
    func cameraFor(_ f: Int) -> (Camera, Float) {
        let t = Double(f) / Double(frameCount)
        let yaw = Float(14 * sin(2 * Double.pi * t))
        let cam = Camera.orbit(target: SIMD3(40, 8, 0),
                               distance: cameraDistance, yaw: yaw, pitch: 8, fov: 30)
        return (cam, yaw)
    }

    func renderFrame(_ f: Int) throws -> (Double, DyeState) {
        let t = Double(f) * Double(delay) / 100.0
        let progress = approachProgress(t: t, duration: seconds)
        let placed = place(scene, progress: progress)
        let state = DyeState(separation: placed.separation)
        let (spheres, cylinders) = sceneGeometry(scene, placed: placed, state: state)
        let (cam, yaw) = cameraFor(f)
        let gpu = try renderer.render(spheres: spheres, cylinders: cylinders, camera: cam,
                                      into: buffer, width: layout.width,
                                      viewHeight: layout.viewHeight, ao: ao, lightYaw: yaw)
        drawOverlay(caption(for: scene, state: state, progress: progress),
                    state: state, into: buffer, layout: layout)
        return (gpu, state)
    }

    // One palette for the whole loop, from frames spread across it.
    var samples: [RGB] = []
    let stride = max(frameCount / 5, 1)
    for f in Swift.stride(from: 0, to: frameCount, by: stride) {
        _ = try renderFrame(f)
        samples += samplePixels(buffer, pixels: pixels, step: 7)
    }
    let palette = medianCutPalette(samples, count: paletteSize)
    let quantizer = try Quantizer(device: device, palette: palette, pixels: pixels)

    let path = "renders/fret.gif"
    let gif = GIFWriter(url: URL(fileURLWithPath: path), width: layout.width,
                        height: layout.height, palette: palette, delayCentiseconds: delay)
    var gpuSeconds = 0.0
    let start = Date()
    for f in 0..<frameCount {
        let (gpu, _) = try renderFrame(f)
        gpuSeconds += gpu
        gif.add(try quantizer.indices(of: buffer))
    }
    try gif.finish()
    let kb = ((try? FileManager.default.attributesOfItem(atPath: path)[.size] as? Int) ?? 0) / 1024
    print(String(format: "fret: %d frames, %.1f s at %.0f fps, %d × %d, %.1f ms GPU/frame, %.0f s total → %@ (%d KB)",
                 frameCount, seconds, 100 / Double(delay), layout.width, layout.height,
                 gpuSeconds * 1000 / Double(frameCount), Date().timeIntervalSince(start), path, kb))
} catch {
    FileHandle.standardError.write("qpcr: \(error)\n".data(using: .utf8)!)
    exit(1)
}
