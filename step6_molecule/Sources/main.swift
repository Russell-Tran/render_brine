// Step 6, up close: one carbonic acid molecule splitting into H⁺ + HCO₃⁻,
// ray-traced on the GPU and saved as an animated GIF.

import Foundation
import simd

let layout = FrameLayout(width: 640, viewHeight: 400, captionHeight: 80)
let fps = 20.0
let timeline = Timeline()

/// A gentle thermal wobble so the molecule looks alive. Each atom moves a few
/// hundredths of an ångström on its own rhythm.
func wobble(_ state: MoleculeState, time: Double) -> MoleculeState {
    var s = state
    for i in 0..<s.atoms.count {
        let phase = Double(i) * 1.7
        let dx = Float(sin(time * 9.0 + phase)) * 0.025
        let dy = Float(cos(time * 7.0 + phase * 1.3)) * 0.025
        s.atoms[i].position += SIMD3<Float>(dx, dy, 0)
    }
    return s
}

do {
    let device = try findDevice()
    let renderer = try MoleculeRenderer(device: device)
    guard let frame = device.makeBuffer(length: layout.width * layout.height * 4, options: .storageModeShared) else {
        throw RenderError.gpu("could not allocate the frame")
    }
    let frameCount = Int((timeline.total * fps).rounded())
    let gif = try GIFWriter(url: URL(fileURLWithPath: "renders/molecule.gif"), frameCount: frameCount,
                            delay: 1 / fps)
    var gpuSeconds = 0.0

    for f in 0..<frameCount {
        let t = Double(f) / fps
        let progress = timeline.progress(at: t)
        let exact = moleculeState(progress: progress, protonDistance: timeline.protonDistance(at: t))
        let shown = wobble(exact, time: t)
        // Swing the camera gently from side to side, so the molecule reads as 3D.
        let yaw = Float(28 * sin(2 * Double.pi * t / timeline.total))
        let camera = Camera.orbit(target: SIMD3(0, -0.1, 0), distance: 11, yaw: yaw, pitch: 12, fov: 30)
        gpuSeconds += try renderer.render(shown, camera: camera, into: frame, width: layout.width,
                                          viewHeight: layout.viewHeight)
        drawOverlay(state: exact, camera: camera, stage: timeline.stage(at: t), progress: progress,
                    into: frame, layout: layout)
        gif.add(frame, layout: layout)
    }
    try gif.finish()

    let start = moleculeState(progress: 0, protonDistance: carbonicAcidGeometry.hydrogenBond)
    let end = moleculeState(progress: 1, protonDistance: 8)
    let bytes = (try? FileManager.default.attributesOfItem(atPath: "renders/molecule.gif")[.size] as? Int) ?? 0
    print("GPU: \(device.name)")
    print("Rendered \(frameCount) frames at \(layout.width) × \(layout.height), 4 rays per pixel")
    print("Carbonic acid: \(bondReadout(start))")
    print("Bicarbonate:   \(bondReadout(end))")
    print(String(format: "GPU time: %.1f ms total, %.2f ms per frame", gpuSeconds * 1000,
                 gpuSeconds * 1000 / Double(frameCount)))
    print("Saved renders/molecule.gif (\(bytes / 1024) KB)")
} catch {
    FileHandle.standardError.write("molecule: \(error)\n".data(using: .utf8)!)
    exit(1)
}
