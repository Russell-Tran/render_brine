// Step 7: one glucose molecule and six O₂ becoming six CO₂ and six H₂O,
// ray-traced on the GPU and saved as an animated GIF.

import Foundation
import simd

let layout = FrameLayout(width: 1280, viewHeight: 800, captionHeight: 160)
let fps = 20.0
let timeline = Timeline()

/// A gentle thermal wobble so the molecules look alive (a few hundredths of an ångström).
func wobble(_ state: MoleculeState, time: Double) -> MoleculeState {
    var s = state
    for i in 0..<s.atoms.count {
        let phase = Double(i) * 1.7
        let dx = Float(sin(time * 5.0 + phase)) * 0.025
        let dy = Float(cos(time * 3.7 + phase * 1.3)) * 0.025
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
    let reaction = buildReaction()
    let frameCount = Int((timeline.total * fps).rounded())
    let gif = try GIFWriter(url: URL(fileURLWithPath: "renders/respiration.gif"), frameCount: frameCount, delay: 1 / fps)
    var gpuSeconds = 0.0

    for f in 0..<frameCount {
        let t = Double(f) / fps
        let progress = timeline.progress(at: t)
        let exact = reactionState(reaction, progress: progress)
        let shown = wobble(exact, time: t)
        // Swing the camera gently from side to side, so the scene reads as 3D.
        let yaw = Float(18 * sin(2 * Double.pi * t / timeline.total))
        let camera = Camera.orbit(target: SIMD3(0, 0, 0), distance: 34, yaw: yaw, pitch: 14, fov: 30)
        gpuSeconds += try renderer.render(shown, camera: camera, into: frame, width: layout.width,
                                          viewHeight: layout.viewHeight)
        drawOverlay(reaction: reaction, state: exact, camera: camera, stage: timeline.stage(at: t),
                    progress: progress, into: frame, layout: layout)
        gif.add(frame, layout: layout)
    }
    try gif.finish()

    let bytes = (try? FileManager.default.attributesOfItem(atPath: "renders/respiration.gif")[.size] as? Int) ?? 0
    let before = oxidationStates(elements: reaction.elements, bonds: reaction.reactantBonds)
    let after = oxidationStates(elements: reaction.elements, bonds: reaction.productBonds)
    print("GPU: \(device.name)")
    print(String(format: "Rendered %d frames at %d × %d, 4 rays per pixel (%.1f s at %.0f fps)",
                 frameCount, layout.width, layout.height, timeline.total, fps))
    print("Carbon oxidation states before: " + reaction.carbons.map { signedLabel(before[$0]) }.joined(separator: " "))
    print("Carbon oxidation states after:  " + reaction.carbons.map { signedLabel(after[$0]) }.joined(separator: " "))
    print(String(format: "Electrons moved from carbon to oxygen: %.0f", electronsTransferred(reaction)))
    print(String(format: "GPU time: %.1f ms total, %.2f ms per frame", gpuSeconds * 1000,
                 gpuSeconds * 1000 / Double(frameCount)))
    print("Saved renders/respiration.gif (\(bytes / 1024) KB)")
} catch {
    FileHandle.standardError.write("respiration: \(error)\n".data(using: .utf8)!)
    exit(1)
}
