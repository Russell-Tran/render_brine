// Step 8: the carbonic acid journey, CO₂ + H₂O → H₂CO₃ → HCO₃⁻ + H₃O⁺,
// ray-traced on the GPU as one seamlessly looping GIF.
//
//   .build/journey             render the full GIF
//   .build/journey --still T   render one PNG frame at T seconds (a quick preview)

import CoreGraphics
import Foundation
import ImageIO
import Metal
import UniformTypeIdentifiers
import simd

let layout = FrameLayout(width: 1280, viewHeight: 800, captionHeight: 160)
let fps = 20.0
let timeline = Timeline()
let poses = buildPoses()

/// A gentle wobble so the molecules look alive. It repeats exactly once per
/// cycle (whole-number frequencies), and each atom's phase comes from its slot
/// in its group, so the next group moves exactly like this one did.
func wobble(_ state: MoleculeState, time: Double) -> MoleculeState {
    var s = state
    let w = 2 * Double.pi / timeline.total
    for i in 0..<s.atoms.count {
        let phase = Double(i % Slot.count) * 1.7
        let dx = Float(sin(time * w * 9 + phase)) * 0.025
        let dy = Float(cos(time * w * 7 + phase * 1.3)) * 0.025
        s.atoms[i].position += SIMD3<Float>(dx, dy, 0)
    }
    return s
}

/// The camera swings gently from side to side, once per cycle.
func camera(at t: Double) -> Camera {
    let yaw = Float(16 * sin(2 * Double.pi * t / timeline.total))
    return Camera.orbit(target: SIMD3(-0.4, -0.6, 0), distance: 16, yaw: yaw, pitch: 8, fov: 30)
}

func renderFrame(_ renderer: MoleculeRenderer, _ frame: MTLBuffer, at t: Double) throws -> Double {
    let exact = journeyState(poses, timeline, at: t)
    let cam = camera(at: t)
    let seconds = try renderer.render(wobble(exact, time: t), camera: cam, into: frame, width: layout.width,
                                      viewHeight: layout.viewHeight)
    drawOverlay(state: exact, timeline: timeline, at: t, camera: cam, into: frame, layout: layout)
    return seconds
}

func savePNG(_ frame: MTLBuffer, to path: String) {
    let data = Data(bytes: frame.contents(), count: layout.width * layout.height * 4)
    guard let provider = CGDataProvider(data: data as CFData),
          let image = CGImage(width: layout.width, height: layout.height, bitsPerComponent: 8, bitsPerPixel: 32,
                              bytesPerRow: layout.width * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                              bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                              provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent),
          let dest = CGImageDestinationCreateWithURL(URL(fileURLWithPath: path) as CFURL,
                                                     UTType.png.identifier as CFString, 1, nil)
    else { return }
    CGImageDestinationAddImage(dest, image, nil)
    CGImageDestinationFinalize(dest)
}

do {
    let device = try findDevice()
    let renderer = try MoleculeRenderer(device: device)
    guard let frame = device.makeBuffer(length: layout.width * layout.height * 4, options: .storageModeShared) else {
        throw RenderError.gpu("could not allocate the frame")
    }
    try? FileManager.default.createDirectory(atPath: "renders", withIntermediateDirectories: true)
    let args = CommandLine.arguments
    if let i = args.firstIndex(of: "--still"), i + 1 < args.count, let t = Double(args[i + 1]) {
        _ = try renderFrame(renderer, frame, at: t)
        let path = String(format: "renders/still_%.2f.png", t)
        savePNG(frame, to: path)
        print("Saved \(path)")
        exit(0)
    }

    let frameCount = Int((timeline.total * fps).rounded())
    let gif = try GIFWriter(url: URL(fileURLWithPath: "renders/journey.gif"), frameCount: frameCount, delay: 1 / fps)
    var gpuSeconds = 0.0
    for f in 0..<frameCount {
        gpuSeconds += try renderFrame(renderer, frame, at: Double(f) / fps)
        gif.add(frame, layout: layout)
    }
    try gif.finish()

    let bytes = (try? FileManager.default.attributesOfItem(atPath: "renders/journey.gif")[.size] as? Int) ?? 0
    print("GPU: \(device.name)")
    print(String(format: "Rendered %d frames at %d × %d, 4 rays per pixel (%.1f s loop at %.0f fps)",
                 frameCount, layout.width, layout.height, timeline.total, fps))
    print(String(format: "GPU time: %.1f ms total, %.2f ms per frame", gpuSeconds * 1000,
                 gpuSeconds * 1000 / Double(frameCount)))
    print("Saved renders/journey.gif (\(bytes / 1024) KB)")
} catch {
    FileHandle.standardError.write("journey: \(error)\n".data(using: .utf8)!)
    exit(1)
}
