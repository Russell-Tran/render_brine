// Step 9: glycolysis as two ray-traced GIFs, "spend" (glucose → two
// glyceraldehyde-3-phosphates, 2 ATP spent) and "payoff" (→ two pyruvates,
// 4 ATP made, 2 NADH). The chemistry and keyframes come from
// Tools/build_timeline.py via Resources/timeline.json.

import Foundation
import simd

let layout = FrameLayout(width: 1280, viewHeight: 800, captionHeight: 160)
let fps = 15.0
let timelineURL = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent().deletingLastPathComponent()
    .appendingPathComponent("Resources/timeline.json")

do {
    let movies = try loadMovies(from: timelineURL)
    let device = try findDevice()
    let renderer = try MoleculeRenderer(device: device)
    guard let buffer = device.makeBuffer(length: layout.width * layout.height * 4, options: .storageModeShared) else {
        throw RenderError.gpu("could not allocate the frame")
    }
    print("GPU: \(device.name)")
    for movie in movies {
        let frameCount = Int((movie.duration * fps).rounded())
        let path = "renders/\(movie.name).gif"
        let gif = try GIFWriter(url: URL(fileURLWithPath: path), frameCount: frameCount, delay: 1 / fps)
        var gpuSeconds = 0.0
        for f in 0..<frameCount {
            let t = Double(f) / fps
            let frame = movie.frame(at: t)
            let cam = movie.camera(at: t)
            let (spheres, cylinders) = sceneGeometry(frame, camera: cam)
            gpuSeconds += try renderer.render(spheres: spheres, cylinders: cylinders, camera: cam, into: buffer,
                                              width: layout.width, viewHeight: layout.viewHeight)
            drawOverlay(frame, camera: cam, into: buffer, layout: layout)
            gif.add(buffer, layout: layout)
        }
        try gif.finish()
        let kb = ((try? FileManager.default.attributesOfItem(atPath: path)[.size] as? Int) ?? 0) / 1024
        print(String(format: "%@: %d frames, %.1f s at %.0f fps, %.2f ms GPU per frame → %@ (%d KB)",
                     movie.name, frameCount, movie.duration, fps, gpuSeconds * 1000 / Double(frameCount), path, kb))
    }
} catch {
    FileHandle.standardError.write("glycolysis: \(error)\n".data(using: .utf8)!)
    exit(1)
}
