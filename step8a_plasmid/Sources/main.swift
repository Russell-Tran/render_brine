// Step 8a: pGLO, Bio-Rad's teaching plasmid, as two looping GIFs.
//
//   plasmid_ring.gif   the whole 5,371 bp circle, turning once
//   plasmid_dive.gif   a continuous zoom from the whole circle into the first
//                      atoms of the GFP gene, 120× closer
//
//   .build/plasmid            both
//   .build/plasmid ring|dive  one
//   .build/plasmid bench      brute force against the grid, and grid resolutions

import Foundation
import simd

let layout = FrameLayout(width: 960, viewHeight: 600, captionHeight: 120)
let delay = 10                    // hundredths of a second: 10 fps, as in step 8
let paletteSize = 96
let samples = 3                   // 3 × 3 rays per pixel: the ring is only ~3 px across
let gridDensity: Float = 2

let pglo = try loadPlasmid(from: URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent().deletingLastPathComponent()
    .appendingPathComponent("Resources/pglo.json"))

/// Where the GFP gene begins, and the stretch of plasmid given real atoms.
func feature(_ name: String) -> Feature {
    for f in pglo.features where f.name == name { return f }
    fatalError("pGLO has no feature called \(name)")
}
let gfp: Feature = feature("GFP")
let diveTarget = pglo.index(ofBasePair: gfp.start)
let atomWindow = max(0, diveTarget - 300)..<min(pglo.length, diveTarget + 301)

let ringPitch: Float = 35

/// How far back the camera has to be for the whole circle to fit, found by
/// projecting it rather than guessed: the ring is tilted, so its near side is
/// much closer than its center and projects a good deal larger.
func fitDistance(pitch: Float, margin: Float) -> Float {
    let ring = (0..<pglo.length).map { pglo.frame(at: $0).origin }
    var lo: Float = 100, hi: Float = 100_000
    for _ in 0..<40 {
        let mid = (lo + hi) / 2
        let camera = Camera.orbit(target: .zero, distance: mid, yaw: 0, pitch: pitch, fov: 30)
        var fits = true
        for p in ring {
            let s = camera.project(p, width: layout.width, height: layout.viewHeight)
            if s.x < margin || s.x > Float(layout.width) - margin
                || s.y < margin || s.y > Float(layout.viewHeight) - margin { fits = false; break }
        }
        if fits { hi = mid } else { lo = mid }
    }
    return hi
}
let ringDistance = fitDistance(pitch: ringPitch, margin: 26)

let device = try findDevice()
let renderer = try SceneRenderer(device: device)
guard let frameBuffer = device.makeBuffer(length: layout.width * layout.height * 4,
                                          options: .storageModeShared) else {
    throw RenderError.gpu("could not allocate the frame")
}
let pixelCount = layout.width * layout.height

func microns(_ angstroms: Float) -> String { String(format: "%.2f µm", angstroms / 10_000) }

// MARK: - GIF 1, the whole circle

func renderRing() throws {
    var scene = buildScene(pglo, atomWindow: 0..<0)
    scene.setDetail(0)
    let built = Date()
    try renderer.buildGrid(scene.widest(), density: gridDensity)
    let grid = renderer.grid!
    print(String(format: "ring: %d shapes, grid %d × %d × %d, %.1f shapes per used box, built in %.2f s",
                 scene.shapes.count, grid.dims.x, grid.dims.y, grid.dims.z,
                 grid.averageOccupancy, -built.timeIntervalSinceNow))

    let caption = Caption(
        title: "pGLO, a plasmid that makes E. coli glow",
        subtitle: "5,371 base pairs · \(microns(pglo.circumference)) around · "
                + "\(pglo.helicalTurns) turns of double helix",
        facts: "green GFP · violet the arabinose switch · amber ampicillin resistance · blue the origin",
        aside: "the tube is drawn 3.4× life; true scale is 2 px")

    let frames = 240
    func draw(_ f: Int) throws -> Double {
        let turn = Float(f) / Float(frames) * 360
        let camera = Camera.orbit(target: .zero, distance: ringDistance, yaw: turn, pitch: ringPitch,
                                  fov: 30, lightSpin: radians(turn))
        let t = try renderer.render(shapes: scene.shapes, camera: camera, into: frameBuffer,
                                    width: layout.width, viewHeight: layout.viewHeight, samplesPerSide: samples)
        drawCaption(caption, into: frameBuffer, layout: layout)
        return t
    }
    try writeGIF(name: "plasmid_ring", frames: frames, draw: draw)
}

// MARK: - GIF 2, the dive

/// The camera at a point `s` (0 to 1) through the dive: an exponential zoom,
/// because each doubling of magnification should take the same time.
func diveCamera(_ s: Float) -> Camera {
    let near: Float = 60
    let distance = ringDistance * pow(near / ringDistance, s)
    // Swing the aim from the middle of the ring onto the target, and finish
    // doing it by the time the frame is narrower than the ring is wide —
    // after that the middle of the ring is empty space, and aiming anywhere
    // near it would show nothing at all. Tying this to distance rather than to
    // elapsed time matters, because the zoom is exponential and races through
    // the middle of its range.
    let ease = smoothstep01((ringDistance - distance) / (ringDistance - 4000))
    let target = pglo.frame(at: diveTarget).origin * ease
    let wide = SIMD3<Float>(0, sin(radians(ringPitch)), cos(radians(ringPitch)))
    let outward = simd_normalize(SIMD3(pglo.frame(at: diveTarget).origin.x, 0,
                                       pglo.frame(at: diveTarget).origin.z))
    let close = simd_normalize(outward * 0.86 + SIMD3<Float>(0, 0.5, 0))
    return Camera(target: target, direction: simd_normalize(wide + (close - wide) * ease),
                  distance: distance, fov: 30)
}

func diveCaption(_ distance: Float, level: Float) -> Caption {
    let scale = 0.8578 * distance / Float(layout.width)      // Å per pixel
    let across = String(format: "%.0f Å across the frame", 0.8578 * distance)
    if level < 0.9 {
        return Caption(title: "pGLO, a plasmid that makes E. coli glow",
                       subtitle: "5,371 base pairs · \(microns(pglo.circumference)) around",
                       facts: "diving into GFP, the gene for the glow — 717 bp, starting at base 1,342",
                       aside: across)
    }
    if level < 1.45 {
        return Caption(title: "Space-filling: every atom at its true size",
                       subtitle: "van der Waals radii (Bondi 1964) · the major and minor grooves are real surface",
                       facts: String(format: "an atom is %.1f px here, so spheres finally carry more than a tube does",
                                     3.4 / scale),
                       aside: across)
    }
    return Caption(title: "The start of GFP",
                   subtitle: "base 1,342 · \(pglo.bases(from: gfp.start, count: 12)) · "
                           + "the first codons of the protein, ATG GCT AGC",
                   facts: "ball-and-stick, with the Watson–Crick hydrogen bonds dashed: "
                        + "three for every G·C pair, two for every A·T",
                   aside: across)
}

func renderDive() throws {
    var scene = buildScene(pglo, atomWindow: atomWindow)
    let built = Date()
    try renderer.buildGrid(scene.widest(), density: gridDensity)
    let grid = renderer.grid!
    print(String(format: "dive: %d shapes (%d atoms and bonds over %d bp), grid %d × %d × %d, "
                       + "%.1f shapes per used box, built in %.2f s",
                 scene.shapes.count, scene.atomCount, atomWindow.count,
                 grid.dims.x, grid.dims.y, grid.dims.z, grid.averageOccupancy, -built.timeIntervalSinceNow))

    let frames = 200
    let zoomFrames = Int(Double(frames) * 0.86)      // then hold on the atoms
    func draw(_ f: Int) throws -> Double {
        let s = min(Float(f) / Float(zoomFrames), 1)
        let camera = diveCamera(smoothstep01(s))
        let distance = simd_distance(camera.origin, camera.target)
        let level = detailLevel(cameraDistance: distance)
        scene.setDetail(level)
        let t = try renderer.render(shapes: scene.shapes, camera: camera, into: frameBuffer,
                                    width: layout.width, viewHeight: layout.viewHeight, samplesPerSide: samples)
        drawCaption(diveCaption(distance, level: level), into: frameBuffer, layout: layout)
        return t
    }
    try writeGIF(name: "plasmid_dive", frames: frames, draw: draw)
}

// MARK: - Shared: palette, encode, report

func writeGIF(name: String, frames: Int, draw: (Int) throws -> Double) throws {
    // One palette for the whole loop, sampled from four frames spread through it.
    var swatches: [RGB] = []
    for f in stride(from: 0, to: frames, by: max(frames / 4, 1)) {
        _ = try draw(f)
        swatches += samplePixels(frameBuffer, pixels: pixelCount, step: 7)
    }
    let palette = plasmidPalette(samples: swatches, count: paletteSize)
    let quantizer = try Quantizer(device: device, palette: palette, pixels: pixelCount)

    let path = "renders/\(name).gif"
    let gif = GIFWriter(url: URL(fileURLWithPath: path), width: layout.width, height: layout.height,
                        palette: palette, delayCentiseconds: delay)
    var gpuSeconds = 0.0
    let start = Date()
    for f in 0..<frames {
        gpuSeconds += try draw(f)
        gif.add(try quantizer.indices(of: frameBuffer))
    }
    try gif.finish()
    let kb = ((try? FileManager.default.attributesOfItem(atPath: path)[.size] as? Int) ?? 0) / 1024
    let seconds: Double = Double(frames) * Double(delay) / 100
    let fps: Double = 100 / Double(delay)
    let msPerFrame: Double = gpuSeconds * 1000 / Double(frames)
    let megabytes: Double = Double(kb) / 1024
    let elapsed: Double = -start.timeIntervalSinceNow
    let line = "%@: %d frames, %.1f s at %.0f fps, %d × %d, %.0f ms GPU per frame, %.0f s total → %@ (%.1f MB)"
    print(String(format: line, name, frames, seconds, fps, layout.width, layout.height,
                 msPerFrame, elapsed, path, megabytes))
}

/// What the grid is worth, and at what resolution.
func benchmark() throws {
    var scene = buildScene(pglo, atomWindow: atomWindow)
    let camera = Camera.orbit(target: .zero, distance: ringDistance, yaw: 0, pitch: ringPitch, fov: 30)
    print("shapes: \(scene.shapes.count)")

    scene.setDetail(0)
    try renderer.buildGrid(scene.widest(), density: 1)
    var brute = 0.0
    for _ in 0..<3 {
        brute += try renderer.render(shapes: scene.shapes, camera: camera, into: frameBuffer,
                                     width: layout.width, viewHeight: layout.viewHeight,
                                     samplesPerSide: samples, useGrid: false)
    }
    print(String(format: "brute force:            %8.0f ms per frame", brute * 1000 / 3))

    for density in [Float(0.5), 1, 2, 4] {
        let t0 = Date()
        try renderer.buildGrid(scene.widest(), density: density)
        let build = -t0.timeIntervalSinceNow
        let g = renderer.grid!
        var gridTime = 0.0
        for _ in 0..<3 {
            gridTime += try renderer.render(shapes: scene.shapes, camera: camera, into: frameBuffer,
                                            width: layout.width, viewHeight: layout.viewHeight,
                                            samplesPerSide: samples, useGrid: true)
        }
        let ms = gridTime * 1000 / 3
        print(String(format: "grid ×%.1f  %4d × %3d × %4d  %7.1f per box  %8.1f ms  (%.0f× faster, %.2f s to build)",
                     density, g.dims.x, g.dims.y, g.dims.z, g.averageOccupancy, ms,
                     brute * 1000 / 3 / ms, build))
    }
}

let mode = CommandLine.arguments.dropFirst().first ?? "all"
print("GPU: \(device.name)")
print(String(format: "ring radius %.0f Å, %d turns, %.3f bp per turn; camera fits it at %.0f Å",
             pglo.ringRadius, pglo.helicalTurns, pglo.basePairsPerTurnDrawn, ringDistance))
switch mode {
case "ring": try renderRing()
case "dive": try renderDive()
case "bench": try benchmark()
default:
    try renderRing()
    try renderDive()
}
