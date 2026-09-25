// Step 10: bacterial transformation — how a plasmid gets into E. coli.
//
//   .build/transformation bench      time brute force against the grid
//   .build/transformation approach   render renders/approach.gif
//   .build/transformation routes     render renders/routes.gif
//   .build/transformation            both GIFs
//
// Environment overrides, used by the benchmark and by short test renders:
//   T_WIDTH   frame width in pixels (default 960)
//   T_FRAMES  frames to render (default: the full loop)
//   T_PROBES  ambient-occlusion probes per hit (default 12)

import CoreGraphics
import Foundation
import Metal
import simd

let env = ProcessInfo.processInfo.environment
let width = Int(env["T_WIDTH"] ?? "") ?? 960
let layout = FrameLayout(width: width, viewHeight: width * 3 / 4, captionHeight: width / 8)
let probes = Int(env["T_PROBES"] ?? "") ?? 12
let delay = 8                                   // hundredths of a second: 12.5 fps
let sceneURL = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent().deletingLastPathComponent()
    .appendingPathComponent("Resources/scene.json")

let ao = AOSettings(probes: probes, distance: 14, strength: 1, contrast: 1.5)

/// The camera looks at the envelope edge-on, square to it, and never moves.
/// A still camera plus a subject that only translates is what keeps the
/// occlusion from crawling (see Render.swift).
func approachCamera(_ scene: Scene) -> Camera {
    // Framed at about 160 nm across, which is the widest that still leaves the
    // 4.7 nm bilayer legible: at 1.7 A a pixel a lipid headgroup is about 5 px.
    // The camera sits a little above the membrane so both its outer surface and
    // its cross-section are visible at once.
    let center = SIMD3<Float>(0, 330, 0)
    return Camera(origin: center + SIMD3(0, 190, 1950), target: center, fov: 40)
}

func approachCaption(_ scene: Scene, t: Float, reach: Float) -> (Caption, Evidence, String) {
    let lowest = plasmidOffset(scene, t: t, reach: reach).y - reach
    let gapNm = (lowest - scene.layers.omOuterFace) / 10
    let caption = Caption(
        title: "A plasmid reaches the cell",
        subtitle: "pGLO, supercoiled · the E. coli outer membrane, in cross-section",
        facts: String(format: "5,371 bp · 1.83 µm of DNA folded to %.0f nm · σ = −0.06 · Ca²⁺ shown hydrated, 4.1 Å",
                      scene.plasmid.extent / 10),
        aside: gapNm > 0.6 ? String(format: "gap %.0f nm", gapNm) : "contact")
    return (caption, .measured, "envelope dimensions from cryo-ET and AFM")
}

func makeFrameBuffer(_ device: MTLDevice) throws -> MTLBuffer {
    guard let b = device.makeBuffer(length: layout.width * layout.height * 4, options: .storageModeShared) else {
        throw RenderError.gpu("could not allocate the frame")
    }
    return b
}

// MARK: - benchmark

func bench(_ scene: Scene, _ renderer: SceneRenderer, _ buffer: MTLBuffer) throws {
    let base = gpuShapes(scene)
    let camera = approachCamera(scene)
    print("scene: \(base.count) spheres")

    print("\nbuilding grids")
    var results: [(String, Double, Double)] = []
    for density in [Float(0.5), 1.0, 2.0] {
        let t0 = Date()
        try renderer.buildGrid(base, density: density)
        let build = Date().timeIntervalSince(t0)
        let g = renderer.grid!
        var ms = 0.0
        for _ in 0..<3 {
            ms += try renderer.render(shapes: base, camera: camera, into: buffer,
                                      width: layout.width, viewHeight: layout.viewHeight,
                                      useGrid: true, ao: ao) * 1000 / 3
        }
        print(String(format: "  density %.1f: %d × %d × %d boxes, %.0f per used box, build %.1f s, %.0f ms/frame",
                     density, g.dims.x, g.dims.y, g.dims.z, g.averageOccupancy, build, ms))
        results.append((String(format: "%.1f", density), ms, g.averageOccupancy))
    }

    // Brute force, at a much smaller frame — at full size it would take minutes.
    let small = FrameLayout(width: 160, viewHeight: 120, captionHeight: 20)
    let smallBuffer = try makeFrameBuffer(renderer.device)
    var bruteMs = 0.0
    bruteMs = try renderer.render(shapes: base, camera: camera, into: smallBuffer,
                                  width: small.width, viewHeight: small.viewHeight,
                                  useGrid: false, ao: ao) * 1000
    let scale = Double(layout.width * layout.viewHeight) / Double(small.width * small.viewHeight)
    print(String(format: "\nbrute force: %.0f ms at %d × %d → about %.0f s at %d × %d",
                 bruteMs, small.width, small.viewHeight, bruteMs * scale / 1000,
                 layout.width, layout.viewHeight))
    if let best = results.min(by: { $0.1 < $1.1 }) {
        print(String(format: "grid is about %.0f× faster", bruteMs * scale / best.1))
    }

    print("\nocclusion cost at density 1")
    try renderer.buildGrid(base, density: 1)
    for p in [0, 6, 12, 24] {
        let s = AOSettings(probes: p, distance: 14, strength: p == 0 ? 0 : 1, contrast: 1.5)
        var ms = 0.0
        for _ in 0..<3 {
            ms += try renderer.render(shapes: base, camera: camera, into: buffer,
                                      width: layout.width, viewHeight: layout.viewHeight,
                                      useGrid: true, ao: s) * 1000 / 3
        }
        print(String(format: "  %2d probes: %6.0f ms/frame", p, ms))
    }
}

// MARK: - GIF 1

func renderApproach(_ scene: Scene, _ renderer: SceneRenderer, _ buffer: MTLBuffer) throws {
    let seconds = 20.0
    let full = Int((seconds * 100 / Double(delay)).rounded())
    let frameCount = Int(env["T_FRAMES"] ?? "") ?? full
    let base = gpuShapes(scene)
    let camera = approachCamera(scene)
    let dnaIdx = scene.indices(of: "plasmid")
    let ionIdx = scene.indices(of: "calcium")
    let onDNA = ionsOnDNA(scene, dnaIndices: dnaIdx, ionIndices: ionIdx)
    let reach = plasmidReach(scene, dnaIndices: dnaIdx)

    // The grid records which box each shape sits in, so it is only valid for
    // the positions it was built from. The plasmid MOVES, so the grid is
    // rebuilt every frame rather than once. That sounds expensive and is not:
    // building over 407,000 spheres is a fraction of the frame's own cost, and
    // the alternative — registering each moving shape in every box it will ever
    // pass through — would smear the plasmid across a third of the grid and
    // slow every ray that went near it.
    func frameAt(_ f: Int) throws -> Double {
        // The plasmid comes in over the first 70% and then rests against the
        // membrane: a conveyor, as since step 8, never a rewind.
        let u = Float(f) / Float(frameCount)
        let t = smoothstep01(min(u / 0.7, 1))
        let shapes = approachShapes(scene, base: base, dnaIndices: dnaIdx,
                                    ionIndices: ionIdx, ionAttachedToDNA: onDNA, t: t, reach: reach)
        try renderer.buildGrid(shapes, density: 2)
        let gpu = try renderer.render(shapes: shapes, camera: camera, into: buffer,
                                      width: layout.width, viewHeight: layout.viewHeight,
                                      useGrid: true, ao: ao)
        let (caption, evidence, note) = approachCaption(scene, t: t, reach: reach)
        drawCaption(caption, into: buffer, layout: layout)
        drawEvidence(evidence, note: note, into: buffer, layout: layout)
        return gpu
    }

    var samples: [RGB] = []
    for f in stride(from: 0, to: frameCount, by: max(frameCount / 4, 1)) {
        _ = try frameAt(f)
        samples += samplePixels(buffer, pixels: layout.width * layout.height, step: 7)
    }
    let palette = medianCutPalette(samples, count: 200)
    let quantizer = try Quantizer(device: renderer.device, palette: palette,
                                  pixels: layout.width * layout.height)

    let path = "renders/approach.gif"
    let gif = GIFWriter(url: URL(fileURLWithPath: path), width: layout.width, height: layout.height,
                        palette: palette, delayCentiseconds: delay)
    var gpuSeconds = 0.0
    let started = Date()
    for f in 0..<frameCount {
        gpuSeconds += try frameAt(f)
        gif.add(try quantizer.indices(of: buffer))
        if f % 25 == 0 { print("  frame \(f)/\(frameCount)") }
    }
    try gif.finish()
    let kb = ((try? FileManager.default.attributesOfItem(atPath: path)[.size] as? Int) ?? 0) / 1024
    print(String(format: "approach: %d frames, %d × %d, %.0f ms GPU/frame, %.0f s total → %@ (%d KB)",
                 frameCount, layout.width, layout.height, gpuSeconds * 1000 / Double(frameCount),
                 Date().timeIntervalSince(started), path, kb))
}

// MARK: - GIF 2: three ways in

func renderRoutes(_ scene: Scene, _ renderer: SceneRenderer, _ buffer: MTLBuffer) throws {
    let seconds = 16.0
    let full = Int((seconds * 100 / Double(delay)).rounded())
    let frameCount = Int(env["T_FRAMES"] ?? "") ?? full
    let base = gpuShapes(scene)
    let dnaIdx = scene.indices(of: "plasmid")
    let ionIdx = scene.indices(of: "calcium")
    let omIdx = scene.indices(of: "outer_membrane")
    let reach = plasmidReach(scene, dnaIndices: dnaIdx)

    let pdb = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        .deletingLastPathComponent().appendingPathComponent("Resources/8DFK.pdb")
    let comEA = try Protein(pdb: pdb, source: "PDB 8DFK, B. subtilis ComEA")
    let protein = comEA.upright(scale: 1.0)
    print("ComEA: \(comEA.beads.count) alpha carbons")

    // Each panel is a third of the frame, rendered on its own and copied in.
    let panelW = layout.width / 3
    let panelLayout = FrameLayout(width: panelW, viewHeight: layout.viewHeight, captionHeight: 0)
    guard let panelBuffer = renderer.device.makeBuffer(length: panelW * layout.viewHeight * 4,
                                                       options: .storageModeShared) else {
        throw RenderError.gpu("could not allocate a panel")
    }
    // Close on the contact point, where all the action is.
    // Level with the membrane, so a way through it is seen as a gap in the
    // cross-section rather than hidden behind the near lipids. Each panel is
    // 320 x 720, a tall narrow window, which suits a membrane: the DNA is above
    // it and the periplasm below.
    // About 21 nm across the panel. Electroporation pores are 1–10 nm wide, so
    // a wider shot would leave the thing the panel is about a few pixels across.
    let camera = Camera(origin: SIMD3(0, scene.layers.omCenter + 30, 640),
                        target: SIMD3(0, scene.layers.omCenter + 10, 0), fov: 40)

    func frameAt(_ f: Int) throws -> Double {
        let u = Float(f) / Float(frameCount)
        // Open over the first 60%, hold, then shut again so the loop is seamless.
        let t = u < 0.6 ? u / 0.6 : max(0, 1 - (u - 0.85) / 0.15)
        var gpu = 0.0
        for route in Route.allCases {
            let shapes = routeShapes(scene, base: base, route: route, t: min(t, 1),
                                     omIndices: omIdx, dnaIndices: dnaIdx, ionIndices: ionIdx,
                                     protein: protein, reach: reach)
            try renderer.buildGrid(shapes, density: 2)
            gpu += try renderer.render(shapes: shapes, camera: camera, into: panelBuffer,
                                       width: panelW, viewHeight: layout.viewHeight,
                                       useGrid: true, ao: ao)
            blit(panelBuffer, into: buffer, at: route.rawValue * panelW,
                 panelWidth: panelW, height: layout.viewHeight, layout: layout)
        }
        let caption = Caption(
            title: "Three ways in, three levels of evidence",
            subtitle: "the same outer membrane in each · only the route differs",
            facts: "the classroom kit uses the left-hand one, and no one knows how it works",
            aside: "pGLO into E. coli")
        drawCaption(caption, into: buffer, layout: layout)
        for route in Route.allCases {
            let cx = CGFloat(route.rawValue * panelW + panelW / 2)
            drawPanelLabel(route.title, into: buffer, layout: layout, centerX: cx,
                           top: 12 * layout.scale, size: 13 * layout.scale)
            drawEvidence(route.evidence, note: route.note, into: buffer, layout: layout,
                         atTop: true, x: CGFloat(route.rawValue * panelW) + 12 * layout.scale,
                         top: 36 * layout.scale)
            if route.rawValue > 0 { drawDivider(into: buffer, layout: layout, x: CGFloat(route.rawValue * panelW)) }
        }
        return gpu
    }

    var samples: [RGB] = []
    for f in stride(from: 0, to: frameCount, by: max(frameCount / 4, 1)) {
        _ = try frameAt(f)
        samples += samplePixels(buffer, pixels: layout.width * layout.height, step: 7)
    }
    let palette = medianCutPalette(samples, count: 200)
    let quantizer = try Quantizer(device: renderer.device, palette: palette,
                                  pixels: layout.width * layout.height)
    let path = "renders/routes.gif"
    let gif = GIFWriter(url: URL(fileURLWithPath: path), width: layout.width, height: layout.height,
                        palette: palette, delayCentiseconds: delay)
    var gpuSeconds = 0.0
    let started = Date()
    for f in 0..<frameCount {
        gpuSeconds += try frameAt(f)
        gif.add(try quantizer.indices(of: buffer))
        if f % 25 == 0 { print("  frame \(f)/\(frameCount)") }
    }
    try gif.finish()
    let kb = ((try? FileManager.default.attributesOfItem(atPath: path)[.size] as? Int) ?? 0) / 1024
    print(String(format: "routes: %d frames, %d × %d, %.0f ms GPU/frame, %.0f s total → %@ (%d KB)",
                 frameCount, layout.width, layout.height, gpuSeconds * 1000 / Double(frameCount),
                 Date().timeIntervalSince(started), path, kb))
}

/// Copies a panel's pixels into the main frame at a horizontal offset.
func blit(_ panel: MTLBuffer, into frame: MTLBuffer, at x: Int,
          panelWidth: Int, height: Int, layout: FrameLayout) {
    let src = panel.contents().assumingMemoryBound(to: UInt32.self)
    let dst = frame.contents().assumingMemoryBound(to: UInt32.self)
    for row in 0..<height {
        let from = row * panelWidth
        let to = row * layout.width + x
        for c in 0..<panelWidth { dst[to + c] = src[from + c] }
    }
}

// MARK: - main

do {
    let scene = try loadScene(from: sceneURL)
    let device = try findDevice()
    let renderer = try SceneRenderer(device: device)
    let buffer = try makeFrameBuffer(device)
    print("GPU: \(device.name)")
    print(String(format: "scene: %d spheres, patch %.0f × %.0f nm, plasmid %.0f nm across",
                 scene.beads.count, scene.patchX / 10, scene.patchZ / 10, scene.plasmid.extent / 10))

    let mode = CommandLine.arguments.dropFirst().first ?? "all"
    switch mode {
    case "bench": try bench(scene, renderer, buffer)
    case "approach": try renderApproach(scene, renderer, buffer)
    case "routes": try renderRoutes(scene, renderer, buffer)
    case "all":
        try renderApproach(scene, renderer, buffer)
        try renderRoutes(scene, renderer, buffer)
    default:
        FileHandle.standardError.write("unknown mode \(mode)\n".data(using: .utf8)!)
        exit(2)
    }
} catch {
    FileHandle.standardError.write("transformation: \(error)\n".data(using: .utf8)!)
    exit(1)
}
