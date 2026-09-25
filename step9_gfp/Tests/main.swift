// Tests for step 9. The structure is checked against the crystal file it came
// from; the space-filling and the ambient occlusion are checked for the
// properties that make them worth having; and the renderer and GIF writer are
// checked as in step 8.

import Foundation
import Metal
import simd

let gfpURL = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent().deletingLastPathComponent()
    .appendingPathComponent("Resources/gfp.json")
let pdbURL = gfpURL.deletingLastPathComponent().appendingPathComponent("1EMA.pdb")
let protein = try loadProtein(from: gfpURL)

/// The crystal file itself, so the tests answer to the source rather than to
/// the JSON that Tools/build_gfp.py produced from it.
let pdbLines = try String(contentsOf: pdbURL, encoding: .utf8).split(separator: "\n").map(String.init)

func near(_ a: Float, _ b: Float, within e: Float) -> Bool { abs(a - b) <= e }

/// Typical bond lengths (Å), Allen et al., J. Chem. Soc. Perkin Trans. 2
/// (1987), widened for a 1.9 Å structure.
func bondRange(_ a: String, _ b: String) -> ClosedRange<Float> {
    let pair = [a, b].sorted().joined()
    switch pair {
    case "CC": return 1.20...1.70
    case "CN", "CN".reversed().map(String.init).joined(): return 1.20...1.60
    case "CO": return 1.15...1.55
    case "NO": return 1.20...1.55
    case "CS", "SC": return 1.70...1.90
    case "CSE", "SEC": return 1.85...2.05
    case "NN": return 1.20...1.55
    case "OO": return 1.30...1.55
    default: return 1.10...2.10
    }
}

section("the structure, against 1EMA.pdb")
test("every non-water atom in the crystal file is here, and nothing else") {
    var fromFile = 0
    for line in pdbLines where line.hasPrefix("ATOM  ") || line.hasPrefix("HETATM") {
        let start = line.index(line.startIndex, offsetBy: 17)
        let end = line.index(line.startIndex, offsetBy: 20)
        if String(line[start..<end]).trimmingCharacters(in: .whitespaces) == "HOH" { continue }
        fromFile += 1
    }
    expectEqual(protein.atoms.count, fromFile)
    expectEqual(protein.atoms.count, 1771)
}
test("226 residues, numbered 2 to 229") {
    let seqs = Set(protein.atoms.map { $0.seq })
    expectEqual(seqs.count, 226)
    expectEqual(seqs.min(), 2)
    expectEqual(seqs.max(), 229)
    expectEqual(protein.residueCount, 226)
}
test("residues 65 and 67 are absent: they fused into the chromophore") {
    let seqs = Set(protein.atoms.map { $0.seq })
    expect(!seqs.contains(65), "65 should not exist on its own")
    expect(!seqs.contains(67), "67 should not exist on its own")
    expect(seqs.contains(66), "66, the chromophore, should")
}
test("the chromophore is one residue, CRO 66, of 22 atoms") {
    let cro = protein.atoms.enumerated().filter { $0.element.res == "CRO" }
    expectEqual(cro.count, 22)
    expect(cro.allSatisfy { $0.element.seq == 66 })
    expectEqual(Set(protein.chromophoreAtoms), Set(cro.map { $0.offset }))
    expect(protein.atoms.filter { $0.isChromophore }.count == 22)
}
test("four selenomethionines, as the file's MODRES records say") {
    let fromFile = pdbLines.filter { $0.hasPrefix("MODRES") && $0.contains("MSE") }.count
    expectEqual(fromFile, 4)
    expectEqual(protein.selenomethionines.count, 4)
    expectEqual(protein.selenomethionines, [78, 88, 153, 218])
    expectEqual(protein.atoms.filter { $0.element == "SE" }.count, 4)
}
test("the barrel has the eleven strands the file annotates") {
    expectEqual(pdbLines.filter { $0.hasPrefix("SHEET") }.count, 11)
    expectEqual(protein.strandCount, 11)
}
test("every bond has a believable length for its elements") {
    var worst: Float = 0, worstAt = ""
    for bond in protein.bonds {
        let a = protein.atoms[bond.a], b = protein.atoms[bond.b]
        let d = simd_distance(a.position, b.position)
        let range = bondRange(a.element, b.element)
        if !range.contains(d) {
            let off = min(abs(d - range.lowerBound), abs(d - range.upperBound))
            if off > worst { worst = off; worstAt = "\(a.res)\(a.seq).\(a.name)–\(b.res)\(b.seq).\(b.name) \(d) Å" }
        }
    }
    expect(worst == 0, "worst bond out of range by \(worst) Å: \(worstAt)")
}

section("the chromophore sits inside the barrel")
test("it is within 3 Å of the molecule's centre") {
    // A barrel exists to hold its chromophore in the middle, away from water.
    expect(simd_length(protein.chromophoreCenter) < 3.0,
           "chromophore is \(simd_length(protein.chromophoreCenter)) Å off centre")
}
test("the barrel wall really is a cylinder about it") {
    // The strands' distance from the axis barely varies: that is what makes it
    // a barrel rather than a bundle.
    expect(protein.wallRadiusSpread < 2.0, "radius varies by \(protein.wallRadiusSpread) Å")
    expect(protein.wallRadius > 10 && protein.wallRadius < 13, "wall radius \(protein.wallRadius) Å")
}

section("space filling")
test("radii are Bondi's van der Waals values, not covalent ones") {
    expectEqual(vdwRadius("C"), 1.70)
    expectEqual(vdwRadius("N"), 1.55)
    expectEqual(vdwRadius("O"), 1.52)
    expectEqual(vdwRadius("S"), 1.80)
    expectEqual(vdwRadius("SE"), 1.90)
    // Covalent radii are about half these; using them by mistake would give a
    // ball-and-stick model with no sticks, full of holes.
    expect(vdwRadius("C") > 1.5, "carbon at \(vdwRadius("C")) Å is too small to be van der Waals")
}
test("every atom in the file has a radius") {
    for atom in protein.atoms {
        expect(vdwRadius(atom.element) > 0.5, "no radius for \(atom.element)")
    }
}
test("bonded neighbours overlap — that is what makes it a solid") {
    var overlapping = 0
    for bond in protein.bonds {
        let a = protein.atoms[bond.a], b = protein.atoms[bond.b]
        let d = simd_distance(a.position, b.position)
        if d < vdwRadius(a.element) + vdwRadius(b.element) { overlapping += 1 }
    }
    expectEqual(overlapping, protein.bonds.count)
}
test("the colour spectrum runs from the N end to the C end") {
    let first = spectrumColor(0), last = spectrumColor(1)
    expect(first.z > first.x, "the N end should be blue, got \(first)")
    expect(last.x > last.z, "the C end should be warm, got \(last)")
    // And it is continuous, so no residue jumps colour against its neighbour.
    var biggest: Float = 0
    var previous = spectrumColor(0)
    for i in 1...200 {
        let c = spectrumColor(Float(i) / 200)
        biggest = max(biggest, simd_length(c - previous))
        previous = c
    }
    expect(biggest < 0.05, "biggest colour step \(biggest)")
}
test("the chromophore is the only saturated colour in the frame") {
    func saturation(_ c: SIMD3<Float>) -> Float { c.max() - c.min() }
    let chromo = saturation(chromophoreColor)
    for i in 0...100 {
        let s = saturation(spectrumColor(Float(i) / 100))
        expect(s < chromo, "a backbone colour is as saturated as the chromophore")
    }
}

section("the cutaway")
let cameraDistanceUnderTest: Float = 118
let closedCamera = Camera.orbit(target: .zero, distance: cameraDistanceUnderTest, yaw: 0, pitch: 6, fov: 30)
test("closed, nothing is removed") {
    let cut = cutFactors(protein, opening: 0, camera: closedCamera)
    expect(cut.allSatisfy { $0 == 1 })
}
test("open, it removes a window and not the whole front") {
    let cut = cutFactors(protein, opening: 1, camera: closedCamera)
    let gone = cut.filter { $0 < 0.5 }.count
    let fraction = Double(gone) / Double(cut.count)
    // A half-space cut would take about half the molecule. A window takes far
    // less, which is the point: the barrel must still read as a barrel.
    expect(fraction > 0.04, "only \(gone) atoms removed, the window may not be opening")
    expect(fraction < 0.30, "\(Int(fraction * 100))% removed, that is a half-space cut not a window")
}
test("the chromophore itself is never cut") {
    for opening in [Float(0), 0.3, 0.7, 1] {
        let cut = cutFactors(protein, opening: opening, camera: closedCamera)
        for i in protein.chromophoreAtoms {
            expectEqual(cut[i], 1)
        }
    }
}
test("when open, nothing opaque stands between the camera and the chromophore") {
    // Trace the actual ray and see what it meets first.
    let cut = cutFactors(protein, opening: 1, camera: closedCamera)
    let target = protein.chromophoreCenter
    let origin = closedCamera.origin
    let dir = simd_normalize(target - origin)
    let reach = simd_length(target - origin)
    var blocked: String? = nil
    for (i, atom) in protein.atoms.enumerated() where !atom.isChromophore {
        let r = vdwRadius(atom.element) * cut[i]
        guard r > 0.01 else { continue }
        let oc = origin - atom.position
        let b: Float = simd_dot(oc, dir)
        let h: Float = b * b - (simd_dot(oc, oc) - r * r)
        guard h >= 0 else { continue }
        let t: Float = -b - h.squareRoot()
        if t > 1e-3 && t < reach { blocked = "\(atom.res)\(atom.seq).\(atom.name)"; break }
    }
    expect(blocked == nil, "the chromophore is hidden behind \(blocked ?? "")")
}
test("when closed, the chromophore is hidden") {
    // The barrel is supposed to wrap it completely; if this fails the cutaway
    // is not actually revealing anything.
    let cut = cutFactors(protein, opening: 0, camera: closedCamera)
    let target = protein.chromophoreCenter
    let origin = closedCamera.origin
    let dir = simd_normalize(target - origin)
    let reach = simd_length(target - origin)
    var blocked = false
    for (i, atom) in protein.atoms.enumerated() where !atom.isChromophore {
        let r = vdwRadius(atom.element) * cut[i]
        let oc = origin - atom.position
        let b: Float = simd_dot(oc, dir)
        let h: Float = b * b - (simd_dot(oc, oc) - r * r)
        guard h >= 0 else { continue }
        let t: Float = -b - h.squareRoot()
        if t > 1e-3 && t < reach { blocked = true; break }
    }
    expect(blocked, "the closed barrel should hide its chromophore")
}
test("the window follows the camera round") {
    // Whichever way we look, the cut opens on the side facing us.
    for yaw in [Float(0), 90, 180, 270] {
        let camera = Camera.orbit(target: .zero, distance: cameraDistanceUnderTest, yaw: yaw, pitch: 6, fov: 30)
        let cut = cutFactors(protein, opening: 1, camera: camera)
        let removed = protein.atoms.enumerated().filter { cut[$0.offset] < 0.5 }
        expect(!removed.isEmpty, "nothing removed at yaw \(yaw)")
        // Everything removed should be on the camera's side of the chromophore.
        let forward = camera.forward
        let plane: Float = simd_dot(protein.chromophoreCenter, forward)
        for r in removed {
            expect(simd_dot(r.element.position, forward) < plane + 2,
                   "at yaw \(yaw), \(r.element.name) was cut from behind the chromophore")
        }
    }
}

section("the loop")
let totalFrames = 250
test("a full turn brings the camera back exactly where it started") {
    let a = Camera.orbit(target: .zero, distance: cameraDistanceUnderTest, yaw: 0, pitch: 6, fov: 30)
    let b = Camera.orbit(target: .zero, distance: cameraDistanceUnderTest, yaw: 360, pitch: 6, fov: 30)
    expect(simd_distance(a.origin, b.origin) < 1e-3, "\(a.origin) vs \(b.origin)")
}
test("the lights come back too, so the last frame joins the first") {
    let a = rotateAboutY(keyLightWorld, degrees: 0)
    let b = rotateAboutY(keyLightWorld, degrees: -360)
    expect(simd_distance(a, b) < 1e-5)
}
test("the cutaway is shut at both ends of the loop") {
    expectEqual(opening(at: 0), 0)
    expectEqual(opening(at: 0.999), 0)
    // and it does open in the middle
    expect(opening(at: 0.5) > 0.99, "it should be fully open half way round")
}
test("the cutaway never jumps: it eases in and out") {
    var biggest: Float = 0
    var previous = opening(at: 0)
    for f in 1...totalFrames {
        let o = opening(at: Double(f) / Double(totalFrames))
        biggest = max(biggest, abs(o - previous))
        previous = o
    }
    expect(biggest < 0.06, "biggest jump between frames was \(biggest)")
}
test("every atom stays inside the frame, all the way round") {
    // Each atom is a sphere, so its edge has to clear the frame, not just its
    // centre. How many pixels an ångström covers follows from the camera.
    let viewH: Float = 600, viewW: Float = 960
    let pxPerAngstrom: Float = viewH / (2 * cameraDistanceUnderTest * tan(radians(15)))
    var worst = ""
    for f in stride(from: 0, to: totalFrames, by: 5) {
        let yaw = Float(360.0 * Double(f) / Double(totalFrames))
        let camera = Camera.orbit(target: .zero, distance: cameraDistanceUnderTest, yaw: yaw, pitch: 6, fov: 30)
        for atom in protein.atoms {
            let p = camera.project(atom.position, width: Int(viewW), height: Int(viewH))
            let margin: Float = vdwRadius(atom.element) * pxPerAngstrom
            if p.x < margin || p.x > viewW - margin || p.y < margin || p.y > viewH - margin {
                if worst.isEmpty {
                    worst = "\(atom.res)\(atom.seq).\(atom.name) at \(p) with \(margin) px of radius, yaw \(yaw)"
                }
            }
        }
    }
    expect(worst.isEmpty, worst)
}

section("the renderer and ambient occlusion")
let device = try findDevice()
let renderer = try MoleculeRenderer(device: device)
let testLayout = FrameLayout(width: 320, viewHeight: 200, captionHeight: 40)
let buffer = device.makeBuffer(length: testLayout.width * testLayout.height * 4, options: .storageModeShared)!
func pixel(_ x: Int, _ y: Int) -> SIMD3<Int> {
    let p = buffer.contents().assumingMemoryBound(to: UInt8.self) + (y * testLayout.width + x) * 4
    return SIMD3(Int(p[0]), Int(p[1]), Int(p[2]))
}
func grey(_ x: Int, _ y: Int) -> Double {
    let c = pixel(x, y)
    return Double(c.x + c.y + c.z) / 3
}
/// Renders the whole protein at one angle and returns the frame.
@discardableResult
func renderAt(yaw: Float, ao: AOSettings, opening: Float = 0) throws -> Double {
    let camera = Camera.orbit(target: .zero, distance: cameraDistanceUnderTest, yaw: yaw, pitch: 6, fov: 30)
    let cut = cutFactors(protein, opening: opening, camera: camera)
    let (s, c) = sceneGeometry(protein, cut: cut, showChromophore: opening > 0.01)
    return try renderer.render(spheres: s, cylinders: c, camera: camera, into: buffer,
                               width: testLayout.width, viewHeight: testLayout.viewHeight,
                               ao: ao, lightYaw: yaw)
}

test("an empty scene is step 2's gradient, light blue above to navy below") {
    let camera = Camera.orbit(target: .zero, distance: cameraDistanceUnderTest, yaw: 0, pitch: 0, fov: 30)
    try renderer.render(spheres: [], cylinders: [], camera: camera, into: buffer,
                        width: testLayout.width, viewHeight: testLayout.viewHeight)
    let top = pixel(160, 0), bottom = pixel(160, testLayout.viewHeight - 1)
    expect(abs(top.x - 140) <= 3 && abs(top.y - 200) <= 3 && abs(top.z - 235) <= 3, "top \(top)")
    expect(abs(bottom.x - 8) <= 3 && abs(bottom.y - 40) <= 3 && abs(bottom.z - 90) <= 3, "bottom \(bottom)")
}
test("the scene is far past setBytes' 4 KB limit and still renders") {
    let (s, c) = sceneGeometry(protein, cut: [Float](repeating: 1, count: protein.atoms.count),
                               showChromophore: false)
    let bytes = s.count * MemoryLayout<GPUSphere>.stride + c.count * MemoryLayout<GPUCylinder>.stride
    expect(bytes > 4096 * 8, "only \(bytes) bytes")
    try renderAt(yaw: 0, ao: AOSettings())
    // Something was actually drawn in the middle of the frame.
    let background = pixel(4, testLayout.viewHeight / 2)
    var drawn = 0
    for x in 120..<200 where pixel(x, testLayout.viewHeight / 2) != background { drawn += 1 }
    expect(drawn > 40, "only \(drawn) pixels of molecule across the middle")
}
test("with occlusion off, the picture is flatter than with it on") {
    // Measured as the spread of brightness over the molecule: occlusion adds
    // dark crevices, so it widens the range.
    func spread(_ ao: AOSettings) throws -> Double {
        try renderAt(yaw: 0, ao: ao)
        var values: [Double] = []
        for y in stride(from: 20, to: testLayout.viewHeight - 20, by: 3) {
            for x in stride(from: 110, to: 210, by: 3) { values.append(grey(x, y)) }
        }
        let mean = values.reduce(0, +) / Double(values.count)
        return (values.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Double(values.count)).squareRoot()
    }
    let off = try spread(AOSettings.off)
    let on = try spread(AOSettings())
    expect(on > off * 1.15, "occlusion barely changed the picture: \(off) → \(on)")
}
test("occlusion is a property of the molecule, not of the image") {
    // Why this test, and not an obvious one about consecutive frames:
    //
    // Occlusion here is a pure function of object-space geometry, and the
    // molecule never moves — the camera orbits it instead. So a given point on
    // the surface is shaded identically in every frame BY CONSTRUCTION, and no
    // amount of turning the camera can make the loop crawl. Trying to catch a
    // crawl by comparing frames turns out to measure nothing: after even a
    // couple of degrees a different atom is frontmost at a given pixel, so
    // consecutive frames differ by about a tenth of their pixels no matter
    // what the occlusion does.
    //
    // The bug that WOULD break it is sampling in screen space — hashing the
    // pixel coordinate to pick probe directions, which is the usual way this
    // is written and the usual reason it crawls. That is what this catches:
    // render the same molecule from the same camera at two image sizes and the
    // shading of a given surface point must not change. Anything keyed to
    // pixels moves; anything keyed to the molecule does not.
    let probe = AOSettings(probes: 12, distance: 8, strength: 1, only: true, contrast: 1.7)
    let camera = Camera.orbit(target: .zero, distance: cameraDistanceUnderTest, yaw: 45, pitch: 6, fov: 30)
    let cut = [Float](repeating: 1, count: protein.atoms.count)
    let (spheres, cylinders) = sceneGeometry(protein, cut: cut, showChromophore: false)

    func readings(width: Int, height: Int) throws -> [Int: Double] {
        let buf = device.makeBuffer(length: width * height * 4, options: .storageModeShared)!
        try renderer.render(spheres: spheres, cylinders: cylinders, camera: camera, into: buf,
                            width: width, viewHeight: height, ao: probe, lightYaw: 45)
        let p = buf.contents().assumingMemoryBound(to: UInt8.self)
        var out: [Int: Double] = [:]
        for (i, atom) in protein.atoms.enumerated() {
            let q = camera.project(atom.position, width: width, height: height)
            let x = Int(q.x), y = Int(q.y)
            guard x > 1, x < width - 1, y > 1, y < height - 1 else { continue }
            let o = (y * width + x) * 4
            let g = Double(Int(p[o]) + Int(p[o + 1]) + Int(p[o + 2])) / 3
            guard g > 6 else { continue }
            out[i] = g
        }
        return out
    }
    let small = try readings(width: 320, height: 200)
    let large = try readings(width: 640, height: 400)
    var differences: [Double] = []
    for (key, a) in small {
        if let b = large[key] { differences.append(abs(a - b)) }
    }
    expect(differences.count > 300, "only \(differences.count) points seen at both sizes")
    differences.sort()
    let median = differences[differences.count / 2]
    // Clean builds measure about 8 (two image grids never sample a curved
    // surface at quite the same points); screen-space sampling measures 15.
    expect(median < 11.5, "the same surface points shaded \(median) grey levels apart at two image sizes")
}
test("a buried point is darker than an exposed one") {
    let probe = AOSettings(probes: 16, distance: 8, strength: 1, only: true, contrast: 1.7)
    try renderAt(yaw: 0, ao: probe)
    // The silhouette's outermost atoms are exposed; the middle of the mass,
    // where atoms crowd on all sides, is not.
    var edge: [Double] = [], middle: [Double] = []
    let camera = Camera.orbit(target: .zero, distance: cameraDistanceUnderTest, yaw: 0, pitch: 6, fov: 30)
    for atom in protein.atoms {
        let p = camera.project(atom.position, width: testLayout.width, height: testLayout.viewHeight)
        let x = Int(p.x), y = Int(p.y)
        guard x > 2, x < testLayout.width - 2, y > 2, y < testLayout.viewHeight - 2 else { continue }
        let neighbours = protein.atoms.filter { simd_distance($0.position, atom.position) < 8 }.count
        if neighbours > 90 { middle.append(grey(x, y)) }
        if neighbours < 35 { edge.append(grey(x, y)) }
    }
    expect(edge.count > 10 && middle.count > 10, "not enough samples: \(edge.count) / \(middle.count)")
    let edgeMean = edge.reduce(0, +) / Double(max(edge.count, 1))
    let middleMean = middle.reduce(0, +) / Double(max(middle.count, 1))
    expect(middleMean < edgeMean - 8,
           "crowded atoms averaged \(Int(middleMean)), exposed ones \(Int(edgeMean))")
}
test("with occlusion off, every surface gets the same ambient term") {
    // Nothing should vary except the two lights, so an occlusion-only render
    // with the probes turned off must be uniformly white where it hits.
    var ao = AOSettings.off
    ao.only = true
    try renderAt(yaw: 0, ao: ao)
    let camera = Camera.orbit(target: .zero, distance: cameraDistanceUnderTest, yaw: 0, pitch: 6, fov: 30)
    var seen = 0
    for atom in protein.atoms.prefix(400) {
        let p = camera.project(atom.position, width: testLayout.width, height: testLayout.viewHeight)
        let x = Int(p.x), y = Int(p.y)
        guard x > 2, x < testLayout.width - 2, y > 2, y < testLayout.viewHeight - 2 else { continue }
        if grey(x, y) > 250 { seen += 1 }
    }
    expect(seen > 100, "only \(seen) points came back fully unoccluded")
}

section("the GIF")
test("the writer makes a looping GIF that decodes back to what went in") {
    let palette: [RGB] = (0..<32).map { RGB(UInt8($0 * 8), UInt8(255 - $0 * 8), 128) }
    let url = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("step9-test.gif")
    let w = 40, h = 24
    let writer = GIFWriter(url: url, width: w, height: h, palette: palette, delayCentiseconds: 8)
    var frames: [[UInt8]] = []
    for f in 0..<6 {
        var frame = [UInt8](repeating: 0, count: w * h)
        for i in 0..<(w * h) { frame[i] = UInt8((i / 3 + f * 5) % 31) }
        frames.append(frame)
        writer.add(frame)
    }
    try writer.finish()
    expectEqual(writer.frameCount, 6)
    let data = try Data(contentsOf: url)
    expect(data.count > 40, "the file is suspiciously small")
    expect(Array(data.prefix(6)) == Array("GIF89a".utf8), "not a GIF89a header")
    expect(data.range(of: "NETSCAPE2.0".data(using: .ascii)!) != nil, "no loop-forever extension")
    try? FileManager.default.removeItem(at: url)
}
test("palette colours map to their own index") {
    let palette: [RGB] = [RGB(10, 20, 30), RGB(200, 30, 40), RGB(50, 200, 60)]
    let q = try Quantizer(device: device, palette: palette, pixels: 3)
    let buf = device.makeBuffer(length: 3 * 4, options: .storageModeShared)!
    let p = buf.contents().assumingMemoryBound(to: UInt8.self)
    for (i, c) in palette.enumerated() {
        p[i * 4] = c.x; p[i * 4 + 1] = c.y; p[i * 4 + 2] = c.z; p[i * 4 + 3] = 255
    }
    expectEqual(try q.indices(of: buf), [0, 1, 2])
}
test("the chromophore's green survives into the palette") {
    // Median cut gives colours to whatever covers the most pixels, and the
    // chromophore is small, so its green has to be reserved. Feed the chooser
    // nothing but grey and check the green is there anyway.
    var greys: [RGB] = []
    for i in 0..<600 {
        let v = UInt8(i % 256)
        greys.append(RGB(v, v, v))
    }
    let palette = gfpPalette(samples: greys, count: 64)
    let target = chromophoreColor * 255
    let closest = palette.map { c -> Float in
        let d = SIMD3<Float>(Float(c.x), Float(c.y), Float(c.z)) - target
        return simd_length(d)
    }.min() ?? 1e9
    expect(closest < 30, "nearest palette colour to the chromophore's green was \(closest) away")
}

finish()
