// Tests for step 8. Structure tests check the molecule is real B-DNA (the
// formula, every atom's bonds, Watson–Crick pairing, the helix); motion tests
// check the loop and framing; the rest check the renderer, the palette and
// the hand-written GIF encoder (decoded back with Apple's ImageIO).

import CoreGraphics
import Foundation
import ImageIO
import Metal
import simd

let dnaURL = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent().deletingLastPathComponent()
    .appendingPathComponent("Resources/dna.json")
let dna = try loadDNA(from: dnaURL)
let camera = dnaCamera()
let view = (width: 960, height: 600)
let frames = 240

func neighbors(_ i: Int) -> [(atom: Int, order: Int)] {
    dna.bonds.compactMap { b in b.a == i ? (b.b, b.order) : b.b == i ? (b.a, b.order) : nil }
}

/// Typical bond lengths (Å) by element pair, from Allen et al., J. Chem. Soc.
/// Perkin Trans. 2 (1987) S1, widened for a 1.9 Å crystal structure.
func bondRange(_ a: String, _ b: String) -> ClosedRange<Float> {
    switch [a, b].sorted().joined() {
    case "CC": return 1.30...1.60
    case "CN": return 1.25...1.55
    case "CO": return 1.15...1.50
    case "CH", "HN", "HO": return 0.93...1.13
    case "OP": return 1.40...1.70
    default: return 0...0
    }
}

section("structure: PDB 1BNA plus hydrogens")
test("758 atoms: C₂₃₂H₂₇₂N₉₂O₁₄₀P₂₂, charge −22 (one per phosphate)") {
    var counts: [String: Int] = [:]
    for a in dna.atoms { counts[a.element, default: 0] += 1 }
    expectEqual(dna.atoms.count, 758)
    expectEqual(counts, ["C": 232, "H": 272, "N": 92, "O": 140, "P": 22])
    expectEqual(dna.atoms.reduce(0) { $0 + $1.charge }, -22)
}
test("the sequence is the Dickerson dodecamer, and it pairs with itself") {
    expectEqual(dna.sequence, "CGCGAATTCGCG")
    let partner: [Character: Character] = ["A": "T", "T": "A", "G": "C", "C": "G"]
    expectEqual(String(dna.sequence.reversed().map { partner[$0]! }), dna.sequence)
}
test("every atom has its proper bonds: C 4, N 3, O 2 (O⁻ 1), P 5, H 1") {
    let valence = ["C": 4, "N": 3, "O": 2, "P": 5, "H": 1]
    for (i, a) in dna.atoms.enumerated() {
        let total = neighbors(i).reduce(0) { $0 + $1.order }
        expect(total == valence[a.element]! + a.charge, "\(a.label) has \(total)")
    }
}
test("every bond has a real length for its elements") {
    for b in dna.bonds {
        let ea = dna.atoms[b.a].element, eb = dna.atoms[b.b].element
        let d = simd_distance(dna.atoms[b.a].position, dna.atoms[b.b].position)
        expect(bondRange(ea, eb).contains(d), "\(dna.atoms[b.a].label)–\(dna.atoms[b.b].label) is \(d) Å")
    }
}

section("Watson–Crick pairing")
test("12 base pairs held by 32 hydrogen bonds: 3 per G–C, 2 per A–T") {
    var perPair: [String: Int] = [:]
    for h in dna.hbonds { perPair[h.pair, default: 0] += 1 }
    expectEqual(perPair.count, 12)
    expectEqual(dna.hbonds.count, 32)
    for (pair, n) in perPair {
        expectEqual(n, pair.contains("G") ? 3 : 2)
    }
}
test("donor–acceptor 2.6–3.3 Å, H…acceptor under 2.3 Å, and the H points at its partner") {
    for h in dna.hbonds {
        let d = dna.atoms[h.donor].position, hp = dna.atoms[h.hydrogen].position, a = dna.atoms[h.acceptor].position
        expect((2.6...3.3).contains(simd_distance(d, a)), "\(h.pair): D…A \(simd_distance(d, a))")
        expect(simd_distance(hp, a) < 2.3, "\(h.pair): H…A \(simd_distance(hp, a))")
        let angle = acos(simd_dot(simd_normalize(d - hp), simd_normalize(a - hp))) * 180 / .pi
        expect(angle > 140, "\(h.pair): D–H…A \(angle)°")
    }
}

section("the helix")
test("right-handed at every step") {
    expectEqual(dna.twist.count, 11)
    for (i, t) in dna.twist.enumerated() { expect(t > 0, "step \(i) twists \(t)°") }
}
test("twist 34–38° and rise 3.1–3.6 Å on average: about 10 base pairs per turn") {
    let twist = mean(dna.twist), rise = mean(dna.rise)
    expect((34...38).contains(twist), "twist \(twist)")
    expect((3.1...3.6).contains(rise), "rise \(rise)")
    expect((9.5...10.6).contains(360 / twist), "\(360 / twist) bp per turn")
}
test("the helix axis is x: base-pair centers stay near it") {
    let c1 = dna.atoms.enumerated().filter { $0.element.label.hasSuffix(".C1'") }.map { $0.element.position }
    for p in c1 { expect(simd_length(SIMD2(p.y, p.z)) < 9, "C1' is \(simd_length(SIMD2(p.y, p.z))) Å off the axis") }
}

section("motion and framing")
test("a full turn brings every atom back exactly: the loop is seamless") {
    for a in dna.atoms {
        expect(simd_distance(place(a.position, turn: 360), place(a.position, turn: 0)) < 1e-3)
    }
    // The frame after the last is 360°, the same as frame 0.
    expectEqual(360.0 * Double(frames) / Double(frames), 360.0)
}
test("turning keeps every distance: the molecule moves rigidly") {
    let a = dna.atoms[0].position, b = dna.atoms[400].position
    for angle in stride(from: Float(0), to: 360, by: 37) {
        expect(abs(simd_distance(place(a, turn: angle), place(b, turn: angle)) - simd_distance(a, b)) < 1e-3)
    }
}
test("on every frame, every atom stays inside the view with room to spare") {
    var worst = ""
    for f in 0..<frames {
        let angle = Float(360.0 * Double(f) / Double(frames))
        for a in dna.atoms {
            let p = camera.project(place(a.position, turn: angle), width: view.width, height: view.height)
            if p.x < 20 || p.x > Float(view.width - 20) || p.y < 20 || p.y > Float(view.height - 20), worst.isEmpty {
                worst = "frame \(f): \(a.label) at \(p)"
            }
        }
    }
    expect(worst.isEmpty, worst)
}

section("geometry for the GPU")
test("dashes are centered: the same gap at both ends") {
    let a = SIMD3<Float>(0, 0, 0), b = SIMD3<Float>(1.9, 0, 0)
    let d = dashes(from: a, to: b)
    expectEqual(d.count, 5)
    expect(abs(simd_distance(a, d.first!.0) - simd_distance(d.last!.1, b)) < 1e-5)
    expect(abs(simd_distance(d[0].0, d[0].1) - dashLength) < 1e-5)
}
test("one ball per atom; one stick per single bond, two per double, plus the dashes") {
    let (spheres, cylinders) = sceneGeometry(dna, turn: 0, camera: camera)
    expectEqual(spheres.count, 758)
    let singles = dna.bonds.filter { $0.order == 1 }.count, doubles = dna.bonds.filter { $0.order == 2 }.count
    let dashCount = cylinders.filter { $0.aRadius.w == hbondRadius }.count
    expectEqual(cylinders.count, singles + 2 * doubles + dashCount)
    expect(dashCount >= 32 * 3, "\(dashCount) dashes")
}
test("the scene is far bigger than setBytes' 4 KB, so it must go in buffers") {
    let (spheres, cylinders) = sceneGeometry(dna, turn: 0, camera: camera)
    expect(spheres.count * MemoryLayout<GPUSphere>.stride > 4096)
    expect(cylinders.count * MemoryLayout<GPUCylinder>.stride > 4096)
}

section("renderer")
let device = try findDevice()
let renderer = try MoleculeRenderer(device: device)
let small = FrameLayout(width: 480, viewHeight: 300, captionHeight: 60)
let buffer = device.makeBuffer(length: small.width * small.height * 4, options: .storageModeShared)!
func pixel(_ x: Int, _ y: Int) -> SIMD3<Int> {
    let p = buffer.contents().assumingMemoryBound(to: UInt8.self) + (y * small.width + x) * 4
    return SIMD3(Int(p[0]), Int(p[1]), Int(p[2]))
}
test("an empty scene is step 2's gradient, light blue at the top to navy at the bottom") {
    try renderer.render(spheres: [], cylinders: [], camera: camera, into: buffer, width: small.width,
                        viewHeight: small.viewHeight)
    let top = pixel(240, 0), bottom = pixel(240, small.viewHeight - 1)
    expect(abs(top.x - 140) <= 3 && abs(top.y - 200) <= 3 && abs(top.z - 235) <= 3, "top \(top)")
    expect(abs(bottom.x - 8) <= 3 && abs(bottom.y - 40) <= 3 && abs(bottom.z - 90) <= 3, "bottom \(bottom)")
}
test("the whole helix renders from buffers, twice in a row (buffers reused)") {
    try renderer.render(spheres: [], cylinders: [], camera: camera, into: buffer, width: small.width,
                        viewHeight: small.viewHeight)
    let empty = (0..<(small.width * small.viewHeight)).map { pixel($0 % small.width, $0 / small.width) }
    for angle: Float in [0, 90] {
        let (s, c) = sceneGeometry(dna, turn: angle, camera: camera)
        try renderer.render(spheres: s, cylinders: c, camera: camera, into: buffer, width: small.width,
                            viewHeight: small.viewHeight)
        var covered = 0
        for i in 0..<empty.count where pixel(i % small.width, i / small.width) != empty[i] { covered += 1 }
        let share = Double(covered) / Double(empty.count)
        expect(share > 0.1 && share < 0.6, "the helix covers \(share) of the view at \(angle)°")
    }
}
test("the caption bar is drawn under the view") {
    drawCaption(Caption(title: "DNA", subtitle: "s", facts: "f", aside: "a"), into: buffer, layout: small)
    let bar = pixel(small.width - 2, small.height - 2)
    expect(bar.x < 30 && bar.y < 30 && bar.z < 40, "\(bar)")
}

section("palette and quantizer")
test("median cut returns the colors asked for, each inside the samples' range") {
    var samples: [RGB] = []
    for r in stride(from: 0, to: 256, by: 5) { for g in stride(from: 0, to: 256, by: 17) { samples.append(RGB(UInt8(r), UInt8(g), 40)) } }
    let palette = medianCutPalette(samples, count: 96)
    expectEqual(palette.count, 96)
    expect(palette.allSatisfy { $0.z == 40 })
}
test("with the real 96-color palette, orange phosphorus stays orange, shaded side too") {
    var samples: [RGB] = []
    for angle: Float in [0, 90, 180, 270] {
        let (s, c) = sceneGeometry(dna, turn: angle, camera: camera)
        try renderer.render(spheres: s, cylinders: c, camera: camera, into: buffer, width: small.width,
                            viewHeight: small.viewHeight)
        samples += samplePixels(buffer, pixels: small.width * small.height, step: 3)
    }
    let palette = dnaPalette(samples: samples, count: 96)
    expectEqual(palette.count, 96)
    let quantizer = try Quantizer(device: device, palette: palette, pixels: small.width * small.height)
    let indices = try quantizer.indices(of: buffer)
    var orange = 0, keptOrange = 0
    for i in 0..<(small.width * small.viewHeight) {
        let p = buffer.contents().assumingMemoryBound(to: UInt8.self) + i * 4
        let r = Int(p[0]), g = Int(p[1]), b = Int(p[2])
        guard r > 180 && g > 90 && g - b > 60 else { continue }   // lit phosphorus
        orange += 1
        let q = palette[Int(indices[i])]
        if Int(q.y) - Int(q.z) > 40 && q.x > 150 { keptOrange += 1 }
    }
    expect(orange > 20, "only \(orange) orange pixels")
    expect(Double(keptOrange) > 0.9 * Double(orange), "\(keptOrange) of \(orange) stayed orange")
}
test("the GPU quantizer maps palette colors to their own index") {
    let palette: [RGB] = [RGB(10, 20, 30), RGB(200, 100, 50), RGB(255, 255, 255)]
    let frame = device.makeBuffer(length: 3 * 4, options: .storageModeShared)!
    let p = frame.contents().assumingMemoryBound(to: UInt8.self)
    for (i, c) in [palette[2], palette[0], palette[1]].enumerated() {
        p[i * 4] = c.x; p[i * 4 + 1] = c.y; p[i * 4 + 2] = c.z; p[i * 4 + 3] = 255
    }
    let q = try Quantizer(device: device, palette: palette, pixels: 3)
    expectEqual(try q.indices(of: frame), [2, 0, 1])
}

section("GIF encoder, decoded back by ImageIO")
/// Decodes every frame of a GIF to RGB triples.
func decode(_ url: URL) -> (frames: [[SIMD3<Int>]], loop: Int?) {
    let source = CGImageSourceCreateWithURL(url as CFURL, nil)!
    var out: [[SIMD3<Int>]] = []
    for i in 0..<CGImageSourceGetCount(source) {
        let image = CGImageSourceCreateImageAtIndex(source, i, nil)!
        let w = image.width, h = image.height
        var raw = [UInt8](repeating: 0, count: w * h * 4)
        let ctx = CGContext(data: &raw, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                            space: CGColorSpace(name: CGColorSpace.sRGB)!,
                            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
        var frame: [SIMD3<Int>] = []
        for i in 0..<(w * h) {
            let r = Int(raw[i * 4]), g = Int(raw[i * 4 + 1]), b = Int(raw[i * 4 + 2])
            frame.append(SIMD3(r, g, b))
        }
        out.append(frame)
    }
    let props = CGImageSourceCopyProperties(source, nil) as? [String: Any]
    let loop = (props?[kCGImagePropertyGIFDictionary as String] as? [String: Any])?[kCGImagePropertyGIFLoopCount as String] as? Int
    return (out, loop)
}
test("frames of changes only, over many LZW table resets, decode to exactly the right pictures") {
    // 200 colors and a busy pattern, so the LZW table fills and clears many times.
    var palette: [RGB] = []
    for i in 0..<200 {
        let blue: Int = (i * 7) % 256
        palette.append(RGB(UInt8(i), UInt8(255 - i), UInt8(blue)))
    }
    let w = 97, h = 61
    var expected: [[UInt8]] = []
    for f in 0..<4 {
        var frame = [UInt8](repeating: 0, count: w * h)
        for y in 0..<h { for x in 0..<w { frame[y * w + x] = UInt8((x * 31 + y * 17 + (x * y) % 13) % 200) } }
        // Each later frame changes a different small patch.
        if f > 0 { for y in (10 * f)..<(10 * f + 7) { for x in (20 * f)..<(20 * f + 9) { frame[y * w + x] = UInt8((f * 50 + x) % 200) } } }
        expected.append(frame)
    }
    let url = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("step8-test.gif")
    let gif = GIFWriter(url: url, width: w, height: h, palette: palette, delayCentiseconds: 10)
    for f in expected { gif.add(f) }
    try gif.finish()
    let (decoded, loop) = decode(url)
    expectEqual(decoded.count, 4)
    expectEqual(loop ?? -1, 0)
    for (f, frame) in expected.enumerated() where f < decoded.count {
        var wrong = 0
        for i in 0..<(w * h) {
            let c = palette[Int(frame[i])]
            if decoded[f][i] != SIMD3(Int(c.x), Int(c.y), Int(c.z)) { wrong += 1 }
        }
        expectEqual(wrong, 0)
    }
    try? FileManager.default.removeItem(at: url)
}
test("LZW on a single repeated value: one clear, few codes, ends with the end code") {
    let out = lzwEncode([UInt8](repeating: 7, count: 10_000))
    expect(out.count < 300, "\(out.count) bytes")
    expectEqual(out.first!, 0x00)   // the clear code 256 in 9 bits starts with 8 zero bits
}

finish()
