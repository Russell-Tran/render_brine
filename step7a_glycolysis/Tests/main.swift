// Tests for step 7a. The chemistry itself (every species against PubChem,
// every reaction balanced) is checked in Python by `make check`; these tests
// check what the Swift side does with it: sensible shapes at every keyframe,
// no atoms crashing into each other, the ledger, the seamless loops, the
// overlay text, and the ray tracer and GIF writer.

import CoreGraphics
import Foundation
import ImageIO
import Metal
import simd

let timelineURL = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent().deletingLastPathComponent()
    .appendingPathComponent("Resources/timeline.json")
let movies = try loadMovies(from: timelineURL)
let spend = movies.first { $0.name == "spend" }!
let payoff = movies.first { $0.name == "payoff" }!
let fps = 15.0

/// Typical bond length ranges (Å) by element pair, from Allen et al., "Tables
/// of bond lengths determined by X-ray and neutron diffraction", J. Chem. Soc.
/// Perkin Trans. 2 (1987), widened a little for force-field geometry.
func bondRange(_ a: String, _ b: String) -> ClosedRange<Float> {
    switch [a, b].sorted().joined() {
    case "CC": return 1.30...1.60
    case "CO": return 1.18...1.47
    case "CH", "HO": return 0.93...1.13
    case "OP": return 1.45...1.70
    default: return 0...0
    }
}

/// Visible atoms and full bonds at one keyframe.
func solid(_ k: Keyframe) -> (atoms: [String: AtomKey], bonds: [(BondKey, Float)]) {
    let atoms = k.atoms.filter { $0.value.visible > 0.99 }
    let bonds = k.bonds.filter { $0.value > 0.99 && atoms[$0.key.a] != nil && atoms[$0.key.b] != nil }
    return (atoms, bonds.map { ($0.key, $0.value) })
}

/// The atoms on screen as sorted (element, rounded position) strings, so two
/// frames can be compared regardless of labels ("C1" vs "n_C1"). Atoms
/// waiting or leaving off-screen don't count.
func picture(_ f: Frame, _ m: Movie) -> [String] {
    let mainCamera = m.camera(at: 0)
    return f.atoms.filter { atom in
        let p = mainCamera.project(atom.key.position, width: 1280, height: 800)
        return atom.key.visible > 0.01 && p.x > -40 && p.x < 1320 && p.y > -40 && p.y < 840
    }.map { atom -> String in
        let p = atom.key.position
        return String(format: "%@ %.3f %.3f %.3f %.2f", atom.element, p.x, p.y, p.z, atom.key.visible)
    }.sorted()
}

section("timeline")
test("two loops, spend then payoff, each at least 10 s") {
    expectEqual(movies.map { $0.name }, ["spend", "payoff"])
    for m in movies { expect(m.duration >= 10, "\(m.name) is \(m.duration) s") }
}
test("keyframes run in order from 0 to the end of each loop") {
    for m in movies {
        expectEqual(m.keys.first!.time, 0)
        expect(abs(m.keys.last!.time - m.duration) < 1e-9)
        for i in 1..<m.keys.count { expect(m.keys[i].time > m.keys[i - 1].time, "\(m.name) key \(i)") }
    }
}
test("every atom has a known element") {
    for m in movies {
        for (label, el) in m.elements { expect(["C", "H", "O", "P"].contains(el), "\(label) is \(el)") }
    }
}

section("shapes at every keyframe")
test("every drawn bond has a real bond length for its elements") {
    for m in movies {
        for (i, k) in m.keys.enumerated() {
            let s = solid(k)
            for (bond, _) in s.bonds {
                let ea = m.elements[bond.a]!, eb = m.elements[bond.b]!
                let d = simd_distance(s.atoms[bond.a]!.position, s.atoms[bond.b]!.position)
                expect(bondRange(ea, eb).contains(d), "\(m.name) key \(i): \(bond.a)–\(bond.b) is \(d) Å")
            }
        }
    }
}
test("no two unbonded atoms closer than 1.5 Å at any keyframe") {
    for m in movies {
        for (i, k) in m.keys.enumerated() {
            let s = solid(k)
            let bonded = Set(s.bonds.map { $0.0 })
            let labels = s.atoms.keys.sorted()
            for x in 0..<labels.count {
                for y in (x + 1)..<labels.count where !bonded.contains(BondKey(labels[x], labels[y])) {
                    let d = simd_distance(s.atoms[labels[x]]!.position, s.atoms[labels[y]]!.position)
                    expect(d >= 1.5, "\(m.name) key \(i): \(labels[x]) and \(labels[y]) are \(d) Å apart")
                }
            }
        }
    }
}
test("no two atoms ever pass through each other (every frame, heavy atoms ≥ 1.0 Å unless bonded)") {
    for m in movies {
        let n = Int((m.duration * fps).rounded())
        var worst: Float = .infinity, worstAt = ""
        for f in 0..<n {
            let frame = m.frame(at: Double(f) / fps)
            let bonded = Set(frame.bonds.map { BondKey($0.a, $0.b) })
            let heavy = frame.atoms.filter { $0.element != "H" && $0.key.visible > 0.5 }
            for x in 0..<heavy.count {
                for y in (x + 1)..<heavy.count where !bonded.contains(BondKey(heavy[x].label, heavy[y].label)) {
                    let d = simd_distance(heavy[x].key.position, heavy[y].key.position)
                    if d < worst { worst = d; worstAt = "\(m.name) frame \(f): \(heavy[x].label)/\(heavy[y].label)" }
                }
            }
        }
        expect(worst >= 1.0, "closest approach \(worst) Å at \(worstAt)")
    }
}

section("the story")
test("spend: glucose starts whole, C3–C4 breaks at aldolase") {
    expect(spend.keys.first!.bonds[BondKey("C3", "C4")] == 1)
    let aldolase = spend.keys.last { $0.enzyme == "aldolase" }!
    expect(aldolase.bonds[BondKey("C3", "C4")] == nil)
}
test("spend ledger: 2 ATP spent, none made, then back to zero for the next glucose") {
    let ledgers = spend.keys.map { $0.ledger }
    expect(ledgers.contains([2, 0, 0]))
    expect(ledgers.allSatisfy { $0[1] == 0 && $0[2] == 0 && $0[0] <= 2 })
    expectEqual(ledgers.first!, [0, 0, 0])
    expectEqual(ledgers.last!, [0, 0, 0])
}
test("payoff ledger: 4 ATP made against 2 spent, and 2 NADH") {
    let ledgers = payoff.keys.map { $0.ledger }
    expect(ledgers.contains([2, 4, 2]), "\(ledgers)")
    expect(ledgers.allSatisfy { $0[1] <= 4 && $0[2] <= 2 })
    expectEqual(netLabel(spent: 2, made: 4), "net +2")
}
test("payoff: each half stays on its own side, so the carbons can be followed into their pyruvate") {
    for k in payoff.keys {
        for (label, atom) in k.atoms where atom.visible > 0.99 {
            guard let n = carbonNumber(label), !label.hasPrefix("n_"), let c = Int(n) else { continue }
            // Carbons 1–3 are half A (right), 4–6 half B (left).
            expect(c <= 3 ? atom.position.x > 0 : atom.position.x < 0, "\(label) at x = \(atom.position.x)")
        }
    }
}
test("payoff ends with methyl carbons: C1 and C6 each carry three H") {
    let end = payoff.keys.last { $0.title.lowercased().contains("pyruvate") }!
    for carbon in ["C1", "C6"] {
        let hs = end.bonds.keys.filter { ($0.a == carbon || $0.b == carbon) && end.bonds[$0]! > 0.99 }
            .map { $0.a == carbon ? $0.b : $0.a }.filter { payoff.elements[$0] == "H" }
        expectEqual(hs.count, 3)
    }
}
test("the free H⁺ and phosphates glow only while charged") {
    for m in movies {
        for k in m.keys {
            for (label, a) in k.atoms where a.glow > 0 { expect(a.charge != 0, "\(m.name) \(label)") }
        }
    }
}

section("seamless loops")
test("the last frame shows exactly the picture of the first (the next molecules are in place)") {
    for m in movies {
        expectEqual(picture(m.frame(at: 0), m), picture(m.frame(at: m.duration - 1e-9), m))
        expectEqual(m.keys.first!.title, m.keys.last!.title)
        expectEqual(m.keys.first!.ledger, m.keys.last!.ledger)
    }
}
test("at every keyframe each atom is well inside the frame or well outside it (waiting or gone)") {
    for m in movies {
        for (i, k) in m.keys.enumerated() {
            let cam = m.camera(at: k.time)
            for (label, atom) in k.atoms where atom.visible > 0.99 {
                let p = cam.project(atom.position, width: 1280, height: 800)
                let inside = p.x > 20 && p.x < 1260 && p.y > 20 && p.y < 780
                let outside = p.x < -30 || p.x > 1310 || p.y < -30 || p.y > 830
                expect(inside || outside, "\(m.name) key \(i): \(label) at \(p)")
            }
        }
    }
}
test("the camera comes back to where it started") {
    for m in movies {
        expect(simd_distance(m.camera(at: 0).origin, m.camera(at: m.duration).origin) < 1e-4)
    }
}
test("frame(at:) wraps around") {
    expectEqual(picture(payoff.frame(at: 3), payoff), picture(payoff.frame(at: 3 + payoff.duration), payoff))
}

section("overlay")
test("net labels use a real minus sign") {
    expectEqual(netLabel(spent: 2, made: 0), "net −2")
    expectEqual(netLabel(spent: 2, made: 2), "net 0")
}
test("carbon numbers come from glucose's labels, including the next molecule's") {
    expectEqual(carbonNumber("C3"), "3")
    expectEqual(carbonNumber("n_C6"), "6")
    expectEqual(carbonNumber("O3"), nil)
    expectEqual(carbonNumber("Pa"), nil)
}

section("geometry for the GPU")
test("bonds fade out as their atoms separate") {
    expectEqual(bondFade(length: 1.5), 1)
    expectEqual(bondFade(length: 2.4), 0)
}
test("a double bond is two sticks, a single one, an invisible atom no ball") {
    let cam = Camera.orbit(target: .zero, distance: 25, yaw: 0, pitch: 8, fov: 30)
    let frame = Frame(atoms: [("C1", "C", AtomKey(position: .zero, visible: 1, charge: 0, glow: 0)),
                              ("O1", "O", AtomKey(position: SIMD3(1.22, 0, 0), visible: 1, charge: 0, glow: 0)),
                              ("H1", "H", AtomKey(position: SIMD3(-1, 0, 0), visible: 0, charge: 0, glow: 0))],
                      bonds: [("C1", "O1", 2)], tokens: [], title: "", enzyme: "", equation: "", ledger: [0, 0, 0])
    let (spheres, cylinders) = sceneGeometry(frame, camera: cam)
    expectEqual(spheres.count, 2)
    expectEqual(cylinders.count, 2)
}
test("the scene always fits Metal's 4 KB setBytes limit") {
    var most = 0
    for m in movies {
        let cam = Camera.orbit(target: .zero, distance: 25, yaw: 0, pitch: 8, fov: 30)
        for f in stride(from: 0, to: Int(m.duration * fps), by: 3) {
            let (s, c) = sceneGeometry(m.frame(at: Double(f) / fps), camera: cam)
            most = max(most, s.count * MemoryLayout<GPUSphere>.stride, c.count * MemoryLayout<GPUCylinder>.stride)
        }
    }
    expect(most <= 4096, "\(most) bytes")
}

section("renderer")
let device = try findDevice()
let renderer = try MoleculeRenderer(device: device)
let layout = FrameLayout(width: 320, viewHeight: 200, captionHeight: 40)
let buffer = device.makeBuffer(length: layout.width * layout.height * 4, options: .storageModeShared)!
func pixel(_ x: Int, _ y: Int) -> SIMD3<Int> {
    let p = buffer.contents().assumingMemoryBound(to: UInt8.self) + (y * layout.width + x) * 4
    return SIMD3(Int(p[0]), Int(p[1]), Int(p[2]))
}
test("an empty scene is step 2's gradient, light blue at the top to navy at the bottom") {
    let cam = Camera.orbit(target: .zero, distance: 25, yaw: 0, pitch: 0, fov: 30)
    try renderer.render(spheres: [], cylinders: [], camera: cam, into: buffer, width: layout.width,
                        viewHeight: layout.viewHeight)
    let top = pixel(160, 0), bottom = pixel(160, layout.viewHeight - 1)
    expect(abs(top.x - 140) <= 3 && abs(top.y - 200) <= 3 && abs(top.z - 235) <= 3, "top \(top)")
    expect(abs(bottom.x - 8) <= 3 && abs(bottom.y - 40) <= 3 && abs(bottom.z - 90) <= 3, "bottom \(bottom)")
}
test("the first spend frame puts glucose in the middle, with the caption drawn below") {
    let frame = spend.frame(at: 0)
    let cam = Camera.orbit(target: .zero, distance: 25, yaw: 0, pitch: 8, fov: 30)
    let (s, c) = sceneGeometry(frame, camera: cam)
    try renderer.render(spheres: s, cylinders: c, camera: cam, into: buffer, width: layout.width,
                        viewHeight: layout.viewHeight)
    let background = pixel(5, layout.viewHeight / 2)
    var differs = 0
    for x in 140..<180 where pixel(x, layout.viewHeight / 2) != background { differs += 1 }
    expect(differs > 5, "only \(differs) molecule pixels")
    drawOverlay(frame, camera: cam, into: buffer, layout: layout)
    let bar = pixel(layout.width - 2, layout.height - 2)
    expect(bar.x < 30 && bar.y < 30 && bar.z < 40, "caption bar \(bar)")
}
test("the GIF writer makes a looping two-frame GIF") {
    let url = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("step7a-test.gif")
    let gif = try GIFWriter(url: url, frameCount: 2, delay: 1 / fps)
    gif.add(buffer, layout: layout)
    gif.add(buffer, layout: layout)
    try gif.finish()
    let source = CGImageSourceCreateWithURL(url as CFURL, nil)!
    expectEqual(CGImageSourceGetCount(source), 2)
    let props = CGImageSourceCopyProperties(source, nil) as? [String: Any]
    let loop = (props?[kCGImagePropertyGIFDictionary as String] as? [String: Any])?[kCGImagePropertyGIFLoopCount as String]
    expectEqual((loop as? Int) ?? -1, 0)
    try? FileManager.default.removeItem(at: url)
}

finish()
