// Tests for step 8. Geometry tests check each key shape against its source;
// bookkeeping tests check every frame of the loop; renderer tests check the
// ray tracer and the GIF writer.

import CoreGraphics
import Foundation
import ImageIO
import Metal
import simd

func near(_ a: Float, _ b: Float, within e: Float) -> Bool { abs(a - b) <= e }

let poses = buildPoses()
let timeline = Timeline()
let fps = 20.0
let frameCount: Int = Int((timeline.total * fps).rounded())
var frameTimes: [Double] = []
for f in 0..<frameCount { frameTimes.append(Double(f) / fps) }

let gathered = poseState(poses.gathered, hydration: 0, transfer: 0)
let acid = poseState(poses.acid, hydration: 1, transfer: 0)
let cisAcid = poseState(poses.cisAcid, hydration: 1, transfer: 0)
let products = poseState(poses.products, hydration: 1, transfer: 1)

/// Is atom `h`'s O–H on the same side of the C–O line as O1 (cis)?
func isCis(_ s: MoleculeState, oxygen o: Int, hydrogen h: Int) -> Bool {
    let c = s.atoms[Slot.carbon].position, op = s.atoms[o].position
    let line = op - c
    let hSide: Float = simd_cross(line, s.atoms[h].position - c).z
    let refSide: Float = simd_cross(line, s.atoms[Slot.o1].position - c).z
    return (hSide > 0) == (refSide > 0)
}

section("reactants: CO₂ and water")
test("CO₂ is straight with both C=O bonds 1.163 Å") {
    expect(near(distance(gathered, Slot.carbon, Slot.o1), 1.163, within: 1e-4))
    expect(near(distance(gathered, Slot.carbon, Slot.o3), 1.163, within: 1e-4))
    expect(near(angle(gathered, Slot.o1, Slot.carbon, Slot.o3), 180, within: 0.01))
}
test("both waters have O–H 0.9572 Å and H–O–H 104.52°") {
    for (o, h1, h2) in [(Slot.o2, Slot.hKept, Slot.hRelayed), (Slot.helperO, Slot.proton, Slot.hHelper)] {
        expect(near(distance(gathered, o, h1), 0.9572, within: 1e-4))
        expect(near(distance(gathered, o, h2), 0.9572, within: 1e-4))
        expect(near(angle(gathered, h1, o, h2), 104.52, within: 0.01), "angle \(angle(gathered, h1, o, h2))")
    }
}
test("the relay ring is set up: water H points at the helper, helper H points at CO₂") {
    let toHelper = simd_normalize(poses.gathered[Slot.helperO] - poses.gathered[Slot.o2])
    let hDir = simd_normalize(poses.gathered[Slot.hRelayed] - poses.gathered[Slot.o2])
    expect(simd_dot(toHelper, hDir) > 0.999)
    let toO3 = simd_normalize(poses.gathered[Slot.o3] - poses.gathered[Slot.helperO])
    let pDir = simd_normalize(poses.gathered[Slot.proton] - poses.gathered[Slot.helperO])
    expect(simd_dot(toO3, pDir) > 0.999)
}

section("carbonic acid, H₂CO₃")
test("C=O 1.222 Å, both C–OH 1.357 Å, O–H 0.980 Å, flat with 125° between C=O and each C–OH") {
    for s in [acid, cisAcid] {
        expect(near(distance(s, Slot.carbon, Slot.o1), 1.222, within: 1e-4))
        expect(near(distance(s, Slot.carbon, Slot.o2), 1.357, within: 1e-4))
        expect(near(distance(s, Slot.carbon, Slot.o3), 1.357, within: 1e-4))
        expect(near(distance(s, Slot.o2, Slot.hKept), 0.980, within: 1e-4))
        expect(near(distance(s, Slot.o3, Slot.proton), 0.980, within: 1e-4))
        expect(near(angle(s, Slot.o1, Slot.carbon, Slot.o2), 125, within: 0.01))
        expect(near(angle(s, Slot.o1, Slot.carbon, Slot.o3), 125, within: 0.01))
        expect(near(angle(s, Slot.carbon, Slot.o3, Slot.proton), 106, within: 0.01))
    }
}
test("CO₂ bends from 180° to 125° during hydration") {
    let start = angle(gathered, Slot.o1, Slot.carbon, Slot.o3)
    let end = angle(acid, Slot.o1, Slot.carbon, Slot.o3)
    expect(near(start, 180, within: 0.01) && near(end, 125, within: 0.01), "\(start)° → \(end)°")
}
test("the relayed proton arrives trans, then the OH turns to cis-cis (the most stable shape)") {
    expect(!isCis(acid, oxygen: Slot.o3, hydrogen: Slot.proton), "should arrive trans")
    expect(isCis(cisAcid, oxygen: Slot.o3, hydrogen: Slot.proton), "should end cis")
    expect(isCis(cisAcid, oxygen: Slot.o2, hydrogen: Slot.hKept), "the other OH is cis too")
}
test("the helper waiting in act 3 is still a proper water, hydrogen-bonded to the proton") {
    expect(near(distance(cisAcid, Slot.helperO, Slot.hRelayed), 0.9572, within: 1e-4))
    expect(near(distance(cisAcid, Slot.helperO, Slot.hHelper), 0.9572, within: 1e-4))
    expect(near(angle(cisAcid, Slot.hRelayed, Slot.helperO, Slot.hHelper), 104.52, within: 0.01))
    let hBond = distance(cisAcid, Slot.helperO, Slot.proton)
    expect(hBond > 1.6 && hBond < 2.0, "H···O \(hBond) Å should be a hydrogen bond (~1.75)")
}

section("products: HCO₃⁻ and H₃O⁺")
test("bicarbonate: two equal C–O 1.25 Å, C–OH 1.36 Å, O–H 0.97 Å, angles 126° and 117°") {
    expect(near(distance(products, Slot.carbon, Slot.o1), 1.25, within: 1e-4))
    expect(near(distance(products, Slot.carbon, Slot.o3), 1.25, within: 1e-4))
    expect(near(distance(products, Slot.carbon, Slot.o2), 1.36, within: 1e-4))
    expect(near(distance(products, Slot.o2, Slot.hKept), 0.97, within: 1e-4))
    expect(near(angle(products, Slot.o1, Slot.carbon, Slot.o3), 126, within: 0.01))
    expect(near(angle(products, Slot.o1, Slot.carbon, Slot.o2), 117, within: 0.01))
}
test("H₃O⁺: three O–H of 0.974 Å at 113.6°, pyramidal (not flat)") {
    let hs = [Slot.proton, Slot.hRelayed, Slot.hHelper]
    for h in hs { expect(near(distance(products, Slot.helperO, h), 0.974, within: 1e-4)) }
    for (i, j) in [(0, 1), (1, 2), (0, 2)] {
        let a = angle(products, hs[i], Slot.helperO, hs[j])
        expect(near(a, 113.6, within: 0.01), "H–O–H \(a)")
    }
    // Flat would put O in the plane of its three H's; a pyramid lifts it out (~0.25 Å here).
    let h0 = poses.products[Slot.proton], h1 = poses.products[Slot.hRelayed], h2 = poses.products[Slot.hHelper]
    let normal = simd_normalize(simd_cross(h1 - h0, h2 - h0))
    let lift: Float = abs(simd_dot(poses.products[Slot.helperO] - h0, normal))
    expect(lift > 0.2 && lift < 0.3, "O sits \(lift) Å from the plane of its H's")
}
test("the new H₃O⁺ still points its fresh O–H back at bicarbonate") {
    let toO3 = simd_normalize(poses.products[Slot.o3] - poses.products[Slot.helperO])
    let h = simd_normalize(poses.products[Slot.proton] - poses.products[Slot.helperO])
    expect(simd_dot(toO3, h) > 0.999)
}

section("bookkeeping on every frame")
test("atoms are conserved: one C, four O, four H per group on screen") {
    for t in frameTimes {
        let s = journeyState(poses, timeline, at: t)
        let groups = s.atoms.count / Slot.count
        expect(s.atoms.count % Slot.count == 0 && (groups == 1 || groups == 2), "odd atom count at \(t)")
        let c = s.atoms.filter { $0.element == .carbon }.count
        let o = s.atoms.filter { $0.element == .oxygen }.count
        let h = s.atoms.filter { $0.element == .hydrogen }.count
        if c != groups || o != 4 * groups || h != 4 * groups { return expect(false, "counts \(c)/\(o)/\(h) at \(t) s") }
    }
}
test("total charge is zero on every frame") {
    for t in frameTimes {
        let total: Float = journeyState(poses, timeline, at: t).atoms.reduce(0) { $0 + $1.charge }
        if abs(total) > 1e-5 { return expect(false, "charge \(total) at \(t) s") }
    }
}
test("bonds match charges on every frame: C has 4, H has 1, O has 2 plus its charge") {
    for t in frameTimes {
        let s = journeyState(poses, timeline, at: t)
        for (i, atom) in s.atoms.enumerated() {
            let v = valence(s, of: i)
            let expected: Float
            switch atom.element {
            case .carbon: expected = 4
            case .hydrogen: expected = 1
            case .oxygen: expected = 2 + atom.charge
            }
            if !near(v, expected, within: 1e-4) {
                return expect(false, "\(atom.element) slot \(i % Slot.count) has \(v), expected \(expected), at \(t) s")
            }
        }
    }
}
test("unbonded atoms never pass through each other") {
    var worst: Float = 99
    var at = ""
    for t in stride(from: 0.0, to: timeline.total, by: 0.02) {
        let s = journeyState(poses, timeline, at: t)
        for i in 0..<s.atoms.count {
            for j in (i + 1)..<s.atoms.count {
                let bonded = s.bonds.contains { ($0.a == i && $0.b == j || $0.a == j && $0.b == i) && $0.order > 0.001 }
                if bonded { continue }
                let d = distance(s, i, j)
                if d < worst { worst = d; at = "slots \(i % Slot.count), \(j % Slot.count) at \(String(format: "%.2f", t)) s" }
            }
        }
    }
    expect(worst > 1.0, "closest unbonded pair \(worst) Å (\(at))")
}
test("the loop is seamless: the frame after the last one is the first") {
    let first = journeyState(poses, timeline, at: 0)
    let almost = journeyState(poses, timeline, at: timeline.total - 1e-4)
    // The next group, arriving, must sit exactly where the first frame starts.
    let arriving = Array(almost.atoms[Slot.count..<(2 * Slot.count)])
    for i in 0..<Slot.count {
        let d = simd_distance(arriving[i].position, first.atoms[i].position)
        expect(d < 1e-3, "slot \(i) off by \(d) Å")
    }
    // And the old products must be far out of the picture by then.
    let oldCarbon = almost.atoms[Slot.carbon].position
    let oldHelper = almost.atoms[Slot.helperO].position
    expect(oldCarbon.y < -9 && oldHelper.x < -12, "old products still near: \(oldCarbon), \(oldHelper)")
}
test("the stage captions run in story order") {
    expectEqual(timeline.stage(at: 1), "CO₂ meets water")
    expectEqual(timeline.stage(at: timeline.act3Start + 0.1), "Carbonic acid, H₂CO₃")
    expectEqual(timeline.stage(at: timeline.act4Start + 0.1), "H₂CO₃ + H₂O → HCO₃⁻ + H₃O⁺")
    expect(abs(timeline.total - 13.5) < 1e-9)
}

section("renderer (GPU)")
func renderOne(_ state: MoleculeState, camera: Camera, width: Int = 96, height: Int = 64) throws -> (Int, Int) -> [Int] {
    let device = try findDevice()
    let renderer = try MoleculeRenderer(device: device)
    let frame = device.makeBuffer(length: width * height * 4, options: .storageModeShared)!
    try renderer.render(state, camera: camera, into: frame, width: width, viewHeight: height)
    let px = frame.contents().bindMemory(to: UInt8.self, capacity: width * height * 4)
    return { x, y in (0..<3).map { Int(px[(y * width + x) * 4 + $0]) } }
}
test("an oxygen renders red over step 2's gradient") {
    let o = MoleculeState(atoms: [Atom(element: .oxygen, position: .zero)], bonds: [])
    let pixel = try renderOne(o, camera: Camera.orbit(target: .zero, distance: 4, yaw: 0, pitch: 0, fov: 30))
    let center = pixel(48, 32)
    expect(center[0] > 120 && center[0] > center[1] * 2, "center should be red, got \(center)")
    let top = pixel(2, 0), bottom = pixel(2, 63)
    expect(abs(top[0] - 140) <= 4 && abs(top[2] - 235) <= 4, "top should be sky blue, got \(top)")
    expect(abs(bottom[0] - 8) <= 4 && abs(bottom[2] - 90) <= 4, "bottom should be deep blue, got \(bottom)")
}
test("a charged atom's glow stays behind it: the atom keeps its own color") {
    let cam = Camera.orbit(target: .zero, distance: 4, yaw: 0, pitch: 0, fov: 30)
    let plain = try renderOne(MoleculeState(atoms: [Atom(element: .oxygen, position: .zero)], bonds: []), camera: cam)
    let charged = try renderOne(MoleculeState(atoms: [Atom(element: .oxygen, position: .zero, charge: 1, glow: 1)], bonds: []),
                                camera: cam)
    expectEqual(charged(48, 32), plain(48, 32))
    let beside = charged(66, 32), besidePlain = plain(66, 32)
    expect(beside[0] > besidePlain[0] + 10, "no glow beside the atom: \(beside) vs \(besidePlain)")
}
test("the GIF writer writes every frame") {
    let device = try findDevice()
    let layout = FrameLayout(width: 32, viewHeight: 20, captionHeight: 4)
    let frame = device.makeBuffer(length: layout.width * layout.height * 4, options: .storageModeShared)!
    let url = FileManager.default.temporaryDirectory.appendingPathComponent("journey_tests-\(getpid()).gif")
    defer { try? FileManager.default.removeItem(at: url) }
    let gif = try GIFWriter(url: url, frameCount: 3, delay: 0.05)
    for _ in 0..<3 { gif.add(frame, layout: layout) }
    try gif.finish()
    expectEqual(CGImageSourceCreateWithURL(url as CFURL, nil).map { CGImageSourceGetCount($0) }, 3)
}

finish()
