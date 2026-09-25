// Tests for the single-molecule render. Geometry tests check the shapes
// against the published bond lengths and angles, and basic chemistry
// (charge, valence); render tests check the ray tracer and the camera.

import CoreGraphics
import Foundation
import ImageIO
import Metal
import simd

func near(_ a: Float, _ b: Float, within e: Float) -> Bool { abs(a - b) <= e }

let acid = moleculeState(progress: 0, protonDistance: carbonicAcidGeometry.hydrogenBond)
let bicarb = moleculeState(progress: 1, protonDistance: 8)

section("carbonic acid, H₂CO₃")
test("C=O is 1.222 Å and both C–OH are 1.357 Å") {
    expect(near(distance(acid, 0, 1), 1.222, within: 1e-4), "C=O \(distance(acid, 0, 1))")
    expect(near(distance(acid, 0, 2), 1.357, within: 1e-4), "C–OH \(distance(acid, 0, 2))")
    expect(near(distance(acid, 0, 3), 1.357, within: 1e-4), "C–OH \(distance(acid, 0, 3))")
}
test("both O–H bonds are 0.980 Å with a C–O–H angle of 106°") {
    expect(near(distance(acid, 2, 4), 0.980, within: 1e-4))
    expect(near(distance(acid, 3, 5), 0.980, within: 1e-4))
    expect(near(angle(acid, 0, 2, 4), 106, within: 0.01), "C–O–H \(angle(acid, 0, 2, 4))")
    expect(near(angle(acid, 0, 3, 5), 106, within: 0.01))
}
test("the O–C–O angles add up to 360° (carbon is flat, trigonal)") {
    let total: Float = angle(acid, 1, 0, 2) + angle(acid, 2, 0, 3) + angle(acid, 3, 0, 1)
    expect(near(total, 360, within: 0.01), "sum \(total)")
    expect(near(angle(acid, 1, 0, 2), 125, within: 0.01))
}
test("it's cis-cis: both H's lean toward the C=O oxygen") {
    // Each H is closer to the carbonyl oxygen than its own oxygen's far side would put it.
    expect(simd_distance(acid.atoms[4].position, acid.atoms[1].position)
           < simd_distance(acid.atoms[2].position, acid.atoms[1].position) + 0.5)
    expect(acid.atoms[4].position.y > acid.atoms[2].position.y, "H on O2 should point up toward C=O")
    expect(acid.atoms[5].position.y > acid.atoms[3].position.y, "H on O3 should point up toward C=O")
}
test("it's mirror-symmetric, as the cis-cis form should be") {
    for (i, j) in [(2, 3), (4, 5)] {
        let a = acid.atoms[i].position, b = acid.atoms[j].position
        expect(near(a.x, -b.x, within: 1e-5) && near(a.y, b.y, within: 1e-5), "atoms \(i), \(j) not mirrored")
    }
}
test("it's flat: every atom lies in one plane") {
    expect(acid.atoms.allSatisfy { $0.position.z == 0 })
}

section("bicarbonate, HCO₃⁻")
test("the two oxygens without H become equal: both 1.25 Å") {
    expect(near(distance(bicarb, 0, 1), 1.25, within: 1e-4), "C–O1 \(distance(bicarb, 0, 1))")
    expect(near(distance(bicarb, 0, 3), 1.25, within: 1e-4), "C–O3 \(distance(bicarb, 0, 3))")
    expect(near(distance(bicarb, 0, 2), 1.36, within: 1e-4), "C–OH \(distance(bicarb, 0, 2))")
}
test("the O–C–O angles are 126° between the equal pair and 117° to the OH") {
    expect(near(angle(bicarb, 1, 0, 3), 126, within: 0.01), "\(angle(bicarb, 1, 0, 3))")
    expect(near(angle(bicarb, 1, 0, 2), 117, within: 0.01))
    expect(near(angle(bicarb, 3, 0, 2), 117, within: 0.01))
}
test("the equal oxygens share the double bond (1.5 each) and the charge (−½ each)") {
    expect(near(bicarb.bonds[0].order, 1.5, within: 1e-6))
    expect(near(bicarb.bonds[2].order, 1.5, within: 1e-6))
    expect(near(bicarb.atoms[1].charge, -0.5, within: 1e-6))
    expect(near(bicarb.atoms[3].charge, -0.5, within: 1e-6))
    expect(near(bicarb.atoms[5].charge, 1, within: 1e-6), "the free proton should carry +1")
}
test("the proton's bond is gone once it's far away") {
    expect(bicarb.bonds[4].order == 0)
}

section("chemistry throughout the change")
test("total charge stays zero (H⁺ + HCO₃⁻ is neutral overall)") {
    for p in stride(from: Float(0), through: 1, by: 0.1) {
        let s = moleculeState(progress: p, protonDistance: 1 + p * 3)
        let total: Float = s.atoms.reduce(0) { $0 + $1.charge }
        expect(abs(total) < 1e-6, "charge \(total) at progress \(p)")
    }
}
test("carbon always has 4 bonds' worth (bond orders add to 4)") {
    for p in stride(from: Float(0), through: 1, by: 0.1) {
        let s = moleculeState(progress: p, protonDistance: 1 + p * 3)
        expect(near(valence(s, of: 0), 4, within: 1e-5), "carbon valence \(valence(s, of: 0)) at \(p)")
    }
}
test("oxygen valence matches its charge at both ends: 2 bonds neutral, 1.5 at −½") {
    expect(near(valence(acid, of: 1), 2, within: 1e-6))
    expect(near(valence(acid, of: 3), 2, within: 1e-6))
    expect(near(valence(bicarb, of: 1), 1.5, within: 1e-6))
    expect(near(valence(bicarb, of: 3), 1.5, within: 1e-6))
    expect(near(valence(bicarb, of: 2), 2, within: 1e-6), "the OH oxygen keeps 2 bonds")
}
test("the proton only moves away, never back") {
    let t = Timeline()
    var previous: Float = 0
    for i in 0...120 {
        let d: Float = t.protonDistance(at: t.total * Double(i) / 120)
        expect(d >= previous - 1e-6, "proton moved back at step \(i)")
        previous = d
    }
    expect(previous > 3 && previous < 4.5, "the proton should end up clearly apart but in view (\(previous) Å)")
}
test("the timeline tells the story in order") {
    let t = Timeline()
    expectEqual(t.stage(at: 0.5), "Carbonic acid, H₂CO₃")
    expectEqual(t.stage(at: t.calm + 0.5), "H₂CO₃ → H⁺ + HCO₃⁻")
    expectEqual(t.stage(at: t.total - 0.1), "Bicarbonate, HCO₃⁻, and a free H⁺")
    expectEqual(t.progress(at: 0), 0)
    expectEqual(t.progress(at: t.total), 1)
}

section("drawing sticks")
test("a double bond is two sticks, a 1.5 bond is a full and a half-thick stick") {
    let (_, acidSticks) = sceneGeometry(acid)
    let (_, bicarbSticks) = sceneGeometry(bicarb)
    // Acid: C=O (2 sticks) + 2 × C–O + 2 × O–H = 6. Bicarbonate: 2 + 2 + 1 + 1 = 6 (proton bond gone).
    expectEqual(acidSticks.count, 6)
    expectEqual(bicarbSticks.count, 6)
    let halfThick = bicarbSticks.filter { near($0.aRadius.w, bondRadius * 0.5, within: 1e-6) }
    expectEqual(halfThick.count, 2)
}

section("camera and ray tracer (GPU)")
test("the camera's target lands in the middle of the image") {
    let cam = Camera.orbit(target: SIMD3(0.3, -0.2, 0), distance: 10, yaw: 20, pitch: 10, fov: 30)
    let p = cam.project(SIMD3(0.3, -0.2, 0), width: 640, height: 400)
    expect(near(p.x, 320, within: 0.01) && near(p.y, 200, within: 0.01), "projected to \(p)")
}
test("a point up and to the right lands up and to the right") {
    let cam = Camera.orbit(target: .zero, distance: 10, yaw: 0, pitch: 0, fov: 30)
    let p = cam.project(SIMD3(1, 1, 0), width: 640, height: 400)
    expect(p.x > 320 && p.y < 200, "projected to \(p)")
}
func renderOne(_ state: MoleculeState, camera: Camera, width: Int = 96, height: Int = 64) throws -> (Int, Int) -> [Int] {
    let device = try findDevice()
    let renderer = try MoleculeRenderer(device: device)
    let frame = device.makeBuffer(length: width * height * 4, options: .storageModeShared)!
    try renderer.render(state, camera: camera, into: frame, width: width, viewHeight: height)
    let px = frame.contents().bindMemory(to: UInt8.self, capacity: width * height * 4)
    return { x, y in (0..<3).map { Int(px[(y * width + x) * 4 + $0]) } }
}
test("a lone oxygen shows up red in the middle, over step 2's gradient") {
    let oxygen = MoleculeState(atoms: [Atom(element: .oxygen, position: .zero)], bonds: [])
    let pixel = try renderOne(oxygen, camera: Camera.orbit(target: .zero, distance: 4, yaw: 0, pitch: 0, fov: 30))
    let center = pixel(48, 32)
    expect(center[0] > 120 && center[0] > center[1] * 2, "center should be red, got \(center)")
    let top = pixel(2, 0), bottom = pixel(2, 63)
    expect(abs(top[0] - 140) <= 3 && abs(top[1] - 200) <= 3 && abs(top[2] - 235) <= 3,
           "top should be step 2's sky blue (140, 200, 235), got \(top)")
    expect(abs(bottom[0] - 8) <= 3 && abs(bottom[1] - 40) <= 3 && abs(bottom[2] - 90) <= 3,
           "bottom should be step 2's deep blue (8, 40, 90), got \(bottom)")
}
test("the nearer atom hides the farther one") {
    let pair = MoleculeState(atoms: [Atom(element: .hydrogen, position: SIMD3(0, 0, 1)),
                                     Atom(element: .oxygen, position: SIMD3(0, 0, -1))], bonds: [])
    let pixel = try renderOne(pair, camera: Camera.orbit(target: .zero, distance: 6, yaw: 0, pitch: 0, fov: 30))
    let center = pixel(48, 32)
    expect(center[1] > 150 && center[2] > 150, "the white H in front should win, got \(center)")
}
test("a bond is drawn between its atoms") {
    let stick = MoleculeState(atoms: [Atom(element: .carbon, position: SIMD3(-1.5, 0, 0)),
                                      Atom(element: .carbon, position: SIMD3(1.5, 0, 0))],
                              bonds: [Bond(a: 0, b: 1, order: 1)])
    let pixel = try renderOne(stick, camera: Camera.orbit(target: .zero, distance: 6, yaw: 0, pitch: 0, fov: 30))
    let middle = pixel(48, 32)
    expect(middle.max()! > 90, "the bond's middle should be lit grey, got \(middle)")
}
test("a positive charge glows gold around its atom") {
    let proton = MoleculeState(atoms: [Atom(element: .hydrogen, position: .zero, charge: 1)], bonds: [])
    let plain = MoleculeState(atoms: [Atom(element: .hydrogen, position: .zero)], bonds: [])
    let cam = Camera.orbit(target: .zero, distance: 4, yaw: 0, pitch: 0, fov: 30)
    let glowing = try renderOne(proton, camera: cam)(62, 32)
    let dark = try renderOne(plain, camera: cam)(62, 32)
    expect(glowing[0] > dark[0] + 20, "glow missing beside the proton: \(glowing) vs \(dark)")
}

finish()
