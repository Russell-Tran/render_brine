// Tests for step 7. Chemistry tests check atom bookkeeping, oxidation states
// and product geometry; glucose tests check PubChem's structure; render tests
// check the ray tracer, including the step 6 feedback (glows stay behind
// atoms, the background is step 2's gradient, dither stays faint).

import Foundation
import Metal
import simd

func near(_ a: Float, _ b: Float, within e: Float) -> Bool { abs(a - b) <= e }

let reaction = buildReaction()
let reactants = reactionState(reaction, progress: 0)
let products = reactionState(reaction, progress: 1)
let before = oxidationStates(elements: reaction.elements, bonds: reaction.reactantBonds)
let after = oxidationStates(elements: reaction.elements, bonds: reaction.productBonds)

func count(_ e: Element, in elements: [Element]) -> Int { elements.filter { $0 == e }.count }
func neighbors(_ i: Int, _ bonds: [Bond]) -> [Int] {
    bonds.compactMap { $0.a == i ? $0.b : ($0.b == i ? $0.a : nil) }
}

section("glucose from PubChem")
test("formula is C₆H₁₂O₆") {
    let elements = glucoseAtoms.map { $0.0 }
    expectEqual(count(.carbon, in: elements), 6)
    expectEqual(count(.hydrogen, in: elements), 12)
    expectEqual(count(.oxygen, in: elements), 6)
}
test("bond lengths are sensible: C–C ≈ 1.52, C–O ≈ 1.42, C–H ≈ 1.10, O–H ≈ 0.97 Å") {
    for (a, b) in glucoseBondPairs {
        let ea = glucoseAtoms[a].0, eb = glucoseAtoms[b].0
        let d = simd_distance(glucoseAtoms[a].1, glucoseAtoms[b].1)
        let pair = Set([ea, eb])
        let expected: Float
        if pair == [.carbon] { expected = 1.52 }
        else if pair == [.carbon, .oxygen] { expected = 1.42 }
        else if pair == [.carbon, .hydrogen] { expected = 1.10 }
        else { expected = 0.97 }
        expect(near(d, expected, within: 0.05), "bond \(a)–\(b): \(d) Å, expected about \(expected)")
    }
}
test("it has a six-membered ring of one O and five C") {
    let bonds = glucoseBondPairs.map { Bond(a: $0.0, b: $0.1, order: 1) }
    for i in 0..<glucoseRing.count {
        let a = glucoseRing[i], b = glucoseRing[(i + 1) % glucoseRing.count]
        expect(neighbors(a, bonds).contains(b), "ring atoms \(a) and \(b) aren't bonded")
    }
    expectEqual(glucoseRing.filter { glucoseAtoms[$0].0 == .carbon }.count, 5)
}
test("every carbon has 4 bonds, every oxygen 2, every hydrogen 1") {
    let bonds = glucoseBondPairs.map { Bond(a: $0.0, b: $0.1, order: 1) }
    for (i, atom) in glucoseAtoms.enumerated() {
        let want = atom.0 == .carbon ? 4 : (atom.0 == .oxygen ? 2 : 1)
        expectEqual(neighbors(i, bonds).count, want)
    }
}

section("atom bookkeeping")
test("the same 36 atoms at every moment: 6 C, 12 H, 18 O") {
    for step in 0...40 {
        let s = reactionState(reaction, progress: Float(step) / 40)
        let elements = s.atoms.map { $0.element }
        expectEqual(elements.count, 36)
        expectEqual(count(.carbon, in: elements), 6)
        expectEqual(count(.hydrogen, in: elements), 12)
        expectEqual(count(.oxygen, in: elements), 18)
    }
}
test("every atom ends in exactly one product molecule, bonded correctly") {
    // Each carbon: 2 double bonds to O. Each H: 1 bond. Each O: in CO₂ (1 bond) or H₂O (2 bonds).
    for i in 0..<36 {
        let n = neighbors(i, reaction.productBonds)
        switch reaction.elements[i] {
        case .carbon: expectEqual(n.count, 2)
        case .hydrogen: expectEqual(n.count, 1)
        case .oxygen: expect(n.count == 1 || n.count == 2, "oxygen \(i) has \(n.count) bonds")
        }
    }
    let waters = (0..<36).filter { reaction.elements[$0] == .oxygen && neighbors($0, reaction.productBonds).count == 2 }
    expectEqual(waters.count, 6)
}

section("oxidation states")
test("glucose carbons: +1 (C1), −1 (CH₂OH), 0 for the other four; average 0") {
    expectEqual(before[10], 1)
    expectEqual(before[11], -1)
    for c in [6, 7, 8, 9] { expectEqual(before[c], 0) }
    let total: Float = reaction.carbons.reduce(0) { $0 + before[$1] }
    expectEqual(total, 0)
}
test("oxygen in O₂ is 0; hydrogen is +1; oxygen in glucose is −2") {
    for o in reaction.o2Oxygens { expectEqual(before[o], 0) }
    for i in 0..<36 where reaction.elements[i] == .hydrogen { expectEqual(before[i], 1) }
    for i in 0..<6 { expectEqual(before[i], -2) }
}
test("after: every carbon +4, every oxygen −2, every hydrogen +1") {
    for i in 0..<36 {
        switch reaction.elements[i] {
        case .carbon: expectEqual(after[i], 4)
        case .oxygen: expectEqual(after[i], -2)
        case .hydrogen: expectEqual(after[i], 1)
        }
    }
}
test("24 electrons move from carbon to oxygen, and all molecules are neutral") {
    expectEqual(electronsTransferred(reaction), 24)
    let oxygenGain: Float = reaction.o2Oxygens.reduce(0) { $0 + before[$1] - after[$1] }
    expectEqual(oxygenGain, 24)
    expectEqual(before.reduce(0, +), 0)
    expectEqual(after.reduce(0, +), 0)
}

section("geometry")
test("O₂ bonds are 1.21 Å") {
    for j in 0..<6 {
        let a = reaction.o2Oxygens[2 * j], b = reaction.o2Oxygens[2 * j + 1]
        expect(near(distance(reactants, a, b), 1.21, within: 1e-4))
    }
}
test("each CO₂ is linear with C=O 1.163 Å") {
    for c in reaction.carbons {
        let o = neighbors(c, reaction.productBonds)
        expect(near(distance(products, c, o[0]), 1.163, within: 1e-3))
        expect(near(distance(products, c, o[1]), 1.163, within: 1e-3))
        expect(near(angle(products, o[0], c, o[1]), 180, within: 0.05), "O=C=O angle \(angle(products, o[0], c, o[1]))")
    }
}
test("each H₂O has O–H 0.9572 Å and a 104.52° angle") {
    for o in 0..<36 where reaction.elements[o] == .oxygen && neighbors(o, reaction.productBonds).count == 2 {
        let h = neighbors(o, reaction.productBonds)
        expect(near(distance(products, o, h[0]), 0.9572, within: 1e-3))
        expect(near(distance(products, o, h[1]), 0.9572, within: 1e-3))
        expect(near(angle(products, h[0], o, h[1]), 104.52, within: 0.05))
    }
}
test("molecules don't overlap, before or after") {
    func closestBetweenMolecules(_ s: MoleculeState, _ bonds: [Bond]) -> Float {
        // Label each atom with its molecule by following bonds.
        var molecule = [Int](repeating: -1, count: 36)
        var next = 0
        for start in 0..<36 where molecule[start] < 0 {
            var stack = [start]
            molecule[start] = next
            while let a = stack.popLast() {
                for b in neighbors(a, bonds) where molecule[b] < 0 { molecule[b] = next; stack.append(b) }
            }
            next += 1
        }
        var closest: Float = 1e9
        for i in 0..<36 { for j in (i + 1)..<36 where molecule[i] != molecule[j] {
            closest = min(closest, distance(s, i, j))
        } }
        return closest
    }
    let r = closestBetweenMolecules(reactants, reaction.reactantBonds)
    let p = closestBetweenMolecules(products, reaction.productBonds)
    expect(r > 2.0, "reactant molecules too close: \(r) Å")
    expect(p > 2.0, "product molecules too close: \(p) Å")
}

section("animation")
test("old bonds are gone and new bonds are complete at the ends") {
    expectEqual(reactants.bonds.count, reaction.reactantBonds.count)
    expectEqual(products.bonds.count, reaction.productBonds.count)
    for b in products.bonds { expect(b.order >= 1, "a product bond is only partly formed") }
}
test("atoms start at the reactants and finish at the products") {
    for i in 0..<36 {
        expect(simd_distance(reactants.atoms[i].position, reaction.start[i]) < 1e-5)
        expect(simd_distance(products.atoms[i].position, reaction.end[i]) < 1e-5)
    }
}
test("glows appear only while atoms are moving, gold on carbon and cyan on O₂'s oxygen") {
    expect(reactants.atoms.allSatisfy { $0.glow == 0 } && products.atoms.allSatisfy { $0.glow == 0 })
    let mid = reactionState(reaction, progress: 0.5)
    expect(reaction.carbons.allSatisfy { mid.atoms[$0].glow > 0 })
    expect(reaction.o2Oxygens.allSatisfy { mid.atoms[$0].glow < 0 })
}
test("the timeline runs in order and lasts over 10 seconds") {
    let t = Timeline()
    expect(t.total > 10)
    expectEqual(t.stage(at: 1), "Glucose + 6 O₂")
    expectEqual(t.stage(at: t.calm + 1), "Carbon hands its electrons to oxygen")
    expectEqual(t.stage(at: t.total - 0.5), "6 CO₂ + 6 H₂O + energy")
}
test("signed labels read like chemistry: +4, 0, −1") {
    expectEqual(signedLabel(4), "+4")
    expectEqual(signedLabel(0), "0")
    expectEqual(signedLabel(-1), "−1")
}

section("ray tracer (GPU)")
func render(_ state: MoleculeState, camera: Camera, width: Int = 96, height: Int = 64) throws -> (Int, Int) -> [Int] {
    let device = try findDevice()
    let renderer = try MoleculeRenderer(device: device)
    let frame = device.makeBuffer(length: width * height * 4, options: .storageModeShared)!
    try renderer.render(state, camera: camera, into: frame, width: width, viewHeight: height)
    let px = frame.contents().bindMemory(to: UInt8.self, capacity: width * height * 4)
    return { x, y in (0..<3).map { Int(px[(y * width + x) * 4 + $0]) } }
}
let closeCamera = Camera.orbit(target: .zero, distance: 4, yaw: 0, pitch: 0, fov: 30)
test("the background is step 2's gradient, with only a faint dither") {
    let empty = MoleculeState(atoms: [], bonds: [])
    let pixel = try render(empty, camera: closeCamera)
    for x in [3, 40, 90] {
        let top = pixel(x, 0), bottom = pixel(x, 63)
        expect(abs(top[0] - 140) <= 3 && abs(top[1] - 200) <= 3 && abs(top[2] - 235) <= 3, "top \(top)")
        expect(abs(bottom[0] - 8) <= 3 && abs(bottom[1] - 40) <= 3 && abs(bottom[2] - 90) <= 3, "bottom \(bottom)")
    }
}
test("a glowing atom keeps its true color; the glow only shows behind it") {
    let plain = MoleculeState(atoms: [Atom(element: .carbon, position: .zero)], bonds: [])
    let glowing = MoleculeState(atoms: [Atom(element: .carbon, position: .zero, glow: 1)], bonds: [])
    let a = try render(plain, camera: closeCamera), b = try render(glowing, camera: closeCamera)
    let centerA = a(48, 32), centerB = b(48, 32)
    for i in 0..<3 { expect(abs(centerA[i] - centerB[i]) <= 3, "atom color changed: \(centerA) vs \(centerB)") }
    let besideA = a(66, 32), besideB = b(66, 32)
    expect(besideB[0] > besideA[0] + 10, "no glow beside the atom: \(besideA) vs \(besideB)")
}
test("the nearer atom hides the farther one") {
    let pair = MoleculeState(atoms: [Atom(element: .hydrogen, position: SIMD3(0, 0, 1)),
                                     Atom(element: .oxygen, position: SIMD3(0, 0, -1))], bonds: [])
    let pixel = try render(pair, camera: Camera.orbit(target: .zero, distance: 6, yaw: 0, pitch: 0, fov: 30))
    let center = pixel(48, 32)
    expect(center[1] > 150 && center[2] > 150, "the white H in front should win, got \(center)")
}
test("a bond along the view direction still draws (no zero-length side vector)") {
    let stick = MoleculeState(atoms: [Atom(element: .carbon, position: SIMD3(0, 0, -1)),
                                      Atom(element: .carbon, position: SIMD3(0, 0, 1))],
                              bonds: [Bond(a: 0, b: 1, order: 2)])
    let (_, cylinders) = sceneGeometry(stick)
    expect(cylinders.allSatisfy { !$0.aRadius.x.isNaN && !$0.aRadius.y.isNaN }, "NaN in bond geometry")
}

finish()
