// Tests for step 6a. Chemistry tests check the model against textbook
// chemistry; particle tests check the dot bookkeeping; GPU tests check the
// Brownian motion, the atomic counts, the drawing, and the GIF.

import CoreGraphics
import Foundation
import ImageIO
import Metal

func near(_ a: Double, _ b: Double, within e: Double) -> Bool { abs(a - b) <= e }

section("constants")
test("hydration constant is about 1.7 × 10⁻³") {
    expect(near(Carbonate.hydrationConstant, 1.7e-3, within: 0.03e-3), "got \(Carbonate.hydrationConstant)")
}
test("carbonic acid's own pKa falls in the published 3.45–3.75 range") {
    let p = Carbonate.carbonicAcidPKa
    expect(p > 3.45 && p < 3.75, "got \(p)")
}
test("the constants agree: Ka(apparent) = Kh × Ka(H₂CO₃)") {
    let apparent: Double = Carbonate.hydrationConstant * Carbonate.carbonicAcidKa / 1000
    expect(near(-log10(apparent), 6.35, within: 1e-9), "got pKa \(-log10(apparent))")
}

section("buffer chemistry")
test("the starting water matches Henderson–Hasselbalch: pH 7.65") {
    let s = BufferState(bicarbonate: 24, co2: 1.2)
    let expected: Double = 6.35 + log10(24.0 / 1.2)
    expect(near(s.pH, expected, within: 1e-3), "pH \(s.pH), expected \(expected)")
    expect(near(s.bicarbonate, 24, within: 1e-9))
}
test("the starting water is at equilibrium and stays put") {
    var s = BufferState(bicarbonate: 24, co2: 1.2)
    expect(near(s.carbonicAcid, Carbonate.hydrationConstant * s.co2, within: 1e-12))
    let before = s.pH
    s.advance(by: 2)
    expect(near(s.pH, before, within: 1e-6), "drifted from \(before) to \(s.pH)")
}
test("charge balances and Ka holds, whatever the mix") {
    for (bi, co2, acid) in [(24.0, 1.2, 0.0), (24.0, 1.2, 10.0), (5.0, 3.0, 4.0), (1.0, 0.5, 20.0)] {
        var s = BufferState(bicarbonate: bi, co2: co2)
        s.addStrongAcid(acid)
        let charge: Double = s.hydrogen + s.sodium - s.bicarbonate - s.chloride
        expect(abs(charge) < 1e-9, "charge off by \(charge)")
        let ka: Double = s.hydrogen * s.bicarbonate / s.carbonicAcid
        expect(abs(ka / Carbonate.carbonicAcidKa - 1) < 1e-6, "Ka off: \(ka)")
    }
}
test("carbon is never created or destroyed") {
    var s = BufferState(bicarbonate: 24, co2: 1.2)
    let start = s.totalCarbon
    s.addStrongAcid(10)
    s.advance(by: 3)
    expect(near(s.totalCarbon, start, within: 1e-9), "\(start) → \(s.totalCarbon)")
}
test("10 mM acid: pH crashes at once, then recovers to Henderson–Hasselbalch") {
    var s = BufferState(bicarbonate: 24, co2: 1.2)
    s.addStrongAcid(10)
    let instant = s.pH
    expect(instant > 3.6 && instant < 3.9, "instant pH \(instant)")
    s.advance(by: 3)
    // Closed system: 10 mM bicarbonate turns into 10 mM CO₂.
    let expected: Double = 6.35 + log10(14.0 / 11.2)
    expect(near(s.pH, expected, within: 0.01), "final pH \(s.pH), expected \(expected)")
    expect(near(s.pH, s.hendersonHasselbalchPH, within: 1e-3))
}
test("the recovery takes about half a second without an enzyme") {
    var s = BufferState(bicarbonate: 24, co2: 1.2)
    s.addStrongAcid(10)
    s.advance(by: 0.1)
    expect(s.pH < 6.0, "recovered too fast: pH \(s.pH) after 0.1 s")
    s.advance(by: 0.9)
    expect(s.pH > 6.4, "recovered too slowly: pH \(s.pH) after 1 s")
}
test("plain water with 10 mM acid is pH 2") {
    expect(near(plainWaterPH(afterAcid: 10), 2, within: 1e-12))
}
test("concentration labels pick sensible units") {
    expectEqual(concentrationLabel(24), "24.0 mM")
    expectEqual(concentrationLabel(0.183), "183 µM")
    expectEqual(concentrationLabel(0.0000224), "22 nM")
}
test("stochastic rounding is right on average") {
    var rng = SplitMix64(seed: 1)
    var total = 0
    for _ in 0..<20000 { total += stochasticRound(2.3, using: &rng) }
    let mean = Double(total) / 20000
    expect(near(mean, 2.3, within: 0.02), "mean \(mean)")
}

section("dot bookkeeping")
test("the box ends up with exactly the dots the chemistry asks for") {
    let box = try MoleculeBox(device: try findDevice(), capacity: 3000)
    var rng = SplitMix64(seed: 7)
    var s = BufferState(bicarbonate: 24, co2: 1.2)
    let carbonDots = Int(dots(s.totalCarbon).rounded())
    let sodiumDots = Int(dots(s.sodium).rounded())
    for step in 0..<40 {
        if step == 5 { s.addStrongAcid(10) } else { s.advance(by: 0.02) }
        let target = targetCounts(s, carbonDots: carbonDots, sodiumDots: sodiumDots, using: &rng)
        try applyTargets(target, to: box, using: &rng)
        expectEqual(box.tally(), target)
        let carbon: Int = target[0] + target[1] + target[2]
        expectEqual(carbon, carbonDots)
    }
}

section("GPU (this Mac)")
func freshBox(_ species: Species, count: Int, at position: SIMD2<Float>) throws -> MoleculeBox {
    let box = try MoleculeBox(device: try findDevice(), capacity: count)
    var rng = SplitMix64(seed: 3)
    try box.add(species, count: count, using: &rng)
    for i in 0..<count { box.slots[i].position = position; box.slots[i].flash = 0 }
    return box
}
func meanSquaredDisplacement(_ box: MoleculeBox, from start: SIMD2<Float>) -> Double {
    var sum = 0.0
    for i in 0..<box.used {
        let d = box.slots[i].position - start
        sum += Double(d.x * d.x + d.y * d.y)
    }
    return sum / Double(box.used)
}
test("Brownian steps keep every molecule inside the box") {
    let box = try freshBox(.hydrogen, count: 2000, at: SIMD2(0.02, 0.98))
    for step in 0..<50 { try box.move(seed: UInt32(step + 1), baseStep: 0.01) }
    let inside = (0..<box.used).allSatisfy { i in
        let p = box.slots[i].position
        return p.x >= 0 && p.x <= 1 && p.y >= 0 && p.y <= 1
    }
    expect(inside, "a molecule left the box")
}
test("spread grows like a random walk: mean squared distance = 2 σ² × steps") {
    let start = SIMD2<Float>(0.5, 0.5)
    let box = try freshBox(.bicarbonate, count: 20000, at: start)
    let base: Float = 0.002
    let steps = 50
    for step in 0..<steps { try box.move(seed: UInt32(step + 1), baseStep: base) }
    let sigma: Double = Double(base) * Double(Species.bicarbonate.diffusion).squareRoot()
    let expected: Double = 2 * sigma * sigma * Double(steps)
    let msd = meanSquaredDisplacement(box, from: start)
    expect(abs(msd / expected - 1) < 0.05, "MSD \(msd), expected \(expected)")
}
test("H⁺ spreads about 7.9× faster than HCO₃⁻, as their diffusion coefficients say") {
    let start = SIMD2<Float>(0.5, 0.5)
    let fast = try freshBox(.hydrogen, count: 20000, at: start)
    let slow = try freshBox(.bicarbonate, count: 20000, at: start)
    for step in 0..<30 {
        try fast.move(seed: UInt32(step + 1), baseStep: 0.001)
        try slow.move(seed: UInt32(step + 1), baseStep: 0.001)
    }
    let ratio: Double = meanSquaredDisplacement(fast, from: start) / meanSquaredDisplacement(slow, from: start)
    let expected: Double = Double(Species.hydrogen.diffusion / Species.bicarbonate.diffusion)
    expect(abs(ratio / expected - 1) < 0.08, "ratio \(ratio), expected \(expected)")
}
test("the GPU's atomic counts match the CPU's tally") {
    let box = try MoleculeBox(device: try findDevice(), capacity: 3000)
    var rng = SplitMix64(seed: 11)
    try box.add(.co2, count: 123, using: &rng)
    try box.add(.bicarbonate, count: 987, using: &rng)
    try box.add(.hydrogen, count: 5, using: &rng)
    box.remove(.bicarbonate, count: 87, using: &rng)
    expectEqual(try box.gpuCounts(), box.tally())
    expectEqual(box.tally(), [123, 0, 900, 5, 0, 0])
}
test("a molecule is drawn where it is, in its color; empty water stays water") {
    let layout = FrameLayout(boxPixels: 64, panelWidth: 16)
    let device = try findDevice()
    let frame = device.makeBuffer(length: layout.width * layout.height * 4, options: .storageModeShared)!
    let box = try freshBox(.bicarbonate, count: 1, at: SIMD2(0.5, 0.5))
    try box.draw(into: frame, frameWidth: layout.width, boxPixels: layout.boxPixels)
    let px = frame.contents().bindMemory(to: UInt8.self, capacity: layout.width * layout.height * 4)
    func rgb(_ x: Int, _ y: Int) -> [Int] { (0..<3).map { Int(px[(y * layout.width + x) * 4 + $0]) } }
    let center = rgb(32, 32)
    expect(center[2] > center[0] && center[2] > 150, "center should be bicarbonate blue, got \(center)")
    let corner = rgb(2, 2)
    expect(corner[0] < 30 && corner[2] < 70, "corner should be dark water, got \(corner)")
    var rng = SplitMix64(seed: 99)
    box.remove(.bicarbonate, count: 1, using: &rng)
    try box.draw(into: frame, frameWidth: layout.width, boxPixels: layout.boxPixels)
    let gone = rgb(32, 32)
    expect(gone[2] < 70, "a removed molecule was still drawn: \(gone)")
}

section("output")
test("the GIF writer writes every frame") {
    let layout = FrameLayout(boxPixels: 32, panelWidth: 8)
    let device = try findDevice()
    let frame = device.makeBuffer(length: layout.width * layout.height * 4, options: .storageModeShared)!
    let url = FileManager.default.temporaryDirectory.appendingPathComponent("buffer_tests-\(getpid()).gif")
    defer { try? FileManager.default.removeItem(at: url) }
    let gif = try GIFWriter(url: url, frameCount: 3, delay: 0.1)
    for _ in 0..<3 { gif.add(frame, layout: layout) }
    try gif.finish()
    let source = CGImageSourceCreateWithURL(url as CFURL, nil)
    expectEqual(source.map { CGImageSourceGetCount($0) }, 3)
}
test("the pH gauge maps 2 → left end, 8 → right end, and clamps") {
    expectEqual(gaugeFraction(2), 0)
    expectEqual(gaugeFraction(8), 1)
    expectEqual(gaugeFraction(5), 0.5)
    expectEqual(gaugeFraction(1), 0)
    expectEqual(gaugeFraction(9), 1)
}

finish()
