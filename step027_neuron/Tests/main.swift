// Tests for step 27: the cell's parts and sizes against the papers, the
// distance function against the definition of a distance, and the picture.
//
// NEURON_MUTANT=twoAxons|taperingAxon builds a wrong cell on purpose;
// `make mutants` requires the suite to fail for each.

import Foundation
import Metal
import simd

let mutant = Mutant(rawValue: ProcessInfo.processInfo.environment["NEURON_MUTANT"] ?? "none") ?? .none
let cell: Neuron = buildNeuron(mutant)
let device: MTLDevice? = try? findDevice()

section("the parts")

test("exactly one axon, from one hillock; a dozen dendrites") {
    expectEqual(cell.axon.filter { $0.kind == .hillock }.count, 1)
    expectEqual(cell.axon.filter { $0.kind == .axon }.count, 1)
    expectEqual(cell.dendrites.filter { $0.order == 1 }.count, primaryDendriteCount)
    expectEqual(cell.dendrites.count, primaryDendriteCount * segmentsPerTree)
    // Every terminal hangs off the one trunk, and every bouton off a terminal.
    for s in cell.axon where s.kind == .terminal {
        expect(s.parent.map { cell.axon[$0].kind } == .axon, "a terminal not on the trunk")
    }
    expectEqual(cell.axon.filter { $0.kind == .bouton }.count, terminalCount)
}

test("dendrites taper along every branch and at every fork; the axon keeps one width") {
    for (i, s) in cell.dendrites.enumerated() {
        expect(s.rb < s.ra, "dendrite segment \(i) does not narrow")
        if let p = s.parent {
            expect(s.ra < cell.dendrites[p].rb, "fork at \(p) does not narrow")
            expectEqual(s.order, cell.dendrites[p].order + 1)
        }
    }
    let tips: [Float] = cell.dendrites.filter { $0.order == branchOrders }.map { $0.rb * 2 }
    print(String(format: "        stems %.1f–%.1f µm, tips %.1f–%.1f µm",
                 stemDiameters.min()!, stemDiameters.max()!, tips.min()!, tips.max()!))
    for s in cell.axon where s.kind == .axon {
        expect(s.ra == axonDiameter / 2 && s.rb == axonDiameter / 2,
               "the axon runs \(s.ra * 2) → \(s.rb * 2) µm; it should stay \(axonDiameter)")
    }
}

section("the sizes, against their sources")

test("soma, stems and axon inside the cat measurements; nucleus from its volume") {
    expect(somaDiameter >= 32 && somaDiameter <= 69, "soma \(somaDiameter) µm (Zwaagstra & Kernell: 32–69)")
    expectEqual(primaryDendriteCount, 12)
    expectEqual(stemDiameters.count, primaryDendriteCount)
    for d in stemDiameters { expect(d >= 0.5 && d <= 19, "stem \(d) µm (Zwaagstra & Kernell: 0.5–19)") }
    expect(axonDiameter >= 4.6 && axonDiameter <= 9.0, "axon \(axonDiameter) µm (Cullheim & Kellerth: 4.6–9.0)")
    expect(abs(nucleusDiameter - 14.69) < 0.02, "nucleus \(nucleusDiameter) µm from 1,660 µm³")
    expect(abs(nucleolusDiameter - 3.66) < 0.02, "nucleolus \(nucleolusDiameter) µm from 25.6 µm³")
    // Rall's 3/2 rule with Kernell & Zwaagstra's 19% excess, and their 12% taper.
    expect(abs(daughterRatio - 0.707) < 0.002, "daughter ratio \(daughterRatio)")
    let excess: Float = 2 * pow(daughterRatio, 1.5)
    expect(abs(excess - rallExcess) < 1e-4, "Σd^1.5 / parent^1.5 = \(excess)")
    expect(abs(taperPerSegment - 0.88) < 1e-6)
    // Everything inside what holds it.
    let nucleolusReach: Float = simd_distance(nucleolusCentre(cell), nucleusCentre(cell)) + nucleolusDiameter / 2
    expect(nucleolusReach < nucleusDiameter / 2, "the nucleolus pokes out of the nucleus")
    expect(nucleusDiameter < somaDiameter / 2, "the nucleus is too big for the soma")
    // The break is earned: the real axon is thousands of somas long.
    expect(axonToSomaRatio > 1000, "axon/soma \(axonToSomaRatio)")
}

section("the distance function is a distance")

test("outside every surface, it never claims more room than there is, beyond what the ray allows") {
    guard let dev = device else { expect(false, "no GPU"); return }
    // Pairs of nearby points across the whole cell. A true distance changes by
    // at most the distance moved; the ray trusts only `stepScale` of each step.
    // Only pairs OUTSIDE count, since a ray only ever asks from outside.
    // The tree-skipping and the break's cut are what this is watching: a skip
    // that is not exact switches a tree on and off in open air and the
    // distance jumps across the switch.
    var a: [SIMD3<Float>] = []
    var b: [SIMD3<Float>] = []
    var rng = Seeded(state: 1)
    let h: Float = 0.2
    for _ in 0..<150_000 {
        let p = SIMD3<Float>(rng.between(-280, 280), rng.between(-160, 160), rng.between(-80, 80))
        let d = SIMD3<Float>(rng.between(-1, 1), rng.between(-1, 1), rng.between(-1, 1))
        a.append(p)
        b.append(p + simd_normalize(d) * h)
    }
    guard let da = try? probeScene(cell, a, on: dev), let db = try? probeScene(cell, b, on: dev) else {
        expect(false, "probe failed"); return
    }
    var worst: Float = 0
    var outside: Int = 0
    for i in 0..<a.count where da[i].x > 0 && db[i].x > 0 {
        worst = max(worst, abs(da[i].x - db[i].x) / h)
        outside += 1
    }
    print(String(format: "        worst over-report outside %.3f over %d pairs; the ray allows %.2f",
                 worst, outside, 1 / stepScale))
    expect(worst * stepScale <= 1.0, "oversteps: \(worst)")
}

section("the picture")

test("background at the corners, and the nucleolus shows dark through the soma") {
    guard let dev = device, let img = try? renderNeuron(cell, width: 480, height: 270, samples: 1, on: dev).image
    else { expect(false, "render failed"); return }
    let bg = Int((backgroundDisplay * 255).rounded())
    for (x, y) in [(0, 0), (479, 0), (0, 269), (479, 269)] {
        expect(abs(Int(img.rgba(x, y).x) - bg) <= 1, "corner (\(x), \(y)) is \(img.rgba(x, y))")
    }
    func luma(_ p: SIMD3<Float>) -> Float {
        let (x, y) = project(p, width: img.width, height: img.height)
        let c: SIMD4<UInt8> = img.rgba(x, y)
        return 0.2126 * Float(c.x) + 0.7152 * Float(c.y) + 0.0722 * Float(c.z)
    }
    let dark: Float = luma(nucleolusCentre(cell))
    // The soma just outside the nucleus, all round.
    var ring: Float = 0
    for k in 0..<8 {
        let t: Float = Float(k) * .pi / 4
        ring += luma(cell.soma + SIMD3<Float>(cos(t), sin(t), 0) * (nucleusDiameter / 2 + 4)) / 8
    }
    print(String(format: "        nucleolus luma %.0f, soma around the nucleus %.0f", dark, ring))
    expect(dark < ring * 0.8, "the nucleolus does not show: \(dark) vs \(ring)")
}

finish()
