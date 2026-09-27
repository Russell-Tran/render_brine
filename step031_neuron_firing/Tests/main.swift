// Tests for step 31: step 27's tests, which still apply to the same cell, and
// the impulse — where it starts, which way it goes, where it never goes, that
// it changes nothing but the shading, and that the loop closes.
//
// NEURON_MUTANT=twoAxons|taperingAxon breaks the cell (step 27's mutants);
// NEURON_MUTANT=startsInDendrite|backwards breaks the impulse. `make mutants`
// requires the suite to fail for each.

import CoreGraphics
import Foundation
import ImageIO
import Metal
import simd

let mutantName: String = ProcessInfo.processInfo.environment["NEURON_MUTANT"] ?? "none"
let mutant = Mutant(rawValue: mutantName) ?? .none
let impulseMutant = ImpulseMutant(rawValue: mutantName) ?? .none
let cell: Neuron = buildNeuron(mutant)
let impulse: Impulse = buildImpulse(cell, impulseMutant)
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

section("the same cell as step 27")

test("the cell's source is step 27's, byte for byte") {
    let mine = try? Data(contentsOf: URL(fileURLWithPath: "Sources/Neuron.swift"))
    let theirs = try? Data(contentsOf: URL(fileURLWithPath: "../step027_neuron/Sources/Neuron.swift"))
    expect(mine != nil && mine == theirs, "Sources/Neuron.swift differs from step 27's")
}

test("a frame at rest is step 27's committed still, pixel for pixel") {
    guard let dev = device, let r = try? NeuronRenderer(cell, impulse, width: 1920, height: 1080, on: dev),
          (try? r.render(time: 0, samples: 3)) != nil
    else { expect(false, "render failed"); return }
    drawScaleBar(r.image)
    guard let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: "../showcase/neuron.png") as CFURL, nil),
          let png = CGImageSourceCreateImageAtIndex(src, 0, nil), let space = CGColorSpace(name: CGColorSpace.sRGB)
    else { expect(false, "could not read ../showcase/neuron.png"); return }
    var theirs = [UInt8](repeating: 0, count: 1920 * 1080 * 4)
    let ok: Bool = theirs.withUnsafeMutableBytes { raw -> Bool in
        guard let ctx = CGContext(data: raw.baseAddress, width: 1920, height: 1080, bitsPerComponent: 8,
                                  bytesPerRow: 1920 * 4, space: space,
                                  bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { return false }
        ctx.draw(png, in: CGRect(x: 0, y: 0, width: 1920, height: 1080))
        return true
    }
    expect(ok, "could not decode the PNG")
    let mine = r.image.pixels.contents().assumingMemoryBound(to: UInt8.self)
    var off: Int = 0
    var worst: Int = 0
    for i in 0..<(1920 * 1080) {
        for c in 0..<3 {
            let d: Int = abs(Int(mine[i * 4 + c]) - Int(theirs[i * 4 + c]))
            worst = max(worst, d)
            if d > 1 { off += 1 }
        }
    }
    print("        \(off) channels off by more than 1, worst \(worst)")
    expectEqual(off, 0)
}

section("the impulse")

/// Points along the impulse's road: the trunk's axis from the initial segment
/// out, skipping the break's gap, then each terminal and its bouton. `s` is
/// the distance along the road as drawn; the break adds nothing to it.
/// `seg` is the piece of axon the point lies on.
struct RoadPoint { var p: SIMD3<Float>; var s: Float; var branch: Int; var seg: Int }
func road() -> [RoadPoint] {
    var out: [RoadPoint] = []
    let trunk: Neurite = cell.axon[1]
    let len: Float = simd_distance(trunk.a, trunk.b)
    let dir: SIMD3<Float> = (trunk.b - trunk.a) / len
    var s: Float = 0
    while s <= len {
        let p: SIMD3<Float> = trunk.a + dir * s
        if abs(simd_dot(p - cell.breakCentre, cell.breakNormal)) > breakHalfGap + 1 {
            out.append(RoadPoint(p: p, s: s, branch: -1, seg: 1))
        }
        s += 1
    }
    var branch: Int = 0
    for (i, t) in cell.axon.enumerated() where t.kind == .terminal {
        guard let bouton = cell.axon.firstIndex(where: { $0.parent == i }) else { continue }
        let b: Neurite = cell.axon[bouton]
        let tl: Float = simd_distance(t.a, t.b)
        var u: Float = 2
        while u <= tl {
            out.append(RoadPoint(p: t.a + (t.b - t.a) * (u / tl), s: len + u, branch: branch, seg: i))
            u += 1
        }
        out.append(RoadPoint(p: b.b, s: len + tl + simd_distance(b.a, b.b), branch: branch, seg: bouton))
        branch += 1
    }
    return out
}

/// Points the impulse must never light: along every dendrite, on the soma,
/// and in the hillock.
func offRoad() -> [SIMD3<Float>] {
    var out: [SIMD3<Float>] = []
    for s in cell.dendrites {
        for k in 1...10 { out.append(s.a + (s.b - s.a) * (Float(k) / 10)) }
    }
    var rng = Seeded(state: 31)
    for _ in 0..<400 {
        let d = SIMD3<Float>(rng.between(-1, 1), rng.between(-1, 1), rng.between(-1, 1))
        out.append(cell.soma + simd_normalize(d) * (somaDiameter / 2))
    }
    for h in cell.axon where h.kind == .hillock {
        for k in 0...8 { out.append(h.a + (h.b - h.a) * (Float(k) / 10)) }
    }
    return out
}

let times: [Float] = (0..<400).map { Float($0) * loopSeconds / 400 }
let roadPoints: [RoadPoint] = road()
let offPoints: [SIMD3<Float>] = offRoad()
let roadGlow: [[Float]]? = device.flatMap { try? probeGlow(cell, impulse, roadPoints.map { $0.p }, times: times, on: $0) }
let offGlow: [[Float]]? = device.flatMap { try? probeGlow(cell, impulse, offPoints, times: times, on: $0) }

test("the kernel's glow is Impulse.swift's, point for point") {
    guard let glow = roadGlow else { expect(false, "probe failed"); return }
    var worst: Float = 0
    for (k, t) in times.enumerated() {
        for (i, r) in roadPoints.enumerated() {
            // Skip the forks, where a point may belong to either piece.
            let piece: Neurite = cell.axon[r.seg]
            let along: Float = simd_distance(r.p, piece.a)
            if piece.kind != .bouton && (along < 6 || simd_distance(r.p, piece.b) < 6) { continue }
            let seg: Int = r.seg
            guard let a = impulse.axon[seg] else { continue }
            var g: Float = pulse(t - arrival(at: r.p, on: cell.axon[seg], a, cell))
            if cell.axon[seg].kind == .bouton { g *= boutonGlowGain }
            worst = max(worst, abs(g - glow[k][i]))
        }
    }
    print(String(format: "        worst difference %.2g", worst))
    expect(worst < 1e-3, "the kernel and the model disagree by \(worst)")
}

test("it starts at the axon's initial segment, just past the hillock") {
    guard let road = roadGlow, let off = offGlow else { expect(false, "probe failed"); return }
    // The first moment anything glows, and where.
    guard let k = times.indices.first(where: { road[$0].max()! > 0.05 || off[$0].max()! > 0.05 }) else {
        expect(false, "nothing ever glows"); return
    }
    let lit: [RoadPoint] = roadPoints.indices.filter { road[k][$0] > 0.05 }.map { roadPoints[$0] }
    let litOff: Int = off[k].filter { $0 > 0.05 }.count
    let neck: SIMD3<Float> = cell.axon[1].a
    print(String(format: "        first glow at t = %.2f s: %d road points, all within %.0f µm of the hillock's end, %.0f µm from the soma's centre",
                 times[k], lit.count, lit.map { $0.s }.max() ?? -1, simd_distance(neck, cell.soma)))
    expectEqual(litOff, 0)
    expect(!lit.isEmpty && lit.allSatisfy { $0.branch < 0 && $0.s <= 10 },
           "the first glow is not at the initial segment: \(lit.map { $0.s })")
    expect(abs(simd_distance(neck, cell.soma) - (somaDiameter / 2 + hillockLength)) < 1e-3)
}

test("it only ever moves outward, and reaches every bouton") {
    guard let road = roadGlow else { expect(false, "probe failed"); return }
    // Where the glow is brightest along the trunk, frame by frame.
    var last: Float = -1
    var backwards: Int = 0
    var seen: Int = 0
    for k in times.indices {
        var best: Float = 0.3
        var at: Float = -1
        for (i, r) in roadPoints.enumerated() where r.branch <= 0 && road[k][i] > best {
            best = road[k][i]
            at = r.s
        }
        guard at >= 0 else { continue }
        seen += 1
        if at < last - 1 { backwards += 1 }
        last = at
    }
    print("        brightest point followed through \(seen) moments; \(backwards) steps back")
    expect(seen > 50, "the glow was seen too rarely to follow")
    expectEqual(backwards, 0)
    // Each bouton lights, after the initial segment did.
    let peak = { (i: Int) -> (Float, Int) in
        var m: Float = 0, at: Int = 0
        for k in times.indices where road[k][i] > m { m = road[k][i]; at = k }
        return (m, at)
    }
    let start: Int = peak(0).1
    for i in roadPoints.indices where i + 1 == roadPoints.count || roadPoints[i + 1].branch != roadPoints[i].branch {
        guard roadPoints[i].branch >= 0 else { continue }
        let (m, at) = peak(i)
        expect(m > 0.9, "bouton \(roadPoints[i].branch) peaks at only \(m)")
        expect(at > start, "bouton \(roadPoints[i].branch) lights before the initial segment")
    }
}

test("it never runs into a dendrite, the soma or the hillock") {
    guard let off = offGlow else { expect(false, "probe failed"); return }
    var worst: Float = 0
    for row in off { worst = max(worst, row.max()!) }
    print("        brightest glow at \(offPoints.count) points off the axon, over the loop: \(worst)")
    expectEqual(worst, 0)
}

test("the loop closes: the end of the loop is its start, at rest, never rewound") {
    guard let dev = device else { expect(false, "no GPU"); return }
    let pts: [SIMD3<Float>] = roadPoints.map { $0.p }
    let shifted: [Float] = times.map { $0 + loopSeconds }
    guard let a = roadGlow, let b = try? probeGlow(cell, impulse, pts, times: shifted, on: dev) else {
        expect(false, "probe failed"); return
    }
    var worst: Float = 0
    for k in times.indices { for i in pts.indices { worst = max(worst, abs(a[k][i] - b[k][i])) } }
    expect(worst < 1e-4, "glow at t and t + loop differ by \(worst)")
    // The GIF's last frame and first are both the resting still.
    let frames: Int = 160
    expect(isResting(0, impulse) && isResting(loopSeconds * Float(frames - 1) / Float(frames), impulse),
           "the loop's seam is not at rest")
    // Every road point lights once a loop: one rise, not a flicker or a rewind.
    for i in pts.indices {
        var rises: Int = 0
        for k in times.indices where a[k][i] > 0.5 && a[(k + times.count - 1) % times.count][i] <= 0.5 { rises += 1 }
        if rises != 1 { expect(false, "road point \(i) lights \(rises) times a loop"); break }
    }
}

test("only the shading changes: same silhouette every frame, and only the axon changes colour") {
    guard let dev = device, let r = try? NeuronRenderer(cell, impulse, width: 480, height: 270, on: dev),
          (try? r.render(time: 0, samples: 1)) != nil
    else { expect(false, "render failed"); return }
    let n: Int = 480 * 270 * 4
    let rest = Array(UnsafeBufferPointer(start: r.image.pixels.contents().assumingMemoryBound(to: UInt8.self), count: n))
    let bg = UInt8((backgroundDisplay * 255).rounded())
    let pieces: [Neurite] = cell.axon.filter { $0.kind == .axon || $0.kind == .terminal || $0.kind == .bouton }
    func nearAxon(_ x: Int, _ y: Int) -> Bool {
        let umPerPx: Float = frameWidth / 480
        let q = SIMD2<Float>((Float(x) + 0.5) * umPerPx - frameWidth / 2, frameWidth * 9 / 32 - (Float(y) + 0.5) * umPerPx)
        for s in pieces {
            let a = SIMD2<Float>(s.a.x, s.a.y), b = SIMD2<Float>(s.b.x, s.b.y)
            let u: Float = min(max(simd_dot(q - a, b - a) / simd_dot(b - a, b - a), 0), 1)
            if simd_distance(q, a + (b - a) * u) <= max(s.ra, s.rb) + 2 * umPerPx { return true }
        }
        return false
    }
    var changed: Int = 0
    var stray: Int = 0
    var silhouette: Int = 0
    for t in [Float(0.8), 1.2, 1.7, 2.3, 2.9, 3.6, 4.2, 4.5, 5.0] {
        guard (try? r.render(time: t, samples: 1)) != nil else { expect(false, "render failed"); return }
        let now = r.image.pixels.contents().assumingMemoryBound(to: UInt8.self)
        for i in stride(from: 0, to: n, by: 4) {
            let wasBG: Bool = rest[i] == bg && rest[i + 1] == bg && rest[i + 2] == bg
            let isBG: Bool = now[i] == bg && now[i + 1] == bg && now[i + 2] == bg
            if wasBG != isBG { silhouette += 1 }
            if now[i] != rest[i] || now[i + 1] != rest[i + 1] || now[i + 2] != rest[i + 2] {
                changed += 1
                if !nearAxon((i / 4) % 480, (i / 4) / 480) { stray += 1 }
            }
        }
    }
    print("        \(changed) pixels changed over 9 frames; \(stray) off the axon; \(silhouette) silhouette changes")
    expect(changed > 500, "the impulse barely shows")
    expectEqual(stray, 0)
    expectEqual(silhouette, 0)
}

section("the numbers, against their sources")

test("velocity, spike and slow-down are what the caption says") {
    expect(abs(conductionVelocity - 70) < 0.01, "velocity \(conductionVelocity) m/s (Hursh 6 × 7 / 0.6)")
    expect(conductionVelocity >= 50 && conductionVelocity <= 100, "outside the 50–100 m/s quoted for alpha motor axons")
    expect(abs(slowdown - 700_000) < 1, "slow-down \(slowdown)")
    expect(captionLines()[0].contains("700,000×"), captionLines()[0])
    expect(abs(realBreakSeconds - 0.01429) < 1e-4, "break \(realBreakSeconds) s")
    // A true-width spike is longer than the whole picture; that is why the
    // glow is drawn short, and the comments say so.
    expect(realSpikeLength * 1_000_000 > frameWidth * 100, "spike length \(realSpikeLength) m")
    // The impulse is done, with rest to spare, well inside one loop.
    let w = activeWindow(impulse)
    expect(w.from > 0 && w.to < loopSeconds - 1, "active window \(w)")
}

finish()
