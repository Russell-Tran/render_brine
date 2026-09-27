// Tests for step 22. The brush is checked where a brush in a picture goes
// wrong: on the wrong tooth, floating a hair off it, buried in it, or with its
// head through the gum. Contact is checked on step 20's own distance function
// and on the tufts exactly as the kernel draws them. Then, as step 20 checked
// its own, that the scene's distance is still a distance with the brush in it,
// and that the picture shows what it should and nothing else changed.
//
// BRUSH_MUTANT=lift|push breaks the brush on purpose; `make mutants` requires
// the suite to fail for each.

import Foundation
import Metal
import simd

let mutant: BrushMutant = {
    switch ProcessInfo.processInfo.environment["BRUSH_MUTANT"] {
    case "lift": return .lift
    case "push": return .push
    default: return .none
    }
}()

let device: MTLDevice? = try? findDevice()
let probe: SceneProbe? = device.flatMap { try? SceneProbe(device: $0) }
let brush: Brush? = probe.flatMap { try? pressedBrush(probe: $0, mutant: mutant) }
let placed: [PlacedTooth] = placeTeeth()
let tooth: PlacedTooth = placed[brushedToothIndex]

/// How close is touching, and how far in is through: 0.02 mm, a tenth of one
/// filament's width.
let tolerance: Float = 0.02

section("which tooth")

test("index 3 is #21, the patient's left first premolar, square to step 20's camera mid-frame") {
    expect(tooth.spec.kind == .firstPremolar, "index 3 is a \(tooth.spec.name)")
    // +x is the patient's right (Anatomy.swift), so −x is the left.
    expect(tooth.side < 0 && tooth.centre.x < 0, "index 3 is on the patient's right: \(tooth.centre)")
    let c: Camera = stillCamera
    let p = SIMD3<Float>(tooth.centre.x, -4, tooth.centre.y)
    let d: SIMD3<Float> = p - c.position
    let depth: Float = simd_dot(d, c.forward)
    let sx: Float = simd_dot(d, c.right) / depth / tanHalfFOV / (1920.0 / 1080.0)
    let sy: Float = simd_dot(d, c.up) / depth / tanHalfFOV
    let px: Float = (sx + 1) / 2 * 1920
    let py: Float = (1 - sy) / 2 * 1080
    print(String(format: "        #21 mid-crown lands at pixel (%.0f, %.0f) of 1920 × 1080", px, py))
    expect(px > 1920 * 0.4 && px < 1920 * 0.6, "not mid-frame across: \(px)")
    let toCamera: SIMD2<Float> = simd_normalize(SIMD2<Float>(c.position.x, c.position.z) - tooth.centre)
    expect(simd_dot(toCamera, tooth.outward) > 0.99, "its buccal face does not face the lens")
}

section("every tuft touches, none goes through")

test("every tuft's tip touches the tooth or gum: grown by 0.02 mm, its tip is in the surface") {
    guard let b = brush, let pr = probe else { expect(false, "no brush"); return }
    var worst: Float = 0
    var onEnamel: Int = 0
    for (i, t) in b.tufts.enumerated() {
        // The exact tip, and the tip grown by the tolerance. The sign of the
        // distance is exact, so "grown tip reaches inside" means the real tip
        // is within 0.02 mm of the surface, whatever the distance's scale.
        guard let exact = try? pr(Array(tuftSurface(t, grow: 0)[0..<tipPointCount])),
              let grown = try? pr.distances(Array(tuftSurface(t, grow: tolerance)[0..<tipPointCount]))
        else { expect(false, "probe failed"); return }
        let nearest: SIMD2<Float> = exact.min { $0.x < $1.x }!
        worst = max(worst, nearest.x)
        if nearest.y == 1 { onEnamel += 1 }
        expect(grown.min()! < 0, "tuft \(i) floats: its tip is \(nearest.x) mm (by the distance) off the surface")
    }
    print(String(format: "        %d tufts; the largest tip-to-surface distance is %.5f mm; %d land on enamel, %d on gum",
                 b.tufts.count, worst, onEnamel, b.tufts.count - onEnamel))
}

test("no part of any tuft is inside the tooth or gum by more than 0.02 mm") {
    guard let b = brush, let pr = probe else { expect(false, "no brush"); return }
    // The whole tuft shrunk by the tolerance — tip ball and body, filled —
    // must be entirely outside. Again only the sign is used.
    var pts: [SIMD3<Float>] = []
    var owner: [Int] = []
    for (i, t) in b.tufts.enumerated() {
        let p: [SIMD3<Float>] = tuftInterior(t, shrink: tolerance)
        pts += p
        owner += Array(repeating: i, count: p.count)
    }
    guard let d = try? pr.distances(pts) else { expect(false, "probe failed"); return }
    var through: Set<Int> = []
    var deepest: Float = 0
    for k in 0..<d.count where d[k] < 0 {
        through.insert(owner[k])
        deepest = min(deepest, d[k])
    }
    print(String(format: "        %d points filling the tufts, shrunk by %.2f mm, checked; %d tufts go through",
                 pts.count, tolerance, through.count))
    expect(through.isEmpty, "tufts \(through.sorted()) pass into the tooth or gum, up to \(-deepest) mm")
}

test("the kernel draws those tufts: its tuft distance is zero at every tip and minus the radius at its centre") {
    guard let b = brush, let dev = device else { expect(false, "no brush"); return }
    let pts: [SIMD3<Float>] = b.tufts.map { $0.apex } + b.tufts.map { $0.tipCentre }
    guard let r = try? probeBrush(pts, brush: b, on: dev) else { expect(false, "probe failed"); return }
    let n: Int = b.tufts.count
    for i in 0..<n {
        expect(abs(r[i].y) < 1e-3, "tuft \(i): the kernel's tuft distance at the tip is \(r[i].y)")
        expect(abs(r[n + i].y + b.tufts[i].tipRadius) < 1e-3, "tuft \(i): at its tip's centre \(r[n + i].y)")
    }
}

test("pressed, not stretched: no tuft is longer than a new one, and the ones pressed hardest splay") {
    guard let b = brush else { expect(false, "no brush"); return }
    let lengths: [Float] = b.tufts.map { $0.length }
    let longest: Float = lengths.max()!
    let hardest: Tuft = b.tufts.max { $0.compression < $1.compression }!
    print(String(format: "        tufts %.2f–%.2f mm against a rest length of %.1f; the hardest pressed is at %.0f%%, tip %.2f mm across against %.2f",
                 lengths.min()!, longest, restLength, (1 - hardest.compression) * 100,
                 2 * hardest.tipRadius, 2 * hardest.rootRadius))
    expect(longest <= restLength + 1e-3, "a tuft is \(longest) mm, longer than new")
    expect(longest > restLength - 0.05, "no tuft is at its rest length, so the head is pressed further than it need be")
    for t in b.tufts {
        expect((t.tipRadius > t.rootRadius) == (t.compression > 0), "splay does not follow compression")
    }
}

section("the head, and where it is")

test("the head and neck touch neither tooth nor gum") {
    guard let b = brush, let dev = device, let pr = probe else { expect(false, "no brush"); return }
    // A grid through the head and the part of the neck in frame, every
    // 0.2 mm: wherever the kernel's head-and-neck distance says inside, step
    // 20's scene must say outside.
    let p: BrushPose = b.pose
    var grid: [SIMD3<Float>] = []
    let h: Float = 0.2
    var a: Float = -7.5
    while a <= 40 {
        var c: Float = -7.5
        while c <= 7.5 {
            var z: Float = -6.0
            while z <= 0.4 {
                grid.append(p.face + p.along * a + p.across * c + p.axis * z)
                z += h
            }
            c += h
        }
        a += h
    }
    guard let head = try? probeBrush(grid, brush: b, on: dev) else { expect(false, "probe failed"); return }
    let inside: [SIMD3<Float>] = zip(grid, head).filter { $0.1.x < 0 }.map { $0.0 }
    guard let scene = try? pr.distances(inside) else { expect(false, "probe failed"); return }
    let clearance: Float = scene.min() ?? -1
    print(String(format: "        %d grid points inside the head and neck; the nearest tooth or gum is at least %.2f mm away",
                 inside.count, clearance))
    expect(inside.count > 10_000, "only \(inside.count) points inside the head")
    expect(clearance > 0, "the head or neck is inside the tooth or gum")
}

test("it is on #21's buccal face at the gum margin") {
    guard let b = brush, let pr = probe else { expect(false, "no brush"); return }
    // The central tuft, which points along the head's axis.
    let centre: Tuft = b.tufts[0]
    let rel: SIMD2<Float> = SIMD2<Float>(centre.apex.x, centre.apex.z) - tooth.centre
    let mesiodistal: Float = simd_dot(rel, tooth.tangent)
    let buccal: Float = simd_dot(rel, tooth.outward)
    let margin: Float = -tooth.spec.crownHeight + gumMarginAboveCEJ
    print(String(format: "        the central tuft lands %.2f mm from #21's midline, %.2f mm buccal of its centre, %.2f mm from the gum margin's height",
                 mesiodistal, buccal, centre.apex.y - margin))
    expect(abs(mesiodistal) < 0.5, "off #21's middle by \(mesiodistal) mm")
    expect(buccal > tooth.spec.cervicalDepth / 2 - 0.5, "not on the buccal face: \(buccal) mm out")
    expect(abs(centre.apex.y - margin) < 0.5, "\(centre.apex.y - margin) mm from the margin")
    // The field straddles the margin: some tufts on enamel, some on gum.
    var mats: Set<Int> = []
    for t in b.tufts {
        guard let d = try? pr(Array(tuftSurface(t, grow: tolerance)[0..<tipPointCount])) else { continue }
        mats.insert(Int(d.min { $0.x < $1.x }!.y))
    }
    expect(mats == [1, 2], "tufts land on \(mats.sorted()), not on both enamel and gum")
}

section("the distance function is still a distance")

test("round the brush, outside every surface, it never claims more room than the ray allows") {
    guard let b = brush, let dev = device else { expect(false, "no brush"); return }
    // Step 20's test, pointed at the brush: pairs of nearby points through a
    // box round the head, the tufts and the neck in frame. The brush is added
    // with a plain minimum and skipped outside its bounds only where the
    // bound proves it cannot be nearest — step 20 found that a skip resting on
    // an under-reported distance jumped 20×, and this is where it would show.
    var rng = SystemRandomNumberGenerator()
    var a: [SIMD3<Float>] = []
    var c: [SIMD3<Float>] = []
    let p: BrushPose = b.pose
    for _ in 0..<150_000 {
        let q: SIMD3<Float> = p.face + p.along * Float.random(in: -12...45, using: &rng)
            + p.across * Float.random(in: -12...12, using: &rng) + p.axis * Float.random(in: -10...12, using: &rng)
        let dir: SIMD3<Float> = simd_normalize(SIMD3<Float>(Float.random(in: -1...1, using: &rng),
                                                            Float.random(in: -1...1, using: &rng),
                                                            Float.random(in: -1...1, using: &rng)))
        a.append(q)
        c.append(q + dir * 0.05)
    }
    let extra: SceneExtra = brushExtra(b)
    guard let da = try? probeScene(a, extra: extra, on: dev), let dc = try? probeScene(c, extra: extra, on: dev) else {
        expect(false, "probe failed"); return
    }
    var worst: [Float] = [0, 0, 0, 0, 0, 0, 0]
    var counts: [Int] = [0, 0, 0, 0, 0, 0, 0]
    for i in 0..<a.count where da[i].y == dc[i].y && da[i].x > 0 && dc[i].x > 0 {
        let m: Int = Int(da[i].y)
        worst[m] = max(worst[m], abs(da[i].x - dc[i].x) / 0.05)
        counts[m] += 1
    }
    print(String(format: "        worst over-report: enamel %.2f, gum %.2f, head %.2f (%d pairs), tufts %.2f (%d pairs); the ray allows %.2f",
                 worst[1], worst[2], worst[5], counts[5], worst[6], counts[6], 1 / stepScale))
    expect(counts[5] > 1000 && counts[6] > 1000, "too few pairs near the brush")
    for m in [1, 2, 5, 6] { expect(worst[m] * stepScale <= 1.0, "material \(m) oversteps: \(worst[m])") }
}

section("the picture")

let render: MouthImage? = {
    guard let dev = device, let b = brush else { return nil }
    return try? renderMouth(width: 640, height: 360, samples: 2, extra: brushExtra(b), on: dev).image
}()

func lab(_ img: MouthImage, _ x: Int, _ y: Int) -> SIMD3<Double> {
    let p: SIMD4<UInt8> = img.rgba(x, y)
    return linearSRGBToLab(SIMD3<Float>(srgbByteToLinear(p.x), srgbByteToLinear(p.y), srgbByteToLinear(p.z)))
}

test("the brush is in the picture: a white head, and tufts both light and dark blue") {
    guard let img = render else { expect(false, "render failed"); return }
    var counts: [Int] = [0, 0, 0, 0, 0, 0, 0]
    var head = SIMD3<Double>(0, 0, 0)
    var light: Int = 0
    var dark: Int = 0
    for y in 0..<img.height {
        for x in 0..<img.width {
            let m: Int = Int(img.seen(x, y).x)
            counts[m] += 1
            if m == 5 { head += lab(img, x, y) }
            if m == 6 {
                let c: SIMD3<Double> = lab(img, x, y)
                if c.z < -15 && c.x > 62 { light += 1 }
                if c.z < -30 && c.x < 50 { dark += 1 }
            }
        }
    }
    let total: Double = Double(img.width * img.height)
    head /= Double(max(counts[5], 1))
    let share: [Double] = counts.map { Double($0) / total * 100 }
    print(String(format: "        head %.1f%%, tufts %.1f%%, enamel %.1f%%, gum %.1f%%",
                 share[5], share[6], share[1], share[2]))
    print(String(format: "        head L* %.1f a* %.1f b* %.1f; light-blue pixels %d, dark-blue %d",
                 head.x, head.y, head.z, light, dark))
    expect(counts[5] > 2000 && counts[6] > 1000, "the brush barely shows")
    expect(Double(counts[1]) / total > 0.15, "the teeth are hidden")
    expect(head.x > 70 && abs(head.y) < 5 && abs(head.z) < 8, "the head is not white")
    expect(light > 100 && dark > 100, "the rim's two blues do not both show")
}

test("far from the brush, the picture is step 20's, pixel for pixel") {
    guard let dev = device, let img = render,
          let plain = try? renderMouth(width: 640, height: 360, samples: 2, on: dev).image
    else { expect(false, "render failed"); return }
    // The top third of the frame — the far side of the arch — is nowhere near
    // the brush, its shadow or its occlusion.
    var differ: Int = 0
    for y in 0..<(img.height / 3) {
        for x in 0..<img.width where img.rgba(x, y) != plain.rgba(x, y) { differ += 1 }
    }
    expectEqual(differ, 0)
}

finish()
