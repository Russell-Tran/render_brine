// Tests for step 65: the camera never moves, the wear only ever deepens, the
// years are the drawn depth over the measured rate, the change is big enough
// to see every second, and each label points at what it names when it is up.
// Step 64's still is tested in its own folder; what this step changes is time.
//
// BRUX_MUTANT=unwear|invisible|cameraMoves|noDentin breaks it on purpose;
// `make mutants` requires the suite to fail for each.

import Foundation
import Metal
import simd

let mutant: PlanMutant = PlanMutant(rawValue: ProcessInfo.processInfo.environment["BRUX_MUTANT"] ?? "") ?? .none
let kernelMutant: Mutant = mutant == .noDentin ? .noDentin : .none

final class MTLDeviceBox {
    let device = try? findDevice()
}
let device: MTLDeviceBox = MTLDeviceBox()

let allFrames: [FramePlan] = (0..<frameCount).map { plan($0, mutant: mutant) }

section("the timeline")

test("16.5 s at 10 frames a second: 1 s healthy, 12 s of wear, 2 s held, 1.5 s dissolve") {
    expectEqual(frameCount, 165)
    expectEqual(healthyFrames + wearFrames + wornFrames + fadeFrames, frameCount)
    expect(abs(loopSeconds - 16.5) < 1e-4)
    expectEqual(frameDelayCentiseconds * framesPerSecond, 100)
}

test("the camera is step 20's, exactly, in every frame") {
    for (f, p) in allFrames.enumerated() {
        expect(p.camera.position == stillCamera.position && p.camera.target == stillCamera.target,
               "frame \(f): camera at \(p.camera.position) looking at \(p.camera.target)")
    }
    expect(stillCamera.position == SIMD3<Float>(-50, 17, 1) && stillCamera.target == SIMD3<Float>(-9, -4.5, 14),
           "not step 20's camera")
}

test("forward only: the wear never gets shallower, from year 0 to the last frame") {
    for f in 1..<frameCount {
        expect(allFrames[f].depth >= allFrames[f - 1].depth, "frame \(f): \(allFrames[f - 1].depth) → \(allFrames[f].depth) mm")
        expect(allFrames[f].years >= allFrames[f - 1].years, "frame \(f): the clock ran back")
    }
    expect(allFrames[0].depth == 0, "frame 0 should be step 20's unworn teeth")
    expect(abs(allFrames[healthyFrames + wearFrames - 1].depth - finalWearDepth) < 1e-5, "the wear should end at step 64's")
}

test("the loop closes by dissolving the picture to year 0, not by un-wearing the teeth") {
    let fadeStart: Int = healthyFrames + wearFrames + wornFrames
    for f in 0..<fadeStart { expect(allFrames[f].dissolve == 0, "frame \(f) dissolves early") }
    for f in fadeStart..<frameCount {
        expect(allFrames[f].dissolve > 0 && allFrames[f].depth == finalWearDepth, "frame \(f)")
        if f > fadeStart { expect(allFrames[f].dissolve >= allFrames[f - 1].dissolve) }
    }
    expect(abs(allFrames[frameCount - 1].dissolve - 1) < 1e-6, "the last frame should be all frame 0")
}

test("the caption's years are the drawn depth over the measured rate: 50 years at the end") {
    for (f, p) in allFrames.enumerated() {
        expect(abs(p.years - p.depth / bruxistWearRate) < 1e-3, "frame \(f): \(p.years) years for \(p.depth) mm")
    }
    expect(abs(finalWearYears - 50) < 0.01)
    // And the tint's band is its years at the same rate.
    expect(abs(glowBand - glowYears * bruxistWearRate) < 1e-6)
}

section("the teeth, frame by frame")

test("no tooth grows at any frame: every surveyed point only ever comes down") {
    guard let dev = device.device else { expect(false, "no GPU"); return }
    var previous: [ToothSurvey]? = nil
    for f in stride(from: 0, to: frameCount, by: 10) + [frameCount - 1] {
        guard let now = try? surveyTeeth(depth: allFrames[f].depth, step: 0.3, on: dev) else { expect(false, "survey"); return }
        if let before = previous {
            for (a, b) in zip(before, now) {
                var heights: [SIMD2<Float>: Float] = [:]
                for p in a.points { heights[p.uv] = p.height }
                // 0.5 µm: the drop's own precision.
                for p in b.points { if let h = heights[p.uv], p.height > h + 5e-4 { expect(false, "\(a.spec.name) grew \(p.height - h) mm by frame \(f)") } }
            }
        }
        previous = now
    }
}

test("the distance stays honest part-way through the wear, as whole and at the end") {
    guard let dev = device.device else { expect(false, "no GPU"); return }
    var rng = SystemRandomNumberGenerator()
    var a: [SIMD3<Float>] = []
    var b: [SIMD3<Float>] = []
    for _ in 0..<60_000 {
        let p = SIMD3<Float>(Float.random(in: -26...26, using: &rng), Float.random(in: -12...4, using: &rng),
                             Float.random(in: -6...50, using: &rng))
        let dir: SIMD3<Float> = simd_normalize(SIMD3<Float>(Float.random(in: -1...1, using: &rng), Float.random(in: -1...1, using: &rng),
                                                            Float.random(in: -1...1, using: &rng)))
        a.append(p)
        b.append(p + dir * 0.05)
    }
    let depth: Float = allFrames[healthyFrames + wearFrames / 2].depth
    guard let da = try? probeScene(a, depth: depth, on: dev), let db = try? probeScene(b, depth: depth, on: dev) else {
        expect(false, "probe failed"); return
    }
    var worst: Float = 0
    for i in 0..<a.count where da[i].y == db[i].y && da[i].x > 0 && db[i].x > 0 && da[i].y == 1 {
        worst = max(worst, abs(da[i].x - db[i].x) / 0.05)
    }
    print(String(format: "        enamel at %.2f mm: worst over-report %.2f, the ray allows %.2f", depth, worst, 1 / stepScale))
    expect(worst * stepScale <= 1.0)
}

section("seeing it")

/// Teeth only, no marks, at a test size: what the eye gets of the wear itself.
func frameImage(_ f: Int, w: Int = 480, h: Int = 270, samples: Int = 1) -> MouthImage? {
    guard let dev = device.device else { return nil }
    let p: FramePlan = allFrames[f]
    return try? renderMouth(width: w, height: h, samples: samples, mutant: kernelMutant, camera: p.camera,
                            depth: p.depth, glow: SIMD2<Float>(glowBand, glowStrength), on: dev).image
}

func isTooth(_ s: SIMD4<Float>) -> Bool { s.x == 1 || s.x == 5 || s.x == 6 }

test("the wear can be seen: every second of it changes a clear share of the teeth's pixels") {
    // Wear is a small change in shape, so it must still read on the screen —
    // step 29's growth was invisible to the eye. One frame per second through
    // the wear; each second must visibly change (≥ 10 levels in some channel)
    // at least 1.5% of the pixels that show a tooth.
    var images: [MouthImage] = []
    for s in 0...Int(wearSeconds) {
        let f: Int = healthyFrames - 1 + s * framesPerSecond
        guard let img = frameImage(min(f, healthyFrames + wearFrames - 1)) else { expect(false, "render failed"); return }
        images.append(img)
    }
    var weakest: Double = 1
    var shares: [String] = []
    for s in 1..<images.count {
        let a: MouthImage = images[s - 1]
        let b: MouthImage = images[s]
        var teeth: Int = 0
        var changed: Int = 0
        for y in 0..<a.height {
            for x in 0..<a.width where isTooth(a.seen(x, y)) || isTooth(b.seen(x, y)) {
                teeth += 1
                let p: SIMD4<UInt8> = a.rgba(x, y)
                let q: SIMD4<UInt8> = b.rgba(x, y)
                let d: Int = max(abs(Int(p.x) - Int(q.x)), abs(Int(p.y) - Int(q.y)), abs(Int(p.z) - Int(q.z)))
                if d >= 10 { changed += 1 }
            }
        }
        let share: Double = Double(changed) / Double(max(teeth, 1))
        weakest = min(weakest, share)
        shares.append(String(format: "%.1f", share * 100))
        expect(share >= 0.015, String(format: "second %d changed only %.2f%% of the tooth pixels", s, share * 100))
    }
    print("        % of tooth pixels changed, second by second: " + shares.joined(separator: " "))
}

test("the newly-worn tint marks worn surface only, and nothing on the healthy teeth") {
    guard let dev = device.device else { expect(false, "no GPU"); return }
    let depth: Float = allFrames[healthyFrames + wearFrames / 2].depth
    guard let with = try? renderMouth(width: 480, height: 270, samples: 1, depth: depth,
                                      glow: SIMD2<Float>(glowBand, glowStrength), on: dev).image,
          let without = try? renderMouth(width: 480, height: 270, samples: 1, depth: depth, on: dev).image,
          let healthyWith = try? renderMouth(width: 480, height: 270, samples: 1, depth: 0,
                                             glow: SIMD2<Float>(glowBand, glowStrength), on: dev).image,
          let healthy = try? renderMouth(width: 480, height: 270, samples: 1, depth: 0, on: dev).image
    else { expect(false, "render failed"); return }
    var tinted: Int = 0
    var stray: Int = 0
    var healthyDiff: Int = 0
    for y in 0..<with.height {
        for x in 0..<with.width {
            if with.rgba(x, y) != without.rgba(x, y) {
                let s: SIMD4<Float> = with.seen(x, y)
                if s.x == 5 || s.x == 6 { tinted += 1 } else { stray += 1 }
            }
            if healthyWith.rgba(x, y) != healthy.rgba(x, y) { healthyDiff += 1 }
        }
    }
    print("        \(tinted) tinted pixels on worn surface, \(stray) elsewhere")
    expect(tinted > 200, "the tint is hardly there: \(tinted) px")
    // A pixel whose centre is off the facet can still catch a tinted sample at its edge.
    expect(stray < tinted / 10, "\(stray) tinted pixels off the worn surface")
    expect(healthyDiff == 0, "the tint changed \(healthyDiff) pixels of step 20's healthy teeth")
}

section("the labels")

let times: FeatureTimes? = {
    guard let dev = device.device else { return nil }
    return try? featureTimes(on: dev)
}()

test("labels go up in the order the wear reaches each feature, at years derived from the rate") {
    guard let t = times else { expect(false, "no times"); return }
    print(String(format: "        facets %.1f, incisor dentine %.1f, canine flat %.1f, premolar dentine %.1f years",
                 t.facets, t.incisorDentine, t.canineFlat, t.premolarDentine))
    expect(t.facets < t.incisorDentine && t.incisorDentine < t.premolarDentine)
    // The thin incisal enamel is worn through well before the thicker cusps'.
    expect(t.incisorDentine * bruxistWearRate < incisalEnamel + 0.05, "incisal dentine at \(t.incisorDentine * bruxistWearRate) mm")
    expect(t.premolarDentine * bruxistWearRate > incisalEnamel + 0.3)
    expect(t.premolarDentine < finalWearYears, "the premolar dentine label would never go up")
}

test("each label, once up, points at a pixel showing what it names") {
    guard let dev = device.device, let t = times,
          let aims = targetAims((try? surveyTeeth(depth: finalWearDepth, step: 0.1, on: dev)) ?? []) else {
        expect(false, "no survey"); return
    }
    for f in [healthyFrames + wearFrames / 2, healthyFrames + 3 * wearFrames / 4, healthyFrames + wearFrames - 1] {
        let p: FramePlan = allFrames[f]
        guard let s = try? surveyTeeth(depth: p.depth, step: 0.1, on: dev), let img = frameImage(f, w: 640, h: 360) else {
            expect(false, "render failed"); continue
        }
        let targets: FrameTargets = frameTargets(s, aims: aims)
        let checks: [(String, SIMD3<Float>?, Float, (SIMD4<Float>) -> Bool)] = [
            ("wear facet", targets.facet, t.facets, { $0.x == 6 }),
            ("incisor dentine", targets.incisorDentine, t.incisorDentine, { $0.x == 5 && $0.z == 0 }),
            ("premolar dentine", targets.premolarDentine, t.premolarDentine, { $0.x == 5 }),
            ("canine", targets.canine, t.canineFlat, { ($0.x == 5 || $0.x == 6) && $0.z == 1 }),
        ]
        for (name, q, appears, ok) in checks where p.years >= appears {
            guard let point = q else { expect(false, "frame \(f): \(name) is up but has no target"); continue }
            let px: SIMD2<Float> = p.camera.project(point, width: img.width, height: img.height)
            let x: Int = Int(px.x)
            let y: Int = Int(px.y)
            guard x >= 1, y >= 1, x < img.width - 1, y < img.height - 1 else { expect(false, "\(name) off the picture"); continue }
            var hit: Bool = false
            for dy in -1...1 { for dx in -1...1 where ok(img.seen(x + dx, y + dy)) { hit = true } }
            expect(hit, "frame \(f): \(name) points at \(img.seen(x, y))")
        }
    }
}

test("exposed dentine is drawn in dentine's colour, not enamel's, as it appears") {
    guard let img = frameImage(healthyFrames + wearFrames - 1, w: 640, h: 360, samples: 2) else { expect(false, "render"); return }
    var dent = SIMD3<Double>(0, 0, 0)
    var fac = SIMD3<Double>(0, 0, 0)
    var nd: Int = 0
    var nf: Int = 0
    for y in 0..<img.height {
        for x in 0..<img.width {
            let s: SIMD4<Float> = img.seen(x, y)
            guard s.x == 5 || s.x == 6 else { continue }
            let p: SIMD4<UInt8> = img.rgba(x, y)
            let lab: SIMD3<Double> = linearSRGBToLab(SIMD3<Float>(srgbByteToLinear(p.x), srgbByteToLinear(p.y), srgbByteToLinear(p.z)))
            if s.x == 5 { dent += lab; nd += 1 } else { fac += lab; nf += 1 }
        }
    }
    guard nd > 100, nf > 100 else { expect(false, "\(nd) dentine, \(nf) facet pixels"); return }
    dent /= Double(nd)
    fac /= Double(nf)
    print(String(format: "        dentine b*/L* %.3f, a* %.1f; facet enamel b*/L* %.3f, a* %.1f",
                 dent.z / dent.x, dent.y, fac.z / fac.x, fac.y))
    expect(dent.z / dent.x > 1.25 * fac.z / fac.x && dent.y > fac.y + 1.5)
}

finish()
