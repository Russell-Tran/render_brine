// Tests for step 28. The motion is checked where an animated brush goes
// wrong: tufts that float off or sink into the tooth once the head turns, a
// head that swings further than a real one, a loop that jumps where it wraps,
// and a frame that is not the still it claims to animate. Contact is checked
// at every one of the loop's frames, on step 20's own distance function and on
// the tufts as the kernel draws them.
//
// OSC_MUTANT=frozen|overswing|open breaks it on purpose; `make mutants`
// requires the suite to fail for each.

import CoreGraphics
import Foundation
import ImageIO
import Metal
import simd

let device: MTLDevice? = try? findDevice()
let probe: SceneProbe? = device.flatMap { try? SceneProbe(device: $0) }
let oscillating: OscillatingBrush? = probe.flatMap { try? OscillatingBrush(probe: $0) }
let placed: [PlacedTooth] = placeTeeth()
let tooth: PlacedTooth = placed[brushedToothIndex]

/// Every frame of the loop, as it will be drawn.
let frames: [Brush] = {
    guard let o = oscillating else { return [] }
    return (0..<frameCount).compactMap { try? o.brush(frame: $0) }
}()

/// How close is touching, and how far in is through: 0.02 mm, a tenth of one
/// filament's width, as in step 22.
let tolerance: Float = 0.02

section("the motion")

test("the swing stays within the cited 45°: never more than 22.5° from rest, and reaches it") {
    let turns: [Float] = (0..<frameCount).map { spinAngle(frame: $0) * 180 / Float.pi }
    let most: Float = turns.map { abs($0) }.max()!
    print(String(format: "        turn from %.2f° to %.2f° over %d frames; the cited swing is %.0f° end to end",
                 turns.min()!, turns.max()!, frameCount, swingDegrees))
    expect(most <= swingDegrees / 2 + 1e-3, "the head turns \(most)° from rest, past the cited \(swingDegrees / 2)°")
    expect(most >= swingDegrees / 2 - 1e-3, "the head never reaches the cited swing: \(most)°")
    expect(turns.min()! < 0 && turns.max()! > 0, "it does not turn both ways")
    for (f, b) in frames.enumerated() {
        expect(abs(b.pose.spin * 180 / Float.pi - turns[f]) < 1e-4, "frame \(f)'s brush is not at its turn")
    }
}

test("the loop closes: frame \(frameCount) is frame 0, and the wrap is a step like any other") {
    let at: (Int) -> Float = { spinAngle(frame: $0) * 180 / Float.pi }
    let wrap: Float = at(frameCount) - at(0)
    var steps: [Float] = []
    for f in 0..<frameCount { steps.append(abs(at(f + 1) - at(f))) }
    // The step into frame 0 must be the continuation of the motion: the
    // same size as the step out of frame 0 (the sine is odd about rest).
    let into: Float = at(0) - at(frameCount - 1)
    let outOf: Float = at(1) - at(0)
    print(String(format: "        turn at frame %d − frame 0: %.5f°; step into frame 0 %.3f°, out of it %.3f°; largest step %.3f°",
                 frameCount, wrap, into, outOf, steps.max()!))
    expect(abs(wrap) < 1e-4, "frame \(frameCount) is \(wrap)° away from frame 0: the loop does not close")
    expect(abs(into - outOf) < 1e-3, "the loop jumps where it wraps: \(into)° in, \(outOf)° out")
    expect(steps.max()! < 2.0, "a step of \(steps.max()!)° between frames is a jerk, not a slow turn")
    // And the brush itself: frame N cast afresh is frame 0's brush.
    guard let o = oscillating, let wrapped = try? o.brush(frame: frameCount), let first = frames.first else {
        expect(false, "no brush"); return
    }
    for (a, b) in zip(wrapped.tufts, first.tufts) {
        expect(simd_distance(a.apex, b.apex) < 1e-3, "frame \(frameCount)'s tufts are not frame 0's")
    }
}

test("the slow-down in the caption is the real rate against the loop's") {
    let real: Double = 1 / oscillationHz
    print(String(format: "        one cycle really takes %.1f ms; the loop takes %.1f s: slowed %.0f×",
                 real * 1000, loopSeconds, slowdown))
    expect(abs(slowdown - loopSeconds * oscillationHz) < 1e-9, "slow-down is not loop ÷ real cycle")
    expect(captionLines()[0].contains("~290×"), "the caption says \(captionLines()[0])")
}

section("at every frame, every tuft touches or hangs free, and none goes through")

test("every tuft's tip touches the tooth or gum, unless it stands free at full length over the gap") {
    guard let pr = probe, frames.count == frameCount else { expect(false, "no brush"); return }
    var worst: Float = 0
    var freeFrames: Int = 0
    var freeGap: Float = 0
    for (f, b) in frames.enumerated() {
        var exact: [SIMD3<Float>] = []
        var grown: [SIMD3<Float>] = []
        for t in b.tufts {
            exact += Array(tuftSurface(t, grow: 0)[0..<tipPointCount])
            grown += Array(tuftSurface(t, grow: tolerance)[0..<tipPointCount])
        }
        guard let de = try? pr.distances(exact), let dg = try? pr.distances(grown) else {
            expect(false, "probe failed"); return
        }
        var anyFree: Bool = false
        for (i, t) in b.tufts.enumerated() {
            let lo: Int = i * tipPointCount
            let hi: Int = lo + tipPointCount
            let nearest: Float = de[lo..<hi].min()!
            let touches: Bool = dg[lo..<hi].min()! < 0
            if t.free {
                anyFree = true
                freeGap = max(freeGap, nearest)
                // Free means: full length, unpressed, straight, and over the
                // #21/#22 embrasure. The embrasure is narrow, so a free tip
                // may still graze its walls (within 0.02 mm near either end
                // of the swing); it is marked free because the surface ahead
                // along its axis is further than it can reach.
                let rel: SIMD2<Float> = SIMD2<Float>(t.apex.x, t.apex.z) - tooth.centre
                let mesiodistal: Float = simd_dot(rel, tooth.tangent)
                expect(abs(t.length - restLength) < 1e-5 && t.compression == 0 && t.tipRadius == t.rootRadius,
                       "frame \(f) tuft \(i) is marked free but is not at its rest length, unpressed")
                expect(abs(mesiodistal) > tooth.spec.width / 2 - 1.0,
                       "frame \(f) tuft \(i) hangs free \(mesiodistal) mm from #21's midline, not over a gap")
            } else {
                worst = max(worst, nearest)
                expect(touches, "frame \(f) tuft \(i) floats: its tip is \(nearest) mm off the surface")
            }
        }
        if anyFree { freeFrames += 1 }
        let freeHere: Int = b.tufts.filter { $0.free }.count
        expect(freeHere <= 1, "frame \(f) has \(freeHere) tufts hanging free")
    }
    print(String(format: "        %d frames × %d tufts; the largest tip-to-surface distance of a touching tuft is %.5f mm",
                 frames.count, frames.first?.tufts.count ?? 0, worst))
    print(String(format: "        %d frames have one tuft free over the #21/#22 gap, up to %.2f mm short of the gum",
                 freeFrames, freeGap))
}

test("no part of any tuft is inside the tooth or gum by more than 0.02 mm, at any frame") {
    guard let pr = probe, frames.count == frameCount else { expect(false, "no brush"); return }
    var checked: Int = 0
    var bad: [String] = []
    for (f, b) in frames.enumerated() {
        var pts: [SIMD3<Float>] = []
        var owner: [Int] = []
        for (i, t) in b.tufts.enumerated() {
            let p: [SIMD3<Float>] = tuftInterior(t, shrink: tolerance)
            pts += p
            owner += Array(repeating: i, count: p.count)
        }
        guard let d = try? pr.distances(pts) else { expect(false, "probe failed"); return }
        checked += pts.count
        var through: Set<Int> = []
        var deepest: Float = 0
        for k in 0..<d.count where d[k] < 0 {
            through.insert(owner[k])
            deepest = min(deepest, d[k])
        }
        if !through.isEmpty { bad.append("frame \(f): tufts \(through.sorted()) up to \(-deepest) mm") }
    }
    print("        \(checked) points filling the tufts, shrunk by \(tolerance) mm, checked over \(frames.count) frames")
    expect(bad.isEmpty, "tufts pass into the tooth or gum: \(bad.prefix(4).joined(separator: "; "))")
}

test("pressed, never stretched: no tuft is longer than a new one at any frame, and splay follows pressing") {
    guard frames.count == frameCount else { expect(false, "no brush"); return }
    var shortest: Float = 99
    var longest: Float = 0
    for (f, b) in frames.enumerated() {
        for t in b.tufts {
            shortest = min(shortest, t.length)
            longest = max(longest, t.length)
            expect((t.tipRadius > t.rootRadius) == (t.compression > 0), "frame \(f): splay does not follow compression")
        }
    }
    print(String(format: "        tufts %.2f–%.2f mm over the loop, against a rest length of %.2f", shortest, longest, restLength))
    expect(longest <= restLength + 1e-3, "a tuft is \(longest) mm, longer than new")
}

test("the tufts really are cast again: their lengths change as the head turns") {
    guard let first = frames.first, frames.count == frameCount else { expect(false, "no brush"); return }
    var most: Float = 0
    for b in frames {
        for (a, t) in zip(first.tufts, b.tufts) { most = max(most, abs(a.length - t.length)) }
    }
    print(String(format: "        a tuft's length changes by up to %.2f mm over the loop", most))
    expect(most > 0.3, "tuft lengths barely change: \(most) mm")
}

test("the kernel draws those tufts, at every frame: zero at each tip, minus the radius at its centre") {
    guard let dev = device, frames.count == frameCount else { expect(false, "no brush"); return }
    for f in stride(from: 0, to: frameCount, by: 10) {
        let b: Brush = frames[f]
        let pts: [SIMD3<Float>] = b.tufts.map { $0.apex } + b.tufts.map { $0.tipCentre }
        guard let r = try? probeBrush(pts, brush: b, on: dev) else { expect(false, "probe failed"); return }
        let n: Int = b.tufts.count
        for i in 0..<n {
            expect(abs(r[i].y) < 1e-3, "frame \(f) tuft \(i): the kernel's tuft distance at the tip is \(r[i].y)")
            expect(abs(r[n + i].y + b.tufts[i].tipRadius) < 1e-3, "frame \(f) tuft \(i): at its centre \(r[n + i].y)")
        }
    }
}

section("the head")

test("the head and neck touch neither tooth nor gum, at the rest pose and both ends of the swing") {
    guard let pr = probe, let dev = device, frames.count == frameCount else { expect(false, "no brush"); return }
    for f in [0, frameCount / 4, 3 * frameCount / 4] {
        let b: Brush = frames[f]
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
        var inside: [SIMD3<Float>] = []
        for k in 0..<grid.count where head[k].x < 0 { inside.append(grid[k]) }
        guard let scene = try? pr.distances(inside) else { expect(false, "probe failed"); return }
        let clearance: Float = scene.min() ?? -1
        print(String(format: "        frame %2d (turn %+.1f°): %d points inside the head and neck; nearest tooth or gum %.2f mm away",
                     f, p.spin * 180 / Float.pi, inside.count, clearance))
        expect(inside.count > 10_000, "only \(inside.count) points inside the head")
        expect(clearance > 0, "frame \(f): the head or neck is inside the tooth or gum")
    }
}

test("only the disc turns: the face, the axis and the neck stay put; the boss turns with the tufts") {
    guard let first = frames.first, frames.count == frameCount else { expect(false, "no brush"); return }
    for (f, b) in frames.enumerated() {
        expect(b.pose.face == first.pose.face && b.pose.axis == first.pose.axis &&
               b.pose.along == first.pose.along && b.pose.across == first.pose.across,
               "frame \(f): the head moved or the neck turned")
        // The central tuft sits on the axis: it does not move at all.
        expect(simd_distance(b.tufts[0].root, first.tufts[0].root) < 1e-5, "frame \(f): the centre tuft moved")
        // A rim tuft's root has turned through the frame's angle.
        let r0: SIMD3<Float> = first.tufts[9].root - first.pose.face
        let r1: SIMD3<Float> = b.tufts[9].root - b.pose.face
        let a0: Float = atan2(simd_dot(r0, b.pose.across), simd_dot(r0, b.pose.along))
        let a1: Float = atan2(simd_dot(r1, b.pose.across), simd_dot(r1, b.pose.along))
        var turned: Float = a1 - a0
        if turned > Float.pi { turned -= 2 * Float.pi }
        if turned < -Float.pi { turned += 2 * Float.pi }
        let boss: Float = atan2(simd_dot(b.pose.spinAlong, b.pose.across), simd_dot(b.pose.spinAlong, b.pose.along))
        expect(abs(turned - b.pose.spin) < 1e-4, "frame \(f): the rim turned \(turned), not \(b.pose.spin)")
        expect(abs(turned - boss) < 1e-4, "frame \(f): the boss and the tufts turn differently")
    }
}

section("the picture")

/// Decode a PNG to RGBA bytes.
func loadPNG(_ path: String) -> (width: Int, height: Int, bytes: [UInt8])? {
    guard let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil),
          let img = CGImageSourceCreateImageAtIndex(src, 0, nil),
          let space = CGColorSpace(name: CGColorSpace.sRGB) else { return nil }
    let w: Int = img.width
    let h: Int = img.height
    var bytes: [UInt8] = Array(repeating: 0, count: w * h * 4)
    let ok: Bool = bytes.withUnsafeMutableBytes { buf -> Bool in
        guard let ctx = CGContext(data: buf.baseAddress, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                                  space: space, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { return false }
        ctx.draw(img, in: CGRect(x: 0, y: 0, width: w, height: h))
        return true
    }
    return ok ? (w, h, bytes) : nil
}

func bytes(_ img: MouthImage) -> [UInt8] {
    let p = img.pixels.contents().assumingMemoryBound(to: UInt8.self)
    return Array(UnsafeBufferPointer(start: p, count: img.width * img.height * 4))
}

/// Pixels (RGB) that differ between two RGBA byte arrays, optionally only
/// those outside a window.
func differing(_ a: [UInt8], _ b: [UInt8], width: Int, outside w: PixelWindow? = nil) -> Int {
    var n: Int = 0
    for i in 0..<(a.count / 4) {
        if let w = w, w.contains(i % width, i / width) { continue }
        if a[4 * i] != b[4 * i] || a[4 * i + 1] != b[4 * i + 1] || a[4 * i + 2] != b[4 * i + 2] { n += 1 }
    }
    return n
}

// Full size, the still's own 16 samples a pixel.
let W: Int = 1920
let H: Int = 1080
let still: MouthImage? = {
    guard let dev = device, let first = frames.first else { return nil }
    return try? renderMouth(width: W, height: H, samples: 4, extra: brushExtra(first), on: dev).image
}()

test("frame 0, at rest, is step 22's still pixel for pixel (showcase/brush.png)") {
    guard let img = still, let ref = loadPNG("../showcase/brush.png") else { expect(false, "render or PNG failed"); return }
    expect(ref.width == W && ref.height == H, "showcase/brush.png is \(ref.width) × \(ref.height)")
    let n: Int = differing(bytes(img), ref.bytes, width: W)
    print("        \(n) of \(W * H) pixels differ from step 22's committed picture")
    expectEqual(n, 0)
}

test("away from the brush every frame is step 20's picture, pixel for pixel (showcase/mouth.png)") {
    guard let img = still, let ref = loadPNG("../showcase/mouth.png"), let dev = device, let o = oscillating
    else { expect(false, "render or PNG failed"); return }
    guard let window = try? movingWindow(brush: o, width: W, height: H, on: dev) else { expect(false, "no window"); return }
    // Outside the window each frame IS the still; the still outside the
    // brush's reach is step 20's. The top third of the frame, the far side
    // of the arch, is nowhere near the brush, its shadow or its occlusion.
    let top: PixelWindow = PixelWindow(x1: W, y0: H / 3, y1: H)
    let n: Int = differing(bytes(img), ref.bytes, width: W, outside: top)
    print("        window \(window.x1) × \(window.y1 - window.y0) from row \(window.y0); \(n) pixels in the top third differ from step 20")
    expectEqual(n, 0)
    expect(window.y0 > H / 3, "the moving window reaches into the top third")
}

test("rendering only the window loses nothing: whole frames at both ends of the swing match, 16 samples") {
    guard let dev = device, let o = oscillating, let img = still, frames.count == frameCount,
          let window = try? movingWindow(brush: o, width: W, height: H, on: dev),
          let work = dev.makeBuffer(length: W * H * 4, options: .storageModeShared),
          let aux = dev.makeBuffer(length: W * H * 16, options: .storageModeShared)
    else { expect(false, "setup failed"); return }
    for f in [frameCount / 4, 5 * frameCount / 8] {
        guard let whole = try? renderMouth(width: W, height: H, samples: 4, extra: brushExtra(frames[f]), on: dev).image
        else { expect(false, "render failed"); return }
        work.contents().copyMemory(from: img.pixels.contents(), byteCount: W * H * 4)
        guard (try? renderWindow(window, width: W, height: H, samples: 4, extra: brushExtra(frames[f]),
                                 into: work, aux: aux, on: dev)) != nil else { expect(false, "window render failed"); return }
        let composite: [UInt8] = Array(UnsafeBufferPointer(start: work.contents().assumingMemoryBound(to: UInt8.self),
                                                            count: W * H * 4))
        let n: Int = differing(bytes(whole), composite, width: W)
        let moved: Int = differing(bytes(whole), bytes(img), width: W)
        print("        frame \(f): \(moved) pixels differ from the still; the windowed frame differs from the whole frame in \(n)")
        expectEqual(n, 0)
        expect(moved > 5000, "frame \(f) barely differs from the still: \(moved) pixels")
    }
}

finish()
