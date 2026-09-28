// Tests for step 54, then step 45's, which still hold.
//
// Step 54: the flowers are white — measured in a render — and stand in axillary
// racemes of the flora's size; the small pods hang only at nodes older than
// the flowers', are the size chosen, and are smooth; flowers, stalks and pods
// keep clear of the stem, the leaves and the pole. The ladybirds have six
// legs, the seven-spot pattern and the sourced size, measured on what is
// drawn; their feet stay planted on the surface under them and never enter
// it; the alternating tripod is read off the drawn feet; they never pass
// through the plant; and the file of ladybirds joins the loop going forward.
//
// Step 45's tests follow: the bean's numbers against Darwin and the floras,
// its twining direction three ways, its contact with the pole, and the loop.
//
// BEAN_MUTANT=purple|pods_above_flowers|eight_legs|sliding_feet|rewind|
// left_handed breaks it on purpose; `make mutants` requires the suite to fail
// for each.

import Foundation
import Metal
import simd

let mutant: Mutant = Mutant.fromEnvironment

final class Box {
    var renderer: PlantRenderer?
    var error: String = ""
    init() {
        do { renderer = try PlantRenderer(look: lookFor(mutant), on: try findDevice()) } catch { self.error = "\(error)" }
    }
}
let box = Box()
if box.renderer == nil { print("no renderer: \(box.error)") }

func probe(_ s: PlantScene, _ pts: [V3], _ mask: UInt32 = allMaterials) -> [SIMD2<Float>]? {
    guard let r = box.renderer else { return nil }
    return try? r.probe(s, pts, mask: mask)
}

/// Everything but the ladybirds.
let plantMask: UInt32 = maskOf([.pole, .stem, .petiole, .leaf, .petal, .pod])
let ladybirdMask: UInt32 = maskOf([.elytra, .beetle, .leg])

/// The stem as drawn in a scene: its chain.
func drawnStem(_ s: PlantScene) -> Chain? { s.chains.first { $0.material == .stem } }

/// Hours to test at: spread over a loop, off the frame lattice.
let testHours: [Float] = [0, 1.37, 4.02, 7.7]

/// Points on the surface of a tube round a polyline.
func tubeSurface(_ pts: [V3], radius r: Float, around: Int = 12) -> [V3] {
    var out: [V3] = []
    if pts.count < 2 { return out }
    for i in 1..<pts.count {
        let a: V3 = pts[i - 1]
        let b: V3 = pts[i]
        guard simd_distance(a, b) > 1e-5 else { continue }
        let t: V3 = simd_normalize(b - a)
        let e1: V3 = perpendicular(to: t)
        let e2: V3 = simd_cross(t, e1)
        for k in 0..<2 {
            let p: V3 = a + (b - a) * (Float(k) / 2)
            for j in 0..<around {
                let th: Float = Float(j) / Float(around) * 2 * Float.pi
                out.append(p + (e1 * cos(th) + e2 * sin(th)) * r)
            }
        }
    }
    return out
}

/// Points spread through a ladybird's body and over its skin, in the world.
func bodySamples(_ b: Beetle, surface: Bool) -> [V3] {
    var out: [V3] = []
    var rng = Lcg(state: 3)
    for part in bodyParts {
        for _ in 0..<(surface ? 160 : 60) {
            let dir: V3 = rng.unit()
            let f: Float = surface ? 1 : rng.next()
            var q: V3 = part.centre + dir * part.radii * f
            if part.centre == elytraPart.centre { q.z = max(q.z, elytraCut) }
            out.append(b.origin + b.forward * (q.x * b.scale) + b.left * (q.y * b.scale) + b.up * (q.z * b.scale))
        }
    }
    return out
}

/// Frames at which to look at the ladybirds: every other frame of the loop.
let walkFrames: [Int] = Array(stride(from: 0, to: frameCount, by: 2))

// MARK: -

section("step 54: the flowers, against the floras")

test("racemes of 1–3 flowers (FTEA) in the axils; pedicels 3–10 mm (FTEA); calyx 3–4 mm (FoC); a 2–3-inch rachis (McGregor); keel coiled 1–5 turns (FTEA)") {
    expect(flowersPerRacemeRange.contains(flowersPerRaceme))
    expect(pedicelRange.contains(pedicelLength))
    expect(calyxRange.contains(calyxLength))
    expect(racemeRachisRange.contains(racemeRachisLength) && peduncleRange.contains(peduncleLength))
    expect(keelTurnsRange.contains(keelTurns))
    expect(standardLengthRange.contains(standardSize.x) && standardWidthRange.contains(standardSize.y))
    var racemes: Int = 0
    for t in testHours {
        for n in nodes(t, mutant) {
            guard let f = inflorescence(n, t, mutant) else { continue }
            expectEqual(f.flowers.count, flowersPerRaceme)
            // In an axil: the stalk starts at its node.
            expect(simd_distance(f.stalk.points[0], n.position) < 4, "raceme stalk starts off its node")
            racemes += 1
        }
    }
    print("        \(racemes) racemes, \(flowersPerRaceme) flowers each")
    expect(racemes >= 6)
}

test("an open flower is 12.7–19 mm long (McGregor 1/2–3/4 inch), with a standard, two wings and a keel whose beak coils") {
    var checked: Int = 0
    var lengths: [Float] = []
    for k in 0..<20 {
        let t: Float = loopHours * Float(k) / 20
        for n in nodes(t, mutant) {
            guard let f = inflorescence(n, t, mutant), flowerStage(reproductiveAge(n.age, mutant)) == .open else { continue }
            for fl in f.flowers {
                expectEqual(fl.petals.map { $0.shape }, [.standard, .wing, .wing])
                guard let keel = fl.keel, fl.petals.count == 3 else { expect(false, "no keel"); continue }
                // From the calyx's base to the farthest petal point.
                var far: Float = 0
                for l in fl.petals {
                    for i in 0...10 {
                        let xi: Float = Float(i) / 10
                        far = max(far, simd_distance(fl.base, l.origin + l.u * (l.length * xi)))
                    }
                }
                for p in keel.points { far = max(far, simd_distance(fl.base, p)) }
                lengths.append(far)
                // The beak turns more than once round: its direction's angle,
                // unwrapped, about the flower's across axis.
                let beak: [V3] = Array(keel.points.dropFirst(2))
                let c: V3 = beak.reduce(V3(0, 0, 0), +) / Float(beak.count)
                let ang: [Float] = unwrappedAngles(beak, origin: c, axis: fl.across, reference: fl.forward)
                let turns: Float = abs(ang.last! - ang.first!) / (2 * Float.pi)
                expect(keelTurnsRange.contains(turns), "keel beak turns \(turns)")
                checked += 1
            }
        }
    }
    let lo: Float = lengths.min() ?? 0
    let hi: Float = lengths.max() ?? 0
    print(String(format: "        %d open flowers, %.1f–%.1f mm long", checked, lo, hi))
    expect(checked >= 6)
    expect(flowerLengthRange.contains(lo) && flowerLengthRange.contains(hi))
}

test("the petals are white — measured in a render of an open flower (FoC, FTEA, McGregor: white; Russell's choice in step 25)") {
    guard let r = box.renderer else { expect(false, "no renderer"); return }
    var counts: [Int] = []
    var worstSaturation: Float = 0
    var hours: [Float] = []
    for k in 0..<40 {
        let t: Float = loopHours * Float(k) / 40
        if nodes(t, mutant).contains(where: { hasRaceme($0.index) && flowerStage(reproductiveAge($0.age, mutant)) == .open && $0.age > 4.6 }) { hours.append(t) }
    }
    for t in hours.prefix(3) {
        guard let n = nodes(t, mutant).first(where: { hasRaceme($0.index) && flowerStage(reproductiveAge($0.age, mutant)) == .open && $0.age > 4.6 }),
              let f = inflorescence(n, t, mutant), let st = f.flowers.first?.petals.first else { continue }
        let centre: V3 = st.origin + st.u * (0.45 * st.length)
        let toward: V3 = simd_normalize(camera(t).position - centre)
        let cam = Camera(position: centre + toward * 150, target: centre, tanHalf: 12.0 / 150)
        guard let out = try? r.render(beanScene(t, mutant), camera: cam, width: 96, height: 96, samples: 1, tubeMin: 0)
        else { expect(false, "render failed"); return }
        var sum = SIMD3<Float>(0, 0, 0)
        var c: Int = 0
        for i in 0..<(96 * 96) where Int(out.frame.seen[i].x) == Material.petal.rawValue && out.frame.rgba[i].w > 0.99 {
            let px: SIMD4<Float> = out.frame.rgba[i]
            sum += SIMD3<Float>(px.x, px.y, px.z)
            c += 1
        }
        guard c > 0 else { counts.append(0); continue }
        let mean: SIMD3<Float> = sum / Float(c)
        let sat: Float = (mean.max() - mean.min()) / max(mean.max(), 1e-3)
        worstSaturation = max(worstSaturation, sat)
        counts.append(c)
        print(String(format: "        hour %.2f: %d petal pixels, mean sRGB (%.2f, %.2f, %.2f)", t, c, mean.x, mean.y, mean.z))
        expect(mean.min() > 0.55, "petals too dark: \(mean)")
        expect(sat < 0.12, "petals coloured: \(mean), saturation \(sat)")
    }
    print(String(format: "        worst saturation %.3f", worstSaturation))
    expect(counts.count == 3 && counts.allSatisfy { $0 > 200 }, "too few petal pixels to judge: \(counts)")
}

section("step 54: small pods below the flowers")

test("pods hang only at nodes older than any flower's, with withering petals round young pods in between") {
    var seenBetween: Int = 0
    var checked: Int = 0
    var podNodes: Int = 0
    for k in 0..<30 {
        let t: Float = loopHours * Float(k) / 30 + 0.29
        var oldestFlower: Float = -1
        var youngestPod: Float = .infinity
        for n in nodes(t, mutant) {
            guard let f = inflorescence(n, t, mutant) else { continue }
            let fresh: Bool = f.flowers.contains { fl in fl.petals.contains { $0.tint < 0.01 } } && f.flowers.allSatisfy { $0.pod == nil }
            let pods: Bool = f.flowers.contains { $0.pod != nil }
            if fresh { oldestFlower = max(oldestFlower, n.age) }
            if pods { youngestPod = min(youngestPod, n.age); podNodes += 1 }
            if pods && f.flowers.contains(where: { fl in fl.petals.contains { $0.tint > 0.2 } }) { seenBetween += 1 }
        }
        expect(oldestFlower < youngestPod, String(format: "hour %.2f: a flower at a node %.1f h old, a pod at %.1f h", t, oldestFlower, youngestPod))
        checked += 1
    }
    print("        \(checked) hours checked; \(podNodes) node-hours with pods; \(seenBetween) with withering petals round a young pod")
    expect(seenBetween > 3 && podNodes > 20)
}

test("a small pod is 33–75 × 3–7 mm (MODEL: a third to a half of the floras' 10–15 cm), smooth — no seed bulges yet (Gómez-Martín 2020) — and beaked") {
    var pods: Int = 0
    for t in testHours {
        for n in nodes(t, mutant) where reproductiveAge(n.age, mutant) >= podFull {
            guard let f = inflorescence(n, t, mutant) else { continue }
            for fl in f.flowers {
                guard let p = fl.pod else { expect(false, "no pod at an old node"); continue }
                let len: Float = polylineLength(p.points)
                let width: Float = 2 * (p.radii.max() ?? 0)
                expect(podLengthRange.contains(len) && podWidthRange.contains(width), "pod \(len) × \(width) mm")
                expect(len < fullPodLengthRange.lowerBound / 2 + 1, "longer than half a full-grown pod")
                // Smooth: once past its rounded base the radius never rises again.
                var rises: Int = 0
                for i in 3..<p.radii.count where p.radii[i] > p.radii[i - 1] + 1e-4 { rises += 1 }
                expectEqual(rises, 0)
                // Beaked: the tip is under a fifth of the widest.
                expect(p.radii.last! < 0.2 * (p.radii.max() ?? 1), "no beak")
                if pods == 0 { print(String(format: "        a small pod: %.1f × %.1f mm, tip %.2f mm", len, width, 2 * p.radii.last!)) }
                pods += 1
            }
        }
    }
    print("        \(pods) small pods checked")
    expect(pods >= 6)
}

test("no flower, flower stalk or pod passes through the stem, a leaf or the pole, at any hour of the loop") {
    var deepest: Float = .infinity
    let others: UInt32 = maskOf([.stem, .leaf, .pole])
    var count: Int = 0
    for k in 0..<16 {
        let t: Float = loopHours * Float(k) / 16 + 0.41
        let s: PlantScene = beanScene(t, mutant)
        var pts: [V3] = []
        for n in nodes(t, mutant) {
            guard let f = inflorescence(n, t, mutant) else { continue }
            // The stalk, but not its first 3 mm, which grow out of the stem.
            let stalk: [V3] = f.stalk.points.filter { simd_distance($0, f.stalk.points[0]) > 3 }
            pts += tubeSurface(stalk, radius: 0.55, around: 8)
            for fl in f.flowers {
                pts += tubeSurface(fl.pedicel.points, radius: 0.4, around: 8)
                pts += tubeSurface(fl.calyx.points, radius: 0.6, around: 8)
                if let kc = fl.keel { pts += tubeSurface(kc.points, radius: 0.3, around: 6) }
                if let p = fl.pod {
                    for i in 0..<p.points.count - 1 {
                        pts += tubeSurface([p.points[i], p.points[i + 1]], radius: p.radii[i], around: 10)
                    }
                }
                for l in fl.petals {
                    for i in 0...8 {
                        let xi: Float = Float(i) / 8
                        let half: Float = l.halfWidth * outlineProfile(l.shape, xi) * 0.9
                        for sgn in [Float(-1), 0, 1] {
                            let across: V3 = l.v * (sgn * cos(l.fold)) + l.w * (abs(sgn) * sin(l.fold))
                            pts.append(l.origin + l.u * (l.length * xi) - l.w * (l.droop * l.length * xi * xi) + across * half)
                        }
                    }
                }
            }
        }
        count += pts.count
        guard let d = probe(s, pts, others) else { expect(false, "probe failed"); return }
        deepest = min(deepest, d.map { $0.x }.min() ?? .infinity)
    }
    print(String(format: "        %d points: the nearest flower or pod point is %.2f mm from stem, leaf or pole", count, deepest))
    expect(deepest > -0.05, "a flower or pod enters something by \(-deepest) mm")
}

section("step 54: the ladybirds' bodies, against the sources")

test("every ladybird drawn has six legs, each joined to its thorax (an insect; Heepe et al. 2016: fore, mid and hind legs)") {
    var beetles: Int = 0
    var wrong: Int = 0
    for f in stride(from: 0, to: frameCount, by: 25) {
        let t: Float = hourOf(frame: f)
        let s: PlantScene = fullScene(t, mutant)
        let legs: [Chain] = s.chains.filter { $0.material == .leg }
        for b in s.beetles {
            // Legs whose hip is under this beetle's body.
            let mine: Int = legs.filter { leg in
                let q: V3 = leg.points[0] - b.origin
                let x: Float = simd_dot(q, b.forward) / b.scale
                let y: Float = simd_dot(q, b.left) / b.scale
                return abs(y) < 1.2 && x > -1.5 && x < 2.0 && simd_length(q) < 3 * b.scale
            }.count
            if mine != insectLegCount { wrong += 1 }
            beetles += 1
        }
        expectEqual(legs.count, insectLegCount * s.beetles.count)
    }
    print("        \(beetles) ladybirds looked at; \(wrong) without exactly six legs")
    expect(beetles >= 10 && wrong == 0)
}

test("seven spots — one on the join of the elytra by the scutellum, three on each elytron, mirrored (ADW, UC IPM) — and white patches beside the scutellum") {
    expectEqual(spots.count, 7)
    let mid: [SIMD3<Float>] = spots.filter { abs($0.y) < 1e-4 }
    let left: [SIMD3<Float>] = spots.filter { $0.y > 1e-4 }
    let right: [SIMD3<Float>] = spots.filter { $0.y < -1e-4 }
    expectEqual(mid.count, 1)
    expect(left.count == spotsPerElytron && right.count == spotsPerElytron)
    for l in left { expect(right.contains { abs($0.x - l.x) < 1e-5 && abs($0.y + l.y) < 1e-5 && abs($0.z - l.z) < 1e-5 }) }
    // The scutellar spot is the frontmost, at the elytra's base.
    if let s = mid.first { expect(s.x > (left.map { $0.x }.max() ?? 0), "the shared spot is not at the front") }
    // Every spot on the elytra, seen from above; none over the rim.
    for s in spots {
        let e: Float = pow((s.x - elytraPart.centre.x) / elytraPart.radii.x, 2) + pow(s.y / elytraPart.radii.y, 2)
        expect(e < 0.85, "spot off the elytra")
    }
    for w in elytraWhite { expect(abs(w.x - scutellarSpot.x) < 0.6 && abs(w.y) > 0.4, "white patch not beside the scutellar spot") }
}

test("seen from above, a drawn ladybird shows seven separate black spots on red elytra: three each side and one on the midline") {
    guard let r = box.renderer else { expect(false, "no renderer"); return }
    let b = Beetle(origin: V3(0, 0, 0), forward: V3(1, 0, 0), left: V3(0, 0, -1), up: V3(0, 1, 0), scale: 1)
    var s = PlantScene()
    s.beetles = [b]
    let cam = Camera(position: V3(0, 60, 0.01), target: V3(0, 0, 0), tanHalf: 4.5 / 60)
    guard let out = try? r.render(s, camera: cam, width: 160, height: 160, samples: 2, tubeMin: 0) else {
        expect(false, "render failed"); return
    }
    let n: Int = 160
    var dark = [Bool](repeating: false, count: n * n)
    var red: Int = 0
    for i in 0..<(n * n) where Int(out.frame.seen[i].x) == Material.elytra.rawValue {
        let c: SIMD4<Float> = out.frame.rgba[i]
        // Black: no red in it (a spot's glossy highlight is grey, not red),
        // and not the white patches.
        if c.x - c.y < 0.15 && c.y < 0.5 { dark[i] = true }
        if c.x > 0.3 && c.x > 2 * c.y { red += 1 }
    }
    // Connected dark regions, and where each one is.
    var label = [Int](repeating: -1, count: n * n)
    var blobs: [(size: Int, cx: Float, cy: Float)] = []
    for i in 0..<(n * n) where dark[i] && label[i] < 0 {
        var stack: [Int] = [i]
        label[i] = blobs.count
        var size: Int = 0
        var sx: Float = 0
        var sy: Float = 0
        while let j = stack.popLast() {
            size += 1
            sx += Float(j % n)
            sy += Float(j / n)
            for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                let x: Int = j % n + dx
                let y: Int = j / n + dy
                guard x >= 0, x < n, y >= 0, y < n else { continue }
                let k: Int = y * n + x
                if dark[k] && label[k] < 0 { label[k] = blobs.count; stack.append(k) }
            }
        }
        blobs.append((size, sx / Float(size), sy / Float(size)))
    }
    let spotsSeen = blobs.filter { $0.size > 12 }
    // The camera looks down with the beetle's forward along +x (screen right)
    // and its left along −z, which is screen up.
    let centreRow: Float = Float(n) / 2
    let midline = spotsSeen.filter { abs($0.cy - centreRow) < 4 }
    let above = spotsSeen.filter { $0.cy < centreRow - 4 }
    let below = spotsSeen.filter { $0.cy > centreRow + 4 }
    print("        \(spotsSeen.count) dark spots on the elytra (\(above.count) + \(midline.count) + \(below.count)); \(red) red pixels")
    expectEqual(spotsSeen.count, 7)
    expect(midline.count == 1 && above.count == 3 && below.count == 3)
    expect(red > 1500, "the elytra are not red")
}

test("each ladybird is the sourced size: 6.5–7.8 mm long (ADW), about 4–5.5 mm wide (UC IPM, MODEL band) — measured on the distance function") {
    let b = Beetle(origin: V3(0, 0, 0), forward: V3(1, 0, 0), left: V3(0, 0, -1), up: V3(0, 1, 0), scale: ladybirdScale)
    var s = PlantScene()
    s.beetles = [b]
    var pts: [V3] = []
    let step: Float = 0.02
    for i in 0..<600 { pts.append(V3(-6 + Float(i) * step, 0.9, 0)) }      // along its length, at mid-height
    for i in 0..<600 { pts.append(V3(-0.5, 0.9, -6 + Float(i) * step)) }   // across it
    for i in 0..<400 { pts.append(V3(-0.5, -1 + Float(i) * step, 0)) }     // up through it
    guard let d = probe(s, pts, ladybirdMask) else { expect(false, "probe failed"); return }
    func extent(_ range: Range<Int>) -> Float {
        let inside: [Int] = range.filter { d[$0].x < 0 }
        guard let a = inside.first, let z = inside.last else { return 0 }
        return Float(z - a) * step
    }
    let length: Float = extent(0..<600)
    let widthM: Float = extent(600..<1200)
    let heightM: Float = extent(1200..<1600)
    // Length is the whole body, head to tail, not only at mid-height: take the
    // longest line through it.
    var longest: Float = 0
    for z in stride(from: Float(0.1), through: 2.0, by: 0.1) {
        var line: [V3] = []
        for i in 0..<600 { line.append(V3(-6 + Float(i) * step, z, 0)) }
        guard let dl = probe(s, line, ladybirdMask) else { break }
        let inside: [Int] = (0..<600).filter { dl[$0].x < 0 }
        if let a = inside.first, let e = inside.last { longest = max(longest, Float(e - a) * step) }
    }
    print(String(format: "        length %.2f mm (at mid-height %.2f), width %.2f mm, height %.2f mm", longest, length, widthM, heightM))
    expect(ladybirdLengthRange.contains(longest), "length \(longest)")
    expect(ladybirdWidthRange.contains(widthM), "width \(widthM)")
}

section("step 54: the ladybirds' walk")

test("the ladybirds' clock is real time, and the frame says so, with the bean about 3600× faster") {
    let f: Float = plantSpeedUp()
    print(String(format: "        the bean runs %.0f× faster than the ladybirds; \"%@\"", f, clockCaptionLine()))
    expect(abs(f - 3605) < 10 && ladybirdClockFactor == 1)
    expect(clockCaptionLine().contains("real time") && clockCaptionLine().contains("3600"))
}

test("feet are planted: a foot in stance touches the surface under it, and no foot or leg ever enters anything") {
    var stanceFeet: Int = 0
    var worstGap: Float = 0
    var deepestFoot: Float = .infinity
    var deepestLeg: Float = .infinity
    for f in walkFrames {
        let t: Float = hourOf(frame: f, mutant: mutant)
        let s: PlantScene = fullScene(t, mutant)
        var feet: [V3] = []
        var down: [Bool] = []
        var legPts: [V3] = []
        for lb in ladybirdsInView(t, mutant) {
            for leg in lb.legs {
                feet.append(leg.foot)
                down.append(leg.stance)
                // The leg's skin, short of the foot's round end.
                let tib: [V3] = [leg.knee, leg.knee + (leg.foot - leg.knee) * 0.85]
                legPts += tubeSurface([leg.hip, leg.knee], radius: femurRadius, around: 6)
                legPts += tubeSurface(tib, radius: tibiaRadius, around: 6)
            }
        }
        guard let df = probe(s, feet, plantMask), let dl = probe(s, legPts, plantMask) else { expect(false, "probe failed"); return }
        for (i, d) in df.enumerated() {
            deepestFoot = min(deepestFoot, d.x - footRadius)
            if down[i] {
                worstGap = max(worstGap, abs(d.x - footRadius))
                stanceFeet += 1
            }
        }
        deepestLeg = min(deepestLeg, dl.map { $0.x }.min() ?? .infinity)
    }
    print(String(format: "        %d stance feet: farthest %.3f mm off the surface; deepest foot %.3f mm in, deepest leg %.3f mm in",
                 stanceFeet, worstGap, max(-deepestFoot, 0), max(-deepestLeg, 0)))
    expect(stanceFeet > 500)
    expect(worstGap < 0.05, "a stance foot is \(worstGap) mm off the surface")
    expect(deepestFoot > -0.03, "a foot enters the plant by \(-deepestFoot) mm")
    expect(deepestLeg > -0.03, "a leg enters the plant by \(-deepestLeg) mm")
}

test("feet do not slide: through a stance a foot keeps its place on the pole, and on the stem it moves only as the stem moves") {
    // Fine steps through a stretch of the walk on the pole, then on the stem.
    var poleChecked: Int = 0
    var stemChecked: Int = 0
    var worstPole: Float = 0
    var worstStem: Float = 0
    let dt: Float = 0.01
    for segment in [Segment.pole, Segment.stem] {
        // Local time within copy 0's walk when it is on this segment.
        let dMid: Float = segment == .pole ? -30 : (walkPlans[0].switchD[0] + walkPlans[0].switchD[1]) / 2
        let tau0: Float = walkPlans[0].time(atDistance: dMid)
        var prev: [LegState] = []
        var prevStem: Chain? = nil
        for i in 0..<120 {
            let tau: Float = tau0 + Float(i) * dt
            let lb: Ladybird = ladybird(copy: 0, tau: tau, mutant: mutant)
            let stemNow: Chain? = drawnStem(beanScene(plantHours(tau), mutant))
            if !prev.isEmpty {
                for j in 0..<min(prev.count, lb.legs.count) where prev[j].stance && lb.legs[j].stance && prev[j].stride == lb.legs[j].stride {
                    if segment == .pole {
                        worstPole = max(worstPole, simd_distance(prev[j].foot, lb.legs[j].foot))
                        poleChecked += 1
                    } else if let a = prevStem, let b = stemNow {
                        // Where the foot was on the drawn stem a moment ago —
                        // segment (by lattice height), along it, round it —
                        // and where that place of the stem is now.
                        let p: V3 = prev[j].foot
                        var bestD: Float = .infinity
                        var bestI: Int = 0
                        for k in 0..<(a.points.count - 1) {
                            let d: Float = segmentDistance(p, a.points[k], a.points[k + 1])
                            if d < bestD { bestD = d; bestI = k }
                        }
                        guard bestD < stemRadius + 0.5 else { continue }
                        let a0: V3 = a.points[bestI]
                        let a1: V3 = a.points[bestI + 1]
                        // The same lattice sample in the new chain: match by height step.
                        guard let bi = b.points.indices.first(where: { abs(b.points[$0].y - a0.y) < 0.8 && simd_distance(V3(b.points[$0].x, 0, b.points[$0].z), V3(a0.x, 0, a0.z)) < 2 }),
                              bi + 1 < b.points.count else { continue }
                        let b0: V3 = b.points[bi]
                        let b1: V3 = b.points[bi + 1]
                        let ha: Float = simd_dot(p - a0, a1 - a0) / simd_length_squared(a1 - a0)
                        let fa = segmentFrame(a0, a1, hint: radialHint(a0, a1))
                        let fb = segmentFrame(b0, b1, hint: radialHint(b0, b1))
                        let off: V3 = p - (a0 + (a1 - a0) * ha)
                        let predicted: V3 = b0 + (b1 - b0) * ha + fb.dir * simd_dot(off, fa.dir) + fb.e1 * simd_dot(off, fa.e1) + fb.e2 * simd_dot(off, fa.e2)
                        worstStem = max(worstStem, simd_distance(predicted, lb.legs[j].foot))
                        stemChecked += 1
                    }
                }
            }
            prev = lb.legs
            prevStem = stemNow
        }
    }
    print(String(format: "        %d foot-steps on the pole: moved at most %.4f mm; %d on the stem: off the stem's own motion by at most %.4f mm",
                 poleChecked, worstPole, stemChecked, worstStem))
    expect(poleChecked > 200 && stemChecked > 50)
    expect(worstPole < 1e-3, "a planted foot slid \(worstPole) mm on the pole")
    expect(worstStem < 0.01, "a planted foot slid \(worstStem) mm over the stem")
}

test("the alternating tripod, read off the drawn feet: right fore, left mid and right hind step together, then the other three") {
    // On the pole, where a planted foot does not move at all.
    let dt: Float = 0.005
    let tau0: Float = walkPlans[0].time(atDistance: -40)
    var prev: [V3] = []
    var downA: Int = 0, downB: Int = 0, both: Int = 0, samples: Int = 0
    var disagree: Int = 0
    var fewest: Int = 6
    var legsSeen: Int = 0
    for i in 0..<600 {
        let lb: Ladybird = ladybird(copy: 0, tau: tau0 + Float(i) * dt, mutant: mutant)
        let feet: [V3] = lb.legs.map { $0.foot }
        legsSeen = feet.count
        if prev.count == feet.count && feet.count >= 6 {
            let still: [Bool] = (0..<feet.count).map { simd_distance(feet[$0], prev[$0]) < 1e-4 }
            let a: [Bool] = tripodA.sorted().map { still[$0] }
            let b: [Bool] = tripodB.sorted().map { still[$0] }
            if Set(a).count > 1 || Set(b).count > 1 { disagree += 1 }
            if a.allSatisfy({ $0 }) { downA += 1 }
            if b.allSatisfy({ $0 }) { downB += 1 }
            if a.allSatisfy({ $0 }) && b.allSatisfy({ $0 }) { both += 1 }
            fewest = min(fewest, still.filter { $0 }.count)
            samples += 1
        }
        prev = feet
    }
    let fa: Float = Float(downA) / Float(max(samples, 1))
    let fb: Float = Float(downB) / Float(max(samples, 1))
    let fboth: Float = Float(both) / Float(max(samples, 1))
    print(String(format: "        %d samples: tripod A down %.0f%%, B down %.0f%%, both %.0f%%; tripod legs out of step in %d; fewest feet down %d",
                 samples, fa * 100, fb * 100, fboth * 100, disagree, fewest))
    expectEqual(legsSeen, 6)
    expect(Float(disagree) < 0.02 * Float(samples), "the legs of a tripod do not step together")
    expect(abs(fa - dutyFactor) < 0.05 && abs(fb - dutyFactor) < 0.05)
    expect(abs(fboth - (2 * dutyFactor - 1)) < 0.05, "the tripods do not alternate")
    expect(fewest >= 3, "fewer than three feet down")
}

test("no leg is ever stretched beyond its reach") {
    var stretched: Int = 0
    var legs: Int = 0
    for f in walkFrames {
        for lb in ladybirdsInView(hourOf(frame: f, mutant: mutant), mutant) {
            for leg in lb.legs {
                legs += 1
                if leg.stretched { stretched += 1 }
            }
        }
    }
    print("        \(legs) legs, \(stretched) stretched")
    expect(legs > 1000 && stretched == 0)
}

test("no ladybird passes through the plant, and no part of the plant through a ladybird, anywhere in the loop") {
    var deepest: Float = .infinity
    var deepestPlant: Float = .infinity
    var worstAt: String = ""
    for f in walkFrames {
        let t: Float = hourOf(frame: f, mutant: mutant)
        let s: PlantScene = fullScene(t, mutant)
        var pts: [V3] = []
        for lb in ladybirdsInView(t, mutant) {
            pts += bodySamples(lb.beetle, surface: true)
            pts += bodySamples(lb.beetle, surface: false)
        }
        guard let d = probe(s, pts, plantMask) else { expect(false, "probe failed"); return }
        for (i, v) in d.enumerated() where v.x < deepest {
            deepest = v.x
            worstAt = "frame \(f), material \(Int(v.y)), point \(pts[i])"
        }
        // The plant's own surfaces near each ladybird, against the ladybirds.
        var near: [V3] = []
        for c in s.chains where c.material != .leg {
            for (k, p) in c.points.enumerated() where s.beetles.contains(where: { simd_distance($0.origin, p) < 8 }) {
                near += tubeSurface([p, k + 1 < c.points.count ? c.points[k + 1] : p + V3(0, 0.01, 0)], radius: c.radii[k], around: 10)
            }
        }
        if !near.isEmpty, let dn = probe(s, near, maskOf([.elytra, .beetle])) {
            deepestPlant = min(deepestPlant, dn.map { $0.x }.min() ?? .infinity)
        }
    }
    print(String(format: "        deepest body point %.3f mm inside the plant (%@); deepest plant point %.3f mm inside a ladybird",
                 max(-deepest, 0), worstAt, max(-deepestPlant, 0)))
    expect(deepest > -0.03, "a ladybird enters the plant")
    expect(deepestPlant > -0.03, "the plant enters a ladybird")
}

test("the ladybirds walk the stem and a leaf in view — pole, stem, leaf stalk and leaflet all seen — and two or three show at once, counted in the picture") {
    guard let r = box.renderer else { expect(false, "no renderer"); return }
    var seen = Set<Int>()
    var inFrame: [Int] = []
    let w: Int = 240
    let h: Int = 312
    for f in stride(from: 0, to: frameCount, by: 5) {
        let t: Float = hourOf(frame: f)
        let cam: Camera = camera(t)
        guard let out = try? r.render(fullScene(t, .none), camera: cam, width: w, height: h, samples: 1, tubeMin: 0.35)
        else { expect(false, "render failed"); return }
        var n: Int = 0
        for lb in ladybirdsInView(t, .none) {
            // Visible: some pixel near where it projects shows ladybird, and
            // it is clear of the caption bands.
            let p: SIMD2<Float> = cam.project(lb.pose.origin + lb.pose.up * 1.2, width: w, height: h)
            guard p.y > 40, p.y < 270 else { continue }
            var shows: Bool = false
            for dy in -3...3 {
                for dx in -3...3 {
                    let x: Int = Int(p.x) + dx
                    let y: Int = Int(p.y) + dy
                    guard x >= 0, x < w, y >= 0, y < h else { continue }
                    let m: Int = Int(out.frame.seen[y * w + x].x)
                    if m == Material.elytra.rawValue || m == Material.beetle.rawValue { shows = true }
                }
            }
            if shows {
                seen.insert(lb.segment.rawValue)
                n += 1
            }
        }
        inFrame.append(n)
    }
    var hist = [Int](repeating: 0, count: 6)
    for n in inFrame { hist[min(n, 5)] += 1 }
    print("        segments seen in the picture: \(seen.sorted()); frames showing 0…5 ladybirds: \(hist)")
    expectEqual(seen.count, 4)
    // A few (2–3) nearly all the time: never none, rarely one or four.
    let fewFraction: Float = Float(hist[2] + hist[3]) / Float(inFrame.count)
    expect(hist[0] == 0 && fewFraction > 0.8, "two or three in view only \(fewFraction * 100)% of the time")
}

test("the file of ladybirds joins the loop going forward: every ladybird's walk only advances, frame to frame and across the join, at a steady pace") {
    var bad: Int = 0
    var steps: [Float] = []
    for f in 0..<frameCount {
        let t0: Float = hourOf(frame: f, mutant: mutant)
        let t1: Float = f + 1 < frameCount ? hourOf(frame: f + 1, mutant: mutant) : hourOf(frame: 0, mutant: mutant) + loopHours
        for c in -1...1 {
            let d0: Float = ladybird(copy: c, tau: filmSeconds(t0), mutant: mutant).distance
            let d1: Float = ladybird(copy: c, tau: filmSeconds(t1), mutant: mutant).distance
            if d1 >= walkPlans[0].endD - 1e-3 { continue }
            if !(d1 > d0) { bad += 1 }
            steps.append(d1 - d0)
        }
    }
    // And a loop later, each ladybird stands where the one ahead of it stood,
    // a loop's rise higher.
    var worst: Float = 0
    for t in testHours {
        let a: Ladybird = ladybird(copy: 0, tau: filmSeconds(t) + loopSeconds, mutant: mutant)
        let b: Ladybird = ladybird(copy: -1, tau: filmSeconds(t), mutant: mutant)
        worst = max(worst, simd_distance(a.pose.origin, b.pose.origin + V3(0, loopRise, 0)))
        for (la, lb) in zip(a.legs, b.legs) { worst = max(worst, simd_distance(la.foot, lb.foot + V3(0, loopRise, 0))) }
    }
    let lo: Float = steps.min() ?? 0
    let hi: Float = steps.max() ?? 0
    print(String(format: "        %d backward steps; each frame %.3f–%.3f mm of walk (%.1f mm/s); a loop on, off by %.5f mm",
                 bad, lo, hi, hi * 20, worst))
    expect(bad == 0, "\(bad) ladybird steps run backwards")
    expect(hi - lo < 1e-3, "uneven pace")
    expect(abs(hi * Float(frameCount) / loopSeconds - walkingSpeed) < 0.01)
    expect(worst < 0.01)
}

section("numbers, against the sources (step 45)")

test("the revolution period is Darwin's: three circles of 2 h 0 m, 1 h 55 m, 1 h 55 m — 'the average of 1 hr. 57 m.'") {
    let minutes: Float = revolutionHours * 60
    print(String(format: "        revolution %.1f min", minutes))
    expect(abs(minutes - darwinAverageMinutes) < 0.5, "revolution is \(minutes) min")
}

test("winding one turn takes longer than one free revolution, by a ratio inside Darwin's two timed cases") {
    let lo: Float = darwinWindingRatios.min()!
    let hi: Float = darwinWindingRatios.max()!
    expect(gyreHours / revolutionHours >= lo && gyreHours / revolutionHours <= hi)
}

test("the label's hours are the loop's real duration, and the time-lapse is the climb's, not the pods'") {
    let caption: String = timeCaption()
    print(String(format: "        loop = %.2f h; caption \"%@\"", loopHours, caption))
    expect(abs(loopHours - 3 * gyreHours) < 1e-4)
    expect(caption.contains(String(format: "%.0f hours", loopHours)) && caption.contains("climb"))
    for c in [selfCaption, aphidCaption, gradientCaption, compressedCaption, nameCaption, ladybirdNameCaption] {
        expect(!c.contains("time-lapse"))
    }
    expect(gradientCaption.contains("flowers above, pods below") && gradientCaption.contains("older nodes"))
    expect(compressedCaption.contains("compressed") && compressedCaption.contains("week"))
    expect(selfCaption.contains("mostly self-pollinated") && selfCaption.contains("0–10%"))
    expect(ladybirdNameCaption.contains("Coccinella septempunctata") && aphidCaption.contains("none drawn"))
}

test("the pole is a size beans are recorded twining round (Darwin 1875: 1/3-inch rods to 4-inch sticks; not 9 inches)") {
    expect(poleDiameter >= 8.4 && poleDiameter <= twinedDiameterRecorded)
    expect(poleDiameter < failedDiameterRecorded)
}

test("the leaf is trifoliate at the floras' sizes: petiole 4–9 cm, leaflets 4–16 × 2.5–11 cm") {
    expect(petioleRange.contains(petioleLength))
    for l in [terminalLeaflet, lateralLeaflet] {
        expect(leafletLengthRange.contains(l.x), "leaflet length \(l.x)")
        expect(leafletWidthRange.contains(l.y), "leaflet width \(l.y)")
    }
    let s: PlantScene = beanScene(2.0, mutant)
    let n: Int = nodes(2.0, mutant).count
    expectEqual(s.leaflets.filter { $0.material == .leaf }.count, 3 * n)
}

section("which way round: anticlockwise seen from above (Darwin: 'against the sun')")

func woundStem(_ t: Float) -> [V3] {
    let s: PlantScene = beanScene(t, mutant)
    guard let c = drawnStem(s) else { return [] }
    let n: Int = stem(t, mutant).frontIndex
    return Array(c.points.prefix(n))
}

test("seen from above, the stem goes round anticlockwise as it climbs, at every step") {
    var bad: Int = 0
    var total: Int = 0
    for t in testHours {
        let pts: [V3] = woundStem(t)
        let a: [Float] = unwrappedAngles(pts, origin: V3(0, 0, 0), axis: V3(0, 1, 0), reference: V3(1, 0, 0))
        for i in 1..<pts.count {
            total += 1
            if !(pts[i].y > pts[i - 1].y && a[i] > a[i - 1]) { bad += 1 }
        }
    }
    print("        \(total - bad) of \(total) steps up the stem turn anticlockwise from above")
    expect(total > 100 && bad == 0, "\(bad) steps turn the wrong way")
}

test("on the face of the pole towards the camera, the stem rises from left to right") {
    var good: Int = 0
    var total: Int = 0
    for t in testHours {
        let cam: Camera = camera(t)
        let pts: [V3] = woundStem(t)
        let toCam: V3 = simd_normalize(V3(cam.position.x, 0, cam.position.z))
        for i in 1..<pts.count {
            let a: V3 = pts[i - 1]
            let b: V3 = pts[i]
            guard simd_dot(V3(a.x, 0, a.z), toCam) > 0.5 * poleRadius,
                  simd_dot(V3(b.x, 0, b.z), toCam) > 0.5 * poleRadius else { continue }
            let pa: SIMD2<Float> = cam.project(a, width: 400, height: 520)
            let pb: SIMD2<Float> = cam.project(b, width: 400, height: 520)
            total += 1
            if pb.y < pa.y && pb.x > pa.x { good += 1 }
        }
    }
    print("        \(good) of \(total) near-face steps rise left to right")
    expect(total > 20 && good == total, "\(total - good) near-face steps rise right to left")
}

test("it is a right-handed helix: the torsion of the wound stem is positive everywhere") {
    var bad: Int = 0
    var total: Int = 0
    for t in testHours {
        let pts: [V3] = woundStem(t)
        for i in 3..<pts.count {
            let d1: V3 = pts[i - 2] - pts[i - 3]
            let d2: V3 = pts[i - 1] - pts[i - 2]
            let d3: V3 = pts[i] - pts[i - 1]
            total += 1
            if simd_dot(simd_cross(d1, d2), d3) <= 0 { bad += 1 }
        }
    }
    expect(total > 100 && bad == 0)
}

section("contact with the pole (step 45)")

func stemSurface(_ c: Chain, from: Int, to: Int) -> [V3] {
    var out: [V3] = []
    for i in max(from, 1)..<min(to, c.points.count) {
        let a: V3 = c.points[i - 1]
        let b: V3 = c.points[i]
        let t: V3 = simd_normalize(b - a)
        let e1: V3 = perpendicular(to: t)
        let e2: V3 = simd_cross(t, e1)
        for k in 0..<3 {
            let p: V3 = a + (b - a) * (Float(k) / 3)
            let r: Float = c.radii[i - 1]
            for j in 0..<24 {
                let th: Float = Float(j) / 24 * 2 * Float.pi
                out.append(p + (e1 * cos(th) + e2 * sin(th)) * r)
            }
        }
    }
    return out
}

test("the wound stem touches the pole and never enters it — on the distance function") {
    for t in [Float(0.6), 5.3] {
        let s: PlantScene = beanScene(t, mutant)
        guard let c = drawnStem(s) else { expect(false, "no stem"); return }
        let front: Int = stem(t, mutant).frontIndex
        let clampStart: Int = front - samplesPerTurn * Int(tighteningTurns + 0.5)
        guard let d = probe(s, stemSurface(c, from: 1, to: c.points.count), maskOf([.pole])) else { expect(false, "probe failed"); return }
        let deepest: Float = d.map { $0.x }.min()!
        guard let dc = probe(s, stemSurface(c, from: 1, to: clampStart), maskOf([.pole])) else { expect(false, "probe failed"); return }
        var worstGap: Float = 0
        var i: Int = 0
        while i + 24 <= dc.count {
            worstGap = max(worstGap, dc[i..<(i + 24)].map { $0.x }.min()!)
            i += 24
        }
        print(String(format: "        hour %.1f: deepest stem point %.3f mm into the pole; clamped turns: widest gap %.3f mm", t, -deepest, worstGap))
        expect(deepest > -0.05, "the stem enters the pole by \(-deepest) mm")
        expect(worstGap < 0.15, "the clamped stem stands \(worstGap) mm off the pole")
    }
}

test("the newest turn is laid loose and tightens onto the wood within a turn (Silk & Hubbard 1991; Isnard et al. 2009)") {
    let atFront: Float = windingRadius(0, mutant) - contactRadius(mutant)
    let turnBack: Float = windingRadius(2 * Float.pi * tighteningTurns, mutant) - contactRadius(mutant)
    expect(atFront > 1.0 && turnBack < 1e-4)
}

test("the spire is close at the front and opens to the mature pitch behind it (Darwin)") {
    let e: Float = 0.01
    let front: Float = (riseBehindFront(e) - riseBehindFront(0)) / e * 2 * Float.pi
    let behind: Float = (riseBehindFront(3 * Float.pi + e) - riseBehindFront(3 * Float.pi)) / e * 2 * Float.pi
    expect(abs(front - closeSpireFraction * maturePitch) < 0.5)
    expect(abs(behind - maturePitch) < 0.5)
}

section("growth (step 45)")

test("the free tip sweeps round the pole anticlockwise, one turn per winding turn, and never touches it") {
    func tipAngle(_ t: Float) -> Float {
        let p: V3 = freeShoot(t, mutant).last!
        return atan2(-p.z, p.x)
    }
    var total: Float = 0
    let steps: Int = 60
    for k in 0..<steps {
        var d: Float = tipAngle(gyreHours * Float(k + 1) / Float(steps)) - tipAngle(gyreHours * Float(k) / Float(steps))
        while d > Float.pi { d -= 2 * Float.pi }
        while d < -Float.pi { d += 2 * Float.pi }
        total += d
    }
    expect(abs(total - 2 * Float.pi) < 0.01, "tip turned \(total) rad")
}

test("leaves: one every internode, alternate, each three-quarters of a turn round from the last") {
    let t: Float = 3.1
    let ns: [Node] = nodes(t, mutant).filter { $0.wound && frontAngle(t) - Float($0.index) * nodeAngleStep > 2 * Float.pi }
    expect(ns.count >= 3)
    for i in 1..<ns.count {
        expect(ns[i].side == -ns[i - 1].side)
        let turn: Float = atan2(-ns[i].position.z, ns[i].position.x) - atan2(-ns[i - 1].position.z, ns[i - 1].position.x)
        expect(cos(turn - 1.5 * Float.pi) > 0.999, "not three-quarters of a turn apart")
    }
}

section("the loop")

test("a loop later the whole scene — plant, flowers, pods, ladybirds and their legs — is the same scene one loop higher (and so is the camera)") {
    var worst: Float = 0
    var beetles: Int = 0
    for t in testHours {
        let a: PlantScene = fullScene(t, mutant).shifted(by: loopRise)
        let b: PlantScene = fullScene(t + loopHours, mutant)
        beetles += a.beetles.count
        worst = max(worst, a.largestDifference(from: b))
        worst = max(worst, simd_distance(camera(t).position + V3(0, loopRise, 0), camera(t + loopHours).position))
    }
    print(String(format: "        largest difference %.5f mm (%d ladybirds compared)", worst, beetles))
    expect(worst < 0.01 && beetles > 0)
}

test("the film only runs forward: every frame later and higher up than the last, the last into the first included") {
    var bad: Int = 0
    var steps: [Float] = []
    for f in 0..<frameCount {
        let t0: Float = hourOf(frame: f, mutant: mutant)
        let t1: Float = f + 1 < frameCount ? hourOf(frame: f + 1, mutant: mutant) : hourOf(frame: 0, mutant: mutant) + loopHours
        if !(t1 > t0) || !(frontHeight(t1) > frontHeight(t0)) { bad += 1 }
        steps.append(t1 - t0)
    }
    let spread: Float = steps.max()! - steps.min()!
    print(String(format: "        %d backward steps; step %.4f h, spread %.6f h", bad, steps[0], spread))
    expect(bad == 0, "\(bad) frames run backwards")
    expect(spread < 1e-4, "the steps are uneven: the join would show")
}

test("rendered, the frame one loop on matches frame 0") {
    guard let r = box.renderer else { expect(false, "no renderer"); return }
    let w: Int = 120
    let h: Int = 156
    let t0: Float = hourOf(frame: 0, mutant: mutant)
    guard let a = try? r.render(fullScene(t0, mutant), camera: camera(t0), width: w, height: h, samples: 1, tubeMin: 0),
          let b = try? r.render(fullScene(t0 + loopHours, mutant), camera: camera(t0 + loopHours), width: w, height: h,
                                samples: 1, tubeMin: 0) else { expect(false, "render failed"); return }
    var worst: Float = 0
    var off: Int = 0
    for i in 0..<(w * h) {
        let d: Float = simd_reduce_max(simd_abs(a.frame.rgba[i] - b.frame.rgba[i]))
        worst = max(worst, d)
        if d > 2.0 / 255 { off += 1 }
    }
    print(String(format: "        largest pixel difference %.4f; %d of %d pixels differ by more than 2/255", worst, off, w * h))
    expect(worst < 0.05 && Float(off) < 0.02 * Float(w * h))
}

test("with the magnified inset drawn in, the frame one loop on still matches frame 0; and the inset always shows a ladybird, magnified") {
    guard let r = box.renderer else { expect(false, "no renderer"); return }
    let w: Int = 160
    let h: Int = 208
    let t0: Float = hourOf(frame: 0, mutant: mutant)
    func composed(_ t: Float) -> Frame? {
        let s: PlantScene = fullScene(t, mutant)
        guard let img = try? r.render(s, camera: camera(t), width: w, height: h, samples: 1, tubeMin: 0),
              let out = try? withInset(img.frame, t, scene: s, renderer: r, samples: 1) else { return nil }
        return out.frame
    }
    guard let a = composed(t0), let b = composed(t0 + loopHours) else { expect(false, "render failed"); return }
    var worst: Float = 0
    var off: Int = 0
    for i in 0..<(w * h) {
        let d: Float = simd_reduce_max(simd_abs(a.rgba[i] - b.rgba[i]))
        worst = max(worst, d)
        if d > 2.0 / 255 { off += 1 }
    }
    var shown: Int = 0
    var mags: [Float] = []
    for f in stride(from: 0, to: frameCount, by: 10) {
        let t: Float = hourOf(frame: f)
        let c = insetChoice(t, width: 480, height: 624)
        if !c.isEmpty { shown += 1 }
        let total: Float = c.reduce(0) { $0 + $1.weight }
        expect(c.isEmpty || abs(total - 1) < 1e-4)
        if let m = c.max(by: { $0.weight < $1.weight }) { mags.append(insetMagnification(m.bird, t, InsetLayout(width: 480, height: 624), height: 624)) }
    }
    print(String(format: "        with the inset: largest pixel difference a loop on %.4f (%d pixels over 2/255); inset shows a ladybird in %d of 20 frames, ×%.1f–%.1f",
                 worst, off, shown, mags.min() ?? 0, mags.max() ?? 0))
    expect(worst < 0.05 && Float(off) < 0.02 * Float(w * h))
    expectEqual(shown, 20)
}

section("the distance function is a distance")

test("outside every surface it never claims more room than there is, beyond what the ray allows — ladybirds and legs included") {
    let names: [String] = ["-", "pole", "stem", "petiole", "leaf", "tendril", "twine", "stipule", "petal", "pod", "elytra", "beetle", "leg"]
    var worst = [Float](repeating: 0, count: materialCount)
    var total: Int = 0
    for t in [Float(2.3), 6.1] {
        let s: PlantScene = fullScene(t, mutant)
        var rng = Lcg(state: 11)
        var a: [V3] = []
        var b: [V3] = []
        func add(_ p: V3, _ step: Float) {
            a.append(p)
            b.append(p + rng.unit() * step)
        }
        let yf: Float = frontHeight(t)
        for _ in 0..<150_000 { add(V3(-180 + 360 * rng.next(), yf - 380 + 520 * rng.next(), -180 + 360 * rng.next()), 0.25) }
        for l in s.leaflets {
            for _ in 0..<600 {
                add(l.origin + l.u * (l.length * (-0.1 + 1.2 * rng.next())) + l.v * (l.halfWidth * (2.4 * rng.next() - 1.2))
                    + l.w * (6 * rng.next() - 3), 0.25)
            }
        }
        // Densely round the ladybirds: bodies and legs, close in and further out.
        for bt in s.beetles {
            for _ in 0..<20_000 {
                let r: Float = 0.02 + 6 * rng.next() * rng.next()
                add(bt.origin + bt.up * 1.2 + rng.unit() * (3.8 * rng.next() + r), 0.05)
            }
        }
        for c in s.chains where c.material == .leg {
            for p in c.points { for _ in 0..<60 { add(p + rng.unit() * (0.1 + 1.5 * rng.next()), 0.05) } }
        }
        guard let da = probe(s, a), let db = probe(s, b) else { expect(false, "probe failed"); return }
        for i in 0..<a.count where da[i].y == db[i].y && da[i].x > 0 && db[i].x > 0 {
            let m: Int = Int(da[i].y)
            worst[m] = max(worst[m], abs(da[i].x - db[i].x) / simd_distance(a[i], b[i]))
        }
        total += a.count
    }
    var line: String = "        worst over-report:"
    for m in [1, 2, 3, 4, 8, 9, 10, 11, 12] { line += String(format: " %@ %.3f", names[m], worst[m]) }
    print(line + String(format: "; the ray allows %.2f (%d pairs)", 1 / stepScale, total))
    for m in 1..<materialCount { expect(worst[m] * stepScale <= 1.0, "\(names[m]) oversteps: \(worst[m])") }
    expect(worst[10] > 0.3 && worst[11] > 0.3, "the ladybirds were not sampled")
}

section("the picture")

test("frame 0 shows the pole, the stem, green leaves, white flowers, pods and ladybirds") {
    guard let r = box.renderer else { expect(false, "no renderer"); return }
    var counts = [Int](repeating: 0, count: materialCount)
    var n: Float = 0
    for f in [0, 100] {
        let t: Float = hourOf(frame: f)
        guard let out = try? r.render(fullScene(t, mutant), camera: camera(t), width: 240, height: 312, samples: 1, tubeMin: 0.35)
        else { expect(false, "render failed"); return }
        for s in out.frame.seen { counts[Int(s.x)] += 1 }
        n += Float(out.frame.seen.count)
    }
    func pc(_ m: Material) -> Float { Float(counts[m.rawValue]) / n * 100 }
    print(String(format: "        pole %.1f%%, stem %.1f%%, leaf %.1f%%, petals %.2f%%, pods %.2f%%, ladybirds %.3f%%",
                 pc(.pole), pc(.stem), pc(.leaf), pc(.petal), pc(.pod), pc(.elytra) + pc(.beetle)))
    expect(pc(.pole) > 3 && pc(.stem) > 0.5 && pc(.leaf) > 3)
    expect(pc(.petal) > 0.05 && pc(.pod) > 0.05, "flowers or pods missing")
    expect(pc(.elytra) + pc(.beetle) > 0.003, "no ladybird in the frame")
}

finish()
