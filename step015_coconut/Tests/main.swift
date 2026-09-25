// Tests for step 15.
//
// Three different things are on trial here.
//
// The first is the CUT. Capping is the whole technique of this step and it is
// the kind of thing that looks right in a still and is quietly wrong one plane
// over: a ray that starts inside a solid at the clip plane must come back as
// cut material, never as background and never as the inside of a far wall. So
// the plane is swept through the entire nut on the CPU, exhaustively, rather
// than judged by eye on the one plane that ships.
//
// The second is the BOTANY. Every claim the picture makes about a coconut is a
// claim about a real plant, and four of them are ones a bean-germination mental
// model gets wrong. A taproot, an inverted shell, a haustorium that grows out
// of nothing and a pinnate first leaf each have a test, and `make mutants`
// makes each of those mistakes on purpose to prove the test catches it:
//
//   COCO_MUTATE=taproot   one root three times the length of the rest
//   COCO_MUTATE=invert    the testa outside the endocarp
//   COCO_MUTATE=budget    the haustorium swells and the endosperm never thins
//   COCO_MUTATE=pinnate   a split frond, months too early
//   COCO_MUTATE=mailbox   a primitive counted once per grid box it touches
//
// The third is the TRAVERSAL, which is step 13's and is reused rather than
// rewritten — so what is tested here is that the reuse is real, and that the
// mailbox still does what step 13 said it does. Step 13 learned the hard way
// that with per-tissue interval unions a duplicated interval merges straight
// back into itself, so turning the mailbox off leaves the picture BIT
// IDENTICAL. The budget is therefore checked against a CPU reference count,
// not against the image — and this suite asserts the image really does not
// move, so that the lesson is on the record rather than in a comment.

import CoreGraphics
import Foundation
import Metal
import simd

let breakage = Breakage.fromEnvironment()

func near(_ a: Float, _ b: Float, within e: Float) -> Bool { abs(a - b) <= e }
func near(_ a: Double, _ b: Double, within e: Double) -> Bool { abs(a - b) <= e }

/// A small frame, for the tests that need the GPU. The pixel scale is the
/// render's own, so a narrower frame simply sees less of the field.
let testWidth = 320
let testViewHeight = testWidth * frameViewHeight / frameWidth
let testLayout = FrameLayout(width: testWidth, viewHeight: testViewHeight,
                             captionHeight: testWidth * frameCaptionHeight / frameWidth)
let testCamera: Camera = cutawayCamera(width: testWidth, height: testViewHeight,
                                       target: cutawayTarget, yaw: cameraYaw)
let testLook = Look()

let device: MTLDevice? = try? findDevice()
let renderer: CutawayRenderer? = {
    guard let d = device else { return nil }
    return try? CutawayRenderer(device: d)
}()
let testFrame: MTLBuffer? = device?.makeBuffer(
    length: testWidth * (testViewHeight + testLayout.captionHeight) * 4,
    options: .storageModeShared)

func renderTest(_ f: Int, plane: Float = cutPlaneZ, grid: Bool = true,
                samples: Int = 2) throws -> Plant {
    guard let r = renderer, let buffer = testFrame else { throw RenderError.noMetalDevice }
    let plant = poseCoconut(frame: f, breakage: breakage)
    let albedo = albedoTable(stages: plant.stages)
    if grid { try r.buildGrid(plant.prims, density: gridDensity) }
    _ = try r.render(prims: plant.prims, albedo: albedo, camera: testCamera, look: testLook,
                     cutZ: plane, into: buffer, width: testWidth, viewHeight: testViewHeight,
                     samplesPerSide: samples, useGrid: grid, breakage: breakage)
    return plant
}

func framePixels() -> [UInt8] {
    guard let b = testFrame else { return [] }
    let p = b.contents().assumingMemoryBound(to: UInt8.self)
    return Array(UnsafeBufferPointer(start: p, count: testWidth * testViewHeight * 4))
}

// MARK: - the traversal this step reuses

section("the traversal this step reuses")

test("the spliced prefix is step 13's traversal and nothing else of step 13's") {
    for name in ["ellipsoidInterval", "capsuleInterval", "primInterval", "addInterval",
                 "inline void gather", "struct Prim", "struct Grid", "MAX_IV"] {
        expect(traversalPrefix.contains(name), "the prefix has lost \(name)")
    }
    // Step 13's own shading must NOT come along: its integrator turns the
    // interval list into optical depth, which is the one thing this step does
    // differently with it.
    for name in ["kernel void render", "inline float3 integrate", "exp(-tau)"] {
        expect(!traversalPrefix.contains(name), "the prefix still carries \(name)")
    }
    // And the closest-approach form, which is the part step 13 learned the hard
    // way and the part a rewrite would silently lose.
    expect(traversalPrefix.contains("float tc = -dot(o, d) / a;"),
           "the ellipsoid intersection is no longer the closest-approach form")
    expect(!traversalPrefix.contains("b * b - 4.0"), "a textbook discriminant has appeared")
    expect(cutawayKernelSource.hasPrefix(traversalPrefix))
    expect(cutawayKernelSource.contains("kernel void cutaway"))
}

test("the kernel compiles, with fast math off") {
    expect(device != nil, "no Metal device")
    expect(renderer != nil, "the cutaway kernel did not compile")
}

// MARK: - the cut is capped

section("the cut is capped")

/// Rays over the nut's whole screen extent, at the test frame's own scale.
func sweepRays(count: Int) -> [(SIMD3<Float>, SIMD3<Float>)] {
    var out: [(SIMD3<Float>, SIMD3<Float>)] = []
    for j in 0..<count {
        for i in 0..<count {
            let sx: Float = Float(i) + 0.5
            let sy: Float = Float(j) + 0.5
            let u: Float = sx / Float(count) * Float(testWidth)
            let v: Float = sy / Float(count) * Float(testViewHeight)
            out.append(testCamera.ray(sx: u, sy: v, width: testWidth, height: testViewHeight))
        }
    }
    return out
}

test("capping is watertight: sweep the plane through the whole nut") {
    // The claim: if the point where a ray crosses the clip plane is inside any
    // solid at all, the renderer returns CUT MATERIAL there — at the plane, of
    // the innermost material present — and never background, never a surface
    // further along, never the inside of a far wall.
    let plant = poseCoconut(frame: 0, breakage: breakage)
    let rays = sweepRays(count: 42)
    var inside = 0
    var wrongDepth = 0
    var wrongMaterial = 0
    var missed = 0
    for step in 0...18 {
        let z: Float = -95 + Float(step) * 190 / 18
        for (o, d) in rays {
            let tCut: Float = (z - o.z) / d.z
            if tCut < 0 { continue }
            let q: SIMD3<Float> = o + d * tCut
            guard let want = cpuMaterialAt(q, prims: plant.prims) else { continue }
            inside += 1
            guard let hit = cpuResolve(origin: o, direction: d, prims: plant.prims, cutZ: z) else {
                missed += 1
                continue
            }
            if !hit.isCut || abs(hit.t - tCut) > 1e-3 { wrongDepth += 1 }
            if hit.material != want { wrongMaterial += 1 }
        }
    }
    expect(inside > 6_000, "only \(inside) rays started inside a solid — the sweep missed the nut")
    expectEqual(missed, 0)
    expectEqual(wrongDepth, 0)
    expectEqual(wrongMaterial, 0)
}

test("capping holds once the plant is grown too") {
    let plant = poseCoconut(frame: coconutFrameCount - 30, breakage: breakage)
    let rays = sweepRays(count: 22)
    var inside = 0
    var bad = 0
    for step in 0...8 {
        let z: Float = -90 + Float(step) * 180 / 8
        for (o, d) in rays {
            let tCut: Float = (z - o.z) / d.z
            if tCut < 0 { continue }
            let q: SIMD3<Float> = o + d * tCut
            guard let want = cpuMaterialAt(q, prims: plant.prims) else { continue }
            inside += 1
            guard let hit = cpuResolve(origin: o, direction: d, prims: plant.prims, cutZ: z) else {
                bad += 1
                continue
            }
            if !hit.isCut || abs(hit.t - tCut) > 1e-3 || hit.material != want { bad += 1 }
        }
    }
    expect(inside > 800, "only \(inside) rays started inside a solid")
    expectEqual(bad, 0)
}

test("no backfaces: every uncut surface faces the camera") {
    let plant = poseCoconut(frame: 100, breakage: breakage)
    var uncut = 0
    var backfacing = 0
    for (o, d) in sweepRays(count: 60) {
        guard let hit = cpuResolve(origin: o, direction: d, prims: plant.prims, cutZ: cutPlaneZ),
              !hit.isCut else { continue }
        uncut += 1
        let n: SIMD3<Float> = cpuSurfaceNormal(origin: o, direction: d, t: hit.t,
                                               material: hit.material, prims: plant.prims)
        if simd_dot(n, d) > 1e-4 { backfacing += 1 }
    }
    expect(uncut > 100, "only \(uncut) rays found an uncut surface")
    expectEqual(backfacing, 0)
}

test("the GPU resolves the cut exactly where the CPU says it does") {
    guard let r = renderer else { return }
    _ = try renderTest(92, samples: 1)
    let plant = poseCoconut(frame: 92, breakage: breakage)
    var checked = 0
    var disagreed = 0
    for y in stride(from: 4, to: testViewHeight, by: 17) {
        for x in stride(from: 4, to: testWidth, by: 13) {
            let px: Float = Float(x) + 0.5
            let py: Float = Float(y) + 0.5
            let (o, d) = testCamera.ray(sx: px, sy: py, width: testWidth, height: testViewHeight)
            let want = cpuResolve(origin: o, direction: d, prims: plant.prims, cutZ: cutPlaneZ)
            let got = r.auxel(atX: x, y: y, width: testWidth)
            checked += 1
            let wantMat: Int = want?.material.rawValue ?? -1
            if got.material != wantMat { disagreed += 1; continue }
            if let w = want, w.isCut != got.isCut { disagreed += 1 }
        }
    }
    expect(checked > 400, "only \(checked) pixels checked")
    expectEqual(disagreed, 0)
}

// MARK: - the nut

section("the nut")

/// Is the ellipsoid `inner` strictly inside `outer`? Sampled on the inner
/// surface, which is the only place the two can touch.
func contained(inner semi: SIMD3<Float>, at centre: SIMD3<Float>,
               outer: SIMD3<Float>, at outerCentre: SIMD3<Float>, samples: Int = 400) -> Float {
    var worst: Float = 0
    for k in 0..<samples {
        let a: Float = Float(k) * 2.39996323
        let v: Float = -1 + 2 * (Float(k) + 0.5) / Float(samples)
        let r: Float = (1 - v * v).squareRoot()
        let u = SIMD3<Float>(r * cos(a), r * sin(a), v)
        let p: SIMD3<Float> = centre + semi * u
        let rel: SIMD3<Float> = (p - outerCentre) / outer
        worst = max(worst, simd_length(rel))
    }
    return worst
}

test("the shells nest, at every frame") {
    for f in stride(from: 0, to: coconutFrameCount, by: 3) {
        let plant = poseCoconut(frame: f, breakage: breakage)
        for i in 0..<(layerStack.count - 1) {
            let outer: SIMD3<Float> = layerSemi(layerStack[i], budget: plant.budget,
                                                breakage: breakage)
            let inner: SIMD3<Float> = layerSemi(layerStack[i + 1], budget: plant.budget,
                                                breakage: breakage)
            let worst: Float = contained(inner: inner, at: nutCentre,
                                         outer: outer, at: nutCentre, samples: 120)
            expect(worst < 1,
                   "frame \(f): \(layerStack[i + 1].name) is not inside \(layerStack[i].name) "
                   + "(reaches \(worst) of the way out)")
            // And strictly: each layer has a real thickness everywhere.
            for k in 0..<3 {
                expect(inner[k] < outer[k] - 0.2,
                       "frame \(f): \(layerStack[i].name) has no thickness on axis \(k)")
            }
        }
    }
}

test("the haustorium never leaves the cavity") {
    for f in stride(from: 0, to: coconutFrameCount, by: 2) {
        let plant = poseCoconut(frame: f, breakage: breakage)
        if plant.budget.fill <= 1e-5 { continue }
        let worst: Float = contained(inner: plant.haustoriumSemi, at: plant.haustoriumCentre,
                                     outer: plant.budget.cavitySemi, at: nutCentre, samples: 260)
        expect(worst < 1, "frame \(f): the haustorium reaches \(worst) of the cavity's wall")
    }
}

test("exactly one of the three pores is ever opened, and the shell is otherwise whole") {
    // Walk the endocarp's whole outer surface at the last frame and ask, at
    // every point, whether anything belonging to the sprout is there. Every
    // such point must lie within one pore's radius of ONE pore.
    let plant = poseCoconut(frame: coconutFrameCount - 25, breakage: breakage)
    let through: [GPUPrim] = plant.prims.filter {
        $0.material == .petiole || $0.material == .shoot || $0.material == .root
    }
    expect(!through.isEmpty, "nothing came out of the nut at all")
    var breached = [Int](repeating: 0, count: poreCount)
    var offPore = 0
    let samples = 20_000
    for k in 0..<samples {
        let a: Float = Float(k) * 2.39996323
        let v: Float = -1 + 2 * (Float(k) + 0.5) / Float(samples)
        let r: Float = (1 - v * v).squareRoot()
        let u = SIMD3<Float>(r * cos(a), r * sin(a), v)
        let p: SIMD3<Float> = nutCentre + endocarpSemi * u
        var hit = false
        for prim in through where cpuContains(prim, point: p) { hit = true; break }
        if !hit { continue }
        var nearest = -1
        var best: Float = .greatestFiniteMagnitude
        for i in 0..<poreCount {
            let d: Float = simd_distance(p, poreCentre(i))
            if d < best { best = d; nearest = i }
        }
        if best <= poreRadius * 2.0 { breached[nearest] += 1 } else { offPore += 1 }
    }
    expectEqual(offPore, 0)
    let opened: Int = breached.filter { $0 > 0 }.count
    expectEqual(opened, 1)
    expect(breached[functionalPore] > 0, "the soft eye is not the one that opened")
}

test("the nut is a closed budget, and the haustorium is mostly absorbed water") {
    let interior: Double = ellipsoidVolume(endospermSemi)
    var previousLost: Double = -1
    for f in 0..<coconutFrameCount {
        let plant = poseCoconut(frame: f, breakage: breakage)
        let b = plant.budget
        // Solid endosperm + coconut water + haustorium fill the seed exactly.
        let total: Double = b.endospermVolume + b.waterVolume + b.haustoriumVolume
        let error: Double = abs(total - interior) / interior
        expect(error < 1e-9, "frame \(f): the interior is \(total), not \(interior)")
        expect(b.waterVolume >= -1e-6, "frame \(f): negative water, \(b.waterVolume)")

        // Nothing grows out of nothing: once the haustorium is past five per
        // cent of the cavity, the meat it is eating has measurably gone.
        let lost: Double = b.cavityVolume - budgetAtStart.cavityVolume
        if b.haustoriumVolume > 0.05 * budgetAtStart.cavityVolume {
            expect(lost > 0, "frame \(f): the haustorium grew and no endosperm was digested")
            let share: Double = lost / b.haustoriumVolume
            expect(share > 0.18 && share < 0.34,
                   "frame \(f): \(share) of the haustorium came from digested meat, "
                   + "which is outside the stated conversion band")
        }
        expect(lost >= previousLost - 1e-6, "frame \(f): the endosperm grew back")
        previousLost = lost
    }
    // And the number that corrects the usual telling: by the end the haustorium
    // has gained about four times the volume of meat that was digested, because
    // most of its bulk is coconut water it absorbed.
    expect(near(haustoriumBulkRatio, 3.98, within: 0.08), "×\(haustoriumBulkRatio)")
    let sum: Double = endospermLost + waterAbsorbed
    expect(near(sum, budgetAtEnd.haustoriumVolume, within: 1.0),
           "\(sum) mm³ accounted for against \(budgetAtEnd.haustoriumVolume) mm³ of haustorium")
    expect(waterAbsorbed > 2.5 * endospermLost,
           "the water is supposed to be the bulk of it: \(waterAbsorbed) vs \(endospermLost)")
}

// MARK: - the plant

section("the plant")

test("there is no taproot") {
    for f in stride(from: dormantFrames + 30, to: coconutFrameCount, by: 5) {
        let plant = poseCoconut(frame: f, breakage: breakage)
        if plant.rootLengths.isEmpty { continue }
        expectEqual(plant.rootLengths.count, primaryRoots)
        let mean: Float = plant.meanRootLength
        let longest: Float = plant.maxRootLength
        expect(longest <= mean * 1.4,
               "frame \(f): the longest root is \(longest / mean)× the mean — that is a taproot")
        // And it really is a fan: no root is a stub either.
        let shortest: Float = plant.rootLengths.min() ?? 0
        expect(shortest >= mean * 0.6, "frame \(f): shortest root is \(shortest / mean)× the mean")
    }
}

test("the roots grow around the husk, not through it") {
    let plant = poseCoconut(frame: coconutFrameCount - 20, breakage: breakage)
    var worst: Float = 0
    for prim in plant.prims where prim.material == .root {
        for end in [SIMD3<Float>(prim.r0.x, prim.r0.y, prim.r0.z),
                    SIMD3<Float>(prim.r1.x, prim.r1.y, prim.r1.z)] {
            let rel: SIMD3<Float> = (end - nutCentre) / exocarpSemi
            worst = max(worst, 1 - simd_length(rel))
        }
    }
    expect(worst < 0.02, "a root endpoint sits \(worst) of the way inside the husk")
}

test("the first true leaf is entire, and no pinnate leaf ever appears") {
    for f in stride(from: 0, to: coconutFrameCount, by: 4) {
        let plant = poseCoconut(frame: f, breakage: breakage)
        expectEqual(plant.leafletCount, leafletCountEntire)
    }
    // "Entire" as a number: the half-width profile is positive everywhere
    // between the base and the tip, and has exactly one hump. A pinnate leaf
    // has one zero per gap between leaflets and one hump per leaflet.
    let n = 400
    var zeros = 0
    var humps = 0
    var previous: Float = bladeHalfWidth(0.5 / Float(n))
    var rising = true
    for k in 1..<n {
        let s: Float = (Float(k) + 0.5) / Float(n)
        let w: Float = bladeHalfWidth(s)
        if w <= 1e-4 { zeros += 1 }
        if rising && w < previous { humps += 1; rising = false }
        if !rising && w > previous { rising = true }
        previous = w
    }
    expectEqual(zeros, 0)
    expectEqual(humps, 1)
    // Lance-shaped: widest nearer the base than the tip.
    var widest: Float = 0
    var at: Float = 0
    for k in 0...200 {
        let s: Float = Float(k) / 200
        let w: Float = bladeHalfWidth(s)
        if w > widest { widest = w; at = s }
    }
    expect(at > 0.25 && at < 0.55, "the blade is widest at \(at) along, which is not lanceolate")
}

test("nothing ever runs backwards") {
    // The one test that catches a rewind. A dissolve is not a rewind, so the
    // geometry must be monotone across ALL 144 frames, dissolve included.
    var shoot: Float = -1
    var roots: Float = -1
    var haustorium: Double = -1
    var blade: Float = -1
    for f in 0..<coconutFrameCount {
        let plant = poseCoconut(frame: f, breakage: breakage)
        expect(plant.shootTipY >= shoot - 1e-4, "frame \(f): the shoot shrank")
        expect(plant.maxRootLength >= roots - 1e-4, "frame \(f): the roots shrank")
        expect(plant.budget.haustoriumVolume >= haustorium - 1e-6,
               "frame \(f): the haustorium shrank")
        expect(plant.bladeLength >= blade - 1e-4, "frame \(f): the leaf shrank")
        shoot = max(shoot, plant.shootTipY)
        roots = max(roots, plant.maxRootLength)
        haustorium = max(haustorium, plant.budget.haustoriumVolume)
        blade = max(blade, plant.bladeLength)
    }
    // Circumnutation is real motion and it must NOT be what makes the shoot
    // rise: it is in x and z only, so the wobble cannot be mistaken for growth
    // and growth cannot be mistaken for wobble.
    for f in 0..<nutationFrames {
        let offset: SIMD3<Float> = nutationOffset(arc: 1, frame: f)
        expectEqual(offset.y, 0)
    }
    let a: SIMD3<Float> = nutationOffset(arc: 1, frame: 0)
    let b: SIMD3<Float> = nutationOffset(arc: 1, frame: nutationFrames / 4)
    expect(simd_distance(a, b) > 1, "the tip is not actually nutating")
}

// MARK: - the loop

section("the loop")

test("the loop closes: the last dissolved frame is frame 0, byte for byte") {
    guard renderer != nil, let buffer = testFrame else { return }
    expectEqual(dissolve(frame: coconutFrameCount - 1), 1)
    for f in 0..<(coconutFrameCount - 1) {
        expect(dissolve(frame: f) < 1, "frame \(f) dissolves fully before the end")
    }
    _ = try renderTest(0)
    let dormant: [UInt8] = framePixels()
    _ = try renderTest(coconutFrameCount - 1)
    let grown: [UInt8] = framePixels()
    expect(dormant != grown, "the last growing frame already looks like the dormant nut")
    crossDissolve(frame: buffer, toward: dormant, alpha: dissolve(frame: coconutFrameCount - 1),
                  pixels: testWidth * testViewHeight)
    expectEqual(framePixels(), dormant)
}

test("the dissolve only ever goes one way, and the growth is flat under it") {
    var previous: Float = 0
    for f in 0..<coconutFrameCount {
        let a: Float = dissolve(frame: f)
        expect(a >= previous, "frame \(f): the dissolve went backwards")
        previous = a
    }
    expectEqual(growth(frame: dormantFrames), 0)
    expectEqual(growth(frame: dormantFrames + growFrames), 1)
    for f in dissolveStartFrame..<coconutFrameCount { expectEqual(growth(frame: f), 1) }
    expectEqual(coconutFrameCount, 144)
    expect(near(loopDisplayedSeconds, 10.08, within: 1e-9), "\(loopDisplayedSeconds) s")
    let held: Double = Double(holdFrames * frameDelayCentiseconds) / 100
    expect(held > 1.7 && held < 2.3, "the finished seedling is held for \(held) s")
}

// MARK: - the renderer

section("the renderer")

test("the grid is rebuilt every frame and matches brute force, pixel for pixel") {
    guard renderer != nil else { return }
    for f in [20, 60, 118] {
        _ = try renderTest(f, grid: true)
        let withGrid: [UInt8] = framePixels()
        _ = try renderTest(f, grid: false)
        let brute: [UInt8] = framePixels()
        var differing = 0
        for i in 0..<min(withGrid.count, brute.count) where withGrid[i] != brute[i] {
            differing += 1
        }
        expectEqual(differing, 0)
    }
    // And the grid really is rebuilt: the primitive count changes as the plant
    // grows, so a grid built once would be the wrong size by the end.
    let early = poseCoconut(frame: 12, breakage: breakage).prims.count
    let late = poseCoconut(frame: 110, breakage: breakage).prims.count
    expect(late > early + 100, "\(early) primitives became \(late) — the scene barely grew")
}

test("the interval budget is what the mailbox says it is") {
    // The CPU reference meets every primitive exactly once, because it has no
    // grid to meet it in twice. With mailboxing on the GPU must agree exactly.
    // With it off the same primitive is met once per grid box it straddles —
    // and the per-material union merges the duplicates straight back into
    // itself, so the PICTURE does not move at all. That is why this is checked
    // against a count and not against an image.
    guard let r = renderer else { return }
    let plant = try renderTest(60, samples: 1)
    var checked = 0
    var mismatch = 0
    var gpuTotal = 0
    var cpuTotal = 0
    for y in stride(from: 6, to: testViewHeight, by: 23) {
        for x in stride(from: 6, to: testWidth, by: 19) {
            let px: Float = Float(x) + 0.5
            let py: Float = Float(y) + 0.5
            let (o, d) = testCamera.ray(sx: px, sy: py, width: testWidth, height: testViewHeight)
            guard let want = cpuResolve(origin: o, direction: d, prims: plant.prims,
                                        cutZ: cutPlaneZ) else { continue }
            let got = r.auxel(atX: x, y: y, width: testWidth)
            checked += 1
            gpuTotal += got.met
            cpuTotal += want.intervals
            if got.met != want.intervals { mismatch += 1 }
        }
    }
    expect(checked > 80, "only \(checked) pixels checked")
    expectEqual(mismatch, 0)
    expect(gpuTotal == cpuTotal, "\(gpuTotal) intervals met against \(cpuTotal) on the CPU")
}

test("switching the mailbox off does not move a single pixel") {
    // Step 13's lesson, on the record. This is why the budget test above exists
    // in the form it does: an image comparison would pass either way.
    guard renderer != nil else { return }
    _ = try renderTest(60)
    let normal: [UInt8] = framePixels()
    guard let r = renderer, let buffer = testFrame else { return }
    let plant = poseCoconut(frame: 60, breakage: breakage)
    try r.buildGrid(plant.prims, density: gridDensity)
    _ = try r.render(prims: plant.prims, albedo: albedoTable(stages: plant.stages),
                     camera: testCamera, look: testLook, cutZ: cutPlaneZ, into: buffer,
                     width: testWidth, viewHeight: testViewHeight, samplesPerSide: 2,
                     useGrid: true, breakage: [.mailbox])
    expectEqual(framePixels(), normal)
}

test("no ray ever overflows its interval list, at the size that ships") {
    // At the size that ships, and not at the test frame's size: overflow is a
    // question of how many rays land in the crowded places, and a third-scale
    // frame simply misses them. This test was added because the shipping render
    // overflowed a quarter of a million times while the small frame said zero.
    guard let r = renderer, let d = device else { return }
    let full: MTLBuffer? = d.makeBuffer(length: frameWidth * frameHeight * 4,
                                        options: .storageModeShared)
    guard let buffer = full else { return }
    let camera: Camera = cutawayCamera(width: frameWidth, height: frameViewHeight,
                                       target: cutawayTarget, yaw: cameraYaw)
    var overflow: UInt32 = 0
    for f in stride(from: 0, to: coconutFrameCount, by: 3) {
        let plant = poseCoconut(frame: f, breakage: breakage)
        try r.buildGrid(plant.prims, density: gridDensity)
        _ = try r.render(prims: plant.prims, albedo: albedoTable(stages: plant.stages),
                         camera: camera, look: testLook, cutZ: cutPlaneZ, into: buffer,
                         width: frameWidth, viewHeight: frameViewHeight,
                         samplesPerSide: 2, useGrid: true, breakage: breakage)
        overflow += r.overflowCount
    }
    expectEqual(overflow, UInt32(0))
}

test("the scene stays inside the mailbox") {
    let plant = poseCoconut(frame: coconutFrameCount - 1, breakage: breakage)
    expect(plant.prims.count <= mailboxCapacity,
           "\(plant.prims.count) primitives against a \(mailboxCapacity)-bit mailbox")
    expect(plant.prims.count > 250, "only \(plant.prims.count) primitives at full growth")
}

test("the output is portrait, 960 × 1280") {
    expectEqual(frameWidth, 960)
    expectEqual(frameHeight, 1280)
    expectEqual(frameViewHeight + frameCaptionHeight, frameHeight)
    expect(frameHeight > frameWidth, "this step is portrait")
    // And the test frame keeps the same aspect to within a pixel of rounding,
    // so nothing above assumed a particular width.
    let wanted: Double = Double(frameViewHeight) / Double(frameWidth)
    let got: Double = Double(testViewHeight) / Double(testWidth)
    expect(near(got, wanted, within: 0.005), "the test frame is \(got), the render \(wanted)")
    let full = FrameLayout(width: frameWidth, viewHeight: frameViewHeight,
                           captionHeight: frameCaptionHeight)
    expectEqual(full.height, 1280)
}

test("one hundred millimetres of scale rule is one hundred millimetres of coconut") {
    let rule: Float = 100 / mmPerPixel
    expect(near(rule, 200, within: 0.01), "the 100 mm rule is \(rule) px")
    let a: SIMD2<Float> = testCamera.project(SIMD3(0, 0, 0), width: testWidth,
                                             height: testViewHeight)
    let b: SIMD2<Float> = testCamera.project(SIMD3(0, 100, 0), width: testWidth,
                                             height: testViewHeight)
    expect(near(simd_distance(a, b), rule, within: 0.01),
           "the camera makes 100 mm \(simd_distance(a, b)) px")
}

// MARK: - the table

section("the constants table")

test("every constant says what kind of claim it is") {
    expect(coconutConstants.count >= 15, "\(coconutConstants.count) constants is a thin table")
    for c in coconutConstants where c.evidence == .model {
        expect(c.source.hasPrefix("MODEL"), "\(c.name) is a model but does not say so")
    }
    for c in coconutConstants where c.evidence == .derived {
        expect(c.source.hasPrefix("DERIVED"), "\(c.name) is derived but does not say so")
    }
    // A MEASURED constant must cite something with a year in it. This is the
    // test the brief asked for: a measured number that cites nothing fails.
    for c in coconutConstants where c.evidence == .measured {
        let cited: Bool = c.source.contains("19") || c.source.contains("20")
        expect(cited, "\(c.name) claims to be measured but cites nothing: \"\(c.source)\"")
        expect(c.source.count > 20, "\(c.name)'s source is too short to be a citation")
    }
    expect(coconutConstants.contains { $0.evidence == .measured })
    expect(coconutConstants.contains { $0.evidence == .model })
    expect(coconutConstants.contains { $0.evidence == .derived })
}

test("the table really is the table the render uses") {
    func value(_ name: String) -> Double? {
        coconutConstants.first { $0.name == name }?.value
    }
    expectEqual(value("germination pores"), Double(poreCount))
    expectEqual(value("functional pores"), 1)
    expectEqual(value("root system"), Double(primaryRoots))
    expectEqual(value("first true leaf"), Double(leafletCountEntire))
    expectEqual(value("meat thickness, mature nut"), Double(meatThicknessStart))
    expectEqual(value("shell thickness"), Double(endocarpThickness))
    expectEqual(value("circumnutation period"), Double(nutationFrames))
    expectEqual(value("cavity volume, dormant"), budgetAtStart.cavityVolume / 1000)
    expectEqual(value("haustorium bulk per mL of meat"), haustoriumBulkRatio)
    // The plant really does have the number of roots the table claims.
    let plant = poseCoconut(frame: coconutFrameCount - 1, breakage: breakage)
    expectEqual(plant.counts["root"], primaryRoots)
    expectEqual(plant.counts["germination pore"], poreCount)
    expectEqual(plant.counts["layer"], layerStack.count)
}

test("the cut plane really does pass through the soft eye's channel") {
    // The off-centre cut is a claim: it goes through the functional pore and
    // NOT through the middle of the embryo. Both halves are checked, because
    // getting the first without the second is how you end up with a bisected
    // embryo, and getting the second without the first buries the whole beat
    // under fifty millimetres of coir.
    let pore: SIMD3<Float> = poreCentre(functionalPore)
    expect(near(pore.z, cutPlaneZ, within: 0.01), "the plane misses the soft eye: \(pore.z)")
    for k in 1..<poreCount {
        expect(abs(poreCentre(k).z - cutPlaneZ) > poreRadius * 0.4,
               "pore \(k) sits on the plane too")
    }
    let poreDir: SIMD3<Float> = simd_normalize(pore - nutCentre)
    let embryo: SIMD3<Float> = nutCentre + poreDir * 64
    let offset: Float = abs(embryo.z - cutPlaneZ)
    expect(offset > 1.0 && offset < 3.4,
           "the plane is \(offset) mm off the embryo's centre — it should graze, not bisect")
    expect(offset < 3.4, "the embryo would be missed entirely")
    // And the plane is genuinely off the nut's own axis.
    expect(cutPlaneZ > 12, "the cut is not off-centre at all: z = \(cutPlaneZ)")
}

test("the caption bar fits inside a 960-pixel frame") {
    let layout = FrameLayout(width: frameWidth, viewHeight: frameViewHeight,
                             captionHeight: frameCaptionHeight)
    let k = layout.scale
    let left: CGFloat = 18 * k
    let right: CGFloat = CGFloat(layout.width) - 18 * k
    expect(left + textWidth(coconutTitle, size: 16 * k, bold: true) < right, "the title runs off")
    for f in [0, 20, 60, 92, 130] {
        let plant = poseCoconut(frame: f, breakage: breakage)
        let caption: Caption = coconutCaption(frame: f, plant: plant)
        let w: CGFloat = textWidth(caption.subtitle, size: 12 * k, bold: false)
        expect(left + w < right,
               "\"\(caption.subtitle)\" runs to \(left + w) in a \(layout.width) px frame")
        let aw: CGFloat = textWidth(caption.aside, size: 11.5 * k, bold: false)
        expect(aw < CGFloat(layout.width) * 0.42, "the aside is \(aw) px wide")
    }
    expect(left + textWidth(coconutFacts, size: 11.5 * k, bold: false) < right,
           "the facts line runs off")
    // Three lines, and the last one has to sit inside the bar.
    let bottom: CGFloat = 54 * k + 11.5 * k
    expect(bottom < CGFloat(layout.captionHeight), "the last line reaches \(bottom)")
}

finish()
