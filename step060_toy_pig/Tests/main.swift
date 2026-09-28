// Tests for step 60, the toy pig. The anatomy is read off the toy as drawn
// — the shapes the kernel receives, and the kernel's own distance function
// probed across each foot — and the optics, the distance function, the
// parting line, the size and the picture by the shared tests.
//
// TOY_MUTANT=wholeHoof|dielectricZero|noSeam breaks the step on purpose;
// `make mutants` requires the suite to fail for each.

import CoreGraphics
import Foundation
import Metal
import simd

let mutant: Mutant = activeMutant
if mutant != .none { print("MUTANT: \(mutant.rawValue)") }

let design: ToyDesign = pigDesign(mutant)
let setup: StillSetup = stillSetup(design, mutant: mutant)
let toy: PosedToy = setup.toy
let renderer: ToyRenderer? = {
    guard let d = gpu.device else { return nil }
    return try? ToyRenderer(device: d, width: 1920, height: 1080, paints: design.paints, studio: setup.studio, mutant: mutant)
}()
let body: Frame = toy.frames[0]

section("the animal: a domestic pig")

testHooves(renderer, toy: toy)

test("erect ears: each ear stands up, its tip above the top of the head") {
    let head: Int = toy.segments.firstIndex { $0.part == .head }!
    let f: Frame = toy.frames[head]
    let ears: [Prim] = toy.segments[head].prims.filter { $0.kind == .plate }
    expectEqual(ears.count, 2)
    let skullTop: Float = toy.segments[head].prims.filter { $0.kind != .plate }.map { primExtent($0, SIMD3<Float>(0, 1, 0)).1 }.max() ?? 0
    for e in ears {
        let up: SIMD3<Float> = f.dirToWorld(e.u)
        let tip: Float = primExtent(e, SIMD3<Float>(0, 1, 0)).1
        print(String(format: "        ear leans %.0f° from upright; tip %.1f mm above the skull's top", acos(up.y) * 180 / .pi, tip - skullTop))
        expect(up.y > 0.8, "an ear droops: its long axis is \(acos(up.y) * 180 / .pi)° from upright")
        expect(tip > skullTop + 2, "an ear's tip is not above the head")
    }
}

test("a flat snout disc is the most forward part of the toy") {
    let e = toy.extent(along: body.x)
    let head: Int = toy.segments.firstIndex { $0.part == .head }!
    let disc: Prim? = toy.segments[head].prims.first { $0.kind == .roundBox && $0.paint == PigPaint.snout.rawValue }
    expect(disc != nil, "no snout disc")
    if let d = disc {
        let one = PosedToy(segments: [Segment(name: "disc", part: .head, prims: [d], blend: 0, seam: false)], frames: [toy.frames[head]],
                           paints: toy.paints, seams: false)
        expect(abs(one.extent(along: body.x).hi - e.hi) < 1e-3, "the snout disc is not the front of the toy")
    }
}

test("a curled tail: the tail's centreline turns through more than half a circle") {
    let tail: Int = toy.segments.firstIndex { $0.part == .tail }!
    let ps: [Prim] = toy.segments[tail].prims
    var turned: Float = 0
    for i in 1..<ps.count {
        let a: SIMD3<Float> = simd_normalize(ps[i - 1].b - ps[i - 1].a)
        let b: SIMD3<Float> = simd_normalize(ps[i].b - ps[i].a)
        turned += acos(min(max(simd_dot(a, b), -1), 1))
    }
    print(String(format: "        the tail turns %.0f°", turned * 180 / .pi))
    expect(turned > .pi, "the tail is not curled")
}

test("pig proportions (MODEL, held to what was chosen): short legs, long deep body") {
    let len = toy.extent(along: body.x)
    let tall = toy.extent(along: SIMD3<Float>(0, 1, 0))
    // The body's underside: the lowest point of the body piece.
    let belly: Float = PosedToy(segments: [toy.segments[0]], frames: [toy.frames[0]], paints: toy.paints, seams: false)
        .extent(along: SIMD3<Float>(0, 1, 0)).lo
    let backTop: Float = PosedToy(segments: [toy.segments[0]], frames: [toy.frames[0]], paints: toy.paints, seams: false)
        .extent(along: SIMD3<Float>(0, 1, 0)).hi
    print(String(format: "        %.1f mm long; back %.1f mm up, belly %.1f mm up (%.0f%% of the back's height); length / back height %.2f",
                 len.hi - len.lo, backTop, belly, belly / backTop * 100, (len.hi - len.lo) / backTop))
    expect(belly / backTop > 0.25 && belly / backTop < 0.45, "legs out of proportion")
    // A pig is long for its height: nose to rump about twice the height at
    // the back. The band is MODEL; it only holds the toy to a pig's build.
    expect((len.hi - len.lo) / backTop > 1.7 && (len.hi - len.lo) / backTop < 2.6, "the body is out of a pig's proportion")
    _ = tall
}

testSize(toy, name: "pig")
testOptics(renderer, mutant: mutant)

section("the distance function")
testDistanceFunction(renderer, toy: toy, label: "the still")

let lowerLF: Int = toy.segments.firstIndex { $0.part == .lowerLeg && $0.limb == 0 }!
let lowerRH: Int = toy.segments.firstIndex { $0.part == .lowerLeg && $0.limb == 3 }!
testSeam(toy, stations: [
    (0, SIMD3<Float>(6, -6, 0), SIMD3<Float>(0, -1, 0), "belly"),
    (0, SIMD3<Float>(4, 12, 0), SIMD3<Float>(0, 1, 0), "back"),
    (1, SIMD3<Float>(6, -3, 0), SIMD3<Float>(0, -1, 0), "jaw"),
    (lowerLF, SIMD3<Float>(1.5, -3, 0), SIMD3<Float>(1, 0, 0), "LF shin"),
    (lowerRH, SIMD3<Float>(1.5, -3, 0), SIMD3<Float>(-1, 0, 0), "RH shin, behind"),
])

let len = toy.extent(along: SIMD3<Float>(1, 0, 0))
testStillPicture(renderer, setup: setup, caption: pigCaption(lengthMM: len.hi - len.lo), mutant: mutant)

finish()
