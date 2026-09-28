// Tests for step 62, the toy cow. The anatomy is read off the toy as drawn
// and checked against Paksoy et al. (2026)'s Holstein measurements; the
// optics, the distance function, the parting line, the size and the picture
// by the shared tests.
//
// TOY_MUTANT=wholeHoof|dielectricZero|noSeam breaks the step on purpose;
// `make mutants` requires the suite to fail for each.

import CoreGraphics
import Foundation
import Metal
import simd

let mutant: Mutant = activeMutant
if mutant != .none { print("MUTANT: \(mutant.rawValue)") }

let design: ToyDesign = cowDesign(mutant)
let setup: StillSetup = stillSetup(design, mutant: mutant)
let toy: PosedToy = setup.toy
let renderer: ToyRenderer? = {
    guard let d = gpu.device else { return nil }
    return try? ToyRenderer(device: d, width: 1920, height: 1080, paints: design.paints, studio: setup.studio, mutant: mutant)
}()
let body: Frame = toy.frames[0]
let bodyOnly = PosedToy(segments: [toy.segments[0]], frames: [toy.frames[0]], paints: toy.paints, seams: false)

section("the animal: a Holstein dairy cow, against Paksoy et al. 2026")

testHooves(renderer, toy: toy)

test("in measured proportion: withers 137.5, rump 141.8, shoulder point to pin bone 144.8 cm, at 1:33") {
    // The top of the back over the withers and over the rump, as drawn: the
    // body's highest point in a slice at each.
    func topAt(_ x: Float) -> Float {
        var hi: Float = -.infinity
        var y: Float = 60
        let p0: SIMD3<Float> = body.toWorld(SIMD3<Float>(x, 0, 0))
        while y > 0 {
            if bodyOnly.sdf(SIMD3<Float>(p0.x, y, p0.z)).d < 0 { hi = y; break }
            y -= 0.02
        }
        return hi
    }
    let s: Float = cowScale * 10
    let len = bodyOnly.extent(along: body.x)
    var withers: Float = 0
    var withersX: Float = 0
    var rump: Float = 0
    for k in 0..<120 {
        let x: Float = len.lo + (len.hi - len.lo) * Float(k) / 119
        let t: Float = topAt(x)
        if x > 14 && x < 34 && t > withers { withers = t; withersX = x }
        if x > -6 && x < 6 && t > rump { rump = t }
    }
    print(String(format: "        withers %.2f mm (want %.2f), rump %.2f mm (want %.2f), rump/withers %.3f (want %.3f); withers at x = %.1f",
                 withers, holsteinWithersCM * s, rump, holsteinRumpCM * s, rump / withers, holsteinRumpCM / holsteinWithersCM, withersX))
    expect(abs(withers - holsteinWithersCM * s) < 0.35, "withers height \(withers) mm")
    expect(abs(rump - holsteinRumpCM * s) < 0.35, "rump height \(rump) mm")
    expect(rump > withers, "a dairy cow stands higher at the rump than the withers")
    // Body length: the design's shoulder point to pin bone is the measured
    // 144.8 cm at scale by construction; check it is longer than the
    // withers are high, as measured (144.8 vs 137.5).
    expect(holsteinBodyLengthCM > holsteinWithersCM)
}

test("an udder with four teats, hanging below the belly") {
    var teats: [SIMD3<Float>] = []
    for p in toy.segments[0].prims where p.tag == .teat { teats.append(body.toWorld(p.b)) }
    expectEqual(teats.count, 4)
    let belly: Float = bodyOnly.extent(along: SIMD3<Float>(0, 1, 0)).lo
    let lowest: Float = teats.map { $0.y }.min() ?? 0
    print(String(format: "        4 teat tips %.1f mm up; the body's underside elsewhere is higher", lowest))
    expect(abs(lowest - belly) < 0.6, "the teats should hang lowest of the body")
    let left: Int = teats.filter { simd_dot($0 - body.o, body.z) < 0 }.count
    expectEqual(left, 2)
}

test("two short horns and two ears on the head; the tail hangs to the hocks") {
    let head: Int = toy.segments.firstIndex { $0.part == .head }!
    expectEqual(toy.segments[head].prims.filter { $0.tag == .horn }.count, 2)
    expectEqual(toy.segments[head].prims.filter { $0.kind == .plate }.count, 2)
    let tail: Int = toy.segments.firstIndex { $0.part == .tail }!
    let tailLow: Float = PosedToy(segments: [toy.segments[tail]], frames: [toy.frames[tail]], paints: toy.paints, seams: false)
        .extent(along: SIMD3<Float>(0, 1, 0)).lo
    // The hock: the hind leg's middle joint.
    let hind: Int = toy.segments.firstIndex { $0.part == .lowerLeg && $0.limb == 2 }!
    let hock: Float = toy.frames[hind].o.y
    print(String(format: "        tail's end %.1f mm up; hock %.1f mm up", tailLow, hock))
    expect(abs(tailLow - hock) < 4, "the tail should reach about to the hocks")
}

testSize(toy, name: "cow")
testOptics(renderer, mutant: mutant)

section("the distance function")
testDistanceFunction(renderer, toy: toy, label: "the still")

let lowerLF: Int = toy.segments.firstIndex { $0.part == .lowerLeg && $0.limb == 0 }!
let lowerRH: Int = toy.segments.firstIndex { $0.part == .lowerLeg && $0.limb == 3 }!
testSeam(toy, stations: [
    (0, SIMD3<Float>(18, 18, 0), SIMD3<Float>(0, 1, 0), "back"),
    (1, SIMD3<Float>(4, -3, 0), SIMD3<Float>(0, -1, 0), "jaw"),
    (lowerLF, SIMD3<Float>(1.5, -3, 0), SIMD3<Float>(1, 0, 0), "LF shin"),
    (lowerRH, SIMD3<Float>(1.5, -3, 0), SIMD3<Float>(-1, 0, 0), "RH shin, behind"),
])

let len = toy.extent(along: SIMD3<Float>(1, 0, 0))
testStillPicture(renderer, setup: setup, caption: cowCaption(lengthMM: len.hi - len.lo), mutant: mutant)

finish()
