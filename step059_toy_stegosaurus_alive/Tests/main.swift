// Tests for step 59, the toy Stegosaurus come to life. The walk is checked
// by the shared animation tests (planted feet, rigid joints, no piece
// through another, the gait read off the drawn feet, the forward-only closed
// loop, the motion visible on screen); the toy itself is step 58's, and a
// few of step 58's checks are repeated on it mid-stride.
//
// TOY_MUTANT=sliding_feet|rewind|frozen|wrongGait breaks the step on purpose;
// `make mutants` requires the suite to fail for each.

import CoreGraphics
import Foundation
import Metal
import simd

let mutant: Mutant = activeMutant
if mutant != .none { print("MUTANT: \(mutant.rawValue)") }

let design: ToyDesign = stegosaurusDesign(mutant)
var spec: WalkSpec = stegoWalk
spec.offsets = gaitOffsets(mutant)
let perf = Performance(design: design, spec: spec, mutant: mutant)
let setup: AnimSetup = animSetup(perf)
let renderer: ToyRenderer? = {
    guard let d = gpu.device else { return nil }
    return try? ToyRenderer(device: d, width: 420, height: 280, paints: design.paints, studio: setup.studio, mutant: mutant)
}()

section("the toy, mid-stride, is still step 58's Stegosaurus")

let midWalk: PosedToy = perf.posed(spec.walkStart + spec.walkSeconds * 0.37)

test("nineteen alternating plates and four tail spikes, as drawn mid-stride") {
    // Each plate's side is read in the frame of the piece it grows from (the
    // tail swings, carrying its plates with it), its order along the back
    // in the body's.
    var plates: [(along: Float, side: Float)] = []
    var spikes: Int = 0
    let body: Frame = midWalk.frames[0]
    for (i, s) in midWalk.segments.enumerated() {
        for p in s.prims {
            if p.tag == .plate { plates.append((simd_dot(midWalk.frames[i].toWorld(p.a) - body.o, body.x), p.a.z)) }
            if p.tag == .spike { spikes += 1 }
        }
    }
    let sorted = plates.sorted { $0.along > $1.along }
    var flips: Int = 0
    for k in 1..<sorted.count where sorted[k].side * sorted[k - 1].side < 0 { flips += 1 }
    expectEqual(plates.count, 19)
    expectEqual(flips, 18)
    expectEqual(spikes, 4)
}

testSize(perf.design.posed(perf.design.restPose()), name: "Stegosaurus")
testOptics(renderer, mutant: mutant)

section("the distance function, mid-stride")
testDistanceFunction(renderer, toy: midWalk, label: "mid-stride")

testAnimation(perf, setup: setup, renderer: renderer, mutant: mutant)

finish()
