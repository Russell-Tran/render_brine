// Tests for step 61, the toy pig come to life. The walk is checked by the
// shared animation tests (planted feet, rigid joints, no piece through
// another, the gait read off the drawn feet, the forward-only closed loop,
// the motion visible on screen); the toy is step 60's pig, and its cloven
// hooves are checked again mid-stride.
//
// TOY_MUTANT=sliding_feet|rewind|frozen|wrongGait breaks the step on purpose;
// `make mutants` requires the suite to fail for each.

import CoreGraphics
import Foundation
import Metal
import simd

let mutant: Mutant = activeMutant
if mutant != .none { print("MUTANT: \(mutant.rawValue)") }

let design: ToyDesign = pigAnimDesign(mutant)
var spec: WalkSpec = pigWalk
spec.offsets = gaitOffsets(mutant)
let perf = Performance(design: design, spec: spec, mutant: mutant)
let setup: AnimSetup = animSetup(perf)
let renderer: ToyRenderer? = {
    guard let d = gpu.device else { return nil }
    return try? ToyRenderer(device: d, width: 420, height: 280, paints: design.paints, studio: setup.studio, mutant: mutant)
}()

section("the toy, mid-stride, is still step 60's pig")
let midWalk: PosedToy = perf.posed(spec.walkStart + spec.walkSeconds * 0.37)
testHooves(renderer, toy: midWalk)
testSize(perf.design.posed(perf.design.restPose()), name: "pig")
testOptics(renderer, mutant: mutant)

section("the distance function, mid-stride")
testDistanceFunction(renderer, toy: midWalk, label: "mid-stride")

testAnimation(perf, setup: setup, renderer: renderer, mutant: mutant)

finish()
