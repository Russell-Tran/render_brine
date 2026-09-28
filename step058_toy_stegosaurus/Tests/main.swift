// Tests for step 58, the toy Stegosaurus. The anatomy is read off the toy
// as drawn — the shapes the kernel receives, in the pose it drew them — and
// checked against Maidment et al. (2015); the optics, the distance function,
// the parting line, the size and the picture by the shared tests.
//
// TOY_MUTANT=pairedPlates|dielectricZero|noSeam breaks the step on purpose;
// `make mutants` requires the suite to fail for each.

import CoreGraphics
import Foundation
import Metal
import simd

let mutant: Mutant = activeMutant
if mutant != .none { print("MUTANT: \(mutant.rawValue)") }

let design: ToyDesign = stegosaurusDesign(mutant)
let setup: StillSetup = stillSetup(design, mutant: mutant)
let toy: PosedToy = setup.toy
let renderer: ToyRenderer? = {
    guard let d = gpu.device else { return nil }
    return try? ToyRenderer(device: d, width: 1920, height: 1080, paints: design.paints, studio: setup.studio, mutant: mutant)
}()

/// Every shape with a tag, placed in the world, with the segment it is on.
func tagged(_ t: PrimTag) -> [(prim: Prim, segment: Int, centre: SIMD3<Float>)] {
    var out: [(Prim, Int, SIMD3<Float>)] = []
    for (i, s) in toy.segments.enumerated() {
        for p in s.prims where p.tag == t {
            let c: SIMD3<Float> = p.kind == .roundCone ? (p.a + p.b) / 2 : p.a
            out.append((p, i, toy.frames[i].toWorld(c)))
        }
    }
    return out.map { (prim: $0.0, segment: $0.1, centre: $0.2) }
}

let body: Frame = toy.frames[0]
func along(_ p: SIMD3<Float>) -> Float { simd_dot(p - body.o, body.x) }
func sideOf(_ p: SIMD3<Float>) -> Float { simd_dot(p - body.o, body.z) }

section("the animal, against Maidment et al. 2015 (NHMUK PV R36730)")

test("nineteen plates, in two rows that ALTERNATE: each plate on the other side from the last, none side by side") {
    let plates = tagged(.plate).sorted { along($0.centre) > along($1.centre) }
    expectEqual(plates.count, 19)
    var flips: Int = 0
    var closest: Float = .infinity
    for k in 1..<plates.count {
        let a: Float = sideOf(plates[k - 1].centre)
        let b: Float = sideOf(plates[k].centre)
        if a * b < 0 { flips += 1 }
        closest = min(closest, abs(along(plates[k - 1].centre) - along(plates[k].centre)))
    }
    let left: Int = plates.filter { sideOf($0.centre) < 0 }.count
    print(String(format: "        %d plates, %d left and %d right; side changes %d of %d; closest two along the back %.2f mm apart",
                 plates.count, left, plates.count - left, flips, plates.count - 1, closest))
    expectEqual(flips, plates.count - 1)
    expect(closest > 0.6, "two plates stand side by side (\(closest) mm apart along the back): paired, not staggered")
    expect(abs(left - (plates.count - left)) <= 1, "the two rows should be near equal")
}

test("the plates are drawn where they are placed: each plate's centre is inside the drawn toy (GPU)") {
    guard let r = renderer else { expect(false, "no GPU"); return }
    let pts: [SIMD3<Float>] = tagged(.plate).map { $0.centre }
    guard let d = try? r.probe(pts, toy: toy) else { expect(false, "probe failed"); return }
    expect(d.allSatisfy { $0.z < 0 }, "a plate's centre is outside the drawn toy")
}

test("the plates' heights are Sophie's at 1:80, and plate 13, over the hips, is the tallest") {
    let placed: [PlatePlacement] = stegoPlatePlacements(mutant)
    for (i, p) in placed.enumerated() {
        expect(abs(p.height / stegoScale - stegoPlateHeights[i]) < 1e-3, "plate \(i + 1)")
    }
    let tallest: Int = placed.indices.max { placed[$0].height < placed[$1].height } ?? -1
    expectEqual(tallest + 1, 13)
    let measured: [Float] = [103, 112, 126, 107, 127, 258, 352, 463, 490, 445, 535, 785, 536, 382, 175]
    expectEqual(stegoPlateHeights.enumerated().filter { stegoPlateHeightsMeasured[$0.offset] }.map { $0.element }, measured)
}

test("four spikes at the end of the tail, two each side, the rear pair shorter, behind the last plate") {
    let spikes = tagged(.spike)
    expectEqual(spikes.count, 4)
    let tailIndex: Int = toy.segments.firstIndex { $0.part == .tail } ?? -1
    expect(spikes.allSatisfy { $0.segment == tailIndex }, "a spike is not on the tail")
    let lastPlate: Float = tagged(.plate).map { along($0.centre) }.min() ?? 0
    expect(spikes.allSatisfy { along($0.centre) < lastPlate }, "the spikes should be behind every plate")
    expectEqual(spikes.filter { sideOf($0.prim.b) < sideOf($0.prim.a) }.count, 2)
    // Each spike's cone starts 0.4 mm inside the tail; its visible length is the rest.
    let lengths: [Float] = spikes.map { simd_distance($0.prim.a, $0.prim.b) - 0.4 }
    let sorted = spikes.sorted { along($0.centre) > along($1.centre) }
    let front: Float = simd_distance(sorted[0].prim.a, sorted[0].prim.b)
    let back: Float = simd_distance(sorted[3].prim.a, sorted[3].prim.b)
    print(String(format: "        spike lengths %@ mm (Sophie at 1:80: 4.72, 4.35)", lengths.map { String(format: "%.2f", $0) }.joined(separator: ", ")))
    expect(back < front, "the rear pair should be shorter")
    // Tips blunt, as a toy's must be.
    expect(spikes.allSatisfy { $0.prim.rb >= 0.2 }, "a spike tip is sharp")
}

test("forelimbs shorter than hind limbs, in Sophie's ratio: (humerus + ulna)/(femur + tibia) = 0.632") {
    let fore: LegSpec = design.legs.first { $0.fore }!
    let hind: LegSpec = design.legs.first { !$0.fore }!
    let ratio: Float = (fore.upper + fore.lower) / (hind.upper + hind.lower)
    let want: Float = (450 + 412) / (868 + 495)
    print(String(format: "        fore %.2f mm, hind %.2f mm: %.3f (want %.3f); shoulder %.1f mm up, hip %.1f mm",
                 fore.upper + fore.lower, hind.upper + hind.lower, ratio, want, fore.hip.y + design.bodyHeight, design.bodyHeight))
    expect(abs(ratio - want) < 0.005, "limb ratio \(ratio)")
    expect(fore.hip.y < -3, "the shoulders should sit well below the hips")
}

test("a small, low head, and a tail held clear of the ground") {
    let len = toy.extent(along: body.x)
    let head: Int = toy.segments.firstIndex { $0.part == .head }!
    let tail: Int = toy.segments.firstIndex { $0.part == .tail }!
    let one = PosedToy(segments: [toy.segments[head]], frames: [toy.frames[head]], paints: toy.paints, seams: false)
    let headLen: Float = one.extent(along: body.x).hi - one.extent(along: body.x).lo
    let headTop: Float = one.extent(along: SIMD3<Float>(0, 1, 0)).hi
    let tailLow: Float = PosedToy(segments: [toy.segments[tail]], frames: [toy.frames[tail]], paints: toy.paints, seams: false)
        .extent(along: SIMD3<Float>(0, 1, 0)).lo
    print(String(format: "        head %.1f mm long (%.0f%% of the toy), top %.1f mm up; tail's lowest point %.1f mm up",
                 headLen, headLen / (len.hi - len.lo) * 100, headTop, tailLow))
    expect(headLen / (len.hi - len.lo) < 0.12, "the head is too big")
    expect(headTop < design.bodyHeight, "the head should be carried below the hips")
    expect(tailLow > 8, "the tail should be held well clear of the ground")
}

test("the toy is Sophie at 1:80: 5.5–6 m long on the animal") {
    let len = toy.extent(along: body.x)
    let animal: Float = (len.hi - len.lo) / stegoScale
    print(String(format: "        %.1f mm × 80 = %.2f m", len.hi - len.lo, animal / 1000))
    expect(animal >= 5500 && animal <= 6000, "\(animal) mm")
}

testSize(toy, name: "Stegosaurus")
testOptics(renderer, mutant: mutant)

section("the distance function")
testDistanceFunction(renderer, toy: toy, label: "the still")

let lowerLF: Int = toy.segments.firstIndex { $0.part == .lowerLeg && $0.limb == 0 }!
let lowerRH: Int = toy.segments.firstIndex { $0.part == .lowerLeg && $0.limb == 3 }!
let upperLH: Int = toy.segments.firstIndex { $0.part == .upperLeg && $0.limb == 2 }!
testSeam(toy, stations: [
    (0, SIMD3<Float>(6, -8, 0), SIMD3<Float>(0, -1, 0), "belly"),
    (1, SIMD3<Float>(3, -2, 0), SIMD3<Float>(0, -1, 0), "jaw"),
    (2, SIMD3<Float>(-10, -4, 0), SIMD3<Float>(0, -1, 0), "tail"),
    (lowerLF, SIMD3<Float>(1, -3, 0), SIMD3<Float>(1, 0, 0), "LF shin"),
    (lowerRH, SIMD3<Float>(1.5, -3, 0), SIMD3<Float>(1, 0, 0), "RH shin"),
    (upperLH, SIMD3<Float>(2, -5, 0), SIMD3<Float>(1, 0, 0), "LH thigh"),
])

let len = toy.extent(along: SIMD3<Float>(1, 0, 0))
testStillPicture(renderer, setup: setup, caption: stegoCaption(lengthMM: len.hi - len.lo), mutant: mutant)

finish()
