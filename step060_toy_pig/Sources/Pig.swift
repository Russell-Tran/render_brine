// The toy pig: a white domestic pig, in the manner of the Large White, as a
// good toy makes one.
//
// THE ANIMAL. A domestic pig; the breed followed is the Large White, "a big,
// white pig" with "erect ears" and a "slightly dished face" (Wikipedia,
// "Large White pig", checked). Pigs walk on cloven hooves: "the two digits
// of cloven-hooved animals are homologous to the third and fourth fingers of
// the hand", two claws either side of a cleft, and pigs are among the
// animals named as having them (Wikipedia, "Cloven hoof", checked). The
// small dewclaws behind (digits II and V) are left off, as most toys of this
// size leave them — MODEL.
//
// PROPORTIONS are MODEL. No measured body dimensions for the breed were
// reached that fix the toy's shape (one sow study's "body length" of 87 cm
// for a 212 kg sow is plainly measured to some other landmark — Animals
// 16:72, 2026, PMC12784878 — and is not used). The shape follows the breed's
// described look: long, deep body; short legs; a long, straight-to-dished
// snout ending in a flat disc; erect ears; a curled tail.
//
// THE TOY is 56 mm long, snout to tail — inside Russell's 2–3 inch spec.

import Foundation
import simd

/// Every length in the pig's design, toy mm, is written at k = 1 and scaled
/// by this to bring the whole toy to its size. MODEL.
let pigScale: Float = 1.22

/// The rest pose's knee flexion. MODEL.
let pigRestFlexion: Float = 15 * .pi / 180

/// Paints. MODEL: the colours of a mass-made toy pig — pale pink body,
/// darker pink snout and ears, dark grey hooves, black eyes.
enum PigPaint: Int {
    case body = 0, snout, hoof, eye, nostril, innerEar
}
let pigPaints: [Paint] = [
    paint("body", 238, 190, 178, rough: 0.26),
    paint("snout", 222, 150, 146, rough: 0.28),
    paint("hoof", 86, 78, 74, rough: 0.28),
    paint("eye", 16, 14, 14, rough: 0.08),
    paint("nostril", 120, 64, 66, rough: 0.30),
    paint("innerEar", 214, 140, 140, rough: 0.30),
]

/// Hip-to-ankle distance for a two-piece leg flexed by `flexion`.
func legReach(_ a: Float, _ b: Float, flexion: Float) -> Float {
    let c: Float = cos(flexion)
    let s: Float = a * a + b * b + 2 * a * b * c
    return s.squareRoot()
}

/// A cloven hoof in the foot's level frame, origin at its centre on the
/// table: two claws, digits III and IV, either side of a cleft that is open
/// at the front and the sole and joined at the back where the claws meet the
/// pastern. The `wholeHoof` mutant makes it one claw. Sizes MODEL; the cleft
/// is widened to 0.45 mm so a mould can form it.
func clovenHoof(length: Float, width: Float, ankle: Float, legRadius: Float, paint hoof: Int, skin: Int,
                mutant: Mutant) -> [Prim] {
    var out: [Prim] = []
    let tipR: Float = width * 0.19
    let heelR: Float = width * 0.31
    let tipZ: Float = tipR + 0.225
    let heelZ: Float = width * 0.19
    // Each claw runs from a raised heel down to a tip on the table; both the
    // tip and the heel's underside touch the table.
    let heel = SIMD3<Float>(-length * 0.28, heelR, 0)
    let tip = SIMD3<Float>(length * 0.40, tipR, 0)
    if mutant == .wholeHoof {
        out.append(Prim.cone(heel * SIMD3(1, 1.45, 1), heelR * 1.45, tip * SIMD3(1, 1.6, 1), tipR * 1.6, paint: hoof).tagged(.toe))
    } else {
        for side in [Float(-1), 1] {
            out.append(Prim.cone(heel + SIMD3(0, 0, side * heelZ), heelR, tip + SIMD3(0, 0, side * tipZ), tipR, paint: hoof).tagged(.toe))
        }
    }
    // The pastern, from above the claws to the fetlock.
    out.append(Prim.cone(SIMD3(-length * 0.12, heelR * 2 + 0.35, 0), legRadius * 0.95, SIMD3(0, ankle, 0), legRadius, paint: skin))
    return out
}

func pigDesign(_ mutant: Mutant = activeMutant) -> ToyDesign {
    let k: Float = pigScale
    let bp: Int = PigPaint.body.rawValue
    func v(_ x: Float, _ y: Float, _ z: Float) -> SIMD3<Float> { SIMD3<Float>(x, y, z) * k }

    // Legs: short. Toy mm at k = 1: forearm or shank, then cannon; the
    // fetlock stands 2.3 up on the hoof. MODEL.
    let hindUpper: Float = 6.2 * k
    let hindLower: Float = 4.2 * k
    let foreUpper: Float = 5.8 * k
    let foreLower: Float = 4.2 * k
    let ankle: Float = 2.3 * k
    let hindReach: Float = legReach(hindUpper, hindLower, flexion: pigRestFlexion)
    let foreReach: Float = legReach(foreUpper, foreLower, flexion: pigRestFlexion)
    let height: Float = hindReach + ankle
    let shoulderY: Float = foreReach + ankle - height
    let shoulderX: Float = 17.0 * k

    var legs: [LegSpec] = []
    for (name, fore, side) in [("LF", true, Float(-1)), ("RF", true, Float(1)), ("LH", false, Float(-1)), ("RH", false, Float(1))] {
        let hip: SIMD3<Float> = fore ? SIMD3(shoulderX, shoulderY, side * 4.4 * k) : SIMD3(0, 0, side * 4.8 * k)
        let a: Float = fore ? foreUpper : hindUpper
        let b: Float = fore ? foreLower : hindLower
        let reach: Float = fore ? foreReach : hindReach
        let foot = SIMD3<Float>(hip.x, hip.y - reach - ankle, hip.z)
        let rTop: Float = (fore ? 2.3 : 2.7) * k
        let rMid: Float = 1.45 * k
        let rLow: Float = 1.15 * k
        let upper: [Prim] = [Prim.cone(.zero, rTop, SIMD3(0, -a, 0), rMid, paint: bp)]
        let lower: [Prim] = [Prim.cone(.zero, rMid, SIMD3(0, -b, 0), rLow, paint: bp)]
        let hoof: [Prim] = clovenHoof(length: 2.7 * k, width: 2.6 * k, ankle: ankle, legRadius: rLow, paint: PigPaint.hoof.rawValue,
                                      skin: bp, mutant: mutant)
        // The hoof's paint line: dark below, carried a little unevenly up the
        // pastern as a hand-painted edge is. Overspill MODEL.
        let hoofLine = Patch(c: SIMD3(0, 1.25 * k, 0), r: 0.09, paint: PigPaint.hoof.rawValue, kind: 1)
        legs.append(LegSpec(name: name, fore: fore, side: side, hip: hip, upper: a, lower: b, ankleHeight: ankle,
                            bendForward: fore, restFoot: foot, upperPrims: upper, lowerPrims: lower, footPrims: hoof,
                            footPatches: [hoofLine]))
    }

    // The body: long and deep, the back nearly level, round hams and
    // shoulders where the legs go in. MODEL shapes.
    let trunk: [Prim] = [
        Prim.cone(v(-4.0, 3.6, 0), 8.0 * k, v(14.0, 3.3, 0), 7.6 * k, paint: bp),
        Prim.cone(v(-1.5, 1.2, 0), 6.9 * k, v(12.5, 1.2, 0), 6.6 * k, paint: bp),
        Prim.ball(v(-2.0, 2.2, 3.1), 6.6 * k, paint: bp), Prim.ball(v(-2.0, 2.2, -3.1), 6.6 * k, paint: bp),
        Prim.ball(v(15.0, 2.2, 2.8), 6.1 * k, paint: bp), Prim.ball(v(15.0, 2.2, -2.8), 6.1 * k, paint: bp),
    ]
    let body = Segment(name: "body", part: .body, prims: trunk, blend: 2.2 * k, seam: true)

    // The head, turned at the neck: jowl, skull tapering to the snout, the
    // flat snout disc, erect ears.
    let headPivot: SIMD3<Float> = v(20.5, 4.4, 0)
    let snoutP: Int = PigPaint.snout.rawValue
    var headPrims: [Prim] = [
        Prim.ball(v(0.8, -1.2, 0), 5.9 * k, paint: bp),
        Prim.cone(v(1.0, 0.6, 0), 5.0 * k, v(9.0, -1.2, 0), 2.6 * k, paint: bp),
        Prim.box(v(11.2, -1.3, 0), half: SIMD3(0.55, 2.25, 2.55) * k, edge: 0.75 * k, paint: snoutP),
    ]
    for side in [Float(-1), 1] {
        let up: SIMD3<Float> = simd_normalize(SIMD3<Float>(0.35, 1, side * 0.42))
        let normal: SIMD3<Float> = simd_normalize(SIMD3<Float>(0.55, -0.1, side * 0.9))
        headPrims.append(Prim.plate(v(1.8, 4.6, side * 2.5) + up * (1.9 * k), up: up, normal: normal, halfHeight: 3.0 * k,
                                    halfWidth: 1.8 * k, halfThickness: 0.36 * k, edge: 0.2 * k, paint: bp))
    }
    let face: [Patch] = [
        Patch(c: v(4.6, 1.75, 3.72), r: 0.5 * k, paint: PigPaint.eye.rawValue, kind: 0),
        Patch(c: v(4.6, 1.75, -3.72), r: 0.5 * k, paint: PigPaint.eye.rawValue, kind: 0),
        Patch(c: v(11.85, -1.2, 0.95), r: 0.48 * k, paint: PigPaint.nostril.rawValue, kind: 0),
        Patch(c: v(11.85, -1.2, -0.95), r: 0.48 * k, paint: PigPaint.nostril.rawValue, kind: 0),
    ]
    let head = Segment(name: "head", part: .head, prims: headPrims, blend: 1.4 * k, seam: true, patches: face)

    // The tail: a short curl, one rigid piece.
    let tailPivot: SIMD3<Float> = v(-11.6, 6.0, 0)
    let curl: [SIMD3<Float>] = [v(0.4, 0, 0), v(-1.3, 0.7, 0), v(-1.8, 2.0, 0.5), v(-1.0, 2.7, 0.9), v(-0.4, 2.1, 0.7)]
    let radii: [Float] = [0.75, 0.62, 0.55, 0.48, 0.42].map { $0 * k }
    var tailPrims: [Prim] = []
    for i in 1..<curl.count { tailPrims.append(Prim.cone(curl[i - 1], radii[i - 1], curl[i], radii[i], paint: bp)) }
    let tail = Segment(name: "tail", part: .tail, prims: tailPrims, blend: 0.3 * k, seam: true)

    return ToyDesign(name: "pig", paints: pigPaints, body: body, head: head, headPivot: headPivot, tail: tail,
                     tailPivot: tailPivot, extra: [], legs: legs, bodyHeight: height, headFillet: 1.6 * k, legFillet: 1.3 * k)
}

func pigCaption(lengthMM: Float) -> Caption {
    Caption(
        title: "A plastic toy pig",
        lines: [
            "One piece of solid PVC, injection-moulded and painted; 2–3 inches long (this one \(Int(lengthMM.rounded())) mm).",
            "Its gloss is PVC's refractive index: n = 1.545 (Zhang et al. 2020) mirrors 4.6% of the light at normal incidence, more at grazing.",
            "A white domestic pig after the Large White: erect ears, a flat snout disc, a curled tail — and cloven hooves, two claws to a foot.",
            "Unbranded; the paint colours and the proportions are a model, not measured.",
        ],
        cutName: "left foreleg",
        seamSegment: .body,
        seamStart: SIMD3<Float>(4, 20, 0))
}
