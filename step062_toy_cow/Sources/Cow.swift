// The toy cow: a Holstein dairy cow, as a good toy makes one.
//
// THE ANIMAL. A Holstein Friesian: "black and white pied" (Wikipedia,
// "Holstein Friesian", infobox, checked; its cows 145–165 cm at the withers).
// The proportions are measured ones: fifty Holstein cows in Türkiye measured
// by tape — withers height 137.5 ± 2.7 cm, rump height 141.8 ± 2.8 cm, and
// body length, "the point of the shoulder [to] the lateral angle of the
// tuber ischii", 144.8 ± 7.7 cm (Paksoy, Erez & Selvi, "Comparison and
// agreement between traditional and smartphone-camera-based morphometric
// measurements in Holstein and Simmental cattle", *Vet Sci* 13:502, 2026,
// doi 10.3390/vetsci13050502, PMC13211479 — full text checked). The toy
// keeps those three in proportion: a dairy cow stands a little higher at
// the rump than the withers and is a little longer than tall.
//
// Cattle walk on cloven hooves — two claws, digits III and IV (Wikipedia,
// "Cloven hoof", checked; cattle named). A dairy cow has an udder with four
// teats. Holsteins are commonly dehorned; toys often keep short horns, and
// this one has two small horns — MODEL.
//
// THE TOY is those measurements at 1:33 — 43.9 mm shoulder to pin bone — and
// about 64 mm nose to rump, inside Russell's 2–3 inch spec.

import Foundation
import simd

/// Paksoy et al. 2026, Holstein means, cm.
let holsteinWithersCM: Float = 137.5
let holsteinRumpCM: Float = 141.8
let holsteinBodyLengthCM: Float = 144.8

/// The toy's scale. MODEL: chosen to bring the toy inside 2–3 inches.
let cowScale: Float = 1.0 / 33.0

/// The rest pose's flexion, fore (a nearly straight column) and hind (the
/// hock's angle, pointing back). MODEL, from how a standing cow looks.
let cowForeFlexion: Float = 6 * .pi / 180
let cowHindFlexion: Float = 38 * .pi / 180

/// Paints. MODEL: a mass-made toy Holstein's — white, black patches, a pink
/// muzzle and udder, dark hooves, pale horns.
enum CowPaint: Int {
    case white = 0, black, muzzle, hoof, horn, udder, eye
}
let cowPaints: [Paint] = [
    paint("white", 238, 236, 230, rough: 0.26),
    paint("black", 30, 30, 32, rough: 0.24),
    paint("muzzle", 206, 158, 154, rough: 0.30),
    paint("hoof", 52, 50, 50, rough: 0.26),
    paint("horn", 226, 214, 186, rough: 0.28),
    paint("udder", 232, 182, 172, rough: 0.30),
    paint("eye", 14, 12, 12, rough: 0.08),
]

func legReach(_ a: Float, _ b: Float, flexion: Float) -> Float {
    let c: Float = cos(flexion)
    let s: Float = a * a + b * b + 2 * a * b * c
    return s.squareRoot()
}

/// A cloven hoof, as step 60's pig has: two claws either side of a cleft
/// open at the front and the sole, joined behind at the pastern. The
/// `wholeHoof` mutant makes it one claw. The cleft is widened so a mould can
/// form it — MODEL.
func clovenHoof(length: Float, width: Float, ankle: Float, legRadius: Float, paint hoof: Int, skin: Int,
                mutant: Mutant) -> [Prim] {
    var out: [Prim] = []
    let tipR: Float = width * 0.19
    let heelR: Float = width * 0.31
    let tipZ: Float = tipR + 0.28
    let heelZ: Float = width * 0.19
    let heel = SIMD3<Float>(-length * 0.28, heelR, 0)
    let tip = SIMD3<Float>(length * 0.40, tipR, 0)
    if mutant == .wholeHoof {
        out.append(Prim.cone(heel * SIMD3(1, 1.45, 1), heelR * 1.45, tip * SIMD3(1, 1.6, 1), tipR * 1.6, paint: hoof).tagged(.toe))
    } else {
        for side in [Float(-1), 1] {
            out.append(Prim.cone(heel + SIMD3(0, 0, side * heelZ), heelR, tip + SIMD3(0, 0, side * tipZ), tipR, paint: hoof).tagged(.toe))
        }
    }
    out.append(Prim.cone(SIMD3(-length * 0.12, heelR * 2 + 0.4, 0), legRadius * 0.95, SIMD3(0, ankle, 0), legRadius, paint: skin))
    return out
}

/// Where the body's landmarks are, in the body frame (origin at the hind
/// leg's joint with the body, the stifle): from the three measurements.
struct CowFrame {
    let stifleHeight: Float     // the body frame's origin above the table
    let withers: SIMD2<Float>   // x, y of the top of the withers
    let rump: SIMD2<Float>      // x, y of the top of the rump
    let pin: SIMD2<Float>       // tuber ischii
    let shoulder: SIMD2<Float>  // point of the shoulder
}

func cowDesign(_ mutant: Mutant = activeMutant) -> ToyDesign {
    let s: Float = cowScale * 10     // cm to toy mm
    let white: Int = CowPaint.white.rawValue

    // Legs, toy mm. MODEL lengths, sized so the shoulder and rump heights
    // come out right: fore, forearm then cannon; hind, shank then cannon.
    let foreUpper: Float = 9.6
    let foreLower: Float = 7.0
    let hindUpper: Float = 11.4
    let hindLower: Float = 8.6
    let ankle: Float = 3.0
    let foreReach: Float = legReach(foreUpper, foreLower, flexion: cowForeFlexion)
    let hindReach: Float = legReach(hindUpper, hindLower, flexion: cowHindFlexion)
    let height: Float = hindReach + ankle
    let withersY: Float = holsteinWithersCM * s - height
    let rumpY: Float = holsteinRumpCM * s - height
    let bodyLength: Float = holsteinBodyLengthCM * s
    // The smooth blend (up to k/4 proud) and the hook bones lift the drawn
    // top line above the shapes' own tops; these offsets, found by measuring
    // the drawn toy (a test does), put the drawn withers and rump at the
    // measured heights.
    let rumpTop: Float = rumpY - 0.64
    let withersTop: Float = withersY - 1.15
    // The pin bones sit below the hooks: the rump slopes down to the tail.
    let pin = SIMD2<Float>(-7.5, rumpTop - 6.6)
    let shoulderPt = SIMD2<Float>(pin.x + (bodyLength * bodyLength - (pin.y - 9.5) * (pin.y - 9.5)).squareRoot(), 9.5)
    let elbow = SIMD3<Float>(shoulderPt.x - 4.2, foreReach + ankle - height, 0)

    var legs: [LegSpec] = []
    for (name, fore, side) in [("LF", true, Float(-1)), ("RF", true, Float(1)), ("LH", false, Float(-1)), ("RH", false, Float(1))] {
        let hip: SIMD3<Float> = fore ? SIMD3(elbow.x, elbow.y, side * 4.6) : SIMD3(0, 0, side * 5.6)
        let a: Float = fore ? foreUpper : hindUpper
        let b: Float = fore ? foreLower : hindLower
        let reach: Float = fore ? foreReach : hindReach
        let foot = SIMD3<Float>(hip.x, hip.y - reach - ankle, hip.z)
        let rTop: Float = fore ? 3.0 : 3.7
        let rMid: Float = fore ? 1.95 : 2.0
        let rLow: Float = 1.65
        let upper: [Prim] = [Prim.cone(.zero, rTop, SIMD3(0, -a, 0), rMid, paint: white)]
        let lower: [Prim] = [Prim.cone(.zero, rMid, SIMD3(0, -b, 0), rLow, paint: white)]
        let hoof: [Prim] = clovenHoof(length: 3.6, width: 3.6, ankle: ankle, legRadius: rLow, paint: CowPaint.hoof.rawValue,
                                      skin: white, mutant: mutant)
        let hoofLine = Patch(c: SIMD3(0, 1.45, 0), r: 0.1, paint: CowPaint.hoof.rawValue, kind: 1)
        legs.append(LegSpec(name: name, fore: fore, side: side, hip: hip, upper: a, lower: b, ankleHeight: ankle,
                            bendForward: fore, restFoot: foot, upperPrims: upper, lowerPrims: lower, footPrims: hoof,
                            footPatches: [hoofLine]))
    }

    // The body. Deep barrel; straight topline from withers to rump, the rump
    // a little higher; hook and pin bones showing, as a dairy cow's do; the
    // udder with four teats. MODEL shapes on the measured landmarks.
    var trunk: [Prim] = [   // SIMD3<Float> spelled out below: inferred, this literal took 5 ms on the mini
        Prim.cone(SIMD3<Float>(2.0, rumpTop - 10.4, 0), 10.4, SIMD3<Float>(shoulderPt.x - 7, withersTop - 11.2, 0), 11.2, paint: white),
        Prim.cone(SIMD3<Float>(pin.x + 1.5, pin.y + 0.5, 0), 5.4, SIMD3<Float>(2.0, rumpTop - 10.4, 0), 10.4, paint: white),
        Prim.ball(SIMD3<Float>(pin.x, pin.y, 2.6), 3.3, paint: white), Prim.ball(SIMD3<Float>(pin.x, pin.y, -2.6), 3.3, paint: white),
        Prim.ball(SIMD3<Float>(1.0, rumpTop - 2.6, 5.4), 2.8, paint: white), Prim.ball(SIMD3<Float>(1.0, rumpTop - 2.6, -5.4), 2.8, paint: white),
        Prim.ball(SIMD3<Float>(shoulderPt.x - 5.5, withersTop - 10.6, 0), 10.6, paint: white),
        Prim.ball(SIMD3<Float>(shoulderPt.x - 1.8, shoulderPt.y - 1.0, 0), 5.2, paint: white),
        // Neck, rising forward from the shoulders.
        Prim.cone(SIMD3<Float>(shoulderPt.x - 3.5, withersTop - 5.8, 0), 6.4, SIMD3<Float>(shoulderPt.x + 3.6, withersTop - 3.2, 0), 4.6, paint: white),
        // Where the legs go in.
        Prim.ball(SIMD3<Float>(0, 0.5, 4.6), 4.0, paint: white), Prim.ball(SIMD3<Float>(0, 0.5, -4.6), 4.0, paint: white),
        Prim.ball(SIMD3<Float>(elbow.x, elbow.y + 1.5, 3.6), 3.2, paint: white), Prim.ball(SIMD3<Float>(elbow.x, elbow.y + 1.5, -3.6), 3.2, paint: white),
        // The udder, forward of the hind legs, and its four teats.
        Prim.ball(SIMD3<Float>(3.2, -2.4, 0), 4.0, paint: CowPaint.udder.rawValue),
    ]
    for (dx, dz) in [(Float(1.4), Float(1.7)), (1.4, -1.7), (-1.4, 1.7), (-1.4, -1.7)] {
        let top = SIMD3<Float>(3.2 + dx, -4.8, dz)
        trunk.append(Prim.cone(top, 0.6, top + SIMD3<Float>(0, -1.9, 0), 0.42, paint: CowPaint.udder.rawValue).tagged(.teat))
    }
    // The pied coat: black blotches, as Holsteins are marked. MODEL placing.
    let black: Int = CowPaint.black.rawValue
    let coat: [Patch] = [
        Patch(c: SIMD3(14, withersTop - 2.0, -7.5), r: 6.0, paint: black, kind: 2),
        Patch(c: SIMD3(24, withersTop - 7.5, 8.0), r: 5.5, paint: black, kind: 2),
        Patch(c: SIMD3(4, rumpTop - 4.0, 8.0), r: 5.0, paint: black, kind: 2),
        Patch(c: SIMD3(-3, rumpTop - 2.0, -6.5), r: 4.8, paint: black, kind: 2),
        Patch(c: SIMD3(shoulderPt.x + 1.5, withersTop - 3.0, 0), r: 4.6, paint: black, kind: 2),
        Patch(c: SIMD3(9, withersTop - 0.8, 3.0), r: 3.6, paint: black, kind: 2),
    ]
    let body = Segment(name: "body", part: .body, prims: trunk, blend: 2.6, seam: true, patches: coat)

    // The head: long face sloping down to the muzzle, ears out to the sides,
    // small horns. Black with a white blaze. MODEL shapes.
    let headPivot = SIMD3<Float>(shoulderPt.x + 3.4, withersTop - 3.1, 0)
    var headPrims: [Prim] = [
        Prim.cone(SIMD3(0.8, 0.8, 0), 3.9, SIMD3(8.6, -3.8, 0), 2.6, paint: black),
        Prim.ball(SIMD3(9.3, -4.6, 0), 2.95, paint: CowPaint.muzzle.rawValue),
    ]
    for side in [Float(-1), 1] {
        let out: SIMD3<Float> = simd_normalize(SIMD3<Float>(-0.25, -0.15, side))
        headPrims.append(Prim.plate(SIMD3(1.6, 1.8, side * 2.6) + out * 2.4, up: out, normal: SIMD3(0.2, 1, 0), halfHeight: 2.5,
                                    halfWidth: 1.25, halfThickness: 0.38, edge: 0.2, paint: black))
        let hornBase = SIMD3<Float>(0.9, 3.9, side * 2.1)
        headPrims.append(Prim.cone(hornBase, 0.58, hornBase + SIMD3<Float>(0.35, 1.45, side * 0.65), 0.28,
                                   paint: CowPaint.horn.rawValue).tagged(.horn))
    }
    let face: [Patch] = [
        Patch(c: SIMD3(4.6, 1.4, 0), r: 2.3, paint: white, kind: 2),
        // Each eye painted dark on a white rim, so it shows on the black head.
        Patch(c: SIMD3(4.2, 0.0, 3.1), r: 0.85, paint: white, kind: 0),
        Patch(c: SIMD3(4.2, 0.0, -3.1), r: 0.85, paint: white, kind: 0),
        Patch(c: SIMD3(4.2, 0.0, 3.1), r: 0.52, paint: CowPaint.eye.rawValue, kind: 0),
        Patch(c: SIMD3(4.2, 0.0, -3.1), r: 0.52, paint: CowPaint.eye.rawValue, kind: 0),
    ]
    let head = Segment(name: "head", part: .head, prims: headPrims, blend: 1.0, seam: true, patches: face)

    // The tail: from the tail head above the pins, hanging to the hocks,
    // with its switch.
    let tailPivot = SIMD3<Float>(pin.x - 0.6, rumpTop - 1.8, 0)
    let tailPrims: [Prim] = [
        Prim.cone(SIMD3(0.3, 0.3, 0), 1.2, SIMD3(-1.6, -7.0, 0), 0.75, paint: white),
        Prim.cone(SIMD3(-1.6, -7.0, 0), 0.75, SIMD3(-1.7, -25.0, 0), 0.6, paint: white),
        Prim.cone(SIMD3(-1.7, -25.0, 0), 1.25, SIMD3(-1.6, -28.8, 0), 0.8, paint: white),
    ]
    let tail = Segment(name: "tail", part: .tail, prims: tailPrims, blend: 0.5, seam: true)

    return ToyDesign(name: "cow", paints: cowPaints, body: body, head: head, headPivot: headPivot, tail: tail,
                     tailPivot: tailPivot, extra: [], legs: legs, bodyHeight: height, headFillet: 1.4, legFillet: 1.4)
}

func cowCaption(lengthMM: Float) -> Caption {
    Caption(
        title: "A plastic toy cow",
        lines: [
            "One piece of solid PVC, injection-moulded and painted; 2–3 inches long (this one \(Int(lengthMM.rounded())) mm, a Holstein at 1:33).",
            "Its gloss is PVC's refractive index: n = 1.545 (Zhang et al. 2020) mirrors 4.6% of the light at normal incidence, more at grazing.",
            "A dairy cow in measured proportion (Paksoy et al. 2026): rump higher than withers, an udder with four teats, cloven hooves.",
            "Unbranded; the paint colours and the coat's patches are a model, not measured.",
        ],
        cutName: "left foreleg",
        seamSegment: .body,
        seamStart: SIMD3<Float>(10, 40, 0))
}
