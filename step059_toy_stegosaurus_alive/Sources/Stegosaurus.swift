// The toy Stegosaurus: a Stegosaurus stenops as a good toy makes one, from
// the most complete skeleton there is.
//
// THE ANIMAL. NHMUK PV R36730 ("Sophie"), described by Maidment, Brassey &
// Barrett, "The postcranial skeleton of an exceptionally complete individual
// of the plated dinosaur Stegosaurus stenops (Dinosauria: Thyreophora) from
// the Upper Jurassic Morrison Formation of Wyoming, U.S.A.", *PLOS ONE*
// 10(10):e0138352 (2015), doi 10.1371/journal.pone.0138352 — full text
// checked. What it fixes here, quoted:
//   * "Eighteen plates and four spines were recovered in place along the
//     spine ... it is likely that the full complement of plates for
//     Stegosaurus was 19, with four spines at the end of the tail."
//   * the plates are "staggered in NHMUK PV R36730 and other individuals of
//     Stegosaurus" (against Dacentrurus, where "the plates are clearly
//     paired") — two rows, ALTERNATING.
//   * "NHMUK PV R36730 is 5.5–6 m in length"; "the right femur ... is 86.8 cm
//     long".
//   * Table 4, plate heights (mm): 103, —, 112, 126, 107, 127, 258, 352, 463,
//     490, 445, 535, 785 (plate 13, "the largest in the series"), plate 14
//     not recovered, 536, 382, —, 175, —.
//   * Table 5, spikes: anterior pair 380 and 375 mm tall, posterior pair 345
//     and 351 mm; "the posterior spines are slightly shorter".
//   * Table 3, limb bones (mm): humerus 450, ulna 412, femur 868 (right),
//     tibia 495 (left). These four were read off the extracted table by
//     column order; the femur is confirmed by the text, the other three
//     are UNVERIFIED readings.
// Wikipedia's "Stegosaurus" (checked) adds, citing the literature: "the fore
// limbs were much shorter than the stocky hind limbs" and "the tail appears
// to have been held well clear of the ground". The head is small and low.
//
// THE TOY is that animal at 1:80 — 5.75 m (the middle of Sophie's 5.5–6 m)
// to 72 mm — inside Russell's 2–3 inch spec. Where a moulded toy must
// differ from the animal it says so: plates and spikes thicker than true
// scale (a mould cannot fill a 0.3 mm sheet), and every tip blunted.

import Foundation
import simd

/// The scale: Sophie's 5.5–6 m, taken at 5.75 m, to the toy's millimetres.
let sophieLengthMM: Float = 5750
let stegoScale: Float = 1.0 / 80.0

/// Limb bones, mm on the animal (Maidment et al. 2015, Table 3 — see above).
let stegoHumerus: Float = 450
let stegoUlna: Float = 412
let stegoFemur: Float = 868
let stegoTibia: Float = 495

/// Plate heights on the animal, mm, Table 4, in order from the neck. Where the
/// table has none (plates 2, 14, 17 and 19 poorly preserved or missing) the
/// value is interpolated from the neighbours — MODEL — and flagged here.
let stegoPlateHeights: [Float] = [103, 108, 112, 126, 107, 127, 258, 352, 463, 490, 445, 535, 785, 650, 536, 382, 280, 175, 120]
let stegoPlateHeightsMeasured: [Bool] = [true, false, true, true, true, true, true, true, true, true, true, true, true,
                                         false, true, true, false, true, false]

/// Spike lengths on the animal, mm: the mean of each pair in Table 5.
let stegoSpikeLengths: [Float] = [377.5, 377.5, 348, 348]

// MARK: - the toy's parts, mm

// The body frame: origin at the hip joint, x forward, y up, z to the right.

/// The rest pose's knee and elbow flexion: nearly straight, as a graviportal
/// animal stands and as such a toy is moulded. MODEL.
let stegoRestFlexion: Float = 18 * .pi / 180

/// Paints. MODEL, all of them: the colours a mass-made toy dinosaur of this
/// kind is commonly painted — a green-grey body, rust plates, pale spikes.
enum StegoPaint: Int {
    case body = 0, plates, spikes, eye, claw, belly
}
let stegoPaints: [Paint] = [
    paint("body", 104, 124, 70, rough: 0.24),
    paint("plates", 178, 94, 44, rough: 0.24),
    paint("spikes", 226, 212, 176, rough: 0.28),
    paint("eye", 18, 16, 14, rough: 0.08),
    paint("claw", 120, 112, 88, rough: 0.30),
    paint("belly", 176, 176, 122, rough: 0.26),
]

/// Hip-to-ankle distance for a two-piece leg flexed by `flexion` at its
/// middle joint.
func legReach(_ a: Float, _ b: Float, flexion: Float) -> Float {
    let c: Float = cos(flexion)
    let s: Float = a * a + b * b + 2 * a * b * c
    return s.squareRoot()
}

/// One plate as placed on the toy: which of Sophie's plates it is, its
/// visible height and fore-aft length (toy mm), where its base sits in the
/// frame of the piece it grows from, and which side of the midline it leans.
struct PlatePlacement {
    var index: Int
    var height: Float
    var length: Float
    var base: SIMD3<Float>
    var side: Float
    var onTail: Bool
}

/// Plate fore-aft lengths on the animal, mm, Table 4 where given (plates
/// 1–12 and 18), else MODEL from the outline of the series.
let stegoPlateLengths: [Float] = [45, 48, 54, 57, 167, 181, 200, 280, 368, 465, 447, 460, 440, 480, 430, 330, 250, 193, 120]

/// The smallest fore-aft length a moulded plate is given, toy mm. The first
/// four of Sophie's plates would be 0.6 mm at 1:80: too slight to fill in a
/// mould. MODEL.
let minimumPlateLength: Float = 1.3
/// How thick every plate is, toy mm. At 1:80 a plate's true thickness is a
/// few tenths of a millimetre; a toy's plates are thicker. MODEL.
let plateThickness: Float = 0.9
/// How far the plate's lens runs down into the back, so its base is buried
/// in the body and not balanced on it. MODEL.
let plateBury: Float = 1.2
/// Half the distance between the two rows, toy mm: the plates sit just off
/// the midline, alternately left and right. MODEL.
let plateRowOffset: Float = 0.65

// The body's top line, in body-frame x and y: where each plate's base sits.
// Neck and back (plates 1–13 on the body piece), then the tail (14–19 on
// the tail piece, in its frame). MODEL placements, spaced along the dorsal
// line as Sophie's are mounted: small plates over the neck, the tallest
// over the hips and the base of the tail.
let stegoPlateStations: [(x: Float, onTail: Bool)] = [
    (22.3, false), (21.0, false), (19.6, false), (18.2, false), (16.6, false), (14.8, false),
    (12.6, false), (10.3, false), (7.8, false), (5.3, false), (2.8, false), (0.2, false), (-2.6, false),
    (-4.0, true), (-7.4, true), (-10.6, true), (-13.6, true), (-16.4, true), (-19.0, true),
]

/// The tail pivot, in the body frame, and the tail's centreline in its own
/// frame: held clear of the ground, drooping a little, then level.
let stegoTailPivot = SIMD3<Float>(-4.5, 1.0, 0)
let stegoTailLine: [(p: SIMD3<Float>, r: Float)] = [
    (SIMD3(1.5, 0.2, 0), 5.0), (SIMD3(-7, -1.6, 0), 3.9), (SIMD3(-16, -3.4, 0), 2.7),
    (SIMD3(-26.5, -4.7, 0), 1.7), (SIMD3(-37.0, -5.3, 0), 0.8),
]

/// Radius of the tail at tail-frame x, and the height of its top.
func stegoTailRadius(_ x: Float) -> (y: Float, r: Float) {
    for i in 1..<stegoTailLine.count where stegoTailLine[i].p.x <= x {
        let a = stegoTailLine[i - 1]
        let b = stegoTailLine[i]
        let t: Float = (x - a.p.x) / (b.p.x - a.p.x)
        return (a.p.y + (b.p.y - a.p.y) * t, a.r + (b.r - a.r) * t)
    }
    let l = stegoTailLine[stegoTailLine.count - 1]
    return (l.p.y, l.r)
}

/// The body's dorsal line: height of the top of the back at body-frame x.
/// MODEL, drawn to the round cones below: highest over the hips, falling to
/// the shoulders, the neck lower still.
func stegoBackTop(_ x: Float) -> Float {
    let pts: [(Float, Float)] = [(-5, 6.6), (0, 6.6), (4, 6.0), (8, 4.5), (12, 2.4), (15, 0.9), (18, -1.0), (21, -2.8), (24, -3.8)]
    if x <= pts[0].0 { return pts[0].1 }
    for i in 1..<pts.count where pts[i].0 >= x {
        let t: Float = (x - pts[i - 1].0) / (pts[i].0 - pts[i - 1].0)
        return pts[i - 1].1 + (pts[i].1 - pts[i - 1].1) * t
    }
    return pts[pts.count - 1].1
}

/// Where the 19 plates go. Each plate alternates side from the one before —
/// or, with the `pairedPlates` mutant, plates are set side by side in pairs
/// at the same station, as some old reconstructions (and Dacentrurus) have.
func stegoPlatePlacements(_ mutant: Mutant = activeMutant) -> [PlatePlacement] {
    var out: [PlatePlacement] = []
    for i in 0..<19 {
        let h: Float = stegoPlateHeights[i] * stegoScale
        let l: Float = max(stegoPlateLengths[i] * stegoScale, minimumPlateLength)
        var station = stegoPlateStations[i]
        var side: Float = i % 2 == 0 ? -1 : 1
        if mutant == .pairedPlates {
            // Pairs (0,1), (2,3) … stand side by side at their mean station;
            // the odd one out (18) keeps its own.
            let j: Int = i - i % 2
            if j + 1 < 19 && stegoPlateStations[j].onTail == stegoPlateStations[j + 1].onTail {
                station.x = (stegoPlateStations[j].x + stegoPlateStations[j + 1].x) / 2
            }
            side = i % 2 == 0 ? -1 : 1
        }
        let top: Float
        if station.onTail {
            let t = stegoTailRadius(station.x)
            top = t.y + t.r
        } else {
            top = stegoBackTop(station.x)
        }
        let base = SIMD3<Float>(station.x, top - 0.25, side * plateRowOffset)
        out.append(PlatePlacement(index: i + 1, height: h, length: l, base: base, side: side, onTail: station.onTail))
    }
    return out
}

/// A plate as a shape: a lens standing on its base, leaning a little out to
/// its own side and, on the tail, a little back. The lean is MODEL: Sophie's
/// describers could not tell which way each plate angled.
func platePrim(_ p: PlatePlacement) -> Prim {
    let lean: Float = 0.14 * p.side
    let back: Float = p.onTail ? -0.18 : -0.05
    let up: SIMD3<Float> = simd_normalize(SIMD3<Float>(back, 1, lean))
    let normal: SIMD3<Float> = simd_normalize(SIMD3<Float>(0, -lean, 1))
    let full: Float = p.height + plateBury
    let centre: SIMD3<Float> = p.base + up * ((p.height - plateBury) / 2)
    return Prim.plate(centre, up: up, normal: normal, halfHeight: full / 2, halfWidth: p.length / 2,
                      halfThickness: plateThickness / 2, edge: 0.3, paint: StegoPaint.plates.rawValue).tagged(.plate)
}

/// The four spikes at the end of the tail, in the tail's frame: two pairs,
/// the rear pair a little shorter, pointing back, up and out. Their
/// direction is MODEL, after modern mounts; Maidment et al. give each
/// spike's angle to its own base (70° and 80°) but could not fix how the
/// base sat on the tail. Tips blunted to 0.22 mm, as a toy's are.
func stegoSpikes() -> [Prim] {
    var out: [Prim] = []
    let stations: [Float] = [-31.4, -31.4, -34.7, -34.7]
    for k in 0..<4 {
        let side: Float = k % 2 == 0 ? -1 : 1
        let x: Float = stations[k]
        let t = stegoTailRadius(x)
        let len: Float = stegoSpikeLengths[k] * stegoScale
        let base = SIMD3<Float>(x, t.y + t.r * 0.45, side * t.r * 0.55)
        let dir: SIMD3<Float> = simd_normalize(SIMD3<Float>(-0.62, 0.48, side * 0.62))
        let baseR: Float = k < 2 ? 0.78 : 0.66
        out.append(Prim.cone(base - dir * 0.4, baseR, base + dir * len, 0.22, paint: StegoPaint.spikes.rawValue).tagged(.spike))
    }
    return out
}

/// The whole design.
func stegosaurusDesign(_ mutant: Mutant = activeMutant) -> ToyDesign {
    let bp: Int = StegoPaint.body.rawValue
    let s: Float = stegoScale

    // Legs, from the bones. The hind piece is femur then tibia; the ankle
    // stands 2.1 mm up on a short, broad foot. The fore piece is humerus then
    // ulna; the wrist 1.9 mm up. Foot heights MODEL (the feet were not
    // measured here).
    let femur: Float = stegoFemur * s
    let tibia: Float = stegoTibia * s
    let humerus: Float = stegoHumerus * s
    let ulna: Float = stegoUlna * s
    let hindAnkle: Float = 2.1
    let foreWrist: Float = 1.9
    let hindReach: Float = legReach(femur, tibia, flexion: stegoRestFlexion)
    let foreReach: Float = legReach(humerus, ulna, flexion: stegoRestFlexion)
    let height: Float = hindReach + hindAnkle
    let shoulderY: Float = foreReach + foreWrist - height
    let shoulderX: Float = 15.2

    func hindFoot(_ side: Float) -> [Prim] {
        var f: [Prim] = [Prim.box(SIMD3(0.3, 0.9, 0), half: SIMD3(1.9, 0.9, 1.55), edge: 0.6, paint: bp)]
        for k in 0..<3 {
            let z: Float = (Float(k) - 1) * 1.0
            f.append(Prim.cone(SIMD3(1.4, 0.55, z), 0.5, SIMD3(2.5, 0.35, z * 1.2), 0.35,
                               paint: StegoPaint.claw.rawValue).tagged(.toe))
        }
        f.append(Prim.cone(SIMD3(0, 1.4, 0), 1.35, SIMD3(0, hindAnkle, 0), 1.3, paint: bp))
        return f
    }
    func foreFoot(_ side: Float) -> [Prim] {
        var f: [Prim] = [Prim.box(SIMD3(0.2, 0.75, 0), half: SIMD3(1.35, 0.75, 1.3), edge: 0.55, paint: bp)]
        for k in 0..<3 {
            let z: Float = (Float(k) - 1) * 0.8
            f.append(Prim.cone(SIMD3(1.1, 0.45, z), 0.42, SIMD3(1.8, 0.3, z * 1.15), 0.3,
                               paint: StegoPaint.claw.rawValue).tagged(.toe))
        }
        f.append(Prim.cone(SIMD3(0, 1.2, 0), 1.15, SIMD3(0, foreWrist, 0), 1.1, paint: bp))
        return f
    }
    var legs: [LegSpec] = []
    for (name, fore, side) in [("LF", true, Float(-1)), ("RF", true, Float(1)), ("LH", false, Float(-1)), ("RH", false, Float(1))] {
        let hip: SIMD3<Float> = fore ? SIMD3(shoulderX, shoulderY, side * 3.0) : SIMD3(0, 0, side * 3.5)
        let a: Float = fore ? humerus : femur
        let b: Float = fore ? ulna : tibia
        let reach: Float = fore ? foreReach : hindReach
        let ankle: Float = fore ? foreWrist : hindAnkle
        let foot = SIMD3<Float>(hip.x, hip.y - reach - ankle, hip.z + side * 0.3)
        let rTop: Float = fore ? 2.2 : 3.1
        let rMid: Float = fore ? 1.55 : 2.05
        let rLow: Float = fore ? 1.35 : 1.7
        let upper: [Prim] = [Prim.cone(.zero, rTop, SIMD3(0, -a, 0), rMid, paint: bp)]
        let lower: [Prim] = [Prim.cone(.zero, rMid, SIMD3(0, -b, 0), rLow, paint: bp)]
        let feet: [Prim] = fore ? foreFoot(side) : hindFoot(side)
        // The claws' paint, carried a little unevenly up the toes: a hand-
        // painted edge. Overspill MODEL.
        let clawEdge = Patch(c: SIMD3(1.05, 0.62, 0), r: 0.12, paint: StegoPaint.claw.rawValue, kind: 3)
        legs.append(LegSpec(name: name, fore: fore, side: side, hip: hip, upper: a, lower: b, ankleHeight: ankle,
                            bendForward: !fore, restFoot: foot, upperPrims: upper, lowerPrims: lower, footPrims: feet,
                            footPatches: [clawEdge]))
    }

    // The trunk: deep and narrow, highest over the hips — two chains of
    // round cones, back and belly, blended. MODEL shapes, sized to the
    // skeleton's hip and shoulder heights above.
    var trunk: [Prim] = [
        Prim.cone(SIMD3(-3.5, 1.6, 0), 5.0, SIMD3(4.5, 1.0, 0), 5.0, paint: bp),
        Prim.cone(SIMD3(4.5, 1.0, 0), 5.0, SIMD3(13.0, -2.4, 0), 3.9, paint: bp),
        Prim.cone(SIMD3(1.0, -2.6, 0), 4.9, SIMD3(11.0, -4.6, 0), 4.1, paint: StegoPaint.belly.rawValue),
        Prim.cone(SIMD3(13.0, -2.4, 0), 3.9, SIMD3(17.5, -4.0, 0), 2.7, paint: bp),
        Prim.cone(SIMD3(17.5, -4.0, 0), 2.7, SIMD3(21.6, -5.1, 0), 2.0, paint: bp),
        // Hips and shoulders, where the legs go in.
        Prim.ball(SIMD3(0, 0, 3.2), 3.4, paint: bp), Prim.ball(SIMD3(0, 0, -3.2), 3.4, paint: bp),
        Prim.ball(SIMD3(shoulderX, shoulderY, 2.7), 2.5, paint: bp),
        Prim.ball(SIMD3(shoulderX, shoulderY, -2.7), 2.5, paint: bp),
    ]
    let placements: [PlatePlacement] = stegoPlatePlacements(mutant)
    for p in placements where !p.onTail { trunk.append(platePrim(p)) }
    // Plate paint spilled a little onto the back at one plate's foot:
    // overspill, as a hand-painted toy has. MODEL.
    let spill = Patch(c: placements[9].base + SIMD3<Float>(0.9, -0.15, -0.35), r: 0.42,
                      paint: StegoPaint.plates.rawValue, kind: 0)
    let body = Segment(name: "body", part: .body, prims: trunk, blend: 0.9, seam: true, patches: [spill])

    // The head: small, long and low, on the end of the neck. The pivot is
    // where head meets neck; the snout points forward and down.
    let headPivot = SIMD3<Float>(21.6, -5.0, 0)
    let headPrims: [Prim] = [
        Prim.cone(SIMD3(0.8, -0.2, 0), 1.55, SIMD3(3.3, -0.9, 0), 1.3, paint: bp),
        Prim.cone(SIMD3(3.3, -0.9, 0), 1.3, SIMD3(5.7, -1.9, 0), 0.7, paint: bp),
        Prim.cone(SIMD3(1.4, -1.0, 0), 1.1, SIMD3(4.4, -1.8, 0), 0.68, paint: StegoPaint.belly.rawValue),
    ]
    let eyes: [Patch] = [Patch(c: SIMD3(2.4, -0.1, 1.25), r: 0.34, paint: StegoPaint.eye.rawValue, kind: 0),
                         Patch(c: SIMD3(2.4, -0.1, -1.25), r: 0.34, paint: StegoPaint.eye.rawValue, kind: 0)]
    let head = Segment(name: "head", part: .head, prims: headPrims, blend: 0.8, seam: true, patches: eyes)

    // The tail, a separate rigid piece at its root, with the six rear plates
    // and the four spikes.
    var tailPrims: [Prim] = []
    for i in 1..<stegoTailLine.count {
        let a = stegoTailLine[i - 1]
        let b = stegoTailLine[i]
        tailPrims.append(Prim.cone(a.p, a.r, b.p, b.r, paint: bp))
    }
    for p in placements where p.onTail { tailPrims.append(plateFor(p)) }
    tailPrims += stegoSpikes()
    let tail = Segment(name: "tail", part: .tail, prims: tailPrims, blend: 0.7, seam: true)

    return ToyDesign(name: "Stegosaurus", paints: stegoPaints, body: body, head: head, headPivot: headPivot,
                     tail: tail, tailPivot: stegoTailPivot, extra: [], legs: legs, bodyHeight: height)
}

/// A tail plate: its base is given in the body frame's x along the tail, so
/// shift it into the tail's frame.
func plateFor(_ p: PlatePlacement) -> Prim {
    var q: PlatePlacement = p
    let t = stegoTailRadius(p.base.x)
    q.base = SIMD3<Float>(p.base.x, t.y + t.r - 0.25, p.base.z)
    return platePrim(q)
}

/// The still's title and argument.
func stegoCaption(lengthMM: Float) -> Caption { Caption(
    title: "A plastic toy Stegosaurus",
    lines: [
        "One piece of solid PVC, injection-moulded and painted; 2–3 inches long (this one \(Int(lengthMM.rounded())) mm: Sophie, NHMUK PV R36730, at 1:80).",
        "Its gloss is PVC's refractive index: n = 1.545 (Zhang et al. 2020) mirrors 4.6% of the light at normal incidence, more at grazing.",
        "Nineteen plates in two alternating rows, four tail spikes, short forelimbs, tail held clear of the ground (Maidment et al. 2015).",
        "Unbranded; the paint colours are a model, not measured.",
    ],
    cutName: "left foreleg") }
