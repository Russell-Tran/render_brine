// One worker standing at the apple's upright edge, head lowered, biting it:
// where the ant stands, how far its jaws close, where its antennae tap — all
// solved from the geometry, not typed. Nothing here touches the GPU.
//
// The world: MILLIMETRES, y up, the card is y = 0. The ant faces +x; the
// bitten edge rises from `cornerBase` (Food.swift).

import Foundation
import simd

/// How far the ant lowers its head at the neck to bite, radians. MODEL: a
/// little — the jaws already reach past the front of the head at its resting
/// 15° pitch; lowered much further, the tips would reach the card before the
/// edge. (Solved alternatives: at 20° the jaws no longer reach the edge.)
let headDown: Float = 4 * Float.pi / 180

/// The gap kept between the edge and every part of the ant but its jaws and
/// antennae, mm. MODEL.
let headGap: Float = 0.04

/// Where each foot stands, in the ant's frame (right side; the left
/// mirrors): step 68's stance centres, all six down. MODEL.
let standingFeet: [SIMD3<Float>] = [
    SIMD3(1.25, 0, 1.40),
    SIMD3(0.26, 0, 1.60),
    SIMD3(-1.60, 0, 1.50),
]

/// The gaster, carried a little raised. Step 44's MODEL.
let restingGasterLift: Float = -6 * Float.pi / 180

/// Every mandible shape of an ant placed at x = `x0` with its jaws open by
/// `open`, in the world.
func mandiblesInWorld(x0: Float, open: Float) -> [Shape] {
    let at = SIMD3<Float>(x0, 0, 0)
    var out: [Shape] = []
    for s in bodyShapes(gasterBend: restingGasterLift, headDown: headDown, open: open) where s.part == .mandible {
        out.append(s.placed(scale: 1, yaw: 0, at: at))
    }
    return out
}

/// The least clearance between a set of shapes and the apple, mm, from
/// points along each shape (dense enough that a flat face's nearest point is
/// found to well under a micrometre).
func clearance(_ shapes: [Shape], _ m: ApplePiece) -> Float {
    var worst: Float = 1e9
    for s in shapes {
        switch s.kind {
        case .roundCone:
            for k in 0...200 {
                let t: Float = Float(k) / 200
                let along: SIMD3<Float> = (s.b - s.a) * t
                let c: SIMD3<Float> = s.a + along
                let taper: Float = (s.rb - s.ra) * t
                let r: Float = s.ra + taper
                worst = min(worst, pieceSDF(c, m) - r)
            }
        case .ellipsoid, .roundBox:
            let z: SIMD3<Float> = simd_cross(s.xAxis, s.yAxis)
            let e: SIMD3<Float> = s.kind == .ellipsoid ? s.b : s.b + SIMD3<Float>(repeating: s.ra)
            for i in 0...24 {
                let th: Float = Float.pi * Float(i) / 24
                for j in 0..<48 {
                    let ph: Float = Float.pi * Float(j) / 24
                    let lx: Float = sin(th) * cos(ph) * e.x
                    let ly: Float = cos(th) * e.y
                    let lz: Float = sin(th) * sin(ph) * e.z
                    let ax: SIMD3<Float> = s.xAxis * lx
                    let ay: SIMD3<Float> = s.yAxis * ly
                    let az: SIMD3<Float> = z * lz
                    let p: SIMD3<Float> = s.a + ax + ay + az
                    worst = min(worst, pieceSDF(p, m))
                }
            }
        }
    }
    return worst
}

/// Where the ant stands and how far its jaws close on the edge, solved:
///   1. with the jaws open wide, the ant comes forward until some part of it
///      other than jaws and antennae is `headGap` from the apple;
///   2. then the jaws close — each turns in about its root — until they meet
///      the flesh, by bisection on the drawn distances. That angle is the grip.
struct Bite {
    let x0: Float          // the ant's origin, x (it stands on the axis, facing +x)
    let grip: Float        // each mandible's opening at the grip, radians
    let piece: ApplePiece
}

func solveBite(mutant: Mutant) -> Bite {
    let m: ApplePiece = buildPiece(mutant: .none)
    func body(_ x0: Float) -> [Shape] {
        bodyShapes(gasterBend: restingGasterLift, headDown: headDown, open: gapeMax)
            .filter { $0.part != .mandible }
            .map { $0.placed(scale: 1, yaw: 0, at: SIMD3<Float>(x0, 0, 0)) }
    }
    var lo: Float = -8
    var hi: Float = 0
    for _ in 0..<40 {
        let mid: Float = (lo + hi) / 2
        let c: Float = min(clearance(body(mid), m), clearance(mandiblesInWorld(x0: mid, open: gapeMax), m) + headGap)
        if c > headGap { lo = mid } else { hi = mid }
    }
    let x0: Float = lo
    // Close from wide open until the jaws just meet the flesh.
    var open: Float = gapeMax
    var shut: Float = 0
    for _ in 0..<40 {
        let mid: Float = (open + shut) / 2
        if clearance(mandiblesInWorld(x0: x0, open: mid), m) > 0 { open = mid } else { shut = mid }
    }
    return Bite(x0: x0, grip: open, piece: m)
}

let bite: Bite = solveBite(mutant: .none)

/// The mandibles' opening at a bite phase: the grip, widening to the full
/// gape and back. The mutants: biteThrough closes 6° past the grip, biteShort
/// stops 6° before it, stillBite opens only 0.3°.
func mandibleOpening(phase: Float, mutant: Mutant) -> Float {
    let six: Float = 6 * Float.pi / 180
    var grip: Float = bite.grip
    if mutant == .biteThrough { grip -= six }
    if mutant == .biteShort { grip += six }
    var widest: Float = gapeMax
    let tiny: Float = 0.3 * Float.pi / 180
    if mutant == .stillBite { widest = grip + tiny }
    let g: Float = gapeAt(phase: phase)
    return grip + (widest - grip) * g
}

// MARK: - the antennae: tapping the two faces beside the edge

/// Where each antenna touches: on its own side's face, 1.2 mm from the edge
/// and 0.8 mm up, where the funiculus reaches with a gentle bow. MODEL.
func antennaContact(_ index: Int) -> (point: SIMD3<Float>, normal: SIMD3<Float>) {
    let right: Bool = index == 0
    let n: SIMD3<Float> = right ? rightFaceNormal : leftFaceNormal
    let along: SIMD3<Float> = right ? pieceV : pieceU      // along that face, away from the edge
    let out: SIMD3<Float> = along * 1.2
    let onFace: SIMD3<Float> = cornerBase + out + SIMD3<Float>(0, 0.8, 0)
    return (onFace, n)
}

/// The scapes raised, the funiculus bowing up and back, away from the face.
/// MODEL pose.
func bitingScape(_ sgn: Float) -> SIMD3<Float> { simd_normalize(SIMD3<Float>(0.35, 0.85, sgn * 0.40)) }
func bitingBow(_ sgn: Float) -> SIMD3<Float> { SIMD3<Float>(-0.6, 0.8, sgn * 0.5) }

// MARK: - the ant at one moment

struct BitingAnt {
    let time: Float
    let tapPhase: Float
    let lift: Float                 // antenna tips off the juice, mm
    let bitePhase: Float
    let opening: Float              // each mandible's opening, radians
    let model: AntModel             // ant frame
    let shapes: [Shape]             // world
    let at: SIMD3<Float>
    let antennaTips: [SIMD3<Float>] // world centres of the tips' round ends
    var gripping: Bool { abs(opening - bite.grip) < 1e-6 }
    var mandibles: [Shape] { shapes.filter { $0.part == .mandible } }
}

func bitingAnt(time t: Float, mutant: Mutant) -> BitingAnt {
    let at = SIMD3<Float>(bite.x0, 0, 0)
    let phase: Float = tapPhase(t, mutant: mutant)
    let lift: Float = liftAt(phase: phase, height: liftHeight)
    let bp: Float = bitePhase(t, mutant: mutant).phase
    let opening: Float = mandibleOpening(phase: bp, mutant: mutant)
    var legs: [Leg] = []
    for j in 0..<6 {
        let side: Float = j < 3 ? 1 : -1
        legs.append(buildLeg(index: j, foot: standingFeet[j % 3] * SIMD3<Float>(1, 1, side)))
    }
    var antennae: [Antenna] = []
    var tips: [SIMD3<Float>] = []
    for i in 0..<2 {
        let sgn: Float = i == 0 ? 1 : -1
        let (p, n) = antennaContact(i)
        let press: Float = (mutant == .press && lift == 0) ? -0.02 : 0
        let tipWorld: SIMD3<Float> = p + n * (funiculusTipRadius + lift + press)
        let a: Antenna = buildAntenna(index: i, scapeDir: bitingScape(sgn), tip: tipWorld - at, bow: bitingBow(sgn),
                                      headDown: headDown, mutant: mutant)
        antennae.append(a)
        tips.append(a.tipCentre + at)
    }
    let body: [Shape] = bodyShapes(gasterBend: restingGasterLift, headDown: headDown, open: opening)
    let model = AntModel(body: body, legs: legs, antennae: antennae)
    let shapes: [Shape] = model.shapes.map { $0.placed(scale: 1, yaw: 0, at: at) }
    return BitingAnt(time: t, tapPhase: phase, lift: lift, bitePhase: bp, opening: opening, model: model, shapes: shapes,
                     at: at, antennaTips: tips)
}
