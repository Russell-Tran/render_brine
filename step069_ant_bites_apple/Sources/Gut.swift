// The inset: where the juice goes and where the bits of flesh go. A drawn
// diagram, not a render — the ant's own body seen from the side (outlines
// from the same shapes the GPU draws), with the path food takes inside it.
//
// What the sources say, and so what is drawn:
//
//   * Ants take their food as liquid, and strain out the solids at the mouth.
//     "Solids are first gripped and processed by the mandibles before
//     transport into the oral cavity by labium and maxillae. Larger particles
//     are filtered at the mouth opening … and stored in the infrabuccal
//     pouch. … Smaller particles and possibly pre-digested substrate from the
//     pouch pass the filter and are taken up by the sucking pump. Fluids are
//     licked up by the glossa and directly pass the filter" (Richter &
//     Economo, *Phil Trans R Soc B* 378: 20220556, 2023, their Fig. 1 legend;
//     read in full via Europe PMC, PMC10577024). "The particles are formed
//     into a pellet which is periodically ejected, about once every 24 h" —
//     in the leaf-cutting ant Acromyrmex, their ref. 41 (Febvay & Kermarrec
//     1981); not a formicine, so the caption gives no interval.
//   * The same review is careful: "some level of solid food uptake is
//     evidently possible in adult ants, but how it can be ingested despite
//     the mouth filter is poorly understood". So the caption does NOT say
//     ants cannot swallow any solid: it says the bits are strained out.
//   * For formicines: in Camponotus, "Ingestion of food particles larger than
//     100 microns is effectively curtailed by the specializations of the
//     epipharynx, hypopharynx, and various maxillary and labial structures",
//     solids collect in the infrabuccal chamber, and "the liquid food passes
//     directly to the pharynx and then through the esophagus for storage in
//     the crop" (Hansen, Spangenberg & Gaver 1999, Proc 3rd Int Conf Urban
//     Pests: 211–219, citing Eisner & Happ 1962, *Psyche* 69: 107, on
//     Camponotus pennsylvanicus). "In formicines, workers mainly feed on
//     liquid food and rely on the IBPs to filter solid food" (Zheng et al.,
//     *Front Microbiol* 12: 785016, 2021, who dissected the infrabuccal
//     pockets of Lasius niger among their four formicines). Together with
//     the proventriculus, the pocket filters "solid particles down to 2 µm"
//     (Tragust et al., *eLife* 9: e60287, 2020).
//   * The crop is in the gaster: Tragust et al. collected crop contents by
//     cutting the gaster off "directly behind the petiole" (Camponotus), and
//     from Lasius "through the mouth by gently pressing the ants' gaster".
//
// So a bite's juice runs from the mouth, through the pharynx and the long
// oesophagus, back through the waist to the crop in the gaster; its bits of
// flesh — Golden Delicious cells are around 0.28 mm apart (Food.swift), far
// over Camponotus's 0.1 mm — go down into the infrabuccal pocket under the
// mouth and join the pellet there. Positions of pharynx, oesophagus, crop and
// pocket in the drawing are MODEL, schematic; the dots are not to scale, and
// their speed is not measured.

import Foundation
import simd

/// A point on the diagram, in the ant's own side-view frame (mm, x forward,
/// y up), before the diagram maps it into its panel.
typealias GutPoint = SIMD2<Float>

/// The ant's side view: its body as the GPU draws it, head lowered as it
/// bites, jaws at the grip.
func sideViewBody() -> [Shape] {
    bodyShapes(gasterBend: restingGasterLift, headDown: headDown, open: bite.grip)
}

/// A head point (in the unlowered head) as the biting ant carries it.
func headPoint(ahead: Float, up: Float) -> GutPoint {
    let p: SIMD3<Float> = headCentre + headForward * ahead + headUp * up
    let q: SIMD3<Float> = lowerHead(p, headDown)
    return GutPoint(q.x, q.y)
}

/// The mouth, between the jaws' roots, under the clypeus.
var mouthPoint: GutPoint { headPoint(ahead: 0.48, up: -0.20) }
/// The pharynx (the sucking pump), inside the head behind the mouth. MODEL.
var pharynxPoint: GutPoint { headPoint(ahead: 0.10, up: -0.02) }
/// The infrabuccal pocket: a sac below and behind the mouth. MODEL.
var pocketCentre: GutPoint { headPoint(ahead: 0.24, up: -0.18) }
let pocketRadius: Float = 0.10
/// The pellet inside it. MODEL: a pellet already forming, the size it keeps.
let pelletRadius: Float = 0.055
/// The crop: the front of the gaster. MODEL.
var cropCentre: GutPoint {
    let g: Shape = gasterShape(angle: restingGasterLift)
    let front: SIMD3<Float> = g.a + g.xAxis * (g.b.x * 0.30)
    return GutPoint(front.x, front.y)
}
let cropRadii = SIMD2<Float>(0.36, 0.26)

/// The juice's road: mouth → pharynx → oesophagus through the neck, the
/// mesosoma and the waist → the crop. The solids': mouth → pocket.
var juicePath: [GutPoint] {
    [mouthPoint, pharynxPoint, GutPoint(0.95, 0.66), GutPoint(0.45, 0.66), GutPoint(0.05, 0.62), GutPoint(-0.50, 0.63),
     cropCentre]
}
var solidsPath: [GutPoint] { [mouthPoint, pocketCentre] }

/// A point `f` (0–1) of the way along a polyline, by length.
func along(_ path: [GutPoint], _ f: Float) -> GutPoint {
    var lengths: [Float] = []
    var total: Float = 0
    for i in 1..<path.count {
        let l: Float = simd_distance(path[i - 1], path[i])
        lengths.append(l)
        total += l
    }
    var want: Float = min(max(f, 0), 1) * total
    for i in 1..<path.count {
        if want <= lengths[i - 1] || i == path.count - 1 {
            let u: Float = lengths[i - 1] > 0 ? min(want / lengths[i - 1], 1) : 0
            let leg: GutPoint = (path[i] - path[i - 1]) * u
            return path[i - 1] + leg
        }
        want -= lengths[i - 1]
    }
    return path[path.count - 1]
}

/// One dot in the diagram.
struct GutDot {
    enum Kind { case juice, solid }
    var kind: Kind
    var position: GutPoint
    var radius: Float           // mm in the side view; 0 = not drawn
    var life: Float             // 0–1 through its trip
    var arrived: Bool           // past the end of its road, shrinking into the crop or pellet
}

struct GutState {
    var dots: [GutDot]
}

/// Each bite sends four drops of juice and two bits of flesh in, a little
/// apart, starting as the jaws close on the flesh (bite phase 0.70). MODEL.
let juiceDots: Int = 4
let solidDots: Int = 2
let dotStart: Float = 0.70
/// A trip takes 90% of a bite; the last 15% of it the dot shrinks into the
/// crop or the pellet. MODEL.
let dotTrip: Float = 0.90
let dotShrink: Float = 0.15
let juiceDotRadius: Float = 0.035
let solidDotRadius: Float = 0.030

func gutState(time t: Float, mutant: Mutant) -> GutState {
    let b: Float = effectiveTime(t, mutant: mutant) / biteSeconds
    var dots: [GutDot] = []
    let road: [GutPoint] = juicePath
    let solidRoad: [GutPoint] = mutant == .solidsToCrop ? juicePath : solidsPath
    for k in 0..<(juiceDots + solidDots) {
        let isJuice: Bool = k < juiceDots
        let juiceDelay: Float = 0.05 * Float(k)
        let solidIndex: Float = Float(k - juiceDots)
        let solidDelay: Float = 0.03 + 0.08 * solidIndex
        let delay: Float = isJuice ? juiceDelay : solidDelay
        var life: Float = (b - dotStart - delay).truncatingRemainder(dividingBy: 1)
        if life < 0 { life += 1 }
        life /= dotTrip
        let r0: Float = isJuice ? juiceDotRadius : solidDotRadius
        guard life < 1 else {
            dots.append(GutDot(kind: isJuice ? .juice : .solid, position: .zero, radius: 0, life: 1, arrived: true))
            continue
        }
        let grow: Float = smoothstep01(life / 0.08)
        let shrinkFrom: Float = 1 - dotShrink
        let shrink: Float = 1 - smoothstep01((life - shrinkFrom) / dotShrink)
        let f: Float = min(life / shrinkFrom, 1)
        // Juice moves at an even pace; the bits of flesh slow as they settle.
        let eased: Float = isJuice ? f : 1 - (1 - f) * (1 - f)
        let p: GutPoint = along(isJuice ? road : solidRoad, eased)
        dots.append(GutDot(kind: isJuice ? .juice : .solid, position: p, radius: r0 * grow * shrink, life: life,
                           arrived: life >= shrinkFrom))
    }
    return GutState(dots: dots)
}
