// A sweet pea climbing garden netting with its tendrils, as numbers: where
// the stem and every leaf are at any hour, and what each tendril is doing —
// searching, touching the net, coiling round it, and then coiling in its free
// length into a spring. Nothing here touches the GPU; it is the part a test
// reads against the sources.
//
// The sweet pea, Lathyrus odoratus, is an ornamental relative of the garden
// pea, Pisum sativum — not the pea anyone eats, and its seeds are toxic.
//
// Unlike step 45's bean it does not twine. Its stem grows up beside the net,
// and each leaf ends in a branched tendril: leaflets turned into grasping
// threads. A tendril sweeps round while it grows (circumnutation), touches a
// string, curls round it, and then — anchored now at both ends, the leaf and
// the string — its free length coils. A thread fixed at both ends cannot coil
// all one way without twisting, so it coils one way for half its length and
// the other way for the other half, with a short straight piece between: a
// PERVERSION. The coils are a spring, and as they form they draw the leaf in
// towards the net.
//
// Millimetres and hours. The net hangs in the plane z = 0, its strings on a
// 100 mm square mesh; the stem grows up in front of it, towards the camera
// (+z), midway between two vertical strings. +y is up, +x the viewer's right.
//
// Sources, checked for this step:
//
//   Flora of China 10 (Lathyrus odoratus): "Annual herbs, 50–200 cm tall.
//     Stem climbing, much branched, somewhat hairy, winged. Leaves with
//     branched tendril at apex; rachis winged; stipules semisagittate;
//     leaflets 1-paired, ovate-oblong or elliptic, 20–60 × 7–30 mm".
//     Flora of Pakistan: "stem pubescent, winged. Leaf paripinnately compound,
//     leaflets 2 ... Stipules 1.5–2.5 cm long, semisagittate." (efloras.org)
//   Hofer et al., Plant Cell 21:420–428 (2009), "Tendril-less regulates
//     tendril formation in pea leaves": the tendril-less mutant of garden pea
//     and the t mutant of sweet pea both turn tendrils into leaflets, which
//     "demonstrates that the pea tendril is a modified leaflet".
//   Spencer & Schaumburg, Neurobehav. Toxicol. Teratol. 5:625–629 (1983):
//     Lathyrus odoratus "contain[s] a compound, beta-aminopropionitrile
//     (BAPN), that induces pathological changes in bone ('osteolathyrism')
//     and blood vessels ('angiolathyrism') of experimental animals". The
//     rat studies of the 1950s called it "sweet pea lathyrism (odoratism)".
//   Darwin, The Movements and Habits of Climbing Plants, 2nd ed. (1875),
//     Project Gutenberg #2485, ch. IV, Pisum sativum: "Dutrochet observed the
//     completion of an ellipse in 1 hr. 20 m.; and I saw one completed in
//     1 hr. 30 m."; tendrils "revolve ... like those made by the internodes";
//     "a single light touch ... caused them to bend quickly"; and "In the
//     common Pea the lateral branches alone contract, and not the central
//     stem." Darwin timed no Lathyrus odoratus; these are the garden pea's
//     numbers, borrowed and flagged.
//     Ch. V, spiral contraction: it "commences in half a day, or in a day or
//     two after their extremities have caught some object"; Echinocystis
//     "spirally flexuous in 7 hrs., and spirally contracted in 18 hrs."; "A
//     tendril ... which has caught a support by its extremity ... invariably
//     becomes twisted in one part in one direction, and in another part in
//     the opposite direction; the oppositely turned spires being separated by
//     a short straight portion"; "there are as many turns in the one
//     direction as in the other"; "the spiral contraction which draws up the
//     stem"; Passiflora gracilis tendrils began to bend 25–39 s after a touch.
//   Gerbode, Puzey, McCormick & Mahadevan, Science 337:1087–1091 (2012), "How
//     the cucumber tendril coils and overwinds": coiling is driven by
//     asymmetric contraction of an internal fibre ribbon, and a tendril held
//     at both ends forms the perversion and overwinds under tension. That is
//     cucumber, measured; the geometry — two helices of opposite hand and a
//     perversion, net twist zero — follows for any thread anchored at both
//     ends, and is what is drawn here. It has not been measured in Lathyrus.
//   Engelberth, Adv. Space Res. 32:1611–1619 (2003): tendrils of Bryonia and
//     Pisum "respond to such a stimulus with a rapid coiling response ...
//     within minutes."
//   Gianoli, AoB Plants 7:plv013 (2015): the usable support diameter of
//     tendril climbers is lower even than twiners' (Putz 1984 and others) —
//     why the support here is thin string, not a pole.

import Foundation
import simd

// MARK: - what can be broken on purpose

enum Mutant: String {
    case none
    case noPerversion = "no_perversion"   // the anchored free coil turns all one way
    case stemTendril = "stem_tendril"     // tendrils growing from the stem, not the leaf tip
    case rewind                           // the loop closes by playing the growth backwards

    static var fromEnvironment: Mutant {
        Mutant(rawValue: ProcessInfo.processInfo.environment["TENDRIL_MUTANT"] ?? "none") ?? .none
    }
}

// MARK: - cited numbers

/// Flora of China: leaflets 1-paired, 20–60 × 7–30 mm. Drawn 40 × 17 mm.
let leafletSize = SIMD2<Float>(40, 17)
let leafletLengthRange: ClosedRange<Float> = 20...60
let leafletWidthRange: ClosedRange<Float> = 7...30
let leafletsPerLeaf: Int = 2
/// Flora of Pakistan: stipules 1.5–2.5 cm, semisagittate. Drawn 17 mm.
let stipuleLength: Float = 17
let stipuleRange: ClosedRange<Float> = 15...25

/// Darwin's garden-pea ellipses: 80 min (Dutrochet) and 90 min (Darwin).
/// Sweet pea is untimed; the pea's range stands in for it (flagged).
let peaEllipseMinutesRange: ClosedRange<Float> = 80...90

/// Contraction after catching: begins about half a day after (Darwin: "in half
/// a day, or in a day or two"; Echinocystis 12–24 h) and is complete by a day
/// (Echinocystis: "spirally contracted in 18 hrs"). MODEL within those.
let contractionStartsAfter: Float = 12
let contractionCompleteAfter: Float = 24

/// Curling round the string once touched: Darwin saw tendrils begin to bend
/// within half a minute of a touch, and Engelberth reports coiling within
/// minutes. Two and a half turns round the string take an hour here (MODEL).
let wrapHours: Float = 1.0
let wrapTurns: Float = 2.5

// MARK: - model sizes

/// Garden netting on a 100 mm square mesh, strings 2 mm thick. MODEL: a thin
/// support because tendrils grip thin things (Gianoli 2015; Darwin's peas held
/// twigs and thread); netting is what sweet peas are commonly given.
let meshSpacing: Float = 100
let twineRadius: Float = 1.0
/// The stem grows 22 mm in front of the net, midway between two strings.
let stemDepth: Float = 22
let stemRadius: Float = 1.5
/// Each wing stands 0.9 mm out from the stem (MODEL; the floras say only
/// "winged").
let wingWidth: Float = 0.9
let wingHalfThickness: Float = 0.15

/// One node per plastochron. MODEL: 50 mm internodes, one every 12 h — 10 cm
/// a day, a vigorous summer pace. No source timed a sweet pea's nodes.
let internodeLength: Float = 50
let plastochronHours: Float = 12
let growthRate: Float = internodeLength / plastochronHours

/// The loop: two nodes, a left leaf and a right, which is also one mesh
/// square, so the net repeats with it. 24 h of growth.
let nodesPerLoop: Int = 2
let loopHours: Float = plastochronHours * Float(nodesPerLoop)
let loopRise: Float = internodeLength * Float(nodesPerLoop)

/// The young internodes and tendrils sweep round in narrow ellipses: 17 in a
/// loop, so a loop holds a whole number of them — 84.7 min each, inside the
/// pea's 80–90 min.
let nutationsPerLoop: Int = 17
let nutationHours: Float = loopHours / Float(nutationsPerLoop)
/// How much of the stem below the tip sways, and by how far at the tip. MODEL.
let swayLength: Float = 40
let swayAmplitude: Float = 9

/// Leaf: winged petiole 28 mm (MODEL), opening over 10 h (MODEL, compressed as
/// in step 45 — a real leaf takes days; the tendril must be grown by the time
/// it reaches the net).
let petioleLength: Float = 28
let petioleRadius: Float = 0.6
let leafOpenHours: Float = 10
let budFraction: Float = 0.1

/// Tendril: 0.8 mm thick (MODEL). A short main axis from the leaf tip, then
/// three branches: one reaches the net and becomes the spring (Darwin: in the
/// pea "the lateral branches alone contract"), two stay short and hooked.
let tendrilRadius: Float = 0.4
let tendrilMainLength: Float = 7
let tendrilSideLengths: [Float] = [16, 12]

/// The tendril touches the net 14 h after its node leaves the tip. MODEL.
let contactAge: Float = 14
/// It touches a vertical string this far above its leaf tip. MODEL.
let contactRise: Float = 18

/// The spring: 1.5 turns each side of the perversion (Darwin: always as many
/// one way as the other), 1.8 mm coil radius. MODEL.
let coilTurnsPerSide: Float = 1.5
let coilRadius: Float = 1.8
/// How short the straight perversion is, as a fraction of the coil's length.
/// MODEL.
let perversionWidth: Float = 0.08

// MARK: - the timeline

/// 10 s at 20 frames a second.
let frameCount: Int = 200
let frameDelayCentiseconds: Int = 5
let loopSeconds: Float = Float(frameCount * frameDelayCentiseconds) / 100

func hourOf(frame f: Int, mutant: Mutant = .none) -> Float {
    let u: Float = Float(f) / Float(frameCount)
    if mutant == .rewind {
        let back: Float = u < 0.5 ? 2 * u : 2 * (1 - u)
        return back * loopHours
    }
    return u * loopHours
}

/// What the frame says about the plant, besides its name.
let relationCaption: String = "an ornamental relative of the garden pea, not the pea you eat;"
let toxicCaption: String = "its seeds are toxic"
let tendrilCaption: String = "tendrils are modified leaflets: they search, catch, coil, and spring"

func timeCaption() -> String {
    String(format: "time-lapse: about %.0f hours in %.0f s", loopHours, loopSeconds)
}

// MARK: - the stem

func apexHeight(_ t: Float) -> Float { growthRate * t }

/// The sway of the young stem at height y: a narrow ellipse, growing towards
/// the tip. Everything attached there sways with it.
func sway(_ y: Float, _ t: Float) -> V3 {
    let below: Float = apexHeight(t) - y
    if below >= swayLength { return V3(0, 0, 0) }
    let f: Float = 1 - max(below, 0) / swayLength
    let a: Float = swayAmplitude * f * f
    let ph: Float = 2 * Float.pi * t / nutationHours
    return V3(a * cos(ph), 0, 0.4 * a * sin(ph))
}

func stemPoint(_ y: Float, _ t: Float) -> V3 { V3(0, y, stemDepth) + sway(y, t) }

/// The stem is sampled every 5 mm on a fixed lattice of heights, plus the tip.
let stemStep: Float = 5
let stemBelow: Float = 420

func stemPoints(_ t: Float) -> [V3] {
    let ya: Float = apexHeight(t)
    let first: Int = Int(floor((ya - stemBelow) / stemStep + 0.37))
    var pts: [V3] = []
    var i: Int = first
    while Float(i) * stemStep < ya - 0.25 * stemStep {
        pts.append(stemPoint(Float(i) * stemStep, t))
        i += 1
    }
    pts.append(stemPoint(ya, t))
    return pts
}

// MARK: - nodes, leaves, tendrils

struct NodeInfo {
    var index: Int
    var y: Float
    var age: Float
    /// +1: the leaf is to the right, towards x = +50; −1 to the left.
    var side: Float
}

func nodes(_ t: Float) -> [NodeInfo] {
    let ya: Float = apexHeight(t)
    let lowest: Int = Int(floor((ya - stemBelow + 20) / internodeLength + 0.37))
    let highest: Int = Int(floor((ya - 1) / internodeLength))
    var out: [NodeInfo] = []
    if highest < lowest { return out }
    for k in lowest...highest {
        let y: Float = Float(k) * internodeLength
        out.append(NodeInfo(index: k, y: y, age: (ya - y) / growthRate, side: (k & 1) == 0 ? 1 : -1))
    }
    return out
}

func leafGrowth(_ age: Float) -> Float { budFraction + (1 - budFraction) * smoothstep(0, leafOpenHours, age) }
func leafOpening(_ age: Float) -> Float { smoothstep(0.1 * leafOpenHours, leafOpenHours, age) }

/// The petiole's direction: out to the leaf's side, up, and a little towards
/// the camera; young, it points up along the shoot.
func petioleDirection(_ side: Float, _ age: Float) -> V3 {
    let o: Float = leafOpening(age)
    let elev: Float = 1.2 + (0.55 - 1.2) * o
    return simd_normalize(V3(side * cos(elev), sin(elev), 0.28))
}

/// Where the leaf's tip — the end of the rachis, where the leaflets sit and
/// the tendril starts — would be with no pull on it.
func restLeafTip(_ n: NodeInfo, _ t: Float) -> V3 {
    stemPoint(n.y, t) + petioleDirection(n.side, n.age) * (petioleLength * leafGrowth(n.age))
}

/// The hour node n was at the tip, and the hour its tendril touches the net.
func birthHour(_ n: NodeInfo, _ t: Float) -> Float { t - n.age }

/// Everything fixed about a tendril's catch, worked out from where its leaf
/// tip is at the moment of contact (by then the leaf is full-grown and out of
/// the swaying zone, so it has stopped moving).
struct Catch {
    /// The leaf tip and the branch point, unpulled.
    var leafTip: V3
    var branchPoint: V3
    /// The string's axis at the contact height, and the point on the
    /// tendril's axis that touches it.
    var stringAxis: V3
    var contact: V3
    /// Main direction at contact, and the reach of the catching branch.
    var direction: V3
    var reach: Float
    /// Which way round the string it wraps.
    var wrapSense: Float
}

func tendrilCatch(_ n: NodeInfo, _ t: Float) -> Catch {
    let tc: Float = birthHour(n, t) + contactAge
    let nc = NodeInfo(index: n.index, y: n.y, age: contactAge, side: n.side)
    let b: V3 = restLeafTip(nc, tc)
    let axis = V3(n.side * meshSpacing / 2, b.y + contactRise, 0)
    var toward = V3(b.x - axis.x, 0, b.z - axis.z)
    toward = simd_normalize(toward)
    let contact: V3 = axis + toward * (twineRadius + tendrilRadius)
    let dir: V3 = simd_normalize(contact - b)
    let bp: V3 = b + dir * tendrilMainLength
    return Catch(leafTip: b, branchPoint: bp, stringAxis: axis, contact: contact, direction: dir,
                 reach: simd_distance(contact, bp), wrapSense: n.side)
}

/// How far the tendril has grown, 0…1: full exactly at contact.
func tendrilGrowth(_ age: Float) -> Float { 0.12 + 0.88 * smoothstep(0, contactAge, age) }

/// The stage of a tendril, for the tests and the labels.
enum Stage: Int { case searching = 0, wrapping, anchored, contracting, sprung }

func stage(_ age: Float) -> Stage {
    if age < contactAge { return .searching }
    if age < contactAge + wrapHours { return .wrapping }
    if age < contactAge + contractionStartsAfter { return .anchored }
    if age < contactAge + contractionCompleteAfter { return .contracting }
    return .sprung
}

/// Contraction of the free coil, 0…1.
func contraction(_ age: Float) -> Float {
    smoothstep(contactAge + contractionStartsAfter, contactAge + contractionCompleteAfter, age)
}

/// The free coil between the branch point and the string: two helices of
/// opposite hand joined by a straight perversion, drawn along the chord from
/// `a` to `b`. θ, the angle round the chord, climbs to the middle and comes
/// back down to where it started — so the net twist between the two anchored
/// ends is zero, as it must be. The no-perversion mutant winds all one way.
func coilCurve(from a: V3, to b: V3, contraction c: Float, mutant: Mutant, samples n: Int = 160) -> [V3] {
    let e: V3 = simd_normalize(b - a)
    let e1: V3 = perpendicular(to: e)
    let e2: V3 = simd_cross(e, e1)
    let D: Float = simd_distance(a, b)
    let d: Float = perversionWidth
    let top: Float = (1 + d * d).squareRoot() - d
    var pts: [V3] = []
    for i in 0...n {
        let u: Float = Float(i) / Float(n)
        // The envelope takes the coil off the chord at both ends.
        let env: Float = smoothstep(0, 0.1, u) * smoothstep(1, 0.9, u)
        let rho: Float = coilRadius * c * env
        var theta: Float
        if mutant == .noPerversion {
            theta = 2 * Float.pi * (2 * coilTurnsPerSide) * c * u
        } else {
            let x: Float = 1 - 2 * u
            let rim: Float = (1 + d * d).squareRoot()
            let here: Float = (x * x + d * d).squareRoot()
            let g: Float = (rim - here) / top
            theta = 2 * Float.pi * coilTurnsPerSide * c * g
        }
        let round: V3 = e1 * cos(theta) + e2 * sin(theta)
        let along: V3 = e * (D * u)
        pts.append(a + along + round * rho)
    }
    return pts
}

/// The chord the free coil spans, for a given contraction: the coil's arc
/// length stays the tendril's length, so the more it coils the shorter the
/// chord — that shortening is the pull. Found by bisection.
func coilChord(length: Float, contraction c: Float, mutant: Mutant) -> Float {
    if c <= 0 { return length }
    var lo: Float = 0.05 * length
    var hi: Float = length
    for _ in 0..<40 {
        let mid: Float = (lo + hi) / 2
        let pts: [V3] = coilCurve(from: V3(0, 0, 0), to: V3(mid, 0, 0), contraction: c, mutant: mutant)
        if polylineLength(pts) > length { hi = mid } else { lo = mid }
    }
    return (lo + hi) / 2
}

struct TendrilGeometry {
    /// Where the tendril starts: the leaf tip, or the stem for the mutant.
    var base: V3
    /// The leaf tip, pulled.
    var leafTip: V3
    var main: [V3]
    /// The catching branch, from the branch point to its tip — through the
    /// free coil and round the string, once caught.
    var catching: [V3]
    /// Of `catching`: the free part (branch point to contact) and the wrap.
    var free: [V3]
    var wrap: [V3]
    var sides: [[V3]]
    var stage: Stage
    var contraction: Float
    var stringAxis: V3
}

func tendril(_ n: NodeInfo, _ t: Float, _ m: Mutant) -> TendrilGeometry {
    let k: Catch = tendrilCatch(n, t)
    let g: Float = tendrilGrowth(n.age)
    let st: Stage = stage(n.age)
    let c: Float = contraction(n.age)
    var leafTip: V3 = restLeafTip(n, t)
    var free: [V3] = []
    var wrap: [V3] = []
    var dir: V3 = k.direction
    var branch: V3
    if st == .searching {
        // Sweeping in a narrow ellipse round the line it will catch along, and
        // exactly on that line at the moment of contact.
        let e1: V3 = simd_normalize(simd_cross(k.direction, V3(0, 0, 1)))
        let e2: V3 = simd_cross(k.direction, e1)
        func ellipse(_ h: Float) -> V3 {
            let ph: Float = 2 * Float.pi * h / nutationHours
            return e1 * (0.30 * cos(ph)) + e2 * (0.12 * sin(ph))
        }
        let tc: Float = birthHour(n, t) + contactAge
        dir = simd_normalize(k.direction + ellipse(t) - ellipse(tc))
        branch = leafTip + dir * (tendrilMainLength * g)
        free = [branch, branch + dir * (k.reach * g)]
    } else {
        // Anchored: the free length coils and draws the leaf in.
        let chord: Float = coilChord(length: k.reach, contraction: c, mutant: m)
        let away: V3 = simd_normalize(k.branchPoint - k.contact)
        branch = k.contact + away * chord
        leafTip = k.leafTip + (branch - k.branchPoint)
        free = coilCurve(from: branch, to: k.contact, contraction: c, mutant: m)
        // Round the string, starting at the contact point.
        let turns: Float = wrapTurns * smoothstep(contactAge, contactAge + wrapHours, n.age)
        let r0: V3 = k.contact - k.stringAxis
        let steps: Int = max(Int(turns * 24), 1)
        let rw: Float = twineRadius + tendrilRadius
        let pitch: Float = 2.4 * tendrilRadius + 0.2
        let a0: Float = atan2(r0.z, r0.x)
        for i in 0...steps {
            let s: Float = turns * Float(i) / Float(steps)
            let a: Float = a0 + k.wrapSense * 2 * Float.pi * s
            wrap.append(k.stringAxis + V3(rw * cos(a), pitch * s, rw * sin(a)))
        }
    }
    let base: V3 = m == .stemTendril ? stemPoint(n.y, t) + V3(n.side * stemRadius, 0, 0) : leafTip
    let main: [V3] = [base, branch]
    // The two short branches, hooked at their tips.
    var sides: [[V3]] = []
    let bend: V3 = simd_normalize(simd_cross(dir, V3(0, 0, 1)))
    for (i, len) in tendrilSideLengths.enumerated() {
        let turn: Float = (i == 0 ? 0.75 : -0.8) * n.side
        let d0: V3 = simd_normalize(rotate(dir, about: V3(0, 0, 1), by: turn))
        let l: Float = len * g
        let p1: V3 = branch + d0 * (0.75 * l)
        let away: Float = (i == 0 ? 1 : -1) * n.side
        let curl: V3 = bend * away
        let hook: V3 = simd_normalize(d0 * 0.3 - curl - V3(0, 0.4, 0))
        sides.append(bezier(branch, p1, p1 + hook * (0.3 * l), segments: 8))
    }
    let catching: [V3] = free + wrap.dropFirst()
    return TendrilGeometry(base: base, leafTip: leafTip, main: main, catching: catching, free: free, wrap: wrap,
                           sides: sides, stage: st, contraction: c, stringAxis: k.stringAxis)
}

/// A winged stalk: capsules down its middle, and a flat wing either side in
/// the plane facing the camera.
func wingedChain(_ pts: [V3], radius: Float, wing: Float, material: Material, into s: inout PlantScene) {
    s.chains.append(Chain(points: pts, radii: Array(repeating: radius, count: pts.count), material: material))
    for i in 1..<pts.count {
        let a: V3 = pts[i - 1]
        let b: V3 = pts[i]
        let along: V3 = simd_normalize(b - a)
        var n0: V3 = V3(1, 0, 0) - along * simd_dot(V3(1, 0, 0), along)
        if simd_length(n0) < 0.2 { n0 = V3(0, 1, 0) - along * simd_dot(V3(0, 1, 0), along) }
        s.ribbons.append(Ribbon(a: a, b: b, n: simd_normalize(n0), halfWidth: radius + wing,
                                halfThickness: wingHalfThickness, material: material))
    }
}

func leafAndTendril(_ n: NodeInfo, _ t: Float, _ m: Mutant, into s: inout PlantScene) {
    let g: Float = leafGrowth(n.age)
    let o: Float = leafOpening(n.age)
    let node: V3 = stemPoint(n.y, t)
    let dir: V3 = petioleDirection(n.side, n.age)
    let td: TendrilGeometry = tendril(n, t, m)
    let tip: V3 = td.leafTip
    let lp: Float = petioleLength * g
    let pet: [V3] = bezier(node, node + dir * (0.5 * lp), tip, segments: 6)
    wingedChain(pet, radius: petioleRadius, wing: 0.9 * g, material: .petiole, into: &s)

    // Stipules: a pair of small half-arrowheads at the node, pointing up.
    for sd in [Float(1), -1] {
        let u: V3 = simd_normalize(V3(0.35 * sd, 1, 0.25))
        let w: V3 = simd_normalize(V3(0, 0, 1) - u * u.z)
        s.leaflets.append(Leaflet(origin: node + V3(sd * stemRadius * 0.8, -1, 0.5), u: u, v: simd_cross(w, u), w: w,
                                  length: stipuleLength * max(g, 0.3), halfWidth: 3.2 * max(g, 0.3), fold: 0.3,
                                  droop: 0.05, shape: .stipule, material: .stipule))
    }
    // The leaflet pair at the leaf tip, spreading as the leaf opens into a V
    // that hangs below the rachis, the tendril carrying on above it. MODEL.
    let up = V3(0, 1, 0)
    for spreadAngle in [Float(0.15 + 0.35 * o), 0.75 + 0.75 * o] {
        let spread: Float = -n.side * spreadAngle
        let u0: V3 = rotate(dir, about: V3(0, 0, 1), by: spread)
        let u: V3 = simd_normalize(u0 + up * (0.15 * (1 - o)))
        let facing: V3 = simd_normalize(V3(0, 0.85, 0.55))
        let w: V3 = simd_normalize(facing - u * simd_dot(facing, u))
        s.leaflets.append(Leaflet(origin: tip, u: u, v: simd_cross(w, u), w: w, length: leafletSize.x * g,
                                  halfWidth: leafletSize.y / 2 * g, fold: 1.3 + (0.25 - 1.3) * o, droop: 0.12 * o,
                                  shape: .peaElliptic, material: .leaf))
    }
    // The tendril: main axis, the catching branch, the two short branches.
    let r: Float = tendrilRadius
    s.chains.append(Chain(points: td.main, radii: [r * 1.3, r * 1.15], material: .tendril))
    s.chains.append(Chain(points: td.catching, radii: Array(repeating: r, count: td.catching.count), material: .tendril))
    for b in td.sides { s.chains.append(Chain(points: b, radii: Array(repeating: r * 0.9, count: b.count), material: .tendril)) }
}

// MARK: - the net and the scene

func net(_ t: Float, into s: inout PlantScene) {
    let ya: Float = apexHeight(t)
    let lo: Float = ya - 520
    let hi: Float = ya + 220
    for i in -2...1 {
        let x: Float = (Float(i) + 0.5) * meshSpacing
        s.chains.append(Chain(points: [V3(x, lo, 0), V3(x, hi, 0)], radii: [twineRadius, twineRadius], material: .twine))
    }
    let j0: Int = Int(ceil(lo / meshSpacing))
    let j1: Int = Int(floor(hi / meshSpacing))
    if j1 >= j0 {
        for j in j0...j1 {
            let y: Float = Float(j) * meshSpacing
            s.chains.append(Chain(points: [V3(-260, y, 0), V3(260, y, 0)], radii: [twineRadius, twineRadius],
                                  material: .twine))
        }
    }
}

func sweetPeaScene(_ t: Float, _ m: Mutant) -> PlantScene {
    var s = PlantScene()
    net(t, into: &s)
    let st: [V3] = stemPoints(t)
    wingedChain(st, radius: stemRadius, wing: wingWidth, material: .stem, into: &s)
    // Taper the last few samples to the tip.
    if let i = s.chains.indices.last {
        let n: Int = s.chains[i].radii.count
        for k in 0..<n {
            let fromTip: Float = Float(n - 1 - k) * stemStep
            s.chains[i].radii[k] = stemRadius * (0.45 + 0.55 * smoothstep(0, 30, fromTip))
        }
    }
    for n in nodes(t) { leafAndTendril(n, t, m, into: &s) }
    return s
}

// MARK: - the camera

/// Rising with the tip at the same steady rate, so a loop later the picture is
/// the same one mesh square higher. MODEL: 0.8 m away, a little to the right
/// and looking down about 21°, so the tendrils' reach back to the net shows.
let cameraDistance: Float = 800
let cameraAzimuth: Float = 0.22
let cameraRise: Float = 0.38
let frameHalfHeight: Float = 108
let cameraLead: Float = -88

func camera(_ t: Float) -> Camera {
    let target = V3(0, apexHeight(t) + cameraLead, 10)
    let pos: V3 = target + simd_normalize(V3(sin(cameraAzimuth), cameraRise, cos(cameraAzimuth))) * cameraDistance
    return Camera(position: pos, target: target, tanHalf: frameHalfHeight / cameraDistance)
}

/// Sweet pea foliage is glaucous: a bluish, waxy green. Jute netting.
let look = Look(leafTop: V3(0.075, 0.17, 0.075), leafUnder: V3(0.12, 0.23, 0.11),
                stem: V3(0.10, 0.21, 0.08), petiole: V3(0.11, 0.22, 0.09), tendril: V3(0.20, 0.33, 0.10),
                support: V3(0.50, 0.40, 0.26), supportKind: 1, period: loopRise, supportRadius: twineRadius,
                leafGloss: 0.6, keyDirection: V3(-0.45, 0.7, 0.55))

let ground = Ground(top: V3(0.87, 0.875, 0.855), bottom: V3(0.79, 0.80, 0.77))
