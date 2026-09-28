// A garden pea climbing garden netting with its tendrils, now in flower and
// setting pods, as numbers: where the stem, every leaf, flower and pod are at
// any hour, and what each tendril is doing. Nothing here touches the GPU; it
// is the part a test reads against the sources.
//
// Step 48 is step 47's pea — the same vine, net, camera and loop — with white
// flowers on stalks from the leaf axils and pods below them. What is new:
//
//   * the WHITE FLOWER is Mendel's white. Flower colour was one of Mendel's
//     seven characters; white is a mutation in gene A, which encodes a bHLH
//     transcription factor that switches on anthocyanin (Hellens et al.
//     2010). The dominant allele gives Mendel's violet-red flowers. The
//     flower is papilionaceous: a standard, two wings, and a keel.
//   * PEAS POLLINATE THEMSELVES IN THE BUD: the anthers shed their pollen on
//     the stigma inside the closed keel, before the flower opens (Mendel
//     1866) — which is why Mendel's lines bred true. Step 25 showed the same
//     for the common bean.
//   * the stages are a GRADIENT ALONG THE STEM, not a time-lapse of a pod
//     growing. Flower to full pod takes weeks; the film's loop is a day. So
//     each node's stage is set by its age — flowers above, withering petals
//     round young pods below them, full pods lower still — and the stages
//     move down with the nodes as the stem grows. Older nodes are further
//     along. This is compressed: in a real pea the gradient spans many more
//     nodes than the four in view (MODEL, and said on the frame).
//
// From step 47, the pea's own anatomy, drawn and tested:
//
//   * the STIPULES — the pair of leafy flaps where each leaf meets the stem —
//     are huge, larger than the leaflets, and clasp the stem. The sweet pea's
//     are small half-arrowheads. This is the pea's most recognisable feature.
//   * two PAIRS of rounded leaflets, not one; bluish, with a waxy bloom.
//   * SEVERAL TENDRILS a leaf: past the leaflets the rachis carries two more
//     pairs of pinnae, and they are tendrils, with a terminal one beyond.
//   * a ROUND stem, not a winged one, and no wing on the leaf stalk.
//   * the circumnutation is the pea's own: Darwin and Dutrochet timed it.
//
// A tendril sweeps round while it grows (circumnutation), touches a string,
// curls round it, and then — anchored now at both ends, the leaf and the
// string — its free length coils. A thread fixed at both ends cannot coil all
// one way without twisting, so it coils one way for half its length and the
// other way for the other half, with a short straight piece between: a
// PERVERSION. The coils are a spring, and as they form they draw the leaf in
// towards the net.
//
// Millimetres and hours. The net hangs in the plane z = 0, its strings on a
// 100 mm square mesh; the stem grows up in front of it, towards the camera
// (+z), midway between two vertical strings. +y is up, +x the viewer's right.
//
// Sources, checked for this step (step 47's, then step 48's):
//
//   Flora of China 10, Pisum (efloras.org, flora_id 2, taxon 125615): "Stem
//     often climbing by means of tendrils, terete, glabrous. Leaves
//     paripinnate with rachis terminating in a tendril; stipules leaflike,
//     cordate, larger than leaflets (to 10 cm); leaflets 1-3-paired, ovate to
//     elliptic". Pisum sativum (taxon 200012282): "stipules to 10 × 6 cm,
//     margin toothed; leaflets ovate, 2-7 × 1-4 cm."
//   Flora of Pakistan, Pisum sativum (efloras.org, flora_id 5, taxon
//     200012282): "Annual, often climbing, stem glabrous, glaucous. Leaf
//     paripinnately compound, rachis ending in a branched tendril, leaflets
//     2-8 ... stipules 1.5-8 cm long, obliquely ovate, toothed at least below,
//     semi-amplexicaul at the base". Var. sativum: "Flower white, stipules
//     without a reddish spot".
//     Sweet pea's stipules, for the mutant: "Stipules 1.5–2.5 cm long,
//     semisagittate" (Flora of Pakistan, Lathyrus odoratus, as in step 46).
//   Rungruangmaitree & Jiraungkoorskul, Pharmacogn. Rev. 11:39–42 (2017),
//     a review: "P. sativum is an herbaceous annual, with a climbing hollow
//     stem". Round from the Flora of China ("terete"); hollow from this review
//     only — a secondary source, flagged. Hollowness is not visible from
//     outside and is not drawn.
//   Gniwotta et al., Plant Physiol. 139:519–530 (2005): pea leaves carry
//     epicuticular wax crystals on both faces, platelets above and ribbons
//     below — the waxy bloom that makes the foliage glaucous.
//   Hofer et al., Plant Cell 21:420–428 (2009), "Tendril-less regulates
//     tendril formation in pea leaves": "The homeotic tendril-less (tl)
//     mutation in garden pea ... transforms tendrils into leaflets"; "the pea
//     tendril is a modified leaflet, inhibited from completing laminar
//     development by Tl", expressed "in organs emerging in the distal region
//     of the leaf primordium". So the tendrils sit at the leaf's distal pinna
//     positions, continuing the leaflets' series.
//   Tayeh et al., New Phytol. 243:1247–1261 (2024): "The afila (af) mutation
//     causes the replacement of leaflets by a branched mass of tendrils" —
//     the mirror case of tl, and the semi-leafless field peas. Not drawn:
//     this is an ordinary leafy pea.
//   Darwin, The Movements and Habits of Climbing Plants, 2nd ed. (1875),
//     Project Gutenberg #2485, ch. IV, Pisum sativum (checked in the text for
//     this step): "the internodes and tendrils revolve in ellipses ...
//     Dutrochet observed the completion of an ellipse in 1 hr. 20 m.; and I
//     saw one completed in 1 hr. 30 m."; with the petiole tied, a tendril
//     alone "completed a perfect ellipse in 1 hr. 30 m."; "the extremities of
//     their two or three pairs of branches become hooked"; "Ultimately the
//     lateral branches contract spirally, but not the middle or main stem."
//     Ch. V, spiral contraction: it "commences in half a day, or in a day or
//     two after their extremities have caught some object"; Echinocystis
//     "spirally contracted in 18 hrs."; a tendril caught at its tip
//     "invariably becomes twisted in one part in one direction, and in another
//     part in the opposite direction; the oppositely turned spires being
//     separated by a short straight portion"; "there are as many turns in the
//     one direction as in the other".
//   Gerbode, Puzey, McCormick & Mahadevan, Science 337:1087–1091 (2012): a
//     tendril held at both ends forms the perversion. Measured in cucumber;
//     the geometry — two helices of opposite hand, net twist zero — follows
//     for any thread anchored at both ends. Not measured in Pisum.
//   Engelberth, Adv. Space Res. 32:1611–1619 (2003): tendrils of Bryonia and
//     Pisum "respond to such a stimulus with a rapid coiling response ...
//     within minutes."
//   Gianoli, AoB Plants 7:plv013 (2015): tendril climbers use thin supports —
//     why the support here is string, not a pole.
//
//   Flora of China, Pisum sativum: "Raceme 1-3-flowered. Corolla variable in
//     color, usually white and/or purple, 15-35 mm ... Legume 2.5-12 × 1-2.5
//     cm. Seeds 2-10." Genus: "Corolla white or otherwise colored; standard
//     obovate ... Legume long elliptic, inflated, apex acute."
//   Flora of Pakistan, Pisum sativum: "peduncle ½ to twice as long as the
//     stipule, 1-3-flowered. Calyx 8-15 mm long ... Vexillum 16-30 mm long.
//     Fruit 40-70 mm long, 12-17 mm broad." Var. sativum: "Flower white ...
//     seeds usually more than 8 mm, globose". Genus: "Inflorescence a
//     solitary or few-flowered axillary raceme ... Vexillum broad, with a
//     claw. Wing bigger than keel and attached to the keel."
//   Hellens et al., PLoS ONE 5:e13230 (2010), "Identification of Mendel's
//     white flower character": "A is the factor determining anthocyanin
//     pigmentation in pea that was used by Gregor Mendel ... The A gene
//     encodes a bHLH transcription factor. The white flowered mutant allele
//     most likely used by Mendel is a simple G to A transition in a splice
//     donor site that leads to a mis-spliced mRNA with a premature stop
//     codon".
//   Mendel, "Versuche über Pflanzen-Hybriden" (1866), Bateson's translation
//     (MendelWeb): Pisum was chosen because "the fertilizing organs are
//     closely packed inside the keel and the anthers burst within the bud, so
//     that the stigma becomes covered with pollen even before the flower
//     opens." Not without exception: he names the beetle Bruchus pisi, which
//     opens the keel, and rare flowers whose parts wither open — so "peas
//     pollinate themselves in the bud", not "never cross". On colour: white
//     seed-coats go with white flowers; grey-brown with "the color of the
//     standards is violet, that of the wings purple" (the purple mutant);
//     the "green coloring of the unripe pod".
//   Zablatzká et al., Int. J. Mol. Sci. 22:4602 (2021): pea seed coats
//     sampled at "13, 21, 27, 30 days after anthesis" while the seeds were
//     still developing — flower to full pod is weeks, not the loop's day.

import Foundation
import simd

// MARK: - what can be broken on purpose

enum Mutant: String {
    case none
    case smallStipules = "small_stipules" // the sweet pea's small stipules on the garden pea
    case oneTendril = "one_tendril"       // a single tendril a leaf, as the sweet pea's step 46 drew
    case noPerversion = "no_perversion"   // the anchored free coil turns all one way
    case stemTendril = "stem_tendril"     // tendrils growing from the stem, not the leaf's pinnae
    case rewind                           // the loop closes by playing the growth backwards
    case purple                           // the flowers given Mendel's violet-red, the dominant A allele
    case podsAboveFlowers = "pods_above_flowers" // the gradient upside down: pods at the young nodes

    static var fromEnvironment: Mutant {
        Mutant(rawValue: ProcessInfo.processInfo.environment["TENDRIL_MUTANT"] ?? "none") ?? .none
    }
}

// MARK: - cited numbers

/// Flora of China: leaflets ovate, 2–7 × 1–4 cm, 1–3-paired. Drawn 30 × 20 mm,
/// rounded, two pairs.
let leafletSize = SIMD2<Float>(30, 20)
let leafletLengthRange: ClosedRange<Float> = 20...70
let leafletWidthRange: ClosedRange<Float> = 10...40
let leafletPairs: Int = 2
let leafletPairRange: ClosedRange<Int> = 1...3

/// Stipules: Flora of China "larger than leaflets", to 10 × 6 cm; Flora of
/// Pakistan 1.5–8 cm, obliquely ovate. Drawn 42 × 28 mm. The small_stipules
/// mutant gives it the sweet pea's 17 × 6.4 mm (step 46, Flora of Pakistan
/// 1.5–2.5 cm).
let stipuleSizeTrue = SIMD2<Float>(42, 28)
let stipuleLengthRange: ClosedRange<Float> = 15...100
let stipuleWidthMax: Float = 60
let sweetPeaStipuleSize = SIMD2<Float>(17, 6.4)

func stipuleSize(_ m: Mutant) -> SIMD2<Float> { m == .smallStipules ? sweetPeaStipuleSize : stipuleSizeTrue }

/// Tendril pinnae past the leaflets: Darwin's "two or three pairs of branches".
/// Two pairs, and the terminal tendril the rachis ends in (Flora of Pakistan:
/// "rachis ending in a branched tendril").
let tendrilPairs: Int = 2
let tendrilPairRange: ClosedRange<Int> = 2...3
func tendrilsPerLeaf(_ m: Mutant) -> Int { m == .oneTendril ? 1 : 2 * tendrilPairs + 1 }

/// Darwin's garden-pea ellipses: 80 min (Dutrochet) and 90 min (Darwin). The
/// pea's own numbers now, not borrowed.
let peaEllipseMinutesRange: ClosedRange<Float> = 80...90

/// Contraction after catching: begins about half a day after (Darwin: "in half
/// a day, or in a day or two"; Echinocystis 12–24 h) and is complete by a day
/// (Echinocystis: "spirally contracted in 18 hrs"). MODEL within those.
let contractionStartsAfter: Float = 12
let contractionCompleteAfter: Float = 24

/// Curling round the string once touched (Engelberth: "within minutes"). Two
/// and a half turns round the string take an hour here (MODEL).
let wrapHours: Float = 1.0
let wrapTurns: Float = 2.5

// MARK: - model sizes

/// Garden netting on a 100 mm square mesh, strings 2 mm thick — step 46's net,
/// so the two films match. MODEL.
let meshSpacing: Float = 100
let twineRadius: Float = 1.0
/// The stem grows 22 mm in front of the net, midway between two strings.
/// Round ("terete"), 2 mm across here (MODEL), and not winged.
let stemDepth: Float = 22
let stemRadius: Float = 2.0

/// One node per plastochron. MODEL, step 46's: 50 mm internodes, one every
/// 12 h. No source timed a pea's nodes for this.
let internodeLength: Float = 50
let plastochronHours: Float = 12
let growthRate: Float = internodeLength / plastochronHours

/// The loop: two nodes, a left leaf and a right, which is also one mesh
/// square, so the net repeats with it. 24 h of growth.
let nodesPerLoop: Int = 2
let loopHours: Float = plastochronHours * Float(nodesPerLoop)
let loopRise: Float = internodeLength * Float(nodesPerLoop)

/// 17 sweeps a loop, so a loop holds a whole number of them — 84.7 min each,
/// between Dutrochet's 80 and Darwin's 90.
let nutationsPerLoop: Int = 17
let nutationHours: Float = loopHours / Float(nutationsPerLoop)
/// How much of the stem below the tip sways, and by how far at the tip. MODEL.
let swayLength: Float = 40
let swayAmplitude: Float = 9

/// Leaf: petiole 24 mm to the first leaflet pair, then pinnae every 11 mm
/// along the rachis (MODEL). It opens over 10 h (MODEL, compressed as in steps
/// 45 and 46 — a real leaf takes days).
let petioleLength: Float = 24
let pinnaSpacing: Float = 11
let petioleRadius: Float = 0.8
let leafOpenHours: Float = 10
let budFraction: Float = 0.1
/// The pinna positions along the rachis: leaflet pairs first, then tendrils.
let pinnaCount: Int = leafletPairs + tendrilPairs
let rachisLength: Float = petioleLength + pinnaSpacing * Float(pinnaCount - 1)

/// Tendrils 0.8 mm thick (MODEL). The catching one — a lateral of the last
/// pair, since in the pea "the lateral branches alone contract" (Darwin) —
/// reaches the net; the others stay free, hooked at their tips (Darwin:
/// "extremities ... become hooked"). Their lengths MODEL: the terminal 26 mm,
/// the laterals 22 mm (Darwin: a full-grown tendril "considerably above two
/// inches" counts the whole branched tendril).
let tendrilRadius: Float = 0.4
let terminalTendrilLength: Float = 26
let lateralTendrilLength: Float = 22

/// The tendril touches the net 14 h after its node leaves the tip (MODEL, as
/// in step 46), 8 mm above its pinna (MODEL) — which with this leaf puts
/// every catch 11 mm clear below a cross-string (step 46's 18 mm would land
/// right-hand catches under one, and the wrap would climb into it; a test
/// checks).
let contactAge: Float = 14
let contactRise: Float = 8

/// The spring: 1.5 turns each side of the perversion (Darwin: as many one way
/// as the other), 1.8 mm coil radius, perversion 8 % of the length. MODEL.
let coilTurnsPerSide: Float = 1.5
let coilRadius: Float = 1.8
let perversionWidth: Float = 0.08

// MARK: - the timeline

/// 10 s at 20 frames a second, as in step 46.
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

/// What the frame says.
let nameCaption: String = "Pisum sativum, garden pea — the pea you eat"
let whiteCaption: String = "white flowers are Mendel's white: a broken gene A, a bHLH factor"
let selfCaption: String = "peas pollinate themselves in the bud, before the flower opens"
let gradientCaption: String = "flowers above, pods below: older nodes are further along"
let compressedCaption: String = "(a real pod takes weeks to fill; the stages here are compressed)"
let budLabel: String = "pollinated in the bud"

func timeCaption() -> String {
    String(format: "time-lapse of the climb: about %.0f hours in %.0f s", loopHours, loopSeconds)
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

// MARK: - nodes and leaves

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

/// The rachis's direction: out to the leaf's side, steeply up, and towards the
/// camera; young, it points up along the shoot. MODEL.
func petioleDirection(_ side: Float, _ age: Float) -> V3 {
    let o: Float = leafOpening(age)
    let elev: Float = 1.25 + (0.6 - 1.25) * o
    return simd_normalize(V3(side * cos(elev), sin(elev), 0.3))
}

/// Where the rachis ends — the last pinna pair, where the catching tendril
/// starts — with no pull on it.
func restLeafTip(_ n: NodeInfo, _ t: Float) -> V3 {
    stemPoint(n.y, t) + petioleDirection(n.side, n.age) * (rachisLength * leafGrowth(n.age))
}

/// The rachis as a curve from the node to the (possibly pulled) tip: a
/// quadratic Bézier with its control halfway along the unpulled direction.
struct Rachis {
    var node: V3
    var control: V3
    var tip: V3

    func point(_ s: Float) -> V3 {
        let r: Float = 1 - s
        let a: V3 = node * (r * r)
        let b: V3 = control * (2 * r * s)
        let c: V3 = tip * (s * s)
        return a + b + c
    }

    func tangent(_ s: Float) -> V3 {
        let a: Float = 2 * (1 - s)
        let b: Float = 2 * s
        let d0: V3 = (control - node) * a
        let d1: V3 = (tip - control) * b
        return simd_normalize(d0 + d1)
    }
}

/// Where pinna i sits on the rachis, as its parameter: 0 is the first leaflet
/// pair, pinnaCount − 1 the last tendril pair (the tip).
func pinnaParameter(_ i: Int) -> Float { (petioleLength + pinnaSpacing * Float(i)) / rachisLength }

func rachis(_ n: NodeInfo, _ t: Float, tip: V3) -> Rachis {
    let node: V3 = stemPoint(n.y, t)
    let dir: V3 = petioleDirection(n.side, n.age)
    let len: Float = rachisLength * leafGrowth(n.age)
    return Rachis(node: node, control: node + dir * (0.5 * len), tip: tip)
}

/// The birth hour of node n.
func birthHour(_ n: NodeInfo, _ t: Float) -> Float { t - n.age }

/// Everything fixed about a tendril's catch, worked out from where the leaf
/// tip is at the moment of contact (by then the leaf is full-grown and out of
/// the swaying zone, so it has stopped moving).
struct Catch {
    /// The rachis tip, unpulled: the catching tendril's base.
    var leafTip: V3
    /// The string's axis at the contact height, and the point on the
    /// tendril's axis that touches it.
    var stringAxis: V3
    var contact: V3
    var direction: V3
    var reach: Float
    var wrapSense: Float
}

func tendrilCatch(_ n: NodeInfo, _ t: Float) -> Catch {
    let tc: Float = birthHour(n, t) + contactAge
    let nc = NodeInfo(index: n.index, y: n.y, age: contactAge, side: n.side)
    let b: V3 = restLeafTip(nc, tc)
    let axis = V3(n.side * meshSpacing / 2, b.y + contactRise, 0)
    let toward: V3 = simd_normalize(V3(b.x - axis.x, 0, b.z - axis.z))
    let contact: V3 = axis + toward * (twineRadius + tendrilRadius)
    let dir: V3 = simd_normalize(contact - b)
    return Catch(leafTip: b, stringAxis: axis, contact: contact, direction: dir,
                 reach: simd_distance(contact, b), wrapSense: n.side)
}

/// How far the tendrils have grown, 0…1: full exactly at contact.
func tendrilGrowth(_ age: Float) -> Float { 0.12 + 0.88 * smoothstep(0, contactAge, age) }

/// The stage of the catching tendril, for the tests and the labels.
enum Stage: Int { case searching = 0, wrapping, anchored, contracting, sprung }

func stage(_ age: Float) -> Stage {
    if age < contactAge { return .searching }
    if age < contactAge + wrapHours { return .wrapping }
    if age < contactAge + contractionStartsAfter { return .anchored }
    if age < contactAge + contractionCompleteAfter { return .contracting }
    return .sprung
}

func contraction(_ age: Float) -> Float {
    smoothstep(contactAge + contractionStartsAfter, contactAge + contractionCompleteAfter, age)
}

/// The free coil between the rachis tip and the string: two helices of
/// opposite hand joined by a straight perversion, drawn along the chord from
/// `a` to `b`. θ, the angle round the chord, climbs to the middle and comes
/// back down to where it started — so the net twist between the two anchored
/// ends is zero, as it must be. The no-perversion mutant winds all one way.
/// (Step 46's curve, unchanged.)
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

/// The chord the free coil spans: the coil keeps its length, so the more it
/// coils the shorter the chord — that shortening is the pull. Bisection.
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

/// A free tendril: straight out, then its tip curled into a hook.
func hookedTendril(from base: V3, direction d: V3, length l: Float, curl: V3) -> [V3] {
    let p1: V3 = base + d * (0.72 * l)
    let hook: V3 = simd_normalize(d * 0.35 + curl - V3(0, 0.35, 0))
    return bezier(base, p1, p1 + hook * (0.32 * l), segments: 8)
}

struct TendrilGeometry {
    /// The rachis, pulled, and its pinna points.
    var rachis: Rachis
    var pinnae: [V3]
    /// The catching tendril's base: the rachis tip (or the stem, mutant).
    var base: V3
    var leafTip: V3
    /// The catching tendril, from its base through the free coil and round the
    /// string once caught; and its free part and wrap.
    var catching: [V3]
    var free: [V3]
    var wrap: [V3]
    /// The other tendrils, free and hooked; each starts at its base.
    var others: [[V3]]
    var stage: Stage
    var contraction: Float
    var stringAxis: V3
}

/// The plane the leaf spreads in, at rachis parameter s: unit normal.
func leafNormal(_ tangent: V3) -> V3 {
    let facing: V3 = simd_normalize(V3(0, 0.75, 0.66))
    return simd_normalize(facing - tangent * simd_dot(facing, tangent))
}

func tendril(_ n: NodeInfo, _ t: Float, _ m: Mutant) -> TendrilGeometry {
    let k: Catch = tendrilCatch(n, t)
    let g: Float = tendrilGrowth(n.age)
    let st: Stage = stage(n.age)
    let c: Float = contraction(n.age)
    var leafTip: V3 = restLeafTip(n, t)
    var free: [V3] = []
    var wrap: [V3] = []
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
        let dir: V3 = simd_normalize(k.direction + ellipse(t) - ellipse(tc))
        free = [leafTip, leafTip + dir * (k.reach * g)]
    } else {
        // Anchored: the free length coils and draws the leaf in.
        let chord: Float = coilChord(length: k.reach, contraction: c, mutant: m)
        let away: V3 = simd_normalize(k.leafTip - k.contact)
        leafTip = k.contact + away * chord
        free = coilCurve(from: leafTip, to: k.contact, contraction: c, mutant: m)
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
    let r: Rachis = rachis(n, t, tip: leafTip)
    let pinnae: [V3] = (0..<pinnaCount).map { r.point(pinnaParameter($0)) }
    let stemBase: V3 = stemPoint(n.y, t) + V3(n.side * stemRadius, 0, 0)
    let base: V3 = m == .stemTendril ? stemBase : leafTip
    if m == .stemTendril { free[0] = stemBase }

    // The other tendrils: the pair on the pinna before the tip, the other
    // lateral at the tip, and the terminal one carrying the rachis on.
    var others: [[V3]] = []
    if m != .oneTendril {
        for i in leafletPairs..<pinnaCount {
            let s: Float = pinnaParameter(i)
            let tg: V3 = r.tangent(s)
            let nrm: V3 = leafNormal(tg)
            let across: V3 = simd_normalize(simd_cross(nrm, tg))
            let at: V3 = m == .stemTendril ? stemBase : pinnae[i]
            let lateralSides: [Float] = i == pinnaCount - 1 ? [-n.side] : [1, -1]
            for sd in lateralSides {
                let d0: V3 = simd_normalize(tg * 0.75 + across * (sd * 0.66))
                others.append(hookedTendril(from: at, direction: d0, length: lateralTendrilLength * g,
                                            curl: across * sd))
            }
            if i == pinnaCount - 1 {
                let d0: V3 = simd_normalize(tg + nrm * 0.25)
                others.append(hookedTendril(from: at, direction: d0, length: terminalTendrilLength * g,
                                            curl: across * (-n.side)))
            }
        }
    }
    let catching: [V3] = free + wrap.dropFirst()
    return TendrilGeometry(rachis: r, pinnae: pinnae, base: base, leafTip: leafTip, catching: catching, free: free,
                           wrap: wrap, others: others, stage: st, contraction: c, stringAxis: k.stringAxis)
}

/// The bases of all of a leaf's tendrils.
func tendrilBases(_ td: TendrilGeometry) -> [V3] { [td.base] + td.others.map { $0[0] } }

func leafAndTendrils(_ n: NodeInfo, _ t: Float, _ m: Mutant, into s: inout PlantScene) {
    let g: Float = leafGrowth(n.age)
    let o: Float = leafOpening(n.age)
    let node: V3 = stemPoint(n.y, t)
    let td: TendrilGeometry = tendril(n, t, m)

    // The rachis, round and unwinged, through every pinna point exactly.
    var params: [Float] = (0...8).map { Float($0) / 8 }
    for i in 0..<pinnaCount { params.append(pinnaParameter(i)) }
    params.sort()
    var pts: [V3] = []
    for p in params where pts.isEmpty || simd_distance(pts[pts.count - 1], td.rachis.point(p)) > 1e-3 {
        pts.append(td.rachis.point(p))
    }
    let radii: [Float] = pts.indices.map { i in petioleRadius * (1 - 0.35 * Float(i) / Float(max(pts.count - 1, 1))) }
    s.chains.append(Chain(points: pts, radii: radii, material: .petiole))

    // Stipules: a big leafy pair at the node, clasping the stem from in front
    // — obliquely ovate, pointed (the machinery's ovate outline); the flora's
    // marginal teeth are not drawn.
    let sz: SIMD2<Float> = stipuleSize(m)
    let sg: Float = max(g, 0.3)
    for sd in [Float(1), -1] {
        let u: V3 = simd_normalize(V3(0.62 * sd, 1, 0.3))
        let w0: V3 = V3(0, 0.25, 1)
        let w: V3 = simd_normalize(w0 - u * simd_dot(w0, u))
        let origin: V3 = node + V3(sd * 0.6, -4, stemRadius + 0.9)
        s.leaflets.append(Leaflet(origin: origin, u: u, v: simd_cross(w, u), w: w, length: sz.x * sg,
                                  halfWidth: sz.y / 2 * sg, fold: 0.35 - 0.1 * o, droop: 0.05,
                                  shape: .beanOvate, material: .stipule))
    }
    // Two pairs of rounded leaflets, spreading in the leaf's plane as it opens.
    for i in 0..<leafletPairs {
        let sp: Float = pinnaParameter(i)
        let tg: V3 = td.rachis.tangent(sp)
        let nrm: V3 = leafNormal(tg)
        let across: V3 = simd_normalize(simd_cross(nrm, tg))
        let angle: Float = 0.25 + 0.75 * o
        for sd in [Float(1), -1] {
            let u: V3 = simd_normalize(tg * cos(angle) + across * (sd * sin(angle)))
            let w: V3 = simd_normalize(nrm - u * simd_dot(nrm, u))
            s.leaflets.append(Leaflet(origin: td.pinnae[i], u: u, v: simd_cross(w, u), w: w,
                                      length: leafletSize.x * g, halfWidth: leafletSize.y / 2 * g,
                                      fold: 1.3 + (0.2 - 1.3) * o, droop: 0.1 * o, shape: .peaElliptic,
                                      material: .leaf))
        }
    }
    // The tendrils.
    let r: Float = tendrilRadius
    s.chains.append(Chain(points: td.catching, radii: Array(repeating: r, count: td.catching.count), material: .tendril))
    for b in td.others { s.chains.append(Chain(points: b, radii: Array(repeating: r * 0.9, count: b.count), material: .tendril)) }
}

// MARK: - flowers and pods (step 48)

/// Flora of China and Flora of Pakistan: "1-3-flowered". Two to a stalk.
let flowersPerStalk: Int = 2
let flowersPerStalkRange: ClosedRange<Int> = 1...3
/// Flora of Pakistan: peduncle "½ to twice as long as the stipule". Drawn
/// 44 mm against the 42 mm stipule.
let peduncleLength: Float = 44
func peduncleRange() -> ClosedRange<Float> { (0.5 * stipuleSizeTrue.x)...(2 * stipuleSizeTrue.x) }
/// Pedicel 7 mm (MODEL; the floras give none).
let pedicelLength: Float = 7
/// Flora of Pakistan: calyx 8–15 mm. Drawn 9 mm, a cone opening to 3 mm
/// radius (MODEL).
let calyxLength: Float = 9
let calyxRange: ClosedRange<Float> = 8...15
/// Flora of Pakistan: vexillum 16–30 mm; Flora of China: corolla 15–35 mm.
/// Standard drawn 24 mm long, 24 wide (MODEL width: "broad", "obovate").
let standardSize = SIMD2<Float>(24, 24)
let standardRange: ClosedRange<Float> = 16...30
/// Wings and keel (MODEL sizes), the wings bigger than the keel (Flora of
/// Pakistan: "Wing bigger than keel").
let wingSize = SIMD2<Float>(18, 11)
let keelSize = SIMD2<Float>(14, 9)
/// Flora of Pakistan: fruit 40–70 × 12–17 mm (Flora of China 2.5–12 × 1–2.5
/// cm). Drawn 13 mm broad, 60 mm from base to tip (about 64 mm along its
/// curve, which the test measures). Flora of China: seeds 2–10; drawn 6 a pod.
/// Both flowers on a stalk set a pod (MODEL; a stalk carries 1–3 flowers).
let podSize = SIMD2<Float>(60, 13)
let podLengthRange: ClosedRange<Float> = 40...70
let podWidthRange: ClosedRange<Float> = 12...17
let seedsPerPod: Int = 6
let seedsPerPodRange: ClosedRange<Int> = 2...10
/// How far each pea bulges the pod wall, as a fraction of the pod's radius
/// (MODEL): between peas the pod is 88 % as wide as over one.
let seedBulge: Float = 0.12

/// The stages by node age, in hours since the node left the tip. All MODEL,
/// and compressed: flower to full pod takes weeks (Zablatzká 2021: seeds
/// still filling 30 days after anthesis). No source here timed a pea bud's
/// opening. Here the stages are laid out over the four nodes in view.
let budAppears: Float = 7        // the flower stalk shows, once its leaf has swung out of the way
let flowerOpens: Float = 9       // the standard starts to lift (self-pollination is already done)
let flowerOpen: Float = 12       // fully open
let witherStarts: Float = 16     // petals start to wither; the pod starts out of the keel
let witherEnds: Float = 24       // petals papery, shrunken
let petalsFall: Float = 30       // the withered corolla is gone
let podFull: Float = 32          // the pod at full length
let seedsSwell: ClosedRange<Float> = 22...38

/// The age a node's flowers and pods go by. It is the node's age; the
/// pods_above_flowers mutant turns the gradient upside down.
func reproductiveAge(_ age: Float, _ m: Mutant) -> Float {
    m == .podsAboveFlowers ? max(0, 44 - age) : age
}

enum FlowerStage: Int { case none = 0, bud, opening, open, withering, pod }

func flowerStage(_ a: Float) -> FlowerStage {
    if a < budAppears { return .none }
    if a < flowerOpens { return .bud }
    if a < flowerOpen { return .opening }
    if a < witherStarts { return .open }
    if a < petalsFall { return .withering }
    return .pod
}

struct Flower {
    /// Where the flower sits on its pedicel, and its frame: `forward` the way
    /// the keel points, `up` the standard's side, `across` the third.
    var base: V3
    var forward: V3
    var up: V3
    var across: V3
    /// Standard, wing, wing, keel — empty once the corolla has fallen.
    var petals: [Leaflet]
    var calyx: Chain
    var pedicel: Chain
    /// The pod, if there is one: its axis and radii, and where the peas sit.
    var pod: Chain?
    var seedCentres: [Float]
}

struct Inflorescence {
    var node: NodeInfo
    var stage: FlowerStage
    var peduncle: Chain
    var flowers: [Flower]
}

/// The pod's radius at s (0…1 along it): pointed at both ends ("apex acute"),
/// swelling over each pea as the seeds fill.
func podRadius(_ s: Float, swell: Float, centres: [Float], halfSpacing: Float) -> Float {
    let ends: Float = smoothstep(0, 0.16, s) * (1 - smoothstep(0.84, 1, s))
    var bump: Float = 0
    for c in centres {
        let x: Float = (s - c) / halfSpacing
        bump = max(bump, max(0, 1 - x * x))
    }
    let wall: Float = 1 - seedBulge + seedBulge * bump * swell
    let r0: Float = podSize.y / 2
    return max(r0 * ends * wall * (1 - (1 - swell) * 0.25), 0.5)
}

func inflorescence(_ n: NodeInfo, _ t: Float, _ m: Mutant) -> Inflorescence? {
    let a: Float = reproductiveAge(n.age, m)
    let st: FlowerStage = flowerStage(a)
    if st == .none { return nil }
    let side: Float = n.side
    let node: V3 = stemPoint(n.y, t)
    // The stalk rises from the axil behind the stipules, then leans out and
    // towards the camera above them. MODEL path. It grows with its leaf and
    // stipules — the same factor — so it clears them at every age as it does
    // full-grown (a test probes it).
    let grow: Float = max(leafGrowth(n.age), 0.3)
    let reach: Float = peduncleLength / 45.6
    let start: V3 = node + V3(side * (stemRadius + 0.2), 3, -0.9)
    let end: V3 = start + V3(side * 15, 36, 22) * (grow * reach)
    let control: V3 = start + V3(side * 10, 30, -4) * grow
    let pedPts: [V3] = bezier(start, control, end, segments: 10)
    let peduncle = Chain(points: pedPts, radii: pedPts.indices.map { _ in 0.7 }, material: .petiole)

    let open: Float = smoothstep(flowerOpens, flowerOpen, a)
    let wither: Float = smoothstep(witherStarts, witherEnds, a)
    let bloom: Float = 0.45 + 0.55 * smoothstep(budAppears, flowerOpen, a)
    var flowers: [Flower] = []
    for j in 0..<flowersPerStalk {
        let fj: Float = Float(j)
        // Pedicels fan out from the stalk's tip; each flower faces out and
        // towards the camera, held level (MODEL).
        let pdir: V3 = simd_normalize(V3(side * (0.35 + 0.5 * fj), 0.25 - 0.55 * fj, 0.8))
        let pl: Float = pedicelLength * grow
        let base: V3 = end + pdir * pl
        let pedicel = Chain(points: [end, base], radii: [0.55, 0.5], material: .petiole)
        let forward: V3 = simd_normalize(V3(side * (0.55 + 0.25 * fj), -0.05 - 0.15 * fj, 0.8))
        let up0 = V3(0, 1, 0)
        let up: V3 = simd_normalize(up0 - forward * simd_dot(up0, forward))
        let across: V3 = simd_cross(forward, up)
        // The calyx: a green cone round the petals' claws.
        let calyxTop: V3 = base + forward * (calyxLength * bloom)
        let calyx = Chain(points: [base, base + forward * (0.5 * calyxLength * bloom), calyxTop],
                          radii: [1.0, 2.3 * bloom, 3.0 * bloom], material: .petiole)
        var petals: [Leaflet] = []
        if a < petalsFall {
            let shrink: Float = bloom * (1 - 0.45 * wither)
            let crumple: Float = 0.9 * wither
            let claw: V3 = calyxTop - forward * 2.5
            // Standard: in the bud folded forward over the wings and keel;
            // opening, it lifts and reflexes back; withering, it folds again.
            let lift: Float = open * (1 - 0.6 * wither)
            let su: V3 = simd_normalize(forward * (0.95 - 1.2 * lift) + up * (0.35 + 0.65 * lift))
            let sw0: V3 = forward * (1 - lift) - up * (0.3 * (1 - lift)) + forward * lift
            let sw: V3 = simd_normalize(sw0 - su * simd_dot(sw0, su))
            petals.append(Leaflet(origin: claw + up * 1.2, u: su, v: simd_cross(sw, su), w: sw,
                                  length: standardSize.x * shrink, halfWidth: standardSize.y / 2 * shrink,
                                  fold: 1.35 - 1.0 * open + crumple, droop: -0.08 * open, shape: .standard,
                                  material: .petal, tint: wither))
            // Wings, one each side of the keel, spreading a little when open.
            for sd in [Float(1), -1] {
                let wu: V3 = simd_normalize(forward + up * 0.18 + across * (sd * 0.22 * open))
                let ww0: V3 = across * sd
                let ww: V3 = simd_normalize(ww0 - wu * simd_dot(ww0, wu))
                petals.append(Leaflet(origin: claw + across * (sd * 1.4) + up * 0.4, u: wu, v: simd_cross(ww, wu),
                                      w: ww, length: wingSize.x * shrink, halfWidth: wingSize.y / 2 * shrink,
                                      fold: 0.25 + crumple, droop: 0.06, shape: .wing, material: .petal, tint: wither))
            }
            // Keel: one blade folded along its midrib into a boat, opening
            // upwards, between the wings.
            let ku: V3 = simd_normalize(forward + up * 0.05)
            let kw: V3 = simd_normalize(up - ku * simd_dot(up, ku))
            petals.append(Leaflet(origin: claw - up * 0.6, u: ku, v: simd_cross(kw, ku), w: kw,
                                  length: keelSize.x * shrink, halfWidth: keelSize.y / 2 * shrink,
                                  fold: 1.15, droop: 0.04, shape: .keel, material: .petal, tint: wither))
        }
        // The pod: out of the keel as the petals wither, lengthening and
        // hanging as it grows; the peas swell it later.
        var pod: Chain? = nil
        var centres: [Float] = []
        if a > witherStarts {
            let g: Float = 0.08 + 0.92 * smoothstep(witherStarts, podFull, a)
            let swell: Float = smoothstep(seedsSwell.lowerBound, seedsSwell.upperBound, a)
            let len: Float = podSize.x * g
            let hang: Float = smoothstep(witherStarts, podFull, a)
            let down0: V3 = V3(side * (0.1 + 0.4 * fj), -1, 0.15 + 0.3 * fj)
            let tipDir: V3 = simd_normalize(forward * (1 - 0.85 * hang) + simd_normalize(down0) * (0.2 + 1.1 * hang))
            let p0: V3 = calyxTop - forward * 1.5
            let p1: V3 = p0 + forward * (0.3 * len)
            let p2: V3 = p0 + tipDir * len
            let spacing: Float = 0.72 / Float(seedsPerPod)
            centres = (0..<seedsPerPod).map { 0.14 + (Float($0) + 0.5) * spacing }
            let perSeed: Int = 8
            let segs: Int = seedsPerPod * perSeed + 12
            var pts: [V3] = []
            var rad: [Float] = []
            for i in 0...segs {
                let s: Float = Float(i) / Float(segs)
                let q: Float = 1 - s
                let pa: V3 = p0 * (q * q)
                let pb: V3 = p1 * (2 * q * s)
                let pc: V3 = p2 * (s * s)
                pts.append(pa + pb + pc)
                rad.append(podRadius(s, swell: swell, centres: centres, halfSpacing: spacing / 2) * (0.35 + 0.65 * g))
            }
            pod = Chain(points: pts, radii: rad, material: .pod)
        }
        flowers.append(Flower(base: base, forward: forward, up: up, across: across, petals: petals, calyx: calyx,
                              pedicel: pedicel, pod: pod, seedCentres: centres))
    }
    return Inflorescence(node: n, stage: st, peduncle: peduncle, flowers: flowers)
}

func addInflorescence(_ f: Inflorescence, into s: inout PlantScene) {
    s.chains.append(f.peduncle)
    for fl in f.flowers {
        s.chains.append(fl.pedicel)
        s.chains.append(fl.calyx)
        s.leaflets += fl.petals
        if let p = fl.pod { s.chains.append(p) }
    }
}

/// The peas in a pod, counted from its drawn outline: the bulges in its
/// radius, between its pointed ends.
func bulges(_ pod: Chain) -> Int {
    var count: Int = 0
    let r: [Float] = pod.radii
    if r.count < 3 { return 0 }
    for i in 1..<(r.count - 1) where r[i] > r[i - 1] + 1e-4 && r[i] >= r[i + 1] {
        count += 1
    }
    return count
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

func peaScene(_ t: Float, _ m: Mutant) -> PlantScene {
    var s = PlantScene()
    net(t, into: &s)
    let st: [V3] = stemPoints(t)
    // Round: a plain chain of capsules, no wing ribbons.
    var radii: [Float] = []
    for k in 0..<st.count {
        let fromTip: Float = Float(st.count - 1 - k) * stemStep
        radii.append(stemRadius * (0.45 + 0.55 * smoothstep(0, 30, fromTip)))
    }
    s.chains.append(Chain(points: st, radii: radii, material: .stem))
    for n in nodes(t) {
        leafAndTendrils(n, t, m, into: &s)
        if let f = inflorescence(n, t, m) { addInflorescence(f, into: &s) }
    }
    return s
}

// MARK: - the camera

/// Step 46's camera exactly, so the two films play side by side.
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

/// Pea foliage and stem are glaucous (Flora of Pakistan; wax crystals,
/// Gniwotta 2005): a bluish, matt, bloomy green — bluer and duller than step
/// 46's sweet pea. Colours MODEL. Jute netting as in step 46.
///
/// Petals: white, var. sativum (Flora of Pakistan: "Flower white") — the a
/// allele, no anthocyanin (Hellens 2010). Withering to a papery tan, and the
/// pod a clear young green (Mendel: "green coloring of the unripe pod").
/// Albedos MODEL. The purple mutant gives the petals Mendel's dominant colour
/// ("the color of the standards is violet, that of the wings purple"), drawn
/// as one violet-purple for all four petals.
let whitePetal = V3(0.64, 0.645, 0.62)
let mendelPurple = V3(0.30, 0.07, 0.34)

func lookFor(_ m: Mutant) -> Look {
    Look(leafTop: V3(0.052, 0.115, 0.078), leafUnder: V3(0.10, 0.17, 0.12),
         stem: V3(0.075, 0.15, 0.09), petiole: V3(0.08, 0.16, 0.095), tendril: V3(0.15, 0.27, 0.12),
         support: V3(0.50, 0.40, 0.26), supportKind: 1, period: loopRise, supportRadius: twineRadius,
         leafGloss: 0.15, keyDirection: V3(-0.45, 0.7, 0.55),
         petal: m == .purple ? mendelPurple : whitePetal, petalWithered: V3(0.40, 0.34, 0.22),
         pod: V3(0.07, 0.17, 0.05))
}

let ground = Ground(top: V3(0.87, 0.875, 0.855), bottom: V3(0.79, 0.80, 0.77))
