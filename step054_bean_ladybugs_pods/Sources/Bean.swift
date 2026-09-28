// A pole bean climbing a pole by twining, as numbers: where the stem is at any
// hour, where its leaves are and how far each has unfolded, and where the
// camera is. Nothing here touches the GPU; it is the part a test reads
// against the sources.
//
// Step 54 is step 45's bean — the same stem, pole, leaves, camera and loop —
// now in flower: white flowers on racemes in the leaf axils, and small young
// pods below them (the flowers and pods are at the end of this file, with
// their sources). The ladybirds walking on it are in Ladybug.swift. Step 45's
// header follows, unchanged.
//
// A bean has no tendrils. It climbs by twining: the growing tip bows over and
// sweeps round in circles (circumnutation); when it meets a pole, the part
// that touches is stopped while the free part beyond goes on sweeping, so the
// stem winds round the pole in a helix — always the same way round for the
// species. The plant grows up; it does not pull itself up.
//
// Millimetres and hours. The pole stands on the y axis (x = z = 0), +y up, and
// the camera looks along −z from +z, so +x is the viewer's right.
//
// Sources, checked for this step:
//
//   Darwin, The Movements and Habits of Climbing Plants, 2nd ed. (1875),
//     Project Gutenberg #2485, ch. I:
//       "Phaseolus vulgaris (Leguminosæ), in greenhouse, moves against the
//       sun. May, 1st circle was made in 2 h 0 m; 2nd 1 h 55 m; 3rd 1 h 55 m",
//       and "Phaseolus vulgaris three at the average of 1 hr. 57 m."
//       "A greater number of twiners revolve in a course opposed to that of
//       the sun, or to the hands of a watch ... and, consequently, the
//       majority ... ascend their supports from left to right."
//       "When a revolving shoot strikes a stick, it winds round it rather more
//       slowly than it revolves. For instance, a shoot of the Ceropegia
//       revolved in 6 hrs., but took 9 hrs. 30 m. to make one complete spire
//       round a stick; Aristolochia gigas revolved in about 5 hrs., but took
//       9 hrs. 15 m."
//       "With all the many plants which were allowed freely to ascend a
//       support, the terminal internodes made at first a close spire ... but
//       as the penultimate internodes grew in length, they pushed themselves
//       up ... round the stick, and the spire became more open."
//       On thickness (citing Mohl): "the Phaseolus [multiflorus] twined round a
//       support of the above thickness [3 to 4 inches], but failed in twining
//       round one 9 inches in diameter"; and his own "kidney-beans ... smooth
//       rods of iron and glass, one-third of an inch in diameter."
//   Silk & Hubbard, J. Biomech. 24:599–606 (1991), morning glory: "When
//     removed from the pole, the helical stem forms a coil of smaller radius,
//     smaller wavelength and larger torsion" — so a wound stem is held open by
//     the pole and presses on it; "the twining stem puts itself into tension
//     and uses a helical geometry to generate contact forces".
//   Isnard, Cobb, Holbrook, Zwieniecki & Dumais, Proc. R. Soc. B 276:2643–2650
//     (2009), "Tensioning the helix": "a delayed stem tensioning ... creates
//     the squeezing forces necessary for twining plants to ascend their
//     supports" (measured in Dioscorea bulbifera, proposed as general).
//   Gianoli, AoB Plants 7:plv013 (2015), "The behavioural ecology of climbing
//     plants": beyond some support diameter twiners "are unable to maintain
//     tensional forces and therefore lose attachment" (Putz 1984 and others);
//     "leaf expansion is delayed relative to stem extension in erect leader
//     shoots of twiners" (Raciborski 1900 and others).
//   Flora of China 10 (Phaseolus vulgaris): "Annual herbs, twining or
//     suberect"; "leaflets broadly ovate or obovate-rhombic, lateral ones
//     oblique, 4–16 × 2.5–11 cm ... apex acuminate". Flora of Pakistan:
//     "Leaf trifoliolate, petiole 4–9 cm long, leaflets 4.5–15 cm long,
//     2.5–6.5 cm broad ... petiolule 1.5–2.5 mm long". (Both at efloras.org.)
//
// WHICH WAY ROUND. "Against the sun" is Darwin's phrase for anticlockwise, and
// he means it seen from above: he equates it with "opposed ... to the hands of
// a watch" and says such twiners "ascend their supports from left to right".
// Here that is one statement three ways:
//   * seen from above, looking down the pole, the stem goes round
//     ANTICLOCKWISE as it climbs;
//   * seen from the side, on the face of the pole towards you, it rises from
//     LEFT TO RIGHT;
//   * it is a RIGHT-HANDED helix in the mathematical sense (a Z-helix, like an
//     ordinary screw thread). Botanists have used "left-handed" and
//     "right-handed" both ways for this; the words above do not depend on
//     which.
// The tests check all three on the drawn stem.

import Foundation
import simd

// MARK: - what can be broken on purpose

/// Each mutant must make the test suite fail; `make mutants` checks that.
enum Mutant: String {
    case none
    case leftHanded = "left_handed"     // twines clockwise from above: the wrong way for the species
    case throughPole = "through_pole"   // the stem's axis on the pole's surface: half of it inside the wood
    case rewind                         // the loop closes by playing the growth (and the ladybirds) backwards
    // Step 54:
    case purple                         // the petals recoloured violet
    case podsAboveFlowers = "pods_above_flowers" // the gradient upside down: pods at the young nodes
    case eightLegs = "eight_legs"       // a ladybird drawn with four pairs of legs, a spider's count
    case slidingFeet = "sliding_feet"   // feet held in the beetle's frame, so they skate over the plant

    static var fromEnvironment: Mutant {
        Mutant(rawValue: ProcessInfo.processInfo.environment["BEAN_MUTANT"] ?? "none") ?? .none
    }
}

// MARK: - cited numbers

/// Darwin's three timed revolutions of Phaseolus vulgaris, in minutes, and the
/// mean he gives: "three at the average of 1 hr. 57 m."
let darwinRevolutionMinutes: [Float] = [120, 115, 115]
let darwinAverageMinutes: Float = 117

/// The revolution period used: the mean of Darwin's three.
let revolutionHours: Float = darwinRevolutionMinutes.reduce(0, +) / Float(darwinRevolutionMinutes.count) / 60

/// Winding round a support is slower than revolving freely: Darwin's two
/// timed cases, Ceropegia 9 h 30 m against 6 h and Aristolochia gigas 9 h 15 m
/// against about 5 h. He timed no bean this way, so the bean takes the mean of
/// the two ratios (MODEL, flagged: the ratio is borrowed from other genera).
let darwinWindingRatios: [Float] = [9.5 / 6.0, 9.25 / 5.0]
let windingSlowdown: Float = darwinWindingRatios.reduce(0, +) / Float(darwinWindingRatios.count)

/// Hours for the stem to make one turn round the pole.
let gyreHours: Float = revolutionHours * windingSlowdown

/// Darwin: twiners "revolve in a course opposed to that of the sun" and
/// Phaseolus vulgaris "moves against the sun" — anticlockwise from above.
/// +1 here is anticlockwise seen from above.
func handedness(_ m: Mutant) -> Float { m == .leftHanded ? -1 : 1 }

/// The pole. MODEL: 25 mm, a garden cane or thin hazel pole. Well inside what
/// beans are recorded twining round — Darwin's kidney beans climbed 1/3-inch
/// (8.5 mm) rods, and Mohl's runner bean managed 3–4 inch (76–102 mm) sticks
/// out of doors but not a 9-inch (229 mm) one — and thick enough to read as a
/// pole beside a 4 mm stem. (No source found for a typical pole-bean pole's
/// diameter; garden guides give heights, 2–2.5 m, not widths.)
let poleDiameter: Float = 25
let poleRadius: Float = poleDiameter / 2
/// Darwin's limits, for the test: twined round up to 4 inches, failed at 9.
let twinedDiameterRecorded: Float = 101.6
let failedDiameterRecorded: Float = 228.6

/// The stem's radius where it is wound. MODEL: a 4 mm stem.
let stemRadius: Float = 2.0
/// And at the very tip. MODEL.
let apexRadius: Float = 0.7

/// The rise of one turn of the mature helix. MODEL: 100 mm on this pole, an
/// ascent angle of 48° — Darwin gives no bean pitch, and no source found
/// does; in the hop the ascent angle falls as the support thickens (Gianoli
/// 2015).
let maturePitch: Float = 100
/// Darwin: the terminal internodes "made at first a close spire", which opened
/// as they grew. MODEL: the new turn is laid at half the mature pitch, and
/// reaches the mature pitch one turn behind the winding front.
let closeSpireFraction: Float = 0.5
let openingTurns: Float = 1.0
/// Silk & Hubbard: off the pole the helix is tighter than the pole; Isnard et
/// al.: the squeeze comes from a delayed tensioning. MODEL: the newest turn is
/// laid 1.5 mm clear of the wood and tightens onto it over one turn.
let looseGap: Float = 1.5
let tighteningTurns: Float = 1.0

/// The free, sweeping part of the shoot beyond the last point on the pole.
/// MODEL: about one close turn's worth of stem, 110 mm.
let freeLength: Float = 110

/// Turns of the pole per loop, and nodes per loop. Four nodes to three turns
/// puts each leaf three-quarters of a turn round from the last, so the leaves
/// spiral round the pole, and the pattern repeats exactly each loop.
let gyresPerLoop: Int = 3
let nodesPerLoop: Int = 4

/// The axis of the wound stem's helix sits at the pole's radius plus the
/// stem's: touching, not inside. The through-pole mutant puts it on the wood.
func contactRadius(_ m: Mutant) -> Float { m == .throughPole ? poleRadius : poleRadius + stemRadius }

/// Stem length in one mature turn.
func matureGyreLength(_ m: Mutant = .none) -> Float {
    let c: Float = 2 * Float.pi * contactRadius(m)
    return (c * c + maturePitch * maturePitch).squareRoot()
}

/// Internode length on the mature helix: three-quarters of a turn's stem.
/// MODEL: ~10 cm; the floras give no internode length for the bean.
let internodeLength: Float = matureGyreLength() * Float(gyresPerLoop) / Float(nodesPerLoop)

/// The leaf. Petiole 65 mm (Flora of Pakistan: 4–9 cm); terminal leaflet
/// 95 × 64 mm and the lateral pair 85 × 56 mm (Flora of China: 4–16 × 2.5–11
/// cm; Pakistan: 4.5–15 × 2.5–6.5 cm). Petiolules drawn 3 mm (Pakistan
/// 1.5–2.5 mm; a hair longer so they show). The rachis beyond the lateral
/// pair, MODEL 18 mm.
let petioleLength: Float = 65
let terminalLeaflet = SIMD2<Float>(95, 64)
let lateralLeaflet = SIMD2<Float>(85, 56)
let petioluleLength: Float = 3
let rachisLength: Float = 18
let petioleRadius: Float = 1.1
let petioleRange: ClosedRange<Float> = 40...90
let leafletLengthRange: ClosedRange<Float> = 40...160
let leafletWidthRange: ClosedRange<Float> = 25...110

/// Hours for a leaf to open fully after its node leaves the tip. MODEL, and
/// compressed: a real bean leaf takes days. Gianoli 2015 records that leaf
/// expansion lags stem extension in twining leader shoots, which is the order
/// drawn; at this time-lapse's pace a leaf taking days would open only far
/// below the frame, so the lag is shortened to 12 h to show it happen.
let leafOpenHours: Float = 12
/// A new leaf's size as a fraction of full. MODEL.
let budFraction: Float = 0.07

// MARK: - the timeline

/// The loop: three turns of the pole. Real time ≈ 10.0 h.
let loopHours: Float = gyreHours * Float(gyresPerLoop)
/// How far the plant rises in one loop: three mature turns.
let loopRise: Float = maturePitch * Float(gyresPerLoop)

/// 10 s at 20 frames a second (see main.swift for why, measured).
let frameCount: Int = 200
let frameDelayCentiseconds: Int = 5
let loopSeconds: Float = Float(frameCount * frameDelayCentiseconds) / 100

/// The hour shown in frame f. Always forward: the frame after the last is the
/// first of the next loop, which is the same picture one loop higher.
func hourOf(frame f: Int, mutant: Mutant = .none) -> Float {
    let u: Float = Float(f) / Float(frameCount)
    if mutant == .rewind {
        // Grow for half the frames, then run the film backwards to the start.
        let back: Float = u < 0.5 ? 2 * u : 2 * (1 - u)
        return back * loopHours
    }
    return u * loopHours
}

// MARK: - the stem

/// Angle round the pole, in radians, of the winding front: the highest point
/// in contact. It advances one full turn per `gyreHours`.
func frontAngle(_ t: Float) -> Float { 2 * Float.pi * t / gyreHours }

/// Height of the winding front: rises one mature pitch per turn.
func frontHeight(_ t: Float) -> Float { maturePitch * t / gyreHours }

/// Rise from a point Δ radians behind the front up to the front: the close
/// spire opening to the mature pitch over `openingTurns`.
func riseBehindFront(_ delta: Float) -> Float {
    let pc: Float = maturePitch * closeSpireFraction
    let L: Float = 2 * Float.pi * openingTurns
    return (pc * delta + (maturePitch - pc) * smoothstepIntegral(delta, L)) / (2 * Float.pi)
}

/// Radius of the stem's axis from the pole's, Δ radians behind the front:
/// laid loose, tightening onto the wood.
func windingRadius(_ delta: Float, _ m: Mutant) -> Float {
    contactRadius(m) + looseGap * (1 - smoothstep(0, 2 * Float.pi * tighteningTurns, delta))
}

/// A point of the wound stem at angle φ (≤ the front's) at hour t.
/// Anticlockwise from above means φ carries +x towards −z.
func woundPoint(_ phi: Float, _ t: Float, _ m: Mutant) -> V3 {
    let delta: Float = max(frontAngle(t) - phi, 0)
    let r: Float = windingRadius(delta, m)
    let y: Float = frontHeight(t) - riseBehindFront(delta)
    let h: Float = handedness(m)
    return V3(r * cos(phi), y, -h * r * sin(phi))
}

/// Stem length per radian in the close spire at the front — how the stem not
/// yet wound is measured out along the free part.
func closeArcPerRadian(_ m: Mutant) -> Float {
    let r: Float = windingRadius(0, m)
    let rise: Float = maturePitch * closeSpireFraction / (2 * Float.pi)
    return (r * r + rise * rise).squareRoot()
}

/// Angle steps the wound stem is sampled at: on a fixed lattice, so a loop
/// later the samples fall on the same places.
let samplesPerTurn: Int = 36

/// The free shoot beyond the front, as points from the front outwards. It
/// leaves along the helix and curves round less and less, so it swings out
/// away from the pole; it leans up, then its tip hooks over (Darwin: "the end
/// of the shoot in many twining plants is completely hooked"). Everything is
/// relative to the front, so it sweeps round the pole with it: that is the
/// circumnutation, slowed by winding to one turn per `gyreHours`.
func freeShoot(_ t: Float, _ m: Mutant) -> [V3] {
    let phi: Float = frontAngle(t)
    let h: Float = handedness(m)
    let p0: V3 = woundPoint(phi, t, m)
    let r: Float = windingRadius(0, m)
    let rise: Float = maturePitch * closeSpireFraction / (2 * Float.pi)
    var d: V3 = simd_normalize(V3(-r * sin(phi), rise, -h * r * cos(phi)))
    let steps: Int = 44
    let ds: Float = freeLength / Float(steps)
    let beta0: Float = asin(d.y)
    var pts: [V3] = [p0]
    var p: V3 = p0
    for i in 0..<steps {
        let s: Float = (Float(i) + 0.5) * ds
        // Leaning: up from the helix's angle to 40°, then the hook.
        let lean: Float = beta0 + (0.70 - beta0) * smoothstep(0, 0.4 * freeLength, s)
        let hook: Float = -1.9 * smoothstep(freeLength - 30, freeLength, s)
        let beta: Float = lean + hook
        // Turning round the pole: at most the helix's own curvature, fading
        // out — so it only ever swings away from the wood. Curvature is of the
        // path seen from above, so it turns by κ times the step's run over the
        // ground, not its length.
        let kappa: Float = (1 / r) * (1 - smoothstep(0, 0.5 * freeLength, s))
        var flat = V3(d.x, 0, d.z)
        if simd_length(flat) < 1e-4 { flat = V3(1, 0, 0) }
        flat = simd_normalize(rotate(flat, about: V3(0, 1, 0), by: h * kappa * ds * cos(beta)))
        d = flat * cos(beta) + V3(0, 1, 0) * sin(beta)
        p += d * ds
        pts.append(p)
    }
    return pts
}

/// The stem's radius at arc length σ from the apex.
func stemRadiusFromApex(_ sigma: Float) -> Float {
    apexRadius + (stemRadius - apexRadius) * smoothstep(0, 0.85 * freeLength, sigma)
}

/// How far below the front the stem is drawn: well under the frame.
let woundTurnsDrawn: Float = 6.5

struct StemCurve {
    var points: [V3]
    var radii: [Float]
    /// Index of the winding front in `points`.
    var frontIndex: Int
}

func stem(_ t: Float, _ m: Mutant) -> StemCurve {
    let phiF: Float = frontAngle(t)
    let dphi: Float = 2 * Float.pi / Float(samplesPerTurn)
    // The lowest sample: floor with an offset, so no frame's front falls
    // exactly on the rounding edge (a loop later it must round the same way).
    let first: Int = Int(floor((phiF - 2 * Float.pi * woundTurnsDrawn) / dphi + 0.37))
    // Capsules between samples cut the corner of the true helix by its
    // sagitta, r(1 − cos(Δφ/2)) — 0.055 mm here, enough to sink into the pole.
    // So the samples stand out by that factor and the capsules' flats touch.
    let circumscribe: Float = 1 / cos(dphi / 2)
    var pts: [V3] = []
    var i: Int = first
    while Float(i) * dphi < phiF - dphi * 0.05 {
        let p: V3 = woundPoint(Float(i) * dphi, t, m)
        pts.append(V3(p.x * circumscribe, p.y, p.z * circumscribe))
        i += 1
    }
    let front: Int = pts.count
    let free: [V3] = freeShoot(t, m)
    pts += free
    var radii: [Float] = Array(repeating: stemRadius, count: front)
    let ds: Float = freeLength / Float(free.count - 1)
    for k in 0..<free.count { radii.append(stemRadiusFromApex(freeLength - Float(k) * ds)) }
    return StemCurve(points: pts, radii: radii, frontIndex: front)
}

// MARK: - nodes and leaves

/// Node k sits where the stem will be at angle k·(3/4 turn) once wound. The
/// stem not yet wound is measured out along the free shoot at the close
/// spire's length per radian, so a node slides along the free shoot and onto
/// the pole continuously.
let nodeAngleStep: Float = 2 * Float.pi * Float(gyresPerLoop) / Float(nodesPerLoop)

/// The apex, as the angle it will have once wound.
func apexAngle(_ t: Float, _ m: Mutant) -> Float { frontAngle(t) + freeLength / closeArcPerRadian(m) }

struct Node {
    var index: Int
    var position: V3
    /// Hours since this node left the apex.
    var age: Float
    /// Outward from the pole, horizontal.
    var outward: V3
    /// Which side of the stem its leaf leans to: alternate leaves.
    var side: Float
    var wound: Bool
}

func nodes(_ t: Float, _ m: Mutant) -> [Node] {
    let phiF: Float = frontAngle(t)
    let phiA: Float = apexAngle(t, m)
    let free: [V3] = freeShoot(t, m)
    let lowest: Int = Int(floor((phiF - 2 * Float.pi * (woundTurnsDrawn - 0.5)) / nodeAngleStep + 0.37))
    let highest: Int = Int(floor(phiA / nodeAngleStep))
    var out: [Node] = []
    if highest < lowest { return out }
    for k in lowest...highest {
        let phi: Float = Float(k) * nodeAngleStep
        let age: Float = (phiA - phi) / (2 * Float.pi) * gyreHours
        var p: V3
        var wound: Bool = true
        if phi <= phiF {
            p = woundPoint(phi, t, m)
        } else {
            let s: Float = (phi - phiF) * closeArcPerRadian(m)
            p = pointAlong(free, s).p
            wound = false
        }
        var out2 = V3(p.x, 0, p.z)
        if simd_length(out2) < 1e-3 { out2 = V3(1, 0, 0) }
        out.append(Node(index: k, position: p, age: age, outward: simd_normalize(out2),
                        side: (k & 1) == 0 ? 1 : -1, wound: wound))
    }
    return out
}

/// Node k at hour t, if it exists yet (step 54: the ladybirds and flowers look
/// nodes up by index). The same arithmetic as `nodes`.
func nodeAt(_ k: Int, _ t: Float, _ m: Mutant) -> Node? {
    let phiF: Float = frontAngle(t)
    let phiA: Float = apexAngle(t, m)
    let phi: Float = Float(k) * nodeAngleStep
    if phi > phiA { return nil }
    let age: Float = (phiA - phi) / (2 * Float.pi) * gyreHours
    var p: V3
    var wound: Bool = true
    if phi <= phiF {
        p = woundPoint(phi, t, m)
    } else {
        let free: [V3] = freeShoot(t, m)
        p = pointAlong(free, (phi - phiF) * closeArcPerRadian(m)).p
        wound = false
    }
    var out2 = V3(p.x, 0, p.z)
    if simd_length(out2) < 1e-3 { out2 = V3(1, 0, 0) }
    return Node(index: k, position: p, age: age, outward: simd_normalize(out2), side: (k & 1) == 0 ? 1 : -1,
                wound: wound)
}

/// How grown a leaf is, 0…1, and how open.
func leafGrowth(_ age: Float) -> Float { budFraction + (1 - budFraction) * smoothstep(0, leafOpenHours, age) }
func leafOpening(_ age: Float) -> Float { smoothstep(0.1 * leafOpenHours, leafOpenHours, age) }

/// A leaf's parts, as they are drawn (step 54 needs them by name: the
/// ladybirds walk on them). Blades in the order terminal, lateral turned away
/// from the camera side (+spread), lateral on the other side (−spread); the
/// petiolules likewise.
struct LeafGeometry {
    var petiole: Chain
    var rachis: Chain
    var petiolules: [Chain]
    var blades: [Leaflet]
}

/// A trifoliate leaf at a node: petiole, rachis, three leaflets on petiolules.
/// Young, it points up along the shoot with its leaflets folded shut; open, it
/// stands out from the pole with the blades spread.
func leaf(at n: Node, into s: inout PlantScene) {
    let g: LeafGeometry = leafGeometry(at: n)
    s.chains.append(g.petiole)
    s.chains.append(g.rachis)
    for (stalk, blade) in zip(g.petiolules, g.blades) {
        s.chains.append(stalk)
        s.leaflets.append(blade)
    }
}

/// Step 45's leaf, unchanged, returned as its parts.
func leafGeometry(at n: Node) -> LeafGeometry {
    var s = PlantScene()
    let g: Float = leafGrowth(n.age)
    let o: Float = leafOpening(n.age)
    let up = V3(0, 1, 0)
    // A little variety between leaves, MODEL. It depends on the node's place
    // in the loop's four, so a loop later the same leaf looks the same.
    let variant: Int = ((n.index % nodesPerLoop) + nodesPerLoop) % nodesPerLoop
    let jitter: [Float] = [0.0, 0.6, -0.4, 0.25]
    let j: Float = jitter[variant]
    // Out from the pole and a little to the leaf's side.
    let flatDir: V3 = simd_normalize(rotate(n.outward, about: up, by: n.side * 0.45 + 0.2 * j))
    let elev: Float = 1.25 + (0.55 + 0.12 * j - 1.25) * o
    let dir: V3 = flatDir * cos(elev) + up * sin(elev)
    let lp: Float = petioleLength * g
    let a: V3 = n.position + n.outward * (stemRadius * 0.6)
    let tip: V3 = a + dir * lp - up * (0.18 * lp * o)
    let ctrl: V3 = a + dir * (lp * 0.55) + up * (0.06 * lp)
    let pet: [V3] = bezier(a, ctrl, tip, segments: 10)
    let pr: Float = petioleRadius * max(g, 0.45)
    var pradii: [Float] = []
    for i in 0...10 {
        let f: Float = Float(i) / 10
        pradii.append(pr * (1.25 - 0.35 * f))
    }
    s.chains.append(Chain(points: pet, radii: pradii, material: .petiole))

    // The rachis carries on past the lateral pair to the terminal leaflet.
    let tdir: V3 = simd_normalize(tip - ctrl)
    let side: V3 = simd_normalize(simd_cross(up, flatDir))
    let rachisEnd: V3 = tip + tdir * (rachisLength * g)
    s.chains.append(Chain(points: [tip, rachisEnd], radii: [pr * 0.9, pr * 0.8], material: .petiole))

    // Blade directions: young, pointing up with the shoot; open, spread and
    // sloping a little down, faces up.
    func blade(_ base: V3, spread: Float, size: SIMD2<Float>, droop: Float) {
        let pitchUp: Float = 1.2 + (-0.35 - 0.15 * j - 1.2) * o
        let flat: V3 = simd_normalize(flatDir * cos(spread) + side * sin(spread))
        let u: V3 = simd_normalize(flat * cos(pitchUp) + up * sin(pitchUp))
        let w0: V3 = up - u * simd_dot(up, u)
        let w1: V3 = simd_length(w0) > 1e-3 ? simd_normalize(w0) : simd_normalize(simd_cross(u, side))
        // Each blade rolled a little about its midrib, the laterals outward.
        let w: V3 = simd_normalize(rotate(w1, about: u, by: (0.25 * spread + 0.15 * j) * o))
        let v: V3 = simd_cross(w, u)
        let stalk: V3 = base + u * (petioluleLength * g)
        s.chains.append(Chain(points: [base, stalk], radii: [pr * 0.8, pr * 0.7], material: .petiole))
        s.leaflets.append(Leaflet(origin: stalk, u: u, v: v, w: w, length: size.x * g, halfWidth: size.y / 2 * g,
                                  fold: 1.4 + (0.22 - 1.4) * o, droop: droop * o, shape: .beanOvate,
                                  material: .leaf))
    }
    // The terminal leaflet hangs a little more than the laterals. MODEL.
    blade(rachisEnd, spread: 0, size: terminalLeaflet, droop: 0.22)
    let lateralSpread: Float = 0.25 + (1.25 - 0.25) * o
    blade(tip, spread: lateralSpread, size: lateralLeaflet, droop: 0.14)
    blade(tip, spread: -lateralSpread, size: lateralLeaflet, droop: 0.14)
    // s.chains: petiole, rachis, then a petiolule per blade.
    return LeafGeometry(petiole: s.chains[0], rachis: s.chains[1], petiolules: Array(s.chains[2...]),
                        blades: s.leaflets)
}

// MARK: - the scene

/// How much of the pole is drawn, above and below the front.
let poleAbove: Float = 400
let poleBelow: Float = 900

func beanScene(_ t: Float, _ m: Mutant) -> PlantScene {
    var s = PlantScene()
    let yf: Float = frontHeight(t)
    // The pole: one long capsule. Its ends are off frame; they move with the
    // loop, and its texture repeats with it.
    s.chains.append(Chain(points: [V3(0, yf - poleBelow, 0), V3(0, yf + poleAbove, 0)],
                          radii: [poleRadius, poleRadius], material: .pole))
    let st: StemCurve = stem(t, m)
    s.chains.append(Chain(points: st.points, radii: st.radii, material: .stem))
    for n in nodes(t, m) {
        leaf(at: n, into: &s)
        if let f = inflorescence(n, t, m) { addInflorescence(f, into: &s) }
    }
    return s
}

/// Step 54's whole scene: the flowering bean and the ladybirds on it.
func fullScene(_ t: Float, _ m: Mutant) -> PlantScene {
    var s: PlantScene = beanScene(t, m)
    addLadybirds(t, m, into: &s)
    return s
}

// MARK: - the camera

/// The camera rises with the winding front at the same steady rate, so the
/// picture a loop later is the same picture one loop higher. MODEL: 1.25 m
/// away, level, a little to the left of square so the near face of the helix
/// runs across the frame; 440 mm of height in frame.
let cameraDistance: Float = 1250
let cameraAzimuth: Float = -0.30
let frameHalfHeight: Float = 220
let cameraLead: Float = -120
/// Looking down a little (about 14°), so open leaves show their faces and
/// the helix reads as going round, not just across.
let cameraRise: Float = 0.25

func camera(_ t: Float) -> Camera {
    let target = V3(0, frontHeight(t) + cameraLead, 0)
    let pos: V3 = target + simd_normalize(V3(sin(cameraAzimuth), cameraRise, cos(cameraAzimuth))) * cameraDistance
    return Camera(position: pos, target: target, tanHalf: frameHalfHeight / cameraDistance)
}

/// Step 45's colours, and step 54's: white petals (MODEL albedo, as step 48's
/// white), a clear young-pod green, and the ladybird's orange-red, black and
/// white (MODEL albedos; ADW: "orangish-red elytra and black spots"). The
/// purple mutant gives the petals a violet.
let whitePetal = V3(0.64, 0.645, 0.62)
let violetPetal = V3(0.30, 0.07, 0.34)

func lookFor(_ m: Mutant) -> Look {
    Look(leafTop: V3(0.045, 0.15, 0.028), leafUnder: V3(0.13, 0.25, 0.07),
         stem: V3(0.16, 0.30, 0.07), petiole: V3(0.17, 0.32, 0.08), tendril: V3(0.3, 0.45, 0.15),
         support: V3(0.40, 0.31, 0.22), supportKind: 0, period: loopRise, supportRadius: poleRadius,
         leafGloss: 0.2, keyDirection: V3(-0.45, 0.7, 0.55),
         petal: m == .purple ? violetPetal : whitePetal, petalWithered: V3(0.42, 0.37, 0.25),
         pod: V3(0.09, 0.22, 0.05),
         elytraRed: V3(0.55, 0.035, 0.012), beetleBlack: V3(0.012, 0.011, 0.010), beetleWhite: V3(0.70, 0.69, 0.64))
}

let ground = Ground(top: V3(0.86, 0.885, 0.87), bottom: V3(0.78, 0.81, 0.78))

/// Real time per GIF second, for the caption. Step 54: it is the climb's
/// time-lapse; the pods are a gradient, and the ladybirds run on their own
/// clock (Ladybug.swift).
func timeCaption() -> String {
    String(format: "time-lapse of the climb: about %.0f hours in %.0f s", loopHours, loopSeconds)
}

// MARK: - flowers and pods (step 54)
//
// Sources, checked for this step:
//
//   Flora of China 10, Phaseolus vulgaris (as quoted on World Flora Online,
//     wfo-0000183282): "Racemes shorter than leaves, usually several flowered
//     at top of rachis. Bracteoles ovate, usually as long as calyx or slightly
//     longer, persistent. Calyx cup-shaped, 3-4 mm ... Corolla white, yellow,
//     violet, or red; standard 9-12 mm wide; wings obovate; keel ca. 1 cm,
//     apex spirally twisted ... Legumes linear-oblong, 10-15 × 1-1.5 cm,
//     slightly curved and turgid, glabrous, beaked. Seeds 4-10".
//   Flora of Tropical East Africa (same WFO page): "Inflorescences 1–3-
//     flowered or flowers solitary, shorter than the leaves; peduncles 0–5 cm
//     long; pedicels 3–10 mm long ... Standard 1(–1.9) × 1.1 cm, oblate-oblong
//     ... keel c. 22 mm long, spirally incurved"; "Corolla white, yellowish,
//     purple or dark or pale pink"; "Pods 11–12.5(–20) × 1–1.3 cm, linear-
//     lanceolate ... beaked".
//   McGregor, Insect Pollination of Cultivated Crop Plants (USDA Agriculture
//     Handbook 496, 1976; the ARS online edition), Bean, Phaseolus vulgaris:
//     "The bean flower is of the typical legume shape; usually whitish ... and
//     is 1/2 to 3/4 inch long. The keel is prolonged in a spirally twisted
//     beak"; "The blooms are loosely scattered along a 2- to 3-inch rachis";
//     "The corolla sheds after a few days"; "The anthers dehisce the evening
//     of the day before the flower opens (Jones and Rosa 1928)"; "Rutger and
//     Beckham (1970) stated that P. vulgaris is primarily a selfing species";
//     "Kristofferson (1921) reported 0 to 1.4 percent crossing, and Emerson
//     (1904), 0 to 10 percent crossing"; the pod "straight or slightly curved,
//     4 to 8 inches long, with a prominent beak".
//   Gómez-Martín et al., Plants 9:545 (2020), "Transcriptional Dynamics and
//     Candidate Genes Involved in Pod Maturation of Common Bean": "developing
//     pods grew exponentially until five DAA [days after anthesis] ... and the
//     first pod reached a length higher than 2.5 cm"; "Pods elongated rapidly
//     until 15 DAA"; then "rapid seed growth and cessation of pod elongation".
//     So a pod a week or so after its flower opened is already several
//     centimetres long, but its seeds have not begun their rapid growth: a
//     small pod is smooth, without bulges. That is the pod drawn here.
//   Step 25 (this repo) drew the white common-bean flower in detail, with its
//     keel coiled 1.75 turns in a nearly flat spiral (MODEL within FTEA's
//     "1–5 turns"). The same coil is used here, smaller.
//
// White: Russell chose white for the common bean in step 25; FoC, FTEA and
// McGregor all list white ("usually whitish"). The stages are a GRADIENT
// ALONG THE STEM, as in step 48: each node's stage is set by its age, so
// flowers stand above and pods hang below, and the stages slide down with the
// nodes as the stem grows. A real pod takes a week or two to grow; the loop is
// ten hours. It is compressed, and the frame says so.

/// FTEA: "1–3-flowered"; FoC: "usually several flowered at top of rachis".
/// Three to a raceme, the most FTEA allows.
let flowersPerRaceme: Int = 3
let flowersPerRacemeRange: ClosedRange<Int> = 1...3
/// Racemes stand in every other leaf axil, the odd-numbered nodes (MODEL: not
/// every axil of a bean flowers at once; alternate nodes keep the flowers,
/// the leaves and the ladybirds' leaf apart, and the even nodes are the ones
/// the ladybirds walk out on).
func hasRaceme(_ k: Int) -> Bool { (k & 1) != 0 }
/// FTEA: peduncle 0–5 cm; drawn 25 mm. McGregor: flowers "loosely scattered
/// along a 2- to 3-inch rachis"; drawn 52 mm, the flowers on its upper part
/// (FoC: "at top of rachis").
let peduncleLength: Float = 25
let peduncleRange: ClosedRange<Float> = 0...50
let racemeRachisLength: Float = 52
let racemeRachisRange: ClosedRange<Float> = 50.8...76.2
/// FTEA: pedicels 3–10 mm; drawn 7.
let pedicelLength: Float = 7
let pedicelRange: ClosedRange<Float> = 3...10
/// FoC: calyx cup-shaped, 3–4 mm; drawn 3.5 mm long, 2.2 mm across its mouth
/// (MODEL width).
let calyxLength: Float = 3.5
let calyxRange: ClosedRange<Float> = 3...4
/// The standard: FTEA 1(–1.9) × 1.1 cm; FoC 9–12 mm wide. Drawn 13 × 11 mm.
/// McGregor: the flower 1/2–3/4 inch (12.7–19 mm) long — the flower here is
/// about 15 mm from calyx to the tip of the standard.
let standardSize = SIMD2<Float>(13, 11)
let standardLengthRange: ClosedRange<Float> = 10...19
let standardWidthRange: ClosedRange<Float> = 9...12
let flowerLengthRange: ClosedRange<Float> = 12.7...19.1
/// Wings: "obovate" (FoC); size MODEL, 12 × 8 mm.
let wingSize = SIMD2<Float>(12, 8)
/// The keel: a boat 6 mm long (MODEL), prolonged in a beak coiled 1.75 turns
/// in a nearly flat spiral (step 25's MODEL, inside FTEA's 1–5 turns), from
/// 2.6 mm out to 0.8 mm in; the beak a tube 0.45 mm in radius, so the turns
/// (1.03 mm apart) do not touch.
let keelBoatLength: Float = 6
let keelTurns: Float = 1.75
let keelTurnsRange: ClosedRange<Float> = 1...5
let keelCoilOuter: Float = 2.6
let keelCoilInner: Float = 0.8
let keelBeakRadius: Float = 0.45
/// The small pod. Full-grown the floras give 10–15 × 1–1.5 cm (FoC) and 11–
/// 12.5(–20) × 1–1.3 cm (FTEA). A pod passes 2.5 cm by five days after the
/// flower opens and lengthens fast to fifteen (Gómez-Martín et al. 2020), so
/// a pod about a week old is a third to a half of full length. MODEL: 46 mm
/// long, 5 mm broad — slender, the seeds not yet swelling, so smooth. Its tip
/// is drawn out into a beak (FoC "beaked"; McGregor "a prominent beak").
let podLength: Float = 46
let podWidth: Float = 5
let podLengthRange: ClosedRange<Float> = 33...75
let podWidthRange: ClosedRange<Float> = 3...7
let fullPodLengthRange: ClosedRange<Float> = 100...150
/// Every flower sets a pod (MODEL).

/// The stages by node age, hours since the node left the tip. All MODEL, and
/// compressed: a bean flower opens once and "the corolla sheds after a few
/// days" (McGregor), and its pod takes a week to reach this size. Laid out
/// here so that the nodes in view carry buds, open flowers, withering ones
/// round young pods, and small pods, top to bottom.
let budAppears: Float = 2.0
let flowerOpens: Float = 3.0
let flowerOpen: Float = 4.2
let witherStarts: Float = 6.0
let witherEnds: Float = 7.6
let petalsFall: Float = 8.6
let podStarts: Float = 6.4
let podFull: Float = 10.5

/// The age a node's flowers and pods go by; the pods_above_flowers mutant
/// turns the gradient upside down.
func reproductiveAge(_ age: Float, _ m: Mutant) -> Float {
    m == .podsAboveFlowers ? max(0, 20 - age) : age
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
    /// Standard, wing, wing — empty once the corolla has fallen.
    var petals: [Leaflet]
    /// The keel's boat and coiled beak, as one chain; nil once fallen.
    var keel: Chain?
    var calyx: Chain
    var pedicel: Chain
    var pod: Chain?
}

struct Inflorescence {
    var node: Node
    var stage: FlowerStage
    /// Peduncle and rachis, one chain.
    var stalk: Chain
    var flowers: [Flower]
}

func inflorescence(_ n: Node, _ t: Float, _ m: Mutant) -> Inflorescence? {
    guard hasRaceme(n.index) else { return nil }
    let a: Float = reproductiveAge(n.age, m)
    let st: FlowerStage = flowerStage(a)
    if st == .none { return nil }
    let up = V3(0, 1, 0)
    // The raceme leaves the axil above the petiole and stands out from the
    // pole steeply, turned past the edge of the nearer lateral leaflet so
    // that its pods hang clear of the leaf (MODEL path; a test probes it).
    // It grows with its leaf, from a third of full length.
    let grow: Float = 0.35 + 0.65 * smoothstep(budAppears, flowerOpen, n.age)
    let flat: V3 = simd_normalize(rotate(n.outward, about: up, by: -n.side * 0.95))
    let elev: Float = 1.2
    let dir: V3 = simd_normalize(flat * cos(elev) + up * sin(elev))
    let start: V3 = n.position + n.outward * (stemRadius * 0.5) + up * 2.5
    let total: Float = (peduncleLength + racemeRachisLength) * grow
    // A gentle curve: out along `dir`, bending over a little at the top.
    let end: V3 = start + dir * total - up * (0.12 * total) + flat * (0.1 * total)
    let ctrl: V3 = start + dir * (0.6 * total)
    let pts: [V3] = bezier(start, ctrl, end, segments: 12)
    let stalk = Chain(points: pts, radii: pts.indices.map { i in 0.85 - 0.3 * Float(i) / 12 }, material: .petiole)

    let open: Float = smoothstep(flowerOpens, flowerOpen, a)
    let wither: Float = smoothstep(witherStarts, witherEnds, a)
    let bloom: Float = 0.45 + 0.55 * smoothstep(budAppears, flowerOpen, a)
    let side: V3 = simd_normalize(simd_cross(dir, up))
    var flowers: [Flower] = []
    for j in 0..<flowersPerRaceme {
        let fj: Float = Float(j)
        // Along the upper rachis: at 45 %, 72 % and the tip (FoC "at top of
        // rachis"), on alternate sides (MODEL).
        let at: Float = [0.45, 0.72, 1.0][j]
        let (onStalk, tangent) = pointAlong(pts, polylineLength(pts) * at)
        let sgn: Float = (j & 1) == 0 ? 1 : -1
        // Pedicels hang out and down from the rachis; the flower faces out
        // from the plant, level-ish (MODEL).
        let pdir: V3 = simd_normalize(side * (0.8 * sgn) - up * 0.45 + flat * 0.35)
        let pl: Float = pedicelLength * max(grow, 0.6)
        let base: V3 = onStalk + pdir * pl
        let pedicel = Chain(points: [onStalk, base], radii: [0.45, 0.4], material: .petiole)
        let forward: V3 = simd_normalize(flat * 0.8 + side * (0.45 * sgn) - up * (0.1 + 0.1 * fj) + tangent * 0.1)
        let fu: V3 = simd_normalize(up - forward * simd_dot(up, forward))
        let across: V3 = simd_cross(forward, fu)
        let calyxTop: V3 = base + forward * (calyxLength * bloom)
        let calyx = Chain(points: [base, base + forward * (0.5 * calyxLength * bloom), calyxTop],
                          radii: [0.6, 1.0 * bloom, 1.2 * bloom], material: .petiole)
        var petals: [Leaflet] = []
        var keel: Chain? = nil
        if a < petalsFall {
            let shrink: Float = bloom * (1 - 0.4 * wither)
            let crumple: Float = 0.9 * wither
            let claw: V3 = calyxTop - forward * 1.0
            // Standard: in the bud folded forward over the rest; opening, it
            // lifts erect and reflexes back; withering, it folds again.
            let lift: Float = open * (1 - 0.6 * wither)
            let su: V3 = simd_normalize(forward * (0.95 - 1.15 * lift) + fu * (0.35 + 0.65 * lift))
            let sw0: V3 = forward * (1 - lift) - fu * (0.3 * (1 - lift)) + forward * lift
            let sw: V3 = simd_normalize(sw0 - su * simd_dot(sw0, su))
            petals.append(Leaflet(origin: claw + fu * 0.9, u: su, v: simd_cross(sw, su), w: sw,
                                  length: standardSize.x * shrink, halfWidth: standardSize.y / 2 * shrink,
                                  fold: 1.3 - 0.95 * open + crumple, droop: -0.08 * open, shape: .standard,
                                  material: .petal, tint: wither))
            // Wings, one each side, spreading forward when open.
            for sd in [Float(1), -1] {
                let wu: V3 = simd_normalize(forward + fu * 0.1 + across * (sd * 0.3 * open))
                let ww0: V3 = across * sd
                let ww: V3 = simd_normalize(ww0 - wu * simd_dot(ww0, wu))
                petals.append(Leaflet(origin: claw + across * (sd * 1.0) + fu * 0.2, u: wu, v: simd_cross(ww, wu),
                                      w: ww, length: wingSize.x * shrink, halfWidth: wingSize.y / 2 * shrink,
                                      fold: 0.3 + crumple, droop: 0.08, shape: .wing, material: .petal, tint: wither))
            }
            // The keel: a boat, then its beak coiled in a flat spiral in the
            // flower's plane of symmetry, curling up towards the standard.
            let k0: V3 = claw - fu * 0.5
            let k1: V3 = k0 + forward * (keelBoatLength * shrink)
            let ro: Float = keelCoilOuter * shrink
            let ri: Float = keelCoilInner * shrink
            let centre: V3 = k1 + fu * ro
            var kp: [V3] = [k0, k0 + forward * (0.5 * keelBoatLength * shrink)]
            var kr: [Float] = [1.3 * shrink, 1.1 * shrink]
            let steps: Int = 28
            for i in 0...steps {
                let th: Float = 2 * Float.pi * keelTurns * Float(i) / Float(steps)
                let r: Float = ro - (ro - ri) * Float(i) / Float(steps)
                kp.append(centre + (fu * (-cos(th)) + forward * sin(th)) * r)
                kr.append(max(keelBeakRadius * shrink * (1 - 0.3 * Float(i) / Float(steps)), 0.2)
                          + 0.5 * shrink * (1 - smoothstep(0, 4, Float(i))))
            }
            keel = Chain(points: kp, radii: kr, material: .petal)
        }
        // The pod: out of the flower as the petals wither, lengthening and
        // hanging down as it grows; smooth; its tip drawn out into a beak.
        var pod: Chain? = nil
        if a > podStarts {
            let g: Float = 0.1 + 0.9 * smoothstep(podStarts, podFull, a)
            let len: Float = podLength * g
            let hang: Float = smoothstep(podStarts, podStarts + 2.5, a)
            let down: V3 = simd_normalize(V3(0, -1, 0) + flat * 0.1 + side * (0.12 * sgn))
            let p0: V3 = calyxTop - forward * 0.5
            let p1: V3 = p0 + forward * (0.25 * len)
            let tipDir: V3 = simd_normalize(forward * (1 - 0.9 * hang) + down * (0.3 + 1.2 * hang))
            let p2: V3 = p0 + tipDir * len
            let segs: Int = 16
            var pp: [V3] = []
            var pr: [Float] = []
            for i in 0...segs {
                let s: Float = Float(i) / Float(segs)
                let q: Float = 1 - s
                let pa: V3 = p0 * (q * q)
                let pb: V3 = p1 * (2 * q * s)
                let pc: V3 = p2 * (s * s)
                pp.append(pa + pb + pc)
                pr.append(podRadius(s) * (0.4 + 0.6 * g))
            }
            pod = Chain(points: pp, radii: pr, material: .pod)
        }
        flowers.append(Flower(base: base, forward: forward, up: fu, across: across, petals: petals, keel: keel,
                              calyx: calyx, pedicel: pedicel, pod: pod))
    }
    return Inflorescence(node: n, stage: st, stalk: stalk, flowers: flowers)
}

/// The young pod's radius at s (0…1 from its base): rounded at the base,
/// even along its length (no seed bulges yet), drawn out to a thin beak.
func podRadius(_ s: Float) -> Float {
    let base: Float = smoothstep(0, 0.08, s)
    let beak: Float = 1 - 0.85 * smoothstep(0.78, 1.0, s)
    return max(podWidth / 2 * (0.45 + 0.55 * base) * beak, 0.3)
}

func addInflorescence(_ f: Inflorescence, into s: inout PlantScene) {
    s.chains.append(f.stalk)
    for fl in f.flowers {
        s.chains.append(fl.pedicel)
        s.chains.append(fl.calyx)
        s.leaflets += fl.petals
        if let k = fl.keel { s.chains.append(k) }
        if let p = fl.pod { s.chains.append(p) }
    }
}

// MARK: - what the frame says (step 54)

let nameCaption: String = "Phaseolus vulgaris, pole bean, in flower"
let ladybirdNameCaption: String = "with seven-spot ladybirds, Coccinella septempunctata"
let selfCaption: String = "mostly self-pollinated: 0–10% of seeds from crossing"
let aphidCaption: String = "ladybirds hunt aphids such as the black bean aphid (none drawn)"
let gradientCaption: String = "flowers above, pods below: older nodes are further along"
let compressedCaption: String = "(a real pod takes a week or two to grow; the stages here are compressed)"
