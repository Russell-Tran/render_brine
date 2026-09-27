// One spinal (alpha) motor neuron as numbers: a cell body with its nucleus, a
// dozen tapering dendrite trees, and one axon of constant width that ends in
// a few boutons. Nothing here touches the GPU; it is the part a test can read
// against the papers.
//
// Micrometres throughout. The cell lies roughly in the z = 0 plane, the camera
// looks down −z, +x is to the right and +y is up.
//
// Most numbers are from the cat, whose lumbar motoneurons are the best
// measured anywhere: they were filled with dye one at a time through a
// microelectrode and reconstructed from serial sections.

import Foundation
import simd

// MARK: - the cell body

/// Soma diameter. Cat hindlimb alpha-motoneurons have cross-sectional areas of
/// 816–3732 µm², "corresponding to diameters of about 32–69 microns"
/// (Zwaagstra & Kernell, *Brain Res* 1981, "Sizes of soma and stem dendrites
/// in intracellularly labelled alpha-motoneurones of the cat"). Mid-range.
let somaDiameter: Float = 50

/// Nucleus and nucleolus, from their measured VOLUMES: 1,660 µm³ and 25.6 µm³
/// (Edström, *J Comp Neurol* 1957, via BioNumbers 112112). CAVEAT: that is a
/// guinea pig neuron with a smaller soma (12,260 µm³, ~29 µm across), not a
/// cat alpha-motoneuron; its cell type was not checked beyond "motor
/// activity" in the title. Used as absolute sizes, since a nucleus grows much
/// less than its cell does.
let nucleusVolume: Double = 1660
let nucleolusVolume: Double = 25.6
func sphereDiameter(volume: Double) -> Float { Float(cbrt(6 * volume / Double.pi)) }
let nucleusDiameter: Float = sphereDiameter(volume: nucleusVolume)       // 14.7 µm
let nucleolusDiameter: Float = sphereDiameter(volume: nucleolusVolume)   // 3.7 µm

// MARK: - the dendrites

/// "On average each neurone had 12 (5–20) dendritic stems" (Zwaagstra &
/// Kernell 1981).
let primaryDendriteCount: Int = 12

/// Stem diameters where each dendrite leaves the soma. Zwaagstra & Kernell
/// measured 0.5–19 µm across all cells, with the mean per cell roughly
/// proportional to the soma's diameter. The individual values are MODEL,
/// spread inside that range for a mid-sized soma.
let stemDiameters: [Float] = [9.0, 6.5, 8.0, 5.5, 10.0, 7.0, 6.0, 8.5, 5.0, 9.5, 7.5, 6.5]

/// Tapering along one branch: "the mean overall degree of diameter decrease
/// per branch was about 12%" (Kernell & Zwaagstra, *J Physiol* 1989,
/// "Dendrites of cat's spinal motoneurones: relationship between stem
/// diameter and predicted input conductance").
let taperPerSegment: Float = 0.88

/// Tapering at a fork. Rall's rule would make the daughters' d^(3/2) add up to
/// the parent's; in the same paper the sum was "on average, 19% greater". Two
/// equal daughters each carrying half of 1.19 × parent^(3/2) are each
/// (1.19 / 2)^(2/3) = 0.707 of the parent's diameter — derived, not typed.
let rallExcess: Float = 1.19
let daughterRatio: Float = pow(rallExcess / 2, 2.0 / 3.0)

/// How many branch orders are drawn: the stem and two generations of forks,
/// 7 segments a tree. MODEL. A real tree forks further and reaches about a
/// millimetre from the soma; drawn in full at this scale it would fill a
/// screen twenty times this one and shrink the soma to a dot.
let branchOrders: Int = 3

/// Length of one segment by branch order. MODEL: chosen so the tree fits the
/// frame. DIAMETERS in this picture are to scale; dendrite LENGTHS are not —
/// they are shortened several-fold.
let segmentLengths: [Float] = [20, 27, 34]

/// Half-angle between two daughters at a fork, degrees. MODEL.
let forkHalfAngle: Float = 28

// MARK: - the axon

/// Axon diameter. Intracellular HRP fills of cat alpha-motoneurons: "a total
/// range being from 4.6 to 9.0 micrometer", pool means 5.2–7.4 µm (Cullheim &
/// Kellerth, *J Physiol* 1978, two papers on the axons of cat
/// alpha-motoneurones). That is the axon itself, measured in the cord, where
/// it runs at one width — no taper.
let axonDiameter: Float = 7.0

/// The axon hillock, a cone from the soma down to the axon's width. MODEL:
/// the base width and length are drawn to look like the classic figure; the
/// papers that measured it (Conradi 1969) were not checked.
let hillockBaseDiameter: Float = 15
let hillockLength: Float = 20

/// The terminal arbor: a few fine branches, each ending in a bouton. MODEL
/// throughout, not checked against a measurement: at a real neuromuscular
/// junction the axon sheds its myelin and splits into short branches with
/// varicosities a few micrometres wide.
let terminalCount: Int = 4
let terminalDiameter: Float = 2.0
let terminalLength: Float = 30
let boutonDiameter: Float = 4.5

/// Why the axon has a break in it. A lumbar motoneuron's axon runs from the
/// cord to the foot, on the order of a metre in a tall adult — MODEL, an order
/// of magnitude, not a measurement — which is about twenty thousand soma
/// diameters. The picture shows about six.
let axonRealLength: Float = 1_000_000
let axonToSomaRatio: Float = axonRealLength / somaDiameter

// MARK: - the cell, assembled

enum Kind: Int {
    case dendrite = 0
    case hillock = 1
    case axon = 2
    case terminal = 3
    case bouton = 4
}

/// One piece of process: a cone with rounded ends, from `a` (radius `ra`) to
/// `b` (radius `rb`). `parent` indexes into the same list.
struct Neurite {
    var a: SIMD3<Float>
    var b: SIMD3<Float>
    var ra: Float
    var rb: Float
    var kind: Kind
    var order: Int
    var parent: Int?
}

/// What to break, for `make mutants`. Each must make the suite fail.
enum Mutant: String {
    case none
    case twoAxons       // a second hillock and axon
    case taperingAxon   // the axon narrows like a dendrite
}

struct Neuron {
    var soma: SIMD3<Float>
    /// Tree after tree, each `segmentsPerTree` long, in the order they were built.
    var dendrites: [Neurite]
    var axon: [Neurite]
    /// Where the break is, and which way its two parallel cuts face.
    var breakCentre: SIMD3<Float>
    var breakNormal: SIMD3<Float>
}

let segmentsPerTree: Int = (1 << branchOrders) - 1
let breakHalfGap: Float = 5.0

/// A small fixed-seed generator, so the tree is the same on every run.
struct Seeded {
    var state: UInt64
    mutating func next() -> Float {
        state = state &* 6364136223846793005 &+ 1442695040888963407
        return Float(state >> 40) / Float(1 << 24)
    }
    mutating func between(_ lo: Float, _ hi: Float) -> Float { lo + (hi - lo) * next() }
}

func buildNeuron(_ mutant: Mutant = .none) -> Neuron {
    let soma = SIMD3<Float>(-140, 8, 0)
    let R: Float = somaDiameter / 2
    var rng = Seeded(state: 27)
    var dendrites: [Neurite] = []

    // Stems spread around the soma, leaving the right-hand side to the axon.
    for i in 0..<primaryDendriteCount {
        let spread: Float = 250 / Float(primaryDendriteCount - 1)
        let deg: Float = 55 + spread * Float(i) + rng.between(-5, 5)
        let angle: Float = deg * .pi / 180
        let tilt: Float = (i % 2 == 0 ? 1 : -1) * rng.between(0.15, 0.55)
        let dir: SIMD3<Float> = simd_normalize(SIMD3<Float>(cos(angle), sin(angle), tilt))
        let r0: Float = stemDiameters[i] / 2
        let first: Int = dendrites.count
        let start: SIMD3<Float> = soma + dir * (R * 0.8)
        let end: SIMD3<Float> = start + dir * segmentLengths[0]
        dendrites.append(Neurite(a: start, b: end, ra: r0, rb: r0 * taperPerSegment,
                                 kind: .dendrite, order: 1, parent: nil))
        // Breadth first, so the tree's segments sit together in the list.
        var k: Int = first
        while k < first + segmentsPerTree && dendrites[k].order < branchOrders {
            let p: Neurite = dendrites[k]
            let pdir: SIMD3<Float> = simd_normalize(p.b - p.a)
            // Forks open mostly in the picture's plane, with a little depth.
            let wobble = SIMD3<Float>(rng.between(-0.4, 0.4), rng.between(-0.4, 0.4), 1)
            // The fork turns about this axis: the viewing direction, tipped a
            // little, made perpendicular to the parent.
            let axis: SIMD3<Float> = simd_normalize(wobble - pdir * simd_dot(wobble, pdir))
            for side in [Float(-1), Float(1)] {
                let half: Float = (forkHalfAngle + rng.between(-8, 8)) * .pi / 180
                let q = simd_quatf(angle: side * half, axis: axis)
                let d: SIMD3<Float> = q.act(pdir)
                let len: Float = segmentLengths[p.order] * rng.between(0.8, 1.2)
                let ra: Float = p.rb * daughterRatio
                dendrites.append(Neurite(a: p.b, b: p.b + d * len, ra: ra, rb: ra * taperPerSegment,
                                         kind: .dendrite, order: p.order + 1, parent: k))
            }
            k += 1
        }
    }

    // The axon: hillock, one straight trunk, a fan of terminals with boutons.
    var axon: [Neurite] = []
    let ra: Float = axonDiameter / 2
    func addAxon(from dir: SIMD3<Float>, to end: SIMD3<Float>) {
        let base: SIMD3<Float> = soma + dir * (R * 0.85)
        let neck: SIMD3<Float> = soma + dir * (R + hillockLength)
        let h: Int = axon.count
        axon.append(Neurite(a: base, b: neck, ra: hillockBaseDiameter / 2, rb: ra,
                            kind: .hillock, order: 0, parent: nil))
        let rb: Float = mutant == .taperingAxon ? ra * taperPerSegment : ra
        axon.append(Neurite(a: neck, b: end, ra: ra, rb: rb, kind: .axon, order: 0, parent: h))
    }
    let axonDir: SIMD3<Float> = simd_normalize(SIMD3<Float>(1, -0.12, 0))
    let trunkEnd = SIMD3<Float>(180, -30, 0)
    addAxon(from: axonDir, to: trunkEnd)
    let trunk: Int = 1
    for i in 0..<terminalCount {
        let deg: Float = -42 + 84 * Float(i) / Float(terminalCount - 1) + rng.between(-6, 6)
        let angle: Float = deg * .pi / 180
        let d: SIMD3<Float> = simd_normalize(SIMD3<Float>(cos(angle), sin(angle), rng.between(-0.3, 0.3)))
        let tip: SIMD3<Float> = trunkEnd + d * terminalLength * rng.between(0.85, 1.15)
        let t: Int = axon.count
        axon.append(Neurite(a: trunkEnd, b: tip, ra: terminalDiameter / 2, rb: terminalDiameter / 2,
                            kind: .terminal, order: 0, parent: trunk))
        axon.append(Neurite(a: tip, b: tip + d * boutonDiameter * 0.55, ra: terminalDiameter / 2,
                            rb: boutonDiameter / 2, kind: .bouton, order: 0, parent: t))
    }
    if mutant == .twoAxons {
        addAxon(from: simd_normalize(SIMD3<Float>(0.2, -1, 0)), to: soma + SIMD3<Float>(40, -140, 0))
    }

    let breakCentre: SIMD3<Float> = soma + axonDir * 150
    let breakNormal: SIMD3<Float> = simd_normalize(SIMD3<Float>(1, -0.6, 0))
    return Neuron(soma: soma, dendrites: dendrites, axon: axon,
                  breakCentre: breakCentre, breakNormal: breakNormal)
}
