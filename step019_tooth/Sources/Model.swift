// Building the tooth as a lattice, loading it, breaking it, and reading the
// answer back out as a field the renderer can paint.
//
// Everything here is bookkeeping around `Lattice`, with one piece of real
// physics in it: `crack`, which grows a crack ONE BOND AT A TIME with a fresh
// solve between breaks. The first version of this step broke every over-limit
// bond in a single round and got 25,444 broken bonds in ten rounds, most of them
// in dentin. That is not a crack, it is an avalanche, and it happens because
// every bond in the round is judged against a stress field that the first break
// already destroyed. A crack sheds load onto its own tip; if you never re-solve,
// nothing ever sheds and the whole over-limit region goes at once.
//
// The other half of that fix is in Tooth.swift: the tooth is held in a ligament
// rather than clamped.

import Foundation
import simd

// MARK: - a small deterministic generator
//
// The scatter in critical strain has to be REPRODUCIBLE across runs and
// different between seeds, because the arrest test is a rate over seeds. Swift's
// own generator is neither, so this is splitmix64, which is four lines.

struct Splitmix {
    private var state: UInt64
    init(seed: UInt64) { state = seed &+ 0x9E3779B97F4A7C15 }

    mutating func next() -> UInt64 {
        state = state &+ 0x9E3779B97F4A7C15
        var z: UInt64 = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }

    /// Uniform in (0, 1).
    mutating func uniform() -> Double {
        let bits: UInt64 = next() >> 11
        let scale: Double = Double(1 << 53)
        return (Double(bits) + 0.5) / scale
    }

    /// Standard normal, Box–Muller. One of the pair is thrown away, which costs
    /// nothing here and keeps the state advance simple to reason about.
    mutating func normal() -> Double {
        let u1: Double = uniform()
        let u2: Double = uniform()
        let r: Double = (-2 * log(u1)).squareRoot()
        let theta: Double = 2 * Double.pi * u2
        return r * cos(theta)
    }
}

// MARK: - the domain

/// The rectangle the lattice covers, in millimetres. Wide enough for the widest
/// part of the crown and for the bone shell around the root, and no wider —
/// every extra column is nodes that solve to zero.
let domainHalfWidth: Float = 5.3
let domainBottom: Float = apexY - pdlThickness - boneShell - 0.15
let domainTop: Float = crownHeight + 0.15

/// A tooth solved at one mesh spacing, under one load case.
final class ToothModel {
    let spacingMM: Double
    let lattice: Lattice
    let cols: Int
    let rows: Int
    let loadCase: LoadCase
    private(set) var contactNodes: [Int] = []
    private(set) var residual: Float = 0
    private(set) var iterations: Int = 0

    /// Node index → its rest position in tooth millimetres. Kept rather than
    /// recomputed: everything downstream asks for it.
    private(set) var mm: [SIMD2<Float>] = []

    init(spacingMM: Double, loadCase: LoadCase, seed: UInt64 = 0,
         scatter: Double = 0.10, rigidRoot: Bool = false,
         pdlModulusOverride: Double? = nil, dentinStrengthOverride: Double? = nil,
         flatPunch: Bool = false) {
        self.flatPunch = flatPunch
        self.spacingMM = spacingMM
        self.loadCase = loadCase
        let a: Double = spacingMM * 1e-3
        let root3: Double = 3.0.squareRoot()
        let widthMM: Double = Double(domainHalfWidth) * 2
        let heightMM: Double = Double(domainTop - domainBottom)
        let c: Int = Int((widthMM / spacingMM).rounded(.up)) + 1
        let r: Int = Int((heightMM / (spacingMM * root3 / 2)).rounded(.up)) + 1
        self.cols = c
        self.rows = r

        var props: [ToothTissue: TissueProperties] = toothProperties
        if let e = pdlModulusOverride {
            props[.pdl] = TissueProperties(youngsModulus: e,
                                           tensileStrength: unbreakableStrength,
                                           name: "periodontal ligament")
        }
        if let s = dentinStrengthOverride {
            let old = props[.dentin]!
            props[.dentin] = TissueProperties(youngsModulus: old.youngsModulus,
                                              tensileStrength: s, name: old.name)
        }

        let ox: Float = -domainHalfWidth
        let oy: Float = domainBottom
        func toMM(_ p: SIMD2<Float>) -> SIMD2<Float> {
            SIMD2<Float>(p.x * 1000 + ox, p.y * 1000 + oy)
        }
        self.lattice = Lattice(cols: c, rows: r, spacing: a, thickness: sectionThickness,
                               properties: props,
                               tissueAt: { tissueAt(toMM($0)) },
                               kindAt: { p, t in kindAt(toMM(p), t, rigidRoot: rigidRoot) })
        self.mm = lattice.rest.map { toMM($0) }

        // Quenched disorder. A perfect lattice is a crystal and cracks along its
        // own symmetry axes; real enamel fails at a distribution of strains.
        // Lognormal so the factor is positive whatever the draw.
        if scatter > 0 {
            var rng = Splitmix(seed: seed &* 0x2545F4914F6CDD1D &+ 12345)
            let sigma: Double = scatter
            lattice.scaleLimits { _ in
                let g: Double = rng.normal()
                return Float(exp(sigma * g))
            }
        }
        applyLoad()
    }

    /// Spread the load over a contact patch on the occlusal surface.
    ///
    /// Two things here are fixture rather than physics, and both were wrong the
    /// first time:
    ///
    ///   * ONLY TRUE SURFACE NODES are loaded. Taking every node within 1.5
    ///     spacings of the surface makes the loaded layer 0.3 mm deep at one
    ///     mesh and 0.075 mm at another, so the load itself changes when the
    ///     mesh does, and a convergence study then measures the fixture. A node
    ///     counts as surface if it is missing a neighbour, which is a property
    ///     of the lattice and not of the spacing.
    ///   * THE PRESSURE IS HERTZIAN, p ∝ √(1 − (x/a)²), not uniform. A uniform
    ///     patch is a flat punch, and a flat punch has a stress singularity at
    ///     each edge (Johnson, *Contact Mechanics*, 1985, §2.8) — a property of
    ///     the punch, not of the tooth. Two curved cusps in contact give the
    ///     elliptical profile, which goes to zero at the edges, so there is
    ///     nothing there to concentrate.
    private(set) var contactWeights: [Float] = []
    private(set) var loadScale: Float = 1

    /// The `punch` mutation: a uniform patch instead of a Hertzian one. A flat
    /// punch has a stress singularity at each edge, so the crack starts under
    /// the loading fixture rather than where the tooth is weakest.
    private let flatPunch: Bool

    private func applyLoad() {
        var nodes: [Int] = []
        var weights: [Float] = []
        let a: Float = loadCase.contactHalfWidth
        for n in 0..<lattice.tissue.count {
            guard lattice.tissue[n] == .enamel else { continue }
            guard isSurfaceNode(n) else { continue }
            let q: SIMD2<Float> = mm[n]
            let offset: Float = q.x - loadCase.contactX
            guard abs(offset) < a else { continue }
            guard q.y > fissureFloorY - 0.5 else { continue }
            let u: Float = offset / a
            let w: Float = flatPunch ? 1 : max(1 - u * u, 0).squareRoot()
            guard w > 0 else { continue }
            nodes.append(n)
            weights.append(w)
        }
        contactNodes = nodes
        let total: Float = weights.reduce(0, +)
        guard total > 0 else { return }
        contactWeights = weights.map { $0 / total }
        setLoadScale(1)
        // The contact patch is a fixture, not a finding. Without this the first
        // thing that fails is the enamel under the load, which is a statement
        // about how the load was applied and nothing about teeth.
        lattice.reinforce(around: nodes)
    }

    /// Re-apply the same contact at a multiple of the case's force. Nothing else
    /// about the model changes, so the ramp below is a pure load ramp.
    func setLoadScale(_ scale: Float) {
        loadScale = scale
        let force: Float = Float(loadCase.newtons) * scale
        for (i, n) in contactNodes.enumerated() {
            lattice.setLoad(loadCase.direction * (force * contactWeights[i]), at: n)
        }
    }

    var currentNewtons: Float { Float(loadCase.newtons) * loadScale }

    /// A solid node with a missing neighbour — the lattice's own definition of a
    /// free surface, independent of the spacing.
    func isSurfaceNode(_ n: Int) -> Bool {
        guard lattice.tissue[n].isSolid else { return false }
        let j: Int = n / cols
        let i: Int = n % cols
        for k in lattice.neighbourSites(col: i, row: j) {
            if k < 0 { return true }
            if !lattice.tissue[k].isSolid { return true }
        }
        return false
    }

    @discardableResult
    func solve(tolerance: Float = 1e-7, maxIterations: Int = 40000,
               warmStart: Bool = false) -> (residual: Float, iterations: Int) {
        let out = lattice.solve(tolerance: tolerance, maxIterations: maxIterations,
                                warmStart: warmStart)
        residual = out.residual
        iterations = out.iterations
        return out
    }

    // MARK: - reading the answer

    /// Is this node on the tooth's OUTER surface — the enamel or cementum you
    /// could touch, or the root face the ligament is attached to?
    ///
    /// The pulp chamber's roof is a free surface too, and a crack cannot start
    /// there because nothing abrades or erodes it. Counting it made the "peak
    /// surface tension" land on the pulp horn under one load case, which is a
    /// true statement about the model and a useless one about teeth, so the
    /// cavity walls are excluded and asked about separately.
    func isToothSurface(_ n: Int) -> Bool {
        guard lattice.tissue[n].isTooth else { return false }
        let j: Int = n / cols
        let i: Int = n % cols
        for k in lattice.neighbourSites(col: i, row: j) {
            if k < 0 { return true }
            let t: ToothTissue = lattice.tissue[k]
            if t == .outside || t == .pdl { return true }
        }
        return false
    }

    /// The pulp chamber's own wall, asked about on its own.
    func isPulpWall(_ n: Int) -> Bool {
        guard lattice.tissue[n].isTooth else { return false }
        let j: Int = n / cols
        let i: Int = n % cols
        for k in lattice.neighbourSites(col: i, row: j) where k >= 0 {
            if lattice.tissue[k] == .pulp { return true }
        }
        return false
    }

    private var surfaceCache: [Int]?

    var surfaceNodes: [Int] {
        if let c = surfaceCache { return c }
        var out: [Int] = []
        for n in 0..<lattice.tissue.count where isToothSurface(n) { out.append(n) }
        surfaceCache = out
        return out
    }

    /// The largest tensile stress anywhere on the tooth's outer surface, and
    /// where. Read off the bonds rather than off the virial, because a node on a
    /// free surface owns half a cell and the bond stress needs no such
    /// correction at all.
    func peakSurfaceTension() -> (stressPa: Float, at: SIMD2<Float>, node: Int) {
        var best: Float = -.greatestFiniteMagnitude
        var where_ = SIMD2<Float>(0, 0)
        var node = -1
        for n in surfaceNodes {
            let s: Float = lattice.peakBondStress(n)
            if s > best { best = s; where_ = mm[n]; node = n }
        }
        return (best, where_, node)
    }

    /// The same peak, but averaged over nearby surface nodes OF THE SAME TISSUE
    /// first.
    ///
    /// A single node's bond stress on a curved free surface is partly a
    /// staircase artefact: the lattice approximates a slope with steps, and the
    /// node at the outside of a step carries more than its share. Averaging over
    /// a third of a millimetre — three lattice spacings at the render's mesh, six
    /// at half of it — is what makes the number comparable BETWEEN meshes.
    ///
    /// The same-tissue restriction is not decoration. At the neck the enamel is
    /// 60 µm thick, which is SMALLER THAN THE MESH, so a disc of radius 0.33 mm
    /// centred there is mostly dentin and the "smoothed peak" landed at the same
    /// place for every load case — the point that catches the most nodes, not
    /// the maximum. Restricted to one tissue it means what it says. The raw
    /// per-region peaks below are what the tests use, because those converge.
    func peakSurfaceTension(smoothedOver radiusMM: Float)
        -> (stressPa: Float, at: SIMD2<Float>) {
        let nodes: [Int] = surfaceNodes
        var raw: [Float] = []
        raw.reserveCapacity(nodes.count)
        for n in nodes { raw.append(lattice.peakBondStress(n)) }
        var best: Float = -.greatestFiniteMagnitude
        var where_ = SIMD2<Float>(0, 0)
        let r2: Float = radiusMM * radiusMM
        for n in nodes {
            let p: SIMD2<Float> = mm[n]
            var sum: Float = 0
            var count: Float = 0
            for (b, k) in nodes.enumerated() {
                let q: SIMD2<Float> = mm[k]
                guard lattice.tissue[k] == lattice.tissue[n] else { continue }
                let d: SIMD2<Float> = q - p
                guard simd_length_squared(d) <= r2 else { continue }
                sum += raw[b]
                count += 1
            }
            let mean: Float = count > 0 ? sum / count : 0
            if mean > best { best = mean; where_ = p }
        }
        return (best, where_)
    }

    /// The stress the ENAMEL SKIN would carry at a surface node, whatever
    /// material the mesh actually put there.
    ///
    /// At the neck the enamel is 60 µm thick and the render's mesh is 100 µm, so
    /// the enamel at the neck is BELOW MESH RESOLUTION: the surface node there
    /// is junction-band material at 38 GPa, and the lattice's 60 MPa is the
    /// junction's stress, not the enamel's. Strain compatibility fixes it
    /// without another solve — the skin sits on the same strain — so the enamel
    /// would carry that stress times 84/38. This is also the whole mechanism in
    /// one line: the thickness of the cervical enamel does not set the stress,
    /// strain compatibility does. The thickness only sets how little energy it
    /// takes to break through it.
    func enamelSkinStress(_ n: Int) -> Float {
        let here: Float = lattice.peakBondStress(n)
        let e: Double = toothProperties[lattice.tissue[n]]?.youngsModulus ?? enamelModulus
        let ratio: Float = Float(enamelModulus / e)
        return here * ratio
    }

    /// The largest skin-corrected tension in a band, and the load at which that
    /// would reach enamel's tensile strength if everything stayed linear.
    func enamelSkinPeak(between lo: Float, and hi: Float) -> (stressPa: Float, atNewtons: Float) {
        var best: Float = 0
        for n in surfaceNodes {
            let q: SIMD2<Float> = mm[n]
            guard q.y >= lo, q.y <= hi else { continue }
            // Only where enamel actually is. Correcting a cementum node by
            // 84/15 would be inventing an enamel skin on the root.
            let t: ToothTissue = lattice.tissue[n]
            guard t == .enamel || t == .dej else { continue }
            let s: Float = enamelSkinStress(n)
            if s > best { best = s }
        }
        guard best > 0 else { return (0, .infinity) }
        let implied: Float = currentNewtons * Float(enamelStrength) / best
        return (best, implied)
    }

    /// The same, restricted to a band of heights — used to ask "is the cervical
    /// region the maximum" without the answer depending on where the frame is
    /// cropped.
    func peakSurfaceTension(between lo: Float, and hi: Float) -> Float {
        var best: Float = 0
        for n in surfaceNodes {
            let q: SIMD2<Float> = mm[n]
            guard q.y >= lo, q.y <= hi else { continue }
            let s: Float = lattice.peakBondStress(n)
            if s > best { best = s }
        }
        return best
    }

    // MARK: - growing a crack

    struct CrackStep {
        var broken: [Int]            // bond indices broken at this step
        var worst: Float             // the worst overload found before breaking
        var totalBroken: Int
        var residual: Float
    }

    struct CrackResult {
        var steps: [CrackStep]
        var arrested: Bool           // it stopped on its own, elastically, nothing flying
        var fragmentDetached: Bool   // a piece came off, which is NOT the same thing
        var breakdown: Bool          // the solver met a zero-energy mode
        var brokenByTissue: [ToothTissue: Int]
        var dentinCore: Int          // broken bonds with dentin at BOTH ends
        var deepestPastDEJ: Float    // mm past the junction, into dentin
        var totalBroken: Int
        var maxDisplacement: Float
    }

    /// Grow the crack. One re-solve per step; `window` lets a tip that is
    /// genuinely two bonds wide open both.
    ///
    /// The runaway check is the part that took two goes to get right, and it is
    /// the difference between this step reporting a result and reporting a bug.
    /// A crack can leave a fragment attached through a SINGLE NODE, which is a
    /// hinge: a central-force lattice has no resistance to it at all, so the
    /// fragment rotates freely — and conjugate gradient will report a converged
    /// residual on a solution with a ten-kilometre displacement in it, because
    /// the load happened to be nearly orthogonal to the null space. Neither a
    /// per-node bond count nor a connected-component test sees this; the
    /// fragment IS connected and every node has three neighbours. What sees it
    /// is the displacement. A tooth under 1000 N moves tens of microns, so
    /// twenty times the elastic answer is not a tooth deforming, it is a piece
    /// of one swinging. That is a legitimate ending — the cusp has broken off —
    /// and it is recorded as its own outcome rather than counted as arrest.
    func crack(maxSteps: Int = 400, window: Float = 0.02, perStep: Int = 4,
               runawayFactor: Float = 20, heal: Bool = false) -> CrackResult {
        var steps: [CrackStep] = []
        var total = 0
        var arrested = false
        var detached = false
        var brokeDown = false
        let elastic: Float = max(lattice.maxDisplacement, 1e-12)
        let ceiling: Float = elastic * runawayFactor
        for _ in 0..<maxSteps {
            let (broken, worst) = lattice.breakWorst(window: window, limit: perStep)
            if broken.isEmpty { arrested = true; break }
            // The mutation: put back the bond that broke two steps ago. Test 7
            // exists to catch exactly this.
            if heal && lattice.breakOrder.count > 2 {
                let old = Int(lattice.breakOrder[lattice.breakOrder.count - 3])
                lattice.healForMutation(old)
            }
            total += broken.count
            total += lattice.pruneDetached()
            // A looser tolerance between breaks than for a reported field: this
            // solve only has to decide WHICH bond is worst next, and warm-started
            // from the last step it is a small correction. The field the render
            // paints is re-solved at the tight tolerance afterwards.
            let out = solve(tolerance: 2e-6, warmStart: true)
            if lattice.breakdown { brokeDown = true }
            steps.append(CrackStep(broken: broken, worst: worst, totalBroken: total,
                                   residual: out.residual))
            // Written as `!(x <= c)` so a NaN displacement takes this branch
            // rather than sailing past it.
            if !(lattice.maxDisplacement <= ceiling) {
                total += lattice.detachRunaway(threshold: ceiling)
                total += lattice.pruneDetached()
                detached = true
                solve(tolerance: 2e-6)
                break
            }
        }
        return CrackResult(steps: steps, arrested: arrested && !detached && !brokeDown,
                           fragmentDetached: detached, breakdown: brokeDown,
                           brokenByTissue: brokenByTissue(), dentinCore: dentinCoreBreaks(),
                           deepestPastDEJ: deepestPastDEJ(), totalBroken: total,
                           maxDisplacement: lattice.maxDisplacement)
    }

    // MARK: - the load ramp, which is where the arrest result actually lives

    /// One rung of the ramp: everything that happened at one load.
    struct Rung {
        var newtons: Float
        var broken: Int              // cumulative
        var brokeThisRung: [Int]     // bond indices, in the order they went
        var deepestPastDEJ: Float
        var dentinCore: Int
        var maxDisplacement: Float
    }

    struct RampResult {
        var rungs: [Rung]
        var initiationNewtons: Float   // the first bond anywhere in the tooth
        var junctionNewtons: Float     // the first bond with a junction end
        var dentinNewtons: Float       // the first bond with dentin at BOTH ends
        var fragmentNewtons: Float     // a piece came off
        var totalBroken: Int
        /// True if the run hit its step ceiling. Anything that breaks nine
        /// hundred bonds one at a time is not arresting.
        var exhausted: Bool

        /// How much harder you have to bite to push the crack out of the
        /// junction and into the dentin. This ONE NUMBER is the arrest result:
        /// above 1 the junction is holding the crack, at 1 it is not holding it
        /// at all. Infinity means the ramp ran out before the dentin went.
        var arrestMargin: Float {
            guard initiationNewtons.isFinite, initiationNewtons > 0 else { return 0 }
            return dentinNewtons / initiationNewtons
        }
        var arrested: Bool { arrestMargin > 1.0 && !exhausted }
    }

    /// Raise the load until the tooth cracks, and keep raising it.
    ///
    /// This replaces asking "does the crack arrest at 1000 N", which turned out
    /// to be the wrong question: at a fixed load a crack in a bending member
    /// either does not start or runs to failure, because the remaining ligament
    /// carries more as the crack grows. Under a RISING load the question has an
    /// answer with a number attached — how much further you have to push before
    /// the crack leaves the junction — and that number is what the junction is
    /// worth.
    ///
    /// At each rung the load is held and the crack allowed to grow to
    /// completion before the load goes up again, so nothing is a transient.
    func ramp(from startScale: Float = 0.2, to endScale: Float = 1.6,
              factor: Float = 1.06, stepsPerRung: Int = 120, maxTotalSteps: Int = 220,
              stopAtDentin: Bool = true,
              window: Float = 0.02, perStep: Int = 4, heal: Bool = false,
              snapshot: ((ToothModel) -> Void)? = nil) -> RampResult {
        var rungs: [Rung] = []
        var initiation: Float = .infinity
        var junction: Float = .infinity
        var dentin: Float = .infinity
        var fragment: Float = .infinity
        var exhausted = false
        var total = 0
        var scale: Float = startScale
        var elastic: Float = 0
        // A hard ceiling on the total number of break-and-re-solve steps. It is
        // not a physics parameter: it is what stops a MUTATED model — the one
        // with the root clamped rigidly, which shatters — from spending two
        // hours proving that it shatters. A run that hits it is by definition
        // not an arrest, and `arrested` says so.
        var totalSteps = 0

        while scale <= endScale {
            setLoadScale(scale)
            solve(warmStart: !rungs.isEmpty)
            snapshot?(self)
            if elastic == 0 { elastic = max(lattice.maxDisplacement, 1e-12) }
            let ceiling: Float = elastic * (scale / startScale) * 20
            var thisRung: [Int] = []
            var steps = 0
            var reachedDentin = false
            while steps < stepsPerRung && totalSteps < maxTotalSteps {
                let (broken, _) = lattice.breakWorst(window: window, limit: perStep)
                if broken.isEmpty { break }
                // The moment a bond with dentin at BOTH ends goes, the question
                // has its answer and there is nothing left to learn by grinding
                // the rest of the tooth to powder. Stopping here is most of what
                // makes this affordable to run six times over.
                for i in broken {
                    let bond = lattice.bonds[i]
                    if lattice.tissue[Int(bond.a)] == .dentin
                        && lattice.tissue[Int(bond.b)] == .dentin { reachedDentin = true }
                }
                // The mutation. It lives here as well as in `crack` because the
                // ramp is where the reported result comes from, and a mutation
                // guarding a path nobody reports from guards nothing.
                if heal && lattice.breakOrder.count > 2 {
                    let old = Int(lattice.breakOrder[lattice.breakOrder.count - 3])
                    lattice.healForMutation(old)
                }
                thisRung += broken
                total += broken.count
                total += lattice.pruneDetached()
                solve(tolerance: 2e-6, warmStart: true)
                snapshot?(self)
                steps += 1
                totalSteps += 1
                if reachedDentin { break }
                if !(lattice.maxDisplacement <= ceiling) {
                    total += lattice.detachRunaway(threshold: ceiling)
                    total += lattice.pruneDetached()
                    if fragment == .infinity { fragment = currentNewtons }
                    solve(tolerance: 2e-6)
                    break
                }
            }
            let newtons: Float = currentNewtons
            if !thisRung.isEmpty && initiation == .infinity { initiation = newtons }
            for i in thisRung {
                let bond = lattice.bonds[i]
                let ta: ToothTissue = lattice.tissue[Int(bond.a)]
                let tb: ToothTissue = lattice.tissue[Int(bond.b)]
                if junction == .infinity && (ta == .dej || tb == .dej) { junction = newtons }
                if dentin == .infinity && ta == .dentin && tb == .dentin { dentin = newtons }
            }
            rungs.append(Rung(newtons: newtons, broken: total, brokeThisRung: thisRung,
                              deepestPastDEJ: deepestPastDEJ(), dentinCore: dentinCoreBreaks(),
                              maxDisplacement: lattice.maxDisplacement))
            if fragment.isFinite { break }
            if stopAtDentin && dentin.isFinite { break }
            if totalSteps >= maxTotalSteps { exhausted = true; break }
            scale *= factor
        }
        total += lattice.pruneDetached()
        return RampResult(rungs: rungs, initiationNewtons: initiation,
                          junctionNewtons: junction, dentinNewtons: dentin,
                          fragmentNewtons: fragment, totalBroken: total,
                          exhausted: exhausted)
    }

    func brokenByTissue() -> [ToothTissue: Int] {
        var out: [ToothTissue: Int] = [:]
        for bond in lattice.bonds where bond.broken {
            let ta = lattice.tissue[Int(bond.a)]
            let tb = lattice.tissue[Int(bond.b)]
            // A bond is counted in its WEAKER tissue: it is only as strong as
            // its weaker end, which is how it was given its limit.
            let key: ToothTissue = rank(ta) <= rank(tb) ? ta : tb
            out[key, default: 0] += 1
        }
        return out
    }

    /// Softest first, so "weaker end" is a lookup rather than a chain of ifs.
    private func rank(_ t: ToothTissue) -> Int {
        switch t {
        case .pdl: return 0
        case .cementum: return 1
        case .dentin: return 2
        case .dej: return 3
        case .enamel: return 4
        default: return 5
        }
    }

    /// Broken bonds with dentin at both ends: a crack that is not merely
    /// touching the junction but running in the core. This is the number that
    /// says arrest failed.
    func dentinCoreBreaks() -> Int {
        var n = 0
        for bond in lattice.bonds where bond.broken {
            if lattice.tissue[Int(bond.a)] == .dentin && lattice.tissue[Int(bond.b)] == .dentin {
                n += 1
            }
        }
        return n
    }

    /// How far, in millimetres, the deepest break sits inside the dentin,
    /// measured from the junction's own radius at that height. Negative means
    /// nothing reached the dentin at all.
    func deepestPastDEJ() -> Float {
        var deepest: Float = -1
        for bond in lattice.bonds where bond.broken {
            for end in [Int(bond.a), Int(bond.b)] {
                guard lattice.tissue[end] == .dentin else { continue }
                let q: SIMD2<Float> = mm[end]
                let r: Float = dentinRadius(y: q.y)
                guard r > 0 else { continue }
                let depth: Float = r - abs(q.x)
                if depth > deepest { deepest = depth }
            }
        }
        return deepest
    }

    // MARK: - the field the renderer paints

    /// Maximum principal stress at every node, in pascals, with the lattice's
    /// own free-surface correction applied. Tension positive.
    func stressField() -> [Float] {
        var out = [Float](repeating: 0, count: lattice.tissue.count)
        for n in 0..<out.count where lattice.tissue[n].isTooth {
            out[n] = lattice.maxPrincipalStress(n)
        }
        return out
    }
}
