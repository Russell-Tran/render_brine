// The tests, in the order the step has to earn them.
//
// The solver is checked against arithmetic done on paper before anything is
// asked about teeth, because a stress field is not a thing you can look at and
// tell whether it is right. Then the model, then the picture.
//
// The mutation check is driven by TOOTH_MUTATE, and the mutations are:
//
//   clamp    pin the root rigidly below −6 mm, the way the first version did
//   brittle  give dentin enamel's tensile strength
//   heal     put a broken bond back every third break
//   freeze   make the evidence bar say the same thing in every beat
//   punch    load a flat uniform patch instead of a Hertzian one
//
// Each must turn the suite red, and `make mutants` prints which test caught it.

import Foundation
import simd

let mutation: String = ProcessInfo.processInfo.environment["TOOTH_MUTATE"] ?? ""
let mutateClamp: Bool = mutation == "clamp"
let mutateBrittle: Bool = mutation == "brittle"
let mutateHeal: Bool = mutation == "heal"
let mutateFreeze: Bool = mutation == "freeze"
let mutatePunch: Bool = mutation == "punch"

/// The dentin strength the model is built with. The `brittle` mutation gives
/// dentin enamel's, which should destroy crack arrest and nothing else.
let dentinStrengthUnderTest: Double? = mutateBrittle ? enamelStrength : nil

func testModel(spacing: Double, load: LoadCase, seed: UInt64 = 1,
               scatter: Double = 0.10) -> ToothModel {
    ToothModel(spacingMM: spacing, loadCase: load, seed: seed, scatter: scatter,
               rigidRoot: mutateClamp,
               dentinStrengthOverride: dentinStrengthUnderTest, flatPunch: mutatePunch)
}

// MARK: - 1. the solver is right before it is pretty

section("the solver, against arithmetic done on paper")

/// A rectangular cantilever of one material, loaded at the free end. Returns the
/// ratio of the lattice's tip deflection to the analytic one.
func cantilever(rows: Int, cols: Int, bending: Bool) -> (ratio: Double, residual: Float) {
    let a: Double = 1e-4
    let t: Double = 1e-3
    let e: Double = 20e9
    let root3: Double = 3.0.squareRoot()
    let props: [ToothTissue: TissueProperties] = [
        .dentin: TissueProperties(youngsModulus: e, tensileStrength: 1e30, name: "beam"),
    ]
    let lat = Lattice(cols: cols, rows: rows, spacing: a, thickness: t, properties: props,
                      tissueAt: { _ in .dentin },
                      kindAt: { p, _ in p.x < Float(1.6 * a) ? .fixed : .free })
    let h: Double = Double(rows - 1) * a * root3 / 2
    let l: Double = Double(cols - 1) * a
    let f: Double = bending ? 0.02 : 20.0
    for j in 0..<rows {
        let share: Float = Float(f / Double(rows))
        let load: SIMD2<Float> = bending ? SIMD2(0, -share) : SIMD2(share, 0)
        lat.setLoad(load, at: lat.index(col: cols - 1, row: j))
    }
    let out = lat.solve(tolerance: 1e-8, maxIterations: 20000)
    let mid: Int = lat.index(col: cols - 1, row: rows / 2)
    let measured: Double = bending ? Double(-lat.displacement(mid).y)
                                   : Double(lat.displacement(mid).x)
    let inertia: Double = t * h * h * h / 12
    let analytic: Double = bending ? f * l * l * l / (3 * e * inertia) : f * l / (e * h * t)
    return (measured / analytic, out.residual)
}

test("a bar in tension approaches δ = FL/EA as the lattice refines") {
    var previous: Double = 0
    for rows in [11, 21, 41] {
        let (ratio, residual) = cantilever(rows: rows, cols: 60, bending: false)
        expect(residual < 1e-7, "residual \(residual) at \(rows) rows")
        expect(ratio > previous, "ratio \(ratio) did not improve on \(previous)")
        expect(ratio < 1.02, "ratio \(ratio) overshot")
        previous = ratio
    }
    expect(previous > 0.93, "finest bar ratio only \(previous)")
}

test("a cantilever's bending error is exactly 3/rows, so the limit is FL³/3EI") {
    // The discrepancy is the free-surface half-cell: the outermost row of nodes
    // owns half a cell but is counted as a whole one, which softens the beam by
    // a fixed fraction of its depth. That predicts error ∝ 1/rows, and it is
    // worth far more than a single number close to 1, because a constant
    // error×rows says the CONTINUUM limit is the analytic answer.
    var products: [Double] = []
    var ratios: [Double] = []
    for rows in [11, 21, 31, 41] {
        let (ratio, residual) = cantilever(rows: rows, cols: 12 * rows, bending: true)
        expect(residual < 1e-7, "residual \(residual) at \(rows) rows")
        products.append((1 - ratio) * Double(rows))
        ratios.append(ratio)
    }
    // Ignore the coarsest, which is where higher-order terms still show.
    let tail: [Double] = Array(products.dropFirst())
    let mean: Double = tail.reduce(0, +) / Double(tail.count)
    for p in tail {
        expect(abs(p - mean) / mean < 0.10,
               "error×rows = \(products), not constant to 10%")
    }
    expect(abs(mean - 3) < 0.4, "error×rows is \(mean), expected about 3")
    // Richardson: with error = c/rows, extrapolating two ratios kills the term.
    let r1: Double = ratios[2]
    let r2: Double = ratios[3]
    let limit: Double = (41 * r2 - 31 * r1) / 10
    expect(abs(limit - 1) < 0.03, "extrapolated limit \(limit), expected 1")
}

test("E survives the round trip through the spring constant") {
    for e in [1.0e9, 18.0e9, 84.0e9, 50.0e6] {
        for t in [1.0e-3, 10.0e-3] {
            let k: Double = springConstant(youngsModulus: e, thickness: t)
            let back: Double = youngsModulus(springConstant: k, thickness: t)
            expect(abs(back - e) / e < 1e-12, "\(e) came back as \(back)")
        }
    }
    expect(abs(latticePoissonRatio - 1.0 / 3.0) < 1e-12)
}

test("a bond between two tissues fails at the weaker one's STRENGTH") {
    // Not at the weaker one's STRAIN. A ligament bonded to cementum has the
    // ligament's stiffness, and if it inherited cementum's critical strain it
    // would fail at 0.12 MPa and the whole interface would tear before the tooth
    // felt anything. It did, for a while.
    let props: [ToothTissue: TissueProperties] = [
        .pdl: TissueProperties(youngsModulus: 50e6, tensileStrength: 1e12, name: "pdl"),
        .cementum: TissueProperties(youngsModulus: 15e9, tensileStrength: 35e6, name: "cem"),
    ]
    let lat = Lattice(cols: 6, rows: 6, spacing: 1e-4, thickness: 1e-3, properties: props,
                      tissueAt: { p in p.x < 2.5e-4 ? .pdl : .cementum },
                      kindAt: { _, _ in .free })
    var mixed = 0
    for (i, bond) in lat.bonds.enumerated() {
        let ta = lat.tissue[Int(bond.a)]
        let tb = lat.tissue[Int(bond.b)]
        guard ta != tb else { continue }
        mixed += 1
        let expected: Float = Float(35e6 / 50e6)
        expect(abs(bond.limit - expected) / expected < 1e-5,
               "mixed bond \(i) fails at strain \(bond.limit), expected \(expected)")
    }
    expect(mixed > 0, "no mixed bonds in the fixture")
}

// MARK: - 2. the tooth solves

section("the tooth, solved")

let bruxModel = testModel(spacing: 0.10, load: lateralBruxing, scatter: 0)
let bruxSolve = bruxModel.solve()

test("conjugate gradient converges on the real tooth") {
    expect(bruxSolve.residual < 1e-7,
           "residual \(bruxSolve.residual) after \(bruxSolve.iterations) iterations")
    expect(bruxSolve.iterations < 40000, "took \(bruxSolve.iterations) iterations")
    expect(!bruxModel.lattice.breakdown, "the operator lost positive definiteness")
}

test("the tooth moves like a tooth, not like a mechanism") {
    // Physiological horizontal mobility is tens of microns. This model gives
    // more than that, and the reason is in the report: a central-force lattice
    // is stuck at Poisson 1/3 and the ligament is nearly incompressible, so the
    // ligament here is far more squeezable than the real one. The bound is
    // deliberately loose and its job is to catch a mechanism, not to claim
    // agreement.
    let micronsMoved: Float = bruxModel.lattice.maxDisplacement * 1e6
    expect(micronsMoved > 20, "only \(micronsMoved) µm — nothing is loading")
    expect(micronsMoved < 900, "\(micronsMoved) µm is not elasticity")
}

test("nothing is loaded anywhere but the contact patch") {
    var loadedNodes = 0
    var total = SIMD2<Float>(0, 0)
    for n in 0..<bruxModel.lattice.load.count {
        let f: SIMD2<Float> = bruxModel.lattice.load[n]
        guard simd_length(f) > 0 else { continue }
        loadedNodes += 1
        total += f
        expect(bruxModel.isSurfaceNode(n), "node \(n) is loaded but is not on a surface")
    }
    expect(loadedNodes >= 5, "only \(loadedNodes) contact nodes")
    let magnitude: Float = simd_length(total)
    let wanted: Float = Float(lateralBruxing.newtons)
    expect(abs(magnitude - wanted) / wanted < 1e-3,
           "the contact adds up to \(magnitude) N, not \(wanted)")
}

test("the contact traction is Hertzian, so it has no edge to concentrate at") {
    // Two curved cusps meeting give p ∝ √(1 − (x/a)²), which goes to zero at the
    // edges. A uniform patch is a FLAT PUNCH and a flat punch has a stress
    // singularity at each of its edges (Johnson, *Contact Mechanics*, 1985,
    // §2.8) — so a model loaded that way puts its worst stress under the corner
    // of its own fixture and calls it a finding. Nothing else in this suite
    // notices the difference, so the assertion belongs here, on the fixture.
    let a: Float = lateralBruxing.contactHalfWidth
    var peak: Float = 0
    var samples: [(Float, Float)] = []
    for (i, n) in bruxModel.contactNodes.enumerated() {
        let magnitude: Float = simd_length(bruxModel.lattice.load[n])
        let u: Float = (bruxModel.mm[n].x - lateralBruxing.contactX) / a
        peak = max(peak, magnitude)
        samples.append((u, magnitude))
        expect(i < 200)
    }
    expect(samples.count >= 5, "only \(samples.count) contact nodes to check")
    guard peak > 0 else { expect(false, "nothing is loaded"); return }
    // Normalised against the node nearest the middle, the profile must be the
    // ellipse and not a rectangle.
    var reference: Float = 0
    var closest: Float = .greatestFiniteMagnitude
    for (u, w) in samples where abs(u) < closest { closest = abs(u); reference = w }
    let shape: Float = max(1 - closest * closest, 0).squareRoot()
    for (u, w) in samples {
        let wanted: Float = reference / shape * max(1 - u * u, 0).squareRoot()
        expect(abs(w - wanted) <= 0.03 * peak,
               "at u = \(u) the contact carries \(w) N, and an ellipse wants \(wanted)")
    }
    // And the edges really do fall away.
    let outermost: Float = samples.max { abs($0.0) < abs($1.0) }?.1 ?? peak
    expect(outermost < 0.75 * peak,
           "the outermost contact node carries \(outermost / peak) of the peak")
}

test("the oblique load's direction is the cusp facet's, not a typed number") {
    // occlusalHeight = floor + D·sin²(πx/S), so dy/dx = D·(π/S)·sin(2πx/S).
    // Differencing the height itself is the check that the derivative is the
    // derivative of the thing actually drawn.
    for x in [Float(0.8), 1.3, 2.0, 3.25, 4.0] {
        let d: Float = 1e-3
        let numeric: Float = (occlusalHeight(x: x + d) - occlusalHeight(x: x - d)) / (2 * d)
        let analytic: Float = occlusalSlope(x: x)
        expect(abs(numeric - analytic) < 2e-3,
               "slope at \(x): \(analytic) against \(numeric) by difference")
    }
    // And the force is the inward normal rotated by atan(friction).
    for x in [innerInclineX, outerInclineX] {
        let m: Float = occlusalSlope(x: x)
        let outward: SIMD2<Float> = simd_normalize(SIMD2<Float>(-m, 1))
        let dir: SIMD2<Float> = facetForceDirection(x: x)
        expect(abs(simd_length(dir) - 1) < 1e-5, "direction is not a unit vector")
        let angle: Float = acos(min(max(simd_dot(dir, -outward), -1), 1))
        let expected: Float = atan(enamelFriction)
        expect(abs(angle - expected) < 1e-3,
               "at x=\(x) the force is \(angle) rad off the normal, not \(expected)")
    }
    // The two inclines pull opposite ways, which is the whole point of the
    // lesion beat.
    expect(facetForceDirection(x: innerInclineX).x > 0.2,
           "the lingual incline should press buccally")
    expect(facetForceDirection(x: outerInclineX).x < -0.2,
           "the buccal incline should press lingually")
}

// MARK: - 3. mesh independence

/// What "cervical" means, as a band of heights. It stops just above the
/// alveolar crest at −1.5 mm: below that is the socket margin, which is a
/// different question, and the crest itself is a support discontinuity.
let cervicalLow: Float = -1.2
let cervicalHigh: Float = 1.0

section("mesh independence")

let coarseModel = testModel(spacing: 0.20, load: lateralBruxing, scatter: 0)
let fineModel = testModel(spacing: 0.05, load: lateralBruxing, scatter: 0)

test("the cervical stress does NOT converge, and the way it fails is a power law") {
    coarseModel.solve()
    fineModel.solve()
    let h20: Float = coarseModel.peakSurfaceTension(between: cervicalLow, and: cervicalHigh)
    let h10: Float = bruxModel.peakSurfaceTension(between: cervicalLow, and: cervicalHigh)
    let h05: Float = fineModel.peakSurfaceTension(between: cervicalLow, and: cervicalHigh)

    // This is the honest result and it is not the one the brief asked for. The
    // cervical stress GROWS as the mesh refines — about 47, 69, 102 MPa at 0.20,
    // 0.10 and 0.05 mm — and it grows by the same FACTOR each time the mesh
    // halves. A constant ratio is a power law, σ ∝ h^−0.55, which is the
    // signature of a material-WEDGE singularity: enamel thins to a knife edge
    // against cementum at the neck, and a wedge of one stiff material into
    // another has an unbounded elastic stress at its tip. No mesh will ever
    // produce a number, because there is no number to produce.
    //
    // This is exactly the limitation the abfraction reviews warn about in
    // published finite element work, and it is why this step reports the
    // cervical stress as "over enamel's strength at bruxing loads" and as a
    // RATIO to the crown, and never as a value.
    expect(h10 > h20 && h05 > h10, "cervical stress \(h20), \(h10), \(h05) is not growing")
    let first: Float = h10 / h20
    let second: Float = h05 / h10
    expect(abs(first - second) / first < 0.15,
           "growth per halving is \(first) then \(second) — not a clean power law")
    let exponent: Float = log(second) / log(2)
    expect(exponent > 0.3 && exponent < 0.8,
           "the exponent is \(exponent); a wedge singularity should be a few tenths")
}

test("what SHOULD converge does: the tooth's displacement") {
    // A singular point stress does not stop the global answer converging. If the
    // displacement were drifting too, the solver would be wrong rather than the
    // question being ill-posed.
    let d20: Float = coarseModel.lattice.maxDisplacement
    let d10: Float = bruxModel.lattice.maxDisplacement
    let d05: Float = fineModel.lattice.maxDisplacement
    let first: Float = abs(d10 - d20)
    let second: Float = abs(d05 - d10)
    expect(d20 > d10 && d10 > d05,
           "displacement is not settling: \(d20), \(d10), \(d05)")
    expect(second < first * 0.85,
           "the steps are \(first * 1e6) then \(second * 1e6) µm — not converging")
    expect(abs(d05 - d20) / d05 < 0.32,
           "the displacement spans \(abs(d05 - d20) / d05 * 100)% across three meshes")
}

// MARK: - 4. where the maximum is

section("where the maximum is")

test("the neck carries the maximum under an oblique load, and the crown does not") {
    let cervical: Float = bruxModel.peakSurfaceTension(between: cervicalLow, and: cervicalHigh)
    let crown: Float = bruxModel.peakSurfaceTension(between: cervicalHigh, and: 7.6)
    expect(cervical > crown,
           "cervical \(cervical / 1e6) MPa against crown \(crown / 1e6) MPa")
    let peak = bruxModel.peakSurfaceTension()
    expect(abs(peak.at.y) < 0.6,
           "the tooth's maximum sits at y = \(peak.at.y) mm, not at the neck")
}

test("oblique beats axial at the neck — but only on one of the two inclines") {
    // This is the honest form of "lateral exceeds axial". It is true for a
    // contact on the cusp's LINGUAL incline, where the horizontal component's
    // moment and the vertical component's eccentricity add, and FALSE for the
    // buccal incline, where they oppose. Which incline a grinding contact lands
    // on therefore decides whether the neck or the crown is the worst place in
    // the tooth — and that is one of the reasons abfraction is contested.
    let axial = testModel(spacing: 0.10, load: axialBruxing, scatter: 0)
    let outer = testModel(spacing: 0.10, load: outerInclineBruxing, scatter: 0)
    axial.solve()
    outer.solve()
    let cervicalOblique: Float = bruxModel.peakSurfaceTension(between: cervicalLow, and: cervicalHigh)
    let cervicalAxial: Float = axial.peakSurfaceTension(between: cervicalLow, and: cervicalHigh)
    let cervicalOuter: Float = outer.peakSurfaceTension(between: cervicalLow, and: cervicalHigh)
    expect(cervicalOblique > cervicalAxial,
           "lingual incline \(cervicalOblique / 1e6) did not beat axial \(cervicalAxial / 1e6)")
    expect(cervicalOuter < cervicalAxial,
           "buccal incline \(cervicalOuter / 1e6) was expected BELOW axial "
           + "\(cervicalAxial / 1e6) — the moments oppose there")
    expect(outer.peakSurfaceTension(between: cervicalHigh, and: 7.6) > cervicalOuter,
           "on the buccal incline the crown should carry more than the neck")
}

test("the lattice agrees with beam theory on the neck, and says where it does not") {
    // Euler–Bernoulli on the CEJ section is the only independent check available
    // for this number. It agrees on WHICH FACE is in tension and on the order of
    // magnitude. It does not agree on the size, and it should not: the neck is a
    // notch and the enamel on it is stiffer than the dentin the beam assumes.
    let beam = cervicalBeamStress(lateralBruxing)
    expect(!beam.tensionOnBuccal, "beam theory puts the tension on the lingual face")
    var latticeBuccal: Float = 0
    var latticeLingual: Float = 0
    for n in bruxModel.surfaceNodes {
        let q: SIMD2<Float> = bruxModel.mm[n]
        guard abs(q.y) < 0.5 else { continue }
        let s: Float = bruxModel.lattice.peakBondStress(n)
        if q.x > 0 { latticeBuccal = max(latticeBuccal, s) }
        if q.x < 0 { latticeLingual = max(latticeLingual, s) }
    }
    expect(latticeLingual > latticeBuccal,
           "the lattice puts the cervical tension on the buccal face, beam theory does not")
    let ratio: Float = latticeLingual / Float(beam.dentinLingual)
    expect(ratio > 1.3 && ratio < 6,
           "the lattice is \(ratio)× beam theory at the neck; expected between 1.3 and 6, "
           + "because the neck is a notch and the skin on it is stiff")
}

test("compression never breaks a bond") {
    // Push the cusp straight down HARD and check that everything that fails is
    // in tension. Enamel fails in tension; a model that let it crumble under
    // compression would be telling a different story from the literature.
    let squash = LoadCase(name: "pure squeeze", newtons: 4000,
                          direction: SIMD2<Float>(0, -1), contactX: 0,
                          contactHalfWidth: 2.4)
    let model = testModel(spacing: 0.20, load: squash, scatter: 0)
    model.solve()
    for i in 0..<model.lattice.bonds.count {
        let strain: Float = model.lattice.bondStrain(i)
        if model.lattice.overload(i) > 1 {
            expect(strain > 0, "bond \(i) is over its limit at strain \(strain)")
        }
    }
    // And the rule itself, directly.
    for i in 0..<model.lattice.bonds.count where model.lattice.bondStrain(i) < 0 {
        expectEqual(model.lattice.overload(i), 0)
    }
}

// MARK: - 5. cracks

section("cracks")

/// One ramp per seed, at a mesh coarse enough that a dozen of them is a test and
/// not an afternoon.
func rampFor(seed: UInt64) -> (ToothModel, ToothModel.RampResult) {
    // 0.10 mm and not coarser. The junction is a 0.18 mm band, so at 0.20 mm it
    // is thinner than one cell and there is nothing there to arrest anything —
    // the margin comes out at 1.00 for every seed, which is a statement about
    // the mesh. A test of arrest at the junction has to resolve the junction.
    let model = testModel(spacing: 0.10, load: lateralBruxing, seed: seed)
    model.solve()
    // No `heal:` here on purpose. The healing mutation is guarded by its own
    // test below, on a quarter-millimetre mesh that runs in seconds; switching
    // it on here as well would make the mutation check spend an hour proving
    // something a twenty-second test already proved, because a crack that heals
    // never terminates and every ramp would run to its step ceiling.
    let result = model.ramp(from: 0.5, to: 1.6, factor: 1.06)
    return (model, result)
}

let seeds: [UInt64] = [1, 2, 3, 4, 5, 6]
let ramps: [(ToothModel, ToothModel.RampResult)] = seeds.map { rampFor(seed: $0) }

test("cracks start in the tooth's mineral and reach the junction") {
    for (i, pair) in ramps.enumerated() {
        let r = pair.1
        expect(r.initiationNewtons.isFinite,
               "seed \(seeds[i]): nothing ever cracked, even at 1600 N")
        expect(r.junctionNewtons.isFinite,
               "seed \(seeds[i]): the crack never reached the junction")
    }
}

test("cracks arrest at the junction, as a rate across seeds") {
    // The load has to rise before the crack leaves the junction for the dentin.
    // That margin IS the arrest: at 1.0 the junction is holding nothing. It is
    // asserted as a rate over eight seeded runs rather than shown once, because
    // one run is an anecdote.
    var arrested = 0
    var margins: [Float] = []
    for (i, pair) in ramps.enumerated() {
        let r = pair.1
        margins.append(r.arrestMargin)
        if r.arrestMargin > 1.05 && !r.exhausted { arrested += 1 }
        expect(!r.exhausted,
               "seed \(seeds[i]) never stopped breaking bonds; that is not an arrest")
        expect(r.arrestMargin >= 1.0,
               "seed \(seeds[i]) has margin \(r.arrestMargin), which is below one")
    }
    let rate: Double = Double(arrested) / Double(seeds.count)
    expect(rate >= 0.8, "only \(arrested)/\(seeds.count) runs arrested; margins \(margins)")
}

test("the crack is a crack and not an avalanche") {
    // Breaking every over-limit bond at once, which is what the first version of
    // this step did, gives tens of thousands of broken bonds in ten rounds. The
    // fuse rule breaks the worst one and re-solves. Below the load that splits
    // the tooth, the crack must still be countable.
    for (i, pair) in ramps.enumerated() {
        let model = pair.0
        let r = pair.1
        var beforeDentin = 0
        for rung in r.rungs where rung.newtons < r.dentinNewtons {
            beforeDentin += rung.brokeThisRung.count
        }
        expect(beforeDentin < 600,
               "seed \(seeds[i]) broke \(beforeDentin) bonds before the dentin went")
        expect(model.lattice.detachedNodes >= 0)
    }
}

test("cracks never heal") {
    // Run a ramp and watch the broken set. It may only grow, ever, and that has
    // to hold for the path the step actually reports from — which is the ramp,
    // not `crack`.
    let model = testModel(spacing: 0.25, load: lateralBruxing, seed: 11)
    model.solve()
    var seen = Set<Int32>()
    var everShrank = false
    var previousCount = 0
    // Bounded hard. A crack that heals never terminates, so without a ceiling
    // this runs until the step cap on every rung; and forty steps is already a
    // dozen times more than the mutation needs to show itself, since it puts a
    // bond back on the third break.
    _ = model.ramp(from: 0.4, to: 1.5, factor: 1.06, stepsPerRung: 20,
                   maxTotalSteps: 40, heal: mutateHeal) { m in
        let broken: Set<Int32> = Set(m.lattice.bonds.enumerated()
            .filter { $0.element.broken }.map { Int32($0.offset) })
        if !seen.isSubset(of: broken) { everShrank = true }
        if broken.count < previousCount { everShrank = true }
        previousCount = broken.count
        seen.formUnion(broken)
    }
    expect(!everShrank, "a bond that had broken was whole again later in the run")
    for id in seen {
        expect(model.lattice.bonds[Int(id)].broken,
               "bond \(id) broke during the run and is whole at the end")
    }
}

test("a dangling node is cut loose rather than left to swing") {
    // Two bonds that are not opposite each other pin a point in the plane;
    // fewer than that and the node is spall. Left in, it makes the stiffness
    // operator singular and conjugate gradient reports a converged residual on
    // a displacement of kilometres.
    let model = testModel(spacing: 0.25, load: lateralBruxing, seed: 5)
    model.solve()
    _ = model.ramp(from: 0.4, to: 1.5, factor: 1.06)
    let lattice = model.lattice
    for n in 0..<lattice.tissue.count {
        guard lattice.tissue[n].isSolid, lattice.kind[n] != .fixed else { continue }
        var directions: [SIMD2<Float>] = []
        for id in lattice.bondsOf[n] where !lattice.bonds[Int(id)].broken {
            var d: SIMD2<Float> = lattice.bondDirection(lattice.bonds[Int(id)])
            if Int(lattice.bonds[Int(id)].b) == n { d = -d }
            directions.append(d)
        }
        if directions.count == 1 {
            expect(false, "node \(n) is held by exactly one bond")
        }
        if directions.count == 2 {
            expect(simd_dot(directions[0], directions[1]) > -0.99,
                   "node \(n) is held by two opposite bonds, which is a hinge")
        }
    }
}

// MARK: - 6. the body and the picture

section("the body, the loop and the bar")

let bodyPrims: [GPUPrim] = toothPrimitives()

test("the revolved body is the section it was solved in") {
    // The picture and the physics have to be the same tooth, or the colour on
    // the cut face is painted on a shape that was never solved. Sampled over the
    // section on a grid, the primitives' priority resolve must agree with
    // `tissueAt` everywhere except within a stack's own discretisation of a
    // boundary.
    var agree = 0
    var disagree = 0
    var farFromBoundary = 0
    var y: Float = -13.6
    while y < 7.7 {
        var x: Float = -6.0
        while x < 6.0 {
            var wanted: ToothTissue = tissueAt(SIMD2(x, y))
            var got: ToothTissue = tissueOf(materialAt(SIMD3(x, y, 0), prims: bodyPrims))
            // The RENDER's bone is a whole jaw block and the LATTICE's is a
            // half-millimetre shell, because bone is held rigid and a thicker
            // shell would change nothing in the solve. Outside the tooth and its
            // ligament the two are not claiming the same thing, so the
            // comparison treats bone and empty space alike.
            if wanted == .bone { wanted = .outside }
            if got == .bone { got = .outside }
            if wanted == got {
                agree += 1
            } else {
                disagree += 1
                // How far is this point from ANY boundary? If it is more than a
                // stack's step away, the disagreement is not discretisation.
                let d: Float = 0.40
                var onBoundary = false
                for (dx, dy) in [(d, 0), (-d, 0), (0, d), (0, -d),
                                 (d, d), (d, -d), (-d, d), (-d, -d)] as [(Float, Float)] {
                    var near: ToothTissue = tissueAt(SIMD2(x + dx, y + dy))
                    if near == .bone { near = .outside }
                    if near != wanted { onBoundary = true }
                }
                if !onBoundary { farFromBoundary += 1 }
            }
            x += 0.05
        }
        y += 0.05
    }
    let rate: Double = Double(disagree) / Double(agree + disagree)
    // A few per cent, all of it within a stack's own step of a boundary, is the
    // discretisation of the stacks and nothing else.
    expect(rate < 0.04, "the body and the section disagree at \(rate * 100)% of points")
    expect(farFromBoundary == 0,
           "\(farFromBoundary) disagreements are not near any boundary")
}

test("no ray in the frame overflows step 13's interval list") {
    // Step 13's ray holds forty intervals and the list is in the spliced prefix,
    // so this step cannot change it. A stack of discs is what spends that
    // budget: a ray down the axis of the root crosses five stacks at once.
    let camera: Camera = cutawayCamera(width: 480, height: 580, target: cameraTarget,
                                       yaw: cameraYaw)
    let worst: Int = worstIntervalCount(prims: bodyPrims, camera: camera,
                                        width: 480, height: 580, stride: 5)
    expect(worst <= maxIntervalsPerRay - 6,
           "the busiest ray collects \(worst) intervals of \(maxIntervalsPerRay)")
    expect(bodyPrims.count < mailboxCapacity,
           "\(bodyPrims.count) primitives, mailbox holds \(mailboxCapacity)")
}

test("the cut face is capped wherever the plane is inside the tooth") {
    // Step 15's result, re-checked on this geometry: a point on the clip plane
    // that is inside a solid must come back as CUT MATERIAL, never as a surface
    // and never as sky.
    let camera: Camera = cutawayCamera(width: 200, height: 240, target: cameraTarget,
                                       yaw: cameraYaw)
    var tested = 0
    var y: Float = -12.5
    while y < 7.0 {
        var x: Float = -4.0
        while x < 4.0 {
            let q = SIMD3<Float>(x, y, cutPlaneZ)
            guard let wanted = materialAt(q, prims: bodyPrims) else { x += 0.4; continue }
            let screen: SIMD2<Float> = camera.project(q, width: 200, height: 240)
            let (o, d) = camera.ray(sx: screen.x, sy: screen.y, width: 200, height: 240)
            guard let hit = cpuResolve(origin: o, direction: d, prims: bodyPrims,
                                       cutZ: cutPlaneZ) else {
                expect(false, "a hole in the cap at (\(x), \(y))")
                x += 0.4
                continue
            }
            expect(hit.isCut, "(\(x), \(y)) came back as a surface, not a cut face")
            expectEqual(hit.material, wanted)
            tested += 1
            x += 0.4
        }
        y += 0.4
    }
    expect(tested > 200, "only \(tested) points were inside the tooth")
}

test("the splice still finds step 13's traversal") {
    expect(swimKernelSource.contains(traversalMarker), "step 13's marker has moved")
    for name in ["ellipsoidInterval", "capsuleInterval", "addInterval", "gather"] {
        expect(traversalPrefix.contains(name), "the spliced prefix has lost \(name)")
    }
    expect(!traversalPrefix.contains("kernel void render"),
           "the splice took step 13's own kernel with it")
    expect(cutawayKernelSource.contains("kernel void cutaway"))
}

test("the Swift and Metal stress ramps are the same ramp") {
    // The key is drawn in Swift and the cut face is coloured in Metal, from two
    // copies of the same arithmetic. They have to agree or the key is a lie.
    for i in 0...12 {
        let f: Float = Float(i) / 12
        let mpa: Float = (f * 2 - 1) * 45
        let c: SIMD3<Float> = stressColour(mpa, range: 45)
        expect(c.x >= 0 && c.x <= 1 && c.y >= 0 && c.y <= 1 && c.z >= 0 && c.z <= 1)
    }
    expect(simd_length(stressColour(0, range: 45) - SIMD3<Float>(0.90, 0.89, 0.86)) < 1e-6,
           "zero stress is not the ramp's neutral")
    // Warmth is red minus blue: the hot end is 0.71 and the neutral 0.04. The
    // red CHANNEL alone is not the test — the hot end is darker than the
    // neutral, so red-versus-red goes the other way.
    func warmth(_ c: SIMD3<Float>) -> Float { c.x - c.z }
    expect(warmth(stressColour(45, range: 45)) > warmth(stressColour(0, range: 45)) + 0.4)
    expect(warmth(stressColour(-45, range: 45)) < warmth(stressColour(0, range: 45)) - 0.3)
    let metal: String = cutawayShadingSource
    for stop in ["float3(0.97, 0.80, 0.30)", "float3(0.85, 0.22, 0.14)",
                 "float3(0.66, 0.78, 0.86)", "float3(0.16, 0.36, 0.60)",
                 "float3(0.90, 0.89, 0.86)"] {
        expect(metal.contains(stop), "the kernel's ramp is missing the stop \(stop)")
    }
}

test("the evidence bar changes across the loop") {
    var seen = Set<String>()
    var standings = Set<Standing>()
    for b in Beat.allCases {
        let rows: [StandingRow] = mutateFreeze
            ? standingRows(.anatomy) : standingRows(b)
        expect(rows.count == 3, "\(b) has \(rows.count) rows")
        let key: String = rows.map { "\($0.standing.rawValue)|\($0.note)" }.joined(separator: "/")
        seen.insert(key)
        for r in rows { standings.insert(r.standing) }
    }
    expect(seen.count == Beat.allCases.count,
           "the bar shows only \(seen.count) different things over \(Beat.allCases.count) beats")
    expect(standings.count == 4, "the bar never uses all four standings")
}

test("the lesion beat is CONTESTED and names the alternatives") {
    expectEqual(evidenceFor(.lesion), .contested)
    let rows: [StandingRow] = standingRows(.lesion)
    expect(rows.contains { $0.standing == .contested })
    let text: String = rows.map { $0.note }.joined(separator: " ")
        + " " + noteFor(.lesion) + " " + captionFor(.lesion)
    expect(text.lowercased().contains("abrasion"), "abrasion is not named")
    expect(text.lowercased().contains("erosion"), "erosion is not named")
    expect(text.lowercased().contains("abfraction"), "abfraction is not named")
    let fact = toothFacts.first { $0.name.contains("abfraction") }
    expect(fact?.standing == .contested, "the abfraction fact is not marked contested")
}

test("the loop closes on a fresh tooth and never runs backwards") {
    expectEqual(toothFrameCount, 144)
    expectEqual(beat(frame: 0), .anatomy)
    expectEqual(beat(frame: toothFrameCount - 1), .lesion)
    expect(dissolve(frame: dissolveStart - 1) == 0, "the dissolve starts early")
    expect(dissolve(frame: toothFrameCount - 1) >= 1,
           "the dissolve does not finish, so the loop does not close")
    var previous: Float = 0
    for f in dissolveStart..<toothFrameCount {
        let a: Float = dissolve(frame: f)
        expect(a >= previous, "the dissolve went backwards at frame \(f)")
        previous = a
    }
}

test("every constant carries a source") {
    expect(toothFacts.count >= 18, "only \(toothFacts.count) facts")
    for fact in toothFacts {
        expect(!fact.source.isEmpty, "\(fact.name) has no source")
        expect(fact.source.count > 8, "\(fact.name)'s source is \(fact.source)")
        expect(!fact.name.isEmpty)
    }
    for standing in Standing.allCases {
        expect(toothFacts.contains { $0.standing == standing },
               "nothing in the table is \(standing.rawValue)")
    }
    // The three things this step could most easily have fudged.
    for needed in ["periodontal ligament modulus", "alveolar crest position",
                   "critical strains", "the three-dimensional body"] {
        expect(toothFacts.contains { $0.name == needed }, "\(needed) is not in the table")
    }
}

finish()
