// The tests for step 18.
//
// The render makes two claims at once — a molecular event on the left, a bulk
// observable on the right — and the only thing that makes that worth doing is
// that the two are not allowed to drift apart. Most of what follows is about
// enforcing that: the reading is counted off the molecules drawn, the twist of
// the beam is measured back off the geometry that was emitted, and the two have
// to agree at every frame.
//
// The rest is about the two soft spots this render has, which are stated on the
// bar because the picture will not show them: 3.40 Å is coarse for a surface
// made of side chains, and no crystal of this enzyme has ever had sucrose in it.

import Foundation
import simd

let sceneURL = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent().deletingLastPathComponent()
    .appendingPathComponent("Resources/invertase.json")

let scene: InvertaseScene
do {
    scene = try loadScene(from: sceneURL)
} catch {
    FileHandle.standardError.write("tests: \(error)\n".data(using: .utf8)!)
    exit(1)
}

// The same loop the renderer encodes, not a second one that happens to look
// like it.
let cellMolecules = defaultCellMolecules
let frameCount = defaultFrameCount

/// Every frame of the loop, as the renderer sees it.
struct Sample {
    var frame: Int
    var beat: Beat
    var prims: [GPUPrim]
    var readout: PolarimeterReadout
    var twist: Double
}

let samples: [Sample] = (0..<frameCount).map { f in
    let b = beat(at: Double(f) / Double(frameCount))
    let (prims, readout) = polarimeterScene(progress: b.progress, fill: b.fill,
                                            molecules: cellMolecules,
                                            lightColour: SIMD3(1, 0.7, 0.2))
    return Sample(frame: f, beat: b, prims: prims, readout: readout,
                  twist: measuredTwist(prims, barIndices: readout.barIndices))
}
let reaction = samples.filter { $0.beat.stage == .reaction }

// MARK: -

section("the sign really flips, and it is derived")

test("invert sugar's −19.9° is computed from the other two, not quoted") {
    let mixture = Composition(sucrose: 0, glucose: cellMolecules, fructose: cellMolecules)
    let derived = specificRotation(of: mixture)
    // The mass-weighted mean, worked out here independently of the code under test.
    let expected: Double = (glucose.specificRotation * glucose.molarMass
                            + fructose.specificRotation * fructose.molarMass)
                           / (glucose.molarMass + fructose.molarMass)
    expect(abs(derived - expected) < 1e-9, "derived \(derived), expected \(expected)")
    expect(abs(derived - (-19.85)) < 0.02, "should land on the tabulated −19.9°, got \(derived)")
    // Anything that returns the mixture's number whatever it is handed fails here.
    let pureSucrose = Composition(sucrose: cellMolecules, glucose: 0, fructose: 0)
    expect(abs(specificRotation(of: pureSucrose) - sucrose.specificRotation) < 1e-9,
           "untouched sucrose must read its own [α], got \(specificRotation(of: pureSucrose))")
    let pureGlucose = Composition(sucrose: 0, glucose: cellMolecules, fructose: 0)
    expect(abs(specificRotation(of: pureGlucose) - glucose.specificRotation) < 1e-9,
           "glucose alone must read its own [α]")
}

test("the reading starts positive, ends negative, and crosses zero once") {
    guard let first = reaction.first, let last = reaction.last else {
        expect(false, "no reaction frames")
        return
    }
    expect(first.readout.rotationDegrees > 0,
           "fresh sucrose must be dextrorotatory, read \(first.readout.rotationDegrees)")
    expect(last.readout.rotationDegrees < 0,
           "inverted sugar must be laevorotatory, read \(last.readout.rotationDegrees)")
    var crossings = 0
    for i in 1..<reaction.count {
        let a = reaction[i - 1].readout.rotationDegrees
        let b = reaction[i].readout.rotationDegrees
        if (a > 0 && b <= 0) || (a < 0 && b >= 0) { crossings += 1 }
    }
    expectEqual(crossings, 1)
}

test("the reading moves the one way each stage allows") {
    // Hydrolysis does not run backwards, so the reading may only fall while it
    // is happening. Draining can only shrink whatever the reading was, toward
    // zero; filling can only build it back up. Anything else in this loop would
    // be the encoder telling a story the chemistry does not.
    for i in 1..<samples.count {
        let previous = samples[i - 1], current = samples[i]
        let before = previous.readout.rotationDegrees
        let after = current.readout.rotationDegrees
        switch current.beat.stage {
        case .reaction:
            if previous.beat.stage != .reaction { continue }      // the stage boundary
            expect(after <= before + 1e-9,
                   "frame \(current.frame) read \(after) after \(before): hydrolysis went backwards")
        case .drain:
            expect(abs(after) <= abs(before) + 1e-9,
                   "frame \(current.frame) read further from zero while draining")
        case .refill:
            if previous.beat.stage != .refill { continue }
            expect(after >= before - 1e-9,
                   "frame \(current.frame) read closer to zero while filling")
        }
    }
}

test("zero is crossed about three quarters of the way, not halfway") {
    let crossing = zeroCrossingFraction(molecules: cellMolecules)
    expect(crossing > 0.70 && crossing < 0.80, "crossing at \(crossing)")
    // Worked out here from the two end points: sucrose starts a long way
    // positive, the products only end a little way negative.
    let positive: Double = sucrose.specificRotation * sucrose.molarMass
    let negative: Double = -(glucose.specificRotation + fructose.specificRotation) * glucose.molarMass
    let expected: Double = positive / (positive + negative)
    expect(abs(crossing - expected) < 1e-6, "\(crossing) against \(expected)")
}

test("hydrolysis takes up a water, and the final reading knows it") {
    let ends = rotationAtRest(molecules: cellMolecules)
    let naive: Double = -19.85 / sucrose.specificRotation
    let actual: Double = ends.end / ends.start
    expect(actual < naive - 0.008,
           "the mass gain must make the end MORE negative: \(actual) against \(naive)")
    let massRatio: Double = 2 * glucose.molarMass / sucrose.molarMass
    expect(abs(actual / naive - massRatio) < 1e-6,
           "the whole difference should be the mass ratio \(massRatio), got \(actual / naive)")
}

// MARK: -

section("the two panels agree")

test("the beam's twist is the reading, one to one, at every frame") {
    var worst: Double = 0
    for s in samples {
        worst = max(worst, abs(s.twist - s.readout.rotationDegrees))
    }
    expect(worst < 0.02, "worst disagreement between the drawn twist and the reading: \(worst)°")
}

test("the fraction the beam implies is the fraction of glyphs actually drawn") {
    var worst: Double = 0
    for s in reaction {
        let drawn: Int = s.readout.sucroseGlyphs + s.readout.splitGlyphs
        let counted: Double = Double(s.readout.splitGlyphs) / Double(drawn)
        let implied: Double = fractionImplied(byRotation: s.twist, molecules: drawn)
        worst = max(worst, abs(counted - implied))
    }
    // One molecule in 96 is 1.04%; the tolerance is that rounding and no more.
    expect(worst <= 1.1 / Double(cellMolecules),
           "worst disagreement between counted and implied conversion: \(worst)")
}

test("the flask never gets ahead of or behind the active site") {
    var worst: Double = 0
    for s in reaction {
        let drawn: Int = s.readout.sucroseGlyphs + s.readout.splitGlyphs
        if drawn == 0 { continue }
        let counted: Double = Double(s.readout.splitGlyphs) / Double(drawn)
        worst = max(worst, abs(counted - s.beat.progress))
    }
    expect(worst <= 1.1 / Double(cellMolecules),
           "the cell drifted \(worst) away from the catalytic progress")
}

test("what is in the cell is what was counted") {
    for s in samples {
        expectEqual(s.readout.composition.sucrose, s.readout.sucroseGlyphs)
        expectEqual(s.readout.composition.glucose, s.readout.splitGlyphs)
        expectEqual(s.readout.composition.fructose, s.readout.splitGlyphs)
    }
}

// MARK: -

section("retention, not inversion — the thing the name gets wrong")

test("the anomeric centre inverts twice and ends where it started") {
    guard let michaelis = anomericConfiguration(scene, state: "michaelis"),
          let covalent = anomericConfiguration(scene, state: "covalent"),
          let hydrolysis = anomericConfiguration(scene, state: "hydrolysis"),
          let product = anomericConfiguration(scene, state: "product") else {
        expect(false, "a state had no measurable configuration")
        return
    }
    expect(michaelis > 0, "the substrate's fructose is β: \(michaelis)")
    expect(covalent < 0, "glycosylation must invert it to α: \(covalent)")
    expect(hydrolysis < 0, "it is still α while the water comes in: \(hydrolysis)")
    expect(product > 0, "deglycosylation must invert it back to β: \(product)")
    expect(michaelis * product > 0, "net retention: the product must be the same hand as the substrate")
    expect(michaelis * covalent < 0, "the intermediate must be the other hand")
}

test("the configuration is measured from coordinates, not copied from the file") {
    // The builder wrote its own answer into the JSON. Recomputing it here from
    // the coordinates alone has to agree, or one of the two is guessing.
    for (name, recorded) in scene.anomericVolumes {
        guard let measured = anomericConfiguration(scene, state: name) else { continue }
        expect(abs(Double(measured) - recorded) < 1e-2,
               "\(name): measured \(measured), the builder recorded \(recorded)")
    }
}

test("the nucleophile attacks in line with the bond that breaks") {
    // Not aimed at: this angle is whatever the superposition produced.
    expect(scene.attackAngle > 150,
           "an SN2-like displacement wants something near 180°, got \(scene.attackAngle)°")
    expect(scene.attackDistance > 2.4 && scene.attackDistance < 3.4,
           "a poised nucleophile sits about 3 Å off, got \(scene.attackDistance) Å")
    expect(scene.acidDistance < 3.2,
           "the acid must be within hydrogen-bonding reach of the oxygen it protonates, "
           + "got \(scene.acidDistance) Å")
}

// MARK: -

section("atoms are conserved")

test("sucrose plus water equals glucose plus fructose, element by element") {
    let left: Formula = formula(sucrose) + waterFormula
    let right: Formula = formula(glucose) + formula(fructose)
    expectEqual(left, right)
    expectEqual(left.text, "C12H24O12")
    expectEqual(formula(sucrose).text, "C12H22O11")
}

test("the drawn heavy atoms balance too") {
    let michaelis = heavyAtomCount(scene, state: "michaelis")
    let product = heavyAtomCount(scene, state: "product")
    let covalent = heavyAtomCount(scene, state: "covalent")
    let hydrolysis = heavyAtomCount(scene, state: "hydrolysis")
    expectEqual(michaelis, formula(sucrose).heavyAtoms)
    // Glycosylation creates nothing: the bridging oxygen simply goes with the
    // glucose and becomes its own O1, so the count across the site is unchanged.
    expectEqual(covalent, formula(sucrose).heavyAtoms)
    // The glucose has gone and the water has arrived: eleven sugar atoms, one
    // water oxygen, which is what a whole fructose will weigh out to.
    expectEqual(hydrolysis, formula(fructose).heavyAtoms - 1 + 1)
    expectEqual(product, formula(fructose).heavyAtoms)
    expect(scene.states["free"]?.fructosyl.isEmpty == true, "free enzyme carries nothing")
}

test("the molar masses come from the formulas") {
    expect(abs(sucrose.molarMass - 342.30) < 0.01, "\(sucrose.molarMass)")
    expect(abs(glucose.molarMass - 180.16) < 0.01, "\(glucose.molarMass)")
    expectEqual(glucose.molarMass, fructose.molarMass)
    expect(abs(sucrose.molarMass + waterMolarMass - 2 * glucose.molarMass) < 1e-9,
           "sucrose + water must weigh what the two hexoses weigh")
}

// MARK: -

section("the covalent intermediate is real")

test("the ester bond is drawn only where it is the right length, and nowhere else") {
    var bondedFrames = 0
    for f in 0..<frameCount {
        let b = beat(at: Double(f) / Double(frameCount))
        if b.stage != .reaction { continue }
        let position = cyclePosition(b.cycleU)
        let site = siteGeometry(scene, at: b.cycleU, scale: 1)
        if let length = site.esterLength {
            bondedFrames += 1
            expect(length >= 1.30 && length <= 1.60,
                   "frame \(f) draws a \(length) Å bond to the nucleophile")
            expect(position.from == "covalent" || position.from == "hydrolysis"
                   || position.from == "michaelis",
                   "frame \(f) is in \(position.from) and should not be esterified")
        } else if position.from == "covalent" {
            expect(false, "frame \(f) is the covalent intermediate and has no bond")
        }
    }
    expect(bondedFrames > 8, "only \(bondedFrames) frames show the intermediate")
}

test("nothing is bonded to the nucleophile when the substrate is merely bound") {
    guard let michaelis = scene.states["michaelis"]?.fructosylC2,
          let product = scene.states["product"]?.fructosylC2 else {
        expect(false, "missing a state")
        return
    }
    let bound: Float = simd_distance(michaelis, scene.nucleophileOxygen)
    let gone: Float = simd_distance(product, scene.nucleophileOxygen)
    expect(bound > 2.0, "the Michaelis complex is not covalent: \(bound) Å")
    expect(gone > 2.0, "the product has left: \(gone) Å")
    expect(scene.esterLength >= 1.30 && scene.esterLength <= 1.60,
           "the builder's ester is \(scene.esterLength) Å")
}

// MARK: -

section("what the bar has to say")

test("the resolution is on the bar in every frame") {
    // 3.40 Å is coarse for a surface made of side chains, and the render will
    // look as confident as step 9's 1.9 Å one. It has to say so.
    let wanted = String(format: "%.2f Å", scene.resolution)
    for f in 0..<frameCount {
        let b = beat(at: Double(f) / Double(frameCount))
        let (prims, readout) = polarimeterScene(progress: b.progress, fill: b.fill,
                                                molecules: cellMolecules,
                                                lightColour: SIMD3(1, 0.7, 0.2))
        _ = prims
        let text = caption(scene, b, readout: readout, invertRotation: -19.85).text
        expect(text.contains(wanted), "frame \(f) does not say \(wanted)")
        expect(text.contains(scene.pdb), "frame \(f) does not name \(scene.pdb)")
        expect(text.contains(scene.organism), "frame \(f) does not name the organism")
    }
}

test("both rotations are on the bar") {
    let b = beat(at: 0.5)
    let (_, readout) = polarimeterScene(progress: b.progress, fill: b.fill,
                                        molecules: cellMolecules, lightColour: SIMD3(1, 0.7, 0.2))
    let text = caption(scene, b, readout: readout, invertRotation: -19.85).text
    expect(text.contains("+66.5"), "sucrose's rotation is missing")
    expect(text.contains("-19.9") || text.contains("−19.9"), "invert sugar's rotation is missing")
    expect(text.contains("RETAINING"), "the bar must say it is a retaining enzyme")
}

test("every frame with sucrose in the site is labelled MODEL") {
    for f in 0..<frameCount {
        let b = beat(at: Double(f) / Double(frameCount))
        guard b.stage == .reaction else { continue }
        let site = siteGeometry(scene, at: b.cycleU, scale: 1)
        guard site.glucoseFraction > 0.05 || site.fructoseFraction > 0.05 else { continue }
        let (_, readout) = polarimeterScene(progress: b.progress, fill: b.fill,
                                            molecules: cellMolecules,
                                            lightColour: SIMD3(1, 0.7, 0.2))
        let text = caption(scene, b, readout: readout, invertRotation: -19.85).text
        expect(text.contains("MODEL"), "frame \(f) shows a docked pose and does not say MODEL")
        expect(text.contains(scene.template.pdb), "frame \(f) does not name the template")
    }
}

test("the pose's entry in the table is MODEL, and says why") {
    let table = structureConstants(scene)
    guard let pose = table.first(where: { $0.name == "sucrose pose" }) else {
        expect(false, "no entry for the pose")
        return
    }
    expectEqual(pose.evidence, .model)
    expect(pose.source.contains("no structure of sucrose bound to invertase exists"),
           "the entry must say what is missing")
    expect(pose.source.contains(scene.template.pdb), "the entry must name the template")
}

// MARK: -

section("the porthole is a real cut")

test("nothing in the frame is a backface, opening or closed") {
    // Nothing is clipped — atoms in the window shrink instead — so every
    // surface is a whole sphere or a whole cylinder with its normal pointing
    // out. This is what that buys, checked by casting rays rather than assumed.
    for u in [0.30, 0.50, 0.70] {
        let b = beat(at: u)
        let prims = moleculeScene(scene, beat: b)
        let camera = moleculeCamera(b, scene: scene)
        var hits = 0, backfaces = 0
        for j in stride(from: 0, to: 620, by: 17) {
            for i in stride(from: 0, to: 640, by: 17) {
                let d = camera.rayDirection(x: i, y: j, width: 640, height: 620)
                guard let hit = intersect(prims, origin: camera.origin, direction: d) else { continue }
                hits += 1
                if simd_dot(hit.normal, d) > 0 { backfaces += 1 }
            }
        }
        expect(hits > 400, "at u=\(u) only \(hits) rays hit anything")
        expectEqual(backfaces, 0)
    }
}

test("the porthole really opens, and the rim really survives") {
    let b = beat(at: 0.5)
    expect(b.opening > 0.95, "the porthole should be wide open at the middle of the loop")
    let axis = scene.siteAxis
    let helper: SIMD3<Float> = abs(axis.y) > 0.9 ? SIMD3(1, 0, 0) : SIMD3(0, 1, 0)
    let right: SIMD3<Float> = simd_normalize(simd_cross(axis, helper))
    let up: SIMD3<Float> = simd_cross(axis, right)
    let origin: SIMD3<Float> = scene.siteCentre + axis * dimerDistance

    func reaches(opening: Float, radiusFraction: Float, rays: Int) -> Int {
        var open = 0
        let scene2 = beat(at: 0.5)
        var b2 = scene2
        b2.opening = opening
        let prims = moleculeScene(scene, beat: b2)
        for k in 0..<rays {
            let angle: Float = Float(k) / Float(rays) * 2 * .pi
            let offset: Float = portholeRadius * radiusFraction
            let target: SIMD3<Float> = scene.siteCentre
                + right * (offset * cos(angle)) + up * (offset * sin(angle))
            let d: SIMD3<Float> = simd_normalize(target - origin)
            guard let hit = intersect(prims, origin: origin, direction: d) else {
                open += 1
                continue
            }
            // Did the ray get past the plane of the site?
            let p: SIMD3<Float> = origin + d * hit.t
            if simd_dot(p - scene.siteCentre, axis) < 2.0 { open += 1 }
        }
        return open
    }

    let throughOpen = reaches(opening: 1.0, radiusFraction: 0.45, rays: 48)
    let throughShut = reaches(opening: 0.0, radiusFraction: 0.45, rays: 48)
    expect(throughOpen > 36, "only \(throughOpen) of 48 rays see into the open site")
    // A third of these rays get in with the porthole shut, and that is not a
    // bug: a GH32 site sits at the bottom of the propeller's funnel and is
    // genuinely open to solvent. The porthole widens a hole that is already
    // there. What it must do is at least double it, or it is not doing anything.
    expect(throughOpen >= 2 * throughShut,
           "the porthole took \(throughShut) rays to \(throughOpen): it is barely cutting")

    // Just outside the window and its 2.6 Å soft edge, the wall must be whole.
    // Further out than about 1.3 porthole radii there is no propeller face left
    // to be whole — the funnel's mouth is only so wide — so testing out there
    // measures the edge of the molecule rather than the edge of the cut.
    let overRim = reaches(opening: 1.0, radiusFraction: 1.25, rays: 48)
    expect(overRim < 8, "the rim is not intact: \(overRim) of 48 rays went straight through it")
}

// MARK: -

section("the loop closes")

test("the catalytic cycle returns to free enzyme, atom for atom") {
    let startOfCycle = cyclePosition(0)
    let endOfCycle = cyclePosition(0.99999)
    expectEqual(startOfCycle.from, "free")
    expectEqual(endOfCycle.to, "free")
    // At either end of a turnover there is nothing in the site, and the two
    // states are the same object, not two that happen to look alike.
    let a = siteGeometry(scene, at: 0.0, scale: 1)
    let b = siteGeometry(scene, at: 0.99999, scale: 1)
    expectEqual(a.spheres.count, 0)
    expectEqual(b.spheres.count, 0)
    expectEqual(a.sticks.count, 0)
    expectEqual(b.sticks.count, 0)
    expect(scene.states["free"]?.glucosyl.isEmpty == true, "free enzyme carries no glucose")
    expect(scene.states["free"]?.water == nil, "free enzyme carries no water")
    expect(scene.states["free"]?.esterBond == false, "free enzyme is not esterified")
}

test("the loop's two ends are the same frame") {
    let first = beat(at: 0)
    let last = beat(at: 1)
    expect(abs(first.fill - last.fill) < 1e-9, "the cell is not equally full")
    expect(abs(first.progress - last.progress) < 1e-9, "the reaction is not equally far along")
    expectEqual(first.zoom, last.zoom)
    expectEqual(first.opening, last.opening)
    // The camera closes because the yaw goes exactly once round.
    let a = moleculeCamera(first, scene: scene)
    let b = moleculeCamera(last, scene: scene)
    expect(simd_distance(a.origin, b.origin) < 0.05,
           "the camera moved \(simd_distance(a.origin, b.origin)) Å across the seam")
    // And the reading closes because the drain starts from where the reaction
    // ended. It cannot land exactly: the cell holds a whole number of molecules
    // and one of them is worth about half a degree. So the test is that the
    // seam is no bigger a step than any other step in the loop — that nothing
    // happens there which does not happen everywhere.
    let seam = abs(samples[0].readout.rotationDegrees
                   - observedRotation(samples[frameCount - 1].readout.composition, fill: 1))
    var biggest: Double = 0
    for i in 1..<reaction.count {
        biggest = max(biggest, abs(reaction[i].readout.rotationDegrees
                                   - reaction[i - 1].readout.rotationDegrees))
    }
    expect(seam <= biggest + 1e-9,
           "the seam steps \(seam)°, larger than the biggest step elsewhere, \(biggest)°")
    let ends = rotationAtRest(molecules: cellMolecules)
    let perMolecule: Double = (ends.start - ends.end) / Double(cellMolecules)
    expect(perMolecule < 0.6, "one molecule in the cell is worth \(perMolecule)°, too coarse")
    expect(biggest <= 2.2 * perMolecule,
           "some frame jumped \(biggest)°, more than a couple of molecules' worth")
}

test("every turnover starts and ends at free enzyme") {
    for k in 0..<turnoversPerLoop {
        let atStart = cyclePosition(0.0)
        _ = atStart
        let progress: Double = Double(k) / Double(turnoversPerLoop)
        let b = beat(at: refillEnds + (1 - refillEnds) * progress + 1e-9)
        expect(b.cycleU < 0.02, "turnover \(k) does not start at the top of the cycle: \(b.cycleU)")
        let site = siteGeometry(scene, at: b.cycleU, scale: 1)
        expectEqual(site.spheres.count, 0)
    }
}

// MARK: -

section("every constant carries a source")

test("nothing is stated without saying where it came from") {
    let table = chemistryConstants(molecules: cellMolecules) + structureConstants(scene)
    expect(table.count >= 20, "only \(table.count) constants in the table")
    for c in table {
        expect(!c.source.isEmpty, "\(c.name) has no source")
        expect(c.source.count > 20, "\(c.name)'s source is too short to be one: \(c.source)")
        expect(!c.unit.isEmpty, "\(c.name) has no unit")
    }
}

test("anything claiming to be MEASURED cites a year") {
    let table = chemistryConstants(molecules: cellMolecules) + structureConstants(scene)
    var measured = 0
    for c in table where c.evidence == .measured {
        measured += 1
        var found = false
        let characters = Array(c.source)
        for i in 0..<max(characters.count - 3, 0) {
            let four = String(characters[i..<(i + 4)])
            if let year = Int(four), year > 1800, year < 2100 { found = true }
        }
        expect(found, "\(c.name) is MEASURED and names no year: \(c.source)")
    }
    expect(measured >= 6, "only \(measured) measured constants")
}

test("the three evidence levels are all used, and the soft ones are honest") {
    let table = chemistryConstants(molecules: cellMolecules) + structureConstants(scene)
    let levels = Set(table.map { $0.evidence })
    expectEqual(levels.count, 3)
    for c in table where c.evidence == .model {
        expect(c.source.contains("MODEL") || c.source.contains("chosen here"),
               "\(c.name) is a model and does not admit it: \(c.source)")
    }
}

// MARK: -

section("the structure was read rather than assumed")

test("the crystal numbers one behind the paper, and that was checked") {
    expectEqual(scene.catalytic.nucleophilePaper - scene.catalytic.nucleophilePDB, 1)
    expectEqual(scene.catalytic.acidBasePaper - scene.catalytic.acidBasePDB, 1)
    expectEqual(scene.catalytic.stabiliserPaper - scene.catalytic.stabiliserPDB, 1)
    expectEqual(scene.catalytic.nucleophilePaper, 23)
    expectEqual(scene.catalytic.acidBasePaper, 204)
    // Taking Asp23 into the coordinates unchecked lands on a proline.
    let atTwentyThree = scene.atoms.first { $0.chain == "A" && $0.seq == 23 }
    expectEqual(atTwentyThree?.res, "PRO")
    let nucleophile = scene.atoms.first {
        $0.chain == "A" && $0.seq == scene.catalytic.nucleophilePDB
    }
    expectEqual(nucleophile?.res, "ASP")
    let acid = scene.atoms.first { $0.chain == "A" && $0.seq == scene.catalytic.acidBasePDB }
    expectEqual(acid?.res, "GLU")
}

test("the octamer is in the file and the dimer is found by counting contacts") {
    expectEqual(scene.chains.count, 8)
    expectEqual(scene.dimerChains.count, 2)
    expect(scene.dimerChains.contains("A"), "the site's own chain must be in the dimer")
    expect(scene.dimerAtomIndices.count > 8000 && scene.dimerAtomIndices.count < scene.atoms.count,
           "the dimer is \(scene.dimerAtomIndices.count) of \(scene.atoms.count) atoms")
    expect(scene.atoms.count > 32000, "\(scene.atoms.count) atoms")
}

test("the template is a close relative and the transfer is tight") {
    expect(scene.template.identityPercent > 40,
           "\(scene.template.identityPercent)% is too far to transfer a pose")
    expect(scene.template.siteRMSD < 0.6,
           "the active sites superpose at \(scene.template.siteRMSD) Å")
    expect(scene.template.siteRMSD < scene.template.wholeChainRMSD,
           "the site should superpose better than the whole chain")
    expect(scene.template.resolution < scene.resolution,
           "the template should be sharper than the enzyme, or there is no point")
}

test("the six chains that are not the dimer fade out rather than being cut away") {
    let wide = survivalFactors(scene, zoom: 0, opening: 0)
    let close = survivalFactors(scene, zoom: 1, opening: 0)
    let dimer = Set(scene.dimerChains)
    for (i, a) in scene.atoms.enumerated() {
        if dimer.contains(a.chain) {
            expectEqual(close[i], 1)
        } else {
            expectEqual(wide[i], 1)
            expectEqual(close[i], 0)
        }
    }
}

_ = finish()
