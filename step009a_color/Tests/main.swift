// Tests for step 9a.
//
// Two groups. The structural ones re-read the deposited PDB files rather than
// the JSON this step builds from them, so a mistake in the build tool cannot
// hide behind its own output. The colour ones check the wavelength-to-sRGB
// conversion against points whose answers are known independently.
//
// The claim the whole step rests on — that mCherry's chromophore carries a
// longer conjugated system, and that this is why it emits red — is checked
// where it is checkable: in the bond lengths.

import CoreGraphics
import Foundation
import Metal
import simd

let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
let pair = try loadPair(from: root.appendingPathComponent("Resources/pair.json"))

func near(_ a: Float, _ b: Float, within e: Float) -> Bool { abs(a - b) <= e }

/// Non-water atoms in the first conformer of a deposited file, read here
/// independently of Tools/build_pair.py.
func readPDB(_ name: String) -> [(name: String, res: String, seq: Int, el: String, pos: SIMD3<Float>)] {
    let text = try! String(contentsOf: root.appendingPathComponent("Resources/\(name).pdb"), encoding: .utf8)
    var out: [(name: String, res: String, seq: Int, el: String, pos: SIMD3<Float>)] = []
    for line in text.split(separator: "\n", omittingEmptySubsequences: false) {
        let l = Array(line)
        guard l.count > 78 else { continue }
        let tag = String(l[0..<6])
        guard tag == "ATOM  " || tag == "HETATM" else { continue }
        let res = String(l[17..<20]).trimmingCharacters(in: .whitespaces)
        guard res != "HOH" else { continue }
        let alt = l[16]
        guard alt == " " || alt == "A" else { continue }
        let x = Float(String(l[30..<38]).trimmingCharacters(in: .whitespaces)) ?? 0
        let y = Float(String(l[38..<46]).trimmingCharacters(in: .whitespaces)) ?? 0
        let z = Float(String(l[46..<54]).trimmingCharacters(in: .whitespaces)) ?? 0
        out.append((String(l[12..<16]).trimmingCharacters(in: .whitespaces),
                    res,
                    Int(String(l[22..<26]).trimmingCharacters(in: .whitespaces)) ?? 0,
                    String(l[76..<78]).trimmingCharacters(in: .whitespaces).uppercased(),
                    SIMD3(x, y, z)))
    }
    return out
}

section("the two structures, read straight from the PDB files")
test("atom counts match the deposited files, waters and second conformers excluded") {
    for (p, file) in [(pair.gfp, "1EMA"), (pair.mCherry, "2H5Q")] {
        expectEqual(p.atoms.count, readPDB(file).count)
    }
    // Their sizes are close enough that the comparison is fair rather than
    // one barrel dwarfing the other.
    let ratio = Float(pair.mCherry.atoms.count) / Float(pair.gfp.atoms.count)
    expect(ratio > 0.9 && ratio < 1.1, "sizes differ by \(ratio)")
}
test("each chromophore is one residue, and two residue numbers are missing where it formed") {
    for (p, file, name) in [(pair.gfp, "1EMA", "CRO"), (pair.mCherry, "2H5Q", "CH6")] {
        expectEqual(p.chromophoreName, name)
        let atoms = readPDB(file)
        let seqs = Set(atoms.map { $0.seq })
        let range = seqs.min()!...seqs.max()!
        let missing = range.filter { !seqs.contains($0) }.sorted()
        expectEqual(missing, p.missingResidues)
        expect(missing.count == 2, "three residues fuse into one, so two numbers go missing")
        expectEqual(atoms.filter { $0.res == name }.count, p.chromophoreAtomCount)
    }
    // The two are NOT at the same place in the sequence, which is why this is
    // read from the files rather than assumed to match.
    expect(pair.gfp.missingResidues != pair.mCherry.missingResidues,
           "GFP loses 65 and 67; mCherry loses 67 and 68")
}
test("GFP is the selenomethionine structure and mCherry is not") {
    expectEqual(pair.gfp.selenomethionines.count, 4)
    expectEqual(pair.mCherry.selenomethionines.count, 0)
    // Selenium is a phasing trick, not biology, but it still needs a radius.
    expect(pair.gfp.atoms.contains { $0.element == "SE" })
    expect(vdwRadius("SE") > vdwRadius("S"))
}
test("the chromophores come from different amino acids, and that is not the colour difference") {
    expectEqual(pair.gfp.formedFrom, ["THR", "TYR", "GLY"])
    expectEqual(pair.mCherry.formedFrom, ["MET", "TYR", "GLY"])
    // mCherry's chromophore has one more atom, and it is tempting to call that
    // the cause. It is not: it is the methionine side chain, which is not
    // conjugated. Assert the spare atom is outside the pi system.
    expectEqual(pair.mCherry.chromophoreAtomCount - pair.gfp.chromophoreAtomCount, 1)
    let extraNames = Set(pair.mCherry.extraPi)
    expect(!extraNames.contains("SD") && !extraNames.contains("CE"),
           "the methionine sulfur and its methyl must not be in the pi system")
}

section("the acylimine: the bond the whole step rests on")
test("N1–CA1 is a single bond in GFP and a double bond in mCherry") {
    // Textbook lengths: C–N single 1.47 Å, C=N double 1.28 Å (Allen et al.,
    // J. Chem. Soc. Perkin Trans. 2, 1987). Measured here from the deposited
    // coordinates of each file, not from this step's own JSON.
    for (file, chromo, expected, tolerance) in [("1EMA", "CRO", Float(1.47), Float(0.06)),
                                                ("2H5Q", "CH6", Float(1.28), Float(0.06))] {
        let atoms = readPDB(file).filter { $0.res == chromo }
        let n1 = atoms.first { $0.name == "N1" }!
        let ca1 = atoms.first { $0.name == "CA1" }!
        let d = simd_distance(n1.pos, ca1.pos)
        expect(near(d, expected, within: tolerance), "\(file) N1–CA1 = \(d), expected \(expected)")
    }
}
test("the two measured lengths differ by far more than crystallographic error") {
    let gap = pair.gfp.acylimineLength - pair.mCherry.acylimineLength
    // 1EMA is 1.9 Å and 2H5Q is 1.36 Å; coordinate error at those resolutions
    // is well under 0.05 Å for ordered atoms, so a 0.17 Å gap is real.
    expect(gap > 0.12, "difference is only \(gap) Å, too small to call a bond order change")
}
test("mCherry's conjugated system is the longer one, by exactly the acylimine") {
    expect(pair.mCherry.piAtoms.count > pair.gfp.piAtoms.count)
    expectEqual(pair.mCherry.piAtoms.count - pair.gfp.piAtoms.count, 2)
    expectEqual(Set(pair.mCherry.extraPi), Set(["CA1", "N1"]))
    expect(pair.gfp.extraPi.isEmpty, "GFP's pi system is the shared one, with nothing extra")
}
test("each conjugated path is actually connected, not a scattering of atoms") {
    for p in pair.both {
        let inPi = Set(p.piAtoms)
        // Walk the pi bonds and check every pi atom is reachable from the first.
        var neighbours: [Int: [Int]] = [:]
        for b in p.piBonds where inPi.contains(b.a) && inPi.contains(b.b) {
            neighbours[b.a, default: []].append(b.b)
            neighbours[b.b, default: []].append(b.a)
        }
        var seen = Set([p.piAtoms[0]])
        var queue = [p.piAtoms[0]]
        while let n = queue.popLast() {
            for m in neighbours[n] ?? [] where !seen.contains(m) {
                seen.insert(m)
                queue.append(m)
            }
        }
        expect(seen.count == p.piAtoms.count, "\(p.label)'s pi system is in pieces")
    }
}
test("the longer conjugated system is the one that emits the longer wavelength") {
    // This is the physics the render claims: more conjugation, smaller gap,
    // redder light. Both halves are measured independently — the atoms from
    // crystallography, the wavelengths from FPbase.
    let longer = pair.mCherry.piAtoms.count > pair.gfp.piAtoms.count ? pair.mCherry : pair.gfp
    let redder = pair.mCherry.emission > pair.gfp.emission ? pair.mCherry : pair.gfp
    expectEqual(longer.key, redder.key)
    expect(pair.gfp.photonEnergyEV > pair.mCherry.photonEnergyEV, "greener light is higher energy")
    expect(near(Float(pair.gfp.photonEnergyEV - pair.mCherry.photonEnergyEV), 0.399, within: 0.01))
}

section("the two folds are the same, so the comparison is fair")
test("both are barrels of the same size and shape") {
    for p in pair.both {
        expect(p.residueCount > 200 && p.residueCount < 240, "\(p.label): \(p.residueCount) residues")
        expect(p.wallRadius > 9 && p.wallRadius < 14, "\(p.label): wall radius \(p.wallRadius) Å")
    }
    expect(abs(pair.gfp.wallRadius - pair.mCherry.wallRadius) < 1.5, "barrels should be the same width")
}
test("the folds superpose: most of mCherry's backbone lies on GFP's") {
    // These two share only about a quarter of their sequence, so this measures
    // the fold rather than the sequence. Aligned only by barrel axis and
    // chromophore direction — no sequence alignment is used or needed.
    expect(pair.foldWithin3A > 0.5, "only \(pair.foldWithin3A) of CAs within 3 Å")
    expect(pair.foldMedian < 3.5, "median nearest-CA distance \(pair.foldMedian) Å")
}
test("both chromophores sit near the middle of their barrel") {
    for p in pair.both {
        expect(simd_length(p.chromophoreCenter) < 6, "\(p.label): \(simd_length(p.chromophoreCenter)) Å off centre")
    }
}

section("colour computed from wavelength")
test("the colour-matching functions peak where the eye does") {
    // The CIE curves peak near 600 (x̄), 555 (ȳ) and 445 nm (z̄).
    var best = (x: Float(0), y: Float(0), z: Float(0))
    var at = (x: Float(0), y: Float(0), z: Float(0))
    for nm in stride(from: Float(380), through: Float(700), by: 1) {
        let c = colourMatching(nanometres: nm)
        if c.x > best.x { best.x = c.x; at.x = nm }
        if c.y > best.y { best.y = c.y; at.y = nm }
        if c.z > best.z { best.z = c.z; at.z = nm }
    }
    expect(near(at.y, 555, within: 15), "ȳ peaks at \(at.y), expected ~555 nm")
    expect(near(at.z, 445, within: 15), "z̄ peaks at \(at.z), expected ~445 nm")
    expect(at.x > 580 && at.x < 615, "x̄'s main peak is at \(at.x), expected ~600 nm")
}
test("known wavelengths come out the right colour") {
    // The mercury lines are convenient because their colours are not a matter
    // of opinion. 700 nm is deliberately NOT used: the analytic fit is weakest
    // in the far tails, where the eye's response is almost nothing.
    let green = spectralRGB8(nanometres: 546.1)      // Hg green line
    expect(green.g > 200 && green.r < 60 && green.b < 60, "546 nm gave \(green)")
    let blue = spectralRGB8(nanometres: 435.8)       // Hg blue line
    expect(blue.b > 200 && blue.g < 60, "436 nm gave \(blue)")
    let red = spectralRGB8(nanometres: 650)
    expect(red.r > 200 && red.g < 60 && red.b < 60, "650 nm gave \(red)")
}
test("hue moves steadily toward red as the wavelength grows") {
    var lastRatio: Float = -1
    for nm in stride(from: Float(500), through: Float(640), by: 20) {
        let c = spectralColour(nanometres: nm).linear
        let ratio: Float = c.x / max(c.y, 1e-4)      // red against green
        expect(ratio >= lastRatio - 1e-3, "at \(nm) nm the red/green ratio fell")
        lastRatio = ratio
    }
}
test("every pure wavelength is outside what a screen can show") {
    // A monochromatic colour always lies outside the sRGB triangle. Both of
    // these proteins clip, which is why a photograph of one never looks as
    // vivid as the eye reports.
    for p in pair.both {
        expect(p.emissionClipped, "\(p.label) at \(p.emission) nm should be out of gamut")
    }
    expect(spectralColour(nanometres: 546.1).clipped)
}
test("the two proteins get different colours, and neither is hard-coded") {
    let g = pair.gfp.emissionColour, m = pair.mCherry.emissionColour
    expect(simd_distance(g, m) > 0.5, "the two colours are too close: \(g) vs \(m)")
    expect(g.y > g.x && g.y > g.z, "GFP's computed colour should be green-dominant, got \(g)")
    expect(m.x > m.y && m.x > m.z, "mCherry's computed colour should be red-dominant, got \(m)")
    // Recomputing from the wavelength must give the same answer: if a literal
    // had been substituted anywhere, these would drift apart.
    expectEqual(g, spectralColour(nanometres: pair.gfp.emission).linear)
    expectEqual(m, spectralColour(nanometres: pair.mCherry.emission).linear)
}
test("the spectra are the ones for these exact variants") {
    // 1EMA is the S65T variant, whose single 490 nm excitation peak is not
    // wild-type avGFP's 395/475 pair. Using the wild-type numbers would be the
    // easy mistake here.
    expectEqual(pair.gfp.excitation, 490)
    expectEqual(pair.gfp.emission, 510)
    expectEqual(pair.mCherry.excitation, 587)
    expectEqual(pair.mCherry.emission, 610)
    for p in pair.both {
        expect(p.emission > p.excitation, "\(p.label): emission must be redshifted from excitation")
    }
}

section("the cutaway and the loop")
test("both portholes open and shut together, so the two are always alike") {
    for u in stride(from: 0.0, to: 1.0, by: 0.01) {
        let open = opening(at: u)
        expect(open >= 0 && open <= 1, "opening out of range at \(u)")
    }
    expectEqual(opening(at: 0), 0)
    expectEqual(opening(at: 0.5), 1)
    expectEqual(opening(at: 0.999), 0)
}
test("the opening never jumps between frames") {
    let frames = 200
    var previous = opening(at: 0)
    for f in 1...frames {
        let now = opening(at: Double(f % frames) / Double(frames))
        expect(abs(now - previous) < 0.2, "jump of \(abs(now - previous)) at frame \(f)")
        previous = now
    }
}
test("shut, the chromophore is hidden; open, a ray reaches it") {
    let camera = Camera.orbit(target: SIMD3(0, 5.5, 0), distance: 134, yaw: 0, pitch: 6, fov: 30)
    for p in pair.both {
        // Trace from the camera to the chromophore and see what is in the way.
        let target = p.chromophoreCenter
        let dir = simd_normalize(target - camera.origin)
        func blocked(_ cut: [Float]) -> Bool {
            let limit: Float = simd_distance(camera.origin, target) - 2
            for (i, atom) in p.atoms.enumerated() {
                if atom.isChromophore { continue }
                let r: Float = vdwRadius(atom.element) * cut[i]
                if r < 0.05 { continue }
                let oc = atom.position - camera.origin
                let along: Float = simd_dot(oc, dir)
                if along <= 0 || along > limit { continue }
                let perp: Float = simd_length(oc - dir * along)
                if perp < r { return true }
            }
            return false
        }
        expect(blocked(cutFactors(p, opening: 0, camera: camera)),
               "\(p.label): the barrel should hide its chromophore when shut")
        expect(!blocked(cutFactors(p, opening: 1, camera: camera)),
               "\(p.label): the open porthole should give a clear line to the chromophore")
    }
}
test("the porthole is a window, not a half-space: most of the barrel survives") {
    let camera = Camera.orbit(target: SIMD3(0, 5.5, 0), distance: 134, yaw: 0, pitch: 6, fov: 30)
    for p in pair.both {
        let cut = cutFactors(p, opening: 1, camera: camera)
        let removed = Float(cut.filter { $0 < 0.5 }.count) / Float(cut.count)
        expect(removed > 0.04 && removed < 0.30, "\(p.label) removed \(removed) of its atoms")
    }
}
test("the chromophore itself is never cut away") {
    let camera = Camera.orbit(target: SIMD3(0, 5.5, 0), distance: 134, yaw: 0, pitch: 6, fov: 30)
    for p in pair.both {
        let cut = cutFactors(p, opening: 1, camera: camera)
        for i in p.chromophoreAtoms { expectEqual(cut[i], 1) }
    }
}
test("every atom stays inside its panel all the way round") {
    for (column, p) in pair.both.enumerated() {
        for f in stride(from: 0, to: 200, by: 5) {
            let yaw = Float(360.0 * Double(f) / 200.0)
            let camera = Camera.orbit(target: SIMD3(0, 5.5, 0), distance: 134, yaw: yaw, pitch: 6, fov: 30)
            for atom in p.atoms {
                let s = camera.project(atom.position, width: 480, height: 520)
                let margin: Float = vdwRadius(atom.element) * 4
                expect(s.x > margin && s.x < 480 - margin && s.y > margin && s.y < 520 - margin,
                       "\(p.label) atom at \(s) in panel \(column), frame \(f)")
            }
        }
    }
}

section("the scene handed to the GPU")
test("shut, the barrel is all spheres and no sticks; open, the chromophore adds both") {
    let camera = Camera.orbit(target: SIMD3(0, 5.5, 0), distance: 134, yaw: 0, pitch: 6, fov: 30)
    for p in pair.both {
        let shut = sceneGeometry(p, cut: cutFactors(p, opening: 0, camera: camera), showChromophore: false)
        expectEqual(shut.cylinders.count, 0)
        expectEqual(shut.spheres.count, p.atoms.count - p.chromophoreAtomCount)
        let open = sceneGeometry(p, cut: cutFactors(p, opening: 1, camera: camera), showChromophore: true)
        expect(open.cylinders.count > 10, "\(p.label): chromophore should have sticks")
    }
}
test("space-filling means van der Waals radii, so bonded neighbours overlap") {
    // The definition of the representation, and what using a covalent radius
    // by mistake would break.
    for p in pair.both {
        var checked = 0
        for bond in p.bonds.prefix(400) {
            let a = p.atoms[bond.a], b = p.atoms[bond.b]
            if a.isChromophore || b.isChromophore { continue }
            let d = simd_distance(a.position, b.position)
            expect(d < vdwRadius(a.element) + vdwRadius(b.element),
                   "\(p.label): \(a.name)–\(b.name) spheres do not touch")
            checked += 1
        }
        expect(checked > 100)
    }
}
test("the chromophore's conjugated atoms wear the emission colour and the rest do not") {
    let camera = Camera.orbit(target: SIMD3(0, 5.5, 0), distance: 134, yaw: 0, pitch: 6, fov: 30)
    for p in pair.both {
        let (spheres, _) = sceneGeometry(p, cut: cutFactors(p, opening: 1, camera: camera), showChromophore: true)
        let emit = p.emissionColour
        let matching = spheres.filter {
            near($0.color.x, emit.x, within: 0.01) && near($0.color.y, emit.y, within: 0.01)
                && near($0.color.z, emit.z, within: 0.01)
        }
        expectEqual(matching.count, p.piAtoms.count)
    }
}
test("the palette reserves room for both emission colours") {
    let grey = [RGB(128, 128, 128), RGB(64, 64, 64), RGB(200, 200, 200)]
    let palette = pairPalette(pair, samples: grey, count: 112)
    for p in pair.both {
        let want = spectralRGB8(nanometres: p.emission)
        var found = false
        for entry in palette {
            let dr: Int = abs(Int(entry.x) - want.r)
            let dg: Int = abs(Int(entry.y) - want.g)
            let db: Int = abs(Int(entry.z) - want.b)
            if dr < 30 && dg < 30 && db < 30 { found = true; break }
        }
        expect(found, "\(p.label)'s colour \(want) is not in the palette")
    }
}

section("renderer")
let device = try findDevice()
let renderer = try MoleculeRenderer(device: device)
let panelW = 240, panelH = 260
let buffer = device.makeBuffer(length: panelW * panelH * 4, options: .storageModeShared)!
func pixel(_ x: Int, _ y: Int) -> SIMD3<Int> {
    let p = buffer.contents().assumingMemoryBound(to: UInt8.self) + (y * panelW + x) * 4
    return SIMD3(Int(p[0]), Int(p[1]), Int(p[2]))
}
test("an empty scene is step 2's gradient") {
    let cam = Camera.orbit(target: .zero, distance: 134, yaw: 0, pitch: 0, fov: 30)
    try renderer.render(spheres: [], cylinders: [], camera: cam, into: buffer,
                        width: panelW, viewHeight: panelH)
    let top = pixel(panelW / 2, 0), bottom = pixel(panelW / 2, panelH - 1)
    expect(abs(top.x - 140) <= 4 && abs(top.y - 200) <= 4 && abs(top.z - 235) <= 4, "top \(top)")
    expect(abs(bottom.x - 8) <= 4 && abs(bottom.y - 40) <= 4 && abs(bottom.z - 90) <= 4, "bottom \(bottom)")
}
test("occlusion darkens a crevice and leaves an exposed bulge alone") {
    // A ring of spheres around a hollow: the inside should come out darker.
    var spheres: [GPUSphere] = []
    for i in 0..<16 {
        let a = Float(i) / 16 * 2 * Float.pi
        let p = SIMD3<Float>(cos(a) * 6, sin(a) * 6, 0)
        spheres.append(GPUSphere(centerRadius: SIMD4(p.x, p.y, p.z, 2.6),
                                 color: SIMD4(0.7, 0.7, 0.7, 1), glow: .zero))
    }
    let cam = Camera.orbit(target: .zero, distance: 40, yaw: 0, pitch: 0, fov: 30)
    let ao = AOSettings(probes: 12, distance: 8, strength: 1, only: true, contrast: 1)
    try renderer.render(spheres: spheres, cylinders: [], camera: cam, into: buffer,
                        width: panelW, viewHeight: panelH, ao: ao)
    // Sample where the ring's inner edge is, against its outer edge.
    var inner = 0, outer = 0, innerN = 0, outerN = 0
    for y in 0..<panelH {
        for x in 0..<panelW {
            let c = pixel(x, y)
            if c.x == c.y && c.y == c.z && c.x > 0 && c.x < 255 {
                let dx = Float(x - panelW / 2), dy = Float(y - panelH / 2)
                let r = (dx * dx + dy * dy).squareRoot()
                if r < 42 { inner += c.x; innerN += 1 } else if r > 58 { outer += c.x; outerN += 1 }
            }
        }
    }
    expect(innerN > 50 && outerN > 50, "not enough samples: \(innerN), \(outerN)")
    let insideMean = Float(inner) / Float(max(innerN, 1))
    let outsideMean = Float(outer) / Float(max(outerN, 1))
    expect(insideMean < outsideMean, "inside \(insideMean) should be darker than outside \(outsideMean)")
}
test("turning occlusion off flattens the picture measurably") {
    let cam = Camera.orbit(target: SIMD3(0, 5.5, 0), distance: 134, yaw: 0, pitch: 6, fov: 30)
    let p = pair.gfp
    let (spheres, cylinders) = sceneGeometry(p, cut: cutFactors(p, opening: 0, camera: cam),
                                             showChromophore: false)
    func spread(_ ao: AOSettings) throws -> Float {
        try renderer.render(spheres: spheres, cylinders: cylinders, camera: cam, into: buffer,
                            width: panelW, viewHeight: panelH, ao: ao)
        var values: [Float] = []
        for y in stride(from: 0, to: panelH, by: 3) {
            for x in stride(from: 0, to: panelW, by: 3) {
                let c = pixel(x, y)
                values.append(Float(c.x + c.y + c.z) / 3)
            }
        }
        let m = values.reduce(0, +) / Float(values.count)
        let v = values.map { ($0 - m) * ($0 - m) }.reduce(0, +) / Float(values.count)
        return v.squareRoot()
    }
    let withAO = try spread(AOSettings(probes: 12, distance: 8, strength: 1, only: false, contrast: 1.7))
    let without = try spread(AOSettings.off)
    expect(withAO > without, "occlusion should add contrast: \(withAO) vs \(without)")
}
test("occlusion is a property of the molecule, not of the image") {
    // Comparing the same atom from two camera angles measures nothing: after a
    // few degrees a different atom is frontmost at a given pixel. Rendering the
    // same view at two sizes is the test that works — anything keyed to the
    // molecule agrees, anything keyed to pixels does not.
    let cam = Camera.orbit(target: SIMD3(0, 5.5, 0), distance: 134, yaw: 30, pitch: 6, fov: 30)
    let p = pair.mCherry
    let (spheres, cylinders) = sceneGeometry(p, cut: cutFactors(p, opening: 0, camera: cam),
                                             showChromophore: false)
    let ao = AOSettings(probes: 12, distance: 8, strength: 1, only: true, contrast: 1)
    let small = device.makeBuffer(length: 120 * 130 * 4, options: .storageModeShared)!
    try renderer.render(spheres: spheres, cylinders: cylinders, camera: cam, into: buffer,
                        width: panelW, viewHeight: panelH, ao: ao)
    try renderer.render(spheres: spheres, cylinders: cylinders, camera: cam, into: small,
                        width: 120, viewHeight: 130, ao: ao)
    var total = 0, count = 0
    for y in 0..<130 {
        for x in 0..<120 {
            let a = small.contents().assumingMemoryBound(to: UInt8.self) + (y * 120 + x) * 4
            let b = buffer.contents().assumingMemoryBound(to: UInt8.self) + ((y * 2) * panelW + x * 2) * 4
            if a[0] > 2 && a[0] < 253 && b[0] > 2 && b[0] < 253 {
                total += abs(Int(a[0]) - Int(b[0]))
                count += 1
            }
        }
    }
    expect(count > 1000, "too few comparable pixels: \(count)")
    let mean = Float(total) / Float(max(count, 1))
    expect(mean < 12, "occlusion differs by \(mean) grey levels between image sizes")
}

finish()
