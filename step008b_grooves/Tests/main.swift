// Tests for step 8b.
//
// The claim this step makes is a measurement, so the tests make the
// measurement rather than reading back the number the builder wrote. The
// occupancy profile is recomputed here from the crystal's own atoms, and the
// groove widths are re-derived from the phosphate positions by scanning every
// cross-strand offset and taking the minima — which is what finds the grooves
// without being told where they are.

import Foundation
import Metal
import simd

let resources = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent().deletingLastPathComponent()
    .appendingPathComponent("Resources")
let (helix, crystal) = try loadDuplexes(from: resources.appendingPathComponent("grooves.json"))

// Bondi, J. Phys. Chem. 68:441 (1964).
let bondi: [String: Float] = ["C": 1.70, "N": 1.55, "O": 1.52, "P": 1.80, "H": 1.20]
// Covalent radii, Cordero et al., Dalton Trans. 2832 (2008) — what a *wrong*
// radius would look like, and roughly half the van der Waals value.
let covalent: [String: Float] = ["C": 0.76, "N": 0.71, "O": 0.66, "P": 1.07, "H": 0.31]

func near(_ a: Float, _ b: Float, within e: Float) -> Bool { abs(a - b) <= e }

// MARK: - the measurement the step rests on

section("occupancy: the core is solid, the outside is not")

/// The fraction of a cylindrical shell about the x axis that lies inside some
/// atom, by Monte Carlo — recomputed here rather than trusted.
func occupancy(_ atoms: [GAtom], shell: Int, samples: Int, seed: UInt64) -> Float {
    var state = seed
    func next() -> Float {           // xorshift, so the test is repeatable
        state ^= state << 13; state ^= state >> 7; state ^= state << 17
        return Float(state % 1_000_003) / 1_000_003.0
    }
    let xs = atoms.map { $0.position.x }
    let lo = (xs.min() ?? 0) + 3, hi = (xs.max() ?? 0) - 3
    let r0 = Float(shell), r1 = Float(shell + 1)
    var inside = 0
    for _ in 0..<samples {
        let x = lo + (hi - lo) * next()
        let theta = 2 * Float.pi * next()
        let r = (r0 * r0 + (r1 * r1 - r0 * r0) * next()).squareRoot()
        let p = SIMD3<Float>(x, r * cos(theta), r * sin(theta))
        for a in atoms {
            let rad = bondi[a.element] ?? 1.7
            if simd_length_squared(p - a.position) < rad * rad { inside += 1; break }
        }
    }
    return Float(inside) / Float(samples)
}

test("the core, out to 3 Å from the axis, is above 70% full — as dense as matter gets") {
    var core: Float = 0
    for shell in 0..<3 { core += occupancy(crystal.atoms, shell: shell, samples: 4000, seed: UInt64(shell + 1) &* 2654435761) }
    core /= 3
    expect(core > 0.70, "core measured \(core * 100)%")
    expect(core < 0.95, "core measured \(core * 100)%, implausibly high")
}

test("the rim, 9–11 Å out, is below 25% — that missing volume is the grooves") {
    var rim: Float = 0
    for shell in 9..<11 { rim += occupancy(crystal.atoms, shell: shell, samples: 4000, seed: UInt64(shell + 7) &* 2654435761) }
    rim /= 2
    expect(rim < 0.25, "rim measured \(rim * 100)%")
}

test("the profile falls monotonically from the core outwards, once past the bases") {
    let p = crystal.occupancy
    expect(p.count >= 11, "no stored profile")
    for i in 3..<10 {
        expect(p[i] >= p[i + 1] - 0.03, "shell \(i)–\(i+1) Å (\(p[i])) is not above \(i+1)–\(i+2) Å (\(p[i+1]))")
    }
}

test("the whole cylinder is well under half full, so most of it is groove") {
    let p = crystal.occupancy
    var weighted: Float = 0, total: Float = 0
    for i in 0..<10 {
        let area = Float((i + 1) * (i + 1) - i * i)
        weighted += p[i] * area
        total += area
    }
    let overall = weighted / total
    expect(overall < 0.5, "whole cylinder measured \(overall * 100)% full")
    expect(overall > 0.2, "whole cylinder measured \(overall * 100)%, implausibly low")
}

// MARK: - groove geometry

section("grooves, found by scanning rather than assumed")

test("the two grooves are where B-DNA says: one near 11.7 Å, one near 5.7 Å") {
    guard let major = crystal.grooveWidth["major"], let minor = crystal.grooveWidth["minor"] else {
        expect(false, "no groove measurements"); return
    }
    // Saenger, Principles of Nucleic Acid Structure (1984): B-DNA's major
    // groove is 11.7 Å wide and the minor 5.7 Å, by this same convention
    // (minimum cross-strand P···P less 5.8 Å for the phosphate's own bulk).
    expect(near(major, 11.7, within: 1.2), "major measured \(major) Å")
    expect(near(minor, 5.7, within: 1.2), "minor measured \(minor) Å")
    expect(major > minor + 3, "the major groove should be much the wider: \(major) vs \(minor)")
}

test("the raw P···P separations are wider than the widths by the phosphate allowance") {
    for name in ["major", "minor"] {
        guard let pp = crystal.groovePP[name], let w = crystal.grooveWidth[name] else { continue }
        expect(near(pp - w, 5.8, within: 0.01), "\(name): \(pp) − \(w) ≠ 5.8")
    }
}

test("the built helix has both grooves too, the major still much the wider") {
    guard let major = helix.grooveWidth["major"], let minor = helix.grooveWidth["minor"] else {
        expect(false, "no groove measurements on the helix"); return
    }
    expect(major > minor + 3, "major \(major) Å, minor \(minor) Å")
    // Idealised at 10.49 bp per turn, it is slightly more unwound than the
    // crystal, which opens the major groove. Wider, but recognisably B-DNA.
    expect(major > 10 && major < 16, "major measured \(major) Å")
    expect(minor > 4 && minor < 9, "minor measured \(minor) Å")
}

// MARK: - space-filling really is space-filling

section("representation")

test("every atom gets a van der Waals radius, not a covalent one") {
    for (element, r) in bondi {
        expect(near(vdw(element), r, within: 0.001), "\(element) has radius \(vdw(element)), expected \(r)")
        let cov = covalent[element] ?? 0
        expect(vdw(element) > cov * 1.5, "\(element)'s radius \(vdw(element)) looks covalent, not van der Waals")
    }
    for a in helix.atoms {
        expect(bondi[a.element] != nil, "no radius for element \(a.element)")
    }
}

test("bonded neighbours overlap — which is what makes it one solid") {
    var overlapping = 0, checked = 0
    for b in helix.bonds where b.a < helix.atoms.count && b.b < helix.atoms.count {
        let x = helix.atoms[b.a], y = helix.atoms[b.b]
        let d = simd_distance(x.position, y.position)
        let sum = vdw(x.element) + vdw(y.element)
        checked += 1
        if d < sum { overlapping += 1 }
    }
    expect(checked > 2000, "only \(checked) bonds")
    expect(overlapping == checked, "\(checked - overlapping) of \(checked) bonded pairs do not overlap")
}

test("the spheres the renderer builds carry the full radius; the comparison's balls do not") {
    let heavy = helix.atoms.filter { $0.element != "H" }
    let full = spaceFilling(helix)
    expectEqual(full.count, heavy.count)
    expect(full.count < helix.atoms.count, "hydrogens should not be drawn")
    for (i, s) in full.enumerated() {
        expect(near(s.centerRadius.w, vdw(heavy[i].element), within: 0.001), "sphere \(i) has the wrong radius")
    }
    let camera = Camera.orbitAboutX(target: .zero, distance: 62, roll: 0, lean: 5, fov: 30)
    let stick = ballAndStick(crystal, camera: camera)
    expect(stick.cylinders.count > 500, "ball-and-stick drew only \(stick.cylinders.count) sticks")
    let heavyCrystal = crystal.atoms.filter { $0.element != "H" }
    for (i, s) in stick.spheres.enumerated() {
        let expected = vdw(heavyCrystal[i].element) * ballScale
        expect(near(s.centerRadius.w, expected, within: 0.001), "ball \(i) has the wrong radius")
    }
    // The whole contrast: space-filling spheres are several times the balls.
    expect(ballScale < 0.5, "the comparison's balls are not distinguishably smaller")
}

test("atoms are coloured by which groove they line, not by element") {
    var seen: Set<String> = []
    for a in helix.atoms {
        let c = partColor(a.part)
        seen.insert("\(a.part.rawValue)")
        expect(c.x >= 0 && c.x <= 1 && c.y >= 0 && c.y <= 1 && c.z >= 0 && c.z <= 1, "colour out of range")
    }
    expect(seen.contains("backbone") && seen.contains("major") && seen.contains("minor"),
           "missing a part: \(seen.sorted())")
    // Two atoms of the same element on different edges must differ in colour,
    // which is exactly what element colouring could not do.
    expect(partColor(.major) != partColor(.minor), "the two groove edges share a colour")
    expect(partColor(.backbone) != partColor(.major), "backbone and major edge share a colour")
}

test("the edge assignment agrees with where the atoms actually are") {
    // Recomputed from geometry: each named edge atom should be nearer its own
    // groove's midline. Uses the built helix, whose strand and pair index are
    // known, and skips the two pairs at each end.
    var phosphorus: [Int: [Int: SIMD3<Float>]] = [:]
    for a in helix.atoms where a.name == "P" {
        phosphorus[a.pair, default: [:]][a.strand] = a.position
    }
    var majorMid: [SIMD3<Float>] = [], minorMid: [SIMD3<Float>] = []
    for (pair, byStrand) in phosphorus {
        if let p0 = byStrand[0], let p1 = phosphorus[pair + 3]?[1] { majorMid.append((p0 + p1) / 2) }
        if let p0 = byStrand[0], let p1 = phosphorus[pair - 4]?[1] { minorMid.append((p0 + p1) / 2) }
    }
    expect(majorMid.count > 20 && minorMid.count > 20, "too few groove midpoints")
    var agree = 0, disagree = 0
    for a in helix.atoms where a.part == .major || a.part == .minor {
        if a.pair < 2 || a.pair > 37 { continue }
        let dMajor = majorMid.map { simd_distance($0, a.position) }.min() ?? .infinity
        let dMinor = minorMid.map { simd_distance($0, a.position) }.min() ?? .infinity
        let nearer: Part = dMajor < dMinor ? .major : .minor
        if nearer == a.part { agree += 1 } else { disagree += 1 }
    }
    let fraction = Float(agree) / Float(agree + disagree)
    expect(fraction > 0.85, "only \(Int(fraction * 100))% of edge atoms are nearer their own groove")
}

// MARK: - the idealised helix

section("the 40-mer")

test("it is right-handed, like B-DNA") {
    var byPair: [Int: SIMD3<Float>] = [:]
    for a in helix.atoms where a.name == "C1'" {
        if byPair[a.pair] == nil { byPair[a.pair] = a.position }
    }
    let pairs = byPair.keys.sorted()
    expect(pairs.count > 30, "only \(pairs.count) pairs")
    var advance: Float = 0, steps = 0
    for i in 1..<pairs.count {
        let a = byPair[pairs[i - 1]]!, b = byPair[pairs[i]]!
        let t0 = atan2(a.z, a.y), t1 = atan2(b.z, b.y)
        var d = t1 - t0
        while d > .pi { d -= 2 * .pi }
        while d < -.pi { d += 2 * .pi }
        advance += d; steps += 1
    }
    let degPerStep = advance / Float(steps) * 180 / .pi
    expect(degPerStep > 0, "the helix turns \(degPerStep)° per step — left-handed")
    expect(near(degPerStep, 34.3, within: 2), "twist measured \(degPerStep)°/bp")
}

test("the rise is 3.4 Å a step, and the duplex is about 20 Å across") {
    var byPair: [Int: [Float]] = [:]
    for a in helix.atoms where !a.name.hasPrefix("H") {
        byPair[a.pair, default: []].append(a.position.x)
    }
    let pairs = byPair.keys.sorted()
    var rises: [Float] = []
    for i in 1..<pairs.count {
        let a = mean(byPair[pairs[i - 1]]!), b = mean(byPair[pairs[i]]!)
        rises.append(b - a)
    }
    expect(near(mean(rises), 3.4, within: 0.15), "rise measured \(mean(rises)) Å")
    // Two diameters, and they are different things. The textbook "20 Å" is
    // backbone to backbone, phosphorus across to phosphorus. The width a
    // space-filling picture actually shows is wider, because it includes the
    // outer van der Waals shell those atoms carry.
    var phosphorusRadius: [Float] = []
    for a in helix.atoms where a.name == "P" {
        phosphorusRadius.append(simd_length(SIMD2(a.position.y, a.position.z)))
    }
    expect(near(2 * mean(phosphorusRadius), 19, within: 2.5),
           "backbone-to-backbone \(2 * mean(phosphorusRadius)) Å")
    var maxR: Float = 0
    for a in helix.atoms where !a.name.hasPrefix("H") {
        maxR = max(maxR, simd_length(SIMD2(a.position.y, a.position.z)) + vdw(a.element))
    }
    expect(2 * maxR > 2 * mean(phosphorusRadius), "the van der Waals width should exceed backbone-to-backbone")
    expect(near(2 * maxR, 24, within: 3), "space-filling width \(2 * maxR) Å")
}

test("the backbone joins up: every O3'–P ester is a real bond length") {
    var long = 0, checked = 0
    for b in helix.bonds where b.a < helix.atoms.count && b.b < helix.atoms.count {
        let x = helix.atoms[b.a], y = helix.atoms[b.b]
        guard (x.name == "O3'" && y.name == "P") || (x.name == "P" && y.name == "O3'") else { continue }
        checked += 1
        let d = simd_distance(x.position, y.position)
        if !near(d, 1.60, within: 0.20) { long += 1 }
    }
    expect(checked >= 70, "only \(checked) ester bonds found")
    expect(long == 0, "\(long) of \(checked) O3'–P bonds are the wrong length")
}

test("base pairing survives from the crystal: every bond is a sane length") {
    var bad: [String] = []
    for b in helix.bonds where b.a < helix.atoms.count && b.b < helix.atoms.count {
        let x = helix.atoms[b.a], y = helix.atoms[b.b]
        let d = simd_distance(x.position, y.position)
        if d < 0.85 || d > 1.85 { bad.append("\(x.name)–\(y.name) \(d) Å") }
    }
    expect(bad.isEmpty, "\(bad.count) odd bonds, e.g. \(bad.prefix(3))")
}

test("the sequence is pGLO's, starting at GFP's start codon") {
    expect(helix.sequence.hasPrefix("ATG"), "starts \(helix.sequence.prefix(6))")
    expectEqual(helix.sequence.count, 40)
    expect(Set(helix.sequence).isSubset(of: Set("ACGT")), "odd letters in \(helix.sequence)")
}

// MARK: - the renderer

section("renderer")

let device = try findDevice()
let renderer = try MoleculeRenderer(device: device)
let testLayout = FrameLayout(width: 320, viewHeight: 200, captionHeight: 40)
let buffer = device.makeBuffer(length: testLayout.width * testLayout.height * 4, options: .storageModeShared)!

func pixel(_ x: Int, _ y: Int) -> SIMD3<Int> {
    let p = buffer.contents().assumingMemoryBound(to: UInt8.self) + (y * testLayout.width + x) * 4
    return SIMD3(Int(p[0]), Int(p[1]), Int(p[2]))
}

/// Mean brightness over a patch, for comparing how dark two places are.
func brightness(_ x0: Int, _ y0: Int, _ w: Int, _ h: Int) -> Float {
    var total: Float = 0
    for y in y0..<(y0 + h) {
        for x in x0..<(x0 + w) {
            let p = pixel(x, y)
            total += Float(p.x + p.y + p.z) / 3
        }
    }
    return total / Float(w * h)
}

test("an empty scene is step 2's gradient") {
    let camera = Camera.orbitAboutX(target: .zero, distance: 100, roll: 0, lean: 0, fov: 30)
    try renderer.render(spheres: [], cylinders: [], camera: camera, into: buffer,
                        width: testLayout.width, viewHeight: testLayout.viewHeight, ao: AOSettings.off)
    let top = pixel(160, 0), bottom = pixel(160, testLayout.viewHeight - 1)
    expect(abs(top.x - 140) <= 4 && abs(top.y - 200) <= 4 && abs(top.z - 235) <= 4, "top \(top)")
    expect(abs(bottom.x - 8) <= 4 && abs(bottom.y - 40) <= 4 && abs(bottom.z - 90) <= 4, "bottom \(bottom)")
}

test("occlusion darkens a groove more than the backbone ridge beside it — the step's whole claim") {
    // A point down in the major groove has far less open sky above it than a
    // point on the phosphate ridge, so it must come out darker. This is what
    // makes a groove read as a channel rather than as paint.
    let spheres = spaceFilling(helix)
    var grooveOcclusion: Float = 0, ridgeOcclusion: Float = 0, samples = 0

    // Measure geometrically rather than through the image: for a point just
    // off an atom's surface, count how many directions in the hemisphere
    // escape without hitting another atom.
    func openness(at p: SIMD3<Float>, normal n: SIMD3<Float>) -> Float {
        var open = 0, total = 0
        for i in 0..<24 {
            let u = (Float(i) + 0.5) / 24
            let r = u.squareRoot()
            let phi = 2 * Float.pi * Float(i) * 0.6180339887
            let t = simd_normalize(abs(n.z) < 0.9 ? simd_cross(n, SIMD3<Float>(0, 0, 1))
                                                  : simd_cross(n, SIMD3<Float>(1, 0, 0)))
            let b = simd_cross(n, t)
            let dir = t * (r * cos(phi)) + b * (r * sin(phi)) + n * (1 - u).squareRoot()
            total += 1
            var blocked = false
            for s in spheres {
                let c = SIMD3(s.centerRadius.x, s.centerRadius.y, s.centerRadius.z)
                let rad = s.centerRadius.w
                let oc = p - c
                let bb = simd_dot(oc, dir)
                let hh = bb * bb - (simd_dot(oc, oc) - rad * rad)
                if hh < 0 { continue }
                let tt = -bb - hh.squareRoot()
                if tt > 1e-3 && tt < 8 { blocked = true; break }
            }
            if !blocked { open += 1 }
        }
        return Float(open) / Float(total)
    }

    for a in helix.atoms where a.pair > 10 && a.pair < 30 && !a.name.hasPrefix("H") {
        let radial = SIMD3<Float>(0, a.position.y, a.position.z)
        guard simd_length(radial) > 0.5 else { continue }
        let n = simd_normalize(radial)
        let p = a.position + n * (vdw(a.element) + 0.02)
        if a.part == .major { grooveOcclusion += openness(at: p, normal: n); samples += 1 }
        if a.part == .backbone && a.name == "P" { ridgeOcclusion += openness(at: p, normal: n) }
    }
    let majorCount = helix.atoms.filter { $0.pair > 10 && $0.pair < 30 && $0.part == .major }.count
    let ridgeCount = helix.atoms.filter { $0.pair > 10 && $0.pair < 30 && $0.part == .backbone && $0.name == "P" }.count
    expect(majorCount > 20 && ridgeCount > 10, "too few sample atoms: \(majorCount)/\(ridgeCount)")
    let grooveMean = grooveOcclusion / Float(majorCount)
    let ridgeMean = ridgeOcclusion / Float(ridgeCount)
    expect(grooveMean < ridgeMean, "groove openness \(grooveMean) is not below the ridge's \(ridgeMean)")
    expect(ridgeMean - grooveMean > 0.05,
           "the difference is only \(ridgeMean - grooveMean) — occlusion would not read")
}

test("occlusion changes the picture substantially, and only ever darkens") {
    // What occlusion does is not "more contrast": it adds an ambient term that
    // every surface loses some of, in proportion to how enclosed it is. So it
    // darkens the molecule overall while deepening its crevices, and a raw
    // contrast measure falls rather than rises. Two earlier versions of this
    // test asserted the opposite and were simply wrong about the physics.
    //
    // The claim that matters — grooves darker than ridges — is measured
    // geometrically in the test above. Here we check only that occlusion is
    // doing real work on the image, and doing it in the one direction it can.
    let spheres = spaceFilling(helix)
    let camera = Camera.orbitAboutX(target: .zero, distance: cameraDistance, roll: 40, lean: 5, fov: 30)
    func sample(_ ao: AOSettings) -> [Float] {
        try? renderer.render(spheres: spheres, cylinders: [], camera: camera, into: buffer,
                             width: testLayout.width, viewHeight: testLayout.viewHeight,
                             ao: ao, lightRoll: 40)
        var out: [Float] = []
        for y in stride(from: 60, to: 140, by: 2) {
            for x in stride(from: 40, to: 280, by: 2) { out.append(brightness(x, y, 2, 2)) }
        }
        return out
    }
    let on = sample(AOSettings(probes: 12, distance: 8, strength: 1, only: false, contrast: 1.7))
    let off = sample(AOSettings.off)
    expectEqual(on.count, off.count)
    // Only where the molecule is. The background is the same gradient either
    // way, and counting it would just dilute the measurement.
    var changed = 0, brighter = 0, totalDelta: Float = 0, onMolecule = 0
    for i in 0..<on.count {
        guard off[i] > 60 else { continue }          // brighter than the gradient behind it
        onMolecule += 1
        let d = on[i] - off[i]
        totalDelta += abs(d)
        if abs(d) > 4 { changed += 1 }
        if d > 4 { brighter += 1 }
    }
    expect(onMolecule > 200, "only \(onMolecule) molecule pixels sampled")
    let fractionChanged = Float(changed) / Float(max(onMolecule, 1))
    expect(fractionChanged > 0.3, "occlusion moved only \(Int(fractionChanged * 100))% of the pixels")
    expect(totalDelta / Float(max(onMolecule, 1)) > 6,
           "mean change of only \(totalDelta / Float(max(onMolecule, 1))) levels")
    expect(brighter == 0, "\(brighter) pixels got brighter — occlusion can only remove light")
}

test("occlusion is steady as the helix turns, because the subject never moves") {
    // Two rolls a long way apart. A given atom's own shading depends only on
    // geometry that never changes, so the overall distribution of brightness
    // on the molecule should barely differ.
    let spheres = spaceFilling(helix)
    func molecule(at roll: Float) -> Float {
        let camera = Camera.orbitAboutX(target: .zero, distance: cameraDistance, roll: roll, lean: 7, fov: 30)
        try? renderer.render(spheres: spheres, cylinders: [], camera: camera, into: buffer,
                             width: testLayout.width, viewHeight: testLayout.viewHeight,
                             ao: AOSettings(probes: 12, distance: 8, strength: 1, only: false, contrast: 1.7),
                             lightRoll: roll)
        return brightness(70, 70, 180, 60)
    }
    let a = molecule(at: 0), b = molecule(at: 90), c = molecule(at: 180)
    expect(abs(a - b) < 14 && abs(a - c) < 14, "mean brightness drifts: \(a), \(b), \(c)")
}

test("the loop is seamless: a full turn returns the camera and lights exactly") {
    let first = Camera.orbitAboutX(target: .zero, distance: cameraDistance, roll: 0, lean: 5, fov: 30)
    let last = Camera.orbitAboutX(target: .zero, distance: cameraDistance, roll: 360, lean: 5, fov: 30)
    expect(simd_distance(first.origin, last.origin) < 1e-3, "camera moved \(simd_distance(first.origin, last.origin)) Å")
    expect(simd_distance(first.up, last.up) < 1e-3, "up drifted")
    let l0 = rotateAboutX(keyLightWorld, degrees: 0)
    let l1 = rotateAboutX(keyLightWorld, degrees: -360)
    expect(simd_distance(l0, l1) < 1e-3, "the key light drifted")
}

test("every atom stays inside the frame all the way round") {
    var worst = ""
    var worstOver: Float = 0
    for step in stride(from: 0, to: 360, by: 15) {
        let camera = Camera.orbitAboutX(target: .zero, distance: cameraDistance,
                                        roll: Float(step), lean: 5, fov: 30)
        for a in helix.atoms {
            let p = camera.project(a.position, width: 960, height: 400)
            let pad = vdw(a.element) * 4          // the sphere's own width in pixels, roughly
            if p.x < pad || p.x > 960 - pad || p.y < pad || p.y > 400 - pad {
                let over = max(pad - p.x, p.x - (960 - pad), pad - p.y, p.y - (400 - pad))
                if over > worstOver { worstOver = over; worst = "roll \(step)°: \(a.name) at \(p), over by \(over) px" }
            }
        }
    }
    expect(worst.isEmpty, worst)
}

finish()
