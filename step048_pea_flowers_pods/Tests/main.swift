// Tests for step 48, then step 47's, which still hold. Step 48: the flowers
// are white — measured in a render — and each is a standard, two wings and a
// keel; the pods hang only at nodes older than the flowers', and hold as many
// peas as the flora allows; the loop still closes with them in it.
//
// Step 47's header follows.
//
// Tests for step 47. The garden pea's numbers are checked against the floras
// and Darwin; what tells it from step 46's sweet pea — stipules larger than
// the leaflets, several tendrils a leaf, a round unwinged stem — on the scene
// as drawn; its tendrils against what a tendril is (a leaflet, so it grows at
// a pinna position past the leaflets) and what an anchored tendril must do
// (coil both ways, with one perversion and no net twist); its grip on the net
// by probing the same distance function the GPU draws with. Then the loop: a
// loop later the scene must be the same one mesh square higher, and the film
// must only ever run forward.
//
// TENDRIL_MUTANT=purple|pods_above_flowers|rewind|small_stipules|one_tendril|
// no_perversion|stem_tendril breaks it on purpose; `make mutants` requires the
// suite to fail for each.

import Foundation
import Metal
import simd

let mutant: Mutant = Mutant.fromEnvironment

final class Box {
    var renderer: PlantRenderer?
    var error: String = ""
    init() {
        do { renderer = try PlantRenderer(look: lookFor(mutant), on: try findDevice()) } catch { self.error = "\(error)" }
    }
}
let box = Box()
if box.renderer == nil { print("no renderer: \(box.error)") }

func probe(_ s: PlantScene, _ pts: [V3], _ mask: UInt32 = allMaterials) -> [SIMD2<Float>]? {
    guard let r = box.renderer else { return nil }
    return try? r.probe(s, pts, mask: mask)
}

/// Hours to test at: spread over a loop, off the frame lattice.
let testHours: [Float] = [0, 3.3, 9.71, 16.2, 21.05]

/// Points on the surface of a tube round a polyline.
func tubeSurface(_ pts: [V3], radius r: Float, around: Int = 12) -> [V3] {
    var out: [V3] = []
    if pts.count < 2 { return out }
    for i in 1..<pts.count {
        let a: V3 = pts[i - 1]
        let b: V3 = pts[i]
        guard simd_distance(a, b) > 1e-5 else { continue }
        let t: V3 = simd_normalize(b - a)
        let e1: V3 = perpendicular(to: t)
        let e2: V3 = simd_cross(t, e1)
        for k in 0..<2 {
            let p: V3 = a + (b - a) * (Float(k) / 2)
            for j in 0..<around {
                let th: Float = Float(j) / Float(around) * 2 * Float.pi
                out.append(p + (e1 * cos(th) + e2 * sin(th)) * r)
            }
        }
    }
    return out
}

// MARK: -

section("step 48: the flowers, against the floras")

test("flowers: 1–3 a stalk (Flora of China, Pakistan); standard 16–30 mm (Pakistan); calyx 8–15 mm; stalk ½–2× the stipule; wings bigger than the keel") {
    expect(flowersPerStalkRange.contains(flowersPerStalk))
    expect(standardRange.contains(standardSize.x))
    expect(calyxRange.contains(calyxLength))
    expect(peduncleRange().contains(peduncleLength))
    expect(wingSize.x * wingSize.y > keelSize.x * keelSize.y)
    var stalks: Int = 0
    for t in testHours {
        for n in nodes(t) {
            guard let f = inflorescence(n, t, mutant) else { continue }
            expectEqual(f.flowers.count, flowersPerStalk)
            stalks += 1
        }
    }
    print("        \(stalks) flower stalks, \(flowersPerStalk) flowers each")
    expect(stalks > 20)
}

test("every flower is papilionaceous: a standard above, two wings either side of the keel, and the keel between them") {
    var checked: Int = 0
    var scenePetals: Int = 0
    var flowerPetals: Int = 0
    for t in testHours {
        let s: PlantScene = peaScene(t, mutant)
        scenePetals += s.leaflets.filter { $0.material == .petal }.count
        for n in nodes(t) {
            guard let f = inflorescence(n, t, mutant) else { continue }
            for fl in f.flowers {
                flowerPetals += fl.petals.count
                if fl.petals.isEmpty { continue }
                expectEqual(fl.petals.map { $0.shape }, [.standard, .wing, .wing, .keel])
                guard fl.petals.count == 4 else { continue }
                let keel: Leaflet = fl.petals[3]
                let keelMid: V3 = keel.origin + keel.u * (0.5 * keel.length)
                // The wings lie on opposite sides of the keel's midplane.
                let w1: Float = simd_dot(fl.petals[1].origin + fl.petals[1].u * (0.5 * fl.petals[1].length) - keelMid, fl.across)
                let w2: Float = simd_dot(fl.petals[2].origin + fl.petals[2].u * (0.5 * fl.petals[2].length) - keelMid, fl.across)
                expect(w1 * w2 < 0, "both wings on one side of the keel")
                // An open standard stands above the keel.
                if flowerStage(reproductiveAge(n.age, mutant)) == .open {
                    let st: Leaflet = fl.petals[0]
                    let top: Float = simd_dot(st.origin + st.u * (0.5 * st.length) - keelMid, fl.up)
                    expect(top > 0.25 * st.length, "the standard is not above the keel")
                }
                checked += 1
            }
        }
    }
    print("        \(checked) flowers with a standard, two wings and a keel")
    expect(checked >= 10)
    expectEqual(scenePetals, flowerPetals)
}

test("the petals are white — measured in a render of an open flower (Flora of Pakistan var. sativum: 'Flower white')") {
    guard let r = box.renderer else { expect(false, "no renderer"); return }
    // An open flower, close up, seen from the film's camera direction.
    var sums: [SIMD3<Float>] = []
    var counts: [Int] = []
    var worstSaturation: Float = 0
    for t in [Float(0.5), 3.0, 14.2] {   // hours when a node is 12–16 h old: open
        guard let n = nodes(t).first(where: { flowerStage(reproductiveAge($0.age, mutant)) == .open }),
              let f = inflorescence(n, t, mutant), let st = f.flowers.first?.petals.first else { continue }
        let centre: V3 = st.origin + st.u * (0.45 * st.length)
        let toward: V3 = simd_normalize(camera(t).position - centre)
        let cam = Camera(position: centre + toward * 150, target: centre, tanHalf: 16.0 / 150)
        guard let out = try? r.render(peaScene(t, mutant), camera: cam, width: 96, height: 96, samples: 1, tubeMin: 0)
        else { expect(false, "render failed"); return }
        var sum = SIMD3<Float>(0, 0, 0)
        var c: Int = 0
        for i in 0..<(96 * 96) where Int(out.frame.seen[i].x) == Material.petal.rawValue && out.frame.rgba[i].w > 0.99 {
            let px: SIMD4<Float> = out.frame.rgba[i]
            sum += SIMD3<Float>(px.x, px.y, px.z)
            c += 1
        }
        guard c > 0 else { continue }
        let mean: SIMD3<Float> = sum / Float(c)
        let sat: Float = (mean.max() - mean.min()) / max(mean.max(), 1e-3)
        worstSaturation = max(worstSaturation, sat)
        sums.append(mean)
        counts.append(c)
        expect(mean.min() > 0.55, "petals too dark: \(mean)")
        expect(sat < 0.12, "petals coloured: \(mean), saturation \(sat)")
    }
    for (m, c) in zip(sums, counts) {
        print(String(format: "        %d petal pixels, mean sRGB (%.2f, %.2f, %.2f)", c, m.x, m.y, m.z))
    }
    print(String(format: "        worst saturation %.3f", worstSaturation))
    expect(counts.count == 3 && counts.allSatisfy { $0 > 300 }, "too few petal pixels to judge: \(counts)")
}

section("step 48: pods below the flowers")

test("pods hang only at nodes older than any flower's, with withering petals round young pods in between") {
    var seenBetween: Int = 0
    var checked: Int = 0
    for k in 0..<24 {
        let t: Float = loopHours * Float(k) / 24 + 0.29
        var oldestFlower: Float = -1
        var youngestPod: Float = .infinity
        for n in nodes(t) {
            guard let f = inflorescence(n, t, mutant) else { continue }
            let fresh: Bool = f.flowers.contains { fl in fl.petals.contains { $0.tint < 0.01 } } && f.flowers.allSatisfy { $0.pod == nil }
            let pods: Bool = f.flowers.contains { $0.pod != nil }
            if fresh { oldestFlower = max(oldestFlower, n.age) }
            if pods { youngestPod = min(youngestPod, n.age) }
            if pods && f.flowers.contains(where: { fl in fl.petals.contains { $0.tint > 0.2 } }) { seenBetween += 1 }
        }
        expect(oldestFlower < youngestPod, String(format: "hour %.2f: a flower at a node %.1f h old, a pod at %.1f h", t, oldestFlower, youngestPod))
        checked += 1
    }
    print("        \(checked) hours checked; \(seenBetween) node-hours with withering petals round a young pod")
    expect(seenBetween > 5)
}

test("a full pod is 40–70 × 12–17 mm (Flora of Pakistan) and holds 2–10 peas (Flora of China), each a bulge in the pod") {
    var pods: Int = 0
    for t in testHours {
        for n in nodes(t) where reproductiveAge(n.age, mutant) > seedsSwell.upperBound {
            guard let f = inflorescence(n, t, mutant) else { continue }
            for fl in f.flowers {
                guard let p = fl.pod else { expect(false, "no pod at an old node"); continue }
                let peas: Int = bulges(p)
                let len: Float = polylineLength(p.points)
                let width: Float = 2 * (p.radii.max() ?? 0)
                expect(seedsPerPodRange.contains(peas), "\(peas) peas")
                expectEqual(peas, seedsPerPod)
                expect(podLengthRange.contains(len) && podWidthRange.contains(width), "pod \(len) × \(width) mm")
                if pods == 0 { print(String(format: "        a full pod: %.1f × %.1f mm, %d peas", len, width, peas)) }
                pods += 1
            }
        }
    }
    print("        \(pods) full pods checked")
    expect(pods >= 4)
}

test("no flower, flower stalk or pod passes through the stem, a leaf, a stipule, a tendril or the net") {
    var deepest: Float = .infinity
    let others: UInt32 = maskOf([.stem, .leaf, .stipule, .tendril, .twine])
    for k in 0..<12 {
        let t: Float = loopHours * Float(k) / 12 + 0.41
        let s: PlantScene = peaScene(t, mutant)
        var pts: [V3] = []
        for n in nodes(t) {
            guard let f = inflorescence(n, t, mutant) else { continue }
            // The stalk, but not its first 3 mm, which grow out of the stem.
            let stalk: [V3] = f.peduncle.points.filter { simd_distance($0, f.peduncle.points[0]) > 3 }
            pts += tubeSurface(stalk, radius: 0.7, around: 8)
            for fl in f.flowers {
                pts += tubeSurface(fl.pedicel.points, radius: 0.55, around: 8)
                if let p = fl.pod {
                    for i in stride(from: 0, to: p.points.count - 1, by: 2) {
                        let a: V3 = p.points[i]
                        let tg: V3 = simd_normalize(p.points[i + 1] - a)
                        let e1: V3 = perpendicular(to: tg)
                        let e2: V3 = simd_cross(tg, e1)
                        for j in 0..<10 {
                            let th: Float = Float(j) / 10 * 2 * Float.pi
                            pts.append(a + (e1 * cos(th) + e2 * sin(th)) * p.radii[i])
                        }
                    }
                }
                for l in fl.petals {
                    for i in 0...6 {
                        let xi: Float = Float(i) / 6
                        pts.append(l.origin + l.u * (l.length * xi))
                    }
                }
            }
        }
        guard let d = probe(s, pts, others) else { expect(false, "probe failed"); return }
        deepest = min(deepest, d.map { $0.x }.min() ?? .infinity)
    }
    print(String(format: "        the nearest flower or pod point is %.2f mm from anything else", deepest))
    expect(deepest > -0.05, "a flower or pod enters something by \(-deepest) mm")
}

test("flowers and pods repeat with the loop: a loop later the same stages stand one mesh square higher") {
    var worst: Float = 0
    var petals: Int = 0
    var pods: Int = 0
    for t in testHours {
        let a: PlantScene = peaScene(t, mutant)
        petals += a.leaflets.filter { $0.material == .petal }.count
        pods += a.chains.filter { $0.material == .pod }.count
        worst = max(worst, a.shifted(by: loopRise).largestDifference(from: peaScene(t + loopHours, mutant)))
    }
    print(String(format: "        %d petals and %d pods across the hours; largest difference a loop later %.5f mm", petals, pods, worst))
    expect(petals > 0 && pods > 0)
    expect(worst < 0.01)
}

section("numbers, against the sources")

/// A node's leaf pieces in the scene: its two stipules and its leaflets, in
/// the order they are added.
func leafPieces(_ s: PlantScene, _ t: Float) -> [(node: NodeInfo, stipules: [Leaflet], leaflets: [Leaflet])] {
    var out: [(node: NodeInfo, stipules: [Leaflet], leaflets: [Leaflet])] = []
    var i: Int = 0
    let blades: [Leaflet] = s.leaflets.filter { $0.material == .leaf || $0.material == .stipule }
    for n in nodes(t) {
        let chunk: [Leaflet] = Array(blades[i..<min(i + 2 + 2 * leafletPairs, blades.count)])
        out.append((n, chunk.filter { $0.material == .stipule }, chunk.filter { $0.material == .leaf }))
        i += 2 + 2 * leafletPairs
    }
    return out
}

test("leaves: 1–3 pairs of leaflets, 2–7 × 1–4 cm (Flora of China); stipules 1.5–8 cm (Flora of Pakistan), to 10 × 6 cm") {
    expect(leafletPairRange.contains(leafletPairs))
    expect(leafletLengthRange.contains(leafletSize.x) && leafletWidthRange.contains(leafletSize.y))
    expect(stipuleLengthRange.contains(stipuleSizeTrue.x) && stipuleSizeTrue.y <= stipuleWidthMax)
    let t: Float = 5
    let s: PlantScene = peaScene(t, mutant)
    let n: Int = nodes(t).count
    expectEqual(s.leaflets.filter { $0.material == .leaf }.count, 2 * leafletPairs * n)
    expectEqual(s.leaflets.filter { $0.material == .stipule }.count, 2 * n)
}

test("on every mature leaf the stipules are larger than the leaflets (Flora of China: 'larger than leaflets')") {
    var checked: Int = 0
    var ratio: Float = .infinity
    for t in testHours {
        let s: PlantScene = peaScene(t, mutant)
        for piece in leafPieces(s, t) where piece.node.age > leafOpenHours {
            let stipLength: Float = piece.stipules.map { $0.length }.min() ?? 0
            let leafLength: Float = piece.leaflets.map { $0.length }.max() ?? .infinity
            // Blade area goes as length × width for the same kind of outline.
            let stipArea: Float = piece.stipules.map { $0.length * $0.halfWidth }.min() ?? 0
            let leafArea: Float = piece.leaflets.map { $0.length * $0.halfWidth }.max() ?? .infinity
            expect(stipLength > leafLength, "node \(piece.node.index): stipule \(stipLength) mm, leaflet \(leafLength) mm")
            expect(stipArea > leafArea, "node \(piece.node.index): stipule smaller in area than a leaflet")
            ratio = min(ratio, stipArea / leafArea)
            checked += 1
        }
    }
    print(String(format: "        %d mature leaves; the smallest stipule is %.2f× the largest leaflet's area", checked, ratio))
    expect(checked >= 20)
}

test("the stem is round and neither it nor the leaf stalks are winged (Flora of China: 'terete'), unlike the sweet pea") {
    let s: PlantScene = peaScene(5, mutant)
    expectEqual(s.ribbons.count, 0)
    let stems: [Chain] = s.chains.filter { $0.material == .stem }
    expectEqual(stems.count, 1)
    // A chain of capsules is round in every cross-section; below the tapering
    // tip its radius is the stem's.
    if let st = stems.first {
        let mature: [Float] = Array(st.radii.dropLast(8))
        expect(mature.allSatisfy { abs($0 - stemRadius) < 1e-4 })
        print(String(format: "        one round stem of %d capsules, radius %.1f mm; %d wing panels", st.points.count - 1,
                     stemRadius, s.ribbons.count))
    }
}

test("the frame says what it is: the garden pea, Mendel's white, pollinated in the bud, and a gradient not a pod time-lapse") {
    expect(nameCaption.contains("Pisum sativum") && nameCaption.contains("the pea you eat"))
    expect(whiteCaption.contains("Mendel") && whiteCaption.contains("gene A") && whiteCaption.contains("bHLH"))
    expect(selfCaption.contains("bud") && selfCaption.contains("before the flower opens"))
    expect(gradientCaption.contains("flowers above, pods below") && gradientCaption.contains("older nodes"))
    expect(compressedCaption.contains("weeks") && compressedCaption.contains("compressed"))
    // The time-lapse is the climb's; nothing calls the pods a time-lapse.
    expect(timeCaption().contains("climb") && timeCaption().contains(String(format: "%.0f hours", loopHours)))
    for c in [whiteCaption, selfCaption, gradientCaption, compressedCaption] { expect(!c.contains("time-lapse")) }
    print("        \"\(whiteCaption)\"; \"\(selfCaption)\"; \"\(gradientCaption)\"; \"\(compressedCaption)\"; \"\(timeCaption())\"")
}

test("the tendrils and young stem sweep round in the pea's own time (Darwin: ellipses of 1 h 20 m and 1 h 30 m)") {
    let minutes: Float = nutationHours * 60
    print(String(format: "        one sweep %.1f min, %d in a loop", minutes, nutationsPerLoop))
    expect(peaEllipseMinutesRange.contains(minutes))
    expect(abs(loopHours / nutationHours - Float(nutationsPerLoop)) < 1e-4)
}

test("the free coil forms half a day to a day after the catch (Darwin: 'in half a day, or in a day or two')") {
    expect(contractionStartsAfter >= 7 && contractionStartsAfter <= 48)
    expect(contractionCompleteAfter > contractionStartsAfter && contractionCompleteAfter <= 48)
}

section("tendrils are leaflets: several a leaf, at the pinnae past the leaflets")

test("every leaf has more than one tendril: two pairs and a terminal one (Darwin: 'two or three pairs of branches')") {
    var fewest: Int = .max
    var counted: Int = 0
    for t in testHours {
        let s: PlantScene = peaScene(t, mutant)
        let ns: [NodeInfo] = nodes(t)
        for n in ns {
            let td: TendrilGeometry = tendril(n, t, mutant)
            fewest = min(fewest, tendrilBases(td).count)
            counted += 1
        }
        // And every one is drawn.
        let drawn: Int = s.chains.filter { $0.material == .tendril }.count
        expect(drawn > ns.count, "\(drawn) tendril chains on \(ns.count) leaves")
    }
    print("        \(counted) leaves: the fewest tendrils on one is \(fewest)")
    expect(fewest > 1, "a leaf with \(fewest) tendril")
    expect(fewest == 2 * tendrilPairs + 1 && tendrilPairRange.contains(tendrilPairs))
}

test("every tendril starts where a leaflet would sit: on a pinna of the rachis, past the leaflet pairs, well out from the stem") {
    var worstLeaflet: Float = 0
    var worstTendril: Float = 0
    var nearest: Float = .infinity
    var spacingSpread: Float = 0
    var count: Int = 0
    for t in testHours {
        let s: PlantScene = peaScene(t, mutant)
        let pieces = leafPieces(s, t)
        for (k, n) in nodes(t).enumerated() {
            let td: TendrilGeometry = tendril(n, t, mutant)
            // The leaflets sit on the first pinnae, one each side.
            for (j, l) in pieces[k].leaflets.enumerated() {
                worstLeaflet = max(worstLeaflet, simd_distance(l.origin, td.pinnae[j / 2]))
            }
            // Every tendril base is a later pinna.
            let tendrilPinnae: [V3] = Array(td.pinnae[leafletPairs...])
            for b in tendrilBases(td) {
                let d: Float = tendrilPinnae.map { simd_distance($0, b) }.min() ?? .infinity
                worstTendril = max(worstTendril, d)
                let stem: V3 = stemPoint(n.y, t)
                nearest = min(nearest, simd_distance(b, stem))
                let drawn: Bool = s.chains.contains { $0.material == .tendril && simd_distance($0.points[0], b) < 1e-4 }
                expect(drawn, "no tendril chain starts at a base on node \(n.index)")
                count += 1
            }
            // The pinnae are one evenly spaced series along the rachis.
            var gaps: [Float] = []
            for i in 1..<td.pinnae.count { gaps.append(simd_distance(td.pinnae[i - 1], td.pinnae[i])) }
            spacingSpread = max(spacingSpread, (gaps.max()! - gaps.min()!) / gaps.max()!)
        }
    }
    print(String(format: "        %d tendrils: base within %.4f mm of a pinna past the leaflets (leaflets within %.4f mm of theirs), at least %.1f mm from the stem; pinna gaps vary %.0f%%",
                 count, worstTendril, worstLeaflet, nearest, spacingSpread * 100))
    expect(worstLeaflet < 1e-3)
    expect(worstTendril < 1e-3, "a tendril starts \(worstTendril) mm from any pinna")
    expect(nearest > petioleLength * 0.5, "a tendril starts on the stem")
    expect(spacingSpread < 0.35)
}

section("the anchored free coil: two hands and one perversion")

/// The handedness runs of a coil: θ about the chord, measured right-handedly,
/// against distance along the chord. Returns the signs of the runs, and the
/// turns in each run.
func handednessRuns(_ pts: [V3]) -> (signs: [Int], turns: [Float], net: Float) {
    let a: V3 = pts.first!
    let e: V3 = simd_normalize(pts.last! - a)
    let theta: [Float] = unwrappedAngles(pts, origin: a, axis: e, reference: perpendicular(to: e))
    var signs: [Int] = []
    var turns: [Float] = []
    var runStart: Float = 0
    var started: Bool = false
    for i in 1..<pts.count {
        let q: V3 = pts[i] - a
        let rho: Float = simd_length(q - e * simd_dot(q, e))
        // Only where the coil is off its chord; the ends ramp in along it.
        guard rho > 0.5 * coilRadius else { continue }
        let dz: Float = simd_dot(pts[i] - pts[i - 1], e)
        let dth: Float = theta[i] - theta[i - 1]
        guard abs(dth) > 1e-3, abs(dz) > 1e-6 else { continue }
        let sgn: Int = (dth / dz) > 0 ? 1 : -1
        if !started { runStart = theta[i - 1]; started = true }
        if signs.last != sgn {
            if !signs.isEmpty { turns.append(abs(theta[i - 1] - runStart) / (2 * Float.pi)); runStart = theta[i - 1] }
            signs.append(sgn)
        }
    }
    if started { turns.append(abs(theta[theta.count - 1] - runStart) / (2 * Float.pi)) }
    return (signs, turns, (theta.last! - theta.first!) / (2 * Float.pi))
}

test("every sprung tendril's free coil has exactly one perversion, a right-handed helix one side and left-handed the other, with as many turns each way (Darwin) and no net twist") {
    var checked: Int = 0
    for t in testHours {
        for n in nodes(t) {
            let td: TendrilGeometry = tendril(n, t, mutant)
            guard td.contraction > 0.95 else { continue }
            let r = handednessRuns(td.free)
            checked += 1
            if checked == 1 {
                print(String(format: "        runs %@; turns %@; net twist %.3f turns", "\(r.signs)",
                             r.turns.map { String(format: "%.2f", $0) }.joined(separator: " / "), r.net))
            }
            expect(r.signs.count == 2, "node \(n.index): \(r.signs.count - 1) perversions")
            guard r.signs.count == 2 else { continue }
            expect(r.signs[0] == -r.signs[1], "both sides the same hand")
            expect(abs(r.turns[0] - r.turns[1]) < 0.25, "turns \(r.turns[0]) one way, \(r.turns[1]) the other")
            expect(r.turns[0] >= 1 && r.turns[1] >= 1, "under a turn a side")
            expect(abs(r.net) < 0.05, "net twist \(r.net) turns between two anchored ends")
        }
    }
    print("        \(checked) sprung tendrils checked")
    expect(checked >= 3)
}

test("the spring draws the leaf in: the coil keeps its length, and its chord shortens as it coils") {
    let n = NodeInfo(index: 3, y: 150, age: 0, side: -1)
    let tBirth: Float = 150 / growthRate
    var lastChord: Float = .infinity
    var worstLength: Float = 0
    var pull: Float = 0
    for k in 0...24 {
        let age: Float = contactAge + contractionStartsAfter + Float(k) / 24 * (contractionCompleteAfter - contractionStartsAfter)
        let nn = NodeInfo(index: n.index, y: n.y, age: age, side: n.side)
        let td: TendrilGeometry = tendril(nn, tBirth + age, mutant)
        let k0: Catch = tendrilCatch(nn, tBirth + age)
        let chord: Float = simd_distance(td.free.first!, td.free.last!)
        worstLength = max(worstLength, abs(polylineLength(td.free) - k0.reach) / k0.reach)
        expect(chord <= lastChord + 1e-4)
        lastChord = chord
        pull = simd_distance(td.leafTip, k0.leafTip)
    }
    print(String(format: "        coil length held to %.2f%%; the leaf tip is drawn %.1f mm towards the net", worstLength * 100, pull))
    expect(worstLength < 0.01)
    expect(pull > 5)
}

section("gripping the net")

test("a searching tendril never enters a string, and its tip arrives touching one exactly at contact") {
    var deepest: Float = .infinity
    var arrival: Float = .infinity
    let rw: Float = twineRadius + tendrilRadius
    for k in 0..<60 {
        let t: Float = loopHours * Float(k) / 60 + 0.013
        for n in nodes(t) where n.age < contactAge {
            let td: TendrilGeometry = tendril(n, t, mutant)
            let tip: V3 = td.free.last!
            let d: Float = simd_distance(V3(tip.x, 0, tip.z), V3(td.stringAxis.x, 0, td.stringAxis.z)) - rw
            deepest = min(deepest, d)
        }
    }
    // At the moment of contact.
    let n = NodeInfo(index: 4, y: 200, age: contactAge, side: 1)
    let tc: Float = 200 / growthRate + contactAge
    let td: TendrilGeometry = tendril(n, tc - 1e-4, mutant)
    let tip: V3 = td.free.last!
    arrival = simd_distance(V3(tip.x, 0, tip.z), V3(td.stringAxis.x, 0, td.stringAxis.z)) - rw
    print(String(format: "        before contact the tip stays %.2f mm or more off the string; at contact %.4f mm", deepest, arrival))
    expect(deepest > 0)
    expect(abs(arrival) < 0.01)
}

test("the coil round the string touches it all the way round and never enters it — on the distance function") {
    var worstIn: Float = 0
    var worstGap: Float = 0
    var rings: Int = 0
    for t in testHours {
        let s: PlantScene = peaScene(t, mutant)
        for n in nodes(t) where n.age > contactAge + wrapHours {
            let td: TendrilGeometry = tendril(n, t, mutant)
            let pts: [V3] = tubeSurface(td.wrap, radius: tendrilRadius, around: 16)
            guard let d = probe(s, pts, maskOf([.twine])) else { expect(false, "probe failed"); return }
            var i: Int = 0
            while i + 16 <= d.count {
                let ring: [Float] = d[i..<(i + 16)].map { $0.x }
                worstIn = min(worstIn, ring.min()!)
                worstGap = max(worstGap, ring.min()!)
                rings += 1
                i += 16
            }
        }
    }
    print(String(format: "        %d rings round the string: deepest %.3f mm in, widest gap %.3f mm", rings, -worstIn, worstGap))
    expect(rings > 100)
    expect(worstIn > -0.03, "a tendril enters the string by \(-worstIn) mm")
    expect(worstGap < 0.06, "the coil stands \(worstGap) mm off the string")
}

test("no part of any tendril passes through the net, at any hour of the loop") {
    var deepest: Float = .infinity
    for k in 0..<24 {
        let t: Float = loopHours * Float(k) / 24 + 0.37
        let s: PlantScene = peaScene(t, mutant)
        var pts: [V3] = []
        for n in nodes(t) {
            let td: TendrilGeometry = tendril(n, t, mutant)
            pts += tubeSurface(td.catching, radius: tendrilRadius, around: 8)
            for b in td.others { pts += tubeSurface(b, radius: tendrilRadius * 0.9, around: 8) }
        }
        guard let d = probe(s, pts, maskOf([.twine])) else { expect(false, "probe failed"); return }
        deepest = min(deepest, d.map { $0.x }.min() ?? .infinity)
    }
    print(String(format: "        deepest tendril point %.3f mm into a string", max(-deepest, 0)))
    expect(deepest > -0.03)
}

test("no leaflet or stipule reaches the net: every blade stays in front of the strings, at any hour") {
    var nearest: Float = .infinity
    for k in 0..<24 {
        let t: Float = loopHours * Float(k) / 24 + 0.37
        let s: PlantScene = peaScene(t, mutant)
        for l in s.leaflets {
            // The folded blade's outline and midrib, drooped (as the kernel draws it).
            for i in 0...10 {
                let xi: Float = Float(i) / 10
                let half: Float = l.halfWidth * outlineProfile(l.shape, xi)
                let mid: V3 = l.origin + l.u * (l.length * xi) - l.w * (l.droop * l.length * xi * xi)
                for side in [Float(1), -1] {
                    let across: V3 = l.v * (side * cos(l.fold)) + l.w * sin(l.fold)
                    let p: V3 = mid + across * half
                    nearest = min(nearest, p.z, mid.z)
                }
            }
        }
    }
    print(String(format: "        the nearest blade point is %.1f mm in front of the net's plane", nearest))
    expect(nearest > twineRadius + 1)
}

test("every stage is on screen somewhere in the loop: searching, curling round the string, anchored, coiling, sprung") {
    var seen = Set<Int>()
    for f in stride(from: 0, to: frameCount, by: 10) {
        let t: Float = hourOf(frame: f)
        let cam: Camera = camera(t)
        for n in nodes(t) {
            let td: TendrilGeometry = tendril(n, t, mutant)
            let p: SIMD2<Float> = cam.project(td.free[td.free.count / 2], width: 480, height: 624)
            if p.x > 0 && p.x < 480 && p.y > 0 && p.y < 624 { seen.insert(td.stage.rawValue) }
        }
    }
    print("        stages seen: \(seen.sorted())")
    expectEqual(seen.count, 5)
    // And in order, for one tendril.
    var last: Int = -1
    for k in 0...100 {
        let s: Int = stage(Float(k) * 0.5).rawValue
        expect(s >= last)
        last = s
    }
}

section("the loop")

test("a loop later the scene is the same scene one mesh square higher (and so is the camera)") {
    var worst: Float = 0
    for t in testHours {
        let a: PlantScene = peaScene(t, mutant).shifted(by: loopRise)
        let b: PlantScene = peaScene(t + loopHours, mutant)
        worst = max(worst, a.largestDifference(from: b))
        worst = max(worst, simd_distance(camera(t).position + V3(0, loopRise, 0), camera(t + loopHours).position))
    }
    print(String(format: "        largest difference %.5f mm", worst))
    expect(worst < 0.01)
    expect(abs(loopRise - meshSpacing) < 1e-4, "the net would not repeat with the loop")
}

test("the film only runs forward: every frame later and higher than the last, the last into the first included") {
    var bad: Int = 0
    var steps: [Float] = []
    for f in 0..<frameCount {
        let t0: Float = hourOf(frame: f, mutant: mutant)
        let t1: Float = f + 1 < frameCount ? hourOf(frame: f + 1, mutant: mutant) : hourOf(frame: 0, mutant: mutant) + loopHours
        if !(t1 > t0) || !(apexHeight(t1) > apexHeight(t0)) { bad += 1 }
        steps.append(t1 - t0)
    }
    let spread: Float = steps.max()! - steps.min()!
    print(String(format: "        %d backward steps; step %.4f h, spread %.6f h", bad, steps[0], spread))
    expect(bad == 0, "\(bad) frames run backwards")
    expect(spread < 1e-4)
}

test("rendered, the frame one loop on matches frame 0") {
    guard let r = box.renderer else { expect(false, "no renderer"); return }
    let w: Int = 120
    let h: Int = 156
    let t0: Float = hourOf(frame: 0, mutant: mutant)
    guard let a = try? r.render(peaScene(t0, mutant), camera: camera(t0), width: w, height: h, samples: 1, tubeMin: 0),
          let b = try? r.render(peaScene(t0 + loopHours, mutant), camera: camera(t0 + loopHours), width: w,
                                height: h, samples: 1, tubeMin: 0) else { expect(false, "render failed"); return }
    var worst: Float = 0
    var off: Int = 0
    for i in 0..<(w * h) {
        let d: Float = simd_reduce_max(simd_abs(a.frame.rgba[i] - b.frame.rgba[i]))
        worst = max(worst, d)
        if d > 2.0 / 255 { off += 1 }
    }
    print(String(format: "        largest pixel difference %.4f; %d of %d pixels differ by more than 2/255", worst, off, w * h))
    expect(worst < 0.05 && Float(off) < 0.02 * Float(w * h))
}

section("the distance function is a distance")

test("outside every surface it never claims more room than there is, beyond what the ray allows") {
    let t: Float = 9.71
    let s: PlantScene = peaScene(t, mutant)
    var rng = Lcg(state: 5)
    var a: [V3] = []
    var b: [V3] = []
    let step: Float = 0.1
    func add(_ p: V3) {
        a.append(p)
        b.append(p + rng.unit() * step)
    }
    let ya: Float = apexHeight(t)
    for _ in 0..<150_000 { add(V3(-110 + 220 * rng.next(), ya - 260 + 300 * rng.next(), -20 + 70 * rng.next())) }
    for l in s.leaflets {
        for _ in 0..<800 {
            add(l.origin + l.u * (l.length * (-0.1 + 1.2 * rng.next())) + l.v * (l.halfWidth * (2.4 * rng.next() - 1.2))
                + l.w * (4 * rng.next() - 2))
        }
    }
    // Close round the stem, the rachises and the tendrils.
    for c in s.chains where c.material != .twine {
        for i in 0..<c.points.count { add(c.points[i] + rng.unit() * (c.radii[i] + 3 * rng.next())) }
    }
    guard let da = probe(s, a), let db = probe(s, b) else { expect(false, "probe failed"); return }
    var worst = [Float](repeating: 0, count: materialCount)
    for i in 0..<a.count where da[i].y == db[i].y && da[i].x > 0 && db[i].x > 0 {
        let m: Int = Int(da[i].y)
        worst[m] = max(worst[m], abs(da[i].x - db[i].x) / step)
    }
    let names: [String] = ["-", "pole", "stem", "petiole", "leaf", "tendril", "twine", "stipule", "petal", "pod"]
    var line: String = "        worst over-report:"
    for m in 2..<materialCount { line += String(format: " %@ %.3f", names[m], worst[m]) }
    print(line + String(format: "; the ray allows %.2f", 1 / stepScale))
    for m in 1..<materialCount { expect(worst[m] * stepScale <= 1.0, "\(names[m]) oversteps: \(worst[m])") }
}

section("the picture")

test("frame 0 shows the net, the stem, stipules, leaflets, tendrils, flowers and pods") {
    guard let r = box.renderer else { expect(false, "no renderer"); return }
    let t: Float = 0
    guard let out = try? r.render(peaScene(t, mutant), camera: camera(t), width: 240, height: 312, samples: 1,
                                  tubeMin: 0.3) else { expect(false, "render failed"); return }
    var counts = [Int](repeating: 0, count: materialCount)
    for s in out.frame.seen { counts[Int(s.x)] += 1 }
    let n: Float = Float(out.frame.seen.count)
    func pc(_ m: Material) -> Float { Float(counts[m.rawValue]) / n * 100 }
    print(String(format: "        twine %.1f%%, stem %.1f%%, stipules %.1f%%, leaflets %.1f%%, tendrils %.2f%%, petals %.2f%%, pods %.2f%%",
                 pc(.twine), pc(.stem), pc(.stipule), pc(.leaf), pc(.tendril), pc(.petal), pc(.pod)))
    expect(pc(.petal) > 0.5 && pc(.pod) > 1, "flowers or pods missing from the frame")
    expect(pc(.twine) > 1 && pc(.stem) > 0.5 && pc(.leaf) > 3 && pc(.stipule) > 3)
    expect(pc(.tendril) > 0.3)
    // The stipules, the pea's mark, take more of the frame than a sweet pea's would.
    expect(pc(.stipule) > 0.5 * pc(.leaf), "stipules \(pc(.stipule))% against leaflets \(pc(.leaf))%")
}

finish()
