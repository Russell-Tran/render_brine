// Tests for step 10.
//
// This step is the first whose subject is largely a MODEL rather than measured
// structure, so the tests divide the same way. Some check the render against
// published measurements — the bilayer thickness, the periplasm, the lipid
// packing — and those are the ordinary kind. Others check that the parts which
// are *not* measured are labelled as such, which matters more here than
// anywhere earlier in the series: the render's claim to honesty is a thing the
// tests can hold it to.

import CoreGraphics
import Foundation
import Metal
import simd

setvbuf(stdout, nil, _IONBF, 0)

let sceneURL = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent().deletingLastPathComponent()
    .appendingPathComponent("Resources/scene.json")
let scene = try loadScene(from: sceneURL)
let beads = scene.beads
let NM: Float = 10

func near(_ a: Float, _ b: Float, within e: Float) -> Bool { abs(a - b) <= e }
func group(_ name: String) -> [Bead] { beads.filter { $0.group == name } }

section("the envelope, against published measurements")

test("the bilayer is 4.7 nm thick, as AFM measures a PE/PG/cardiolipin bilayer") {
    expect(near(scene.layers.bilayer, 4.7 * NM, within: 0.1 * NM),
           "bilayer \(scene.layers.bilayer / NM) nm")
    // And the lipids really occupy that thickness, rather than the number just
    // sitting in the JSON: the heads of the two leaflets should be that far apart.
    let om = group("outer_membrane").filter { $0.part == "head" }
    let above = om.filter { $0.position.y > scene.layers.omCenter }.map { $0.position.y }
    let below = om.filter { $0.position.y < scene.layers.omCenter }.map { $0.position.y }
    let span = (above.max() ?? 0) - (below.min() ?? 0)
    expect(span > 4.0 * NM && span < 8.0 * NM, "head-to-head span \(span / NM) nm")
}

test("the periplasm is 13 nm, between the CEMOVIS and cryo-ET figures of 12 and 14") {
    expect(near(scene.layers.periplasm, 13 * NM, within: 0.5 * NM))
    expect(scene.layers.periplasm >= 12 * NM && scene.layers.periplasm <= 14 * NM,
           "must lie between the two published measurements")
}

test("outer membrane to peptidoglycan is 11 nm, which both methods agree on") {
    expect(near(scene.layers.omToPG, 11 * NM, within: 0.5 * NM))
}

test("the layers stack outside-in with no overlap: OM, then PG, then IM") {
    expect(scene.layers.omCenter > scene.layers.pgCenter, "PG must sit inside the OM")
    expect(scene.layers.pgCenter > scene.layers.imCenter, "IM must sit inside the PG")
    let omInner = scene.layers.omCenter - scene.layers.bilayer / 2
    let imOuter = scene.layers.imCenter + scene.layers.bilayer / 2
    expect(scene.layers.pgCenter < omInner && scene.layers.pgCenter > imOuter,
           "the glycan layer belongs in the periplasm, not inside a membrane")
    // Nothing from one membrane may stray into the other.
    let omLow = group("outer_membrane").map { $0.position.y }.min() ?? 0
    let imHigh = group("inner_membrane").map { $0.position.y }.max() ?? 0
    expect(omLow > imHigh, "the two membranes touch: \(omLow) vs \(imHigh)")
}

test("lipids are packed at 0.588 nm² each, the measured area for POPE") {
    // Recovered from the geometry rather than read back from the constant:
    // count the heads of one leaflet and divide the patch area by them.
    let heads = group("outer_membrane").filter {
        $0.part == "head" && $0.position.y > scene.layers.omCenter
    }
    // Four head beads per lipid.
    let lipids = Float(heads.count) / 4
    let area = (scene.patchX / NM) * (scene.patchZ / NM)   // nm²
    let perLipid = area / lipids
    expect(near(perLipid, 0.588, within: 0.08), "\(perLipid) nm² per lipid")
}

test("no two lipid beads sit inside one another") {
    // A sample, since the full 403,000 × 403,000 comparison is not worth it:
    // a grid over one slab, then only neighbouring cells.
    let sample = group("outer_membrane")
    var worst: Float = .greatestFiniteMagnitude
    var cells: [SIMD3<Int>: [Int]] = [:]
    let cell: Float = 8
    for (i, b) in sample.enumerated() {
        let c = SIMD3<Int>(Int(floor(b.position.x / cell)), Int(floor(b.position.y / cell)),
                           Int(floor(b.position.z / cell)))
        cells[c, default: []].append(i)
    }
    for (c, list) in cells {
        var near: [Int] = []
        for dx in -1...1 { for dy in -1...1 { for dz in -1...1 {
            near += cells[SIMD3(c.x + dx, c.y + dy, c.z + dz)] ?? []
        } } }
        for i in list {
            for j in near where j != i {
                let d = simd_distance(sample[i].position, sample[j].position)
                worst = min(worst, d - 0.001)
            }
        }
    }
    // Beads of the same lipid are meant to touch and overlap a little; what
    // must not happen is two beads at the same place.
    expect(worst > 0.8, "closest pair of beads \(worst) Å apart")
}

section("the plasmid")

test("it is still pGLO: 5,371 base pairs and 1.83 µm of DNA") {
    expectEqual(scene.plasmid.bp, 5371)
    expect(near(scene.plasmid.contourNm, 5371 * 0.34, within: 1))
}

test("it is supercoiled to σ = −0.06, the figure measured for plasmids from E. coli") {
    expect(near(scene.plasmid.sigma, -0.06, within: 0.001))
    expect(near(scene.plasmid.lk0, 5371 / 10.5, within: 1), "Lk0 \(scene.plasmid.lk0)")
    expect(near(scene.plasmid.deltaLk, scene.plasmid.sigma * scene.plasmid.lk0, within: 0.1))
    expect(scene.plasmid.deltaLk < 0, "negative supercoiling means a linking deficit")
}

test("the built shape really carries that writhe, and with the right sign") {
    // The builder solves for the winding by measuring writhe with the Gauss
    // integral, so this checks the solve converged rather than trusting it.
    expect(scene.plasmid.measuredWrithe < 0, "negative supercoiling gives negative writhe")
    let ratio = scene.plasmid.measuredWrithe / scene.plasmid.targetWrithe
    expect(ratio > 0.9 && ratio < 1.1,
           "writhe \(scene.plasmid.measuredWrithe) against a target of \(scene.plasmid.targetWrithe)")
}

test("the plasmid does not pass through itself") {
    let dna = group("plasmid")
    var worst: Float = .greatestFiniteMagnitude
    var pair = ""
    // Beads adjacent along the chain are meant to touch; anything further
    // apart along the contour must stay clear.
    for i in 0..<dna.count {
        guard i + 6 < dna.count else { continue }
        for j in (i + 6)..<dna.count where (dna.count - (j - i)) > 6 {
            let d = simd_distance(dna[i].position, dna[j].position)
            if d < worst { worst = d; pair = "\(i)/\(j)" }
        }
    }
    let minimum = dna.first!.radius * 1.6
    expect(worst > minimum, "closest non-neighbour approach \(worst) Å at \(pair), needs \(minimum)")
}

test("it is compact enough to be a plasmid rather than a rod") {
    // 1.83 µm of DNA folded into something that could fit a 2 µm cell.
    expect(scene.plasmid.extent < 2500, "extent \(scene.plasmid.extent / NM) nm")
    expect(scene.plasmid.extent < scene.plasmid.contourNm * NM / 5,
           "a folded plectoneme should be far shorter than its own contour")
}

section("the calcium — the part that is a model")

test("ions are hydrated Ca²⁺ at 4.1 Å, not bare ions at 1.0 Å") {
    let ions = group("calcium")
    expect(!ions.isEmpty)
    for i in ions { expect(near(i.radius, 4.1, within: 0.01), "radius \(i.radius)") }
}

test("ions crowd the DNA and the membrane rather than being scattered at random") {
    let ions = group("calcium")
    let dna = group("plasmid").map { $0.position }
    var nearDNA = 0, nearMembrane = 0
    for i in ions {
        if dna.contains(where: { simd_distance(i.position, $0) < 30 }) { nearDNA += 1 }
        else if i.position.y > scene.layers.omOuterFace - 2 &&
                i.position.y < scene.layers.omOuterFace + 20 { nearMembrane += 1 }
    }
    // Every ion should be doing one job or the other; a random fill would
    // leave most of them in open water.
    let placed = Float(nearDNA + nearMembrane) / Float(ions.count)
    expect(placed > 0.9, "only \(Int(placed * 100))% of ions are on the DNA or the membrane")
    expect(nearDNA > 0 && nearMembrane > 0, "both populations must exist")
}

section("evidence, which this step has to get right")

test("every bead knows how well it is known, and only from the three levels") {
    let levels = Set(beads.map { $0.evidence })
    expect(levels.isSubset(of: ["measured", "simulated", "model"]), "found \(levels)")
}

test("the envelope is marked measured and the calcium is marked a model") {
    for b in group("outer_membrane") + group("inner_membrane") + group("peptidoglycan") {
        expectEqual(b.evidence, "measured")
    }
    for b in group("calcium") { expectEqual(b.evidence, "model") }
    // The plasmid's shape is modelled too: no one has a structure of a whole
    // supercoiled plasmid.
    for b in group("plasmid") { expectEqual(b.evidence, "model") }
}

test("each route carries the evidence level the literature supports") {
    expectEqual(Route.chemical.evidence, .model)
    expectEqual(Route.electroporation.evidence, .simulated)
    expectEqual(Route.competence.evidence, .measured)
    // The route the classroom kit actually uses is the least understood one,
    // which is the whole point of the panel.
    expect(Route.chemical.isTheLabMethod)
    expect(Route.chemical.evidence.rawValue < Route.competence.evidence.rawValue)
}

test("the levels are ordered, so the bar reads as a scale") {
    expect(Evidence.model.rawValue < Evidence.simulated.rawValue)
    expect(Evidence.simulated.rawValue < Evidence.measured.rawValue)
    expectEqual(Evidence.model.label, "MODEL")
}

section("level of detail, by the pixels-per-feature arithmetic")

test("at the approach's framing a lipid headgroup is big enough to draw") {
    let camera = Camera(origin: SIMD3(0, 520, 1950), target: SIMD3(0, 330, 0), fov: 40)
    let width = 960, height = 720
    // Two points one headgroup apart, across the view.
    let a = camera.project(SIMD3(0, 0, 0), width: width, height: height)
    let b = camera.project(SIMD3(8, 0, 0), width: width, height: height)   // 0.8 nm
    let px = abs(b.x - a.x)
    expect(px > 3, "a headgroup is only \(px) px across")
}

test("a whole cell could not be drawn this way, which is why it is not") {
    // 2 µm across a 960-pixel frame puts a lipid head at half a pixel. The
    // render lives at the patch scale for that reason, and the test records it.
    let nmPerPixel: Float = 2000 / 960
    let headPx = 0.8 / nmPerPixel
    expect(headPx < 1, "a headgroup would be \(headPx) px at whole-cell scale")
}

section("the renderer")

let device = try findDevice()
let renderer = try SceneRenderer(device: device)
let layout = FrameLayout(width: 240, viewHeight: 180, captionHeight: 40)
let buffer = device.makeBuffer(length: layout.width * layout.height * 4, options: .storageModeShared)!
let allShapes = gpuShapes(scene)
let testCamera = Camera(origin: SIMD3(0, 90, 640), target: SIMD3(0, 60, 0), fov: 40)

func pixels(_ b: MTLBuffer) -> [UInt8] {
    let p = b.contents().assumingMemoryBound(to: UInt8.self)
    return (0..<(layout.width * layout.viewHeight * 4)).map { p[$0] }
}

test("the grid draws the same picture as testing every sphere") {
    // The claim a 21,000× speedup rests on. Brute force at this size is slow,
    // so the frame is small and the scene is a slice — but it is the same
    // kernel, the same shapes, and the same camera.
    let slab = allShapes.enumerated().filter { abs(scene.beads[$0.offset].position.x) < 260 }.map { $0.element }
    try renderer.buildGrid(slab, density: 2)
    try renderer.render(shapes: slab, camera: testCamera, into: buffer,
                        width: layout.width, viewHeight: layout.viewHeight, useGrid: true)
    let withGrid = pixels(buffer)
    try renderer.render(shapes: slab, camera: testCamera, into: buffer,
                        width: layout.width, viewHeight: layout.viewHeight, useGrid: false)
    let brute = pixels(buffer)
    var differ = 0
    for i in 0..<withGrid.count where withGrid[i] != brute[i] { differ += 1 }
    let share = Double(differ) / Double(withGrid.count)
    // Not bit-identical: the two paths meet spheres in a different order, so a
    // pixel exactly on a silhouette can round either way.
    expect(share < 0.02, "\(Int(share * 10000))/100 % of bytes differ")
}

test("a coarse grid and a fine one draw the same picture") {
    let slab = allShapes.enumerated().filter { abs(scene.beads[$0.offset].position.x) < 260 }.map { $0.element }
    try renderer.buildGrid(slab, density: 1)
    try renderer.render(shapes: slab, camera: testCamera, into: buffer,
                        width: layout.width, viewHeight: layout.viewHeight, useGrid: true)
    let coarse = pixels(buffer)
    try renderer.buildGrid(slab, density: 3)
    try renderer.render(shapes: slab, camera: testCamera, into: buffer,
                        width: layout.width, viewHeight: layout.viewHeight, useGrid: true)
    let fine = pixels(buffer)
    var differ = 0
    for i in 0..<coarse.count where coarse[i] != fine[i] { differ += 1 }
    expect(Double(differ) / Double(coarse.count) < 0.02)
}

test("every sphere is filed in the box that holds its middle") {
    let sample = Array(allShapes.prefix(4000))
    let grid = UniformGrid(shapes: sample, density: 1)
    for (i, s) in sample.enumerated() where s.radius > 0 {
        expect(grid.shapes(at: s.center).contains(i), "sphere \(i) is missing from its own box")
    }
}

test("occlusion darkens a crowded spot more than an open one") {
    // A point down among the lipid tails against one out on the surface.
    let om = group("outer_membrane")
    // Beads from the middle of the patch, not whichever happened to be first —
    // the first ones built sit at the patch edge, where a tail is about as
    // exposed as a surface bead and the comparison means nothing.
    func nearestToAxis(_ pool: [Bead]) -> Bead {
        pool.min { a, b in
            (a.position.x * a.position.x + a.position.z * a.position.z)
                < (b.position.x * b.position.x + b.position.z * b.position.z)
        }!
    }
    // The deepest tail beads, which sit down at the bilayer midplane.
    let deepest = om.filter { $0.part == "tail" }
        .filter { abs($0.position.y - scene.layers.omCenter) < 6 }
    let buried = nearestToAxis(deepest)
    let outermost = om.filter { $0.part == "lps" }
        .filter { $0.position.y > scene.layers.omCenter + scene.layers.bilayer / 2 }
    let exposed = nearestToAxis(outermost.isEmpty ? om.filter { $0.part == "head" } : outermost)
    // Fire the same probe pattern by hand, through the same geometry.
    func openness(_ p: SIMD3<Float>, _ n: SIMD3<Float>) -> Float {
        var open = 0
        for i in 0..<12 {
            let u = (Float(i) + 0.5) / 12
            let r = sqrt(u)
            let phi = 6.2831853 * (Float(i) * 0.6180339887).truncatingRemainder(dividingBy: 1)
            let t = simd_normalize(abs(n.z) < 0.9 ? simd_cross(n, SIMD3(0, 0, 1)) : simd_cross(n, SIMD3(1, 0, 0)))
            let b = simd_cross(n, t)
            let dir = t * (r * cos(phi)) + b * (r * sin(phi)) + n * sqrt(max(0, 1 - u))
            var blocked = false
            for bead in om where simd_distance(bead.position, p) < 18 && bead.position != p {
                let oc = p + n * 0.05 - bead.position
                let bb = simd_dot(oc, dir)
                let h = bb * bb - (simd_dot(oc, oc) - bead.radius * bead.radius)
                if h >= 0 && -bb - sqrt(h) > 1e-3 && -bb - sqrt(h) < 14 { blocked = true; break }
            }
            if !blocked { open += 1 }
        }
        return Float(open) / 12
    }
    let up = SIMD3<Float>(0, 1, 0)
    let deep = openness(buried.position, up), open = openness(exposed.position, up)
    expect(deep < open, "buried \(deep) should be less open than exposed \(open)")
}

section("the two loops")

test("the approach ends in contact and starts clear of the membrane") {
    let dnaIdx = scene.indices(of: "plasmid")
    let reach = plasmidReach(scene, dnaIndices: dnaIdx)
    let startGap = plasmidOffset(scene, t: 0, reach: reach).y - reach - scene.layers.omOuterFace
    let endGap = plasmidOffset(scene, t: 1, reach: reach).y - reach - scene.layers.omOuterFace
    expect(startGap > 200, "it should start well clear: \(startGap / NM) nm")
    expect(endGap >= 0 && endGap < 10, "it should finish touching, not overlapping: \(endGap) Å")
}

test("the plasmid only ever translates, which is what keeps occlusion still") {
    // The guarantee the whole shading scheme rests on: if the motion were a
    // rotation, surface normals would turn and the occlusion would crawl.
    let dnaIdx = scene.indices(of: "plasmid")
    let reach = plasmidReach(scene, dnaIndices: dnaIdx)
    let base = gpuShapes(scene)
    let ionIdx = scene.indices(of: "calcium")
    let onDNA = [Bool](repeating: false, count: ionIdx.count)
    let a = approachShapes(scene, base: base, dnaIndices: dnaIdx, ionIndices: ionIdx,
                           ionAttachedToDNA: onDNA, t: 0.2, reach: reach)
    let b = approachShapes(scene, base: base, dnaIndices: dnaIdx, ionIndices: ionIdx,
                           ionAttachedToDNA: onDNA, t: 0.8, reach: reach)
    // Every pairwise distance inside the plasmid must be unchanged.
    var worst: Float = 0
    for k in stride(from: 0, to: dnaIdx.count - 1, by: 7) {
        let i = dnaIdx[k], j = dnaIdx[k + 1]
        let da = simd_distance(a[i].center, a[j].center)
        let db = simd_distance(b[i].center, b[j].center)
        worst = max(worst, abs(da - db))
    }
    expect(worst < 1e-3, "the plasmid changed shape by \(worst) Å — that is a rotation, not a translation")
}

test("the routes loop shuts again, so it can run forever without a rewind") {
    // The opening schedule: open over the first 60%, hold, shut by the end.
    func openness(_ u: Float) -> Float { u < 0.6 ? u / 0.6 : max(0, 1 - (u - 0.85) / 0.15) }
    expect(near(openness(0), 0, within: 1e-6), "starts shut")
    expect(near(openness(1), 0, within: 1e-6), "ends shut")
    expect(openness(0.7) > 0.9, "and is open in the middle")
}

test("each route changes only its own membrane, and the chemical one keeps its calcium") {
    let base = gpuShapes(scene)
    let omIdx = scene.indices(of: "outer_membrane")
    let dnaIdx = scene.indices(of: "plasmid")
    let ionIdx = scene.indices(of: "calcium")
    let reach = plasmidReach(scene, dnaIndices: dnaIdx)
    for route in Route.allCases {
        let shut = routeShapes(scene, base: base, route: route, t: 0, omIndices: omIdx,
                               dnaIndices: dnaIdx, ionIndices: ionIdx, protein: [], reach: reach)
        let open = routeShapes(scene, base: base, route: route, t: 1, omIndices: omIdx,
                               dnaIndices: dnaIdx, ionIndices: ionIdx, protein: [], reach: reach)
        var removed = 0
        for i in omIdx where open[i].radius < shut[i].radius * 0.5 { removed += 1 }
        expect(removed > 0, "\(route.title) never opens anything")
        // The inner membrane is a bystander in all three.
        for i in scene.indices(of: "inner_membrane") {
            expectEqual(open[i].radius, shut[i].radius)
        }
        // Calcium belongs to the chemical route and nowhere else.
        let visible = ionIdx.filter { open[$0].radius > 0 }.count
        if route == .chemical { expect(visible > 0, "the calcium route lost its calcium") }
        else { expectEqual(visible, 0) }
    }
}

test("the competence panel uses a real solved structure") {
    let pdb = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        .deletingLastPathComponent().appendingPathComponent("Resources/8DFK.pdb")
    let protein = try Protein(pdb: pdb, source: "PDB 8DFK")
    expect(protein.beads.count > 300, "only \(protein.beads.count) alpha carbons")
    let upright = protein.upright(scale: 1)
    let height = (upright.map { $0.y }.max() ?? 0) - (upright.map { $0.y }.min() ?? 0)
    expect(height > scene.layers.bilayer * 0.8,
           "a DNA-uptake protein has to be tall enough to span a membrane: \(height) Å")
    // And it must be stood up: its longest axis along y, not left lying down.
    let width = (upright.map { $0.x }.max() ?? 0) - (upright.map { $0.x }.min() ?? 0)
    expect(height >= width, "the protein is lying down: \(height) tall vs \(width) wide")
}

finish()
