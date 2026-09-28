// Tests for step 70, the closed blue hardback. The construction is read off
// the book as drawn — the kernel's own distance function, probed — and held
// to the sources; the cloth's lobe is read back from the GPU and held to
// Estevez & Kulla's formula and to what cloth is not (a mirror); the
// distance function against the definition of a distance; and the picture,
// inset and scale bars off the rendered pixels.
//
// BOOK_MUTANT=noSquares|plasticSheen|pagesNotLeaves breaks the step on
// purpose; `make mutants` requires the suite to fail for each.

import CoreGraphics
import Foundation
import Metal
import simd

let mutant: Mutant = activeMutant
if mutant != .none { print("MUTANT: \(mutant.rawValue)") }

let setup: StillSetup = stillSetup(mutant)
let L: BookLayout = setup.layout
let scene: Scene = setup.scene
let renderer: BookRenderer? = {
    guard let d = try? findDevice() else { return nil }
    return try? BookRenderer(device: d, width: 1920, height: 1080, studio: setup.studio, mutant: mutant)
}()

/// Walk a line of points and return the runs of the innermost material
/// along it: (material, start, end) in the line's parameter.
func runs(from a: SIMD3<Float>, to b: SIMD3<Float>, step: Float) -> [(Int, Float, Float)] {
    guard let r = renderer else { return [] }
    let len: Float = simd_distance(a, b)
    let n: Int = Int(len / step) + 1
    let dir: SIMD3<Float> = (b - a) / len
    let pts: [SIMD3<Float>] = (0..<n).map { a + dir * (Float($0) * step) }
    guard let pr = try? r.probe(pts, scene: scene) else { return [] }
    var out: [(Int, Float, Float)] = []
    for i in 0..<n {
        let m: Int = Int(pr[i].w)
        let t: Float = Float(i) * step
        if let last = out.last, last.0 == m { out[out.count - 1].2 = t } else { out.append((m, t, t)) }
    }
    return out
}

func extent(of m: Material, _ rs: [(Int, Float, Float)]) -> (Float, Float)? {
    let hits = rs.filter { $0.0 == Int(m.rawValue) }
    guard let f = hits.first, let l = hits.last else { return nil }
    return (f.1, l.2)
}

section("the construction, against the sources")

test("the text block is \(leafCount) leaves × \(paperCaliper) mm: \(pageCount) pages, two to a leaf, measured off the drawn book") {
    // Straight down through the middle of the cover, at mid-height.
    let rs = runs(from: SIMD3<Float>(leafWidth / 2, L.top + 1, 0), to: SIMD3<Float>(leafWidth / 2, -0.5, 0), step: 0.002)
    guard let e = extent(of: .paper, rs) else { expect(false, "no paper found"); return }
    let measured: Float = e.1 - e.0
    let wanted: Float = Float(leafCount) * paperCaliper
    print(String(format: "        text block %.3f mm; %d leaves × %.4f mm = %.3f mm", measured, leafCount, paperCaliper, wanted))
    expect(abs(measured - wanted) < 0.01, "text block \(measured) mm, want \(wanted)")
    expect(paperCaliperRange.contains(paperCaliper), "caliper outside Wikipedia's 0.07–0.18 mm")
    expectEqual(pageCount % 16, 0)
}

test("squares of ⅛–¼ in at head, tail and fore-edge: the case overhangs the leaves [Bailey, Bean]") {
    // Head and tail: along z through the front board's cloth, and through the leaves.
    // The case's edge, at the board's mid-thickness (where the cloth's
    // rounded edge reaches farthest), against the leaves' edge.
    let x: Float = leafWidth / 2
    let yb: Float = L.top - shellThickness - boardThickness / 2
    let caseZ = runs(from: SIMD3<Float>(x, yb, 0), to: SIMD3<Float>(x, yb, leafHeight / 2 + 12), step: 0.004)
    let leafZ = runs(from: SIMD3<Float>(x, L.ym, 0), to: SIMD3<Float>(x, L.ym, leafHeight / 2 + 12), step: 0.004)
    let caseX = runs(from: SIMD3<Float>(leafWidth / 2, yb, 0), to: SIMD3<Float>(leafWidth + 12, yb, 0), step: 0.004)
    let leafX = runs(from: SIMD3<Float>(leafWidth / 2, L.y1 - 0.05, 0), to: SIMD3<Float>(leafWidth + 12, L.y1 - 0.05, 0), step: 0.004)
    guard let cz = extent(of: .cloth, caseZ), let lz = extent(of: .paper, leafZ),
          let cx = extent(of: .cloth, caseX), let lx = extent(of: .paper, leafX) else { expect(false, "probe failed"); return }
    let head: Float = cz.1 - lz.1
    let fore: Float = cx.1 - lx.1
    print(String(format: "        head square %.3f mm, fore-edge square %.3f mm; sourced %.3f–%.3f mm", head, fore,
                 squaresRange.lowerBound, squaresRange.upperBound))
    expect(squaresRange.contains(head) || abs(head - squaresRange.lowerBound) < 0.02, "head square \(head) mm")
    expect(squaresRange.contains(fore) || abs(fore - squaresRange.lowerBound) < 0.02, "fore-edge square \(fore) mm")
}

test("the boards are binder's board inside E&R's 0.030–0.300 in, under a cloth shell with glue between") {
    let rs = runs(from: SIMD3<Float>(leafWidth / 2, L.top + 0.5, 0), to: SIMD3<Float>(leafWidth / 2, L.y1 + 0.02, 0), step: 0.001)
    let seq: [Int] = rs.map { $0.0 }
    print("        down through the front board: " + rs.map { String(format: "%d %.3f", $0.0, $0.2 - $0.1) }.joined(separator: ", "))
    expectEqual(Array(seq.prefix(6)), [0, 2, 6, 5, 6, 2].map { Int($0) })
    guard let b = extent(of: .board, rs) else { expect(false, "no board"); return }
    let t: Float = b.1 - b.0
    expect(abs(t - boardThickness) < 0.01, "board \(t) mm")
    expect(boardThicknessRange.contains(t), "board \(t) mm outside E&R's range")
}

test("a French joint: the board stands off the backing shoulder by ⅛–¼ in, and the cloth sinks into a groove there [E&R]") {
    expect(jointGapRange.contains(jointGap))
    guard let r = renderer else { expect(false, "no GPU"); return }
    // The case's outer surface height across the joint, from above.
    func surface(_ x: Float) -> Float {
        var y: Float = L.top + 2
        for _ in 0..<400 {
            guard let d = try? r.probe([SIMD3<Float>(x, y, 0)], scene: scene) else { return .nan }
            if d[0].z < 0.001 { break }
            y -= max(d[0].z * 0.9, 0.0005)
        }
        return y
    }
    let board: Float = surface(jointGap + 5)
    let groove: Float = surface(jointGap * 0.45)
    print(String(format: "        cover surface %.3f mm; in the joint %.3f mm (a groove %.3f mm deep)", board, groove, board - groove))
    expect(board - groove > 0.5, "no groove at the joint")
    // The board's spine edge is the gap away from the shoulder (x = 0).
    let rs = runs(from: SIMD3<Float>(-4, L.top - shellThickness - boardThickness / 2, 0),
                  to: SIMD3<Float>(20, L.top - shellThickness - boardThickness / 2, 0), step: 0.002)
    guard let b = extent(of: .board, rs) else { expect(false, "no board"); return }
    let edge: Float = -4 + b.0 - shellThickness
    print(String(format: "        board's spine edge (with its cloth) at x = %.3f mm; the gap is %.3f mm", edge, jointGap))
    expect(abs(edge - jointGap) < 0.02, "board edge at \(edge)")
}

test("the spine is rounded to a third of a circle, and the fore-edge is concave by the same arc [E&R]") {
    let mid = runs(from: SIMD3<Float>(-8, L.ym, 0), to: SIMD3<Float>(leafWidth + 2, L.ym, 0), step: 0.002)
    guard let e = extent(of: .paper, mid) else { expect(false, "no paper"); return }
    let spineX: Float = -8 + e.0
    let foreX: Float = -8 + e.1
    let sag: Float = -spineX
    let angle: Float = 4 * atan(2 * sag / L.T) * 180 / .pi
    print(String(format: "        spine bulges %.3f mm past the shoulders over a %.3f mm chord: an arc of %.1f°; fore-edge %.3f mm in",
                 sag, L.T, angle, leafWidth - foreX))
    expect(abs(angle - 120) < 1.5, "arc \(angle)°")
    expect(abs((leafWidth - foreX) - sag) < 0.02, "fore-edge not concave by the spine's sagitta")
}

test("headbands at head and tail project slightly beyond the leaves, inside the squares [E&R]") {
    let x: Float = -L.sag + headbandRadius * 0.8 + 0.3
    let rs = runs(from: SIMD3<Float>(x, L.ym, leafHeight / 2 - 4), to: SIMD3<Float>(x, L.ym, leafHeight / 2 + 6), step: 0.002)
    guard let h = extent(of: .headband, rs) else { expect(false, "no headband at the head"); return }
    let beyond: Float = leafHeight / 2 - 4 + h.1 - leafHeight / 2
    print(String(format: "        the headband reaches %.3f mm beyond the head of the leaves; the square is %.3f mm", beyond, L.s))
    expect(beyond > 0.3 && beyond < max(L.s, 0.5), "headband projects \(beyond) mm")
}

test("book cloth at Group B's counts, warp across the cover [E&R, Bailey]") {
    expect(warpPerInch >= 104 && fillingPerInch >= 77)
    print(String(format: "        warp every %.3f mm, filling every %.3f mm", warpPitch, fillingPitch))
}

section("the cloth: a sheen, not a gloss")

test("Estevez & Kulla's D is a normalised distribution: ∫ D cos θm dω = 1 for r = 0.2 … 1") {
    for r in [0.2, 0.4, 0.7, 1.0] {
        var s: Double = 0
        let n: Int = 20_000
        for i in 0..<n {
            let th: Double = (Double(i) + 0.5) / Double(n) * Double.pi / 2
            s += sheenD(cosThetaM: cos(th), r: r) * cos(th) * sin(th) * (Double.pi / 2 / Double(n)) * 2 * Double.pi
        }
        expect(abs(s - 1) < 1e-3, "r \(r): ∫ = \(s)")
    }
}

test("the kernel's lobe is Estevez & Kulla's sheen, read back from the GPU") {
    guard let r = renderer else { expect(false, "no GPU"); return }
    var pairs: [(SIMD3<Float>, SIMD3<Float>)] = []
    var want: [Double] = []
    for tv in stride(from: 10.0, through: 85.0, by: 15.0) {
        for tl in stride(from: 5.0, through: 85.0, by: 20.0) {
            for ph in [0.0, 90.0, 180.0] {
                let v = SIMD3<Double>(sin(tv * .pi / 180), 0, cos(tv * .pi / 180))
                let l = SIMD3<Double>(sin(tl * .pi / 180) * cos(ph * .pi / 180), sin(tl * .pi / 180) * sin(ph * .pi / 180), cos(tl * .pi / 180))
                pairs.append((SIMD3<Float>(v), SIMD3<Float>(l)))
                want.append(sheenBRDF(v: v, l: l))
            }
        }
    }
    guard let g = try? r.brdfOnGPU(pairs) else { expect(false, "probe failed"); return }
    var worst: Double = 0
    for i in 0..<g.count { worst = max(worst, abs(Double(g[i]) - want[i]) / max(want[i], 1e-3)) }
    print(String(format: "        worst relative difference GPU vs formula %.2e over %d pairs", worst, g.count))
    expect(worst < 1e-3, "the kernel's lobe differs from the sheen formula by \(worst)")
}

test("no mirror: at the mirror direction the cloth's lobe is no brighter than 1.5× its value 30° away") {
    guard let r = renderer else { expect(false, "no GPU"); return }
    let t: Float = 45 * .pi / 180
    let v = SIMD3<Float>(sin(t), 0, cos(t))
    let mirror = SIMD3<Float>(-sin(t), 0, cos(t))
    let t2: Float = 75 * .pi / 180
    let off = SIMD3<Float>(-sin(t2), 0, cos(t2))
    let t3: Float = 15 * .pi / 180
    let off2 = SIMD3<Float>(-sin(t3), 0, cos(t3))
    guard let g = try? r.brdfOnGPU([(v, mirror), (v, off), (v, off2)]) else { expect(false, "probe failed"); return }
    print(String(format: "        f(mirror) %.4f; f(30° off, two ways) %.4f, %.4f (a plastic's would spike)", g[0], g[1], g[2]))
    expect(g[0] < 1.5 * max(g[1], g[2]), "a mirror peak: \(g[0]) vs \(g[1]), \(g[2])")
    expect(g[0] < 1 / Float.pi, "the lobe at the mirror is brighter than white diffuse")
}

test("the sheen grows towards grazing: the lobe's albedo rises monotonically as the view tilts") {
    let tab: [Float] = topAlbedoTable(mutant)
    var mono: Bool = true
    for i in 1..<tab.count where tab[i] > tab[i - 1] + 1e-6 { mono = false }
    print(String(format: "        albedo per unit F: %.4f at grazing, %.4f head-on", tab[0], tab[tab.count - 1]))
    expect(mono, "not monotone: \(tab)")
    expect(tab[0] > 5 * tab[tab.count - 1], "no grazing sheen")
}

section("the distance function")

func samplePoints(_ count: Int) -> [SIMD3<Float>] {
    var rng = SystemRandomNumberGenerator()
    var pts: [SIMD3<Float>] = []
    for k in 0..<count {
        if k % 2 == 0 {
            pts.append(SIMD3<Float>(Float.random(in: -12...(leafWidth + 12), using: &rng), Float.random(in: 0.001...(L.top + 8), using: &rng),
                                    Float.random(in: -(leafHeight / 2 + 12)...(leafHeight / 2 + 12), using: &rng)))
        } else {
            // Near the spine, joints, edges and corners, where the shapes meet.
            let x: Float = [Float.random(in: -6...8, using: &rng), Float.random(in: (leafWidth - 6)...(leafWidth + 6), using: &rng)].randomElement(using: &rng)!
            let z: Float = [Float.random(in: -8...8, using: &rng), Float.random(in: (leafHeight / 2 - 5)...(leafHeight / 2 + 6), using: &rng)].randomElement(using: &rng)!
            pts.append(SIMD3<Float>(x, Float.random(in: 0.001...(L.top + 2), using: &rng), z))
        }
    }
    return pts
}

test("outside every surface, no distance claims more room than the ray allows") {
    guard let r = renderer else { expect(false, "no GPU"); return }
    var rng = SystemRandomNumberGenerator()
    let a: [SIMD3<Float>] = samplePoints(200_000)
    var worst: [Int: Float] = [:]
    var worstBook: Float = 0
    for step in [Float(0.003), 0.05, 0.4] {
        let b: [SIMD3<Float>] = a.map { p in
            let d: SIMD3<Float> = simd_normalize(SIMD3<Float>(Float.random(in: -1...1, using: &rng), Float.random(in: -1...1, using: &rng),
                                                              Float.random(in: -1...1, using: &rng)))
            var q: SIMD3<Float> = p + d * step
            q.y = max(q.y, 0.0005)
            return q
        }
        guard let pa = try? r.probe(a, scene: scene), let pb = try? r.probe(b, scene: scene) else { expect(false, "probe failed"); return }
        for i in 0..<a.count {
            // Only points outside count.
            guard pa[i].x > 0, pb[i].x > 0 else { continue }
            let dist: Float = simd_distance(a[i], b[i])
            if Int(pa[i].y) == Int(pb[i].y) {
                let v: Float = abs(pa[i].x - pb[i].x) / dist
                worst[Int(pa[i].y)] = max(worst[Int(pa[i].y)] ?? 0, v)
            }
            if pa[i].z > 0 && pb[i].z > 0 { worstBook = max(worstBook, abs(pa[i].z - pb[i].z) / dist) }
        }
    }
    let report: String = worst.keys.sorted().map { String(format: "material %d %.3f", $0, worst[$0]!) }.joined(separator: ", ")
    print("        worst |Δd|/|Δp| outside: " + report + String(format: "; book alone %.3f; the ray allows %.3f", worstBook, 1 / stepScale))
    for m in [1, 2, 3] { expect(worst[m] != nil, "material \(m) never sampled") }
    for (m, v) in worst { expect(v * stepScale <= 1.0, "material \(m) oversteps: \(v)") }
    expect(worstBook * stepScale <= 1.0, "the book alone oversteps: \(worstBook)")
}

test("the bounding-sphere shortcut never changes the answer") {
    guard let r = renderer else { expect(false, "no GPU"); return }
    let pts: [SIMD3<Float>] = samplePoints(50_000)
    guard let g = try? r.probe(pts, scene: scene) else { expect(false, "probe failed"); return }
    let naive: [Float] = r.lastNaive
    var worst: Float = 0
    for i in 0..<pts.count where naive[i] < 1000 { worst = max(worst, abs(g[i].z - naive[i])) }
    print(String(format: "        worst difference with and without the shortcut: %.2e mm", worst))
    expect(worst < 1e-5, "the shortcut changes the distance by \(worst)")
}

section("the picture")

func pixelLinear(_ img: BookImage, _ x: Int, _ y: Int) -> SIMD3<Double> {
    let p: SIMD4<UInt8> = img.rgba(x, y)
    return SIMD3<Double>(srgbByteToLinear(p.x), srgbByteToLinear(p.y), srgbByteToLinear(p.z))
}

var rendered: Bool = false
if let r = renderer, (try? r.render(scene, camera: setup.camera, samples: 1)) != nil { rendered = true }

test("the frame holds the book and the table; the cloth reads as cloth — matte, with its sheen at the grazing edges") {
    guard rendered, let img = renderer?.image else { expect(false, "render failed"); return }
    var counts: [Int: Int] = [:]
    var topGraze: Double = 0, allGraze: Double = 0, topFace: Double = 0, allFace: Double = 0
    for y in 0..<img.height {
        for x in 0..<img.width {
            let a: SIMD4<Float> = img.seen(x, y)
            counts[Int(a.x), default: 0] += 1
            guard Int(a.x) == 2 else { continue }
            let l: SIMD4<Float> = img.light(x, y)
            if a.z < 0.35 { topGraze += Double(l.x); allGraze += Double(l.y) }
            if a.z > 0.5 { topFace += Double(l.x); allFace += Double(l.y) }
        }
    }
    let total: Double = Double(img.width * img.height)
    let cloth: Double = Double(counts[2] ?? 0) / total
    let paper: Double = Double(counts[3] ?? 0) / total
    let fg: Double = topGraze / max(allGraze, 1e-9)
    let ff: Double = topFace / max(allFace, 1e-9)
    print(String(format: "        cloth %.1f%%, paper edges %.2f%%, table %.1f%%; sheen's share of the light: %.2f at grazing, %.2f facing",
                 cloth * 100, paper * 100, Double(counts[1] ?? 0) / total * 100, fg, ff))
    expect(cloth > 0.08 && cloth < 0.5, "cloth covers \(cloth)")
    expect(paper > 0.002, "the text block's edges are not seen")
    expect(Double(counts[1] ?? 0) / total > 0.3, "not enough table")
    expect(fg > ff, "no grazing sheen")
    expect(ff < 0.5, "the cover facing the camera is mostly reflection: \(ff)")
}

if rendered, let r = renderer { annotate(r.image, setup: setup, renderer: r) }

test("the inset is on the picture: cloth, glue, greyboard, glue, cloth and pastedown, each as thick as the model, to the pixel") {
    guard rendered, let img = renderer?.image else { expect(false, "render failed"); return }
    let rect: CGRect = insetRect(height: img.height)
    let mm: Float = insetMillimetresPerPixel(height: img.height)
    // A column through the board's flat part.
    let col: CGPoint = insetPixel(SIMD2<Float>(insetTestColumnX(L), L.y1), layout: L, height: img.height)
    let bottom: CGPoint = insetPixel(SIMD2<Float>(insetTestColumnX(L), L.y1 - 0.4), layout: L, height: img.height)
    let x: Int = Int(col.x)
    let palette: [(String, SIMD3<Float>)] = [("air", insetAir), ("cloth", insetCloth), ("glue", insetGlue), ("board", insetBoard),
                                             ("pastedown", insetPastedown), ("leaf", insetLeafA), ("leaf", insetLeafB)]
    var seq: [(String, Int)] = []
    for y in Int(rect.minY) + 3..<Int(bottom.y) {
        let p: SIMD4<UInt8> = img.rgba(x, y)
        var best: String = "other"
        var bd: Int = 12
        for (nm, c) in palette {
            let e: SIMD3<Double> = SIMD3<Double>(srgbEncode(Double(c.x)), srgbEncode(Double(c.y)), srgbEncode(Double(c.z))) * 255
            let d: Int = abs(Int(p.x) - Int(e.x.rounded())) + abs(Int(p.y) - Int(e.y.rounded())) + abs(Int(p.z) - Int(e.z.rounded()))
            if d < bd { bd = d; best = nm }
        }
        if let last = seq.last, last.0 == best { seq[seq.count - 1].1 += 1 } else { seq.append((best, 1)) }
    }
    // Text in the air above the cover crosses the column; merge the air.
    var clean: [(String, Int)] = []
    for run in seq where run.0 != "other" {
        if let last = clean.last, last.0 == run.0, run.0 == "air" { clean[clean.count - 1].1 += run.1 } else { clean.append(run) }
    }
    print("        down the inset: " + clean.prefix(9).map { "\($0.0) \($0.1)" }.joined(separator: ", "))
    let names: [String] = clean.map { $0.0 }
    expectEqual(Array(names.prefix(7)), ["air", "cloth", "glue", "board", "glue", "cloth", "pastedown"])
    if clean.count >= 7 {
        let want: [Float] = [clothThickness, adhesiveThickness, boardThickness, adhesiveThickness, clothThickness, pastedownThickness]
        for i in 0..<6 {
            let px: Float = want[i] / mm
            expect(abs(Float(clean[i + 1].1) - px) <= 1.5, "\(clean[i + 1].0): \(clean[i + 1].1) px, want \(px)")
        }
    }
}

test("the scale bars are true: 50 mm at the cover's middle, projected, and 1 mm in the inset") {
    guard rendered, let img = renderer?.image else { expect(false, "render failed"); return }
    let a: SIMD2<Float> = project(setup.centre, width: img.width, height: img.height, cam: setup.camera)
    let b: SIMD2<Float> = project(setup.centre + setup.camera.right * mainBarMillimetres, width: img.width, height: img.height, cam: setup.camera)
    let projected: Float = simd_distance(a, b)
    let bar: ScaleBar = mainScaleBar(setup, width: img.width, height: img.height)
    let yy: Int = Int(bar.y + 1.5)
    var painted: Int = 0
    for x in max(Int(bar.x) - 4, 0)..<min(Int(bar.x + bar.pixels) + 5, img.width) {
        let p: SIMD4<UInt8> = img.rgba(x, yy)
        if Int(p.x) + Int(p.y) + Int(p.z) < 3 * 60 { painted += 1 }
    }
    let insetBar: ScaleBar = insetScaleBar(height: img.height)
    let insetExpected: Float = insetBarMillimetres / insetMillimetresPerPixel(height: img.height)
    print(String(format: "        50 mm: projected %.1f px, bar %.1f px, painted %d px; inset 1 mm: %.1f px, bar %.1f px",
                 projected, Float(bar.pixels), painted, insetExpected, Float(insetBar.pixels)))
    expect(abs(Float(bar.pixels) - projected) < 0.5, "bar \(bar.pixels) vs projected \(projected)")
    expect(abs(Float(painted) - projected) <= 2.5, "bar painted \(painted) px")
    expect(abs(Float(insetBar.pixels) - insetExpected) < 0.01)
}

test("the cut is on the book, in view, left of the inset") {
    guard rendered, let img = renderer?.image else { expect(false, "render failed"); return }
    let p: SIMD2<Float> = project(SIMD3<Float>(L.boardX1 - 1.0, L.y1 + 1.5, 0), width: img.width, height: img.height, cam: setup.camera)
    let r: CGRect = insetRect(height: img.height)
    expect(p.x > 0 && p.x < Float(r.minX) && p.y > 0 && p.y < Float(img.height), "cut point off the view: \(p)")
    // No part of the book under the inset or the caption.
    var covered: Int = 0
    for y in stride(from: Int(r.minY), to: Int(r.maxY), by: 4) {
        for x in stride(from: Int(r.minX), to: min(Int(r.maxX), img.width), by: 4) where Int(img.seen(x, y).x) >= 2 { covered += 1 }
    }
    expect(covered == 0, "the inset hides \(covered) samples of the book")
}

finish()
