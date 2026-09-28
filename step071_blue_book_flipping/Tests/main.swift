// Tests for step 71, the blue book turning its own pages. The book is held
// to step 70's sources at frame 0; every turning leaf is measured off the
// primitives the GPU draws and held to what paper can do (bend, not
// stretch); nothing may pass through anything; the motion must run forward
// and be seen to run; and the distance function, the scale bar and the
// caption are checked as in step 70.
//
// BOOK_MUTANT=stretchyPage|frozen|rewind|ghostPage|noSquares breaks the
// step on purpose; `make mutants` requires the suite to fail for each.

import CoreGraphics
import Foundation
import Metal
import simd

let mutant: Mutant = activeMutant
if mutant != .none { print("MUTANT: \(mutant.rawValue)") }

let theShot: Shot = shot()
let device: MTLDevice? = try? findDevice()
let renderer: BookRenderer? = {
    guard let d = device else { return nil }
    return try? BookRenderer(device: d, width: 160, height: 90, studio: theShot.studio, mutant: mutant)
}()

func frameTime(_ f: Int) -> Float { Float(f) / framesPerSecond }
let allFrames: [Int] = Array(0..<frameCount)
let states: [BookState] = allFrames.map { bookState(at: frameTime($0), mutant) }

// MARK: - the closed book, against step 70's sources

section("frame 0: step 70's book, closed")

let scene0: Scene = buildScene(states[0], mutant)

func runs(_ scene: Scene, from a: SIMD3<Float>, to b: SIMD3<Float>, step: Float) -> [(Int, Float, Float)] {
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

let closedTop: Float = 2 * boardStack + spineLength

test("the text block is \(leafCount) leaves × \(paperCaliper) mm, measured off the drawn book") {
    let rs = runs(scene0, from: SIMD3<Float>(leafWidth / 2, closedTop + 1, 0), to: SIMD3<Float>(leafWidth / 2, -0.5, 0), step: 0.002)
    guard let e = extent(of: .paper, rs) else { expect(false, "no paper"); return }
    let measured: Float = e.1 - e.0
    let wanted: Float = Float(leafCount) * paperCaliper
    print(String(format: "        text block %.3f mm; %d × %.4f mm = %.3f mm", measured, leafCount, paperCaliper, wanted))
    expect(abs(measured - wanted) < 0.01, "text block \(measured) mm")
    expect(paperCaliperRange.contains(paperCaliper))
}

test("squares of ⅛–¼ in at head and fore-edge, and boards inside E&R's range, as in step 70") {
    let yb: Float = closedTop - boardStack / 2
    let caseZ = runs(scene0, from: SIMD3<Float>(80, yb, 0), to: SIMD3<Float>(80, yb, leafHeight / 2 + 12), step: 0.004)
    let leafZ = runs(scene0, from: SIMD3<Float>(80, boardStack + 1, 0), to: SIMD3<Float>(80, boardStack + 1, leafHeight / 2 + 12), step: 0.004)
    let caseX = runs(scene0, from: SIMD3<Float>(80, yb, 0), to: SIMD3<Float>(leafWidth + 12, yb, 0), step: 0.004)
    let leafX = runs(scene0, from: SIMD3<Float>(80, boardStack + 1, 0), to: SIMD3<Float>(leafWidth + 12, boardStack + 1, 0), step: 0.004)
    guard let cz = extent(of: .cloth, caseZ), let lz = extent(of: .paper, leafZ),
          let cx = extent(of: .cloth, caseX), let lx = extent(of: .paper, leafX) else { expect(false, "probe failed"); return }
    let head: Float = cz.1 - lz.1
    let fore: Float = cx.1 - lx.1
    let down = runs(scene0, from: SIMD3<Float>(80, closedTop + 0.5, 0), to: SIMD3<Float>(80, closedTop - boardStack - 0.2, 0), step: 0.001)
    let cloth: Float = extent(of: .cloth, down).map { $0.1 - $0.0 } ?? 0
    let board: Float = cloth - 2 * shellThickness
    print(String(format: "        head square %.3f mm, fore-edge square %.3f mm; board %.3f mm under its cloth", head, fore, board))
    expect(squaresRange.contains(head) || abs(head - squaresRange.lowerBound) < 0.02, "head square \(head)")
    expect(squaresRange.contains(fore) || abs(fore - squaresRange.lowerBound) < 0.02, "fore-edge square \(fore)")
    expect(abs(board - boardThickness) < 0.02 && boardThicknessRange.contains(board), "board \(board)")
}

// MARK: - paper bends but does not stretch

section("paper bends but does not stretch")

/// A drawn leaf's centre line, rebuilt from the primitives the GPU gets:
/// arc pieces (centre, middle direction, radius, half-angle) and straight
/// ones (end points), in order.
struct DrawnPiece {
    var points: (Int) -> SIMD2<Float>
    var length: Float
    var start: SIMD2<Float>
    var end: SIMD2<Float>
    var z0: Float
    var z1: Float
}

func drawnPiece(_ g: GPrim) -> DrawnPiece {
    if g.info.x == PrimKind.arc.rawValue {
        // By its chord, as the kernel reads it: the point at angle φ from
        // the middle is M + x̂ r sin φ + ŷ r (cos φ − cos h).
        let m = SIMD2<Float>(g.a.x, g.a.y)
        let mid = SIMD2<Float>(g.a.z, g.a.w)
        let xh = SIMD2<Float>(mid.y, -mid.x)
        let r: Float = g.b.x
        let h: Float = g.b.z
        let f: (Float) -> SIMD2<Float> = { phi in
            let y: Float = 2 * r * sin((h + phi) / 2) * sin((h - phi) / 2)
            return m + xh * (r * sin(phi)) + mid * y
        }
        return DrawnPiece(points: { i in f(-h + 2 * h * Float(i) / 400) }, length: 2 * h * r,
                          start: f(-h), end: f(h), z0: g.z.x, z1: g.z.y)
    }
    let a = SIMD2<Float>(g.a.x, g.a.y)
    let b = SIMD2<Float>(g.a.z, g.a.w)
    return DrawnPiece(points: { i in a + (b - a) * (Float(i) / 400) }, length: simd_distance(a, b), start: a, end: b,
                      z0: g.z.x, z1: g.z.y)
}

/// The drawn leaves of a scene: leaf index → its pieces, in order, with the
/// arcs oriented to run from the root.
func drawnLeaves(_ scene: Scene, _ s: BookState) -> [Int: [DrawnPiece]] {
    var out: [Int: [DrawnPiece]] = [:]
    for (i, nm) in scene.names.enumerated() where nm.hasPrefix("leaf ") {
        let parts = nm.split(separator: " ")
        guard parts.count >= 2, let k = Int(parts[1]) else { continue }
        out[k, default: []].append(drawnPiece(scene.prims[i]))
    }
    // Orient each chain from the root outwards.
    for (k, ps) in out {
        let d = SIMD2<Float>(cos(s.psi), sin(s.psi))
        var cursor: SIMD2<Float> = s.jb + d * ((Float(k) + 0.5) * paperCaliper)
        var chain: [DrawnPiece] = []
        for var p in ps {
            if simd_distance(p.end, cursor) < simd_distance(p.start, cursor) {
                let f = p.points
                let st = p.start
                p = DrawnPiece(points: { i in f(400 - i) }, length: p.length, start: p.end, end: st, z0: p.z0, z1: p.z1)
            }
            chain.append(p)
            cursor = p.end
        }
        out[k] = chain
    }
    return out
}

let isoFrames: [Int] = allFrames.filter { $0 % 2 == 0 }

test("every turning leaf, as drawn, is exactly as wide and as tall as the flat leaf: 152.4 × 228.6 mm, no stretch") {
    var worstLen: Float = 0
    var worstGap: Float = 0
    var worstRoot: Float = 0
    var worstArea: Float = 0
    var count: Int = 0
    for f in isoFrames {
        let s: BookState = states[f]
        let sc: Scene = buildScene(s, mutant)
        for (k, chain) in drawnLeaves(sc, s) {
            count += 1
            let len: Float = chain.reduce(0) { $0 + $1.length }
            worstLen = max(worstLen, abs(len - leafWidth))
            // Sum of the little quads' areas over the sampled surface.
            var area: Float = 0
            for p in chain {
                var prev: SIMD2<Float> = p.points(0)
                for i in 1...400 {
                    let q: SIMD2<Float> = p.points(i)
                    area += simd_distance(prev, q) * (p.z1 - p.z0)
                    prev = q
                }
            }
            worstArea = max(worstArea, abs(area - leafWidth * leafHeight) / (leafWidth * leafHeight))
            for i in 1..<chain.count { worstGap = max(worstGap, simd_distance(chain[i - 1].end, chain[i].start)) }
            let d = SIMD2<Float>(cos(s.psi), sin(s.psi))
            let root: SIMD2<Float> = s.jb + d * ((Float(k) + 0.5) * paperCaliper)
            worstRoot = max(worstRoot, simd_distance(chain[0].start, root))
            for p in chain { worstLen = max(worstLen, abs((p.z1 - p.z0) - leafHeight) > 1e-3 ? 999 : 0) }
        }
    }
    print(String(format: "        %d drawn flying leaves over %d frames: worst width error %.2e mm, area error %.2e (relative), joins %.2e mm, root %.2e mm",
                 count, isoFrames.count, worstLen, worstArea, worstGap, worstRoot))
    expect(count > 200, "too few flying leaves sampled: \(count)")
    expect(worstLen < 2e-3, "a leaf's width changed by \(worstLen) mm")
    expect(worstArea < 1e-4, "a leaf's area changed by \(worstArea)")
    expect(worstGap < 2e-3, "a leaf's pieces don't join: \(worstGap) mm")
    expect(worstRoot < 2e-3, "a leaf left its root: \(worstRoot) mm")
}

test("every drawn leaf has zero Gaussian curvature, and its rulings run square to its curve (it unrolls flat)") {
    // Sample the surface S(u, v) = c(u) + v ẑ off the drawn pieces and take
    // K = (LN − M²)/(EG − F²) by finite differences.
    var worstK: Double = 0
    var worstF: Double = 0
    var samples: Int = 0
    for f in isoFrames where f % 8 == 0 {
        let s: BookState = states[f]
        let sc: Scene = buildScene(s, mutant)
        for (_, chain) in drawnLeaves(sc, s) {
            for p in chain {
                for i in stride(from: 5, to: 395, by: 30) {
                    let h: Double = Double(p.length) / 400
                    guard h > 1e-4 else { continue }
                    let dv: Double = 1.0
                    func S(_ di: Int, _ vv: Double) -> SIMD3<Double> {
                        let q: SIMD2<Float> = p.points(i + di)
                        return SIMD3<Double>(Double(q.x), Double(q.y), vv)
                    }
                    let su: SIMD3<Double> = (S(1, 0) - S(-1, 0)) / (2 * h)
                    let sv: SIMD3<Double> = (S(0, dv) - S(0, -dv)) / (2 * dv)
                    let suu: SIMD3<Double> = (S(1, 0) - 2 * S(0, 0) + S(-1, 0)) / (h * h)
                    let svv: SIMD3<Double> = (S(0, dv) - 2 * S(0, 0) + S(0, -dv)) / (dv * dv)
                    let suv: SIMD3<Double> = (S(1, dv) - S(1, -dv) - S(-1, dv) + S(-1, -dv)) / (4 * h * dv)
                    let n: SIMD3<Double> = simd_normalize(simd_cross(su, sv))
                    let E: Double = simd_dot(su, su), F: Double = simd_dot(su, sv), G: Double = simd_dot(sv, sv)
                    let L2: Double = simd_dot(suu, n), M: Double = simd_dot(suv, n), N: Double = simd_dot(svv, n)
                    let K: Double = (L2 * N - M * M) / (E * G - F * F)
                    worstK = max(worstK, abs(K))
                    worstF = max(worstF, abs(F) / (E * G).squareRoot())
                    samples += 1
                }
            }
        }
    }
    print(String(format: "        %d surface samples: worst |K| %.2e per mm², worst cos(ruling, curve) %.2e", samples, worstK, worstF))
    expect(samples > 1000, "too few samples")
    expect(worstK < 1e-9, "Gaussian curvature \(worstK)")
    expect(worstF < 1e-6, "rulings not square to the curve")
}

test("the leaves at rest are the same curves: each as wide as the leaf, and the stacks drawn are exactly their union") {
    guard let r = renderer else { expect(false, "no GPU"); return }
    var worstLen: Float = 0
    var inside: Int = 0
    var total: Int = 0
    var outsideWrong: Int = 0
    for f in [0, 40, Int(riffleStart * framesPerSecond) + 30, Int(riffleStart * framesPerSecond) + 60, Int(closeStart * framesPerSecond) + 10] {
        let s: BookState = states[f]
        let sc: Scene = buildScene(s, mutant)
        var pts: [SIMD3<Float>] = []
        var outPts: [SIMD3<Float>] = []
        for k in 0..<leafCount where s.tau[k] <= 0 || s.tau[k] >= 1 {
            let ps: [Piece] = leafPieces(k, s, mutant)
            worstLen = max(worstLen, abs(ps.reduce(0) { $0 + $1.length } - leafWidth))
            for q in leafPoints(ps, step: 2.0) { pts.append(SIMD3<Float>(q.x, q.y, Float.random(in: -100...100))) }
        }
        // Just outside the outermost leaf of each stack: a thickness beyond it.
        let lying: Int = s.tau.filter { $0 <= 0 }.count
        let turned: Int = s.tau.filter { $0 >= 1 }.count
        if lying > 0 {
            let b: BoardFrame = backFrame(s)
            for a in stride(from: Float(20), to: leafWidth - 10, by: 5) { outPts.append(SIMD3<Float>(b.world(a, Float(lying) * paperCaliper + 0.05).x, b.world(a, Float(lying) * paperCaliper + 0.05).y, 0)) }
        }
        if turned > 0 {
            let fr: BoardFrame = frontFrame(s)
            for a in stride(from: Float(20), to: leafWidth - 10, by: 5) { outPts.append(SIMD3<Float>(fr.world(a, Float(turned) * paperCaliper + 0.05).x, fr.world(a, Float(turned) * paperCaliper + 0.05).y, 0)) }
        }
        guard let g = try? r.probe(pts, scene: sc), let go = try? r.probe(outPts, scene: sc) else { expect(false, "probe failed"); return }
        for i in 0..<pts.count { total += 1; if g[i].z <= 1e-4 { inside += 1 } }
        for i in 0..<outPts.count where go[i].z < 0.02 && Int(go[i].w) == Int(Material.paper.rawValue) { outsideWrong += 1 }
    }
    print(String(format: "        resting leaves: worst width error %.2e mm; %d of %d centre-line points inside the drawn stacks; %d points above a stack wrongly inside",
                 worstLen, inside, total, outsideWrong))
    expect(worstLen < 2e-3)
    expectEqual(inside, total)
    expectEqual(outsideWrong, 0)
}

// MARK: - nothing passes through anything

section("nothing passes through anything")

/// The signed distance from p to a board's slab (2D rounded box), in mm.
func boxDistance(_ p: SIMD2<Float>, _ b: BoardBox, radius r: Float) -> Float {
    let d: SIMD2<Float> = p - b.centre
    let q = SIMD2<Float>(abs(simd_dot(d, b.axis)), abs(simd_dot(d, SIMD2<Float>(-b.axis.y, b.axis.x)))) - b.half + r
    return simd_length(simd_max(q, .zero)) + min(max(q.x, q.y), 0) - r
}

test("no leaf comes within a leaf's thickness of another, nor into a board, in any frame") {
    let t: Float = paperCaliper
    var worstLeaf: Float = .infinity
    var worstBoard: Float = .infinity
    var where_: String = ""
    let rBoard: Float = boardEdgeRadius + shellThickness
    for f in allFrames {
        let s: BookState = states[f]
        // Every leaf's centre line, sampled finely near the spine, and
        // bucketed on a grid.
        var curves: [[SIMD2<Float>]] = []
        for k in 0..<leafCount { curves.append(leafPoints(leafPieces(k, s, mutant), step: 0.25)) }
        let cell: Float = 0.6
        var grid: [SIMD2<Int32>: [(Int, Int)]] = [:]
        for (k, c) in curves.enumerated() {
            for (i, p) in c.enumerated() { grid[SIMD2<Int32>(Int32((p.x / cell).rounded(.down)), Int32((p.y / cell).rounded(.down))), default: []].append((k, i)) }
        }
        func segDist(_ p: SIMD2<Float>, _ a: SIMD2<Float>, _ b: SIMD2<Float>) -> Float {
            let ab: SIMD2<Float> = b - a
            let h: Float = max(0, min(1, simd_dot(p - a, ab) / max(simd_dot(ab, ab), 1e-12)))
            return simd_distance(p, a + ab * h)
        }
        for (k, c) in curves.enumerated() {
            for p in c {
                let cx = Int32((p.x / cell).rounded(.down)), cy = Int32((p.y / cell).rounded(.down))
                for dx in -1...1 { for dy in -1...1 {
                    guard let bucket = grid[SIMD2<Int32>(cx + Int32(dx), cy + Int32(dy))] else { continue }
                    for (j, i) in bucket where j != k {
                        // Two resting leaves are checked every tenth frame;
                        // anything with a flying leaf, every frame.
                        let restK: Bool = s.tau[k] <= 0 || s.tau[k] >= 1
                        let restJ: Bool = s.tau[j] <= 0 || s.tau[j] >= 1
                        if restK && restJ && f % 10 != 0 { continue }
                        let other: [SIMD2<Float>] = curves[j]
                        var d: Float = .infinity
                        if i + 1 < other.count { d = min(d, segDist(p, other[i], other[i + 1])) }
                        if i > 0 { d = min(d, segDist(p, other[i - 1], other[i])) }
                        let need: Float = t * Float(abs(j - k) >= 1 ? 1 : 0)
                        let slack: Float = d - need
                        if slack < worstLeaf { worstLeaf = slack; where_ = "frame \(f), leaves \(k) and \(j)" }
                    }
                } }
                // Boards and pastedowns: the leaf's surface (half a thickness
                // out from its centre line) must stay outside.
                for fr in [backFrame(s), frontFrame(s)] {
                    let bd: Float = boxDistance(p, boardBox(fr, mutant), radius: rBoard) - t / 2
                    worstBoard = min(worstBoard, bd)
                }
            }
        }
    }
    print(String(format: "        closest two leaves come: %.4f mm beyond touching (%@); closest a leaf comes to a board: %.4f mm",
                 worstLeaf, where_, worstBoard))
    expect(worstLeaf > -0.002, "leaves pass through each other: \(worstLeaf) mm at \(where_)")
    expect(worstBoard > -0.002, "a leaf enters a board: \(worstBoard) mm")
}

test("the boards never pass through each other or the table") {
    var worst: Float = .infinity
    var lowest: Float = .infinity
    let rBoard: Float = boardEdgeRadius + shellThickness
    for s in states {
        let a: BoardBox = boardBox(backFrame(s), mutant)
        let b: BoardBox = boardBox(frontFrame(s), mutant)
        for (x, y) in [(a, b), (b, a)] {
            let ax = x.axis
            let ay = SIMD2<Float>(-ax.y, ax.x)
            for i in 0...200 {
                let u: Float = Float(i) / 200 * 2 - 1
                for (sa, sb) in [(u, Float(1)), (u, -1), (Float(1), u), (Float(-1), u)] {
                    let p: SIMD2<Float> = x.centre + ax * (x.half.x * sa) + ay * (x.half.y * sb)
                    worst = min(worst, boxDistance(p, y, radius: rBoard))
                    lowest = min(lowest, p.y)
                }
            }
        }
    }
    print(String(format: "        the boards come within %.3f mm of each other; lowest board point %.4f mm", worst, lowest))
    expect(worst > -0.002, "boards overlap by \(-worst) mm")
    expect(lowest > -0.01, "a board is \(-lowest) mm into the table")
}

// MARK: - the motion runs forward, and is seen to

section("the motion")

test("forward only: every leaf turns once, front to back; the covers only open and close; the book ends closed on its other side") {
    var backward: Int = 0
    for k in 0..<leafCount {
        for i in 1..<states.count where states[i].tau[k] < states[i - 1].tau[k] - 1e-6 { backward += 1 }
    }
    var spineBack: Int = 0
    for i in 1..<states.count where states[i].psi < states[i - 1].psi - 1e-6 { spineBack += 1 }
    let openEnd: Int = Int(riffleStart * framesPerSecond)
    var coverBack: Int = 0
    for i in 1..<openEnd where states[i].alpha < states[i - 1].alpha - 1e-6 { coverBack += 1 }
    for i in Int(closeStart * framesPerSecond) + 1..<states.count where states[i].beta < states[i - 1].beta - 1e-6 { coverBack += 1 }
    let last: BookState = bookState(at: fadeStart, mutant)
    let first: BookState = states[0]
    print(String(format: "        leaf steps backwards: %d; spine %d; covers %d; at the dissolve %d of %d leaves turned, back cover at %.1f°",
                 backward, spineBack, coverBack, last.tau.filter { $0 >= 1 }.count, leafCount, last.beta * 180 / .pi))
    expectEqual(backward, 0)
    expectEqual(spineBack, 0)
    expectEqual(coverBack, 0)
    expect(first.tau.allSatisfy { $0 == 0 } && first.alpha == 0, "frame 0 is not the closed book")
    expect(last.tau.allSatisfy { $0 >= 1 } && abs(last.beta - .pi) < 1e-4, "the book does not end closed on its other side")
    var fadeBack: Int = 0
    for i in 1..<states.count where states[i].fade < states[i - 1].fade - 1e-6 { fadeBack += 1 }
    expectEqual(fadeBack, 0)
    expect(states[states.count - 1].fade > 0.9, "the loop's last frame is not nearly frame 0")
    // Pages turn in order: the first page first.
    for k in 1..<leafCount { expect(leafStart(k) < leafStart(k - 1), "leaf \(k) out of order") }
}

test("the motion is visible: at least 2% of the picture changes between any two frames a second apart") {
    guard let r = renderer else { expect(false, "no GPU"); return }
    let w: Int = r.width, h: Int = r.height
    var images: [Int: [UInt8]] = [:]
    let step: Int = 4
    var first: [UInt8] = []
    for f in stride(from: 0, to: frameCount, by: step) {
        guard (try? r.render(buildScene(states[f], mutant), camera: theShot.camera, samples: 1)) != nil else { expect(false, "render failed"); return }
        let p = r.image.pixels.contents().assumingMemoryBound(to: UInt8.self)
        var img: [UInt8] = Array(UnsafeBufferPointer(start: p, count: w * h * 4))
        if f == 0 { first = img }
        let fw: Float = states[f].fade
        if fw > 0 {
            for i in 0..<img.count {
                let v: Float = Float(img[i]) + (Float(first[i]) - Float(img[i])) * fw
                img[i] = UInt8(max(min(v.rounded(), 255), 0))
            }
        }
        images[f] = img
    }
    let second: Int = Int(framesPerSecond)
    var worst: Double = 1
    var worstAt: Int = -1
    for f in stride(from: 0, to: frameCount, by: step) {
        let g: Int = (f + second) % frameCount
        let gg: Int = (g / step) * step
        guard let a = images[f], let b = images[gg] else { continue }
        var changed: Int = 0
        for i in 0..<(w * h) {
            let o: Int = i * 4
            let d: Int = abs(Int(a[o]) - Int(b[o])) + abs(Int(a[o + 1]) - Int(b[o + 1])) + abs(Int(a[o + 2]) - Int(b[o + 2]))
            if d > 24 { changed += 1 }
        }
        let frac: Double = Double(changed) / Double(w * h)
        if frac < worst { worst = frac; worstAt = f }
    }
    print(String(format: "        least change over a second: %.1f%% of pixels (from frame %d)", worst * 100, worstAt))
    expect(worst >= 0.02, "only \(worst * 100)% of the picture changes in the second from frame \(worstAt)")
}

test("the caption says the motion is fiction and the bending is not") {
    let all: String = ([captionTitle] + captionLines).joined(separator: " ")
    expect(all.contains("fiction"), "no 'fiction' in the caption")
    expect(all.contains("never stretched"))
}

// MARK: - the distance function

section("the distance function")

func samplePoints(_ s: BookState, _ count: Int) -> [SIMD3<Float>] {
    var rng = SystemRandomNumberGenerator()
    var pts: [SIMD3<Float>] = []
    for i in 0..<count {
        if i % 2 == 0 {
            pts.append(SIMD3<Float>(Float.random(in: -175...175, using: &rng), Float.random(in: 0.001...165, using: &rng),
                                    Float.random(in: -130...130, using: &rng)))
        } else {
            // Near the spine and the joints, where everything meets.
            let c: SIMD2<Float> = (s.jb + s.jf) / 2
            pts.append(SIMD3<Float>(c.x + Float.random(in: -14...14, using: &rng), max(c.y + Float.random(in: -12...14, using: &rng), 0.001),
                                    Float.random(in: -120...120, using: &rng)))
        }
    }
    return pts
}

test("outside every surface, in every kind of frame, no distance claims more room than the ray allows") {
    guard let r = renderer else { expect(false, "no GPU"); return }
    var rng = SystemRandomNumberGenerator()
    var worst: Float = 0
    var worstShortcut: Float = 0
    var sampled: Set<Int> = []
    let picks: [Int] = [0, 30, Int(riffleStart * framesPerSecond) + 20, Int(riffleStart * framesPerSecond) + 50,
                        Int(riffleStart * framesPerSecond) + 80, Int(closeStart * framesPerSecond) + 14]
    for f in picks {
        let s: BookState = states[min(f, states.count - 1)]
        let sc: Scene = buildScene(s, mutant)
        let a: [SIMD3<Float>] = samplePoints(s, 60_000)
        for step in [Float(0.003), 0.05, 0.5] {
            let b: [SIMD3<Float>] = a.map { p in
                let d: SIMD3<Float> = simd_normalize(SIMD3<Float>(Float.random(in: -1...1, using: &rng), Float.random(in: -1...1, using: &rng),
                                                                  Float.random(in: -1...1, using: &rng)))
                var q: SIMD3<Float> = p + d * step
                q.y = max(q.y, 0.0005)
                return q
            }
            guard let pa = try? r.probe(a, scene: sc) else { expect(false, "probe failed"); return }
            let naive: [Float] = r.lastNaive
            guard let pb = try? r.probe(b, scene: sc) else { expect(false, "probe failed"); return }
            for i in 0..<a.count {
                if naive[i] < 1000 { worstShortcut = max(worstShortcut, abs(pa[i].z - naive[i])) }
                guard pa[i].x > 0, pb[i].x > 0 else { continue }
                sampled.insert(Int(pa[i].y))
                if Int(pa[i].y) == Int(pb[i].y) { worst = max(worst, abs(pa[i].x - pb[i].x) / simd_distance(a[i], b[i])) }
                if pa[i].z > 0 && pb[i].z > 0 { worst = max(worst, abs(pa[i].z - pb[i].z) / simd_distance(a[i], b[i])) }
            }
        }
    }
    print(String(format: "        worst |Δd|/|Δp| outside %.3f (the ray allows %.3f); the bounding-sphere shortcut changes nothing: %.1e mm; materials %@",
                 worst, 1 / stepScale, worstShortcut, sampled.sorted().map { String($0) }.joined(separator: ",")))
    expect(worst * stepScale <= 1, "a distance oversteps: \(worst)")
    expect(worstShortcut < 1e-5, "the shortcut changes the distance by \(worstShortcut)")
    for m in [1, 2, 3] { expect(sampled.contains(m), "material \(m) never sampled") }
}

// MARK: - the picture

section("the picture")

test("the scale bar is true: 50 mm at the spine's head, projected") {
    let w: Int = 640, h: Int = 360
    let a: SIMD2<Float> = project(theShot.centre, width: w, height: h, cam: theShot.camera)
    let b: SIMD2<Float> = project(theShot.centre + theShot.camera.right * barMillimetres, width: w, height: h, cam: theShot.camera)
    let bar: ScaleBar = scaleBar(theShot, width: w, height: h)
    print(String(format: "        50 mm projects to %.2f px; the bar is %.2f px", simd_distance(a, b), Float(bar.pixels)))
    expect(abs(Float(bar.pixels) - simd_distance(a, b)) < 0.5)
}

test("the book stays in the frame and clear of the caption and the scale bar in every frame") {
    let w: Int = 640, h: Int = 360
    var worst: String = ""
    var ok: Bool = true
    for (f, s) in states.enumerated() {
        var pts: [SIMD3<Float>] = []
        for k in stride(from: 0, to: leafCount, by: 5) {
            let e: SIMD2<Float> = leafPieces(k, s, mutant).last!.end
            for z in [-leafHeight / 2, leafHeight / 2] { pts.append(SIMD3<Float>(e.x, e.y, z)) }
        }
        for fr in [backFrame(s), frontFrame(s)] {
            for a in [Float(0), boardEnd(mutant)] {
                let p: SIMD2<Float> = fr.world(a, 0)
                for z in [-caseHalfHeight(mutant), caseHalfHeight(mutant)] { pts.append(SIMD3<Float>(p.x, p.y, z)) }
            }
        }
        for p in pts {
            let q: SIMD2<Float> = project(p, width: w, height: h, cam: theShot.camera)
            let inFrame: Bool = q.x > 4 && q.x < Float(w - 4) && q.y > 4 && q.y < Float(h - 4)
            let underBar: Bool = q.x < 150 && q.y > Float(h - 30)
            let cap: CGRect = captionBox(height: h).insetBy(dx: -4, dy: -4)
            let underCaption: Bool = cap.contains(CGPoint(x: CGFloat(q.x), y: CGFloat(q.y)))
            if !inFrame || underCaption || underBar { ok = false; worst = "frame \(f): \(q)" }
        }
    }
    expect(ok, "the book leaves the frame or runs under the caption: \(worst)")
}

finish()
