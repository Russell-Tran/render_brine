// Tests for step 52. The fibre is measured through the GPU's own distance
// function — the one that draws the picture — for its cross-section, its
// flatness, its twist and the hand of that twist either side of a reversal.
// The knit is checked as geometry (every loop joined to the next in its
// row, touching the row below, passing through it), and against the drawn
// distance. The molecule is checked atom by atom against the crystal.
//
// COTTON_MUTANT=roundFibre|alphaLinks|noReversal|gappyKnit breaks it on
// purpose; `make mutants` requires the suite to fail for each.

import CoreGraphics
import Foundation
import Metal
import simd

let mutant: Mutant = Mutant.fromEnvironment

final class DeviceBox {
    let device: MTLDevice? = try? findDevice()
}
let gpu = DeviceBox()

let scene: Scene? = try? Scene(mutant: mutant)
let library: MTLLibrary? = {
    guard let d = gpu.device, let s = scene else { return nil }
    do { return try makeLibrary(d, s) } catch { print("library: \(error)"); return nil }
}()

func probe(_ pts: [SIMD3<Float>]) -> [Probe]? {
    guard let d = gpu.device, let s = scene, let l = library else { return nil }
    return try? probeScene(pts, scene: s, library: l, on: d)
}

func deg(_ r: Float) -> Float { r * 180 / Float.pi }

section("the scene")

test("the scene builds and its kernel compiles") {
    expect(scene != nil, "could not build the scene (is Resources/ there?)")
    expect(library != nil, "the kernel did not compile")
}

// MARK: - the fibre

section("one fibre: a flat, twisted ribbon with a collapsed lumen")

/// A station on the fuzz fibre: centre, tangent, and an in-plane frame
/// (width direction W and thickness direction N from the ribbon's data).
struct Station { var c: SIMD3<Float>; var t: SIMD3<Float>; var w: SIMD3<Float>; var n: SIMD3<Float>; var s: Float }

/// Halfway along segment k→k+1, clear of the mitre planes at the vertices
/// (inside a ribbon the distance to a shared mitre face reads ~0 there).
func station(_ r: Ribbon, vertex k: Int) -> Station {
    let t: SIMD3<Float> = simd_normalize(r.points[k + 1] - r.points[k])
    let wm: SIMD3<Float> = r.widths[k] + r.widths[k + 1]
    let w: SIMD3<Float> = simd_normalize(wm - t * simd_dot(wm, t))
    return Station(c: (r.points[k] + r.points[k + 1]) / 2, t: t, w: w, n: simd_cross(t, w), s: (r.arc[k] + r.arc[k + 1]) / 2)
}

/// The vertex nearest arc length s.
func vertex(_ r: Ribbon, at s: Float) -> Int {
    var best: Int = 1
    for k in 1..<(r.points.count - 1) where abs(r.arc[k] - s) < abs(r.arc[best] - s) { best = k }
    return best
}

/// The drawn cross-section at a station: a grid in the plane ⟂ the fibre,
/// probed on the GPU. Returns wall area, lumen area, outer perimeter (by
/// Crofton's formula), the extent along the widest direction, and the full
/// thickness through the fibre's axis.
struct Section { var wall: Float; var lumen: Float; var perimeter: Float; var width: Float; var thickness: Float; var bow: Float }

func measureSection(_ st: Station) -> Section? {
    let h: Float = 0.1
    let n: Int = 340
    var pts: [SIMD3<Float>] = []
    pts.reserveCapacity(n * n)
    for i in 0..<n {
        for j in 0..<n {
            let u: Float = (Float(i) - Float(n) / 2 + 0.5) * h
            let v: Float = (Float(j) - Float(n) / 2 + 0.5) * h
            pts.append(st.c + st.w * u + st.n * v)
        }
    }
    guard let r = probe(pts) else { return nil }
    var solid: [Bool] = []
    var wall: Int = 0, lumen: Int = 0
    for p in r {
        let inWall: Bool = p.insetRibbons < 0
        let inLumen: Bool = !inWall && p.insetSub == 2
        if inWall { wall += 1 }
        if inLumen { lumen += 1 }
        solid.append(inWall || inLumen)
    }
    var changes: Int = 0
    for i in 0..<n { for j in 1..<n where solid[i * n + j] != solid[i * n + j - 1] { changes += 1 } }
    for j in 0..<n { for i in 1..<n where solid[i * n + j] != solid[(i - 1) * n + j] { changes += 1 } }
    // Extent along the widest direction: second moments of the solid set.
    var mu = SIMD2<Float>(0, 0)
    var count: Float = 0
    for i in 0..<n { for j in 0..<n where solid[i * n + j] {
        mu += SIMD2<Float>(Float(i), Float(j)); count += 1
    } }
    guard count > 0 else { return Section(wall: 0, lumen: 0, perimeter: 0, width: 0, thickness: 0, bow: 0) }
    mu /= count
    var sxx: Float = 0, sxy: Float = 0, syy: Float = 0
    for i in 0..<n { for j in 0..<n where solid[i * n + j] {
        let dx: Float = Float(i) - mu.x, dy: Float = Float(j) - mu.y
        sxx += dx * dx; sxy += dx * dy; syy += dy * dy
    } }
    let angle: Float = 0.5 * atan2(2 * sxy, sxx - syy)
    let major = SIMD2<Float>(cos(angle), sin(angle))
    var lo: Float = .infinity, hi: Float = -.infinity
    for i in 0..<n { for j in 0..<n where solid[i * n + j] {
        let a: Float = simd_dot(SIMD2<Float>(Float(i), Float(j)), major)
        lo = min(lo, a); hi = max(hi, a)
    } }
    // Thickness: along the minor axis, through the fibre's axis (the grid
    // centre), the run of solid samples containing it.
    let minor = SIMD2<Float>(-major.y, major.x)
    var run: Int = 0
    for sgn in [Float(1), Float(-1)] {
        var k: Int = sgn > 0 ? 0 : 1
        while true {
            let q: SIMD2<Float> = SIMD2<Float>(Float(n) / 2 - 0.5, Float(n) / 2 - 0.5) + minor * (Float(k) * sgn)
            let i: Int = Int(q.x.rounded()), j: Int = Int(q.y.rounded())
            guard i >= 0, i < n, j >= 0, j < n, solid[i * n + j] else { break }
            run += 1
            k += 1
        }
    }
    // Bow: how far the section's centroid sits off the fibre's axis, across
    // the width — zero for a flat strip or a round fibre, the kidney's
    // curvature otherwise.
    let off: SIMD2<Float> = mu - SIMD2<Float>(Float(n) / 2 - 0.5, Float(n) / 2 - 0.5)
    return Section(wall: Float(wall) * h * h, lumen: Float(lumen) * h * h, perimeter: Float(changes) * h * Float.pi / 4,
                   width: (hi - lo + 1) * h, thickness: Float(run) * h, bow: abs(simd_dot(off, minor)) * h)
}

let freeStart: Float = (scene?.yarn.peelArc ?? 0) + 60

test("the cross-section is LaFave et al.'s: perimeter 59.1 µm, wall 135.4 µm², lumen 13.5 µm², as drawn") {
    guard let s = scene else { expect(false); return }
    let f: Ribbon = s.yarn.fuzz
    guard let m = measureSection(station(f, vertex: vertex(f, at: freeStart + 40))) else { expect(false, "probe failed"); return }
    print(String(format: "        drawn: perimeter %.1f µm, wall %.1f µm², lumen %.2f µm²", m.perimeter, m.wall, m.lumen))
    expect(abs(m.perimeter - measuredPerimeter) / measuredPerimeter < 0.04, "perimeter \(m.perimeter)")
    expect(abs(m.wall - measuredWallArea) / measuredWallArea < 0.04, "wall area \(m.wall)")
    expect(abs(m.lumen - measuredLumenArea) / measuredLumenArea < 0.12, "lumen area \(m.lumen)")
}

test("it is a flat ribbon: width over thickness above 3.5, kidney-curved") {
    guard let s = scene else { expect(false); return }
    let f: Ribbon = s.yarn.fuzz
    guard let m = measureSection(station(f, vertex: vertex(f, at: freeStart + 40))) else { expect(false, "probe failed"); return }
    let ratio: Float = m.width / max(m.thickness, 1e-3)
    print(String(format: "        width %.1f µm, thickness %.1f µm through the axis: ratio %.2f", m.width, m.thickness, ratio))
    expect(ratio > 3.5, "width/thickness only \(ratio)")
    // A kidney, not a flat strip: the section's arms bend to one side, so
    // its centroid sits off the axis.
    print(String(format: "        centroid %.2f µm off the axis, across the width: kidney-curved", m.bow))
    expect(m.bow > 0.8, "the section is not curved: centroid \(m.bow) µm off the axis")
}

test("the lumen is there: hollow on the fibre's axis, cell wall either side of it") {
    guard let s = scene else { expect(false); return }
    let f: Ribbon = s.yarn.fuzz
    let st: Station = station(f, vertex: vertex(f, at: freeStart + 100))
    let wallMid: Float = (s.yarn.section.halfThickness + s.yarn.section.lumenHalfThickness) / 2
    guard let r = probe([st.c, st.c + st.n * wallMid, st.c - st.n * wallMid]) else { expect(false, "probe failed"); return }
    print(String(format: "        axis %.3f µm (sub %d), wall above %.3f, below %.3f", r[0].insetRibbons, r[0].insetSub,
                 r[1].insetRibbons, r[2].insetRibbons))
    expect(r[0].insetRibbons > 0 && r[0].insetSub == 2, "no lumen at the axis")
    expect(r[1].insetRibbons < 0 && r[2].insetRibbons < 0, "no wall either side of the lumen")
    expect(s.yarn.section.lumenHalfThickness * 2 < 0.25 * s.yarn.section.halfThickness * 2, "the lumen is not collapsed")
}

/// The ribbon's orientation along its free part, measured from the drawn
/// distance: at each station, probe a ring of radius 9 µm in the plane ⟂ the
/// fibre, and take the axial mean direction of the samples inside the wall,
/// against a frame carried along the fibre without spin.
func measuredTwist(_ f: Ribbon, from s0: Float, to s1: Float, every ds: Float) -> (s: [Float], beta: [Float], inside: [Float])? {
    let k0: Int = vertex(f, at: s0), k1: Int = vertex(f, at: s1)
    let stride: Int = max(Int((ds / fibreStep).rounded()), 1)
    var ks: [Int] = []
    var k: Int = k0
    while k <= k1 { ks.append(k); k += stride }
    let ring: Int = 360
    let rho: Float = 9
    var pts: [SIMD3<Float>] = []
    var frames: [(SIMD3<Float>, SIMD3<Float>)] = []
    var ref: SIMD3<Float> = perpendicular(to: station(f, vertex: ks[0]).t)
    var lastT: SIMD3<Float> = station(f, vertex: ks[0]).t
    for kk in ks {
        let st: Station = station(f, vertex: kk)
        // carry `ref` from lastT to st.t
        let axis: SIMD3<Float> = simd_cross(lastT, st.t)
        let sn: Float = simd_length(axis)
        if sn > 1e-7 { ref = simd_quatf(angle: atan2(sn, simd_dot(lastT, st.t)), axis: axis / sn).act(ref) }
        ref = simd_normalize(ref - st.t * simd_dot(ref, st.t))
        lastT = st.t
        let e2: SIMD3<Float> = simd_cross(st.t, ref)
        frames.append((ref, e2))
        for a in 0..<ring {
            let phi: Float = 2 * Float.pi * Float(a) / Float(ring)
            pts.append(st.c + (ref * cos(phi) + e2 * sin(phi)) * rho)
        }
    }
    guard let r = probe(pts) else { return nil }
    var ss: [Float] = [], beta: [Float] = [], inside: [Float] = []
    var prev: Float = 0
    for (i, kk) in ks.enumerated() {
        var c2: Float = 0, s2: Float = 0, n: Float = 0
        for a in 0..<ring where r[i * ring + a].insetRibbons < 0 {
            let phi: Float = 2 * Float.pi * Float(a) / Float(ring)
            c2 += cos(2 * phi); s2 += sin(2 * phi); n += 1
        }
        var b: Float = 0.5 * atan2(s2, c2)
        if i > 0 {
            while b - prev > Float.pi / 2 { b -= Float.pi }
            while b - prev < -Float.pi / 2 { b += Float.pi }
        }
        prev = b
        ss.append(station(f, vertex: kk).s); beta.append(b); inside.append(n / Float(ring))
    }
    return (ss, beta, inside)
}

/// Least-squares slope of y against x.
func slope(_ x: [Float], _ y: [Float]) -> Float {
    let n: Float = Float(x.count)
    let mx: Float = x.reduce(0, +) / n, my: Float = y.reduce(0, +) / n
    var sxy: Float = 0, sxx: Float = 0
    for i in 0..<x.count { sxy += (x[i] - mx) * (y[i] - my); sxx += (x[i] - mx) * (x[i] - mx) }
    return sxy / max(sxx, 1e-9)
}

test("it twists at the convolution rate, and at the reversal the hand of the twist really flips") {
    guard let s = scene else { expect(false); return }
    let f: Ribbon = s.yarn.fuzz
    let revAt: Float = s.yarn.peelArc + fuzzReversalAfter
    guard let m = measuredTwist(f, from: freeStart, to: f.length - 12, every: 8) else { expect(false, "probe failed"); return }
    // A ribbon: at radius 9 µm the ring crosses the wall on two short arcs.
    let meanInside: Float = m.inside.reduce(0, +) / Float(m.inside.count)
    expect(meanInside > 0.03 && meanInside < 0.5, "the ring is \(meanInside * 100)% inside: not a flat ribbon")
    // Runs of one hand: the sign of the local slope, over 40 µm windows.
    var signs: [Int] = []
    var flips: [Float] = []
    let w: Int = 5
    for i in w..<(m.s.count - w) {
        let sl: Float = slope(Array(m.s[(i - w)...(i + w)]), Array(m.beta[(i - w)...(i + w)]))
        guard abs(sl) > 0.3 * twistRate else { continue }
        let sg: Int = sl > 0 ? 1 : -1
        if let last = signs.last, last != sg { flips.append(m.s[i]) }
        signs.append(sg)
    }
    // Away from the reversal, the rate is the convolution rate.
    let before: [Int] = m.s.indices.filter { m.s[$0] < revAt - 4 * reversalHalfWidth }
    let after: [Int] = m.s.indices.filter { m.s[$0] > revAt + 4 * reversalHalfWidth }
    let rb: Float = before.count > 3 ? slope(before.map { m.s[$0] }, before.map { m.beta[$0] }) : 0
    let ra: Float = after.count > 3 ? slope(after.map { m.s[$0] }, after.map { m.beta[$0] }) : 0
    print(String(format: "        twist %.4f rad/µm before the reversal, %.4f after (convolutions: %.4f); hand flips at %@ µm (reversal drawn at %.0f)",
                 rb, ra, twistRate, flips.map { String(format: "%.0f", $0) }.joined(separator: ", "), revAt))
    expect(abs(abs(rb) - twistRate) < 0.15 * twistRate, "twist before \(rb)")
    expect(abs(abs(ra) - twistRate) < 0.15 * twistRate, "twist after \(ra)")
    expect(rb * ra < 0, "the same hand both sides: no reversal")
    expect(flips.count == 1, "\(flips.count) changes of hand along the free fibre")
    if let fl = flips.first { expect(abs(fl - revAt) < 3 * reversalHalfWidth, "the hand flips at \(fl), not at the reversal") }
}

test("every fibre in the picture reverses, at about Gould & Seagull's 17.9 per cm") {
    guard let s = scene else { expect(false); return }
    var total: Float = 0
    var count: Int = 0
    for r in s.yarn.surface + s.hairs.map({ $0.ribbon }) {
        total += r.length / r.scale
        count += r.reversals.count
    }
    let perCm: Float = Float(count) / (total / 10_000)
    print(String(format: "        %d reversals over %.1f mm of fibre: %.1f per cm", count, total / 1000, perCm))
    expect(abs(perCm - reversalsPerCentimetre) / reversalsPerCentimetre < 0.35, "\(perCm) reversals per cm")
}

// MARK: - the yarn

section("the yarn")

test("~96 fibres in a section: 19.68 tex of 0.206-tex fibres; radius 86.9 µm at packing 0.6") {
    print(String(format: "        fibre %.4f tex; %.1f fibres; yarn radius %.1f µm", fibreTex, fibresPerYarnSection, yarnRadius))
    expect(abs(fibreTex - 0.2058) < 0.001)
    expect(abs(fibresPerYarnSection - 95.6) < 0.5)
    expect(abs(yarnRadius - 86.9) < 0.3)
}

test("its surface fibres follow a right-handed (Z) helix at tan α = 2πRT, and never touch each other") {
    guard let s = scene else { expect(false); return }
    let want: Float = helixAngle(radius: surfaceRadius(s.yarn.section))
    var worst: Float = 0
    var hand: Float = 0
    for r in s.yarn.surface {
        for k in stride(from: 1, to: r.points.count - 1, by: 7) {
            let t: SIMD3<Float> = simd_normalize(r.points[k + 1] - r.points[k - 1])
            worst = max(worst, abs(acos(min(abs(simd_dot(t, yarnAxis)), 1)) - want))
            let q: SIMD3<Float> = r.points[k] - yarnOrigin
            let radial: SIMD3<Float> = simd_normalize(q - yarnAxis * simd_dot(q, yarnAxis))
            hand += simd_dot(simd_cross(radial, t), yarnAxis) * (simd_dot(t, yarnAxis) > 0 ? 1 : -1)
        }
    }
    print(String(format: "        helix %.2f° (want %.2f°), worst off %.3f°; hand %@", deg(want), deg(want), deg(worst), hand > 0 ? "right (Z)" : "left (S)"))
    expect(deg(worst) < 0.5)
    expect(hand > 0, "an S-twist yarn")
    // Neighbours stay two reaches apart, square to their direction.
    let a: [SIMD3<Float>] = s.yarn.surface[0].points, b: [SIMD3<Float>] = s.yarn.surface[1].points
    let gap: Float = polylineDistance(Array(a[40..<60]), Array(b[20..<80]))
    print(String(format: "        neighbouring centre lines %.1f µm apart; two reaches %.1f µm", gap, 2 * s.yarn.section.reach))
    expect(gap > 2 * s.yarn.section.reach, "neighbouring surface fibres can touch")
}

// MARK: - the knit

section("the knit: plain jersey, every loop held")

test("stitch density is Hossain et al.'s: 32.2 wales and 47.8 courses per inch; loop length within 2% of 2.77 mm") {
    guard let s = scene else { expect(false); return }
    let wpi: Float = 25.4 / s.knit.w, cpi: Float = 25.4 / s.knit.c
    print(String(format: "        %.1f wales, %.1f courses per inch; loop %.3f mm of yarn (measured %.2f)", wpi, cpi, s.knit.loopLength, measuredLoopLength))
    expect(abs(wpi - walesPerInch) < 0.05 && abs(cpi - coursesPerInch) < 0.05)
    expect(abs(s.knit.loopLength - measuredLoopLength) / measuredLoopLength < 0.02)
}

test("each course is one yarn: every loop ends exactly where the next in its row begins, heading the same way") {
    guard let s = scene else { expect(false); return }
    let k: Knit = s.knit
    var worstGap: Float = 0, worstTurn: Float = 0
    for j in -2...2 {
        for i in -3...3 {
            let end: SIMD3<Float> = k.point(i, j, Float.pi)
            let start: SIMD3<Float> = k.point(i + 1, j, -Float.pi)
            worstGap = max(worstGap, simd_distance(end, start))
            let te: SIMD3<Float> = simd_normalize(k.point(i, j, Float.pi) - k.point(i, j, Float.pi - 1e-3))
            let ts: SIMD3<Float> = simd_normalize(k.point(i + 1, j, -Float.pi + 1e-3) - k.point(i + 1, j, -Float.pi))
            worstTurn = max(worstTurn, acos(min(simd_dot(te, ts), 1)))
        }
    }
    print(String(format: "        worst join gap %.2e mm, turn %.3f°", worstGap, deg(worstTurn)))
    expect(worstGap < 1e-5)
    expect(deg(worstTurn) < 0.5)
}

/// Projected crossings (in the fabric's plane) between two polylines, with
/// which one is in front (smaller knit z is nearer the face).
func crossings(_ A: [SIMD3<Float>], _ B: [SIMD3<Float>]) -> [(sA: Int, aFront: Bool)] {
    var out: [(Int, Bool)] = []
    for i in 1..<A.count {
        let p0 = SIMD2<Float>(A[i - 1].x, A[i - 1].y), p1 = SIMD2<Float>(A[i].x, A[i].y)
        for j in 1..<B.count {
            let q0 = SIMD2<Float>(B[j - 1].x, B[j - 1].y), q1 = SIMD2<Float>(B[j].x, B[j].y)
            let d1: SIMD2<Float> = p1 - p0, d2: SIMD2<Float> = q1 - q0
            let den: Float = d1.x * d2.y - d1.y * d2.x
            guard abs(den) > 1e-12 else { continue }
            let w: SIMD2<Float> = q0 - p0
            let s: Float = (w.x * d2.y - w.y * d2.x) / den
            let u: Float = (w.x * d1.y - w.y * d1.x) / den
            guard s >= 0, s < 1, u >= 0, u < 1 else { continue }
            let za: Float = A[i - 1].z + (A[i].z - A[i - 1].z) * s
            let zb: Float = B[j - 1].z + (B[j].z - B[j - 1].z) * u
            out.append((i, za < zb))
        }
    }
    return out
}

test("each loop holds the row below: touching it (centre lines 2r apart, no gap, no overlap) and passing through it") {
    guard let s = scene else { expect(false); return }
    let k: Knit = s.knit
    let line: [SIMD3<Float>] = k.polyline(160)
    func loop(_ i: Int, _ j: Int) -> [SIMD3<Float>] { line.map { $0 + SIMD3<Float>(Float(i) * k.w, Float(j) * k.c, 0) } }
    let upper: [SIMD3<Float>] = loop(0, 1)
    var nearest: Float = .infinity
    for i in -1...1 { nearest = min(nearest, polylineDistance(upper, loop(i, 0))) }
    print(String(format: "        closest approach to the row below %.4f mm; 2r = %.4f mm", nearest, 2 * k.r))
    expect(nearest >= 2 * k.r * 0.995, "loops pass into each other: \(nearest)")
    expect(nearest <= 2 * k.r * 1.01, "a gap between the rows: \(nearest) against \(2 * k.r)")
    // Nothing else closer: the rows two apart, and the rest of its own row.
    for i in -1...1 { expect(polylineDistance(upper, loop(i, -1)) > 2 * k.r, "a loop reaches two rows down") }
    let course: [SIMD3<Float>] = loop(-1, 0) + loop(1, 0)
    let own: [SIMD3<Float>] = Array(loop(0, 0)[28..<133])   // clear of the joins, > 1.5πr of yarn away
    expect(polylineDistance(own, course) > 2 * k.r, "a loop overlaps its neighbours in the row")
    // Passing through: the upper loop crosses the lower loop four times in
    // the fabric's plane — behind its legs low down, in front of its head
    // higher up — so it goes in at the back and out at the front.
    let c: [(sA: Int, aFront: Bool)] = crossings(upper, loop(0, 0)).sorted { $0.sA < $1.sA }
    print("        crossings with the loop below, along the upper loop: \(c.map { $0.aFront ? "front" : "back" })")
    expectEqual(c.count, 4)
    if c.count == 4 { expect(!c[0].aFront && c[1].aFront && c[2].aFront && !c[3].aFront, "the loop does not pass through the one below") }
}

test("the drawn knit is that geometry: the GPU's distance is zero at the contact and on the yarn, matching the CPU's") {
    guard let s = scene else { expect(false); return }
    let k: Knit = s.knit
    let line: [SIMD3<Float>] = k.polyline()
    // The contact: the midpoint of the closest pair between loop (0,1) and the row below.
    var best: Float = .infinity
    var mid = SIMD3<Float>(0, 0, 0)
    let upper: [SIMD3<Float>] = k.polyline(400).map { $0 + SIMD3<Float>(0, k.c, 0) }
    for i in -1...1 {
        let lower: [SIMD3<Float>] = k.polyline(400).map { $0 + SIMD3<Float>(Float(i) * k.w, 0, 0) }
        for a in upper { for b in lower where simd_distance(a, b) < best { best = simd_distance(a, b); mid = (a + b) / 2 } }
    }
    var pts: [SIMD3<Float>] = [k.world(mid)]
    var rng = Seeded(7)
    for _ in 0..<2000 {
        pts.append(SIMD3<Float>(rng.uniform(-2, 2), rng.uniform(-0.3, 0.4), rng.uniform(-3, 1)))
    }
    guard let r = probe(pts) else { expect(false, "probe failed"); return }
    print(String(format: "        drawn distance at the contact %.5f mm", r[0].knit))
    expect(abs(r[0].knit) < 0.002, "the contact is not on both tubes: \(r[0].knit)")
    var worst: Float = 0
    for i in 1..<pts.count { worst = max(worst, abs(r[i].knit - knitDistance(pts[i], k, line))) }
    expect(worst < 1e-4, "GPU and CPU knit distances differ by \(worst) mm")
}

test("fuzz: every hair grows out of a yarn, clears the knit, and stands up from the face") {
    guard let s = scene else { expect(false); return }
    let line: [SIMD3<Float>] = s.knit.polyline()
    expect(s.hairs.count > 40, "only \(s.hairs.count) hairs")
    var rooted: Int = 0, clear: Int = 0, up: Int = 0
    for h in s.hairs {
        let pts: [SIMD3<Float>] = h.ribbon.points
        if knitDistance(pts[0], s.knit, line) < 0 { rooted += 1 }
        if pts.dropFirst(4).allSatisfy({ knitDistance($0, s.knit, line) > 0 }) { clear += 1 }
        if pts[pts.count - 1].y > pts[0].y { up += 1 }
    }
    print("        \(s.hairs.count) hairs: \(rooted) rooted in the yarn, \(clear) clear of it, \(up) rising")
    expectEqual(rooted, s.hairs.count)
    expectEqual(clear, s.hairs.count)
    expect(up >= s.hairs.count * 9 / 10)
}

// MARK: - cellulose

section("cellulose, from the crystal")

test("four glucose units from COD 4114994: C24H42O21, every atom its valence, four rings") {
    guard let s = scene else { expect(false); return }
    let m: Molecule = s.molecule
    let f: [String: Int] = m.formula
    expect(f["C"] == 24 && f["H"] == 42 && f["O"] == 21 && f.count == 3, "formula \(f)")
    let v: [Int] = m.valences
    for (i, a) in m.atoms.enumerated() {
        let want: Int = a.element == "C" ? 4 : (a.element == "O" ? 2 : 1)
        expect(v[i] == want, "\(a.name) of unit \(a.residue) has \(v[i]) bonds")
    }
    var parent: [Int] = Array(0..<m.atoms.count)
    func find(_ x: Int) -> Int { var x = x; while parent[x] != x { x = parent[x] }; return x }
    for b in m.bonds { let a = find(b.0), c = find(b.1); if a != c { parent[a] = c } }
    let pieces: Int = Set((0..<m.atoms.count).map { find($0) }).count
    expectEqual(pieces, 1)
    expectEqual(m.bonds.count - m.atoms.count + pieces, chainUnits)
}

test("bond lengths are the crystal's: C–C 1.49–1.56 Å, C–O 1.40–1.45 Å, C–H 0.95–0.99 Å") {
    guard let s = scene else { expect(false); return }
    let m: Molecule = s.molecule
    for (a, b) in m.bonds {
        let ea: String = m.atoms[a].element, eb: String = m.atoms[b].element
        let d: Float = simd_distance(m.atoms[a].position, m.atoms[b].position)
        let pair: String = [ea, eb].sorted().joined()
        let range: ClosedRange<Float>
        switch pair {
        case "CC": range = 1.49...1.56
        case "CO": range = 1.40...1.45
        case "CH": range = 0.95...0.99
        default: range = (hydroxylLength - 0.01)...(hydroxylLength + 0.01)
        }
        expect(range.contains(d), "\(m.atoms[a].name)\(m.atoms[a].residue)–\(m.atoms[b].name)\(m.atoms[b].residue) \(d) Å")
    }
}

/// The side of the ring a substituent X on ring atom A is on: the sign of
/// (prev − A) × (next − A) · (X − A), with the ring walked O5 C1 C2 C3 C4 C5.
func face(_ m: Molecule, _ n: Int, atom: String, prev: String, next: String, x: (String, Int)) -> Float {
    guard let a = m.index(atom, n), let p = m.index(prev, n), let q = m.index(next, n), let xi = m.index(x.0, x.1) else { return 0 }
    let A: SIMD3<Float> = m.atoms[a].position
    return simd_dot(simd_cross(m.atoms[p].position - A, m.atoms[q].position - A), m.atoms[xi].position - A)
}

test("every link is β(1→4): the bridging oxygen on C6's face of the ring, equatorial, and bonded to the next unit's C4") {
    guard let s = scene else { expect(false); return }
    let m: Molecule = s.molecule
    for n in 0..<chainUnits {
        let o: (String, Int) = n + 1 < chainUnits ? ("O4", n + 1) : ("O1", n)
        let f1: Float = face(m, n, atom: "C1", prev: "O5", next: "C2", x: o)
        let f5: Float = face(m, n, atom: "C5", prev: "C4", next: "O5", x: ("C6", n))
        expect(f1 * f5 > 0, "unit \(n): the link is on the far face from C6 — α")
        // Equatorial: the C1–O bond lies near the ring's mean plane.
        let ring: [Int] = ["C1", "C2", "C3", "C4", "C5", "O5"].compactMap { m.index($0, n) }
        var c = SIMD3<Float>(0, 0, 0)
        for i in ring { c += m.atoms[i].position }
        c /= Float(ring.count)
        var normal = SIMD3<Float>(0, 0, 0)
        for k in 0..<ring.count {
            normal += simd_cross(m.atoms[ring[k]].position - c, m.atoms[ring[(k + 1) % ring.count]].position - c)
        }
        normal = simd_normalize(normal)
        if let c1 = m.index("C1", n), let oi = m.index(o.0, o.1) {
            let bond: SIMD3<Float> = simd_normalize(m.atoms[oi].position - m.atoms[c1].position)
            let tilt: Float = deg(acos(abs(simd_dot(bond, normal))))
            if n == 0 { print(String(format: "        unit 0: C1–O %.0f° from the ring's normal (equatorial > 55°, axial < 30°)", tilt)) }
            expect(tilt > 55, "unit \(n): C1–O axial, \(tilt)° from the normal — α")
        }
        if n + 1 < chainUnits, let o4 = m.index("O4", n + 1), let c4 = m.index("C4", n + 1), let c1 = m.index("C1", n) {
            let dC4: Float = simd_distance(m.atoms[o4].position, m.atoms[c4].position)
            let dC1: Float = simd_distance(m.atoms[o4].position, m.atoms[c1].position)
            expect(abs(dC4 - 1.44) < 0.03 && abs(dC1 - 1.414) < 0.03, "link \(n): C4′–O4′ \(dC4), O4′–C1 \(dC1)")
        }
    }
}

test("each glucose is turned 180° from the next, 5.19 Å further along: the 2₁ screw") {
    guard let s = scene else { expect(false); return }
    let m: Molecule = s.molecule
    var dirs: [SIMD3<Float>] = [], cents: [SIMD3<Float>] = []
    guard let a0 = m.index("C1", 0), let a3 = m.index("C1", chainUnits - 1) else { expect(false); return }
    let axis: SIMD3<Float> = simd_normalize(m.atoms[a3].position - m.atoms[a0].position)
    for n in 0..<chainUnits {
        guard let c1 = m.index("C1", n), let c4 = m.index("C4", n), let c6 = m.index("C6", n) else { continue }
        let ring: SIMD3<Float> = (m.atoms[c1].position + m.atoms[c4].position) / 2
        var d: SIMD3<Float> = m.atoms[c6].position - ring
        d -= axis * simd_dot(d, axis)
        dirs.append(simd_normalize(d)); cents.append(ring)
    }
    for n in 1..<dirs.count {
        let turn: Float = deg(acos(max(min(simd_dot(dirs[n], dirs[n - 1]), 1), -1)))
        let rise: Float = simd_dot(cents[n] - cents[n - 1], axis)
        if n == 1 { print(String(format: "        unit to unit: turned %.1f°, %.2f Å along the chain", turn, rise)) }
        expect(turn > 165, "units \(n - 1)–\(n) turned only \(turn)°")
        expect(abs(abs(rise) - 5.19) < 0.05, "rise \(rise) Å")
    }
    for n in 2..<dirs.count { expect(simd_dot(dirs[n], dirs[n - 2]) > cos(15 * Float.pi / 180), "units \(n - 2) and \(n) differ") }
}

// MARK: - distances

section("the distance functions are distances")

/// Worst over-report per material over pairs of nearby points both outside
/// every surface: |d(a) − d(b)| / |a − b|.
func worstOverReport(_ a: [SIMD3<Float>], step: Float, inset: Bool) -> [Int: Float] {
    var rng = Seeded(99)
    var b: [SIMD3<Float>] = []
    for p in a {
        let d: SIMD3<Float> = simd_normalize(SIMD3<Float>(rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1)))
        b.append(p + d * step)
    }
    guard let pa = probe(a), let pb = probe(b) else { return [:] }
    var worst: [Int: Float] = [:]
    for i in 0..<a.count {
        let (da, db, ma, mb): (Float, Float, Int, Int) = inset
            ? (pa[i].insetDistance, pb[i].insetDistance, pa[i].insetMaterial, pb[i].insetMaterial)
            : (pa[i].mainDistance, pb[i].mainDistance, pa[i].mainMaterial, pb[i].mainMaterial)
        guard ma == mb, da > 0, db > 0 else { continue }
        let v: Float = abs(da - db) / step
        if ProcessInfo.processInfo.environment["COTTON_DEBUG"] != nil && v * stepScale > 1 { print("        m\(ma) \(v) at \(a[i]) d \(da)") }
        worst[ma] = max(worst[ma] ?? 0, v)
    }
    return worst
}

func merge(_ a: inout [Int: Float], _ b: [Int: Float]) { for (k, v) in b { a[k] = max(a[k] ?? 0, v) } }

test("main view: the backing, the yarn tubes and the fuzz never claim more room than the ray allows") {
    guard let s = scene else { expect(false); return }
    var rng = Seeded(3)
    var box: [SIMD3<Float>] = []
    for _ in 0..<400_000 { box.append(SIMD3<Float>(rng.uniform(-3.5, 3.5), rng.uniform(-1.3, 1.6), rng.uniform(-5, 2))) }
    var w: [Int: Float] = worstOverReport(box, step: 0.004, inset: false)
    // A band 0.1–1.5 mm out from every hair: where the ribbon's far field
    // and the chunk culling meet (the first version jumped 4.9× here).
    var far: [SIMD3<Float>] = []
    for h in s.hairs {
        for p in h.ribbon.points {
            let d: SIMD3<Float> = simd_normalize(SIMD3<Float>(rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1)))
            far.append(p + d * rng.uniform(0.1, 1.5))
        }
    }
    merge(&w, worstOverReport(far, step: 0.002, inset: false))
    // Packed round the hairs, where the twisted ribbons are.
    var near: [SIMD3<Float>] = []
    for h in s.hairs {
        for p in h.ribbon.points {
            for _ in 0..<10 { near.append(p + SIMD3<Float>(rng.uniform(-0.03, 0.03), rng.uniform(-0.03, 0.03), rng.uniform(-0.03, 0.03))) }
        }
    }
    merge(&w, worstOverReport(near, step: 0.0015, inset: false))
    // And round the yarn surface, where the tubes of neighbouring loops meet.
    var yarn: [SIMD3<Float>] = []
    for _ in 0..<300_000 { yarn.append(SIMD3<Float>(rng.uniform(-1.5, 1.5), rng.uniform(-0.25, 0.3), rng.uniform(-2, 1))) }
    merge(&w, worstOverReport(yarn, step: 0.002, inset: false))
    print(String(format: "        worst over-report: backing %.3f, yarn %.3f, fuzz %.3f; the ray allows %.3f", w[1] ?? 0, w[2] ?? 0, w[3] ?? 0, 1 / stepScale))
    expect(w.count == 3, "not every material was sampled: \(w)")
    for (m, v) in w { expect(v * stepScale <= 1.0, "material \(m) oversteps: \(v)") }
}

test("insets: the core, the twisted fibres of the yarn and the cut fibre are honest outside too — densely, round every fibre") {
    guard let s = scene else { expect(false); return }
    var rng = Seeded(4)
    var box: [SIMD3<Float>] = []
    for _ in 0..<300_000 { box.append(SIMD3<Float>(rng.uniform(-320, 320), rng.uniform(-260, 260), rng.uniform(-200, 200))) }
    var w: [Int: Float] = worstOverReport(box, step: 1.0, inset: true)
    // A dense shell round every fibre: where neighbouring twisted ribbons
    // come closest, and where a ray skims along a ribbon's edge.
    var shell: [SIMD3<Float>] = []
    for r in s.yarn.surface + [s.yarn.fuzz] {
        for k in 0..<r.points.count {
            for _ in 0..<8 {
                shell.append(r.points[k] + SIMD3<Float>(rng.uniform(-18, 18), rng.uniform(-18, 18), rng.uniform(-18, 18)))
            }
        }
    }
    merge(&w, worstOverReport(shell, step: 0.3, inset: true))
    // The cut end and the lumen's mouth, finely.
    let f: Ribbon = s.yarn.fuzz
    let end: SIMD3<Float> = f.points[f.points.count - 1]
    var cut: [SIMD3<Float>] = []
    for _ in 0..<100_000 { cut.append(end + SIMD3<Float>(rng.uniform(-16, 16), rng.uniform(-16, 16), rng.uniform(-16, 16))) }
    merge(&w, worstOverReport(cut, step: 0.08, inset: true))
    print(String(format: "        worst over-report: core %.3f, yarn fibres %.3f, the cut fibre %.3f; the ray allows %.3f", w[1] ?? 0, w[2] ?? 0, w[3] ?? 0, 1 / stepScale))
    expect(w.count == 3, "not every material was sampled: \(w)")
    for (m, v) in w { expect(v * stepScale <= 1.0, "material \(m) oversteps: \(v)") }
}

// MARK: - the picture

section("the picture")

test("each scale bar matches its view: 1 mm, 100 µm, 10 µm, 0.5 nm in that view's pixels") {
    guard let s = scene else { expect(false); return }
    let w: Int = 1920, h: Int = 1080
    let p = SIMD3<Float>(0.3, 0.1, -1)
    let dx: Float = simd_distance(projectMain(p, width: w, height: h), projectMain(p + mainCamera.right, width: w, height: h))
    expect(abs(dx - 1 / mainMillimetresPerPixel(width: w)) < 0.01, "1 mm is \(dx) px")
    let q = SIMD3<Float>(0, 10, 0)
    let du: Float = simd_distance(projectInset(q, height: h), projectInset(q + insetCamera.right * 100, height: h))
    expect(abs(du - 100 / insetMicrometresPerPixel(height: h)) < 0.01, "100 µm is \(du) px")
    let cam: OrthoCamera = cutCamera(s.yarn)
    let dc: Float = simd_distance(projectCut(q, camera: cam, height: h), projectCut(q + cam.right * 10, camera: cam, height: h))
    expect(abs(dc - 10 / cutMicrometresPerPixel(height: h)) < 0.01, "10 µm is \(dc) px")
    let barPx: Float = moleculeBarNanometres * 10 / moleculeAngstromsPerPixel(height: h)
    expect(barPx < 2 * moleculeRadius * Float(h) * 0.8, "the molecule bar does not fit")
    // Each view is several times finer than the one it magnifies: no shared bar.
    expect(mainMillimetresPerPixel(width: w) * 1000 / insetMicrometresPerPixel(height: h) > 4)
    expect(insetMicrometresPerPixel(height: h) / cutMicrometresPerPixel(height: h) > 4)
    // The insets don't overlap each other.
    let hf = Float(h)
    expect(simd_distance(insetCentre, cutCentre) * hf > (insetRadius + cutRadius) * hf + 8)
    expect(simd_distance(insetCentre, moleculeCentre) * hf > (insetRadius + moleculeRadius) * hf + 8)
    expect(simd_distance(cutCentre, moleculeCentre) * hf > (cutRadius + moleculeRadius) * hf + 8)
}

let rendered: (SockImage, [Label])? = {
    guard let d = gpu.device, let s = scene else { return nil }
    guard let img = try? renderSock(width: 960, height: 540, samples: 1, scene: s, on: d).image else { return nil }
    let labels: [Label] = annotate(img, scene: s)
    return (img, labels)
}()

test("the picture shows the knit and its fuzz, the yarn and the tinted fibre, the cut wall and its lumen, and C, O and H") {
    guard let (img, _) = rendered else { expect(false, "render failed"); return }
    var main: Set<Int> = [], yarn: Set<Int> = [], cut: Set<String> = [], mol: Set<Int> = []
    for y in 0..<img.height {
        for x in 0..<img.width {
            let v: SIMD4<Float> = img.seen(x, y)
            switch Int(v.x) {
            case 1: main.insert(Int(v.y))
            case 2: yarn.insert(Int(v.y))
            case 3: cut.insert("\(Int(v.y)).\(Int(v.z))")
            case 4: mol.insert(Int(v.y))
            default: break
            }
        }
    }
    print("        main \(main.sorted()), yarn inset \(yarn.sorted()), cut inset \(cut.sorted()), molecule \(mol.sorted())")
    expect(main.isSuperset(of: [2, 3]), "main view: yarn and fuzz")
    expect(yarn.isSuperset(of: [1, 2, 3]), "yarn inset: core, fibres and the tinted fibre")
    expect(cut.contains("3.1") && cut.contains("3.2"), "cut inset: the cut face and the lumen")
    expect(mol.isSuperset(of: [1, 2, 3]), "molecule: C, O and H")
}

test("every label is on the page, none collides with another, and each view has its own") {
    guard let (img, labels) = rendered else { expect(false, "render failed"); return }
    let page = CGRect(x: 0, y: 0, width: img.width, height: img.height)
    for l in labels { expect(page.contains(l.box), "\"\(l.text)\" runs off the page: \(l.box)") }
    for i in 0..<labels.count {
        for j in (i + 1)..<labels.count where labels[i].box.intersects(labels[j].box) {
            expect(false, "\"\(labels[i].text)\" overlaps \"\(labels[j].text)\"")
        }
    }
    let texts: Set<String> = Set(labels.map { $0.text })
    for want in ["knit loops", "fuzz", "reversal", "convolutions", "one fibre = one cell", "cut across", "cellulose", "caption"] {
        expect(texts.contains(want), "no \"\(want)\" label")
    }
    for v in 1...4 { expect(labels.contains { $0.view == v }, "view \(v) has no label") }
}

finish()
