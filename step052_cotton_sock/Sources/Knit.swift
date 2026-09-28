// The knit: plain (jersey) weft knitting, in MILLIMETRES.
//
// Each course (row) is one continuous yarn making a line of loops; each
// loop is pulled through the loop of the course below. On the face you see
// the loops' legs, as rows of Vs; the heads and the sinker loops between
// them lie behind.

import Foundation
import simd

// MARK: - the fabric, measured

// Hossain et al. 2025 (PMC12080846), Table 2, sample 1 (regular sinker
// timing, dry-relaxed 48 h): loop length 2.77 mm, 32.2 wales and 47.8
// courses per 2.54 cm, from the 30/1 Ne cotton yarn in Yarn.swift.
let measuredLoopLength: Float = 2.77          // mm of yarn per stitch
let walesPerInch: Float = 32.2
let coursesPerInch: Float = 47.8
let waleSpacing: Float = 25.4 / walesPerInch  // 0.789 mm, centre to centre across
let courseSpacing: Float = 25.4 / coursesPerInch  // 0.531 mm, row to row

/// Yarn radius in the fabric, mm (Yarn.swift: ~96 fibres at packing 0.6).
let knitYarnRadius: Float = yarnRadius / 1000

// MARK: - one loop's centre line

// The loop is the standard parametric stitch used for yarn-level knit
// models: over one stitch, t from −π to π,
//     X = w/2π · (t + a sin 2t),   Y = h cos t,   Z = d cos 2t,
// X along the course, Y up the wale, Z through the fabric (+Z away from the
// face). t = 0 is the head, t = ±π the sinker loops that join the
// neighbours; a > 0.5 makes the head bulge and the legs pinch in below it —
// the jersey loop's shape. MODEL shape parameters: a = 1.5, h = 0.41 mm,
// chosen by a numerical search during development (the tests check the
// outcome) so the loop comes out close to the measured 2.77 mm; d is then SOLVED so that
// each row's loops just touch the row below, 2r apart (see solveDepth).
// This curve family cannot get below ~2.80 mm at this spacing and yarn
// radius without the rows colliding: 1.1% over the measurement.
let loopBulge: Float = 1.5
let loopHalfHeight: Float = 0.41

struct Knit {
    var w: Float                // wale spacing
    var c: Float                // course spacing
    var a: Float
    var h: Float
    var d: Float                // solved depth amplitude
    var r: Float                // yarn radius

    /// Loop-local centre line at t.
    func local(_ t: Float) -> SIMD3<Float> {
        let x: Float = w / (2 * Float.pi) * (t + a * sin(2 * t))
        return SIMD3<Float>(x, h * cos(t), d * cos(2 * t))
    }

    /// Loop (i, j) in knit coordinates: loop centres on a lattice.
    func point(_ i: Int, _ j: Int, _ t: Float) -> SIMD3<Float> {
        local(t) + SIMD3<Float>(Float(i) * w, Float(j) * c, 0)
    }

    /// Knit coordinates → world. World y is up out of the face, towards the
    /// viewer; wales run towards −z (away), courses along x.
    func world(_ k: SIMD3<Float>) -> SIMD3<Float> { SIMD3<Float>(k.x, -k.z, -k.y) }
    func knitCoords(_ p: SIMD3<Float>) -> SIMD3<Float> { SIMD3<Float>(p.x, -p.z, -p.y) }

    /// The loop's centre line as a polyline, `n` segments, loop-local.
    func polyline(_ n: Int = knitSegments) -> [SIMD3<Float>] {
        (0...n).map { local(-Float.pi + 2 * Float.pi * Float($0) / Float(n)) }
    }

    /// Arc length of one loop's centre line.
    var loopLength: Float {
        let p: [SIMD3<Float>] = polyline(4000)
        var s: Float = 0
        for k in 1..<p.count { s += simd_distance(p[k], p[k - 1]) }
        return s
    }
}

/// Segments per loop in the drawn polyline. At 64 the chords stray < 3 µm
/// from the curve, against an 87 µm yarn radius.
let knitSegments: Int = 64

/// Distance from p to the segment ab.
func segmentDistance(_ p: SIMD3<Float>, _ a: SIMD3<Float>, _ b: SIMD3<Float>) -> Float {
    let ab: SIMD3<Float> = b - a
    let t: Float = simd_clamp(simd_dot(p - a, ab) / simd_dot(ab, ab), 0, 1)
    return simd_distance(p, a + ab * t)
}

/// Shortest distance between two polylines (segment to segment, sampled
/// finely enough for tests: each segment split in 8).
func polylineDistance(_ A: [SIMD3<Float>], _ B: [SIMD3<Float>]) -> Float {
    var best: Float = .infinity
    for i in 1..<A.count {
        for k in 0...8 {
            let p: SIMD3<Float> = A[i - 1] + (A[i] - A[i - 1]) * (Float(k) / 8)
            for j in 1..<B.count { best = min(best, segmentDistance(p, B[j - 1], B[j])) }
        }
    }
    return best
}

/// The depth amplitude at which a row's loops just touch the row below:
/// the smallest centre-line distance between loop (0,1) and loops (−1…1, 0)
/// is exactly 2r. Found by bisection on the polyline.
func solveDepth(w: Float, c: Float, a: Float, h: Float, r: Float) -> Float {
    func gap(_ d: Float) -> Float {
        let k = Knit(w: w, c: c, a: a, h: h, d: d, r: r)
        let upper: [SIMD3<Float>] = k.polyline(96).map { $0 + SIMD3<Float>(0, c, 0) }
        var best: Float = .infinity
        for i in -1...1 {
            let lower: [SIMD3<Float>] = k.polyline(96).map { $0 + SIMD3<Float>(Float(i) * w, 0, 0) }
            best = min(best, polylineDistance(upper, lower))
        }
        return best - 2 * r
    }
    var lo: Float = 0.0, hi: Float = 0.4
    for _ in 0..<30 {
        let m: Float = (lo + hi) / 2
        if gap(m) < 0 { lo = m } else { hi = m }
    }
    return hi
}

func buildKnit(mutant: Mutant) -> Knit {
    let d: Float = solveDepth(w: waleSpacing, c: courseSpacing, a: loopBulge, h: loopHalfHeight, r: knitYarnRadius)
    // The mutant pulls the courses 20% further apart AFTER the fit: the
    // rows no longer hold each other.
    let c: Float = mutant == .gappyKnit ? courseSpacing * 1.2 : courseSpacing
    return Knit(w: waleSpacing, c: c, a: loopBulge, h: loopHalfHeight, d: d, r: knitYarnRadius)
}

/// CPU distance to the knit's yarn surface, world coordinates — the same
/// arithmetic as the kernel's, for building hairs and for tests.
func knitDistance(_ p: SIMD3<Float>, _ k: Knit, _ line: [SIMD3<Float>]) -> Float {
    let q: SIMD3<Float> = k.knitCoords(p)
    let i0: Int = Int((q.x / k.w).rounded())
    let j0: Int = Int((q.y / k.c).rounded())
    var best: Float = .infinity
    for j in (j0 - 1)...(j0 + 1) {
        for i in (i0 - 1)...(i0 + 1) {
            let l: SIMD3<Float> = q - SIMD3<Float>(Float(i) * k.w, Float(j) * k.c, 0)
            for s in 1..<line.count { best = min(best, segmentDistance(l, line[s - 1], line[s])) }
        }
    }
    return best - k.r
}

// MARK: - fuzz: loose fibre ends standing up from the yarn

/// One hair: a loose end of a fibre, rooted in a loop's yarn.
struct Hair {
    var loop: (i: Int, j: Int)
    var t: Float                  // where on the loop it roots
    var ribbon: Ribbon            // in mm
}

// How many hairs and how long. MODEL, UNVERIFIED: no hairiness measurement
// of this yarn was found. Real ring-spun cotton is hairier than this — far
// more short ends lying close along the surface; these are the ends that
// stand up and catch the light, about 1.6 per loop, 0.15–1.1 mm long.
let hairsPerLoop: Float = 1.6
let hairLengthRange: ClosedRange<Float> = 0.15...1.1     // mm

/// Loops the main view can see (knit lattice indices). Set from the view.
let visibleLoopsI: ClosedRange<Int> = -6...6
let visibleLoopsJ: ClosedRange<Int> = -4...9

func buildHairs(_ k: Knit, mutant: Mutant) -> [Hair] {
    let line: [SIMD3<Float>] = k.polyline()
    var rng = Seeded(1052)
    var out: [Hair] = []
    let section: CrossSection = mutant == .roundFibre ? .round() : .cotton()
    let reachMM: Float = section.reach / 1000
    let rate: Float = (mutant == .roundFibre ? 0 : twistRate) * 1000   // rad per mm
    let loops: Int = visibleLoopsI.count * visibleLoopsJ.count
    let wanted: Int = Int(Float(loops) * hairsPerLoop)
    var tries: Int = 0
    while out.count < wanted && tries < wanted * 40 {
        tries += 1
        let i: Int = Int(rng.uniform(Float(visibleLoopsI.lowerBound), Float(visibleLoopsI.upperBound) + 0.999).rounded(.down))
        let j: Int = Int(rng.uniform(Float(visibleLoopsJ.lowerBound), Float(visibleLoopsJ.upperBound) + 0.999).rounded(.down))
        let t: Float = rng.uniform(-Float.pi, Float.pi)
        let c: SIMD3<Float> = k.world(k.point(i, j, t))
        let tan: SIMD3<Float> = simd_normalize(k.world(k.point(i, j, t + 1e-3)) - c)
        // Root on the side of the yarn facing out of the fabric (+y), turned
        // a random amount round the yarn.
        let up = SIMD3<Float>(0, 1, 0)
        let out0: SIMD3<Float> = simd_normalize(up - tan * simd_dot(up, tan))
        let side: SIMD3<Float> = simd_cross(tan, out0)
        let spin: Float = rng.uniform(-1.0, 1.0)
        let n: SIMD3<Float> = simd_normalize(out0 * cos(spin) + side * sin(spin))
        // Rooted just inside the yarn surface, so it grows out of it.
        let root: SIMD3<Float> = c + n * (k.r - reachMM * 1.2)
        let length: Float = hairLengthRange.lowerBound + (hairLengthRange.upperBound - hairLengthRange.lowerBound)
            * pow(rng.uniform(0, 1), 1.8)
        let az: Float = rng.uniform(0, 2 * Float.pi)
        let drift = SIMD3<Float>(cos(az), 0, sin(az))
        let p1: SIMD3<Float> = root + n * length * 0.45
        let d2a: SIMD3<Float> = n * 0.6 + up * 0.5
        let d2: SIMD3<Float> = simd_normalize(d2a + drift * 0.7)
        let d3a: SIMD3<Float> = n * 0.3 + up * 0.35
        let d3: SIMD3<Float> = simd_normalize(d3a + drift)
        let p2: SIMD3<Float> = root + d2 * (length * 0.75)
        let p3: SIMD3<Float> = root + d3 * length
        let pts: [SIMD3<Float>] = bezierPoints(root, p1, p2, p3, step: 0.025)
        // Keep it only if, once clear of its root, it never enters the knit.
        var clear = true
        for p in pts.dropFirst(4) where knitDistance(p, k, line) < reachMM * 1.5 { clear = false; break }
        if !clear { continue }
        let len: Float = Float(pts.count - 1) * 0.025
        let revs: [Float] = mutant == .noReversal ? [] : randomReversals(length: len, spacing: meanReversalSpacing / 1000, rng: &rng)
        let hand: Float = rng.uniform(0, 1) < 0.5 ? 1 : -1
        let rib: Ribbon = makeRibbon(pts, startWidth: perpendicular(to: simd_normalize(pts[1] - pts[0])),
                                     startTwist: rng.uniform(0, 2 * Float.pi), reversals: revs, hand: hand, rate: rate, scale: 0.001)
        out.append(Hair(loop: (i, j), t: t, ribbon: rib))
    }
    return out
}
