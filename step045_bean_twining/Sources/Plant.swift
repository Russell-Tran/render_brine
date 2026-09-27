// The shared machinery for a climbing plant, as numbers: the kinds of shape a
// plant here is built from, and the curve arithmetic both climbing steps use.
// Nothing here touches the GPU and nothing here is one species; the species
// files say where each thing goes and when.
//
// This file is the same in step 45 (the twining bean) and step 46 (the sweet
// pea with tendrils). It was written once and copied into both folders, so
// each step builds on its own.
//
// Millimetres throughout. +y is up; the camera looks along −z from +z, so +x
// is the viewer's right. Time is in hours of real growth.
//
// The shapes:
//
//   * a Chain is a polyline of capsules — the stem, the petioles, a tendril, a
//     pole, a string. Distance to a capsule is exact (step 25), so a chain can
//     follow any curve with a distance function that is honest everywhere.
//   * a Ribbon is a thin flat strip along a segment: the wings of a sweet pea's
//     stem and leaf stalk. With its width at right angles to its length the
//     nearest point is found by clamping each coordinate on its own, which is
//     exact.
//   * a Leaflet is a thin blade with an outline, a fold along its midrib, and a
//     droop — the one shape whose distance is an estimate, and the tests
//     measure how far it can be trusted.

import Foundation
import simd

typealias V3 = SIMD3<Float>

/// What a surface is made of, as the kernel numbers them.
enum Material: Int {
    case none = 0, pole, stem, petiole, leaf, tendril, twine, stipule
}

/// Outlines a leaflet can have; the kernel numbers them the same way.
enum LeafShape: Int {
    /// Phaseolus vulgaris: broadly ovate, widest below the middle, rounded at
    /// the base, acuminate — drawn out to a point — at the apex.
    case beanOvate = 0
    /// Lathyrus odoratus leaflet: ovate-oblong to elliptic, blunt-tipped.
    case peaElliptic = 1
    /// A stipule: a small half-arrowhead at the base of a sweet pea leaf.
    case stipule = 2
}

struct Chain {
    var points: [V3]
    var radii: [Float]
    var material: Material
}

struct Ribbon {
    var a: V3
    var b: V3
    /// Unit, at right angles to b − a: the direction the strip is wide in.
    var n: V3
    var halfWidth: Float
    var halfThickness: Float
    var material: Material
}

struct Leaflet {
    /// The base of the midrib.
    var origin: V3
    /// Along the midrib, across the blade, and out of its upper face.
    var u: V3
    var v: V3
    var w: V3
    var length: Float
    var halfWidth: Float
    /// Radians each half of the blade is folded up from flat about the midrib:
    /// near π/2 for a leaflet still folded in the bud, small once it is open.
    var fold: Float
    /// How far the midrib curves away from the upper face, as a fraction of the
    /// length, at the tip.
    var droop: Float
    var shape: LeafShape
    var material: Material
}

/// Everything in one frame.
struct PlantScene {
    var chains: [Chain] = []
    var ribbons: [Ribbon] = []
    var leaflets: [Leaflet] = []

    /// The same scene moved up by `dy`: what the loop test compares.
    func shifted(by dy: Float) -> PlantScene {
        let d = V3(0, dy, 0)
        var s = self
        s.chains = chains.map { c in Chain(points: c.points.map { $0 + d }, radii: c.radii, material: c.material) }
        s.ribbons = ribbons.map { r in
            Ribbon(a: r.a + d, b: r.b + d, n: r.n, halfWidth: r.halfWidth, halfThickness: r.halfThickness, material: r.material)
        }
        s.leaflets = leaflets.map { l in
            var m = l
            m.origin += d
            return m
        }
        return s
    }

    /// The largest distance between matching points of two scenes with the
    /// same structure, or infinity if the structures differ.
    func largestDifference(from o: PlantScene) -> Float {
        guard chains.count == o.chains.count, ribbons.count == o.ribbons.count,
              leaflets.count == o.leaflets.count else { return .infinity }
        var worst: Float = 0
        for (a, b) in zip(chains, o.chains) {
            guard a.points.count == b.points.count, a.material == b.material else { return .infinity }
            for (p, q) in zip(a.points, b.points) { worst = max(worst, simd_distance(p, q)) }
            for (r, s) in zip(a.radii, b.radii) { worst = max(worst, abs(r - s)) }
        }
        for (a, b) in zip(ribbons, o.ribbons) {
            worst = max(worst, simd_distance(a.a, b.a), simd_distance(a.b, b.b), abs(a.halfWidth - b.halfWidth))
        }
        for (a, b) in zip(leaflets, o.leaflets) {
            worst = max(worst, simd_distance(a.origin, b.origin), abs(a.length - b.length), abs(a.fold - b.fold))
            worst = max(worst, simd_distance(a.u, b.u) * a.length, simd_distance(a.w, b.w) * a.length)
        }
        return worst
    }
}

// MARK: - small arithmetic

func smoothstep(_ a: Float, _ b: Float, _ x: Float) -> Float {
    let t: Float = min(max((x - a) / (b - a), 0), 1)
    return t * t * (3 - 2 * t)
}

/// ∫₀ˣ smoothstep(0, L, ξ) dξ, in closed form: L(u³ − u⁴/2) with u = x/L up to
/// L, then L/2 plus the straight run after it.
func smoothstepIntegral(_ x: Float, _ L: Float) -> Float {
    if x <= 0 { return 0 }
    if x >= L { return L / 2 + (x - L) }
    let u: Float = x / L
    return L * (u * u * u - u * u * u * u / 2)
}

/// Rotate `p` about unit axis `k` by `angle` (Rodrigues).
func rotate(_ p: V3, about k: V3, by angle: Float) -> V3 {
    let c: Float = cos(angle)
    let s: Float = sin(angle)
    return p * c + simd_cross(k, p) * s + k * (simd_dot(k, p) * (1 - c))
}

/// A unit vector at right angles to `a`.
func perpendicular(to a: V3) -> V3 {
    let t: V3 = abs(a.y) < 0.9 ? V3(0, 1, 0) : V3(1, 0, 0)
    return simd_normalize(simd_cross(a, t))
}

/// Distance from a point to a segment's axis.
func segmentDistance(_ p: V3, _ a: V3, _ b: V3) -> Float {
    let ab: V3 = b - a
    let h: Float = min(max(simd_dot(p - a, ab) / max(simd_dot(ab, ab), 1e-12), 0), 1)
    return simd_distance(p, a + ab * h)
}

/// The arc length of a polyline.
func polylineLength(_ pts: [V3]) -> Float {
    var s: Float = 0
    for i in 1..<max(pts.count, 1) { s += simd_distance(pts[i - 1], pts[i]) }
    return s
}

/// The point at arc length `s` along a polyline, and the direction there.
func pointAlong(_ pts: [V3], _ s: Float) -> (p: V3, t: V3) {
    var left: Float = s
    for i in 1..<pts.count {
        let d: Float = simd_distance(pts[i - 1], pts[i])
        if left <= d || i == pts.count - 1 {
            let f: Float = d > 0 ? min(max(left / d, 0), 1) : 0
            let t: V3 = d > 0 ? (pts[i] - pts[i - 1]) / d : V3(0, 1, 0)
            return (pts[i - 1] + (pts[i] - pts[i - 1]) * f, t)
        }
        left -= d
    }
    return (pts[0], V3(0, 1, 0))
}

/// A quadratic Bézier from a to c with control b, as `n` segments.
func bezier(_ a: V3, _ b: V3, _ c: V3, segments n: Int) -> [V3] {
    (0...n).map { i in
        let t: Float = Float(i) / Float(n)
        let s: Float = 1 - t
        return a * (s * s) + b * (2 * s * t) + c * (t * t)
    }
}

/// Signed turning of a curve about an axis: the angle, unwrapped, that each
/// point makes round the line through `origin` along unit `axis`, measured
/// right-handedly about `axis` (anticlockwise seen looking back down it).
/// Points on the axis carry the previous angle forward.
func unwrappedAngles(_ pts: [V3], origin: V3, axis: V3, reference: V3) -> [Float] {
    let e1: V3 = simd_normalize(reference - axis * simd_dot(reference, axis))
    let e2: V3 = simd_cross(axis, e1)
    var out: [Float] = []
    var last: Float = 0
    var have: Bool = false
    for p in pts {
        let q: V3 = p - origin
        let x: Float = simd_dot(q, e1)
        let y: Float = simd_dot(q, e2)
        if x * x + y * y < 1e-8 { out.append(last); continue }
        var a: Float = atan2(y, x)
        if have {
            while a - last > Float.pi { a -= 2 * Float.pi }
            while a - last < -Float.pi { a += 2 * Float.pi }
        }
        out.append(a)
        last = a
        have = true
    }
    return out
}

/// A small repeatable random source for the tests.
struct Lcg {
    var state: UInt64
    mutating func next() -> Float {
        state = state &* 6364136223846793005 &+ 1442695040888963407
        return Float(state >> 40) / Float(1 << 24)
    }
    mutating func unit() -> V3 {
        while true {
            let v = V3(next() * 2 - 1, next() * 2 - 1, next() * 2 - 1)
            let l: Float = simd_length(v)
            if l > 0.05 && l <= 1 { return v / l }
        }
    }
}
