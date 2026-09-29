// A toy boat as the render and the tests see it: a few RIGID segments (the
// hull and what is moulded onto it), each made of a few shapes in its own
// frame, placed in the world by one rigid transform. Copied from steps
// 58–63's toy renderer, with the animal rig taken out and two shapes added
// that boats need: a HULL and a PROFILE plate (a sail, an end piece).
//
// Millimetres throughout. The world: y up, the table top is y = 0, the boat
// faces +x (bow forward), +z is starboard.
//
// Every shape has an exact distance, or a 1-Lipschitz lower bound, outside:
//   * round cone, rounded box, vesica plate — exact (as steps 58–63);
//   * profile plate: an exact 2D triangle, cut by circles with max, extruded
//     to a thickness that may taper, edges rounded; the taper and the
//     rounding's non-square gradients are divided out by per-shape
//     CONSTANTS, so the field never claims more room than there is;
//   * hull: intersections (max) and set-differences (max with a negation)
//     of exact 1-Lipschitz fields — a tapered plan, a keel circle with
//     deadrise, a sheer circle or deck cylinder — the bilge rounded by a
//     quadratic smooth-max of CONSTANT width. Max, min and smooth-max of
//     1-Lipschitz fields are 1-Lipschitz, and a 1-Lipschitz field that is
//     zero on the surface is never more than the true distance.
// Within one segment the shapes are joined by smooth-min with a constant
// width; segments by plain min. The mould's parting line is a ridge SEAM_H
// proud, either in a segment's local z = 0 plane or, on a hull, round its
// gunwale (see `hullSDF`).

import Foundation
import simd

// MARK: - frames

/// A rigid placement: an origin and three orthonormal axes, right-handed
/// (z = x × y). A point's local coordinates (a, b, c) sit at o + a x + b y + c z.
struct Frame {
    var o: SIMD3<Float>
    var x: SIMD3<Float>
    var y: SIMD3<Float>
    var z: SIMD3<Float>

    static let identity = Frame(o: .zero, x: SIMD3<Float>(1, 0, 0), y: SIMD3<Float>(0, 1, 0), z: SIMD3<Float>(0, 0, 1))

    /// A frame whose local z is the world's up, so a segment's z = 0 seam
    /// is a horizontal parting plane at height h (x forward, y to port).
    static func horizontalParting(at h: Float) -> Frame {
        Frame(o: SIMD3<Float>(0, h, 0), x: SIMD3<Float>(1, 0, 0), y: SIMD3<Float>(0, 0, -1), z: SIMD3<Float>(0, 1, 0))
    }

    func toWorld(_ p: SIMD3<Float>) -> SIMD3<Float> {
        let a: SIMD3<Float> = x * p.x + y * p.y
        return o + a + z * p.z
    }
    func dirToWorld(_ v: SIMD3<Float>) -> SIMD3<Float> {
        let a: SIMD3<Float> = x * v.x + y * v.y
        return a + z * v.z
    }
    func toLocal(_ q: SIMD3<Float>) -> SIMD3<Float> {
        let d: SIMD3<Float> = q - o
        return SIMD3<Float>(simd_dot(d, x), simd_dot(d, y), simd_dot(d, z))
    }
    func dirToLocal(_ v: SIMD3<Float>) -> SIMD3<Float> {
        SIMD3<Float>(simd_dot(v, x), simd_dot(v, y), simd_dot(v, z))
    }
}

// MARK: - shapes

/// What a shape is, for the tests that count them off the drawn toy.
enum PrimTag: Int {
    case none = 0, hull, thwart, oar, blade, rowlock, mast, sail, boom, crossbeam, deck, manu, spar, sweep, rudder
}

enum PrimKind: Int32 {
    case roundCone = 0
    case roundBox = 1
    case plate = 2
    case profile = 3
    case hull = 4
}

/// One shape, in its segment's frame.
///   roundCone: sphere radius `ra` at `a` swept to radius `rb` at `b`.
///   roundBox: centre `a`, axes `u` (x) and `n` (z), half-extents `size.xyz`,
///             edge radius `size.w`.
///   plate: centre `a`; outline long axis `u`; thickness axis `n`; `size` =
///          (half-height along u, half-width across, half-thickness, edge).
///   profile: see `ProfileSpec` — packed into a, u, n, size, b, e, f.
///   hull: see `HullSpec` — packed into a, b, u, n, size, e, f, g.
struct Prim {
    var kind: PrimKind
    var a: SIMD3<Float>
    var ra: Float = 0
    var b: SIMD3<Float> = .zero
    var rb: Float = 0
    var u = SIMD3<Float>(0, 1, 0)
    var n = SIMD3<Float>(0, 0, 1)
    var size = SIMD4<Float>(0, 0, 0, 0)
    var e = SIMD4<Float>(0, 0, 0, 0)
    var f = SIMD4<Float>(0, 0, 0, 0)
    var g = SIMD4<Float>(0, 0, 0, 0)
    var paint: Int
    /// A second paint: a hull's inside (below its gunwale, and its thwarts).
    var paint2: Int = -1
    /// Its own parting line: a plate's or profile's round its rim in its
    /// mid-plane; a hull's round its gunwale.
    var ownSeam: Bool = false
    var tag: PrimTag = .none
    /// A bounding sphere given by the builder (hull, profile), in the frame.
    var boundGiven: SIMD4<Float> = .zero

    func tagged(_ t: PrimTag) -> Prim {
        var c: Prim = self
        c.tag = t
        return c
    }

    static func cone(_ a: SIMD3<Float>, _ ra: Float, _ b: SIMD3<Float>, _ rb: Float, paint: Int) -> Prim {
        Prim(kind: .roundCone, a: a, ra: ra, b: b, rb: rb, paint: paint)
    }
    static func ball(_ c: SIMD3<Float>, _ r: Float, paint: Int) -> Prim {
        Prim(kind: .roundCone, a: c, ra: r, b: c, rb: r, paint: paint)
    }
    static func box(_ c: SIMD3<Float>, half: SIMD3<Float>, edge: Float, xAxis: SIMD3<Float> = SIMD3<Float>(1, 0, 0),
                    zAxis: SIMD3<Float> = SIMD3<Float>(0, 0, 1), paint: Int) -> Prim {
        Prim(kind: .roundBox, a: c, u: simd_normalize(xAxis), n: simd_normalize(zAxis),
             size: SIMD4<Float>(half.x, half.y, half.z, edge), paint: paint)
    }

    /// A sphere round the shape, in its segment's frame.
    var bound: (centre: SIMD3<Float>, radius: Float) {
        switch kind {
        case .roundCone:
            let c: SIMD3<Float> = (a + b) / 2
            return (c, simd_distance(a, b) / 2 + max(ra, rb))
        case .roundBox:
            return (a, simd_length(SIMD3<Float>(size.x, size.y, size.z)) + size.w)
        case .plate:
            return (a, max(size.x, size.y) + size.z + size.w)
        case .profile, .hull:
            return (SIMD3<Float>(boundGiven.x, boundGiven.y, boundGiven.z), boundGiven.w)
        }
    }
}

// MARK: - exact 2D and 3D pieces (the kernel has the same arithmetic)

/// Exact distance to a round cone (Quílez's formulation).
func sdRoundCone(_ p: SIMD3<Float>, _ a: SIMD3<Float>, _ b: SIMD3<Float>, _ r1: Float, _ r2: Float) -> Float {
    let ba: SIMD3<Float> = b - a
    let l2: Float = simd_dot(ba, ba)
    if l2 < 1e-12 { return simd_distance(p, a) - max(r1, r2) }
    let rr: Float = r1 - r2
    let a2: Float = l2 - rr * rr
    let il2: Float = 1 / l2
    let pa: SIMD3<Float> = p - a
    let y: Float = simd_dot(pa, ba)
    let z: Float = y - l2
    let xv: SIMD3<Float> = pa * l2 - ba * y
    let x2: Float = simd_dot(xv, xv)
    let y2: Float = y * y * l2
    let z2: Float = z * z * l2
    let k: Float = (rr >= 0 ? 1 : -1) * rr * rr * x2
    if (z >= 0 ? 1 : -1) * a2 * z2 > k { return (x2 + z2).squareRoot() * il2 - r2 }
    if (y >= 0 ? 1 : -1) * a2 * y2 < k { return (x2 + y2).squareRoot() * il2 - r1 }
    return ((x2 * a2 * il2).squareRoot() + y * rr) * il2 - r1
}

/// Exact 2D distance to a vesica of half-height h (along y) and half-width
/// w (along x), h ≥ w.
func sdVesica2(_ p0: SIMD2<Float>, h: Float, w: Float) -> Float {
    let r: Float = (w + h * h / w) / 2
    let d: Float = r - w
    let p = SIMD2<Float>(abs(p0.x), abs(p0.y))
    let bb: Float = (r * r - d * d).squareRoot()
    if (p.y - bb) * d > p.x * bb { return simd_distance(p, SIMD2<Float>(0, bb)) }
    return simd_distance(p, SIMD2<Float>(-d, 0)) - r
}

/// Exact 2D distance to a triangle (Quílez), negative inside.
func sdTriangle2(_ p: SIMD2<Float>, _ p0: SIMD2<Float>, _ p1: SIMD2<Float>, _ p2: SIMD2<Float>) -> Float {
    let e0: SIMD2<Float> = p1 - p0
    let e1: SIMD2<Float> = p2 - p1
    let e2: SIMD2<Float> = p0 - p2
    let v0: SIMD2<Float> = p - p0
    let v1: SIMD2<Float> = p - p1
    let v2: SIMD2<Float> = p - p2
    let t0: Float = min(max(simd_dot(v0, e0) / simd_dot(e0, e0), 0), 1)
    let t1: Float = min(max(simd_dot(v1, e1) / simd_dot(e1, e1), 0), 1)
    let t2: Float = min(max(simd_dot(v2, e2) / simd_dot(e2, e2), 0), 1)
    let pq0: SIMD2<Float> = v0 - e0 * t0
    let pq1: SIMD2<Float> = v1 - e1 * t1
    let pq2: SIMD2<Float> = v2 - e2 * t2
    let orient: Float = e0.x * e2.y - e0.y * e2.x
    let s: Float = orient >= 0 ? 1 : -1
    let c0: Float = s * (v0.x * e0.y - v0.y * e0.x)
    let c1: Float = s * (v1.x * e1.y - v1.y * e1.x)
    let c2: Float = s * (v2.x * e2.y - v2.y * e2.x)
    let dd: Float = min(simd_dot(pq0, pq0), min(simd_dot(pq1, pq1), simd_dot(pq2, pq2)))
    let sg: Float = min(c0, min(c1, c2))
    return -dd.squareRoot() * (sg >= 0 ? 1 : -1)
}

/// Quadratic smooth minimum with a constant blend width k (Quílez). Its
/// value is a convex mix of its inputs' gradients, so it is 1-Lipschitz when
/// they are.
func smin(_ a: Float, _ b: Float, _ k: Float) -> Float {
    if k <= 0 { return min(a, b) }
    let h: Float = max(k - abs(a - b), 0) / k
    return min(a, b) - h * h * k * 0.25
}

/// Smooth maximum, the mirror of `smin`: rounds an intersection's edge.
func smax(_ a: Float, _ b: Float, _ k: Float) -> Float { -smin(-a, -b, k) }

// MARK: - the profile plate: a sail, a boat's end piece

/// A flat piece standing in a plane: a triangle's outline, optionally with a
/// circle cut out of it (a concave edge) and kept inside another circle (a
/// convex one), extruded to a thickness that can thin towards the top (the
/// draft a mould needs when it pulls along the plate's face), edges rounded.
struct ProfileSpec {
    var origin: SIMD3<Float>
    var xAxis: SIMD3<Float>             // in the plane
    var normal: SIMD3<Float>            // square to the plane
    var a: SIMD2<Float>
    var b: SIMD2<Float>
    var c: SIMD2<Float>
    var cut: SIMD3<Float> = SIMD3<Float>(0, 0, 0)     // centre x, y, radius (0: none)
    var keep: SIMD3<Float> = SIMD3<Float>(0, 0, 0)    // centre x, y, radius (0: none)
    var halfThickness: Float            // at the plane's local y = 0
    var taper: Float = 0                // half-thickness lost per mm of local y
    var edge: Float

    func prim(paint: Int, seam: Bool) -> Prim {
        let nn: SIMD3<Float> = simd_normalize(normal)
        let uu: SIMD3<Float> = simd_normalize(xAxis - nn * simd_dot(xAxis, nn))
        let v: SIMD3<Float> = simd_cross(nn, uu)
        let cx: Float = (a.x + b.x + c.x) / 3
        let cy: Float = (a.y + b.y + c.y) / 3
        let cen: SIMD3<Float> = origin + uu * cx + v * cy
        var r: Float = 0
        var thick: Float = halfThickness
        for q in [a, b, c] {
            let w: SIMD3<Float> = origin + uu * q.x + v * q.y
            r = max(r, simd_distance(w, cen))
            thick = max(thick, halfThickness - taper * q.y)
        }
        return Prim(kind: .profile, a: origin, ra: halfThickness, b: SIMD3<Float>(c.x, c.y, edge), rb: 0,
                    u: uu, n: nn, size: SIMD4<Float>(a.x, a.y, b.x, b.y),
                    e: SIMD4<Float>(cut.x, cut.y, cut.z, 0), f: SIMD4<Float>(keep.x, keep.y, keep.z, taper),
                    paint: paint, ownSeam: seam, boundGiven: SIMD4<Float>(cen, r + thick + edge + 0.2))
    }
}

func profileSDF(_ s: Prim, _ q: SIMD3<Float>, seams: Bool) -> Float {
    let v: SIMD3<Float> = simd_cross(s.n, s.u)
    let d: SIMD3<Float> = q - s.a
    let p2 = SIMD2<Float>(simd_dot(d, s.u), simd_dot(d, v))
    let pz: Float = simd_dot(d, s.n)
    let edge: Float = s.b.z
    var d2: Float = sdTriangle2(p2, SIMD2<Float>(s.size.x, s.size.y), SIMD2<Float>(s.size.z, s.size.w), SIMD2<Float>(s.b.x, s.b.y))
    if s.e.z > 0 { d2 = max(d2, s.e.z - simd_distance(p2, SIMD2<Float>(s.e.x, s.e.y))) }
    if s.f.z > 0 { d2 = max(d2, simd_distance(p2, SIMD2<Float>(s.f.x, s.f.y)) - s.f.z) }
    let taper: Float = s.f.w
    let sT: Float = (1 + taper * taper).squareRoot()
    let th: Float = s.ra - taper * p2.y
    let dz: Float = (abs(pz) - th) / sT
    let w = SIMD2<Float>(d2 + edge, dz + edge)
    let outside: Float = simd_length(simd_max(w, SIMD2<Float>(0, 0)))
    let raw: Float = min(max(w.x, w.y), 0) + outside - edge
    // The outline's and the thickness's gradients meet at cos ≤ taper/sT, so
    // the rounded corner's gradient is at most √(1 + taper/sT): divide it out.
    var dist: Float = raw / (1 + abs(taper) / sT).squareRoot()
    if seams && s.ownSeam { dist = min(dist, max(dist - seamHeight, abs(pz) - seamHalfWidth)) }
    return dist
}

// MARK: - the hull

/// A hull, as the intersection of simple exact fields (hull frame: x
/// forward, y up from the keel's lowest point, z starboard):
///   * PLAN — a vesica (the lens two circles make: a fine bow) of half-length
///     `planHalfLength` and half-beam `beam`, centred at x = `planCentre`,
///     cut square at a transom x = `transom`; this outline holds at height
///     `planHeight` and shrinks downwards by tan(`flare`) per mm (and grows
///     upwards) — the topsides' flare, which also rakes the stem and transom;
///   * KEEL — inside a circle in the side view (the rocker: the bottom rising
///     to bow and stern), lowest at x = `keelLowX`, radius `keelRadius`; the
///     bottom rising from the keel by tan(`deadrise`) each side (a V);
///   * the two joined by a smooth-max of width `bilge` (a rounded bilge);
///   * TOP — open boat: outside a circle in the side view (the SHEER line,
///     rising to bow and stern), lowest at x = `sheerLowX`, height
///     `sheerLowY`, radius `sheerRadius`; decked boat: inside a cylinder
///     along x, crowned `deckCrown` above the keel at x = 0, rising forward
///     by `deckSlope`, radius `deckCamberRadius` (the deck's camber);
///   * optional HOLLOW — the plan and keel moved in by `wall` and `floor`, open
///     at the top, with solid THWARTS left standing across it.
struct HullSpec {
    var planHalfLength: Float
    var beam: Float                     // half-beam at planHeight
    var planCentre: Float
    var transom: Float = -1e4
    var planHeight: Float
    var flare: Float                    // radians
    var keelLowX: Float
    var keelRadius: Float
    var deadrise: Float                 // radians
    var bilge: Float
    var decked: Bool = false
    var sheerLowX: Float = 0
    var sheerLowY: Float = 0
    var sheerRadius: Float = 0
    var deckCrown: Float = 0
    var deckSlope: Float = 0
    var deckCamberRadius: Float = 0
    var topRound: Float = 0             // smooth-max width where the top meets the sides
    var cap: Float                      // nothing above this height
    var wall: Float = 0                 // 0: solid
    var floor: Float = 0
    var innerRound: Float = 0
    var thwartX: [Float] = []           // up to three
    var thwartTop: [Float] = []
    var thwartHalfWidth: Float = 0
    var thwartDraft: Float = 0          // radians: each face leans this far from vertical
    /// How far below the sheer the parting line runs round the outside.
    var seamDrop: Float = 0.12

    func prim(paint: Int, inside: Int, seam: Bool) -> Prim {
        let tx: [Float] = thwartX + [0, 0, 0]
        let ty: [Float] = thwartTop + [0, 0, 0]
        let topMode: Float = decked ? 1 : 0
        let top: SIMD4<Float> = decked ? SIMD4<Float>(topMode, deckSlope, deckCrown - deckCamberRadius, deckCamberRadius)
                                       : SIMD4<Float>(topMode, sheerLowX, sheerLowY + sheerRadius, sheerRadius)
        let reachX: Float = planHalfLength + abs(planCentre) + cap * tan(flare) + 2
        let reachZ: Float = beam + cap * tan(flare) + 2
        let bc = SIMD3<Float>(0, cap / 2, 0)
        let br: Float = simd_length(SIMD3<Float>(reachX, cap / 2 + 1, reachZ))
        return Prim(kind: .hull,
                    a: SIMD3<Float>(planHalfLength, beam, planCentre), ra: transom,
                    b: SIMD3<Float>(tan(flare), planHeight, bilge), rb: tan(deadrise),
                    u: SIMD3<Float>(keelLowX, keelRadius, keelRadius), n: SIMD3<Float>(top.x, top.y, top.z),
                    size: SIMD4<Float>(wall, floor, tan(thwartDraft), topRound),
                    e: SIMD4<Float>(tx[0], tx[1], tx[2], thwartHalfWidth),
                    f: SIMD4<Float>(ty[0], ty[1], ty[2], seamDrop),
                    g: SIMD4<Float>(innerRound, Float(thwartX.count), cap, top.w),
                    paint: paint, paint2: inside, ownSeam: seam, tag: .hull,
                    boundGiven: SIMD4<Float>(bc, br))
    }
}

/// The hull's distance, and its hollow's (negative inside the hollow; +∞
/// for a solid hull). The parting line, when on, is a ridge round the
/// outside of an open hull just below its gunwale — the widest line, where a
/// mould that opens up and down must split — or, for a decked hull, the
/// caller uses the segment's mid-plane seam instead.
func hullSDF(_ s: Prim, _ q: SIMD3<Float>, seams: Bool) -> (d: Float, hollow: Float) {
    let lh: Float = s.a.x
    let bm: Float = s.a.y
    let xm: Float = s.a.z
    let xt: Float = s.ra
    let tanF: Float = s.b.x
    let y0: Float = s.b.y
    let kB: Float = s.b.z
    let tanD: Float = s.rb
    let plan2: Float = max(sdVesica2(SIMD2<Float>(q.z, q.x - xm), h: lh, w: bm), xt - q.x)
    let sF: Float = (1 + tanF * tanF).squareRoot()
    let dp: Float = (plan2 + (y0 - q.y) * tanF) / sF
    let keelC = SIMD2<Float>(s.u.x, s.u.y)
    let dk: Float = simd_distance(SIMD2<Float>(q.x, q.y), keelC) - s.u.z
    let sD: Float = (1 + tanD * tanD).squareRoot()
    let kb: Float = (dk + abs(q.z) * tanD) / sD
    let lower: Float = smax(dp, kb, kB)
    var top: Float
    let rTop: Float = s.g.w
    if s.n.x < 0.5 {
        top = rTop - simd_distance(SIMD2<Float>(q.x, q.y), SIMD2<Float>(s.n.y, s.n.z))
    } else {
        let axis: SIMD3<Float> = simd_normalize(SIMD3<Float>(1, s.n.y, 0))
        let w: SIMD3<Float> = q - SIMD3<Float>(0, s.n.z, 0)
        let radial: SIMD3<Float> = w - axis * simd_dot(w, axis)
        top = simd_length(radial) - rTop
    }
    var outer: Float = smax(lower, top, s.size.w)
    outer = max(outer, q.y - s.g.z)
    var hollow: Float = .infinity
    let wall: Float = s.size.x
    if wall > 0 {
        let vp: Float = dp + wall
        let vk: Float = kb + s.size.y
        var vd: Float = smax(vp, vk, s.g.x)
        let tanT: Float = s.size.z
        let sT: Float = (1 + tanT * tanT).squareRoot()
        let count: Int = Int(s.g.y + 0.5)
        var thw: Float = .infinity
        let xs: [Float] = [s.e.x, s.e.y, s.e.z]
        let ys: [Float] = [s.f.x, s.f.y, s.f.z]
        for i in 0..<min(count, 3) {
            let side: Float = (abs(q.x - xs[i]) - s.e.w - (ys[i] - q.y) * tanT) / sT
            thw = min(thw, max(side, q.y - ys[i]))
        }
        vd = max(vd, -thw)
        hollow = vd
        outer = max(outer, -vd)
    }
    if seams && s.ownSeam && s.n.x < 0.5 {
        let ring: Float = abs(simd_distance(SIMD2<Float>(q.x, q.y), SIMD2<Float>(s.n.y, s.n.z)) - (rTop + s.f.w)) - seamHalfWidth
        var ridge: Float = max(outer - seamHeight, ring)
        if wall > 0 { ridge = max(ridge, wall * 0.5 - hollow) }
        outer = min(outer, ridge)
    }
    return (outer, hollow)
}

func primSDF(_ s: Prim, _ p: SIMD3<Float>, seams: Bool) -> Float {
    switch s.kind {
    case .roundCone:
        return sdRoundCone(p, s.a, s.b, s.ra, s.rb)
    case .roundBox:
        let v: SIMD3<Float> = simd_cross(s.n, s.u)
        let d: SIMD3<Float> = p - s.a
        let q0 = SIMD3<Float>(abs(simd_dot(d, s.u)), abs(simd_dot(d, v)), abs(simd_dot(d, s.n)))
        let q: SIMD3<Float> = q0 - SIMD3<Float>(s.size.x, s.size.y, s.size.z) + s.size.w
        let outside: Float = simd_length(simd_max(q, SIMD3<Float>(0, 0, 0)))
        return outside + min(max(q.x, max(q.y, q.z)), 0) - s.size.w
    case .plate:
        let across: SIMD3<Float> = simd_cross(s.u, s.n)
        let d: SIMD3<Float> = p - s.a
        let e: Float = s.size.w
        let hh: Float = s.size.x - e
        let ww: Float = s.size.y - e
        let p2 = SIMD2<Float>(simd_dot(d, across), simd_dot(d, s.u))
        let d2: Float = hh >= ww ? sdVesica2(p2, h: hh, w: ww) : sdVesica2(SIMD2<Float>(p2.y, p2.x), h: ww, w: hh)
        let wv = SIMD2<Float>(d2, abs(simd_dot(d, s.n)) - (s.size.z - e))
        let outside: Float = simd_length(simd_max(wv, SIMD2<Float>(0, 0)))
        var dist: Float = min(max(wv.x, wv.y), 0) + outside - e
        if seams && s.ownSeam {
            let off: Float = abs(simd_dot(p - s.a, s.n))
            dist = min(dist, max(dist - seamHeight, off - seamHalfWidth))
        }
        return dist
    case .profile:
        return profileSDF(s, p, seams: seams)
    case .hull:
        return hullSDF(s, p, seams: seams).d
    }
}

// MARK: - the mould's parting line

/// The parting line: a ridge SEAM_H proud of the surface and 2 × SEAM_W wide,
/// where the two halves of the mould met (Wikipedia, "Injection moulding":
/// "a parting line ... usually present on the final part"). Its size is
/// MODEL: no measured figure for toys was reached. The source's example
/// tolerance, ±0.2 mm on an inch, bounds how far two mould halves can be out.
let seamHeight: Float = 0.03
let seamHalfWidth: Float = 0.045

// MARK: - paint on the surface

/// Paint laid over part of a segment, overriding its shapes' own paint where
/// a point falls inside:
///   0 sphere — centre `c`, radius `r`;
///   1 below a level — y < c.y (+ r·sin wobble round the piece);
///   2 blotch — a sphere whose radius wanders ±28%.
struct Patch {
    var c: SIMD3<Float>
    var r: Float
    var paint: Int
    var kind: Int32
}

// MARK: - segments

enum Part: Int {
    case hull = 0, oars, rig, sails, beams, deck, fittings
}

/// One rigid piece: its shapes, the blend that joins them, whether the
/// parting line runs round it (in its local z = 0 plane), and its paint.
struct Segment {
    var name: String
    var part: Part
    var limb: Int = -1
    var prims: [Prim]
    var blend: Float
    var seam: Bool
    var patches: [Patch] = []
    /// Kept from steps 58–63's renderer (a fillet to segment 0); 0 here.
    var bodyFillet: Float = 0

    /// A sphere round every shape, in the segment's frame, with room for the
    /// blend's bulge (at most k/4) and the seam's ridge.
    var bound: (centre: SIMD3<Float>, radius: Float) {
        var lo = SIMD3<Float>(repeating: .infinity)
        var hi = SIMD3<Float>(repeating: -.infinity)
        for p in prims {
            let b = p.bound
            lo = simd_min(lo, b.centre - b.radius)
            hi = simd_max(hi, b.centre + b.radius)
        }
        let c: SIMD3<Float> = (lo + hi) / 2
        var r: Float = 0
        for p in prims { r = max(r, simd_distance(p.bound.centre, c) + p.bound.radius) }
        return (c, r + blend / 4 + bodyFillet + seamHeight + 0.01)
    }

    /// Distance from a point in the segment's frame, and which shape is
    /// nearest. The same arithmetic as the kernel's `segSDF`.
    func sdf(local q: SIMD3<Float>, seams: Bool) -> (d: Float, prim: Int) {
        var d: Float = .infinity
        var nearest: Int = 0
        var best: Float = .infinity
        for (i, s) in prims.enumerated() {
            let di: Float = primSDF(s, q, seams: seams)
            if di < best { best = di; nearest = i }
            d = d == .infinity ? di : smin(d, di, blend)
        }
        if seams && seam { d = min(d, max(d - seamHeight, abs(q.z) - seamHalfWidth)) }
        return (d, nearest)
    }
}

/// A toy in one pose: its segments and where each one is.
struct PosedToy {
    var segments: [Segment]
    var frames: [Frame]
    var paints: [Paint]
    var seams: Bool

    /// Distance to the toy from a world point, the nearest segment and the
    /// shape in it. No bound-sphere shortcut: the reference for the kernel's.
    func sdf(_ p: SIMD3<Float>) -> (d: Float, segment: Int, prim: Int) {
        var d: Float = .infinity
        var seg: Int = -1
        var prim: Int = -1
        for (i, s) in segments.enumerated() {
            let r = s.sdf(local: frames[i].toLocal(p), seams: seams)
            if r.d < d { d = r.d; seg = i; prim = r.prim }
        }
        return (d, seg, prim)
    }

    /// Which paint is at a surface point: the nearest shape's (a hull's
    /// inside paint where the point is on its hollow), unless a patch covers
    /// it. The same rules as the kernel's `paintAt`.
    func paint(at p: SIMD3<Float>) -> Int {
        let r = sdf(p)
        guard r.segment >= 0 else { return -1 }
        let s: Segment = segments[r.segment]
        let q: SIMD3<Float> = frames[r.segment].toLocal(p)
        let pr: Prim = s.prims[r.prim]
        var pi: Int = pr.paint
        if pr.kind == .hull && pr.paint2 >= 0 && hullSDF(pr, q, seams: false).hollow < pr.size.x * 0.5 { pi = pr.paint2 }
        for c in s.patches {
            var inside: Bool = false
            switch c.kind {
            case 0: inside = simd_distance(q, c.c) < c.r
            case 1: inside = q.y < c.c.y + c.r * sin(atan2(q.z, q.x) * 5 + 0.7)
            case 2:
                let a: Float = sin(0.9 * q.x + 1.3) * sin(0.8 * q.y + 0.4)
                let wob: Float = a * sin(1.0 * q.z + 2.1)
                inside = simd_distance(q, c.c) < c.r * (1 + 0.28 * wob)
            default: inside = false
            }
            if inside { pi = c.paint }
        }
        return pi
    }

    /// The toy's extent along a world direction: the lowest and highest
    /// values of dot(point, dir) over its surface. Round cones and boxes
    /// exactly; plates, profiles and hulls by marching their own distance
    /// in from outside over a fine grid (to ~0.01 mm).
    func extent(along dir0: SIMD3<Float>) -> (lo: Float, hi: Float) {
        let dir: SIMD3<Float> = simd_normalize(dir0)
        var lo: Float = .infinity
        var hi: Float = -.infinity
        for (i, s) in segments.enumerated() {
            let f: Frame = frames[i]
            let ld: SIMD3<Float> = f.dirToLocal(dir)
            let base: Float = simd_dot(f.o, dir)
            for p in s.prims {
                let e: (Float, Float) = primExtent(p, ld)
                lo = min(lo, base + e.0)
                hi = max(hi, base + e.1)
            }
        }
        return (lo, hi)
    }
}

/// How far along a unit local direction a shape reaches, the highest value
/// of dot(point, d) on it, found by marching in from beyond its bound.
func marchedSupport(_ s: Prim, _ d: SIMD3<Float>) -> Float {
    let b = s.bound
    let helper: SIMD3<Float> = abs(d.y) < 0.9 ? SIMD3<Float>(0, 1, 0) : SIMD3<Float>(1, 0, 0)
    let e1: SIMD3<Float> = simd_normalize(simd_cross(d, helper))
    let e2: SIMD3<Float> = simd_cross(d, e1)
    let far: Float = b.radius + 1
    func hit(_ s1: Float, _ s2: Float) -> Float {
        let start: SIMD3<Float> = b.centre + d * far + e1 * s1 + e2 * s2
        var t: Float = 0
        for _ in 0..<300 {
            let dist: Float = primSDF(s, start - d * t, seams: false)
            if dist < 1e-4 { return simd_dot(start - d * t, d) }
            t += dist
            if t > 2 * far { break }
        }
        return -.infinity
    }
    // Coarse, then finer round the best, twice.
    var best: Float = -.infinity
    var c1: Float = 0
    var c2: Float = 0
    var span: Float = b.radius
    var step: Float = max(b.radius / 60, 0.05)
    for _ in 0..<3 {
        let n: Int = Int((span / step).rounded(.up))
        let o1: Float = c1
        let o2: Float = c2
        for i in -n...n {
            for j in -n...n {
                let s1: Float = o1 + Float(i) * step
                let s2: Float = o2 + Float(j) * step
                let h: Float = hit(s1, s2)
                if h > best { best = h; c1 = s1; c2 = s2 }
            }
        }
        span = step * 2
        step = step / 8
    }
    return best
}

/// A shape's extent along a unit local direction.
func primExtent(_ s: Prim, _ d: SIMD3<Float>) -> (Float, Float) {
    switch s.kind {
    case .roundCone:
        let a: Float = simd_dot(s.a, d)
        let b: Float = simd_dot(s.b, d)
        return (min(a - s.ra, b - s.rb), max(a + s.ra, b + s.rb))
    case .roundBox:
        let v: SIMD3<Float> = simd_cross(s.n, s.u)
        let c: Float = simd_dot(s.a, d)
        let hx: Float = abs(simd_dot(s.u, d)) * (s.size.x - s.size.w)
        let hy: Float = abs(simd_dot(v, d)) * (s.size.y - s.size.w)
        let hz: Float = abs(simd_dot(s.n, d)) * (s.size.z - s.size.w)
        let r: Float = hx + hy + hz + s.size.w
        return (c - r, c + r)
    case .plate, .profile, .hull:
        return (-marchedSupport(s, -d), marchedSupport(s, d))
    }
}

// MARK: - a boat, placed

/// A toy boat's design: its paints and its segments, each with the frame it
/// sits in. The boat stands on the table on its keel.
struct BoatDesign {
    var name: String
    var paints: [Paint]
    var segments: [Segment]
    var frames: [Frame]

    func posed(seams: Bool = activeMutant != .noSeam) -> PosedToy {
        PosedToy(segments: segments, frames: frames, paints: paints, seams: seams)
    }
}

// MARK: - size, against Russell's spec

/// Russell's spec for these toys: "like 2-3 inches lengthwise" — 50.8 to
/// 76.2 mm, measured along the boat's length, bow to stern.
let specLengthRange: ClosedRange<Float> = 50.8...76.2

/// Wall thickness a moulder recommends: Protolabs' design guidelines for
/// plastic injection moulding (protolabs.com, "Plastic Injection Molding
/// Design Guidelines", checked 2026-09-28) give 0.035–0.150 in for
/// polystyrene and 0.045–0.140 in for ABS; PVC is not in their table, so the
/// polystyrene range stands in — 0.89 to 3.81 mm.
let mouldWallRange: ClosedRange<Float> = 0.889...3.81
/// The draft Protolabs recommends "for most situations": 2°; 0.5° is their
/// least, on vertical faces.
let draftMostDegrees: Float = 2.0
let draftLeastDegrees: Float = 0.5
