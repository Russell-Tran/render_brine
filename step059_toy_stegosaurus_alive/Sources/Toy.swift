// A toy as the render and the tests see it: a set of RIGID segments — body,
// head, tail, and three pieces per leg — each made of a few simple shapes in
// its own frame, placed in the world by one rigid transform. A still places
// them once, in the pose the toy was moulded in. An animation moves the
// frames and never the shapes: a hard plastic piece can turn about a joint,
// it cannot bend or stretch.
//
// Millimetres throughout. The world: y up, the table top is y = 0.
//
// Every shape has an exact distance, or a lower bound, outside it:
//   * round cone (a capsule whose two end radii differ) — exact;
//   * rounded box — exact;
//   * plate: a lens (vesica) outline extruded to a thickness, edges rounded —
//     exact, from the exact 2D vesica distance.
// Within one segment the shapes are joined with a quadratic smooth-min whose
// blend width is a CONSTANT of that segment (never a local slope), which is
// 1-Lipschitz when its inputs are; segments are joined with plain min. The
// mould's parting line is max(d − h, |z| − w) on the segment's own mid-plane,
// unioned in with min: a max and a min of 1-Lipschitz fields, so it is too.

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

    static let identity = Frame(o: .zero, x: SIMD3(1, 0, 0), y: SIMD3(0, 1, 0), z: SIMD3(0, 0, 1))

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
    /// A frame whose y axis runs along `up` and whose x axis is as close to
    /// `forward` as a right angle to y allows.
    static func along(origin: SIMD3<Float>, up: SIMD3<Float>, forward: SIMD3<Float>) -> Frame {
        let yy: SIMD3<Float> = simd_normalize(up)
        let f: SIMD3<Float> = forward - yy * simd_dot(forward, yy)
        let xx: SIMD3<Float> = simd_normalize(f)
        return Frame(o: origin, x: xx, y: yy, z: simd_cross(xx, yy))
    }
    /// Level, facing `yaw` radians anticlockwise seen from above (yaw 0 faces +x).
    static func level(origin: SIMD3<Float>, yaw: Float) -> Frame {
        let xx = SIMD3<Float>(cos(yaw), 0, -sin(yaw))
        let yy = SIMD3<Float>(0, 1, 0)
        return Frame(o: origin, x: xx, y: yy, z: simd_cross(xx, yy))
    }
    /// This frame turned by `angle` about a local axis through a local point.
    func rotated(about axis: SIMD3<Float>, through pivot: SIMD3<Float>, by angle: Float) -> Frame {
        let q = simd_quatf(angle: angle, axis: simd_normalize(dirToWorld(axis)))
        let pw: SIMD3<Float> = toWorld(pivot)
        return Frame(o: pw + q.act(o - pw), x: q.act(x), y: q.act(y), z: q.act(z))
    }
}

// MARK: - shapes

/// What a shape is, for the tests that count them off the drawn toy.
enum PrimTag: Int {
    case none = 0, plate, spike, toe, eye
}

enum PrimKind: Int32 {
    case roundCone = 0
    case roundBox = 1
    case plate = 2
}

/// One shape, in its segment's frame.
///   roundCone: sphere radius `ra` at `a` swept to radius `rb` at `b`.
///   roundBox: centre `a`, axes `u` (x) and `n` (z), half-extents `size.xyz`,
///             edge radius `size.w`.
///   plate: centre `a`; its outline's long axis `u`; its thickness axis `n`;
///          `size` = (half-height along u, half-width across, half-thickness,
///          edge radius). The outline is a vesica — the lens two circles make —
///          pointed at both ends of its long axis.
struct Prim {
    var kind: PrimKind
    var a: SIMD3<Float>
    var ra: Float = 0
    var b: SIMD3<Float> = .zero
    var rb: Float = 0
    var u = SIMD3<Float>(0, 1, 0)
    var n = SIMD3<Float>(0, 0, 1)
    var size = SIMD4<Float>(0, 0, 0, 0)
    var paint: Int
    /// A plate's own parting line, round its rim in its mid-plane.
    var ownSeam: Bool = false
    var tag: PrimTag = .none

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
    static func box(_ c: SIMD3<Float>, half: SIMD3<Float>, edge: Float, xAxis: SIMD3<Float> = SIMD3(1, 0, 0),
                    zAxis: SIMD3<Float> = SIMD3(0, 0, 1), paint: Int) -> Prim {
        Prim(kind: .roundBox, a: c, u: simd_normalize(xAxis), n: simd_normalize(zAxis),
             size: SIMD4<Float>(half.x, half.y, half.z, edge), paint: paint)
    }
    static func plate(_ c: SIMD3<Float>, up: SIMD3<Float>, normal: SIMD3<Float>, halfHeight: Float, halfWidth: Float,
                      halfThickness: Float, edge: Float, paint: Int) -> Prim {
        let nn: SIMD3<Float> = simd_normalize(normal)
        let uu: SIMD3<Float> = simd_normalize(up - nn * simd_dot(up, nn))
        return Prim(kind: .plate, a: c, u: uu, n: nn,
                    size: SIMD4<Float>(halfHeight, halfWidth, halfThickness, edge), paint: paint, ownSeam: true)
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
        }
    }
}

/// Exact distance to a round cone (Quílez's formulation, the same one the
/// kernel uses).
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

func primSDF(_ s: Prim, _ p: SIMD3<Float>) -> Float {
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
        return min(max(wv.x, wv.y), 0) + outside - e
    }
}

/// Quadratic smooth minimum with a constant blend width k (Quílez). Its
/// value is a convex mix of its inputs' gradients, so it is 1-Lipschitz when
/// they are, and it never exceeds min(a, b): outside, a lower bound.
func smin(_ a: Float, _ b: Float, _ k: Float) -> Float {
    if k <= 0 { return min(a, b) }
    let h: Float = max(k - abs(a - b), 0) / k
    return min(a, b) - h * h * k * 0.25
}

// MARK: - the mould's parting line

/// The parting line: a ridge SEAM_H proud of the surface and 2 × SEAM_W wide,
/// where the two halves of the mould met. Injection-moulded parts carry one
/// (Wikipedia, "Injection moulding": "a parting line ... usually present on
/// the final part"). Its size is MODEL: no measured figure for toys was
/// reached. The source's example tolerance for a moulded part, ±0.2 mm on
/// an inch, bounds how far two mould halves can be out; a flash line of a
/// few hundredths of a millimetre catches the light without being a flaw.
let seamHeight: Float = 0.03
let seamHalfWidth: Float = 0.045

// MARK: - paint on the surface

/// Paint laid over part of a segment, overriding its shapes' own paint where
/// a point falls inside:
///   0 sphere — centre `c`, radius `r`;
///   1 below a level — y < c.y, wobbling by ±r round the piece (a painted
///     hoof's or foot's edge: hand-painted edges are never quite level);
///   2 blotch — a sphere whose radius wanders ±35% (a painted coat pattern).
struct Patch {
    var c: SIMD3<Float>
    var r: Float
    var paint: Int
    var kind: Int32
}

// MARK: - segments

enum Part: Int {
    case body = 0, head, tail, plates, upperLeg, lowerLeg, foot
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
        return (c, r + blend / 4 + seamHeight + 0.01)
    }

    /// Distance from a point in the segment's frame, and which shape is
    /// nearest. The same arithmetic as the kernel's `segSDF`.
    func sdf(local q: SIMD3<Float>, seams: Bool) -> (d: Float, prim: Int) {
        var d: Float = .infinity
        var nearest: Int = 0
        var best: Float = .infinity
        for (i, s) in prims.enumerated() {
            var di: Float = primSDF(s, q)
            if di < best { best = di; nearest = i }
            if seams && s.ownSeam {
                let off: Float = abs(simd_dot(q - s.a, s.n))
                di = min(di, max(di - seamHeight, off - seamHalfWidth))
            }
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
    /// shape in it. No bound-sphere shortcut: this is the reference the
    /// kernel's shortcut is tested against.
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

    /// Which paint is at a surface point: the nearest shape's, unless a patch
    /// covers it. The same rules as the kernel's `paintAt`.
    func paint(at p: SIMD3<Float>) -> Int {
        let r = sdf(p)
        guard r.segment >= 0 else { return -1 }
        let s: Segment = segments[r.segment]
        var pi: Int = s.prims[r.prim].paint
        let q: SIMD3<Float> = frames[r.segment].toLocal(p)
        for c in s.patches {
            var inside: Bool = false
            switch c.kind {
            case 0: inside = simd_distance(q, c.c) < c.r
            case 1: inside = q.y < c.c.y + c.r * sin(atan2(q.z, q.x) * 5 + 0.7)
            case 2:
                let a: Float = sin(3.1 * q.x + 1.3) * sin(2.7 * q.y + 0.4)
                let wob: Float = a * sin(3.3 * q.z + 2.1)
                inside = simd_distance(q, c.c) < c.r * (1 + 0.35 * wob)
            default: inside = q.x > c.c.x + c.r * sin(q.z * 9 + q.y * 5)
            }
            if inside { pi = c.paint }
        }
        return pi
    }

    /// Is a world point inside segment i (strictly, by `margin`)?
    func inside(_ p: SIMD3<Float>, segment i: Int, margin: Float = 0) -> Bool {
        segments[i].sdf(local: frames[i].toLocal(p), seams: false).d < -margin
    }

    /// The toy's extent along a world direction: the lowest and highest
    /// values of dot(point, dir) over every surface point, from each shape's
    /// exact extreme. Blends and seams add a few hundredths at most, and
    /// never at an extremity of the toy.
    func extent(along dir0: SIMD3<Float>) -> (lo: Float, hi: Float) {
        let dir: SIMD3<Float> = simd_normalize(dir0)
        var lo: Float = .infinity
        var hi: Float = -.infinity
        for (i, s) in segments.enumerated() {
            let f: Frame = frames[i]
            let ld = SIMD3<Float>(simd_dot(dir, f.x), simd_dot(dir, f.y), simd_dot(dir, f.z))  // dir in local
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
    case .plate:
        // The outline sampled finely, extruded, and rounded by the edge.
        let across: SIMD3<Float> = simd_cross(s.u, s.n)
        let e: Float = s.size.w
        let hh: Float = s.size.x - e
        let ww: Float = s.size.y - e
        let t: Float = abs(simd_dot(s.n, d)) * (s.size.z - e)
        var lo: Float = .infinity
        var hi: Float = -.infinity
        for k in 0..<720 {
            let ang: Float = Float(k) / 720 * 2 * .pi
            // A point on the vesica's outline in direction ang, found by
            // bisection on the exact 2D distance.
            let dir2 = SIMD2<Float>(cos(ang), sin(ang))
            var r0: Float = 0
            var r1: Float = max(hh, ww) * 1.01
            for _ in 0..<30 {
                let m: Float = (r0 + r1) / 2
                let pp: SIMD2<Float> = dir2 * m
                let dd: Float = hh >= ww ? sdVesica2(pp, h: hh, w: ww) : sdVesica2(SIMD2<Float>(pp.y, pp.x), h: ww, w: hh)
                if dd < 0 { r0 = m } else { r1 = m }
            }
            let q: SIMD3<Float> = s.a + across * (dir2.x * r0) + s.u * (dir2.y * r0)
            let v: Float = simd_dot(q, d)
            lo = min(lo, v - t - e)
            hi = max(hi, v + t + e)
        }
        return (lo, hi)
    }
}

// MARK: - the rig

/// One leg: where it joins the body, how long its two pieces are, which way
/// its middle joint points, and the foot at its end. Upper and lower pieces
/// are drawn along their frames' −y axis from the joint at the origin; the
/// foot's shapes sit in a level frame whose origin is the foot's centre on
/// the ground, facing forward.
struct LegSpec {
    var name: String            // "LF", "RF", "LH", "RH"
    var fore: Bool
    var side: Float             // +1 right (+z), −1 left
    var hip: SIMD3<Float>       // in the body's frame
    var upper: Float            // hip to middle joint
    var lower: Float            // middle joint to the ankle, fetlock or wrist
    var ankleHeight: Float      // that joint's height above the foot's ground centre
    var bendForward: Bool       // the middle joint's point faces forward (fore knee, dinosaur knee)
    var restFoot: SIMD3<Float>  // foot's ground centre at rest, in the body's frame (y = −body height)
    var upperPrims: [Prim]
    var lowerPrims: [Prim]
    var footPrims: [Prim]
    var footPatches: [Patch] = []
    var lowerPatches: [Patch] = []
}

/// A toy's design: its body pieces (in the body's frame), the neck and tail
/// pivots, its legs, and how it stands at rest.
struct ToyDesign {
    var name: String
    var paints: [Paint]
    var body: Segment
    var head: Segment               // in the head's frame, origin at the neck pivot
    var headPivot: SIMD3<Float>     // in the body frame
    var tail: Segment?              // origin at the tail pivot
    var tailPivot: SIMD3<Float>
    var extra: [Segment]            // more pieces fixed to the body (the plates)
    var legs: [LegSpec]
    /// Height of the body frame's origin above the table at rest.
    var bodyHeight: Float
}

/// Where the toy is and how its movable pieces are set.
struct Pose {
    var body: Frame                 // the body frame in the world
    var headYaw: Float = 0
    var headPitch: Float = 0
    var tailYaw: Float = 0
    var feet: [Frame]               // each foot's level frame on the ground
}

extension ToyDesign {
    /// The pose the toy was moulded in, standing at `at` facing `yaw`.
    func restPose(at: SIMD3<Float> = .zero, yaw: Float = 0) -> Pose {
        let body = Frame.level(origin: at + SIMD3<Float>(0, bodyHeight, 0), yaw: yaw)
        let feet: [Frame] = legs.map { l in
            Frame.level(origin: body.toWorld(l.restFoot), yaw: yaw)
        }
        return Pose(body: body, feet: feet)
    }

    /// Two-piece leg reaching from the hip to the ankle: where the middle
    /// joint goes. The joint bends in the plane that holds hip, ankle and the
    /// body's forward direction; the law of cosines puts it there.
    func middleJoint(_ l: LegSpec, hip h: SIMD3<Float>, ankle p: SIMD3<Float>, forward f: SIMD3<Float>) -> SIMD3<Float> {
        let d0: Float = simd_distance(h, p)
        let d: Float = min(d0, l.upper + l.lower - 1e-4)
        let u: SIMD3<Float> = (p - h) / max(d0, 1e-6)
        let w0: SIMD3<Float> = f - u * simd_dot(f, u)
        let w: SIMD3<Float> = simd_normalize(w0)
        let ca: Float = (l.upper * l.upper + d * d - l.lower * l.lower) / (2 * l.upper * d)
        let alpha: Float = acos(min(max(ca, -1), 1))
        let sgn: Float = l.bendForward ? 1 : -1
        let along: SIMD3<Float> = u * (l.upper * cos(alpha))
        return h + along + w * (sgn * l.upper * sin(alpha))
    }

    /// Every segment of the toy, and where each is, in a pose.
    func posed(_ pose: Pose, seams: Bool = activeMutant != .noSeam) -> PosedToy {
        var segs: [Segment] = [body]
        var frames: [Frame] = [pose.body]
        // Head: turned about the neck pivot, yaw about the body's up, then pitch.
        var hf: Frame = pose.body
        hf.o = pose.body.toWorld(headPivot)
        hf = hf.rotated(about: SIMD3<Float>(0, 1, 0), through: .zero, by: pose.headYaw)
        hf = hf.rotated(about: SIMD3<Float>(0, 0, 1), through: .zero, by: pose.headPitch)
        segs.append(head)
        frames.append(hf)
        if let t = tail {
            var tf: Frame = pose.body
            tf.o = pose.body.toWorld(tailPivot)
            tf = tf.rotated(about: SIMD3<Float>(0, 1, 0), through: .zero, by: pose.tailYaw)
            segs.append(t)
            frames.append(tf)
        }
        for e in extra {
            segs.append(e)
            frames.append(pose.body)
        }
        for (j, l) in legs.enumerated() {
            let foot: Frame = pose.feet[j]
            let h: SIMD3<Float> = pose.body.toWorld(l.hip)
            let p: SIMD3<Float> = foot.toWorld(SIMD3<Float>(0, l.ankleHeight, 0))
            let k: SIMD3<Float> = middleJoint(l, hip: h, ankle: p, forward: foot.x)
            let upperF = Frame.along(origin: h, up: h - k, forward: foot.x)
            let lowerF = Frame.along(origin: k, up: k - p, forward: foot.x)
            segs.append(Segment(name: l.name + " upper", part: .upperLeg, limb: j, prims: l.upperPrims, blend: 0, seam: true))
            frames.append(upperF)
            segs.append(Segment(name: l.name + " lower", part: .lowerLeg, limb: j, prims: l.lowerPrims, blend: 0.3,
                                seam: true, patches: l.lowerPatches))
            frames.append(lowerF)
            segs.append(Segment(name: l.name + " foot", part: .foot, limb: j, prims: l.footPrims, blend: 0.15,
                                seam: true, patches: l.footPatches))
            frames.append(foot)
        }
        return PosedToy(segments: segs, frames: frames, paints: paints, seams: seams)
    }
}

// MARK: - size, against Russell's spec

/// Russell's spec for these toys: "like 2-3 inches lengthwise" — 50.8 to
/// 76.2 mm from nose to tail tip, measured along the animal's length.
let specLengthRange: ClosedRange<Float> = 50.8...76.2
