// The ray tracer. Three things here are new to the project.
//
// A rounded box, analytic
// -----------------------
// A bus seat and a window frame are the same shape: a box with its corners
// taken off. Until now this renderer had spheres and cylinders, so that shape
// had to be faked. The intersector is Inigo Quilez's exact one
// (https://iquilezles.org/articles/intersectors/): clip against the box grown
// by the corner radius, and if the entry point is not on a flat face, solve the
// three edge cylinders and the corner sphere in the first octant. No iteration,
// no distance field, no marching.
//
// The aperture mask
// -----------------
// The camera is bolted to the bus, so the interior is rigid for the whole
// render. A pixel can therefore only change if its ray leaves the bus through a
// window, or if it lands on a piece of interior that the sun can reach through
// a window — in which case a tree going past can shadow it. Both tests are made
// against the BUS ALONE, so the mask is a property of the geometry and not of
// any particular frame, and it is exact rather than conservative-by-guesswork.
// Frame 0 is traced whole; later frames trace only the masked pixels and copy
// the rest.
//
// Motion blur for nothing
// -----------------------
// The world transform is a rigid motion, so applying it to the ray is the same
// as applying it to the vertices. That means each of the four anti-aliasing
// samples a pixel already takes can be given its OWN time inside the frame's
// exposure, at the cost of one vector subtract. Blur length then comes out as
// f·d·v/z² — proportional to angular rate, and so inversely proportional to
// distance — without a single extra ray.
//
// Amanatides & Woo, "A Fast Voxel Traversal Algorithm for Ray Tracing",
// Eurographics (1987), for the grid walk, as in every step since 8a.

import Foundation
import Metal
import simd

func radians(_ degrees: Float) -> Float { degrees * .pi / 180 }

// MARK: - shapes

enum ShapeKind: UInt32 {
    case sphere = 0
    case cylinder = 1
    case roundedBox = 2
}

/// One primitive, laid out the way the GPU reads it.
///   sphere       a.xyz centre,   a.w radius
///   cylinder     a.xyz base,     a.w radius,       b.xyz base→top axis
///   rounded box  a.xyz centre,   a.w corner radius, b.xyz half extents of the inner box
struct GPUShape {
    var a: SIMD4<Float>
    var b: SIMD4<Float>
    var color: SIMD4<Float>
    var meta: SIMD4<UInt32>          // x = kind, y = tag

    static func sphere(center: SIMD3<Float>, radius: Float, color: SIMD3<Float>,
                       tag: UInt32 = 0) -> GPUShape {
        GPUShape(a: SIMD4(center, radius), b: SIMD4(repeating: 0), color: SIMD4(color, 0),
                 meta: SIMD4(ShapeKind.sphere.rawValue, tag, 0, 0))
    }

    static func cylinder(base: SIMD3<Float>, axis: SIMD3<Float>, radius: Float,
                         color: SIMD3<Float>, tag: UInt32 = 0) -> GPUShape {
        GPUShape(a: SIMD4(base, radius), b: SIMD4(axis, 0), color: SIMD4(color, 0),
                 meta: SIMD4(ShapeKind.cylinder.rawValue, tag, 0, 0))
    }

    /// `half` is the half extent of the inner box; the shape reaches `radius`
    /// further in every direction, so the whole thing is `half + radius` across.
    static func box(center: SIMD3<Float>, half: SIMD3<Float>, radius: Float,
                    color: SIMD3<Float>, tag: UInt32 = 0) -> GPUShape {
        let inner = simd_max(half - SIMD3(repeating: radius), SIMD3(repeating: 1e-4))
        return GPUShape(a: SIMD4(center, radius), b: SIMD4(inner, 0), color: SIMD4(color, 0),
                        meta: SIMD4(ShapeKind.roundedBox.rawValue, tag, 0, 0))
    }

    var kind: ShapeKind { ShapeKind(rawValue: meta.x) ?? .sphere }

    func bounds() -> (lo: SIMD3<Float>, hi: SIMD3<Float>) {
        let p = SIMD3(a.x, a.y, a.z)
        switch kind {
        case .sphere:
            let r = SIMD3<Float>(repeating: a.w)
            return (p - r, p + r)
        case .cylinder:
            let q = p + SIMD3(b.x, b.y, b.z)
            let r = SIMD3<Float>(repeating: a.w)
            return (simd_min(p, q) - r, simd_max(p, q) + r)
        case .roundedBox:
            let e = SIMD3(b.x, b.y, b.z) + SIMD3(repeating: a.w)
            return (p - e, p + e)
        }
    }
}

// MARK: - camera

struct Camera {
    var origin: SIMD3<Float>
    var target: SIMD3<Float>
    var tanHalfFOV: Float

    var forward: SIMD3<Float> { simd_normalize(target - origin) }
    var right: SIMD3<Float> {
        let f = forward
        let up = abs(f.y) > 0.999 ? SIMD3<Float>(0, 0, 1) : SIMD3<Float>(0, 1, 0)
        return simd_normalize(simd_cross(f, up))
    }
    var up: SIMD3<Float> { simd_cross(right, forward) }

    init(origin: SIMD3<Float>, target: SIMD3<Float>, fov: Float) {
        self.origin = origin
        self.target = target
        self.tanHalfFOV = tan(radians(fov / 2))
    }

    /// Where a point lands in a width × height image, in pixels from the top left.
    func project(_ p: SIMD3<Float>, width: Int, height: Int) -> SIMD2<Float> {
        let v = p - origin
        let z: Float = simd_dot(v, forward)
        let aspect: Float = Float(width) / Float(height)
        let ndcX: Float = simd_dot(v, right) / (z * tanHalfFOV * aspect)
        let ndcY: Float = simd_dot(v, up) / (z * tanHalfFOV)
        let px: Float = (ndcX + 1) / 2 * Float(width)
        let py: Float = (1 - ndcY) / 2 * Float(height)
        return SIMD2(px, py)
    }
}

// MARK: - GPU structures

struct GPUParams {
    var origin: SIMD4<Float>       // xyz camera origin, w tanHalfFOV
    var forward: SIMD4<Float>      // xyz, w aspect
    var right: SIMD4<Float>        // xyz, w exposure in seconds
    var up: SIMD4<Float>           // xyz, w speed in m/s
    var sun: SIMD4<Float>          // xyz direction toward the sun, w strength
    var sunColor: SIMD4<Float>     // rgb, w ambient strength
    var skyTop: SIMD4<Float>       // rgb, w travel wrapped into one loop
    var skyHorizon: SIMD4<Float>   // rgb, w the road surface's y
    var glass: SIMD4<Float>        // rgb glazing tint, w road centreline x
    var road: SIMD4<Float>         // x half width, y dash period, z dash length, w AO reach
    var counts: SIMD4<UInt32>      // x width, y height, z bus shapes, w world shapes
    var flags: SIMD4<UInt32>       // x samples per side, y AO probes, z mode, w list length
}

struct GPUGrid {
    var origin: SIMD4<Float>
    var cell: SIMD4<Float>
    var dims: SIMD4<UInt32>
}

enum RenderMode: UInt32 {
    case full = 0            // every pixel
    case mask = 1            // write the aperture-and-sunlight mask
    case list = 2            // only the pixels in the index list
}

let keyLightWorld = SIMD3<Float>(0, 1, 0)

// MARK: - the kernel

let raytraceKernelSource = """
    #include <metal_stdlib>
    using namespace metal;

    struct Shape { float4 a; float4 b; float4 color; uint4 meta; };
    struct Grid { float4 origin; float4 cell; uint4 dims; };
    struct Params {
        float4 origin; float4 forward; float4 right; float4 up;
        float4 sun; float4 sunColor; float4 skyTop; float4 skyHorizon;
        float4 glass; float4 road;
        uint4 counts; uint4 flags;
    };

    constant uint KIND_SPHERE = 0u;
    constant uint KIND_CYLINDER = 1u;
    constant uint KIND_BOX = 2u;
    constant float HAZE_RANGE = 1500.0;
    constant float3 HAZE_COLOR = float3(0.694, 0.757, 0.812);

    constant float BAYER[16] = { 0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5 };
    inline float3 dither(uint2 gid) {
        float n = (BAYER[(gid.y & 3u) * 4u + (gid.x & 3u)] + 0.5) / 16.0 - 0.5;
        return float3(n * 2.0 / 255.0);
    }

    // A per-pixel offset for the shutter, so that four time samples read as
    // grain rather than as four ghosts. It depends on the pixel alone — never
    // on the frame — so the noise does not crawl, and frame 120 can still come
    // out bit-identical to frame 0.
    inline float hash01(uint2 g) {
        uint h = g.x * 1973u + g.y * 9277u + 26699u;
        h = (h ^ (h >> 15)) * 2246822519u;
        h = (h ^ (h >> 13)) * 3266489917u;
        h ^= (h >> 16);
        return float(h & 0x00FFFFFFu) / float(0x01000000u);
    }

    inline float3 safeInverse(float3 rd) {
        float3 d = select(rd, float3(1e-9), abs(rd) < float3(1e-9));
        return 1.0 / d;
    }

    // ---- primitives

    inline bool hitSphere(float3 ro, float3 rd, float3 c, float r, float best,
                          thread float &t, thread float3 &n) {
        if (r <= 0.0) return false;
        float3 oc = ro - c;
        float b = dot(oc, rd);
        float h = b * b - (dot(oc, oc) - r * r);
        if (h < 0.0) return false;
        float tt = -b - sqrt(h);
        if (tt <= 1e-4 || tt >= best) return false;
        t = tt;
        n = (ro + tt * rd - c) / r;
        return true;
    }

    // A capped cylinder from `pa` along `ba`, radius `ra`.
    inline bool hitCylinder(float3 ro, float3 rd, float3 pa, float3 ba, float ra, float best,
                            thread float &t, thread float3 &n) {
        float3 oc = ro - pa;
        float baba = dot(ba, ba);
        if (baba <= 0.0) return false;
        float bard = dot(ba, rd);
        float baoc = dot(ba, oc);
        float k2 = baba - bard * bard;
        float k1 = baba * dot(oc, rd) - baoc * bard;
        float k0 = baba * dot(oc, oc) - baoc * baoc - ra * ra * baba;

        if (k2 < 1e-9) {
            // The ray runs along the axis: only the end caps can be hit.
            if (abs(bard) < 1e-9) return false;
            float d2 = dot(oc, oc) - baoc * baoc / baba;
            if (d2 > ra * ra) return false;
            float t0 = (0.0 - baoc) / bard;
            float t1 = (baba - baoc) / bard;
            float near = min(t0, t1);
            float far = max(t0, t1);
            float tt = near > 1e-4 ? near : far;
            if (tt <= 1e-4 || tt >= best) return false;
            t = tt;
            n = ba * ((tt == t0) ? -1.0 : 1.0) / sqrt(baba);
            return true;
        }

        float h = k1 * k1 - k2 * k0;
        if (h < 0.0) return false;
        h = sqrt(h);
        float tt = (-k1 - h) / k2;
        float y = baoc + tt * bard;
        if (y > 0.0 && y < baba) {
            if (tt <= 1e-4 || tt >= best) return false;
            t = tt;
            n = (oc + tt * rd - ba * y / baba) / ra;
            return true;
        }
        if (abs(bard) < 1e-9) return false;
        float capTarget = (y < 0.0) ? 0.0 : baba;
        tt = (capTarget - baoc) / bard;
        if (abs(k1 + k2 * tt) >= h) return false;
        if (tt <= 1e-4 || tt >= best) return false;
        t = tt;
        n = ba * ((y < 0.0) ? -1.0 : 1.0) / sqrt(baba);
        return true;
    }

    // The new primitive: a box of half extents `siz` with every corner and edge
    // rounded off by `rad`. Exact, following Quilez. A ray whose origin is
    // inside the shape is reported as a miss; nothing in this scene starts
    // inside a solid, and a test checks that.
    inline bool hitRoundBox(float3 ro, float3 rd, float3 c, float3 siz, float rad, float best,
                            thread float &t, thread float3 &n) {
        float3 o = ro - c;
        float3 m = safeInverse(rd);
        float3 k = abs(m) * (siz + rad);
        float3 nn = m * o;
        float3 ta = -nn - k;
        float3 tb = -nn + k;
        float tN = max(max(ta.x, ta.y), ta.z);
        float tF = min(min(tb.x, tb.y), tb.z);
        if (tN > tF || tF < 0.0 || tN <= 1e-4 || tN >= best) return false;

        float tt = tN;
        float3 p = o + tt * rd;
        float3 s = sign(p);
        // Everything below works in the first octant.
        float3 op = o * s;
        float3 dp = rd * s;
        float3 face = p * s - siz;
        float3 mx = max(face.xyz, face.yzx);
        if (min(min(mx.x, mx.y), mx.z) < 0.0) {
            t = tt;
            n = s * normalize(max(abs(p) - siz, float3(1e-9)));
            return true;
        }

        float3 oc = op - siz;
        float3 dd = dp * dp;
        float3 oo = oc * oc;
        float3 od = oc * dp;
        float ra2 = rad * rad;
        tt = 1e20;
        {   // the corner sphere
            float b = od.x + od.y + od.z;
            float cc = oo.x + oo.y + oo.z - ra2;
            float h = b * b - cc;
            if (h > 0.0) tt = -b - sqrt(h);
        }
        {   // the edge running along x
            float a = dd.y + dd.z;
            float b = od.y + od.z;
            float cc = oo.y + oo.z - ra2;
            float h = b * b - a * cc;
            if (h > 0.0 && a > 1e-12) {
                float hh = (-b - sqrt(h)) / a;
                if (hh > 0.0 && hh < tt && abs(op.x + dp.x * hh) < siz.x) tt = hh;
            }
        }
        {   // along y
            float a = dd.z + dd.x;
            float b = od.z + od.x;
            float cc = oo.z + oo.x - ra2;
            float h = b * b - a * cc;
            if (h > 0.0 && a > 1e-12) {
                float hh = (-b - sqrt(h)) / a;
                if (hh > 0.0 && hh < tt && abs(op.y + dp.y * hh) < siz.y) tt = hh;
            }
        }
        {   // along z
            float a = dd.x + dd.y;
            float b = od.x + od.y;
            float cc = oo.x + oo.y - ra2;
            float h = b * b - a * cc;
            if (h > 0.0 && a > 1e-12) {
                float hh = (-b - sqrt(h)) / a;
                if (hh > 0.0 && hh < tt && abs(op.z + dp.z * hh) < siz.z) tt = hh;
            }
        }
        if (tt > 1e19 || tt <= 1e-4 || tt >= best) return false;
        t = tt;
        float3 q = o + tt * rd;
        n = sign(q) * normalize(max(abs(q) - siz, float3(1e-9)));
        return true;
    }

    inline bool hitShape(float3 ro, float3 rd, device const Shape &s, float best,
                         thread float &t, thread float3 &n) {
        uint kind = s.meta.x;
        if (kind == KIND_SPHERE) return hitSphere(ro, rd, s.a.xyz, s.a.w, best, t, n);
        if (kind == KIND_CYLINDER) return hitCylinder(ro, rd, s.a.xyz, s.b.xyz, s.a.w, best, t, n);
        return hitRoundBox(ro, rd, s.a.xyz, s.b.xyz, s.a.w, best, t, n);
    }

    // ---- the uniform grid

    float traceAll(float3 ro, float3 rd, device const Shape *shapes, uint count, float maxT,
                   bool anyHit, thread float3 &normal, thread float3 &color) {
        float best = maxT;
        bool got = false;
        for (uint i = 0; i < count; i++) {
            float t; float3 n;
            if (hitShape(ro, rd, shapes[i], best, t, n)) {
                best = t; normal = n; color = shapes[i].color.rgb; got = true;
                if (anyHit) return best;
            }
        }
        return got ? best : 1e30;
    }

    float traceGrid(float3 ro, float3 rd, device const Shape *shapes,
                    device const uint *cellStart, device const uint *cellItems,
                    device const Grid &grid, float maxT, bool anyHit,
                    thread float3 &normal, thread float3 &color) {
        float3 lo = grid.origin.xyz;
        float3 cell = grid.cell.xyz;
        int3 dims = int3(grid.dims.xyz);
        float3 hi = lo + float3(dims) * cell;
        float3 inv = safeInverse(rd);

        float3 ta = (lo - ro) * inv, tb = (hi - ro) * inv;
        float3 tsmall = min(ta, tb), tbig = max(ta, tb);
        float tEnter = max(max(tsmall.x, tsmall.y), max(tsmall.z, 0.0));
        float tExit = min(min(tbig.x, tbig.y), tbig.z);
        if (tEnter > tExit || tEnter > maxT) return 1e30;

        float3 p = ro + rd * (tEnter + 1e-4);
        int3 c = clamp(int3(floor((p - lo) / cell)), int3(0), dims - 1);
        int3 stp;
        float3 tMax, tDelta;
        for (int k = 0; k < 3; k++) {
            if (rd[k] > 0.0) {
                stp[k] = 1;
                tMax[k] = (lo[k] + float(c[k] + 1) * cell[k] - ro[k]) * inv[k];
                tDelta[k] = cell[k] * inv[k];
            } else if (rd[k] < 0.0) {
                stp[k] = -1;
                tMax[k] = (lo[k] + float(c[k]) * cell[k] - ro[k]) * inv[k];
                tDelta[k] = -cell[k] * inv[k];
            } else {
                stp[k] = 0;
                tMax[k] = 1e30;
                tDelta[k] = 1e30;
            }
        }

        float best = 1e30;
        for (int guard = 0; guard < 4096; guard++) {
            uint index = (uint(c.z) * grid.dims.y + uint(c.y)) * grid.dims.x + uint(c.x);
            uint from = cellStart[index], to = cellStart[index + 1];
            for (uint i = from; i < to; i++) {
                uint s = cellItems[i];
                float t; float3 n;
                if (hitShape(ro, rd, shapes[s], best, t, n)) {
                    if (t > maxT) continue;
                    best = t; normal = n; color = shapes[s].color.rgb;
                    if (anyHit) return best;
                }
            }
            float leave = min(tMax.x, min(tMax.y, tMax.z));
            if (best <= leave) break;
            if (leave > maxT) break;
            if (tMax.x < tMax.y && tMax.x < tMax.z) {
                c.x += stp.x; if (c.x < 0 || c.x >= dims.x) break; tMax.x += tDelta.x;
            } else if (tMax.y < tMax.z) {
                c.y += stp.y; if (c.y < 0 || c.y >= dims.y) break; tMax.y += tDelta.y;
            } else {
                c.z += stp.z; if (c.z < 0 || c.z >= dims.z) break; tMax.z += tDelta.z;
            }
        }
        return best;
    }

    // ---- the outside world

    inline float3 skyColor(float3 rd, device const Params &p) {
        float h = clamp(rd.y, 0.0, 1.0);
        float3 c = mix(p.skyHorizon.rgb, p.skyTop.rgb, pow(h, 0.55));
        float s = max(dot(rd, p.sun.xyz), 0.0);
        c += p.sunColor.rgb * pow(s, 400.0) * 7.0;
        c += p.sunColor.rgb * pow(s, 9.0) * 0.16;
        return c;
    }

    // The road is a plane, not geometry: a surface that runs to the horizon has
    // no bounding box worth putting in a grid. Its markings move because their
    // pattern coordinate is the world's, not the bus's.
    inline float3 roadColor(float x, float zPattern, float range, device const Params &p) {
        float u = x - p.glass.w;
        float au = abs(u);
        float halfRoad = p.road.x;
        float3 asphalt = float3(0.157, 0.161, 0.172);
        float3 gravel = float3(0.325, 0.306, 0.278);
        float3 grass = float3(0.243, 0.353, 0.176);
        float3 c;
        float3 marking = float3(-1.0);
        if (au <= halfRoad) {
            c = asphalt;
            if (au <= 0.075) {
                float d = fmod(zPattern, p.road.y);
                if (d < 0.0) d += p.road.y;
                if (d < p.road.z) marking = float3(0.816, 0.686, 0.192);
            }
            if (au > halfRoad - 0.22 && au < halfRoad - 0.10) marking = float3(0.800, 0.804, 0.780);
        } else if (au <= halfRoad + 0.65) {
            c = gravel;
        } else {
            c = grass;
            // A little texture on the verge, faded out before it is fine enough
            // to alias into moire.
            float blade = fmod(zPattern * 0.9 + au * 1.7, 1.0);
            if (blade < 0.0) blade += 1.0;
            c *= 1.0 + 0.12 * (blade - 0.5) * exp(-range / 55.0);
        }
        // Road and verge stay distinct all the way to the horizon — that is the
        // contraction the rear window is there to show. Only the markings
        // dissolve, and only once they are thinner than a pixel.
        if (marking.r >= 0.0) {
            float blend = clamp((range - 300.0) / 900.0, 0.0, 1.0);
            c = mix(marking, c, blend);
        }
        return c;
    }

    inline float3 applyHaze(float3 color, float range) {
        float f = 1.0 - exp(-range / HAZE_RANGE);
        return mix(color, HAZE_COLOR, f);
    }

    // ---- shading

    // Probe directions built from the surface normal alone — no pixel, no
    // frame, no position — so the interior's occlusion is fixed for the whole
    // render, as it has been since step 9.
    inline float3 hemisphere(uint i, uint total, float3 n) {
        float u = (float(i) + 0.5) / float(total);
        float r = sqrt(u);
        float phi = 6.2831853 * fract(float(i) * 0.6180339887);
        float3 t = normalize(abs(n.z) < 0.9 ? cross(n, float3(0.0, 0.0, 1.0))
                                            : cross(n, float3(1.0, 0.0, 0.0)));
        float3 b = cross(n, t);
        return t * (r * cos(phi)) + b * (r * sin(phi)) + n * sqrt(max(0.0, 1.0 - u));
    }

    struct Scene {
        device const Shape *busShapes;
        device const uint *busStart;
        device const uint *busItems;
        device const Grid *busGrid;
        device const Shape *worldShapes;
        device const uint *worldStart;
        device const uint *worldItems;
        device const Grid *worldGrid;
    };

    inline bool busBlocks(float3 p, float3 dir, Scene sc, device const Params &prm) {
        float3 n, c;
        float t = traceGrid(p, dir, sc.busShapes, sc.busStart, sc.busItems, *sc.busGrid,
                            1e30, true, n, c);
        return t < 1e29;
    }

    inline bool worldBlocks(float3 p, float3 dir, Scene sc, device const Params &prm) {
        if (prm.counts.w == 0u) return false;
        float3 n, c;
        float t = traceGrid(p, dir, sc.worldShapes, sc.worldStart, sc.worldItems, *sc.worldGrid,
                            1e30, true, n, c);
        return t < 1e29;
    }

    inline float busOcclusion(float3 p, float3 n, Scene sc, device const Params &prm) {
        uint probes = prm.flags.y;
        if (probes == 0u) return 1.0;
        float3 start = p + n * 0.004;
        uint open = 0u;
        for (uint i = 0; i < probes; i++) {
            float3 dir = hemisphere(i, probes, n);
            float3 nn, cc;
            float t = traceGrid(start, dir, sc.busShapes, sc.busStart, sc.busItems,
                                *sc.busGrid, prm.road.w, true, nn, cc);
            if (t > prm.road.w) open++;
        }
        return float(open) / float(probes);
    }

    // What one sample of one pixel sees, at its own moment inside the exposure.
    inline float3 sampleColor(float3 ro, float3 rd, float tau, Scene sc, device const Params &prm) {
        float3 slide = float3(0.0, 0.0, prm.up.w * tau);      // v·τ, the whole animation
        float3 rw = ro - slide;

        float3 nb = float3(0, 1, 0), cb = float3(1);
        float tb = traceGrid(ro, rd, sc.busShapes, sc.busStart, sc.busItems, *sc.busGrid,
                             1e30, false, nb, cb);

        float3 nw = float3(0, 1, 0), cw = float3(1);
        float tw = 1e30;
        if (prm.counts.w > 0u) {
            tw = traceGrid(rw, rd, sc.worldShapes, sc.worldStart, sc.worldItems, *sc.worldGrid,
                           tb, false, nw, cw);
        }
        float groundY = prm.skyHorizon.w;
        float tg = 1e30;
        if (rd.y < -1e-7) {
            float cand = (groundY - ro.y) / rd.y;
            if (cand > 1e-4) tg = cand;
        }

        float3 sun = prm.sun.xyz;
        float sunStrength = prm.sun.w;
        float ambient = prm.sunColor.w;

        if (tb <= tw && tb <= tg) {
            if (tb > 1e29) return skyColor(rd, prm) * prm.glass.rgb;   // no bus, no world, no road
            float3 p = ro + tb * rd;
            float ao = busOcclusion(p, nb, sc, prm);
            float3 start = p + nb * 0.004;
            float lit = 0.0;
            if (dot(nb, sun) > 0.0 && !busBlocks(start, sun, sc, prm)
                && !worldBlocks(start - slide, sun, sc, prm)) {
                lit = dot(nb, sun);
            }
            // Skylight arrives sideways, through the windows, so a surface
            // facing a wall collects more of it than one facing the ceiling.
            float sky = 0.66 + 0.34 * abs(nb.x) + 0.10 * max(nb.y, 0.0);
            float3 c = cb * (ambient * sky * (0.40 + 0.60 * ao) + sunStrength * lit * prm.sunColor.rgb);
            if (lit > 0.0) {
                float3 h = normalize(sun - rd);
                float spec = pow(max(dot(nb, h), 0.0), 40.0);
                c += 0.10 * spec * prm.sunColor.rgb;
            }
            return c;
        }

        // Past this point the ray has left the bus through a window, so it is
        // looking through glazing.
        if (tw <= tg) {
            float3 p = rw + tw * rd;
            float3 start = p + nw * 0.02;
            float lit = 0.0;
            if (dot(nw, sun) > 0.0 && !worldBlocks(start, sun, sc, prm)) lit = dot(nw, sun);
            float sky = 0.55 + 0.45 * clamp(nw.y, -1.0, 1.0);
            float3 c = cw * (ambient * sky + sunStrength * lit * prm.sunColor.rgb);
            return applyHaze(c, tw) * prm.glass.rgb;
        }
        if (tg < 1e29) {
            float3 p = rw + tg * rd;
            float zPattern = p.z - prm.skyTop.w;
            float3 base = roadColor(p.x, zPattern, tg, prm);
            float3 up = float3(0, 1, 0);
            float lit = 0.0;
            if (!worldBlocks(p + up * 0.03, sun, sc, prm)) lit = max(sun.y, 0.0);
            float3 c = base * (ambient + sunStrength * lit * prm.sunColor.rgb);
            return applyHaze(c, tg) * prm.glass.rgb;
        }
        return skyColor(rd, prm) * prm.glass.rgb;
    }

    inline float3 rayDirection(float sx, float sy, device const Params &prm) {
        float ndcX = sx / float(prm.counts.x) * 2.0 - 1.0;
        float ndcY = 1.0 - sy / float(prm.counts.y) * 2.0;
        float3 d = prm.forward.xyz
                 + prm.right.xyz * ndcX * prm.forward.w * prm.origin.w
                 + prm.up.xyz * ndcY * prm.origin.w;
        return normalize(d);
    }

    inline float4 shadePixel(uint2 gid, Scene sc, device const Params &prm) {
        uint k = prm.flags.x;
        float3 sum = float3(0);
        float rotation = hash01(gid);
        for (uint j = 0; j < k; j++) {
            for (uint i = 0; i < k; i++) {
                float sx = float(gid.x) + (float(i) + 0.5) / float(k);
                float sy = float(gid.y) + (float(j) + 0.5) / float(k);
                float3 rd = rayDirection(sx, sy, prm);
                uint sIdx = j * k + i;
                float u = fract((float(sIdx) + rotation) / float(k * k));
                float tau = prm.right.w * u;
                sum += sampleColor(prm.origin.xyz, rd, tau, sc, prm);
            }
        }
        return float4(sum / float(k * k), 1.0);
    }

    // Is this pixel one that can ever change? Two ways in, both tested against
    // the BUS ALONE so that the answer is a fact about the geometry:
    //   1. some sample leaves through a window, or
    //   2. some sample lands on interior that the sun can reach through one.
    inline bool pixelIsLive(uint2 gid, Scene sc, device const Params &prm) {
        uint k = prm.flags.x;
        float3 sun = prm.sun.xyz;
        for (uint j = 0; j < k; j++) {
            for (uint i = 0; i < k; i++) {
                float sx = float(gid.x) + (float(i) + 0.5) / float(k);
                float sy = float(gid.y) + (float(j) + 0.5) / float(k);
                float3 rd = rayDirection(sx, sy, prm);
                float3 n = float3(0, 1, 0), c = float3(1);
                float t = traceGrid(prm.origin.xyz, rd, sc.busShapes, sc.busStart, sc.busItems,
                                    *sc.busGrid, 1e30, false, n, c);
                if (t > 1e29) return true;                       // straight out of a window
                if (prm.sun.w <= 0.0) continue;                  // no sun, so no stripe either
                float3 p = prm.origin.xyz + t * rd + n * 0.004;
                if (dot(n, sun) > 0.0 && !busBlocks(p, sun, sc, prm)) return true;
            }
        }
        return false;
    }

    inline void writePixel(device uchar4 *pixels, uint2 gid, float4 c, device const Params &prm) {
        float3 col = clamp(c.rgb + dither(gid), 0.0, 1.0);
        pixels[gid.y * prm.counts.x + gid.x] = uchar4(uchar3(round(col * 255.0)), 255);
    }

    kernel void render(device uchar4 *pixels [[buffer(0)]],
                       device const Params &prm [[buffer(1)]],
                       device const Shape *busShapes [[buffer(2)]],
                       device const uint *busStart [[buffer(3)]],
                       device const uint *busItems [[buffer(4)]],
                       device const Grid &busGrid [[buffer(5)]],
                       device const Shape *worldShapes [[buffer(6)]],
                       device const uint *worldStart [[buffer(7)]],
                       device const uint *worldItems [[buffer(8)]],
                       device const Grid &worldGrid [[buffer(9)]],
                       uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= prm.counts.x || gid.y >= prm.counts.y) return;
        Scene sc = { busShapes, busStart, busItems, &busGrid,
                     worldShapes, worldStart, worldItems, &worldGrid };
        writePixel(pixels, gid, shadePixel(gid, sc, prm), prm);
    }

    kernel void renderList(device uchar4 *pixels [[buffer(0)]],
                           device const Params &prm [[buffer(1)]],
                           device const Shape *busShapes [[buffer(2)]],
                           device const uint *busStart [[buffer(3)]],
                           device const uint *busItems [[buffer(4)]],
                           device const Grid &busGrid [[buffer(5)]],
                           device const Shape *worldShapes [[buffer(6)]],
                           device const uint *worldStart [[buffer(7)]],
                           device const uint *worldItems [[buffer(8)]],
                           device const Grid &worldGrid [[buffer(9)]],
                           device const uint *list [[buffer(10)]],
                           uint i [[thread_position_in_grid]]) {
        if (i >= prm.flags.w) return;
        uint index = list[i];
        uint2 gid = uint2(index % prm.counts.x, index / prm.counts.x);
        Scene sc = { busShapes, busStart, busItems, &busGrid,
                     worldShapes, worldStart, worldItems, &worldGrid };
        writePixel(pixels, gid, shadePixel(gid, sc, prm), prm);
    }

    kernel void maskPass(device uchar *mask [[buffer(11)]],
                         device const Params &prm [[buffer(1)]],
                         device const Shape *busShapes [[buffer(2)]],
                         device const uint *busStart [[buffer(3)]],
                         device const uint *busItems [[buffer(4)]],
                         device const Grid &busGrid [[buffer(5)]],
                         uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= prm.counts.x || gid.y >= prm.counts.y) return;
        Scene sc = { busShapes, busStart, busItems, &busGrid,
                     busShapes, busStart, busItems, &busGrid };
        mask[gid.y * prm.counts.x + gid.x] = pixelIsLive(gid, sc, prm) ? 1 : 0;
    }
    """

enum RenderError: Error, CustomStringConvertible {
    case noMetalDevice
    case kernelCompile(String)
    case gpu(String)

    var description: String {
        switch self {
        case .noMetalDevice: return "no Metal GPU found"
        case .kernelCompile(let detail): return "could not compile the kernel: \(detail)"
        case .gpu(let detail): return "GPU error: \(detail)"
        }
    }
}

func findDevice() throws -> MTLDevice {
    if let device = MTLCreateSystemDefaultDevice() ?? MTLCopyAllDevices().first { return device }
    throw RenderError.noMetalDevice
}

// MARK: - the uniform grid

/// Which shapes lie in each box of a regular grid over the scene, flattened the
/// way the GPU wants it: `start[c]..<start[c+1]` indexes into `items`.
struct UniformGrid {
    var origin: SIMD3<Float>
    var cellSize: SIMD3<Float>
    var dims: SIMD3<Int>
    var start: [UInt32]
    var items: [UInt32]

    var cellCount: Int { dims.x * dims.y * dims.z }
    var averageOccupancy: Double {
        var used = 0
        for c in 0..<cellCount where start[c + 1] > start[c] { used += 1 }
        return used == 0 ? 0 : Double(items.count) / Double(used)
    }

    /// `density` scales the resolution: 1 aims at roughly one box per shape
    /// (Pharr, Jakob & Humphreys, *Physically Based Rendering*, §4.4).
    init(shapes: [GPUShape], density: Float = 1, maxCells: Int = 4_000_000) {
        var lo = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
        var hi = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)
        for s in shapes {
            let b = s.bounds()
            lo = simd_min(lo, b.lo)
            hi = simd_max(hi, b.hi)
        }
        if lo.x > hi.x { lo = .zero; hi = SIMD3(repeating: 1) }
        let pad = simd_max((hi - lo) * 0.001, SIMD3(repeating: 0.01))
        lo -= pad
        hi += pad
        let extent = hi - lo
        let longest = max(extent.x, max(extent.y, extent.z))
        let cubeRoot: Float = Float(pow(Double(max(shapes.count, 1)), 1.0 / 3.0))
        let perUnit: Float = density * 3 * cubeRoot / longest
        var dims = SIMD3<Int>(1, 1, 1)
        for k in 0..<3 {
            let raw: Float = extent[k] * perUnit
            dims[k] = max(1, min(1024, Int(raw.rounded())))
        }
        while dims.x * dims.y * dims.z > maxCells {
            let k = dims.x >= dims.y && dims.x >= dims.z ? 0 : (dims.y >= dims.z ? 1 : 2)
            dims[k] = max(1, dims[k] / 2)
        }
        self.origin = lo
        self.dims = dims
        self.cellSize = SIMD3(extent.x / Float(dims.x), extent.y / Float(dims.y),
                              extent.z / Float(dims.z))

        let cells = dims.x * dims.y * dims.z
        var counts = [UInt32](repeating: 0, count: cells)
        let gridOrigin = lo, cellSize = self.cellSize
        func span(_ s: GPUShape) -> (SIMD3<Int>, SIMD3<Int>) {
            let b = s.bounds()
            var a = SIMD3<Int>(0, 0, 0), z = SIMD3<Int>(0, 0, 0)
            for k in 0..<3 {
                let fa: Float = (b.lo[k] - gridOrigin[k]) / cellSize[k]
                let fz: Float = (b.hi[k] - gridOrigin[k]) / cellSize[k]
                a[k] = max(0, min(dims[k] - 1, Int(fa.rounded(.down))))
                z[k] = max(0, min(dims[k] - 1, Int(fz.rounded(.down))))
            }
            return (a, z)
        }
        for s in shapes {
            let (a, z) = span(s)
            for iz in a.z...z.z {
                for iy in a.y...z.y {
                    let row = (iz * dims.y + iy) * dims.x
                    for ix in a.x...z.x { counts[row + ix] += 1 }
                }
            }
        }
        var start = [UInt32](repeating: 0, count: cells + 1)
        var running: UInt32 = 0
        for c in 0..<cells {
            start[c] = running
            running += counts[c]
        }
        start[cells] = running
        var items = [UInt32](repeating: 0, count: Int(running))
        var cursor = start
        for (i, s) in shapes.enumerated() {
            let (a, z) = span(s)
            for iz in a.z...z.z {
                for iy in a.y...z.y {
                    let row = (iz * dims.y + iy) * dims.x
                    for ix in a.x...z.x {
                        items[Int(cursor[row + ix])] = UInt32(i)
                        cursor[row + ix] += 1
                    }
                }
            }
        }
        self.start = start
        self.items = items
    }

    /// The shapes recorded in the box containing `p` (for tests).
    func shapes(at p: SIMD3<Float>) -> [Int] {
        var c = SIMD3<Int>(0, 0, 0)
        for k in 0..<3 {
            let f: Float = (p[k] - origin[k]) / cellSize[k]
            let v = Int(f.rounded(.down))
            if v < 0 || v >= dims[k] { return [] }
            c[k] = v
        }
        let index = (c.z * dims.y + c.y) * dims.x + c.x
        return (Int(start[index])..<Int(start[index + 1])).map { Int(items[$0]) }
    }
}

// MARK: - settings

struct RenderSettings {
    var samplesPerSide: Int = 2
    var aoProbes: Int = 6
    var aoDistance: Float = 0.85
    var exposure: Float = 0.04
    var speed: Float = 15.3
    var travelWrapped: Float = 0
    var groundY: Float = -0.9144
    var roadCentre: Float = -1.83
    var roadHalfWidth: Float = 3.65
    var dashPeriod: Float = 12.24
    var dashLength: Float = 3.06
    var sun: SIMD3<Float> = SIMD3(0, 1, 0)
    var sunStrength: Float = 1.60
    var sunColor: SIMD3<Float> = SIMD3(1.0, 0.957, 0.882)
    var ambient: Float = 1.15
    var skyTop: SIMD3<Float> = SIMD3(0.325, 0.518, 0.776)
    var skyHorizon: SIMD3<Float> = SIMD3(0.729, 0.816, 0.882)
    var glass: SIMD3<Float> = SIMD3(0.800, 0.851, 0.812)
}

// MARK: - the renderer

final class SceneRenderer {
    let device: MTLDevice
    private let fullPipeline: MTLComputePipelineState
    private let listPipeline: MTLComputePipelineState
    private let maskPipeline: MTLComputePipelineState
    private let queue: MTLCommandQueue

    private var busShapeBuffer: MTLBuffer?
    private var busStartBuffer: MTLBuffer?
    private var busItemBuffer: MTLBuffer?
    private var busGridBuffer: MTLBuffer?
    private var worldShapeBuffer: MTLBuffer?
    private var worldStartBuffer: MTLBuffer?
    private var worldItemBuffer: MTLBuffer?
    private var worldGridBuffer: MTLBuffer?
    private var paramsBuffer: MTLBuffer?
    private var listBuffer: MTLBuffer?
    private var maskBuffer: MTLBuffer?

    private(set) var busGrid: UniformGrid?
    private(set) var worldGrid: UniformGrid?
    private(set) var busCount: Int = 0
    private(set) var worldCount: Int = 0
    private(set) var lastGridBuildSeconds: Double = 0

    init(device: MTLDevice) throws {
        self.device = device
        do {
            let options = MTLCompileOptions()
            // Off since step 3: fast math would let the compiler reassociate the
            // intersection arithmetic and the two Macs would stop agreeing.
            options.fastMathEnabled = false
            let library = try device.makeLibrary(source: raytraceKernelSource, options: options)
            guard let full = library.makeFunction(name: "render"),
                  let list = library.makeFunction(name: "renderList"),
                  let mask = library.makeFunction(name: "maskPass") else {
                throw RenderError.kernelCompile("a kernel is missing")
            }
            fullPipeline = try device.makeComputePipelineState(function: full)
            listPipeline = try device.makeComputePipelineState(function: list)
            maskPipeline = try device.makeComputePipelineState(function: mask)
        } catch let error as RenderError {
            throw error
        } catch {
            throw RenderError.kernelCompile(error.localizedDescription)
        }
        guard let queue = device.makeCommandQueue() else { throw RenderError.gpu("no command queue") }
        self.queue = queue
    }

    /// The bus. Sorted into its own grid once, because it never moves again.
    func setBus(_ shapes: [GPUShape], density: Float = 1) throws {
        let g = UniformGrid(shapes: shapes, density: density)
        busGrid = g
        busCount = shapes.count
        busShapeBuffer = try fill(busShapeBuffer, with: shapes)
        busStartBuffer = try fill(busStartBuffer, with: g.start)
        busItemBuffer = try fill(busItemBuffer, with: g.items.isEmpty ? [UInt32(0)] : g.items)
        busGridBuffer = try fill(busGridBuffer, with: [gpuGrid(g)])
    }

    /// The world, transformed into the bus's frame for this frame and sorted
    /// again. This is the one rebuild the animation costs.
    func setWorld(_ shapes: [GPUShape], density: Float = 3) throws {
        let start = Date()
        let g = UniformGrid(shapes: shapes.isEmpty ? [GPUShape.sphere(center: .zero, radius: 0.001,
                                                                     color: .zero)] : shapes,
                            density: density)
        lastGridBuildSeconds = Date().timeIntervalSince(start)
        worldGrid = g
        worldCount = shapes.count
        let list = shapes.isEmpty ? [GPUShape.sphere(center: .zero, radius: 0, color: .zero)] : shapes
        worldShapeBuffer = try fill(worldShapeBuffer, with: list)
        worldStartBuffer = try fill(worldStartBuffer, with: g.start)
        worldItemBuffer = try fill(worldItemBuffer, with: g.items.isEmpty ? [UInt32(0)] : g.items)
        worldGridBuffer = try fill(worldGridBuffer, with: [gpuGrid(g)])
    }

    private func gpuGrid(_ g: UniformGrid) -> GPUGrid {
        GPUGrid(origin: SIMD4(g.origin, 0), cell: SIMD4(g.cellSize, 0),
                dims: SIMD4(UInt32(g.dims.x), UInt32(g.dims.y), UInt32(g.dims.z), 0))
    }

    private func params(camera: Camera, settings: RenderSettings, width: Int, viewHeight: Int,
                        mode: RenderMode, listCount: Int) -> GPUParams {
        let aspect: Float = Float(width) / Float(viewHeight)
        return GPUParams(
            origin: SIMD4(camera.origin, camera.tanHalfFOV),
            forward: SIMD4(camera.forward, aspect),
            right: SIMD4(camera.right, settings.exposure),
            up: SIMD4(camera.up, settings.speed),
            sun: SIMD4(settings.sun, settings.sunStrength),
            sunColor: SIMD4(settings.sunColor, settings.ambient),
            skyTop: SIMD4(settings.skyTop, settings.travelWrapped),
            skyHorizon: SIMD4(settings.skyHorizon, settings.groundY),
            glass: SIMD4(settings.glass, settings.roadCentre),
            road: SIMD4(settings.roadHalfWidth, settings.dashPeriod, settings.dashLength,
                        settings.aoDistance),
            counts: SIMD4(UInt32(width), UInt32(viewHeight), UInt32(busCount), UInt32(worldCount)),
            flags: SIMD4(UInt32(settings.samplesPerSide), UInt32(settings.aoProbes),
                         mode.rawValue, UInt32(listCount)))
    }

    private func bind(_ e: MTLComputeCommandEncoder, frame: MTLBuffer?) {
        if let frame { e.setBuffer(frame, offset: 0, index: 0) }
        e.setBuffer(paramsBuffer, offset: 0, index: 1)
        e.setBuffer(busShapeBuffer, offset: 0, index: 2)
        e.setBuffer(busStartBuffer, offset: 0, index: 3)
        e.setBuffer(busItemBuffer, offset: 0, index: 4)
        e.setBuffer(busGridBuffer, offset: 0, index: 5)
        e.setBuffer(worldShapeBuffer ?? busShapeBuffer, offset: 0, index: 6)
        e.setBuffer(worldStartBuffer ?? busStartBuffer, offset: 0, index: 7)
        e.setBuffer(worldItemBuffer ?? busItemBuffer, offset: 0, index: 8)
        e.setBuffer(worldGridBuffer ?? busGridBuffer, offset: 0, index: 9)
    }

    /// Trace every pixel. Returns GPU seconds.
    @discardableResult
    func renderFull(camera: Camera, settings: RenderSettings, into frame: MTLBuffer,
                    width: Int, viewHeight: Int) throws -> Double {
        let p = params(camera: camera, settings: settings, width: width, viewHeight: viewHeight,
                       mode: .full, listCount: 0)
        paramsBuffer = try fill(paramsBuffer, with: [p])
        guard let commands = queue.makeCommandBuffer(), let e = commands.makeComputeCommandEncoder()
        else { throw RenderError.gpu("could not create a command encoder") }
        e.setComputePipelineState(fullPipeline)
        bind(e, frame: frame)
        let w = fullPipeline.threadExecutionWidth
        let group = MTLSize(width: w, height: fullPipeline.maxTotalThreadsPerThreadgroup / w, depth: 1)
        e.dispatchThreads(MTLSize(width: width, height: viewHeight, depth: 1),
                          threadsPerThreadgroup: group)
        e.endEncoding()
        commands.commit()
        commands.waitUntilCompleted()
        if let error = commands.error { throw RenderError.gpu(error.localizedDescription) }
        return commands.gpuEndTime - commands.gpuStartTime
    }

    /// Trace only the pixels named in `indices`, leaving the rest of the buffer
    /// exactly as it was.
    @discardableResult
    func renderMasked(camera: Camera, settings: RenderSettings, into frame: MTLBuffer,
                      width: Int, viewHeight: Int, indices: [UInt32]) throws -> Double {
        if indices.isEmpty { return 0 }
        let p = params(camera: camera, settings: settings, width: width, viewHeight: viewHeight,
                       mode: .list, listCount: indices.count)
        paramsBuffer = try fill(paramsBuffer, with: [p])
        listBuffer = try fill(listBuffer, with: indices)
        guard let commands = queue.makeCommandBuffer(), let e = commands.makeComputeCommandEncoder()
        else { throw RenderError.gpu("could not create a command encoder") }
        e.setComputePipelineState(listPipeline)
        bind(e, frame: frame)
        e.setBuffer(listBuffer, offset: 0, index: 10)
        let w = listPipeline.threadExecutionWidth
        e.dispatchThreads(MTLSize(width: indices.count, height: 1, depth: 1),
                          threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
        e.endEncoding()
        commands.commit()
        commands.waitUntilCompleted()
        if let error = commands.error { throw RenderError.gpu(error.localizedDescription) }
        return commands.gpuEndTime - commands.gpuStartTime
    }

    /// The aperture-and-sunlight mask, computed from the bus alone and so valid
    /// for every frame of the render.
    func computeMask(camera: Camera, settings: RenderSettings,
                     width: Int, viewHeight: Int) throws -> [UInt8] {
        let p = params(camera: camera, settings: settings, width: width, viewHeight: viewHeight,
                       mode: .mask, listCount: 0)
        paramsBuffer = try fill(paramsBuffer, with: [p])
        let pixels = width * viewHeight
        if maskBuffer == nil || maskBuffer!.length < pixels {
            maskBuffer = device.makeBuffer(length: max(pixels, 16), options: .storageModeShared)
        }
        guard let mask = maskBuffer else { throw RenderError.gpu("could not allocate the mask") }
        guard let commands = queue.makeCommandBuffer(), let e = commands.makeComputeCommandEncoder()
        else { throw RenderError.gpu("could not create a command encoder") }
        e.setComputePipelineState(maskPipeline)
        bind(e, frame: nil)
        e.setBuffer(mask, offset: 0, index: 11)
        let w = maskPipeline.threadExecutionWidth
        let group = MTLSize(width: w, height: maskPipeline.maxTotalThreadsPerThreadgroup / w, depth: 1)
        e.dispatchThreads(MTLSize(width: width, height: viewHeight, depth: 1),
                          threadsPerThreadgroup: group)
        e.endEncoding()
        commands.commit()
        commands.waitUntilCompleted()
        if let error = commands.error { throw RenderError.gpu(error.localizedDescription) }
        return Array(UnsafeBufferPointer(start: mask.contents().assumingMemoryBound(to: UInt8.self),
                                         count: pixels))
    }

    private func fill<T>(_ buffer: MTLBuffer?, with items: [T]) throws -> MTLBuffer {
        let bytes = max(items.count * MemoryLayout<T>.stride, 16)
        var target = buffer
        if target == nil || target!.length < bytes {
            target = device.makeBuffer(length: bytes, options: .storageModeShared)
        }
        guard let out = target else { throw RenderError.gpu("could not allocate a scene buffer") }
        items.withUnsafeBytes { out.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
        return out
    }
}

/// The masked pixels, as a flat list the GPU can be dispatched over.
func maskedIndices(_ mask: [UInt8]) -> [UInt32] {
    var out: [UInt32] = []
    out.reserveCapacity(mask.count / 3)
    for (i, m) in mask.enumerated() where m != 0 { out.append(UInt32(i)) }
    return out
}
