// The toy, rendered: one GPU thread per pixel marching rays into the toy's
// distance function, as every step since the ocean has.
//
// The toy reaches the GPU as data — its rigid segments (each a world-to-
// local transform and a bounding sphere), their shapes, and their paint
// patches — so a still and every frame of an animation use one kernel, and
// the tests probe exactly what was drawn.
//
// The new work is the material. A painted plastic surface does two things
// with light: its paint scatters some back diffusely in the paint's colour,
// and its top surface mirrors a little of every colour — the Fresnel
// reflectance of a dielectric of index n, ((n − 1)/(n + 1))² at normal
// incidence and rising to 1 at grazing. For PVC's n = 1.545 that is 4.6%. How
// sharp that mirror is depends on the surface's roughness: the kernel draws a
// microfacet normal from the GGX distribution for each sample (Heitz's
// visible-normal sampling), reflects the view about it, weights the light
// found there by Fresnel × Smith masking, and averages. Rough paint gives a
// broad sheen of the softbox; a glossy eye, a small bright glint. The
// Fresnel values come from a table the CPU computed from PVC's measured n
// (Plastic.swift). The diffuse part is weighted by 1 − F, the light the
// surface did not mirror.

import CoreGraphics
import Foundation
import ImageIO
import Metal
import UniformTypeIdentifiers
import simd

/// Ray steps trust this fraction of each distance. The field is exact or a
/// lower bound outside (tested), so 1 would do in principle; 0.9 absorbs
/// float rounding.
let stepScale: Float = 0.9

func metal(_ v: SIMD3<Float>) -> String { "float3(\(v.x), \(v.y), \(v.z))" }

// MARK: - the toy as GPU data

struct GSeg {
    var r0: SIMD4<Float>
    var r1: SIMD4<Float>
    var r2: SIMD4<Float>
    var bound: SIMD4<Float>
    var misc: SIMD4<Float>      // blend k, seam on, fillet to the body, 0
    var range: SIMD4<Int32>     // first shape, shape count, first patch, patch count
}

struct GPrim {
    var a: SIMD4<Float>
    var b: SIMD4<Float>
    var u: SIMD4<Float>
    var n: SIMD4<Float>
    var size: SIMD4<Float>
    var e: SIMD4<Float>
    var f: SIMD4<Float>
    var g: SIMD4<Float>
    var info: SIMD4<Int32>      // kind, paint, own seam, second (inside) paint
}

struct GPatch {
    var c: SIMD4<Float>
    var info: SIMD4<Int32>      // paint, kind, 0, 0
}

/// Flatten a posed toy into the three arrays the kernel reads.
func gpuData(_ toy: PosedToy) -> (segs: [GSeg], prims: [GPrim], patches: [GPatch]) {
    var segs: [GSeg] = []
    var prims: [GPrim] = []
    var patches: [GPatch] = []
    for (i, s) in toy.segments.enumerated() {
        let f: Frame = toy.frames[i]
        let b = s.bound
        let bc: SIMD3<Float> = f.toWorld(b.centre)
        let r0 = SIMD4<Float>(f.x, -simd_dot(f.x, f.o))
        let r1 = SIMD4<Float>(f.y, -simd_dot(f.y, f.o))
        let r2 = SIMD4<Float>(f.z, -simd_dot(f.z, f.o))
        let seam: Float = (toy.seams && s.seam) ? 1 : 0
        let range = SIMD4<Int32>(Int32(prims.count), Int32(s.prims.count), Int32(patches.count), Int32(s.patches.count))
        segs.append(GSeg(r0: r0, r1: r1, r2: r2, bound: SIMD4<Float>(bc, b.radius), misc: SIMD4<Float>(s.blend, seam, s.bodyFillet, 0),
                         range: range))
        for p in s.prims {
            let own: Int32 = (toy.seams && p.ownSeam) ? 1 : 0
            prims.append(GPrim(a: SIMD4<Float>(p.a, p.ra), b: SIMD4<Float>(p.b, p.rb), u: SIMD4<Float>(p.u, 0),
                               n: SIMD4<Float>(p.n, 0), size: p.size, e: p.e, f: p.f, g: p.g,
                               info: SIMD4<Int32>(p.kind.rawValue, Int32(p.paint), own, Int32(p.paint2))))
        }
        for q in s.patches {
            patches.append(GPatch(c: SIMD4<Float>(q.c, q.r), info: SIMD4<Int32>(Int32(q.paint), q.kind, 0, 0)))
        }
    }
    if patches.isEmpty { patches.append(GPatch(c: .zero, info: .zero)) }
    return (segs, prims, patches)
}

// MARK: - the kernel

func kernelSource(paints: [Paint], studio: Studio, mutant: Mutant) -> String {
    let lut: String = fresnelTable(mutant).map { metal($0) }.joined(separator: ",\n        ")
    let albedo: String = paints.map { metal($0.albedo) }.joined(separator: ", ")
    let rough: String = paints.map { "\($0.roughness)" }.joined(separator: ", ")
    let k: Softbox = studio.key
    let f: Softbox = studio.fill
    let r: Softbox = studio.rim
    func box(_ n: String, _ b: Softbox) -> String {
        """
        constant float3 \(n)_C = \(metal(b.centre));
        constant float3 \(n)_A = \(metal(b.axisA));
        constant float3 \(n)_B = \(metal(b.axisB));
        constant float \(n)_TA = \(b.tanA);
        constant float \(n)_TB = \(b.tanB);
        constant float \(n)_L = \(b.radiance);
        constant float \(n)_OMEGA = \(b.solidAngle);
        """
    }
    return """
    #include <metal_stdlib>
    using namespace metal;

    struct Params { uint width; uint height; uint rowOffset; uint samples;
                    uint segCount; uint pad0; uint pad1; uint pad2;
                    float4 camPos; float4 camFwd; float4 camRight; float4 camUp; };
    struct GSeg { float4 r0; float4 r1; float4 r2; float4 bound; float4 misc; int4 range; };
    struct GPrim { float4 a; float4 b; float4 u; float4 n; float4 size; float4 e; float4 f; float4 g; int4 info; };
    struct GPatch { float4 c; int4 info; };

    constant float STEP_SCALE = \(stepScale);
    constant float FAR = 2000.0;
    constant float SEAM_H = \(seamHeight);
    constant float SEAM_W = \(seamHalfWidth);
    constant float TONE_GAIN = \(toneGain);

    constant int PAINT_N = \(paints.count);
    constant float3 ALBEDO[\(paints.count)] = { \(albedo) };
    constant float ROUGH[\(paints.count)] = { \(rough) };

    \(box("KEY", k))
    \(box("FILL", f))
    \(box("RIM", r))
    constant float TENT_H = \(studio.tentHorizon);
    constant float TENT_Z = \(studio.tentZenith);
    constant float AMBIENT_E = \(studio.ambientIrradiance);
    constant float3 TABLE_ALBEDO = \(metal(studio.tableAlbedo));
    constant float TABLE_FALL = \(studio.tableFalloff);

    constant int LUT_N = \(fresnelTableSize);
    constant float3 FRESNEL[\(fresnelTableSize)] = {
        \(lut)
    };

    // ---------------------------------------------------------------- shapes

    float sdRoundCone(float3 p, float3 a, float3 b, float r1, float r2) {
        float3 ba = b - a;
        float l2 = dot(ba, ba);
        if (l2 < 1e-12) return length(p - a) - max(r1, r2);
        float rr = r1 - r2;
        float a2 = l2 - rr * rr;
        float il2 = 1.0 / l2;
        float3 pa = p - a;
        float y = dot(pa, ba);
        float z = y - l2;
        float3 xv = pa * l2 - ba * y;
        float x2 = dot(xv, xv);
        float y2 = y * y * l2;
        float z2 = z * z * l2;
        float k = sign(rr) * rr * rr * x2;
        if ((z >= 0.0 ? 1.0 : -1.0) * a2 * z2 > k) return sqrt(x2 + z2) * il2 - r2;
        if ((y >= 0.0 ? 1.0 : -1.0) * a2 * y2 < k) return sqrt(x2 + y2) * il2 - r1;
        return (sqrt(x2 * a2 * il2) + y * rr) * il2 - r1;
    }

    float sdVesica2(float2 p, float h, float w) {
        float r = (w + h * h / w) * 0.5;
        float d = r - w;
        p = abs(p);
        float b = sqrt(r * r - d * d);
        if ((p.y - b) * d > p.x * b) return length(p - float2(0.0, b));
        return length(p - float2(-d, 0.0)) - r;
    }

    float sdTriangle2(float2 p, float2 p0, float2 p1, float2 p2) {
        float2 e0 = p1 - p0, e1 = p2 - p1, e2 = p0 - p2;
        float2 v0 = p - p0, v1 = p - p1, v2 = p - p2;
        float2 pq0 = v0 - e0 * clamp(dot(v0, e0) / dot(e0, e0), 0.0, 1.0);
        float2 pq1 = v1 - e1 * clamp(dot(v1, e1) / dot(e1, e1), 0.0, 1.0);
        float2 pq2 = v2 - e2 * clamp(dot(v2, e2) / dot(e2, e2), 0.0, 1.0);
        float s = (e0.x * e2.y - e0.y * e2.x) >= 0.0 ? 1.0 : -1.0;
        float c0 = s * (v0.x * e0.y - v0.y * e0.x);
        float c1 = s * (v1.x * e1.y - v1.y * e1.x);
        float c2 = s * (v2.x * e2.y - v2.y * e2.x);
        float dd = min(dot(pq0, pq0), min(dot(pq1, pq1), dot(pq2, pq2)));
        float sg = min(c0, min(c1, c2));
        return -sqrt(dd) * (sg >= 0.0 ? 1.0 : -1.0);
    }

    float smin(float a, float b, float k) {
        if (k <= 0.0) return min(a, b);
        float h = max(k - abs(a - b), 0.0) / k;
        return min(a, b) - h * h * k * 0.25;
    }

    float smax(float a, float b, float k) { return -smin(-a, -b, k); }

    // A flat piece in a plane: triangle, cut and kept by circles, tapered
    // thickness, rounded edges (Toy.swift's profileSDF).
    float profileSDF(constant GPrim &s, float3 q, bool seams) {
        float3 v = cross(s.n.xyz, s.u.xyz);
        float3 d = q - s.a.xyz;
        float2 p2 = float2(dot(d, s.u.xyz), dot(d, v));
        float pz = dot(d, s.n.xyz);
        float edge = s.b.z;
        float d2 = sdTriangle2(p2, s.size.xy, s.size.zw, s.b.xy);
        if (s.e.z > 0.0) d2 = max(d2, s.e.z - length(p2 - s.e.xy));
        if (s.f.z > 0.0) d2 = max(d2, length(p2 - s.f.xy) - s.f.z);
        float taper = s.f.w;
        float sT = sqrt(1.0 + taper * taper);
        float th = s.a.w - taper * p2.y;
        float dz = (abs(pz) - th) / sT;
        float2 w = float2(d2 + edge, dz + edge);
        float raw = min(max(w.x, w.y), 0.0) + length(max(w, 0.0)) - edge;
        float dist = raw / sqrt(1.0 + abs(taper) / sT);
        if (seams && s.info.z != 0) dist = min(dist, max(dist - SEAM_H, abs(pz) - SEAM_W));
        return dist;
    }

    // The hull (Toy.swift's hullSDF): tapered plan, keel circle with
    // deadrise, sheer circle or deck cylinder, optional hollow with thwarts,
    // the parting line round the gunwale.
    float hullSDF(constant GPrim &s, float3 q, bool seams, thread float &hollow, thread bool &onTop) {
        float lh = s.a.x, bm = s.a.y, xm = s.a.z, xt = s.a.w;
        float tanF = s.b.x, y0 = s.b.y, kB = s.b.z, tanD = s.b.w;
        float tanV = s.u.y;
        float transomFace = (xt - q.x + abs(q.z) * tanV) / sqrt(1.0 + tanV * tanV);
        float plan2 = max(sdVesica2(float2(q.z, q.x - xm), lh, bm), transomFace);
        float sF = sqrt(1.0 + tanF * tanF);
        float dp = (plan2 + (y0 - q.y) * tanF) / sF;
        float dk = length(q.xy - s.u.xz) - s.u.z;
        float sD = sqrt(1.0 + tanD * tanD);
        float kb = (dk + abs(q.z) * tanD) / sD;
        float lower = smax(dp, kb, kB);
        float rTop = s.g.w;
        float top;
        if (s.n.x < 0.5) {
            top = rTop - length(q.xy - s.n.yz);
        } else {
            float3 axis = normalize(float3(1.0, s.n.y, 0.0));
            float3 w = q - float3(0.0, s.n.z, 0.0);
            top = length(w - axis * dot(w, axis)) - rTop;
        }
        float outer = smax(lower, top, s.size.w);
        outer = max(outer, q.y - s.g.z);
        onTop = top >= lower;
        hollow = 1e9;
        float wall = s.size.x;
        if (wall > 0.0) {
            float vd = smax(dp + wall, kb + s.size.y, s.g.x);
            float tanT = s.size.z;
            float sT = sqrt(1.0 + tanT * tanT);
            int count = int(s.g.y + 0.5);
            float thw = 1e9;
            float xs[3] = { s.e.x, s.e.y, s.e.z };
            float ys[3] = { s.f.x, s.f.y, s.f.z };
            for (int i = 0; i < min(count, 3); i++) {
                float side = (abs(q.x - xs[i]) - s.e.w - (ys[i] - q.y) * tanT) / sT;
                thw = min(thw, max(side, q.y - ys[i]));
            }
            vd = max(vd, -thw);
            hollow = vd;
            outer = max(outer, -vd);
        }
        if (seams && s.info.z != 0 && s.n.x < 0.5) {
            float ring = abs(length(q.xy - s.n.yz) - (rTop + s.f.w)) - SEAM_W;
            float ridge = max(outer - SEAM_H, ring);
            if (wall > 0.0) ridge = max(ridge, wall * 0.5 - hollow);
            outer = min(outer, ridge);
        }
        return outer;
    }

    float primSDF(constant GPrim &s, float3 p, bool seams) {
        int kind = s.info.x;
        if (kind == 0) return sdRoundCone(p, s.a.xyz, s.b.xyz, s.a.w, s.b.w);
        if (kind == 1) {
            float3 v = cross(s.n.xyz, s.u.xyz);
            float3 d = p - s.a.xyz;
            float3 q = abs(float3(dot(d, s.u.xyz), dot(d, v), dot(d, s.n.xyz))) - s.size.xyz + s.size.w;
            return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0) - s.size.w;
        }
        if (kind == 3) return profileSDF(s, p, seams);
        if (kind == 4) { float h; bool t; return hullSDF(s, p, seams, h, t); }
        float3 across = cross(s.u.xyz, s.n.xyz);
        float3 d = p - s.a.xyz;
        float e = s.size.w;
        float hh = s.size.x - e;
        float ww = s.size.y - e;
        float2 p2 = float2(dot(d, across), dot(d, s.u.xyz));
        float d2 = hh >= ww ? sdVesica2(p2, hh, ww) : sdVesica2(p2.yx, ww, hh);
        float2 w = float2(d2, abs(dot(d, s.n.xyz)) - (s.size.z - e));
        float dist = min(max(w.x, w.y), 0.0) + length(max(w, 0.0)) - e;
        if (seams && s.info.z != 0) {
            float off = abs(dot(p - s.a.xyz, s.n.xyz));
            dist = min(dist, max(dist - SEAM_H, off - SEAM_W));
        }
        return dist;
    }

    float3 toLocal(constant GSeg &g, float3 p) {
        return float3(dot(g.r0.xyz, p) + g.r0.w, dot(g.r1.xyz, p) + g.r1.w, dot(g.r2.xyz, p) + g.r2.w);
    }

    // One rigid segment: its shapes blended with its constant k, then the
    // parting line in its mid-plane.
    float segSDF(constant GSeg &g, constant GPrim *prims, float3 p, thread int &nearest) {
        float3 q = toLocal(g, p);
        float d = 1e9;
        float best = 1e9;
        nearest = 0;
        int first = g.range.x;
        int count = g.range.y;
        float k = g.misc.x;
        for (int i = 0; i < count; i++) {
            constant GPrim &s = prims[first + i];
            float di = primSDF(s, q, true);
            if (di < best) { best = di; nearest = i; }
            d = (i == 0) ? di : smin(d, di, k);
        }
        if (g.misc.y > 0.5) d = min(d, max(d - SEAM_H, abs(q.z) - SEAM_W));
        return d;
    }

    // The toy: the nearest segment, skipping any whose bounding sphere is
    // farther than the best so far (which can never change the minimum).
    // A piece with a fillet to the body is smooth-min'd with the body's own
    // distance, so the body is always evaluated; a filleted piece's bound
    // includes its fillet width, so skipping it is still safe.
    float toySDF(float3 p, constant GSeg *segs, constant GPrim *prims, uint n, float cap,
                 thread int &seg, thread int &prim) {
        float best = cap;
        seg = -1;
        prim = -1;
        int pr0;
        float d0 = segSDF(segs[0], prims, p, pr0);
        if (d0 < best) { best = d0; seg = 0; prim = pr0; }
        for (uint i = 1; i < n; i++) {
            float dc = length(p - segs[i].bound.xyz) - segs[i].bound.w;
            if (dc > best) continue;
            int pr;
            float d = segSDF(segs[i], prims, p, pr);
            float k = segs[i].misc.z;
            float dd = k > 0.0 ? smin(d0, d, k) : d;
            if (dd < best) {
                best = dd;
                if (k > 0.0 && d > d0) { seg = 0; prim = pr0; } else { seg = int(i); prim = pr; }
            }
        }
        return best;
    }

    // Material: 1 the table, 2 the toy.
    float sceneSDF(float3 p, constant GSeg *segs, constant GPrim *prims, uint n,
                   thread int &mat, thread int &seg, thread int &prim) {
        float dt = p.y;
        float d = toySDF(p, segs, prims, n, dt, seg, prim);
        if (seg >= 0) { mat = 2; return d; }
        mat = 1;
        return dt;
    }

    bool march(float3 ro, float3 rd, float tMax, constant GSeg *segs, constant GPrim *prims, uint n,
               thread float &t, thread int &mat, thread int &seg, thread int &prim) {
        t = 0.0;
        for (int i = 0; i < 300; i++) {
            float3 p = ro + rd * t;
            float d = sceneSDF(p, segs, prims, n, mat, seg, prim);
            float eps = 0.0015 + t * 0.00001;
            if (d < eps) return true;
            t += d * STEP_SCALE;
            if (t > tMax) break;
        }
        return false;
    }

    float toyOnly(float3 p, constant GSeg *segs, constant GPrim *prims, uint n) {
        int s, q;
        return toySDF(p, segs, prims, n, FAR, s, q);
    }

    float3 toyNormal(float3 p, constant GSeg *segs, constant GPrim *prims, uint n) {
        const float h = 0.0012;
        float3 k0 = float3(1, -1, -1), k1 = float3(-1, -1, 1), k2 = float3(-1, 1, -1), k3 = float3(1, 1, 1);
        return normalize(k0 * toyOnly(p + k0 * h, segs, prims, n) + k1 * toyOnly(p + k1 * h, segs, prims, n)
                       + k2 * toyOnly(p + k2 * h, segs, prims, n) + k3 * toyOnly(p + k3 * h, segs, prims, n));
    }

    // Which paint is at a surface point: the nearest shape's, unless a patch
    // on its segment covers the point.
    int paintAt(float3 p, int seg, int prim, constant GSeg *segs, constant GPrim *prims, constant GPatch *patches) {
        constant GSeg &g = segs[seg];
        constant GPrim &pr = prims[g.range.x + prim];
        int pi = pr.info.y;
        float3 q = toLocal(g, p);
        if (pr.info.x == 4 && pr.info.w >= 0) {
            float h;
            bool onTop;
            hullSDF(pr, q, false, h, onTop);
            // An open hull's second paint is its inside; a decked hull's, its deck.
            if (pr.size.x > 0.0 ? h < pr.size.x * 0.5 : onTop) pi = pr.info.w;
        }
        for (int i = 0; i < g.range.w; i++) {
            constant GPatch &c = patches[g.range.z + i];
            int kind = c.info.y;
            bool inside = false;
            if (kind == 0) inside = length(q - c.c.xyz) < c.c.w;
            else if (kind == 1) inside = q.y < c.c.y + c.c.w * sin(atan2(q.z, q.x) * 5.0 + 0.7);
            else if (kind == 2) {
                float wob = sin(0.9 * q.x + 1.3) * sin(0.8 * q.y + 0.4) * sin(1.0 * q.z + 2.1);
                inside = length(q - c.c.xyz) < c.c.w * (1.0 + 0.28 * wob);
            }
            if (inside) pi = c.info.x;
        }
        return pi;
    }

    // ---------------------------------------------------------------- light

    float boxWeight(float3 r, float3 C, float3 A, float3 B, float ta, float tb) {
        float c = dot(r, C);
        if (c <= 0.0) return 0.0;
        float a = dot(r, A) / (ta * c);
        float b = dot(r, B) / (tb * c);
        if (abs(a) >= 1.0 || abs(b) >= 1.0) return 0.0;
        // Brightest in the middle of the diffuser, 40% at the edge. MODEL.
        return mix(0.4, 1.0, (1.0 - a * a) * (1.0 - b * b));
    }

    // Radiance arriving from a direction. tag: 1 key, 2 fill, 3 rim, 0 tent.
    float3 environment(float3 r, thread int &tag) {
        float w = boxWeight(r, KEY_C, KEY_A, KEY_B, KEY_TA, KEY_TB);
        if (w > 0.0) { tag = 1; return float3(KEY_L * w); }
        w = boxWeight(r, FILL_C, FILL_A, FILL_B, FILL_TA, FILL_TB);
        if (w > 0.0) { tag = 2; return float3(FILL_L * w); }
        w = boxWeight(r, RIM_C, RIM_A, RIM_B, RIM_TA, RIM_TB);
        if (w > 0.0) { tag = 3; return float3(RIM_L * w); }
        tag = 0;
        float L = r.y < 0.0 ? TENT_H * 0.8 : mix(TENT_H, TENT_Z, sqrt(r.y));
        return float3(L);
    }

    float softShadow(float3 ro, float3 rd, float k, constant GSeg *segs, constant GPrim *prims, uint n) {
        float res = 1.0;
        float t = 0.02;
        for (int i = 0; i < 70; i++) {
            float h = toyOnly(ro + rd * t, segs, prims, n);
            res = min(res, k * h / t);
            if (res < 0.002) return 0.0;
            t += clamp(h, 0.03, 4.0);
            if (t > 120.0) break;
        }
        return clamp(res, 0.0, 1.0);
    }

    // How much of the sky above a point the toy blocks, from the distance
    // field: a few taps along the normal. MODEL (the usual SDF occlusion).
    float occlusion(float3 p, float3 nrm, constant GSeg *segs, constant GPrim *prims, uint n, float scale) {
        float occ = 0.0;
        float w = 1.0;
        for (int i = 1; i <= 5; i++) {
            float h = scale * float(i);
            float3 q = p + nrm * h;
            float d = min(toyOnly(q, segs, prims, n), q.y + 1e3 * step(0.5, nrm.y));
            occ += (h - clamp(d, 0.0, h)) * w;
            w *= 0.6;
        }
        return clamp(1.0 - occ * 0.4 / scale, 0.0, 1.0);
    }

    // Light arriving on a surface with normal nrm: three softboxes (from
    // their centres, with soft shadows as wide as each box) and the tent.
    float3 irradiance(float3 p, float3 nrm, constant GSeg *segs, constant GPrim *prims, uint n, float ao) {
        float3 o = p + nrm * 0.01;
        float e = 0.0;
        float c = dot(nrm, KEY_C);
        if (c > 0.0) e += KEY_L * 0.7 * KEY_OMEGA * c * softShadow(o, KEY_C, 1.0 / KEY_TB, segs, prims, n);
        c = dot(nrm, FILL_C);
        if (c > 0.0) e += FILL_L * 0.7 * FILL_OMEGA * c * softShadow(o, FILL_C, 1.0 / FILL_TB, segs, prims, n);
        c = dot(nrm, RIM_C);
        if (c > 0.0) e += RIM_L * 0.7 * RIM_OMEGA * c * softShadow(o, RIM_C, 1.0 / RIM_TB, segs, prims, n);
        // The tent from above, the pale table from below.
        float up = 0.5 + 0.5 * nrm.y;
        float ground = 0.55 * (KEY_L * 0.7 * KEY_OMEGA * KEY_C.y + AMBIENT_E) * 0.3;
        e += mix(ground, AMBIENT_E, up) * ao;
        return float3(e);
    }

    float3 tableRadiance(float3 p, constant GSeg *segs, constant GPrim *prims, uint n) {
        float ao = occlusion(p, float3(0, 1, 0), segs, prims, n, 1.2);
        float3 E = irradiance(p, float3(0, 1, 0), segs, prims, n, ao);
        // A gentle falloff away from the set, so the table reads as a surface
        // and not a flat colour. MODEL.
        float fall = 1.0 / (1.0 + dot(p.xz, p.xz) / (TABLE_FALL * TABLE_FALL));
        return TABLE_ALBEDO / M_PI_F * E * fall;
    }

    float3 fresnel(float c) {
        float x = clamp(c, 0.0, 1.0) * float(LUT_N - 1);
        int i = min(int(x), LUT_N - 2);
        return mix(FRESNEL[i], FRESNEL[i + 1], x - float(i));
    }

    // Heitz (2018), "Sampling the GGX distribution of visible normals",
    // JCGT 7(4): a microfacet normal, in the frame where the surface normal
    // is +z, drawn in proportion to how much of it the viewer sees.
    float3 sampleVNDF(float3 ve, float alpha, float u1, float u2) {
        float3 vh = normalize(float3(alpha * ve.x, alpha * ve.y, ve.z));
        float lensq = vh.x * vh.x + vh.y * vh.y;
        float3 t1 = lensq > 0.0 ? float3(-vh.y, vh.x, 0.0) * rsqrt(lensq) : float3(1.0, 0.0, 0.0);
        float3 t2 = cross(vh, t1);
        float r = sqrt(u1);
        float phi = 2.0 * M_PI_F * u2;
        float a = r * cos(phi);
        float b = r * sin(phi);
        float s = 0.5 * (1.0 + vh.z);
        b = (1.0 - s) * sqrt(max(1.0 - a * a, 0.0)) + s * b;
        float3 nh = a * t1 + b * t2 + sqrt(max(0.0, 1.0 - a * a - b * b)) * vh;
        return normalize(float3(alpha * nh.x, alpha * nh.y, max(0.0, nh.z)));
    }

    // Smith masking for GGX, one direction.
    float smithG1(float c, float alpha) {
        float a2 = alpha * alpha;
        return 2.0 * c / (c + sqrt(a2 + (1.0 - a2) * c * c));
    }

    float hash(uint x) {
        x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16;
        return float(x & 0xffffffU) / 16777216.0;
    }

    // A painted surface: diffuse paint under a dielectric top surface.
    // Returns the radiance and, separately, the part that was mirrored.
    float3 shadeToy(float3 p, float3 rd, float3 nrm, int paint, float u1, float u2,
                    constant GSeg *segs, constant GPrim *prims, constant GPatch *patches, uint n,
                    thread float3 &specOut) {
        float3 v = -rd;
        float cv = max(dot(nrm, v), 1e-4);
        float3 alb = ALBEDO[paint];
        float alpha = ROUGH[paint];
        float ao = occlusion(p, nrm, segs, prims, n, 0.35);
        float3 E = irradiance(p, nrm, segs, prims, n, ao);
        float3 diffuse = alb / M_PI_F * E * (1.0 - fresnel(cv));

        // The mirror part, one GGX sample.
        float3 t = normalize(abs(nrm.y) < 0.9 ? cross(nrm, float3(0, 1, 0)) : cross(nrm, float3(1, 0, 0)));
        float3 b = cross(nrm, t);
        float3 ve = float3(dot(v, t), dot(v, b), cv);
        float3 me = sampleVNDF(ve, alpha, u1, u2);
        float3 m = normalize(t * me.x + b * me.y + nrm * me.z);
        float3 l = reflect(rd, m);
        float cl = dot(nrm, l);
        float3 spec = float3(0.0);
        if (cl > 0.0) {
            float3 w = fresnel(dot(v, m)) * smithG1(cl, alpha);
            float3 o = p + nrm * 0.004;
            float tt;
            int mat, sg, pr;
            float3 Li;
            if (march(o, l, 150.0, segs, prims, n, tt, mat, sg, pr)) {
                float3 h = o + l * tt;
                if (mat == 1) Li = tableRadiance(h, segs, prims, n);
                else {
                    // The toy seen in the toy: its paint lit by the tent. MODEL.
                    int pp = paintAt(h, sg, pr, segs, prims, patches);
                    Li = ALBEDO[pp] / M_PI_F * AMBIENT_E * 0.6;
                }
            } else {
                int tag;
                Li = environment(l, tag);
            }
            spec = w * Li;
        }
        specOut = spec;
        return diffuse + spec;
    }

    // Step 20's tone curve: a shoulder on luminance, the colour scaled to
    // match, so the hue survives; above white it bleeds towards white.
    float3 toneMap(float3 x) {
        float L = dot(x, float3(0.2126, 0.7152, 0.0722));
        float Lm = L * (1.0 + L / 16.0) / (1.0 + L);
        float3 c = x * (Lm / max(L, 1e-5)) * TONE_GAIN;
        float over = max(max(c.r, max(c.g, c.b)) - 1.0, 0.0);
        c = mix(c, float3(1.0), clamp(over, 0.0, 1.0));
        c = clamp(c, 0.0, 1.0);
        return select(1.055 * pow(c, 1.0 / 2.4) - 0.055, 12.92 * c, c <= 0.0031308);
    }

    float3 cameraRay(float2 pixel, constant Params &P) {
        float aspect = float(P.width) / float(P.height);
        float sx = (2.0 * pixel.x / float(P.width) - 1.0) * aspect * P.camFwd.w;
        float sy = (1.0 - 2.0 * pixel.y / float(P.height)) * P.camFwd.w;
        return normalize(P.camFwd.xyz + sx * P.camRight.xyz + sy * P.camUp.xyz);
    }

    // ---------------------------------------------------------------- kernels

    kernel void render(device uchar4 *pixels [[buffer(0)]],
                       device float4 *aux [[buffer(1)]],
                       constant Params &P [[buffer(2)]],
                       constant GSeg *segs [[buffer(3)]],
                       constant GPrim *prims [[buffer(4)]],
                       constant GPatch *patches [[buffer(5)]],
                       uint2 gid [[thread_position_in_grid]]) {
        uint x = gid.x;
        uint y = gid.y + P.rowOffset;
        if (x >= P.width || y >= P.height) return;
        uint n = P.segCount;
        float3 ro = P.camPos.xyz;
        float3 sum = float3(0.0);
        float specSum = 0.0;
        float diffSum = 0.0;
        uint S = P.samples;
        for (uint sy = 0; sy < S; sy++) {
            for (uint sx = 0; sx < S; sx++) {
                uint idx = (y * P.width + x) * 64u + sy * S + sx;
                float2 jitter = float2((float(sx) + 0.5) / float(S), (float(sy) + 0.5) / float(S));
                float3 rd = cameraRay(float2(x, y) + jitter, P);
                float t;
                int mat, seg, prim;
                float3 c;
                if (!march(ro, rd, FAR, segs, prims, n, t, mat, seg, prim)) {
                    int tag;
                    c = environment(rd, tag);
                } else {
                    float3 p = ro + rd * t;
                    if (mat == 1) c = tableRadiance(p, segs, prims, n);
                    else {
                        float3 nrm = toyNormal(p, segs, prims, n);
                        int paint = paintAt(p, seg, prim, segs, prims, patches);
                        float3 spec;
                        c = shadeToy(p, rd, nrm, paint, hash(idx * 2u + 1u), hash(idx * 2u + 2u), segs, prims, patches, n, spec);
                        specSum += dot(spec, float3(0.2126, 0.7152, 0.0722));
                        diffSum += dot(c - spec, float3(0.2126, 0.7152, 0.0722));
                    }
                }
                sum += toneMap(c);
            }
        }
        float3 c = sum / float(S * S);
        pixels[y * P.width + x] = uchar4(uchar3(round(clamp(c, 0.0, 1.0) * 255.0)), 255);

        // What the centre of the pixel saw, for the tests: material, segment,
        // paint, cos of the view to the normal; then the mean mirrored and
        // diffuse luminance over the samples, and F at the centre's angle.
        float3 rdc = cameraRay(float2(x, y) + 0.5, P);
        float t;
        int mat = 0, seg = -1, prim = -1;
        float4 a0 = float4(0.0, -1.0, -1.0, 0.0);
        float4 a1 = float4(specSum / float(S * S), diffSum / float(S * S), 0.0, 0.0);
        if (march(ro, rdc, FAR, segs, prims, n, t, mat, seg, prim)) {
            float3 p = ro + rdc * t;
            if (mat == 2) {
                float3 nrm = toyNormal(p, segs, prims, n);
                float cv = dot(nrm, -rdc);
                a0 = float4(2.0, float(seg), float(paintAt(p, seg, prim, segs, prims, patches)), cv);
                a1.z = fresnel(cv).y;
                a1.w = t;
            } else {
                a0 = float4(1.0, -1.0, -1.0, 0.0);
                a1.w = t;
            }
        }
        aux[2 * (y * P.width + x)] = a0;
        aux[2 * (y * P.width + x) + 1] = a1;
    }

    kernel void probe(device const float4 *points [[buffer(0)]],
                      device float4 *out [[buffer(1)]],
                      constant uint &count [[buffer(2)]],
                      constant GSeg *segs [[buffer(3)]],
                      constant GPrim *prims [[buffer(4)]],
                      constant uint &segCount [[buffer(5)]],
                      uint id [[thread_position_in_grid]]) {
        if (id >= count) return;
        float3 p = points[id].xyz;
        int mat, seg, prim;
        float d = sceneSDF(p, segs, prims, segCount, mat, seg, prim);
        float dt = toyOnly(p, segs, prims, segCount);
        int s2, p2;
        toySDF(p, segs, prims, segCount, FAR, s2, p2);
        out[id] = float4(d, float(mat), dt, float(s2));
    }

    kernel void fresnelProbe(device const float *cosines [[buffer(0)]],
                             device float4 *out [[buffer(1)]],
                             uint id [[thread_position_in_grid]]) {
        out[id] = float4(fresnel(cosines[id]), 0.0);
    }
    """
}

// MARK: - running it

enum ToyError: Error, CustomStringConvertible {
    case noMetalDevice
    case kernelCompile(String)
    case gpu(String)
    case png(String)

    var description: String {
        switch self {
        case .noMetalDevice: return "no Metal GPU found"
        case .kernelCompile(let s): return "could not compile the kernel: \(s)"
        case .gpu(let s): return "GPU error: \(s)"
        case .png(let s): return "could not write PNG: \(s)"
        }
    }
}

struct Params {
    var width: UInt32
    var height: UInt32
    var rowOffset: UInt32
    var samples: UInt32
    var segCount: UInt32
    var pad0: UInt32 = 0
    var pad1: UInt32 = 0
    var pad2: UInt32 = 0
    var camPos: SIMD4<Float>
    var camFwd: SIMD4<Float>     // w: tan of half the vertical field of view
    var camRight: SIMD4<Float>
    var camUp: SIMD4<Float>
}

/// The finished picture, and what each pixel's centre saw.
struct ToyImage {
    let width: Int
    let height: Int
    let pixels: MTLBuffer
    let aux: MTLBuffer

    func rgba(_ x: Int, _ y: Int) -> SIMD4<UInt8> {
        let p = pixels.contents().advanced(by: (y * width + x) * 4).assumingMemoryBound(to: UInt8.self)
        return SIMD4<UInt8>(p[0], p[1], p[2], p[3])
    }
    /// (material: 0 background, 1 table, 2 toy; segment; paint; cos θ).
    func seen(_ x: Int, _ y: Int) -> SIMD4<Float> {
        aux.contents().assumingMemoryBound(to: SIMD4<Float>.self)[2 * (y * width + x)]
    }
    /// (mean mirrored luminance, mean diffuse luminance, F at the centre, ray length).
    func light(_ x: Int, _ y: Int) -> SIMD4<Float> {
        aux.contents().assumingMemoryBound(to: SIMD4<Float>.self)[2 * (y * width + x) + 1]
    }
}

func findDevice() throws -> MTLDevice {
    if let d = MTLCreateSystemDefaultDevice() ?? MTLCopyAllDevices().first { return d }
    throw ToyError.noMetalDevice
}

final class ToyRenderer {
    let device: MTLDevice
    let queue: MTLCommandQueue
    let renderPSO: MTLComputePipelineState
    let probePSO: MTLComputePipelineState
    let fresnelPSO: MTLComputePipelineState
    let width: Int
    let height: Int
    let image: ToyImage

    init(device: MTLDevice, width: Int, height: Int, paints: [Paint], studio: Studio, mutant: Mutant = activeMutant) throws {
        self.device = device
        self.width = width
        self.height = height
        let options = MTLCompileOptions()
        // Precise maths, as in every step.
        options.fastMathEnabled = false
        let library: MTLLibrary
        do {
            library = try device.makeLibrary(source: kernelSource(paints: paints, studio: studio, mutant: mutant), options: options)
        } catch {
            throw ToyError.kernelCompile("\(error)")
        }
        func pipeline(_ name: String) throws -> MTLComputePipelineState {
            guard let f = library.makeFunction(name: name) else { throw ToyError.kernelCompile("no kernel \(name)") }
            do { return try device.makeComputePipelineState(function: f) } catch { throw ToyError.kernelCompile("\(error)") }
        }
        renderPSO = try pipeline("render")
        probePSO = try pipeline("probe")
        fresnelPSO = try pipeline("fresnelProbe")
        guard let q = device.makeCommandQueue(),
              let pixels = device.makeBuffer(length: width * height * 4, options: .storageModeShared),
              let aux = device.makeBuffer(length: width * height * 32, options: .storageModeShared)
        else { throw ToyError.gpu("could not allocate buffers") }
        queue = q
        image = ToyImage(width: width, height: height, pixels: pixels, aux: aux)
    }

    private func buffers(_ toy: PosedToy) throws -> (MTLBuffer, MTLBuffer, MTLBuffer, Int) {
        var data = gpuData(toy)
        guard let sb = device.makeBuffer(bytes: &data.segs, length: MemoryLayout<GSeg>.stride * data.segs.count, options: .storageModeShared),
              let pb = device.makeBuffer(bytes: &data.prims, length: MemoryLayout<GPrim>.stride * data.prims.count, options: .storageModeShared),
              let qb = device.makeBuffer(bytes: &data.patches, length: MemoryLayout<GPatch>.stride * data.patches.count,
                                         options: .storageModeShared)
        else { throw ToyError.gpu("could not upload the toy") }
        return (sb, pb, qb, data.segs.count)
    }

    /// Render the toy in horizontal bands, one command buffer each, so no
    /// single piece of GPU work trips the watchdog on a shared GPU.
    func render(_ toy: PosedToy, camera: Camera, samples: Int) throws -> Double {
        let (sb, pb, qb, n) = try buffers(toy)
        let band: Int = 24
        var gpu: Double = 0
        var row: Int = 0
        while row < height {
            let rows: Int = min(band, height - row)
            guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else {
                throw ToyError.gpu("could not make a command buffer")
            }
            var params = Params(width: UInt32(width), height: UInt32(height), rowOffset: UInt32(row), samples: UInt32(samples),
                                segCount: UInt32(n), camPos: SIMD4<Float>(camera.position, 0),
                                camFwd: SIMD4<Float>(camera.forward, camera.tanHalfFOV),
                                camRight: SIMD4<Float>(camera.right, 0), camUp: SIMD4<Float>(camera.up, 0))
            enc.setComputePipelineState(renderPSO)
            enc.setBuffer(image.pixels, offset: 0, index: 0)
            enc.setBuffer(image.aux, offset: 0, index: 1)
            enc.setBytes(&params, length: MemoryLayout<Params>.stride, index: 2)
            enc.setBuffer(sb, offset: 0, index: 3)
            enc.setBuffer(pb, offset: 0, index: 4)
            enc.setBuffer(qb, offset: 0, index: 5)
            let w: Int = renderPSO.threadExecutionWidth
            let group = MTLSize(width: w, height: max(renderPSO.maxTotalThreadsPerThreadgroup / w / 4, 1), depth: 1)
            enc.dispatchThreads(MTLSize(width: width, height: rows, depth: 1), threadsPerThreadgroup: group)
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            if let e = cb.error { throw ToyError.gpu(e.localizedDescription) }
            gpu += cb.gpuEndTime - cb.gpuStartTime
            row += rows
        }
        return gpu
    }

    /// The scene's distance at points, from the kernel that draws: (scene
    /// distance, material, toy-only distance, nearest segment).
    func probe(_ points: [SIMD3<Float>], toy: PosedToy) throws -> [SIMD4<Float>] {
        let (sb, pb, _, n) = try buffers(toy)
        var pts: [SIMD4<Float>] = points.map { SIMD4<Float>($0, 0) }
        var count = UInt32(pts.count)
        var segCount = UInt32(n)
        guard let inb = device.makeBuffer(bytes: &pts, length: 16 * max(pts.count, 1), options: .storageModeShared),
              let outb = device.makeBuffer(length: 16 * max(pts.count, 1), options: .storageModeShared),
              let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder()
        else { throw ToyError.gpu("could not set up the probe") }
        enc.setComputePipelineState(probePSO)
        enc.setBuffer(inb, offset: 0, index: 0)
        enc.setBuffer(outb, offset: 0, index: 1)
        enc.setBytes(&count, length: 4, index: 2)
        enc.setBuffer(sb, offset: 0, index: 3)
        enc.setBuffer(pb, offset: 0, index: 4)
        enc.setBytes(&segCount, length: 4, index: 5)
        enc.dispatchThreads(MTLSize(width: pts.count, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(probePSO.maxTotalThreadsPerThreadgroup, 256), height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        if let e = cb.error { throw ToyError.gpu(e.localizedDescription) }
        let o = outb.contents().assumingMemoryBound(to: SIMD4<Float>.self)
        return (0..<pts.count).map { o[$0] }
    }

    /// The kernel's own Fresnel reflectance at the given cosines.
    func fresnelOnGPU(_ cosines: [Float]) throws -> [SIMD3<Float>] {
        var c: [Float] = cosines
        guard let inb = device.makeBuffer(bytes: &c, length: 4 * c.count, options: .storageModeShared),
              let outb = device.makeBuffer(length: 16 * c.count, options: .storageModeShared),
              let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder()
        else { throw ToyError.gpu("could not set up the Fresnel probe") }
        enc.setComputePipelineState(fresnelPSO)
        enc.setBuffer(inb, offset: 0, index: 0)
        enc.setBuffer(outb, offset: 0, index: 1)
        enc.dispatchThreads(MTLSize(width: c.count, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        let o = outb.contents().assumingMemoryBound(to: SIMD4<Float>.self)
        return (0..<c.count).map { SIMD3<Float>(o[$0].x, o[$0].y, o[$0].z) }
    }
}

func savePNG(_ image: ToyImage, to url: URL) throws {
    try? FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    guard let provider = CGDataProvider(dataInfo: nil, data: image.pixels.contents(), size: image.width * image.height * 4,
                                        releaseData: { _, _, _ in }),
          let space = CGColorSpace(name: CGColorSpace.sRGB),
          let cg = CGImage(width: image.width, height: image.height, bitsPerComponent: 8, bitsPerPixel: 32,
                           bytesPerRow: image.width * 4, space: space,
                           bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                           provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent),
          let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)
    else { throw ToyError.png("could not set up the encoder for \(url.path)") }
    CGImageDestinationAddImage(dest, cg, nil)
    guard CGImageDestinationFinalize(dest) else { throw ToyError.png("could not write \(url.path)") }
}
