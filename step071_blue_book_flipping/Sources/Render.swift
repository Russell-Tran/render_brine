// The book, rendered: one GPU thread per pixel marching rays into the
// book's distance function, as every step since the ocean has.
//
// The book reaches the GPU as data — a list of primitives, each a 2D shape
// in the book's cross-section extruded along the spine, or a 3D tube — so
// the tests probe exactly what was drawn, and the inset is drawn from the
// same distance function by the probe kernel.
//
// Every primitive's distance is exact outside it, or a lower bound (the
// extrusion's edge rounding); the scene is their plain minimum — no blends
// at all — so it is a lower bound everywhere outside (tested).
//
// The cloth is shaded as Cloth.swift describes: dyed diffuse under
// Estevez & Kulla's sheen, lit by each softbox (evaluated at its centre,
// with a soft shadow) and by the tent (through the sheen's albedo). Its
// weave is a height field of warp and filling threads at Group B's counts
// [E&R], perturbing the normal, faded where a pixel is wider than a thread.

import CoreGraphics
import Foundation
import ImageIO
import Metal
import UniformTypeIdentifiers
import simd

/// Ray steps trust this fraction of each distance.
let stepScale: Float = 0.9

func metal(_ v: SIMD3<Float>) -> String { "float3(\(v.x), \(v.y), \(v.z))" }

func kernelSource(studio: Studio, mutant: Mutant) -> String {
    let lut: String = topAlbedoTable(mutant).map { "\($0)" }.joined(separator: ", ")
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
    let plastic: Int = 0
    return """
    #include <metal_stdlib>
    using namespace metal;

    struct Params { uint width; uint height; uint rowOffset; uint samples;
                    uint primCount; uint pad0; uint pad1; uint pad2;
                    float4 camPos; float4 camFwd; float4 camRight; float4 camUp; };
    struct GPrim { float4 a; float4 b; float4 c; float4 z; float4 bound; int4 info; };

    constant float STEP_SCALE = \(stepScale);
    constant float FAR = 4000.0;
    constant float TONE_GAIN = \(toneGain);
    constant int PLASTIC = \(plastic);
    constant float SHEEN_R = \(sheenRoughness);
    constant float3 TOP_COL = \(metal(topColour(mutant)));
    constant float GRAIN = \(threadGrain);
    constant float PLASTIC_A = \(plasticAlpha);
    constant float WARP_P = \(warpPitch);
    constant float FILL_P = \(fillingPitch);
    constant float STRIPE = \(headbandStripe);
    constant float3 CLOTH_ALB = \(metal(clothAlbedo));
    constant float3 PAPER_ALB = \(metal(paperAlbedo));
    constant float3 HB_A = \(metal(headbandAlbedoA));
    constant float3 HB_B = \(metal(headbandAlbedoB));

    \(box("KEY", k))
    \(box("FILL", f))
    \(box("RIM", r))
    constant float TENT_H = \(studio.tentHorizon);
    constant float TENT_Z = \(studio.tentZenith);
    constant float AMBIENT_E = \(studio.ambientIrradiance);
    constant float3 TABLE_ALBEDO = \(metal(studio.tableAlbedo));

    constant int LUT_N = \(albedoTableSize);
    constant float TOP_ALBEDO[\(albedoTableSize)] = { \(lut) };

    // ---------------------------------------------------------------- shapes

    float2 toFrame(float2 p, float2 c, float2 axis) {
        float2 d = p - c;
        return float2(dot(d, axis), dot(d, float2(-axis.y, axis.x)));
    }

    // A ring sector (flat ends) or, with rounded = true, an arc of a thick
    // line (round ends). Local frame: the middle direction is +y.
    float sdSector(float2 p, float2 sc, float r, float th, bool rounded) {
        p.x = abs(p.x);
        float2 e = float2(sc.x, sc.y);            // the end's direction
        float2 en = float2(sc.y, -sc.x);          // outward normal of the end ray
        float side = dot(p, en);
        if (side <= 0.0) return abs(length(p) - r) - th * 0.5;
        if (rounded) return length(p - e * r) - th * 0.5;
        float along = dot(p, e);
        return length(float2(side, max(abs(along - r) - th * 0.5, 0.0)));
    }

    float sdPolygon(float2 p, constant float2 *v, int first, int n) {
        float d = dot(p - v[first], p - v[first]);
        float s = 1.0;
        for (int i = 0, j = n - 1; i < n; j = i, i++) {
            float2 vi = v[first + i];
            float2 vj = v[first + j];
            float2 e = vj - vi;
            float2 w = p - vi;
            float2 b = w - e * clamp(dot(w, e) / dot(e, e), 0.0, 1.0);
            d = min(d, dot(b, b));
            bool c1 = p.y >= vi.y;
            bool c2 = p.y < vj.y;
            bool c3 = e.x * w.y > e.y * w.x;
            if ((c1 && c2 && c3) || (!c1 && !c2 && !c3)) s = -s;
        }
        return s * sqrt(d);
    }

    float primSDF2(constant GPrim &g, float2 p, constant float2 *verts) {
        int kind = g.info.x;
        if (kind == 0) {
            float2 a = g.a.xy, b = g.a.zw;
            float2 pa = p - a, ba = b - a;
            float h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-12), 0.0, 1.0);
            return length(pa - ba * h) - g.b.x;
        }
        if (kind == 1) {
            float2 q = abs(toFrame(p, g.a.xy, g.a.zw)) - g.b.xy + g.b.z;
            return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - g.b.z;
        }
        if (kind == 2) {
            float2 mid = g.a.zw;
            float2 d = p - g.a.xy;
            float2 q = float2(dot(d, float2(mid.y, -mid.x)), dot(d, mid));
            return sdSector(q, g.c.xy, g.b.x, g.b.y, false);
        }
        if (kind == 4) {
            // A thick arc by its chord: x along the chord (folded), y from
            // the chord towards the arc's middle; the centre is at
            // (0, −r cos h), far off when the arc is nearly straight, so
            // |p − centre| − r is taken as (|p − c|² − r²)/(|p − c| + r)
            // with r² − (r cos h)² written (r sin h)².
            float2 mid = g.a.zw;
            float2 d = p - g.a.xy;
            float lx = abs(dot(d, float2(mid.y, -mid.x)));
            float ly = dot(d, mid);
            float rs = g.c.z;               // r sin h, half the chord
            float rc = g.c.w;               // r cos h, the centre's distance
            float side = lx * g.c.y - ly * g.c.x - rc * g.c.x;
            if (side > 0.0) return length(float2(lx - rs, ly)) - g.b.y * 0.5;
            float num = lx * lx + ly * ly + 2.0 * ly * rc - rs * rs;
            float len = sqrt(lx * lx + (ly + rc) * (ly + rc));
            return abs(num / (len + g.b.x)) - g.b.y * 0.5;
        }
        return sdPolygon(p, verts, g.info.w, int(g.z.w));
    }

    float primSDF(constant GPrim &g, float3 p, constant float2 *verts) {
        int kind = g.info.x;
        if (kind == 5) {
            float3 a = g.a.xyz, b = g.b.xyz;
            float3 pa = p - a, ba = b - a;
            float h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-12), 0.0, 1.0);
            return length(pa - ba * h) - g.c.x;
        }
        float d2 = primSDF2(g, p.xy, verts);
        float zc = 0.5 * (g.z.x + g.z.y);
        float hz = 0.5 * (g.z.y - g.z.x);
        float rr = g.z.z;
        float2 w = float2(d2 + rr, abs(p.z - zc) - hz + rr);
        return min(max(w.x, w.y), 0.0) + length(max(w, 0.0)) - rr;
    }

    // The book: the nearest visible primitive, skipping any whose bounding
    // sphere is farther than the best so far.
    float bookSDF(float3 p, constant GPrim *prims, constant float2 *verts, uint n, float cap, thread int &mat) {
        float best = cap;
        mat = 0;
        for (uint i = 0; i < n; i++) {
            constant GPrim &g = prims[i];
            if (g.info.z != 0) continue;
            float dc = length(p - g.bound.xyz) - g.bound.w;
            if (dc > best) continue;
            float d = primSDF(g, p, verts);
            if (d < best) { best = d; mat = g.info.y; }
        }
        return best;
    }

    float sceneSDF(float3 p, constant GPrim *prims, constant float2 *verts, uint n, thread int &mat) {
        float dt = p.y;
        float d = bookSDF(p, prims, verts, n, dt, mat);
        if (mat != 0) return d;
        mat = 1;
        return dt;
    }

    bool march(float3 ro, float3 rd, float tMax, constant GPrim *prims, constant float2 *verts, uint n,
               thread float &t, thread int &mat) {
        t = 0.0;
        for (int i = 0; i < 400; i++) {
            float3 p = ro + rd * t;
            float d = sceneSDF(p, prims, verts, n, mat);
            float eps = 0.002 + t * 0.000004;
            if (d < eps) return true;
            t += d * STEP_SCALE;
            if (t > tMax) break;
        }
        return false;
    }

    float bookOnly(float3 p, constant GPrim *prims, constant float2 *verts, uint n) {
        int m;
        return bookSDF(p, prims, verts, n, FAR, m);
    }

    float3 bookNormal(float3 p, constant GPrim *prims, constant float2 *verts, uint n) {
        const float h = 0.004;
        float3 k0 = float3(1, -1, -1), k1 = float3(-1, -1, 1), k2 = float3(-1, 1, -1), k3 = float3(1, 1, 1);
        return normalize(k0 * bookOnly(p + k0 * h, prims, verts, n) + k1 * bookOnly(p + k1 * h, prims, verts, n)
                       + k2 * bookOnly(p + k2 * h, prims, verts, n) + k3 * bookOnly(p + k3 * h, prims, verts, n));
    }

    // ---------------------------------------------------------------- light

    float boxWeight(float3 r, float3 C, float3 A, float3 B, float ta, float tb) {
        float c = dot(r, C);
        if (c <= 0.0) return 0.0;
        float a = dot(r, A) / (ta * c);
        float b = dot(r, B) / (tb * c);
        if (abs(a) >= 1.0 || abs(b) >= 1.0) return 0.0;
        return mix(0.4, 1.0, (1.0 - a * a) * (1.0 - b * b));
    }

    float3 environment(float3 r) {
        float w = boxWeight(r, KEY_C, KEY_A, KEY_B, KEY_TA, KEY_TB);
        if (w > 0.0) return float3(KEY_L * w);
        w = boxWeight(r, FILL_C, FILL_A, FILL_B, FILL_TA, FILL_TB);
        if (w > 0.0) return float3(FILL_L * w);
        w = boxWeight(r, RIM_C, RIM_A, RIM_B, RIM_TA, RIM_TB);
        if (w > 0.0) return float3(RIM_L * w);
        float L = r.y < 0.0 ? TENT_H * 0.8 : mix(TENT_H, TENT_Z, sqrt(r.y));
        return float3(L);
    }

    float softShadow(float3 ro, float3 rd, float k, constant GPrim *prims, constant float2 *verts, uint n) {
        float res = 1.0;
        float t = 0.03;
        for (int i = 0; i < 90; i++) {
            float h = bookOnly(ro + rd * t, prims, verts, n);
            res = min(res, k * h / t);
            if (res < 0.002) return 0.0;
            t += clamp(h, 0.04, 8.0);
            if (t > 400.0) break;
        }
        return clamp(res, 0.0, 1.0);
    }

    float occlusion(float3 p, float3 nrm, constant GPrim *prims, constant float2 *verts, uint n, float scale) {
        float occ = 0.0;
        float w = 1.0;
        for (int i = 1; i <= 5; i++) {
            float h = scale * float(i);
            float3 q = p + nrm * h;
            float d = min(bookOnly(q, prims, verts, n), q.y + 1e3 * step(0.5, nrm.y));
            occ += (h - clamp(d, 0.0, h)) * w;
            w *= 0.6;
        }
        return clamp(1.0 - occ * 0.4 / scale, 0.0, 1.0);
    }

    // Each softbox as seen from p: its direction, radiance × solid angle,
    // and how much of it is unshadowed.
    struct Lights { float3 dir[3]; float power[3]; };

    Lights lightsAt(float3 p, float3 nrm, constant GPrim *prims, constant float2 *verts, uint n) {
        Lights L;
        float3 o = p + nrm * 0.02;
        L.dir[0] = KEY_C; L.dir[1] = FILL_C; L.dir[2] = RIM_C;
        L.power[0] = dot(nrm, KEY_C) > 0.0 ? KEY_L * 0.7 * KEY_OMEGA * softShadow(o, KEY_C, 1.0 / KEY_TB, prims, verts, n) : 0.0;
        L.power[1] = dot(nrm, FILL_C) > 0.0 ? FILL_L * 0.7 * FILL_OMEGA * softShadow(o, FILL_C, 1.0 / FILL_TB, prims, verts, n) : 0.0;
        L.power[2] = dot(nrm, RIM_C) > 0.0 ? RIM_L * 0.7 * RIM_OMEGA * softShadow(o, RIM_C, 1.0 / RIM_TB, prims, verts, n) : 0.0;
        return L;
    }

    float ambientAt(float3 nrm, float ao) {
        float up = 0.5 + 0.5 * nrm.y;
        float ground = 0.55 * (KEY_L * 0.7 * KEY_OMEGA * KEY_C.y + AMBIENT_E) * 0.3;
        return mix(ground, AMBIENT_E, up) * ao;
    }

    float3 tableRadiance(float3 p, constant GPrim *prims, constant float2 *verts, uint n) {
        float3 nrm = float3(0, 1, 0);
        float ao = occlusion(p, nrm, prims, verts, n, 3.0);
        Lights L = lightsAt(p, nrm, prims, verts, n);
        float e = ambientAt(nrm, ao);
        for (int i = 0; i < 3; i++) e += L.power[i] * max(dot(nrm, L.dir[i]), 0.0);
        float fall = 1.0 / (1.0 + dot(p.xz - float2(80.0, 0.0), p.xz - float2(80.0, 0.0)) / 400000.0);
        return TABLE_ALBEDO / M_PI_F * e * fall;
    }

    // ---------------------------------------------------------------- the cloth's lobe

    float fresnelDiel(float n, float c) {
        c = clamp(c, 0.0, 1.0);
        float s2 = 1.0 - c * c;
        float t2 = 1.0 - s2 / (n * n);
        if (t2 <= 0.0) return 1.0;
        float ct = sqrt(t2);
        float rs = (c - n * ct) / (c + n * ct);
        float rp = (n * c - ct) / (n * c + ct);
        return 0.5 * (rs * rs + rp * rp);
    }

    float sheenLfit(float x, float r) {
        float w = (1.0 - r) * (1.0 - r);
        float a = w * 25.3245 + (1.0 - w) * 21.5473;
        float b = w * 3.32435 + (1.0 - w) * 3.82987;
        float c = w * 0.16801 + (1.0 - w) * 0.19823;
        float d = w * -1.27393 + (1.0 - w) * -1.97760;
        float e = w * -4.85967 + (1.0 - w) * -4.32054;
        return a / (1.0 + b * pow(x, c)) + d * x + e;
    }

    float sheenLambda(float c, float r) {
        if (c < 0.5) return exp(sheenLfit(c, r));
        return exp(2.0 * sheenLfit(0.5, r) - sheenLfit(1.0 - c, r));
    }

    // The top lobe: Estevez & Kulla's sheen, or the mutant's plastic GGX.
    float topBRDF(float3 v, float3 l, float3 nrm) {
        float cv = dot(v, nrm);
        float cl = dot(l, nrm);
        if (cv <= 0.0 || cl <= 0.0) return 0.0;
        float3 h = normalize(v + l);
        float ch = clamp(dot(h, nrm), 0.0, 1.0);
        if (PLASTIC == 1) {
            float a2 = PLASTIC_A * PLASTIC_A;
            float t = ch * ch * (a2 - 1.0) + 1.0;
            float D = a2 / (M_PI_F * t * t);
            float g1v = 2.0 * cv / (cv + sqrt(a2 + (1.0 - a2) * cv * cv));
            float g1l = 2.0 * cl / (cl + sqrt(a2 + (1.0 - a2) * cl * cl));
            return fresnelDiel(1.5, dot(v, h)) * g1v * g1l * D / (4.0 * cv * cl);
        }
        float s = sqrt(max(1.0 - ch * ch, 0.0));
        float inv = 1.0 / SHEEN_R;
        float D = (2.0 + inv) * pow(s, inv) / (2.0 * M_PI_F);
        float G = 1.0 / (1.0 + sheenLambda(cv, SHEEN_R) + sheenLambda(cl, SHEEN_R));
        return G * D / (4.0 * cv * cl);
    }

    // Heitz (2018), "Sampling the GGX distribution of visible normals",
    // JCGT 7(4), as step 60 uses it: for the plastic mutant's mirror.
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

    float topAlbedo(float c) {
        float x = clamp(c, 0.0, 1.0) * float(LUT_N - 1);
        int i = min(int(x), LUT_N - 2);
        return mix(TOP_ALBEDO[i], TOP_ALBEDO[i + 1], x - float(i));
    }

    // ---------------------------------------------------------------- the weave

    // A plain weave's height: warp threads across the cover (along x, per
    // Bailey), filling along the spine (z), each passing over one and under
    // the next. u runs along the warp, w along the filling; heights in mm.
    float yarn(float f) { float x = 2.0 * f - 1.0; return sqrt(max(1.0 - x * x, 0.0)); }

    float weaveHeight(float u, float w) {
        float iu = floor(w / WARP_P);      // which warp thread (they are spaced along w)
        float iw = floor(u / FILL_P);      // which filling thread (spaced along u)
        float fw = w / WARP_P - iu;
        float fu = u / FILL_P - iw;
        // Warp: a yarn across w, rising over alternate filling threads along u.
        float warp = yarn(fw) * (0.55 + 0.45 * cos(M_PI_F * (u / FILL_P - 0.5) + M_PI_F * iu));
        float fill = yarn(fu) * (0.55 + 0.45 * cos(M_PI_F * (w / WARP_P - 0.5) + M_PI_F * (iw + 1.0)));
        return 0.06 * max(warp, fill);
    }

    // The weave's slope in (u, w), by central differences.
    float2 weaveSlope(float u, float w) {
        float e = 0.012;
        float hu = weaveHeight(u + e, w) - weaveHeight(u - e, w);
        float hw = weaveHeight(u, w + e) - weaveHeight(u, w - e);
        return float2(hu, hw) / (2.0 * e);
    }

    // The normal with the weave on it. The cloth's (u, w) axes follow the
    // face it is on: on the covers u = x, w = z; on the spine and fore-edge
    // u = y (the warp wraps round the edge), w = z; on head and tail u = x,
    // w = y. Blended by the normal. `fade` goes to zero where a pixel spans
    // a thread or more.
    float3 weaveNormal(float3 p, float3 nrm, float fade) {
        if (fade <= 0.0) return nrm;
        float3 a = abs(nrm);
        float3 wts = pow(a, float3(4.0));
        wts /= (wts.x + wts.y + wts.z);
        float3 g = float3(0.0);
        // Cover faces (normal ±y): slope in x (u) and z (w).
        float2 sy = weaveSlope(p.x, p.z);
        g += wts.y * float3(sy.x, 0.0, sy.y);
        // Spine / fore-edge faces (normal ±x): u along y, w along z.
        float2 sx = weaveSlope(p.y, p.z);
        g += wts.x * float3(0.0, sx.x, sx.y);
        // Head / tail (normal ±z): u along x, w along y.
        float2 sz = weaveSlope(p.x, p.y);
        g += wts.z * float3(sz.x, sz.y, 0.0);
        g -= nrm * dot(g, nrm);
        return normalize(nrm - g * fade);
    }

    float hash(uint x) {
        x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16;
        return float(x & 0xffffffU) / 16777216.0;
    }

    // One thread's shade: a per-thread offset plus a slow wander along it
    // (value noise every 2.5 mm), in ±1.
    float threadShade(float index, float along, uint salt) {
        uint i = uint(int(index) + 100000) * 2u + salt;
        float a = along / 2.5;
        float fa = floor(a);
        float t = a - fa;
        t = t * t * (3.0 - 2.0 * t);
        float n0 = hash(i * 7919u + uint(int(fa) + 100000));
        float n1 = hash(i * 7919u + uint(int(fa) + 100001));
        return (hash(i) - 0.5) + (mix(n0, n1, t) - 0.5) * 0.8;
    }

    // The grain at a point, by which thread is on top there.
    float clothGrain(float3 p, float3 nrm) {
        float3 a = abs(nrm);
        float u, w;
        if (a.y >= a.x && a.y >= a.z) { u = p.x; w = p.z; }
        else if (a.x >= a.z) { u = p.y; w = p.z; }
        else { u = p.x; w = p.y; }
        float iu = floor(w / WARP_P);
        float iw = floor(u / FILL_P);
        // Which is on top: the warp where it rises over a filling thread.
        float warpUp = cos(M_PI_F * (u / FILL_P - 0.5) + M_PI_F * iu);
        float s = warpUp > 0.0 ? threadShade(iu, u, 0u) : threadShade(iw, w, 1u);
        return 1.0 + GRAIN * 2.0 * s;
    }

    // Cloth: dyed diffuse scaled by 1 − the top lobe's albedo, plus the top
    // lobe from each light and from the tent.
    float3 shadeCloth(float3 p, float3 rd, float3 nrm, float3 geomN, float fade, float u1, float u2,
                      constant GPrim *prims, constant float2 *verts, uint n, thread float &topOut) {
        float3 v = -rd;
        float3 sn = weaveNormal(p, nrm, fade);
        float cv = max(dot(sn, v), 1e-4);
        float ao = occlusion(p, geomN, prims, verts, n, 0.8);
        Lights L = lightsAt(p, geomN, prims, verts, n);
        float amb = ambientAt(geomN, ao);
        float3 scale = 1.0 - TOP_COL * topAlbedo(cv);
        float e = amb;
        float top = 0.0;
        for (int i = 0; i < 3; i++) {
            float cl = dot(sn, L.dir[i]);
            if (cl <= 0.0) continue;
            e += L.power[i] * cl;
            if (PLASTIC == 0) top += topBRDF(v, L.dir[i], sn) * L.power[i] * cl;
        }
        if (PLASTIC == 0) {
            // The sheen is broad, so each softbox counts from its centre;
            // the tent through the lobe's albedo: its radiance is smooth, so
            // the lobe's integral over it is its albedo times the mean. MODEL.
            top += topAlbedo(cv) * amb / M_PI_F;
        } else {
            // A mirror-like lobe sees the softboxes' shapes: one sample of
            // the visible normals, reflected into the studio, per ray.
            float3 tt = normalize(abs(sn.y) < 0.9 ? cross(sn, float3(0, 1, 0)) : cross(sn, float3(1, 0, 0)));
            float3 bb = cross(sn, tt);
            float3 ve = float3(dot(v, tt), dot(v, bb), cv);
            float3 me = sampleVNDF(ve, PLASTIC_A, u1, u2);
            float3 m = normalize(tt * me.x + bb * me.y + sn * me.z);
            float3 l = reflect(rd, m);
            float cl = dot(sn, l);
            if (cl > 0.0) {
                float a2 = PLASTIC_A * PLASTIC_A;
                float g1 = 2.0 * cl / (cl + sqrt(a2 + (1.0 - a2) * cl * cl));
                float sh = softShadow(p + geomN * 0.02, l, 40.0, prims, verts, n);
                float3 env = environment(l);
                float Li = l.y < 0.0 ? 0.3 : env.x * mix(0.35, 1.0, sh);
                top += fresnelDiel(1.5, dot(v, m)) * g1 * Li;
            }
        }
        float3 topC = TOP_COL * top;
        topOut = dot(topC, float3(0.2126, 0.7152, 0.0722));
        float3 alb = CLOTH_ALB * clothGrain(p, geomN);
        return alb / M_PI_F * e * scale + topC;
    }

    float3 shadeMatte(float3 p, float3 nrm, float3 alb, constant GPrim *prims, constant float2 *verts, uint n) {
        float ao = occlusion(p, nrm, prims, verts, n, 0.6);
        Lights L = lightsAt(p, nrm, prims, verts, n);
        float e = ambientAt(nrm, ao);
        for (int i = 0; i < 3; i++) e += L.power[i] * max(dot(nrm, L.dir[i]), 0.0);
        return alb / M_PI_F * e;
    }

    float3 headbandAlbedo(float3 p) {
        // Stripes along the cord, which runs up the spine: by height.
        float s = floor(p.y / STRIPE);
        return fmod(abs(s), 2.0) < 1.0 ? HB_A : HB_B;
    }

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
                       constant GPrim *prims [[buffer(3)]],
                       constant float2 *verts [[buffer(4)]],
                       uint2 gid [[thread_position_in_grid]]) {
        uint x = gid.x;
        uint y = gid.y + P.rowOffset;
        if (x >= P.width || y >= P.height) return;
        uint n = P.primCount;
        float3 ro = P.camPos.xyz;
        float3 sum = float3(0.0);
        float topSum = 0.0;
        float allSum = 0.0;
        uint S = P.samples;
        // A pixel's width in mm per mm of distance, for the weave's fade.
        float pixAngle = 2.0 * P.camFwd.w / float(P.height);
        for (uint sy = 0; sy < S; sy++) {
            for (uint sx = 0; sx < S; sx++) {
                float2 jitter = float2((float(sx) + 0.5) / float(S), (float(sy) + 0.5) / float(S));
                float3 rd = cameraRay(float2(x, y) + jitter, P);
                float t;
                int mat;
                float3 c;
                if (!march(ro, rd, FAR, prims, verts, n, t, mat)) {
                    c = environment(rd);
                } else {
                    float3 p = ro + rd * t;
                    if (mat == 1) c = tableRadiance(p, prims, verts, n);
                    else {
                        float3 nrm = bookNormal(p, prims, verts, n);
                        if (mat == 2) {
                            float footprint = pixAngle * t / max(abs(dot(nrm, rd)), 0.2);
                            float fade = clamp(1.6 - footprint / WARP_P, 0.0, 1.0);
                            float top;
                            uint idx = (y * P.width + x) * 64u + sy * S + sx;
                            c = shadeCloth(p, rd, nrm, nrm, fade, hash(idx * 2u + 1u), hash(idx * 2u + 2u), prims, verts, n, top);
                            topSum += top;
                            allSum += dot(c, float3(0.2126, 0.7152, 0.0722));
                        } else if (mat == 4) c = shadeMatte(p, nrm, headbandAlbedo(p), prims, verts, n);
                        else c = shadeMatte(p, nrm, PAPER_ALB, prims, verts, n);
                    }
                }
                sum += toneMap(c);
            }
        }
        float3 c = sum / float(S * S);
        pixels[y * P.width + x] = uchar4(uchar3(round(clamp(c, 0.0, 1.0) * 255.0)), 255);

        // What the pixel's centre saw, for the tests: material, ray length,
        // cos of view to normal; mean top-lobe and total luminance on cloth.
        float3 rdc = cameraRay(float2(x, y) + 0.5, P);
        float t;
        int mat = 0;
        float4 a0 = float4(0.0, 0.0, 0.0, 0.0);
        if (march(ro, rdc, FAR, prims, verts, n, t, mat)) {
            float3 p = ro + rdc * t;
            float cv = 0.0;
            if (mat >= 2) cv = dot(bookNormal(p, prims, verts, n), -rdc);
            a0 = float4(float(mat), t, cv, 0.0);
        }
        aux[2 * (y * P.width + x)] = a0;
        aux[2 * (y * P.width + x) + 1] = float4(topSum / float(S * S), allSum / float(S * S), 0.0, 0.0);
    }

    // The scene at points: (scene distance, visible material, book-only
    // distance, the innermost material, internal parts included), then the
    // book's distance again without the bounding-sphere shortcut.
    kernel void probe(device const float4 *points [[buffer(0)]],
                      device float4 *out [[buffer(1)]],
                      constant uint &count [[buffer(2)]],
                      constant GPrim *prims [[buffer(3)]],
                      constant float2 *verts [[buffer(4)]],
                      constant uint &primCount [[buffer(5)]],
                      uint id [[thread_position_in_grid]]) {
        if (id >= count) return;
        float3 p = points[id].xyz;
        int mat;
        float d = sceneSDF(p, prims, verts, primCount, mat);
        float db = bookOnly(p, prims, verts, primCount);
        int inner = 0;
        float naive = FAR;
        for (uint i = 0; i < primCount; i++) {
            float di = primSDF(prims[i], p, verts);
            if (di < 0.0) inner = prims[i].info.y;
            if (prims[i].info.z == 0) naive = min(naive, di);
        }
        out[2 * id] = float4(d, float(mat), db, float(inner));
        out[2 * id + 1] = float4(naive, 0.0, 0.0, 0.0);
    }

    // The kernel's own top lobe, for (v, l) pairs about the normal +z.
    kernel void brdfProbe(device const float4 *vl [[buffer(0)]],
                          device float *out [[buffer(1)]],
                          uint id [[thread_position_in_grid]]) {
        float3 v = vl[2 * id].xyz;
        float3 l = vl[2 * id + 1].xyz;
        out[id] = topBRDF(v, l, float3(0.0, 0.0, 1.0));
    }
    """
}

// MARK: - running it

enum BookError: Error, CustomStringConvertible {
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
    var primCount: UInt32
    var pad0: UInt32 = 0
    var pad1: UInt32 = 0
    var pad2: UInt32 = 0
    var camPos: SIMD4<Float>
    var camFwd: SIMD4<Float>
    var camRight: SIMD4<Float>
    var camUp: SIMD4<Float>
}

struct BookImage {
    let width: Int
    let height: Int
    let pixels: MTLBuffer
    let aux: MTLBuffer

    func rgba(_ x: Int, _ y: Int) -> SIMD4<UInt8> {
        let p = pixels.contents().advanced(by: (y * width + x) * 4).assumingMemoryBound(to: UInt8.self)
        return SIMD4<UInt8>(p[0], p[1], p[2], p[3])
    }
    /// (material, ray length, cos θ, 0).
    func seen(_ x: Int, _ y: Int) -> SIMD4<Float> {
        aux.contents().assumingMemoryBound(to: SIMD4<Float>.self)[2 * (y * width + x)]
    }
    /// (mean top-lobe radiance, mean luminance) on cloth.
    func light(_ x: Int, _ y: Int) -> SIMD4<Float> {
        aux.contents().assumingMemoryBound(to: SIMD4<Float>.self)[2 * (y * width + x) + 1]
    }
}

func findDevice() throws -> MTLDevice {
    if let d = MTLCreateSystemDefaultDevice() ?? MTLCopyAllDevices().first { return d }
    throw BookError.noMetalDevice
}

final class BookRenderer {
    let device: MTLDevice
    let queue: MTLCommandQueue
    let renderPSO: MTLComputePipelineState
    let probePSO: MTLComputePipelineState
    let brdfPSO: MTLComputePipelineState
    let width: Int
    let height: Int
    let image: BookImage

    init(device: MTLDevice, width: Int, height: Int, studio: Studio, mutant: Mutant = activeMutant) throws {
        self.device = device
        self.width = width
        self.height = height
        let options = MTLCompileOptions()
        // Precise maths, as in every step.
        options.fastMathEnabled = false
        let library: MTLLibrary
        do {
            library = try device.makeLibrary(source: kernelSource(studio: studio, mutant: mutant), options: options)
        } catch {
            throw BookError.kernelCompile("\(error)")
        }
        func pipeline(_ name: String) throws -> MTLComputePipelineState {
            guard let f = library.makeFunction(name: name) else { throw BookError.kernelCompile("no kernel \(name)") }
            do { return try device.makeComputePipelineState(function: f) } catch { throw BookError.kernelCompile("\(error)") }
        }
        renderPSO = try pipeline("render")
        probePSO = try pipeline("probe")
        brdfPSO = try pipeline("brdfProbe")
        guard let q = device.makeCommandQueue(),
              let pixels = device.makeBuffer(length: width * height * 4, options: .storageModeShared),
              let aux = device.makeBuffer(length: width * height * 32, options: .storageModeShared)
        else { throw BookError.gpu("could not allocate buffers") }
        queue = q
        image = BookImage(width: width, height: height, pixels: pixels, aux: aux)
    }

    private func buffers(_ scene: Scene) throws -> (MTLBuffer, MTLBuffer, Int) {
        var prims: [GPrim] = scene.prims
        var verts: [SIMD2<Float>] = scene.verts.isEmpty ? [SIMD2<Float>(0, 0)] : scene.verts
        guard let pb = device.makeBuffer(bytes: &prims, length: MemoryLayout<GPrim>.stride * prims.count, options: .storageModeShared),
              let vb = device.makeBuffer(bytes: &verts, length: MemoryLayout<SIMD2<Float>>.stride * verts.count, options: .storageModeShared)
        else { throw BookError.gpu("could not upload the book") }
        return (pb, vb, prims.count)
    }

    /// Render in horizontal bands, one command buffer each, so no single
    /// piece of GPU work trips the watchdog on a shared GPU.
    func render(_ scene: Scene, camera: Camera, samples: Int) throws -> Double {
        let (pb, vb, n) = try buffers(scene)
        let band: Int = 16
        var gpu: Double = 0
        var row: Int = 0
        while row < height {
            let rows: Int = min(band, height - row)
            guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else {
                throw BookError.gpu("could not make a command buffer")
            }
            var params = Params(width: UInt32(width), height: UInt32(height), rowOffset: UInt32(row), samples: UInt32(samples),
                                primCount: UInt32(n), camPos: SIMD4<Float>(camera.position, 0),
                                camFwd: SIMD4<Float>(camera.forward, camera.tanHalfFOV),
                                camRight: SIMD4<Float>(camera.right, 0), camUp: SIMD4<Float>(camera.up, 0))
            enc.setComputePipelineState(renderPSO)
            enc.setBuffer(image.pixels, offset: 0, index: 0)
            enc.setBuffer(image.aux, offset: 0, index: 1)
            enc.setBytes(&params, length: MemoryLayout<Params>.stride, index: 2)
            enc.setBuffer(pb, offset: 0, index: 3)
            enc.setBuffer(vb, offset: 0, index: 4)
            let w: Int = renderPSO.threadExecutionWidth
            let group = MTLSize(width: w, height: max(renderPSO.maxTotalThreadsPerThreadgroup / w / 4, 1), depth: 1)
            enc.dispatchThreads(MTLSize(width: width, height: rows, depth: 1), threadsPerThreadgroup: group)
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            if let e = cb.error { throw BookError.gpu(e.localizedDescription) }
            gpu += cb.gpuEndTime - cb.gpuStartTime
            row += rows
        }
        return gpu
    }

    /// The book's distance without the bounding-sphere shortcut, from the
    /// last probe.
    var lastNaive: [Float] = []

    /// (scene distance, visible material, book-only distance, innermost material).
    func probe(_ points: [SIMD3<Float>], scene: Scene) throws -> [SIMD4<Float>] {
        let (pb, vb, n) = try buffers(scene)
        var out: [SIMD4<Float>] = []
        var naiveOut: [Float] = []
        out.reserveCapacity(points.count)
        let chunk: Int = 262_144
        var start: Int = 0
        while start < points.count {
            let end: Int = min(start + chunk, points.count)
            var pts: [SIMD4<Float>] = points[start..<end].map { SIMD4<Float>($0, 0) }
            var count = UInt32(pts.count)
            var primCount = UInt32(n)
            guard let inb = device.makeBuffer(bytes: &pts, length: 16 * max(pts.count, 1), options: .storageModeShared),
                  let outb = device.makeBuffer(length: 32 * max(pts.count, 1), options: .storageModeShared),
                  let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder()
            else { throw BookError.gpu("could not set up the probe") }
            enc.setComputePipelineState(probePSO)
            enc.setBuffer(inb, offset: 0, index: 0)
            enc.setBuffer(outb, offset: 0, index: 1)
            enc.setBytes(&count, length: 4, index: 2)
            enc.setBuffer(pb, offset: 0, index: 3)
            enc.setBuffer(vb, offset: 0, index: 4)
            enc.setBytes(&primCount, length: 4, index: 5)
            enc.dispatchThreads(MTLSize(width: pts.count, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: min(probePSO.maxTotalThreadsPerThreadgroup, 256), height: 1, depth: 1))
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            if let e = cb.error { throw BookError.gpu(e.localizedDescription) }
            let o = outb.contents().assumingMemoryBound(to: SIMD4<Float>.self)
            for i in 0..<pts.count { out.append(o[2 * i]); naiveOut.append(o[2 * i + 1].x) }
            start = end
        }
        lastNaive = naiveOut
        return out
    }

    /// The kernel's own top lobe at (v, l) pairs about +z.
    func brdfOnGPU(_ pairs: [(SIMD3<Float>, SIMD3<Float>)]) throws -> [Float] {
        var vl: [SIMD4<Float>] = []
        for (v, l) in pairs { vl.append(SIMD4<Float>(v, 0)); vl.append(SIMD4<Float>(l, 0)) }
        guard let inb = device.makeBuffer(bytes: &vl, length: 16 * vl.count, options: .storageModeShared),
              let outb = device.makeBuffer(length: 4 * pairs.count, options: .storageModeShared),
              let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder()
        else { throw BookError.gpu("could not set up the BRDF probe") }
        enc.setComputePipelineState(brdfPSO)
        enc.setBuffer(inb, offset: 0, index: 0)
        enc.setBuffer(outb, offset: 0, index: 1)
        enc.dispatchThreads(MTLSize(width: pairs.count, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        let o = outb.contents().assumingMemoryBound(to: Float.self)
        return (0..<pairs.count).map { o[$0] }
    }
}

func savePNG(_ image: BookImage, to url: URL) throws {
    try? FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    guard let provider = CGDataProvider(dataInfo: nil, data: image.pixels.contents(), size: image.width * image.height * 4,
                                        releaseData: { _, _, _ in }),
          let space = CGColorSpace(name: CGColorSpace.sRGB),
          let cg = CGImage(width: image.width, height: image.height, bitsPerComponent: 8, bitsPerPixel: 32,
                           bytesPerRow: image.width * 4, space: space,
                           bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                           provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent),
          let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)
    else { throw BookError.png("could not set up the encoder for \(url.path)") }
    CGImageDestinationAddImage(dest, cg, nil)
    guard CGImageDestinationFinalize(dest) else { throw BookError.png("could not write \(url.path)") }
}
