// The canoe, rendered: one GPU thread per pixel marching rays into the
// canoe's distance function, as every step since the ocean has, and a flat
// sea as a plane the rays meet exactly.
//
// The distance function is built only from shapes whose distance is exact
// outside or a lower bound — cylinders (the hull's section, plan and keel
// line are each where big cylinders overlap), capsules, rounded boxes, a
// triangle, a ring — combined by min (union), max (intersection), max(a, −b)
// (a cut) and extrusion, each of which keeps a lower bound a lower bound,
// and, for the mutant's larger canoe, by uniform scaling, which does too.
// No blends. The tests check it.
//
// The same function floats the canoe: a `columns` kernel finds, for each
// vertical line through the hulls, where it enters and leaves them, and
// Buoyancy.swift sums those into the volume under any waterline.

import CoreGraphics
import Foundation
import ImageIO
import Metal
import UniformTypeIdentifiers
import simd

let stepScale: Float = 0.9

func metal(_ v: SIMD3<Float>) -> String { "float3(\(v.x), \(v.y), \(v.z))" }
func metal2(_ v: SIMD2<Float>) -> String { "float2(\(v.x), \(v.y))" }

func kernelSource(sky: Sky, mutant: Mutant) -> String {
    let fore: (SIMD2<Float>, Float) = headCircle(foresail)
    let aft: (SIMD2<Float>, Float) = headCircle(aftsail)
    let beams: String = crossbeamX.map { "\($0)" }.joined(separator: ", ")
    let deckFront: Float = crossbeamX[0] + 0.25
    let deckBack: Float = crossbeamX[crossbeamX.count - 1] - 0.25
    return """
    #include <metal_stdlib>
    using namespace metal;

    struct Params { uint width; uint height; uint rowOffset; uint samples;
                    float sink; float pad0; float pad1; float pad2;
                    float4 camPos; float4 camFwd; float4 camRight; float4 camUp; };

    constant float STEP_SCALE = \(stepScale);
    constant float TONE_GAIN = \(toneGain);
    constant float SCALE = \(canoeScale(mutant));
    constant bool SINGLE = \(mutant == .singleHull ? "true" : "false");

    constant float HULL_Z = \(hullCentreZ);
    constant float GUNWALE = \(gunwaleHeight);
    constant float WIDEST = \(widestAboveKeel);
    constant float SEC_R = \(sectionRadius);
    constant float SEC_C = \(sectionOffset);
    constant float PLAN_R = \(planRadius);
    constant float PLAN_C = \(planOffset);
    constant float ROCKER_R = \(rockerRadius);
    constant float3 SPLAY = \(metal(simd_normalize(SIMD3<Float>(1, 0, sectionSplay))));

    constant float BOW_X0 = \(manuStart(bowManu));
    constant float BOW_R = \(bowManu.radius);
    constant float BOW_S = \(bowManu.sweep);
    constant float STERN_X0 = \(manuStart(sternManu));
    constant float STERN_R = \(sternManu.radius);
    constant float STERN_S = \(sternManu.sweep);
    constant float MANU_T = \(manuHalfThickness);
    constant float MANU_W = \(manuHalfWidth);

    constant int BEAMS = \(crossbeamX.count);
    constant float BEAM_X[\(crossbeamX.count)] = { \(beams) };
    constant float BEAM_R = \(crossbeamRadius);
    constant float BEAM_H = \(crossbeamHalfSpan);
    constant float DECK_TOP = \(deckTop);
    constant float DECK_T = \(deckHalfThickness);
    constant float DECK_X0 = \(deckBack);
    constant float DECK_X1 = \(deckFront);

    constant float MAST_R = \(mastRadius);
    constant float SPAR_R = \(sparRadius);
    constant float MAST_F = \(mastHeightFraction);
    constant float YAW = \(sailYaw);
    constant float F_X = \(foresail.mastX);
    constant float2 F_PEAK = \(metal2(foresail.peak));
    constant float2 F_CLEW = \(metal2(foresail.clew));
    constant float2 F_HC = \(metal2(fore.0));
    constant float F_HR = \(fore.1);
    constant float A_X = \(aftsail.mastX);
    constant float2 A_PEAK = \(metal2(aftsail.peak));
    constant float2 A_CLEW = \(metal2(aftsail.clew));
    constant float2 A_HC = \(metal2(aft.0));
    constant float A_HR = \(aft.1);
    constant float SAIL_T = 0.012;

    constant float3 PAD_TOP = \(metal(paddleTop));
    constant float3 PAD_BOT = \(metal(paddleBottom));
    constant float PAD_R = \(paddleShaftRadius);
    constant float PAD_L = \(paddleBladeLength);
    constant float PAD_W = \(paddleBladeHalfWidth);
    constant float PAD_T = \(paddleBladeHalfThickness);

    constant float3 HULL_ALB = \(metal(hullAlbedo));
    constant float3 WOOD_ALB = \(metal(woodAlbedo));
    constant float3 DECK_ALB = \(metal(deckAlbedo));
    constant float3 SAIL_ALB = \(metal(sailAlbedo));
    constant float SAIL_TRANS = \(sailTranslucency);

    constant float3 SUN = \(metal(sky.sun));
    constant float SUN_E = \(sky.sunIrradiance);
    constant float SUN_L = \(sky.sunRadiance);
    constant float SUN_COS = \(sky.sunCosRadius);
    constant float3 SKY_H = \(metal(sky.horizon));
    constant float3 SKY_Z = \(metal(sky.zenith));
    constant float SKY_E = \(sky.skyIrradiance);

    constant float N_WATER = \(waterIndex);
    constant float SIGMA = \(waterExtinction);
    constant float3 DEEP = \(metal(deepWater));

    // ---------------------------------------------------------------- shapes

    float sdRoundBox(float3 p, float3 b, float r) {
        float3 q = abs(p) - b + r;
        return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0) - r;
    }

    float sdCapsule(float3 p, float3 a, float3 b, float r) {
        float3 pa = p - a, ba = b - a;
        float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
        return length(pa - ba * h) - r;
    }

    // A 2D distance d2 extruded to ±h along the third axis (a lower bound).
    float extrude(float d2, float w, float h) {
        float2 q = float2(d2, abs(w) - h);
        return min(max(q.x, q.y), 0.0) + length(max(q, 0.0));
    }

    // Exact distance to a triangle (Quílez).
    float sdTriangle(float2 p, float2 p0, float2 p1, float2 p2) {
        float2 e0 = p1 - p0, e1 = p2 - p1, e2 = p0 - p2;
        float2 v0 = p - p0, v1 = p - p1, v2 = p - p2;
        float2 pq0 = v0 - e0 * clamp(dot(v0, e0) / dot(e0, e0), 0.0, 1.0);
        float2 pq1 = v1 - e1 * clamp(dot(v1, e1) / dot(e1, e1), 0.0, 1.0);
        float2 pq2 = v2 - e2 * clamp(dot(v2, e2) / dot(e2, e2), 0.0, 1.0);
        float s = sign(e0.x * e2.y - e0.y * e2.x);
        float c0 = v0.x * e0.y - v0.y * e0.x;
        float c1 = v1.x * e1.y - v1.y * e1.x;
        float c2 = v2.x * e2.y - v2.y * e2.x;
        float2 d = min(min(float2(dot(pq0, pq0), s * c0), float2(dot(pq1, pq1), s * c1)), float2(dot(pq2, pq2), s * c2));
        return -sqrt(d.x) * sign(d.y);
    }

    // Distance from p to a line through o along unit a.
    float toAxis(float3 p, float3 o, float3 a) {
        float3 v = p - o;
        return length(v - a * dot(v, a));
    }

    // One hull, centred at z = zc: a vesica in section (two cylinders
    // splaying out from midships, mirrored fore and aft), a lens in plan
    // (two vertical cylinders), an arc in profile (one cylinder across),
    // cut flat at the gunwale. Mirroring by |x| keeps a lower bound: the
    // unmirrored splayed cylinders contain the mirrored hull.
    float hull(float3 p, float zc) {
        float3 q = p - float3(0.0, 0.0, zc);
        float3 m = float3(abs(q.x), q.y, q.z);
        float s1 = toAxis(m, float3(0.0, WIDEST, SEC_C), float3(SPLAY.x, 0.0, SPLAY.z)) - SEC_R;
        float s2 = toAxis(m, float3(0.0, WIDEST, -SEC_C), float3(SPLAY.x, 0.0, -SPLAY.z)) - SEC_R;
        float p1 = length(float2(q.x, q.z + PLAN_C)) - PLAN_R;
        float p2 = length(float2(q.x, q.z - PLAN_C)) - PLAN_R;
        float rk = length(float2(q.x, q.y - ROCKER_R)) - ROCKER_R;
        float top = q.y - GUNWALE;
        return max(max(max(s1, s2), max(p1, p2)), max(rk, top));
    }

    // A manu: an arc leaving the gunwale at x0 (outward along `side`) and
    // sweeping out and up through `sweep` radians, radius r, as a blade.
    float manu(float3 p, float zc, float side, float x0, float r, float sweep) {
        float2 v = float2(side * p.x - x0, p.y - GUNWALE - r);
        float ring = abs(length(v) - r) - MANU_T;
        float h0 = -v.x;
        float h1 = dot(v, float2(cos(sweep), sin(sweep)));
        float d2 = max(ring, max(h0, h1));
        return extrude(d2, p.z - zc, MANU_W);
    }

    float hulls(float3 p) {
        float d = hull(p, HULL_Z);
        if (!SINGLE) d = min(d, hull(p, -HULL_Z));
        return d;
    }

    float manus(float3 p) {
        float d = min(manu(p, HULL_Z, 1.0, BOW_X0, BOW_R, BOW_S), manu(p, HULL_Z, -1.0, STERN_X0, STERN_R, STERN_S));
        if (!SINGLE) {
            d = min(d, min(manu(p, -HULL_Z, 1.0, BOW_X0, BOW_R, BOW_S), manu(p, -HULL_Z, -1.0, STERN_X0, STERN_R, STERN_S)));
        }
        return d;
    }

    float paddle(float3 p) {
        float shaft = sdCapsule(p, PAD_TOP, PAD_BOT, PAD_R);
        float3 a = normalize(PAD_BOT - PAD_TOP);
        float3 b = normalize(cross(float3(0.0, 0.0, 1.0), a));
        float3 d = p - (PAD_BOT - a * (0.5 * PAD_L));
        float blade = sdRoundBox(float3(dot(d, a), dot(d, b), d.z), float3(0.5 * PAD_L, PAD_W, PAD_T), 0.02);
        return min(shaft, blade);
    }

    // What can be in the water: the hulls and the steering paddle.
    float wet(float3 p) { return min(hulls(p), paddle(p)); }

    // A sail about its mast at x = mx: tack on the deck, the sail's plane
    // swung YAW off the centreline, luff spar to the peak, boom to the clew.
    float sail(float3 p, float mx, float2 peak, float2 clew, float2 hc, float hr, thread float &spars) {
        float3 d = p - float3(mx, DECK_TOP, 0.0);
        float3 U = float3(cos(YAW), 0.0, -sin(YAW));
        float3 N = float3(sin(YAW), 0.0, cos(YAW));
        float2 uv = float2(dot(d, U), d.y);
        float tri = sdTriangle(uv, float2(0.0), peak, clew);
        float head = length(uv - hc) - hr;
        float s = extrude(max(tri, -head), dot(d, N), SAIL_T);
        float3 o = float3(mx, DECK_TOP, 0.0);
        float3 P = o + U * peak.x + float3(0.0, peak.y, 0.0);
        float3 C = o + U * clew.x + float3(0.0, clew.y, 0.0);
        spars = min(sdCapsule(p, o, P, SPAR_R), sdCapsule(p, o, C, SPAR_R));
        float mast = sdCapsule(p, o, o + float3(0.0, MAST_F * peak.y, 0.0), MAST_R);
        spars = min(spars, mast);
        return s;
    }

    float crossbeams(float3 p) {
        float beams = 1e9;
        for (int i = 0; i < BEAMS; i++) {
            float y = GUNWALE + BEAM_R;
            beams = min(beams, sdCapsule(p, float3(BEAM_X[i], y, -BEAM_H + BEAM_R), float3(BEAM_X[i], y, BEAM_H - BEAM_R), BEAM_R));
        }
        return beams;
    }

    float sails(float3 p) {
        float a, b;
        return min(sail(p, F_X, F_PEAK, F_CLEW, F_HC, F_HR, a), sail(p, A_X, A_PEAK, A_CLEW, A_HC, A_HR, b));
    }

    // Materials: 2 hull, 3 manu, 4 ʻiako, 5 pola, 6 masts and spars,
    // 7 sails, 8 steering paddle.
    struct Hit { float d; int mat; };

    void take(thread Hit &h, float d, int m) { if (d < h.d) { h.d = d; h.mat = m; } }

    // The canoe at unit scale (PVS's size), keel at y = 0.
    Hit canoe(float3 p) {
        Hit h; h.d = 1e9; h.mat = 0;
        take(h, hulls(p), 2);
        take(h, manus(p), 3);
        take(h, crossbeams(p), 4);
        float3 dc = float3(0.5 * (DECK_X0 + DECK_X1), DECK_TOP - DECK_T, 0.0);
        take(h, sdRoundBox(p - dc, float3(0.5 * (DECK_X1 - DECK_X0), DECK_T, HULL_Z), 0.01), 5);
        float sparsF, sparsA;
        float sails = min(sail(p, F_X, F_PEAK, F_CLEW, F_HC, F_HR, sparsF), sail(p, A_X, A_PEAK, A_CLEW, A_HC, A_HR, sparsA));
        take(h, min(sparsF, sparsA), 6);
        take(h, sails, 7);
        take(h, paddle(p), 8);
        return h;
    }

    // World: the drawn canoe (scaled) sunk `sink` below the sea at y = 0.
    Hit scene(float3 p, float sink) {
        Hit h = canoe((p + float3(0.0, sink, 0.0)) / SCALE);
        h.d *= SCALE;
        return h;
    }

    float sceneD(float3 p, float sink) { return scene(p, sink).d; }

    // The ray's span inside the canoe's bounding box (world).
    bool boxSpan(float3 ro, float3 rd, float sink, thread float &t0, thread float &t1) {
        float3 lo = float3(-12.2, -0.2, -3.2) * SCALE - float3(0.0, sink, 0.0);
        float3 hi = float3(10.2, 16.5, 3.2) * SCALE - float3(0.0, sink, 0.0);
        float3 inv = 1.0 / rd;
        float3 a = (lo - ro) * inv, b = (hi - ro) * inv;
        float3 mn = min(a, b), mx = max(a, b);
        t0 = max(max(mn.x, mn.y), max(mn.z, 0.0));
        t1 = min(min(mx.x, mx.y), mx.z);
        return t1 > t0;
    }

    bool march(float3 ro, float3 rd, float tMax, float sink, thread float &t, thread int &mat) {
        float t0, t1;
        if (!boxSpan(ro, rd, sink, t0, t1)) return false;
        t = t0;
        float tEnd = min(t1, tMax);
        for (int i = 0; i < 400; i++) {
            float3 p = ro + rd * t;
            Hit h = scene(p, sink);
            float eps = 0.0008 + t * 0.00008;
            if (h.d < eps) { mat = h.mat; return true; }
            t += h.d * STEP_SCALE;
            if (t > tEnd) break;
        }
        return false;
    }

    float3 normalAt(float3 p, float sink) {
        const float e = 0.002;
        float3 k0 = float3(1, -1, -1), k1 = float3(-1, -1, 1), k2 = float3(-1, 1, -1), k3 = float3(1, 1, 1);
        return normalize(k0 * sceneD(p + k0 * e, sink) + k1 * sceneD(p + k1 * e, sink)
                       + k2 * sceneD(p + k2 * e, sink) + k3 * sceneD(p + k3 * e, sink));
    }

    // ---------------------------------------------------------------- light

    float3 skyRadiance(float3 r) {
        float3 c = mix(SKY_H, SKY_Z, 1.0 - exp(-6.0 * max(r.y, 0.0))) * (SKY_E / M_PI_F);
        if (dot(r, SUN) > SUN_COS) c += float3(SUN_L);
        return c;
    }

    float softShadow(float3 ro, float3 rd, float k, float sink) {
        float t0, t1;
        if (!boxSpan(ro, rd, sink, t0, t1)) return 1.0;
        float res = 1.0;
        float t = max(t0, 0.01);
        for (int i = 0; i < 120; i++) {
            float h = sceneD(ro + rd * t, sink);
            res = min(res, k * h / t);
            if (res < 0.002) return 0.0;
            t += clamp(h, 0.01, 1.0);
            if (t > t1) break;
        }
        return clamp(res, 0.0, 1.0);
    }

    float occlusion(float3 p, float3 nrm, float sink) {
        float occ = 0.0;
        float w = 1.0;
        for (int i = 1; i <= 5; i++) {
            float h = 0.06 * float(i);
            float d = sceneD(p + nrm * h, sink);
            occ += (h - clamp(d, 0.0, h)) * w;
            w *= 0.6;
        }
        return clamp(1.0 - occ * 0.4 / 0.06, 0.0, 1.0);
    }

    float3 albedoOf(int mat) {
        if (mat == 2) return HULL_ALB;
        if (mat == 5) return DECK_ALB;
        if (mat == 7) return SAIL_ALB;
        return WOOD_ALB;
    }

    // A point of the canoe, lit by the sun and the sky (and, below the
    // waterline, dimmed by the water above it).
    float3 shadeCanoe(float3 p, float3 rd, int mat, float sink) {
        float3 nrm = normalAt(p, sink);
        bool sail = mat == 7;
        if (sail && dot(nrm, rd) > 0.0) nrm = -nrm;
        float3 alb = albedoOf(mat);
        float ao = occlusion(p, nrm, sink);
        float c = dot(nrm, SUN);
        float direct = max(c, 0.0);
        if (sail) direct += SAIL_TRANS * max(-c, 0.0);
        float depth = max(-p.y, 0.0);
        float e;
        if (depth > 0.0) {
            // Under water: sunlight and skylight both cross `depth` of it.
            float down = exp(-SIGMA * depth);
            float slant = exp(-SIGMA * depth / max(SUN.y, 0.1));
            float sunPart = SUN_E * direct * slant;
            float skyPart = SKY_E * (0.5 + 0.5 * nrm.y);
            e = sunPart + skyPart * ao * down;
        } else {
            float sh = direct > 0.0 ? softShadow(p + nrm * 0.01, SUN, 24.0, sink) : 0.0;
            float up = 0.5 + 0.5 * nrm.y;
            float sunPart = SUN_E * direct * sh;
            float skyPart = SKY_E * (up + 0.08 * (1.0 - up));
            e = sunPart + skyPart * ao;
        }
        float3 col = alb / M_PI_F * e;
        if (mat == 2) {
            // Painted, resined hulls: a little sky in them. MODEL.
            float cv = max(dot(nrm, -rd), 0.0);
            float f = 0.04 + 0.96 * pow(1.0 - cv, 5.0);
            col += f * skyRadiance(reflect(rd, nrm)) * (depth > 0.0 ? 0.0 : 1.0) * 0.6;
        }
        return col;
    }

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

    // What a camera ray sees: 0 sky, 1 open water, 2–8 the canoe above the
    // water, 12–18 the canoe seen through it.
    float3 trace(float3 ro, float3 rd, float sink, thread int &seen) {
        float tb;
        int mat;
        bool hit = march(ro, rd, 400.0, sink, tb, mat);
        float tw = (ro.y > 0.0 && rd.y < 0.0) ? -ro.y / rd.y : 1e9;
        if (hit && tb < tw) { seen = mat; return shadeCanoe(ro + rd * tb, rd, mat, sink); }
        if (tw > 1e8) { seen = 0; return skyRadiance(rd); }
        // The sea's surface: reflection plus what is seen through it.
        float3 pw = ro + rd * tw;
        float F = fresnelDiel(N_WATER, -rd.y);
        float3 rr = float3(rd.x, -rd.y, rd.z);
        float tr;
        int mr;
        float3 refl = march(pw + rr * 0.002, rr, 200.0, sink, tr, mr) ? shadeCanoe(pw + rr * (0.002 + tr), rr, mr, sink)
                                                                     : skyRadiance(rr);
        float3 below;
        if (hit) {
            float s = tb - tw;
            float att = exp(-SIGMA * s);
            below = shadeCanoe(ro + rd * tb, rd, mat, sink) * att + DEEP * (1.0 - att);
            seen = 10 + mat;
        } else {
            below = DEEP;
            seen = 1;
        }
        return F * refl + (1.0 - F) * below;
    }

    kernel void render(device uchar4 *pixels [[buffer(0)]],
                       device float4 *aux [[buffer(1)]],
                       constant Params &P [[buffer(2)]],
                       uint2 gid [[thread_position_in_grid]]) {
        uint x = gid.x;
        uint y = gid.y + P.rowOffset;
        if (x >= P.width || y >= P.height) return;
        float3 ro = P.camPos.xyz;
        float3 sum = float3(0.0);
        uint S = P.samples;
        int seen = 0;
        for (uint sy = 0; sy < S; sy++) {
            for (uint sx = 0; sx < S; sx++) {
                float2 jitter = float2((float(sx) + 0.5) / float(S), (float(sy) + 0.5) / float(S));
                float3 rd = cameraRay(float2(x, y) + jitter, P);
                sum += toneMap(trace(ro, rd, P.sink, seen));
            }
        }
        float3 c = sum / float(S * S);
        pixels[y * P.width + x] = uchar4(uchar3(round(clamp(c, 0.0, 1.0) * 255.0)), 255);
        float3 rdc = cameraRay(float2(x, y) + 0.5, P);
        float tb;
        int mat = 0;
        float4 a0 = float4(0.0);
        bool hit = march(ro, rdc, 400.0, P.sink, tb, mat);
        float tw = (ro.y > 0.0 && rdc.y < 0.0) ? -ro.y / rdc.y : 1e9;
        if (hit && tb < tw) a0 = float4(float(mat), tb, 0.0, 0.0);
        else if (hit) a0 = float4(float(10 + mat), tb, tw, 0.0);
        else if (tw < 1e8) a0 = float4(1.0, tw, tw, 0.0);
        aux[y * P.width + x] = a0;
    }

    // (scene distance, material, hulls-and-manu distance, wet distance) at
    // each world point, the canoe sunk `sink`.
    kernel void probe(device const float4 *points [[buffer(0)]],
                      device float4 *out [[buffer(1)]],
                      constant uint &count [[buffer(2)]],
                      constant float &sink [[buffer(3)]],
                      uint id [[thread_position_in_grid]]) {
        if (id >= count) return;
        float3 p = points[id].xyz;
        Hit h = scene(p, sink);
        float3 q = (p + float3(0.0, sink, 0.0)) / SCALE;
        out[id] = float4(h.d, float(h.mat), min(hulls(q), manus(q)) * SCALE, wet(q) * SCALE);
    }

    // (sails, crossbeams, 0, 0) at each world point: parts the tests count.
    kernel void parts(device const float4 *points [[buffer(0)]],
                      device float4 *out [[buffer(1)]],
                      constant uint &count [[buffer(2)]],
                      constant float &sink [[buffer(3)]],
                      uint id [[thread_position_in_grid]]) {
        if (id >= count) return;
        float3 q = (points[id].xyz + float3(0.0, sink, 0.0)) / SCALE;
        out[id] = float4(sails(q) * SCALE, crossbeams(q) * SCALE, 0.0, 0.0);
    }

    // For each vertical line (x, z) through the drawn canoe (keel at y = 0),
    // where it first meets what can be wet coming up from below, and coming
    // down from above: (−1e9 / 1e9 for a miss).
    kernel void columns(device const float2 *xz [[buffer(0)]],
                        device float2 *out [[buffer(1)]],
                        constant uint &count [[buffer(2)]],
                        uint id [[thread_position_in_grid]]) {
        if (id >= count) return;
        float2 c = xz[id];
        float lo = -1.0 * SCALE, hi = 4.0 * SCALE;
        float y = lo;
        float yb = 1e9;
        for (int i = 0; i < 600; i++) {
            float d = wet(float3(c.x, y, c.y) / SCALE) * SCALE;
            if (d < 1e-6) { yb = y; break; }
            y += d;
            if (y > hi) break;
        }
        y = hi;
        float yt = -1e9;
        for (int i = 0; i < 600; i++) {
            float d = wet(float3(c.x, y, c.y) / SCALE) * SCALE;
            if (d < 1e-6) { yt = y; break; }
            y -= d;
            if (y < lo) break;
        }
        out[id] = float2(yb, yt);
    }
    """
}

// MARK: - running it

enum CanoeError: Error, CustomStringConvertible {
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
    var sink: Float
    var pad0: Float = 0
    var pad1: Float = 0
    var pad2: Float = 0
    var camPos: SIMD4<Float>
    var camFwd: SIMD4<Float>
    var camRight: SIMD4<Float>
    var camUp: SIMD4<Float>
}

struct CanoeImage {
    let width: Int
    let height: Int
    let pixels: MTLBuffer
    let aux: MTLBuffer

    func rgba(_ x: Int, _ y: Int) -> SIMD4<UInt8> {
        let p = pixels.contents().advanced(by: (y * width + x) * 4).assumingMemoryBound(to: UInt8.self)
        return SIMD4<UInt8>(p[0], p[1], p[2], p[3])
    }
    /// (what is seen, ray length to it, ray length to the sea, 0).
    func seen(_ x: Int, _ y: Int) -> SIMD4<Float> {
        aux.contents().assumingMemoryBound(to: SIMD4<Float>.self)[y * width + x]
    }
}

func findDevice() throws -> MTLDevice {
    if let d = MTLCreateSystemDefaultDevice() ?? MTLCopyAllDevices().first { return d }
    throw CanoeError.noMetalDevice
}

final class CanoeRenderer {
    let device: MTLDevice
    let queue: MTLCommandQueue
    let renderPSO: MTLComputePipelineState
    let probePSO: MTLComputePipelineState
    let columnsPSO: MTLComputePipelineState
    let partsPSO: MTLComputePipelineState
    let width: Int
    let height: Int
    let image: CanoeImage

    init(device: MTLDevice, width: Int, height: Int, sky: Sky, mutant: Mutant = activeMutant) throws {
        self.device = device
        self.width = width
        self.height = height
        let options = MTLCompileOptions()
        // Precise maths, as in every step.
        options.fastMathEnabled = false
        let library: MTLLibrary
        do {
            library = try device.makeLibrary(source: kernelSource(sky: sky, mutant: mutant), options: options)
        } catch {
            throw CanoeError.kernelCompile("\(error)")
        }
        func pipeline(_ name: String) throws -> MTLComputePipelineState {
            guard let f = library.makeFunction(name: name) else { throw CanoeError.kernelCompile("no kernel \(name)") }
            do { return try device.makeComputePipelineState(function: f) } catch { throw CanoeError.kernelCompile("\(error)") }
        }
        renderPSO = try pipeline("render")
        probePSO = try pipeline("probe")
        columnsPSO = try pipeline("columns")
        partsPSO = try pipeline("parts")
        guard let q = device.makeCommandQueue(),
              let pixels = device.makeBuffer(length: width * height * 4, options: .storageModeShared),
              let aux = device.makeBuffer(length: width * height * 16, options: .storageModeShared)
        else { throw CanoeError.gpu("could not allocate buffers") }
        queue = q
        image = CanoeImage(width: width, height: height, pixels: pixels, aux: aux)
    }

    func render(camera: Camera, samples: Int, sink: Float) throws -> Double {
        let band: Int = 16
        var gpu: Double = 0
        var row: Int = 0
        while row < height {
            let rows: Int = min(band, height - row)
            guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else {
                throw CanoeError.gpu("could not make a command buffer")
            }
            var params = Params(width: UInt32(width), height: UInt32(height), rowOffset: UInt32(row), samples: UInt32(samples),
                                sink: sink, camPos: SIMD4<Float>(camera.position, 0),
                                camFwd: SIMD4<Float>(camera.forward, camera.tanHalfFOV),
                                camRight: SIMD4<Float>(camera.right, 0), camUp: SIMD4<Float>(camera.up, 0))
            enc.setComputePipelineState(renderPSO)
            enc.setBuffer(image.pixels, offset: 0, index: 0)
            enc.setBuffer(image.aux, offset: 0, index: 1)
            enc.setBytes(&params, length: MemoryLayout<Params>.stride, index: 2)
            let w: Int = renderPSO.threadExecutionWidth
            let group = MTLSize(width: w, height: max(renderPSO.maxTotalThreadsPerThreadgroup / w / 4, 1), depth: 1)
            enc.dispatchThreads(MTLSize(width: width, height: rows, depth: 1), threadsPerThreadgroup: group)
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            if let e = cb.error { throw CanoeError.gpu(e.localizedDescription) }
            gpu += cb.gpuEndTime - cb.gpuStartTime
            row += rows
        }
        return gpu
    }

    /// (scene distance, material, hulls-and-manu distance, wet distance) at
    /// each world point, the canoe sunk `sink`.
    func probe(_ points: [SIMD3<Float>], sink: Float) throws -> [SIMD4<Float>] {
        try run(probePSO, points, sink: sink)
    }

    /// (sails, crossbeams, 0, 0) distances at each world point.
    func parts(_ points: [SIMD3<Float>], sink: Float) throws -> [SIMD4<Float>] {
        try run(partsPSO, points, sink: sink)
    }

    func run(_ pso: MTLComputePipelineState, _ points: [SIMD3<Float>], sink: Float) throws -> [SIMD4<Float>] {
        var out: [SIMD4<Float>] = []
        out.reserveCapacity(points.count)
        let chunk: Int = 262_144
        var start: Int = 0
        while start < points.count {
            let end: Int = min(start + chunk, points.count)
            var pts: [SIMD4<Float>] = points[start..<end].map { SIMD4<Float>($0, 0) }
            var count = UInt32(pts.count)
            var s: Float = sink
            guard let inb = device.makeBuffer(bytes: &pts, length: 16 * max(pts.count, 1), options: .storageModeShared),
                  let outb = device.makeBuffer(length: 16 * max(pts.count, 1), options: .storageModeShared),
                  let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder()
            else { throw CanoeError.gpu("could not set up the probe") }
            enc.setComputePipelineState(pso)
            enc.setBuffer(inb, offset: 0, index: 0)
            enc.setBuffer(outb, offset: 0, index: 1)
            enc.setBytes(&count, length: 4, index: 2)
            enc.setBytes(&s, length: 4, index: 3)
            enc.dispatchThreads(MTLSize(width: pts.count, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: min(pso.maxTotalThreadsPerThreadgroup, 256), height: 1, depth: 1))
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            if let e = cb.error { throw CanoeError.gpu(e.localizedDescription) }
            let o = outb.contents().assumingMemoryBound(to: SIMD4<Float>.self)
            for i in 0..<pts.count { out.append(o[i]) }
            start = end
        }
        return out
    }

    /// For each (x, z), where the vertical line meets the wet parts of the
    /// drawn canoe (keel at 0) from below and from above.
    func columns(_ xz: [SIMD2<Float>]) throws -> [SIMD2<Float>] {
        var out: [SIMD2<Float>] = []
        out.reserveCapacity(xz.count)
        let chunk: Int = 262_144
        var start: Int = 0
        while start < xz.count {
            let end: Int = min(start + chunk, xz.count)
            var pts: [SIMD2<Float>] = Array(xz[start..<end])
            var count = UInt32(pts.count)
            guard let inb = device.makeBuffer(bytes: &pts, length: 8 * max(pts.count, 1), options: .storageModeShared),
                  let outb = device.makeBuffer(length: 8 * max(pts.count, 1), options: .storageModeShared),
                  let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder()
            else { throw CanoeError.gpu("could not set up the columns") }
            enc.setComputePipelineState(columnsPSO)
            enc.setBuffer(inb, offset: 0, index: 0)
            enc.setBuffer(outb, offset: 0, index: 1)
            enc.setBytes(&count, length: 4, index: 2)
            enc.dispatchThreads(MTLSize(width: pts.count, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: min(columnsPSO.maxTotalThreadsPerThreadgroup, 256), height: 1, depth: 1))
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            if let e = cb.error { throw CanoeError.gpu(e.localizedDescription) }
            let o = outb.contents().assumingMemoryBound(to: SIMD2<Float>.self)
            for i in 0..<pts.count { out.append(o[i]) }
            start = end
        }
        return out
    }
}

func savePNG(_ image: CanoeImage, to url: URL) throws {
    try? FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    guard let provider = CGDataProvider(dataInfo: nil, data: image.pixels.contents(), size: image.width * image.height * 4,
                                        releaseData: { _, _, _ in }),
          let space = CGColorSpace(name: CGColorSpace.sRGB),
          let cg = CGImage(width: image.width, height: image.height, bitsPerComponent: 8, bitsPerPixel: 32,
                           bytesPerRow: image.width * 4, space: space,
                           bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                           provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent),
          let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)
    else { throw CanoeError.png("could not set up the encoder for \(url.path)") }
    CGImageDestinationAddImage(dest, cg, nil)
    guard CGImageDestinationFinalize(dest) else { throw CanoeError.png("could not write \(url.path)") }
}
