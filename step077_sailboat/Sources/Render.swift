// The boat, rendered: one GPU thread per pixel marching rays into the boat's
// distance function, as every step since the ocean has; the water is a box
// the ray's path through is found exactly, and what is seen through it is
// dimmed by Beer–Lambert and tinted by the water's own light.
//
// The distance function: the hull is an implicit shape (Boat.swift) divided
// by a constant bound on its gradient and held above its bounding box's
// exact distance; the spars, foils and sail are exact primitives (cylinders,
// capsules, stadium extrusions, a triangle). They are combined by min, max
// and max(a, −b), each of which keeps a lower bound a lower bound, and, for
// the mutant's larger boat, by uniform scaling, which does too. No blends,
// no local slopes. The tests check it.
//
// Materials are shaded like step 72's plastic: a diffuse body under a rough
// dielectric skin with visible-normal GGX sampling (Heitz 2018), copied.

import CoreGraphics
import Foundation
import ImageIO
import Metal
import UniformTypeIdentifiers
import simd

let stepScale: Float = 0.9

func metal(_ v: SIMD3<Float>) -> String { "float3(\(v.x), \(v.y), \(v.z))" }
func metalD(_ v: SIMD3<Double>) -> String { "float3(\(Float(v.x)), \(Float(v.y)), \(Float(v.z)))" }
func f(_ v: Double) -> String { "\(Float(v))" }

/// Gelcoat, sail and spar looks. All MODEL (index 1.5 for the gelcoat's
/// polyester skin; albedos so white reads as white under the tent).
let hullAlbedo = SIMD3<Float>(0.80, 0.81, 0.80)
let deckAlbedo = SIMD3<Float>(0.66, 0.67, 0.68)
let sailAlbedo = SIMD3<Float>(0.86, 0.86, 0.84)
let sparAlbedo = SIMD3<Float>(0.42, 0.43, 0.45)
let foilAlbedo = SIMD3<Float>(0.74, 0.75, 0.74)
let fittingAlbedo = SIMD3<Float>(0.08, 0.08, 0.09)
let gelcoatIndex: Double = 1.5

func kernelSource(studio: Studio, water: WaterBlock, draught: Double, mutant: Mutant) -> String {
    let k: Softbox = studio.key
    let fl: Softbox = studio.fill
    let r: Softbox = studio.rim
    let h: HullShape = hullShape
    let sail: Sail = sailGeometry()
    let th: Double = tackAngle
    let boomDir = SIMD3<Double>(-sin(th), cos(th), 0)
    let boomEnd: SIMD3<Double> = sail.tack + boomDir * boomLength
    let boardTip: Double = keelHeight(boardX) - boardDepthBelowHull
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
                    float4 camPos; float4 camFwd; float4 camRight; float4 camUp; };

    constant float STEP_SCALE = \(stepScale);
    constant float FAR = 80.0;
    constant float TONE_GAIN = \(toneGain);
    constant float SCALE = \(f(boatScale(mutant)));
    constant float DRAUGHT = \(f(draught));
    constant bool HAS_BOARD = \(mutant == .noKeel ? "false" : "true");

    // The hull (Boat.swift, HullShape)
    constant float STERN = \(f(h.stern));
    constant float BOW = \(f(h.bow));
    constant float HB = \(f(h.halfBeam));
    constant float DEPTH = \(f(h.depth));
    constant float TRIM_T = \(f(h.trimDraught));
    constant float KEEL_X = \(f(h.keelLowX));
    constant float AFT_P = \(f(h.aftPower));
    constant float FORE_P = \(f(hullForePower));
    constant float A1 = \(f(h.bottomHalf));
    constant float H1 = \(f(h.deadrise));
    constant float BILGE = \(f(h.bilge));
    constant float SIDE_S = \(f(hullSideSlope));
    constant float WIDEST_X = \(f(h.widestX));
    constant float TRANSOM_HALF = \(f(h.transomHalf));
    constant float STEM_HALF = \(f(h.stemHalf));
    constant float STEM_FOOT = \(f(h.stemFoot));
    constant float MIN_NARROW = \(f(h.minNarrow));
    constant float LIPSCHITZ = \(f(h.lipschitz));
    constant float3 COCKPIT_C = \(metalD(cockpitCentre));
    constant float3 COCKPIT_H = \(metalD(cockpitHalf));

    // The rig and the foils
    constant float MAST_X = \(f(mastX));
    constant float MAST_FOOT = \(f(mastFootY));
    constant float MAST_LOWER = \(f(mastLower));
    constant float MAST_LEN = \(f(mastLength));
    constant float MAST_R0 = \(f(mastLowerRadius));
    constant float MAST_R1 = \(f(mastTopRadius));
    constant float3 TACK = \(metalD(sail.tack));
    constant float3 HEAD = \(metalD(sail.head));
    constant float3 CLEW = \(metalD(sail.clew));
    constant float3 BOOM_END = \(metalD(boomEnd));
    constant float BOARD_X = \(f(boardX));
    constant float BOARD_CHORD = \(f(boardChord));
    constant float BOARD_THICK = \(f(boardThickness));
    constant float BOARD_TIP = \(f(boardTip));
    constant float BOARD_TOP = \(f(boardTopY));
    constant float RUDDER_X = \(f(rudderX));
    constant float RUDDER_CHORD = \(f(rudderChord));
    constant float RUDDER_THICK = \(f(rudderThickness));
    constant float RUDDER_BOTTOM = \(f(rudderBottomY));
    constant float RUDDER_TOP = \(f(rudderTopY));

    constant float N_GELCOAT = \(f(gelcoatIndex));
    constant float N_WATER = \(f(waterIndex));
    constant float3 HULL_ALB = \(metal(hullAlbedo));
    constant float3 DECK_ALB = \(metal(deckAlbedo));
    constant float3 SAIL_ALB = \(metal(sailAlbedo));
    constant float3 SPAR_ALB = \(metal(sparAlbedo));
    constant float3 FOIL_ALB = \(metal(foilAlbedo));
    constant float3 FITTING_ALB = \(metal(fittingAlbedo));
    constant float3 W_LO = \(metal(water.lo));
    constant float3 W_HI = \(metal(water.hi));
    constant float3 W_SIGMA = \(metal(water.sigma));
    constant float3 W_GLOW = \(metal(water.glow));

    \(box("KEY", k))
    \(box("FILL", fl))
    \(box("RIM", r))
    constant float TENT_H = \(studio.tentHorizon);
    constant float TENT_Z = \(studio.tentZenith);
    constant float AMBIENT_E = \(studio.ambientIrradiance);

    // ---------------------------------------------------------------- shapes

    float sdBox(float3 p, float3 b) {
        float3 q = abs(p) - b;
        return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
    }

    float sdRoundBox(float3 p, float3 b, float r) { return sdBox(p, b - r) - r; }

    float sdCapsule(float3 p, float3 a, float3 b, float r) {
        float3 pa = p - a, ba = b - a;
        float hh = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
        return length(pa - ba * hh) - r;
    }

    // A vertical cylinder at (x, 0) in xz from y0 to y1.
    float sdVCyl(float3 p, float x, float r, float y0, float y1) {
        float2 w = float2(length(float2(p.x - x, p.z)) - r, abs(p.y - 0.5 * (y0 + y1)) - 0.5 * (y1 - y0));
        return min(max(w.x, w.y), 0.0) + length(max(w, 0.0));
    }

    // Inigo Quilez's unsigned distance to a triangle.
    float udTriangle(float3 p, float3 a, float3 b, float3 c) {
        float3 ba = b - a, pa = p - a, cb = c - b, pb = p - b, ac = a - c, pc = p - c;
        float3 nor = cross(ba, ac);
        float s = sign(dot(cross(ba, nor), pa)) + sign(dot(cross(cb, nor), pb)) + sign(dot(cross(ac, nor), pc));
        if (s < 2.0) {
            float3 e0 = ba * clamp(dot(ba, pa) / dot(ba, ba), 0.0, 1.0) - pa;
            float3 e1 = cb * clamp(dot(cb, pb) / dot(cb, cb), 0.0, 1.0) - pb;
            float3 e2 = ac * clamp(dot(ac, pc) / dot(ac, ac), 0.0, 1.0) - pc;
            return sqrt(min(min(dot(e0, e0), dot(e1, e1)), dot(e2, e2)));
        }
        return sqrt(dot(nor, pa) * dot(nor, pa) / dot(nor, nor));
    }

    // The section: a flat V rising H1 over A1, a round bilge, flared sides.
    float sdSection(float y, float az) {
        float ln = sqrt(A1 * A1 + H1 * H1);
        float ez = A1 / ln, ey = H1 / ln;
        float u = clamp(az * ez + (y - BILGE) * ey, 0.0, ln);
        float dSeg = length(float2(az - u * ez, y - BILGE - u * ey));
        float n = sqrt(SIDE_S * SIDE_S + 1.0);
        float dz = SIDE_S / n, dy = 1.0 / n;
        float yc = BILGE + H1;
        float t = max((az - A1) * dz + (y - yc) * dy, 0.0);
        float dRay = length(float2(az - A1 - t * dz, y - yc - t * dy));
        bool inside = (y - BILGE) * A1 > az * H1 && az < A1 + SIDE_S * (y - yc);
        float d = min(dSeg, dRay);
        return (inside ? -d : d) - BILGE;
    }

    float keelHeight(float x) {
        if (x < KEEL_X) return TRIM_T * pow(max((KEEL_X - x) / (KEEL_X - STERN), 0.0), AFT_P);
        return STEM_FOOT * pow(max((x - KEEL_X) / (BOW - KEEL_X), 0.0), FORE_P);
    }

    float planHalf(float x) {
        if (x < WIDEST_X) {
            float u = clamp((WIDEST_X - x) / (WIDEST_X - STERN), 0.0, 1.0);
            return TRANSOM_HALF + (HB - TRANSOM_HALF) * (1.0 - u * u);
        }
        float u = clamp((x - WIDEST_X) / (BOW - WIDEST_X), 0.0, 1.0);
        return HB * (1.0 - u * u) + STEM_HALF * u * u;
    }

    float hullDist(float3 p) {
        float3 lo = float3(STERN, 0.0, -HB), hi = float3(BOW, DEPTH, HB);
        float3 q = clamp(p, lo - 0.1, hi + 0.1);
        float B = planHalf(q.x);
        float k = max(B / HB, MIN_NARROW);
        float az = abs(q.z);
        float sec = sdSection(q.y - keelHeight(q.x), az / k) * k;
        float imp = max(max(sec, az - B), max(max(STERN - q.x, q.x - BOW), q.y - DEPTH));
        float bx = sdBox(p - 0.5 * (lo + hi), 0.5 * (hi - lo));
        float d = max(imp / LIPSCHITZ, bx);
        return max(d, -sdRoundBox(p - COCKPIT_C, COCKPIT_H, 0.05));
    }

    float sdFoil(float3 p, float x, float chord, float thick, float y0, float y1) {
        float rr = 0.5 * thick;
        float dx = max(abs(p.x - x) - (0.5 * chord - rr), 0.0);
        float d2 = length(float2(dx, p.z)) - rr;
        float2 w = float2(d2, abs(p.y - 0.5 * (y0 + y1)) - 0.5 * (y1 - y0));
        return min(max(w.x, w.y), 0.0) + length(max(w, 0.0));
    }

    // Materials: 1 hull, 2 mast and boom, 3 sail, 4 centreboard, 5 rudder,
    // 6 rudder head and tiller.
    struct Hit { float d; int mat; };
    void take(thread Hit &h, float d, int m) { if (d < h.d) { h.d = d; h.mat = m; } }

    float boardDist(float3 q) {
        return HAS_BOARD ? sdFoil(q, BOARD_X, BOARD_CHORD, BOARD_THICK, BOARD_TIP, BOARD_TOP) : 1e9;
    }
    float rudderDist(float3 q) { return sdFoil(q, RUDDER_X, RUDDER_CHORD, RUDDER_THICK, RUDDER_BOTTOM, RUDDER_TOP); }

    // The boat in its own frame at unit scale.
    Hit boatUnit(float3 q) {
        Hit h; h.d = 1e9; h.mat = 0;
        take(h, hullDist(q), 1);
        take(h, min(sdVCyl(q, MAST_X, MAST_R0, MAST_FOOT, MAST_FOOT + MAST_LOWER),
                    sdVCyl(q, MAST_X, MAST_R1, MAST_FOOT + MAST_LOWER - 0.01, MAST_FOOT + MAST_LEN)), 2);
        take(h, sdCapsule(q, TACK, BOOM_END, MAST_R1), 2);
        take(h, udTriangle(q, TACK, HEAD, CLEW) - 0.0015, 3);
        take(h, boardDist(q), 4);
        take(h, rudderDist(q), 5);
        // The rudder head and the tiller, over the aft deck. MODEL.
        take(h, sdRoundBox(q - float3(RUDDER_X - 0.01, RUDDER_TOP - 0.06, 0.0), float3(0.13, 0.09, 0.03), 0.01), 6);
        take(h, sdCapsule(q, float3(RUDDER_X + 0.08, RUDDER_TOP - 0.02, 0.0), float3(STERN + 1.05, DEPTH + 0.16, 0.0), 0.016), 6);
        return h;
    }

    float3 toBoat(float3 p) { return (p + float3(0.0, DRAUGHT, 0.0)) / SCALE; }

    Hit boat(float3 p) {
        Hit h = boatUnit(toBoat(p));
        h.d *= SCALE;
        return h;
    }

    float boatD(float3 p) { return boat(p).d; }

    bool march(float3 ro, float3 rd, float tMax, thread float &t, thread int &mat) {
        t = 0.0;
        for (int i = 0; i < 500; i++) {
            float3 p = ro + rd * t;
            Hit h = boat(p);
            float eps = 0.0002 + t * 0.00004;
            if (h.d < eps) { mat = h.mat; return true; }
            t += h.d * STEP_SCALE;
            if (t > tMax) break;
        }
        return false;
    }

    float3 boatNormal(float3 p) {
        const float e = 0.0008;
        float3 k0 = float3(1, -1, -1), k1 = float3(-1, -1, 1), k2 = float3(-1, 1, -1), k3 = float3(1, 1, 1);
        return normalize(k0 * boatD(p + k0 * e) + k1 * boatD(p + k1 * e) + k2 * boatD(p + k2 * e) + k3 * boatD(p + k3 * e));
    }

    // ---------------------------------------------------------------- water

    // The ray's interval inside the water block (slab method).
    bool waterSpan(float3 ro, float3 rd, thread float &t0, thread float &t1, thread int &face) {
        float3 inv = 1.0 / rd;
        float3 a = (W_LO - ro) * inv, b = (W_HI - ro) * inv;
        float3 tmin = min(a, b), tmax = max(a, b);
        t0 = max(max(tmin.x, tmin.y), max(tmin.z, 0.0));
        t1 = min(tmax.x, min(tmax.y, tmax.z));
        if (t1 <= t0) return false;
        face = 0;
        if (tmin.y >= tmin.x && tmin.y >= tmin.z) face = 1;          // through the surface
        else if (tmin.z >= tmin.x) face = 2;                          // through the cut face
        return true;
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
        float L = r.y < 0.0 ? TENT_H * 0.85 : mix(TENT_H, TENT_Z, sqrt(r.y));
        return float3(L);
    }

    // What the camera sees behind everything: a pale, even studio sweep
    // (the lights themselves are out of frame). MODEL.
    float3 backdrop(float3 r) {
        return float3(mix(1.45, 2.1, clamp(0.5 + 1.5 * r.y, 0.0, 1.0)));
    }

    float softShadow(float3 ro, float3 rd, float k) {
        float res = 1.0;
        float t = 0.004;
        for (int i = 0; i < 120; i++) {
            float h = boatD(ro + rd * t);
            res = min(res, k * h / t);
            if (res < 0.002) return 0.0;
            t += clamp(h, 0.003, 0.4);
            if (t > 12.0) break;
        }
        return clamp(res, 0.0, 1.0);
    }

    float occlusion(float3 p, float3 nrm, float scale) {
        float occ = 0.0;
        float w = 1.0;
        for (int i = 1; i <= 5; i++) {
            float h = scale * float(i);
            float d = boatD(p + nrm * h);
            occ += (h - clamp(d, 0.0, h)) * w;
            w *= 0.6;
        }
        return clamp(1.0 - occ * 0.4 / scale, 0.0, 1.0);
    }

    // Light at a point, dimmed by the water above it on the way down (MODEL:
    // straight down, the mean of the three channels).
    float irradiance(float3 p, float3 nrm, float ao) {
        float3 o = p + nrm * 0.002;
        float e = 0.0;
        float c = dot(nrm, KEY_C);
        if (c > 0.0) e += KEY_L * 0.7 * KEY_OMEGA * c * softShadow(o, KEY_C, 1.0 / KEY_TB);
        c = dot(nrm, FILL_C);
        if (c > 0.0) e += FILL_L * 0.7 * FILL_OMEGA * c * softShadow(o, FILL_C, 1.0 / FILL_TB);
        c = dot(nrm, RIM_C);
        if (c > 0.0) e += RIM_L * 0.7 * RIM_OMEGA * c * softShadow(o, RIM_C, 1.0 / RIM_TB);
        float up = 0.5 + 0.5 * nrm.y;
        float ground = 0.35 * AMBIENT_E;
        e += mix(ground, AMBIENT_E, up) * ao;
        if (p.y < 0.0) e *= exp(-dot(W_SIGMA, float3(1.0 / 3.0)) * (-p.y));
        return e;
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

    // Heitz (2018), "Sampling the GGX distribution of visible normals",
    // JCGT 7(4), as steps 60 and 72 use it.
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

    float hash(uint x) {
        x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16;
        return float(x & 0xffffffU) / 16777216.0;
    }

    float3 shadeGloss(float3 p, float3 rd, float3 nrm, float3 alb, float alpha, float u1, float u2) {
        float3 v = -rd;
        float cv = max(dot(nrm, v), 1e-4);
        float ao = occlusion(p, nrm, 0.02);
        float e = irradiance(p, nrm, ao);
        float3 diffuse = alb / M_PI_F * e * (1.0 - fresnelDiel(N_GELCOAT, cv));
        float3 t = normalize(abs(nrm.y) < 0.9 ? cross(nrm, float3(0, 1, 0)) : cross(nrm, float3(1, 0, 0)));
        float3 b = cross(nrm, t);
        float3 ve = float3(dot(v, t), dot(v, b), cv);
        float3 me = sampleVNDF(ve, alpha, u1, u2);
        float3 m = normalize(t * me.x + b * me.y + nrm * me.z);
        float3 l = reflect(rd, m);
        float cl = dot(nrm, l);
        float spec = 0.0;
        if (cl > 0.0) {
            float a2 = alpha * alpha;
            float g1 = 2.0 * cl / (cl + sqrt(a2 + (1.0 - a2) * cl * cl));
            float Li;
            float tt;
            int mt;
            if (march(p + nrm * 0.002, l, 10.0, tt, mt)) Li = 0.25 * AMBIENT_E / M_PI_F;
            else Li = environment(l).x;
            if (p.y < 0.0) Li *= 0.3;
            spec = fresnelDiel(N_GELCOAT, dot(v, m)) * g1 * Li;
        }
        return diffuse + float3(spec);
    }

    float3 shadeMatte(float3 p, float3 nrm, float3 alb) {
        float ao = occlusion(p, nrm, 0.02);
        return alb / M_PI_F * irradiance(p, nrm, ao);
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

    float3 shadeHit(float3 p, float3 rd, int mat, float u1, float u2, float lineW) {
        float3 nrm = boatNormal(p);
        float3 c;
        if (mat == 1) {
            float3 q = toBoat(p);
            bool deck = q.y > DEPTH - 0.004 && nrm.y > 0.7;
            c = shadeGloss(p, rd, nrm, deck ? DECK_ALB : HULL_ALB, 0.22, u1, u2);
        } else if (mat == 2) c = shadeGloss(p, rd, nrm, SPAR_ALB, 0.35, u1, u2);
        else if (mat == 3) {
            // Dacron: matte, lit from both faces a little (thin cloth). MODEL.
            float3 n2 = dot(nrm, rd) > 0.0 ? -nrm : nrm;
            c = shadeMatte(p, n2, SAIL_ALB) + 0.25 * shadeMatte(p, -n2, SAIL_ALB);
        } else if (mat == 4 || mat == 5) c = shadeGloss(p, rd, nrm, FOIL_ALB, 0.25, u1, u2);
        else c = shadeGloss(p, rd, nrm, FITTING_ALB, 0.3, u1, u2);
        // The waterline Archimedes gives, drawn on the boat.
        if ((mat == 1 || mat >= 4) && abs(p.y) < lineW) c = float3(0.02, 0.10, 0.30);
        return c;
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
        float pixAngle = 2.0 * P.camFwd.w / float(P.height);
        for (uint sy = 0; sy < S; sy++) {
            for (uint sx = 0; sx < S; sx++) {
                uint idx = (y * P.width + x) * 64u + sy * S + sx;
                float2 jitter = float2((float(sx) + 0.5) / float(S), (float(sy) + 0.5) / float(S));
                float3 rd = cameraRay(float2(x, y) + jitter, P);
                float u1 = hash(idx * 2u + 1u), u2 = hash(idx * 2u + 2u);
                float t;
                int mat;
                bool hit = march(ro, rd, FAR, t, mat);
                float3 base = hit ? shadeHit(ro + rd * t, rd, mat, u1, u2, 1.1 * pixAngle * t) : backdrop(rd);
                float t0, t1;
                int face;
                float3 c = base;
                if (waterSpan(ro, rd, t0, t1, face) && (!hit || t > t0)) {
                    float path = (hit ? min(t, t1) : t1) - t0;
                    float3 tr = exp(-W_SIGMA * path);
                    float3 inside = base * tr + W_GLOW * (1.0 - tr);
                    if (face == 1) {
                        float R = fresnelDiel(N_WATER, -rd.y);
                        c = R * environment(reflect(rd, float3(0, 1, 0))) + (1.0 - R) * inside;
                    } else c = inside;
                }
                sum += toneMap(c);
            }
        }
        float3 c = sum / float(S * S);
        pixels[y * P.width + x] = uchar4(uchar3(round(clamp(c, 0.0, 1.0) * 255.0)), 255);
        float3 rdc = cameraRay(float2(x, y) + 0.5, P);
        float t;
        int mat = 0;
        float4 a0 = float4(0.0);
        if (march(ro, rdc, FAR, t, mat)) {
            float3 p = ro + rdc * t;
            a0 = float4(float(mat), t, p.y, 0.0);
        }
        float t0, t1;
        int face;
        if (waterSpan(ro, rdc, t0, t1, face) && (a0.x == 0.0 || t > t0)) a0.w = float(face + 1);
        aux[y * P.width + x] = a0;
    }

    kernel void probe(device const float4 *points [[buffer(0)]],
                      device float4 *out [[buffer(1)]],
                      constant uint &count [[buffer(2)]],
                      uint id [[thread_position_in_grid]]) {
        if (id >= count) return;
        float3 p = points[id].xyz;
        Hit h = boat(p);
        float3 q = toBoat(p);
        float under = min(min(hullDist(q), rudderDist(q)), boardDist(q)) * SCALE;
        out[id] = float4(h.d, float(h.mat), under, hullDist(q) * SCALE);
    }
    """
}

// MARK: - running it

enum BoatError: Error, CustomStringConvertible {
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
    var camPos: SIMD4<Float>
    var camFwd: SIMD4<Float>
    var camRight: SIMD4<Float>
    var camUp: SIMD4<Float>
}

struct BoatImage {
    let width: Int
    let height: Int
    let pixels: MTLBuffer
    let aux: MTLBuffer

    func rgba(_ x: Int, _ y: Int) -> SIMD4<UInt8> {
        let p = pixels.contents().advanced(by: (y * width + x) * 4).assumingMemoryBound(to: UInt8.self)
        return SIMD4<UInt8>(p[0], p[1], p[2], p[3])
    }
    /// (material, ray length, world height of the hit, water: 0 none, 1 side, 2 surface, 3 cut face).
    func seen(_ x: Int, _ y: Int) -> SIMD4<Float> {
        aux.contents().assumingMemoryBound(to: SIMD4<Float>.self)[y * width + x]
    }
}

func findDevice() throws -> MTLDevice {
    if let d = MTLCreateSystemDefaultDevice() ?? MTLCopyAllDevices().first { return d }
    throw BoatError.noMetalDevice
}

final class BoatRenderer {
    let device: MTLDevice
    let queue: MTLCommandQueue
    let renderPSO: MTLComputePipelineState
    let probePSO: MTLComputePipelineState
    let width: Int
    let height: Int
    let image: BoatImage

    init(device: MTLDevice, width: Int, height: Int, setup: StillSetup, draught: Double, mutant: Mutant = activeMutant) throws {
        self.device = device
        self.width = width
        self.height = height
        let options = MTLCompileOptions()
        // Precise maths, as in every step.
        options.fastMathEnabled = false
        let library: MTLLibrary
        do {
            let src: String = kernelSource(studio: setup.studio, water: setup.water, draught: draught, mutant: mutant)
            library = try device.makeLibrary(source: src, options: options)
        } catch {
            throw BoatError.kernelCompile("\(error)")
        }
        func pipeline(_ name: String) throws -> MTLComputePipelineState {
            guard let fn = library.makeFunction(name: name) else { throw BoatError.kernelCompile("no kernel \(name)") }
            do { return try device.makeComputePipelineState(function: fn) } catch { throw BoatError.kernelCompile("\(error)") }
        }
        renderPSO = try pipeline("render")
        probePSO = try pipeline("probe")
        guard let q = device.makeCommandQueue(),
              let pixels = device.makeBuffer(length: width * height * 4, options: .storageModeShared),
              let aux = device.makeBuffer(length: width * height * 16, options: .storageModeShared)
        else { throw BoatError.gpu("could not allocate buffers") }
        queue = q
        image = BoatImage(width: width, height: height, pixels: pixels, aux: aux)
    }

    func render(camera: Camera, samples: Int) throws -> Double {
        let band: Int = 16
        var gpu: Double = 0
        var row: Int = 0
        while row < height {
            let rows: Int = min(band, height - row)
            guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else {
                throw BoatError.gpu("could not make a command buffer")
            }
            var params = Params(width: UInt32(width), height: UInt32(height), rowOffset: UInt32(row), samples: UInt32(samples),
                                camPos: SIMD4<Float>(camera.position, 0),
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
            if let e = cb.error { throw BoatError.gpu(e.localizedDescription) }
            gpu += cb.gpuEndTime - cb.gpuStartTime
            row += rows
        }
        return gpu
    }

    /// (boat distance, material, underwater-body distance, hull distance) at each world point.
    func probe(_ points: [SIMD3<Float>]) throws -> [SIMD4<Float>] {
        var out: [SIMD4<Float>] = []
        out.reserveCapacity(points.count)
        let chunk: Int = 1 << 20
        var start: Int = 0
        while start < points.count {
            let end: Int = min(start + chunk, points.count)
            var pts: [SIMD4<Float>] = points[start..<end].map { SIMD4<Float>($0, 0) }
            var count = UInt32(pts.count)
            guard let inb = device.makeBuffer(bytes: &pts, length: 16 * max(pts.count, 1), options: .storageModeShared),
                  let outb = device.makeBuffer(length: 16 * max(pts.count, 1), options: .storageModeShared),
                  let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder()
            else { throw BoatError.gpu("could not set up the probe") }
            enc.setComputePipelineState(probePSO)
            enc.setBuffer(inb, offset: 0, index: 0)
            enc.setBuffer(outb, offset: 0, index: 1)
            enc.setBytes(&count, length: 4, index: 2)
            enc.dispatchThreads(MTLSize(width: pts.count, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: min(probePSO.maxTotalThreadsPerThreadgroup, 256), height: 1, depth: 1))
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            if let e = cb.error { throw BoatError.gpu(e.localizedDescription) }
            let o = outb.contents().assumingMemoryBound(to: SIMD4<Float>.self)
            for i in 0..<pts.count { out.append(o[i]) }
            start = end
        }
        return out
    }
}

func savePNG(_ image: BoatImage, to url: URL) throws {
    try? FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    guard let provider = CGDataProvider(dataInfo: nil, data: image.pixels.contents(), size: image.width * image.height * 4,
                                        releaseData: { _, _, _ in }),
          let space = CGColorSpace(name: CGColorSpace.sRGB),
          let cg = CGImage(width: image.width, height: image.height, bitsPerComponent: 8, bitsPerPixel: 32,
                           bytesPerRow: image.width * 4, space: space,
                           bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                           provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent),
          let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)
    else { throw BoatError.png("could not set up the encoder for \(url.path)") }
    CGImageDestinationAddImage(dest, cg, nil)
    guard CGImageDestinationFinalize(dest) else { throw BoatError.png("could not write \(url.path)") }
}
