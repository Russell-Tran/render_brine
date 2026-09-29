// The boat, rendered: one GPU thread per pixel marching rays into the
// boat's distance function, as every step since the ocean has, over a flat
// sea at y = 0.
//
// The distance function is built from shapes whose distance is exact or a
// lower bound — rounded rectangles, planes, boxes, capsules, a torus — and
// the hull's sections, whose 2D distance is exact across the boat but whose
// parameters (beam, bottom, sheer, the ellipse's centre and depth) change
// along it. The hull's
// part is divided by one constant for the whole hull, √(1 + M²) from
// HullShape.lipschitz, never by a local slope. Combined by min (union), max
// (intersection) and max(a, −b) (a cut), each of which keeps a lower bound
// a lower bound. No blends. The tests check it.
//
// The sea is a plane, not a shape: a ray reaching it is split by water's
// Fresnel (n = 1.333, [HQ]) into a reflection of the studio (and of the
// boat) and a ray that goes on, straight — no refraction, a diagram's
// choice (MODEL) so the keel stands where it is — through water that dims
// it (Beer's law, MODEL coefficients) towards the deep sea's colour.
//
// The wood is varnished: a diffuse body under a GGX clear coat of index
// 1.5 (MODEL), step 60's visible-normal sampling, copied via step 72.

import CoreGraphics
import Foundation
import ImageIO
import Metal
import UniformTypeIdentifiers
import simd

let stepScale: Float = 0.9

func metal(_ v: SIMD3<Float>) -> String { "float3(\(v.x), \(v.y), \(v.z))" }

func kernelSource(studio: Studio, layout: Layout, draft: Float, mutant: Mutant) -> String {
    let k: Softbox = studio.key
    let f: Softbox = studio.fill
    let r: Softbox = studio.rim
    let h: HullShape = layout.hull
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
    // Seats: the two thwarts and the stern seat, as (centre x, half depth,
    // top, thickness).
    var seats: [SIMD4<Float>] = layout.stations.map { st in
        SIMD4<Float>(st.seatCentreX, st.seatDepth / 2, st.seatTop, st.seatThickness)
    }
    let sternHalf: Float = 150 * h.scale
    seats.append(SIMD4<Float>(layout.sternSeatX, sternHalf, layout.sternSeatTop, 22 * h.scale))
    var seatCode: String = ""
    for s in seats {
        seatCode += "    take(hh, thwart(p, open, \(s.x), \(s.y), \(s.z), \(s.w)), 4);\n"
    }
    var blockCode: String = ""
    var lockCode: String = ""
    for l in layout.oarlocks {
        let half: Float = (blockHeight + 10) / 2
        let c: SIMD3<Float> = l.pin - SIMD3<Float>(0, half, 0)
        blockCode += "    take(hh, sdBox(p - \(metal(c)), float3(\(blockHalfLength), \(half), \(blockHalfWidth))), 3);\n"
        if mutant != .noOarlocks {
            lockCode += "    take(hh, oarlock(p, \(metal(l.pin)), \(metal(l.pivot)), \(metal(l.axis)), \(l.ringRadius), \(l.tube)), 6);\n"
        }
    }
    var oarCode: String = ""
    for o in layout.oars {
        oarCode += "    take(hh, oar(p, \(metal(o.pivot)), \(metal(o.axis)), \(metal(o.across)), \(metal(o.normal)), "
            + "\(o.inboard), \(o.outboard), \(bladeWidth(forLength: o.length) / 2)), 5);\n"
    }
    // The boat's bounds in the world, for skipping rays that miss it.
    let reach: Float = layout.oars.map { $0.outboard }.max() ?? 0
    let lo = SIMD3<Float>(-h.halfLength - reach, -draft - 5, -h.halfBeam - reach)
    let hi = SIMD3<Float>(h.halfLength + 10, h.bowDepth - draft + 400, h.halfBeam + reach)

    return """
    #include <metal_stdlib>
    using namespace metal;

    struct Params { uint width; uint height; uint rowOffset; uint samples;
                    float4 camPos; float4 camFwd; float4 camRight; float4 camUp; };

    constant float STEP_SCALE = \(stepScale);
    constant float FAR = 60000.0;
    constant float TONE_GAIN = \(toneGain);
    constant float DRAFT = \(draft);

    constant float HL = \(h.halfLength);
    constant float HB = \(h.halfBeam);
    constant float D0 = \(h.centerDepth);
    constant float DB = \(h.bowDepth);
    constant float DS = \(h.sternDepth);
    constant float XM = \(h.maxBeamX);
    constant float TR = \(h.transomRatio);
    constant float KD = \(h.keelDepth);
    constant float KW = \(h.keelHalfWidth);
    constant float X1 = \(h.bowRiseStart);
    constant float KB = \(h.bowFoot);
    constant float EB = \(h.bowRisePower);
    constant float X2 = \(h.sternRiseStart);
    constant float KT = \(h.transomFoot);
    constant float PB = \(h.bowBeamPower);
    constant float ET = \(h.sternRisePower);
    constant float TOPSIDES = \(h.topsides);
    constant float SKIN = \(h.skin);
    constant float LIP = \(h.lipschitz);
    constant float3 BOX_LO = \(metal(lo));
    constant float3 BOX_HI = \(metal(hi));

    constant float N_WATER = \(waterIndex);
    constant float N_VARNISH = \(varnishIndex);
    constant float VARNISH_A = \(varnishRoughness);
    constant float3 CEDAR_ALB = \(metal(cedarAlbedo));
    constant float3 MAHOGANY_ALB = \(metal(mahoganyAlbedo));
    constant float3 SEAT_ALB = \(metal(seatAlbedo));
    constant float3 SPRUCE_ALB = \(metal(spruceAlbedo));
    constant float3 BRONZE_F0 = \(metal(bronzeF0));
    constant float3 DEEP = \(metal(deepWater));
    constant float3 SIGMA = \(metal(waterExtinction));

    \(box("KEY", k))
    \(box("FILL", f))
    \(box("RIM", r))
    constant float TENT_H = \(studio.tentHorizon);
    constant float TENT_Z = \(studio.tentZenith);
    constant float AMBIENT_E = \(studio.ambientIrradiance);

    // ---------------------------------------------------------------- the hull's lines (Boat.swift, mirrored)

    float sheerH(float x) {
        if (x >= 0.0) { float u = min(x / HL, 1.0); return D0 + (DB - D0) * u * u; }
        float v = min(-x / HL, 1.0);
        return D0 + (DS - D0) * v * v;
    }

    float beamH(float x) {
        if (x >= XM) { float u = min((x - XM) / (HL - XM), 1.0); return HB * (1.0 - pow(u, PB)); }
        float v = min((XM - x) / (XM + HL), 1.0);
        return HB * (1.0 - (1.0 - TR) * v * v);
    }

    float bottomH(float x) {
        float u = clamp((x - X1) / (HL - X1), 0.0, 1.0);
        float v = clamp((X2 - x) / (X2 + HL), 0.0, 1.0);
        return KD + (KB - KD) * pow(u, EB) + (KT - KD) * pow(v, ET);
    }

    float centreH(float x) { return max(sheerH(x) - TOPSIDES, bottomH(x) + 1.0); }

    float keelBottomH(float x) { return x > 0.0 ? bottomH(x) - KD : 0.0; }

    // ---------------------------------------------------------------- shapes

    float sdBox(float3 p, float3 b) {
        float3 q = abs(p) - b;
        return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
    }

    float sdRoundBox(float3 p, float3 b, float r) {
        float3 q = abs(p) - b + r;
        return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0) - r;
    }

    float sdCapsule(float3 p, float3 a, float3 b, float r) {
        float3 pa = p - a, ba = b - a;
        float hh = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
        return length(pa - ba * hh) - r;
    }

    // Distance from y (first quadrant) to the ellipse with semi-axes
    // e0 ≥ e1: D. Eberly, "Distance from a Point to an Ellipse, an
    // Ellipsoid, or a Hyperellipsoid" (Geometric Tools, 2020), whose
    // bisection is robust inside and out — here on u = s + 1 rather than
    // s, so that float32 keeps its precision where the root is near s = −1
    // (inside, close to the major axis; checked against a dense search to
    // 0.0002 mm, and by the tests).
    float ellipseDist(float2 y, float e0, float e1) {
        if (y.y > 0.0) {
            if (y.x > 0.0) {
                float z0 = y.x / e0, z1 = y.y / e1;
                float g = z0 * z0 + z1 * z1 - 1.0;
                if (g == 0.0) return 0.0;
                float r0 = (e0 / e1) * (e0 / e1);
                float r1 = r0 - 1.0;
                float n0 = r0 * z0;
                float u0 = z1;
                float u1 = g < 0.0 ? 1.0 : length(float2(n0, z1));
                float u = u0;
                for (int i = 0; i < 80; i++) {
                    u = 0.5 * (u0 + u1);
                    if (u == u0 || u == u1) break;
                    float ra = n0 / (u + r1), rb = z1 / u;
                    float gg = ra * ra + rb * rb - 1.0;
                    if (gg > 0.0) u0 = u; else if (gg < 0.0) u1 = u; else break;
                }
                float x0 = r0 * y.x / (u + r1), x1 = y.y / u;
                return length(float2(x0 - y.x, x1 - y.y));
            }
            return abs(y.y - e1);
        }
        float numer0 = e0 * y.x, denom0 = e0 * e0 - e1 * e1;
        if (numer0 < denom0) {
            float xde0 = numer0 / denom0;
            float x0 = e0 * xde0, x1 = e1 * sqrt(max(1.0 - xde0 * xde0, 0.0));
            return length(float2(x0 - y.x, x1));
        }
        return abs(y.x - e0);
    }

    // Signed distance to the ellipse with semi-axes ab (along x, along y).
    float sdEllipse(float2 p, float2 ab) {
        float2 q = abs(p);
        float d = ab.x >= ab.y ? ellipseDist(q, ab.x, ab.y) : ellipseDist(q.yx, ab.y, ab.x);
        float2 u = q / ab;
        return dot(u, u) > 1.0 ? d : -d;
    }

    // The hull body's section at p.x: `open` without the sheer's cap (for
    // the cavity), `closed` with it (the outside). Unscaled by LIP. The
    // section: an ellipse centred c (half-beam b across, a down to the
    // bottom) joined to straight sides from c up.
    void hullBody(float3 p, thread float &open, thread float &closed) {
        float x = p.x;
        float b = max(beamH(x), 1.0), k = bottomH(x), s = sheerH(x), c = centreH(x);
        float a = c - k;
        float2 q = float2(p.z, p.y - c);
        float ell = sdEllipse(q, float2(b, a));
        float sides = sdBox(float3(q.x, q.y - 150.0, 0.0), float3(b, 150.0, 1.0));
        float d = min(ell, sides);
        d = max(d, max(-HL - x, x - HL));
        open = d;
        closed = max(d, p.y - s);
    }

    float keelD(float3 p) {
        float k = bottomH(p.x);
        float d = max(abs(p.z) - KW, max(keelBottomH(p.x) - p.y, p.y - (k + 20.0)));
        return max(d, max(-HL - p.x, p.x - HL));
    }

    float stemD(float3 p) {
        float d = max(abs(p.z) - KW, max(p.x - HL, (HL - 45.0) - p.x));
        return max(d, max((KB - KD) - p.y, p.y - DB));
    }

    // The outer envelope (what displaces water): body, keel, stem.
    float envelopeH(float3 p) {
        float open, closed;
        hullBody(p, open, closed);
        return min(closed, min(keelD(p), stemD(p))) / LIP;
    }

    float thwart(float3 p, float open, float cx, float hd, float top, float th) {
        float d = max(abs(p.x - cx) - hd, abs(p.y - (top - 0.5 * th)) - 0.5 * th);
        return max(d, open + 0.5 * SKIN) / LIP;
    }

    // A bronze oarlock: a pin down into its block, and a U whose ring
    // (radius R, rod t) is centred on the oar's pivot, square to `ax`.
    float oarlock(float3 p, float3 pin, float3 pivot, float3 ax, float R, float t) {
        float3 w = p - pivot;
        float along = dot(w, ax);
        float3 perp = w - ax * along;
        float ring = length(float2(length(perp) - R, along)) - t;
        ring = max(ring, w.y);
        float3 u = normalize(cross(float3(0.0, 1.0, 0.0), ax));
        float arms = min(sdCapsule(p, pivot + u * R, pivot + u * R + float3(0.0, 22.0, 0.0), t),
                         sdCapsule(p, pivot - u * R, pivot - u * R + float3(0.0, 22.0, 0.0), t));
        float stem = sdCapsule(p, pin - float3(0.0, 18.0, 0.0), pivot - float3(0.0, R, 0.0), 6.0);
        return min(min(ring, arms), stem);
    }

    // An oar in its own frame: s along the axis from the pivot, the blade
    // flat in the (axis, across) plane.
    float oar(float3 p, float3 pivot, float3 a, float3 c, float3 n, float inb, float outb, float bw) {
        float3 w = p - pivot;
        float3 q = float3(dot(w, a), dot(w, c), dot(w, n));
        float grip = sdCapsule(q, float3(-inb + \(gripRadius), 0.0, 0.0), float3(-inb + \(gripLength), 0.0, 0.0), \(gripRadius));
        float loom = sdCapsule(q, float3(-inb + \(gripLength), 0.0, 0.0), float3(outb - \(bladeLength) + 40.0, 0.0, 0.0), \(loomRadius));
        float blade = sdRoundBox(q - float3(outb - \(bladeLength / 2), 0.0, 0.0),
                                 float3(\(bladeLength / 2), bw, \(bladeThickness / 2)), 5.0);
        return min(min(grip, loom), blade);
    }

    // Materials: 2 hull (cedar), 3 mahogany (inwale, keel, stem, oar
    // blocks), 4 seats, 5 oars, 6 bronze oarlocks.
    struct Hit { float d; int mat; };

    void take(thread Hit &hh, float d, int m) { if (d < hh.d) { hh.d = d; hh.mat = m; } }

    // The boat in the world (the hull frame lowered by the draft): first
    // everything but the oars, then the oars.
    Hit fixedParts(float3 pw) {
        float3 p = pw + float3(0.0, DRAFT, 0.0);
        Hit hh; hh.d = 1e9; hh.mat = 0;
        float open, closed;
        hullBody(p, open, closed);
        float s = sheerH(p.x);
        float shell = max(closed, -(open + SKIN));
        take(hh, shell / LIP, 2);
        float inwale = max(abs(open + SKIN + 14.0) - 14.0, max(p.y - s, (s - 24.0) - p.y));
        take(hh, min(inwale, min(keelD(p), stemD(p))) / LIP, 3);
    \(seatCode)\(blockCode)\(lockCode)    return hh;
    }

    Hit boat(float3 pw) {
        Hit hh = fixedParts(pw);
        float3 p = pw + float3(0.0, DRAFT, 0.0);
    \(oarCode)    return hh;
    }

    float boatD(float3 pw) { return boat(pw).d; }

    bool boxRange(float3 ro, float3 rd, thread float &t0, thread float &t1) {
        float3 inv = 1.0 / rd;
        float3 a = (BOX_LO - ro) * inv, b = (BOX_HI - ro) * inv;
        float3 lo = min(a, b), hi = max(a, b);
        t0 = max(max(lo.x, lo.y), max(lo.z, 0.0));
        t1 = min(min(hi.x, hi.y), hi.z);
        return t1 > t0;
    }

    bool march(float3 ro, float3 rd, float tMax, thread float &t, thread int &mat) {
        float t0, t1;
        if (!boxRange(ro, rd, t0, t1)) return false;
        t1 = min(t1, tMax);
        t = t0;
        for (int i = 0; i < 500; i++) {
            if (t > t1) break;
            float3 p = ro + rd * t;
            Hit hh = boat(p);
            float eps = 0.02 + t * 0.00006;
            if (hh.d < eps) { mat = hh.mat; return true; }
            t += hh.d * STEP_SCALE;
        }
        return false;
    }

    float3 boatNormal(float3 p) {
        const float e = 0.05;
        float3 k0 = float3(1, -1, -1), k1 = float3(-1, -1, 1), k2 = float3(-1, 1, -1), k3 = float3(1, 1, 1);
        return normalize(k0 * boatD(p + k0 * e) + k1 * boatD(p + k1 * e) + k2 * boatD(p + k2 * e) + k3 * boatD(p + k3 * e));
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

    float softShadow(float3 ro, float3 rd, float k) {
        float res = 1.0;
        float t = 1.0;
        for (int i = 0; i < 100; i++) {
            float h = boatD(ro + rd * t);
            res = min(res, k * h / t);
            if (res < 0.002) return 0.0;
            t += clamp(h, 0.5, 60.0);
            if (t > 4000.0) break;
        }
        return clamp(res, 0.0, 1.0);
    }

    float occlusion(float3 p, float3 nrm, float scale) {
        float occ = 0.0;
        float w = 1.0;
        for (int i = 1; i <= 5; i++) {
            float h = scale * float(i);
            // The distance is a lower bound (the hull's is divided by LIP);
            // for this shading estimate only, undo that. MODEL.
            float d = boatD(p + nrm * h) * LIP;
            occ += (h - clamp(d, 0.0, h)) * w;
            w *= 0.6;
        }
        return clamp(1.0 - occ * 0.4 / scale, 0.0, 1.0);
    }

    float irradiance(float3 p, float3 nrm, float ao) {
        float3 o = p + nrm * 0.5;
        float e = 0.0;
        float c = dot(nrm, KEY_C);
        if (c > 0.0) e += KEY_L * 0.7 * KEY_OMEGA * c * softShadow(o, KEY_C, 1.0 / KEY_TB);
        c = dot(nrm, FILL_C);
        if (c > 0.0) e += FILL_L * 0.7 * FILL_OMEGA * c * softShadow(o, FILL_C, 1.0 / FILL_TB);
        c = dot(nrm, RIM_C);
        if (c > 0.0) e += RIM_L * 0.7 * RIM_OMEGA * c * softShadow(o, RIM_C, 1.0 / RIM_TB);
        float up = 0.5 + 0.5 * nrm.y;
        float ground = 0.35 * AMBIENT_E;
        return e + mix(ground, AMBIENT_E, up) * ao;
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
    // JCGT 7(4), as step 60 uses it.
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

    // Varnished wood: diffuse × (1 − F) plus one GGX sample of the studio
    // (what the coat mirrors is the tent, or the sea's colour below the
    // horizon). Bronze: a metal, F0 tinted, the same sample.
    float3 shadeBoat(float3 p, float3 rd, int mat, float u1, float u2) {
        float3 nrm = boatNormal(p);
        float3 v = -rd;
        if (dot(nrm, v) < 0.0) nrm = normalize(nrm + v * (0.01 - dot(nrm, v)));
        float cv = max(dot(nrm, v), 1e-4);
        float ao = occlusion(p, nrm, 8.0);
        float e = irradiance(p, nrm, ao);
        float3 alb = mat == 2 ? CEDAR_ALB : mat == 3 ? MAHOGANY_ALB : mat == 4 ? SEAT_ALB : SPRUCE_ALB;
        bool metal = mat == 6;
        float alpha = metal ? 0.25 : VARNISH_A;
        float3 t = normalize(abs(nrm.y) < 0.9 ? cross(nrm, float3(0, 1, 0)) : cross(nrm, float3(1, 0, 0)));
        float3 b = cross(nrm, t);
        float3 ve = float3(dot(v, t), dot(v, b), cv);
        float3 me = sampleVNDF(ve, alpha, u1, u2);
        float3 m = normalize(t * me.x + b * me.y + nrm * me.z);
        float3 l = reflect(rd, m);
        float cl = dot(nrm, l);
        float3 Li = float3(0.0);
        float g1 = 0.0;
        if (cl > 0.0) {
            float a2 = alpha * alpha;
            g1 = 2.0 * cl / (cl + sqrt(a2 + (1.0 - a2) * cl * cl));
            float tt;
            int mt;
            if (march(p + nrm * 0.5, l, 8000.0, tt, mt)) Li = float3(0.04);
            else if (l.y < 0.0) Li = DEEP * 1.5 + 0.03 * environment(reflect(l, float3(0, 1, 0)));
            else Li = environment(l);
        }
        float vm = dot(v, m);
        if (metal) {
            float3 F = BRONZE_F0 + (1.0 - BRONZE_F0) * pow(1.0 - clamp(vm, 0.0, 1.0), 5.0);
            return F * g1 * Li + BRONZE_F0 * 0.05 / M_PI_F * e;
        }
        float F = fresnelDiel(N_VARNISH, vm);
        float3 diffuse = alb / M_PI_F * e * (1.0 - fresnelDiel(N_VARNISH, cv));
        return diffuse + F * g1 * Li;
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

    // One camera ray: what it sees, and (for the tests) the material and
    // whether it was seen through the water (9 = open water).
    float3 trace(float3 ro, float3 rd, uint idx, thread int &seenMat, thread bool &under) {
        float tEnter = rd.y < 0.0 ? -ro.y / rd.y : FAR;
        float t;
        int mat;
        under = false;
        float u1 = hash(idx * 2u + 1u), u2 = hash(idx * 2u + 2u);
        if (march(ro, rd, tEnter, t, mat)) {
            seenMat = mat;
            return shadeBoat(ro + rd * t, rd, mat, u1, u2);
        }
        if (tEnter >= FAR) { seenMat = 0; return environment(rd); }
        float3 s = ro + rd * tEnter;
        // Inside the hull's envelope the surface is not sea: the boat is dry
        // inside, its floor below the waterline outside. Go on to the floor.
        if (envelopeH(s + float3(0.0, DRAFT, 0.0)) < 0.0) {
            if (march(ro, rd, FAR, t, mat)) {
                seenMat = mat;
                return shadeBoat(ro + rd * t, rd, mat, u1, u2);
            }
        }
        float F = fresnelDiel(N_WATER, -rd.y);
        // Reflection: the boat above the water, or the studio.
        float3 refl;
        float3 rr = reflect(rd, float3(0, 1, 0));
        float tr;
        int mr;
        if (march(s + rr * 0.1, rr, 20000.0, tr, mr)) refl = shadeBoat(s + rr * tr, rr, mr, u1, u2);
        else refl = environment(rr);
        // Transmission, straight on: the hull below the surface, dimmed.
        float3 below;
        float tb;
        int mb;
        seenMat = 9;
        if (march(s + rd * 0.1, rd, 8000.0, tb, mb)) {
            float3 q = s + rd * tb;
            // Light down to the hull through the same water, as far again as
            // it is deep: MODEL.
            float3 att = exp(-SIGMA * (tb - q.y * 1.3));
            below = shadeBoat(q, rd, mb, u1, u2) * att + DEEP * (1.0 - att);
            seenMat = mb;
            under = true;
        } else below = DEEP;
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
        for (uint sy = 0; sy < S; sy++) {
            for (uint sx = 0; sx < S; sx++) {
                uint idx = (y * P.width + x) * 64u + sy * S + sx;
                float2 jitter = float2((float(sx) + 0.5) / float(S), (float(sy) + 0.5) / float(S));
                float3 rd = cameraRay(float2(x, y) + jitter, P);
                int m;
                bool u;
                sum += toneMap(trace(ro, rd, idx, m, u));
            }
        }
        float3 c = sum / float(S * S);
        pixels[y * P.width + x] = uchar4(uchar3(round(clamp(c, 0.0, 1.0) * 255.0)), 255);
        int m = 0;
        bool u = false;
        float3 rdc = cameraRay(float2(x, y) + 0.5, P);
        trace(ro, rdc, 7u, m, u);
        aux[y * P.width + x] = float4(float(m), u ? 1.0 : 0.0, 0.0, 0.0);
    }

    // (boat distance, material, envelope distance, distance to all but the
    // oars) at world points.
    kernel void probe(device const float4 *points [[buffer(0)]],
                      device float4 *out [[buffer(1)]],
                      constant uint &count [[buffer(2)]],
                      uint id [[thread_position_in_grid]]) {
        if (id >= count) return;
        float3 p = points[id].xyz;
        Hit hh = boat(p);
        out[id] = float4(hh.d, float(hh.mat), envelopeH(p + float3(0.0, DRAFT, 0.0)), fixedParts(p).d);
    }

    // The volume below the water, column by column: each thread walks one
    // (x, z) column down from the surface in steps of `dy` (midpoints) and
    // counts the length inside the envelope. The hull frame is lowered by
    // `draft` (not necessarily the drawn one).
    struct Grid { float x0; float z0; float dx; float dz; uint nx; uint nz; float draft; float dy; uint ny; uint pad; };

    kernel void submerged(device float *out [[buffer(0)]],
                          constant Grid &G [[buffer(1)]],
                          uint id [[thread_position_in_grid]]) {
        if (id >= G.nx * G.nz) return;
        uint i = id % G.nx, j = id / G.nx;
        float x = G.x0 + (float(i) + 0.5) * G.dx;
        float z = G.z0 + (float(j) + 0.5) * G.dz;
        float len = 0.0;
        for (uint k = 0; k < G.ny; k++) {
            float yw = -(float(k) + 0.5) * G.dy;
            if (envelopeH(float3(x, yw + G.draft, z)) < 0.0) len += G.dy;
        }
        out[id] = len;
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

struct Grid {
    var x0: Float
    var z0: Float
    var dx: Float
    var dz: Float
    var nx: UInt32
    var nz: UInt32
    var draft: Float
    var dy: Float
    var ny: UInt32
    var pad: UInt32 = 0
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
    /// (material seen by the centre ray, 1 if seen through the water, 0, 0).
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
    let submergedPSO: MTLComputePipelineState
    let width: Int
    let height: Int
    let image: BoatImage

    init(device: MTLDevice, width: Int, height: Int, studio: Studio, layout: Layout, draft: Float,
         mutant: Mutant = activeMutant) throws {
        self.device = device
        self.width = width
        self.height = height
        let options = MTLCompileOptions()
        // Precise maths, as in every step.
        options.fastMathEnabled = false
        let library: MTLLibrary
        do {
            let src: String = kernelSource(studio: studio, layout: layout, draft: draft, mutant: mutant)
            library = try device.makeLibrary(source: src, options: options)
        } catch {
            throw BoatError.kernelCompile("\(error)")
        }
        func pipeline(_ name: String) throws -> MTLComputePipelineState {
            guard let f = library.makeFunction(name: name) else { throw BoatError.kernelCompile("no kernel \(name)") }
            do { return try device.makeComputePipelineState(function: f) } catch { throw BoatError.kernelCompile("\(error)") }
        }
        renderPSO = try pipeline("render")
        probePSO = try pipeline("probe")
        submergedPSO = try pipeline("submerged")
        guard let q = device.makeCommandQueue(),
              let pixels = device.makeBuffer(length: width * height * 4, options: .storageModeShared),
              let aux = device.makeBuffer(length: width * height * 16, options: .storageModeShared)
        else { throw BoatError.gpu("could not allocate buffers") }
        queue = q
        image = BoatImage(width: width, height: height, pixels: pixels, aux: aux)
    }

    func render(camera: Camera, samples: Int) throws -> Double {
        let band: Int = 24
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

    /// (boat distance, material, envelope distance, distance to all but the
    /// oars) at each world point.
    func probe(_ points: [SIMD3<Float>]) throws -> [SIMD4<Float>] {
        var out: [SIMD4<Float>] = []
        out.reserveCapacity(points.count)
        let chunk: Int = 262_144
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

    /// The envelope's volume below the water (m³) with the hull lowered by
    /// `draft`, counted on the GPU from the kernel's own distance function:
    /// columns `cell` mm square, `dy` mm steps.
    func submergedVolume(hull: HullShape, draft: Float, cell: Float = 2, dy: Float = 0.25) throws -> Double {
        let x0: Float = -hull.halfLength - 4
        let z0: Float = -hull.halfBeam - 4
        let nx = UInt32(((hull.length + 8) / cell).rounded(.up))
        let zSpan: Float = 2 * hull.halfBeam + 8
        let nz = UInt32((zSpan / cell).rounded(.up))
        let ny = UInt32(((draft + 2) / dy).rounded(.up))
        var g = Grid(x0: x0, z0: z0, dx: cell, dz: cell, nx: nx, nz: nz, draft: draft, dy: dy, ny: ny)
        let n: Int = Int(nx * nz)
        guard let outb = device.makeBuffer(length: 4 * n, options: .storageModeShared),
              let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder()
        else { throw BoatError.gpu("could not set up the volume count") }
        enc.setComputePipelineState(submergedPSO)
        enc.setBuffer(outb, offset: 0, index: 0)
        enc.setBytes(&g, length: MemoryLayout<Grid>.stride, index: 1)
        enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(submergedPSO.maxTotalThreadsPerThreadgroup, 256), height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        if let e = cb.error { throw BoatError.gpu(e.localizedDescription) }
        let o = outb.contents().assumingMemoryBound(to: Float.self)
        var sum: Double = 0
        for i in 0..<n { sum += Double(o[i]) }
        let area: Double = Double(cell) * Double(cell)
        return sum * area * 1e-9
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
