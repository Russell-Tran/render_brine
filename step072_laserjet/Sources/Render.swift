// The printer, rendered: one GPU thread per pixel marching rays into the
// printer's distance function, as every step since the ocean has.
//
// The distance function is built only from shapes whose distance is exact
// outside or a lower bound (rounded boxes, capsules, cylinders), combined by
// min (union), max (intersection) and max(a, −b) (a cut) — each of which
// keeps a lower bound a lower bound — and, for the mutant's larger printer,
// by uniform scaling, which does too. No blends. The tests check it.
//
// The housing is matte black plastic: a dark diffuse under a dielectric top
// surface with polystyrene's Fresnel (Printer.swift), rough (GGX α 0.42), so
// it shows the studio's softboxes as broad, soft reflections — step 60's
// visible-normal sampling, copied. The lit lights are emitters.

import CoreGraphics
import Foundation
import ImageIO
import Metal
import UniformTypeIdentifiers
import simd

let stepScale: Float = 0.9

func metal(_ v: SIMD3<Float>) -> String { "float3(\(v.x), \(v.y), \(v.z))" }

func kernelSource(studio: Studio, mutant: Mutant) -> String {
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
                    uint trayOpen; uint pad0; uint pad1; uint pad2;
                    float4 camPos; float4 camFwd; float4 camRight; float4 camUp; };

    constant float STEP_SCALE = \(stepScale);
    constant float FAR = 6000.0;
    constant float TONE_GAIN = \(toneGain);
    constant float SCALE = \(printerScale(mutant));
    constant float W2 = \(hpWidth / 2);
    constant float D2 = \(hpDepth / 2);
    constant float H = \(hpHeight);
    constant float N_HOUSING = \(housingIndex);
    constant float3 HOUSING_ALB = \(metal(housingAlbedo));
    constant float HOUSING_A = \(housingRoughness);
    constant float3 BUTTON_ALB = \(metal(buttonAlbedo));
    constant float BUTTON_A = \(buttonRoughness);
    constant float3 BADGE_ALB = \(metal(badgeAlbedo));
    constant float3 PAPER_ALB = \(metal(paperAlbedo));
    constant float3 LED_WIRELESS = \(metal(wirelessLight));
    constant float3 LED_READY = \(metal(readyLight));
    constant float3 LED_ATTENTION = \(metal(attentionOff));

    \(box("KEY", k))
    \(box("FILL", f))
    \(box("RIM", r))
    constant float TENT_H = \(studio.tentHorizon);
    constant float TENT_Z = \(studio.tentZenith);
    constant float AMBIENT_E = \(studio.ambientIrradiance);
    constant float3 TABLE_ALBEDO = \(metal(studio.tableAlbedo));

    // ---------------------------------------------------------------- shapes

    float sdRoundBox(float3 p, float3 b, float r) {
        float3 q = abs(p) - b + r;
        return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0) - r;
    }

    float sdRoundRect(float2 p, float2 b, float r) {
        float2 q = abs(p) - b + r;
        return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - r;
    }

    // A 2D shape's distance d2 extruded over [y0, y1] with its edges rounded
    // by re (a lower bound: see the header).
    float extrudeY(float d2, float y, float y0, float y1, float re) {
        float2 w = float2(d2 + re, abs(y - 0.5 * (y0 + y1)) - 0.5 * (y1 - y0) + re);
        return min(max(w.x, w.y), 0.0) + length(max(w, 0.0)) - re;
    }

    // A cylinder along an axis: centre c, unit axis a, radius r, half-length h.
    float sdCylinder(float3 p, float3 c, float3 a, float r, float h) {
        float3 d = p - c;
        float along = dot(d, a);
        float radial = length(d - a * along);
        float2 w = float2(radial - r, abs(along) - h);
        return min(max(w.x, w.y), 0.0) + length(max(w, 0.0));
    }

    float3 rotX(float3 p, float a) {
        float c = cos(a), s = sin(a);
        return float3(p.x, c * p.y - s * p.z, s * p.y + c * p.z);
    }

    // Materials: 1 table, 2 housing, 3 button, 4 badge, 5 paper,
    // 10 wireless light, 11 attention light, 12 ready light, 13 cancel button.
    struct Hit { float d; int mat; };

    void take(thread Hit &h, float d, int m) { if (d < h.d) { h.d = d; h.mat = m; } }

    // The printer at unit scale (HP's size), trays as `open` says.
    Hit printer(float3 p, bool open) {
        Hit h; h.d = 1e9; h.mat = 0;
        // The body: a box rounded at its upright corners, its top and bottom
        // edges softened.
        float body = extrudeY(sdRoundRect(p.xz, float2(W2, D2), 24.0), p.y, 0.0, H, 7.0);
        // The output bin: a recess in the top whose floor falls towards the
        // back, where the paper comes out.
        float3 qb = rotX(p - float3(0.0, H - 2.0, -16.0), -0.12);
        float bin = sdRoundBox(qb, float3(118.0, 16.0, 74.0), 4.0);
        body = max(body, -bin);
        // The exit slot at the back of the bin.
        body = max(body, -sdRoundBox(p - float3(0.0, H - 24.0, -92.0), float3(106.0, 3.5, 12.0), 1.5));
        // The print-cartridge door's seam: round the front and sides at
        // y = 104, and up the sides to the top at z = −40.
        float shell = -(body + 1.2);
        float seamA = max(max(abs(p.y - 104.0) - 0.6, -(p.z + 40.0)), shell);
        float seamB = max(max(abs(p.z + 40.0) - 0.6, -(p.y - 104.0)), shell);
        body = max(body, -min(seamA, seamB));
        // The door's lift-tab: a finger recess at the top of the front.
        body = max(body, -sdRoundBox(p - float3(0.0, H - 5.0, D2), float3(28.0, 5.0, 7.0), 3.0));
        // The input opening and, above it, the priority input slot.
        body = max(body, -sdRoundBox(p - float3(0.0, 42.0, D2), float3(118.0, 28.0, 30.0), 3.0));
        body = max(body, -sdRoundBox(p - float3(0.0, 80.5, D2), float3(112.0, 2.2, 34.0), 1.0));
        // The control panel's well, sunk into the top at the left front.
        body = max(body, -sdRoundBox(p - float3(-148.0, H, 21.0), float3(9.0, 2.5, 34.0), 2.0));
        take(h, body, 2);

        // The output tray extension at the bin's front edge: flipped up
        // (open), or folded flat into the bin.
        float3 hinge = float3(0.0, H - 11.0, 57.0);
        float3 qf = rotX(p - hinge, open ? 1.05 : -0.12);
        take(h, sdRoundBox(qf - float3(0.0, 1.2, -19.0), float3(46.0, 1.2, 19.0), 1.0), 2);

        // The badge: a plain disc where the logo would be, flush with the
        // front.
        take(h, sdCylinder(p, float3(0.0, 150.0, D2 - 0.6), float3(0.0, 0.0, 1.0), 8.0, 0.6), 4);

        // The control panel, back to front: wireless button, wireless light,
        // attention light, ready light, cancel button [HP-UG], in the well,
        // none standing above the top.
        take(h, sdRoundBox(p - float3(-148.0, H - 1.5, -4.0), float3(5.5, 1.3, 5.5), 1.2), 3);
        take(h, sdCylinder(p, float3(-148.0, H - 2.0, 10.0), float3(0.0, 1.0, 0.0), 1.7, 0.8), 10);
        take(h, sdCylinder(p, float3(-148.0, H - 2.0, 20.0), float3(0.0, 1.0, 0.0), 1.7, 0.8), 11);
        take(h, sdCylinder(p, float3(-148.0, H - 2.0, 30.0), float3(0.0, 1.0, 0.0), 1.7, 0.8), 12);
        take(h, sdRoundBox(p - float3(-148.0, H - 1.5, 45.0), float3(5.5, 1.3, 5.5), 1.2), 13);
        // The power button, on the front at the left, flush.
        take(h, sdCylinder(p, float3(-138.0, 40.0, D2 - 1.0), float3(0.0, 0.0, 1.0), 5.0, 1.0), 3);

        if (open) {
            // The main input tray, folded down from the bottom of the
            // opening and resting on its far edge, with its paper guides,
            // ribbed floor and a few sheets of paper.
            float3 th = float3(0.0, 16.0, D2 - 4.0);
            float ang = asin((16.0 - 0.5) / 165.0);
            float3 q = rotX(p - th, -ang);                      // x across, y up from the tray, z along it
            float tray = sdRoundBox(q - float3(0.0, 1.5, 82.5), float3(121.0, 1.5, 82.5), 1.0);
            float rib = abs(fmod(abs(q.x) + 8.0, 16.0) - 8.0) - 1.5;
            tray = max(tray, -max(max(rib, abs(q.y - 3.0) - 0.8), abs(q.z - 90.0) - 60.0));
            float guides = min(sdRoundBox(q - float3(-110.0, 8.0, 65.0), float3(1.5, 6.0, 55.0), 1.0),
                               sdRoundBox(q - float3(110.0, 8.0, 65.0), float3(1.5, 6.0, 55.0), 1.0));
            take(h, min(tray, guides), 2);
            take(h, sdRoundBox(q - float3(0.0, 4.2, 55.0), float3(107.9, 1.2, 80.0), 0.2), 5);
        } else {
            // Closed: the tray stands up, filling the opening.
            take(h, sdRoundBox(p - float3(0.0, 42.0, D2 - 1.6), float3(117.0, 27.0, 1.5), 1.0), 2);
        }
        return h;
    }

    Hit scene(float3 p, bool open) {
        Hit h = printer(p / SCALE, open);
        h.d *= SCALE;
        if (p.y < h.d) { h.d = p.y; h.mat = 1; }
        return h;
    }

    float printerOnly(float3 p, bool open) {
        Hit h = printer(p / SCALE, open);
        return h.d * SCALE;
    }

    bool march(float3 ro, float3 rd, float tMax, bool open, thread float &t, thread int &mat) {
        t = 0.0;
        for (int i = 0; i < 400; i++) {
            float3 p = ro + rd * t;
            Hit h = scene(p, open);
            float eps = 0.004 + t * 0.00002;
            if (h.d < eps) { mat = h.mat; return true; }
            t += h.d * STEP_SCALE;
            if (t > tMax) break;
        }
        return false;
    }

    float3 printerNormal(float3 p, bool open) {
        const float e = 0.01;
        float3 k0 = float3(1, -1, -1), k1 = float3(-1, -1, 1), k2 = float3(-1, 1, -1), k3 = float3(1, 1, 1);
        return normalize(k0 * printerOnly(p + k0 * e, open) + k1 * printerOnly(p + k1 * e, open)
                       + k2 * printerOnly(p + k2 * e, open) + k3 * printerOnly(p + k3 * e, open));
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

    float softShadow(float3 ro, float3 rd, float k, bool open) {
        float res = 1.0;
        float t = 0.1;
        for (int i = 0; i < 90; i++) {
            float h = printerOnly(ro + rd * t, open);
            res = min(res, k * h / t);
            if (res < 0.002) return 0.0;
            t += clamp(h, 0.1, 12.0);
            if (t > 800.0) break;
        }
        return clamp(res, 0.0, 1.0);
    }

    float occlusion(float3 p, float3 nrm, bool open, float scale) {
        float occ = 0.0;
        float w = 1.0;
        for (int i = 1; i <= 5; i++) {
            float h = scale * float(i);
            float3 q = p + nrm * h;
            float d = min(printerOnly(q, open), q.y + 1e3 * step(0.5, nrm.y));
            occ += (h - clamp(d, 0.0, h)) * w;
            w *= 0.6;
        }
        return clamp(1.0 - occ * 0.4 / scale, 0.0, 1.0);
    }

    float irradiance(float3 p, float3 nrm, bool open, float ao) {
        float3 o = p + nrm * 0.05;
        float e = 0.0;
        float c = dot(nrm, KEY_C);
        if (c > 0.0) e += KEY_L * 0.7 * KEY_OMEGA * c * softShadow(o, KEY_C, 1.0 / KEY_TB, open);
        c = dot(nrm, FILL_C);
        if (c > 0.0) e += FILL_L * 0.7 * FILL_OMEGA * c * softShadow(o, FILL_C, 1.0 / FILL_TB, open);
        c = dot(nrm, RIM_C);
        if (c > 0.0) e += RIM_L * 0.7 * RIM_OMEGA * c * softShadow(o, RIM_C, 1.0 / RIM_TB, open);
        float up = 0.5 + 0.5 * nrm.y;
        float ground = 0.55 * (KEY_L * 0.7 * KEY_OMEGA * KEY_C.y + AMBIENT_E) * 0.3;
        return e + mix(ground, AMBIENT_E, up) * ao;
    }

    float3 tableRadiance(float3 p, bool open) {
        float ao = occlusion(p, float3(0, 1, 0), open, 6.0);
        float e = irradiance(p, float3(0, 1, 0), open, ao);
        float fall = 1.0 / (1.0 + dot(p.xz, p.xz) / 900000.0);
        return TABLE_ALBEDO / M_PI_F * e * fall;
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

    // Plastic: diffuse × (1 − F) plus one GGX sample of the studio.
    float3 shadePlastic(float3 p, float3 rd, float3 nrm, float3 alb, float alpha, float u1, float u2, bool open,
                        thread float &specOut) {
        float3 v = -rd;
        float cv = max(dot(nrm, v), 1e-4);
        float ao = occlusion(p, nrm, open, 1.2);
        float e = irradiance(p, nrm, open, ao);
        float3 diffuse = alb / M_PI_F * e * (1.0 - fresnelDiel(N_HOUSING, cv));
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
            float3 o = p + nrm * 0.05;
            float Li;
            float tt;
            int mt;
            if (march(o, l, 600.0, open, tt, mt)) {
                // Something of the scene seen in the plastic: the table or
                // the printer's own dark housing, lit by the tent. MODEL.
                Li = mt == 1 ? dot(TABLE_ALBEDO, float3(0.33)) / M_PI_F * AMBIENT_E : 0.02;
            } else Li = environment(l).x;
            spec = fresnelDiel(N_HOUSING, dot(v, m)) * g1 * Li;
        }
        specOut = spec;
        return diffuse + float3(spec);
    }

    float3 shadeMatte(float3 p, float3 nrm, float3 alb, bool open) {
        float ao = occlusion(p, nrm, open, 1.2);
        return alb / M_PI_F * irradiance(p, nrm, open, ao);
    }

    // The cancel button's red ✕ and the wireless button's blue mark, printed
    // on their tops. MODEL shapes after HP's panel drawing.
    float3 buttonPrint(float3 p, int mat, thread bool &printed) {
        float3 q = p / SCALE;
        printed = false;
        if (mat == 13) {
            float2 c = q.xz - float2(-148.0, 45.0);
            float d = min(abs(c.x - c.y), abs(c.x + c.y)) * 0.7071;
            if (d < 0.55 && max(abs(c.x), abs(c.y)) < 2.6) { printed = true; return float3(0.55, 0.03, 0.02); }
        }
        if (mat == 3 && q.y > H - 0.5 && abs(q.x + 148.0) < 6.0 && abs(q.z + 4.0) < 6.0) {
            float2 c = q.xz - float2(-148.0, -4.0);
            float rr = length(c);
            bool arc = (abs(rr - 1.7) < 0.35 || abs(rr - 3.0) < 0.35) && abs(c.x) > 0.8 * rr;
            if (rr < 0.6 || arc) { printed = true; return float3(0.05, 0.25, 0.85); }
        }
        return float3(0.0);
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

    kernel void render(device uchar4 *pixels [[buffer(0)]],
                       device float4 *aux [[buffer(1)]],
                       constant Params &P [[buffer(2)]],
                       uint2 gid [[thread_position_in_grid]]) {
        uint x = gid.x;
        uint y = gid.y + P.rowOffset;
        if (x >= P.width || y >= P.height) return;
        bool open = P.trayOpen != 0;
        float3 ro = P.camPos.xyz;
        float3 sum = float3(0.0);
        float specSum = 0.0, allSum = 0.0;
        uint S = P.samples;
        for (uint sy = 0; sy < S; sy++) {
            for (uint sx = 0; sx < S; sx++) {
                uint idx = (y * P.width + x) * 64u + sy * S + sx;
                float2 jitter = float2((float(sx) + 0.5) / float(S), (float(sy) + 0.5) / float(S));
                float3 rd = cameraRay(float2(x, y) + jitter, P);
                float t;
                int mat;
                float3 c;
                if (!march(ro, rd, FAR, open, t, mat)) c = environment(rd);
                else {
                    float3 p = ro + rd * t;
                    if (mat == 1) c = tableRadiance(p, open);
                    else {
                        float3 nrm = printerNormal(p, open);
                        float u1 = hash(idx * 2u + 1u), u2 = hash(idx * 2u + 2u);
                        float spec = 0.0;
                        if (mat == 2) {
                            c = shadePlastic(p, rd, nrm, HOUSING_ALB, HOUSING_A, u1, u2, open, spec);
                            specSum += spec;
                            allSum += dot(c, float3(0.2126, 0.7152, 0.0722));
                        } else if (mat == 3 || mat == 13) {
                            bool printed;
                            float3 ink = buttonPrint(p, mat, printed);
                            c = shadePlastic(p, rd, nrm, printed ? ink : BUTTON_ALB, BUTTON_A, u1, u2, open, spec);
                        } else if (mat == 4) c = shadePlastic(p, rd, nrm, BADGE_ALB, 0.25, u1, u2, open, spec);
                        else if (mat == 5) c = shadeMatte(p, nrm, PAPER_ALB, open);
                        else if (mat == 10) c = LED_WIRELESS;
                        else if (mat == 11) c = LED_ATTENTION + shadePlastic(p, rd, nrm, float3(0.3, 0.15, 0.02), 0.2, u1, u2, open, spec);
                        else c = LED_READY;
                    }
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
        if (march(ro, rdc, FAR, open, t, mat)) a0 = float4(float(mat), t, 0.0, 0.0);
        aux[2 * (y * P.width + x)] = a0;
        aux[2 * (y * P.width + x) + 1] = float4(specSum / float(S * S), allSum / float(S * S), 0.0, 0.0);
    }

    kernel void probe(device const float4 *points [[buffer(0)]],
                      device float4 *out [[buffer(1)]],
                      constant uint &count [[buffer(2)]],
                      constant uint &trayOpen [[buffer(3)]],
                      uint id [[thread_position_in_grid]]) {
        if (id >= count) return;
        float3 p = points[id].xyz;
        Hit h = scene(p, trayOpen != 0);
        out[id] = float4(h.d, float(h.mat), printerOnly(p, trayOpen != 0), 0.0);
    }
    """
}

// MARK: - running it

enum PrinterError: Error, CustomStringConvertible {
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
    var trayOpen: UInt32
    var pad0: UInt32 = 0
    var pad1: UInt32 = 0
    var pad2: UInt32 = 0
    var camPos: SIMD4<Float>
    var camFwd: SIMD4<Float>
    var camRight: SIMD4<Float>
    var camUp: SIMD4<Float>
}

struct PrinterImage {
    let width: Int
    let height: Int
    let pixels: MTLBuffer
    let aux: MTLBuffer

    func rgba(_ x: Int, _ y: Int) -> SIMD4<UInt8> {
        let p = pixels.contents().advanced(by: (y * width + x) * 4).assumingMemoryBound(to: UInt8.self)
        return SIMD4<UInt8>(p[0], p[1], p[2], p[3])
    }
    /// (material, ray length, 0, 0).
    func seen(_ x: Int, _ y: Int) -> SIMD4<Float> {
        aux.contents().assumingMemoryBound(to: SIMD4<Float>.self)[2 * (y * width + x)]
    }
    /// (mean mirrored luminance on the housing, mean luminance there).
    func light(_ x: Int, _ y: Int) -> SIMD4<Float> {
        aux.contents().assumingMemoryBound(to: SIMD4<Float>.self)[2 * (y * width + x) + 1]
    }
}

func findDevice() throws -> MTLDevice {
    if let d = MTLCreateSystemDefaultDevice() ?? MTLCopyAllDevices().first { return d }
    throw PrinterError.noMetalDevice
}

final class PrinterRenderer {
    let device: MTLDevice
    let queue: MTLCommandQueue
    let renderPSO: MTLComputePipelineState
    let probePSO: MTLComputePipelineState
    let width: Int
    let height: Int
    let image: PrinterImage

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
            throw PrinterError.kernelCompile("\(error)")
        }
        func pipeline(_ name: String) throws -> MTLComputePipelineState {
            guard let f = library.makeFunction(name: name) else { throw PrinterError.kernelCompile("no kernel \(name)") }
            do { return try device.makeComputePipelineState(function: f) } catch { throw PrinterError.kernelCompile("\(error)") }
        }
        renderPSO = try pipeline("render")
        probePSO = try pipeline("probe")
        guard let q = device.makeCommandQueue(),
              let pixels = device.makeBuffer(length: width * height * 4, options: .storageModeShared),
              let aux = device.makeBuffer(length: width * height * 32, options: .storageModeShared)
        else { throw PrinterError.gpu("could not allocate buffers") }
        queue = q
        image = PrinterImage(width: width, height: height, pixels: pixels, aux: aux)
    }

    func render(camera: Camera, samples: Int, trayOpen: Bool = true) throws -> Double {
        let band: Int = 16
        var gpu: Double = 0
        var row: Int = 0
        while row < height {
            let rows: Int = min(band, height - row)
            guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else {
                throw PrinterError.gpu("could not make a command buffer")
            }
            var params = Params(width: UInt32(width), height: UInt32(height), rowOffset: UInt32(row), samples: UInt32(samples),
                                trayOpen: trayOpen ? 1 : 0, camPos: SIMD4<Float>(camera.position, 0),
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
            if let e = cb.error { throw PrinterError.gpu(e.localizedDescription) }
            gpu += cb.gpuEndTime - cb.gpuStartTime
            row += rows
        }
        return gpu
    }

    /// (scene distance, material, printer-only distance, 0) at each point.
    func probe(_ points: [SIMD3<Float>], trayOpen: Bool = true) throws -> [SIMD4<Float>] {
        var out: [SIMD4<Float>] = []
        out.reserveCapacity(points.count)
        let chunk: Int = 262_144
        var start: Int = 0
        while start < points.count {
            let end: Int = min(start + chunk, points.count)
            var pts: [SIMD4<Float>] = points[start..<end].map { SIMD4<Float>($0, 0) }
            var count = UInt32(pts.count)
            var open: UInt32 = trayOpen ? 1 : 0
            guard let inb = device.makeBuffer(bytes: &pts, length: 16 * max(pts.count, 1), options: .storageModeShared),
                  let outb = device.makeBuffer(length: 16 * max(pts.count, 1), options: .storageModeShared),
                  let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder()
            else { throw PrinterError.gpu("could not set up the probe") }
            enc.setComputePipelineState(probePSO)
            enc.setBuffer(inb, offset: 0, index: 0)
            enc.setBuffer(outb, offset: 0, index: 1)
            enc.setBytes(&count, length: 4, index: 2)
            enc.setBytes(&open, length: 4, index: 3)
            enc.dispatchThreads(MTLSize(width: pts.count, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: min(probePSO.maxTotalThreadsPerThreadgroup, 256), height: 1, depth: 1))
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            if let e = cb.error { throw PrinterError.gpu(e.localizedDescription) }
            let o = outb.contents().assumingMemoryBound(to: SIMD4<Float>.self)
            for i in 0..<pts.count { out.append(o[i]) }
            start = end
        }
        return out
    }
}

func savePNG(_ image: PrinterImage, to url: URL) throws {
    try? FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    guard let provider = CGDataProvider(dataInfo: nil, data: image.pixels.contents(), size: image.width * image.height * 4,
                                        releaseData: { _, _, _ in }),
          let space = CGColorSpace(name: CGColorSpace.sRGB),
          let cg = CGImage(width: image.width, height: image.height, bitsPerComponent: 8, bitsPerPixel: 32,
                           bytesPerRow: image.width * 4, space: space,
                           bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                           provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent),
          let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)
    else { throw PrinterError.png("could not set up the encoder for \(url.path)") }
    CGImageDestinationAddImage(dest, cg, nil)
    guard CGImageDestinationFinalize(dest) else { throw PrinterError.png("could not write \(url.path)") }
}
