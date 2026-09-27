// The lower arch, rendered as one still: one GPU thread per pixel marching rays
// into a scene made entirely of distance functions, as the step 3 ocean did.
//
// Every surface is a formula that answers "how far is the nearest surface
// from here?" A ray steps forward by that distance, over and over, until the
// answer is nearly zero — it has arrived. There are no meshes and no files of
// geometry: a tooth is a rounded box that narrows at the neck, with a height
// field of rounded cusps for a top.
//
// The work is in the light, which is the point of the step:
//
//   * enamel under a saliva film — a crisp, weak highlight from the water on
//     top, a softer, weaker one from the enamel below it (Anatomy.swift says
//     why each is the size it is);
//   * translucency at the thin incisal edge, read straight off the distance
//     function: step inward from the surface and see how soon you come out;
//   * colour that changes from neck to edge, from measured CIELAB;
//   * wet, faintly stippled gums, and a matte, papillated tongue;
//   * one big soft key light with soft shadows, a fill, and ambient occlusion
//     where teeth meet — which the distance function also gives for free.

import CoreGraphics
import Foundation
import ImageIO
import Metal
import UniformTypeIdentifiers
import simd

// MARK: - the crown's shape, beyond the table

/// Where the crown is widest mesiodistally, as a fraction of its height from
/// the neck: the contact areas are in the incisal third on incisors and the
/// occlusal third on posterior teeth (Wheeler's, each tooth's chapter).
func contactHeight(_ kind: CrownKind) -> Float {
    switch kind {
    case .incisor: return 0.82
    case .canine: return 0.75
    default: return 0.78
    }
}

/// Half the labiolingual thickness at the very top of the crown. An incisal
/// edge is about a millimetre thick; a posterior occlusal table is roughly 55–
/// 65% of the crown's depth (Wheeler's). MODEL within those.
func topHalfDepth(_ spec: ToothSpec) -> Float {
    switch spec.kind {
    case .incisor: return 0.55
    case .canine: return 0.9
    case .firstPremolar, .secondPremolar: return 0.30 * spec.depth
    case .firstMolar, .secondMolar: return 0.32 * spec.depth
    }
}

// MARK: - the camera and the lights

/// Looking down and across the lower arch from in front of the patient's
/// right premolars, the angle of the reference: incisors turning away on the
/// left, the near premolars square to the camera, a molar at the right edge.
let cameraPosition = SIMD3<Float>(-50, 17, 1)
let cameraTarget = SIMD3<Float>(-9, -4.5, 14)
let tanHalfFOV: Float = 0.30

/// A softbox above and in front, and a second, dimmer one beside the camera —
/// the way dental photographs are lit, with the flash at the lens, which is
/// what puts a highlight on every labial face.
///
/// Each light is a real disc in the sky, not just a direction: its radiance is
/// its irradiance divided by the solid angle it covers, so the highlight a wet
/// tooth mirrors back is exactly as bright as the light that also lights it
/// diffusely. A small, bright box gives crisp highlights; the same light spread
/// over a big dim one gives none — which is what the first render did.
let keyDirection = simd_normalize(SIMD3<Float>(-0.45, 1.0, -0.55))
let fillDirection = simd_normalize(simd_normalize(cameraPosition - cameraTarget) + SIMD3<Float>(0, 0.25, 0))
let keyColour = SIMD3<Float>(1.0, 0.98, 0.95) * 1.7
let fillColour = SIMD3<Float>(0.96, 0.97, 1.0) * 0.8
/// Angular radius of each softbox, as the cosine of it.
let keyDiscCosine: Float = 0.985     // about 10°
let fillDiscCosine: Float = 0.990    // about 8°

/// Irradiance over solid angle: the radiance of a disc of that size.
func discRadiance(_ irradiance: SIMD3<Float>, cosine: Float) -> SIMD3<Float> {
    let solidAngle: Float = 2 * Float.pi * (1 - cosine)
    return irradiance / solidAngle
}

/// The picture's ground, in display values, as in the reference: near white.
let backgroundDisplay: Float = 0.965

// MARK: - GPU layout

/// One tooth as the kernel sees it: six float4s, so there is no padding to get
/// wrong between Swift and Metal.
struct GPUTooth {
    var frame: SIMD4<Float>      // centre x, centre z, tangent x, tangent z
    var outward: SIMD4<Float>    // outward x, outward z, contact height, side
    var halves: SIMD4<Float>     // half width, half cervical width, half depth, half cervical depth
    var shape: SIMD4<Float>      // crown height, half depth at top, CEJ curve, kind
    var toothBound: SIMD4<Float> // sphere enclosing the crown and the root shown
    var gumBound: SIMD4<Float>   // sphere enclosing this tooth's block of gum
}

/// How much root is modelled below the neck: only enough to sit in the gum.
let rootShown: Float = 6.0
/// How deep the gum body runs below the neck.
let gumDepth: Float = 16.0

func gpuTeeth(_ placed: [PlacedTooth]) -> [GPUTooth] {
    placed.map { t in
        let s: ToothSpec = t.spec
        let h: Float = s.crownHeight
        let tb = SIMD4<Float>(t.centre.x, -(h + rootShown) / 2, t.centre.y,
                              simd_length(SIMD3<Float>(s.width / 2, (h + rootShown) / 2, s.depth / 2)) + 0.5)
        let gHalfDepth: Float = s.cervicalDepth / 2 + gumWidth
        let gHalfHeight: Float = (h + gumDepth) / 2
        let gb = SIMD4<Float>(t.centre.x, -gHalfHeight, t.centre.y,
                              simd_length(SIMD3<Float>(s.width / 2 + 1.5, gHalfHeight, gHalfDepth)) + 1.5)
        return GPUTooth(
            frame: SIMD4<Float>(t.centre.x, t.centre.y, t.tangent.x, t.tangent.y),
            outward: SIMD4<Float>(t.outward.x, t.outward.y, contactHeight(s.kind), t.side),
            halves: SIMD4<Float>(s.width / 2, s.cervicalWidth / 2, s.depth / 2, s.cervicalDepth / 2),
            shape: SIMD4<Float>(h, topHalfDepth(s), s.cejCurve, Float(s.kind.rawValue)),
            toothBound: tb, gumBound: gb)
    }
}

// MARK: - the kernel

/// What to break, for the mutation check. Each one must be caught by a test.
enum Mutant: Int {
    case none = 0
    case flat = 1       // one colour from neck to edge
    case opaque = 2     // no translucency at the thin edge
}

/// How far a ray may trust a distance. The crown's cusp height field and the
/// gum's slope make the distance functions over-report by up to ~1.5×, so a
/// ray steps 60% of what it is told; the tests measure the real factor.
let stepScale: Float = 0.6

func metal(_ v: SIMD3<Float>) -> String { "float3(\(v.x), \(v.y), \(v.z))" }

func kernelSource(mutant: Mutant) -> String {
    let forward: SIMD3<Float> = simd_normalize(cameraTarget - cameraPosition)
    let right: SIMD3<Float> = simd_normalize(simd_cross(forward, SIMD3<Float>(0, 1, 0)))
    let up: SIMD3<Float> = simd_cross(right, forward)
    let kinds: [CrownKind] = CrownKind.allCases
    let mids: String = kinds.map { metal(labToLinearSRGB(middleThirdLab[$0]!)) }.joined(separator: ", ")
    let cervs: String = kinds.map { metal(labToLinearSRGB(middleThirdLab[$0]! + cervicalShift)) }.joined(separator: ", ")
    return """
    #include <metal_stdlib>
    using namespace metal;

    struct Tooth { float4 frame; float4 outward; float4 halves; float4 shape; float4 toothBound; float4 gumBound; };
    struct Params { uint width; uint height; uint rowOffset; uint samples; };

    constant uint TOOTH_COUNT = 14;
    constant int MUTANT = \(mutant.rawValue);
    constant float STEP_SCALE = \(stepScale);
    constant float HIT_EPS = 0.003;
    constant float FAR = 260.0;

    constant float3 CAM_POS = \(metal(cameraPosition));
    constant float3 CAM_FWD = \(metal(forward));
    constant float3 CAM_RIGHT = \(metal(right));
    constant float3 CAM_UP = \(metal(up));
    constant float TAN_HALF_FOV = \(tanHalfFOV);

    constant float3 KEY_DIR = \(metal(keyDirection));
    constant float3 FILL_DIR = \(metal(fillDirection));
    constant float3 KEY_COL = \(metal(keyColour));
    constant float3 FILL_COL = \(metal(fillColour));
    constant float3 KEY_RADIANCE = \(metal(discRadiance(keyColour, cosine: keyDiscCosine)));
    constant float3 FILL_RADIANCE = \(metal(discRadiance(fillColour, cosine: fillDiscCosine)));
    constant float KEY_DISC = \(keyDiscCosine);
    constant float FILL_DISC = \(fillDiscCosine);
    constant float BG = \(backgroundDisplay);

    constant float ROOT_SHOWN = \(rootShown);
    constant float EDGE_ROUND = 0.45;
    constant float CUSP_K = 0.3;          // dome curvature, per mm²
    constant float CUSP_BLEND = 0.7;      // how softly neighbouring cusps meet in a groove
    constant float CUSP_SLOPE = 1.0;      // steepest cusp incline, 45°; real inclines run ~25–45°
    constant float TOP_SLOPE = 1.25;      // steepest the whole occlusal height field gets, canine ridges included

    constant float GUM_MARGIN = \(gumMarginAboveCEJ);
    constant float PAPILLA_EXTRA = \(papillaExtra);
    constant float GUM_WIDTH = \(gumWidth);
    constant float GUM_SLOPE = \(gumSlope);
    constant float GUM_DEPTH = \(gumDepth);
    constant float GUM_ROUND = 0.8;
    constant float GUM_COLLAR = 4.5;      // mm over which the gum goes from a knife edge to full thickness
    constant float GUM_BLEND = 2.0;

    constant float3 TONGUE_C = float3(0.0, -14.5, 28.0);
    constant float3 TONGUE_R = float3(14.0, 8.0, 21.0);
    constant float FLOOR_Y = -14.0;

    constant float3 MID[6] = { \(mids) };
    constant float3 CERV[6] = { \(cervs) };
    constant float3 INCISAL = \(metal(labToLinearSRGB(incisalLab)));
    constant float3 GUM = \(metal(labToLinearSRGB(gingivaLab)));
    constant float3 TONGUE = \(metal(labToLinearSRGB(tongueLab)));

    constant float FILM_F0 = \(filmF0);
    constant float ENAMEL_F0 = \(enamelUnderFilmF0);

    // ---------------------------------------------------------------- helpers

    float smin(float a, float b, float k) {
        float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
        return mix(b, a, h) - k * h * (1.0 - h);
    }
    float smax(float a, float b, float k) { return -smin(-a, -b, k); }

    float hash3(float3 p) {
        p = fract(p * 0.3183099 + 0.1);
        p *= 17.0;
        return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
    }
    float noise3(float3 x) {
        float3 i = floor(x);
        float3 f = fract(x);
        f = f * f * (3.0 - 2.0 * f);
        float a = mix(mix(hash3(i + float3(0,0,0)), hash3(i + float3(1,0,0)), f.x),
                      mix(hash3(i + float3(0,1,0)), hash3(i + float3(1,1,0)), f.x), f.y);
        float b = mix(mix(hash3(i + float3(0,0,1)), hash3(i + float3(1,0,1)), f.x),
                      mix(hash3(i + float3(0,1,1)), hash3(i + float3(1,1,1)), f.x), f.y);
        return mix(a, b, f.z);
    }
    float3 noiseGrad(float3 x) {
        float e = 0.05;
        float c = noise3(x);
        return float3(noise3(x + float3(e,0,0)) - c, noise3(x + float3(0,e,0)) - c, noise3(x + float3(0,0,e)) - c) / e;
    }

    // ---------------------------------------------------------------- a tooth

    float2 localUV(constant Tooth &T, float3 p) {
        float2 rel = p.xz - T.frame.xy;
        return float2(dot(rel, T.frame.zw), dot(rel, T.outward.xy));
    }

    // The crown's walls: a rounded rectangle in plan, narrow at the neck,
    // widest at the contacts mesiodistally and in the cervical third
    // labiolingually, converging to the edge or the occlusal table.
    float crownSide(constant Tooth &T, float2 uv, float y) {
        float H = T.shape.x;
        float t = (y + H) / H;
        float a;
        float b;
        if (t >= 0.0) {
            a = mix(T.halves.y, T.halves.x, smoothstep(0.0, T.outward.z, t));
            b = t < 0.3 ? mix(T.halves.w, T.halves.z, smoothstep(0.0, 0.3, t))
                        : mix(T.halves.z, T.shape.y, smoothstep(0.3, 1.0, t));
        } else {
            float below = y + H;                  // negative: mm into the root
            a = T.halves.y * max(1.0 + 0.05 * below, 0.3);
            b = T.halves.w * max(1.0 + 0.05 * below, 0.3);
        }
        float r = min(a, b) * 0.7;
        float2 q = abs(uv) - float2(a, b) + r;
        return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - r;
    }

    // A rounded cusp: a paraboloid near its tip that becomes a cone of slope
    // CUSP_SLOPE further out. A pure paraboloid keeps steepening forever,
    // which made the distance function claim far more room than there was.
    float dome(float2 uv, float cu, float cv, float h) {
        float r = length(uv - float2(cu, cv));
        float r0 = CUSP_SLOPE / (2.0 * CUSP_K);
        return r < r0 ? h - CUSP_K * r * r : h - CUSP_K * r0 * r0 - CUSP_SLOPE * (r - r0);
    }

    // The top as a height field. u runs distally, v buccally or labially.
    float crownTop(constant Tooth &T, float2 uv) {
        int kind = int(T.shape.w + 0.5);
        float a = T.halves.x;
        float b = T.halves.z;
        if (kind == 0) {
            float un = uv.x / a;
            return -0.45 * un * un * un * un;
        }
        if (kind == 1) {
            // The mandibular canine's tip sits a little mesial, and its mesial
            // ridge is the shorter, steeper one.
            float du = uv.x + 0.12 * a;
            float k = du < 0.0 ? 1.05 : 0.8;
            return -k * (sqrt(du * du + 0.5) - sqrt(0.5)) - 0.25 * max(-uv.y, 0.0);
        }
        float h;
        if (kind == 2) {
            h = smax(dome(uv, 0.0, 0.3 * b, 0.0), dome(uv, 0.0, -0.5 * b, -2.2), CUSP_BLEND);
        } else if (kind == 3) {
            h = smax(dome(uv, 0.0, 0.3 * b, 0.0), dome(uv, -0.35 * a, -0.5 * b, -1.0), CUSP_BLEND);
            h = smax(h, dome(uv, 0.4 * a, -0.5 * b, -1.3), CUSP_BLEND);
        } else if (kind == 4) {
            h = smax(dome(uv, -0.45 * a, -0.5 * b, 0.0), dome(uv, 0.35 * a, -0.5 * b, -0.2), CUSP_BLEND);
            h = smax(h, dome(uv, -0.55 * a, 0.5 * b, -0.3), CUSP_BLEND);
            h = smax(h, dome(uv, 0.0, 0.55 * b, -0.4), CUSP_BLEND);
            h = smax(h, dome(uv, 0.62 * a, 0.3 * b, -0.8), CUSP_BLEND);
        } else {
            h = smax(dome(uv, -0.45 * a, -0.5 * b, 0.0), dome(uv, 0.45 * a, -0.5 * b, -0.2), CUSP_BLEND);
            h = smax(h, dome(uv, -0.45 * a, 0.5 * b, -0.3), CUSP_BLEND);
            h = smax(h, dome(uv, 0.45 * a, 0.5 * b, -0.4), CUSP_BLEND);
        }
        return h;
    }

    float crownSDF(constant Tooth &T, float3 p, thread float &side) {
        float2 uv = localUV(T, p);
        side = crownSide(T, uv, p.y);
        // The top is a height field, so its height difference over-reports
        // distance by up to √(1 + slope²). Read it inside the crown's own
        // outline only, and divide by that bound, so a ray can trust it.
        float2 inside = clamp(uv, -T.halves.xz, T.halves.xz);
        float top = (p.y - crownTop(T, inside)) / sqrt(1.0 + TOP_SLOPE * TOP_SLOPE);
        float d = smax(side, top, EDGE_ROUND);
        return max(d, (-T.shape.x - ROOT_SHOWN) - p.y);
    }

    // One tooth's share of gum: a shell around its neck and root that is a
    // knife edge at the gingival margin and thickens below it into the
    // alveolar ridge. The margin rises between teeth as the junction does, so
    // where two shells meet they fill the gap between the teeth: a papilla.
    // Neighbouring shells melt together into one gum.
    // The shape is built in real millimetres, so every blend radius means what
    // it says. What comes back alongside is how much this block's distance can
    // over-report — its steepest gradient — so the scene can divide by it once,
    // at the end. Dividing inside, before the blends, made the blends four
    // times fatter between incisors and the gum swallowed the crowns.
    float gumBlock(constant Tooth &T, float3 p, float side, thread float &steep) {
        float2 uv = localUV(T, p);
        float H = T.shape.x;
        float un = clamp(uv.x / T.halves.x, -1.25, 1.25);
        float climb = T.shape.z + PAPILLA_EXTRA;
        float top = -H + GUM_MARGIN + climb * un * un;
        float below = top - p.y;
        float x = clamp(below / GUM_COLLAR, 0.0, 1.0);
        float thick = GUM_WIDTH * x * x * (3.0 - 2.0 * x) + GUM_SLOPE * max(below - GUM_COLLAR, 0.0);
        float shell = side - thick;
        float bottom = (-H - GUM_DEPTH) - p.y;
        // The margin's steepest climb on this tooth — between incisors the
        // papilla rises about four millimetres per millimetre — and the
        // collar's steepest thickening. The shell's gradient is at most
        // 1 + thickening × √(1 + climb²).
        float slope = 2.0 * climb * 1.25 / T.halves.x;
        float lean = sqrt(1.0 + slope * slope);
        float thickRate = max(GUM_WIDTH * 1.5 / GUM_COLLAR, GUM_SLOPE);
        steep = 1.0 + thickRate * lean;
        return max(smax(shell, p.y - top, GUM_ROUND), bottom);
    }

    float tongueSDF(float3 p) {
        float3 q = p - TONGUE_C;
        float k0 = length(q / TONGUE_R);
        float k1 = length(q / (TONGUE_R * TONGUE_R));
        return k0 * (k0 - 1.0) / k1;
    }

    // The floor of the mouth, inside the arch only, so nothing shows through.
    float floorSDF(float3 p) {
        float e = length((p.xz - float2(0.0, 22.0)) / float2(20.0, 26.0)) - 1.0;
        return max(p.y - FLOOR_Y, e * 18.0);
    }

    // ---------------------------------------------------------------- the scene

    // Materials: 1 enamel, 2 gum, 3 tongue, 4 floor of mouth.
    float sceneSDF(float3 p, thread int &mat, constant Tooth *teeth) {
        float dT = 1e9;
        float dG = 1e9;
        // Every tooth, every time. Skipping teeth whose bounding sphere is far
        // away looks free, but it is only exact if no distance ever
        // under-reports — and the gum's, divided by its slope so rays can
        // trust it, does. The skip then switched blocks on and off in open
        // air and the distance jumped 20× across the switch.
        float steepG = 1.0;
        for (uint i = 0; i < TOOTH_COUNT; i++) {
            float side;
            float c = crownSDF(teeth[i], p, side);
            dT = min(dT, c);
            float steep;
            float g = gumBlock(teeth[i], p, side, steep);
            // smin, written out so the steepness can ride on the same weight:
            // it follows whichever block dominates, and never jumps.
            float h = clamp(0.5 + 0.5 * (g - dG) / GUM_BLEND, 0.0, 1.0);
            dG = mix(g, dG, h) - GUM_BLEND * h * (1.0 - h);
            steepG = mix(steep, steepG, h);
        }
        dG /= steepG;
        float d = dT;
        mat = 1;
        if (dG < d) { d = dG; mat = 2; }
        float dTo = tongueSDF(p);
        if (dTo < d) { d = dTo; mat = 3; }
        float dF = floorSDF(p);
        if (dF < d) { d = dF; mat = 4; }
        return d;
    }

    // Teeth alone, and which one is nearest, for the shading.
    float teethSDF(float3 p, constant Tooth *teeth, thread int &nearest) {
        float dT = 1e9;
        nearest = 0;
        for (uint i = 0; i < TOOTH_COUNT; i++) {
            // Safe here: a crown's distance under-reports by at most the
            // occlusal height field's √(1 + 1.25²) = 1.6, so a tooth whose
            // bounding sphere is 1.6× further than the best so far cannot win.
            float tb = length(p - teeth[i].toothBound.xyz) - teeth[i].toothBound.w;
            if (tb > dT * 1.61) continue;
            float side;
            float c = crownSDF(teeth[i], p, side);
            if (c < dT) { dT = c; nearest = int(i); }
        }
        return dT;
    }

    float3 sceneNormal(float3 p, constant Tooth *teeth) {
        const float e = 0.01;
        int m;
        float3 k1 = float3(1, -1, -1), k2 = float3(-1, -1, 1), k3 = float3(-1, 1, -1), k4 = float3(1, 1, 1);
        return normalize(k1 * sceneSDF(p + k1 * e, m, teeth) + k2 * sceneSDF(p + k2 * e, m, teeth)
                       + k3 * sceneSDF(p + k3 * e, m, teeth) + k4 * sceneSDF(p + k4 * e, m, teeth));
    }

    bool march(float3 ro, float3 rd, constant Tooth *teeth, thread float &t, thread int &mat) {
        t = 0.0;
        for (int i = 0; i < 400; i++) {
            float d = sceneSDF(ro + rd * t, mat, teeth);
            if (d < HIT_EPS) return true;
            t += d * STEP_SCALE;
            if (t > FAR) break;
        }
        return false;
    }

    float softShadow(float3 ro, float3 rd, constant Tooth *teeth) {
        float res = 1.0;
        float t = 0.08;
        int m;
        for (int i = 0; i < 80; i++) {
            float h = sceneSDF(ro + rd * t, m, teeth);
            res = min(res, 10.0 * h / t);
            t += clamp(h * STEP_SCALE, 0.03, 2.5);
            if (res < 0.002 || t > 70.0) break;
        }
        return clamp(res, 0.0, 1.0);
    }

    float ambientOcclusion(float3 p, float3 n, constant Tooth *teeth) {
        float occ = 0.0;
        float weight = 1.0;
        int m;
        for (int i = 1; i <= 5; i++) {
            float h = 0.35 * float(i) * float(i) * 0.5 + 0.2;
            float d = sceneSDF(p + n * h, m, teeth);
            occ += weight * clamp((h - d) / h, 0.0, 1.0);
            weight *= 0.75;
        }
        return clamp(1.0 - 0.45 * occ, 0.0, 1.0);
    }

    // ---------------------------------------------------------------- light

    // The room: dim below, brighter above, and the two softboxes as discs.
    // The discs' edges are softened by a quarter of their radius so a
    // highlight has a soft rim rather than aliasing.
    float3 environment(float3 d) {
        float up = smoothstep(-0.4, 0.8, d.y);
        float3 c = mix(float3(0.22, 0.19, 0.19), float3(0.75, 0.75, 0.78), up);
        float kSoft = (1.0 - KEY_DISC) * 0.5;
        float fSoft = (1.0 - FILL_DISC) * 0.5;
        c += KEY_RADIANCE * smoothstep(KEY_DISC - kSoft, KEY_DISC + kSoft, dot(d, KEY_DIR));
        c += FILL_RADIANCE * smoothstep(FILL_DISC - fSoft, FILL_DISC + fSoft, dot(d, FILL_DIR));
        return c;
    }

    float3 irradiance(float3 n) {
        return mix(float3(0.30, 0.25, 0.25), float3(0.85, 0.86, 0.9), 0.5 + 0.5 * n.y);
    }

    float ggx(float3 n, float3 v, float3 l, float alpha, float f0) {
        float3 h = normalize(v + l);
        float ndl = max(dot(n, l), 0.0);
        float ndv = max(dot(n, v), 1e-3);
        float ndh = max(dot(n, h), 0.0);
        float vdh = max(dot(v, h), 0.0);
        float a2 = alpha * alpha;
        float den = ndh * ndh * (a2 - 1.0) + 1.0;
        float D = a2 / (M_PI_F * den * den);
        float k = alpha * 0.5;
        float G = (ndv / (ndv * (1.0 - k) + k)) * (ndl / (ndl * (1.0 - k) + k));
        float F = f0 + (1.0 - f0) * pow(1.0 - vdh, 5.0);
        return D * G * F / (4.0 * ndv * ndl + 1e-4) * ndl;
    }

    float schlick(float f0, float c) { return f0 + (1.0 - f0) * pow(1.0 - c, 5.0); }

    // The crown-height fraction at a point, 0 at the neck and 1 at the
    // occlusal plane, and the tooth's kind.
    float toothT(float3 p, constant Tooth *teeth, thread int &kind) {
        int idx;
        teethSDF(p, teeth, idx);
        kind = int(teeth[idx].shape.w + 0.5);
        float H = teeth[idx].shape.x;
        return clamp((p.y + H) / H, 0.0, 1.0);
    }

    float3 shade(float3 p, float3 rd, int mat, constant Tooth *teeth) {
        float3 n = sceneNormal(p, teeth);
        float3 v = -rd;
        float occ = ambientOcclusion(p, n, teeth);
        float sh = softShadow(p + n * 0.05, KEY_DIR, teeth);

        float3 albedo;
        float wrap = 0.15;
        float3 sssTint = float3(0.0);
        float filmAlpha = 0.07;
        float baseAlpha = 0.3;
        float baseF0 = ENAMEL_F0;

        if (mat == 1) {
            int kind;
            float t = toothT(p, teeth, kind);
            float3 mid = MID[kind];
            float3 cerv = CERV[kind];
            // The measurements are by thirds of the crown, so the cervical
            // colour holds through the cervical third and hands over to the
            // middle third's across the boundary between them.
            albedo = MUTANT == 1 ? mid : mix(cerv, mid, smoothstep(0.22, 0.5, t));
            float anterior = kind <= 1 ? 1.0 : 0.35;
            if (MUTANT != 1) albedo = mix(albedo, INCISAL, smoothstep(0.72, 1.0, t) * anterior);
            // Translucency: step inward and see how soon the tooth ends. A
            // thin edge ends almost at once and lets the dark mouth behind it
            // show through, greyer and bluer; enamel scatters short
            // wavelengths back and passes long ones on.
            int idx;
            const float probe = 1.1;
            float inside = clamp(-teethSDF(p - n * probe, teeth, idx) / probe, 0.0, 1.0);
            float translucency = MUTANT == 2 ? 0.0 : (1.0 - inside) * smoothstep(0.55, 0.95, t);
            albedo = mix(albedo, albedo * float3(0.55, 0.62, 0.78), translucency * 0.75);
            wrap = 0.25;
        } else if (mat == 2) {
            float3 bump = noiseGrad(p * 3.0);
            n = normalize(n + 0.05 * (bump - n * dot(bump, n)));
            albedo = GUM;
            wrap = 0.5;
            sssTint = float3(0.45, 0.08, 0.05);
            filmAlpha = mix(0.06, 0.22, noise3(p * 0.7));
            baseAlpha = 0.5;
            baseF0 = 0.0;
        } else {
            float3 bump = noiseGrad(p * 4.5);
            n = normalize(n + 0.18 * (bump - n * dot(bump, n)));
            albedo = mat == 3 ? mix(TONGUE, float3(0.62, 0.42, 0.40), 0.25 * noise3(p * 4.5)) : TONGUE * 0.8;
            wrap = 0.45;
            sssTint = float3(0.35, 0.06, 0.04);
            filmAlpha = 0.35;
            baseAlpha = 0.6;
            baseF0 = 0.0;
        }

        float ndv = max(dot(n, v), 1e-3);
        float nk = dot(n, KEY_DIR);
        float nf = dot(n, FILL_DIR);
        float key = max((nk + wrap) / (1.0 + wrap), 0.0) * sh;
        float fill = max((nf + wrap) / (1.0 + wrap), 0.0);
        float terminator = max(wrap - abs(nk), 0.0) / max(wrap, 1e-3) * sh;
        float Ffilm = schlick(FILM_F0, ndv);
        float3 diffuse = albedo * (KEY_COL * key + FILL_COL * fill + irradiance(n) * occ)
                       + albedo * sssTint * KEY_COL * terminator;
        diffuse *= (1.0 - Ffilm);

        // The saliva film is a mirror, so its highlight is the environment —
        // softboxes included — seen in reflection, dimmed by Fresnel. Where
        // the film is rougher (the gum's patches, the tongue) the reflection
        // is blurred towards the room's average; the analytic lobes carry the
        // lights for those rough surfaces, and for the enamel beneath the film.
        float3 r = reflect(-v, n);
        float blur = smoothstep(0.08, 0.35, filmAlpha);
        float3 mirror = environment(r);
        float keyVisible = mix(sh, 1.0, 0.0);
        float3 sharp = mirror * Ffilm * mix(1.0, occ, 0.5);
        if (dot(r, KEY_DIR) > KEY_DISC - 0.02) sharp *= keyVisible;
        float3 rough = KEY_COL * ggx(n, v, KEY_DIR, max(filmAlpha, 0.12), FILM_F0) * sh
                     + FILL_COL * ggx(n, v, FILL_DIR, max(filmAlpha, 0.12), FILM_F0)
                     + irradiance(r) * Ffilm * occ;
        float3 specular = mix(sharp, rough, blur);
        if (baseF0 > 0.0) specular += KEY_COL * ggx(n, v, KEY_DIR, baseAlpha, baseF0) * sh
                                    + FILL_COL * ggx(n, v, FILL_DIR, baseAlpha, baseF0);
        return diffuse + specular;
    }

    // A filmic shoulder applied to LUMINANCE, then the colour scaled to match,
    // so a measured tooth colour keeps its hue. A per-channel curve (ACES and
    // the like) clips the brightest channel first and drags ivory towards
    // green-yellow, which is what the first render did. Only above white does
    // the colour bleed towards white, as film does.
    float3 toneMap(float3 x) {
        float L = dot(x, float3(0.2126, 0.7152, 0.0722));
        float Lm = L * (1.0 + L / 16.0) / (1.0 + L);
        float3 c = x * (Lm / max(L, 1e-5)) * 1.12;
        float over = max(max(c.r, max(c.g, c.b)) - 1.0, 0.0);
        c = mix(c, float3(1.0), clamp(over, 0.0, 1.0));
        c = clamp(c, 0.0, 1.0);
        return select(1.055 * pow(c, 1.0 / 2.4) - 0.055, 12.92 * c, c <= 0.0031308);
    }

    float3 cameraRay(float2 pixel, uint width, uint height) {
        float aspect = float(width) / float(height);
        float sx = (2.0 * pixel.x / float(width) - 1.0) * aspect * TAN_HALF_FOV;
        float sy = (1.0 - 2.0 * pixel.y / float(height)) * TAN_HALF_FOV;
        return normalize(CAM_FWD + sx * CAM_RIGHT + sy * CAM_UP);
    }

    // ---------------------------------------------------------------- kernels

    kernel void mouth(device uchar4 *pixels [[buffer(0)]],
                      device float4 *aux [[buffer(1)]],
                      constant Params &P [[buffer(2)]],
                      constant Tooth *teeth [[buffer(3)]],
                      uint2 gid [[thread_position_in_grid]]) {
        uint x = gid.x;
        uint y = gid.y + P.rowOffset;
        if (x >= P.width || y >= P.height) return;
        float3 sum = float3(0.0);
        uint S = P.samples;
        for (uint sy = 0; sy < S; sy++) {
            for (uint sx = 0; sx < S; sx++) {
                float2 jitter = float2((float(sx) + 0.5) / float(S), (float(sy) + 0.5) / float(S));
                float3 rd = cameraRay(float2(x, y) + jitter, P.width, P.height);
                float t;
                int mat;
                if (march(CAM_POS, rd, teeth, t, mat)) {
                    sum += toneMap(shade(CAM_POS + rd * t, rd, mat, teeth));
                } else {
                    sum += float3(BG);
                }
            }
        }
        float3 c = sum / float(S * S);
        pixels[y * P.width + x] = uchar4(uchar3(round(clamp(c, 0.0, 1.0) * 255.0)), 255);

        // What the centre of the pixel sees, for the tests: material, crown
        // fraction, tooth kind.
        float3 rd = cameraRay(float2(x, y) + 0.5, P.width, P.height);
        float t;
        int mat;
        float4 a = float4(0.0);
        if (march(CAM_POS, rd, teeth, t, mat)) {
            a.x = float(mat);
            if (mat == 1) {
                int kind;
                a.y = toothT(CAM_POS + rd * t, teeth, kind);
                a.z = float(kind);
            }
        }
        aux[y * P.width + x] = a;
    }

    kernel void probe(device const float4 *points [[buffer(0)]],
                      device float2 *out [[buffer(1)]],
                      constant Tooth *teeth [[buffer(2)]],
                      uint id [[thread_position_in_grid]]) {
        int mat;
        float d = sceneSDF(points[id].xyz, mat, teeth);
        out[id] = float2(d, float(mat));
    }
    """
}

// MARK: - running it

enum MouthError: Error, CustomStringConvertible {
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
}

/// The finished picture, and what each pixel's centre saw.
struct MouthImage {
    let width: Int
    let height: Int
    let pixels: MTLBuffer
    let aux: MTLBuffer

    func rgba(_ x: Int, _ y: Int) -> SIMD4<UInt8> {
        let p = pixels.contents().advanced(by: (y * width + x) * 4).assumingMemoryBound(to: UInt8.self)
        return SIMD4<UInt8>(p[0], p[1], p[2], p[3])
    }

    /// (material, crown fraction, tooth kind, 0).
    func seen(_ x: Int, _ y: Int) -> SIMD4<Float> {
        aux.contents().assumingMemoryBound(to: SIMD4<Float>.self)[y * width + x]
    }
}

func findDevice() throws -> MTLDevice {
    if let d = MTLCreateSystemDefaultDevice() ?? MTLCopyAllDevices().first { return d }
    throw MouthError.noMetalDevice
}

func makeLibrary(_ device: MTLDevice, mutant: Mutant) throws -> MTLLibrary {
    let options = MTLCompileOptions()
    // Precise maths, as in every step: the distance functions are subtracted
    // from one another to within a few microns.
    options.fastMathEnabled = false
    do {
        return try device.makeLibrary(source: kernelSource(mutant: mutant), options: options)
    } catch {
        throw MouthError.kernelCompile("\(error)")
    }
}

func pipeline(_ device: MTLDevice, _ library: MTLLibrary, _ name: String) throws -> MTLComputePipelineState {
    guard let f = library.makeFunction(name: name) else { throw MouthError.kernelCompile("no kernel \(name)") }
    do { return try device.makeComputePipelineState(function: f) } catch {
        throw MouthError.kernelCompile("\(error)")
    }
}

/// Render in horizontal bands, one command buffer each, so no single piece of
/// GPU work runs long enough to trip the system's watchdog.
func renderMouth(width: Int, height: Int, samples: Int, mutant: Mutant = .none,
                 on device: MTLDevice) throws -> (image: MouthImage, gpuSeconds: Double) {
    let library = try makeLibrary(device, mutant: mutant)
    let pso = try pipeline(device, library, "mouth")
    var teeth: [GPUTooth] = gpuTeeth(placeTeeth())
    guard let pixels = device.makeBuffer(length: width * height * 4, options: .storageModeShared),
          let aux = device.makeBuffer(length: width * height * 16, options: .storageModeShared),
          let toothBuf = device.makeBuffer(bytes: &teeth, length: MemoryLayout<GPUTooth>.stride * teeth.count,
                                           options: .storageModeShared),
          let queue = device.makeCommandQueue()
    else { throw MouthError.gpu("could not allocate buffers") }

    let band: Int = 60
    var gpu: Double = 0
    var row: Int = 0
    while row < height {
        let rows: Int = min(band, height - row)
        guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else {
            throw MouthError.gpu("could not make a command buffer")
        }
        var params = Params(width: UInt32(width), height: UInt32(height),
                            rowOffset: UInt32(row), samples: UInt32(samples))
        enc.setComputePipelineState(pso)
        enc.setBuffer(pixels, offset: 0, index: 0)
        enc.setBuffer(aux, offset: 0, index: 1)
        enc.setBytes(&params, length: MemoryLayout<Params>.stride, index: 2)
        enc.setBuffer(toothBuf, offset: 0, index: 3)
        let w: Int = pso.threadExecutionWidth
        let group = MTLSize(width: w, height: max(pso.maxTotalThreadsPerThreadgroup / w / 4, 1), depth: 1)
        enc.dispatchThreads(MTLSize(width: width, height: rows, depth: 1), threadsPerThreadgroup: group)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        if let e = cb.error { throw MouthError.gpu(e.localizedDescription) }
        gpu += cb.gpuEndTime - cb.gpuStartTime
        row += rows
    }
    return (MouthImage(width: width, height: height, pixels: pixels, aux: aux), gpu)
}

/// The scene's distance and material at arbitrary points, from the same kernel
/// source the render uses — so a test of the distance function is a test of
/// the thing that drew the picture, not of a copy of it.
func probeScene(_ points: [SIMD3<Float>], on device: MTLDevice) throws -> [SIMD2<Float>] {
    let library = try makeLibrary(device, mutant: .none)
    let pso = try pipeline(device, library, "probe")
    var pts: [SIMD4<Float>] = points.map { SIMD4<Float>($0.x, $0.y, $0.z, 0) }
    var teeth: [GPUTooth] = gpuTeeth(placeTeeth())
    guard let pb = device.makeBuffer(bytes: &pts, length: 16 * pts.count, options: .storageModeShared),
          let ob = device.makeBuffer(length: 8 * pts.count, options: .storageModeShared),
          let tb = device.makeBuffer(bytes: &teeth, length: MemoryLayout<GPUTooth>.stride * teeth.count,
                                     options: .storageModeShared),
          let queue = device.makeCommandQueue(), let cb = queue.makeCommandBuffer(),
          let enc = cb.makeComputeCommandEncoder()
    else { throw MouthError.gpu("could not set up the probe") }
    enc.setComputePipelineState(pso)
    enc.setBuffer(pb, offset: 0, index: 0)
    enc.setBuffer(ob, offset: 0, index: 1)
    enc.setBuffer(tb, offset: 0, index: 2)
    enc.dispatchThreads(MTLSize(width: pts.count, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: min(pso.maxTotalThreadsPerThreadgroup, 256), height: 1, depth: 1))
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    if let e = cb.error { throw MouthError.gpu(e.localizedDescription) }
    let out = ob.contents().assumingMemoryBound(to: SIMD2<Float>.self)
    return (0..<pts.count).map { out[$0] }
}

func savePNG(_ image: MouthImage, to url: URL) throws {
    try? FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    guard let provider = CGDataProvider(dataInfo: nil, data: image.pixels.contents(), size: image.width * image.height * 4,
                                        releaseData: { _, _, _ in }),
          let space = CGColorSpace(name: CGColorSpace.sRGB),
          let cg = CGImage(width: image.width, height: image.height, bitsPerComponent: 8, bitsPerPixel: 32,
                           bytesPerRow: image.width * 4, space: space,
                           bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                           provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent),
          let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)
    else { throw MouthError.png("could not set up the encoder for \(url.path)") }
    CGImageDestinationAddImage(dest, cg, nil)
    guard CGImageDestinationFinalize(dest) else { throw MouthError.png("could not write \(url.path)") }
}
