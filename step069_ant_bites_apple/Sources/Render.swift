// Step 69: step 68's kernel (the GPU program), copied — step 44's many-ant
// shape buffer with its bounding spheres, step 68's peeled Golden Delicious —
// with its micrometre and molecule insets taken out: this step's inset is a
// drawn diagram (Annotate.swift), not a render. One ant, one piece of apple,
// one orthographic view, every surface a distance function.

import CoreGraphics
import Foundation
import ImageIO
import Metal
import UniformTypeIdentifiers
import simd

// MARK: - framing

/// The frame is 16:9. View positions are given in units of the frame's
/// HEIGHT, so any resolution puts them in the same places.
let frameAspect: Float = 16.0 / 9.0

/// The main view: millimetres across the full frame width. MODEL: close on
/// the head so the jaws' swing is tens of pixels; the whole ant still fits.
let mainViewWidth: Float = 9.0
/// The world point at the centre of the frame, and the view direction: from
/// above, in front and to the ant's right, looking down on the jaws. MODEL.
let mainCentre = SIMD3<Float>(-3.010, 1.041, -0.667)
let mainDirection: SIMD3<Float> = simd_normalize(SIMD3<Float>(-0.40, -1.25, -1.0))

/// An orthographic camera: where the frame centre looks, and its axes.
struct OrthoCamera {
    var centre: SIMD3<Float>
    var forward: SIMD3<Float>
    var right: SIMD3<Float>
    var up: SIMD3<Float>
    /// World units per unit of normalised screen coordinate.
    var halfWidth: Float

    init(centre: SIMD3<Float>, forward: SIMD3<Float>, halfWidth: Float) {
        self.centre = centre
        self.forward = forward
        self.right = simd_normalize(simd_cross(forward, SIMD3<Float>(0, 1, 0)))
        self.up = simd_cross(self.right, forward)
        self.halfWidth = halfWidth
    }

    /// World point to pixel, for a view whose centre is at `viewCentre` and
    /// whose half-width is `viewHalf`, both in pixels.
    func project(_ p: SIMD3<Float>, viewCentre: SIMD2<Float>, viewHalf: Float) -> SIMD2<Float> {
        let q: SIMD3<Float> = p - centre
        let sx: Float = simd_dot(q, right) / halfWidth
        let sy: Float = simd_dot(q, up) / halfWidth
        return SIMD2<Float>(viewCentre.x + sx * viewHalf, viewCentre.y - sy * viewHalf)
    }
}

let mainCamera = OrthoCamera(centre: mainCentre, forward: mainDirection, halfWidth: mainViewWidth / 2)

/// Main-view pixel of a world point, for a frame `width` × `height`.
func projectMain(_ p: SIMD3<Float>, width: Int, height: Int) -> SIMD2<Float> {
    mainCamera.project(p, viewCentre: SIMD2(Float(width) / 2, Float(height) / 2), viewHalf: Float(width) / 2)
}

/// The magnified view of the jaws: a circle in the frame (in units of the
/// frame's height), and an orthographic camera looking down on the jaws from
/// above and a little behind — the one direction nothing hides them from, as
/// they swing in the plane of the head. MODEL framing; the same scene, the
/// same kernel, only a closer camera.
let zoomCentre = SIMD2<Float>(1.395, 0.285)
let zoomRadius: Float = 0.245
/// Millimetres across the circle's diameter.
let zoomField: Float = 0.9
let zoomLookAt = SIMD3<Float>(0.08, 0.36, 0)
let zoomDirection: SIMD3<Float> = simd_normalize(SIMD3<Float>(0.12, -1.0, 0.04))
let zoomCamera = OrthoCamera(centre: zoomLookAt, forward: zoomDirection, halfWidth: zoomField / 2)

/// Magnified-view pixel of a world point.
func projectZoom(_ p: SIMD3<Float>, height: Int) -> SIMD2<Float> {
    let h: Float = Float(height)
    return zoomCamera.project(p, viewCentre: zoomCentre * h, viewHalf: zoomRadius * h)
}

/// How much bigger the magnified view draws things than the main view.
func zoomMagnification(width: Int, height: Int) -> Float {
    let diameter: Float = 2 * zoomRadius
    let zoomPxPerMM: Float = diameter * Float(height) / zoomField
    let mainPxPerMM: Float = Float(width) / mainViewWidth
    return zoomPxPerMM / mainPxPerMM
}

// MARK: - light

/// A softbox above and in front-left, and a dimmer fill from beside the
/// camera. Radiance = irradiance / solid angle, as in step 20, so a mirror
/// highlight is exactly as bright as the light that also lights the diffuse.
let keyDirection: SIMD3<Float> = simd_normalize(SIMD3<Float>(-0.35, 1.0, 0.50))
let fillDirection: SIMD3<Float> = simd_normalize(-mainDirection + SIMD3<Float>(0.3, 0.6, 0))
let keyColour = SIMD3<Float>(1.0, 0.985, 0.96) * 1.6
let fillColour = SIMD3<Float>(0.96, 0.97, 1.0) * 0.55
let keyDiscCosine: Float = 0.975     // about 13°: a big box, a soft but defined highlight
let fillDiscCosine: Float = 0.985

func discRadiance(_ irradiance: SIMD3<Float>, cosine: Float) -> SIMD3<Float> {
    irradiance / (2 * Float.pi * (1 - cosine))
}

/// The table: a plain white card. MODEL albedo, 0.80.
let tableAlbedo: Float = 0.80

// MARK: - GPU layout

/// One shape: six float4s, so nothing pads differently in Swift and Metal.
struct GPUShape {
    var a: SIMD4<Float>      // end a (or centre), radius a
    var b: SIMD4<Float>      // end b (or semi-axes), radius b
    var x: SIMD4<Float>      // ellipsoid x axis
    var y: SIMD4<Float>      // ellipsoid y axis
    var meta: SIMD4<Float>   // kind, part, light, blend group
    var bound: SIMD4<Float>  // bounding sphere
}

func blendGroup(_ s: Shape) -> Float {
    switch s.part {
    case .mesosoma: return 1
    case .head, .mandible: return 2
    default: return 0
    }
}

func gpuShapes(_ shapes: [Shape]) -> [GPUShape] {
    shapes.map { s in
        let bound: SIMD4<Float>
        if s.kind == .roundCone {
            let c: SIMD3<Float> = (s.a + s.b) / 2
            bound = SIMD4<Float>(c, simd_distance(s.a, s.b) / 2 + max(s.ra, s.rb))
        } else if s.kind == .ellipsoid {
            bound = SIMD4<Float>(s.a, max(s.b.x, max(s.b.y, s.b.z)))
        } else {
            bound = SIMD4<Float>(s.a, simd_length(s.b) + s.ra)
        }
        return GPUShape(a: SIMD4<Float>(s.a, s.ra), b: SIMD4<Float>(s.b, s.rb),
                        x: SIMD4<Float>(s.xAxis, 0), y: SIMD4<Float>(s.yAxis, 0),
                        meta: SIMD4<Float>(Float(s.kind.rawValue), Float(s.part.rawValue), s.light ? 1 : 0, blendGroup(s)),
                        bound: bound)
    }
}

/// The apple for the GPU: (centre, rounding), (half extents, film), (u axis
/// x, z, v axis x, z) — the piece's own axes, each 45° from the ant's heading.
func gpuFood(_ m: ApplePiece) -> [SIMD4<Float>] {
    let u: SIMD3<Float> = pieceU
    let v: SIMD3<Float> = pieceV
    return [SIMD4<Float>(m.centre, m.rounding), SIMD4<Float>(m.half, m.film), SIMD4<Float>(u.x, u.z, v.x, v.z)]
}

/// The ants flattened for the GPU: every shape, grouped by ant; per ant a
/// bounding sphere and (first shape, count). Step 44's FrameData.
func gpuAnts(_ ants: [[Shape]]) -> (shapes: [GPUShape], index: [SIMD4<Float>]) {
    var shapes: [GPUShape] = []
    var index: [SIMD4<Float>] = []
    for a in ants {
        var c = SIMD3<Float>(0, 0, 0)
        for x in a { c += x.bound.centre }
        c /= Float(max(a.count, 1))
        var r: Float = 0
        for x in a { r = max(r, simd_distance(x.bound.centre, c) + x.bound.radius) }
        index.append(SIMD4<Float>(c, r))
        index.append(SIMD4<Float>(Float(shapes.count), Float(a.count), 0, 0))
        shapes += gpuShapes(a)
    }
    return (shapes, index)
}

// MARK: - the kernel

/// How far a ray may trust a distance. The ellipsoid distances are
/// approximations; the tests measure how much they over-report.
let stepScale: Float = 0.75

func metal(_ v: SIMD3<Float>) -> String { "float3(\(v.x), \(v.y), \(v.z))" }
func kernelSource() -> String {
    let mc: OrthoCamera = mainCamera
    return """
    #include <metal_stdlib>
    using namespace metal;

    struct Shape { float4 a; float4 b; float4 x; float4 y; float4 meta; float4 bound; };
    struct Params { uint width; uint height; uint rowOffset; uint samples; uint colOffset; uint colEnd; uint rowEnd; uint ants; };

    constant float STEP_SCALE = \(stepScale);

    constant float3 M_CENTRE = \(metal(mc.centre));
    constant float3 M_FWD = \(metal(mc.forward));
    constant float3 M_RIGHT = \(metal(mc.right));
    constant float3 M_UP = \(metal(mc.up));
    constant float M_HALF = \(mc.halfWidth);

    constant float2 Z_C = float2(\(zoomCentre.x), \(zoomCentre.y));
    constant float Z_R = \(zoomRadius);
    constant float3 Z_CENTRE = \(metal(zoomCamera.centre));
    constant float3 Z_FWD = \(metal(zoomCamera.forward));
    constant float3 Z_RIGHT = \(metal(zoomCamera.right));
    constant float3 Z_UP = \(metal(zoomCamera.up));
    constant float Z_HALF = \(zoomCamera.halfWidth);

    constant float3 KEY_DIR = \(metal(keyDirection));
    constant float3 FILL_DIR = \(metal(fillDirection));
    constant float3 KEY_COL = \(metal(keyColour));
    constant float3 FILL_COL = \(metal(fillColour));
    constant float3 KEY_RAD = \(metal(discRadiance(keyColour, cosine: keyDiscCosine)));
    constant float3 FILL_RAD = \(metal(discRadiance(fillColour, cosine: fillDiscCosine)));
    constant float KEY_DISC = \(keyDiscCosine);
    constant float FILL_DISC = \(fillDiscCosine);

    constant float TABLE = \(tableAlbedo);
    constant float CUTICLE_F0 = \(cuticleF0);
    constant float3 FLESH = \(metal(fleshAlbedo));
    constant float CELL = \(fleshCellSpacing);
    constant float JUICE_F0 = \(juiceF0);

    // Cuticle colours, linear sRGB. MODEL: Seifert (2020) describes the
    // L. niger worker as dark brown on head and gaster, with the mandibles
    // and scape "yellowish-reddish brown"; these are those words as numbers.
    // PALE is lighter than step 26's (0.048, 0.027, 0.015), which rendered
    // as a near-black brown, and is drawn with a satin rather than glassy
    // finish (PALE_ALPHA), so a scape seen at a grazing angle shows its own
    // colour instead of mirroring the white table.
    constant float3 DARK = float3(0.011, 0.0082, 0.0066);
    constant float3 PALE = float3(0.105, 0.048, 0.020);
    constant float PALE_ALPHA = 0.34;
    constant float3 EYE = float3(0.014, 0.013, 0.013);

    // ------------------------------------------------------------ helpers

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

    // Exact distance to a round cone (Quílez).
    float roundCone(float3 p, float3 a, float3 b, float r1, float r2) {
        float3 ba = b - a;
        float l2 = dot(ba, ba);
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
        if (sign(z) * a2 * z2 > k) return sqrt(x2 + z2) * il2 - r2;
        if (sign(y) * a2 * y2 < k) return sqrt(x2 + y2) * il2 - r1;
        return (sqrt(x2 * a2 * il2) + y * rr) * il2 - r1;
    }

    // A rounded box with its own axes: exact outside.
    float roundBox(float3 p, float3 c, float3 b, float r, float3 xa, float3 ya) {
        float3 q0 = p - c;
        float3 za = cross(xa, ya);
        float3 q = abs(float3(dot(q0, xa), dot(q0, ya), dot(q0, za))) - b;
        return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0) - r;
    }

    // An ellipsoid's distance, Quílez's bound-like approximation.
    float ellipsoid(float3 p, float3 c, float3 r, float3 xa, float3 ya) {
        float3 q0 = p - c;
        float3 za = cross(xa, ya);
        float3 q = float3(dot(q0, xa), dot(q0, ya), dot(q0, za));
        float k0 = length(q / r);
        float k1 = length(q / (r * r));
        return k0 * (k0 - 1.0) / max(k1, 1e-6);
    }

    // ------------------------------------------------------------ the ants (mm)

    // One ant: step 26's three blend groups (step 44's). Round cones are
    // exact, so one far from the current best is skipped; ellipsoids never are.
    float oneAnt(float3 p, constant Shape *S, int first, int count, thread int &part, thread int &light, thread int &idx) {
        float g0 = 1e9, g1 = 1e9, g2 = 1e9;
        float best = 1e9;
        for (int i = first; i < first + count; i++) {
            Shape s = S[i];
            int group = int(s.meta.w + 0.5);
            float k = group == 1 ? 0.10 : (group == 2 ? 0.04 : 0.012);
            float cur = group == 1 ? g1 : (group == 2 ? g2 : g0);
            float d;
            if (s.meta.x < 0.5) {
                float bd = length(p - s.bound.xyz) - s.bound.w;
                if (bd > cur + k && bd > best) continue;
                d = roundCone(p, s.a.xyz, s.b.xyz, s.a.w, s.b.w);
            } else if (s.meta.x < 1.5) {
                d = ellipsoid(p, s.a.xyz, s.b.xyz, s.x.xyz, s.y.xyz);
            } else {
                d = roundBox(p, s.a.xyz, s.b.xyz, s.a.w, s.x.xyz, s.y.xyz);
            }
            if (d < best) { best = d; part = int(s.meta.y + 0.5); light = int(s.meta.z + 0.5); idx = i; }
            if (group == 1) g1 = smin(g1, d, k); else if (group == 2) g2 = smin(g2, d, k); else g0 = smin(g0, d, k);
        }
        return smin(smin(g1, g2, 0.03), g0, 0.012);
    }

    // All the ants: the nearest one wins; an ant whose bounding sphere is
    // further than the best so far (less the most its blends can pull a
    // surface out, 0.03 mm) is skipped. Step 44's. `cap`: beyond it the
    // caller does not care, and cap comes back.
    float antsSDF(float3 p, constant Shape *S, constant float4 *N, int ants, float cap, thread int &part, thread int &light, thread int &idx) {
        float best = cap;
        part = 0; light = 0; idx = -1;
        for (int k = 0; k < ants; k++) {
            float4 bs = N[2 * k];
            float bd = length(p - bs.xyz) - bs.w;
            if (bd - 0.03 > best) continue;
            int pt, lt, ix;
            float d = oneAnt(p, S, int(N[2 * k + 1].x + 0.5), int(N[2 * k + 1].y + 0.5), pt, lt, ix);
            if (d < best) { best = d; part = pt; light = lt; idx = ix; }
        }
        return best;
    }

    // ------------------------------------------------------------ the apple (mm)

    // Food.swift's pieceSDF, the same arithmetic: a rounded box, exact
    // outside. G[0] = (centre, rounding), G[1] = (half extents, film),
    // G[2] = (u axis x, z, v axis x, z). Its surface is the juice's.
    float3 pieceLocal(float3 p, constant float4 *G) {
        float3 d = p - G[0].xyz;
        return float3(d.x * G[2].x + d.z * G[2].y, d.x * G[2].z + d.z * G[2].w, d.y);
    }
    float foodSDF(float3 p, constant float4 *G) {
        float r = G[0].w;
        float3 q = abs(pieceLocal(p, G)) - (G[1].xyz - float3(r));
        return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0) - r;
    }

    // Materials: 1 table, 2 ant, 3 apple. `idx`: the ant shape hit.
    float mainSDF(float3 p, constant Shape *S, constant float4 *N, int ants, constant float4 *G, thread int &mat, thread int &sub, thread int &light, thread int &idx) {
        float d = p.y;
        mat = 1; sub = 0; light = 0; idx = -1;
        float g = foodSDF(p, G);
        if (g < d) { d = g; mat = 3; }
        int part, lt, ix;
        float a = antsSDF(p, S, N, ants, max(d, 0.0) + 0.001, part, lt, ix);
        if (a < d) { d = a; mat = 2; sub = part; light = lt; idx = ix; }
        return d;
    }

    float mainDist(float3 p, constant Shape *S, constant float4 *N, int ants, constant float4 *G) {
        int m, s, l, i;
        return mainSDF(p, S, N, ants, G, m, s, l, i);
    }

    float3 mainNormal(float3 p, constant Shape *S, constant float4 *N, int ants, constant float4 *G) {
        const float e = 0.0015;
        float3 k1 = float3(1, -1, -1), k2 = float3(-1, -1, 1), k3 = float3(-1, 1, -1), k4 = float3(1, 1, 1);
        return normalize(k1 * mainDist(p + k1 * e, S, N, ants, G) + k2 * mainDist(p + k2 * e, S, N, ants, G)
                       + k3 * mainDist(p + k3 * e, S, N, ants, G) + k4 * mainDist(p + k4 * e, S, N, ants, G));
    }

    bool marchMain(float3 ro, float3 rd, constant Shape *S, constant float4 *N, int ants, constant float4 *G,
                   thread float &t, thread int &mat, thread int &sub, thread int &light, thread int &ix) {
        t = 0.0;
        for (int i = 0; i < 900; i++) {
            float3 p = ro + rd * t;
            float d = mainSDF(p, S, N, ants, G, mat, sub, light, ix);
            if (d < 0.0006) return true;
            t += max(d * STEP_SCALE, 0.0003);
            if (t > 60.0) break;
        }
        return false;
    }

    // Shadows: the ant and the kernel both block the light.
    float softShadow(float3 ro, float3 rd, constant Shape *S, constant float4 *N, int ants, constant float4 *G) {
        float res = 1.0;
        float t = 0.004;
        for (int i = 0; i < 100; i++) {
            float3 p = ro + rd * t;
            int part, lt, ix;
            float a = min(antsSDF(p, S, N, ants, 1.0, part, lt, ix), foodSDF(p, G));
            res = min(res, 8.0 * a / t);
            t += clamp(a * STEP_SCALE, 0.002, 0.2);
            if (res < 0.002 || t > 6.0) break;
        }
        return clamp(res, 0.0, 1.0);
    }

    float occlusion(float3 p, float3 n, constant Shape *S, constant float4 *N, int ants, constant float4 *G, float scale) {
        float occ = 0.0, w = 1.0;
        for (int i = 1; i <= 5; i++) {
            float h = scale * (0.02 + 0.06 * float(i * i));
            occ += w * clamp((h - mainDist(p + n * h, S, N, ants, G)) / h, 0.0, 1.0);
            w *= 0.7;
        }
        return clamp(1.0 - 0.5 * occ, 0.0, 1.0);
    }

    // ------------------------------------------------------------ light

    float3 environment(float3 d) {
        // A pale studio: bright above, the white table's glow below.
        // A studio: the white table below, dim surroundings above, and the
        // two softboxes.
        float up = smoothstep(-0.15, 0.25, d.y);
        float3 c = mix(float3(0.62, 0.61, 0.60), float3(0.24, 0.24, 0.25), up);
        float kSoft = (1.0 - KEY_DISC) * 0.35;
        float fSoft = (1.0 - FILL_DISC) * 0.35;
        c += KEY_RAD * smoothstep(KEY_DISC - kSoft, KEY_DISC + kSoft, dot(d, KEY_DIR));
        c += FILL_RAD * smoothstep(FILL_DISC - fSoft, FILL_DISC + fSoft, dot(d, FILL_DIR));
        return c;
    }

    // The inset's surroundings: brighter, so the micrometre world sits on
    // the same plain light ground as the rest of the page.
    float3 envInset(float3 d) {
        float up = smoothstep(-0.3, 0.6, d.y);
        float3 c = mix(float3(0.70, 0.71, 0.72), float3(0.90, 0.91, 0.92), up);
        float kSoft = (1.0 - KEY_DISC) * 0.35;
        c += KEY_RAD * smoothstep(KEY_DISC - kSoft, KEY_DISC + kSoft, dot(d, KEY_DIR));
        return c;
    }

    float3 irradiance(float3 n) {
        return mix(float3(0.50, 0.49, 0.48), float3(0.42, 0.42, 0.44), 0.5 + 0.5 * n.y);
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
        float Gs = (ndv / (ndv * (1.0 - k) + k)) * (ndl / (ndl * (1.0 - k) + k));
        float F = f0 + (1.0 - f0) * pow(1.0 - vdh, 5.0);
        return D * Gs * F / (4.0 * ndv * ndl + 1e-4) * ndl;
    }

    float schlick(float f0, float c) { return f0 + (1.0 - f0) * pow(clamp(1.0 - c, 0.0, 1.0), 5.0); }

    // Glossy dielectric over a coloured body: cuticle in both views.
    float3 shadeCuticle(float3 n, float3 v, float3 albedo, float sh, float occ, float alpha, float f0, bool inset) {
        float ndv = max(dot(n, v), 1e-3);
        float key = max(dot(n, KEY_DIR), 0.0) * sh;
        float fill = max(dot(n, FILL_DIR), 0.0);
        float F = schlick(f0, ndv);
        float3 diffuse = albedo * (KEY_COL * key + FILL_COL * fill + irradiance(n) * occ) * (1.0 - F);
        float3 r = reflect(-v, n);
        float3 mirror = inset ? envInset(r) * 0.6 : environment(r);
        if (dot(r, KEY_DIR) > KEY_DISC - 0.02) mirror = mix(irradiance(r), mirror, sh);
        float3 spec = mirror * F * mix(1.0, occ, 0.6) * (alpha < 0.2 ? 1.0 : (alpha < 0.3 ? 0.5 : 0.22));
        spec += KEY_COL * ggx(n, v, KEY_DIR, alpha * 1.6, f0) * sh * 0.3 + FILL_COL * ggx(n, v, FILL_DIR, alpha * 1.6, f0) * 0.3;
        return diffuse + spec;
    }

    // ------------------------------------------------------------ main view shading

    float3 shadeTable(float3 p, constant Shape *S, constant float4 *N, int ants, constant float4 *G) {
        float sh = softShadow(p + float3(0, 0.002, 0), KEY_DIR, S, N, ants, G);
        float occ = occlusion(p, float3(0, 1, 0), S, N, ants, G, 1.0);
        float grain = 0.97 + 0.03 * noise3(p * 60.0);
        float3 alb = float3(TABLE) * grain * float3(1.0, 0.99, 0.975);
        return alb * (KEY_COL * max(KEY_DIR.y, 0.0) * sh + FILL_COL * max(FILL_DIR.y, 0.0) + irradiance(float3(0, 1, 0)) * occ * 0.55);
    }

    float3 shadeAnt(float3 p, float3 rd, int part, int light, int idx, constant Shape *S, constant float4 *N, int ants, constant float4 *G) {
        float3 n = mainNormal(p, S, N, ants, G);
        float3 v = -rd;
        float alpha = light == 1 ? PALE_ALPHA : 0.16;
        float3 alb = light == 1 ? PALE : DARK;
        if (part == 7) {
            // Compound eye: a hexagonal field of facets, each a tiny lens.
            float3 bump = noiseGrad(p * 200.0);
            n = normalize(n + 0.12 * (bump - n * dot(bump, n)));
            alb = EYE;
            alpha = 0.07;
        } else {
            // Fine reticulate microsculpture: a satin sheen, not a mirror.
            float3 bump = noiseGrad(p * 260.0);
            n = normalize(n + 0.06 * (bump - n * dot(bump, n)));
        }
        if (part == 4 && idx >= 0 && S[idx].meta.x > 0.5 && S[idx].meta.x < 1.5) {
            // Tergite margins (step 44's, in this gaster's own frame): where
            // each plate's hind edge laps the next, a narrow step.
            Shape g = S[idx];
            float x = dot(p - g.a.xyz, g.x.xyz) / g.b.x;
            float m = 0.0;
            for (int i = 0; i < 3; i++) {
                float at = 0.42 - 0.40 * float(i);
                m = max(m, exp(-pow((x - at) / 0.022, 2.0)));
            }
            alb *= 1.0 - 0.45 * m;
            n = normalize(n + g.x.xyz * 0.35 * m);
        }
        float sh = softShadow(p + n * 0.003, KEY_DIR, S, N, ants, G);
        float occ = occlusion(p, n, S, N, ants, G, 0.6);
        return shadeCuticle(n, v, alb, sh, occ, alpha, CUTICLE_F0, false);
    }

    // Cells: distance to the nearest of jittered lattice points, and to the
    // second nearest — the cell walls are where the two are equal.
    float2 cells(float3 p) {
        float3 i = floor(p);
        float3 f = fract(p);
        float d1 = 8.0, d2 = 8.0;
        for (int z = -1; z <= 1; z++) {
            for (int y = -1; y <= 1; y++) {
                for (int x = -1; x <= 1; x++) {
                    float3 g = float3(x, y, z);
                    float3 o = float3(hash3(i + g), hash3(i + g + 17.3), hash3(i + g + 41.1));
                    float d = length(g + o - f);
                    if (d < d1) { d2 = d1; d1 = d; } else if (d < d2) { d2 = d; }
                }
            }
        }
        return float2(d1, d2);
    }

    // The apple: peeled, so cut flesh everywhere, under a glossy film of its
    // juice (water's gloss). The flesh is Golden Delicious's measured colour,
    // translucent — light wraps a little into it — with a faint mottle of
    // cells at the measured 0.28 mm spacing (Food.swift).
    float3 shadeFood(float3 p, float3 rd, constant Shape *S, constant float4 *N, int ants, constant float4 *G) {
        float3 n = mainNormal(p, S, N, ants, G);
        float3 v = -rd;
        float sh = softShadow(p + n * 0.004, KEY_DIR, S, N, ants, G);
        float occ = occlusion(p, n, S, N, ants, G, 0.6);
        float fill = max(dot(n, FILL_DIR), 0.0);
        float wrap = 0.3;
        float keyW = max((dot(n, KEY_DIR) + wrap) / (1.0 + wrap), 0.0) * sh;
        float2 c = cells(p / CELL);
        float wall = 1.0 - smoothstep(0.0, 0.10, c.y - c.x);
        float3 alb = FLESH * (0.95 + 0.05 * noise3(p * 6.0)) * (1.0 - 0.10 * wall);
        float3 col = alb * (KEY_COL * keyW + FILL_COL * fill + irradiance(n) * occ * 1.1) * mix(0.62, 1.0, occ);
        col += alb * alb * KEY_COL * 0.10 * occ;
        // The juice's surface follows the cut cells a little, so the gloss
        // breaks up instead of mirroring the studio whole.
        float3 g = noiseGrad(p / CELL * 1.3);
        float3 bn = normalize(n + 0.10 * (g - n * dot(g, n)));
        float F = schlick(JUICE_F0, max(dot(bn, v), 0.0));
        float3 rr = reflect(rd, bn);
        float3 refl = environment(rr);
        if (dot(rr, KEY_DIR) > KEY_DISC - 0.02) refl = mix(irradiance(rr), refl, sh);
        col = col * (1.0 - F) + refl * F * mix(1.0, occ, 0.5);
        col += KEY_COL * ggx(bn, v, KEY_DIR, 0.08, JUICE_F0) * sh * 0.35 + FILL_COL * ggx(bn, v, FILL_DIR, 0.08, JUICE_F0) * 0.3;
        return col;
    }

    float3 shadeMain(float3 ro, float3 rd, constant Shape *S, constant float4 *N, int ants, constant float4 *G, thread float4 &aux) {
        float t;
        int mat, sub, light, idx;
        aux = float4(0.0);
        if (!marchMain(ro, rd, S, N, ants, G, t, mat, sub, light, idx)) return environment(rd);
        float3 p = ro + rd * t;
        aux = float4(1.0, float(mat), float(sub), float(idx));
        if (mat == 1) return shadeTable(p, S, N, ants, G);
        if (mat == 2) return shadeAnt(p, rd, sub, light, idx, S, N, ants, G);
        return shadeFood(p, rd, S, N, ants, G);
    }

    // ------------------------------------------------------------ output

    // A filmic shoulder on luminance, hue kept (step 20's curve).
    float3 toneMap(float3 x) {
        float L = dot(x, float3(0.2126, 0.7152, 0.0722));
        float Lm = L * (1.0 + L / 16.0) / (1.0 + L);
        float3 c = x * (Lm / max(L, 1e-5)) * 1.18;
        float over = max(max(c.r, max(c.g, c.b)) - 1.0, 0.0);
        c = mix(c, float3(1.0), clamp(over, 0.0, 1.0));
        c = clamp(c, 0.0, 1.0);
        return select(1.055 * pow(c, 1.0 / 2.4) - 0.055, 12.92 * c, c <= 0.0031308);
    }

    kernel void ant(device uchar4 *pixels [[buffer(0)]],
                    device float4 *auxOut [[buffer(1)]],
                    constant Params &P [[buffer(2)]],
                    constant Shape *S [[buffer(3)]],
                    constant float4 *G [[buffer(4)]],
                    constant float4 *AI [[buffer(5)]],
                    uint2 gid [[thread_position_in_grid]]) {
        uint x = gid.x + P.colOffset;
        uint y = gid.y + P.rowOffset;
        if (x >= P.colEnd || y >= P.rowEnd) return;
        float W = float(P.width), Hh = float(P.height);
        int ants = int(P.ants);
        float3 sum = float3(0.0);
        uint NS = P.samples;
        float4 aux = float4(0.0);
        for (uint sy = 0; sy < NS; sy++) {
            for (uint sx = 0; sx < NS; sx++) {
                float2 px = float2(x, y) + float2((float(sx) + 0.5) / float(NS), (float(sy) + 0.5) / float(NS));
                float4 a;
                float3 c;
                float2 f = px / Hh;
                if (length(f - Z_C) < Z_R) {
                    // The magnified view: the same scene through a closer camera.
                    float2 z = (f - Z_C) / Z_R;
                    float3 ro = Z_CENTRE + (Z_RIGHT * z.x - Z_UP * z.y) * Z_HALF - Z_FWD * 20.0;
                    c = shadeMain(ro, Z_FWD, S, AI, ants, G, a);
                    a.x = a.x > 0.5 ? 2.0 : 0.0;
                } else {
                    float2 s = float2(2.0 * px.x / W - 1.0, (1.0 - 2.0 * px.y / Hh) * (Hh / W));
                    float3 ro = M_CENTRE + (M_RIGHT * s.x + M_UP * s.y) * M_HALF - M_FWD * 20.0;
                    c = shadeMain(ro, M_FWD, S, AI, ants, G, a);
                }
                sum += toneMap(c);
                if (sx == NS / 2 && sy == NS / 2) aux = a;
            }
        }
        float3 c = sum / float(NS * NS);
        pixels[y * P.width + x] = uchar4(uchar3(round(clamp(c, 0.0, 1.0) * 255.0)), 255);
        auxOut[y * P.width + x] = aux;
    }

    // The distance functions at arbitrary points, from the same source the
    // picture is drawn with: (main distance, main material, the apple alone,
    // the ant alone).
    kernel void probe(device const float4 *points [[buffer(0)]],
                      device float4 *out [[buffer(1)]],
                      constant Shape *S [[buffer(3)]],
                      constant float4 *G [[buffer(4)]],
                      constant float4 *AI [[buffer(5)]],
                      constant uint &antCount [[buffer(6)]],
                      uint id [[thread_position_in_grid]]) {
        float3 p = points[id].xyz;
        int ants = int(antCount);
        int m, s, l, ix, part, lt;
        float dm = mainSDF(p, S, AI, ants, G, m, s, l, ix);
        out[id] = float4(dm, float(m), foodSDF(p, G), antsSDF(p, S, AI, ants, 1e9, part, lt, ix));
    }
    """
}

// MARK: - running it

enum AntError: Error, CustomStringConvertible {
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

/// Step 8's GIF encoder names its errors RenderError; here they are AntError.
typealias RenderError = AntError

struct Params {
    var width: UInt32
    var height: UInt32
    var rowOffset: UInt32
    var samples: UInt32
    var colOffset: UInt32
    var colEnd: UInt32
    var rowEnd: UInt32
    var ants: UInt32
}

/// What never moves, built once.
struct Scene {
    let mutant: Mutant
    let piece: ApplePiece

    init(mutant: Mutant) {
        self.mutant = mutant
        piece = buildPiece(mutant: mutant)
    }

    var source: String { kernelSource() }

    /// Everything that moves, at time `t` seconds into the loop.
    func frame(at t: Float) -> FrameState {
        FrameState(time: t, ant: bitingAnt(time: t, mutant: mutant), gut: gutState(time: t, mutant: mutant))
    }
}

/// One moment of the loop.
struct FrameState {
    var time: Float
    var ant: BitingAnt
    var gut: GutState               // the inset's juice and bits of flesh
}

/// The finished picture, and what each pixel's centre sample saw:
/// (1 main-view hit / 2 magnified-view hit, material 1 card / 2 ant / 3 apple, part, shape index).
struct AntImage {
    let width: Int
    let height: Int
    let pixels: MTLBuffer
    let aux: MTLBuffer

    func rgba(_ x: Int, _ y: Int) -> SIMD4<UInt8> {
        let p = pixels.contents().advanced(by: (y * width + x) * 4).assumingMemoryBound(to: UInt8.self)
        return SIMD4<UInt8>(p[0], p[1], p[2], p[3])
    }

    func seen(_ x: Int, _ y: Int) -> SIMD4<Float> {
        aux.contents().assumingMemoryBound(to: SIMD4<Float>.self)[y * width + x]
    }
}

func findDevice() throws -> MTLDevice {
    if let d = MTLCreateSystemDefaultDevice() ?? MTLCopyAllDevices().first { return d }
    throw AntError.noMetalDevice
}

func makeLibrary(_ device: MTLDevice, _ scene: Scene) throws -> MTLLibrary {
    let options = MTLCompileOptions()
    // Precise maths, as in every step: pores 0.08 µm wide are carved out of
    // distances a few µm long.
    options.fastMathEnabled = false
    do {
        return try device.makeLibrary(source: scene.source, options: options)
    } catch {
        throw AntError.kernelCompile("\(error)")
    }
}

func pipeline(_ device: MTLDevice, _ library: MTLLibrary, _ name: String) throws -> MTLComputePipelineState {
    guard let f = library.makeFunction(name: name) else { throw AntError.kernelCompile("no kernel \(name)") }
    do { return try device.makeComputePipelineState(function: f) } catch {
        throw AntError.kernelCompile("\(error)")
    }
}

func buffer<T>(_ device: MTLDevice, _ array: [T]) throws -> MTLBuffer {
    let length: Int = max(MemoryLayout<T>.stride * array.count, 16)
    let made: MTLBuffer? = array.withUnsafeBytes { raw -> MTLBuffer? in
        guard let base = raw.baseAddress else { return device.makeBuffer(length: length, options: .storageModeShared) }
        return device.makeBuffer(bytes: base, length: length, options: .storageModeShared)
    }
    guard let b = made else { throw AntError.gpu("could not allocate a buffer") }
    return b
}

/// The scene's buffers for one frame, in kernel argument order 3…5, and the
/// number of ants (one).
func sceneBuffers(_ device: MTLDevice, _ scene: Scene, _ frame: FrameState) throws -> (buffers: [MTLBuffer], ants: Int) {
    let ants = gpuAnts([frame.ant.shapes])
    return ([try buffer(device, ants.shapes), try buffer(device, gpuFood(scene.piece)), try buffer(device, ants.index)], 1)
}

/// A pixel rectangle, [x0, x1) × [y0, y1).
struct PixelRect {
    var x0: Int
    var y0: Int
    var x1: Int
    var y1: Int

    func clipped(width: Int, height: Int) -> PixelRect {
        PixelRect(x0: max(x0, 0), y0: max(y0, 0), x1: min(x1, width), y1: min(y1, height))
    }
    var area: Int { max(x1 - x0, 0) * max(y1 - y0, 0) }
}

/// One compiled kernel and one frame-sized pair of buffers, reused for every
/// frame of the loop.
final class Renderer {
    let device: MTLDevice
    let scene: Scene
    let width: Int
    let height: Int
    let pixels: MTLBuffer
    let aux: MTLBuffer
    private let pso: MTLComputePipelineState
    private let queue: MTLCommandQueue

    init(device: MTLDevice, scene: Scene, width: Int, height: Int) throws {
        self.device = device
        self.scene = scene
        self.width = width
        self.height = height
        let library = try makeLibrary(device, scene)
        pso = try pipeline(device, library, "ant")
        guard let p = device.makeBuffer(length: width * height * 4, options: .storageModeShared),
              let a = device.makeBuffer(length: width * height * 16, options: .storageModeShared),
              let q = device.makeCommandQueue()
        else { throw AntError.gpu("could not allocate buffers") }
        pixels = p
        aux = a
        queue = q
    }

    var image: AntImage { AntImage(width: width, height: height, pixels: pixels, aux: aux) }

    /// Render `frame` into the buffers — the whole picture, or only `region`.
    /// In bands, one command buffer each, so no single piece of GPU work runs
    /// long enough to trip the watchdog. Returns GPU seconds.
    @discardableResult
    func render(_ frame: FrameState, samples: Int, region given: PixelRect? = nil) throws -> Double {
        let region: PixelRect = (given ?? PixelRect(x0: 0, y0: 0, x1: width, y1: height)).clipped(width: width, height: height)
        if region.area == 0 { return 0 }
        let (bufs, ants) = try sceneBuffers(device, scene, frame)
        // Step 33's fix: a band is 24 full-width rows' worth of pixels, however
        // narrow the region.
        let regionWidth: Int = max(region.x1 - region.x0, 1)
        let band: Int = max(24, 24 * width / regionWidth)
        var gpu: Double = 0
        var row: Int = region.y0
        while row < region.y1 {
            let rows: Int = min(band, region.y1 - row)
            guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else {
                throw AntError.gpu("could not make a command buffer")
            }
            var params = Params(width: UInt32(width), height: UInt32(height), rowOffset: UInt32(row), samples: UInt32(samples),
                                colOffset: UInt32(region.x0), colEnd: UInt32(region.x1), rowEnd: UInt32(region.y1),
                                ants: UInt32(ants))
            enc.setComputePipelineState(pso)
            enc.setBuffer(pixels, offset: 0, index: 0)
            enc.setBuffer(aux, offset: 0, index: 1)
            enc.setBytes(&params, length: MemoryLayout<Params>.stride, index: 2)
            for (i, b) in bufs.enumerated() { enc.setBuffer(b, offset: 0, index: 3 + i) }
            let w: Int = pso.threadExecutionWidth
            let group = MTLSize(width: w, height: max(pso.maxTotalThreadsPerThreadgroup / w / 4, 1), depth: 1)
            enc.dispatchThreads(MTLSize(width: region.x1 - region.x0, height: rows, depth: 1), threadsPerThreadgroup: group)
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            if let e = cb.error { throw AntError.gpu(e.localizedDescription) }
            gpu += cb.gpuEndTime - cb.gpuStartTime
            row += rows
        }
        return gpu
    }
}

/// One frame, whole: for the tests.
func renderAnt(width: Int, height: Int, samples: Int, scene: Scene, frame: FrameState,
               on device: MTLDevice) throws -> (image: AntImage, gpuSeconds: Double) {
    let r = try Renderer(device: device, scene: scene, width: width, height: height)
    let gpu: Double = try r.render(frame, samples: samples)
    return (r.image, gpu)
}

/// One probe result.
struct Probe {
    var mainDistance: Float
    var mainMaterial: Int
    var food: Float
    var ant: Float
}

/// The scene's distances at arbitrary points, from the same kernel source the
/// render uses — so a test of the distance function is a test of the thing
/// that drew the picture, not of a copy of it.
func probeScene(_ points: [SIMD3<Float>], scene: Scene, frame: FrameState, library: MTLLibrary,
                on device: MTLDevice) throws -> [Probe] {
    let pso = try pipeline(device, library, "probe")
    let (bufs, ants) = try sceneBuffers(device, scene, frame)
    let pts: [SIMD4<Float>] = points.map { SIMD4<Float>($0, 0) }
    let pb: MTLBuffer = try buffer(device, pts)
    guard let ob = device.makeBuffer(length: 16 * max(pts.count, 1), options: .storageModeShared),
          let queue = device.makeCommandQueue(), let cb = queue.makeCommandBuffer(),
          let enc = cb.makeComputeCommandEncoder()
    else { throw AntError.gpu("could not set up the probe") }
    var count: UInt32 = UInt32(ants)
    enc.setComputePipelineState(pso)
    enc.setBuffer(pb, offset: 0, index: 0)
    enc.setBuffer(ob, offset: 0, index: 1)
    for i in 0..<3 { enc.setBuffer(bufs[i], offset: 0, index: 3 + i) }
    enc.setBytes(&count, length: 4, index: 6)
    enc.dispatchThreads(MTLSize(width: pts.count, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: min(pso.maxTotalThreadsPerThreadgroup, 256), height: 1, depth: 1))
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    if let e = cb.error { throw AntError.gpu(e.localizedDescription) }
    let o = ob.contents().assumingMemoryBound(to: SIMD4<Float>.self)
    return (0..<pts.count).map { i in
        Probe(mainDistance: o[i].x, mainMaterial: Int(o[i].y), food: o[i].z, ant: o[i].w)
    }
}

func savePNG(_ image: AntImage, to url: URL) throws {
    try? FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    guard let provider = CGDataProvider(dataInfo: nil, data: image.pixels.contents(), size: image.width * image.height * 4,
                                        releaseData: { _, _, _ in }),
          let space = CGColorSpace(name: CGColorSpace.sRGB),
          let cg = CGImage(width: image.width, height: image.height, bitsPerComponent: 8, bitsPerPixel: 32,
                           bytesPerRow: image.width * 4, space: space,
                           bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                           provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent),
          let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)
    else { throw AntError.png("could not set up the encoder for \(url.path)") }
    CGImageDestinationAddImage(dest, cg, nil)
    guard CGImageDestinationFinalize(dest) else { throw AntError.png("could not write \(url.path)") }
}
