// One frame of a file of ants on a trail (step 44's renderer, the frame
// widened for more ants), marched per pixel on the GPU. Every
// surface is a distance function, as in step 26; the ant's shapes, materials
// and light are step 32's. What is new is that the shapes change every frame
// — so they go to the GPU as a buffer each frame, grouped by ant, with a
// bounding sphere per ant so a ray far from an ant never looks at its parts —
// and the table carries the trail.
//
// The camera is ORTHOGRAPHIC and looks square across the trail, so a
// millimetre along the trail is the same number of pixels anywhere in the
// frame and the scale bar is true everywhere.

import CoreGraphics
import Foundation
import ImageIO
import Metal
import UniformTypeIdentifiers
import simd

// MARK: - framing

/// Frame proportions: height over width. MODEL, a wide strip — the file is
/// long and low.
let frameAspect: Float = 0.2
/// Millimetres across the full frame width, chosen so that exactly
/// `fileCount` ants have their bodies in view at every instant — one crosses
/// out on the right just as the next crosses in on the left. An ant takes
/// W / cruise + (its stops in view) to cross a frame W wide, and the ants
/// come one loop apart, so that crossing must take exactly `fileCount` loops:
/// W = fileCount · spacing − (stops in view) · cruise · dabLoss. (A test
/// counts them.)
let frameWorldWidth: Float = Float(fileCount) * spacing - Float(dabSites.count) * cruise * dabLoss
/// The camera looks from the ants' right side, 32° above the ground, square
/// to the trail. MODEL.
let cameraElevation: Float = 32 * Float.pi / 180
let cameraDirection = SIMD3<Float>(0, -sin(cameraElevation), -cos(cameraElevation))
/// The world point at the centre of the frame: on the trail, a little above
/// the ground, and slightly behind it so the ants sit below the title.
let frameCentre = SIMD3<Float>(0, 0.55, 0.5)

/// The x range the camera sees (the camera looks square to x).
let viewRange: ClosedRange<Float> = (-frameWorldWidth / 2)...(frameWorldWidth / 2)

/// An orthographic camera.
struct OrthoCamera {
    var centre: SIMD3<Float>
    var forward: SIMD3<Float>
    var right: SIMD3<Float>
    var up: SIMD3<Float>
    var halfWidth: Float

    init(centre: SIMD3<Float>, forward: SIMD3<Float>, halfWidth: Float) {
        self.centre = centre
        self.forward = forward
        self.right = simd_normalize(simd_cross(forward, SIMD3<Float>(0, 1, 0)))
        self.up = simd_cross(self.right, forward)
        self.halfWidth = halfWidth
    }
}

let camera = OrthoCamera(centre: frameCentre, forward: cameraDirection, halfWidth: frameWorldWidth / 2)

/// The camera a renderer uses: the one above — or, for the subpixel mutant,
/// one taking in 40× more trail, so the same motion shrinks below a pixel.
func cameraFor(_ mutant: Mutant) -> OrthoCamera {
    if mutant != .subpixel { return camera }
    let half: Float = 40 * frameWorldWidth / 2
    return OrthoCamera(centre: frameCentre, forward: cameraDirection, halfWidth: half)
}

/// World point to pixel, for a frame `width` × `height`.
func project(_ p: SIMD3<Float>, width: Int, height: Int, camera: OrthoCamera = camera) -> SIMD2<Float> {
    let q: SIMD3<Float> = p - camera.centre
    let half: Float = Float(width) / 2
    let sx: Float = simd_dot(q, camera.right) / camera.halfWidth
    let sy: Float = simd_dot(q, camera.up) / camera.halfWidth
    return SIMD2<Float>(half + sx * half, Float(height) / 2 - sy * half)
}

func millimetresPerPixel(width: Int) -> Float { frameWorldWidth / Float(width) }

// MARK: - light and materials (step 32's)

let keyDirection: SIMD3<Float> = simd_normalize(SIMD3<Float>(-0.35, 1.0, 0.50))
let fillDirection: SIMD3<Float> = simd_normalize(-cameraDirection + SIMD3<Float>(0.3, 0.6, 0))
let keyColour = SIMD3<Float>(1.0, 0.985, 0.96) * 1.6
let fillColour = SIMD3<Float>(0.96, 0.97, 1.0) * 0.55
let keyDiscCosine: Float = 0.975
let fillDiscCosine: Float = 0.985

func discRadiance(_ irradiance: SIMD3<Float>, cosine: Float) -> SIMD3<Float> {
    irradiance / (2 * Float.pi * (1 - cosine))
}

/// The table: a plain white card. MODEL albedo, 0.80 (step 26).
let tableAlbedo: Float = 0.80
/// Chitin, n = 1.56 (step 26): normal-incidence reflectance ((n−1)/(n+1))².
let cuticleIndex: Float = 1.56
let cuticleF0: Float = ((cuticleIndex - 1) / (cuticleIndex + 1)) * ((cuticleIndex - 1) / (cuticleIndex + 1))

/// The trail, drawn — it is invisible in life. A soft blue band along the
/// trail line (half-width σ, mm) and a stronger spot where a fresh mark is.
/// MODEL: colour and widths are chosen to read, not measured; the real trail
/// is dots a few ants wide.
let trailColour = SIMD3<Float>(0.30, 0.52, 0.95)
let trailSigma: Float = 0.24
let trailStrength: Float = 0.34
let markRadius: Float = 0.16
let markStrength: Float = 0.60

// MARK: - GPU layout

/// One shape: six float4s, so nothing pads differently in Swift and Metal.
struct GPUShape {
    var a: SIMD4<Float>
    var b: SIMD4<Float>
    var x: SIMD4<Float>
    var y: SIMD4<Float>
    var meta: SIMD4<Float>   // kind, part, light, blend group
    var bound: SIMD4<Float>
}

func blendGroup(_ s: Shape) -> Float {
    switch s.part {
    case .mesosoma: return 1
    case .head, .mandible: return 2
    default: return 0
    }
}

func gpuShape(_ s: Shape) -> GPUShape {
    let (c, r) = s.bound
    return GPUShape(a: SIMD4<Float>(s.a, s.ra), b: SIMD4<Float>(s.b, s.rb),
                    x: SIMD4<Float>(s.xAxis, 0), y: SIMD4<Float>(s.yAxis, 0),
                    meta: SIMD4<Float>(Float(s.kind.rawValue), Float(s.part.rawValue), s.light ? 1 : 0, blendGroup(s)),
                    bound: SIMD4<Float>(c, r))
}

/// A frame's scene flattened for the GPU: shapes grouped by ant; per ant a
/// bounding sphere and (first shape, count); the fresh marks.
struct FrameData {
    var shapes: [GPUShape] = []
    var ants: [SIMD4<Float>] = []     // pairs: bound sphere, then (first, count, 0, 0)
    var marks: [SIMD4<Float>] = []    // (x, z, strength, 0)
    var antCount: Int = 0

    init(ants list: [WorldAnt], marks m: [Mark]) {
        for a in list {
            let (c, r) = a.bound
            ants.append(SIMD4<Float>(c, r))
            ants.append(SIMD4<Float>(Float(shapes.count), Float(a.shapes.count), 0, 0))
            shapes += a.shapes.map(gpuShape)
        }
        antCount = list.count
        marks = m.map { SIMD4<Float>($0.point.x, $0.point.z, $0.strength, 0) }
    }
}

// MARK: - the kernel

/// How far a ray may trust a distance (step 26): the ellipsoid distances are
/// approximations, and the tests measure how much they over-report.
let stepScale: Float = 0.75

func metal(_ v: SIMD3<Float>) -> String { "float3(\(v.x), \(v.y), \(v.z))" }

func kernelSource(camera c: OrthoCamera) -> String {
    return """
    #include <metal_stdlib>
    using namespace metal;

    struct Shape { float4 a; float4 b; float4 x; float4 y; float4 meta; float4 bound; };
    struct Params { uint width; uint height; uint rowOffset; uint samples; uint ants; uint marks; };

    constant float STEP_SCALE = \(stepScale);
    constant float3 C_CENTRE = \(metal(c.centre));
    constant float3 C_FWD = \(metal(c.forward));
    constant float3 C_RIGHT = \(metal(c.right));
    constant float3 C_UP = \(metal(c.up));
    constant float C_HALF = \(c.halfWidth);

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
    constant float3 TRAIL = \(metal(trailColour));
    constant float TRAIL_SIGMA = \(trailSigma);
    constant float TRAIL_K = \(trailStrength);
    constant float MARK_R = \(markRadius);
    constant float MARK_K = \(markStrength);

    // Cuticle colours, linear sRGB (step 32): Seifert (2020) — dark brown,
    // mandibles and scape "yellowish-reddish brown".
    constant float3 DARK = float3(0.011, 0.0082, 0.0066);
    constant float3 PALE = float3(0.105, 0.048, 0.020);
    constant float PALE_ALPHA = 0.34;
    constant float3 EYE = float3(0.014, 0.013, 0.013);

    float smin(float a, float b, float k) {
        float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
        return mix(b, a, h) - k * h * (1.0 - h);
    }

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

    float roundBox(float3 p, float3 c, float3 b, float r, float3 xa, float3 ya) {
        float3 q0 = p - c;
        float3 za = cross(xa, ya);
        float3 q = abs(float3(dot(q0, xa), dot(q0, ya), dot(q0, za))) - b;
        return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0) - r;
    }

    float ellipsoid(float3 p, float3 c, float3 r, float3 xa, float3 ya) {
        float3 q0 = p - c;
        float3 za = cross(xa, ya);
        float3 q = float3(dot(q0, xa), dot(q0, ya), dot(q0, za));
        float k0 = length(q / r);
        float k1 = length(q / (r * r));
        return k0 * (k0 - 1.0) / max(k1, 1e-6);
    }

    // One ant: step 26's three blend groups. Round cones are exact, so one
    // far from the current best is skipped; ellipsoids never are.
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

    // All the ants: the nearest one wins. An ant whose bounding sphere is
    // further than the best so far (less the most its blends can pull a
    // surface out, 0.03 mm) cannot be nearer, and is skipped.
    // `cap`: the caller does not care about distances beyond it (the table
    // is already nearer, or the step would be clamped anyway), so an ant
    // further than that is never opened; if none is nearer, cap comes back.
    float antsSDF(float3 p, constant Shape *S, constant float4 *A, int ants, float cap, thread int &part, thread int &light, thread int &idx) {
        float best = cap;
        part = 0; light = 0; idx = -1;
        for (int k = 0; k < ants; k++) {
            float4 bs = A[2 * k];
            float bd = length(p - bs.xyz) - bs.w;
            if (bd - 0.03 > best) continue;
            int pt, lt, ix;
            float d = oneAnt(p, S, int(A[2 * k + 1].x + 0.5), int(A[2 * k + 1].y + 0.5), pt, lt, ix);
            if (d < best) { best = d; part = pt; light = lt; idx = ix; }
        }
        return best;
    }

    // Materials: 1 table, 2 ant.
    float mainSDF(float3 p, constant Shape *S, constant float4 *A, int ants, thread int &mat, thread int &part, thread int &light, thread int &idx) {
        float d = p.y;
        mat = 1; part = 0; light = 0; idx = -1;
        int pt, lt, ix;
        float a = antsSDF(p, S, A, ants, max(d, 0.0) + 0.001, pt, lt, ix);
        if (a < d) { d = a; mat = 2; part = pt; light = lt; idx = ix; }
        return d;
    }

    float mainDist(float3 p, constant Shape *S, constant float4 *A, int ants) {
        int m, s, l, i;
        return mainSDF(p, S, A, ants, m, s, l, i);
    }

    float3 mainNormal(float3 p, constant Shape *S, constant float4 *A, int ants) {
        const float e = 0.0015;
        float3 k1 = float3(1, -1, -1), k2 = float3(-1, -1, 1), k3 = float3(-1, 1, -1), k4 = float3(1, 1, 1);
        return normalize(k1 * mainDist(p + k1 * e, S, A, ants) + k2 * mainDist(p + k2 * e, S, A, ants)
                       + k3 * mainDist(p + k3 * e, S, A, ants) + k4 * mainDist(p + k4 * e, S, A, ants));
    }

    bool march(float3 ro, float3 rd, constant Shape *S, constant float4 *A, int ants,
               thread float &t, thread int &mat, thread int &part, thread int &light, thread int &idx) {
        t = 0.0;
        for (int i = 0; i < 360; i++) {
            float3 p = ro + rd * t;
            float d = mainSDF(p, S, A, ants, mat, part, light, idx);
            if (d < 0.0006) return true;
            t += max(d * STEP_SCALE, 0.0003);
            if (t > 60.0) break;
        }
        return false;
    }

    float softShadow(float3 ro, float3 rd, constant Shape *S, constant float4 *A, int ants) {
        float res = 1.0;
        float t = 0.004;
        for (int i = 0; i < 90; i++) {
            float3 p = ro + rd * t;
            int part, lt, ix;
            float a = antsSDF(p, S, A, ants, 1.0, part, lt, ix);
            res = min(res, 8.0 * a / t);
            t += clamp(a * STEP_SCALE, 0.002, 0.2);
            if (res < 0.002 || t > 6.0) break;
        }
        return clamp(res, 0.0, 1.0);
    }

    float occlusion(float3 p, float3 n, constant Shape *S, constant float4 *A, int ants, float scale) {
        float occ = 0.0, w = 1.0;
        for (int i = 1; i <= 5; i++) {
            float h = scale * (0.02 + 0.06 * float(i * i));
            occ += w * clamp((h - mainDist(p + n * h, S, A, ants)) / h, 0.0, 1.0);
            w *= 0.7;
        }
        return clamp(1.0 - 0.5 * occ, 0.0, 1.0);
    }

    float3 environment(float3 d) {
        float up = smoothstep(-0.15, 0.25, d.y);
        float3 c = mix(float3(0.62, 0.61, 0.60), float3(0.24, 0.24, 0.25), up);
        float kSoft = (1.0 - KEY_DISC) * 0.35;
        float fSoft = (1.0 - FILL_DISC) * 0.35;
        c += KEY_RAD * smoothstep(KEY_DISC - kSoft, KEY_DISC + kSoft, dot(d, KEY_DIR));
        c += FILL_RAD * smoothstep(FILL_DISC - fSoft, FILL_DISC + fSoft, dot(d, FILL_DIR));
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

    float3 shadeCuticle(float3 n, float3 v, float3 albedo, float sh, float occ, float alpha, float f0) {
        float ndv = max(dot(n, v), 1e-3);
        float key = max(dot(n, KEY_DIR), 0.0) * sh;
        float fill = max(dot(n, FILL_DIR), 0.0);
        float F = schlick(f0, ndv);
        float3 diffuse = albedo * (KEY_COL * key + FILL_COL * fill + irradiance(n) * occ) * (1.0 - F);
        float3 r = reflect(-v, n);
        float3 mirror = environment(r);
        if (dot(r, KEY_DIR) > KEY_DISC - 0.02) mirror = mix(irradiance(r), mirror, sh);
        float3 spec = mirror * F * mix(1.0, occ, 0.6) * (alpha < 0.2 ? 1.0 : (alpha < 0.3 ? 0.5 : 0.22));
        spec += KEY_COL * ggx(n, v, KEY_DIR, alpha * 1.6, f0) * sh * 0.3 + FILL_COL * ggx(n, v, FILL_DIR, alpha * 1.6, f0) * 0.3;
        return diffuse + spec;
    }

    // The table, with the trail drawn on it: a soft band along z = 0 and a
    // brighter spot at each fresh mark.
    float3 shadeTable(float3 p, constant Shape *S, constant float4 *A, int ants, constant float4 *M, int marks) {
        float sh = softShadow(p + float3(0, 0.002, 0), KEY_DIR, S, A, ants);
        // Occlusion reaches 1.6 mm up; further from every ant it is exactly 1.
        int pt, lt, ix;
        float near = antsSDF(p, S, A, ants, 2.0, pt, lt, ix);
        float occ = near >= 2.0 ? 1.0 : occlusion(p, float3(0, 1, 0), S, A, ants, 1.0);
        float grain = 0.97 + 0.03 * noise3(p * 60.0);
        float3 alb = float3(TABLE) * grain * float3(1.0, 0.99, 0.975);
        float tint = TRAIL_K * exp(-0.5 * (p.z * p.z) / (TRAIL_SIGMA * TRAIL_SIGMA));
        for (int i = 0; i < marks; i++) {
            float2 d = p.xz - M[i].xy;
            tint += MARK_K * M[i].z * exp(-0.5 * dot(d, d) / (MARK_R * MARK_R));
        }
        alb = mix(alb, TRAIL * TABLE, clamp(tint, 0.0, 0.85));
        return alb * (KEY_COL * max(KEY_DIR.y, 0.0) * sh + FILL_COL * max(FILL_DIR.y, 0.0) + irradiance(float3(0, 1, 0)) * occ * 0.55);
    }

    float3 shadeAnt(float3 p, float3 rd, int part, int light, int idx, constant Shape *S, constant float4 *A, int ants) {
        float3 n = mainNormal(p, S, A, ants);
        float3 v = -rd;
        float alpha = light == 1 ? PALE_ALPHA : 0.16;
        float3 alb = light == 1 ? PALE : DARK;
        if (part == 7) {
            float3 bump = noiseGrad(p * 200.0);
            n = normalize(n + 0.12 * (bump - n * dot(bump, n)));
            alb = EYE;
            alpha = 0.07;
        } else {
            float3 bump = noiseGrad(p * 260.0);
            n = normalize(n + 0.06 * (bump - n * dot(bump, n)));
        }
        if (part == 4 && idx >= 0) {
            // Tergite margins (step 26), in this gaster's own frame.
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
        float sh = softShadow(p + n * 0.003, KEY_DIR, S, A, ants);
        float occ = occlusion(p, n, S, A, ants, 0.6);
        return shadeCuticle(n, v, alb, sh, occ, alpha, CUTICLE_F0);
    }

    float3 toneMap(float3 x) {
        float L = dot(x, float3(0.2126, 0.7152, 0.0722));
        float Lm = L * (1.0 + L / 16.0) / (1.0 + L);
        float3 c = x * (Lm / max(L, 1e-5)) * 1.18;
        float over = max(max(c.r, max(c.g, c.b)) - 1.0, 0.0);
        c = mix(c, float3(1.0), clamp(over, 0.0, 1.0));
        c = clamp(c, 0.0, 1.0);
        return select(1.055 * pow(c, 1.0 / 2.4) - 0.055, 12.92 * c, c <= 0.0031308);
    }

    kernel void frame(device uchar4 *pixels [[buffer(0)]],
                      device float4 *auxOut [[buffer(1)]],
                      constant Params &P [[buffer(2)]],
                      constant Shape *S [[buffer(3)]],
                      constant float4 *A [[buffer(4)]],
                      constant float4 *M [[buffer(5)]],
                      uint2 gid [[thread_position_in_grid]]) {
        uint x = gid.x;
        uint y = gid.y + P.rowOffset;
        if (x >= P.width || y >= P.height) return;
        float W = float(P.width), Hh = float(P.height);
        int ants = int(P.ants), marks = int(P.marks);
        float3 sum = float3(0.0);
        uint N = P.samples;
        float4 aux = float4(0.0);
        for (uint sy = 0; sy < N; sy++) {
            for (uint sx = 0; sx < N; sx++) {
                float2 px = float2(x, y) + float2((float(sx) + 0.5) / float(N), (float(sy) + 0.5) / float(N));
                float2 s = float2(2.0 * px.x / W - 1.0, (Hh - 2.0 * px.y) / W);
                float3 ro = C_CENTRE + (C_RIGHT * s.x + C_UP * s.y) * C_HALF - C_FWD * 20.0;
                float t;
                int mat, part, light, idx;
                float3 c;
                float4 a = float4(0.0);
                if (!march(ro, C_FWD, S, A, ants, t, mat, part, light, idx)) {
                    c = environment(C_FWD);
                } else {
                    float3 p = ro + C_FWD * t;
                    a = float4(1.0, float(mat), float(part), float(idx));
                    c = mat == 1 ? shadeTable(p, S, A, ants, M, marks) : shadeAnt(p, C_FWD, part, light, idx, S, A, ants);
                }
                sum += toneMap(c);
                if (sx == N / 2 && sy == N / 2) aux = a;
            }
        }
        float3 c = sum / float(N * N);
        pixels[y * P.width + x] = uchar4(uchar3(round(clamp(c, 0.0, 1.0) * 255.0)), 255);
        auxOut[y * P.width + x] = aux;
    }

    // The distance function at arbitrary points, from the same source the
    // picture is drawn with: (distance, material, ant distance, 0).
    kernel void probe(device const float4 *points [[buffer(0)]],
                      device float4 *out [[buffer(1)]],
                      constant Params &P [[buffer(2)]],
                      constant Shape *S [[buffer(3)]],
                      constant float4 *A [[buffer(4)]],
                      uint id [[thread_position_in_grid]]) {
        float3 p = points[id].xyz;
        int m, part, lt, ix;
        float d = mainSDF(p, S, A, int(P.ants), m, part, lt, ix);
        float a = antsSDF(p, S, A, int(P.ants), 1e9, part, lt, ix);
        out[id] = float4(d, float(m), a, 0.0);
    }
    """
}

// MARK: - running it

enum RenderError: Error, CustomStringConvertible {
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
    var ants: UInt32
    var marks: UInt32
}

/// A finished frame, and what each pixel's centre sample saw:
/// (hit, material 1 table / 2 ant, part, shape index).
struct TrailImage {
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
    throw RenderError.noMetalDevice
}

func buffer<T>(_ device: MTLDevice, _ array: [T]) throws -> MTLBuffer {
    let length: Int = max(MemoryLayout<T>.stride * array.count, 16)
    let made: MTLBuffer? = array.withUnsafeBytes { raw -> MTLBuffer? in
        guard let base = raw.baseAddress else { return device.makeBuffer(length: length, options: .storageModeShared) }
        return device.makeBuffer(bytes: base, length: length, options: .storageModeShared)
    }
    guard let b = made else { throw RenderError.gpu("could not allocate a buffer") }
    return b
}

/// Compiles the kernels once and renders frames into one reused image.
final class TrailRenderer {
    let device: MTLDevice
    let image: TrailImage
    private let framePSO: MTLComputePipelineState
    private let probePSO: MTLComputePipelineState
    private let queue: MTLCommandQueue
    let mutant: Mutant
    let camera: OrthoCamera

    init(width: Int, height: Int, mutant: Mutant, on device: MTLDevice) throws {
        self.device = device
        self.mutant = mutant
        self.camera = cameraFor(mutant)
        let options = MTLCompileOptions()
        // Precise maths, as in every step.
        options.fastMathEnabled = false
        let library: MTLLibrary
        do { library = try device.makeLibrary(source: kernelSource(camera: self.camera), options: options) } catch {
            throw RenderError.kernelCompile("\(error)")
        }
        func pso(_ name: String) throws -> MTLComputePipelineState {
            guard let f = library.makeFunction(name: name) else { throw RenderError.kernelCompile("no kernel \(name)") }
            do { return try device.makeComputePipelineState(function: f) } catch { throw RenderError.kernelCompile("\(error)") }
        }
        framePSO = try pso("frame")
        probePSO = try pso("probe")
        guard let pixels = device.makeBuffer(length: width * height * 4, options: .storageModeShared),
              let aux = device.makeBuffer(length: width * height * 16, options: .storageModeShared),
              let queue = device.makeCommandQueue()
        else { throw RenderError.gpu("could not allocate buffers") }
        self.queue = queue
        image = TrailImage(width: width, height: height, pixels: pixels, aux: aux)
    }

    /// The scene at time t, as the GPU gets it.
    func frameData(time t: Float) -> FrameData {
        FrameData(ants: antsInView(time: t, range: viewRange, mutant: mutant), marks: marks(time: t, range: viewRange))
    }

    /// Render the frame at picture time t. Returns GPU seconds.
    func render(time t: Float, samples: Int) throws -> Double {
        let data: FrameData = frameData(time: t)
        let sb: MTLBuffer = try buffer(device, data.shapes)
        let ab: MTLBuffer = try buffer(device, data.ants)
        let mb: MTLBuffer = try buffer(device, data.marks)
        let band: Int = 32
        var gpu: Double = 0
        var row: Int = 0
        while row < image.height {
            let rows: Int = min(band, image.height - row)
            guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else {
                throw RenderError.gpu("could not make a command buffer")
            }
            var params = Params(width: UInt32(image.width), height: UInt32(image.height), rowOffset: UInt32(row),
                                samples: UInt32(samples), ants: UInt32(data.antCount), marks: UInt32(data.marks.count))
            enc.setComputePipelineState(framePSO)
            enc.setBuffer(image.pixels, offset: 0, index: 0)
            enc.setBuffer(image.aux, offset: 0, index: 1)
            enc.setBytes(&params, length: MemoryLayout<Params>.stride, index: 2)
            enc.setBuffer(sb, offset: 0, index: 3)
            enc.setBuffer(ab, offset: 0, index: 4)
            enc.setBuffer(mb, offset: 0, index: 5)
            let w: Int = framePSO.threadExecutionWidth
            let group = MTLSize(width: w, height: max(framePSO.maxTotalThreadsPerThreadgroup / w / 4, 1), depth: 1)
            enc.dispatchThreads(MTLSize(width: image.width, height: rows, depth: 1), threadsPerThreadgroup: group)
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            if let e = cb.error { throw RenderError.gpu(e.localizedDescription) }
            gpu += cb.gpuEndTime - cb.gpuStartTime
            row += rows
        }
        return gpu
    }

    /// The drawn distance function at arbitrary points, at time t: (distance,
    /// material, distance to the nearest ant).
    func probe(_ points: [SIMD3<Float>], time t: Float) throws -> [SIMD3<Float>] {
        let data: FrameData = FrameData(ants: antsInView(time: t, range: -40...40, mutant: mutant), marks: [])
        let pts: [SIMD4<Float>] = points.map { SIMD4<Float>($0, 0) }
        let pb: MTLBuffer = try buffer(device, pts)
        let sb: MTLBuffer = try buffer(device, data.shapes)
        let ab: MTLBuffer = try buffer(device, data.ants)
        guard let ob = device.makeBuffer(length: 16 * max(pts.count, 1), options: .storageModeShared),
              let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder()
        else { throw RenderError.gpu("could not set up the probe") }
        var params = Params(width: 0, height: 0, rowOffset: 0, samples: 0, ants: UInt32(data.antCount), marks: 0)
        enc.setComputePipelineState(probePSO)
        enc.setBuffer(pb, offset: 0, index: 0)
        enc.setBuffer(ob, offset: 0, index: 1)
        enc.setBytes(&params, length: MemoryLayout<Params>.stride, index: 2)
        enc.setBuffer(sb, offset: 0, index: 3)
        enc.setBuffer(ab, offset: 0, index: 4)
        enc.dispatchThreads(MTLSize(width: pts.count, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(probePSO.maxTotalThreadsPerThreadgroup, 256), height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        if let e = cb.error { throw RenderError.gpu(e.localizedDescription) }
        let o = ob.contents().assumingMemoryBound(to: SIMD4<Float>.self)
        return (0..<pts.count).map { SIMD3<Float>(o[$0].x, o[$0].y, o[$0].z) }
    }
}

func savePNG(_ image: TrailImage, to url: URL) throws {
    try? FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    guard let provider = CGDataProvider(dataInfo: nil, data: image.pixels.contents(), size: image.width * image.height * 4,
                                        releaseData: { _, _, _ in }),
          let space = CGColorSpace(name: CGColorSpace.sRGB),
          let cg = CGImage(width: image.width, height: image.height, bitsPerComponent: 8, bitsPerPixel: 32,
                           bytesPerRow: image.width * 4, space: space,
                           bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                           provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent),
          let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)
    else { throw RenderError.png("could not set up the encoder for \(url.path)") }
    CGImageDestinationAddImage(dest, cg, nil)
    guard CGImageDestinationFinalize(dest) else { throw RenderError.png("could not write \(url.path)") }
}
