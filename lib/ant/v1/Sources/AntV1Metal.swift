// lib/ant/v1 — what a renderer needs to draw the ant: the shapes packed for
// the GPU, and the Metal source of their distance functions, blends and
// cuticle, to paste into a scene's kernel. Nothing here needs a device; the
// scene compiles `AntV1.metalSource` together with its own kernel text.
//
// Where it comes from: step 55's renderer (step055_ant_trail_5/Sources/
// Render.swift), which is step 44's, which draws step 32's ant with step 26's
// distance functions and three blend groups. The Metal functions are step
// 55's, renamed with an `antv1_` prefix so they cannot collide with a scene's
// own; their arithmetic is unchanged (a test compares them with step 55's
// kernel at the same points). Scene-specific parts stay with the scene: the
// ground, the camera, the lights and the shading model.
//
// Buffers, as step 55's kernel took them:
//   S: AntV1Shape[]  — every ant's shapes, one ant after another
//   A: float4[]      — per ant two entries: its bounding sphere (centre, r),
//                      then (first shape, shape count, 0, 0)

import Foundation
import simd

extension AntV1 {

    /// How far a ray may trust a distance (step 26): the ellipsoid distances
    /// are approximations, and the module's tests measure how much they
    /// over-report. A marcher should step `d × stepScale`.
    static let stepScale: Float = 0.75

    // The blends (step 26's three groups, as step 55's kernel): the mesosoma's
    // parts melt into one another with k = 0.10 mm, the head's and mandibles'
    // with 0.04, everything else with 0.012; the mesosoma and head groups join
    // with 0.03, and the rest with 0.012. MODEL (step 26), chosen to read as
    // one body.
    static let blendMesosoma: Float = 0.10
    static let blendHead: Float = 0.04
    static let blendOther: Float = 0.012
    static let blendGroups: Float = 0.03
    /// The most the blends can pull a surface out beyond its shapes, mm — an
    /// ant whose bounding sphere is further than this past the best distance
    /// so far cannot be nearer (step 44).
    static let blendReach: Float = 0.03

    // Cuticle (step 32, via 44/55). Chitin, n = 1.56 (step 26): normal-
    // incidence reflectance ((n−1)/(n+1))². Colours linear sRGB: Seifert
    // (2020) — dark brown, mandibles and scape "yellowish-reddish brown".
    // The values are step 26/32's MODEL renderings of those words.
    static let cuticleIndex: Float = 1.56
    static let cuticleF0: Float = ((cuticleIndex - 1) / (cuticleIndex + 1)) * ((cuticleIndex - 1) / (cuticleIndex + 1))
    static let cuticleDark = SIMD3<Float>(0.011, 0.0082, 0.0066)
    static let cuticlePale = SIMD3<Float>(0.105, 0.048, 0.020)
    static let cuticleEye = SIMD3<Float>(0.014, 0.013, 0.013)
    /// GGX roughness: dark cuticle, pale cuticle, eye (step 32). MODEL.
    static let alphaDark: Float = 0.16
    static let alphaPale: Float = 0.34
    static let alphaEye: Float = 0.07

    // MARK: - packing for the GPU

    /// One shape: six float4s, so nothing pads differently in Swift and Metal
    /// (step 44's layout).
    struct GPUShape {
        var a: SIMD4<Float>
        var b: SIMD4<Float>
        var x: SIMD4<Float>
        var y: SIMD4<Float>
        var meta: SIMD4<Float>   // kind, part, light, blend group
        var bound: SIMD4<Float>
    }

    /// Which blend group a part melts into: 1 mesosoma, 2 head and
    /// mandibles, 0 everything else (step 26).
    static func blendGroup(_ s: Shape) -> Float {
        switch s.part {
        case .mesosoma: return 1
        case .head, .mandible: return 2
        default: return 0
        }
    }

    static func gpuShape(_ s: Shape) -> GPUShape {
        let (c, r) = s.bound
        let meta = SIMD4<Float>(Float(s.kind.rawValue), Float(s.part.rawValue), s.light ? 1 : 0, blendGroup(s))
        return GPUShape(a: SIMD4<Float>(s.a, s.ra), b: SIMD4<Float>(s.b, s.rb),
                        x: SIMD4<Float>(s.xAxis, 0), y: SIMD4<Float>(s.yAxis, 0),
                        meta: meta, bound: SIMD4<Float>(c, r))
    }

    /// Posed ants flattened for the GPU: buffers S and A above.
    struct Packed {
        var shapes: [GPUShape] = []
        var ants: [SIMD4<Float>] = []
        var count: Int = 0

        init(_ list: [Posed]) {
            for p in list {
                let (c, r) = p.bound
                ants.append(SIMD4<Float>(c, r))
                ants.append(SIMD4<Float>(Float(shapes.count), Float(p.shapes.count), 0, 0))
                shapes += p.shapes.map(AntV1.gpuShape)
            }
            count = list.count
        }
    }

    // MARK: - the Metal source

    static func metal(_ v: SIMD3<Float>) -> String { "float3(\(v.x), \(v.y), \(v.z))" }

    /// Paste before the scene's kernels (after `#include <metal_stdlib>` and
    /// `using namespace metal;`). It defines:
    ///
    ///   struct AntV1Shape
    ///   float antv1_antsSDF(float3 p, constant AntV1Shape *S, constant float4 *A, int ants,
    ///                       float cap, thread int &part, thread int &light, thread int &idx)
    ///       distance to the nearest ant, or `cap` if none is nearer; part is
    ///       an AntV1.Part raw value, light 1 for pale cuticle, idx the
    ///       nearest shape's index in S
    ///   float antv1_oneAnt(float3 p, constant AntV1Shape *S, int first, int count,
    ///                      thread int &part, thread int &light, thread int &idx)
    ///   float3 antv1_surface(float3 p, float3 n, int part, int light, int idx,
    ///                        constant AntV1Shape *S, thread float3 &albedo, thread float &alpha)
    ///       the cuticle at a hit: returns the bumped normal, sets albedo
    ///       (linear sRGB) and GGX alpha; tergite margins on the gaster
    ///   constant float ANTV1_STEP_SCALE, ANTV1_CUTICLE_F0
    ///   and helpers antv1_smin, antv1_roundCone, antv1_roundBox,
    ///   antv1_ellipsoid, antv1_hash3, antv1_noise3, antv1_noiseGrad.
    static var metalSource: String {
        return """
        // ---- lib/ant/v1: the shared black garden ant (Lasius niger worker) ----
        struct AntV1Shape { float4 a; float4 b; float4 x; float4 y; float4 meta; float4 bound; };

        constant float ANTV1_STEP_SCALE = \(stepScale);
        constant float ANTV1_CUTICLE_F0 = \(cuticleF0);
        constant float ANTV1_K_MESO = \(blendMesosoma);
        constant float ANTV1_K_HEAD = \(blendHead);
        constant float ANTV1_K_OTHER = \(blendOther);
        constant float ANTV1_K_GROUPS = \(blendGroups);
        constant float ANTV1_REACH = \(blendReach);
        constant float3 ANTV1_DARK = \(metal(cuticleDark));
        constant float3 ANTV1_PALE = \(metal(cuticlePale));
        constant float3 ANTV1_EYE = \(metal(cuticleEye));
        constant float ANTV1_ALPHA_DARK = \(alphaDark);
        constant float ANTV1_ALPHA_PALE = \(alphaPale);
        constant float ANTV1_ALPHA_EYE = \(alphaEye);

        float antv1_smin(float a, float b, float k) {
            float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
            return mix(b, a, h) - k * h * (1.0 - h);
        }

        float antv1_hash3(float3 p) {
            p = fract(p * 0.3183099 + 0.1);
            p *= 17.0;
            return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
        }
        float antv1_noise3(float3 x) {
            float3 i = floor(x);
            float3 f = fract(x);
            f = f * f * (3.0 - 2.0 * f);
            float a = mix(mix(antv1_hash3(i + float3(0,0,0)), antv1_hash3(i + float3(1,0,0)), f.x),
                          mix(antv1_hash3(i + float3(0,1,0)), antv1_hash3(i + float3(1,1,0)), f.x), f.y);
            float b = mix(mix(antv1_hash3(i + float3(0,0,1)), antv1_hash3(i + float3(1,0,1)), f.x),
                          mix(antv1_hash3(i + float3(0,1,1)), antv1_hash3(i + float3(1,1,1)), f.x), f.y);
            return mix(a, b, f.z);
        }
        float3 antv1_noiseGrad(float3 x) {
            float e = 0.05;
            float c = antv1_noise3(x);
            return float3(antv1_noise3(x + float3(e,0,0)) - c, antv1_noise3(x + float3(0,e,0)) - c, antv1_noise3(x + float3(0,0,e)) - c) / e;
        }

        float antv1_roundCone(float3 p, float3 a, float3 b, float r1, float r2) {
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

        float antv1_roundBox(float3 p, float3 c, float3 b, float r, float3 xa, float3 ya) {
            float3 q0 = p - c;
            float3 za = cross(xa, ya);
            float3 q = abs(float3(dot(q0, xa), dot(q0, ya), dot(q0, za))) - b;
            return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0) - r;
        }

        float antv1_ellipsoid(float3 p, float3 c, float3 r, float3 xa, float3 ya) {
            float3 q0 = p - c;
            float3 za = cross(xa, ya);
            float3 q = float3(dot(q0, xa), dot(q0, ya), dot(q0, za));
            float k0 = length(q / r);
            float k1 = length(q / (r * r));
            return k0 * (k0 - 1.0) / max(k1, 1e-6);
        }

        // One ant: step 26's three blend groups. Round cones are exact, so one
        // far from the current best is skipped; ellipsoids never are.
        float antv1_oneAnt(float3 p, constant AntV1Shape *S, int first, int count, thread int &part, thread int &light, thread int &idx) {
            float g0 = 1e9, g1 = 1e9, g2 = 1e9;
            float best = 1e9;
            for (int i = first; i < first + count; i++) {
                AntV1Shape s = S[i];
                int group = int(s.meta.w + 0.5);
                float k = group == 1 ? ANTV1_K_MESO : (group == 2 ? ANTV1_K_HEAD : ANTV1_K_OTHER);
                float cur = group == 1 ? g1 : (group == 2 ? g2 : g0);
                float d;
                if (s.meta.x < 0.5) {
                    float bd = length(p - s.bound.xyz) - s.bound.w;
                    if (bd > cur + k && bd > best) continue;
                    d = antv1_roundCone(p, s.a.xyz, s.b.xyz, s.a.w, s.b.w);
                } else if (s.meta.x < 1.5) {
                    d = antv1_ellipsoid(p, s.a.xyz, s.b.xyz, s.x.xyz, s.y.xyz);
                } else {
                    d = antv1_roundBox(p, s.a.xyz, s.b.xyz, s.a.w, s.x.xyz, s.y.xyz);
                }
                if (d < best) { best = d; part = int(s.meta.y + 0.5); light = int(s.meta.z + 0.5); idx = i; }
                if (group == 1) g1 = antv1_smin(g1, d, k); else if (group == 2) g2 = antv1_smin(g2, d, k); else g0 = antv1_smin(g0, d, k);
            }
            return antv1_smin(antv1_smin(g1, g2, ANTV1_K_GROUPS), g0, ANTV1_K_OTHER);
        }

        // All the ants: the nearest one wins. An ant whose bounding sphere is
        // further than the best so far (less the most its blends can pull a
        // surface out) cannot be nearer, and is skipped. `cap`: the caller
        // does not care about distances beyond it, so an ant further than
        // that is never opened; if none is nearer, cap comes back.
        float antv1_antsSDF(float3 p, constant AntV1Shape *S, constant float4 *A, int ants, float cap, thread int &part, thread int &light, thread int &idx) {
            float best = cap;
            part = 0; light = 0; idx = -1;
            for (int k = 0; k < ants; k++) {
                float4 bs = A[2 * k];
                float bd = length(p - bs.xyz) - bs.w;
                if (bd - ANTV1_REACH > best) continue;
                int pt, lt, ix;
                float d = antv1_oneAnt(p, S, int(A[2 * k + 1].x + 0.5), int(A[2 * k + 1].y + 0.5), pt, lt, ix);
                if (d < best) { best = d; part = pt; light = lt; idx = ix; }
            }
            return best;
        }

        // The cuticle at a hit (step 32's shadeAnt, before its lighting): a
        // fine bump on every part, a coarser one and a glossier coat on the
        // eye, and on the gaster three dark tergite margins in its own frame.
        float3 antv1_surface(float3 p, float3 n, int part, int light, int idx, constant AntV1Shape *S,
                             thread float3 &albedo, thread float &alpha) {
            alpha = light == 1 ? ANTV1_ALPHA_PALE : ANTV1_ALPHA_DARK;
            albedo = light == 1 ? ANTV1_PALE : ANTV1_DARK;
            if (part == 7) {
                float3 bump = antv1_noiseGrad(p * 200.0);
                n = normalize(n + 0.12 * (bump - n * dot(bump, n)));
                albedo = ANTV1_EYE;
                alpha = ANTV1_ALPHA_EYE;
            } else {
                float3 bump = antv1_noiseGrad(p * 260.0);
                n = normalize(n + 0.06 * (bump - n * dot(bump, n)));
            }
            if (part == 4 && idx >= 0) {
                AntV1Shape g = S[idx];
                float x = dot(p - g.a.xyz, g.x.xyz) / g.b.x;
                float m = 0.0;
                for (int i = 0; i < 3; i++) {
                    float at = 0.42 - 0.40 * float(i);
                    m = max(m, exp(-pow((x - at) / 0.022, 2.0)));
                }
                albedo *= 1.0 - 0.45 * m;
                n = normalize(n + g.x.xyz * 0.35 * m);
            }
            return n;
        }
        // ---- end lib/ant/v1 ----

        """
    }
}
