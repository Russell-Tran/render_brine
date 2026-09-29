// Step 79: one frame of the ant world, marched per pixel on the GPU.
//
// Every surface is a distance function, as in steps 26 and 44–57: the card
// with the nest entrance cut in it, step 26's sucrose grains (clear: rays
// refract in and bounce out, which is their sparkle), and the ants — drawn
// by the SHARED ANT, lib/ant/v1, whose Metal source (AntV1.metalSource) is
// pasted in front of this kernel: its shapes, blends and cuticle. The light
// and shading are step 55's; the pheromone comes from the simulation's grid,
// sampled under each ground point and drawn as step 44's blue.
//
// Two ways to find the nearest ant, chosen per frame (`WorldRenderer.bounds`):
//   * BOUNDS ON — lib/ant/v1's antv1_antsSDF: each ant's bounding sphere is
//     tested first, and an ant whose sphere is further than the best
//     distance so far is never opened (steps 44/55's speed-up);
//   * BOUNDS OFF — every ant's shapes at every step (antv1_oneAnt on each):
//     no acceleration at all.
// Both give the same picture bit for bit (a test checks it); only the time
// differs. That is prediction P3's measurement.
//
// Scene-wide, for both: nothing stands higher than `top` (the tallest ant
// or grain), so a ray starts marching where it comes down through that
// height, and a shadow ray stops once it rises above it.

import CoreGraphics
import Foundation
import ImageIO
import Metal
import UniformTypeIdentifiers
import simd

// MARK: - light (step 55's, which is step 32's)

let worldKeyDirection: SIMD3<Float> = simd_normalize(SIMD3<Float>(-0.35, 1.0, 0.50))
let worldKeyColour: SIMD3<Float> = SIMD3<Float>(1.0, 0.985, 0.96) * 1.6
let worldFillColour: SIMD3<Float> = SIMD3<Float>(0.96, 0.97, 1.0) * 0.55
let worldKeyDisc: Float = 0.975
let worldFillDisc: Float = 0.985

func worldDiscRadiance(_ irradiance: SIMD3<Float>, cosine: Float) -> SIMD3<Float> {
    let solid: Float = 2 * Float.pi * (1 - cosine)
    return irradiance / solid
}

// MARK: - what to break, for the mutation check

/// Each must make the suite fail (SIM_MUTANT, shared with the simulation's
/// mutants; each enum ignores the other's names).
enum WorldMutant: String {
    case none
    /// Every ant's bounding sphere drawn at half its radius: the bounds
    /// then skip ants that are nearer than they seem.
    case shrunkBounds

    static func fromEnvironment() -> WorldMutant {
        guard let s = ProcessInfo.processInfo.environment["SIM_MUTANT"] else { return .none }
        return WorldMutant(rawValue: s) ?? .none
    }
}

/// Posed ants packed for the GPU (lib/ant/v1), with the mutant applied.
func worldPack(_ ants: [AntV1.Posed], mutant: WorldMutant) -> AntV1.Packed {
    var packed = AntV1.Packed(ants)
    if mutant == .shrunkBounds {
        for k in stride(from: 0, to: packed.ants.count, by: 2) { packed.ants[k].w *= 0.5 }
    }
    return packed
}

// MARK: - what the GPU gets

/// Per frame. The layout matches the kernel's `WorldParams` (eight uints,
/// four floats, then float4s, so nothing pads differently).
struct WorldParams {
    var width: UInt32
    var height: UInt32
    var rowOffset: UInt32
    var samples: UInt32
    var ants: UInt32
    var grains: UInt32
    var fieldWidth: UInt32
    var fieldHeight: UInt32
    var fieldCell: Float
    var bounds: Float          // 1 = per-ant bounding spheres, 0 = none
    var top: Float             // nothing is higher than this, mm
    var pad: Float
    var camCentre: SIMD4<Float>   // xyz, halfWidth
    var camForward: SIMD4<Float>
    var camRight: SIMD4<Float>
    var camUp: SIMD4<Float>
    var pile: SIMD4<Float>        // bounding sphere of all grains: centre, radius
    var nest: SIMD4<Float>        // x, z, radius, depth
}

func worldMetal(_ v: SIMD3<Float>) -> String { "float3(\(v.x), \(v.y), \(v.z))" }

/// The scene's kernel text (after AntV1.metalSource).
func worldKernelSource() -> String {
    let keyRad: SIMD3<Float> = worldDiscRadiance(worldKeyColour, cosine: worldKeyDisc)
    let fillDir: SIMD3<Float> = simd_normalize(SIMD3<Float>(0.3, 1.0, -0.6))
    let fillRad: SIMD3<Float> = worldDiscRadiance(worldFillColour, cosine: worldFillDisc)
    return """
    struct WorldParams {
        uint width; uint height; uint rowOffset; uint samples;
        uint ants; uint grains; uint fieldWidth; uint fieldHeight;
        float fieldCell; float bounds; float top; float pad;
        float4 camCentre; float4 camForward; float4 camRight; float4 camUp;
        float4 pile; float4 nest;
    };

    constant float3 KEY_DIR = \(worldMetal(worldKeyDirection));
    constant float3 FILL_DIR = \(worldMetal(fillDir));
    constant float3 KEY_COL = \(worldMetal(worldKeyColour));
    constant float3 FILL_COL = \(worldMetal(worldFillColour));
    constant float3 KEY_RAD = \(worldMetal(keyRad));
    constant float3 FILL_RAD = \(worldMetal(fillRad));
    constant float KEY_DISC = \(worldKeyDisc);
    constant float FILL_DISC = \(worldFillDisc);
    constant float CARD = \(worldCardAlbedo);
    constant float3 SOIL = \(worldMetal(worldSoilAlbedo));
    constant float3 TRAIL = \(worldMetal(worldTrailColour));
    constant float TRAIL_MAX = \(worldTrailMax);
    constant float TRAIL_SCALE = \(worldTrailScale);
    constant float SUCROSE_N = \(sucroseIndex);
    constant float SUCROSE_F0 = \(sucroseF0);
    constant float GRAIN_ROUND = \(sugarRounding);
    constant int FACES = \(sugarFaces);

    struct Scene {
        constant WorldParams *P;
        constant AntV1Shape *S;
        constant float4 *A;
        constant float4 *G;
        constant float *F;
    };

    // ------------------------------------------------------------ the card and the nest

    // The card is the half-space y <= 0 with a round hole at the nest; the
    // hole has a floor `depth` below. max/min of exact distances: never more
    // than the true distance.
    float groundSDF(float3 p, float4 nest) {
        float rho = length(p.xz - nest.xy);
        return min(max(p.y, nest.z - rho), p.y + nest.w);
    }

    // ------------------------------------------------------------ the sugar (step 26's)

    float smax(float a, float b, float k) {
        float h = clamp(0.5 + 0.5 * (a - b) / k, 0.0, 1.0);
        return mix(b, a, h) + k * h * (1.0 - h);
    }

    float grainSDF(float3 p, constant float4 *G, int g) {
        float3 c = G[g * (FACES + 1)].xyz;
        float3 q = p - c;
        float4 f0 = G[g * (FACES + 1) + 1];
        float d = dot(q, f0.xyz) - f0.w;
        for (int i = 1; i < FACES; i++) {
            float4 f = G[g * (FACES + 1) + 1 + i];
            d = smax(d, dot(q, f.xyz) - f.w, GRAIN_ROUND);
        }
        return d;
    }

    // The nearest grain, or `cap` if none is nearer. The pile's sphere, then
    // each grain's, are tested first: a grain further than the best so far
    // is never opened. `skip`: a grain a ray is leaving.
    float grainsSDF(float3 p, Scene s, float cap, int skip, thread int &which) {
        float best = cap;
        which = -1;
        if (length(p - s.P->pile.xyz) - s.P->pile.w > best) return best;
        int n = int(s.P->grains);
        for (int g = 0; g < n; g++) {
            if (g == skip) continue;
            float4 c = s.G[g * (FACES + 1)];
            if (length(p - c.xyz) - c.w > best) continue;
            float e = grainSDF(p, s.G, g);
            if (e < best) { best = e; which = g; }
        }
        return best;
    }

    // ------------------------------------------------------------ the ants (lib/ant/v1)

    // The nearest ant, or `cap`: with the lib's bounding spheres, or
    // (bounds off) every ant opened at every call.
    float antsSDF(float3 p, Scene s, float cap, thread int &part, thread int &light, thread int &idx) {
        int n = int(s.P->ants);
        if (s.P->bounds > 0.5) return antv1_antsSDF(p, s.S, s.A, n, cap, part, light, idx);
        float best = cap;
        part = 0; light = 0; idx = -1;
        for (int k = 0; k < n; k++) {
            int pt, lt, ix;
            float d = antv1_oneAnt(p, s.S, int(s.A[2 * k + 1].x + 0.5), int(s.A[2 * k + 1].y + 0.5), pt, lt, ix);
            if (d < best) { best = d; part = pt; light = lt; idx = ix; }
        }
        return best;
    }

    // ------------------------------------------------------------ everything

    // Materials: 1 card (and the nest's hole), 2 ant, 3 sugar.
    float mainSDF(float3 p, Scene s, int skipGrain, thread int &mat, thread int &sub, thread int &light) {
        float d = groundSDF(p, s.P->nest);
        mat = 1; sub = 0; light = 0;
        int w;
        float g = grainsSDF(p, s, max(d, 0.0) + 0.001, skipGrain, w);
        if (g < d) { d = g; mat = 3; sub = w; }
        int pt, lt, ix;
        float a = antsSDF(p, s, max(d, 0.0) + 0.001, pt, lt, ix);
        if (a < d) { d = a; mat = 2; sub = ix; light = lt; }
        return d;
    }

    float mainDist(float3 p, Scene s) {
        int m, u, l;
        return mainSDF(p, s, -1, m, u, l);
    }

    float3 mainNormal(float3 p, Scene s) {
        const float e = 0.0015;
        float3 k1 = float3(1, -1, -1), k2 = float3(-1, -1, 1), k3 = float3(-1, 1, -1), k4 = float3(1, 1, 1);
        return normalize(k1 * mainDist(p + k1 * e, s) + k2 * mainDist(p + k2 * e, s)
                       + k3 * mainDist(p + k3 * e, s) + k4 * mainDist(p + k4 * e, s));
    }

    float3 grainNormal(float3 p, constant float4 *G, int g) {
        const float e = 0.0008;
        float3 k1 = float3(1, -1, -1), k2 = float3(-1, -1, 1), k3 = float3(-1, 1, -1), k4 = float3(1, 1, 1);
        return normalize(k1 * grainSDF(p + k1 * e, G, g) + k2 * grainSDF(p + k2 * e, G, g)
                       + k3 * grainSDF(p + k3 * e, G, g) + k4 * grainSDF(p + k4 * e, G, g));
    }

    // Where a ray first comes down to height `top`: nothing is above it.
    float entry(float3 ro, float3 rd, float top) {
        if (ro.y <= top || rd.y >= 0.0) return 0.0;
        return (ro.y - top) / (-rd.y);
    }

    bool march(float3 ro, float3 rd, Scene s, int skipGrain, thread float &t, thread int &mat, thread int &sub, thread int &light) {
        t = entry(ro, rd, s.P->top);
        float far = t + 400.0;
        for (int i = 0; i < 400; i++) {
            float3 p = ro + rd * t;
            float d = mainSDF(p, s, skipGrain, mat, sub, light);
            if (d < 0.0006) return true;
            t += max(d * ANTV1_STEP_SCALE, 0.0003);
            if (t > far) break;
        }
        return false;
    }

    // Shadows (step 26/55): an ant blocks the key light; a clear grain dims
    // it (MODEL 45%, step 26). The ray stops once it is above everything.
    float softShadow(float3 ro, float3 rd, Scene s) {
        float resA = 1.0, resG = 1.0;
        float t = 0.004;
        float top = s.P->top;
        for (int i = 0; i < 90; i++) {
            float3 p = ro + rd * t;
            if (p.y > top) break;
            int pt, lt, ix, w;
            float a = antsSDF(p, s, 1.0, pt, lt, ix);
            float g = grainsSDF(p, s, 1.0, -1, w);
            resA = min(resA, 8.0 * a / t);
            resG = min(resG, 8.0 * g / t);
            t += clamp(min(a, g) * ANTV1_STEP_SCALE, 0.002, 0.2);
            if (resA < 0.002 || t > 6.0) break;
        }
        return clamp(resA, 0.0, 1.0) * mix(0.55, 1.0, clamp(resG, 0.0, 1.0));
    }

    float occlusion(float3 p, float3 n, Scene s, float scale) {
        float occ = 0.0, w = 1.0;
        for (int i = 1; i <= 5; i++) {
            float h = scale * (0.02 + 0.06 * float(i * i));
            occ += w * clamp((h - mainDist(p + n * h, s)) / h, 0.0, 1.0);
            w *= 0.7;
        }
        return clamp(1.0 - 0.5 * occ, 0.0, 1.0);
    }

    // ------------------------------------------------------------ light (step 55's)

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

    // ------------------------------------------------------------ shading

    // The pheromone under a ground point, marks/mm²: bilinear between the
    // simulation's cell centres (cell i spans [i·h, (i+1)·h)), 0 off the grid.
    float fieldAt(float2 xz, Scene s) {
        float h = s.P->fieldCell;
        int W = int(s.P->fieldWidth), H = int(s.P->fieldHeight);
        if (W == 0) return 0.0;
        float fx = xz.x / h - 0.5, fy = xz.y / h - 0.5;
        int i0 = int(floor(fx)), j0 = int(floor(fy));
        float u = fx - float(i0), v = fy - float(j0);
        float c = 0.0;
        for (int dj = 0; dj < 2; dj++) {
            for (int di = 0; di < 2; di++) {
                int i = i0 + di, j = j0 + dj;
                if (i < 0 || j < 0 || i >= W || j >= H) continue;
                float wgt = (di == 1 ? u : 1.0 - u) * (dj == 1 ? v : 1.0 - v);
                c += wgt * s.F[j * W + i];
            }
        }
        return c;
    }

    float3 shadeGround(float3 p, Scene s) {
        float3 n = float3(0, 1, 0);
        float4 nest = s.P->nest;
        float rho = length(p.xz - nest.xy);
        bool inHole = rho < nest.z + 0.002 && p.y < -0.001;
        if (inHole) n = p.y < -nest.w + 0.002 ? float3(0, 1, 0) : normalize(float3(nest.x - p.x, 0.0, nest.y - p.z));
        float sh = softShadow(p + n * 0.002, KEY_DIR, s);
        // Occlusion reaches 1.6 mm; further from everything it is exactly 1.
        int pt, lt, ix, w;
        float nearA = antsSDF(p, s, 2.0, pt, lt, ix);
        float nearG = grainsSDF(p, s, 2.0, -1, w);
        bool nearNest = rho < nest.z + 2.0;
        float occ = (nearA >= 2.0 && nearG >= 2.0 && !nearNest) ? 1.0 : occlusion(p, n, s, 1.0);
        float3 alb;
        if (inHole) {
            // Below the card: soil, darkening with depth as less light gets in.
            alb = SOIL * exp(p.y / 1.5);
        } else {
            alb = float3(CARD) * float3(1.0, 0.99, 0.975);
            float c = fieldAt(p.xz, s);
            float tint = TRAIL_MAX * (1.0 - exp(-c / TRAIL_SCALE));
            alb = mix(alb, TRAIL * CARD, tint);
        }
        float key = max(dot(n, KEY_DIR), 0.0) * sh;
        float fill = max(dot(n, FILL_DIR), 0.0);
        float3 lit = KEY_COL * key + FILL_COL * fill + irradiance(n) * occ * 0.55;
        return alb * lit;
    }

    float3 shadeAnt(float3 p, float3 rd, int idx, int light, Scene s) {
        float3 n = mainNormal(p, s);
        int part = idx >= 0 ? int(s.S[idx].meta.y + 0.5) : 0;
        float3 alb;
        float alpha;
        n = antv1_surface(p, n, part, light, idx, s.S, alb, alpha);
        float sh = softShadow(p + n * 0.003, KEY_DIR, s);
        float occ = occlusion(p, n, s, 0.6);
        return shadeCuticle(n, -rd, alb, sh, occ, alpha, ANTV1_CUTICLE_F0);
    }

    // What a ray sees after it leaves a grain: shaded simply (step 26).
    float3 shadeBehind(float3 ro, float3 rd, int fromGrain, Scene s) {
        float t;
        int mat, sub, light;
        if (!march(ro, rd, s, fromGrain, t, mat, sub, light)) return environment(rd);
        float3 p = ro + rd * t;
        if (mat == 1) return shadeGround(p, s);
        if (mat == 2) return shadeAnt(p, rd, sub, light, s);
        float3 n = grainNormal(p, s.G, sub);
        float F = schlick(SUCROSE_F0, max(dot(n, -rd), 0.0));
        return mix(shadeGround(float3(p.x, 0.0, p.z), s) * 0.95, environment(reflect(rd, n)), F);
    }

    // A sugar grain (step 26): reflect; refract in; bounce inside by total
    // internal reflection until a face lets the ray out; see what is behind.
    float3 shadeGrain(float3 p, float3 rd, int g, Scene s) {
        float3 n = grainNormal(p, s.G, g);
        float c = max(dot(n, -rd), 0.0);
        float F = schlick(SUCROSE_F0, c);
        float3 refl = environment(reflect(rd, n));
        float sh = softShadow(p + n * 0.002, KEY_DIR, s);
        if (dot(reflect(rd, n), KEY_DIR) > KEY_DISC - 0.02) refl = mix(irradiance(reflect(rd, n)), refl, sh);
        float3 dir = refract(rd, n, 1.0 / SUCROSE_N);
        float3 q = p - n * 0.001;
        float3 through = float3(0.0);
        float carry = 1.0 - F;
        for (int bounce = 0; bounce < 5; bounce++) {
            float t = 0.0;
            for (int i = 0; i < 64; i++) {
                float d = -grainSDF(q + dir * t, s.G, g);
                if (d < 0.0005) break;
                t += max(d, 0.0005);
            }
            q += dir * t;
            float3 m = grainNormal(q, s.G, g);
            float3 out = refract(dir, -m, SUCROSE_N);
            float cosIn = max(dot(dir, m), 0.0);
            if (dot(out, out) > 0.0) {
                float Fe = schlick(SUCROSE_F0, cosIn);
                float3 behind = shadeBehind(q + m * 0.002, normalize(out), g, s);
                through += carry * (1.0 - Fe) * behind;
                carry *= Fe;
            }
            dir = reflect(dir, -m);
            q -= m * 0.001;
            if (carry < 0.02) break;
        }
        through += carry * irradiance(dir);
        return refl * F + through * float3(0.985, 0.99, 1.0);
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

    kernel void worldFrame(device uchar4 *pixels [[buffer(0)]],
                           device float4 *auxOut [[buffer(1)]],
                           constant WorldParams &P [[buffer(2)]],
                           constant AntV1Shape *S [[buffer(3)]],
                           constant float4 *A [[buffer(4)]],
                           constant float4 *G [[buffer(5)]],
                           constant float *F [[buffer(6)]],
                           uint2 gid [[thread_position_in_grid]]) {
        uint x = gid.x;
        uint y = gid.y + P.rowOffset;
        if (x >= P.width || y >= P.height) return;
        Scene s = { &P, S, A, G, F };
        float W = float(P.width), Hh = float(P.height);
        float3 C = P.camCentre.xyz, FWD = P.camForward.xyz, RIGHT = P.camRight.xyz, UP = P.camUp.xyz;
        float halfW = P.camCentre.w;
        float3 sum = float3(0.0);
        uint N = P.samples;
        float4 aux = float4(0.0);
        for (uint sy = 0; sy < N; sy++) {
            for (uint sx = 0; sx < N; sx++) {
                float2 px = float2(x, y) + float2((float(sx) + 0.5) / float(N), (float(sy) + 0.5) / float(N));
                float2 q = float2(2.0 * px.x / W - 1.0, (Hh - 2.0 * px.y) / W);
                float3 ro = C + (RIGHT * q.x + UP * q.y) * halfW - FWD * 200.0;
                float t;
                int mat, sub, light;
                float3 c;
                float4 a = float4(0.0);
                if (!march(ro, FWD, s, -1, t, mat, sub, light)) {
                    c = environment(FWD);
                } else {
                    float3 p = ro + FWD * t;
                    a = float4(1.0, float(mat), float(sub), 0.0);
                    if (mat == 1) c = shadeGround(p, s);
                    else if (mat == 2) c = shadeAnt(p, FWD, sub, light, s);
                    else c = shadeGrain(p, FWD, sub, s);
                }
                sum += toneMap(c);
                if (sx == N / 2 && sy == N / 2) aux = a;
            }
        }
        float3 c = sum / float(N * N);
        pixels[y * P.width + x] = uchar4(uchar3(round(clamp(c, 0.0, 1.0) * 255.0)), 255);
        auxOut[y * P.width + x] = aux;
    }

    // The drawn distance functions at arbitrary points, from the same source
    // the picture is drawn with: (scene distance, material, nearest ant
    // uncapped, nearest grain uncapped).
    kernel void worldProbe(device const float4 *points [[buffer(0)]],
                           device float4 *out [[buffer(1)]],
                           constant WorldParams &P [[buffer(2)]],
                           constant AntV1Shape *S [[buffer(3)]],
                           constant float4 *A [[buffer(4)]],
                           constant float4 *G [[buffer(5)]],
                           constant float *F [[buffer(6)]],
                           uint id [[thread_position_in_grid]]) {
        Scene s = { &P, S, A, G, F };
        float3 p = points[id].xyz;
        int m, u, l, pt, lt, ix, w;
        float d = mainSDF(p, s, -1, m, u, l);
        float a = antsSDF(p, s, 1e9, pt, lt, ix);
        float g = grainsSDF(p, s, 1e9, -1, w);
        out[id] = float4(d, float(m), a, g);
    }
    """
}

// MARK: - running it

enum WorldRenderError: Error, CustomStringConvertible {
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

func worldBuffer<T>(_ device: MTLDevice, _ array: [T]) throws -> MTLBuffer {
    let length: Int = max(MemoryLayout<T>.stride * array.count, 16)
    let made: MTLBuffer? = array.withUnsafeBytes { raw -> MTLBuffer? in
        guard let base = raw.baseAddress else { return device.makeBuffer(length: length, options: .storageModeShared) }
        return device.makeBuffer(bytes: base, length: length, options: .storageModeShared)
    }
    guard let b = made else { throw WorldRenderError.gpu("could not allocate a buffer") }
    return b
}

/// Everything one frame draws.
struct WorldFrameScene {
    var camera: WorldCamera
    var ants: [AntV1.Posed]
    /// The pheromone grid (marks/mm², row-major), or empty for none.
    var field: [Float]
}

/// The static world: the grains, the nest, the grid size.
struct WorldSet {
    let config: SimWorldConfig
    let grains: [SugarGrain]
    let grainData: [SIMD4<Float>]
    let pile: SIMD4<Float>
    let grainTop: Float

    init(config: SimWorldConfig, grains simGrains: [SimGrain]) {
        self.config = config
        grains = simGrains.map(sugarGrain)
        grainData = sugarGPU(grains)
        let c = SIMD3<Float>(config.sugar.x, 0, config.sugar.y)
        var r: Float = 0
        var top: Float = 0
        for g in grains {
            let reach: Float = simd_distance(g.centre, c) + g.boundRadius
            r = max(r, reach)
            top = max(top, g.centre.y + g.boundRadius)
        }
        pile = SIMD4<Float>(c, r)
        grainTop = top
    }

    var nest: SIMD4<Float> { SIMD4<Float>(config.nest.x, config.nest.y, config.nestRadius, worldNestDepth) }
}

/// A finished frame, and what each pixel's centre sample saw:
/// (hit, material 1 ground / 2 ant / 3 sugar, shape or grain index, 0).
struct WorldImage {
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

    func bytes() -> [UInt8] {
        let n: Int = width * height * 4
        let p = pixels.contents().assumingMemoryBound(to: UInt8.self)
        return Array(UnsafeBufferPointer(start: p, count: n))
    }
}

/// Compiles the shared ant's Metal and the scene's kernel once; renders
/// frames into one reused image.
final class WorldRenderer {
    let device: MTLDevice
    let image: WorldImage
    let set: WorldSet
    /// Per-ant bounding spheres (lib/ant/v1's antv1_antsSDF) or none.
    var bounds: Bool = true
    /// Rows per GPU command (a watchdog-friendly slice of the frame).
    var band: Int = 32
    var mutant: WorldMutant = WorldMutant.fromEnvironment()
    private let framePSO: MTLComputePipelineState
    private let probePSO: MTLComputePipelineState
    private let queue: MTLCommandQueue
    private let grainBuffer: MTLBuffer

    init(width: Int, height: Int, set: WorldSet) throws {
        guard let d = MTLCreateSystemDefaultDevice() ?? MTLCopyAllDevices().first else { throw WorldRenderError.noMetalDevice }
        device = d
        self.set = set
        let options = MTLCompileOptions()
        // Precise maths, as in every step.
        options.fastMathEnabled = false
        let source: String = "#include <metal_stdlib>\nusing namespace metal;\n" + AntV1.metalSource + worldKernelSource()
        let library: MTLLibrary
        do { library = try d.makeLibrary(source: source, options: options) } catch {
            throw WorldRenderError.kernelCompile("\(error)")
        }
        func pso(_ name: String) throws -> MTLComputePipelineState {
            guard let f = library.makeFunction(name: name) else { throw WorldRenderError.kernelCompile("no kernel \(name)") }
            do { return try d.makeComputePipelineState(function: f) } catch { throw WorldRenderError.kernelCompile("\(error)") }
        }
        framePSO = try pso("worldFrame")
        probePSO = try pso("worldProbe")
        guard let pixels = d.makeBuffer(length: width * height * 4, options: .storageModeShared),
              let aux = d.makeBuffer(length: width * height * 16, options: .storageModeShared),
              let q = d.makeCommandQueue()
        else { throw WorldRenderError.gpu("could not allocate buffers") }
        queue = q
        image = WorldImage(width: width, height: height, pixels: pixels, aux: aux)
        grainBuffer = try worldBuffer(d, set.grainData)
    }

    /// The tallest thing in the frame, mm, with a margin for the blends.
    func sceneTop(_ ants: [AntV1.Posed]) -> Float {
        var top: Float = set.grainTop
        for a in ants {
            for sh in a.shapes { top = max(top, sh.extent(along: SIMD3<Float>(0, 1, 0)).hi) }
        }
        return top + AntV1.blendReach + 0.01
    }

    func params(_ scene: WorldFrameScene, packed: AntV1.Packed, samples: Int, top: Float) -> WorldParams {
        let cam: WorldCamera = scene.camera
        let hasField: Bool = !scene.field.isEmpty
        let fw: Int = hasField ? set.config.gridWidth : 0
        let fh: Int = hasField ? set.config.gridHeight : 0
        return WorldParams(width: UInt32(image.width), height: UInt32(image.height), rowOffset: 0,
                           samples: UInt32(samples), ants: UInt32(packed.count), grains: UInt32(set.grains.count),
                           fieldWidth: UInt32(fw), fieldHeight: UInt32(fh), fieldCell: SimConst.cell,
                           bounds: bounds ? 1 : 0, top: top, pad: 0,
                           camCentre: SIMD4<Float>(cam.target, cam.halfWidth), camForward: SIMD4<Float>(cam.forward, 0),
                           camRight: SIMD4<Float>(cam.right, 0), camUp: SIMD4<Float>(cam.up, 0),
                           pile: set.pile, nest: set.nest)
    }

    /// Render one frame. Returns GPU seconds.
    @discardableResult
    func render(_ scene: WorldFrameScene, samples: Int) throws -> Double {
        if !scene.field.isEmpty && scene.field.count != set.config.gridWidth * set.config.gridHeight {
            throw WorldRenderError.gpu("field is \(scene.field.count) cells, grid is \(set.config.gridWidth)×\(set.config.gridHeight)")
        }
        let packed: AntV1.Packed = worldPack(scene.ants, mutant: mutant)
        let sb: MTLBuffer = try worldBuffer(device, packed.shapes)
        let ab: MTLBuffer = try worldBuffer(device, packed.ants)
        let fb: MTLBuffer = try worldBuffer(device, scene.field)
        var params: WorldParams = self.params(scene, packed: packed, samples: samples, top: sceneTop(scene.ants))
        var gpu: Double = 0
        var row: Int = 0
        while row < image.height {
            let rows: Int = min(band, image.height - row)
            guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else {
                throw WorldRenderError.gpu("could not make a command buffer")
            }
            params.rowOffset = UInt32(row)
            enc.setComputePipelineState(framePSO)
            enc.setBuffer(image.pixels, offset: 0, index: 0)
            enc.setBuffer(image.aux, offset: 0, index: 1)
            enc.setBytes(&params, length: MemoryLayout<WorldParams>.stride, index: 2)
            enc.setBuffer(sb, offset: 0, index: 3)
            enc.setBuffer(ab, offset: 0, index: 4)
            enc.setBuffer(grainBuffer, offset: 0, index: 5)
            enc.setBuffer(fb, offset: 0, index: 6)
            let w: Int = framePSO.threadExecutionWidth
            let tall: Int = max(framePSO.maxTotalThreadsPerThreadgroup / w / 4, 1)
            let group = MTLSize(width: w, height: tall, depth: 1)
            enc.dispatchThreads(MTLSize(width: image.width, height: rows, depth: 1), threadsPerThreadgroup: group)
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            if let e = cb.error { throw WorldRenderError.gpu(e.localizedDescription) }
            gpu += cb.gpuEndTime - cb.gpuStartTime
            row += rows
        }
        return gpu
    }

    /// The drawn distance functions at arbitrary points: (scene distance,
    /// material, nearest ant, nearest grain).
    func probe(_ points: [SIMD3<Float>], ants: [AntV1.Posed]) throws -> [SIMD4<Float>] {
        let packed: AntV1.Packed = worldPack(ants, mutant: mutant)
        let pts: [SIMD4<Float>] = points.map { SIMD4<Float>($0, 0) }
        let pb: MTLBuffer = try worldBuffer(device, pts)
        let sb: MTLBuffer = try worldBuffer(device, packed.shapes)
        let ab: MTLBuffer = try worldBuffer(device, packed.ants)
        let fb: MTLBuffer = try worldBuffer(device, [Float(0)])
        let noField: [Float] = []
        let scene = WorldFrameScene(camera: WorldCamera.wide, ants: ants, field: noField)
        var params: WorldParams = self.params(scene, packed: packed, samples: 1, top: sceneTop(ants))
        guard let ob = device.makeBuffer(length: 16 * max(pts.count, 1), options: .storageModeShared),
              let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder()
        else { throw WorldRenderError.gpu("could not set up the probe") }
        enc.setComputePipelineState(probePSO)
        enc.setBuffer(pb, offset: 0, index: 0)
        enc.setBuffer(ob, offset: 0, index: 1)
        enc.setBytes(&params, length: MemoryLayout<WorldParams>.stride, index: 2)
        enc.setBuffer(sb, offset: 0, index: 3)
        enc.setBuffer(ab, offset: 0, index: 4)
        enc.setBuffer(grainBuffer, offset: 0, index: 5)
        enc.setBuffer(fb, offset: 0, index: 6)
        let width: Int = min(probePSO.maxTotalThreadsPerThreadgroup, 256)
        enc.dispatchThreads(MTLSize(width: pts.count, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        if let e = cb.error { throw WorldRenderError.gpu(e.localizedDescription) }
        let o = ob.contents().assumingMemoryBound(to: SIMD4<Float>.self)
        return (0..<pts.count).map { o[$0] }
    }
}

func worldSavePNG(_ image: WorldImage, to url: URL) throws {
    try? FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    guard let provider = CGDataProvider(dataInfo: nil, data: image.pixels.contents(), size: image.width * image.height * 4,
                                        releaseData: { _, _, _ in }),
          let space = CGColorSpace(name: CGColorSpace.sRGB),
          let cg = CGImage(width: image.width, height: image.height, bitsPerComponent: 8, bitsPerPixel: 32,
                           bytesPerRow: image.width * 4, space: space,
                           bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                           provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent),
          let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)
    else { throw WorldRenderError.png("could not set up the encoder for \(url.path)") }
    CGImageDestinationAddImage(dest, cg, nil)
    guard CGImageDestinationFinalize(dest) else { throw WorldRenderError.png("could not write \(url.path)") }
}
