// The Metal source: one thread per pixel marching rays into a plant made of
// distance functions, as in step 25 — chains of capsules for everything round
// and thin, flat ribbons for wings, folded blades for leaflets. The numbers
// that differ between the two climbing steps (colours, the support's texture,
// the loop's period) are written in as constants by Swift from the species
// file's `Look`, so the tests read the numbers the GPU draws.
//
// Written for step 45 and copied unchanged into steps 46 and 47. Step 48 adds
// petals (white, translucent, withering to a papery tan by a per-blade tint)
// and pods (green capsule chains), and three petal outlines. The distance
// functions are unchanged: a petal is a leaflet blade, a pod a capsule chain.
//
// What is new against step 25:
//
//   * the scene changes every frame, so its shapes come in as buffers sized per
//     frame and their counts ride in the View, not as compiled-in constants.
//   * leaflets: a blade is an outline in its own plane, folded up about its
//     midrib and drooping along it. The outline is a polyline, so its distance
//     is exact; the fold is a mirror and a rotation, which move no distances;
//     the droop bends the blade and its slope is divided out. A first version
//     divided a formula's edge distance by the edge's slope instead, and the
//     test below measured it over-reporting 3.5× far off the blade, because
//     the slope it divided by changed along the blade.
//     The tests measure what is left against what a ray allows.
//   * textures that repeat with the loop. Every pattern on a support is built
//     from noise whose lattice wraps with the loop's period, so the frame after
//     the last one is exactly the first.

import Foundation
import simd

func metal(_ v: SIMD3<Float>) -> String { "float3(\(v.x), \(v.y), \(v.z))" }
func metal(_ v: Float) -> String { "\(v)" }

/// The look of one step: everything the kernel needs that is not geometry.
/// Colours are linear albedos.
struct Look {
    var leafTop: V3
    var leafUnder: V3
    var stem: V3
    var petiole: V3
    var tendril: V3
    var support: V3
    /// 0: a wooden pole round the y axis; 1: jute twine.
    var supportKind: Int
    /// The loop's rise in millimetres: support textures repeat over it.
    var period: Float
    /// Radius of the support, for its texture.
    var supportRadius: Float
    /// How strongly a leaf's upper surface shines: the bean's is hairy and
    /// matt, the sweet pea's waxy.
    var leafGloss: Float
    var keyDirection: V3
    /// Step 48: petals fresh and withered, and the pod.
    var petal: V3
    var petalWithered: V3
    var pod: V3
}

/// How far a ray trusts a distance. Chains and ribbons are exact; what
/// over-reports is the leaflet outline. The tests measure the real factor.
let stepScale: Float = 0.8

/// A leaflet's half-outline: half-width as a fraction of the widest at ξ
/// along the midrib, ξ^a (1 − ξ)^b scaled so its peak is 1. b > 1 draws the tip
/// out to a point (acuminate); a < 1 rounds the base.
func outlineExponents(_ shape: LeafShape) -> SIMD2<Float> {
    switch shape {
    case .beanOvate: return SIMD2<Float>(0.85, 1.6)     // broadly ovate, acuminate
    case .peaElliptic: return SIMD2<Float>(0.75, 0.8)   // ovate-oblong, blunt
    case .stipule: return SIMD2<Float>(0.45, 1.1)       // narrow, pointed
    case .standard: return SIMD2<Float>(1.4, 0.45)      // clawed base, broad rounded top
    case .wing: return SIMD2<Float>(0.9, 0.55)          // obovate-oblong, blunt
    case .keel: return SIMD2<Float>(0.8, 0.9)           // boat-shaped once folded
    }
}

func outlineProfile(_ shape: LeafShape, _ xi: Float) -> Float {
    let e: SIMD2<Float> = outlineExponents(shape)
    let a: Float = e.x
    let b: Float = e.y
    let peak: Float = pow(a / (a + b), a) * pow(b / (a + b), b)
    let x: Float = min(max(xi, 0), 1)
    return pow(x, a) * pow(1 - x, b) / peak
}

/// Segments in each outline polyline, closer together at the two ends where
/// the outline turns fastest.
let outlineSegments: Int = 16

func outlinePolyline(_ shape: LeafShape) -> [SIMD2<Float>] {
    (0...outlineSegments).map { i in
        let xi: Float = 0.5 - 0.5 * cos(Float.pi * Float(i) / Float(outlineSegments))
        return SIMD2<Float>(xi, outlineProfile(shape, xi))
    }
}

func kernelConstants(_ look: Look) -> String {
    var c: [String] = []
    func add(_ name: String, _ value: String) { c.append("constant \(name) = \(value);") }
    add("float STEP_SCALE", metal(stepScale))
    add("float3 LEAF_TOP", metal(look.leafTop))
    add("float3 LEAF_UNDER", metal(look.leafUnder))
    add("float3 STEM_COL", metal(look.stem))
    add("float3 PETIOLE_COL", metal(look.petiole))
    add("float3 TENDRIL_COL", metal(look.tendril))
    add("float3 SUPPORT_COL", metal(look.support))
    add("int SUPPORT_KIND", "\(look.supportKind)")
    add("float PERIOD", metal(look.period))
    add("float SUPPORT_R", metal(look.supportRadius))
    add("float LEAF_GLOSS", metal(look.leafGloss))
    add("float3 KEY_DIR", metal(simd_normalize(look.keyDirection)))
    add("float3 PETAL_COL", metal(look.petal))
    add("float3 WITHER_COL", metal(look.petalWithered))
    add("float3 POD_COL", metal(look.pod))
    add("int OUTLINE_N", "\(outlineSegments)")
    var pts: [String] = []
    for shape in [LeafShape.beanOvate, .peaElliptic, .stipule, .standard, .wing, .keel] {
        for q in outlinePolyline(shape) { pts.append("float2(\(q.x), \(q.y))") }
    }
    c.append("constant float2 OUTLINE[\(pts.count)] = { " + pts.joined(separator: ", ") + " };")
    return c.joined(separator: "\n")
}

let kernelBody: String = """
struct Chunk { float4 sphere; uint first; uint count; uint mat; uint pad; };
struct RibbonG { float4 a; float4 b; float4 n; };
struct LeafG { float4 o; float4 u; float4 v; float4 w; float4 bound; float4 kind; };
struct View { float4 pos; float4 fwd; float4 right; float4 up;
              float tanHalf; uint width; uint height; uint rowOffset;
              uint samples; uint chunkN; uint ribbonN; uint leafN;
              float tubeMin; uint mask; float pad0; float pad1; };

#define ARGS device const float4 *cverts, device const Chunk *chunks, \\
    device const RibbonG *ribbons, device const LeafG *leaves, constant View &V
#define PASS cverts, chunks, ribbons, leaves, V

constant int M_POLE = 1, M_STEM = 2, M_PETIOLE = 3, M_LEAF = 4, M_TENDRIL = 5, M_TWINE = 6, M_STIPULE = 7;
constant int M_PETAL = 8, M_POD = 9;

bool isBlade(int m) { return m == M_LEAF || m == M_STIPULE || m == M_PETAL; }

// ------------------------------------------------------------------ helpers

float capsule(float3 p, float3 a, float3 b, float ra, float rb) {
    float3 ab = b - a;
    float h = clamp(dot(p - a, ab) / max(dot(ab, ab), 1e-12), 0.0, 1.0);
    return length(p - a - ab * h) - mix(ra, rb, h);
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

// 3D value noise whose lattice wraps in y with the loop: the frequency is
// rounded so a whole number of cells fits in PERIOD, and the cell index in y
// wraps there. Anything fixed in the world — the support, the old stem — is
// textured with it, so a loop later it wears the same pattern. (Plain noise3
// broke the loop: a loop later every surface had moved into new cells.)
float hashWrap(float3 i, float cells) {
    i.y = i.y - cells * floor(i.y / cells);
    return hash3(i);
}
float noiseLoop(float3 p, float freq) {
    float cells = max(round(PERIOD * freq), 1.0);
    float3 x = p * (cells / PERIOD);
    float3 i = floor(x);
    float3 f = fract(x);
    f = f * f * (3.0 - 2.0 * f);
    float a = mix(mix(hashWrap(i + float3(0,0,0), cells), hashWrap(i + float3(1,0,0), cells), f.x),
                  mix(hashWrap(i + float3(0,1,0), cells), hashWrap(i + float3(1,1,0), cells), f.x), f.y);
    float b = mix(mix(hashWrap(i + float3(0,0,1), cells), hashWrap(i + float3(1,0,1), cells), f.x),
                  mix(hashWrap(i + float3(0,1,1), cells), hashWrap(i + float3(1,1,1), cells), f.x), f.y);
    return mix(a, b, f.z);
}

// Value noise on a lattice that wraps: `per` cells in each direction, so the
// pattern repeats exactly every `per` units. The support's textures use it
// with the loop's period, which is what makes the loop close.
float wrapHash(float2 i, float2 per) {
    float2 w = i - per * floor(i / per);
    return hash3(float3(w, 7.0));
}
float pnoise(float2 x, float2 per) {
    float2 i = floor(x);
    float2 f = fract(x);
    f = f * f * (3.0 - 2.0 * f);
    float a = wrapHash(i, per), b = wrapHash(i + float2(1, 0), per);
    float c = wrapHash(i + float2(0, 1), per), d = wrapHash(i + float2(1, 1), per);
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

// ------------------------------------------------------------------ shapes

float ribbonD(float3 p, RibbonG r) {
    float3 a = r.a.xyz, b = r.b.xyz, n = r.n.xyz;
    float3 ab = b - a;
    float t = clamp(dot(p - a, ab) / max(dot(ab, ab), 1e-12), 0.0, 1.0);
    float3 c = a + ab * t;
    float u = clamp(dot(p - c, n), -r.a.w, r.a.w);
    return length(p - c - n * u) - r.b.w;
}

// Outlines: each shape's half-outline as a polyline of (ξ along the midrib,
// half-width as a fraction of the widest), written in by Swift from
// `outlineProfile`. Distance to a polyline is exact, and scaling its vertices
// by the blade's length and width before measuring keeps it exact.
float outlineDistance(float x, float s, float L, float W, int shape, thread float &halfWidth) {
    float best = 1e9;
    halfWidth = 0.0;
    float2 p = float2(x, s);
    for (int i = 0; i < OUTLINE_N; i++) {
        float2 a = OUTLINE[shape * (OUTLINE_N + 1) + i] * float2(L, W);
        float2 b = OUTLINE[shape * (OUTLINE_N + 1) + i + 1] * float2(L, W);
        float2 ab = b - a;
        float h = clamp(dot(p - a, ab) / max(dot(ab, ab), 1e-12), 0.0, 1.0);
        best = min(best, length(p - a - ab * h));
        if (x >= a.x && x <= b.x && b.x > a.x) halfWidth = mix(a.y, b.y, (x - a.x) / (b.x - a.x));
    }
    bool inside = x > 0.0 && x < L && s < halfWidth;
    return inside ? -best : best;
}

// Distance to a leaflet. `local` returns (ξ along the midrib, s across the
// half-blade as a fraction of its half-width, which face is nearer: +1 upper).
float leafletD(float3 p, LeafG g, thread float3 &local) {
    float3 q = p - g.o.xyz;
    float L = g.o.w, W = g.u.w, fold = g.v.w, droop = g.w.w;
    float x = dot(q, g.u.xyz);
    float y = dot(q, g.v.xyz);
    float z = dot(q, g.w.xyz);
    float xi = x / L;
    // Droop: the blade's surface sags to z = −droop·L·ξ², so measure from it.
    float xc = clamp(xi, 0.0, 1.0);
    float zs = z + droop * L * xc * xc;
    float dropSlope = 2.0 * droop * xc;
    // Fold: mirror across the midrib, then turn the half-blade flat.
    float ya = abs(y);
    float cf = cos(fold), sf = sin(fold);
    float s = ya * cf + zs * sf;        // across the half-blade
    float h = -ya * sf + zs * cf;       // out of its face
    int shape = int(g.kind.x);
    float wHalf;
    float edge = outlineDistance(x, s, L, W, shape, wHalf);
    edge = max(edge, -s);               // this half only; the mirror gives the other
    float thick = 0.15;
    float hd = (abs(h) - thick) / sqrt(1.0 + dropSlope * dropSlope);
    float2 e = float2(edge, hd);
    local = float3(xi, s / max(W, 1e-3), h >= 0.0 ? 1.0 : -1.0);
    return length(max(e, 0.0)) + min(max(e.x, e.y), 0.0);
}

// ------------------------------------------------------------------ the scene

float sceneSDF(float3 p, ARGS, thread int &mat) {
    float best = 1e9;
    mat = 0;
    for (uint k = 0; k < V.chunkN; k++) {
        Chunk ch = chunks[k];
        int m = int(ch.mat);
        if (((V.mask >> uint(m)) & 1u) == 0u) continue;
        if (length(p - ch.sphere.xyz) - ch.sphere.w > best) continue;
        float rmin = m == M_TENDRIL ? V.tubeMin : 0.0;
        for (uint i = ch.first; i < ch.first + ch.count; i++) {
            float4 a = cverts[i], b = cverts[i + 1];
            float d = capsule(p, a.xyz, b.xyz, max(a.w, rmin), max(b.w, rmin));
            if (d < best) { best = d; mat = m; }
        }
    }
    for (uint k = 0; k < V.ribbonN; k++) {
        RibbonG r = ribbons[k];
        int m = int(r.n.w);
        if (((V.mask >> uint(m)) & 1u) == 0u) continue;
        float d = ribbonD(p, r);
        if (d < best) { best = d; mat = m; }
    }
    for (uint k = 0; k < V.leafN; k++) {
        LeafG g = leaves[k];
        int m = int(g.kind.y);
        if (((V.mask >> uint(m)) & 1u) == 0u) continue;
        float sphere = length(p - g.bound.xyz) - g.bound.w;
        if (sphere > best) continue;
        float3 loc;
        // Far from a blade its estimate under-reports; the bounding sphere is a
        // true lower bound, so take the larger. This also keeps the skip above
        // honest: a skipped blade could never have come in under `best`.
        float d = max(leafletD(p, g, loc), sphere);
        if (d < best) { best = d; mat = m; }
    }
    return best;
}

// The nearest blade's own coordinates, for veins and which face is showing,
// and (w) its tint.
float4 leafLocal(float3 p, ARGS) {
    float best = 1e9;
    float4 out = float4(0.5, 0.5, 1.0, 0.0);
    for (uint k = 0; k < V.leafN; k++) {
        LeafG g = leaves[k];
        if (length(p - g.bound.xyz) - g.bound.w > best) continue;
        float3 loc;
        float d = leafletD(p, g, loc);
        if (d < best) { best = d; out = float4(loc, g.kind.z); }
    }
    return out;
}

// ------------------------------------------------------------------ marching

float3 sceneNormal(float3 p, float e, ARGS) {
    int m;
    float3 k1 = float3(1, -1, -1), k2 = float3(-1, -1, 1), k3 = float3(-1, 1, -1), k4 = float3(1, 1, 1);
    return normalize(k1 * sceneSDF(p + k1 * e, PASS, m) + k2 * sceneSDF(p + k2 * e, PASS, m)
                   + k3 * sceneSDF(p + k3 * e, PASS, m) + k4 * sceneSDF(p + k4 * e, PASS, m));
}

bool march(float3 ro, float3 rd, float pixAngle, thread float &t, thread int &mat, ARGS) {
    t = 50.0;
    float far = 3000.0;
    for (int i = 0; i < 400; i++) {
        float d = sceneSDF(ro + rd * t, PASS, mat);
        if (d < 0.3 * pixAngle * t) return true;
        t += d * STEP_SCALE;
        if (t > far) break;
    }
    return false;
}

// Soft shadows; a leaf passes some light (MODEL: a leaf transmits of the order
// of a tenth to a fifth of the visible light falling on it, more of the green),
// so a shadow ray that crosses one is dimmed and carries on.
float softShadow(float3 ro, float3 rd, float t0, ARGS) {
    float res = 1.0;
    float through = 1.0;
    float t = t0;
    int m;
    for (int i = 0; i < 90; i++) {
        float h = sceneSDF(ro + rd * t, PASS, m);
        bool thin = isBlade(m);
        if (thin && h < 0.3) {
            through *= 0.3;
            t += 1.0;
            if (through < 0.03) break;
            continue;
        }
        if (!thin) res = min(res, 10.0 * h / t);
        t += clamp(h * STEP_SCALE, 0.3, 25.0);
        if (res < 0.003 || t > 600.0) break;
    }
    return clamp(res, 0.0, 1.0) * through;
}

float ambientOcclusion(float3 p, float3 n, float scale, ARGS) {
    float occ = 0.0;
    float weight = 1.0;
    int m;
    for (int i = 1; i <= 5; i++) {
        float h = scale * (0.5 * float(i * i) + 0.3);
        float d = sceneSDF(p + n * h, PASS, m);
        occ += weight * clamp((h - d) / h, 0.0, 1.0);
        weight *= 0.7;
    }
    return clamp(1.0 - 0.45 * occ, 0.0, 1.0);
}

// ------------------------------------------------------------------ light

// Soft daylight like step 25's: a warm key, a cool fill from the camera, and a
// sky over everything.
constant float3 KEY_COL = float3(1.0, 0.97, 0.92) * 2.6;
constant float3 FILL_COL = float3(0.90, 0.94, 1.0) * 0.35;

float3 skyIrradiance(float3 n) {
    return mix(float3(0.22, 0.24, 0.21), float3(0.62, 0.67, 0.74), 0.5 + 0.5 * n.y);
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

float3 shade(float3 p, float3 rd, int mat, float pixSize, ARGS) {
    float e = max(pixSize * 0.4, 0.02);
    float3 n = sceneNormal(p, e, PASS);
    float3 v = -rd;
    if (dot(n, v) < 0.0 && isBlade(mat)) n = -n;
    float occ = ambientOcclusion(p, n, max(pixSize * 1.5, 0.6), PASS);

    float3 albedo = float3(0.5);
    float trans = 0.0;
    float wrap = 0.2;
    float alpha = 0.5;
    float f0 = 0.035;
    float sheen = 0.0;

    if (mat == M_POLE) {
        // A peeled, weathered wooden pole: long grain streaks round it and up
        // it, repeating with the loop.
        float th = atan2(p.z, p.x) / (2.0 * M_PI_F) + 0.5;
        float yy = p.y / PERIOD;
        float g1 = pnoise(float2(th * 40.0, yy * 6.0), float2(40.0, 6.0));
        float g2 = pnoise(float2(th * 90.0, yy * 20.0), float2(90.0, 20.0));
        float g3 = pnoise(float2(th * 12.0, yy * 30.0), float2(12.0, 30.0));
        float grain = 0.55 * g1 + 0.3 * g2 + 0.15 * g3;
        albedo = SUPPORT_COL * (0.55 + 0.9 * grain * grain);
        albedo = mix(albedo, SUPPORT_COL * float3(0.75, 0.78, 0.80), smoothstep(0.62, 0.8, g3) * 0.5);
        alpha = 0.6;
        f0 = 0.03;
    } else if (mat == M_TWINE) {
        // Jute twine: three plies twisted, and fibres.
        float fib = noiseLoop(p, 3.0);
        albedo = SUPPORT_COL * (0.8 + 0.35 * fib);
        alpha = 0.8;
        sheen = 0.15;
    } else if (mat == M_STEM) {
        albedo = STEM_COL * (0.92 + 0.12 * noiseLoop(p, 0.8));
        sheen = 0.25;
        alpha = 0.45;
        trans = 0.15;
    } else if (mat == M_PETIOLE) {
        albedo = PETIOLE_COL;
        sheen = 0.2;
        trans = 0.15;
    } else if (mat == M_TENDRIL) {
        albedo = TENDRIL_COL;
        trans = 0.3;
        wrap = 0.35;
        alpha = 0.35;
        f0 = 0.04;
    } else if (mat == M_POD) {
        // A young pea pod: smooth, green, a little waxy (Mendel's "green
        // colouring of the unripe pod").
        albedo = POD_COL * (0.94 + 0.1 * noiseLoop(p, 0.5));
        alpha = 0.4;
        f0 = 0.045;
        sheen = 0.25;
        trans = 0.2;
        wrap = 0.3;
    } else if (mat == M_PETAL) {
        // White petals: thin, translucent, faint veins running out from the
        // claw; a withering petal goes papery tan (MODEL colours).
        float4 loc = leafLocal(p, PASS);
        albedo = mix(PETAL_COL, WITHER_COL, clamp(loc.w, 0.0, 1.0));
        float vein = 1.0 - smoothstep(0.03, 0.08, abs(fract(loc.y * 5.0 + 0.5) - 0.5));
        albedo *= 1.0 - 0.06 * vein * smoothstep(0.1, 0.4, loc.x);
        trans = 0.6;
        wrap = 0.45;
        alpha = 0.55;
        f0 = 0.03;
        sheen = 0.1;
    } else if (mat == M_LEAF || mat == M_STIPULE) {
        float3 loc = leafLocal(p, PASS).xyz;
        bool upper = loc.z > 0.0;
        albedo = upper ? LEAF_TOP : LEAF_UNDER;
        // Veins: the midrib, and pinnate side veins curving out to the margin.
        float s = loc.y;
        float mid = 1.0 - smoothstep(0.02, 0.06, s);
        float side = abs(fract(loc.x * 7.0 - s * 1.6) - 0.5);
        float sideVein = (1.0 - smoothstep(0.03, 0.07, side)) * smoothstep(0.08, 0.2, s) * (1.0 - smoothstep(0.75, 0.95, s));
        float vein = max(mid, sideVein * 0.6);
        albedo = mix(albedo, albedo * float3(1.25, 1.2, 1.05) + float3(0.02, 0.03, 0.0), vein * (upper ? 0.55 : 0.8));
        // Mottling in the blade's own coordinates, so it grows and turns with
        // the leaf instead of swimming across it.
        albedo *= 0.93 + 0.1 * noise3(float3(loc.x * 24.0, loc.y * 9.0, 3.0));
        trans = 0.45;
        wrap = 0.3;
        alpha = upper ? mix(0.6, 0.3, LEAF_GLOSS) : 0.7;
        f0 = upper ? mix(0.03, 0.05, LEAF_GLOSS) : 0.02;
        sheen = upper ? 0.05 : 0.2;
    }

    float nk = dot(n, KEY_DIR);
    float3 faceL = (trans > 0.0 && nk < 0.0) ? -n : n;
    float sh = softShadow(p + faceL * e * 4.0, KEY_DIR, max(e * 6.0, 0.4), PASS);
    float front = max((nk + wrap) / (1.0 + wrap), 0.0);
    float back = max(-nk, 0.0) * trans;
    float nf = dot(n, v);
    float fill = max((nf + wrap) / (1.0 + wrap), 0.0);
    // Light through a leaf comes out greener (MODEL).
    float3 through = float3(0.75, 1.0, 0.55);
    float3 diffuse = albedo * (KEY_COL * (front + back * through) * sh + FILL_COL * fill + skyIrradiance(n) * occ);
    float3 spec = KEY_COL * ggx(n, v, KEY_DIR, alpha, f0) * sh + FILL_COL * ggx(n, v, v, alpha, f0) * 0.5;
    float rimAmt = pow(1.0 - max(nf, 0.0), 3.0) * sheen;
    return diffuse + spec + float3(rimAmt) * occ * 0.6;
}

// Luminance filmic shoulder, then the colour scaled to match (step 20).
float3 toneMap(float3 x) {
    float L = dot(x, float3(0.2126, 0.7152, 0.0722));
    float Lm = L * (1.0 + L / 16.0) / (1.0 + L);
    float3 c = x * (Lm / max(L, 1e-5)) * 1.12;
    float over = max(max(c.r, max(c.g, c.b)) - 1.0, 0.0);
    c = mix(c, float3(1.0), clamp(over, 0.0, 1.0));
    c = clamp(c, 0.0, 1.0);
    return select(1.055 * pow(c, 1.0 / 2.4) - 0.055, 12.92 * c, c <= 0.0031308);
}

float3 cameraRay(constant View &W, float2 pixel) {
    float aspect = float(W.width) / float(W.height);
    float sx = (2.0 * pixel.x / float(W.width) - 1.0) * aspect * W.tanHalf;
    float sy = (1.0 - 2.0 * pixel.y / float(W.height)) * W.tanHalf;
    return normalize(W.fwd.xyz + sx * W.right.xyz + sy * W.up.xyz);
}

kernel void render(device float4 *pixels [[buffer(0)]],
                   device float4 *aux [[buffer(1)]],
                   constant View &V [[buffer(2)]],
                   device const float4 *cverts [[buffer(3)]],
                   device const Chunk *chunks [[buffer(4)]],
                   device const RibbonG *ribbons [[buffer(5)]],
                   device const LeafG *leaves [[buffer(6)]],
                   uint2 gid [[thread_position_in_grid]]) {
    uint x = gid.x;
    uint y = gid.y + V.rowOffset;
    if (x >= V.width || y >= V.height) return;
    float pixAngle = 2.0 * V.tanHalf / float(V.height);
    float3 sum = float3(0.0);
    float cover = 0.0;
    uint S = V.samples;
    for (uint sy = 0; sy < S; sy++) {
        for (uint sx = 0; sx < S; sx++) {
            float2 jitter = float2((float(sx) + 0.5) / float(S), (float(sy) + 0.5) / float(S));
            float3 rd = cameraRay(V, float2(x, y) + jitter);
            float t;
            int mat;
            if (march(V.pos.xyz, rd, pixAngle, t, mat, PASS)) {
                float3 p = V.pos.xyz + rd * t;
                sum += toneMap(shade(p, rd, mat, pixAngle * t, PASS));
                cover += 1.0;
            }
        }
    }
    float n = float(S * S);
    pixels[y * V.width + x] = float4(sum / n, cover / n);

    float3 rd = cameraRay(V, float2(x, y) + 0.5);
    float t;
    int mat;
    float4 a = float4(0.0);
    if (march(V.pos.xyz, rd, pixAngle, t, mat, PASS)) a = float4(float(mat), t, 0.0, 0.0);
    aux[y * V.width + x] = a;
}

kernel void probe(device const float4 *points [[buffer(0)]],
                  device float2 *out [[buffer(1)]],
                  constant View &V [[buffer(2)]],
                  device const float4 *cverts [[buffer(3)]],
                  device const Chunk *chunks [[buffer(4)]],
                  device const RibbonG *ribbons [[buffer(5)]],
                  device const LeafG *leaves [[buffer(6)]],
                  uint id [[thread_position_in_grid]]) {
    int mat;
    float d = sceneSDF(points[id].xyz, PASS, mat);
    out[id] = float2(d, float(mat));
}
"""
