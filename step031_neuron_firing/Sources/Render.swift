// Step 27's neuron, with one nerve impulse running out along its axon: one
// GPU thread per pixel marching rays into a scene made of distance functions,
// as step 20's teeth were. Copied from step 27 and changed only where the
// impulse needs it — the geometry is untouched, and a frame with no glow in
// it is step 27's still, bit for bit.
//
// The whole cell is two kinds of shape: a sphere for the soma, and cones with
// rounded ends for every piece of process — a dendrite segment that narrows,
// an axon that doesn't, a bouton that swells. Each is an exact distance, and
// they are melted together with a smooth minimum, so where a dendrite leaves
// the soma it flares instead of meeting it at a crease.
//
// The soma is softly translucent: when a ray lands on it, the shader looks
// on through the cytoplasm for the nucleus and its nucleolus, which are plain
// spheres met analytically, and lets them show in proportion to how much
// cytoplasm lies in front of them.
//
// The camera is orthographic, so a micrometre is the same number of pixels
// everywhere in the frame and the scale bar is true across the whole picture.
//
// The impulse is shading only. Each piece of process carries the times the
// impulse reaches its two ends (Impulse.swift); a lit point finds the piece
// it belongs to — the nearest — reads its arrival time there, and tints the
// membrane by how recently the impulse passed.

import CoreGraphics
import CoreText
import Foundation
import ImageIO
import Metal
import UniformTypeIdentifiers
import simd

// MARK: - the frame and the light

/// Width of the picture in micrometres; the height follows from the aspect.
let frameWidth: Float = 540
let frameCentre = SIMD2<Float>(0, 0)

/// One big soft key from the upper left and in front, a dimmer fill from the
/// lower right — the look of step 20.
let keyDirection = simd_normalize(SIMD3<Float>(-0.55, 0.75, 0.6))
let fillDirection = simd_normalize(SIMD3<Float>(0.6, -0.25, 0.8))
let keyColour = SIMD3<Float>(1.0, 0.98, 0.95) * 1.9
let fillColour = SIMD3<Float>(0.95, 0.97, 1.0) * 0.35

/// The ground, in display values: near white.
let backgroundDisplay: Float = 0.965

/// Colours. STYLISED: a living neuron is colourless and nearly transparent;
/// it only has a colour once it is stained. A calm sea-glass green for the
/// cell, a cooler blue for the nucleus, a deep indigo for the nucleolus.
/// Given as sRGB hex, the way a designer would pick them.
let cellHex: UInt32 = 0x7DB5AA
let nucleusHex: UInt32 = 0x8C9FE0
let nucleolusHex: UInt32 = 0x2E3470

func linear(_ hex: UInt32) -> SIMD3<Float> {
    func channel(_ shift: UInt32) -> Float {
        let byte: UInt32 = (hex >> shift) & 0xFF
        let c: Float = Float(byte) / 255
        return c <= 0.04045 ? c / 12.92 : pow((c + 0.055) / 1.055, 2.4)
    }
    return SIMD3<Float>(channel(16), channel(8), channel(0))
}

/// How far into the cytoplasm the eye can see before the nucleus fades out,
/// µm. MODEL — stylised, like the colour.
let cytoplasmClearDepth: Float = 60
let somaClarity: Float = 0.9

/// Where the nucleus and nucleolus sit. A motoneuron's nucleus is central;
/// the nucleolus is placed a little off centre so it reads. MODEL.
func nucleusCentre(_ n: Neuron) -> SIMD3<Float> { n.soma }
func nucleolusCentre(_ n: Neuron) -> SIMD3<Float> { n.soma + SIMD3<Float>(1.8, 1.5, 2.5) }

/// Blend radii, µm: how far the soma melts into its stems, and forks into
/// their branches. MODEL.
let somaBlend: Float = 7.0
let branchBlend: Float = 1.2

/// The glow. STYLISED, like the cell's colour: a warm gold against the
/// sea-glass green, mixed into the membrane's colour and adding a little
/// light of its own, so the axon stays round and lit as it brightens.
let glowHex: UInt32 = 0xF2B54A
let glowMix: Float = 0.6
let glowEmission: Float = 0.35

/// The boutons brighten more than the axon as the impulse arrives: that is
/// where it ends and where transmitter would be released. STYLISED.
let boutonGlowGain: Float = 1.8

/// Every distance here is exact or a lower bound (sphere, rounded cone,
/// smooth minimum, a slab cut away), so a ray can trust nearly all of it.
let stepScale: Float = 0.9

// MARK: - GPU layout

/// One rounded cone: (a, ra), (b, rb), (kind, 0, 0, 0). Three float4s, so
/// there is no padding to get wrong between Swift and Metal.
struct GPUSegment {
    var a: SIMD4<Float>
    var b: SIMD4<Float>
    var info: SIMD4<Float>
}

struct Params {
    var width: UInt32
    var height: UInt32
    var rowOffset: UInt32
    var samples: UInt32
    var axonCount: UInt32
    var time: Float
}

/// info = (kind, arrival at a, arrival at b, jump past the break); an arrival
/// of −1 means the impulse never enters this piece. Only info.x shapes the
/// cell, so the geometry is the same whatever the impulse does.
func gpuSegments(_ list: [Neurite], _ arrivals: [Arrival?]) -> [GPUSegment] {
    zip(list, arrivals).map { (s: Neurite, arr: Arrival?) -> GPUSegment in
        let a: Arrival = arr ?? Arrival(ta: -1, tb: -1, breakShift: 0)
        return GPUSegment(a: SIMD4<Float>(s.a, s.ra), b: SIMD4<Float>(s.b, s.rb),
                          info: SIMD4<Float>(Float(s.kind.rawValue), a.ta, a.tb, a.breakShift))
    }
}

/// A sphere around each dendrite tree, so a point far from a tree can skip it.
func treeBounds(_ n: Neuron) -> [SIMD4<Float>] {
    stride(from: 0, to: n.dendrites.count, by: segmentsPerTree).map { first in
        let tree: ArraySlice<Neurite> = n.dendrites[first..<(first + segmentsPerTree)]
        var c = SIMD3<Float>(0, 0, 0)
        for s in tree { c += (s.a + s.b) / 2 }
        c /= Float(tree.count)
        var r: Float = 0
        for s in tree { r = max(r, simd_distance(c, s.a) + s.ra, simd_distance(c, s.b) + s.rb) }
        return SIMD4<Float>(c, r)
    }
}

// MARK: - the kernel

func metal(_ v: SIMD3<Float>) -> String { "float3(\(v.x), \(v.y), \(v.z))" }

func kernelSource(_ n: Neuron) -> String {
    let aspectHeight: Float = frameWidth * 9 / 16
    return """
    #include <metal_stdlib>
    using namespace metal;

    struct Seg { float4 a; float4 b; float4 info; };
    struct Params { uint width; uint height; uint rowOffset; uint samples; uint axonCount; float time; };

    constant uint TREES = \(n.dendrites.count / segmentsPerTree);
    constant uint PER_TREE = \(segmentsPerTree);
    constant float STEP_SCALE = \(stepScale);
    constant float HIT_EPS = 0.01;

    constant float FRAME_W = \(frameWidth);
    constant float FRAME_H = \(aspectHeight);
    constant float2 FRAME_C = float2(\(frameCentre.x), \(frameCentre.y));

    constant float3 SOMA = \(metal(n.soma));
    constant float SOMA_R = \(somaDiameter / 2);
    constant float3 NUC = \(metal(nucleusCentre(n)));
    constant float NUC_R = \(nucleusDiameter / 2);
    constant float3 NUCLEOLUS = \(metal(nucleolusCentre(n)));
    constant float NUCLEOLUS_R = \(nucleolusDiameter / 2);
    constant float SOMA_K = \(somaBlend);
    constant float BRANCH_K = \(branchBlend);

    constant float3 BREAK_C = \(metal(n.breakCentre));
    constant float3 BREAK_N = \(metal(n.breakNormal));
    constant float BREAK_HALF = \(breakHalfGap);

    constant float3 KEY_DIR = \(metal(keyDirection));
    constant float3 FILL_DIR = \(metal(fillDirection));
    constant float3 KEY_COL = \(metal(keyColour));
    constant float3 FILL_COL = \(metal(fillColour));
    constant float BG = \(backgroundDisplay);

    constant float3 CELL = \(metal(linear(cellHex)));
    constant float3 NUCLEUS = \(metal(linear(nucleusHex)));
    constant float3 NUCLEOLUS_COL = \(metal(linear(nucleolusHex)));
    constant float CLEAR_DEPTH = \(cytoplasmClearDepth);
    constant float CLARITY = \(somaClarity);

    constant float LOOP = \(loopSeconds);
    constant float RISE = \(glowRise);
    constant float DECAY = \(glowDecay);
    constant float FADE0 = \(glowFadeStart);
    constant float FADE1 = \(glowFadeEnd);
    constant float3 GLOW = \(metal(linear(glowHex)));
    constant float GLOW_MIX = \(glowMix);
    constant float GLOW_EMIT = \(glowEmission);
    constant float BOUTON_GAIN = \(boutonGlowGain);

    float smin(float a, float b, float k) {
        float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
        return mix(b, a, h) - k * h * (1.0 - h);
    }

    // Exact distance to a cone with spherical caps of radii r1 at a and r2
    // at b (Inigo Quilez's round cone).
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

    // Materials: 1 soma, 2 process.
    float sceneSDF(float3 p, thread int &mat, constant Seg *dend, constant float4 *bounds,
                   constant Seg *axon, uint axonCount) {
        float dS = length(p - SOMA) - SOMA_R;
        float dN = 1e9;
        for (uint t = 0; t < TREES; t++) {
            // Exact skip: a tree further than dN + BRANCH_K away cannot move
            // a smooth minimum of radius BRANCH_K at all.
            if (length(p - bounds[t].xyz) - bounds[t].w > dN + BRANCH_K) continue;
            for (uint i = t * PER_TREE; i < (t + 1) * PER_TREE; i++) {
                Seg s = dend[i];
                dN = smin(dN, roundCone(p, s.a.xyz, s.b.xyz, s.a.w, s.b.w), BRANCH_K);
            }
        }
        for (uint i = 0; i < axonCount; i++) {
            Seg s = axon[i];
            float d = roundCone(p, s.a.xyz, s.b.xyz, s.a.w, s.b.w);
            // The break: two parallel slanted cuts through the trunk.
            if (int(s.info.x) == 2) d = max(d, BREAK_HALF - abs(dot(p - BREAK_C, BREAK_N)));
            dN = smin(dN, d, BRANCH_K);
        }
        mat = dS < dN ? 1 : 2;
        return smin(dS, dN, SOMA_K);
    }

    #define SCENE(q, m) sceneSDF(q, m, dend, bounds, axon, axonCount)
    #define SCENE_ARGS constant Seg *dend, constant float4 *bounds, constant Seg *axon, uint axonCount

    float3 sceneNormal(float3 p, SCENE_ARGS) {
        const float e = 0.02;
        int m;
        float3 k1 = float3(1, -1, -1), k2 = float3(-1, -1, 1), k3 = float3(-1, 1, -1), k4 = float3(1, 1, 1);
        return normalize(k1 * SCENE(p + k1 * e, m) + k2 * SCENE(p + k2 * e, m)
                       + k3 * SCENE(p + k3 * e, m) + k4 * SCENE(p + k4 * e, m));
    }

    bool march(float3 ro, float3 rd, thread float &t, thread int &mat, SCENE_ARGS) {
        t = 0.0;
        for (int i = 0; i < 300; i++) {
            float d = SCENE(ro + rd * t, mat);
            if (d < HIT_EPS) return true;
            t += d * STEP_SCALE;
            if (t > 600.0) break;
        }
        return false;
    }

    float softShadow(float3 ro, float3 rd, SCENE_ARGS) {
        float res = 1.0;
        float t = 0.3;
        int m;
        for (int i = 0; i < 64; i++) {
            float h = SCENE(ro + rd * t, m);
            res = min(res, 8.0 * h / t);
            t += clamp(h * STEP_SCALE, 0.2, 12.0);
            if (res < 0.002 || t > 250.0) break;
        }
        return clamp(res, 0.0, 1.0);
    }

    float ambientOcclusion(float3 p, float3 n, SCENE_ARGS) {
        float occ = 0.0;
        float w = 1.0;
        int m;
        for (int i = 1; i <= 5; i++) {
            float h = 1.2 * float(i * i) + 0.5;
            float d = SCENE(p + n * h, m);
            occ += w * clamp((h - d) / h, 0.0, 1.0);
            w *= 0.7;
        }
        return clamp(1.0 - 0.4 * occ, 0.0, 1.0);
    }

    // The room: a bright white ceiling and a grey floor, for fill from all round.
    float3 ambient(float3 n) {
        return mix(float3(0.16, 0.16, 0.17), float3(0.42, 0.43, 0.45), 0.5 + 0.5 * n.y);
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

    // Where a ray from ro first enters a sphere, or -1.
    float sphereHit(float3 ro, float3 rd, float3 c, float r) {
        float3 oc = ro - c;
        float b = dot(oc, rd);
        float h = b * b - (dot(oc, oc) - r * r);
        if (h < 0.0) return -1.0;
        float t = -b - sqrt(h);
        return t > 0.0 ? t : -1.0;
    }

    // How much of a sphere the ray passes through, as a fraction of its
    // diameter: 1 through the middle, 0 at a grazing edge. Seen through
    // cytoplasm, a nucleus has no hard rim, so this softens its outline.
    float chordFraction(float3 ro, float3 rd, float3 c, float r) {
        float3 oc = ro - c;
        float b = dot(oc, rd);
        float h = b * b - (dot(oc, oc) - r * r);
        return h > 0.0 ? sqrt(h) / r : 0.0;
    }

    // Light inside the cell is scattered, so a sphere seen in there is lit
    // softly: half by the key's side, half from all round.
    float3 innerLight(float3 n) {
        return 0.6 * KEY_COL * (0.45 + 0.55 * max(dot(n, KEY_DIR), 0.0)) + ambient(n);
    }

    // Impulse.swift's `pulse`: the glow d seconds after the impulse arrives.
    float pulse(float since) {
        float d = since - LOOP * floor(since / LOOP);
        if (d > LOOP - RISE) d -= LOOP;
        float up = smoothstep(-RISE, 0.0, d);
        float down = d < 0.0 ? 1.0 : exp(-d / DECAY);
        return up * down * (1.0 - smoothstep(FADE0, FADE1, d));
    }

    // Impulse.swift's `arrival`: along the piece, plus the break's jump.
    float arrivalAt(float3 p, Seg s) {
        float3 ba = s.b.xyz - s.a.xyz;
        float u = clamp(dot(p - s.a.xyz, ba) / dot(ba, ba), 0.0, 1.0);
        float t = mix(s.info.y, s.info.z, u);
        if (s.info.w != 0.0 && dot(p - BREAK_C, BREAK_N) > 0.0) t += s.info.w;
        return t;
    }

    // How brightly the membrane at p glows at this moment: it belongs to the
    // nearest piece of the cell, and glows as that piece's arrival says. The
    // soma, and any piece the impulse never enters, stay dark.
    float glowAt(float3 p, float time, SCENE_ARGS) {
        float best = length(p - SOMA) - SOMA_R;
        float g = 0.0;
        for (uint i = 0; i < TREES * PER_TREE; i++) {
            Seg s = dend[i];
            float d = roundCone(p, s.a.xyz, s.b.xyz, s.a.w, s.b.w);
            if (d < best) { best = d; g = s.info.y < 0.0 ? 0.0 : pulse(time - arrivalAt(p, s)); }
        }
        for (uint i = 0; i < axonCount; i++) {
            Seg s = axon[i];
            float d = roundCone(p, s.a.xyz, s.b.xyz, s.a.w, s.b.w);
            if (d < best) {
                best = d;
                g = s.info.y < 0.0 ? 0.0 : pulse(time - arrivalAt(p, s));
                if (int(s.info.x) == 4) g *= BOUTON_GAIN;
            }
        }
        return g;
    }

    float3 shade(float3 p, float3 rd, int mat, float time, SCENE_ARGS) {
        float3 n = sceneNormal(p, dend, bounds, axon, axonCount);
        float3 v = -rd;
        float occ = ambientOcclusion(p, n, dend, bounds, axon, axonCount);
        float sh = softShadow(p + n * 0.1, KEY_DIR, dend, bounds, axon, axonCount);
        const float wrap = 0.2;
        float key = max((dot(n, KEY_DIR) + wrap) / (1.0 + wrap), 0.0) * sh;
        float fill = max((dot(n, FILL_DIR) + wrap) / (1.0 + wrap), 0.0);
        float3 diffuse = CELL * (KEY_COL * key + FILL_COL * fill + ambient(n) * occ);
        // Only where the impulse is: everywhere else this is step 27 exactly.
        float g = glowAt(p, time, dend, bounds, axon, axonCount);
        if (g > 0.0) {
            float3 albedo = mix(CELL, GLOW, min(GLOW_MIX * g, 1.0));
            diffuse = albedo * (KEY_COL * key + FILL_COL * fill + ambient(n) * occ) + GLOW * (GLOW_EMIT * g);
        }
        float3 spec = KEY_COL * ggx(n, v, KEY_DIR, 0.35, 0.02) * sh
                    + FILL_COL * ggx(n, v, FILL_DIR, 0.35, 0.02);
        if (mat == 1) {
            float tn = sphereHit(p, rd, NUC, NUC_R);
            if (tn > 0.0) {
                float3 q = p + rd * tn;
                float3 inner = NUCLEUS * innerLight(normalize(q - NUC));
                float tl = sphereHit(q, rd, NUCLEOLUS, NUCLEOLUS_R);
                if (tl > 0.0) {
                    float3 r = q + rd * tl;
                    float3 dark = NUCLEOLUS_COL * innerLight(normalize(r - NUCLEOLUS));
                    inner = mix(inner, dark, 0.95 * smoothstep(0.0, 0.7, chordFraction(q, rd, NUCLEOLUS, NUCLEOLUS_R)));
                }
                float edge = smoothstep(0.0, 0.6, chordFraction(p, rd, NUC, NUC_R));
                diffuse = mix(diffuse, inner * mix(1.0, occ, 0.5), CLARITY * exp(-tn / CLEAR_DEPTH) * edge);
            }
        }
        return diffuse + spec;
    }

    // Step 20's tone map: a filmic shoulder on LUMINANCE, the colour scaled
    // to match, so a chosen colour keeps its hue as it brightens.
    float3 toneMap(float3 x) {
        float L = dot(x, float3(0.2126, 0.7152, 0.0722));
        float Lm = L * (1.0 + L / 16.0) / (1.0 + L);
        float3 c = x * (Lm / max(L, 1e-5)) * 1.12;
        float over = max(max(c.r, max(c.g, c.b)) - 1.0, 0.0);
        c = mix(c, float3(1.0), clamp(over, 0.0, 1.0));
        c = clamp(c, 0.0, 1.0);
        return select(1.055 * pow(c, 1.0 / 2.4) - 0.055, 12.92 * c, c <= 0.0031308);
    }

    kernel void neuron(device uchar4 *pixels [[buffer(0)]],
                       constant Params &P [[buffer(1)]],
                       constant Seg *dend [[buffer(2)]],
                       constant float4 *bounds [[buffer(3)]],
                       constant Seg *axon [[buffer(4)]],
                       uint2 gid [[thread_position_in_grid]]) {
        uint x = gid.x;
        uint y = gid.y + P.rowOffset;
        if (x >= P.width || y >= P.height) return;
        uint axonCount = P.axonCount;
        float3 sum = float3(0.0);
        uint S = P.samples;
        for (uint sy = 0; sy < S; sy++) {
            for (uint sx = 0; sx < S; sx++) {
                float2 px = float2(x, y) + float2((float(sx) + 0.5) / float(S), (float(sy) + 0.5) / float(S));
                float3 ro = float3(FRAME_C.x + (px.x / float(P.width) - 0.5) * FRAME_W,
                                   FRAME_C.y + (0.5 - px.y / float(P.height)) * FRAME_H, 300.0);
                float3 rd = float3(0.0, 0.0, -1.0);
                float t;
                int mat;
                if (march(ro, rd, t, mat, dend, bounds, axon, axonCount)) {
                    sum += toneMap(shade(ro + rd * t, rd, mat, P.time, dend, bounds, axon, axonCount));
                } else {
                    sum += float3(BG);
                }
            }
        }
        float3 c = sum / float(S * S);
        pixels[y * P.width + x] = uchar4(uchar3(round(clamp(c, 0.0, 1.0) * 255.0)), 255);
    }

    kernel void probe(device const float4 *points [[buffer(0)]],
                      device float2 *out [[buffer(1)]],
                      constant Seg *dend [[buffer(2)]],
                      constant float4 *bounds [[buffer(3)]],
                      constant Seg *axon [[buffer(4)]],
                      constant uint &axonCount [[buffer(5)]],
                      uint id [[thread_position_in_grid]]) {
        int mat;
        float d = SCENE(points[id].xyz, mat);
        out[id] = float2(d, float(mat));
    }

    // The glow at arbitrary points and one moment, for the tests.
    kernel void probeGlow(device const float4 *points [[buffer(0)]],
                          device float *out [[buffer(1)]],
                          constant Seg *dend [[buffer(2)]],
                          constant float4 *bounds [[buffer(3)]],
                          constant Seg *axon [[buffer(4)]],
                          constant uint &axonCount [[buffer(5)]],
                          constant float &time [[buffer(6)]],
                          uint id [[thread_position_in_grid]]) {
        out[id] = glowAt(points[id].xyz, time, dend, bounds, axon, axonCount);
    }
    """
}

// MARK: - running it

enum NeuronError: Error, CustomStringConvertible {
    case noMetalDevice
    case kernel(String)
    case gpu(String)
    case png(String)

    var description: String {
        switch self {
        case .noMetalDevice: return "no Metal GPU found"
        case .kernel(let s): return "could not compile the kernel: \(s)"
        case .gpu(let s): return "GPU error: \(s)"
        case .png(let s): return "could not write PNG: \(s)"
        }
    }
}

/// The errors step 8's GIF encoder throws.
enum RenderError: Error {
    case kernelCompile(String)
    case gpu(String)
}

/// The finished picture: RGBA bytes, top row first.
struct NeuronImage {
    let width: Int
    let height: Int
    let pixels: MTLBuffer

    func rgba(_ x: Int, _ y: Int) -> SIMD4<UInt8> {
        let p = pixels.contents().advanced(by: (y * width + x) * 4).assumingMemoryBound(to: UInt8.self)
        return SIMD4<UInt8>(p[0], p[1], p[2], p[3])
    }
}

/// Where a point in the cell lands in a picture of this size, in pixels.
func project(_ p: SIMD3<Float>, width: Int, height: Int) -> (x: Int, y: Int) {
    let frameHeight: Float = frameWidth * 9 / 16
    let u: Float = (p.x - frameCentre.x) / frameWidth + 0.5
    let v: Float = 0.5 - (p.y - frameCentre.y) / frameHeight
    return (Int(u * Float(width)), Int(v * Float(height)))
}

func findDevice() throws -> MTLDevice {
    if let d = MTLCreateSystemDefaultDevice() ?? MTLCopyAllDevices().first { return d }
    throw NeuronError.noMetalDevice
}

/// The kernel, its pipelines, and the cell's buffers, ready to dispatch.
struct Scene {
    let queue: MTLCommandQueue
    let library: MTLLibrary
    let dendrites: MTLBuffer
    let bounds: MTLBuffer
    let axon: MTLBuffer
    let axonCount: Int

    init(_ n: Neuron, _ imp: Impulse, on device: MTLDevice) throws {
        let options = MTLCompileOptions()
        // Precise maths, as in every step.
        options.fastMathEnabled = false
        do { library = try device.makeLibrary(source: kernelSource(n), options: options) } catch {
            throw NeuronError.kernel("\(error)")
        }
        var d: [GPUSegment] = gpuSegments(n.dendrites, imp.dendrites)
        var b: [SIMD4<Float>] = treeBounds(n)
        var a: [GPUSegment] = gpuSegments(n.axon, imp.axon)
        let stride: Int = MemoryLayout<GPUSegment>.stride
        guard let q = device.makeCommandQueue(),
              let db = device.makeBuffer(bytes: &d, length: stride * d.count, options: .storageModeShared),
              let bb = device.makeBuffer(bytes: &b, length: 16 * b.count, options: .storageModeShared),
              let ab = device.makeBuffer(bytes: &a, length: stride * a.count, options: .storageModeShared)
        else { throw NeuronError.gpu("could not allocate buffers") }
        queue = q
        dendrites = db
        bounds = bb
        axon = ab
        axonCount = a.count
    }

    func pipeline(_ name: String) throws -> MTLComputePipelineState {
        guard let f = library.makeFunction(name: name) else { throw NeuronError.kernel("no kernel \(name)") }
        do { return try library.device.makeComputePipelineState(function: f) } catch {
            throw NeuronError.kernel("\(error)")
        }
    }
}

/// The kernel compiled once and an image buffer, for drawing frame after
/// frame. Renders in horizontal bands, one command buffer each, so no single
/// piece of GPU work runs long enough to trip the watchdog — or hog a shared
/// GPU.
final class NeuronRenderer {
    let device: MTLDevice
    let scene: Scene
    let pso: MTLComputePipelineState
    let image: NeuronImage

    init(_ n: Neuron, _ imp: Impulse, width: Int, height: Int, on device: MTLDevice) throws {
        self.device = device
        scene = try Scene(n, imp, on: device)
        pso = try scene.pipeline("neuron")
        guard let pixels = device.makeBuffer(length: width * height * 4, options: .storageModeShared) else {
            throw NeuronError.gpu("could not allocate the image")
        }
        image = NeuronImage(width: width, height: height, pixels: pixels)
    }

    /// Draws the cell at `time` seconds into the loop; returns GPU seconds.
    @discardableResult
    func render(time: Float, samples: Int) throws -> Double {
        let width: Int = image.width
        let height: Int = image.height
        var gpu: Double = 0
        var row: Int = 0
        while row < height {
            let rows: Int = min(60, height - row)
            guard let cb = scene.queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else {
                throw NeuronError.gpu("could not make a command buffer")
            }
            var params = Params(width: UInt32(width), height: UInt32(height), rowOffset: UInt32(row),
                                samples: UInt32(samples), axonCount: UInt32(scene.axonCount), time: time)
            enc.setComputePipelineState(pso)
            enc.setBuffer(image.pixels, offset: 0, index: 0)
            enc.setBytes(&params, length: MemoryLayout<Params>.stride, index: 1)
            enc.setBuffer(scene.dendrites, offset: 0, index: 2)
            enc.setBuffer(scene.bounds, offset: 0, index: 3)
            enc.setBuffer(scene.axon, offset: 0, index: 4)
            let w: Int = pso.threadExecutionWidth
            let group = MTLSize(width: w, height: max(pso.maxTotalThreadsPerThreadgroup / w / 4, 1), depth: 1)
            enc.dispatchThreads(MTLSize(width: width, height: rows, depth: 1), threadsPerThreadgroup: group)
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            if let e = cb.error { throw NeuronError.gpu(e.localizedDescription) }
            gpu += cb.gpuEndTime - cb.gpuStartTime
            row += rows
        }
        return gpu
    }
}

/// One frame, from scratch: step 27's call, with a moment and an impulse.
/// At rest (the default) it is step 27's still.
func renderNeuron(_ n: Neuron, width: Int, height: Int, samples: Int, time: Float = 0,
                  impulse: Impulse? = nil, on device: MTLDevice) throws -> (image: NeuronImage, gpuSeconds: Double) {
    let r = try NeuronRenderer(n, impulse ?? buildImpulse(n), width: width, height: height, on: device)
    let gpu: Double = try r.render(time: time, samples: samples)
    return (r.image, gpu)
}

/// The scene's distance and material at arbitrary points, from the same kernel
/// source the render uses, so the test reads the thing that drew the picture.
func probeScene(_ n: Neuron, _ points: [SIMD3<Float>], on device: MTLDevice) throws -> [SIMD2<Float>] {
    let scene = try Scene(n, buildImpulse(n), on: device)
    let pso = try scene.pipeline("probe")
    var pts: [SIMD4<Float>] = points.map { SIMD4<Float>($0, 0) }
    var count = UInt32(scene.axonCount)
    guard let pb = device.makeBuffer(bytes: &pts, length: 16 * pts.count, options: .storageModeShared),
          let ob = device.makeBuffer(length: 8 * pts.count, options: .storageModeShared),
          let cb = scene.queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder()
    else { throw NeuronError.gpu("could not set up the probe") }
    enc.setComputePipelineState(pso)
    enc.setBuffer(pb, offset: 0, index: 0)
    enc.setBuffer(ob, offset: 0, index: 1)
    enc.setBuffer(scene.dendrites, offset: 0, index: 2)
    enc.setBuffer(scene.bounds, offset: 0, index: 3)
    enc.setBuffer(scene.axon, offset: 0, index: 4)
    enc.setBytes(&count, length: 4, index: 5)
    enc.dispatchThreads(MTLSize(width: pts.count, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: min(pso.maxTotalThreadsPerThreadgroup, 256), height: 1, depth: 1))
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    if let e = cb.error { throw NeuronError.gpu(e.localizedDescription) }
    let out = ob.contents().assumingMemoryBound(to: SIMD2<Float>.self)
    return (0..<pts.count).map { out[$0] }
}

/// The glow the kernel would paint at each point, at each of several moments
/// — the very function the render calls — so the tests read what the picture
/// shows. One row of results per moment.
func probeGlow(_ n: Neuron, _ imp: Impulse, _ points: [SIMD3<Float>], times: [Float],
               on device: MTLDevice) throws -> [[Float]] {
    let scene = try Scene(n, imp, on: device)
    let pso = try scene.pipeline("probeGlow")
    var pts: [SIMD4<Float>] = points.map { SIMD4<Float>($0, 0) }
    var count = UInt32(scene.axonCount)
    guard let pb = device.makeBuffer(bytes: &pts, length: 16 * pts.count, options: .storageModeShared),
          let ob = device.makeBuffer(length: 4 * pts.count * times.count, options: .storageModeShared),
          let cb = scene.queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder()
    else { throw NeuronError.gpu("could not set up the glow probe") }
    enc.setComputePipelineState(pso)
    enc.setBuffer(pb, offset: 0, index: 0)
    enc.setBuffer(scene.dendrites, offset: 0, index: 2)
    enc.setBuffer(scene.bounds, offset: 0, index: 3)
    enc.setBuffer(scene.axon, offset: 0, index: 4)
    enc.setBytes(&count, length: 4, index: 5)
    for (k, time) in times.enumerated() {
        var t: Float = time
        enc.setBuffer(ob, offset: 4 * pts.count * k, index: 1)
        enc.setBytes(&t, length: 4, index: 6)
        enc.dispatchThreads(MTLSize(width: pts.count, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(pso.maxTotalThreadsPerThreadgroup, 256), height: 1, depth: 1))
    }
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    if let e = cb.error { throw NeuronError.gpu(e.localizedDescription) }
    let out = ob.contents().assumingMemoryBound(to: Float.self)
    return (0..<times.count).map { k in (0..<pts.count).map { out[k * pts.count + $0] } }
}

// MARK: - the scale bar, and saving

/// Length of the scale bar, µm: one soma diameter.
let scaleBarMicrons: Float = 50

/// Draws a 50 µm bar and its label into the bottom-right corner, on the CPU,
/// straight into the image's pixels. The camera is orthographic, so the bar
/// is true anywhere in the frame.
func drawScaleBar(_ image: NeuronImage) {
    guard let space = CGColorSpace(name: CGColorSpace.sRGB),
          let ctx = CGContext(data: image.pixels.contents(), width: image.width, height: image.height,
                              bitsPerComponent: 8, bytesPerRow: image.width * 4, space: space,
                              bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
    else { return }
    let s: CGFloat = CGFloat(image.width) / 1920
    let length: CGFloat = CGFloat(scaleBarMicrons / frameWidth) * CGFloat(image.width)
    let right: CGFloat = CGFloat(image.width) - 90 * s
    let grey = CGColor(srgbRed: 0.33, green: 0.34, blue: 0.36, alpha: 1)
    ctx.setFillColor(grey)
    ctx.fill(CGRect(x: right - length, y: 70 * s, width: length, height: 6 * s))
    let font = CTFontCreateWithName("Helvetica Neue" as CFString, 26 * s, nil)
    let attrs: [NSAttributedString.Key: Any] = [
        NSAttributedString.Key(kCTFontAttributeName as String): font,
        NSAttributedString.Key(kCTForegroundColorAttributeName as String): grey,
    ]
    let line = CTLineCreateWithAttributedString(NSAttributedString(string: "50 µm", attributes: attrs))
    let w: CGFloat = CTLineGetTypographicBounds(line, nil, nil, nil)
    ctx.textPosition = CGPoint(x: right - length / 2 - w / 2, y: 88 * s)
    CTLineDraw(line, ctx)
}

/// The caption, bottom left: that the impulse is slowed, by how much, and
/// the two honesties a viewer would otherwise miss — no myelin, and the
/// metre inside the break cut short. Numbers come from Impulse.swift.
func captionLines() -> [String] {
    let factor: Int = Int((slowdown / 100_000).rounded()) * 100_000
    var grouped: String = String(factor % 1000).leftPadded(3)
    var rest: Int = factor / 1000
    while rest > 0 {
        grouped = (rest >= 1000 ? String(rest % 1000).leftPadded(3) : String(rest)) + "," + grouped
        rest /= 1000
    }
    return [
        "One nerve impulse, slowed ~\(grouped)× (real: ~\(Int(conductionVelocity.rounded())) m/s, ~\(Int((spikeDuration * 1000).rounded())) ms spike)",
        "Myelin not drawn, so it glides; a real motor axon jumps node to node",
        "The break stands for ~1 m of axon; its \(Int((realBreakSeconds * 1000).rounded())) ms is cut short",
    ]
}

extension String {
    func leftPadded(_ n: Int) -> String { String(repeating: "0", count: max(n - count, 0)) + self }
}

func drawCaption(_ image: NeuronImage) {
    guard let space = CGColorSpace(name: CGColorSpace.sRGB),
          let ctx = CGContext(data: image.pixels.contents(), width: image.width, height: image.height,
                              bitsPerComponent: 8, bytesPerRow: image.width * 4, space: space,
                              bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
    else { return }
    let s: CGFloat = CGFloat(image.width) / 1920
    let grey = CGColor(srgbRed: 0.33, green: 0.34, blue: 0.36, alpha: 1)
    let font = CTFontCreateWithName("Helvetica Neue" as CFString, 22 * s, nil)
    let attrs: [NSAttributedString.Key: Any] = [
        NSAttributedString.Key(kCTFontAttributeName as String): font,
        NSAttributedString.Key(kCTForegroundColorAttributeName as String): grey,
    ]
    let lines: [String] = captionLines()
    for (i, text) in lines.enumerated() {
        let line = CTLineCreateWithAttributedString(NSAttributedString(string: text, attributes: attrs))
        let fromBottom: CGFloat = CGFloat(lines.count - 1 - i)
        let base: CGFloat = 70
        let lead: CGFloat = 32
        let y: CGFloat = (base + lead * fromBottom) * s
        ctx.textPosition = CGPoint(x: 90 * s, y: y)
        CTLineDraw(line, ctx)
    }
}

func savePNG(_ image: NeuronImage, to url: URL) throws {
    try? FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    guard let provider = CGDataProvider(dataInfo: nil, data: image.pixels.contents(), size: image.width * image.height * 4,
                                        releaseData: { _, _, _ in }),
          let space = CGColorSpace(name: CGColorSpace.sRGB),
          let cg = CGImage(width: image.width, height: image.height, bitsPerComponent: 8, bitsPerPixel: 32,
                           bytesPerRow: image.width * 4, space: space,
                           bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                           provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent),
          let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)
    else { throw NeuronError.png("could not set up the encoder for \(url.path)") }
    CGImageDestinationAddImage(dest, cg, nil)
    guard CGImageDestinationFinalize(dest) else { throw NeuronError.png("could not write \(url.path)") }
}
