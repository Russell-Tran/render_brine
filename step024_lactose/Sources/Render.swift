// Two cutaways of the same gut, marched per pixel in Metal: the terminal ileum
// running through the ileocecal valve into the caecum and colon, a break, and
// the rectum. Everything solid is a distance function, as in step 20; the
// particles are a second set of primitives — spheres — intersected exactly and
// composited by depth against what the march found.
//
//   * The gut is two hollow tubes: the ileum a strip of wall in the (x, radius)
//     plane with a rounded lip where it ends inside the colon, the colon a
//     tube with a spherical caecal end. The ileum's lumen is carved through
//     the caecal wall, so the ileum opens into the colon as a papilla — the
//     valve.
//   * Villi are ONE capsule, repeated around and along the ileum by domain
//     repetition, and they stop at the last villus before the valve. Crypts are
//     ONE pit, repeated the same way — in the ileum between the villi, in the
//     colon across a flat lining.
//   * The peristaltic wave is a travelling change of radius: a ring of
//     contraction with a widened, relaxed segment ahead of it.
//   * The cutaway is max(scene, z): everything nearer the camera than z = 0 is
//     removed, and the cut face is coloured by tissue layer.
//
// A third region of the frame is a close-up of the brush border — a row of
// columnar cells, cut the same way — half of them with lactase and half
// without.
//
// This is step 23's renderer, copied. What changed for step 24: the lower
// panel's contents are coloured as loose, milky-then-fermenting stool instead
// of rice-water; the villus tips of the upper panel carry a gold band for
// lactase; bubbles are shaded as bubbles; and the close-up's cells carry
// lactase, SGLT1 and GLUT2 instead of CFTR and NKCC1.

import CoreGraphics
import Foundation
import ImageIO
import Metal
import UniformTypeIdentifiers
import simd

// MARK: - layout

/// Where everything sits in the frame, for any frame width. Laid out at 1280
/// wide: two 360-pixel panels, then a 240-pixel strip holding the close-up on
/// the left and the key on the right.
struct Layout {
    let width: Int
    var panelHeight: Int { width * 360 / 1280 }
    var stripHeight: Int { width * 240 / 1280 }
    var height: Int { 2 * panelHeight + stripHeight }
    var k: Float { Float(width) / 1280 }
    /// mm per pixel in the two gut panels.
    var scale: Float { panelViewSpan / Float(width) }
    var insetX: Int { Int(16 * k) }
    var insetY: Int { 2 * panelHeight + Int(14 * k) }
    var insetW: Int { Int(600 * k) }
    var insetH: Int { Int(212 * k) }
    /// µm per pixel in the close-up.
    var insetScale: Float { insetViewSpan / Float(insetW) }
}

/// The gut panels show 262 mm across; the close-up 66 µm.
let panelViewSpan: Float = 262
let insetViewSpan: Float = 66
/// Where each camera looks, and how far down it tilts.
let panelCentre = SIMD3<Float>((gutStartX + gutEndX) * 0.5 - 0.5, -5.0, 0)
let panelTilt: Float = 13 * Float.pi / 180
let insetCentre = SIMD3<Float>(0, 0.6, 0)
let insetTilt: Float = 10 * Float.pi / 180

func cameraUp(_ tilt: Float) -> SIMD3<Float> { SIMD3(0, cos(tilt), -sin(tilt)) }
func cameraForward(_ tilt: Float) -> SIMD3<Float> { SIMD3(0, -sin(tilt), -cos(tilt)) }

/// Where a world point lands in the frame, in top-down pixels.
func project(_ p: SIMD3<Float>, panel: Int, layout: Layout) -> SIMD2<Float> {
    if panel == 2 {
        let rel: SIMD3<Float> = p - insetCentre
        let sx: Float = rel.x
        let sy: Float = simd_dot(rel, cameraUp(insetTilt))
        let px: Float = Float(layout.insetX) + Float(layout.insetW) * 0.5 + sx / layout.insetScale
        let py: Float = Float(layout.insetY) + Float(layout.insetH) * 0.5 - sy / layout.insetScale
        return SIMD2(px, py)
    }
    let rel: SIMD3<Float> = p - panelCentre
    let sy: Float = simd_dot(rel, cameraUp(panelTilt))
    let px: Float = Float(layout.width) * 0.5 + rel.x / layout.scale
    let top: Float = Float(panel * layout.panelHeight)
    let py: Float = top + Float(layout.panelHeight) * 0.5 - sy / layout.scale
    return SIMD2(px, py)
}

// MARK: - GPU layout

struct GPUParticle {
    var posR: SIMD4<Float>      // centre, radius
    var colour: SIMD4<Float>    // linear rgb, alpha
    var info: SIMD4<Float>      // panel, species, route, 0
}

struct Params {
    var width: UInt32
    var height: UInt32
    var rowOffset: UInt32
    var samples: UInt32
    var time: Float
    var tilesX: UInt32
    var lumpCount: UInt32
    var pad: UInt32
}

let tileSize: Int = 16

/// How far a ray may trust a distance. The wave's changing radius, the
/// repeated villi and pits and the stool's stretched ellipsoids all make the
/// distance over-report a little; the test measures by how much.
let stepScale: Float = 0.65

func srgbToLinear(_ c: Float) -> Float {
    c <= 0.04045 ? c / 12.92 : pow((c + 0.055) / 1.055, 2.4)
}

func metal(_ v: SIMD3<Float>) -> String { "float3(\(v.x), \(v.y), \(v.z))" }

// MARK: - the stool (upper panel only)

/// Lumps of stool in the normal colon: spaced 18 mm apart, creeping one spacing
/// a loop, growing as they go downstream — water leaving, solids consolidating —
/// and melting together into one formed mass by the rectum. MODEL shapes.
let lumpSpacing: Float = 18
let lumpCount: Int = 8
let lumpStartX: Float = 8

func stoolLumps(at t: Float) -> [SIMD4<Float>] {
    let w: Wave = normalWave
    var out: [SIMD4<Float>] = []
    let span: Float = lumpSpacing * Float(lumpCount)
    for j in 0..<lumpCount {
        let xb: Float = lumpStartX + fract((Float(j) * lumpSpacing + w.contentMM * t / loopSeconds) / span) * span
        let d: Float = waveOffset(xb, panel: 0, t: t)
        let x: Float = xb + w.surge * surgeFraction(d, wavelength: w.wavelength)
        let grow: Float = smoothstep(lumpStartX, 100, x)
        let r: Float = colonRadius * 0.66 * pow(grow, 0.9)
        let blend: Float = 0.6 + 6.0 * smoothstep(75, 110, x)
        let y: Float = colonAxisY - colonRadius * 0.18 * (1 - grow)
        out.append(SIMD4(x, y, blend, r))
    }
    return out
}

// MARK: - the kernel

func kernelSource(layout: Layout) -> String {
    let nWave: Wave = normalWave
    let cWave: Wave = lowerWave
    return """
    #include <metal_stdlib>
    using namespace metal;

    struct Params { uint width; uint height; uint rowOffset; uint samples; float time; uint tilesX; uint lumpCount; uint pad; };
    struct Particle { float4 posR; float4 colour; float4 info; };

    constant float STEP_SCALE = \(stepScale);
    constant uint TILE = \(tileSize);

    // layout
    constant float W = \(Float(layout.width));
    constant float PH = \(Float(layout.panelHeight));
    constant float SCALE = \(layout.scale);
    constant float IX = \(Float(layout.insetX));
    constant float IY = \(Float(layout.insetY));
    constant float IW = \(Float(layout.insetW));
    constant float IHH = \(Float(layout.insetH));
    constant float ISCALE = \(layout.insetScale);
    constant float3 PC = \(metal(panelCentre));
    constant float3 PUP = \(metal(cameraUp(panelTilt)));
    constant float3 PFWD = \(metal(cameraForward(panelTilt)));
    constant float3 ICN = \(metal(insetCentre));
    constant float3 IUP = \(metal(cameraUp(insetTilt)));
    constant float3 IFWD = \(metal(cameraForward(insetTilt)));

    // the tubes
    constant float RI0 = \(ileumRadius);
    constant float WI0 = \(ileumWall);
    constant float RC0 = \(colonRadius);
    constant float WC0 = \(colonWall);
    constant float YI = \(ileumAxisY);
    constant float YC = \(colonAxisY);
    constant float XLIP = \(lipX);
    constant float XB1 = \(breakStartX);
    constant float XB2 = \(breakEndX);

    // the mucosa
    constant float VH = \(villusHeight);
    constant float VRAD = \(villusRadius);
    constant float VP = \(villusPitch);
    constant float VRING = \(Float(villusRing));
    constant float VX0 = \(lastVillusX);
    constant float ICD = \(ilealCryptDepth);
    constant float CRAD = \(cryptRadius);
    constant float CCD = \(colonCryptDepth);
    constant float CP = \(colonCryptPitch);
    constant float CRING = \(Float(colonCryptRing));
    constant float CSTART = \(colonCryptStartX);
    constant float IMUC = \(ilealMucosa);
    constant float CMUC = \(colonMucosa);
    constant float SUBM = \(submucosa);
    constant float EPI = 0.26;

    // the wave, per panel
    constant float WL0 = \(nWave.wavelength);
    constant float WL1 = \(cWave.wavelength);
    constant float WS0 = \(nWave.speed);
    constant float WS1 = \(cWave.speed);
    constant float WA0 = \(nWave.contraction);
    constant float WA1 = \(cWave.contraction);
    constant float WB0 = \(nWave.relaxation);
    constant float WB1 = \(cWave.relaxation);
    constant float CW = \(contractionWidth);
    constant float RL = \(relaxationLead);
    constant float RW = \(relaxationWidth);

    // the close-up (µm)
    constant float CELL_P = \(insetCellPitch);
    constant float CELL_G = \(insetGap);
    constant float CELL_H = \(insetCellHeight);
    constant float CELL_N = \(Float(insetCellCount));
    constant float CELL_X0 = \(insetFirstCellX);
    constant float CELL_D = 6.0;
    constant float LPH_CELLS = \(Float(insetLactaseCells));

    constant float3 KEY = \(metal(simd_normalize(SIMD3<Float>(-0.35, 0.78, 0.52))));
    constant float3 BG = float3(0.010, 0.013, 0.017);

    // ---------------------------------------------------------------- helpers

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

    // ---------------------------------------------------------------- the wave

    float waveOffset(float x, int panel, float t) {
        float lam = panel == 0 ? WL0 : WL1;
        float sp = panel == 0 ? WS0 : WS1;
        float ph = (x - sp * t) / lam;
        return (ph - floor(ph + 0.5)) * lam;
    }
    // (total fractional change of radius, contraction alone)
    float2 waveShape(float d, int panel) {
        float A = panel == 0 ? WA0 : WA1;
        float B = panel == 0 ? WB0 : WB1;
        float a = d / CW;
        float c = A * exp(-a * a);
        float b = (d - RL) / RW;
        float r = B * exp(-b * b);
        return float2(r - c, c);
    }
    float envI(float x) { return 1.0 - smoothstep(-34.0, -24.0, x); }
    float envC(float x) { return smoothstep(4.0, 18.0, x); }

    struct Local { float RI; float WI; float RC; float WC; float con; };

    Local localGut(float x, int panel, float t) {
        float2 g = waveShape(waveOffset(x, panel, t), panel);
        float eI = envI(x);
        float eC = envC(x);
        Local L;
        L.RI = RI0 * (1.0 + eI * g.x);
        L.WI = WI0 * (1.0 + 0.9 * eI * g.y);
        L.RC = RC0 * (1.0 + eC * g.x);
        L.WC = WC0 * (1.0 + 0.9 * eC * g.y);
        L.con = max(eI, eC) * g.y / (panel == 0 ? WA0 : WA1);
        return L;
    }

    // ---------------------------------------------------------------- pieces

    // A point's (radial, tangential) coordinates in the nearest of n equal
    // sectors around an axis — the repetition that makes one villus a ring.
    float2 ringLocal(float2 yz, float n) {
        float cell = 2.0 * M_PI_F / n;
        float th = atan2(yz.y, yz.x);
        float ang = round(th / cell) * cell;
        float c = cos(ang);
        float s = sin(ang);
        return float2(c * yz.x + s * yz.y, -s * yz.x + c * yz.y);
    }

    float villi(float3 p, float R) {
        float2 ab = ringLocal(float2(p.y - YI, p.z), VRING);
        float u = p.x - VX0;
        float du = u - VP * round(u / VP);
        float a = clamp(ab.x, R - VH + VRAD, R + 0.4);
        float d = length(float3(du, ab.x - a, ab.y)) - VRAD;
        return max(d, p.x - (VX0 + 0.5 * VP));
    }

    float ilealPits(float3 p, float R) {
        float2 ab = ringLocal(float2(p.y - YI, p.z), VRING);
        float u = p.x - (VX0 - 0.5 * VP);
        float du = u - VP * round(u / VP);
        float a = clamp(ab.x, R - 0.6, R + ICD - CRAD);
        float d = length(float3(du, ab.x - a, ab.y)) - CRAD;
        return max(d, p.x - VX0);
    }

    float colonPits(float3 p, float R) {
        float2 ab = ringLocal(float2(p.y - YC, p.z), CRING);
        float du = p.x - CP * round(p.x / CP);
        float a = clamp(ab.x, R - 0.6, R + CCD - CRAD);
        float d = length(float3(du, ab.x - a, ab.y)) - CRAD;
        return max(d, CSTART - p.x);
    }

    // The ileal wall in the (x, radius) plane: a strip from R to R + W that
    // ends in a half-round lip at XLIP.
    float ileumWall(float3 p, float R, float Wl) {
        float rho = length(float2(p.y - YI, p.z));
        float h = 0.5 * Wl;
        float dx = max(p.x - (XLIP - h), 0.0);
        return length(float2(dx, rho - (R + h))) - h;
    }

    // The colonic wall: a tube for x >= 0, a spherical shell (the caecum) behind.
    float colonRadial(float3 p) {
        float3 q = float3(p.x, p.y - YC, p.z);
        return q.x >= 0.0 ? length(q.yz) : length(q);
    }
    float colonWall(float3 p, float R, float Wl) {
        return abs(colonRadial(p) - (R + 0.5 * Wl)) - 0.5 * Wl;
    }

    float stoolSDF(float3 p, constant float4 *lumps, uint n) {
        float d = 1e9;
        for (uint i = 0; i < n; i++) {
            float4 L = lumps[i];
            if (L.w < 0.3) continue;
            float3 q = p - float3(L.x, L.y, 0.0);
            float3 rad = float3(L.w * 1.3, L.w, L.w);
            float e = (length(q / rad) - 1.0) * L.w;
            d = smin(d, e, L.z);
        }
        return d;
    }

    // ---------------------------------------------------------------- the gut

    // Materials: 1 ileal wall, 2 villus, 3 colonic wall, 5 stool.
    // `cut` says the cut plane is what bounds the point.
    float gutUncut(float3 p, int panel, float t, constant float4 *lumps, uint nl, thread int &mat) {
        Local L = localGut(p.x, panel, t);
        float rhoI = length(float2(p.y - YI, p.z));

        float ileW = ileumWall(p, L.RI, L.WI);
        // Pits only matter within reach of the inner surface; skipping them
        // elsewhere only ever under-reports, which is safe.
        float pitI = abs(rhoI - L.RI) < ICD + 1.5 ? ilealPits(p, L.RI) : 1e9;
        float ileumPiece = max(ileW, -pitI);

        float d = ileumPiece;
        mat = 1;
        // The villi live in a band of radius; outside it, the band's own
        // distance is a lower bound, so skip them when it cannot win.
        float band = max(max((L.RI - VH) - rhoI, rhoI - (L.RI + 0.4)), p.x - (VX0 + 0.5 * VP + VRAD));
        if (band < d) {
            float v = villi(p, L.RI);
            if (v < d) { d = v; mat = 2; }
        }

        float ileLumen = max(rhoI - L.RI, p.x - XLIP);
        float radial = colonRadial(p);
        float colW = colonWall(p, L.RC, L.WC);
        float colonPiece = colW;
        float Rloc = p.x < 0.0 ? sqrt(max(RC0 * RC0 - p.x * p.x, 1.0)) : L.RC;
        if (abs(radial - L.RC) < CCD + 1.5 && p.x > CSTART - 1.0) colonPiece = max(colW, -colonPits(p, Rloc));
        // The ileum's lumen and its crypts run on through the caecal wall.
        colonPiece = max(max(colonPiece, -ileLumen), -pitI);
        float slab = max(XB1 - p.x, p.x - XB2);
        colonPiece = max(colonPiece, -slab);
        if (colonPiece < d) { d = colonPiece; mat = 3; }

        if (panel == 0 && nl > 0 && p.x > 0.0) {
            float s = stoolSDF(p, lumps, nl);
            s = max(s, (radial - L.RC) + 0.5);
            s = max(s, -slab);
            if (s < d) { d = s; mat = 5; }
        }
        return d;
    }

    float gutSDF(float3 p, int panel, float t, constant float4 *lumps, uint nl, thread int &mat, thread bool &cut) {
        float d = gutUncut(p, panel, t, lumps, nl, mat);
        cut = p.z > d;
        return max(d, p.z);
    }

    // Inside the lumen (negative) — for the fluid.
    float lumenSDF(float3 p, int panel, float t) {
        Local L = localGut(p.x, panel, t);
        float rhoI = length(float2(p.y - YI, p.z));
        float ileLumen = max(rhoI - L.RI, p.x - XLIP);
        float colLumen = colonRadial(p) - L.RC;
        float slab = max(XB1 - p.x, p.x - XB2);
        return max(min(ileLumen, colLumen), -slab);
    }

    // ---------------------------------------------------------------- the close-up

    // Materials: 6 cell, 7 the backdrop behind (lumen above, blood side below).
    float insetSDF(float3 p, thread int &mat, thread bool &cut) {
        float k = clamp(round((p.x - CELL_X0) / CELL_P), 0.0, CELL_N - 1.0);
        float3 q = p - float3(CELL_X0 + k * CELL_P, 0.0, 0.0);
        float3 b = float3(0.5 * (CELL_P - CELL_G), 0.5 * CELL_H, CELL_D);
        float r = 0.9;
        float3 qq = abs(q) - b + r;
        float d = length(max(qq, 0.0)) + min(max(qq.x, max(qq.y, qq.z)), 0.0) - r;
        mat = 6;
        cut = p.z > d;
        d = max(d, p.z);
        float back = p.z + CELL_D + 2.5;
        if (back < d) { d = back; mat = 7; cut = false; }
        return d;
    }

    // ---------------------------------------------------------------- scene

    float sceneSDF(float3 p, int panel, float t, constant float4 *lumps, uint nl, thread int &mat, thread bool &cut) {
        if (panel == 2) return insetSDF(p, mat, cut);
        return gutSDF(p, panel, t, lumps, nl, mat, cut);
    }

    float3 sceneNormal(float3 p, int panel, float t, constant float4 *lumps, uint nl) {
        float e = panel == 2 ? 0.01 : 0.02;
        int m;
        bool c;
        float3 k1 = float3(1, -1, -1), k2 = float3(-1, -1, 1), k3 = float3(-1, 1, -1), k4 = float3(1, 1, 1);
        return normalize(k1 * sceneSDF(p + k1 * e, panel, t, lumps, nl, m, c) + k2 * sceneSDF(p + k2 * e, panel, t, lumps, nl, m, c)
                       + k3 * sceneSDF(p + k3 * e, panel, t, lumps, nl, m, c) + k4 * sceneSDF(p + k4 * e, panel, t, lumps, nl, m, c));
    }

    bool march(float3 ro, float3 rd, int panel, float t, constant float4 *lumps, uint nl,
               float tStart, float tEnd, thread float &th, thread int &mat, thread bool &cut) {
        th = tStart;
        float eps = panel == 2 ? 0.004 : 0.008;
        for (int i = 0; i < 300; i++) {
            float d = sceneSDF(ro + rd * th, panel, t, lumps, nl, mat, cut);
            if (d < eps) return true;
            th += d * STEP_SCALE;
            if (th > tEnd) break;
        }
        return false;
    }

    float ambientOcclusion(float3 p, float3 n, int panel, float t, constant float4 *lumps, uint nl) {
        float occ = 0.0;
        float w = 1.0;
        int m;
        bool c;
        float s = panel == 2 ? 0.5 : 1.0;
        for (int i = 1; i <= 4; i++) {
            float h = s * (0.25 + 0.55 * float(i * i) * 0.5);
            float d = sceneSDF(p + n * h, panel, t, lumps, nl, m, c);
            occ += w * clamp((h - d) / h, 0.0, 1.0);
            w *= 0.7;
        }
        return clamp(1.0 - 0.5 * occ, 0.0, 1.0);
    }

    float softShadow(float3 ro, float3 rd, int panel, float t, constant float4 *lumps, uint nl) {
        float res = 1.0;
        float s = 0.15;
        int m;
        bool c;
        for (int i = 0; i < 28; i++) {
            float h = sceneSDF(ro + rd * s, panel, t, lumps, nl, m, c);
            res = min(res, 8.0 * h / s);
            s += clamp(h * STEP_SCALE, 0.1, 3.0);
            if (res < 0.01 || s > 40.0) break;
        }
        return clamp(res, 0.0, 1.0);
    }

    // ---------------------------------------------------------------- colour

    // Tissue in section, by layer, as a textbook cutaway colours it.
    constant float3 C_EPI = float3(0.36, 0.07, 0.16);
    constant float3 C_LP = float3(0.70, 0.27, 0.29);
    constant float3 C_SUB = float3(0.80, 0.60, 0.48);
    constant float3 C_CIRC = float3(0.42, 0.06, 0.06);
    constant float3 C_LONG = float3(0.54, 0.10, 0.08);
    constant float3 C_SER = float3(0.82, 0.70, 0.64);
    constant float3 C_LACT = float3(0.93, 0.86, 0.66);
    constant float3 C_ACTIVE = float3(0.85, 0.10, 0.06);
    // Lactase (LPH) on the brush border of the villus tips: a marker, not a molecule.
    constant float3 C_LPH = float3(0.95, 0.66, 0.16);
    constant float LPH_BAND = 0.8;

    float3 layerColour(float depth, float wallT, float mucosa, float con, float nearSurface) {
        if (nearSurface < EPI && depth < mucosa + 0.5) return C_EPI;
        if (depth < mucosa) return C_LP;
        if (depth < mucosa + SUBM) return C_SUB;
        if (depth > wallT - 0.18) return C_SER;
        float muscle = wallT - mucosa - SUBM;
        float f = (depth - mucosa - SUBM) / max(muscle, 0.1);
        float3 m = f < 0.62 ? C_CIRC : C_LONG;
        // Contracting circular muscle flushes brighter: this is the ring.
        if (f < 0.62) m = mix(m, C_ACTIVE, clamp(con, 0.0, 1.0) * 0.8);
        return m;
    }

    float3 sectionColour(float3 p, int panel, float t, int mat, constant float4 *lumps, uint nl) {
        int m2;
        float inside = -gutUncut(p, panel, t, lumps, nl, m2);   // distance to the nearest tissue surface
        Local L = localGut(p.x, panel, t);
        if (mat == 5) {
            // Low-frequency mottling only: the lumps move, and fine texture
            // fixed in space would shimmer across them.
            float n = noise3(p * 0.35);
            float rim = smoothstep(0.0, 1.2, inside);
            return mix(float3(0.10, 0.045, 0.015), mix(float3(0.22, 0.11, 0.04), float3(0.30, 0.16, 0.06), n), rim);
        }
        if (mat == 2) {
            // A villus in section: epithelium round the edge, lamina propria
            // inside, and the central lacteal up its middle.
            float2 ab = ringLocal(float2(p.y - YI, p.z), VRING);
            float u = p.x - VX0;
            float du = u - VP * round(u / VP);
            if (inside < EPI) return (panel == 0 && ab.x < L.RI - VH + LPH_BAND) ? C_LPH : C_EPI;
            bool lacteal = abs(du) < 0.09 && ab.x > L.RI - VH + 0.6;
            return lacteal ? C_LACT : C_LP;
        }
        if (mat == 1) {
            float rho = length(float2(p.y - YI, p.z));
            float depth = rho - L.RI;
            // The valve's lips: their outer face is in the colon, lined by colonic epithelium.
            if (p.x > -24.0 && depth > L.WI - EPI - 0.05) return C_EPI;
            return layerColour(depth, L.WI, IMUC, L.con, inside);
        }
        float radial = colonRadial(p);
        float R = p.x < 0.0 ? RC0 : L.RC;
        return layerColour(radial - R, L.WC, CMUC, L.con, inside);
    }

    float3 surfaceAlbedo(float3 p, int panel, float t, int mat, float3 n) {
        Local L = localGut(p.x, panel, t);
        if (mat == 5) {
            float k = noise3(p * 1.3);
            return mix(float3(0.13, 0.055, 0.02), float3(0.24, 0.12, 0.045), k);
        }
        if (mat == 2 && panel == 0) {
            float2 ab = ringLocal(float2(p.y - YI, p.z), VRING);
            if (ab.x < L.RI - VH + LPH_BAND) return mix(float3(0.80, 0.52, 0.12), C_LPH, noise3(p * 3.0));
        }
        bool serosa;
        if (mat == 3) serosa = colonRadial(p) > (p.x < 0.0 ? RC0 : L.RC) + 0.5 * L.WC;
        else serosa = length(float2(p.y - YI, p.z)) > L.RI + 0.5 * L.WI && p.x < -24.0;
        if (serosa) return float3(0.50, 0.30, 0.26) * (0.9 + 0.2 * noise3(p * 0.8));
        float k = noise3(p * 2.2);
        if (mat == 3) return mix(float3(0.50, 0.19, 0.16), float3(0.60, 0.26, 0.21), k);
        return mix(float3(0.50, 0.12, 0.12), float3(0.62, 0.18, 0.17), k);
    }

    float3 insetSection(float3 p) {
        float k = clamp(round((p.x - CELL_X0) / CELL_P), 0.0, CELL_N - 1.0);
        float3 q = p - float3(CELL_X0 + k * CELL_P, 0.0, 0.0);
        float bx = 0.5 * (CELL_P - CELL_G);
        float top = 0.5 * CELL_H;
        // SGLT1 on the apical membrane and GLUT2 on the basolateral, in every
        // cell; lactase in the brush border of the left four only.
        if (abs(q.x) < 1.5 && q.y > top - 1.25 && q.y <= top - 0.8) return float3(0.03, 0.52, 0.40);
        if (abs(q.x) < 1.5 && q.y < -top + 1.05) return float3(0.18, 0.30, 0.66);
        if (k < LPH_CELLS && q.y > top - 0.8) return mix(float3(0.80, 0.52, 0.12), float3(0.98, 0.72, 0.22), step(0.5, fract(q.x * 2.6)));
        // Tight junctions: the seal at the top of each lateral membrane.
        if (abs(q.x) > bx - 0.55 && q.y > top - 2.0 && q.y < top - 0.6) return float3(0.16, 0.03, 0.06);
        // Brush border.
        if (q.y > top - 0.8) return mix(float3(0.55, 0.26, 0.36), float3(0.72, 0.42, 0.50), step(0.5, fract(q.x * 2.6)));
        // Nucleus, basal.
        float2 nq = (q.xy - float2(0.0, -2.6)) / float2(2.0, 2.7);
        if (dot(nq, nq) < 1.0) return mix(float3(0.24, 0.12, 0.36), float3(0.34, 0.18, 0.48), noise3(p * 3.0));
        // Membrane.
        float3 qq = abs(q) - float3(bx, top, CELL_D) + 0.9;
        float d = length(max(qq, 0.0)) + min(max(qq.x, max(qq.y, qq.z)), 0.0) - 0.9;
        if (d > -0.22) return float3(0.48, 0.22, 0.26);
        return mix(float3(0.78, 0.52, 0.50), float3(0.86, 0.62, 0.58), noise3(p * 1.5));
    }

    float3 insetBackdrop(float3 p) {
        float top = 0.5 * CELL_H;
        if (p.y > top) return float3(0.05, 0.10, 0.14);          // crypt lumen, watery
        if (p.y < -top) return float3(0.26, 0.08, 0.09);        // lamina propria: the blood side
        return float3(0.03, 0.02, 0.03);                        // down between cells
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

    float3 shade(float3 p, float3 rd, int panel, float t, int mat, bool cut, constant float4 *lumps, uint nl) {
        float3 v = -rd;
        if (panel == 2) {
            if (mat == 7) return insetBackdrop(p);
            float3 n = sceneNormal(p, panel, t, lumps, nl);
            float occ = ambientOcclusion(p, n, panel, t, lumps, nl);
            float3 albedo = cut ? insetSection(p) : float3(0.70, 0.40, 0.44);
            float key = max(dot(n, KEY), 0.0);
            float3 c = albedo * (0.55 + 0.6 * key) * occ;
            return c + (cut ? 0.0 : 0.25 * ggx(n, v, KEY, 0.35, 0.02));
        }
        float3 n = sceneNormal(p, panel, t, lumps, nl);
        float occ = ambientOcclusion(p, n, panel, t, lumps, nl);
        if (cut) {
            float3 albedo = sectionColour(p, panel, t, mat, lumps, nl);
            // A cut face is flat and matte; light it evenly so the layers read.
            return albedo * (0.78 + 0.25 * occ);
        }
        float sh = softShadow(p + n * 0.05, KEY, panel, t, lumps, nl);
        float3 albedo = surfaceAlbedo(p, panel, t, mat, n);
        float3 bump = float3(noise3(p * 2.5), noise3(p * 2.5 + 11.0), noise3(p * 2.5 + 23.0)) - 0.5;
        float3 nb = normalize(n + 0.12 * (bump - n * dot(bump, n)));
        float wrap = 0.35;
        float key = max((dot(nb, KEY) + wrap) / (1.0 + wrap), 0.0) * mix(0.35, 1.0, sh);
        float amb = mix(0.20, 0.42, 0.5 + 0.5 * nb.y) * occ;
        float3 c = albedo * (1.05 * key + amb);
        // Wet mucosa: a sharp, weak highlight; stool is duller.
        float wet = mat == 5 ? 0.25 : 1.0;
        c += wet * 1.4 * ggx(nb, v, KEY, mat == 5 ? 0.5 : 0.18, 0.02) * sh;
        return c;
    }

    // Contents of the lumen, per mm of path: absorption, and the colour it scatters.
    void fluid(float x, int panel, thread float3 &sigma, thread float3 &glow) {
        if (panel == 1) {
            // Milky chyme in the ileum — the lactose is still in it — turning
            // to loose, frothy, yellow-brown stool as it ferments in the colon.
            float c = smoothstep(-12.0, 40.0, x);
            sigma = mix(float3(0.022, 0.022, 0.024), float3(0.040, 0.050, 0.075), c);
            glow = mix(float3(0.40, 0.38, 0.33), float3(0.36, 0.25, 0.10), c);
        } else {
            // Chyme: watery and yellow-brown at the valve, gone by mid-colon.
            float wet = 1.0 - smoothstep(-5.0, 70.0, x);
            sigma = wet * float3(0.050, 0.070, 0.12) + float3(0.002);
            glow = float3(0.30, 0.19, 0.06);
        }
    }

    float3 toneMap(float3 x) {
        float L = dot(x, float3(0.2126, 0.7152, 0.0722));
        float Lm = L * (1.0 + L / 9.0) / (1.0 + L);
        float3 c = x * (Lm / max(L, 1e-5)) * 1.08;
        c = clamp(c, 0.0, 1.0);
        return select(1.055 * pow(c, 1.0 / 2.4) - 0.055, 12.92 * c, c <= 0.0031308);
    }

    // ---------------------------------------------------------------- particles

    float3 particleColour(Particle P, float3 hit, float3 rd, float edge) {
        float3 n = normalize(hit - P.posR.xyz);
        float lam = 0.38 + 0.8 * max(dot(n, KEY), 0.0);
        float spec = pow(max(dot(reflect(-KEY, n), -rd), 0.0), 28.0) * 0.55;
        float3 c = P.colour.rgb * lam + spec;
        int species = int(P.info.y + 0.5);
        if (species == 3) return mix(c, float3(0.85, 0.95, 1.0), smoothstep(0.7, 0.98, edge) * 0.6);
        // A bubble: nearly clear in the middle, a bright rim and a highlight.
        if (species == 7) return mix(c * 0.6, float3(0.97, 0.99, 1.0), smoothstep(0.55, 0.95, edge)) + spec;
        // A dark rim, so a symbol reads against pink tissue as well as fluid.
        return mix(c, P.colour.rgb * 0.12, smoothstep(0.74, 0.9, edge) * 0.9);
    }

    // ---------------------------------------------------------------- one ray

    struct Sample { float3 colour; float mat; float x; float species; };

    Sample traceRay(float3 ro, float3 rd, int panel, float t,
                    constant float4 *lumps, uint nl,
                    constant Particle *parts, constant uint *tileStart, constant uint *tileItems, uint tile) {
        Sample S;
        float tStart, tEnd;
        if (panel == 2) {
            tStart = (ro.z - 1.5) / -rd.z;
            tEnd = (ro.z + CELL_D + 3.0) / -rd.z;
        } else {
            tStart = (ro.z - 0.5) / -rd.z;
            tEnd = (ro.z + RC0 + WC0 + 12.0) / -rd.z;
        }
        float th;
        int mat = 0;
        bool cut = false;
        float3 col;
        bool hit = march(ro, rd, panel, t, lumps, nl, tStart, tEnd, th, mat, cut);
        if (hit) {
            col = shade(ro + rd * th, rd, panel, t, mat, cut, lumps, nl);
        } else {
            th = 1e9;
            col = BG;
            mat = 0;
        }
        S.mat = hit ? float(mat + (cut ? 100 : 0)) : 0.0;
        S.x = hit ? (ro + rd * th).x : 0.0;
        S.species = 0.0;

        // The contents: a ray that crosses the cut plane inside the lumen
        // travels through them to whatever it hits.
        float tPlane = ro.z / -rd.z;
        float3 sigma = float3(0.0);
        float3 glow = float3(0.0);
        bool inFluid = false;
        if (panel < 2) {
            float3 pp = ro + rd * tPlane;
            if (lumenSDF(pp, panel, t) < 0.0) {
                inFluid = true;
                fluid(pp.x, panel, sigma, glow);
                float len = min(th, tEnd) - tPlane;
                if (len > 0.0) {
                    float3 tr = exp(-sigma * len);
                    col = col * tr + glow * (1.0 - tr);
                }
            }
        }

        // Particles in this tile, nearest few, composited back to front.
        float hitT[6];
        float3 hitC[6];
        float hitA[6];
        int hitS[6];
        int nh = 0;
        uint s0 = tileStart[tile];
        uint s1 = tileStart[tile + 1];
        for (uint i = s0; i < s1; i++) {
            Particle P = parts[tileItems[i]];
            if (int(P.info.x + 0.5) != panel) continue;
            float3 oc = ro - P.posR.xyz;
            float b = dot(oc, rd);
            float cc = dot(oc, oc) - P.posR.w * P.posR.w;
            float h = b * b - cc;
            if (h < 0.0) continue;
            float tp = -b - sqrt(h);
            if (tp > th) continue;
            float edge = sqrt(max(dot(oc, oc) - b * b, 0.0)) / P.posR.w;
            float3 pc = particleColour(P, ro + rd * tp, rd, edge);
            if (inFluid && tp > tPlane) {
                float3 tr = exp(-sigma * (tp - tPlane));
                pc = pc * tr + glow * (1.0 - tr);
            }
            // Insert, nearest first, keeping at most six.
            int j = min(nh, 5);
            if (nh == 6 && tp >= hitT[5]) continue;
            while (j > 0 && hitT[j - 1] > tp) {
                hitT[j] = hitT[j - 1]; hitC[j] = hitC[j - 1]; hitA[j] = hitA[j - 1]; hitS[j] = hitS[j - 1];
                j--;
            }
            float pa = P.colour.a;
            // A bubble's rim is opaque and its middle nearly clear.
            if (int(P.info.y + 0.5) == 7) pa *= mix(0.45, 2.6, smoothstep(0.55, 0.95, edge));
            hitT[j] = tp; hitC[j] = pc; hitA[j] = min(pa, 1.0); hitS[j] = int(P.info.y + 0.5);
            nh = min(nh + 1, 6);
        }
        for (int j = nh - 1; j >= 0; j--) col = mix(col, hitC[j], hitA[j]);
        if (nh > 0 && hitA[0] > 0.5) S.species = float(hitS[0] + 1);
        S.colour = col;
        return S;
    }

    // ---------------------------------------------------------------- kernels

    kernel void gut(device uchar4 *pixels [[buffer(0)]],
                    device float4 *aux [[buffer(1)]],
                    constant Params &P [[buffer(2)]],
                    constant Particle *parts [[buffer(3)]],
                    constant uint *tileStart [[buffer(4)]],
                    constant uint *tileItems [[buffer(5)]],
                    constant float4 *lumps [[buffer(6)]],
                    uint2 gid [[thread_position_in_grid]]) {
        uint x = gid.x;
        uint y = gid.y + P.rowOffset;
        if (x >= P.width || y >= P.height) return;
        int panel;
        float2 local;
        float fy = float(y);
        float fx = float(x);
        if (fy < PH) { panel = 0; local = float2(fx, fy); }
        else if (fy < 2.0 * PH) { panel = 1; local = float2(fx, fy - PH); }
        else if (fx >= IX && fx < IX + IW && fy >= IY && fy < IY + IHH) { panel = 2; local = float2(fx - IX, fy - IY); }
        else {
            pixels[y * P.width + x] = uchar4(12, 16, 20, 255);
            aux[y * P.width + x] = float4(-1.0, 0.0, 0.0, 0.0);
            return;
        }
        uint tile = (y / TILE) * P.tilesX + (x / TILE);
        float3 sum = float3(0.0);
        uint S = P.samples;
        Sample centre;
        uint nl = panel == 0 ? P.lumpCount : 0;
        for (uint sy = 0; sy < S; sy++) {
            for (uint sx = 0; sx < S; sx++) {
                float2 j = float2((float(sx) + 0.5) / float(S), (float(sy) + 0.5) / float(S));
                float2 px = local + j;
                float3 ro, rd;
                if (panel == 2) {
                    float ssx = (px.x - IW * 0.5) * ISCALE;
                    float ssy = (IHH * 0.5 - px.y) * ISCALE;
                    rd = IFWD;
                    ro = ICN + float3(ssx, 0, 0) + ssy * IUP - IFWD * 60.0;
                } else {
                    float ssx = (px.x - W * 0.5) * SCALE;
                    float ssy = (PH * 0.5 - px.y) * SCALE;
                    rd = PFWD;
                    ro = PC + float3(ssx, 0, 0) + ssy * PUP - PFWD * 200.0;
                }
                Sample s = traceRay(ro, rd, panel, P.time, lumps, nl, parts, tileStart, tileItems, tile);
                sum += toneMap(s.colour);
                if (sx == S / 2 && sy == S / 2) centre = s;
            }
        }
        float3 c = sum / float(S * S);
        pixels[y * P.width + x] = uchar4(uchar3(round(clamp(c, 0.0, 1.0) * 255.0)), 255);
        aux[y * P.width + x] = float4(float(panel), centre.mat, centre.x, centre.species);
    }

    // The scene's distance and material at arbitrary points (xyz, panel), at
    // one time — the same functions the picture was drawn with.
    kernel void probe(device const float4 *points [[buffer(0)]],
                      device float2 *out [[buffer(1)]],
                      constant float &time [[buffer(2)]],
                      constant float4 *lumps [[buffer(3)]],
                      constant uint &lumpCount [[buffer(4)]],
                      uint id [[thread_position_in_grid]]) {
        float4 q = points[id];
        int panel = int(q.w + 0.5);
        int mat;
        bool cut;
        float d;
        if (panel == 2) {
            // Cells only, no backdrop, no cut: is this point inside a cell?
            float k = clamp(round((q.x - CELL_X0) / CELL_P), 0.0, CELL_N - 1.0);
            float3 c = q.xyz - float3(CELL_X0 + k * CELL_P, 0.0, 0.0);
            float3 b = float3(0.5 * (CELL_P - CELL_G), 0.5 * CELL_H, CELL_D);
            float3 qq = abs(c) - b + 0.9;
            d = length(max(qq, 0.0)) + min(max(qq.x, max(qq.y, qq.z)), 0.0) - 0.9;
            mat = 6;
        } else {
            d = gutSDF(q.xyz, panel, time, lumps, panel == 0 ? lumpCount : 0, mat, cut);
        }
        out[id] = float2(d, float(mat));
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

func findDevice() throws -> MTLDevice {
    if let d = MTLCreateSystemDefaultDevice() ?? MTLCopyAllDevices().first { return d }
    throw RenderError.noMetalDevice
}

/// The finished picture, and what each pixel's centre saw.
struct Frame {
    let layout: Layout
    let pixels: MTLBuffer
    let aux: MTLBuffer

    func rgba(_ x: Int, _ y: Int) -> SIMD4<UInt8> {
        let p = pixels.contents().advanced(by: (y * layout.width + x) * 4).assumingMemoryBound(to: UInt8.self)
        return SIMD4<UInt8>(p[0], p[1], p[2], p[3])
    }

    /// (panel or −1, material (+100 on the cut face), world x, species + 1 of a front particle).
    func seen(_ x: Int, _ y: Int) -> SIMD4<Float> {
        aux.contents().assumingMemoryBound(to: SIMD4<Float>.self)[y * layout.width + x]
    }
}

/// Holds the compiled kernel and the buffers for one frame size.
final class Renderer {
    let device: MTLDevice
    let layout: Layout
    let queue: MTLCommandQueue
    let pso: MTLComputePipelineState
    let probePSO: MTLComputePipelineState
    let pixels: MTLBuffer
    let aux: MTLBuffer

    init(device: MTLDevice, layout: Layout) throws {
        self.device = device
        self.layout = layout
        let options = MTLCompileOptions()
        // Precise maths, as in every step: the tube's walls are differences of
        // radii to a few microns.
        options.fastMathEnabled = false
        let library: MTLLibrary
        do { library = try device.makeLibrary(source: kernelSource(layout: layout), options: options) } catch {
            throw RenderError.kernelCompile("\(error)")
        }
        guard let f = library.makeFunction(name: "gut"), let g = library.makeFunction(name: "probe"),
              let q = device.makeCommandQueue(),
              let px = device.makeBuffer(length: layout.width * layout.height * 4, options: .storageModeShared),
              let ax = device.makeBuffer(length: layout.width * layout.height * 16, options: .storageModeShared)
        else { throw RenderError.gpu("could not set up the renderer") }
        do {
            pso = try device.makeComputePipelineState(function: f)
            probePSO = try device.makeComputePipelineState(function: g)
        } catch { throw RenderError.kernelCompile("\(error)") }
        queue = q
        pixels = px
        aux = ax
    }

    /// Bins every particle into the screen tiles its disc touches.
    func tiles(_ ps: [Particle]) -> (gpu: [GPUParticle], start: [UInt32], items: [UInt32], tilesX: Int) {
        let tx: Int = (layout.width + tileSize - 1) / tileSize
        let ty: Int = (layout.height + tileSize - 1) / tileSize
        var lists: [[UInt32]] = Array(repeating: [], count: tx * ty)
        var gpu: [GPUParticle] = []
        gpu.reserveCapacity(ps.count)
        for p in ps {
            let c: SIMD4<Float> = speciesColour(p.species)
            let lin = SIMD3<Float>(srgbToLinear(c.x), srgbToLinear(c.y), srgbToLinear(c.z))
            let alpha: Float = c.w * p.alpha
            let idx = UInt32(gpu.count)
            gpu.append(GPUParticle(posR: SIMD4(p.position, p.radius), colour: SIMD4(lin, alpha),
                                   info: SIMD4(Float(p.panel), Float(p.species.rawValue), Float(p.route.rawValue), 0)))
            let s: SIMD2<Float> = project(p.position, panel: p.panel, layout: layout)
            let rp: Float = p.radius / (p.panel == 2 ? layout.insetScale : layout.scale) + 1.5
            let x0: Int = max(Int((s.x - rp) / Float(tileSize)), 0)
            let x1: Int = min(Int((s.x + rp) / Float(tileSize)), tx - 1)
            let y0: Int = max(Int((s.y - rp) / Float(tileSize)), 0)
            let y1: Int = min(Int((s.y + rp) / Float(tileSize)), ty - 1)
            if x1 < x0 || y1 < y0 { continue }
            for yy in y0...y1 { for xx in x0...x1 { lists[yy * tx + xx].append(idx) } }
        }
        var start: [UInt32] = [0]
        var items: [UInt32] = []
        for l in lists {
            items += l
            start.append(UInt32(items.count))
        }
        if items.isEmpty { items = [0] }
        if gpu.isEmpty { gpu = [GPUParticle(posR: .zero, colour: .zero, info: SIMD4(-9, 0, 0, 0))] }
        return (gpu, start, items, tx)
    }

    /// Renders the frame at time t into `pixels`; returns GPU seconds.
    @discardableResult
    func render(time t: Float, samples: Int, mutant: Mutant = .none) throws -> Double {
        let ps: [Particle] = particles(at: t, mutant: mutant)
        var (gpuParts, start, items, tilesX) = tiles(ps)
        var lumps: [SIMD4<Float>] = stoolLumps(at: t)
        guard let pb = device.makeBuffer(bytes: &gpuParts, length: MemoryLayout<GPUParticle>.stride * gpuParts.count,
                                         options: .storageModeShared),
              let sb = device.makeBuffer(bytes: &start, length: 4 * start.count, options: .storageModeShared),
              let ib = device.makeBuffer(bytes: &items, length: 4 * items.count, options: .storageModeShared),
              let lb = device.makeBuffer(bytes: &lumps, length: 16 * lumps.count, options: .storageModeShared)
        else { throw RenderError.gpu("could not allocate particle buffers") }
        let band: Int = 64
        var gpu: Double = 0
        var row: Int = 0
        while row < layout.height {
            let rows: Int = min(band, layout.height - row)
            guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else {
                throw RenderError.gpu("could not make a command buffer")
            }
            var params = Params(width: UInt32(layout.width), height: UInt32(layout.height),
                                rowOffset: UInt32(row), samples: UInt32(samples), time: t,
                                tilesX: UInt32(tilesX), lumpCount: UInt32(lumps.count), pad: 0)
            enc.setComputePipelineState(pso)
            enc.setBuffer(pixels, offset: 0, index: 0)
            enc.setBuffer(aux, offset: 0, index: 1)
            enc.setBytes(&params, length: MemoryLayout<Params>.stride, index: 2)
            enc.setBuffer(pb, offset: 0, index: 3)
            enc.setBuffer(sb, offset: 0, index: 4)
            enc.setBuffer(ib, offset: 0, index: 5)
            enc.setBuffer(lb, offset: 0, index: 6)
            let w: Int = pso.threadExecutionWidth
            let group = MTLSize(width: w, height: max(pso.maxTotalThreadsPerThreadgroup / w / 4, 1), depth: 1)
            enc.dispatchThreads(MTLSize(width: layout.width, height: rows, depth: 1), threadsPerThreadgroup: group)
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            if let e = cb.error { throw RenderError.gpu(e.localizedDescription) }
            gpu += cb.gpuEndTime - cb.gpuStartTime
            row += rows
        }
        return gpu
    }

    var frame: Frame { Frame(layout: layout, pixels: pixels, aux: aux) }

    /// Distance and material at arbitrary points (xyz, panel) at time t, from
    /// the kernel source that drew the picture.
    func probe(_ points: [SIMD4<Float>], time: Float) throws -> [SIMD2<Float>] {
        var pts: [SIMD4<Float>] = points
        var lumps: [SIMD4<Float>] = stoolLumps(at: time)
        var t: Float = time
        var n = UInt32(lumps.count)
        guard let pb = device.makeBuffer(bytes: &pts, length: 16 * max(pts.count, 1), options: .storageModeShared),
              let ob = device.makeBuffer(length: 8 * max(pts.count, 1), options: .storageModeShared),
              let lb = device.makeBuffer(bytes: &lumps, length: 16 * lumps.count, options: .storageModeShared),
              let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder()
        else { throw RenderError.gpu("could not set up the probe") }
        enc.setComputePipelineState(probePSO)
        enc.setBuffer(pb, offset: 0, index: 0)
        enc.setBuffer(ob, offset: 0, index: 1)
        enc.setBytes(&t, length: 4, index: 2)
        enc.setBuffer(lb, offset: 0, index: 3)
        enc.setBytes(&n, length: 4, index: 4)
        enc.dispatchThreads(MTLSize(width: pts.count, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(probePSO.maxTotalThreadsPerThreadgroup, 256), height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        if let e = cb.error { throw RenderError.gpu(e.localizedDescription) }
        let out = ob.contents().assumingMemoryBound(to: SIMD2<Float>.self)
        return (0..<pts.count).map { out[$0] }
    }
}

func savePNG(_ frame: Frame, to url: URL) throws {
    try? FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    let w: Int = frame.layout.width
    let h: Int = frame.layout.height
    guard let provider = CGDataProvider(dataInfo: nil, data: frame.pixels.contents(), size: w * h * 4,
                                        releaseData: { _, _, _ in }),
          let space = CGColorSpace(name: CGColorSpace.sRGB),
          let cg = CGImage(width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 32,
                           bytesPerRow: w * 4, space: space,
                           bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                           provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent),
          let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)
    else { throw RenderError.png("could not set up the encoder for \(url.path)") }
    CGImageDestinationAddImage(dest, cg, nil)
    guard CGImageDestinationFinalize(dest) else { throw RenderError.png("could not write \(url.path)") }
}
