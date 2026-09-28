// The Metal source: one thread per pixel marching rays into a scene made of
// distance functions, as in step 20. The constants come first, written out by
// Swift from Flower.swift, so the numbers the tests read are the numbers the
// GPU draws; the body below them has no interpolation in it at all.
//
// What is new against step 20:
//
//   * thin tubes as capsule chains. The keel, the style, ten filaments and the
//     pollen tube all follow curves. A helix has no exact distance function,
//     and dividing an approximate one by its steepest slope is where step 20's
//     discontinuities came from. A polyline of capsules IS exact — distance to
//     a segment is a closed form — so the Lipschitz bound is 1 by
//     construction, and the only cost is looping over segments. Bounding
//     spheres over runs of 16 segments skip most of them, and the skip is
//     exact: a run is skipped only when even its sphere is further than the
//     best surface found so far, so it could not have won the min.
//   * the cutaway is a plane, z > 0 removed, applied by max() to the petals,
//     calyx, ovary and sheath — and not to the style, stamens, ovules and
//     pollen, which are drawn whole, as a dissection would leave them.
//   * translucency of thin petals (light through from behind), and of the
//     style, through which the pollen tube shows.
//
// Step 29 copies this kernel and changes three things: the pollen tube's
// vertices carry arc length in w (its radius is the constant TUBE_R), the
// View carries how much of it has grown, and the tube — and its glow through
// the style — is drawn only that far, cut exactly; and a column offset lets a
// frame re-render just a rectangle. With the tube fully grown the picture is
// step 25's, bit for bit. And (tubeP.z) a second cut, used only by the inset
// that follows the tip: the plane z = cutZ opens the style, filaments and
// anthers along the tube, so the tube can be seen inside the style at its
// true width. It is a max() with a plane, so still a distance bound; at
// cutZ = 1e9 it changes nothing.

import Foundation
import simd

func metal(_ v: SIMD3<Float>) -> String { "float3(\(v.x), \(v.y), \(v.z))" }
func metal(_ v: Float) -> String { "\(v)" }

/// The camera's light, shared by the kernel and the tests.
let keyDirection: SIMD3<Float> = simd_normalize(SIMD3<Float>(-0.3, 0.6, 0.85))

/// How far a ray trusts a distance. The capsule chains are exact; what
/// over-reports is the ellipsoids' outside approximation and the tapers. The
/// tests measure the real factor.
let stepScale: Float = 0.85

/// Everything the kernel needs to know that is a number, as Metal `constant`s.
func kernelConstants(_ m: FlowerModel, _ buffers: SceneLayout) -> String {
    var c: [String] = []
    func add(_ name: String, _ value: String) { c.append("constant \(name) = \(value);") }
    add("uint SPINE_N", "\(spine.count)")
    add("uint SPINE_CHUNKS", "\(buffers.spineChunkCount)")
    add("uint CHUNK_N", "\(buffers.chunkCount)")
    add("uint TUBE_C0", "\(buffers.tubeChunkStart)")
    add("uint ANTHER_N", "\(m.anthers.count)")
    add("uint GRAIN_N", "\(m.grains.count)")
    add("uint OVULE_N", "\(m.ovules.count)")
    add("int HAIR_ROWS", "\(hairRows)")
    add("int HAIR_COLS", "\(hairColumns)")
    add("float STEP_SCALE", metal(stepScale))
    add("float3 TIP_C", metal(buffers.tipCentre))
    add("float TIP_R", metal(buffers.tipRadius))
    add("float BRUSH_S0", metal(brushStartS))
    add("float HAIR_ROW", metal(hairRowSpacing))
    add("float HAIR_STEP", metal(hairColumnStepDegrees))
    add("float3 EF_C", metal(envelopeFrontCentre))
    add("float3 EF_R", metal(envelopeFrontRadii))
    add("float3 EB_C", metal(envelopeBackCentre))
    add("float3 EB_R", metal(envelopeBackRadii))
    add("float SPLIT_Y", metal(petalSplitY))
    add("float BANNER_GAP", metal(bannerBottomGapHalfDegrees * Float.pi / 180))
    let wingMid: Float = (wingFromDegrees + wingToDegrees) / 2 * Float.pi / 180
    let wingHalf: Float = (wingToDegrees - wingFromDegrees) / 2 * Float.pi / 180
    add("float WING_MID", metal(wingMid))
    add("float WING_HALF", metal(wingHalf))
    add("float BANNER_OFFSET", metal(bannerOffset))
    add("float PETAL_HALF", metal(petalThickness / 2))
    add("float BANNER_OPEN", metal(bannerOpenAngle(m.mutant)))
    add("float WING_OPEN", metal(wingOpenAngle(m.mutant)))
    add("float KEEL_START_X", metal(keelStartX))
    add("int SHEATH", hasSheath(m.stamens) ? "1" : "0")
    add("float SHEATH_R", metal(sheathRadius))
    add("float SHEATH_X0", metal(sheathStartX))
    add("float SHEATH_X1", metal(sheathEndX))
    add("float SHEATH_SLIT", metal(sheathSlitHalfAngle * Float.pi / 180))
    add("float OV_X0", metal(ovaryStartX))
    add("float OV_X1", metal(ovaryEndX))
    add("float OV_A", metal(ovaryHalfHeight))
    add("float OV_B", metal(ovaryHalfWidth))
    add("float OV_WALL", metal(ovaryWall))
    add("float OV_TAPER", metal(ovaryTaperStartX))
    add("float3 OVULE_R", metal(ovuleSemiAxes))
    add("float3 ANTHER_R", metal(antherSemiAxes))
    add("float3 STIG_C", metal(stigmaCentre))
    add("float STIG_R", metal(stigmaRadius))
    add("float TUBE_R", metal(pollenTubeRadius))
    add("float3 KEY_DIR", metal(keyDirection))
    let pod: PodFrame = podFrame()
    add("float3 POD_O", metal(pod.origin))
    add("float3 POD_X", metal(pod.x))
    add("float3 POD_Y", metal(pod.y))
    add("float3 POD_Z", metal(pod.z))
    add("float POD_L", metal(podLength))
    add("float POD_A", metal(podWidth / 2))
    add("int POD_SEEDS", "\(m.ovules.count)")
    return c.joined(separator: "\n")
}

let kernelBody: String = """
struct SpineVert { float4 p; float4 n; float4 b; float4 r; };
struct Chunk { float4 sphere; uint first; uint count; uint mat; uint pad; };
struct AntherG { float4 c; float4 ax; float4 sd; };
struct GrainG { float4 c; float4 pole; float4 pore; };
struct HairG { float4 base; float4 tip; };
struct View { float4 pos; float4 fwd; float4 right; float4 up;
              float tanHalf; uint width; uint height; uint rowOffset;
              uint samples; uint layer; float tubeMinV; float tubeGrownV;
              uint colOffset; float cutZ; uint pad1; uint pad2; };

#define SCENE_ARGS device const SpineVert *spine, device const float4 *spineChunks, \\
    device const float4 *cverts, device const Chunk *chunks, device const AntherG *anthers, \\
    device const GrainG *grains, device const float4 *ovules, device const HairG *hairs
#define SCENE spine, spineChunks, cverts, chunks, anthers, grains, ovules, hairs

constant float PETAL_TRANSMIT = 0.5;

// Materials.
constant int M_BANNER = 1, M_WING = 2, M_KEEL = 3, M_GREEN = 4, M_OVARY = 5, M_OVULE = 6,
             M_STAMEN = 7, M_ANTHER = 8, M_STYLE = 9, M_HAIR = 10, M_POLLEN = 11, M_TUBE = 12,
             M_POD = 13;

// Probe modes. 0 is what the picture marches.
constant int P_RENDER = 0, P_UNCUT = 1, P_PISTIL = 2, P_OVULES = 3, P_GRAINS = 4,
             P_ENCLOSURE = 5, P_STAMENS = 6, P_KEELCAV = 7, P_TUBE = 8, P_BANNER = 9;

// ------------------------------------------------------------------ helpers

float smin(float a, float b, float k) {
    float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    return mix(b, a, h) - k * h * (1.0 - h);
}

// An ellipsoid. Outside: the usual k0(k0−1)/k1 estimate. Inside, where the
// bud's air is, that estimate has a singularity at the centre — and the
// centre of the front ellipsoid is where the stigma sits. So inside it is
// (k0 − 1)·r_min instead: k0's gradient is at most 1/r_min, so this never
// claims more room than there is, and it meets the outside form at zero.
float ellipsoidD(float3 q, float3 r) {
    float k0 = length(q / r);
    if (k0 < 1.0) return (k0 - 1.0) * min(r.x, min(r.y, r.z));
    float k1 = length(q / (r * r));
    return k0 * (k0 - 1.0) / k1;
}

float capsule(float3 p, float3 a, float3 b, float ra, float rb) {
    float3 ab = b - a;
    float h = clamp(dot(p - a, ab) / max(dot(ab, ab), 1e-12), 0.0, 1.0);
    return length(p - a - ab * h) - mix(ra, rb, h);
}

// A 2D wedge of half-angle w round unit direction u: negative inside.
float wedge(float2 v, float2 u, float w) {
    float2 perp = float2(-u.y, u.x);
    float2 q = float2(abs(dot(v, perp)), dot(v, u));
    return q.x * cos(w) - q.y * sin(w);
}

float3 rotX(float3 p, float a) { float c = cos(a), s = sin(a); return float3(p.x, c * p.y - s * p.z, s * p.y + c * p.z); }
float3 rotZ(float3 p, float a) { float c = cos(a), s = sin(a); return float3(c * p.x - s * p.y, s * p.x + c * p.y, p.z); }

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
float3 hash33(float3 p) {
    return float3(hash3(p), hash3(p + float3(17.1, 3.3, 9.7)), hash3(p + float3(5.2, 29.4, 13.1)));
}
// Cellular noise: distance to the nearest and second-nearest feature point.
float2 voronoi(float3 x) {
    float3 i = floor(x);
    float3 f = fract(x);
    float f1 = 8.0, f2 = 8.0;
    for (int k = -1; k <= 1; k++) for (int j = -1; j <= 1; j++) for (int m = -1; m <= 1; m++) {
        float3 g = float3(m, j, k);
        float3 o = hash33(i + g);
        float d = length(g + o - f);
        if (d < f1) { f2 = f1; f1 = d; } else if (d < f2) { f2 = d; }
    }
    return float2(f1, f2);
}

// ------------------------------------------------------------------ the petals

float envelope(float3 p) {
    return smin(ellipsoidD(p - EF_C, EF_R), ellipsoidD(p - EB_C, EB_R), 1.0);
}

// The banner: outer shell over the top and sides, everything but a 110°
// wedge underneath. For the open mutant it is swung back on its claw.
float banner(float3 p) {
    float3 hinge = float3(2.2, SPLIT_Y, 0.0);
    float3 q = rotZ(p - hinge, BANNER_OPEN) + hinge;
    float shell = abs(envelope(q) - BANNER_OFFSET) - PETAL_HALF;
    float bottom = wedge(float2(q.y - SPLIT_Y, q.z), float2(-1.0, 0.0), BANNER_GAP);
    return max(shell, -bottom);
}

// One wing: 40°–185° round one side, so it tucks under the banner at the
// top and crosses the other wing underneath.
float wing(float3 p, float side) {
    float3 axis = float3(0.0, SPLIT_Y, 0.0);
    float3 q = rotX(p - axis, side * WING_OPEN) + axis;
    float shell = abs(envelope(q)) - PETAL_HALF;
    float2 u = float2(cos(WING_MID), side * sin(WING_MID));
    return max(shell, wedge(float2(q.y - SPLIT_Y, q.z), u, WING_HALF));
}

// The calyx: a green cone with five teeth. Its slope and the teeth make the
// raw formula over-report by up to ~1.25; the divisions put it back.
float calyx(float3 p) {
    float r = 0.5 + (p.x + 0.6) * 0.45;
    float dr = (length(p.yz) - r) * 0.912;
    float sh = abs(dr) - 0.06;
    float th = atan2(p.z, p.y);
    float tri = abs(fract(th * 5.0 / (2.0 * M_PI_F) + 0.5) - 0.5) * 2.0;
    float xEnd = 2.2 + 0.85 * (1.0 - tri);
    return max(sh, max(-0.6 - p.x, (p.x - xEnd) * 0.8));
}

float bracteole(float3 p, float side) {
    float3 c = float3(0.3, -0.75, side * 1.5);
    float3 w = normalize(float3(0.0, 0.35, side));
    float3 u = normalize(float3(1.0, -0.3, 0.0) - w * dot(float3(1.0, -0.3, 0.0), w));
    float3 v = cross(w, u);
    float3 q = p - c;
    float a = dot(q, u), b = dot(q, v), cc = dot(q, w) + 0.12 * (a / 2.75) * (a / 2.75);
    float e = length(float2(a / 2.75, b / 1.5));
    return max((e - 1.0) * 1.5, abs(cc) - 0.05) / 1.05;
}

float greenParts(float3 p, bool cut, thread bool &isCut) {
    float d = calyx(p);
    float b1 = bracteole(p, 1.0);
    float b2 = bracteole(p, -1.0);
    float cz = p.z;
    float cutD = min(d, b1);
    isCut = false;
    if (cut && cz > cutD) { cutD = cz; isCut = true; }
    float stalk = capsule(p, float3(-5.5, -1.5, 0.0), float3(-0.4, 0.0, 0.0), 0.3, 0.34);
    float recept = length(p - float3(-0.35, 0.0, 0.0)) - 0.42;
    float whole = min(smin(stalk, recept, 0.25), b2);
    if (whole < cutD) isCut = false;
    return min(cutD, whole);
}

// ------------------------------------------------------------------ the pistil

// Ovary cross-section: an ellipse whose size tapers at both ends. Written as
// b·(E₀ − t) with E₀ the untapered ellipse coordinate, so the taper enters
// linearly; its steepest slope makes the whole over-report by ≤ 1.3, divided.
float ovaryTaper(float x) {
    float tipT = mix(1.0, 0.14, smoothstep(OV_TAPER, OV_X1, x));
    float baseT = mix(0.5, 1.0, smoothstep(OV_X0, OV_X0 + 0.4, x));
    return tipT * baseT;
}
float ovaryOuter(float3 p) {
    float t = ovaryTaper(p.x);
    float e0 = length(float2(p.y / OV_A, p.z / OV_B));
    float d = OV_B * (e0 - t);
    return max(d, max(OV_X0 - p.x, p.x - OV_X1)) / 1.3;
}
float locule(float3 p) {
    float a = OV_A - OV_WALL, b = OV_B - OV_WALL;
    float t = ovaryTaper(p.x);
    float e0 = length(float2(p.y / a, p.z / b));
    float d = b * (e0 - t);
    return max(d, max(OV_X0 + 0.3 - p.x, p.x - (OV_X1 - 0.25))) / 1.3;
}

float sheath(float3 p) {
    float r = length(p.yz);
    float shell = abs(r - SHEATH_R) - 0.035;
    float d = max(shell, max(SHEATH_X0 - p.x, p.x - SHEATH_X1));
    float slit = wedge(float2(p.y, p.z), float2(1.0, 0.0), SHEATH_SLIT);
    return max(d, -slit);
}

// ------------------------------------------------------------------ the spine

struct SpineInfo {
    float keelCav;   // distance to the keel's cavity wall; negative inside
    float style;     // distance to the style
    float s;         // arc length of the style's nearest point
    float3 c;        // that point
    float3 T, N, B;  // its frame
    float styleR;
};

SpineInfo noSpine() {
    SpineInfo si;
    si.keelCav = 1e9; si.style = 1e9; si.s = 0.0; si.c = float3(0.0);
    si.T = float3(1, 0, 0); si.N = float3(0, 1, 0); si.B = float3(0, 0, 1); si.styleR = 0.1;
    return si;
}

SpineInfo spineEval(float3 p, SCENE_ARGS) {
    SpineInfo si = noSpine();
    for (uint k = 0; k < SPINE_CHUNKS; k++) {
        float4 sph = spineChunks[k];
        float lb = length(p - sph.xyz) - sph.w;
        if (lb > si.keelCav && lb > si.style) continue;
        uint i0 = k * 16;
        uint i1 = min(i0 + 16, SPINE_N - 1);
        for (uint i = i0; i < i1; i++) {
            float3 a = spine[i].p.xyz;
            float3 b = spine[i + 1].p.xyz;
            float3 ab = b - a;
            float h = clamp(dot(p - a, ab) / dot(ab, ab), 0.0, 1.0);
            float3 c = a + ab * h;
            float dist = length(p - c);
            float rk = mix(spine[i].r.x, spine[i + 1].r.x, h);
            si.keelCav = min(si.keelCav, dist - rk);
            float r0 = spine[i].r.y, r1 = spine[i + 1].r.y;
            if (r0 > 0.0 && r1 > 0.0) {
                float rs = mix(r0, r1, h);
                float ds = dist - rs;
                if (ds < si.style) {
                    si.style = ds;
                    si.s = mix(spine[i].p.w, spine[i + 1].p.w, h);
                    si.c = c;
                    si.T = normalize(ab);
                    float3 n = normalize(mix(spine[i].n.xyz, spine[i + 1].n.xyz, h));
                    si.N = normalize(n - si.T * dot(n, si.T));
                    si.B = cross(si.T, si.N);
                    si.styleR = rs;
                }
            }
        }
    }
    return si;
}

// The brush: explicit hairs, looked up by the nearest style point's row and
// angle, so only a handful are tested. Rows below lean up into this one, so
// three rows back are included.
float hairField(float3 p, SpineInfo si, device const HairG *hairs) {
    float3 o = p - si.c;
    float th = atan2(dot(o, si.B), dot(o, si.N)) * 180.0 / M_PI_F;
    int k0 = int(round((si.s - BRUSH_S0) / HAIR_ROW));
    float best = 1e9;
    for (int dk = -4; dk <= 1; dk++) {
        int k = k0 + dk;
        if (k < 0 || k >= HAIR_ROWS) continue;
        float stagger = (k & 1) == 1 ? 0.5 : 0.0;
        int jc = int(round(th / HAIR_STEP + float(HAIR_COLS / 2) - stagger + 0.25));
        for (int dj = -1; dj <= 1; dj++) {
            int j = jc + dj;
            if (j < 0 || j >= HAIR_COLS) continue;
            HairG h = hairs[k * HAIR_COLS + j];
            best = min(best, capsule(p, h.base.xyz, h.tip.xyz, h.base.w, h.tip.w));
        }
    }
    return best;
}

float antherD(float3 p, AntherG a) {
    float3 d = p - a.c.xyz;
    float3 w = cross(a.ax.xyz, a.sd.xyz);
    return ellipsoidD(float3(dot(d, a.ax.xyz), dot(d, a.sd.xyz), dot(d, w)), ANTHER_R);
}

float grainD(float3 p, GrainG g) {
    float3 q = p - g.c.xyz;
    float qp = dot(q, g.pole.xyz);
    float qe = length(q - g.pole.xyz * qp);
    return ellipsoidD(float3(qe, 0.0, qp), float3(g.c.w, g.c.w, g.pole.w));
}

// ------------------------------------------------------------------ the scene

void consider(thread float &best, thread int &mat, thread int &cap, float d, int m, bool cutIt, float3 p) {
    if (cutIt && p.z > d) {
        if (p.z < best) { best = p.z; mat = m; cap = 1; }
        return;
    }
    if (d < best) { best = d; mat = m; cap = 0; }
}

float podSDF(float3 p, thread int &mat);

float sceneSDF(float3 p, int mode, float3 tubeP, thread int &mat, thread int &cap, SCENE_ARGS) {
    float best = 1e9;
    mat = 0; cap = 0;
    bool all = mode <= P_UNCUT;
    bool cut = mode == P_RENDER;

    if (all || mode == P_ENCLOSURE || mode == P_BANNER) consider(best, mat, cap, banner(p), M_BANNER, cut, p);
    if (mode == P_BANNER) return best;
    if (all || mode == P_ENCLOSURE) {
        consider(best, mat, cap, wing(p, 1.0), M_WING, cut, p);
        consider(best, mat, cap, wing(p, -1.0), M_WING, cut, p);
    }
    if (mode == P_ENCLOSURE) { consider(best, mat, cap, calyx(p), M_GREEN, false, p); return best; }
    if (all) {
        bool isCut;
        float g = greenParts(p, cut, isCut);
        if (g < best) { best = g; mat = M_GREEN; cap = isCut ? 1 : 0; }
    }

    bool needSpine = all || mode == P_PISTIL || mode == P_KEELCAV;
    SpineInfo si = noSpine();
    if (needSpine) si = spineEval(p, SCENE);
    if (mode == P_KEELCAV) return si.keelCav;
    if (all) {
        float keel = max(abs(si.keelCav) - PETAL_HALF, KEEL_START_X - p.x);
        consider(best, mat, cap, keel, M_KEEL, cut, p);
    }
    if (all || mode == P_PISTIL) {
        float st = min(si.style, length(p - STIG_C) - STIG_R);
        consider(best, mat, cap, max(st, p.z - tubeP.z), M_STYLE, false, p);
        float ov = ovaryOuter(p);
        if (all) ov = max(ov, -locule(p));
        consider(best, mat, cap, ov, M_OVARY, cut, p);
    }
    if (mode == P_PISTIL) return best;
    if (all || mode == P_OVULES) {
        for (uint i = 0; i < OVULE_N; i++)
            consider(best, mat, cap, ellipsoidD(p - ovules[i].xyz, OVULE_R), M_OVULE, false, p);
    }
    if (mode == P_OVULES) return best;
    if ((all || mode == P_STAMENS) && SHEATH == 1) consider(best, mat, cap, max(sheath(p), p.z - tubeP.z), M_STAMEN, cut, p);

    // The chains: filaments, funiculi, the pollen tube.
    if (all || mode == P_STAMENS || mode == P_TUBE) {
        for (uint k = 0; k < CHUNK_N; k++) {
            Chunk ch = chunks[k];
            int m = int(ch.mat);
            if (mode == P_STAMENS && m != M_STAMEN) continue;
            if (mode == P_TUBE && m != M_TUBE) continue;
            if (length(p - ch.sphere.xyz) - ch.sphere.w > best) continue;
            if (m == M_TUBE) {
                // Step 29: the tube's w is arc length, and only the part grown
                // so far is drawn. The last segment is cut exactly where the
                // growth has reached, and its round end is the growing tip —
                // still a union of capsules, so still an exact distance.
                float r = max(TUBE_R, tubeP.x);
                for (uint i = ch.first; i < ch.first + ch.count; i++) {
                    float4 a = cverts[i], b = cverts[i + 1];
                    if (a.w >= tubeP.y) break;
                    float3 bp = b.w > tubeP.y ? a.xyz + (b.xyz - a.xyz) * ((tubeP.y - a.w) / (b.w - a.w)) : b.xyz;
                    consider(best, mat, cap, capsule(p, a.xyz, bp, r, r), m, false, p);
                }
                continue;
            }
            for (uint i = ch.first; i < ch.first + ch.count; i++) {
                float4 a = cverts[i], b = cverts[i + 1];
                float d = capsule(p, a.xyz, b.xyz, a.w, b.w);
                if (m == M_STAMEN) d = max(d, p.z - tubeP.z);
                consider(best, mat, cap, d, m, false, p);
            }
        }
    }
    if (mode == P_STAMENS || mode == P_TUBE) return best;

    // The cluster at the tip: hairs, anthers, pollen. Skipped only when its
    // bounding sphere is further than what has already been found.
    if (length(p - TIP_C) - TIP_R < best) {
        if (all) {
            consider(best, mat, cap, hairField(p, si, hairs), M_HAIR, false, p);
            for (uint i = 0; i < ANTHER_N; i++) consider(best, mat, cap, max(antherD(p, anthers[i]), p.z - tubeP.z), M_ANTHER, false, p);
        }
        for (uint i = 0; i < GRAIN_N; i++) {
            GrainG g = grains[i];
            if (length(p - g.c.xyz) - g.c.w > best) continue;
            consider(best, mat, cap, grainD(p, g), M_POLLEN, false, p);
        }
    }
    return best;
}

// ------------------------------------------------------------------ the pod

// The same ovary two weeks on, far behind the flower and blurred afterwards:
// a long, flattened, slightly upcurved tube with a beak, swelling a little
// over each of its six young seeds.
float podSDF(float3 p, thread int &mat) {
    mat = M_POD;
    float3 q = p - POD_O;
    float u = dot(q, POD_X);
    float v = dot(q, POD_Y);
    float w = dot(q, POD_Z);
    float un = clamp(u / POD_L, 0.0, 1.0);
    float vc = 0.07 * POD_L * un * un;
    float taper = smoothstep(-2.0, 12.0, u) * (1.0 - 0.9 * smoothstep(POD_L - 20.0, POD_L, u));
    float bulge = 0.0;
    for (int i = 0; i < POD_SEEDS; i++) {
        float ui = POD_L * (0.14 + 0.145 * float(i));
        float x = (u - ui) / 5.5;
        bulge += 0.9 * exp(-x * x);
    }
    float a = POD_A * taper + 0.15;
    float b = (3.6 + bulge) * taper + 0.15;
    float e = length(float2((v - vc) / a, w / b));
    float d = (e - 1.0) * min(a, b);
    d = max(d, max(-u, u - POD_L)) / 1.5;
    float3 tip = POD_O + POD_X * POD_L + POD_Y * (0.07 * POD_L);
    float beak = capsule(p, tip - POD_X * 2.0, tip + POD_X * 7.0 + POD_Y * 4.0, 0.7, 0.2);
    float stalk = capsule(p, POD_O - POD_X * 14.0 - POD_Y * 3.0, POD_O + POD_X * 1.0, 0.9, 1.1);
    float cup = max(length(q + POD_X * 1.0) - 4.2, -(length(q + POD_X * 1.0) - 3.8));
    cup = max(cup, u - 2.5);
    cup = max(cup, -u - 3.5);
    return min(min(d, beak), min(stalk, cup));
}

// ------------------------------------------------------------------ marching

float sdf(float3 p, uint layer, float3 tubeP, thread int &mat, thread int &cap, SCENE_ARGS) {
    if (layer == 1) { cap = 0; return podSDF(p, mat); }
    return sceneSDF(p, P_RENDER, tubeP, mat, cap, SCENE);
}

float3 sceneNormal(float3 p, uint layer, float3 tubeP, float e, SCENE_ARGS) {
    int m, c;
    float3 k1 = float3(1, -1, -1), k2 = float3(-1, -1, 1), k3 = float3(-1, 1, -1), k4 = float3(1, 1, 1);
    return normalize(k1 * sdf(p + k1 * e, layer, tubeP, m, c, SCENE) + k2 * sdf(p + k2 * e, layer, tubeP, m, c, SCENE)
                   + k3 * sdf(p + k3 * e, layer, tubeP, m, c, SCENE) + k4 * sdf(p + k4 * e, layer, tubeP, m, c, SCENE));
}

bool march(float3 ro, float3 rd, uint layer, float3 tubeP, float pixAngle, thread float &t,
           thread int &mat, thread int &cap, SCENE_ARGS) {
    t = layer == 1 ? 150.0 : 20.0;
    float far = layer == 1 ? 800.0 : 110.0;
    for (int i = 0; i < 900; i++) {
        float d = sdf(ro + rd * t, layer, tubeP, mat, cap, SCENE);
        if (d < 0.3 * pixAngle * t) return true;
        t += d * STEP_SCALE;
        if (t > far) break;
    }
    return false;
}

// Soft shadows — except that a petal is not opaque. A shadow ray that meets
// a petal loses about half its light crossing it and carries on (MODEL: a
// white petal a few cells thick transmits of the order of half the light), so
// the inside of the closed bud is lit through its own walls, as a real bud
// is. Only opaque tissue makes a penumbra.
float softShadow(float3 ro, float3 rd, uint layer, float3 tubeP, float t0, SCENE_ARGS) {
    float res = 1.0;
    float through = 1.0;
    float t = t0;
    int m, c;
    float maxT = layer == 1 ? 200.0 : 25.0;
    for (int i = 0; i < 96; i++) {
        float h = sdf(ro + rd * t, layer, tubeP, m, c, SCENE);
        bool petal = m == M_BANNER || m == M_WING || m == M_KEEL;
        if (petal) {
            if (h < t0 * 0.5) {
                through *= PETAL_TRANSMIT;
                t += PETAL_HALF * 4.0;
                if (through < 0.02) break;
                continue;
            }
        } else {
            res = min(res, 8.0 * h / t);
        }
        t += clamp(h * STEP_SCALE, t0 * 0.5, maxT * 0.1);
        if (res < 0.002 || t > maxT) break;
    }
    return clamp(res, 0.0, 1.0) * through;
}

float ambientOcclusion(float3 p, float3 n, uint layer, float3 tubeP, float scale, SCENE_ARGS) {
    float occ = 0.0;
    float weight = 1.0;
    int m, c;
    for (int i = 1; i <= 5; i++) {
        float h = scale * (0.5 * float(i * i) + 0.3);
        float d = sdf(p + n * h, layer, tubeP, m, c, SCENE);
        occ += weight * clamp((h - d) / h, 0.0, 1.0);
        weight *= 0.72;
    }
    return clamp(1.0 - 0.42 * occ, 0.0, 1.0);
}

float tubeAxisDistance(float3 p, float grown, device const float4 *cverts, device const Chunk *chunks) {
    float best = 1e9;
    for (uint k = TUBE_C0; k < CHUNK_N; k++) {
        Chunk ch = chunks[k];
        if (length(p - ch.sphere.xyz) - ch.sphere.w > best) continue;
        for (uint i = ch.first; i < ch.first + ch.count; i++) {
            float4 a = cverts[i], b = cverts[i + 1];
            if (a.w >= grown) break;
            float3 bp = b.w > grown ? a.xyz + (b.xyz - a.xyz) * ((grown - a.w) / (b.w - a.w)) : b.xyz;
            best = min(best, capsule(p, a.xyz, bp, 0.0, 0.0));
        }
    }
    return best;
}

// ------------------------------------------------------------------ light

constant float3 KEY_COL = float3(1.0, 0.98, 0.94) * 2.7;
constant float3 FILL_COL = float3(0.92, 0.95, 1.0) * 0.9;

float3 skyIrradiance(float3 n) {
    return mix(float3(0.45, 0.46, 0.44), float3(0.9, 0.93, 0.96), 0.5 + 0.5 * n.y) * 1.0;
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

int nearestGrain(float3 p, device const GrainG *grains) {
    int best = 0;
    float bd = 1e9;
    for (uint i = 0; i < GRAIN_N; i++) {
        float d = grainD(p, grains[i]);
        if (d < bd) { bd = d; best = int(i); }
    }
    return best;
}

// The exine: a reticulum of ridges (muri) round pits (lumina), and three
// pores on the equator. Drawn as a bump on the shading normal.
float exineHeight(float3 dir) {
    float2 f = voronoi(dir * 7.5);
    return 1.0 - smoothstep(0.0, 0.16, f.y - f.x);
}

float3 shade(float3 p, float3 rd, int mat, int cap, uint layer, float3 tubeP, float pixSize, SCENE_ARGS) {
    float e = max(pixSize * 0.5, 0.0002);
    float3 n = sceneNormal(p, layer, tubeP, e, SCENE);
    float3 v = -rd;
    float occ = ambientOcclusion(p, n, layer, tubeP, max(pixSize * 1.6, 0.0015), SCENE);

    float3 albedo = float3(0.8);
    float trans = 0.0;
    float wrap = 0.2;
    float alpha = 0.45;
    float f0 = 0.03;
    float3 emit = float3(0.0);
    float sheen = 0.0;

    if (mat == M_BANNER || mat == M_WING || mat == M_KEEL) {
        albedo = float3(0.80, 0.81, 0.77);
        albedo = mix(albedo, float3(0.62, 0.72, 0.50), smoothstep(4.0, 1.6, p.x) * 0.6);
        float vein = noise3(float3(p.x * 1.2, p.y * 9.0, p.z * 9.0));
        albedo *= 1.0 - 0.05 * vein;
        trans = 0.5;
        wrap = 0.35;
        // The keel a shade creamier than banner and wings, greener at its beak.
        if (mat == M_KEEL) albedo *= float3(0.97, 0.96, 0.90);
        if (cap == 1) { albedo = float3(0.70, 0.76, 0.60); trans = 0.0; }
    } else if (mat == M_GREEN) {
        albedo = float3(0.10, 0.23, 0.06) * (0.85 + 0.3 * noise3(p * 6.0));
        if (cap == 1) albedo = float3(0.30, 0.45, 0.18);
        alpha = 0.4;
    } else if (mat == M_OVARY) {
        albedo = float3(0.33, 0.52, 0.21);
        sheen = 0.35;
        if (cap == 1) albedo = float3(0.58, 0.72, 0.42);
    } else if (mat == M_OVULE) {
        albedo = float3(0.80, 0.84, 0.64);
        alpha = 0.25;
        f0 = 0.04;
    } else if (mat == M_STAMEN) {
        albedo = float3(0.86, 0.84, 0.70);
        trans = 0.25;
        if (cap == 1) albedo = float3(0.70, 0.73, 0.60);
    } else if (mat == M_ANTHER) {
        albedo = float3(0.82, 0.68, 0.30) * (0.9 + 0.2 * noise3(p * 120.0));
        alpha = 0.5;
    } else if (mat == M_STYLE) {
        albedo = float3(0.55, 0.74, 0.36);
        trans = 0.3;
        wrap = 0.3;
        if (length(p - STIG_C) < STIG_R * 1.05) {
            // Wet and non-papillate: a smooth, glistening surface.
            albedo = float3(0.70, 0.80, 0.55);
            alpha = 0.08;
            f0 = 0.02;
        }
        // The pollen tube inside, seen through the translucent style.
        float dmin = 1e9;
        float depthW = 0.0;
        float stepIn = max(pixSize * 0.5, 0.003);
        for (int k = 0; k < 40; k++) {
            float3 q = p + rd * (float(k) * stepIn);
            float dk = tubeAxisDistance(q, tubeP.y, cverts, chunks);
            if (dk < dmin) { dmin = dk; depthW = float(k); }
        }
        float w = max(TUBE_R, pixSize * 1.2) * 1.3;
        float glow = exp(-(dmin * dmin) / (w * w)) * exp(-depthW * stepIn * 6.0);
        albedo = mix(albedo, float3(0.95, 0.66, 0.12), 0.85 * glow);
        emit += float3(0.9, 0.55, 0.08) * 0.35 * glow;
    } else if (mat == M_HAIR) {
        albedo = float3(0.90, 0.90, 0.86);
        sheen = 0.3;
        trans = 0.3;
    } else if (mat == M_POLLEN) {
        GrainG g = grains[nearestGrain(p, grains)];
        float3 q = p - g.c.xyz;
        float3 dir = normalize(q);
        // Bump from the reticulum, in the grain's own frame so it turns with it.
        float3 t1 = normalize(cross(n, abs(n.y) < 0.9 ? float3(0, 1, 0) : float3(1, 0, 0)));
        float3 t2 = cross(n, t1);
        float eps = 0.004;
        float h0 = exineHeight(dir);
        float h1 = exineHeight(normalize(dir + t1 * eps));
        float h2 = exineHeight(normalize(dir + t2 * eps));
        float3 grad = t1 * (h1 - h0) + t2 * (h2 - h0);
        n = normalize(n - grad * 2.2);
        albedo = float3(0.88, 0.62, 0.14) * (0.8 + 0.25 * h0);
        // Three pores, 120° apart on the equator.
        float3 pole = g.pole.xyz;
        float3 e1 = g.pore.xyz;
        float3 e2 = cross(pole, e1);
        for (int k = 0; k < 3; k++) {
            float a = float(k) * 2.0943951;
            float3 pd = e1 * cos(a) + e2 * sin(a);
            float c = dot(dir, pd);
            float rim = smoothstep(0.970, 0.980, c);
            float hole = smoothstep(0.986, 0.992, c);
            albedo = mix(albedo, float3(0.97, 0.80, 0.35), rim * (1.0 - hole) * 0.6);
            albedo = mix(albedo, float3(0.55, 0.36, 0.08), hole);
        }
        alpha = 0.35;
        f0 = 0.04;
        wrap = 0.35;
    } else if (mat == M_TUBE) {
        albedo = float3(0.96, 0.72, 0.20);
        emit += float3(0.6, 0.38, 0.06) * 0.3;
        trans = 0.3;
        // Step 29: in the cut-open inset, a deeper gold — the key's colour —
        // so a 10 µm thread reads against the pale cut face. (A living tube
        // is colourless; microscopists stain it to see it. MODEL.)
        if (tubeP.z < 1e8) { albedo = float3(0.92, 0.50, 0.02); emit = float3(0.35, 0.16, 0.0); }
    } else if (mat == M_POD) {
        albedo = float3(0.13, 0.30, 0.09) * (0.9 + 0.15 * noise3(p * 0.4));
        alpha = 0.3;
        f0 = 0.04;
        sheen = 0.15;
    }

    // Thin tissue is lit from whichever side the light is on, and passes some
    // of it through: the shadow ray leaves from that side.
    float nk = dot(n, KEY_DIR);
    float3 faceL = (trans > 0.0 && nk < 0.0) ? -n : n;
    float sh = softShadow(p + faceL * e * 6.0, KEY_DIR, layer, tubeP, max(e * 10.0, 0.004), SCENE);
    float front = max((nk + wrap) / (1.0 + wrap), 0.0);
    float back = max(-nk, 0.0) * trans;
    float nf = dot(n, v);
    float fill = max((nf + wrap) / (1.0 + wrap), 0.0);
    float3 diffuse = albedo * (KEY_COL * (front + back * float3(0.95, 1.0, 0.85)) * sh
                             + FILL_COL * fill + skyIrradiance(n) * occ);
    float3 spec = KEY_COL * ggx(n, v, KEY_DIR, alpha, f0) * sh + FILL_COL * ggx(n, v, v, alpha, f0) * 0.5;
    float rimAmt = pow(1.0 - max(nf, 0.0), 3.0) * sheen;
    return diffuse + spec + emit + float3(rimAmt) * occ;
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

float3 cameraRay(constant View &V, float2 pixel) {
    float aspect = float(V.width) / float(V.height);
    float sx = (2.0 * pixel.x / float(V.width) - 1.0) * aspect * V.tanHalf;
    float sy = (1.0 - 2.0 * pixel.y / float(V.height)) * V.tanHalf;
    return normalize(V.fwd.xyz + sx * V.right.xyz + sy * V.up.xyz);
}

kernel void render(device float4 *pixels [[buffer(0)]],
                   device float4 *aux [[buffer(1)]],
                   constant View &V [[buffer(2)]],
                   device const SpineVert *spine [[buffer(3)]],
                   device const float4 *spineChunks [[buffer(4)]],
                   device const float4 *cverts [[buffer(5)]],
                   device const Chunk *chunks [[buffer(6)]],
                   device const AntherG *anthers [[buffer(7)]],
                   device const GrainG *grains [[buffer(8)]],
                   device const float4 *ovules [[buffer(9)]],
                   device const HairG *hairs [[buffer(10)]],
                   uint2 gid [[thread_position_in_grid]]) {
    uint x = gid.x + V.colOffset;
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
            int mat, cap;
            if (march(V.pos.xyz, rd, V.layer, float3(V.tubeMinV, V.tubeGrownV, V.cutZ), pixAngle, t, mat, cap, SCENE)) {
                float3 p = V.pos.xyz + rd * t;
                sum += toneMap(shade(p, rd, mat, cap, V.layer, float3(V.tubeMinV, V.tubeGrownV, V.cutZ), pixAngle * t, SCENE));
                cover += 1.0;
            }
        }
    }
    float n = float(S * S);
    pixels[y * V.width + x] = float4(sum / n, cover / n);

    float3 rd = cameraRay(V, float2(x, y) + 0.5);
    float t;
    int mat, cap;
    float4 a = float4(0.0);
    if (march(V.pos.xyz, rd, V.layer, float3(V.tubeMinV, V.tubeGrownV, V.cutZ), pixAngle, t, mat, cap, SCENE)) a = float4(float(mat), float(cap), t, 0.0);
    aux[y * V.width + x] = a;
}

kernel void probe(device const float4 *points [[buffer(0)]],
                  device float2 *out [[buffer(1)]],
                  constant int &mode [[buffer(2)]],
                  device const SpineVert *spine [[buffer(3)]],
                  device const float4 *spineChunks [[buffer(4)]],
                  device const float4 *cverts [[buffer(5)]],
                  device const Chunk *chunks [[buffer(6)]],
                  device const AntherG *anthers [[buffer(7)]],
                  device const GrainG *grains [[buffer(8)]],
                  device const float4 *ovules [[buffer(9)]],
                  device const HairG *hairs [[buffer(10)]],
                  constant float &grown [[buffer(11)]],
                  uint id [[thread_position_in_grid]]) {
    int mat, cap;
    float d = sceneSDF(points[id].xyz, mode, float3(0.0, grown, 1e9), mat, cap, SCENE);
    out[id] = float2(d, float(mat));
}
"""
