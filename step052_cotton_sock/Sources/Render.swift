// One still, four scales, one kernel. Each pixel's GPU thread asks which
// view it belongs to — the knit (main view, mm), the yarn inset (µm), the
// cut-end inset (µm, the same world as the yarn inset, looked at closer) or
// the molecule inset (Å) — and marches its ray into that view's scene.
//
// Every surface is a distance function. The yarn loops are tubes round a
// polyline (exact); the fibres are twisted kidney-section ribbons, measured
// in each segment's own cross-section plane and bounded between mitre
// planes, then divided once by a constant for the twist (see ribbonBound).
// All cameras are ORTHOGRAPHIC, so each view's scale bar is true everywhere
// in it.

import CoreGraphics
import Foundation
import ImageIO
import Metal
import UniformTypeIdentifiers
import simd

// MARK: - framing

/// The frame is 16:9; view positions are in units of the frame's HEIGHT.
let frameAspect: Float = 16.0 / 9.0

/// The main view: millimetres across the full frame width, and the camera.
let mainViewWidth: Float = 7.0
let mainCentre = SIMD3<Float>(-0.9, 0.0, -1.6)
let mainDirection: SIMD3<Float> = simd_normalize(SIMD3<Float>(0.10, -0.62, -0.78))

/// The yarn inset: a circle, how many micrometres its diameter spans, and
/// where it looks in the yarn's world.
let insetCentre = SIMD2<Float>(1.325, 0.375)
let insetRadius: Float = 0.335
let insetField: Float = 560
let insetLookAt = SIMD3<Float>(20, -10, 0)
let insetDirection: SIMD3<Float> = simd_normalize(SIMD3<Float>(-0.06, -0.30, -1.0))

/// The cut-end inset: the same world, 42 µm across, looking into the cut.
let cutCentre = SIMD2<Float>(1.615, 0.835)
let cutRadius: Float = 0.135
let cutField: Float = 42

/// The molecule inset, Å across its diameter.
let moleculeCentre = SIMD2<Float>(1.105, 0.835)
let moleculeRadius: Float = 0.145
let moleculeField: Float = 25.0
let moleculeBarNanometres: Float = 0.5

/// An orthographic camera: where the frame centre looks, and its axes.
struct OrthoCamera {
    var centre: SIMD3<Float>
    var forward: SIMD3<Float>
    var right: SIMD3<Float>
    var up: SIMD3<Float>
    var halfWidth: Float

    init(centre: SIMD3<Float>, forward: SIMD3<Float>, halfWidth: Float, upHint: SIMD3<Float> = SIMD3<Float>(0, 1, 0)) {
        self.centre = centre
        self.forward = forward
        self.right = simd_normalize(simd_cross(forward, upHint))
        self.up = simd_cross(self.right, forward)
        self.halfWidth = halfWidth
    }

    func project(_ p: SIMD3<Float>, viewCentre: SIMD2<Float>, viewHalf: Float) -> SIMD2<Float> {
        let q: SIMD3<Float> = p - centre
        let sx: Float = simd_dot(q, right) / halfWidth
        let sy: Float = simd_dot(q, up) / halfWidth
        return SIMD2<Float>(viewCentre.x + sx * viewHalf, viewCentre.y - sy * viewHalf)
    }
}

let mainCamera = OrthoCamera(centre: mainCentre, forward: mainDirection, halfWidth: mainViewWidth / 2)
let insetCamera = OrthoCamera(centre: insetLookAt, forward: insetDirection, halfWidth: insetField / 2)

/// The cut-end camera looks back along the fibre into its cut face, from a
/// little above and to the side, so the face and the twisting ribbon behind
/// it both show. Built from the fibre as drawn.
func cutCamera(_ yarn: YarnPiece) -> OrthoCamera {
    let n: Int = yarn.fuzz.points.count
    let end: SIMD3<Float> = yarn.fuzz.points[n - 1]
    let t: SIMD3<Float> = simd_normalize(yarn.fuzz.points[n - 1] - yarn.fuzz.points[n - 2])
    // The ribbon's own axes at the cut: W across its width, N through it.
    let w0: SIMD3<Float> = yarn.fuzz.widths[n - 1]
    let w: SIMD3<Float> = simd_normalize(w0 - t * simd_dot(w0, t))
    let nrm: SIMD3<Float> = simd_cross(t, w)
    // 28° off the face's normal, tilted across the width so the lumen's
    // slit opens to the eye; the width runs level in the inset.
    let look: SIMD3<Float> = simd_normalize(-t * 0.88 + nrm * 0.40 + w * 0.22)
    return OrthoCamera(centre: end - t * 16, forward: look, halfWidth: cutField / 2, upHint: nrm)
}

func projectMain(_ p: SIMD3<Float>, width: Int, height: Int) -> SIMD2<Float> {
    mainCamera.project(p, viewCentre: SIMD2(Float(width) / 2, Float(height) / 2), viewHalf: Float(width) / 2)
}

func projectInset(_ p: SIMD3<Float>, height: Int) -> SIMD2<Float> {
    let h: Float = Float(height)
    return insetCamera.project(p, viewCentre: insetCentre * h, viewHalf: insetRadius * h)
}

func projectCut(_ p: SIMD3<Float>, camera: OrthoCamera, height: Int) -> SIMD2<Float> {
    let h: Float = Float(height)
    return camera.project(p, viewCentre: cutCentre * h, viewHalf: cutRadius * h)
}

// MARK: - light

/// A softbox above, front-left; a dim fill from beside the camera; and a
/// rim light from behind, which is what makes fuzz visible against a knit.
let keyDirection: SIMD3<Float> = simd_normalize(SIMD3<Float>(-0.45, 0.85, 0.35))
let fillDirection: SIMD3<Float> = simd_normalize(-mainDirection + SIMD3<Float>(0.35, 0.3, 0))
let rimDirection: SIMD3<Float> = simd_normalize(SIMD3<Float>(0.25, 0.45, -1.0))
let keyColour = SIMD3<Float>(1.0, 0.975, 0.94) * 1.55
let fillColour = SIMD3<Float>(0.90, 0.94, 1.0) * 0.42
let rimColour = SIMD3<Float>(1.0, 0.98, 0.95) * 0.9

/// Cotton's colour. MODEL: undyed, unbleached cotton is creamy off-white;
/// these are that as linear-light numbers, not a measured reflectance.
let cottonAlbedo = SIMD3<Float>(0.74, 0.69, 0.60)
/// The one fibre the inset follows is tinted, a drawing choice the caption
/// states; all cotton fibres are the same colour.
let highlightAlbedo = SIMD3<Float>(0.86, 0.64, 0.30)

/// Behind the fabric: the dim inside of the sock. MODEL: a plane 1.2 mm
/// below the face.
let backingDepth: Float = 1.2

// MARK: - GPU packing

/// A ribbon set for the GPU: chunks of up to 8 segments with bounding
/// spheres, and vertices (position+arc, width+twist, mitre).
struct RibbonPack {
    var chunks: [SIMD4<Float>] = []      // 2 per chunk
    var vertices: [SIMD4<Float>] = []    // 3 per vertex
    var chunkCount: Int { chunks.count / 2 }
}

let chunkSegments: Int = 8

func packRibbons(_ ribbons: [Ribbon], reach: Float) -> RibbonPack {
    var pack = RibbonPack()
    for rib in ribbons {
        let base: Int = pack.vertices.count / 3
        let m: [SIMD3<Float>] = rib.mitres
        for k in 0..<rib.points.count {
            pack.vertices.append(SIMD4<Float>(rib.points[k], rib.arc[k]))
            pack.vertices.append(SIMD4<Float>(rib.widths[k], rib.twist[k]))
            pack.vertices.append(SIMD4<Float>(m[k], 0))
        }
        let segs: Int = rib.points.count - 1
        var first: Int = 0
        while first < segs {
            let count: Int = min(chunkSegments, segs - first)
            var c = SIMD3<Float>(0, 0, 0)
            for k in first...(first + count) { c += rib.points[k] }
            c /= Float(count + 1)
            var r: Float = 0
            for k in first...(first + count) { r = max(r, simd_distance(c, rib.points[k])) }
            r += reach * rib.scale * 1.15
            var flags: Float = 0
            if first == 0 { flags += 1 }
            if first + count == segs { flags += 2 }
            if rib.highlight { flags += 4 }
            pack.chunks.append(SIMD4<Float>(c, r))
            pack.chunks.append(SIMD4<Float>(Float(base + first), Float(count), flags, 0))
            first += count
        }
    }
    return pack
}

/// The per-object bound the ribbon distance is divided by. Measured in a
/// segment's cross-section plane, the distance to a TWISTED ribbon can
/// exceed the true distance by up to √(1 + (τρ)²), ρ the reach from the
/// axis; the chord polyline and the mitre planes add a little: 5% margin.
func ribbonBound(_ s: CrossSection, rate: Float) -> Float {
    let tr: Float = rate * s.reach
    return (1 + tr * tr).squareRoot() * 1.05
}

/// The knit loop polyline, and bounding spheres for chunks of 8 segments.
func knitChunks(_ line: [SIMD3<Float>], r: Float) -> [SIMD4<Float>] {
    var out: [SIMD4<Float>] = []
    var first: Int = 0
    while first < line.count - 1 {
        let count: Int = min(8, line.count - 1 - first)
        var c = SIMD3<Float>(0, 0, 0)
        for k in first...(first + count) { c += line[k] }
        c /= Float(count + 1)
        var rad: Float = 0
        for k in first...(first + count) { rad = max(rad, simd_distance(c, line[k])) }
        out.append(SIMD4<Float>(c, rad + r))
        first += count
    }
    return out
}

func elementCode(_ e: String) -> Float {
    switch e {
    case "C": return 0
    case "O": return 1
    default: return 2
    }
}

func gpuMolecule(_ m: Molecule) -> (atoms: [SIMD4<Float>], bonds: [SIMD4<Float>]) {
    (m.atoms.map { SIMD4<Float>($0.position, elementCode($0.element)) },
     m.bonds.map { SIMD4<Float>(Float($0.0), Float($0.1), 0, 0) })
}

// MARK: - the kernel

/// How far a ray may trust a distance. Every distance here is a bound, and
/// the tests measure how much each over-reports outside its surface.
let stepScale: Float = 0.8

func metal(_ v: SIMD3<Float>) -> String { "float3(\(v.x), \(v.y), \(v.z))" }
func metal2(_ v: SIMD2<Float>) -> String { "float2(\(v.x), \(v.y))" }
func metalArray(_ v: [SIMD4<Float>]) -> String {
    v.map { "float4(\($0.x), \($0.y), \($0.z), \($0.w))" }.joined(separator: ", ")
}

func kernelSource(scene: Scene) -> String {
    let mc: OrthoCamera = mainCamera
    let ic: OrthoCamera = insetCamera
    let cc: OrthoCamera = cutCamera(scene.yarn)
    let s: CrossSection = scene.yarn.section
    let segLength: Float = scene.knit.loopLength / Float(scene.knitLine.count - 1)
    let line: [SIMD4<Float>] = scene.knitLine.map { SIMD4<Float>($0, segLength) }
    let chunks: [SIMD4<Float>] = knitChunks(scene.knitLine, r: scene.knit.r)
    let k: Knit = scene.knit
    return """
    #include <metal_stdlib>
    using namespace metal;

    struct Params { uint width; uint height; uint rowOffset; uint samples; };

    constant float STEP_SCALE = \(stepScale);
    constant int MAIN_CHUNKS = \(scene.hairPack.chunkCount);
    constant int INSET_CHUNKS = \(scene.insetPack.chunkCount);
    constant int ATOMS = \(scene.molecule.atoms.count);
    constant int BONDS = \(scene.molecule.bonds.count);

    // The knit (mm).
    constant float KW = \(k.w);
    constant float KC = \(k.c);
    constant float KD = \(k.d);
    constant float KR = \(k.r);
    constant int KSEG = \(scene.knitLine.count - 1);
    constant float4 KLINE[\(line.count)] = { \(metalArray(line)) };
    constant int KCHUNKS = \(chunks.count);
    constant float4 KCHUNK[\(chunks.count)] = { \(metalArray(chunks)) };
    constant float BACK_Y = \(-backingDepth);

    // The fibre's cross-section (µm) and the ribbon bound.
    constant float RB = \(s.bendRadius);
    constant float2 SCO = float2(\(sin(s.halfArc)), \(cos(s.halfArc)));
    constant float BT = \(s.halfThickness);
    constant float2 SCL = float2(\(sin(s.lumenHalfArc)), \(cos(s.lumenHalfArc)));
    constant float BL = \(s.lumenHalfThickness);
    constant float RIB_K = \(scene.ribbonK);

    // The yarn inset (µm): the core of hidden fibres, and the twist.
    constant float3 YARN_O = \(metal(yarnOrigin));
    constant float3 YARN_A = \(metal(yarnAxis));
    constant float CORE_R = \(scene.yarn.core);
    constant float YARN_TWIST = \(yarnTwistPerMillimetre / 1000 * yarnHand);   // turns per µm, signed
    constant float KNIT_TWIST = \(yarnTwistPerMillimetre * yarnHand);          // turns per mm
    constant float SURF_N = \(Float(scene.yarn.fibreCount));

    // Cameras.
    constant float3 M_CENTRE = \(metal(mc.centre));
    constant float3 M_FWD = \(metal(mc.forward));
    constant float3 M_RIGHT = \(metal(mc.right));
    constant float3 M_UP = \(metal(mc.up));
    constant float M_HALF = \(mc.halfWidth);
    constant float2 I_C = \(metal2(insetCentre));
    constant float I_R = \(insetRadius);
    constant float3 I_CENTRE = \(metal(ic.centre));
    constant float3 I_FWD = \(metal(ic.forward));
    constant float3 I_RIGHT = \(metal(ic.right));
    constant float3 I_UP = \(metal(ic.up));
    constant float I_HALF = \(ic.halfWidth);
    constant float2 X_C = \(metal2(cutCentre));
    constant float X_R = \(cutRadius);
    constant float3 X_CENTRE = \(metal(cc.centre));
    constant float3 X_FWD = \(metal(cc.forward));
    constant float3 X_RIGHT = \(metal(cc.right));
    constant float3 X_UP = \(metal(cc.up));
    constant float X_HALF = \(cc.halfWidth);
    constant float2 Q_C = \(metal2(moleculeCentre));
    constant float Q_R = \(moleculeRadius);
    constant float Q_HALF = \(moleculeField / 2);

    // Light.
    constant float3 KEY_DIR = \(metal(keyDirection));
    constant float3 FILL_DIR = \(metal(fillDirection));
    constant float3 RIM_DIR = \(metal(rimDirection));
    constant float3 KEY_COL = \(metal(keyColour));
    constant float3 FILL_COL = \(metal(fillColour));
    constant float3 RIM_COL = \(metal(rimColour));
    constant float3 COTTON = \(metal(cottonAlbedo));
    constant float3 HILITE = \(metal(highlightAlbedo));

    // ------------------------------------------------------------ helpers

    float hash3(float3 p) {
        p = fract(p * 0.3183099 + 0.1);
        p *= 17.0;
        return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
    }
    float hash1(float x) { return fract(sin(x * 127.1) * 43758.5453); }
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

    float segDist(float3 p, float3 a, float3 b) {
        float3 ab = b - a;
        float t = clamp(dot(p - a, ab) / dot(ab, ab), 0.0, 1.0);
        return length(p - a - ab * t);
    }

    // ------------------------------------------------------------ the cell wall (µm)

    // Exact distance to a bent capsule (Quílez's sdArc), p folded, relative
    // to the arc's circle centre.
    float arcDist(float2 p, float2 sc, float ra, float rb) {
        return ((sc.y * p.x > sc.x * p.y) ? length(p - sc * ra) : abs(length(p) - ra)) - rb;
    }
    // The wall in its cross-section: u across the width, v across the
    // thickness. sub = 2 when p is inside the lumen.
    float wallDist(float u, float v, thread int &sub) {
        float2 q = float2(abs(u), RB - v);
        float outer = arcDist(q, SCO, RB, BT);
        float lumen = arcDist(q, SCL, RB, BL);
        sub = (outer < 0.0 && lumen < 0.0) ? 2 : 0;
        return max(outer, -lumen);
    }

    // ------------------------------------------------------------ ribbons

    // One set of ribbons: chunks (sphere; first vertex, count, flags) and
    // vertices (P, arc; W, twist; mitre). Returns the distance in scene
    // units, already divided by RIB_K; `hl` is 1 for the highlighted fibre;
    // `sub` 1 on a cut face, 2 in the lumen.
    float ribbons(float3 p, constant float4 *C, constant float4 *V, int nChunks, float scale,
                  thread int &hl, thread int &sub) {
        float best = 1e9;
        hl = 0; sub = 0;
        for (int c = 0; c < nChunks; c++) {
            float4 sph = C[c * 2];
            // The chunk's bounding sphere is a lower bound on the distance
            // to anything in it. Each segment's value is floored at it (a
            // max of two lower bounds is one), so skipping the chunk when
            // the bound exceeds the best so far can never change the answer.
            float lb = length(p - sph.xyz) - sph.w;
            if (lb > best) continue;
            float4 meta = C[c * 2 + 1];
            int first = int(meta.x + 0.5);
            int count = int(meta.y + 0.5);
            int flags = int(meta.z + 0.5);
            for (int k = 0; k < count; k++) {
                int i = first + k;
                float3 a = V[i * 3].xyz, b = V[(i + 1) * 3].xyz;
                float3 ab = b - a;
                float L2 = dot(ab, ab);
                float t = clamp(dot(p - a, ab) / L2, 0.0, 1.0);
                float3 T = ab * rsqrt(L2);
                float3 W = mix(V[i * 3 + 1].xyz, V[(i + 1) * 3 + 1].xyz, t);
                W = normalize(W - T * dot(W, T));
                float3 N = cross(T, W);
                float3 q = p - (a + ab * t);
                int s2;
                float d2 = wallDist(dot(q, W) / scale, dot(q, N) / scale, s2) * scale;
                float s0 = -dot(p - a, V[i * 3 + 2].xyz);
                float s1 = dot(p - b, V[(i + 1) * 3 + 2].xyz);
                // Inside the segment's mitre slab: the in-plane distance —
                // capped by a plane only at the fibre's real ends (its root
                // and its cut), never at the mitres it shares with its
                // neighbours, which are not surfaces. Past a mitre plane: the
                // distance to that end face, √(d2² + slab²) — continuous
                // with the inside at the plane, and never below the sphere
                // bound, so culling can't make it jump.
                bool firstSeg = (flags & 1) != 0 && k == 0;
                bool lastSeg = (flags & 2) != 0 && k == count - 1;
                float cap = max(firstSeg ? s0 : -1e9, lastSeg ? s1 : -1e9);
                float slab = max(s0, s1);
                float d = slab <= 0.0 ? max(d2, cap) : (d2 > 0.0 ? sqrt(d2 * d2 + slab * slab) : slab);
                d = max(d, lb);
                if (d < best) {
                    best = d;
                    hl = (flags & 4) != 0 ? 1 : 0;
                    sub = (lastSeg && s1 > d2 - 1e-4 * scale) ? 1 : s2;
                }
            }
        }
        return best / RIB_K;
    }

    // ------------------------------------------------------------ the knit (mm)

    // Knit coordinates: x along the course, y up the wale, z through the
    // fabric (+z away from the face). World y is out of the face.
    float3 toKnit(float3 p) { return float3(p.x, -p.z, -p.y); }

    // Exact distance to the yarn tubes: the 3×3 loops round the point's
    // lattice cell, each a polyline tube, chunks culled by sphere. (No
    // early out far from the fabric: a slab bound there is lower than the
    // exact distance, and switching to it is a jump the tests reject.)
    float knitSDF(float3 p) {
        float3 q = toKnit(p);
        float i0 = round(q.x / KW), j0 = round(q.y / KC);
        float best = 1e9;
        for (int dj = -1; dj <= 1; dj++) {
            for (int di = -1; di <= 1; di++) {
                float3 l = q - float3((i0 + float(di)) * KW, (j0 + float(dj)) * KC, 0.0);
                for (int c = 0; c < KCHUNKS; c++) {
                    if (length(l - KCHUNK[c].xyz) - KCHUNK[c].w > best) continue;
                    int s0 = c * 8;
                    int s1 = min(s0 + 8, KSEG);
                    for (int s = s0; s < s1; s++) best = min(best, segDist(l, KLINE[s].xyz, KLINE[s + 1].xyz) - KR);
                }
            }
        }
        return best;
    }

    // Where on the yarn a surface point is: arc along the loop (mm), angle
    // round the yarn, for the fibre texture.
    float2 knitDetail(float3 p) {
        float3 q = toKnit(p);
        float i0 = round(q.x / KW), j0 = round(q.y / KC);
        float best = 1e9;
        float2 res = float2(0.0);
        for (int dj = -1; dj <= 1; dj++) {
            for (int di = -1; di <= 1; di++) {
                float3 l = q - float3((i0 + float(di)) * KW, (j0 + float(dj)) * KC, 0.0);
                for (int s = 0; s < KSEG; s++) {
                    float3 a = KLINE[s].xyz, b = KLINE[s + 1].xyz;
                    float3 ab = b - a;
                    float t = clamp(dot(l - a, ab) / dot(ab, ab), 0.0, 1.0);
                    float3 c = a + ab * t;
                    float d = length(l - c);
                    if (d < best) {
                        best = d;
                        float3 T = normalize(ab);
                        float3 e1 = normalize(cross(T, float3(0.0, 0.0, 1.0)));
                        float3 e2 = cross(T, e1);
                        float3 r = l - c;
                        float along = (float(s) + t) * KLINE[KSEG].w + (i0 + float(di)) * 3.1 + (j0 + float(dj)) * 7.7;
                        res = float2(along, atan2(dot(r, e2), dot(r, e1)));
                    }
                }
            }
        }
        return res;
    }

    // Materials: 1 backing, 2 yarn, 3 fuzz.
    float mainSDF(float3 p, constant float4 *HC, constant float4 *HV, thread int &mat, thread int &hl, thread int &sub) {
        float d = p.y - BACK_Y;
        mat = 1; hl = 0; sub = 0;
        float k = knitSDF(p);
        if (k < d) { d = k; mat = 2; }
        int h1, s1;
        float h = ribbons(p, HC, HV, MAIN_CHUNKS, 0.001, h1, s1);
        if (h < d) { d = h; mat = 3; hl = h1; sub = s1; }
        return d;
    }
    float mainDist(float3 p, constant float4 *HC, constant float4 *HV) { int m, h, s; return mainSDF(p, HC, HV, m, h, s); }

    float3 mainNormal(float3 p, constant float4 *HC, constant float4 *HV) {
        const float e = 0.0012;
        float3 k1 = float3(1, -1, -1), k2 = float3(-1, -1, 1), k3 = float3(-1, 1, -1), k4 = float3(1, 1, 1);
        return normalize(k1 * mainDist(p + k1 * e, HC, HV) + k2 * mainDist(p + k2 * e, HC, HV)
                       + k3 * mainDist(p + k3 * e, HC, HV) + k4 * mainDist(p + k4 * e, HC, HV));
    }

    bool marchMain(float3 ro, float3 rd, constant float4 *HC, constant float4 *HV,
                   thread float &t, thread int &mat, thread int &hl, thread int &sub) {
        t = 0.0;
        for (int i = 0; i < 400; i++) {
            float d = mainSDF(ro + rd * t, HC, HV, mat, hl, sub);
            if (d < 0.0004) return true;
            t += max(d * STEP_SCALE, 0.0002);
            if (t > 60.0) break;
        }
        return false;
    }

    float mainShadow(float3 ro, float3 rd, constant float4 *HC, constant float4 *HV) {
        float res = 1.0;
        float t = 0.004;
        for (int i = 0; i < 72; i++) {
            float h = mainDist(ro + rd * t, HC, HV);
            res = min(res, 7.0 * h / t);
            t += clamp(h * STEP_SCALE, 0.003, 0.25);
            if (res < 0.003 || t > 4.0) break;
        }
        return clamp(res, 0.0, 1.0);
    }

    float mainOcclusion(float3 p, float3 n, constant float4 *HC, constant float4 *HV) {
        float occ = 0.0, w = 1.0;
        for (int i = 1; i <= 5; i++) {
            float h = 0.02 + 0.035 * float(i * i);
            occ += w * clamp((h - mainDist(p + n * h, HC, HV)) / h, 0.0, 1.0);
            w *= 0.7;
        }
        return clamp(1.0 - 0.55 * occ, 0.0, 1.0);
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

    float3 skyIrr(float3 n) { return mix(float3(0.20, 0.19, 0.18), float3(0.36, 0.37, 0.40), 0.5 + 0.5 * n.y); }

    // Cotton: a matte, slightly translucent fibre with a faint waxy sheen.
    // Light wraps a little round it, and a rim light behind picks out edges.
    float3 shadeCotton(float3 n, float3 v, float3 alb, float sh, float occ, float rimSh) {
        float wrap = 0.3;
        float key = max((dot(n, KEY_DIR) + wrap) / (1.0 + wrap), 0.0) * sh;
        float fill = max(dot(n, FILL_DIR), 0.0);
        float rim = pow(1.0 - max(dot(n, v), 0.0), 2.5) * max(dot(-v, RIM_DIR) * 0.5 + 0.6, 0.0);
        float3 col = alb * (KEY_COL * key + FILL_COL * fill + skyIrr(n) * occ);
        col += RIM_COL * rim * 0.55 * rimSh * mix(alb, float3(1.0), 0.4);
        col += KEY_COL * ggx(n, v, KEY_DIR, 0.35, 0.04) * sh * 0.5;
        return col;
    }

    float3 shadeMain(float3 ro, float3 rd, constant float4 *HC, constant float4 *HV, thread float4 &aux) {
        float t;
        int mat, hl, sub;
        aux = float4(1.0, 0.0, 0.0, 0.0);
        if (!marchMain(ro, rd, HC, HV, t, mat, hl, sub)) return float3(0.02);
        float3 p = ro + rd * t;
        aux.y = float(mat);
        float3 v = -rd;
        if (mat == 1) {
            // The inside of the sock, deep in shadow.
            float occ = mainOcclusion(p, float3(0, 1, 0), HC, HV);
            return COTTON * 0.10 * occ;
        }
        float3 n = mainNormal(p, HC, HV);
        float sh = mainShadow(p + n * 0.003, KEY_DIR, HC, HV);
        float occ = mainOcclusion(p, n, HC, HV);
        float3 alb = COTTON;
        if (mat == 2) {
            // The yarn's surface fibres: fine streaks lying along the
            // twist's helix, about as many as the surface layer has, each a
            // little different in brightness.
            float2 dt = knitDetail(p);
            float phase = dt.y * SURF_N / 6.2831853 - dt.x * KNIT_TWIST * SURF_N;
            float cell = floor(phase);
            float f = fract(phase);
            float ridge = sin(3.14159265 * f);
            float vary = 0.88 + 0.22 * hash1(cell + floor(dt.x * 1.3) * 17.0);
            alb *= vary * (0.86 + 0.14 * ridge);
            float3 bump = float3(noise3(p * 900.0), noise3(p * 900.0 + 3.1), noise3(p * 900.0 + 7.3)) - 0.5;
            n = normalize(n + 0.18 * (bump - n * dot(bump, n)));
        }
        float rimSh = mat == 3 ? 1.0 : 0.35;
        return shadeCotton(n, v, alb, sh, occ, rimSh);
    }

    // ------------------------------------------------------------ the yarn inset (µm)

    float coreSDF(float3 p) {
        float3 q = p - YARN_O;
        return length(q - YARN_A * dot(q, YARN_A)) - CORE_R;
    }

    // Materials: 1 core, 2 surface fibre, 3 the highlighted fibre.
    float insetSDF(float3 p, constant float4 *C, constant float4 *V, thread int &mat, thread int &sub) {
        float d = coreSDF(p);
        mat = 1; sub = 0;
        int hl, s;
        float r = ribbons(p, C, V, INSET_CHUNKS, 1.0, hl, s);
        if (r < d) { d = r; mat = hl == 1 ? 3 : 2; sub = s; }
        return d;
    }
    float insetDist(float3 p, constant float4 *C, constant float4 *V) { int m, s; return insetSDF(p, C, V, m, s); }

    float3 insetNormal(float3 p, constant float4 *C, constant float4 *V, float e) {
        float3 k1 = float3(1, -1, -1), k2 = float3(-1, -1, 1), k3 = float3(-1, 1, -1), k4 = float3(1, 1, 1);
        return normalize(k1 * insetDist(p + k1 * e, C, V) + k2 * insetDist(p + k2 * e, C, V)
                       + k3 * insetDist(p + k3 * e, C, V) + k4 * insetDist(p + k4 * e, C, V));
    }

    bool marchInset(float3 ro, float3 rd, constant float4 *C, constant float4 *V, float eps,
                    thread float &t, thread int &mat, thread int &sub) {
        t = 0.0;
        for (int i = 0; i < 400; i++) {
            float d = insetSDF(ro + rd * t, C, V, mat, sub);
            if (d < eps) return true;
            t += max(d * STEP_SCALE, eps * 0.5);
            if (t > 2400.0) break;
        }
        return false;
    }

    float insetShadow(float3 ro, float3 rd, constant float4 *C, constant float4 *V) {
        float res = 1.0;
        float t = 1.0;
        for (int i = 0; i < 64; i++) {
            float h = insetDist(ro + rd * t, C, V);
            res = min(res, 8.0 * h / t);
            t += clamp(h * STEP_SCALE, 0.8, 40.0);
            if (res < 0.003 || t > 500.0) break;
        }
        return clamp(res, 0.0, 1.0);
    }

    float insetOcclusion(float3 p, float3 n, constant float4 *C, constant float4 *V, float scale) {
        float occ = 0.0, w = 1.0;
        for (int i = 1; i <= 5; i++) {
            float h = scale * (0.5 + 1.2 * float(i * i));
            occ += w * clamp((h - insetDist(p + n * h, C, V)) / h, 0.0, 1.0);
            w *= 0.7;
        }
        return clamp(1.0 - 0.6 * occ, 0.0, 1.0);
    }

    float3 insetBackground(float2 s) {
        return mix(float3(0.80, 0.82, 0.85), float3(0.93, 0.94, 0.95), 0.5 - 0.5 * s.y);
    }

    // `cut` is true in the cut-end inset (finer detail, smaller epsilon).
    float3 shadeInset(float3 ro, float3 rd, constant float4 *C, constant float4 *V, float2 s, bool cut, thread float4 &aux) {
        float t;
        int mat, sub;
        float eps = cut ? 0.01 : 0.04;
        aux = float4(cut ? 3.0 : 2.0, 0.0, 0.0, 0.0);
        if (!marchInset(ro, rd, C, V, eps, t, mat, sub)) return insetBackground(s);
        float3 p = ro + rd * t;
        aux.y = float(mat);
        aux.z = float(sub);
        float3 v = -rd;
        float3 n = insetNormal(p, C, V, cut ? 0.02 : 0.08);
        float sh = insetShadow(p + n * 0.6, KEY_DIR, C, V);
        float occ = insetOcclusion(p, n, C, V, cut ? 0.25 : 1.0);
        float3 alb = mat == 3 ? HILITE : COTTON;
        if (mat == 1) {
            // The core: the fibres beneath, as streaks along the twist.
            float3 q = p - YARN_O;
            float x = dot(q, YARN_A);
            float3 rad = q - YARN_A * x;
            float3 e1 = normalize(cross(YARN_A, float3(0.0, 1.0, 0.0)));
            float3 e2 = cross(YARN_A, e1);
            float th = atan2(dot(rad, e2), dot(rad, e1));
            float ph = th / 6.2831853 * 34.0 - x * YARN_TWIST * 34.0;
            alb *= 0.62 + 0.18 * sin(6.2831853 * ph) + 0.08 * hash1(floor(ph));
        } else {
            // Fine surface texture: the primary wall's wrinkles.
            float3 bump = float3(noise3(p * 0.9), noise3(p * 0.9 + 3.1), noise3(p * 0.9 + 7.3)) - 0.5;
            n = normalize(n + 0.10 * (bump - n * dot(bump, n)));
        }
        if (sub == 1) {
            // The cut face: the cell wall seen end-on, a flat pale section.
            alb = mix(alb, float3(0.93, 0.90, 0.84), 0.55);
        } else if (sub == 2) {
            // Down the lumen: the dried-out hollow, dark.
            return alb * 0.12 * occ;
        }
        return shadeCotton(n, v, alb, sh, occ, 1.0);
    }

    // ------------------------------------------------------------ the molecule (Å)

    float3 atomColour(float code) {
        if (code < 0.5) return float3(0.30, 0.30, 0.32);
        if (code < 1.5) return float3(0.80, 0.07, 0.05);
        return float3(0.92, 0.92, 0.92);
    }
    float atomRadius(float code) {
        if (code < 0.5) return 0.38;
        if (code < 1.5) return 0.36;
        return 0.24;
    }

    bool sphereHit(float3 ro, float3 rd, float3 c, float r, thread float &t) {
        float3 oc = ro - c;
        float b = dot(oc, rd);
        float h = b * b - (dot(oc, oc) - r * r);
        if (h < 0.0) return false;
        t = -b - sqrt(h);
        return t > 0.0;
    }

    bool cylinderHit(float3 ro, float3 rd, float3 a, float3 b, float r, thread float &t, thread float3 &n) {
        float3 ba = b - a;
        float3 oc = ro - a;
        float baba = dot(ba, ba);
        float bard = dot(ba, rd);
        float baoc = dot(ba, oc);
        float k2 = baba - bard * bard;
        float k1 = baba * dot(oc, rd) - baoc * bard;
        float k0 = baba * dot(oc, oc) - baoc * baoc - r * r * baba;
        float h = k1 * k1 - k2 * k0;
        if (h < 0.0 || k2 < 1e-6) return false;
        float tt = (-k1 - sqrt(h)) / k2;
        float y = baoc + tt * bard;
        if (tt <= 0.0 || y < 0.0 || y > baba) return false;
        t = tt;
        n = (oc + tt * rd - ba * y / baba) / r;
        return true;
    }

    // Ball and stick, with a soft glow behind each atom.
    float3 shadeMolecule(float3 ro, float3 rd, constant float4 *A, constant float4 *B, float2 uv, thread float4 &aux) {
        float best = 1e9;
        float3 n = float3(0.0);
        float3 col = float3(0.0);
        float glow = 0.0;
        aux = float4(4.0, 0.0, 0.0, 0.0);
        for (int i = 0; i < ATOMS; i++) {
            float3 c = A[i].xyz;
            float code = A[i].w;
            float r = atomRadius(code);
            float t;
            if (sphereHit(ro, rd, c, r, t) && t < best) {
                best = t; n = normalize(ro + rd * t - c); col = atomColour(code);
                aux.y = 1.0 + code;
            }
            float3 oc = c - ro;
            float along = dot(oc, rd);
            float miss = length(oc - rd * along);
            glow += exp(-pow(max(miss - r, 0.0) / 0.45, 2.0)) * 0.12;
        }
        for (int i = 0; i < BONDS; i++) {
            int ia = int(B[i].x + 0.5), ib = int(B[i].y + 0.5);
            float3 a = A[ia].xyz, b = A[ib].xyz;
            float t;
            float3 cn;
            if (cylinderHit(ro, rd, a, b, 0.11, t, cn) && t < best) {
                best = t; n = cn;
                float y = dot(ro + rd * t - a, b - a) / dot(b - a, b - a);
                col = mix(atomColour(A[ia].w), atomColour(A[ib].w), step(0.5, y)) * 0.9;
                aux.y = 9.0;
            }
        }
        float3 bg = mix(float3(0.95, 0.93, 0.88), float3(0.99, 0.98, 0.95), 0.5 + 0.5 * uv.y);
        float g = min(glow, 0.4);
        if (best > 1e8) return bg * (1.0 - g) + float3(0.30, 0.55, 0.85) * g * 0.5;
        float3 v = -rd;
        float key = max(dot(n, KEY_DIR), 0.0);
        float fill = max(dot(n, FILL_DIR), 0.0);
        float3 h = normalize(v + KEY_DIR);
        float spec = pow(max(dot(n, h), 0.0), 60.0) * 0.5;
        return col * (0.35 + 0.75 * key + 0.35 * fill) + float3(spec);
    }

    // ------------------------------------------------------------ output

    float3 toneMap(float3 x) {
        float L = dot(x, float3(0.2126, 0.7152, 0.0722));
        float Lm = L * (1.0 + L / 16.0) / (1.0 + L);
        float3 c = x * (Lm / max(L, 1e-5)) * 1.18;
        float over = max(max(c.r, max(c.g, c.b)) - 1.0, 0.0);
        c = mix(c, float3(1.0), clamp(over, 0.0, 1.0));
        c = clamp(c, 0.0, 1.0);
        return select(1.055 * pow(c, 1.0 / 2.4) - 0.055, 12.92 * c, c <= 0.0031308);
    }

    kernel void sock(device uchar4 *pixels [[buffer(0)]],
                     device float4 *auxOut [[buffer(1)]],
                     constant Params &P [[buffer(2)]],
                     constant float4 *HC [[buffer(3)]],
                     constant float4 *HV [[buffer(4)]],
                     constant float4 *IC [[buffer(5)]],
                     constant float4 *IV [[buffer(6)]],
                     constant float4 *A [[buffer(7)]],
                     constant float4 *B [[buffer(8)]],
                     uint2 gid [[thread_position_in_grid]]) {
        uint x = gid.x;
        uint y = gid.y + P.rowOffset;
        if (x >= P.width || y >= P.height) return;
        float W = float(P.width), Hh = float(P.height);
        float3 sum = float3(0.0);
        uint N = P.samples;
        float4 aux = float4(0.0);
        for (uint sy = 0; sy < N; sy++) {
            for (uint sx = 0; sx < N; sx++) {
                float2 px = float2(x, y) + float2((float(sx) + 0.5) / float(N), (float(sy) + 0.5) / float(N));
                float2 f = px / Hh;
                float4 a;
                float3 c;
                if (length(f - I_C) < I_R) {
                    float2 s = (f - I_C) / I_R;
                    float3 ro = I_CENTRE + (I_RIGHT * s.x - I_UP * s.y) * I_HALF - I_FWD * 900.0;
                    c = shadeInset(ro, I_FWD, IC, IV, s, false, a);
                } else if (length(f - X_C) < X_R) {
                    float2 s = (f - X_C) / X_R;
                    float3 ro = X_CENTRE + (X_RIGHT * s.x - X_UP * s.y) * X_HALF - X_FWD * 400.0;
                    c = shadeInset(ro, X_FWD, IC, IV, s, true, a);
                } else if (length(f - Q_C) < Q_R) {
                    float2 s = (f - Q_C) / Q_R;
                    float3 ro = float3(s.x * Q_HALF, -s.y * Q_HALF, 30.0);
                    c = shadeMolecule(ro, float3(0.0, 0.0, -1.0), A, B, s, a);
                } else {
                    float2 s = float2(2.0 * px.x / W - 1.0, (1.0 - 2.0 * px.y / Hh) * (Hh / W));
                    float3 ro = M_CENTRE + (M_RIGHT * s.x + M_UP * s.y) * M_HALF - M_FWD * 25.0;
                    c = shadeMain(ro, M_FWD, HC, HV, a);
                }
                sum += toneMap(c);
                if (sx == N / 2 && sy == N / 2) aux = a;
            }
        }
        float3 c = sum / float(N * N);
        pixels[y * P.width + x] = uchar4(uchar3(round(clamp(c, 0.0, 1.0) * 255.0)), 255);
        auxOut[y * P.width + x] = aux;
    }

    // The distance functions at arbitrary points, from the source the
    // picture is drawn with: (main distance, material, inset distance,
    // material) and (knit alone, hairs alone, inset ribbons alone, inset sub).
    kernel void probe(device const float4 *points [[buffer(0)]],
                      device float4 *out [[buffer(1)]],
                      device float4 *out2 [[buffer(2)]],
                      constant float4 *HC [[buffer(3)]],
                      constant float4 *HV [[buffer(4)]],
                      constant float4 *IC [[buffer(5)]],
                      constant float4 *IV [[buffer(6)]],
                      uint id [[thread_position_in_grid]]) {
        float3 p = points[id].xyz;
        int m, hl, sub, mi, si, h2, s2, h3, s3;
        float dm = mainSDF(p, HC, HV, m, hl, sub);
        float di = insetSDF(p, IC, IV, mi, si);
        out[id] = float4(dm, float(m), di, float(mi));
        out2[id] = float4(knitSDF(p), ribbons(p, HC, HV, MAIN_CHUNKS, 0.001, h2, s2),
                          ribbons(p, IC, IV, INSET_CHUNKS, 1.0, h3, s3), float(si));
    }
    """
}

// MARK: - running it

enum SockError: Error, CustomStringConvertible {
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

/// Everything one render needs.
struct Scene {
    let mutant: Mutant
    let knit: Knit
    let knitLine: [SIMD3<Float>]
    let hairs: [Hair]
    let yarn: YarnPiece
    let molecule: Molecule
    let hairPack: RibbonPack
    let insetPack: RibbonPack
    let ribbonK: Float

    init(mutant: Mutant, resources: String = "Resources") throws {
        self.mutant = mutant
        knit = buildKnit(mutant: mutant)
        knitLine = knit.polyline()
        hairs = buildHairs(knit, mutant: mutant)
        yarn = buildYarn(mutant: mutant)
        molecule = placeForInset(try buildCellulose(resources: resources, mutant: mutant))
        let reach: Float = yarn.section.reach
        hairPack = packRibbons(hairs.map { $0.ribbon }, reach: reach)
        insetPack = packRibbons(yarn.surface + [yarn.fuzz], reach: reach)
        ribbonK = ribbonBound(yarn.section, rate: mutant == .roundFibre ? 0 : twistRate)
    }

    var source: String { kernelSource(scene: self) }
}

/// The finished picture, and what each pixel's centre sample saw:
/// (view 1 main / 2 yarn inset / 3 cut inset / 4 molecule, material, sub, 0).
struct SockImage {
    let width: Int
    let height: Int
    let pixels: MTLBuffer
    let aux: MTLBuffer

    func seen(_ x: Int, _ y: Int) -> SIMD4<Float> {
        aux.contents().assumingMemoryBound(to: SIMD4<Float>.self)[y * width + x]
    }
}

func findDevice() throws -> MTLDevice {
    if let d = MTLCreateSystemDefaultDevice() ?? MTLCopyAllDevices().first { return d }
    throw SockError.noMetalDevice
}

func makeLibrary(_ device: MTLDevice, _ scene: Scene) throws -> MTLLibrary {
    let options = MTLCompileOptions()
    // Precise maths, as in every step: a 0.9 µm lumen inside distances
    // hundreds of µm long.
    options.fastMathEnabled = false
    do {
        return try device.makeLibrary(source: scene.source, options: options)
    } catch {
        throw SockError.kernelCompile("\(error)")
    }
}

func pipeline(_ device: MTLDevice, _ library: MTLLibrary, _ name: String) throws -> MTLComputePipelineState {
    guard let f = library.makeFunction(name: name) else { throw SockError.kernelCompile("no kernel \(name)") }
    do { return try device.makeComputePipelineState(function: f) } catch {
        throw SockError.kernelCompile("\(error)")
    }
}

func buffer<T>(_ device: MTLDevice, _ array: [T]) throws -> MTLBuffer {
    let length: Int = max(MemoryLayout<T>.stride * array.count, 16)
    let made: MTLBuffer? = array.withUnsafeBytes { raw -> MTLBuffer? in
        guard let base = raw.baseAddress else { return device.makeBuffer(length: length, options: .storageModeShared) }
        return device.makeBuffer(bytes: base, length: length, options: .storageModeShared)
    }
    guard let b = made else { throw SockError.gpu("could not allocate a buffer") }
    return b
}

/// The scene's buffers, in kernel argument order 3…8.
func sceneBuffers(_ device: MTLDevice, _ scene: Scene) throws -> [MTLBuffer] {
    let mol = gpuMolecule(scene.molecule)
    return [try buffer(device, scene.hairPack.chunks), try buffer(device, scene.hairPack.vertices),
            try buffer(device, scene.insetPack.chunks), try buffer(device, scene.insetPack.vertices),
            try buffer(device, mol.atoms), try buffer(device, mol.bonds)]
}

/// Render in horizontal bands, one command buffer each, so no single piece
/// of GPU work runs long enough to trip the watchdog. The GPU is shared
/// with other renders; bands keep it responsive.
func renderSock(width: Int, height: Int, samples: Int, scene: Scene, on device: MTLDevice,
                progress: Bool = false) throws -> (image: SockImage, gpuSeconds: Double) {
    let library = try makeLibrary(device, scene)
    let pso = try pipeline(device, library, "sock")
    let bufs: [MTLBuffer] = try sceneBuffers(device, scene)
    guard let pixels = device.makeBuffer(length: width * height * 4, options: .storageModeShared),
          let aux = device.makeBuffer(length: width * height * 16, options: .storageModeShared),
          let queue = device.makeCommandQueue()
    else { throw SockError.gpu("could not allocate buffers") }
    let band: Int = 8
    var gpu: Double = 0
    var row: Int = 0
    while row < height {
        let rows: Int = min(band, height - row)
        guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else {
            throw SockError.gpu("could not make a command buffer")
        }
        var params = Params(width: UInt32(width), height: UInt32(height), rowOffset: UInt32(row), samples: UInt32(samples))
        enc.setComputePipelineState(pso)
        enc.setBuffer(pixels, offset: 0, index: 0)
        enc.setBuffer(aux, offset: 0, index: 1)
        enc.setBytes(&params, length: MemoryLayout<Params>.stride, index: 2)
        for (i, b) in bufs.enumerated() { enc.setBuffer(b, offset: 0, index: 3 + i) }
        let w: Int = pso.threadExecutionWidth
        let group = MTLSize(width: w, height: max(pso.maxTotalThreadsPerThreadgroup / w / 4, 1), depth: 1)
        enc.dispatchThreads(MTLSize(width: width, height: rows, depth: 1), threadsPerThreadgroup: group)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        if let e = cb.error { throw SockError.gpu(e.localizedDescription) }
        gpu += cb.gpuEndTime - cb.gpuStartTime
        row += rows
        if progress && (row / band) % 10 == 0 {
            FileHandle.standardError.write(String(format: "\r  %3d%%", row * 100 / height).data(using: .utf8)!)
        }
    }
    if progress { FileHandle.standardError.write("\r      \r".data(using: .utf8)!) }
    return (SockImage(width: width, height: height, pixels: pixels, aux: aux), gpu)
}

struct Probe {
    var mainDistance: Float
    var mainMaterial: Int
    var insetDistance: Float
    var insetMaterial: Int
    var knit: Float
    var hairs: Float
    var insetRibbons: Float
    var insetSub: Int
}

/// The scene's distances at arbitrary points, from the same kernel source
/// the render uses — a test of the distance function is a test of the thing
/// that drew the picture. Large batches go in pieces.
func probeScene(_ points: [SIMD3<Float>], scene: Scene, library: MTLLibrary, on device: MTLDevice) throws -> [Probe] {
    let pso = try pipeline(device, library, "probe")
    let bufs: [MTLBuffer] = try sceneBuffers(device, scene)
    guard let queue = device.makeCommandQueue() else { throw SockError.gpu("no queue") }
    var out: [Probe] = []
    out.reserveCapacity(points.count)
    let batch: Int = 65_536
    var start: Int = 0
    while start < points.count {
        let pts: [SIMD4<Float>] = points[start..<min(start + batch, points.count)].map { SIMD4<Float>($0, 0) }
        let pb: MTLBuffer = try buffer(device, pts)
        guard let ob = device.makeBuffer(length: 16 * pts.count, options: .storageModeShared),
              let ob2 = device.makeBuffer(length: 16 * pts.count, options: .storageModeShared),
              let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder()
        else { throw SockError.gpu("could not set up the probe") }
        enc.setComputePipelineState(pso)
        enc.setBuffer(pb, offset: 0, index: 0)
        enc.setBuffer(ob, offset: 0, index: 1)
        enc.setBuffer(ob2, offset: 0, index: 2)
        for i in 0..<4 { enc.setBuffer(bufs[i], offset: 0, index: 3 + i) }
        enc.dispatchThreads(MTLSize(width: pts.count, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(pso.maxTotalThreadsPerThreadgroup, 256), height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        if let e = cb.error { throw SockError.gpu(e.localizedDescription) }
        let o = ob.contents().assumingMemoryBound(to: SIMD4<Float>.self)
        let o2 = ob2.contents().assumingMemoryBound(to: SIMD4<Float>.self)
        for i in 0..<pts.count {
            out.append(Probe(mainDistance: o[i].x, mainMaterial: Int(o[i].y), insetDistance: o[i].z, insetMaterial: Int(o[i].w),
                             knit: o2[i].x, hairs: o2[i].y, insetRibbons: o2[i].z, insetSub: Int(o2[i].w)))
        }
        start += batch
    }
    return out
}

func savePNG(_ image: SockImage, to url: URL) throws {
    try? FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    guard let provider = CGDataProvider(dataInfo: nil, data: image.pixels.contents(), size: image.width * image.height * 4,
                                        releaseData: { _, _, _ in }),
          let space = CGColorSpace(name: CGColorSpace.sRGB),
          let cg = CGImage(width: image.width, height: image.height, bitsPerComponent: 8, bitsPerPixel: 32,
                           bytesPerRow: image.width * 4, space: space,
                           bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                           provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent),
          let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)
    else { throw SockError.png("could not set up the encoder for \(url.path)") }
    CGImageDestinationAddImage(dest, cg, nil)
    guard CGImageDestinationFinalize(dest) else { throw SockError.png("could not write \(url.path)") }
}
