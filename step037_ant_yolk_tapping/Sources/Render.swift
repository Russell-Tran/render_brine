// Step 37: step 36's kernel, copied, with step 30's changes: the yolk's
// transform in the inset arrives per frame (buffer 8), and a frame can be
// rendered in part. New here: the odour dots arrive per frame too, in the
// same buffer after the transform, instead of as compiled-in constants.
// The molecule insets are step 36's, unchanged: nothing in them moves.
//
// One still, three scales, one kernel. Each pixel's GPU thread first asks
// which view it belongs to — the main macro view, the round micrometre inset
// or the small molecule inset — and then marches its ray into that view's
// scene. Every surface is a distance function, as in step 20; the molecule
// is spheres and cylinders hit exactly.
//
// All three cameras are ORTHOGRAPHIC. A macro lens and an SEM are both close
// to it, and it is what makes each view's scale bar true everywhere in that
// view rather than only at one depth.
//
// The light is what carries the three materials:
//
//   * ant cuticle — dark, glossy chitin, n = 1.56, so a crisp 4.8% highlight
//     of the softbox over a nearly black body;
//   * cooked yolk — matte, crumbly, pale yellow from its CIELAB colour, a
//     heap of yolk spheres each lit as a soft, slightly translucent lump;
//   * the water film on the yolk where the taste hair touches.

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

/// The main view: millimetres across the full frame width.
let mainViewWidth: Float = 8.4
/// The world point at the centre of the frame, and the view direction: from
/// the ant's right, in front and above.
let mainCentre = SIMD3<Float>(1.48, 0.76, 0.31)
let mainDirection: SIMD3<Float> = simd_normalize(SIMD3<Float>(-0.30, -0.62, -1.0))

/// The micrometre inset: a circle, and how many micrometres its diameter spans.
let insetCentre = SIMD2<Float>(1.472, 0.312)
let insetRadius: Float = 0.262
let insetField: Float = 14.5
let insetLookAt = SIMD3<Float>(1.9, 4.2, -1.0)
let insetDirection: SIMD3<Float> = simd_normalize(SIMD3<Float>(-0.10, -0.24, -1.0))

/// The molecule inset at the taste pore, in ångströms across its diameter:
/// one oleic acid, 2 nm long, and a few waters. Its bar is 0.5 nm.
let moleculeCentre = SIMD2<Float>(1.545, 0.775)
let moleculeRadius: Float = 0.176
let moleculeField: Float = 27.0
let moleculeBarNanometres: Float = 0.5

/// The second molecule inset, for the smell hair: one odorant molecule in
/// air. Smaller, above the ant's head and left of the micrometre inset, with
/// its own 0.5 nm bar.
let odourCentre = SIMD2<Float>(0.985, 0.178)
let odourRadius: Float = 0.12
let odourField: Float = 13.0
let odourBarNanometres: Float = 0.5

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
let insetCamera = OrthoCamera(centre: insetLookAt, forward: insetDirection, halfWidth: insetField / 2)

/// Main-view pixel of a world point, for a frame `width` × `height`.
func projectMain(_ p: SIMD3<Float>, width: Int, height: Int) -> SIMD2<Float> {
    mainCamera.project(p, viewCentre: SIMD2(Float(width) / 2, Float(height) / 2), viewHalf: Float(width) / 2)
}

/// Inset pixel of a micrometre-space point.
func projectInset(_ p: SIMD3<Float>, height: Int) -> SIMD2<Float> {
    let h: Float = Float(height)
    return insetCamera.project(p, viewCentre: insetCentre * h, viewHalf: insetRadius * h)
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

/// The crumb for the kernel: the core ellipsoid (centre, smallest semi-axis;
/// then its semi-axes), then every yolk sphere (centre, radius).
func gpuFood(_ spheres: [YolkSphere], core: (centre: SIMD3<Float>, semi: SIMD3<Float>)) -> [SIMD4<Float>] {
    [SIMD4<Float>(core.centre, min(core.semi.x, min(core.semi.y, core.semi.z))), SIMD4<Float>(core.semi, 0)]
        + spheres.map { SIMD4<Float>($0.centre, $0.radius) }
}

/// A sphere holding the whole crumb and its bits, so rays far away skip it.
func foodBound(_ spheres: [YolkSphere]) -> (centre: SIMD3<Float>, radius: Float) {
    var c = SIMD3<Float>(0, 0, 0)
    for s in spheres { c += s.centre }
    c /= Float(max(spheres.count, 1))
    let r: Float = spheres.map { simd_distance($0.centre, c) + $0.radius }.max() ?? 0
    return (c, r + 0.01)
}

/// Hairs flattened: six float4s each.
func gpuHairs(_ hairs: [Sensillum]) -> [SIMD4<Float>] {
    var out: [SIMD4<Float>] = []
    for h in hairs {
        let u: SIMD3<Float> = simd_normalize(h.tip - h.base)
        let helper: SIMD3<Float> = abs(u.y) < 0.9 ? SIMD3(0, 1, 0) : SIMD3(1, 0, 0)
        let e1: SIMD3<Float> = simd_normalize(simd_cross(u, helper))
        let e2: SIMD3<Float> = simd_cross(u, e1)
        out.append(SIMD4<Float>(h.base, h.baseRadius))
        out.append(SIMD4<Float>(h.tip, h.tipRadius))
        out.append(SIMD4<Float>(h.tipPoreRadius, h.wallPoreRadius, h.poreSpacing, h.lumenFraction))
        out.append(SIMD4<Float>(h.poreStart, h.poreEnd, Float(h.poreColumns), h.socket ? 1 : 0))
        out.append(SIMD4<Float>(e1, Float(h.poreRows)))
        out.append(SIMD4<Float>(e2, 0))
    }
    return out
}

/// Element codes for the kernel: 0 C, 1 O, 2 H.
func elementCode(_ e: String) -> Float {
    switch e {
    case "C": return 0
    case "O": return 1
    default: return 2
    }
}

/// Atoms (position Å, element code + 10 × which inset) and bonds (atom
/// indices). The atoms are already placed in their inset's own coordinates.
func gpuMolecule(_ m: Molecule) -> (atoms: [SIMD4<Float>], bonds: [SIMD4<Float>]) {
    let atoms: [SIMD4<Float>] = m.atoms.map { SIMD4<Float>($0.position, elementCode($0.element) + 10 * Float($0.view)) }
    let bonds: [SIMD4<Float>] = m.bonds.map { SIMD4<Float>(Float($0.0), Float($0.1), 0, 0) }
    return (atoms, bonds)
}

// MARK: - the kernel

/// How far a ray may trust a distance. The ellipsoid distances are
/// approximations; the tests measure how much they over-report.
let stepScale: Float = 0.75

func metal(_ v: SIMD3<Float>) -> String { "float3(\(v.x), \(v.y), \(v.z))" }
func metal2(_ v: SIMD2<Float>) -> String { "float2(\(v.x), \(v.y))" }

func metalMatrix(_ q: simd_quatf) -> String {
    let m = simd_float3x3(q)
    return "float3x3(\(metal(m.columns.0)), \(metal(m.columns.1)), \(metal(m.columns.2)))"
}

func kernelSource(shapeCount: Int, sphereCount: Int, bound: (centre: SIMD3<Float>, radius: Float), atomCount: Int, bondCount: Int, volatileSlots: Int,
                  dome: (centre: SIMD3<Float>, radius: Float)) -> String {
    let mc: OrthoCamera = mainCamera
    let ic: OrthoCamera = insetCamera
    let moleculeForward = SIMD3<Float>(0, 0, -1)
    return """
    #include <metal_stdlib>
    using namespace metal;

    struct Shape { float4 a; float4 b; float4 x; float4 y; float4 meta; float4 bound; };
    struct Params { uint width; uint height; uint rowOffset; uint samples; uint colOffset; uint colEnd; uint rowEnd; uint pad; };

    constant int SHAPES = \(shapeCount);
    constant int ATOMS = \(atomCount);
    constant int BONDS = \(bondCount);
    constant int VOLS = \(volatileSlots);
    constant float STEP_SCALE = \(stepScale);

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

    constant float2 Q_C = \(metal2(moleculeCentre));
    constant float Q_R = \(moleculeRadius);
    constant float Q_HALF = \(moleculeField / 2);
    constant float3 Q_FWD = \(metal(moleculeForward));

    constant float3 KEY_DIR = \(metal(keyDirection));
    constant float3 FILL_DIR = \(metal(fillDirection));
    constant float3 KEY_COL = \(metal(keyColour));
    constant float3 FILL_COL = \(metal(fillColour));
    constant float3 KEY_RAD = \(metal(discRadiance(keyColour, cosine: keyDiscCosine)));
    constant float3 FILL_RAD = \(metal(discRadiance(fillColour, cosine: fillDiscCosine)));
    constant float KEY_DISC = \(keyDiscCosine);
    constant float FILL_DISC = \(fillDiscCosine);

    constant float TABLE = \(tableAlbedo);
    constant float3 GASTER_C = \(metal(gasterCentre));
    constant float3 GASTER_X = \(metal(gasterAxis));
    constant float GASTER_A = \(gasterSemiAxes.x);
    constant float CUTICLE_F0 = \(cuticleF0);
    constant int SPHERES = \(sphereCount);
    constant float3 BOUND_C = \(metal(bound.centre));
    constant float BOUND_R = \(bound.radius);
    constant float CRUMB_BLEND = \(crumbBlend);
    constant float3 YOLK = \(metal(yolkAlbedo));
    constant float WATER_F0 = \(waterF0);
    constant float WATER_N = \(Float(waterIndex));

    constant float FILM_H = \(filmThickness);
    constant float FILM_R = \(filmRadius);
    constant float MENISCUS_H = \(meniscusHeight);
    constant float3 DOME_C = \(metal(dome.centre));
    constant float DOME_R = \(dome.radius);
    constant float2 O_C = \(metal2(odourCentre));
    constant float O_R = \(odourRadius);
    constant float O_HALF = \(odourField / 2);

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
    constant float3 HAIR = float3(0.20, 0.125, 0.065);
    constant float3 ODOUR = float3(0.95, 0.55, 0.08);

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

    // ------------------------------------------------------------ the ant (mm)

    // Three blend groups: the mesosoma's three parts melt together broadly,
    // the head and mouthparts moderately, everything else barely — enough
    // that a joint is a joint, not a seam. Round cones are exact, so one far
    // from the current best (by more than its blend radius) cannot change the
    // answer and is skipped; the ellipsoids are approximate and never skipped.
    float antSDF(float3 p, constant Shape *S, thread int &part, thread int &light) {
        float g0 = 1e9, g1 = 1e9, g2 = 1e9, g3 = 1e9;
        float best = 1e9;
        part = 0;
        light = 0;
        for (int i = 0; i < SHAPES; i++) {
            Shape s = S[i];
            int group = int(s.meta.w + 0.5);
            float k = group == 1 ? 0.10 : (group == 2 ? 0.04 : (group == 3 ? 0.04 : 0.012));
            float cur = group == 1 ? g1 : (group == 2 ? g2 : (group == 3 ? g3 : g0));
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
            if (d < best) { best = d; part = int(s.meta.y + 0.5); light = int(s.meta.z + 0.5); }
            if (group == 1) g1 = smin(g1, d, k); else if (group == 2) g2 = smin(g2, d, k); else if (group == 3) g3 = smin(g3, d, k); else g0 = smin(g0, d, k);
        }
        return smin(smin(min(g1, g3), g2, 0.03), g0, 0.012);
    }

    // ------------------------------------------------------------ the yolk (mm)

    // The crumb: a core ellipsoid (bounded, never over-reporting) blended
    // with every yolk sphere, cut by the card. Far away (beyond anything
    // the occlusion or shadow tests probe), the bounding sphere's distance
    // stands in — a lower bound, so still safe, and too far to leave rings.
    float foodSDF(float3 p, constant float4 *G) {
        float far = length(p - BOUND_C) - BOUND_R;
        if (far > 1.6) return far;
        float3 q = (p - G[0].xyz) / G[1].xyz;
        float d = (length(q) - 1.0) * G[0].w;
        for (int i = 0; i < SPHERES; i++) {
            float4 s = G[2 + i];
            d = smin(d, length(p - s.xyz) - s.w, CRUMB_BLEND);
        }
        return max(d, -p.y);
    }

    // Materials: 1 table, 2 ant, 3 yolk.
    float mainSDF(float3 p, constant Shape *S, constant float4 *G, thread int &mat, thread int &sub, thread int &light) {
        float d = p.y;
        mat = 1; sub = 0; light = 0;
        int part, lt;
        float a = antSDF(p, S, part, lt);
        if (a < d) { d = a; mat = 2; sub = part; light = lt; }
        float g = foodSDF(p, G);
        if (g < d) { d = g; mat = 3; sub = 0; light = 0; }
        return d;
    }

    float mainDist(float3 p, constant Shape *S, constant float4 *G) {
        int m, s, l;
        return mainSDF(p, S, G, m, s, l);
    }

    float3 mainNormal(float3 p, constant Shape *S, constant float4 *G) {
        const float e = 0.0015;
        float3 k1 = float3(1, -1, -1), k2 = float3(-1, -1, 1), k3 = float3(-1, 1, -1), k4 = float3(1, 1, 1);
        return normalize(k1 * mainDist(p + k1 * e, S, G) + k2 * mainDist(p + k2 * e, S, G)
                       + k3 * mainDist(p + k3 * e, S, G) + k4 * mainDist(p + k4 * e, S, G));
    }

    bool marchMain(float3 ro, float3 rd, constant Shape *S, constant float4 *G,
                   thread float &t, thread int &mat, thread int &sub, thread int &light) {
        t = 0.0;
        for (int i = 0; i < 320; i++) {
            float3 p = ro + rd * t;
            float d = mainSDF(p, S, G, mat, sub, light);
            if (d < 0.0006) return true;
            t += max(d * STEP_SCALE, 0.0003);
            if (t > 40.0) break;
        }
        return false;
    }

    // Shadows: the ant and the yolk both block the light.
    float softShadow(float3 ro, float3 rd, constant Shape *S, constant float4 *G) {
        float res = 1.0;
        float t = 0.004;
        for (int i = 0; i < 100; i++) {
            float3 p = ro + rd * t;
            int part, lt;
            float a = min(antSDF(p, S, part, lt), foodSDF(p, G));
            res = min(res, 8.0 * a / t);
            t += clamp(a * STEP_SCALE, 0.002, 0.2);
            if (res < 0.002 || t > 6.0) break;
        }
        return clamp(res, 0.0, 1.0);
    }

    float occlusion(float3 p, float3 n, constant Shape *S, constant float4 *G, float scale) {
        float occ = 0.0, w = 1.0;
        for (int i = 1; i <= 5; i++) {
            float h = scale * (0.02 + 0.06 * float(i * i));
            occ += w * clamp((h - mainDist(p + n * h, S, G)) / h, 0.0, 1.0);
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

    float3 shadeTable(float3 p, constant Shape *S, constant float4 *G) {
        float sh = softShadow(p + float3(0, 0.002, 0), KEY_DIR, S, G);
        float occ = occlusion(p, float3(0, 1, 0), S, G, 1.0);
        float grain = 0.97 + 0.03 * noise3(p * 60.0);
        float3 alb = float3(TABLE) * grain * float3(1.0, 0.99, 0.975);
        return alb * (KEY_COL * max(KEY_DIR.y, 0.0) * sh + FILL_COL * max(FILL_DIR.y, 0.0) + irradiance(float3(0, 1, 0)) * occ * 0.55);
    }

    float3 shadeAnt(float3 p, float3 rd, int part, int light, constant Shape *S, constant float4 *G) {
        float3 n = mainNormal(p, S, G);
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
        if (part == 4) {
            // Tergite margins: where each plate's hind edge laps the next, a
            // narrow step — darker in the crease, a glint on the lip.
            float x = dot(p - GASTER_C, GASTER_X) / GASTER_A;      // -1 tail … +1 front
            float m = 0.0;
            for (int i = 0; i < 3; i++) {
                float at = 0.42 - 0.40 * float(i);
                m = max(m, exp(-pow((x - at) / 0.022, 2.0)));
            }
            alb *= 1.0 - 0.45 * m;
            n = normalize(n + GASTER_X * 0.35 * m);
        }
        float sh = softShadow(p + n * 0.003, KEY_DIR, S, G);
        float occ = occlusion(p, n, S, G, 0.6);
        return shadeCuticle(n, v, alb, sh, occ, alpha, CUTICLE_F0, false);
    }

    // Cooked yolk: matte and crumbly. Each yolk sphere's surface is rough
    // at a finer scale still (noise bump), the light wraps a little round
    // its edges as in any soft, fatty, translucent-ish solid, and the
    // creases between spheres fall into shadow by the occlusion term.
    float3 shadeFood(float3 p, float3 rd, constant Shape *S, constant float4 *G) {
        float3 n = mainNormal(p, S, G);
        float3 bump = noiseGrad(p * 160.0) + 0.6 * noiseGrad(p * 520.0);
        n = normalize(n + 0.16 * (bump - n * dot(bump, n)));
        float3 v = -rd;
        float sh = softShadow(p + n * 0.003, KEY_DIR, S, G);
        float occ = occlusion(p, n, S, G, 0.35);
        float wrap = 0.25;
        float key = max((dot(n, KEY_DIR) + wrap) / (1.0 + wrap), 0.0) * sh;
        float fill = max(dot(n, FILL_DIR), 0.0);
        float speck = 0.92 + 0.16 * noise3(p * 90.0);
        float3 alb = YOLK * speck;
        float3 col = alb * (KEY_COL * key + FILL_COL * fill + irradiance(n) * occ * 1.1) * mix(0.55, 1.0, occ);
        // A faint, broad sheen from the fat — nothing like a gloss.
        col += KEY_COL * ggx(n, v, KEY_DIR, 0.55, 0.03) * sh * 0.25;
        return col;
    }

    float3 shadeMain(float3 ro, float3 rd, constant Shape *S, constant float4 *G, thread float4 &aux) {
        float t;
        int mat, sub, light;
        aux = float4(0.0);
        if (!marchMain(ro, rd, S, G, t, mat, sub, light)) return environment(rd);
        float3 p = ro + rd * t;
        aux = float4(1.0, float(mat), float(sub), 0.0);
        if (mat == 1) return shadeTable(p, S, G);
        if (mat == 2) return shadeAnt(p, rd, sub, light, S, G);
        return shadeFood(p, rd, S, G);
    }

    // ------------------------------------------------------------ the inset (µm)

    // A sense hair: a hollow, tapered cuticle tube with a rounded apex,
    // drilled by its pores. Far from its surface the pores cannot matter, so
    // the plain outer distance comes back exactly.
    float hairSDF(float3 p, constant float4 *H, int h) {
        float3 a = H[h * 6 + 0].xyz;  float ra = H[h * 6 + 0].w;
        float3 b = H[h * 6 + 1].xyz;  float rb = H[h * 6 + 1].w;
        float4 pore = H[h * 6 + 2];
        float4 range = H[h * 6 + 3];
        float3 e1 = H[h * 6 + 4].xyz; float rows = H[h * 6 + 4].w;
        float3 e2 = H[h * 6 + 5].xyz;
        float outer = roundCone(p, a, b, ra, rb);
        if (outer > 0.25) return outer;
        float3 axis = b - a;
        float L = length(axis);
        float3 u = axis / L;
        float lf = pore.w;
        float inner = roundCone(p, a - u * 2.0, b, ra * lf, rb * lf);
        float d = max(outer, -inner);
        float t = dot(p - a, u);
        float3 radial = (p - a) - u * t;
        // One pore at the apex, opening the tip into the lumen.
        if (pore.x > 0.0) {
            float cut = max(length(radial) - pore.x, L - t);
            d = max(d, -cut);
        }
        // Wall pores on a staggered lattice, rows along the hair, columns round it.
        if (pore.y > 0.0) {
            float sp = pore.z;
            float t0 = range.x * L;
            float cols = range.z;
            float i = clamp(floor((t - t0) / sp + 0.5), 0.0, rows - 1.0);
            float tc = t0 + i * sp;
            float phi = atan2(dot(radial, e2), dot(radial, e1));
            float stagger = fmod(i, 2.0) * 0.5;
            float cell = 6.2831853 / cols;
            float j = floor(phi / cell - stagger + 0.5);
            float dphi = phi - (j + stagger) * cell;
            // Distance to the pore's own axis, a radial line: exact, so the
            // carve never over-reports. (Measuring the arc at the outer
            // radius instead read 1.7× down in the pores, where the wall is
            // nearer the axis.)
            float arc = length(radial) * sin(clamp(dphi, -1.5, 1.5));
            float dt = t - tc;
            float hole = sqrt(dt * dt + arc * arc) - pore.y;
            d = max(d, -hole);
        }
        return d;
    }

    // The flexible socket a chaeticum stands in: a raised collar.
    float socketSDF(float3 p, constant float4 *H, int h) {
        if (H[h * 6 + 3].w < 0.5) return 1e9;
        float3 a = H[h * 6 + 0].xyz;
        float ra = H[h * 6 + 0].w;
        float3 u = normalize(H[h * 6 + 1].xyz - a);
        float3 q = p - (a + u * 0.9);
        float y = dot(q, u);
        float r = length(q - u * y);
        float2 tq = float2(r - (ra + 0.45), y);
        return length(tq) - 0.42;
    }

    // The yolk's surface at this scale: the skin of one yolk sphere, studded
    // with yolk granules 0.3–2 µm across (Anton 2013) — here 0.5–1.0 µm
    // bumps on a jittered 1.1 µm grid over a floor. The granule under the
    // origin is placed so its top is exactly y = 0, where the hair touches;
    // every other granule tops out lower.
    float granule(float2 cell, thread float3 &c) {
        float h1 = fract(sin(dot(cell, float2(127.1, 311.7))) * 43758.5453);
        float h2 = fract(sin(dot(cell, float2(269.5, 183.3))) * 43758.5453);
        float h3 = fract(sin(dot(cell, float2(419.2, 371.9))) * 43758.5453);
        float h4 = fract(sin(dot(cell, float2(83.7, 529.1))) * 43758.5453);
        float r = 0.22 + 0.30 * h3;
        c = float3((cell.x + (h1 - 0.5) * 0.45) * 1.1, -r - 0.05 - 0.22 * h4, (cell.y + (h2 - 0.5) * 0.45) * 1.1);
        if (cell.x == 0.0 && cell.y == 0.0) { r = 0.45; c = float3(0.0, -r, 0.0); }
        return r;
    }
    // Step 37: far above the granules the field hands over, smoothly, to the
    // height above y = 0 (every granule top is at or below it), less a little
    // for the blends. Step 36's field looks only at the 3 × 3 cells round a
    // point; high up, many granules are nearly equally far, and a cell seam
    // that drops one of them out of the blend jumped the distance by up to
    // 0.03 µm — an over-report on a 0.02 µm step that the lifted frames, with
    // the yolk 70 µm below, turned up. Below 1.5 µm (the surface, its film,
    // anything the contact frame's rays and shadows lean on) it is step 36's.
    float crystalSDF(float3 p) {
        float far = p.y - 0.03;
        if (p.y > 5.5) return far;
        float d = p.y + 0.30;
        float2 g = floor(p.xz / 1.1 + 0.5);
        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                float3 c;
                float r = granule(g + float2(i, j), c);
                d = smin(d, length(p - c) - r, 0.12);
            }
        }
        return mix(d, far, smoothstep(1.5, 5.5, p.y));
    }

    // A thin lens of water on the face, and a meniscus climbing the hair:
    // the surface height rises towards the hair as exp(−distance / 0.3 µm).
    // Built in real micrometres, then divided once by its steepest slope,
    // √(1 + (MENISCUS_H / 0.3)²), so a ray can trust it.
    float filmSDF(float3 p, float dHair) {
        // A shell of water following the yolk's bumpy surface, thickest at
        // the contact and thinning to nothing at FILM_R, plus the meniscus
        // climbing the hair. The yolk distance changes at most one per unit,
        // the added height at most MENISCUS_H / 0.3, so dividing by their
        // sum keeps it a safe bound.
        float r = length(p.xz);
        float base = crystalSDF(p);
        float h = FILM_H * (1.0 - smoothstep(0.55 * FILM_R, FILM_R, r)) + 0.004;
        h += MENISCUS_H * exp(-max(dHair, 0.0) / 0.3) * (1.0 - smoothstep(0.0, 1.2, p.y));
        float lip = MENISCUS_H / 0.3;
        float d = (base - h) / (1.0 + lip);
        d = max(d, -base);
        return max(d, r - FILM_R);
    }

    // Materials: 1 yolk, 2 antenna cuticle, 3 taste hair, 4 smell hair, 5 water film, 6 odour dot.
    // Step 37, as step 30: the inset rides on the antenna tip, so the yolk
    // and its film are found through the frame's transform: inset point p
    // lies over yolk point R p + T (FR[0..2] the columns of R, FR[3] T, µm).
    // The hairs, the dome and the odour dots are the inset's own: untouched.
    float3 toCrystal(float3 p, constant float4 *FR) {
        return float3x3(FR[0].xyz, FR[1].xyz, FR[2].xyz) * p + FR[3].xyz;
    }

    float insetSDF(float3 p, constant float4 *H, constant float4 *FR, bool film, thread int &mat) {
        float3 pc = toCrystal(p, FR);
        float d = crystalSDF(pc);
        mat = 1;
        float dome = length(p - DOME_C) - DOME_R;
        float sock = socketSDF(p, H, 0);
        dome = smin(dome, sock, 0.25);
        if (dome < d) { d = dome; mat = 2; }
        float t0 = hairSDF(p, H, 0);
        if (t0 < d) { d = t0; mat = 3; }
        float t1 = hairSDF(p, H, 1);
        if (t1 < d) { d = t1; mat = 4; }
        if (film) {
            // The film is water around the hair, never inside it.
            float f = max(filmSDF(pc, t0), -t0);
            if (f < d) { d = f; mat = 5; }
        }
        // Odour molecules in the air, as dots: step 37 sends them per frame,
        // after the transform; a slot of radius 0 holds none.
        for (int v = 0; v < VOLS; v++) {
            float4 o = FR[4 + v];
            if (o.w <= 0.0) continue;
            float e = length(p - o.xyz) - o.w;
            if (e < d) { d = e; mat = 6; }
        }
        return d;
    }

    float insetDist(float3 p, constant float4 *H, constant float4 *FR) { int m; return insetSDF(p, H, FR, false, m); }

    float3 insetNormal(float3 p, constant float4 *H, constant float4 *FR, bool film) {
        const float e = 0.004;
        int m;
        float3 k1 = float3(1, -1, -1), k2 = float3(-1, -1, 1), k3 = float3(-1, 1, -1), k4 = float3(1, 1, 1);
        return normalize(k1 * insetSDF(p + k1 * e, H, FR, film, m) + k2 * insetSDF(p + k2 * e, H, FR, film, m)
                       + k3 * insetSDF(p + k3 * e, H, FR, film, m) + k4 * insetSDF(p + k4 * e, H, FR, film, m));
    }

    bool marchInset(float3 ro, float3 rd, constant float4 *H, constant float4 *FR, bool film, thread float &t, thread int &mat) {
        t = 0.0;
        for (int i = 0; i < 400; i++) {
            float d = insetSDF(ro + rd * t, H, FR, film, mat);
            if (d < 0.002) return true;
            t += max(d * STEP_SCALE, 0.001);
            if (t > 120.0) break;
        }
        return false;
    }

    float insetShadow(float3 ro, float3 rd, constant float4 *H, constant float4 *FR) {
        float res = 1.0;
        float t = 0.02;
        for (int i = 0; i < 80; i++) {
            float h = insetDist(ro + rd * t, H, FR);
            res = min(res, 10.0 * h / t);
            t += clamp(h * STEP_SCALE, 0.01, 1.5);
            if (res < 0.002 || t > 40.0) break;
        }
        return clamp(res, 0.0, 1.0);
    }

    float insetOcclusion(float3 p, float3 n, constant float4 *H, constant float4 *FR) {
        float occ = 0.0, w = 1.0;
        for (int i = 1; i <= 5; i++) {
            float h = 0.03 + 0.05 * float(i * i);
            occ += w * clamp((h - insetDist(p + n * h, H, FR)) / h, 0.0, 1.0);
            w *= 0.7;
        }
        return clamp(1.0 - 0.55 * occ, 0.0, 1.0);
    }

    // Yolk at this scale: matte, pale yellow, soft; wet under the film,
    // darker and a touch more saturated, as wet solids are.
    float3 shadeYolk(float3 p, float3 rd, float3 n, float sh, float occ, float wet) {
        float3 alb = YOLK * mix(1.0, 0.72, wet) * (0.94 + 0.12 * noise3(p * 3.0));
        alb = mix(alb, alb * alb / max(dot(alb, float3(0.333)), 1e-3) * 0.9, 0.25 * wet);
        float wrap = 0.3;
        float key = max((dot(n, KEY_DIR) + wrap) / (1.0 + wrap), 0.0) * sh;
        return alb * (KEY_COL * key * 0.85 + irradiance(n) * occ * 1.35);
    }

    float3 shadeInsetSolid(float3 p, float3 rd, int mat, constant float4 *H, constant float4 *FR, float wet) {
        float3 n = insetNormal(p, H, FR, false);
        float3 v = -rd;
        if (mat == 6) {
            // An odour dot: a small bright bead, lit from the key.
            return ODOUR * (0.55 + 0.7 * max(dot(n, KEY_DIR), 0.0)) + float3(0.25) * pow(max(dot(n, normalize(v + KEY_DIR)), 0.0), 40.0);
        }
        float sh = insetShadow(p + n * 0.01, KEY_DIR, H, FR);
        float occ = insetOcclusion(p, n, H, FR);
        if (mat == 1) return shadeYolk(p, rd, n, sh, occ, wet);
        float3 alb = mat == 2 ? DARK : HAIR;
        if (mat == 2) {
            float3 bump = noiseGrad(p * 1.3);
            n = normalize(n + 0.05 * (bump - n * dot(bump, n)));
        }
        // Inside a hair — seen only down a pore — is the lumen, full of
        // sensillum lymph: dark.
        if (mat >= 3) {
            int h = mat - 3;
            float outer = roundCone(p, H[h * 6].xyz, H[h * 6 + 1].xyz, H[h * 6].w, H[h * 6 + 1].w);
            if (outer < -0.015) return HAIR * 0.08 * occ;
        }
        if (mat == 3) {
            // A chaeticum's wall is fluted: fine ridges running its length.
            float3 a = H[0].xyz;
            float3 u = normalize(H[1].xyz - a);
            float3 e1 = H[4].xyz, e2 = H[5].xyz;
            float3 rad = (p - a) - u * dot(p - a, u);
            float phi = atan2(dot(rad, e2), dot(rad, e1));
            float3 tang = normalize(cross(u, rad));
            n = normalize(n + tang * 0.22 * sin(phi * 14.0));
        }
        // Thin hair cuticle passes some light: a little warm translucency.
        float3 col = shadeCuticle(n, v, alb, sh, occ, 0.22, CUTICLE_F0, true);
        if (mat >= 3) col += HAIR * float3(1.0, 0.7, 0.45) * KEY_COL * 0.10 * max(-dot(n, KEY_DIR) + 0.3, 0.0);
        return col;
    }

    float3 shadeInset(float3 ro, float3 rd, constant float4 *H, constant float4 *FR, thread float4 &aux) {
        float t;
        int mat;
        aux = float4(2.0, 0.0, 0.0, 0.0);
        if (!marchInset(ro, rd, H, FR, true, t, mat)) return envInset(rd);
        float3 p = ro + rd * t;
        aux.y = float(mat);
        if (mat != 5) return shadeInsetSolid(p, rd, mat, H, FR, 0.0);
        // The water film: 2% reflection off its top, the rest refracted in,
        // lensed slightly by its curved surface, onto what lies beneath.
        float3 n = insetNormal(p, H, FR, true);
        float c = max(dot(n, -rd), 0.0);
        float F = schlick(WATER_F0, c);
        float3 refl = envInset(reflect(rd, n));
        float3 dir = refract(rd, n, 1.0 / WATER_N);
        float t2;
        int m2;
        float3 below = float3(0.8);
        float3 q = p + dir * 0.004;
        if (marchInset(q, dir, H, FR, false, t2, m2)) {
            below = shadeInsetSolid(q + dir * t2, dir, m2, H, FR, 1.0);
            aux.z = float(m2);
        }
        return refl * F + below * (1.0 - F) * float3(0.92, 0.96, 1.0);
    }

    // ------------------------------------------------------------ the molecule (Å)

    float3 atomColour(float code) {
        // CPK: carbon grey, oxygen red, hydrogen white.
        return code < 0.5 ? float3(0.30, 0.30, 0.32) : (code < 1.5 ? float3(0.80, 0.07, 0.05) : float3(0.92, 0.92, 0.92));
    }
    float atomRadius(float code) { return code < 0.5 ? 0.38 : (code < 1.5 ? 0.36 : 0.24); }

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

    float3 shadeMolecule(float3 ro, float3 rd, constant float4 *A, constant float4 *B, float2 uv, int view, thread float4 &aux) {
        float best = 1e9;
        float3 n = float3(0.0);
        float3 col = float3(0.0);
        float glow = 0.0;
        aux = float4(3.0 + float(view), 0.0, 0.0, 0.0);
        for (int i = 0; i < ATOMS; i++) {
            if (int(A[i].w / 10.0) != view) continue;
            float3 c = A[i].xyz;
            float code = fmod(A[i].w, 10.0);
            float r = atomRadius(code);
            float t;
            if (sphereHit(ro, rd, c, r, t) && t < best) {
                best = t; n = normalize(ro + rd * t - c); col = atomColour(code);
                aux.y = 1.0 + code;
            }
            // A soft halo behind each atom, only where the ray misses it.
            float3 oc = c - ro;
            float along = dot(oc, rd);
            float miss = length(oc - rd * along);
            glow += exp(-pow(max(miss - r, 0.0) / 0.35, 2.0)) * 0.10;
        }
        for (int i = 0; i < BONDS; i++) {
            int ia = int(B[i].x + 0.5), ib = int(B[i].y + 0.5);
            if (int(A[ia].w / 10.0) != view) continue;
            float3 a = A[ia].xyz, b = A[ib].xyz;
            float t;
            float3 cn;
            if (cylinderHit(ro, rd, a, b, 0.11, t, cn) && t < best) {
                best = t; n = cn;
                float y = dot(ro + rd * t - a, b - a) / dot(b - a, b - a);
                col = mix(atomColour(fmod(A[ia].w, 10.0)), atomColour(fmod(A[ib].w, 10.0)), step(0.5, y)) * 0.9;
                aux.y = 9.0;
            }
        }
        // The taste inset's ground is the moisture film, pale blue; the
        // smell inset's is air, pale grey.
        float3 bg = view == 0 ? mix(float3(0.80, 0.88, 0.93), float3(0.90, 0.94, 0.97), 0.5 + 0.5 * uv.y)
                              : mix(float3(0.88, 0.89, 0.91), float3(0.95, 0.95, 0.96), 0.5 + 0.5 * uv.y);
        if (best > 1e8) return bg * (1.0 - min(glow, 0.35)) + float3(0.02, 0.05, 0.07) * min(glow, 0.35);
        float3 v = -rd;
        float key = max(dot(n, KEY_DIR), 0.0);
        float fill = max(dot(n, FILL_DIR), 0.0);
        float3 h = normalize(v + KEY_DIR);
        float spec = pow(max(dot(n, h), 0.0), 60.0) * 0.5;
        return col * (0.35 + 0.75 * key + 0.35 * fill) + float3(spec);
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
                    constant float4 *H [[buffer(5)]],
                    constant float4 *A [[buffer(6)]],
                    constant float4 *B [[buffer(7)]],
                    constant float4 *FR [[buffer(8)]],
                    uint2 gid [[thread_position_in_grid]]) {
        uint x = gid.x + P.colOffset;
        uint y = gid.y + P.rowOffset;
        if (x >= P.colEnd || y >= P.rowEnd) return;
        float W = float(P.width), Hh = float(P.height);
        float3 sum = float3(0.0);
        uint N = P.samples;
        float4 aux = float4(0.0);
        for (uint sy = 0; sy < N; sy++) {
            for (uint sx = 0; sx < N; sx++) {
                float2 px = float2(x, y) + float2((float(sx) + 0.5) / float(N), (float(sy) + 0.5) / float(N));
                float2 f = px / Hh;                  // frame-height units
                float4 a;
                float3 c;
                if (length(f - I_C) < I_R) {
                    float2 s = (f - I_C) / I_R;
                    float3 ro = I_CENTRE + (I_RIGHT * s.x - I_UP * s.y) * I_HALF - I_FWD * 60.0;
                    c = shadeInset(ro, I_FWD, H, FR, a);
                } else if (length(f - Q_C) < Q_R) {
                    float2 s = (f - Q_C) / Q_R;
                    float3 ro = float3(s.x * Q_HALF, -s.y * Q_HALF, 30.0);
                    c = shadeMolecule(ro, Q_FWD, A, B, s, 0, a);
                } else if (length(f - O_C) < O_R) {
                    float2 s = (f - O_C) / O_R;
                    float3 ro = float3(s.x * O_HALF, -s.y * O_HALF, 30.0);
                    c = shadeMolecule(ro, Q_FWD, A, B, s, 1, a);
                } else {
                    float2 s = float2(2.0 * px.x / W - 1.0, (1.0 - 2.0 * px.y / Hh) * (Hh / W));
                    float3 ro = M_CENTRE + (M_RIGHT * s.x + M_UP * s.y) * M_HALF - M_FWD * 20.0;
                    c = shadeMain(ro, M_FWD, S, G, a);
                }
                sum += toneMap(c);
                if (sx == N / 2 && sy == N / 2) aux = a;
            }
        }
        float3 c = sum / float(N * N);
        pixels[y * P.width + x] = uchar4(uchar3(round(clamp(c, 0.0, 1.0) * 255.0)), 255);
        auxOut[y * P.width + x] = aux;
    }

    // The distance functions at arbitrary points, from the same source the
    // picture is drawn with: (main distance, main material, inset distance,
    // inset material), plus each hair alone and the yolk alone.
    kernel void probe(device const float4 *points [[buffer(0)]],
                      device float4 *out [[buffer(1)]],
                      device float4 *out2 [[buffer(2)]],
                      constant Shape *S [[buffer(3)]],
                      constant float4 *G [[buffer(4)]],
                      constant float4 *H [[buffer(5)]],
                      constant float4 *FR [[buffer(6)]],
                      uint id [[thread_position_in_grid]]) {
        float3 p = points[id].xyz;
        int m, s, l, mi, part, lt;
        float dm = mainSDF(p, S, G, m, s, l);
        float di = insetSDF(p, H, FR, true, mi);
        out[id] = float4(dm, float(m), di, float(mi));
        out2[id] = float4(hairSDF(p, H, 0), hairSDF(p, H, 1), foodSDF(p, G), antSDF(p, S, part, lt));
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
    var pad: UInt32 = 0
}


/// What never moves, built once from the anatomy.
struct Scene {
    let mutant: Mutant
    let spheres: [YolkSphere]       // the crumb's and the bits'
    let core: (centre: SIMD3<Float>, semi: SIMD3<Float>)
    let hairs: [Sensillum]
    let chemistry: YolkChemistry
    /// What both molecule insets draw, each in its own Å. Step 37: nothing
    /// in them moves — step 36's molecules, as they were.
    let molecule: Molecule
    let dome: (centre: SIMD3<Float>, radius: Float)
    /// The antenna as it rests on the yolk — step 36's pose.
    let contactAnt: AntModel
    let contact: (point: SIMD3<Float>, normal: SIMD3<Float>)
    let shapeCount: Int

    init(mutant: Mutant, resources: String = "Resources") throws {
        self.mutant = mutant
        let crumb = buildCrumb()
        spheres = crumb.spheres + buildBits()
        core = crumb.core
        hairs = buildHairs(mutant: mutant)
        chemistry = try buildChemistry(resources: resources, mutant: mutant)
        molecule = chemistry.all
        dome = antennaDome(hairs)
        contactAnt = buildAnt(mutant: .none)
        contact = contactPoint()
        shapeCount = buildAnt(mutant: mutant).shapes.count
    }

    var source: String {
        kernelSource(shapeCount: shapeCount, sphereCount: spheres.count, bound: foodBound(spheres),
                     atomCount: molecule.atoms.count, bondCount: molecule.bonds.count, volatileSlots: odourSlots, dome: dome)
    }

    /// The odour dots at the moment the loop starts: step 36's `volatiles`.
    var volatiles: [Volatile] { frame(at: 0).volatiles }

    /// Everything that moves, at time `t` seconds into the loop.
    func frame(at t: Float) -> FrameState {
        let phase: Float = tapPhase(t, mutant: mutant).phase
        let lift: Float = liftAt(phase: phase)
        let ant: AntModel = buildAnt(mutant: mutant, lift: lift, sway: swayAt(t, mutant: mutant))
        let cf: CrystalFrame = crystalFrame(contact: contactAnt.antennae[0], now: ant.antennae[0],
                                            contactPoint: contact.point, normal: contact.normal)
        let u: Float = odourProgress(t, mutant: mutant)
        let vols: [Volatile] = buildVolatiles(hairs: hairs, progress: u, mutant: mutant, facing: insetDirection)
        return FrameState(time: t, phase: phase, lift: lift, progress: u, ant: ant, crystal: cf, volatiles: vols)
    }
}

/// One moment of the loop.
struct FrameState {
    var time: Float
    var phase: Float
    var lift: Float
    var progress: Float
    var ant: AntModel
    var crystal: CrystalFrame
    var volatiles: [Volatile]

    /// The per-frame buffer: the yolk's transform, R's columns then T; then
    /// the odour slots (position µm, drawn radius; radius 0 = none).
    var gpuFrame: [SIMD4<Float>] {
        let r: simd_float3x3 = crystal.rotation
        var out: [SIMD4<Float>] = [SIMD4<Float>(r.columns.0, 0), SIMD4<Float>(r.columns.1, 0),
                                   SIMD4<Float>(r.columns.2, 0), SIMD4<Float>(crystal.translation, 0)]
        for k in 0..<odourSlots {
            if k < volatiles.count {
                out.append(SIMD4<Float>(volatiles[k].position, volatiles[k].radius))
            } else {
                out.append(SIMD4<Float>(0, 0, 0, 0))
            }
        }
        return out
    }
}

/// The finished picture, and what each pixel's centre sample saw:
/// (view 1 main / 2 inset / 3 molecule, material, sub-part or material under the film, 0).
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

/// The scene's buffers for one frame, in kernel argument order 3…8.
func sceneBuffers(_ device: MTLDevice, _ scene: Scene, _ frame: FrameState) throws -> [MTLBuffer] {
    let mol = gpuMolecule(scene.molecule)
    return [try buffer(device, gpuShapes(frame.ant.shapes)), try buffer(device, gpuFood(scene.spheres, core: scene.core)),
            try buffer(device, gpuHairs(scene.hairs)), try buffer(device, mol.atoms), try buffer(device, mol.bonds),
            try buffer(device, frame.gpuFrame)]
}

/// A pixel rectangle, [x0, x1) × [y0, y1).
struct PixelRect {
    var x0: Int
    var y0: Int
    var x1: Int
    var y1: Int

    func union(_ o: PixelRect) -> PixelRect {
        PixelRect(x0: min(x0, o.x0), y0: min(y0, o.y0), x1: max(x1, o.x1), y1: max(y1, o.y1))
    }
    func clipped(width: Int, height: Int) -> PixelRect {
        PixelRect(x0: max(x0, 0), y0: max(y0, 0), x1: min(x1, width), y1: min(y1, height))
    }
    func contains(_ x: Int, _ y: Int) -> Bool { x >= x0 && x < x1 && y >= y0 && y < y1 }
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
        let bufs: [MTLBuffer] = try sceneBuffers(device, scene, frame)
        // Step 33's fix: a band is 24 full-width rows' worth of pixels, however
        // narrow the region — a narrow region in 24-row bands gives the GPU
        // too few threads at a time to fill it, and ran several times slower.
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
                                colOffset: UInt32(region.x0), colEnd: UInt32(region.x1), rowEnd: UInt32(region.y1))
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

/// Step 36's one-shot entry point, kept for the tests: one frame, whole.
func renderAnt(width: Int, height: Int, samples: Int, scene: Scene, frame: FrameState,
               on device: MTLDevice) throws -> (image: AntImage, gpuSeconds: Double) {
    let r = try Renderer(device: device, scene: scene, width: width, height: height)
    let gpu: Double = try r.render(frame, samples: samples)
    return (r.image, gpu)
}

/// The parts of the picture that can change during the loop: everywhere the
/// moving antennae project to over every frame, grown by `margin` mm for
/// their shadows, their faint sheen on the yolk and the occlusion they
/// cast; and the micrometre inset whole. The molecule insets never change.
/// Step 37: the margin is 1 mm, not step 33's 0.65 — at 0.65 the soft
/// shadow of the lifted antenna on the lumpy crumb changed pixels by up to
/// 4 levels outside the region (the test below found it). Outside these every frame is identical, so
/// only these are rendered after the first frame; a test renders frames
/// whole and checks that nothing outside them ever differs.
struct MovingRegion {
    var rects: [PixelRect]

    func contains(_ x: Int, _ y: Int) -> Bool { rects.contains { $0.contains(x, y) } }
    var bounds: PixelRect { rects.dropFirst().reduce(rects[0]) { $0.union($1) } }
    var area: Int { rects.reduce(0) { $0 + $1.area } }   // an upper bound: rects may overlap
}

func movingRegion(scene: Scene, frames: [FrameState], width: Int, height: Int, margin: Float = 1.0) -> MovingRegion {
    let pxPerMM: Float = Float(width) / mainViewWidth
    var lo = SIMD2<Float>(repeating: 1e9)
    var hi = SIMD2<Float>(repeating: -1e9)
    // The funiculi only: the scapes never move.
    for f in frames {
        for a in f.ant.antennae {
            for s in a.segments.dropFirst() {
                for p in [s.a, s.b] {
                    let q: SIMD2<Float> = projectMain(p, width: width, height: height)
                    lo = simd_min(lo, q)
                    hi = simd_max(hi, q)
                }
            }
        }
    }
    let m: Float = margin * pxPerMM
    var rects: [PixelRect] = [PixelRect(x0: Int(lo.x - m), y0: Int(lo.y - m), x1: Int(hi.x + m) + 1, y1: Int(hi.y + m) + 1)]
    let h: Float = Float(height)
    for (c, rad) in [(insetCentre, insetRadius)] {
        let pad: Float = 4
        rects.append(PixelRect(x0: Int(c.x * h - rad * h - pad), y0: Int(c.y * h - rad * h - pad),
                               x1: Int(c.x * h + rad * h + pad) + 1, y1: Int(c.y * h + rad * h + pad) + 1))
    }
    return MovingRegion(rects: rects.map { $0.clipped(width: width, height: height) })
}

/// One probe result.
struct Probe {
    var mainDistance: Float
    var mainMaterial: Int
    var insetDistance: Float
    var insetMaterial: Int
    var tasteHair: Float
    var smellHair: Float
    var food: Float
    var ant: Float
}

/// The scene's distances at arbitrary points, from the same kernel source the
/// render uses — so a test of the distance function is a test of the thing
/// that drew the picture, not of a copy of it.
func probeScene(_ points: [SIMD3<Float>], scene: Scene, frame: FrameState, library: MTLLibrary,
                on device: MTLDevice) throws -> [Probe] {
    let pso = try pipeline(device, library, "probe")
    let bufs: [MTLBuffer] = try sceneBuffers(device, scene, frame)
    let pts: [SIMD4<Float>] = points.map { SIMD4<Float>($0, 0) }
    let pb: MTLBuffer = try buffer(device, pts)
    guard let ob = device.makeBuffer(length: 16 * pts.count, options: .storageModeShared),
          let ob2 = device.makeBuffer(length: 16 * pts.count, options: .storageModeShared),
          let queue = device.makeCommandQueue(), let cb = queue.makeCommandBuffer(),
          let enc = cb.makeComputeCommandEncoder()
    else { throw AntError.gpu("could not set up the probe") }
    enc.setComputePipelineState(pso)
    enc.setBuffer(pb, offset: 0, index: 0)
    enc.setBuffer(ob, offset: 0, index: 1)
    enc.setBuffer(ob2, offset: 0, index: 2)
    for i in 0..<3 { enc.setBuffer(bufs[i], offset: 0, index: 3 + i) }
    enc.setBuffer(bufs[5], offset: 0, index: 6)
    enc.dispatchThreads(MTLSize(width: pts.count, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: min(pso.maxTotalThreadsPerThreadgroup, 256), height: 1, depth: 1))
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    if let e = cb.error { throw AntError.gpu(e.localizedDescription) }
    let o = ob.contents().assumingMemoryBound(to: SIMD4<Float>.self)
    let o2 = ob2.contents().assumingMemoryBound(to: SIMD4<Float>.self)
    return (0..<pts.count).map { i in
        Probe(mainDistance: o[i].x, mainMaterial: Int(o[i].y), insetDistance: o[i].z, insetMaterial: Int(o[i].w),
              tasteHair: o2[i].x, smellHair: o2[i].y, food: o2[i].z, ant: o2[i].w)
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
