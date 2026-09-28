// Step 67: step 66's Render.swift, copied unchanged so step 66 stays exactly as
// committed. The camera that moves is in Shot.swift.
//
// The upper arch, rendered as one still: one GPU thread per pixel marching rays
// into a scene made entirely of distance functions. Step 20's renderer, copied
// and turned over: the teeth are the maxillary ones, hanging from their gum,
// and the tongue and floor of the mouth are gone — above the teeth, where a
// camera below them looks, is the hard palate with its rugae and the incisive
// papilla.
//
// What is kept from step 20, and why each piece is honest, is described there
// and repeated where it matters below: the saliva film over the enamel (2.0%)
// and the enamel beneath it (0.94%); translucency at the thin incisal edge read
// off the distance function; colour from measured CIELAB, neck to edge; cusps
// that become cones past a radius so the height field cannot steepen forever;
// the occlusal height field divided by a constant √(1 + TOP_SLOPE²); the gum
// built in real millimetres and divided by its steepness once, at the end.
//
// THE FLIP. Anatomy.swift builds the arch in step 20's frame, crowns up. The
// kernel reads every world point through `archFrame`, which turns y over, so
// the crowns hang down in the world and everything else in step 20's code
// works unchanged in the arch's frame. A reflection is an isometry, so every
// distance stays exactly as honest as it was.

import CoreGraphics
import Foundation
import ImageIO
import Metal
import UniformTypeIdentifiers
import simd

// MARK: - the crown's shape, beyond the table

/// Where the crown is widest mesiodistally, as a fraction of its height from
/// the neck: the contact areas are in the incisal third on incisors and the
/// occlusal third on posterior teeth (Wheeler's, each tooth's chapter). Step 20.
func contactHeight(_ kind: CrownKind) -> Float {
    switch kind {
    case .incisor: return 0.82
    case .canine: return 0.75
    default: return 0.78
    }
}

/// Half the labiolingual thickness at the very top of the crown. An incisal
/// edge is about a millimetre thick; a posterior occlusal table is roughly 55–
/// 65% of the crown's depth (Wheeler's). MODEL within those, step 20's.
func topHalfDepth(_ spec: ToothSpec) -> Float {
    switch spec.kind {
    case .incisor: return 0.55
    case .canine: return 0.9
    case .firstPremolar, .secondPremolar: return 0.30 * spec.depth
    case .firstMolar, .secondMolar, .thirdMolar: return 0.32 * spec.depth
    }
}

/// The largest stretch of the shear (u, v) → (u + k v, v): its top singular
/// value. A distance read in sheared coordinates is divided by it to stay a
/// lower bound.
func shearStretch(_ k: Float) -> Float {
    let k2: Float = k * k
    let root: Float = (4 + k2).squareRoot()
    let big: Float = (2 + k2 + k * root) / 2
    return big.squareRoot()
}

// MARK: - the camera and the lights

/// Looking up into the upper arch from below and in front, a little off to
/// the patient's left (−x, the viewer's right), as step 20 looked down into the
/// lower arch from above and in front of the same side. MODEL. The exact
/// mirror of step 20's camera, (−50, −17, 1), was tried first: it framed the
/// near premolars as step 20 did but could not see past the incisors into the
/// vault, where the rugae and the incisive papilla are — the things that make
/// an upper arch look like one. So the lens comes round to the front and down,
/// about 55° below the occlusal plane, and looks up past the incisal edges
/// into the palate, the way a dentist's occlusal mirror shot does.
let cameraPosition = SIMD3<Float>(-16, -56, -30)
let cameraTarget = SIMD3<Float>(-1, 8, 18)
let tanHalfFOV: Float = 0.30

/// Where the camera stands and what it looks at, read by the kernel at run
/// time (step 21's change to step 20), so step 67 can move it.
struct Camera {
    var position: SIMD3<Float>
    var target: SIMD3<Float>

    var forward: SIMD3<Float> { simd_normalize(target - position) }
    var right: SIMD3<Float> { simd_normalize(simd_cross(forward, SIMD3<Float>(0, 1, 0))) }
    var up: SIMD3<Float> { simd_cross(right, forward) }
    /// The fill softbox sits beside the lens — just below it now, as step 20's
    /// sat just above — so it goes where the camera goes.
    var fill: SIMD3<Float> { simd_normalize(simd_normalize(position - target) + SIMD3<Float>(0, -0.25, 0)) }
}

let stillCamera = Camera(position: cameraPosition, target: cameraTarget)

/// Step 20's lights, turned over with the scene: the key softbox below and in
/// front, as a photographer re-aims it for the upper arch, and the room's
/// bright side towards it. Each light is a disc whose radiance is its
/// irradiance over its solid angle, so the highlight a wet tooth mirrors is
/// exactly as bright as the light that lights it (step 20).
let keyDirection = simd_normalize(SIMD3<Float>(-0.45, -1.0, -0.55))
let keyColour = SIMD3<Float>(1.0, 0.98, 0.95) * 1.7
let fillColour = SIMD3<Float>(0.96, 0.97, 1.0) * 0.8
let keyDiscCosine: Float = 0.985     // about 10°
let fillDiscCosine: Float = 0.990    // about 8°

func discRadiance(_ irradiance: SIMD3<Float>, cosine: Float) -> SIMD3<Float> {
    let solidAngle: Float = 2 * Float.pi * (1 - cosine)
    return irradiance / solidAngle
}

/// The picture's ground, in display values: step 20's near white.
let backgroundDisplay: Float = 0.965

// MARK: - GPU layout

/// One tooth as the kernel sees it: seven float4s, so there is no padding to
/// get wrong between Swift and Metal. All in the arch's frame.
struct GPUTooth {
    var frame: SIMD4<Float>      // centre x, centre z, tangent x, tangent z
    var outward: SIMD4<Float>    // outward x, outward z, contact height, side
    var halves: SIMD4<Float>     // half width, half cervical width, half depth, half cervical depth
    var shape: SIMD4<Float>      // crown height, half depth at top, CEJ curve, kind
    var toothBound: SIMD4<Float> // sphere enclosing the crown and the root shown
    var gumBound: SIMD4<Float>   // sphere enclosing this tooth's block of gum
    var extra: SIMD4<Float>      // occlusal shear, shade class, shear stretch, Universal number
}

/// How much root is modelled beyond the neck: only enough to sit in the gum.
let rootShown: Float = 6.0
/// How deep the gum body runs beyond the neck: step 20's 16 mm.
let gumDepth: Float = 16.0
/// Where the gum body ends, for every tooth: 16 mm beyond the tallest crown's
/// neck (the central incisor's, 10.5 mm), so no neck has less than step 20's.
var gumFloor: Float { (toothTable.map { $0.crownHeight }.max() ?? 10.5) + gumDepth }

func gpuTeeth(_ placed: [PlacedTooth]) -> [GPUTooth] {
    placed.enumerated().map { (index, t) in
        let s: ToothSpec = t.spec
        let h: Float = s.crownHeight
        let k: Float = occlusalShear(s.kind)
        let slack: Float = k * s.depth / 2
        let tb = SIMD4<Float>(t.centre.x, -(h + rootShown) / 2, t.centre.y,
                              simd_length(SIMD3<Float>(s.width / 2 + slack, (h + rootShown) / 2, s.depth / 2)) + 0.5)
        let gHalfDepth: Float = s.cervicalDepth / 2 + gumWidth
        let gHalfHeight: Float = gumFloor / 2
        let gb = SIMD4<Float>(t.centre.x, -gHalfHeight, t.centre.y,
                              simd_length(SIMD3<Float>(s.width / 2 + 1.5, gHalfHeight, gHalfDepth)) + 1.5)
        let shade: Int = shadeClass(index: index % 8).rawValue
        return GPUTooth(
            frame: SIMD4<Float>(t.centre.x, t.centre.y, t.tangent.x, t.tangent.y),
            outward: SIMD4<Float>(t.outward.x, t.outward.y, contactHeight(s.kind), t.side),
            halves: SIMD4<Float>(s.width / 2, s.cervicalWidth / 2, s.depth / 2, s.cervicalDepth / 2),
            shape: SIMD4<Float>(h, topHalfDepth(s), s.cejCurve, Float(s.kind.rawValue)),
            toothBound: tb, gumBound: gb,
            extra: SIMD4<Float>(k, Float(shade), shearStretch(k), Float(t.number)))
    }
}

/// A point in the world, in the arch's frame — and back: the same flip.
func archFrame(_ p: SIMD3<Float>) -> SIMD3<Float> { SIMD3<Float>(p.x, -p.y, p.z) }
func world(_ p: SIMD3<Float>) -> SIMD3<Float> { archFrame(p) }

// MARK: - the kernel

/// Shading mutants, as step 20's. Each one must be caught by a test.
enum Mutant: Int {
    case none = 0
    case flat = 1       // one colour from neck to edge
    case opaque = 2     // no translucency at the thin edge
}

/// How far a ray may trust a distance: 60% of it, as in step 20. The tests
/// measure the real over-report.
let stepScale: Float = 0.6

func metal(_ v: SIMD3<Float>) -> String { "float3(\(v.x), \(v.y), \(v.z))" }

// MARK: - the maxillary tuberosity

// Behind the last upper molar the alveolar ridge ends in the maxillary
// tuberosity, a rounded eminence covered in the same mucosa. It is step 21's
// retromolar pad, re-sized: an ellipsoid lying along the ridge, short in front
// and long behind, sunk below the gum's floor so its sides lean in as they
// rise, and cut off at that floor.
//
// Every size is MODEL: the tuberosity is described as a rounded bulge of the
// ridge behind the last molar, about as wide as the ridge. Step 21's pad
// numbers are kept for its length and width; its top is put half a millimetre
// beyond the third molar's gum margin (−5.5 mm here), so it closes the ridge
// round the tooth without standing proud of it as the lower pad does.
let padBehind: Float = 2.0
let padFront: Float = 7.0
let padBack: Float = 8.0
let padAcross: Float = 6.0
let padSink: Float = 8.0
let padTop: Float = -6.0

// MARK: - the palate as the kernel sees it

/// How far the palate's shelf reaches beyond the vault's rim, under the gum.
/// MODEL: enough to meet the gum body all round the arch.
let palateRim: Float = 3.0
/// How softly the vault's rim turns into the shelf. MODEL.
let rimRound: Float = 2.0
/// How softly the palate and the gum meet: they are one continuous mucosa.
/// MODEL: wider than the gum's own blend so no seam shows.
let palateBlend: Float = 3.0
/// How softly each ruga and the papilla rise out of the palate. MODEL.
let rugaBlend: Float = 0.6

func kernelSource(mutant: Mutant) -> String {
    let placed: [PlacedTooth] = placeTeeth()
    let toothCount: Int = placed.count
    let lastLeft: Int = toothIndex(16, in: placed) ?? 7
    let lastRight: Int = toothIndex(1, in: placed) ?? 15
    let shades: [ShadeClass] = ShadeClass.allCases
    let mids: String = shades.map { metal(labToLinearSRGB(middleThirdLab[$0]!)) }.joined(separator: ", ")
    let cervs: String = shades.map { metal(labToLinearSRGB(middleThirdLab[$0]! + cervicalShift)) }.joined(separator: ", ")
    let rugaCount: Int = rugae.count
    var rugaPoints: [SIMD3<Float>] = rugae.flatMap { $0.points }
    if rugaPoints.isEmpty { rugaPoints = [SIMD3<Float>(0, 0, 0)] }
    let rugaText: String = rugaPoints.map { metal($0) }.joined(separator: ", ")
    let papillaZ: Float = incisivePapillaZ
    let papillaRoof: Float = vault.roof(0, papillaZ) ?? vault.rimY
    let papillaY: Float = papillaRoof + papillaHeight - papillaHalfWidth
    let slabDepth: Float = vault.radii.y + 6
    let vaultMin: Float = min(vault.radii.x, min(vault.radii.y, vault.radii.z))
    // The shelf reaches `palateRim` beyond the vault in front and at the
    // sides, where the gum covers it, and not at all behind, where nothing does.
    let footR = SIMD2<Float>(vault.radii.x + palateRim, vault.radii.z + palateRim / 2)
    return """
    #include <metal_stdlib>
    using namespace metal;

    struct Tooth { float4 frame; float4 outward; float4 halves; float4 shape; float4 toothBound; float4 gumBound; float4 extra; };
    struct Params { uint width; uint height; uint rowOffset; uint samples;
                    float4 camPos; float4 camFwd; float4 camRight; float4 camUp; float4 fillDir; };

    constant uint TOOTH_COUNT = \(toothCount);
    constant int MUTANT = \(mutant.rawValue);
    constant float STEP_SCALE = \(stepScale);
    constant float HIT_EPS = 0.003;
    constant float FAR = 260.0;

    constant float TAN_HALF_FOV = \(tanHalfFOV);

    constant float3 KEY_DIR = \(metal(keyDirection));
    constant float3 KEY_COL = \(metal(keyColour));
    constant float3 FILL_COL = \(metal(fillColour));
    constant float3 KEY_RADIANCE = \(metal(discRadiance(keyColour, cosine: keyDiscCosine)));
    constant float3 FILL_RADIANCE = \(metal(discRadiance(fillColour, cosine: fillDiscCosine)));
    constant float KEY_DISC = \(keyDiscCosine);
    constant float FILL_DISC = \(fillDiscCosine);
    constant float BG = \(backgroundDisplay);

    constant float ROOT_SHOWN = \(rootShown);
    constant float EDGE_ROUND = 0.45;
    constant float CUSP_K = 0.3;          // dome curvature, per mm²
    constant float CUSP_BLEND = 0.7;      // how softly neighbouring cusps meet in a groove
    constant float CUSP_SLOPE = 1.0;      // steepest cusp incline, 45°; real inclines run ~25–45°
    constant float TOP_SLOPE = 1.25;      // steepest the whole occlusal height field gets, canine ridges included

    constant float GUM_MARGIN = \(gumMarginAboveCEJ);
    constant float PAPILLA_EXTRA = \(papillaExtra);
    constant float GUM_WIDTH = \(gumWidth);
    constant float GUM_SLOPE = \(gumSlope);
    constant float GUM_DEPTH = \(gumDepth);
    constant float GUM_ROUND = 0.8;
    constant float GUM_COLLAR = 4.5;      // mm over which the gum goes from a knife edge to full thickness
    constant float GUM_BLEND = 2.0;
    constant float GUM_FLOOR = \(gumFloor);

    constant float RIM_Y = \(vault.rimY);
    constant float VAULT_Z = \(vault.centreZ);
    constant float3 VAULT_R = \(metal(vault.radii));
    constant float VAULT_MIN_R = \(vaultMin);
    constant float FOOT_Z = \(vault.centreZ - palateRim / 2);
    constant float2 FOOT_R = float2(\(footR.x), \(footR.y));
    constant float FOOT_MIN_R = \(min(footR.x, footR.y));
    constant float SLAB_DEPTH = \(slabDepth);
    constant float RIM_ROUND = \(rimRound);
    constant float PALATE_BLEND = \(palateBlend);
    constant uint RUGA_COUNT = \(rugaCount);
    constant float3 RUGA_P[\(rugaPoints.count)] = { \(rugaText) };
    constant float RUGA_R = \(rugaRadius);
    constant float RUGA_BLEND = \(rugaBlend);
    constant float3 PAPILLA_C = float3(0.0, \(papillaY), \(papillaZ));
    constant float PAPILLA_HALF = \(papillaHalfLength - papillaHalfWidth);
    constant float PAPILLA_R = \(papillaHalfWidth);

    constant float3 MID[4] = { \(mids) };
    constant float3 CERV[4] = { \(cervs) };
    constant float3 INCISAL = \(metal(labToLinearSRGB(incisalLab)));
    constant float3 GUM = \(metal(labToLinearSRGB(gingivaLab)));
    constant float3 PALATE = \(metal(labToLinearSRGB(palateLab)));

    constant float FILM_F0 = \(filmF0);
    constant float ENAMEL_F0 = \(enamelUnderFilmF0);

    // ---------------------------------------------------------------- helpers

    float3 archFrame(float3 p) { return float3(p.x, -p.y, p.z); }

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

    float segment(float3 p, float3 a, float3 b) {
        float3 pa = p - a;
        float3 ba = b - a;
        float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
        return length(pa - ba * h);
    }

    // ---------------------------------------------------------------- a tooth
    // Everything from here to the scene works in the arch's frame.

    float2 localUV(constant Tooth &T, float3 p) {
        float2 rel = p.xz - T.frame.xy;
        return float2(dot(rel, T.frame.zw), dot(rel, T.outward.xy));
    }

    // The molars' rhomboid: the buccal side slid mesially (Anatomy.swift).
    float2 sheared(constant Tooth &T, float2 uv) { return float2(uv.x + T.extra.x * uv.y, uv.y); }

    // Step 20's crown walls: a rounded rectangle in plan, narrow at the neck.
    float crownSide(constant Tooth &T, float2 uv, float y) {
        float H = T.shape.x;
        float t = (y + H) / H;
        float a;
        float b;
        if (t >= 0.0) {
            a = mix(T.halves.y, T.halves.x, smoothstep(0.0, T.outward.z, t));
            b = t < 0.3 ? mix(T.halves.w, T.halves.z, smoothstep(0.0, 0.3, t))
                        : mix(T.halves.z, T.shape.y, smoothstep(0.3, 1.0, t));
        } else {
            float below = y + H;                  // negative: mm into the root
            a = T.halves.y * max(1.0 + 0.05 * below, 0.3);
            b = T.halves.w * max(1.0 + 0.05 * below, 0.3);
        }
        float r = min(a, b) * 0.7;
        float2 q = abs(uv) - float2(a, b) + r;
        return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - r;
    }

    // Step 20's rounded cusp: a paraboloid near its tip, a cone of slope
    // CUSP_SLOPE further out, so the height field cannot steepen forever.
    float dome(float2 uv, float cu, float cv, float h) {
        float r = length(uv - float2(cu, cv));
        float r0 = CUSP_SLOPE / (2.0 * CUSP_K);
        return r < r0 ? h - CUSP_K * r * r : h - CUSP_K * r0 * r0 - CUSP_SLOPE * (r - r0);
    }

    // The top as a height field. u runs distally, v buccally or labially.
    // The cusps of the upper teeth (Wheeler's, each tooth's occlusal aspect):
    //   first premolar  — buccal and lingual, the buccal the longer by ~1 mm
    //                     and its tip a little distal, the lingual's mesial;
    //   second premolar — two cusps of nearly equal length;
    //   first molar     — mesiolingual the largest, then mesiobuccal,
    //                     distobuccal, and a small distolingual;
    //   second molar    — the distolingual smaller still;
    //   third molar     — three: the distolingual usually missing.
    // Heights between cusps are MODEL.
    float crownTop(constant Tooth &T, float2 uv) {
        int kind = int(T.shape.w + 0.5);
        float a = T.halves.x;
        float b = T.halves.z;
        if (kind == 0) {
            float un = uv.x / a;
            return -0.45 * un * un * un * un;
        }
        if (kind == 1) {
            // The maxillary canine's tip sits a little mesial, and its mesial
            // cusp ridge is the shorter, steeper one (Wheeler's).
            float du = uv.x + 0.12 * a;
            float k = du < 0.0 ? 1.05 : 0.8;
            return -k * (sqrt(du * du + 0.5) - sqrt(0.5)) - 0.25 * max(-uv.y, 0.0);
        }
        float h;
        if (kind == 2) {
            h = smax(dome(uv, 0.05 * a, 0.35 * b, 0.0), dome(uv, -0.1 * a, -0.4 * b, -1.0), CUSP_BLEND);
        } else if (kind == 3) {
            h = smax(dome(uv, 0.0, 0.35 * b, 0.0), dome(uv, 0.0, -0.4 * b, -0.3), CUSP_BLEND);
        } else if (kind == 4 || kind == 5) {
            float dl = kind == 4 ? -1.3 : -1.8;
            h = smax(dome(uv, -0.3 * a, -0.45 * b, 0.0), dome(uv, -0.45 * a, 0.5 * b, -0.3), CUSP_BLEND);
            h = smax(h, dome(uv, 0.45 * a, 0.5 * b, -0.6), CUSP_BLEND);
            h = smax(h, dome(uv, 0.5 * a, -0.5 * b, dl), CUSP_BLEND);
        } else {
            h = smax(dome(uv, -0.2 * a, -0.4 * b, 0.0), dome(uv, -0.4 * a, 0.5 * b, -0.3), CUSP_BLEND);
            h = smax(h, dome(uv, 0.4 * a, 0.45 * b, -0.7), CUSP_BLEND);
        }
        return h;
    }

    float crownSDF(constant Tooth &T, float3 p, thread float &side) {
        float2 uv = sheared(T, localUV(T, p));
        float stretch = T.extra.z;
        // Read in sheared coordinates, a distance can over-report by the
        // shear's stretch; divide by it (1 on every tooth but the molars).
        side = crownSide(T, uv, p.y) / stretch;
        // Step 20: the top's height difference over-reports by up to
        // √(1 + slope²) — and by the stretch again on a sheared crown.
        float2 inside = clamp(uv, -T.halves.xz, T.halves.xz);
        float slope = TOP_SLOPE * stretch;
        float top = (p.y - crownTop(T, inside)) / sqrt(1.0 + slope * slope);
        float d = smax(side, top, EDGE_ROUND);
        return max(d, (-T.shape.x - ROOT_SHOWN) - p.y);
    }

    // Step 20's gum: one tooth's share, a shell round the neck and root that is
    // a knife edge at the margin and thickens beyond it, in real millimetres,
    // returning alongside how steep it can be so the scene divides once.
    float gumBlock(constant Tooth &T, float3 p, float side, thread float &steep) {
        float2 uv = localUV(T, p);
        float H = T.shape.x;
        float un = clamp(uv.x / T.halves.x, -1.25, 1.25);
        float climb = T.shape.z + PAPILLA_EXTRA;
        float top = -H + GUM_MARGIN + climb * un * un;
        float below = top - p.y;
        float x = clamp(below / GUM_COLLAR, 0.0, 1.0);
        float thick = GUM_WIDTH * x * x * (3.0 - 2.0 * x) + GUM_SLOPE * max(below - GUM_COLLAR, 0.0);
        float shell = side - thick;
        // One floor for the whole jaw, beyond the tallest crown's gum, rather
        // than step 20's floor per tooth: seen from below, step 20's steps
        // between neighbouring blocks showed as layered sheets.
        float bottom = -GUM_FLOOR - p.y;
        float slope = 2.0 * climb * 1.25 / T.halves.x;
        float lean = sqrt(1.0 + slope * slope);
        float thickRate = max(GUM_WIDTH * 1.5 / GUM_COLLAR, GUM_SLOPE);
        steep = 1.0 + thickRate * lean;
        return max(smax(shell, p.y - top, GUM_ROUND), bottom);
    }

    // The maxillary tuberosity behind each last molar (#16 and #1, teeth
    // \(lastLeft) and \(lastRight)): step 21's pad, in the molar's own frame.
    float tuberositySDF(float3 p, constant Tooth *teeth) {
        float d = 1e9;
        for (uint k = 0; k < 2; k++) {
            constant Tooth &T = teeth[k == 0 ? \(lastLeft) : \(lastRight)];
            float base = -GUM_FLOOR;
            float2 c = T.frame.xy + T.frame.zw * (T.halves.x + \(padBehind));
            float3 q = p - float3(c.x, base - \(padSink), c.y);
            float3 l = float3(dot(q.xz, T.frame.zw), q.y, dot(q.xz, T.outward.xy));
            float along = l.x < 0.0 ? \(padFront) : \(padBack);
            float3 r = float3(along, \(padTop) - base + \(padSink), \(padAcross));
            float k0 = length(l / r);
            float k1 = length(l / (r * r));
            float e = k0 * (k0 - 1.0) / k1;
            d = min(d, max(e, base - p.y));
        }
        return d;
    }

    // The hard palate. A shelf of tissue on the far side of the gum margin
    // from the crowns, inside an ellipse a little bigger than the vault, with
    // the vault carved out of it: an ellipsoid of air whose widest section is
    // the rim. The vault's distance is (|q/r| − 1) × its smallest radius, which
    // never over-reports on either side of its surface; the footprint's the
    // same. Then the rugae and the incisive papilla, as rounded ridges.
    float palateSDF(float3 p) {
        float2 fq = (p.xz - float2(0.0, FOOT_Z)) / FOOT_R;
        float foot = (length(fq) - 1.0) * FOOT_MIN_R;
        float slab = max(p.y - RIM_Y, foot);
        slab = max(slab, (RIM_Y - SLAB_DEPTH) - p.y);
        float3 q = (p - float3(0.0, RIM_Y, VAULT_Z)) / VAULT_R;
        float cavity = (length(q) - 1.0) * VAULT_MIN_R;
        float d = smax(slab, -cavity, RIM_ROUND);
        float ridge = 1e9;
        for (uint i = 0; i < RUGA_COUNT; i++) {
            ridge = min(ridge, segment(p, RUGA_P[3 * i], RUGA_P[3 * i + 1]));
            ridge = min(ridge, segment(p, RUGA_P[3 * i + 1], RUGA_P[3 * i + 2]));
        }
        ridge -= RUGA_R;
        float papilla = segment(p, PAPILLA_C - float3(0.0, 0.0, PAPILLA_HALF), PAPILLA_C + float3(0.0, 0.0, PAPILLA_HALF)) - PAPILLA_R;
        ridge = min(ridge, papilla);
        return smin(d, ridge, RUGA_BLEND);
    }

    // ---------------------------------------------------------------- the scene

    // Materials: 1 enamel, 2 gum, 3 palate. `pw` is a world point.
    float sceneSDF(float3 pw, thread int &mat, constant Tooth *teeth) {
        float3 p = archFrame(pw);
        float dT = 1e9;
        float dG = 1e9;
        // Every tooth, every time (step 20 explains why skipping is unsafe).
        float steepG = 1.0;
        for (uint i = 0; i < TOOTH_COUNT; i++) {
            float side;
            float c = crownSDF(teeth[i], p, side);
            dT = min(dT, c);
            float steep;
            float g = gumBlock(teeth[i], p, side, steep);
            float h = clamp(0.5 + 0.5 * (g - dG) / GUM_BLEND, 0.0, 1.0);
            dG = mix(g, dG, h) - GUM_BLEND * h * (1.0 - h);
            steepG = mix(steep, steepG, h);
        }
        dG /= steepG;
        dG = smin(dG, tuberositySDF(p, teeth), GUM_BLEND);
        float dP = palateSDF(p);
        // Gum and palate are one mucosa, blended; the label goes to the nearer.
        float dM = smin(dG, dP, PALATE_BLEND);
        float d = dT;
        mat = 1;
        if (dM < d) { d = dM; mat = dG <= dP ? 2 : 3; }
        return d;
    }

    // Teeth alone, and which one is nearest, for the shading. `pw` is world.
    float teethSDF(float3 pw, constant Tooth *teeth, thread int &nearest) {
        float3 p = archFrame(pw);
        float dT = 1e9;
        nearest = 0;
        for (uint i = 0; i < TOOTH_COUNT; i++) {
            // A crown's distance under-reports by at most √(1 + (1.25 × 1.09)²)
            // = 1.69 on the most sheared molar; 1.8 is safe.
            float tb = length(p - teeth[i].toothBound.xyz) - teeth[i].toothBound.w;
            if (tb > dT * 1.8) continue;
            float side;
            float c = crownSDF(teeth[i], p, side);
            if (c < dT) { dT = c; nearest = int(i); }
        }
        return dT;
    }

    float3 sceneNormal(float3 p, constant Tooth *teeth) {
        const float e = 0.01;
        int m;
        float3 k1 = float3(1, -1, -1), k2 = float3(-1, -1, 1), k3 = float3(-1, 1, -1), k4 = float3(1, 1, 1);
        return normalize(k1 * sceneSDF(p + k1 * e, m, teeth) + k2 * sceneSDF(p + k2 * e, m, teeth)
                       + k3 * sceneSDF(p + k3 * e, m, teeth) + k4 * sceneSDF(p + k4 * e, m, teeth));
    }

    // A ray that skims a surface takes tiny steps and can run out of them
    // before it arrives; step 20 then drew background, which showed as thin
    // white slivers along the gum's silhouette, seen from below. Here a ray
    // that runs out while still within a tenth of a millimetre of a surface
    // is counted as having reached it.
    bool march(float3 ro, float3 rd, constant Tooth *teeth, thread float &t, thread int &mat) {
        t = 0.0;
        float d = 1e9;
        for (int i = 0; i < 600; i++) {
            d = sceneSDF(ro + rd * t, mat, teeth);
            if (d < HIT_EPS) return true;
            t += d * STEP_SCALE;
            if (t > FAR) return false;
        }
        return d < 0.1;
    }

    float softShadow(float3 ro, float3 rd, constant Tooth *teeth) {
        float res = 1.0;
        float t = 0.08;
        int m;
        for (int i = 0; i < 80; i++) {
            float h = sceneSDF(ro + rd * t, m, teeth);
            res = min(res, 10.0 * h / t);
            t += clamp(h * STEP_SCALE, 0.03, 2.5);
            if (res < 0.002 || t > 70.0) break;
        }
        return clamp(res, 0.0, 1.0);
    }

    float ambientOcclusion(float3 p, float3 n, constant Tooth *teeth) {
        float occ = 0.0;
        float weight = 1.0;
        int m;
        for (int i = 1; i <= 5; i++) {
            float h = 0.35 * float(i) * float(i) * 0.5 + 0.2;
            float d = sceneSDF(p + n * h, m, teeth);
            occ += weight * clamp((h - d) / h, 0.0, 1.0);
            weight *= 0.75;
        }
        return clamp(1.0 - 0.45 * occ, 0.0, 1.0);
    }

    // ---------------------------------------------------------------- light

    // Step 20's room, turned over with the scene: bright towards the key.
    float3 environment(float3 d, float3 fillDir) {
        float up = smoothstep(-0.4, 0.8, -d.y);
        float3 c = mix(float3(0.22, 0.19, 0.19), float3(0.75, 0.75, 0.78), up);
        float kSoft = (1.0 - KEY_DISC) * 0.5;
        float fSoft = (1.0 - FILL_DISC) * 0.5;
        c += KEY_RADIANCE * smoothstep(KEY_DISC - kSoft, KEY_DISC + kSoft, dot(d, KEY_DIR));
        c += FILL_RADIANCE * smoothstep(FILL_DISC - fSoft, FILL_DISC + fSoft, dot(d, fillDir));
        return c;
    }

    float3 irradiance(float3 n) {
        return mix(float3(0.30, 0.25, 0.25), float3(0.85, 0.86, 0.9), 0.5 - 0.5 * n.y);
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

    float schlick(float f0, float c) { return f0 + (1.0 - f0) * pow(1.0 - c, 5.0); }

    // The crown-height fraction at a world point, 0 at the neck and 1 at the
    // occlusal plane; the tooth's kind, shade class and Universal number.
    float toothT(float3 pw, constant Tooth *teeth, thread int &kind, thread int &shade, thread int &number) {
        int idx;
        teethSDF(pw, teeth, idx);
        kind = int(teeth[idx].shape.w + 0.5);
        shade = int(teeth[idx].extra.y + 0.5);
        number = int(teeth[idx].extra.w + 0.5);
        float H = teeth[idx].shape.x;
        float y = archFrame(pw).y;
        return clamp((y + H) / H, 0.0, 1.0);
    }

    float3 shade(float3 p, float3 rd, int mat, constant Tooth *teeth, float3 fillDir) {
        float3 n = sceneNormal(p, teeth);
        float3 v = -rd;
        float occ = ambientOcclusion(p, n, teeth);
        float sh = softShadow(p + n * 0.05, KEY_DIR, teeth);

        float3 albedo;
        float wrap = 0.15;
        float3 sssTint = float3(0.0);
        float filmAlpha = 0.07;
        float baseAlpha = 0.3;
        float baseF0 = ENAMEL_F0;

        if (mat == 1) {
            int kind;
            int sc;
            int number;
            float t = toothT(p, teeth, kind, sc, number);
            float3 mid = MID[sc];
            float3 cerv = CERV[sc];
            albedo = MUTANT == 1 ? mid : mix(cerv, mid, smoothstep(0.22, 0.5, t));
            float anterior = kind <= 1 ? 1.0 : 0.35;
            if (MUTANT != 1) albedo = mix(albedo, INCISAL, smoothstep(0.72, 1.0, t) * anterior);
            // Translucency (step 20): step inward and see how soon the tooth
            // ends. A thin edge lets the dark mouth behind show through.
            int idx;
            const float probe = 1.1;
            float inside = clamp(-teethSDF(p - n * probe, teeth, idx) / probe, 0.0, 1.0);
            float translucency = MUTANT == 2 ? 0.0 : (1.0 - inside) * smoothstep(0.55, 0.95, t);
            albedo = mix(albedo, albedo * float3(0.55, 0.62, 0.78), translucency * 0.75);
            wrap = 0.25;
        } else {
            // Gum and palate: wet, faintly stippled keratinized mucosa.
            float3 bump = noiseGrad(p * 3.0);
            n = normalize(n + 0.05 * (bump - n * dot(bump, n)));
            albedo = mat == 2 ? GUM : PALATE;
            wrap = 0.5;
            sssTint = float3(0.45, 0.08, 0.05);
            filmAlpha = mix(0.06, 0.22, noise3(p * 0.7));
            baseAlpha = 0.5;
            baseF0 = 0.0;
        }

        float ndv = max(dot(n, v), 1e-3);
        float nk = dot(n, KEY_DIR);
        float nf = dot(n, fillDir);
        float key = max((nk + wrap) / (1.0 + wrap), 0.0) * sh;
        float fill = max((nf + wrap) / (1.0 + wrap), 0.0);
        float terminator = max(wrap - abs(nk), 0.0) / max(wrap, 1e-3) * sh;
        float Ffilm = schlick(FILM_F0, ndv);
        float3 diffuse = albedo * (KEY_COL * key + FILL_COL * fill + irradiance(n) * occ)
                       + albedo * sssTint * KEY_COL * terminator;
        diffuse *= (1.0 - Ffilm);

        // Step 20's saliva film: a mirror of the room, softboxes included,
        // dimmed by Fresnel; blurred where the film is rough.
        float3 r = reflect(-v, n);
        float blur = smoothstep(0.08, 0.35, filmAlpha);
        float3 mirror = environment(r, fillDir);
        float3 sharp = mirror * Ffilm * mix(1.0, occ, 0.5);
        if (dot(r, KEY_DIR) > KEY_DISC - 0.02) sharp *= sh;
        float3 rough = KEY_COL * ggx(n, v, KEY_DIR, max(filmAlpha, 0.12), FILM_F0) * sh
                     + FILL_COL * ggx(n, v, fillDir, max(filmAlpha, 0.12), FILM_F0)
                     + irradiance(r) * Ffilm * occ;
        float3 specular = mix(sharp, rough, blur);
        if (baseF0 > 0.0) specular += KEY_COL * ggx(n, v, KEY_DIR, baseAlpha, baseF0) * sh
                                    + FILL_COL * ggx(n, v, fillDir, baseAlpha, baseF0);
        return diffuse + specular;
    }

    // Step 20's filmic shoulder on luminance, so a measured colour keeps its hue.
    float3 toneMap(float3 x) {
        float L = dot(x, float3(0.2126, 0.7152, 0.0722));
        float Lm = L * (1.0 + L / 16.0) / (1.0 + L);
        float3 c = x * (Lm / max(L, 1e-5)) * 1.12;
        float over = max(max(c.r, max(c.g, c.b)) - 1.0, 0.0);
        c = mix(c, float3(1.0), clamp(over, 0.0, 1.0));
        c = clamp(c, 0.0, 1.0);
        return select(1.055 * pow(c, 1.0 / 2.4) - 0.055, 12.92 * c, c <= 0.0031308);
    }

    float3 cameraRay(float2 pixel, constant Params &P) {
        float aspect = float(P.width) / float(P.height);
        float sx = (2.0 * pixel.x / float(P.width) - 1.0) * aspect * TAN_HALF_FOV;
        float sy = (1.0 - 2.0 * pixel.y / float(P.height)) * TAN_HALF_FOV;
        return normalize(P.camFwd.xyz + sx * P.camRight.xyz + sy * P.camUp.xyz);
    }

    // ---------------------------------------------------------------- kernels

    kernel void mouth(device uchar4 *pixels [[buffer(0)]],
                      device float4 *aux [[buffer(1)]],
                      constant Params &P [[buffer(2)]],
                      constant Tooth *teeth [[buffer(3)]],
                      uint2 gid [[thread_position_in_grid]]) {
        uint x = gid.x;
        uint y = gid.y + P.rowOffset;
        if (x >= P.width || y >= P.height) return;
        float3 camPos = P.camPos.xyz;
        float3 fillDir = P.fillDir.xyz;
        float3 sum = float3(0.0);
        uint S = P.samples;
        for (uint sy = 0; sy < S; sy++) {
            for (uint sx = 0; sx < S; sx++) {
                float2 jitter = float2((float(sx) + 0.5) / float(S), (float(sy) + 0.5) / float(S));
                float3 rd = cameraRay(float2(x, y) + jitter, P);
                float t;
                int mat;
                if (march(camPos, rd, teeth, t, mat)) {
                    sum += toneMap(shade(camPos + rd * t, rd, mat, teeth, fillDir));
                } else {
                    sum += float3(BG);
                }
            }
        }
        float3 c = sum / float(S * S);
        pixels[y * P.width + x] = uchar4(uchar3(round(clamp(c, 0.0, 1.0) * 255.0)), 255);

        // What the centre of the pixel sees, for the tests: material, crown
        // fraction, tooth kind, Universal number.
        float3 rd = cameraRay(float2(x, y) + 0.5, P);
        float t;
        int mat;
        float4 a = float4(0.0);
        if (march(camPos, rd, teeth, t, mat)) {
            a.x = float(mat);
            if (mat == 1) {
                int kind;
                int sc;
                int number;
                a.y = toothT(camPos + rd * t, teeth, kind, sc, number);
                a.z = float(kind);
                a.w = float(number);
            }
        }
        aux[y * P.width + x] = a;
    }

    kernel void probe(device const float4 *points [[buffer(0)]],
                      device float2 *out [[buffer(1)]],
                      constant Tooth *teeth [[buffer(2)]],
                      uint id [[thread_position_in_grid]]) {
        int mat;
        float d = sceneSDF(points[id].xyz, mat, teeth);
        out[id] = float2(d, float(mat));
    }
    """
}

// MARK: - running it

enum MouthError: Error, CustomStringConvertible {
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
    var camPos: SIMD4<Float>
    var camFwd: SIMD4<Float>
    var camRight: SIMD4<Float>
    var camUp: SIMD4<Float>
    var fillDir: SIMD4<Float>
}

/// The finished picture, and what each pixel's centre saw.
struct MouthImage {
    let width: Int
    let height: Int
    let pixels: MTLBuffer
    let aux: MTLBuffer

    func rgba(_ x: Int, _ y: Int) -> SIMD4<UInt8> {
        let p = pixels.contents().advanced(by: (y * width + x) * 4).assumingMemoryBound(to: UInt8.self)
        return SIMD4<UInt8>(p[0], p[1], p[2], p[3])
    }

    /// (material, crown fraction, tooth kind, Universal number).
    func seen(_ x: Int, _ y: Int) -> SIMD4<Float> {
        aux.contents().assumingMemoryBound(to: SIMD4<Float>.self)[y * width + x]
    }
}

func findDevice() throws -> MTLDevice {
    if let d = MTLCreateSystemDefaultDevice() ?? MTLCopyAllDevices().first { return d }
    throw MouthError.noMetalDevice
}

func makeLibrary(_ device: MTLDevice, mutant: Mutant) throws -> MTLLibrary {
    let options = MTLCompileOptions()
    // Precise maths, as in every step: the distance functions are subtracted
    // from one another to within a few microns.
    options.fastMathEnabled = false
    do {
        return try device.makeLibrary(source: kernelSource(mutant: mutant), options: options)
    } catch {
        throw MouthError.kernelCompile("\(error)")
    }
}

func pipeline(_ device: MTLDevice, _ library: MTLLibrary, _ name: String) throws -> MTLComputePipelineState {
    guard let f = library.makeFunction(name: name) else { throw MouthError.kernelCompile("no kernel \(name)") }
    do { return try device.makeComputePipelineState(function: f) } catch {
        throw MouthError.kernelCompile("\(error)")
    }
}

/// The compiled scene, kept so a sequence of frames compiles the kernel once.
final class MouthRenderer {
    let device: MTLDevice
    let pso: MTLComputePipelineState
    let probePSO: MTLComputePipelineState
    let queue: MTLCommandQueue
    let toothBuf: MTLBuffer

    init(mutant: Mutant = .none, on device: MTLDevice) throws {
        self.device = device
        let library = try makeLibrary(device, mutant: mutant)
        pso = try pipeline(device, library, "mouth")
        probePSO = try pipeline(device, library, "probe")
        var teeth: [GPUTooth] = gpuTeeth(placeTeeth())
        guard let tb = device.makeBuffer(bytes: &teeth, length: MemoryLayout<GPUTooth>.stride * teeth.count,
                                         options: .storageModeShared),
              let q = device.makeCommandQueue()
        else { throw MouthError.gpu("could not allocate buffers") }
        toothBuf = tb
        queue = q
    }

    /// Render in horizontal bands, one command buffer each, so no single piece
    /// of GPU work runs long enough to trip the system's watchdog.
    func render(width: Int, height: Int, samples: Int, camera: Camera) throws -> (image: MouthImage, gpuSeconds: Double) {
        guard let pixels = device.makeBuffer(length: width * height * 4, options: .storageModeShared),
              let aux = device.makeBuffer(length: width * height * 16, options: .storageModeShared)
        else { throw MouthError.gpu("could not allocate buffers") }
        let band: Int = 60
        var gpu: Double = 0
        var row: Int = 0
        while row < height {
            let rows: Int = min(band, height - row)
            guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else {
                throw MouthError.gpu("could not make a command buffer")
            }
            var params = Params(width: UInt32(width), height: UInt32(height),
                                rowOffset: UInt32(row), samples: UInt32(samples),
                                camPos: SIMD4<Float>(camera.position, 0), camFwd: SIMD4<Float>(camera.forward, 0),
                                camRight: SIMD4<Float>(camera.right, 0), camUp: SIMD4<Float>(camera.up, 0),
                                fillDir: SIMD4<Float>(camera.fill, 0))
            enc.setComputePipelineState(pso)
            enc.setBuffer(pixels, offset: 0, index: 0)
            enc.setBuffer(aux, offset: 0, index: 1)
            enc.setBytes(&params, length: MemoryLayout<Params>.stride, index: 2)
            enc.setBuffer(toothBuf, offset: 0, index: 3)
            let w: Int = pso.threadExecutionWidth
            let group = MTLSize(width: w, height: max(pso.maxTotalThreadsPerThreadgroup / w / 4, 1), depth: 1)
            enc.dispatchThreads(MTLSize(width: width, height: rows, depth: 1), threadsPerThreadgroup: group)
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            if let e = cb.error { throw MouthError.gpu(e.localizedDescription) }
            gpu += cb.gpuEndTime - cb.gpuStartTime
            row += rows
        }
        return (MouthImage(width: width, height: height, pixels: pixels, aux: aux), gpu)
    }

    /// The scene's distance and material at world points, from the same kernel
    /// source the render uses — so a test of the distance function is a test
    /// of the thing that drew the picture, not of a copy of it.
    func probe(_ points: [SIMD3<Float>]) throws -> [SIMD2<Float>] {
        var pts: [SIMD4<Float>] = points.map { SIMD4<Float>($0.x, $0.y, $0.z, 0) }
        guard let pb = device.makeBuffer(bytes: &pts, length: 16 * pts.count, options: .storageModeShared),
              let ob = device.makeBuffer(length: 8 * pts.count, options: .storageModeShared),
              let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder()
        else { throw MouthError.gpu("could not set up the probe") }
        enc.setComputePipelineState(probePSO)
        enc.setBuffer(pb, offset: 0, index: 0)
        enc.setBuffer(ob, offset: 0, index: 1)
        enc.setBuffer(toothBuf, offset: 0, index: 2)
        enc.dispatchThreads(MTLSize(width: pts.count, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(probePSO.maxTotalThreadsPerThreadgroup, 256), height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        if let e = cb.error { throw MouthError.gpu(e.localizedDescription) }
        let out = ob.contents().assumingMemoryBound(to: SIMD2<Float>.self)
        return (0..<pts.count).map { out[$0] }
    }
}

func savePNG(_ image: MouthImage, to url: URL) throws {
    try? FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    guard let provider = CGDataProvider(dataInfo: nil, data: image.pixels.contents(), size: image.width * image.height * 4,
                                        releaseData: { _, _, _ in }),
          let space = CGColorSpace(name: CGColorSpace.sRGB),
          let cg = CGImage(width: image.width, height: image.height, bitsPerComponent: 8, bitsPerPixel: 32,
                           bytesPerRow: image.width * 4, space: space,
                           bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                           provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent),
          let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)
    else { throw MouthError.png("could not set up the encoder for \(url.path)") }
    CGImageDestinationAddImage(dest, cg, nil)
    guard CGImageDestinationFinalize(dest) else { throw MouthError.png("could not write \(url.path)") }
}
