// A transmitted-light ray tracer: no lights, no normals, no shading at all.
//
// Every step of this project so far has asked "what is the nearest thing this
// ray hits, and how is it lit?" A brightfield micrograph asks a different
// question, and it is the whole point of step 13. In brightfield the lamp is
// BEHIND the specimen. The background is not scenery, it IS the light source,
// and every tone in the animal is light that got through it. So the ray does
// not stop at the first surface. It keeps going, collects every interval of
// tissue it crosses, and the pixel is
//
//     colour = backlight · exp(−τ),   τ = Σ σ · ℓ
//
// Beer–Lambert. σ is an RGB triple per tissue — three independent extinction
// coefficients — and that triple is the entire colour model here. There is no
// albedo, no specular, no ambient occlusion. A tissue is "olive" because it
// takes blue out of a cyan lamp, not because it is painted olive.
//
// Three things make that harder than it sounds.
//
// 1. ORDERED MULTI-HIT. The ray needs every interval, not the nearest one, so
//    the grid walk never terminates early and the intervals are kept in a
//    sorted list.
//
// 2. UNION, NOT SUM. Two primitives that overlap are ONE piece of tissue in
//    the overlap, not two. Summing would make every joint in the model a dark
//    seam. So intervals are merged into a union before integrating. The union
//    is taken PER TISSUE: σ is a property of tissue, and the union of a
//    tissue's primitives is the solid that tissue occupies. Different tissues
//    do add, and where one sits inside another (gut inside body, eye inside
//    head) its σ is written as the EXCESS over what surrounds it.
//
// 3. MAILBOXING. A primitive that straddles several grid boxes would be met
//    once per box, and each meeting would add its length again. Each ray
//    carries a bitmask of primitives it has already counted.
//
// Amanatides & Woo, "A Fast Voxel Traversal Algorithm for Ray Tracing",
// Eurographics (1987), for the grid walk.
// Beer–Lambert as used in quantitative phase/absorption microscopy; see any
// optics text — the model here is pure absorption, no scattering and no phase.

import Foundation
import Metal
import simd

func radians(_ degrees: Float) -> Float { degrees * .pi / 180 }

// MARK: - Tissues

/// Every primitive belongs to a tissue, and σ belongs to the tissue rather
/// than to the primitive. That is what makes the per-tissue union well
/// defined: the union of a tissue's primitives has one σ, so its length can be
/// integrated once no matter how many overlapping lobes drew it.
enum Tissue: Int, CaseIterable {
    case body = 0
    case limb
    case seta
    case compoundEye
    case naupliarEye
    case gut
    case appendage
    case alga
    case egg

    var name: String {
        switch self {
        case .body: return "body"
        case .limb: return "limb"
        case .seta: return "seta"
        case .compoundEye: return "compound eye"
        case .naupliarEye: return "naupliar eye"
        case .gut: return "gut"
        case .appendage: return "appendage"
        case .alga: return "alga"
        case .egg: return "egg"
        }
    }
}

let tissueCount = Tissue.allCases.count

// MARK: - Primitives

enum PrimKind: Int {
    case ellipsoid = 0
    case capsule = 1
}

/// One primitive, packed the way the kernel reads it.
///
///   ellipsoid   r0,r1,r2 = the rows of M⁻¹, c.xyz = centre
///   capsule     r0.xyz = end A, r0.w = radius, r1.xyz = end B
///   both        r2.w = tissue index, c.w = kind
///
/// The general ellipsoid is the new primitive of this step. M's three columns
/// are its semi-axis vectors, so ONE matrix carries size, flattening and pose
/// at once; the ray is pushed into unit-sphere space by M⁻¹, met with the unit
/// sphere there, and the two roots come back as world ray parameters unchanged
/// (the map is affine, so t is preserved). That same matrix machinery poses
/// all 22 limbs.
struct GPUPrim {
    var r0: SIMD4<Float>
    var r1: SIMD4<Float>
    var r2: SIMD4<Float>
    var c: SIMD4<Float>

    /// An ellipsoid with semi-axis vectors as the columns of `m`.
    static func ellipsoid(centre: SIMD3<Float>, m: simd_float3x3, tissue: Tissue) -> GPUPrim {
        let inv: simd_float3x3 = m.inverse
        // simd_float3x3 is column-major, so row k is (inv[0][k], inv[1][k], inv[2][k]).
        let row0 = SIMD3<Float>(inv[0][0], inv[1][0], inv[2][0])
        let row1 = SIMD3<Float>(inv[0][1], inv[1][1], inv[2][1])
        let row2 = SIMD3<Float>(inv[0][2], inv[1][2], inv[2][2])
        return GPUPrim(r0: SIMD4(row0, 0),
                       r1: SIMD4(row1, 0),
                       r2: SIMD4(row2, Float(tissue.rawValue)),
                       c: SIMD4(centre, Float(PrimKind.ellipsoid.rawValue)))
    }

    /// An axis-aligned sphere is the simplest ellipsoid: M = rI.
    static func sphere(centre: SIMD3<Float>, radius: Float, tissue: Tissue) -> GPUPrim {
        let m = simd_float3x3(diagonal: SIMD3<Float>(radius, radius, radius))
        return .ellipsoid(centre: centre, m: m, tissue: tissue)
    }

    /// A capsule: a segment swept by a ball. Used for everything long and
    /// round — setae, the gut tube, antennae, eye stalks.
    static func capsule(from a: SIMD3<Float>, to b: SIMD3<Float>, radius: Float,
                        tissue: Tissue) -> GPUPrim {
        let mid: SIMD3<Float> = (a + b) * 0.5
        return GPUPrim(r0: SIMD4(a, radius),
                       r1: SIMD4(b, 0),
                       r2: SIMD4(0, 0, 0, Float(tissue.rawValue)),
                       c: SIMD4(mid, Float(PrimKind.capsule.rawValue)))
    }

    var kind: PrimKind { c.w < 0.5 ? .ellipsoid : .capsule }
    var tissue: Int { Int(r2.w.rounded()) }

    /// The axis-aligned box the primitive occupies, which is what the grid
    /// files it by. For an ellipsoid p = c + M u with |u| = 1, the extent along
    /// axis k is the length of row k of M — and M's rows are the columns of
    /// M⁻ᵀ, so they are recovered by inverting back.
    func bounds() -> (lo: SIMD3<Float>, hi: SIMD3<Float>) {
        switch kind {
        case .capsule:
            let a = SIMD3<Float>(r0.x, r0.y, r0.z)
            let b = SIMD3<Float>(r1.x, r1.y, r1.z)
            let r = SIMD4<Float>(repeating: r0.w)
            let lo = simd_min(a, b) - SIMD3(r.x, r.y, r.z)
            let hi = simd_max(a, b) + SIMD3(r.x, r.y, r.z)
            return (lo, hi)
        case .ellipsoid:
            let inv = simd_float3x3(columns: (SIMD3(r0.x, r1.x, r2.x),
                                              SIMD3(r0.y, r1.y, r2.y),
                                              SIMD3(r0.z, r1.z, r2.z)))
            let m: simd_float3x3 = inv.inverse
            let ex: Float = simd_length(SIMD3(m[0][0], m[1][0], m[2][0]))
            let ey: Float = simd_length(SIMD3(m[0][1], m[1][1], m[2][1]))
            let ez: Float = simd_length(SIMD3(m[0][2], m[1][2], m[2][2]))
            let e = SIMD3<Float>(ex, ey, ez)
            let centre = SIMD3<Float>(c.x, c.y, c.z)
            return (centre - e, centre + e)
        }
    }
}

// MARK: - The camera

/// An orthographic camera, because a microscope objective is (near enough)
/// telecentric and because it makes the picture's scale a single exact number:
/// one pixel is `micronsPerPixel` everywhere in the field. The sub-pixel seta
/// clamp depends on that being one number and not a function of depth.
///
/// And it is dead sharp: no depth of field, no blur. A real objective at this
/// magnification has a depth of field of a few microns and would throw most of
/// a 10 mm animal out of focus; drawing it all sharp is a declared departure
/// from the optics, not an oversight.
struct Camera {
    var origin: SIMD3<Float>       // the centre of the ray grid
    var forward: SIMD3<Float>
    var right: SIMD3<Float>
    var up: SIMD3<Float>
    var halfWidth: Float           // world µm from centre to the left/right edge
    var halfHeight: Float

    /// A level camera looking along −Z at `centre`, with the field set by the
    /// pixel scale. Screen x is world x and screen y is world y, so the animal
    /// is aimed by posing the animal, not by rolling the camera.
    init(centre: SIMD3<Float>, micronsPerPixel: Float, width: Int, height: Int, standOff: Float) {
        let halfW: Float = Float(width) * micronsPerPixel * 0.5
        let halfH: Float = Float(height) * micronsPerPixel * 0.5
        self.origin = SIMD3(centre.x, centre.y, centre.z + standOff)
        self.forward = SIMD3(0, 0, -1)
        self.right = SIMD3(1, 0, 0)
        self.up = SIMD3(0, 1, 0)
        self.halfWidth = halfW
        self.halfHeight = halfH
    }

    /// Where a world point lands, in pixels from the top left.
    func project(_ p: SIMD3<Float>, width: Int, height: Int) -> SIMD2<Float> {
        let v: SIMD3<Float> = p - origin
        let x: Float = simd_dot(v, right) / halfWidth
        let y: Float = simd_dot(v, up) / halfHeight
        let px: Float = (x + 1) * 0.5 * Float(width)
        let py: Float = (1 - y) * 0.5 * Float(height)
        return SIMD2(px, py)
    }

    /// The ray for a sample at pixel coordinates (sx, sy).
    func ray(sx: Float, sy: Float, width: Int, height: Int) -> (SIMD3<Float>, SIMD3<Float>) {
        let ndcX: Float = sx / Float(width) * 2 - 1
        let ndcY: Float = 1 - sy / Float(height) * 2
        let o: SIMD3<Float> = origin + right * (ndcX * halfWidth) + up * (ndcY * halfHeight)
        return (o, forward)
    }
}

// MARK: - The backlight

/// Köhler illumination: a bright even field that falls off gently toward the
/// edges. The house gradient is deliberately broken here — a brightfield lamp
/// is not a sky, it is a disc of light behind the specimen.
struct Backlight {
    var centre: SIMD3<Float>
    var edge: SIMD3<Float>
    var falloff: Float

    static let standard = Backlight(centre: SIMD3(150, 222, 235) / 255,
                                    edge: SIMD3(95, 195, 215) / 255,
                                    falloff: 1.45)

    /// The lamp colour under a whole pixel. It is evaluated once per PIXEL, not
    /// once per sample: the lamp does not vary within a pixel, and keeping it
    /// per-pixel is what lets a ray through empty space return the backlight
    /// exactly, with no epsilon anywhere.
    func colour(px: Int, py: Int, width: Int, height: Int, halfWidth: Float, halfHeight: Float)
        -> SIMD3<Float> {
        let u: Float = (Float(px) + 0.5) / Float(width) * 2 - 1
        let v: Float = (Float(py) + 0.5) / Float(height) * 2 - 1
        let qx: Float = u * halfWidth
        let qy: Float = v * halfHeight
        let rMax: Float = (halfWidth * halfWidth + halfHeight * halfHeight).squareRoot()
        let r: Float = (qx * qx + qy * qy).squareRoot() / rMax
        let f: Float = pow(r, falloff)
        return centre + (edge - centre) * f
    }
}

// MARK: - Mutations, for the mutation check

/// Deliberate breakages, switched on with SWIM_MUTATE so that the tests can be
/// shown to catch them. Nothing here is ever on in a real render.
struct Mutations: OptionSet {
    let rawValue: UInt32
    static let noUnion = Mutations(rawValue: 1 << 0)     // sum intervals instead of merging
    static let noMailbox = Mutations(rawValue: 1 << 1)   // count a primitive once per grid box
    static let noClamp = Mutations(rawValue: 1 << 2)     // draw sub-pixel setae at true radius
    static let badPhase = Mutations(rawValue: 1 << 3)    // 1/10 of a cycle of lag, not 1/11

    static func fromEnvironment() -> Mutations {
        let text: String = ProcessInfo.processInfo.environment["SWIM_MUTATE"] ?? ""
        var out = Mutations([])
        for word in text.split(separator: ",") {
            switch word.trimmingCharacters(in: .whitespaces) {
            case "union": out.insert(.noUnion)
            case "mailbox": out.insert(.noMailbox)
            case "clamp": out.insert(.noClamp)
            case "phase": out.insert(.badPhase)
            default: break
            }
        }
        return out
    }
}

// MARK: - GPU-side structs

struct GPUCamera {
    var origin: SIMD4<Float>
    var forward: SIMD4<Float>
    var right: SIMD4<Float>
    var up: SIMD4<Float>
    var bgCentre: SIMD4<Float>
    var bgEdge: SIMD4<Float>
    var halfWidth: Float
    var halfHeight: Float
    var falloff: Float
    var pad: Float
    var width: UInt32
    var height: UInt32
    var primCount: UInt32
    var samplesPerSide: UInt32
    var useGrid: UInt32
    var mutations: UInt32
}

struct GPUGrid {
    var origin: SIMD4<Float>
    var cell: SIMD4<Float>
    var dims: SIMD4<UInt32>
}

/// How many intervals one ray may hold. A ray crossing the limb fan meets
/// about twenty; the kernel counts anything beyond this as an overflow and the
/// tests insist the real scene never produces one.
let maxIntervalsPerRay = 40

/// Mailboxing is a bitmask, one bit per primitive, so it is exact rather than
/// a "last ray seen" cache. 1,024 bits is 128 bytes per ray.
let mailboxCapacity = 1024

let swimKernelSource = """
    #include <metal_stdlib>
    using namespace metal;

    #define MAX_IV \(maxIntervalsPerRay)
    #define MAX_TISSUE 16
    #define MAILBOX_WORDS \(mailboxCapacity / 32)

    constant uint MUT_NO_UNION = 1u;
    constant uint MUT_NO_MAILBOX = 2u;

    struct Prim { float4 r0; float4 r1; float4 r2; float4 c; };
    struct Camera {
        float4 origin; float4 forward; float4 right; float4 up;
        float4 bgCentre; float4 bgEdge;
        float halfWidth; float halfHeight; float falloff; float pad;
        uint width; uint height; uint primCount; uint samplesPerSide;
        uint useGrid; uint mutations;
    };
    struct Grid { float4 origin; float4 cell; uint4 dims; };

    // The lamp under a whole pixel. Evaluated once per pixel, so a pixel the
    // animal misses entirely comes back as this value and nothing else.
    inline float3 backlight(constant Camera &cam, uint2 gid) {
        float u = (float(gid.x) + 0.5) / float(cam.width) * 2.0 - 1.0;
        float v = (float(gid.y) + 0.5) / float(cam.height) * 2.0 - 1.0;
        float qx = u * cam.halfWidth;
        float qy = v * cam.halfHeight;
        float rMax = sqrt(cam.halfWidth * cam.halfWidth + cam.halfHeight * cam.halfHeight);
        float r = sqrt(qx * qx + qy * qy) / rMax;
        float f = pow(r, cam.falloff);
        return cam.bgCentre.rgb + (cam.bgEdge.rgb - cam.bgCentre.rgb) * f;
    }

    // The general ellipsoid. M⁻¹ takes the ray into unit-sphere space; the
    // roots there are already world ray parameters, because an affine map
    // leaves t alone.
    inline bool ellipsoidInterval(float3 ro, float3 rd, device const Prim &p,
                                  thread float &t0, thread float &t1) {
        float3 rel = ro - p.c.xyz;
        float3 o = float3(dot(p.r0.xyz, rel), dot(p.r1.xyz, rel), dot(p.r2.xyz, rel));
        float3 d = float3(dot(p.r0.xyz, rd), dot(p.r1.xyz, rd), dot(p.r2.xyz, rd));
        float a = dot(d, d);
        if (a <= 0.0) return false;
        // The GEOMETRIC form, not b² − 4ac. The camera stands 40 mm off a
        // 10 µm feature, so |o| is four thousand times the radius and the
        // textbook discriminant is the difference of two numbers that agree
        // to seven digits — which is every digit a float has. Finding the
        // closest approach first and measuring the miss distance there keeps
        // the cancellation inside `perp`, where it is worth 1e-4 µm instead
        // of the whole answer.
        float tc = -dot(o, d) / a;
        float3 perp = o + d * tc;
        float disc = 1.0 - dot(perp, perp);
        if (disc < 0.0) return false;
        float dt = sqrt(disc / a);
        t0 = tc - dt;
        t1 = tc + dt;
        return true;
    }

    // A capsule is convex, so its entry is the earliest entry of its three
    // pieces (the barrel clipped by the end planes, and the two cap balls) and
    // its exit is the latest exit. `rd` must be a unit vector.
    inline bool capsuleInterval(float3 ro, float3 rd, device const Prim &p,
                                thread float &t0, thread float &t1) {
        float3 A = p.r0.xyz;
        float3 B = p.r1.xyz;
        float r = p.r0.w;
        float3 ba = B - A;
        float3 oa = ro - A;
        float baba = dot(ba, ba);
        float bard = dot(ba, rd);
        float baoa = dot(ba, oa);
        float lo = 1e30;
        float hi = -1e30;

        // The barrel, in the plane across the axis — again by closest
        // approach rather than by a discriminant, for the same reason.
        float3 axis = ba * rsqrt(max(baba, 1e-20));
        float3 dPerp = rd - axis * dot(rd, axis);
        float3 oPerp = oa - axis * dot(oa, axis);
        float aa = dot(dPerp, dPerp);
        if (aa > 1e-12) {
            float tc = -dot(oPerp, dPerp) / aa;
            float3 perp = oPerp + dPerp * tc;
            float disc = r * r - dot(perp, perp);
            if (disc >= 0.0) {
                float dt = sqrt(disc / aa);
                float u0 = tc - dt;
                float u1 = tc + dt;
                float s0 = -1e30;
                float s1 = 1e30;
                if (fabs(bard) > 1e-9) {
                    float v0 = -baoa / bard;
                    float v1 = (baba - baoa) / bard;
                    s0 = min(v0, v1);
                    s1 = max(v0, v1);
                } else if (baoa < 0.0 || baoa > baba) {
                    s0 = 1.0; s1 = -1.0;
                }
                float e0 = max(u0, s0);
                float e1 = min(u1, s1);
                if (e0 <= e1) { lo = min(lo, e0); hi = max(hi, e1); }
            }
        }
        for (int k = 0; k < 2; k++) {
            float3 ctr = (k == 0) ? A : B;
            float3 oc = ro - ctr;
            float tc = -dot(oc, rd);
            float3 perp = oc + rd * tc;
            float disc = r * r - dot(perp, perp);
            if (disc >= 0.0) {
                float s = sqrt(disc);
                lo = min(lo, tc - s);
                hi = max(hi, tc + s);
            }
        }
        if (hi < lo) return false;
        t0 = lo;
        t1 = hi;
        return true;
    }

    inline bool primInterval(float3 ro, float3 rd, device const Prim &p,
                             thread float &t0, thread float &t1) {
        if (p.c.w < 0.5) return ellipsoidInterval(ro, rd, p, t0, t1);
        return capsuleInterval(ro, rd, p, t0, t1);
    }

    // Keeps the interval list sorted by entry, which is what lets the merge
    // below be a single pass.
    inline void addInterval(thread float2 *iv, thread uchar *tis, thread uint &n,
                            float a, float b, uint tissue, thread uint &overflow,
                            thread uint &met) {
        if (b <= 0.0) return;
        float s = max(a, 0.0);
        if (b <= s) return;
        met++;
        if (n >= uint(MAX_IV)) { overflow++; return; }
        uint i = n;
        while (i > 0u && iv[i - 1].x > s) {
            iv[i] = iv[i - 1];
            tis[i] = tis[i - 1];
            i--;
        }
        iv[i] = float2(s, b);
        tis[i] = uchar(tissue);
        n++;
    }

    // Walks the grid, collecting EVERY interval the ray crosses — never
    // stopping at the first hit, because in transmitted light nothing is
    // hidden behind anything.
    inline void gather(float3 ro, float3 rd, device const Prim *prims,
                       device const uint *cellStart, device const uint *cellItems,
                       constant Grid &grid, constant Camera &cam,
                       thread float2 *iv, thread uchar *tis, thread uint &n,
                       thread uint &overflow, thread uint &met) {
        bool mailbox = (cam.mutations & MUT_NO_MAILBOX) == 0u;
        uint seen[MAILBOX_WORDS];
        for (uint i = 0; i < uint(MAILBOX_WORDS); i++) seen[i] = 0u;

        if (cam.useGrid == 0u) {
            for (uint i = 0; i < cam.primCount; i++) {
                float t0, t1;
                if (primInterval(ro, rd, prims[i], t0, t1)) {
                    addInterval(iv, tis, n, t0, t1, uint(prims[i].r2.w + 0.5), overflow, met);
                }
            }
            return;
        }

        float3 lo = grid.origin.xyz;
        float3 cell = grid.cell.xyz;
        int3 dims = int3(grid.dims.xyz);
        float3 hi = lo + float3(dims) * cell;
        float3 inv = 1.0 / rd;
        float3 ta = (lo - ro) * inv;
        float3 tb = (hi - ro) * inv;
        float3 tsmall = min(ta, tb);
        float3 tbig = max(ta, tb);
        float tEnter = max(max(tsmall.x, tsmall.y), max(tsmall.z, 0.0));
        float tExit = min(min(tbig.x, tbig.y), tbig.z);
        if (tEnter > tExit) return;

        float3 p = ro + rd * (tEnter + 1e-4);
        int3 c = clamp(int3(floor((p - lo) / cell)), int3(0), dims - 1);
        int3 stp;
        float3 tMax, tDelta;
        for (int k = 0; k < 3; k++) {
            if (rd[k] > 0.0) {
                stp[k] = 1;
                tMax[k] = (lo[k] + float(c[k] + 1) * cell[k] - ro[k]) * inv[k];
                tDelta[k] = cell[k] * inv[k];
            } else if (rd[k] < 0.0) {
                stp[k] = -1;
                tMax[k] = (lo[k] + float(c[k]) * cell[k] - ro[k]) * inv[k];
                tDelta[k] = -cell[k] * inv[k];
            } else {
                stp[k] = 0;
                tMax[k] = 1e30;
                tDelta[k] = 1e30;
            }
        }

        for (int guard = 0; guard < 4096; guard++) {
            uint index = (uint(c.z) * grid.dims.y + uint(c.y)) * grid.dims.x + uint(c.x);
            uint from = cellStart[index];
            uint to = cellStart[index + 1];
            for (uint i = from; i < to; i++) {
                uint s = cellItems[i];
                if (mailbox && s < uint(MAILBOX_WORDS) * 32u) {
                    uint word = s >> 5;
                    uint bit = 1u << (s & 31u);
                    if ((seen[word] & bit) != 0u) continue;
                    seen[word] |= bit;
                }
                float t0, t1;
                if (primInterval(ro, rd, prims[s], t0, t1)) {
                    addInterval(iv, tis, n, t0, t1, uint(prims[s].r2.w + 0.5), overflow, met);
                }
            }
            if (tMax.x < tMax.y && tMax.x < tMax.z) {
                c.x += stp.x; if (c.x < 0 || c.x >= dims.x) break; tMax.x += tDelta.x;
            } else if (tMax.y < tMax.z) {
                c.y += stp.y; if (c.y < 0 || c.y >= dims.y) break; tMax.y += tDelta.y;
            } else {
                c.z += stp.z; if (c.z < 0 || c.z >= dims.z) break; tMax.z += tDelta.z;
            }
        }
    }

    // Merge each tissue's intervals into a union, then integrate. The list is
    // already sorted by entry, and a list sorted overall is sorted within any
    // subset of it, so one pass with a running open run per tissue does it.
    inline float3 integrate(thread float2 *iv, thread uchar *tis, uint n,
                            device const float4 *sigma, constant Camera &cam) {
        float3 tau = float3(0.0);
        if ((cam.mutations & MUT_NO_UNION) != 0u) {
            for (uint i = 0; i < n; i++) tau += sigma[tis[i]].rgb * (iv[i].y - iv[i].x);
            return tau;
        }
        float curS[MAX_TISSUE];
        float curE[MAX_TISSUE];
        bool open[MAX_TISSUE];
        for (uint k = 0; k < uint(MAX_TISSUE); k++) { curS[k] = 0.0; curE[k] = 0.0; open[k] = false; }
        for (uint i = 0; i < n; i++) {
            uint k = uint(tis[i]);
            if (!open[k]) {
                curS[k] = iv[i].x; curE[k] = iv[i].y; open[k] = true;
            } else if (iv[i].x > curE[k]) {
                tau += sigma[k].rgb * (curE[k] - curS[k]);
                curS[k] = iv[i].x; curE[k] = iv[i].y;
            } else {
                curE[k] = max(curE[k], iv[i].y);
            }
        }
        for (uint k = 0; k < uint(MAX_TISSUE); k++) {
            if (open[k]) tau += sigma[k].rgb * (curE[k] - curS[k]);
        }
        return tau;
    }

    kernel void render(device uchar4 *pixels [[buffer(0)]],
                       constant Camera &cam [[buffer(1)]],
                       device const Prim *prims [[buffer(2)]],
                       device const uint *cellStart [[buffer(3)]],
                       device const uint *cellItems [[buffer(4)]],
                       constant Grid &grid [[buffer(5)]],
                       device const float4 *sigma [[buffer(6)]],
                       device float4 *tauOut [[buffer(7)]],
                       device atomic_uint *stats [[buffer(8)]],
                       uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= cam.width || gid.y >= cam.height) return;
        uint k = cam.samplesPerSide;
        float3 lamp = backlight(cam, gid);
        float3 sum = float3(0.0);
        float3 tauSum = float3(0.0);
        uint overflow = 0;
        uint met = 0;
        for (uint j = 0; j < k; j++) {
            for (uint i = 0; i < k; i++) {
                float sx = float(gid.x) + (float(i) + 0.5) / float(k);
                float sy = float(gid.y) + (float(j) + 0.5) / float(k);
                float ndcX = sx / float(cam.width) * 2.0 - 1.0;
                float ndcY = 1.0 - sy / float(cam.height) * 2.0;
                float3 ro = cam.origin.xyz + cam.right.xyz * (ndcX * cam.halfWidth)
                          + cam.up.xyz * (ndcY * cam.halfHeight);
                float3 rd = cam.forward.xyz;

                float2 iv[MAX_IV];
                uchar tis[MAX_IV];
                uint n = 0;
                gather(ro, rd, prims, cellStart, cellItems, grid, cam, iv, tis, n, overflow, met);
                float3 tau = integrate(iv, tis, n, sigma, cam);
                tauSum += tau;
                sum += lamp * exp(-tau);
            }
        }
        float inv = 1.0 / float(k * k);
        float3 colour = clamp(sum * inv, 0.0, 1.0);
        // w carries how many intervals this pixel's rays collected. That is
        // what mailboxing actually controls: without it the same primitive is
        // met once per grid box it touches, and although the per-tissue union
        // then merges the duplicates away — so τ does not change — the list
        // fills up more than twice as fast and eventually overflows.
        tauOut[gid.y * cam.width + gid.x] = float4(tauSum * inv, float(met));
        if (overflow > 0) atomic_fetch_add_explicit(stats, overflow, memory_order_relaxed);
        pixels[gid.y * cam.width + gid.x] = uchar4(uchar3(round(colour * 255.0)), 255);
    }
    """

enum RenderError: Error, CustomStringConvertible {
    case noMetalDevice
    case kernelCompile(String)
    case gpu(String)
    case tooManyPrimitives(Int)

    var description: String {
        switch self {
        case .noMetalDevice: return "no Metal GPU found"
        case .kernelCompile(let detail): return "could not compile the kernel: \(detail)"
        case .gpu(let detail): return "GPU error: \(detail)"
        case .tooManyPrimitives(let n):
            return "\(n) primitives, but the mailbox only holds \(mailboxCapacity)"
        }
    }
}

func findDevice() throws -> MTLDevice {
    if let device = MTLCreateSystemDefaultDevice() ?? MTLCopyAllDevices().first { return device }
    throw RenderError.noMetalDevice
}

// MARK: - The uniform grid

/// Which primitives lie in each box of a regular grid, flattened the way the
/// GPU wants it: `start[c]..<start[c+1]` indexes into `items`. Straight from
/// step 8a, with one change: a primitive here has a general box rather than a
/// sphere's, so the box comes from `bounds()`.
struct UniformGrid {
    var origin: SIMD3<Float>
    var cellSize: SIMD3<Float>
    var dims: SIMD3<Int>
    var start: [UInt32]
    var items: [UInt32]

    var cellCount: Int { dims.x * dims.y * dims.z }
    var averageOccupancy: Double {
        var used = 0
        for c in 0..<cellCount where start[c + 1] > start[c] { used += 1 }
        return used == 0 ? 0 : Double(items.count) / Double(used)
    }
    /// How many boxes the average primitive is filed in — the number that says
    /// how much work mailboxing is saving.
    var averageSpan: Double {
        items.isEmpty ? 0 : Double(items.count)
    }

    /// `density` scales the resolution: 1 aims at roughly one box per
    /// primitive (Pharr, Jakob & Humphreys, *Physically Based Rendering*, §4.4).
    init(prims: [GPUPrim], density: Float = 1, maxCells: Int = 4_000_000) {
        var lo = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
        var hi = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)
        var boxes: [(SIMD3<Float>, SIMD3<Float>)] = []
        boxes.reserveCapacity(prims.count)
        for p in prims {
            let b = p.bounds()
            boxes.append(b)
            lo = simd_min(lo, b.lo)
            hi = simd_max(hi, b.hi)
        }
        if prims.isEmpty || lo.x > hi.x { lo = .zero; hi = SIMD3(repeating: 1) }
        let pad: SIMD3<Float> = simd_max((hi - lo) * 0.001, SIMD3(repeating: 0.01))
        lo -= pad
        hi += pad
        let extent: SIMD3<Float> = hi - lo
        let longest: Float = max(extent.x, max(extent.y, extent.z))
        let n: Double = Double(max(prims.count, 1))
        let cube: Float = Float(pow(n, 1.0 / 3.0))
        let perUnit: Float = density * 3 * cube / longest
        var dims = SIMD3<Int>(1, 1, 1)
        for k in 0..<3 {
            let raw: Float = extent[k] * perUnit
            dims[k] = max(1, min(512, Int(raw.rounded())))
        }
        while dims.x * dims.y * dims.z > maxCells {
            let k = dims.x >= dims.y && dims.x >= dims.z ? 0 : (dims.y >= dims.z ? 1 : 2)
            dims[k] = max(1, dims[k] / 2)
        }
        self.origin = lo
        self.dims = dims
        self.cellSize = SIMD3(extent.x / Float(dims.x), extent.y / Float(dims.y),
                              extent.z / Float(dims.z))

        let cells = dims.x * dims.y * dims.z
        var counts = [UInt32](repeating: 0, count: cells)
        let gridOrigin = lo
        let cellSize = self.cellSize
        func span(_ b: (lo: SIMD3<Float>, hi: SIMD3<Float>)) -> (SIMD3<Int>, SIMD3<Int>) {
            var a = SIMD3<Int>(0, 0, 0)
            var z = SIMD3<Int>(0, 0, 0)
            for k in 0..<3 {
                let fa: Float = (b.lo[k] - gridOrigin[k]) / cellSize[k]
                let fz: Float = (b.hi[k] - gridOrigin[k]) / cellSize[k]
                a[k] = max(0, min(dims[k] - 1, Int(fa.rounded(.down))))
                z[k] = max(0, min(dims[k] - 1, Int(fz.rounded(.down))))
            }
            return (a, z)
        }
        for b in boxes {
            let (a, z) = span(b)
            for iz in a.z...z.z {
                for iy in a.y...z.y {
                    let row = (iz * dims.y + iy) * dims.x
                    for ix in a.x...z.x { counts[row + ix] += 1 }
                }
            }
        }
        var start = [UInt32](repeating: 0, count: cells + 1)
        var running: UInt32 = 0
        for c in 0..<cells {
            start[c] = running
            running += counts[c]
        }
        start[cells] = running
        var items = [UInt32](repeating: 0, count: Int(running))
        var cursor = start
        for (i, b) in boxes.enumerated() {
            let (a, z) = span(b)
            for iz in a.z...z.z {
                for iy in a.y...z.y {
                    let row = (iz * dims.y + iy) * dims.x
                    for ix in a.x...z.x {
                        items[Int(cursor[row + ix])] = UInt32(i)
                        cursor[row + ix] += 1
                    }
                }
            }
        }
        self.start = start
        self.items = items
    }

    /// The primitives filed in the box containing `p` (for tests).
    func prims(at p: SIMD3<Float>) -> [Int] {
        var c = SIMD3<Int>(0, 0, 0)
        for k in 0..<3 {
            let f: Float = (p[k] - origin[k]) / cellSize[k]
            let v = Int(f.rounded(.down))
            if v < 0 || v >= dims[k] { return [] }
            c[k] = v
        }
        let index = (c.z * dims.y + c.y) * dims.x + c.x
        return (Int(start[index])..<Int(start[index + 1])).map { Int(items[$0]) }
    }
}

// MARK: - A CPU reference for the same physics

/// The same Beer–Lambert integral on the CPU, brute force, no grid. The tests
/// use it because it is exact and cheap, and one test renders the same scene
/// both ways and checks they agree — which is what makes the CPU version a
/// reference and not a second opinion.
struct CPUTrace {
    var tau: SIMD3<Float>
    var intervals: Int
}

func cpuOpticalDepth(origin: SIMD3<Float>, direction: SIMD3<Float>,
                     prims: [GPUPrim], sigma: [SIMD3<Float>],
                     mutations: Mutations = []) -> CPUTrace {
    var list: [(Float, Float, Int)] = []
    list.reserveCapacity(32)
    for p in prims {
        guard let (a, b) = cpuInterval(origin: origin, direction: direction, prim: p) else { continue }
        if b <= 0 { continue }
        let s: Float = max(a, 0)
        if b <= s { continue }
        list.append((s, b, p.tissue))
    }
    list.sort { $0.0 < $1.0 }
    var tau = SIMD3<Float>(repeating: 0)
    if mutations.contains(.noUnion) {
        for (a, b, k) in list { tau += sigma[k] * (b - a) }
        return CPUTrace(tau: tau, intervals: list.count)
    }
    var curS = [Float](repeating: 0, count: sigma.count)
    var curE = [Float](repeating: 0, count: sigma.count)
    var open = [Bool](repeating: false, count: sigma.count)
    for (a, b, k) in list {
        if !open[k] {
            curS[k] = a; curE[k] = b; open[k] = true
        } else if a > curE[k] {
            tau += sigma[k] * (curE[k] - curS[k])
            curS[k] = a; curE[k] = b
        } else {
            curE[k] = max(curE[k], b)
        }
    }
    for k in 0..<sigma.count where open[k] {
        tau += sigma[k] * (curE[k] - curS[k])
    }
    return CPUTrace(tau: tau, intervals: list.count)
}

/// The interval a ray spends inside one primitive. `direction` must be a unit
/// vector, as on the GPU.
func cpuInterval(origin ro: SIMD3<Float>, direction rd: SIMD3<Float>,
                 prim p: GPUPrim) -> (Float, Float)? {
    if p.kind == .ellipsoid {
        let rel: SIMD3<Float> = ro - SIMD3(p.c.x, p.c.y, p.c.z)
        let r0 = SIMD3<Float>(p.r0.x, p.r0.y, p.r0.z)
        let r1 = SIMD3<Float>(p.r1.x, p.r1.y, p.r1.z)
        let r2 = SIMD3<Float>(p.r2.x, p.r2.y, p.r2.z)
        let o = SIMD3<Float>(simd_dot(r0, rel), simd_dot(r1, rel), simd_dot(r2, rel))
        let d = SIMD3<Float>(simd_dot(r0, rd), simd_dot(r1, rd), simd_dot(r2, rd))
        let a: Float = simd_dot(d, d)
        if a <= 0 { return nil }
        let tc: Float = -simd_dot(o, d) / a
        let perp: SIMD3<Float> = o + d * tc
        let disc: Float = 1 - simd_dot(perp, perp)
        if disc < 0 { return nil }
        let dt: Float = (disc / a).squareRoot()
        return (tc - dt, tc + dt)
    }
    let A = SIMD3<Float>(p.r0.x, p.r0.y, p.r0.z)
    let B = SIMD3<Float>(p.r1.x, p.r1.y, p.r1.z)
    let r: Float = p.r0.w
    let ba: SIMD3<Float> = B - A
    let oa: SIMD3<Float> = ro - A
    let baba: Float = simd_dot(ba, ba)
    let bard: Float = simd_dot(ba, rd)
    let baoa: Float = simd_dot(ba, oa)
    var lo: Float = .greatestFiniteMagnitude
    var hi: Float = -.greatestFiniteMagnitude

    let axis: SIMD3<Float> = ba / max(baba, 1e-20).squareRoot()
    let dPerp: SIMD3<Float> = rd - axis * simd_dot(rd, axis)
    let oPerp: SIMD3<Float> = oa - axis * simd_dot(oa, axis)
    let aa: Float = simd_dot(dPerp, dPerp)
    if aa > 1e-12 {
        let tc: Float = -simd_dot(oPerp, dPerp) / aa
        let perp: SIMD3<Float> = oPerp + dPerp * tc
        let disc: Float = r * r - simd_dot(perp, perp)
        if disc >= 0 {
            let dt: Float = (disc / aa).squareRoot()
            let u0: Float = tc - dt
            let u1: Float = tc + dt
            var s0: Float = -.greatestFiniteMagnitude
            var s1: Float = .greatestFiniteMagnitude
            if abs(bard) > 1e-9 {
                let v0: Float = -baoa / bard
                let v1: Float = (baba - baoa) / bard
                s0 = min(v0, v1)
                s1 = max(v0, v1)
            } else if baoa < 0 || baoa > baba {
                s0 = 1; s1 = -1
            }
            let e0: Float = max(u0, s0)
            let e1: Float = min(u1, s1)
            if e0 <= e1 { lo = min(lo, e0); hi = max(hi, e1) }
        }
    }
    for k in 0..<2 {
        let ctr: SIMD3<Float> = k == 0 ? A : B
        let oc: SIMD3<Float> = ro - ctr
        let tc: Float = -simd_dot(oc, rd)
        let perp: SIMD3<Float> = oc + rd * tc
        let disc: Float = r * r - simd_dot(perp, perp)
        if disc >= 0 {
            let s: Float = disc.squareRoot()
            lo = min(lo, tc - s)
            hi = max(hi, tc + s)
        }
    }
    if hi < lo { return nil }
    return (lo, hi)
}

/// Is `p` inside this primitive? Used by the interpenetration tests.
func cpuContains(_ p: GPUPrim, point q: SIMD3<Float>) -> Bool {
    if p.kind == .ellipsoid {
        let rel: SIMD3<Float> = q - SIMD3(p.c.x, p.c.y, p.c.z)
        let r0 = SIMD3<Float>(p.r0.x, p.r0.y, p.r0.z)
        let r1 = SIMD3<Float>(p.r1.x, p.r1.y, p.r1.z)
        let r2 = SIMD3<Float>(p.r2.x, p.r2.y, p.r2.z)
        let u = SIMD3<Float>(simd_dot(r0, rel), simd_dot(r1, rel), simd_dot(r2, rel))
        return simd_length_squared(u) < 1
    }
    let A = SIMD3<Float>(p.r0.x, p.r0.y, p.r0.z)
    let B = SIMD3<Float>(p.r1.x, p.r1.y, p.r1.z)
    let ba: SIMD3<Float> = B - A
    let denom: Float = max(simd_dot(ba, ba), 1e-12)
    var t: Float = simd_dot(q - A, ba) / denom
    t = min(max(t, 0), 1)
    let closest: SIMD3<Float> = A + ba * t
    return simd_distance(q, closest) < p.r0.w
}

// MARK: - The renderer

final class SceneRenderer {
    let device: MTLDevice
    private let pipeline: MTLComputePipelineState
    private let queue: MTLCommandQueue
    private var primBuffer: MTLBuffer?
    private var startBuffer: MTLBuffer?
    private var itemBuffer: MTLBuffer?
    private var sigmaBuffer: MTLBuffer?
    private var tauBuffer: MTLBuffer?
    private var statsBuffer: MTLBuffer?
    private(set) var grid: UniformGrid?
    private(set) var overflowCount: UInt32 = 0
    private(set) var lastBuildSeconds: Double = 0

    init(device: MTLDevice) throws {
        self.device = device
        do {
            let options = MTLCompileOptions()
            // Off, as in every step since 3: fast math would let the compiler
            // reassociate the intersection arithmetic, and here it would also
            // be free to contract the interval endpoints, which is exactly
            // where a union merge must not drift.
            options.fastMathEnabled = false
            let library = try device.makeLibrary(source: swimKernelSource, options: options)
            guard let function = library.makeFunction(name: "render") else {
                throw RenderError.kernelCompile("no kernel named render")
            }
            pipeline = try device.makeComputePipelineState(function: function)
        } catch let error as RenderError {
            throw error
        } catch {
            throw RenderError.kernelCompile(error.localizedDescription)
        }
        guard let queue = device.makeCommandQueue() else { throw RenderError.gpu("no command queue") }
        self.queue = queue
    }

    func buildGrid(_ prims: [GPUPrim], density: Float = 1) throws {
        let t0 = Date()
        let g = UniformGrid(prims: prims, density: density)
        grid = g
        startBuffer = try fill(startBuffer, with: g.start)
        itemBuffer = try fill(itemBuffer, with: g.items.isEmpty ? [UInt32(0)] : g.items)
        lastBuildSeconds = Date().timeIntervalSince(t0)
    }

    /// Ray-traces into the top `viewHeight` rows of `frame`. Returns GPU seconds.
    @discardableResult
    func render(prims: [GPUPrim], sigma: [SIMD3<Float>], camera: Camera, backlight: Backlight,
                into frame: MTLBuffer, width: Int, viewHeight: Int,
                samplesPerSide: Int = 2, useGrid: Bool = true,
                mutations: Mutations = []) throws -> Double {
        if prims.count > mailboxCapacity { throw RenderError.tooManyPrimitives(prims.count) }
        var cam = GPUCamera(origin: SIMD4(camera.origin, 1),
                            forward: SIMD4(camera.forward, 0),
                            right: SIMD4(camera.right, 0),
                            up: SIMD4(camera.up, 0),
                            bgCentre: SIMD4(backlight.centre, 0),
                            bgEdge: SIMD4(backlight.edge, 0),
                            halfWidth: camera.halfWidth, halfHeight: camera.halfHeight,
                            falloff: backlight.falloff, pad: 0,
                            width: UInt32(width), height: UInt32(viewHeight),
                            primCount: UInt32(prims.count),
                            samplesPerSide: UInt32(samplesPerSide),
                            useGrid: useGrid ? 1 : 0,
                            mutations: mutations.rawValue)
        if useGrid && grid == nil { throw RenderError.gpu("no grid has been built") }
        var g = GPUGrid(origin: SIMD4(grid?.origin ?? .zero, 0),
                        cell: SIMD4(grid?.cellSize ?? SIMD3(repeating: 1), 0),
                        dims: SIMD4(UInt32(grid?.dims.x ?? 1), UInt32(grid?.dims.y ?? 1),
                                    UInt32(grid?.dims.z ?? 1), 0))
        let list = prims.isEmpty
            ? [GPUPrim.sphere(centre: SIMD3(0, 0, 1e9), radius: 0.001, tissue: .body)] : prims
        primBuffer = try fill(primBuffer, with: list)
        let sig: [SIMD4<Float>] = (0..<16).map { i in
            i < sigma.count ? SIMD4(sigma[i], 0) : SIMD4(repeating: 0)
        }
        sigmaBuffer = try fill(sigmaBuffer, with: sig)
        if startBuffer == nil { startBuffer = try fill(nil, with: [UInt32(0), UInt32(0)]) }
        if itemBuffer == nil { itemBuffer = try fill(nil, with: [UInt32(0)]) }
        let pixels = width * viewHeight
        if tauBuffer == nil || tauBuffer!.length < pixels * 16 {
            tauBuffer = device.makeBuffer(length: pixels * 16, options: .storageModeShared)
        }
        if statsBuffer == nil {
            statsBuffer = device.makeBuffer(length: 16, options: .storageModeShared)
        }
        statsBuffer!.contents().assumingMemoryBound(to: UInt32.self).pointee = 0

        guard let commands = queue.makeCommandBuffer(),
              let e = commands.makeComputeCommandEncoder() else {
            throw RenderError.gpu("could not create a command encoder")
        }
        e.setComputePipelineState(pipeline)
        e.setBuffer(frame, offset: 0, index: 0)
        e.setBytes(&cam, length: MemoryLayout<GPUCamera>.stride, index: 1)
        e.setBuffer(primBuffer, offset: 0, index: 2)
        e.setBuffer(startBuffer, offset: 0, index: 3)
        e.setBuffer(itemBuffer, offset: 0, index: 4)
        e.setBytes(&g, length: MemoryLayout<GPUGrid>.stride, index: 5)
        e.setBuffer(sigmaBuffer, offset: 0, index: 6)
        e.setBuffer(tauBuffer, offset: 0, index: 7)
        e.setBuffer(statsBuffer, offset: 0, index: 8)
        let w = pipeline.threadExecutionWidth
        let group = MTLSize(width: w, height: pipeline.maxTotalThreadsPerThreadgroup / w, depth: 1)
        e.dispatchThreads(MTLSize(width: width, height: viewHeight, depth: 1),
                          threadsPerThreadgroup: group)
        e.endEncoding()
        commands.commit()
        commands.waitUntilCompleted()
        if let error = commands.error { throw RenderError.gpu(error.localizedDescription) }
        overflowCount = statsBuffer!.contents().assumingMemoryBound(to: UInt32.self).pointee
        return commands.gpuEndTime - commands.gpuStartTime
    }

    /// The optical depth the last render measured at one pixel, averaged over
    /// its samples. The tests read this rather than un-doing the 8-bit colour.
    func tau(atX x: Int, y: Int, width: Int) -> SIMD3<Float> {
        guard let b = tauBuffer else { return .zero }
        let p = b.contents().assumingMemoryBound(to: SIMD4<Float>.self)
        let v = p[y * width + x]
        return SIMD3(v.x, v.y, v.z)
    }

    /// How many intervals the rays of one pixel collected — the count the
    /// mailbox exists to hold down.
    func intervalsMet(atX x: Int, y: Int, width: Int) -> Int {
        guard let b = tauBuffer else { return 0 }
        let p = b.contents().assumingMemoryBound(to: SIMD4<Float>.self)
        return Int(p[y * width + x].w.rounded())
    }

    private func fill<T>(_ buffer: MTLBuffer?, with items: [T]) throws -> MTLBuffer {
        let bytes = max(items.count * MemoryLayout<T>.stride, 16)
        var target = buffer
        if target == nil || target!.length < bytes {
            target = device.makeBuffer(length: bytes, options: .storageModeShared)
        }
        guard let out = target else { throw RenderError.gpu("could not allocate a scene buffer") }
        items.withUnsafeBytes { out.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
        return out
    }
}
