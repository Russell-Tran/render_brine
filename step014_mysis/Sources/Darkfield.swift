// Darkfield: step 13's ray traversal with the other operator.
//
// Step 13 was brightfield. The lamp was behind the specimen, the background WAS
// the light source, and the picture was
//
//     I = I_bg · exp(−∫ σ_a dℓ)
//
// Tissue could only subtract. Here the direct beam is stopped before the
// objective — a dark stop in the condenser sends the illumination in as a
// hollow cone whose zeroth order misses the front lens entirely — so the only
// light that reaches the sensor is light the specimen SCATTERED out of that
// cone. In Fourier terms darkfield deletes the zeroth order from the
// diffraction pattern at the back focal plane and builds the image out of the
// higher orders alone. So
//
//     I = ∫ σ_s(ℓ) · g(θ) dℓ,   I_bg = 0
//
// Same traversal, emission instead of absorption, and a background that is
// exactly black rather than nearly black.
//
// Three things make that look like the photograph rather than like a glow
// filter laid over step 13.
//
// 1. SCATTERING LIVES AT INTERFACES, NOT IN VOLUMES. Light scatters off the
//    refractive-index discontinuity at a surface, not out of the middle of a
//    uniform medium. So σ_s is deposited where the ray ENTERS or LEAVES a
//    body and nowhere along the inside. That single choice is what gives
//    bright outlines, dim middles, and a body that reads as a stack of shells
//    instead of a lump of jelly. It also means the picture is a count of
//    surfaces crossed, which is why the interval list matters more here than
//    it did in step 13: a dropped interval is a missing outline.
//
// 2. GRAZING INTERFACES SCATTER MORE. A surface seen edge-on presents a long
//    slice of boundary to the ray and throws light sideways into the
//    objective; a surface seen face-on throws it forward, into the stop. The
//    weight here is
//
//        g(θ) = (1 − |n·v|)^k
//
//    which is 0 face-on and 1 edge-on. It is a stand-in for a real angular
//    scattering distribution and is labelled MODEL on the bar. It is what
//    makes the carapace rim and the joints between abdominal somites the
//    brightest lines in the frame, exactly as they are in a real darkfield
//    shot.
//
// 3. THE WAVELENGTH BIAS IS DERIVED, NOT PICKED. Scattering by structures
//    small against λ goes as λ⁻⁴ (Rayleigh; Strutt 1871). So σ_s carries a
//    wavelength EXPONENT rather than three hand-chosen RGB numbers, and the
//    pale blue-white of the body falls out of the exponent instead of being
//    painted on. Pigmented tissue multiplies that by the pigment's own
//    spectrum, because the light the granule throws sideways has been through
//    the pigment on the way out.
//
// The union is still per tissue and is still the point: two coincident shells
// are ONE surface pair, not two, so a body drawn from overlapping lobes does
// not grow a bright seam at every joint. The deposit happens at the union's
// boundaries, never at a primitive's own boundary that lies inside another
// primitive of the same tissue.

import Foundation
import Metal
import simd

// MARK: - Tissues

/// σ_s belongs to the tissue, not to the primitive — that is what makes the
/// per-tissue union well defined, exactly as in step 13.
enum MTissue: Int, CaseIterable {
    case cuticle = 0      // carapace, pleonites, telson, uropods: thin hard shell
    case innerWall        // the body wall inside the carapace — the second shell
    case pereopod         // thoracopod protopod and endopod
    case exopod           // the natatory paddle that beats
    case seta             // the setal fringes
    case antenna          // antennules, antennal flagella, the antennal scale
    case eye              // the dark screening pigment of the compound eye
    case tapetum          // the bright reflective ring around it
    case hepatopancreas   // the orange digestive gland, and the stomach
    case gut              // the tube behind it
    case marsupium        // the brood pouch wall — the oostegites
    case embryo           // what is in the pouch
    case statolith        // the dense bead in the uropod endopod
    case snow             // marine snow and detritus drifting past
    case floc             // the larger, brighter aggregates

    var name: String {
        switch self {
        case .cuticle: return "cuticle"
        case .innerWall: return "inner body wall"
        case .pereopod: return "pereopod"
        case .exopod: return "exopod"
        case .seta: return "seta"
        case .antenna: return "antenna"
        case .eye: return "compound eye"
        case .tapetum: return "tapetum"
        case .hepatopancreas: return "hepatopancreas"
        case .gut: return "gut"
        case .marsupium: return "marsupium"
        case .embryo: return "embryo"
        case .statolith: return "statolith"
        case .snow: return "marine snow"
        case .floc: return "floc"
        }
    }

    /// Cuticle in the anatomical sense: the chitinous exoskeleton and the
    /// appendages made of it. These are the surfaces the Rayleigh test looks
    /// at, because they carry no pigment of their own.
    var isCuticle: Bool {
        switch self {
        case .cuticle, .innerWall, .pereopod, .exopod, .seta, .antenna, .marsupium: return true
        default: return false
        }
    }
}

let mTissueCount = MTissue.allCases.count

// MARK: - GPUPrim, with this step's tissue numbering
//
// The primitive itself, its M⁻¹ packing, its bounds and its closest-approach
// intersection all come from step 13 unchanged. Only the tissue index differs,
// so these three build through step 13's own factories and then overwrite the
// one field that carries it.

extension GPUPrim {
    static func ell(_ centre: SIMD3<Float>, _ m: simd_float3x3, _ t: MTissue) -> GPUPrim {
        var p: GPUPrim = GPUPrim.ellipsoid(centre: centre, m: m, tissue: .body)
        p.r2.w = Float(t.rawValue)
        return p
    }

    static func sph(_ centre: SIMD3<Float>, _ radius: Float, _ t: MTissue) -> GPUPrim {
        var p: GPUPrim = GPUPrim.sphere(centre: centre, radius: radius, tissue: .body)
        p.r2.w = Float(t.rawValue)
        return p
    }

    static func cap(_ a: SIMD3<Float>, _ b: SIMD3<Float>, _ radius: Float,
                    _ t: MTissue) -> GPUPrim {
        var p: GPUPrim = GPUPrim.capsule(from: a, to: b, radius: radius, tissue: .body)
        p.r2.w = Float(t.rawValue)
        return p
    }

    /// The outward normal at a point on this primitive's surface. For an
    /// ellipsoid p = c + M u the gradient of |M⁻¹(p−c)|² is 2 M⁻ᵀ u, and M⁻ᵀ's
    /// columns are M⁻¹'s rows — which are exactly r0, r1, r2 as packed.
    func normal(at q: SIMD3<Float>) -> SIMD3<Float> {
        let r0 = SIMD3<Float>(r0.x, r0.y, r0.z)
        let r1 = SIMD3<Float>(r1.x, r1.y, r1.z)
        let r2 = SIMD3<Float>(r2.x, r2.y, r2.z)
        if kind == .ellipsoid {
            let rel: SIMD3<Float> = q - SIMD3(c.x, c.y, c.z)
            let ux: Float = simd_dot(r0, rel)
            let uy: Float = simd_dot(r1, rel)
            let uz: Float = simd_dot(r2, rel)
            let g: SIMD3<Float> = r0 * ux + r1 * uy + r2 * uz
            let len: Float = simd_length(g)
            return len > 0 ? g / len : SIMD3(0, 0, 1)
        }
        let a = SIMD3<Float>(self.r0.x, self.r0.y, self.r0.z)
        let b = SIMD3<Float>(self.r1.x, self.r1.y, self.r1.z)
        let ba: SIMD3<Float> = b - a
        let denom: Float = max(simd_dot(ba, ba), 1e-12)
        var s: Float = simd_dot(q - a, ba) / denom
        s = min(max(s, 0), 1)
        let closest: SIMD3<Float> = a + ba * s
        let g: SIMD3<Float> = q - closest
        let len: Float = simd_length(g)
        return len > 0 ? g / len : SIMD3(0, 0, 1)
    }
}

// MARK: - The wavelength law
//
// One exponent, not three numbers. σ_s(λ) ∝ λ^(−p), p = 4 for structures small
// against the wavelength (Rayleigh) and smaller for bodies that are not. The
// three channels are evaluated at the sRGB primaries' band centres and
// normalised on green, so p = 0 is exactly neutral and p = 4 is the blue-white
// the body actually has.

let bandCentresNanometres = SIMD3<Float>(612, 549, 465)
/// Green, the channel everything is normalised against; also, to within a
/// nanometre or two, where this animal's own eye peaks (520 nm, Jokela-Määttä
/// et al. 2005 — a coincidence, but a pleasing one).
let referenceNanometres: Float = 549

/// (λ_ref / λ_c)^p, channel by channel.
func rayleighWeights(exponent p: Float) -> SIMD3<Float> {
    let r: Float = pow(referenceNanometres / bandCentresNanometres.x, p)
    let g: Float = pow(referenceNanometres / bandCentresNanometres.y, p)
    let b: Float = pow(referenceNanometres / bandCentresNanometres.z, p)
    return SIMD3(r, g, b)
}

/// One tissue's scattering strength: an amplitude, a wavelength exponent, and
/// the pigment that filters what it throws sideways.
struct Scatterer {
    var amplitude: Float
    var exponent: Float
    var tint: SIMD3<Float>

    var sigma: SIMD3<Float> {
        let w: SIMD3<Float> = rayleighWeights(exponent: exponent)
        return w * tint * amplitude
    }
}

// MARK: - Mutations

/// Deliberate breakages, switched on with MYSIS_MUTATE so the tests can be
/// shown to catch them. Nothing here is ever on in a real render.
struct MMutations: OptionSet {
    let rawValue: UInt32
    /// Every interface scatters the same however the ray meets it.
    static let flatGrazing = MMutations(rawValue: 1 << 0)
    /// The swept-bound mask is built from frame 0 alone, so it stops being a
    /// bound on the other 119.
    static let frameZeroMask = MMutations(rawValue: 1 << 1)
    /// The beat phase is reduced modulo the wrong number of frames, so the
    /// loop no longer closes.
    static let badLoopPhase = MMutations(rawValue: 1 << 2)
    /// The metachronal wave runs head to tail instead of tail to head.
    static let reversedWave = MMutations(rawValue: 1 << 3)
    /// Intervals summed instead of merged, so coincident shells double.
    static let noUnion = MMutations(rawValue: 1 << 4)

    static func fromEnvironment() -> MMutations {
        let text: String = ProcessInfo.processInfo.environment["MYSIS_MUTATE"] ?? ""
        var out = MMutations([])
        for word in text.split(separator: ",") {
            switch word.trimmingCharacters(in: .whitespaces) {
            case "grazing": out.insert(.flatGrazing)
            case "mask": out.insert(.frameZeroMask)
            case "phase": out.insert(.badLoopPhase)
            case "wave": out.insert(.reversedWave)
            case "union": out.insert(.noUnion)
            default: break
            }
        }
        return out
    }
}

// MARK: - GPU-side structs

struct DarkCamera {
    var origin: SIMD4<Float>
    var forward: SIMD4<Float>
    var right: SIMD4<Float>
    var up: SIMD4<Float>
    var halfWidth: Float
    var halfHeight: Float
    var grazingExponent: Float
    var exposure: Float
    var width: UInt32
    var height: UInt32
    var primCount: UInt32
    var samplesPerSide: UInt32
    var useGrid: UInt32
    var mutations: UInt32
    var useMask: UInt32
    var tileShift: UInt32
    var tilesX: UInt32
    var tilesY: UInt32
    var pad0: UInt32
    var pad1: UInt32
}

struct DarkGrid {
    var origin: SIMD4<Float>
    var cell: SIMD4<Float>
    var dims: SIMD4<UInt32>
}

/// How many intervals one ray may hold. A ray through the limb fan, the
/// carapace, the marsupium and a few flakes of snow meets about thirty; the
/// kernel counts anything past this as an overflow and a test insists the real
/// render never produces one.
let maxSurfacesPerRay = 48

/// The grazing exponent k in (1 − |n·v|)^k.
let grazingExponent: Float = 2.4

/// The one free number in the tone curve: colour = 1 − exp(−exposure · I).
/// Monotone, so it cannot reorder anything the physics said, and exactly zero
/// at I = 0, so black stays black with no epsilon.
let darkfieldExposure: Float = 1.0

/// 16 × 16 pixels to a tile of the swept-bound mask.
let maskTileShift: UInt32 = 4
let maskTileSize = 1 << Int(maskTileShift)

let darkfieldKernelSource = """
    #include <metal_stdlib>
    using namespace metal;

    #define MAX_IV \(maxSurfacesPerRay)
    #define MAX_TISSUE 16
    #define MAILBOX_WORDS \(mailboxCapacity / 32)

    constant uint MUT_FLAT_GRAZING = 1u;
    constant uint MUT_NO_UNION = 16u;

    struct Prim { float4 r0; float4 r1; float4 r2; float4 c; };
    struct Camera {
        float4 origin; float4 forward; float4 right; float4 up;
        float halfWidth; float halfHeight; float grazingExponent; float exposure;
        uint width; uint height; uint primCount; uint samplesPerSide;
        uint useGrid; uint mutations; uint useMask; uint tileShift;
        uint tilesX; uint tilesY; uint pad0; uint pad1;
    };
    struct Grid { float4 origin; float4 cell; uint4 dims; };

    // ---- intersections -----------------------------------------------------
    // Both of these are step 13's, unchanged, and for step 13's reason: the
    // camera stands 40 mm off a 200 µm feature, so |o| is hundreds of times the
    // radius and the textbook discriminant b² − ac is the difference of two
    // numbers that agree to seven digits — every digit a float has. Finding the
    // closest approach first and measuring the miss distance there keeps the
    // cancellation inside `perp`, where it costs 1e-4 µm instead of the answer.

    inline bool ellipsoidInterval(float3 ro, float3 rd, device const Prim &p,
                                  thread float &t0, thread float &t1) {
        float3 rel = ro - p.c.xyz;
        float3 o = float3(dot(p.r0.xyz, rel), dot(p.r1.xyz, rel), dot(p.r2.xyz, rel));
        float3 d = float3(dot(p.r0.xyz, rd), dot(p.r1.xyz, rd), dot(p.r2.xyz, rd));
        float a = dot(d, d);
        if (a <= 0.0) return false;
        float tc = -dot(o, d) / a;
        float3 perp = o + d * tc;
        float disc = 1.0 - dot(perp, perp);
        if (disc < 0.0) return false;
        float dt = sqrt(disc / a);
        t0 = tc - dt;
        t1 = tc + dt;
        return true;
    }

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

    // The outward normal, which brightfield never needed and darkfield cannot
    // do without: the whole picture is the angle between this and the ray.
    inline float3 primNormal(device const Prim &p, float3 q) {
        if (p.c.w < 0.5) {
            float3 rel = q - p.c.xyz;
            float3 u = float3(dot(p.r0.xyz, rel), dot(p.r1.xyz, rel), dot(p.r2.xyz, rel));
            float3 g = p.r0.xyz * u.x + p.r1.xyz * u.y + p.r2.xyz * u.z;
            return normalize(g);
        }
        float3 A = p.r0.xyz;
        float3 B = p.r1.xyz;
        float3 ba = B - A;
        float s = clamp(dot(q - A, ba) / max(dot(ba, ba), 1e-12), 0.0, 1.0);
        return normalize(q - (A + ba * s));
    }

    // g(θ). Face-on is 0, edge-on is 1. The mutation makes it 1 everywhere,
    // which is the same picture with every rim gone.
    inline float grazing(constant Camera &cam, float3 n, float3 v) {
        if ((cam.mutations & MUT_FLAT_GRAZING) != 0u) return 1.0;
        float c = fabs(dot(n, v));
        float e = 1.0 - min(c, 1.0);
        return pow(e, cam.grazingExponent);
    }

    // ---- the ordered multi-hit list ---------------------------------------
    // Step 13's, with one field added: which primitive each interval came from,
    // so that after the union merge the surviving boundaries still know whose
    // surface they are and can be asked for a normal.

    inline void addInterval(thread float2 *iv, thread uchar *tis, thread ushort *pid,
                            thread uint &n, float a, float b, uint tissue, uint prim,
                            thread uint &overflow, thread uint &met) {
        if (b <= 0.0) return;
        float s = max(a, 0.0);
        if (b <= s) return;
        met++;
        if (n >= uint(MAX_IV)) { overflow++; return; }
        uint i = n;
        while (i > 0u && iv[i - 1].x > s) {
            iv[i] = iv[i - 1];
            tis[i] = tis[i - 1];
            pid[i] = pid[i - 1];
            i--;
        }
        iv[i] = float2(s, b);
        tis[i] = uchar(tissue);
        pid[i] = ushort(prim);
        n++;
    }

    inline void gather(float3 ro, float3 rd, device const Prim *prims,
                       device const uint *cellStart, device const uint *cellItems,
                       constant Grid &grid, constant Camera &cam,
                       thread float2 *iv, thread uchar *tis, thread ushort *pid,
                       thread uint &n, thread uint &overflow, thread uint &met) {
        uint seen[MAILBOX_WORDS];
        for (uint i = 0; i < uint(MAILBOX_WORDS); i++) seen[i] = 0u;

        if (cam.useGrid == 0u) {
            for (uint i = 0; i < cam.primCount; i++) {
                float t0, t1;
                if (primInterval(ro, rd, prims[i], t0, t1)) {
                    addInterval(iv, tis, pid, n, t0, t1,
                                uint(prims[i].r2.w + 0.5), i, overflow, met);
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
                if (s < uint(MAILBOX_WORDS) * 32u) {
                    uint word = s >> 5;
                    uint bit = 1u << (s & 31u);
                    if ((seen[word] & bit) != 0u) continue;
                    seen[word] |= bit;
                }
                float t0, t1;
                if (primInterval(ro, rd, prims[s], t0, t1)) {
                    addInterval(iv, tis, pid, n, t0, t1,
                                uint(prims[s].r2.w + 0.5), s, overflow, met);
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

    // ---- the darkfield integral -------------------------------------------
    // Merge each tissue's intervals into a union, then deposit at the union's
    // two ends and NOWHERE in between. What is inside a body is not a surface,
    // whatever primitive drew it.

    inline float3 depositAt(float3 ro, float3 rd, device const Prim *prims, uint prim,
                            float t, float3 sigma, constant Camera &cam) {
        float3 q = ro + rd * t;
        float3 n = primNormal(prims[prim], q);
        return sigma * grazing(cam, n, rd);
    }

    inline float3 integrate(float3 ro, float3 rd, device const Prim *prims,
                            thread float2 *iv, thread uchar *tis, thread ushort *pid,
                            uint n, device const float4 *sigma, constant Camera &cam) {
        float3 sum = float3(0.0);
        if ((cam.mutations & MUT_NO_UNION) != 0u) {
            for (uint i = 0; i < n; i++) {
                float3 s = sigma[tis[i]].rgb;
                sum += depositAt(ro, rd, prims, uint(pid[i]), iv[i].x, s, cam);
                sum += depositAt(ro, rd, prims, uint(pid[i]), iv[i].y, s, cam);
            }
            return sum;
        }
        float curS[MAX_TISSUE];
        float curE[MAX_TISSUE];
        ushort sPid[MAX_TISSUE];
        ushort ePid[MAX_TISSUE];
        bool open[MAX_TISSUE];
        for (uint k = 0; k < uint(MAX_TISSUE); k++) {
            curS[k] = 0.0; curE[k] = 0.0; sPid[k] = 0; ePid[k] = 0; open[k] = false;
        }
        for (uint i = 0; i < n; i++) {
            uint k = uint(tis[i]);
            if (!open[k]) {
                curS[k] = iv[i].x; curE[k] = iv[i].y;
                sPid[k] = pid[i]; ePid[k] = pid[i];
                open[k] = true;
            } else if (iv[i].x > curE[k]) {
                float3 s = sigma[k].rgb;
                sum += depositAt(ro, rd, prims, uint(sPid[k]), curS[k], s, cam);
                sum += depositAt(ro, rd, prims, uint(ePid[k]), curE[k], s, cam);
                curS[k] = iv[i].x; curE[k] = iv[i].y;
                sPid[k] = pid[i]; ePid[k] = pid[i];
            } else if (iv[i].y > curE[k]) {
                curE[k] = iv[i].y;
                ePid[k] = pid[i];
            }
        }
        for (uint k = 0; k < uint(MAX_TISSUE); k++) {
            if (!open[k]) continue;
            float3 s = sigma[k].rgb;
            sum += depositAt(ro, rd, prims, uint(sPid[k]), curS[k], s, cam);
            sum += depositAt(ro, rd, prims, uint(ePid[k]), curE[k], s, cam);
        }
        return sum;
    }

    kernel void render(device uchar4 *pixels [[buffer(0)]],
                       constant Camera &cam [[buffer(1)]],
                       device const Prim *prims [[buffer(2)]],
                       device const uint *cellStart [[buffer(3)]],
                       device const uint *cellItems [[buffer(4)]],
                       constant Grid &grid [[buffer(5)]],
                       device const float4 *sigma [[buffer(6)]],
                       device float4 *scatterOut [[buffer(7)]],
                       device atomic_uint *stats [[buffer(8)]],
                       device const uchar *tileMask [[buffer(9)]],
                       uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= cam.width || gid.y >= cam.height) return;
        uint out = gid.y * cam.width + gid.x;

        // The camera never moves, so a pixel no primitive can ever reach in any
        // of the 120 frames is decided once, before the loop, and costs one
        // byte to look up. In brightfield this optimisation is worthless —
        // every pixel still has to evaluate the lamp. In darkfield the answer
        // for such a pixel is exactly black, so the whole ray can go.
        if (cam.useMask != 0u) {
            uint tx = gid.x >> cam.tileShift;
            uint ty = gid.y >> cam.tileShift;
            if (tx < cam.tilesX && ty < cam.tilesY && tileMask[ty * cam.tilesX + tx] == 0u) {
                pixels[out] = uchar4(0, 0, 0, 255);
                scatterOut[out] = float4(0.0);
                return;
            }
        }

        uint k = cam.samplesPerSide;
        float3 sum = float3(0.0);
        float3 rawSum = float3(0.0);
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
                ushort pid[MAX_IV];
                uint n = 0;
                gather(ro, rd, prims, cellStart, cellItems, grid, cam,
                       iv, tis, pid, n, overflow, met);
                float3 I = integrate(ro, rd, prims, iv, tis, pid, n, sigma, cam);
                rawSum += I;
                // 1 − exp(−I). Monotone, so it cannot reorder anything the
                // physics said, and exactly 0 at I = 0 — no epsilon glow.
                float3 e = exp(-I * cam.exposure);
                sum += float3(1.0) - e;
            }
        }
        float inv = 1.0 / float(k * k);
        float3 colour = clamp(sum * inv, 0.0, 1.0);
        scatterOut[out] = float4(rawSum * inv, float(met));
        if (overflow > 0) atomic_fetch_add_explicit(stats, overflow, memory_order_relaxed);
        pixels[out] = uchar4(uchar3(round(colour * 255.0)), 255);
    }
    """

// MARK: - A CPU reference for the same physics

struct CPUScatter {
    var intensity: SIMD3<Float>
    var intervals: Int
    var interfaces: Int
}

/// The same darkfield integral on the CPU, brute force, no grid. Exact and
/// cheap, so the tests can check the GPU against it rather than against a
/// picture — and so the interval COUNT has a reference, which is the thing
/// mailboxing actually controls.
func cpuScatter(origin ro: SIMD3<Float>, direction rd: SIMD3<Float>,
                prims: [GPUPrim], sigma: [SIMD3<Float>],
                mutations: MMutations = [],
                grazingK: Float = grazingExponent) -> CPUScatter {
    var list: [(Float, Float, Int, Int)] = []
    list.reserveCapacity(64)
    for (i, p) in prims.enumerated() {
        guard let (a, b) = cpuInterval(origin: ro, direction: rd, prim: p) else { continue }
        if b <= 0 { continue }
        let s: Float = max(a, 0)
        if b <= s { continue }
        list.append((s, b, p.tissue, i))
    }
    list.sort { $0.0 < $1.0 }

    var sum = SIMD3<Float>(repeating: 0)
    var interfaces = 0
    func deposit(_ prim: Int, _ t: Float, _ sig: SIMD3<Float>) {
        let q: SIMD3<Float> = ro + rd * t
        let n: SIMD3<Float> = prims[prim].normal(at: q)
        var g: Float = 1
        if !mutations.contains(.flatGrazing) {
            let c: Float = abs(simd_dot(n, rd))
            let e: Float = 1 - min(c, 1)
            g = pow(e, grazingK)
        }
        sum += sig * g
        interfaces += 1
    }

    if mutations.contains(.noUnion) {
        for (a, b, k, i) in list {
            deposit(i, a, sigma[k])
            deposit(i, b, sigma[k])
        }
        return CPUScatter(intensity: sum, intervals: list.count, interfaces: interfaces)
    }

    let slots = max(sigma.count, mTissueCount)
    var curS = [Float](repeating: 0, count: slots)
    var curE = [Float](repeating: 0, count: slots)
    var sPid = [Int](repeating: 0, count: slots)
    var ePid = [Int](repeating: 0, count: slots)
    var open = [Bool](repeating: false, count: slots)
    for (a, b, k, i) in list {
        if !open[k] {
            curS[k] = a; curE[k] = b; sPid[k] = i; ePid[k] = i; open[k] = true
        } else if a > curE[k] {
            deposit(sPid[k], curS[k], sigma[k])
            deposit(ePid[k], curE[k], sigma[k])
            curS[k] = a; curE[k] = b; sPid[k] = i; ePid[k] = i
        } else if b > curE[k] {
            curE[k] = b; ePid[k] = i
        }
    }
    for k in 0..<slots where open[k] {
        deposit(sPid[k], curS[k], sigma[k])
        deposit(ePid[k], curE[k], sigma[k])
    }
    return CPUScatter(intensity: sum, intervals: list.count, interfaces: interfaces)
}

/// The tone curve, on the CPU, so a test can state that black stays black
/// without going through the GPU at all.
func darkfieldTone(_ i: SIMD3<Float>) -> SIMD3<Float> {
    let e = SIMD3<Float>(exp(-i.x * darkfieldExposure), exp(-i.y * darkfieldExposure),
                         exp(-i.z * darkfieldExposure))
    return SIMD3<Float>(repeating: 1) - e
}

// MARK: - The swept bound
//
// The camera is bolted down for all 120 frames, which is the premise of the
// shot and also, for free, an optimisation nothing earlier in this project
// could use. If no primitive in ANY frame projects onto a pixel, that pixel is
// black in every frame, and the ray never has to be traced at all.
//
// Two forms of the same idea are built here. The world-space swept AABB is the
// literal bounding volume: the union over all frames of every primitive's box.
// The screen-space TILE MASK is what the kernel actually reads, and it is much
// tighter, because the projection of the swept volume is a rectangle while the
// projection of the animal is a long diagonal streak with a lot of empty
// corner around it.
//
// The mask is conservative by construction. The camera is orthographic with
// right = +x and up = +y, so the screen footprint of a world AABB is exactly
// the rectangle between its projected corners — never smaller. Rounding out to
// whole 16-pixel tiles can only add area. A test renders a frame with the mask
// off and demands the two pictures be identical pixel for pixel.

struct SweptBound {
    var lo: SIMD3<Float>
    var hi: SIMD3<Float>
    var mask: [UInt8]
    var tilesX: Int
    var tilesY: Int
    var liveTiles: Int

    var tileCount: Int { tilesX * tilesY }
    var liveFraction: Double {
        tileCount == 0 ? 0 : Double(liveTiles) / Double(tileCount)
    }
}

func buildSweptBound(frames: [Int], camera: Camera, width: Int, height: Int,
                     tile: Int = maskTileSize,
                     pose: (Int) -> [GPUPrim]) -> SweptBound {
    let tilesX: Int = (width + tile - 1) / tile
    let tilesY: Int = (height + tile - 1) / tile
    var mask = [UInt8](repeating: 0, count: tilesX * tilesY)
    var lo = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
    var hi = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)

    for f in frames {
        for p in pose(f) {
            let b = p.bounds()
            lo = simd_min(lo, b.lo)
            hi = simd_max(hi, b.hi)
            // Orthographic, axis-aligned basis: the box's screen rectangle is
            // the projection of its two extreme corners, and nothing else has
            // to be checked.
            let a: SIMD2<Float> = camera.project(b.lo, width: width, height: height)
            let c: SIMD2<Float> = camera.project(b.hi, width: width, height: height)
            let x0: Float = min(a.x, c.x)
            let x1: Float = max(a.x, c.x)
            let y0: Float = min(a.y, c.y)
            let y1: Float = max(a.y, c.y)
            if x1 < 0 || y1 < 0 || x0 >= Float(width) || y0 >= Float(height) { continue }
            let tx0: Int = max(0, Int(x0.rounded(.down)) / tile)
            let tx1: Int = min(tilesX - 1, Int(x1.rounded(.down)) / tile)
            let ty0: Int = max(0, Int(y0.rounded(.down)) / tile)
            let ty1: Int = min(tilesY - 1, Int(y1.rounded(.down)) / tile)
            if tx1 < tx0 || ty1 < ty0 { continue }
            for ty in ty0...ty1 {
                let row: Int = ty * tilesX
                for tx in tx0...tx1 { mask[row + tx] = 1 }
            }
        }
    }
    if lo.x > hi.x { lo = .zero; hi = .zero }
    var live = 0
    for m in mask where m != 0 { live += 1 }
    return SweptBound(lo: lo, hi: hi, mask: mask, tilesX: tilesX, tilesY: tilesY,
                      liveTiles: live)
}

// MARK: - The renderer

final class DarkfieldRenderer {
    let device: MTLDevice
    private let pipeline: MTLComputePipelineState
    private let queue: MTLCommandQueue
    private var primBuffer: MTLBuffer?
    private var startBuffer: MTLBuffer?
    private var itemBuffer: MTLBuffer?
    private var sigmaBuffer: MTLBuffer?
    private var camBuffer: MTLBuffer?
    private var gridBuffer: MTLBuffer?
    private var maskBuffer: MTLBuffer?
    private var scatterBuffer: MTLBuffer?
    private var statsBuffer: MTLBuffer?
    private(set) var grid: UniformGrid?
    private(set) var overflowCount: UInt32 = 0
    private(set) var lastBuildSeconds: Double = 0

    init(device: MTLDevice) throws {
        self.device = device
        do {
            let options = MTLCompileOptions()
            // Off, as in every step since 3. Fast math would let the compiler
            // reassociate the intersection arithmetic, and here it would also
            // be free to contract the interval endpoints — which is exactly
            // where a union merge must not drift, and where a drifting endpoint
            // means a surface deposited twice.
            options.fastMathEnabled = false
            let library = try device.makeLibrary(source: darkfieldKernelSource, options: options)
            guard let function = library.makeFunction(name: "render") else {
                throw RenderError.kernelCompile("no kernel named render")
            }
            pipeline = try device.makeComputePipelineState(function: function)
        } catch let error as RenderError {
            throw error
        } catch {
            throw RenderError.kernelCompile(error.localizedDescription)
        }
        guard let queue = device.makeCommandQueue() else {
            throw RenderError.gpu("no command queue")
        }
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

    func setMask(_ bound: SweptBound?) throws {
        let bytes: [UInt8] = bound?.mask ?? [UInt8(1)]
        maskBuffer = try fill(maskBuffer, with: bytes)
    }

    /// Ray-traces into the top `viewHeight` rows of `frame`. Returns GPU seconds.
    @discardableResult
    func render(prims: [GPUPrim], sigma: [SIMD3<Float>], camera: Camera,
                into frame: MTLBuffer, width: Int, viewHeight: Int,
                samplesPerSide: Int = 2, useGrid: Bool = true,
                mask: SweptBound? = nil,
                mutations: MMutations = []) throws -> Double {
        if prims.count > mailboxCapacity { throw RenderError.tooManyPrimitives(prims.count) }
        if useGrid && grid == nil { throw RenderError.gpu("no grid has been built") }

        let tilesX: UInt32 = UInt32(mask?.tilesX ?? 0)
        let tilesY: UInt32 = UInt32(mask?.tilesY ?? 0)
        let cam = DarkCamera(origin: SIMD4(camera.origin, 1),
                             forward: SIMD4(camera.forward, 0),
                             right: SIMD4(camera.right, 0),
                             up: SIMD4(camera.up, 0),
                             halfWidth: camera.halfWidth, halfHeight: camera.halfHeight,
                             grazingExponent: grazingExponent, exposure: darkfieldExposure,
                             width: UInt32(width), height: UInt32(viewHeight),
                             primCount: UInt32(prims.count),
                             samplesPerSide: UInt32(samplesPerSide),
                             useGrid: useGrid ? 1 : 0,
                             mutations: mutations.rawValue,
                             useMask: mask == nil ? 0 : 1,
                             tileShift: maskTileShift,
                             tilesX: tilesX, tilesY: tilesY, pad0: 0, pad1: 0)
        let g = DarkGrid(origin: SIMD4(grid?.origin ?? .zero, 0),
                         cell: SIMD4(grid?.cellSize ?? SIMD3(repeating: 1), 0),
                         dims: SIMD4(UInt32(grid?.dims.x ?? 1), UInt32(grid?.dims.y ?? 1),
                                     UInt32(grid?.dims.z ?? 1), 0))
        // Everything the kernel reads lives in a device buffer, including the
        // two small structs: setBytes has a size ceiling the scene would walk
        // into sooner or later, and one rule is easier to keep than two.
        camBuffer = try fill(camBuffer, with: [cam])
        gridBuffer = try fill(gridBuffer, with: [g])
        let list: [GPUPrim] = prims.isEmpty
            ? [GPUPrim.sph(SIMD3(0, 0, 1e9), 0.001, .cuticle)] : prims
        primBuffer = try fill(primBuffer, with: list)
        let sig: [SIMD4<Float>] = (0..<16).map { i -> SIMD4<Float> in
            i < sigma.count ? SIMD4(sigma[i], 0) : SIMD4(repeating: 0)
        }
        sigmaBuffer = try fill(sigmaBuffer, with: sig)
        if startBuffer == nil { startBuffer = try fill(nil, with: [UInt32(0), UInt32(0)]) }
        if itemBuffer == nil { itemBuffer = try fill(nil, with: [UInt32(0)]) }
        if let m = mask {
            maskBuffer = try fill(maskBuffer, with: m.mask)
        } else if maskBuffer == nil {
            maskBuffer = try fill(nil, with: [UInt8(1)])
        }
        let pixels: Int = width * viewHeight
        if scatterBuffer == nil || scatterBuffer!.length < pixels * 16 {
            scatterBuffer = device.makeBuffer(length: pixels * 16, options: .storageModeShared)
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
        e.setBuffer(camBuffer, offset: 0, index: 1)
        e.setBuffer(primBuffer, offset: 0, index: 2)
        e.setBuffer(startBuffer, offset: 0, index: 3)
        e.setBuffer(itemBuffer, offset: 0, index: 4)
        e.setBuffer(gridBuffer, offset: 0, index: 5)
        e.setBuffer(sigmaBuffer, offset: 0, index: 6)
        e.setBuffer(scatterBuffer, offset: 0, index: 7)
        e.setBuffer(statsBuffer, offset: 0, index: 8)
        e.setBuffer(maskBuffer, offset: 0, index: 9)
        let w: Int = pipeline.threadExecutionWidth
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

    /// The scattered intensity the last render measured at one pixel, before
    /// the tone curve. The tests read this rather than un-doing 8-bit colour.
    func intensity(atX x: Int, y: Int, width: Int) -> SIMD3<Float> {
        guard let b = scatterBuffer else { return .zero }
        let p = b.contents().assumingMemoryBound(to: SIMD4<Float>.self)
        let v = p[y * width + x]
        return SIMD3(v.x, v.y, v.z)
    }

    /// How many intervals the rays of one pixel collected — the count the
    /// mailbox exists to hold down.
    func intervalsMet(atX x: Int, y: Int, width: Int) -> Int {
        guard let b = scatterBuffer else { return 0 }
        let p = b.contents().assumingMemoryBound(to: SIMD4<Float>.self)
        return Int(p[y * width + x].w.rounded())
    }

    private func fill<T>(_ buffer: MTLBuffer?, with items: [T]) throws -> MTLBuffer {
        let bytes: Int = max(items.count * MemoryLayout<T>.stride, 16)
        var target = buffer
        if target == nil || target!.length < bytes {
            target = device.makeBuffer(length: bytes, options: .storageModeShared)
        }
        guard let out = target else { throw RenderError.gpu("could not allocate a scene buffer") }
        items.withUnsafeBytes { out.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
        return out
    }
}
