// A cutaway that is not a hole.
//
// Clipping with a plane is easy. Making the cut LOOK like a cut is the problem.
// Naive clipping throws away every surface in front of the plane and leaves you
// staring at the inside of the far wall of each shell: a hollow chocolate egg,
// not a cross-section. What a cross-section does is CAP the cut — the sliced
// face is filled with the material it sliced through, so husk reads as solid
// husk and meat as solid meat.
//
// With step 13's ordered multi-hit traversal that is nearly free, and this step
// exists partly to show how nearly. Step 13's ray already collects EVERY entry
// and exit along its path, sorted, merged into a per-tissue union, with
// per-ray mailboxing so a primitive straddling several grid boxes is counted
// once. Given that list, capping is three lines of bookkeeping:
//
//   1. take the merged runs,
//   2. throw away everything in front of the clip plane,
//   3. any run that STRADDLES the plane has its entry clamped to the plane and
//      is shaded as CUT MATERIAL rather than as a surface.
//
// Nothing else changes. No second pass, no cap geometry, no stencil.
//
// This file does not rewrite step 13's traversal; it literally reuses it. The
// kernel below is built by taking step 13's Metal source up to the point where
// it starts integrating Beer–Lambert, and appending this step's own resolve and
// shading to it. So `ellipsoidInterval`, `capsuleInterval`, `addInterval` and
// `gather` here are the same characters that shipped in step 13 — including the
// closest-approach intersection form, which step 13 learned the hard way: the
// textbook discriminant b² − ac cancels to every digit a float has when the
// camera stands a long way off a small feature.
//
// Two things this step adds on top of the traversal:
//
//   PRIORITY. Fifteen materials, and where two occupy the same point the higher
//   index wins. That is what makes a stack of nested SOLID ellipsoids behave
//   like a stack of shells — none of them has to be hollow, and there is no
//   constructive-solid-geometry tree. A root running through soil and a shoot
//   running through husk fall out of the same rule.
//
//   CUT SHADING. Cut faces are flat, matte and slightly desaturated, with a
//   boundary line; uncut surfaces keep normal shading with ambient occlusion.
//   The CONTRAST between the two is what makes a cutaway legible instead of
//   confusing, and it is deliberately not subtle.
//
// Amanatides & Woo, "A Fast Voxel Traversal Algorithm for Ray Tracing",
// Eurographics (1987) — via step 13, unchanged.

import Foundation
import Metal
import simd

// MARK: - Splicing step 13's traversal in
//
// Step 13's kernel source is one string: includes, structs, the two
// intersection routines, the sorted insert, the grid walk, and then its own
// Beer–Lambert integrator and kernel. Everything up to the integrator is
// exactly what this step needs and nothing this step needs is missing from it,
// so the split is a text split at a named marker rather than a copy. A test
// checks the marker is still there and that what comes out has the four
// functions in it and none of step 13's shading.

let traversalMarker: String = "// Merge each tissue's intervals into a union"

func spliceTraversal() -> String {
    let source: String = swimKernelSource
    let marker: String = traversalMarker
    let found: Range<String.Index>? = source.range(of: marker)
    guard let r: Range<String.Index> = found else {
        fatalError("step 13's kernel no longer contains \(marker)")
    }
    let head: Substring = source[source.startIndex..<r.lowerBound]
    return String(head)
}

let traversalPrefix: String = spliceTraversal()

// MARK: - The look

/// Everything about the lighting and the cut that the kernel needs and the
/// camera struct has no room for. It rides in its own `device`-address-space
/// buffer, like all the other scene data in this project.
struct GPULook {
    var sun: SIMD4<Float>          // xyz toward the sun, w intensity
    var sunColour: SIMD4<Float>
    var ambient: SIMD4<Float>      // rgb ambient, w AO strength
    var skyTop: SIMD4<Float>
    var skyBottom: SIMD4<Float>
    var cut: SIMD4<Float>          // x zCut, y AO radius, z AO rays, w soil line
    var tex: SIMD4<Float>          // x desaturation, y lift, z fibre pitch, w speckle
}

struct Look {
    var sun = simd_normalize(SIMD3<Float>(-0.42, 0.74, 0.53))
    var sunIntensity: Float = 0.80
    var sunColour = SIMD3<Float>(1.00, 0.96, 0.88)
    var ambient = SIMD3<Float>(0.46, 0.50, 0.56)
    var aoStrength: Float = 0.55
    var aoRadius: Float = 26
    var aoRays: Int = 5
    var skyTop = SIMD3<Float>(0.145, 0.235, 0.330)
    var skyBottom = SIMD3<Float>(0.360, 0.450, 0.500)
    var cutDesaturation: Float = 0.34
    var cutLift: Float = 1.06
    var fibrePitch: Float = 0.62
    var speckle: Float = 0.26

    func packed(cutZ: Float, soilY: Float) -> GPULook {
        GPULook(sun: SIMD4(sun, sunIntensity),
                sunColour: SIMD4(sunColour, 0),
                ambient: SIMD4(ambient, aoStrength),
                skyTop: SIMD4(skyTop, 0),
                skyBottom: SIMD4(skyBottom, 0),
                cut: SIMD4(cutZ, aoRadius, Float(aoRays), soilY),
                tex: SIMD4(cutDesaturation, cutLift, fibrePitch, speckle))
    }
}

// MARK: - The kernel this step adds on top of step 13's traversal

let cutawayShadingSource = """
    // ----------------------------------------------------------------------
    // Everything above this line is step 13's, spliced in unchanged. Everything
    // below is step 15's.
    // ----------------------------------------------------------------------

    #define MAX_MAT 16
    #define TOPSOIL 0
    #define SUBSOIL 1
    #define MESOCARP 3
    #define ENDOSPERM 6
    #define HAUSTORIUM 9

    struct Look {
        float4 sun; float4 sunColour; float4 ambient;
        float4 skyTop; float4 skyBottom; float4 cut; float4 tex;
    };

    inline float hash13(float3 p) {
        float h = dot(p, float3(127.1, 311.7, 74.7));
        return fract(sin(h) * 43758.5453123);
    }

    // The per-material union, exactly as step 13 merges it, but emitting the
    // merged runs instead of integrating them away. The input list is already
    // sorted by entry, and a list sorted overall is sorted within any subset of
    // it, so one pass with a running open run per material does it.
    inline uint mergeRuns(thread float2 *iv, thread uchar *tis, uint n,
                          thread float2 *run, thread uchar *runMat) {
        float curS[MAX_MAT];
        float curE[MAX_MAT];
        bool open[MAX_MAT];
        for (uint k = 0; k < uint(MAX_MAT); k++) { curS[k] = 0.0; curE[k] = 0.0; open[k] = false; }
        uint nr = 0;
        for (uint i = 0; i < n; i++) {
            uint k = uint(tis[i]);
            if (k >= uint(MAX_MAT)) continue;
            if (!open[k]) {
                curS[k] = iv[i].x; curE[k] = iv[i].y; open[k] = true;
            } else if (iv[i].x > curE[k]) {
                if (nr < uint(MAX_IV)) { run[nr] = float2(curS[k], curE[k]); runMat[nr] = uchar(k); nr++; }
                curS[k] = iv[i].x; curE[k] = iv[i].y;
            } else {
                curE[k] = max(curE[k], iv[i].y);
            }
        }
        for (uint k = 0; k < uint(MAX_MAT); k++) {
            if (open[k] && nr < uint(MAX_IV)) {
                run[nr] = float2(curS[k], curE[k]); runMat[nr] = uchar(k); nr++;
            }
        }
        return nr;
    }

    // Priority IS the material index: where two materials cover the same point
    // the inner one wins. Six nested solid ellipsoids become six shells with no
    // shell ever being hollow.
    inline int coverAt(thread float2 *run, thread uchar *runMat, uint nr, float t) {
        int best = -1;
        for (uint i = 0; i < nr; i++) {
            if (run[i].x <= t && t < run[i].y) {
                int m = int(runMat[i]);
                if (m > best) best = m;
            }
        }
        return best;
    }

    // Which primitive of material `mat` put a surface at ray parameter `t`. The
    // run that starts at t was started by exactly one primitive's entry, so the
    // one whose own entry is nearest t is the one whose normal this is. Looked
    // up in the single grid box the hit point lies in — the primitive's bounding
    // box contains the hit, so the grid has it filed there — which makes the
    // answer identical whether the frame was traced through the grid or brute
    // force, and the grid-rebuild test checks exactly that.
    inline float3 surfaceNormal(float3 ro, float3 rd, float t, uint mat,
                                device const Prim *prims,
                                device const uint *cellStart, device const uint *cellItems,
                                constant Grid &grid, constant Camera &cam) {
        float3 p = ro + rd * t;
        float best = 1e30;
        int bi = -1;
        if (cam.useGrid == 0u) {
            for (uint i = 0; i < cam.primCount; i++) {
                if (uint(prims[i].r2.w + 0.5) != mat) continue;
                float t0, t1;
                if (!primInterval(ro, rd, prims[i], t0, t1)) continue;
                float d = fabs(t0 - t);
                if (d < best) { best = d; bi = int(i); }
            }
        } else {
            int3 dims = int3(grid.dims.xyz);
            float3 f = (p - grid.origin.xyz) / grid.cell.xyz;
            int3 c = clamp(int3(floor(f)), int3(0), dims - 1);
            uint index = (uint(c.z) * grid.dims.y + uint(c.y)) * grid.dims.x + uint(c.x);
            for (uint i = cellStart[index]; i < cellStart[index + 1]; i++) {
                uint s = cellItems[i];
                if (uint(prims[s].r2.w + 0.5) != mat) continue;
                float t0, t1;
                if (!primInterval(ro, rd, prims[s], t0, t1)) continue;
                float d = fabs(t0 - t);
                if (d < best) { best = d; bi = int(s); }
            }
        }
        if (bi < 0) return -rd;
        device const Prim &q = prims[bi];
        if (q.c.w < 0.5) {
            // ∇|M⁻¹(p − c)| ∝ M⁻ᵀ M⁻¹ (p − c), and the rows of M⁻¹ are r0,r1,r2,
            // so M⁻ᵀ's columns are r0,r1,r2 and the second product is a sum.
            float3 rel = p - q.c.xyz;
            float3 u = float3(dot(q.r0.xyz, rel), dot(q.r1.xyz, rel), dot(q.r2.xyz, rel));
            float3 g = q.r0.xyz * u.x + q.r1.xyz * u.y + q.r2.xyz * u.z;
            return normalize(g);
        }
        float3 A = q.r0.xyz;
        float3 B = q.r1.xyz;
        float3 ba = B - A;
        float h = clamp(dot(p - A, ba) / max(dot(ba, ba), 1e-12), 0.0, 1.0);
        float3 axisPoint = A + ba * h;
        float3 d = p - axisPoint;
        float len = length(d);
        if (len < 1e-6) return -rd;
        return d / len;
    }

    // Ambient occlusion, and it costs one more gather per ray. Cut faces get
    // none — they are flat by convention — so this only runs on the husk's own
    // curved surface, the sprout and the leaf, which between them are a small
    // part of a portrait frame.
    inline float ambientOcclusion(float3 p, float3 n, float rot,
                                  device const Prim *prims,
                                  device const uint *cellStart, device const uint *cellItems,
                                  constant Grid &grid, constant Camera &cam,
                                  constant Look &look,
                                  thread float2 *iv, thread uchar *tis) {
        uint rays = uint(look.cut.z + 0.5);
        if (rays == 0u) return 1.0;
        float radius = look.cut.y;
        float3 guide = (fabs(n.z) < 0.9) ? float3(0.0, 0.0, 1.0) : float3(1.0, 0.0, 0.0);
        float3 tx = normalize(cross(guide, n));
        float3 ty = cross(n, tx);
        float3 origin = p + n * 0.08;
        float occluded = 0.0;
        for (uint k = 0; k < rays; k++) {
            float slot = (float(k) + 0.5) / float(rays);
            float a = slot * 6.28318530718 + rot;
            float ct = sqrt(1.0 - slot * 0.78);
            float st = sqrt(max(1.0 - ct * ct, 0.0));
            float3 d = normalize(tx * (cos(a) * st) + ty * (sin(a) * st) + n * ct);
            uint m = 0;
            uint spill = 0;
            uint met = 0;
            gather(origin, d, prims, cellStart, cellItems, grid, cam, iv, tis, m, spill, met);
            float nearest = 1e30;
            for (uint i = 0; i < m; i++) {
                float s = iv[i].x;
                if (s > 0.05 && s < nearest) nearest = s;
            }
            if (nearest < radius) occluded += 1.0 - nearest / radius;
        }
        float mean = occluded / float(rays);
        return 1.0 - look.ambient.w * mean;
    }

    // The sky behind everything, a plain vertical ramp. It is not a light
    // source in this render — the ambient term is — it is just what is there
    // when a ray leaves without touching anything.
    inline float3 sky(constant Camera &cam, constant Look &look, float ndcY) {
        float v = clamp(ndcY * 0.5 + 0.5, 0.0, 1.0);
        return look.skyBottom.rgb + (look.skyTop.rgb - look.skyBottom.rgb) * v;
    }

    // Flat, matte, a little desaturated, with just enough texture to say what
    // the material is: fibre streaks for coir, grain for soil, a fine sponge
    // for the haustorium. This is the diagram convention, and the contrast
    // against a lit uncut surface is the whole reason a cutaway reads.
    inline float3 cutShade(float3 albedo, uint mat, float3 p, constant Look &look) {
        float luma = dot(albedo, float3(0.299, 0.587, 0.114));
        float3 flat = mix(albedo, float3(luma), look.tex.x) * look.tex.y;
        if (mat == uint(MESOCARP)) {
            // Coir runs the length of the nut, so in this section it streaks.
            float s = sin(p.y * look.tex.z + p.x * 0.035);
            float t = sin(p.y * look.tex.z * 2.7 + 1.9);
            flat *= 0.90 + 0.13 * (0.7 * s + 0.3 * t);
        } else if (mat == uint(TOPSOIL) || mat == uint(SUBSOIL)) {
            float grain = (mat == uint(TOPSOIL)) ? 2.6 : 5.2;
            float h = hash13(floor(p / grain));
            float h2 = hash13(floor(p / (grain * 3.1)) + 17.0);
            flat *= 1.0 - look.tex.w * 0.5 + look.tex.w * (0.6 * h + 0.4 * h2);
        } else if (mat == uint(HAUSTORIUM)) {
            float h = hash13(floor(p / 1.9));
            flat *= 0.96 + 0.06 * h;
        } else if (mat == uint(ENDOSPERM)) {
            float h = hash13(floor(p / 3.4));
            flat *= 0.985 + 0.025 * h;
        }
        return flat;
    }

    kernel void cutaway(device uchar4 *pixels [[buffer(0)]],
                        constant Camera &cam [[buffer(1)]],
                        device const Prim *prims [[buffer(2)]],
                        device const uint *cellStart [[buffer(3)]],
                        device const uint *cellItems [[buffer(4)]],
                        constant Grid &grid [[buffer(5)]],
                        device const float4 *albedo [[buffer(6)]],
                        device float4 *aux [[buffer(7)]],
                        device atomic_uint *stats [[buffer(8)]],
                        constant Look &look [[buffer(9)]],
                        uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= cam.width || gid.y >= cam.height) return;
        uint k = cam.samplesPerSide;
        float3 sum = float3(0.0);
        uint overflow = 0;
        uint met = 0;
        int firstMat = -1;
        float firstCut = 0.0;
        float firstT = 0.0;
        float rot = hash13(float3(float(gid.x), float(gid.y), 1.0)) * 6.28318530718;

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

                float2 run[MAX_IV];
                uchar runMat[MAX_IV];
                uint nr = mergeRuns(iv, tis, n, run, runMat);

                // The clip plane, in this ray's own parameter. Every ray in this
                // camera has rd.z < 0, so this is always ahead of the origin.
                float tCut = (look.cut.x - ro.z) / rd.z;
                if (tCut < 0.0) tCut = 0.0;

                float eps = 1e-3;
                int mat = coverAt(run, runMat, nr, tCut + eps);
                float t = tCut;
                bool isCut = true;
                if (mat < 0) {
                    // Nothing straddles the plane here: find the first surface
                    // beyond it, the ordinary way.
                    float ts = 1e30;
                    int tsMat = -1;
                    for (uint r = 0; r < nr; r++) {
                        if (run[r].x >= tCut && run[r].x < ts) { ts = run[r].x; tsMat = int(runMat[r]); }
                    }
                    if (ts < 1e29) {
                        t = ts;
                        isCut = false;
                        int inner = coverAt(run, runMat, nr, ts + eps);
                        mat = (inner >= 0) ? inner : tsMat;
                    }
                }

                float3 colour;
                if (mat < 0) {
                    colour = sky(cam, look, ndcY);
                } else {
                    float3 p = ro + rd * t;
                    float3 alb = albedo[mat].rgb;
                    if (isCut) {
                        colour = cutShade(alb, uint(mat), p, look);
                    } else {
                        float3 nrm = surfaceNormal(ro, rd, t, uint(mat), prims,
                                                   cellStart, cellItems, grid, cam);
                        if (dot(nrm, rd) > 0.0) nrm = -nrm;
                        float ao = ambientOcclusion(p, nrm, rot, prims, cellStart, cellItems,
                                                    grid, cam, look, iv, tis);
                        float ndl = max(dot(nrm, look.sun.xyz), 0.0);
                        float3 direct = alb * (ndl * look.sun.w) * look.sunColour.rgb;
                        float dome = 0.5 + 0.5 * nrm.y;
                        float3 indirect = alb * look.ambient.rgb * dome * ao;
                        colour = direct + indirect;
                    }
                }
                if (i == 0 && j == 0) {
                    firstMat = mat;
                    firstCut = isCut ? 1.0 : 0.0;
                    firstT = t;
                }
                sum += colour;
            }
        }
        float inv = 1.0 / float(k * k);
        float3 out = clamp(sum * inv, 0.0, 1.0);
        // w carries how many intervals this pixel's rays collected. That is
        // what mailboxing actually controls: with a per-material union a
        // duplicated interval merges straight back into itself, so switching
        // the mailbox off leaves the PICTURE bit-identical and only this number
        // moves. Step 13 learned that the hard way; the budget test here is
        // against a CPU reference count, not against the image.
        aux[gid.y * cam.width + gid.x] = float4(float(firstMat), firstCut, firstT, float(met));
        if (overflow > 0) atomic_fetch_add_explicit(stats, overflow, memory_order_relaxed);
        pixels[gid.y * cam.width + gid.x] = uchar4(uchar3(round(out * 255.0)), 255);
    }
    """

let cutawayKernelSource: String = traversalPrefix + cutawayShadingSource

// MARK: - The camera
//
// Orthographic, level, yawed off the cut plane's normal. Level matters: with
// any tilt at all the soil surface would recede up the frame and swallow the
// sky, because the ground here is a dome eight metres across. Seen edge-on it
// is a line, which is what a section wants.

func cutawayCamera(width: Int, height: Int, target: SIMD3<Float>, yaw: Float) -> Camera {
    var camera = Camera(centre: .zero, micronsPerPixel: mmPerPixel,
                        width: width, height: height, standOff: 0)
    let cy: Float = cos(yaw)
    let sy: Float = sin(yaw)
    camera.forward = SIMD3(-sy, 0, -cy)
    camera.right = SIMD3(cy, 0, -sy)
    camera.up = SIMD3(0, 1, 0)
    camera.origin = target - camera.forward * cameraStandOff
    return camera
}

/// The world point that lands in the middle of the frame. Chosen so the nut
/// sits left of centre, the sprout right of it, and the soil line about a third
/// of the way up.
let cutawayTarget = SIMD3<Float>(19.3, 97, 0)

// MARK: - A CPU reference for the same resolve
//
// Same rule, same priority, no grid: the tests use this to check capping is
// watertight, because sweeping a plane through a nut is a thing you can do
// exhaustively on the CPU and cannot do exhaustively by looking at pictures.

struct CutHit {
    var t: Float
    var material: Material
    var isCut: Bool
    var intervals: Int
}

func cpuResolve(origin ro: SIMD3<Float>, direction rd: SIMD3<Float>,
                prims: [GPUPrim], cutZ: Float) -> CutHit? {
    var list: [(Float, Float, Int)] = []
    list.reserveCapacity(48)
    for p in prims {
        guard let (a, b) = cpuInterval(origin: ro, direction: rd, prim: p) else { continue }
        if b <= 0 { continue }
        let s: Float = max(a, 0)
        if b <= s { continue }
        list.append((s, b, p.tissue))
    }
    let intervals: Int = list.count
    list.sort { $0.0 < $1.0 }

    var runs: [(Float, Float, Int)] = []
    var curS = [Float](repeating: 0, count: materialCount)
    var curE = [Float](repeating: 0, count: materialCount)
    var open = [Bool](repeating: false, count: materialCount)
    for (a, b, k) in list {
        if k < 0 || k >= materialCount { continue }
        if !open[k] {
            curS[k] = a; curE[k] = b; open[k] = true
        } else if a > curE[k] {
            runs.append((curS[k], curE[k], k))
            curS[k] = a; curE[k] = b
        } else {
            curE[k] = max(curE[k], b)
        }
    }
    for k in 0..<materialCount where open[k] { runs.append((curS[k], curE[k], k)) }

    func coverAt(_ t: Float) -> Int {
        var best = -1
        for (a, b, k) in runs where a <= t && t < b {
            if k > best { best = k }
        }
        return best
    }

    var tCut: Float = (cutZ - ro.z) / rd.z
    if tCut < 0 { tCut = 0 }
    let eps: Float = 1e-3
    var material: Int = coverAt(tCut + eps)
    var t: Float = tCut
    var isCut = true
    if material < 0 {
        var ts: Float = .greatestFiniteMagnitude
        var tsMat = -1
        for (a, _, k) in runs where a >= tCut && a < ts { ts = a; tsMat = k }
        if ts < .greatestFiniteMagnitude {
            t = ts
            isCut = false
            let inner: Int = coverAt(ts + eps)
            material = inner >= 0 ? inner : tsMat
        }
    }
    guard material >= 0, let m = Material(rawValue: material) else { return nil }
    return CutHit(t: t, material: m, isCut: isCut, intervals: intervals)
}

/// The same normal the kernel picks: the primitive of `material` whose own
/// entry is nearest `t`, because the run that starts at `t` was started by
/// exactly one primitive's entry. Brute force, for the tests.
func cpuSurfaceNormal(origin ro: SIMD3<Float>, direction rd: SIMD3<Float>, t: Float,
                      material: Material, prims: [GPUPrim]) -> SIMD3<Float> {
    var best: Float = .greatestFiniteMagnitude
    var chosen: GPUPrim? = nil
    for p in prims where p.tissue == material.rawValue {
        guard let (t0, _) = cpuInterval(origin: ro, direction: rd, prim: p) else { continue }
        let d: Float = abs(t0 - t)
        if d < best { best = d; chosen = p }
    }
    guard let q = chosen else { return -rd }
    let point: SIMD3<Float> = ro + rd * t
    if q.kind == .ellipsoid {
        let rel: SIMD3<Float> = point - SIMD3(q.c.x, q.c.y, q.c.z)
        let r0 = SIMD3<Float>(q.r0.x, q.r0.y, q.r0.z)
        let r1 = SIMD3<Float>(q.r1.x, q.r1.y, q.r1.z)
        let r2 = SIMD3<Float>(q.r2.x, q.r2.y, q.r2.z)
        let u = SIMD3<Float>(simd_dot(r0, rel), simd_dot(r1, rel), simd_dot(r2, rel))
        let g: SIMD3<Float> = r0 * u.x + r1 * u.y + r2 * u.z
        return simd_normalize(g)
    }
    let a = SIMD3<Float>(q.r0.x, q.r0.y, q.r0.z)
    let b = SIMD3<Float>(q.r1.x, q.r1.y, q.r1.z)
    let ba: SIMD3<Float> = b - a
    let denom: Float = max(simd_dot(ba, ba), 1e-12)
    var h: Float = simd_dot(point - a, ba) / denom
    h = min(max(h, 0), 1)
    let axis: SIMD3<Float> = a + ba * h
    let d: SIMD3<Float> = point - axis
    if simd_length(d) < 1e-6 { return -rd }
    return simd_normalize(d)
}

/// Is `q` inside any primitive at all, and if so which material wins there?
/// Used by the capping test: if a point on the clip plane is inside a solid,
/// the renderer must return cut material there and nothing else.
func cpuMaterialAt(_ q: SIMD3<Float>, prims: [GPUPrim]) -> Material? {
    var best = -1
    for p in prims where cpuContains(p, point: q) {
        if p.tissue > best { best = p.tissue }
    }
    guard best >= 0 else { return nil }
    return Material(rawValue: best)
}

// MARK: - The renderer

final class CutawayRenderer {
    let device: MTLDevice
    private let pipeline: MTLComputePipelineState
    private let queue: MTLCommandQueue
    private var primBuffer: MTLBuffer?
    private var startBuffer: MTLBuffer?
    private var itemBuffer: MTLBuffer?
    private var albedoBuffer: MTLBuffer?
    private var auxBuffer: MTLBuffer?
    private var statsBuffer: MTLBuffer?
    private(set) var grid: UniformGrid?
    private(set) var overflowCount: UInt32 = 0
    private(set) var lastBuildSeconds: Double = 0

    init(device: MTLDevice) throws {
        self.device = device
        do {
            let options = MTLCompileOptions()
            // Off, as in every step since 3. Here it matters for a new reason:
            // a cut face is decided by whether a run's entry is before or after
            // the clip plane, and reassociating that comparison's arithmetic is
            // exactly the kind of drift that punches a hole through a cap.
            options.fastMathEnabled = false
            let library = try device.makeLibrary(source: cutawayKernelSource, options: options)
            guard let function = library.makeFunction(name: "cutaway") else {
                throw RenderError.kernelCompile("no kernel named cutaway")
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

    /// The grid is rebuilt every frame, because primitives are added as the
    /// plant grows: the shoot gains segments, the roots gain laterals, the
    /// haustorium appears. There is no static scene here to build once.
    func buildGrid(_ prims: [GPUPrim], density: Float = 1) throws {
        let t0 = Date()
        let g = UniformGrid(prims: prims, density: density)
        grid = g
        startBuffer = try fill(startBuffer, with: g.start)
        itemBuffer = try fill(itemBuffer, with: g.items.isEmpty ? [UInt32(0)] : g.items)
        lastBuildSeconds = Date().timeIntervalSince(t0)
    }

    @discardableResult
    func render(prims: [GPUPrim], albedo: [SIMD3<Float>], camera: Camera, look: Look,
                cutZ: Float, into frame: MTLBuffer, width: Int, viewHeight: Int,
                samplesPerSide: Int = 2, useGrid: Bool = true,
                breakage: Breakage = []) throws -> Double {
        if prims.count > mailboxCapacity { throw RenderError.tooManyPrimitives(prims.count) }
        var cam = GPUCamera(origin: SIMD4(camera.origin, 1),
                            forward: SIMD4(camera.forward, 0),
                            right: SIMD4(camera.right, 0),
                            up: SIMD4(camera.up, 0),
                            bgCentre: SIMD4(repeating: 0),
                            bgEdge: SIMD4(repeating: 0),
                            halfWidth: camera.halfWidth, halfHeight: camera.halfHeight,
                            falloff: 1, pad: 0,
                            width: UInt32(width), height: UInt32(viewHeight),
                            primCount: UInt32(prims.count),
                            samplesPerSide: UInt32(samplesPerSide),
                            useGrid: useGrid ? 1 : 0,
                            mutations: breakage.kernelBits)
        if useGrid && grid == nil { throw RenderError.gpu("no grid has been built") }
        var g = GPUGrid(origin: SIMD4(grid?.origin ?? .zero, 0),
                        cell: SIMD4(grid?.cellSize ?? SIMD3(repeating: 1), 0),
                        dims: SIMD4(UInt32(grid?.dims.x ?? 1), UInt32(grid?.dims.y ?? 1),
                                    UInt32(grid?.dims.z ?? 1), 0))
        var packedLook: GPULook = look.packed(cutZ: cutZ, soilY: soilSurfaceY)
        let list = prims.isEmpty
            ? [GPUPrim.sphere(centre: SIMD3(0, 0, 1e9), radius: 0.001, material: .topsoil)] : prims
        primBuffer = try fill(primBuffer, with: list)
        let table: [SIMD4<Float>] = (0..<16).map { i in
            i < albedo.count ? SIMD4(albedo[i], 0) : SIMD4(repeating: 0)
        }
        albedoBuffer = try fill(albedoBuffer, with: table)
        if startBuffer == nil { startBuffer = try fill(nil, with: [UInt32(0), UInt32(0)]) }
        if itemBuffer == nil { itemBuffer = try fill(nil, with: [UInt32(0)]) }
        let pixels = width * viewHeight
        if auxBuffer == nil || auxBuffer!.length < pixels * 16 {
            auxBuffer = device.makeBuffer(length: pixels * 16, options: .storageModeShared)
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
        e.setBuffer(albedoBuffer, offset: 0, index: 6)
        e.setBuffer(auxBuffer, offset: 0, index: 7)
        e.setBuffer(statsBuffer, offset: 0, index: 8)
        e.setBytes(&packedLook, length: MemoryLayout<GPULook>.stride, index: 9)
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

    /// What the last render decided at one pixel: material, whether it was a cut
    /// face, the ray parameter, and how many intervals the ray collected.
    func auxel(atX x: Int, y: Int, width: Int) -> (material: Int, isCut: Bool, t: Float, met: Int) {
        guard let b = auxBuffer else { return (-1, false, 0, 0) }
        let p = b.contents().assumingMemoryBound(to: SIMD4<Float>.self)
        let v = p[y * width + x]
        return (Int(v.x.rounded()), v.y > 0.5, v.z, Int(v.w.rounded()))
    }

    var auxPointer: UnsafeMutablePointer<SIMD4<Float>>? {
        auxBuffer?.contents().assumingMemoryBound(to: SIMD4<Float>.self)
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

// MARK: - The boundary line
//
// The other half of the diagram convention. A cut face on its own is a flat
// patch of colour; what makes it read as a cut is a line where the material
// changes. The kernel already wrote each pixel's material and whether it was
// cut, so this is a one-pass compare against the right and bottom neighbours —
// cheaper than asking the GPU a second question about geometry it has already
// answered.

func drawCutBoundaries(frame: MTLBuffer, aux: UnsafeMutablePointer<SIMD4<Float>>,
                       width: Int, viewHeight: Int) {
    let pixels = frame.contents().assumingMemoryBound(to: UInt8.self)
    let inkCutCut: Float = 0.58        // material meets material, both in section
    let inkEdge: Float = 0.34          // a silhouette against sky or an uncut face
    func darken(_ i: Int, _ strength: Float) {
        for c in 0..<3 {
            let v: Float = Float(pixels[i * 4 + c])
            let mixed: Float = v * (1 - strength)
            pixels[i * 4 + c] = UInt8(max(min(mixed.rounded(), 255), 0))
        }
    }
    for y in 0..<viewHeight {
        for x in 0..<width {
            let i: Int = y * width + x
            let a: SIMD4<Float> = aux[i]
            var edge: Float = 0
            if x + 1 < width {
                let b: SIMD4<Float> = aux[i + 1]
                if b.x != a.x || b.y != a.y {
                    edge = max(edge, (a.y > 0.5 && b.y > 0.5) ? inkCutCut : inkEdge)
                }
            }
            if y + 1 < viewHeight {
                let b: SIMD4<Float> = aux[i + width]
                if b.x != a.x || b.y != a.y {
                    edge = max(edge, (a.y > 0.5 && b.y > 0.5) ? inkCutCut : inkEdge)
                }
            }
            if edge > 0 { darken(i, edge) }
        }
    }
}

/// A straight cross-dissolve between two frames, in place on `frame`.
/// `alpha` = 1 means `frame` becomes `other` exactly, which is what closes the
/// loop: the last dissolved frame IS frame 0, byte for byte, and nothing ever
/// had to run backwards to get there.
func crossDissolve(frame: MTLBuffer, toward other: [UInt8], alpha: Float, pixels count: Int) {
    if alpha <= 0 { return }
    let p = frame.contents().assumingMemoryBound(to: UInt8.self)
    if alpha >= 1 {
        for i in 0..<(count * 4) { p[i] = other[i] }
        return
    }
    for i in 0..<count {
        for c in 0..<3 {
            let j: Int = i * 4 + c
            let a: Float = Float(p[j])
            let b: Float = Float(other[j])
            let v: Float = a + (b - a) * alpha
            p[j] = UInt8(max(min(v.rounded(), 255), 0))
        }
        p[i * 4 + 3] = 255
    }
}
