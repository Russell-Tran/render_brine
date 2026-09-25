// A cutaway with the physics painted on the cut face.
//
// The traversal is step 13's and the capping is step 15's, and neither is
// rewritten here: the kernel below is step 13's Metal source up to the point
// where it begins integrating Beer–Lambert, with this step's resolve and shading
// appended. So `ellipsoidInterval`, `addInterval` and `gather` are the same
// characters step 13 shipped, including the closest-approach intersection form
// it learned the hard way.
//
// What step 19 adds to step 15's cut is a FIELD. The cut face is not just
// capped, it is coloured by the maximum principal stress the lattice solved at
// that point, and the cracks are drawn on it as they run. That is the whole
// argument of the step: a cutaway of a tooth that shows only anatomy is a
// diagram, and a tooth is a structure. The field arrives as one more `device`
// buffer — 217 × 435 samples of (stress in MPa, crack coverage) over the same
// millimetre coordinates the section was solved in — and is bilinearly sampled,
// so the crack lines antialias and the stress does not band.
//
// Amanatides & Woo, "A Fast Voxel Traversal Algorithm for Ray Tracing",
// Eurographics (1987) — via step 13, unchanged.

import Foundation
import Metal
import simd

// MARK: - splicing step 13's traversal in

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

// MARK: - the frame

let frameWidth = 960
let frameViewHeight = 1160
let frameCaptionHeight = 120
let frameHeight = frameViewHeight + frameCaptionHeight

/// Millimetres to the pixel, one number over the whole field because the camera
/// is orthographic. The scale bar and the crack widths are both derived from it.
let mmPerPixel: Float = 0.0204
let cameraStandOff: Float = 90
let cameraYaw: Float = 0.20            // radians off the cut plane's normal
let cutPlaneZ: Float = 0
let cameraTarget = SIMD3<Float>(0, -3.2, 0)

// MARK: - the look

struct GPULook {
    var sun: SIMD4<Float>
    var sunColour: SIMD4<Float>
    var ambient: SIMD4<Float>
    var skyTop: SIMD4<Float>
    var skyBottom: SIMD4<Float>
    var cut: SIMD4<Float>          // x zCut, y AO radius, z AO rays, w cut desaturation
    var field: SIMD4<Float>        // x,y field origin mm; z,w cells per mm
    var fieldDims: SIMD4<Float>    // x,y dims; z stress range MPa; w stress mix
    var ink: SIMD4<Float>          // x crack darkness, y cut lift, z contact glow, w unused
}

struct Look {
    var sun = simd_normalize(SIMD3<Float>(-0.38, 0.72, 0.58))
    var sunIntensity: Float = 0.82
    var sunColour = SIMD3<Float>(1.00, 0.97, 0.91)
    var ambient = SIMD3<Float>(0.44, 0.47, 0.53)
    var aoStrength: Float = 0.38
    var aoRadius: Float = 1.6
    var aoRays: Int = 4
    var skyTop = SIMD3<Float>(0.105, 0.150, 0.205)
    var skyBottom = SIMD3<Float>(0.215, 0.265, 0.305)
    var cutDesaturation: Float = 0.26
    var cutLift: Float = 1.05
    var stressRangeMPa: Float = 45
    var stressMix: Float = 0
    var crackInk: Float = 0.90
    var contactGlow: Float = 0

    func packed(field: StressField) -> GPULook {
        GPULook(sun: SIMD4(sun, sunIntensity),
                sunColour: SIMD4(sunColour, 0),
                ambient: SIMD4(ambient, aoStrength),
                skyTop: SIMD4(skyTop, 0),
                skyBottom: SIMD4(skyBottom, 0),
                cut: SIMD4(cutPlaneZ, aoRadius, Float(aoRays), cutDesaturation),
                field: SIMD4(field.originX, field.originY, field.perMM, field.perMM),
                fieldDims: SIMD4(Float(field.width), Float(field.height),
                                 stressRangeMPa, stressMix),
                ink: SIMD4(crackInk, cutLift, contactGlow, 0))
    }
}

// MARK: - the stress field, as the GPU takes it

/// A regular grid over the same millimetre coordinates the section was solved
/// in. x carries the maximum principal stress in MPa, y the crack coverage.
struct StressField {
    var originX: Float
    var originY: Float
    var perMM: Float            // samples per millimetre
    var width: Int
    var height: Int
    var texels: [SIMD2<Float>]

    init(originX: Float, originY: Float, perMM: Float, width: Int, height: Int) {
        self.originX = originX
        self.originY = originY
        self.perMM = perMM
        self.width = width
        self.height = height
        self.texels = [SIMD2<Float>](repeating: .zero, count: width * height)
    }

    /// The field the render uses: 0.05 mm samples over the solved rectangle.
    static func standard() -> StressField {
        let perMM: Float = 20
        let x0: Float = -domainHalfWidth - 0.1
        let y0: Float = domainBottom - 0.1
        let w: Int = Int(((domainHalfWidth * 2 + 0.2) * perMM).rounded(.up)) + 1
        let h: Int = Int(((domainTop - domainBottom + 0.2) * perMM).rounded(.up)) + 1
        return StressField(originX: x0, originY: y0, perMM: perMM, width: w, height: h)
    }

    func cell(x: Float, y: Float) -> (Int, Int) {
        let i: Int = Int(((x - originX) * perMM).rounded())
        let j: Int = Int(((y - originY) * perMM).rounded())
        return (i, j)
    }

    func inside(_ i: Int, _ j: Int) -> Bool { i >= 0 && i < width && j >= 0 && j < height }

    subscript(i: Int, j: Int) -> SIMD2<Float> {
        get { inside(i, j) ? texels[j * width + i] : .zero }
        set { if inside(i, j) { texels[j * width + i] = newValue } }
    }
}

// MARK: - the kernel this step adds on top of step 13's traversal

let cutawayShadingSource = """
    // ----------------------------------------------------------------------
    // Everything above this line is step 13's, spliced in unchanged. Everything
    // below is step 19's.
    // ----------------------------------------------------------------------

    #define MAX_MAT 16
    #define M_BONE 0
    #define M_PDL 1
    #define M_CEMENTUM 2
    #define M_ENAMEL 3
    #define M_DEJ 4
    #define M_DENTIN 5
    #define M_PULP 6
    #define M_FISSURE 7

    struct Look {
        float4 sun; float4 sunColour; float4 ambient;
        float4 skyTop; float4 skyBottom; float4 cut;
        float4 field; float4 fieldDims; float4 ink;
    };

    inline float hash13(float3 p) {
        float h = dot(p, float3(127.1, 311.7, 74.7));
        return fract(sin(h) * 43758.5453123);
    }

    // Step 13's per-tissue union, emitting the merged runs instead of
    // integrating them away. Step 15's, unchanged but for the name.
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
    // the inner one wins. Eight nested solids of revolution become eight layers
    // with none of them hollow.
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

    // Which primitive of material `mat` put a surface at ray parameter `t`.
    // Step 15's, unchanged.
    inline float3 surfaceNormal(float3 ro, float3 rd, float t, uint mat,
                                device const Prim *prims,
                                device const uint *cellStart, device const uint *cellItems,
                                constant Grid &grid, constant Camera &cam) {
        float3 p = ro + rd * t;
        float best = 1e30;
        int bi = -1;
        // Averaged over every disc of this material whose own entry is near t,
        // weighted by how near. The body is a STACK of overlapping discs, so at
        // the seam between two of them the single nearest primitive flips from
        // one to the other and the normal jumps — which reads as a row of dark
        // beads down the silhouette. Two discs that meet there have nearly equal
        // entries, so this returns their bisector and the seam disappears. It
        // costs nothing: the loop was already running.
        float3 blended = float3(0.0);
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
                if (d < 0.30) {
                    float3 rel2 = (ro + rd * t) - prims[s].c.xyz;
                    float3 u2 = float3(dot(prims[s].r0.xyz, rel2), dot(prims[s].r1.xyz, rel2),
                                       dot(prims[s].r2.xyz, rel2));
                    float3 g2 = prims[s].r0.xyz * u2.x + prims[s].r1.xyz * u2.y
                              + prims[s].r2.xyz * u2.z;
                    float l2 = length(g2);
                    if (l2 > 1e-9) blended += (g2 / l2) * (1.0 / (d + 0.01));
                }
            }
        }
        if (bi < 0) return -rd;
        float bl = length(blended);
        if (bl > 1e-6) return blended / bl;
        device const Prim &q = prims[bi];
        float3 rel = p - q.c.xyz;
        float3 u = float3(dot(q.r0.xyz, rel), dot(q.r1.xyz, rel), dot(q.r2.xyz, rel));
        float3 g = q.r0.xyz * u.x + q.r1.xyz * u.y + q.r2.xyz * u.z;
        float len = length(g);
        if (len < 1e-9) return -rd;
        return g / len;
    }

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
        float3 origin = p + n * 0.01;
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
                if (s > 0.008 && s < nearest) nearest = s;
            }
            if (nearest < radius) occluded += 1.0 - nearest / radius;
        }
        float mean = occluded / float(rays);
        return 1.0 - look.ambient.w * mean;
    }

    inline float3 sky(constant Look &look, float ndcY) {
        float v = clamp(ndcY * 0.5 + 0.5, 0.0, 1.0);
        return look.skyBottom.rgb + (look.skyTop.rgb - look.skyBottom.rgb) * v;
    }

    // The field, bilinear. Sampling it bilinearly is what makes a crack line
    // drawn at 0.05 mm read as a line rather than as a staircase, and it costs
    // four taps of a buffer the size of a thumbnail.
    inline float2 sampleField(float2 mm, device const float2 *field, constant Look &look) {
        float fx = (mm.x - look.field.x) * look.field.z;
        float fy = (mm.y - look.field.y) * look.field.w;
        int w = int(look.fieldDims.x);
        int h = int(look.fieldDims.y);
        if (fx < 0.0 || fy < 0.0 || fx > float(w - 1) || fy > float(h - 1)) return float2(0.0);
        int i0 = int(floor(fx));
        int j0 = int(floor(fy));
        int i1 = min(i0 + 1, w - 1);
        int j1 = min(j0 + 1, h - 1);
        float ax = fx - float(i0);
        float ay = fy - float(j0);
        float2 a = mix(field[j0 * w + i0], field[j0 * w + i1], ax);
        float2 b = mix(field[j1 * w + i0], field[j1 * w + i1], ax);
        return mix(a, b, ay);
    }

    // A diverging ramp centred on zero: compression cool, tension warm, and the
    // far end of tension the colour of the number that matters. Maximum
    // PRINCIPAL stress, not von Mises — a brittle solid is broken by tension and
    // a von Mises map would paint the compressed side of the neck as if it were
    // in danger.
    inline float3 stressColour(float mpa, float range) {
        float t = clamp(mpa / range, -1.0, 1.0);
        float3 neutral = float3(0.90, 0.89, 0.86);
        if (t >= 0.0) {
            float3 warm = float3(0.97, 0.80, 0.30);
            float3 hot = float3(0.85, 0.22, 0.14);
            if (t < 0.5) return mix(neutral, warm, t * 2.0);
            return mix(warm, hot, (t - 0.5) * 2.0);
        }
        float s = -t;
        float3 pale = float3(0.66, 0.78, 0.86);
        float3 deep = float3(0.16, 0.36, 0.60);
        if (s < 0.5) return mix(neutral, pale, s * 2.0);
        return mix(pale, deep, (s - 0.5) * 2.0);
    }

    // Flat, matte, a little desaturated, with a fine grain that says which
    // mineral it is. The contrast against a lit uncut surface is what makes a
    // cutaway read, and it is deliberately not subtle.
    inline float3 cutShade(float3 albedo, uint mat, float3 p, constant Look &look) {
        float luma = dot(albedo, float3(0.299, 0.587, 0.114));
        float3 flat = mix(albedo, float3(luma), look.cut.w) * look.ink.y;
        if (mat == uint(M_ENAMEL)) {
            // Enamel rods run out from the junction toward the surface, so in
            // this section they streak radially.
            float ang = atan2(p.y + 3.0, p.x);
            float s = sin(ang * 90.0);
            flat *= 0.9965 + 0.0035 * s;
        } else if (mat == uint(M_DENTIN)) {
            // Tubules, likewise radial, and coarser.
            float ang = atan2(p.y + 3.0, p.x);
            float s = sin(ang * 64.0 + 0.7);
            flat *= 0.9955 + 0.0045 * s;
        } else if (mat == uint(M_BONE)) {
            float g = hash13(floor(p * 14.0));
            float g2 = hash13(floor(p * 3.3) + 11.0);
            flat *= 0.93 + 0.09 * (0.55 * g + 0.45 * g2);
        } else if (mat == uint(M_PDL)) {
            float s = sin(p.y * 26.0 + p.x * 4.0);
            flat *= 0.94 + 0.08 * s;
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
                        device const float2 *field [[buffer(10)]],
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

                float tCut = (look.cut.x - ro.z) / rd.z;
                if (tCut < 0.0) tCut = 0.0;

                float eps = 1e-4;
                int mat = coverAt(run, runMat, nr, tCut + eps);
                float t = tCut;
                bool isCut = true;
                if (mat < 0) {
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
                    colour = sky(look, ndcY);
                } else {
                    float3 p = ro + rd * t;
                    float3 alb = albedo[mat].rgb;
                    if (isCut) {
                        colour = cutShade(alb, uint(mat), p, look);
                        bool carries = (mat == M_ENAMEL || mat == M_DEJ
                                        || mat == M_DENTIN || mat == M_CEMENTUM);
                        if (carries) {
                            float2 fv = sampleField(p.xy, field, look);
                            float mixAmount = look.fieldDims.w;
                            if (mixAmount > 0.0) {
                                // Multiplied in as a TINT, not lerped over the
                                // top. Lerping toward the ramp paints every
                                // unstressed part of the tooth the ramp's
                                // neutral colour, which is nearly white, and the
                                // anatomy disappears under the physics. Dividing
                                // the ramp by its own neutral gives a factor
                                // that is exactly 1 where the stress is zero, so
                                // the enamel still looks like enamel and only
                                // the loaded parts change colour.
                                float3 s = stressColour(fv.x, look.fieldDims.z);
                                float3 tint = s / float3(0.90, 0.89, 0.86);
                                colour *= mix(float3(1.0), tint, mixAmount);
                            }
                            float crack = clamp(fv.y, 0.0, 1.0);
                            if (crack > 0.0) {
                                float3 seam = float3(0.05, 0.04, 0.04);
                                colour = mix(colour, seam, crack * look.ink.x);
                            }
                        }
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
        aux[gid.y * cam.width + gid.x] = float4(float(firstMat), firstCut, firstT, float(met));
        if (overflow > 0) atomic_fetch_add_explicit(stats, overflow, memory_order_relaxed);
        pixels[gid.y * cam.width + gid.x] = uchar4(uchar3(round(out * 255.0)), 255);
    }
    """

let cutawayKernelSource: String = traversalPrefix + cutawayShadingSource

// MARK: - the camera

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

// MARK: - a CPU reference for the same resolve

struct CutHit {
    var t: Float
    var material: Material
    var isCut: Bool
    var intervals: Int
}

func cpuResolve(origin ro: SIMD3<Float>, direction rd: SIMD3<Float>,
                prims: [GPUPrim], cutZ: Float) -> CutHit? {
    var list: [(Float, Float, Int)] = []
    list.reserveCapacity(64)
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
    let eps: Float = 1e-4
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

/// How many intervals the busiest ray in a frame collects. Step 13's list holds
/// forty; this step's stacks are sized against this number, and a test fails if
/// the worst ray in the real frame comes within a few of the limit.
func worstIntervalCount(prims: [GPUPrim], camera: Camera, width: Int, height: Int,
                        stride: Int = 7) -> Int {
    var worst = 0
    var y = 0
    while y < height {
        var x = 0
        while x < width {
            let (o, d) = camera.ray(sx: Float(x) + 0.5, sy: Float(y) + 0.5,
                                    width: width, height: height)
            var n = 0
            for p in prims {
                guard let (a, b) = cpuInterval(origin: o, direction: d, prim: p) else { continue }
                if b <= 0 { continue }
                if b <= max(a, 0) { continue }
                n += 1
            }
            if n > worst { worst = n }
            x += stride
        }
        y += stride
    }
    return worst
}

// MARK: - the renderer

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
    private var fieldBuffer: MTLBuffer?
    private(set) var grid: UniformGrid?
    private(set) var overflowCount: UInt32 = 0
    private(set) var lastBuildSeconds: Double = 0

    init(device: MTLDevice) throws {
        self.device = device
        do {
            let options = MTLCompileOptions()
            // Off, as in every step since 3. Here it matters because a cut face
            // is decided by whether a run's entry is before or after the clip
            // plane, and reassociating that comparison is exactly the kind of
            // drift that punches a hole through a cap.
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

    /// The geometry never moves in this step — only the field painted on it
    /// does — so the grid is built once and reused for every frame.
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
                field: StressField, into frame: MTLBuffer, width: Int, viewHeight: Int,
                samplesPerSide: Int = 2, useGrid: Bool = true) throws -> Double {
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
                            mutations: 0)
        if useGrid && grid == nil { throw RenderError.gpu("no grid has been built") }
        var g = GPUGrid(origin: SIMD4(grid?.origin ?? .zero, 0),
                        cell: SIMD4(grid?.cellSize ?? SIMD3(repeating: 1), 0),
                        dims: SIMD4(UInt32(grid?.dims.x ?? 1), UInt32(grid?.dims.y ?? 1),
                                    UInt32(grid?.dims.z ?? 1), 0))
        var packedLook: GPULook = look.packed(field: field)
        let list = prims.isEmpty
            ? [GPUPrim.disc(y: 1e6, radius: 0.001, half: 0.001, material: .bone)] : prims
        primBuffer = try fill(primBuffer, with: list)
        let table: [SIMD4<Float>] = (0..<16).map { i in
            i < albedo.count ? SIMD4(albedo[i], 0) : SIMD4(repeating: 0)
        }
        albedoBuffer = try fill(albedoBuffer, with: table)
        fieldBuffer = try fill(fieldBuffer, with: field.texels)
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
        e.setBuffer(fieldBuffer, offset: 0, index: 10)
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

// MARK: - the boundary line
//
// The other half of the diagram convention, straight from step 15: a cut face on
// its own is a flat patch of colour; what makes it read as a cut is a line where
// the material changes.

func drawCutBoundaries(frame: MTLBuffer, aux: UnsafeMutablePointer<SIMD4<Float>>,
                       width: Int, viewHeight: Int) {
    let pixels = frame.contents().assumingMemoryBound(to: UInt8.self)
    let inkCutCut: Float = 0.50
    let inkEdge: Float = 0.32
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

/// A straight cross-dissolve between two frames, in place. `alpha` = 1 means
/// `frame` becomes `other` exactly, which is what closes the loop: a cracked
/// tooth cannot become an uncracked one by running anything backwards, so the
/// seam is a dissolve to a FRESH TOOTH — the same device step 18 used when it
/// drained and refilled the polarimeter cell.
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
