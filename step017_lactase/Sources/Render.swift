// A GPU ray tracer for a space-filling protein with a hole cut in it.
//
// Two things are carried over and one is new.
//
// Carried over from step 12: a uniform grid, so a 33,000-sphere tetramer and
// its ambient-occlusion probes are not 33,000 tests each. Carried over from
// step 9: the occlusion itself, without which a packed protein surface is a
// coloured blob — there are no sticks to read and no cast shadows, so every
// bit of the shape is carried by the darkening in its dimples.
//
// New here: the PORTHOLE IS A REAL CUT.
// ------------------------------------
// Step 9 opened its cutaway by shrinking the radius of every atom in the way
// to nothing. That is honest enough — no surface is ever left open, because no
// sphere is ever cut — but it dissolves the rim into a haze of small beads, and
// at this scale that reads as fog rather than as an opening.
//
// So this does the subtraction properly. The porthole is a solid: a cylinder
// bored along the view axis through the active site, closed off by a plane just
// behind it. Every protein sphere is rendered as (sphere MINUS that solid).
// Because the solid is convex, a ray crosses it in exactly one interval, which
// makes the subtraction exact rather than approximate: find where the ray
// enters the sphere; if that point is inside the porthole, slide along to where
// the ray LEAVES the porthole, and if that is still inside the sphere, that is
// the hit.
//
// The face you then see is a genuine cut face, and it is capped: it is shaded
// with the porthole's own surface normal (inward on the bore wall, toward the
// camera on the back plane), never with the sphere's inward-pointing one. So no
// ray ever sees the inside of a sphere through the opening. `capNormal` is
// where that happens, and Tests/main.swift checks it over a grid of rays.
//
// What is NOT cut: the ball-and-stick cast inside the porthole, which is the
// whole reason for cutting. Each shape carries a clip flag in `color.w`.
//
// Amanatides & Woo, "A Fast Voxel Traversal Algorithm for Ray Tracing",
// Eurographics (1987), for the grid walk.
// Tarini, Cignoni & Montani, IEEE TVCG 12:1237 (2006), for molecular occlusion.
// Roth, "Ray casting for modeling solids", CGIP 18:109 (1982), for the CSG.

import Foundation
import Metal
import simd

func radians(_ degrees: Float) -> Float { degrees * .pi / 180 }

/// A sphere or a capped cylinder. One struct for both keeps the grid, the
/// traversal and the occlusion probe to a single code path.
///
///   a     sphere: xyz centre, w radius      cylinder: xyz end A, w radius
///   b     sphere: unused,     w kind = 0    cylinder: xyz end B, w kind = 1
///   color rgb, and w = 1 when the porthole is allowed to cut this shape
struct GPUShape {
    var a: SIMD4<Float>
    var b: SIMD4<Float>
    var color: SIMD4<Float>

    static func sphere(center: SIMD3<Float>, radius: Float, color: SIMD3<Float>,
                       clippable: Bool) -> GPUShape {
        GPUShape(a: SIMD4(center, radius), b: SIMD4(0, 0, 0, 0),
                 color: SIMD4(color, clippable ? 1 : 0))
    }

    static func stick(from p: SIMD3<Float>, to q: SIMD3<Float>, radius: Float,
                      color: SIMD3<Float>) -> GPUShape {
        GPUShape(a: SIMD4(p, radius), b: SIMD4(q, 1), color: SIMD4(color, 0))
    }

    var isCylinder: Bool { b.w > 0.5 }
    var radius: Float { a.w }

    /// The axis-aligned box the shape occupies, for sorting it into the grid.
    func bounds() -> (lo: SIMD3<Float>, hi: SIMD3<Float>) {
        let p = SIMD3(a.x, a.y, a.z)
        let r = a.w
        if !isCylinder { return (p - r, p + r) }
        let q = SIMD3(b.x, b.y, b.z)
        return (simd_min(p, q) - r, simd_max(p, q) + r)
    }
}

/// The porthole: a cylinder of radius `radius` about the line through `center`
/// along `axis`, closed off `back` ångströms behind the centre. `radius` of 0
/// means the porthole is shut and nothing is cut.
struct Porthole {
    var center: SIMD3<Float>
    var axis: SIMD3<Float>       // unit, pointing out of the protein at the camera
    var radius: Float
    var back: Float              // how far behind the centre the bore stops

    static let shut = Porthole(center: .zero, axis: SIMD3(0, 0, 1), radius: 0, back: 0)
    var isOpen: Bool { radius > 0.001 }

    /// Whether a point is inside the solid being subtracted. The CPU mirror of
    /// the kernel's test, used by the tests.
    func contains(_ p: SIMD3<Float>) -> Bool {
        if !isOpen { return false }
        let v: SIMD3<Float> = p - center
        let along: Float = simd_dot(v, axis)
        if along < back { return false }
        let perp: SIMD3<Float> = v - axis * along
        return simd_length(perp) < radius
    }
}

struct GPUCamera {
    var origin: SIMD4<Float>
    var forward: SIMD4<Float>
    var right: SIMD4<Float>
    var up: SIMD4<Float>
    var key: SIMD4<Float>
    var fill: SIMD4<Float>
    var portholeCenter: SIMD4<Float>     // xyz centre, w radius (0 = shut)
    var portholeAxis: SIMD4<Float>       // xyz axis, w back plane
    var tanHalfFOV: Float
    var width: UInt32
    var height: UInt32
    var shapeCount: UInt32
    var samplesPerSide: UInt32
    var useGrid: UInt32
    var aoProbes: UInt32
    var aoDistance: Float
    var aoStrength: Float
    var aoContrast: Float
    var capShade: Float                  // how much darker a cut face is drawn
}

struct GPUGrid {
    var origin: SIMD4<Float>
    var cell: SIMD4<Float>
    var dims: SIMD4<UInt32>
}

struct Camera {
    var origin: SIMD3<Float>
    var target: SIMD3<Float>
    var tanHalfFOV: Float

    var forward: SIMD3<Float> { simd_normalize(target - origin) }
    var right: SIMD3<Float> {
        let f: SIMD3<Float> = forward
        let up: SIMD3<Float> = abs(f.y) > 0.999 ? SIMD3(0, 0, 1) : SIMD3(0, 1, 0)
        return simd_normalize(simd_cross(f, up))
    }
    var up: SIMD3<Float> { simd_cross(right, forward) }

    init(origin: SIMD3<Float>, target: SIMD3<Float>, fov: Float) {
        self.origin = origin
        self.target = target
        self.tanHalfFOV = tan(radians(fov / 2))
    }

    /// A camera `distance` Å from `target`, turned `yaw` degrees about it and
    /// raised `pitch` degrees.
    static func orbit(target: SIMD3<Float>, distance: Float, yaw: Float,
                      pitch: Float, fov: Float) -> Camera {
        let y: Float = radians(yaw), p: Float = radians(pitch)
        let ox: Float = sin(y) * cos(p)
        let oy: Float = sin(p)
        let oz: Float = cos(y) * cos(p)
        let offset: SIMD3<Float> = SIMD3(ox, oy, oz) * distance
        return Camera(origin: target + offset, target: target, fov: fov)
    }

    /// Where a point lands in a width × height image (pixels, from the top left).
    func project(_ p: SIMD3<Float>, width: Int, height: Int) -> SIMD2<Float> {
        let v: SIMD3<Float> = p - origin
        let z: Float = simd_dot(v, forward)
        let aspect: Float = Float(width) / Float(height)
        let ndcX: Float = simd_dot(v, right) / (z * tanHalfFOV * aspect)
        let ndcY: Float = simd_dot(v, up) / (z * tanHalfFOV)
        let px: Float = (ndcX + 1) / 2 * Float(width)
        let py: Float = (1 - ndcY) / 2 * Float(height)
        return SIMD2(px, py)
    }
}

let keyLightWorld = simd_normalize(SIMD3<Float>(-0.45, 0.66, 0.70))
let fillLightWorld = simd_normalize(SIMD3<Float>(0.78, -0.20, 0.58))

let raytraceKernelSource = """
    #include <metal_stdlib>
    using namespace metal;

    struct Shape { float4 a; float4 b; float4 color; };
    struct Camera {
        float4 origin; float4 forward; float4 right; float4 up; float4 key; float4 fill;
        float4 portholeCenter; float4 portholeAxis;
        float tanHalfFOV; uint width; uint height; uint shapeCount; uint samplesPerSide;
        uint useGrid; uint aoProbes; float aoDistance; float aoStrength; float aoContrast;
        float capShade;
    };
    struct Grid { float4 origin; float4 cell; uint4 dims; };

    constant float3 SKY_BLUE = float3(140.0, 200.0, 235.0) / 255.0;
    constant float3 DEEP_BLUE = float3(8.0, 40.0, 90.0) / 255.0;

    constant float BAYER[16] = { 0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5 };
    float3 dither(uint2 gid) {
        float n = (BAYER[(gid.y & 3u) * 4u + (gid.x & 3u)] + 0.5) / 16.0 - 0.5;
        return float3(n * 3.0 / 255.0);
    }

    // ---------------------------------------------------------- the porthole
    //
    // The interval of the ray that lies inside the solid being subtracted: an
    // infinite cylinder about the view axis, cut off by a plane behind the
    // active site. Both are convex, so their intersection is one interval and
    // the CSG subtraction below is exact.
    //
    // Returns (enter, leave); leave <= enter means the ray misses it entirely.
    float2 portholeSpan(float3 ro, float3 rd, constant Camera &cam) {
        float R = cam.portholeCenter.w;
        if (R <= 0.001) return float2(1.0, -1.0);
        float3 C = cam.portholeCenter.xyz;
        float3 A = cam.portholeAxis.xyz;
        float back = cam.portholeAxis.w;

        float3 o = ro - C;
        float oA = dot(o, A), dA = dot(rd, A);
        float3 op = o - A * oA;
        float3 dp = rd - A * dA;

        // the infinite cylinder
        float lo = -1e30, hi = 1e30;
        float k2 = dot(dp, dp);
        float k1 = dot(op, dp);
        float k0 = dot(op, op) - R * R;
        if (k2 < 1e-12) {
            if (k0 > 0.0) return float2(1.0, -1.0);     // parallel and outside
        } else {
            float h = k1 * k1 - k2 * k0;
            if (h < 0.0) return float2(1.0, -1.0);
            float s = sqrt(h);
            lo = (-k1 - s) / k2;
            hi = (-k1 + s) / k2;
        }
        // the half space dot(p - C, A) > back
        if (abs(dA) < 1e-12) {
            if (oA < back) return float2(1.0, -1.0);
        } else {
            float tp = (back - oA) / dA;
            if (dA > 0.0) lo = max(lo, tp); else hi = min(hi, tp);
        }
        if (hi <= lo) return float2(1.0, -1.0);
        return float2(lo, hi);
    }

    // The outward normal of the material left behind where the porthole cuts
    // it, at the point the ray LEAVES the porthole. On the back plane that is
    // the axis itself; on the bore wall it points in at the axis. Either way it
    // faces the ray, which is what makes the cut a cap and not a backface.
    float3 capNormal(float3 p, constant Camera &cam) {
        float3 C = cam.portholeCenter.xyz;
        float3 A = cam.portholeAxis.xyz;
        float R = cam.portholeCenter.w;
        float3 v = p - C;
        float along = dot(v, A);
        float3 perp = v - A * along;
        float lp = length(perp);
        // Whichever boundary the point is on: the plane if it is at the back,
        // the wall if it is at the rim.
        if (lp > R - 1e-3) return -perp / max(lp, 1e-6);
        return A;
    }

    // --------------------------------------------------------- intersections

    inline bool hitSphereRaw(float3 ro, float3 rd, float4 s,
                             thread float &t0, thread float &t1) {
        float r = s.w;
        if (r <= 0.0) return false;
        float3 oc = ro - s.xyz;
        float b = dot(oc, rd);
        float h = b * b - (dot(oc, oc) - r * r);
        if (h < 0.0) return false;
        float q = sqrt(h);
        t0 = -b - q;
        t1 = -b + q;
        return true;
    }

    // One shape, with the porthole subtracted from it when it is clippable.
    // `span` is the ray's porthole interval, computed once per ray.
    inline bool hitShape(float3 ro, float3 rd, device const Shape &s, float2 span,
                         float best, constant Camera &cam,
                         thread float &t, thread float3 &n, thread bool &capped) {
        capped = false;
        if (s.b.w > 0.5) {
            // a capped cylinder: bonds, never cut
            float3 a = s.a.xyz;
            float r = s.a.w;
            float3 ba = s.b.xyz - a;
            float3 oc = ro - a;
            float baba = dot(ba, ba);
            float bard = dot(ba, rd);
            float baoc = dot(ba, oc);
            float k2 = baba - bard * bard;
            float k1 = baba * dot(oc, rd) - baoc * bard;
            float k0 = baba * dot(oc, oc) - baoc * baoc - r * r * baba;
            float h = k1 * k1 - k2 * k0;
            if (h < 0.0 || k2 < 1e-8) return false;
            float tt = (-k1 - sqrt(h)) / k2;
            float y = baoc + tt * bard;
            if (tt <= 1e-3 || tt >= best || y < 0.0 || y > baba) return false;
            t = tt;
            n = (oc + tt * rd - ba * y / baba) / r;
            return true;
        }
        float t0, t1;
        if (!hitSphereRaw(ro, rd, s.a, t0, t1)) return false;
        float enter = max(t0, 1e-3);
        if (enter >= t1) return false;
        if (s.color.w < 0.5 || cam.portholeCenter.w <= 0.001) {
            if (enter >= best) return false;
            t = enter;
            n = (ro + enter * rd - s.a.xyz) / s.a.w;
            return true;
        }
        // subtract the porthole
        if (span.y <= span.x || enter < span.x || enter > span.y) {
            if (enter >= best) return false;
            t = enter;
            n = (ro + enter * rd - s.a.xyz) / s.a.w;
            return true;
        }
        float cut = span.y;                 // where the ray leaves the porthole
        if (cut >= t1 || cut >= best || cut <= 1e-3) return false;
        t = cut;
        n = capNormal(ro + cut * rd, cam);
        capped = true;
        return true;
    }

    float traceAll(float3 ro, float3 rd, device const Shape *shapes, constant Camera &cam,
                   float2 span, thread float3 &normal, thread float3 &color,
                   thread bool &wasCut) {
        float best = 1e30;
        for (uint i = 0; i < cam.shapeCount; i++) {
            float t; float3 n; bool capped;
            if (hitShape(ro, rd, shapes[i], span, best, cam, t, n, capped)) {
                best = t;
                normal = n;
                color = shapes[i].color.rgb;
                wasCut = capped;
            }
        }
        return best;
    }

    float traceGrid(float3 ro, float3 rd, device const Shape *shapes,
                    device const uint *cellStart, device const uint *cellItems,
                    constant Grid &grid, constant Camera &cam, float2 span,
                    float maxT, bool anyHit, bool proteinOnly,
                    thread float3 &normal, thread float3 &color, thread bool &wasCut) {
        float3 lo = grid.origin.xyz;
        float3 cell = grid.cell.xyz;
        int3 dims = int3(grid.dims.xyz);
        float3 hi = lo + float3(dims) * cell;
        float3 inv = 1.0 / rd;

        float3 ta = (lo - ro) * inv, tb = (hi - ro) * inv;
        float3 tsmall = min(ta, tb), tbig = max(ta, tb);
        float tEnter = max(max(tsmall.x, tsmall.y), max(tsmall.z, 0.0));
        float tExit = min(min(tbig.x, tbig.y), tbig.z);
        if (tEnter > tExit || tEnter > maxT) return 1e30;

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
                stp[k] = 0; tMax[k] = 1e30; tDelta[k] = 1e30;
            }
        }

        float best = 1e30;
        for (int guard = 0; guard < 8192; guard++) {
            uint index = (uint(c.z) * grid.dims.y + uint(c.y)) * grid.dims.x + uint(c.x);
            uint from = cellStart[index], to = cellStart[index + 1];
            for (uint i = from; i < to; i++) {
                uint s = cellItems[i];
                // Occlusion is cast by the SPACE-FILLING protein and by nothing
                // else. The ball-and-stick molecules are a schematic drawn at
                // radii chosen to show connectivity — a 0.5 A ball is not a
                // thing that can shade a 1.7 A van der Waals surface, and
                // letting it try makes the whole bore shimmer as the sugar
                // moves. (Step 9 drew the same line, with its occlusion
                // ignoring cylinders.)
                if (proteinOnly && shapes[s].color.w < 0.5) continue;
                float t; float3 n; bool capped;
                if (hitShape(ro, rd, shapes[s], span, best, cam, t, n, capped)) {
                    if (t > maxT) continue;
                    best = t;
                    normal = n;
                    color = shapes[s].color.rgb;
                    wasCut = capped;
                    if (anyHit) return best;
                }
            }
            float leave = min(tMax.x, min(tMax.y, tMax.z));
            if (best <= leave) break;
            if (leave > maxT) break;
            if (tMax.x < tMax.y && tMax.x < tMax.z) {
                c.x += stp.x; if (c.x < 0 || c.x >= dims.x) break; tMax.x += tDelta.x;
            } else if (tMax.y < tMax.z) {
                c.y += stp.y; if (c.y < 0 || c.y >= dims.y) break; tMax.y += tDelta.y;
            } else {
                c.z += stp.z; if (c.z < 0 || c.z >= dims.z) break; tMax.z += tDelta.z;
            }
        }
        return best;
    }

    // A cosine-weighted direction in the hemisphere about n, depending on
    // nothing but i and n — not the pixel, not the frame, not the position — so
    // a surface that does not move keeps exactly the shading it had.
    float3 hemisphere(uint i, uint total, float3 n) {
        float u = (float(i) + 0.5) / float(total);
        float r = sqrt(u);
        float phi = 6.2831853 * fract(float(i) * 0.6180339887);
        float3 t = normalize(abs(n.z) < 0.9 ? cross(n, float3(0.0, 0.0, 1.0))
                                            : cross(n, float3(1.0, 0.0, 0.0)));
        float3 b = cross(n, t);
        return t * (r * cos(phi)) + b * (r * sin(phi)) + n * sqrt(max(0.0, 1.0 - u));
    }

    // The probes see the porthole too. They have to: without it the bore would
    // be pitch black, because every probe fired from the cut face would hit the
    // protein that the porthole is supposed to have taken away.
    float ambientOcclusion(float3 p, float3 n, device const Shape *shapes,
                           device const uint *cellStart, device const uint *cellItems,
                           constant Grid &grid, constant Camera &cam) {
        if (cam.aoProbes == 0) return 1.0;
        float3 start = p + n * 0.05;
        uint open = 0;
        for (uint i = 0; i < cam.aoProbes; i++) {
            float3 dir = hemisphere(i, cam.aoProbes, n);
            float2 span = portholeSpan(start, dir, cam);
            float3 nn, cc;
            bool cut = false;
            float t;
            if (cam.useGrid != 0) {
                t = traceGrid(start, dir, shapes, cellStart, cellItems, grid, cam, span,
                              cam.aoDistance, true, true, nn, cc, cut);
            } else {
                t = 1e30;
                for (uint k = 0; k < cam.shapeCount; k++) {
                    if (shapes[k].color.w < 0.5) continue;
                    float tt; float3 n2; bool cap;
                    if (hitShape(start, dir, shapes[k], span, cam.aoDistance, cam,
                                 tt, n2, cap)) { t = tt; break; }
                }
            }
            if (t > cam.aoDistance) open++;
        }
        return pow(float(open) / float(cam.aoProbes), cam.aoContrast);
    }

    float3 shadeRay(float3 ro, float3 rd, float backgroundY, device const Shape *shapes,
                    device const uint *cellStart, device const uint *cellItems,
                    constant Grid &grid, constant Camera &cam, thread bool &hit) {
        float2 span = portholeSpan(ro, rd, cam);
        float3 n = float3(0, 1, 0), base = float3(1);
        bool capped = false;
        float t = cam.useGrid != 0
            ? traceGrid(ro, rd, shapes, cellStart, cellItems, grid, cam, span, 1e30, false,
                        false, n, base, capped)
            : traceAll(ro, rd, shapes, cam, span, n, base, capped);
        hit = t < 1e29;
        if (!hit) return mix(SKY_BLUE, DEEP_BLUE, backgroundY);

        float3 p = ro + t * rd;

        // A CUT FACE is not a surface the molecule has — it is where the
        // porthole sliced an atom open. Occlusion sampled on it is meaningless
        // and, worse, it speckles: the probes fired from a flat disc buried in
        // the protein catch whatever happens to be nearby, so the bore wall
        // boils from frame to frame. So cut faces are shaded flat, the way a
        // sectioned model is drawn, with one depth cue: the further down the
        // bore a face is, the darker. That is fixed geometry, so it does not
        // move, and a still bore wall costs the GIF nothing.
        if (capped) {
            float along = dot(p - cam.portholeCenter.xyz, cam.portholeAxis.xyz);
            float depth = clamp((along - cam.portholeAxis.w) / 26.0, 0.0, 1.0);
            float lamb = 0.62 + 0.38 * max(dot(n, cam.key.xyz), 0.0);
            return base * cam.capShade * lamb * (0.62 + 0.38 * depth);
        }

        float ao = ambientOcclusion(p, n, shapes, cellStart, cellItems, grid, cam);
        ao = mix(1.0, ao, cam.aoStrength);

        float key = max(dot(n, cam.key.xyz), 0.0);
        float fill = max(dot(n, cam.fill.xyz), 0.0);
        float3 h = normalize(cam.key.xyz - rd);
        float spec = pow(max(dot(n, h), 0.0), 24.0);
        float3 color = base * (0.13 + 0.64 * ao + 0.60 * key * mix(0.30, 1.0, ao)
                               + 0.15 * fill * ao);
        color += 0.05 * spec * ao;
        return color;
    }

    kernel void render(device uchar4 *pixels [[buffer(0)]],
                       constant Camera &cam [[buffer(1)]],
                       device const Shape *shapes [[buffer(2)]],
                       device const uint *cellStart [[buffer(3)]],
                       device const uint *cellItems [[buffer(4)]],
                       constant Grid &grid [[buffer(5)]],
                       uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= cam.width || gid.y >= cam.height) return;
        float aspect = float(cam.width) / float(cam.height);
        uint k = cam.samplesPerSide;
        float3 sum = float3(0);
        float misses = 0;
        for (uint j = 0; j < k; j++) {
            for (uint i = 0; i < k; i++) {
                float sx = float(gid.x) + (float(i) + 0.5) / float(k);
                float sy = float(gid.y) + (float(j) + 0.5) / float(k);
                float ndcX = sx / float(cam.width) * 2.0 - 1.0;
                float ndcY = 1.0 - sy / float(cam.height) * 2.0;
                float3 rd = normalize(cam.forward.xyz + cam.right.xyz * ndcX * aspect * cam.tanHalfFOV
                                      + cam.up.xyz * ndcY * cam.tanHalfFOV);
                bool hit;
                sum += shadeRay(cam.origin.xyz, rd, sy / float(cam.height), shapes,
                                cellStart, cellItems, grid, cam, hit);
                misses += hit ? 0.0 : 1.0;
            }
        }
        float3 color = clamp(sum / float(k * k) + dither(gid) * (misses / float(k * k)), 0.0, 1.0);
        pixels[gid.y * cam.width + gid.x] = uchar4(uchar3(round(color * 255)), 255);
    }
    """

enum RenderError: Error, CustomStringConvertible {
    case noMetalDevice
    case kernelCompile(String)
    case gpu(String)

    var description: String {
        switch self {
        case .noMetalDevice: return "no Metal GPU found"
        case .kernelCompile(let detail): return "could not compile the kernel: \(detail)"
        case .gpu(let detail): return "GPU error: \(detail)"
        }
    }
}

func findDevice() throws -> MTLDevice {
    if let device = MTLCreateSystemDefaultDevice() ?? MTLCopyAllDevices().first { return device }
    throw RenderError.noMetalDevice
}

// MARK: - The uniform grid

/// Which shapes lie in each box of a regular grid over the scene, flattened the
/// way the GPU wants it: `start[c]..<start[c+1]` indexes into `items`.
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

    /// `density` scales the resolution: 1 aims at roughly one box per shape
    /// (Pharr, Jakob & Humphreys, *Physically Based Rendering*, §4.4).
    init(shapes: [GPUShape], density: Float = 1, maxCells: Int = 8_000_000) {
        var lo = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
        var hi = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)
        for s in shapes where s.radius > 0 {
            let b = s.bounds()
            lo = simd_min(lo, b.lo)
            hi = simd_max(hi, b.hi)
        }
        if lo.x > hi.x {
            lo = .zero
            hi = SIMD3(repeating: 1)
        }
        let pad: SIMD3<Float> = simd_max((hi - lo) * 0.001, SIMD3(repeating: 0.01))
        lo -= pad
        hi += pad
        let extent: SIMD3<Float> = hi - lo
        let longest: Float = max(extent.x, max(extent.y, extent.z))
        let cube: Double = pow(Double(max(shapes.count, 1)), 1.0 / 3.0)
        let perUnit: Float = density * 3 * Float(cube) / longest
        var dims = SIMD3<Int>(1, 1, 1)
        for k in 0..<3 {
            let n: Float = extent[k] * perUnit
            dims[k] = max(1, min(1024, Int(n.rounded())))
        }
        while dims.x * dims.y * dims.z > maxCells {
            let k = dims.x >= dims.y && dims.x >= dims.z ? 0 : (dims.y >= dims.z ? 1 : 2)
            dims[k] = max(1, dims[k] / 2)
        }
        self.origin = lo
        self.dims = dims
        let cx: Float = extent.x / Float(dims.x)
        let cy: Float = extent.y / Float(dims.y)
        let cz: Float = extent.z / Float(dims.z)
        self.cellSize = SIMD3(cx, cy, cz)

        let cells = dims.x * dims.y * dims.z
        var counts = [UInt32](repeating: 0, count: cells)
        let gridOrigin = lo, cellSize = self.cellSize

        func span(_ s: GPUShape) -> (SIMD3<Int>, SIMD3<Int>) {
            let b = s.bounds()
            var a = SIMD3<Int>(0, 0, 0), z = SIMD3<Int>(0, 0, 0)
            for k in 0..<3 {
                let u: Float = (b.lo[k] - gridOrigin[k]) / cellSize[k]
                let v: Float = (b.hi[k] - gridOrigin[k]) / cellSize[k]
                a[k] = max(0, min(dims[k] - 1, Int(u.rounded(.down))))
                z[k] = max(0, min(dims[k] - 1, Int(v.rounded(.down))))
            }
            return (a, z)
        }

        for s in shapes where s.radius > 0 {
            let (a, z) = span(s)
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
        for (i, s) in shapes.enumerated() where s.radius > 0 {
            let (a, z) = span(s)
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

    /// The shapes recorded in the box containing `p` (for tests).
    func shapes(at p: SIMD3<Float>) -> [Int] {
        var c = SIMD3<Int>(0, 0, 0)
        for k in 0..<3 {
            let u: Float = (p[k] - origin[k]) / cellSize[k]
            let v = Int(u.rounded(.down))
            if v < 0 || v >= dims[k] { return [] }
            c[k] = v
        }
        let index = (c.z * dims.y + c.y) * dims.x + c.x
        return (Int(start[index])..<Int(start[index + 1])).map { Int(items[$0]) }
    }
}

// MARK: - Occlusion settings

struct AOSettings {
    var probes: Int = 12
    var distance: Float = 9        // ångströms; about three atom diameters
    var strength: Float = 1
    var contrast: Float = 1.6

    static let off = AOSettings(probes: 0, distance: 0, strength: 0, contrast: 1)
}

// MARK: - A CPU mirror of the intersector
//
// Slow and plain, so the tests have something to check the kernel's CSG
// against without a GPU readback, and so the porthole's caps can be checked ray
// by ray.

struct CPUHit {
    var t: Float
    var normal: SIMD3<Float>
    var shape: Int
    var capped: Bool
}

/// The interval of the ray inside the porthole solid. The CPU twin of
/// `portholeSpan`; the tests compare the two.
func portholeSpan(origin ro: SIMD3<Float>, direction rd: SIMD3<Float>,
                  _ hole: Porthole) -> (enter: Float, leave: Float)? {
    if !hole.isOpen { return nil }
    let o: SIMD3<Float> = ro - hole.center
    let a: SIMD3<Float> = hole.axis
    let oA: Float = simd_dot(o, a)
    let dA: Float = simd_dot(rd, a)
    let op: SIMD3<Float> = o - a * oA
    let dp: SIMD3<Float> = rd - a * dA
    var lo: Float = -1e30
    var hi: Float = 1e30
    let k2: Float = simd_dot(dp, dp)
    let k1: Float = simd_dot(op, dp)
    let k0: Float = simd_dot(op, op) - hole.radius * hole.radius
    if k2 < 1e-12 {
        if k0 > 0 { return nil }
    } else {
        let h: Float = k1 * k1 - k2 * k0
        if h < 0 { return nil }
        let s: Float = sqrt(h)
        lo = (-k1 - s) / k2
        hi = (-k1 + s) / k2
    }
    if abs(dA) < 1e-12 {
        if oA < hole.back { return nil }
    } else {
        let tp: Float = (hole.back - oA) / dA
        if dA > 0 { lo = max(lo, tp) } else { hi = min(hi, tp) }
    }
    if hi <= lo { return nil }
    return (lo, hi)
}

/// The outward normal of the cut face where the ray leaves the porthole.
func capNormal(at p: SIMD3<Float>, _ hole: Porthole) -> SIMD3<Float> {
    let v: SIMD3<Float> = p - hole.center
    let along: Float = simd_dot(v, hole.axis)
    let perp: SIMD3<Float> = v - hole.axis * along
    let lp: Float = simd_length(perp)
    if lp > hole.radius - 1e-3 { return -perp / max(lp, 1e-6) }
    return hole.axis
}

/// The nearest hit along a ray, with the porthole subtracted from every
/// clippable sphere. Brute force over every shape.
func traceCPU(origin ro: SIMD3<Float>, direction rd: SIMD3<Float>,
              shapes: [GPUShape], hole: Porthole) -> CPUHit? {
    let span = portholeSpan(origin: ro, direction: rd, hole)
    var best: CPUHit? = nil
    for (i, s) in shapes.enumerated() {
        guard s.radius > 0 else { continue }
        if s.isCylinder {
            let a = SIMD3(s.a.x, s.a.y, s.a.z)
            let r: Float = s.a.w
            let ba: SIMD3<Float> = SIMD3(s.b.x, s.b.y, s.b.z) - a
            let oc: SIMD3<Float> = ro - a
            let baba: Float = simd_dot(ba, ba)
            let bard: Float = simd_dot(ba, rd)
            let baoc: Float = simd_dot(ba, oc)
            let k2: Float = baba - bard * bard
            let k1: Float = baba * simd_dot(oc, rd) - baoc * bard
            let k0: Float = baba * simd_dot(oc, oc) - baoc * baoc - r * r * baba
            let h: Float = k1 * k1 - k2 * k0
            if h < 0 || k2 < 1e-8 { continue }
            let t: Float = (-k1 - sqrt(h)) / k2
            let y: Float = baoc + t * bard
            if t <= 1e-3 || y < 0 || y > baba { continue }
            if let b = best, t >= b.t { continue }
            let n: SIMD3<Float> = (oc + rd * t - ba * (y / baba)) / r
            best = CPUHit(t: t, normal: n, shape: i, capped: false)
            continue
        }
        let c = SIMD3(s.a.x, s.a.y, s.a.z)
        let r: Float = s.a.w
        let oc: SIMD3<Float> = ro - c
        let b: Float = simd_dot(oc, rd)
        let h: Float = b * b - (simd_dot(oc, oc) - r * r)
        if h < 0 { continue }
        let q: Float = sqrt(h)
        let t0: Float = -b - q
        let t1: Float = -b + q
        let enter: Float = max(t0, 1e-3)
        if enter >= t1 { continue }
        let clippable: Bool = s.color.w > 0.5
        var t: Float = enter
        var capped = false
        if clippable, let sp = span, enter >= sp.enter, enter <= sp.leave {
            let cut: Float = sp.leave
            if cut >= t1 || cut <= 1e-3 { continue }
            t = cut
            capped = true
        }
        if let bb = best, t >= bb.t { continue }
        let p: SIMD3<Float> = ro + rd * t
        let n: SIMD3<Float> = capped ? capNormal(at: p, hole) : (p - c) / r
        best = CPUHit(t: t, normal: n, shape: i, capped: capped)
    }
    return best
}

// MARK: - The renderer

final class SceneRenderer {
    let device: MTLDevice
    private let pipeline: MTLComputePipelineState
    private let queue: MTLCommandQueue
    private var shapeBuffer: MTLBuffer?
    private var startBuffer: MTLBuffer?
    private var itemBuffer: MTLBuffer?
    private(set) var grid: UniformGrid?

    init(device: MTLDevice) throws {
        self.device = device
        do {
            let options = MTLCompileOptions()
            // Precise math, as in every step since 3. The laptop's older
            // toolchain knows this name; `mathMode = .safe` is the new one and
            // is not available there.
            options.fastMathEnabled = false
            let library = try device.makeLibrary(source: raytraceKernelSource, options: options)
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

    /// Sorts the scene into a grid. Positions must not change afterwards.
    func buildGrid(_ shapes: [GPUShape], density: Float = 1) throws {
        let g = UniformGrid(shapes: shapes, density: density)
        grid = g
        startBuffer = try fill(nil, with: g.start)
        itemBuffer = try fill(nil, with: g.items.isEmpty ? [UInt32(0)] : g.items)
    }

    /// Ray-traces into the top `viewHeight` rows of `frame`. Returns GPU seconds.
    @discardableResult
    func render(shapes: [GPUShape], camera: Camera, into frame: MTLBuffer,
                width: Int, viewHeight: Int, samplesPerSide: Int = 2,
                useGrid: Bool = true, ao: AOSettings = AOSettings(),
                porthole: Porthole = .shut, capShade: Float = 0.95) throws -> Double {
        let holeCentre = SIMD4<Float>(porthole.center, porthole.radius)
        let holeAxis = SIMD4<Float>(porthole.axis, porthole.back)
        var cam = GPUCamera(origin: SIMD4(camera.origin, 1), forward: SIMD4(camera.forward, 0),
                            right: SIMD4(camera.right, 0), up: SIMD4(camera.up, 0),
                            key: SIMD4(keyLightWorld, 0), fill: SIMD4(fillLightWorld, 0),
                            portholeCenter: holeCentre, portholeAxis: holeAxis,
                            tanHalfFOV: camera.tanHalfFOV, width: UInt32(width),
                            height: UInt32(viewHeight), shapeCount: UInt32(shapes.count),
                            samplesPerSide: UInt32(samplesPerSide), useGrid: useGrid ? 1 : 0,
                            aoProbes: UInt32(ao.probes), aoDistance: ao.distance,
                            aoStrength: ao.strength, aoContrast: ao.contrast,
                            capShade: capShade)
        if useGrid && grid == nil { throw RenderError.gpu("no grid has been built") }
        let gOrigin: SIMD3<Float> = grid?.origin ?? .zero
        let gCell: SIMD3<Float> = grid?.cellSize ?? SIMD3(repeating: 1)
        var g = GPUGrid(origin: SIMD4(gOrigin, 0), cell: SIMD4(gCell, 0),
                        dims: SIMD4(UInt32(grid?.dims.x ?? 1), UInt32(grid?.dims.y ?? 1),
                                    UInt32(grid?.dims.z ?? 1), 0))
        let empty = GPUShape.sphere(center: .zero, radius: 0, color: .zero, clippable: false)
        let list = shapes.isEmpty ? [empty] : shapes
        shapeBuffer = try fill(shapeBuffer, with: list)
        if startBuffer == nil { startBuffer = try fill(nil, with: [UInt32(0), UInt32(0)]) }
        if itemBuffer == nil { itemBuffer = try fill(nil, with: [UInt32(0)]) }

        guard let commands = queue.makeCommandBuffer(),
              let e = commands.makeComputeCommandEncoder() else {
            throw RenderError.gpu("could not create a command encoder")
        }
        e.setComputePipelineState(pipeline)
        e.setBuffer(frame, offset: 0, index: 0)
        e.setBytes(&cam, length: MemoryLayout<GPUCamera>.stride, index: 1)
        e.setBuffer(shapeBuffer, offset: 0, index: 2)
        e.setBuffer(startBuffer, offset: 0, index: 3)
        e.setBuffer(itemBuffer, offset: 0, index: 4)
        e.setBytes(&g, length: MemoryLayout<GPUGrid>.stride, index: 5)
        let w = pipeline.threadExecutionWidth
        let group = MTLSize(width: w, height: pipeline.maxTotalThreadsPerThreadgroup / w, depth: 1)
        e.dispatchThreads(MTLSize(width: width, height: viewHeight, depth: 1),
                          threadsPerThreadgroup: group)
        e.endEncoding()
        commands.commit()
        commands.waitUntilCompleted()
        if let error = commands.error { throw RenderError.gpu(error.localizedDescription) }
        return commands.gpuEndTime - commands.gpuStartTime
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
