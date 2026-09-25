// A GPU ray tracer with a uniform grid AND ambient occlusion, on a scene that
// moves.
//
// The two techniques arrive from different steps and, until now, both leaned on
// the same convenience: the scene never moved.
//
//   Step 8a built its grid ONCE, because a turning plasmid is a rigid body —
//   you can orbit the camera instead of moving 92,000 shapes.
//   Step 9 kept ambient occlusion from crawling the same way: the molecule sat
//   still in its own coordinates while the camera and lights turned around it,
//   so a given point on the surface drew the same probe rays every frame.
//
// Neither holds here. The DNA genuinely deforms as the loop opens and closes,
// so:
//
//   * the grid is REBUILT EVERY FRAME (see `SceneRenderer.frame`), and the cost
//     of doing so is measured separately from the cost of tracing through it;
//   * occlusion is recomputed against geometry that really did move. The probe
//     pattern is still built from the surface normal with no per-point jitter,
//     which is what keeps it steady: as the geometry turns, the frame built on
//     the normal turns smoothly with it.
//
// Probe rays go through the grid too. They are short, so they usually touch one
// or two boxes and stop — which is why 12 probes a hit is affordable on 20,000
// shapes when step 9 could only afford it on 1,771 by brute force.
//
// References: Amanatides & Woo, "A Fast Voxel Traversal Algorithm for Ray
// Tracing", Eurographics (1987) for the grid walk; Tarini, Cignoni & Montani,
// IEEE TVCG 12:1237 (2006) for molecular ambient occlusion.

import Foundation
import Metal
import simd

func radians(_ degrees: Float) -> Float { degrees * .pi / 180 }

/// A sphere (kind 0: `a` is centre and radius) or a cylinder (kind 1: from `a`
/// to `b`, with `a.w` the radius). Radius zero means "skip me".
struct GPUShape {
    var a: SIMD4<Float>
    var b: SIMD4<Float>       // w = kind
    var color: SIMD4<Float>

    static func sphere(center: SIMD3<Float>, radius: Float, color: SIMD3<Float>) -> GPUShape {
        GPUShape(a: SIMD4(center, radius), b: SIMD4(0, 0, 0, 0), color: SIMD4(color, 1))
    }
    static func cylinder(from p: SIMD3<Float>, to q: SIMD3<Float>, radius: Float, color: SIMD3<Float>) -> GPUShape {
        GPUShape(a: SIMD4(p, radius), b: SIMD4(q, 1), color: SIMD4(color, 1))
    }

    var isSphere: Bool { b.w < 0.5 }

    func bounds() -> (lo: SIMD3<Float>, hi: SIMD3<Float>) {
        let r = a.w
        let p = SIMD3(a.x, a.y, a.z)
        if isSphere { return (p - r, p + r) }
        let q = SIMD3(b.x, b.y, b.z)
        return (simd_min(p, q) - r, simd_max(p, q) + r)
    }
}

struct GPUCamera {
    var origin: SIMD4<Float>
    var forward: SIMD4<Float>
    var right: SIMD4<Float>
    var up: SIMD4<Float>
    var key: SIMD4<Float>
    var fill: SIMD4<Float>
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
    var aoOnly: UInt32
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
        let f = forward
        let up = abs(f.y) > 0.999 ? SIMD3<Float>(0, 0, 1) : SIMD3<Float>(0, 1, 0)
        return simd_normalize(simd_cross(f, up))
    }
    var up: SIMD3<Float> { simd_cross(right, forward) }

    static func orbit(target: SIMD3<Float>, distance: Float, yaw: Float, pitch: Float, fov: Float) -> Camera {
        let y = radians(yaw), p = radians(pitch)
        let d = SIMD3<Float>(sin(y) * cos(p), sin(p), cos(y) * cos(p))
        return Camera(origin: target + d * distance, target: target, tanHalfFOV: tan(radians(fov / 2)))
    }

    func project(_ p: SIMD3<Float>, width: Int, height: Int) -> SIMD2<Float> {
        let v = p - origin
        let z: Float = simd_dot(v, forward)
        let aspect: Float = Float(width) / Float(height)
        let ndcX: Float = simd_dot(v, right) / (z * tanHalfFOV * aspect)
        let ndcY: Float = simd_dot(v, up) / (z * tanHalfFOV)
        return SIMD2((ndcX + 1) / 2 * Float(width), (1 - ndcY) / 2 * Float(height))
    }

    /// Is this point inside the view, allowing `margin` extra at each edge?
    func sees(_ p: SIMD3<Float>, width: Int, height: Int, margin: Float) -> Bool {
        let v = p - origin
        if simd_dot(v, forward) <= 0 { return false }
        let s = project(p, width: width, height: height)
        return s.x > -margin && s.x < Float(width) + margin && s.y > -margin && s.y < Float(height) + margin
    }
}

let bondRadius: Float = 0.30

let raytraceKernelSource = """
    #include <metal_stdlib>
    using namespace metal;

    struct Shape { float4 a; float4 b; float4 color; };
    struct Camera {
        float4 origin; float4 forward; float4 right; float4 up; float4 key; float4 fill;
        float tanHalfFOV; uint width; uint height; uint shapeCount; uint samplesPerSide; uint useGrid;
        uint aoProbes; float aoDistance; float aoStrength; float aoContrast; uint aoOnly;
    };
    struct Grid { float4 origin; float4 cell; uint4 dims; };

    constant float3 SKY_BLUE = float3(140.0, 200.0, 235.0) / 255.0;
    constant float3 DEEP_BLUE = float3(8.0, 40.0, 90.0) / 255.0;

    constant float BAYER[16] = { 0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5 };
    float3 dither(uint2 gid) {
        float n = (BAYER[(gid.y & 3u) * 4u + (gid.x & 3u)] + 0.5) / 16.0 - 0.5;
        return float3(n * 3.0 / 255.0);
    }

    // Nearest hit on one shape, nearer than `best`.
    bool hitShape(float3 ro, float3 rd, Shape s, float best, thread float &tt, thread float3 &n) {
        float r = s.a.w;
        if (r <= 0.0) return false;
        if (s.b.w < 0.5) {
            float3 c = s.a.xyz;
            float3 oc = ro - c;
            float bb = dot(oc, rd);
            float h = bb * bb - (dot(oc, oc) - r * r);
            if (h < 0.0) return false;
            float t = -bb - sqrt(h);
            if (t <= 1e-3 || t >= best) return false;
            tt = t; n = normalize(ro + t * rd - c);
            return true;
        }
        float3 a = s.a.xyz;
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
        float t = (-k1 - sqrt(h)) / k2;
        float y = baoc + t * bard;
        if (t <= 1e-3 || t >= best || y <= 0.0 || y >= baba) return false;
        tt = t;
        n = (oc + t * rd - ba * y / baba) / r;
        return true;
    }

    float traceAll(float3 ro, float3 rd, device const Shape *shapes, uint count,
                   thread float3 &normal, thread float3 &color) {
        float best = 1e30;
        for (uint i = 0; i < count; i++) {
            float t; float3 n;
            if (hitShape(ro, rd, shapes[i], best, t, n)) { best = t; normal = n; color = shapes[i].color.rgb; }
        }
        return best;
    }

    // Set up the DDA walk. Returns false if the ray misses the grid entirely.
    bool gridSetup(float3 ro, float3 rd, constant Grid &grid,
                   thread int3 &c, thread int3 &stp, thread float3 &tMax, thread float3 &tDelta,
                   thread float &tEnter) {
        float3 lo = grid.origin.xyz;
        float3 cell = grid.cell.xyz;
        int3 dims = int3(grid.dims.xyz);
        float3 hi = lo + float3(dims) * cell;
        float3 inv = 1.0 / rd;
        float3 ta = (lo - ro) * inv, tb = (hi - ro) * inv;
        float3 tsmall = min(ta, tb), tbig = max(ta, tb);
        tEnter = max(max(tsmall.x, tsmall.y), max(tsmall.z, 0.0));
        float tExit = min(min(tbig.x, tbig.y), tbig.z);
        if (tEnter > tExit) return false;
        float3 p = ro + rd * (tEnter + 1e-4);
        c = clamp(int3(floor((p - lo) / cell)), int3(0), dims - 1);
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
        return true;
    }

    float traceGrid(float3 ro, float3 rd, device const Shape *shapes,
                    device const uint *cellStart, device const uint *cellItems,
                    constant Grid &grid, thread float3 &normal, thread float3 &color) {
        int3 c, stp; float3 tMax, tDelta; float tEnter;
        if (!gridSetup(ro, rd, grid, c, stp, tMax, tDelta, tEnter)) return 1e30;
        int3 dims = int3(grid.dims.xyz);
        float best = 1e30;
        for (int guard = 0; guard < 4096; guard++) {
            uint index = (uint(c.z) * grid.dims.y + uint(c.y)) * grid.dims.x + uint(c.x);
            for (uint i = cellStart[index]; i < cellStart[index + 1]; i++) {
                uint s = cellItems[i];
                float t; float3 n;
                if (hitShape(ro, rd, shapes[s], best, t, n)) { best = t; normal = n; color = shapes[s].color.rgb; }
            }
            float leave = min(tMax.x, min(tMax.y, tMax.z));
            if (best <= leave) break;
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

    // Is anything within `maxDist`? Probe rays want a yes or no, so this stops
    // at the first hit. Through the grid, a short probe usually visits one or
    // two boxes — which is what makes 12 probes a hit affordable here.
    bool occluded(float3 ro, float3 rd, float maxDist, device const Shape *shapes,
                  device const uint *cellStart, device const uint *cellItems,
                  constant Grid &grid, constant Camera &cam) {
        if (cam.useGrid == 0) {
            for (uint i = 0; i < cam.shapeCount; i++) {
                float t; float3 n;
                if (hitShape(ro, rd, shapes[i], maxDist, t, n)) return true;
            }
            return false;
        }
        int3 c, stp; float3 tMax, tDelta; float tEnter;
        if (!gridSetup(ro, rd, grid, c, stp, tMax, tDelta, tEnter)) return false;
        int3 dims = int3(grid.dims.xyz);
        for (int guard = 0; guard < 256; guard++) {
            uint index = (uint(c.z) * grid.dims.y + uint(c.y)) * grid.dims.x + uint(c.x);
            for (uint i = cellStart[index]; i < cellStart[index + 1]; i++) {
                float t; float3 n;
                if (hitShape(ro, rd, shapes[cellItems[i]], maxDist, t, n)) return true;
            }
            float leave = min(tMax.x, min(tMax.y, tMax.z));
            if (leave > maxDist) break;
            if (tMax.x < tMax.y && tMax.x < tMax.z) {
                c.x += stp.x; if (c.x < 0 || c.x >= dims.x) break; tMax.x += tDelta.x;
            } else if (tMax.y < tMax.z) {
                c.y += stp.y; if (c.y < 0 || c.y >= dims.y) break; tMax.y += tDelta.y;
            } else {
                c.z += stp.z; if (c.z < 0 || c.z >= dims.z) break; tMax.z += tDelta.z;
            }
        }
        return false;
    }

    // A cosine-weighted direction in the hemisphere about n. Deterministic in
    // i and n, with NO per-point jitter: neighbouring points then draw nearly
    // the same spray, which is what keeps the surface smooth rather than
    // speckled, and what keeps it steady as the geometry moves.
    float3 hemisphere(uint i, uint n_total, float3 n) {
        float u = (float(i) + 0.5) / float(n_total);
        float r = sqrt(u);
        float phi = 6.2831853 * fract(float(i) * 0.6180339887);
        float3 t = normalize(abs(n.z) < 0.9 ? cross(n, float3(0.0, 0.0, 1.0)) : cross(n, float3(1.0, 0.0, 0.0)));
        float3 b = cross(n, t);
        return t * (r * cos(phi)) + b * (r * sin(phi)) + n * sqrt(max(0.0, 1.0 - u));
    }

    float ambientOcclusion(float3 p, float3 nrm, device const Shape *shapes,
                           device const uint *cellStart, device const uint *cellItems,
                           constant Grid &grid, constant Camera &cam) {
        if (cam.aoProbes == 0) return 1.0;
        float3 start = p + nrm * 0.03;
        uint open = 0;
        for (uint i = 0; i < cam.aoProbes; i++) {
            float3 dir = hemisphere(i, cam.aoProbes, nrm);
            if (!occluded(start, dir, cam.aoDistance, shapes, cellStart, cellItems, grid, cam)) open++;
        }
        return pow(float(open) / float(cam.aoProbes), cam.aoContrast);
    }

    float3 shadeRay(float3 ro, float3 rd, float backgroundY, device const Shape *shapes,
                    device const uint *cellStart, device const uint *cellItems,
                    constant Grid &grid, constant Camera &cam, thread bool &hit) {
        float3 n = float3(0, 1, 0), base = float3(1);
        float t = cam.useGrid != 0
            ? traceGrid(ro, rd, shapes, cellStart, cellItems, grid, n, base)
            : traceAll(ro, rd, shapes, cam.shapeCount, n, base);
        hit = t < 1e29;
        if (!hit) return mix(SKY_BLUE, DEEP_BLUE, backgroundY);
        float3 p = ro + t * rd;
        float ao = ambientOcclusion(p, n, shapes, cellStart, cellItems, grid, cam);
        ao = mix(1.0, ao, cam.aoStrength);
        if (cam.aoOnly != 0) return float3(ao);
        float key = max(dot(n, cam.key.xyz), 0.0);
        float fill = max(dot(n, cam.fill.xyz), 0.0);
        float3 h = normalize(cam.key.xyz - rd);
        float spec = pow(max(dot(n, h), 0.0), 24.0);
        float3 color = base * (0.14 + 0.66 * ao + 0.58 * key * mix(0.30, 1.0, ao) + 0.16 * fill * ao);
        color += 0.06 * spec * ao;
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
        case .kernelCompile(let d): return "could not compile the kernel: \(d)"
        case .gpu(let d): return "GPU error: \(d)"
        }
    }
}

func findDevice() throws -> MTLDevice {
    if let device = MTLCreateSystemDefaultDevice() ?? MTLCopyAllDevices().first { return device }
    throw RenderError.noMetalDevice
}

// MARK: - The uniform grid

/// Which shapes lie in each box of a regular grid over the scene.
///
/// Step 8a built one of these once and kept it. Here a new one is built for
/// every frame, because the scene changes shape; `SceneRenderer.frame` times
/// that separately from the trace so the split can be reported.
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

    init(shapes: [GPUShape], density: Float = 1, maxCells: Int = 4_000_000) {
        var lo = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
        var hi = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)
        for s in shapes where s.a.w > 0 {
            let b = s.bounds()
            lo = simd_min(lo, b.lo)
            hi = simd_max(hi, b.hi)
        }
        if shapes.isEmpty || lo.x > hi.x { lo = .zero; hi = SIMD3(repeating: 1) }
        let pad = simd_max((hi - lo) * 0.001, SIMD3(repeating: 0.01))
        lo -= pad
        hi += pad
        let extent = hi - lo
        let longest = max(extent.x, max(extent.y, extent.z))
        let perUnit = density * 3 * Float(pow(Double(max(shapes.count, 1)), 1.0 / 3.0)) / longest
        var dims = SIMD3<Int>(1, 1, 1)
        for k in 0..<3 { dims[k] = max(1, min(512, Int((extent[k] * perUnit).rounded()))) }
        while dims.x * dims.y * dims.z > maxCells {
            let k = dims.x >= dims.y && dims.x >= dims.z ? 0 : (dims.y >= dims.z ? 1 : 2)
            dims[k] = max(1, dims[k] / 2)
        }
        self.origin = lo
        self.dims = dims
        self.cellSize = SIMD3(extent.x / Float(dims.x), extent.y / Float(dims.y), extent.z / Float(dims.z))

        let cells = dims.x * dims.y * dims.z
        var counts = [UInt32](repeating: 0, count: cells)
        let org = lo, cs = self.cellSize
        func span(_ s: GPUShape) -> (SIMD3<Int>, SIMD3<Int>) {
            let b = s.bounds()
            var a = SIMD3<Int>(0, 0, 0), z = SIMD3<Int>(0, 0, 0)
            for k in 0..<3 {
                a[k] = max(0, min(dims[k] - 1, Int(((b.lo[k] - org[k]) / cs[k]).rounded(.down))))
                z[k] = max(0, min(dims[k] - 1, Int(((b.hi[k] - org[k]) / cs[k]).rounded(.down))))
            }
            return (a, z)
        }
        for s in shapes where s.a.w > 0 {
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
        for c in 0..<cells { start[c] = running; running += counts[c] }
        start[cells] = running
        var items = [UInt32](repeating: 0, count: Int(running))
        var cursor = start
        for (i, s) in shapes.enumerated() where s.a.w > 0 {
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
            let v = Int(((p[k] - origin[k]) / cellSize[k]).rounded(.down))
            if v < 0 || v >= dims[k] { return [] }
            c[k] = v
        }
        let index = (c.z * dims.y + c.y) * dims.x + c.x
        return (Int(start[index])..<Int(start[index + 1])).map { Int(items[$0]) }
    }
}

/// How occlusion is sampled.
struct AOSettings {
    var probes: Int = 12
    var distance: Float = 8.0
    var strength: Float = 1.0
    var contrast: Float = 1.7
    var only: Bool = false
    static let off = AOSettings(probes: 0, distance: 0, strength: 0, contrast: 1, only: false)
}

/// What one frame cost, split the way this step cares about.
struct FrameCost {
    var build: Double = 0     // seconds spent sorting the scene into boxes
    var trace: Double = 0     // seconds the GPU spent tracing through them
    var shapes: Int = 0
    var cells: Int = 0
    var occupancy: Double = 0
}

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
            // Precise math, as every step since 3. (Newer SDKs call this
            // mathMode = .safe; the old name builds on the laptop's toolchain too.)
            options.fastMathEnabled = false
            let library = try device.makeLibrary(source: raytraceKernelSource, options: options)
            guard let function = library.makeFunction(name: "render") else {
                throw RenderError.kernelCompile("no kernel named render")
            }
            pipeline = try device.makeComputePipelineState(function: function)
        } catch let e as RenderError {
            throw e
        } catch {
            throw RenderError.kernelCompile(error.localizedDescription)
        }
        guard let q = device.makeCommandQueue() else { throw RenderError.gpu("no command queue") }
        self.queue = q
    }

    /// Sorts the scene into boxes. Called once per frame here, not once per run.
    @discardableResult
    func buildGrid(_ shapes: [GPUShape], density: Float = 1) throws -> Double {
        let t0 = Date()
        let g = UniformGrid(shapes: shapes, density: density)
        grid = g
        startBuffer = try fill(startBuffer, with: g.start)
        itemBuffer = try fill(itemBuffer, with: g.items.isEmpty ? [UInt32(0)] : g.items)
        return Date().timeIntervalSince(t0)
    }

    /// Builds the grid for this frame's geometry and then traces it.
    func frame(shapes: [GPUShape], camera: Camera, into buffer: MTLBuffer,
               width: Int, viewHeight: Int, samplesPerSide: Int = 2,
               ao: AOSettings = AOSettings(), density: Float = 1) throws -> FrameCost {
        var cost = FrameCost()
        cost.build = try buildGrid(shapes, density: density)
        cost.trace = try render(shapes: shapes, camera: camera, into: buffer,
                                width: width, viewHeight: viewHeight,
                                samplesPerSide: samplesPerSide, ao: ao)
        cost.shapes = shapes.count
        cost.cells = grid?.cellCount ?? 0
        cost.occupancy = grid?.averageOccupancy ?? 0
        return cost
    }

    /// Ray-traces into the top `viewHeight` rows. Returns GPU seconds.
    @discardableResult
    func render(shapes: [GPUShape], camera: Camera, into frame: MTLBuffer,
                width: Int, viewHeight: Int, samplesPerSide: Int = 2,
                ao: AOSettings = AOSettings(), useGrid: Bool = true) throws -> Double {
        let keyDir = simd_normalize(SIMD3<Float>(-0.45, 0.62, 0.72))
        let fillDir = simd_normalize(SIMD3<Float>(0.80, -0.23, 0.57))
        var cam = GPUCamera(origin: SIMD4(camera.origin, 1), forward: SIMD4(camera.forward, 0),
                            right: SIMD4(camera.right, 0), up: SIMD4(camera.up, 0),
                            key: SIMD4(keyDir, 0), fill: SIMD4(fillDir, 0),
                            tanHalfFOV: camera.tanHalfFOV, width: UInt32(width), height: UInt32(viewHeight),
                            shapeCount: UInt32(shapes.count), samplesPerSide: UInt32(samplesPerSide),
                            useGrid: useGrid ? 1 : 0,
                            aoProbes: UInt32(ao.probes), aoDistance: ao.distance,
                            aoStrength: ao.strength, aoContrast: ao.contrast, aoOnly: ao.only ? 1 : 0)
        if useGrid && grid == nil { throw RenderError.gpu("no grid has been built") }
        var g = GPUGrid(origin: SIMD4(grid?.origin ?? .zero, 0),
                        cell: SIMD4(grid?.cellSize ?? SIMD3(repeating: 1), 0),
                        dims: SIMD4(UInt32(grid?.dims.x ?? 1), UInt32(grid?.dims.y ?? 1),
                                    UInt32(grid?.dims.z ?? 1), 0))
        let list = shapes.isEmpty ? [GPUShape.sphere(center: .zero, radius: 0, color: .zero)] : shapes
        shapeBuffer = try fill(shapeBuffer, with: list)
        if startBuffer == nil { startBuffer = try fill(nil, with: [UInt32(0), UInt32(0)]) }
        if itemBuffer == nil { itemBuffer = try fill(nil, with: [UInt32(0)]) }

        guard let commands = queue.makeCommandBuffer(), let e = commands.makeComputeCommandEncoder() else {
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
        e.dispatchThreads(MTLSize(width: width, height: viewHeight, depth: 1), threadsPerThreadgroup: group)
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
