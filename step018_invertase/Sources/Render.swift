// A GPU ray tracer with a uniform grid, ambient occlusion, and BOTH spheres and
// cylinders in one primitive array.
//
// Step 9 had spheres and cylinders but tested every ray against every one of
// them, which is fine for a 1,771-atom protein. Step 12 had the grid and the
// occlusion together but only spheres, because everything in that scene was a
// bead. This step needs all three at once: the octamer is 32,992 atoms drawn at
// van der Waals radii, and inside the porthole there is ball-and-stick, whose
// sticks are cylinders. So the grid sorts one array that holds both kinds and
// the probe rays walk it exactly as the camera rays do.
//
// One array rather than two is not just tidiness. A grid over spheres and a
// second grid over cylinders would mean two walks per ray and two sets of
// mailboxes; with one array a primitive is a sphere when b.w is zero and a
// capped-free cylinder when it is one, and the traversal never has to know.
//
// Amanatides & Woo, "A Fast Voxel Traversal Algorithm for Ray Tracing",
// Eurographics (1987), for the grid walk.
// Tarini, Cignoni & Montani, IEEE TVCG 12:1237 (2006), for molecular occlusion.

import Foundation
import Metal
import simd

func radians(_ degrees: Float) -> Float { degrees * .pi / 180 }

/// One primitive. `b.w == 0` means a sphere centred at `a.xyz` of radius `a.w`;
/// `b.w == 1` means a cylinder from `a.xyz` to `b.xyz` of radius `a.w`.
struct GPUPrim {
    var a: SIMD4<Float>
    var b: SIMD4<Float>
    var color: SIMD4<Float>

    static func sphere(_ centre: SIMD3<Float>, _ radius: Float, _ colour: SIMD3<Float>) -> GPUPrim {
        GPUPrim(a: SIMD4(centre, radius), b: SIMD4(0, 0, 0, 0), color: SIMD4(colour, 1))
    }

    static func stick(_ from: SIMD3<Float>, _ to: SIMD3<Float>, _ radius: Float,
                      _ colour: SIMD3<Float>) -> GPUPrim {
        GPUPrim(a: SIMD4(from, radius), b: SIMD4(to, 1), color: SIMD4(colour, 1))
    }

    var isCylinder: Bool { b.w > 0.5 }
    var radius: Float { a.w }

    func bounds() -> (lo: SIMD3<Float>, hi: SIMD3<Float>) {
        let p = SIMD3(a.x, a.y, a.z)
        let r = SIMD3<Float>(repeating: a.w)
        if !isCylinder { return (p - r, p + r) }
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
    var background: SIMD4<Float>     // rgb of the top of the gradient, w = how dark the bottom goes
    var tanHalfFOV: Float
    var width: UInt32
    var height: UInt32
    var primCount: UInt32
    var samplesPerSide: UInt32
    var useGrid: UInt32
    var aoProbes: UInt32             // 0 turns occlusion off
    var aoDistance: Float
    var aoStrength: Float
    var aoContrast: Float
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
        let worldUp: SIMD3<Float> = abs(f.y) > 0.999 ? SIMD3(0, 0, 1) : SIMD3(0, 1, 0)
        return simd_normalize(simd_cross(f, worldUp))
    }
    var up: SIMD3<Float> { simd_cross(right, forward) }

    init(origin: SIMD3<Float>, target: SIMD3<Float>, fov: Float) {
        self.origin = origin
        self.target = target
        self.tanHalfFOV = tan(radians(fov / 2))
    }

    /// A camera `distance` away from `target`, turned `yaw` degrees about it and
    /// raised `pitch` degrees.
    static func orbit(target: SIMD3<Float>, distance: Float, yaw: Float, pitch: Float,
                      fov: Float) -> Camera {
        let y: Float = radians(yaw), p: Float = radians(pitch)
        let dx: Float = sin(y) * cos(p)
        let dy: Float = sin(p)
        let dz: Float = cos(y) * cos(p)
        let offset = SIMD3<Float>(dx, dy, dz) * distance
        return Camera(origin: target + offset, target: target, fov: fov)
    }

    /// Where a point lands in a width × height image, in pixels from the top left.
    func project(_ p: SIMD3<Float>, width: Int, height: Int) -> SIMD2<Float> {
        let v = p - origin
        let z: Float = simd_dot(v, forward)
        let aspect: Float = Float(width) / Float(height)
        let ndcX: Float = simd_dot(v, right) / (z * tanHalfFOV * aspect)
        let ndcY: Float = simd_dot(v, up) / (z * tanHalfFOV)
        let px: Float = (ndcX + 1) / 2 * Float(width)
        let py: Float = (1 - ndcY) / 2 * Float(height)
        return SIMD2(px, py)
    }

    /// The direction of the ray through the centre of pixel (x, y).
    func rayDirection(x: Int, y: Int, width: Int, height: Int) -> SIMD3<Float> {
        let aspect: Float = Float(width) / Float(height)
        let sx: Float = Float(x) + 0.5, sy: Float = Float(y) + 0.5
        let ndcX: Float = sx / Float(width) * 2 - 1
        let ndcY: Float = 1 - sy / Float(height) * 2
        let dx: Float = ndcX * aspect * tanHalfFOV
        let dy: Float = ndcY * tanHalfFOV
        return simd_normalize(forward + right * dx + up * dy)
    }
}

let keyLightWorld = simd_normalize(SIMD3<Float>(-0.45, 0.66, 0.70))
let fillLightWorld = simd_normalize(SIMD3<Float>(0.78, -0.20, 0.58))

let raytraceKernelSource = """
    #include <metal_stdlib>
    using namespace metal;

    struct Prim { float4 a; float4 b; float4 color; };
    struct Camera {
        float4 origin; float4 forward; float4 right; float4 up; float4 key; float4 fill;
        float4 background;
        float tanHalfFOV; uint width; uint height; uint primCount; uint samplesPerSide; uint useGrid;
        uint aoProbes; float aoDistance; float aoStrength; float aoContrast;
    };
    struct Grid { float4 origin; float4 cell; uint4 dims; };

    constant float BAYER[16] = { 0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5 };
    float3 dither(uint2 gid) {
        float n = (BAYER[(gid.y & 3u) * 4u + (gid.x & 3u)] + 0.5) / 16.0 - 0.5;
        return float3(n * 3.0 / 255.0);
    }

    // Nearest hit on one primitive, sphere or cylinder, nearer than `best`.
    inline bool hitPrim(float3 ro, float3 rd, device const Prim &s, float best,
                        thread float &t, thread float3 &n) {
        float r = s.a.w;
        if (r <= 0.0) return false;
        if (s.b.w < 0.5) {
            float3 oc = ro - s.a.xyz;
            float bb = dot(oc, rd);
            float h = bb * bb - (dot(oc, oc) - r * r);
            if (h < 0.0) return false;
            float tt = -bb - sqrt(h);
            if (tt <= 1e-3 || tt >= best) return false;
            t = tt;
            n = (ro + tt * rd - s.a.xyz) / r;
            return true;
        }
        // An open cylinder (no end caps: bonds always end inside an atom).
        float3 pa = s.a.xyz;
        float3 ba = s.b.xyz - pa;
        float3 oc = ro - pa;
        float baba = dot(ba, ba);
        float bard = dot(ba, rd);
        float baoc = dot(ba, oc);
        float k2 = baba - bard * bard;
        if (k2 < 1e-8) return false;
        float k1 = baba * dot(oc, rd) - baoc * bard;
        float k0 = baba * dot(oc, oc) - baoc * baoc - r * r * baba;
        float h = k1 * k1 - k2 * k0;
        if (h < 0.0) return false;
        float tt = (-k1 - sqrt(h)) / k2;
        if (tt <= 1e-3 || tt >= best) return false;
        float y = baoc + tt * bard;
        if (y < 0.0 || y > baba) return false;
        t = tt;
        n = (oc + tt * rd - ba * y / baba) / r;
        return true;
    }

    float traceAll(float3 ro, float3 rd, device const Prim *prims, constant Camera &cam,
                   float maxT, bool anyHit, thread float3 &normal, thread float3 &color) {
        float best = maxT;
        for (uint i = 0; i < cam.primCount; i++) {
            float t; float3 n;
            if (hitPrim(ro, rd, prims[i], best, t, n)) {
                best = t; normal = n; color = prims[i].color.rgb;
                if (anyHit) return best;
            }
        }
        return best;
    }

    // Only the primitives in the boxes the ray crosses. `anyHit` stops at the
    // first thing in the way, which is all an occlusion probe needs.
    float traceGrid(float3 ro, float3 rd, device const Prim *prims,
                    device const uint *cellStart, device const uint *cellItems,
                    constant Grid &grid, float maxT, bool anyHit,
                    thread float3 &normal, thread float3 &color) {
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
                float t; float3 n;
                if (hitPrim(ro, rd, prims[s], best, t, n)) {
                    if (t > maxT) continue;
                    best = t; normal = n; color = prims[s].color.rgb;
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

    // A cosine-weighted direction in the hemisphere about `n`, depending on
    // nothing but `i` and `n` — not the pixel, not the frame, not the position.
    float3 hemisphere(uint i, uint total, float3 n) {
        float u = (float(i) + 0.5) / float(total);
        float r = sqrt(u);
        float phi = 6.2831853 * fract(float(i) * 0.6180339887);
        float3 t = normalize(abs(n.z) < 0.9 ? cross(n, float3(0.0, 0.0, 1.0))
                                            : cross(n, float3(1.0, 0.0, 0.0)));
        float3 b = cross(n, t);
        return t * (r * cos(phi)) + b * (r * sin(phi)) + n * sqrt(max(0.0, 1.0 - u));
    }

    float ambientOcclusion(float3 p, float3 n, device const Prim *prims,
                           device const uint *cellStart, device const uint *cellItems,
                           constant Grid &grid, constant Camera &cam) {
        if (cam.aoProbes == 0) return 1.0;
        float3 start = p + n * 0.05;
        uint open = 0;
        for (uint i = 0; i < cam.aoProbes; i++) {
            float3 dir = hemisphere(i, cam.aoProbes, n);
            float3 nn, cc;
            float t = cam.useGrid != 0
                ? traceGrid(start, dir, prims, cellStart, cellItems, grid, cam.aoDistance, true, nn, cc)
                : traceAll(start, dir, prims, cam, cam.aoDistance, true, nn, cc);
            if (t > cam.aoDistance) open++;
        }
        float ao = float(open) / float(cam.aoProbes);
        return pow(ao, cam.aoContrast);
    }

    float3 shadeRay(float3 ro, float3 rd, float backgroundY, device const Prim *prims,
                    device const uint *cellStart, device const uint *cellItems,
                    constant Grid &grid, constant Camera &cam, thread bool &hit) {
        float3 n = float3(0, 1, 0), base = float3(1);
        float t = cam.useGrid != 0
            ? traceGrid(ro, rd, prims, cellStart, cellItems, grid, 1e30, false, n, base)
            : traceAll(ro, rd, prims, cam, 1e30, false, n, base);
        hit = t < 1e29;
        if (!hit) return mix(cam.background.rgb, cam.background.rgb * cam.background.w, backgroundY);

        float3 p = ro + t * rd;
        float ao = ambientOcclusion(p, n, prims, cellStart, cellItems, grid, cam);
        ao = mix(1.0, ao, cam.aoStrength);

        float key = max(dot(n, cam.key.xyz), 0.0);
        float fill = max(dot(n, cam.fill.xyz), 0.0);
        float3 h = normalize(cam.key.xyz - rd);
        float spec = pow(max(dot(n, h), 0.0), 24.0);
        // Occlusion drives the ambient term and damps the lights, which is what
        // turns crevices into depth rather than dark paint.
        float3 color = base * (0.13 + 0.64 * ao + 0.60 * key * mix(0.30, 1.0, ao) + 0.15 * fill * ao);
        color += 0.05 * spec * ao;
        return color;
    }

    kernel void render(device uchar4 *pixels [[buffer(0)]],
                       constant Camera &cam [[buffer(1)]],
                       device const Prim *prims [[buffer(2)]],
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
                sum += shadeRay(cam.origin.xyz, rd, sy / float(cam.height), prims,
                                cellStart, cellItems, grid, cam, hit);
                misses += hit ? 0.0 : 1.0;
            }
        }
        // Dither only the background: it hides banding in the gradient, and on
        // the subject it would be churn the GIF has to store.
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

/// The GPU to use. (Needs CoreGraphics linked; the Makefile does that.)
func findDevice() throws -> MTLDevice {
    if let device = MTLCreateSystemDefaultDevice() ?? MTLCopyAllDevices().first { return device }
    throw RenderError.noMetalDevice
}

// MARK: - The uniform grid

/// Which primitives lie in each box of a regular grid over the scene, flattened
/// the way the GPU wants it: `start[c]..<start[c+1]` indexes into `items`.
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

    /// Sorts `prims` into boxes. `density` scales how many boxes per primitive;
    /// the usual rule of thumb is one box per primitive.
    init(prims: [GPUPrim], density: Float = 1) {
        var lo = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
        var hi = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)
        for p in prims where p.radius > 0 {
            let b = p.bounds()
            lo = simd_min(lo, b.lo)
            hi = simd_max(hi, b.hi)
        }
        if lo.x > hi.x { lo = .zero; hi = SIMD3(repeating: 1) }
        lo -= SIMD3(repeating: 0.5)
        hi += SIMD3(repeating: 0.5)
        let extent: SIMD3<Float> = hi - lo
        let volume: Float = max(extent.x * extent.y * extent.z, 1e-6)
        let wanted: Float = max(Float(prims.count) * density, 1)
        let side: Float = pow(volume / wanted, 1.0 / 3.0)
        var d = SIMD3<Int>(0, 0, 0)
        for k in 0..<3 {
            let n: Int = Int((extent[k] / max(side, 1e-4)).rounded(.up))
            d[k] = max(1, min(n, 256))
        }
        origin = lo
        dims = d
        cellSize = SIMD3(extent.x / Float(d.x), extent.y / Float(d.y), extent.z / Float(d.z))

        let cells: Int = d.x * d.y * d.z
        var counts = [UInt32](repeating: 0, count: cells + 1)
        let gridOrigin = origin, cell = cellSize
        func span(_ p: GPUPrim) -> (SIMD3<Int>, SIMD3<Int>) {
            let b = p.bounds()
            var a = SIMD3<Int>(0, 0, 0), z = SIMD3<Int>(0, 0, 0)
            for k in 0..<3 {
                let f0: Float = (b.lo[k] - gridOrigin[k]) / cell[k]
                let f1: Float = (b.hi[k] - gridOrigin[k]) / cell[k]
                a[k] = max(0, min(d[k] - 1, Int(f0.rounded(.down))))
                z[k] = max(0, min(d[k] - 1, Int(f1.rounded(.down))))
            }
            return (a, z)
        }
        for p in prims where p.radius > 0 {
            let (a, z) = span(p)
            for cz in a.z...z.z {
                for cy in a.y...z.y {
                    for cx in a.x...z.x {
                        counts[(cz * d.y + cy) * d.x + cx] += 1
                    }
                }
            }
        }
        var offsets = [UInt32](repeating: 0, count: cells + 1)
        var running: UInt32 = 0
        for c in 0..<cells {
            offsets[c] = running
            running += counts[c]
        }
        offsets[cells] = running
        start = offsets
        var cursor = offsets
        items = [UInt32](repeating: 0, count: Int(running))
        for (i, p) in prims.enumerated() where p.radius > 0 {
            let (a, z) = span(p)
            for cz in a.z...z.z {
                for cy in a.y...z.y {
                    for cx in a.x...z.x {
                        let c = (cz * d.y + cy) * d.x + cx
                        items[Int(cursor[c])] = UInt32(i)
                        cursor[c] += 1
                    }
                }
            }
        }
    }
}

/// How occlusion is sampled.
struct AOSettings {
    var probes: Int = 12
    var distance: Float = 8.0
    var strength: Float = 1.0
    var contrast: Float = 1.7
    static let off = AOSettings(probes: 0, distance: 0, strength: 0, contrast: 1)
}

/// The two panels are lit and backed differently: the molecule sits against the
/// project's blue, the polarimeter against a dark bench.
struct Backdrop {
    var top: SIMD3<Float>
    var bottomScale: Float
    static let sky = Backdrop(top: SIMD3(140.0 / 255, 200.0 / 255, 235.0 / 255), bottomScale: 0.22)
    static let bench = Backdrop(top: SIMD3(0.10, 0.12, 0.16), bottomScale: 0.45)
}

final class SceneRenderer {
    let device: MTLDevice
    private let pipeline: MTLComputePipelineState
    private let queue: MTLCommandQueue
    private var primBuffer: MTLBuffer?
    private var startBuffer: MTLBuffer?
    private var itemBuffer: MTLBuffer?
    private(set) var grid: UniformGrid?
    private(set) var lastGridSeconds: Double = 0

    init(device: MTLDevice) throws {
        self.device = device
        do {
            let options = MTLCompileOptions()
            // Precise math, as in steps 3 to 5. (Newer SDKs call this
            // mathMode = .safe; the older name builds on every SDK here.)
            options.fastMathEnabled = false
            let library = try device.makeLibrary(source: raytraceKernelSource, options: options)
            guard let function = library.makeFunction(name: "render") else {
                throw RenderError.kernelCompile("no kernel named render")
            }
            pipeline = try device.makeComputePipelineState(function: function)
        } catch let error as RenderError {
            throw error
        } catch {
            throw RenderError.kernelCompile("\(error)")
        }
        guard let queue = device.makeCommandQueue() else { throw RenderError.gpu("no command queue") }
        self.queue = queue
    }

    /// Sorts the scene into a grid. Positions must not change afterwards.
    func buildGrid(_ prims: [GPUPrim], density: Float = 1) throws {
        let start = Date()
        let g = UniformGrid(prims: prims, density: density)
        grid = g
        lastGridSeconds = Date().timeIntervalSince(start)
        startBuffer = try fill(startBuffer, with: g.start)
        itemBuffer = try fill(itemBuffer, with: g.items.isEmpty ? [UInt32(0)] : g.items)
    }

    /// Ray-traces into a `width` × `height` buffer. Returns GPU seconds.
    @discardableResult
    func render(prims: [GPUPrim], camera: Camera, into frame: MTLBuffer,
                width: Int, height: Int, samplesPerSide: Int = 2,
                ao: AOSettings = AOSettings(), backdrop: Backdrop = .sky,
                lightYaw: Float = 0, useGrid: Bool = true) throws -> Double {
        guard let g = grid, useGrid else {
            return try dispatch(prims: prims, camera: camera, into: frame, width: width,
                                height: height, samplesPerSide: samplesPerSide, ao: ao,
                                backdrop: backdrop, lightYaw: lightYaw, useGrid: false,
                                grid: GPUGrid(origin: .zero, cell: SIMD4(1, 1, 1, 0),
                                              dims: SIMD4(1, 1, 1, 0)))
        }
        let gpuGrid = GPUGrid(origin: SIMD4(g.origin, 0),
                              cell: SIMD4(g.cellSize, 0),
                              dims: SIMD4(UInt32(g.dims.x), UInt32(g.dims.y), UInt32(g.dims.z), 0))
        return try dispatch(prims: prims, camera: camera, into: frame, width: width, height: height,
                            samplesPerSide: samplesPerSide, ao: ao, backdrop: backdrop,
                            lightYaw: lightYaw, useGrid: true, grid: gpuGrid)
    }

    private func dispatch(prims: [GPUPrim], camera: Camera, into frame: MTLBuffer,
                          width: Int, height: Int, samplesPerSide: Int, ao: AOSettings,
                          backdrop: Backdrop, lightYaw: Float, useGrid: Bool,
                          grid gpuGrid: GPUGrid) throws -> Double {
        let keyDir = rotateAboutY(keyLightWorld, degrees: -lightYaw)
        let fillDir = rotateAboutY(fillLightWorld, degrees: -lightYaw)
        var cam = GPUCamera(origin: SIMD4(camera.origin, 1),
                            forward: SIMD4(camera.forward, 0),
                            right: SIMD4(camera.right, 0),
                            up: SIMD4(camera.up, 0),
                            key: SIMD4(keyDir, 0),
                            fill: SIMD4(fillDir, 0),
                            background: SIMD4(backdrop.top, backdrop.bottomScale),
                            tanHalfFOV: camera.tanHalfFOV,
                            width: UInt32(width), height: UInt32(height),
                            primCount: UInt32(prims.count),
                            samplesPerSide: UInt32(samplesPerSide),
                            useGrid: useGrid ? 1 : 0,
                            aoProbes: UInt32(ao.probes), aoDistance: ao.distance,
                            aoStrength: ao.strength, aoContrast: ao.contrast)
        var grid = gpuGrid
        guard let commands = queue.makeCommandBuffer(),
              let e = commands.makeComputeCommandEncoder() else {
            throw RenderError.gpu("could not create a command encoder")
        }
        // Metal won't bind an empty buffer, so always send at least one element.
        let p = prims.isEmpty ? [GPUPrim.sphere(.zero, 0, .zero)] : prims
        primBuffer = try fill(primBuffer, with: p)
        if startBuffer == nil { startBuffer = try fill(startBuffer, with: [UInt32(0), UInt32(0)]) }
        if itemBuffer == nil { itemBuffer = try fill(itemBuffer, with: [UInt32(0)]) }
        e.setComputePipelineState(pipeline)
        e.setBuffer(frame, offset: 0, index: 0)
        e.setBytes(&cam, length: MemoryLayout<GPUCamera>.stride, index: 1)
        e.setBuffer(primBuffer, offset: 0, index: 2)
        e.setBuffer(startBuffer, offset: 0, index: 3)
        e.setBuffer(itemBuffer, offset: 0, index: 4)
        e.setBytes(&grid, length: MemoryLayout<GPUGrid>.stride, index: 5)
        let w = pipeline.threadExecutionWidth
        let group = MTLSize(width: w, height: pipeline.maxTotalThreadsPerThreadgroup / w, depth: 1)
        e.dispatchThreads(MTLSize(width: width, height: height, depth: 1),
                          threadsPerThreadgroup: group)
        e.endEncoding()
        commands.commit()
        commands.waitUntilCompleted()
        if let error = commands.error { throw RenderError.gpu(error.localizedDescription) }
        return commands.gpuEndTime - commands.gpuStartTime
    }

    private func fill<T>(_ buffer: MTLBuffer?, with items: [T]) throws -> MTLBuffer {
        let bytes = items.count * MemoryLayout<T>.stride
        var target = buffer
        if target == nil || target!.length < bytes {
            target = device.makeBuffer(length: max(bytes, 16), options: .storageModeShared)
        }
        guard let out = target else { throw RenderError.gpu("could not allocate a scene buffer") }
        items.withUnsafeBytes { out.contents().copyMemory(from: $0.baseAddress!, byteCount: bytes) }
        return out
    }
}

/// Turns a vector `degrees` about the y axis.
func rotateAboutY(_ v: SIMD3<Float>, degrees: Float) -> SIMD3<Float> {
    let a: Float = radians(degrees)
    let c: Float = cos(a), s: Float = sin(a)
    let x: Float = v.x * c + v.z * s
    let z: Float = -v.x * s + v.z * c
    return SIMD3(x, v.y, z)
}

/// Turns `v` about the unit axis `axis` by `degrees`. Rodrigues' formula.
func rotate(_ v: SIMD3<Float>, about axis: SIMD3<Float>, degrees: Float) -> SIMD3<Float> {
    let a: Float = radians(degrees)
    let c: Float = cos(a), s: Float = sin(a)
    let dotted: Float = simd_dot(axis, v)
    let crossed: SIMD3<Float> = simd_cross(axis, v)
    let term1: SIMD3<Float> = v * c
    let term2: SIMD3<Float> = crossed * s
    let term3: SIMD3<Float> = axis * (dotted * (1 - c))
    return term1 + term2 + term3
}

// MARK: - The CPU reference intersector
//
// The same arithmetic as the kernel, in Swift, so the tests can ask what a ray
// actually hits without going near the GPU. The porthole test uses it: a cut
// that leaves a backface showing is a cut you can see through into the inside
// of the far wall, and that is a question about normals along a ray.

struct RayHit {
    var t: Float
    var normal: SIMD3<Float>
    var index: Int
}

func intersect(_ prims: [GPUPrim], origin: SIMD3<Float>, direction: SIMD3<Float>) -> RayHit? {
    var best: Float = .greatestFiniteMagnitude
    var hit: RayHit?
    for (i, s) in prims.enumerated() {
        let r: Float = s.a.w
        if r <= 0 { continue }
        if !s.isCylinder {
            let centre = SIMD3(s.a.x, s.a.y, s.a.z)
            let oc: SIMD3<Float> = origin - centre
            let b: Float = simd_dot(oc, direction)
            let h: Float = b * b - (simd_dot(oc, oc) - r * r)
            if h < 0 { continue }
            let t: Float = -b - sqrt(h)
            if t <= 1e-3 || t >= best { continue }
            best = t
            hit = RayHit(t: t, normal: (origin + direction * t - centre) / r, index: i)
        } else {
            let pa = SIMD3(s.a.x, s.a.y, s.a.z)
            let ba = SIMD3(s.b.x, s.b.y, s.b.z) - pa
            let oc: SIMD3<Float> = origin - pa
            let baba: Float = simd_dot(ba, ba)
            let bard: Float = simd_dot(ba, direction)
            let baoc: Float = simd_dot(ba, oc)
            let k2: Float = baba - bard * bard
            if k2 < 1e-8 { continue }
            let k1: Float = baba * simd_dot(oc, direction) - baoc * bard
            let k0: Float = baba * simd_dot(oc, oc) - baoc * baoc - r * r * baba
            let h: Float = k1 * k1 - k2 * k0
            if h < 0 { continue }
            let t: Float = (-k1 - sqrt(h)) / k2
            if t <= 1e-3 || t >= best { continue }
            let y: Float = baoc + t * bard
            if y < 0 || y > baba { continue }
            best = t
            let n: SIMD3<Float> = (oc + direction * t - ba * (y / baba)) / r
            hit = RayHit(t: t, normal: n, index: i)
        }
    }
    return hit
}
