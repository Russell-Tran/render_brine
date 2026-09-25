// A GPU ray tracer with a uniform grid AND ambient occlusion — the two
// techniques of steps 8a and 9 in one kernel, which this step needs because it
// is both the biggest scene so far and the one whose subject is all crevices.
//
// Putting them together is not just a matter of pasting. Step 9's occlusion
// tested every probe ray against every sphere, which was fine for a 1,771-atom
// protein. Here the scene is ~407,000 spheres and each visible point fires 12
// probes, so brute-force probes would be about 5 billion sphere tests per
// *pixel*. The probe rays therefore walk the same grid the camera rays do, and
// stop at the first hit rather than looking for the nearest one — an occlusion
// probe only needs to know *whether* something is in the way.
//
// Keeping occlusion from crawling
// -------------------------------
// Step 9 could guarantee stillness: the molecule never moved and the camera
// orbited it. Here the plasmid drifts toward the membrane, so something does
// move. The guarantee is kept a different way: the probe directions are built
// from the surface normal alone, in world space, with no dependence on the
// pixel, the frame number or the point's position. A rigid body that
// TRANSLATES therefore carries its own shading with it unchanged — p and its
// neighbours move together and n does not change at all. So the plasmid
// translates and never tumbles, and the camera never moves. Occlusion between
// the plasmid and the membrane does change as the gap closes, which is the one
// thing here that should change.
//
// Amanatides & Woo, "A Fast Voxel Traversal Algorithm for Ray Tracing",
// Eurographics (1987), for the grid walk.
// Tarini, Cignoni & Montani, IEEE TVCG 12:1237 (2006), for molecular occlusion.

import Foundation
import Metal
import simd

func radians(_ degrees: Float) -> Float { degrees * .pi / 180 }

/// A sphere. This step has no cylinders: everything — lipid beads, glycan,
/// DNA, ions — is a sphere, which keeps the grid and the probe loop simple.
struct GPUShape {
    var a: SIMD4<Float>           // xyz = center, w = radius (0 means skip)
    var color: SIMD4<Float>       // rgb, w unused

    static func sphere(center: SIMD3<Float>, radius: Float, color: SIMD3<Float>) -> GPUShape {
        GPUShape(a: SIMD4(center, radius), color: SIMD4(color, 1))
    }

    var center: SIMD3<Float> { SIMD3(a.x, a.y, a.z) }
    var radius: Float {
        get { a.w }
        set { a.w = newValue }
    }

    func bounds() -> (lo: SIMD3<Float>, hi: SIMD3<Float>) {
        (center - a.w, center + a.w)
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
    var aoProbes: UInt32          // 0 turns occlusion off
    var aoDistance: Float         // how far a probe looks, in ångströms
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
        let up = abs(f.y) > 0.999 ? SIMD3<Float>(0, 0, 1) : SIMD3<Float>(0, 1, 0)
        return simd_normalize(simd_cross(f, up))
    }
    var up: SIMD3<Float> { simd_cross(right, forward) }

    init(origin: SIMD3<Float>, target: SIMD3<Float>, fov: Float) {
        self.origin = origin
        self.target = target
        self.tanHalfFOV = tan(radians(fov / 2))
    }

    /// Where a point lands in a width × height image (pixels, from the top left).
    func project(_ p: SIMD3<Float>, width: Int, height: Int) -> SIMD2<Float> {
        let v = p - origin
        let z: Float = simd_dot(v, forward)
        let aspect: Float = Float(width) / Float(height)
        let ndcX: Float = simd_dot(v, right) / (z * tanHalfFOV * aspect)
        let ndcY: Float = simd_dot(v, up) / (z * tanHalfFOV)
        return SIMD2((ndcX + 1) / 2 * Float(width), (1 - ndcY) / 2 * Float(height))
    }
}

let keyLightWorld = simd_normalize(SIMD3<Float>(-0.45, 0.66, 0.70))
let fillLightWorld = simd_normalize(SIMD3<Float>(0.78, -0.20, 0.58))

let raytraceKernelSource = """
    #include <metal_stdlib>
    using namespace metal;

    struct Shape { float4 a; float4 color; };
    struct Camera {
        float4 origin; float4 forward; float4 right; float4 up; float4 key; float4 fill;
        float tanHalfFOV; uint width; uint height; uint shapeCount; uint samplesPerSide; uint useGrid;
        uint aoProbes; float aoDistance; float aoStrength; float aoContrast;
    };
    struct Grid { float4 origin; float4 cell; uint4 dims; };

    constant float3 SKY_BLUE = float3(140.0, 200.0, 235.0) / 255.0;
    constant float3 DEEP_BLUE = float3(8.0, 40.0, 90.0) / 255.0;

    constant float BAYER[16] = { 0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5 };
    float3 dither(uint2 gid) {
        float n = (BAYER[(gid.y & 3u) * 4u + (gid.x & 3u)] + 0.5) / 16.0 - 0.5;
        return float3(n * 3.0 / 255.0);
    }

    inline bool hitSphere(float3 ro, float3 rd, device const Shape &s, float best,
                          thread float &t, thread float3 &n) {
        float r = s.a.w;
        if (r <= 0.0) return false;
        float3 oc = ro - s.a.xyz;
        float b = dot(oc, rd);
        float h = b * b - (dot(oc, oc) - r * r);
        if (h < 0.0) return false;
        float tt = -b - sqrt(h);
        if (tt <= 1e-3 || tt >= best) return false;
        t = tt;
        n = (ro + tt * rd - s.a.xyz) / r;
        return true;
    }

    // Every shape in order: the reference the grid is checked against.
    float traceAll(float3 ro, float3 rd, device const Shape *shapes, constant Camera &cam,
                   thread float3 &normal, thread float3 &color) {
        float best = 1e30;
        for (uint i = 0; i < cam.shapeCount; i++) {
            float t; float3 n;
            if (hitSphere(ro, rd, shapes[i], best, t, n)) {
                best = t; normal = n; color = shapes[i].color.rgb;
            }
        }
        return best;
    }

    // Only the shapes in the boxes the ray crosses. `anyHit` stops at the first
    // thing in the way rather than hunting for the nearest, which is all an
    // occlusion probe needs and is far cheaper.
    float traceGrid(float3 ro, float3 rd, device const Shape *shapes,
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
                stp[k] = 0;
                tMax[k] = 1e30;
                tDelta[k] = 1e30;
            }
        }

        float best = 1e30;
        for (int guard = 0; guard < 8192; guard++) {
            uint index = (uint(c.z) * grid.dims.y + uint(c.y)) * grid.dims.x + uint(c.x);
            uint from = cellStart[index], to = cellStart[index + 1];
            for (uint i = from; i < to; i++) {
                uint s = cellItems[i];
                float t; float3 n;
                if (hitSphere(ro, rd, shapes[s], best, t, n)) {
                    if (t > maxT) continue;
                    best = t; normal = n; color = shapes[s].color.rgb;
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

    // A cosine-weighted direction in the hemisphere about `n`. It depends on
    // nothing but `i` and `n` — not the pixel, not the frame, not the position —
    // so a surface that translates keeps exactly the shading it had.
    float3 hemisphere(uint i, uint total, float3 n) {
        float u = (float(i) + 0.5) / float(total);
        float r = sqrt(u);
        float phi = 6.2831853 * fract(float(i) * 0.6180339887);
        float3 t = normalize(abs(n.z) < 0.9 ? cross(n, float3(0.0, 0.0, 1.0)) : cross(n, float3(1.0, 0.0, 0.0)));
        float3 b = cross(n, t);
        return t * (r * cos(phi)) + b * (r * sin(phi)) + n * sqrt(max(0.0, 1.0 - u));
    }

    float ambientOcclusion(float3 p, float3 n, device const Shape *shapes,
                           device const uint *cellStart, device const uint *cellItems,
                           constant Grid &grid, constant Camera &cam) {
        if (cam.aoProbes == 0) return 1.0;
        float3 start = p + n * 0.05;
        uint open = 0;
        for (uint i = 0; i < cam.aoProbes; i++) {
            float3 dir = hemisphere(i, cam.aoProbes, n);
            float3 nn, cc;
            float t = cam.useGrid != 0
                ? traceGrid(start, dir, shapes, cellStart, cellItems, grid, cam.aoDistance, true, nn, cc)
                : 1e30;
            if (cam.useGrid == 0) {
                // The brute-force path, kept so the grid can be checked against it.
                t = 1e30;
                for (uint k = 0; k < cam.shapeCount; k++) {
                    float tt; float3 n2;
                    if (hitSphere(start, dir, shapes[k], cam.aoDistance, tt, n2)) { t = tt; break; }
                }
            }
            if (t > cam.aoDistance) open++;
        }
        float ao = float(open) / float(cam.aoProbes);
        return pow(ao, cam.aoContrast);
    }

    float3 shadeRay(float3 ro, float3 rd, float backgroundY, device const Shape *shapes,
                    device const uint *cellStart, device const uint *cellItems,
                    constant Grid &grid, constant Camera &cam, thread bool &hit) {
        float3 n = float3(0, 1, 0), base = float3(1);
        float t = cam.useGrid != 0
            ? traceGrid(ro, rd, shapes, cellStart, cellItems, grid, 1e30, false, n, base)
            : traceAll(ro, rd, shapes, cam, n, base);
        hit = t < 1e29;
        if (!hit) return mix(SKY_BLUE, DEEP_BLUE, backgroundY);

        float3 p = ro + t * rd;
        float ao = ambientOcclusion(p, n, shapes, cellStart, cellItems, grid, cam);
        ao = mix(1.0, ao, cam.aoStrength);

        float key = max(dot(n, cam.key.xyz), 0.0);
        float fill = max(dot(n, cam.fill.xyz), 0.0);
        float3 h = normalize(cam.key.xyz - rd);
        float spec = pow(max(dot(n, h), 0.0), 24.0);
        // Occlusion drives the ambient term and damps the lights, which is what
        // turns crevices into depth rather than dark paint. Specular is kept
        // very low: a highlight on each of 400,000 beads is a field of glitter.
        float3 color = base * (0.13 + 0.64 * ao + 0.60 * key * mix(0.30, 1.0, ao) + 0.15 * fill * ao);
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
        if lo.x > hi.x { lo = .zero; hi = SIMD3(repeating: 1) }
        let pad = simd_max((hi - lo) * 0.001, SIMD3(repeating: 0.01))
        lo -= pad
        hi += pad
        let extent = hi - lo
        let longest = max(extent.x, max(extent.y, extent.z))
        let perUnit = density * 3 * Float(pow(Double(max(shapes.count, 1)), 1.0 / 3.0)) / longest
        var dims = SIMD3<Int>(1, 1, 1)
        for k in 0..<3 {
            dims[k] = max(1, min(1024, Int((extent[k] * perUnit).rounded())))
        }
        while dims.x * dims.y * dims.z > maxCells {
            let k = dims.x >= dims.y && dims.x >= dims.z ? 0 : (dims.y >= dims.z ? 1 : 2)
            dims[k] = max(1, dims[k] / 2)
        }
        self.origin = lo
        self.dims = dims
        self.cellSize = SIMD3(extent.x / Float(dims.x), extent.y / Float(dims.y), extent.z / Float(dims.z))

        let cells = dims.x * dims.y * dims.z
        var counts = [UInt32](repeating: 0, count: cells)
        let gridOrigin = lo, cellSize = self.cellSize
        func span(_ s: GPUShape) -> (SIMD3<Int>, SIMD3<Int>) {
            let b = s.bounds()
            var a = SIMD3<Int>(0, 0, 0), z = SIMD3<Int>(0, 0, 0)
            for k in 0..<3 {
                a[k] = max(0, min(dims[k] - 1, Int(((b.lo[k] - gridOrigin[k]) / cellSize[k]).rounded(.down))))
                z[k] = max(0, min(dims[k] - 1, Int(((b.hi[k] - gridOrigin[k]) / cellSize[k]).rounded(.down))))
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
            let v = Int(((p[k] - origin[k]) / cellSize[k]).rounded(.down))
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
    var distance: Float = 14      // ångströms; about three bead diameters
    var strength: Float = 1
    var contrast: Float = 1.5

    static let off = AOSettings(probes: 0, distance: 0, strength: 0, contrast: 1)
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
            // The laptop's older toolchain wants this off, as in every step
            // since 3: fast math would let the compiler reassociate the
            // intersection arithmetic and the results drift between machines.
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

    /// Sorts the scene into a grid. Positions must not change afterwards;
    /// radii may shrink freely, because the boxes were measured at full size.
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
                useGrid: Bool = true, ao: AOSettings = AOSettings()) throws -> Double {
        var cam = GPUCamera(origin: SIMD4(camera.origin, 1), forward: SIMD4(camera.forward, 0),
                            right: SIMD4(camera.right, 0), up: SIMD4(camera.up, 0),
                            key: SIMD4(keyLightWorld, 0), fill: SIMD4(fillLightWorld, 0),
                            tanHalfFOV: camera.tanHalfFOV, width: UInt32(width), height: UInt32(viewHeight),
                            shapeCount: UInt32(shapes.count), samplesPerSide: UInt32(samplesPerSide),
                            useGrid: useGrid ? 1 : 0, aoProbes: UInt32(ao.probes),
                            aoDistance: ao.distance, aoStrength: ao.strength, aoContrast: ao.contrast)
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
