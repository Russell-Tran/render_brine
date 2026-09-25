// A GPU ray tracer with a uniform grid, the first acceleration structure in
// the series.
//
// Step 8 tested every ray against every shape. That is fine for a molecule of
// 1,700 spheres and sticks, but the plasmid is ~40,000, and the cost of brute
// force is (pixels × samples × shapes) — about 40 billion tests a frame.
//
// So the scene is sorted into a grid of boxes once, each box holding the
// shapes that overlap it, and each ray walks only the boxes it actually
// crosses (a 3D DDA — the same idea as drawing a line on a pixel grid, in
// three dimensions). A ray that passes through empty space touches almost
// nothing. The brute-force path is kept, both as the thing to measure against
// and as the reference the grid is tested for agreement with.
//
// Shapes are one type now rather than two arrays, because the grid has to put
// spheres and cylinders in the same boxes and walk them together.

import Foundation
import Metal
import simd

/// A sphere (kind 0: `a` is center and radius) or a cylinder (kind 1: from `a`
/// to `b`, with `a.w` the radius). A radius of zero means "skip me", which is
/// how a shape fades out without disturbing the grid built around it.
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
    var radius: Float {
        get { a.w }
        set { a.w = newValue }
    }

    /// The box around this shape, grown by `pad`.
    func bounds(pad: Float = 0) -> (lo: SIMD3<Float>, hi: SIMD3<Float>) {
        let r = a.w + pad
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
    var key: SIMD4<Float>         // the two light directions, so the scene can
    var fill: SIMD4<Float>        // stay still while camera and lights turn
    var tanHalfFOV: Float
    var width: UInt32
    var height: UInt32
    var shapeCount: UInt32
    var samplesPerSide: UInt32
    var useGrid: UInt32
}

struct GPUGrid {
    var origin: SIMD4<Float>      // the low corner of the whole grid
    var cell: SIMD4<Float>        // the size of one box, per axis
    var dims: SIMD4<UInt32>       // how many boxes along each axis
}

struct Camera {
    var origin: SIMD3<Float>
    var target: SIMD3<Float>
    var tanHalfFOV: Float
    /// Turned this far (radians) about y, so a still scene can be lit as if it
    /// were the scene that turned.
    var lightSpin: Float = 0

    var forward: SIMD3<Float> { simd_normalize(target - origin) }
    var right: SIMD3<Float> {
        let f = forward
        let up = abs(f.y) > 0.999 ? SIMD3<Float>(0, 0, 1) : SIMD3<Float>(0, 1, 0)
        return simd_normalize(simd_cross(f, up))
    }
    var up: SIMD3<Float> { simd_cross(right, forward) }

    /// Looking at `target` from `distance` away, along `direction`.
    init(target: SIMD3<Float>, direction: SIMD3<Float>, distance: Float, fov: Float, lightSpin: Float = 0) {
        self.target = target
        self.origin = target + simd_normalize(direction) * distance
        self.tanHalfFOV = tan(radians(fov / 2))
        self.lightSpin = lightSpin
    }

    /// A camera `distance` away, turned `yaw` degrees about the target and
    /// raised `pitch` degrees above it.
    static func orbit(target: SIMD3<Float>, distance: Float, yaw: Float, pitch: Float,
                      fov: Float, lightSpin: Float = 0) -> Camera {
        let y = radians(yaw), p = radians(pitch)
        let direction = SIMD3<Float>(sin(y) * cos(p), sin(p), cos(y) * cos(p))
        return Camera(target: target, direction: direction, distance: distance, fov: fov, lightSpin: lightSpin)
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

    /// Is this point inside the view, allowing `margin` extra at each edge?
    func sees(_ p: SIMD3<Float>, width: Int, height: Int, margin: Float) -> Bool {
        let v = p - origin
        let z: Float = simd_dot(v, forward)
        if z <= 0 { return false }
        let s = project(p, width: width, height: height)
        return s.x > -margin && s.x < Float(width) + margin && s.y > -margin && s.y < Float(height) + margin
    }
}

let bondRadius: Float = 0.085

let raytraceKernelSource = """
    #include <metal_stdlib>
    using namespace metal;

    struct Shape { float4 a; float4 b; float4 color; };
    struct Camera {
        float4 origin; float4 forward; float4 right; float4 up; float4 key; float4 fill;
        float tanHalfFOV; uint width; uint height; uint shapeCount; uint samplesPerSide; uint useGrid;
    };
    struct Grid { float4 origin; float4 cell; uint4 dims; };

    // Step 2's gradient colors, (140, 200, 235) and (8, 40, 90) out of 255.
    constant float3 SKY_BLUE = float3(140.0, 200.0, 235.0) / 255.0;
    constant float3 DEEP_BLUE = float3(8.0, 40.0, 90.0) / 255.0;

    // A faint fixed ordered dither (4 × 4 Bayer, ±1.5/255), as in step 8: it
    // hides the banding a 256-color GIF would otherwise show in the gradient,
    // and a repeating pattern compresses far better than noise.
    constant float BAYER[16] = { 0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5 };
    float3 dither(uint2 gid) {
        float n = (BAYER[(gid.y & 3u) * 4u + (gid.x & 3u)] + 0.5) / 16.0 - 0.5;
        return float3(n * 3.0 / 255.0);
    }

    // One shape. Returns true and fills t and the normal if this ray hits it
    // nearer than `best`. A radius of zero means the shape is switched off.
    inline bool hitShape(float3 ro, float3 rd, device const Shape &s, float best,
                         thread float &t, thread float3 &n) {
        float r = s.a.w;
        if (r <= 0.0) return false;
        if (s.b.w < 0.5) {
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
        float tt = (-k1 - sqrt(h)) / k2;
        float y = baoc + tt * bard;
        if (tt <= 1e-3 || tt >= best || y < 0.0 || y > baba) return false;
        t = tt;
        n = (oc + tt * rd - ba * y / baba) / r;
        return true;
    }

    // Every shape, in order. The reference the grid is checked against.
    float traceAll(float3 ro, float3 rd, device const Shape *shapes, constant Camera &cam,
                   thread float3 &normal, thread float3 &color) {
        float best = 1e30;
        for (uint i = 0; i < cam.shapeCount; i++) {
            float t; float3 n;
            if (hitShape(ro, rd, shapes[i], best, t, n)) {
                best = t; normal = n; color = shapes[i].color.rgb;
            }
        }
        return best;
    }

    // Only the shapes in the boxes the ray crosses (Amanatides & Woo, "A Fast
    // Voxel Traversal Algorithm for Ray Tracing", Eurographics 1987).
    float traceGrid(float3 ro, float3 rd, device const Shape *shapes,
                    device const uint *cellStart, device const uint *cellItems,
                    constant Grid &grid, thread float3 &normal, thread float3 &color) {
        float3 lo = grid.origin.xyz;
        float3 cell = grid.cell.xyz;
        int3 dims = int3(grid.dims.xyz);
        float3 hi = lo + float3(dims) * cell;
        float3 inv = 1.0 / rd;              // an infinity here is harmless below

        // Where the ray is inside the grid's own box at all.
        float3 ta = (lo - ro) * inv, tb = (hi - ro) * inv;
        float3 tsmall = min(ta, tb), tbig = max(ta, tb);
        float tEnter = max(max(tsmall.x, tsmall.y), max(tsmall.z, 0.0));
        float tExit = min(min(tbig.x, tbig.y), tbig.z);
        if (tEnter > tExit) return 1e30;

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
        for (int guard = 0; guard < 4096; guard++) {
            uint index = (uint(c.z) * grid.dims.y + uint(c.y)) * grid.dims.x + uint(c.x);
            uint from = cellStart[index], to = cellStart[index + 1];
            for (uint i = from; i < to; i++) {
                uint s = cellItems[i];
                float t; float3 n;
                if (hitShape(ro, rd, shapes[s], best, t, n)) {
                    best = t; normal = n; color = shapes[s].color.rgb;
                }
            }
            // A hit inside this box is final: nothing in a later box can be nearer.
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

    float3 shadeRay(float3 ro, float3 rd, float backgroundY, device const Shape *shapes,
                    device const uint *cellStart, device const uint *cellItems,
                    constant Grid &grid, constant Camera &cam, thread bool &hit) {
        float3 n = float3(0, 1, 0), base = float3(1);
        float t = cam.useGrid != 0
            ? traceGrid(ro, rd, shapes, cellStart, cellItems, grid, n, base)
            : traceAll(ro, rd, shapes, cam, n, base);
        hit = t < 1e29;
        if (!hit) return mix(SKY_BLUE, DEEP_BLUE, backgroundY);
        float key = max(dot(n, cam.key.xyz), 0.0);
        float fill = max(dot(n, cam.fill.xyz), 0.0);
        float3 h = normalize(cam.key.xyz - rd);
        float spec = pow(max(dot(n, h), 0.0), 48.0);
        float rim = pow(1.0 - max(dot(n, -rd), 0.0), 3.0);
        // Much less white than steps 6 to 8 used. A cylinder always presents
        // some part of its curved side at the mirror angle, so a highlight
        // runs down every stick; on a tube only three pixels wide that covered
        // the whole thing and turned all four gene colors white.
        return base * (0.26 + 0.86 * key + 0.26 * fill) + 0.16 * spec + 0.05 * rim;
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
        // Dither only where the background shows through: it is there to hide
        // banding in the gradient, and on the molecule it would only add churn
        // for the GIF to store.
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

/// Which shapes lie in each box of a regular grid over the scene, as the flat
/// pair of arrays the GPU wants: `start[c]..<start[c+1]` indexes into `items`.
struct UniformGrid {
    var origin: SIMD3<Float>
    var cellSize: SIMD3<Float>
    var dims: SIMD3<Int>
    var start: [UInt32]
    var items: [UInt32]

    var cellCount: Int { dims.x * dims.y * dims.z }
    /// How many shapes an average non-empty box holds — the number that decides
    /// whether the grid is worth anything.
    var averageOccupancy: Double {
        var used = 0
        for c in 0..<cellCount where start[c + 1] > start[c] { used += 1 }
        return used == 0 ? 0 : Double(items.count) / Double(used)
    }

    /// `density` scales the resolution: 1 aims at roughly one box per shape,
    /// the usual starting point (Pharr, Jakob & Humphreys, *Physically Based
    /// Rendering*, §4.4). Larger means finer boxes and more of them.
    init(shapes: [GPUShape], density: Float = 1, maxCells: Int = 4_000_000) {
        var lo = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
        var hi = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)
        for s in shapes {
            let b = s.bounds()
            lo = simd_min(lo, b.lo)
            hi = simd_max(hi, b.hi)
        }
        if shapes.isEmpty { lo = .zero; hi = SIMD3(repeating: 1) }
        // A little air around the scene, so nothing sits exactly on a face.
        let pad = simd_max((hi - lo) * 0.001, SIMD3(repeating: 0.01))
        lo -= pad
        hi += pad
        let extent = hi - lo
        let longest = max(extent.x, max(extent.y, extent.z))
        let perUnit = density * 3 * Float(pow(Double(max(shapes.count, 1)), 1.0 / 3.0)) / longest
        var dims = SIMD3<Int>(1, 1, 1)
        for k in 0..<3 {
            dims[k] = max(1, min(512, Int((extent[k] * perUnit).rounded())))
        }
        while dims.x * dims.y * dims.z > maxCells {
            let k = dims.x >= dims.y && dims.x >= dims.z ? 0 : (dims.y >= dims.z ? 1 : 2)
            dims[k] = max(1, dims[k] / 2)
        }
        self.origin = lo
        self.dims = dims
        self.cellSize = SIMD3(extent.x / Float(dims.x), extent.y / Float(dims.y), extent.z / Float(dims.z))

        // Count what lands in each box, total them up, then fill.
        let cells = dims.x * dims.y * dims.z
        var counts = [UInt32](repeating: 0, count: cells)
        let origin = lo, cellSize = self.cellSize
        func span(_ s: GPUShape) -> (SIMD3<Int>, SIMD3<Int>) {
            let b = s.bounds()
            var a = SIMD3<Int>(0, 0, 0), z = SIMD3<Int>(0, 0, 0)
            for k in 0..<3 {
                a[k] = max(0, min(dims[k] - 1, Int(((b.lo[k] - origin[k]) / cellSize[k]).rounded(.down))))
                z[k] = max(0, min(dims[k] - 1, Int(((b.hi[k] - origin[k]) / cellSize[k]).rounded(.down))))
            }
            return (a, z)
        }
        for s in shapes {
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
        for (i, s) in shapes.enumerated() {
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
            let library = try device.makeLibrary(source: raytraceKernelSource, options: nil)
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

    /// Sorts the scene into a grid. Do this once for a scene whose shapes keep
    /// their positions; radii may then change freely, because the boxes were
    /// measured at full size and so stay correct for anything smaller.
    func buildGrid(_ shapes: [GPUShape], density: Float = 1) throws {
        let g = UniformGrid(shapes: shapes, density: density)
        grid = g
        startBuffer = try fill(nil, with: g.start)
        itemBuffer = try fill(nil, with: g.items.isEmpty ? [UInt32(0)] : g.items)
    }

    /// Ray-traces into the top `viewHeight` rows of `frame`. Returns GPU seconds.
    @discardableResult
    func render(shapes: [GPUShape], camera: Camera, into frame: MTLBuffer,
                width: Int, viewHeight: Int, samplesPerSide: Int = 2, useGrid: Bool = true) throws -> Double {
        // The lights turn with the camera, so a scene that never moves can be
        // lit as though it were the scene doing the turning.
        let c = cos(camera.lightSpin), s = sin(camera.lightSpin)
        func spin(_ v: SIMD3<Float>) -> SIMD3<Float> { SIMD3(v.x * c + v.z * s, v.y, -v.x * s + v.z * c) }
        let keyLight = spin(simd_normalize(SIMD3(-0.5, 0.7, 0.8)))
        let fillLight = spin(simd_normalize(SIMD3(0.7, -0.2, 0.5)))

        var cam = GPUCamera(origin: SIMD4(camera.origin, 1), forward: SIMD4(camera.forward, 0),
                            right: SIMD4(camera.right, 0), up: SIMD4(camera.up, 0),
                            key: SIMD4(keyLight, 0), fill: SIMD4(fillLight, 0),
                            tanHalfFOV: camera.tanHalfFOV, width: UInt32(width), height: UInt32(viewHeight),
                            shapeCount: UInt32(shapes.count), samplesPerSide: UInt32(samplesPerSide),
                            useGrid: useGrid ? 1 : 0)
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
