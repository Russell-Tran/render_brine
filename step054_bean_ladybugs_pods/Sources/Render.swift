// Running the kernel: packing a frame's plant into buffers, the camera,
// rendering, and probing the distance function at arbitrary points for the
// tests — from the same kernel source the picture uses.
//
// Written for step 45 and copied unchanged into steps 46 and 47; step 48
// passed each blade's tint to the kernel. Step 54 adds a buffer of beetles
// (the ladybirds' bodies) and their count in the View.

import CoreGraphics
import Foundation
import Metal
import simd

// MARK: - GPU layouts (float4s throughout, so nothing pads differently)

struct GPUChunk { var sphere: SIMD4<Float>; var first: UInt32; var count: UInt32; var mat: UInt32; var pad: UInt32 }
struct GPURibbon { var a: SIMD4<Float>; var b: SIMD4<Float>; var n: SIMD4<Float> }
struct GPULeaf {
    var o: SIMD4<Float>
    var u: SIMD4<Float>
    var v: SIMD4<Float>
    var w: SIMD4<Float>
    var bound: SIMD4<Float>
    var kind: SIMD4<Float>
}
struct GPUBeetle {
    var o: SIMD4<Float>
    var u: SIMD4<Float>
    var v: SIMD4<Float>
    var w: SIMD4<Float>
    var bound: SIMD4<Float>
}
struct GPUView {
    var pos: SIMD4<Float>
    var fwd: SIMD4<Float>
    var right: SIMD4<Float>
    var up: SIMD4<Float>
    var tanHalf: Float
    var width: UInt32
    var height: UInt32
    var rowOffset: UInt32
    var samples: UInt32
    var chunkN: UInt32
    var ribbonN: UInt32
    var leafN: UInt32
    var tubeMin: Float
    var mask: UInt32
    var beetleN: UInt32
    var pad1: Float
}

func v4(_ v: V3, _ w: Float = 0) -> SIMD4<Float> { SIMD4<Float>(v.x, v.y, v.z, w) }

/// Every material, for a probe that sees everything.
let allMaterials: UInt32 = 0xFFFF

func maskOf(_ ms: [Material]) -> UInt32 {
    ms.reduce(UInt32(0)) { $0 | (UInt32(1) << UInt32($1.rawValue)) }
}

/// A frame's plant packed for the GPU.
struct Packed {
    var verts: [SIMD4<Float>] = []
    var chunks: [GPUChunk] = []
    var ribbons: [GPURibbon] = []
    var leaves: [GPULeaf] = []
    var beetles: [GPUBeetle] = []

    init(_ s: PlantScene) {
        for chain in s.chains where chain.points.count >= 2 {
            let base: Int = verts.count
            for (i, p) in chain.points.enumerated() { verts.append(v4(p, chain.radii[i])) }
            let segments: Int = chain.points.count - 1
            for k in stride(from: 0, to: segments, by: 12) {
                let count: Int = min(12, segments - k)
                var lo = V3(repeating: 1e9)
                var hi = V3(repeating: -1e9)
                var rmax: Float = 0
                for i in k...(k + count) {
                    lo = simd_min(lo, chain.points[i])
                    hi = simd_max(hi, chain.points[i])
                    rmax = max(rmax, chain.radii[i])
                }
                let c: V3 = (lo + hi) / 2
                var rad: Float = 0
                for i in k...(k + count) { rad = max(rad, simd_distance(c, chain.points[i])) }
                // + 1.0: a tendril is drawn at least a pixel wide in the picture,
                // wider than its true radius; its sphere must still hold it.
                chunks.append(GPUChunk(sphere: v4(c, rad + rmax + 1.0), first: UInt32(base + k),
                                       count: UInt32(count), mat: UInt32(chain.material.rawValue), pad: 0))
            }
        }
        ribbons = s.ribbons.map { r in
            GPURibbon(a: v4(r.a, r.halfWidth), b: v4(r.b, r.halfThickness), n: v4(r.n, Float(r.material.rawValue)))
        }
        leaves = s.leaflets.map { l in
            // A sphere round the whole blade, drooped or not.
            let centre: V3 = l.origin + l.u * (l.length * 0.5) - l.w * (l.droop * l.length * 0.25)
            let radius: Float = l.length * 0.5 + l.halfWidth + l.droop * l.length + 0.5
            return GPULeaf(o: v4(l.origin, l.length), u: v4(l.u, l.halfWidth), v: v4(l.v, l.fold),
                           w: v4(l.w, l.droop), bound: v4(centre, radius),
                           kind: SIMD4<Float>(Float(l.shape.rawValue), Float(l.material.rawValue), l.tint, 0))
        }
        beetles = s.beetles.map { b in
            let bound: (centre: V3, radius: Float) = beetleBound(b)
            return GPUBeetle(o: v4(b.origin, b.scale), u: v4(b.forward), v: v4(b.left), w: v4(b.up),
                             bound: v4(bound.centre, bound.radius))
        }
    }
}

// MARK: - the camera

struct Camera {
    var position: V3
    var forward: V3
    var right: V3
    var up: V3
    var tanHalf: Float        // of the vertical field

    init(position: V3, target: V3, tanHalf: Float) {
        self.position = position
        forward = simd_normalize(target - position)
        right = simd_normalize(simd_cross(forward, V3(0, 1, 0)))
        up = simd_cross(right, forward)
        self.tanHalf = tanHalf
    }

    /// Pixel coordinates (top-left origin) of a world point.
    func project(_ p: V3, width: Int, height: Int) -> SIMD2<Float> {
        let d: V3 = p - position
        let z: Float = simd_dot(d, forward)
        let x: Float = simd_dot(d, right) / z
        let y: Float = simd_dot(d, up) / z
        let aspect: Float = Float(width) / Float(height)
        return SIMD2<Float>((x / (aspect * tanHalf) + 1) / 2 * Float(width), (1 - y / tanHalf) / 2 * Float(height))
    }

    /// Millimetres per pixel at the distance of a point.
    func millimetresPerPixel(at p: V3, height: Int) -> Float {
        let z: Float = simd_dot(p - position, forward)
        return 2 * tanHalf * z / Float(height)
    }
}

// MARK: - running it

/// Step 8's GIF encoder names its errors this way.
enum RenderError: Error, CustomStringConvertible {
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

func findDevice() throws -> MTLDevice {
    if let d = MTLCreateSystemDefaultDevice() ?? MTLCopyAllDevices().first { return d }
    throw RenderError.noMetalDevice
}

/// One frame's colour, coverage and what each pixel's centre saw.
struct Frame {
    let width: Int
    let height: Int
    /// Premultiplied display colour and coverage, row-major from the top.
    let rgba: [SIMD4<Float>]
    /// (material, distance, 0, 0) at each pixel's centre.
    let seen: [SIMD4<Float>]
}

final class PlantRenderer {
    let device: MTLDevice
    let queue: MTLCommandQueue
    let renderPSO: MTLComputePipelineState
    let probePSO: MTLComputePipelineState

    init(look: Look, on device: MTLDevice) throws {
        self.device = device
        let source: String = "#include <metal_stdlib>\nusing namespace metal;\n"
            + kernelConstants(look) + "\n" + kernelBody
        let options = MTLCompileOptions()
        // Precise maths: the distance functions are compared to within microns.
        options.fastMathEnabled = false
        let library: MTLLibrary
        do { library = try device.makeLibrary(source: source, options: options) } catch {
            throw RenderError.kernelCompile("\(error)")
        }
        guard let r = library.makeFunction(name: "render"), let p = library.makeFunction(name: "probe"),
              let q = device.makeCommandQueue() else { throw RenderError.kernelCompile("missing kernel") }
        do {
            renderPSO = try device.makeComputePipelineState(function: r)
            probePSO = try device.makeComputePipelineState(function: p)
        } catch { throw RenderError.kernelCompile("\(error)") }
        queue = q
    }

    private func buffer<T>(_ a: [T], dummy: T) throws -> MTLBuffer {
        let arr: [T] = a.isEmpty ? [dummy] : a
        let made: MTLBuffer? = arr.withUnsafeBytes { raw in
            device.makeBuffer(bytes: raw.baseAddress!, length: raw.count, options: .storageModeShared)
        }
        guard let b = made else { throw RenderError.gpu("could not allocate") }
        return b
    }

    private func sceneBuffers(_ p: Packed) throws -> [MTLBuffer] {
        let z = SIMD4<Float>(0, 0, 0, 0)
        return [try buffer(p.verts, dummy: z),
                try buffer(p.chunks, dummy: GPUChunk(sphere: z, first: 0, count: 0, mat: 0, pad: 0)),
                try buffer(p.ribbons, dummy: GPURibbon(a: z, b: z, n: z)),
                try buffer(p.leaves, dummy: GPULeaf(o: z, u: z, v: z, w: z, bound: z, kind: z)),
                try buffer(p.beetles, dummy: GPUBeetle(o: z, u: z, v: z, w: z, bound: z))]
    }

    private func view(_ cam: Camera, _ p: Packed, width: Int, height: Int, row: Int, samples: Int,
                      tubeMin: Float, mask: UInt32) -> GPUView {
        GPUView(pos: v4(cam.position), fwd: v4(cam.forward), right: v4(cam.right), up: v4(cam.up),
                tanHalf: cam.tanHalf, width: UInt32(width), height: UInt32(height), rowOffset: UInt32(row),
                samples: UInt32(samples), chunkN: UInt32(p.chunks.count), ribbonN: UInt32(p.ribbons.count),
                leafN: UInt32(p.leaves.count), tubeMin: tubeMin, mask: mask, beetleN: UInt32(p.beetles.count), pad1: 0)
    }

    /// Render one frame. `tubeMin` holds tendrils to at least that radius so a
    /// thread thinner than a pixel does not flicker in and out.
    func render(_ scene: PlantScene, camera: Camera, width: Int, height: Int, samples: Int,
                tubeMin: Float) throws -> (frame: Frame, gpuSeconds: Double) {
        let packed = Packed(scene)
        let bufs: [MTLBuffer] = try sceneBuffers(packed)
        guard let pixels = device.makeBuffer(length: width * height * 16, options: .storageModeShared),
              let aux = device.makeBuffer(length: width * height * 16, options: .storageModeShared)
        else { throw RenderError.gpu("could not allocate image buffers") }
        let band: Int = 48
        var gpu: Double = 0
        var row: Int = 0
        while row < height {
            let rows: Int = min(band, height - row)
            guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else {
                throw RenderError.gpu("could not make a command buffer")
            }
            var v: GPUView = view(camera, packed, width: width, height: height, row: row, samples: samples,
                                  tubeMin: tubeMin, mask: allMaterials)
            enc.setComputePipelineState(renderPSO)
            enc.setBuffer(pixels, offset: 0, index: 0)
            enc.setBuffer(aux, offset: 0, index: 1)
            enc.setBytes(&v, length: MemoryLayout<GPUView>.stride, index: 2)
            for (i, b) in bufs.enumerated() { enc.setBuffer(b, offset: 0, index: 3 + i) }
            let w: Int = renderPSO.threadExecutionWidth
            let group = MTLSize(width: w, height: max(renderPSO.maxTotalThreadsPerThreadgroup / w / 4, 1), depth: 1)
            enc.dispatchThreads(MTLSize(width: width, height: rows, depth: 1), threadsPerThreadgroup: group)
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            if let e = cb.error { throw RenderError.gpu(e.localizedDescription) }
            gpu += cb.gpuEndTime - cb.gpuStartTime
            row += rows
        }
        let n: Int = width * height
        let pp = pixels.contents().assumingMemoryBound(to: SIMD4<Float>.self)
        let ap = aux.contents().assumingMemoryBound(to: SIMD4<Float>.self)
        let frame = Frame(width: width, height: height, rgba: Array(UnsafeBufferPointer(start: pp, count: n)),
                          seen: Array(UnsafeBufferPointer(start: ap, count: n)))
        return (frame, gpu)
    }

    /// The distance function at arbitrary points, seeing only the materials in
    /// `mask`, at true radii. Returns (distance, material) per point.
    func probe(_ scene: PlantScene, _ points: [V3], mask: UInt32 = allMaterials) throws -> [SIMD2<Float>] {
        if points.isEmpty { return [] }
        let packed = Packed(scene)
        let bufs: [MTLBuffer] = try sceneBuffers(packed)
        var out: [SIMD2<Float>] = []
        out.reserveCapacity(points.count)
        let batch: Int = 1 << 18
        var start: Int = 0
        while start < points.count {
            let end: Int = min(start + batch, points.count)
            var pts: [SIMD4<Float>] = points[start..<end].map { v4($0) }
            let cam = Camera(position: V3(0, 0, 1), target: V3(0, 0, 0), tanHalf: 1)
            var v: GPUView = view(cam, packed, width: 1, height: 1, row: 0, samples: 1, tubeMin: 0, mask: mask)
            guard let pb = device.makeBuffer(bytes: &pts, length: 16 * pts.count, options: .storageModeShared),
                  let ob = device.makeBuffer(length: 8 * pts.count, options: .storageModeShared),
                  let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder()
            else { throw RenderError.gpu("could not set up the probe") }
            enc.setComputePipelineState(probePSO)
            enc.setBuffer(pb, offset: 0, index: 0)
            enc.setBuffer(ob, offset: 0, index: 1)
            enc.setBytes(&v, length: MemoryLayout<GPUView>.stride, index: 2)
            for (i, b) in bufs.enumerated() { enc.setBuffer(b, offset: 0, index: 3 + i) }
            enc.dispatchThreads(MTLSize(width: pts.count, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: min(probePSO.maxTotalThreadsPerThreadgroup, 128), height: 1, depth: 1))
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            if let e = cb.error { throw RenderError.gpu(e.localizedDescription) }
            let o = ob.contents().assumingMemoryBound(to: SIMD2<Float>.self)
            out += Array(UnsafeBufferPointer(start: o, count: pts.count))
            start = end
        }
        return out
    }
}
