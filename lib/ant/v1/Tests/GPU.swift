// The module's Metal source, compiled on its own with a probe kernel, so the
// tests can ask the drawn distance function about arbitrary points.

import Foundation
import Metal
import simd

enum ProbeError: Error, CustomStringConvertible {
    case noDevice
    case compile(String)
    case gpu(String)
    var description: String {
        switch self {
        case .noDevice: return "no Metal GPU found"
        case .compile(let s): return "could not compile: \(s)"
        case .gpu(let s): return "GPU error: \(s)"
        }
    }
}

/// The probe: (distance to the nearest ant, part, light, shape index), and
/// the surface at a point: (albedo, alpha).
let probeKernels: String = """
kernel void antv1_probe(device const float4 *points [[buffer(0)]],
                        device float4 *out [[buffer(1)]],
                        constant uint &ants [[buffer(2)]],
                        constant AntV1Shape *S [[buffer(3)]],
                        constant float4 *A [[buffer(4)]],
                        uint id [[thread_position_in_grid]]) {
    float3 p = points[id].xyz;
    int part, lt, ix;
    float d = antv1_antsSDF(p, S, A, int(ants), 1e9, part, lt, ix);
    out[id] = float4(d, float(part), float(lt), float(ix));
}

kernel void antv1_probeSurface(device const float4 *points [[buffer(0)]],
                               device float4 *out [[buffer(1)]],
                               constant uint &ants [[buffer(2)]],
                               constant AntV1Shape *S [[buffer(3)]],
                               constant float4 *A [[buffer(4)]],
                               uint id [[thread_position_in_grid]]) {
    float3 p = points[id].xyz;
    int part, lt, ix;
    antv1_antsSDF(p, S, A, int(ants), 1e9, part, lt, ix);
    float3 albedo;
    float alpha;
    antv1_surface(p, float3(0, 1, 0), part, lt, ix, S, albedo, alpha);
    out[id] = float4(albedo, alpha);
}
"""

func makeBuf<T>(_ device: MTLDevice, _ array: [T]) throws -> MTLBuffer {
    let length: Int = max(MemoryLayout<T>.stride * array.count, 16)
    let made: MTLBuffer? = array.withUnsafeBytes { raw -> MTLBuffer? in
        guard let base = raw.baseAddress else { return device.makeBuffer(length: length, options: .storageModeShared) }
        return device.makeBuffer(bytes: base, length: length, options: .storageModeShared)
    }
    guard let b = made else { throw ProbeError.gpu("could not allocate a buffer") }
    return b
}

final class AntProbe {
    let device: MTLDevice
    private let queue: MTLCommandQueue
    private let probePSO: MTLComputePipelineState
    private let surfacePSO: MTLComputePipelineState

    init() throws {
        guard let d = MTLCreateSystemDefaultDevice() ?? MTLCopyAllDevices().first else { throw ProbeError.noDevice }
        device = d
        let options = MTLCompileOptions()
        // Precise maths, as in every step.
        options.fastMathEnabled = false
        let source: String = "#include <metal_stdlib>\nusing namespace metal;\n" + AntV1.metalSource + probeKernels
        let library: MTLLibrary
        do { library = try d.makeLibrary(source: source, options: options) } catch { throw ProbeError.compile("\(error)") }
        func pso(_ name: String) throws -> MTLComputePipelineState {
            guard let f = library.makeFunction(name: name) else { throw ProbeError.compile("no kernel \(name)") }
            do { return try d.makeComputePipelineState(function: f) } catch { throw ProbeError.compile("\(error)") }
        }
        probePSO = try pso("antv1_probe")
        surfacePSO = try pso("antv1_probeSurface")
        guard let q = d.makeCommandQueue() else { throw ProbeError.gpu("no queue") }
        queue = q
    }

    private func run(_ pso: MTLComputePipelineState, _ points: [SIMD3<Float>], ants: [AntV1.Posed]) throws -> [SIMD4<Float>] {
        let packed = AntV1.Packed(ants)
        let pts: [SIMD4<Float>] = points.map { SIMD4<Float>($0, 0) }
        let pb: MTLBuffer = try makeBuf(device, pts)
        let sb: MTLBuffer = try makeBuf(device, packed.shapes)
        let ab: MTLBuffer = try makeBuf(device, packed.ants)
        guard let ob = device.makeBuffer(length: 16 * max(pts.count, 1), options: .storageModeShared),
              let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder()
        else { throw ProbeError.gpu("could not set up the probe") }
        var count = UInt32(packed.count)
        enc.setComputePipelineState(pso)
        enc.setBuffer(pb, offset: 0, index: 0)
        enc.setBuffer(ob, offset: 0, index: 1)
        enc.setBytes(&count, length: MemoryLayout<UInt32>.stride, index: 2)
        enc.setBuffer(sb, offset: 0, index: 3)
        enc.setBuffer(ab, offset: 0, index: 4)
        let width: Int = min(pso.maxTotalThreadsPerThreadgroup, 256)
        enc.dispatchThreads(MTLSize(width: pts.count, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        if let e = cb.error { throw ProbeError.gpu(e.localizedDescription) }
        let o = ob.contents().assumingMemoryBound(to: SIMD4<Float>.self)
        return (0..<pts.count).map { o[$0] }
    }

    /// (distance to the nearest ant, part, light, shape index) at each point.
    func distance(_ points: [SIMD3<Float>], ants: [AntV1.Posed]) throws -> [SIMD4<Float>] {
        try run(probePSO, points, ants: ants)
    }

    /// (albedo, alpha) of the cuticle nearest each point.
    func surface(_ points: [SIMD3<Float>], ants: [AntV1.Posed]) throws -> [SIMD4<Float>] {
        try run(surfacePSO, points, ants: ants)
    }
}
