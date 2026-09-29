// Step 79: the pheromone field, a grid on the GPU (Metal compute). Each tick:
//
//   1. DEPOSIT — this tick's dabs are added, each split bilinearly over the
//      four cells around its point, so each adds exactly its marks.
//   2. DIFFUSE and EVAPORATE, fused, from one buffer into the other:
//        c' = (c + α · Σ_neighbours (c_n − c)) · f,
//      α = D·dt/h², f = e^(−dt/τ). Zero flux through the edges (a neighbour
//      that doesn't exist contributes nothing), so diffusion moves pheromone
//      but never makes or loses any; evaporation is an exact factor per tick.
//
// Units: marks per mm² (SimConstants.swift). Cell (i, j) covers ground
// x ∈ [i·h, (i+1)·h), z ∈ [j·h, (j+1)·h); row j is stored at j·width.
//
// Determinism: every cell is written by one thread, in a fixed order of
// operations, and dabs are summed in the order given — no atomics — so the
// same dabs give the same field bit for bit.

import Foundation
import Metal

enum SimFieldError: Error {
    case noDevice
    case compile(String)
    case gpu(String)
}

/// A dab as the GPU takes it: the lower-left cell of its four and the four
/// bilinear weights times its marks.
struct SimDabGPU {
    var ix: Float
    var iy: Float
    var w00: Float
    var w10: Float
    var w01: Float
    var w11: Float
    var pad0: Float = 0
    var pad1: Float = 0
}

/// Must match `FieldParams` in the kernel.
struct SimFieldParams {
    var width: UInt32
    var height: UInt32
    var alpha: Float
    var decay: Float
    var dabCount: UInt32
}

let simFieldKernelSource: String = """
#include <metal_stdlib>
using namespace metal;

struct FieldParams { uint width; uint height; float alpha; float decay; uint dabCount; };
struct Dab { float ix; float iy; float w00; float w10; float w01; float w11; float pad0; float pad1; };

kernel void deposit(device float* c [[buffer(0)]],
                    constant FieldParams& p [[buffer(1)]],
                    device const Dab* dabs [[buffer(2)]],
                    uint2 g [[thread_position_in_grid]]) {
    if (g.x >= p.width || g.y >= p.height) return;
    uint i = g.y * p.width + g.x;
    float add = 0.0f;
    for (uint k = 0; k < p.dabCount; k++) {
        Dab d = dabs[k];
        int dx = int(g.x) - int(d.ix);
        int dy = int(g.y) - int(d.iy);
        if (dx == 0 && dy == 0) add += d.w00;
        if (dx == 1 && dy == 0) add += d.w10;
        if (dx == 0 && dy == 1) add += d.w01;
        if (dx == 1 && dy == 1) add += d.w11;
    }
    c[i] = c[i] + add;
}

kernel void spread(device const float* src [[buffer(0)]],
                   device float* dst [[buffer(1)]],
                   constant FieldParams& p [[buffer(2)]],
                   uint2 g [[thread_position_in_grid]]) {
    if (g.x >= p.width || g.y >= p.height) return;
    uint i = g.y * p.width + g.x;
    float c = src[i];
    float lap = 0.0f;
    if (g.x > 0) lap += src[i - 1] - c;
    if (g.x + 1 < p.width) lap += src[i + 1] - c;
    if (g.y > 0) lap += src[i - p.width] - c;
    if (g.y + 1 < p.height) lap += src[i + p.width] - c;
    float moved = c + p.alpha * lap;
    dst[i] = moved * p.decay;
}
"""

/// The Metal pipelines, compiled once and shared by every field.
final class SimGPU {
    let device: MTLDevice
    let queue: MTLCommandQueue
    let depositPSO: MTLComputePipelineState
    let spreadPSO: MTLComputePipelineState

    init() throws {
        guard let d = MTLCreateSystemDefaultDevice() ?? MTLCopyAllDevices().first else { throw SimFieldError.noDevice }
        device = d
        let options = MTLCompileOptions()
        // Precise maths, as in every step (its deprecation warning is expected).
        options.fastMathEnabled = false
        let library: MTLLibrary
        do { library = try d.makeLibrary(source: simFieldKernelSource, options: options) } catch {
            throw SimFieldError.compile("\(error)")
        }
        func pso(_ name: String) throws -> MTLComputePipelineState {
            guard let f = library.makeFunction(name: name) else { throw SimFieldError.compile("no kernel \(name)") }
            do { return try d.makeComputePipelineState(function: f) } catch { throw SimFieldError.compile("\(error)") }
        }
        depositPSO = try pso("deposit")
        spreadPSO = try pso("spread")
        guard let q = d.makeCommandQueue() else { throw SimFieldError.gpu("no command queue") }
        queue = q
    }

    /// One per process.
    static func shared() throws -> SimGPU {
        if let g = sharedGPU { return g }
        let g = try SimGPU()
        sharedGPU = g
        return g
    }
}

private var sharedGPU: SimGPU?

/// A pheromone grid on the GPU.
final class SimField {
    let width: Int
    let height: Int
    let cell: Float
    /// α = D·dt/h² and f = e^(−dt/τ) per tick. Set by the simulation from
    /// SimConst; tests set them to isolate diffusion or evaporation.
    var alpha: Float
    var decay: Float
    private let gpu: SimGPU
    private var buffers: [MTLBuffer]
    private var current: Int = 0
    private let dabBuffer: MTLBuffer
    let maxDabsPerTick: Int = 256

    init(width: Int, height: Int, cell: Float, alpha: Float, decay: Float) throws {
        self.width = width
        self.height = height
        self.cell = cell
        self.alpha = alpha
        self.decay = decay
        gpu = try SimGPU.shared()
        let bytes: Int = width * height * MemoryLayout<Float>.stride
        var made: [MTLBuffer] = []
        for _ in 0..<2 {
            guard let b = gpu.device.makeBuffer(length: bytes, options: .storageModeShared) else {
                throw SimFieldError.gpu("could not allocate the field")
            }
            memset(b.contents(), 0, bytes)
            made.append(b)
        }
        buffers = made
        let dabBytes: Int = maxDabsPerTick * MemoryLayout<SimDabGPU>.stride
        guard let db = gpu.device.makeBuffer(length: dabBytes, options: .storageModeShared) else {
            throw SimFieldError.gpu("could not allocate the dab list")
        }
        dabBuffer = db
    }

    /// The standard field for the simulation's constants.
    static func forSimulation(width: Int, height: Int) throws -> SimField {
        let dt: Double = SimConst.tick
        let h: Float = SimConst.cell
        let dtF: Float = Float(dt)
        let area: Float = h * h
        let a: Float = SimConst.diffusion * dtF / area
        let ratio: Double = -dt / SimConst.pheromoneLifetime
        let f: Float = Float(exp(ratio))
        return try SimField(width: width, height: height, cell: h, alpha: a, decay: f)
    }

    var count: Int { width * height }

    /// The live values (valid between ticks; the GPU has finished).
    var values: UnsafeBufferPointer<Float> {
        let p: UnsafeMutablePointer<Float> = buffers[current].contents().assumingMemoryBound(to: Float.self)
        return UnsafeBufferPointer(start: UnsafePointer(p), count: count)
    }

    func snapshot() -> [Float] { Array(values) }

    func load(_ v: [Float]) {
        precondition(v.count == count)
        let p: UnsafeMutablePointer<Float> = buffers[current].contents().assumingMemoryBound(to: Float.self)
        v.withUnsafeBufferPointer { src in
            if let base = src.baseAddress { p.update(from: base, count: count) }
        }
    }

    /// Σ over cells × cell area: total marks on the ground (Double sum).
    func totalMarks() -> Double {
        var s: Double = 0
        for v in values { s += Double(v) }
        let area: Double = Double(cell) * Double(cell)
        return s * area
    }

    /// Bilinear sample at ground point (x, z), marks/mm²; zero outside.
    func sample(x: Float, z: Float) -> Float {
        let fx: Float = x / cell - 0.5
        let fy: Float = z / cell - 0.5
        let x0: Float = fx.rounded(.down)
        let y0: Float = fy.rounded(.down)
        let tx: Float = fx - x0
        let ty: Float = fy - y0
        let i0: Int = Int(x0)
        let j0: Int = Int(y0)
        let v: UnsafeBufferPointer<Float> = values
        func at(_ i: Int, _ j: Int) -> Float {
            if i < 0 || j < 0 || i >= width || j >= height { return 0 }
            return v[j * width + i]
        }
        let a: Float = at(i0, j0) + (at(i0 + 1, j0) - at(i0, j0)) * tx
        let b: Float = at(i0, j0 + 1) + (at(i0 + 1, j0 + 1) - at(i0, j0 + 1)) * tx
        return a + (b - a) * ty
    }

    /// A dab of `marks` at ground point (x, z) as the GPU takes it: bilinear
    /// weights over the four cell centres around it, divided by the cell area
    /// so the field's integral grows by exactly `marks`.
    func gpuDab(x: Float, z: Float, marks: Float) -> SimDabGPU {
        let fx: Float = x / cell - 0.5
        let fy: Float = z / cell - 0.5
        let x0: Float = fx.rounded(.down)
        let y0: Float = fy.rounded(.down)
        let tx: Float = fx - x0
        let ty: Float = fy - y0
        let density: Float = marks / (cell * cell)
        let ux: Float = 1 - tx
        let uy: Float = 1 - ty
        let w00: Float = ux * uy * density
        let w10: Float = tx * uy * density
        let w01: Float = ux * ty * density
        let w11: Float = tx * ty * density
        return SimDabGPU(ix: x0, iy: y0, w00: w00, w10: w10, w01: w01, w11: w11)
    }

    /// One tick: deposit `dabs`, then diffuse and evaporate.
    func step(dabs: [SimDabGPU]) throws {
        precondition(dabs.count <= maxDabsPerTick, "too many dabs in one tick")
        guard let cb = gpu.queue.makeCommandBuffer() else { throw SimFieldError.gpu("no command buffer") }
        var params = SimFieldParams(width: UInt32(width), height: UInt32(height), alpha: alpha, decay: decay,
                                    dabCount: UInt32(dabs.count))
        let grid = MTLSize(width: width, height: height, depth: 1)
        let group = MTLSize(width: 16, height: 16, depth: 1)
        if !dabs.isEmpty {
            let dst: UnsafeMutablePointer<SimDabGPU> = dabBuffer.contents().assumingMemoryBound(to: SimDabGPU.self)
            for (k, d) in dabs.enumerated() { dst[k] = d }
            guard let enc = cb.makeComputeCommandEncoder() else { throw SimFieldError.gpu("no encoder") }
            enc.setComputePipelineState(gpu.depositPSO)
            enc.setBuffer(buffers[current], offset: 0, index: 0)
            enc.setBytes(&params, length: MemoryLayout<SimFieldParams>.stride, index: 1)
            enc.setBuffer(dabBuffer, offset: 0, index: 2)
            enc.dispatchThreads(grid, threadsPerThreadgroup: group)
            enc.endEncoding()
        }
        guard let enc2 = cb.makeComputeCommandEncoder() else { throw SimFieldError.gpu("no encoder") }
        enc2.setComputePipelineState(gpu.spreadPSO)
        enc2.setBuffer(buffers[current], offset: 0, index: 0)
        enc2.setBuffer(buffers[1 - current], offset: 0, index: 1)
        enc2.setBytes(&params, length: MemoryLayout<SimFieldParams>.stride, index: 2)
        enc2.dispatchThreads(grid, threadsPerThreadgroup: group)
        enc2.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        if let e = cb.error { throw SimFieldError.gpu("\(e)") }
        current = 1 - current
    }
}
