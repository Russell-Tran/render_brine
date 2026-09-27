// Running the kernel: the scene's buffers, the cameras, rendering a layer, and
// probing the distance function at arbitrary points for the tests — from the
// same kernel source the picture uses.

import CoreGraphics
import Foundation
import ImageIO
import Metal
import UniformTypeIdentifiers
import simd

// MARK: - GPU layouts (float4s throughout, so nothing pads differently)

struct GPUSpine { var p: SIMD4<Float>; var n: SIMD4<Float>; var b: SIMD4<Float>; var r: SIMD4<Float> }
struct GPUChunk { var sphere: SIMD4<Float>; var first: UInt32; var count: UInt32; var mat: UInt32; var pad: UInt32 }
struct GPUAnther { var c: SIMD4<Float>; var ax: SIMD4<Float>; var sd: SIMD4<Float> }
struct GPUGrain { var c: SIMD4<Float>; var pole: SIMD4<Float>; var pore: SIMD4<Float> }
struct GPUHair { var base: SIMD4<Float>; var tip: SIMD4<Float> }
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
    var layer: UInt32
    var tubeMin: Float
    var pad: Float
}

func v4(_ v: SIMD3<Float>, _ w: Float = 0) -> SIMD4<Float> { SIMD4<Float>(v.x, v.y, v.z, w) }

/// Hair radii at base and tip. MODEL.
let hairBaseRadius: Float = 0.0065
let hairTipRadius: Float = 0.003

/// Everything packed for the GPU, and the counts the kernel is compiled with.
struct SceneLayout {
    var spine: [GPUSpine] = []
    var spineChunks: [SIMD4<Float>] = []
    var verts: [SIMD4<Float>] = []
    var chunks: [GPUChunk] = []
    var anthers: [GPUAnther] = []
    var grains: [GPUGrain] = []
    var ovules: [SIMD4<Float>] = []
    var hairs: [GPUHair] = []
    var tubeChunkStart: Int = 0
    var tipCentre = SIMD3<Float>(0, 0, 0)
    var tipRadius: Float = 0

    var spineChunkCount: Int { spineChunks.count }
    var chunkCount: Int { chunks.count }

    init(_ m: FlowerModel) {
        spine = Bean_spine()
        for k in stride(from: 0, to: spine.count - 1, by: 16) {
            let last: Int = min(k + 16, spine.count - 1)
            var lo = SIMD3<Float>(repeating: 1e9)
            var hi = SIMD3<Float>(repeating: -1e9)
            var rmax: Float = 0
            for i in k...last {
                let p = SIMD3<Float>(spine[i].p.x, spine[i].p.y, spine[i].p.z)
                lo = simd_min(lo, p)
                hi = simd_max(hi, p)
                rmax = max(rmax, spine[i].r.x)
            }
            let c: SIMD3<Float> = (lo + hi) / 2
            var rad: Float = 0
            for i in k...last {
                rad = max(rad, simd_distance(c, SIMD3<Float>(spine[i].p.x, spine[i].p.y, spine[i].p.z)))
            }
            spineChunks.append(v4(c, rad + rmax + 0.01))
        }
        for chain in m.chains {
            if chain.material == materialTube { tubeChunkStart = chunks.count }
            let base: Int = verts.count
            for (i, p) in chain.points.enumerated() { verts.append(v4(p, chain.radii[i])) }
            let segments: Int = chain.points.count - 1
            for k in stride(from: 0, to: segments, by: 16) {
                let count: Int = min(16, segments - k)
                var lo = SIMD3<Float>(repeating: 1e9)
                var hi = SIMD3<Float>(repeating: -1e9)
                var rmax: Float = 0
                for i in k...(k + count) {
                    lo = simd_min(lo, chain.points[i])
                    hi = simd_max(hi, chain.points[i])
                    rmax = max(rmax, chain.radii[i])
                }
                let c: SIMD3<Float> = (lo + hi) / 2
                var rad: Float = 0
                for i in k...(k + count) { rad = max(rad, simd_distance(c, chain.points[i])) }
                // + 0.012 so a tube drawn wider than life for the main view is still inside.
                chunks.append(GPUChunk(sphere: v4(c, rad + rmax + 0.012), first: UInt32(base + k),
                                       count: UInt32(count), mat: chain.material, pad: 0))
            }
        }
        anthers = m.anthers.map { GPUAnther(c: v4($0.centre), ax: v4($0.axis), sd: v4($0.side)) }
        grains = m.grains.map { g in
            // The pore reference: for the germinating grain, the pore its tube
            // leaves by; for the rest, any direction on the equator.
            var e: SIMD3<Float> = germinationPore - g.centre
            if simd_distance(g.centre, germinatingGrain.centre) > 1e-6 {
                e = simd_cross(g.pole, SIMD3<Float>(0.31, 0.52, 0.79))
            }
            e = simd_normalize(e - g.pole * simd_dot(e, g.pole))
            return GPUGrain(c: v4(g.centre, grainEquatorialRadius), pole: v4(g.pole, grainPolarRadius), pore: v4(e))
        }
        ovules = m.ovules.map { v4($0) }
        hairs = m.hairs.map { GPUHair(base: v4($0.base, hairBaseRadius), tip: v4($0.tip, hairTipRadius)) }
        // One sphere round the tip's hairs, anthers and pollen.
        var pts: [SIMD3<Float>] = m.anthers.map { $0.centre } + m.grains.map { $0.centre }
        pts += m.hairs.flatMap { [$0.base, $0.tip] }
        let c: SIMD3<Float> = pts.reduce(SIMD3<Float>(0, 0, 0), +) / Float(pts.count)
        tipCentre = c
        tipRadius = pts.map { simd_distance($0, c) }.max()! + antherSemiAxes.x + 0.03
    }
}

func Bean_spine() -> [GPUSpine] {
    spine.map { s in
        GPUSpine(p: v4(s.p, s.s), n: v4(s.n), b: v4(s.b), r: SIMD4<Float>(s.keelRadius, s.styleRadius, 0, 0))
    }
}

// MARK: - cameras

struct Camera {
    var position: SIMD3<Float>
    var forward: SIMD3<Float>
    var right: SIMD3<Float>
    var up: SIMD3<Float>
    var tanHalf: Float        // of the vertical field

    init(target: SIMD3<Float>, direction: SIMD3<Float>, distance: Float, halfHeightAtTarget: Float) {
        let back: SIMD3<Float> = simd_normalize(direction)
        position = target + back * distance
        forward = -back
        right = simd_normalize(simd_cross(forward, SIMD3<Float>(0, 1, 0)))
        up = simd_cross(right, forward)
        tanHalf = halfHeightAtTarget / distance
    }

    /// Pixel coordinates (top-left origin) of a world point.
    func project(_ p: SIMD3<Float>, width: Int, height: Int) -> SIMD2<Float> {
        let d: SIMD3<Float> = p - position
        let z: Float = simd_dot(d, forward)
        let x: Float = simd_dot(d, right) / z
        let y: Float = simd_dot(d, up) / z
        let aspect: Float = Float(width) / Float(height)
        let px: Float = (x / (aspect * tanHalf) + 1) / 2 * Float(width)
        let py: Float = (1 - y / tanHalf) / 2 * Float(height)
        return SIMD2<Float>(px, py)
    }

    /// Pixels per millimetre at the distance of a point.
    func pixelsPerMillimetre(at p: SIMD3<Float>, height: Int) -> Float {
        let z: Float = simd_dot(p - position, forward)
        return Float(height) / (2 * tanHalf * z)
    }
}

let cameraDistance: Float = 60
/// 16 mm of frame height at the flower: 67.5 px/mm at 1080 lines, so a
/// 45 µm grain is 3 px — fine gold dust, at true scale.
let mainHalfHeight: Float = 6.8
let mainTarget = SIMD3<Float>(10.3, 1.0, 0)

func mainCamera() -> Camera {
    Camera(target: mainTarget, direction: viewDirection, distance: cameraDistance, halfHeightAtTarget: mainHalfHeight)
}

/// The inset: the same direction, centred just behind the stigma, 0.56 mm
/// across its diameter.
let insetFieldDiameter: Float = 0.56
let insetTarget: SIMD3<Float> = stigmaCentre - stigmaFrame.t * 0.07 + stigmaFrame.n * 0.02

func insetCamera() -> Camera {
    Camera(target: insetTarget, direction: viewDirection, distance: cameraDistance, halfHeightAtTarget: insetFieldDiameter / 2)
}

struct PodFrame {
    var origin: SIMD3<Float>
    var x: SIMD3<Float>
    var y: SIMD3<Float>
    var z: SIMD3<Float>
}

/// The pod lies far behind the flower — 9× the camera distance, so 12 cm of
/// pod spans about 13 mm at the flower's depth — below it, and drawn parallel
/// to the ovary as the picture shows it, pedicel end left and beak right, so
/// the two read as the same organ at two ages. Its flat face is turned to the
/// camera.
func podFrame() -> PodFrame {
    let cam: Camera = mainCamera()
    let k: Float = 9.0
    let depth: Float = cameraDistance * k
    let centre: SIMD3<Float> = cam.position + cam.forward * depth
        + cam.up * (-4.4 * k) + cam.right * (-0.9 * k)
    // The ovary's direction on screen.
    let a: SIMD2<Float> = cam.project(SIMD3<Float>(ovaryStartX, 0, 0), width: 1920, height: 1080)
    let b: SIMD2<Float> = cam.project(SIMD3<Float>(ovaryEndX, 0, 0), width: 1920, height: 1080)
    let d: SIMD2<Float> = simd_normalize(b - a)
    let x: SIMD3<Float> = simd_normalize(cam.right * d.x - cam.up * d.y)
    let z: SIMD3<Float> = -cam.forward
    let y: SIMD3<Float> = simd_cross(z, x)
    return PodFrame(origin: centre - x * (podLength / 2), x: x, y: y, z: z)
}

// MARK: - running it

enum BeanError: Error, CustomStringConvertible {
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
    throw BeanError.noMetalDevice
}

/// The compiled kernel and the scene's buffers, for one flower model.
final class BeanScene {
    let device: MTLDevice
    let model: FlowerModel
    let layout: SceneLayout
    let library: MTLLibrary
    let buffers: [MTLBuffer]

    init(_ model: FlowerModel, on device: MTLDevice) throws {
        self.device = device
        self.model = model
        layout = SceneLayout(model)
        let source: String = "#include <metal_stdlib>\nusing namespace metal;\n"
            + kernelConstants(model, layout) + "\n" + kernelBody
        let options = MTLCompileOptions()
        // Precise maths: the distance functions are compared to within a micron.
        options.fastMathEnabled = false
        do {
            library = try device.makeLibrary(source: source, options: options)
        } catch {
            throw BeanError.kernelCompile("\(error)")
        }
        func buffer<T>(_ a: [T]) throws -> MTLBuffer {
            if a.isEmpty { throw BeanError.gpu("empty scene buffer") }
            let made: MTLBuffer? = a.withUnsafeBytes { raw in
                device.makeBuffer(bytes: raw.baseAddress!, length: raw.count, options: .storageModeShared)
            }
            guard let b = made else { throw BeanError.gpu("could not allocate") }
            return b
        }
        buffers = [try buffer(layout.spine), try buffer(layout.spineChunks), try buffer(layout.verts),
                   try buffer(layout.chunks), try buffer(layout.anthers), try buffer(layout.grains),
                   try buffer(layout.ovules), try buffer(layout.hairs)]
    }

    func pipeline(_ name: String) throws -> MTLComputePipelineState {
        guard let f = library.makeFunction(name: name) else { throw BeanError.kernelCompile("no kernel \(name)") }
        do { return try device.makeComputePipelineState(function: f) } catch {
            throw BeanError.kernelCompile("\(error)")
        }
    }

    /// One layer: 0 the flower, 1 the pod. Premultiplied display colour and
    /// coverage per pixel, and what each pixel's centre saw.
    func render(camera: Camera, width: Int, height: Int, samples: Int, layer: Int,
                tubeMin: Float) throws -> (image: LayerImage, gpuSeconds: Double) {
        let pso = try pipeline("render")
        guard let pixels = device.makeBuffer(length: width * height * 16, options: .storageModeShared),
              let aux = device.makeBuffer(length: width * height * 16, options: .storageModeShared),
              let queue = device.makeCommandQueue()
        else { throw BeanError.gpu("could not allocate image buffers") }
        let band: Int = 24
        var gpu: Double = 0
        var row: Int = 0
        while row < height {
            let rows: Int = min(band, height - row)
            guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else {
                throw BeanError.gpu("could not make a command buffer")
            }
            var view = GPUView(pos: v4(camera.position), fwd: v4(camera.forward), right: v4(camera.right),
                               up: v4(camera.up), tanHalf: camera.tanHalf, width: UInt32(width),
                               height: UInt32(height), rowOffset: UInt32(row), samples: UInt32(samples),
                               layer: UInt32(layer), tubeMin: tubeMin, pad: 0)
            enc.setComputePipelineState(pso)
            enc.setBuffer(pixels, offset: 0, index: 0)
            enc.setBuffer(aux, offset: 0, index: 1)
            enc.setBytes(&view, length: MemoryLayout<GPUView>.stride, index: 2)
            for (i, b) in buffers.enumerated() { enc.setBuffer(b, offset: 0, index: 3 + i) }
            let w: Int = pso.threadExecutionWidth
            let group = MTLSize(width: w, height: max(pso.maxTotalThreadsPerThreadgroup / w / 4, 1), depth: 1)
            enc.dispatchThreads(MTLSize(width: width, height: rows, depth: 1), threadsPerThreadgroup: group)
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            if let e = cb.error { throw BeanError.gpu(e.localizedDescription) }
            gpu += cb.gpuEndTime - cb.gpuStartTime
            row += rows
        }
        return (LayerImage(width: width, height: height, pixels: pixels, aux: aux), gpu)
    }

    /// The distance function at arbitrary points, in one of the kernel's probe
    /// modes. Returns (distance, material).
    func probe(_ points: [SIMD3<Float>], mode: ProbeMode) throws -> [SIMD2<Float>] {
        let pso = try pipeline("probe")
        var pts: [SIMD4<Float>] = points.map { v4($0) }
        var m: Int32 = Int32(mode.rawValue)
        guard let pb = device.makeBuffer(bytes: &pts, length: 16 * max(pts.count, 1), options: .storageModeShared),
              let ob = device.makeBuffer(length: 8 * max(pts.count, 1), options: .storageModeShared),
              let queue = device.makeCommandQueue(), let cb = queue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder()
        else { throw BeanError.gpu("could not set up the probe") }
        enc.setComputePipelineState(pso)
        enc.setBuffer(pb, offset: 0, index: 0)
        enc.setBuffer(ob, offset: 0, index: 1)
        enc.setBytes(&m, length: 4, index: 2)
        for (i, b) in buffers.enumerated() { enc.setBuffer(b, offset: 0, index: 3 + i) }
        enc.dispatchThreads(MTLSize(width: pts.count, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(pso.maxTotalThreadsPerThreadgroup, 128), height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        if let e = cb.error { throw BeanError.gpu(e.localizedDescription) }
        let out = ob.contents().assumingMemoryBound(to: SIMD2<Float>.self)
        return (0..<pts.count).map { out[$0] }
    }
}

/// The kernel's probe modes; the raw values match the P_ constants in it.
enum ProbeMode: Int {
    case render = 0, uncut, pistil, ovules, grains, enclosure, stamens, keelCavity, tube, banner
}

/// Materials, as the kernel numbers them.
enum Material: Int {
    case none = 0, banner, wing, keel, green, ovary, ovule, stamen, anther, style, hair, pollen, tube, pod
}

struct LayerImage {
    let width: Int
    let height: Int
    let pixels: MTLBuffer
    let aux: MTLBuffer

    /// Premultiplied display colour and coverage.
    func rgba(_ x: Int, _ y: Int) -> SIMD4<Float> {
        pixels.contents().assumingMemoryBound(to: SIMD4<Float>.self)[y * width + x]
    }

    /// (material, cut face?, distance, 0) at the pixel's centre.
    func seen(_ x: Int, _ y: Int) -> SIMD4<Float> {
        aux.contents().assumingMemoryBound(to: SIMD4<Float>.self)[y * width + x]
    }
}
