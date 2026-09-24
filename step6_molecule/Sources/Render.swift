// A small GPU ray tracer for ball-and-stick molecules.
//
// One thread per pixel. Each thread shoots a few rays through its pixel
// (2 × 2, as in step 5), finds the nearest atom (sphere) or bond (cylinder),
// and shades it with a key light, a fill light and a highlight. Charged atoms
// also get a soft glow.

import Foundation
import Metal
import simd

// These three structs must match the kernel. SIMD4<Float> keeps the Swift and
// Metal layouts identical (16-byte aligned).
struct GPUSphere {
    var centerRadius: SIMD4<Float>
    var color: SIMD4<Float>
    var glow: SIMD4<Float>        // rgb = glow color × strength, w = glow width (Å)
}
struct GPUCylinder {
    var aRadius: SIMD4<Float>
    var b: SIMD4<Float>
    var color: SIMD4<Float>
}
struct GPUCamera {
    var origin: SIMD4<Float>
    var forward: SIMD4<Float>
    var right: SIMD4<Float>
    var up: SIMD4<Float>
    var tanHalfFOV: Float
    var width: UInt32             // image width in pixels
    var height: UInt32            // height of the molecule view in pixels
    var sphereCount: UInt32
    var cylinderCount: UInt32
    var samplesPerSide: UInt32
}

struct Camera {
    var origin: SIMD3<Float>
    var target: SIMD3<Float>
    var tanHalfFOV: Float

    var forward: SIMD3<Float> { simd_normalize(target - origin) }
    var right: SIMD3<Float> { simd_normalize(simd_cross(forward, SIMD3(0, 1, 0))) }
    var up: SIMD3<Float> { simd_cross(right, forward) }

    /// A camera `distance` Å from `target`, turned `yaw` degrees around it and
    /// raised `pitch` degrees.
    static func orbit(target: SIMD3<Float>, distance: Float, yaw: Float, pitch: Float, fov: Float) -> Camera {
        let y = radians(yaw), p = radians(pitch)
        let offset = SIMD3<Float>(sin(y) * cos(p), sin(p), cos(y) * cos(p)) * distance
        return Camera(origin: target + offset, target: target, tanHalfFOV: tan(radians(fov / 2)))
    }

    /// Where a point lands in a width × height image (pixels, from the top left).
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
}

let bondRadius: Float = 0.085
let doubleBondOffset: Float = 0.11
let positiveGlow = SIMD3<Float>(1.0, 0.78, 0.25)   // warm gold for +
let negativeGlow = SIMD3<Float>(0.30, 0.80, 1.0)   // cyan for −

/// Turns a molecule into spheres and cylinders. A bond of order 1 is one
/// stick; order 2 is two sticks; order 1.5 is one stick plus a half-thick one.
/// A bond fading out (order below 1) gets thinner.
func sceneGeometry(_ state: MoleculeState) -> (spheres: [GPUSphere], cylinders: [GPUCylinder]) {
    var spheres: [GPUSphere] = []
    for atom in state.atoms {
        let p = atom.position
        let c = atom.element.color
        var glow = SIMD4<Float>(0, 0, 0, 0)
        if atom.charge > 0.01 {
            let g = positiveGlow * atom.charge * 1.1
            glow = SIMD4(g.x, g.y, g.z, 0.45)
        } else if atom.charge < -0.01 {
            let g = negativeGlow * (-atom.charge) * 1.3
            glow = SIMD4(g.x, g.y, g.z, 0.40)
        }
        spheres.append(GPUSphere(centerRadius: SIMD4(p.x, p.y, p.z, atom.element.ballRadius),
                                 color: SIMD4(c.x, c.y, c.z, 1), glow: glow))
    }
    var cylinders: [GPUCylinder] = []
    let grey = SIMD4<Float>(0.62, 0.64, 0.67, 1)
    for bond in state.bonds where bond.order > 0.02 {
        let a = state.atoms[bond.a].position
        let b = state.atoms[bond.b].position
        let along = simd_normalize(b - a)
        let side = simd_normalize(simd_cross(along, SIMD3<Float>(0, 0, 1)))
        if bond.order <= 1 {
            cylinders.append(GPUCylinder(aRadius: SIMD4(a.x, a.y, a.z, bondRadius * bond.order),
                                         b: SIMD4(b.x, b.y, b.z, 0), color: grey))
        } else {
            let extra: Float = bond.order - 1
            let shiftMain = side * (-doubleBondOffset * extra)
            let shiftExtra = side * doubleBondOffset * extra
            let a1 = a + shiftMain, b1 = b + shiftMain
            let a2 = a + shiftExtra, b2 = b + shiftExtra
            cylinders.append(GPUCylinder(aRadius: SIMD4(a1.x, a1.y, a1.z, bondRadius),
                                         b: SIMD4(b1.x, b1.y, b1.z, 0), color: grey))
            cylinders.append(GPUCylinder(aRadius: SIMD4(a2.x, a2.y, a2.z, bondRadius * extra),
                                         b: SIMD4(b2.x, b2.y, b2.z, 0), color: grey))
        }
    }
    return (spheres, cylinders)
}

let raytraceKernelSource = """
    #include <metal_stdlib>
    using namespace metal;

    struct Sphere { float4 centerRadius; float4 color; float4 glow; };
    struct Cylinder { float4 aRadius; float4 b; float4 color; };
    struct Camera {
        float4 origin; float4 forward; float4 right; float4 up;
        float tanHalfFOV; uint width; uint height; uint sphereCount; uint cylinderCount; uint samplesPerSide;
    };

    // Unit vectors (Metal constants can't call normalize).
    constant float3 KEY = float3(-0.4256, 0.5959, 0.681);
    constant float3 FILL = float3(0.7926, -0.2265, 0.5661);
    // Step 2's gradient colors, (140, 200, 235) and (8, 40, 90) out of 255.
    constant float3 SKY_BLUE = float3(140.0, 200.0, 235.0) / 255.0;
    constant float3 DEEP_BLUE = float3(8.0, 40.0, 90.0) / 255.0;

    // Nearest hit along the ray. Returns the distance (or a huge number), the
    // surface normal and the color.
    float trace(float3 ro, float3 rd, constant Sphere *spheres, constant Cylinder *cylinders,
                constant Camera &cam, thread float3 &normal, thread float3 &color) {
        float best = 1e30;
        for (uint i = 0; i < cam.sphereCount; i++) {
            float3 c = spheres[i].centerRadius.xyz;
            float r = spheres[i].centerRadius.w;
            float3 oc = ro - c;
            float b = dot(oc, rd);
            float h = b * b - (dot(oc, oc) - r * r);
            if (h < 0) continue;
            float t = -b - sqrt(h);
            if (t > 1e-3 && t < best) {
                best = t;
                normal = normalize(ro + t * rd - c);
                color = spheres[i].color.rgb;
            }
        }
        for (uint i = 0; i < cam.cylinderCount; i++) {
            float3 a = cylinders[i].aRadius.xyz;
            float r = cylinders[i].aRadius.w;
            float3 ba = cylinders[i].b.xyz - a;
            float3 oc = ro - a;
            float baba = dot(ba, ba);
            float bard = dot(ba, rd);
            float baoc = dot(ba, oc);
            float k2 = baba - bard * bard;
            float k1 = baba * dot(oc, rd) - baoc * bard;
            float k0 = baba * dot(oc, oc) - baoc * baoc - r * r * baba;
            float h = k1 * k1 - k2 * k0;
            if (h < 0 || k2 < 1e-8) continue;
            float t = (-k1 - sqrt(h)) / k2;
            float y = baoc + t * bard;
            if (t > 1e-3 && t < best && y > 0 && y < baba) {
                best = t;
                normal = (oc + t * rd - ba * y / baba) / r;
                color = cylinders[i].color.rgb;
            }
        }
        return best;
    }

    float3 shadeRay(float3 ro, float3 rd, float backgroundY, constant Sphere *spheres,
                    constant Cylinder *cylinders, constant Camera &cam) {
        float3 n, base;
        float t = trace(ro, rd, spheres, cylinders, cam, n, base);
        float3 color;
        if (t > 1e29) {
            // The step 2 gradient: sky blue at the top to deep blue at the bottom.
            color = mix(SKY_BLUE, DEEP_BLUE, backgroundY);
        } else {
            float key = max(dot(n, KEY), 0.0);
            float fill = max(dot(n, FILL), 0.0);
            float3 h = normalize(KEY - rd);
            float spec = pow(max(dot(n, h), 0.0), 48.0);
            float rim = pow(1.0 - max(dot(n, -rd), 0.0), 3.0);
            color = base * (0.22 + 0.85 * key + 0.25 * fill) + 0.45 * spec + 0.12 * rim;
        }
        // Glows around charged atoms, from how close the ray passes to them.
        for (uint i = 0; i < cam.sphereCount; i++) {
            float4 g = spheres[i].glow;
            if (g.w <= 0) continue;
            float3 v = spheres[i].centerRadius.xyz - ro;
            float along = dot(v, rd);
            if (along <= 0) continue;
            float d2 = dot(v, v) - along * along;
            color += g.rgb * exp(-d2 / (2.0 * g.w * g.w)) * 0.6;
        }
        return color;
    }

    kernel void render(device uchar4 *pixels [[buffer(0)]],
                       constant Camera &cam [[buffer(1)]],
                       constant Sphere *spheres [[buffer(2)]],
                       constant Cylinder *cylinders [[buffer(3)]],
                       uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= cam.width || gid.y >= cam.height) return;
        float aspect = float(cam.width) / float(cam.height);
        uint k = cam.samplesPerSide;
        float3 sum = float3(0);
        for (uint j = 0; j < k; j++) {
            for (uint i = 0; i < k; i++) {
                float sx = float(gid.x) + (float(i) + 0.5) / float(k);
                float sy = float(gid.y) + (float(j) + 0.5) / float(k);
                float ndcX = sx / float(cam.width) * 2.0 - 1.0;
                float ndcY = 1.0 - sy / float(cam.height) * 2.0;
                float3 rd = normalize(cam.forward.xyz + cam.right.xyz * ndcX * aspect * cam.tanHalfFOV
                                      + cam.up.xyz * ndcY * cam.tanHalfFOV);
                sum += shadeRay(cam.origin.xyz, rd, sy / float(cam.height), spheres, cylinders, cam);
            }
        }
        float3 color = clamp(sum / float(k * k), 0.0, 1.0);
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
    if let device = MTLCreateSystemDefaultDevice() ?? MTLCopyAllDevices().first {
        return device
    }
    throw RenderError.noMetalDevice
}

final class MoleculeRenderer {
    let device: MTLDevice
    private let pipeline: MTLComputePipelineState
    private let queue: MTLCommandQueue

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

    /// Ray-traces the molecule into the top `viewHeight` rows of `frame`
    /// (which is `width` pixels wide). Returns the GPU time in seconds.
    @discardableResult
    func render(_ state: MoleculeState, camera: Camera, into frame: MTLBuffer, width: Int, viewHeight: Int,
                samplesPerSide: Int = 2) throws -> Double {
        let (spheres, cylinders) = sceneGeometry(state)
        var cam = GPUCamera(origin: SIMD4(camera.origin, 1), forward: SIMD4(camera.forward, 0),
                            right: SIMD4(camera.right, 0), up: SIMD4(camera.up, 0),
                            tanHalfFOV: camera.tanHalfFOV, width: UInt32(width), height: UInt32(viewHeight),
                            sphereCount: UInt32(spheres.count), cylinderCount: UInt32(cylinders.count),
                            samplesPerSide: UInt32(samplesPerSide))
        guard let commands = queue.makeCommandBuffer(), let e = commands.makeComputeCommandEncoder() else {
            throw RenderError.gpu("could not create a command encoder")
        }
        // Metal won't bind an empty array, so always send at least one element.
        let s = spheres.isEmpty ? [GPUSphere(centerRadius: .zero, color: .zero, glow: .zero)] : spheres
        let c = cylinders.isEmpty ? [GPUCylinder(aRadius: .zero, b: .zero, color: .zero)] : cylinders
        e.setComputePipelineState(pipeline)
        e.setBuffer(frame, offset: 0, index: 0)
        e.setBytes(&cam, length: MemoryLayout<GPUCamera>.stride, index: 1)
        s.withUnsafeBytes { e.setBytes($0.baseAddress!, length: $0.count, index: 2) }
        c.withUnsafeBytes { e.setBytes($0.baseAddress!, length: $0.count, index: 3) }
        let w = pipeline.threadExecutionWidth
        let group = MTLSize(width: w, height: pipeline.maxTotalThreadsPerThreadgroup / w, depth: 1)
        e.dispatchThreads(MTLSize(width: width, height: viewHeight, depth: 1), threadsPerThreadgroup: group)
        e.endEncoding()
        commands.commit()
        commands.waitUntilCompleted()
        if let error = commands.error { throw RenderError.gpu(error.localizedDescription) }
        return commands.gpuEndTime - commands.gpuStartTime
    }
}
