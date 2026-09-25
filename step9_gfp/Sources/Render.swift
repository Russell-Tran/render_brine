// A GPU ray tracer for space-filling molecules (from step 8, which brought the
// scene into MTLBuffers so it could outgrow setBytes' 4 KB).
//
// New in step 9: AMBIENT OCCLUSION. Every earlier step cast one ray per sample
// and stopped. Here each ray that hits the molecule casts a further spray of
// short probe rays into the hemisphere above the surface, and counts how many
// escape. Few escapes means a deep crevice, so the point is darkened.
//
// Without it a space-filling protein is a featureless coloured blob: with no
// sticks to read and no cast shadows, every bit of its shape is carried by the
// darkening in its dimples and channels.
//
// Keeping the loop from crawling
// ------------------------------
// Occlusion sampled afresh in screen space every frame would shimmer horribly. So
// the whole render happens in the molecule's OWN coordinates, which never
// move: the camera orbits it and the two light directions are rotated into
// object space for each frame. The image is identical to turning the molecule
// under a fixed camera and light, but a given point on the surface now gets
// the same probe directions, and so the same shade, in every frame.
//
// Reference for molecular ambient occlusion: Tarini, Cignoni & Montani,
// "Ambient occlusion and edge cueing for enhancing real-time molecular
// visualization", IEEE TVCG 12:1237 (2006).

import Foundation
import Metal
import simd

// These structs must match the kernel. SIMD4<Float> keeps the Swift and Metal
// layouts identical (16-byte aligned).
struct GPUSphere {
    var centerRadius: SIMD4<Float>
    var color: SIMD4<Float>
    var glow: SIMD4<Float>        // unused here, kept so the struct matches step 8's
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
    var keyLight: SIMD4<Float>    // in object space, so the light looks fixed while the molecule turns
    var fillLight: SIMD4<Float>
    var tanHalfFOV: Float
    var width: UInt32             // image width in pixels
    var height: UInt32            // height of the molecule view in pixels
    var sphereCount: UInt32
    var cylinderCount: UInt32
    var samplesPerSide: UInt32
    var aoProbes: UInt32          // 0 turns ambient occlusion off
    var aoDistance: Float         // how far a probe ray looks, in ångströms
    var aoStrength: Float         // 0 none, 1 full
    var aoOnly: UInt32            // debug: show the occlusion term by itself
    var aoContrast: Float         // raises the occlusion to this power, to use more of the range
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

let bondRadius: Float = 0.26

/// The two lights, in world terms. The renderer turns these into the
/// molecule's own coordinates each frame, so they stay put while it spins.
let keyLightWorld = simd_normalize(SIMD3<Float>(-0.45, 0.62, 0.72))
let fillLightWorld = simd_normalize(SIMD3<Float>(0.80, -0.23, 0.57))

let raytraceKernelSource = """
    #include <metal_stdlib>
    using namespace metal;

    struct Sphere { float4 centerRadius; float4 color; float4 glow; };
    struct Cylinder { float4 aRadius; float4 b; float4 color; };
    struct Camera {
        float4 origin; float4 forward; float4 right; float4 up;
        float4 keyLight; float4 fillLight;
        float tanHalfFOV; uint width; uint height; uint sphereCount; uint cylinderCount; uint samplesPerSide;
        uint aoProbes; float aoDistance; float aoStrength; uint aoOnly; float aoContrast;
    };

    // Step 2's gradient colors, (140, 200, 235) and (8, 40, 90) out of 255.
    constant float3 SKY_BLUE = float3(140.0, 200.0, 235.0) / 255.0;
    constant float3 DEEP_BLUE = float3(8.0, 40.0, 90.0) / 255.0;

    // A faint, fixed ordered dither (a 4 x 4 Bayer matrix). GIFs have at most
    // 256 colors, so smooth gradients band; this breaks the bands up, and a
    // repeating pattern compresses far better than noise (step 8).
    constant float BAYER[16] = { 0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5 };
    float3 dither(uint2 gid) {
        float n = (BAYER[(gid.y & 3u) * 4u + (gid.x & 3u)] + 0.5) / 16.0 - 0.5;
        return float3(n * 3.0 / 255.0);
    }

    uint hash(uint x) {
        x ^= x >> 16; x *= 0x7feb352du;
        x ^= x >> 15; x *= 0x846ca68bu;
        x ^= x >> 16;
        return x;
    }

    // Nearest hit along the ray: distance (or a huge number), normal and color.
    float trace(float3 ro, float3 rd, device const Sphere *spheres, device const Cylinder *cylinders,
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

    // Is anything within `maxDist` along this ray? Probe rays only need a yes
    // or no, so this stops at the first hit instead of finding the nearest.
    bool occluded(float3 ro, float3 rd, float maxDist, device const Sphere *spheres, uint count) {
        for (uint i = 0; i < count; i++) {
            float3 c = spheres[i].centerRadius.xyz;
            float r = spheres[i].centerRadius.w;
            float3 oc = ro - c;
            float b = dot(oc, rd);
            // Skip anything already behind us or too far to matter.
            if (b > 0.0 && dot(oc, oc) > r * r) continue;
            float h = b * b - (dot(oc, oc) - r * r);
            if (h < 0) continue;
            float t = -b - sqrt(h);
            if (t > 1e-3 && t < maxDist) return true;
        }
        return false;
    }

    // A cosine-weighted direction in the hemisphere about `n`. Deterministic
    // in i and n, with a twist taken from the point's own position, so a given
    // spot on the molecule always gets the same spray of probes.
    float3 hemisphere(uint i, uint n_total, float twist, float3 n) {
        float u = (float(i) + 0.5) / float(n_total);
        float r = sqrt(u);
        float phi = 6.2831853 * (fract(float(i) * 0.6180339887 + twist));
        // A frame around n. The branch keeps the cross product well conditioned.
        float3 t = normalize(abs(n.z) < 0.9 ? cross(n, float3(0.0, 0.0, 1.0)) : cross(n, float3(1.0, 0.0, 0.0)));
        float3 b = cross(n, t);
        return t * (r * cos(phi)) + b * (r * sin(phi)) + n * sqrt(max(0.0, 1.0 - u));
    }

    // The fraction of the hemisphere above `p` that is open sky.
    //
    // The probe pattern is fixed in the frame built from the surface normal,
    // with no per-point jitter. Jitter was the first thing I tried and it
    // produced exactly what it sounds like: neighbouring points drew different
    // directions, so the surface came out speckled. Because the frame turns
    // smoothly as the normal does, a fixed pattern varies smoothly too, and it
    // is if anything steadier from frame to frame.
    float ambientOcclusion(float3 p, float3 n, device const Sphere *spheres, constant Camera &cam) {
        if (cam.aoProbes == 0) return 1.0;
        float3 start = p + n * 0.02;
        uint open = 0;
        for (uint i = 0; i < cam.aoProbes; i++) {
            float3 dir = hemisphere(i, cam.aoProbes, 0.0, n);
            if (!occluded(start, dir, cam.aoDistance, spheres, cam.sphereCount)) open++;
        }
        // Deepen the midtones: raw openness clusters near 1 on a convex
        // surface, which wastes most of the range.
        float ao = float(open) / float(cam.aoProbes);
        return pow(ao, cam.aoContrast);
    }

    float3 shadeRay(float3 ro, float3 rd, float backgroundY, device const Sphere *spheres,
                    device const Cylinder *cylinders, constant Camera &cam, thread bool &hit) {
        float3 n, base;
        float t = trace(ro, rd, spheres, cylinders, cam, n, base);
        hit = t < 1e29;
        if (!hit) {
            // The step 2 gradient: sky blue at the top to deep blue at the bottom.
            return mix(SKY_BLUE, DEEP_BLUE, backgroundY);
        }
        float3 p = ro + t * rd;
        float ao = ambientOcclusion(p, n, spheres, cam);
        ao = mix(1.0, ao, cam.aoStrength);
        if (cam.aoOnly != 0) return float3(ao);

        float key = max(dot(n, cam.keyLight.xyz), 0.0);
        float fill = max(dot(n, cam.fillLight.xyz), 0.0);
        float3 h = normalize(cam.keyLight.xyz - rd);
        float spec = pow(max(dot(n, h), 0.0), 24.0);
        // Occlusion drives the ambient term outright and damps the two lights,
        // which is what makes crevices read as depth rather than as dark paint.
        float3 color = base * (0.14 + 0.66 * ao + 0.58 * key * mix(0.30, 1.0, ao) + 0.16 * fill * ao);
        // Kept low on purpose: a bright highlight on every one of 1,771 spheres
        // turns the molecule into a heap of glossy beads.
        color += 0.07 * spec * ao;
        return color;
    }

    kernel void render(device uchar4 *pixels [[buffer(0)]],
                       constant Camera &cam [[buffer(1)]],
                       device const Sphere *spheres [[buffer(2)]],
                       device const Cylinder *cylinders [[buffer(3)]],
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
                sum += shadeRay(cam.origin.xyz, rd, sy / float(cam.height), spheres, cylinders, cam, hit);
                misses += hit ? 0.0 : 1.0;
            }
        }
        // Dither only the background: it is there to hide banding in the
        // gradient, and on the molecule it would be churn the GIF must store.
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
    if let device = MTLCreateSystemDefaultDevice() ?? MTLCopyAllDevices().first {
        return device
    }
    throw RenderError.noMetalDevice
}

/// How occlusion is sampled. Defaults chosen by measurement, see the results page.
struct AOSettings {
    var probes: Int = 12
    var distance: Float = 8.0
    var strength: Float = 1.0
    var only: Bool = false
    var contrast: Float = 1.7
    static let off = AOSettings(probes: 0, distance: 0, strength: 0, only: false, contrast: 1)
}

final class MoleculeRenderer {
    let device: MTLDevice
    private let pipeline: MTLComputePipelineState
    private let queue: MTLCommandQueue
    private var sphereBuffer: MTLBuffer?
    private var cylinderBuffer: MTLBuffer?

    init(device: MTLDevice) throws {
        self.device = device
        do {
            let options = MTLCompileOptions()
            // Precise math, as in steps 3 to 5. (Newer SDKs call this
            // mathMode = .safe; this older name works on every SDK, and this
            // step has to build on the laptop's older toolchain too.)
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

    /// Ray-traces into the top `viewHeight` rows of `frame`. Returns GPU seconds.
    ///
    /// The scene is in the molecule's own coordinates and never moves; the
    /// camera orbits instead, and the lights are rotated to match, so ambient
    /// occlusion is the same from frame to frame.
    @discardableResult
    func render(spheres: [GPUSphere], cylinders: [GPUCylinder], camera: Camera, into frame: MTLBuffer,
                width: Int, viewHeight: Int, samplesPerSide: Int = 2,
                ao: AOSettings = AOSettings(), lightYaw: Float = 0) throws -> Double {
        // Turn the world-space lights into object space by unwinding the same
        // yaw the camera was given.
        let keyDir = rotateAboutY(keyLightWorld, degrees: -lightYaw)
        let fillDir = rotateAboutY(fillLightWorld, degrees: -lightYaw)
        var cam = GPUCamera(origin: SIMD4(camera.origin, 1), forward: SIMD4(camera.forward, 0),
                            right: SIMD4(camera.right, 0), up: SIMD4(camera.up, 0),
                            keyLight: SIMD4(keyDir, 0), fillLight: SIMD4(fillDir, 0),
                            tanHalfFOV: camera.tanHalfFOV, width: UInt32(width), height: UInt32(viewHeight),
                            sphereCount: UInt32(spheres.count), cylinderCount: UInt32(cylinders.count),
                            samplesPerSide: UInt32(samplesPerSide),
                            aoProbes: UInt32(ao.probes), aoDistance: ao.distance,
                            aoStrength: ao.strength, aoOnly: ao.only ? 1 : 0,
                            aoContrast: ao.contrast)
        guard let commands = queue.makeCommandBuffer(), let e = commands.makeComputeCommandEncoder() else {
            throw RenderError.gpu("could not create a command encoder")
        }
        // Metal won't bind an empty buffer, so always send at least one element.
        let s = spheres.isEmpty ? [GPUSphere(centerRadius: .zero, color: .zero, glow: .zero)] : spheres
        let c = cylinders.isEmpty ? [GPUCylinder(aRadius: .zero, b: .zero, color: .zero)] : cylinders
        sphereBuffer = try fill(sphereBuffer, with: s)
        cylinderBuffer = try fill(cylinderBuffer, with: c)
        e.setComputePipelineState(pipeline)
        e.setBuffer(frame, offset: 0, index: 0)
        e.setBytes(&cam, length: MemoryLayout<GPUCamera>.stride, index: 1)
        e.setBuffer(sphereBuffer, offset: 0, index: 2)
        e.setBuffer(cylinderBuffer, offset: 0, index: 3)
        let w = pipeline.threadExecutionWidth
        let group = MTLSize(width: w, height: pipeline.maxTotalThreadsPerThreadgroup / w, depth: 1)
        e.dispatchThreads(MTLSize(width: width, height: viewHeight, depth: 1), threadsPerThreadgroup: group)
        e.endEncoding()
        commands.commit()
        commands.waitUntilCompleted()
        if let error = commands.error { throw RenderError.gpu(error.localizedDescription) }
        return commands.gpuEndTime - commands.gpuStartTime
    }

    /// Copies `items` into a shared buffer, reusing `buffer` when it's big
    /// enough. (Safe to overwrite: each frame waits for the GPU to finish.)
    private func fill<T>(_ buffer: MTLBuffer?, with items: [T]) throws -> MTLBuffer {
        let bytes = items.count * MemoryLayout<T>.stride
        var target = buffer
        if target == nil || target!.length < bytes {
            target = device.makeBuffer(length: bytes, options: .storageModeShared)
        }
        guard let out = target else { throw RenderError.gpu("could not allocate a scene buffer") }
        items.withUnsafeBytes { out.contents().copyMemory(from: $0.baseAddress!, byteCount: bytes) }
        return out
    }
}

/// Turns a vector `degrees` about the y axis.
func rotateAboutY(_ v: SIMD3<Float>, degrees: Float) -> SIMD3<Float> {
    let a = radians(degrees)
    let c = cos(a), s = sin(a)
    let x: Float = v.x * c + v.z * s
    let z: Float = -v.x * s + v.z * c
    return SIMD3(x, v.y, z)
}
