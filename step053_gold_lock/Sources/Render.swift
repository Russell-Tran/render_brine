// The gold padlock, rendered as one still: one GPU thread per pixel marching
// rays into a scene of distance functions, as every step since the ocean has.
//
// The new work is the metal. A polished metal has no diffuse colour at all:
// every bit of light leaving it was reflected, mirror-fashion, and how much of
// each colour was reflected is the conductor Fresnel reflectance at that
// angle. The kernel reads that from a table the CPU computed spectrally from
// gold's measured n and k (Optics.swift), 129 angles from grazing to normal,
// and multiplies it into whatever the reflected ray finds: the softboxes, the
// dim backdrop, the table, or the lock itself — up to four bounces, so the
// shackle shows in the face and the face in the shackle.
//
// The shape is simple on purpose, so its distance function is exact outside
// every surface:
//
//   * the body is a rounded box — an exact distance;
//   * the shackle is a half-torus on two capsules — exact, derived below;
//   * the keyway and the ring round the plug are cut with max(d, −cut), which
//     never over-reports;
//   * the pieces are joined with min, which is exact outside a union.
//
// No smooth blends at all, so there is no blend constant to get wrong; every
// piece is 1-Lipschitz and so is the scene.

import CoreGraphics
import Foundation
import ImageIO
import Metal
import UniformTypeIdentifiers
import simd

/// Ray steps trust this fraction of each distance. The field is exact or a
/// lower bound everywhere outside, so 1.0 would be safe in principle; 0.9
/// absorbs float rounding at the keyway's corners.
let stepScale: Float = 0.9

func metal(_ v: SIMD3<Float>) -> String { "float3(\(v.x), \(v.y), \(v.z))" }

/// The kernel's source, with every constant printed into it from Swift, so the
/// numbers the tests read are the numbers the GPU used.
func kernelSource(mutant: Mutant) -> String {
    let lut: [SIMD3<Float>] = fresnelTable(mutant)
    let lutText: String = lut.map { metal($0) }.joined(separator: ",\n        ")
    let k: Softbox = keyLight
    let s: Softbox = stripLight
    return """
    #include <metal_stdlib>
    using namespace metal;

    struct Params { uint width; uint height; uint rowOffset; uint samples;
                    float4 camPos; float4 camFwd; float4 camRight; float4 camUp; };

    constant float STEP_SCALE = \(stepScale);
    constant float FAR = 900.0;
    constant float TAN_HALF_FOV = \(camera.tanHalfFOV);
    constant int MAX_BOUNCES = \(maxBounces);

    constant float3 LOCK_O = \(metal(lockOrigin));
    constant float3 LOCK_U = \(metal(lockU));
    constant float3 LOCK_V = \(metal(lockV));

    constant float BODY_HW = \(bodyWidth / 2);
    constant float BODY_H = \(bodyHeight);
    constant float BODY_T = \(bodyThickness);
    constant float EDGE_R = \(bodyEdgeRadius);
    constant float SH_R = \(shackleRadius);
    constant float SH_LEG = \(shackleLegOffset);
    constant float SH_CY = \(shackleBowCentre);
    constant float SH_BOTTOM = \(bodyHeight - shackleInsertion);
    constant float PLUG_R = \(plugDiameter / 2);
    constant float RING_HW = \(plugClearance / 2);
    constant float RING_D = \(plugRingDepth);
    constant float SLOT_HL = \(keywayLength / 2);
    constant float SLOT_HW = \(keywayWidth / 2);
    constant float SLOT_D = \(keywayDepth);

    constant float3 KEY_C = \(metal(k.centre));
    constant float3 KEY_A = \(metal(k.axisA));
    constant float3 KEY_B = \(metal(k.axisB));
    constant float KEY_TA = \(k.tanA);
    constant float KEY_TB = \(k.tanB);
    constant float KEY_L = \(k.radiance);
    constant float KEY_E = \(k.tableIrradiance(meanFraction: keyMeanFraction));
    constant float KEY_EDGE = \(keyEdgeFraction);
    constant float3 STRIP_C = \(metal(s.centre));
    constant float3 STRIP_A = \(metal(s.axisA));
    constant float3 STRIP_B = \(metal(s.axisB));
    constant float STRIP_TA = \(s.tanA);
    constant float STRIP_TB = \(s.tanB);
    constant float STRIP_L = \(s.radiance);
    constant float STRIP_E = \(s.tableIrradiance());
    constant float BG_HORIZON = \(backdropHorizon);
    constant float BG_ZENITH = \(backdropZenith);
    constant float BG_LOW = \(belowHorizon);
    constant float AMBIENT_E = \(ambientIrradiance);
    constant float TABLE_ALBEDO = \(tableAlbedo);
    constant float TONE_GAIN = \(toneGain);

    constant int LUT_N = \(fresnelTableSize);
    constant float3 FRESNEL[\(fresnelTableSize)] = {
        \(lutText)
    };

    // ---------------------------------------------------------------- shape

    float3 toLock(float3 p) {
        float3 d = p - LOCK_O;
        return float3(dot(d, LOCK_U), dot(d, LOCK_V), d.y);
    }

    float sdRoundBox(float3 p, float3 b, float r) {
        float3 q = abs(p) - b + r;
        return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0) - r;
    }

    float sdRect(float2 p, float2 b) {
        float2 q = abs(p) - b;
        return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0);
    }

    // The body, with the keyway cut into its v = 0 face. q is lock-local:
    // (across the face, along the lock, height off the table).
    float bodySDF(float3 q) {
        float d = sdRoundBox(q - float3(0.0, BODY_H * 0.5, BODY_T * 0.5),
                             float3(BODY_HW, BODY_H * 0.5, BODY_T * 0.5), EDGE_R);
        float2 f = float2(q.x, q.z - BODY_T * 0.5);          // on the keyway face
        // The ring of clearance round the plug: an annulus, extruded into the
        // face to RING_D. max of exact fields is a lower bound inside and out.
        float ring = max(abs(length(f) - PLUG_R) - RING_HW, q.y - RING_D);
        // The keyway: a slot with a step along one side, ending blind.
        float slotA = sdRect(f, float2(SLOT_HL, SLOT_HW));
        float slotB = sdRect(f - float2(0.9, SLOT_HW * 0.9), float2(SLOT_HL - 1.4, SLOT_HW * 0.75));
        float slot = max(min(slotA, slotB), q.y - SLOT_D);
        return max(d, -min(ring, slot));
    }

    // The shackle: a bow of bar radius SH_R whose centreline is a semicircle
    // of radius SH_LEG about (0, SH_CY) in the lock's mid-plane, continued
    // straight down both legs into the body. Above the bow's centre the
    // nearest centreline point is on the semicircle (a circle's nearest point
    // lies along the ray to it, which points up); below, it is on a leg,
    // because the semicircle's nearest points to anything below its centre
    // are its two ends, which are the legs' tops. So the split is exact.
    float shackleSDF(float3 q) {
        float3 s = float3(q.x, q.y - SH_CY, q.z - BODY_T * 0.5);
        if (s.y > 0.0) {
            float2 t = float2(length(s.xy) - SH_LEG, s.z);
            return length(t) - SH_R;
        }
        float y = s.y - clamp(s.y, SH_BOTTOM - SH_CY, 0.0);
        return length(float3(abs(s.x) - SH_LEG, y, s.z)) - SH_R;
    }

    // Material: 1 body, 2 shackle, 3 table.
    float sceneSDF(float3 p, thread int &mat) {
        float3 q = toLock(p);
        float d = bodySDF(q);
        mat = 1;
        float s = shackleSDF(q);
        if (s < d) { d = s; mat = 2; }
        if (p.y < d) { d = p.y; mat = 3; }
        return d;
    }

    float lockSDF(float3 p) {
        float3 q = toLock(p);
        return min(bodySDF(q), shackleSDF(q));
    }

    bool march(float3 ro, float3 rd, thread float &t, thread int &mat) {
        t = 0.0;
        for (int i = 0; i < 400; i++) {
            float3 p = ro + rd * t;
            float d = sceneSDF(p, mat);
            float eps = 0.0015 + t * 0.00002;
            if (d < eps) return true;
            t += d * STEP_SCALE;
            if (t > FAR) break;
        }
        return false;
    }

    float3 sceneNormal(float3 p) {
        const float h = 0.0015;
        int m;
        float3 k0 = float3(1, -1, -1), k1 = float3(-1, -1, 1), k2 = float3(-1, 1, -1), k3 = float3(1, 1, 1);
        return normalize(k0 * sceneSDF(p + k0 * h, m) + k1 * sceneSDF(p + k1 * h, m)
                       + k2 * sceneSDF(p + k2 * h, m) + k3 * sceneSDF(p + k3 * h, m));
    }

    // ---------------------------------------------------------------- light

    // Which light a direction sees: 1 the key softbox, 2 the strip, 0 backdrop.
    int envTag(float3 r) {
        float c = dot(r, KEY_C);
        if (c > 0.0 && abs(dot(r, KEY_A)) < KEY_TA * c && abs(dot(r, KEY_B)) < KEY_TB * c) return 1;
        float c2 = dot(r, STRIP_C);
        if (c2 > 0.0 && abs(dot(r, STRIP_A)) < STRIP_TA * c2 && abs(dot(r, STRIP_B)) < STRIP_TB * c2) return 2;
        return 0;
    }

    // Radiance arriving from a direction, neutral: equal R, G and B.
    float3 environment(float3 r, thread int &tag) {
        tag = envTag(r);
        if (tag == 1) {
            float c = dot(r, KEY_C);
            float a = dot(r, KEY_A) / (KEY_TA * c);
            float b = dot(r, KEY_B) / (KEY_TB * c);
            float g = (1.0 - a * a) * (1.0 - b * b);
            return float3(KEY_L * mix(KEY_EDGE, 1.0, g));
        }
        if (tag == 2) return float3(STRIP_L);
        float L = r.y < 0.0 ? BG_LOW : mix(BG_HORIZON, BG_ZENITH, sqrt(r.y));
        return float3(L);
    }

    float softShadow(float3 ro, float3 rd, float k) {
        float res = 1.0;
        float t = 0.05;
        for (int i = 0; i < 90; i++) {
            float h = lockSDF(ro + rd * t);
            res = min(res, k * h / t);
            if (res < 0.002) return 0.0;
            t += clamp(h, 0.05, 6.0);
            if (t > 160.0) break;
        }
        return clamp(res, 0.0, 1.0);
    }

    // The lock's contact shadow on the table: how much of the sky above a
    // table point the lock blocks, from the distance field.
    float tableOcclusion(float3 p) {
        float occ = 0.0;
        float w = 1.0;
        for (int i = 1; i <= 5; i++) {
            float h = 1.6 * float(i);
            float d = lockSDF(p + float3(0, h, 0));
            occ += (h - clamp(d, 0.0, h)) * w;
            w *= 0.6;
        }
        return clamp(1.0 - occ * 0.09, 0.0, 1.0);
    }

    // The matte table: Lambert, lit by the two softboxes (with soft shadows
    // whose width follows each box's angular size) and the backdrop.
    float3 tableRadiance(float3 p) {
        float3 o = p + float3(0, 0.02, 0);
        float key = softShadow(o, KEY_C, 1.0 / KEY_TB);
        float strip = softShadow(o, STRIP_C, 1.0 / STRIP_TA);
        float E = KEY_E * key + STRIP_E * strip + AMBIENT_E * tableOcclusion(p);
        return float3(TABLE_ALBEDO / M_PI_F * E);
    }

    float3 fresnel(float c) {
        float x = clamp(c, 0.0, 1.0) * float(LUT_N - 1);
        int i = min(int(x), LUT_N - 2);
        return mix(FRESNEL[i], FRESNEL[i + 1], x - float(i));
    }

    // Polished metal: follow the mirror ray, multiplying in the reflectance
    // at each bounce, until it leaves for the sky or lands on the table.
    // firstTag records what the FIRST reflection saw: 0 backdrop, 1 key,
    // 2 strip, 3 table, 4 the lock itself.
    float3 shadeMetal(float3 p, float3 rd, thread int &firstTag) {
        float3 throughput = float3(1.0);
        for (int b = 0; b < MAX_BOUNCES; b++) {
            float3 n = sceneNormal(p);
            float c = dot(n, -rd);
            throughput *= fresnel(c);
            float3 r = reflect(rd, n);
            float t;
            int mat;
            if (!march(p + n * 0.006, r, t, mat)) {
                int tag;
                float3 L = environment(r, tag);
                if (b == 0) firstTag = tag;
                return throughput * L;
            }
            float3 h = p + n * 0.006 + r * t;
            if (mat == 3) {
                if (b == 0) firstTag = 3;
                return throughput * tableRadiance(h);
            }
            if (b == 0) firstTag = 4;
            p = h;
            rd = r;
        }
        return float3(0.0);
    }

    // Step 20's tone curve: a shoulder on luminance, the colour scaled to
    // match, so the hue survives; above white it bleeds towards white.
    float3 toneMap(float3 x) {
        float L = dot(x, float3(0.2126, 0.7152, 0.0722));
        float Lm = L * (1.0 + L / 16.0) / (1.0 + L);
        float3 c = x * (Lm / max(L, 1e-5)) * TONE_GAIN;
        float over = max(max(c.r, max(c.g, c.b)) - 1.0, 0.0);
        c = mix(c, float3(1.0), clamp(over, 0.0, 1.0));
        c = clamp(c, 0.0, 1.0);
        return select(1.055 * pow(c, 1.0 / 2.4) - 0.055, 12.92 * c, c <= 0.0031308);
    }

    float3 cameraRay(float2 pixel, constant Params &P) {
        float aspect = float(P.width) / float(P.height);
        float sx = (2.0 * pixel.x / float(P.width) - 1.0) * aspect * TAN_HALF_FOV;
        float sy = (1.0 - 2.0 * pixel.y / float(P.height)) * TAN_HALF_FOV;
        return normalize(P.camFwd.xyz + sx * P.camRight.xyz + sy * P.camUp.xyz);
    }

    // One camera ray: its display colour, and what it saw.
    float3 trace(float3 ro, float3 rd, thread int &mat, thread int &tag, thread float &cosV, thread float3 &nrm) {
        float t;
        tag = -1;
        cosV = 0.0;
        nrm = float3(0.0);
        if (!march(ro, rd, t, mat)) {
            mat = 0;
            int et;
            return environment(rd, et);
        }
        float3 p = ro + rd * t;
        if (mat == 3) return tableRadiance(p);
        nrm = sceneNormal(p);
        cosV = dot(nrm, -rd);
        return shadeMetal(p, rd, tag);
    }

    // ---------------------------------------------------------------- kernels

    kernel void render(device uchar4 *pixels [[buffer(0)]],
                       device float4 *aux [[buffer(1)]],
                       constant Params &P [[buffer(2)]],
                       uint2 gid [[thread_position_in_grid]]) {
        uint x = gid.x;
        uint y = gid.y + P.rowOffset;
        if (x >= P.width || y >= P.height) return;
        float3 ro = P.camPos.xyz;
        float3 sum = float3(0.0);
        uint S = P.samples;
        int keyCount = 0;
        for (uint sy = 0; sy < S; sy++) {
            for (uint sx = 0; sx < S; sx++) {
                float2 jitter = float2((float(sx) + 0.5) / float(S), (float(sy) + 0.5) / float(S));
                int mat, tag;
                float cv;
                float3 n;
                float3 c = trace(ro, cameraRay(float2(x, y) + jitter, P), mat, tag, cv, n);
                if (tag == 1 && mat != 3 && dot(n, float3(0, 1, 0)) > 0.9999) keyCount++;
                sum += toneMap(c);
            }
        }
        float3 c = sum / float(S * S);
        pixels[y * P.width + x] = uchar4(uchar3(round(clamp(c, 0.0, 1.0) * 255.0)), 255);

        // What the centre of the pixel saw, for the tests: material, cos of the
        // angle to the normal, how many samples mirrored the key softbox off
        // the flat face, and what the first reflection found.
        int mat, tag;
        float cv;
        float3 n;
        float3 rdc = cameraRay(float2(x, y) + 0.5, P);
        trace(ro, rdc, mat, tag, cv, n);
        aux[2 * (y * P.width + x)] = float4(float(mat), cv, float(keyCount) / float(S * S), float(tag));
        // And the radiance the centre's first reflection arrived from, when it
        // was a light: neutral, so one number.
        float Lseen = 0.0;
        if (mat == 1 || mat == 2) {
            int et;
            Lseen = environment(reflect(rdc, n), et).x;
        }
        aux[2 * (y * P.width + x) + 1] = float4(Lseen, n.y, 0.0, 0.0);
    }

    kernel void probe(device const float4 *points [[buffer(0)]],
                      device float2 *out [[buffer(1)]],
                      uint id [[thread_position_in_grid]]) {
        int mat;
        float d = sceneSDF(points[id].xyz, mat);
        out[id] = float2(d, float(mat));
    }
    """
}

// MARK: - running it

enum LockError: Error, CustomStringConvertible {
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

struct Params {
    var width: UInt32
    var height: UInt32
    var rowOffset: UInt32
    var samples: UInt32
    var camPos: SIMD4<Float>
    var camFwd: SIMD4<Float>
    var camRight: SIMD4<Float>
    var camUp: SIMD4<Float>
}

/// The finished picture, and what each pixel's centre saw.
struct LockImage {
    let width: Int
    let height: Int
    let pixels: MTLBuffer
    let aux: MTLBuffer

    func rgba(_ x: Int, _ y: Int) -> SIMD4<UInt8> {
        let p = pixels.contents().advanced(by: (y * width + x) * 4).assumingMemoryBound(to: UInt8.self)
        return SIMD4<UInt8>(p[0], p[1], p[2], p[3])
    }

    /// (material, cos θ to the normal, fraction of samples mirroring the key
    /// off the flat face, first reflection's tag).
    func seen(_ x: Int, _ y: Int) -> SIMD4<Float> {
        aux.contents().assumingMemoryBound(to: SIMD4<Float>.self)[2 * (y * width + x)]
    }

    /// (radiance the centre's first reflection came from if it was the sky,
    /// the normal's y component, 0, 0).
    func seenLight(_ x: Int, _ y: Int) -> SIMD4<Float> {
        aux.contents().assumingMemoryBound(to: SIMD4<Float>.self)[2 * (y * width + x) + 1]
    }
}

func findDevice() throws -> MTLDevice {
    if let d = MTLCreateSystemDefaultDevice() ?? MTLCopyAllDevices().first { return d }
    throw LockError.noMetalDevice
}

func makeLibrary(_ device: MTLDevice, mutant: Mutant) throws -> MTLLibrary {
    let options = MTLCompileOptions()
    // Precise maths, as in every step.
    options.fastMathEnabled = false
    do {
        return try device.makeLibrary(source: kernelSource(mutant: mutant), options: options)
    } catch {
        throw LockError.kernelCompile("\(error)")
    }
}

func pipeline(_ device: MTLDevice, _ library: MTLLibrary, _ name: String) throws -> MTLComputePipelineState {
    guard let f = library.makeFunction(name: name) else { throw LockError.kernelCompile("no kernel \(name)") }
    do { return try device.makeComputePipelineState(function: f) } catch {
        throw LockError.kernelCompile("\(error)")
    }
}

/// Render in horizontal bands, one command buffer each, so no single piece of
/// GPU work runs long enough to trip the watchdog on a shared GPU.
func renderLock(width: Int, height: Int, samples: Int, mutant: Mutant = activeMutant,
                on device: MTLDevice) throws -> (image: LockImage, gpuSeconds: Double) {
    let library = try makeLibrary(device, mutant: mutant)
    let pso = try pipeline(device, library, "render")
    guard let pixels = device.makeBuffer(length: width * height * 4, options: .storageModeShared),
          let aux = device.makeBuffer(length: width * height * 32, options: .storageModeShared),
          let queue = device.makeCommandQueue()
    else { throw LockError.gpu("could not allocate buffers") }

    let band: Int = 40
    var gpu: Double = 0
    var row: Int = 0
    while row < height {
        let rows: Int = min(band, height - row)
        guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else {
            throw LockError.gpu("could not make a command buffer")
        }
        var params = Params(width: UInt32(width), height: UInt32(height),
                            rowOffset: UInt32(row), samples: UInt32(samples),
                            camPos: SIMD4<Float>(camera.position, 0), camFwd: SIMD4<Float>(camera.forward, 0),
                            camRight: SIMD4<Float>(camera.right, 0), camUp: SIMD4<Float>(camera.up, 0))
        enc.setComputePipelineState(pso)
        enc.setBuffer(pixels, offset: 0, index: 0)
        enc.setBuffer(aux, offset: 0, index: 1)
        enc.setBytes(&params, length: MemoryLayout<Params>.stride, index: 2)
        let w: Int = pso.threadExecutionWidth
        let group = MTLSize(width: w, height: max(pso.maxTotalThreadsPerThreadgroup / w / 4, 1), depth: 1)
        enc.dispatchThreads(MTLSize(width: width, height: rows, depth: 1), threadsPerThreadgroup: group)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        if let e = cb.error { throw LockError.gpu(e.localizedDescription) }
        gpu += cb.gpuEndTime - cb.gpuStartTime
        row += rows
    }
    return (LockImage(width: width, height: height, pixels: pixels, aux: aux), gpu)
}

/// The scene's distance and material at arbitrary points, from the same kernel
/// source the render uses — so a test of the distance function is a test of
/// the thing that drew the picture.
func probeScene(_ points: [SIMD3<Float>], on device: MTLDevice) throws -> [SIMD2<Float>] {
    let library = try makeLibrary(device, mutant: .none)
    let pso = try pipeline(device, library, "probe")
    var pts: [SIMD4<Float>] = points.map { SIMD4<Float>($0.x, $0.y, $0.z, 0) }
    guard let pb = device.makeBuffer(bytes: &pts, length: 16 * pts.count, options: .storageModeShared),
          let ob = device.makeBuffer(length: 8 * pts.count, options: .storageModeShared),
          let queue = device.makeCommandQueue(), let cb = queue.makeCommandBuffer(),
          let enc = cb.makeComputeCommandEncoder()
    else { throw LockError.gpu("could not set up the probe") }
    enc.setComputePipelineState(pso)
    enc.setBuffer(pb, offset: 0, index: 0)
    enc.setBuffer(ob, offset: 0, index: 1)
    enc.dispatchThreads(MTLSize(width: pts.count, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: min(pso.maxTotalThreadsPerThreadgroup, 256), height: 1, depth: 1))
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    if let e = cb.error { throw LockError.gpu(e.localizedDescription) }
    let out = ob.contents().assumingMemoryBound(to: SIMD2<Float>.self)
    return (0..<pts.count).map { out[$0] }
}

func savePNG(_ image: LockImage, to url: URL) throws {
    try? FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    guard let provider = CGDataProvider(dataInfo: nil, data: image.pixels.contents(), size: image.width * image.height * 4,
                                        releaseData: { _, _, _ in }),
          let space = CGColorSpace(name: CGColorSpace.sRGB),
          let cg = CGImage(width: image.width, height: image.height, bitsPerComponent: 8, bitsPerPixel: 32,
                           bytesPerRow: image.width * 4, space: space,
                           bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                           provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent),
          let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)
    else { throw LockError.png("could not set up the encoder for \(url.path)") }
    CGImageDestinationAddImage(dest, cg, nil)
    guard CGImageDestinationFinalize(dest) else { throw LockError.png("could not write \(url.path)") }
}
