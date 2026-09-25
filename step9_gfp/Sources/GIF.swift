// A small GIF encoder, written because ImageIO's was too big for this scene.
//
// ImageIO quantizes every frame to its own palette and stores every frame
// whole: ~41 MB for this loop. But between two frames of a slowly turning
// molecule only ~7% of the pixels change. So this writer:
//   1. picks one palette for the whole GIF (median cut over sample frames),
//   2. maps each frame to it on the GPU (nearest color, no extra dithering),
//   3. stores only the pixels that changed since the last frame, cropped to
//      the box around them, with every unchanged pixel "transparent" so the
//      previous frame shows through (GIF disposal method 1),
//   4. compresses each frame with GIF's LZW.
// GIF format: https://www.w3.org/Graphics/GIF/spec-gif89a.txt

import Foundation
import Metal

typealias RGB = SIMD3<UInt8>

/// A palette of up to 255 colors chosen by median cut: split the box of colors
/// along its widest channel at the median, again and again, then average each
/// box. (Index 255 is kept free to mean "transparent".)
func medianCutPalette(_ samples: [RGB], count: Int) -> [RGB] {
    var boxes: [[RGB]] = [samples]
    func spread(_ box: [RGB]) -> (channel: Int, range: Int) {
        var best = (channel: 0, range: -1)
        for ch in 0..<3 {
            var lo = 255, hi = 0
            for c in box {
                let u: UInt8 = c[ch]
                let v = Int(u)
                lo = min(lo, v)
                hi = max(hi, v)
            }
            if hi - lo > best.range { best = (ch, hi - lo) }
        }
        return best
    }
    while boxes.count < count {
        // Split the box with the widest spread (weighted by how many pixels it holds).
        var pick = -1, score = 0
        for (i, box) in boxes.enumerated() where box.count > 1 {
            let s = spread(box).range * Int(Double(box.count).squareRoot())
            if s > score { pick = i; score = s }
        }
        if pick < 0 { break }
        let box = boxes.remove(at: pick)
        let ch = spread(box).channel
        let sorted = box.sorted { $0[ch] < $1[ch] }
        let half = sorted.count / 2
        boxes.append(Array(sorted[..<half]))
        boxes.append(Array(sorted[half...]))
    }
    return boxes.map { box -> RGB in
        var sum = SIMD3<Int>(0, 0, 0)
        for c in box { sum &+= SIMD3<Int>(Int(c.x), Int(c.y), Int(c.z)) }
        let n = max(box.count, 1)
        return RGB(UInt8(sum.x / n), UInt8(sum.y / n), UInt8(sum.z / n))
    }
}

/// Every `step`th pixel of an RGBA frame, as colors.
func samplePixels(_ frame: MTLBuffer, pixels: Int, step: Int) -> [RGB] {
    let p = frame.contents().assumingMemoryBound(to: UInt8.self)
    var out: [RGB] = []
    out.reserveCapacity(pixels / step + 1)
    for i in stride(from: 0, to: pixels, by: step) {
        out.append(RGB(p[i * 4], p[i * 4 + 1], p[i * 4 + 2]))
    }
    return out
}

let quantizeKernelSource = """
    #include <metal_stdlib>
    using namespace metal;

    // Each pixel becomes the index of the nearest palette color.
    kernel void quantize(device const uchar4 *pixels [[buffer(0)]],
                         constant float4 *palette [[buffer(1)]],
                         constant uint &count [[buffer(2)]],
                         device uchar *indices [[buffer(3)]],
                         uint i [[thread_position_in_grid]]) {
        float3 c = float3(pixels[i].rgb);
        uint best = 0;
        float bestD = 1e30;
        for (uint k = 0; k < count; k++) {
            float3 d = c - palette[k].rgb;
            float dd = dot(d, d);
            if (dd < bestD) { bestD = dd; best = k; }
        }
        indices[i] = uchar(best);
    }
    """

/// Maps frames to palette indices on the GPU.
final class Quantizer {
    let palette: [RGB]
    private let pipeline: MTLComputePipelineState
    private let queue: MTLCommandQueue
    private let paletteBuffer: MTLBuffer
    private let indexBuffer: MTLBuffer
    private let pixels: Int

    init(device: MTLDevice, palette: [RGB], pixels: Int) throws {
        precondition(palette.count >= 1 && palette.count <= 255)
        self.palette = palette
        self.pixels = pixels
        let library = try device.makeLibrary(source: quantizeKernelSource, options: nil)
        guard let function = library.makeFunction(name: "quantize"), let queue = device.makeCommandQueue() else {
            throw RenderError.kernelCompile("no quantize kernel")
        }
        pipeline = try device.makeComputePipelineState(function: function)
        self.queue = queue
        let floats = palette.map { SIMD4<Float>(Float($0.x), Float($0.y), Float($0.z), 0) }
        guard let pb = device.makeBuffer(bytes: floats, length: floats.count * MemoryLayout<SIMD4<Float>>.stride,
                                         options: .storageModeShared),
              let ib = device.makeBuffer(length: pixels, options: .storageModeShared) else {
            throw RenderError.gpu("could not allocate quantizer buffers")
        }
        paletteBuffer = pb
        indexBuffer = ib
    }

    func indices(of frame: MTLBuffer) throws -> [UInt8] {
        guard let commands = queue.makeCommandBuffer(), let e = commands.makeComputeCommandEncoder() else {
            throw RenderError.gpu("could not create a command encoder")
        }
        var count = UInt32(palette.count)
        e.setComputePipelineState(pipeline)
        e.setBuffer(frame, offset: 0, index: 0)
        e.setBuffer(paletteBuffer, offset: 0, index: 1)
        e.setBytes(&count, length: 4, index: 2)
        e.setBuffer(indexBuffer, offset: 0, index: 3)
        e.dispatchThreads(MTLSize(width: pixels, height: 1, depth: 1),
                          threadsPerThreadgroup: MTLSize(width: pipeline.threadExecutionWidth, height: 1, depth: 1))
        e.endEncoding()
        commands.commit()
        commands.waitUntilCompleted()
        if let error = commands.error { throw RenderError.gpu(error.localizedDescription) }
        return Array(UnsafeBufferPointer(start: indexBuffer.contents().assumingMemoryBound(to: UInt8.self), count: pixels))
    }
}

/// GIF's LZW compression (the classic variable-width scheme, 8-bit pixels,
/// codes of 9 to 12 bits, a clear code when the table fills up).
func lzwEncode(_ data: [UInt8]) -> [UInt8] {
    let minCodeSize = 8
    let clearCode = 1 << minCodeSize, endCode = clearCode + 1
    var bits = minCodeSize + 1
    var maxCode = (1 << bits) - 1
    var next = clearCode + 2
    var clearing = false
    var table = [Int16](repeating: -1, count: 4096 * 256)   // (prefix code, byte) → code
    var out: [UInt8] = []
    out.reserveCapacity(data.count / 2)
    var acc: UInt32 = 0, accBits = 0

    func emit(_ code: Int) {
        acc |= UInt32(code) << UInt32(accBits)
        accBits += bits
        while accBits >= 8 {
            out.append(UInt8(acc & 0xff))
            acc >>= 8
            accBits -= 8
        }
        // Widen the codes once the table outgrows them (or reset after a clear).
        if next > maxCode || clearing {
            if clearing {
                bits = minCodeSize + 1
                clearing = false
            } else {
                bits += 1
            }
            maxCode = bits == 12 ? 4096 : (1 << bits) - 1
        }
    }

    emit(clearCode)
    guard let first = data.first else { emit(endCode); return out }
    var prefix = Int(first)
    table.withUnsafeMutableBufferPointer { t in
        for i in 1..<data.count {
            let c = Int(data[i])
            let key = prefix << 8 | c
            let found = Int(t[key])
            if found >= 0 { prefix = found; continue }
            emit(prefix)
            prefix = c
            if next < 4096 {
                t[key] = Int16(next)
                next += 1
            } else {
                for k in 0..<t.count { t[k] = -1 }
                next = clearCode + 2
                clearing = true
                emit(clearCode)
            }
        }
    }
    emit(prefix)
    emit(endCode)
    if accBits > 0 { out.append(UInt8(acc & 0xff)) }
    return out
}

/// Writes an animated GIF that loops forever, one palette-index frame at a time.
final class GIFWriter {
    let width: Int
    let height: Int
    private var bytes: [UInt8] = []
    private var previous: [UInt8]?
    private let url: URL
    private let delay: Int                  // hundredths of a second
    static let transparent: UInt8 = 255
    private(set) var frameCount = 0

    init(url: URL, width: Int, height: Int, palette: [RGB], delayCentiseconds: Int) {
        self.url = url
        self.width = width
        self.height = height
        self.delay = delayCentiseconds
        func u16(_ v: Int) -> [UInt8] { [UInt8(v & 0xff), UInt8(v >> 8)] }
        bytes += Array("GIF89a".utf8)
        bytes += u16(width) + u16(height)
        bytes += [0xF7, 0, 0]               // a global color table of 256 entries
        for i in 0..<256 {
            let c = i < palette.count ? palette[i] : RGB(0, 0, 0)
            bytes += [c.x, c.y, c.z]
        }
        // Loop forever (the NETSCAPE2.0 extension).
        bytes += [0x21, 0xFF, 0x0B] + Array("NETSCAPE2.0".utf8) + [0x03, 0x01, 0x00, 0x00, 0x00]
    }

    /// Adds a frame of palette indices (width × height, top row first).
    func add(_ indices: [UInt8]) {
        precondition(indices.count == width * height)
        var x0 = 0, y0 = 0, x1 = width - 1, y1 = height - 1
        var pixels = indices
        if let prev = previous {
            // Keep only what changed; everything else shows the frame before.
            x0 = width; y0 = height; x1 = -1; y1 = -1
            for y in 0..<height {
                let row = y * width
                for x in 0..<width where indices[row + x] != prev[row + x] {
                    x0 = min(x0, x); x1 = max(x1, x); y0 = min(y0, y); y1 = max(y1, y)
                }
            }
            if x1 < 0 { x0 = 0; y0 = 0; x1 = 0; y1 = 0 }   // nothing changed: one transparent pixel
            var crop: [UInt8] = []
            crop.reserveCapacity((x1 - x0 + 1) * (y1 - y0 + 1))
            for y in y0...y1 {
                let row = y * width
                for x in x0...x1 {
                    crop.append(indices[row + x] == prev[row + x] ? GIFWriter.transparent : indices[row + x])
                }
            }
            pixels = crop
        }
        previous = indices
        func u16(_ v: Int) -> [UInt8] { [UInt8(v & 0xff), UInt8(v >> 8)] }
        // Graphic control: disposal 1 (leave in place), transparent index 255, delay.
        bytes += [0x21, 0xF9, 0x04, 0x05] + u16(delay) + [GIFWriter.transparent, 0x00]
        bytes += [0x2C] + u16(x0) + u16(y0) + u16(x1 - x0 + 1) + u16(y1 - y0 + 1) + [0x00]
        bytes.append(8)                     // LZW minimum code size
        let data = lzwEncode(pixels)
        var i = 0
        while i < data.count {
            let n = min(255, data.count - i)
            bytes.append(UInt8(n))
            bytes += data[i..<(i + n)]
            i += n
        }
        bytes.append(0)
        frameCount += 1
    }

    func finish() throws {
        bytes.append(0x3B)
        try? FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data(bytes).write(to: url)
    }
}
