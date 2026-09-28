// Rendering only what moves. Everything in the frame but the turning disc,
// its tufts and what they shade is step 22's still, so each frame starts from
// that still and only a window round the brush is rendered again — by step
// 20's own kernel, with step 20's own parameters, pixel for pixel what a full
// render would give there (each pixel's samples depend only on its own
// coordinates). The tests render whole frames and check that nothing outside
// the window differs.

import CoreGraphics
import CoreText
import Foundation
import Metal
import simd

/// A block of pixels: columns 0 ..< x1 and rows y0 ..< y1. The kernel reads
/// its column straight from the thread's position, so a window always starts
/// at column 0; rows can start anywhere.
struct PixelWindow: Equatable {
    var x1: Int
    var y0: Int
    var y1: Int
    func contains(_ x: Int, _ y: Int) -> Bool { x < x1 && y >= y0 && y < y1 }
}

/// Render `window` of a frame into `pixels`, leaving every other pixel as it
/// is. Step 20's `renderMouth`, restricted to the window.
func renderWindow(_ window: PixelWindow, width: Int, height: Int, samples: Int, extra: SceneExtra,
                  into pixels: MTLBuffer, aux: MTLBuffer, on device: MTLDevice) throws -> Double {
    let library: MTLLibrary = try makeLibrary(device, mutant: .none, extra: extra)
    let pso: MTLComputePipelineState = try pipeline(device, library, "mouth")
    var teeth: [GPUTooth] = gpuTeeth(placeTeeth())
    guard let toothBuf = device.makeBuffer(bytes: &teeth, length: MemoryLayout<GPUTooth>.stride * teeth.count,
                                           options: .storageModeShared),
          let queue = device.makeCommandQueue()
    else { throw MouthError.gpu("could not allocate buffers") }
    let camera: Camera = stillCamera
    let band: Int = 60
    var gpu: Double = 0
    var row: Int = window.y0
    while row < window.y1 {
        let rows: Int = min(band, window.y1 - row)
        guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else {
            throw MouthError.gpu("could not make a command buffer")
        }
        var params = Params(width: UInt32(width), height: UInt32(height),
                            rowOffset: UInt32(row), samples: UInt32(samples),
                            camPos: SIMD4<Float>(camera.position, 0), camFwd: SIMD4<Float>(camera.forward, 0),
                            camRight: SIMD4<Float>(camera.right, 0), camUp: SIMD4<Float>(camera.up, 0),
                            fillDir: SIMD4<Float>(camera.fill, 0))
        enc.setComputePipelineState(pso)
        enc.setBuffer(pixels, offset: 0, index: 0)
        enc.setBuffer(aux, offset: 0, index: 1)
        enc.setBytes(&params, length: MemoryLayout<Params>.stride, index: 2)
        enc.setBuffer(toothBuf, offset: 0, index: 3)
        let w: Int = pso.threadExecutionWidth
        let group = MTLSize(width: w, height: max(pso.maxTotalThreadsPerThreadgroup / w / 4, 1), depth: 1)
        enc.dispatchThreads(MTLSize(width: window.x1, height: rows, depth: 1), threadsPerThreadgroup: group)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        if let e = cb.error { throw MouthError.gpu(e.localizedDescription) }
        gpu += cb.gpuEndTime - cb.gpuStartTime
        row += rows
    }
    return gpu
}

/// The box round every pixel that differs between two pictures, or nil.
func differingBox(_ a: MouthImage, _ b: MouthImage) -> (x0: Int, y0: Int, x1: Int, y1: Int)? {
    var x0: Int = a.width
    var y0: Int = a.height
    var x1: Int = -1
    var y1: Int = -1
    for y in 0..<a.height {
        for x in 0..<a.width where a.rgba(x, y) != b.rgba(x, y) {
            x0 = min(x0, x); x1 = max(x1, x); y0 = min(y0, y); y1 = max(y1, y)
        }
    }
    return x1 < 0 ? nil : (x0, y0, x1, y1)
}

/// Where the moving brush can change the picture: whole frames at one sample
/// per pixel, at rest and at turns spread across the swing, compared with the
/// frame at rest; the box round every difference, grown by a margin of 3% of
/// the width for the samples between pixel centres and the soft edges of
/// shadows. Measured, not guessed — and the tests check it at full quality.
func movingWindow(brush: OscillatingBrush, width: Int, height: Int, on device: MTLDevice) throws -> PixelWindow {
    let still: MouthImage = try renderMouth(width: width, height: height, samples: 1,
                                            extra: brushExtra(brush.rest), on: device).image
    var x1: Int = 0
    var y0: Int = height
    var y1: Int = 0
    let turns: [Float] = [-1.0, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1.0].map { $0 * swingHalf }
    for spin in turns {
        let b: Brush = try brush.brush(spin: spin, mutant: .none)
        let img: MouthImage = try renderMouth(width: width, height: height, samples: 1,
                                              extra: brushExtra(b), on: device).image
        if let box = differingBox(still, img) {
            x1 = max(x1, box.x1 + 1)
            y0 = min(y0, box.y0)
            y1 = max(y1, box.y1 + 1)
        }
    }
    let margin: Int = max(width * 3 / 100, 4)
    return PixelWindow(x1: min(x1 + margin, width), y0: max(y0 - margin, 0), y1: min(y1 + margin, height))
}

// MARK: - the caption

/// The caption, top left: what the motion is, how much it is slowed, and the
/// one thing a viewer might take for a mistake.
func captionLines() -> [String] {
    let factor: Int = Int((slowdown / 10).rounded()) * 10
    return [
        "Oscillating head slowed ~\(factor)× (real: \(Int(oscillationHz)) back-and-forth turns a second through \(Int(swingDegrees))°)",
        "Every tuft is cast onto the tooth each frame; over the gap between teeth one hangs free",
    ]
}

func drawCaption(_ pixels: MTLBuffer, width: Int, height: Int) {
    guard let space = CGColorSpace(name: CGColorSpace.sRGB),
          let ctx = CGContext(data: pixels.contents(), width: width, height: height,
                              bitsPerComponent: 8, bytesPerRow: width * 4, space: space,
                              bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
    else { return }
    let s: CGFloat = CGFloat(width) / 1920
    let grey = CGColor(srgbRed: 0.33, green: 0.34, blue: 0.36, alpha: 1)
    let font = CTFontCreateWithName("Helvetica Neue" as CFString, 24 * s, nil)
    let attrs: [NSAttributedString.Key: Any] = [
        NSAttributedString.Key(kCTFontAttributeName as String): font,
        NSAttributedString.Key(kCTForegroundColorAttributeName as String): grey,
    ]
    for (i, text) in captionLines().enumerated() {
        let line = CTLineCreateWithAttributedString(NSAttributedString(string: text, attributes: attrs))
        // CoreGraphics counts up from the bottom; the caption sits at the top.
        let fromTop: CGFloat = 52 + 34 * CGFloat(i)
        ctx.textPosition = CGPoint(x: 48 * s, y: CGFloat(height) - fromTop * s)
        CTLineDraw(line, ctx)
    }
}
