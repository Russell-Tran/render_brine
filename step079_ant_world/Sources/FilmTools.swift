// Step 79: measuring the film — how deep ants ever go into one another in 3D
// (the crowding bound), whether the pheromone's formation shows on screen
// (step 29's rule), and whether frames drawn on another machine match the
// reference frames drawn here.

import CoreGraphics
import Foundation
import ImageIO
import Metal
import UniformTypeIdentifiers
import simd

// MARK: - interpenetration, in 3D, from the drawn distance functions

/// How far two ants' drawn shapes go into each other: for each pair whose
/// bounding spheres meet, a grid of points (`spacing` mm) over the box the two
/// spheres share, from the ground to 1.4 mm up (nothing of an ant is higher),
/// and at each point both ants' own distances; where both are negative the
/// point is inside both, and min(−dA, −dB) is how deep. Returns the pairs
/// that touch and the deepest point over them.
struct FilmOverlap {
    var pairsChecked: Int = 0
    var pairsInside: Int = 0
    var deepest: Float = 0
    /// Ids of each pair found inside one another, with its depth.
    var inside: [(Int, Int, Float)] = []
}

/// The crowding bound the tests hold the film to. NOT "no interpenetration":
/// the simulation lets ants crowd at the food and squeeze past one another
/// (lib/ant/v1 can neither back up nor turn on the spot, so two ants that meet
/// head on could otherwise wait for ever; making them queue at the pile was
/// tried and made it worse — see the report). Measured over the whole film
/// (`render overlap 1`): some ants' drawn shapes are inside another's in 1212
/// of 1800 frames (5289 pair-frames, 4340 of them at the food), at most
/// 0.601 mm deep. The bound on the test's sample (every 60th frame) is that,
/// with a margin: a regression guard, stated as what it is.
let filmOverlapFrameShareBound: Double = 0.80
let filmOverlapDepthBound: Float = 0.65

func filmOverlap(_ r: WorldRenderer, ants: [(id: Int, posed: AntV1.Posed)], spacing: Float = 0.08) throws -> FilmOverlap {
    var out = FilmOverlap()
    for i in 0..<ants.count {
        for j in (i + 1)..<ants.count {
            let a: AntV1.Posed = ants[i].posed
            let b: AntV1.Posed = ants[j].posed
            let (ca, ra) = a.bound
            let (cb, rb) = b.bound
            if simd_distance(ca, cb) > ra + rb { continue }
            out.pairsChecked += 1
            let lo: SIMD3<Float> = simd_max(ca - SIMD3<Float>(repeating: ra), cb - SIMD3<Float>(repeating: rb))
            let hi: SIMD3<Float> = simd_min(ca + SIMD3<Float>(repeating: ra), cb + SIMD3<Float>(repeating: rb))
            let y0: Float = max(lo.y, 0)
            let y1: Float = min(hi.y, 1.4)
            if hi.x <= lo.x || hi.z <= lo.z || y1 <= y0 { continue }
            var pts: [SIMD3<Float>] = []
            var x: Float = lo.x
            while x <= hi.x {
                var z: Float = lo.z
                while z <= hi.z {
                    var y: Float = y0
                    while y <= y1 {
                        pts.append(SIMD3<Float>(x, y, z))
                        y += spacing
                    }
                    z += spacing
                }
                x += spacing
            }
            let da: [SIMD4<Float>] = try r.probe(pts, ants: [a])
            let db: [SIMD4<Float>] = try r.probe(pts, ants: [b])
            var depth: Float = 0
            for k in 0..<pts.count where da[k].z < 0 && db[k].z < 0 {
                depth = max(depth, min(-da[k].z, -db[k].z))
            }
            if depth > 0 {
                out.pairsInside += 1
                out.deepest = max(out.deepest, depth)
                out.inside.append((ants[i].id, ants[j].id, depth))
            }
        }
    }
    return out
}

// MARK: - the visibility rule: the trail's formation shows on screen

/// Pixels of the card (the centre sample saw ground, material 1, in both)
/// whose blue lead over red grew by more than `threshold` levels from frame
/// A to frame B: pheromone that appeared on screen. Frames as RGBA8 with
/// their aux (hit, material, …) per pixel.
func filmTrailPixels(_ a: [UInt8], _ auxA: [SIMD4<Float>], _ b: [UInt8], _ auxB: [SIMD4<Float>],
                     threshold: Int = 12) -> Int {
    var n: Int = 0
    for k in 0..<auxA.count {
        let groundA: Bool = auxA[k].y > 0.5 && auxA[k].y < 1.5
        let groundB: Bool = auxB[k].y > 0.5 && auxB[k].y < 1.5
        if !(groundA && groundB) { continue }
        let leadA: Int = Int(a[4 * k + 2]) - Int(a[4 * k])
        let leadB: Int = Int(b[4 * k + 2]) - Int(b[4 * k])
        if leadB - leadA > threshold { n += 1 }
    }
    return n
}

func filmAux(_ image: WorldImage) -> [SIMD4<Float>] {
    let p = image.aux.contents().assumingMemoryBound(to: SIMD4<Float>.self)
    return Array(UnsafeBufferPointer(start: p, count: image.width * image.height))
}

// MARK: - reference frames, for other machines

/// Film frames kept as references: across the film (wide, the move, the
/// trail forming, the close real-time end), at a small size so the committed
/// bytes stay modest.
let filmReferenceFrames: [Int] = [150, 700, 1431, 1799]
let filmReferenceWidth: Int = 480
let filmReferenceHeight: Int = 270
let filmReferenceSamples: Int = 2
let filmReferenceDirectory: String = "records/refs"
/// What another machine's GPU may differ by and still count as the same
/// film (the mini's proposal): mean |difference| below 0.5/255 and max below
/// 8/255 in every channel.
let filmReferenceMeanTolerance: Double = 0.5
let filmReferenceMaxTolerance: Int = 8

struct FilmChannelDifference {
    var maxAbs: [Int] = [0, 0, 0]
    var meanAbs: [Double] = [0, 0, 0]
    var within: Bool {
        let meanOK: Bool = meanAbs.allSatisfy { $0 < filmReferenceMeanTolerance }
        let maxOK: Bool = maxAbs.allSatisfy { $0 < filmReferenceMaxTolerance }
        return meanOK && maxOK
    }
}

func filmCompare(_ a: [UInt8], _ b: [UInt8]) -> FilmChannelDifference {
    var d = FilmChannelDifference()
    var sums: [Double] = [0, 0, 0]
    let n: Int = min(a.count, b.count) / 4
    for k in 0..<n {
        for c in 0..<3 {
            let e: Int = abs(Int(a[4 * k + c]) - Int(b[4 * k + c]))
            d.maxAbs[c] = max(d.maxAbs[c], e)
            sums[c] += Double(e)
        }
    }
    let count: Double = Double(max(n, 1))
    for c in 0..<3 { d.meanAbs[c] = sums[c] / count }
    return d
}

/// Draw the reference frames from the record, in the given order, on a
/// fresh renderer; (frame, RGBA) each.
func filmDrawReferences(_ plan: FilmPlan, order: [Int] = filmReferenceFrames) throws -> [(Int, [UInt8])] {
    let frames = try FilmFrames(plan: plan, width: filmReferenceWidth, height: filmReferenceHeight,
                                samples: filmReferenceSamples)
    var out: [(Int, [UInt8])] = []
    for f in order {
        try frames.draw(f)
        out.append((f, frames.renderer.image.bytes()))
    }
    return out
}

// MARK: - files

/// Save RGBA8 pixels (top row first) as a PNG.
func filmSavePNG(_ rgba: [UInt8], width: Int, height: Int, to url: URL) throws {
    try? FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    let data = Data(rgba) as CFData
    guard let provider = CGDataProvider(data: data),
          let space = CGColorSpace(name: CGColorSpace.sRGB),
          let cg = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
                           bytesPerRow: width * 4, space: space,
                           bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                           provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent),
          let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)
    else { throw WorldRenderError.png("could not set up the encoder for \(url.path)") }
    CGImageDestinationAddImage(dest, cg, nil)
    guard CGImageDestinationFinalize(dest) else { throw WorldRenderError.png("could not write \(url.path)") }
}

/// A frame from the store, scaled to `width` × `height` (Core Graphics, high
/// quality interpolation), RGBA8.
func filmScaled(_ store: FilmStore, _ f: Int, width w: Int, height h: Int) throws -> [UInt8] {
    guard let src = CGImageSourceCreateWithURL(store.url(f) as CFURL, nil),
          let img = CGImageSourceCreateImageAtIndex(src, 0, nil),
          let space = CGColorSpace(name: CGColorSpace.sRGB) else { throw WorldRenderError.png("cannot read frame \(f)") }
    var out = [UInt8](repeating: 255, count: w * h * 4)
    let ok: Bool = out.withUnsafeMutableBytes { raw -> Bool in
        guard let ctx = CGContext(data: raw.baseAddress, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                                  space: space, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { return false }
        ctx.interpolationQuality = .high
        ctx.draw(img, in: CGRect(x: 0, y: 0, width: w, height: h))
        return true
    }
    guard ok else { throw WorldRenderError.png("cannot scale frame \(f)") }
    return out
}

/// A GIF excerpt from frames on disk, with step 8's writer (step 51's split
/// copy): one palette from frames across the excerpt, only changed pixels
/// stored per frame.
func filmGIF(store: FilmStore, frames list: [Int], width w: Int, delayCentiseconds: Int, to url: URL, copyTo: URL?) throws {
    let h: Int = w * 9 / 16
    guard let device = MTLCreateSystemDefaultDevice() ?? MTLCopyAllDevices().first else { throw WorldRenderError.noMetalDevice }
    let pixels: Int = w * h
    guard let buf = device.makeBuffer(length: pixels * 4, options: .storageModeShared) else { throw WorldRenderError.gpu("buffer") }
    func load(_ f: Int) throws {
        let rgba: [UInt8] = try filmScaled(store, f, width: w, height: h)
        rgba.withUnsafeBytes { raw in
            if let base = raw.baseAddress { buf.contents().copyMemory(from: base, byteCount: pixels * 4) }
        }
    }
    var sampled: [RGB] = []
    for f in stride(from: 0, to: list.count, by: max(list.count / 10, 1)).map({ list[$0] }) {
        try load(f)
        sampled += samplePixels(buf, pixels: pixels, step: 13)
        sampled += samplePixels(buf, pixels: pixels, step: 397)
    }
    let palette: [RGB] = medianCutPalette(sampled, count: 250)
    let quantizer = try Quantizer(device: device, palette: palette, pixels: pixels)
    let gif = GIFWriter(url: url, width: w, height: h, palette: palette, delayCentiseconds: delayCentiseconds)
    for f in list {
        try load(f)
        gif.add(try quantizer.indices(of: buf))
    }
    try gif.finish()
    let bytes: Int = (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? Int) ?? 0
    let mb: Double = Double(bytes) / 1_048_576
    print(String(format: "GIF: %d frames (film frames %d–%d), %d × %d, %d cs each → %@ (%.2f MB)",
                 list.count, list.first ?? 0, list.last ?? 0, w, h, delayCentiseconds, url.path, mb))
    if let dest = copyTo {
        try? FileManager.default.removeItem(at: dest)
        try FileManager.default.copyItem(at: url, to: dest)
    }
}
