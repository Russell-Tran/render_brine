// The animation's fixed camera and what the CPU draws over each frame: the
// title, the honest caption, a scale bar, and a footfall diagram of one
// stride of the gait. Everything here is the same in every frame, so the
// GIF stores only the toy's moving pixels.

import CoreGraphics
import CoreText
import Foundation
import simd

/// Frame shape for the animations: 3:2. MODEL.
let animAspect: Float = 1.5
/// The GIFs' frame width and delay: 800 px, 6 hundredths of a second a
/// frame (16.7 fps). Chosen after measuring: at 20 fps a walking frame costs
/// ~33 KB and the loop would pass 10 MB.
let animWidth: Int = 800
let gifDelayCentiseconds: Int = 6

struct AnimSetup {
    let performance: Performance
    let camera: Camera
    let studio: Studio
    /// The circle's centre at the toy's mid-height: where the scale bar is true.
    let barPoint: SIMD3<Float>
}

/// A camera that sees the whole circle and the toy at every point on it,
/// from the same side as the stills (the toy's left, a little in front) and
/// higher, so the circle opens out. Its distance and aim are fitted so the
/// whole walk fills the frame below the caption. MODEL.
func animSetup(_ perf: Performance) -> AnimSetup {
    let toy: PosedToy = perf.design.posed(perf.design.restPose())
    let ey = toy.extent(along: SIMD3<Float>(0, 1, 0))
    let centre: SIMD3<Float> = perf.path.centre + SIMD3<Float>(0, (ey.hi - ey.lo) * 0.42, 0)
    let tanHalf: Float = 0.2
    let az: Float = stillCameraAzimuthDegrees * .pi / 180
    let el: Float = 34 * .pi / 180
    let dir = SIMD3<Float>(cos(az) * cos(el), sin(el), -sin(az) * cos(el))
    // Every shape's bounding sphere, with the toy stood at 36 places round
    // the circle, and the crouched walk's lifted feet besides.
    var spheres: [(SIMD3<Float>, Float)] = []
    for k in 0..<36 {
        let s: Float = perf.spec.circumference * Float(k) / 36
        let (p, yaw) = perf.path.at(s)
        let placed: PosedToy = perf.design.posed(perf.design.restPose(at: p, yaw: yaw))
        for (i, seg) in placed.segments.enumerated() {
            for pr in seg.prims {
                let b = pr.bound
                spheres.append((placed.frames[i].toWorld(b.centre), b.radius + 1.5))
            }
        }
    }
    let W: Float = 840
    let H: Float = W / animAspect
    let region = SIMD4<Float>(0.02 * W, 0.235 * H, 0.98 * W, 0.965 * H)   // x0, y0, x1, y1
    var distance: Float = 400
    var pan = SIMD2<Float>(0, 0)
    var cam = Camera(position: centre + dir * distance, target: centre, tanHalfFOV: tanHalf)
    for _ in 0..<30 {
        let right: SIMD3<Float> = simd_normalize(simd_cross(-dir, SIMD3<Float>(0, 1, 0)))
        let up: SIMD3<Float> = simd_cross(right, -dir)
        let shift: SIMD3<Float> = right * pan.x + up * pan.y
        cam = Camera(position: centre + dir * distance + shift, target: centre + shift, tanHalfFOV: tanHalf)
        var lo = SIMD2<Float>(repeating: .infinity)
        var hi = SIMD2<Float>(repeating: -.infinity)
        for (c, r) in spheres {
            let p: SIMD2<Float> = project(c, width: Int(W), height: Int(H), cam: cam)
            let rp: Float = r * pixelsPerMillimetre(at: c, height: Int(H), cam: cam)
            lo = simd_min(lo, p - rp)
            hi = simd_max(hi, p + rp)
        }
        let scale: Float = max((hi.x - lo.x) / (region.z - region.x), (hi.y - lo.y) / (region.w - region.y))
        let mmPerPx: Float = 1 / pixelsPerMillimetre(at: centre, height: Int(H), cam: cam)
        let dx: Float = ((lo.x + hi.x) / 2 - (region.x + region.z) / 2) * mmPerPx
        let dy: Float = ((lo.y + hi.y) / 2 - (region.y + region.w) / 2) * mmPerPx
        pan += SIMD2<Float>(dx, -dy) * 0.8
        distance *= 1 + (scale - 1) * 0.8
    }
    return AnimSetup(performance: perf, camera: cam, studio: Studio(cameraAzimuthDegrees: stillCameraAzimuthDegrees),
                     barPoint: centre)
}

/// The caption for the animation, per toy.
struct AnimCaption {
    var title: String
    var lines: [String]
}

func animBar(_ a: AnimSetup, width: Int, height: Int) -> ScaleBar {
    let k: CGFloat = CGFloat(height) / 560
    let px: CGFloat = CGFloat(mainBarMillimetres * pixelsPerMillimetre(at: a.barPoint, height: height, cam: a.camera))
    return ScaleBar(label: String(format: "%g mm", mainBarMillimetres), x: 22 * k, y: CGFloat(height) - 24 * k, pixels: px)
}

/// The footfall diagram's frame (y down).
func gaitDiagramRect(width: Int, height: Int) -> CGRect {
    let k: CGFloat = CGFloat(height) / 560
    return CGRect(x: CGFloat(width) - 262 * k, y: 14 * k, width: 244 * k, height: 96 * k)
}

func annotateFrame(_ image: ToyImage, setup a: AnimSetup, caption: AnimCaption) {
    let w: Int = image.width
    let hI: Int = image.height
    guard let ctx = CGContext(data: image.pixels.contents(), width: w, height: hI, bitsPerComponent: 8,
                              bytesPerRow: w * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                              bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return }
    let h: CGFloat = CGFloat(hI)
    let k: CGFloat = h / 560
    ctx.setShouldAntialias(true)
    put(caption.title, ctx, x: 22 * k, top: 16 * k, size: 19 * k, bold: true, height: h)
    for (i, s) in caption.lines.enumerated() {
        put(s, ctx, x: 22 * k, top: (42 + 15 * CGFloat(i)) * k, size: 11 * k, color: softInk, height: h)
    }

    // The scale bar, true at the circle's centre.
    let bar: ScaleBar = animBar(a, width: w, height: hI)
    ctx.setFillColor(ink)
    ctx.fill(flip(CGRect(x: bar.x, y: bar.y, width: bar.pixels, height: 2 * k), h))
    ctx.fill(flip(CGRect(x: bar.x, y: bar.y - 4 * k, width: 1.5 * k, height: 6 * k), h))
    ctx.fill(flip(CGRect(x: bar.x + bar.pixels - 1.5 * k, y: bar.y - 4 * k, width: 1.5 * k, height: 6 * k), h))
    put(bar.label, ctx, x: bar.x + bar.pixels + 6 * k, top: bar.y - 7 * k, size: 11 * k, bold: true, height: h)

    // One stride of the gait: when each foot is on the ground.
    let spec: WalkSpec = a.performance.spec
    let box: CGRect = gaitDiagramRect(width: w, height: hI)
    ctx.setFillColor(CGColor(srgbRed: 1, green: 1, blue: 1, alpha: 0.55))
    ctx.fill(flip(box, h))
    put("footfalls, one stride (bars: foot down)", ctx, x: box.minX + 8 * k, top: box.minY + 5 * k, size: 10 * k, bold: true, height: h)
    let rowsTop: CGFloat = box.minY + 22 * k
    let rowH: CGFloat = 15 * k
    let barX: CGFloat = box.minX + 30 * k
    let barW: CGFloat = box.width - 40 * k
    for (r, name) in spec.footfallOrder.enumerated() {
        guard let j = a.performance.design.legs.firstIndex(where: { $0.name == name }) else { continue }
        let y: CGFloat = rowsTop + CGFloat(r) * rowH
        put(name, ctx, x: box.minX + 8 * k, top: y + 1 * k, size: 10 * k, height: h)
        // Stance runs from the leg's touchdown for the duty factor, wrapped
        // into one stride starting at the left hind's touchdown.
        let off: Float = spec.offsets[j] - spec.offsets[a.performance.design.legs.firstIndex { $0.name == spec.footfallOrder[0] } ?? 0]
        var start: Float = off - off.rounded(.down)
        let end: Float = start + spec.dutyFactor
        ctx.setFillColor(ink)
        if end <= 1 {
            ctx.fill(flip(CGRect(x: barX + CGFloat(start) * barW, y: y + 3 * k, width: CGFloat(spec.dutyFactor) * barW, height: 8 * k), h))
        } else {
            ctx.fill(flip(CGRect(x: barX + CGFloat(start) * barW, y: y + 3 * k, width: CGFloat(1 - start) * barW, height: 8 * k), h))
            start = 0
            ctx.fill(flip(CGRect(x: barX, y: y + 3 * k, width: CGFloat(end - 1) * barW, height: 8 * k), h))
        }
    }
}

/// Step 8's GIF encoder throws `RenderError`; here that is the toy's error.
typealias RenderError = ToyError
