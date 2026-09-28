// Putting the picture together on the CPU: the plain ground, the pod blurred
// behind, the flower over it, the magnified inset in its circle, and the few
// marks drawn on top — two scale bars, the inset's magnification, and a
// dotted line saying the pod is the same ovary later.
//
// Step 29 copies this from step 25 and adds: the pod blurred once and kept,
// frames that re-render only the rectangles round the newly grown tube, and
// the time-lapse caption under the inset.
//
// The pod is rendered small and blurred rather than with a thin-lens camera:
// it is entirely behind the flower, so "flower over blurred pod" is exactly
// what a lens focused on the flower would give, without sixteen samples of
// lens noise to pay for. How strong the blur is, is a choice (MODEL).

import CoreGraphics
import CoreText
import Foundation
import ImageIO
import UniformTypeIdentifiers
import simd

/// The ground, in display values: a plain pale grey-green, dark enough that
/// white petals read white against it. Top and bottom of a faint gradient.
let groundTop = SIMD3<Float>(0.765, 0.80, 0.795)
let groundBottom = SIMD3<Float>(0.69, 0.73, 0.715)

/// Where the inset sits in the frame, as fractions of the frame, and its size.
struct InsetPlacement {
    var centre: SIMD2<Float>
    var radius: Float
}

func insetPlacement(width: Int, height: Int) -> InsetPlacement {
    InsetPlacement(centre: SIMD2<Float>(Float(width) * 0.815, Float(height) * 0.33),
                   radius: Float(height) * 0.213)
}

/// The inset is rendered square, its side the circle's diameter.
func insetPixels(width: Int, height: Int) -> Int { Int(insetPlacement(width: width, height: height).radius * 2) }

/// Micrometres per pixel in each view, from the cameras themselves.
func mainPixelsPerMillimetre(height: Int) -> Float {
    mainCamera().pixelsPerMillimetre(at: mainTarget, height: height)
}
func insetPixelsPerMillimetre(side: Int, view: InsetView = stillInsetView) -> Float {
    insetCamera(view).pixelsPerMillimetre(at: view.target, height: side)
}

/// The scale bars: 1 mm in the main view, 100 µm in the inset.
let mainBarMillimetres: Float = 1.0
let insetBarMillimetres: Float = 0.1

/// The magnification written on the inset, to one decimal: the ratio of the
/// two views' pixels per millimetre.
func magnification(width: Int, height: Int, view: InsetView = stillInsetView) -> Float {
    let side: Int = insetPixels(width: width, height: height)
    let main: Float = mainCamera().pixelsPerMillimetre(at: view.mainReference, height: height)
    return insetPixelsPerMillimetre(side: side, view: view) / main
}
func magnificationLabel(width: Int, height: Int, view: InsetView = stillInsetView) -> String {
    String(format: "×%.1f", magnification(width: width, height: height, view: view))
}

/// Step 29: the inset's scale bar — 100 µm as in step 25 while that is at
/// least 45 px long (at 1080 lines), else the next of 200 or 500 µm that is.
func insetBarLength(width: Int, height: Int, view: InsetView = stillInsetView) -> Float {
    let side: Int = insetPixels(width: width, height: height)
    let ppm: Float = insetPixelsPerMillimetre(side: side, view: view)
    let least: Float = 45 * Float(height) / 1080
    for bar in [Float(0.1), 0.2, 0.5] where bar * ppm >= least { return bar }
    return 0.5
}

// MARK: - blur and composite

/// Separable Gaussian on premultiplied RGBA.
func blur(_ img: [SIMD4<Float>], width: Int, height: Int, sigma: Float) -> [SIMD4<Float>] {
    let r: Int = Int(ceil(sigma * 3))
    var weights: [Float] = []
    for i in -r...r {
        let x: Float = Float(i) / sigma
        weights.append(exp(-0.5 * x * x))
    }
    let total: Float = weights.reduce(0, +)
    weights = weights.map { $0 / total }
    var tmp = [SIMD4<Float>](repeating: .zero, count: img.count)
    var out = [SIMD4<Float>](repeating: .zero, count: img.count)
    for y in 0..<height {
        for x in 0..<width {
            var s = SIMD4<Float>(0, 0, 0, 0)
            for k in -r...r {
                let xx: Int = min(max(x + k, 0), width - 1)
                let wk: Float = weights[k + r]
                let px: SIMD4<Float> = img[y * width + xx]
                s += px * wk
            }
            tmp[y * width + x] = s
        }
    }
    for y in 0..<height {
        for x in 0..<width {
            var s = SIMD4<Float>(0, 0, 0, 0)
            for k in -r...r {
                let yy: Int = min(max(y + k, 0), height - 1)
                let wk: Float = weights[k + r]
                let px: SIMD4<Float> = tmp[yy * width + x]
                s += px * wk
            }
            out[y * width + x] = s
        }
    }
    return out
}

func sampleBilinear(_ img: [SIMD4<Float>], width: Int, height: Int, _ x: Float, _ y: Float) -> SIMD4<Float> {
    let fx: Float = min(max(x - 0.5, 0), Float(width - 1))
    let fy: Float = min(max(y - 0.5, 0), Float(height - 1))
    let x0: Int = Int(fx)
    let y0: Int = Int(fy)
    let x1: Int = min(x0 + 1, width - 1)
    let y1: Int = min(y0 + 1, height - 1)
    let tx: Float = fx - Float(x0)
    let ty: Float = fy - Float(y0)
    let p00: SIMD4<Float> = img[y0 * width + x0]
    let p01: SIMD4<Float> = img[y0 * width + x1]
    let p10: SIMD4<Float> = img[y1 * width + x0]
    let p11: SIMD4<Float> = img[y1 * width + x1]
    let a: SIMD4<Float> = p00 * (1 - tx) + p01 * tx
    let b: SIMD4<Float> = p10 * (1 - tx) + p11 * tx
    return a * (1 - ty) + b * ty
}

func layerArray(_ l: LayerImage) -> [SIMD4<Float>] {
    let p = l.pixels.contents().assumingMemoryBound(to: SIMD4<Float>.self)
    return Array(UnsafeBufferPointer(start: p, count: l.width * l.height))
}

/// Ground, blurred pod, flower, inset. Display values, row-major from the top.
/// The pod layer, blurred. Step 29 makes this once and reuses it every frame:
/// the pod does not change.
struct BlurredPod {
    var pixels: [SIMD4<Float>]
    var width: Int
    var height: Int
    var scale: Int
}

func blurPod(_ podSmall: LayerImage, podScale: Int, podSigma: Float) -> BlurredPod {
    BlurredPod(pixels: blur(layerArray(podSmall), width: podSmall.width, height: podSmall.height, sigma: podSigma),
               width: podSmall.width, height: podSmall.height, scale: podScale)
}

func composite(flower: LayerImage, pod blurred: BlurredPod, inset: LayerImage) -> [SIMD3<Float>] {
    let w: Int = flower.width
    let h: Int = flower.height
    let pod: [SIMD4<Float>] = blurred.pixels
    let podSmall: BlurredPod = blurred
    let podScale: Int = blurred.scale
    let fl: [SIMD4<Float>] = layerArray(flower)
    let ins: [SIMD4<Float>] = layerArray(inset)
    let place: InsetPlacement = insetPlacement(width: w, height: h)
    let side: Int = inset.width
    var out = [SIMD3<Float>](repeating: .zero, count: w * h)
    for y in 0..<h {
        let g: Float = Float(y) / Float(h - 1)
        let ground: SIMD3<Float> = groundTop + (groundBottom - groundTop) * g
        for x in 0..<w {
            let ps: SIMD4<Float> = sampleBilinear(pod, width: podSmall.width, height: podSmall.height,
                                                  (Float(x) + 0.5) / Float(podScale), (Float(y) + 0.5) / Float(podScale))
            // Distance haze on the pod: a third of the way to the ground colour.
            let podC: SIMD3<Float> = SIMD3<Float>(ps.x, ps.y, ps.z) * 0.67 + ground * (ps.w * 0.33)
            var c: SIMD3<Float> = podC + ground * (1 - ps.w)
            let f: SIMD4<Float> = fl[y * w + x]
            c = SIMD3<Float>(f.x, f.y, f.z) + c * (1 - f.w)
            // The inset: its own backdrop, a shade lighter, inside a circle.
            let dx: Float = Float(x) + 0.5 - place.centre.x
            let dy: Float = Float(y) + 0.5 - place.centre.y
            let r: Float = (dx * dx + dy * dy).squareRoot()
            let inside: Float = min(max(place.radius - r + 0.5, 0), 1)
            if inside > 0 {
                let ix: Int = min(max(Int(dx + Float(side) / 2), 0), side - 1)
                let iy: Int = min(max(Int(dy + Float(side) / 2), 0), side - 1)
                let v: SIMD4<Float> = ins[iy * side + ix]
                let back: SIMD3<Float> = SIMD3<Float>(0.87, 0.895, 0.89)
                let ic: SIMD3<Float> = SIMD3<Float>(v.x, v.y, v.z) + back * (1 - v.w)
                c = c * (1 - inside) + ic * inside
            }
            out[y * w + x] = c
        }
    }
    return out
}

// MARK: - marks

private func ctLine(_ s: String, size: CGFloat, font: String, color: CGColor) -> CTLine {
    let f = CTFontCreateWithName(font as CFString, size, nil)
    let attributes: [NSAttributedString.Key: Any] = [
        NSAttributedString.Key(kCTFontAttributeName as String): f,
        NSAttributedString.Key(kCTForegroundColorAttributeName as String): color,
    ]
    return CTLineCreateWithAttributedString(NSAttributedString(string: s, attributes: attributes))
}

private func put(_ s: String, _ ctx: CGContext, x: CGFloat, baseline: CGFloat, size: CGFloat, height: CGFloat,
                 font: String = "HelveticaNeue", color: CGColor, centred: Bool = false) {
    let line: CTLine = ctLine(s, size: size, font: font, color: color)
    var dx: CGFloat = 0
    if centred { dx = CGFloat(CTLineGetTypographicBounds(line, nil, nil, nil)) / 2 }
    ctx.textPosition = CGPoint(x: x - dx, y: height - baseline)
    CTLineDraw(line, ctx)
}

private let ink = CGColor(srgbRed: 0.18, green: 0.22, blue: 0.22, alpha: 0.9)
private let faint = CGColor(srgbRed: 0.18, green: 0.22, blue: 0.22, alpha: 0.45)

/// Step 29: the grown tube drawn over the main view, wider than life so it
/// can be seen, with a marker on its growing tip; and a ring round the tip in
/// the inset.
struct TubeOverlay {
    var points: [SIMD3<Float>]
}

/// The widened line's width in the main view, px at 1080 lines. MODEL: about
/// five times the tube's true 10 µm at this scale (see `tubeWidening`).
let tubeLineWidth: Float = 4.0

/// How many times wider than life the main view's tube line is drawn.
func tubeWidening(height: Int) -> Float {
    let truePx: Float = 2 * pollenTubeRadius * mainPixelsPerMillimetre(height: height)
    return tubeLineWidth * Float(height) / 1080 / truePx
}

private let tubeGold = CGColor(srgbRed: 1.0, green: 0.70, blue: 0.08, alpha: 1)
private let tubeEdge = CGColor(srgbRed: 0.50, green: 0.26, blue: 0.0, alpha: 0.85)
private let tipHalo = CGColor(srgbRed: 1.0, green: 0.62, blue: 0.0, alpha: 0.35)
private let tipCore = CGColor(srgbRed: 1.0, green: 0.93, blue: 0.55, alpha: 1)

private func drawTube(_ ctx: CGContext, _ o: TubeOverlay, view: InsetView, width: Int, height: Int) {
    let k: CGFloat = CGFloat(height) / 1080
    let h: CGFloat = CGFloat(height)
    let cam: Camera = mainCamera()
    let pts: [CGPoint] = o.points.map { p in
        let q: SIMD2<Float> = cam.project(p, width: width, height: height)
        return CGPoint(x: CGFloat(q.x), y: h - CGFloat(q.y))
    }
    guard let tip = pts.last else { return }
    ctx.setLineCap(.round)
    ctx.setLineJoin(.round)
    if pts.count > 1 {
        let w: CGFloat = CGFloat(tubeLineWidth) * k
        for (color, width) in [(tubeEdge, w + 2 * k), (tubeGold, w)] {
            ctx.setStrokeColor(color)
            ctx.setLineWidth(width)
            ctx.move(to: pts[0])
            for p in pts.dropFirst() { ctx.addLine(to: p) }
            ctx.strokePath()
        }
    }
    // The growing tip: a soft halo and a bright dot.
    ctx.setFillColor(tipHalo)
    ctx.fillEllipse(in: CGRect(x: tip.x - 12 * k, y: tip.y - 12 * k, width: 24 * k, height: 24 * k))
    ctx.setFillColor(tipCore)
    ctx.setStrokeColor(tubeEdge)
    ctx.setLineWidth(1.5 * k)
    let core: CGRect = CGRect(x: tip.x - 5 * k, y: tip.y - 5 * k, width: 10 * k, height: 10 * k)
    ctx.fillEllipse(in: core)
    ctx.strokeEllipse(in: core)
    // The same tip in the inset, ringed.
    let side: Int = insetPixels(width: width, height: height)
    let place: InsetPlacement = insetPlacement(width: width, height: height)
    let qi: SIMD2<Float> = insetCamera(view).project(o.points[o.points.count - 1], width: side, height: side)
    let ix: CGFloat = CGFloat(place.centre.x - Float(side) / 2 + qi.x)
    let iy: CGFloat = h - CGFloat(place.centre.y - Float(side) / 2 + qi.y)
    let dx: CGFloat = ix - CGFloat(place.centre.x)
    let dy: CGFloat = iy - (h - CGFloat(place.centre.y))
    if (dx * dx + dy * dy).squareRoot() < CGFloat(place.radius) - 20 * k {
        ctx.setStrokeColor(tubeGold)
        ctx.setLineWidth(2.5 * k)
        ctx.strokeEllipse(in: CGRect(x: ix - 16 * k, y: iy - 16 * k, width: 32 * k, height: 32 * k))
    }
}

/// The finished picture as sRGB bytes, marks drawn. Step 29 adds the inset's
/// view (step 25's by default) and the widened tube (none by default).
func finish(_ display: [SIMD3<Float>], width: Int, height: Int, view: InsetView = stillInsetView,
            tube: TubeOverlay? = nil) -> [UInt8] {
    var bytes = [UInt8](repeating: 255, count: width * height * 4)
    for i in 0..<(width * height) {
        let c: SIMD3<Float> = simd_clamp(display[i], SIMD3<Float>(0, 0, 0), SIMD3<Float>(1, 1, 1))
        bytes[i * 4] = UInt8((c.x * 255).rounded())
        bytes[i * 4 + 1] = UInt8((c.y * 255).rounded())
        bytes[i * 4 + 2] = UInt8((c.z * 255).rounded())
    }
    let k: CGFloat = CGFloat(height) / 1080
    let h: CGFloat = CGFloat(height)
    bytes.withUnsafeMutableBytes { raw in
        guard let ctx = CGContext(data: raw.baseAddress, width: width, height: height, bitsPerComponent: 8,
                                  bytesPerRow: width * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return }
        if let o = tube { drawTube(ctx, o, view: view, width: width, height: height) }
        ctx.setLineCap(.round)
        let cam: Camera = mainCamera()
        let place: InsetPlacement = insetPlacement(width: width, height: height)

        // The region the inset magnifies, and two lines out to it.
        let s: SIMD2<Float> = cam.project(view.target, width: width, height: height)
        let markR: CGFloat = CGFloat(view.field / 2 * cam.pixelsPerMillimetre(at: view.mainReference, height: height))
        let big: CGFloat = CGFloat(place.radius)
        let a = CGPoint(x: CGFloat(s.x), y: h - CGFloat(s.y))
        let b = CGPoint(x: CGFloat(place.centre.x), y: h - CGFloat(place.centre.y))
        ctx.setStrokeColor(faint)
        ctx.setLineWidth(1.2 * k)
        ctx.strokeEllipse(in: CGRect(x: a.x - markR, y: a.y - markR, width: 2 * markR, height: 2 * markR))
        // External tangents of the two circles.
        let dx: CGFloat = b.x - a.x
        let dy: CGFloat = b.y - a.y
        let d: CGFloat = (dx * dx + dy * dy).squareRoot()
        let base: CGFloat = atan2(dy, dx)
        let off: CGFloat = acos((big - markR) / d)
        for sign in [CGFloat(1), CGFloat(-1)] {
            let ang: CGFloat = base + sign * (CGFloat.pi - off)
            let p1 = CGPoint(x: a.x + markR * cos(ang), y: a.y + markR * sin(ang))
            let p2 = CGPoint(x: b.x + big * cos(ang), y: b.y + big * sin(ang))
            ctx.move(to: p1)
            ctx.addLine(to: p2)
        }
        ctx.strokePath()
        ctx.setStrokeColor(ink)
        ctx.setLineWidth(1.6 * k)
        ctx.strokeEllipse(in: CGRect(x: b.x - big, y: b.y - big, width: 2 * big, height: 2 * big))

        // Inset scale bar and magnification, inside the circle's lower edge.
        let side: Int = insetPixels(width: width, height: height)
        let barMM: Float = insetBarLength(width: width, height: height, view: view)
        let insetBar: CGFloat = CGFloat(barMM * insetPixelsPerMillimetre(side: side, view: view))
        let ib = CGPoint(x: b.x - big * 0.62 - insetBar / 2, y: b.y - big * 0.66)
        ctx.setLineWidth(3 * k)
        ctx.move(to: ib)
        ctx.addLine(to: CGPoint(x: ib.x + insetBar, y: ib.y))
        ctx.strokePath()
        put("\(Int((barMM * 1000).rounded())) µm", ctx, x: ib.x + insetBar / 2, baseline: h - ib.y - 8 * k, size: 17 * k,
            height: h, color: ink, centred: true)
        put(magnificationLabel(width: width, height: height, view: view), ctx, x: b.x + big * 0.62, baseline: h - b.y + big * 0.70,
            size: 17 * k, height: h, color: ink, centred: true)

        // Main scale bar, bottom left.
        let mainBar: CGFloat = CGFloat(mainBarMillimetres * mainPixelsPerMillimetre(height: height))
        let mb = CGPoint(x: 60 * k, y: 70 * k)
        ctx.move(to: mb)
        ctx.addLine(to: CGPoint(x: mb.x + mainBar, y: mb.y))
        ctx.strokePath()
        put("1 mm", ctx, x: mb.x + mainBar / 2, baseline: h - mb.y - 10 * k, size: 17 * k, height: h,
            color: ink, centred: true)

        // The ovary and the pod it becomes.
        let ov: SIMD2<Float> = cam.project(SIMD3<Float>(3.9, -0.55, 0), width: width, height: height)
        let pf: PodFrame = podFrame()
        let podMid: SIMD3<Float> = pf.origin + pf.x * (podLength * 0.42) + pf.y * 3.0
        let pm: SIMD2<Float> = cam.project(podMid, width: width, height: height)
        ctx.setStrokeColor(faint)
        ctx.setLineWidth(1.4 * k)
        ctx.setLineDash(phase: 0, lengths: [2 * k, 6 * k])
        ctx.move(to: CGPoint(x: CGFloat(ov.x), y: h - CGFloat(ov.y)))
        ctx.addLine(to: CGPoint(x: CGFloat(pm.x), y: h - CGFloat(pm.y)))
        ctx.strokePath()
        ctx.setLineDash(phase: 0, lengths: [])
        put("the same ovary, ~\(podDaysAfterAnthesis) days later", ctx, x: CGFloat(pm.x) + 14 * k,
            baseline: CGFloat(pm.y) + 34 * k, size: 17 * k, height: h, font: "HelveticaNeue-Italic", color: ink)

        put("Phaseolus vulgaris", ctx, x: 60 * k, baseline: h - 110 * k, size: 19 * k, height: h,
            font: "HelveticaNeue-Italic", color: ink)
    }
    return bytes
}

func savePNG(_ bytes: [UInt8], width: Int, height: Int, to url: URL) throws {
    try? FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    let data = Data(bytes)
    guard let provider = CGDataProvider(data: data as CFData),
          let space = CGColorSpace(name: CGColorSpace.sRGB),
          let cg = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
                           bytesPerRow: width * 4, space: space,
                           bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                           provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent),
          let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)
    else { throw BeanError.png("could not set up the encoder for \(url.path)") }
    CGImageDestinationAddImage(dest, cg, nil)
    guard CGImageDestinationFinalize(dest) else { throw BeanError.png("could not write \(url.path)") }
}

/// Everything about a frame that does not change as the tube grows — the
/// scene, the sizes, the blurred pod — so each frame renders only the flower
/// and the inset. (Step 25 rendered all three once; step 29 renders the pod
/// once and the other two per frame.)
final class BeanFrames {
    let scene: BeanScene
    let width: Int
    let height: Int
    let samples: Int
    let pod: BlurredPod
    private(set) var gpuSeconds: Double = 0
    /// The flower and inset layers from the last frame. Only the rectangles
    /// round the tube change as it grows, so after the first frame only those
    /// are rendered again (a test checks this gives the same picture as
    /// rendering everything).
    private var flower: LayerImage? = nil
    private var inset: LayerImage? = nil
    /// When false, every frame is rendered whole (for that test).
    var reuse: Bool = true

    /// How far the tube had grown in the layers held from the last frame.
    private var lastGrown: Float = 0
    let mainRectMarginMillimetres: Float = 1.5
    let insetRectMarginMillimetres: Float = 0.25

    /// The rectangles round the stretch of tube between two lengths in each
    /// view — all that differs between the last frame and this one — widened
    /// by a margin for its shading and the soft shadow it casts on what lies
    /// round and behind it. MODEL margins — 1.5 mm in the main view, where the
    /// tube is drawn wider than life and its shadow falls on the keel behind,
    /// 0.25 mm in the inset — checked against a whole render by the tests.
    func mainTubeRect(from a: Float, to b: Float) -> PixelRect {
        tubeRect(mainCamera(), width: width, height: height, from: a, to: b,
                 margin: mainRectMarginMillimetres * mainPixelsPerMillimetre(height: height))
    }
    func insetTubeRect(from a: Float, to b: Float) -> PixelRect {
        let side: Int = insetPixels(width: width, height: height)
        return tubeRect(insetCamera(), width: side, height: side, from: a, to: b,
                        margin: insetRectMarginMillimetres * insetPixelsPerMillimetre(side: side))
    }
    private func tubeRect(_ cam: Camera, width w: Int, height h: Int, from a: Float, to b: Float,
                          margin: Float) -> PixelRect {
        var lo = SIMD2<Float>(repeating: 1e9)
        var hi = SIMD2<Float>(repeating: -1e9)
        let tube: Chain = scene.model.tube
        let s0: Float = min(a, b)
        let s1: Float = max(a, b)
        // The vertices between the two lengths, and the two cut points.
        var stretch: [SIMD3<Float>] = []
        let arc: [Float] = tubeArcLengths(tube.points)
        for i in 0..<tube.points.count where arc[i] >= s0 && arc[i] <= s1 { stretch.append(tube.points[i]) }
        let cutA: [SIMD3<Float>] = grownTube(tube, s0)
        let cutB: [SIMD3<Float>] = grownTube(tube, s1)
        if let p = cutA.last { stretch.append(p) }
        if let p = cutB.last { stretch.append(p) }
        for p in stretch {
            let q: SIMD2<Float> = cam.project(p, width: w, height: h)
            // Only the part of the tube this view can see.
            if q.x < -margin || q.y < -margin || q.x > Float(w) + margin || q.y > Float(h) + margin { continue }
            lo = simd_min(lo, q)
            hi = simd_max(hi, q)
        }
        let x0: Int = max(Int(lo.x - margin), 0)
        let y0: Int = max(Int(lo.y - margin), 0)
        let x1: Int = min(Int(hi.x + margin) + 1, w)
        let y1: Int = min(Int(hi.y + margin) + 1, h)
        if x1 <= x0 || y1 <= y0 { return PixelRect(x0: 0, y0: 0, x1: 0, y1: 0) }
        return PixelRect(x0: x0, y0: y0, x1: x1, y1: y1)
    }

    init(_ scene: BeanScene, width: Int, height: Int, samples: Int) throws {
        self.scene = scene
        self.width = width
        self.height = height
        self.samples = samples
        let podScale: Int = 4
        let (podSmall, g) = try scene.render(camera: mainCamera(), width: width / podScale, height: height / podScale,
                                             samples: 2, layer: 1, tubeMin: 0, tubeGrown: 0)
        pod = blurPod(podSmall, podScale: podScale, podSigma: Float(height) / 1080 * 5.0)
        gpuSeconds = g
    }

    /// The inset's view in the layer held from the last frame.
    private var lastView: InsetView = stillInsetView

    /// The picture with the tube grown to `grown` mm, marks drawn (step 25's
    /// marks, the inset looking through `view`, and the widened tube if
    /// given; the time-lapse caption is drawn separately).
    func render(grown: Float, view: InsetView = stillInsetView,
                tube overlay: TubeOverlay? = nil) throws -> (bytes: [UInt8], flower: LayerImage, inset: LayerImage) {
        let footprint: Float = 1 / mainPixelsPerMillimetre(height: height)
        // The pollen tube is 10 µm across and a main-view pixel is ~15: drawn at
        // true width it would flicker in and out. So in the main view only, it is
        // held to at least 0.6 px wide (as step 13 held its setae), and at its true
        // width in the inset.
        let side: Int = insetPixels(width: width, height: height)
        let again: Bool = reuse && self.flower != nil
        let mainRect: PixelRect? = again ? mainTubeRect(from: lastGrown, to: grown) : nil
        // A moved inset is rendered whole; a still one only round the new tube.
        let insetAgain: Bool = again && view == stillInsetView && lastView == stillInsetView
        let insetRect: PixelRect? = insetAgain ? insetTubeRect(from: lastGrown, to: grown) : nil
        let (flower, g1) = try scene.render(camera: mainCamera(), width: width, height: height, samples: samples,
                                            layer: 0, tubeMin: footprint * 0.6, tubeGrown: grown,
                                            into: again ? self.flower : nil, rect: mainRect)
        let (inset, g2) = try scene.render(camera: insetCamera(view), width: side, height: side, samples: samples,
                                           layer: 0, tubeMin: 0, tubeGrown: grown,
                                           into: again ? self.inset : nil, rect: insetRect, cutZ: view.cutZ)
        self.flower = flower
        self.inset = inset
        lastGrown = grown
        lastView = view
        gpuSeconds += g1 + g2
        let display: [SIMD3<Float>] = composite(flower: flower, pod: pod, inset: inset)
        return (finish(display, width: width, height: height, view: view, tube: overlay), flower, inset)
    }
}

/// The whole picture, start to finish, with the tube fully grown — step 25's
/// still. Returns bytes and GPU seconds.
func renderBean(_ scene: BeanScene, width: Int, height: Int, samples: Int) throws -> (bytes: [UInt8], gpu: Double, flower: LayerImage, inset: LayerImage) {
    let frames = try BeanFrames(scene, width: width, height: height, samples: samples)
    let r = try frames.render(grown: scene.layout.tubeLength)
    return (r.bytes, frames.gpuSeconds, r.flower, r.inset)
}

// MARK: - step 29: the time-lapse caption

/// The clock, to the nearest 10 minutes so it reads as a clock rather than a
/// blur of digits.
func clockText(hours: Float) -> String {
    let minutes: Int = Int((hours * 6).rounded()) * 10
    return "\(minutes / 60) h \(String(format: "%02d", minutes % 60)) min"
}

/// Two lines under the inset: what is shown and how fast, and the real time
/// since the tube emerged.
func drawCaption(_ bytes: inout [UInt8], width: Int, height: Int, hours: Float, rate: Float,
                 totalHours: Float, growthSeconds: Float, legend: Bool) {
    let k: CGFloat = CGFloat(height) / 1080
    let h: CGFloat = CGFloat(height)
    let place: InsetPlacement = insetPlacement(width: width, height: height)
    let x: CGFloat = CGFloat(place.centre.x)
    let top: CGFloat = CGFloat(place.centre.y + place.radius)
    bytes.withUnsafeMutableBytes { raw in
        guard let ctx = CGContext(data: raw.baseAddress, width: width, height: height, bitsPerComponent: 8,
                                  bytesPerRow: width * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return }
        let line1: String = String(format: "time-lapse: ~%.0f h of pollen-tube growth in %.1f s  ·  ~%.1f mm/h",
                                   totalHours, growthSeconds, rate)
        put(line1, ctx, x: x, baseline: top + 48 * k, size: 17 * k, height: h,
            font: "HelveticaNeue-Italic", color: ink, centred: true)
        put(clockText(hours: hours) + " since the tube emerged", ctx, x: x, baseline: top + 76 * k,
            size: 17 * k, height: h, color: ink, centred: true)
        guard legend else { return }
        // A key for the widened line: a short sample of it, then what it is.
        let text: String = String(format: "pollen tube, drawn ~%.0f× wider to be seen", tubeWidening(height: height))
        put("the inset follows its tip, cut open along the style", ctx, x: x, baseline: top + 132 * k,
            size: 17 * k, height: h, color: ink, centred: true)
        let y: CGFloat = h - (top + 104 * k) + 6 * k
        let x0: CGFloat = x - 170 * k
        ctx.setLineCap(.round)
        for (color, w) in [(tubeEdge, CGFloat(tubeLineWidth + 2) * k), (tubeGold, CGFloat(tubeLineWidth) * k)] {
            ctx.setStrokeColor(color)
            ctx.setLineWidth(w)
            ctx.move(to: CGPoint(x: x0, y: y))
            ctx.addLine(to: CGPoint(x: x0 + 28 * k, y: y))
            ctx.strokePath()
        }
        put(text, ctx, x: x0 + 40 * k, baseline: top + 104 * k, size: 17 * k, height: h, color: ink)
    }
}
