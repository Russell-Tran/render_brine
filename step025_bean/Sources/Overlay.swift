// Putting the picture together on the CPU: the plain ground, the pod blurred
// behind, the flower over it, the magnified inset in its circle, and the few
// marks drawn on top — two scale bars, the inset's magnification, and a
// dotted line saying the pod is the same ovary later.
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
func insetPixelsPerMillimetre(side: Int) -> Float {
    insetCamera().pixelsPerMillimetre(at: insetTarget, height: side)
}

/// The scale bars: 1 mm in the main view, 100 µm in the inset.
let mainBarMillimetres: Float = 1.0
let insetBarMillimetres: Float = 0.1

/// The magnification written on the inset, to one decimal: the ratio of the
/// two views' pixels per millimetre.
func magnification(width: Int, height: Int) -> Float {
    let side: Int = insetPixels(width: width, height: height)
    return insetPixelsPerMillimetre(side: side) / mainPixelsPerMillimetre(height: height)
}
func magnificationLabel(width: Int, height: Int) -> String {
    String(format: "×%.1f", magnification(width: width, height: height))
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
    let a: SIMD4<Float> = img[y0 * width + x0] * (1 - tx) + img[y0 * width + x1] * tx
    let b: SIMD4<Float> = img[y1 * width + x0] * (1 - tx) + img[y1 * width + x1] * tx
    return a * (1 - ty) + b * ty
}

func layerArray(_ l: LayerImage) -> [SIMD4<Float>] {
    let p = l.pixels.contents().assumingMemoryBound(to: SIMD4<Float>.self)
    return Array(UnsafeBufferPointer(start: p, count: l.width * l.height))
}

/// Ground, blurred pod, flower, inset. Display values, row-major from the top.
func composite(flower: LayerImage, podSmall: LayerImage, podScale: Int, podSigma: Float,
               inset: LayerImage) -> [SIMD3<Float>] {
    let w: Int = flower.width
    let h: Int = flower.height
    let pod: [SIMD4<Float>] = blur(layerArray(podSmall), width: podSmall.width, height: podSmall.height, sigma: podSigma)
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

/// The finished picture as sRGB bytes, marks drawn.
func finish(_ display: [SIMD3<Float>], width: Int, height: Int) -> [UInt8] {
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
        ctx.setLineCap(.round)
        let cam: Camera = mainCamera()
        let place: InsetPlacement = insetPlacement(width: width, height: height)

        // The region the inset magnifies, and two lines out to it.
        let s: SIMD2<Float> = cam.project(insetTarget, width: width, height: height)
        let markR: CGFloat = CGFloat(insetFieldDiameter / 2 * mainPixelsPerMillimetre(height: height))
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
        let insetBar: CGFloat = CGFloat(insetBarMillimetres * insetPixelsPerMillimetre(side: side))
        let ib = CGPoint(x: b.x - big * 0.62 - insetBar / 2, y: b.y - big * 0.66)
        ctx.setLineWidth(3 * k)
        ctx.move(to: ib)
        ctx.addLine(to: CGPoint(x: ib.x + insetBar, y: ib.y))
        ctx.strokePath()
        put("100 µm", ctx, x: ib.x + insetBar / 2, baseline: h - ib.y - 8 * k, size: 17 * k, height: h, color: ink, centred: true)
        put(magnificationLabel(width: width, height: height), ctx, x: b.x + big * 0.62, baseline: h - b.y + big * 0.70,
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

/// The whole picture, start to finish. Returns bytes and GPU seconds.
func renderBean(_ scene: BeanScene, width: Int, height: Int, samples: Int) throws -> (bytes: [UInt8], gpu: Double, flower: LayerImage, inset: LayerImage) {
    let cam: Camera = mainCamera()
    let footprint: Float = 1 / mainPixelsPerMillimetre(height: height)
    // The pollen tube is 10 µm across and a main-view pixel is ~15: drawn at
    // true width it would flicker in and out. So in the main view only, it is
    // held to at least 0.6 px wide (as step 13 held its setae), and at its true
    // width in the inset.
    let (flower, g1) = try scene.render(camera: cam, width: width, height: height, samples: samples,
                                        layer: 0, tubeMin: footprint * 0.6)
    let podScale: Int = 4
    let (pod, g2) = try scene.render(camera: cam, width: width / podScale, height: height / podScale,
                                     samples: 2, layer: 1, tubeMin: 0)
    let side: Int = insetPixels(width: width, height: height)
    let (inset, g3) = try scene.render(camera: insetCamera(), width: side, height: side, samples: samples,
                                       layer: 0, tubeMin: 0)
    let display: [SIMD3<Float>] = composite(flower: flower, podSmall: pod, podScale: podScale,
                                            podSigma: Float(height) / 1080 * 5.0, inset: inset)
    return (finish(display, width: width, height: height), g1 + g2 + g3, flower, inset)
}
