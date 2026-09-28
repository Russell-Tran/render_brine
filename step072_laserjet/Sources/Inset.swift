// The inset: how a laser printer prints, at the scale it happens — the
// edge of a 12-point letter on the page, seen from above, 0.8 mm across.
//
// Electrophotography, from Wikipedia ("Laser printing", read for this
// step): a laser beam passes "back and forth over a negatively charged
// cylinder called a 'drum' to define a differentially charged image. The
// drum then selectively collects electrically charged powdered ink (toner),
// and transfers the image to paper, which is then heated to permanently
// fuse" it; "Toner is fused onto paper with heat and pressure." (Wikipedia
// gives a fuser temperature "up to 427 °C"; that is not a figure for this
// printer and is not used.)
//
// What is drawn, and from where:
//   • the dot grid: FastRes 600, 600 dots per inch [HP-UG], so dots sit
//     25.4 mm / 600 = 42.33 µm apart. (FastRes 1200 is HP's "effective"
//     1,200 dpi enhancement; it is not what is shown.)
//   • which dots are on: the letter "A" in Helvetica at 12 pt, rasterised by
//     asking of each dot's centre whether it lies inside the glyph — the
//     simplest rasteriser, MODEL (printers' own add edge smoothing).
//   • the toner: each dot a heap of toner particles fused together. Particle
//     size MODEL: 5–9 µm across. Wikipedia ("Toner (printing)") gives the
//     bounds: toner "originally ... averaged 14–16 micrometres", and "for
//     the perfect reproduction of dots and print features at 600 dpi, a
//     particle size of about 5 μm is required". HP's own toner size for the
//     85A cartridge was not found: UNVERIFIED.
//   • the dot's size: MODEL, a fused heap 1.24 × the pitch across, so
//     neighbouring dots merge into solid black, as printed text is, and
//     the letter's diagonal edge shows the grid as a staircase.
//   • the paper: 75 g/m² [HP-UG], a mat of cellulose fibres — the same
//     molecule as step 52's cotton — drawn 10–30 µm wide (Wikipedia,
//     "Paper": a micrograph caption, fibres "around 10 μm in diameter" in
//     tissue; office paper's wood fibres are wider, MODEL).

import CoreGraphics
import CoreText
import Foundation
import simd

let micrometresPerInch: Float = 25_400

/// Dot pitch in µm: 1/600 in, or the mutant's 1/300.
func dotPitch(_ m: Mutant = activeMutant) -> Float {
    var dpi: Float = dotsPerInch
    if case .wrongDPI = m { dpi = 300 }
    return micrometresPerInch / dpi
}

/// Toner particle diameters, µm: MODEL 5–9 (the mutant's are a dot wide).
func tonerDiameterRange(_ m: Mutant = activeMutant) -> ClosedRange<Float> {
    m == .tonerTooBig ? 38...46 : 5...9
}
/// Wikipedia's bounds on toner size, µm: ~5 needed for 600 dpi, 14–16 for
/// older toners.
let tonerSourcedRange: ClosedRange<Float> = 4.5...16

/// The inset's field: 800 µm square.
let insetFieldMicrometres: Float = 800
/// The letter's size: 12 pt = 12/72 in.
let letterPoints: Float = 12

/// A deterministic random stream, so the inset is the same every run.
struct Stream {
    var state: UInt64
    mutating func next() -> Float {
        state = state &* 6364136223846793005 &+ 1442695040888963407
        let x: UInt64 = (state >> 33) & 0xFFFFFF
        return Float(x) / Float(0x1000000)
    }
    mutating func normal() -> Float {
        let u1: Float = max(next(), 1e-7)
        let u2: Float = next()
        return (-2 * log(u1)).squareRoot() * cos(2 * Float.pi * u2)
    }
}

/// The letter's outline, in µm, with its baseline's left end at the origin,
/// y up.
func letterPath() -> CGPath {
    let pointsPerMicrometre: CGFloat = 72.0 / 25_400.0
    let size: CGFloat = CGFloat(letterPoints)
    let font = CTFontCreateWithName("Helvetica" as CFString, size, nil)
    var glyph: CGGlyph = 0
    var ch: UniChar = UniChar(("A" as String).utf16.first!)
    CTFontGetGlyphsForCharacters(font, &ch, &glyph, 1)
    let scale: CGFloat = 1 / pointsPerMicrometre
    var t = CGAffineTransform(scaleX: scale, y: scale)
    return CTFontCreatePathForGlyph(font, glyph, &t) ?? CGMutablePath()
}

/// The inset's window onto the page, in µm: where on the letter it looks.
/// Over the right-hand leg of the "A", where it meets the crossbar's
/// underside: a diagonal edge, a horizontal edge and a corner.
func insetWindowOrigin(_ path: CGPath) -> SIMD2<Float> {
    let b: CGRect = path.boundingBoxOfPath
    // Right leg, a little below the crossbar.
    return SIMD2<Float>(Float(b.minX + b.width * 0.62), Float(b.minY + b.height * 0.12))
}

struct TonerDot {
    var centre: SIMD2<Float>          // µm, inset window coordinates (y up)
    var particles: [SIMD3<Float>]     // x, y, diameter, µm
}

struct Fibre {
    var points: [SIMD2<Float>]        // µm, window coordinates
    var width: Float
    var shade: Float
}

struct InsetContent {
    var pitch: Float
    var dots: [TonerDot]
    var strays: [SIMD3<Float>]
    var fibres: [Fibre]
    /// The grid's origin in the window (a dot centre), µm.
    var gridOrigin: SIMD2<Float>
}

func buildInset(_ m: Mutant = activeMutant) -> InsetContent {
    let path: CGPath = letterPath()
    let pitch: Float = dotPitch(m)
    let win: SIMD2<Float> = insetWindowOrigin(path)
    let field: Float = insetFieldMicrometres
    var rng = Stream(state: 20260928)
    // Dot centres on the page's 600-dpi grid (page origin at a dot), those
    // inside the glyph turned on; kept if they fall near the window.
    let i0: Int = Int(((win.x - 60) / pitch).rounded(.down))
    let i1: Int = Int(((win.x + field + 60) / pitch).rounded(.up))
    let j0: Int = Int(((win.y - 60) / pitch).rounded(.down))
    let j1: Int = Int(((win.y + field + 60) / pitch).rounded(.up))
    var dots: [TonerDot] = []
    let dr: ClosedRange<Float> = tonerDiameterRange(m)
    let heap: Float = 0.62 * pitch
    for j in j0...j1 {
        for i in i0...i1 {
            let fi: Float = Float(i)
            let fj: Float = Float(j)
            let page = SIMD2<Float>(fi * pitch, fj * pitch)
            guard path.contains(CGPoint(x: CGFloat(page.x), y: CGFloat(page.y))) else { continue }
            var ps: [SIMD3<Float>] = []
            let meanD: Float = (dr.lowerBound + dr.upperBound) / 2
            let area: Float = Float.pi * heap * heap
            let count: Int = Int(area / (meanD * meanD) * 0.5)
            for _ in 0..<count {
                let rr: Float = heap * rng.next().squareRoot()
                let a: Float = 2 * Float.pi * rng.next()
                let d: Float = dr.lowerBound + (dr.upperBound - dr.lowerBound) * rng.next()
                ps.append(SIMD3<Float>(page.x - win.x + rr * cos(a), page.y - win.y + rr * sin(a), d))
            }
            dots.append(TonerDot(centre: page - win, particles: ps))
        }
    }
    // Stray particles that landed off the dots: a few near the edge. MODEL.
    var strays: [SIMD3<Float>] = []
    for d in dots where rng.next() < 0.35 {
        let a: Float = 2 * Float.pi * rng.next()
        let rr: Float = heap + 4 + 10 * rng.next()
        let dd: Float = dr.lowerBound + (dr.upperBound - dr.lowerBound) * rng.next()
        strays.append(SIMD3<Float>(d.centre.x + rr * cos(a), d.centre.y + rr * sin(a), dd))
    }
    // Fibres: gently curving ribbons crossing the field.
    var fibres: [Fibre] = []
    for _ in 0..<150 {
        var p = SIMD2<Float>(-200 + (field + 400) * rng.next(), -200 + (field + 400) * rng.next())
        var a: Float = 2 * Float.pi * rng.next()
        var pts: [SIMD2<Float>] = []
        let length: Float = 400 + 900 * rng.next()
        let steps: Int = 40
        for _ in 0...steps {
            pts.append(p)
            a += 0.08 * rng.normal()
            p += SIMD2<Float>(cos(a), sin(a)) * (length / Float(steps))
        }
        fibres.append(Fibre(points: pts, width: 10 + 20 * rng.next(), shade: rng.next()))
    }
    // The grid origin: the page's (0, 0) dot, expressed in the window.
    let gx: Float = (win.x / pitch).rounded(.up) * pitch - win.x
    let gy: Float = (win.y / pitch).rounded(.up) * pitch - win.y
    return InsetContent(pitch: pitch, dots: dots, strays: strays, fibres: fibres, gridOrigin: SIMD2<Float>(gx, gy))
}

// MARK: - drawing it

/// Draw the inset into `rect` (image coordinates, y down) of a CGContext
/// whose y axis is flipped (origin bottom-left), `h` the image height.
func drawInset(_ ctx: CGContext, _ c: InsetContent, rect r: CGRect, imageHeight h: CGFloat) {
    let px: CGFloat = r.width / CGFloat(insetFieldMicrometres)      // pixels per µm
    // Window µm (y up) to context coordinates (y up from the image bottom).
    func pt(_ p: SIMD2<Float>) -> CGPoint {
        CGPoint(x: r.minX + CGFloat(p.x) * px, y: (h - r.maxY) + CGFloat(p.y) * px)
    }
    ctx.saveGState()
    ctx.clip(to: CGRect(x: r.minX, y: h - r.maxY, width: r.width, height: r.height))
    // Paper: a warm white ground, then the fibres, each a pale ribbon with
    // darker edges where it curls down to the next.
    ctx.setFillColor(CGColor(srgbRed: 0.90, green: 0.90, blue: 0.87, alpha: 1))
    ctx.fill(CGRect(x: r.minX, y: h - r.maxY, width: r.width, height: r.height))
    ctx.setLineCap(.round)
    ctx.setLineJoin(.round)
    for f in c.fibres {
        let path = CGMutablePath()
        path.addLines(between: f.points.map { pt($0) })
        let g: CGFloat = 0.80 + 0.14 * CGFloat(f.shade)
        ctx.setStrokeColor(CGColor(srgbRed: g * 0.86, green: g * 0.86, blue: g * 0.83, alpha: 0.55))
        ctx.setLineWidth(CGFloat(f.width) * px)
        ctx.addPath(path)
        ctx.strokePath()
        ctx.setStrokeColor(CGColor(srgbRed: g * 1.04, green: g * 1.04, blue: g, alpha: 0.75))
        ctx.setLineWidth(CGFloat(f.width) * px * 0.55)
        ctx.addPath(path)
        ctx.strokePath()
    }
    // Toner: fused heaps of near-black particles, each with a small
    // highlight where its melted top catches the light.
    func particle(_ q: SIMD3<Float>, alpha: CGFloat) {
        let cpt: CGPoint = pt(SIMD2<Float>(q.x, q.y))
        let rad: CGFloat = CGFloat(q.z / 2) * px
        let e = CGRect(x: cpt.x - rad, y: cpt.y - rad, width: 2 * rad, height: 2 * rad)
        ctx.setFillColor(CGColor(srgbRed: 0.15, green: 0.15, blue: 0.16, alpha: alpha))
        ctx.fillEllipse(in: e)
        ctx.setStrokeColor(CGColor(srgbRed: 0.02, green: 0.02, blue: 0.02, alpha: alpha))
        ctx.setLineWidth(0.8 * h / 1080)
        ctx.strokeEllipse(in: e)
    }
    // The fused film: melted toner, one disc per dot (0.62 of the pitch in
    // radius, MODEL), the particles' round tops still showing at its rim.
    let melt: CGFloat = CGFloat(c.pitch * 0.62) * px
    ctx.setFillColor(CGColor(srgbRed: 0.05, green: 0.05, blue: 0.06, alpha: 1))
    for d in c.dots {
        let q: CGPoint = pt(d.centre)
        ctx.fillEllipse(in: CGRect(x: q.x - melt, y: q.y - melt, width: 2 * melt, height: 2 * melt))
    }
    for d in c.dots { for q in d.particles { particle(q, alpha: 1) } }
    for q in c.strays { particle(q, alpha: 1) }
    for d in c.dots {
        for q in d.particles.prefix(8) {
            let cpt: CGPoint = pt(SIMD2<Float>(q.x - q.z * 0.15, q.y + q.z * 0.15))
            let rad: CGFloat = CGFloat(q.z * 0.12) * px
            ctx.setFillColor(CGColor(srgbRed: 0.30, green: 0.30, blue: 0.32, alpha: 0.8))
            ctx.fillEllipse(in: CGRect(x: cpt.x - rad, y: cpt.y - rad, width: 2 * rad, height: 2 * rad))
        }
    }
    // The 600-dpi grid: a small cross at every dot position, in blue.
    ctx.setStrokeColor(CGColor(srgbRed: 0.10, green: 0.35, blue: 0.85, alpha: 0.75))
    ctx.setLineWidth(1.0 * h / 1080)
    var y: Float = c.gridOrigin.y - c.pitch * 2
    while y < insetFieldMicrometres + c.pitch {
        var x: Float = c.gridOrigin.x - c.pitch * 2
        while x < insetFieldMicrometres + c.pitch {
            let q: CGPoint = pt(SIMD2<Float>(x, y))
            let a: CGFloat = 3.5 * h / 1080
            ctx.move(to: CGPoint(x: q.x - a, y: q.y)); ctx.addLine(to: CGPoint(x: q.x + a, y: q.y))
            ctx.move(to: CGPoint(x: q.x, y: q.y - a)); ctx.addLine(to: CGPoint(x: q.x, y: q.y + a))
            x += c.pitch
        }
        y += c.pitch
    }
    ctx.strokePath()
    ctx.restoreGState()
}
