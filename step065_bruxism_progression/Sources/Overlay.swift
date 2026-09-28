// The marks drawn over the render: leader-line labels on the features the wear
// made, a caption saying what stage and how many years, and an inset — a cut
// through the worn central incisor, to scale, sampled from the same distance
// functions that drew the teeth.
//
// Every label's target is found by the survey (Survey.swift), not placed by
// eye, and a test checks each lands on a pixel that shows what it names.

import CoreGraphics
import CoreText
import Foundation
import Metal
import simd

// MARK: - where a point lands on the picture

extension Camera {
    /// A world point's pixel position (x right, y down), as the kernel's
    /// `cameraRay` would send a ray through it.
    func project(_ p: SIMD3<Float>, width: Int, height: Int) -> SIMD2<Float> {
        let d: SIMD3<Float> = p - position
        let z: Float = simd_dot(d, forward)
        let sx: Float = simd_dot(d, right) / z
        let sy: Float = simd_dot(d, up) / z
        let aspect: Float = Float(width) / Float(height)
        let px: Float = (sx / (aspect * tanHalfFOV) + 1) * Float(width) / 2
        let py: Float = (1 - sy / tanHalfFOV) * Float(height) / 2
        return SIMD2<Float>(px, py)
    }
}

// MARK: - what the labels point at

/// The world points the labels point at, chosen from the survey.
struct LabelTargets {
    var dentine: SIMD3<Float>
    var facet: SIMD3<Float>
    var canine: SIMD3<Float>
}

/// Which teeth: the patient's left side faces the camera, so its first
/// premolar (#21, index 3) for dentine, second premolar (#20, index 4) for a
/// facet, canine (#22, index 2) for the canine.
let dentineLabelTooth: Int = 3
let facetLabelTooth: Int = 4
let canineLabelTooth: Int = 2

/// The middle of the first premolar's dentine; the point of the second premolar's
/// facet furthest from any dentine — on its lingual side, which the camera
/// looks down on — so the leader lands on facet enamel and not on the dentine
/// island inside it; the canine's cut tip.
func labelTargets(_ survey: [ToothSurvey]) -> LabelTargets? {
    let d: ToothSurvey = survey[dentineLabelTooth]
    let dent: [SurveyPoint] = d.dentine
    guard !dent.isEmpty else { return nil }
    var c = SIMD2<Float>(0, 0)
    for p in dent { c += p.uv }
    c /= Float(dent.count)
    // The dentine point nearest the middle, on the buccal side the camera sees.
    let target: SurveyPoint = dent.min { simd_distance($0.uv, c + SIMD2<Float>(0, 0.3)) < simd_distance($1.uv, c + SIMD2<Float>(0, 0.3)) }!
    let f: ToothSurvey = survey[facetLabelTooth]
    let fdent: [SIMD2<Float>] = f.dentine.map { $0.uv }
    var best: SurveyPoint? = nil
    var bestGap: Float = -1
    for p in f.points where p.surface == 1 {
        var gap: Float = 1e9
        for q in fdent { gap = min(gap, simd_distance(p.uv, q)) }
        if gap > bestGap { bestGap = gap; best = p }
    }
    guard let facetPoint = best else { return nil }
    let k: ToothSurvey = survey[canineLabelTooth]
    let a: SIMD3<Float> = facetAnchorLocal(k.spec)
    let tip: SurveyPoint = k.worn.min { simd_distance($0.uv, SIMD2<Float>(a.x, a.y)) < simd_distance($1.uv, SIMD2<Float>(a.x, a.y)) }!
    return LabelTargets(dentine: d.world(target.uv, target.height),
                        facet: f.world(facetPoint.uv, facetPoint.height),
                        canine: k.world(tip.uv, tip.height))
}

// MARK: - the section inset

/// The cut: through the patient's left central incisor (#24, index 0), in the
/// labiolingual plane through the middle of its edge (u = 0) — where Smith &
/// Knight's incisal criteria are read.
let sectionToothIndex: Int = 0
/// What the cut spans, tooth-local: v (lingual − to labial +) and y, mm.
let sectionV: ClosedRange<Float> = -4.2...4.2
let sectionY: ClosedRange<Float> = -10.5...0.8

/// Tissue on a grid over the cut, worn and unworn: 0 air, 1 enamel, 2 dentine.
struct SectionGrid {
    var columns: Int
    var rows: Int
    var worn: [Float]
    var whole: [Float]
    func at(_ c: Int, _ r: Int, worn w: Bool) -> Float { (w ? worn : whole)[r * columns + c] }
}

func sampleSection(depth: Float, perMillimetre: Float, on device: MTLDevice) throws -> SectionGrid {
    let t: PlacedTooth = placeTeeth()[sectionToothIndex]
    let s = ToothSurvey(index: sectionToothIndex, tooth: t, points: [])
    let cols: Int = Int(((sectionV.upperBound - sectionV.lowerBound) * perMillimetre).rounded())
    let rows: Int = Int(((sectionY.upperBound - sectionY.lowerBound) * perMillimetre).rounded())
    var pts: [SIMD3<Float>] = []
    pts.reserveCapacity(cols * rows)
    for r in 0..<rows {
        let y: Float = sectionY.upperBound - (Float(r) + 0.5) / perMillimetre
        for c in 0..<cols {
            let v: Float = sectionV.lowerBound + (Float(c) + 0.5) / perMillimetre
            pts.append(s.world(SIMD2<Float>(0, v), y))
        }
    }
    let worn: [Float] = try sectionTooth(pts, tooth: sectionToothIndex, depth: depth, on: device)
    let whole: [Float] = try sectionTooth(pts, tooth: sectionToothIndex, depth: 0, on: device)
    return SectionGrid(columns: cols, rows: rows, worn: worn, whole: whole)
}

// MARK: - drawing

func ctLine(_ s: String, size: CGFloat, font: String, color: CGColor) -> CTLine {
    let f = CTFontCreateWithName(font as CFString, size, nil)
    let attributes: [NSAttributedString.Key: Any] = [
        NSAttributedString.Key(kCTFontAttributeName as String): f,
        NSAttributedString.Key(kCTForegroundColorAttributeName as String): color,
    ]
    return CTLineCreateWithAttributedString(NSAttributedString(string: s, attributes: attributes))
}

func lineWidth(_ line: CTLine) -> CGFloat { CGFloat(CTLineGetTypographicBounds(line, nil, nil, nil)) }

/// Text with its baseline at (x, y) in picture pixels (y down).
func put(_ s: String, _ ctx: CGContext, x: CGFloat, y: CGFloat, size: CGFloat, height: CGFloat,
         font: String = "HelveticaNeue", color: CGColor = ink) {
    let line: CTLine = ctLine(s, size: size, font: font, color: color)
    ctx.textPosition = CGPoint(x: x, y: height - y)
    CTLineDraw(line, ctx)
}

let ink = CGColor(srgbRed: 0.13, green: 0.14, blue: 0.16, alpha: 1)
let softInk = CGColor(srgbRed: 0.30, green: 0.31, blue: 0.34, alpha: 1)
let leader = CGColor(srgbRed: 0.10, green: 0.11, blue: 0.13, alpha: 0.9)
let paper = CGColor(srgbRed: 1, green: 1, blue: 1, alpha: 0.86)

/// A label: a few lines of text on a pale card, and a leader from the card to
/// a dot on the feature.
struct Label {
    var lines: [String]
    var card: CGPoint          // top-left of the card, picture pixels
    var target: SIMD2<Float>   // picture pixels
}

func drawLabel(_ l: Label, _ ctx: CGContext, k: CGFloat, height: CGFloat) {
    let size: CGFloat = 25 * k
    let small: CGFloat = 19 * k
    let pad: CGFloat = 12 * k
    var w: CGFloat = 0
    for (i, s) in l.lines.enumerated() {
        let lw: CGFloat = lineWidth(ctLine(s, size: i == 0 ? size : small, font: i == 0 ? "HelveticaNeue-Medium" : "HelveticaNeue", color: ink))
        w = max(w, lw)
    }
    let lineH: CGFloat = size * 1.25
    let smallH: CGFloat = small * 1.3
    let hgt: CGFloat = lineH + CGFloat(max(l.lines.count - 1, 0)) * smallH + pad * 1.2
    let box = CGRect(x: l.card.x, y: height - l.card.y - hgt, width: w + 2 * pad, height: hgt)
    // The leader: from the card's nearest edge to the dot.
    let tx: CGFloat = CGFloat(l.target.x)
    let ty: CGFloat = height - CGFloat(l.target.y)
    let sx: CGFloat = min(max(tx, box.minX), box.maxX)
    let sy: CGFloat = min(max(ty, box.minY), box.maxY)
    ctx.setLineCap(.round)
    ctx.setStrokeColor(CGColor(srgbRed: 1, green: 1, blue: 1, alpha: 0.8))
    ctx.setLineWidth(5 * k)
    ctx.move(to: CGPoint(x: sx, y: sy)); ctx.addLine(to: CGPoint(x: tx, y: ty)); ctx.strokePath()
    ctx.setStrokeColor(leader)
    ctx.setLineWidth(2 * k)
    ctx.move(to: CGPoint(x: sx, y: sy)); ctx.addLine(to: CGPoint(x: tx, y: ty)); ctx.strokePath()
    ctx.setFillColor(CGColor(srgbRed: 1, green: 1, blue: 1, alpha: 1))
    ctx.fillEllipse(in: CGRect(x: tx - 6 * k, y: ty - 6 * k, width: 12 * k, height: 12 * k))
    ctx.setFillColor(leader)
    ctx.fillEllipse(in: CGRect(x: tx - 3.5 * k, y: ty - 3.5 * k, width: 7 * k, height: 7 * k))
    ctx.setFillColor(paper)
    ctx.addPath(CGPath(roundedRect: box, cornerWidth: 8 * k, cornerHeight: 8 * k, transform: nil))
    ctx.fillPath()
    var y: CGFloat = l.card.y + pad * 0.6 + size
    for (i, s) in l.lines.enumerated() {
        put(s, ctx, x: l.card.x + pad, y: y, size: i == 0 ? size : small, height: height,
            font: i == 0 ? "HelveticaNeue-Medium" : "HelveticaNeue", color: i == 0 ? ink : softInk)
        y += i == 0 ? smallH + 2 * k : smallH
    }
}

/// sRGB display value of a CIELAB colour, for the inset's flat fills, lit as
/// brightly as the render lights the teeth's tops (MODEL: ×1.45 in linear
/// light, so the inset's enamel reads about as it does in the picture). Both
/// tissues get the same light, so their difference is the measured one.
let insetLight: Float = 1.45
func displayColour(_ lab: SIMD3<Double>, alpha: CGFloat = 1) -> CGColor {
    let lin: SIMD3<Float> = labToLinearSRGB(lab) * insetLight
    func enc(_ c: Float) -> CGFloat {
        let x: Float = min(max(c, 0), 1)
        let v: Float = x <= 0.0031308 ? 12.92 * x : 1.055 * pow(x, 1 / 2.4) - 0.055
        return CGFloat(v)
    }
    return CGColor(srgbRed: enc(lin.x), green: enc(lin.y), blue: enc(lin.z), alpha: alpha)
}

/// Where the inset sits, picture pixels at 1080 lines (scaled by k).
struct InsetFrame {
    var origin: CGPoint        // top-left of the section drawing
    var perMillimetre: CGFloat
}

func insetFrame(k: CGFloat) -> InsetFrame {
    InsetFrame(origin: CGPoint(x: 1640 * k, y: 58 * k), perMillimetre: 30 * k)
}

/// The cut through the worn incisor: enamel and dentine in their measured
/// colours, what grinding took away as a pale ghost inside the unworn
/// outline, the pulp's depth below the original edge, a 1 mm bar.
///
/// Step 65: `depth` is the frame's wear; notes appear only once what they name
/// exists.
func drawSection(_ g: SectionGrid, _ ctx: CGContext, frame f: InsetFrame, k: CGFloat, height: CGFloat,
                 depth: Float = finalWearDepth) {
    let px: CGFloat = 1   // one grid cell per picture pixel
    let enamel: CGColor = displayColour(middleThirdLab[.incisor]!)
    let dentine: CGColor = displayColour(dentineLab)
    let ghost = CGColor(srgbRed: 0.80, green: 0.86, blue: 0.93, alpha: 1)
    let w: CGFloat = CGFloat(g.columns) * px
    let h: CGFloat = CGFloat(g.rows) * px
    // A card behind it all, with the caption above.
    let card = CGRect(x: f.origin.x - 250 * k, y: height - f.origin.y - h - 64 * k, width: w + 272 * k, height: h + 110 * k)
    ctx.setFillColor(CGColor(srgbRed: 1, green: 1, blue: 1, alpha: 0.92))
    ctx.addPath(CGPath(roundedRect: card, cornerWidth: 10 * k, cornerHeight: 10 * k, transform: nil))
    ctx.fillPath()
    for r in 0..<g.rows {
        for c in 0..<g.columns {
            let now: Float = g.at(c, r, worn: true)
            let was: Float = g.at(c, r, worn: false)
            var col: CGColor? = nil
            if now == 1 { col = enamel } else if now == 2 { col = dentine } else if was > 0 { col = ghost }
            guard let colour = col else { continue }
            ctx.setFillColor(colour)
            ctx.fill(CGRect(x: f.origin.x + CGFloat(c) * px, y: height - f.origin.y - CGFloat(r + 1) * px, width: px, height: px))
        }
    }
    // The unworn outline over the ghost, dashed.
    ctx.setStrokeColor(CGColor(srgbRed: 0.35, green: 0.45, blue: 0.6, alpha: 1))
    ctx.setLineWidth(1.5 * k)
    for r in 1..<g.rows {
        for c in 1..<g.columns {
            let a: Bool = g.at(c, r, worn: false) > 0
            let edge: Bool = a != (g.at(c - 1, r, worn: false) > 0) || a != (g.at(c, r - 1, worn: false) > 0)
            let now: Bool = g.at(c, r, worn: true) > 0 || g.at(c - 1, r, worn: true) > 0 || g.at(c, r - 1, worn: true) > 0
            if edge && !now && (c + r) % 6 < 3 {
                ctx.fill(CGRect(x: f.origin.x + CGFloat(c) - 0.5, y: height - f.origin.y - CGFloat(r) - 0.5, width: 1.5 * k, height: 1.5 * k))
            }
        }
    }
    let mm: CGFloat = f.perMillimetre
    // Where rows and columns land, in picture pixels (y down).
    func yAt(_ y: Float) -> CGFloat { f.origin.y + CGFloat(sectionY.upperBound - y) * mm }
    func xAt(_ v: Float) -> CGFloat { f.origin.x + CGFloat(v - sectionV.lowerBound) * mm }
    // The pulp's depth below the unworn edge (Al-Zahawi 2023): a dashed line.
    let unwornTop: Float = crownTopCPU(placeTeeth()[sectionToothIndex].spec, SIMD2<Float>(0, 0))
    let pulpY: CGFloat = yAt(unwornTop - incisalEnamelToPulp)
    ctx.setStrokeColor(CGColor(srgbRed: 0.72, green: 0.2, blue: 0.2, alpha: 0.9))
    ctx.setLineWidth(1.5 * k)
    ctx.setLineDash(phase: 0, lengths: [6 * k, 5 * k])
    ctx.move(to: CGPoint(x: xAt(-1.6), y: height - pulpY)); ctx.addLine(to: CGPoint(x: xAt(1.6), y: height - pulpY))
    ctx.strokePath()
    ctx.setLineDash(phase: 0, lengths: [])
    // Labels down the left of the drawing, with short leaders.
    let size: CGFloat = 18 * k
    let lx: CGFloat = f.origin.x - 238 * k
    func note(_ s: String, _ y: CGFloat, to target: CGPoint, colour: CGColor = ink) {
        put(s, ctx, x: lx, y: y + 6 * k, size: size, height: height, color: colour)
        ctx.setStrokeColor(leader)
        ctx.setLineWidth(1.2 * k)
        let startX: CGFloat = lx + lineWidth(ctLine(s, size: size, font: "HelveticaNeue", color: ink)) + 6 * k
        ctx.move(to: CGPoint(x: startX, y: height - y)); ctx.addLine(to: CGPoint(x: target.x, y: height - target.y))
        ctx.strokePath()
    }
    let topNow: Float = unwornTop - depth
    // Dentine bared: some column whose topmost tissue is dentine.
    var bared: Bool = false
    for c in 0..<g.columns {
        if let r = (0..<g.rows).first(where: { g.at(c, $0, worn: true) > 0 }), g.at(c, r, worn: true) == 2 { bared = true }
    }
    if depth > 0.05 {
        note(String(format: "ground away, %.2f mm", depth), yAt(unwornTop + 0.2) - 4 * k,
             to: CGPoint(x: xAt(0.15), y: yAt(unwornTop - min(0.9, depth * 0.4))))
    }
    if bared {
        note("dentine, flush with the facet", yAt(topNow) + 22 * k, to: CGPoint(x: xAt(0.0), y: yAt(topNow - 0.3)))
    }
    let red = CGColor(srgbRed: 0.6, green: 0.15, blue: 0.15, alpha: 1)
    note("pulp 5.2 mm below the", pulpY - 10 * k, to: CGPoint(x: xAt(-1.6), y: pulpY), colour: red)
    put("original edge (not drawn)", ctx, x: lx, y: pulpY + 16 * k, size: size, height: height, color: red)
    // The labial enamel low on the crown, ~0.7 mm there (Al-Zahawi's 1–3 mm values).
    let enamelY: Float = -7.2
    note("enamel shell", yAt(enamelY), to: CGPoint(x: xAt(2.35), y: yAt(enamelY)))
    note("root: dentine, no enamel", yAt(-9.8), to: CGPoint(x: xAt(0.0), y: yAt(-9.8)))
    // Lingual and labial, under the drawing.
    let base: CGFloat = f.origin.y + CGFloat(g.rows) + 30 * k
    put("lingual", ctx, x: f.origin.x, y: base, size: 16 * k, height: height, font: "HelveticaNeue-Italic", color: softInk)
    let lab: CTLine = ctLine("labial", size: 16 * k, font: "HelveticaNeue-Italic", color: softInk)
    put("labial", ctx, x: f.origin.x + w - lineWidth(lab), y: base, size: 16 * k, height: height, font: "HelveticaNeue-Italic", color: softInk)
    // 1 mm bar.
    let barY: CGFloat = base + 4 * k
    let bx: CGFloat = f.origin.x + w / 2 - mm / 2
    ctx.setStrokeColor(ink)
    ctx.setLineWidth(3 * k)
    ctx.move(to: CGPoint(x: bx, y: height - barY)); ctx.addLine(to: CGPoint(x: bx + mm, y: height - barY)); ctx.strokePath()
    put("1 mm", ctx, x: bx + mm + 8 * k, y: barY + 6 * k, size: 16 * k, height: height, color: softInk)
    // The heading, above.
    put("Cut through the worn central incisor", ctx, x: lx, y: f.origin.y - 18 * k, size: 22 * k, height: height,
        font: "HelveticaNeue-Medium")
}

/// The whole overlay onto a rendered image's pixels.
func drawStillOverlay(_ image: MouthImage, targets t: LabelTargets, section: SectionGrid) {
    let width: Int = image.width
    let height: Int = image.height
    let k: CGFloat = CGFloat(height) / 1080
    let h: CGFloat = CGFloat(height)
    guard let ctx = CGContext(data: image.pixels.contents(), width: width, height: height, bitsPerComponent: 8,
                              bytesPerRow: width * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                              bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return }
    let cam: Camera = stillCamera
    let pd: SIMD2<Float> = cam.project(t.dentine, width: width, height: height)
    let pf: SIMD2<Float> = cam.project(t.facet, width: width, height: height)
    let pc: SIMD2<Float> = cam.project(t.canine, width: width, height: height)
    let labels: [Label] = [
        Label(lines: ["Canine tip ground flat",
                      String(format: "%.2f mm off the tip; the crown %.2f mm shorter", finalWearDepth, canineHeightLoss)],
              card: CGPoint(x: 60 * k, y: 880 * k), target: pc),
        Label(lines: ["Exposed dentine",
                      "yellower than enamel; flush with the facet —",
                      "grinding leaves it flat, not cupped"],
              card: CGPoint(x: 700 * k, y: 860 * k), target: pd),
        Label(lines: ["Wear facets",
                      "flat, shiny, sharp-edged, each matched by one",
                      "on the upper tooth that ground against it"],
              card: CGPoint(x: 1300 * k, y: 880 * k), target: pf),
    ]
    for l in labels { drawLabel(l, ctx, k: k, height: h) }
    // The caption, top left.
    put("Bruxism, ~\(Int(finalWearYears.rounded())) years on", ctx, x: 48 * k, y: 70 * k, size: 40 * k, height: h,
        font: "HelveticaNeue-Medium")
    let lines: [String] = [
        String(format: "%.2f mm ground off every lower tooth at %.0f µm a year, the rate", finalWearDepth, bruxistWearRate * 1000),
        "measured in untreated night-time grinders (Korkut et al. 2020);",
        String(format: "ordinary chewing wears %.0f–%.0f µm a year (Lambrechts et al. 1989).",
               normalPremolarWearRate * 1000, normalMolarWearRate * 1000),
        "Smith & Knight Tooth Wear Index: 3 on incisal edges and canines,",
        "2 on premolars and molars. Same teeth, camera and light as step 20.",
    ]
    var y: CGFloat = 112 * k
    for s in lines {
        put(s, ctx, x: 50 * k, y: y, size: 21 * k, height: h, color: softInk)
        y += 29 * k
    }
    drawSection(section, ctx, frame: insetFrame(k: k), k: k, height: h)
}
