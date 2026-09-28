// Step 65's marks, frame by frame: the year and a time bar, labels that come
// up as each feature forms (the facets' label carries the key to the newly
// worn tint), the cut through the incisor as it wears, and — during the
// dissolve — a note that the loop is starting over, not the teeth growing back.

import CoreGraphics
import CoreText
import Foundation
import Metal
import simd

/// Where each label points this frame: the survey at this frame's depth, and
/// the point nearest where the feature will be at the end, so each dot sits on
/// its feature from the moment it appears and slides only as the feature grows.
struct FrameTargets {
    var facet: SIMD3<Float>?
    var incisorDentine: SIMD3<Float>?
    var premolarDentine: SIMD3<Float>?
    var canine: SIMD3<Float>?
}

/// The points the labels aim for, in each tooth's frame, from the final survey.
struct TargetAims {
    var facet: SIMD2<Float>
    var incisorDentine: SIMD2<Float>
    var premolarDentine: SIMD2<Float>
}

func targetAims(_ final: [ToothSurvey]) -> TargetAims? {
    guard let t = labelTargets(final) else { return nil }
    func local(_ s: ToothSurvey, _ p: SIMD3<Float>) -> SIMD2<Float> {
        let rel = SIMD2<Float>(p.x - s.tooth.centre.x, p.z - s.tooth.centre.y)
        return SIMD2<Float>(simd_dot(rel, s.tooth.tangent), simd_dot(rel, s.tooth.outward))
    }
    let inc: [SurveyPoint] = final[incisorLabelTooth].dentine
    guard !inc.isEmpty else { return nil }
    var c = SIMD2<Float>(0, 0)
    for p in inc { c += p.uv }
    c /= Float(inc.count)
    return TargetAims(facet: local(final[facetLabelTooth], t.facet), incisorDentine: c,
                      premolarDentine: local(final[dentineLabelTooth], t.dentine))
}

func frameTargets(_ s: [ToothSurvey], aims: TargetAims) -> FrameTargets {
    func nearest(_ t: ToothSurvey, _ aim: SIMD2<Float>, _ surface: Int) -> SIMD3<Float>? {
        var best: SurveyPoint? = nil
        var bestDistance: Float = Float.greatestFiniteMagnitude
        for p in t.points where p.surface == surface {
            let d: Float = simd_distance(p.uv, aim)
            if d < bestDistance { bestDistance = d; best = p }
        }
        guard let p = best else { return nil }
        return t.world(p.uv, p.height)
    }
    let k: ToothSurvey = s[canineLabelTooth]
    let a: SIMD3<Float> = facetAnchorLocal(k.spec)
    let tip: SurveyPoint? = k.worn.min { simd_distance($0.uv, SIMD2<Float>(a.x, a.y)) < simd_distance($1.uv, SIMD2<Float>(a.x, a.y)) }
    return FrameTargets(facet: nearest(s[facetLabelTooth], aims.facet, 1),
                        incisorDentine: nearest(s[incisorLabelTooth], aims.incisorDentine, 2),
                        premolarDentine: nearest(s[dentineLabelTooth], aims.premolarDentine, 2),
                        canine: tip.map { k.world($0.uv, $0.height) })
}

/// How opaque a label is, `years` after its feature appeared: it fades in over
/// about a second of the loop.
func labelAlpha(years: Float, appears: Float) -> CGFloat {
    let perSecond: Float = finalWearYears / wearSeconds
    return CGFloat(smooth((years - appears) / perSecond))
}

/// The marks for one frame.
func drawFrameOverlay(_ image: MouthImage, plan p: FramePlan, times: FeatureTimes, targets t: FrameTargets,
                      section: SectionGrid) {
    let width: Int = image.width
    let height: Int = image.height
    let k: CGFloat = CGFloat(height) / 1080
    let h: CGFloat = CGFloat(height)
    guard let ctx = CGContext(data: image.pixels.contents(), width: width, height: height, bitsPerComponent: 8,
                              bytesPerRow: width * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                              bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return }
    let cam: Camera = p.camera
    func at(_ q: SIMD3<Float>?) -> SIMD2<Float>? { q.map { cam.project($0, width: width, height: height) } }
    // Labels, as their features form.
    var labels: [(Label, CGFloat)] = []
    if let q = at(t.facet) {
        labels.append((Label(lines: ["Wear facets", "flat, shiny, sharp-edged: ground by",
                                     "the upper teeth, each matched by one;",
                                     String(format: "blue: ground away in the last %.0f years", glowYears)],
                             card: CGPoint(x: 1330 * k, y: 870 * k), target: q),
                       labelAlpha(years: p.years, appears: times.facets)))
    }
    if let q = at(t.incisorDentine) {
        labels.append((Label(lines: ["Dentine bared first on the incisal edges",
                                     String(format: "year %.0f: 1.0 mm of enamel there", times.incisorDentine)],
                             card: CGPoint(x: 50 * k, y: 560 * k), target: q),
                       labelAlpha(years: p.years, appears: times.incisorDentine)))
    }
    if let q = at(t.canine) {
        labels.append((Label(lines: ["Canine tip ground flat",
                                     String(format: "%.2f mm off its tip so far", p.depth)],
                             card: CGPoint(x: 50 * k, y: 880 * k), target: q),
                       labelAlpha(years: p.years, appears: times.canineFlat)))
    }
    if let q = at(t.premolarDentine) {
        labels.append((Label(lines: ["Dentine on the premolars",
                                     String(format: "year %.0f, under thicker cusp enamel", times.premolarDentine),
                                     "flush with the facet, not cupped"],
                             card: CGPoint(x: 700 * k, y: 870 * k), target: q),
                       labelAlpha(years: p.years, appears: times.premolarDentine)))
    }
    for (l, alpha) in labels where alpha > 0.01 {
        ctx.saveGState()
        ctx.setAlpha(alpha)
        drawLabel(l, ctx, k: k, height: h)
        ctx.restoreGState()
    }
    // The clock and the time bar, top left.
    put(String(format: "Bruxism: %.0f years of night-time grinding in %.0f s", finalWearYears, wearSeconds),
        ctx, x: 48 * k, y: 66 * k, size: 40 * k, height: h, font: "HelveticaNeue-Medium")
    put(String(format: "year %2.0f", p.years), ctx, x: 48 * k, y: 146 * k, size: 64 * k, height: h,
        font: "HelveticaNeue-Bold")
    put(String(format: "%.2f mm ground off", p.depth), ctx, x: 330 * k, y: 146 * k, size: 30 * k, height: h,
        color: softInk)
    let barX: CGFloat = 50 * k
    let barW: CGFloat = 560 * k
    let barY: CGFloat = h - 178 * k
    ctx.setFillColor(CGColor(srgbRed: 0.85, green: 0.86, blue: 0.88, alpha: 1))
    ctx.fill(CGRect(x: barX, y: barY, width: barW, height: 10 * k))
    ctx.setFillColor(ink)
    ctx.fill(CGRect(x: barX, y: barY, width: barW * CGFloat(p.years / finalWearYears), height: 10 * k))
    for yr in stride(from: 0, through: Int(finalWearYears.rounded()), by: 10) {
        let x: CGFloat = barX + barW * CGFloat(Float(yr) / finalWearYears)
        ctx.fill(CGRect(x: x - 1 * k, y: barY - 6 * k, width: 2 * k, height: 22 * k))
        put("\(yr)", ctx, x: x - 8 * k, y: 212 * k, size: 17 * k, height: h, color: softInk)
    }
    // One line of where the rate comes from, above the incisors.
    put(String(format: "%.0f µm/yr in untreated grinders (Korkut 2020); chewing alone %.0f–%.0f (Lambrechts 1989)",
               bruxistWearRate * 1000, normalPremolarWearRate * 1000, normalMolarWearRate * 1000),
        ctx, x: 50 * k, y: 244 * k, size: 17 * k, height: h, color: softInk)
    drawSection(section, ctx, frame: insetFrame(k: k), k: k, height: h, depth: p.depth)
}

/// During the dissolve: the loop starting over, said plainly, on top.
func drawRestart(_ image: MouthImage, strength: Float) {
    let width: Int = image.width
    let height: Int = image.height
    let k: CGFloat = CGFloat(height) / 1080
    let h: CGFloat = CGFloat(height)
    guard let ctx = CGContext(data: image.pixels.contents(), width: width, height: height, bitsPerComponent: 8,
                              bytesPerRow: width * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                              bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return }
    let lines: [String] = ["↺  Restarting at year 0",
                           "the loop starts over; worn teeth never grow back"]
    let w: CGFloat = 640 * k
    let box = CGRect(x: CGFloat(width) / 2 - w / 2, y: h - 560 * k, width: w, height: 110 * k)
    ctx.setAlpha(CGFloat(min(strength * 3, 1)))
    ctx.setFillColor(CGColor(srgbRed: 1, green: 1, blue: 1, alpha: 0.94))
    ctx.addPath(CGPath(roundedRect: box, cornerWidth: 12 * k, cornerHeight: 12 * k, transform: nil))
    ctx.fillPath()
    put(lines[0], ctx, x: box.minX + 30 * k, y: 500 * k, size: 38 * k, height: h, font: "HelveticaNeue-Medium")
    put(lines[1], ctx, x: box.minX + 30 * k, y: 538 * k, size: 22 * k, height: h, color: softInk)
}
