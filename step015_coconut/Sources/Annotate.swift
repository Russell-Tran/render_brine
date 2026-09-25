// Labels on the picture, because a cutaway with no labels is a photograph of a
// broken coconut.
//
// The house caption bar comes from step 13 unchanged. What this file adds is
// the part a section drawing needs and a micrograph does not: leaders to the
// layers, so the six shells of the pericarp and the seed can be named where
// they are rather than in a legend somebody has to map back onto the picture.
//
// Every anchor below is a WORLD point run through the same camera the kernel
// uses. None of them is a hand-placed pixel, so none of them can drift away
// from what is drawn underneath it.

import CoreGraphics
import CoreText
import Foundation
import Metal
import simd

private let plate = CGColor(srgbRed: 0.04, green: 0.07, blue: 0.10, alpha: 0.74)
private let leaderInk = CGColor(srgbRed: 0.82, green: 0.87, blue: 0.90, alpha: 0.85)
private let labelInk = CGColor(srgbRed: 0.93, green: 0.96, blue: 0.97, alpha: 1)
private let labelKey = CGColor(srgbRed: 0.55, green: 0.88, blue: 0.90, alpha: 1)

private func annotationContext(_ buffer: MTLBuffer, _ layout: FrameLayout) -> CGContext? {
    CGContext(data: buffer.contents(), width: layout.width, height: layout.height,
              bitsPerComponent: 8, bytesPerRow: layout.width * 4,
              space: CGColorSpace(name: CGColorSpace.sRGB)!,
              bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
}

/// Where one label goes: the text, the world point it points at, and which side
/// of that point the plate sits on.
struct Annotation {
    var text: String
    var anchor: SIMD3<Float>
    var at: CGPoint          // the plate's own corner, in view pixels
    var leftOfAnchor: Bool
    var key: Bool = false    // the labels the render is actually about
}

/// Where the boundary between layer `i` and layer `i + 1` sits, measured along a
/// ray from the nut's centre in direction `dir`, on the cut plane. The layers
/// are concentric ellipsoids, so this is one square root and no search.
func layerRadius(semi: SIMD3<Float>, dir: SIMD2<Float>, z: Float) -> Float? {
    let zz: Float = z / semi.z
    let left: Float = 1 - zz * zz
    if left <= 0 { return nil }
    let dx: Float = dir.x / semi.x
    let dy: Float = dir.y / semi.y
    let den: Float = dx * dx + dy * dy
    if den <= 0 { return nil }
    return (left / den).squareRoot()
}

/// A point halfway through layer `m`'s own thickness, along `dir`, on the cut
/// plane — which is exactly where a leader should land.
func layerAnchor(_ m: Material, budget: Budget, dir: SIMD2<Float>) -> SIMD3<Float> {
    let outer: SIMD3<Float> = layerSemi(m, budget: budget, breakage: [])
    let rOuter: Float = layerRadius(semi: outer, dir: dir, z: cutPlaneZ) ?? 0
    var rInner: Float = 0
    if let index = layerStack.firstIndex(of: m), index + 1 < layerStack.count {
        let next: Material = layerStack[index + 1]
        let inner: SIMD3<Float> = layerSemi(next, budget: budget, breakage: [])
        rInner = layerRadius(semi: inner, dir: dir, z: cutPlaneZ) ?? 0
    }
    let r: Float = (rOuter + rInner) * 0.5
    return nutCentre + SIMD3(dir.x * r, dir.y * r, cutPlaneZ)
}

/// Draws one leader: a short horizontal stub off the plate, a straight run to
/// the anchor, and a dot on the anchor itself.
private func drawLeader(_ ctx: CGContext, from: CGPoint, to: CGPoint, height: CGFloat,
                        scale k: CGFloat) {
    let stub: CGFloat = from.x < to.x ? 8 * k : -8 * k
    ctx.setStrokeColor(leaderInk)
    ctx.setLineWidth(1.1 * k)
    ctx.beginPath()
    ctx.move(to: CGPoint(x: from.x, y: height - from.y))
    ctx.addLine(to: CGPoint(x: from.x + stub, y: height - from.y))
    ctx.addLine(to: CGPoint(x: to.x, y: height - to.y))
    ctx.strokePath()
    ctx.setFillColor(leaderInk)
    ctx.fillEllipse(in: CGRect(x: to.x - 2.1 * k, y: height - to.y - 2.1 * k,
                               width: 4.2 * k, height: 4.2 * k))
}

func drawAnnotations(_ items: [Annotation], into buffer: MTLBuffer, layout: FrameLayout,
                     camera: Camera) {
    guard let ctx = annotationContext(buffer, layout) else { return }
    let k = layout.scale
    let h = CGFloat(layout.height)
    let size: CGFloat = 10.5 * k
    let padX: CGFloat = 5 * k
    let padY: CGFloat = 3 * k
    for item in items {
        let w: CGFloat = textWidth(item.text, size: size, bold: item.key)
        let boxW: CGFloat = w + padX * 2
        let boxH: CGFloat = size + padY * 2
        let x: CGFloat = item.leftOfAnchor ? item.at.x - boxW : item.at.x
        let y: CGFloat = item.at.y
        let projected: SIMD2<Float> = camera.project(item.anchor, width: layout.width,
                                                     height: layout.viewHeight)
        let target = CGPoint(x: CGFloat(projected.x), y: CGFloat(projected.y))
        let edge = CGPoint(x: item.leftOfAnchor ? x : x + boxW, y: y + boxH / 2)
        drawLeader(ctx, from: edge, to: target, height: h, scale: k)
        ctx.setFillColor(plate)
        ctx.fill(CGRect(x: x, y: h - y - boxH, width: boxW, height: boxH))
        drawLabel(item.text, into: buffer, layout: layout, x: x + padX, top: y + padY,
                  size: size, bold: item.key, color: item.key ? labelKey : labelInk)
    }
}

/// The soil line, drawn as a hairline right across the frame so the viewer can
/// see that the nut really is half buried rather than resting on the ground.
func drawSoilLine(into buffer: MTLBuffer, layout: FrameLayout, camera: Camera) {
    guard let ctx = annotationContext(buffer, layout) else { return }
    let k = layout.scale
    let h = CGFloat(layout.height)
    let p: SIMD2<Float> = camera.project(SIMD3(cutawayTarget.x, soilSurfaceY, cutPlaneZ),
                                         width: layout.width, height: layout.viewHeight)
    let y = CGFloat(p.y)
    ctx.setStrokeColor(CGColor(srgbRed: 0.85, green: 0.89, blue: 0.91, alpha: 0.40))
    ctx.setLineWidth(1.0 * k)
    ctx.setLineDash(phase: 0, lengths: [6 * k, 6 * k])
    ctx.beginPath()
    ctx.move(to: CGPoint(x: 0, y: h - y))
    ctx.addLine(to: CGPoint(x: CGFloat(layout.width), y: h - y))
    ctx.strokePath()
    ctx.setLineDash(phase: 0, lengths: [])
}

/// A scale rule in pale ink, low on the left where the soil is dark. Step 13's
/// scale bar is drawn in dark ink because a brightfield background is a lamp;
/// here the background is soil and sky, so the ink has to go the other way.
/// Its length is worked out from the same millimetres-per-pixel the camera
/// uses, so it cannot drift from the render.
func drawScaleRule(into buffer: MTLBuffer, layout: FrameLayout, millimetres: Float) {
    guard let ctx = annotationContext(buffer, layout) else { return }
    let k = layout.scale
    let h = CGFloat(layout.height)
    let length = CGFloat(millimetres / mmPerPixel)
    let x: CGFloat = 18 * k
    let top: CGFloat = CGFloat(layout.viewHeight) - 34 * k
    let ink = CGColor(srgbRed: 0.90, green: 0.93, blue: 0.95, alpha: 0.88)
    ctx.setFillColor(ink)
    ctx.fill(CGRect(x: x, y: h - top, width: length, height: 2.5 * k))
    ctx.fill(CGRect(x: x, y: h - top - 4 * k, width: 2 * k, height: 10.5 * k))
    ctx.fill(CGRect(x: x + length - 2 * k, y: h - top - 4 * k, width: 2 * k, height: 10.5 * k))
    let label = String(format: "%.0f mm", millimetres)
    let w = textWidth(label, size: 11 * k, bold: true)
    drawLabel(label, into: buffer, layout: layout, x: x + length / 2 - w / 2,
              top: top - 19 * k, size: 11 * k, bold: true, color: ink)
}

/// The annotation set for one frame. Labels appear when the thing they name
/// does, and every anchor is recomputed from the frame's own geometry.
func coconutAnnotations(plant: Plant, layout: FrameLayout) -> [Annotation] {
    let k = CGFloat(layout.scale)
    let h = CGFloat(layout.viewHeight)
    let right: CGFloat = CGFloat(layout.width) - 14 * k
    var out: [Annotation] = []

    // The layer stack, up the left of the frame, leaders fanning into the upper
    // left quarter of the cut face so they never cross.
    let names: [(Material, String)] = [
        (.exocarp, "exocarp — skin"),
        (.mesocarp, "mesocarp — husk, coir"),
        (.endocarp, "endocarp — shell"),
        (.testa, "testa — seed coat"),
        (.endosperm, "solid endosperm — the meat"),
        (.water, "liquid endosperm — coconut water"),
    ]
    let bearings: [Float] = [126, 141, 156, 169, 182, 196]
    for (i, entry) in names.enumerated() {
        let a: Float = radians(bearings[i])
        let dir = SIMD2<Float>(cos(a), sin(a))
        let anchor: SIMD3<Float> = layerAnchor(entry.0, budget: plant.budget, dir: dir)
        let top: CGFloat = h * (0.085 + CGFloat(i) * 0.031)
        out.append(Annotation(text: entry.1, anchor: anchor,
                              at: CGPoint(x: 14 * k, y: top), leftOfAnchor: false))
    }

    // The interior story.
    if plant.budget.fill > 0.02 {
        let anchor: SIMD3<Float> = plant.haustoriumCentre
            + SIMD3(0, plant.haustoriumSemi.y * 0.35, cutPlaneZ * 0.0)
        out.append(Annotation(text: "haustorium — the swollen cotyledon",
                              anchor: SIMD3(anchor.x, anchor.y, min(anchor.z, cutPlaneZ - 1)),
                              at: CGPoint(x: right, y: h * 0.455),
                              leftOfAnchor: true, key: true))
    }
    let poreDir: SIMD3<Float> = simd_normalize(poreCentre(functionalPore) - nutCentre)
    let embryo: SIMD3<Float> = nutCentre + poreDir * 64
    out.append(Annotation(text: "embryo, under the one soft eye", anchor: embryo,
                          at: CGPoint(x: right, y: h * 0.505),
                          leftOfAnchor: true))
    if plant.stages.petiole > 0.25 {
        let mid: SIMD3<Float> = (poreCentre(functionalPore) + sproutBase) * 0.5
        out.append(Annotation(text: "cotyledonary petiole, through the pore", anchor: mid,
                              at: CGPoint(x: right, y: h * 0.555),
                              leftOfAnchor: true))
    }

    // Outside.
    if plant.stages.leaf > 0.30 {
        let tip = SIMD3<Float>(sproutBase.x + 60, plant.shootTipY - 30, sproutBase.z)
        out.append(Annotation(text: "first true leaf — entire, not pinnate", anchor: tip,
                              at: CGPoint(x: right, y: h * 0.055),
                              leftOfAnchor: true, key: true))
    }
    if plant.stages.root > 0.35 {
        let fan = SIMD3<Float>(sproutBase.x + 6, -85, cutPlaneZ)
        out.append(Annotation(text: "adventitious roots — no taproot", anchor: fan,
                              at: CGPoint(x: right, y: h * 0.815),
                              leftOfAnchor: true, key: true))
    }
    return out
}
