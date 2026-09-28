// A round inset: one of the ladybirds magnified, at the same moment, seen from
// the film's own camera direction — so its six legs, its tripod gait and its
// seven spots can be seen. At true scale in the main picture a ladybird is
// about ten pixels long and its legs are under half a pixel across.
//
// Which ladybird: the lowest one in view above a hand-over line; when it
// passes the line the inset crossfades to the next one up. The choice is a
// function of the frame's hour alone, so a loop later it is the same choice,
// one loop higher, and the loop still closes (a test renders it).

import Foundation
import simd

/// The inset's place and size in a frame `height` pixels high (480 × 624 at
/// full size): a disc on the right, below the captions. MODEL layout.
struct InsetLayout {
    var centre: SIMD2<Float>
    var radius: Float
    /// The field of view across the disc, mm, and the inset camera's distance.
    var fieldHalf: Float = 5.5
    var distance: Float = 60

    init(width: Int, height: Int) {
        let k: Float = Float(height) / 624
        centre = SIMD2<Float>(Float(width) - 78 * k, 190 * k)
        radius = 64 * k
    }
}

/// The screen line past which the inset hands over to the next ladybird up,
/// as a fraction of the frame's height, and the crossfade's width.
let insetHandOver: Float = 0.84
let insetFade: Float = 0.02

/// The ladybirds the inset shows at hour t, with their weights (summing to
/// 1, or empty if none is in view).
func insetChoice(_ t: Float, width: Int, height: Int) -> [(bird: Ladybird, weight: Float)] {
    let cam: Camera = camera(t)
    let h: Float = Float(height)
    let top: Float = 0.12 * h
    var shown: [(Ladybird, Float)] = []
    for lb in ladybirdsInView(t, .none) {
        let p: SIMD2<Float> = cam.project(lb.pose.origin, width: width, height: height)
        if p.x > 0 && p.x < Float(width) && p.y > top && p.y < insetHandOver * h { shown.append((lb, p.y)) }
    }
    shown.sort { $0.1 > $1.1 }
    guard let low = shown.first else { return [] }
    // Crossfade to the next one up as the lowest nears the line.
    let w: Float = smoothstep(insetHandOver * h, (insetHandOver - insetFade) * h, low.1)
    if w < 1, shown.count > 1 {
        return [(low.0, w), (shown[1].0, 1 - w)]
    }
    return [(low.0, 1)]
}

/// The camera the inset looks through for one ladybird: from the film's
/// camera direction, close in.
func insetCamera(_ lb: Ladybird, _ t: Float, _ layout: InsetLayout) -> Camera {
    let c: V3 = lb.pose.origin + lb.pose.up * 1.3
    let toward: V3 = simd_normalize(camera(t).position - c)
    return Camera(position: c + toward * layout.distance, target: c, tanHalf: layout.fieldHalf / layout.distance)
}

/// How many times larger the inset draws the ladybird than the main picture.
func insetMagnification(_ lb: Ladybird, _ t: Float, _ layout: InsetLayout, height: Int) -> Float {
    let main: Float = camera(t).millimetresPerPixel(at: lb.pose.origin, height: height)
    let inset: Float = 2 * layout.fieldHalf / (2 * layout.radius)
    return main / inset
}

let insetBackground = V3(0.83, 0.855, 0.84)

/// The frame with the inset drawn in, and its caption and leader lines.
func withInset(_ f: Frame, _ t: Float, scene: PlantScene, renderer: PlantRenderer, samples: Int)
    throws -> (frame: Frame, captions: [Caption]) {
    let layout = InsetLayout(width: f.width, height: f.height)
    let choice = insetChoice(t, width: f.width, height: f.height)
    guard !choice.isEmpty else { return (f, []) }
    let d: Int = Int(2 * layout.radius) + 2
    var blend = [SIMD4<Float>](repeating: SIMD4<Float>(0, 0, 0, 0), count: d * d)
    for (lb, w) in choice {
        let cam: Camera = insetCamera(lb, t, layout)
        let (img, _) = try renderer.render(scene, camera: cam, width: d, height: d, samples: samples, tubeMin: 0)
        for i in 0..<(d * d) { blend[i] += img.rgba[i] * w }
    }
    var px: [SIMD4<Float>] = f.rgba
    let x0: Int = Int(layout.centre.x - Float(d) / 2)
    let y0: Int = Int(layout.centre.y - Float(d) / 2)
    let bg: V3 = insetBackground
    let ring: V3 = V3(0.16, 0.20, 0.18)
    for j in 0..<d {
        for i in 0..<d {
            let x: Int = x0 + i
            let y: Int = y0 + j
            guard x >= 0, x < f.width, y >= 0, y < f.height else { continue }
            let r: Float = simd_distance(SIMD2<Float>(Float(x) + 0.5, Float(y) + 0.5), layout.centre)
            let inside: Float = min(max(layout.radius - r + 0.5, 0), 1)
            if inside <= 0 && r > layout.radius + 1.8 { continue }
            let q: SIMD4<Float> = blend[j * d + i]
            let c: V3 = V3(q.x, q.y, q.z) + bg * (1 - q.w)
            let k: Int = y * f.width + x
            let old: SIMD4<Float> = px[k]
            var mixed: SIMD4<Float> = old * (1 - inside) + SIMD4<Float>(c.x, c.y, c.z, 1) * inside
            // A thin ring round the disc.
            let edge: Float = max(0, 1 - abs(r - layout.radius) / 1.1)
            mixed = mixed * (1 - edge * 0.7) + SIMD4<Float>(ring.x, ring.y, ring.z, 1) * (edge * 0.7)
            px[k] = mixed
        }
    }
    // The label, and a leader from it to the ladybird in the main picture.
    let k: Float = Float(f.height) / 624
    let cam: Camera = camera(t)
    var caps: [Caption] = []
    let main: Ladybird = choice.max { $0.weight < $1.weight }!.bird
    let mag: Float = insetMagnification(main, t, layout, height: f.height)
    for (lb, w) in choice where w > 0.02 {
        let p: SIMD2<Float> = cam.project(lb.pose.origin + lb.pose.up * 1.2, width: f.width, height: f.height)
        caps.append(Caption(text: "", x: layout.centre.x - layout.radius * 0.72, baseline: layout.centre.y + layout.radius * 0.72,
                            size: 1, alpha: w, pointAt: p))
    }
    caps.append(Caption(text: String(format: "the same ladybird, ×%.0f", mag), x: layout.centre.x,
                        baseline: layout.centre.y + layout.radius + 15 * k, size: 12 * k, italic: true, centred: true))
    return (Frame(width: f.width, height: f.height, rgba: px, seen: f.seen), caps)
}
