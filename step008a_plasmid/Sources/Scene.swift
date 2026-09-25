// Turning the plasmid into shapes the ray tracer can draw, at three levels of
// detail, each appearing exactly where it stops being sub-pixel.
//
// At full-ring scale the circle is 5,813 Å across. Drawn 900 px wide that is
// 6.5 Å per pixel, so:
//
//     the duplex                    20 Å    3.1 px
//     one base-pair step           3.4 Å    0.5 px
//     a carbon atom (van der Waals) 3.4 Å   0.5 px
//
// Space-filling at that scale would mean 338,373 spheres, each covering half a
// pixel, to produce a three-pixel tube: enormously expensive, and it would
// sparkle rather than resolve. That is why every published plasmid picture is
// a smooth tube — not an artistic simplification but what space-filling
// actually converges to at that resolution. Ten times closer, the duplex is
// 31 px and an atom is 5 px, and space-filling becomes the most informative
// thing to draw, because the major and minor grooves turn into real surface.
// Closer still, the spheres merge into a wall and ball-and-stick wins.
//
// So the dive crosses two thresholds, both picked from that arithmetic and
// not by eye:
//
//   tube → space-filling   camera 2,100 Å → 950 Å   (an atom goes 1.8 → 4.0 px)
//   space-filling → sticks camera   240 Å → 110 Å   (a bond goes 1.5 → 3.3 px)
//
// The scene is built once and never moves. Per frame only the camera changes
// (and the lights, which turn with it, so a still ring can be lit as though it
// were the ring turning) and each shape's radius, which cross-fades the tiers.
// Radius is the only thing allowed to vary, because the grid was built from
// the shapes at their largest and so stays correct for anything smaller.

import Foundation
import simd

/// Ink, not measurement: positions are the plasmid's real geometry, but how
/// thick to draw something is a choice, as in any molecular model.
///
/// At ring scale the duplex is 2.4 px across, and a two-pixel cord blends into
/// the background before its color can be read — rendering it proved that more
/// convincingly than arithmetic did. So the tube is drawn 3.4× life in
/// cross-section and thins to true scale as the camera comes in, before any
/// atom appears. Every published plasmid figure exaggerates this; this one
/// says by how much, in the caption, and stops doing it once it no longer
/// needs to.
let duplexRadius: Float = 10.0            // true: 20 Å across
let tubeExaggeration: Float = 3.4
let atomBondRadius: Float = 0.16
let hbondRadius: Float = 0.055
let dashLength: Float = 0.22
let dashGap: Float = 0.16

/// Van der Waals radii (Å), Bondi, *J. Phys. Chem.* 68:441 (1964) — the size
/// an atom actually occupies, which is what a space-filling model draws.
func vanDerWaalsRadius(_ element: String) -> Float {
    switch element {
    case "C": return 1.70
    case "N": return 1.55
    case "O": return 1.52
    case "P": return 1.80
    default:  return 1.20    // H
    }
}

/// Camera distances (Å) where one representation gives way to the next.
let spaceFillStart: Float = 2100
let spaceFillFull: Float = 950
let ballStart: Float = 240
let ballFull: Float = 110

/// How far through the three tiers a given camera distance is: 0 is all tube,
/// 1 all space-filling, 2 all ball-and-stick.
func detailLevel(cameraDistance d: Float) -> Float {
    let toSpace = 1 - smoothstep01((d - spaceFillFull) / (spaceFillStart - spaceFillFull))
    let toBall = 1 - smoothstep01((d - ballFull) / (ballStart - ballFull))
    return toSpace + toBall
}

struct PlasmidScene {
    var shapes: [GPUShape]
    var tubeRadius: [Float]        // each shape's radius in tier 1
    var spaceRadius: [Float]       // ... in tier 2
    var ballRadius: [Float]        // ... in tier 3
    var tubeCount: Int             // shapes[0..<tubeCount) are the ring itself
    var atomWindow: Range<Int>
    var atomAnchor: [Int: Int] = [:]   // base pair -> where its atoms begin

    var atomCount: Int { shapes.count - tubeCount }
    /// The largest each shape ever gets, which is what the grid must be built from.
    var widestRadius: [Float] {
        var out = [Float](repeating: 0, count: shapes.count)
        for i in 0..<shapes.count {
            let a: Float = tubeRadius[i]
            let b: Float = spaceRadius[i]
            let c: Float = ballRadius[i]
            out[i] = max(a, max(b, c))
        }
        return out
    }

    /// Cross-fades the tiers in place. A shape at radius 0 is skipped by the
    /// kernel, so a tier costs nothing while it is invisible.
    mutating func setDetail(_ level: Float) {
        let d = min(max(level, 0), 2)
        for i in 0..<shapes.count {
            let r: Float
            if d <= 1 {
                r = tubeRadius[i] + (spaceRadius[i] - tubeRadius[i]) * d
            } else {
                r = spaceRadius[i] + (ballRadius[i] - spaceRadius[i]) * (d - 1)
            }
            shapes[i].a.w = r
        }
    }

    /// The scene with every shape at full size, for building the grid.
    func widest() -> [GPUShape] {
        let r = widestRadius
        return (0..<shapes.count).map { i -> GPUShape in
            var s = shapes[i]
            s.a.w = r[i]
            return s
        }
    }
}

extension Plasmid {
    /// The two-letter template key for the base pair at index `i`.
    func templateKey(at i: Int) -> String {
        switch sequence[i] {
        case UInt8(ascii: "A"): return "AT"
        case UInt8(ascii: "T"): return "TA"
        case UInt8(ascii: "G"): return "GC"
        default:                return "CG"
        }
    }

    /// The color of base pair `i`: its feature's, or neutral grey.
    func color(at i: Int) -> SIMD3<Float> {
        let f = featureAt[i]
        return featureColor(f < 0 ? "grey" : features[f].color)
    }
}

/// The dashes along a line, centered so both ends match (as in step 8).
func dashes(from a: SIMD3<Float>, to b: SIMD3<Float>) -> [(SIMD3<Float>, SIMD3<Float>)] {
    let length = simd_distance(a, b)
    guard length > 1e-5 else { return [] }
    let period = dashLength + dashGap
    let count = max(Int((length + dashGap) / period), 1)
    let used = Float(count) * period - dashGap
    let dir = (b - a) / length
    var start = a + dir * ((length - used) / 2)
    var out: [(SIMD3<Float>, SIMD3<Float>)] = []
    for _ in 0..<count {
        out.append((start, start + dir * dashLength))
        start += dir * period
    }
    return out
}

/// The whole ring as a tube of sticks, plus real atoms for one window of base
/// pairs. `atomWindow` empty means the ring alone, which is all GIF 1 needs.
func buildScene(_ plasmid: Plasmid, atomWindow: Range<Int>) -> PlasmidScene {
    let n = plasmid.length
    var shapes: [GPUShape] = []
    var tube: [Float] = [], space: [Float] = [], ball: [Float] = []
    shapes.reserveCapacity(n * 3 + atomWindow.count * 140)

    func add(_ s: GPUShape, tier1: Float, tier2: Float, tier3: Float) {
        shapes.append(s)
        tube.append(tier1)
        space.append(tier2)
        ball.append(tier3)
    }

    // The ring itself: one tube segment per base-pair step, down the helix's
    // own axis, colored by whichever feature owns that base pair. Drawing the
    // two backbone strands and their rungs instead was the first attempt, and
    // at 2.4 px it came out as a dashed white line — two thin strands and the
    // gaps between them are mostly empty space, which is exactly what a tube
    // is the correct resolution of at this scale.
    for i in 0..<n {
        let j = (i + 1) % n
        let fades = atomWindow.contains(i)
        add(.cylinder(from: plasmid.frame(at: i).origin, to: plasmid.frame(at: j).origin,
                      radius: 1, color: plasmid.color(at: i)),
            tier1: duplexRadius * tubeExaggeration,
            tier2: fades ? 0 : duplexRadius,
            tier3: fades ? 0 : duplexRadius)
    }
    let tubeCount = shapes.count

    // Real atoms for the window, stamped from the crystal templates.
    let grey = SIMD3<Float>(0.62, 0.64, 0.67)
    let hbondColor = SIMD3<Float>(0.80, 0.88, 0.97)
    var anchor: [Int: Int] = [:]
    var placed: [Int: [String: (index: Int, position: SIMD3<Float>)]] = [:]
    for i in atomWindow {
        guard let template = plasmid.templates[plasmid.templateKey(at: i)] else { continue }
        let f = plasmid.frame(at: i)
        anchor[i] = shapes.count
        var here: [String: (index: Int, position: SIMD3<Float>)] = [:]
        var world = [SIMD3<Float>](repeating: .zero, count: template.atoms.count)
        for (k, a) in template.atoms.enumerated() {
            let p = f.origin + f.tangent * a.position.x + f.y * a.position.y + f.z * a.position.z
            world[k] = p
            here["\(a.strand).\(a.name)"] = (shapes.count, p)
            add(.sphere(center: p, radius: 1, color: elementColor(a.element)),
                tier1: 0, tier2: vanDerWaalsRadius(a.element), tier3: ballRadius(a.element))
        }
        for b in template.bonds {
            add(.cylinder(from: world[b.a], to: world[b.b], radius: 1, color: grey),
                tier1: 0, tier2: 0, tier3: atomBondRadius)
        }
        for h in template.hbonds {
            for (s, e) in dashes(from: world[h.hydrogen], to: world[h.acceptor]) {
                add(.cylinder(from: s, to: e, radius: 1, color: hbondColor), tier1: 0, tier2: 0, tier3: hbondRadius)
            }
        }
        placed[i] = here
    }
    // The bonds from one nucleotide to the next: O3' to P. Strand 0 runs 5'→3'
    // as the index grows and strand 1 runs the other way, so the two strands
    // join up in opposite directions.
    for i in atomWindow.dropLast() {
        guard let a = placed[i], let b = placed[i + 1] else { continue }
        for (fromName, toName, fromSet, toSet) in [("0.O3'", "0.P", a, b), ("1.O3'", "1.P", b, a)] {
            guard let p = fromSet[fromName], let q = toSet[toName] else { continue }
            add(.cylinder(from: p.position, to: q.position, radius: 1, color: grey),
                tier1: 0, tier2: 0, tier3: atomBondRadius)
        }
    }

    var scene = PlasmidScene(shapes: shapes, tubeRadius: tube, spaceRadius: space, ballRadius: ball,
                             tubeCount: tubeCount, atomWindow: atomWindow, atomAnchor: anchor)
    scene.setDetail(0)
    return scene
}

/// Smooth 0→1 easing, as in steps 7a and 8.
func smoothstep01(_ t: Float) -> Float {
    let x = min(max(t, 0), 1)
    return x * x * (3 - 2 * x)
}

/// The GIF's palette. Median cut alone hands colors to whatever covers the
/// most pixels, so the four gene colors — which are the point of the ring
/// view but cover few pixels each — would be rounded into the background.
/// A few shades of every colour that carries meaning are reserved first.
func plasmidPalette(samples: [RGB], count: Int) -> [RGB] {
    var reserved: [RGB] = []
    let meaningful = ["gfp", "switch", "bla", "ori", "grey"].map(featureColor)
        + ["C", "N", "O", "P", "H"].map(elementColor)
    for c in meaningful {
        for light: Float in [0.35, 0.62, 0.9, 1.15] {
            let v = simd_clamp(c * light, SIMD3<Float>(repeating: 0), SIMD3<Float>(repeating: 1)) * 255
            reserved.append(RGB(UInt8(v.x.rounded()), UInt8(v.y.rounded()), UInt8(v.z.rounded())))
        }
    }
    return reserved + medianCutPalette(samples, count: max(count - reserved.count, 1))
}
