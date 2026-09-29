// Step 79: measuring whether a trail formed — from the record alone, never
// from the rules that made it.
//
// CONCENTRATION INDEX. The route is the straight corridor between the nest
// entrance and the sugar pile's centre, `corridorHalfWidth` either side. Of
// all the OUTBOUND ant positions (searching or following — not carrying, not
// tasting, not in the nest) over the last 30 simulated seconds, what
// fraction lie inside that corridor? Positions close to the nest or the pile
// are left out of both counts, since every ant passes there whatever it does.
// Ants spread evenly over the arena would score the corridor's share of the
// area (`uniformShare`, about 0.06); ants all walking the route score near 1.
// Outbound ants are the ones pheromone steers: returning ants home straight by
// path integration whether or not there is a trail, so they would score high
// either way; the index over ALL visible ants is reported too.
//
// The trail has FORMED at the first frame where the windowed index reaches
// `formedThreshold` and it stays at or above `holdThreshold` from then on.
//
// Thresholds and widths are MODEL, fixed before the rules were tuned: 0.5
// means half of all outbound ant-time is spent on one 10 mm-wide route that
// covers 4.5% of the counted ground — eleven times what chance gives. (The
// trails that form are not straight — they bow and loop round the pile — so
// this straight-corridor index understates them; it is kept because it needs
// no knowledge of where the trail went.)

import Foundation
import simd

enum SimMetric {
    static let corridorHalfWidth: Float = 5.0
    static let nestExclusion: Float = 15.0
    static let sugarExclusion: Float = 12.0
    /// The sliding window, simulated seconds.
    static let windowSeconds: Double = 30
    static let formedThreshold: Double = 0.5
    static let holdThreshold: Double = 0.4
}

struct SimCorridor {
    let a: SIMD2<Float>
    let b: SIMD2<Float>
    let config: SimWorldConfig

    init(_ c: SimWorldConfig) {
        a = c.nest
        b = c.sugar
        config = c
    }

    var length: Float { simd_distance(a, b) }

    /// Is this position counted at all (away from nest and pile)?
    func counted(_ p: SIMD2<Float>) -> Bool {
        simd_distance(p, a) > SimMetric.nestExclusion && simd_distance(p, b) > SimMetric.sugarExclusion
    }

    /// Perpendicular distance from the route, and how far along it (mm).
    func coordinates(_ p: SIMD2<Float>) -> (along: Float, off: Float) {
        let u: SIMD2<Float> = simd_normalize(b - a)
        let d: SIMD2<Float> = p - a
        let along: Float = simd_dot(d, u)
        let off: Float = abs(d.x * u.y - d.y * u.x)
        return (along, off)
    }

    /// Inside the corridor (between the ends, within the half-width).
    func inside(_ p: SIMD2<Float>) -> Bool {
        let (s, o) = coordinates(p)
        return s >= 0 && s <= length && o <= SimMetric.corridorHalfWidth
    }

    /// The corridor's share of the counted ground, by a fixed grid of points:
    /// what evenly spread ants would score.
    var uniformShare: Double {
        var n: Int = 0
        var k: Int = 0
        let step: Float = 0.5
        var x: Float = config.edgeMargin
        while x <= config.width - config.edgeMargin {
            var z: Float = config.edgeMargin
            while z <= config.depth - config.edgeMargin {
                let p = SIMD2<Float>(x, z)
                if counted(p) {
                    n += 1
                    if inside(p) { k += 1 }
                }
                z += step
            }
            x += step
        }
        return n > 0 ? Double(k) / Double(n) : 0
    }
}

struct SimConcentration {
    /// Per film frame: the windowed index over outbound ants (nil where no
    /// outbound ant was counted in the window), and over all visible ants.
    var outbound: [Double?]
    var all: [Double?]
    /// First frame the trail counts as formed, if it does.
    var formedFrame: Int?
    var uniformShare: Double
}

func simIsOutbound(_ a: SimAntFrame) -> Bool {
    (a.state == .searching || a.state == .following) && a.carrying == 0
}

func simConcentration(_ r: SimFilmRecord) -> SimConcentration {
    let corridor = SimCorridor(r.config)
    let n: Int = r.frameCount
    // Per frame counts, then a sliding window.
    var outIn: [Int] = Array(repeating: 0, count: n)
    var outAll: [Int] = Array(repeating: 0, count: n)
    var allIn: [Int] = Array(repeating: 0, count: n)
    var allAll: [Int] = Array(repeating: 0, count: n)
    for f in 0..<n {
        for a in r.ants(atFrame: f) where a.visible && a.state != .tasting {
            let p: SIMD2<Float> = a.position
            if !corridor.counted(p) { continue }
            let hit: Bool = corridor.inside(p)
            allAll[f] += 1
            if hit { allIn[f] += 1 }
            if simIsOutbound(a) {
                outAll[f] += 1
                if hit { outIn[f] += 1 }
            }
        }
    }
    func windowed(_ num: [Int], _ den: [Int]) -> [Double?] {
        var out: [Double?] = []
        for f in 0..<n {
            let now: Double = r.simSeconds(atFrame: f)
            var sn: Int = 0
            var sd: Int = 0
            var g: Int = f
            while g >= 0 && r.simSeconds(atFrame: g) > now - SimMetric.windowSeconds {
                sn += num[g]
                sd += den[g]
                g -= 1
            }
            out.append(sd > 0 ? Double(sn) / Double(sd) : nil)
        }
        return out
    }
    let outbound: [Double?] = windowed(outIn, outAll)
    let all: [Double?] = windowed(allIn, allAll)
    var formed: Int? = nil
    for f in 0..<n {
        guard let v = outbound[f], v >= SimMetric.formedThreshold else { continue }
        var holds: Bool = true
        for g in f..<n {
            if let w = outbound[g], w < SimMetric.holdThreshold { holds = false; break }
        }
        if holds { formed = f; break }
    }
    return SimConcentration(outbound: outbound, all: all, formedFrame: formed, uniformShare: corridor.uniformShare)
}

/// Mean of the non-nil values in frames [from, to).
func simMean(_ v: [Double?], from: Int, to: Int) -> Double? {
    var s: Double = 0
    var k: Int = 0
    for f in max(from, 0)..<min(to, v.count) {
        if let x = v[f] { s += x; k += 1 }
    }
    return k > 0 ? s / Double(k) : nil
}

// MARK: - can the trail be drawn?

/// How the formed trail's pheromone stands out, for the film's picture.
struct SimTrailVisibility {
    /// Median concentration sampled along the route's centreline (middle part).
    var onTrailMedian: Float
    /// 99th percentile of the field off the route (farther than 3 half-widths).
    var offTrail99: Float
    /// Fraction of the centreline samples above a fifth of the median: how
    /// continuous the trail is.
    var continuity: Float
    /// Fraction of all pheromone that lies within the corridor.
    var massInCorridor: Double
    /// onTrailMedian / max(offTrail99, a floor): the contrast.
    var contrast: Float
    /// Peak value anywhere, marks/mm².
    var peak: Float
}

func simTrailVisibility(field v: [Float], width: Int, height: Int, config c: SimWorldConfig) -> SimTrailVisibility {
    let corridor = SimCorridor(c)
    let h: Float = SimConst.cell
    func at(_ p: SIMD2<Float>) -> Float {
        let i: Int = Int(p.x / h)
        let j: Int = Int(p.y / h)
        if i < 0 || j < 0 || i >= width || j >= height { return 0 }
        return v[j * width + i]
    }
    // Centreline samples every 0.5 mm, between the exclusions; the best of a
    // ±2 mm cross-section at each (the trail need not be dead straight).
    var line: [Float] = []
    let u: SIMD2<Float> = simd_normalize(corridor.b - corridor.a)
    let nrm = SIMD2<Float>(-u.y, u.x)
    var s: Float = SimMetric.nestExclusion
    while s <= corridor.length - SimMetric.sugarExclusion {
        var best: Float = 0
        var o: Float = -2
        while o <= 2 {
            let p: SIMD2<Float> = corridor.a + u * s + nrm * o
            best = max(best, at(p))
            o += 0.5
        }
        line.append(best)
        s += 0.5
    }
    let sortedLine: [Float] = line.sorted()
    let median: Float = sortedLine.isEmpty ? 0 : sortedLine[sortedLine.count / 2]
    var off: [Float] = []
    var inMass: Double = 0
    var total: Double = 0
    var peak: Float = 0
    for j in 0..<height {
        for i in 0..<width {
            let x: Float = (Float(i) + 0.5) * h
            let z: Float = (Float(j) + 0.5) * h
            let p = SIMD2<Float>(x, z)
            let val: Float = v[j * width + i]
            peak = max(peak, val)
            total += Double(val)
            let (along, o) = corridor.coordinates(p)
            let inCorr: Bool = along >= 0 && along <= corridor.length && o <= SimMetric.corridorHalfWidth
            if inCorr { inMass += Double(val) }
            let farOff: Bool = o > 3 * SimMetric.corridorHalfWidth || along < 0 || along > corridor.length
            if farOff && corridor.counted(p) { off.append(val) }
        }
    }
    off.sort()
    let rank: Float = Float(off.count) * 0.99
    let at99: Int = min(off.count - 1, Int(rank))
    let off99: Float = off.isEmpty ? 0 : off[at99]
    let cut: Float = median / 5
    let above: Int = line.filter { $0 > cut }.count
    let aboveF: Float = Float(above)
    let lineF: Float = Float(max(line.count, 1))
    let cont: Float = line.isEmpty || median <= 0 ? 0 : aboveF / lineF
    let floorValue: Float = 1e-6
    return SimTrailVisibility(onTrailMedian: median, offTrail99: off99, continuity: cont,
                              massInCorridor: total > 0 ? inMass / total : 0,
                              contrast: median / max(off99, floorValue), peak: peak)
}
