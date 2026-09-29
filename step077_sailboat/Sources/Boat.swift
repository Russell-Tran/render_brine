// The boat: an ILCA 7 (the Laser, standard rig), Bruce Kirby's single-handed
// Olympic dinghy, drawn as a few clean shapes — hull, mast, boom, a flat
// triangular mainsail, the centreboard and the rudder — at its published
// size, floating exactly where Archimedes puts it in fresh water.
//
// Sources, every one read for this step:
//   [DN]     A. H. Day and E. Nixon, "Measurement and prediction of the
//            resistance of a Laser sailing dinghy", Int. J. Small Craft
//            Technology (Trans. RINA) 2014, read from Strathclyde's
//            repository (strathprints 46407). The hull's lines were measured
//            off a real Laser (RMS error < 1 mm) and tank-tested:
//            • "The nominal hull weight including fittings is quoted by the
//              manufacturer as 58kg. Added to that are the weight of the
//              mast, boom, foils, rudder stock, tiller and extension, ropes
//              and sail. These were weighed at approximately 22kg. The
//              optimal weight for a standard rig Laser sailor is often quoted
//              at around 80kg. Hence the benchmark displacement was chosen to
//              be 160kg".
//            • Table 1, "Standard" (80 kg crew, level trim): hull volume
//              displacement 0.160 m³, waterline length 3.791 m, waterline
//              beam 1.103 m, hull draught (neglecting appendages) 0.094 m,
//              waterplane area 2.859 m², Cp 0.552, Cm 0.757,
//              LCB 0.532 Lwl aft of the forward perpendicular.
//            • Level trim: "the vessel is trimmed so that the transom stern
//              was just touching the water surface"; results "scaled to fresh
//              water at 15.0 degrees Celsius".
//   [WP-L]   Wikipedia, "Laser (dinghy)" (read 2026-09-28): LOA 4.23 m,
//            beam 1.37 m, draft 0.787 m, hull weight 58.97 kg, ILCA 7 sail
//            area 7.06 m². (Wikipedia's "Laser Standard" page gives beam
//            1.39 m and LOA 4.2 m; lasersailingtips.com 1.42 m — sources
//            disagree on the beam; 1.37 m is used.)
//   [LST]    lasersailingtips.com, "Laser Sailboat Specs" (read): ILCA 7 sail
//            "Luff: 5.13m, Leech: 5.57m, Foot: 2.74m", area 7.06 m².
//   [SF-m]   sailingforums.com, "Radial or full" (read): "The bottom mast
//            section for a Laser Standard is 2865 mm long … The top section …
//            is 3600 mm long", and a reply: "the inserted part is 305mm".
//            A forum, not the builder: unverified.
//   [SF-p]   sailingforums.com, "Mast info" (read): quoting the ILCA handbook,
//            "945 mm" from the bottom of the mast to the boom pin. Forum.
//   [SF-d]   sailingforums.com, "Mast step dimensions and repair" (read),
//            Performance Sailcraft's figures: standard lower mast OD 62.9–
//            64.5 mm. [YBW] forums.ybw.com, "laser mast how do i know what
//            diameter it is?" (read): "Top section: 51.00mm OD". Forums.
//   [SF-c]   sailingforums.com, "Laser daggerboard size and dimensions?"
//            (read): "341 mm x 680 mm" — the board's chord and the depth it
//            reaches below the hull. Forum. The ILCA class rules (2025, read)
//            call it the centreboard and give no dimensions for it: the
//            boat is one-design, built to the builder's moulds.
//   [WP-w]   Wikipedia, "Water (data page)", density of liquid water:
//            "0.9991026 g/cm3" at 15 °C.
//   [WP-n]   Wikipedia, "Refractive index", selected indices at 589 nm:
//            water 1.333 (20 °C).
//
// Metres. Boat frame: x along the boat (bow +), y up from the hull's lowest
// point, z across (starboard +). The world puts the water surface at y = 0.

import Foundation
import simd

enum Mutant: String {
    case none
    case floatsWrong        // the waterline placed by eye: 30% too deep
    case wrongSize          // the whole boat 20% too big
    case noKeel             // no centreboard
}

let activeMutant: Mutant = Mutant(rawValue: ProcessInfo.processInfo.environment["BOAT_MUTANT"] ?? "") ?? .none

// MARK: - the class's numbers

/// Length overall and beam [WP-L].
let classLength: Double = 4.23
let classBeam: Double = 1.37
/// Draft with the board down [WP-L]; the test compares, the drawing does not use it.
let classDraft: Double = 0.787
/// Sail [LST], area [WP-L][LST].
let sailLuff: Double = 5.13
let sailFoot: Double = 2.74
let sailLeech: Double = 5.57
let sailArea: Double = 7.06
/// Mast [SF-m]: lower section 2865 mm with its plug, top 3600 mm of which
/// 305 mm slides into the lower: 6160 mm assembled.
let mastLower: Double = 2.865
let mastTop: Double = 3.600
let mastInsert: Double = 0.305
var mastLength: Double { mastLower + mastTop - mastInsert }
/// Boom pin 945 mm above the mast's foot [SF-p].
let boomPinHeight: Double = 0.945
/// Tube diameters [SF-d][YBW]: lower 63.7 mm (the middle of PSA's 62.9–64.5),
/// top 51 mm.
let mastLowerRadius: Double = 0.0637 / 2
let mastTopRadius: Double = 0.051 / 2
/// Centreboard [SF-c]: 341 mm chord, reaching 680 mm below the hull; 31 mm
/// thick (the same thread: "a bit over 31 mm", a 9% section) — drawn as a
/// stadium, the foil's NACA shape not modelled.
let boardChord: Double = 0.341
let boardDepthBelowHull: Double = 0.680
let boardThickness: Double = 0.031

// MARK: - the load [DN]

let hullMass: Double = 58
let rigAndFoilsMass: Double = 22
/// The sailor's mass is counted (Day & Nixon's benchmark) but the sailor is
/// not drawn, so the rig and the underwater body read.
let sailorMass: Double = 80
var totalMass: Double { hullMass + rigAndFoilsMass + sailorMass }

/// Fresh water at 15 °C [WP-w] — Day & Nixon's condition, so their measured
/// hydrostatics are a like-for-like check. (Sea water would float her
/// ~2.5% higher.)
let waterDensity: Double = 999.1026
/// For the surface's reflection only [WP-n]. Refraction is left out on
/// purpose: the cut-away shows the underwater body true to scale.
let waterIndex: Double = 1.333

// MARK: - the simplified hull
//
// A constant cross-section — a flat-bottomed V (deadrise) with a round
// bilge and flared sides — narrowed across by the plan shape and lifted by a
// keel line (rocker), cut by the deck and the transom. Every one of these
// parameters is MODEL: chosen (by a small fit, recorded in the step's
// report) so the simplified hull reproduces Day & Nixon's measured
// hydrostatics at 160 kg, not the real lines.

struct HullShape {
    var length: Double = classLength
    var halfBeam: Double = classBeam / 2
    /// Deck height above the lowest point. MODEL (not published).
    var depth: Double = 0.34
    /// The rocker: the transom's lowest edge and the forward end of the
    /// waterline both at Day & Nixon's level-trim draught, 3.791 m apart.
    var trimDraught: Double = 0.094
    var trimWaterline: Double = 3.791
    /// Where the keel line is lowest, and the aft curve's power. MODEL.
    var keelLowX: Double = -0.73
    var aftPower: Double = 1.2
    /// Section: flat-V half-width a1 rising h1 (deadrise), bilge radius r. MODEL.
    var bottomHalf: Double = 0.40
    var deadrise: Double = 0.03
    var bilge: Double = 0.21
    /// Plan: widest at x = widestX, transom half-width at the deck. MODEL.
    var widestX: Double = -1.35
    var transomHalf: Double = 0.42
    /// The stem: a narrow rounded face, 30 mm across, standing 100 mm down
    /// from the deck at the bow. MODEL.
    var stemHalf: Double = 0.015
    var stemDrop: Double = 0.10
    /// The section is not narrowed below this share of its width; nearer
    /// the bow the plan cuts it.
    var minNarrow: Double = 0.25
    /// Divisor keeping the implicit hull a lower bound on distance: its
    /// steepest gradient near the hull, sampled in the prototype, was 1.22;
    /// 1.4 leaves room. MODEL (tested: see the distance-function tests).
    var lipschitz: Double = 1.4

    var stern: Double { -length / 2 }
    var bow: Double { length / 2 }
    /// Side slope dz/dy, solved so the section is halfBeam wide at the deck
    /// at the widest station (where the keel line stands `lift` above the
    /// lowest point).
    func sideSlope(lift: Double) -> Double {
        let top: Double = depth - lift
        var lo: Double = -2, hi: Double = 5
        for _ in 0..<80 {
            let s: Double = (lo + hi) / 2
            let rise: Double = top - bilge - deadrise
            let side: Double = s * rise
            let s2: Double = s * s
            let round: Double = bilge * (1 + s2).squareRoot()
            let w: Double = bottomHalf + side + round
            if w > halfBeam { hi = s } else { lo = s }
        }
        return (lo + hi) / 2
    }
    /// The bow's keel curve rises from the waterline's forward end to the
    /// foot of the stem: its power follows.
    var stemFoot: Double { depth - stemDrop }
    var forePower: Double {
        let xw: Double = stern + trimWaterline
        return log(trimDraught / stemFoot) / log((xw - keelLowX) / (bow - keelLowX))
    }
}

let hullShape = HullShape()
let hullForePower: Double = hullShape.forePower
let hullSideSlope: Double = hullShape.sideSlope(lift: keelHeight(hullShape.widestX))

// MARK: - the rest of the boat, placed. All positions MODEL unless noted.

/// Mast 1.0 m aft of the bow, its foot 0.30 m below the deck (in its tube).
let mastX: Double = hullShape.bow - 1.0
let mastFootY: Double = hullShape.depth - 0.30
/// Centreboard: its middle 0.30 m forward of midships; its top a handle's
/// height above the deck.
let boardX: Double = 0.30
let boardTopY: Double = hullShape.depth + 0.10
/// Rudder blade (MODEL: not sourced): 220 mm chord, 24 mm thick, hung 60 mm
/// aft of the transom, reaching 0.50 m below the hull's lowest point.
let rudderChord: Double = 0.22
let rudderThickness: Double = 0.024
let rudderX: Double = hullShape.stern - 0.06 - 0.11
let rudderBottomY: Double = -0.50
let rudderTopY: Double = hullShape.depth + 0.12
/// Boom: 2.9 m long, top-section tube. MODEL.
let boomLength: Double = 2.90
/// Cockpit well. MODEL.
let cockpitCentre = SIMD3<Double>(-0.75, hullShape.depth, 0)
let cockpitHalf = SIMD3<Double>(0.45, 0.16, 0.24)

/// The angle at the tack between the luff (up the mast) and the foot, from
/// the three sides [LST] by the law of cosines.
var tackAngle: Double {
    let a2: Double = sailLuff * sailLuff
    let b2: Double = sailFoot * sailFoot
    let c2: Double = sailLeech * sailLeech
    let ab: Double = sailLuff * sailFoot
    let c: Double = (a2 + b2 - c2) / (2 * ab)
    return acos(c)
}

struct Sail {
    var tack: SIMD3<Double>
    var head: SIMD3<Double>
    var clew: SIMD3<Double>
    var area: Double {
        let a: SIMD3<Double> = head - tack
        let b: SIMD3<Double> = clew - tack
        return simd_length(simd_cross(a, b)) / 2
    }
}

func sailGeometry() -> Sail {
    let tack = SIMD3<Double>(mastX, mastFootY + boomPinHeight, 0)
    let head: SIMD3<Double> = tack + SIMD3<Double>(0, sailLuff, 0)
    let th: Double = tackAngle
    let dir = SIMD3<Double>(-sin(th), cos(th), 0)
    return Sail(tack: tack, head: head, clew: tack + dir * sailFoot)
}

func boatScale(_ m: Mutant = activeMutant) -> Double { m == .wrongSize ? 1.2 : 1 }

// MARK: - the distance function, on the CPU (the kernel's twin; a test holds them equal)

func sdSection(_ y: Double, _ az: Double) -> Double {
    let h: HullShape = hullShape
    let s: Double = hullSideSlope
    let a1: Double = h.bottomHalf
    let r: Double = h.bilge
    let h1: Double = h.deadrise
    let ln: Double = (a1 * a1 + h1 * h1).squareRoot()
    let ez: Double = a1 / ln
    let ey: Double = h1 / ln
    let u: Double = min(max(az * ez + (y - r) * ey, 0), ln)
    let gz: Double = az - u * ez
    let gy: Double = y - r - u * ey
    let dSeg: Double = (gz * gz + gy * gy).squareRoot()
    let s2: Double = s * s
    let n: Double = (s2 + 1).squareRoot()
    let dz: Double = s / n
    let dy: Double = 1 / n
    let yc: Double = r + h1
    let t: Double = max((az - a1) * dz + (y - yc) * dy, 0)
    let rz: Double = az - a1 - t * dz
    let ry: Double = y - yc - t * dy
    let dRay: Double = (rz * rz + ry * ry).squareRoot()
    let aboveBottom: Bool = (y - r) * a1 > az * h1
    let sideZ: Double = a1 + s * (y - yc)
    let inside: Bool = aboveBottom && az < sideZ
    let d: Double = min(dSeg, dRay)
    return (inside ? -d : d) - r
}

func keelHeight(_ x: Double) -> Double {
    let h: HullShape = hullShape
    if x < h.keelLowX {
        let u: Double = max((h.keelLowX - x) / (h.keelLowX - h.stern), 0)
        return h.trimDraught * pow(u, h.aftPower)
    }
    let u: Double = max((x - h.keelLowX) / (h.bow - h.keelLowX), 0)
    return h.stemFoot * pow(u, hullForePower)
}

func planHalfWidth(_ x: Double) -> Double {
    let h: HullShape = hullShape
    if x < h.widestX {
        let u: Double = min(max((h.widestX - x) / (h.widestX - h.stern), 0), 1)
        let u2: Double = u * u
        let span: Double = h.halfBeam - h.transomHalf
        return h.transomHalf + span * (1 - u2)
    }
    let u: Double = min(max((x - h.widestX) / (h.bow - h.widestX), 0), 1)
    let u2: Double = u * u
    return h.halfBeam * (1 - u2) + h.stemHalf * u2
}

func sdBox(_ p: SIMD3<Double>, _ b: SIMD3<Double>) -> Double {
    let q: SIMD3<Double> = abs(p) - b
    let outside: Double = simd_length(simd_max(q, SIMD3<Double>(repeating: 0)))
    return outside + min(max(q.x, max(q.y, q.z)), 0)
}

func sdRoundBox(_ p: SIMD3<Double>, _ b: SIMD3<Double>, _ r: Double) -> Double {
    sdBox(p, b - SIMD3<Double>(repeating: r)) - r
}

/// The hull, in the boat frame at unit scale.
func hullDistance(_ p: SIMD3<Double>) -> Double {
    let h: HullShape = hullShape
    let lo = SIMD3<Double>(h.stern, 0, -h.halfBeam)
    let hi = SIMD3<Double>(h.bow, h.depth, h.halfBeam)
    let m: Double = 0.1
    let margin = SIMD3<Double>(repeating: m)
    let q: SIMD3<Double> = simd_clamp(p, lo - margin, hi + margin)
    let B: Double = planHalfWidth(q.x)
    let k: Double = max(B / h.halfBeam, h.minNarrow)
    let az: Double = abs(q.z)
    let sec: Double = sdSection(q.y - keelHeight(q.x), az / k) * k
    let across: Double = max(sec, az - B)
    let ends: Double = max(h.stern - q.x, q.x - h.bow)
    let top: Double = q.y - h.depth
    let implicit: Double = max(across, max(ends, top))
    let centre: SIMD3<Double> = (lo + hi) / 2
    let halfSize: SIMD3<Double> = (hi - lo) / 2
    let box: Double = sdBox(p - centre, halfSize)
    var d: Double = max(implicit / h.lipschitz, box)
    d = max(d, -sdRoundBox(p - cockpitCentre, cockpitHalf, 0.05))
    return d
}

/// A vertical foil: a stadium of `chord` × `thick` in plan, from y0 to y1.
func sdFoil(_ p: SIMD3<Double>, x: Double, chord: Double, thick: Double, y0: Double, y1: Double) -> Double {
    let rr: Double = thick / 2
    let half: Double = chord / 2 - rr
    let dx: Double = max(abs(p.x - x) - half, 0)
    let d2: Double = (dx * dx + p.z * p.z).squareRoot() - rr
    let dy: Double = abs(p.y - (y0 + y1) / 2) - (y1 - y0) / 2
    let w = SIMD2<Double>(d2, dy)
    return min(max(w.x, w.y), 0) + simd_length(simd_max(w, SIMD2<Double>(repeating: 0)))
}

func boardDistance(_ p: SIMD3<Double>) -> Double {
    let tip: Double = keelHeight(boardX) - boardDepthBelowHull
    return sdFoil(p, x: boardX, chord: boardChord, thick: boardThickness, y0: tip, y1: boardTopY)
}

func rudderDistance(_ p: SIMD3<Double>) -> Double {
    sdFoil(p, x: rudderX, chord: rudderChord, thick: rudderThickness, y0: rudderBottomY, y1: rudderTopY)
}

/// Everything that can be under water, in the boat frame at the boat's
/// drawn scale: hull ∪ centreboard ∪ rudder (the spars and sail are far above).
func underwaterBody(_ p: SIMD3<Double>, mutant m: Mutant = activeMutant) -> Double {
    let s: Double = boatScale(m)
    let q: SIMD3<Double> = p / s
    var d: Double = min(hullDistance(q), rudderDistance(q))
    if m != .noKeel { d = min(d, boardDistance(q)) }
    return d * s
}

// MARK: - Archimedes: the waterline from the shape

/// Every vertical column's inside intervals, found by sphere-tracing up the
/// column with the distance function itself and refining each crossing by
/// bisection. The submerged volume at any waterline is then exact up to the
/// column grid.
struct ColumnVolume {
    var cellArea: Double
    var intervals: [SIMD2<Double>]         // (y in, y out), boat frame
    var columns: Int

    func volume(below y: Double) -> Double {
        var v: Double = 0
        for iv in intervals where iv.x < y { v += min(iv.y, y) - iv.x }
        return v * cellArea
    }
}

func columnVolume(_ f: (SIMD3<Double>) -> Double, x0: Double, x1: Double, halfZ: Double, y0: Double, y1: Double,
                  dx: Double, dz: Double) -> ColumnVolume {
    var intervals: [SIMD2<Double>] = []
    let nx: Int = Int(((x1 - x0) / dx).rounded(.up))
    let nz: Int = Int((2 * halfZ / dz).rounded(.up))
    for i in 0..<nx {
        let x: Double = x0 + (Double(i) + 0.5) * dx
        for j in 0..<nz {
            let z: Double = -halfZ + (Double(j) + 0.5) * dz
            var y: Double = y0
            var d: Double = f(SIMD3<Double>(x, y, z))
            var inside: Bool = d < 0
            var start: Double = y0
            while y < y1 {
                let step: Double = max(abs(d), 1e-4)
                let yn: Double = min(y + step, y1)
                let dn: Double = f(SIMD3<Double>(x, yn, z))
                if (dn < 0) != inside {
                    var a: Double = y, b: Double = yn
                    for _ in 0..<30 {
                        let mid: Double = (a + b) / 2
                        if (f(SIMD3<Double>(x, mid, z)) < 0) == inside { a = mid } else { b = mid }
                    }
                    let cross: Double = (a + b) / 2
                    if inside { intervals.append(SIMD2<Double>(start, cross)) } else { start = cross }
                    inside = !inside
                }
                y = yn
                d = dn
            }
            if inside { intervals.append(SIMD2<Double>(start, y1)) }
        }
    }
    return ColumnVolume(cellArea: dx * dz, intervals: intervals, columns: nx * nz)
}

struct Flotation {
    /// The waterline's height in the boat frame: the hull's draught.
    var draught: Double
    var volume: Double
    var mass: Double
    var body: ColumnVolume
}

/// Sink the boat until the water it displaces weighs what it does.
func archimedes(mutant m: Mutant = activeMutant, dx: Double = 0.005, dz: Double = 0.0025) -> Flotation {
    let s: Double = boatScale(m)
    let h: HullShape = hullShape
    let body: ColumnVolume = columnVolume({ underwaterBody($0, mutant: m) },
                                          x0: (rudderX - rudderChord) * s, x1: h.bow * s, halfZ: h.halfBeam * s,
                                          y0: (rudderBottomY - 0.3) * s, y1: h.depth * s, dx: dx, dz: dz)
    let want: Double = totalMass / waterDensity
    var lo: Double = 0, hi: Double = h.depth * s
    for _ in 0..<60 {
        let mid: Double = (lo + hi) / 2
        if body.volume(below: mid) > want { hi = mid } else { lo = mid }
    }
    let t: Double = (lo + hi) / 2
    return Flotation(draught: t, volume: body.volume(below: t), mass: totalMass, body: body)
}

/// The waterline the picture uses: Archimedes', or the mutant's by eye.
func drawnDraught(_ f: Flotation, _ m: Mutant = activeMutant) -> Double {
    m == .floatsWrong ? f.draught * 1.3 : f.draught
}

// MARK: - tone and colour, as in steps 20, 53, 60 and 72

func srgbEncode(_ v: Double) -> Double {
    let c: Double = min(max(v, 0), 1)
    return c <= 0.0031308 ? 12.92 * c : 1.055 * pow(c, 1.0 / 2.4) - 0.055
}

let toneGain: Double = 1.12
