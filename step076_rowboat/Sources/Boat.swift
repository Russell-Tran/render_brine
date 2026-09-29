// The boat: a Whitehall pulling boat, the classic 19th-century harbour
// rowing boat, in Newfound Woodworks' 16 ft 8 in cedar-strip version, drawn
// as few clean shapes to its published size: the hull (with its keel, full
// skeg and transom), two rowing thwarts and a stern seat, two pairs of
// oarlocks on oar blocks ("double oarlocks"), and one pair of oars at rest,
// blades trailing. It floats where Archimedes puts it: the draft is solved
// here from the hull's own shape.
//
// Sources, every one opened for this step:
//   [NW]      Newfound Woodworks, "16′8″ Whitehall Pulling Boat", kit
//             specifications, newfound.com/boat/whitehall-pulling-boat/
//             (read 2026-09-28): "Length (LOA) 16'8"; Length (LWL) 15.23';
//             Beam (BOA) 42.00"; Beam (BWL) 37.80"; Weight 113 lbs.;
//             Displacement (Capacity) 725 lbs; Draft (at Capacity) 8.13";
//             Center Depth 16.20"; Depth at Bow 27.99"". Text: "based on an
//             old New England design ... a traditional skeg and elegant
//             'wine glass' transom ... fitted with double oarlocks". Kit:
//             "Spanish Cedar Thwarts and Stern fan seat parts ... oar blocks,
//             Spanish Cedar transom materials", Northern White and Western
//             Red Cedar strips, mahogany wales. Its photographs (the boat
//             rowed on a lake) were looked at for the sheer and the transom.
//   [Wiki-W]  Wikipedia, "Whitehall rowboat": "first made in the U.S. at the
//             foot of Whitehall Street in New York City to ferry goods, and
//             people to ships in New York Harbor"; "a keel running the entire
//             length of the bottom and a distinctive wine glass transom with
//             a full skeg".
//   [Wiki-S]  Wikipedia, "Seawater": "The average density at the surface is
//             1.025 kg/L"; "ranges from about 1020 to 1029 kg/m3".
//   [Walpole] S. C. Walpole et al., "The weight of nations: an estimation of
//             adult human biomass", BMC Public Health 12:439 (2012), Table 3
//             and text: "North America has the highest average body mass of
//             any continent (80.7 kg)"; world 62.0 kg.
//   [S&T]     Shaw & Tenney, "Oar Sizing" (shawandtenney.com/pages/oar-
//             sizing): span between the oar sockets; "Divide the span by 2,
//             then add 2. This gives you the inboard loom length. Multiply
//             the inboard loom length by 25, then divide by 7 ... Round to
//             the nearest 6" increment"; "7/25 of the oar inboard of the
//             oarlocks, 18/25 outboard". "Flat Blade Oars": blade "5-1/2" -
//             Standard", narrow 4-5/8" "recommended for oars 7' 10" and over".
//   [Angus]   Angus Rowboats, "Fixed Seat Rowing Geometry": "The oarlocks
//             should be placed approximately 13" back from the aft edge of the
//             rowing seat"; "at least eight inches higher than the seat to a
//             maximum of 11.5""; seat "from the lowest point of the bilge
//             should be 5.5"- 8"".
//   [WDB]     The Wood Database, "Sitka Spruce": "Average Dried Weight 27
//             lbs/ft3 (425 kg/m3)".
//   [HQ]      G. M. Hale and M. R. Querry, Appl. Opt. 12, 555 (1973), water
//             at 25 °C, refractiveindex.info `main/H2O/nk/Hale.yml`:
//             n = 1.333 at 0.550 µm (read for this step).
//
// Units: millimetres. Hull frame: x along the boat (bow +x, transom −x,
// x = 0 amidships), y up from the baseline (the keel's bottom amidships),
// z across. The world is the hull frame lowered by the draft, so the water
// is the plane y = 0.

import Foundation
import simd

enum Mutant: String {
    case none
    case floatsWrong        // the waterline placed by eye: 30% too deep
    case wrongSize          // the boat 20% too big
    case noOarlocks         // the oars rest on nothing
}

let activeMutant: Mutant = Mutant(rawValue: ProcessInfo.processInfo.environment["BOAT_MUTANT"] ?? "") ?? .none

// MARK: - the published numbers [NW], converted exactly (1 in = 25.4 mm,
// 1 lb = 0.45359237 kg)

let inch: Float = 25.4
let poundKilograms: Double = 0.45359237
let nwLOA: Float = 200 * 25.4           // 16'8" = 200 in = 5080 mm
let nwLWLFeet: Float = 15.23
let nwLWL: Float = nwLWLFeet * 304.8     // 4642.1 mm (1 ft = 304.8 mm)
let nwBOA: Float = 42.00 * 25.4         // 1066.8 mm
let nwBWL: Float = 37.80 * 25.4         // 960.1 mm
let nwCenterDepth: Float = 16.20 * 25.4 // 411.5 mm
let nwBowDepth: Float = 27.99 * 25.4    // 711.0 mm
let nwDraftAtCapacity: Float = 8.13 * 25.4   // 206.5 mm
let nwWeightPounds: Double = 113
let nwCapacityPounds: Double = 725

/// Seawater: a Whitehall is a harbour boat [Wiki-W], so it floats in the
/// sea, at the surface average [Wiki-S].
let seawaterDensity: Double = 1025          // kg/m³
/// Fresh water, for the check against Newfound's lake-tested figures:
/// 1000 kg/m³ is the round value (pure water is 997 at 25 °C); MODEL for a
/// comparison only, never for the drawn waterline.
let freshwaterDensity: Double = 1000
/// One adult: North America's average [Walpole] (the boat is American).
/// Not drawn and not in the drawn waterline — the picture floats the boat
/// as drawn; the rower's draft is given beside it.
let rowerKilograms: Double = 80.7
let spruceDensity: Double = 425             // kg/m³ [WDB]
let waterIndex: Double = 1.333              // [HQ]

// MARK: - the hull's lines

/// The hull's shape. Newfound publish the six numbers above, not the lines,
/// so the lines are MODEL: simple curves whose free parameters were fitted
/// (by hand, in a Python sketch, then held by the tests) so that at
/// Newfound's full-load draft the hull has Newfound's waterline length and
/// waterline beam and displaces Newfound's 725 lb. Sections are round:
/// the lower half of an ellipse from the keel up to a little below the
/// sheer, then straight sides to the sheer. (A first fit with a V bottom
/// and a hard turn of the bilge met the same numbers but hid the light
/// waterline under a chine; a Whitehall has "rounded sides" [Wiki-W], as
/// Newfound's photos show.)
struct HullShape {
    let scale: Float
    let length: Float           // LOA [NW]
    let halfBeam: Float         // BOA / 2 [NW]
    let centerDepth: Float      // sheer above the baseline amidships [NW]
    let bowDepth: Float         // stem head above the baseline [NW]
    /// Sheer at the transom: unpublished; MODEL from Newfound's side photo,
    /// where the transom's top stands a little higher than amidships.
    let sternDepth: Float
    /// Station of greatest beam: a little aft of amidships. MODEL.
    let maxBeamX: Float
    /// Transom half-width at the sheer as a fraction of the half-beam. MODEL
    /// from the photos.
    let transomRatio: Float = 0.60
    /// The bow's lines: the half-beam falls as 1 − u^1.5 to the stem.
    /// MODEL, fitted.
    let bowBeamPower: Float = 1.5
    /// The keel: its depth below the hull's bottom amidships and its width.
    /// MODEL.
    let keelDepth: Float
    let keelHalfWidth: Float
    /// The bottom's rise towards the bow: flat until bowRiseStart, rising as
    /// u^1.8 to bowFoot at the stem. MODEL, fitted to LWL.
    let bowRiseStart: Float
    let bowFoot: Float
    let bowRisePower: Float = 1.8
    /// The bottom's rise towards the transom, as v^1.5, to transomFoot, so
    /// the transom's foot just meets the water at full load. MODEL, fitted.
    let sternRiseStart: Float
    let transomFoot: Float
    let sternRisePower: Float = 1.5
    /// Straight topsides: the ellipse's centre (where the section is
    /// widest) this far below the sheer. MODEL, fitted to BWL.
    let topsides: Float
    /// Skin: cedar strips with glass both sides. MODEL (Newfound's kit uses
    /// ¼-inch-class strips; not published as a thickness).
    let skin: Float

    init(scale s: Float) {
        scale = s
        length = nwLOA * s
        halfBeam = nwBOA / 2 * s
        centerDepth = nwCenterDepth * s
        bowDepth = nwBowDepth * s
        sternDepth = 470 * s
        maxBeamX = -200 * s
        keelDepth = 25 * s
        keelHalfWidth = 16 * s
        bowRiseStart = 250 * s
        bowFoot = 330 * s
        sternRiseStart = -600 * s
        transomFoot = 195 * s
        skin = 9 * s
        topsides = 60 * s
    }

    var halfLength: Float { length / 2 }

    /// Height of the sheer above the baseline: lowest amidships (x = 0).
    func sheer(_ x: Float) -> Float {
        let h: Float = halfLength
        if x >= 0 {
            let u: Float = min(x / h, 1)
            return centerDepth + (bowDepth - centerDepth) * u * u
        }
        let v: Float = min(-x / h, 1)
        return centerDepth + (sternDepth - centerDepth) * v * v
    }

    /// Half-beam at the sheer: B/2 at maxBeamX, to nothing at the stem as
    /// 1 − u^1.5, to transomRatio at the transom as 1 − (1 − r)v².
    func beam(_ x: Float) -> Float {
        let h: Float = halfLength
        if x >= maxBeamX {
            let u: Float = min((x - maxBeamX) / (h - maxBeamX), 1)
            return halfBeam * (1 - pow(u, bowBeamPower))
        }
        let v: Float = min((maxBeamX - x) / (maxBeamX + h), 1)
        let k: Float = (1 - transomRatio) * v * v
        return halfBeam * (1 - k)
    }

    /// Height of the hull's bottom (outside the skin, at the centreline,
    /// the keel apart).
    func bottom(_ x: Float) -> Float {
        let h: Float = halfLength
        let u: Float = min(max((x - bowRiseStart) / (h - bowRiseStart), 0), 1)
        let v: Float = min(max((sternRiseStart - x) / (sternRiseStart + h), 0), 1)
        let bow: Float = (bowFoot - keelDepth) * pow(u, bowRisePower)
        let stern: Float = (transomFoot - keelDepth) * pow(v, sternRisePower)
        return keelDepth + bow + stern
    }

    /// Where the section is widest (the ellipse's centre), and the
    /// ellipse's semi-axes: across, the half-beam (never below 1 mm, so the
    /// kernel's ellipse stays an ellipse at the stem); down, to the bottom.
    func centre(_ x: Float) -> Float { max(sheer(x) - topsides, bottom(x) + 1) }
    func semiAcross(_ x: Float) -> Float { max(beam(x), 1) }
    func semiDown(_ x: Float) -> Float { centre(x) - bottom(x) }

    /// The keel's bottom: the baseline aft of amidships (the keel deepening
    /// into the full skeg as the bottom rises to the transom), and a keel
    /// of constant depth following the bottom up to the stem forward.
    func keelBottom(_ x: Float) -> Float { x > 0 ? bottom(x) - keelDepth : 0 }

    /// Outer half-width of the hull body at height y, station x (0 where
    /// there is none): the half-ellipse below the centre, straight sides
    /// above it — the zero set of the kernel's section distance.
    func halfWidth(x: Float, y: Float) -> Float {
        if abs(x) > halfLength { return 0 }
        let k: Float = bottom(x)
        let s: Float = sheer(x)
        if y < k || y > s { return 0 }
        let b: Float = semiAcross(x)
        let c: Float = centre(x)
        if y >= c { return b }
        let t: Float = (c - y) / semiDown(x)
        let t2: Float = t * t
        let root: Float = max(1 - t2, 0).squareRoot()
        return b * root
    }

    /// Area of the station's section below height `level`: the body plus
    /// the keel (and skeg) below it. Trapezoidal rule in y over `n` strips.
    func sectionArea(x: Float, below level: Float, n: Int = 400) -> Double {
        if abs(x) > halfLength { return 0 }
        let k: Float = bottom(x)
        let top: Float = min(level, sheer(x))
        var area: Double = 0
        if top > k {
            let dy: Double = Double(top - k) / Double(n)
            var sum: Double = 0
            for i in 0...n {
                let di: Double = Double(i) * dy
                let y: Float = k + Float(di)
                let wt: Double = (i == 0 || i == n) ? 0.5 : 1
                let w: Float = halfWidth(x: x, y: y)
                let width: Double = Double(2 * w)
                sum += wt * width
            }
            area += sum * dy
        }
        let kb: Float = keelBottom(x)
        let keelTop: Float = min(level, k)
        if keelTop > kb {
            let kw: Double = Double(2 * keelHalfWidth)
            let kh: Double = Double(keelTop - kb)
            area += kw * kh
        }
        return area
    }

    /// Volume of the hull's outer envelope below height `level` (hull
    /// frame), in m³: Simpson's rule over `stations` slices. The cavity is
    /// dry — the sheer is everywhere above the water — so it displaces too.
    func volume(below level: Float, stations: Int = 400) -> Double {
        let n: Int = stations % 2 == 0 ? stations : stations + 1
        let a: Double = Double(-halfLength)
        let dx: Double = Double(length) / Double(n)
        var sum: Double = 0
        for i in 0...n {
            let x: Float = Float(a + Double(i) * dx)
            let w: Double = (i == 0 || i == n) ? 1 : (i % 2 == 1 ? 4 : 2)
            sum += w * sectionArea(x: x, below: level)
        }
        let mm3: Double = sum * dx / 3
        return mm3 * 1e-9
    }

    /// The draft (baseline to waterline) at which the envelope displaces
    /// `kilograms` of water of `density`: bisection to 1 µm.
    func draft(forMass kilograms: Double, density: Double) -> Float {
        let want: Double = kilograms / density
        var lo: Float = 0
        var hi: Float = centerDepth
        for _ in 0..<40 {
            let mid: Float = (lo + hi) / 2
            if volume(below: mid) < want { lo = mid } else { hi = mid }
            if hi - lo < 0.001 { break }
        }
        return (lo + hi) / 2
    }

    /// Length and greatest beam of the waterplane at a draft.
    func waterline(at level: Float) -> (length: Float, beam: Float) {
        var first: Float = .nan, last: Float = .nan
        var widest: Float = 0
        let n: Int = 10_160
        for i in 0...n {
            let x: Float = -halfLength + length * Float(i) / Float(n)
            let w: Float = halfWidth(x: x, y: level)
            let kb: Float = keelBottom(x)
            let wet: Bool = w > 0 || (kb < level && level < bottom(x))
            if wet {
                if first.isNaN { first = x }
                last = x
            }
            widest = max(widest, w)
        }
        return (last - first, 2 * widest)
    }

    /// A constant bound on how fast the section's parameters change along
    /// the boat: M ≥ |b′| + |c′| + |a′| + |s′| + |k′| everywhere (finite
    /// differences, 1 mm apart, with 15% to spare). The section's own
    /// distance (an exact ellipse, an exact box, planes) has unit slope
    /// across; along the boat each parameter moves its boundary no faster
    /// than its own slope, so √(1 + M²) bounds the whole gradient.
    /// The kernel divides by it: a per-object constant, never a local
    /// slope.
    var lipschitz: Float {
        var m: Float = 0
        let n: Int = Int(length)
        for i in 0...n {
            let x: Float = -halfLength + Float(i) - 0.5
            let x2: Float = x + 1
            let db: Float = abs(semiAcross(x2) - semiAcross(x))
            let dk: Float = abs(bottom(x2) - bottom(x))
            let ds: Float = abs(sheer(x2) - sheer(x))
            let dc: Float = abs(centre(x2) - centre(x))
            let da: Float = abs(semiDown(x2) - semiDown(x))
            m = max(m, db + dk + ds + dc + da)
        }
        let mm: Float = m * 1.15
        let m2: Float = mm * mm
        return (1 + m2).squareRoot()
    }
}

func boatScale(_ m: Mutant = activeMutant) -> Float { m == .wrongSize ? 1.2 : 1 }

// MARK: - the fittings

/// Two rowing stations [NW "double oarlocks"]. Where they are along the
/// boat is MODEL (Newfound's plans are not public): the aft station a
/// little aft of amidships, where one rower sits, and one forward of it for
/// a second rower. Each thwart's aft edge is 13 in forward of its oarlocks
/// [Angus]; its top is 6.5 in above the lowest point of the bilge there
/// ([Angus]: 5.5–8 in; 6.5 keeps the forward pair's oarlocks 8 in above
/// it where the bottom rises); it is 9 in deep fore and aft (MODEL) and 7/8 in
/// thick (MODEL).
struct Station {
    let oarlockX: Float
    var seatAftEdge: Float { oarlockX + 13 * inch * sizeScale }
    let sizeScale: Float
    let seatDepth: Float
    let seatTop: Float
    let seatThickness: Float
    var seatCentreX: Float { seatAftEdge + seatDepth / 2 }
}

struct Oarlock {
    /// Where the pin stands: on the oar block, on the sheer line.
    var pin: SIMD3<Float>          // hull frame, top of the block
    /// The ring's centre: the oar's pivot, where its axis crosses the U.
    var pivot: SIMD3<Float>
    /// The ring's axis (horizontal, along the loom's horizontal direction).
    var axis: SIMD3<Float>
    var ringRadius: Float          // centreline of the bronze U
    var tube: Float                // the U's rod radius
}

/// One oar: along `axis` from the handle end (−inboard) through the pivot to
/// the blade's tip (+outboard); the blade lies flat in the plane of the
/// axis and `across` (horizontal).
struct Oar {
    var pivot: SIMD3<Float>
    var axis: SIMD3<Float>
    var across: SIMD3<Float>
    var normal: SIMD3<Float>
    var inboard: Float
    var outboard: Float
    var length: Float { inboard + outboard }
}

/// Oar dimensions other than [S&T]'s length rule and blade width: MODEL,
/// in proportion to common spruce oars (loom 1¾ in, a 1¼ in grip 5 in
/// long, a blade 22 in long, ½ in thick).
let loomRadius: Float = 22.2
let gripRadius: Float = 15.9
let gripLength: Float = 127
let bladeLength: Float = 559
let bladeThickness: Float = 12.7
/// [S&T]: 5½ in for oars under 7 ft 10 in.
func bladeWidth(forLength l: Float) -> Float {
    let limit: Float = 94 * inch
    let standard: Float = 5.5 * inch
    let narrow: Float = 4.625 * inch
    return l < limit ? standard : narrow
}
/// Oarlock: an oar block on the gunwale (mahogany, [NW] "oar blocks"), a
/// bronze pin and U. Sizes MODEL (a common 2-in-throat horn).
let blockHalfLength: Float = 76
let blockHalfWidth: Float = 24
let blockHeight: Float = 25
let oarlockTube: Float = 5
let oarlockCollar: Float = 10
/// Oars at rest: trailed aft alongside, the blades held just clear of the
/// water. The trailing angle is MODEL; the blades' height is set from the
/// waterline (solved), so the oars rest on the physics, not by eye.
let trailDegrees: Float = 12
let bladeClearance: Float = 10

struct Layout {
    let hull: HullShape
    let stations: [Station]
    let oarlocks: [Oarlock]
    let oars: [Oar]
    let oarLengthRule: Float        // [S&T] before rounding
    let span: Float                 // between the aft station's pins
    let sternSeatX: Float
    let sternSeatTop: Float
}

/// The oar's pitch at rest: the lowest point of the blade 10 mm above the
/// water (world y = 0, hull y = drawnDraft). Bisection on the angle.
func restPitch(pivot: SIMD3<Float>, horiz: SIMD3<Float>, across: SIMD3<Float>, outboard: Float, drawnDraft: Float) -> Float {
    func lowest(_ phi: Float) -> Float {
        let a: SIMD3<Float> = horiz * cos(phi) + SIMD3<Float>(0, -sin(phi), 0)
        let n: SIMD3<Float> = simd_cross(a, across)
        let tip: SIMD3<Float> = pivot + a * outboard
        let drop: Float = abs(n.y) * bladeThickness / 2
        return tip.y - drop
    }
    var lo: Float = 0, hi: Float = 0.8
    let target: Float = drawnDraft + bladeClearance
    for _ in 0..<60 {
        let mid: Float = (lo + hi) / 2
        if lowest(mid) > target { lo = mid } else { hi = mid }
    }
    return (lo + hi) / 2
}

/// The pair of oars' geometry in the hull frame, given the draft at which
/// the boat is drawn (the blades rest just clear of that waterline).
func buildLayout(scale s: Float, drawnDraft: Float) -> Layout {
    let hull = HullShape(scale: s)
    func station(_ x: Float) -> Station {
        let seatDepth: Float = 9 * inch * s
        let aft: Float = x + 13 * inch * s
        let mid: Float = aft + seatDepth / 2
        let bilge: Float = hull.bottom(mid) + hull.skin
        return Station(oarlockX: x, sizeScale: s, seatDepth: seatDepth, seatTop: bilge + 6.5 * inch * s,
                       seatThickness: 22 * s)
    }
    let aftStation: Station = station(-300 * s)
    let foreStation: Station = station(1000 * s)
    let stations: [Station] = [aftStation, foreStation]

    // Oar length by [S&T] from the aft station's span.
    let zPin0: Float = hull.beam(aftStation.oarlockX) - hull.skin / 2
    let span: Float = 2 * zPin0
    let inboardRule: Float = (span / inch / 2 + 2) * inch
    let lengthRule: Float = inboardRule * 25 / 7
    let halfFeet: Float = (lengthRule / inch / 6).rounded()
    let oarLength: Float = halfFeet * 6 * inch
    let inboard: Float = oarLength * 7 / 25
    let outboard: Float = oarLength - inboard

    var locks: [Oarlock] = []
    var oars: [Oar] = []
    for st in stations {
        let x: Float = st.oarlockX
        let zp: Float = hull.beam(x) - hull.skin / 2
        for side in [Float(1), Float(-1)] {
            let pin = SIMD3<Float>(x, hull.sheer(x) + blockHeight, side * zp)
            let isAft: Bool = st.oarlockX == aftStation.oarlockX
            // The ring's axis: along the trailed oar's horizontal direction at
            // the aft station (the U swivels on its pin to square up to the
            // loom); across the boat at the forward, idle pair.
            let th: Float = trailDegrees * .pi / 180
            let horiz: SIMD3<Float> = isAft ? SIMD3<Float>(-cos(th), 0, side * sin(th)) : SIMD3<Float>(0, 0, side)
            // The U is sized so the loom rests on it: the loom, pitched φ,
            // meets the ring's plane as an ellipse loomRadius/cos φ deep,
            // and across the rod's thickness it drops tube·tan φ more; a
            // 0.3 mm margin. φ comes from the blades' rest (below), so
            // solve twice. The forward, idle pair is the same bronze.
            var ringR: Float = 0
            var pivot: SIMD3<Float> = pin
            var phi: Float = 15 * .pi / 180
            let across: SIMD3<Float> = simd_normalize(simd_cross(SIMD3<Float>(0, 1, 0), horiz))
            for _ in 0..<3 {
                ringR = loomRadius / cos(phi) + oarlockTube + oarlockTube * tan(phi) + 0.3
                pivot = pin + SIMD3<Float>(0, oarlockCollar + oarlockTube + ringR, 0)
                phi = restPitch(pivot: pivot, horiz: horiz, across: across, outboard: outboard, drawnDraft: drawnDraft)
            }
            locks.append(Oarlock(pin: pin, pivot: pivot, axis: horiz, ringRadius: ringR, tube: oarlockTube))
            if isAft {
                let a: SIMD3<Float> = horiz * cos(phi) + SIMD3<Float>(0, -sin(phi), 0)
                var n: SIMD3<Float> = simd_cross(a, across)
                if n.y < 0 { n = -n }
                oars.append(Oar(pivot: pivot, axis: a, across: across, normal: n, inboard: inboard, outboard: outboard))
            }
        }
    }
    let sternX: Float = -hull.halfLength + 480 * s
    let sternTop: Float = hull.bottom(sternX) + hull.skin + 7 * inch * s
    return Layout(hull: hull, stations: stations, oarlocks: locks, oars: oars, oarLengthRule: lengthRule, span: span,
                  sternSeatX: sternX, sternSeatTop: sternTop)
}

/// An oar's wood volume (m³) as drawn: grip and loom as cylinders (their
/// round ends ignored), the blade as a flat slab.
func oarVolume(_ o: Oar) -> Double {
    let bw: Float = bladeWidth(forLength: o.length)
    let loomLen: Float = o.length - gripLength - bladeLength
    let gripArea: Float = Float.pi * gripRadius * gripRadius
    let loomArea: Float = Float.pi * loomRadius * loomRadius
    let grip: Double = Double(gripArea * gripLength)
    let loom: Double = Double(loomArea * loomLen)
    let blade: Double = Double(bw * bladeThickness * bladeLength)
    return (grip + loom + blade) * 1e-9
}

// MARK: - what floats

struct Buoyancy {
    let hullKilograms: Double
    let oarKilograms: Double
    var totalKilograms: Double { hullKilograms + oarKilograms }
    let density: Double
    /// Draft from buoyancy: baseline (keel bottom) to waterline.
    let draft: Float
    let displacedCubicMetres: Double
    /// With one rower [Walpole] aboard.
    let draftWithRower: Float
    /// At Newfound's 725-lb full load, in fresh and in sea water.
    let draftAtCapacityFresh: Float
    let draftAtCapacitySea: Float
}

func solveBuoyancy(scale s: Float) -> Buoyancy {
    let hull = HullShape(scale: s)
    // The oars' pose does not change their mass; lay them out at any draft.
    let layout: Layout = buildLayout(scale: s, drawnDraft: 100)
    // A bigger boat of the same build weighs as its volume: s³. Newfound's
    // weight is for their size.
    let s3: Double = Double(s * s * s)
    let hullKg: Double = nwWeightPounds * poundKilograms * s3
    var oarKg: Double = 0
    for o in layout.oars { oarKg += oarVolume(o) * spruceDensity }
    let total: Double = hullKg + oarKg
    let t: Float = hull.draft(forMass: total, density: seawaterDensity)
    let v: Double = hull.volume(below: t)
    let tr: Float = hull.draft(forMass: total + rowerKilograms, density: seawaterDensity)
    let cap: Double = nwCapacityPounds * poundKilograms
    let tf: Float = hull.draft(forMass: cap, density: freshwaterDensity)
    let ts: Float = hull.draft(forMass: cap, density: seawaterDensity)
    return Buoyancy(hullKilograms: hullKg, oarKilograms: oarKg, density: seawaterDensity, draft: t,
                    displacedCubicMetres: v, draftWithRower: tr, draftAtCapacityFresh: tf, draftAtCapacitySea: ts)
}

/// The draft the picture is drawn at: buoyancy's, or the mutant's by-eye
/// 30% deeper.
func drawnDraft(_ b: Buoyancy, _ m: Mutant = activeMutant) -> Float { m == .floatsWrong ? b.draft * 1.3 : b.draft }

// MARK: - materials, all MODEL except where cited

/// Varnished cedar strips: warm orange-brown under a clear coat.
let cedarAlbedo = SIMD3<Float>(0.66, 0.32, 0.12)
/// Mahogany wales, keel, stem and oar blocks: darker, redder.
let mahoganyAlbedo = SIMD3<Float>(0.30, 0.11, 0.05)
/// Spanish cedar thwarts: a paler brown.
let seatAlbedo = SIMD3<Float>(0.50, 0.30, 0.16)
/// Spruce oars, varnished: pale straw.
let spruceAlbedo = SIMD3<Float>(0.70, 0.52, 0.30)
/// Bronze oarlocks: a metal's reflectance, warm. MODEL.
let bronzeF0 = SIMD3<Float>(0.80, 0.55, 0.32)
/// Clear varnish's index: 1.5, MODEL (typical of clear coatings).
let varnishIndex: Double = 1.5
let varnishRoughness: Float = 0.18
/// Sea water's colour: the radiance of deep water seen from above, and how
/// fast it dims the hull below the surface (per mm). MODEL, chosen so the
/// keel still shows through at full draft.
let deepWater = SIMD3<Float>(0.020, 0.075, 0.090)
let waterExtinction = SIMD3<Float>(0.0120, 0.0050, 0.0060)

// MARK: - tone and colour, as in steps 20, 53, 60 and 72

func srgbEncode(_ v: Double) -> Double {
    let c: Double = min(max(v, 0), 1)
    return c <= 0.0031308 ? 12.92 * c : 1.055 * pow(c, 1.0 / 2.4) - 0.055
}

/// Dielectric Fresnel reflectance, unpolarised.
func fresnelDielectric(n: Double, cosTheta c0: Double) -> Double {
    let c: Double = min(max(c0, 0), 1)
    let s2: Double = 1 - c * c
    let t2: Double = 1 - s2 / (n * n)
    if t2 <= 0 { return 1 }
    let ct: Double = t2.squareRoot()
    let rs: Double = (c - n * ct) / (c + n * ct)
    let rp: Double = (n * c - ct) / (n * c + ct)
    return (rs * rs + rp * rp) / 2
}

let toneGain: Double = 1.05
