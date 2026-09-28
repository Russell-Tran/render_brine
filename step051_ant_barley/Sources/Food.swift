// Step 51: the object — one grain of cooked pearled barley lying on the card,
// at true scale beside a 4 mm ant, in millimetres; its thin wet film; its
// colour; and why it smells.

import Foundation
import simd

// MARK: - how big a cooked barley grain is

// Raw grain. Wang et al. 2018 (*Processes* 6: 230, doi:10.3390/pr6110230,
// §3.1, read there) measured dehulled barley grains with a vernier caliper:
// "a = 2.89 mm, b = 1.59 mm, c = 1.06 mm", and modelled the grain as an
// ellipsoid ("a one-eighth ellipsoid was created", §2) — so a, b, c are read
// here as HALF-axes: a raw grain about 5.8 × 3.2 × 2.1 mm. (That reading is
// an inference from the one-eighth model; and the paper's grains are
// dehulled, not stated to be pearled. UNVERIFIED for pearled grain.)
let rawHalfAxes = SIMD3<Float>(2.89, 1.59, 1.06)     // mm: long, wide, thick

// Swelling on cooking. USDA FoodData Central, SR Legacy: pearled barley,
// raw, water 10.09 g/100 g (fdcId 170284); cooked, 68.8 g/100 g (fdcId
// 170285). The dry matter is unchanged, so a cooked grain weighs
// 0.8991 / 0.312 = 2.88 × the raw one. MODEL: that gain is water taken up
// evenly, the grain's density little changed, so every length grows by the
// cube root of 2.88, 1.42×. Wang et al. measured a volume expansion ratio of
// 2.62 after 16 min of cooking (§4, their Fig. 6) — a cube root of 1.38 —
// which agrees. 1.4 is used.
let rawWater: Float = 10.09          // g/100 g
let cookedWater: Float = 68.8        // g/100 g
var massGain: Float { (100 - rawWater) / (100 - cookedWater) }
let linearSwelling: Float = 1.4

/// The cooked grain's half-axes as the grain lies — along it, up, across —
/// mm: 4.05, 1.48 (Wang's c, the thickness, upright) and 2.23 (b, the
/// width): a grain about 8.1 long, 4.5 wide and 3.0 mm thick.
var cookedHalfAxes: SIMD3<Float> { SIMD3<Float>(rawHalfAxes.x, rawHalfAxes.z, rawHalfAxes.y) * linearSwelling }

/// The wet film on the cooked grain, mm. MODEL: Wang et al. weighed their
/// grains "After the removal of the surface water", so a drained grain
/// carries some; no measurement of its thickness was found. 5 µm
/// is chosen: invisible in the main view, as a real film is, and in the
/// micrometre inset a layer of water the grain's surface shows through.
let filmThicknessMM: Float = 0.005

// The science mutant draws the grain far too small — 0.45 of its size,
// 3.6 mm long, shorter than the ant: every length shrunk by this factor.
func grainScale(_ m: Mutant) -> Float { m == .grainSize ? 0.45 : 1.0 }

/// The grain: an ellipsoid (Wang et al.'s shape) resting on the card, its
/// long axis level and turned by `yaw` about the vertical, its thickness
/// upright — lying on its flattest face, as a grain settles. The film is the
/// same ellipsoid's surface pushed out by `film` along its normal.
struct Grain {
    var centre: SIMD3<Float>
    var axes: SIMD3<Float>       // half-axes: along the grain, up, across
    var yaw: Float
    var film: Float

    var long: SIMD3<Float> { SIMD3<Float>(cos(yaw), 0, sin(yaw)) }
    var across: SIMD3<Float> { SIMD3<Float>(-sin(yaw), 0, cos(yaw)) }

    /// World to the grain's own frame (x along it, y up, z across).
    func local(_ p: SIMD3<Float>) -> SIMD3<Float> {
        let q: SIMD3<Float> = p - centre
        return SIMD3<Float>(simd_dot(q, long), q.y, simd_dot(q, across))
    }
    func world(_ q: SIMD3<Float>) -> SIMD3<Float> {
        let along: SIMD3<Float> = long * q.x
        let side: SIMD3<Float> = across * q.z
        return centre + along + SIMD3<Float>(0, q.y, 0) + side
    }
    func worldDirection(_ q: SIMD3<Float>) -> SIMD3<Float> {
        let along: SIMD3<Float> = long * q.x
        let side: SIMD3<Float> = across * q.z
        return along + SIMD3<Float>(0, q.y, 0) + side
    }
}

// MARK: - distance to an ellipsoid, exactly

/// The distance from a point to an ellipsoid with half-axes `e`, centred at
/// the origin, in its own frame. Outside, it is exact: the nearest surface
/// point x lies where the line to q is along the surface normal (Lagrange's
/// condition), q − x = t (x_i / e_i²), so x_i = e_i² y_i / (t + e_i²), and t
/// is the root of G(t) = Σ (e_i y_i / (t + e_i²))² − 1 (derived here, not
/// copied; the tests check it against a brute-force search of the surface).
/// G falls and is convex for t ≥ 0, so
/// Newton's method started below the root climbs to it without overshoot;
/// the root lies between S − e_max² and S − e_min², S = |(e_i y_i)|, and it
/// starts at the lower end. Inside, a safe bound, (|q/e| − 1)·e_min: never
/// deeper than the truth. The kernel runs the same arithmetic (Render.swift).
let ellipsoidNewtonSteps: Int = 28

func ellipsoidDistance(_ q: SIMD3<Float>, _ e: SIMD3<Float>) -> Float {
    let y: SIMD3<Float> = abs(q)
    let k: Float = simd_length(y / e)
    let eMin: Float = min(e.x, min(e.y, e.z))
    let eMax: Float = max(e.x, max(e.y, e.z))
    if k <= 1 { return (k - 1) * eMin }
    let e2: SIMD3<Float> = e * e
    let ey: SIMD3<Float> = e * y
    let s: Float = simd_length(ey)
    var t: Float = max(s - eMax * eMax, 0)
    for _ in 0..<ellipsoidNewtonSteps {
        let d: SIMD3<Float> = t + e2
        let r: SIMD3<Float> = ey / d
        let g: Float = simd_dot(r, r) - 1
        let r3: SIMD3<Float> = r * r / d
        let dg: Float = -2 * (r3.x + r3.y + r3.z)
        if g <= 0 || dg >= 0 { break }
        t -= g / dg
    }
    let x: SIMD3<Float> = e2 * y / (t + e2)
    return simd_distance(x, y)
}

/// The grain with its film, as a distance (mm): the ellipsoid's own distance
/// less the film's thickness — exact outside, since the film is the surface
/// pushed out by a constant along its normal — cut by the card.
func grainSDF(_ p: SIMD3<Float>, _ g: Grain) -> Float {
    let d: Float = ellipsoidDistance(g.local(p), g.axes) - g.film
    return max(d, -p.y)
}

/// The bare grain, under its film.
func bareGrainSDF(_ p: SIMD3<Float>, _ g: Grain) -> Float { ellipsoidDistance(g.local(p), g.axes) }

// MARK: - where it lies, and where the antenna touches

/// The contact's height on the grain, mm. MODEL: a little below the grain's
/// waist, about level with the ant's antennal elbow, as step 42 set it on
/// the macaroni — a grain 3 mm tall overhangs an ant 1 mm tall, and a
/// funiculus bent up and over it would pass through it.
let contactHeight: Float = 1.25
/// Where on the grain's outline: the angle round its waist from the +x end
/// of its long axis, in the grain's own frame (±180° is the −x end). MODEL:
/// −135°, on the shoulder of the −x end, so the grain lies ahead of the ant
/// and runs away from it, and all of it stays in the picture.
let contactAngle: Float = -135 * Float.pi / 180
/// Which way the film's normal at the contact faces, level: back at the ant
/// (−x) and turned 26° towards the camera (+z), so the camera sees the tip
/// on the grain rather than behind its edge. The grain is turned to make it
/// so. MODEL.
let contactDirection: SIMD2<Float> = simd_normalize(SIMD2<Float>(-0.9, 0.44))
/// The antenna reaches from its elbow forward and out to the right; the
/// tip's centre 0.97 of the funiculus's length away, as in step 42. MODEL.
let reachDirection: SIMD2<Float> = simd_normalize(SIMD2<Float>(0.80, 0.60))
/// Which way the right funiculus bows: back and up, away from the grain.
let rightFuniculusBow = SIMD3<Float>(-1, 0.5, 0)
let reachFraction: Float = 0.97

/// The contact on the bare grain in its own frame, and the normal there.
func localContact(_ axes: SIMD3<Float>) -> (point: SIMD3<Float>, normal: SIMD3<Float>) {
    let y0: Float = contactHeight - axes.y
    let s: Float = max(1 - (y0 * y0) / (axes.y * axes.y), 0).squareRoot()
    let q = SIMD3<Float>(axes.x * s * cos(contactAngle), y0, axes.z * s * sin(contactAngle))
    let n: SIMD3<Float> = simd_normalize(q / (axes * axes))
    return (q, n)
}

/// The turn that makes the contact's normal face `contactDirection`.
func grainYaw(_ axes: SIMD3<Float>) -> Float {
    let n: SIMD3<Float> = localContact(axes).normal
    let alpha: Float = atan2(n.z, n.x)
    let want: Float = atan2(contactDirection.y, contactDirection.x)
    return want - alpha
}

/// The contact point in x, z: out along reachDirection from the elbow until
/// the tip's centre (the contact plus a tip radius along the normal) is
/// reachFraction of the funiculus's length from the elbow. Bisection.
func solveContactXZ(normal n: SIMD3<Float>, height y: Float) -> SIMD2<Float> {
    let elbow: SIMD3<Float> = rightElbow()
    let total: Float = funiculusLengths(count: workerAntennaSegments - 1).reduce(0, +)
    var lo: Float = 0
    var hi: Float = 3
    for _ in 0..<60 {
        let mid: Float = (lo + hi) / 2
        let xz: SIMD2<Float> = SIMD2<Float>(elbow.x, elbow.z) + reachDirection * mid
        let tip: SIMD3<Float> = SIMD3<Float>(xz.x, y, xz.y) + n * funiculusTipRadius
        if simd_distance(tip, elbow) < reachFraction * total { lo = mid } else { hi = mid }
    }
    return SIMD2<Float>(elbow.x, elbow.z) + reachDirection * lo
}

func buildGrain(mutant: Mutant) -> Grain {
    let full: SIMD3<Float> = cookedHalfAxes
    let yaw: Float = grainYaw(full)
    let probe = Grain(centre: .zero, axes: full, yaw: yaw, film: filmThicknessMM)
    let (q, nLocal) = localContact(full)
    let n: SIMD3<Float> = probe.worldDirection(nLocal)
    let filmPoint: SIMD3<Float> = probe.world(q) + n * filmThicknessMM       // relative to a centre at the origin
    let xz: SIMD2<Float> = solveContactXZ(normal: n, height: filmPoint.y + full.y)
    var g = probe
    g.centre = SIMD3<Float>(xz.x - filmPoint.x, full.y, xz.y - filmPoint.z)
    // The grainSize mutant keeps the contact and shrinks the grain about it
    // (and settles it on the card).
    let k: Float = grainScale(mutant)
    if k != 1 {
        let contact: SIMD3<Float> = g.world(q) + n * filmThicknessMM
        var small = g
        small.axes = full * k
        let (qs, _) = localContact(small.axes)
        let ns: SIMD3<Float> = small.worldDirection(localContact(small.axes).normal)
        small.centre = .zero
        let rel: SIMD3<Float> = small.world(qs) + ns * filmThicknessMM
        small.centre = SIMD3<Float>(contact.x - rel.x, small.axes.y, contact.z - rel.z)
        return small
    }
    return g
}

/// Where the right antenna's tip touches the film: the contact point on the
/// film's surface, and the exact normal there (the ellipsoid's own, which the
/// film shares).
func contactPoint(_ g: Grain) -> (point: SIMD3<Float>, normal: SIMD3<Float>) {
    let (q, nLocal) = localContact(g.axes)
    let n: SIMD3<Float> = simd_normalize(g.worldDirection(nLocal))
    return (g.world(q) + n * g.film, n)
}

func contactPoint() -> (point: SIMD3<Float>, normal: SIMD3<Float>) { contactPoint(buildGrain(mutant: .none)) }

/// The grain's tightest curvature at the contact, as a radius (mm): the
/// normal curvature κ(d) = Σ dᵢ²/eᵢ² / |∇| over tangent directions d, at its
/// largest. The micrometre inset curves the film by this radius — at over a
/// millimetre it is flat across the inset's 14.5 µm either way.
func contactCurvatureRadius(_ g: Grain) -> Float {
    let (q, _) = localContact(g.axes)
    let grad: SIMD3<Float> = q / (g.axes * g.axes)
    let n: SIMD3<Float> = simd_normalize(grad)
    let helper: SIMD3<Float> = abs(n.y) < 0.9 ? SIMD3<Float>(0, 1, 0) : SIMD3<Float>(1, 0, 0)
    let t1: SIMD3<Float> = simd_normalize(simd_cross(n, helper))
    let t2: SIMD3<Float> = simd_cross(n, t1)
    var kMax: Float = 0
    for i in 0..<360 {
        let a: Float = Float(i) * Float.pi / 180
        let c: Float = cos(a)
        let sn: Float = sin(a)
        let d1: SIMD3<Float> = t1 * c
        let d2: SIMD3<Float> = t2 * sn
        let d: SIMD3<Float> = d1 + d2
        let dd: SIMD3<Float> = d * d
        let e2: SIMD3<Float> = g.axes * g.axes
        let inv: SIMD3<Float> = dd / e2
        let k: Float = (inv.x + inv.y + inv.z) / simd_length(grad)
        kMax = max(kMax, k)
    }
    return 1 / kMax
}

// MARK: - colour

/// CIELAB (D65) to linear sRGB, the standard formulas.
func labToLinearSRGB(_ lab: SIMD3<Float>) -> SIMD3<Float> {
    let fy: Float = (lab.x + 16) / 116
    let fx: Float = fy + lab.y / 500
    let fz: Float = fy - lab.z / 200
    func finv(_ t: Float) -> Float { t > 6.0 / 29.0 ? t * t * t : 3 * pow(6.0 / 29.0, 2) * (t - 4.0 / 29.0) }
    let X: Float = 0.95047 * finv(fx)
    let Y: Float = 1.0 * finv(fy)
    let Z: Float = 1.08883 * finv(fz)
    let r: Float = 3.2406 * X - 1.5372 * Y - 0.4986 * Z
    let g: Float = -0.9689 * X + 1.8758 * Y + 0.0415 * Z
    let b: Float = 0.0557 * X - 0.2040 * Y + 1.0570 * Z
    return simd_clamp(SIMD3<Float>(r, g, b), SIMD3<Float>(repeating: 0), SIMD3<Float>(repeating: 1))
}

/// Cooked pearled barley: a pale, slightly greyish cream. MODEL colour — no
/// measured L*a*b* for cooked pearled barley was found.
let grainLab = SIMD3<Float>(74, 1, 10)
let grainAlbedo: SIMD3<Float> = labToLinearSRGB(grainLab)

// MARK: - optics

/// Normal-incidence Fresnel reflectance between two media.
func fresnelF0(_ n1: Double, _ n2: Double) -> Float {
    let r: Double = (n1 - n2) / (n1 + n2)
    return Float(r * r)
}

/// Insect cuticle, n ≈ 1.56: Leertouwer, Wilts & Stavenga, *Opt Express* 19:
/// 24061 (2011), measured butterfly-scale chitin at 1.56 in the visible and
/// that value is the one used across insect optics. No measurement of ant
/// cuticle itself was found; the ant is taken to be chitin like the rest.
let cuticleIndex: Double = 1.56
/// The film is water: its gloss is water's, n = 1.333.
let waterIndex: Double = 1.333

let cuticleF0: Float = fresnelF0(1.0, cuticleIndex)
let waterF0: Float = fresnelF0(1.0, waterIndex)

// MARK: - what the grain is made of, and what the ant can find there

// Pearled barley carries (1,3;1,4)-β-glucan, the cell-wall fibre: 5.8 ± 0.3%
// and 6.2 ± 0.3% of dry weight in the two cultivars Kohyama & Yanagisawa
// measured (*Food Sci Technol Res* 30: 223, 2024, doi:10.3136/fstr.FSTR-D-
// 23-00120). Cereal β-glucans are chains of "~25 and ~75% of (1,3)- and
// (1,4)-β-linked glucosyl units", made "largely of cellotriosyl and
// cellotetraosyl units separated by single (1,3)-β-linkages" (Purushotham et
// al., *Sci Adv* 8: eadd1596, 2022). The taste inset shows one such piece,
// a cellotriosyl and a cellotetraosyl unit joined by one β-(1→3) link —
// as fibre, NOT as a taste: no ant or insect taste response to β-glucan was
// found, and starch, the grain's main carbohydrate, is not shown to be
// tasted either — in the blowfly it inhibits the sugar receptor site
// (Ahamed et al., *Chem Senses* 26: 507, 2001). Nothing enters the taste
// pore here.
let fibreIsATaste: Bool = false

// MARK: - why it smells

// Cooked barley smells. Kohyama & Yanagisawa 2024 (above), quoting Kaneko et
// al. 2013's GC-olfactometry of cooked rolled barley, name "Four aldehydes,
// hexanal, 2-nonenal, 2,4-nonadienal (E,E), and 2,4-decadienal (E,E)" as its
// key odorants, and hexanal as one with "a green or grassy odor" (also "no
// pyrazines or pyrroles were found" in cooked, unlike roasted, barley).
// Hexanal is the odorant drawn reaching the smell hair.
let objectHasOdour: Bool = true

/// The film's surface where the hair touches, as the micrometre inset sees
/// it: a sphere of the grain's tightest radius there, in µm.
var filmSurfaceRadiusMicrometres: Float {
    let g: Grain = buildGrain(mutant: .none)
    return (contactCurvatureRadius(g) + g.film) * 1000
}
/// The film's thickness in the inset, µm: the same film.
var filmThicknessMicrometres: Float { filmThicknessMM * 1000 }
