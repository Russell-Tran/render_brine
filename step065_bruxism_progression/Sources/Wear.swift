// Bruxism as numbers: where the grinding flattens each lower tooth, how deep,
// how thick the enamel is that it wears through, and how many years that takes.
// Nothing here touches the GPU; it is the part a test reads against the papers.
//
// Tooth-local coordinates, as step 20's kernel uses them: u runs distally along
// the arch, v runs outward (labially or buccally, away from the tongue), y is
// up, and y = 0 is the occlusal plane, the top of the unworn crowns.
//
// What the papers say, and what this file takes from them:
//
//   * Attrition — wear from tooth grinding on tooth, with no food between —
//     "is characterised by the facet that is matched by a corresponding facet
//     on a tooth in the opposing arch. When dentine is exposed, it remains flat
//     with no 'cupping' or 'scooping' … In general, well-defined, shiny facets
//     is a good measure for active attrition." Cupped dentine belongs to
//     abrasion (food) and erosion (acid), not to grinding (Kaidonis, "Tooth
//     wear: the view of the anthropologist", Clin Oral Investig 2008, 12 Suppl
//     1:S21–S26, PMC2563149). So a facet here is ONE PLANE, cut straight
//     through enamel and dentine alike, and the dentine is flush with it.
//   * As wear goes on, the occlusion goes "from a 'canine-rise' occlusion, to
//     group function, to a flat occlusal plane with an associated edge-to-edge
//     anterior bite", and "the buccal cusps of the lower molars … wear faster"
//     (Kaidonis 2008). That sets which way each facet leans.
//   * The stage is Smith & Knight's Tooth Wear Index (Br Dent J 1984;156:435,
//     "An index for measuring the wear of teeth"), per surface, criteria as
//     tabulated in Bardsley, Clin Oral Investig 2008 (PMC2238784) and Saczuk et
//     al., J Clin Med 2019 (PMC6678144), which agree word for word.

import Foundation
import simd

// MARK: - rates, and so the years

/// How fast a bruxist's incisal edges lose height: "Annual decrease in mean
/// crown length was determined as 20–30 µm for group 1 and 40–50 µm for group
/// 2" — group 2 being nocturnal bruxists without an occlusal splint, measured on
/// maxillary incisors for four years (Korkut, Tagtekin, Murat & Yanikoglu,
/// "Clinical Quantitative Evaluation of Tooth Wear: A 4-year Longitudinal
/// Study", Oral Health Prev Dent 2020, PMC11654528; abstract checked, full text
/// not reachable). The middle of 40–50.
///
/// Applied to every lower tooth. MODEL twice over: the lower incisors are taken
/// to lose what the upper ones they grind against lose (attrition facets come
/// in matched pairs, Kaidonis 2008), and no in-vivo rate for a bruxist's
/// posterior teeth was found, so they wear at the same rate. The rate is held
/// constant through dentine too: no source gives dentine's attrition rate, and
/// in attrition it wears flush with the enamel around it rather than faster.
let bruxistWearRate: Float = 0.045           // mm per year

/// For comparison, ordinary chewing: "The average steady-wear rate on occlusal
/// contact areas was about 29 microns per year for molars and about 15 microns
/// per year for premolars" (Lambrechts, Braem, Vuylsteke-Wauters & Vanherle,
/// "Quantitative in vivo wear of human enamel", J Dent Res 1989;68:1752, abstract
/// checked). Pintado et al. (J Prosthet Dent 1997;77:313) measured 10.7 µm of mean
/// depth in a year across young adults' teeth. Not drawn; quoted in the caption.
let normalMolarWearRate: Float = 0.029
let normalPremolarWearRate: Float = 0.015

// MARK: - enamel, measured

/// Maxillary central and lateral incisors on CBCT, 324 teeth, ages 14–68
/// (Al-Zahawi et al., "Age and sex related change in tooth enamel thickness of
/// maxillary incisors measured by cone beam computed tomography", BMC Oral
/// Health 2023, PMC10701974, Table 2): mid-incisal enamel 1.03 ± 0.4 mm; incisal
/// edge to pulp 5.2 ± 1.06 mm; facial enamel 0.58, 0.79 and 0.99 mm at 1, 3 and
/// 5 mm above the cementoenamel junction. The lower incisors are taken as the
/// same — MODEL: they are smaller teeth, and no mandibular measurement was
/// reachable.
let incisalEnamel: Float = 1.03
let incisalEnamelToPulp: Float = 5.2
let facialEnamelAt1mm: Float = 0.58
let facialEnamelAt5mm: Float = 0.99

/// Enamel over the canine's cusp tip. MODEL: the incisors' measured incisal
/// thickness; no canine measurement was reachable.
let canineCuspEnamel: Float = 1.03

/// Enamel over a premolar's or molar's buccal cusp, measured vertically. MODEL,
/// and the one number here that decides whether dentine shows on the back
/// teeth: oral histology texts put enamel at its thickest, up to ~2.5 mm, over
/// cusps, but no measurement of the lower buccal cusps could be reached to
/// check. 1.6 mm is below that maximum, as the worn, functional cusps are.
let posteriorCuspEnamel: Float = 1.6

/// The vertical enamel over the top of each kind of crown.
func topEnamel(_ kind: CrownKind) -> Float {
    switch kind {
    case .incisor: return incisalEnamel
    case .canine: return canineCuspEnamel
    default: return posteriorCuspEnamel
    }
}

/// Enamel on the crown's walls, as a line through Al-Zahawi's 1 mm and 5 mm
/// facial values — 0.10 mm more per mm up the crown — from nothing at the
/// junction (enamel ends in a feather edge there), and no thicker than over the
/// top. The walls of the premolars and molars get the same line: MODEL.
let wallEnamelSlope: Float = (facialEnamelAt5mm - facialEnamelAt1mm) / 4
let wallEnamelAtJunction: Float = facialEnamelAt1mm - wallEnamelSlope

func wallEnamel(heightAboveJunction h: Float, kind: CrownKind) -> Float {
    guard h > 0 else { return 0 }
    return min(wallEnamelAtJunction + wallEnamelSlope * h, topEnamel(kind))
}

// MARK: - where the facets go, and which way they face

/// How far each facet leans outward from horizontal, degrees: its normal tips
/// toward the lip or cheek by this much.
///
/// MODEL, in direction sourced and in size chosen. The upper incisors and
/// canines overlap the lower ones on the outside, so the lower anterior teeth
/// are worn on the outer edge of their incisal edges and cusp tips by the
/// upper teeth's inner surfaces sliding over them — a facet facing up and
/// labially. Heavy wear goes on towards a flat, edge-to-edge bite (Kaidonis
/// 2008), so the lean is shallower than the ~40° or more of an unworn incisal
/// guidance. The lower molars' buccal cusps wear faster than their lingual ones
/// (Kaidonis 2008), so the back teeth's facets lean buccally, gently.
func facetTiltDegrees(_ kind: CrownKind) -> Float {
    switch kind {
    case .incisor, .canine: return 20
    default: return 10
    }
}

/// The facet's unit normal in tooth-local (u, v, y): pointing out of the tooth,
/// up and outward.
func facetNormalLocal(_ kind: CrownKind) -> SIMD3<Float> {
    let a: Float = facetTiltDegrees(kind) * Float.pi / 180
    return SIMD3<Float>(0, sin(a), cos(a))
}

// MARK: - the crown's top, on the CPU

// Step 20's occlusal height field, the same formulas as its kernel's
// `crownTop`, so the facet can be anchored at the surface's real height. A test
// checks this copy against the kernel, point by point.
let cuspK: Float = 0.3
let cuspBlend: Float = 0.7
let cuspSlope: Float = 1.0

func sminCPU(_ a: Float, _ b: Float, _ k: Float) -> Float {
    let h: Float = min(max(0.5 + 0.5 * (b - a) / k, 0), 1)
    let mixed: Float = b + (a - b) * h
    return mixed - k * h * (1 - h)
}
func smaxCPU(_ a: Float, _ b: Float, _ k: Float) -> Float { -sminCPU(-a, -b, k) }

func domeCPU(_ uv: SIMD2<Float>, _ cu: Float, _ cv: Float, _ h: Float) -> Float {
    let r: Float = simd_length(uv - SIMD2<Float>(cu, cv))
    let r0: Float = cuspSlope / (2 * cuspK)
    if r < r0 { return h - cuspK * r * r }
    let base: Float = h - cuspK * r0 * r0
    return base - cuspSlope * (r - r0)
}

/// The unworn crown's top height at tooth-local (u, v), clamped into the
/// crown's outline as the kernel clamps it.
func crownTopCPU(_ spec: ToothSpec, _ uvIn: SIMD2<Float>) -> Float {
    let a: Float = spec.width / 2
    let b: Float = spec.depth / 2
    let uv = SIMD2<Float>(min(max(uvIn.x, -a), a), min(max(uvIn.y, -b), b))
    switch spec.kind {
    case .incisor:
        let un: Float = uv.x / a
        return -0.45 * un * un * un * un
    case .canine:
        let du: Float = uv.x + 0.12 * a
        let k: Float = du < 0 ? 1.05 : 0.8
        let ridge: Float = (du * du + 0.5).squareRoot() - Float(0.5).squareRoot()
        return -k * ridge - 0.25 * max(-uv.y, 0)
    case .firstPremolar:
        return smaxCPU(domeCPU(uv, 0, 0.3 * b, 0), domeCPU(uv, 0, -0.5 * b, -2.2), cuspBlend)
    case .secondPremolar:
        var h: Float = smaxCPU(domeCPU(uv, 0, 0.3 * b, 0), domeCPU(uv, -0.35 * a, -0.5 * b, -1.0), cuspBlend)
        h = smaxCPU(h, domeCPU(uv, 0.4 * a, -0.5 * b, -1.3), cuspBlend)
        return h
    case .firstMolar:
        var h: Float = smaxCPU(domeCPU(uv, -0.45 * a, -0.5 * b, 0), domeCPU(uv, 0.35 * a, -0.5 * b, -0.2), cuspBlend)
        h = smaxCPU(h, domeCPU(uv, -0.55 * a, 0.5 * b, -0.3), cuspBlend)
        h = smaxCPU(h, domeCPU(uv, 0, 0.55 * b, -0.4), cuspBlend)
        h = smaxCPU(h, domeCPU(uv, 0.62 * a, 0.3 * b, -0.8), cuspBlend)
        return h
    case .secondMolar:
        var h: Float = smaxCPU(domeCPU(uv, -0.45 * a, -0.5 * b, 0), domeCPU(uv, 0.45 * a, -0.5 * b, -0.2), cuspBlend)
        h = smaxCPU(h, domeCPU(uv, -0.45 * a, 0.5 * b, -0.3), cuspBlend)
        h = smaxCPU(h, domeCPU(uv, 0.45 * a, 0.5 * b, -0.4), cuspBlend)
        return h
    }
}

/// Step 20's crown walls, the same formulas as its kernel's `crownSide`.
func crownSideCPU(_ spec: ToothSpec, _ uv: SIMD2<Float>, _ y: Float) -> Float {
    let H: Float = spec.crownHeight
    let t: Float = (y + H) / H
    let hx: Float = spec.width / 2
    let hcx: Float = spec.cervicalWidth / 2
    let hz: Float = spec.depth / 2
    let hcz: Float = spec.cervicalDepth / 2
    var a: Float
    var b: Float
    if t >= 0 {
        a = hcx + (hx - hcx) * smoothstepCPU(0, contactHeight(spec.kind), t)
        if t < 0.3 {
            b = hcz + (hz - hcz) * smoothstepCPU(0, 0.3, t)
        } else {
            let top: Float = topHalfDepth(spec)
            b = hz + (top - hz) * smoothstepCPU(0.3, 1, t)
        }
    } else {
        let below: Float = y + H
        a = hcx * max(1 + 0.05 * below, 0.3)
        b = hcz * max(1 + 0.05 * below, 0.3)
    }
    let r: Float = min(a, b) * 0.7
    let q: SIMD2<Float> = simd_abs(uv) - SIMD2<Float>(a, b) + SIMD2<Float>(r, r)
    let outside: Float = simd_length(simd_max(q, SIMD2<Float>(0, 0)))
    let inside: Float = min(max(q.x, q.y), 0)
    return outside + inside - r
}

func smoothstepCPU(_ e0: Float, _ e1: Float, _ x: Float) -> Float {
    let t: Float = min(max((x - e0) / (e1 - e0), 0), 1)
    return t * t * (3 - 2 * t)
}

/// Step 20's whole crown distance, unworn, as its kernel's `crownSDF` — the
/// walls and the top height field meeting in a rounded edge.
let topSlope: Float = 1.25
let topSlopeSquaredPlusOne: Float = 1 + topSlope * topSlope
let topSlopeBound: Float = topSlopeSquaredPlusOne.squareRoot()
func crownSDFCPU(_ spec: ToothSpec, _ uv: SIMD2<Float>, _ y: Float) -> Float {
    let side: Float = crownSideCPU(spec, uv, y)
    let top: Float = (y - crownTopCPU(spec, uv)) / topSlopeBound
    let d: Float = smaxCPU(side, top, 0.45)
    return max(d, (-spec.crownHeight - rootShown) - y)
}

/// The unworn surface's height at tooth-local (u, v), found by bisection down a
/// vertical line, or nil if the line misses the crown's top.
func surfaceHeightCPU(_ spec: ToothSpec, _ uv: SIMD2<Float>) -> Float? {
    var hi: Float = 2
    var lo: Float = -0.5 * spec.crownHeight
    guard crownSDFCPU(spec, uv, lo) < 0, crownSDFCPU(spec, uv, hi) > 0 else { return nil }
    for _ in 0..<32 {
        let mid: Float = (hi + lo) / 2
        if crownSDFCPU(spec, uv, mid) > 0 { hi = mid } else { lo = mid }
    }
    return (hi + lo) / 2
}

/// Where the grinding starts, tooth-local (u, v, y): the point of the unworn
/// crown that stands furthest out along the facet's normal — the first point a
/// plane of that slope touches as it comes down. Found on a 0.025 mm grid.
///
/// Anchoring there, rather than at a typed cusp tip, means the first sliver of
/// wear is a sliver: a tilted plane put through the middle of a flat incisal
/// edge would already cut 0.2 mm off its labial rim at the first frame.
func findFacetAnchor(_ spec: ToothSpec) -> SIMD3<Float> {
    let n: SIMD3<Float> = facetNormalLocal(spec.kind)
    let a: Float = spec.width / 2
    let b: Float = spec.depth / 2
    let step: Float = 0.025
    var best: Float = -1e9
    var anchor = SIMD3<Float>(0, 0, 0)
    var u: Float = -a
    while u <= a {
        var v: Float = -b
        while v <= b {
            if let y = surfaceHeightCPU(spec, SIMD2<Float>(u, v)) {
                let s: Float = n.y * v + n.z * y
                if s > best { best = s; anchor = SIMD3<Float>(u, v, y) }
            }
            v += step
        }
        u += step
    }
    return anchor
}

/// The anchors, found once per kind of crown.
let facetAnchors: [CrownKind: SIMD3<Float>] = {
    var out: [CrownKind: SIMD3<Float>] = [:]
    for spec in mandibularTeeth { out[spec.kind] = findFacetAnchor(spec) }
    return out
}()

func facetAnchorLocal(_ spec: ToothSpec) -> SIMD3<Float> { facetAnchors[spec.kind]! }

// MARK: - the stage, and the depth it takes

/// Smith & Knight's incisal criteria, which are the ones for incisors and
/// canines: 1 "loss of enamel surface characteristics"; 2 "loss of enamel just
/// exposing dentine"; 3 "loss of enamel and substantial loss of dentine"; 4
/// "pulp exposure or exposure of secondary dentine". For occlusal surfaces: 2
/// "loss of enamel exposing dentine for less than one third of surface"; 3
/// "for more than one third"; 4 "complete enamel loss – pulp exposure –
/// secondary dentine exposure".
///
/// "Substantial" is not given a number by the index. MODEL: dentine lost to at
/// least the depth of the enamel cap that covered it — as much dentine gone as
/// enamel — so the wear is at least twice the incisal enamel deep.
let substantialDentineLoss: Float = incisalEnamel

/// How deep the grinding has gone at the end: straight down at each facet's
/// anchor, the same on every tooth, as the rate is. MODEL, chosen inside
/// incisal stage 3: past 2 × 1.03 = 2.06 mm, where the dentine loss becomes
/// "substantial", and well short of the 5.2 mm to the pulp (stage 4).
let finalWearDepth: Float = 2.25

/// How much lower the canine's highest point stands at the end: its tip is
/// ground 2.25 mm, but the facet leans 20° labially, so the facet's lingual
/// edge, now the top of the canine, stands higher than the cut tip. Measured
/// by the survey (the tests hold it to this), written here for the label.
let canineHeightLoss: Float = 1.64

/// The years that depth takes at the bruxist's rate: derived, not typed.
let finalWearYears: Float = finalWearDepth / bruxistWearRate

/// The Tooth Wear Index surface each crown kind is scored on.
enum TWISurface: String {
    case incisal, occlusal
}
func twiSurface(_ kind: CrownKind) -> TWISurface {
    kind == .incisor || kind == .canine ? .incisal : .occlusal
}

/// The score the incisal criteria give, from what the wear has done: whether
/// dentine shows, how deep the facet has gone into it (mm below where the
/// enamel cap ended), and how deep the wear is overall, straight down.
func incisalStage(worn: Bool, dentineExposed: Bool, dentineLoss: Float, depth: Float) -> Int {
    if !worn { return 0 }
    if depth >= incisalEnamelToPulp { return 4 }
    if !dentineExposed { return 1 }
    return dentineLoss >= substantialDentineLoss ? 3 : 2
}

/// The score the occlusal criteria give for a surface with `dentineFraction`
/// of its area worn to dentine.
func occlusalStage(worn: Bool, dentineFraction: Float) -> Int {
    if !worn { return 0 }
    if dentineFraction <= 0 { return 1 }
    if dentineFraction >= 0.999 { return 4 }
    return dentineFraction < 1.0 / 3.0 ? 2 : 3
}

/// The stage this still claims, per crown kind — what the tests hold the
/// drawn wear to. Incisal: 2.25 mm through 1.03 mm of enamel is 1.22 mm of
/// dentine, "substantial", so 3. Occlusal: 2.25 mm through 1.6 mm of cusp
/// enamel exposes dentine only on the cusps the facet reaches, under a third
/// of the occlusal table, so 2.
let claimedStage: [CrownKind: Int] = [
    .incisor: 3, .canine: 3,
    .firstPremolar: 2, .secondPremolar: 2, .firstMolar: 2, .secondMolar: 2,
]

// MARK: - the colours of a worn tooth

/// Exposed dentine. MODEL: no in-vivo CIELAB of attrition-exposed dentine could
/// be reached (Pecho et al., J Dent 2012, measured 0.5 mm slices of human
/// dentine, but only colour differences are in the abstract). Taken as step
/// 20's cervical colour — which is dentine seen through thin enamel (Hasegawa
/// et al. 2000) — carried the same way again: darker, redder and yellower than
/// any enamel in the scene, which is how exposed dentine reads clinically.
let dentineLab: SIMD3<Double> = SIMD3(60.0, 4.0, 28.0)

/// How rough the enamel reflection is on a facet, as the GGX alpha, against
/// step 20's 0.3 for unworn enamel. MODEL: facets are polished by the grinding —
/// "well-defined, shiny facets" (Kaidonis 2008) — so a small value.
let facetRoughness: Float = 0.08
/// The same for exposed dentine, which grinding burnishes to a smear layer
/// (Kaidonis 2008). MODEL: a little less glossy than enamel.
let dentineRoughness: Float = 0.14

// MARK: - one tooth's wear, for the GPU

/// A facet as the kernel wants it: world-space unit normal and the plane's
/// offset, f(p) = dot(n, p) − offset, the tooth cut away where f > 0. With no
/// wear the plane is parked far above everything.
func facetPlane(_ t: PlacedTooth, depth: Float) -> SIMD4<Float> {
    let nl: SIMD3<Float> = facetNormalLocal(t.spec.kind)
    let tan3 = SIMD3<Float>(t.tangent.x, 0, t.tangent.y)
    let out3 = SIMD3<Float>(t.outward.x, 0, t.outward.y)
    let up = SIMD3<Float>(0, 1, 0)
    let n: SIMD3<Float> = simd_normalize(tan3 * nl.x + out3 * nl.y + up * nl.z)
    guard depth > 0 else { return SIMD4<Float>(n, 1e4) }
    let a: SIMD3<Float> = facetAnchorLocal(t.spec)
    let centre = SIMD3<Float>(t.centre.x, 0, t.centre.y)
    let anchorWorld: SIMD3<Float> = centre + tan3 * a.x + out3 * a.y + up * (a.z - depth)
    return SIMD4<Float>(n, simd_dot(n, anchorWorld))
}

/// The enamel the kernel needs, per tooth: (top, wall at the junction, wall
/// slope per mm, most the wall reaches).
func enamelShell(_ kind: CrownKind) -> SIMD4<Float> {
    SIMD4<Float>(topEnamel(kind), wallEnamelAtJunction, wallEnamelSlope, topEnamel(kind))
}
