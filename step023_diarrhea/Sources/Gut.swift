// The gut as numbers: where the tubes are, how big the villi and crypts are
// drawn, how the peristaltic wave moves, what cholera stool is made of, and how
// much water crosses the wall each way. Nothing here touches the GPU; it is the
// part a test can read against its sources. The kernel in Render.swift is
// generated from these same constants, and Particles.swift uses the Swift
// versions of the wave and the radii below so the particles ride the wall the
// kernel draws.
//
// Millimetres throughout. x runs downstream (left to right on screen), y is up,
// and the cutaway plane is z = 0: everything with z > 0 has been cut away, and
// the camera looks in from +z. Both panels show the SAME stretch of gut; only
// what happens inside it differs.
//
// Every number carries a source or the word MODEL and a reason. "Drawn" sizes
// are deliberately enlarged: at true scale a villus would be under a pixel wide.

import Foundation
import simd

// MARK: - the loop

/// Seconds per loop. The peristaltic wave's period divides this exactly, and
/// every particle is spawned periodically in time, so the last frame flows into
/// the first — forward, never a rewind (the project's standing rule). MODEL.
let loopSeconds: Float = 9.6

// MARK: - the tubes

/// Inner radius of the terminal ileum at rest. The small bowel is normally no
/// more than 3 cm across (Radiopaedia, "3-6-9 rule (bowel)"); 21 mm across is
/// MODEL within that.
let ileumRadius: Float = 10.5
/// Ileal wall thickness. MODEL: a little over the ~3 mm upper limit of normal
/// on imaging, so the four layers can be seen in section.
let ileumWall: Float = 4.0
/// Inner radius of the colon. The colon is normally no more than 6 cm across,
/// the caecum 9 cm (Radiopaedia, "3-6-9 rule"); 44 mm across is MODEL within
/// that, and keeps the colon about twice the ileum's width, as the rule has it.
let colonRadius: Float = 22.0
/// Colonic wall thickness. MODEL, as for the ileum.
let colonWall: Float = 4.5

/// The two axes. The ileum enters the caecal end of the colon above the colon's
/// axis, so the blind caecal pouch hangs below the valve, as it does in the
/// body. MODEL geometry; anatomy only in its arrangement.
let ileumAxisY: Float = 1.5
// (5.5 mm above the colon's axis: the ileum's outer wall clears the colon's
// inner wall by 2 mm, so both lips of the valve stand free.)
let colonAxisY: Float = -4.0
/// Where the ileal papilla — the ileocecal valve — ends inside the colon. The
/// ileal wall runs on past the caecal wall for about 9 mm as two lips.
let lipX: Float = -10.0

/// The colon is shortened: it stops at breakStartX and resumes, as the rectum,
/// at breakEndX. A break mark is drawn over the gap.
let breakStartX: Float = 90.0
let breakEndX: Float = 106.0

/// What the frame shows: the ileum enters at the left edge, the rectum leaves
/// at the right. The particles that wrap around the loop wrap over this span.
let gutStartX: Float = -112.0
let gutEndX: Float = 148.0
var gutSpan: Float { gutEndX - gutStartX }

// MARK: - the mucosa, drawn enlarged

/// Villus height as drawn. Real villi are about 0.5–1.6 mm long (Wikipedia,
/// "Intestinal villus"; ScienceDirect Topics, "Intestine villus"), so these are
/// drawn roughly 2–7× enlarged. Everything in the mucosa is enlarged together.
let villusHeight: Float = 3.6
/// Villus radius as drawn. MODEL.
let villusRadius: Float = 0.55
/// Villus spacing along the gut, and how many ring the gut. MODEL; the count
/// around must be even so a villus sits exactly in the cut plane top and bottom.
let villusPitch: Float = 2.6
let villusRing: Int = 22
/// The last villus before the valve. Past this there are none: the lips of the
/// valve are where small-intestinal mucosa gives way to colonic.
let lastVillusX: Float = lipX - 2.4

/// Ileal crypt depth: a third of the villus. Normal adult small bowel has a
/// villus-to-crypt ratio of about 3:1 ("A practical approach to small bowel
/// biopsy interpretation: celiac disease and its mimics", Semin Diagn Pathol
/// 31(2), 2014: "a villus to crypt ratio of 3:1 is felt to be normal").
let ilealCryptDepth: Float = villusHeight / 3.0
/// Crypt radius as drawn. MODEL.
let cryptRadius: Float = 0.42

/// Colonic crypts: the colon's lining is flat, with crypts and no villi.
/// Depth and spacing are MODEL, enlarged like everything else in the mucosa.
let colonCryptDepth: Float = 1.6
let colonCryptPitch: Float = 1.8
let colonCryptRing: Int = 70
/// Colonic crypts are drawn from here on (the caecal tip is left smooth, where
/// a cylinder's lattice cannot follow a sphere). MODEL.
let colonCryptStartX: Float = -10.0

/// How thick the mucosa and submucosa are in section; muscle takes the rest of
/// the wall and thickens as it contracts. MODEL, in the textbook proportions.
let ilealMucosa: Float = 1.7
let colonMucosa: Float = 2.0
let submucosa: Float = 0.8

// MARK: - the particles, as oversized symbols

/// Shannon, Acta Cryst A 32:751 (1976), six-coordinate ionic radii in pm.
let shannonRadiusPm: [Species: Float] = [.sodium: 102, .potassium: 138, .chloride: 181]
/// Water, as the 1.4 Å probe of Lee & Richards, J Mol Biol 55:379 (1971).
let waterRadiusPm: Float = 140
/// Bicarbonate has no Shannon radius — it is not a single atom. MODEL: drawn at
/// 160 pm, between K⁺ and Cl⁻, and it is not part of any size claim.
let bicarbonateRadiusPm: Float = 160

/// Drawn radius in mm per pm of real radius in the main panels: a chloride ion
/// is drawn with a 1.0 mm radius — about 5.5 million times too big. A water
/// molecule is ~0.28 nm across and a villus ~0.5 mm tall: a millionth.
let panelMMPerPm: Float = 1.0 / 181
/// In the close-up of the crypt wall, µm per pm.
let insetUMPerPm: Float = 0.95 / 181

enum Species: Int, CaseIterable {
    case chloride = 0, sodium, potassium, bicarbonate, water, fleck
}

func radiusPm(_ s: Species) -> Float {
    switch s {
    case .chloride, .sodium, .potassium: return shannonRadiusPm[s]!
    case .water: return waterRadiusPm
    case .bicarbonate: return bicarbonateRadiusPm
    case .fleck: return 150          // a fleck of mucus is not a molecule; MODEL size
    }
}

/// Display colours, sRGB. CPK-style: chlorine green, sodium violet. CPK makes
/// potassium (8F40D4) nearly the same purple as sodium (AB5CF2), which would
/// make the sparse K⁺ unreadable next to the plentiful Na⁺, so K⁺ is AMBER
/// here — a deliberate break from CPK. Bicarbonate is oxygen-red, since it is
/// mostly oxygen. Water is a translucent blue. Mucus flecks are off-white:
/// cholera stool is "rice-water" — clear, with flecks of mucus (WHO).
func speciesColour(_ s: Species) -> SIMD4<Float> {
    switch s {
    case .chloride: return SIMD4(0.16, 0.82, 0.24, 1.0)
    case .sodium: return SIMD4(0.67, 0.36, 0.95, 1.0)
    case .potassium: return SIMD4(0.97, 0.66, 0.10, 1.0)
    case .bicarbonate: return SIMD4(0.92, 0.26, 0.22, 1.0)
    case .water: return SIMD4(0.48, 0.74, 1.0, 0.62)
    case .fleck: return SIMD4(0.93, 0.92, 0.86, 0.55)
    }
}

// MARK: - what cholera stool is made of

/// Adult cholera stool, mmol/L (WHO cholera treatment guidance; Merck Manual
/// Professional, "Cholera"): Na⁺ 130–135, Cl⁻ 100, K⁺ 15–20, HCO₃⁻ ~44–45,
/// osmolality ~300–308. The lumen of the lower panel holds exactly this many of
/// each, one particle per mmol/L. Bicarbonate IS drawn, as a fourth colour: it
/// makes the charges balance (150 cations against 145 anions) and its loss is
/// the acidosis of cholera.
struct StoolElectrolytes {
    var sodium: Int
    var chloride: Int
    var potassium: Int
    var bicarbonate: Int
}
let choleraStool = StoolElectrolytes(sodium: 135, chloride: 100, potassium: 15, bicarbonate: 45)

/// Stool osmotic gap, mOsm/kg: 290 − 2(Na⁺ + K⁺). Under 50 means secretory
/// diarrhoea; over 100, osmotic.
func osmoticGap(sodium: Float, potassium: Float) -> Float {
    let cations: Float = sodium + potassium
    return 290 - 2 * cations
}
let secretoryGapLimit: Float = 50

// MARK: - how much water crosses the wall

/// Fluid budget, litres per day.
///
/// Normally ~2 L/day crosses the ileocecal valve (Ghishan & Kiela, Curr Opin
/// Gastroenterol 28:130, 2012: "the typical ileocecal daily flow of 2 l"), and
/// the colon absorbs all but ~0.1–0.2 L of it; 1.35 L/day is used here. At
/// most the colon absorbs ~5 L/day, up to ~5.7 L measured with caecal infusion
/// (Debongnie & Phillips, Gastroenterology 74:698, 1978). Severe cholera can
/// pour out more than 1 L an HOUR — 24 L/day — which is what arrives at the
/// colon in the lower panel. Jejunum and ileum both secrete in cholera
/// (Banwell et al., J Clin Invest 49:183, 1970); the ileum drawn here stands
/// for the whole secreting small intestine.
let normalColonAbsorbsLitres: Float = 1.35
let maximalColonAbsorbsLitres: Float = 5.0
let choleraInflowLitres: Float = 24.0

/// Water symbols per crypt burst in the lower panel. Every ileal crypt in the
/// cut face bursts once a loop: Cl⁻, then Na⁺, then this much water.
let choleraWaterPerBurst: Int = 2

/// The number of ileal crypts in the cut face (top and bottom walls).
func ilealCryptXs() -> [Float] {
    var xs: [Float] = []
    var x: Float = lastVillusX - villusPitch * 0.5
    while x > gutStartX + 1 {
        xs.append(x)
        x -= villusPitch
    }
    return xs
}

func villusXs() -> [Float] {
    var xs: [Float] = []
    var x: Float = lastVillusX
    while x > gutStartX + 1 {
        xs.append(x)
        x -= villusPitch
    }
    return xs
}

/// Absorbing surface between two colonic crypts, in the cut face.
func colonSurfaceXs() -> [Float] {
    var xs: [Float] = []
    var k: Float = 2
    while true {
        let x: Float = (k + 0.5) * colonCryptPitch
        if x > gutEndX - 2 { break }
        if x < breakStartX - 1.5 || x > breakEndX + 1.5 { xs.append(x) }
        k += 1
    }
    return xs
}

/// Water particles per loop per litre-per-day, fixed by the cholera inflow:
/// every secreted water symbol in the lower panel is 24/N of a litre a day.
var waterParticlesPerLitre: Float {
    let secreted: Float = Float(ilealCryptXs().count * 2 * choleraWaterPerBurst)
    return secreted / choleraInflowLitres
}

// MARK: - the peristaltic wave

/// One panel's wave. A ring of contraction travels downstream with a relaxed,
/// widened segment just AHEAD of it — Bayliss & Starling's "law of the
/// intestine" (J Physiol 24:99, 1899): excitation above, inhibition below — so
/// the wave pushes what is in front of it instead of pinching it.
///
/// Speeds are compressed into a 9.6 s loop. Real small-intestinal waves
/// propagate at millimetres to a couple of centimetres per second over short
/// distances (ScienceDirect Topics, "Intestine motility"); the shown speed is
/// MODEL, and is faster in the lower panel.
struct Wave {
    var wavelength: Float          // mm; divides gutSpan exactly
    var wavesPerLoop: Int          // how many wavelengths pass a point per loop
    var contraction: Float         // fractional narrowing at the ring
    var relaxation: Float          // fractional widening ahead of it
    var contentLoops: Int          // diarrhoea: lumen contents cross the span this many times per loop
    var contentMM: Float           // normal: how far the contents creep per loop
    var surge: Float               // mm each passing wave shoves the contents forward
    var speed: Float { wavelength * Float(wavesPerLoop) / loopSeconds }
}

/// Upper panel: one slow wave; the contents creep 18 mm a loop — one stool
/// lump's spacing — plus a 5 mm shove per wave.
let normalWave = Wave(wavelength: 260, wavesPerLoop: 1, contraction: 0.34, relaxation: 0.13,
                      contentLoops: 0, contentMM: 18, surge: 5)
/// Lower panel: two waves on screen, travelling faster than the watery
/// contents, which cross the whole span once a loop.
let choleraWave = Wave(wavelength: 130, wavesPerLoop: 3, contraction: 0.40, relaxation: 0.16,
                       contentLoops: 1, contentMM: 0, surge: 18)

func wave(_ panel: Int) -> Wave { panel == 0 ? normalWave : choleraWave }

/// Shape of the ring, in mm from its centre (positive = downstream, ahead).
let contractionWidth: Float = 10     // MODEL
let relaxationLead: Float = 17       // MODEL: how far ahead the relaxed segment sits
let relaxationWidth: Float = 12      // MODEL

/// Where x sits in the wave at time t: signed mm from the nearest ring centre,
/// in [−λ/2, λ/2). Positive means the ring has not reached x yet.
func waveOffset(_ x: Float, panel: Int, t: Float) -> Float {
    let w: Wave = wave(panel)
    let phase: Float = (x - w.speed * t) / w.wavelength
    let wrapped: Float = phase - (phase + 0.5).rounded(.down)
    return wrapped * w.wavelength
}

/// Fractional change of radius at a wave offset: negative in the ring, positive
/// in the relaxed segment ahead of it.
func waveShape(_ d: Float, panel: Int) -> (total: Float, contraction: Float) {
    let w: Wave = wave(panel)
    let a: Float = d / contractionWidth
    let c: Float = w.contraction * exp(-a * a)
    let b: Float = (d - relaxationLead) / relaxationWidth
    let r: Float = w.relaxation * exp(-b * b)
    return (r - c, c)
}

func smoothstep(_ e0: Float, _ e1: Float, _ x: Float) -> Float {
    let t: Float = min(max((x - e0) / (e1 - e0), 0), 1)
    return t * t * (3 - 2 * t)
}

/// The wave fades out approaching the valve from both sides: the ileocecal
/// junction keeps its own tone. MODEL.
func ileumEnvelope(_ x: Float) -> Float { 1 - smoothstep(-34, -24, x) }
func colonEnvelope(_ x: Float) -> Float { smoothstep(4, 18, x) }

func ileumRadiusAt(_ x: Float, panel: Int, t: Float) -> Float {
    let g = waveShape(waveOffset(x, panel: panel, t: t), panel: panel)
    return ileumRadius * (1 + ileumEnvelope(x) * g.total)
}

func colonRadiusAt(_ x: Float, panel: Int, t: Float) -> Float {
    if x < 0 { return sqrt(max(colonRadius * colonRadius - x * x, 1)) }
    let g = waveShape(waveOffset(x, panel: panel, t: t), panel: panel)
    return colonRadius * (1 + colonEnvelope(x) * g.total)
}

/// The free lumen a particle may occupy: its axis height and radius, blending
/// from the ileum through the valve's jet into the colon.
func lumenFrame(_ x: Float, panel: Int, t: Float) -> (axisY: Float, radius: Float, fill: Float) {
    let s: Float = smoothstep(lipX, lipX + 16, x)
    let rI: Float = ileumRadiusAt(min(x, lipX), panel: panel, t: t)
    let rC: Float = colonRadiusAt(max(x, 0), panel: panel, t: t)
    let axis: Float = ileumAxisY + (colonAxisY - ileumAxisY) * s
    let radius: Float = rI + (rC - rI) * s
    // Keep clear of the villi in the ileum; the colon's lining is flat.
    let fill: Float = 0.42 + (0.82 - 0.42) * s
    return (axis, radius, fill)
}

/// How far the contents are shoved at a wave offset: 1 just behind the ring
/// (already pushed), falling to 0 over the half-wavelength behind it, and
/// rising again sharply in the 25 mm ahead of the ring as it arrives. Because
/// the ring outruns the contents, a particle's offset only ever DEcreases, so
/// this is a shove forward followed by a slow give-back that never reverses
/// the particle: the tests check every lumen particle moves forward every frame.
func surgeFraction(_ d: Float, wavelength: Float) -> Float {
    if d >= 0 { return 1 - smoothstep(0, 25, d) }
    return 1 + d / (wavelength * 0.5)
}

// MARK: - the flow, by continuity

/// Where the contents are, as the flow carries them. The same volume passes
/// every cross-section each second, so the contents race through the narrow
/// ileum and the jet of the valve and slow in the wide colon. Measured along
/// the gut by a coordinate s that advances uniformly in time, x(s) spends time
/// in proportion to the lumen's width — which also keeps the particles evenly
/// spread per unit of screen area, rather than crowding the ileum.
/// (Continuity weights by area; weighting by width, as the cut face shows it,
/// is MODEL — area would make the ileum's particles five times sparser.)
let flowTable: [Float] = {
    var cumulative: [Float] = [0]
    var x: Float = gutStartX
    while x < gutEndX {
        let s: Float = smoothstep(lipX, lipX + 16, x + 0.5)
        let width: Float = ileumRadius + (colonRadius - ileumRadius) * s
        cumulative.append(cumulative.last! + width)
        x += 1
    }
    let total: Float = cumulative.last!
    return cumulative.map { $0 / total }
}()

/// x at flow coordinate s in [0, 1).
func flowX(_ s: Float) -> Float {
    var lo = 0, hi = flowTable.count - 1
    while hi - lo > 1 {
        let mid = (lo + hi) / 2
        if flowTable[mid] <= s { lo = mid } else { hi = mid }
    }
    let f: Float = (s - flowTable[lo]) / max(flowTable[hi] - flowTable[lo], 1e-9)
    return gutStartX + Float(lo) + f
}

/// The time of frame f of n: evenly spaced, strictly increasing, and frame n —
/// which is never drawn — would be exactly one loop, i.e. frame 0 again. No
/// frame is ever played backwards.
func frameTime(_ f: Int, of n: Int) -> Float { loopSeconds * Float(f) / Float(n) }
