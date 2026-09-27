// The gut as numbers: where the tubes are, how big the villi and crypts are
// drawn, how the peristaltic wave moves, what the stool is made of in each
// panel, and how much water and sugar crosses the wall each way. Nothing here
// touches the GPU; it is the part a test can read against its sources. The
// kernel in Render.swift is generated from these same constants, and
// Particles.swift uses the Swift versions of the wave and the radii below so
// the particles ride the wall the kernel draws.
//
// The anatomy — tubes, villi, crypts, valve, wave, flow — is step 23's,
// copied rather than shared (see the Makefile for why) and unchanged: the same
// stretch of gut, now after two glasses of milk. Only the chemistry differs.
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
/// Glucose, as its effective hydrodynamic radius, 3.8 Å (Pappenheimer's
/// viscometric value, as quoted alongside Schultz & Solomon, J Gen Physiol
/// 44:1189, 1961 — read second-hand, not from the paper's table). Galactose
/// has the same formula and nearly the same shape, so it is drawn the same
/// size (MODEL).
let glucoseRadiusPm: Float = 380
/// A short-chain fatty acid. MODEL: 240 pm, between water and glucose —
/// acetate (the commonest) is a two-carbon acid, butyrate a four-carbon one;
/// no single radius stands for all three.
let scfaRadiusPm: Float = 240

/// Drawn radius in mm per pm of real radius in the main panels: a chloride ion
/// would be drawn with a 1.0 mm radius — about 5.5 million times too big. A
/// water molecule is ~0.28 nm across and a villus ~0.5 mm tall: a millionth.
/// (Step 23's scale, kept so the two renders' symbols compare.)
let panelMMPerPm: Float = 1.0 / 181
/// In the close-up of the brush border, µm per pm.
let insetUMPerPm: Float = 0.95 / 181

/// What the particles are. Lactose is not a species of its own: it is drawn as
/// what it is, a galactose bonded to a glucose — the pair of spheres touching —
/// so that splitting it is visibly the pair coming apart.
///
/// Chloride is here only so the mutation check can switch chloride secretion
/// on and a test can see it; in the real scene there is none.
enum Species: Int, CaseIterable {
    case chloride = 0, sodium, potassium, water, glucose, galactose, scfa, gas, bacterium
}

/// Gas bubbles and bacteria are not molecules, and are drawn at their own
/// symbol sizes (MODEL): a bacterium is ~1–2 µm, some 10,000 times a water
/// molecule, and is drawn only a few times larger. Not to scale, and the key
/// says so.
let bacteriumSphereMM: Float = 0.62        // radius of each of the rod's three beads
let bacteriumBeadSpacing: Float = 0.62     // centre to centre: a rod about 2.5 mm long
let bubbleMinMM: Float = 0.55              // a new bubble
let bubbleMaxMM: Float = 1.7               // one that has grown and merged on its way down

func radiusPm(_ s: Species) -> Float {
    switch s {
    case .chloride, .sodium, .potassium: return shannonRadiusPm[s]!
    case .water: return waterRadiusPm
    case .glucose, .galactose: return glucoseRadiusPm
    case .scfa: return scfaRadiusPm
    case .gas: return bubbleMinMM / panelMMPerPm
    case .bacterium: return bacteriumSphereMM / panelMMPerPm
    }
}

/// Display colours, sRGB. Glucose yellow and galactose light green are step
/// 17's, so the two sugars look the same wherever they appear in the project.
/// Na⁺ violet, K⁺ amber, Cl⁻ green and water blue are step 23's. SCFAs are
/// coral: acids, and unlike anything else on screen. Gas is a pale, clear
/// bubble. Bacteria are teal rods (colour MODEL; they are not coloured).
func speciesColour(_ s: Species) -> SIMD4<Float> {
    switch s {
    case .chloride: return SIMD4(0.16, 0.82, 0.24, 1.0)
    case .sodium: return SIMD4(0.67, 0.36, 0.95, 1.0)
    case .potassium: return SIMD4(0.97, 0.66, 0.10, 1.0)
    case .water: return SIMD4(0.48, 0.74, 1.0, 0.62)
    case .glucose: return SIMD4(0.96, 0.86, 0.42, 1.0)
    case .galactose: return SIMD4(0.62, 0.90, 0.52, 1.0)
    case .scfa: return SIMD4(0.96, 0.40, 0.36, 1.0)
    case .gas: return SIMD4(0.90, 0.95, 1.0, 0.34)
    case .bacterium: return SIMD4(0.14, 0.66, 0.60, 1.0)
    }
}

// MARK: - the dose

/// Two glasses of milk. One 240 mL cup holds about 12 g of lactose. The NIH
/// Consensus Development Conference Statement, "Lactose Intolerance and
/// Health" (Suchy et al., 2010), found that most people with lactose
/// intolerance tolerate up to ~12 g; symptoms become more prominent above
/// 12 g and appreciable after 24 g, and 50 g induces them in the vast majority.
/// Two glasses is where the lower panel's story starts to be typical.
let lactoseGramsPerCup: Float = 12
let cups: Int = 2

// MARK: - where lactase is

/// Human lactase is lactase-phlorizin hydrolase (LPH, gene LCT): an ectoenzyme
/// anchored in the microvillar membrane of the villus brush border by a
/// C-terminal hydrophobic stretch (Norén & Sjöström, Scand J Nutr 45:156,
/// 2001). It is NOT the E. coli β-galactosidase (LacZ) that step 17 rendered —
/// that is a different protein that makes the same cut, and step 17 found no
/// structure of LPH deposited at all. So LPH is drawn here as a marker on the
/// villus tips, not as a molecule. In lactase non-persistence its mRNA is
/// turned down after weaning (same source); the lower panel draws it absent.
func lactasePresent(panel: Int, mutant: Mutant = .none) -> Bool {
    panel == 0 || mutant == .lactaseWorking
}

/// Chloride secretion — the engine of step 23 — is off in both panels. In
/// osmotic diarrhoea the water is pulled in by what is in the lumen, not
/// pushed out by pumped ions. The mutant switches it on.
func chlorideSecretion(panel: Int, mutant: Mutant = .none) -> Bool {
    panel == 1 && mutant == .chlorideOn
}

// MARK: - what the stool is made of

/// Stool osmotic gap, mOsm/kg: 290 − 2(Na⁺ + K⁺). Over 125 means osmotic
/// diarrhoea, under 50 secretory (Fine & Schiller, AGA technical review on the
/// evaluation and management of chronic diarrhea, Gastroenterology 116:1464,
/// 1999). Stool water is taken as isotonic with plasma, 290 mOsm/kg, as the
/// rule does: measured stool osmolality climbs after passage as bacteria go on
/// fermenting in the jar (Hammer et al., below: 306 → 346 in 72 h, cold).
func osmoticGap(sodium: Float, potassium: Float) -> Float {
    let cations: Float = sodium + potassium
    return 290 - 2 * cations
}
let osmoticGapLimit: Float = 125
let secretoryGapLimit: Float = 50
let stoolOsmolality: Float = 290

/// The lower panel's stool, one particle for every 3 mOsm/kg (97 particles,
/// 291 mOsm/kg).
///
/// Na⁺ 33 and K⁺ 30 mmol/L are measured — but in LACTULOSE diarrhoea, not
/// lactose: Hammer, Santa Ana, Schiller & Fordtran, J Clin Invest 84:1056
/// (1989), Table II, 95 g/day of lactulose (33 ± 7 and 30 ± 6 meq/L). Lactulose
/// is the standard experimental stand-in for a sugar the small intestine cannot
/// absorb and colonic bacteria ferment; no comparable table for lactose was
/// found, so using it for lactose is MODEL. 11 and 10 particles are 33 and 30.
///
/// The rest of the osmoles — the gap — are organic acids and unfermented
/// sugar. How they split is MODEL, in the right order: at low lactulose doses
/// the colon's bacteria ferment almost all of it (45 g/day: 1 g left in the
/// stool) and "diarrhea ... was mainly due to unabsorbed organic acids and
/// associated cations" (same paper). So 73 SCFA (219 mOsm) and 3 lactose
/// (9 mOsm). Ca²⁺, Mg²⁺, Cl⁻ and HCO₃⁻ are all small in that stool (Table II)
/// and are not drawn.
///
/// A check the numbers pass without being forced: about 30% of organic acids
/// are ionized at pH 4.4 (same paper), and 0.3 × 219 = 66 meq/L of organic
/// anions is within 5% of the 63 of Na⁺ + K⁺ they hold in the lumen.
struct StoolMix {
    var sodium: Int
    var potassium: Int
    var scfa: Int
    var lactose: Int
    var total: Int { sodium + potassium + scfa + lactose }
    /// mmol/L implied by a count, if the whole mix is 290 mOsm/kg.
    func concentration(_ n: Int) -> Float { Float(n) * stoolOsmolality / Float(total) }
    var gap: Float { osmoticGap(sodium: concentration(sodium), potassium: concentration(potassium)) }
}
let mOsmPerParticle: Float = 3
let lowerStool = StoolMix(sodium: 11, potassium: 10, scfa: 73, lactose: 3)
/// The mutant: a secretory-looking stool (step 23's cholera Na⁺ and K⁺, scaled
/// the same way) — its gap comes out far under 125.
let secretoryLookingStool = StoolMix(sodium: 45, potassium: 5, scfa: 44, lactose: 3)
func stool(mutant: Mutant) -> StoolMix { mutant == .lowGap ? secretoryLookingStool : lowerStool }

/// Normal stool water, for the upper panel: Na⁺ about 30 and K⁺ about 75
/// mmol/L, osmolality close to plasma's (ARUP Laboratories, "Electrolyte and
/// Osmolality Profile, Fecal", test directory 0020699). Its gap, 80, is
/// between the two cut-offs: the gap is a test for diarrhoeal stool, and in
/// formed stool the unmeasured osmoles are mostly SCFAs from the fibre the
/// colon ferments every day. Not drawn as particles: the stool is solid here.
let normalStoolSodium: Float = 30
let normalStoolPotassium: Float = 75

/// Fecal pH in the lower panel: 4.5 ± 0.1 at 95 g/day of lactulose (Hammer et
/// al., Table II). When diarrhoea is caused by carbohydrate malabsorption,
/// fecal fluid pH is always under 5.6 and usually under 5.3; other causes
/// rarely reach 5.6 and never go under 5.3 (Eherer & Fordtran, Gastroenterology
/// 103:545, 1992 — thresholds from the abstract as quoted in secondary
/// sources; the paper itself was not read). Normal colonic contents run from
/// pH 5.6 in the caecum to 6.6 in the descending colon (Cummings et al., Gut
/// 28:1221, 1987).
let lowerStoolPH: Float = 4.5
let carbohydrateMalabsorptionPH: Float = 5.6

// MARK: - fermentation, and taking some of it back

/// One mole of disaccharide yields about 3.7 moles of organic acids (Hammer et
/// al. 1989, citing their ref. 25). Drawn exactly: in every ten lactose
/// symbols fermented, seven give four SCFA and three give three — 37 per 10.
func scfaYield(parcel j: Int) -> Int { (j % 10) < 7 ? 4 : 3 }

/// Gas bubbles per lactose fermented: hydrogen and carbon dioxide. MODEL count,
/// one of each. Methane is made too in about a third of adults (34% of healthy
/// adults 35 years apart, 36% now: Levitt et al., Clin Gastroenterol Hepatol
/// 4:123, 2006) — by archaea, which eat the hydrogen — and is not drawn.
/// About 14% of the hydrogen is absorbed into the blood and breathed out (Levitt,
/// N Engl J Med 281:122, 1969): the hydrogen breath test.
let bubblesPerLactose: Int = 2

/// Lactose symbols that reach the colon in the lower panel, and how many the
/// bacteria ferment. 40 fermented make 148 SCFA; 73 are left in the stool, so
/// 75 (51%) are absorbed back — "colonic salvage".
///
/// Understated on purpose: at 125 g/day of lactulose the colon absorbed about
/// 745 of the ~848 meq of organic acids made, ~88% (Hammer et al.). Drawn at
/// the measured fraction, the survivors that make the gap would be too few to
/// see. MODEL.
let lactoseParcels: Int = 43
var fermentedParcels: Int { lactoseParcels - lowerStool.lactose }
func scfaProduced(mutant: Mutant = .none) -> Int {
    (0..<(lactoseParcels - stool(mutant: mutant).lactose)).reduce(0) { $0 + scfaYield(parcel: $1) }
}
func scfaAbsorbed(mutant: Mutant = .none) -> Int { scfaProduced(mutant: mutant) - stool(mutant: mutant).scfa }

// MARK: - how much water crosses the wall

/// Water symbols per loop, by route. MODEL counts, in the right order.
///
/// Upper panel: the villi absorb far more than anything adds; the colon takes
/// back most of what is left and the stool forms.
///
/// Lower panel: the lactose stays in the lumen and holds water there, and the
/// small intestine, which is leaky, lets water in until the contents are
/// isotonic — osmosis, and nothing else pushing. The villi still absorb what
/// they can of everything else. In the colon, water goes back with the Na⁺
/// and the salvaged SCFAs, but not enough to firm the stool.
let upperVillusWater: Int = 40
let upperVillusSodium: Int = 12
let upperColonWater: Int = 16
let lowerVillusWater: Int = 12
let lowerVillusSodium: Int = 6
let lowerOsmoticWater: Int = 44
let lowerColonWater: Int = 22

/// Lactose split at the villus tips of the upper panel per loop. MODEL count.
let upperSplitEvents: Int = 30

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


// MARK: - the peristaltic wave

/// One panel's wave. A ring of contraction travels downstream with a relaxed,
/// widened segment just AHEAD of it — Bayliss & Starling's "law of the
/// intestine" (J Physiol 24:99, 1899): excitation above, inhibition below — so
/// the wave pushes what is in front of it instead of pinching it.
///
/// Speeds are compressed into a 9.6 s loop. Real small-intestinal waves
/// propagate at millimetres to a couple of centimetres per second over short
/// distances (ScienceDirect Topics, "Intestine motility"); the shown speed is
/// MODEL, and is faster in the lower panel (step 23's cholera wave, kept: the contents race through in both kinds of diarrhoea).
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
/// Lower panel: two waves on screen, travelling faster than the loose
/// contents, which cross the whole span once a loop. Step 23's cholera wave,
/// unchanged. MODEL: osmotic diarrhoea after two glasses of milk is usually
/// milder than cholera; the faster transit is what matters here.
let lowerWave = Wave(wavelength: 130, wavesPerLoop: 3, contraction: 0.40, relaxation: 0.16,
                       contentLoops: 1, contentMM: 0, surge: 18)

func wave(_ panel: Int) -> Wave { panel == 0 ? normalWave : lowerWave }

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
