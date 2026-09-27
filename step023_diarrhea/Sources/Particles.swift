// Every particle on screen, as a pure function of time. Nothing is simulated
// and nothing carries state from frame to frame: a particle's position is
// computed from its emitter, its phase and the clock, and every emitter repeats
// with a period that divides the loop. So frame N is frame 0 again, exactly,
// and nothing ever runs backwards.
//
// Routes, which the tests read:
//   lumen          the contents, carried downstream by the flow and the wave
//   crypt          secretion: up out of the tissue through a crypt into the lumen
//   villusAbsorb   absorption: from the lumen into a villus tip and down its core
//   colonAbsorb    absorption: from the lumen into the colon's surface and away
//   transcellular  close-up: basolateral membrane → cytoplasm → CFTR → lumen
//   paracellular   close-up: up the space between cells, through the tight junction

import Foundation
import simd

enum Route: Int {
    case lumen, crypt, villusAbsorb, colonAbsorb, transcellular, paracellular
}

/// What to break, for the mutation check. Each one must be caught by a test.
enum Mutant: String {
    case none
    case tips          // secretion from villus tips instead of crypts
    case naThroughCells = "na-through-cells"   // Na⁺ takes Cl⁻'s transcellular route
    case lazyColon = "lazy-colon"              // the cholera colon absorbs nothing
}

struct Particle {
    var position: SIMD3<Float>
    var radius: Float
    var species: Species
    var alpha: Float
    var panel: Int         // 0 normal, 1 cholera, 2 close-up
    var route: Route
}

/// A deterministic hash in [0, 1).
func hash01(_ a: Int, _ b: Int = 0, _ c: Int = 0) -> Float {
    var h: UInt64 = UInt64(bitPattern: Int64(a &* 73856093 ^ b &* 19349663 ^ c &* 83492791))
    h ^= h >> 33
    h = h &* 0xff51afd7ed558ccd
    h ^= h >> 33
    h = h &* 0xc4ceb9fe1a85ec53
    h ^= h >> 33
    return Float(h >> 40) / Float(1 << 24)
}

func fract(_ x: Float) -> Float { x - x.rounded(.down) }

// MARK: - emitters

/// One source of particles: where, what, and when in the loop.
struct Emitter {
    var route: Route
    var species: Species
    var panel: Int
    var x: Float            // site along the gut (or across the close-up)
    var side: Float         // +1 top wall, −1 bottom wall
    var phase: Float        // when in the loop it fires, [0, 1)
    var life: Float         // lifetime as a fraction of the loop
    var perLoop: Int        // how many times it fires per loop
}

/// Particles per burst from one crypt, with their delay (fraction of a loop).
/// Chloride goes first, sodium follows the negative lumen it leaves behind, and
/// water follows both. In cholera each burst carries two water symbols — at
/// true proportions it would be hundreds per ion (MODEL count, right order).
func burstOrder(panel: Int) -> [(Species, Float)] {
    var b: [(Species, Float)] = [(.chloride, 0.0), (.sodium, 0.05), (.water, 0.10)]
    if panel == 1 { b.append((.water, 0.14)) }
    return b
}

/// How long a secreted particle is followed, as a fraction of the loop: up
/// through the crypt (the first `cryptCrossing` of its life) and a little way
/// into the lumen, where it joins the contents.
let cryptLife: Float = 0.40
let cryptCrossing: Float = 0.45
/// Absorbed particles: from the lumen into the wall.
let absorbLife: Float = 0.48

/// The part of each life during which a water symbol carries an arrow — while
/// it crosses the wall. Every route's window lasts the same 0.24 of a loop, so
/// the number of arrows on screen at any moment is proportional to the flux
/// and the two panels' arrows can be compared by eye. A test holds this.
func arrowWindow(_ r: Route) -> (from: Float, to: Float, life: Float) {
    switch r {
    case .crypt: return (0.05, 0.65, cryptLife)
    default: return (0.25, 0.75, absorbLife)
    }
}

func emitters(panel: Int, mutant: Mutant = .none) -> [Emitter] {
    var out: [Emitter] = []
    let crypts: [Float] = ilealCryptXs()
    let villi: [Float] = villusXs()
    let colon: [Float] = colonSurfaceXs()
    let perL: Float = waterParticlesPerLitre

    // Secretion from the crypts. In cholera every crypt in the cut face
    // fires; normally a quarter of them fire once a loop (basal secretion).
    for (i, x) in crypts.enumerated() {
        for side: Float in [1, -1] {
            let s: Int = side > 0 ? 0 : 1
            let fires: Bool = panel == 1 || (i + s * 2) % 4 == 0
            guard fires else { continue }
            let bursts: Int = 1
            let phase: Float = hash01(i, s, 11 + panel)
            // The tips mutant moves the source onto the nearest villus.
            let site: Float = mutant == .tips && panel == 1 ? x + villusPitch * 0.5 : x
            for (sp, delay) in burstOrder(panel: panel) {
                out.append(Emitter(route: .crypt, species: sp, panel: panel, x: site, side: side,
                                   phase: fract(phase + delay), life: cryptLife, perLoop: bursts))
            }
        }
    }

    // Absorption at villus tips. Normally the villi absorb far more than the
    // crypts secrete. In cholera they still absorb — the toxin does not touch
    // sodium–glucose absorption at the tips, which is why oral rehydration
    // works (Hirschhorn et al., N Engl J Med 279:176, 1968) — but far less
    // than the crypts pour out. Counts are MODEL, in the right order.
    let villusWater: Int = panel == 0 ? 40 : 12
    let villusSodium: Int = panel == 0 ? 12 : 6
    let tipSites: [(Float, Float)] = villi.flatMap { [($0, Float(1)), ($0, Float(-1))] }
    for k in 0..<(villusWater + villusSodium) {
        let pick: Int = (k * 37 + panel * 11) % tipSites.count
        let (x, side) = tipSites[pick]
        out.append(Emitter(route: .villusAbsorb, species: k < villusWater ? .water : .sodium,
                           panel: panel, x: x, side: side, phase: hash01(k, panel, 23),
                           life: absorbLife, perLoop: 1))
    }

    // Colonic absorption: sodium through the surface cells, water after it.
    // Normally 1.35 L/day; in cholera the colon runs at its ~5 L/day maximum.
    // It is overwhelmed, not idle.
    var litres: Float = panel == 0 ? normalColonAbsorbsLitres : maximalColonAbsorbsLitres
    if mutant == .lazyColon && panel == 1 { litres = 0 }
    let colonWater: Int = Int((litres * perL).rounded())
    let colonSodium: Int = colonWater / 3
    let surfaceSites: [(Float, Float)] = colon.flatMap { [($0, Float(1)), ($0, Float(-1))] }
    for k in 0..<(colonWater + colonSodium) {
        let pick: Int = (k * 53 + panel * 7) % surfaceSites.count
        let (x, side) = surfaceSites[pick]
        out.append(Emitter(route: .colonAbsorb, species: k < colonWater ? .water : .sodium,
                           panel: panel, x: x, side: side, phase: hash01(k, panel, 31),
                           life: absorbLife, perLoop: 1))
    }
    return out
}

/// Water symbols per loop into the lumen, out of it, and out through the colon.
func waterBudget(panel: Int, mutant: Mutant = .none) -> (into: Int, out: Int, colonOut: Int) {
    var into = 0, out = 0, colon = 0
    for e in emitters(panel: panel, mutant: mutant) where e.species == .water {
        switch e.route {
        case .crypt: into += e.perLoop
        case .villusAbsorb: out += e.perLoop
        case .colonAbsorb: out += e.perLoop; colon += e.perLoop
        default: break
        }
    }
    return (into, out, colon)
}

// MARK: - paths in the main panels

func mix(_ a: Float, _ b: Float, _ t: Float) -> Float { a + (b - a) * t }

/// The lumen's downstream drift, in mm, after `seconds`, at x.
func drift(panel: Int, seconds: Float) -> Float {
    let w: Wave = wave(panel)
    let v: Float = w.contentLoops > 0 ? gutSpan * Float(w.contentLoops) / loopSeconds
                                       : w.contentMM / loopSeconds
    return v * seconds
}

/// Fade in over the first 6% of a life and out over the last 18%.
func lifeAlpha(_ age: Float) -> Float {
    smoothstep(0, 0.06, age) * (1 - smoothstep(0.82, 1.0, age))
}

/// Where an emitter's particle is at a given age (0..1 of its life), at time t.
func wallPath(_ e: Emitter, age: Float, t: Float, mutant: Mutant = .none) -> SIMD3<Float> {
    let inIleum: Bool = e.route == .crypt || e.route == .villusAbsorb
    let axis: Float = inIleum ? ileumAxisY : colonAxisY
    let R: Float = inIleum ? ileumRadiusAt(e.x, panel: e.panel, t: t)
                           : colonRadiusAt(e.x, panel: e.panel, t: t)
    let front: Float = 0.25                // just in front of the cut face
    var rho: Float
    var z: Float = front
    var x: Float = e.x
    switch e.route {
    case .crypt:
        let tips: Bool = mutant == .tips && e.panel == 1
        // Deep in the lamina propria, up through the crypt, out of its mouth…
        let start: Float = tips ? R - villusHeight * 0.4 : R + ilealCryptDepth + 1.4
        let mouth: Float = tips ? R - villusHeight - 0.3 : R - 0.4
        if age < cryptCrossing {
            rho = mix(start, mouth, age / cryptCrossing)
        } else {
            // …then out into the lumen, carried downstream.
            let s: Float = (age - cryptCrossing) / (1 - cryptCrossing)
            let ease: Float = 1 - (1 - s) * (1 - s)
            rho = mix(mouth, R * 0.28, ease)
            z = mix(front, -R * 0.35, ease)
            x += drift(panel: e.panel, seconds: s * (1 - cryptCrossing) * e.life * loopSeconds)
        }
    case .villusAbsorb:
        let tip: Float = R - villusHeight
        if age < 0.45 {
            let s: Float = age / 0.45
            rho = mix(tip - 2.4, tip + 0.4, s)
            z = mix(-1.6, front, s)
            x += mix(-1.8, 0, s)
        } else {
            rho = mix(tip + 0.4, R + 0.9, (age - 0.45) / 0.55)
        }
    case .colonAbsorb:
        if age < 0.45 {
            let s: Float = age / 0.45
            rho = mix(R - 4.0, R + 0.3, s)
            z = mix(-1.8, front, s)
            x += mix(-2.2, 0, s)
        } else {
            rho = mix(R + 0.3, R + 2.9, (age - 0.45) / 0.55)
        }
    default:
        rho = R
    }
    return SIMD3<Float>(x, axis + e.side * rho, z)
}

// MARK: - the lumen contents

/// The lower panel's lumen: the stool table in exact proportion, three
/// particles for every 5 mmol/L (81 Na⁺, 60 Cl⁻, 9 K⁺, 27 HCO₃⁻ — every entry
/// of the table is a multiple of 5, so nothing is rounded), plus water and
/// mucus flecks (MODEL counts: at true proportions there would be ~185 water
/// molecules per ion, and the ions would vanish).
let lumenParticlesPer5mM: Int = 3
func lumenPopulation(panel: Int) -> [(Species, Int)] {
    if panel == 1 {
        let s: StoolElectrolytes = choleraStool
        let k: Int = lumenParticlesPer5mM
        return [(.sodium, s.sodium / 5 * k), (.chloride, s.chloride / 5 * k),
                (.potassium, s.potassium / 5 * k), (.bicarbonate, s.bicarbonate / 5 * k),
                (.water, 90), (.fleck, 24)]
    }
    // Upper panel: watery chyme at the valve, drying downstream. MODEL counts.
    return [(.water, 110), (.sodium, 10), (.chloride, 8)]
}

/// How much water is left in the normal lumen at x: all of it at the valve,
/// almost none by the rectum, where the stool is formed. MODEL profile.
func normalWetness(_ x: Float) -> Float {
    1 - 0.9 * smoothstep(-10, 95, x)
}

func lumenParticle(_ s: Species, index: Int, panel: Int, t: Float) -> Particle? {
    let w: Wave = wave(panel)
    let h1: Float = hash01(index, Int(s.rawValue), 101 + panel)
    let h2: Float = hash01(index, Int(s.rawValue), 202 + panel)
    let h3: Float = hash01(index, Int(s.rawValue), 303 + panel)
    var xb: Float
    var alpha: Float = 1
    if panel == 1 {
        // Carried across the whole span once per loop, wrapping, fast where
        // the gut is narrow.
        xb = flowX(fract(h1 + Float(w.contentLoops) * t / loopSeconds))
    } else {
        // Creeps; each particle lives one loop, starting where it was spawned.
        let age: Float = fract(h1 + t / loopSeconds)
        let spawn: Float = gutStartX + h2 * (60 - gutStartX)
        xb = spawn + w.contentMM * age
        alpha = sin(Float.pi * age) * normalWetness(xb)
    }
    let d: Float = waveOffset(xb, panel: panel, t: t)
    var x: Float = xb + w.surge * surgeFraction(d, wavelength: w.wavelength)
    if x >= gutEndX { x -= gutSpan }
    let r: Float = radiusPm(s) * panelMMPerPm * (s == .fleck ? mix(0.8, 1.6, h3) : 1)
    if x > breakStartX - r - 0.5 && x < breakEndX + r + 0.5 { alpha = 0 }
    guard alpha > 0.02 else { return nil }
    let f = lumenFrame(x, panel: panel, t: t)
    let rho: Float = f.fill * f.radius * sqrt(h2 * 0.92 + 0.08)
    let ang: Float = Float.pi * mix(0.12, 0.88, h3)
    var z: Float = -rho * sin(ang)
    z = min(z, -r - 0.3)
    let y: Float = f.axisY + rho * cos(ang)
    return Particle(position: SIMD3(x, y, z), radius: r, species: s, alpha: alpha,
                    panel: panel, route: .lumen)
}

// MARK: - the close-up of the crypt wall (panel 2), in µm

/// Enterocyte-like crypt cells in a row, apical side up. All MODEL: a schematic
/// with the proportions of a columnar epithelium, not a measured one.
let insetCellPitch: Float = 7.2
let insetGap: Float = 1.0
let insetCellHeight: Float = 13.0
let insetCellCount: Int = 8
var insetFirstCellX: Float { -Float(insetCellCount - 1) * insetCellPitch * 0.5 }
var insetTop: Float { insetCellHeight * 0.5 }
var insetBottom: Float { -insetCellHeight * 0.5 }

func insetCellX(_ k: Int) -> Float { insetFirstCellX + Float(k) * insetCellPitch }
func insetGapX(_ k: Int) -> Float { insetCellX(k) + insetCellPitch * 0.5 }

/// Two pulses per loop; within a pulse, Cl⁻ crosses first, Na⁺ starts once the
/// lumen has gone negative, and water starts last.
let insetPulses: Int = 2
let insetClStart: Float = 0.0
let insetNaStart: Float = 0.24
let insetWaterStart: Float = 0.44
let insetCrossing: Float = 0.46

func insetEmitters(mutant: Mutant = .none) -> [Emitter] {
    var out: [Emitter] = []
    for k in 0..<insetCellCount {
        out.append(Emitter(route: .transcellular, species: .chloride, panel: 2, x: insetCellX(k), side: 1,
                           phase: insetClStart + 0.08 * hash01(k, 1), life: insetCrossing, perLoop: insetPulses))
    }
    for k in 0..<(insetCellCount - 1) {
        let through: Bool = mutant == .naThroughCells
        out.append(Emitter(route: through ? .transcellular : .paracellular, species: .sodium, panel: 2,
                           x: through ? insetCellX(k) + 0.9 : insetGapX(k), side: 1,
                           phase: insetNaStart + 0.08 * hash01(k, 2), life: insetCrossing, perLoop: insetPulses))
    }
    for k in 0..<insetCellCount {
        let viaGap: Bool = k % 2 == 1 && k < insetCellCount - 1
        out.append(Emitter(route: viaGap ? .paracellular : .transcellular, species: .water, panel: 2,
                           x: viaGap ? insetGapX(k) : insetCellX(k) - 1.4, side: 1,
                           phase: insetWaterStart + 0.08 * hash01(k, 3), life: insetCrossing, perLoop: insetPulses))
    }
    return out
}

/// Close-up path: from the blood side, across the epithelium, into the crypt
/// lumen. Transcellular particles cross at their cell's centre line (through
/// NKCC1 at the base and CFTR at the top); paracellular ones up the gap.
func insetPath(_ e: Emitter, age: Float) -> SIMD3<Float> {
    let y0: Float = insetBottom - 3.2
    let y1: Float = insetTop + 3.6
    let s: Float = age
    let y: Float = mix(y0, y1, s)
    let wobble: Float = e.route == .transcellular ? 0.35 * sin(s * 9.0 + e.x) : 0
    return SIMD3<Float>(e.x + wobble * (1 - abs(2 * s - 1)), y, 0.6)
}

/// Every particle of every panel at time t.
func particles(at t: Float, mutant: Mutant = .none) -> [Particle] {
    var out: [Particle] = []
    for panel in 0..<2 {
        for e in emitters(panel: panel, mutant: mutant) {
            for k in 0..<e.perLoop {
                let age: Float = fract(t / loopSeconds * Float(e.perLoop) - e.phase * Float(e.perLoop) - Float(k))
                let norm: Float = age / (e.life * Float(e.perLoop))
                guard norm < 1 else { continue }
                let a: Float = lifeAlpha(norm)
                guard a > 0.02 else { continue }
                let r: Float = radiusPm(e.species) * panelMMPerPm
                out.append(Particle(position: wallPath(e, age: norm, t: t, mutant: mutant), radius: r,
                                    species: e.species, alpha: a, panel: panel, route: e.route))
            }
        }
        for (s, n) in lumenPopulation(panel: panel) {
            for i in 0..<n {
                if let p = lumenParticle(s, index: i, panel: panel, t: t) { out.append(p) }
            }
        }
    }
    for e in insetEmitters(mutant: mutant) {
        let age: Float = fract(t / loopSeconds * Float(e.perLoop) - e.phase)
        let norm: Float = age / e.life
        guard norm < 1 else { continue }
        let a: Float = lifeAlpha(norm)
        guard a > 0.02 else { continue }
        out.append(Particle(position: insetPath(e, age: norm), radius: radiusPm(e.species) * insetUMPerPm,
                            species: e.species, alpha: a, panel: 2, route: e.route))
    }
    return out
}

/// The fraction of its life at which a close-up particle reaches the apical
/// side (the lumen): its path runs from 3.2 µm below the cells to 3.6 above.
let insetApicalAge: Float = (insetTop - (insetBottom - 3.2)) / ((insetTop + 3.6) - (insetBottom - 3.2))
/// Pulse phase by which every particle of a group started at `start` (with up
/// to 0.08 of jitter) has reached the lumen.
func insetArrival(_ start: Float) -> (first: Float, last: Float) {
    (start + insetApicalAge * insetCrossing, start + 0.08 + insetApicalAge * insetCrossing)
}

/// How negative the crypt lumen is in the close-up, 0..1: rising as the Cl⁻
/// arrives, falling as the Na⁺ follows it through the junctions. For the "−"
/// signs drawn there.
func insetLumenCharge(at t: Float) -> Float {
    let p: Float = fract(t / loopSeconds * Float(insetPulses))
    let cl = insetArrival(insetClStart)
    let na = insetArrival(insetNaStart)
    let rise: Float = smoothstep(cl.first - 0.02, cl.last, p)
    let fall: Float = 1 - smoothstep(na.first - 0.02, na.last, p)
    return rise * fall
}

/// Which step of the close-up is happening: 0 while Cl⁻ is crossing to the
/// lumen, 1 once it is out and Na⁺ is being drawn through, 2 once Na⁺ is out
/// and water is following.
func insetStep(at t: Float) -> Int {
    let p: Float = fract(t / loopSeconds * Float(insetPulses))
    if p < insetArrival(insetClStart).last { return 0 }
    if p < insetArrival(insetNaStart).last { return 1 }
    return 2
}
