// Every particle on screen, as a pure function of time. Nothing is simulated
// and nothing carries state from frame to frame: a particle's position is
// computed from its emitter or its item number, its phase and the clock, and
// everything repeats with a period that divides the loop. So frame N is frame
// 0 again, exactly, and nothing ever runs backwards.
//
// Routes, which the tests read:
//   lumen          the contents, carried downstream by the flow and the wave
//   crypt          chloride secretion — present ONLY in the chloride mutant
//   villusAbsorb   absorption: from the lumen into a villus tip and down its core
//   villusSplit    lactose meets lactase at a villus tip, comes apart, and the
//                  glucose and galactose go down into the villus (upper panel)
//   osmoticInflow  water drawn out of a villus into the lumen, by osmosis alone
//   colonAbsorb    absorption: from the lumen into the colon's surface and away
//   scfaSalvage    an SCFA leaving the colon's lumen for its wall
//   transcellular  close-up: through a cell
//   paracellular   close-up: up the space between cells
//
// Lactose is drawn as a galactose sphere touching a glucose sphere. A bacterium
// is a rod of three beads. Both are made of spheres, because spheres are what
// the kernel intersects.

import Foundation
import simd

enum Route: Int {
    case lumen, crypt, villusAbsorb, villusSplit, osmoticInflow, colonAbsorb, scfaSalvage, transcellular, paracellular
}

/// What to break, for the mutation check. Each one must be caught by a test.
enum Mutant: String {
    case none
    case lactaseWorking = "lactase-working"   // the lower panel's villi split lactose too
    case chlorideOn = "chloride-on"           // step 23's crypt chloride secretion, switched on below
    case lowGap = "low-gap"                   // a stool mix whose osmotic gap is under 125
}

struct Particle {
    var position: SIMD3<Float>
    var radius: Float
    var species: Species
    var alpha: Float
    var panel: Int         // 0 lactase working, 1 lactase missing, 2 close-up
    var route: Route
    /// Which logical item this sphere belongs to (a lactose's two halves and a
    /// rod's three beads share one), for following it from frame to frame;
    /// −1 for wall traffic. `parent` is the lactose an SCFA or bubble came from.
    var item: Int = -1
    var parent: Int = -1
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
func mix(_ a: Float, _ b: Float, _ t: Float) -> Float { a + (b - a) * t }

/// Sugars and SCFAs are drawn at 0.6 of the ions' scale, so a glucose fits
/// beside a villus instead of hiding four of them. The order of sizes among
/// what is drawn is kept — glucose > SCFA > water > K⁺ > Na⁺ — and the key
/// says so. MODEL.
let organicScale: Float = 0.6

func isOrganic(_ s: Species) -> Bool { s == .glucose || s == .galactose || s == .scfa }

/// Drawn radius in the main panels, mm.
func drawnRadius(_ s: Species) -> Float {
    switch s {
    case .gas: return bubbleMinMM
    case .bacterium: return bacteriumSphereMM
    default: return radiusPm(s) * panelMMPerPm * (isOrganic(s) ? organicScale : 1)
    }
}

/// Drawn radius in the close-up, µm.
func insetRadius(_ s: Species) -> Float {
    radiusPm(s) * insetUMPerPm * (isOrganic(s) ? organicScale : 1)
}

// MARK: - emitters: traffic across the wall

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
    var tag: Int = 0        // close-up: which cell; split: which event
}

/// Step 23's crypt burst, for the chloride mutant only: Cl⁻, then Na⁺, then water.
func burstOrder() -> [(Species, Float)] {
    [(.chloride, 0.0), (.sodium, 0.05), (.water, 0.10)]
}

let cryptLife: Float = 0.40
let cryptCrossing: Float = 0.45
/// Absorbed, split and osmotic particles: across the wall in 0.48 of a loop.
let absorbLife: Float = 0.48
/// When in its life a lactose at a villus tip comes apart.
let splitAge: Float = 0.45

/// The part of each life during which a water symbol carries an arrow — while
/// it crosses the wall. Every route's window lasts the same 0.24 of a loop, so
/// the number of arrows on screen is proportional to the flux and the two
/// panels can be compared by eye. A test holds this.
func arrowWindow(_ r: Route) -> (from: Float, to: Float, life: Float) {
    switch r {
    case .crypt: return (0.05, 0.65, cryptLife)
    default: return (0.25, 0.75, absorbLife)
    }
}

func emitters(panel: Int, mutant: Mutant = .none) -> [Emitter] {
    var out: [Emitter] = []
    let villi: [Float] = villusXs()
    let colon: [Float] = colonSurfaceXs()
    let tipSites: [(Float, Float)] = villi.filter { $0 > gutStartX + 4 }.flatMap { [($0, Float(1)), ($0, Float(-1))] }

    // Chloride secretion: none. Only the mutant turns step 23's crypts on.
    if chlorideSecretion(panel: panel, mutant: mutant) {
        for (i, x) in ilealCryptXs().enumerated() {
            for side: Float in [1, -1] {
                let s: Int = side > 0 ? 0 : 1
                let phase: Float = hash01(i, s, 11 + panel)
                for (sp, delay) in burstOrder() {
                    out.append(Emitter(route: .crypt, species: sp, panel: panel, x: x, side: side,
                                       phase: fract(phase + delay), life: cryptLife, perLoop: 1))
                }
            }
        }
    }

    // Lactose split by lactase on the brush border — only where lactase is.
    // Each event is a glucose and a galactose that arrive together, bonded,
    // and leave apart, down the villus: SGLT1 carries both across the apical
    // membrane with Na⁺ (Wright, Loo & Hirayama, Physiol Rev 91:733, 2011; its
    // mutations cause glucose–galactose malabsorption).
    if lactasePresent(panel: panel, mutant: mutant) {
        for k in 0..<upperSplitEvents {
            let pick: Int = (k * 29 + 5 + panel * 3) % tipSites.count
            let (x, side) = tipSites[pick]
            let phase: Float = hash01(k, panel, 41)
            for sp: Species in [.glucose, .galactose] {
                out.append(Emitter(route: .villusSplit, species: sp, panel: panel, x: x, side: side,
                                   phase: phase, life: absorbLife, perLoop: 1, tag: k))
            }
        }
    }

    // Absorption at villus tips: water, and Na⁺ (much of it riding with the
    // sugars on SGLT1). The lower panel's villi still absorb what they can.
    let villusWater: Int = panel == 0 ? upperVillusWater : lowerVillusWater
    let villusSodium: Int = panel == 0 ? upperVillusSodium : lowerVillusSodium
    for k in 0..<(villusWater + villusSodium) {
        let pick: Int = (k * 37 + panel * 11) % tipSites.count
        let (x, side) = tipSites[pick]
        out.append(Emitter(route: .villusAbsorb, species: k < villusWater ? .water : .sodium,
                           panel: panel, x: x, side: side, phase: hash01(k, panel, 23),
                           life: absorbLife, perLoop: 1))
    }

    // Osmosis: water drawn into the lumen by the lactose that stays there.
    // Water only — no ion goes first, or at all.
    if panel == 1 {
        for k in 0..<lowerOsmoticWater {
            let pick: Int = (k * 17 + 3) % tipSites.count
            let (x, side) = tipSites[pick]
            out.append(Emitter(route: .osmoticInflow, species: .water, panel: panel, x: x, side: side,
                               phase: hash01(k, panel, 57), life: absorbLife, perLoop: 1))
        }
    }

    // Colonic absorption: water, with Na⁺ through the surface cells.
    let colonWater: Int = panel == 0 ? upperColonWater : lowerColonWater
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
        case .crypt, .osmoticInflow: into += e.perLoop
        case .villusAbsorb: out += e.perLoop
        case .colonAbsorb: out += e.perLoop; colon += e.perLoop
        default: break
        }
    }
    return (into, out, colon)
}

// MARK: - paths across the wall

/// The lumen's downstream drift, in mm, after `seconds`.
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
    let inIleum: Bool = e.route != .colonAbsorb
    let axis: Float = inIleum ? ileumAxisY : colonAxisY
    let R: Float = inIleum ? ileumRadiusAt(e.x, panel: e.panel, t: t)
                           : colonRadiusAt(e.x, panel: e.panel, t: t)
    let front: Float = 0.25                // just in front of the cut face
    let tip: Float = R - villusHeight
    var rho: Float
    var z: Float = front
    var x: Float = e.x
    switch e.route {
    case .crypt:
        let start: Float = R + ilealCryptDepth + 1.4
        let mouth: Float = R - 0.4
        if age < cryptCrossing {
            rho = mix(start, mouth, age / cryptCrossing)
        } else {
            let s: Float = (age - cryptCrossing) / (1 - cryptCrossing)
            let ease: Float = 1 - (1 - s) * (1 - s)
            rho = mix(mouth, R * 0.28, ease)
            z = mix(front, -R * 0.35, ease)
            x += drift(panel: e.panel, seconds: s * (1 - cryptCrossing) * e.life * loopSeconds)
        }
    case .villusAbsorb:
        if age < 0.45 {
            let s: Float = age / 0.45
            rho = mix(tip - 2.4, tip + 0.4, s)
            z = mix(-1.6, front, s)
            x += mix(-1.8, 0, s)
        } else {
            rho = mix(tip + 0.4, R + 0.9, (age - 0.45) / 0.55)
        }
    case .villusSplit:
        // Bonded, in from the lumen until the pair touches the tip, where LPH
        // sits; then apart, down into the villus.
        let r: Float = drawnRadius(e.species)
        let sign: Float = e.species == .glucose ? 1 : -1
        if age < splitAge {
            let s: Float = age / splitAge
            rho = mix(tip - 3.4, tip - r * 0.55, s)
            z = mix(-1.8, front, s)
            x += mix(-2.2, 0, s) + sign * r * 0.92
        } else {
            let s: Float = (age - splitAge) / (1 - splitAge)
            let ease: Float = 1 - (1 - s) * (1 - s)
            rho = mix(tip - r * 0.55, R + 0.9, ease)
            x += sign * mix(r * 0.92, 0.22, smoothstep(0, 0.5, s))
        }
    case .osmoticInflow:
        // Up the villus core to its tip, then out into the lumen and away.
        if age < 0.55 {
            rho = mix(R + 0.9, tip + 0.4, age / 0.55)
        } else {
            let s: Float = (age - 0.55) / 0.45
            rho = mix(tip + 0.4, tip - 2.6, s)
            z = mix(front, -1.6, s)
            x += mix(0, 2.4, s)
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

// MARK: - the lumen

/// A place in the lumen: along the gut, how far out (0..1) and at what angle
/// round the back half of the tube.
struct Lateral {
    var reach: Float      // 0 at the axis, 1 at the fill limit
    var angle: Float      // 0 the top of the lumen, π the bottom, through the back

    static func random(_ a: Int, _ b: Int, _ c: Int) -> Lateral {
        Lateral(reach: sqrt(hash01(a, b, c) * 0.92 + 0.08), angle: Float.pi * mix(0.12, 0.88, hash01(a, b, c + 1)))
    }
    func blend(_ o: Lateral, _ t: Float) -> Lateral {
        Lateral(reach: mix(reach, o.reach, t), angle: mix(angle, o.angle, t))
    }
}

func lumenPoint(_ x: Float, _ l: Lateral, panel: Int, t: Float) -> SIMD3<Float> {
    let f = lumenFrame(x, panel: panel, t: t)
    let rho: Float = f.fill * f.radius * l.reach
    let y: Float = f.axisY + rho * cos(l.angle)
    let z: Float = -rho * sin(l.angle)
    return SIMD3(x, y, z)
}

/// A sphere, kept behind the cut plane, and hidden in the colon's break.
func sphere(_ p: SIMD3<Float>, _ s: Species, radius r: Float, alpha: Float, panel: Int,
            route: Route, item: Int, parent: Int = -1) -> Particle? {
    var q: SIMD3<Float> = p
    if route == .lumen { q.z = min(q.z, -r - 0.3) }
    if q.x > breakStartX - r - 0.5 && q.x < breakEndX + r + 0.5 { return nil }
    guard alpha > 0.02 else { return nil }
    return Particle(position: q, radius: r, species: s, alpha: alpha, panel: panel, route: route,
                    item: item, parent: parent)
}

/// Lactose: galactose and glucose touching, along a direction in the x–y plane.
func lactose(_ p: SIMD3<Float>, turn: Float, alpha: Float, panel: Int, route: Route, item: Int) -> [Particle] {
    let r: Float = drawnRadius(.glucose)
    let d = SIMD3<Float>(cos(turn), sin(turn), 0) * (r * 0.92)
    // Both halves or neither: a lactose is never drawn half-hidden.
    guard let a = sphere(p + d, .glucose, radius: r, alpha: alpha, panel: panel, route: route, item: item),
          let b = sphere(p - d, .galactose, radius: r, alpha: alpha, panel: panel, route: route, item: item)
    else { return [] }
    return [a, b]
}

/// A rod-shaped bacterium: three beads in a line.
func bacterium(_ p: SIMD3<Float>, turn: Float, alpha: Float, panel: Int, item: Int) -> [Particle] {
    let d = SIMD3<Float>(cos(turn), sin(turn), 0) * bacteriumBeadSpacing
    var out: [Particle] = []
    for k in -1...1 {
        guard let b = sphere(p + d * Float(k), .bacterium, radius: bacteriumSphereMM, alpha: alpha,
                             panel: panel, route: .lumen, item: item) else { return [] }
        out.append(b)
    }
    return out
}

/// Item numbers: panel, kind, index — unique, so a test can follow each one.
enum Kind: Int { case water = 1, sodium, potassium, lactose, bacterium, scfa, gas, freeBacterium }
func itemID(_ panel: Int, _ kind: Kind, _ i: Int, _ sub: Int = 0) -> Int {
    ((panel * 16 + kind.rawValue) * 10_000 + i) * 8 + sub
}

/// Upper panel: watery chyme at the valve, drying downstream. MODEL counts.
let upperWater: Int = 110
let upperSodium: Int = 10
/// Lactose still in the upper panel's lumen upstream: gone well before the
/// valve, split and absorbed. MODEL count.
let upperLactose: Int = 22
/// Bacteria in the colon: in BOTH panels — the colon always has them. In the
/// upper panel no lactose reaches them, so they make nothing from it. (They
/// ferment dietary fibre every day, and that is not drawn.) MODEL counts.
let upperBacteria: Int = 26
let freeBacteria: Int = 14
/// Lower panel water. MODEL count.
let lowerWater: Int = 70

/// How much water is left in the normal lumen at x: all of it at the valve,
/// almost none by the rectum, where the stool is formed. MODEL profile.
func normalWetness(_ x: Float) -> Float {
    1 - 0.9 * smoothstep(-10, 95, x)
}

/// Bacteria fade in across the valve: the colon holds vastly more than the
/// ileum, and those upstream are not drawn. MODEL.
func bacteriaPresence(_ x: Float) -> Float { smoothstep(-12, -2, x) }

/// Upper panel: each item creeps and lives one loop, as in step 23.
func upperLumen(t: Float) -> [Particle] {
    let w: Wave = normalWave
    var out: [Particle] = []
    func place(_ i: Int, _ salt: Int, spawnFrom a: Float, to b: Float) -> (x: Float, age: Float, lat: Lateral) {
        let h1: Float = hash01(i, salt, 101)
        let h2: Float = hash01(i, salt, 202)
        let age: Float = fract(h1 + t / loopSeconds)
        let xb: Float = mix(a, b, h2) + w.contentMM * age
        let d: Float = waveOffset(xb, panel: 0, t: t)
        var x: Float = xb + w.surge * surgeFraction(d, wavelength: w.wavelength)
        if x >= gutEndX { x -= gutSpan }
        return (x, age, Lateral.random(i, salt, 303))
    }
    let simple: [(Species, Kind, Int)] = [(.water, .water, upperWater), (.sodium, .sodium, upperSodium)]
    for (s, kind, n) in simple {
        for i in 0..<n {
            let p = place(i, kind.rawValue, spawnFrom: gutStartX, to: 60)
            let a: Float = sin(Float.pi * p.age) * normalWetness(p.x)
            if let q = sphere(lumenPoint(p.x, p.lat, panel: 0, t: t), s, radius: drawnRadius(s), alpha: a,
                              panel: 0, route: .lumen, item: itemID(0, kind, i)) { out.append(q) }
        }
    }
    for i in 0..<upperLactose {
        let p = place(i, Kind.lactose.rawValue, spawnFrom: gutStartX, to: -62)
        let a: Float = sin(Float.pi * p.age) * (1 - smoothstep(-70, -48, p.x))
        out += lactose(lumenPoint(p.x, p.lat, panel: 0, t: t), turn: 6.283 * hash01(i, 7, 0), alpha: a,
                       panel: 0, route: .lumen, item: itemID(0, .lactose, i))
    }
    for i in 0..<upperBacteria {
        let p = place(i, Kind.bacterium.rawValue, spawnFrom: -6, to: 140)
        let a: Float = sin(Float.pi * p.age) * bacteriaPresence(p.x)
        out += bacterium(lumenPoint(p.x, p.lat, panel: 0, t: t), turn: 6.283 * hash01(i, 9, 0), alpha: a,
                         panel: 0, item: itemID(0, .bacterium, i))
    }
    return out
}

/// Where lactose parcel j (lower panel) is taken up by its bacterium; nil if
/// it survives to the stool. The last `lactose` parcels of the stool mix survive.
func fermentX(_ j: Int, mutant: Mutant = .none) -> Float? {
    if j >= lactoseParcels - stool(mutant: mutant).lactose { return nil }
    return mix(4, 70, hash01(j, 3, 61))
}

/// Every SCFA the lower colon makes, in order: (parcel, child).
func scfaChildren(mutant: Mutant = .none) -> [(Int, Int)] {
    var out: [(Int, Int)] = []
    for j in 0..<lactoseParcels where fermentX(j, mutant: mutant) != nil {
        for k in 0..<scfaYield(parcel: j) { out.append((j, k)) }
    }
    return out
}

/// Whether the c-th SCFA is taken back by the colon. Multiplying by 41 mod n
/// is a permutation of 0..<n when 41 does not divide n, so exactly
/// `scfaAbsorbed` of them are chosen, spread evenly through the loop.
func scfaSalvaged(_ c: Int, of n: Int, absorbed: Int) -> Bool { (c * 41) % n < absorbed }

/// Where a salvaged SCFA starts for the wall: somewhere downstream of where it
/// was made, and always before the break. MODEL.
/// How deep into the colonic wall a salvaged SCFA is followed: past the
/// crypts, into the tissue under them. MODEL.
let salvageDepth: Float = colonCryptDepth + 0.8

func salvageX(_ c: Int, from xf: Float) -> Float { min(xf + mix(10, 46, hash01(c, 5, 71)), 78) }

/// Lower panel: everything is carried across the span once a loop, as in step
/// 23's lower panel, and what a thing IS depends on where it has got to.
func lowerLumen(t: Float, mutant: Mutant = .none) -> [Particle] {
    let w: Wave = lowerWave
    var out: [Particle] = []
    func flow(_ h1: Float) -> Float {
        let xb: Float = flowX(fract(h1 + Float(w.contentLoops) * t / loopSeconds))
        let d: Float = waveOffset(xb, panel: 1, t: t)
        var x: Float = xb + w.surge * surgeFraction(d, wavelength: w.wavelength)
        if x >= gutEndX { x -= gutSpan }
        return x
    }
    let mixNow: StoolMix = stool(mutant: mutant)
    let simple: [(Species, Kind, Int)] = [(.water, .water, lowerWater), (.sodium, .sodium, mixNow.sodium),
                                          (.potassium, .potassium, mixNow.potassium)]
    for (s, kind, n) in simple {
        for i in 0..<n {
            let x: Float = flow(hash01(i, kind.rawValue, 111))
            if let q = sphere(lumenPoint(x, Lateral.random(i, kind.rawValue, 313), panel: 1, t: t), s,
                              radius: drawnRadius(s), alpha: 1, panel: 1, route: .lumen,
                              item: itemID(1, kind, i)) { out.append(q) }
        }
    }

    let children: [(Int, Int)] = scfaChildren(mutant: mutant)
    let absorbed: Int = scfaAbsorbed(mutant: mutant)
    var childIndex: [Int: Int] = [:]
    for (c, jk) in children.enumerated() { childIndex[jk.0 * 8 + jk.1] = c }

    for j in 0..<lactoseParcels {
        let x: Float = flow(hash01(j, Kind.lactose.rawValue, 111))
        let own: Lateral = Lateral.random(j, Kind.lactose.rawValue, 313)
        let bug: Lateral = Lateral.random(j, Kind.bacterium.rawValue, 313)
        let bugTurn: Float = 6.283 * hash01(j, 9, 1)
        let lactoseID: Int = itemID(1, .lactose, j)
        guard let xf = fermentX(j, mutant: mutant) else {
            out += lactose(lumenPoint(x, own, panel: 1, t: t), turn: 6.283 * hash01(j, 7, 1), alpha: 1,
                           panel: 1, route: .lumen, item: lactoseID)
            continue
        }
        // Its bacterium travels with it, alongside.
        out += bacterium(lumenPoint(x, bug, panel: 1, t: t), turn: bugTurn, alpha: bacteriaPresence(x),
                         panel: 1, item: itemID(1, .bacterium, j))
        // The lactose closes in on the bacterium and is taken up at xf.
        let meet: Float = smoothstep(xf - 16, xf - 1, x)
        let aL: Float = 1 - smoothstep(xf - 2.5, xf, x)
        if aL > 0.02 {
            out += lactose(lumenPoint(x, own.blend(bug, meet), panel: 1, t: t), turn: 6.283 * hash01(j, 7, 1),
                           alpha: aL, panel: 1, route: .lumen, item: lactoseID)
        }
        guard x > xf else { continue }
        let born: Float = smoothstep(xf, xf + 2.5, x)
        // SCFAs out of the bacterium, spreading into the contents.
        for k in 0..<scfaYield(parcel: j) {
            let c: Int = childIndex[j * 8 + k]!
            let target: Lateral = Lateral.random(j * 8 + k, Kind.scfa.rawValue, 313)
            var p: SIMD3<Float> = lumenPoint(x, bug.blend(target, smoothstep(xf, xf + 12, x)), panel: 1, t: t)
            var a: Float = born
            var route: Route = .lumen
            if scfaSalvaged(c, of: children.count, absorbed: absorbed) {
                let xs: Float = salvageX(c, from: xf)
                let go: Float = smoothstep(xs, xs + 8, x)
                if go > 0 {
                    route = .scfaSalvage
                    let side: Float = target.angle < Float.pi / 2 ? 1 : -1
                    let R: Float = colonRadiusAt(x, panel: 1, t: t)
                    let wall = SIMD3<Float>(x, colonAxisY + side * (R + salvageDepth), 0.25)
                    p = p + (wall - p) * go
                    a *= 1 - smoothstep(xs + 6, xs + 9, x)
                }
            }
            if let q = sphere(p, .scfa, radius: drawnRadius(.scfa), alpha: a, panel: 1, route: route,
                              item: itemID(1, .scfa, j, k), parent: lactoseID) { out.append(q) }
        }
        // Gas: rising to the top of the lumen and growing as bubbles merge.
        for b in 0..<bubblesPerLactose {
            let top = Lateral(reach: mix(0.72, 0.95, hash01(j, b, 81)), angle: Float.pi * mix(0.13, 0.32, hash01(j, b, 83)))
            let p: SIMD3<Float> = lumenPoint(x + Float(b) * 1.4 - 0.7, bug.blend(top, smoothstep(xf, xf + 26, x)), panel: 1, t: t)
            let r: Float = mix(bubbleMinMM, bubbleMaxMM, smoothstep(xf, xf + 70, x))
            if let q = sphere(p, .gas, radius: r, alpha: born, panel: 1, route: .lumen,
                              item: itemID(1, .gas, j, b), parent: lactoseID) { out.append(q) }
        }
    }
    for i in 0..<freeBacteria {
        let x: Float = flow(hash01(i, Kind.freeBacterium.rawValue, 111))
        out += bacterium(lumenPoint(x, Lateral.random(i, Kind.freeBacterium.rawValue, 313), panel: 1, t: t),
                         turn: 6.283 * hash01(i, 9, 2), alpha: bacteriaPresence(x), panel: 1,
                         item: itemID(1, .freeBacterium, i))
    }
    return out
}

// MARK: - the close-up of the brush border (panel 2), in µm

/// Enterocytes in a row, apical side up. All MODEL: a schematic with the
/// proportions of a columnar epithelium, not a measured one. The left four
/// have lactase in their brush border; the right four do not. All eight have
/// SGLT1 on top and GLUT2 below — only lactase differs.
let insetCellPitch: Float = 7.2
let insetGap: Float = 1.0
let insetCellHeight: Float = 13.0
let insetCellCount: Int = 8
let insetLactaseCells: Int = 4
var insetFirstCellX: Float { -Float(insetCellCount - 1) * insetCellPitch * 0.5 }
var insetTop: Float { insetCellHeight * 0.5 }
var insetBottom: Float { -insetCellHeight * 0.5 }

func insetCellX(_ k: Int) -> Float { insetFirstCellX + Float(k) * insetCellPitch }
func insetGapX(_ k: Int) -> Float { insetCellX(k) + insetCellPitch * 0.5 }
func insetHasLactase(_ k: Int) -> Bool { k < insetLactaseCells }

let insetPulses: Int = 2
let insetCrossing: Float = 0.46
/// A lactose's whole journey, lumen to blood, takes longer: most of each
/// pulse, so the lactase cells are seldom idle on screen. MODEL.
let insetSugarLife: Float = 0.85
/// When in its life a close-up lactose touches the brush border and splits.
let insetSplitAge: Float = 0.3
/// Where the two sugars cross the apical membrane, either side of the cell's
/// centre line, through SGLT1.
let insetSugarOffset: Float = 1.75

func insetEmitters(mutant: Mutant = .none) -> [Emitter] {
    var out: [Emitter] = []
    for k in 0..<insetCellCount {
        if insetHasLactase(k) {
            // Lactose split at this cell's brush border; each sugar crosses
            // with two Na⁺ (SGLT1's 2:1 — Wright et al. 2011).
            let phase: Float = 0.06 * hash01(k, 1)
            for sp: Species in [.glucose, .galactose, .sodium, .sodium, .sodium, .sodium] {
                out.append(Emitter(route: .transcellular, species: sp, panel: 2, x: insetCellX(k), side: 1,
                                   phase: phase, life: insetSugarLife, perLoop: insetPulses, tag: k))
            }
        } else {
            // No lactase: water comes the other way, by osmosis, through the
            // cells and between them.
            for j in 0..<2 {
                let viaGap: Bool = j == 1 && k < insetCellCount - 1
                out.append(Emitter(route: viaGap ? .paracellular : .transcellular, species: .water, panel: 2,
                                   x: viaGap ? insetGapX(k) : insetCellX(k) - 1.4, side: -1,
                                   phase: 0.04 + 0.4 * hash01(k, j, 3), life: insetCrossing, perLoop: insetPulses, tag: k))
            }
        }
    }
    return out
}

/// The positions of one close-up emitter's particle. Sugar and Na⁺ come down
/// from the lumen; water goes up from the blood side. `index` numbers the Na⁺
/// of one cell (0, 1 ride with the glucose; 2, 3 with the galactose).
func insetPath(_ e: Emitter, age: Float, index: Int = 0) -> SIMD3<Float> {
    let yLumen: Float = insetTop + 4.4
    let yBlood: Float = insetBottom - 3.2
    if e.species == .water {
        let y: Float = mix(yBlood, yLumen, age)
        let wobble: Float = e.route == .transcellular ? 0.35 * sin(age * 9.0 + e.x) : 0
        return SIMD3(e.x + wobble * (1 - abs(2 * age - 1)), y, 0.6)
    }
    let rs: Float = insetRadius(.glucose)
    let withGlucose: Bool = e.species == .glucose || (e.species == .sodium && index < 2)
    let sign: Float = withGlucose ? 1 : -1
    let touchY: Float = insetTop + rs * 0.8
    var x: Float
    var y: Float
    if age < insetSplitAge {
        let s: Float = age / insetSplitAge
        y = mix(yLumen, touchY, s)
        x = e.x + sign * rs * 0.92
    } else {
        let s: Float = (age - insetSplitAge) / (1 - insetSplitAge)
        y = mix(touchY, yBlood, s)
        x = e.x + sign * mix(rs * 0.92, insetSugarOffset, smoothstep(0, 0.2, s))
    }
    if e.species == .sodium {
        // Riding above its sugar, through the same carrier.
        let stack: Float = Float(index % 2)
        y += rs + insetRadius(.sodium) * (1.2 + 2.2 * stack)
    }
    return SIMD3(x, y, 0.6)
}

/// Lactose drifting past the lactase-free cells, untouched. MODEL count.
let insetDrifters: Int = 3
func insetDrifter(_ i: Int, t: Float) -> (position: SIMD3<Float>, alpha: Float) {
    let a: Float = insetGapX(insetLactaseCells - 1) + 2.0
    let b: Float = insetCellX(insetCellCount - 1) + 3.2
    let u: Float = fract(Float(i) / Float(insetDrifters) + t / loopSeconds * Float(insetPulses))
    let x: Float = mix(a, b, u)
    let y: Float = insetTop + 4.0 + 0.5 * sin(6.283 * (u + Float(i) * 0.3))
    let alpha: Float = smoothstep(0, 0.08, u) * (1 - smoothstep(0.9, 1, u))
    return (SIMD3(x, y, 0.6), alpha)
}

// MARK: - everything

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
                out.append(Particle(position: wallPath(e, age: norm, t: t, mutant: mutant), radius: drawnRadius(e.species),
                                    species: e.species, alpha: a, panel: panel, route: e.route))
            }
        }
    }
    out += upperLumen(t: t)
    out += lowerLumen(t: t, mutant: mutant)

    var sodiumSeen: [Int: Int] = [:]
    for e in insetEmitters(mutant: mutant) {
        var index: Int = 0
        if e.species == .sodium { index = sodiumSeen[e.tag, default: 0]; sodiumSeen[e.tag] = index + 1 }
        let age: Float = fract(t / loopSeconds * Float(e.perLoop) - e.phase)
        let norm: Float = age / e.life
        guard norm < 1 else { continue }
        var a: Float = lifeAlpha(norm)
        // Na⁺ joins at the carrier, once the lactose has been split.
        if e.species == .sodium { a *= smoothstep(insetSplitAge - 0.02, insetSplitAge + 0.04, norm) }
        guard a > 0.02 else { continue }
        out.append(Particle(position: insetPath(e, age: norm, index: index), radius: insetRadius(e.species),
                            species: e.species, alpha: a, panel: 2, route: e.route))
    }
    let rs: Float = insetRadius(.glucose)
    for i in 0..<insetDrifters {
        let d = insetDrifter(i, t: t)
        guard d.alpha > 0.02 else { continue }
        let off = SIMD3<Float>(rs * 0.92, 0, 0)
        out.append(Particle(position: d.position + off, radius: rs, species: .glucose, alpha: d.alpha,
                            panel: 2, route: .lumen, item: 900 + i))
        out.append(Particle(position: d.position - off, radius: rs, species: .galactose, alpha: d.alpha,
                            panel: 2, route: .lumen, item: 900 + i))
    }
    return out
}
