// Tests for step 76, the Whitehall pulling boat. Its size is measured off
// the kernel's own distance function and held to Newfound's published
// figures; its waterline is held to Archimedes — the water the hull
// displaces, counted on the GPU from the same distance function at the
// drawn draft, weighs what the boat weighs; the fitted lines are held to
// Newfound's full-load waterline figures; the oars and oarlocks to Shaw &
// Tenney's and Angus Rowboats' rules; the distance function to the
// definition of a distance; and the picture and its scale bars to the
// camera.
//
// BOAT_MUTANT=floatsWrong|wrongSize|noOarlocks breaks the step on purpose;
// `make mutants` requires the suite to fail for each.

import CoreGraphics
import Foundation
import Metal
import simd

let mutant: Mutant = activeMutant
if mutant != .none { print("MUTANT: \(mutant.rawValue)") }

let scale: Float = boatScale(mutant)
let buoy: Buoyancy = solveBuoyancy(scale: scale)
let draft: Float = drawnDraft(buoy, mutant)
let layout: Layout = buildLayout(scale: scale, drawnDraft: draft)
let hull: HullShape = layout.hull
let setup: StillSetup = stillSetup(mutant)
let renderer: BoatRenderer? = {
    guard let d = try? findDevice() else { return nil }
    return try? BoatRenderer(device: d, width: 1920, height: 1080, studio: setup.studio, layout: layout, draft: draft, mutant: mutant)
}()
/// World ↔ hull frame.
let lift = SIMD3<Float>(0, draft, 0)

section("the boat, against Newfound's published figures")

/// Sphere-trace a batch of rays against the envelope (probe .z) and return
/// each ray's hit distance (∞ for a miss).
func traceEnvelope(_ starts: [SIMD3<Float>], _ dir: SIMD3<Float>, maxT: Float) -> [Float] {
    guard let r = renderer else { return starts.map { _ in Float.nan } }
    var t: [Float] = Array(repeating: 0, count: starts.count)
    var done: [Bool] = Array(repeating: false, count: starts.count)
    for _ in 0..<600 {
        let pts: [SIMD3<Float>] = (0..<starts.count).map { starts[$0] + dir * t[$0] }
        guard let pr = try? r.probe(pts) else { return starts.map { _ in Float.nan } }
        var active: Int = 0
        for i in 0..<starts.count where !done[i] {
            let d: Float = pr[i].z
            if d < 0.001 { done[i] = true } else { t[i] += d * 0.95; active += 1 }
            if t[i] > maxT { done[i] = true; t[i] = .infinity }
        }
        if active == 0 { break }
    }
    return t
}

/// How far the envelope reaches along `axis`: parallel rays from 3 m out,
/// on a grid across `u` and `v`, the nearest hit.
func reach(along axis: SIMD3<Float>, across u: SIMD3<Float>, _ v: SIMD3<Float>, uRange: ClosedRange<Float>, vRange: ClosedRange<Float>) -> Float {
    let n: Int = 60
    var starts: [SIMD3<Float>] = []
    for i in 0..<n {
        for j in 0..<n {
            let a: Float = uRange.lowerBound + (uRange.upperBound - uRange.lowerBound) * Float(i) / Float(n - 1)
            let b: Float = vRange.lowerBound + (vRange.upperBound - vRange.lowerBound) * Float(j) / Float(n - 1)
            starts.append(axis * 3000 + u * a + v * b - lift)
        }
    }
    let t: [Float] = traceEnvelope(starts, -axis, maxT: 6000)
    return 3000 - (t.min() ?? .infinity)
}

test("5080 × 1066.8 mm (16 ft 8 in × 42 in), 16.2 in deep amidships, 27.99 in at the bow [NW]: off the kernel, to 1 mm") {
    guard renderer != nil else { expect(false, "no GPU"); return }
    let X = SIMD3<Float>(1, 0, 0), Y = SIMD3<Float>(0, 1, 0), Z = SIMD3<Float>(0, 0, 1)
    let big: Float = 1.3 * scale
    let yr: ClosedRange<Float> = -10...(750 * big)
    let xr: ClosedRange<Float> = (-2600 * big)...(2600 * big)
    let zr: ClosedRange<Float> = (-560 * big)...(560 * big)
    let loa: Float = reach(along: X, across: Y, Z, uRange: yr, vRange: zr) + reach(along: -X, across: Y, Z, uRange: yr, vRange: zr)
    let boa: Float = reach(along: Z, across: X, Y, uRange: xr, vRange: yr) + reach(along: -Z, across: X, Y, uRange: xr, vRange: yr)
    // Amidships: down onto the sheer beside the centreline, up onto the
    // keel at it; at the bow, down onto the stem head.
    let down: [Float] = traceEnvelope([SIMD3<Float>(0, 1500, 480 * scale) - lift, SIMD3<Float>(hull.halfLength - 10, 1500, 0) - lift],
                                      SIMD3<Float>(0, -1, 0), maxT: 3000)
    let up: [Float] = traceEnvelope([SIMD3<Float>(0, -1500, 0) - lift], SIMD3<Float>(0, 1, 0), maxT: 3000)
    let sheerMid: Float = 1500 - down[0]
    let stemHead: Float = 1500 - down[1]
    let keel: Float = -1500 + up[0]
    let depth: Float = sheerMid - keel
    let bow: Float = stemHead - keel
    print(String(format: "        drawn %.2f × %.2f mm, depth %.2f mm amidships, %.2f mm at the bow; Newfound %.1f × %.1f, %.1f, %.1f",
                 loa, boa, depth, bow, nwLOA, nwBOA, nwCenterDepth, nwBowDepth))
    expect(abs(loa - nwLOA) < 1, "length \(loa)")
    expect(abs(boa - nwBOA) < 1, "beam \(boa)")
    expect(abs(depth - nwCenterDepth) < 1, "depth \(depth)")
    expect(abs(bow - nwBowDepth) < 1, "bow \(bow)")
}

test("the fitted lines at Newfound's full load (725 lb): waterline 15.23 ft and 37.8 in within 1%, draft 8.13 in within 2%") {
    let wl = hull.waterline(at: buoy.draftAtCapacityFresh)
    print(String(format: "        at 725 lb in fresh water: draft %.1f mm (Newfound %.1f); LWL %.0f mm (%.0f); BWL %.1f mm (%.1f); in sea water %.1f mm",
                 buoy.draftAtCapacityFresh, nwDraftAtCapacity, wl.length, nwLWL, wl.beam, nwBWL, buoy.draftAtCapacitySea))
    expect(abs(wl.length - nwLWL) < 0.01 * nwLWL, "LWL \(wl.length)")
    expect(abs(wl.beam - nwBWL) < 0.01 * nwBWL, "BWL \(wl.beam)")
    expect(abs(buoy.draftAtCapacityFresh - nwDraftAtCapacity) < 0.02 * nwDraftAtCapacity, "draft \(buoy.draftAtCapacityFresh)")
}

test("the kernel's hull is Boat.swift's: points on the CPU's surface are on the GPU's, to 0.3 mm") {
    guard let r = renderer else { expect(false, "no GPU"); return }
    var pts: [SIMD3<Float>] = []
    var i: Int = 0
    while pts.count < 4000 {
        i += 1
        let x: Float = -hull.halfLength + hull.length * Float((i * 7919) % 1000) / 1000
        let y0: Float = hull.bottom(x) + 2
        let y: Float = y0 + (hull.sheer(x) - 2 - y0) * Float((i * 104729) % 997) / 997
        let w: Float = hull.halfWidth(x: x, y: y)
        if w > 20 { pts.append(SIMD3<Float>(x, y, (i % 2 == 0 ? w : -w)) - lift) }
    }
    guard let pr = try? r.probe(pts) else { expect(false, "probe failed"); return }
    var worst: Float = 0
    for p in pr { worst = max(worst, abs(p.z) * hull.lipschitz) }
    print(String(format: "        %d points; worst |envelope| × LIP %.3f mm", pts.count, worst))
    expect(worst < 0.3, "the kernel's hull is off the CPU's by \(worst) mm")
}

section("where it floats: Archimedes")

test("the solved draft balances: displaced volume × 1025 kg/m³ = the boat's mass, to 0.01%") {
    let m: Double = buoy.displacedCubicMetres * buoy.density
    let err: Double = abs(m - buoy.totalKilograms) / buoy.totalKilograms
    print(String(format: "        mass %.3f kg (hull %.3f + oars %.3f); %.5f m³ × %.0f = %.3f kg; error %.4f%%; draft %.2f mm",
                 buoy.totalKilograms, buoy.hullKilograms, buoy.oarKilograms, buoy.displacedCubicMetres, buoy.density, m, err * 100,
                 buoy.draft))
    expect(err < 1e-4)
    // More stations change nothing that matters: the slices have converged.
    let fine: Double = hull.volume(below: buoy.draft, stations: 1600)
    expect(abs(fine - buoy.displacedCubicMetres) / fine < 5e-4, "slices not converged: \(fine)")
}

test("counted on the GPU from the kernel's own distance function, the drawn hull displaces its own weight, to 1%") {
    guard let r = renderer else { expect(false, "no GPU"); return }
    guard let v = try? r.submergedVolume(hull: hull, draft: draft) else { expect(false, "count failed"); return }
    let m: Double = v * seawaterDensity
    let err: Double = (m - buoy.totalKilograms) / buoy.totalKilograms
    print(String(format: "        drawn at %.2f mm: %.2f L below the water = %.2f kg of sea water; the boat is %.2f kg (%+.2f%%)",
                 draft, v * 1000, m, buoy.totalKilograms, err * 100))
    expect(abs(err) < 0.01, "the drawn waterline displaces \(m) kg, not \(buoy.totalKilograms)")
}

test("loads sink it deeper, fresh water deeper than sea: 54 kg < + rower < 725 lb; fresh 2.5% deeper-ish") {
    print(String(format: "        %.1f mm empty, %.1f mm with one 80.7-kg rower, %.1f mm (sea) / %.1f mm (fresh) at 725 lb",
                 buoy.draft, buoy.draftWithRower, buoy.draftAtCapacitySea, buoy.draftAtCapacityFresh))
    expect(buoy.draft < buoy.draftWithRower && buoy.draftWithRower < buoy.draftAtCapacitySea)
    expect(buoy.draftAtCapacitySea < buoy.draftAtCapacityFresh)
    // Nothing floods: at full load the sheer still stands clear everywhere.
    var freeboard: Float = .infinity
    for i in 0...200 {
        let x: Float = -hull.halfLength + hull.length * Float(i) / 200
        freeboard = min(freeboard, hull.sheer(x) - buoy.draftAtCapacityFresh)
    }
    expect(freeboard > 100, "least freeboard at full load \(freeboard) mm")
}

section("the rowing gear")

test("oars by Shaw & Tenney: (span/2 + 2 in) × 25/7, to the nearest 6 in; 7/25 inboard; 5½-in blades") {
    let o: Oar = layout.oars[0]
    let spanIn: Float = layout.span / inch
    let rule: Float = (spanIn / 2 + 2) * 25 / 7
    let rounded: Float = (rule / 6).rounded() * 6
    print(String(format: "        span %.1f in → %.1f in → %.0f in; drawn %.1f in, inboard %.1f in (%.3f), blade %.2f in",
                 spanIn, rule, rounded, o.length / inch, o.inboard / inch, o.inboard / o.length, bladeWidth(forLength: o.length) / inch))
    expect(abs(o.length / inch - rounded) < 0.01)
    expect(abs(o.inboard / o.length - 7.0 / 25.0) < 1e-4)
    expect(abs(bladeWidth(forLength: o.length) - 5.5 * inch) < 0.01)
    expect(layout.oars.count == 2)
}

test("rowing geometry by Angus: oarlocks 13 in aft of the seat, 8–11.5 in above it; seat 5.5–8 in above the bilge") {
    for (i, st) in layout.stations.enumerated() {
        let lock: Oarlock = layout.oarlocks[2 * i]
        let rest: Float = lock.pivot.y - loomRadius          // where the loom's underside rests
        let aboveSeat: Float = (rest - st.seatTop) / inch / scale
        let aft: Float = (st.seatAftEdge - lock.pin.x) / inch / scale
        let bilge: Float = hull.bottom(st.seatCentreX) + hull.skin
        let seatAbove: Float = (st.seatTop - bilge) / inch / scale
        print(String(format: "        station %d: oarlock %.2f in aft of the seat, rest %.2f in above it; seat %.2f in above the bilge",
                     i + 1, aft, aboveSeat, seatAbove))
        expect(abs(aft - 13) < 0.01)
        expect(aboveSeat >= 8 && aboveSeat <= 11.5, "rest height \(aboveSeat) in")
        expect(seatAbove >= 5.5 && seatAbove <= 8, "seat \(seatAbove) in")
    }
}

test("four oarlocks stand on the gunwales, and each oar's loom rests in one: bronze under it, a gap of 0–2.5 mm") {
    guard let r = renderer else { expect(false, "no GPU"); return }
    // Each pin's top: bronze just above the block.
    var bronze: Int = 0
    let above: [SIMD3<Float>] = layout.oarlocks.map { $0.pin + SIMD3<Float>(0, oarlockCollar / 2, 0) - lift }
    if let pr = try? r.probe(above) { for p in pr where Int(p.y) == 6 && p.x < 0 { bronze += 1 } }
    // Each block sits on the sheer: its underside is below the sheer.
    var seated: Int = 0
    for l in layout.oarlocks where l.pin.y - blockHeight - 10 < hull.sheer(l.pin.x) && l.pin.y - blockHeight >= hull.sheer(l.pin.x) - 0.01 {
        seated += 1
    }
    print("        bronze at \(bronze) of \(layout.oarlocks.count) pins; \(seated) blocks on the sheer")
    expect(bronze == 4, "oarlocks missing")
    expect(seated == 4)
    // Under each oar's pivot, straight down: the oar's wood, then a gap,
    // then bronze.
    for o in layout.oars {
        var pts: [SIMD3<Float>] = []
        for i in 0..<1200 { pts.append(o.pivot - SIMD3<Float>(0, Float(i) * 0.05, 0) - lift) }
        guard let pr = try? r.probe(pts) else { expect(false, "probe failed"); return }
        var woodEnd: Float = .nan, bronzeStart: Float = .nan
        for (i, p) in pr.enumerated() {
            let d: Float = Float(i) * 0.05
            if p.x < 0 && Int(p.y) == 5 { woodEnd = d }
            if p.x < 0 && Int(p.y) == 6 && bronzeStart.isNaN { bronzeStart = d }
        }
        let gap: Float = bronzeStart - woodEnd
        print(String(format: "        oar at z = %+.0f: wood to %.2f mm below the pivot, bronze from %.2f mm: gap %.2f mm",
                     o.pivot.z, woodEnd, bronzeStart, gap))
        expect(!gap.isNaN && gap > 0 && gap <= 2.5, "the oar does not rest in an oarlock (gap \(gap))")
    }
}

test("the oars at rest: blades 10 mm clear of the water, and nothing of an oar inside the boat's wood or bronze") {
    guard let r = renderer else { expect(false, "no GPU"); return }
    var lowest: Float = .infinity
    var inside: Int = 0
    var closest: Float = .infinity
    for o in layout.oars {
        // Points on the oar's surface: round the loom and grip, over the
        // blade's faces and edges.
        var pts: [SIMD3<Float>] = []
        let n: Int = 1200
        for i in 0...n {
            let s: Float = -o.inboard + o.length * Float(i) / Float(n)
            let inBlade: Bool = s > o.outboard - bladeLength
            let rad: Float = s < -o.inboard + gripLength ? gripRadius : loomRadius
            for j in 0..<16 {
                let a: Float = Float(j) * .pi / 8
                var q: SIMD3<Float>
                if inBlade {
                    let half: Float = bladeWidth(forLength: o.length) / 2
                    q = o.pivot + o.axis * s + o.across * (half * cos(a)) + o.normal * (bladeThickness / 2 * (j % 2 == 0 ? 1 : -1))
                } else {
                    q = o.pivot + o.axis * s + o.across * (rad * cos(a)) + o.normal * (rad * sin(a))
                }
                pts.append(q - lift)
                lowest = min(lowest, q.y - draft)
            }
        }
        guard let pr = try? r.probe(pts) else { expect(false, "probe failed"); return }
        for p in pr {
            closest = min(closest, p.w)
            if p.w < -0.05 { inside += 1 }
        }
    }
    print(String(format: "        lowest point of the oars %.2f mm above the water; nearest other part %.2f mm; %d points inside",
                 lowest, closest, inside))
    expect(lowest > bladeClearance - 1 && lowest < bladeClearance + 1)
    expect(inside == 0, "\(inside) points of the oars inside the boat")
    // Resting, not hovering: the looms touch their oarlocks (to 1 mm).
    expect(closest >= -0.05 && closest < 1, "nearest \(closest) mm")
}

section("the distance function")

test("outside every surface, no distance claims more room than there is (|Δd| ≤ |Δp|, + 0.01 mm of float32 rounding), boat and envelope") {
    guard let r = renderer else { expect(false, "no GPU"); return }
    var rng = SystemRandomNumberGenerator()
    let s: Float = scale
    var a: [SIMD3<Float>] = []
    for i in 0..<240_000 {
        switch i % 4 {
        case 0:
            a.append(SIMD3<Float>(Float.random(in: -3000...3000, using: &rng) * s, Float.random(in: -300...900, using: &rng) * s,
                                  Float.random(in: -1200...1200, using: &rng) * s))
        case 1:
            // Close to the hull's skin anywhere.
            let x: Float = Float.random(in: -hull.halfLength...hull.halfLength, using: &rng)
            let y: Float = Float.random(in: hull.bottom(x)...hull.sheer(x), using: &rng)
            let w: Float = hull.halfWidth(x: x, y: y)
            let side: Float = Bool.random(using: &rng) ? 1 : -1
            a.append(SIMD3<Float>(x, y, side * (w + Float.random(in: -12...12, using: &rng))) - lift)
        case 2:
            // The ends: stem, transom, keel and skeg.
            let bow: Bool = Bool.random(using: &rng)
            let x: Float = (bow ? hull.halfLength : -hull.halfLength) + Float.random(in: -300...60, using: &rng) * s
            // Half of them on the centreline itself, where the ellipse's
            // distance is hardest for float32 (a first version failed here).
            let zc: Float = i % 8 == 2 ? Float.random(in: -0.5...0.5, using: &rng) : Float.random(in: -80...80, using: &rng) * s
            a.append(SIMD3<Float>(x, Float.random(in: -20...760, using: &rng) * s, zc) - lift)
        default:
            // Round the oarlocks and along the oars.
            if i % 8 == 3 {
                let l: Oarlock = layout.oarlocks[Int.random(in: 0..<layout.oarlocks.count, using: &rng)]
                a.append(l.pivot + SIMD3<Float>(Float.random(in: -60...60, using: &rng), Float.random(in: -80...40, using: &rng),
                                                Float.random(in: -60...60, using: &rng)) - lift)
            } else {
                let o: Oar = layout.oars[Int.random(in: 0..<layout.oars.count, using: &rng)]
                let sAlong: Float = Float.random(in: -o.inboard...o.outboard, using: &rng)
                a.append(o.pivot + o.axis * sAlong + SIMD3<Float>(Float.random(in: -90...90, using: &rng), Float.random(in: -40...40, using: &rng),
                                                                  Float.random(in: -90...90, using: &rng)) - lift)
            }
        }
    }
    var worst: Float = 0
    var worstEnv: Float = 0
    var mats: Set<Int> = []
    var worstAbs: Float = 0
    for step in [Float(0.01), 0.25, 3.0, 20.0] {
        let b: [SIMD3<Float>] = a.map { p in
            let d: SIMD3<Float> = simd_normalize(SIMD3<Float>(Float.random(in: -1...1, using: &rng), Float.random(in: -1...1, using: &rng),
                                                              Float.random(in: -1...1, using: &rng)))
            return p + d * step
        }
        guard let pa = try? r.probe(a), let pb = try? r.probe(b) else { expect(false, "probe failed"); return }
        for i in 0..<a.count {
            let dist: Float = simd_distance(a[i], b[i])
            if pa[i].x > 0 && pb[i].x > 0 {
                mats.insert(Int(pa[i].y))
                let over: Float = abs(pa[i].x - pb[i].x) - dist
                // At the 0.01 mm step, float32 rounding (~0.0002 mm) is
                // not small beside the step: allow it absolutely.
                if step < 0.1 { worstAbs = max(worstAbs, over) } else { worst = max(worst, abs(pa[i].x - pb[i].x) / dist) }
            }
            if pa[i].z > 0 && pb[i].z > 0 {
                if step < 0.1 { worstAbs = max(worstAbs, abs(pa[i].z - pb[i].z) - dist) } else {
                    worstEnv = max(worstEnv, abs(pa[i].z - pb[i].z) / dist)
                }
            }
        }
    }
    print(String(format: "        worst |Δd|/|Δp| outside: boat %.4f, envelope %.4f (a distance allows 1); at 0.01 mm, |Δd| − |Δp| ≤ %.4f mm; materials %@",
                 worst, worstEnv, worstAbs, mats.sorted().map { String($0) }.joined(separator: ",")))
    expect(worstAbs <= 0.01, "rounding beyond 0.01 mm: \(worstAbs)")
    expect(worst <= 1.01, "the boat's distance oversteps: \(worst)")
    expect(worstEnv <= 1.01, "the envelope's distance oversteps: \(worstEnv)")
    for m in [2, 3, 4, 5] { expect(mats.contains(m), "material \(m) never sampled") }
}

section("the picture")

var rendered: Bool = false
if let r = renderer, (try? r.render(camera: setup.camera, samples: 1)) != nil {
    rendered = true
    annotate(r.image, setup: setup, layout: layout, buoyancy: buoy, draft: draft)
}

test("the scale bar is true: 1 m amidships at the water, projected") {
    guard rendered, let img = renderer?.image else { expect(false, "render failed"); return }
    let a: SIMD2<Float> = project(setup.centre, width: img.width, height: img.height, cam: setup.camera)
    let b: SIMD2<Float> = project(setup.centre + setup.camera.right * mainBarMillimetres, width: img.width, height: img.height, cam: setup.camera)
    let projected: Float = simd_distance(a, b)
    let bar: ScaleBar = mainScaleBar(setup, width: img.width, height: img.height)
    var painted: Int = 0
    let yy: Int = Int(bar.y + 1.5)
    for x in max(Int(bar.x) - 4, 0)..<min(Int(bar.x + bar.pixels) + 5, img.width) {
        let p: SIMD4<UInt8> = img.rgba(x, yy)
        if Int(p.x) + Int(p.y) + Int(p.z) < 3 * 60 { painted += 1 }
    }
    let sectionPx: Float = 200 / insetMillimetresPerPixel(height: img.height)
    print(String(format: "        1 m: projected %.1f px, bar %.1f px, painted %d px; the insets: 200 mm = %.1f px, 1 m = %.1f px",
                 projected, Float(bar.pixels), painted, sectionPx, 1000 / profileMillimetresPerPixel(height: img.height)))
    expect(abs(Float(bar.pixels) - projected) < 0.5)
    expect(abs(Float(painted) - projected) <= 2.5)
}

test("the picture: the boat, the sea, the hull meeting the water at the computed waterline, the draft marked") {
    guard rendered, let img = renderer?.image else { expect(false, "render failed"); return }
    var counts: [Int: Int] = [:]
    var underCount: Int = 0
    for y in 0..<img.height {
        for x in 0..<img.width {
            let a: SIMD4<Float> = img.seen(x, y)
            counts[Int(a.x), default: 0] += 1
            if a.y > 0.5 { underCount += 1 }
        }
    }
    let total: Double = Double(img.width * img.height)
    let hullShare: Double = Double(counts[2] ?? 0) / total
    let sea: Double = Double(counts[9] ?? 0) / total
    // Halfway between the computed waterline and the sheer on the near
    // side (lower, the round bilge turns under, towards the silhouette),
    // the camera sees the hull directly, not through water.
    var dry: Int = 0, looked: Int = 0
    for i in stride(from: 20, to: 220, by: 10) {
        let x: Float = -hull.halfLength + hull.length * Float(i) / 240
        let y: Float = (draft + hull.sheer(x)) / 2
        let w: Float = hull.halfWidth(x: x, y: y)
        if w < 50 { continue }
        let q: SIMD2<Float> = project(SIMD3<Float>(x, y, w) - lift, width: img.width, height: img.height, cam: setup.camera)
        let px: Int = Int(q.x), py: Int = Int(q.y)
        guard px >= 0, px < img.width, py >= 0, py < img.height else { continue }
        looked += 1
        let a: SIMD4<Float> = img.seen(px, py)
        // The hull itself, or a fitting or oar in front of it; never water.
        if Int(a.x) >= 2 && Int(a.x) <= 6 && a.y < 0.5 { dry += 1 }
    }
    // The yellow draft mark where it is drawn.
    let dd = draftDimensionPoints(layout, draft: draft)
    let mid: SIMD2<Float> = project(dd.top * 0.7 + dd.bottom * 0.3, width: img.width, height: img.height, cam: setup.camera)
    let p: SIMD4<UInt8> = img.rgba(Int(mid.x.rounded()), Int(mid.y.rounded()))
    let yellow: Bool = p.x > 200 && p.y > 180 && p.z < 140
    print(String(format: "        hull %.1f%% of the frame, open sea %.1f%%, %d px seen through water; %d of %d points above the waterline dry; draft mark %@",
                 hullShare * 100, sea * 100, underCount, dry, looked, yellow ? "drawn" : "missing"))
    expect(hullShare > 0.06)
    expect(sea > 0.2)
    expect(looked >= 8 && dry == looked, "the hull above the waterline is not seen dry")
    expect(yellow, "no draft mark")
}

finish()
