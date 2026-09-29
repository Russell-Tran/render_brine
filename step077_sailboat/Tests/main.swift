// Tests for step 77, the ILCA 7 (Laser). The boat's size is measured off
// the kernel's own distance function and held to the class figures; the
// waterline is held to Archimedes — the water displaced at the drawn
// waterline must weigh what the boat and its load weigh — on the CPU's
// columns and, independently, by counting the GPU kernel's inside points;
// the simplified hull is held to Day & Nixon's measured hydrostatics; the
// distance function to the definition of a distance; the picture to what
// it must show.
//
// BOAT_MUTANT=floatsWrong|wrongSize|noKeel breaks the step on purpose;
// `make mutants` requires the suite to fail for each.

import CoreGraphics
import Foundation
import Metal
import simd

let mutant: Mutant = activeMutant
if mutant != .none { print("MUTANT: \(mutant.rawValue)") }

let flo: Flotation = archimedes(mutant: mutant)
let drawn: Double = drawnDraught(flo, mutant)
let scale: Double = boatScale(mutant)
let setup: StillSetup = stillSetup(mutant)
let renderer: BoatRenderer? = {
    guard let d = try? findDevice() else { return nil }
    return try? BoatRenderer(device: d, width: 1920, height: 1080, setup: setup, draught: drawn, mutant: mutant)
}()

/// World point from a boat-frame point (unit scale).
func world(_ p: SIMD3<Double>) -> SIMD3<Float> {
    let q: SIMD3<Double> = p * scale - SIMD3<Double>(0, drawn, 0)
    return SIMD3<Float>(Float(q.x), Float(q.y), Float(q.z))
}

/// Sphere-trace many rays at once against one channel of the probe
/// (0 whole boat, 2 underwater body, 3 hull only); the distance travelled.
func trace(_ starts: [SIMD3<Float>], _ dir: SIMD3<Float>, channel: Int, maxT: Float) -> [Float] {
    guard let r = renderer else { return [] }
    var t: [Float] = Array(repeating: 0, count: starts.count)
    var done: [Bool] = Array(repeating: false, count: starts.count)
    for _ in 0..<400 {
        let pts: [SIMD3<Float>] = (0..<starts.count).map { starts[$0] + dir * t[$0] }
        guard let pr = try? r.probe(pts) else { return [] }
        var all: Bool = true
        for i in 0..<starts.count where !done[i] {
            let d: Float = pr[i][channel]
            if d < 0.00005 { done[i] = true } else { t[i] += d * 0.95; all = false }
            if t[i] > maxT { done[i] = true; t[i] = .infinity }
        }
        if all { break }
    }
    return t
}

/// How far the chosen channel reaches along an axis: trace a grid of lines
/// in from `back` metres out, take the nearest hit.
func reach(along axis: SIMD3<Float>, across u: SIMD3<Float>, _ v: SIMD3<Float>, uRange: ClosedRange<Float>,
           vRange: ClosedRange<Float>, centre: SIMD3<Float>, channel: Int, back: Float = 4) -> Float {
    let n: Int = 48
    var starts: [SIMD3<Float>] = []
    for i in 0..<n {
        for j in 0..<n {
            let fi: Float = Float(i) / Float(n - 1)
            let fj: Float = Float(j) / Float(n - 1)
            let a: Float = uRange.lowerBound + (uRange.upperBound - uRange.lowerBound) * fi
            let b: Float = vRange.lowerBound + (vRange.upperBound - vRange.lowerBound) * fj
            let s: SIMD3<Float> = centre + axis * back + u * a + v * b
            starts.append(s)
        }
    }
    let t: [Float] = trace(starts, -axis, channel: channel, maxT: 2 * back)
    return back - (t.min() ?? .infinity)
}

section("the ILCA 7, against the class figures")

test("the hull, measured off the kernel: \(classLength) m long, \(classBeam) m beam (Wikipedia), to 5 mm") {
    let X = SIMD3<Float>(1, 0, 0), Y = SIMD3<Float>(0, 1, 0), Z = SIMD3<Float>(0, 0, 1)
    // Lines along the boat at the centreline, over the hull's height; lines
    // across it near the widest station, over the top centimetre below the
    // deck edge (where the flared sides are widest).
    let s: Float = Float(scale)
    let c: SIMD3<Float> = world(SIMD3<Double>(0, hullShape.depth / 2, 0))
    let yr: ClosedRange<Float> = (-0.2 * s)...(0.2 * s)
    let zr: ClosedRange<Float> = (-0.01 * s)...(0.01 * s)
    let fore: Float = reach(along: X, across: Y, Z, uRange: yr, vRange: zr, centre: c, channel: 3)
    let aft: Float = reach(along: -X, across: Y, Z, uRange: yr, vRange: zr, centre: c, channel: 3)
    let cw: SIMD3<Float> = world(SIMD3<Double>(hullShape.widestX, hullShape.depth, 0))
    let xr: ClosedRange<Float> = (-0.3 * s)...(0.3 * s)
    let dr: ClosedRange<Float> = (-0.01 * s)...(-0.0002 * s)
    let port: Float = reach(along: -Z, across: X, Y, uRange: xr, vRange: dr, centre: cw, channel: 3)
    let stbd: Float = reach(along: Z, across: X, Y, uRange: xr, vRange: dr, centre: cw, channel: 3)
    let length: Float = fore + aft
    let beam: Float = port + stbd
    print(String(format: "        drawn %.4f m long, %.4f m beam; class %.2f × %.2f m", length, beam, classLength, classBeam))
    expect(abs(Double(length) - classLength) < 0.005, "length \(length)")
    expect(abs(Double(beam) - classBeam) < 0.005, "beam \(beam)")
}

test("the mast stands \(String(format: "%.3f", mastLength)) m from its foot: 2.865 + 3.600 − 0.305 m (forum figures)") {
    let top: SIMD3<Float> = world(SIMD3<Double>(mastX, mastFootY + mastLength + 1, 0))
    let t: [Float] = trace([top], SIMD3<Float>(0, -1, 0), channel: 0, maxT: 3)
    guard let hit = t.first, hit.isFinite else { expect(false, "mast top not found"); return }
    let topY: Double = Double(top.y - hit)
    let foot: Double = mastFootY * scale - drawn
    let len: Double = topY - foot
    print(String(format: "        mast top %.4f m above its foot (world y %.3f)", len, topY))
    expect(abs(len - mastLength) < 0.003, "mast \(len) m")
}

test("the mainsail: luff 5.13, foot 2.74, leech 5.57 m; area within 2% of the class's 7.06 m² (its roach not drawn); on the kernel's sail") {
    let s: Sail = sailGeometry()
    let luff: Double = simd_distance(s.tack, s.head)
    let foot: Double = simd_distance(s.tack, s.clew)
    let leech: Double = simd_distance(s.head, s.clew)
    print(String(format: "        luff %.4f, foot %.4f, leech %.4f m; area %.3f m² (class %.2f)", luff, foot, leech, s.area, sailArea))
    expect(abs(luff - sailLuff) < 0.001 && abs(foot - sailFoot) < 0.001 && abs(leech - sailLeech) < 0.001)
    expect(abs(s.area - sailArea) / sailArea < 0.02, "area \(s.area)")
    guard let r = renderer, let pr = try? r.probe([world(s.tack), world(s.head), world(s.clew)]) else { expect(false, "probe"); return }
    for p in pr { expect(p.x < 0.001, "a sail corner is \(p.x) m off the drawn boat") }
}

var boardTipWorld: Double = .nan
test("the centreboard, measured off the kernel: 341 mm chord, reaching 680 mm below the hull (forum figures), to 2 mm") {
    // The hull's bottom under the board, hull only; the board's tip; its chord.
    let below: SIMD3<Float> = world(SIMD3<Double>(boardX, -1.5, 0))
    let hullUp: [Float] = trace([below], SIMD3<Float>(0, 1, 0), channel: 3, maxT: 3)
    let bodyUp: [Float] = trace([below], SIMD3<Float>(0, 1, 0), channel: 2, maxT: 3)
    guard let hh = hullUp.first, let bb = bodyUp.first, hh.isFinite else { expect(false, "no hull above"); return }
    let depth: Double = Double(hh - bb) / scale
    boardTipWorld = Double(below.y + bb)
    let mid: SIMD3<Float> = world(SIMD3<Double>(boardX, keelHeight(boardX) - boardDepthBelowHull / 2, 0))
    let fwd: [Float] = trace([mid + SIMD3<Float>(1, 0, 0)], SIMD3<Float>(-1, 0, 0), channel: 2, maxT: 2)
    let aft: [Float] = trace([mid - SIMD3<Float>(1, 0, 0)], SIMD3<Float>(1, 0, 0), channel: 2, maxT: 2)
    let chord: Double = (2 - Double(fwd.first ?? 0) - Double(aft.first ?? 0)) / scale
    print(String(format: "        board reaches %.4f m below the hull; chord %.4f m", depth, chord))
    expect(abs(depth - boardDepthBelowHull) < 0.002, "depth \(depth)")
    expect(abs(chord - boardChord) < 0.002, "chord \(chord)")
}

test("draught with the board down: the class's 0.787 m (Wikipedia), to 3%") {
    let d: Double = -boardTipWorld
    print(String(format: "        waterline to board tip %.4f m", d))
    expect(d.isFinite && abs(d - classDraft) / classDraft < 0.03, "draught \(d)")
}

section("Archimedes: the waterline")

test("the water displaced at the drawn waterline weighs what the boat does: ρV = M to 0.1% (CPU columns)") {
    let v: Double = flo.body.volume(below: drawn)
    let m: Double = v * waterDensity
    print(String(format: "        %.1f kg/m³ × %.5f m³ = %.2f kg; the boat and load %.2f kg; drawn draught %.1f mm",
                 waterDensity, v, m, totalMass, drawn * 1000))
    expect(abs(m - totalMass) / totalMass < 0.001, "displaces \(m) kg")
}

test("counted independently on the GPU: the kernel's own body below the drawn water, ρV = M to 1%") {
    guard let r = renderer else { expect(false, "no GPU"); return }
    let dx: Float = 0.01, dz: Float = 0.005, dy: Float = 0.004
    let x0: Float = -2.65 * Float(scale), x1: Float = 2.2 * Float(scale)
    let z0: Float = -0.72 * Float(scale), y0: Float = -1.0 * Float(scale)
    let nx: Int = Int(((x1 - x0) / dx).rounded()), nz: Int = Int((-2 * z0 / dz).rounded()), ny: Int = Int((-y0 / dy).rounded())
    var inside: Int = 0
    for i in 0..<nx {
        var pts: [SIMD3<Float>] = []
        pts.reserveCapacity(nz * ny)
        let x: Float = x0 + (Float(i) + 0.5) * dx
        for j in 0..<nz {
            let z: Float = z0 + (Float(j) + 0.5) * dz
            for k in 0..<ny {
                let y: Float = y0 + (Float(k) + 0.5) * dy
                pts.append(SIMD3<Float>(x, y, z))
            }
        }
        guard let pr = try? r.probe(pts) else { expect(false, "probe"); return }
        for p in pr where p.z < 0 { inside += 1 }
    }
    let cell: Double = Double(dx) * Double(dz) * Double(dy)
    let v: Double = Double(inside) * cell
    let m: Double = v * waterDensity
    print(String(format: "        %d cells inside → %.5f m³ → %.2f kg (want %.2f)", inside, v, m, totalMass))
    expect(abs(m - totalMass) / totalMass < 0.01, "displaces \(m) kg")
}

test("the column grid has converged: a grid twice as coarse floats her within 0.5 mm") {
    let coarse: Flotation = archimedes(mutant: mutant, dx: 0.01, dz: 0.005)
    print(String(format: "        draught %.3f mm at 5 × 2.5 mm columns, %.3f mm at 10 × 5 mm", flo.draught * 1000, coarse.draught * 1000))
    expect(abs(coarse.draught - flo.draught) < 0.0005)
}

test("the CPU's distance function is the kernel's (the one Archimedes used is the one drawn), to 0.2 mm") {
    guard let r = renderer else { expect(false, "no GPU"); return }
    var rng = SystemRandomNumberGenerator()
    var pts: [SIMD3<Float>] = []
    var cpu: [Double] = []
    for _ in 0..<60_000 {
        let p = SIMD3<Double>(Double.random(in: -2.7...2.4, using: &rng), Double.random(in: -0.9...0.5, using: &rng),
                              Double.random(in: -0.8...0.8, using: &rng))
        let q: SIMD3<Double> = p * scale
        pts.append(SIMD3<Float>(Float(q.x), Float(q.y - drawn), Float(q.z)))
        cpu.append(underwaterBody(q, mutant: mutant))
    }
    guard let pr = try? r.probe(pts) else { expect(false, "probe"); return }
    var worst: Double = 0
    for i in 0..<pts.count { worst = max(worst, abs(Double(pr[i].z) - cpu[i])) }
    print(String(format: "        worst |GPU − CPU| %.2e m over %d points", worst, pts.count))
    expect(worst < 0.0002)
}

test("the simplified hull against Day & Nixon's tank-test hull at their 160 kg (Table 1): volume, LWL, BWL, waterplane, draught") {
    // Hull only (appendages neglected, as they did), at their draught 0.094 m.
    let t: Double = 0.094 * scale
    let hullOnly: ColumnVolume = columnVolume({ hullDistance($0 / scale) * scale }, x0: hullShape.stern * scale, x1: hullShape.bow * scale,
                                              halfZ: hullShape.halfBeam * scale, y0: -0.01, y1: 0.2 * scale, dx: 0.005, dz: 0.0025)
    let v: Double = hullOnly.volume(below: t)
    // The waterplane at that draught.
    var area: Double = 0
    var xmin: Double = .infinity, xmax: Double = -.infinity, bmax: Double = 0
    let dx: Double = 0.004, dz: Double = 0.002
    var x: Double = hullShape.stern * scale + dx / 2
    while x < hullShape.bow * scale {
        var n: Int = 0
        var z: Double = -hullShape.halfBeam * scale + dz / 2
        while z < hullShape.halfBeam * scale {
            if hullDistance(SIMD3<Double>(x, t - 1e-6, z) / scale) < 0 { n += 1 }
            z += dz
        }
        if n > 0 { xmin = min(xmin, x - dx / 2); xmax = max(xmax, x + dx / 2) }
        area += Double(n) * dx * dz
        bmax = max(bmax, Double(n) * dz)
        x += dx
    }
    let lwl: Double = xmax - xmin
    print(String(format: "        V %.4f m³ (0.160), LWL %.3f m (3.791), BWL %.3f m (1.103), Awp %.3f m² (2.859); our draught %.1f mm (94, board out)",
                 v, lwl, bmax, area, flo.draught * 1000))
    expect(abs(v - 0.160) / 0.160 < 0.04, "volume \(v)")
    expect(abs(lwl - 3.791) / 3.791 < 0.015, "LWL \(lwl)")
    expect(abs(bmax - 1.103) / 1.103 < 0.04, "BWL \(bmax)")
    expect(abs(area - 2.859) / 2.859 < 0.04, "Awp \(area)")
    expect(abs(flo.draught - 0.094) / 0.094 < 0.08, "draught \(flo.draught)")
}

test("level trim as Day & Nixon define it: the transom just touching the water (its lowest edge within 10 mm of the waterline)") {
    let transom: Double = keelHeight(hullShape.stern) * scale
    print(String(format: "        transom's lowest edge %.1f mm above the drawn waterline", (transom - drawn) * 1000))
    expect(abs(transom - drawn) < 0.010)
}

section("the distance function")

test("outside every surface, no distance claims more room than the ray allows") {
    guard let r = renderer else { expect(false, "no GPU"); return }
    var rng = SystemRandomNumberGenerator()
    let s: Float = Float(scale)
    let sail: Sail = sailGeometry()
    var a: [SIMD3<Float>] = []
    for i in 0..<300_000 {
        var p: SIMD3<Double>
        switch i % 5 {
        case 0: p = SIMD3<Double>(Double.random(in: -2.4...2.4, using: &rng), Double.random(in: -0.1...0.5, using: &rng),
                                  Double.random(in: -0.8...0.8, using: &rng))
        case 1: p = SIMD3<Double>(Double.random(in: 1.8...2.3, using: &rng), Double.random(in: -0.05...0.45, using: &rng),
                                  Double.random(in: -0.3...0.3, using: &rng))
        case 2: p = SIMD3<Double>(boardX + Double.random(in: -0.3...0.3, using: &rng), Double.random(in: -0.8...0.5, using: &rng),
                                  Double.random(in: -0.08...0.08, using: &rng))
        case 3: p = SIMD3<Double>(rudderX + Double.random(in: -0.3...0.3, using: &rng), Double.random(in: -0.6...0.6, using: &rng),
                                  Double.random(in: -0.08...0.08, using: &rng))
        default:
            let u: Double = Double.random(in: 0...1, using: &rng), v: Double = Double.random(in: 0...1, using: &rng)
            let w: Double = u + v > 1 ? 0 : 1
            let q: SIMD3<Double> = sail.tack + (sail.head - sail.tack) * (w > 0 ? u : 1 - u) + (sail.clew - sail.tack) * (w > 0 ? v : 1 - v)
            p = q + SIMD3<Double>(Double.random(in: -0.1...0.1, using: &rng), Double.random(in: -0.1...0.1, using: &rng),
                                  Double.random(in: -0.05...0.05, using: &rng))
        }
        a.append(world(p))
    }
    var worst: Float = 0
    var mats: Set<Int> = []
    for step in [Float(0.0005), 0.005, 0.05] {
        let b: [SIMD3<Float>] = a.map { p in
            let d: SIMD3<Float> = simd_normalize(SIMD3<Float>(Float.random(in: -1...1, using: &rng), Float.random(in: -1...1, using: &rng),
                                                              Float.random(in: -1...1, using: &rng)))
            return p + d * step * s
        }
        guard let pa = try? r.probe(a), let pb = try? r.probe(b) else { expect(false, "probe failed"); return }
        for i in 0..<a.count {
            guard pa[i].x > 0, pb[i].x > 0 else { continue }
            mats.insert(Int(pa[i].y))
            let dist: Float = simd_distance(a[i], b[i])
            worst = max(worst, abs(pa[i].x - pb[i].x) / dist)
            if pa[i].z > 0 && pb[i].z > 0 { worst = max(worst, abs(pa[i].z - pb[i].z) / dist) }
        }
    }
    print(String(format: "        worst |Δd|/|Δp| outside %.3f (the ray allows %.3f); materials seen %@", worst, 1 / stepScale,
                 mats.sorted().map { String($0) }.joined(separator: ",")))
    expect(worst * stepScale <= 1, "oversteps: \(worst)")
    for m in [1, 2, 3, 5] { expect(mats.contains(m), "material \(m) never sampled") }
}

test("the hull's bound is a lower bound: marching the hull's distance never passes through its surface") {
    // Along random lines through the hull's box, the distance never claims
    // more room than the next sign change (found by fine stepping).
    var rng = SystemRandomNumberGenerator()
    var worst: Double = 0
    for _ in 0..<3000 {
        let o = SIMD3<Double>(Double.random(in: -2.3...2.3, using: &rng), Double.random(in: -0.1...0.45, using: &rng),
                              Double.random(in: -0.9...0.9, using: &rng))
        let d: SIMD3<Double> = simd_normalize(SIMD3<Double>(Double.random(in: -1...1, using: &rng), Double.random(in: -1...1, using: &rng),
                                                            Double.random(in: -1...1, using: &rng)))
        let d0: Double = hullDistance(o)
        guard d0 > 0 else { continue }
        var t: Double = 0
        while t < 1.0 {
            t += 0.0005
            if hullDistance(o + d * t) < 0 { break }
        }
        if t < 1.0 { worst = max(worst, d0 / t) }
    }
    print(String(format: "        worst claimed distance ÷ distance to the surface along a line: %.3f", worst))
    expect(worst <= 1.0 + 1e-3)
}

section("the picture")

var rendered: Bool = false
if let r = renderer, (try? r.render(camera: setup.camera, samples: 1)) != nil {
    rendered = true
    annotate(r.image, setup: setup, report: Report(flotation: flo, drawn: drawn), mutant: mutant)
}

test("the scale bar is true: 1 m at the boat's centre plane, projected") {
    guard rendered, let img = renderer?.image else { expect(false, "render failed"); return }
    let a: SIMD2<Float> = project(setup.centre, width: img.width, height: img.height, cam: setup.camera)
    let b: SIMD2<Float> = project(setup.centre + setup.camera.right * mainBarMetres, width: img.width, height: img.height, cam: setup.camera)
    let projected: Float = simd_distance(a, b)
    let bar: ScaleBar = mainScaleBar(setup, width: img.width, height: img.height)
    var painted: Int = 0
    let yy: Int = Int(bar.y + 1.5)
    for x in max(Int(bar.x) - 4, 0)..<min(Int(bar.x + bar.pixels) + 5, img.width) {
        let p: SIMD4<UInt8> = img.rgba(x, yy)
        if Int(p.x) + Int(p.y) + Int(p.z) < 3 * 60 { painted += 1 }
    }
    let insetWant: Float = insetBarMetres * insetPixelsPerMetre(height: img.height)
    let insetBar: ScaleBar = insetScaleBar(height: img.height)
    print(String(format: "        1 m: projected %.1f px, bar %.1f px, painted %d px; inset 100 mm: %.1f px", projected, Float(bar.pixels),
                 painted, Float(insetBar.pixels)))
    expect(abs(Float(bar.pixels) - projected) < 0.5)
    expect(abs(Float(painted) - projected) <= 2.5)
    expect(abs(Float(insetBar.pixels) - insetWant) < 0.01)
}

test("the picture shows the boat, its sail, and — through the cut-away water — its underwater hull and centreboard") {
    guard rendered, let img = renderer?.image else { expect(false, "render failed"); return }
    var counts: [Int: Int] = [:]
    var hullUnder: Int = 0, boardUnder: Int = 0, waterSeen: Int = 0, overInset: Int = 0
    let rect: CGRect = insetRect(height: img.height)
    for y in 0..<img.height {
        for x in 0..<img.width {
            let a: SIMD4<Float> = img.seen(x, y)
            let m: Int = Int(a.x)
            counts[m, default: 0] += 1
            if a.w > 0 { waterSeen += 1 }
            if m == 1 && a.z < -0.005 && a.w >= 2 { hullUnder += 1 }
            if m == 4 && a.z < -0.005 && a.w >= 2 { boardUnder += 1 }
            if m > 0 && rect.contains(CGPoint(x: x, y: y)) { overInset += 1 }
        }
    }
    print(String(format: "        hull %d px (%d under water), sail %d px, centreboard under water %d px, water %d px, under the inset %d px",
                 counts[1] ?? 0, hullUnder, counts[3] ?? 0, boardUnder, waterSeen, overInset))
    expect((counts[1] ?? 0) > 20_000, "the hull is too small")
    expect((counts[3] ?? 0) > 50_000, "the sail is not seen")
    expect(hullUnder > 800, "the underwater hull is not seen through the water")
    expect(boardUnder > 3_000, "the centreboard is not seen under the water")
    expect(waterSeen > 100_000, "no water")
    expect(overInset == 0, "the inset covers the boat")
}

test("the inset's draught, measured off its pixels, is the draught Archimedes gave (to 2.5 px), and the section its label is measured from sits there") {
    guard rendered, let img = renderer?.image else { expect(false, "render failed"); return }
    let r: CGRect = insetRect(height: img.height)
    let ppm: Float = insetPixelsPerMetre(height: img.height)
    let waterRow: Float = Float(r.maxY) - insetBelow * ppm
    // Down the inset's centre column (the draught arrow is drawn there, so
    // look 12 px beside it): the lowest dark outline pixel.
    let cx: Int = Int(r.midX) + 12
    var lowest: Int = -1
    for y in Int(waterRow)..<Int(r.maxY) - 30 {
        let p: SIMD4<UInt8> = img.rgba(cx, y)
        if Int(p.x) + Int(p.y) + Int(p.z) < 3 * 110 { lowest = y }
    }
    let measured: Float = (Float(lowest) + 1 - waterRow) / ppm
    let bottom: Double = sectionBottom(draught: drawn, mutant: mutant)
    print(String(format: "        inset: hull bottom %.1f px below the waterline = %.1f mm; section bottom %.1f mm; Archimedes %.1f mm",
                 Float(lowest) + 1 - waterRow, measured * 1000, -bottom * 1000, flo.draught * 1000))
    expect(lowest > 0, "no hull bottom found in the inset")
    expect(abs(Double(measured) - flo.draught) * Double(ppm) < 2.5, "inset bottom \(measured) m")
    expect(abs(-bottom - flo.draught) < 0.0015, "the section's bottom is not at Archimedes' draught")
}

finish()
