// Tests for step 68: three ants visit a piece of peeled Golden Delicious
// apple. Step 44's tests of the walk (the gait read off the drawn feet, no
// sliding, no leg stretched, no ant touching another), step 50's tests of
// the tap and the insets (every touch-down meets the juice, nothing enters
// it, the pores, the odour, the sugar going in, the distance functions
// honest), and new ones: the glossa meets the juice while the ant drinks;
// no ant ever enters the apple; each trip starts and ends out of view; the
// sugars and smells are Golden Delicious's own; and the NEW RULE — the tap
// shows on screen: at least 10 px of movement and many pixels changed where
// the caption points.
//
// ANT_MUTANT=segments13|sliding|rewind|press|hover|tastePores|noOdour|
// pieceSize|formula|dryGlossa|stillTap breaks the scene on purpose; `make
// mutants` requires the suite to fail for each.

import Foundation
import Metal
import simd

let mutant: Mutant = {
    switch ProcessInfo.processInfo.environment["ANT_MUTANT"] {
    case "segments13": return .segments13
    case "sliding": return .sliding
    case "rewind": return .rewind
    case "press": return .press
    case "hover": return .hover
    case "tastePores": return .tastePores
    case "noOdour": return .noOdour
    case "pieceSize": return .pieceSize
    case "formula": return .formula
    case "dryGlossa": return .dryGlossa
    case "stillTap": return .stillTap
    default: return .none
    }
}()

/// Holds the device so a failure to find one is a test failure, not a crash.
final class DeviceBox {
    let device: MTLDevice? = try? findDevice()
}
let gpu = DeviceBox()

let scene: Scene? = try? Scene(mutant: mutant)
let library: MTLLibrary? = {
    guard let d = gpu.device, let s = scene else { return nil }
    return try? makeLibrary(d, s)
}()

/// Every frame of the GIF's loop; the first is ant 0's first touch-down.
let gifFrames: Int = Int((loopSeconds * 100).rounded()) / defaultDelayCentiseconds
let frames: [FrameState] = scene.map { s in (0..<gifFrames).map { s.frame(at: frameTime($0, of: gifFrames)) } } ?? []
let contactFrame: FrameState? = frames.first

/// Fine time samples across one loop, every ant (on the card or not) at each.
let fineStep: Float = 0.01
let fineTimes: [Float] = (0..<Int(loopSeconds / fineStep)).map { Float($0) * fineStep }
let fine: [[WorldAnt]] = fineTimes.map { t in trips.map { worldAnt($0, time: t, mutant: mutant) } }
/// Those samples of each ant while it is on its trip, in trip order: the
/// loop's samples rotated to begin where its trip begins, the away ones dropped.
func onTrip(_ id: Int) -> [WorldAnt] {
    let all: [WorldAnt] = fine.map { $0[id] }
    guard let start = all.indices.min(by: { all[$0].tau < all[$1].tau }) else { return [] }
    return (Array(all[start...]) + Array(all[..<start])).filter { $0.stage != .away }
}

func probe(_ pts: [SIMD3<Float>], frame given: FrameState? = nil) -> [Probe]? {
    guard let d = gpu.device, let s = scene, let l = library, let f = given ?? contactFrame else { return nil }
    return try? probeScene(pts, scene: s, frame: f, library: l, on: d)
}

func deg(_ r: Float) -> Float { r * 180 / Float.pi }

/// A foot is down when its lowest point is on the card.
func down(_ a: WorldAnt, _ j: Int) -> Bool { a.feet[j].y < 1e-6 }

/// Points on a shape's surface, for testing it against the apple and other ants.
func surfacePoints(_ s: Shape, _ n: Int) -> [SIMD3<Float>] {
    var out: [SIMD3<Float>] = []
    switch s.kind {
    case .roundCone:
        let ax: SIMD3<Float> = simd_normalize(s.b - s.a)
        let helper: SIMD3<Float> = abs(ax.y) < 0.9 ? SIMD3(0, 1, 0) : SIMD3(1, 0, 0)
        let e1: SIMD3<Float> = simd_normalize(simd_cross(ax, helper))
        let e2: SIMD3<Float> = simd_cross(ax, e1)
        for i in 0...n {
            let t: Float = Float(i) / Float(n)
            let c: SIMD3<Float> = s.a + (s.b - s.a) * t
            let r: Float = s.ra + (s.rb - s.ra) * t
            for k in 0..<8 {
                let ang: Float = Float(k) * Float.pi / 4
                out.append(c + (e1 * cos(ang) + e2 * sin(ang)) * r)
            }
        }
        out.append(s.a - ax * s.ra)
        out.append(s.b + ax * s.rb)
    case .ellipsoid, .roundBox:
        let z: SIMD3<Float> = simd_cross(s.xAxis, s.yAxis)
        for i in 0...n {
            let th: Float = Float.pi * Float(i) / Float(n)
            for k in 0..<(2 * n) {
                let ph: Float = Float.pi * Float(k) / Float(n)
                let l = SIMD3<Float>(sin(th) * cos(ph), cos(th), sin(th) * sin(ph))
                let e: SIMD3<Float> = s.kind == .ellipsoid ? s.b : s.b + SIMD3<Float>(repeating: s.ra)
                out.append(s.a + s.xAxis * (l.x * e.x) + s.yAxis * (l.y * e.y) + z * (l.z * e.z))
            }
        }
    }
    return out
}

section("the ants, against the literature")

test("the scene loads") {
    expect(scene != nil, "could not build the scene (are the Resources/ structure files there?)")
}

test("every ant, every moment: each antenna has 12 segments — scape plus an 11-segment funiculus") {
    var bad: Int = 0
    for k in stride(from: 0, to: fine.count, by: 13) {
        for a in fine[k] {
            for i in 0..<2 {
                let drawn: Int = a.shapes.filter { $0.part == .antenna && $0.index == i }.count
                if drawn != workerAntennaSegments { bad += 1 }
            }
        }
    }
    expectEqual(bad, 0)
}

test("the antennae are elbowed, and the funiculus has no club") {
    guard let f = contactFrame else { expect(false); return }
    for a in f.focus.model.antennae {
        let scape: Shape = a.segments[0]
        let first: Shape = a.segments[1]
        let bend: Float = deg(acos(simd_dot(simd_normalize(scape.b - scape.a), simd_normalize(first.b - first.a))))
        expect(bend > 35, "elbow is only \(bend)°")
        expect(abs(simd_distance(scape.a, scape.b) - scapeLength) < 1e-4, "scape length")
        for i in 2..<a.segments.count {
            let ratio: Float = a.segments[i].rb / a.segments[i - 1].rb
            expect(ratio < 1.12, "segment \(i) jumps by \(ratio)×")
        }
    }
}

test("the head is Seifert's: CS 976 µm, CL/CW 1.074, scape SL/CS 0.979") {
    expect(abs(headWidth - 0.941) < 0.002, "CW \(headWidth)")
    expect(abs(headLength - 1.011) < 0.002, "CL \(headLength)")
    expect(abs(scapeLength / cephalicSize - 0.979) < 1e-4)
    let head: Shape = bodyShapes(gasterBend: 0)[0]
    expect(abs(head.b.z * 2 - headWidth) < 1e-4, "drawn head width \(head.b.z * 2)")
}

test("six legs, every one joined to the mesosoma; one petiole between mesosoma and gaster") {
    for a in fine[0] {
        expectEqual(a.model.legs.count, 6)
        let meso: [Shape] = a.model.body.filter { $0.part == .mesosoma }
        let others: [Shape] = a.model.body.filter { $0.part == .head || $0.part == .petiole || $0.part == .gaster }
        for leg in a.model.legs {
            expect(meso.contains { $0.contains(leg.root) }, "\(leg.name) root is not inside the mesosoma")
            expect(!others.contains { $0.contains(leg.root) }, "\(leg.name) root is inside another tagma")
            expect(simd_distance(leg.shapes[0].a, leg.root) < 1e-6, "\(leg.name) coxa does not start at its root")
        }
        expectEqual(a.shapes.filter { $0.part == .leg }.count, 6 * 9)
        expectEqual(a.model.body.filter { $0.part == .petiole }.count, 1)
    }
}

test("every worker is 3.4–5.0 mm long, mandibles to gaster tip") {
    let base: Float = AntModel(body: bodyShapes(gasterBend: 0), legs: [], antennae: []).length
    for t in trips {
        let l: Float = base * t.scale
        print(String(format: "        ant %d: %.2f mm", t.id, l))
        expect(workerLengthRange.contains(l), "ant \(t.id) is \(l) mm")
    }
}

test("no leg is ever stretched or squashed: femur and tibia keep their lengths, walking and turning") {
    var worst: Float = 0
    for ants in fine {
        for a in ants where a.stage != .away {
            for (j, leg) in a.model.legs.enumerated() {
                let spec: LegSpec = legSpecs[j % 3]
                worst = max(worst, abs(simd_distance(leg.hip, leg.knee) - spec.femur))
                worst = max(worst, abs(simd_distance(leg.knee, leg.ankle) - spec.tibia))
            }
        }
    }
    print(String(format: "        worst femur/tibia length error %.5f mm", worst))
    expect(worst < 1e-3, "a leg segment is off its length by \(worst) mm: the foot is out of reach")
}

section("the apple: Golden Delicious")

/// The drawn piece's extent along one of its own axes, through its centre, by
/// bisection on the CPU distance (the one the GPU copies).
func reach(_ m: ApplePiece, along d: SIMD3<Float>) -> Float {
    var lo: Float = 0
    var hi: Float = 40
    for _ in 0..<60 {
        let mid: Float = (lo + hi) / 2
        if pieceSDF(m.centre + d * mid, m) < 0 { lo = mid } else { hi = mid }
    }
    return lo
}

test("the piece is 14 × 9 × 7 mm as drawn, fits inside a Golden Delicious over 76 mm across, and sits on the card") {
    guard let s = scene else { expect(false); return }
    let m: ApplePiece = s.piece
    let along: Float = 2 * reach(m, along: faceTangent)
    let deep: Float = 2 * reach(m, along: -faceNormal)
    let high: Float = 2 * reach(m, along: SIMD3(0, 1, 0))
    print(String(format: "        drawn %.2f × %.2f × %.2f mm (juice included)", along, deep, high))
    expect(abs(along - pieceLength) < 0.01 && abs(deep - pieceDepth) < 0.01 && abs(high - pieceHeight) < 0.01,
           "drawn \(along) × \(deep) × \(high)")
    let diagonal: Float = (pieceLength * pieceLength + pieceDepth * pieceDepth + pieceHeight * pieceHeight).squareRoot()
    expect(diagonal < fruitWidthAtLeast / 2, "a \(diagonal) mm piece cannot come from the fruit's flesh")
    // On the card: the bottom touches y = 0, no deeper.
    expect(abs(m.centre.y - m.half.z) < 1e-5, "the piece floats or sinks: bottom at \(m.centre.y - m.half.z)")
    // The drinking face's juice passes through faceBase, square to faceNormal.
    let onFace: SIMD3<Float> = faceBase + SIMD3<Float>(0, 3, 0)
    expect(abs(pieceSDF(onFace, m)) < 1e-4, "the face is not where the ants drink: \(pieceSDF(onFace, m))")
    // True scale: much taller than an ant, and longer than two.
    guard let f = contactFrame else { expect(false); return }
    let antTop: Float = f.focus.shapes.filter { $0.part != .antenna }.map { $0.extent(along: SIMD3(0, 1, 0)).hi }.max() ?? 0
    expect(high > 4 * antTop, "the piece is only \(high / antTop)× an ant's height")
}

test("the juice's sugars are the cultivar's own: fructose most abundant in both Golden Delicious sources") {
    // Ferreira et al. 2024, Golden Delicious pulp at harvest, g/kg.
    expect(gdFructose > gdSucrose && gdSucrose > gdGlucose && gdGlucose > gdSorbitol)
    expect(abs(gdFructose + gdSucrose + gdGlucose + gdSorbitol - gdTotalSugars) < 0.05, "the table does not add up")
    print(String(format: "        Ferreira 2024: fructose %.0f%% of the sugars; USDA 168202: %.0f%%",
                 100 * gdFructose / gdTotalSugars, 100 * usdaFructose / (usdaFructose + usdaSucrose + usdaGlucose)))
    expect(usdaFructose > usdaSucrose && usdaFructose > usdaGlucose)
    // And the molecule drawn going into the pore is that sugar in the form
    // it mostly takes dissolved.
    expect(fructoseBetaPyranoseShare > 50)
    expect(objectHasOdour)
}

test("the flesh is Golden Delicious's measured colour: pale, yellow-cream (L* 81.5, a* 1.6, b* 25.4)") {
    let c: SIMD3<Float> = fleshAlbedo
    expect(c.x >= c.y && c.y > c.z, "flesh \(c) is not yellow-cream")
    expect(c.z > 0.25 * c.x, "flesh \(c) is too saturated for b* 25")
    expect(abs(fleshLab.x - 81.5) < 1e-4 && abs(fleshLab.z - 25.4) < 1e-4)
}

section("the walk (step 44's gait)")

test("alternating tripod: whenever the feet are not all down, the ones down are one tripod, and they take turns") {
    var bad: Int = 0
    for id in 0..<3 {
        var sequence: [Int] = []
        var previous: Int = -1
        for a in onTrip(id) {
            let set: Set<Int> = Set((0..<6).filter { down(a, $0) })
            var now: Int = -2
            if set.count == 6 { now = -1 } else if set == tripodA { now = 0 } else if set == tripodB { now = 1 }
            if now == -2 {
                bad += 1
                if bad < 4 { expect(false, "ant \(id) has feet \(set.sorted()) down at trip time \(a.tau)") }
                previous = -1
                continue
            }
            if now >= 0 && now != previous { sequence.append(now) }
            previous = now
        }
        for k in 1..<max(sequence.count, 1) where sequence[k] == sequence[k - 1] {
            expect(false, "ant \(id): tripod \(sequence[k]) stands alone twice running (phase \(k))")
        }
        expect(sequence.count > 10, "ant \(id): only \(sequence.count) single-tripod phases")
    }
    expectEqual(bad, 0)
}

test("stance feet do not slide: a foot on the card stays within 0.1 µm of where it landed") {
    var worst: Float = 0
    var stances: Int = 0
    for id in 0..<3 {
        for j in 0..<6 {
            var landed: SIMD3<Float>? = nil
            for a in onTrip(id) {
                if down(a, j) {
                    if let l = landed { worst = max(worst, simd_distance(l, a.feet[j])) } else { landed = a.feet[j]; stances += 1 }
                } else {
                    landed = nil
                }
            }
        }
    }
    print(String(format: "        %d stances, worst drift %.6f mm", stances, worst))
    expect(worst < 1e-4, "a stance foot moved \(worst) mm across the card")
}

test("feet in stance touch the card, swing feet lift, nothing goes below it; the timetable agrees with the feet") {
    var lowest: Float = 0
    var highestSwing: Float = 0
    var mismatches: Int = 0
    for id in 0..<3 {
        for a in onTrip(id) {
            for j in 0..<6 {
                if !down(a, j) { highestSwing = max(highestSwing, a.feet[j].y) }
                if a.stance[j] != down(a, j) { mismatches += 1 }
            }
            lowest = min(lowest, a.shapes.map { $0.extent(along: SIMD3(0, 1, 0)).lo }.min() ?? 0)
        }
    }
    print(String(format: "        lowest point of any ant %.6f mm; swing lift up to %.3f mm", lowest, highestSwing))
    expect(lowest > -1e-4, "part of an ant is \(-lowest) mm below the card")
    expect(highestSwing > 0.1, "swing feet barely lift")
    expectEqual(mismatches, 0)
}

test("standing at the apple, every ant has all six feet down and does not move") {
    for id in 0..<3 {
        let standing: [WorldAnt] = onTrip(id).filter { $0.stage == .standing }
        expect(standing.count > 100, "ant \(id) hardly stands")
        guard let first = standing.first else { continue }
        for a in standing {
            expectEqual((0..<6).filter { down(a, $0) }.count, 6)
            expect(simd_distance(a.at, first.at) < 1e-5 && abs(a.yaw - first.yaw) < 1e-6, "ant \(id) moves while standing")
        }
    }
}

section("the visits")

test("no part of any ant ever enters the apple — walking, standing, turning (every shape's surface, CPU)") {
    guard let s = scene else { expect(false); return }
    var worst: Float = 9
    var worstWhere: String = ""
    for k in stride(from: 0, to: fine.count, by: 2) {
        for a in fine[k] where a.stage != .away {
            // Only shapes that could be near: bound sphere within 0.5 mm.
            for shape in a.shapes {
                let (c, r) = shape.bound
                if pieceSDF(c, s.piece) > r + 0.05 { continue }
                // The glossa's tip is meant to rest on the juice: tested on its own.
                for p in surfacePoints(shape, 6) {
                    let d: Float = pieceSDF(p, s.piece)
                    if d < worst { worst = d; worstWhere = "ant \(a.id) \(shape.part) at trip time \(a.tau)" }
                }
            }
        }
    }
    print(String(format: "        nearest ant surface to the apple %.4f mm (%@)", worst, worstWhere))
    expect(worst > -0.002, "an ant reaches \(-worst) mm into the apple: \(worstWhere)")
}

test("the ants never touch one another") {
    var overlaps: Int = 0
    var closest: Float = 9
    for k in stride(from: 0, to: fine.count, by: 5) {
        let ants: [WorldAnt] = fine[k].filter { $0.stage != .away }
        for i in 0..<ants.count {
            for j in (i + 1)..<ants.count {
                let (ci, ri) = ants[i].bound
                let (cj, rj) = ants[j].bound
                if simd_distance(ci, cj) > ri + rj { continue }
                for a in ants[i].shapes {
                    for b in ants[j].shapes {
                        if simd_distance(a.bound.centre, b.bound.centre) > a.bound.radius + b.bound.radius { continue }
                        for p in surfacePoints(a, 5) where b.contains(p) { overlaps += 1; break }
                        for p in surfacePoints(b, 5) where a.contains(p) { overlaps += 1; break }
                    }
                }
                // How close their bodies come (legs and antennae excluded).
                for a in ants[i].shapes where a.part != .leg && a.part != .antenna {
                    for b in ants[j].shapes where b.part != .leg && b.part != .antenna {
                        closest = min(closest, simd_distance(a.bound.centre, b.bound.centre) - a.bound.radius - b.bound.radius)
                    }
                }
            }
        }
    }
    print(String(format: "        closest two bodies' bounding spheres come: %.2f mm", closest))
    expectEqual(overlaps, 0)
}

/// An ant's picture footprint: every shape's projected bound and its shadow
/// on the card, grown 8 px, in a 1600 × 900 frame.
func footprint(_ a: WorldAnt) -> (lo: SIMD2<Float>, hi: SIMD2<Float>) {
    var lo = SIMD2<Float>(repeating: 1e9)
    var hi = SIMD2<Float>(repeating: -1e9)
    let pxPerMM: Float = 1600 / mainViewWidth
    for s in a.shapes {
        let (c, r) = s.bound
        let shadow: SIMD3<Float> = c - keyDirection * (c.y / keyDirection.y)
        for q in [c, shadow] {
            let p: SIMD2<Float> = projectMain(q, width: 1600, height: 900)
            lo = simd_min(lo, p - SIMD2<Float>(repeating: r * pxPerMM + 8))
            hi = simd_max(hi, p + SIMD2<Float>(repeating: r * pxPerMM + 8))
        }
    }
    return (lo, hi)
}

test("nothing appears or vanishes in view: each trip starts and ends with the ant and its shadow off the frame") {
    for trip in trips {
        for tau in [Float(0), trip.endTime] {
            let t: Float = tau + trip.offset
            let a: WorldAnt = worldAnt(trip, time: t, mutant: .none)
            let (lo, hi) = footprint(a)
            let outside: Bool = hi.x < 0 || lo.x > 1600 || hi.y < 0 || lo.y > 900
            print(String(format: "        ant %d at trip time %.2f: x %.0f…%.0f, y %.0f…%.0f", trip.id, tau, lo.x, hi.x, lo.y, hi.y))
            expect(outside, "ant \(trip.id) is in view at trip time \(tau)")
        }
        expect(trip.endTime < loopSeconds, "ant \(trip.id)'s trip is longer than the loop")
    }
}

test("each visit: taste first, then drink — three touches, the glossa out after the first and in before the ant turns") {
    for trip in trips {
        let standing: [WorldAnt] = fine.map { $0[trip.id] }.filter { $0.stage == .standing }
        var touches: Int = 0
        var was: Bool = false
        var firstTouchTau: Float = -1
        var glossaFirstTau: Float = -1
        for a in standing.sorted(by: { $0.tau < $1.tau }) {
            if a.touching && !was { touches += 1; if firstTouchTau < 0 { firstTouchTau = a.tau } }
            was = a.touching
            if a.glossa > 0.01 && glossaFirstTau < 0 { glossaFirstTau = a.tau }
        }
        expectEqual(touches, touchesPerVisit)
        expect(glossaFirstTau > firstTouchTau, "ant \(trip.id) drinks before it tastes")
        // In by the time it moves again.
        for a in fine.map({ $0[trip.id] }) where a.stage != .standing { expect(a.glossa == 0, "the glossa is out while walking") }
        expect(trip.glossaReach <= glossaMaxReach && trip.glossaReach > 0, "glossa reach \(trip.glossaReach) mm")
    }
}

section("the tap (step 30's, on the juice)")

test("one clock: the tap is 4 strokes/s slowed ×10, the walk 30 mm/s slowed ×10, and the GIF plays the loop at its length") {
    expect(realStrokeRange.contains(realStrokesPerSecond))
    expect(abs(slowdown - 10) < 1e-5, "slowed ×\(slowdown)")
    expect(abs(pictureSpeed * slowdown - realSpeed) < 1e-4)
    expect(abs(Float(gifFrames * defaultDelayCentiseconds) / 100 - loopSeconds) < 1e-4)
    // The odour's trips close with the loop.
    expectEqual(tapsPerLoop % odourSlots, 0)
}

/// Tip-to-juice distances on the GPU at one frame: for each standing ant's
/// two antennae the tip's nearest point, and the deepest any antenna point
/// reaches into the apple among `n` points packed round each tip.
func tipSurvey(_ f: FrameState, points n: Int) -> [(id: Int, lowest: Float, deepest: Float, lift: Float)]? {
    var out: [(Int, Float, Float, Float)] = []
    var rng = SystemRandomNumberGenerator()
    for a in f.ants where a.stage == .standing {
        let sc: Float = trips[a.id].scale
        for c in a.antennaTips {
            var pts: [SIMD3<Float>] = [c - faceNormal * funiculusTipRadius * sc]
            for _ in 0..<n {
                pts.append(c + SIMD3<Float>(Float.random(in: -0.16...0.16, using: &rng), Float.random(in: -0.16...0.16, using: &rng),
                                            Float.random(in: -0.16...0.16, using: &rng)))
            }
            guard let q = probe(pts, frame: f) else { return nil }
            var deepest: Float = 1
            for p in q.dropFirst() where p.ant < 0 { deepest = min(deepest, p.food) }
            out.append((a.id, q[0].food, deepest, a.lift * sc))
        }
    }
    return out
}

/// Every standing touch-down moment in the loop, finely.
let touchFrames: [FrameState] = {
    guard let s = scene else { return [] }
    var out: [FrameState] = []
    for t in stride(from: Float(0), to: loopSeconds, by: 0.25) {
        let f: FrameState = s.frame(at: t)
        if f.ants.contains(where: { $0.touching }) { out.append(f) }
    }
    return out
}()

test("at every touch-down both antenna tips meet the juice: 0 within 2 µm, on the GPU") {
    var worst: Float = 0
    var n: Int = 0
    for f in touchFrames {
        guard let r = tipSurvey(f, points: 0) else { expect(false, "probe failed"); return }
        for x in r where x.lift == 0 { worst = max(worst, abs(x.lowest)); n += 1 }
    }
    print(String(format: "        %d touching tips; worst tip-to-juice %.4f mm", n, worst))
    expect(n >= 3 * touchesPerVisit * 2, "only \(n) touching tips")
    expect(worst < 0.002, "a touching tip is \(worst) mm from the juice")
}

test("at no frame does any part of an antenna go into the juice (12,000 points round each tip)") {
    var worst: Float = 1
    for f in frames where f.ants.contains(where: { $0.stage == .standing }) {
        guard let r = tipSurvey(f, points: 6000) else { expect(false, "probe failed"); return }
        for x in r { worst = min(worst, x.deepest) }
    }
    print(String(format: "        nearest antenna point to the inside of the juice %.4f mm", worst))
    expect(worst > -0.002, "an antenna reaches \(-worst) mm into the juice")
}

test("lifted, the tips clear the juice by the lift — 0.15 mm at the top") {
    var checked: Int = 0
    var top: Float = 0
    for f in frames {
        guard f.ants.contains(where: { $0.stage == .standing && $0.lift > 0.5 * liftHeight(.none) }) else { continue }
        guard let r = tipSurvey(f, points: 1000) else { expect(false, "probe failed"); return }
        for x in r where x.lift > 0.5 * liftHeight(.none) {
            checked += 1
            top = max(top, x.lowest)
            expect(abs(x.lowest - x.lift) < 0.002, "lift \(x.lift) mm but the tip is \(x.lowest) mm off")
            expect(x.deepest > 0, "a lifted antenna point is in the juice")
        }
    }
    print(String(format: "        %d lifted tips; highest %.3f mm", checked, top))
    expect(checked > 0, "no lifted frames")
}

test("standing, every antenna reaches the tip asked for: no funiculus stretched") {
    for ants in fine {
        for a in ants where a.stage == .standing {
            for s in a.model.antennae { expect(s.reached, "ant \(a.id) cannot reach its tip at trip time \(a.tau)") }
        }
    }
}

test("drinking, the glossa rests on the juice: its tip 0 within 2 µm of it, never in it (GPU)") {
    guard let s = scene else { expect(false); return }
    var worst: Float = 0
    var deepest: Float = 1
    var n: Int = 0
    for t in stride(from: Float(0), to: loopSeconds, by: 0.2) {
        let f: FrameState = s.frame(at: t)
        let drinking: [WorldAnt] = f.ants.filter { a in
            a.stage == .standing && a.tapPhase >= glossaOut.full && a.tapPhase <= glossaOut.back
        }
        if drinking.isEmpty { continue }
        var pts: [SIMD3<Float>] = []
        for a in drinking {
            let g: Shape = a.shapes.first { $0.part == .glossa }!
            pts.append(g.b - faceNormal * g.rb)
            for k in 0..<200 {
                let u: Float = Float(k) / 200
                let c: SIMD3<Float> = g.a + (g.b - g.a) * (0.7 + 0.3 * u)
                pts.append(c + simd_normalize(SIMD3<Float>(sin(Float(k)), cos(Float(k) * 1.3), sin(Float(k) * 0.7))) * g.rb)
            }
        }
        guard let q = probe(pts, frame: f) else { expect(false, "probe failed"); return }
        var i: Int = 0
        for _ in drinking {
            worst = max(worst, abs(q[i].food))
            n += 1
            for k in 1...200 { deepest = min(deepest, q[i + k].food) }
            i += 201
        }
    }
    print(String(format: "        %d drinking moments; glossa tip to juice worst %.4f mm; deepest glossa point %.4f mm", n, worst, deepest))
    expect(n > 10, "hardly any drinking")
    expect(worst < 0.002, "the glossa is \(worst) mm off the juice while drinking")
    expect(deepest > -0.002, "the glossa goes \(-deepest) mm into the apple")
}

section("the micrometre inset: ant 0's right antenna tip")

test("the taste hair meets the juice at touch-down and leaves it when lifted — never dipping in") {
    guard let s = scene, let c = contactFrame else { expect(false); return }
    let h: Sensillum = s.hairs[0]
    let apexLow = SIMD3<Float>(h.tip.x, h.tip.y - h.tipRadius, h.tip.z)
    let beside: SIMD3<Float> = apexLow + SIMD3<Float>(0.7, 0.05, 0)
    expect(abs(c.crystal.toCrystal(apexLow).y) < 0.002, "at touch-down the apex is \(c.crystal.toCrystal(apexLow).y) µm off the juice")
    guard let r0 = probe([beside], frame: c) else { expect(false, "probe failed"); return }
    expect(r0[0].insetMaterial == 5 && r0[0].insetDistance < 0, "beside the apex at touch-down: material \(r0[0].insetMaterial)")
    var deepest: Float = 9
    var liftedChecked: Int = 0
    for f in frames {
        let y: Float = f.crystal.toCrystal(apexLow).y
        deepest = min(deepest, y)
        if f.focus.stage == .standing && f.focus.lift > 0.001 {
            liftedChecked += 1
            let sc: Float = trips[0].scale
            // The inset's juice drops as the main view's tip lifts: the same motion.
            expect(abs(y - 1000 * f.focus.lift * sc) < 3, "inset \(y) µm against a lift of \(f.focus.lift * sc) mm")
            guard let r = probe([beside], frame: f) else { expect(false, "probe failed"); return }
            if f.focus.lift > 0.01 { expect(r[0].insetMaterial != 5, "lifted, the meniscus is still at the apex") }
        }
    }
    print(String(format: "        apex's deepest point over the loop %.4f µm; %d lifted frames agree with the main view", deepest, liftedChecked))
    expect(deepest > -0.005, "the hair dips \(-deepest) µm into the juice")
    expect(liftedChecked > 5)
}

/// Survey a hair's wall on the GPU (step 50's): sample just under its outer
/// surface on a fine grid, count connected holes, and test the apex.
func poreSurvey(_ hairIndex: Int) -> (wallPores: Int, apexHole: Bool, largest: Int)? {
    guard let s = scene else { return nil }
    let h: Sensillum = s.hairs[hairIndex]
    let u: SIMD3<Float> = simd_normalize(h.tip - h.base)
    let helper: SIMD3<Float> = abs(u.y) < 0.9 ? SIMD3(0, 1, 0) : SIMD3(1, 0, 0)
    let e1: SIMD3<Float> = simd_normalize(simd_cross(u, helper))
    let e2: SIMD3<Float> = simd_cross(u, e1)
    let L: Float = h.length
    let step: Float = 0.012
    let rows: Int = Int(0.95 * L / step)
    let meanR: Float = (h.baseRadius + h.tipRadius) / 2
    let cols: Int = Int(2 * Float.pi * meanR / step)
    let depth: Float = 0.03
    var pts: [SIMD3<Float>] = []
    for i in 0..<rows {
        let t: Float = 0.05 * L + Float(i) * step
        let r: Float = h.baseRadius + (h.tipRadius - h.baseRadius) * (t / L) - depth
        for j in 0..<cols {
            let phi: Float = 2 * Float.pi * Float(j) / Float(cols)
            pts.append(h.base + u * t + (e1 * cos(phi) + e2 * sin(phi)) * r)
        }
    }
    pts.append(h.tip + u * (h.tipRadius - depth))
    guard let res = probe(pts) else { return nil }
    let dist: (Int) -> Float = { hairIndex == 0 ? res[$0].tasteHair : res[$0].smellHair }
    let hole: [Bool] = (0..<(rows * cols)).map { dist($0) > 0 }
    var parent: [Int] = Array(0..<(rows * cols))
    func find(_ x: Int) -> Int {
        var x = x
        while parent[x] != x { parent[x] = parent[parent[x]]; x = parent[x] }
        return x
    }
    func join(_ a: Int, _ b: Int) { let ra = find(a), rb = find(b); if ra != rb { parent[ra] = rb } }
    for i in 0..<rows {
        for j in 0..<cols where hole[i * cols + j] {
            let right: Int = i * cols + (j + 1) % cols
            if hole[right] { join(i * cols + j, right) }
            if i + 1 < rows && hole[(i + 1) * cols + j] { join(i * cols + j, (i + 1) * cols + j) }
        }
    }
    var sizes: [Int: Int] = [:]
    for k in 0..<(rows * cols) where hole[k] { sizes[find(k), default: 0] += 1 }
    return (sizes.count, dist(rows * cols) > 0, sizes.values.max() ?? 0)
}

test("the taste hair has exactly one pore, and it is at the tip") {
    guard let s = scene, let survey = poreSurvey(0) else { expect(false, "survey failed"); return }
    print("        taste hair: \(survey.wallPores) wall pores drawn, tip pore \(survey.apexHole)")
    expectEqual(survey.wallPores, 0)
    expect(survey.apexHole, "no opening at the apex")
    expectEqual(s.hairs[0].wallPoreCount, 0)
}

test("the smell hair has many pores in its wall and none at the tip") {
    guard let s = scene, let survey = poreSurvey(1) else { expect(false, "survey failed"); return }
    let spec: Int = s.hairs[1].wallPoreCount
    print("        smell hair: \(survey.wallPores) wall pores drawn (lattice \(spec)), tip pore \(survey.apexHole)")
    expect(survey.wallPores >= 50, "only \(survey.wallPores) wall pores")
    expect(Double(survey.wallPores) >= 0.85 * Double(spec))
    expect(!survey.apexHole, "the smell hair is open at the apex")
    let r: Float = s.hairs[1].wallPoreRadius
    expect(2 * r >= 0.07 && 2 * r <= 0.09, "pore diameter \(2 * r) µm")
}

test("a cut apple gives the smell hair something at every frame: odour molecules at or heading for wall pores") {
    guard let s = scene else { expect(false); return }
    let smell: Sensillum = s.hairs[1]
    expectEqual(s.odourPaths.count, odourSlots)
    var fewest: Int = 99
    var worstWall: Float = 9
    for f in frames {
        fewest = min(fewest, f.volatiles.filter { $0.drawn > 0.5 }.count)
        for v in f.volatiles where v.drawn > 0.01 {
            let (mouth, outward) = smell.wallPore(row: v.pore.row, column: v.pore.column)
            expect(simd_dot(v.position - mouth, outward) >= 0, "an odour molecule inside the hair")
            worstWall = min(worstWall, roundConeDistance(v.position, smell.base, smell.tip, smell.baseRadius, smell.tipRadius) - v.radius)
        }
    }
    print(String(format: "        at least %d dots shown in every frame; nearest dot surface to the wall %.3f µm", fewest, worstWall))
    expect(fewest >= 3, "a frame shows only \(fewest) odour molecules")
    expect(worstWall > -0.002)
    for p in s.odourPaths {
        guard let r = probe([p.mouth - p.outward * 0.02]) else { expect(false, "probe failed"); return }
        expect(r[0].smellHair > 0, "no pore at the mouth a molecule is heading for")
    }
}

section("the molecules")

/// Is an atom inside its molecule circle, allowing for its own drawn size?
func inCircle(_ a: Atom) -> Bool {
    let field: Float = a.view == 0 ? moleculeField : odourField
    return (a.position.x * a.position.x + a.position.y * a.position.y).squareRoot() < field / 2 + 1.2
}

func neighbours(_ m: Molecule) -> [[Int]] {
    var nb: [[Int]] = Array(repeating: [], count: m.atoms.count)
    for (a, b) in m.bonds { nb[a].append(b); nb[b].append(a) }
    return nb
}

test("fructose is C6H12O6, a six-ring pyranose: five carbons and one oxygen in the ring; C2 carries two oxygens and the CH2OH") {
    guard let s = scene else { expect(false); return }
    let m: Molecule = s.chemistry.fructose
    let f: [String: Int] = m.formula
    expect(f["C"] == 6 && f["H"] == 12 && f["O"] == 6 && f.count == 3, "fructose is \(f)")
    expectEqual(m.rings, 1)
    let nb: [[Int]] = neighbours(m)
    func count(_ i: Int, _ e: String) -> Int { nb[i].filter { m.atoms[$0].element == e }.count }
    // The anomeric carbon of a ketose: bonded to two oxygens and two carbons.
    let anomeric: [Int] = (0..<m.atoms.count).filter { m.atoms[$0].element == "C" && count($0, "O") == 2 }
    expectEqual(anomeric.count, 1)
    if let c2 = anomeric.first { expectEqual(count(c2, "C"), 2) }
    // Ring size from the ring oxygen: shortest cycle through it over heavy atoms.
    let ringO: [Int] = (0..<m.atoms.count).filter { m.atoms[$0].element == "O" && count($0, "C") == 2 }
    expectEqual(ringO.count, 1)
    if let o = ringO.first {
        let a: Int = nb[o][0], b: Int = nb[o][1]
        var dist: [Int: Int] = [a: 0]
        var queue: [Int] = [a]
        while !queue.isEmpty {
            let x: Int = queue.removeFirst()
            for y in nb[x] where y != o && m.atoms[y].element != "H" && dist[y] == nil { dist[y] = dist[x]! + 1; queue.append(y) }
        }
        expectEqual((dist[b] ?? 0) + 2, 6)
    }
}

test("the odorants: 2-hexenal C6H10O with C=C next to C=O; butyl acetate C6H12O2, an ester") {
    guard let s = scene else { expect(false); return }
    let h: Molecule = s.chemistry.hexenal
    let hf: [String: Int] = h.formula
    expect(hf["C"] == 6 && hf["H"] == 10 && hf["O"] == 1, "2-hexenal is \(hf)")
    let doubles: [(Int, Int)] = h.bonds.indices.filter { h.orders[$0] == 2 }.map { h.bonds[$0] }
    expectEqual(doubles.count, 2)
    let cc: [(Int, Int)] = doubles.filter { h.atoms[$0.0].element == "C" && h.atoms[$0.1].element == "C" }
    let co: [(Int, Int)] = doubles.filter { h.atoms[$0.0].element == "O" || h.atoms[$0.1].element == "O" }
    expect(cc.count == 1 && co.count == 1)
    if let x = cc.first, let y = co.first {
        let carbonylC: Int = h.atoms[y.0].element == "C" ? y.0 : y.1
        let nb: [[Int]] = neighbours(h)
        // Conjugated: one carbon of the C=C is bonded to the carbonyl carbon.
        expect(nb[carbonylC].contains(x.0) || nb[carbonylC].contains(x.1), "the C=C is not next to the C=O")
    }
    let b: Molecule = s.chemistry.butylAcetate
    let bf: [String: Int] = b.formula
    expect(bf["C"] == 6 && bf["H"] == 12 && bf["O"] == 2, "butyl acetate is \(bf)")
    // An ester: an oxygen bonded to two carbons, one of which also carries a C=O.
    let nb: [[Int]] = neighbours(b)
    let ether: [Int] = (0..<b.atoms.count).filter { b.atoms[$0].element == "O" && nb[$0].filter { b.atoms[$0].element == "C" }.count == 2 }
    expectEqual(ether.count, 1)
    for a in h.atoms + b.atoms { expectEqual(a.view, 1) }
}

test("every atom has its valence (bond orders summed): C 4, O 2, H 1 — every frame") {
    var bad: Int = 0
    for f in frames {
        let v: [Int] = f.molecule.valences
        for (i, a) in f.molecule.atoms.enumerated() {
            let want: Int = a.element == "C" ? 4 : (a.element == "O" ? 2 : 1)
            if v[i] != want { bad += 1 }
        }
    }
    expectEqual(bad, 0)
}

test("each touch of ant 0 takes one fructose a place into the pore; between touches the fructose is still") {
    guard let s = scene else { expect(false); return }
    // Over a loop: exactly one place per touch.
    let gained: Float = s.sugarProgress(loopSeconds - 1e-3) - s.sugarProgress(0)
    expect(abs(gained - Float(touchesPerVisit)) < 0.01, "a loop takes \(gained) fructose in, not \(touchesPerVisit)")
    // Moving only while ant 0's right antenna is on the juice.
    var bad: Int = 0
    for i in 1..<fine.count {
        let a: WorldAnt = fine[i][0]
        let du: Float = s.sugarProgress(fineTimes[i]) - s.sugarProgress(fineTimes[i - 1])
        if du > 1e-6 && !(a.touching || fine[i - 1][0].touching) { bad += 1 }
        if du < -1e-6 { bad += 1 }
    }
    expectEqual(bad, 0)
    let h: Sensillum = s.hairs[0]
    let into: SIMD3<Float> = simd_normalize(h.base - h.tip)
    let screen = SIMD2<Float>(simd_dot(into, insetCamera.right), simd_dot(into, insetCamera.up))
    expect(simd_dot(simd_normalize(screen), s.drift) > 0.999, "the fructose does not drift towards the pore")
}

test("every frame: each fructose and odorant moves rigidly; no two come within 2 Å; the odorants stay in their circle") {
    guard let s = scene, let c = contactFrame else { expect(false); return }
    func distances(_ atoms: [Atom]) -> [Float] {
        var out: [Float] = []
        for i in 0..<atoms.count { for j in (i + 1)..<atoms.count { out.append(simd_distance(atoms[i].position, atoms[j].position)) } }
        return out
    }
    let still = simd_quatf(angle: 0, axis: SIMD3<Float>(0, 1, 0))
    let reference: [Float] = distances(sugarUnit(chemistry: s.chemistry, turn: still, centre: .zero).sugar.atoms)
    let odorantReference: [Float] = distances(c.apple.odorants.atoms)
    var worstRigid: Float = 0
    var closest: Float = 1e9
    var worstOut: Float = -9
    let hexReference: [Float] = distances(c.apple.odorants.atoms.filter { $0.position.y > 0.5 * (hexenalCentre.y + butylAcetateCentre.y) })
    for f in frames {
        for u in f.apple.units {
            for (x, y) in zip(distances(u.sugar.atoms), reference) { worstRigid = max(worstRigid, abs(x - y)) }
        }
        // Each odorant alone keeps its shape (the two turn about their own centres).
        let n: Int = s.chemistry.hexenal.atoms.count
        let hexNow: [Atom] = Array(f.apple.odorants.atoms[0..<n])
        _ = hexReference
        for (x, y) in zip(distances(hexNow), distances(Array(c.apple.odorants.atoms[0..<n]))) { worstRigid = max(worstRigid, abs(x - y)) }
        let butNow: [Atom] = Array(f.apple.odorants.atoms[n...])
        for (x, y) in zip(distances(butNow), distances(Array(c.apple.odorants.atoms[n...]))) { worstRigid = max(worstRigid, abs(x - y)) }
        let parts: [Molecule] = f.apple.sugars
        for i in 0..<parts.count {
            for j in (i + 1)..<parts.count {
                for a in parts[i].atoms { for b in parts[j].atoms { closest = min(closest, simd_distance(a.position, b.position)) } }
            }
        }
        _ = odorantReference
        for a in f.apple.odorants.atoms { worstOut = max(worstOut, simd_length(SIMD2<Float>(a.position.x, a.position.y)) + 0.4 - odourField / 2) }
    }
    print(String(format: "        rigid to %.1e Å; closest atoms of two fructoses %.2f Å; odorants' reach past their circle %.2f Å",
                 worstRigid, closest, worstOut))
    expect(worstRigid < 1e-3)
    expect(closest > 2.0, "two fructoses come within \(closest) Å")
    expect(worstOut < 0, "an odorant leaves its circle")
}

section("the loop")

test("forward: every ant only walks on, time L is time 0, and the step across the seam is an ordinary step") {
    guard let s = scene, let c = contactFrame else { expect(false); return }
    // Gait distance never decreases within a trip.
    for trip in trips {
        var last: Float = -1
        for t in fineTimes {
            let tau: Float = trip.tripTime(t, mutant: mutant)
            if tau > trip.endTime { last = -1; continue }
            let d: Float = trip.gaitDistance(tau)
            if last >= 0 && d < last - 1e-4 && tau > 0.02 { expect(false, "ant \(trip.id) walks backwards at \(t)"); break }
            last = d
        }
    }
    let end: FrameState = s.frame(at: loopSeconds)
    expectEqual(end.ants.count, c.ants.count)
    var worst: Float = 0
    for (a, b) in zip(end.ants, c.ants) { for (x, y) in zip(a.shapes, b.shapes) { worst = max(worst, simd_distance(x.a, y.a)) } }
    for (a, b) in zip(end.molecule.atoms, c.molecule.atoms) { worst = max(worst, simd_distance(a.position, b.position) / 1000) }
    expect(worst < 1e-3, "time L differs from time 0 by \(worst)")
    // The seam step: every ant's shapes, the fructose, the odour.
    // What is drawn depends on the fructose's progress past a whole place and
    // the odour's past a whole trip: compare those, across the seam too.
    func wrapped(_ x: Float) -> Float { abs(x - x.rounded()) }
    func step(_ a: FrameState, _ b: FrameState) -> (ants: Float, sugar: Float, odour: Float) {
        var m: Float = 0
        for x in a.ants { if let y = b.ants.first(where: { $0.id == x.id }) { m = max(m, simd_distance(x.at, y.at)) } }
        let o: Float = (b.odourProgress - a.odourProgress) / Float(odourSlots)
        return (m, wrapped(b.progress - a.progress), wrapped(o) * Float(odourSlots))
    }
    var biggest: (ants: Float, sugar: Float, odour: Float) = (0, 0, 0)
    for i in 1..<frames.count {
        let x = step(frames[i - 1], frames[i])
        biggest = (max(biggest.ants, x.ants), max(biggest.sugar, x.sugar), max(biggest.odour, x.odour))
    }
    let seamEnd: FrameState = s.frame(at: loopSeconds)
    let last: FrameState = frames[frames.count - 1]
    let seam = step(last, seamEnd)
    print(String(format: "        seam step: ants %.4f mm, sugar %.3f, odour %.3f; largest inside %.4f, %.3f, %.3f",
                 seam.ants, seam.sugar, seam.odour, biggest.ants, biggest.sugar, biggest.odour))
    expect(seam.ants <= biggest.ants * 1.01 + 1e-5 && seam.sugar <= biggest.sugar * 1.01 + 1e-5 && seam.odour <= biggest.odour * 1.01 + 1e-5,
           "the seam jumps")
    // Progress only forward, frame to frame; over the loop the fructose gains
    // exactly one place per touch of ant 0 and the odour a whole number of
    // trips, so the drop at the seam changes nothing drawn.
    var lastSugar: Float = frames[0].progress
    var lastOdour: Float = frames[0].odourProgress
    for f in frames.dropFirst() {
        expect(f.progress >= lastSugar - 1e-5, "the fructose goes back at \(f.time)")
        expect(f.odourProgress >= lastOdour - 1e-5, "the odour goes back at \(f.time)")
        lastSugar = f.progress
        lastOdour = f.odourProgress
    }
    let justBefore: FrameState = s.frame(at: loopSeconds - 1e-4)
    expect(abs(justBefore.progress - Float(touchesPerVisit)) < 0.01, "the loop takes \(justBefore.progress) fructose in")
    expect(abs(justBefore.odourProgress - Float(tapsPerLoop)) < 0.01 && tapsPerLoop % odourSlots == 0)
}

test("nothing pops: 1 ms apart, every ant, atom and odour dot moves only a little — at every frame and every relabel") {
    guard let s = scene else { expect(false); return }
    var times: [Float] = (0..<frames.count).map { frameTime($0, of: frames.count) }
    // Around each ant's touches (the fructose slots relabel), and trip ends.
    for trip in trips {
        for k in 0...touchesPerVisit {
            for extra in [Float(0), contactFraction] {
                let tau: Float = trip.holdStart + (Float(k) + extra - standStartPhase) * tapSeconds
                let t: Float = tau + trip.offset
                times += [t - 2e-3, t - 1e-3, t, t + 1e-3]
            }
        }
    }
    times.append(loopSeconds - 1e-3)
    var pops: Int = 0
    var worstAnt: Float = 0
    var worstAtom: Float = 0
    for t in times {
        let a: FrameState = s.frame(at: t)
        let b: FrameState = s.frame(at: t + 1e-3)
        for x in a.ants {
            guard let y = b.ants.first(where: { $0.id == x.id }) else { continue }
            var m: Float = 0
            for (p, q) in zip(x.shapes, y.shapes) { m = max(m, simd_distance(p.a, q.a), simd_distance(p.b, q.b)) }
            worstAnt = max(worstAnt, m)
            if m > 0.02 { pops += 1; if pops < 5 { print("        ant \(x.id) jumps \(m) mm at \(t)") } }
        }
        // Any ant present in one and not the other must be out of view.
        let ida: Set<Int> = Set(a.ants.map { $0.id })
        let idb: Set<Int> = Set(b.ants.map { $0.id })
        for id in ida.symmetricDifference(idb) {
            let w: WorldAnt = worldAnt(trips[id], time: t, mutant: mutant)
            let (lo, hi) = footprint(w)
            if !(hi.x < 0 || lo.x > 1600 || hi.y < 0 || lo.y > 900) { pops += 1; if pops < 5 { print("        ant \(id) appears/vanishes in view at \(t)") } }
        }
        for (x, y) in zip(a.molecule.atoms, b.molecule.atoms) where inCircle(x) && inCircle(y) {
            let d: Float = simd_distance(x.position, y.position)
            // Slots relabel: an atom may instead match the one ahead of it.
            if d > 0.2 {
                let near: Float = b.molecule.atoms.filter { $0.element == x.element && $0.view == x.view }
                    .map { simd_distance($0.position, x.position) }.min() ?? 9
                if near > 0.2 { pops += 1; if pops < 5 { print("        an atom jumps \(near) Å at \(t)") } } else { worstAtom = max(worstAtom, near) }
            } else { worstAtom = max(worstAtom, d) }
        }
        for (x, y) in zip(a.volatiles, b.volatiles) where x.drawn > 0.01 && y.drawn > 0.01 {
            if simd_distance(x.position, y.position) > 0.02 || abs(x.radius - y.radius) > 0.005 { pops += 1; if pops < 5 { print("        a dot jumps at \(t)") } }
        }
    }
    print(String(format: "        %d moments: largest ant move in 1 ms %.4f mm, atom %.3f Å; %d pops", times.count, worstAnt, worstAtom, pops))
    expectEqual(pops, 0)
}

section("the distance functions are distances")

func worstOverReport(box lo: SIMD3<Float>, _ hi: SIMD3<Float>, step: Float, inset: Bool, count: Int = 120_000,
                     points given: [SIMD3<Float>] = [], frame: FrameState? = nil) -> [Int: Float] {
    var rng = SystemRandomNumberGenerator()
    var a: [SIMD3<Float>] = []
    var b: [SIMD3<Float>] = []
    for k in 0..<(given.isEmpty ? count : given.count) {
        let p: SIMD3<Float> = given.isEmpty
            ? SIMD3<Float>(Float.random(in: lo.x...hi.x, using: &rng), Float.random(in: lo.y...hi.y, using: &rng),
                           Float.random(in: lo.z...hi.z, using: &rng))
            : given[k]
        let d: SIMD3<Float> = simd_normalize(SIMD3<Float>(Float.random(in: -1...1, using: &rng), Float.random(in: -1...1, using: &rng),
                                                          Float.random(in: -1...1, using: &rng)))
        a.append(p)
        b.append(p + d * step)
    }
    guard let pa = probe(a, frame: frame), let pb = probe(b, frame: frame) else { return [:] }
    var worst: [Int: Float] = [:]
    for i in 0..<a.count {
        let (da, db, ma, mb): (Float, Float, Int, Int) = inset
            ? (pa[i].insetDistance, pb[i].insetDistance, pa[i].insetMaterial, pb[i].insetMaterial)
            : (pa[i].mainDistance, pb[i].mainDistance, pa[i].mainMaterial, pb[i].mainMaterial)
        guard ma == mb, da > 0, db > 0 else { continue }
        worst[ma] = max(worst[ma] ?? 0, abs(da - db) / step)
    }
    return worst
}

test("main view: outside every surface, no distance claims more room than the ray allows") {
    var w: [Int: Float] = [:]
    for f in [frames[0], frames[frames.count / 3], frames[2 * frames.count / 3]] {
        for a in f.ants {
            for (m, v) in worstOverReport(box: a.at - SIMD3<Float>(2.6, 0, 2.6), a.at + SIMD3<Float>(2.6, 1.6, 2.6), step: 0.006,
                                          inset: false, count: 60_000, frame: f) {
                w[m] = max(w[m] ?? 0, v)
            }
        }
    }
    for (m, v) in worstOverReport(box: SIMD3(-10, 0, -10), SIMD3(10, 9, 10), step: 0.02, inset: false) { w[m] = max(w[m] ?? 0, v) }
    print(String(format: "        worst over-report: card %.2f, ant %.2f, apple %.2f; the ray allows %.2f", w[1] ?? 0, w[2] ?? 0, w[3] ?? 0, 1 / stepScale))
    expect(w.count == 3, "not every material was sampled: \(w)")
    for (m, v) in w { expect(v * stepScale <= 1.0, "material \(m) oversteps: \(v)") }
}

test("inset: the hairs, the dome, the juice and its meniscus, the odour dots are honest outside too") {
    var all: [Int: Float] = worstOverReport(box: SIMD3(-4, -0.5, -12), SIMD3(8, 12, 4), step: 0.02, inset: true)
    for (m, v) in worstOverReport(box: SIMD3(-1.5, -0.2, -1.5), SIMD3(1.5, 1.5, 1.5), step: 0.005, inset: true) { all[m] = max(all[m] ?? 0, v) }
    if frames.count > 37 {
        let f: FrameState = frames[37]
        var pts: [SIMD3<Float>] = []
        var rng = SystemRandomNumberGenerator()
        for v in f.volatiles where v.drawn > 0.01 {
            for _ in 0..<20_000 {
                pts.append(v.position + SIMD3<Float>(Float.random(in: -0.4...0.4, using: &rng), Float.random(in: -0.4...0.4, using: &rng),
                                                     Float.random(in: -0.4...0.4, using: &rng)))
            }
        }
        for (m, v) in worstOverReport(box: .zero, .zero, step: 0.004, inset: true, points: pts, frame: f) { all[m] = max(all[m] ?? 0, v) }
    }
    print(String(format: "        worst over-report: juice %.2f, cuticle %.2f, taste hair %.2f, smell hair %.2f, meniscus %.2f, odour %.2f",
                 all[1] ?? 0, all[2] ?? 0, all[3] ?? 0, all[4] ?? 0, all[5] ?? 0, all[6] ?? 0))
    expect(all.count >= 5, "not every material was sampled: \(all)")
    for (m, v) in all { expect(v * stepScale <= 1.0, "material \(m) oversteps: \(v)") }
}

section("the picture")

let render: AntImage? = {
    guard let d = gpu.device, let s = scene, let f = contactFrame else { return nil }
    return try? renderAnt(width: 800, height: 450, samples: 1, scene: s, frame: f, on: d).image
}()

test("the touch is visible: antenna and apple both within 3 px of ant 0's projected contact") {
    guard let img = render, let s = scene else { expect(false, "render failed"); return }
    let p: SIMD2<Float> = projectMain(s.contact.point, width: img.width, height: img.height)
    var sawAnt = false
    var sawApple = false
    for dy in -3...3 {
        for dx in -3...3 {
            let v: SIMD4<Float> = img.seen(Int(p.x) + dx, Int(p.y) + dy)
            if v.x == 1 && v.y == 2 { sawAnt = true }
            if v.x == 1 && v.y == 3 { sawApple = true }
        }
    }
    expect(sawAnt && sawApple, "near the contact: antenna \(sawAnt), apple \(sawApple)")
}

test("the inset shows both hairs, the meniscus and odour dots; both molecule insets show C, O and H") {
    guard let img = render else { expect(false, "render failed"); return }
    var seen: Set<Int> = []
    var taste: Set<Int> = []
    var smell: Set<Int> = []
    for y in 0..<img.height {
        for x in 0..<img.width {
            let v: SIMD4<Float> = img.seen(x, y)
            if v.x == 2 { seen.insert(Int(v.y)) }
            if v.x == 3 && v.y >= 1 && v.y <= 3 { taste.insert(Int(v.y)) }
            if v.x == 4 && v.y >= 1 && v.y <= 3 { smell.insert(Int(v.y)) }
        }
    }
    expect(seen.isSuperset(of: [3, 4, 5, 6]), "inset materials seen: \(seen.sorted())")
    expectEqual(taste, [1, 2, 3])
    expectEqual(smell, [1, 2, 3])
}

test("each scale bar matches its own view: 1 mm, 5 µm, 0.5 nm") {
    let w: Int = 1600
    let h: Int = 900
    let p = SIMD3<Float>(1, 0.3, 0.5)
    let dx: Float = simd_distance(projectMain(p, width: w, height: h), projectMain(p + mainCamera.right, width: w, height: h))
    expect(abs(dx - 1 / mainMillimetresPerPixel(width: w)) < 0.01, "1 mm is \(dx) px")
    let q = SIMD3<Float>(0, 1, 0)
    let du: Float = simd_distance(projectInset(q, height: h), projectInset(q + insetCamera.right * 5, height: h))
    expect(abs(du - 5 / insetMicrometresPerPixel(height: h)) < 0.01, "5 µm is \(du) px")
    let hf = Float(h)
    expect(simd_distance(odourCentre, insetCentre) * hf > (odourRadius + insetRadius) * hf + 8)
    expect(simd_distance(odourCentre, moleculeCentre) * hf > (odourRadius + moleculeRadius) * hf + 8)
    expect(simd_distance(moleculeCentre, insetCentre) * hf > (moleculeRadius + insetRadius) * hf + 8)
}

// The new rule: an animation must be visibly animated. The tap is the
// motion the insets and the caption talk about, so it is measured on screen:
// render ant 0 touching and at the top of the next lift, at the GIF's size,
// and count what changed round its right antenna tip.
test("VISIBLE: ant 0's tap moves its tip at least 10 px on screen and changes hundreds of pixels round it") {
    guard let d = gpu.device, let s = scene, let touch = contactFrame else { expect(false); return }
    // Ant 0 touches down at t = 0 (phase 1.0); its lift tops out at phase 1.7.
    let topPhase: Float = (contactFraction + 1) / 2
    let up: FrameState = s.frame(at: topPhase * tapSeconds)
    guard up.focus.stage == .standing, up.focus.lift > 0 || s.mutant == .stillTap else { expect(false, "no lifted frame"); return }
    let w: Int = 1600
    let h: Int = 900
    guard let r = try? Renderer(device: d, scene: s, width: w, height: h) else { expect(false, "renderer"); return }
    _ = try? r.render(touch, samples: 1)
    let n: Int = w * h * 4
    let a: [UInt8] = Array(UnsafeBufferPointer(start: r.pixels.contents().assumingMemoryBound(to: UInt8.self), count: n))
    _ = try? r.render(up, samples: 1)
    let b: [UInt8] = Array(UnsafeBufferPointer(start: r.pixels.contents().assumingMemoryBound(to: UInt8.self), count: n))
    let t0: SIMD2<Float> = projectMain(touch.focus.antennaTips[0], width: w, height: h)
    let t1: SIMD2<Float> = projectMain(up.focus.antennaTips[0], width: w, height: h)
    let moved: Float = simd_distance(t0, t1)
    // Pixels round the tip (a 60 px box about both positions) that changed
    // by more than 40 levels in any channel.
    let c: SIMD2<Float> = (t0 + t1) / 2
    var changed: Int = 0
    for y in max(Int(c.y) - 60, 0)..<min(Int(c.y) + 60, h) {
        for x in max(Int(c.x) - 60, 0)..<min(Int(c.x) + 60, w) {
            let i: Int = (y * w + x) * 4
            var m: Int = 0
            for k in 0..<3 { m = max(m, abs(Int(a[i + k]) - Int(b[i + k]))) }
            if m > 40 { changed += 1 }
        }
    }
    print(String(format: "        tip moved %.1f px (lift %.3f mm); %d pixels changed by > 40 levels within 60 px of it", moved,
                 up.focus.lift * trips[0].scale, changed))
    expect(moved >= 10, "the tap moves the tip only \(moved) px: invisible")
    expect(changed >= 150, "only \(changed) pixels change where the tap is")
}

test("text and insets stay off the ants: no ant or its shadow passes under a label, a plate or an inset, at any frame") {
    let w: Int = 1600
    let h: Int = 900
    let rects: [PixelRect] = [titleBlock(height: h), fructoseBlock(height: h), odourNameBlock(height: h), scaleBarBlock(height: h)]
    let hf: Float = Float(h)
    let circles: [(SIMD2<Float>, Float)] = [(insetCentre * hf, insetRadius * hf + 8), (moleculeCentre * hf, moleculeRadius * hf + 8),
                                           (odourCentre * hf, odourRadius * hf + 8)]
    let pxPerMM: Float = Float(w) / mainViewWidth
    var hits: Int = 0
    var nearest: Float = 1e9
    var tally: [String: Int] = [:]
    for f in frames {
        for a in f.ants {
            for s in a.shapes {
                let (c, r) = s.bound
                let shadow: SIMD3<Float> = c - keyDirection * (c.y / keyDirection.y)
                for q in [c, shadow] {
                    let p: SIMD2<Float> = projectMain(q, width: w, height: h)
                    let rp: Float = r * pxPerMM + (q == c ? 0 : 6)
                    for b in rects {
                        let dx: Float = max(Float(b.x0) - p.x, 0, p.x - Float(b.x1))
                        let dy: Float = max(Float(b.y0) - p.y, 0, p.y - Float(b.y1))
                        let d: Float = (dx * dx + dy * dy).squareRoot() - rp
                        nearest = min(nearest, d)
                        if d < 0 { hits += 1; tally["label \(b.x0),\(b.y0) ant \(a.id)", default: 0] += 1 }
                    }
                    for (cc, rr) in circles {
                        let d: Float = simd_distance(p, cc) - rr - rp
                        nearest = min(nearest, d)
                        if d < 0 { hits += 1; tally["inset \(Int(cc.x)) ant \(a.id)", default: 0] += 1 }
                    }
                }
            }
        }
    }
    print(String(format: "        nearest any ant (or its shadow) comes to a label or inset: %.1f px", nearest))
    for (k, v) in tally.sorted(by: { $0.key < $1.key }) { print("        \(k): \(v)") }
    expectEqual(hits, 0)
}

test("lifted, ant 0's inset shows no juice at all") {
    guard let d = gpu.device, let s = scene else { expect(false); return }
    let top: FrameState = s.frame(at: ((contactFraction + 1) / 2) * tapSeconds)
    guard top.focus.lift > 0.1 else { expect(s.mutant == .stillTap, "not lifted"); return }
    guard let img = try? renderAnt(width: 800, height: 450, samples: 1, scene: s, frame: top, on: d).image else {
        expect(false, "render failed"); return
    }
    var juice: Int = 0
    for y in 0..<450 {
        for x in 0..<800 where img.seen(x, y).x == 2 {
            let m: Int = Int(img.seen(x, y).y)
            if m == 1 || m == 5 { juice += 1 }
        }
    }
    expectEqual(juice, 0)
}

finish()
