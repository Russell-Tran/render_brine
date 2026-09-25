// The right panel: a polarimeter, which is how the chemistry on the left was
// found in the first place. Nobody watched Asp23 do anything. They watched a
// number go negative and reasoned backwards.
//
// Sodium light goes through a polariser, so it is vibrating in one plane. It
// crosses a cell of sugar solution, and the plane turns. An analyser at the far
// end reads how far. That is the whole instrument.
//
// WHAT IS DRAWN, AND WHAT IS COMPUTED
//
// The beam is drawn as a ribbon: a row of cross-bars down the axis of the cell,
// each one the direction the light is vibrating at that point. The ribbon
// twists along the cell, because that is what optical rotation IS — not a
// property of the exit, but a turn accumulating through the solution. The twist
// from the entry bar to the exit bar is the polarimeter's reading, ONE TO ONE:
// no gain, no exaggeration. 34.6 degrees of rotation is 34.6 degrees of twist.
//
// The molecules in the cell are drawn too, out in the annulus around the beam,
// as many as the panel can hold. The reading is computed by COUNTING THOSE
// DRAWN MOLECULES — sucrose glyphs joined, split ones apart — and putting the
// count through Chemistry.swift. The two panels cannot disagree, because there
// is only one number and the picture is where it comes from.
//
// One thing this cannot claim. The cell holds a sample standing in for the
// bulk, and the single active site on the left is one of something like 10^17.
// The rate at which the flask converts is therefore a display mapping, marked
// MODEL in the table: what the tests enforce is that the reading always matches
// the molecules drawn, and that the molecules drawn always match how far the
// catalytic cycle has got — never that a real site would empty a real flask in
// four turnovers.

import Foundation
import simd

// The bench, in the same units the molecule panel uses so one camera works for
// both. The cell is 120 units long and stands for the standard 200 mm tube.
let cellHalfLength: Float = 60
let cellRadius: Float = 21
let beamRadius: Float = 10.5
let ribbonBars = 33
let polariserX: Float = -74
let analyserX: Float = 74
let lampX: Float = -97

/// What the panel drew, so the tests can read it back.
struct PolarimeterReadout {
    var composition: Composition
    var fill: Double
    var rotationDegrees: Double
    /// Indices into the primitive array of the ribbon's cross-bars, in order
    /// along the beam. The twist measured off these is the render's own answer.
    var barIndices: [Int]
    var sucroseGlyphs: Int
    var splitGlyphs: Int
}

/// A deterministic scatter. The same molecule slot is in the same place in
/// every frame, so nothing in the cell jitters and the GIF has less to store.
private func slotPosition(_ i: Int, count: Int) -> SIMD3<Float> {
    // A golden-angle spiral down the tube, in the annulus outside the beam.
    let golden: Float = 2.39996323
    let t: Float = (Float(i) + 0.5) / Float(count)
    let angle: Float = Float(i) * golden
    let ringSpread: Float = 0.62 + 0.30 * Float((i * 7919) % 97) / 97.0
    let r: Float = beamRadius + 4.2 + (cellRadius - beamRadius - 6.4) * ringSpread
    let x: Float = -cellHalfLength + 6 + (2 * cellHalfLength - 12) * t
    let y: Float = r * cos(angle)
    let z: Float = r * sin(angle)
    return SIMD3(x, y, z)
}

/// A tilt for each glyph, so the cell does not look like a row of dumbbells all
/// facing the same way.
private func slotAxis(_ i: Int) -> SIMD3<Float> {
    let a: Float = Float((i * 5087) % 360) * .pi / 180
    let b: Float = Float((i * 2731) % 180) * .pi / 180
    return simd_normalize(SIMD3(cos(a) * sin(b), sin(a) * sin(b), cos(b) + 0.35))
}

/// How many of the cell's `count` slots have been split, from the catalytic
/// progress on the other panel. This is the single place the two panels are
/// joined, and it is a rounding of one number.
func convertedSlots(progress: Double, count: Int) -> Int {
    let clamped: Double = min(max(progress, 0), 1)
    return Int((clamped * Double(count)).rounded())
}

/// Builds the right panel. `progress` is how far the catalytic cycle on the
/// left has got, 0 to 1; `fill` is how much of the cell holds solution.
func polarimeterScene(progress: Double, fill: Double, molecules: Int,
                      lightColour: SIMD3<Float>) -> (prims: [GPUPrim], readout: PolarimeterReadout) {
    var prims: [GPUPrim] = []
    let split: Int = convertedSlots(progress: progress, count: molecules)
    let clampedFill: Double = min(max(fill, 0), 1)
    let liquidEnd: Float = -cellHalfLength + 2 * cellHalfLength * Float(clampedFill)

    // --- the molecules, first, because the reading is counted off them ------
    // Sucrose is one glyph: a glucose bead and a fructose bead with a bond
    // between them. Split, the bond is gone and the two drift apart. Counting
    // bonds is counting sucrose. Whatever is left in the cell — while it is
    // draining, there is less of it — is what the reading is computed from.
    var sucroseGlyphs = 0, splitGlyphs = 0
    for i in 0..<molecules {
        let centre = slotPosition(i, count: molecules)
        if centre.x > liquidEnd { continue }
        let axis = slotAxis(i)
        let isSplit: Bool = i < split
        let gap: Float = isSplit ? 4.6 : 2.15
        let g = centre + axis * gap
        let f = centre - axis * gap
        prims.append(.sphere(g, 1.7, glucoseColour))
        prims.append(.sphere(f, 1.8, fructoseColour))
        if isSplit {
            splitGlyphs += 1
        } else {
            prims.append(.stick(g, f, 0.65, stickColour * 0.8))
            sucroseGlyphs += 1
        }
    }
    let composition = Composition(sucrose: sucroseGlyphs, glucose: splitGlyphs,
                                  fructose: splitGlyphs)
    let rotation: Double = observedRotation(composition, fill: clampedFill)

    // --- the cell, as a cage rather than glass -----------------------------
    // A ray tracer of opaque solids has no transparency, so a solid tube would
    // hide everything the panel is for. Rails and rings read as a tube and let
    // the light through, which is the honest trade.
    let glass = SIMD3<Float>(0.52, 0.60, 0.66)
    let rails = 10
    for k in 0..<rails {
        let a: Float = Float(k) / Float(rails) * 2 * .pi
        let y: Float = cellRadius * cos(a), z: Float = cellRadius * sin(a)
        prims.append(.stick(SIMD3(-cellHalfLength, y, z), SIMD3(cellHalfLength, y, z), 0.5, glass))
    }
    for end in [-cellHalfLength, cellHalfLength, -cellHalfLength / 3, cellHalfLength / 3] {
        let segments = 44
        for k in 0..<segments {
            let a0: Float = Float(k) / Float(segments) * 2 * .pi
            let a1: Float = Float(k + 1) / Float(segments) * 2 * .pi
            let p0 = SIMD3<Float>(end, cellRadius * cos(a0), cellRadius * sin(a0))
            let p1 = SIMD3<Float>(end, cellRadius * cos(a1), cellRadius * sin(a1))
            prims.append(.stick(p0, p1, 0.55, glass))
        }
    }

    // --- the lamp, the polariser, the analyser ------------------------------
    prims.append(.sphere(SIMD3(lampX, 0, 0), 7.5, lightColour))
    func disc(_ x: Float, barAngle: Float, colour: SIMD3<Float>) {
        let segments = 40
        let r: Float = cellRadius + 2.5
        for k in 0..<segments {
            let a0: Float = Float(k) / Float(segments) * 2 * .pi
            let a1: Float = Float(k + 1) / Float(segments) * 2 * .pi
            prims.append(.stick(SIMD3(x, r * cos(a0), r * sin(a0)),
                                SIMD3(x, r * cos(a1), r * sin(a1)), 0.9, colour))
        }
        let c: Float = cos(barAngle), s: Float = sin(barAngle)
        prims.append(.stick(SIMD3(x, r * c, r * s), SIMD3(x, -r * c, -r * s), 1.3, colour))
    }
    let frame = SIMD3<Float>(0.66, 0.70, 0.76)
    disc(polariserX, barAngle: 0, colour: frame)
    disc(analyserX, barAngle: Float(rotation) * .pi / 180, colour: SIMD3(0.92, 0.86, 0.62))

    // The beam outside the cell: still plane-polarised, not yet turned.
    let entryColour: SIMD3<Float> = lightColour * 0.85
    prims.append(.stick(SIMD3(lampX + 7, 0, 0), SIMD3(polariserX, 0, 0), 2.2, entryColour * 0.7))

    // --- the ribbon ---------------------------------------------------------
    // Each bar is the plane of vibration where it stands. The bars only turn
    // inside the solution: over the empty part of the cell the light is not
    // rotated at all, which is why a half-filled tube reads half as much.
    var barIndices: [Int] = []
    var tips: [(SIMD3<Float>, SIMD3<Float>)] = []
    for k in 0..<ribbonBars {
        let f: Float = Float(k) / Float(ribbonBars - 1)
        let x: Float = -cellHalfLength - 14 + (2 * cellHalfLength + 28) * f
        // How much solution the light has already been through at this point.
        let travelled: Float = min(max((min(x, liquidEnd) + cellHalfLength)
                                       / max(2 * cellHalfLength, 1e-3), 0), 1)
        // `decouple` drives the beam from something other than the molecules
        // counted, which is the failure this whole render is built to make
        // impossible: two panels that look right and mean different things.
        let gain: Float = mutation == .decouple ? 1.3 : 1.0
        let turned: Float = Float(rotation) * gain * (clampedFill > 1e-6
                                                      ? travelled / Float(clampedFill) : 0)
        let a: Float = turned * .pi / 180
        let dir = SIMD3<Float>(0, cos(a), sin(a))
        let half: Float = beamRadius
        let axisPoint = SIMD3<Float>(x, 0, 0)
        barIndices.append(prims.count)
        prims.append(.stick(axisPoint - dir * half, axisPoint + dir * half, 0.8, lightColour))
        tips.append((axisPoint - dir * half, axisPoint + dir * half))
    }
    // The ribbon's two edges. Without them the bars read as a picket fence and
    // a 35-degree twist spread over thirty-odd of them is invisible; joined up,
    // the same 35 degrees is a visible spiral, and it is still 35 degrees.
    let edge: SIMD3<Float> = lightColour * 0.75
    for k in 1..<tips.count {
        prims.append(.stick(tips[k - 1].0, tips[k].0, 0.75, edge))
        prims.append(.stick(tips[k - 1].1, tips[k].1, 0.75, edge))
    }

    let readout = PolarimeterReadout(composition: composition, fill: clampedFill,
                                     rotationDegrees: rotation, barIndices: barIndices,
                                     sucroseGlyphs: sucroseGlyphs, splitGlyphs: splitGlyphs)
    return (prims, readout)
}

/// The twist actually drawn, measured back off the ribbon's cross-bars: the
/// signed angle from the first bar to the last, about the beam axis, summed bar
/// by bar so half-turns cannot be lost. This is the render's own answer to what
/// the polarimeter reads, and it is compared against the chemistry's.
func measuredTwist(_ prims: [GPUPrim], barIndices: [Int]) -> Double {
    var total: Double = 0
    var previous: SIMD3<Float>?
    for index in barIndices {
        let p = prims[index]
        let a = SIMD3(p.a.x, p.a.y, p.a.z)
        let b = SIMD3(p.b.x, p.b.y, p.b.z)
        let d: SIMD3<Float> = simd_normalize(b - a)
        if let q = previous {
            // Both lie in the y-z plane; the signed angle between them about x.
            let cosine: Float = q.y * d.y + q.z * d.z
            let sine: Float = q.y * d.z - q.z * d.y
            total += Double(atan2(sine, cosine)) * 180 / .pi
        }
        previous = d
    }
    return total
}
