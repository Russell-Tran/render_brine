// Step 65's timeline: step 64's wear growing from nothing, frame by frame,
// with the camera never moving.
//
//   1.0 s   the healthy arch — step 20's teeth — at year 0
//  12.0 s   the wear grows, 0 → 2.25 mm, 0 → 50 years, straight through
//   2.0 s   holding at year 50, step 64's teeth
//   1.5 s   a dissolve back to the first frame, captioned as a restart
//
// The wear only ever deepens. The loop closes by dissolving the picture back
// to year 0, as steps 21 and 29 close theirs, never by un-wearing the teeth.
// The years on the caption are depth ÷ rate (Wear.swift), not typed.

import Foundation
import Metal
import simd

/// Frames a second, and the GIF delay that gives it.
let framesPerSecond: Int = 10
let frameDelayCentiseconds: Int = 100 / framesPerSecond

let healthySeconds: Float = 1.0
let wearSeconds: Float = 12.0
let wornSeconds: Float = 2.0
let fadeSeconds: Float = 1.5

func frames(_ seconds: Float) -> Int { Int((seconds * Float(framesPerSecond)).rounded()) }
let healthyFrames: Int = frames(healthySeconds)
let wearFrames: Int = frames(wearSeconds)
let wornFrames: Int = frames(wornSeconds)
let fadeFrames: Int = frames(fadeSeconds)
let frameCount: Int = healthyFrames + wearFrames + wornFrames + fadeFrames
let loopSeconds: Float = Float(frameCount) / Float(framesPerSecond)

/// How newly worn surface is marked: tinted where the original surface stood
/// less than this far above it, i.e. ground away within the last few years.
/// The years are chosen (MODEL, a teaching aid, not anatomy); the band is the
/// depth those years take at the rate.
let glowYears: Float = 5
let glowBand: Float = glowYears * bruxistWearRate
let glowStrength: Float = 0.6

/// What to break, for the mutation check. Each one must be caught by a test.
enum PlanMutant: String {
    case none
    case unwear          // the wear runs backwards in the second half of its span
    case invisible       // the wear a tenth as deep: too small to see
    case cameraMoves     // the camera drifts during the wear
    case noDentin        // passed through to the kernel: dentine drawn as enamel
}

/// One frame: how deep the wear is, the year the caption says, how far into
/// the dissolve back to year 0, and where the camera stands.
struct FramePlan {
    var depth: Float
    var years: Float
    var dissolve: Float
    var camera: Camera
    var restarting: Bool { dissolve > 0 }
}

func smooth(_ x: Float) -> Float {
    let t: Float = min(max(x, 0), 1)
    return t * t * (3 - 2 * t)
}

func plan(_ f: Int, mutant: PlanMutant = .none) -> FramePlan {
    var depth: Float = 0
    var dissolve: Float = 0
    let wearStart: Int = healthyFrames
    let wearEnd: Int = healthyFrames + wearFrames
    if f >= wearStart && f < wearEnd {
        // Linear in time: a constant rate is a constant speed.
        let u: Float = Float(f - wearStart + 1) / Float(wearFrames)
        depth = finalWearDepth * u
        if mutant == .unwear && u > 0.5 { depth = finalWearDepth * (1 - u) }
    } else if f >= wearEnd {
        depth = finalWearDepth
        let fadeStart: Int = wearEnd + wornFrames
        if f >= fadeStart { dissolve = smooth(Float(f - fadeStart + 1) / Float(fadeFrames)) }
    }
    if mutant == .unwear && f >= wearEnd { depth = 0 }
    if mutant == .invisible { depth *= 0.1 }
    var camera: Camera = stillCamera
    if mutant == .cameraMoves {
        let drift: Float = 0.6 * Float(max(f - wearStart, 0)) / Float(wearFrames)
        camera = Camera(position: stillCamera.position + SIMD3<Float>(0, drift, drift), target: stillCamera.target)
    }
    // The caption's year is the drawn depth over the rate, so the label can
    // never disagree with the teeth.
    return FramePlan(depth: depth, years: depth / bruxistWearRate, dissolve: dissolve, camera: camera)
}

/// The year each feature first shows, for its label to appear: depth over
/// rate again, with the depths the survey finds (`firstDentineDepth`).
struct FeatureTimes {
    var facets: Float           // facets first reach a visible 0.1 mm
    var incisorDentine: Float   // dentine first bared on the lateral incisor's edge
    var canineFlat: Float       // the canine's tip ground 1 mm
    var premolarDentine: Float  // dentine first bared on the first premolar
}

/// A facet is "formed" for labelling at 0.1 mm deep, and the canine "flat"
/// once 1 mm is gone from its tip. MODEL: thresholds for when to put a label
/// up, not anatomy.
let facetLabelDepth: Float = 0.1
let canineFlatDepth: Float = 1.0

/// Finds, by bisection on the survey, the depth at which tooth `tooth` first
/// shows dentine.
func firstDentineDepth(tooth: Int, on device: MTLDevice) throws -> Float {
    var lo: Float = 0
    var hi: Float = finalWearDepth
    for _ in 0..<12 {
        let mid: Float = (lo + hi) / 2
        let s: [ToothSurvey] = try surveyTeeth(depth: mid, step: 0.05, on: device)
        if s[tooth].dentine.isEmpty { lo = mid } else { hi = mid }
    }
    return hi
}

/// The years at which each label goes up.
func featureTimes(on device: MTLDevice) throws -> FeatureTimes {
    let incisor: Float = try firstDentineDepth(tooth: incisorLabelTooth, on: device)
    let premolar: Float = try firstDentineDepth(tooth: dentineLabelTooth, on: device)
    return FeatureTimes(facets: facetLabelDepth / bruxistWearRate,
                        incisorDentine: incisor / bruxistWearRate,
                        canineFlat: canineFlatDepth / bruxistWearRate,
                        premolarDentine: premolar / bruxistWearRate)
}

/// The patient's left lateral incisor (#23), whose edge the camera sees.
let incisorLabelTooth: Int = 1
