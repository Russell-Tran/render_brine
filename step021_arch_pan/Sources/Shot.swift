// The shot: a camera that travels alongside the lower teeth, from the midline
// between the two central incisors (#25 and #24) back along the patient's left,
// past the second molar (#18), to the wisdom tooth (#17), holds there, and
// dissolves back to where it began.
//
// Everything here is plain arithmetic on step 20's arch, so the tests can read
// it without a GPU. The camera is the only thing that moves; the teeth, the
// gum, the colours and the lights are step 20's, with two of its options
// switched on: the wisdom teeth, and the retromolar pad behind them.
//
// Millimetres, and step 20's axes: y up, y = 0 the occlusal plane, +z back
// towards the throat, the midline at x = 0, z = 0.
//
// WHICH SIDE IS LEFT. Step 20's comments used to call +x the patient's left.
// But the renderer is right-handed — the camera's right is forward × up — and
// facing the patient from the front that puts +x on the viewer's LEFT, which
// is the patient's RIGHT. (Face someone: their left hand is on your right.)
// Step 20's arch is an exact mirror image across x = 0, so its still was
// correct either way and only its comments needed to change. Here it matters: the camera starts
// facing the patient and moves off to one side, and a dentist watching would
// name the tooth it ends on by which way it went. To end on #17, the lower
// LEFT third molar, as seen, the camera goes to −x: towards the viewer's
// right in the first frame. That is the tooth step 20's
// `placeTeeth(thirdMolars: true)` lists at index 14 (`tooth17Index`); #18,
// the second molar in front of it, is index 6.

import Foundation
import simd

// MARK: - the rail

/// How far out from the arch the camera rides, along the arch's outward
/// normal. MODEL: close enough that the tooth in the middle of the frame fills
/// about a third of its width (the frame is ±19 mm wide at this distance with
/// step 20's lens), far enough that its neighbours on both sides are in shot.
let railOffset: Float = 35.0

/// How high above the occlusal plane the camera rides. MODEL: high enough to
/// see onto the occlusal tables of the premolars and molars — the cusps are
/// what make each tooth recognisable — low enough that the labial and buccal
/// faces, which carry the colour and the highlights, still face the lens.
let railHeight: Float = 15.0

/// The height the camera aims at: about the middle of the crown. The crowns in
/// Wheeler's table run 7 to 11 mm tall, so y = −4 is mid-crown on all of them.
let aimHeight: Float = -4.0

/// Half the window over which the arch is averaged to steer the camera.
///
/// Step 20's Bonwill–Hawley arch is a circle round the front teeth and a
/// straight line from the canine back, and the two do not quite meet head on:
/// the line is aimed so the first molars land 38 mm apart, which leaves it
/// turned about 20° from the circle's direction where they join, at the distal
/// end of the canine (17.5 mm). A camera that followed the raw arch would swing
/// 20° in no time at all there — a jolt — and even a camera that only followed
/// the arch's curvature would snap from turning to not turning.
///
/// So the camera follows the arch AVERAGED over ±7 mm (a canine's width either
/// side), which rounds the corner off, and takes its direction from that
/// averaged arch averaged once more. Each average smooths by one order: the
/// arch has a corner; averaged once it has none; averaged twice its direction
/// turns without a snap. MODEL: a wider window softens more but leaves the
/// camera less square to the tooth it is looking at.
let smoothingHalfWindow: Float = 7.0

/// Arc length from the midline to #17's centre: step 20's
/// `placeTeeth(thirdMolars: true)` puts the third molar exactly here on both
/// sides, touching the second molar in front of it.
let endArc: Float = thirdMolarArc

/// The side the camera travels along: −x, the patient's left as seen (see the
/// head of this file), so the shot ends on #17.
let travelSide: Float = -1
/// #17 in step 20's list with the wisdom teeth on: the third molar on the −x
/// side. #18, the second molar, is still index 6.
let tooth17: Int = tooth17Index
let tooth18: Int = 6

/// Step 20's `archPoint` covers one half of the arch. Negative arc lengths are
/// the other half, its mirror image, so the rail can look across the midline.
func archPointSigned(_ s: Float) -> SIMD2<Float> {
    let p: SIMD2<Float> = archPoint(abs(s)).point
    return s < 0 ? SIMD2<Float>(-p.x, p.y) : p
}

/// The running total of the arch point from the midline, ∫₀ˢ P(u) du, in closed
/// form — so an average over any window is a difference of two of these, with
/// no sampling to get wrong at the corner. Round the circle P is
/// (r sin(u/r), r(1 − cos(u/r))); along the line it is linear.
func archIntegral(_ s: Float) -> SIMD2<Float> {
    let a: Float = abs(s)
    let r: Float = hawleyRadius
    let arc: Float = min(a, r)
    let angle: Float = arc / r
    var q = SIMD2<Float>(r * r * (1 - cos(angle)), r * arc - r * r * sin(angle))
    if a > r {
        let corner: SIMD2<Float> = archPoint(r).point
        let dir: SIMD2<Float> = archPoint(a).tangent
        let run: Float = a - r
        q += corner * run + dir * (run * run / 2)
    }
    // The mirror half: x is odd in s, so its integral is even; z the reverse.
    return s < 0 ? SIMD2<Float>(q.x, -q.y) : q
}

/// The arch averaged over the window around `s`: no corner.
func smoothedArch(_ s: Float) -> SIMD2<Float> {
    let w: Float = smoothingHalfWindow
    return (archIntegral(s + w) - archIntegral(s - w)) / (2 * w)
}

/// The direction of the averaged arch, averaged again: the chord across the
/// window of the averaged arch, since the average of a direction along a curve
/// is the chord divided by its length.
func smoothedTangent(_ s: Float) -> SIMD2<Float> {
    let w: Float = smoothingHalfWindow
    return simd_normalize(smoothedArch(s + w) - smoothedArch(s - w))
}

/// Where the camera aims, before it is lifted to mid-crown: the averaged arch,
/// except within 10 mm of either end, where it hands back to the arch itself
/// so the shot starts and ends EXACTLY on the named teeth. The two differ by
/// under half a millimetre there — the average sits a little inside the curve
/// of the incisors — so the hand-over cannot be seen. Both ends are 10 mm clear
/// of the ±7 mm window round the corner (10.5 to 24.5 mm), so the raw arch is
/// never used where it bends.
func aimPoint(_ s: Float) -> SIMD2<Float> {
    let nearStart: Float = 1 - smoothstep(0, 10, abs(s))
    let nearEnd: Float = smoothstep(endArc - 10, endArc, s)
    let onArch: Float = max(nearStart, nearEnd)
    return archPointSigned(s) * onArch + smoothedArch(s) * (1 - onArch)
}

func smoothstep(_ a: Float, _ b: Float, _ x: Float) -> Float {
    let t: Float = min(max((x - a) / (b - a), 0), 1)
    return t * t * (3 - 2 * t)
}

/// What to break, for the mutation check. Each must be caught by a test.
enum ShotMutant {
    case none
    case raw        // follow the raw arch, corner and all
    case rewind     // close the loop by playing the travel backwards
    case noWisdom   // leave the wisdom teeth out of the scene
}

/// Whether the scene has the wisdom teeth: always, except for the mutant that
/// takes them out to check a test notices the shot no longer ends on one.
func withThirdMolars(_ mutant: ShotMutant = .none) -> Bool {
    mutant != .noWisdom
}

/// The camera for a look-at point `s` along the arch: aimed at the arch at
/// mid-crown, standing `railOffset` out along the smoothed outward normal and
/// `railHeight` above the occlusal plane. Built on step 20's +x half, where
/// `archPoint` lives, then mirrored onto `travelSide`.
func railCamera(_ s: Float, mutant: ShotMutant = .none) -> Camera {
    var p: SIMD2<Float> = aimPoint(s)
    var t: SIMD2<Float> = smoothedTangent(s)
    if mutant == .raw {
        p = archPointSigned(s)
        let raw: SIMD2<Float> = archPoint(abs(s)).tangent
        t = s < 0 ? SIMD2<Float>(raw.x, -raw.y) : raw
    }
    // A quarter turn clockwise from the direction of travel points out of the
    // +x half: at the midline, travel is +x and out is −z.
    let outward = SIMD2<Float>(t.y, -t.x)
    let stand: SIMD2<Float> = p + outward * railOffset
    return Camera(position: SIMD3<Float>(stand.x * travelSide, railHeight, stand.y),
                  target: SIMD3<Float>(p.x * travelSide, aimHeight, p.y))
}

// MARK: - the timeline

/// 5 centiseconds a frame: 20 frames a second. With the camera moving, every
/// pixel changes every frame and each one costs the GIF a whole frame, so the
/// frame rate is paid for directly in file size; 20 is about the fewest at
/// which a move this size still reads as a glide rather than a series of steps.
let frameDelayCentiseconds: Int = 5
let framesPerSecond: Float = 100 / Float(frameDelayCentiseconds)

/// Seconds of travel, of hold on #17, and of dissolve back to the start.
/// The travel is a tooth longer than it was when the shot ended on #18 (58.0
/// mm of arch instead of 47.75), so it takes a little longer: 8 s instead of
/// 7.5, the glide 14% quicker than before. Keeping the old pace would have
/// needed 9.1 s and a GIF well over 10 MB; see main.swift for the measurement.
let travelSeconds: Float = 8.0
let holdSeconds: Float = 1.0
let dissolveSeconds: Float = 1.0

let travelFrames: Int = Int((travelSeconds * framesPerSecond).rounded())
let holdFrames: Int = Int((holdSeconds * framesPerSecond).rounded())
let dissolveFrames: Int = Int((dissolveSeconds * framesPerSecond).rounded())
let frameCount: Int = travelFrames + 1 + holdFrames + dissolveFrames
let loopSeconds: Float = Float(frameCount) / framesPerSecond

/// The fraction of the travel spent speeding up, and slowing down. MODEL: a
/// quarter each way reads as a deliberate start and a settled arrival, and
/// leaves half the move as an even glide past the premolars.
let easeIn: Float = 0.25
let easeOut: Float = 0.25

/// How far along the travel, 0 to 1, at time fraction `u`. The speed rises
/// from rest along half a cosine, holds, and falls back to rest the same way;
/// this is that speed integrated, in closed form. Speed and position are both
/// continuous, so the camera neither lurches off nor stops dead.
func eased(_ u: Float) -> Float {
    let x: Float = min(max(u, 0), 1)
    let a: Float = easeIn
    let b: Float = easeOut
    let top: Float = 1 / (1 - a / 2 - b / 2)    // the glide speed, so the total is 1
    if x < a {
        let phase: Float = Float.pi * x / a
        let ramp: Float = x - a / Float.pi * sin(phase)
        return top * ramp / 2
    }
    let afterIn: Float = top * a / 2
    if x <= 1 - b {
        return afterIn + top * (x - a)
    }
    let y: Float = 1 - x
    let phase: Float = Float.pi * y / b
    let ramp: Float = y - b / Float.pi * sin(phase)
    return 1 - top * ramp / 2
}

/// One frame: where the camera looks along the arch, and how much of frame 0
/// is dissolved over it.
struct FramePlan {
    var arc: Float
    var dissolve: Float
}

func plan(_ f: Int, mutant: ShotMutant = .none) -> FramePlan {
    if f <= travelFrames {
        let u: Float = Float(f) / Float(travelFrames)
        return FramePlan(arc: endArc * eased(u), dissolve: 0)
    }
    if f <= travelFrames + holdFrames {
        return FramePlan(arc: endArc, dissolve: 0)
    }
    let k: Int = f - travelFrames - holdFrames      // 1 … dissolveFrames
    if mutant == .rewind {
        let u: Float = 1 - Float(k) / Float(dissolveFrames)
        return FramePlan(arc: endArc * eased(u), dissolve: 0)
    }
    // Stop one step short of all the way, so the frame after the last one —
    // frame 0 itself — is the step that completes the dissolve.
    let weight: Float = Float(k) / Float(dissolveFrames + 1)
    return FramePlan(arc: endArc, dissolve: weight)
}
