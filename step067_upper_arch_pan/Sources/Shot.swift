// The shot: step 21's move, one jaw up. A camera that travels alongside the
// upper teeth, from the midline between the two central incisors (#8 and #9)
// back along the patient's left, past the second molar (#15), to the wisdom
// tooth, #16 — directly above step 21's #17 — holds there, and dissolves
// forward to where it began. Never a rewind.
//
// Step 21's Shot.swift, copied and adapted: the rail rides BELOW the occlusal
// plane and looks up at the crowns, the arch is step 66's maxillary one, and
// the tooth it ends on is found by its Universal number. Everything here is
// plain arithmetic, so the tests can read it without a GPU.
//
// World millimetres: y up, y = 0 the occlusal plane (the upper crowns hang
// below the gum, towards −y), +z back towards the throat, the midline at
// x = 0, z = 0. +x is the patient's RIGHT; the renderer is right-handed, so
// facing the patient the camera's right is −x. To end on #16, the upper LEFT
// third molar, the camera goes to −x: towards the viewer's right in the first
// frame, as step 21 went to #17.

import Foundation
import simd

// MARK: - what to break

/// Shot mutants, read from SHOT_MUTANT. Each must make a test fail.
enum ShotMutant: String {
    case none
    case raw            // follow the raw arch, canine kink and all
    case rewind         // close the loop by playing the travel backwards
    case frozenCamera   // the camera never moves
    case mirrored       // travel the patient's right side, ending on #1
}

let shotMutant: ShotMutant = {
    let raw: String = ProcessInfo.processInfo.environment["SHOT_MUTANT"] ?? ""
    return ShotMutant(rawValue: raw) ?? .none
}()

// MARK: - the rail

/// How far out from the arch the camera rides, along its outward normal:
/// step 21's 35 mm. MODEL.
let railOffset: Float = 35.0

/// How far BELOW the occlusal plane the camera rides: step 21's 15 mm above,
/// turned over. MODEL: low enough to see onto the occlusal tables of the
/// premolars and molars, high enough that the buccal faces still face the lens.
let railHeight: Float = -15.0

/// The height the camera aims at: about the middle of the crown. The upper
/// crowns in Wheeler's table run 6.5 to 10.5 mm tall, so y = +4 is mid-crown
/// on all of them.
let aimHeight: Float = 4.0

/// Half the window over which the arch is averaged to steer the camera.
///
/// Step 66's arch, like step 20's, is a circle round the front six and a
/// straight line from the canine back, and the line is aimed at the first
/// molars' measured span, which leaves it about 15° off the circle's direction
/// where they join, at the distal end of the canine (22.5 mm from the midline).
/// A camera that followed the raw arch would swing 15° in no time there.
/// Step 21's cure, unchanged: follow the arch averaged over ± this much, and
/// take the heading from that averaged once more. MODEL: 7.5 mm is the upper
/// canine's own width, as step 21's 7 mm was about the lower canine's.
let smoothingHalfWindow: Float = 7.5

/// The side the camera travels along: −x, the patient's left — or, under the
/// `mirrored` mutant, the right, so the shot ends on #1.
let travelSide: Float = shotMutant == .mirrored ? 1 : -1

/// Arc length from the midline to the third molar's centre: where step 66
/// places #16 and #1.
let endArc: Float = archCentres[7]

/// #16 and #15 in `placeTeeth()`, found by number rather than assumed.
let tooth16: Int = toothIndex(16, in: placeTeeth()) ?? 7
let tooth15: Int = toothIndex(15, in: placeTeeth()) ?? 6

/// The arch's +x half as a point on the signed arc: negative arc lengths are
/// the mirror half, so the rail can look across the midline.
func archPointSigned(_ s: Float) -> SIMD2<Float> {
    let p: SIMD2<Float> = archPoint(abs(s)).point
    return s < 0 ? SIMD2<Float>(-p.x, p.y) : p
}

/// ∫₀ˢ P(u) du in closed form, as step 21 has it: round the circle P is
/// (r sin(u/r), r(1 − cos(u/r))); along the line it is linear.
func archIntegral(_ s: Float) -> SIMD2<Float> {
    let a: Float = abs(s)
    let r: Float = hawleyRadius
    let arc: Float = min(a, r)
    let angle: Float = arc / r
    let r2: Float = r * r
    var q = SIMD2<Float>(r2 * (1 - cos(angle)), r * arc - r2 * sin(angle))
    if a > r {
        let corner: SIMD2<Float> = archPoint(r).point
        let dir: SIMD2<Float> = archPoint(a).tangent
        let run: Float = a - r
        let half: Float = run * run / 2
        q += corner * run + dir * half
    }
    return s < 0 ? SIMD2<Float>(q.x, -q.y) : q
}

/// The arch averaged over the window around `s`: no corner.
func smoothedArch(_ s: Float) -> SIMD2<Float> {
    let w: Float = smoothingHalfWindow
    return (archIntegral(s + w) - archIntegral(s - w)) / (2 * w)
}

/// The direction of the averaged arch, averaged again.
func smoothedTangent(_ s: Float) -> SIMD2<Float> {
    let w: Float = smoothingHalfWindow
    return simd_normalize(smoothedArch(s + w) - smoothedArch(s - w))
}

func smoothstep(_ a: Float, _ b: Float, _ x: Float) -> Float {
    let t: Float = min(max((x - a) / (b - a), 0), 1)
    return t * t * (3 - 2 * t)
}

/// Where the camera aims: the averaged arch, handed back to the arch itself
/// within 10 mm of either end so the shot starts and ends EXACTLY on the named
/// teeth (step 21). Both hand-overs are clear of the ±7.5 mm window round the
/// corner (15 to 30 mm): the start's ends at 10 mm, the end's begins at 49.75.
func aimPoint(_ s: Float) -> SIMD2<Float> {
    let nearStart: Float = 1 - smoothstep(0, 10, abs(s))
    let nearEnd: Float = smoothstep(endArc - 10, endArc, s)
    let onArch: Float = max(nearStart, nearEnd)
    return archPointSigned(s) * onArch + smoothedArch(s) * (1 - onArch)
}

/// The camera for a look-at point `s` along the arch: aimed at mid-crown,
/// standing `railOffset` out along the smoothed outward normal and
/// `railHeight` below the occlusal plane. Built on the +x half, where
/// `archPoint` lives, then mirrored onto `travelSide`.
func railCamera(_ s: Float, mutant: ShotMutant = shotMutant) -> Camera {
    let at: Float = mutant == .frozenCamera ? 0 : s
    var p: SIMD2<Float> = aimPoint(at)
    var t: SIMD2<Float> = smoothedTangent(at)
    if mutant == .raw {
        p = archPointSigned(at)
        let raw: SIMD2<Float> = archPoint(abs(at)).tangent
        t = at < 0 ? SIMD2<Float>(raw.x, -raw.y) : raw
    }
    // A quarter turn clockwise from the direction of travel points out of the
    // +x half: at the midline, travel is +x and out is −z.
    let outward = SIMD2<Float>(t.y, -t.x)
    let stand: SIMD2<Float> = p + outward * railOffset
    return Camera(position: SIMD3<Float>(stand.x * travelSide, railHeight, stand.y),
                  target: SIMD3<Float>(p.x * travelSide, aimHeight, p.y))
}

// MARK: - the timeline (step 21's)

/// 5 centiseconds a frame, 20 frames a second: step 21's rate. With the
/// camera moving every pixel changes every frame, so the GIF pays for each.
let frameDelayCentiseconds: Int = 5
let framesPerSecond: Float = 100 / Float(frameDelayCentiseconds)

/// Seconds of travel, of hold on #16, and of dissolve back to the start.
/// Step 21's: 8, 1 and 1. The travel here is 59.75 mm of arch against step
/// 21's 58.0, so the glide is 3% quicker at the same length of loop.
let travelSeconds: Float = 8.0
let holdSeconds: Float = 1.0
let dissolveSeconds: Float = 1.0

let travelFrames: Int = Int((travelSeconds * framesPerSecond).rounded())
let holdFrames: Int = Int((holdSeconds * framesPerSecond).rounded())
let dissolveFrames: Int = Int((dissolveSeconds * framesPerSecond).rounded())
let frameCount: Int = travelFrames + 1 + holdFrames + dissolveFrames
let loopSeconds: Float = Float(frameCount) / framesPerSecond

/// Step 21's ease: a quarter of the travel speeding up, a quarter slowing.
let easeIn: Float = 0.25
let easeOut: Float = 0.25

/// How far along the travel, 0 to 1, at time fraction `u`: speed rises along
/// half a cosine, holds, and falls back the same way, integrated in closed form.
func eased(_ u: Float) -> Float {
    let x: Float = min(max(u, 0), 1)
    let a: Float = easeIn
    let b: Float = easeOut
    let top: Float = 1 / (1 - a / 2 - b / 2)
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

func plan(_ f: Int, mutant: ShotMutant = shotMutant) -> FramePlan {
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
    // One step short of all the way, so frame 0 itself completes the dissolve.
    let weight: Float = Float(k) / Float(dissolveFrames + 1)
    return FramePlan(arc: endArc, dissolve: weight)
}

/// Every frame's camera, as the loop renders it.
func frameCamera(_ f: Int, mutant: ShotMutant = shotMutant) -> Camera {
    railCamera(plan(f, mutant: mutant).arc, mutant: mutant)
}
