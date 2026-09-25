// The loop: six beats, and the one thing about it that is not ordinary.
//
// A cracked tooth cannot become an uncracked one. Every loop in this series so
// far has closed by running the same process to its start — the coconut's
// growth dissolving back to the dormant nut, the polarimeter draining and
// refilling. Elastic strain can do that: take the bite away and the field goes
// to zero, exactly, because the model is linear. CRACKS CANNOT. So this loop
// closes the way step 18's did when it could not un-hydrolyse sucrose: it
// dissolves to a FRESH TOOTH, and the dissolve is a cut, not a rewind. A test
// asserts that no bond that has broken is ever restored inside a run.

import Foundation

let framesPerBeat = 24
let toothFrameCount = framesPerBeat * Beat.allCases.count      // 144
let frameDelayCentiseconds = 7
let loopSeconds: Double = Double(frameDelayCentiseconds) * Double(toothFrameCount) / 100

/// The last stretch of the final beat cross-dissolves back to frame 0.
let dissolveFrames = 16
let dissolveStart = toothFrameCount - dissolveFrames

func beat(frame f: Int) -> Beat {
    let i: Int = min(max(f / framesPerBeat, 0), Beat.allCases.count - 1)
    return Beat(rawValue: i) ?? .anatomy
}

/// How far through its own beat a frame is, in [0, 1).
func beatProgress(frame f: Int) -> Float {
    let within: Int = f % framesPerBeat
    return Float(within) / Float(framesPerBeat)
}

func dissolve(frame f: Int) -> Float {
    guard f >= dissolveStart else { return 0 }
    let num: Float = Float(f - (dissolveStart - 1))
    let den: Float = Float(dissolveFrames)
    return min(num / den, 1)
}

func smoothstep(_ a: Float, _ b: Float, _ x: Float) -> Float {
    if b <= a { return x < a ? 0 : 1 }
    let t: Float = min(max((x - a) / (b - a), 0), 1)
    let u: Float = t * t
    let v: Float = 3 - 2 * t
    return u * v
}

/// What one frame asks for. `state` indexes the precomputed physics states;
/// `gain` scales a linear elastic field, which is legitimate because the model
/// IS linear and one solve therefore gives the whole ramp below cracking.
struct Shot {
    var beat: Beat
    var state: Int
    var gain: Float
    var stressMix: Float
    var labelFade: Float
    var keyFade: Float
    var newtons: Float
}
