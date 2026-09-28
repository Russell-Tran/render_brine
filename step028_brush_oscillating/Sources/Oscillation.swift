// Step 28: step 22's brush head doing what an oscillating-rotating head does —
// turning back and forth about its own axis — slowed down enough to see.
//
// Nothing else moves: step 20's mouth, camera and light, and step 22's neck,
// press and tufts are used as they are. Only the disc turns, carrying its
// tufts and the boss on its back, and every frame each tuft is cast again
// against step 20's distance function, from where the turn has put its root,
// so its tip lands on the tooth or gum exactly — shorter where the surface in
// front of it is nearer, splayed where it is pressed harder.
//
// Every tuft but one, that is. As the disc turns, one rim tuft at a time
// passes over the gap between #21 and the canine, #22, where the papilla lies
// deeper than a new tuft is long; there it springs out to its full length and
// hangs free, short of the gum, as a real one would. The head is not pushed in
// to make it touch: step 22's press is held, as a hand holds it, and a tuft
// never grows.
//
// Millimetres, seconds and step 20's axes throughout.

import Foundation
import simd

// MARK: - the real motion, from the literature

/// How far the head turns, end to end: 45°. "Oral-B Professional Care Series,
/// oscillation angle of 45 degrees, 73 Hz, with a pulsation frequency of 340
/// Hz" — the device as described in Strate J, Cugini MA, Warren PR, Qaqish JG,
/// Galustians HJ, Sharma NC, "A comparison of the plaque removal efficacy of
/// two power toothbrushes: Oral-B Professional Care Series versus Sonicare
/// Elite", Int Dent J 2005;55:151–156, as tabulated by Lewis et al., "The
/// Effect of Different Electric Toothbrush Technologies on Interdental Plaque
/// Removal: A Systematic Review with a Meta-Analysis", Healthcare 2024;12:1035
/// (PMC11121692), Table 2. The primary paper was not reachable; the figure is
/// taken from the review's table, and read as the whole arc — 22.5° either
/// side of rest — because an "oscillation angle" is the swing, not the
/// half-swing. A lower-priced Oral-B body is described as making "16 degree
/// movements" at 7,600 oscillations a minute (Klonowicz et al., BMC Oral
/// Health 2018, PMC6220499); the 45° figure is taken because it is the one
/// given with the frequency that matches the maker's 8,800 a minute below,
/// and it is flagged as a second-hand number.
let swingDegrees: Float = 45.0
/// Half the swing, in radians: the most the head is ever turned from rest.
let swingHalf: Float = swingDegrees / 2 * Float.pi / 180

/// How often: 73 full back-and-forth cycles a second (the same table, same
/// device). It agrees with the maker's figure quoted in the same review for
/// Oral-B's round heads, "8800 oscillations/40,000 pulsations per minute"
/// for the Oral-B Triumph (Biesbrock et al., Am J Dent 2008;21:185–188, in
/// the review's Table 2), and 8800 a minute again for the PRO 700 (Lv et al.,
/// Am J Dent 2018): 8,800 strokes a minute, each one way, is 73.3 full cycles
/// a second.
let oscillationHz: Double = 73.0
/// A real cycle: 13.7 ms.
let realCycleSeconds: Double = 1.0 / oscillationHz

/// The motion within a cycle. MODEL: a sine — the head is driven at a steady
/// rate through a fixed arc, fastest through rest and stopping to reverse at
/// each end. The pulsation (the head's small in-and-out tapping at 340 Hz) is
/// not drawn: its amplitude is not in the sources found, and at this slow-down
/// it would be a 1.2 s wobble laid over the turn.

// MARK: - the loop

/// One full cycle — rest, 22.5° one way, back through rest, 22.5° the other,
/// back to rest — takes the whole loop, so the GIF's last frame flows into its
/// first. Frames are 5 cs apart (20 frames a second), and the loop is 80
/// frames: 4 s for what really takes 13.7 ms.
let frameDelayCentiseconds: Int = 5
let frameCount: Int = 80
let loopSeconds: Double = Double(frameCount * frameDelayCentiseconds) / 100
/// How much slower than life: 4 s against 13.7 ms, 292×.
let slowdown: Double = loopSeconds / realCycleSeconds

/// What to break, for the mutation check. Each must be caught by a test.
enum OscillationMutant {
    case none
    case frozen     // tufts not cast again: frame 0's lengths turned rigidly with the disc
    case overswing  // swing half again the cited 45°
    case open       // the cycle takes 10% longer than the loop, so it never closes
}

let oscillationMutant: OscillationMutant = {
    switch ProcessInfo.processInfo.environment["OSC_MUTANT"] {
    case "frozen": return .frozen
    case "overswing": return .overswing
    case "open": return .open
    default: return .none
    }
}()

/// The head's turn at frame f, in radians. Frame `frameCount` is frame 0
/// again.
func spinAngle(frame f: Int, mutant: OscillationMutant = oscillationMutant) -> Float {
    let amplitude: Float = mutant == .overswing ? swingHalf * 1.5 : swingHalf
    let period: Double = mutant == .open ? Double(frameCount) * 1.1 : Double(frameCount)
    let phase: Double = 2 * Double.pi * Double(f) / period
    return amplitude * Float(sin(phase))
}

// MARK: - the brush, turned and cast again

/// Step 22's press, held: the face stays where step 22 pressed it, and at each
/// turn the tufts are cast again exactly as step 22 cast them — straight from
/// a trial face to find how far each one reaches, then splayed by how hard it
/// is pressed and cast along its splayed direction from its root on the face.
final class OscillatingBrush {
    let probe: SceneProbe
    let axis: SIMD3<Float>
    let along: SIMD3<Float>
    let across: SIMD3<Float>
    let trial: SIMD3<Float>
    /// How far step 22 moved the face in from the trial: its longest straight
    /// reach, less the rest length. Held for every turn — the hand does not
    /// push harder or softer as the head turns.
    let pressIn: Float
    let face: SIMD3<Float>
    /// The brush at rest, turn 0: step 22's.
    let rest: Brush

    init(probe: SceneProbe) throws {
        self.probe = probe
        let frame = brushFrame()
        axis = frame.axis
        along = frame.along
        across = frame.across
        trial = marginAim() - axis * (restLength + 3)
        let straight: [Float] = try castLengths(OscillatingBrush.roots(trial, along, across, spin: 0).map {
            ($0, frame.axis, tuftDiameter / 2, tuftDiameter / 2)
        }, probe: probe)
        pressIn = straight.max()! - restLength
        face = trial + axis * pressIn
        rest = try OscillatingBrush.cast(probe: probe, trial: trial, face: face, pressIn: pressIn,
                                         axis: axis, along: along, across: across, spin: 0)
    }

    static func roots(_ face: SIMD3<Float>, _ along: SIMD3<Float>, _ across: SIMD3<Float>,
                      spin: Float) -> [SIMD3<Float>] {
        let (a, b) = spinFrame(along: along, across: across, spin: spin)
        return tuftLayout().map { face + a * $0.position.x + b * $0.position.y }
    }

    /// Step 22's `pressedBrush`, steps 1 and 3, with the tufts laid out on the
    /// turned disc and the press held at `pressIn`.
    static func cast(probe: SceneProbe, trial: SIMD3<Float>, face: SIMD3<Float>, pressIn: Float,
                     axis: SIMD3<Float>, along: SIMD3<Float>, across: SIMD3<Float>,
                     spin: Float) throws -> Brush {
        let layout = tuftLayout()
        let r0: Float = tuftDiameter / 2
        let (sa, sb) = spinFrame(along: along, across: across, spin: spin)
        let straight: [Float] = try castLengths(roots(trial, along, across, spin: spin).map { ($0, axis, r0, r0) },
                                                probe: probe)
        let pressed: [Float] = straight.map { $0 - pressIn }
        var specs: [(root: SIMD3<Float>, direction: SIMD3<Float>, rootRadius: Float, tipRadius: Float)] = []
        var compressions: [Float] = []
        for (i, l) in layout.enumerated() {
            let c: Float = max(restLength - pressed[i], 0) / restLength
            let radial: Float = simd_length(l.position)
            var dir: SIMD3<Float> = axis
            if radial > 1e-3 {
                let outward: SIMD3<Float> = simd_normalize(sa * l.position.x + sb * l.position.y)
                let lean: Float = splayLean * c
                dir = simd_normalize(axis * cos(lean) + outward * sin(lean))
            }
            let root: SIMD3<Float> = face + sa * l.position.x + sb * l.position.y
            specs.append((root, dir, r0, r0 * (1 + splayGrowth * c)))
            compressions.append(c)
        }
        let lengths: [Float] = try castLengths(specs, probe: probe)
        var tufts: [Tuft] = []
        for (i, s) in specs.enumerated() {
            // A tuft cannot grow. Where the turn carries it over the gap
            // between #21 and #22 the gum is further than a new tuft reaches
            // (up to 1.2 mm further along its axis, measured — though the
            // gap is narrow, and its tip is never more than 0.11 mm from the
            // walls either side), and there it stands free at
            // its rest length, unpressed and straight, touching nothing —
            // as a real tuft does over an embrasure. At turn 0 no tuft is
            // free (step 22 pressed the head until the one with furthest to
            // go just arrived), so this changes nothing in step 22's pose.
            // (Within the cast's own precision, a micron: step 22's longest
            // tuft is cast to its rest length and must stay exactly as cast.)
            let free: Bool = lengths[i] > restLength + 1e-3
            tufts.append(Tuft(root: s.root, direction: s.direction, length: free ? restLength : lengths[i],
                              rootRadius: s.rootRadius, tipRadius: s.tipRadius,
                              dye: layout[i].dye, compression: compressions[i], free: free))
        }
        let pose = BrushPose(face: face, axis: axis, along: along, across: across,
                             spinAlong: sa, spinAcross: sb, spin: spin)
        return Brush(pose: pose, tufts: tufts)
    }

    /// Turn a point that rides on the disc by `spin` about the head's axis,
    /// the way the disc turns: (x, y) on the disc goes to
    /// (x cos − y sin, x sin + y cos).
    func turn(_ v: SIMD3<Float>, spin: Float, isDirection: Bool) -> SIMD3<Float> {
        let r: SIMD3<Float> = isDirection ? v : v - face
        let x: Float = simd_dot(r, along)
        let y: Float = simd_dot(r, across)
        let z: Float = simd_dot(r, axis)
        let c: Float = cos(spin)
        let s: Float = sin(spin)
        let out: SIMD3<Float> = along * (x * c - y * s) + across * (x * s + y * c) + axis * z
        return isDirection ? out : face + out
    }

    /// The brush at a turn: cast again — or, for the `frozen` mutant, frame
    /// 0's tufts carried round rigidly, lengths and all.
    func brush(spin: Float, mutant: OscillationMutant = oscillationMutant) throws -> Brush {
        if mutant == .frozen {
            var b: Brush = rest
            let (sa, sb) = spinFrame(along: along, across: across, spin: spin)
            b.pose.spinAlong = sa
            b.pose.spinAcross = sb
            b.pose.spin = spin
            b.tufts = rest.tufts.map { t in
                var u: Tuft = t
                u.root = turn(t.root, spin: spin, isDirection: false)
                u.direction = simd_normalize(turn(t.direction, spin: spin, isDirection: true))
                return u
            }
            return b
        }
        return try OscillatingBrush.cast(probe: probe, trial: trial, face: face, pressIn: pressIn,
                                         axis: axis, along: along, across: across, spin: spin)
    }

    func brush(frame f: Int, mutant: OscillationMutant = oscillationMutant) throws -> Brush {
        try brush(spin: spinAngle(frame: f, mutant: mutant), mutant: mutant)
    }
}
