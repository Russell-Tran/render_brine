// What moves: the front cover swings open about its joint, the leaves turn
// one after another as if blown, the back cover swings closed over the last
// one, and the loop dissolves forward to the start. Nothing ever turns
// back.
//
// Paper bends but does not stretch. Every leaf is drawn as a generalised
// cylinder: a curve in the book's cross-section (x, y), run unchanged along
// the spine (z). Such a surface is developable — its Gaussian curvature is
// zero (Wikipedia, "Developable surface": cylinders "and, more generally,
// the 'generalized' cylinder; its cross-section may be any smooth curve";
// developable = "can be flattened onto a plane without distortion") — and
// if the cross-section curve is traced at unit speed for exactly the leaf's
// width, the leaf unrolls onto the flat leaf, lengths and areas unchanged.
// Each leaf's curve is built from circular arcs and straight pieces, each
// of a length the leaf's width is shared out between, so it can be nothing
// else. The tests measure it anyway, off the primitives the GPU draws.
//
// The kinematics (all MODEL, chosen to be forward, gap-free and
// collision-free, and checked for each):
//
// • The leaves' roots lie along the spine, a straight line of length
//   L = 96 × 0.1016 mm from the back board's joint J_b to the front's J_f;
//   leaf k's root is (k + ½) leaf-thicknesses from J_b. Every leaf leaves
//   its root square to the spine.
// • A leaf lying on the back half wraps round J_b on a circle of radius
//   σ_k (its distance from J_b along the spine), then runs straight along
//   the back board; one lying on the front half wraps round J_f on radius
//   L − σ_k, then runs along the front board. Neighbouring leaves are thus
//   concentric, one thickness apart, and cannot cross.
// • A leaf in flight keeps the same gutter circle (round J_b while it still
//   leans back, round J_f once past upright) and beyond it is one more arc:
//   its direction sweeps from the back board's to the front board's, and it
//   curls, tip lagging, by at most 0.7 rad (MODEL: page stiffness).
// • The spine turns as leaves cross it, from upright (book closed, front
//   cover up) through flat (open at the middle) to upright the other way,
//   in step with the fraction of leaves turned; each cover rests on the
//   table at its outer edge, sloping up to its joint, as a real cover does.

import Foundation
import simd

// MARK: - the timeline (MODEL)

let framesPerSecond: Float = 20
let frameDelayCentiseconds: Int = 5
let holdStart: Float = 0.4            // the closed book, still
let openDuration: Float = 1.4         // the front cover swings open
let riffleGap: Float = 0.1            // before the first leaf lifts
let leafInterval: Float = 0.045       // between one leaf lifting and the next
let flightDuration: Float = 0.6       // one leaf's flight
let closeGap: Float = 0.1
let closeDuration: Float = 1.4        // the back cover swings closed
let holdEnd: Float = 0.3
let fadeDuration: Float = 0.8         // the dissolve forward to frame 0
/// How far a flying leaf curls, tip lagging, at most (radians). MODEL.
let curlMax: Float = 0.7

let riffleStart: Float = holdStart + openDuration + riffleGap
var riffleEnd: Float { riffleStart + Float(leafCount - 1) * leafInterval + flightDuration }
var closeStart: Float { riffleEnd + closeGap }
var closeEnd: Float { closeStart + closeDuration }
var fadeStart: Float { closeEnd + holdEnd }
var loopSeconds: Float { fadeStart + fadeDuration }
var frameCount: Int { Int((loopSeconds * framesPerSecond).rounded()) }

func smooth(_ x: Float) -> Float {
    let c: Float = min(max(x, 0), 1)
    return c * c * (3 - 2 * c)
}

/// When leaf k lifts: the top leaf (k = N − 1, the first page) first.
func leafStart(_ k: Int) -> Float {
    let order: Float = Float(leafCount - 1 - k)
    return riffleStart + order * leafInterval
}

// MARK: - the book's state at a moment

struct BookState {
    var time: Float
    /// Each leaf's progress, 0 lying on the back half, 1 on the front.
    var tau: [Float]
    /// The spine's direction (from J_b to J_f), front and back boards'
    /// directions (from joint towards fore-edge), all radians.
    var psi: Float
    var alpha: Float
    var beta: Float
    var jb: SIMD2<Float>
    var jf: SIMD2<Float>
    /// The dissolve towards frame 0, 0…1.
    var fade: Float
}

let spineLength: Float = Float(leafCount) * paperCaliper

/// A cover's slope, resting on its outer edge with its joint at height y:
/// the angle δ below the horizontal at which the board's far outer corner
/// touches the table. Solved by bisection.
func restingDip(jointHeight y: Float, _ m: Mutant = activeMutant) -> Float {
    let a: Float = boardEnd(m)
    let b: Float = boardStack
    var lo: Float = -0.5
    var hi: Float = 1.5
    for _ in 0..<60 {
        let mid: Float = (lo + hi) / 2
        let sd: Float = a * sin(mid)
        let cd: Float = b * cos(mid)
        let h: Float = y - sd - cd
        if h > 0 { lo = mid } else { hi = mid }
    }
    return (lo + hi) / 2
}

/// Where the joints are for a spine direction ψ: the lower joint's board
/// lies on the table (its joint at the board's height), J_b stays at x = 0.
func joints(psi: Float) -> (jb: SIMD2<Float>, jf: SIMD2<Float>) {
    let d = SIMD2<Float>(cos(psi), sin(psi))
    let sL: Float = spineLength
    if psi <= Float.pi {
        let jb = SIMD2<Float>(0, boardStack)
        return (jb, jb + d * sL)
    }
    let sinPsi: Float = sin(psi)
    let jb = SIMD2<Float>(0, boardStack - sL * sinPsi)
    return (jb, jb + d * sL)
}

/// The book at time t. The `rewind` mutant plays the loop backwards; the
/// `frozen` one never leaves t = 0.
func bookState(at time: Float, _ m: Mutant = activeMutant) -> BookState {
    var t: Float = time
    if m == .frozen { t = 0 }
    if m == .rewind { t = fadeStart - min(time, fadeStart) }
    let fadeW: Float = m == .frozen ? 0 : smooth((time - fadeStart) / fadeDuration)
    var tau: [Float] = []
    tau.reserveCapacity(leafCount)
    for k in 0..<leafCount { tau.append(smooth((t - leafStart(k)) / flightDuration)) }
    let f: Float = tau.reduce(0, +) / Float(leafCount)
    let psi: Float = Float.pi / 2 + Float.pi * f
    let (jb, jf) = joints(psi: psi)
    var alpha: Float
    var beta: Float
    if psi <= Float.pi {
        beta = 0
        let rest: Float = Float.pi + restingDip(jointHeight: jf.y, m)
        let open: Float = smooth((t - holdStart) / openDuration)
        alpha = rest * open
    } else {
        alpha = Float.pi
        beta = -restingDip(jointHeight: jb.y, m)
    }
    if t > closeStart {
        let dip: Float = restingDip(jointHeight: jb.y, m)
        let c: Float = smooth((t - closeStart) / closeDuration)
        beta = -dip + (Float.pi + dip) * c
    }
    return BookState(time: time, tau: tau, psi: psi, alpha: alpha, beta: beta, jb: jb, jf: jf, fade: fadeW)
}

// MARK: - a leaf's curve

/// A piece of a leaf's cross-section: a circular arc (or a straight line,
/// curvature 0) from `start`, heading `angle`, for `length`.
struct Piece {
    var start: SIMD2<Float>
    var angle: Float
    var curvature: Float
    var length: Float

    /// The point s along: the chord to it has length s·sinc(κs/2) and
    /// points along the tangent at s/2 — a form that stays exact as the
    /// curvature goes to zero (no difference of nearly equal sines).
    func point(_ s: Float) -> SIMD2<Float> {
        let half: Float = curvature * s / 2
        let a: Float = angle + half
        return start + SIMD2<Float>(cos(a), sin(a)) * (s * sinc(half))
    }
    var end: SIMD2<Float> { point(length) }
    var endAngle: Float { angle + curvature * length }
}

/// sin(x)/x, by its series near zero.
func sinc(_ x: Float) -> Float {
    let x2: Float = x * x
    if x2 < 1e-4 { return 1 - x2 / 6 + x2 * x2 / 120 }
    return sin(x) / x
}

/// The counter-clockwise turn from angle a to angle b, in [0, 2π).
func ccw(_ a: Float, _ b: Float) -> Float {
    var d: Float = (b - a).truncatingRemainder(dividingBy: 2 * Float.pi)
    if d < -1e-5 { d += 2 * Float.pi }
    if d > 2 * Float.pi - 1e-5 { d -= 2 * Float.pi }
    return max(d, 0)
}

/// Leaf k's curve in state s: its gutter arc, then its body.
func leafPieces(_ k: Int, _ s: BookState, _ m: Mutant = activeMutant) -> [Piece] {
    let t: Float = paperCaliper
    let sigma: Float = (Float(k) + 0.5) * t
    let rho: Float = spineLength - sigma
    let d = SIMD2<Float>(cos(s.psi), sin(s.psi))
    let root: SIMD2<Float> = s.jb + d * sigma
    let rootAngle: Float = s.psi - Float.pi / 2
    // How far the leaf's body turns: from the back board's direction (β)
    // counter-clockwise to the front's (α).
    let sweep: Float = ccw(s.beta, s.alpha)
    let tau: Float = s.tau[k]
    let body: Float = s.beta + sweep * tau
    let turn: Float = body - rootAngle                      // the gutter's turn, signed
    var gutterLen: Float
    var gutterK: Float
    if turn >= 0 {
        gutterLen = rho * turn
        gutterK = rho > 1e-6 ? 1 / rho : 0
    } else {
        gutterLen = sigma * -turn
        gutterK = -1 / sigma
    }
    var width: Float = leafWidth
    if m == .stretchyPage { width *= 1 + 0.12 * sin(Float.pi * tau) }
    let rest: Float = width - gutterLen
    var curl: Float = -curlMax * sin(Float.pi * tau)
    if m == .ghostPage && k == leafCount / 2 { curl = 3.0 * sin(Float.pi * tau) }
    var pieces: [Piece] = []
    if gutterLen > 1e-6 { pieces.append(Piece(start: root, angle: rootAngle, curvature: gutterK, length: gutterLen)) }
    let bodyStart: SIMD2<Float> = pieces.last?.end ?? root
    let bodyK: Float = rest > 1e-6 ? curl / rest : 0
    pieces.append(Piece(start: bodyStart, angle: body, curvature: bodyK, length: rest))
    return pieces
}

/// Points along a leaf's curve every `step` mm and every 0.02 rad of turn
/// (so a tight gutter arc's chords stay within 0.005% of its radius), and
/// its end.
func leafPoints(_ pieces: [Piece], step: Float) -> [SIMD2<Float>] {
    var out: [SIMD2<Float>] = []
    for p in pieces {
        let byLength: Int = Int((p.length / step).rounded(.up))
        let byTurn: Int = Int((abs(p.curvature * p.length) / 0.02).rounded(.up))
        let n: Int = max(byLength, byTurn, 1)
        for i in 0..<n { out.append(p.point(p.length * Float(i) / Float(n))) }
    }
    if let last = pieces.last { out.append(last.end) }
    return out
}

// MARK: - the scene at a moment

/// A board's frame: origin at its joint, `u` along it towards the fore-edge,
/// `n` towards the leaves.
struct BoardFrame {
    var joint: SIMD2<Float>
    var u: SIMD2<Float>
    var n: SIMD2<Float>
    func world(_ along: Float, _ across: Float) -> SIMD2<Float> { joint + u * along + n * across }
}

func frontFrame(_ s: BookState) -> BoardFrame {
    let u = SIMD2<Float>(cos(s.alpha), sin(s.alpha))
    return BoardFrame(joint: s.jf, u: u, n: SIMD2<Float>(u.y, -u.x))
}

func backFrame(_ s: BookState) -> BoardFrame {
    let u = SIMD2<Float>(cos(s.beta), sin(s.beta))
    return BoardFrame(joint: s.jb, u: u, n: SIMD2<Float>(-u.y, u.x))
}

/// A board's cloth-covered slab in its frame: along [g, end], across
/// [−boardStack, −pastedown].
struct BoardBox {
    var centre: SIMD2<Float>
    var axis: SIMD2<Float>
    var half: SIMD2<Float>
}

func boardBox(_ f: BoardFrame, _ m: Mutant = activeMutant) -> BoardBox {
    let a0: Float = jointGap
    let a1: Float = boardEnd(m)
    let c0: Float = -boardStack
    let c1: Float = -pastedownThickness
    let c: SIMD2<Float> = f.world((a0 + a1) / 2, (c0 + c1) / 2)
    // The box's own axes: x along u, y along the box's second axis, which
    // the kernel takes as u turned +90°.
    return BoardBox(centre: c, axis: f.u, half: SIMD2<Float>((a1 - a0) / 2, (c1 - c0) / 2))
}

/// The case spine's strip: parallel to the spine, outside it by the hollow.
func spineBox(_ s: BookState) -> BoardBox {
    let d = SIMD2<Float>(cos(s.psi), sin(s.psi))
    let o = SIMD2<Float>(-d.y, d.x)
    let mid: SIMD2<Float> = (s.jb + s.jf) / 2 + o * (spineHollow + spineStrip / 2)
    let halfLen: Float = spineLength / 2 + boardStack - spineStrip / 2
    return BoardBox(centre: mid, axis: d, half: SIMD2<Float>(halfLen, spineStrip / 2))
}

/// The scene the kernel draws at state s.
func buildScene(_ s: BookState, _ m: Mutant = activeMutant) -> Scene {
    var scene = Scene()
    let hz: Float = caseHalfHeight(m)
    let e: Float = shellThickness
    let r: Float = boardEdgeRadius + e
    let back: BoardFrame = backFrame(s)
    let front: BoardFrame = frontFrame(s)
    for (f, nm) in [(back, "back"), (front, "front")] {
        let b: BoardBox = boardBox(f, m)
        scene.add(boxPrim(centre: b.centre, half: b.half, radius: r, z0: -hz, z1: hz, zRound: r, material: .cloth, axis: b.axis),
                  "\(nm) board")
        // The pastedown, the squares' width short of the board's edges.
        let p0: Float = jointGap + 1.0
        let p1: Float = boardEnd(m) - squares(m)
        let pc: SIMD2<Float> = f.world((p0 + p1) / 2, -pastedownThickness / 2)
        scene.add(boxPrim(centre: pc, half: SIMD2<Float>((p1 - p0) / 2, pastedownThickness / 2), radius: 0,
                          z0: -(leafHeight / 2), z1: leafHeight / 2, zRound: 0, material: .pastedown, axis: f.u), "\(nm) pastedown")
    }
    let sp: BoardBox = spineBox(s)
    scene.add(boxPrim(centre: sp.centre, half: sp.half, radius: spineStrip / 2, z0: -hz, z1: hz, zRound: spineStrip / 2,
                      material: .cloth, axis: sp.axis), "case spine")

    // The joints: cloth from each board's spine edge down into a groove and
    // across to the case spine's end.
    let d = SIMD2<Float>(cos(s.psi), sin(s.psi))
    let o = SIMD2<Float>(-d.y, d.x)
    let cr: Float = clothThickness / 2
    let spineOut: Float = spineHollow + spineStrip / 2
    let ends: [(BoardFrame, SIMD2<Float>, String)] = [
        (back, s.jb - d * (boardStack - spineStrip / 2) + o * spineOut, "back"),
        (front, s.jf + d * (boardStack - spineStrip / 2) + o * spineOut, "front"),
    ]
    for (f, spineEnd, nm) in ends {
        let outer: Float = -boardStack + cr
        let edge: SIMD2<Float> = f.world(jointGap + e, outer)
        let groove: SIMD2<Float> = f.world(jointGap * 0.45, outer + grooveDepthFraction * boardThickness)
        scene.add(capsulePrim(edge, groove, radius: cr, z0: -hz, z1: hz, zRound: cr, material: .cloth), "\(nm) joint, board side")
        scene.add(capsulePrim(groove, spineEnd, radius: cr, z0: -hz, z1: hz, zRound: cr, material: .cloth), "\(nm) joint, spine side")
    }

    // Headbands: cords along the spine at head and tail, just inside it.
    for z in [leafHeight / 2, -leafHeight / 2] {
        let a: SIMD2<Float> = s.jb + d * 0.6 + o * 0.1
        let b: SIMD2<Float> = s.jf - d * 0.6 + o * 0.1
        scene.add(capsule3Prim(SIMD3<Float>(a.x, a.y, z), SIMD3<Float>(b.x, b.y, z), radius: headbandRadius * 0.55,
                               material: .headband), z > 0 ? "head headband" : "tail headband")
    }

    // The leaves.
    let z0: Float = -leafHeight / 2
    let z1: Float = leafHeight / 2
    let t: Float = paperCaliper
    let lying: [Int] = (0..<leafCount).filter { s.tau[$0] <= 0 }
    let turned: [Int] = (0..<leafCount).filter { s.tau[$0] >= 1 }
    let flying: [Int] = (0..<leafCount).filter { s.tau[$0] > 0 && s.tau[$0] < 1 }
    // Lying leaves are 0..<nLying (they lift from the top down); turned
    // leaves are the top ones.
    let nLying: Int = lying.count
    let nTurned: Int = turned.count
    if nLying > 0 {
        addStack(&scene, centre: s.jb, frame: back, count: nLying, fromAngle: s.psi, sweep: -(s.psi - s.beta - Float.pi / 2),
                 z0: z0, z1: z1, name: "leaves lying on the back half")
    }
    if nTurned > 0 {
        let sweep: Float = ccw(s.psi + Float.pi, s.alpha - Float.pi / 2)
        addStack(&scene, centre: s.jf, frame: front, count: nTurned, fromAngle: s.psi + Float.pi, sweep: sweep,
                 z0: z0, z1: z1, name: "leaves turned onto the front half")
    }
    for k in flying {
        for (i, p) in leafPieces(k, s, m).enumerated() {
            scene.add(piecePrim(p, thickness: t, z0: z0, z1: z1), "leaf \(k) piece \(i)")
        }
    }
    return scene
}

/// A stack of `count` leaves lying concentric about a joint: a ring sector
/// of radius 0…count × t turning `sweep` from `fromAngle`, then their
/// straight parts along the board, a trapezoid (each leaf's straight part is
/// shorter by what its arc used).
func addStack(_ scene: inout Scene, centre: SIMD2<Float>, frame f: BoardFrame, count: Int, fromAngle a0: Float, sweep: Float,
              z0: Float, z1: Float, name: String) {
    let t: Float = paperCaliper
    let depth: Float = Float(count) * t
    let absSweep: Float = abs(sweep)
    if absSweep > 1e-5 {
        let midA: Float = a0 + sweep / 2
        scene.add(ringPrim(centre: centre, mid: SIMD2<Float>(cos(midA), sin(midA)), radius: depth / 2, thickness: depth,
                           halfAngle: absSweep / 2, z0: z0, z1: z1, zRound: 0, material: .paper), "\(name), gutter")
    }
    let w: Float = leafWidth
    let quad: [SIMD2<Float>] = [f.world(0, 0), f.world(w, 0), f.world(w - depth * absSweep, depth), f.world(0, depth)]
    scene.addPolygon(quad, z0: z0, z1: z1, zRound: 0, material: .paper, name)
}

/// A piece of a flying leaf as a primitive: a thick arc, stored by its
/// chord (so a nearly straight arc of huge radius keeps its precision), or a
/// capsule when it is straight to float precision.
func piecePrim(_ p: Piece, thickness t: Float, z0: Float, z1: Float) -> GPrim {
    let bend: Float = abs(p.curvature * p.length)
    if bend < 1e-6 {
        return capsulePrim(p.start, p.end, radius: t / 2, z0: z0, z1: z1, zRound: 0, material: .paper)
    }
    let r: Float = 1 / abs(p.curvature)
    let side: Float = p.curvature > 0 ? 1 : -1
    let chordAngle: Float = p.angle + p.curvature * p.length / 2
    let chordDir = SIMD2<Float>(cos(chordAngle), sin(chordAngle))
    // The arc bulges away from its centre, which is on the curving side.
    let mid: SIMD2<Float> = SIMD2<Float>(-chordDir.y, chordDir.x) * -side
    let chordMid: SIMD2<Float> = p.start + chordDir * (p.length / 2 * sinc(bend / 2))
    return arcPrim(chordMid: chordMid, mid: mid, radius: r, thickness: t, halfAngle: bend / 2, z0: z0, z1: z1, material: .paper)
}
