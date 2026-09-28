// The book: a plain clothbound hardback, closed, lying on its back cover,
// built from the construction bookbinders describe — a text block of
// leaves, rounded at the spine, cased in two boards and a spine strip
// covered in dyed cotton cloth, with a French (open) joint groove between
// each board and the spine, squares all round, and headbands.
//
// Everything is in millimetres. The book lies with its spine along the z
// axis at x ≈ 0, its fore-edge towards +x, its head towards +z, on a table
// at y = 0.
//
// Sources, every one read for this step:
//   [Wiki-size]  Wikipedia, "Book size": the octavo row of the table,
//                6 × 9 in; octavo = 8 leaves (16 pages) per gathering.
//   [Case]       enterprise-press.com "paper thickness chart" (Case Paper
//                chart, as cited by Wikipedia "Paper"): offset, regular
//                finish, 50 lb basis → caliper 0.004 in per sheet.
//   [Wiki-paper] Wikipedia, "Paper": "Paper may be between 0.07 and
//                0.18 mm thick."
//   [Bailey]     A. L. Bailey, Library Bookbinding (1916), Project Gutenberg
//                #38387: "the boards must project an eighth of an inch on
//                all edges (except the back) forming what is called the
//                'squares'"; cloth "practically all ... made of cotton";
//                dark blue among the standard shades; "advisable ... to
//                cover books with the warp running across the cover".
//   [Bean]       F. O. Bean & J. C. Brodhead, Bookbinding for Beginners
//                (1914), Gutenberg #68844: the cover "should project beyond
//                the pages from ⅛ to ¼ of an inch"; a "½ inch lap to fold
//                over" the board's edges (the turn-in).
//   [E&R]        Etherington & Roberts, Bookbinding and the Conservation of
//                Books: A Dictionary of Descriptive Terminology (AIC/CoOL):
//                "binder's board" — machine boards "range in thickness from
//                0.030 to 0.300 inch"; "joint" — "the exterior juncture of
//                the spine and covers"; "French joint" — the board set
//                "approximately 1/8 to 1/4 inch ... away from the backing
//                shoulder"; "rounding" — the spine molded "into an arc of
//                approximately one-third of a circle, which ... produces the
//                characteristic concave fore edge"; "headband" — "projects
//                slightly beyond the head and tail"; "book cloth" — Group B
//                (medium) cloth: warp not less than 104, filling not less
//                than 77 threads per inch; "turn-ins" — covering material
//                "turned over the edges of the board and glued to the inside
//                surface"; "squares" — "the marginal difference between the
//                edges of the text block and the edges of the case".

import Foundation
import simd

// MARK: - mutants

/// Ways to break the step on purpose; each must make the suite fail.
enum Mutant: String {
    case none
    case noSquares          // the case cut flush with the text block
    case plasticSheen       // the cloth given a dielectric plastic's gloss
    case pagesNotLeaves     // the text block's thickness from pages, not leaves
}

let activeMutant: Mutant = Mutant(rawValue: ProcessInfo.processInfo.environment["BOOK_MUTANT"] ?? "") ?? .none

// MARK: - sourced constants

let millimetresPerInch: Float = 25.4

/// The leaf: an octavo, 6 × 9 in [Wiki-size].
let leafWidth: Float = 6 * millimetresPerInch          // 152.4 mm, spine to fore-edge
let leafHeight: Float = 9 * millimetresPerInch         // 228.6 mm, head to tail

/// Pages in the book. MODEL: 192 pages is a slim novel, and a whole number
/// of octavo gatherings (12 × 16 pages) [Wiki-size].
let pageCount: Int = 192
/// Two pages to a leaf.
let leafCount: Int = pageCount / 2

/// One leaf's thickness: 50 lb offset (a common book paper; the choice of
/// grade is MODEL), 0.004 in [Case] — inside Wikipedia's 0.07–0.18 mm
/// [Wiki-paper].
let paperCaliper: Float = 0.004 * millimetresPerInch   // 0.1016 mm
let paperCaliperRange: ClosedRange<Float> = 0.07...0.18

/// The text block's thickness at the fore-edge: leaves × caliper. The
/// `pagesNotLeaves` mutant makes the classic mistake of pages × caliper.
func textBlockThickness(_ m: Mutant = activeMutant) -> Float {
    let n: Int = m == .pagesNotLeaves ? pageCount : leafCount
    let nf: Float = Float(n)
    return nf * paperCaliper
}

/// The squares: ⅛ in [Bailey], the low end of [Bean]'s ⅛–¼ in.
func squares(_ m: Mutant = activeMutant) -> Float {
    m == .noSquares ? 0 : 0.125 * millimetresPerInch   // 3.175 mm
}
let squaresRange: ClosedRange<Float> = (0.125 * millimetresPerInch)...(0.25 * millimetresPerInch)

/// Binder's board: 2.5 mm, MODEL — a common greyboard; E&R's range for
/// machine boards is 0.030–0.300 in (0.76–7.62 mm) [E&R].
let boardThickness: Float = 2.5
let boardThicknessRange: ClosedRange<Float> = (0.030 * millimetresPerInch)...(0.300 * millimetresPerInch)

/// The joint's gap between board and backing shoulder: ⅛ in, the low end of
/// E&R's French joint, 1/8–1/4 in [E&R].
let jointGap: Float = 0.125 * millimetresPerInch
let jointGapRange: ClosedRange<Float> = (0.125 * millimetresPerInch)...(0.25 * millimetresPerInch)

/// The round of the spine: a third of a circle, 120° of arc [E&R].
let spineRoundDegrees: Float = 120

/// Book cloth, Group B (medium): at least 104 warp and 77 filling threads
/// per inch [E&R]. The warp runs across the cover [Bailey].
let warpPerInch: Float = 104
let fillingPerInch: Float = 77
var warpPitch: Float { millimetresPerInch / warpPerInch }         // 0.244 mm, between warp threads
var fillingPitch: Float { millimetresPerInch / fillingPerInch }   // 0.330 mm, between filling threads

/// The turn-in: ½ in of cloth folded over the board's edges onto its inside
/// [Bean]. Only the part inside the squares is ever seen.
let turnIn: Float = 0.5 * millimetresPerInch

// MARK: - model constants (each MODEL, with the reason)

/// The cloth's thickness: MODEL. No measured figure reached; two crossing
/// threads of a medium cotton cloth, ~0.3 mm.
let clothThickness: Float = 0.30
/// The glue film under the cloth and under the pastedown: MODEL, 50 µm.
let adhesiveThickness: Float = 0.05
/// The pastedown (the endpaper's half glued to the board): MODEL, the same
/// paper as the text.
let pastedownThickness: Float = 0.004 * millimetresPerInch
/// The board's own edge radius under the cloth: MODEL, a cut board's edge
/// crushed a little by the casing-in.
let boardEdgeRadius: Float = 0.25
/// The hollow between the text block's spine and the case spine: MODEL.
let spineHollow: Float = 0.5
/// The case spine's stiffener (inlay), under the cloth: MODEL, thin card.
let spineInlay: Float = 0.4
/// The headband: a round cord, radius 1.0 mm, so it "projects slightly
/// beyond the head and tail" [E&R] by its own radius. MODEL size.
let headbandRadius: Float = 1.0
/// The headband's stripes, each this long along the spine. MODEL.
let headbandStripe: Float = 0.9

/// How deep the joint's groove sinks below the board's outer face, as a
/// fraction of the board's thickness. MODEL: a groove pressed in by the
/// casing-in, visible as a channel.
let grooveDepthFraction: Float = 0.6

// MARK: - the layout, derived

/// The cloth-and-glue shell over the board.
let shellThickness: Float = clothThickness + adhesiveThickness

struct BookLayout {
    let mutant: Mutant
    /// Text block: thickness, bottom and top face heights.
    let T: Float
    let y0: Float
    let y1: Float
    /// The squares.
    let s: Float
    /// The case's outer faces, bottom (0) and top.
    let top: Float
    /// Spine round: the circle's centre (x) and radius, and its sagitta.
    let spineCentreX: Float
    let spineRadius: Float
    let sag: Float

    init(_ m: Mutant = activeMutant) {
        mutant = m
        T = textBlockThickness(m)
        s = squares(m)
        let board: Float = boardThickness + 2 * shellThickness
        y0 = board + pastedownThickness
        y1 = y0 + T
        top = y1 + pastedownThickness + board
        // A 120° arc on the chord T: R = T / (2 sin 60°), sagitta R(1 − cos 60°).
        let half: Float = spineRoundDegrees / 2 * .pi / 180
        spineRadius = T / (2 * sin(half))
        sag = spineRadius * (1 - cos(half))
        spineCentreX = spineRadius - sag
    }

    /// The text block's middle height.
    var ym: Float { (y0 + y1) / 2 }

    /// How far left of the shoulders the round bulges at height y: the
    /// spine's arc (and, shifted by the leaf's width, the fore-edge's).
    func bulge(_ y: Float) -> Float {
        let dy: Float = y - ym
        let r2: Float = spineRadius * spineRadius
        let inside: Float = max(r2 - dy * dy, 0)
        return inside.squareRoot() - spineCentreX
    }

    /// The case's outline, x: board's spine edge (after the joint gap) and
    /// fore-edge (the leaves' fore-edge plus the squares).
    var boardX0: Float { jointGap }
    var boardX1: Float { leafWidth + s }
    /// And z: head and tail, the leaves' plus the squares.
    var caseHalfHeight: Float { leafHeight / 2 + s }
}

// MARK: - the scene as primitives for the kernel

/// A primitive: a 2D shape in the (x, y) cross-section extruded along z, or
/// a 3D shape. The kernel's distance function is the minimum over the
/// visible primitives; internal ones (board core, glue) are only read by the
/// probe, to draw the inset.
enum PrimKind: Int32 {
    case capsule = 0       // a: p0.xy p1.xy; b.x radius
    case box = 1           // a: centre.xy axis.xy; b: half.xy, corner radius
    case ring = 2          // a: centre.xy middle direction.xy; b: mid radius, thickness, half-angle; c: sin, cos
    case polygon = 3       // vertices in the vertex buffer
    case arcTube = 6       // a: centre.xy middle direction.xy; b: radius, tube radius, half-angle, z; c: sin, cos
}

/// Materials. The inset colours by the same numbers.
enum Material: Int32 {
    case air = 0
    case table = 1
    case cloth = 2
    case paper = 3
    case headband = 4
    case board = 5
    case adhesive = 6
    case pastedown = 7
}

struct GPrim {
    var a: SIMD4<Float>
    var b: SIMD4<Float>
    var c: SIMD4<Float>
    var z: SIMD4<Float>         // z0, z1, edge rounding, vertex count
    var bound: SIMD4<Float>     // centre, radius
    var info: SIMD4<Int32>      // kind, material, internal (1) or visible (0), first vertex
}

struct Scene {
    var prims: [GPrim] = []
    var verts: [SIMD2<Float>] = []
    var names: [String] = []

    mutating func add(_ p: GPrim, _ name: String) {
        prims.append(p)
        names.append(name)
    }
}

/// A 2D rounded box extruded between z0 and z1 with its 3D edges rounded.
func boxPrim(centre: SIMD2<Float>, half: SIMD2<Float>, radius: Float, z0: Float, z1: Float, zRound: Float,
             material: Material, hidden: Bool = false) -> GPrim {
    let hz: Float = (z1 - z0) / 2
    let r3: Float = (half.x * half.x + half.y * half.y + hz * hz).squareRoot()
    return GPrim(a: SIMD4<Float>(centre.x, centre.y, 1, 0), b: SIMD4<Float>(half.x, half.y, radius, 0), c: .zero,
                 z: SIMD4<Float>(z0, z1, zRound, 0),
                 bound: SIMD4<Float>(centre.x, centre.y, (z0 + z1) / 2, r3 + 0.01),
                 info: SIMD4<Int32>(PrimKind.box.rawValue, material.rawValue, hidden ? 1 : 0, 0))
}

func capsulePrim(_ p0: SIMD2<Float>, _ p1: SIMD2<Float>, radius: Float, z0: Float, z1: Float, zRound: Float,
                 material: Material) -> GPrim {
    let c: SIMD2<Float> = (p0 + p1) / 2
    let hz: Float = (z1 - z0) / 2
    let hl: Float = simd_distance(p0, p1) / 2 + radius
    let r3: Float = (hl * hl + hz * hz).squareRoot()
    return GPrim(a: SIMD4<Float>(p0.x, p0.y, p1.x, p1.y), b: SIMD4<Float>(radius, 0, 0, 0), c: .zero,
                 z: SIMD4<Float>(z0, z1, zRound, 0), bound: SIMD4<Float>(c.x, c.y, (z0 + z1) / 2, r3 + 0.01),
                 info: SIMD4<Int32>(PrimKind.capsule.rawValue, material.rawValue, 0, 0))
}

/// An annular sector about `centre`, symmetric about direction `mid`.
func ringPrim(centre: SIMD2<Float>, mid: SIMD2<Float>, radius: Float, thickness: Float, halfAngle: Float,
              z0: Float, z1: Float, zRound: Float, material: Material) -> GPrim {
    let hz: Float = (z1 - z0) / 2
    let ro: Float = radius + thickness / 2
    let r3: Float = (ro * ro + hz * hz).squareRoot()
    return GPrim(a: SIMD4<Float>(centre.x, centre.y, mid.x, mid.y), b: SIMD4<Float>(radius, thickness, halfAngle, 0),
                 c: SIMD4<Float>(sin(halfAngle), cos(halfAngle), 0, 0), z: SIMD4<Float>(z0, z1, zRound, 0),
                 bound: SIMD4<Float>(centre.x, centre.y, (z0 + z1) / 2, r3 + 0.01),
                 info: SIMD4<Int32>(PrimKind.ring.rawValue, material.rawValue, 0, 0))
}

func arcTubePrim(centre: SIMD2<Float>, mid: SIMD2<Float>, radius: Float, tube: Float, halfAngle: Float, z: Float,
                 material: Material) -> GPrim {
    let ro: Float = radius + tube
    return GPrim(a: SIMD4<Float>(centre.x, centre.y, mid.x, mid.y), b: SIMD4<Float>(radius, tube, halfAngle, z),
                 c: SIMD4<Float>(sin(halfAngle), cos(halfAngle), 0, 0), z: .zero,
                 bound: SIMD4<Float>(centre.x, centre.y, z, ro + tube + 0.01),
                 info: SIMD4<Int32>(PrimKind.arcTube.rawValue, material.rawValue, 0, 0))
}

extension Scene {
    /// A polygon (counter-clockwise or not; the kernel's test is winding-free)
    /// extruded between z0 and z1.
    mutating func addPolygon(_ vs: [SIMD2<Float>], z0: Float, z1: Float, zRound: Float, material: Material, _ name: String) {
        var lo = SIMD2<Float>(repeating: .infinity)
        var hi = SIMD2<Float>(repeating: -.infinity)
        for v in vs { lo = simd_min(lo, v); hi = simd_max(hi, v) }
        let c: SIMD2<Float> = (lo + hi) / 2
        let e: SIMD2<Float> = (hi - lo) / 2
        let hz: Float = (z1 - z0) / 2
        let r3: Float = (e.x * e.x + e.y * e.y + hz * hz).squareRoot()
        let first: Int32 = Int32(verts.count)
        verts += vs
        add(GPrim(a: .zero, b: .zero, c: .zero, z: SIMD4<Float>(z0, z1, zRound, Float(vs.count)),
                  bound: SIMD4<Float>(c.x, c.y, (z0 + z1) / 2, r3 + 0.01),
                  info: SIMD4<Int32>(PrimKind.polygon.rawValue, material.rawValue, 0, first)), name)
    }
}

/// A board: its cloth shell (what the camera sees), and, for the inset only,
/// the glue film and the greyboard core inside it. `y` is the core's bottom.
func addBoard(_ scene: inout Scene, _ L: BookLayout, coreBottom y: Float, name: String) {
    let x0: Float = L.boardX0 + shellThickness
    let x1: Float = L.boardX1 - shellThickness
    let cx: Float = (x0 + x1) / 2
    let hx: Float = (x1 - x0) / 2
    let hy: Float = boardThickness / 2
    let cy: Float = y + hy
    let hz: Float = L.caseHalfHeight - shellThickness
    let r: Float = boardEdgeRadius
    let e: Float = shellThickness
    let a: Float = adhesiveThickness
    scene.add(boxPrim(centre: SIMD2<Float>(cx, cy), half: SIMD2<Float>(hx + e, hy + e), radius: r + e,
                      z0: -hz - e, z1: hz + e, zRound: r + e, material: .cloth), "\(name) board, in its cloth")
    scene.add(boxPrim(centre: SIMD2<Float>(cx, cy), half: SIMD2<Float>(hx + a, hy + a), radius: r + a,
                      z0: -hz - a, z1: hz + a, zRound: r + a, material: .adhesive, hidden: true), "\(name) glue film")
    scene.add(boxPrim(centre: SIMD2<Float>(cx, cy), half: SIMD2<Float>(hx, hy), radius: r,
                      z0: -hz, z1: hz, zRound: r, material: .board, hidden: true), "\(name) greyboard")
}

/// Arc points from angle a0 to a1 (radians, from +x) about c, radius r.
func arcPoints(_ c: SIMD2<Float>, _ r: Float, from a0: Float, to a1: Float, count n: Int) -> [SIMD2<Float>] {
    (0...n).map { i in
        let u: Float = Float(i) / Float(n)
        let a: Float = a0 + (a1 - a0) * u
        return c + SIMD2<Float>(cos(a), sin(a)) * r
    }
}

/// The whole closed book.
func buildBook(_ L: BookLayout = BookLayout()) -> Scene {
    var scene = Scene()
    let e: Float = shellThickness

    // The boards: back on the table, front on top.
    addBoard(&scene, L, coreBottom: e, name: "back")
    addBoard(&scene, L, coreBottom: L.top - e - boardThickness, name: "front")

    // The pastedowns, glued to each board's inside, the squares' width short
    // of every edge so the turn-in shows in the squares (MODEL placement).
    let pdx0: Float = L.boardX0 + 1.0
    let pdx1: Float = L.boardX1 - L.s
    let pdHalfZ: Float = L.caseHalfHeight - L.s
    for (yb, nm) in [(L.y0 - pastedownThickness, "back"), (L.y1, "front")] {
        let c = SIMD2<Float>((pdx0 + pdx1) / 2, yb + pastedownThickness / 2)
        scene.add(boxPrim(centre: c, half: SIMD2<Float>((pdx1 - pdx0) / 2, pastedownThickness / 2), radius: 0,
                          z0: -pdHalfZ, z1: pdHalfZ, zRound: 0, material: .pastedown), "\(nm) pastedown")
    }

    // The text block: flat head and tail, its spine rounded to a third of a
    // circle bulging left of the shoulders at x = 0, and the fore-edge
    // concave by the same arc [E&R, "rounding"].
    let half: Float = spineRoundDegrees / 2 * .pi / 180
    let cSpine = SIMD2<Float>(L.spineCentreX, L.ym)
    let cFore = SIMD2<Float>(L.spineCentreX + leafWidth, L.ym)
    var tb: [SIMD2<Float>] = []
    // Spine arc, from the top shoulder round the left to the bottom one.
    tb += arcPoints(cSpine, L.spineRadius, from: .pi - half, to: .pi + half, count: 24)
    // Fore-edge arc, bottom to top (the same circle, shifted by the leaf).
    tb += arcPoints(cFore, L.spineRadius, from: .pi + half, to: .pi - half, count: 24)
    scene.addPolygon(tb, z0: -leafHeight / 2, z1: leafHeight / 2, zRound: 0, material: .paper, "text block")

    // The case spine: cloth over a thin inlay, an arc standing off the text
    // block's spine by the hollow, from the back board's outer face to the
    // front's. Its circle passes through the two joints and the bulge.
    let rim: Float = spineInlay + clothThickness
    let xOut: Float = -L.sag - spineHollow - rim / 2        // mid-surface at the bulge
    let xJoint: Float = -0.2                                 // mid-surface at the joints (MODEL)
    let yLo: Float = rim / 2
    let yHi: Float = L.top - rim / 2
    // Circle through (xJoint, yLo), (xJoint, yHi), (xOut, ym').
    let ymid: Float = (yLo + yHi) / 2
    let hh: Float = (yHi - yLo) / 2
    let dx: Float = xJoint - xOut
    let R: Float = (hh * hh + dx * dx) / (2 * dx)
    let caseCentre = SIMD2<Float>(xOut + R, ymid)
    let caseHalf: Float = asin(min(hh / R, 1))
    scene.add(ringPrim(centre: caseCentre, mid: SIMD2<Float>(-1, 0), radius: R, thickness: rim, halfAngle: caseHalf,
                       z0: -L.caseHalfHeight, z1: L.caseHalfHeight, zRound: rim / 2, material: .cloth), "case spine")

    // The joints: the cloth runs from the board's spine edge down into the
    // groove and up to the case spine's edge [E&R, "French joint"]. Two
    // strips of cloth each side.
    let depth: Float = grooveDepthFraction * boardThickness
    let cr: Float = clothThickness / 2
    for side in 0..<2 {
        let outer: Float = side == 0 ? cr : L.top - cr
        let inward: Float = side == 0 ? 1 : -1
        let boardEdge = SIMD2<Float>(L.boardX0 + e, outer)
        let groove = SIMD2<Float>(L.boardX0 * 0.45, outer + inward * depth)
        let spineEdge = SIMD2<Float>(xJoint + 0.1, outer)
        let nm: String = side == 0 ? "back" : "front"
        scene.add(capsulePrim(boardEdge, groove, radius: cr, z0: -L.caseHalfHeight, z1: L.caseHalfHeight, zRound: cr,
                              material: .cloth), "\(nm) joint, board side")
        scene.add(capsulePrim(groove, spineEdge, radius: cr, z0: -L.caseHalfHeight, z1: L.caseHalfHeight, zRound: cr,
                              material: .cloth), "\(nm) joint, spine side")
    }

    // Headbands at head and tail: cords along the spine's round, their
    // centres at the leaves' edge so each projects its own radius beyond.
    let hbRadius: Float = L.spineRadius - headbandRadius * 0.8
    for z in [leafHeight / 2, -leafHeight / 2] {
        scene.add(arcTubePrim(centre: cSpine, mid: SIMD2<Float>(-1, 0), radius: hbRadius, tube: headbandRadius,
                              halfAngle: half * 0.92, z: z, material: .headband), z > 0 ? "head headband" : "tail headband")
    }
    return scene
}
