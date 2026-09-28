// The book: step 70's plain clothbound hardback — the same octavo, the same
// 96 leaves of 50 lb offset, the same squares, boards, cloth and joints —
// now able to open. What moves is in Flip.swift; this file holds the
// sourced construction and the primitives the kernel draws.
//
// One change from step 70, a judgement call: while the book moves, its
// spine is drawn flat-backed (the roots of the leaves on a straight line),
// not rounded, so the text block can flex at the spine as the book opens.
// A rounded spine's leaves would have to slide past each other at the
// gutter, which this model does not attempt.
//
// Everything is in millimetres. The spine runs along the z axis near
// x = 0, the head towards +z, on a table at y = 0.
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
    case stretchyPage       // a turning page that stretches as it flies
    case frozen             // nothing moves
    case rewind             // the pages turn backwards
    case ghostPage          // one page curls through its neighbours
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

/// The text block's thickness: leaves × caliper.
func textBlockThickness() -> Float {
    let nf: Float = Float(leafCount)
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
/// A board's whole thickness, cloth both sides, with its pastedown: from
/// the table to the first leaf when the book lies closed.
let boardStack: Float = boardThickness + 2 * shellThickness + pastedownThickness
/// The case spine: inlay and cloth.
let spineStrip: Float = spineInlay + clothThickness
/// Where the board ends, measured along it from the joint: the leaf's width
/// plus the square.
func boardEnd(_ m: Mutant = activeMutant) -> Float { leafWidth + squares(m) }
/// The case's half-height: the leaves' plus the squares.
func caseHalfHeight(_ m: Mutant = activeMutant) -> Float { leafHeight / 2 + squares(m) }

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
    case arc = 4           // a thick arc with round ends, by its chord: see arcPrim
    case capsule3 = 5      // 3D: a.xyz to b.xyz, radius c.x
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
             material: Material, hidden: Bool = false, axis: SIMD2<Float> = SIMD2<Float>(1, 0)) -> GPrim {
    let hz: Float = (z1 - z0) / 2
    let r3: Float = (half.x * half.x + half.y * half.y + hz * hz).squareRoot()
    return GPrim(a: SIMD4<Float>(centre.x, centre.y, axis.x, axis.y), b: SIMD4<Float>(half.x, half.y, radius, 0), c: .zero,
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

/// A thick arc of a circle with round ends, stored by its chord: the
/// chord's middle, the direction from it to the arc's middle, the radius
/// and the half-angle. The kernel works relative to the chord, never to a
/// far-off centre, so a leaf curled only a little keeps float precision.
func arcPrim(chordMid: SIMD2<Float>, mid: SIMD2<Float>, radius r: Float, thickness: Float, halfAngle h: Float,
             z0: Float, z1: Float, material: Material) -> GPrim {
    let halfChord: Float = r * sin(h)
    let sag: Float = 2 * r * sin(h / 2) * sin(h / 2)
    let reach: Float = (halfChord * halfChord + sag * sag).squareRoot() + thickness
    let hz: Float = (z1 - z0) / 2
    let r3: Float = (reach * reach + hz * hz).squareRoot()
    return GPrim(a: SIMD4<Float>(chordMid.x, chordMid.y, mid.x, mid.y), b: SIMD4<Float>(r, thickness, h, sag),
                 c: SIMD4<Float>(sin(h), cos(h), halfChord, r * cos(h)), z: SIMD4<Float>(z0, z1, 0, 0),
                 bound: SIMD4<Float>(chordMid.x, chordMid.y, (z0 + z1) / 2, r3 + 0.01),
                 info: SIMD4<Int32>(PrimKind.arc.rawValue, material.rawValue, 0, 0))
}

func capsule3Prim(_ p0: SIMD3<Float>, _ p1: SIMD3<Float>, radius: Float, material: Material) -> GPrim {
    let c: SIMD3<Float> = (p0 + p1) / 2
    return GPrim(a: SIMD4<Float>(p0, 0), b: SIMD4<Float>(p1, 0), c: SIMD4<Float>(radius, 0, 0, 0), z: .zero,
                 bound: SIMD4<Float>(c, simd_distance(p0, p1) / 2 + radius + 0.01),
                 info: SIMD4<Int32>(PrimKind.capsule3.rawValue, material.rawValue, 0, 0))
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

