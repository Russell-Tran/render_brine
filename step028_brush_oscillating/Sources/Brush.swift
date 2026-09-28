// A round electric toothbrush head, new, pressed against the buccal face of
// one lower premolar at the gum line — in step 20's mouth, under step 20's
// camera and light, which are used unchanged.
//
// STEP 28: this file is step 22's Brush.swift, copied, so that step 22 stays
// exactly as committed. What step 28 added, and nothing else: the disc can be
// turned about its own axis by `spin` radians — the tufts and the hexagonal
// boss on its back turn with it, the neck does not — through `tuftLayout`,
// `spinFrame`, `BrushPose.spinAlong/spinAcross` and the kernel's
// `BR_SPIN_ALONG/ACROSS`. At spin 0 every one of those is bit-for-bit the
// value step 22 used, and step 22's `pressedBrush` is kept whole as the
// reference; the tests hold the spin-0 frame to step 22's picture.
//
// Which tooth. The renderer is right-handed and +x is the patient's RIGHT
// (Anatomy.swift's header). Step 20's camera stands at −x, so the premolar
// square to it in the middle of the frame is the patient's LEFT mandibular
// first premolar: Universal #21, `placeTeeth()` index 3. Projected through
// step 20's camera it lands at pixel (940, 655) of 1920 × 1080, and its
// buccal face points at the lens to within 4° (cos 0.997).
//
// The brush is modelled on photographs of one real head: a white disc on a
// white tapering neck, a hexagonal boss on the back of the disc with a small
// hole in it, a diamond-shaped recess on the back of the neck and a slot on
// its front, and tufts — white in the middle, light and dark blue round the
// rim. No brand, no lettering. No ruler measurement was available, so sizes
// are sourced typical values where a source was found, and MODEL otherwise;
// the photos fix proportions only.
//
// The one rule that matters: each tuft's length is not chosen by eye. Its tip
// is cast along its axis against step 20's own distance function until it
// first touches the enamel or the gum, and stops there — touching, neither
// floating nor passing through. `castLengths` does that; the tests check it on
// the geometry the kernel actually draws.
//
// Millimetres and step 20's axes throughout: y up, y = 0 the occlusal plane,
// +z back towards the throat, the midline at x = 0.

import Foundation
import Metal
import simd

// MARK: - the tooth being brushed

/// #21, the patient's left mandibular first premolar, in `placeTeeth()`.
let brushedToothIndex: Int = 3

// MARK: - the head, in sourced sizes

/// Diameter of the round head: 11.5 mm. The first render used 13 mm, from
/// "the cup-shaped brush head has a diameter of approximately 13mm" (Oral-B's
/// professional education site, dentalcare.ca, "Round for a reason"), and it
/// read too big beside a 7 mm premolar. Russell asked for it slightly smaller.
/// What the sources give: the 13 mm figure is for Oral-B's classic round
/// cup; the newer iO head is 2 mm WIDER than the CrossAction ("Cleansing
/// efficacy of the electric toothbrush Oral-B iO...", Clin Oral Investig 2024,
/// PMC11339098); and reviewers call the Precision Clean — the design in the
/// photos — the smallest of the maker's standard round heads (animated-teeth
/// .com). No maker publishes its millimetres. MODEL: 11.5 mm, a little under
/// the classic 13, and every other part of the head scales with it by
/// `headScale`, so the brush keeps the proportions of the one in the photos.
let headDiameter: Float = 11.5
/// How much smaller than the first render's 13 mm head everything is.
let headScale: Float = headDiameter / 13.0

/// Thickness of the disc that carries the tufts. MODEL: in the side-on photo
/// the disc is about a quarter of its own diameter thick.
let carrierThickness: Float = 3.0 * headScale

/// The hexagonal boss on the back of the disc: across its flats, and how far
/// it stands proud. MODEL: in the photo of the back it spans a little under
/// six tenths of the disc, and stands about half the disc's thickness.
let hexAcrossFlats: Float = 7.5 * headScale
let hexHeight: Float = 1.5 * headScale

/// The neck: width and thickness where it meets the head and where it leaves
/// the frame, and how long a piece of it is modelled. MODEL, from the photos:
/// the neck is about four tenths of the disc wide at the head and widens
/// steadily towards the handle.
let neckWidthNear: Float = 5.0 * headScale
let neckWidthFar: Float = 8.0 * headScale
let neckThickNear: Float = 3.8 * headScale
let neckThickFar: Float = 5.5 * headScale
let neckLength: Float = 60.0
/// How far behind the disc's front face the neck's front face sits. MODEL:
/// in the side-on photo the neck runs along the back half of the disc.
let neckSetBack: Float = 0.8 * headScale

// MARK: - the tufts, in sourced sizes

/// Diameter of one tuft. "Standard tufts with a diameter of between 1.5 mm
/// and 1.7 mm" (Braun GmbH, US 6,957,468 B2, "Toothbrush head with
/// anchor-free bristle tufting", 2005). The bottom of that range, so sixteen
/// still fit round the smaller head (the first render used the middle, 1.6).
let tuftDiameter: Float = 1.5

/// Diameter of one filament: "nylon 6,12 filaments typically having a
/// diameter of 0.15–0.25 mm, often 0.2 mm" (Unilever, EP 1 014 830 B1,
/// "Toothbrush", 2002); for electric toothbrushes Braun gives 0.10–0.25 mm
/// (US 6,094,769, "Bristle for a toothbrush", 2000). Filaments are not drawn
/// one by one — tufts are — but their size sets the fine striation on each tuft.
let filamentDiameter: Float = 0.2

/// A new tuft's free length, from the disc to its tip, before it is pressed.
/// MODEL: in the side-on photo the tufts stand 0.5–0.7 of the disc's diameter
/// (6.5–9 mm against 13 mm). Manual-brush patents give 9 ± 1 mm for short
/// tufts and 13 ± 1 mm for long ones (US 2010/0180392 A1, "Toothbrush with long
/// tapered bristles and short non-tapered bristles"); a round oscillating head
/// is smaller, and 8 mm sat at the photo's middle for a 13 mm head — scaled
/// with the head, so the tufts keep the photo's proportions.
let restLength: Float = 8.0 * headScale

/// The rim: a ring of 16 equal tufts, as in the round oscillating head of
/// US 9,332,828 B2 ("Brush head for an electric toothbrush", Ranir LLC, 2016),
/// whose outer series is "16 equally circumferentially spaced bristle tufts".
/// Its radius is derived, not typed: the smallest circle on which 16 tufts of
/// `tuftDiameter` fit with `tuftGap` between neighbours.
let rimTuftCount: Int = 16
/// Clear space between neighbouring tufts' bases. MODEL: the moulded walls
/// between tuft holes are 0.2–0.3 mm thick (US 6,957,468 B2, as above). The
/// thin end, for the smaller head.
let tuftGap: Float = 0.2
let rimRadius: Float = Float(rimTuftCount) * (tuftDiameter + tuftGap) / (2 * Float.pi)
/// The white centre: one tuft and a ring of eight round it. MODEL: the photo
/// of the face shows a white centre about half the field across; the patent
/// above puts two rings of six there, which the photo does not show.
let innerTuftCount: Int = 8
let innerRadius: Float = 2.6 * headScale

/// How much a pressed tuft splays: its tip widens by `splayGrowth` × its
/// fractional compression, and leans outward from the head's centre by
/// `splayLean` radians × the same. MODEL, for the look of a pressed brush: a
/// tuft pressed to 70% of its length widens by a third and leans about 9°.
let splayGrowth: Float = 1.1
let splayLean: Float = 0.5

/// How far each tuft's base is buried in the disc, so the join never shows.
let tuftBuried: Float = 0.5

// MARK: - colour

/// The dyes and plastics, as CIELAB. MODEL: read off Russell's photos of the
/// real head (averages of small patches, in the side-on photo where the tufts
/// are best lit) and corrected for the room's warm light — the white neck in
/// the same photos reads b* +6 to +12, so about −6 b* is taken off every
/// colour. The brush is rendered new, so the blues are the saturated ones seen
/// on unworn tufts: dark blue read L* 41–44, a* 30–33, b* −58 to −62; light blue
/// L* 69–75, a* −9 to −11, b* −15 to −20; white tufts L* 69–76 in shadow. The
/// dark blue is taken at the darker end (the photo of the face reads L* 25–32
/// on it) because step 20's lights are bright, and at L* 42 it rendered as
/// periwinkle beside the teeth.
let darkBlueLab: SIMD3<Double> = SIMD3(33.0, 31.0, -63.0)
let lightBlueLab: SIMD3<Double> = SIMD3(68.0, -11.0, -26.0)
let whiteTuftLab: SIMD3<Double> = SIMD3(88.0, 0.0, 1.0)
/// White polypropylene. MODEL: the head's rim reads L* 82 under the room's
/// light next to a white bristle's 75; unfilled white PP mouldings are not
/// quite paper white.
let headLab: SIMD3<Double> = SIMD3(91.0, -0.5, 2.0)

/// Refractive indices, for the reflectance of the wet plastics. Nylon: 1.53,
/// the value commonly quoted for nylon 6; a measured 1.525 for isotropic
/// nylon 6 is in Gaur et al. on drawn nylon 6 yarns (J Polym Sci Polym Phys
/// 1975) — the drawn filament is birefringent round that. NOT checked against
/// a primary table: flagged. Polypropylene: 1.49, likewise commonly quoted and
/// not checked. In the mouth both are wet, so each reflects against saliva
/// (water, 1.333), exactly as step 20's enamel does.
let nylonIndex: Double = 1.53
let polypropyleneIndex: Double = 1.49
let nylonUnderFilmF0: Float = fresnelF0(salivaIndex, nylonIndex)
let polypropyleneUnderFilmF0: Float = fresnelF0(salivaIndex, polypropyleneIndex)

// MARK: - the pose

/// Where the brush is, as a frame: `face` is the centre of the disc's front
/// face, `axis` points from it towards the tooth (the way the tufts point),
/// `along` points down the neck, `across` completes the frame.
struct BrushPose {
    var face: SIMD3<Float>
    var axis: SIMD3<Float>
    var along: SIMD3<Float>
    var across: SIMD3<Float>
    /// STEP 28: the disc's own frame, turned by the spin about `axis`. The
    /// tufts and the boss are laid out on these; the neck stays on `along`.
    var spinAlong: SIMD3<Float>
    var spinAcross: SIMD3<Float>
    var spin: Float
}

/// STEP 28: `along` and `across` turned by `spin` radians about the axis.
/// A point laid out at (x, y) on the disc lands at spinAlong·x + spinAcross·y,
/// which is (x cos − y sin, x sin + y cos) in the unturned frame. At spin 0,
/// cos is exactly 1 and sin exactly 0, so these are `along` and `across` to
/// the last bit.
func spinFrame(along: SIMD3<Float>, across: SIMD3<Float>, spin: Float) -> (SIMD3<Float>, SIMD3<Float>) {
    let c: Float = cos(spin)
    let s: Float = sin(spin)
    let a: SIMD3<Float> = along * c + across * s
    let b: SIMD3<Float> = across * c - along * s
    return (a, b)
}

/// One tuft, as drawn: a round cone from a buried base to a rounded tip.
/// `root` is where it leaves the disc's face, `length` is from there to the
/// very end of its tip, and `direction` is the way it runs.
struct Tuft {
    var root: SIMD3<Float>
    var direction: SIMD3<Float>
    var length: Float
    var rootRadius: Float
    var tipRadius: Float
    var dye: Int                // 0 white, 1 light blue, 2 dark blue
    var compression: Float      // 0 unpressed … 1 flat
    /// STEP 28: standing free at its full new length, because the surface in
    /// front of it is further away than a new tuft reaches — over the gap
    /// between two teeth. Never true in step 22's pose.
    var free: Bool = false
    /// The centre of the rounded tip.
    var tipCentre: SIMD3<Float> { root + direction * (length - tipRadius) }
    /// The very end of the tuft.
    var apex: SIMD3<Float> { root + direction * length }
    /// The centre of the buried base.
    var base: SIMD3<Float> { root - direction * tuftBuried }
}

/// The brush in the mouth.
struct Brush {
    var pose: BrushPose
    var tufts: [Tuft]
}

/// What to break, for the mutation check. Each must be caught by a test.
enum BrushMutant {
    case none
    case lift       // the whole brush 1 mm back off the tooth
    case push       // the whole brush 1 mm into it
}

/// The tufts' positions on the disc, in the disc's own plane (along, across),
/// with their dyes. The rim alternates in fours, light then dark, as in the
/// photo of the face; it is turned so that a light and a dark group meet at
/// the top of the head, where the camera looks.
func tuftLayout() -> [(position: SIMD2<Float>, dye: Int)] {
    var out: [(position: SIMD2<Float>, dye: Int)] = [(SIMD2<Float>(0, 0), 0)]
    for k in 0..<innerTuftCount {
        let angle: Float = 2 * Float.pi * (Float(k) + 0.5) / Float(innerTuftCount)
        out.append((SIMD2<Float>(cos(angle), sin(angle)) * innerRadius, 0))
    }
    for k in 0..<rimTuftCount {
        let angle: Float = 2 * Float.pi * (Float(k) + 0.5) / Float(rimTuftCount)
        let dye: Int = (k / 4) % 2 == 0 ? 1 : 2
        out.append((SIMD2<Float>(cos(angle), sin(angle)) * rimRadius, dye))
    }
    return out
}

/// How steeply the neck climbs away from the head, in radians. MODEL: just
/// enough that the neck leaves step 20's frame through its right edge, as the
/// brief asks, rather than through the bottom; any more and it crosses the
/// crowns of #20 and #19. The disc is round, so this only turns it in its own
/// plane: it changes nothing about where the tufts land.
let neckRise: Float = 0.16

/// The brush's frame, before it is pressed on: the tufts point straight into
/// #21's buccal face (level, along the arch's inward normal), and the neck runs
/// distally along the arch, rising a little — out of the right of step 20's
/// frame. MODEL: held square on and level, as a round head is held. In a real
/// mouth the handle leaves between the lips, so it would come from the front;
/// the neck is sent distally here because the brief placed it so.
func brushFrame() -> (axis: SIMD3<Float>, along: SIMD3<Float>, across: SIMD3<Float>) {
    let t: PlacedTooth = placeTeeth()[brushedToothIndex]
    let axis = SIMD3<Float>(-t.outward.x, 0, -t.outward.y)
    let flat = SIMD3<Float>(t.tangent.x, 0, t.tangent.y)
    let along: SIMD3<Float> = flat * cos(neckRise) + SIMD3<Float>(0, 1, 0) * sin(neckRise)
    let across: SIMD3<Float> = simd_cross(along, axis)
    return (axis, along, across)
}

/// The mid-facial point of #21's gum margin: at the margin's height, where the
/// gum follows the cementoenamel junction `gumMarginAboveCEJ` above it
/// (Gargiulo et al., via Anatomy.swift). Brushing an oscillating head is aimed
/// there. The central tuft's axis passes through this point.
func marginAim() -> SIMD3<Float> {
    let t: PlacedTooth = placeTeeth()[brushedToothIndex]
    let y: Float = -t.spec.crownHeight + gumMarginAboveCEJ
    return SIMD3<Float>(t.centre.x, y, t.centre.y)
}

// MARK: - asking step 20's distance function

/// Step 20's scene — teeth, gum, tongue, floor, no brush — evaluated at many
/// points on the GPU, from a kernel compiled once. `probeScene` compiles on
/// every call, and the casting below asks a few dozen times.
final class SceneProbe {
    let device: MTLDevice
    let pso: MTLComputePipelineState
    let teeth: MTLBuffer
    let queue: MTLCommandQueue

    init(device: MTLDevice) throws {
        self.device = device
        let library: MTLLibrary = try makeLibrary(device, mutant: .none)
        pso = try pipeline(device, library, "probe")
        var t: [GPUTooth] = gpuTeeth(placeTeeth())
        guard let tb = device.makeBuffer(bytes: &t, length: MemoryLayout<GPUTooth>.stride * t.count,
                                         options: .storageModeShared),
              let q = device.makeCommandQueue()
        else { throw MouthError.gpu("could not set up the scene probe") }
        teeth = tb
        queue = q
    }

    /// (distance, material) at each point.
    func callAsFunction(_ points: [SIMD3<Float>]) throws -> [SIMD2<Float>] {
        if points.isEmpty { return [] }
        var pts: [SIMD4<Float>] = points.map { SIMD4<Float>($0.x, $0.y, $0.z, 0) }
        guard let pb = device.makeBuffer(bytes: &pts, length: 16 * pts.count, options: .storageModeShared),
              let ob = device.makeBuffer(length: 8 * pts.count, options: .storageModeShared),
              let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder()
        else { throw MouthError.gpu("could not run the scene probe") }
        enc.setComputePipelineState(pso)
        enc.setBuffer(pb, offset: 0, index: 0)
        enc.setBuffer(ob, offset: 0, index: 1)
        enc.setBuffer(teeth, offset: 0, index: 2)
        let w: Int = min(pso.maxTotalThreadsPerThreadgroup, 256)
        enc.dispatchThreads(MTLSize(width: pts.count, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        if let e = cb.error { throw MouthError.gpu(e.localizedDescription) }
        let out = ob.contents().assumingMemoryBound(to: SIMD2<Float>.self)
        return (0..<pts.count).map { out[$0] }
    }

    /// Just the distances.
    func distances(_ points: [SIMD3<Float>]) throws -> [Float] {
        try self(points).map { $0.x }
    }
}

// MARK: - points on a tuft

/// Two unit vectors square to `d` and to each other.
func perpendiculars(_ d: SIMD3<Float>) -> (SIMD3<Float>, SIMD3<Float>) {
    let helper: SIMD3<Float> = abs(d.y) < 0.9 ? SIMD3<Float>(0, 1, 0) : SIMD3<Float>(1, 0, 0)
    let e1: SIMD3<Float> = simd_normalize(simd_cross(d, helper))
    let e2: SIMD3<Float> = simd_cross(d, e1)
    return (e1, e2)
}

/// How many of `tuftSurface`'s points are on the rounded tip — they come first.
let tipPointCount: Int = 1 + 12 * 32

/// Points on the outside of a tuft, grown outward by `grow` (negative shrinks
/// it): a dense cap over the rounded tip — the forward hemisphere and a band
/// past it, where a tuft that meets a surface at a slant touches first — and
/// rings down the flank to where it leaves the disc. Sample spacing on the
/// tip is ~0.13 mm, so between samples the tip bulges by under 0.003 mm.
func tuftSurface(root: SIMD3<Float>, direction: SIMD3<Float>, length: Float,
                 rootRadius: Float, tipRadius: Float, grow: Float) -> [SIMD3<Float>] {
    let (e1, e2) = perpendiculars(direction)
    let tip: SIMD3<Float> = root + direction * (length - tipRadius)
    let r: Float = tipRadius + grow
    var out: [SIMD3<Float>] = [tip + direction * r]
    let rings: Int = 12
    let around: Int = 32
    for i in 1...rings {
        // From the apex (0) round to 20° behind the equator.
        let polar: Float = Float(i) / Float(rings) * (Float.pi / 2 + 0.35)
        let axial: Float = cos(polar) * r
        let radial: Float = sin(polar) * r
        for j in 0..<around {
            let phi: Float = 2 * Float.pi * Float(j) / Float(around)
            let side: SIMD3<Float> = e1 * cos(phi) + e2 * sin(phi)
            out.append(tip + direction * axial + side * radial)
        }
    }
    let flankRings: Int = 10
    let flankLength: Float = length - tipRadius
    for i in 0..<flankRings {
        let s: Float = Float(i) / Float(flankRings)
        let c: SIMD3<Float> = root + direction * (flankLength * s)
        let radius: Float = rootRadius + (tipRadius - rootRadius) * s + grow
        for j in 0..<24 {
            let phi: Float = 2 * Float.pi * Float(j) / 24
            out.append(c + (e1 * cos(phi) + e2 * sin(phi)) * radius)
        }
    }
    return out
}

/// Points filling the inside of a tuft shrunk by `shrink`: the tip ball in
/// shells and the body in discs. Used to prove nothing of the tuft is inside
/// the tooth or gum.
func tuftInterior(_ t: Tuft, shrink: Float) -> [SIMD3<Float>] {
    var out: [SIMD3<Float>] = []
    for f in [Float(1.0), 0.75, 0.5, 0.25, 0.0] {
        let rt: Float = max(t.tipRadius - shrink, 0) * f
        let rr: Float = max(t.rootRadius - shrink, 0) * f
        out += tuftSurface(root: t.root, direction: t.direction, length: t.length - t.tipRadius + rt,
                           rootRadius: rr, tipRadius: rt, grow: 0)
    }
    return out
}

func tuftSurface(_ t: Tuft, grow: Float) -> [SIMD3<Float>] {
    tuftSurface(root: t.root, direction: t.direction, length: t.length,
                rootRadius: t.rootRadius, tipRadius: t.tipRadius, grow: grow)
}

// MARK: - casting the tufts onto the tooth

/// For each tuft (root, direction, radii), the free length at which it first
/// touches step 20's scene: marched out in 0.1 mm steps until any point on its
/// surface is inside, then bisected to a micron. Only the SIGN of the distance
/// is used, and the sign is exact even where the gum's distance is scaled down
/// to be safe for rays — so the answer does not depend on how honest the
/// distance's size is.
func castLengths(_ specs: [(root: SIMD3<Float>, direction: SIMD3<Float>, rootRadius: Float, tipRadius: Float)],
                 probe: SceneProbe) throws -> [Float] {
    let step: Float = 0.1
    let shortest: Float = 1.5
    let longest: Float = 16.0
    let steps: Int = Int((longest - shortest) / step) + 1
    // Coarse march, every tuft and every length at once.
    var points: [SIMD3<Float>] = []
    var spans: [Int] = []
    for s in specs {
        for k in 0..<steps {
            let L: Float = shortest + Float(k) * step
            let pts: [SIMD3<Float>] = tuftSurface(root: s.root, direction: s.direction, length: L,
                                                  rootRadius: s.rootRadius, tipRadius: s.tipRadius, grow: 0)
            points += pts
            spans.append(pts.count)
        }
    }
    let d: [Float] = try probe.distances(points)
    var lo: [Float] = []
    var hi: [Float] = []
    var cursor: Int = 0
    for (i, _) in specs.enumerated() {
        var found: Float = -1
        for k in 0..<steps {
            let n: Int = spans[i * steps + k]
            var inside: Bool = false
            for m in cursor..<(cursor + n) where d[m] < 0 { inside = true; break }
            cursor += n
            if inside && found < 0 { found = shortest + Float(k) * step }
        }
        guard found > shortest else {
            throw MouthError.gpu("tuft \(i) never reaches the tooth within \(longest) mm, or starts inside it")
        }
        lo.append(found - step)
        hi.append(found)
    }
    // Bisect: `lo` is always clear, `hi` always touching.
    for _ in 0..<17 {
        var pts: [SIMD3<Float>] = []
        var counts: [Int] = []
        for (i, s) in specs.enumerated() {
            let mid: Float = (lo[i] + hi[i]) / 2
            let p: [SIMD3<Float>] = tuftSurface(root: s.root, direction: s.direction, length: mid,
                                                rootRadius: s.rootRadius, tipRadius: s.tipRadius, grow: 0)
            pts += p
            counts.append(p.count)
        }
        let dd: [Float] = try probe.distances(pts)
        var c: Int = 0
        for i in 0..<specs.count {
            var inside: Bool = false
            for m in c..<(c + counts[i]) where dd[m] < 0 { inside = true; break }
            c += counts[i]
            let mid: Float = (lo[i] + hi[i]) / 2
            if inside { hi[i] = mid } else { lo[i] = mid }
        }
    }
    return lo
}

/// Where the disc's face sits along the central axis, as the distance from
/// the face back to the margin point. The head is pressed in until the tuft
/// that has furthest to go just reaches the tooth at its rest length — no tuft
/// can be longer than a new tuft is — so every other tuft is pressed shorter
/// than that, by however much nearer the tooth or gum is in front of it.
/// Found by casting the tufts straight from a trial face and moving the face
/// by the difference; the cast lengths change exactly with the face's move.
func pressedBrush(probe: SceneProbe, mutant: BrushMutant = .none) throws -> Brush {
    let (axis, along, across) = brushFrame()
    let aim: SIMD3<Float> = marginAim()
    let layout = tuftLayout()
    let r0: Float = tuftDiameter / 2

    func roots(_ face: SIMD3<Float>) -> [SIMD3<Float>] {
        layout.map { face + along * $0.position.x + across * $0.position.y }
    }

    // 1. Straight tufts from a trial face well clear of the tooth.
    let trial: SIMD3<Float> = aim - axis * (restLength + 3)
    let straight: [Float] = try castLengths(roots(trial).map { ($0, axis, r0, r0) }, probe: probe)
    // 2. Move the face so the longest reach equals the rest length.
    let reach: Float = straight.max()!
    let face: SIMD3<Float> = trial + axis * (reach - restLength)
    let pressed: [Float] = straight.map { $0 - (reach - restLength) }

    // 3. Each tuft splays by how hard it is pressed, and is cast again along
    //    its splayed direction, from its root on the pressed face.
    var specs: [(root: SIMD3<Float>, direction: SIMD3<Float>, rootRadius: Float, tipRadius: Float)] = []
    var compressions: [Float] = []
    for (i, l) in layout.enumerated() {
        let c: Float = max(restLength - pressed[i], 0) / restLength
        let radial: Float = simd_length(l.position)
        var dir: SIMD3<Float> = axis
        if radial > 1e-3 {
            let outward: SIMD3<Float> = simd_normalize(along * l.position.x + across * l.position.y)
            let lean: Float = splayLean * c
            dir = simd_normalize(axis * cos(lean) + outward * sin(lean))
        }
        let root: SIMD3<Float> = face + along * l.position.x + across * l.position.y
        specs.append((root, dir, r0, r0 * (1 + splayGrowth * c)))
        compressions.append(c)
    }
    let lengths: [Float] = try castLengths(specs, probe: probe)

    // The mutants move the finished brush, tufts and all, along its axis.
    var shift: Float = 0
    if mutant == .lift { shift = -1 }
    if mutant == .push { shift = 1 }
    let offset: SIMD3<Float> = axis * shift
    var tufts: [Tuft] = []
    for (i, s) in specs.enumerated() {
        tufts.append(Tuft(root: s.root + offset, direction: s.direction, length: lengths[i],
                          rootRadius: s.rootRadius, tipRadius: s.tipRadius,
                          dye: layout[i].dye, compression: compressions[i]))
    }
    return Brush(pose: BrushPose(face: face + offset, axis: axis, along: along, across: across,
                                 spinAlong: along, spinAcross: across, spin: 0), tufts: tufts)
}

// MARK: - the brush as Metal

func metal4(_ v: SIMD3<Float>, _ w: Float) -> String { "float4(\(v.x), \(v.y), \(v.z), \(w))" }

/// The brush, spliced into step 20's kernel: its distance functions, its place
/// in the scene's distance, and how it is shaded. Materials 5 (the head and
/// neck, white polypropylene) and 6 (the tufts, dyed nylon).
func brushExtra(_ brush: Brush) -> SceneExtra {
    let p: BrushPose = brush.pose
    let n: Int = brush.tufts.count
    let bases: String = brush.tufts.map { metal4($0.base, $0.rootRadius) }.joined(separator: ", ")
    let tips: String = brush.tufts.map { metal4($0.tipCentre, $0.tipRadius) }.joined(separator: ", ")
    let dyes: String = brush.tufts.map { "\($0.dye)" }.joined(separator: ", ")
    // A sphere round the head and every tuft, and a capsule round the neck:
    // outside both, the brush cannot be nearer than they are, so it is skipped
    // exactly (a plain minimum, not a blend, so skipping cannot change it).
    var boundR: Float = headDiameter / 2 + 1
    let boundC: SIMD3<Float> = p.face + p.axis * (restLength / 2 - 2)
    for t in brush.tufts {
        boundR = max(boundR, simd_distance(t.apex, boundC) + t.tipRadius + 0.2)
        boundR = max(boundR, simd_distance(t.base, boundC) + t.rootRadius + 0.2)
    }
    boundR = max(boundR, simd_distance(p.face - p.axis * (carrierThickness + hexHeight), boundC) + headDiameter / 2 + 0.5)
    let neckEnd: SIMD3<Float> = p.face + p.along * neckLength
    let neckRadius: Float = simd_length(SIMD2<Float>(neckWidthFar / 2, neckThickFar + neckSetBack)) + 0.5
    let white: SIMD3<Float> = labToLinearSRGB(whiteTuftLab)
    let light: SIMD3<Float> = labToLinearSRGB(lightBlueLab)
    let dark: SIMD3<Float> = labToLinearSRGB(darkBlueLab)

    let functions: String = """


        // ------------------------------------------------------------ the brush

        constant float3 BR_FACE = \(metal(p.face));
        constant float3 BR_AXIS = \(metal(p.axis));
        constant float3 BR_ALONG = \(metal(p.along));
        constant float3 BR_ACROSS = \(metal(p.across));
        constant float3 BR_SPIN_ALONG = \(metal(p.spinAlong));
        constant float3 BR_SPIN_ACROSS = \(metal(p.spinAcross));
        constant float4 BR_BOUND = \(metal4(boundC, boundR));
        constant float3 BR_NECK_END = \(metal(neckEnd));
        constant float BR_NECK_R = \(neckRadius);
        constant float HEAD_R = \(headDiameter / 2);
        constant float DIAMOND_X = \(13.0 * headScale);
        constant float SLOT_X = \(17.0 * headScale);
        constant float CARRIER_T = \(carrierThickness);
        constant float HEX_F = \(hexAcrossFlats / 2);
        constant float HEX_H = \(hexHeight);
        constant float NECK_LEN = \(neckLength);
        constant float NECK_W0 = \(neckWidthNear / 2);
        constant float NECK_W1 = \(neckWidthFar / 2);
        constant float NECK_T0 = \(neckThickNear / 2);
        constant float NECK_T1 = \(neckThickFar / 2);
        constant float NECK_BACK = \(neckSetBack);
        constant uint TUFT_COUNT = \(n);
        constant float4 TUFT_BASE[\(n)] = { \(bases) };
        constant float4 TUFT_TIP[\(n)] = { \(tips) };
        constant int TUFT_DYE[\(n)] = { \(dyes) };
        constant float3 DYE[3] = { \(metal(white)), \(metal(light)), \(metal(dark)) };
        constant float3 HEAD_ALBEDO = \(metal(labToLinearSRGB(headLab)));
        constant float NYLON_F0 = \(nylonUnderFilmF0);
        constant float PP_F0 = \(polypropyleneUnderFilmF0);
        constant float FILAMENT = \(filamentDiameter);

        // A cone with a ball at each end, of different sizes — exact
        // (Quílez's round cone, for arbitrary end points).
        float roundCone(float3 p, float3 a, float3 b, float r1, float r2) {
            float3 ba = b - a;
            float l2 = dot(ba, ba);
            float rr = r1 - r2;
            float a2 = l2 - rr * rr;
            float il2 = 1.0 / l2;
            float3 pa = p - a;
            float y = dot(pa, ba);
            float z = y - l2;
            float3 xv = pa * l2 - ba * y;
            float x2 = dot(xv, xv);
            float y2 = y * y * l2;
            float z2 = z * z * l2;
            float k = sign(rr) * rr * rr * x2;
            if (sign(z) * a2 * z2 > k) return sqrt(x2 + z2) * il2 - r2;
            if (sign(y) * a2 * y2 < k) return sqrt(x2 + y2) * il2 - r1;
            return (sqrt(x2 * a2 * il2) + y * rr) * il2 - r1;
        }

        // Every tuft; which is nearest, and how far along it (0 base, 1 tip).
        float tuftsSDF(float3 p, thread int &nearest) {
            float d = 1e9;
            nearest = 0;
            for (uint i = 0; i < TUFT_COUNT; i++) {
                float t = roundCone(p, TUFT_BASE[i].xyz, TUFT_TIP[i].xyz, TUFT_BASE[i].w, TUFT_TIP[i].w);
                if (t < d) { d = t; nearest = int(i); }
            }
            return d;
        }

        // A hexagonal prism: apothem h.x in its plane, half height h.y.
        float hexPrism(float3 p, float2 h) {
            const float3 k = float3(-0.8660254, 0.5, 0.57735);
            p = abs(p);
            p.xy -= 2.0 * min(dot(k.xy, p.xy), 0.0) * k.xy;
            float2 d = float2(length(p.xy - float2(clamp(p.x, -k.z * h.x, k.z * h.x), h.x)) * sign(p.y - h.x), p.z - h.y);
            return min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
        }

        float roundBox(float3 p, float3 b, float r) {
            float3 q = abs(p) - b + r;
            return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0) - r;
        }

        // The head and neck, in the brush's frame: x down the neck, y across,
        // z towards the tooth, 0 at the disc's front face.
        float headSDF(float3 p) {
            float3 r = p - BR_FACE;
            float3 q = float3(dot(r, BR_ALONG), dot(r, BR_ACROSS), dot(r, BR_AXIS));
            // The disc, edges rounded.
            const float round0 = 0.6;
            float2 dc = float2(length(q.xy) - (HEAD_R - round0), abs(q.z + CARRIER_T * 0.5) - (CARRIER_T * 0.5 - round0));
            float disc = min(max(dc.x, dc.y), 0.0) + length(max(dc, 0.0)) - round0;
            // The hexagonal boss on its back, and the small hole in it. They
            // turn with the disc (step 28), so they are read in its turned
            // frame; the round disc itself looks the same at any turn.
            float3 hq = float3(dot(r, BR_SPIN_ALONG), dot(r, BR_SPIN_ACROSS), q.z);
            float hex = hexPrism(float3(hq.x, hq.y, hq.z + CARRIER_T + HEX_H * 0.5 - 0.2), float2(HEX_F - 0.3, HEX_H * 0.5 + 0.2 - 0.3)) - 0.3;
            float hole = roundBox(hq - float3(1.9, 0.0, -CARRIER_T - HEX_H), float3(0.5, 0.3, 0.9), 0.1);
            hex = max(hex, -hole);
            // The neck: a rounded bar that widens and thickens towards the
            // handle, its front face a little behind the disc's.
            float s = clamp(q.x / NECK_LEN, 0.0, 1.0);
            float hw = mix(NECK_W0, NECK_W1, s);
            float ht = mix(NECK_T0, NECK_T1, s);
            float3 nq = float3(q.x - NECK_LEN * 0.5, q.y, q.z + NECK_BACK + ht);
            float neck = roundBox(nq, float3(NECK_LEN * 0.5, hw, ht), min(hw, ht) * 0.7);
            // The diamond-shaped recess on the back of the neck, and the slot
            // on its front.
            float2 dq = abs(float2(q.x - DIAMOND_X, q.y));
            float rhomb = (dq.x * 1.0 + dq.y * 2.2 - 2.2) / sqrt(1.0 + 2.2 * 2.2);
            float back = -(q.z + NECK_BACK + 2.0 * ht);
            float diamond = max(rhomb, abs(back) - 0.35);
            float2 sq = float2(max(abs(q.x - SLOT_X) - 1.4, 0.0), q.y);
            float slot = max(length(sq) - 0.55, abs(q.z + NECK_BACK) - 0.9);
            neck = max(neck, -diamond);
            neck = max(neck, -slot);
            float d = smin(disc, neck, 0.8);
            return min(d, hex);
        }

        // The brush can be no nearer than a sphere round its head and tufts,
        // or a capsule round its neck.
        float brushBound(float3 p) {
            float sphere = length(p - BR_BOUND.xyz) - BR_BOUND.w;
            float3 pa = p - BR_FACE;
            float3 ba = BR_NECK_END - BR_FACE;
            float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
            float capsule = length(pa - ba * h) - BR_NECK_R;
            return min(sphere, capsule);
        }

        kernel void brushProbe(device const float4 *points [[buffer(0)]],
                               device float4 *out [[buffer(1)]],
                               uint id [[thread_position_in_grid]]) {
            float3 p = points[id].xyz;
            int ti;
            float dt = tuftsSDF(p, ti);
            out[id] = float4(headSDF(p), dt, float(ti), brushBound(p));
        }
    """

    let sceneCall: String = """

            if (brushBound(p) < d) {
                float dh = headSDF(p);
                if (dh < d) { d = dh; mat = 5; }
                int ti;
                float dt = tuftsSDF(p, ti);
                if (dt < d) { d = dt; mat = 6; }
            }
    """

    let shadeBranch: String = """
     else if (mat == 5) {
                // White polypropylene: matte, a little translucent, so its
                // shadowed side glows faintly rather than going grey. Wet.
                albedo = HEAD_ALBEDO;
                wrap = 0.35;
                sssTint = float3(0.10, 0.09, 0.08);
                filmAlpha = 0.16;
                baseAlpha = 0.45;
                baseF0 = PP_F0;
            } else if (mat == 6) {
                // Dyed nylon. A tuft is a bundle of 0.2 mm filaments: a fine
                // striation runs along it, and its thin, rounded-off tips let
                // light through, so they read paler than the tuft's body.
                int ti;
                tuftsSDF(p, ti);
                float3 a = TUFT_BASE[ti].xyz;
                float3 b = TUFT_TIP[ti].xyz;
                float3 ax = normalize(b - a);
                float along = clamp(dot(p - a, ax) / (length(b - a) + TUFT_TIP[ti].w), 0.0, 1.0);
                float3 e1 = normalize(cross(ax, abs(ax.y) < 0.9 ? float3(0, 1, 0) : float3(1, 0, 0)));
                float3 e2 = cross(ax, e1);
                float3 rel = p - a;
                // The filaments round the tuft: as many as fit round its base
                // at 0.2 mm each, running its length and fanning out with it.
                // Each is a small cylinder of its own — the normal turns
                // across it, so it carries its own thin highlight — with a
                // dark gap to the next, and a slightly different brightness.
                float theta = atan2(dot(rel, e2), dot(rel, e1)) / 6.2831853 + 0.5;
                float count = round(6.2831853 * TUFT_BASE[ti].w / FILAMENT);
                float wander = 0.35 * noise3(float3(theta * 9.0, dot(rel, ax) * 0.6, float(ti) * 7.0));
                float fpos = theta * count + wander;
                float strand = fract(fpos) - 0.5;
                float which = hash3(float3(floor(fpos), float(ti), 3.0));
                float3 around = cross(ax, n);
                around = around / max(length(around), 1e-4);
                n = normalize(n + around * strand * 1.1);
                float gap = smoothstep(0.5, 0.3, abs(strand));
                float3 dye = DYE[TUFT_DYE[ti]];
                float tip = smoothstep(0.72, 1.0, along);
                albedo = mix(dye, mix(dye, float3(0.92), 0.45), tip) * mix(0.5, 1.0, gap) * mix(0.8, 1.08, which);
                wrap = 0.45;
                sssTint = dye * 0.35;
                filmAlpha = 0.45;
                baseAlpha = 0.55;
                baseF0 = NYLON_F0;
            }
    """
    return SceneExtra(functions: functions, sceneCall: sceneCall, shadeBranch: shadeBranch)
}

// MARK: - asking the brush's own distance

/// The brush's own distance functions at many points, from the same kernel
/// source the picture uses: (head and neck, nearest tuft, which tuft, bound).
func probeBrush(_ points: [SIMD3<Float>], brush: Brush, on device: MTLDevice) throws -> [SIMD4<Float>] {
    let library: MTLLibrary = try makeLibrary(device, mutant: .none, extra: brushExtra(brush))
    let pso: MTLComputePipelineState = try pipeline(device, library, "brushProbe")
    var pts: [SIMD4<Float>] = points.map { SIMD4<Float>($0.x, $0.y, $0.z, 0) }
    guard let pb = device.makeBuffer(bytes: &pts, length: 16 * pts.count, options: .storageModeShared),
          let ob = device.makeBuffer(length: 16 * pts.count, options: .storageModeShared),
          let queue = device.makeCommandQueue(), let cb = queue.makeCommandBuffer(),
          let enc = cb.makeComputeCommandEncoder()
    else { throw MouthError.gpu("could not set up the brush probe") }
    enc.setComputePipelineState(pso)
    enc.setBuffer(pb, offset: 0, index: 0)
    enc.setBuffer(ob, offset: 0, index: 1)
    enc.dispatchThreads(MTLSize(width: pts.count, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: min(pso.maxTotalThreadsPerThreadgroup, 256), height: 1, depth: 1))
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    if let e = cb.error { throw MouthError.gpu(e.localizedDescription) }
    let out = ob.contents().assumingMemoryBound(to: SIMD4<Float>.self)
    return (0..<pts.count).map { out[$0] }
}
