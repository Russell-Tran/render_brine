// The padlock and its plating as numbers: how big a 40 mm brass padlock is,
// how thick each metal layer on it is and why the middle one is there, and
// where the lights and the camera stand. Nothing here touches the GPU.
//
// Millimetres throughout for the lock, micrometres for the plating. The lock
// lies on its back on a plain table: y is up and y = 0 is the table top.

import Foundation
import simd

// MARK: - the padlock

// A solid-brass padlock 40 mm wide, the commonest size there is. The four
// numbers a maker publishes are from Master Lock's specification for its
// No. 140D "Solid Brass Padlock, 1-9/16 in Wide" (masterlock.com product
// page, checked): body width 1-9/16 in (40 mm), shackle diameter 1/4 in
// (6 mm), shackle length 7/8 in (22 mm), shackle width 13/16 in (21 mm).
// "Length" is the clear height inside the shackle above the body and "width"
// the clear span between its legs: a 21 mm OUTSIDE width round 6 mm bar would
// leave the legs 9 mm apart on a 40 mm body, which no padlock of this class
// has. That reading is inferred, not stated on the page — UNVERIFIED.
// Nothing else about that product is copied: this lock is an unbranded
// member of the class.

/// Body width, across the face. Master Lock 140D: 40 mm.
let bodyWidth: Float = 40.0
/// Shackle bar diameter. Master Lock 140D: 6 mm.
let shackleDiameter: Float = 6.0
/// Clear height inside the shackle, body top to the inside of the bow.
/// Master Lock 140D: 22 mm.
let shackleClearHeight: Float = 22.0
/// Clear span between the shackle's legs. Master Lock 140D: 21 mm.
let shackleInsideWidth: Float = 21.0

/// Body height, bottom face to shoulders. MODEL: the maker gives no height;
/// solid-brass bodies of this width are close to square in photographs.
let bodyHeight: Float = 38.0
/// Body thickness, front to back. MODEL: likewise unpublished; brass padlock
/// bodies are roughly 3/8 as thick as they are wide.
let bodyThickness: Float = 15.0
/// Radius on every edge of the body: machined brass is broken and polished
/// at the edges before plating. MODEL.
let bodyEdgeRadius: Float = 2.2
/// How far each leg of the closed shackle runs down into the body. Hidden,
/// but it keeps the shackle one piece with the body. MODEL.
let shackleInsertion: Float = 9.0

/// The keyway is on the bottom face, as on pin-tumbler padlocks of this class:
/// the key goes in along the lock's long axis. MODEL throughout, since the
/// brief leaves the mechanism for later: a plug of this diameter, the thin
/// ring of clearance round it, and a stepped slot that ends blind, with no
/// pins behind it.
let plugDiameter: Float = 11.0
let plugClearance: Float = 0.25       // width of the visible ring round the plug
let plugRingDepth: Float = 0.6
let keywayLength: Float = 7.2         // the slot's long side, across the face
let keywayWidth: Float = 1.3          // its narrow side
let keywayDepth: Float = 7.0

/// Where the shackle's centreline runs, derived from the four published
/// numbers — the legs are one bar radius outside the clear span, and the bow's
/// centre sits so its inside edge is the clear height above the body.
let shackleRadius: Float = shackleDiameter / 2
let shackleLegOffset: Float = shackleInsideWidth / 2 + shackleRadius
let shackleBowCentre: Float = bodyHeight + shackleClearHeight - (shackleLegOffset - shackleRadius)

/// The lock's outline, for the tests: overall height, bottom to top of bow.
let lockOverallHeight: Float = shackleBowCentre + shackleLegOffset + shackleRadius

// MARK: - the plating

/// One layer of the cut through the surface, top first.
struct Layer: Equatable {
    var name: String
    var micrometres: Float
}

/// Gold on a decorative lock. The US Federal Trade Commission's jewelry
/// guides (16 CFR 23.3(c)(3), eCFR text checked) allow "gold electroplate"
/// only for a coating at least 0.175 µm of fine gold thick, and "heavy gold
/// electroplate" at 2.5 µm; below 0.175 µm it is a flash or wash. ASTM B488,
/// which the brief suggested, turns out to cover gold "used for engineering
/// applications" (its published scope, checked) — contacts and bonding
/// pads, not decorative finishes — so it is not the reference here. 0.5 µm is
/// MODEL: a mid-range decorative plate inside the FTC's 0.175–2.5 µm band,
/// three times the minimum.
let goldMicrometres: Float = 0.5
let ftcGoldElectroplateMinimum: Float = 0.175
let ftcHeavyGoldElectroplate: Float = 2.5

/// Nickel under the gold. Copper diffuses into and through electroplated gold
/// (Pinnel & Bennett, "Mass diffusion in polycrystalline copper/electroplated
/// gold planar couples", *Metall Trans* 3:1989–1997, 1972), and nickel is the
/// barrier put between them (Pinnel & Bennett, "Qualitative observations on
/// the diffusion of copper and gold through a nickel barrier", *Metall Trans
/// A* 7:629–635, 1976). Both citations checked; their text was not reachable,
/// so only what their titles say is claimed. That zinc from brass is also
/// held back is the brief's claim, and no source for it was reached —
/// UNVERIFIED. Copper that reaches the gold's surface tarnishes it; a pore-
/// free barrier needs a few micrometres. 3 µm is MODEL: no published value for
/// decorative nickel under gold could be reached online.
let nickelMicrometres: Float = 3.0

/// The cut through the surface, top first, as the inset draws it. The
/// `noNickel` mutant loses the barrier; `thickGold` makes the gold 100× too
/// thick.
func platingStack(_ mutant: Mutant = activeMutant) -> [Layer] {
    let gold = Layer(name: "gold", micrometres: mutant == .thickGold ? 100 * goldMicrometres : goldMicrometres)
    let nickel = Layer(name: "nickel", micrometres: nickelMicrometres)
    let brass = Layer(name: "brass", micrometres: .infinity)
    return mutant == .noNickel ? [gold, brass] : [gold, nickel, brass]
}

/// A cotton fibre's width, for scale: "the average diameter of single cotton
/// fibers in all four yarns is about 15 μm", measured by microscopy (Xiong et
/// al., "An Experiment and Simulation Study on the Tensile Behavior of Cotton
/// Ring-Spun Yarn with Twisted Staple Fibers", *Materials* 19(3):560, 2026,
/// PMC12897929, full text checked). A cotton fibre is a collapsed, twisted
/// ribbon rather than a cylinder; the inset draws it as the circle of that
/// diameter.
let cottonFibreMicrometres: Float = 15.0

// MARK: - where the lock lies

/// The lock lies on its back, turned this far from square to the camera, so
/// the front face, one side, the keyway face and the shackle all show. MODEL.
let lockYawDegrees: Float = 28.0

/// Where the lock's axes point in the world. u runs across the face, v along
/// the lock from the keyway face to the shackle, and the face looks up.
let lockU: SIMD3<Float> = {
    let a: Float = lockYawDegrees * .pi / 180
    return SIMD3<Float>(cos(a), 0, -sin(a))
}()
let lockV: SIMD3<Float> = {
    let a: Float = lockYawDegrees * .pi / 180
    return SIMD3<Float>(sin(a), 0, cos(a))
}()
/// The middle of the keyway face, on the table.
let lockOrigin = SIMD3<Float>(0, 0, 0)

/// A lock-local point (u, v, w) in the world: w is height off the table.
func lockToWorld(_ q: SIMD3<Float>) -> SIMD3<Float> {
    lockOrigin + lockU * q.x + lockV * q.y + SIMD3<Float>(0, q.z, 0)
}

/// Where the inset's cut is taken, on the front face near its corner.
let cutPointLocal = SIMD3<Float>(9.0, 27.0, bodyThickness)

// MARK: - camera

/// From the keyway end, above and a little to one side, looking down at 35–40°
/// the way a product photograph of a small object is taken. MODEL.
struct Camera {
    var position: SIMD3<Float>
    var target: SIMD3<Float>
    var tanHalfFOV: Float
    var forward: SIMD3<Float> { simd_normalize(target - position) }
    var right: SIMD3<Float> { simd_normalize(simd_cross(forward, SIMD3<Float>(0, 1, 0))) }
    var up: SIMD3<Float> { simd_cross(right, forward) }
}

let lockCentre: SIMD3<Float> = lockToWorld(SIMD3<Float>(0, lockOverallHeight / 2, bodyThickness / 2))

let camera: Camera = {
    let back: SIMD3<Float> = -lockV * 0.80 + lockU * 0.42
    let dir: SIMD3<Float> = simd_normalize(SIMD3<Float>(back.x, 0, back.z)) * cos(Float(38) * .pi / 180)
        + SIMD3<Float>(0, sin(Float(38) * .pi / 180), 0)
    let distance: Float = 222
    let pos: SIMD3<Float> = lockCentre + dir * distance
    // Aim right of the lock so it sits in the left of the frame and the inset
    // has the right.
    let f: SIMD3<Float> = simd_normalize(lockCentre - pos)
    let r: SIMD3<Float> = simd_normalize(simd_cross(f, SIMD3<Float>(0, 1, 0)))
    let target: SIMD3<Float> = lockCentre + r * 46 - SIMD3<Float>(0, 8, 0)
    return Camera(position: pos, target: target, tanHalfFOV: 0.25)
}()

/// Where a world point lands in the image, in pixels (y down). The same ray
/// geometry the kernel uses, inverted.
func project(_ p: SIMD3<Float>, width: Int, height: Int, cam: Camera = camera) -> SIMD2<Float> {
    let d: SIMD3<Float> = p - cam.position
    let z: Float = simd_dot(d, cam.forward)
    let sx: Float = simd_dot(d, cam.right) / (z * cam.tanHalfFOV)
    let sy: Float = simd_dot(d, cam.up) / (z * cam.tanHalfFOV)
    let aspect: Float = Float(width) / Float(height)
    let x: Float = (sx / aspect + 1) * 0.5 * Float(width)
    let y: Float = (1 - sy) * 0.5 * Float(height)
    return SIMD2<Float>(x, y)
}

/// Pixels per millimetre for a length square to the camera at the lock's
/// centre depth: what the main view's scale bar means.
func mainPixelsPerMillimetre(height: Int, cam: Camera = camera) -> Float {
    let z: Float = simd_dot(lockCentre - cam.position, cam.forward)
    return Float(height) / (2 * z * cam.tanHalfFOV)
}

// MARK: - lights

// A small studio: a dark, neutral backdrop, a big softbox overhead and behind
// the lock for the front face to mirror, and a tall strip light at the side for
// the shackle's streaks. MODEL throughout — a metal shows nothing but what it
// reflects, so this is the choice of what it reflects. Every light is NEUTRAL
// (D65, equal in linear R, G and B), which makes the render's RGB reflection
// exact: the metal's colour was integrated against D65 spectrally, and white
// times that colour is that colour. A coloured light would need the product of
// spectra, which RGB multiplication only approximates.

/// A rectangle of light on the sky: its centre direction, two axes across it,
/// its half-extents as tangents of half-angles, and its radiance.
struct Softbox {
    var centre: SIMD3<Float>
    var axisA: SIMD3<Float>
    var axisB: SIMD3<Float>
    var tanA: Float
    var tanB: Float
    var radiance: Float

    init(azimuthFromLockV az: Float, elevation el: Float, halfA: Float, halfB: Float, radiance: Float) {
        let a: Float = az * .pi / 180
        let e: Float = el * .pi / 180
        let flat: SIMD3<Float> = lockV * cos(a) + lockU * sin(a)
        centre = simd_normalize(flat * cos(e) + SIMD3<Float>(0, sin(e), 0))
        axisA = simd_normalize(simd_cross(SIMD3<Float>(0, 1, 0), centre))
        axisB = simd_cross(centre, axisA)
        tanA = tan(halfA * .pi / 180)
        tanB = tan(halfB * .pi / 180)
        self.radiance = radiance
    }

    /// Solid angle of the rectangle, exactly: 4 atan(ab / √(1 + a² + b²)).
    var solidAngle: Float {
        let s: Float = (1 + tanA * tanA + tanB * tanB).squareRoot()
        return 4 * atan(tanA * tanB / s)
    }
    /// Irradiance on the table, lit from this box's centre direction, for a
    /// box whose mean radiance is this fraction of its peak.
    func tableIrradiance(meanFraction: Float = 1) -> Float { radiance * meanFraction * solidAngle * centre.y }
}

/// The key: a softbox beyond the lock, which the polished front face mirrors.
/// Its peak radiance is chosen so a face mirroring it lands below white after
/// the tone curve, keeping the gold's hue measurable. MODEL.
let keyLight = Softbox(azimuthFromLockV: -24, elevation: 41, halfA: 9, halfB: 12, radiance: 0.95)
/// A real softbox is brightest in the middle of its diffuser and dimmer at the
/// edges; this is the fraction of the peak left at the edge, falling off as
/// (1 − a²)(1 − b²) across it. MODEL. The table's irradiance uses the box's
/// mean radiance, which for this profile is edge + (1 − edge)·4/9.
let keyEdgeFraction: Float = 0.4
let keyMeanFraction: Float = keyEdgeFraction + (1 - keyEdgeFraction) * 4 / 9
/// The strip: tall, narrow and brighter, to one side. It blows out to near
/// white in the highlights, as a real strip light does on gold. MODEL.
let stripLight = Softbox(azimuthFromLockV: 100, elevation: 22, halfA: 3.5, halfB: 20, radiance: 5.0)

/// The backdrop's radiance: dim and neutral, a little brighter near the
/// horizon than overhead, as a studio sweep is. MODEL.
let backdropHorizon: Float = 0.085
let backdropZenith: Float = 0.035
let belowHorizon: Float = 0.02

/// The table: plain, matte, neutral grey. Albedo MODEL.
let tableAlbedo: Float = 0.85
/// What the table receives from the backdrop, open sky: π × its mean radiance.
let ambientIrradiance: Float = Float.pi * (backdropHorizon + backdropZenith) / 2

/// How many times a ray may bounce between polished surfaces before it is
/// given up as trapped (inside the keyway, say) and counted black. MODEL:
/// four bounces at ≤ 97% each is where the light that would still escape is
/// under the 8-bit step in the dark slot.
let maxBounces: Int = 4

// MARK: - the inset

/// The inset, in units of the image height (x, y from the top-left): a
/// rectangle on the right of the frame showing the cut at the cut point.
let insetOrigin = SIMD2<Float>(1.08, 0.215)
let insetSize = SIMD2<Float>(0.62, 0.56)
/// How many micrometres the inset's height spans.
let insetFieldMicrometres: Float = 24.0
/// Where the plated surface sits, micrometres below the inset's top edge:
/// the mounting resin above it has room for the cotton fibre's circle.
let insetSurfaceMicrometres: Float = 16.8

func insetMicrometresPerPixel(height: Int) -> Float { insetFieldMicrometres / (insetSize.y * Float(height)) }

/// Scale bars: 10 mm in the main view, 5 µm in the inset.
let mainBarMillimetres: Float = 10
let insetBarMicrometres: Float = 5
