// Where the boat floats, where the camera is, and how the studio is lit.
// Camera, softboxes and tent are step 72's (from step 60), copied.
//
// The water is a cut-away block, as in a textbook diagram: fresh water up to
// y = 0, its near side sliced away just outside the hull so the underwater
// body — the hull below the waterline, the centreboard, the rudder — is seen
// through it, tinted by the water it looks through. Lighting and the block's
// size are MODEL, chosen so it reads.

import Foundation
import simd

// MARK: - camera

struct Camera {
    var position: SIMD3<Float>
    var target: SIMD3<Float>
    var tanHalfFOV: Float
    var forward: SIMD3<Float> { simd_normalize(target - position) }
    var right: SIMD3<Float> { simd_normalize(simd_cross(forward, SIMD3<Float>(0, 1, 0))) }
    var up: SIMD3<Float> { simd_cross(right, forward) }
}

/// A camera looking at `centre` from azimuth `azimuth` (radians, 0 = from +x,
/// anticlockwise from above) and `elevation`, `distance` away, aimed `shift`
/// to the right of the centre and `drop` below it.
func studioCamera(centre: SIMD3<Float>, azimuth: Float, elevation: Float, distance: Float, tanHalfFOV: Float,
                  shift: Float, drop: Float) -> Camera {
    let flat = SIMD3<Float>(cos(azimuth), 0, -sin(azimuth))
    let dir: SIMD3<Float> = flat * cos(elevation) + SIMD3<Float>(0, sin(elevation), 0)
    let pos: SIMD3<Float> = centre + dir * distance
    let f: SIMD3<Float> = simd_normalize(centre - pos)
    let r: SIMD3<Float> = simd_normalize(simd_cross(f, SIMD3<Float>(0, 1, 0)))
    let side: SIMD3<Float> = r * shift
    let target: SIMD3<Float> = centre + side - SIMD3<Float>(0, drop, 0)
    return Camera(position: pos, target: target, tanHalfFOV: tanHalfFOV)
}

/// Where a world point lands in the image, in pixels (y down).
func project(_ p: SIMD3<Float>, width: Int, height: Int, cam: Camera) -> SIMD2<Float> {
    let d: SIMD3<Float> = p - cam.position
    let z: Float = simd_dot(d, cam.forward)
    let sx: Float = simd_dot(d, cam.right) / (z * cam.tanHalfFOV)
    let sy: Float = simd_dot(d, cam.up) / (z * cam.tanHalfFOV)
    let aspect: Float = Float(width) / Float(height)
    let w: Float = Float(width)
    let h: Float = Float(height)
    let x: Float = (sx / aspect + 1) * 0.5 * w
    let y: Float = (1 - sy) * 0.5 * h
    return SIMD2<Float>(x, y)
}

/// Pixels per metre for a length square to the camera at a point.
func pixelsPerMetre(at p: SIMD3<Float>, height: Int, cam: Camera) -> Float {
    let z: Float = simd_dot(p - cam.position, cam.forward)
    return Float(height) / (2 * z * cam.tanHalfFOV)
}

// MARK: - lights

struct Softbox {
    var centre: SIMD3<Float>
    var axisA: SIMD3<Float>
    var axisB: SIMD3<Float>
    var tanA: Float
    var tanB: Float
    var radiance: Float

    init(azimuth az: Float, elevation el: Float, halfA: Float, halfB: Float, radiance: Float) {
        let a: Float = az * .pi / 180
        let e: Float = el * .pi / 180
        let flat = SIMD3<Float>(cos(a), 0, -sin(a))
        centre = simd_normalize(flat * cos(e) + SIMD3<Float>(0, sin(e), 0))
        axisA = simd_normalize(simd_cross(SIMD3<Float>(0, 1, 0), centre))
        axisB = simd_cross(centre, axisA)
        let ra: Float = halfA * Float.pi / 180
        let rb: Float = halfB * Float.pi / 180
        tanA = tan(ra)
        tanB = tan(rb)
        self.radiance = radiance
    }

    /// Solid angle, exactly: 4 atan(ab / √(1 + a² + b²)).
    var solidAngle: Float {
        let a2: Float = tanA * tanA
        let b2: Float = tanB * tanB
        let s: Float = (1 + a2 + b2).squareRoot()
        return 4 * atan(tanA * tanB / s)
    }
}

/// The studio, relative to the camera's azimuth. MODEL, all of it.
struct Studio {
    var key: Softbox
    var fill: Softbox
    var rim: Softbox
    var tentHorizon: Float = 0.80
    var tentZenith: Float = 1.30

    init(cameraAzimuthDegrees c: Float) {
        key = Softbox(azimuth: c + 30, elevation: 55, halfA: 18, halfB: 13, radiance: 6.0)
        fill = Softbox(azimuth: c - 70, elevation: 30, halfA: 14, halfB: 12, radiance: 2.2)
        rim = Softbox(azimuth: c + 175, elevation: 20, halfA: 30, halfB: 6, radiance: 2.6)
    }

    var ambientIrradiance: Float { Float.pi * (tentHorizon + (tentZenith - tentHorizon) * 4 / 5) }
}

// MARK: - the water block (world frame; the surface at y = 0)

struct WaterBlock {
    var lo: SIMD3<Float>
    var hi: SIMD3<Float>
    /// Attenuation per metre, red green blue: a clear, slightly green-blue
    /// fresh water. MODEL (so a metre of it tints but does not hide).
    var sigma = SIMD3<Float>(0.42, 0.16, 0.12)
    /// The water's own scattered light, as a radiance. MODEL.
    var glow = SIMD3<Float>(0.20, 0.42, 0.50)
}

/// Near side cut 0.25 m outside the hull's widest point, so nothing of the
/// boat is sliced. The block's other sides are far enough to frame it.
func waterBlock(_ scale: Float) -> WaterBlock {
    let near: Float = Float(hullShape.halfBeam) * scale + 0.25
    return WaterBlock(lo: SIMD3<Float>(-3.6, -1.25, -3.0), hi: SIMD3<Float>(3.4, 0, near))
}

// MARK: - the still's camera

/// From starboard, a little forward of the beam and a little above the
/// water, so the sail reads as a triangle and the cut face shows the
/// underwater body. MODEL.
let stillCameraAzimuthDegrees: Float = 282
let stillCameraElevationDegrees: Float = 5

struct StillSetup {
    let camera: Camera
    let studio: Studio
    let water: WaterBlock
    /// The point the main scale bar is true at: midships, on the centreline,
    /// at the waterline.
    let centre: SIMD3<Float>
}

func stillSetup(_ mutant: Mutant = activeMutant) -> StillSetup {
    let s: Float = Float(boatScale(mutant))
    let centre = SIMD3<Float>(0.2, 2.35, 0)
    let tanHalf: Float = 0.2
    let distance: Float = 22
    let az: Float = stillCameraAzimuthDegrees * .pi / 180
    let el: Float = stillCameraElevationDegrees * .pi / 180
    // Framed for the class's size; the mutant's bigger boat is seen from the
    // same place.
    let shiftAmount: Float = distance * 0.118
    let dropAmount: Float = distance * 0.004
    let cam: Camera = studioCamera(centre: centre, azimuth: az, elevation: el, distance: distance, tanHalfFOV: tanHalf,
                                   shift: shiftAmount, drop: dropAmount)
    return StillSetup(camera: cam, studio: Studio(cameraAzimuthDegrees: stillCameraAzimuthDegrees),
                      water: waterBlock(s), centre: SIMD3<Float>(0, 0, 0))
}
