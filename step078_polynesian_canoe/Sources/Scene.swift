// Where the canoe floats, where the camera is, and how the day is lit.
// The camera maths is step 72's (from steps 60 and 70), copied; the sky and
// sun replace its studio.
//
// Light is MODEL throughout: a clear tropical sky, the sun high on the
// camera's side so the hulls, crossbeams and canvas read, and a flat sea.

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
func sceneCamera(centre: SIMD3<Float>, azimuth: Float, elevation: Float, distance: Float, tanHalfFOV: Float,
                 shift: Float, drop: Float) -> Camera {
    let flat = SIMD3<Float>(cos(azimuth), 0, -sin(azimuth))
    let dir: SIMD3<Float> = flat * cos(elevation) + SIMD3<Float>(0, sin(elevation), 0)
    let pos: SIMD3<Float> = centre + dir * distance
    let f: SIMD3<Float> = simd_normalize(centre - pos)
    let r: SIMD3<Float> = simd_normalize(simd_cross(f, SIMD3<Float>(0, 1, 0)))
    let target: SIMD3<Float> = centre + r * shift - SIMD3<Float>(0, drop, 0)
    return Camera(position: pos, target: target, tanHalfFOV: tanHalfFOV)
}

/// Where a world point lands in the image, in pixels (y down).
func project(_ p: SIMD3<Float>, width: Int, height: Int, cam: Camera) -> SIMD2<Float> {
    let d: SIMD3<Float> = p - cam.position
    let z: Float = simd_dot(d, cam.forward)
    let sx: Float = simd_dot(d, cam.right) / (z * cam.tanHalfFOV)
    let sy: Float = simd_dot(d, cam.up) / (z * cam.tanHalfFOV)
    let aspect: Float = Float(width) / Float(height)
    let x: Float = (sx / aspect + 1) * 0.5 * Float(width)
    let y: Float = (1 - sy) * 0.5 * Float(height)
    return SIMD2<Float>(x, y)
}

/// Pixels per metre for a length square to the camera at a point.
func pixelsPerMetre(at p: SIMD3<Float>, height: Int, cam: Camera) -> Float {
    let z: Float = simd_dot(p - cam.position, cam.forward)
    return Float(height) / (2 * z * cam.tanHalfFOV)
}

// MARK: - the day (MODEL)

struct Sky {
    /// Towards the sun.
    var sun: SIMD3<Float>
    /// Sun's irradiance on a square-on surface and its disc's radiance.
    var sunIrradiance: Float = 4.0
    var sunRadiance: Float = 60
    var sunCosRadius: Float = 0.99996
    var horizon = SIMD3<Float>(0.80, 0.87, 0.95)
    var zenith = SIMD3<Float>(0.16, 0.36, 0.80)
    /// Irradiance from the whole sky on an upward surface.
    var skyIrradiance: Float = 2.2

    init(cameraAzimuthDegrees c: Float) {
        let toRad: Float = Float.pi / 180
        let a: Float = (c - 35) * toRad
        let e: Float = 48 * toRad
        sun = simd_normalize(SIMD3<Float>(cos(a) * cos(e), sin(e), -sin(a) * cos(e)))
    }
}

// MARK: - the still's camera

/// From off the port bow, a little above the sea: both hulls, the manu,
/// the ʻiako and both sails in view, and the waterline along the near hull.
/// MODEL.
let stillCameraAzimuthDegrees: Float = 68
let stillCameraElevationDegrees: Float = 7

struct StillSetup {
    let camera: Camera
    let sky: Sky
    /// The point the main scale bar is true at: the canoe's middle, on the
    /// waterline.
    let centre: SIMD3<Float>
}

func stillSetup(_ mutant: Mutant = activeMutant) -> StillSetup {
    let centre = SIMD3<Float>(0, 6.4, 0)
    let tanHalf: Float = 0.25
    let distance: Float = 48
    let az: Float = stillCameraAzimuthDegrees * .pi / 180
    let el: Float = stillCameraElevationDegrees * .pi / 180
    // The camera frames PVS's size; the mutant's bigger canoe is seen from
    // the same place.
    let cam: Camera = sceneCamera(centre: centre, azimuth: az, elevation: el, distance: distance, tanHalfFOV: tanHalf,
                                  shift: 5.6, drop: 0)
    return StillSetup(camera: cam, sky: Sky(cameraAzimuthDegrees: stillCameraAzimuthDegrees),
                      centre: SIMD3<Float>(0, 0, 0))
}
