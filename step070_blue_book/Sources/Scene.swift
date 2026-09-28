// Where the book lies, where the camera is, and how the studio is lit.
// Camera, softboxes and tent are step 60's, copied.
//
// Lighting is MODEL throughout, chosen so the material reads (step 53's
// lesson). Cloth is recognised by its sheen: a soft brightening where it is
// seen or lit at a grazing angle, over a matte dyed colour, and by its weave
// catching the light at edges and corners. So a large key softbox lights the
// cover from the front-left, a strip light low behind the book skims the
// cover and the head at a grazing angle (where the sheen lives), and a pale
// tent gives every face something to reflect. Every light is neutral white.

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
/// mm to the right of the centre and `drop` mm below it.
func studioCamera(centre: SIMD3<Float>, azimuth: Float, elevation: Float, distance: Float, tanHalfFOV: Float,
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

/// Pixels per millimetre for a length square to the camera at a point.
func pixelsPerMillimetre(at p: SIMD3<Float>, height: Int, cam: Camera) -> Float {
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
        tanA = tan(halfA * .pi / 180)
        tanB = tan(halfB * .pi / 180)
        self.radiance = radiance
    }

    /// Solid angle, exactly: 4 atan(ab / √(1 + a² + b²)).
    var solidAngle: Float {
        let s: Float = (1 + tanA * tanA + tanB * tanB).squareRoot()
        return 4 * atan(tanA * tanB / s)
    }
}

/// The studio, relative to the camera's azimuth. MODEL, all of it.
struct Studio {
    var key: Softbox
    var fill: Softbox
    var rim: Softbox
    var tentHorizon: Float = 0.46
    var tentZenith: Float = 0.80
    /// The table: plain, matte, pale warm grey. Albedo MODEL.
    var tableAlbedo = SIMD3<Float>(0.62, 0.60, 0.57)

    init(cameraAzimuthDegrees c: Float) {
        key = Softbox(azimuth: c + 35, elevation: 52, halfA: 18, halfB: 13, radiance: 4.4)
        fill = Softbox(azimuth: c - 65, elevation: 28, halfA: 14, halfB: 12, radiance: 1.3)
        // Low and behind: skims the cover at a grazing angle for the sheen.
        rim = Softbox(azimuth: c + 175, elevation: 14, halfA: 32, halfB: 5, radiance: 3.2)
    }

    var ambientIrradiance: Float { Float.pi * (tentHorizon + (tentZenith - tentHorizon) * 4 / 5) }
}

// MARK: - the still's camera

/// From the spine-and-head corner, above: the cover, the spine with its
/// joint grooves, and the head with its squares and headband all in view.
/// MODEL.
let stillCameraAzimuthDegrees: Float = 232
let stillCameraElevationDegrees: Float = 25

struct StillSetup {
    let layout: BookLayout
    let scene: Scene
    let camera: Camera
    let studio: Studio
    /// The point the scale bar is true at: the middle of the cover.
    let centre: SIMD3<Float>
}

func stillSetup(_ mutant: Mutant = activeMutant) -> StillSetup {
    let L = BookLayout(mutant)
    let scene: Scene = buildBook(L)
    let centre = SIMD3<Float>(leafWidth / 2, L.top / 2, 0)
    let tanHalf: Float = 0.2
    let distance: Float = 780
    let az: Float = stillCameraAzimuthDegrees * .pi / 180
    let el: Float = stillCameraElevationDegrees * .pi / 180
    let cam: Camera = studioCamera(centre: centre, azimuth: az, elevation: el, distance: distance, tanHalfFOV: tanHalf,
                                   shift: distance * 0.16, drop: distance * 0.02)
    return StillSetup(layout: L, scene: scene, camera: cam, studio: Studio(cameraAzimuthDegrees: stillCameraAzimuthDegrees),
                      centre: centre)
}
