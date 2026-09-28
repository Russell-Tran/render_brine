// Where the toy stands, where the camera is, and how the studio is lit.
// Nothing here touches the GPU.
//
// Lighting is MODEL throughout, chosen so the material reads: a plastic toy
// is recognised by its soft, bright reflections of the lights round it — a
// broad sheen on matte-ish paint, tight glints on the glossy eyes — over
// colour that comes from the paint. So the toy sits under a large softbox
// (whose reflection is the sheen), with a fill opposite and a narrow strip
// behind for a rim, in a pale tent that gives every face something to
// reflect. Every light is neutral white (equal R, G, B).

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
/// anticlockwise from above) and `elevation` above the horizontal, `distance`
/// away, aimed `shift` mm to the right of the centre (so the subject sits
/// left of the frame) and `drop` mm below it.
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

/// Where a world point lands in the image, in pixels (y down): the kernel's
/// ray geometry, inverted.
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

/// Pixels per millimetre for a length square to the camera at depth of a
/// point: what a scale bar there means.
func pixelsPerMillimetre(at p: SIMD3<Float>, height: Int, cam: Camera) -> Float {
    let z: Float = simd_dot(p - cam.position, cam.forward)
    return Float(height) / (2 * z * cam.tanHalfFOV)
}

// MARK: - lights

/// A rectangle of light on the sky: centre direction, two axes across it,
/// half-extents as tangents, and radiance.
struct Softbox {
    var centre: SIMD3<Float>
    var axisA: SIMD3<Float>
    var axisB: SIMD3<Float>
    var tanA: Float
    var tanB: Float
    var radiance: Float

    /// `azimuth` anticlockwise from +x seen from above, `elevation` above the
    /// horizontal, half-angles in degrees.
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

/// The studio, relative to the camera's azimuth so every toy is lit the same
/// way from where it is seen. MODEL, all of it.
struct Studio {
    var key: Softbox
    var fill: Softbox
    var rim: Softbox
    /// The tent's radiance at the horizon and overhead.
    var tentHorizon: Float = 0.46
    var tentZenith: Float = 0.80
    /// The table: plain, matte, pale warm grey. Albedo MODEL.
    var tableAlbedo = SIMD3<Float>(0.70, 0.68, 0.65)

    init(cameraAzimuthDegrees c: Float) {
        // The key: large, high, from the camera's left and a little in front.
        key = Softbox(azimuth: c + 40, elevation: 50, halfA: 17, halfB: 12, radiance: 4.2)
        // The fill: from the camera's right, lower and dimmer.
        fill = Softbox(azimuth: c - 60, elevation: 25, halfA: 14, halfB: 12, radiance: 1.4)
        // The rim: a strip behind the toy, for an edge on the back.
        rim = Softbox(azimuth: c + 170, elevation: 30, halfA: 25, halfB: 4, radiance: 2.4)
    }

    /// π × the tent's mean radiance over the sky, cosine-weighted, for
    /// L = horizon + (zenith − horizon)·√sin(elevation): horizon + (zenith −
    /// horizon)·4/5 (as step 53).
    var ambientIrradiance: Float { Float.pi * (tentHorizon + (tentZenith - tentHorizon) * 4 / 5) }
}

// MARK: - the inset

/// The inset: a cut straight across one leg, drawn to scale from the same
/// distance function the render uses. In units of image height (x, y from
/// the top-left).
let insetOrigin = SIMD2<Float>(1.16, 0.24)
let insetSize = SIMD2<Float>(0.52, 0.52)
/// How many millimetres the inset's height spans.
let insetFieldMillimetres: Float = 5.4
func insetMillimetresPerPixel(height: Int) -> Float { insetFieldMillimetres / (insetSize.y * Float(height)) }

/// Scale bars: 10 mm in the main view, 0.5 mm in the inset.
let mainBarMillimetres: Float = 10
let insetBarMillimetres: Float = 0.5
