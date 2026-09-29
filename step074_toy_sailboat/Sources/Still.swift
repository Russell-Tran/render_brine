// The still: the toy boat standing on the table on its keel, the camera
// where a product photograph would put it (the boat's file sets the angle),
// and the cut the inset shows.

import Foundation
import simd

struct StillSetup {
    let design: BoatDesign
    let toy: PosedToy
    let camera: Camera
    let studio: Studio
    /// The toy's middle, which the camera centres on and the scale bar is true at.
    let centre: SIMD3<Float>
    let cut: Cut
}

/// Frame the toy large: its length across a share of the frame's width, or
/// its height a share of the frame's, whichever is the tighter (the boat's
/// file sets both), sitting left of centre so the inset has the right.
func stillSetup(_ design: BoatDesign, mutant: Mutant = activeMutant) -> StillSetup {
    let toy: PosedToy = design.posed(seams: mutant != .noSeam)
    let ex = toy.extent(along: SIMD3<Float>(1, 0, 0))
    let ey = toy.extent(along: SIMD3<Float>(0, 1, 0))
    let ez = toy.extent(along: SIMD3<Float>(0, 0, 1))
    let centre = SIMD3<Float>((ex.lo + ex.hi) / 2, (ey.lo + ey.hi) / 2, (ez.lo + ez.hi) / 2)
    let length: Float = ex.hi - ex.lo
    let tall: Float = ey.hi - ey.lo
    let tanHalf: Float = 0.2
    let byLength: Float = length / (stillFraming.length * 2 * tanHalf * 16 / 9)
    let byHeight: Float = tall / (stillFraming.height * 2 * tanHalf)
    let distance: Float = max(byLength, byHeight)
    let az: Float = stillCameraAzimuthDegrees * .pi / 180
    let el: Float = stillCameraElevationDegrees * .pi / 180
    let cam: Camera = studioCamera(centre: centre, azimuth: az, elevation: el, distance: distance, tanHalfFOV: tanHalf,
                                   shift: distance * stillFraming.shift, drop: distance * stillFraming.drop)
    return StillSetup(design: design, toy: toy, camera: cam, studio: Studio(cameraAzimuthDegrees: stillCameraAzimuthDegrees),
                      centre: centre, cut: boatCut(toy))
}
