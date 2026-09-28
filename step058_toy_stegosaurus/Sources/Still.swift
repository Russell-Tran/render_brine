// The still: the toy standing on the table in the pose it was moulded in,
// the camera where a product photograph would put it, and the leg the inset
// cuts through.

import Foundation
import simd

/// The camera's azimuth for the stills: from the toy's left side and a
/// little in front, so it faces the left of the frame and the inset has the
/// right. MODEL.
let stillCameraAzimuthDegrees: Float = 62
/// Elevation, degrees: a product shot's gentle look down. MODEL.
let stillCameraElevationDegrees: Float = 20

struct StillSetup {
    let design: ToyDesign
    let pose: Pose
    let toy: PosedToy
    let camera: Camera
    let studio: Studio
    /// The toy's middle, which the camera centres on and the scale bar is true at.
    let centre: SIMD3<Float>
    /// Which leg the inset cuts, and the height of the cut above the table.
    let cutLimb: Int
    let cutHeight: Float
}

/// Frame the toy: its length across about half the frame's width, sitting
/// left of centre.
func stillSetup(_ design: ToyDesign, mutant: Mutant = activeMutant) -> StillSetup {
    let pose: Pose = design.restPose()
    let toy: PosedToy = design.posed(pose, seams: mutant != .noSeam)
    let ex = toy.extent(along: SIMD3<Float>(1, 0, 0))
    let ey = toy.extent(along: SIMD3<Float>(0, 1, 0))
    let ez = toy.extent(along: SIMD3<Float>(0, 0, 1))
    let centre = SIMD3<Float>((ex.lo + ex.hi) / 2, (ey.lo + ey.hi) / 2, (ez.lo + ez.hi) / 2)
    let length: Float = ex.hi - ex.lo
    let tanHalf: Float = 0.2
    // Distance so the toy's length spans ~0.52 of the frame's width (16:9).
    let distance: Float = length / (0.52 * 2 * tanHalf * 16 / 9) * 1.02
    let az: Float = stillCameraAzimuthDegrees * .pi / 180
    let el: Float = stillCameraElevationDegrees * .pi / 180
    let cam: Camera = studioCamera(centre: centre, azimuth: az, elevation: el, distance: distance, tanHalfFOV: tanHalf,
                                   shift: length * 0.36, drop: length * 0.02)
    // The inset cuts the left foreleg (the one nearest the camera), across
    // its lower piece, just above the foot.
    let limb: Int = design.legs.firstIndex { $0.name == "LF" } ?? 0
    let l: LegSpec = design.legs[limb]
    let cut: Float = l.ankleHeight + l.lower * 0.45
    return StillSetup(design: design, pose: pose, toy: toy, camera: cam, studio: Studio(cameraAzimuthDegrees: stillCameraAzimuthDegrees),
                      centre: centre, cutLimb: limb, cutHeight: cut)
}
