// The scene: a Type C school bus interior, and the world that goes past it.
//
// The idea of this step is that nothing in it is animated. The camera is bolted
// to the vehicle, so the bus's own frame is simply the right choice of
// coordinates: the interior gets the identity transform and never moves a
// nanometre for the whole loop. What moves is the world, and it moves by one
// rigid transform,
//
//     x_bus = R⁻¹ (x_world − v t)
//
// with R the identity here because the bus does not turn. Applying it to the
// ray instead of to the vertices is the same transform and costs nothing, which
// is what lets every sample carry its own time and gives motion blur for free
// (see Render.swift).
//
// Coordinates, chosen so the camera is a plain OpenGL-style camera:
//   +x  starboard (the bus's right when it is facing forward)
//   +y  up, with y = 0 the interior floor line
//   +z  toward the REAR of the bus — the way the camera looks
// The bus therefore travels toward −z, and a fixed point in the world drifts
// toward +z, falling behind. Looking rearward, camera-right is −x, so the
// bus's port side appears on the right of the picture, exactly as it does when
// you turn round in your seat.
//
// Dimensions come from Resources/bus.json, which records where each one comes
// from. The regulated ones are the Washington OSPI School Bus Specification
// Manual and FMVSS 222; the body dimensions are Blue Bird's published Type C
// figures.

import Foundation
import simd

// MARK: - the loop

/// The one velocity vector, and the periods that have to divide into it for the
/// loop to close.
struct Loop {
    /// 15.3 m/s — 55 km/h, a school bus on a suburban road.
    let speed: Float = 15.3
    let frames: Int = 120
    /// GIF stores frame delays in hundredths of a second, so 82 ms cannot be
    /// asked for; 8 cs is the nearest thing the format can say.
    let delayCentiseconds: Int = 8
    /// Three roadside tiles per loop: a plausible suburban block rhythm.
    let tilesPerLoop: Int = 3

    var dt: Float { Float(delayCentiseconds) / 100 }              // 0.08 s
    var duration: Float { Float(frames) * dt }                    // 9.6 s
    var distance: Float { speed * duration }                      // 146.88 m
    var tile: Float { distance / Float(tilesPerLoop) }            // 48.96 m
    /// A 180° shutter: each frame's four samples are spread over half the
    /// frame interval, which is what a film camera does.
    var exposure: Float { dt * 0.5 }

    /// How far the bus has gone since frame 0, unwrapped.
    func travel(frame: Int) -> Float {
        let seconds: Float = Float(frame) * dt
        return speed * seconds
    }

    /// The same, wrapped into one loop. Everything that builds the world uses
    /// this, so that frame 120 is not merely similar to frame 0 but is built
    /// from bit-identical numbers.
    func travelWrapped(frame: Int) -> Float {
        var f: Int = frame % frames
        if f < 0 { f += frames }
        let seconds: Float = Float(f) * dt
        return speed * seconds
    }
}

// MARK: - the measured bus

struct SpecEntry: Decodable {
    var inches: Double
    var evidence: String
    var source: String
}

struct SpecCounts: Decodable {
    var seat_rows: Int
    var window_bays: Int
    var seats_per_row: Int
}

struct BusSpecs: Decodable {
    var vehicle: String
    var note: String
    var sources: [String: String]
    var dimensions: [String: SpecEntry]
    var counts: SpecCounts

    /// A dimension in metres. Missing keys are a programming error, not a
    /// runtime condition, so this traps rather than returning a default.
    func metres(_ name: String) -> Float {
        guard let entry = dimensions[name] else {
            preconditionFailure("no dimension named \(name) in bus.json")
        }
        let inches: Double = entry.inches
        return Float(inches * 0.0254)
    }

    func evidence(_ name: String) -> String {
        dimensions[name]?.evidence ?? "unknown"
    }
}

func loadSpecs(from url: URL) throws -> BusSpecs {
    let data = try Data(contentsOf: url)
    return try JSONDecoder().decode(BusSpecs.self, from: data)
}

// MARK: - colours

private let vinyl = SIMD3<Float>(0.322, 0.384, 0.451)         // grey-blue seat covering
private let vinylDark = SIMD3<Float>(0.255, 0.310, 0.372)
private let seatFrame = SIMD3<Float>(0.196, 0.204, 0.216)
private let runner = SIMD3<Float>(0.086, 0.090, 0.098)        // ribbed rubber aisle
private let floorPlate = SIMD3<Float>(0.235, 0.243, 0.251)
private let ceilingWhite = SIMD3<Float>(0.902, 0.906, 0.890)
private let wallWhite = SIMD3<Float>(0.847, 0.855, 0.839)
private let wallLower = SIMD3<Float>(0.796, 0.804, 0.788)
private let pillarWhite = SIMD3<Float>(0.870, 0.874, 0.858)
private let decalYellow = SIMD3<Float>(0.953, 0.729, 0.106)
private let doorGrey = SIMD3<Float>(0.827, 0.835, 0.816)
private let railChrome = SIMD3<Float>(0.62, 0.64, 0.66)

private let barkGrey = SIMD3<Float>(0.396, 0.353, 0.298)
private let postWood = SIMD3<Float>(0.451, 0.384, 0.286)
private let wireGrey = SIMD3<Float>(0.29, 0.29, 0.30)
private let leafGreen = SIMD3<Float>(0.255, 0.400, 0.176)
private let leafDark = SIMD3<Float>(0.192, 0.318, 0.141)
private let distantGreen = SIMD3<Float>(0.255, 0.337, 0.243)
private let steelGrey = SIMD3<Float>(0.478, 0.498, 0.518)
private let mailboxRed = SIMD3<Float>(0.678, 0.145, 0.118)

// MARK: - the interior

/// What a piece of the bus is. The tests use this to take the seats out and
/// leave a bare shell, which is the only way to predict where a sunbeam lands
/// in closed form.
enum BusTag: UInt32 {
    case shell = 0
    case seat = 1
    case door = 2
    case decal = 3
    case trim = 4
}


/// Everything about the bus that the render and the tests both need. Built once
/// and never touched again: the interior is rigid by construction.
struct Bus {
    let specs: BusSpecs
    let shapes: [GPUShape]

    let halfWidth: Float          // interior half width
    let ceiling: Float            // interior head room above the floor line
    let floorAboveGround: Float   // the floor line's height above the road
    let aisleHalf: Float
    let sill: Float
    let header: Float
    let seatPitch: Float
    let seatBackTop: Float
    let cushion: Float
    let firstRowZ: Float
    let rearZ: Float              // the rear bulkhead
    let rows: Int

    let cameraOrigin: SIMD3<Float>
    let fovDegrees: Float = 46

    /// The road surface, in bus coordinates.
    var groundY: Float { -floorAboveGround }

    /// The box the interior occupies. Nothing in the world may enter it — that
    /// is what makes the aperture mask provably correct, and a test checks it.
    var interiorBox: (lo: SIMD3<Float>, hi: SIMD3<Float>) {
        (SIMD3(-halfWidth, 0, 0), SIMD3(halfWidth, ceiling, rearZ))
    }

    func camera() -> Camera {
        Camera(origin: cameraOrigin,
               target: SIMD3<Float>(0, cameraOrigin.y, rearZ),
               fov: fovDegrees)
    }

    /// A camera that moves, for the control render. It pans and dollies a
    /// little, the way every render in this project since step 5 has, so that
    /// nearly every pixel changes between frames.
    func orbitCamera(frame: Int, loop: Loop) -> Camera {
        let phase: Float = 2 * .pi * Float(frame) / Float(loop.frames)
        let swing: Float = 0.17
        let rise: Float = 0.10
        let origin = SIMD3<Float>(swing * cos(phase),
                                  cameraOrigin.y + rise * sin(phase),
                                  cameraOrigin.z)
        let target = SIMD3<Float>(0.42 * sin(phase),
                                  cameraOrigin.y + 0.14 * cos(phase),
                                  rearZ)
        return Camera(origin: origin, target: target, fov: fovDegrees)
    }

    /// Focal length in pixels, for the parallax arithmetic.
    func focalPixels(viewHeight: Int) -> Float {
        let half: Float = Float(viewHeight) / 2
        return half / tan(radians(fovDegrees / 2))
    }
}

func buildBus(_ specs: BusSpecs) -> Bus {
    let halfWidth: Float = specs.metres("interior_width") / 2
    let ceiling: Float = specs.metres("interior_headroom")
    let floorAboveGround: Float = specs.metres("floor_height")
    let aisleHalf: Float = specs.metres("aisle_width") / 2
    let cushion: Float = specs.metres("seat_cushion_height")
    let backAboveSRP: Float = specs.metres("seat_back_height_above_srp")
    let seatBackTop: Float = cushion + backAboveSRP
    let cushionDepth: Float = specs.metres("seat_cushion_depth")
    let pitch: Float = specs.metres("seat_pitch")
    let sill: Float = specs.metres("window_sill_height")
    let glassHeight: Float = specs.metres("window_glass_height")
    let header: Float = sill + glassHeight
    let pillar: Float = specs.metres("window_pillar_width")
    let rows: Int = specs.counts.seat_rows
    let ribPitch: Float = specs.metres("aisle_rib_pitch")

    let firstRowZ: Float = 0.90
    let lastRowZ: Float = firstRowZ + Float(rows) * pitch
    let rearZ: Float = lastRowZ + 0.30

    var shapes: [GPUShape] = []

    // ---- floor, aisle runner and its ribs
    let floorMidZ: Float = rearZ / 2
    shapes.append(.box(center: SIMD3(0, -0.035, floorMidZ),
                       half: SIMD3(halfWidth + 0.04, 0.035, floorMidZ + 0.06),
                       radius: 0.010, color: floorPlate))
    shapes.append(.box(center: SIMD3(0, 0.006, floorMidZ),
                       half: SIMD3(aisleHalf, 0.006, floorMidZ - 0.02),
                       radius: 0.004, color: runner))
    var ribZ: Float = 0.45
    while ribZ < rearZ - 0.10 {
        shapes.append(.box(center: SIMD3(0, 0.014, ribZ),
                           half: SIMD3(aisleHalf - 0.006, 0.004, 0.016),
                           radius: 0.003, color: runner * 1.45))
        ribZ += ribPitch
    }

    // ---- ceiling and its ribs, which run across the bus
    shapes.append(.box(center: SIMD3(0, ceiling + 0.04, floorMidZ),
                       half: SIMD3(halfWidth + 0.04, 0.04, floorMidZ + 0.06),
                       radius: 0.010, color: ceilingWhite))
    var ceilZ: Float = 0.55
    while ceilZ < rearZ - 0.10 {
        shapes.append(.box(center: SIMD3(0, ceiling - 0.014, ceilZ),
                           half: SIMD3(halfWidth - 0.02, 0.012, 0.020),
                           radius: 0.008, color: ceilingWhite * 0.93))
        ceilZ += 0.3556                       // 14 in, the usual bow spacing
    }

    // ---- side walls. The window bays are holes: the wall is the panel below
    // the sill, the panel above the header, and the pillars between bays.
    // Nothing is drawn where the glass is, so a ray that leaves through a
    // window meets no bus geometry at all, which is exactly the test the
    // aperture mask makes.
    let bays: Int = specs.counts.window_bays
    for side in [Float(-1), Float(1)] {
        let wallX: Float = side * (halfWidth + 0.035)
        shapes.append(.box(center: SIMD3(wallX, sill / 2, floorMidZ),
                           half: SIMD3(0.035, sill / 2, floorMidZ + 0.06),
                           radius: 0.008, color: wallLower))
        let upperMid: Float = (header + ceiling) / 2
        shapes.append(.box(center: SIMD3(wallX, upperMid, floorMidZ),
                           half: SIMD3(0.035, (ceiling - header) / 2, floorMidZ + 0.06),
                           radius: 0.008, color: wallWhite))
        // Solid wall ahead of the first bay and behind the last one.
        let bayStart: Float = firstRowZ
        let bayEnd: Float = firstRowZ + Float(bays) * pitch
        let frontLen: Float = bayStart
        shapes.append(.box(center: SIMD3(wallX, (sill + header) / 2, frontLen / 2),
                           half: SIMD3(0.035, (header - sill) / 2, frontLen / 2),
                           radius: 0.006, color: wallWhite))
        let rearLen: Float = rearZ + 0.06 - bayEnd
        shapes.append(.box(center: SIMD3(wallX, (sill + header) / 2, bayEnd + rearLen / 2),
                           half: SIMD3(0.035, (header - sill) / 2, rearLen / 2),
                           radius: 0.006, color: wallWhite))
        for i in 0...bays {
            let z: Float = bayStart + Float(i) * pitch
            shapes.append(.box(center: SIMD3(wallX, (sill + header) / 2, z),
                               half: SIMD3(0.036, (header - sill) / 2, pillar / 2),
                               radius: 0.008, color: pillarWhite))
        }
        // A sill ledge under each bay, and a header cap over it.
        for i in 0..<bays {
            let z: Float = bayStart + (Float(i) + 0.5) * pitch
            let glassLen: Float = pitch - pillar
            shapes.append(.box(center: SIMD3(side * (halfWidth - 0.015), sill + 0.012, z),
                               half: SIMD3(0.030, 0.012, glassLen / 2),
                               radius: 0.006, color: wallLower * 0.92,
                               tag: BusTag.trim.rawValue))
            shapes.append(.box(center: SIMD3(side * (halfWidth - 0.010), header - 0.010, z),
                               half: SIMD3(0.025, 0.010, glassLen / 2),
                               radius: 0.005, color: wallWhite * 0.95,
                               tag: BusTag.trim.rawValue))
        }
    }

    // ---- the bulkhead behind the driver. The camera sits just aft of it and
    // every camera ray goes the other way, so it is never drawn — but it closes
    // the front of the bus, and without it a sunbeam traced from a seat near the
    // front would escape forward and report light that is not there.
    shapes.append(.box(center: SIMD3(0, ceiling / 2, 0.02),
                       half: SIMD3(halfWidth + 0.04, ceiling / 2, 0.020),
                       radius: 0.006, color: wallWhite))

    // ---- seats
    let benchInner: Float = aisleHalf
    let benchOuter: Float = halfWidth - 0.020
    let benchHalfX: Float = (benchOuter - benchInner) / 2
    let benchMidX: Float = (benchOuter + benchInner) / 2
    for row in 0..<rows {
        let backZ: Float = firstRowZ + Float(row) * pitch
        for side in [Float(-1), Float(1)] {
            let x: Float = side * benchMidX
            shapes.append(.box(center: SIMD3(x, (cushion + seatBackTop) / 2, backZ + 0.045),
                               half: SIMD3(benchHalfX - 0.030, (seatBackTop - cushion) / 2 - 0.030,
                                           0.014),
                               radius: 0.030, color: vinyl, tag: BusTag.seat.rawValue))
            shapes.append(.box(center: SIMD3(x, cushion - 0.032, backZ + 0.095 + cushionDepth / 2),
                               half: SIMD3(benchHalfX - 0.030, 0.014, cushionDepth / 2 - 0.030),
                               radius: 0.030, color: vinylDark, tag: BusTag.seat.rawValue))
            let legZ: Float = backZ + 0.095 + cushionDepth * 0.55
            for lx in [benchInner + 0.075, benchOuter - 0.11] {
                shapes.append(.cylinder(base: SIMD3(side * lx, 0.004, legZ),
                                        axis: SIMD3(0, cushion - 0.068, 0),
                                        radius: 0.017, color: seatFrame,
                                        tag: BusTag.seat.rawValue))
            }
            // The grab bar along the top of the seat back.
            shapes.append(.cylinder(base: SIMD3(side * (benchInner + 0.020), seatBackTop - 0.012,
                                                backZ + 0.045),
                                    axis: SIMD3(side * (benchOuter - benchInner - 0.040), 0, 0),
                                    radius: 0.012, color: railChrome,
                                    tag: BusTag.seat.rawValue))
        }
    }

    // ---- rear bulkhead with a door-shaped hole in it
    let doorWidth: Float = specs.metres("rear_door_width")
    let doorHalf: Float = doorWidth / 2
    let doorTop: Float = 1.72
    for side in [Float(-1), Float(1)] {
        let inner: Float = doorHalf
        let outer: Float = halfWidth + 0.04
        shapes.append(.box(center: SIMD3(side * (inner + outer) / 2, ceiling / 2, rearZ + 0.03),
                           half: SIMD3((outer - inner) / 2, ceiling / 2, 0.030),
                           radius: 0.008, color: wallWhite))
    }
    shapes.append(.box(center: SIMD3(0, (doorTop + ceiling) / 2, rearZ + 0.03),
                       half: SIMD3(doorHalf, (ceiling - doorTop) / 2, 0.030),
                       radius: 0.008, color: wallWhite))

    // ---- the rear emergency door leaf, set back a little, with its window
    let doorZ: Float = rearZ + 0.10
    let winW: Float = specs.metres("rear_door_window_width")
    let winH: Float = specs.metres("rear_door_window_height")
    let winMidY: Float = 1.30
    let winLo: Float = winMidY - winH / 2
    let winHi: Float = winMidY + winH / 2
    shapes.append(.box(center: SIMD3(0, winLo / 2, doorZ),
                       half: SIMD3(doorHalf, winLo / 2, 0.028),
                       radius: 0.010, color: doorGrey, tag: BusTag.door.rawValue))
    shapes.append(.box(center: SIMD3(0, (winHi + doorTop) / 2, doorZ),
                       half: SIMD3(doorHalf, (doorTop - winHi) / 2, 0.028),
                       radius: 0.010, color: doorGrey, tag: BusTag.door.rawValue))
    for side in [Float(-1), Float(1)] {
        let inner: Float = winW / 2
        shapes.append(.box(center: SIMD3(side * (inner + doorHalf) / 2, winMidY, doorZ),
                           half: SIMD3((doorHalf - inner) / 2, winH / 2, 0.028),
                           radius: 0.010, color: doorGrey, tag: BusTag.door.rawValue))
    }

    // ---- EMERGENCY EXIT decals. Three of them, which is how many a real bus
    // has where the camera can see them: one over the rear door and one over
    // the rearmost emergency window on each side. The plate is sized for the
    // 2-inch lettering the specification calls for.
    let letterHeight: Float = specs.metres("emergency_exit_letter_height")
    shapes.append(.box(center: SIMD3(0, winHi + 0.075, doorZ - 0.024),
                       half: SIMD3(0.195, letterHeight * 0.75, 0.004),
                       radius: 0.003, color: decalYellow, tag: BusTag.decal.rawValue))
    for side in [Float(-1), Float(1)] {
        let z: Float = firstRowZ + (Float(bays) - 0.5) * pitch
        shapes.append(.box(center: SIMD3(side * (halfWidth - 0.008), header + 0.055, z),
                           half: SIMD3(0.004, letterHeight * 0.70, 0.175),
                           radius: 0.003, color: decalYellow, tag: BusTag.decal.rawValue))
    }

    // A camera bolted to the bulkhead behind the driver, at about the height of
    // a standing adult's eye. High enough that the road is visible through the
    // rear door window, low enough to look down the aisle rather than over it.
    let cameraOrigin = SIMD3<Float>(0, 1.32, 0.30)

    return Bus(specs: specs, shapes: shapes, halfWidth: halfWidth, ceiling: ceiling,
               floorAboveGround: floorAboveGround, aisleHalf: aisleHalf, sill: sill,
               header: header, seatPitch: pitch, seatBackTop: seatBackTop, cushion: cushion,
               firstRowZ: firstRowZ, rearZ: rearZ, rows: rows, cameraOrigin: cameraOrigin)
}

// MARK: - the sun

/// The sun is at infinity and the bus does not turn, so in the bus's frame the
/// sun direction is a constant. That is the whole reason the interior can be
/// treated as rigid: the patch of light through each window is fixed, and the
/// only thing that moves it is the world outside passing in front of it.
struct Sun {
    /// 38° up. Lower than that and the beam from a window sill never crosses
    /// the aisle before it reaches the floor, so there are no stripes on the
    /// runner at all; higher and it falls straight down the window wall. The
    /// number is set by the sill height and the width of the bus, not by taste.
    let elevationDegrees: Float = 38
    /// Measured round from the port beam toward the FRONT of the bus. It has to
    /// be ahead of the bus rather than behind it: the camera looks rearward, so
    /// the faces of the seat backs it can see are the ones pointing forward,
    /// and only a sun ahead of the bus lights them.
    let azimuthDegrees: Float = 40

    var direction: SIMD3<Float> {
        let el: Float = radians(elevationDegrees)
        let az: Float = radians(azimuthDegrees)
        let horizontal: Float = cos(el)
        let x: Float = -horizontal * cos(az)      // toward port
        let y: Float = sin(el)
        let z: Float = -horizontal * sin(az)      // toward the front of the bus
        return simd_normalize(SIMD3(x, y, z))
    }
}

// MARK: - the world

/// One kind of roadside thing, repeated along the track.
struct Prop {
    var name: String
    /// Perpendicular distance from the bus centreline: the `d` of the parallax
    /// ladder. Always positive.
    var lateral: Float
    /// +1 starboard (the picture's left), −1 port (the picture's right).
    var side: Float
    /// The along-track repeat. Every one of these must divide the loop distance
    /// exactly or the loop will not close.
    var period: Float
    var phase: Float
    /// The along-track band this prop is emitted in, in bus coordinates.
    var near: Float
    var far: Float
    /// Tagged distinctive means it must not repeat inside one loop.
    var distinctive: Bool
    /// Shown on the caption bar's parallax ladder.
    var ladderLabel: String?
    var build: (Float, Int, Float) -> [GPUShape]   // (z, instance, side) → shapes
}

private func positiveRemainder(_ x: Float, _ m: Float) -> Float {
    let q: Float = (x / m).rounded(.down)
    let r: Float = x - q * m
    return r < 0 ? r + m : r
}

/// Where a prop's instances sit at a given point in the loop, in bus
/// coordinates. A fixed point in the world has z = w + travel, so as the bus
/// goes the instances slide toward +z — toward the camera and past it. Nothing
/// is ever moved backwards and nothing is ever rewound: instances that leave
/// the far end are simply not emitted, and new ones appear at the near end.
func instancePositions(_ prop: Prop, travel: Float) -> [(z: Float, index: Int)] {
    var out: [(z: Float, index: Int)] = []
    let base: Float = positiveRemainder(prop.phase + travel, prop.period)
    var j: Int = 0
    while true {
        let z: Float = base + Float(j) * prop.period
        if z > prop.far { break }
        if z >= prop.near {
            // The instance's identity in the world, so that anything keyed to
            // it (a house colour, a tree height) travels with it rather than
            // flickering as the conveyor turns.
            let world: Float = z - travel - prop.phase
            let index: Int = Int((world / prop.period).rounded())
            out.append((z, index))
        }
        j += 1
    }
    return out
}

/// The roadside, from the bus outward. Distances are the ladder of the caption
/// bar: a fence post at 4 m and a pylon line at 1500 m in the same still frame.
func buildProps(_ loop: Loop, bus: Bus) -> [Prop] {
    let g: Float = bus.groundY
    let tile: Float = loop.tile
    let lap: Float = loop.distance

    var props: [Prop] = []

    // 4 m — the near rung, and the fastest thing in the picture at 219 °/s
    // abeam. From this camera anything at 4 m that is less than about 0.65 m
    // above the road is hidden by the window sill, so the posts are drawn at the
    // 1.75 m of a stock fence: the top metre is what sweeps past.
    props.append(Prop(name: "fence post", lateral: 4.0, side: 1,
                      period: tile / 20, phase: 0.6, near: 4, far: 46,
                      distinctive: false, ladderLabel: "fence post") { z, _, side in
        let x: Float = side * 4.0
        var out: [GPUShape] = []
        out.append(.cylinder(base: SIMD3(x, g, z), axis: SIMD3(0, 1.75, 0),
                             radius: 0.068, color: postWood))
        for wireY in [Float(1.62), Float(1.10)] {
            out.append(.cylinder(base: SIMD3(x, g + wireY, z),
                                 axis: SIMD3(0, 0, tile / 20),
                                 radius: 0.009, color: wireGrey))
        }
        return out
    })

    // 4 m, once a loop — a landmark near enough that a single instance really
    // can be alone in shot (see the note on `distinctive` in the tests).
    props.append(Prop(name: "mailbox", lateral: 4.0, side: 1,
                      period: lap, phase: 21.4, near: 4, far: 46,
                      distinctive: true, ladderLabel: nil) { z, _, side in
        let x: Float = side * 4.0
        var out: [GPUShape] = []
        out.append(.cylinder(base: SIMD3(x, g, z), axis: SIMD3(0, 1.15, 0),
                             radius: 0.045, color: postWood * 0.8))
        out.append(.box(center: SIMD3(x, g + 1.24, z), half: SIMD3(0.085, 0.075, 0.150),
                        radius: 0.070, color: mailboxRed))
        return out
    })

    // 6.2 m, across the road — the trees whose shadows chop the sunlight inside
    // the bus. They are the only reason the interior is not literally frozen.
    props.append(Prop(name: "poplar", lateral: 6.2, side: -1,
                      period: tile / 4, phase: 3.1, near: 6, far: 74,
                      distinctive: false, ladderLabel: nil) { z, index, side in
        let x: Float = side * 6.2
        let lean: Float = Float((index % 5) - 2) * 0.06
        var out: [GPUShape] = []
        out.append(.cylinder(base: SIMD3(x, g, z), axis: SIMD3(lean, 8.4, lean * 0.4),
                             radius: 0.155, color: barkGrey))
        out.append(.sphere(center: SIMD3(x + lean * 0.4, g + 4.3, z), radius: 1.02,
                           color: leafDark))
        out.append(.sphere(center: SIMD3(x + lean * 0.7, g + 6.1, z + 0.1), radius: 1.24,
                           color: leafGreen))
        out.append(.sphere(center: SIMD3(x + lean, g + 7.8, z - 0.1), radius: 0.88,
                           color: leafGreen))
        return out
    })

    // 12 m — a pickup on a driveway, once a loop.
    props.append(Prop(name: "parked pickup", lateral: 12.0, side: 1,
                      period: lap, phase: 74.0, near: 10, far: 132,
                      distinctive: true, ladderLabel: "parked car") { z, _, side in
        let x: Float = side * 12.0
        var out: [GPUShape] = []
        out.append(.box(center: SIMD3(x, g + 0.92, z), half: SIMD3(0.86, 0.30, 2.42),
                        radius: 0.14, color: SIMD3(0.545, 0.176, 0.153)))
        out.append(.box(center: SIMD3(x, g + 1.44, z - 0.55), half: SIMD3(0.78, 0.26, 0.78),
                        radius: 0.12, color: SIMD3(0.475, 0.153, 0.133)))
        for wz in [Float(1.62), Float(-1.62)] {
            out.append(.cylinder(base: SIMD3(x - 0.90, g + 0.38, z + wz),
                                 axis: SIMD3(1.80, 0, 0), radius: 0.36,
                                 color: SIMD3(0.10, 0.10, 0.11)))
        }
        return out
    })

    // 40 m — a row of houses. Three fronts over the loop, so the block does not
    // read as one house stamped out three times.
    props.append(Prop(name: "house", lateral: 40.0, side: 1,
                      period: tile, phase: 12.0, near: 40, far: 420,
                      distinctive: false, ladderLabel: "house") { z, index, side in
        var variant: Int = index % 3
        if variant < 0 { variant += 3 }
        let x: Float = side * 40.0
        let walls: [SIMD3<Float>] = [SIMD3(0.686, 0.663, 0.596),
                                     SIMD3(0.545, 0.514, 0.475),
                                     SIMD3(0.612, 0.580, 0.533)]
        let roofs: [SIMD3<Float>] = [SIMD3(0.271, 0.259, 0.255),
                                     SIMD3(0.361, 0.263, 0.220),
                                     SIMD3(0.231, 0.239, 0.251)]
        let widths: [Float] = [4.6, 5.4, 4.0]
        let heights: [Float] = [2.5, 2.9, 2.3]
        let w: Float = widths[variant]
        let h: Float = heights[variant]
        var out: [GPUShape] = []
        out.append(.box(center: SIMD3(x, g + h, z), half: SIMD3(w, h, 5.2),
                        radius: 0.20, color: walls[variant]))
        out.append(.box(center: SIMD3(x, g + 2 * h + 0.75, z),
                        half: SIMD3(w * 0.72, 0.75, 5.5),
                        radius: 0.55, color: roofs[variant]))
        // Two windows on the road-facing wall.
        for wz in [Float(-2.2), Float(2.2)] {
            out.append(.box(center: SIMD3(x - w - 0.04, g + h + 0.20, z + wz),
                            half: SIMD3(0.05, 0.60, 0.72),
                            radius: 0.04, color: SIMD3(0.129, 0.153, 0.176)))
        }
        return out
    })

    // 200 m — a treeline. Repetition at this distance is invisible, which is
    // just as well, because nothing this far away can be made to appear only
    // once in a 147 m loop.
    props.append(Prop(name: "treeline", lateral: 200.0, side: 1,
                      period: tile / 8, phase: 0, near: 200, far: 1900,
                      distinctive: false, ladderLabel: "treeline") { z, index, side in
        var k: Int = index % 8
        if k < 0 { k += 8 }
        let x: Float = side * 200.0
        let jitter: Float = Float(k) * 0.37
        let h: Float = 7.0 + jitter
        var out: [GPUShape] = []
        out.append(.sphere(center: SIMD3(x + jitter, g + h * 0.62, z), radius: 3.5,
                           color: distantGreen))
        out.append(.sphere(center: SIMD3(x - jitter * 0.5, g + h, z + 1.1), radius: 2.5,
                           color: distantGreen * 1.08))
        return out
    })

    // 1500 m — the far rung. A transmission line, not a water tower: at this
    // distance a single landmark cannot be made to loop (see the tests), but a
    // line of pylons genuinely is periodic, so its repeat is the truth rather
    // than a tell.
    props.append(Prop(name: "pylon", lateral: 1500.0, side: -1,
                      period: lap, phase: 0, near: 2000, far: 14000,
                      distinctive: false, ladderLabel: "pylon line") { z, _, side in
        let x: Float = side * 1500.0
        var out: [GPUShape] = []
        out.append(.cylinder(base: SIMD3(x, g, z), axis: SIMD3(0, 42, 0),
                             radius: 1.6, color: steelGrey))
        for armY in [Float(30), Float(38)] {
            out.append(.box(center: SIMD3(x, g + armY, z), half: SIMD3(6.5, 0.5, 0.5),
                            radius: 0.4, color: steelGrey))
        }
        return out
    })

    return props
}

/// The world as the bus sees it at one instant: every prop instance placed by
/// the single transform, ready to be sorted into a grid.
func buildWorld(_ props: [Prop], travel: Float) -> [GPUShape] {
    var out: [GPUShape] = []
    out.reserveCapacity(2048)
    for prop in props {
        for instance in instancePositions(prop, travel: travel) {
            out += prop.build(instance.z, instance.index, prop.side)
        }
    }
    return out
}

// MARK: - the road surface

/// The road is an analytic plane rather than geometry, because a plane that
/// stretches to the horizon has no useful bounding box to put in a grid. Its
/// markings move because their pattern coordinate is the world one.
struct Road {
    /// The road centreline, in bus coordinates: the bus sits in the right-hand
    /// lane, so the centreline is to port.
    let centre: Float = -1.83
    let laneWidth: Float = 3.65
    /// A 10 ft line and a 30 ft gap is the American standard 40 ft (12.192 m)
    /// cycle. 12.24 m is used instead — 0.4% longer — because it divides the
    /// loop distance exactly and 12.192 m does not.
    var dashPeriod: Float { 12.24 }
    var dashLength: Float { 3.06 }
}

// MARK: - the parallax arithmetic

/// The rate at which the line of sight to a fixed point swings, for a camera
/// carried along a straight track at `speed`.
///
///     ω = v sin θ / r
///
/// with r the range and θ the angle between the velocity and the line of sight.
/// Writing that in track coordinates — `lateral` the perpendicular distance and
/// `along` the distance measured down the track — gives ω = v·d/(d² + z²).
func angularRate(speed: Float, lateral: Float, along: Float) -> Float {
    let r2: Float = lateral * lateral + along * along
    return speed * lateral / r2
}

/// The largest that rate ever gets, which happens as the thing goes abeam:
/// ω_max = v/d, pure inverse distance. This is the number the caption bar
/// quotes, because it belongs to the object rather than to the moment.
func abeamRate(speed: Float, lateral: Float) -> Float {
    speed / lateral
}

func degreesPerSecond(_ radiansPerSecond: Float) -> Float {
    radiansPerSecond * 180 / .pi
}

/// How fast a fixed point slides across the picture, in pixels a second.
///
/// The camera looks *along* the track, not across it, so the familiar
/// side-window law — displacement = v·Δt·f/d — does not apply. What the camera
/// sees is a radial contraction toward the point straight behind: a point at
/// perpendicular distance d and along-track distance z sits at radius
/// ρ = f·d/z from the centre of the picture, so
///
///     dρ/dt = −f·d·v/z²
///
/// It still falls off as 1/d at a given place on the screen, because reaching
/// the same ρ from twice the distance means twice the z.
func screenRate(focalPixels: Float, speed: Float, lateral: Float, along: Float) -> Float {
    let numerator: Float = focalPixels * lateral * speed
    let denominator: Float = along * along
    return numerator / denominator
}

/// Where a fixed point is on the screen, as a radius from the picture's centre.
func screenRadius(focalPixels: Float, lateral: Float, along: Float) -> Float {
    focalPixels * lateral / along
}

/// The band of track over which a thing at perpendicular distance `lateral` is
/// visible at all: from where it enters at the edge of the field of view to
/// where the rear bulkhead hides it. Both limits are proportional to the
/// distance, so the band is too — which is the fact that decides which props
/// can be made to appear once a loop and which cannot.
struct VisibleBand {
    let nearTangent: Float      // tan of the widest angle the camera sees
    let farTangent: Float       // tan of the narrowest angle a side window allows

    func range(lateral: Float) -> (near: Float, far: Float) {
        (lateral / nearTangent, lateral / farTangent)
    }

    func length(lateral: Float) -> Float {
        let r = range(lateral: lateral)
        return r.far - r.near
    }
}

func visibleBand(_ bus: Bus, aspect: Float) -> VisibleBand {
    let tanHalfVertical: Float = tan(radians(bus.fovDegrees / 2))
    let tanHalfHorizontal: Float = tanHalfVertical * aspect
    // The rearmost window bay is the shallowest angle a side window offers.
    let lastBayZ: Float = bus.firstRowZ + Float(bus.specs.counts.window_bays) * bus.seatPitch
    let reach: Float = lastBayZ - bus.cameraOrigin.z
    let narrowest: Float = bus.halfWidth / reach
    return VisibleBand(nearTangent: tanHalfHorizontal, farTangent: narrowest)
}
