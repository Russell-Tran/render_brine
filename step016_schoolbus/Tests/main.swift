// Tests for step 16. The render makes one big claim — that a camera bolted to
// a vehicle turns the whole animation into a single rigid transform, and that
// almost nothing on the screen can therefore change — and these are the checks
// that the claim is true rather than merely plausible. The critical one is the
// mask: a mask that is wrong by a single pixel silently corrupts every frame
// after the first, and nothing in the picture would show it.

import CoreGraphics
import Foundation
import Metal
import simd

func near(_ a: Float, _ b: Float, within e: Float) -> Bool { abs(a - b) <= e }
func relative(_ a: Float, _ b: Float) -> Float { abs(a - b) / max(abs(b), 1e-9) }

let specsURL = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent().deletingLastPathComponent()
    .appendingPathComponent("Resources/bus.json")
let specs = try loadSpecs(from: specsURL)
let bus = buildBus(specs)
let loop = Loop()
let road = Road()
let sun = Sun()
let props = buildProps(loop, bus: bus)
let inch: Float = 0.0254

// A small view, so that the tests that render 120 frames stay quick.
let tw = 256, th = 192
let layout = FrameLayout(width: tw, viewHeight: th, captionHeight: 0)

func testSettings(frame: Int) -> RenderSettings {
    var s = RenderSettings()
    s.samplesPerSide = 2
    s.aoProbes = 4
    s.aoDistance = 0.85
    s.exposure = loop.exposure
    s.speed = loop.speed
    s.travelWrapped = loop.travelWrapped(frame: frame)
    s.groundY = bus.groundY
    s.roadCentre = road.centre
    s.roadHalfWidth = road.laneWidth
    s.dashPeriod = road.dashPeriod
    s.dashLength = road.dashLength
    s.sun = sun.direction
    return s
}

// MARK: - the loop and its periods

section("the loop")
test("the loop is 120 frames of 80 ms, and the numbers it implies are the ones used") {
    expectEqual(loop.frames, 120)
    expectEqual(loop.delayCentiseconds, 8)
    expect(near(loop.dt, 0.08, within: 1e-7), "\(loop.dt) s per frame")
    expect(near(loop.duration, 9.6, within: 1e-4), "\(loop.duration) s")
    expect(near(loop.distance, 146.88, within: 1e-2), "\(loop.distance) m")
    expect(near(loop.tile, 48.96, within: 1e-2), "\(loop.tile) m per tile")
    expect(near(loop.speed * 3.6, 55.08, within: 0.01), "\(loop.speed * 3.6) km/h")
}
test("every period in the scene divides the loop exactly") {
    // If it does not, the loop cannot close, and a render that does not close
    // is a rewind rather than a conveyor.
    func divides(_ period: Float, _ name: String) {
        let n: Float = loop.distance / period
        expect(relative(n, n.rounded()) < 1e-4,
               "\(name): \(loop.distance) / \(period) = \(n), not a whole number")
    }
    for prop in props { divides(prop.period, prop.name) }
    divides(road.dashPeriod, "centre line")
    divides(loop.tile, "roadside tile")
    // The American standard 40 ft (12.192 m) cycle does NOT divide 146.88 m, so
    // the dashes are drawn 0.4% longer. Worth stating out loud.
    let standard: Float = 12.192
    expect(relative(road.dashPeriod, standard) < 0.005,
           "the dash cycle is \(road.dashPeriod) m against the 12.192 m standard")
    let wouldNotDivide: Float = loop.distance / standard
    expect(relative(wouldNotDivide, wouldNotDivide.rounded()) > 1e-3,
           "12.192 m would have divided the loop after all — the deviation is not needed")
}
test("the world at frame 120 is the world at frame 0, built from identical numbers") {
    let a: Float = loop.travelWrapped(frame: 0)
    let b: Float = loop.travelWrapped(frame: loop.frames)
    expectEqual(a, b)
    // And it really is one whole lap further on, not a rewind.
    let unwrapped: Float = loop.travel(frame: loop.frames) - loop.travel(frame: 0)
    expect(relative(unwrapped, loop.distance) < 1e-5, "\(unwrapped) m travelled in one loop")
    expect(relative(unwrapped, Float(loop.tilesPerLoop) * loop.tile) < 1e-5,
           "one loop should be exactly \(loop.tilesPerLoop) tiles")
    let w0 = buildWorld(props, travel: a)
    let w1 = buildWorld(props, travel: b)
    expectEqual(w0.count, w1.count)
    var differing = 0
    for i in 0..<min(w0.count, w1.count) where w0[i].a != w1[i].a || w0[i].b != w1[i].b {
        differing += 1
    }
    expectEqual(differing, 0)
}

// MARK: - the bus

section("the measured bus")
test("the dimensions on screen are the ones bus.json records") {
    expect(near(bus.halfWidth * 2, 90.75 * inch, within: 1e-4), "\(bus.halfWidth * 2) m wide")
    expect(near(bus.ceiling, 77 * inch, within: 1e-4), "\(bus.ceiling) m of headroom")
    expect(near(bus.seatBackTop - bus.cushion, 24 * inch, within: 1e-4),
           "the seat back should stand 24 in above the seating reference point")
    expect(near(bus.header - bus.sill, 28 * inch, within: 1e-4), "glass \(bus.header - bus.sill) m")
    expectEqual(bus.rows, 13)
}
test("the regulated minima are met, not merely approached") {
    // Washington OSPI: aisles at least 12 in.
    expect(bus.aisleHalf * 2 >= 12 * inch, "aisle \(bus.aisleHalf * 2 / inch) in")
    // Type C and D passenger windows must give a 12 in by 22 in opening when
    // lowered, so a half-drop window needs at least 24 in of glass.
    let glass: Float = bus.header - bus.sill
    expect(glass >= 24 * inch, "glass \(glass / inch) in — a 12 in opening needs 24")
    let glassWidth: Float = bus.seatPitch - specs.metres("window_pillar_width")
    expect(glassWidth >= 22 * inch, "bay \(glassWidth / inch) in wide")
    // FMVSS 222 wants a seat back within 24 in of the seating reference point.
    expect(bus.seatPitch <= 30 * inch, "seat pitch \(bus.seatPitch / inch) in")
}
test("the rear door's window is not being passed off as a rear emergency window") {
    // The specification's 16 in by 54 in minimum is for a rear-emergency
    // *window*, which is a different body. This bus has a door.
    let w: Float = specs.metres("rear_door_window_width")
    expect(w < specs.metres("rear_exit_window_min_width"),
           "the door's window is \(w / inch) in wide, and would not qualify as an exit window")
}
test("the EMERGENCY EXIT decals are sized for the lettering and used sparingly") {
    let letter: Float = specs.metres("emergency_exit_letter_height")
    let decals = bus.shapes.filter { $0.meta.y == BusTag.decal.rawValue }
    expectEqual(decals.count, 3)                     // rear door, and one per side
    for d in decals {
        let halfHeight: Float = d.b.y + d.a.w
        expect(halfHeight * 2 >= letter, "a decal plate \(halfHeight * 2 / inch) in high")
    }
}
test("the camera is inside the bus and inside nothing else") {
    let box = bus.interiorBox
    let p = bus.cameraOrigin
    expect(p.x > box.lo.x && p.x < box.hi.x)
    expect(p.y > box.lo.y && p.y < box.hi.y)
    expect(p.z > box.lo.z && p.z < box.hi.z)
    for s in bus.shapes {
        let b = s.bounds()
        let inside = p.x > b.lo.x && p.x < b.hi.x && p.y > b.lo.y && p.y < b.hi.y
            && p.z > b.lo.z && p.z < b.hi.z
        expect(!inside, "the camera is inside a \(s.kind) at \(s.a)")
    }
}
test("nothing in the world ever comes inside the bus") {
    // This is what makes the aperture mask exact: a pixel whose ray stops on
    // interior geometry cannot have anything in front of it, at any frame.
    let box = bus.interiorBox
    var worst = ""
    for f in stride(from: 0, to: loop.frames, by: 5) {
        let world = buildWorld(props, travel: loop.travelWrapped(frame: f))
        for s in world {
            let b = s.bounds()
            let overlaps = b.lo.x < box.hi.x && b.hi.x > box.lo.x
                && b.lo.y < box.hi.y && b.hi.y > box.lo.y
                && b.lo.z < box.hi.z && b.hi.z > box.lo.z
            if overlaps { worst = "a \(s.kind) at \(s.a) reaches into the bus at frame \(f)" }
        }
    }
    expect(worst.isEmpty, worst)
    // The road surface, which is a plane rather than a shape, must also stay
    // under the floor.
    expect(bus.groundY < 0, "the road is at \(bus.groundY), which is not below the floor line")
}

// MARK: - the parallax arithmetic

section("the parallax ladder")
let ladder: [(String, Float, Float)] = [("fence post", 4, 219), ("parked car", 12, 73),
                                        ("house", 40, 22), ("treeline", 200, 4.4),
                                        ("pylon line", 1500, 0.6)]
test("ω = v/d abeam reproduces the ladder, over nearly four decades") {
    for (name, d, expected) in ladder {
        let got: Float = degreesPerSecond(abeamRate(speed: loop.speed, lateral: d))
        // The caption prints two significant figures, so the check is against
        // the printed number to that precision: 15.3/1500 is 0.584 °/s, which
        // the table rounds to 0.6.
        expect(relative(got, expected) < 0.03, "\(name) at \(d) m: \(got) °/s, expected \(expected)")
    }
    let spread: Float = abeamRate(speed: loop.speed, lateral: 4)
        / abeamRate(speed: loop.speed, lateral: 1500)
    expect(spread > 370 && spread < 380, "the ladder should span about 375:1, not \(spread)")
}
test("every rung of the ladder is actually in the scene") {
    for (name, d, _) in ladder {
        let match = props.first { $0.ladderLabel == name }
        expect(match != nil, "nothing in the scene is labelled \(name)")
        if let m = match { expect(near(m.lateral, d, within: 0.01), "\(name) is at \(m.lateral) m") }
    }
}
test("the angular rate of a tracked marker matches v·d/(d² + z²) to within 1%") {
    // Measured by a central difference across one frame, which is why the
    // prediction is evaluated at the midpoint.
    let dt: Float = loop.dt
    for (name, d, _) in ladder {
        let z0: Float = 2.5 * d
        // In double precision and through atan, because at 1500 m the angle
        // moves by 1e-7 rad a frame and a float dot product cannot see it.
        let alpha0: Double = atan(Double(d) / Double(z0))
        let alpha1: Double = atan(Double(d) / Double(z0 + loop.speed * dt))
        let measured: Float = Float((alpha0 - alpha1) / Double(dt))
        let mid: Float = z0 + loop.speed * dt / 2
        let predicted: Float = angularRate(speed: loop.speed, lateral: d, along: mid)
        expect(relative(measured, predicted) < 0.01,
               "\(name): measured \(degreesPerSecond(measured)) °/s, predicted \(degreesPerSecond(predicted)) °/s")
    }
}
test("on the screen the rate is f·d·v/z², not the side-window law") {
    // The camera looks along the track, so a fixed point contracts toward the
    // point straight behind rather than sliding past. Displacement therefore
    // falls off as 1/z², and the familiar v·Δt·f/d is simply the wrong law here.
    let cam = bus.camera()
    let f: Float = bus.focalPixels(viewHeight: 720)
    let dt: Float = loop.dt
    for (name, d, _) in ladder {
        let z0: Float = 2.5 * d
        let p0 = SIMD3<Float>(d, bus.cameraOrigin.y, bus.cameraOrigin.z + z0)
        let p1 = SIMD3<Float>(d, bus.cameraOrigin.y, bus.cameraOrigin.z + z0 + loop.speed * dt)
        let a = cam.project(p0, width: 960, height: 720)
        let b = cam.project(p1, width: 960, height: 720)
        let measured: Float = simd_distance(a, b)
        let mid: Float = z0 + loop.speed * dt / 2
        let predicted: Float = screenRate(focalPixels: f, speed: loop.speed, lateral: d,
                                          along: mid) * dt
        expect(relative(measured, predicted) < 0.01,
               "\(name): measured \(measured) px, predicted \(predicted) px")
        let sideWindowLaw: Float = loop.speed * dt * f / d
        if d <= 40 {
            expect(relative(measured, sideWindowLaw) > 0.5,
                   "\(name): the side-window law gives \(sideWindowLaw) px, and it should be nowhere near \(measured)")
        }
    }
}
test("at a fixed place on the screen the rate still falls off as 1/d") {
    // The ladder has to survive projection, not just exist in angle.
    let f: Float = bus.focalPixels(viewHeight: 720)
    var product: [Float] = []
    for (_, d, _) in ladder {
        let z: Float = 2.5 * d
        product.append(screenRate(focalPixels: f, speed: loop.speed, lateral: d, along: z) * d)
    }
    for v in product { expect(relative(v, product[0]) < 1e-4, "\(product)") }
}

// MARK: - the conveyor

section("the conveyor")
test("every visible thing moves by one and the same velocity vector") {
    let expected = SIMD3<Float>(0, 0, loop.speed * loop.dt)
    for prop in props {
        for f in [0, 37, 88] {
            let a = instancePositions(prop, travel: loop.travel(frame: f))
            let b = instancePositions(prop, travel: loop.travel(frame: f + 1))
            let lookup = Dictionary(uniqueKeysWithValues: b.map { ($0.index, $0.z) })
            for item in a {
                guard let next = lookup[item.index] else { continue }
                let step: Float = next - item.z
                // The pylons sit 14 km down the track, where one float32 step is
                // about a millimetre, so the tolerance is stated in units of
                // that rather than as a fixed fraction.
                let slack: Float = max(1e-4, item.z.ulp * 2)
                expect(abs(step - expected.z) <= slack,
                       "\(prop.name) instance \(item.index) moved \(step) m, not \(expected.z)")
            }
        }
    }
}
test("nothing ever drifts backwards") {
    // Measured as distance still ahead of the bus, which must only ever shrink.
    for prop in props {
        var previous: [Int: Float] = [:]
        for f in 0..<(2 * loop.frames) {
            let now = instancePositions(prop, travel: loop.travel(frame: f))
            for item in now {
                let ahead: Float = -item.z
                if let before = previous[item.index] {
                    expect(ahead < before,
                           "\(prop.name) instance \(item.index) went from \(before) m ahead to \(ahead) m at frame \(f)")
                }
                previous[item.index] = ahead
            }
        }
    }
}
test("one lap later the world is back exactly where it started") {
    // The reason every period has to divide the loop. If one does not, the
    // conveyor does not line up when it wraps and the loop becomes a rewind —
    // and because the render builds every frame from `travel mod loop`, frame
    // 120 would still match frame 0 while frame 119 to 120 jumped.
    for prop in props {
        let a = instancePositions(prop, travel: 3.7)
        let b = instancePositions(prop, travel: 3.7 + loop.distance)
        expectEqual(a.count, b.count)
        for (x, y) in zip(a, b) {
            expect(abs(x.z - y.z) < 2e-3,
                   "\(prop.name) is at \(x.z) m and, one lap later, at \(y.z) m")
        }
    }
}
test("the conveyor keeps a steady population — nothing pops in mid-shot") {
    for prop in props {
        var counts = Set<Int>()
        for f in 0..<loop.frames {
            counts.insert(instancePositions(prop, travel: loop.travelWrapped(frame: f)).count)
        }
        let span = (counts.max() ?? 0) - (counts.min() ?? 0)
        expect(span <= 1, "\(prop.name) swings between \(counts.min()!) and \(counts.max()!) instances")
    }
}

section("landmarks")
let band = visibleBand(bus, aspect: 4.0 / 3.0)
test("a thing at distance d stays in shot for about 6.8 d of track") {
    // The limits are the edge of the field of view at the near end and the rear
    // bulkhead at the far end, and both scale with the distance.
    expect(relative(band.length(lateral: 1), 6.78) < 0.05,
           "the band is \(band.length(lateral: 1)) d long")
    for d in [Float(4), 40, 1500] {
        let r = band.range(lateral: d)
        expect(relative(r.far - r.near, 6.78 * d) < 0.05, "\(d) m: \(r)")
    }
}
test("anything tagged distinctive sits on the full loop and really is alone in shot") {
    let distinctive = props.filter { $0.distinctive }
    expect(!distinctive.isEmpty, "nothing is tagged distinctive")
    for prop in distinctive {
        expect(relative(prop.period, loop.distance) < 1e-5,
               "\(prop.name) repeats every \(prop.period) m, not once a loop")
        let r = band.range(lateral: prop.lateral)
        expect(r.far - r.near < loop.distance,
               "\(prop.name) at \(prop.lateral) m is in shot for \(r.far - r.near) m, longer than the \(loop.distance) m loop")
        for f in 0..<loop.frames {
            let seen = instancePositions(prop, travel: loop.travelWrapped(frame: f))
                .filter { $0.z >= r.near && $0.z <= r.far }
            expect(seen.count <= 1, "\(seen.count) \(prop.name)s in shot at once at frame \(f)")
        }
    }
}
test("nothing beyond about 22 m is allowed to claim it is distinctive") {
    // The hard limit of the whole idea. A rearward camera keeps a thing in shot
    // for 6.78 d of track, so a single non-repeating instance is only possible
    // while 6.78 d is shorter than one loop — that is d < 146.88 / 6.78 ≈ 21.7 m.
    // Beyond that, exact looping forces genuine repetition, and the only honest
    // far-field content is content that repeats in life too.
    let limit: Float = loop.distance / band.length(lateral: 1)
    expect(relative(limit, 21.7) < 0.05, "the limit works out at \(limit) m")
    for prop in props where prop.distinctive {
        expect(prop.lateral < limit, "\(prop.name) at \(prop.lateral) m cannot be distinctive")
    }
    for prop in props where prop.lateral >= limit {
        expect(!prop.distinctive, "\(prop.name) is tagged distinctive but is \(prop.lateral) m out")
        expect(prop.period <= loop.distance + 1e-3,
               "\(prop.name) must repeat at least once a loop to close it")
    }
}

// MARK: - the new primitive

section("the rounded box")
let device = try findDevice()
let renderer = try SceneRenderer(device: device)
let bw = 320, bh = 240
guard let scratch = device.makeBuffer(length: bw * bh * 4, options: .storageModeShared),
      let scratch2 = device.makeBuffer(length: bw * bh * 4, options: .storageModeShared) else {
    fatalError("could not allocate the test buffers")
}

/// Renders one black shape against the sky from far enough away that the view
/// is nearly orthographic, and returns which pixels it covers.
func silhouette(_ shapes: [GPUShape]) throws -> (count: Int, bytes: [UInt8]) {
    var s = RenderSettings()
    s.samplesPerSide = 1
    s.aoProbes = 0
    s.exposure = 0
    s.speed = 0
    s.groundY = -100000
    s.sun = SIMD3(0, 1, 0)
    try renderer.setBus(shapes)
    try renderer.setWorld([])
    let cam = Camera(origin: SIMD3(0, 0, -40), target: SIMD3(0, 0, 0), fov: 2)
    _ = try renderer.renderFull(camera: cam, settings: s, into: scratch, width: bw, viewHeight: bh)
    let p = scratch.contents().assumingMemoryBound(to: UInt8.self)
    var bytes = [UInt8](repeating: 0, count: bw * bh)
    var count = 0
    for i in 0..<(bw * bh) {
        let lum = Int(p[i * 4]) + Int(p[i * 4 + 1]) + Int(p[i * 4 + 2])
        if lum < 200 { bytes[i] = 1; count += 1 }
    }
    return (count, bytes)
}

/// Pixels per metre at the plane through the origin, for the camera above.
let orthoScale: Float = (Float(bh) / 2 / tan(radians(1))) / 40

test("a rounded box with no box left in it is exactly a sphere") {
    let r: Float = 0.7
    let sphere = try silhouette([GPUShape.sphere(center: .zero, radius: r, color: .zero)])
    let boxy = try silhouette([GPUShape.box(center: .zero, half: SIMD3(repeating: r), radius: r,
                                           color: .zero)])
    var differing = 0
    for i in 0..<sphere.bytes.count where sphere.bytes[i] != boxy.bytes[i] { differing += 1 }
    expect(sphere.count > 3000, "the sphere covered only \(sphere.count) pixels")
    expect(Double(differing) / Double(sphere.count) < 0.01,
           "\(differing) pixels differ out of \(sphere.count)")
}
test("a rounded box reaches exactly half + radius, and its corners really are cut") {
    let half = SIMD3<Float>(0.35, 0.35, 0.35)
    let r: Float = 0.35
    let shot = try silhouette([GPUShape.box(center: .zero, half: half + SIMD3(repeating: r),
                                            radius: r, color: .zero)])
    // Area of a rounded rectangle: the full rectangle less what the four
    // quarter-circles take off the corners.
    let a: Float = half.x + r, b: Float = half.y + r
    let predicted: Float = 4 * a * b - (4 - .pi) * r * r
    let sharp: Float = 4 * a * b
    let measured: Float = Float(shot.count) / (orthoScale * orthoScale)
    expect(relative(measured, predicted) < 0.03,
           "silhouette \(measured) m², rounded predicts \(predicted) m²")
    expect(relative(measured, sharp) > 0.03,
           "it is indistinguishable from a sharp box of \(sharp) m² — the corners are not being cut")
    // And the extent across the middle is half + radius on each side.
    var widest = 0
    for y in 0..<bh {
        var lo = bw, hi = -1
        for x in 0..<bw where shot.bytes[y * bw + x] == 1 { lo = min(lo, x); hi = max(hi, x) }
        widest = max(widest, hi - lo + 1)
    }
    let measuredWidth: Float = Float(widest) / orthoScale
    expect(relative(measuredWidth, 2 * a) < 0.02, "\(measuredWidth) m across, expected \(2 * a) m")
}
test("a rounded box with almost no rounding is a plain box") {
    let half = SIMD3<Float>(0.5, 0.3, 0.4)
    let shot = try silhouette([GPUShape.box(center: .zero, half: half, radius: 0.002,
                                            color: .zero)])
    let predicted: Float = 4 * half.x * half.y
    let measured: Float = Float(shot.count) / (orthoScale * orthoScale)
    expect(relative(measured, predicted) < 0.02, "\(measured) m², expected \(predicted) m²")
}

// MARK: - the aperture mask

section("the aperture mask")
try renderer.setBus(bus.shapes)
let cam = bus.camera()
let mask = try renderer.computeMask(camera: cam, settings: testSettings(frame: 0),
                                    width: tw, viewHeight: th)
let indices = maskedIndices(mask)
let tracedFraction = Double(indices.count) / Double(tw * th)

var apertureOnly = testSettings(frame: 0)
apertureOnly.sunStrength = 0                // no sun, so no stripe — windows only
let windowMask = try renderer.computeMask(camera: cam, settings: apertureOnly,
                                          width: tw, viewHeight: th)

test("the mask is the windows plus the region a sun stripe can reach") {
    let windows = windowMask.reduce(0) { $0 + Int($1) }
    expect(windows > 0 && windows < tw * th)
    for i in 0..<mask.count where windowMask[i] != 0 {
        expectEqual(mask[i], 1)             // every window pixel is in the full mask
    }
    expect(indices.count > windows, "the sun adds nothing to the mask")
    print(String(format: "        windows %.1f%%, plus sunlight %.1f%%, total %.1f%%",
                 Double(windows) / Double(tw * th) * 100,
                 Double(indices.count - windows) / Double(tw * th) * 100,
                 tracedFraction * 100))
}

/// Renders a frame whole, into `into`.
func full(_ f: Int, into buffer: MTLBuffer) throws {
    try renderer.setWorld(buildWorld(props, travel: loop.travelWrapped(frame: f)))
    _ = try renderer.renderFull(camera: cam, settings: testSettings(frame: f), into: buffer,
                                width: tw, viewHeight: th)
}

try full(0, into: scratch)
let frameZero = Data(bytes: scratch.contents(), count: tw * th * 4)

test("the premise: outside the mask, every one of the 120 frames is bit-identical") {
    var worstFrame = -1
    var offenders = 0
    for f in 1..<loop.frames {
        try full(f, into: scratch)
        let now = scratch.contents().assumingMemoryBound(to: UInt8.self)
        frameZero.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            let base = raw.bindMemory(to: UInt8.self)
            for i in 0..<(tw * th) where mask[i] == 0 {
                if now[i * 4] != base[i * 4] || now[i * 4 + 1] != base[i * 4 + 1]
                    || now[i * 4 + 2] != base[i * 4 + 2] {
                    offenders += 1
                    if worstFrame < 0 { worstFrame = f }
                }
            }
        }
    }
    expectEqual(offenders, 0)
    if offenders > 0 { print("        first offending frame \(worstFrame)") }
}

test("mask correctness: a masked frame is pixel-identical to one traced whole") {
    // The critical test. A mask that is a little too small corrupts every frame
    // after the first and nothing in the picture says so.
    for f in [17, 53, 91] {
        frameZero.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            scratch2.contents().copyMemory(from: raw.baseAddress!, byteCount: tw * th * 4)
        }
        try renderer.setWorld(buildWorld(props, travel: loop.travelWrapped(frame: f)))
        _ = try renderer.renderMasked(camera: cam, settings: testSettings(frame: f),
                                      into: scratch2, width: tw, viewHeight: th, indices: indices)
        try full(f, into: scratch)
        let a = scratch.contents().assumingMemoryBound(to: UInt8.self)
        let b = scratch2.contents().assumingMemoryBound(to: UInt8.self)
        var differing = 0
        for i in 0..<(tw * th * 4) where a[i] != b[i] { differing += 1 }
        expectEqual(differing, 0)
        if differing > 0 { print("        frame \(f): \(differing) bytes differ") }
    }
}

test("the loop closes: frame 120 is the same picture as frame 0, pixel for pixel") {
    try full(loop.frames, into: scratch)
    let now = Data(bytes: scratch.contents(), count: tw * th * 4)
    expect(now == frameZero, "frame \(loop.frames) does not match frame 0")
}
test("the wrap is an ordinary step, not a rewind") {
    // Frame 120 matching frame 0 is not enough on its own: the render wraps the
    // travel, so it would match even if the conveyor did not line up. What says
    // the loop is seamless is that the jump from 119 to 120 is the same size as
    // any other frame's.
    func changedBetween(_ a: Int, _ b: Int) throws -> Int {
        try full(a, into: scratch)
        let first = Data(bytes: scratch.contents(), count: tw * th * 4)
        try full(b, into: scratch2)
        let second = scratch2.contents().assumingMemoryBound(to: UInt8.self)
        var n = 0
        first.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            let base = raw.bindMemory(to: UInt8.self)
            for i in 0..<(tw * th) where base[i * 4] != second[i * 4]
                || base[i * 4 + 1] != second[i * 4 + 1] || base[i * 4 + 2] != second[i * 4 + 2] {
                n += 1
            }
        }
        return n
    }
    let ordinary = (try changedBetween(40, 41) + (try changedBetween(75, 76))) / 2
    let wrap = try changedBetween(loop.frames - 1, loop.frames)
    expect(wrap < ordinary * 3 / 2,
           "\(wrap) pixels change at the wrap against \(ordinary) at an ordinary step")
}

// MARK: - the sun stripe

section("the sun stripe")
test("the stripe lands where the sun direction and the window aperture put it") {
    // The seats and the window trim are taken out, which leaves a shell whose
    // sunlit band on the floor can be written down in closed form: a floor point
    // sees the sun exactly when the ray from it crosses the port wall between
    // the sill and the header.
    let shell = bus.shapes.filter {
        $0.meta.y != BusTag.seat.rawValue && $0.meta.y != BusTag.trim.rawValue
    }
    try renderer.setBus(shell)
    defer { try? renderer.setBus(bus.shapes) }

    for elevation in [Float(38), Float(50)] {
        var s = testSettings(frame: 0)
        var custom = Sun()
        custom = Sun()
        let el: Float = radians(elevation)
        let az: Float = radians(custom.azimuthDegrees)
        s.sun = simd_normalize(SIMD3(-cos(el) * cos(az), sin(el), -cos(el) * sin(az)))
        let k: Float = s.sun.y / abs(s.sun.x)          // rise per metre travelled to port
        let shear: Float = -s.sun.z / abs(s.sun.x)     // and how far forward it goes

        let lo: Float = -bus.halfWidth + bus.sill / k
        let hi: Float = -bus.halfWidth + bus.header / k
        let m = try renderer.computeMask(camera: cam, settings: s, width: tw, viewHeight: th)

        var outside = 0, insideLit = 0, insideTotal = 0
        for z0 in [Float(4.3), 5.0, 5.7, 6.4, 7.1] {
            var x: Float = -1.10
            while x < 1.10 {
                defer { x += 0.01 }
                if abs(x) < bus.aisleHalf + 0.03 { continue }   // the runner sits proud
                // Skip anywhere a window pillar is in the way, which is the one
                // thing the closed form does not describe.
                let crossing: Float = z0 - shear * (x + bus.halfWidth)
                var nearPillar = false
                for i in 0...bus.specs.counts.window_bays {
                    let pz: Float = bus.firstRowZ + Float(i) * bus.seatPitch
                    if abs(crossing - pz) < 0.13 { nearPillar = true }
                }
                if nearPillar { continue }
                let px = cam.project(SIMD3(x, 0, z0), width: tw, height: th)
                let ix = Int(px.x.rounded()), iy = Int(px.y.rounded())
                if ix < 0 || ix >= tw || iy < 0 || iy >= th { continue }
                let live = m[iy * tw + ix] != 0
                let predicted = x >= lo + 0.02 && x <= hi - 0.02
                if predicted {
                    insideTotal += 1
                    if live { insideLit += 1 }
                } else if live && (x < lo - 0.02 || x > hi + 0.02) {
                    outside += 1
                }
            }
        }
        expect(insideTotal > 30, "only \(insideTotal) floor samples fell inside the predicted band")
        expect(Double(insideLit) / Double(max(insideTotal, 1)) > 0.85,
               "at \(elevation)° only \(insideLit) of \(insideTotal) predicted-lit samples are lit")
        expect(outside <= 4, "at \(elevation)° there are \(outside) lit samples outside the band [\(lo), \(hi)]")
    }
}
test("raising the sun moves the stripe by the amount the geometry says") {
    let k38: Float = tan(radians(Float(38))) / cos(radians(sun.azimuthDegrees))
    _ = k38
    func edge(_ elevation: Float) -> Float {
        let el: Float = radians(elevation)
        let az: Float = radians(sun.azimuthDegrees)
        let d = simd_normalize(SIMD3<Float>(-cos(el) * cos(az), sin(el), -cos(el) * sin(az)))
        let k: Float = d.y / abs(d.x)
        return -bus.halfWidth + bus.sill / k
    }
    // A higher sun pulls the port edge of the stripe back toward the window.
    expect(edge(50) < edge(38), "the stripe should retreat as the sun rises: \(edge(38)) → \(edge(50))")
    expect(near(edge(38), -0.306, within: 0.02), "at 38° the port edge is at x = \(edge(38))")
}

// MARK: - motion blur

section("motion blur")
let mw = 960, mh = 720
guard let blurBuffer = device.makeBuffer(length: mw * mh * 4, options: .storageModeShared) else {
    fatalError("could not allocate the blur buffer")
}
let blurFocal: Float = bus.focalPixels(viewHeight: mh)

/// A single dark marker in an otherwise empty world, seen from a camera at the
/// origin looking down the track. Returns how wide it is on the screen.
func markerWidth(lateral d: Float, exposure: Float) throws -> Float {
    let along: Float = 2.5 * d
    let radius: Float = d / 25
    var s = RenderSettings()
    s.samplesPerSide = 2
    s.aoProbes = 0
    s.exposure = exposure
    s.speed = loop.speed
    s.groundY = -100000
    s.sun = SIMD3(0, 1, 0)
    let camera = Camera(origin: .zero, target: SIMD3(0, 0, 1), fov: bus.fovDegrees)
    try renderer.setBus([GPUShape.sphere(center: SIMD3(0, 0, -1000), radius: 0.001, color: .zero)])

    try renderer.setWorld([])
    _ = try renderer.renderFull(camera: camera, settings: s, into: blurBuffer,
                                width: mw, viewHeight: mh)
    let background = Data(bytes: blurBuffer.contents(), count: mw * mh * 4)

    let marker = GPUShape.sphere(center: SIMD3(d, 0, along), radius: radius, color: .zero)
    try renderer.setWorld([marker])
    _ = try renderer.renderFull(camera: camera, settings: s, into: blurBuffer,
                                width: mw, viewHeight: mh)
    let now = blurBuffer.contents().assumingMemoryBound(to: UInt8.self)
    var lo = mw, hi = -1
    background.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
        let base = raw.bindMemory(to: UInt8.self)
        for y in 0..<mh {
            for x in 0..<mw {
                let i = (y * mw + x) * 4
                let delta = abs(Int(now[i]) - Int(base[i])) + abs(Int(now[i + 1]) - Int(base[i + 1]))
                    + abs(Int(now[i + 2]) - Int(base[i + 2]))
                if delta > 24 { lo = min(lo, x); hi = max(hi, x) }
            }
        }
    }
    return hi < 0 ? 0 : Float(hi - lo + 1)
}

test("the streak is exactly the distance the thing sweeps during the exposure") {
    for d in [Float(4), 12, 40] {
        let along: Float = 2.5 * d
        let sharp: Float = try markerWidth(lateral: d, exposure: 0)
        let blurred: Float = try markerWidth(lateral: d, exposure: loop.exposure)
        let streak: Float = blurred - sharp
        // Exact, not linearised: where the thing starts and where it ends.
        let travelled: Float = loop.speed * loop.exposure
        let predicted: Float = screenRadius(focalPixels: blurFocal, lateral: d, along: along)
            - screenRadius(focalPixels: blurFocal, lateral: d, along: along + travelled)
        expect(sharp > 8, "the marker at \(d) m is only \(sharp) px wide with no blur")
        // Four time samples cannot quite reach the ends of the sweep, and both
        // edges of the silhouette are read to the nearest pixel, so a short
        // streak comes out two or three pixels under.
        expect(abs(streak - predicted) < max(3.5, predicted * 0.25),
               "\(d) m: streak \(streak) px, predicted \(predicted) px")
    }
}
test("blur length is inversely proportional to distance") {
    // Which is the same statement as ω = v/d, seen on the screen rather than in
    // angle — and it is what makes the amount of blur evidence of how far away
    // a thing is.
    var products: [Float] = []
    var streaks: [Float] = []
    for d in [Float(8), 24, 72] {
        let sharp: Float = try markerWidth(lateral: d, exposure: 0)
        let blurred: Float = try markerWidth(lateral: d, exposure: 0.15)
        streaks.append(blurred - sharp)
        products.append((blurred - sharp) * d)
    }
    // There has to be a streak at all, and it has to shrink with distance —
    // a blur applied in screen space would be the same width for every rung.
    expect(streaks[0] > 4 && streaks[1] > 2, "no measurable streak: \(streaks)")
    expect(streaks[0] > streaks[1] && streaks[1] > streaks[2], "streaks \(streaks)")
    for v in products {
        expect(relative(v, products[1]) < 0.25, "streak × distance should be constant: \(products)")
    }
}
test("with the shutter closed there is no blur at all") {
    let a: Float = try markerWidth(lateral: 4, exposure: 0)
    let b: Float = try markerWidth(lateral: 4, exposure: 0)
    expectEqual(a, b)
    let c: Float = try markerWidth(lateral: 4, exposure: loop.exposure)
    expect(c > a + 8, "the shutter is open and nothing smeared: \(a) → \(c)")
}

// MARK: - the payoff

section("the payoff")
test("the bolted camera changes far fewer pixels than a moving one") {
    try renderer.setBus(bus.shapes)
    let frames = 12
    let pixels = tw * th

    func changedFraction(masked: Bool) throws -> (Double, Int, Double) {
        var samples: [RGB] = []
        func draw(_ f: Int, full: Bool) throws {
            try renderer.setWorld(buildWorld(props, travel: loop.travelWrapped(frame: f * 10)))
            let camera = masked ? cam : bus.orbitCamera(frame: f * 10, loop: loop)
            let s = testSettings(frame: f * 10)
            if full {
                _ = try renderer.renderFull(camera: camera, settings: s, into: scratch,
                                            width: tw, viewHeight: th)
            } else {
                _ = try renderer.renderMasked(camera: camera, settings: s, into: scratch,
                                              width: tw, viewHeight: th, indices: indices)
            }
        }
        try draw(0, full: true)
        samples += samplePixels(scratch, pixels: pixels, step: 5)
        let palette = medianCutPalette(samples, count: 96)
        let quantizer = try Quantizer(device: device, palette: palette, pixels: pixels)
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("step016_\(masked ? "bolted" : "moving").gif")
        let gif = GIFWriter(url: url, width: tw, height: th, palette: palette,
                            delayCentiseconds: loop.delayCentiseconds)
        var previous: [UInt8]?
        var changed = 0, compared = 0
        var boxArea = 0.0, boxFrames = 0
        for f in 0..<frames {
            try draw(f, full: !masked || f == 0)
            let idx = try quantizer.indices(of: scratch)
            if let prev = previous {
                var x0 = tw, y0 = th, x1 = -1, y1 = -1
                for y in 0..<th {
                    let row = y * tw
                    for x in 0..<tw where idx[row + x] != prev[row + x] {
                        changed += 1
                        x0 = min(x0, x); x1 = max(x1, x); y0 = min(y0, y); y1 = max(y1, y)
                    }
                }
                compared += idx.count
                if x1 >= 0 { boxArea += Double((x1 - x0 + 1) * (y1 - y0 + 1)); boxFrames += 1 }
            }
            previous = idx
            gif.add(idx)
        }
        try gif.finish()
        let size = (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? Int) ?? 0
        let box = boxFrames == 0 ? 0 : boxArea / Double(boxFrames) / Double(pixels)
        return (Double(changed) / Double(max(compared, 1)), size ?? 0, box)
    }

    let (boltedChanged, boltedBytes, boltedBox) = try changedFraction(masked: true)
    let (movingChanged, movingBytes, movingBox) = try changedFraction(masked: false)
    print(String(format: "        bolted  %.1f%% changed, change box %.1f%% of frame, %d bytes",
                 boltedChanged * 100, boltedBox * 100, boltedBytes))
    print(String(format: "        moving  %.1f%% changed, change box %.1f%% of frame, %d bytes  (%.2f× bigger)",
                 movingChanged * 100, movingBox * 100, movingBytes,
                 Double(movingBytes) / Double(max(boltedBytes, 1))))
    expect(boltedChanged < movingChanged * 0.6,
           "\(boltedChanged * 100)% against \(movingChanged * 100)%")
    expect(boltedBytes < movingBytes,
           "the GIF should be smaller: \(boltedBytes) against \(movingBytes)")
    expect(tracedFraction < 0.5, "the mask covers \(tracedFraction * 100)% of the frame")
}

finish()
