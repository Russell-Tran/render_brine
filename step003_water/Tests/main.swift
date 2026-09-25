// Tests for step 3. Unit tests check the wave math and the CPU reference
// renderer; GPU tests compare the kernel's output to the CPU reference.
//
// Unlike step 2, this uses floating point (sin, cos, pow), and the GPU's
// versions round slightly differently from the CPU's. So the GPU is allowed
// to differ by at most `tolerance` levels (out of 255) per color channel.

import CoreGraphics
import Foundation
import ImageIO
import Metal
import simd

let tolerance = 2

func near(_ a: Float, _ b: Float, within e: Float = 1e-4) -> Bool { abs(a - b) <= e }

/// Largest per-channel difference between the GPU image and the CPU reference,
/// and where it is.
func worstDifference(_ image: GPUImage, waves: [Wave]) -> (diff: Int, at: String) {
    var worst = (diff: 0, at: "")
    for y in 0..<image.height {
        for x in 0..<image.width {
            let cpu = RGBA(shadePixel(px: x, py: y, width: image.width, height: image.height, waves: waves))
            let gpu = image.pixel(x: x, y: y)
            let dr: Int = abs(Int(cpu.r) - Int(gpu.r))
            let dg: Int = abs(Int(cpu.g) - Int(gpu.g))
            let db: Int = abs(Int(cpu.b) - Int(gpu.b))
            let d = max(dr, dg, db)
            if d > worst.diff { worst = (d, "(\(x), \(y)): CPU \(cpu), GPU \(gpu)") }
        }
    }
    return worst
}

section("waves")
test("a single wave peaks at its amplitude and is zero at its start") {
    let w = [Wave(angleDegrees: 0, wavelength: 4, amplitude: 0.5)]
    expect(near(waveHeight(x: 0, z: 0, waves: w), 0))
    expect(near(waveHeight(x: 1, z: 0, waves: w), 0.5), "a quarter wavelength in should be the crest")
}
test("waves add up") {
    let a = Wave(angleDegrees: 0, wavelength: 4, amplitude: 0.5)
    let b = Wave(angleDegrees: 90, wavelength: 8, amplitude: 0.25)
    expect(near(waveHeight(x: 1, z: 2, waves: [a, b]),
                waveHeight(x: 1, z: 2, waves: [a]) + waveHeight(x: 1, z: 2, waves: [b])))
}
test("a flat sea's normal points straight up") {
    expectEqual(waveNormal(x: 3, z: 7, waves: []), SIMD3<Float>(0, 1, 0))
}
test("the normal matches the slope measured numerically") {
    let (x, z, e): (Float, Float, Float) = (1.3, 2.7, 1e-3)
    let n = waveNormal(x: x, z: z, waves: defaultWaves)
    let east: Float = waveHeight(x: x + e, z: z, waves: defaultWaves)
    let west: Float = waveHeight(x: x - e, z: z, waves: defaultWaves)
    let north: Float = waveHeight(x: x, z: z + e, waves: defaultWaves)
    let south: Float = waveHeight(x: x, z: z - e, waves: defaultWaves)
    let dx: Float = (east - west) / (2 * e)
    let dz: Float = (north - south) / (2 * e)
    let numeric = simd_normalize(SIMD3<Float>(-dx, 1, -dz))
    expect(simd_distance(n, numeric) < 1e-2, "analytic \(n) vs numeric \(numeric)")
}
test("a wave built from an angle has a unit direction") {
    let w = Wave(angleDegrees: 37, wavelength: 3, amplitude: 1)
    let lengthSquared: Float = w.directionX * w.directionX + w.directionZ * w.directionZ
    expect(near(lengthSquared, 1))
    expect(near(w.wavenumber, 2 * .pi / 3))
}

section("CPU reference renderer")
test("the top of the image is sky and the bottom is water") {
    let top = shadePixel(px: 0, py: 0, width: 192, height: 108, waves: defaultWaves)
    let bottom = shadePixel(px: 0, py: 107, width: 192, height: 108, waves: defaultWaves)
    expect(simd_distance(top, skyBlue) < 0.05, "top-left should be close to sky blue, got \(top)")
    expect(bottom.z > bottom.x, "water should be more blue than red, got \(bottom)")
}
test("the horizon sits about 37% of the way down") {
    // The ray through row y points up by (1 - 2v) * tanHalfFOV - tilt, with v
    // the row's position from 0 (top) to 1 (bottom). Water starts where that
    // turns negative. (Split into steps so older Swift compilers can type-check it.)
    let h = 1000
    var firstWater = -1
    for y in 0..<h {
        let v: Float = (Float(y) + 0.5) / Float(h)
        let up: Float = (1 - 2 * v) * tanHalfFOV - tilt
        if up < 0 { firstWater = y; break }
    }
    expect(abs(firstWater - 370) <= 1, "first water row \(firstWater)")
}
test("a flat sea looks the same on the left and right") {
    let (w, h) = (64, 36)
    for y in 0..<h {
        for x in 0..<w / 2 {
            let left = RGBA(shadePixel(px: x, py: y, width: w, height: h, waves: []))
            let right = RGBA(shadePixel(px: w - 1 - x, py: y, width: w, height: h, waves: []))
            if left != right { return expect(false, "row \(y): \(left) vs \(right)") }
        }
    }
}
test("the sun appears in the sky, straight ahead") {
    let (w, h) = (400, 400)
    let sunPixelY = (0..<h).first { y in
        shadePixel(px: w / 2, py: y, width: w, height: h, waves: defaultWaves) == sunColor
    }
    expect(sunPixelY != nil, "no sun found in the middle column")
}
test("waves make the water vary; a flat sea doesn't") {
    let (w, h, y) = (64, 36, 30)
    let flat = Set((0..<w).map { RGBA(shadePixel(px: $0, py: y, width: w, height: h, waves: [])).description })
    let wavy = Set((0..<w).map { RGBA(shadePixel(px: $0, py: y, width: w, height: h, waves: defaultWaves)).description })
    expect(wavy.count > flat.count, "wavy row has \(wavy.count) colors, flat row \(flat.count)")
}
test("colors stay between 0 and 1") {
    for y in stride(from: 0, to: 108, by: 7) {
        for x in stride(from: 0, to: 192, by: 7) {
            let c = shadePixel(px: x, py: y, width: 192, height: 108, waves: defaultWaves)
            expect(simd_reduce_min(c) >= 0 && simd_reduce_max(c) <= 1, "out of range at (\(x), \(y)): \(c)")
        }
    }
}

section("parseOptions")
test("defaults to 1920 × 1080 in renders/") {
    expectEqual(try parseOptions([]), Options())
    expectEqual(Options().out, "renders/water.png")
}
test("rejects bad input") {
    expectThrows { _ = try parseOptions(["--width"]) }
    expectThrows { _ = try parseOptions(["--width", "0"]) }
    expectThrows { _ = try parseOptions(["--depth", "3"]) }
}

section("GPU (this Mac)")
test("the GPU matches the CPU reference within \(tolerance) levels") {
    let image = try renderWater(width: 192, height: 108, on: try findDevice())
    let worst = worstDifference(image, waves: defaultWaves)
    print("        worst difference: \(worst.diff) \(worst.at)")
    expect(worst.diff <= tolerance, "worst difference \(worst.diff) at \(worst.at)")
}
test("odd sizes and a flat sea match too") {
    let device = try findDevice()
    for (w, h, waves) in [(33, 17, defaultWaves), (1, 1, defaultWaves), (50, 30, [Wave]())] {
        let worst = worstDifference(try renderWater(width: w, height: h, waves: waves, on: device), waves: waves)
        expect(worst.diff <= tolerance, "\(w) × \(h), \(waves.count) waves: \(worst.diff) at \(worst.at)")
    }
}
test("the wave buffer reaches the kernel") {
    let device = try findDevice()
    let flat = try renderWater(width: 64, height: 36, waves: [], on: device)
    let wavy = try renderWater(width: 64, height: 36, on: device)
    var differ = false
    for y in 0..<36 {
        for x in 0..<64 where flat.pixel(x: x, y: y) != wavy.pixel(x: x, y: y) { differ = true }
    }
    expect(differ, "the image didn't change when waves were added")
}
test("a saved PNG reads back with the same size") {
    let image = try renderWater(width: 32, height: 18, on: try findDevice())
    let url = FileManager.default.temporaryDirectory.appendingPathComponent("water_tests-\(getpid()).png")
    defer { try? FileManager.default.removeItem(at: url) }
    try savePNG(image, to: url)
    guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
          let decoded = CGImageSourceCreateImageAtIndex(source, 0, nil)
    else { return expect(false, "could not read the PNG back") }
    expectEqual(decoded.width, 32)
    expectEqual(decoded.height, 18)
}

finish()
