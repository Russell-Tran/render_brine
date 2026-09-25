// Tests for step 5. Unit tests check the Fresnel, reflection and sampling
// math and the CPU reference; GPU tests compare the kernel to the CPU
// reference (within 2 levels out of 255, as in step 3) and check the timing.

import Foundation
import Metal
import simd

let tolerance = 2

func near(_ a: Float, _ b: Float, within e: Float = 1e-4) -> Bool { abs(a - b) <= e }

func worstDifference(_ image: GPUImage, waves: [Wave], samplesPerSide k: Int) -> (diff: Int, at: String) {
    var worst = (diff: 0, at: "")
    for y in 0..<image.height {
        for x in 0..<image.width {
            let c = shadeReflectionPixel(px: x, py: y, width: image.width, height: image.height,
                                         waves: waves, samplesPerSide: k)
            let cpu = RGBA(c)
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

section("Fresnel")
test("water reflects 2% looking straight down and everything at a grazing angle") {
    expect(near(fresnel(cosine: 1), 0.02))
    expect(near(fresnel(cosine: 0), 1))
}
test("reflection only grows as the view gets more glancing") {
    var previous: Float = fresnel(cosine: 1)
    for i in stride(from: 99, through: 0, by: -1) {
        let f: Float = fresnel(cosine: Float(i) / 100)
        expect(f >= previous, "dropped at cosine \(Float(i) / 100)")
        previous = f
    }
}

section("reflection")
test("a ray hitting flat water at 45° leaves at 45° upward") {
    let down = simd_normalize(SIMD3<Float>(0, -1, 1))
    let r = reflectOffWater(down, normal: SIMD3<Float>(0, 1, 0))
    expect(simd_distance(r, simd_normalize(SIMD3<Float>(0, 1, 1))) < 1e-5, "got \(r)")
}
test("a reflection that would point into the water is flipped up") {
    let ray = simd_normalize(SIMD3<Float>(0, -0.2, 1))
    let steep = simd_normalize(SIMD3<Float>(0, 0.3, -1))
    let r = reflectOffWater(ray, normal: steep)
    expect(r.y >= 0, "reflection points down: \(r)")
    expect(near(simd_length(r), 1), "not unit length: \(r)")
}
test("the sky includes a bright sun disk") {
    let sun = skyRadiance(sunDirection)
    expect(sun.x > 1, "sun should be brighter than 1 so its reflection shows: \(sun)")
    let up = skyRadiance(SIMD3<Float>(0, 1, 0))
    expect(simd_distance(up, skyBlue) < 1e-5, "straight up should be sky blue, got \(up)")
}

section("samples per pixel")
test("one sample sits in the middle of the pixel") {
    expectEqual(sampleOffsets(perSide: 1), [SIMD2<Float>(0.5, 0.5)])
}
test("k × k samples cover the pixel evenly") {
    let o = sampleOffsets(perSide: 4)
    expectEqual(o.count, 16)
    expect(o.allSatisfy { $0.x > 0 && $0.x < 1 && $0.y > 0 && $0.y < 1 }, "a sample is outside the pixel")
    let mean = o.reduce(SIMD2<Float>(0, 0), +) / 16
    expect(simd_distance(mean, SIMD2(0.5, 0.5)) < 1e-6, "samples aren't centered: \(mean)")
}
test("a pixel's color is the average of its samples") {
    let (w, h) = (96, 54)
    let pixel = shadeReflectionPixel(px: 40, py: 40, width: w, height: h, waves: defaultWaves, samplesPerSide: 2)
    var sum = SIMD3<Float>(0, 0, 0)
    for o in sampleOffsets(perSide: 2) {
        sum += shadeReflectionSample(sx: 40 + o.x, sy: 40 + o.y, width: w, height: h, waves: defaultWaves)
    }
    expect(simd_distance(pixel, sum / 4) < 1e-6)
}

section("CPU reference renderer")
test("flat water looking steeply down is mostly the water's own color") {
    // Bottom center of the image looks down at about 45°, where Fresnel is small.
    let c = shadeReflectionSample(sx: 48, sy: 53.5, width: 96, height: 54, waves: [])
    let facing: Float = simd_clamp(simd_dot(SIMD3<Float>(0, 1, 0), sunDirection), 0, 1)
    let body = simd_mix(deepBlue, turquoise, SIMD3(repeating: facing * facing))
    expect(simd_distance(c, body) < 0.12, "got \(c), body color \(body)")
}
test("flat water gets brighter toward the horizon (more reflection, plus haze)") {
    let (w, h) = (96, 540)
    let near0 = shadeReflectionSample(sx: 10.5, sy: 539.5, width: w, height: h, waves: [])
    let far0 = shadeReflectionSample(sx: 10.5, sy: 205.5, width: w, height: h, waves: [])
    expect(simd_reduce_add(far0) > simd_reduce_add(near0), "far \(far0) should be brighter than near \(near0)")
}
test("a flat sea looks the same on the left and right") {
    let (w, h) = (64, 36)
    for y in 0..<h {
        for x in 0..<w / 2 {
            let left = RGBA(shadeReflectionPixel(px: x, py: y, width: w, height: h, waves: [], samplesPerSide: 2))
            let right = RGBA(shadeReflectionPixel(px: w - 1 - x, py: y, width: w, height: h, waves: [], samplesPerSide: 2))
            if left != right { return expect(false, "row \(y): \(left) vs \(right)") }
        }
    }
}
test("the sun's reflection makes glints on the waves") {
    let (w, h) = (192, 108)
    var brightest: Float = 0
    for y in 45..<h {
        for x in 0..<w {
            let c = shadeReflectionSample(sx: Float(x) + 0.5, sy: Float(y) + 0.5, width: w, height: h, waves: defaultWaves)
            brightest = max(brightest, simd_reduce_min(c))
        }
    }
    expect(brightest > 0.8, "no bright glint found below the horizon (brightest \(brightest))")
}
test("colors stay between 0 and 1") {
    for y in stride(from: 0, to: 108, by: 7) {
        for x in stride(from: 0, to: 192, by: 7) {
            let c = shadeReflectionPixel(px: x, py: y, width: 192, height: 108, waves: defaultWaves, samplesPerSide: 2)
            expect(simd_reduce_min(c) >= 0 && simd_reduce_max(c) <= 1, "out of range at (\(x), \(y)): \(c)")
        }
    }
}

section("GPU (this Mac)")
test("the GPU matches the CPU reference at 1 and 4 samples per pixel") {
    let renderer = try ReflectionRenderer(device: try findDevice())
    for k in [1, 2] {
        let worst = worstDifference(try renderer.renderImage(width: 96, height: 54, samplesPerSide: k),
                                    waves: defaultWaves, samplesPerSide: k)
        print("        \(k * k) spp worst difference: \(worst.diff) \(worst.at)")
        expect(worst.diff <= tolerance, "\(k * k) spp: \(worst.diff) at \(worst.at)")
    }
}
test("odd sizes, 16 samples and a flat sea match too") {
    let device = try findDevice()
    let wavy = try ReflectionRenderer(device: device)
    let flat = try ReflectionRenderer(device: device, waves: [])
    let a = worstDifference(try wavy.renderImage(width: 33, height: 17, samplesPerSide: 4), waves: defaultWaves, samplesPerSide: 4)
    let b = worstDifference(try flat.renderImage(width: 50, height: 30, samplesPerSide: 1), waves: [], samplesPerSide: 1)
    expect(a.diff <= tolerance, "33 × 17 at 16 spp: \(a.diff) at \(a.at)")
    expect(b.diff <= tolerance, "flat 50 × 30: \(b.diff) at \(b.at)")
}
test("more samples change the image (the edges get smoothed)") {
    let renderer = try ReflectionRenderer(device: try findDevice())
    let one = try renderer.renderImage(width: 96, height: 54, samplesPerSide: 1)
    let many = try renderer.renderImage(width: 96, height: 54, samplesPerSide: 4)
    let differ = memcmp(one.buffer.contents(), many.buffer.contents(), one.byteCount) != 0
    expect(differ, "16 spp image is identical to the 1 spp image")
}
test("16 samples per pixel take longer than 1") {
    let renderer = try ReflectionRenderer(device: try findDevice())
    try warmUpGPU(renderer, seconds: 0.1)
    let one: Double = try medianGPUTime(renderer, width: 960, height: 540, samplesPerSide: 1, runs: 5)
    let sixteen: Double = try medianGPUTime(renderer, width: 960, height: 540, samplesPerSide: 4, runs: 5)
    expect(sixteen > one * 4, "16 spp took \(sixteen * 1000) ms, 1 spp took \(one * 1000) ms")
}

section("report")
test("the timing table compares work and time") {
    let text = timingTable([SampleTiming(samplesPerSide: 1, gpuSeconds: 0.0005),
                            SampleTiming(samplesPerSide: 4, gpuSeconds: 0.0078)], width: 1920, height: 1080)
    expect(text.contains("1 sample per pixel:     0.500 ms  (1× the work, 1.0× the time)"), text)
    expect(text.contains("16 samples per pixel:   7.800 ms  (16× the work, 15.6× the time)"), text)
    let fours = timingTable([SampleTiming(samplesPerSide: 1, gpuSeconds: 0.0005),
                             SampleTiming(samplesPerSide: 2, gpuSeconds: 0.0016)], width: 1920, height: 1080)
    expect(fours.contains("\n  4 samples per pixel:    1.600 ms"), fours)
}

finish()
