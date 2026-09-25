// Tests for step 2. Unit tests check the pure helpers; GPU tests compare the
// kernel's output against the CPU version of the same formula, byte for byte.

import CoreGraphics
import Foundation
import ImageIO
import Metal

func tempURL(_ name: String) -> URL {
    FileManager.default.temporaryDirectory.appendingPathComponent("paint_gpu_tests-\(getpid())-\(name)")
}

func expectMatchesCPU(_ image: GPUImage, file: StaticString = #fileID, line: UInt = #line) {
    var mismatches = 0
    var first = ""
    for y in 0..<image.height {
        let expected = gradientPixel(y: y, height: image.height)
        for x in 0..<image.width where image.pixel(x: x, y: y) != expected {
            if mismatches == 0 { first = "(\(x), \(y)): expected \(expected), got \(image.pixel(x: x, y: y))" }
            mismatches += 1
        }
    }
    expect(mismatches == 0, "\(mismatches) pixels differ, first at \(first)", file: file, line: line)
}

section("gradientPixel")
test("the top row is sky blue and the bottom row is deep blue") {
    expectEqual(gradientPixel(y: 0, height: 1080), skyBlue)
    expectEqual(gradientPixel(y: 1079, height: 1080), deepBlue)
}
test("the middle row is halfway between") {
    expectEqual(gradientPixel(y: 1, height: 3), RGBA(r: 74, g: 120, b: 162, a: 255))
}
test("a one-row image is sky blue instead of dividing by zero") {
    expectEqual(gradientPixel(y: 0, height: 1), skyBlue)
}
test("colors only get darker going down") {
    var previous = gradientPixel(y: 0, height: 100)
    for y in 1..<100 {
        let p = gradientPixel(y: y, height: 100)
        expect(p.r <= previous.r && p.g <= previous.g && p.b <= previous.b, "row \(y) got lighter")
        previous = p
    }
}

section("grouped")
test("groups digits in threes") {
    expectEqual(grouped(0), "0")
    expectEqual(grouped(999), "999")
    expectEqual(grouped(1000), "1,000")
    expectEqual(grouped(2_073_600), "2,073,600")
}

section("parseOptions")
test("defaults to 1920 × 1080 in renders/") {
    expectEqual(try parseOptions([]), Options())
    expectEqual(Options().out, "renders/gradient.png")
}
test("reads width, height and output path") {
    expectEqual(try parseOptions(["--width", "64", "--height", "32", "--out", "x.png"]),
                Options(width: 64, height: 32, out: "x.png"))
}
test("rejects bad input") {
    expectThrows { _ = try parseOptions(["--width"]) }
    expectThrows { _ = try parseOptions(["--width", "0"]) }
    expectThrows { _ = try parseOptions(["--width", "wide"]) }
    expectThrows { _ = try parseOptions(["--height", "99999"]) }
    expectThrows { _ = try parseOptions(["--depth", "3"]) }
}

section("GPU (this Mac)")
test("the GPU's pixels match the CPU formula exactly") {
    let image = try paintGradient(width: 64, height: 48, on: try findDevice())
    expectMatchesCPU(image)
}
test("sizes that aren't a multiple of the threadgroup still fill every pixel") {
    let device = try findDevice()
    expectMatchesCPU(try paintGradient(width: 33, height: 17, on: device))
    expectMatchesCPU(try paintGradient(width: 1, height: 1, on: device))
    expectMatchesCPU(try paintGradient(width: 1, height: 300, on: device))
}
// dispatchThreads already stops at the image edge, so this guards the
// dispatch setup rather than the kernel's own bounds check.
test("the GPU doesn't write past the end of the image") {
    let device = try findDevice()
    let (width, height, spare) = (33, 17, 4096)
    let buffer = device.makeBuffer(length: width * height * 4 + spare, options: .storageModeShared)!
    let bytes = buffer.contents().assumingMemoryBound(to: UInt8.self)
    for i in 0..<buffer.length { bytes[i] = 0xAB }
    try paintGradient(into: buffer, width: width, height: height, on: device)
    let untouched = (width * height * 4..<buffer.length).allSatisfy { bytes[$0] == 0xAB }
    expect(untouched, "bytes after the image were overwritten")
}
test("the pixel buffer is shared between CPU and GPU") {
    let image = try paintGradient(width: 8, height: 8, on: try findDevice())
    expectEqual(image.buffer.storageMode, .shared)
}

section("PNG")
test("a saved PNG reads back with the same size and colors") {
    let image = try paintGradient(width: 16, height: 8, on: try findDevice())
    let url = tempURL("roundtrip.png")
    defer { try? FileManager.default.removeItem(at: url) }
    try savePNG(image, to: url)
    guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
          let decoded = CGImageSourceCreateImageAtIndex(source, 0, nil),
          let data = decoded.dataProvider?.data,
          let bytes = CFDataGetBytePtr(data)
    else { return expect(false, "could not read the PNG back") }
    expectEqual(decoded.width, 16)
    expectEqual(decoded.height, 8)
    let stride = decoded.bytesPerRow
    let step = decoded.bitsPerPixel / 8
    func rgb(_ x: Int, _ y: Int) -> [UInt8] { (0..<3).map { bytes[y * stride + x * step + $0] } }
    expectEqual(rgb(0, 0), [skyBlue.r, skyBlue.g, skyBlue.b])
    expectEqual(rgb(15, 7), [deepBlue.r, deepBlue.g, deepBlue.b])
}
test("the summary reports the size, thread count and zero copies") {
    let image = try paintGradient(width: 1920, height: 1080, on: try findDevice())
    let text = summary(gpuName: "Test GPU", unifiedMemory: true, image: image,
                       storageMode: storageModeName(.shared), savedTo: "renders/gradient.png")
    for line in [
        "Painted 1920 × 1080 pixels, one GPU thread per pixel (2,073,600 threads)",
        "Pixel buffer: 8,294,400 bytes, storage mode: shared (CPU and GPU use the same memory)",
        "Bytes copied from GPU to CPU: 0",
        "Top-left pixel: (140, 200, 235, 255), bottom-left pixel: (8, 40, 90, 255)",
    ] {
        expect(text.contains(line), "missing line: \(line)")
    }
}

finish()
