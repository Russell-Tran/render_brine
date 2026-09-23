// Tests for step 1. Unit tests use made-up inputs; hardware tests run
// against this Mac's real chip and GPU.

import Foundation
import Metal

// Made-up values for testing formatting, not a real chip.
func sampleReport() -> GPUReport {
    GPUReport(
        chip: "Test Chip", performanceCores: 12, efficiencyCores: 4,
        totalMemory: 48 << 30, gpuCores: 40,
        gpuName: "Test GPU", hasUnifiedMemory: true,
        recommendedMaxWorkingSetSize: 36 << 30, maxBufferLength: 24 << 30,
        maxThreadsPerThreadgroup: (1024, 1024, 1024), maxThreadgroupMemory: 32 << 10,
        highestAppleFamily: 9, supportsRaytracing: true,
        simdWidth: 32, kernelMaxThreadsPerThreadgroup: 1024
    )
}

section("formatBytes")
test("bytes under 1 KB are shown exactly") {
    expectEqual(formatBytes(0), "0 B")
    expectEqual(formatBytes(1023), "1023 B")
}
test("larger sizes use binary units with one decimal") {
    expectEqual(formatBytes(1024), "1.0 KB")
    expectEqual(formatBytes(1536), "1.5 KB")
    expectEqual(formatBytes(32 << 10), "32.0 KB")
    expectEqual(formatBytes(16 << 30), "16.0 GB")
    expectEqual(formatBytes(48 << 30), "48.0 GB")
}

section("percent")
test("rounds to the nearest whole percent") {
    expectEqual(percent(2, of: 3), 67)
    expectEqual(percent(36 << 30, of: 48 << 30), 75)
}
test("a zero total gives 0 instead of crashing") {
    expectEqual(percent(5, of: 0), 0)
}

section("parseGPUCoreCount")
test("reads the count from ioreg output") {
    let text = """
        +-o AGXAcceleratorG16G  <class AGXAcceleratorG16G>
            {
              "model" = "Apple M4"
              "gpu-core-count" = 10
            }
        """
    expectEqual(parseGPUCoreCount(fromIORegistry: text), 10)
    expectEqual(parseGPUCoreCount(fromIORegistry: #""gpu-core-count" = 40"#), 40)
}
test("returns nil when the key is missing or malformed") {
    expectEqual(parseGPUCoreCount(fromIORegistry: ""), nil)
    expectEqual(parseGPUCoreCount(fromIORegistry: #""model" = "Apple M4""#), nil)
    expectEqual(parseGPUCoreCount(fromIORegistry: #""gpu-core-count" = many"#), nil)
}

section("highestAppleFamily")
test("picks the newest family the GPU supports") {
    let upToApple9 = highestAppleFamily { $0.rawValue <= MTLGPUFamily.apple9.rawValue }
    expectEqual(upToApple9, 9)
    let onlyApple1 = highestAppleFamily { $0 == .apple1 }
    expectEqual(onlyApple1, 1)
}
test("returns nil when no Apple family is supported") {
    expectEqual(highestAppleFamily { _ in false }, nil)
}

section("render")
test("pairs each Apple term with its CUDA equivalent") {
    let text = render(sampleReport())
    for line in [
        "SIMD group width (≈ CUDA warp): 32",
        "Threadgroup size limit per dimension (≈ CUDA block): 1024 × 1024 × 1024",
        "Threadgroup memory (≈ CUDA shared memory per block): 32.0 KB",
    ] {
        expect(text.contains(line), "missing line: \(line)")
    }
}
test("shows memory, working-set share and core counts") {
    let text = render(sampleReport())
    for line in [
        "Memory (shared by CPU and GPU): 48.0 GB",
        "GPU working-set limit: 36.0 GB (75% of RAM)",
        "CPU cores: 12 performance + 4 efficiency",
        "GPU cores: 40",
        "Apple GPU family: Apple 9",
        "Unified memory: yes",
    ] {
        expect(text.contains(line), "missing line: \(line)")
    }
}
test("unknown values say unknown") {
    var report = sampleReport()
    report.gpuCores = nil
    report.highestAppleFamily = nil
    let text = render(report)
    expect(text.contains("GPU cores: unknown"))
    expect(text.contains("Apple GPU family: unknown"))
}

section("hardware (this Mac)")
test("the system default Metal GPU is available") {
    // Needs CoreGraphics linked; see the import in GPUReport.swift.
    expect(MTLCreateSystemDefaultDevice() != nil, "MTLCreateSystemDefaultDevice() returned nil")
}
test("findDevice returns a GPU") {
    expect((try? findDevice()) != nil)
}
test("a bad kernel reports a compile error") {
    let device = try findDevice()
    expectThrows { _ = try compileKernel("not metal", named: "probe", on: device) }
    expectThrows { _ = try compileKernel(probeKernelSource, named: "missing", on: device) }
}
test("the GPU core count can be read from the IO registry") {
    let cores = readGPURegistry().flatMap(parseGPUCoreCount(fromIORegistry:))
    expect((cores ?? 0) > 0, "got \(String(describing: cores))")
}
test("the gathered report is consistent") {
    let r = try gatherReport()
    expect(r.hasUnifiedMemory, "Apple silicon should have unified memory")
    expect(r.totalMemory > 0)
    expect(r.recommendedMaxWorkingSetSize <= r.totalMemory, "working set exceeds RAM")
    expect(r.performanceCores > 0 && r.efficiencyCores >= 0)
    expect(r.simdWidth > 0 && r.simdWidth & (r.simdWidth - 1) == 0,
           "SIMD width \(r.simdWidth) should be a power of two")
    expect(r.kernelMaxThreadsPerThreadgroup >= r.simdWidth)
    expect(r.kernelMaxThreadsPerThreadgroup <= r.maxThreadsPerThreadgroup.width)
    expect(r.highestAppleFamily != nil)
}

finish()
