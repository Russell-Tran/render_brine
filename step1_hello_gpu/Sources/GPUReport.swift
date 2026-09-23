// Gathers facts about this Mac's chip and GPU, and formats them as a report
// that puts Apple's Metal terms next to the CUDA terms they correspond to.
//
// The pure helpers (formatting, parsing, rendering) are kept separate from
// the functions that query the hardware, so the tests can check them with
// made-up inputs.

// Build with -framework CoreGraphics (the Makefile does). Apple's docs say
// command-line tools must link it for MTLCreateSystemDefaultDevice() to
// return a GPU; without it, that call returned nil on an M3 Max.
import Foundation
import Metal

struct GPUReport {
    // From the Mac (sysctl and the IO registry).
    var chip: String
    var performanceCores: Int
    var efficiencyCores: Int
    var totalMemory: UInt64
    var gpuCores: Int?

    // From Metal's device object.
    var gpuName: String
    var hasUnifiedMemory: Bool
    var recommendedMaxWorkingSetSize: UInt64
    var maxBufferLength: UInt64
    var maxThreadsPerThreadgroup: (width: Int, height: Int, depth: Int)
    var maxThreadgroupMemory: UInt64
    var highestAppleFamily: Int?
    var supportsRaytracing: Bool

    // From a compiled kernel (these belong to the kernel, not the chip).
    var simdWidth: Int
    var kernelMaxThreadsPerThreadgroup: Int
}

enum ReportError: Error, CustomStringConvertible {
    case noMetalDevice
    case kernelCompile(String)

    var description: String {
        switch self {
        case .noMetalDevice:
            return "no Metal GPU found (MTLCreateSystemDefaultDevice and MTLCopyAllDevices returned nothing)"
        case .kernelCompile(let detail): return "could not compile the probe kernel: \(detail)"
        }
    }
}

// MARK: - Pure helpers

/// Formats a byte count with binary units, the way macOS reports RAM ("16.0 GB").
func formatBytes(_ bytes: UInt64) -> String {
    if bytes < 1024 { return "\(bytes) B" }
    let units = ["KB", "MB", "GB", "TB", "PB"]
    var value = Double(bytes) / 1024
    var unit = 0
    while value >= 1024 && unit < units.count - 1 {
        value /= 1024
        unit += 1
    }
    return String(format: "%.1f %@", value, units[unit])
}

/// `part` as a whole-number percentage of `whole`, rounded to nearest.
func percent(_ part: UInt64, of whole: UInt64) -> Int {
    guard whole > 0 else { return 0 }
    return Int((Double(part) / Double(whole) * 100).rounded())
}

/// Pulls the GPU core count out of `ioreg -rc AGXAccelerator` output.
/// Metal itself doesn't expose this number.
func parseGPUCoreCount(fromIORegistry text: String) -> Int? {
    guard let range = text.range(of: #""gpu-core-count" = (\d+)"#, options: .regularExpression) else {
        return nil
    }
    return Int(text[range].split(separator: " ").last ?? "")
}

/// The newest Apple GPU family (apple1, apple2, ...) that `supports` accepts.
/// Taking the check as a closure lets the tests fake a GPU.
func highestAppleFamily(supports: (MTLGPUFamily) -> Bool) -> Int? {
    let appleBase = MTLGPUFamily.apple1.rawValue - 1
    for n in stride(from: 20, through: 1, by: -1) {
        if let family = MTLGPUFamily(rawValue: appleBase + n), supports(family) {
            return n
        }
    }
    return nil
}

/// Renders the report, one fact per line, with the CUDA equivalent in parentheses.
func render(_ r: GPUReport) -> String {
    let t = r.maxThreadsPerThreadgroup
    let lines = [
        "== The chip ==",
        "Chip: \(r.chip)",
        "CPU cores: \(r.performanceCores) performance + \(r.efficiencyCores) efficiency",
        "GPU cores: \(r.gpuCores.map(String.init) ?? "unknown")",
        "Memory (shared by CPU and GPU): \(formatBytes(r.totalMemory))",
        "",
        "== What Metal reports (Metal ≈ CUDA) ==",
        "GPU: \(r.gpuName)",
        "Unified memory: \(r.hasUnifiedMemory ? "yes" : "no")",
        "GPU working-set limit: \(formatBytes(r.recommendedMaxWorkingSetSize)) "
            + "(\(percent(r.recommendedMaxWorkingSetSize, of: r.totalMemory))% of RAM)",
        "Largest single buffer: \(formatBytes(r.maxBufferLength))",
        "Threadgroup size limit per dimension (≈ CUDA block): \(t.width) × \(t.height) × \(t.depth)",
        "Threadgroup memory (≈ CUDA shared memory per block): \(formatBytes(r.maxThreadgroupMemory))",
        "Apple GPU family: \(r.highestAppleFamily.map { "Apple \($0)" } ?? "unknown")",
        "Ray tracing API available: \(r.supportsRaytracing ? "yes" : "no")",
        "",
        "== A compiled kernel ==",
        "SIMD group width (≈ CUDA warp): \(r.simdWidth)",
        "Max threads per threadgroup for this kernel: \(r.kernelMaxThreadsPerThreadgroup)",
    ]
    return lines.joined(separator: "\n")
}

// MARK: - Querying the hardware

let probeKernelSource = """
    #include <metal_stdlib>
    using namespace metal;
    kernel void probe(device float *out [[buffer(0)]], uint id [[thread_position_in_grid]]) {
        out[id] = float(id);
    }
    """

/// Compiles Metal source at runtime. This needs no Xcode or offline Metal compiler.
func compileKernel(_ source: String, named name: String, on device: MTLDevice) throws -> MTLComputePipelineState {
    do {
        let library = try device.makeLibrary(source: source, options: nil)
        guard let function = library.makeFunction(name: name) else {
            throw ReportError.kernelCompile("no kernel named \(name)")
        }
        return try device.makeComputePipelineState(function: function)
    } catch let error as ReportError {
        throw error
    } catch {
        throw ReportError.kernelCompile(error.localizedDescription)
    }
}

func sysctlString(_ name: String) -> String? {
    var size = 0
    guard sysctlbyname(name, nil, &size, nil, 0) == 0, size > 0 else { return nil }
    var buffer = [CChar](repeating: 0, count: size)
    guard sysctlbyname(name, &buffer, &size, nil, 0) == 0 else { return nil }
    return String(cString: buffer)
}

func sysctlInt(_ name: String) -> Int64? {
    var value: Int64 = 0
    var size = MemoryLayout<Int64>.size
    guard sysctlbyname(name, &value, &size, nil, 0) == 0 else { return nil }
    // Some sysctls are 32-bit; only the low `size` bytes were written.
    return size == 4 ? Int64(Int32(truncatingIfNeeded: value)) : value
}

func readGPURegistry() -> String? {
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/usr/sbin/ioreg")
    process.arguments = ["-rc", "AGXAccelerator", "-d", "1"]
    let pipe = Pipe()
    process.standardOutput = pipe
    process.standardError = FileHandle.nullDevice
    do { try process.run() } catch { return nil }
    let data = pipe.fileHandleForReading.readDataToEndOfFile()
    process.waitUntilExit()
    return String(data: data, encoding: .utf8)
}

/// The GPU to report on: the system default, or else the first GPU Metal lists.
func findDevice() throws -> MTLDevice {
    if let device = MTLCreateSystemDefaultDevice() ?? MTLCopyAllDevices().first {
        return device
    }
    throw ReportError.noMetalDevice
}

func gatherReport() throws -> GPUReport {
    let device = try findDevice()
    let pipeline = try compileKernel(probeKernelSource, named: "probe", on: device)
    let t = device.maxThreadsPerThreadgroup
    return GPUReport(
        chip: sysctlString("machdep.cpu.brand_string") ?? "unknown",
        performanceCores: Int(sysctlInt("hw.perflevel0.physicalcpu") ?? 0),
        efficiencyCores: Int(sysctlInt("hw.perflevel1.physicalcpu") ?? 0),
        totalMemory: UInt64(sysctlInt("hw.memsize") ?? 0),
        gpuCores: readGPURegistry().flatMap(parseGPUCoreCount(fromIORegistry:)),
        gpuName: device.name,
        hasUnifiedMemory: device.hasUnifiedMemory,
        recommendedMaxWorkingSetSize: device.recommendedMaxWorkingSetSize,
        maxBufferLength: UInt64(device.maxBufferLength),
        maxThreadsPerThreadgroup: (t.width, t.height, t.depth),
        maxThreadgroupMemory: UInt64(device.maxThreadgroupMemoryLength),
        highestAppleFamily: highestAppleFamily(supports: device.supportsFamily),
        supportsRaytracing: device.supportsRaytracing,
        simdWidth: pipeline.threadExecutionWidth,
        kernelMaxThreadsPerThreadgroup: pipeline.maxTotalThreadsPerThreadgroup
    )
}
