// Molecules as dots on the GPU.
//
// Each dot stands for 50 µM of one species. Every frame:
//   1. the CPU matches the dot counts to the chemistry (Chemistry.swift) by
//      converting, adding or removing dots, writing straight into the GPU's
//      buffer (unified memory, step 2);
//   2. `move`: one GPU thread per molecule takes a random Brownian step,
//      sized by that species' real diffusion coefficient;
//   3. `countSpecies`: one thread per molecule adds 1 to its species' counter
//      with an atomic add, to check the GPU's picture against the CPU's;
//   4. `draw`: one thread per pixel shades the molecules as small spheres.

import Foundation
import Metal

enum Species: UInt32, CaseIterable {
    case co2, carbonicAcid, bicarbonate, hydrogen, sodium, chloride

    /// Marks an unused slot in the particle buffer.
    static let inactive: UInt32 = 99

    var name: String {
        switch self {
        case .co2: return "CO₂"
        case .carbonicAcid: return "H₂CO₃"
        case .bicarbonate: return "HCO₃⁻"
        case .hydrogen: return "H⁺"
        case .sodium: return "Na⁺"
        case .chloride: return "Cl⁻"
        }
    }

    /// Diffusion coefficient in water at 25 °C, in 10⁻⁹ m²/s (CRC Handbook
    /// values). H₂CO₃ isn't tabulated; it's given HCO₃⁻'s value, a close cousin.
    var diffusion: Float {
        switch self {
        case .co2: return 1.91
        case .carbonicAcid: return 1.185
        case .bicarbonate: return 1.185
        case .hydrogen: return 9.311
        case .sodium: return 1.334
        case .chloride: return 2.032
        }
    }

    /// Drawing color (sRGB, 0...1) and radius in pixels.
    var color: SIMD3<Float> {
        switch self {
        case .co2: return SIMD3(0.86, 0.88, 0.90)
        case .carbonicAcid: return SIMD3(0.96, 0.65, 0.14)
        case .bicarbonate: return SIMD3(0.29, 0.56, 0.89)
        case .hydrogen: return SIMD3(0.95, 0.25, 0.30)
        case .sodium: return SIMD3(0.62, 0.50, 0.82)
        case .chloride: return SIMD3(0.35, 0.78, 0.50)
        }
    }
    var radius: Float {
        switch self {
        case .co2, .carbonicAcid, .bicarbonate: return 3.4
        case .hydrogen: return 2.6
        case .sodium, .chloride: return 1.7
        }
    }
}

/// How many µM one dot stands for.
let micromolarPerDot: Double = 50

/// Dots for a concentration in mM.
func dots(_ mM: Double) -> Double { mM * 1000 / micromolarPerDot }

// Must match `Particle` in the kernels.
struct Particle {
    var position: SIMD2<Float>   // 0...1 across the box
    var species: UInt32
    var flash: Float             // 1 right after reacting, fading to 0
}

// Must match the structs in the kernels.
struct MoveParams {
    var count: UInt32
    var seed: UInt32
    var baseStep: Float
}
struct DrawParams {
    var count: UInt32
    var frameWidth: UInt32
    var boxPixels: UInt32
}

func metalList(_ values: [Float]) -> String { values.map { "\($0)" }.joined(separator: ", ") }
func metalColors(_ values: [SIMD3<Float>]) -> String {
    values.map { "float3(\($0.x), \($0.y), \($0.z))" }.joined(separator: ", ")
}

let particleKernelSource = """
    #include <metal_stdlib>
    using namespace metal;

    struct Particle { float2 position; uint species; float flash; };
    struct MoveParams { uint count; uint seed; float baseStep; };
    struct DrawParams { uint count; uint frameWidth; uint boxPixels; };

    constant float DIFFUSION[6] = { \(metalList(Species.allCases.map { $0.diffusion })) };
    constant float RADIUS[6] = { \(metalList(Species.allCases.map { $0.radius })) };
    constant float3 COLOR[6] = { \(metalColors(Species.allCases.map { $0.color })) };
    constant uint HYDROGEN = \(Species.hydrogen.rawValue);

    uint hash(uint x) {
        x ^= x >> 16; x *= 0x7feb352du;
        x ^= x >> 15; x *= 0x846ca68bu;
        x ^= x >> 16;
        return x;
    }
    float uniform01(thread uint &state) {
        state = hash(state);
        return (float(state) + 0.5) / 4294967296.0;
    }

    // One thread per molecule: a random Brownian step. Each step's spread
    // grows with the square root of the diffusion coefficient.
    kernel void move(device Particle *particles [[buffer(0)]],
                     constant MoveParams &m [[buffer(1)]],
                     uint i [[thread_position_in_grid]]) {
        if (i >= m.count) return;
        Particle p = particles[i];
        if (p.species > 5) return;
        uint state = hash(i * 747796405u + m.seed * 2891336453u + 1u);
        float u1 = uniform01(state);
        float u2 = uniform01(state);
        // Box–Muller: two uniform numbers make one 2D Gaussian step.
        float r = sqrt(-2.0 * log(u1));
        float a = 6.28318530718 * u2;
        float2 step = float2(r * cos(a), r * sin(a)) * m.baseStep * sqrt(DIFFUSION[p.species]);
        float2 pos = p.position + step;
        // Bounce off the walls of the box.
        pos = abs(pos);
        pos = 1.0 - abs(1.0 - pos);
        p.position = clamp(pos, 0.0, 1.0);
        p.flash = max(p.flash - 0.1, 0.0);
        particles[i] = p;
    }

    // One thread per molecule: add 1 to its species' counter.
    kernel void countSpecies(device const Particle *particles [[buffer(0)]],
                             constant uint &count [[buffer(1)]],
                             device atomic_uint *counts [[buffer(2)]],
                             uint i [[thread_position_in_grid]]) {
        if (i >= count) return;
        uint s = particles[i].species;
        if (s <= 5) atomic_fetch_add_explicit(&counts[s], 1, memory_order_relaxed);
    }

    // One thread per pixel of the box: shade every molecule that covers it.
    kernel void draw(device uchar4 *pixels [[buffer(0)]],
                     device const Particle *particles [[buffer(1)]],
                     constant DrawParams &d [[buffer(2)]],
                     uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= d.boxPixels || gid.y >= d.boxPixels) return;
        float2 px = float2(gid) + 0.5;
        float size = float(d.boxPixels);
        float depth = px.y / size;
        float3 color = mix(float3(0.05, 0.13, 0.20), float3(0.03, 0.08, 0.14), depth);
        for (uint i = 0; i < d.count; i++) {
            Particle p = particles[i];
            if (p.species > 5) continue;
            float2 center = p.position * size;
            float dist = distance(px, center);
            float r = RADIUS[p.species];
            if (p.species == HYDROGEN) {
                // H⁺ is rare, so give it a glow to make it findable.
                float g = r * 3.0;
                color += COLOR[HYDROGEN] * 0.55 * exp(-dist * dist / (2.0 * g * g));
            }
            if (dist < r + 1.0) {
                float coverage = clamp(r - dist + 0.5, 0.0, 1.0);
                float t = clamp(dist / r, 0.0, 1.0);
                float z = sqrt(1.0 - t * t);
                float3 sphere = COLOR[p.species] * (0.45 + 0.55 * z) + 0.25 * pow(z, 8.0);
                color = mix(color, sphere, coverage);
            }
            if (p.flash > 0 && dist > r && dist < r + 1.8) {
                color = mix(color, float3(1.0), p.flash * 0.9);
            }
        }
        pixels[gid.y * d.frameWidth + gid.x] = uchar4(uchar3(round(clamp(color, 0.0, 1.0) * 255)), 255);
    }
    """

enum SimulationError: Error, CustomStringConvertible {
    case noMetalDevice
    case kernelCompile(String)
    case gpu(String)
    case full

    var description: String {
        switch self {
        case .noMetalDevice: return "no Metal GPU found"
        case .kernelCompile(let detail): return "could not compile the kernels: \(detail)"
        case .gpu(let detail): return "GPU error: \(detail)"
        case .full: return "the particle buffer is full"
        }
    }
}

/// The GPU to use. (Needs CoreGraphics linked; the Makefile does that.)
func findDevice() throws -> MTLDevice {
    if let device = MTLCreateSystemDefaultDevice() ?? MTLCopyAllDevices().first {
        return device
    }
    throw SimulationError.noMetalDevice
}

final class MoleculeBox {
    let device: MTLDevice
    let capacity: Int
    let particles: MTLBuffer
    private(set) var used = 0   // slots handed out so far (some may be inactive)
    private let queue: MTLCommandQueue
    private let movePipeline: MTLComputePipelineState
    private let countPipeline: MTLComputePipelineState
    private let drawPipeline: MTLComputePipelineState
    private let countBuffer: MTLBuffer

    init(device: MTLDevice, capacity: Int) throws {
        self.device = device
        self.capacity = capacity
        let library: MTLLibrary
        do {
            library = try device.makeLibrary(source: particleKernelSource, options: nil)
        } catch {
            throw SimulationError.kernelCompile(error.localizedDescription)
        }
        func pipeline(_ name: String) throws -> MTLComputePipelineState {
            guard let f = library.makeFunction(name: name) else { throw SimulationError.kernelCompile("no kernel \(name)") }
            do { return try device.makeComputePipelineState(function: f) } catch {
                throw SimulationError.kernelCompile(error.localizedDescription)
            }
        }
        movePipeline = try pipeline("move")
        countPipeline = try pipeline("countSpecies")
        drawPipeline = try pipeline("draw")
        guard let queue = device.makeCommandQueue(),
              let particles = device.makeBuffer(length: capacity * MemoryLayout<Particle>.stride, options: .storageModeShared),
              let countBuffer = device.makeBuffer(length: 6 * MemoryLayout<UInt32>.stride, options: .storageModeShared)
        else { throw SimulationError.gpu("could not allocate GPU buffers") }
        self.queue = queue
        self.particles = particles
        self.countBuffer = countBuffer
    }

    /// The particle array, read and written directly by the CPU (unified memory).
    var slots: UnsafeMutablePointer<Particle> {
        particles.contents().bindMemory(to: Particle.self, capacity: capacity)
    }

    // MARK: CPU bookkeeping

    func tally() -> [Int] {
        var counts = [Int](repeating: 0, count: 6)
        for i in 0..<used where slots[i].species <= 5 {
            counts[Int(slots[i].species)] += 1
        }
        return counts
    }

    func add<R: RandomNumberGenerator>(_ species: Species, count: Int, using rng: inout R) throws {
        var remaining = count
        var i = 0
        // Reuse inactive slots first.
        while remaining > 0 && i < used {
            if slots[i].species == Species.inactive {
                slots[i] = randomParticle(species, using: &rng)
                remaining -= 1
            }
            i += 1
        }
        while remaining > 0 {
            guard used < capacity else { throw SimulationError.full }
            slots[used] = randomParticle(species, using: &rng)
            used += 1
            remaining -= 1
        }
    }

    private func randomParticle<R: RandomNumberGenerator>(_ species: Species, using rng: inout R) -> Particle {
        let x = Float.random(in: 0...1, using: &rng)
        let y = Float.random(in: 0...1, using: &rng)
        return Particle(position: SIMD2(x, y), species: species.rawValue, flash: 1)
    }

    /// Picks `count` random molecules of one species.
    private func pick<R: RandomNumberGenerator>(_ species: Species, count: Int, using rng: inout R) -> [Int] {
        var candidates: [Int] = []
        for i in 0..<used where slots[i].species == species.rawValue { candidates.append(i) }
        candidates.shuffle(using: &rng)
        return Array(candidates.prefix(count))
    }

    func remove<R: RandomNumberGenerator>(_ species: Species, count: Int, using rng: inout R) {
        for i in pick(species, count: count, using: &rng) { slots[i].species = Species.inactive }
    }

    /// Turns `count` molecules of one species into another, where they stand
    /// (a reaction), and makes them flash.
    func convert<R: RandomNumberGenerator>(_ from: Species, to: Species, count: Int, using rng: inout R) {
        for i in pick(from, count: count, using: &rng) {
            slots[i].species = to.rawValue
            slots[i].flash = 1
        }
    }

    // MARK: GPU work

    private func threadgroup(_ p: MTLComputePipelineState) -> MTLSize {
        let w = p.threadExecutionWidth
        return MTLSize(width: w, height: 1, depth: 1)
    }

    private func run(_ encode: (MTLComputeCommandEncoder) -> Void) throws {
        guard let commands = queue.makeCommandBuffer(), let encoder = commands.makeComputeCommandEncoder() else {
            throw SimulationError.gpu("could not create a command encoder")
        }
        encode(encoder)
        encoder.endEncoding()
        commands.commit()
        commands.waitUntilCompleted()
        if let error = commands.error { throw SimulationError.gpu(error.localizedDescription) }
    }

    /// One Brownian step for every molecule. `baseStep` is the step spread,
    /// in box widths, for a diffusion coefficient of 1.
    func move(seed: UInt32, baseStep: Float) throws {
        var params = MoveParams(count: UInt32(used), seed: seed, baseStep: baseStep)
        try run { e in
            e.setComputePipelineState(movePipeline)
            e.setBuffer(particles, offset: 0, index: 0)
            e.setBytes(&params, length: MemoryLayout<MoveParams>.stride, index: 1)
            e.dispatchThreads(MTLSize(width: max(used, 1), height: 1, depth: 1),
                              threadsPerThreadgroup: threadgroup(movePipeline))
        }
    }

    /// Counts the molecules of each species on the GPU.
    func gpuCounts() throws -> [Int] {
        memset(countBuffer.contents(), 0, countBuffer.length)
        var count = UInt32(used)
        try run { e in
            e.setComputePipelineState(countPipeline)
            e.setBuffer(particles, offset: 0, index: 0)
            e.setBytes(&count, length: MemoryLayout<UInt32>.stride, index: 1)
            e.setBuffer(countBuffer, offset: 0, index: 2)
            e.dispatchThreads(MTLSize(width: max(used, 1), height: 1, depth: 1),
                              threadsPerThreadgroup: threadgroup(countPipeline))
        }
        let c = countBuffer.contents().bindMemory(to: UInt32.self, capacity: 6)
        return (0..<6).map { Int(c[$0]) }
    }

    /// Draws the box into the left `boxPixels` × `boxPixels` of a frame buffer
    /// that is `frameWidth` pixels wide. Returns the GPU time in seconds.
    @discardableResult
    func draw(into frame: MTLBuffer, frameWidth: Int, boxPixels: Int) throws -> Double {
        var params = DrawParams(count: UInt32(used), frameWidth: UInt32(frameWidth), boxPixels: UInt32(boxPixels))
        guard let commands = queue.makeCommandBuffer(), let e = commands.makeComputeCommandEncoder() else {
            throw SimulationError.gpu("could not create a command encoder")
        }
        e.setComputePipelineState(drawPipeline)
        e.setBuffer(frame, offset: 0, index: 0)
        e.setBuffer(particles, offset: 0, index: 1)
        e.setBytes(&params, length: MemoryLayout<DrawParams>.stride, index: 2)
        let w = drawPipeline.threadExecutionWidth
        let group = MTLSize(width: w, height: drawPipeline.maxTotalThreadsPerThreadgroup / w, depth: 1)
        e.dispatchThreads(MTLSize(width: boxPixels, height: boxPixels, depth: 1), threadsPerThreadgroup: group)
        e.endEncoding()
        commands.commit()
        commands.waitUntilCompleted()
        if let error = commands.error { throw SimulationError.gpu(error.localizedDescription) }
        return commands.gpuEndTime - commands.gpuStartTime
    }
}

// MARK: - Matching dots to the chemistry

/// Dot counts the chemistry calls for right now. Carbon is conserved: the
/// CO₂, H₂CO₃ and HCO₃⁻ dots always add up to `carbonDots`.
func targetCounts<R: RandomNumberGenerator>(_ state: BufferState, carbonDots: Int, sodiumDots: Int,
                                            using rng: inout R) -> [Int] {
    let co2 = stochasticRound(dots(state.co2), using: &rng)
    let acid = stochasticRound(dots(state.carbonicAcid), using: &rng)
    let bicarbonate = max(carbonDots - co2 - acid, 0)
    let hydrogen = stochasticRound(dots(state.hydrogen), using: &rng)
    let chloride = Int(dots(state.chloride).rounded())
    return [co2, acid, bicarbonate, hydrogen, sodiumDots, chloride]
}

/// Changes the box to match `target`, following the reactions: bicarbonate
/// picks up H⁺ to become carbonic acid, and carbonic acid gives off CO₂ (and
/// the reverse). Conversions into H₂CO₃ happen before those out of it, so
/// there's always enough to convert.
func applyTargets<R: RandomNumberGenerator>(_ target: [Int], to box: MoleculeBox, using rng: inout R) throws {
    var now = box.tally()
    // Filling a new box: create the carbon molecules first. After that, carbon
    // only changes form, never amount.
    let carbonNow: Int = now[0] + now[1] + now[2]
    let carbonTarget: Int = target[0] + target[1] + target[2]
    if carbonNow < carbonTarget {
        for s in [Species.co2, .carbonicAcid, .bicarbonate] {
            let missing: Int = target[Int(s.rawValue)] - now[Int(s.rawValue)]
            if missing > 0 { try box.add(s, count: missing, using: &rng) }
        }
        now = box.tally()
    }
    let co2 = Species.co2.rawValue, bi = Species.bicarbonate.rawValue
    let bicarbonateToAcid: Int = now[Int(bi)] - target[Int(bi)]     // > 0: HCO₃⁻ + H⁺ → H₂CO₃
    let acidToCO2: Int = target[Int(co2)] - now[Int(co2)]           // > 0: H₂CO₃ → CO₂ + H₂O
    if bicarbonateToAcid > 0 { box.convert(.bicarbonate, to: .carbonicAcid, count: bicarbonateToAcid, using: &rng) }
    if acidToCO2 < 0 { box.convert(.co2, to: .carbonicAcid, count: -acidToCO2, using: &rng) }
    if bicarbonateToAcid < 0 { box.convert(.carbonicAcid, to: .bicarbonate, count: -bicarbonateToAcid, using: &rng) }
    if acidToCO2 > 0 { box.convert(.carbonicAcid, to: .co2, count: acidToCO2, using: &rng) }

    for s in [Species.hydrogen, .sodium, .chloride] {
        let change: Int = target[Int(s.rawValue)] - now[Int(s.rawValue)]
        if change > 0 { try box.add(s, count: change, using: &rng) }
        if change < 0 { box.remove(s, count: -change, using: &rng) }
    }
}
