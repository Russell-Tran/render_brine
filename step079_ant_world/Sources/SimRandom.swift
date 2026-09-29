// Step 79: a small seeded random-number generator for the simulation.
// SplitMix64 (Steele, Lea & Flood 2014, the generator Java's SplittableRandom
// and many PRNG seeders use): 64-bit state, one add and a mix per draw, so the
// same seed gives the same stream on every machine. Each ant owns its own
// stream, seeded from the world seed and its id, so the ants' draws never
// depend on the order anything else consumed numbers.

import Foundation

struct SimRandom {
    private(set) var state: UInt64

    init(seed: UInt64, stream: UInt64) {
        // Mix the stream id in so neighbouring ids start far apart.
        var s: UInt64 = seed ^ (stream &* 0xD1B5_4A32_D192_ED03)
        s = s &+ 0x9E37_79B9_7F4A_7C15
        state = s
        _ = next()
    }

    mutating func next() -> UInt64 {
        state = state &+ 0x9E37_79B9_7F4A_7C15
        var z: UInt64 = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }

    /// Uniform in [0, 1), 53 bits.
    mutating func uniform() -> Double {
        let bits: UInt64 = next() >> 11
        return Double(bits) / 9_007_199_254_740_992.0
    }

    /// Uniform Float in [lo, hi).
    mutating func float(_ lo: Float, _ hi: Float) -> Float {
        let u: Float = Float(uniform())
        return lo + (hi - lo) * u
    }

    /// Standard normal (Box–Muller; one value per call, the other discarded
    /// so the stream's use never depends on history).
    mutating func normal() -> Float {
        let u1: Double = max(uniform(), 1e-300)
        let u2: Double = uniform()
        let r: Double = (-2.0 * log(u1)).squareRoot()
        let a: Double = 2.0 * Double.pi * u2
        return Float(r * cos(a))
    }

    /// Waiting time of a Poisson process of `rate` per second.
    mutating func exponential(rate: Double) -> Float {
        let u: Double = max(uniform(), 1e-300)
        return Float(-log(u) / rate)
    }
}
