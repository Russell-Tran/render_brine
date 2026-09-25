// The chemistry of the carbonic acid–bicarbonate buffer, solved exactly.
//
//     CO₂ + H₂O  ⇌  H₂CO₃  ⇌  H⁺ + HCO₃⁻
//        (slow: seconds)   (fast: effectively instant)
//
// Concentrations are in millimolar (mM), at 25 °C. The water starts with
// sodium bicarbonate (baking soda) and some dissolved CO₂. Strong acid (HCl)
// is added as H⁺ plus Cl⁻, which just watches.
//
// Constants (Wikipedia, "Carbonic acid", 25 °C):
//   hydration   CO₂ + H₂O → H₂CO₃   0.039 s⁻¹
//   dehydration H₂CO₃ → CO₂ + H₂O   23 s⁻¹      (ratio ≈ 1.7 × 10⁻³)
//   apparent pKa of CO₂ + HCO₃⁻     6.35
// Carbonic acid's own pKa is derived from those, so everything agrees:
// Ka(apparent) = Kh × Ka(H₂CO₃), giving pKa(H₂CO₃) ≈ 3.58 (published: 3.45–3.75).
//
// Simplifications: carbonate (CO₃²⁻, pKa 10.3) and hydroxide are left out,
// which is fine below about pH 8.5. The system is closed: no CO₂ escapes.

import Foundation

enum Carbonate {
    static let hydrationRate: Double = 0.039   // s⁻¹
    static let dehydrationRate: Double = 23    // s⁻¹
    static let apparentPKa: Double = 6.35

    /// [H₂CO₃] / [CO₂] at equilibrium.
    static var hydrationConstant: Double { hydrationRate / dehydrationRate }
    /// Ka of carbonic acid itself, in mM.
    static var carbonicAcidKa: Double { pow(10, -apparentPKa) / hydrationConstant * 1000 }
    static var carbonicAcidPKa: Double { -log10(carbonicAcidKa / 1000) }
}

/// pH from an H⁺ concentration in mM.
func phFromHydrogen(_ mM: Double) -> Double { -log10(mM / 1000) }

struct BufferState {
    var co2: Double            // dissolved CO₂, mM
    var carbonicPool: Double   // H₂CO₃ + HCO₃⁻, mM (these two settle almost instantly)
    var sodium: Double         // Na⁺ from the baking soda, mM
    var chloride: Double = 0   // Cl⁻ from added HCl, mM
    var time: Double = 0       // seconds

    /// Water with this much bicarbonate and CO₂, already at equilibrium.
    init(bicarbonate: Double, co2: Double) {
        let acid: Double = Carbonate.hydrationConstant * co2
        let h: Double = Carbonate.carbonicAcidKa * acid / bicarbonate
        self.co2 = co2
        carbonicPool = bicarbonate + acid
        sodium = bicarbonate - h   // charge balance: H⁺ + Na⁺ = HCO₃⁻
    }

    /// H⁺ in mM. The fast step obeys Ka = [H⁺][HCO₃⁻] / [H₂CO₃] and charge
    /// balance [H⁺] + [Na⁺] = [HCO₃⁻] + [Cl⁻], which combine into
    /// h² + (q + Ka) h − Ka (pool − q) = 0 with q = [Na⁺] − [Cl⁻].
    var hydrogen: Double {
        let q: Double = sodium - chloride
        let ka: Double = Carbonate.carbonicAcidKa
        let b: Double = q + ka
        let c: Double = ka * (carbonicPool - q)
        // The stable form of the quadratic's positive root.
        return 2 * c / (b + (b * b + 4 * c).squareRoot())
    }
    var bicarbonate: Double { hydrogen + sodium - chloride }
    var carbonicAcid: Double { carbonicPool - bicarbonate }
    var pH: Double { phFromHydrogen(hydrogen) }
    var totalCarbon: Double { co2 + carbonicPool }

    /// The pH the Henderson–Hasselbalch equation gives for the current mix.
    var hendersonHasselbalchPH: Double { Carbonate.apparentPKa + log10(bicarbonate / co2) }

    mutating func addStrongAcid(_ mM: Double) { chloride += mM }

    /// Lets CO₂ and H₂CO₃ convert into each other for `seconds`.
    mutating func advance(by seconds: Double, step: Double = 1e-4) {
        var remaining = seconds
        while remaining > 1e-12 {
            let dt: Double = min(step, remaining)
            let flux: Double = Carbonate.hydrationRate * co2 - Carbonate.dehydrationRate * carbonicAcid
            co2 -= flux * dt
            carbonicPool += flux * dt
            time += dt
            remaining -= dt
        }
    }
}

/// pH of plain water after adding this much strong acid.
func plainWaterPH(afterAcid mM: Double) -> Double { phFromHydrogen(mM) }

/// "24.0 mM", "183 µM", "22 nM".
func concentrationLabel(_ mM: Double) -> String {
    if mM >= 1 { return String(format: "%.1f mM", mM) }
    if mM >= 0.001 { return String(format: "%.0f µM", mM * 1000) }
    return String(format: "%.0f nM", mM * 1_000_000)
}

// MARK: - Randomness

/// A small, seedable random number generator (SplitMix64), so runs repeat exactly.
struct SplitMix64: RandomNumberGenerator {
    var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}

/// Rounds 2.3 to 3 with probability 0.3 and to 2 otherwise, so the average is right.
func stochasticRound<R: RandomNumberGenerator>(_ x: Double, using rng: inout R) -> Int {
    let whole: Double = x.rounded(.down)
    let fraction: Double = x - whole
    let extra = Double.random(in: 0..<1, using: &rng) < fraction ? 1 : 0
    return Int(whole) + extra
}
