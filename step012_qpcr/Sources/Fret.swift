// Förster resonance energy transfer, and the one new thing in this step:
// a brightness that is COMPUTED from the geometry of the scene on every
// frame, rather than keyframed by hand.
//
// A donor dye next to an acceptor can hand its energy over without ever
// emitting a photon. How efficiently it does that depends on their
// separation raised to the sixth power:
//
//     E = 1 / (1 + (r / R0)^6)
//
// R0, the Förster radius, is the separation at which half the energy
// transfers. The sixth power is what makes this worth rendering: it is not a
// dimmer but a switch, and almost all of the action happens over about four
// nanometres.
//
// WHAT THIS IS NOT USED FOR, which matters. The step was proposed around a
// TaqMan hydrolysis probe - one probe with a reporter and a quencher, cut by
// the polymerase. The numbers rule that out. When the polymerase arrives, the
// probe is hybridised, and hybridised DNA is stiff: a 25-mer is a rigid rod
// 8.5 nm long, so reporter and quencher sit at opposite ends of it. Transfer
// at 8.5 nm with R0 near 5 nm is about 4%, i.e. the intact probe would be
// almost fully BRIGHT. Real TaqMan probes are dark until cut, so FRET is not
// what darkens them; contact quenching is, and contact quenching is a binding
// equilibrium with no distance law to compute.
//
// This render therefore uses the LightCycler HybProbe chemistry, where two
// probes land head-to-tail a few nucleotides apart and FRET between them is
// the designed mechanism - and where the separation is set by base-pair
// geometry, which is measured.

import Foundation
import simd

/// The Förster radius for the donor/acceptor pair used here, in ångströms.
///
/// R0 is a property of the PAIR, not of either dye alone: it depends on how
/// far the donor's emission spectrum overlaps the acceptor's absorption, on
/// the donor's quantum yield, on the refractive index of the medium, and on
/// how the two transition dipoles happen to be oriented.
///
/// HONESTY NOTE. The assay this render follows uses a fluorescein donor and an
/// LC Red 640 acceptor, and I could not reach a tabulated R0 for that exact
/// pair. 55 Å is the tabulated value for fluorescein with a rhodamine
/// acceptor (fluorescein-tetramethylrhodamine), and pairs in this family run
/// about 53-61 Å - Cy3-Cy5 at 53, fluorescein-QSY7 at 61. So this is a
/// representative figure for the class of pair, not a measurement of the one
/// named on screen, and the results page says so.
let forsterRadius: Float = 55.0

/// Transfer efficiency at separation `r` ångströms. This is the whole law.
func transferEfficiency(separation r: Float, R0: Float = forsterRadius) -> Float {
    guard r > 0 else { return 1 }
    let ratio: Float = r / R0
    let r2: Float = ratio * ratio
    let r6: Float = r2 * r2 * r2
    return 1 / (1 + r6)
}

/// The separation at which a given efficiency is reached, in ångströms.
/// The inverse of the law, used by the tests to check it end to end.
func separation(forEfficiency e: Float, R0: Float = forsterRadius) -> Float {
    guard e > 0, e < 1 else { return e <= 0 ? .infinity : 0 }
    let inner: Float = (1 - e) / e
    return R0 * pow(inner, 1.0 / 6.0)
}

// MARK: - The two dyes

/// Emission wavelengths, nanometres. The donor is a fluorescein; the acceptor
/// is a red dye of the LC Red 640 family. Both colours in this render are
/// computed from these numbers through the CIE matching functions, as step 9a
/// established - no colour here was picked by eye.
let donorEmission: Float = 520
let acceptorEmission: Float = 640

/// What the viewer should see at a given transfer efficiency.
///
/// This is the physical content of the picture. Energy that transfers leaves
/// as the ACCEPTOR's colour; energy that does not leaves as the DONOR's. So
/// as the two probes come together the green fades and the red comes up, and
/// the sum stays roughly constant - which is exactly what the instrument sees
/// and why the ratio of the two channels is what gets measured.
struct DyeState {
    var donorBrightness: Float     // 0...1
    var acceptorBrightness: Float  // 0...1
    var efficiency: Float
    var separation: Float          // ångströms

    init(separation r: Float, R0: Float = forsterRadius) {
        let e: Float = transferEfficiency(separation: r, R0: R0)
        self.efficiency = e
        self.separation = r
        self.donorBrightness = 1 - e
        self.acceptorBrightness = e
    }
}

/// The donor's colour, computed from its emission wavelength.
func donorColour() -> SIMD3<Float> { spectralColour(nanometres: donorEmission).linear }

/// The acceptor's colour, computed from its emission wavelength.
func acceptorColour() -> SIMD3<Float> { spectralColour(nanometres: acceptorEmission).linear }
