// How the toy pig walks when it comes to life.
//
// THE GAIT IS SOURCED. Pigs walk with the lateral sequence — left hind, left
// fore, right hind, right fore: Yucatan miniature pigs on a treadmill "use a
// lateral sequence footfall pattern across all speeds" (Boakye et al.,
// "Treadmill-based gait kinematics in the Yucatan mini pig", *J Neurotrauma*
// 37:2277–2291, 2020, PMC9836690 — abstract checked; the full text was not
// released). On a concrete floor "the pigs walked with a four-beat gait"
// (Thorup et al., "Biomechanical gait analysis of pigs walking on solid
// concrete floor", *Animal* 1:708–715, 2007 — abstract checked). No pig duty
// factor was reached; the typical walk's 0.65, with each footfall a quarter
// stride after the last, is Usherwood & Smith's ("The grazing gait ...",
// *Biol Lett* 14:20180137, 2018, PMC6012707 — full text checked). Thorup et
// al. also found the forelimbs' stance lasts longer than the hind limbs';
// that difference is not modelled — UNVERIFIED how large it is.
//
// Stride and the body's drop while walking are MODEL: 14 mm on a 15.4 mm
// hip (λ/h ≈ 0.9, a slow walk), with the body lowered 1.9 mm so the short
// legs reach (a test checks none is ever stretched). The last hold is 1.54 s
// so the loop is a whole number of 6-hundredth frames.

import Foundation

/// The animated pig: step 60's toy, without the still's fillets where the
/// head and legs meet the body — come to life, those are separate pieces
/// that turn, and a fillet would have to bend like flesh.
func pigAnimDesign(_ mutant: Mutant = activeMutant) -> ToyDesign {
    var d: ToyDesign = pigDesign(mutant)
    d.headFillet = 0
    d.legFillet = 0
    return d
}

let pigWalk = WalkSpec(
    dutyFactor: 0.65,
    offsets: gaitOffsets(activeMutant),
    footfallOrder: ["LH", "LF", "RH", "RF"],
    stride: 14.0,
    strides: 14,
    strideSeconds: 1.0,
    crouch: 1.9,
    footLift: 1.4,
    lookYaw: 0.55,
    tailSway: 0.18,
    holdEnd: 1.54)

let pigAnimCaption = AnimCaption(
    title: "A toy pig comes to life",
    lines: [
        "Fiction: a one-piece toy has no joints. All else is kept true: rigid",
        "plastic pieces turning at hips, knees and neck; feet that never slide.",
        "The gait is the pig's own: a lateral-sequence walk, left hind, left fore,",
        "right hind, right fore (Boakye et al. 2020), each foot down about 65%",
        "of the stride (a typical walk: Usherwood & Smith 2018).",
    ])
