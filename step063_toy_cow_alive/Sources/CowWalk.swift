// How the toy cow walks when it comes to life.
//
// THE GAIT IS SOURCED. Cattle walk with the lateral sequence — left hind,
// left fore, right hind, right fore. Usherwood & Smith filmed grazing cattle,
// sheep and horses and state that "the footfall order during grazing is the
// same as for walking", and that in grazing livestock "a step forward with a
// forefoot is consistently and immediately followed by a step forward from
// the hind" — the lateral sequence's fore-then-opposite-hind; they give "a
// typical walk" a duty factor "around 0.65" and each footfall a quarter
// stride after the last ("The grazing gait, and implications of toppling
// table geometry for primate footfall sequences", *Biol Lett* 14:20180137,
// 2018, PMC6012707 — full text checked). No cattle-specific duty factor was
// reached; the typical walk's is used — flagged.
//
// Stride and the body's drop while walking are MODEL: 20 mm on a 22 mm hind
// leg (λ/h ≈ 0.9, a slow walk), the body lowered 3.0 mm so the nearly
// straight forelegs reach (a test checks none is ever stretched).

import Foundation

/// The animated cow: step 62's toy without the still's neck and hip fillets
/// — come to life, those are separate pieces that turn.
func cowAnimDesign(_ mutant: Mutant = activeMutant) -> ToyDesign {
    var d: ToyDesign = cowDesign(mutant)
    d.headFillet = 0
    d.legFillet = 0
    return d
}

let cowWalk = WalkSpec(
    dutyFactor: 0.65,
    offsets: gaitOffsets(activeMutant),
    footfallOrder: ["LH", "LF", "RH", "RF"],
    stride: 20.0,
    strides: 12,
    strideSeconds: 1.0,
    crouch: 3.0,
    footLift: 1.9,
    lookYaw: 0.55,
    tailSway: 0.16)

let cowAnimCaption = AnimCaption(
    title: "A toy cow comes to life",
    lines: [
        "Fiction: a one-piece toy has no joints. All else is kept true: rigid",
        "plastic pieces turning at hips, knees and neck; feet that never slide.",
        "The gait is the cow's own: a lateral-sequence walk, left hind, left fore,",
        "right hind, right fore, each foot down about 65% of the stride",
        "(Usherwood & Smith 2018: cattle, grazing or walking, step in this order).",
    ])
