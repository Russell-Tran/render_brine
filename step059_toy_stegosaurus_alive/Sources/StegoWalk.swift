// How the toy Stegosaurus walks when it comes to life.
//
// WHAT IS KNOWN. Stegosaurs walked on all four feet: their tracks, the
// ichnogenus Deltapodus, preserve hand and foot prints together (Whyte &
// Romano 1994, via Wikipedia's "Deltapodus", checked). No trackway reached
// gives a stegosaur's footfall order or duty factor, and none could — tracks
// record where feet landed, not when.
//
// WHAT IS INFERRED. The footfall order and timing are borrowed from living
// quadrupeds' ordinary walk: the lateral sequence, left hind, left fore,
// right hind, right fore, "the usual lateral sequence pattern of horses",
// with "a typical walk ha[ving] a duty factor around 0.65 ... and HL–FL phase
// of 25%" (Usherwood & Smith, "The grazing gait, and implications of
// toppling table geometry for primate footfall sequences", *Biol Lett*
// 14:20180137, 2018, PMC6012707 — full text checked). For Stegosaurus this
// is INFERRED and the caption says so.
//
// Stride length. Relative to hip height it is MODEL: 15 mm on a 19 mm hip,
// λ/h ≈ 0.8 — a slow walk — kept inside what the legs reach with the body
// lowered 1.8 mm (a test checks no leg is ever stretched past its length),
// and short enough that the hind knee, coming forward, never meets the
// foreleg lifting away in front of it (a test checks that too; at 18 mm on
// a tighter circle it did).

import Foundation

let stegoWalk = WalkSpec(
    dutyFactor: 0.65,
    offsets: gaitOffsets(activeMutant),
    footfallOrder: ["LH", "LF", "RH", "RF"],
    stride: 15.0,
    strides: 15,
    strideSeconds: 1.0,
    crouch: 1.8,
    footLift: 1.6,
    lookYaw: 0.6,
    tailSway: 0.07)

let stegoAnimCaption = AnimCaption(
    title: "A toy Stegosaurus comes to life",
    lines: [
        "Fiction: a one-piece toy has no joints. All else is kept true: rigid",
        "plastic pieces turning at hips, knees and neck; feet that never slide.",
        "The gait is a real quadruped walk: left hind, left fore, right hind,",
        "right fore, each foot down 65% of the stride (Usherwood & Smith 2018).",
        "For Stegosaurus the order is inferred: its tracks show four feet, not when.",
    ])
