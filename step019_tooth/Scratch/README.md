Probes, kept because each one is a check that is cheaper to run by hand than to
carry in the suite, and because each one caught something.

Not part of the build. Build any of them with, e.g.:

    SDK=/Library/Developer/CommandLineTools/SDKs/MacOSX26.sdk
    swiftc -O -sdk $SDK Sources/Lattice.swift Sources/Tooth.swift Sources/Model.swift \
           Scratch/yardstick_cervical.swift -o .build/yardstick

(the last file on the command line is treated as main, so top-level code is legal
in it).

* `probe_beam.swift` — the analytic cantilever and bar. Now a test, but still the
  quickest way to see the 3/rows behaviour on one screen. Note that it predates
  the `TissueProperties` change from critical strain to tensile strength and
  needs one edit to compile.
* `probe_tooth.swift` — the run that produced the catastrophic-failure result
  described in the handover: 25,444 bonds broken in ten rounds with the root
  clamped rigidly at −6 mm. Kept as the record of what was wrong. It no longer
  compiles against the current sources, which is the point.
* `yardstick_cervical.swift` — Euler–Bernoulli on the CEJ section, against the
  lattice. Written by the coordinating session; it is the check that found that
  the 60 µm cervical enamel is below mesh resolution at every mesh this step can
  afford, so what the lattice breaks first at the neck is the junction band and
  not the enamel. `cervicalBeamStress` in Tooth.swift is this, lifted, and a test
  uses it.
