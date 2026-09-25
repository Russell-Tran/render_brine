# Step 10: Getting a plasmid in

Bacterial transformation — how pGLO crosses into *E. coli*. Two looping GIFs:

- `renders/approach.gif` — a supercoiled pGLO drifting onto the outer membrane, in cross-section
- `renders/routes.gif` — three ways in, side by side, each labelled with how well it is actually known

## The one thing to know about this step

It is the first render in the series whose central event is **not measured**.
The CaCl₂-and-heat-shock method dates to 1970 and its molecular mechanism has
never been established. The render draws the leading model in full — a model
that has guided fifty years of successful protocol has earned a depiction — and
says on screen that it is a model. An **evidence bar** in the corner of every
frame carries three marks: filled for measured, part-filled for simulated,
hollow for a model.

## Running it

```sh
make run        # both GIFs, about 50 s on an M4 mini
make bench      # brute force against the uniform grid
make test       # 28 tests
make structure  # build Resources/scene.json — REQUIRED before the first run
```

If swiftc complains that the SDK doesn't match the compiler:

```sh
make test SDK=/Library/Developer/CommandLineTools/SDKs/MacOSX26.sdk
```

### On the M3 Max

The laptop is **not needed for this step** — both GIFs render in under a minute
on the mini. The code does build and run there (no macOS 15+ APIs,
`fastMathEnabled = false`, clean slow-expression scan) if you want the
comparison:

```sh
git pull && cd step10_transformation
make test SDK=$(xcrun --show-sdk-path)
make bench SDK=$(xcrun --show-sdk-path)
make run  SDK=$(xcrun --show-sdk-path)
```

`Resources/scene.json` is generated, not committed: it is 46 MB of bead
positions that `make structure` rebuilds in about four seconds. Run it once
after cloning.

## What is measured, and what is not

| Part | Level | Source |
|---|---|---|
| Bilayer 4.7 nm | measured | AFM of a PE/PG/cardiolipin bilayer, *Langmuir* 41:12301 (2025) |
| Periplasm 13 nm | measured | 12 nm CEMOVIS / 14 nm cryo-ET, *J. Electron Microsc.* 59:419 (2010) |
| Outer membrane to peptidoglycan 11 nm | measured | same, both methods agree |
| Lipid area 0.588 nm² | measured | POPE, Hills et al., *J. Comput. Chem.* 37 (2016) |
| σ = −0.06 | measured | plasmids from *E. coli* in mid-exponential growth |
| ComEA structure | measured | [PDB 8DFK](https://www.rcsb.org/structure/8DFK), *B. subtilis* |
| Electroporation pore | simulated | shapes molecular-dynamics studies produce under field |
| Supercoiled plasmid shape | model | no structure of a whole plasmid exists |
| Ca²⁺ arrangement, and the crossing itself | model | inferred from bulk behaviour, never observed |
