# Handoff: "A minute of ant world" (step 79) for the M3 Max laptop

Written 2026-09-28 by the M4 mini session. Russell gave the go ("I'm ready for A Minute
of Ant World"). It's a pilot for experiment 2. **Read
[`experiments/world/PREREGISTRATION.md`](../experiments/world/PREREGISTRATION.md) first:**
the pilot exists to test its predictions P1–P6, so measure them and report each one
honestly, including where I'm wrong. Russell's page: https://claude.ai/artifact/L1XauFiED4fssp4bFu9Lxt

All the usual rules apply (see `2026-09-27_laptop_ant_animations.md`): cite or MODEL,
tests and mutants, honest distance functions, no macOS 15+ APIs, `fastMathEnabled =
false`, your risky-line checker, the Swift 5.10 GIF.swift copy, never touching
README.md, `git pull --rebase`, never force-push. Open each commit message with
"Found and decided:".

## What to build, in this order (commit each part separately)

1. **`lib/ant/v1/`: the shared ant (P4).** Extract the black garden ant (body, legs,
   antennae, the tripod walk, planted feet) into one module, taken from the steps 44/55–57
   lineage. It must compile on both toolchains. Old steps stay **untouched**, keeping
   their copies. Only step 79 imports the module, via its Makefile. Version-pin it: a
   future change makes `v2` and never edits `v1`. Add module tests.
2. **The MP4 writer (P1).** `AVAssetWriter` with H.264 via the hardware encoder,
   1920×1080, 30 fps. Verify it compiles and runs on both machines (the mini will check).
   Measure MP4 against GIF on the same frames.
3. **The simulation (P2).** A GPU pheromone grid (deposit, diffuse, evaporate) plus ant
   agents with simple sourced rules: random-walk search, taste on contact with sugar (step
   26's science), lay pheromone on the way home, follow the gradient when on scent.
   Research *Lasius niger* recruitment (e.g. Beckers, Deneubourg & Goss's trail work),
   deposition, and evaporation (step 44 cites a ~47 min mark lifetime). Decide honestly
   between real time and time-lapse: trail formation in real colonies takes minutes,
   longer than a minute of film. **Put the clock on screen.** The simulation must be
   deterministic (seeded) so renders are reproducible. Test that the trail emerges, e.g.
   paths concentrate onto one route over time. Mutant `noPheromone` must fail it.
4. **Scaling measurement (P3).** Frame time at 1, 5, 10, 20 and 40 ants at 1080p.
   Report the curve. If it's too slow, a simple speed-up (per-ant bounding spheres,
   skipping far ants) is in scope; say what you did.
5. **The film.** About 60 s, 1080p30 MP4. Nest entrance, a sugar pile (step 26's grains),
   a plain ground surface (step 57's wood or a card surface; your call), and a dozen to
   twenty workers. The camera starts wide and ends on the formed trail. A caption line
   states the clock and the "pheromone drawn visible; invisible in life" convention from step 44.
   **Resumable rendering (P5):** frames written to disk, a restart skips finished ones,
   and a test checks that resumed frames match fresh ones. Report the wall-clock time.
   Also make a **short GIF excerpt** (≤ 10 MB) for the README, and put the MP4 at
   `showcase/ant_world.mp4`. Keep it small enough for git, aiming for ≤ 20 MB, and
   report the size.

## Tests and mutants (minimum)

The shared module's tests; the MP4 is valid and has the right frame count and duration;
the trail emerges; feet planted; no interpenetration; forward-only (no rewind); the
visibility rule (the trail's formation shows on screen). Mutants: `noPheromone`,
`sliding_feet`, `rewind`, plus one on the MP4 (e.g. `wrongFrameCount`).

## Report (to the mini session, "Render Brine")

After each part: the commit, what was measured, and **each prediction P1–P5 marked
right, wrong or partly right, with numbers.** The mini runs P6 (the stranger viewer)
and verifies everything before the README.
