# World-building experiment: results

Hypothesis and predictions: [`PREREGISTRATION.md`](PREREGISTRATION.md) (ae6f1e6), plus the P6
protocol [`P6_PROTOCOL.md`](P6_PROTOCOL.md) (b6af3ff). Both were committed before the thing
they predict was built or viewed. The running log with more detail is [`FINDINGS.md`](FINDINGS.md).
The pilot is step 79, [`step079_ant_world/`](../../step079_ant_world/): a one-minute, 1080p30
MP4 of *Lasius niger* workers forming a pheromone trail, with nothing scripted. Built on the
M3 Max laptop and verified on the M4 mini.

## Verdict

**The hypothesis holds: the stack could draw a world but not run one, and the gaps were
about time.** Filling the five named gaps was enough to make a minute of world that works,
and a stranger reads its story correctly. Building it also found **three limits I didn't
predict**, and they are the real lessons (below).

| # | Prediction | Result |
|---|---|---|
| P1 | MP4 works; ≥ 10× smaller than GIF | **Partly right.** It works on both toolchains with the hardware encoder. 12–21× smaller at the film's bitrate, but softer; **3–6× at matched quality.** |
| P2 | A trail emerges from rules alone | **Right.** It formed in 7/8 seeds, and in 0/8 with pheromone off. The mutant `noPheromone` is caught. |
| P3 | Linear cost; impractical at 20–40 ants | **Partly right.** Linear, yes, but impractical already at **10**. The shared ant's per-ant bounding spheres make it 30–47× faster with identical pixels, so no heavier structure was needed. |
| P4 | One shared, versioned ant; the biggest change to how we work | **Right** on the module: `lib/ant/v1`, old steps untouched and passing, and the pilot builds against it. Whether it was the *biggest* change is Russell's call; in practice it took one new rule (`lib/README`). |
| P5 | Several hours to render; resume works | **Wrong on time:** about **50 min** for the minute (1.65 s/frame at 4 spp on the laptop). **Right on resume:** interrupted, rebuilt and resumed, with 8 frames redrawn in reverse, all bit-identical. |
| P6 | A stranger can tell the trail formed on its own | **Right.** The grader scored subject 1, change 2, where 1, emergence 1. The viewer: "the blue goes from nothing to dots to a continuous road… that irregularity reads as the product of the ants' own footsteps rather than a drawn path." |

## Three things I didn't predict

1. **A world must be recorded, not re-run.** The same seed gives a different world on the M3
   laptop and the M4 mini (trail at 716 s vs 506 s of ant time). The colony is chaotic, and
   last-bit floating-point differences grow into different paths. So step 79 simulates once,
   saves the record (7.97 MB, SHA-256 in a manifest), and renders **only** from it. A test
   checks that the renderer contains no simulation code.
2. **Renders from the record are near-identical, not bit-identical, across GPUs.** Three of
   four reference frames match to 1/255. Frame 1799, the close-up with the refracting sugar
   crystals large in frame, has rare pixels off by up to 10/255, outside the max < 8 tolerance
   I set beforehand. The test stays as recorded and fails on the mini (47/48). The outlier
   count and location are pending the laptop's analysis. **Future films** adopt a two-part
   tolerance, stated before the second machine renders: the 99.9th percentile ≤ 2/255, and the
   max ≤ 16/255, reported with a count.
3. **The shared ant can't back up.** Free-roaming ants meet head-on, which scripted trails
   never did. v1 ants can't reverse or turn on the spot, so at the food they **pass through
   each other** in 1212 of 1800 frames, up to 0.60 mm deep. Queueing made it worse. This is
   the one requirement the pilot doesn't meet, and it names the next build: **`lib/ant/v2`, an
   ant that can back up.**

## What it means for world-building

Before this, every render was a snapshot. Now the stack has what a longer world needs: video,
a shared cast, a simulation clock, a speed-up for crowds, and recorded, resumable renders.
The next limits are named rather than guessed: ant behaviour (v2), and pixel-level
reproducibility across GPUs.

## Honest limits of this experiment

- One pilot, one seed chosen before results (seed 79), one minute. P2 rests on 8 seeds.
- P6 is one AI viewer, not a person. The legibility experiment found AI viewers generous but
  able to detect invisibility. Russell dropped the human check there, and there's none here.
- The failing cross-machine test on the mini is reported, not fixed.
