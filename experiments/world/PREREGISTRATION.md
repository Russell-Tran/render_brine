# World-building experiment: pre-registration

Written and committed on 2026-09-28, **before any of the pilot is built**. Russell's
proposal page: https://claude.ai/artifact/L1XauFiED4fssp4bFu9Lxt

## Hypothesis

Our stack can already **draw** a world: the renderer, a tested ant, props and science
all exist. It can't yet **run** one. The gaps are about time, not pictures:

1. no video format beyond short GIFs;
2. no shared ant, since it has been copied into 25+ folders;
3. no simulation clock: nothing happens unless it's scripted;
4. unknown scaling with many characters;
5. no pause-and-resume rendering for long jobs.

## The pilot

**Step 79, "A minute of ant world"**, `step079_ant_world/`: about 60 seconds of 1080p
**MP4**. Workers leave a nest and wander. One finds a sugar pile, tastes it and heads
home laying pheromone. Others meet the scent, follow it and reinforce it, and a trail
**emerges**. No path is scripted; the story comes from the rules. The pheromone field is
simulated on the GPU as a grid that is deposited, diffused and evaporated each tick.

## Predictions (checked after the build)

| # | Question | Prediction | How it's checked |
|---|---|---|---|
| P1 | Can we write MP4? | Yes, with AVFoundation (`AVAssetWriter`, H.264, hardware encoder), no macOS 15+ APIs, compiling on both machines' toolchains. At the same frames and resolution, the MP4 is **≥ 10× smaller** than a GIF. | File sizes compared on the same frames |
| P2 | Does a trail emerge? | Yes, from the rules alone, within the film. With pheromone deposition switched off, no trail forms. | A test measuring trail formation (e.g. ants' paths concentrating onto one route), and a mutant `noPheromone` that must fail it |
| P3 | How many ants can the renderer hold? | Frame time grows roughly **linearly** with ant count. Somewhere between **20 and 40 ants** it gets impractically slow without an acceleration structure. | Time one frame at 1, 5, 10, 20 and 40 ants on each machine |
| P4 | Can the ant be shared? | Yes: one versioned ant module (`lib/ant/v1/`) that the pilot imports while old steps stay untouched. The **biggest change** to how we work, because it bends copy-don't-modify. | The pilot builds against the shared module; every old ant step's suite still passes unchanged |
| P5 | How long does a minute take? | **Several hours** of rendering. A paused job resumes without re-rendering finished frames, and output is identical either way. | Wall-clock time; a test that a resumed render matches an uninterrupted one on sample frames |
| P6 | Can a stranger follow it? | The legibility method from experiment 1: a context-free viewer, shown frames across the minute, can say that **a trail formed on its own**. | The same viewer and grader protocol as `experiments/legibility/` |

## What would count as the hypothesis being wrong

- **"The stack is already enough"** if the pilot needs **none** of the five gaps filled
  beyond trivial glue. I don't expect that.
- **"The gap is deeper than time"** if something outside the five breaks badly: e.g.
  the renderer can't hold even 10 ants at 1080p in reasonable time, or trails won't
  emerge believably with any sourced parameters. That would name the real limit.

## Honesty rules carried over

Every constant is cited or marked MODEL. Real-time versus time-lapse is labelled on
screen. The film plays once through and never rewinds. Tests and mutants apply, and so
does step 29's rule that the change must be visible.
