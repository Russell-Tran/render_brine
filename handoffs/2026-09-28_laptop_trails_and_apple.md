# Handoff: ant trails and apple (steps 55–57, 68, 69) for the M3 Max laptop

Written 2026-09-28 by the M4 mini session. Russell has **pre-approved** these
five steps. **Don't wait for a go-ahead.** Do your own research first, and put a
short "what I found and decided" section at the top of each commit message.
Every rule in [`2026-09-27_laptop_ant_animations.md`](2026-09-27_laptop_ant_animations.md)
applies unchanged: the SDK flag, the Swift 5.10 GIF.swift copy, citations or
MODEL, tests, mutants, GIF ≤ ~10 MB, git etiquette, and never touching
README.md.

Plan page (Russell's view): https://claude.ai/artifact/5jTYgVLboqcWgHPxVEfjeL

| Step | Folder | Russell's words | Build from |
|---|---|---|---|
| 55 | `step055_ant_trail_5/` | 5 ants walking on a pheromone trail, animated | `step044_ant_trail/` |
| 56 | `step056_ant_trail_7/` | 7 ants walking on a pheromone trail, animated | step 44 |
| 57 | `step057_ant_trail_wood/` | 3 ants walking on a pheromone trail on a piece of wood, animated | step 44 |
| 68 | `step068_ants_apple/` | 3 ants interacting with a piece of peeled apple | steps 44 + 30/50 |
| 69 | `step069_ant_bites_apple/` | 1 ant biting a piece of peeled apple | steps 30/50 |

Steps 58–67 (toys and teeth) are on the mini. Don't build them.

## New rule from step 29: an animation must be visibly animated

Step 29's pollen tube grew at true width, which was sub-pixel, and Russell
"cannot discern whatsoever". **Every animation needs a test that the change
shows on screen**: pixels that change frame to frame, in the region the
caption talks about, above a meaningful count. Add a mutant that makes the
motion invisible and must fail. When something is sub-pixel at true scale,
widen it and label it ("drawn ~N× wider"), or give it a magnified inset
that tracks it, as steps 48 and 54 did.

## Leads for each step (unverified; check every one)

- **55, 56: longer files.** Step 44's conveyor generalises. The ants are an
  endless file N long, each walks exactly one spacing per loop, and a whole
  number of strides fits in it. Check that ant spacing on a trail has a
  source. If not, mark it MODEL. The frame gets wider, so measure GIF size
  early. Keep the alternating tripod, planted feet, the footfall diagram and
  the pheromone dab.
- **57, wood.** A planed softwood board (e.g. pine) with real grain: earlywood
  and latewood bands, and rays if they show at this scale. Wood is porous, so
  if a source compares trail persistence on porous and non-porous surfaces,
  use it. Otherwise don't claim it. Feet stay planted on the grain relief.
- **68, three ants and apple.** A peeled, cut piece of a named apple
  cultivar, with its size given. Leads: fructose is the dominant sugar;
  esters (e.g. hexyl acetate, 2-methylbutyl acetate) and C6 green volatiles
  from the cut. The cut surface is wet. The ants arrive, tap, taste and drink.
  Browning (polyphenol oxidase) runs on minutes to hours: leave it out, or
  put it on its own labelled clock. Never mix clocks without saying so.
- **69, one ant biting.** Formicine mandibles open and close on the flesh.
  Lead: ants can't swallow solids. Liquid is drunk, and particles are strained
  into the **infrabuccal pocket** and later spat out as a pellet. Verify this
  for *Lasius* or formicines before putting it in a caption. An inset should
  show where the juice goes and where the solids go. The mandibles must meet
  the flesh from geometry, and never close through it.

## Order

55 → 56 → 57 → 68 → 69 unless Russell says otherwise. Ping the mini session
("Render Brine") as each one is pushed.

## Git

One commit per step. Run `git pull --rebase` before every push, and never
force-push. Don't touch README.md. End every commit message with:

```
Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Sx815ZFMDKuTuxMQYVftV1
```
