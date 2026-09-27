# Handoff: six ant animations for the M3 Max laptop

Written 2026-09-27 by the Claude Code session on the M4 mini, for a Claude Code
session on Russell's M3 Max laptop. The mini's GPU is saturated with other
renders, so these six steps are **assigned to the laptop**. The mini session
will not build them.

## The job

Build six animated steps. Each one is a still that already exists, **slightly
and slowly animated** — Russell's words — using the antenna-tapping motion
worked out in step 30.

| Step | Folder to create | Animates | The still's folder |
|---|---|---|---|
| 33 | `step033_ant_salt_tapping/` | the ant tasting table salt | `step032_ant_salt/` |
| 35 | `step035_ant_honey_tapping/` | the ant with a drop of honey | `step034_ant_honey/` |
| 37 | `step037_ant_yolk_tapping/` | the ant with egg yolk | `step036_ant_yolk/` |
| 39 | `step039_ant_water_tapping/` | the ant touching a water droplet | `step038_ant_water/` |
| 41 | `step041_ant_banana_leaf_tapping/` | the ant touching a banana leaf | `step040_ant_banana_leaf/` |
| 43 | `step043_ant_macaroni_tapping/` | the ant touching a macaroni | `step042_ant_macaroni/` |

### Dependencies — check before starting each one

`git pull` first, then:

- **Step 30** (`step030_ant_tapping/`, the sugar version) is the template for
  all six. It was still rendering on the mini when this was written. Start
  only once `git log --oneline | grep "Add step 30"` finds it. Read its
  Sources, Tests, Makefile and commit message closely: the tap motion, its
  cited antennation source and slow-down factor, how contact is re-solved each
  frame, how the molecule drifts into the pore, how the GIF was sized, and its
  mutants.
- **Steps 32, 34, 36** are already committed, so 33, 35, 37 can start as soon
  as step 30 lands.
- **Steps 38, 40, 42** are being built on the mini. At the time of writing
  step 38 was committed; check for "Add step 40" and "Add step 42" before
  starting 41 and 43.

## What each animation must do

Same cameras, composition, light and insets as its still. The touching
antenna's distal part lifts slightly and comes back down in a slow rhythmic
tap, exactly as step 30 does it, labelled "slowed ×N" with step 30's cited
real frequency. At each touch-down the tip meets the object exactly (contact
from geometry, as the stills measured it) and never penetrates it at any
frame. The insets follow the object's own science from its still:

- **33 salt** — on touch-down the taste hair meets the moisture film and an
  Na⁺/Cl⁻ pair leaves the crystal face into the film. No odour, ever (salt is
  not volatile — the still's test).
- **35 honey** — odour molecules keep drifting to the smell hair's wall
  pores; on touch-down the taste hair meets the honey surface and a sugar
  enters the taste pore. The drop's surface may dimple very slightly at
  contact (only if you can bound it honestly; otherwise keep it rigid).
- **37 yolk** — hexanal keeps arriving at the smell hair; on touch-down the
  taste hair meets the granular yolk surface.
- **39 water** — the taste hair meets the droplet's surface (water-repellent
  cuticle: a touch, not immersion); follow whatever the still chose for
  humidity sensing. No odour at the smell hair.
- **41 banana leaf** — the tip taps the waxy surface; follow the still's
  choice on volatiles (intact leaf: none or very few).
- **43 macaroni** — cheese odour keeps reaching the smell hair; on touch-down
  the taste hair meets the sauce.

Everything else stays still. **Forward loop, never a reverse rewind** — a
standing project rule: the tap is periodic so the loop closes naturally, and
anything that drifts (molecules, odour dots) must loop forward too, e.g. a new
one arriving each cycle.

## Rules (non-negotiable, all from Russell)

- **Copy, don't modify.** Never edit another step's folder. Each new step is
  its own self-contained folder (Sources, Tests with `Harness.swift`,
  Makefile, `.gitignore` containing `.build/`). Copy from the still and from
  step 30.
- **SDK on the laptop:** pass `SDK=$(xcrun --show-sdk-path)` to every make
  (see the note in `step019_tooth/Makefile`). On the mini it is a different
  path — don't copy that one.
- No macOS 15+ APIs. `options.fastMathEnabled = false` (its deprecation
  warning is expected). Long arithmetic in explicitly typed steps.
  `make typecheck` clean at `-warn-long-expression-type-checking=5` (only the
  fastMathEnabled warning allowed).
- **Every constant cited** (from a source you actually checked) or marked
  `MODEL` with a reason, in the house comment style — read step 20's or step
  26's comments to see it.
- **GIF size:** the writer is `../step008_dna/Sources/GIF.swift`, which stores
  ONE changed rectangle per frame, so the changed *bounding box* decides the
  size. The antenna and both insets change, so the box spans them. Measure 10
  frames first, then choose frame size, frame count and delay so each GIF is
  ≤ ~10 MB. Include frames from across the loop in the palette sample. The GIF
  goes to `stepNNN_…/renders/<name>.gif` and is copied to
  `showcase/<name>.gif`; `renders/` is gitignored, so only `showcase/` is
  committed.
- **Tests + mutants for every step:** keep the still's tests that apply, and
  add: touch-down frames touch (≈0 within tolerance), no penetration at any
  frame, lift frames clear the object; 12 antennal segments with connected
  joints every frame; periodic forward loop (frame N flows into frame 0);
  molecules keep their geometry (rigid motion only); the smell state matches
  the still (odour only for honey, yolk, macaroni); distance function honest
  outside surfaces. `make mutants`, each must make the suite FAIL: (1) the
  tip pushing into the object at touch-down; (2) the loop rewinding; (3) the
  wrong smell state (odour added where there is none, or removed where there
  is).
- Look at your frames yourself before calling a step done.

## Git

- One commit per step. `git pull --rebase` before every push; **never
  force-push**. Commit only your own files. The mini session is pushing to
  the same `main` at the same time, in other folders.
- End every commit message with exactly these two lines:

  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01Sx815ZFMDKuTuxMQYVftV1
  ```

- **Do not touch README.md.** The mini session verifies each step (fresh test
  run, mutants, a look at the GIF) and writes its README entry. Never rename
  folders — in particular never `step008a_plasmid`.

## Reporting

After each step, report to Russell plainly: commit hash, GIF path, frame size,
frame count, delay, loop length and file size (and why), render time, tests
and mutants, the slow-down factor, every new constant and its source (flag
anything unverified), judgement calls, and anything that didn't work.
