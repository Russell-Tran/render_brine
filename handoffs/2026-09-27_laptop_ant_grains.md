# Handoff: three ant-and-grain animations for the M3 Max laptop

Written 2026-09-27 by the Claude Code session on the M4 mini. Russell assigned
these three steps to **the laptop** ("Announcement to the M3 Max & M4 team").
The mini session will not build them. This brief adds to
[`2026-09-27_laptop_ant_animations.md`](2026-09-27_laptop_ant_animations.md).
Every rule in that brief (SDK flag, toolchain, citations, GIF sizing, tests,
mutants, git etiquette, never touching README.md) applies here unchanged. Read
it first.

## The job

| Step | Folder to create | Russell's words |
|---|---|---|
| 49 | `step049_ant_rice/` | animation of an ant interacting with a grain of rice |
| 50 | `step050_ant_corn/` | animation of an ant interacting with a corn kernel |
| 51 | `step051_ant_barley/` | animation of an ant interacting with a piece of cooked barley |

These differ from 33–43: **no still exists for them.** Each one is a new
object in the ant scene, animated from the start. Build each from:

- **`step030_ant_tapping/`**: the tap motion, contact re-solved every frame,
  the molecule entering the pore once per touch, the loop, and the mutants
  (`segments13`, `hover`, `tastePores`, `press`, `rewind`). It is committed
  and verified.
- **`step042_ant_macaroni/`**: how to stage a food object *larger than the
  ant*. The camera, the elbowed antenna reaching an overhanging surface, and
  the scale bar all carry over. A corn kernel especially will dwarf a 4 mm
  worker.
- The still steps 32–42 for the pattern of the two insets (the sense hairs at
  µm scale, the molecules at nm scale) and the smell-state test.

## Propose first

Russell's standing rule: **propose before building, and build only on his
explicit go-ahead.** These are new scenes, so show him a short proposal for
each (or one page for all three) and wait for his answer. Keep it simple; he
has said "the goal is simplicity". Put the proposal in an artifact, not
terminal text.

## Science to settle in the proposal (verify all of it; nothing here is checked)

The mini session wrote these as leads, not facts. Confirm or correct each
from a source you actually read, as steps 36 and 38 did when their briefs
were wrong.

- **Starch is the common thread.** All three grains are mostly starch, a
  glucose polymer. It is probably not tasted as sweet until it is broken down
  to sugars. Find what is measured for ants or insects and say only that. This
  links back to steps 7, 17 and 18, where glucose appears.
- **Smell state: decide per grain, from sources.** A raw polished white rice
  grain is plausibly near odourless (aromatic rices carry
  2-acetyl-1-pyrroline; plain white rice mostly doesn't). Cooked barley
  plausibly has a cooked-grain aroma. Whichever you find, the smell-state test
  and mutant follow it, as in steps 32 (none) and 34 (some).
- **Rice.** Grain size for a named type (e.g. long-grain white), taken from
  a source. The inset could show rice's small, angular compound starch
  granules. Raw or cooked is a proposal call for Russell; his words are just
  "a grain of rice".
- **Corn.** Field (dent) corn or sweet corn is a real choice. Sweet corn
  keeps sugar instead of turning it to starch (the *sugary* mutations), so
  its taste story differs. Put it to Russell. A kernel is roughly 2–3× the
  ant's length. Check this.
- **Cooked barley.** Pearled barley swells when cooked. Get the cooked size
  and the swelling from a source, or mark it MODEL. Barley's β-glucan could
  be the molecular inset. The surface is wet and soft, so decide honestly
  whether the tip touches a moisture film, as with honey and macaroni.

## Order

Russell hasn't said how 49–51 rank against the six already assigned (33,
35, 37, 39, 41, 43). Ask him when you send the proposal.

## Git

As in the first brief: one commit per step, `git pull --rebase` before every
push, never force-push, never touch README.md. End every commit message with:

```
Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Sx815ZFMDKuTuxMQYVftV1
```
