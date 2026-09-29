# Handoff: three real, extremely simplified boats (steps 76–78) for the M3 Max laptop

Written 2026-09-28 by the M4 mini session. Russell assigned these ("Subagents/laptop"),
and they're **pre-approved**, so don't wait for a go-ahead. Do your own research first
and open each commit message with "Found and decided:". Every rule in
[`2026-09-27_laptop_ant_animations.md`](2026-09-27_laptop_ant_animations.md) applies:
the SDK flag, citations or MODEL, tests and mutants, honest distance functions, git
etiquette, and never touching README.md. The toy versions (73–75) are being built on
the mini. Don't build them.

| Step | Folder | Russell's words |
|---|---|---|
| 76 | `step076_rowboat/` | real but extremely simplified rowboat |
| 77 | `step077_sailboat/` | real but extremely simplified sailboat |
| 78 | `step078_polynesian_canoe/` | real but extremely simplified Polynesian boat |

All three are **stills**, 1920×1080. "Extremely simplified" means few, clean shapes, but
**true proportions** from a real, named boat type with sourced dimensions. It's a
diagram-like render, not a detailed model.

## The one physics idea for all three: the waterline is computed, not placed

Each boat floats where Archimedes puts it. The hull sinks until the water it displaces
weighs as much as the boat plus its load. Source or state the boat's mass (hull
material and weight from a real design, or MODEL with a reason) and seawater or fresh
water density (cite). Compute the draft numerically from the hull's own distance
function or its shape. **Test it:** the displaced volume × water density equals the
total mass, to a stated tolerance. **Mutant `floatsWrong`** (the waterline placed by
eye, e.g. 30% too deep) must FAIL. Show the water as a simple flat surface, a cut-away,
or a translucent plane, whichever makes the waterline read, and label the draft.

## Each boat (leads, not facts: verify everything)

- **76, rowboat.** A small wooden rowing boat, e.g. a classic ~12–14 ft dinghy or
  skiff. Hull, thwarts (seats), oarlocks, and a pair of oars at rest. One rower's
  weight can be the load (cite an average adult mass) or left out; say which.
- **77, sailboat.** A small single-masted dinghy or sloop: hull, mast, boom, a
  triangular mainsail (optionally a jib), and a centreboard or keel below the
  waterline. The underwater part is where the physics is, so the cut-away or
  translucent water should show the keel/centreboard. Sail area and mast height from
  a named class, if one can be sourced.
- **78, Polynesian boat.** The iconic form is the **double-hulled voyaging canoe**
  (e.g. Hawaiian *waʻa kaulua*): two hulls joined by crossbeams and a deck, with
  **crab-claw (oceanic lateen) sails**. The best-documented modern example is
  *Hōkūleʻa* (Polynesian Voyaging Society; ~62 ft); check its published dimensions.
  An outrigger canoe (*vaʻa*) is the other option. Pick one and say why.
  **Be respectful and accurate:** name the tradition correctly, avoid caricature, and
  caption it as a simplified depiction informed by published replicas and scholarship,
  not a specific sacred vessel. The waterline test applies to both hulls together.

## Tests (each step)

Dimensions match the sourced type within tolerance. The waterline comes from buoyancy
(above). Distance functions are honest outside every surface (only outside points
count; blend by per-object constant bounds, never a local slope). Scale bar true. The
picture shows the boat, the water and the draft label. Mutants: `floatsWrong`,
`wrongSize` (20% off), plus one per boat, e.g. `noKeel` (77), `singleHull` (78, if
double-hulled), `noOarlocks` (76).

## Typecheck

The mini's compiler is stricter about slow expressions than Swift 5.10. From the last
batch, these patterns hit 5 ms there: several `n * k` or `Float(...)` products inline;
a conversion wrapping a call inside a comparison; a long array literal of untyped
`SIMD3(...)`. Type each scalar first.

## Git

One commit per step. Run `git pull --rebase` before every push, never force-push, and
don't touch README.md. Ping the mini session ("Render Brine") as each step is pushed.
End every commit message with:

```
Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Sx815ZFMDKuTuxMQYVftV1
```
