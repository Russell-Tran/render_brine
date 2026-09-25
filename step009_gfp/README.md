# Step 9: Green fluorescent protein, space-filling

GFP from its crystal structure ([PDB 1EMA](https://www.rcsb.org/structure/1EMA)),
drawn space-filling — every atom a sphere at its van der Waals radius — turning
once about the barrel's axis, with a window opening in the front to show the
chromophore inside.

New technique: **ambient occlusion**. Every step up to now cast one ray per
sample and stopped. Here each ray that lands on the molecule casts a further
spray of short probe rays and counts how many escape, so crevices darken.
Without it a space-filling protein is a flat coloured blob.

## Rendering this on the M3 Max

The code is written and tested on the M4 mini, but it builds and runs on the
laptop's older toolchain too: no macOS 15+ APIs, `fastMathEnabled = false`, and
the slow-expression scan is clean.

```sh
git pull
cd step9_gfp

# The laptop needs its own SDK path. Find it with:
xcrun --show-sdk-path
# then pass it to make, for example:
make test SDK=$(xcrun --show-sdk-path)
make run  SDK=$(xcrun --show-sdk-path)
```

`make run` prints the GPU it found and the milliseconds per frame, which is the
number worth reporting back.

**Is the laptop actually needed here?** No — and the measurement says so
plainly. The full loop is **63 seconds of GPU time on the M4 mini**, so the
laptop would bring it to roughly 20 seconds. Both are fine. Ambient occlusion
costs about 8× (50 ms a frame without it, 315 ms with it at 12 probes), which
is a large multiplier on a small number. Where the laptop will genuinely matter
is the ribosome: about 147,000 atoms against this molecule's 1,771, with the
same 8× on top.

To compare the two machines on the same work, run this on both and note the
milliseconds per frame:

```sh
GFP_FRAMES=10 make run SDK=$(xcrun --show-sdk-path)
```

## Knobs

Set these in the environment; no recompile needed.

| Variable | Default | What it does |
|---|---|---|
| `GFP_FRAMES` | all | Render only the first N frames |
| `GFP_AT` | – | Render one specific frame, for looking at a moment |
| `GFP_AO` | `1` | `0` turns ambient occlusion off |
| `GFP_PROBES` | `12` | Probe rays per hit |
| `GFP_AO_DIST` | `8.0` | How far a probe ray looks, in ångströms |
| `GFP_AO_POW` | `1.7` | Contrast curve on the occlusion term |
| `GFP_AO_ONLY` | `0` | `1` shows the occlusion term by itself, in grey |

## Targets

```sh
make run        # render renders/gfp.gif
make test       # 34 tests
make structure  # rebuild Resources/gfp.json from Resources/1EMA.pdb
```

## What the structure turned out to be

Checked against the file rather than assumed, and three of these differ from
what I expected going in:

- **1,771 heavy atoms**, not the 1,866 the PDB entry reports — that figure
  counts the 95 crystal waters, which are left out here.
- **226 residues, numbered 2–229, with gaps at 65 and 67.** The gaps are the
  whole story: Thr65, Tyr66 and Gly67 fuse into one residue, `CRO 66`, which is
  the chromophore. GFP builds its own light-emitting group out of its own
  chain, with no enzyme and no cofactor, and the barrel exists to hold it rigid
  and keep water away from it.
- **Four selenomethionines** (78, 88, 153, 218), holding selenium rather than
  sulfur. That is a crystallographer's trick for solving the phase problem, not
  a feature of the real protein.
- **Eleven β-strands**, straight from the file's `SHEET` records. Their wall is
  23.1 Å across and its radius varies by only ±1.4 Å, which is what makes it a
  barrel rather than a bundle. The chromophore sits 2.1 Å from the centre.

No hydrogens are added, unlike step 8: X-rays at 1.9 Å cannot see them, nothing
here depends on them, and the ~1,800 extra spheres would double the cost of the
occlusion that makes the shape readable.
