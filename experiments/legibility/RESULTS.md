# Legibility experiment: results

Pre-registration: [`PREREGISTRATION.md`](PREREGISTRATION.md), committed at 5856d60
(2026-09-28 19:15:57 -0700), before any viewer looked. Everything in part 1 follows that
file's rules. Parts 2 and 3 were **not** pre-registered and are labelled as such.
Part 4 is still waiting on Russell's ratings.

## 1. Pre-registered result: the hypothesis is **rejected**

| | Change score 2 (got it) | 1 (partial) | 0 (missed) |
|---|---|---|---|
| Animations (52) | **52** | 0 | 0 |
| Stills (23) | 21 | 2 | 0 |

The decision rule said **rejected if ≥ 90% of animations score 2**. The result is 100%.
Every item also scored 1 on subject and 1 on where. The two partial stills:

- **Step 5**, the sea reflecting the sky: the viewer saw the sun's glint path but not the sky's reflection.
- **Step 20**, the lower arch: the viewer took *crowded, out-of-line front teeth* as the point. That's worth a look as a real visual flaw.

**My predictions did badly.** I predicted 15 animations would fail, and none did (0 of 15).
I predicted still 38, the colourless drop, would fail, and it was read correctly. The toy
walks, which I expected to be too small to read, were described clearly: "the pig walks
around in a loop on the floor".

## 2. Exploratory, not pre-registered: *how* the tapping loops were understood

On every ant tapping loop (steps 30, 33, 35, 37, 39, 41, 43, 49, 50, 51), the viewer
said the tap **itself** was barely visible or invisible in the main view, then read the
change from the **inset** and its labels:

> "The main ant hardly moves. The antenna tip on the crystal shifts only a tiny amount,
> if at all. The change shows up in the upper-right inset…" (step 30)
>
> "The ant's body doesn't move, and its antenna tip barely moves at full size. The big
> middle inset alternates… ('↓ juice 53 µm away')." (step 50)

So my prediction about the *physical motion at true scale* was right: it isn't legible.
But the renders as a whole were legible, because the **annotation layer** (insets, labels,
captions, zoom circles) carried the meaning. I'm reporting this as an observation. It
doesn't change the verdict in part 1.

## 3. Positive controls, added after the fact

**A design flaw I missed:** the two cases that motivated the hypothesis, step 29's
invisible pollen tube and step 53's bronze lock, had both been **fixed before** the
experiment, so the corpus contained no render already known to fail a person. To check
that the viewer *can* fail something, I ran the **original** versions from git history
(29 at 39b65fe, 53 at d8aee49) past a fresh viewer and a fresh grader, keyed by the alt
text each had when first published.

| Control | Viewer said | Change score |
|---|---|---|
| Original step 29 | "It looks the same length in frames 2 through 5, so I cannot see it grow… a faint yellow line… subtle and hard to follow at this size." | **1 (partial)** |
| Original step 53 | "It explains why the lock looks gold…" | **2** |

**Disclosure:** the control grader's rubric had one clause the main grader's did not.
A viewer who "reports the key's main change as too subtle to follow" scores 1. It's a
reasonable reading of "vague", but it isn't identical.

What the controls suggest, as an observation only:

- The viewer **can** detect an animation whose change is too small, the problem Russell
  hit with step 29, though it scored partial, not a clean fail.
- Step 53's problem was never legibility. The viewer got its point. It was **how it
  looked** (bronze, not gold). A reader-for-meaning won't catch that kind of failure.

## 4. Validation against Russell: *pending*

Russell rates 10 items (seed 20260929) on the real GIFs and stills. If his ratings and
the grader's disagree on more than 3 of the 10 (a fail/pass mismatch on the change
question), the conclusions above are marked unreliable.

## What I now think

The pre-registered hypothesis, "correct renders are often illegible", is **wrong for
this corpus**. The renders are legible, largely because every step carries insets and
labels that explain what's happening. The failures Russell caught were of two different
kinds: one real legibility failure (29, since fixed), which the viewer partly detects,
and one **aesthetic** failure (53), which it can't detect at all. If there's a follow-up
hypothesis, it's about beauty rather than legibility, and it would need a different
instrument: people, or a viewer asked "does this look right?" rather than "what does this show?"
