# Legibility experiment: pre-registration

Written and committed on 2026-09-28, **before any viewer has looked at any item**.
The git timestamp on this file is the proof of order.

Russell's plain-language version of the hypothesis is on the proposal page:
https://claude.ai/artifact/Wekfw8r6m48urz1vzK3F6o

## Hypothesis

Our tests grade whether a render is **correct**. Nothing grades whether a person
can **see** what it shows. These are independent: a render can pass every
correctness test and still fail to show its point to a viewer.

Every item in the corpus passes its full correctness suite and all its mutants.
So any item that a fresh viewer can't read is direct evidence that correctness
doesn't guarantee legibility.

## Corpus

`corpus.json`: 75 items, one per README step that has a raster image (step 1 has
no image; step 4's is an SVG chart). There are 52 animations and 23 stills.
Order is shuffled with seed 20260928, and items are renamed `item01`…`item75` so
file names give nothing away. Each animation is shown as **6 frames evenly
spaced across its loop, in time order**. Each still is shown as its single image.

## The answer key

Each item's key is its **README title and image alt text**. Both were written when
the step was published, before this hypothesis existed, so they can't be
tuned to the result. The alt text describes what the render shows and, for
animations, what moves.

## Viewers

Fresh agents with **no project context**. Each sees only its items' frames and
answers three questions per item:

1. **What is this?** (the subject)
2. For an animation, **what changes across the frames?** For a still, **what
   is the main point being shown?**
3. **Where should a viewer look?** (region of the picture)

Viewers are told not to open any other file or use the web. About 10 items go
to each viewer, mixed across families.

## Grading (blind to the predictions below)

A separate grader agent scores each answer against its key. It never sees this
file's predictions.

- **subject**: 1 if the main subject matches the key's, 0 if not.
- **change / point**: 2 if it names the specific change or point the key
  describes, 1 if it notices something but misidentifies it or stays vague, 0 if
  it says nothing changes or is wrong.
- **where**: 1 if the region named contains the key feature, 0 if not.

An item **fails** if its change/point score is 0. A score of 1 counts as
**partial**.

## Decision rules

- **Confirmed** if **≥ 20% of animations fail** (change score 0).
- **Rejected** if **≥ 90% of animations score 2**.
- Anything in between is **inconclusive**, and is reported as such.

## Validation against a person

Russell rates 10 items chosen by seed 20260929 from the corpus, watching the real
GIF or still, on the same three questions. Agreement between the grader's scores
and Russell's is reported. If they disagree on more than 3 of the 10, the
viewer is a poor stand-in for a person, and every conclusion above is marked
unreliable.

**Known bias:** Russell has seen every render before, so he isn't a stranger, and
he'll tend to find them *more* legible. If even he finds an item hard to read,
that's strong evidence.

## My predictions (the viewer and grader never see these)

**Animations predicted to fail (15 of 52, 29%):**
- The ant tapping loops, where the antenna lift is ~6–14 px: steps **30, 33,
  35, 37, 39, 41, 43, 49, 50, 51**
- The apple steps: **68** (the tap cleared the pixel test by 0.4 px) and **69**
  (the bite inset doesn't show contact)
- The toy walks, small in empty frames: **59, 61, 63**

**Stills predicted to fail:** **38** (the water drop is honestly colourless and
nearly invisible).

**Predicted to pass:** the camera pans (21, 67), the trails (44, 55, 56, 57), the
plant growth loops (29 after its fix, 45–48, 54), the neuron firing (31) and
bruxism's progression (65).

I'll report the hit rate of these predictions honestly, including misses in
both directions.

## Known limits, stated in advance

- Six frames aren't watching the GIF. Slow or small motion may be harder to
  judge from frames than live. That's one reason Russell's ratings, made on
  the real GIFs, are the check.
- Captions and labels are part of each render and are left in. A viewer may
  read the answer from a caption. That's fair: captions are part of what a
  person sees.
- The viewer is a model. The validation step exists because of that.
