# Laptop vs mini: the recorded world rendered on two GPUs

Frames rendered from `records/world79.rec` at 480×270 by `.build/render refs`:
`records/refs/` on the M3 Max laptop (macOS 14.5), `records/refs_mini/` on the M4 mini
(macOS 26.6.2; compiler in commit e8e2f80). |diff| is the absolute difference per channel, out of 255.

| frame | max R G B | mean | pixels ≠ | > 2/255 | > 4/255 | ≥ 8/255 |
|---|---|---|---|---|---|---|
| 150  | 1 1 1  | 0.00005 | 10 | 0 | 0 | 0 |
| 700  | 1 1 1  | 0.00007 | 15 | 0 | 0 | 0 |
| 1431 | 1 1 1  | 0.00006 | 11 | 0 | 0 | 0 |
| 1799 | 8 7 10 | 0.00021 | 29 | 2 | 2 | 2 |

The two outliers in frame 1799 are single pixels, (415, 110) off by 10 and (432, 96) off by 8,
both **inside the sugar pile**, where rays refract and reflect inside the sucrose crystals and
last-bit GPU differences are amplified. None is on an ant, a caption or the pheromone field.
`1799_diff.png` shows the frame darkened with |diff| × 25 in red.

The test's tolerance (mean < 0.5/255, max < 8/255) was stated before the mini rendered, and it
stays: frame 1799 fails it on the mini, honestly. P5's cross-machine claim is partly right: the
recorded world renders the same on both GPUs to 1/255 almost everywhere, with rare single-pixel
outliers (here in refracting crystals) up to 10/255. Record the world; expect near-identical,
not bit-identical, renders across GPUs. A rule for future multi-machine films is in lib/README.
