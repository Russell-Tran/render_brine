# Step 5 results: the sea mirrors the sky

Run with `make run`, and run the tests with `make test`. On the M4 mini, add
`SDK=/Library/Developer/CommandLineTools/SDKs/MacOSX26.sdk`.

`make run` saves `renders/reflections.png` (16 samples per pixel) and
`renders/reflections_1spp.png` (1 sample per pixel), then times 1080p at 1, 4
and 16 samples per pixel (median of 10 runs after a 500 ms warm-up).

## Timing

| Samples per pixel | M4 mini GPU time | vs 1 sample | M3 Max GPU time | vs 1 sample |
|---|---|---|---|---|
| 1 | 0.55 ms | 1.0× | | |
| 4 | 1.78 ms | 3.2× | | |
| 16 | 6.61 ms | 11.9× | | |

M4 mini, 2026-09-24. For comparison, step 3's render (no reflections, one
sample) took 0.49 ms at 1080p, so reflections cost about 13% more.

## Notes

- **16× the samples took only 12× the time.** Each pixel has a fixed cost that
  extra samples don't repeat. The check: the same 8.3 million samples took
  1.78 ms as 1080p × 4 samples, but 2.17 ms as 4K × 1 sample (4× the pixels).
  Likely causes: starting each thread, and writing each finished pixel to
  memory (4K writes 33 MB of pixels to 1080p's 8 MB).
- **Why more samples matter:** at 1 sample per pixel, waves smaller than a
  pixel near the horizon turn into scattered sparkles. At 16 they average into
  smooth ripples.
- **GPU vs CPU reference:** worst difference 1 level out of 255.
