# Step 4 results: time it on both chips

Run with `make run`, and run the tests with `make test`. On the M4 mini, add
`SDK=/Library/Developer/CommandLineTools/SDKs/MacOSX26.sdk`.

This times the step 3 water render. Before timing, the GPU renders
continuously for 500 ms, and each size then gets 20 timed runs. Times are
medians, by the GPU's own clock.

## 1. Predict first

The M3 Max has 40 GPU cores to the M4 mini's 10. Fill this in **before**
running it on the M3 Max.

| Size | M4 mini GPU time | Your guess for the M3 Max | Speedup you expect |
|---|---|---|---|
| tiny (256 × 144) | 0.013 ms | | |
| 1080p (1920 × 1080) | 0.49 ms | | |
| 4K (3840 × 2160) | 1.9 ms | | |

## 2. Results

Medians of three separate runs on each machine.

| Size | M4 mini GPU time | M4 mini wall time | M3 Max GPU time | M3 Max wall time | Speedup (GPU time) |
|---|---|---|---|---|---|
| tiny | 0.013 ms | 0.19 ms | | | |
| 1080p | 0.49 ms | 0.74 ms | | | |
| 4K | 1.9 ms | 2.2 ms | | | |

M4 mini, 2026-09-23: 4K vs 1080p was 4.0× the pixels in 3.9–4.0× the time.

## 3. Notes

- **Warm-up matters.** From idle, a 4K frame on the mini took ~3.7 ms; after
  ~200 ms of steady rendering it took ~1.9 ms, because the GPU raises its clock
  speed only once it's kept busy. With just 5 warm-up runs, two back-to-back
  benchmarks disagreed by 2×. Warming up for 500 ms fixed that.
- **Big images scale perfectly on the mini.** 4× the pixels took 4× the time,
  so the GPU was fully busy.
- **Tiny images are all overhead.** The GPU finished in 0.013 ms, but sending
  the work and waiting for it back took 0.19 ms.
-
