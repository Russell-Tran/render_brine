# Step 1 results: Hello, GPU

Build and run with `make run`, and run the tests with `make test`. On the M4 mini, the default SDK doesn't match the compiler, so add
`SDK=/Library/Developer/CommandLineTools/SDKs/MacOSX26.sdk`.

## 1. Predict first

Fill in the M3 Max column **before** running it there.

| Question | Your prediction for the M3 Max |
|---|---|
| GPU cores | |
| SIMD group width (≈ CUDA warp) | |
| GPU working-set limit, out of 48 GB | |
| Largest single buffer | |
| Threadgroup memory (≈ CUDA shared memory per block) | |
| Anything that will be the *same* as the M4 mini? | |

## 2. Results

| | M4 mini (2026-09-23) | M3 Max (2026-09-23) |
|---|---|---|
| CPU cores | 4 performance + 6 efficiency | 12 performance + 4 efficiency |
| GPU cores | 10 | 40 |
| Memory (shared by CPU and GPU) | 16.0 GB | 48.0 GB |
| Unified memory | yes | yes |
| GPU working-set limit | 11.8 GB (74% of RAM) | 36.0 GB (75% of RAM) |
| Largest single buffer | 8.9 GB | 27.0 GB |
| Threadgroup size limit per dimension (≈ CUDA block) | 1024 × 1024 × 1024 | 1024 × 1024 × 1024 |
| Threadgroup memory (≈ CUDA shared memory per block) | 32.0 KB | 32.0 KB |
| Apple GPU family | Apple 9 | Apple 9 |
| Ray tracing API available | yes | yes |
| SIMD group width (≈ CUDA warp) | 32 | 32 |
| Max threads per threadgroup for the probe kernel | 1024 | 1024 |

## 3. What surprised you?

Notes after comparing the two columns:

- **Same on both:** SIMD width 32, threadgroup limits, 32 KB threadgroup memory, and GPU family Apple 9. Metal puts both chips in the same GPU family, so a kernel sees the same limits on each.
- **What scales:** 4× the GPU cores (10 → 40) and 3× the memory (16 → 48 GB).
- **Same fractions:** the GPU may use about 75% of RAM on both, and the largest single buffer is about 56% of RAM on both.
- **Question for step 4:** with 4× the cores, will the water render run 4× faster?
