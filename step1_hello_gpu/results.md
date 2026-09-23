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

| | M4 mini (2026-09-23) | M3 Max |
|---|---|---|
| CPU cores | 4 performance + 6 efficiency | |
| GPU cores | 10 | |
| Memory (shared by CPU and GPU) | 16.0 GB | 48.0 GB |
| Unified memory | yes | |
| GPU working-set limit | 11.8 GB (74% of RAM) | |
| Largest single buffer | 8.9 GB | |
| Threadgroup size limit per dimension (≈ CUDA block) | 1024 × 1024 × 1024 | |
| Threadgroup memory (≈ CUDA shared memory per block) | 32.0 KB | |
| Apple GPU family | Apple 9 | |
| Ray tracing API available | yes | |
| SIMD group width (≈ CUDA warp) | 32 | |
| Max threads per threadgroup for the probe kernel | 1024 | |

## 3. What surprised you?

Notes after comparing the two columns:

-
