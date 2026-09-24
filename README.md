# render_brine

Learning to render water on Apple silicon GPUs with Metal, one small step at a time.

## Showcase

### Sine-wave ocean with sun (step 3)

![Open sea under a pale sky, with a sun and a path of glints on the waves leading to the horizon](showcase/water.png)

Each of the 2,073,600 pixels gets its own GPU thread. The thread adds up five sine waves to find which way the sea's surface tilts at that spot, then shades it toward the sun, from deep blue to turquoise with white glints. Code: [`step3_water/`](step3_water/)

### Gradient painted on the GPU (step 2)

![A vertical gradient from sky blue at the top to deep navy at the bottom](showcase/gradient.png)

The first image painted on the GPU: one thread per pixel, written into memory the CPU and GPU share, so saving it needed no copying. Code: [`step2_paint_gpu/`](step2_paint_gpu/)

Both images are 1920 × 1080, rendered on an Apple M4 Mac mini.

### What the two GPUs report (step 1)

The first program asks each Mac's GPU what it can do. The two chips share the same GPU design, and the M3 Max has more of it. Code: [`step1_hello_gpu/`](step1_hello_gpu/)

| | M4 Mac mini | M3 Max |
|---|---|---|
| GPU cores | 10 | 40 |
| Memory, shared by CPU and GPU | 16 GB | 48 GB |
| Memory the GPU may use | 11.8 GB (74%) | 36.0 GB (75%) |
| SIMD group width (≈ CUDA warp) | 32 | 32 |
| Threadgroup memory (≈ CUDA shared memory) | 32 KB | 32 KB |
| Apple GPU family | Apple 9 | Apple 9 |

### How fast the water renders (step 4)

The step 3 water render, timed on the GPU's own clock (median of 20 runs). Code: [`step4_timing/`](step4_timing/)

| Size | Pixels | M4 mini | M3 Max |
|---|---|---|---|
| 256 × 144 | 36,864 | 0.013 ms | not run yet |
| 1080p | 2,073,600 | 0.49 ms | not run yet |
| 4K | 8,294,400 | 1.9 ms | not run yet |

On the mini, 4K has 4× the pixels of 1080p and took 4× the time. Getting steady numbers first required warming the GPU up, because it runs slowly until it's been kept busy for about 200 ms:

![Line chart: a 4K frame takes about 3.7 to 4.5 ms right after the GPU has been idle, then drops to about 1.9 ms after roughly 200 ms of continuous rendering](showcase/warmup.svg)
