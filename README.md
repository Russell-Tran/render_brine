# render_brine

Learning to render water on Apple silicon GPUs with Metal, one small step at a time.

## Showcase

### Step 1: What the two GPUs report

The first program asks each Mac's GPU what it can do. The two chips share the same GPU design, and the M3 Max has more of it. Code: [`step1_hello_gpu/`](step1_hello_gpu/)

| | M4 Mac mini | M3 Max |
|---|---|---|
| GPU cores | 10 | 40 |
| Memory, shared by CPU and GPU | 16 GB | 48 GB |
| Memory the GPU may use | 11.8 GB (74%) | 36.0 GB (75%) |
| SIMD group width (≈ CUDA warp) | 32 | 32 |
| Threadgroup memory (≈ CUDA shared memory) | 32 KB | 32 KB |
| Apple GPU family | Apple 9 | Apple 9 |

### Step 2: Gradient painted on the GPU

![A vertical gradient from sky blue at the top to deep navy at the bottom](showcase/gradient.png)

The first image painted on the GPU: one thread per pixel, written into memory the CPU and GPU share, so saving it needed no copying. Code: [`step2_paint_gpu/`](step2_paint_gpu/)

### Step 3: Sine-wave ocean with sun

![Open sea under a pale sky, with a sun and a path of glints on the waves leading to the horizon](showcase/water.png)

Each of the 2,073,600 pixels gets its own GPU thread. The thread adds up five sine waves to find which way the sea's surface tilts at that spot, then shades it toward the sun, from deep blue to turquoise with white glints. Code: [`step3_water/`](step3_water/)

Both images are 1920 × 1080, rendered on an Apple M4 Mac mini.

### Step 4: How fast the water renders

The step 3 water render, timed on the GPU's own clock (median of 20 runs). Code: [`step4_timing/`](step4_timing/)

| Size | Pixels | M4 mini | M3 Max |
|---|---|---|---|
| 256 × 144 | 36,864 | 0.013 ms | not run yet |
| 1080p | 2,073,600 | 0.49 ms | not run yet |
| 4K | 8,294,400 | 1.9 ms | not run yet |

On the mini, 4K has 4× the pixels of 1080p and took 4× the time. Getting steady numbers first required warming the GPU up, because it runs slowly until it's been kept busy for about 200 ms:

![Line chart: a 4K frame takes about 3.7 to 4.5 ms right after the GPU has been idle, then drops to about 1.9 ms after roughly 200 ms of continuous rendering](showcase/warmup.svg)

### Step 5: The sea mirrors the sky

![The same sea as step 3, now reflecting the pale sky, brighter toward the horizon, with sharp glints from the sun's reflection](showcase/reflections.png)

Each water pixel now bounces its ray off the waves and looks up the sky in that direction. How much it reflects depends on the angle (the Fresnel effect): about 2% looking straight down, nearly everything at a glancing angle. Each pixel also averages 16 samples, which turns the sparkly noise near the horizon into smooth ripples. Code: [`step5_reflections/`](step5_reflections/)

| 1 sample per pixel | 16 samples per pixel |
|---|---|
| ![Close-up with sparkly, noisy ripples near the horizon](showcase/reflections_crop_1spp.png) | ![The same close-up with smooth ripples](showcase/reflections_crop_16spp.png) |

On the M4 mini at 1080p, 16 samples per pixel took 6.6 ms, only 12× the time of 1 sample, because part of each pixel's cost doesn't grow with its sample count.
