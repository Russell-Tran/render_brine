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
