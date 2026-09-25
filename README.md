# render_brine

Learning Apple silicon GPUs with Metal, one small step at a time: from ray-traced water to scientifically accurate renders of the chemistry of life, building toward a brine shrimp.

Two Macs take part: an **M4 Mac mini** (10 GPU cores, 16 GB), where the code is written and first run, and an **M3 Max laptop** (40 GPU cores, 48 GB). Steps 1–5 ran on both.

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

Both images are 1920 × 1080. The ones shown were rendered on the M4 Mac mini; the M3 Max laptop produced the same gradient exactly and a water image differing in 73 of 2 million pixels, each by 1 level out of 255.

### Step 4: How fast the water renders

The step 3 water render, timed on the GPU's own clock (median of 20 runs) on both Macs. Code: [`step4_timing/`](step4_timing/)

| Size | Pixels | M4 Mac mini (10 GPU cores) | M3 Max laptop (40 GPU cores) | Speedup |
|---|---|---|---|---|
| 256 × 144 | 36,864 | 0.013 ms | 0.010 ms | 1.3× |
| 1080p | 2,073,600 | 0.49 ms | 0.151 ms | 3.2× |
| 4K | 8,294,400 | 1.9 ms | 0.563 ms | 3.4× |

**4× the cores gave about 3.3×, not 4×.** The chips also differ in GPU clock speed and core generation, so core count alone doesn't set the speed. Small jobs barely speed up at all: a tiny image can't keep either GPU busy.

On the mini, 4K has 4× the pixels of 1080p and took 4× the time. Getting steady numbers first required warming the GPU up, because it runs slowly until it's been kept busy for about 200 ms:

![Line chart: a 4K frame takes about 3.7 to 4.5 ms right after the GPU has been idle, then drops to about 1.9 ms after roughly 200 ms of continuous rendering](showcase/warmup.svg)

### Step 5: The sea mirrors the sky

![The same sea as step 3, now reflecting the pale sky, brighter toward the horizon, with sharp glints from the sun's reflection](showcase/reflections.png)

Each water pixel now bounces its ray off the waves and looks up the sky in that direction. How much it reflects depends on the angle (the Fresnel effect): about 2% looking straight down, nearly everything at a glancing angle. Each pixel also averages 16 samples, which turns the sparkly noise near the horizon into smooth ripples. Code: [`step5_reflections/`](step5_reflections/)

| 1 sample per pixel | 16 samples per pixel |
|---|---|
| ![Close-up with sparkly, noisy ripples near the horizon](showcase/reflections_crop_1spp.png) | ![The same close-up with smooth ripples](showcase/reflections_crop_16spp.png) |

1080p, median GPU time of 10 runs:

| Samples per pixel | M4 Mac mini | M3 Max laptop | Speedup |
|---|---|---|---|
| 1 | 0.55 ms | 0.169 ms | 3.3× |
| 4 | 1.78 ms | 0.545 ms | 3.3× |
| 16 | 6.61 ms | 2.038 ms | 3.2× |

On both Macs, 16 samples per pixel took only 12× the time of 1 sample, because part of each pixel's cost doesn't grow with its sample count.

### Step 6a: One carbonic acid molecule splits

![Ball-and-stick animation: carbonic acid loses a proton (gold, H⁺) and becomes bicarbonate, whose two free oxygens end up with equal bonds and share the negative charge](showcase/molecule.gif)

H₂CO₃ → H⁺ + HCO₃⁻, ray-traced on the GPU with real bond lengths. Code: [`step6_carbonic_acid/step6a_carbonic_acid/molecule/`](step6_carbonic_acid/step6a_carbonic_acid/molecule/)

### Step 6b: The carbonic acid journey

![Looping ball-and-stick animation: CO₂ and water become carbonic acid with a helper water relaying the proton, then bicarbonate and hydronium, while fresh molecules keep arriving](showcase/journey.gif)

CO₂ + H₂O → H₂CO₃ → HCO₃⁻ + H₃O⁺, one molecule at a time. Code: [`step6_carbonic_acid/step6b_carbonic_journey/`](step6_carbonic_acid/step6b_carbonic_journey/)

### Step 7: Glucose meets oxygen

![Ball-and-stick animation: one glucose molecule and six O₂ molecules rearrange into six CO₂ and six H₂O](showcase/respiration.gif)

C₆H₁₂O₆ + 6 O₂ → 6 CO₂ + 6 H₂O, the overall accounting of cellular respiration: 24 electrons move from carbon to oxygen. Code: [`step7_glucose/step7_respiration/`](step7_glucose/step7_respiration/)

### Step 7a: Glycolysis: spend two, earn four

![Looping ball-and-stick animation: glucose gets two phosphates from two ATP, then splits between carbons 3 and 4 into two glyceraldehyde-3-phosphates](showcase/glycolysis_spend.gif)

![Looping ball-and-stick animation: two glyceraldehyde-3-phosphates side by side make 2 NADH and 4 ATP on the way to two pyruvates](showcase/glycolysis_payoff.gif)

glucose + 2 NAD⁺ + 2 ADP + 2 Pᵢ → 2 pyruvate + 2 NADH + 2 H⁺ + 2 ATP + 2 H₂O. Code: [`step7_glucose/step7a_glycolysis/`](step7_glucose/step7a_glycolysis/)

### Step 8: DNA, slowly turning

![Looping ball-and-stick animation: a DNA double helix lying diagonally turns once about its own axis, with grey carbons, blue nitrogens, red oxygens, orange phosphorus and faint dashed hydrogen bonds between the paired bases](showcase/dna.gif)

The Dickerson–Drew dodecamer, CGCGAATTCGCG, from its X-ray crystal structure ([PDB 1BNA](https://www.rcsb.org/structure/1BNA)): 758 atoms, right-handed, 10.1 base pairs per turn. Code: [`step8_dna/`](step8_dna/)

### Step 8a: pGLO, whole and up close

![Looping animation: the pGLO plasmid as a coloured ring turning once, with green GFP, violet arabinose switch, amber ampicillin resistance and blue origin against grey for the rest](showcase/plasmid_ring.gif)

![Looping animation: a continuous 150× zoom from the whole plasmid ring into the first atoms of the GFP gene, passing through a tube, then space-filling spheres, then ball-and-stick](showcase/plasmid_dive.gif)

The plasmid that makes *E. coli* glow, at its real size: 5,371 base pairs, 1.83 µm around, 512 turns of double helix. The dive changes how it draws the molecule twice on the way down, each time at the distance where the finer detail stops being smaller than a pixel. First acceleration structure in the series: a uniform grid, 369× faster than testing every ray against every shape. Code: [`step8a_plasmid/`](step8a_plasmid/)

### Step 9: The protein that makes its own light

![Looping animation: green fluorescent protein drawn as overlapping van der Waals spheres turns about its axis; a round window opens in the front to reveal the chromophore in ball-and-stick inside, then closes](showcase/gfp.gif)

GFP from its crystal structure ([PDB 1EMA](https://www.rcsb.org/structure/1EMA)), space-filling: every atom a sphere at its full van der Waals radius. The chromophore inside is not a cofactor — the protein builds it out of three consecutive amino acids of its own chain, and the barrel exists to hold it rigid and keep water away. First render with ambient occlusion, which is what makes a space-filling surface readable at all. Code: [`step9_gfp/`](step9_gfp/)

### Step 10: Getting a plasmid into a bacterium

![Looping animation: a supercoiled plasmid, drawn as a branched interwound coil with calcium ions around it, drifts down onto a cross-section of the E. coli envelope — an outer membrane of lipids, a peptidoglycan mesh, and an inner membrane below](showcase/transformation_approach.gif)

![Looping animation: three panels of the same membrane side by side — CaCl₂ and heat shock labelled MODEL, electroporation labelled SIMULATED, natural competence labelled MEASURED with a protein spanning the membrane](showcase/transformation_routes.gif)

Bacterial transformation, and the first render here whose central event has never been observed: the CaCl₂ and heat-shock method dates to 1970 and its molecular mechanism is still not established. So every frame carries an **evidence bar** saying how well the thing on screen is actually known — measured, simulated, or model — and the tests enforce it, failing if the chemical route ever claims more evidence than it has. 407,000 spheres at 20 ms a frame, with the grid rebuilt every frame. Code: [`step10_transformation/`](step10_transformation/)

### Step 11: Arabinose, the sugar that switches on pGLO's gene in E. coli

![Looping animation: a loop of DNA held shut by a protein bridging two distant sites. A small sugar arrives and binds it, the grip moves along the DNA, the loop springs open, and a shape settles onto the newly exposed promoter — then the sugar leaves and the loop re-forms](showcase/switch.gif)

How pGLO's GFP gene gets switched on. AraC holds *araO2* and *araI1* — 210 base pairs apart — at the same time, tying the DNA in a loop that blocks RNA polymerase. Arabinose binds, AraC's grip moves to the adjacent site, the loop opens, and the gene can be read. The two half-sites occur in pGLO verbatim, and araO2 placed by its published offset lands exactly 210 bp away, independently. First scene here that genuinely deforms, so the acceleration grid is rebuilt every frame — which costs only 7.7% of it. Code: [`step11_switch/`](step11_switch/)
