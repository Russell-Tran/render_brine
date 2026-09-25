# render_brine

Learning Apple silicon GPUs with Metal, one small step at a time: from ray-traced water to scientifically accurate renders of the chemistry of life, building toward a brine shrimp.

Two Macs take part: an **M4 Mac mini** (10 GPU cores, 16 GB), where the code is written and first run, and an **M3 Max laptop** (40 GPU cores, 48 GB). Steps 1–5 ran on both.

## Showcase

### Step 1: What the two GPUs report

The first program asks each Mac's GPU what it can do. The two chips share the same GPU design, and the M3 Max has more of it. Code: [`step001_hello_gpu/`](step001_hello_gpu/)

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

The first image painted on the GPU: one thread per pixel, written into memory the CPU and GPU share, so saving it needed no copying. Code: [`step002_paint_gpu/`](step002_paint_gpu/)

### Step 3: Sine-wave ocean with sun

![Open sea under a pale sky, with a sun and a path of glints on the waves leading to the horizon](showcase/water.png)

Each of the 2,073,600 pixels gets its own GPU thread. The thread adds up five sine waves to find which way the sea's surface tilts at that spot, then shades it toward the sun, from deep blue to turquoise with white glints. Code: [`step003_water/`](step003_water/)

Both images are 1920 × 1080. The ones shown were rendered on the M4 Mac mini; the M3 Max laptop produced the same gradient exactly and a water image differing in 73 of 2 million pixels, each by 1 level out of 255.

### Step 4: How fast the water renders

The step 3 water render, timed on the GPU's own clock (median of 20 runs) on both Macs. Code: [`step004_timing/`](step004_timing/)

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

Each water pixel now bounces its ray off the waves and looks up the sky in that direction. How much it reflects depends on the angle (the Fresnel effect): about 2% looking straight down, nearly everything at a glancing angle. Each pixel also averages 16 samples, which turns the sparkly noise near the horizon into smooth ripples. Code: [`step005_reflections/`](step005_reflections/)

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

H₂CO₃ → H⁺ + HCO₃⁻, ray-traced on the GPU with real bond lengths. Code: [`step006_carbonic_acid/step006a_carbonic_acid/molecule/`](step006_carbonic_acid/step006a_carbonic_acid/molecule/)

### Step 6b: The carbonic acid journey

![Looping ball-and-stick animation: CO₂ and water become carbonic acid with a helper water relaying the proton, then bicarbonate and hydronium, while fresh molecules keep arriving](showcase/journey.gif)

CO₂ + H₂O → H₂CO₃ → HCO₃⁻ + H₃O⁺, one molecule at a time. Code: [`step006_carbonic_acid/step006b_carbonic_journey/`](step006_carbonic_acid/step006b_carbonic_journey/)

### Step 7: Glucose meets oxygen

![Ball-and-stick animation: one glucose molecule and six O₂ molecules rearrange into six CO₂ and six H₂O](showcase/respiration.gif)

C₆H₁₂O₆ + 6 O₂ → 6 CO₂ + 6 H₂O, the overall accounting of cellular respiration: 24 electrons move from carbon to oxygen. Code: [`step007_glucose/step007_respiration/`](step007_glucose/step007_respiration/)

### Step 7a: Glycolysis: spend two, earn four

![Looping ball-and-stick animation: glucose gets two phosphates from two ATP, then splits between carbons 3 and 4 into two glyceraldehyde-3-phosphates](showcase/glycolysis_spend.gif)

![Looping ball-and-stick animation: two glyceraldehyde-3-phosphates side by side make 2 NADH and 4 ATP on the way to two pyruvates](showcase/glycolysis_payoff.gif)

glucose + 2 NAD⁺ + 2 ADP + 2 Pᵢ → 2 pyruvate + 2 NADH + 2 H⁺ + 2 ATP + 2 H₂O. Code: [`step007_glucose/step007a_glycolysis/`](step007_glucose/step007a_glycolysis/)

### Step 8: DNA, slowly turning

![Looping ball-and-stick animation: a DNA double helix lying diagonally turns once about its own axis, with grey carbons, blue nitrogens, red oxygens, orange phosphorus and faint dashed hydrogen bonds between the paired bases](showcase/dna.gif)

The Dickerson–Drew dodecamer, CGCGAATTCGCG, from its X-ray crystal structure ([PDB 1BNA](https://www.rcsb.org/structure/1BNA)): 758 atoms, right-handed, 10.1 base pairs per turn. Code: [`step008_dna/`](step008_dna/)

### Step 8a: pGLO, whole and up close

![Looping animation: the pGLO plasmid as a coloured ring turning once, with green GFP, violet arabinose switch, amber ampicillin resistance and blue origin against grey for the rest](showcase/plasmid_ring.gif)

![Looping animation: a continuous 150× zoom from the whole plasmid ring into the first atoms of the GFP gene, passing through a tube, then space-filling spheres, then ball-and-stick](showcase/plasmid_dive.gif)

The plasmid that makes *E. coli* glow, at its real size: 5,371 base pairs, 1.83 µm around, 512 turns of double helix. The dive changes how it draws the molecule twice on the way down, each time at the distance where the finer detail stops being smaller than a pixel. First acceleration structure in the series: a uniform grid, 369× faster than testing every ray against every shape. Code: [`step008a_plasmid/`](step008a_plasmid/)

### Step 8b: The grooves are real

![Looping animation: a length of DNA double helix drawn space-filling, every atom a sphere at its full van der Waals radius, turning slowly so the major and minor grooves spiral past as real channels in the surface](showcase/grooves.gif)

The same DNA as step 8, drawn space-filling instead of ball-and-stick. Two facts appear that ball-and-stick cannot show: the core of the duplex is packed solid, and the space outside it is not filler — it is the major and minor grooves, and the major groove is where proteins reach in to read the sequence. Measured from the real 1BNA atoms: 78% of the space within 3 Å of the axis is inside an atom, falling to 16% at the rim. The groove widths come out at 11.71 Å and 5.35 Å against published values of 11.7 and 5.7, found by scanning every cross-strand phosphate offset rather than being told where to look. Code: [`step008b_grooves/`](step008b_grooves/)

### Step 9: The protein that makes its own light

![Looping animation: green fluorescent protein drawn as overlapping van der Waals spheres turns about its axis; a round window opens in the front to reveal the chromophore in ball-and-stick inside, then closes](showcase/gfp.gif)

GFP from its crystal structure ([PDB 1EMA](https://www.rcsb.org/structure/1EMA)), space-filling: every atom a sphere at its full van der Waals radius. The chromophore inside is not a cofactor — the protein builds it out of three consecutive amino acids of its own chain, and the barrel exists to hold it rigid and keep water away. First render with ambient occlusion, which is what makes a space-filling surface readable at all. Code: [`step009_gfp/`](step009_gfp/)

### Step 9a: GFP compared with mCherry protein

![Looping animation: two space-filling protein barrels side by side, GFP on the left and mCherry on the right, turning together. Part-way round a round porthole opens in each, showing a green chromophore inside the left barrel and a red one inside the right, then both close again](showcase/color.gif)

Two barrels of almost the same size and fold, and the single bond that separates green light from red. It isn't refraction — they fluoresce, and a chromophore's colour is set by how far its π electrons can spread. mCherry's run is longer by an acylimine, and that is measurable in the coordinates: the N1–CA1 bond is **1.471 Å** in GFP (a single bond) and **1.305 Å** in mCherry (a double), 0.166 Å apart and far beyond coordinate error. The π system runs 14 atoms in one and 16 in the other. Both proteins wear the colour of the light they actually emit, computed from its wavelength through the CIE 1931 matching functions rather than chosen — and both clip, because no screen can show a pure wavelength. Code: [`step009a_color/`](step009a_color/)

### Step 10: Getting a plasmid into a bacterium

![Looping animation: a supercoiled plasmid, drawn as a branched interwound coil with calcium ions around it, drifts down onto a cross-section of the E. coli envelope — an outer membrane of lipids, a peptidoglycan mesh, and an inner membrane below](showcase/transformation_approach.gif)

![Looping animation: three panels of the same membrane side by side — CaCl₂ and heat shock labelled MODEL, electroporation labelled SIMULATED, natural competence labelled MEASURED with a protein spanning the membrane](showcase/transformation_routes.gif)

Bacterial transformation, and the first render here whose central event has never been observed: the CaCl₂ and heat-shock method dates to 1970 and its molecular mechanism is still not established. So every frame carries an **evidence bar** saying how well the thing on screen is actually known — measured, simulated, or model — and the tests enforce it, failing if the chemical route ever claims more evidence than it has. 407,000 spheres at 20 ms a frame, with the grid rebuilt every frame. Code: [`step010_transformation/`](step010_transformation/)

### Step 11: Arabinose, the sugar that switches on pGLO's gene in E. coli

![Looping animation: a loop of DNA held shut by a protein bridging two distant sites. A small sugar arrives and binds it, the grip moves along the DNA, the loop springs open, and a shape settles onto the newly exposed promoter — then the sugar leaves and the loop re-forms](showcase/switch.gif)

How pGLO's GFP gene gets switched on. AraC holds *araO2* and *araI1* — 210 base pairs apart — at the same time, tying the DNA in a loop that blocks RNA polymerase. Arabinose binds, AraC's grip moves to the adjacent site, the loop opens, and the gene can be read. The two half-sites occur in pGLO verbatim, and araO2 placed by its published offset lands exactly 210 bp away, independently. First scene here that genuinely deforms, so the acceleration grid is rebuilt every frame — which costs only 7.7% of it. Code: [`step011_switch/`](step011_switch/)

### Step 12: Bacteriophage infecting E. coli

![Looping animation: a blue protein baseplate above a layered bacterial envelope flips from a compact hexagonal dome into a flat six-pointed star, orange fibres splay down onto the surface, and a violet needle drives down through the outer membrane, the peptidoglycan mesh and the inner membrane](showcase/phage.gif)

A T4 phage lands on *E. coli* and fires. Both ends of the movement are solved structures — [PDB 5IV5](https://www.rcsb.org/structure/5IV5) hexagonal before attachment, [5IV7](https://www.rcsb.org/structure/5IV7) star-shaped after — so only the path between them is interpolated, which makes this the best-evidenced moving part the project has rendered. 5IV7 turns out to be a strict subset of 5IV5, and "hubless" in its title is literal: after firing the hub is no longer *in* the baseplate, because it has been driven into the cell. The missing half of the file is the event. Measured from the coordinates rather than quoted: the baseplate spreads 49.0 → 60.9 nm and flattens 275 → 151 Å, while the sheath contracts 206 → 130 Å. At 613,332 spheres this is the largest scene here, with the grid rebuilt every frame — 9,625× faster than testing every ray against every sphere, and the first scene where *building* the grid costs more than tracing it, at 44% of the frame. Code: [`step012_phage/`](step012_phage/)

### Step 13: Rendered model of one brine shrimp swimming

![Looping animation: a brine shrimp lies diagonally across a bright cyan field as a transmitted-light micrograph. Eleven pairs of leaf-shaped limbs beat in a wave running from tail to head, the gut shows as a soft olive line inside a translucent blue-grey body, the compound eyes are near-black, and specks of algae drift past and are drawn forward along the ventral food groove](showcase/swim.gif)

*Artemia franciscana*, adult female, about 10 mm. The first render here that stops being a reflected-light renderer. A brightfield micrograph is transmitted light, so the cyan field **is the lamp** — every tone in the animal is light that got through. Instead of shading the nearest hit, each ray now collects every interval it crosses, merges them into a union so that overlapping limbs cannot absorb twice, and returns `background × exp(−τ)`. Colour is nothing but σ per channel: the gut reads olive because its absorption is higher in blue.

The eleven pairs do not beat in unison. Each limb leads the one in front of it by exactly 1/11 of a cycle, so one metachronal wave sits on the body at any instant and the loop closes exactly. The wave runs tail to head — adlocomotory metachrony. One beat does three jobs: the limbs are oars, gills and a filter at once, and algal cells drawn in at the front travel forward along the midventral food groove to the mouth.

At 9.1 µm per pixel a seta is 0.2–0.5 px across and would flicker between frames. Clamping each to a half-pixel floor while scaling σ by the same factor holds the axial optical depth, and the effect was measured rather than asserted: rendering *only* the 546 setae and sliding the camera across one whole pixel in eighths moves the total absorbed light **1.23% at true size and 0.09% clamped**. 747 primitives, 6.6 ms a frame, 63× faster through the grid than brute force — the 9.6 s loop renders in about two seconds. Code: [`step013_swim/`](step013_swim/)
