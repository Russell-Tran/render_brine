# World-building experiment: findings so far

A running log against [`PREREGISTRATION.md`](PREREGISTRATION.md) (ae6f1e6). Each entry is
verified on the mini as well as the laptop. The final write-up comes after the film.

## P1, MP4: **partly right** (lib/video/v1, fcb4f05, fixed in 0822041)

- The MP4 writes, as predicted: AVAssetWriter, H.264 on the hardware encoder, no
  macOS 15+ APIs, building on both toolchains (Swift 5.10 and 6.x). Verified here: 16
  tests, 4 mutants.
- **"≥ 10× smaller" was wrong in spirit.** At the film's bitrate it's 12–21× smaller than
  the showcase GIFs' frames, but at 30–37 dB PSNR, with fine texture (step 57's wood
  grain) going soft. **At comparable quality it's 3–6× smaller.**
- Found on the way: tagging only the output with BT.709 darkened every channel by about
  9/255, so the source buffers need the tag too. On the mini's compiler, with AVFoundation
  in scope, even `trackCount == 1` was slow to type-check.

## P2, a trail emerges: **right**

- Laptop, seed 79: concentration index 0.06 → 0.47 (chance 0.045). The trail formed at
  film 47.7 s (ant time 716 s). Over 8 seeds it formed in 7/8, and in 0/8 without
  pheromone. The `noPheromone` mutant is caught.
- Mini, seed 79: index 0.06 → **0.70**, formed at film **33.7 s** (ant time 506 s).

## Cross-machine divergence: a finding not predicted

The **same seed gives a different world on each machine.** Each machine is
bit-for-bit repeatable with itself, but the M3 laptop and the M4 mini diverge: trail
timing 716 s vs 506 s, overlap 2.1% vs 2.8%. The colony is chaotic, so last-bit
floating-point differences (GPU or compiler) grow into different paths. The trail
*forms* on both, and *when* depends on the machine.

**Consequence for world-building:** a world can't be re-run from its seed and trusted
to match. It has to be **recorded**, simulated once and rendered from the record. That
changes the P5 plan: splitting one film's frames across machines only works from a
shared record. Requested of the laptop before the film is rendered.

## P3, ant-count scaling: **partly right** (step 79 part 4, aac3701)

One 1080p frame at 4 samples per pixel. Measured on **both** machines.

| Mini, 40 ants | wide shot | close-up |
|---|---|---|
| no speed-up | 61.9 s/frame (31 h per 1800-frame film) | 243.7 s/frame (122 h) |
| per-ant bounding spheres | **1.3 s/frame (0.7 h)** | **8.2 s/frame (4.1 h)** |

- **Linear:** right. On the mini, frame time grows about 1.5 s per ant (wide) and 6 s per
  ant (close) without the speed-up.
- **"Impractical at 20–40 ants":** wrong. It's already impractical at **10 ants** (7–26 h
  per film on the mini; the laptop's figures are similar).
- **"Needs an acceleration structure":** wrong in a good way. The shared ant library's
  per-ant bounding spheres, a simple skip, are **30–47× faster with bit-identical pixels**,
  and nothing more elaborate is needed. The laptop measures 1.2–4.3 s per frame at 40
  ants, and the mini's close-up is about 2× slower.

## P4, a shared ant: **right** (lib/ant/v1, b5384ba, fixed in b43a47f; used by step 79 in aac3701)

- The versioned module exists and builds on both toolchains. Its shapes match step
  55's to 1.4 × 10⁻⁶ mm over 1600 poses. Verified here: 24 tests, 3 mutants.
- The old ant steps 44, 55, 56 and 57 are byte-untouched and still pass (23, 29, 29, 32).
  Step 44 can only build on the mini.
- **New limits the scripted trails never hit:** the gait holds only down to a 5 mm turn
  radius, and a frame is final only once the track runs a stride ahead. v1 ants **can't
  back up**, so head-on ants squeeze past each other (2–3% overlap away from the food).
- Step 79 builds against the module (Metal plus poses) and links lib/video/v1. Whether this was "the biggest change to how we work" is Russell's judgement.
- Process change adopted: `lib/README`, where a version freezes once any step imports it.

## Still to come

P5 (render time and resume, plus whether the recorded world renders the same on both machines), P6 (can a stranger tell a trail
formed on its own).
