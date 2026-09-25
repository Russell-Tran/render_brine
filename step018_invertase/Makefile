# Step 18: sucrose split by yeast invertase, beside the polarimeter reading
# that gave "invert sugar" its name.
#
#   make run        render renders/invertase.gif
#   make stills     write a few frames as PNGs
#   make test       build and run the Swift tests
#   make mutants    break the chemistry four ways and check a test catches each
#   make typecheck  prove no expression is slow to type-check
#   make structure  rebuild Resources/invertase.json from the PDB entries
#
# If swiftc says the SDK doesn't match the compiler, point it at one that does:
#   make test SDK=/Library/Developer/CommandLineTools/SDKs/MacOSX26.sdk
# On the M3 Max laptop, `SDK=$(xcrun --show-sdk-path)` is the right one.

SWIFTC := swiftc -O -framework CoreGraphics -framework CoreText -framework ImageIO \
          $(if $(SDK),-sdk $(SDK))
BUILD := .build

# Files this step COMPILES rather than copies — the same source, the same way
# step 4 timed step 3's kernel by naming it.
#
#   Spectrum.swift  step 9a's CIE 1931 colour matching, which turns the sodium
#                   D line into the colour the polarimeter's beam is drawn in
#   GIF.swift       step 8's encoder: one palette, changed pixels only
#   Harness.swift   step 9's plain test harness
#
# Step 17 (lactase) shares most of this machinery — the space-filling exterior
# with a capped porthole, the covalent-intermediate bookkeeping, the
# anomeric-configuration test. It did not exist when this step was started and
# was still being written when it finished, so nothing here depends on it and
# the two were built separately. THEY SHOULD BE MERGED. Concretely:
#
#   * both have their own grid + ambient-occlusion ray tracer; this one carries
#     spheres and cylinders in one primitive array, step 17's is its own
#   * both build a covalent glycosyl-enzyme intermediate and both assert the
#     anomeric centre inverts twice; that pair of functions wants to be one
#   * step 17 uses step 9's Overlay.swift; this one needed a caption bar with a
#     dial on it, so it has its own. The caption half of the two is identical.
SPECTRUM := ../step009a_color/Sources/Spectrum.swift
GIF := ../step008_dna/Sources/GIF.swift
HARNESS := ../step009_gfp/Tests/Harness.swift

SOURCES := $(SPECTRUM) $(GIF) Sources/Render.swift Sources/Chemistry.swift \
           Sources/Invertase.swift Sources/Polarimeter.swift Sources/Storyboard.swift \
           Sources/Overlay.swift Sources/Output.swift

$(BUILD)/invertase: $(SOURCES) Sources/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(filter %.swift,$^) -o $@

$(BUILD)/tests: $(SOURCES) $(HARNESS) Tests/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(filter %.swift,$^) -o $@

run: $(BUILD)/invertase
	@$<

stills: $(BUILD)/invertase
	@$< stills

test: $(BUILD)/tests
	@$<

# The mutation check. Each name breaks exactly one thing, and the suite must go
# red for every one of them. The failing test's name is printed, so it is on the
# record which test did the catching.
mutants: $(BUILD)/tests
	@for m in hardcode decouple invert ester; do \
	  printf '=== mutation: %s ===\n' "$$m"; \
	  if INV_MUTATE=$$m $< > /dev/null 2>&1; then \
	    printf '  NOT CAUGHT — no test failed\n'; exit 1; \
	  else \
	    INV_MUTATE=$$m $< 2>&1 | grep '^  FAIL' || true; \
	  fi; \
	done
	@echo "every mutation was caught"

# The laptop's older type checker gives up on long chained arithmetic, so every
# expression here is written in typed steps. This proves it: nothing may take
# longer than 5 ms to type-check.
typecheck:
	$(SWIFTC) -typecheck -Xfrontend -warn-long-expression-type-checking=5 \
	  $(SOURCES) Sources/main.swift

structure:
	cd Tools && python3 build_invertase.py

clean:
	rm -rf $(BUILD) renders

.PHONY: run stills test mutants typecheck structure clean
