# Step 17: lactose cut into glucose and galactose by beta-galactosidase.
#
#   make run        render renders/lactase.gif
#   make stills     write the still frames for the results page
#   make bench      time the grid against brute force, and the occlusion probes
#   make constants  print every constant with its evidence level and source
#   make test       build and run the Swift tests
#   make mutants    break the retention, the covalent bond, the loop closure and
#                   Glu461's one-job-at-a-time rule in turn, and check that a
#                   test catches each one
#   make structure  rebuild Resources/scene.json from the PDB entries
#
# If swiftc says the SDK doesn't match the compiler, point it at one that does:
#   make test SDK=/Library/Developer/CommandLineTools/SDKs/MacOSX26.sdk
# On the M3 Max laptop, `SDK=$(xcrun --show-sdk-path)` is the right one.

SWIFTC := swiftc -O -framework CoreGraphics -framework CoreText -framework ImageIO \
          $(if $(SDK),-sdk $(SDK))
BUILD := .build

# Two files are COMPILED here rather than copied, the way step 4 timed step 3's
# kernel by naming its source and step 14 built on step 13's:
#   step 8's GIF encoder   one palette for the loop, changed pixels only
#   step 9's overlay       the house caption bar, and nothing else — this step
#                          has no evidence bar, for step 13's reason
GIF := ../step008_dna/Sources/GIF.swift
OVERLAY := ../step009_gfp/Sources/Overlay.swift
HARNESS := ../step009_gfp/Tests/Harness.swift

SOURCES := $(GIF) $(OVERLAY) Sources/Render.swift Sources/Lactase.swift

$(BUILD)/lactase: $(SOURCES) Sources/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(filter %.swift,$^) -o $@

$(BUILD)/tests: $(SOURCES) $(HARNESS) Tests/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(filter %.swift,$^) -o $@

run: $(BUILD)/lactase
	@$<

stills: $(BUILD)/lactase
	@$< stills

bench: $(BUILD)/lactase
	@$< bench

constants: $(BUILD)/lactase
	@$< constants

test: $(BUILD)/tests
	@$<

# The mutation check. Each name breaks exactly one thing; the suite must go red
# for every one of them, and the failing test's name is printed so it is on the
# record which test did the catching.
mutants: $(BUILD)/tests
	@for m in invert bond loop roles; do \
	  printf '=== mutation: %s ===\n' "$$m"; \
	  if LAC_MUTATE=$$m $< > /dev/null 2>&1; then \
	    printf '  NOT CAUGHT — no test failed\n'; exit 1; \
	  else \
	    LAC_MUTATE=$$m $< 2>&1 | grep '^  FAIL' || true; \
	  fi; \
	done
	@echo "every mutation was caught"

# The type checker on the laptop's older toolchain gives up on long chained
# arithmetic, so every expression here is written in typed steps. This proves
# it: nothing may take longer than 5 ms to type-check.
typecheck:
	$(SWIFTC) -typecheck -Xfrontend -warn-long-expression-type-checking=5 \
	  $(SOURCES) Sources/main.swift

structure:
	python3 Tools/build_lactase.py

clean:
	rm -rf $(BUILD) renders

.PHONY: run stills bench constants test mutants typecheck structure clean
