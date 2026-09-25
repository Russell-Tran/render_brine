# Step 14: one mysis shrimp swimming — a DARKFIELD micrograph of Mysis
# diluviana, the opossum shrimp, from a camera that never moves.
#
#   make run        render renders/mysis.gif
#   make stills     write a few frames as PNGs
#   make bench      time the swept-bound mask against no mask, and the grid
#                   against brute force
#   make test       build and run the Swift tests
#   make mutants    break the grazing weight, the swept-bound mask, the loop
#                   phase and the direction of the metachronal wave in turn,
#                   and check a test catches each one
#
# If swiftc says the SDK doesn't match the compiler, point it at one that does:
#   make test SDK=/Library/Developer/CommandLineTools/SDKs/MacOSX26.sdk
# On the M3 Max laptop, `SDK=$(xcrun --show-sdk-path)` is the right one.

SWIFTC := swiftc -O -framework CoreGraphics -framework CoreText -framework ImageIO \
          $(if $(SDK),-sdk $(SDK))
BUILD := .build

# Step 13 built the ordered multi-hit traversal this step needs: every entry
# and exit along the ray, sorted, merged into a per-tissue union, mailboxed
# across grid boxes. It is not copied here, it is COMPILED here — the same
# file, the same way step 4 timed step 3's kernel by naming its source.
#
#   Render.swift   GPUPrim, the closest-approach intersections, the orthographic
#                  Camera, the uniform grid, the CPU reference intersector
#   Overlay.swift  the caption bar and the three-level evidence bar
#   GIF.swift      the step 8 encoder: one palette, changed pixels only
SWIM := ../step013_swim/Sources/Render.swift ../step013_swim/Sources/Overlay.swift
GIF := ../step008_dna/Sources/GIF.swift
HARNESS := ../step013_swim/Tests/Harness.swift

SOURCES := $(SWIM) $(GIF) Sources/Darkfield.swift Sources/Mysis.swift

$(BUILD)/mysis: $(SOURCES) Sources/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(filter %.swift,$^) -o $@

$(BUILD)/tests: $(SOURCES) $(HARNESS) Tests/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(filter %.swift,$^) -o $@

run: $(BUILD)/mysis
	@$<

stills: $(BUILD)/mysis
	@$< stills

bench: $(BUILD)/mysis
	@$< bench

test: $(BUILD)/tests
	@$<

# The mutation check. Each name breaks exactly one thing; the suite must go
# red for every one of them, and the failing test's name is printed so it is
# on the record which test did the catching.
mutants: $(BUILD)/tests
	@for m in grazing mask phase wave union; do \
	  printf '=== mutation: %s ===\n' "$$m"; \
	  if MYSIS_MUTATE=$$m $< > /dev/null 2>&1; then \
	    printf '  NOT CAUGHT — no test failed\n'; exit 1; \
	  else \
	    MYSIS_MUTATE=$$m $< 2>&1 | grep '^  FAIL' || true; \
	  fi; \
	done
	@echo "every mutation was caught"

# The type checker on the laptop's older toolchain gives up on long chained
# arithmetic, so every expression here is written in typed steps. This proves
# it: nothing may take longer than 5 ms to type-check.
typecheck:
	$(SWIFTC) -typecheck -Xfrontend -warn-long-expression-type-checking=5 \
	  $(SOURCES) Sources/main.swift

clean:
	rm -rf $(BUILD) renders

.PHONY: run stills bench test mutants typecheck clean
