# Step 15: a coconut sprouting — a vertical cutaway of a germinating nut, in
# portrait. The white ball inside is not meat: it is the haustorium, a new organ
# the embryo grows to eat the nut from the inside.
#
#   make run        render renders/coconut.gif
#   make stills     write a few frames as PNGs
#   make sweep      sweep the clip plane through the nut, as PNGs
#   make bench      time brute force against the uniform grid
#   make test       build and run the Swift tests
#   make mutants    put a taproot in, invert a shell, let the haustorium grow
#                   without the endosperm shrinking, make a pinnate leaf appear,
#                   and turn the mailbox off — and check a test catches each
#
# If swiftc says the SDK doesn't match the compiler, point it at one that does:
#   make test SDK=/Library/Developer/CommandLineTools/SDKs/MacOSX26.sdk
# On the M3 Max laptop, `SDK=$(xcrun --show-sdk-path)` is the right one.

# Command-line tools must link CoreGraphics for MTLCreateSystemDefaultDevice()
# to find the GPU (per Apple's docs). CoreText draws the labels, ImageIO the
# PNG stills.
SWIFTC := swiftc -O -framework CoreGraphics -framework CoreText -framework ImageIO \
          $(if $(SDK),-sdk $(SDK))
BUILD := .build

# The ordered multi-hit traversal — every entry and exit along the ray, sorted,
# merged into a union, with per-ray mailboxing — is step 13's, and this step
# uses it rather than a copy of it. Cutaway.swift splices step 13's kernel
# source in ahead of its own shading, so the intersection routines, the sorted
# insert and the grid walk on the GPU are the same characters step 13 shipped.
# The caption bar is step 13's too, and the GIF encoder is step 8's, as in
# every step since.
TRAVERSAL := ../step013_swim/Sources/Render.swift
OVERLAY := ../step013_swim/Sources/Overlay.swift
GIF := ../step008_dna/Sources/GIF.swift
OWN := Sources/Coconut.swift Sources/Cutaway.swift Sources/Annotate.swift
SOURCES := $(TRAVERSAL) $(OVERLAY) $(GIF) $(OWN)

$(BUILD)/coconut: $(SOURCES) Sources/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(filter %.swift,$^) -o $@

$(BUILD)/tests: $(SOURCES) Tests/Harness.swift Tests/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(filter %.swift,$^) -o $@

run: $(BUILD)/coconut
	@$<

stills: $(BUILD)/coconut
	@$< stills

sweep: $(BUILD)/coconut
	@$< sweep

bench: $(BUILD)/coconut
	@$< bench

test: $(BUILD)/tests
	@$<

# The mutation check. Each name breaks exactly one thing; the suite must go red
# for every one of them, and the failing test's name is printed so it is on the
# record which test did the catching.
mutants: $(BUILD)/tests
	@for m in taproot invert budget pinnate mailbox; do \
	  printf '=== mutation: %s ===\n' "$$m"; \
	  if COCO_MUTATE=$$m $< > /dev/null 2>&1; then \
	    printf '  NOT CAUGHT — no test failed\n'; exit 1; \
	  else \
	    COCO_MUTATE=$$m $< 2>&1 | grep '^  FAIL' || true; \
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

.PHONY: run stills sweep bench test mutants typecheck clean
