# Step 19: the neck of a tooth — a molar in section with the stress field it is
# carrying painted on the cut face, and the cracks drawn as they run.
#
#   make run        render renders/tooth.gif
#   make stills     write a few frames as PNGs
#   make physics    solve and print the numbers, render nothing
#   make bench      time brute force against the uniform grid
#   make test       build and run the Swift tests
#   make mutants    pin the root rigidly, give dentin enamel's strength, let
#                   broken bonds heal, freeze the evidence bar — and check a
#                   test catches each
#   make typecheck  prove no expression is slow to type-check
#
# If swiftc says the SDK doesn't match the compiler, point it at one that does:
#   make test SDK=/Library/Developer/CommandLineTools/SDKs/MacOSX26.sdk
# On the M3 Max laptop, `SDK=$(xcrun --show-sdk-path)` is the right one.

SWIFTC := swiftc -O -framework CoreGraphics -framework CoreText -framework ImageIO \
          $(if $(SDK),-sdk $(SDK))
BUILD := .build

# The ordered multi-hit traversal — every entry and exit along the ray, sorted,
# merged into a union, with per-ray mailboxing — is step 13's, and this step uses
# it rather than a copy of it. Cutaway.swift splices step 13's kernel source in
# ahead of its own shading, exactly as step 15 did, so the intersection routines,
# the sorted insert and the grid walk on the GPU are the same characters step 13
# shipped. The caption bar is step 13's too, and the GIF encoder is step 8's, as
# in every step since.
#
# The evidence bar is NOT step 13's: this step needs a fourth standing,
# CONTESTED, and adding a level to step 13's file would change step 13's
# picture. See the head of Annotate.swift.
TRAVERSAL := ../step013_swim/Sources/Render.swift
OVERLAY := ../step013_swim/Sources/Overlay.swift
GIF := ../step008_dna/Sources/GIF.swift
OWN := Sources/Lattice.swift Sources/Tooth.swift Sources/Model.swift Sources/Paint.swift \
       Sources/Body.swift Sources/Cutaway.swift Sources/Storyboard.swift Sources/Annotate.swift
SOURCES := $(TRAVERSAL) $(OVERLAY) $(GIF) $(OWN)

$(BUILD)/tooth: $(SOURCES) Sources/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(filter %.swift,$^) -o $@

$(BUILD)/tests: $(SOURCES) Tests/Harness.swift Tests/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(filter %.swift,$^) -o $@

run: $(BUILD)/tooth
	@$<

stills: $(BUILD)/tooth
	@$< stills

physics: $(BUILD)/tooth
	@$< physics

bench: $(BUILD)/tooth
	@$< bench

test: $(BUILD)/tests
	@$<

# The mutation check. Each name breaks exactly one thing, and the suite must go
# red for every one of them. The failing test's name is printed, so it is on the
# record which test did the catching.
#
# The suite takes minutes on this model — it solves the tooth at three meshes and
# runs six load ramps — so each mutant is run ONCE and its output kept, rather
# than run once for the exit code and again for the message. Budget most of an
# hour for all five: `clamp` and `punch` both make the tooth shatter, and a
# shattering tooth runs every ramp to its step ceiling before it gives up.
mutants: $(BUILD)/tests
	@ok=1; \
	for m in clamp brittle heal freeze punch; do \
	  printf '=== mutation: %s ===\n' "$$m"; \
	  if TOOTH_MUTATE=$$m $< > $(BUILD)/mutant-$$m.log 2>&1; then \
	    printf '  NOT CAUGHT — no test failed\n'; ok=0; \
	  else \
	    grep '^  FAIL' $(BUILD)/mutant-$$m.log || true; \
	  fi; \
	done; \
	[ $$ok -eq 1 ] || exit 1
	@echo "every mutation was caught"

# The laptop's older type checker gives up on long chained arithmetic, so every
# expression here is written in typed steps. This proves it: nothing may take
# longer than 5 ms to type-check.
typecheck:
	$(SWIFTC) -typecheck -Xfrontend -warn-long-expression-type-checking=5 \
	  $(SOURCES) Sources/main.swift

clean:
	rm -rf $(BUILD) renders

.PHONY: run stills physics bench test mutants typecheck clean
