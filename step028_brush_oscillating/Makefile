# Step 28: step 22's still, with the round head oscillating about its own axis
# as a real oscillating-rotating head does — 45° end to end, 73 times a second
# — slowed ~290×, as a forward-only GIF loop of one full cycle.
#
#   make run        render 1920 × 1080 at 16 samples per pixel
#                   → renders/brush_oscillating.gif
#   make stills     write a few frames as PNGs, to look at
#   make test       build and run the tests
#   make mutants    freeze the tufts at frame 0's lengths, swing past the cited
#                   angle, leave the loop open — and check a test catches each
#   make typecheck  prove no expression is slow to type-check
#
# If swiftc says the SDK doesn't match the compiler, point it at one that does:
#   make test SDK=/Library/Developer/CommandLineTools/SDKs/MacOSX26.sdk

SWIFTC := swiftc -O -framework CoreGraphics $(if $(SDK),-sdk $(SDK))
BUILD := .build

# The mouth and its renderer are step 20's, used rather than copied, as steps
# 21 and 22 do, and step 20 is not changed at all for this step. The brush is
# step 22's Brush.swift COPIED into this folder and extended (the disc can
# turn), so step 22 stays exactly as committed. The GIF encoder is step 8's.
SCENE := ../step020_mouth/Sources/Anatomy.swift ../step020_mouth/Sources/Render.swift
GIF := ../step008_dna/Sources/GIF.swift
SOURCES := $(SCENE) Sources/Brush.swift Sources/Oscillation.swift Sources/Frames.swift

$(BUILD)/brush_oscillating: $(SOURCES) $(GIF) Sources/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(SOURCES) $(GIF) Sources/main.swift -o $@

$(BUILD)/tests: $(SOURCES) Tests/Harness.swift Tests/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(SOURCES) Tests/Harness.swift Tests/main.swift -o $@

run: $(BUILD)/brush_oscillating
	@$< 1920 1080 4

stills: $(BUILD)/brush_oscillating
	@$< stills 1920 1080 4

test: $(BUILD)/tests
	@$<

# Each mutant must make the suite FAIL. A mutant that passes is a test missing.
mutants: $(BUILD)/tests
	@ok=1; \
	for m in frozen overswing open; do \
	  printf '=== mutation: %s ===\n' "$$m"; \
	  if OSC_MUTANT=$$m $< > $(BUILD)/mutant-$$m.log 2>&1; then \
	    printf '  NOT CAUGHT — no test failed\n'; ok=0; \
	  else \
	    grep '^  FAIL' $(BUILD)/mutant-$$m.log || true; \
	  fi; \
	done; \
	[ $$ok -eq 1 ] || exit 1
	@echo "every mutation was caught"

typecheck:
	$(SWIFTC) -typecheck -Xfrontend -warn-long-expression-type-checking=5 \
	  $(SOURCES) $(GIF) Sources/main.swift

clean:
	rm -rf $(BUILD)

.PHONY: run stills test mutants typecheck clean
