# Step 67: the upper arch, with the camera travelling along it — from the
# midline between the upper central incisors (#8 and #9) back past the second
# molar (#15) to the wisdom tooth, #16, then a dissolve forward to the start.
# Step 21's move, one jaw up.
#
#   make run        render renders/upper_arch_pan.gif and copy it to ../showcase
#   make measure    ten frames only, to see what a frame costs in the GIF
#   make stills     write a few frames as PNGs, to look at
#   make test       build and run the tests
#   make mutants    break the teeth, the steering, the loop, the camera and the
#                   side — and check a test catches each
#   make typecheck  prove no expression is slow to type-check
#
# If swiftc says the SDK doesn't match the compiler, point it at one that does:
#   make test SDK=/Library/Developer/CommandLineTools/SDKs/MacOSX26.sdk

SWIFTC := swiftc -O -framework CoreGraphics $(if $(SDK),-sdk $(SDK))
BUILD := .build

# The scene is step 66's, copied (not shared) so step 66 stays exactly as
# committed. The GIF encoder is step 8's, as in every animated step since.
GIF := ../step008_dna/Sources/GIF.swift
SOURCES := Sources/Anatomy.swift Sources/Render.swift Sources/Shot.swift

$(BUILD)/upper_arch_pan: $(SOURCES) $(GIF) Sources/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(SOURCES) $(GIF) Sources/main.swift -o $@

$(BUILD)/tests: $(SOURCES) Tests/Harness.swift Tests/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(SOURCES) Tests/Harness.swift Tests/main.swift -o $@

run: $(BUILD)/upper_arch_pan
	@$< 544 306 3
	@cp renders/upper_arch_pan.gif ../showcase/upper_arch_pan.gif

measure: $(BUILD)/upper_arch_pan
	@$< 544 306 3 10 60

stills: $(BUILD)/upper_arch_pan
	@$< stills 544 306 3

test: $(BUILD)/tests
	@$<

# Each mutant must make the suite FAIL. A mutant that passes is a test missing.
mutants: $(BUILD)/tests
	@ok=1; \
	printf '=== mutation: mandibularSizes ===\n'; \
	if UPPER_MUTANT=mandibularSizes $< > $(BUILD)/mutant-mandibularSizes.log 2>&1; then \
	  printf '  NOT CAUGHT — no test failed\n'; ok=0; \
	else grep '^  FAIL' $(BUILD)/mutant-mandibularSizes.log || true; fi; \
	for m in raw rewind frozenCamera mirrored; do \
	  printf '=== mutation: %s ===\n' "$$m"; \
	  if SHOT_MUTANT=$$m $< > $(BUILD)/mutant-$$m.log 2>&1; then \
	    printf '  NOT CAUGHT — no test failed\n'; ok=0; \
	  else grep '^  FAIL' $(BUILD)/mutant-$$m.log || true; fi; \
	done; \
	[ $$ok -eq 1 ] || exit 1
	@echo "every mutation was caught"

typecheck:
	$(SWIFTC) -typecheck -Xfrontend -warn-long-expression-type-checking=5 \
	  $(SOURCES) $(GIF) Sources/main.swift

clean:
	rm -rf $(BUILD)

.PHONY: run measure stills test mutants typecheck clean
