# Step 21: the lower arch, with the camera travelling along it — from the
# midline between the lower central incisors back to the second molar, #18,
# then a dissolve back to the start.
#
#   make run        render renders/arch_pan.gif
#   make stills     write a few frames as PNGs, to look at
#   make test       build and run the tests
#   make mutants    steer by the raw arch, rewind instead of dissolving — and
#                   check a test catches each
#   make typecheck  prove no expression is slow to type-check
#
# If swiftc says the SDK doesn't match the compiler, point it at one that does:
#   make test SDK=/Library/Developer/CommandLineTools/SDKs/MacOSX26.sdk

SWIFTC := swiftc -O -framework CoreGraphics $(if $(SDK),-sdk $(SDK))
BUILD := .build

# The scene and its renderer are step 20's, used rather than copied: the
# teeth, gum, colours and lights are the same characters step 20 shipped. The
# only change step 20 took for this step is that its camera is read at run
# time instead of being typed into the kernel, and its own picture came out
# bit-for-bit the same. The GIF encoder is step 8's, as in every step since.
SCENE := ../step020_mouth/Sources/Anatomy.swift ../step020_mouth/Sources/Render.swift
GIF := ../step008_dna/Sources/GIF.swift
SOURCES := $(SCENE) Sources/Shot.swift

$(BUILD)/arch_pan: $(SOURCES) $(GIF) Sources/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(SOURCES) $(GIF) Sources/main.swift -o $@

$(BUILD)/tests: $(SOURCES) Tests/Harness.swift Tests/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(SOURCES) Tests/Harness.swift Tests/main.swift -o $@

run: $(BUILD)/arch_pan
	@$< 576 324 3

stills: $(BUILD)/arch_pan
	@$< stills 576 324 3

test: $(BUILD)/tests
	@$<

# Each mutant must make the suite FAIL. A mutant that passes is a test missing.
mutants: $(BUILD)/tests
	@ok=1; \
	for m in raw rewind; do \
	  printf '=== mutation: %s ===\n' "$$m"; \
	  if SHOT_MUTANT=$$m $< > $(BUILD)/mutant-$$m.log 2>&1; then \
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
