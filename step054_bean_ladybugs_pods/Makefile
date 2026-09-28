# Step 54: step 45's pole bean in flower — white flowers on axillary racemes
# above, small young pods below, a gradient along the stem by node age — with
# seven-spot ladybirds walking on it in real time, on the same climbing stem,
# pole, camera and loop.
#
#   make run        render renders/bean_ladybugs_pods.gif and copy it to ../showcase/
#   make measure    render ten frames only, to see what a frame costs in the GIF
#   make stills     write five frames across the loop as PNGs, to look at
#   make test       build and run the tests
#   make mutants    recolour the petals violet, put the pods above the flowers,
#                   give a ladybird eight legs, let its feet slide, rewind the
#                   loop, twine the wrong way — and check a test catches each
#   make typecheck  prove no expression is slow to type-check
#
# Always build against the command-line tools' SDK here:
#   make test SDK=/Library/Developer/CommandLineTools/SDKs/MacOSX26.sdk

SDK ?= /Library/Developer/CommandLineTools/SDKs/MacOSX26.sdk
SWIFTC := swiftc -O -framework CoreGraphics $(if $(SDK),-sdk $(SDK))
BUILD := .build

# Plant.swift, Kernel.swift, Render.swift and Picture.swift are the shared
# climbing-plant machinery from steps 45-48; step 54 adds the ladybird's body
# (a Beetle shape) to Plant, Kernel and Render. Bean.swift is step 45's bean
# with flowers and pods; Ladybug.swift the ladybirds. The GIF encoder is step
# 8's, as in every animated step since. Inset.swift draws the magnified
# ladybird.
SOURCES := Sources/Plant.swift Sources/Kernel.swift Sources/Render.swift Sources/Picture.swift Sources/Bean.swift Sources/Ladybug.swift Sources/Inset.swift
GIF := ../step008_dna/Sources/GIF.swift

$(BUILD)/bean_ladybugs_pods: $(SOURCES) $(GIF) Sources/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(SOURCES) $(GIF) Sources/main.swift -o $@

$(BUILD)/tests: $(SOURCES) $(GIF) Tests/Harness.swift Tests/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(SOURCES) $(GIF) Tests/Harness.swift Tests/main.swift -o $@

run: $(BUILD)/bean_ladybugs_pods
	@mkdir -p renders
	@$< 480 624 2
	cp renders/bean_ladybugs_pods.gif ../showcase/bean_ladybugs_pods.gif

measure: $(BUILD)/bean_ladybugs_pods
	@mkdir -p renders
	@$< 480 624 2 10

stills: $(BUILD)/bean_ladybugs_pods
	@mkdir -p renders
	@$< stills 480 624 2

test: $(BUILD)/tests
	@$<

# Each mutant must make the suite FAIL. A mutant that passes is a test missing.
mutants: $(BUILD)/tests
	@ok=1; \
	for m in purple pods_above_flowers eight_legs sliding_feet rewind left_handed; do \
	  printf '=== mutation: %s ===\n' "$$m"; \
	  if BEAN_MUTANT=$$m $< > $(BUILD)/mutant-$$m.log 2>&1; then \
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

.PHONY: run measure stills test mutants typecheck clean
