# Step 60: a plastic toy pig, one still
#
#   make run        render 1920 × 1080 at 25 samples per pixel → renders/toy_pig.png
#   make preview    960 × 540 at 4 samples → renders/preview.png
#   make test       build and run the tests
#   make mutants    break it four ways; the tests must catch each
#   make typecheck  prove no expression is slow to type-check
#
# Always build against the command-line tools' SDK here:
#   make test SDK=/Library/Developer/CommandLineTools/SDKs/MacOSX26.sdk

SDK ?= /Library/Developer/CommandLineTools/SDKs/MacOSX26.sdk
SWIFTC := swiftc -O -framework CoreGraphics $(if $(SDK),-sdk $(SDK))
BUILD := .build

# Plastic, Toy, Scene, Render, Still and Annotate are the shared toy renderer
# (steps 58–63 each carry a copy); Pig.swift is this step's animal; Annotate's parting-line pointer is generalised from step 58's.
SOURCES := Sources/Plastic.swift Sources/Toy.swift Sources/Scene.swift Sources/Render.swift Sources/Still.swift \
           Sources/Annotate.swift Sources/Pig.swift

$(BUILD)/toy: $(SOURCES) Sources/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(SOURCES) Sources/main.swift -o $@

$(BUILD)/tests: $(SOURCES) Tests/Harness.swift Tests/Common.swift Tests/Hooves.swift Tests/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(SOURCES) Tests/Harness.swift Tests/Common.swift Tests/Hooves.swift Tests/main.swift -o $@

run: $(BUILD)/toy
	@mkdir -p renders
	@$< 1920 1080 5 renders/toy_pig.png
	cp renders/toy_pig.png ../showcase/toy_pig.png

preview: $(BUILD)/toy
	@mkdir -p renders
	@$< 960 540 2 renders/preview.png

test: $(BUILD)/tests
	@$<

# Each mutant must make the suite FAIL. A mutant that passes is a test missing.
mutants: $(BUILD)/tests
	@ok=1; \
	for m in wholeHoof dielectricZero noSeam; do \
	  printf '=== mutation: %s ===\n' "$$m"; \
	  if TOY_MUTANT=$$m $< > $(BUILD)/mutant-$$m.log 2>&1; then \
	    printf '  NOT CAUGHT — no test failed\n'; ok=0; \
	  else \
	    grep '^  FAIL' $(BUILD)/mutant-$$m.log || true; \
	  fi; \
	done; \
	[ $$ok -eq 1 ] || exit 1
	@echo "every mutation was caught"

typecheck:
	$(SWIFTC) -typecheck -Xfrontend -warn-long-expression-type-checking=5 \
	  $(SOURCES) Sources/main.swift

clean:
	rm -rf $(BUILD)

.PHONY: run preview test mutants typecheck clean
