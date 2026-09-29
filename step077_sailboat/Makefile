# Step 77: an ILCA 7 (Laser) sailing dinghy, one still, floating where
# Archimedes puts it, with its underwater body seen through cut-away water
#
#   make run        render 1920 × 1080 at 25 samples per pixel → renders/sailboat.png
#   make preview    960 × 540 at 4 samples → renders/preview.png
#   make test       build and run the tests
#   make mutants    break it three ways; the tests must catch each
#   make typecheck  prove no expression is slow to type-check
#
# Always build against the command-line tools' SDK on the machine at hand:
#   make test SDK=$(xcrun --show-sdk-path)

SDK ?= /Library/Developer/CommandLineTools/SDKs/MacOSX26.sdk
SWIFTC := swiftc -O -framework CoreGraphics $(if $(SDK),-sdk $(SDK))
BUILD := .build

# Scene and Render are step 72's camera, studio and plastic, copied and
# adapted; the rest is this step's.
SOURCES := Sources/Boat.swift Sources/Scene.swift Sources/Render.swift Sources/Annotate.swift

$(BUILD)/sailboat: $(SOURCES) Sources/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(SOURCES) Sources/main.swift -o $@

$(BUILD)/tests: $(SOURCES) Tests/Harness.swift Tests/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(SOURCES) Tests/Harness.swift Tests/main.swift -o $@

run: $(BUILD)/sailboat
	@mkdir -p renders
	@$< 1920 1080 5 renders/sailboat.png
	cp renders/sailboat.png ../showcase/sailboat.png

preview: $(BUILD)/sailboat
	@mkdir -p renders
	@$< 960 540 2 renders/preview.png

test: $(BUILD)/tests
	@$<

# Each mutant must make the suite FAIL. A mutant that passes is a test missing.
mutants: $(BUILD)/tests
	@ok=1; \
	for m in floatsWrong wrongSize noKeel; do \
	  printf '=== mutation: %s ===\n' "$$m"; \
	  if BOAT_MUTANT=$$m $< > $(BUILD)/mutant-$$m.log 2>&1; then \
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
