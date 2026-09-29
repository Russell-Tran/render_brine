# Step 78: a waʻa kaulua (Hawaiian double-hulled voyaging canoe), extremely
# simplified, to the proportions of PVS's published plans for Hōkūleʻa; one
# still, its waterline computed by buoyancy.
#
#   make run        render 1920 × 1080 at 25 samples per pixel → renders/polynesian_canoe.png
#   make preview    960 × 540 at 4 samples → renders/preview.png
#   make test       build and run the tests
#   make mutants    break it three ways; the tests must catch each
#   make typecheck  prove no expression is slow to type-check
#
# Always build against the command-line tools' SDK here, e.g. on the laptop:
#   make test SDK=$(xcrun --show-sdk-path)

SDK ?= /Library/Developer/CommandLineTools/SDKs/MacOSX26.sdk
SWIFTC := swiftc -O -framework CoreGraphics $(if $(SDK),-sdk $(SDK))
BUILD := .build

# Camera maths is step 72's, copied; the rest is this step's.
SOURCES := Sources/Canoe.swift Sources/Scene.swift Sources/Render.swift Sources/Buoyancy.swift Sources/Section.swift Sources/Annotate.swift

$(BUILD)/canoe: $(SOURCES) Sources/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(SOURCES) Sources/main.swift -o $@

$(BUILD)/tests: $(SOURCES) Tests/Harness.swift Tests/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(SOURCES) Tests/Harness.swift Tests/main.swift -o $@

run: $(BUILD)/canoe
	@mkdir -p renders
	@$< 1920 1080 5 renders/polynesian_canoe.png
	cp renders/polynesian_canoe.png ../showcase/polynesian_canoe.png

preview: $(BUILD)/canoe
	@mkdir -p renders
	@$< 960 540 2 renders/preview.png

test: $(BUILD)/tests
	@$<

# Each mutant must make the suite FAIL. A mutant that passes is a test missing.
mutants: $(BUILD)/tests
	@ok=1; \
	for m in floatsWrong wrongSize singleHull; do \
	  printf '=== mutation: %s ===\n' "$$m"; \
	  if CANOE_MUTANT=$$m $< > $(BUILD)/mutant-$$m.log 2>&1; then \
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
