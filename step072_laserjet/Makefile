# Step 72: an HP LaserJet P1102w, one still, with an inset of how it prints
#
#   make run        render 1920 × 1080 at 25 samples per pixel → renders/laserjet.png
#   make preview    960 × 540 at 4 samples → renders/preview.png
#   make test       build and run the tests
#   make mutants    break it three ways; the tests must catch each
#   make typecheck  prove no expression is slow to type-check
#
# Always build against the command-line tools' SDK here:
#   make test SDK=/Library/Developer/CommandLineTools/SDKs/MacOSX26.sdk

SDK ?= /Library/Developer/CommandLineTools/SDKs/MacOSX26.sdk
SWIFTC := swiftc -O -framework CoreGraphics $(if $(SDK),-sdk $(SDK))
BUILD := .build

# Scene is step 70's camera and studio, copied; the rest is this step's.
SOURCES := Sources/Printer.swift Sources/Scene.swift Sources/Render.swift Sources/Inset.swift Sources/Annotate.swift

$(BUILD)/printer: $(SOURCES) Sources/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(SOURCES) Sources/main.swift -o $@

$(BUILD)/tests: $(SOURCES) Tests/Harness.swift Tests/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(SOURCES) Tests/Harness.swift Tests/main.swift -o $@

run: $(BUILD)/printer
	@mkdir -p renders
	@$< 1920 1080 5 renders/laserjet.png
	cp renders/laserjet.png ../showcase/laserjet.png

preview: $(BUILD)/printer
	@mkdir -p renders
	@$< 960 540 2 renders/preview.png

test: $(BUILD)/tests
	@$<

# Each mutant must make the suite FAIL. A mutant that passes is a test missing.
mutants: $(BUILD)/tests
	@ok=1; \
	for m in wrongDPI wrongSize tonerTooBig; do \
	  printf '=== mutation: %s ===\n' "$$m"; \
	  if PRINTER_MUTANT=$$m $< > $(BUILD)/mutant-$$m.log 2>&1; then \
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
