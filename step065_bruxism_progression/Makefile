# Step 65: step 20's lower arch wearing down under bruxism, the camera held
# exactly still — step 64's wear growing from nothing to year 50, then a
# dissolve back to year 0 that says it is a restart.
#
#   make run        render renders/bruxism_progression.gif
#   make measure    render the first 10 wear frames only, to size the GIF
#   make stills     write a few frames as PNGs, to look at
#   make test       build and run the tests
#   make mutants    run the wear backwards, make it too small to see, let the
#                   camera drift, draw dentine as enamel — a test must catch each
#   make typecheck  prove no expression is slow to type-check
#
# If swiftc says the SDK doesn't match the compiler, point it at one that does:
#   make test SDK=/Library/Developer/CommandLineTools/SDKs/MacOSX26.sdk

SWIFTC := swiftc -O -framework CoreGraphics $(if $(SDK),-sdk $(SDK))
BUILD := .build
# Step 64's scene, copied: the teeth, the wear, the survey, the marks. The
# GIF encoder is step 8's, as in every animated step.
SOURCES := Sources/Anatomy.swift Sources/Wear.swift Sources/Render.swift Sources/Survey.swift \
           Sources/Overlay.swift Sources/Plan.swift Sources/Frames.swift
GIF := ../step008_dna/Sources/GIF.swift

$(BUILD)/progression: $(SOURCES) $(GIF) Sources/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(SOURCES) $(GIF) Sources/main.swift -o $@

$(BUILD)/tests: $(SOURCES) Tests/Harness.swift Tests/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(SOURCES) Tests/Harness.swift Tests/main.swift -o $@

run: $(BUILD)/progression
	@$< 1280 720 3

measure: $(BUILD)/progression
	@$< measure 10 1280 720 3

stills: $(BUILD)/progression
	@$< stills 1280 720 3

test: $(BUILD)/tests
	@$<

# Each mutant must make the suite FAIL. A mutant that passes is a test missing.
mutants: $(BUILD)/tests
	@ok=1; \
	for m in unwear invisible cameraMoves noDentin; do \
	  printf '=== mutation: %s ===\n' "$$m"; \
	  if BRUX_MUTANT=$$m $< > $(BUILD)/mutant-$$m.log 2>&1; then \
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
