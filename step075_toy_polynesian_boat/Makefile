# Step 75: a plastic toy voyaging canoe (waʻa kaulua), one still
#
#   make run        render 1920 × 1080 at 25 samples per pixel → renders/toy_polynesian_boat.png
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

# Plastic, Toy, Scene, Render, Still and Annotate are the toy-boat renderer
# (steps 73–75 each carry a copy, grown from steps 58–63's: a hull and a
# profile plate added, the animal rig taken out); Canoe.swift is this
# step's boat.
SOURCES := Sources/Plastic.swift Sources/Toy.swift Sources/Scene.swift Sources/Render.swift Sources/Still.swift \
           Sources/Annotate.swift Sources/Canoe.swift

$(BUILD)/toy: $(SOURCES) Sources/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(SOURCES) Sources/main.swift -o $@

$(BUILD)/tests: $(SOURCES) Tests/Harness.swift Tests/Common.swift Tests/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(SOURCES) Tests/Harness.swift Tests/Common.swift Tests/main.swift -o $@

run: $(BUILD)/toy
	@mkdir -p renders
	@$< 1920 1080 5 renders/toy_polynesian_boat.png
	cp renders/toy_polynesian_boat.png ../showcase/toy_polynesian_boat.png

preview: $(BUILD)/toy
	@mkdir -p renders
	@$< 960 540 2 renders/preview.png

test: $(BUILD)/tests
	@$<

# Each mutant must make the suite FAIL. A mutant that passes is a test missing.
mutants: $(BUILD)/tests
	@ok=1; \
	for m in dielectricZero noSeam wrongSize singleHull; do \
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
