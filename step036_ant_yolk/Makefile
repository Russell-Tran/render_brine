# Step 36: an ant smells and tastes hard-boiled egg yolk, one still
#
#   make run        render 1920 × 1080 at 9 samples per pixel → renders/ant_yolk.png
#   make test       build and run the tests
#   make mutants    break the scene five ways; the tests must catch each
#   make typecheck  prove no expression is slow to type-check
#
# If swiftc says the SDK doesn't match the compiler, point it at one that does:
#   make test SDK=/Library/Developer/CommandLineTools/SDKs/MacOSX26.sdk

SWIFTC := swiftc -O -framework CoreGraphics $(if $(SDK),-sdk $(SDK))
BUILD := .build
SOURCES := Sources/Anatomy.swift Sources/Food.swift Sources/Chemistry.swift Sources/Render.swift Sources/Annotate.swift

$(BUILD)/ant: $(SOURCES) Sources/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(SOURCES) Sources/main.swift -o $@

$(BUILD)/tests: $(SOURCES) Tests/Harness.swift Tests/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(SOURCES) Tests/Harness.swift Tests/main.swift -o $@

run: $(BUILD)/ant
	@$< 1920 1080 3 renders/ant_yolk.png
	cp renders/ant_yolk.png ../showcase/ant_yolk.png

preview: $(BUILD)/ant
	@$< 960 540 1 renders/preview.png

test: $(BUILD)/tests
	@$<

# Each mutant must make the suite FAIL. A mutant that passes is a test missing.
mutants: $(BUILD)/tests
	@for m in segments13 hover tastePores volatiles formula; do \
	  if ANT_MUTANT=$$m $< > /dev/null 2>&1; then echo "mutant $$m ESCAPED"; exit 1; \
	  else echo "mutant $$m caught"; fi; \
	done

typecheck:
	$(SWIFTC) -typecheck -Xfrontend -warn-long-expression-type-checking=5 \
	  $(SOURCES) Sources/main.swift

clean:
	rm -rf $(BUILD)

.PHONY: run preview test mutants typecheck clean
