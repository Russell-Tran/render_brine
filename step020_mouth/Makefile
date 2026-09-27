# Step 20: the lower arch, one still
#
#   make run        render 1920 × 1080 at 16 samples per pixel → renders/mouth.png
#   make test       build and run the tests
#   make mutants    break the shading two ways; the tests must catch both
#   make typecheck  prove no expression is slow to type-check
#
# If swiftc says the SDK doesn't match the compiler, point it at one that does:
#   make test SDK=/Library/Developer/CommandLineTools/SDKs/MacOSX26.sdk

SWIFTC := swiftc -O -framework CoreGraphics $(if $(SDK),-sdk $(SDK))
BUILD := .build
SOURCES := Sources/Anatomy.swift Sources/Render.swift

$(BUILD)/mouth: $(SOURCES) Sources/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(SOURCES) Sources/main.swift -o $@

$(BUILD)/tests: $(SOURCES) Tests/Harness.swift Tests/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(SOURCES) Tests/Harness.swift Tests/main.swift -o $@

run: $(BUILD)/mouth
	@$< 1920 1080 4 renders/mouth.png

test: $(BUILD)/tests
	@$<

# Each mutant must make the suite FAIL. A mutant that passes is a test missing.
mutants: $(BUILD)/tests
	@for m in flat opaque; do \
	  if MOUTH_MUTANT=$$m $< > /dev/null 2>&1; then echo "mutant $$m ESCAPED"; exit 1; \
	  else echo "mutant $$m caught"; fi; \
	done

typecheck:
	$(SWIFTC) -typecheck -Xfrontend -warn-long-expression-type-checking=5 \
	  $(SOURCES) Sources/main.swift

clean:
	rm -rf $(BUILD)

.PHONY: run test mutants typecheck clean
