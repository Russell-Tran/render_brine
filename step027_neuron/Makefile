# Step 27: one neuron, one still
#
#   make run        render 1920 × 1080 at 9 samples per pixel → renders/neuron.png
#   make test       build and run the tests
#   make mutants    break the cell two ways; the tests must catch both
#   make typecheck  prove no expression is slow to type-check
#
# If swiftc says the SDK doesn't match the compiler, point it at one that does:
#   make test SDK=/Library/Developer/CommandLineTools/SDKs/MacOSX26.sdk

SWIFTC := swiftc -O -framework CoreGraphics $(if $(SDK),-sdk $(SDK))
BUILD := .build
SOURCES := Sources/Neuron.swift Sources/Render.swift

$(BUILD)/neuron: $(SOURCES) Sources/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(SOURCES) Sources/main.swift -o $@

$(BUILD)/tests: $(SOURCES) Tests/Harness.swift Tests/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(SOURCES) Tests/Harness.swift Tests/main.swift -o $@

run: $(BUILD)/neuron
	@$< 1920 1080 3 renders/neuron.png

test: $(BUILD)/tests
	@$<

# Each mutant must make the suite FAIL. A mutant that passes is a test missing.
mutants: $(BUILD)/tests
	@for m in twoAxons taperingAxon; do \
	  if NEURON_MUTANT=$$m $< > /dev/null 2>&1; then echo "mutant $$m ESCAPED"; exit 1; \
	  else echo "mutant $$m caught"; fi; \
	done

typecheck:
	$(SWIFTC) -typecheck -Xfrontend -warn-long-expression-type-checking=5 \
	  $(SOURCES) Sources/main.swift

clean:
	rm -rf $(BUILD)

.PHONY: run test mutants typecheck clean
