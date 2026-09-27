# Step 31: step 27's neuron, with one nerve impulse running out along its axon,
# slowed about 700,000 times, then rest, then the next.
#
#   make run        render renders/neuron_firing.gif (1920 × 1080)
#   make stills     write a few frames as PNGs to renders/, to look at
#   make test       build and run the tests
#   make mutants    break the cell two ways and the impulse two ways; the tests
#                   must catch all four
#   make typecheck  prove no expression is slow to type-check
#
# If swiftc says the SDK doesn't match the compiler, point it at one that does:
#   make test SDK=/Library/Developer/CommandLineTools/SDKs/MacOSX26.sdk

SWIFTC := swiftc -O -framework CoreGraphics $(if $(SDK),-sdk $(SDK))
BUILD := .build
# Neuron.swift is step 27's, copied byte for byte; Render.swift is step 27's
# with the impulse's shading added. The GIF encoder is step 8's, as in every
# animated step since.
GIF := ../step008_dna/Sources/GIF.swift
SOURCES := $(GIF) Sources/Neuron.swift Sources/Impulse.swift Sources/Render.swift

$(BUILD)/neuron_firing: $(SOURCES) Sources/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(SOURCES) Sources/main.swift -o $@

$(BUILD)/tests: $(SOURCES) Tests/Harness.swift Tests/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(SOURCES) Tests/Harness.swift Tests/main.swift -o $@

run: $(BUILD)/neuron_firing
	@$<

stills: $(BUILD)/neuron_firing
	@$< stills

test: $(BUILD)/tests
	@$<

# Each mutant must make the suite FAIL. A mutant that passes is a test missing.
mutants: $(BUILD)/tests
	@for m in twoAxons taperingAxon startsInDendrite backwards; do \
	  if NEURON_MUTANT=$$m $< > /dev/null 2>&1; then echo "mutant $$m ESCAPED"; exit 1; \
	  else echo "mutant $$m caught"; fi; \
	done

typecheck:
	$(SWIFTC) -typecheck -Xfrontend -warn-long-expression-type-checking=5 \
	  $(SOURCES) Sources/main.swift

clean:
	rm -rf $(BUILD)

.PHONY: run stills test mutants typecheck clean
