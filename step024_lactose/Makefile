# Step 24: lactose intolerance — osmotic diarrhoea. The same stretch of gut as
# step 23, twice, after two glasses of milk: lactase working above, lactase
# missing below.
#
#   make run        render renders/lactose.gif
#   make stills     write a few frames as PNGs to renders/
#   make test       build and run the tests
#   make mutants    break the physiology three ways; the tests must catch each
#   make typecheck  prove no expression is slow to type-check
#
# If swiftc says the SDK doesn't match the compiler, point it at one that does:
#   make test SDK=/Library/Developer/CommandLineTools/SDKs/MacOSX26.sdk
#
# Step 23's sources are COPIED here, not referenced. Its chemistry is in every
# file — the species list, the emitters, the stool table, the fluid colour,
# the close-up's CFTR and NKCC1 — so sharing them would have meant threading
# options through all five files of a step that has just shipped, and
# re-proving its picture byte for byte afterwards. The anatomy (tubes, villi,
# crypts, valve, wave, flow) is carried over unchanged; step 23 is untouched.

SWIFTC := swiftc -O -framework CoreGraphics -framework CoreText -framework ImageIO \
          $(if $(SDK),-sdk $(SDK))
BUILD := .build
# The GIF encoder is step 8's, as in every animated step since.
GIF := ../step008_dna/Sources/GIF.swift
OWN := Sources/Gut.swift Sources/Particles.swift Sources/Render.swift Sources/Overlay.swift
SOURCES := $(GIF) $(OWN)

$(BUILD)/lactose: $(SOURCES) Sources/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(SOURCES) Sources/main.swift -o $@

$(BUILD)/tests: $(SOURCES) Tests/Harness.swift Tests/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(SOURCES) Tests/Harness.swift Tests/main.swift -o $@

run: $(BUILD)/lactose
	@$<

stills: $(BUILD)/lactose
	@$< stills

test: $(BUILD)/tests
	@$<

# Each mutant must make the suite FAIL. A mutant that passes is a test missing.
mutants: $(BUILD)/tests
	@for m in lactase-working chloride-on low-gap; do \
	  if GUT_MUTANT=$$m $< > /dev/null 2>&1; then echo "mutant $$m ESCAPED"; exit 1; \
	  else echo "mutant $$m caught"; fi; \
	done

typecheck:
	$(SWIFTC) -typecheck -Xfrontend -warn-long-expression-type-checking=5 \
	  $(SOURCES) Sources/main.swift

clean:
	rm -rf $(BUILD)

.PHONY: run stills test mutants typecheck clean
