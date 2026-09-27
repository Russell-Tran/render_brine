# Step 23: secretory diarrhoea (cholera-type) — the same stretch of gut twice,
# normal above and cholera below, as animated cutaways.
#
#   make run        render renders/diarrhea.gif
#   make stills     write a few frames as PNGs to renders/
#   make test       build and run the tests
#   make mutants    break the physiology three ways; the tests must catch each
#   make typecheck  prove no expression is slow to type-check
#
# If swiftc says the SDK doesn't match the compiler, point it at one that does:
#   make test SDK=/Library/Developer/CommandLineTools/SDKs/MacOSX26.sdk

SWIFTC := swiftc -O -framework CoreGraphics -framework CoreText -framework ImageIO \
          $(if $(SDK),-sdk $(SDK))
BUILD := .build
# The GIF encoder is step 8's, as in every animated step since.
GIF := ../step008_dna/Sources/GIF.swift
OWN := Sources/Gut.swift Sources/Particles.swift Sources/Render.swift Sources/Overlay.swift
SOURCES := $(GIF) $(OWN)

$(BUILD)/diarrhea: $(SOURCES) Sources/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(SOURCES) Sources/main.swift -o $@

$(BUILD)/tests: $(SOURCES) Tests/Harness.swift Tests/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(SOURCES) Tests/Harness.swift Tests/main.swift -o $@

run: $(BUILD)/diarrhea
	@$<

stills: $(BUILD)/diarrhea
	@$< stills

test: $(BUILD)/tests
	@$<

# Each mutant must make the suite FAIL. A mutant that passes is a test missing.
mutants: $(BUILD)/tests
	@for m in tips na-through-cells lazy-colon; do \
	  if GUT_MUTANT=$$m $< > /dev/null 2>&1; then echo "mutant $$m ESCAPED"; exit 1; \
	  else echo "mutant $$m caught"; fi; \
	done

typecheck:
	$(SWIFTC) -typecheck -Xfrontend -warn-long-expression-type-checking=5 \
	  $(SOURCES) Sources/main.swift

clean:
	rm -rf $(BUILD)

.PHONY: run stills test mutants typecheck clean
