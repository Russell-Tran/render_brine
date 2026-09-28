# Step 37: an ant taps egg yolk — step 36's still, slightly and slowly animated
#
#   make run        render 1920 × 1080 at 9 samples per pixel → renders/ant_yolk_tapping.gif
#   make stills     a few frames across one tap, as PNGs, to look at
#   make test       build and run the tests
#   make mutants    break the scene seven ways; the tests must catch each
#   make typecheck  prove no expression is slow to type-check
#
# If swiftc says the SDK doesn't match the compiler, point it at one that does:
#   make test SDK=$(xcrun --show-sdk-path)

SWIFTC := swiftc -O -framework CoreGraphics -framework CoreText -framework ImageIO $(if $(SDK),-sdk $(SDK))
BUILD := .build
# The GIF encoder is step 8's, as in every animated step since — copied here
# with one expression split, because Swift 5.10 cannot type-check it in time.
GIF := Sources/GIF.swift
SOURCES := $(GIF) Sources/Anatomy.swift Sources/Food.swift Sources/Chemistry.swift Sources/Motion.swift \
           Sources/Render.swift Sources/Annotate.swift

$(BUILD)/ant_yolk_tapping: $(SOURCES) Sources/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(SOURCES) Sources/main.swift -o $@

$(BUILD)/tests: $(SOURCES) Tests/Harness.swift Tests/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(SOURCES) Tests/Harness.swift Tests/main.swift -o $@

run: $(BUILD)/ant_yolk_tapping
	@$<
	@mkdir -p ../showcase && cp renders/ant_yolk_tapping.gif ../showcase/ant_yolk_tapping.gif

stills: $(BUILD)/ant_yolk_tapping
	@$< stills

test: $(BUILD)/tests
	@$<

# Each mutant must make the suite FAIL. A mutant that passes is a test missing.
mutants: $(BUILD)/tests
	@for m in segments13 hover tastePores volatiles formula press rewind; do \
	  if ANT_MUTANT=$$m $< > /dev/null 2>&1; then echo "mutant $$m ESCAPED"; exit 1; \
	  else echo "mutant $$m caught"; fi; \
	done

typecheck:
	$(SWIFTC) -typecheck -Xfrontend -warn-long-expression-type-checking=5 \
	  $(SOURCES) Sources/main.swift

clean:
	rm -rf $(BUILD)

.PHONY: run stills test mutants typecheck clean
