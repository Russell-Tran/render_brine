# Step 68: three black garden ants come to a piece of peeled Golden Delicious
# apple, tap its wet face, taste it and drink — a forward-only looping GIF.
#
#   make run        render 1600 × 900 → renders/ants_apple.gif, copied to showcase/
#   make stills     a few frames across the loop, as PNGs, to look at
#   make test       build and run the tests
#   make mutants    break the scene eleven ways; the tests must catch each
#   make typecheck  prove no expression is slow to type-check
#
# If swiftc says the SDK doesn't match the compiler, point it at one that does:
#   make test SDK=$(xcrun --show-sdk-path)

SWIFTC := swiftc -O -framework CoreGraphics -framework CoreText -framework ImageIO $(if $(SDK),-sdk $(SDK))
BUILD := .build
# The GIF encoder is step 8's, as in every animated step since — copied here
# with one expression split, because Swift 5.10 cannot type-check it in time.
GIF := Sources/GIF.swift
SOURCES := $(GIF) Sources/Anatomy.swift Sources/Food.swift Sources/Motion.swift Sources/Walk.swift \
           Sources/Chemistry.swift Sources/Render.swift Sources/Annotate.swift

$(BUILD)/ants_apple: $(SOURCES) Sources/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(SOURCES) Sources/main.swift -o $@

$(BUILD)/tests: $(SOURCES) Tests/Harness.swift Tests/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(SOURCES) Tests/Harness.swift Tests/main.swift -o $@

run: $(BUILD)/ants_apple
	@$<
	@mkdir -p ../showcase && cp renders/ants_apple.gif ../showcase/ants_apple.gif

stills: $(BUILD)/ants_apple
	@$< stills

test: $(BUILD)/tests
	@$<

# Each mutant must make the suite FAIL. A mutant that passes is a test missing.
mutants: $(BUILD)/tests
	@for m in segments13 sliding rewind press hover tastePores noOdour pieceSize formula dryGlossa stillTap; do \
	  if ANT_MUTANT=$$m $< > /dev/null 2>&1; then echo "mutant $$m ESCAPED"; exit 1; \
	  else echo "mutant $$m caught"; fi; \
	done

typecheck:
	$(SWIFTC) -typecheck -Xfrontend -warn-long-expression-type-checking=5 \
	  $(SOURCES) Sources/main.swift

clean:
	rm -rf $(BUILD)

.PHONY: run stills test mutants typecheck clean
