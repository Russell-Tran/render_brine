# Step 69: one black garden ant bites a piece of peeled Golden Delicious
# apple — its jaws open and close on the cut edge — with an inset of where the
# juice goes and where the bits of flesh go. A forward-only looping GIF.
#
#   make run        render 1600 × 900 → renders/ant_bites_apple.gif, copied to showcase/
#   make stills     a few frames across one bite, as PNGs, to look at
#   make test       build and run the tests
#   make mutants    break the scene eight ways; the tests must catch each
#   make typecheck  prove no expression is slow to type-check
#
# If swiftc says the SDK doesn't match the compiler, point it at one that does:
#   make test SDK=$(xcrun --show-sdk-path)

SWIFTC := swiftc -O -framework CoreGraphics -framework CoreText -framework ImageIO $(if $(SDK),-sdk $(SDK))
BUILD := .build
# The GIF encoder is step 8's, as in every animated step since — copied here
# with one expression split, because Swift 5.10 cannot type-check it in time.
GIF := Sources/GIF.swift
SOURCES := $(GIF) Sources/Anatomy.swift Sources/Food.swift Sources/Motion.swift Sources/Pose.swift \
           Sources/Gut.swift Sources/Render.swift Sources/Annotate.swift

$(BUILD)/ant_bites_apple: $(SOURCES) Sources/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(SOURCES) Sources/main.swift -o $@

$(BUILD)/tests: $(SOURCES) Tests/Harness.swift Tests/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(SOURCES) Tests/Harness.swift Tests/main.swift -o $@

run: $(BUILD)/ant_bites_apple
	@$<
	@mkdir -p ../showcase && cp renders/ant_bites_apple.gif ../showcase/ant_bites_apple.gif

stills: $(BUILD)/ant_bites_apple
	@$< stills

test: $(BUILD)/tests
	@$<

# Each mutant must make the suite FAIL. A mutant that passes is a test missing.
mutants: $(BUILD)/tests
	@for m in segments13 rewind biteThrough biteShort stillBite solidsToCrop press pieceSize; do \
	  if ANT_MUTANT=$$m $< > /dev/null 2>&1; then echo "mutant $$m ESCAPED"; exit 1; \
	  else echo "mutant $$m caught"; fi; \
	done

typecheck:
	$(SWIFTC) -typecheck -Xfrontend -warn-long-expression-type-checking=5 \
	  $(SOURCES) Sources/main.swift

clean:
	rm -rf $(BUILD)

.PHONY: run stills test mutants typecheck clean
