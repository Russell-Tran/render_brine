# Step 57: step 44's three black garden ant workers walking a pheromone trail
# in single file — now across a planed Scots pine board — a forward-only
# looping GIF.
#
#   make run        render renders/ant_trail_wood_wood.gif and copy it to showcase/
#   make stills     write a few frames as PNGs to renders/, to look at
#   make test       build and run the tests
#   make mutants    break the gait, the loop, the wood and the motion six
#                   ways; the tests must catch each
#   make typecheck  prove no expression is slow to type-check
#
# If swiftc says the SDK doesn't match the compiler, point it at one that does:
#   make test SDK=$(xcrun --show-sdk-path)

SWIFTC := swiftc -O -framework CoreGraphics $(if $(SDK),-sdk $(SDK))
BUILD := .build
# The ant is step 32's (Anatomy.swift, made poseable); the GIF encoder is
# step 8's, as in every animated step since — copied here (Sources/GIF.swift)
# with one expression split so Swift 5.10 can type-check it, as step 51 did.
GIF := Sources/GIF.swift
SOURCES := $(GIF) Sources/Anatomy.swift Sources/Wood.swift Sources/Walk.swift Sources/Render.swift Sources/Annotate.swift

$(BUILD)/ant_trail_wood: $(SOURCES) Sources/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(SOURCES) Sources/main.swift -o $@

$(BUILD)/tests: $(SOURCES) Tests/Harness.swift Tests/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(SOURCES) Tests/Harness.swift Tests/main.swift -o $@

run: $(BUILD)/ant_trail_wood
	@$<
	cp renders/ant_trail_wood.gif ../showcase/ant_trail_wood.gif

stills: $(BUILD)/ant_trail_wood
	@$< stills

test: $(BUILD)/tests
	@$<

# Each mutant must make the suite FAIL. A mutant that passes is a test missing.
mutants: $(BUILD)/tests
	@for m in nonTripod sliding rewind sink swapRoughness frozen; do \
	  if TRAIL_MUTANT=$$m $< > /dev/null 2>&1; then echo "mutant $$m ESCAPED"; exit 1; \
	  else echo "mutant $$m caught"; fi; \
	done

typecheck:
	$(SWIFTC) -typecheck -Xfrontend -warn-long-expression-type-checking=5 \
	  $(SOURCES) Sources/main.swift

clean:
	rm -rf $(BUILD)

.PHONY: run stills test mutants typecheck clean
