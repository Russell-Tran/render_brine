# Step 56: seven black garden ant workers walking a pheromone trail in single
# file, a forward-only looping GIF. Step 44's scene, the file widened.
#
#   make run        render renders/ant_trail_7.gif and copy it to showcase/
#   make stills     write a few frames as PNGs to renders/, to look at
#   make test       build and run the tests
#   make mutants    break the gait, the loop and the motion's visibility;
#                   the tests must catch each
#   make typecheck  prove no expression is slow to type-check
#
# On the laptop, point swiftc at the SDK that matches the compiler:
#   make test SDK=$(xcrun --show-sdk-path)

SWIFTC := swiftc -O -framework CoreGraphics $(if $(SDK),-sdk $(SDK))
BUILD := .build
# The ant is step 32's (Anatomy.swift, made poseable in step 44); the GIF
# encoder is step 8's, copied here with the one expression split that Swift
# 5.10 needs (as step 51's and step 55's copies).
GIF := Sources/GIF.swift
SOURCES := $(GIF) Sources/Anatomy.swift Sources/Walk.swift Sources/Render.swift Sources/Annotate.swift
NAME := ant_trail_7

$(BUILD)/ant_trail: $(SOURCES) Sources/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(SOURCES) Sources/main.swift -o $@

$(BUILD)/tests: $(SOURCES) Tests/Harness.swift Tests/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(SOURCES) Tests/Harness.swift Tests/main.swift -o $@

run: $(BUILD)/ant_trail
	@mkdir -p renders
	@$<
	cp renders/$(NAME).gif ../showcase/$(NAME).gif

stills: $(BUILD)/ant_trail
	@mkdir -p renders
	@$< stills

test: $(BUILD)/tests
	@$<

# Each mutant must make the suite FAIL. A mutant that passes is a test missing.
mutants: $(BUILD)/tests
	@for m in nonTripod sliding rewind subpixel; do \
	  if TRAIL_MUTANT=$$m $< > /dev/null 2>&1; then echo "mutant $$m ESCAPED"; exit 1; \
	  else echo "mutant $$m caught"; fi; \
	done

typecheck:
	$(SWIFTC) -typecheck -Xfrontend -warn-long-expression-type-checking=5 \
	  $(SOURCES) Sources/main.swift

clean:
	rm -rf $(BUILD)

.PHONY: run stills test mutants typecheck clean
