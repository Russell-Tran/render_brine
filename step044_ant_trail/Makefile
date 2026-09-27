# Step 44: three black garden ant workers walking a pheromone trail in single
# file, a forward-only looping GIF.
#
#   make run        render renders/ant_trail.gif and copy it to showcase/
#   make stills     write a few frames as PNGs to renders/, to look at
#   make test       build and run the tests
#   make mutants    break the gait and the loop three ways; the tests must
#                   catch each
#   make typecheck  prove no expression is slow to type-check
#
# If swiftc says the SDK doesn't match the compiler, point it at one that does:
#   make test SDK=/Library/Developer/CommandLineTools/SDKs/MacOSX26.sdk

SWIFTC := swiftc -O -framework CoreGraphics $(if $(SDK),-sdk $(SDK))
BUILD := .build
# The ant is step 32's (Anatomy.swift, made poseable); the GIF encoder is
# step 8's, as in every animated step since.
GIF := ../step008_dna/Sources/GIF.swift
SOURCES := $(GIF) Sources/Anatomy.swift Sources/Walk.swift Sources/Render.swift Sources/Annotate.swift

$(BUILD)/ant_trail: $(SOURCES) Sources/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(SOURCES) Sources/main.swift -o $@

$(BUILD)/tests: $(SOURCES) Tests/Harness.swift Tests/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(SOURCES) Tests/Harness.swift Tests/main.swift -o $@

run: $(BUILD)/ant_trail
	@$<
	cp renders/ant_trail.gif ../showcase/ant_trail.gif

stills: $(BUILD)/ant_trail
	@$< stills

test: $(BUILD)/tests
	@$<

# Each mutant must make the suite FAIL. A mutant that passes is a test missing.
mutants: $(BUILD)/tests
	@for m in nonTripod sliding rewind; do \
	  if TRAIL_MUTANT=$$m $< > /dev/null 2>&1; then echo "mutant $$m ESCAPED"; exit 1; \
	  else echo "mutant $$m caught"; fi; \
	done

typecheck:
	$(SWIFTC) -typecheck -Xfrontend -warn-long-expression-type-checking=5 \
	  $(SOURCES) Sources/main.swift

clean:
	rm -rf $(BUILD)

.PHONY: run stills test mutants typecheck clean
