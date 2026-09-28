# Step 63: the plastic toy cow comes to life — frozen, it wakes,
# looks round, walks once round a circle and settles back into its pose.
#
#   make run        render renders/toy_cow_alive.gif and copy it to ../showcase/
#   make measure    render ten frames only, to see what a frame costs in the GIF
#   make stills     write frames across the loop as PNGs, to look at
#   make test       build and run the tests
#   make mutants    slide a planted foot, rewind the loop, freeze the toy, trot
#                   instead of walk — and check a test catches each
#   make typecheck  prove no expression is slow to type-check
#
# Always build against the command-line tools' SDK here:
#   make test SDK=/Library/Developer/CommandLineTools/SDKs/MacOSX26.sdk

SDK ?= /Library/Developer/CommandLineTools/SDKs/MacOSX26.sdk
SWIFTC := swiftc -O -framework CoreGraphics $(if $(SDK),-sdk $(SDK))
BUILD := .build

# Plastic, Toy, Scene, Render, Still and Annotate are step 58's shared toy
# renderer as step 62 carries it; Walk and Overlay are step 59's shared animation (steps 59, 61,
# 63 carry copies); Cow is step 62's animal; CowWalk its gait. The
# GIF encoder is step 8's, as in every animated step.
SOURCES := Sources/Plastic.swift Sources/Toy.swift Sources/Scene.swift Sources/Render.swift Sources/Still.swift \
           Sources/Annotate.swift Sources/Cow.swift Sources/Walk.swift Sources/Overlay.swift Sources/CowWalk.swift
GIF := ../step008_dna/Sources/GIF.swift

$(BUILD)/toy_alive: $(SOURCES) $(GIF) Sources/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(SOURCES) $(GIF) Sources/main.swift -o $@

$(BUILD)/tests: $(SOURCES) $(GIF) Tests/Harness.swift Tests/Common.swift Tests/Hooves.swift Tests/Anim.swift Tests/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(SOURCES) $(GIF) Tests/Harness.swift Tests/Common.swift Tests/Hooves.swift Tests/Anim.swift Tests/main.swift -o $@

run: $(BUILD)/toy_alive
	@mkdir -p renders
	@$<
	cp renders/toy_cow_alive.gif ../showcase/toy_cow_alive.gif

measure: $(BUILD)/toy_alive
	@mkdir -p renders
	@TOY_LIMIT=10 TOY_FIRST=200 TOY_OUT=renders/measure.gif $<

stills: $(BUILD)/toy_alive
	@mkdir -p renders
	@$< stills

test: $(BUILD)/tests
	@$<

# Each mutant must make the suite FAIL. A mutant that passes is a test missing.
mutants: $(BUILD)/tests
	@ok=1; \
	for m in sliding_feet rewind frozen wrongGait; do \
	  printf '=== mutation: %s ===\n' "$$m"; \
	  if TOY_MUTANT=$$m $< > $(BUILD)/mutant-$$m.log 2>&1; then \
	    printf '  NOT CAUGHT — no test failed\n'; ok=0; \
	  else \
	    grep '^  FAIL' $(BUILD)/mutant-$$m.log || true; \
	  fi; \
	done; \
	[ $$ok -eq 1 ] || exit 1
	@echo "every mutation was caught"

typecheck:
	$(SWIFTC) -typecheck -Xfrontend -warn-long-expression-type-checking=5 \
	  $(SOURCES) $(GIF) Sources/main.swift

clean:
	rm -rf $(BUILD)

.PHONY: run measure stills test mutants typecheck clean
