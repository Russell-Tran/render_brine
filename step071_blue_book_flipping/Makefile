# Step 71: step 70's blue hardback opens, turns its pages on its own, and
# closes on its other side; then a dissolve forward to the start.
#
#   make run        render renders/blue_book_flipping.gif and copy it to ../showcase
#   make measure    ten frames only, to see what a frame costs in the GIF
#   make stills     write a few frames as PNGs, to look at
#   make test       build and run the tests
#   make mutants    break it five ways; the tests must catch each
#   make typecheck  prove no expression is slow to type-check
#
# Always build against the command-line tools' SDK here:
#   make test SDK=/Library/Developer/CommandLineTools/SDKs/MacOSX26.sdk

SDK ?= /Library/Developer/CommandLineTools/SDKs/MacOSX26.sdk
SWIFTC := swiftc -O -framework CoreGraphics $(if $(SDK),-sdk $(SDK))
BUILD := .build

# Book, Cloth, Scene and Render are step 70's, copied (not shared) so step 70
# stays exactly as committed; Flip is the motion. The GIF encoder is step 8's.
GIF := ../step008_dna/Sources/GIF.swift
SOURCES := Sources/Book.swift Sources/Cloth.swift Sources/Scene.swift Sources/Render.swift Sources/Annotate.swift Sources/Flip.swift

$(BUILD)/flip: $(SOURCES) $(GIF) Sources/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(SOURCES) $(GIF) Sources/main.swift -o $@

$(BUILD)/tests: $(SOURCES) Tests/Harness.swift Tests/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(SOURCES) Tests/Harness.swift Tests/main.swift -o $@

run: $(BUILD)/flip
	@$< 640 360 3
	@cp renders/blue_book_flipping.gif ../showcase/blue_book_flipping.gif

measure: $(BUILD)/flip
	@$< 640 360 3 10 60

stills: $(BUILD)/flip
	@$< stills 640 360 3

test: $(BUILD)/tests
	@$<

# Each mutant must make the suite FAIL. A mutant that passes is a test missing.
mutants: $(BUILD)/tests
	@ok=1; \
	for m in stretchyPage frozen rewind ghostPage noSquares; do \
	  printf '=== mutation: %s ===\n' "$$m"; \
	  if BOOK_MUTANT=$$m $< > $(BUILD)/mutant-$$m.log 2>&1; then \
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
