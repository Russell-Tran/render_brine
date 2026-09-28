# Step 29: step 25's bean flower, slightly and slowly animated — the pollen
# tube grows from its grain on the stigma down the coiled style to the ovule,
# holds, and dissolves back to the start.
#
#   make run        render renders/bean_growing.gif and copy it to ../showcase
#   make stills     first, middle and last frames as PNGs in renders/
#   make test       build and run the tests
#   make mutants    break the growth three ways (and the flower step 25's three);
#                   the tests must catch each
#   make typecheck  prove no expression is slow to type-check
#
# If swiftc says the SDK doesn't match the compiler, point it at one that does:
#   make test SDK=/Library/Developer/CommandLineTools/SDKs/MacOSX26.sdk

SWIFTC := swiftc -O -framework CoreGraphics $(if $(SDK),-sdk $(SDK))
BUILD := .build
# The flower, kernel and picture are step 25's, copied (not shared) so step 25
# stays exactly as committed; the changes are marked "Step 29". The GIF
# encoder is step 8's, as in every animated step since.
GIF := ../step008_dna/Sources/GIF.swift
SOURCES := Sources/Flower.swift Sources/Kernel.swift Sources/Render.swift Sources/Overlay.swift \
           Sources/Animation.swift

$(BUILD)/bean_growing: $(SOURCES) $(GIF) Sources/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(SOURCES) $(GIF) Sources/main.swift -o $@

$(BUILD)/tests: $(SOURCES) Tests/Harness.swift Tests/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(SOURCES) Tests/Harness.swift Tests/main.swift -o $@

run: $(BUILD)/bean_growing
	@$<
	@cp renders/bean_growing.gif ../showcase/bean_growing.gif

stills: $(BUILD)/bean_growing
	@$< still 0,25,75,125,140 renders/frame

test: $(BUILD)/tests
	@$<

# Each mutant must make the suite FAIL. A mutant that passes is a test missing.
mutants: $(BUILD)/tests
	@ok=1; \
	for m in rewind strays invisible tube_leaves open all_free; do \
	  printf '=== mutation: %s ===\n' "$$m"; \
	  if BEAN_MUTANT=$$m $< > $(BUILD)/mutant-$$m.log 2>&1; then \
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

.PHONY: run stills test mutants typecheck clean
