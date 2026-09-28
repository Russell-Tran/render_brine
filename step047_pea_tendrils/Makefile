# Step 47: a garden pea climbing netting with its tendrils — a time-lapse loop,
# step 46's view, so the sweet pea and the pea you eat play side by side.
#
#   make run        render renders/pea_tendrils.gif and copy it to ../showcase/
#   make measure    render ten frames only, to see what a frame costs in the GIF
#   make stills     write four frames across the loop as PNGs, to look at
#   make test       build and run the tests
#   make mutants    shrink the stipules to a sweet pea's, give each leaf one
#                   tendril, coil the anchored tendril all one way, grow
#                   tendrils from the stem, rewind the loop — and check a test
#                   catches each
#   make typecheck  prove no expression is slow to type-check
#
# If swiftc says the SDK doesn't match the compiler, point it at one that does:
#   make test SDK=/Library/Developer/CommandLineTools/SDKs/MacOSX26.sdk

SWIFTC := swiftc -O -framework CoreGraphics $(if $(SDK),-sdk $(SDK))
BUILD := .build

# Plant.swift, Kernel.swift, Render.swift and Picture.swift are the shared
# climbing-plant machinery, copied unchanged from steps 45 and 46. The GIF
# encoder is step 8's, as in every animated step since.
SOURCES := Sources/Plant.swift Sources/Kernel.swift Sources/Render.swift Sources/Picture.swift Sources/Pea.swift
GIF := ../step008_dna/Sources/GIF.swift

$(BUILD)/pea_tendrils: $(SOURCES) $(GIF) Sources/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(SOURCES) $(GIF) Sources/main.swift -o $@

$(BUILD)/tests: $(SOURCES) $(GIF) Tests/Harness.swift Tests/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(SOURCES) $(GIF) Tests/Harness.swift Tests/main.swift -o $@

run: $(BUILD)/pea_tendrils
	@mkdir -p renders
	@$< 480 624 2
	cp renders/pea_tendrils.gif ../showcase/pea_tendrils.gif

measure: $(BUILD)/pea_tendrils
	@mkdir -p renders
	@$< 480 624 2 10

stills: $(BUILD)/pea_tendrils
	@$< stills 480 624 2

test: $(BUILD)/tests
	@$<

# Each mutant must make the suite FAIL. A mutant that passes is a test missing.
mutants: $(BUILD)/tests
	@ok=1; \
	for m in small_stipules one_tendril no_perversion stem_tendril rewind; do \
	  printf '=== mutation: %s ===\n' "$$m"; \
	  if TENDRIL_MUTANT=$$m $< > $(BUILD)/mutant-$$m.log 2>&1; then \
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
