# Step 48: step 47's garden pea, now in flower and setting pods: Mendel's white
# flowers above, pods below — a gradient along the stem by node age — on the
# same climbing vine, net, camera and loop.
#
#   make run        render renders/pea_flowers_pods.gif and copy it to ../showcase/
#   make measure    render ten frames only, to see what a frame costs in the GIF
#   make stills     write four frames across the loop as PNGs, to look at
#   make test       build and run the tests
#   make mutants    colour the flowers Mendel's purple, put the pods above the
#                   flowers, rewind the loop, and step 47's mutants (sweet pea
#                   stipules, one tendril, no perversion, stem tendrils) — and
#                   check a test catches each
#   make typecheck  prove no expression is slow to type-check
#
# If swiftc says the SDK doesn't match the compiler, point it at one that does:
#   make test SDK=/Library/Developer/CommandLineTools/SDKs/MacOSX26.sdk

SWIFTC := swiftc -O -framework CoreGraphics $(if $(SDK),-sdk $(SDK))
BUILD := .build

# Plant.swift, Kernel.swift, Render.swift and Picture.swift are the shared
# climbing-plant machinery from steps 45-47; step 48 adds petals and pods to
# Plant, Kernel and Render (Picture is unchanged). The GIF
# encoder is step 8's, as in every animated step since.
SOURCES := Sources/Plant.swift Sources/Kernel.swift Sources/Render.swift Sources/Picture.swift Sources/Pea.swift
GIF := ../step008_dna/Sources/GIF.swift

$(BUILD)/pea_flowers_pods: $(SOURCES) $(GIF) Sources/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(SOURCES) $(GIF) Sources/main.swift -o $@

$(BUILD)/tests: $(SOURCES) $(GIF) Tests/Harness.swift Tests/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(SOURCES) $(GIF) Tests/Harness.swift Tests/main.swift -o $@

run: $(BUILD)/pea_flowers_pods
	@mkdir -p renders
	@$< 480 624 2
	cp renders/pea_flowers_pods.gif ../showcase/pea_flowers_pods.gif

measure: $(BUILD)/pea_flowers_pods
	@mkdir -p renders
	@$< 480 624 2 10

stills: $(BUILD)/pea_flowers_pods
	@$< stills 480 624 2

test: $(BUILD)/tests
	@$<

# Each mutant must make the suite FAIL. A mutant that passes is a test missing.
mutants: $(BUILD)/tests
	@ok=1; \
	for m in purple pods_above_flowers rewind small_stipules one_tendril no_perversion stem_tendril; do \
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
