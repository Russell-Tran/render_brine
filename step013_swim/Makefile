# Step 13: one brine shrimp swimming — a transmitted-light (brightfield)
# micrograph of an adult Artemia, where the background IS the lamp and every
# tone in the animal is light that got through it.
#
#   make run        render renders/swim.gif
#   make stills     write a few frames as PNGs
#   make bench      time brute force against the uniform grid
#   make test       build and run the Swift tests
#   make mutants    break the union, the mailbox, the phase lag and the radius
#                   clamp in turn, and check a test catches each one
#
# If swiftc says the SDK doesn't match the compiler, point it at one that does:
#   make test SDK=/Library/Developer/CommandLineTools/SDKs/MacOSX26.sdk
# On the M3 Max laptop, `SDK=$(xcrun --show-sdk-path)` is the right one.

# Command-line tools must link CoreGraphics for MTLCreateSystemDefaultDevice()
# to find the GPU (per Apple's docs). CoreText draws the labels, ImageIO the
# PNG stills.
SWIFTC := swiftc -O -framework CoreGraphics -framework CoreText -framework ImageIO \
          $(if $(SDK),-sdk $(SDK))
BUILD := .build
SOURCES := Sources/Artemia.swift Sources/Render.swift Sources/Overlay.swift Sources/GIF.swift

$(BUILD)/swim: $(SOURCES) Sources/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(filter %.swift,$^) -o $@

$(BUILD)/tests: $(SOURCES) Tests/Harness.swift Tests/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(filter %.swift,$^) -o $@

run: $(BUILD)/swim
	@$<

stills: $(BUILD)/swim
	@$< stills

bench: $(BUILD)/swim
	@$< bench

test: $(BUILD)/tests
	@$<

# The mutation check. Each name breaks exactly one thing; the suite must go
# red for every one of them, and the failing test's name is printed so it is
# on the record which test did the catching.
mutants: $(BUILD)/tests
	@for m in union mailbox phase clamp; do \
	  printf '=== mutation: %s ===\n' "$$m"; \
	  if SWIM_MUTATE=$$m $< > /dev/null 2>&1; then \
	    printf '  NOT CAUGHT — no test failed\n'; exit 1; \
	  else \
	    SWIM_MUTATE=$$m $< 2>&1 | grep '^  FAIL' || true; \
	  fi; \
	done
	@echo "every mutation was caught"

# The type checker on the laptop's older toolchain gives up on long chained
# arithmetic, so every expression here is written in typed steps. This proves
# it: nothing may take longer than 5 ms to type-check.
typecheck:
	$(SWIFTC) -typecheck -Xfrontend -warn-long-expression-type-checking=5 \
	  $(SOURCES) Sources/main.swift

clean:
	rm -rf $(BUILD) renders

.PHONY: run stills bench test mutants typecheck clean
