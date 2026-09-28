# Step 66: the upper arch, one still
#
#   make run        render 1920 × 1080 at 16 samples per pixel → renders/upper_arch.png,
#                   and copy it to ../showcase/upper_arch.png
#   make test       build and run the tests
#   make mutants    break the anatomy three ways and the shading two; the tests
#                   must catch each
#   make typecheck  prove no expression is slow to type-check
#
# If swiftc says the SDK doesn't match the compiler, point it at one that does:
#   make test SDK=/Library/Developer/CommandLineTools/SDKs/MacOSX26.sdk
#
# The scene is step 20's, copied (not shared) and turned into the upper jaw;
# step 20 stays exactly as committed.

SWIFTC := swiftc -O -framework CoreGraphics $(if $(SDK),-sdk $(SDK))
BUILD := .build
SOURCES := Sources/Anatomy.swift Sources/Render.swift

$(BUILD)/upper_arch: $(SOURCES) Sources/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(SOURCES) Sources/main.swift -o $@

$(BUILD)/tests: $(SOURCES) Tests/Harness.swift Tests/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(SOURCES) Tests/Harness.swift Tests/main.swift -o $@

run: $(BUILD)/upper_arch
	@$< 1920 1080 4 renders/upper_arch.png
	@cp renders/upper_arch.png ../showcase/upper_arch.png

test: $(BUILD)/tests
	@$<

# Each mutant must make the suite FAIL. A mutant that passes is a test missing.
# The anatomy mutants are read from UPPER_MUTANT, the shading ones from
# MOUTH_MUTANT, as step 20's were.
mutants: $(BUILD)/tests
	@ok=1; \
	for m in mandibularSizes mirrored noRugae; do \
	  printf '=== mutation: %s ===\n' "$$m"; \
	  if UPPER_MUTANT=$$m $< > $(BUILD)/mutant-$$m.log 2>&1; then \
	    printf '  NOT CAUGHT — no test failed\n'; ok=0; \
	  else grep '^  FAIL' $(BUILD)/mutant-$$m.log || true; fi; \
	done; \
	for m in flat opaque; do \
	  printf '=== mutation: %s ===\n' "$$m"; \
	  if MOUTH_MUTANT=$$m $< > $(BUILD)/mutant-$$m.log 2>&1; then \
	    printf '  NOT CAUGHT — no test failed\n'; ok=0; \
	  else grep '^  FAIL' $(BUILD)/mutant-$$m.log || true; fi; \
	done; \
	[ $$ok -eq 1 ] || exit 1
	@echo "every mutation was caught"

typecheck:
	$(SWIFTC) -typecheck -Xfrontend -warn-long-expression-type-checking=5 \
	  $(SOURCES) Sources/main.swift

clean:
	rm -rf $(BUILD)

.PHONY: run test mutants typecheck clean
