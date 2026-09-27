# Step 22: step 20's still with a round electric toothbrush head pressed on
# #21, the patient's left lower first premolar, at the gum line.
#
#   make run        render 1920 × 1080 at 16 samples per pixel → renders/brush.png
#   make test       build and run the tests
#   make mutants    lift the brush 1 mm off the tooth, push it 1 mm in — and
#                   check a test catches each
#   make typecheck  prove no expression is slow to type-check
#
# If swiftc says the SDK doesn't match the compiler, point it at one that does:
#   make test SDK=/Library/Developer/CommandLineTools/SDKs/MacOSX26.sdk

SWIFTC := swiftc -O -framework CoreGraphics $(if $(SDK),-sdk $(SDK))
BUILD := .build

# The mouth and its renderer are step 20's, used rather than copied, as step 21
# does. The only change step 20 took for this step is a place in its kernel
# for another step's objects (`SceneExtra`), empty by default — and with it
# empty the kernel is the same text and step 20's picture the same bytes.
SCENE := ../step020_mouth/Sources/Anatomy.swift ../step020_mouth/Sources/Render.swift
SOURCES := $(SCENE) Sources/Brush.swift

$(BUILD)/brush: $(SOURCES) Sources/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(SOURCES) Sources/main.swift -o $@

$(BUILD)/tests: $(SOURCES) Tests/Harness.swift Tests/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(SOURCES) Tests/Harness.swift Tests/main.swift -o $@

run: $(BUILD)/brush
	@$< 1920 1080 4 renders/brush.png

test: $(BUILD)/tests
	@$<

# Each mutant must make the suite FAIL. A mutant that passes is a test missing.
mutants: $(BUILD)/tests
	@ok=1; \
	for m in lift push; do \
	  printf '=== mutation: %s ===\n' "$$m"; \
	  if BRUSH_MUTANT=$$m $< > $(BUILD)/mutant-$$m.log 2>&1; then \
	    printf '  NOT CAUGHT — no test failed\n'; ok=0; \
	  else \
	    grep '^  FAIL' $(BUILD)/mutant-$$m.log || true; \
	  fi; \
	done; \
	[ $$ok -eq 1 ] || exit 1
	@echo "every mutation was caught"

typecheck:
	$(SWIFTC) -typecheck -Xfrontend -warn-long-expression-type-checking=5 \
	  $(SOURCES) Sources/main.swift

clean:
	rm -rf $(BUILD)

.PHONY: run test mutants typecheck clean
