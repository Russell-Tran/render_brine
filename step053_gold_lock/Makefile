# Step 53: a gold-plated padlock, one still
#
#   make run        render 1920 × 1080 at 16 samples per pixel → renders/gold_lock.png
#   make preview    960 × 540 at 1 sample → renders/preview.png
#   make test       build and run the tests
#   make mutants    break the physics four ways; the tests must catch each
#   make typecheck  prove no expression is slow to type-check
#
# If swiftc says the SDK doesn't match the compiler, point it at one that does:
#   make test SDK=/Library/Developer/CommandLineTools/SDKs/MacOSX26.sdk

SWIFTC := swiftc -O -framework CoreGraphics $(if $(SDK),-sdk $(SDK))
BUILD := .build
SOURCES := Sources/Optics.swift Sources/Lock.swift Sources/Render.swift Sources/Annotate.swift

$(BUILD)/lock: $(SOURCES) Sources/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(SOURCES) Sources/main.swift -o $@

$(BUILD)/tests: $(SOURCES) Tests/Harness.swift Tests/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(SOURCES) Tests/Harness.swift Tests/main.swift -o $@

run: $(BUILD)/lock
	@$< 1920 1080 4 renders/gold_lock.png
	cp renders/gold_lock.png ../showcase/gold_lock.png

preview: $(BUILD)/lock
	@$< 960 540 1 renders/preview.png

test: $(BUILD)/tests
	@$<

# Each mutant must make the suite FAIL. A mutant that passes is a test missing.
mutants: $(BUILD)/tests
	@for m in silver dielectric noNickel thickGold; do \
	  if LOCK_MUTANT=$$m $< > /dev/null 2>&1; then echo "mutant $$m ESCAPED"; exit 1; \
	  else echo "mutant $$m caught"; fi; \
	done

typecheck:
	$(SWIFTC) -typecheck -Xfrontend -warn-long-expression-type-checking=5 \
	  $(SOURCES) Sources/main.swift

clean:
	rm -rf $(BUILD)

.PHONY: run preview test mutants typecheck clean
