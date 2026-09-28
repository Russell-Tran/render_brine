# Step 52: the fibres of a cotton sock — knit, yarn, one fibre, its cut end,
# and cellulose. One still, 1920 × 1080.
#
#   make run        render 1920 × 1080 at 9 samples per pixel → renders/cotton_sock.png,
#                   copied to ../showcase/cotton_sock.png
#   make preview    960 × 540, one sample per pixel → renders/preview.png
#   make test       build and run the tests
#   make mutants    break the science four ways; the tests must catch each
#   make typecheck  prove no expression is slow to type-check
#
# If swiftc says the SDK doesn't match the compiler, point it at one that does:
#   make test SDK=/Library/Developer/CommandLineTools/SDKs/MacOSX26.sdk

SWIFTC := swiftc -O -framework CoreGraphics $(if $(SDK),-sdk $(SDK))
BUILD := .build
SOURCES := Sources/Fibre.swift Sources/Yarn.swift Sources/Knit.swift Sources/Cellulose.swift Sources/Render.swift Sources/Annotate.swift

$(BUILD)/cotton_sock: $(SOURCES) Sources/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(SOURCES) Sources/main.swift -o $@

$(BUILD)/tests: $(SOURCES) Tests/Harness.swift Tests/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(SOURCES) Tests/Harness.swift Tests/main.swift -o $@

run: $(BUILD)/cotton_sock
	@$< 1920 1080 3 renders/cotton_sock.png
	cp renders/cotton_sock.png ../showcase/cotton_sock.png

preview: $(BUILD)/cotton_sock
	@$< 960 540 1 renders/preview.png

test: $(BUILD)/tests
	@$<

# Each mutant must make the suite FAIL. A mutant that passes is a test missing.
mutants: $(BUILD)/tests
	@ok=1; \
	for m in roundFibre alphaLinks noReversal gappyKnit; do \
	  printf '=== mutation: %s ===\n' "$$m"; \
	  if COTTON_MUTANT=$$m $< > $(BUILD)/mutant-$$m.log 2>&1; then \
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

.PHONY: run preview test mutants typecheck clean
