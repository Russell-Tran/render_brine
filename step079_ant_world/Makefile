# Step 79, "A minute of ant world" — PART 3: the simulation (prediction P2).
# Seeded Lasius niger workers, a nest, a sugar pile, and a pheromone field on
# the GPU (deposit, diffuse, evaporate every tick). The film (a later part)
# reads its record; nothing here draws.
#
#   make test       build and run the tests (about a minute)
#   make report     the P2 numbers: with and without pheromone, and formation
#                   times over eight seeds (a few minutes)
#   make mutants    break the simulation three ways; the tests must catch each
#   make typecheck  prove no expression is slow to type-check
#
# If swiftc says the SDK doesn't match the compiler, point it at one that does:
#   make test SDK=$(xcrun --show-sdk-path)

SWIFTC := swiftc -O -framework Metal $(if $(SDK),-sdk $(SDK))
BUILD := .build
SIM := Sources/SimConstants.swift Sources/SimRandom.swift Sources/SimField.swift Sources/SimClock.swift \
       Sources/SimWorld.swift Sources/SimRecord.swift Sources/SimMetrics.swift

$(BUILD)/tests: $(SIM) Tests/Harness.swift Tests/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(SIM) Tests/Harness.swift Tests/main.swift -o $@

test: $(BUILD)/tests
	@$<

report: $(BUILD)/tests
	@$< report

# Each mutant must make the suite FAIL. A mutant that passes is a test missing.
mutants: $(BUILD)/tests
	@for m in noPheromone rewind tightTurn; do \
	  if SIM_MUTANT=$$m $< > /dev/null 2>&1; then echo "mutant $$m ESCAPED"; exit 1; \
	  else echo "mutant $$m caught"; fi; \
	done

typecheck:
	$(SWIFTC) -typecheck -Xfrontend -warn-long-expression-type-checking=5 \
	  $(SIM) Tests/Harness.swift Tests/main.swift

clean:
	rm -rf $(BUILD)

.PHONY: test report mutants typecheck clean
