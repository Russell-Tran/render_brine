# Step 9: Glycolysis — spend two ATP, earn four
#
#   make run        render renders/spend.gif and renders/payoff.gif
#   make test       build and run the Swift tests
#   make timeline   rebuild Resources/timeline.json (needs Python RDKit: pip3 install --user rdkit)
#   make check      check every reaction balances (needs Python RDKit)
#
# If swiftc says the SDK doesn't match the compiler, point it at one that does:
#   make test SDK=/Library/Developer/CommandLineTools/SDKs/MacOSX26.sdk

# Command-line tools must link CoreGraphics for MTLCreateSystemDefaultDevice()
# to find the GPU (per Apple's docs). CoreText draws the labels.
SWIFTC := swiftc -O -framework CoreGraphics -framework CoreText $(if $(SDK),-sdk $(SDK))
BUILD := .build
SOURCES := Sources/Movie.swift Sources/Render.swift Sources/Overlay.swift

$(BUILD)/glycolysis: $(SOURCES) Sources/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(filter %.swift,$^) -o $@

$(BUILD)/tests: $(SOURCES) Tests/Harness.swift Tests/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(filter %.swift,$^) -o $@

run: $(BUILD)/glycolysis
	@$<

test: $(BUILD)/tests
	@$<

timeline:
	cd Tools && python3 build_timeline.py

check:
	cd Tools && python3 check_balances.py

clean:
	rm -rf $(BUILD) renders

.PHONY: run test timeline check clean
