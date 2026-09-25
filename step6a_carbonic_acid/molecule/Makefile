# Step 6a, up close: one H₂CO₃ molecule splitting into H⁺ + HCO₃⁻
#
#   make run    ray-trace the molecule and save renders/molecule.gif
#   make test   build and run the tests
#
# If swiftc says the SDK doesn't match the compiler, point it at one that does:
#   make test SDK=/Library/Developer/CommandLineTools/SDKs/MacOSX26.sdk

# Command-line tools must link CoreGraphics for MTLCreateSystemDefaultDevice()
# to find the GPU (per Apple's docs). CoreText draws the labels.
SWIFTC := swiftc -O -framework CoreGraphics -framework CoreText $(if $(SDK),-sdk $(SDK))
BUILD := .build
SOURCES := Sources/Molecule.swift Sources/Render.swift Sources/Output.swift

$(BUILD)/molecule: $(SOURCES) Sources/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(filter %.swift,$^) -o $@

$(BUILD)/tests: $(SOURCES) Tests/Harness.swift Tests/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(filter %.swift,$^) -o $@

run: $(BUILD)/molecule
	@$<

test: $(BUILD)/tests
	@$<

clean:
	rm -rf $(BUILD) renders

.PHONY: run test clean
