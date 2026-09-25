# Step 9a: GFP beside mCherry — the extra conjugation that turns green light
# red, with both proteins' colours computed from the light they emit
#
#   make run        render renders/color.gif
#   make test       build and run the Swift tests
#   make structure  rebuild Resources/pair.json from the two PDB files
#
# If swiftc says the SDK doesn't match the compiler, point it at one that does:
#   make test SDK=/Library/Developer/CommandLineTools/SDKs/MacOSX26.sdk

# Command-line tools must link CoreGraphics for MTLCreateSystemDefaultDevice()
# to find the GPU (per Apple's docs). CoreText draws the labels.
SWIFTC := swiftc -O -framework CoreGraphics -framework CoreText $(if $(SDK),-sdk $(SDK))
BUILD := .build
SOURCES := Sources/Spectrum.swift Sources/Pair.swift Sources/Render.swift Sources/Overlay.swift Sources/GIF.swift

$(BUILD)/color: $(SOURCES) Sources/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(filter %.swift,$^) -o $@

$(BUILD)/tests: $(SOURCES) Tests/Harness.swift Tests/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(filter %.swift,$^) -o $@

run: $(BUILD)/color
	@$<

test: $(BUILD)/tests
	@$<

structure:
	cd Tools && python3 build_pair.py

clean:
	rm -rf $(BUILD) renders

.PHONY: run test structure clean
