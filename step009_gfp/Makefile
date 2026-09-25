# Step 9: green fluorescent protein (PDB 1EMA), space-filling with ambient
# occlusion, turning once with a cutaway to the chromophore
#
#   make run        render renders/gfp.gif
#   make test       build and run the Swift tests
#   make structure  rebuild Resources/gfp.json from Resources/1EMA.pdb
#
# If swiftc says the SDK doesn't match the compiler, point it at one that does:
#   make test SDK=/Library/Developer/CommandLineTools/SDKs/MacOSX26.sdk

# Command-line tools must link CoreGraphics for MTLCreateSystemDefaultDevice()
# to find the GPU (per Apple's docs). CoreText draws the labels.
SWIFTC := swiftc -O -framework CoreGraphics -framework CoreText $(if $(SDK),-sdk $(SDK))
BUILD := .build
SOURCES := Sources/Protein.swift Sources/Render.swift Sources/Overlay.swift Sources/GIF.swift

$(BUILD)/gfp: $(SOURCES) Sources/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(filter %.swift,$^) -o $@

$(BUILD)/tests: $(SOURCES) Tests/Harness.swift Tests/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(filter %.swift,$^) -o $@

run: $(BUILD)/gfp
	@$<

test: $(BUILD)/tests
	@$<

structure:
	cd Tools && python3 build_gfp.py

clean:
	rm -rf $(BUILD) renders



.PHONY: run test structure clean
