# Step 8: the DNA double helix (PDB 1BNA), turning once as a looping GIF
#
#   make run        render renders/dna.gif
#   make test       build and run the Swift tests
#   make structure  rebuild Resources/dna.json from Resources/1BNA.pdb (needs RDKit)
#
# If swiftc says the SDK doesn't match the compiler, point it at one that does:
#   make test SDK=/Library/Developer/CommandLineTools/SDKs/MacOSX26.sdk

# Command-line tools must link CoreGraphics for MTLCreateSystemDefaultDevice()
# to find the GPU (per Apple's docs). CoreText draws the labels.
SWIFTC := swiftc -O -framework CoreGraphics -framework CoreText $(if $(SDK),-sdk $(SDK))
BUILD := .build
SOURCES := Sources/Scene.swift Sources/Render.swift Sources/Overlay.swift Sources/GIF.swift

$(BUILD)/dna: $(SOURCES) Sources/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(filter %.swift,$^) -o $@

$(BUILD)/tests: $(SOURCES) Tests/Harness.swift Tests/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(filter %.swift,$^) -o $@

run: $(BUILD)/dna
	@$<

test: $(BUILD)/tests
	@$<

structure:
	cd Tools && python3 build_dna.py

clean:
	rm -rf $(BUILD) renders



.PHONY: run test structure clean
