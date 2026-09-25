# Step 8a: the pGLO plasmid — the whole 5,371 bp circle, then a dive into GFP
#
#   make run        render renders/plasmid_ring.gif and renders/plasmid_dive.gif
#   make test       build and run the Swift tests
#   make structure  rebuild Resources/pglo.json from the published pGLO sequence
#
# If swiftc says the SDK doesn't match the compiler, point it at one that does:
#   make test SDK=/Library/Developer/CommandLineTools/SDKs/MacOSX26.sdk

# Command-line tools must link CoreGraphics for MTLCreateSystemDefaultDevice()
# to find the GPU (per Apple's docs). CoreText draws the labels.
SWIFTC := swiftc -O -framework CoreGraphics -framework CoreText $(if $(SDK),-sdk $(SDK))
BUILD := .build
SOURCES := Sources/Plasmid.swift Sources/Scene.swift Sources/Render.swift Sources/Overlay.swift Sources/GIF.swift

$(BUILD)/plasmid: $(SOURCES) Sources/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(filter %.swift,$^) -o $@

$(BUILD)/tests: $(SOURCES) Tests/Harness.swift Tests/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(filter %.swift,$^) -o $@

run: $(BUILD)/plasmid
	@$<

test: $(BUILD)/tests
	@$<

structure:
	cd Tools && python3 build_plasmid.py

clean:
	rm -rf $(BUILD) renders



.PHONY: run test structure clean
