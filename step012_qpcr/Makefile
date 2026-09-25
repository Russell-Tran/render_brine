# Step 12: Förster transfer in a real-time PCR probe pair, with brightness
# computed from the geometry on every frame
#
#   make run        render renders/fret.gif
#   make test       build and run the Swift tests
#   make structure  rebuild Resources/scene.json (needs step008a_plasmid)
#
# If swiftc says the SDK doesn't match the compiler, point it at one that does:
#   make test SDK=/Library/Developer/CommandLineTools/SDKs/MacOSX26.sdk

# Command-line tools must link CoreGraphics for MTLCreateSystemDefaultDevice()
# to find the GPU (per Apple's docs). CoreText draws the labels.
SWIFTC := swiftc -O -framework CoreGraphics -framework CoreText $(if $(SDK),-sdk $(SDK))
BUILD := .build
SOURCES := Sources/Scene.swift Sources/Fret.swift Sources/Spectrum.swift \
           Sources/Render.swift Sources/Overlay.swift Sources/GIF.swift

$(BUILD)/qpcr: $(SOURCES) Sources/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(filter %.swift,$^) -o $@

$(BUILD)/tests: $(SOURCES) Tests/Harness.swift Tests/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(filter %.swift,$^) -o $@

run: $(BUILD)/qpcr
	@$<

test: $(BUILD)/tests
	@$<

structure:
	cd Tools && python3 build_scene.py

clean:
	rm -rf $(BUILD) renders

.PHONY: run test structure clean
