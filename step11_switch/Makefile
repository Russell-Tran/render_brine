# Step 11: AraC's light switch — the protein that ties pGLO's DNA in a loop to
# keep GFP off, and unties it when arabinose arrives
#
#   make run        render renders/switch.gif
#   make test       build and run the Swift tests
#   make structure  rebuild Resources/switch.json from the PDB files and step 8a's pGLO
#
# If swiftc says the SDK doesn't match the compiler, point it at one that does:
#   make test SDK=/Library/Developer/CommandLineTools/SDKs/MacOSX26.sdk

# Command-line tools must link CoreGraphics for MTLCreateSystemDefaultDevice()
# to find the GPU (per Apple's docs). CoreText draws the labels.
SWIFTC := swiftc -O -framework CoreGraphics -framework CoreText $(if $(SDK),-sdk $(SDK))
BUILD := .build
SOURCES := Sources/Switch.swift Sources/Render.swift Sources/Overlay.swift Sources/GIF.swift

$(BUILD)/switch: $(SOURCES) Sources/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(filter %.swift,$^) -o $@

$(BUILD)/tests: $(SOURCES) Tests/Harness.swift Tests/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(filter %.swift,$^) -o $@

run: $(BUILD)/switch
	@$<

test: $(BUILD)/tests
	@$<

structure:
	cd Tools && python3 build_switch.py

clean:
	rm -rf $(BUILD) renders



.PHONY: run test structure clean
