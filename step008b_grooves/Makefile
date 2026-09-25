# Step 8b: DNA space-filling, so the major and minor grooves read as the real
# channels they are — 40 base pairs turning, plus a side-by-side still that
# compares the two representations on the same twelve crystal base pairs.
#
#   make run        render renders/grooves.gif and the comparison stills
#   make test       build and run the Swift tests
#   make structure  rebuild Resources/grooves.json from steps 8 and 8a
#
# If swiftc says the SDK doesn't match the compiler, point it at one that does:
#   make test SDK=/Library/Developer/CommandLineTools/SDKs/MacOSX26.sdk

# Command-line tools must link CoreGraphics for MTLCreateSystemDefaultDevice()
# to find the GPU (per Apple's docs). CoreText draws the labels.
SWIFTC := swiftc -O -framework CoreGraphics -framework CoreText $(if $(SDK),-sdk $(SDK))
BUILD := .build
SOURCES := Sources/Grooves.swift Sources/Render.swift Sources/Overlay.swift Sources/GIF.swift

$(BUILD)/grooves: $(SOURCES) Sources/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(filter %.swift,$^) -o $@

$(BUILD)/tests: $(SOURCES) Tests/Harness.swift Tests/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(filter %.swift,$^) -o $@

run: $(BUILD)/grooves
	@$<

test: $(BUILD)/tests
	@$<

structure:
	cd Tools && python3 build_grooves.py

clean:
	rm -rf $(BUILD) renders

.PHONY: run test structure clean
