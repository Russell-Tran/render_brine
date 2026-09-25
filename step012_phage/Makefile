# Step 12: a T4 phage baseplate firing into E. coli
#
#   make run        render renders/firing.gif
#   make stills     write the still frames for the results page
#   make bench      time brute force against the uniform grid
#   make test       build and run the Swift tests
#   make structure  rebuild Resources/scene.json from the PDB entries
#
# If swiftc says the SDK doesn't match the compiler, point it at one that does:
#   make test SDK=/Library/Developer/CommandLineTools/SDKs/MacOSX26.sdk
# On the M3 Max laptop, `SDK=$(xcrun --show-sdk-path)` is the right one.

SWIFTC := swiftc -O -framework CoreGraphics -framework CoreText $(if $(SDK),-sdk $(SDK))
BUILD := .build
SOURCES := Sources/Phage.swift Sources/Render.swift Sources/Overlay.swift Sources/GIF.swift

$(BUILD)/phage: $(SOURCES) Sources/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(filter %.swift,$^) -o $@

$(BUILD)/tests: $(SOURCES) Tests/Harness.swift Tests/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(filter %.swift,$^) -o $@

run: $(BUILD)/phage
	@$<

stills: $(BUILD)/phage
	@$< stills

bench: $(BUILD)/phage
	@$< bench

test: $(BUILD)/tests
	@$<

structure:
	cd Tools && python3 build_phage.py

clean:
	rm -rf $(BUILD) renders

.PHONY: run stills bench test structure clean
