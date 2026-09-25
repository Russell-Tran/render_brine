# Step 10: bacterial transformation — how a plasmid gets into E. coli
#
#   make run        render renders/approach.gif
#   make bench      time brute force against the uniform grid
#   make test       build and run the Swift tests
#   make structure  rebuild Resources/scene.json (needs Python with numpy)
#
# If swiftc says the SDK doesn't match the compiler, point it at one that does:
#   make test SDK=/Library/Developer/CommandLineTools/SDKs/MacOSX26.sdk
# On the M3 Max laptop, `SDK=$(xcrun --show-sdk-path)` is the right one.

SWIFTC := swiftc -O -framework CoreGraphics -framework CoreText $(if $(SDK),-sdk $(SDK))
BUILD := .build
SOURCES := Sources/Scene.swift Sources/Routes.swift Sources/Render.swift Sources/Overlay.swift Sources/GIF.swift

$(BUILD)/transformation: $(SOURCES) Sources/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(filter %.swift,$^) -o $@

$(BUILD)/tests: $(SOURCES) Tests/Harness.swift Tests/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(filter %.swift,$^) -o $@

run: $(BUILD)/transformation
	@$<

bench: $(BUILD)/transformation
	@$< bench

test: $(BUILD)/tests
	@$<

structure:
	cd Tools && python3 build_scene.py

clean:
	rm -rf $(BUILD) renders

.PHONY: run bench test structure clean
