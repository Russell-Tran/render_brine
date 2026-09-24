# Step 5: The sea mirrors the sky
#
#   make run    render renders/reflections.png, then time 1, 4 and 16 samples per pixel
#   make test   build and run the tests
#
# If swiftc says the SDK doesn't match the compiler, point it at one that does:
#   make test SDK=/Library/Developer/CommandLineTools/SDKs/MacOSX26.sdk

# Command-line tools must link CoreGraphics for MTLCreateSystemDefaultDevice()
# to find the GPU (per Apple's docs). It also writes the PNGs.
SWIFTC := swiftc -O -framework CoreGraphics $(if $(SDK),-sdk $(SDK))
BUILD := .build

# Waves, camera, colors and PNG saving come from step 3.
WATER := ../step3_water/Sources/Water.swift

$(BUILD)/reflections: $(WATER) Sources/Reflections.swift Sources/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(filter %.swift,$^) -o $@

$(BUILD)/tests: $(WATER) Sources/Reflections.swift Tests/Harness.swift Tests/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(filter %.swift,$^) -o $@

run: $(BUILD)/reflections
	@$<

test: $(BUILD)/tests
	@$<

clean:
	rm -rf $(BUILD) renders

.PHONY: run test clean
