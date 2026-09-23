# Step 2: Paint one image on the GPU
#
#   make run    paint a 1920 × 1080 gradient and save renders/gradient.png
#   make test   build and run the tests
#
# If swiftc says the SDK doesn't match the compiler, point it at one that does:
#   make test SDK=/Library/Developer/CommandLineTools/SDKs/MacOSX26.sdk

# Command-line tools must link CoreGraphics for MTLCreateSystemDefaultDevice()
# to find the GPU (per Apple's docs). This step also uses it to write PNGs.
SWIFTC := swiftc -framework CoreGraphics $(if $(SDK),-sdk $(SDK))
BUILD := .build

$(BUILD)/paint_gpu: Sources/Paint.swift Sources/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(filter %.swift,$^) -o $@

$(BUILD)/tests: Sources/Paint.swift Tests/Harness.swift Tests/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(filter %.swift,$^) -o $@

run: $(BUILD)/paint_gpu
	@$<

test: $(BUILD)/tests
	@$<

clean:
	rm -rf $(BUILD) renders

.PHONY: run test clean
