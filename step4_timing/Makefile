# Step 4: Time it on both chips
#
#   make run    time the step 3 water render at three sizes
#   make test   build and run the tests
#
# If swiftc says the SDK doesn't match the compiler, point it at one that does:
#   make test SDK=/Library/Developer/CommandLineTools/SDKs/MacOSX26.sdk

# Command-line tools must link CoreGraphics for MTLCreateSystemDefaultDevice()
# to find the GPU (per Apple's docs).
SWIFTC := swiftc -O -framework CoreGraphics $(if $(SDK),-sdk $(SDK))
BUILD := .build

# The renderer being timed comes straight from step 3, so both steps always
# use the same kernel.
WATER := ../step3_water/Sources/Water.swift

$(BUILD)/timing: $(WATER) Sources/Benchmark.swift Sources/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(filter %.swift,$^) -o $@

$(BUILD)/tests: $(WATER) Sources/Benchmark.swift Tests/Harness.swift Tests/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(filter %.swift,$^) -o $@

run: $(BUILD)/timing
	@$<

test: $(BUILD)/tests
	@$<

clean:
	rm -rf $(BUILD)

.PHONY: run test clean
