# Step 1: Hello, GPU
#
#   make run    build and print this Mac's GPU report
#   make test   build and run the tests
#
# If swiftc says the SDK doesn't match the compiler, point it at one that does:
#   make test SDK=/Library/Developer/CommandLineTools/SDKs/MacOSX26.sdk

# Command-line tools must link CoreGraphics for MTLCreateSystemDefaultDevice()
# to find the GPU (per Apple's docs), even though nothing calls it directly.
SWIFTC := swiftc -framework CoreGraphics $(if $(SDK),-sdk $(SDK))
BUILD := .build

$(BUILD)/hello_gpu: Sources/GPUReport.swift Sources/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(filter %.swift,$^) -o $@

$(BUILD)/tests: Sources/GPUReport.swift Tests/Harness.swift Tests/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(filter %.swift,$^) -o $@

run: $(BUILD)/hello_gpu
	@$<

test: $(BUILD)/tests
	@$<

clean:
	rm -rf $(BUILD)

.PHONY: run test clean
