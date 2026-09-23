# Step 1: Hello, GPU
#
#   make run    build and print this Mac's GPU report
#   make test   build and run the tests
#
# If swiftc says the SDK doesn't match the compiler, point it at one that does:
#   make test SDK=/Library/Developer/CommandLineTools/SDKs/MacOSX26.sdk

SWIFTC := swiftc $(if $(SDK),-sdk $(SDK))
BUILD := .build

$(BUILD)/hello_gpu: Sources/GPUReport.swift Sources/main.swift
	@mkdir -p $(BUILD)
	$(SWIFTC) $^ -o $@

$(BUILD)/tests: Sources/GPUReport.swift Tests/Harness.swift Tests/main.swift
	@mkdir -p $(BUILD)
	$(SWIFTC) $^ -o $@

run: $(BUILD)/hello_gpu
	@$<

test: $(BUILD)/tests
	@$<

clean:
	rm -rf $(BUILD)

.PHONY: run test clean
