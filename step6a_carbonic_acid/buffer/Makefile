# Step 6a: The carbonic acid–bicarbonate buffer
#
#   make run    simulate the buffer, render renders/buffer.gif and renders/ph.csv
#   make test   build and run the tests
#
# If swiftc says the SDK doesn't match the compiler, point it at one that does:
#   make test SDK=/Library/Developer/CommandLineTools/SDKs/MacOSX26.sdk

# Command-line tools must link CoreGraphics for MTLCreateSystemDefaultDevice()
# to find the GPU (per Apple's docs). CoreText draws the labels.
SWIFTC := swiftc -O -framework CoreGraphics -framework CoreText $(if $(SDK),-sdk $(SDK))
BUILD := .build
SOURCES := Sources/Chemistry.swift Sources/Particles.swift Sources/Frames.swift

$(BUILD)/buffer: $(SOURCES) Sources/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(filter %.swift,$^) -o $@

$(BUILD)/tests: $(SOURCES) Tests/Harness.swift Tests/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(filter %.swift,$^) -o $@

run: $(BUILD)/buffer
	@$<

test: $(BUILD)/tests
	@$<

clean:
	rm -rf $(BUILD) renders

.PHONY: run test clean
