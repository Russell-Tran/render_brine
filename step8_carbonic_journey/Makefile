# Step 8: the carbonic acid journey, CO₂ + H₂O → H₂CO₃ → HCO₃⁻ + H₃O⁺, looping
#
#   make run    ray-trace the loop and save renders/journey.gif (make still T=5 for one frame)
#   make test   build and run the tests
#
# If swiftc says the SDK doesn't match the compiler, point it at one that does:
#   make test SDK=/Library/Developer/CommandLineTools/SDKs/MacOSX26.sdk

# Command-line tools must link CoreGraphics for MTLCreateSystemDefaultDevice()
# to find the GPU (per Apple's docs). CoreText draws the labels.
SWIFTC := swiftc -O -framework CoreGraphics -framework CoreText $(if $(SDK),-sdk $(SDK))
BUILD := .build
SOURCES := Sources/Journey.swift Sources/Render.swift Sources/Output.swift

$(BUILD)/journey: $(SOURCES) Sources/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(filter %.swift,$^) -o $@

$(BUILD)/tests: $(SOURCES) Tests/Harness.swift Tests/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(filter %.swift,$^) -o $@

run: $(BUILD)/journey
	@$<

test: $(BUILD)/tests
	@$<

still: $(BUILD)/journey
	@$< --still $(T)

clean:
	rm -rf $(BUILD) renders

.PHONY: run test still clean
