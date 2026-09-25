# Step 16: inside a moving schoolbus — a camera bolted to the vehicle, and the
# world moving past the windows
#
#   make run        render renders/schoolbus.gif
#   make control    render renders/orbit_control.gif, the moving-camera control
#   make stills     write the still frames for the results page
#   make bench      measure what the aperture mask buys, against the control
#   make test       build and run the Swift tests
#
# If swiftc says the SDK doesn't match the compiler, point it at one that does:
#   make test SDK=/Library/Developer/CommandLineTools/SDKs/MacOSX26.sdk
# On the M3 Max laptop, `SDK=$(xcrun --show-sdk-path)` is the right one.

# Command-line tools must link CoreGraphics for MTLCreateSystemDefaultDevice()
# to find the GPU (per Apple's docs). CoreText draws the labels.
SWIFTC := swiftc -O -framework CoreGraphics -framework CoreText $(if $(SDK),-sdk $(SDK))
BUILD := .build
SOURCES := Sources/Bus.swift Sources/Render.swift Sources/Overlay.swift Sources/GIF.swift

$(BUILD)/schoolbus: $(SOURCES) Sources/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(filter %.swift,$^) -o $@

$(BUILD)/tests: $(SOURCES) Tests/Harness.swift Tests/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(filter %.swift,$^) -o $@

run: $(BUILD)/schoolbus
	@$<

control: $(BUILD)/schoolbus
	@$< control

stills: $(BUILD)/schoolbus
	@$< stills

bench: $(BUILD)/schoolbus
	@$< bench

test: $(BUILD)/tests
	@$<

clean:
	rm -rf $(BUILD) renders

.PHONY: run control stills bench test clean
