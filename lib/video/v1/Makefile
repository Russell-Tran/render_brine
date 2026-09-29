# lib/video/v1: an H.264 MP4 writer (AVAssetWriter, hardware encoder) for
# rendered RGBA8 frames, and the readers that test it. Version-pinned: a step
# uses it by listing $(VIDEO) in its own Makefile's sources, e.g.
#
#   VIDEO := ../lib/video/v1/Sources/MP4Writer.swift
#   SWIFTC := swiftc -O ... -framework AVFoundation -framework CoreMedia \
#             -framework CoreVideo -framework VideoToolbox -framework Accelerate
#
# (add ../lib/video/v1/Sources/MP4Inspect.swift to its tests to read the film back).
#
#   make test       write a 1.5 s 1080p30 clip, read it back, check it
#   make mutants    break the writer four ways; the tests must catch each
#   make typecheck  prove no expression is slow to type-check
#   make p1         prediction P1: re-encode a showcase GIF's frames as MP4, compare
#                   (GIF=../../../showcase/ant_trail_5.gif to pick another)
#
# If swiftc says the SDK doesn't match the compiler, point it at one that does:
#   make test SDK=$(xcrun --show-sdk-path)
# To try another compiler: make test SWIFT=/path/to/swiftc SDK=/path/to/MacOSX.sdk

SWIFT ?= swiftc
FRAMEWORKS := -framework AVFoundation -framework CoreMedia -framework CoreVideo \
              -framework VideoToolbox -framework Accelerate -framework CoreGraphics -framework ImageIO
SWIFTC := $(SWIFT) -O $(FRAMEWORKS) $(if $(SDK),-sdk $(SDK))
BUILD := .build
SOURCES := Sources/MP4Writer.swift Sources/MP4Inspect.swift
GIF ?= ../../../showcase/ant_trail_wood.gif

$(BUILD)/tests: $(SOURCES) Tests/Harness.swift Tests/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(SOURCES) Tests/Harness.swift Tests/main.swift -o $@

$(BUILD)/gif_vs_mp4: $(SOURCES) Tools/gif_vs_mp4/main.swift Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(SOURCES) Tools/gif_vs_mp4/main.swift -o $@

test: $(BUILD)/tests
	@$<

# Each mutant must make the suite FAIL. A mutant that passes is a test missing.
mutants: $(BUILD)/tests
	@for m in wrongFrameCount wrongFps swapChannels softwareAllowed; do \
	  if VIDEO_MUTANT=$$m $< > /dev/null 2>&1; then echo "mutant $$m ESCAPED"; exit 1; \
	  else echo "mutant $$m caught"; fi; \
	done

typecheck:
	$(SWIFTC) -typecheck -Xfrontend -warn-long-expression-type-checking=5 \
	  $(SOURCES) Tests/Harness.swift Tests/main.swift
	$(SWIFTC) -typecheck -Xfrontend -warn-long-expression-type-checking=5 \
	  $(SOURCES) Tools/gif_vs_mp4/main.swift

p1: $(BUILD)/gif_vs_mp4
	@$< $(GIF)

clean:
	rm -rf $(BUILD)

.PHONY: test mutants typecheck p1 clean
