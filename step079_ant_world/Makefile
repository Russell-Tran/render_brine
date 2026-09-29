# Step 79, "A minute of ant world".
#   PART 3: the simulation (prediction P2). Seeded Lasius niger workers, a nest,
#   a sugar pile, and a pheromone field on the GPU (deposit, diffuse, evaporate
#   every tick).
#   PART 4: the renderer (prediction P3), drawing with the SHARED ant
#   (../lib/ant/v1).
#   PART 5: the film (prediction P5): 60 s, 1920×1080, 30 fps, resumable, the
#   MP4 written with the shared writer (../lib/video/v1) → showcase/ant_world.mp4.
#
# The simulation is chaotic: on another machine seed 79 is a different world.
# So it is run ONCE (make record, on the laptop) into records/world79.rec,
# whose SHA-256 is in records/MANIFEST, and the renderer reads only that file.
# The renderer is built WITHOUT the simulation's sources (RUN below), and
# `make isolation` proves its binary has none of it.
#
#   make test       the tests (a few minutes; needs the record)
#   make film       render the film's frames into renders/frames (about an
#                   hour); stop it any time, run it again to carry on
#   make encode     the MP4 from the frames → showcase/ant_world.mp4, checked
#   make gif        the GIF excerpt → showcase/ant_world.gif
#   make refcheck   draw the reference frames again and compare with records/refs
#                   (for another machine: how close is its GPU's film to ours?)
#   make record     run the simulation once and save the film's record
#   make scale      the P3 numbers: one 1080p frame at 1, 5, 10, 20, 40 ants
#   make report     the P2 numbers: with and without pheromone, and formation
#                   times over eight seeds (a few minutes)
#   make mutants    break things on purpose; the tests must catch each
#   make typecheck  prove no expression is slow to type-check
#   make swift6     build and run the tests with Xcode's newer Swift as well
#
# If swiftc says the SDK doesn't match the compiler, point it at one that does:
#   make test SDK=$(xcrun --show-sdk-path)

FRAMEWORKS := -framework Metal -framework CoreGraphics -framework ImageIO -framework CoreText
VIDEO_FRAMEWORKS := -framework AVFoundation -framework CoreMedia -framework CoreVideo -framework VideoToolbox \
                    -framework Accelerate -framework CoreGraphics -framework ImageIO
SWIFT ?= swiftc
SWIFTC := $(SWIFT) -O $(FRAMEWORKS) $(if $(SDK),-sdk $(SDK))
SWIFTC_VIDEO := $(SWIFT) -O $(VIDEO_FRAMEWORKS) $(if $(SDK),-sdk $(SDK))
BUILD := .build

# The shared modules, version-pinned (../lib/README).
ANT := ../lib/ant/v1/Sources
ANT_SRC := $(ANT)/AntV1Anatomy.swift $(ANT)/AntV1Pose.swift $(ANT)/AntV1Metal.swift
VIDEO := ../lib/video/v1/Sources
VIDEO_SRC := $(VIDEO)/MP4Writer.swift $(VIDEO)/MP4Inspect.swift

# What the record holds (types and file format; no simulation in them).
DATA := Sources/SimConstants.swift Sources/SimClock.swift Sources/SimRandom.swift Sources/SimTypes.swift \
        Sources/SimFilmRecord.swift Sources/FilmRecord.swift
# The simulation itself: never in the renderer.
RUN := Sources/SimField.swift Sources/SimWorld.swift Sources/SimRecord.swift Sources/SimMetrics.swift
# The film's drawing.
WORLD := Sources/WorldSugar.swift Sources/WorldScene.swift Sources/WorldRender.swift \
         Sources/Film.swift Sources/FilmTools.swift Sources/GIF.swift

# lib/video/v1 (AVFoundation) goes only into the small MP4 tool: once any
# file in a module imports AVFoundation, its operators slow the type-checking
# of every == in the module (the mini measured 5 ms a line).
RENDER_SRC := $(DATA) $(WORLD) $(ANT_SRC) Sources/main.swift
RECORD_SRC := $(DATA) $(RUN) $(WORLD) $(ANT_SRC) Tools/record/main.swift
MP4_SRC := $(VIDEO_SRC) Tools/mp4/main.swift
TEST_SRC := $(DATA) $(RUN) $(WORLD) $(ANT_SRC) Tests/Harness.swift Tests/RenderTests.swift Tests/FilmTests.swift Tests/main.swift

$(BUILD)/tests: $(TEST_SRC) Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(TEST_SRC) -o $@

$(BUILD)/render: $(RENDER_SRC) Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(RENDER_SRC) -o $@

$(BUILD)/record: $(RECORD_SRC) Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(RECORD_SRC) -o $@

$(BUILD)/mp4: $(MP4_SRC) Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC_VIDEO) $(MP4_SRC) -o $@

# The suite, the isolation check, and the MP4 checks: a synthetic film of the
# plan's frame count through the same writer, and the finished film if it is
# there. SIM_MUTANT passes through to all of them.
test: $(BUILD)/tests $(BUILD)/mp4 isolation
	@$(BUILD)/tests
	@n=$$($(BUILD)/render count); $(BUILD)/mp4 synth $$n
	@n=$$($(BUILD)/render count); if [ -f ../showcase/ant_world.mp4 ]; then $(BUILD)/mp4 check ../showcase/ant_world.mp4 $$n 20; \
	  else echo "(no showcase/ant_world.mp4 yet)"; fi

record: $(BUILD)/record
	@$<

# THE FILM. Stop it whenever; `make film` again carries on where it stopped.
film: $(BUILD)/render
	@$< film

encode: $(BUILD)/render $(BUILD)/mp4
	@n=$$($(BUILD)/render count); $(BUILD)/mp4 encode renders/frames $$n renders/ant_world.mp4 && \
	  cp renders/ant_world.mp4 ../showcase/ant_world.mp4 && $(BUILD)/mp4 check ../showcase/ant_world.mp4 $$n 20

# The GIF excerpt: the trail forming (film 34–50 s, the time-lapse just before
# real time), every 2nd frame at 7 cs (about film speed), 1280 wide.
gif: $(BUILD)/render
	@$< gif 1020 1500 2 1280 7

# Frames drawn here, kept in records/refs; another machine draws them again.
refcheck: $(BUILD)/render
	@$< refcheck

scale: $(BUILD)/render
	@$< scale

report: $(BUILD)/tests
	@$< report

# The renderer's binary must contain nothing of the simulation: no simRun,
# simFilmRun, SimWorld, SimAnt, SimField, SimFieldPlayer, simSugarPile (Swift
# mangles a name with its length first, e.g. 8SimWorldC for the class). The
# tests binary, which has them all, shows the pattern finds them.
SIM_SYMBOLS := '6simRun|10simFilmRun|8SimWorldC|6SimAntV|8SimFieldC|14SimFieldPlayerC|12simSugarPile'
isolation: $(BUILD)/render
	@if nm $(BUILD)/render | grep -E $(SIM_SYMBOLS) > /dev/null; then \
	  echo "isolation FAILED: the renderer links the simulation"; exit 1; \
	else echo "isolation ok: the renderer has no simulation in it"; fi
	@if [ -f $(BUILD)/tests ] && ! nm $(BUILD)/tests | grep -E $(SIM_SYMBOLS) > /dev/null; then \
	  echo "isolation check is blind: it finds nothing in the tests either"; exit 1; fi

# Each mutant must make the suite FAIL. A mutant that passes is a test missing.
mutants: $(BUILD)/tests $(BUILD)/mp4 $(BUILD)/render
	@for m in noPheromone rewind tightTurn shrunkBounds sliding_feet wrongFrameCount; do \
	  if SIM_MUTANT=$$m $(MAKE) -s test > /dev/null 2>&1; then echo "mutant $$m ESCAPED"; exit 1; \
	  else echo "mutant $$m caught"; fi; \
	done

typecheck:
	$(SWIFTC) -typecheck -Xfrontend -warn-long-expression-type-checking=5 $(TEST_SRC)
	$(SWIFTC) -typecheck -Xfrontend -warn-long-expression-type-checking=5 $(RENDER_SRC)
	$(SWIFTC) -typecheck -Xfrontend -warn-long-expression-type-checking=5 $(RECORD_SRC)
	$(SWIFTC_VIDEO) -typecheck -Xfrontend -warn-long-expression-type-checking=5 $(MP4_SRC)

XCODE := /Applications/Xcode.app/Contents/Developer
SWIFT6 := $(XCODE)/Toolchains/XcodeDefault.xctoolchain/usr/bin/swiftc
SDK6 := $(XCODE)/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk
SWIFTC6 := $(SWIFT6) -O $(FRAMEWORKS) -sdk $(SDK6)
swift6:
	@mkdir -p $(BUILD)
	$(SWIFTC6) -Xfrontend -warn-long-expression-type-checking=5 $(RENDER_SRC) -o $(BUILD)/render6
	$(SWIFTC6) -Xfrontend -warn-long-expression-type-checking=5 $(RECORD_SRC) -o $(BUILD)/record6
	$(SWIFT6) -O $(VIDEO_FRAMEWORKS) -sdk $(SDK6) -Xfrontend -warn-long-expression-type-checking=5 $(MP4_SRC) -o $(BUILD)/mp4_6
	$(SWIFTC6) -Xfrontend -warn-long-expression-type-checking=5 $(TEST_SRC) -o $(BUILD)/tests6
	@$(BUILD)/tests6
	@$(BUILD)/mp4_6 synth 1800

clean:
	rm -rf $(BUILD)

.PHONY: test record film encode gif refcheck scale report isolation mutants typecheck swift6 clean
