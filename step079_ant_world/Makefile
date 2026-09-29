# Step 79, "A minute of ant world".
#   PART 3: the simulation (prediction P2). Seeded Lasius niger workers, a nest,
#   a sugar pile, and a pheromone field on the GPU (deposit, diffuse, evaporate
#   every tick).
#   PART 4: the renderer (prediction P3), drawing with the SHARED ant
#   (../lib/ant/v1) and linking the shared MP4 writer (../lib/video/v1).
#
# The simulation is chaotic: on another machine seed 79 is a different world.
# So it is run ONCE (make record, on the laptop) into records/world79.rec,
# whose SHA-256 is in records/MANIFEST, and the renderer reads only that file.
# The renderer is built WITHOUT the simulation's sources (RUN below), and
# `make isolation` proves its binary has none of it.
#
#   make test       build and run the tests (a few minutes; needs the record)
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

FRAMEWORKS := -framework Metal -framework CoreGraphics -framework ImageIO -framework AVFoundation \
              -framework CoreMedia -framework CoreVideo -framework VideoToolbox -framework Accelerate
SWIFT ?= swiftc
SWIFTC := $(SWIFT) -O $(FRAMEWORKS) $(if $(SDK),-sdk $(SDK))
BUILD := .build

# The shared modules, version-pinned (../lib/README).
ANT := ../lib/ant/v1/Sources
ANT_SRC := $(ANT)/AntV1Anatomy.swift $(ANT)/AntV1Pose.swift $(ANT)/AntV1Metal.swift
VIDEO := ../lib/video/v1/Sources
VIDEO_SRC := $(VIDEO)/MP4Writer.swift
VIDEO_INSPECT := $(VIDEO)/MP4Inspect.swift

# What the record holds (types and file format; no simulation in them).
DATA := Sources/SimConstants.swift Sources/SimClock.swift Sources/SimRandom.swift Sources/SimTypes.swift \
        Sources/SimFilmRecord.swift Sources/FilmRecord.swift
# The simulation itself: never in the renderer.
RUN := Sources/SimField.swift Sources/SimWorld.swift Sources/SimRecord.swift Sources/SimMetrics.swift
# The film's drawing.
WORLD := Sources/WorldSugar.swift Sources/WorldScene.swift Sources/WorldRender.swift

RENDER_SRC := $(DATA) $(WORLD) $(ANT_SRC) $(VIDEO_SRC) Sources/main.swift
RECORD_SRC := $(DATA) $(RUN) $(WORLD) $(ANT_SRC) Tools/record/main.swift
TEST_SRC := $(DATA) $(RUN) $(WORLD) $(ANT_SRC) $(VIDEO_SRC) $(VIDEO_INSPECT) \
            Tests/Harness.swift Tests/Render.swift Tests/main.swift

$(BUILD)/tests: $(TEST_SRC) Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(TEST_SRC) -o $@

$(BUILD)/render: $(RENDER_SRC) Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(RENDER_SRC) -o $@

$(BUILD)/record: $(RECORD_SRC) Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(RECORD_SRC) -o $@

test: $(BUILD)/tests isolation
	@$(BUILD)/tests

record: $(BUILD)/record
	@$<

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
mutants: $(BUILD)/tests
	@for m in noPheromone rewind tightTurn shrunkBounds; do \
	  if SIM_MUTANT=$$m $(BUILD)/tests > /dev/null 2>&1; then echo "mutant $$m ESCAPED"; exit 1; \
	  else echo "mutant $$m caught"; fi; \
	done

typecheck:
	$(SWIFTC) -typecheck -Xfrontend -warn-long-expression-type-checking=5 $(TEST_SRC)
	$(SWIFTC) -typecheck -Xfrontend -warn-long-expression-type-checking=5 $(RENDER_SRC)
	$(SWIFTC) -typecheck -Xfrontend -warn-long-expression-type-checking=5 $(RECORD_SRC)

XCODE := /Applications/Xcode.app/Contents/Developer
SWIFTC6 := $(XCODE)/Toolchains/XcodeDefault.xctoolchain/usr/bin/swiftc -O $(FRAMEWORKS) \
  -sdk $(XCODE)/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk
swift6:
	@mkdir -p $(BUILD)
	$(SWIFTC6) -Xfrontend -warn-long-expression-type-checking=5 $(RENDER_SRC) -o $(BUILD)/render6
	$(SWIFTC6) -Xfrontend -warn-long-expression-type-checking=5 $(TEST_SRC) -o $(BUILD)/tests6
	@$(BUILD)/tests6

clean:
	rm -rf $(BUILD)

.PHONY: test record scale report isolation mutants typecheck swift6 clean
