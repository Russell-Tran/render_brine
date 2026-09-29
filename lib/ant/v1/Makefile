# lib/ant/v1: the shared black garden ant (Lasius niger worker), version 1.
# Frozen once committed: a change makes lib/ant/v2 and never edits this
# folder (../../README).
#
# A scene uses it by compiling its three sources with its own, e.g. in the
# scene's Makefile:
#   ANT := ../lib/ant/v1/Sources
#   SOURCES := $(ANT)/AntV1Anatomy.swift $(ANT)/AntV1Pose.swift $(ANT)/AntV1Metal.swift ...
# and pasting AntV1.metalSource into its kernel text.
#
#   make test       build and run the module's tests (these read step 55's
#                   Sources, unchanged, to check the module against its lineage)
#   make mutants    break the gait and the antenna; the tests must catch each
#   make typecheck  prove no expression is slow to type-check
#   make swift6     build the tests with Xcode's newer Swift as well
#
# On the laptop, point swiftc at the SDK that matches the compiler:
#   make test SDK=$(xcrun --show-sdk-path)

SWIFTC := swiftc -O -framework CoreGraphics -framework Metal $(if $(SDK),-sdk $(SDK))
BUILD := .build
SOURCES := Sources/AntV1Anatomy.swift Sources/AntV1Pose.swift Sources/AntV1Metal.swift
# The lineage, compiled in read-only for the golden tests.
LINEAGE := ../../../step055_ant_trail_5/Sources
GOLDEN := $(LINEAGE)/Anatomy.swift $(LINEAGE)/Walk.swift $(LINEAGE)/Render.swift
TESTS := Tests/Harness.swift Tests/GPU.swift Tests/main.swift

# A newer compiler, where one is installed (Xcode's, on the laptop).
XCODE := /Applications/Xcode.app/Contents/Developer
SWIFTC6 := $(XCODE)/Toolchains/XcodeDefault.xctoolchain/usr/bin/swiftc -O -framework CoreGraphics -framework Metal \
  -sdk $(XCODE)/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk

$(BUILD)/tests: $(SOURCES) $(GOLDEN) $(TESTS) Makefile
	@mkdir -p $(BUILD)
	$(SWIFTC) $(SOURCES) $(GOLDEN) $(TESTS) -o $@

test: $(BUILD)/tests
	@$<

# Each mutant must make the suite FAIL. A mutant that passes is a test missing.
mutants: $(BUILD)/tests
	@for m in nonTripod sliding segments13; do \
	  if ANT_MUTANT=$$m $< > /dev/null 2>&1; then echo "mutant $$m ESCAPED"; exit 1; \
	  else echo "mutant $$m caught"; fi; \
	done

typecheck:
	$(SWIFTC) -typecheck -Xfrontend -warn-long-expression-type-checking=5 $(SOURCES)

swift6:
	@mkdir -p $(BUILD)
	$(SWIFTC6) -Xfrontend -warn-long-expression-type-checking=5 $(SOURCES) $(GOLDEN) $(TESTS) -o $(BUILD)/tests6
	@$(BUILD)/tests6

clean:
	rm -rf $(BUILD)

.PHONY: test mutants typecheck swift6 clean
