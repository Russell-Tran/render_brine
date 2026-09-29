// Step 79: the simulation's data types — what an ant is doing, the grains,
// the world's layout, a path sample, a dab. Moved here from SimWorld.swift
// (unchanged) so the film's renderer can read a saved record without
// compiling the simulation itself.

import Foundation
import simd

/// What an ant is doing. Stored as a byte in the record.
enum SimAntState: UInt8 {
    case inNest = 0
    case searching = 1
    case following = 2
    case tasting = 3
    case returning = 4
}

/// One sugar grain in the pile, for contact and for the film to draw with
/// step 26's crystal (sieve size, rest face and turn as step 26 places one).
struct SimGrain: Equatable {
    /// Centre on the ground, mm.
    var x: Float
    var z: Float
    /// Sieve size (step 26's `grainSieveRange` 0.30–0.67 mm).
    var size: Float
    /// Turn about the vertical, radians.
    var yaw: Float
    /// Which of step 26's four rest faces it lies on: 0 (−1,0,0), 1 (0,0,−1),
    /// 2 (1,0,0), 3 (0,0,1).
    var restFace: UInt8
}

/// Where everything is. Defaults are the film's world. MODEL throughout: a
/// composition, not a measurement.
struct SimWorldConfig: Equatable {
    var seed: UInt64 = 79
    /// Arena, mm (16:9, the film's wide shot).
    var width: Float = 240
    var depth: Float = 135
    /// Nest entrance centre and radius, mm.
    var nest: SIMD2<Float> = SIMD2<Float>(40, 78)
    var nestRadius: Float = 2.0
    /// Sugar pile centre and radius, mm, and how many grains.
    var sugar: SIMD2<Float> = SIMD2<Float>(192, 56)
    var sugarRadius: Float = 5.0
    var grainCount: Int = 200
    /// Workers in all, and how many of them start outside as scouts; the
    /// scouts leave the nest `scoutSpacing` seconds apart.
    var antCount: Int = 20
    var scouts: Int = 4
    var scoutSpacing: Float = 1.5
    /// Keep antennae and gaster inside the arena: the edge turns an ant back
    /// this far in, mm.
    var edgeMargin: Float = 4.0
    var mutant: SimMutant = SimMutant.fromEnvironment()

    /// Grid size for this arena.
    var gridWidth: Int { Int((width / SimConst.cell).rounded(.up)) }
    var gridHeight: Int { Int((depth / SimConst.cell).rounded(.up)) }
}

/// One sample of an ant's path, taken every tick it moves (and when it comes
/// out of the nest): what the film turns into a lib/ant/v1 Track, one per
/// (ant, outing). Distances strictly increase within an outing.
struct SimTrackPoint: Equatable {
    var tick: UInt32
    var outing: UInt32
    var distance: Float
    var x: Float
    var z: Float
    var yaw: Float
}

/// A dab as recorded: when and where, and how much.
struct SimDabEvent: Equatable {
    var tick: UInt32
    var x: Float
    var z: Float
    var marks: Float
}
