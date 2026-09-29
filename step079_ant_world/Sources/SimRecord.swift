// Step 79: running the simulation into a record. Run the world once through
// the film's clock and keep, for every film frame, each ant's ground position,
// heading, distance walked and state; keep every dab; and replay the
// pheromone field from the dabs on demand (it depends on nothing else, so the
// replay is bit-identical to the live field on the same machine — a test
// checks it). The record's types and file format are in SimFilmRecord.swift;
// the film renders from a SAVED record (FilmRecord.swift), never from a run.

import Foundation
import simd

/// Everything a run produces: the record, and any field snapshots asked for.
struct SimRunResult {
    var record: SimFilmRecord
    var fields: [Int: [Float]]
}

func simAntFrame(_ a: SimAnt) -> SimAntFrame {
    let amount: Float = SimAnt.dabProfile(a.dabClock).dab
    let dab: Float = a.dabbing ? max(amount, 1e-6) : 0
    let k: Int = max(a.outing - 1, 0)
    let outing: UInt32 = UInt32(k)
    return SimAntFrame(x: a.pos.x, z: a.pos.y, yaw: a.yaw, distance: a.distance, stateRaw: a.state.rawValue,
                       carrying: a.carrying ? 1 : 0, layer: a.layer ? 1 : 0, fed: a.fed ? 1 : 0, dab: dab,
                       outing: outing)
}

/// Ticks the tracks run past the last film frame: 1 s of the ants' time,
/// 30 mm of walking — far more than the 2.53 mm stride a swinging foot looks
/// ahead (lib/ant/v1).
let simRunAheadTicks: Int = 60

/// Run the world through the film's clock. `keepFieldsAt` names film frames
/// whose pheromone field is kept as well.
func simRun(config: SimWorldConfig, clock: SimClock, keepFieldsAt: Set<Int> = []) throws -> SimRunResult {
    let world = try SimWorld(config: config)
    let n: Int = clock.frameCount
    var ticks: [Int] = []
    ticks.reserveCapacity(n)
    for f in 0..<n { ticks.append(clock.tick(atFrame: f)) }
    // Snapshots are taken on the way forward, at every tick a frame needs
    // (with the rewind mutant a later frame can need an earlier tick).
    let needed: [Int] = Array(Set(ticks)).sorted()
    var fieldTicks: Set<Int> = []
    for f in keepFieldsAt where f >= 0 && f < n { fieldTicks.insert(ticks[f]) }
    var byTick: [Int: [SimAntFrame]] = [:]
    var fieldsByTick: [Int: [Float]] = [:]
    for t in needed {
        while world.tick < t { try world.step() }
        byTick[t] = world.ants.map { simAntFrame($0) }
        if fieldTicks.contains(t) { fieldsByTick[t] = world.field.snapshot() }
    }
    var frames: [SimAntFrame] = []
    frames.reserveCapacity(n * config.antCount)
    var fields: [Int: [Float]] = [:]
    for f in 0..<n {
        frames.append(contentsOf: byTick[ticks[f]] ?? [])
        if keepFieldsAt.contains(f) { fields[f] = fieldsByTick[ticks[f]] }
    }
    // Only dabs up to the last tick shown belong to the film.
    let last: Int = ticks.max() ?? 0
    let dabs: [SimDabEvent] = world.dabs.filter { Int($0.tick) < last }
    // Run on past the end for the tracks only.
    for _ in 0..<simRunAheadTicks { try world.step() }
    let rec = SimFilmRecord(config: config, clock: clock, grains: world.grains, antCount: config.antCount,
                            ticks: ticks, antFrames: frames, dabs: dabs, tracks: world.track,
                            firstContact: world.firstContact, returns: world.returns)
    return SimRunResult(record: rec, fields: fields)
}

/// The film's run: the default world (20 workers, seed 79) through the film
/// clock. `mutant` defaults to SIM_MUTANT from the environment.
func simFilmRun(seed: UInt64 = 79, mutant: SimMutant = SimMutant.fromEnvironment(),
                keepFieldsAt: Set<Int> = []) throws -> SimRunResult {
    var config = SimWorldConfig()
    config.seed = seed
    config.mutant = mutant
    var clock: SimClock = SimClock.film
    clock.mutant = mutant
    return try simRun(config: config, clock: clock, keepFieldsAt: keepFieldsAt)
}

// MARK: - replaying the field

/// Rebuilds the pheromone field at any film frame from the record's dabs,
/// running the same GPU kernels in the same order as the live run. Moving
/// forward is incremental; asking for an earlier frame starts again.
final class SimFieldPlayer {
    let record: SimFilmRecord
    let field: SimField
    private var tick: Int = 0
    private var nextDab: Int = 0

    init(record: SimFilmRecord) throws {
        self.record = record
        field = try SimField.forSimulation(width: record.config.gridWidth, height: record.config.gridHeight)
    }

    var width: Int { field.width }
    var height: Int { field.height }
    /// Millimetres per cell.
    var cell: Float { field.cell }

    private func reset() throws {
        field.load([Float](repeating: 0, count: field.count))
        tick = 0
        nextDab = 0
    }

    /// Advance to simulation tick `t`.
    func advance(toTick t: Int) throws {
        if t < tick { try reset() }
        while tick < t {
            var batch: [SimDabGPU] = []
            while nextDab < record.dabs.count && Int(record.dabs[nextDab].tick) == tick {
                let d: SimDabEvent = record.dabs[nextDab]
                batch.append(field.gpuDab(x: d.x, z: d.z, marks: d.marks))
                nextDab += 1
            }
            try field.step(dabs: batch)
            tick += 1
        }
    }

    /// The field (marks/mm², row-major, `width` × `height`) at film frame `f`.
    func field(atFrame f: Int) throws -> [Float] {
        try advance(toTick: record.ticks[f])
        return field.snapshot()
    }
}
