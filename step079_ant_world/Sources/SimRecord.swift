// Step 79: what the film reads. Run the simulation once through the film's
// clock and keep, for every film frame, each ant's ground position, heading,
// distance walked and state; keep every dab; and replay the pheromone field
// from the dabs on demand (it depends on nothing else, so the replay is
// bit-identical to the live field — a test checks it). The record saves to one
// small binary file and loads back exactly, for the resumable renderer.

import Foundation
import simd

/// One ant at one film frame. 28 bytes (no padding), stored raw.
struct SimAntFrame: Equatable {
    /// Ground position, mm (x across, z the ground's other axis; y = 0).
    var x: Float
    var z: Float
    /// Heading, radians: body +x points along (cos yaw, 0, sin yaw).
    var yaw: Float
    /// Distance walked since the start, mm — drives the gait (feet by
    /// distance, as in step 44). Constant while the ant stands.
    var distance: Float
    /// SimAntState raw value.
    var stateRaw: UInt8
    /// 1 while carrying sugar home.
    var carrying: UInt8
    /// 1 if this ant ever lays trail (86% of workers do).
    var layer: UInt8
    /// 1 once it has fed at the sugar.
    var fed: UInt8
    /// The dab amount, lib/ant/v1's `State.dab`: 1 = gaster tip on the
    /// ground; 0 = not dabbing. While an ant is in a dab's 0.2 s stop this is
    /// at least 1e-6, so `dabbing` tells a stop from ordinary walking.
    var dab: Float
    /// Which time out of the nest this is (0 = the first): one film track per
    /// (ant, outing). Meaningless while in the nest.
    var outing: UInt32

    var state: SimAntState { SimAntState(rawValue: stateRaw) ?? .inNest }
    var visible: Bool { state != .inNest }
    var dabbing: Bool { dab > 0 }
    var position: SIMD2<Float> { SIMD2<Float>(x, z) }
}

struct SimFilmRecord: Equatable {
    var config: SimWorldConfig
    var clock: SimClock
    var grains: [SimGrain]
    var antCount: Int
    /// Simulation tick shown at each film frame.
    var ticks: [Int]
    /// frames × antCount, frame-major.
    var antFrames: [SimAntFrame]
    var dabs: [SimDabEvent]
    /// Every ant's path, one sample per tick it walked (SimTrackPoint), run
    /// `runAheadTicks` past the last frame so every frame's swinging feet
    /// have the path one stride ahead of them.
    var tracks: [[SimTrackPoint]]
    /// Simulated second of the first sugar contact, and successful returns by the end.
    var firstContact: Double?
    var returns: Int

    var frameCount: Int { ticks.count }

    func ants(atFrame f: Int) -> ArraySlice<SimAntFrame> {
        let a: Int = f * antCount
        return antFrames[a..<(a + antCount)]
    }

    func ant(_ i: Int, atFrame f: Int) -> SimAntFrame { antFrames[f * antCount + i] }

    func simSeconds(atFrame f: Int) -> Double { Double(ticks[f]) * SimConst.tick }
}

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

extension SimFilmRecord {
    /// Ant `i`'s path on outing `k`, for a lib/ant/v1 Track: append each
    /// point as `track.append(distance: p.distance, AntV1.Ground(SIMD2(p.x,
    /// p.z), heading: p.yaw))`. Distances strictly increase.
    func trackPoints(ant i: Int, outing k: Int) -> [SimTrackPoint] {
        tracks[i].filter { Int($0.outing) == k }
    }

    /// How many outings ant `i` made (including one still under way).
    func outings(ant i: Int) -> Int {
        Int(tracks[i].last?.outing ?? 0) + (tracks[i].isEmpty ? 0 : 1)
    }
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

// MARK: - saving and loading

enum SimRecordError: Error {
    case badFile(String)
}

private struct SimWriter {
    var data = Data()
    mutating func put<T>(_ v: T) {
        var x: T = v
        withUnsafeBytes(of: &x) { data.append(contentsOf: $0) }
    }
    mutating func putArray<T>(_ a: [T]) {
        put(UInt64(a.count))
        a.withUnsafeBytes { data.append(contentsOf: $0) }
    }
}

private struct SimReader {
    let data: Data
    var at: Int = 0
    mutating func get<T>(_: T.Type) throws -> T {
        let n: Int = MemoryLayout<T>.size
        guard at + n <= data.count else { throw SimRecordError.badFile("truncated") }
        var v: T? = nil
        data.withUnsafeBytes { raw in
            v = raw.loadUnaligned(fromByteOffset: at, as: T.self)
        }
        at += n
        guard let out = v else { throw SimRecordError.badFile("read") }
        return out
    }
    mutating func getArray<T>(_: T.Type) throws -> [T] {
        let count: Int = Int(try get(UInt64.self))
        let n: Int = count * MemoryLayout<T>.stride
        guard count >= 0, at + n <= data.count else { throw SimRecordError.badFile("truncated array") }
        var out: [T] = []
        out.reserveCapacity(count)
        data.withUnsafeBytes { raw in
            for k in 0..<count {
                out.append(raw.loadUnaligned(fromByteOffset: at + k * MemoryLayout<T>.stride, as: T.self))
            }
        }
        at += n
        return out
    }
}

private let simMagic: UInt64 = 0x3139_4B52_5437_3953   // "S79TRK19"
private let simFormatVersion: UInt32 = 1

extension SimFilmRecord {
    func encoded() -> Data {
        var w = SimWriter()
        w.put(simMagic)
        w.put(simFormatVersion)
        let c: SimWorldConfig = config
        w.put(c.seed)
        w.putArray([c.width, c.depth, c.nest.x, c.nest.y, c.nestRadius, c.sugar.x, c.sugar.y,
                    c.sugarRadius, c.scoutSpacing, c.edgeMargin])
        w.putArray([Int64(c.grainCount), Int64(c.antCount), Int64(c.scouts)])
        w.putArray(Array(c.mutant.rawValue.utf8))
        w.putArray(clock.segments.map { SIMD2<Int64>(Int64($0.frames), Int64($0.ticksPerFrame)) })
        w.putArray(Array(clock.mutant.rawValue.utf8))
        // Field by field: SimGrain's stride has padding bytes whose contents
        // are undefined, which would make equal records encode differently.
        w.putArray(grains.map { SIMD4<Float>($0.x, $0.z, $0.size, $0.yaw) })
        w.putArray(grains.map { $0.restFace })
        w.put(Int64(antCount))
        w.putArray(ticks.map { Int64($0) })
        w.putArray(antFrames)
        w.putArray(dabs)
        w.put(Int64(tracks.count))
        for tr in tracks { w.putArray(tr) }
        w.put(firstContact ?? -1.0)
        w.put(Int64(returns))
        return w.data
    }

    func save(to url: URL) throws {
        try encoded().write(to: url, options: .atomic)
    }

    static func load(from url: URL) throws -> SimFilmRecord {
        try decode(try Data(contentsOf: url))
    }

    static func decode(_ data: Data) throws -> SimFilmRecord {
        var r = SimReader(data: data)
        guard try r.get(UInt64.self) == simMagic else { throw SimRecordError.badFile("not a step 79 track") }
        guard try r.get(UInt32.self) == simFormatVersion else { throw SimRecordError.badFile("version") }
        var c = SimWorldConfig()
        c.seed = try r.get(UInt64.self)
        let f: [Float] = try r.getArray(Float.self)
        guard f.count == 10 else { throw SimRecordError.badFile("config") }
        c.width = f[0]; c.depth = f[1]
        c.nest = SIMD2<Float>(f[2], f[3]); c.nestRadius = f[4]
        c.sugar = SIMD2<Float>(f[5], f[6]); c.sugarRadius = f[7]
        c.scoutSpacing = f[8]; c.edgeMargin = f[9]
        let i: [Int64] = try r.getArray(Int64.self)
        guard i.count == 3 else { throw SimRecordError.badFile("config counts") }
        c.grainCount = Int(i[0]); c.antCount = Int(i[1]); c.scouts = Int(i[2])
        let m: [UInt8] = try r.getArray(UInt8.self)
        c.mutant = SimMutant(rawValue: String(decoding: m, as: UTF8.self)) ?? .none
        let segs: [SIMD2<Int64>] = try r.getArray(SIMD2<Int64>.self)
        let cm: [UInt8] = try r.getArray(UInt8.self)
        var clock = SimClock(segments: segs.map { SimClockSegment(frames: Int($0.x), ticksPerFrame: Int($0.y)) })
        clock.mutant = SimMutant(rawValue: String(decoding: cm, as: UTF8.self)) ?? .none
        let gv: [SIMD4<Float>] = try r.getArray(SIMD4<Float>.self)
        let gf: [UInt8] = try r.getArray(UInt8.self)
        guard gv.count == gf.count else { throw SimRecordError.badFile("grains") }
        var grains: [SimGrain] = []
        for k in 0..<gv.count {
            grains.append(SimGrain(x: gv[k].x, z: gv[k].y, size: gv[k].z, yaw: gv[k].w, restFace: gf[k]))
        }
        let antCount: Int = Int(try r.get(Int64.self))
        let ticks: [Int] = try r.getArray(Int64.self).map { Int($0) }
        let frames: [SimAntFrame] = try r.getArray(SimAntFrame.self)
        let dabs: [SimDabEvent] = try r.getArray(SimDabEvent.self)
        let trackCount: Int = Int(try r.get(Int64.self))
        var tracks: [[SimTrackPoint]] = []
        for _ in 0..<max(trackCount, 0) { tracks.append(try r.getArray(SimTrackPoint.self)) }
        let fc: Double = try r.get(Double.self)
        let returns: Int = Int(try r.get(Int64.self))
        guard frames.count == ticks.count * antCount else { throw SimRecordError.badFile("frame count") }
        return SimFilmRecord(config: c, clock: clock, grains: grains, antCount: antCount, ticks: ticks,
                             antFrames: frames, dabs: dabs, tracks: tracks, firstContact: fc < 0 ? nil : fc, returns: returns)
    }
}
