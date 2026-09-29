// Step 79: the record the film reads — for every film frame each ant's ground
// position, heading, distance walked and state; every dab; every ant's path
// (one lib/ant/v1 Track per outing). It saves to one small binary file and
// loads back exactly. Moved here from SimRecord.swift (unchanged) so the
// renderer can read a record without compiling the simulation.

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
