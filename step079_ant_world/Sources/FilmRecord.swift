// Step 79: THE RECORDED WORLD — everything the film shows, saved once, so
// every frame on every machine draws the same world.
//
// Found and decided (part 4): the simulation is bit-repeatable on one
// machine but not across machines. The mini ran seed 79 and got a different
// world (its trail formed at film 33.7 s, ours at 47.7 s): the ants' paths are
// chaotic, and last-bit differences between GPUs and compilers grow into
// different routes. So the film is rendered from a saved record, never by
// re-running the simulation, and the renderer is built WITHOUT the
// simulation's sources (the Makefile's render target; a test checks it).
//
// The file (records/world79.rec) holds:
//   * the simulation's record (SimFilmRecord): every ant at every film frame,
//     every ant's path with its stride of look-ahead, every dab, the grains;
//   * the pheromone field every `fieldStride` film frames, stored at a
//     fixed quantum of `fieldQuantum` marks/mm² in 16 bits.
// Between two stored fields a frame's field is rebuilt ON THE CPU, tick by
// tick, with the simulation's own two steps in a fixed order: the dabs of
// that tick deposited over their four cells, then diffusion and evaporation
// (SimField's `deposit` and `spread` kernels, as plain Float arithmetic), from
// the last stored field, with the simulation's α and decay factor stored in
// the file (not recomputed, so no libm difference can creep in). Nothing
// appears before it is laid. What differs from the live GPU field is the
// storage quantum and the last bits of Float arithmetic (the recorder
// measures it and writes it into records/MANIFEST).
//
// The whole file is compressed (LZMA, Apple's Compression framework, macOS
// 10.11+). Decompression is exact, so the bytes the renderer reads are the
// same everywhere; the file's SHA-256 is in records/MANIFEST.

import CryptoKit
import Foundation
import simd

enum FilmRecordError: Error, CustomStringConvertible {
    case badFile(String)
    var description: String {
        switch self { case .badFile(let s): return "film record: \(s)" }
    }
}

struct FilmRecord {
    /// A stored field every this many film frames (and at the last frame):
    /// 1/3 s of film, at most 300 ticks for the CPU to run on. MODEL, a
    /// trade of file size against CPU time per frame.
    static let fieldStride: Int = 10
    /// The stored field's step, marks/mm². MODEL: 1/200 of the simulation's
    /// detection threshold (0.02), far below anything the drawing can show
    /// (the tint's slope at zero moves an 8-bit pixel by < 0.2 of a level
    /// per step). 16 bits hold up to 6.55 marks/mm²; a test checks nothing
    /// is clipped.
    static let fieldQuantum: Float = 1e-4

    let sim: SimFilmRecord
    /// The simulation's diffusion number α = D·dt/h² and evaporation factor
    /// per tick, as the GPU field used them.
    let alpha: Float
    let decay: Float
    /// Film frames with a stored field, ascending, first 0 and last the
    /// film's last frame.
    let fieldFrames: [Int]
    /// Stored fields, quantised: value = q × fieldQuantum.
    let fields: [[UInt16]]

    var frameCount: Int { sim.frameCount }
    var gridWidth: Int { sim.config.gridWidth }
    var gridHeight: Int { sim.config.gridHeight }

    /// The film frames that get a stored field.
    static func fieldFrames(frameCount n: Int) -> [Int] {
        var out: [Int] = Array(stride(from: 0, to: n, by: fieldStride))
        if out.last != n - 1 { out.append(n - 1) }
        return out
    }

    static func quantise(_ v: [Float]) -> [UInt16] {
        v.map { x -> UInt16 in
            let q: Float = (max(x, 0) / fieldQuantum).rounded()
            return UInt16(min(q, 65535))
        }
    }

    /// The pheromone field at film frame `f`, marks/mm², row-major
    /// `gridWidth` × `gridHeight`: the stored field at or before `f`, run on
    /// tick by tick to `f` with the simulation's deposit and spread.
    /// `fromStored` picks an earlier stored field to start from (index into
    /// `fieldFrames`), for the tests.
    func field(atFrame f: Int, fromStored start: Int? = nil) -> [Float] {
        precondition(f >= 0 && f < frameCount, "frame \(f) outside the film")
        var k: Int = 0
        while k + 1 < fieldFrames.count && fieldFrames[k + 1] <= f { k += 1 }
        if let s = start, s >= 0, s <= k { k = s }
        let base: Int = fieldFrames[k]
        let q: Float = FilmRecord.fieldQuantum
        var cur: [Float] = fields[k].map { Float($0) * q }
        let from: Int = sim.ticks[base]
        let to: Int = sim.ticks[f]
        if to <= from { return cur }
        var next: [Float] = cur
        var d: Int = 0
        while d < sim.dabs.count && Int(sim.dabs[d].tick) < from { d += 1 }
        for t in from..<to {
            var batch: [SimDabEvent] = []
            while d < sim.dabs.count && Int(sim.dabs[d].tick) == t {
                batch.append(sim.dabs[d])
                d += 1
            }
            filmDeposit(&cur, width: gridWidth, height: gridHeight, dabs: batch)
            filmSpread(cur, into: &next, width: gridWidth, height: gridHeight, alpha: alpha, decay: decay)
            swap(&cur, &next)
        }
        return cur
    }

    // MARK: - the file

    private static let magic: UInt64 = 0x314D_4C49_4637_3953   // "S79FILM1"
    private static let version: UInt32 = 2

    /// The uncompressed payload.
    func payload() -> Data {
        var d = Data()
        func put<T>(_ v: T) {
            var x: T = v
            withUnsafeBytes(of: &x) { d.append(contentsOf: $0) }
        }
        let simBytes: Data = sim.encoded()
        put(alpha)
        put(decay)
        put(UInt64(simBytes.count))
        d.append(simBytes)
        put(UInt64(fieldFrames.count))
        for f in fieldFrames { put(Int64(f)) }
        let cells: Int = gridWidth * gridHeight
        put(UInt64(cells))
        // Each field as its high bytes then its low bytes: the high bytes are
        // mostly zero and compress far better apart.
        for v in fields {
            d.append(contentsOf: v.map { UInt8($0 >> 8) })
            d.append(contentsOf: v.map { UInt8($0 & 0xFF) })
        }
        return d
    }

    func encoded() throws -> Data {
        let raw = payload() as NSData
        let packed: Data
        do { packed = try raw.compressed(using: .lzma) as Data } catch {
            throw FilmRecordError.badFile("compression failed: \(error)")
        }
        var d = Data()
        var m: UInt64 = FilmRecord.magic
        var v: UInt32 = FilmRecord.version
        withUnsafeBytes(of: &m) { d.append(contentsOf: $0) }
        withUnsafeBytes(of: &v) { d.append(contentsOf: $0) }
        d.append(packed)
        return d
    }

    func save(to url: URL) throws {
        try? FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try encoded().write(to: url, options: .atomic)
    }

    static func decode(_ data: Data) throws -> FilmRecord {
        guard data.count > 12 else { throw FilmRecordError.badFile("too short") }
        let m: UInt64 = data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: 0, as: UInt64.self) }
        let v: UInt32 = data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: 8, as: UInt32.self) }
        guard m == magic else { throw FilmRecordError.badFile("not a step 79 film record") }
        guard v == version else { throw FilmRecordError.badFile("version \(v)") }
        let body = data.subdata(in: 12..<data.count) as NSData
        let raw: Data
        do { raw = try body.decompressed(using: .lzma) as Data } catch {
            throw FilmRecordError.badFile("decompression failed: \(error)")
        }
        var at: Int = 0
        func get<T>(_: T.Type) throws -> T {
            let n: Int = MemoryLayout<T>.size
            guard at + n <= raw.count else { throw FilmRecordError.badFile("truncated") }
            let x: T = raw.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: at, as: T.self) }
            at += n
            return x
        }
        let alpha: Float = try get(Float.self)
        let decay: Float = try get(Float.self)
        let simCount: Int = Int(try get(UInt64.self))
        guard simCount >= 0, at + simCount <= raw.count else { throw FilmRecordError.badFile("truncated record") }
        let sim: SimFilmRecord = try SimFilmRecord.decode(raw.subdata(in: at..<(at + simCount)))
        at += simCount
        let nf: Int = Int(try get(UInt64.self))
        var frames: [Int] = []
        for _ in 0..<nf { frames.append(Int(try get(Int64.self))) }
        let cells: Int = Int(try get(UInt64.self))
        guard cells == sim.config.gridWidth * sim.config.gridHeight else { throw FilmRecordError.badFile("grid size") }
        guard at + nf * cells * 2 == raw.count else { throw FilmRecordError.badFile("field bytes") }
        var fields: [[UInt16]] = []
        raw.withUnsafeBytes { buf in
            for k in 0..<nf {
                let hi: Int = at + k * cells * 2
                let lo: Int = hi + cells
                var v: [UInt16] = []
                v.reserveCapacity(cells)
                for c in 0..<cells {
                    let a: UInt16 = UInt16(buf[hi + c])
                    let b: UInt16 = UInt16(buf[lo + c])
                    v.append((a << 8) | b)
                }
                fields.append(v)
            }
        }
        let lastFrame: Int = sim.frameCount - 1
        let firstStored: Int = frames.first ?? -1
        let lastStored: Int = frames.last ?? -1
        guard firstStored == 0, lastStored == lastFrame else { throw FilmRecordError.badFile("field frames") }
        return FilmRecord(sim: sim, alpha: alpha, decay: decay, fieldFrames: frames, fields: fields)
    }

    static func load(from url: URL) throws -> FilmRecord {
        try decode(try Data(contentsOf: url))
    }

    /// Hex SHA-256 of a file's bytes, for records/MANIFEST.
    static func sha256(of url: URL) throws -> String {
        let data: Data = try Data(contentsOf: url)
        return SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
}

/// Where the film's record lives, relative to step079_ant_world/.
let filmRecordPath: String = "records/world79.rec"
let filmManifestPath: String = "records/MANIFEST"

/// The SHA-256 the manifest names (its "sha256 <hex>" line).
func filmManifestHash(_ url: URL) throws -> String {
    let text: String = try String(contentsOf: url, encoding: .utf8)
    for line in text.split(separator: "\n") {
        let parts = line.split(separator: " ")
        if parts.count >= 2 && parts[0] == "sha256" { return String(parts[1]) }
    }
    throw FilmRecordError.badFile("no sha256 line in the manifest")
}

/// Load the film's record, refusing one whose hash is not the manifest's.
func filmLoadVerified(directory: URL) throws -> FilmRecord {
    let rec: URL = directory.appendingPathComponent(filmRecordPath)
    let man: URL = directory.appendingPathComponent(filmManifestPath)
    let want: String = try filmManifestHash(man)
    let have: String = try FilmRecord.sha256(of: rec)
    guard want == have else { throw FilmRecordError.badFile("\(filmRecordPath) hash \(have) is not the manifest's \(want)") }
    return try FilmRecord.load(from: rec)
}

/// One tick's dabs into the field: SimField.gpuDab's bilinear weights, and
/// the deposit kernel's order — per cell, the dabs' weights summed in the
/// order given, then added.
func filmDeposit(_ c: inout [Float], width w: Int, height h: Int, dabs: [SimDabEvent]) {
    if dabs.isEmpty { return }
    let cell: Float = SimConst.cell
    var adds: [Int: Float] = [:]
    var order: [Int] = []
    for dab in dabs {
        let fx: Float = dab.x / cell - 0.5
        let fy: Float = dab.z / cell - 0.5
        let x0: Float = fx.rounded(.down)
        let y0: Float = fy.rounded(.down)
        let tx: Float = fx - x0
        let ty: Float = fy - y0
        let density: Float = dab.marks / (cell * cell)
        let ux: Float = 1 - tx
        let uy: Float = 1 - ty
        let w00: Float = ux * uy * density
        let w10: Float = tx * uy * density
        let w01: Float = ux * ty * density
        let w11: Float = tx * ty * density
        let wts: [Float] = [w00, w10, w01, w11]
        let i0: Int = Int(x0)
        let j0: Int = Int(y0)
        for k in 0..<4 {
            let i: Int = i0 + k % 2
            let j: Int = j0 + k / 2
            if i < 0 || j < 0 || i >= w || j >= h { continue }
            let at: Int = j * w + i
            if let a = adds[at] { adds[at] = a + wts[k] } else { adds[at] = wts[k]; order.append(at) }
        }
    }
    for at in order { c[at] = c[at] + (adds[at] ?? 0) }
}

/// Diffusion and evaporation, one tick: the spread kernel's arithmetic, with
/// no flux through the edges.
func filmSpread(_ src: [Float], into dst: inout [Float], width w: Int, height h: Int, alpha: Float, decay: Float) {
    src.withUnsafeBufferPointer { s in
        dst.withUnsafeMutableBufferPointer { o in
            for j in 0..<h {
                for i in 0..<w {
                    let k: Int = j * w + i
                    let c: Float = s[k]
                    var lap: Float = 0
                    if i > 0 { lap += s[k - 1] - c }
                    if i + 1 < w { lap += s[k + 1] - c }
                    if j > 0 { lap += s[k - w] - c }
                    if j + 1 < h { lap += s[k + w] - c }
                    let moved: Float = c + alpha * lap
                    o[k] = moved * decay
                }
            }
        }
    }
}
