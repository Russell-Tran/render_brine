// Step 79: record the world ONCE — run the simulation (seed 79) through the
// film's clock and save everything the film shows to records/world79.rec,
// with its SHA-256 in records/MANIFEST. The renderer reads only that file.
//
//   make record SDK=$(xcrun --show-sdk-path)
//
// Re-running this on another machine makes a DIFFERENT world (see
// FilmRecord.swift); the committed record is the film's.

import Foundation
import simd

func fmt(_ x: Double, _ d: Int = 3) -> String { String(format: "%.\(d)f", x) }

let dir = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
let started: Date = Date()
let n: Int = SimClock.film.frameCount
let stored: [Int] = FilmRecord.fieldFrames(frameCount: n)
// Also keep the live field at the frames furthest from a stored one, to
// measure what the rebuild between stored fields leaves out.
var probes: [Int] = []
for s in stride(from: 0, to: n - FilmRecord.fieldStride, by: 50 * FilmRecord.fieldStride) {
    probes.append(s + FilmRecord.fieldStride - 1)
}
for s in stride(from: 1500, to: n - FilmRecord.fieldStride, by: 5 * FilmRecord.fieldStride) {
    probes.append(s + FilmRecord.fieldStride - 1)
}
let run: SimRunResult = try simFilmRun(seed: 79, mutant: .none, keepFieldsAt: Set(stored + probes))
let simSeconds: Double = Date().timeIntervalSince(started)
var fields: [[UInt16]] = []
var peak: Float = 0
for f in stored {
    guard let v = run.fields[f] else { print("missing field at frame \(f)"); exit(1) }
    peak = max(peak, v.max() ?? 0)
    fields.append(FilmRecord.quantise(v))
}
let live: SimField = try SimField.forSimulation(width: run.record.config.gridWidth, height: run.record.config.gridHeight)
let film = FilmRecord(sim: run.record, alpha: live.alpha, decay: live.decay, fieldFrames: stored, fields: fields)
let url: URL = dir.appendingPathComponent(filmRecordPath)
try film.save(to: url)
let size: Int = (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? Int) ?? -1
let raw: Int = film.payload().count
let hash: String = try FilmRecord.sha256(of: url)

// What the rebuild between stored fields leaves out, against the live field.
var worstC: Float = 0
var worstTint: Float = 0
var sumTint: Double = 0
var cells: Int = 0
for f in probes {
    guard let live = run.fields[f] else { continue }
    let rebuilt: [Float] = film.field(atFrame: f)
    for k in 0..<live.count {
        let dc: Float = abs(rebuilt[k] - live[k])
        worstC = max(worstC, dc)
        let dt: Float = abs(worldTrailTint(rebuilt[k]) - worldTrailTint(live[k]))
        worstTint = max(worstTint, dt)
        sumTint += Double(dt)
        cells += 1
    }
}
let cellCount: Int = max(cells, 1)
let meanTint: Double = sumTint / Double(cellCount)
let conc: SimConcentration = simConcentration(run.record)
let formed: String = conc.formedFrame.map { "film frame \($0) = film \(fmt(Double($0) / 30, 1)) s = ant time \(fmt(run.record.simSeconds(atFrame: $0), 0)) s" } ?? "never"

var m: String = ""
m += "# step 79's recorded world. The film renders from this file only; re-running the\n"
m += "# simulation on another machine gives a different world (see Sources/FilmRecord.swift).\n"
m += "file \(filmRecordPath)\n"
m += "sha256 \(hash)\n"
m += "bytes \(size)\n"
m += "uncompressed \(raw)\n"
m += "seed 79\n"
m += "frames \(n)\n"
m += "stored_fields \(stored.count) (every \(FilmRecord.fieldStride) frames and the last; quantum \(FilmRecord.fieldQuantum) marks/mm2; peak \(peak))\n"
m += "trail_formed \(formed)\n"
m += "dabs \(run.record.dabs.count)\n"
m += "rebuild_vs_live worst \(worstC) marks/mm2; drawn tint worst \(worstTint), mean \(meanTint) over \(probes.count) frames\n"
m += "made_on M3 Max laptop (Swift 5.10 toolchain, macOS 14.5)\n"
m += "regenerate: make record SDK=$(xcrun --show-sdk-path)   (on the laptop only; elsewhere it is a different world)\n"
try m.write(to: dir.appendingPathComponent(filmManifestPath), atomically: true, encoding: .utf8)
print(m, terminator: "")
print("simulation \(fmt(simSeconds, 1)) s; record \(size / 1024) KB compressed, \(raw / 1024) KB raw")
if peak >= Float(65535) * FilmRecord.fieldQuantum { print("WARNING: the field peak \(peak) was clipped"); exit(1) }
