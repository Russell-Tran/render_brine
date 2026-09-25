// Step 6a: the carbonic acid–bicarbonate buffer, rendered as an animated GIF.
//
// Water starts with 24 mM bicarbonate and 1.2 mM CO₂ (the amounts in human
// blood), at equilibrium. At t = 0.25 s, 10 mM of hydrochloric acid goes in.
// Each frame is 0.01 s of chemistry; played at 0.08 s per frame, the GIF runs
// 8× slower than real time.

import Foundation

let layout = FrameLayout(boxPixels: 400, panelWidth: 280)
let frameStep = 0.01      // seconds of chemistry per frame
let gifDelay = 0.08       // seconds each frame is shown
let totalFrames = 120
let doseFrame = 25
let doseMM = 10.0
let brownianStep: Float = 0.0035   // box widths per frame, for a diffusion coefficient of 1

do {
    let device = try findDevice()
    let box = try MoleculeBox(device: device, capacity: 4000)
    var rng = SplitMix64(seed: 2026)
    var state = BufferState(bicarbonate: 24, co2: 1.2)
    let carbonAtStart = state.totalCarbon
    let carbonDots = Int(dots(state.totalCarbon).rounded())
    let sodiumDots = Int(dots(state.sodium).rounded())

    try applyTargets(targetCounts(state, carbonDots: carbonDots, sodiumDots: sodiumDots, using: &rng), to: box, using: &rng)
    for i in 0..<box.used { box.slots[i].flash = 0 }   // no flashing on the first frame

    guard let frame = device.makeBuffer(length: layout.width * layout.height * 4, options: .storageModeShared) else {
        throw SimulationError.gpu("could not allocate the frame")
    }
    let gif = try GIFWriter(url: URL(fileURLWithPath: "renders/buffer.gif"), frameCount: totalFrames, delay: gifDelay)
    var history: [(time: Double, pH: Double)] = []
    var csv = "time_s,pH,co2_mM,carbonic_acid_mM,bicarbonate_mM,hydrogen_mM\n"
    var mismatches = 0
    var drawSeconds = 0.0
    var mostMolecules = 0
    let pHBefore = state.pH
    var pHRightAfter = 0.0

    for f in 0..<totalFrames {
        if f > 0 { state.advance(by: frameStep) }
        if f == doseFrame {
            state.addStrongAcid(doseMM)
            pHRightAfter = state.pH
        }
        let target = targetCounts(state, carbonDots: carbonDots, sodiumDots: sodiumDots, using: &rng)
        try applyTargets(target, to: box, using: &rng)
        try box.move(seed: UInt32(f + 1), baseStep: brownianStep)
        if try box.gpuCounts() != box.tally() { mismatches += 1 }
        mostMolecules = max(mostMolecules, box.tally().reduce(0, +))
        drawSeconds += try box.draw(into: frame, frameWidth: layout.width, boxPixels: layout.boxPixels)

        history.append((state.time, state.pH))
        let concentrations: [Species: Double] = [
            .co2: state.co2, .carbonicAcid: state.carbonicAcid, .bicarbonate: state.bicarbonate,
            .hydrogen: state.hydrogen, .sodium: state.sodium, .chloride: state.chloride,
        ]
        let info = PanelInfo(time: state.time, pH: state.pH, concentrations: concentrations, history: history,
                             doseTime: Double(doseFrame) * frameStep, doseMM: doseMM,
                             endTime: Double(totalFrames - 1) * frameStep, slowdown: gifDelay / frameStep)
        drawPanel(info, into: frame, layout: layout)
        gif.add(frame, layout: layout)
        csv += String(format: "%.3f,%.4f,%.5f,%.6f,%.5f,%.8f\n", state.time, state.pH, state.co2,
                      state.carbonicAcid, state.bicarbonate, state.hydrogen)
    }
    try gif.finish()
    try csv.write(toFile: "renders/ph.csv", atomically: true, encoding: .utf8)

    let bytes = (try? FileManager.default.attributesOfItem(atPath: "renders/buffer.gif")[.size] as? Int) ?? 0
    print("GPU: \(device.name)")
    print("Rendered \(totalFrames) frames at \(layout.width) × \(layout.height), up to \(mostMolecules) molecules")
    print(String(format: "pH before the acid:            %.2f", pHBefore))
    print(String(format: "pH right after 10 mM HCl:      %.2f", pHRightAfter))
    print(String(format: "pH at the end (t = %.2f s):     %.2f   (Henderson–Hasselbalch: %.2f)",
                 state.time, state.pH, state.hendersonHasselbalchPH))
    print(String(format: "Same acid in plain water:      %.2f", plainWaterPH(afterAcid: doseMM)))
    print(String(format: "Total carbon: %.6f mM at the start, %.6f mM at the end", carbonAtStart, state.totalCarbon))
    print(mismatches == 0 ? "GPU molecule counts matched the CPU's on every frame"
                          : "GPU molecule counts differed from the CPU's on \(mismatches) frames")
    print(String(format: "GPU drawing time: %.1f ms total, %.2f ms per frame", drawSeconds * 1000,
                 drawSeconds * 1000 / Double(totalFrames)))
    print("Saved renders/buffer.gif (\(bytes / 1024) KB) and renders/ph.csv")
} catch {
    FileHandle.standardError.write("buffer: \(error)\n".data(using: .utf8)!)
    exit(1)
}
