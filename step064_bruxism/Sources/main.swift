// Step 64: step 20's lower arch after decades of bruxism — one still, rendered
// on the GPU, labelled, and saved as a PNG.
//
//   .build/bruxism [width height samples-per-side out]

import Foundation

let args: [String] = Array(CommandLine.arguments.dropFirst())

/// The nth argument as a whole number, or a default.
func argument(_ n: Int, _ fallback: Int) -> Int {
    guard n < args.count, let v = Int(args[n]) else { return fallback }
    return v
}

let width: Int = argument(0, 1920)
let height: Int = argument(1, 1080)
let samples: Int = argument(2, 4)
let out: String = args.count > 3 ? args[3] : "renders/bruxism.png"

do {
    let device = try findDevice()
    let start = Date()
    let (image, gpu) = try renderMouth(width: width, height: height, samples: samples, depth: finalWearDepth, on: device)
    let survey: [ToothSurvey] = try surveyTeeth(depth: finalWearDepth, step: 0.1, on: device)
    guard let targets = labelTargets(survey) else { throw MouthError.gpu("no dentine or facet to label") }
    let k: Float = Float(height) / 1080
    let section: SectionGrid = try sampleSection(depth: finalWearDepth, perMillimetre: 30 * k, on: device)
    drawStillOverlay(image, targets: targets, section: section)
    try savePNG(image, to: URL(fileURLWithPath: out))
    let wall: Double = Date().timeIntervalSince(start)
    print("GPU: \(device.name)")
    print(String(format: "%d × %d, %d samples per pixel, 14 teeth worn %.2f mm (~%.0f years at %.0f µm/yr)",
                 width, height, samples * samples, finalWearDepth, finalWearYears, bruxistWearRate * 1000))
    print(String(format: "%.2f s on the GPU for the picture, %.2f s wall", gpu, wall))
    print("Saved \(out)")
} catch {
    FileHandle.standardError.write("bruxism: \(error)\n".data(using: .utf8)!)
    exit(1)
}
