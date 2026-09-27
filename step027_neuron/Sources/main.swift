// Step 27: one neuron, rendered on the GPU and saved as a PNG.
//
//   .build/neuron [width height samples-per-side out]

import Foundation

let args: [String] = Array(CommandLine.arguments.dropFirst())

/// The nth argument as a whole number, or a default.
func argument(_ n: Int, _ fallback: Int) -> Int {
    guard n < args.count, let v = Int(args[n]) else { return fallback }
    return v
}

let width: Int = argument(0, 1920)
let height: Int = argument(1, 1080)
let samples: Int = argument(2, 3)
let out: String = args.count > 3 ? args[3] : "renders/neuron.png"

do {
    let device = try findDevice()
    let start = Date()
    let cell: Neuron = buildNeuron()
    let (image, gpu) = try renderNeuron(cell, width: width, height: height, samples: samples, on: device)
    drawScaleBar(image)
    try savePNG(image, to: URL(fileURLWithPath: out))
    let wall: Double = Date().timeIntervalSince(start)
    print("GPU: \(device.name)")
    print("\(width) × \(height), \(samples * samples) samples per pixel, \(cell.dendrites.count + cell.axon.count) pieces of process")
    print(String(format: "%.2f s on the GPU, %.2f s wall", gpu, wall))
    print("Saved \(out)")
} catch {
    FileHandle.standardError.write("neuron: \(error)\n".data(using: .utf8)!)
    exit(1)
}
