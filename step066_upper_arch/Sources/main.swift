// Step 66: one still of the upper arch, rendered on the GPU and saved as a PNG.
//
//   .build/upper_arch [width height samples-per-side out]

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
let out: String = args.count > 3 ? args[3] : "renders/upper_arch.png"

if archMutant != .none {
    FileHandle.standardError.write("upper_arch: UPPER_MUTANT=\(archMutant.rawValue) is set; this picture is broken on purpose\n".data(using: .utf8)!)
}

do {
    let device = try findDevice()
    let start = Date()
    let renderer = try MouthRenderer(on: device)
    // CAMERA="x,y,z,tx,ty,tz" overrides the still's camera, for trying framings.
    var camera: Camera = stillCamera
    if let spec = ProcessInfo.processInfo.environment["CAMERA"] {
        let v: [Float] = spec.split(separator: ",").compactMap { Float($0) }
        if v.count == 6 {
            camera = Camera(position: SIMD3<Float>(v[0], v[1], v[2]), target: SIMD3<Float>(v[3], v[4], v[5]))
        }
    }
    let (image, gpu) = try renderer.render(width: width, height: height, samples: samples, camera: camera)
    try savePNG(image, to: URL(fileURLWithPath: out))
    let wall: Double = Date().timeIntervalSince(start)
    print("GPU: \(device.name)")
    print("\(width) × \(height), \(samples * samples) samples per pixel, \(placeTeeth().count) teeth, \(rugae.count) rugae")
    print(String(format: "%.2f s on the GPU, %.2f s wall", gpu, wall))
    print("Saved \(out)")
} catch {
    FileHandle.standardError.write("upper_arch: \(error)\n".data(using: .utf8)!)
    exit(1)
}
