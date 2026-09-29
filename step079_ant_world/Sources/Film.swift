// Step 79: THE FILM — about a minute, 1920 × 1080, 30 frames a second, drawn
// frame by frame from the recorded world (records/world79.rec), with the
// camera, the captions, the resumable frame store and the MP4.
//
// What it shows, frame f: record frame f (the record has one frame per film
// frame; its clock is time-lapse ×15 for 50 s, then real time), seen by the
// camera at f, captioned with that frame's clock.
//
// THE CAMERA starts wide (the whole arena: nest, sugar, the empty card),
// moves to take in the route between nest and sugar while the trail forms
// (it forms at film 47.7 s in this record), and ends close on the formed
// trail at the sugar end in real time. Each move eases in and out
// (smoothstep); its progress only ever grows, so the camera never goes back.
// The shots are MODEL: framing, not measurement; the close shot's centre is
// found from the record (where the last frame's pheromone lies), not typed.
//
// MOTION IN THE TIME-LAPSE, decided: each time-lapse frame is 0.5 s of ant
// time, 15 mm of walking, six strides and three antenna sweeps. The frame
// shows the ants as they are at that instant (a 0° shutter), so legs and
// antennae change phase from frame to frame without moving smoothly — the
// strobing a real time-lapse camera shows too. Motion blur would mean
// rendering each frame many times over (the legs cycle 6× within one
// frame); it is not faked, and the time-lapse is labelled on screen.
//
// CAPTIONS sit in two plain bands across the top and bottom of the frame,
// painted over the picture, so no caption is ever drawn over an ant.

import CoreGraphics
import CoreText
import Foundation
import ImageIO
import UniformTypeIdentifiers
import simd

// MARK: - what to break, for the mutation check

/// SIM_MUTANT, shared with the simulation's and the renderer's mutants.
enum FilmMutant: String {
    case none
    /// The film draws no pheromone (the record's field replaced by nothing).
    case noPheromone
    /// Stance feet carried along with the body (lib/ant/v1's `.sliding`).
    case slidingFeet = "sliding_feet"
    /// The last fifth of the film shows the fifth before it backwards.
    case rewind
    /// One frame short.
    case wrongFrameCount

    static func fromEnvironment() -> FilmMutant {
        guard let s = ProcessInfo.processInfo.environment["SIM_MUTANT"] else { return .none }
        return FilmMutant(rawValue: s) ?? .none
    }
}

// MARK: - the plan: which record frame, which camera

struct FilmPlan {
    let film: FilmRecord
    let mutant: FilmMutant
    static let fps: Int = 30

    init(film: FilmRecord, mutant: FilmMutant = FilmMutant.fromEnvironment()) {
        self.film = film
        self.mutant = mutant
        close = FilmPlan.closeShot(film)
    }

    /// Frames in the film: one per record frame.
    var frameCount: Int { mutant == .wrongFrameCount ? film.frameCount - 1 : film.frameCount }
    var seconds: Double { Double(frameCount) / Double(FilmPlan.fps) }

    /// The record frame shown at film frame `f`.
    func recordFrame(_ f: Int) -> Int {
        if mutant == .rewind {
            let cut: Int = film.frameCount * 4 / 5
            if f > cut { return max(2 * cut - f, 0) }
        }
        return f
    }

    // The shots. MODEL framing.
    /// The whole arena (240 × 135 mm), from 60° up.
    static let wide: WorldCamera = WorldCamera.wide
    /// The route: nest (x 40) to sugar (x 192), 195 mm across, 52° up.
    static let route = WorldCamera(target: SIMD3<Float>(116, 0, 67), azimuth: WorldCamera.lookingBack,
                                   elevation: WorldCamera.degrees(52), width: 195)
    /// The trail's sugar end, 90 mm across (x 110–200, the whole pile in
    /// view), 38° up; its centre's z from the record.
    let close: WorldCamera

    /// Where the formed trail runs at x = 155 mm: the pheromone-weighted mean z
    /// over x 145–165 mm, z 45–90 mm, at the record's last frame.
    static func closeShot(_ film: FilmRecord) -> WorldCamera {
        let field: [Float] = film.field(atFrame: film.frameCount - 1)
        let w: Int = film.gridWidth
        let h: Float = SimConst.cell
        var sum: Float = 0
        var mass: Float = 0
        for j in Int(45 / h)..<Int(90 / h) {
            for i in Int(145 / h)..<Int(165 / h) {
                let c: Float = field[j * w + i]
                let z: Float = (Float(j) + 0.5) * h
                sum += c * z
                mass += c
            }
        }
        let z: Float = mass > 0 ? sum / mass : 67
        return WorldCamera(target: SIMD3<Float>(155, 0, z), azimuth: WorldCamera.lookingBack,
                           elevation: WorldCamera.degrees(38), width: 90)
    }

    /// The moves, film seconds: hold wide, ease to the route, hold while the
    /// trail forms, ease in close as real time begins, hold.
    static let moves: [(from: Double, to: Double)] = [(8, 18), (48, 54)]

    /// How far along the camera's path frame `f` is: 0 wide, 1 route, 2
    /// close. Never decreases.
    func progress(_ f: Int) -> Float {
        let t: Double = Double(f) / Double(FilmPlan.fps)
        var p: Float = 0
        for m in FilmPlan.moves {
            let span: Double = m.to - m.from
            let along: Double = (t - m.from) / span
            let u: Double = min(max(along, 0), 1)
            let twice: Double = 2 * u
            let rise: Double = 3 - twice
            let square: Double = u * u
            let eased: Double = square * rise
            p += Float(eased)
        }
        return p
    }

    static func blend(_ a: WorldCamera, _ b: WorldCamera, _ u: Float) -> WorldCamera {
        let step: SIMD3<Float> = b.target - a.target
        let target: SIMD3<Float> = a.target + step * u
        let elevation: Float = a.elevation + (b.elevation - a.elevation) * u
        // Width eases geometrically, so the zoom looks even.
        let ratio: Float = b.width / a.width
        let width: Float = a.width * pow(ratio, u)
        return WorldCamera(target: target, azimuth: a.azimuth, elevation: elevation, width: width)
    }

    func camera(_ f: Int) -> WorldCamera {
        let p: Float = progress(f)
        if p <= 1 { return FilmPlan.blend(FilmPlan.wide, FilmPlan.route, p) }
        return FilmPlan.blend(FilmPlan.route, close, p - 1)
    }

    func clockLabel(_ f: Int) -> String { film.sim.clock.label(atFrame: recordFrame(f)) }
}

// MARK: - captions

/// Caption bands, as a fraction of the frame height, top and bottom. MODEL.
let filmBandFraction: Float = 0.06
let filmTitle: String = "A minute of ant world · Lasius niger workers, 25 °C"
let filmTrailNote: String = "trail pheromone: drawn visible; invisible in life"
/// The scale bar's length, mm.
let filmScaleBar: Float = 10

func filmBandHeight(_ height: Int) -> Int {
    let h: Float = Float(height) * filmBandFraction
    return Int(h.rounded())
}

/// Paint the two bands and their captions onto a finished RGBA frame.
func filmCaption(_ pixels: UnsafeMutableRawPointer, width: Int, height: Int, clock: String,
                 camera: WorldCamera) {
    guard let space = CGColorSpace(name: CGColorSpace.sRGB),
          let ctx = CGContext(data: pixels, width: width, height: height, bitsPerComponent: 8,
                              bytesPerRow: width * 4, space: space,
                              bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return }
    let band: CGFloat = CGFloat(filmBandHeight(height))
    let w: CGFloat = CGFloat(width)
    let h: CGFloat = CGFloat(height)
    // The paper of the bands. CG's origin is the bottom left.
    ctx.setFillColor(CGColor(srgbRed: 0.93, green: 0.925, blue: 0.915, alpha: 1))
    ctx.fill(CGRect(x: 0, y: 0, width: w, height: band))
    ctx.fill(CGRect(x: 0, y: h - band, width: w, height: band))
    ctx.setFillColor(CGColor(srgbRed: 0.80, green: 0.79, blue: 0.78, alpha: 1))
    let rule: CGFloat = max(1, h / 1080)
    ctx.fill(CGRect(x: 0, y: band, width: w, height: rule))
    ctx.fill(CGRect(x: 0, y: h - band - rule, width: w, height: rule))

    let size: CGFloat = band * 0.42
    let margin: CGFloat = band * 0.55
    let ink = CGColor(srgbRed: 0.16, green: 0.16, blue: 0.17, alpha: 1)
    func draw(_ text: String, x: CGFloat, midY: CGFloat, alignRight: Bool = false) -> CGFloat {
        let font: CTFont = CTFontCreateWithName("Helvetica Neue" as CFString, size, nil)
        let attrs: [NSAttributedString.Key: Any] = [
            NSAttributedString.Key(kCTFontAttributeName as String): font,
            NSAttributedString.Key(kCTForegroundColorAttributeName as String): ink,
        ]
        let line: CTLine = CTLineCreateWithAttributedString(NSAttributedString(string: text, attributes: attrs))
        let bounds: CGRect = CTLineGetBoundsWithOptions(line, [])
        let x0: CGFloat = alignRight ? x - bounds.width : x
        let y0: CGFloat = midY - (bounds.height / 2 + bounds.minY)
        ctx.textPosition = CGPoint(x: x0, y: y0)
        CTLineDraw(line, ctx)
        return bounds.width
    }
    let topMid: CGFloat = h - band / 2
    let bottomMid: CGFloat = band / 2
    _ = draw(filmTitle, x: margin, midY: topMid)
    _ = draw(clock, x: w - margin, midY: topMid, alignRight: true)

    // The trail note, after a swatch of the drawn pheromone's colour.
    let sw: CGFloat = size * 1.6
    let tint: SIMD3<Float> = worldTrailColour
    // Linear to sRGB for the swatch, at the drawing's densest tint on the card.
    func encode(_ c: Float) -> CGFloat {
        let base: Float = worldCardAlbedo * (1 - worldTrailMax)
        let scaled: Float = c * worldCardAlbedo
        let v: Float = base + scaled * worldTrailMax
        let e: Float = v <= 0.0031308 ? 12.92 * v : 1.055 * pow(v, 1 / 2.4) - 0.055
        return CGFloat(min(max(e, 0), 1))
    }
    ctx.setFillColor(CGColor(srgbRed: encode(tint.x), green: encode(tint.y), blue: encode(tint.z), alpha: 1))
    ctx.fill(CGRect(x: margin, y: bottomMid - size * 0.4, width: sw, height: size * 0.8))
    _ = draw(filmTrailNote, x: margin + sw + size * 0.5, midY: bottomMid)

    // The scale bar: true along the frame's width (an orthographic camera
    // looking square to x), on the ground.
    let pxPerMM: CGFloat = w / CGFloat(camera.width)
    let barLength: CGFloat = pxPerMM * CGFloat(filmScaleBar)
    let labelWidth: CGFloat = draw("\(Int(filmScaleBar)) mm", x: w - margin, midY: bottomMid, alignRight: true)
    let barRight: CGFloat = w - margin - labelWidth - size * 0.6
    ctx.setFillColor(ink)
    ctx.fill(CGRect(x: barRight - barLength, y: bottomMid - rule * 1.5, width: barLength, height: rule * 3))
    ctx.fill(CGRect(x: barRight - barLength, y: bottomMid - size * 0.3, width: rule * 2, height: size * 0.6))
    ctx.fill(CGRect(x: barRight - rule * 2, y: bottomMid - size * 0.3, width: rule * 2, height: size * 0.6))
}

// MARK: - drawing a frame

/// Draws film frames: the world renderer, the ants posed from the record,
/// the field run on from the record, the captions.
final class FilmFrames {
    let plan: FilmPlan
    let renderer: WorldRenderer
    let samples: Int
    private let cast: WorldCast
    private let cursor: FilmFieldCursor

    init(plan: FilmPlan, width: Int, height: Int, samples: Int) throws {
        self.plan = plan
        self.samples = samples
        let set = WorldSet(config: plan.film.sim.config, grains: plan.film.sim.grains)
        renderer = try WorldRenderer(width: width, height: height, set: set)
        cast = WorldCast(record: plan.film.sim)
        cursor = FilmFieldCursor(film: plan.film)
    }

    var width: Int { renderer.image.width }
    var height: Int { renderer.image.height }

    /// The posed ants at film frame `f`, with their ids.
    func ants(_ f: Int) -> [(id: Int, posed: AntV1.Posed)] {
        let antMutant: AntV1.Mutant = plan.mutant == .slidingFeet ? .sliding : .none
        return cast.ants(atFrame: plan.recordFrame(f), mutant: antMutant)
    }

    /// The scene at film frame `f`, before captions.
    func scene(_ f: Int) -> WorldFrameScene {
        let r: Int = plan.recordFrame(f)
        let field: [Float] = plan.mutant == .noPheromone ? [] : cursor.field(atFrame: r)
        return WorldFrameScene(camera: plan.camera(f), ants: ants(f).map { $0.posed }, field: field)
    }

    /// Render film frame `f` into the renderer's image, captions and all.
    /// Returns GPU seconds.
    @discardableResult
    func draw(_ f: Int) throws -> Double {
        let gpu: Double = try renderer.render(scene(f), samples: samples)
        filmCaption(renderer.image.pixels.contents(), width: width, height: height, clock: plan.clockLabel(f),
                    camera: plan.camera(f))
        return gpu
    }
}

// MARK: - the frame store: resumable

/// Finished frames on disk, one PNG each (lossless: read back, the pixels
/// are exactly the rendered ones). A frame is written to a temporary name
/// and renamed when complete, so an interrupted run never leaves a partial
/// frame under a real name; a restart renders only the frames not there.
struct FilmStore {
    let directory: URL

    func url(_ f: Int) -> URL { directory.appendingPathComponent(String(format: "%04d.png", f)) }

    func has(_ f: Int) -> Bool { FileManager.default.fileExists(atPath: url(f).path) }

    func write(_ image: WorldImage, frame f: Int) throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let tmp: URL = directory.appendingPathComponent(String(format: ".%04d.partial.png", f))
        try worldSavePNG(image, to: tmp)
        if has(f) { try FileManager.default.removeItem(at: url(f)) }
        try FileManager.default.moveItem(at: tmp, to: url(f))
    }

    /// A stored frame's pixels, RGBA8, top row first.
    func read(_ f: Int) throws -> (width: Int, height: Int, rgba: [UInt8]) {
        try filmReadPNG(url(f))
    }
}

func filmReadPNG(_ url: URL) throws -> (width: Int, height: Int, rgba: [UInt8]) {
    guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
          let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else { throw WorldRenderError.png("cannot read \(url.path)") }
    let w: Int = img.width
    let h: Int = img.height
    var out = [UInt8](repeating: 0, count: w * h * 4)
    guard let space = CGColorSpace(name: CGColorSpace.sRGB) else { throw WorldRenderError.png("no sRGB") }
    let ok: Bool = out.withUnsafeMutableBytes { raw -> Bool in
        guard let ctx = CGContext(data: raw.baseAddress, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                                  space: space, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { return false }
        ctx.draw(img, in: CGRect(x: 0, y: 0, width: w, height: h))
        return true
    }
    guard ok else { throw WorldRenderError.png("cannot decode \(url.path)") }
    // Alpha is not stored; report it as the renderer writes it.
    for k in stride(from: 3, to: out.count, by: 4) { out[k] = 255 }
    return (w, h, out)
}

/// Render every frame in `frames` not already in the store. Returns
/// (rendered, skipped, wall seconds).
func filmRender(_ frames: FilmFrames, store: FilmStore, frames list: [Int],
                progress: ((Int, Double) -> Void)? = nil) throws -> (rendered: Int, skipped: Int, seconds: Double) {
    let t0: Date = Date()
    var rendered: Int = 0
    var skipped: Int = 0
    for f in list {
        if store.has(f) { skipped += 1; continue }
        let gpu: Double = try frames.draw(f)
        try store.write(frames.renderer.image, frame: f)
        rendered += 1
        progress?(f, gpu)
    }
    return (rendered, skipped, Date().timeIntervalSince(t0))
}

// MARK: - the MP4

// Encoded by a separate small tool, Tools/mp4 (make encode), which compiles
// lib/video/v1: AVFoundation's operators slow the type-checking of every
// == in any module that imports it, so the renderer and the tests are
// built without it.
