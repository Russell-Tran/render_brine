// Tests for step 79's film (part 5, prediction P5), drawn from the recorded
// world. Called from RenderTests.swift once the record has loaded.
//
// Mutants (SIM_MUTANT) this file must catch:
//   noPheromone     — the film draws no pheromone → the visibility rule fails
//   sliding_feet    — stance feet ride along with the body → feet planted fails
//   rewind          — the last fifth shows the fifth before it backwards →
//                     forward only fails
//   wrongFrameCount — one frame short → the frame count fails (and make
//                     test's MP4 checks)

import Foundation
import simd

func filmTests(_ film: FilmRecord) {
    section("The film (P5): the plan")

    let plan = FilmPlan(film: film)

    test("60 s at 30 fps: 1800 film frames, one per record frame") {
        expectEqual(plan.frameCount, 1800)
        expectEqual(plan.frameCount, film.frameCount)
        expect(abs(plan.seconds - 60) < 1e-9, "\(plan.seconds) s")
    }

    test("forward only: record frame, ant time and the camera's progress never go back") {
        var back: Int = 0
        for f in 1..<plan.frameCount {
            let a: Int = plan.recordFrame(f - 1)
            let b: Int = plan.recordFrame(f)
            if b <= a { back += 1 }
            if film.sim.ticks[b] <= film.sim.ticks[a] { back += 1 }
            if plan.progress(f) < plan.progress(f - 1) { back += 1 }
        }
        expect(back == 0, "\(back) steps back")
        // The camera moves only within the planned moves, and ends where it
        // was going.
        expectEqual(plan.progress(0), 0)
        expectEqual(plan.progress(plan.frameCount - 1), 2)
    }

    test("the camera starts wide on the whole arena and ends close on the formed trail") {
        let first: WorldCamera = plan.camera(0)
        let last: WorldCamera = plan.camera(plan.frameCount - 1)
        expect(first.width >= film.sim.config.width, "the first shot is \(first.width) mm across")
        expect(last.width < 100, "the last shot is \(last.width) mm across")
        // The trail's centre and the whole sugar pile are in the last shot.
        let c: SimWorldConfig = film.sim.config
        let sugar: SIMD2<Float> = last.project(SIMD3<Float>(c.sugar.x + c.sugarRadius, 0, c.sugar.y), width: 1920, height: 1080)
        let centre: SIMD2<Float> = last.project(last.target, width: 1920, height: 1080)
        expect(sugar.x < 1920 && sugar.x > 0, "the pile's far edge is off screen at x = \(sugar.x)")
        // The formed trail is what the last shot shows: of the pheromone in
        // the stretch it frames (x 110–195 mm), most is in view, between the
        // caption bands, and the densest of it is well above detection.
        let field: [Float] = film.field(atFrame: film.frameCount - 1)
        let band: Float = Float(filmBandHeight(1080))
        var inView: Float = 0
        var stretch: Float = 0
        var peak: Float = 0
        for j in 0..<film.gridHeight {
            for i in 0..<film.gridWidth {
                let x: Float = (Float(i) + 0.5) * SimConst.cell
                let z: Float = (Float(j) + 0.5) * SimConst.cell
                if x < 110 || x > 195 { continue }
                let v: Float = field[j * film.gridWidth + i]
                stretch += v
                let px: SIMD2<Float> = last.project(SIMD3<Float>(x, 0, z), width: 1920, height: 1080)
                let inside: Bool = px.x >= 0 && px.x < 1920 && px.y >= band && px.y < 1080 - band
                if inside {
                    inView += v
                    peak = max(peak, v)
                }
            }
        }
        let share: Float = stretch > 0 ? inView / stretch : 0
        expect(share > 0.8, "only \(share) of the stretch's pheromone is in the last shot")
        expect(peak > 5 * SimConst.detectThreshold, "the densest pheromone in view is \(peak) marks/mm²")
        print("        last shot centred on (\(last.target.x), \(last.target.z)), pixel \(centre): \(String(format: "%.0f", share * 100))% of the pheromone at x 110–195 mm in view, peak \(peak) marks/mm²")
    }

    test("feet planted: from frame to frame a stance foot stays where it was put") {
        let frames: FilmFrames
        do { frames = try FilmFrames(plan: plan, width: 16, height: 16, samples: 1) } catch { expect(false, "\(error)"); return }
        var checked: Int = 0
        var moved: Int = 0
        var worst: Float = 0
        var previous: [Int: AntV1.Posed] = [:]
        // Real time (frames 1500–1799) and a stretch of the time-lapse.
        let list: [Int] = Array(1500..<1800) + Array(600..<660)
        var last: Int = -2
        for f in list {
            let now = frames.ants(f)
            var current: [Int: AntV1.Posed] = [:]
            for (id, p) in now { current[id] = p }
            if f == last + 1 {
                for (id, p) in current {
                    guard let q = previous[id] else { continue }
                    let a: SimAntFrame = film.sim.ant(id, atFrame: plan.recordFrame(f))
                    let b: SimAntFrame = film.sim.ant(id, atFrame: plan.recordFrame(f - 1))
                    if a.outing != b.outing { continue }
                    let st: AntV1.State = worldAntState(id: id, dab: 0)
                    for j in 0..<6 where p.stance[j] && q.stance[j] {
                        // The same stance: the same stride of this leg.
                        let off: Float = AntV1.legOffset(j, gaitOffset: st.gaitOffset, mutant: .none)
                        let kNow: Float = (p.distance / AntV1.strideLength + off).rounded(.down)
                        let kBefore: Float = (q.distance / AntV1.strideLength + off).rounded(.down)
                        if kNow != kBefore { continue }
                        checked += 1
                        let d: Float = simd_distance(p.feet[j], q.feet[j])
                        worst = max(worst, d)
                        if d > 1e-4 { moved += 1 }
                    }
                }
            }
            previous = current
            last = f
        }
        expect(checked > 1000, "only \(checked) stance feet compared")
        expect(moved == 0, "\(moved) of \(checked) stance feet moved (up to \(worst) mm)")
        print("        \(checked) stance feet followed from one frame to the next; the most any moved: \(worst) mm")
    }

    section("The film (P5): on screen")

    let frames: FilmFrames
    do { frames = try FilmFrames(plan: plan, width: 480, height: 270, samples: 1) } catch {
        test("the film's frames can be drawn") { throw error }
        return
    }

    test("the clock and the pheromone note are on screen, in bands clear of the scene") {
        let f: Int = 1431
        try frames.draw(f)
        expectEqual(plan.clockLabel(f), film.sim.clock.label(atFrame: f))
        expect(plan.clockLabel(f).hasPrefix("time-lapse ×15"), plan.clockLabel(f))
        expect(plan.clockLabel(plan.frameCount - 1).hasPrefix("real time"), plan.clockLabel(plan.frameCount - 1))
        let band: Int = filmBandHeight(270)
        let img: WorldImage = frames.renderer.image
        var ink: [Int] = [0, 0]
        var colour: Int = 0
        let swatchEnd: Int = 60
        for (k, rows) in [0..<band, (270 - band)..<270].enumerated() {
            for y in rows {
                for x in 0..<480 {
                    let p: SIMD4<UInt8> = img.rgba(x, y)
                    if p.x < 110 { ink[k] += 1 }
                    let top: UInt8 = max(p.x, p.y, p.z)
                    let bottom: UInt8 = min(p.x, p.y, p.z)
                    let spread: Int = Int(top) - Int(bottom)
                    if spread > 10 && !(k == 1 && x < swatchEnd) { colour += 1 }
                }
            }
        }
        expect(ink[0] > 100 && ink[1] > 100, "caption ink top \(ink[0]), bottom \(ink[1])")
        expect(colour == 0, "\(colour) coloured pixels in the bands: the scene shows through")
    }

    test("the visibility rule: the trail's forming shows on screen (18 s → 48 s, same shot)") {
        expect(plan.camera(540) == plan.camera(1440), "the camera moved between the two frames")
        try frames.draw(540)
        let a: [UInt8] = frames.renderer.image.bytes()
        let auxA: [SIMD4<Float>] = filmAux(frames.renderer.image)
        try frames.draw(1440)
        let b: [UInt8] = frames.renderer.image.bytes()
        let auxB: [SIMD4<Float>] = filmAux(frames.renderer.image)
        let n: Int = filmTrailPixels(a, auxA, b, auxB)
        let share: Double = Double(n) / Double(480 * 270)
        expect(share > 0.01, "only \(n) pixels turned trail-blue")
        print("        \(n) card pixels turned trail-blue (\(String(format: "%.1f", share * 100))% of the frame)")
    }

    section("The film (P5): resumable, and the same whenever drawn")

    test("frames drawn after an interruption match an uninterrupted run bit for bit") {
        let root = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("step079_resume")
        try? FileManager.default.removeItem(at: root)
        let list: [Int] = [0, 449, 450, 451, 1431, 1432, 1499, 1500, 1501, 1799]
        let whole = FilmStore(directory: root.appendingPathComponent("whole"))
        let broken = FilmStore(directory: root.appendingPathComponent("broken"))
        let backwards = FilmStore(directory: root.appendingPathComponent("backwards"))
        func fresh() throws -> FilmFrames { try FilmFrames(plan: plan, width: 320, height: 180, samples: 1) }
        let one = try filmRender(try fresh(), store: whole, frames: list)
        expectEqual(one.rendered, list.count)
        // Interrupted after four frames, with a half-written fifth left behind…
        _ = try filmRender(try fresh(), store: broken, frames: Array(list.prefix(4)))
        try Data([0, 1, 2]).write(to: broken.directory.appendingPathComponent(String(format: ".%04d.partial.png", list[4])))
        // …then resumed by a new renderer.
        let resumed = try filmRender(try fresh(), store: broken, frames: list)
        expectEqual(resumed.skipped, 4)
        expectEqual(resumed.rendered, list.count - 4)
        // And drawn in the opposite order.
        _ = try filmRender(try fresh(), store: backwards, frames: list.reversed())
        for f in list {
            let a: [UInt8] = try whole.read(f).rgba
            expect(try broken.read(f).rgba == a, "frame \(f): resumed differs")
            expect(try backwards.read(f).rgba == a, "frame \(f): drawn out of order differs")
        }
        try? FileManager.default.removeItem(at: root)
    }

    test("the stored frame reads back exactly as drawn (PNG is lossless here)") {
        let root = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("step079_png")
        let store = FilmStore(directory: root)
        try frames.draw(1600)
        let drawn: [UInt8] = frames.renderer.image.bytes()
        try store.write(frames.renderer.image, frame: 1600)
        expect(try store.read(1600).rgba == drawn, "PNG round trip changed the pixels")
        try? FileManager.default.removeItem(at: root)
    }

    test("reference frames: drawn here, they match records/refs (tolerance for another GPU)") {
        let drawn: [(Int, [UInt8])] = try filmDrawReferences(plan)
        let here = URL(fileURLWithPath: FileManager.default.currentDirectoryPath).appendingPathComponent(filmReferenceDirectory)
        for (f, rgba) in drawn {
            let ref = try filmReadPNG(here.appendingPathComponent(String(format: "%04d.png", f)))
            let d: FilmChannelDifference = filmCompare(rgba, ref.rgba)
            let means: [String] = d.meanAbs.map { String(format: "%.4f", $0) }
            print("        frame \(f): max |diff| R G B \(d.maxAbs) /255, mean \(means) /255")
            expect(d.within, "frame \(f) outside the tolerance (mean < \(filmReferenceMeanTolerance), max < \(filmReferenceMaxTolerance))")
        }
    }

    section("The film (P5): crowding — the honest bound")

    test("interpenetration in 3D stays within the bound the simulation allows") {
        let probe: FilmFrames
        do { probe = try FilmFrames(plan: plan, width: 16, height: 16, samples: 1) } catch { expect(false, "\(error)"); return }
        var framesInside: Int = 0
        var sampled: Int = 0
        var deepest: Float = 0
        for f in stride(from: 0, to: plan.frameCount, by: 60) {
            let o: FilmOverlap = try filmOverlap(probe.renderer, ants: probe.ants(f))
            sampled += 1
            if o.pairsInside > 0 { framesInside += 1 }
            deepest = max(deepest, o.deepest)
        }
        let share: Double = Double(framesInside) / Double(sampled)
        print("        \(framesInside) of \(sampled) sampled frames have ants inside one another; deepest \(String(format: "%.3f", deepest)) mm")
        expect(share <= filmOverlapFrameShareBound, "\(share) of frames")
        expect(deepest <= filmOverlapDepthBound, "\(deepest) mm deep")
    }
}
