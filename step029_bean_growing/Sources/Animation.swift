// Step 29: step 25's still, with one thing moving — the pollen tube grows from
// its grain on the stigma, down inside the coiled style, to the ovule. When it
// arrives the picture holds, then dissolves back to the start. Never a rewind:
// the tube only ever gets longer, and the way back to the start is a
// crossfade of whole pictures.
//
// Sources for the clock:
//
//   Johnson & Bernard (1963), "Soybean genetics and breeding", in The Soybean
//     (Norman, ed.), as quoted in US soybean variety patents (e.g. US 8,319,039,
//     "Soybean variety A1024340"): the
//     anthers dehisce, pollen falls on the stigma, and "within 10 h the pollen
//     tubes reach the ovary and fertilization is completed". Soybean is the
//     nearest legume found with a measured pollination-to-fertilization time.
//   Weinstein (1926), "Cytological studies on Phaseolus vulgaris", Am. J. Bot.
//     13:248–263, as cited by Chacón-Sánchez et al., Front. Ecol. Evol. 9:618709
//     (2021): in the common bean "fertilization normally occurs 8 h after
//     anthesis" — and the anthers dehisce the evening before anthesis
//     (McGregor, USDA Handbook 496, after Jones & Rosa 1928). So in the bean the
//     tube's whole journey fits inside that window of a night and a morning.
//     No direct measurement of a bean tube's growth rate in the style turned up.
//   Williams, Int. J. Plant Sci. 173:649–661 (2012), "Pollen tube growth rates
//     and the diversification of flowering plant reproductive cycles":
//     angiosperm pollen tubes grow from about 10 µm/h to over 20 mm/h.

import Foundation

/// MODEL, from the two legume timings above: the tube's journey from grain to
/// ovule takes about 10 hours. Growth at a steady rate over this tube's length
/// gives the rate printed on the frame: about 2.1 mm/h for this 20.8 mm tube,
/// inside Williams's angiosperm range.
let realGrowthHours: Float = 10.0

// MARK: - the loop

/// Seconds in one loop, and where in it each phase ends, as fractions of the
/// loop. MODEL (choices of pacing, not biology): a short look at the grain
/// before the tube emerges, 9.6 s of growth, a hold once it arrives, then a
/// 1.3 s dissolve back to the start.
let loopSeconds: Float = 12.0
let growthStart: Float = 0.4 / 12.0
let growthEnd: Float = 10.0 / 12.0
let holdEnd: Float = 10.7 / 12.0

/// How the playback clock maps to the real one. The first render ran the
/// time-lapse 20× faster at the end than the start, so the inset's part was
/// seen and the rest of the journey flew by. Now the inset follows the tip,
/// and the advance is nearly steady: the time-lapse eases in from a third of
/// its speed over the first quarter of the growth (so the tube can be seen
/// leaving its grain), and is steady after that. The real time — and so the
/// tube's length, which grows at a steady real rate — is written on the frame
/// as a clock. MODEL.
let lapseSpeedRatio: Float = 3.0
let lapseSlowUntil: Float = 0.0
let lapseFastFrom: Float = 0.25

/// Fraction of the tube grown at a fraction `v` of the growth phase: the
/// integral of the playback speed, normalised to reach 1 exactly at v = 1.
/// Monotonic because the speed is never negative.
func grownFraction(_ v: Float) -> Float {
    let x: Float = min(max(v, 0), 1)
    let k: Float = lapseSpeedRatio - 1
    let a: Float = lapseSlowUntil
    let w: Float = lapseFastFrom - lapseSlowUntil
    // ∫ smoothstep(a, a + w, t) dt from 0 to x: 0 before a; w(y³ − y⁴/2)
    // with y = (x − a)/w across the rise; w/2 + (x − a − w) after it.
    func ramp(_ x: Float) -> Float {
        if x <= a { return 0 }
        if x >= a + w { return w / 2 + (x - a - w) }
        let y: Float = (x - a) / w
        return w * (y * y * y - y * y * y * y / 2)
    }
    if x >= 1 { return 1 }
    return (x + k * ramp(x)) / (1 + k * ramp(1))
}

/// What a frame shows: how much tube has grown, and how far the picture has
/// dissolved toward the loop's first frame (0 = not at all).
struct FrameState {
    var grown: Float
    var fade: Float
}

/// Frames in the loop and the GIF delay; `frameCount × delay` is the loop.
struct Timeline {
    var frameCount: Int
    var tubeLength: Float
    var mutant: Mutant

    /// Loop fraction of frame f.
    func u(_ f: Int) -> Float { Float(f) / Float(frameCount) }

    /// The frame that growth ends on: the first whose loop fraction reaches
    /// `growthEnd`. The tube arrives exactly there.
    var arrivalFrame: Int { Int((growthEnd * Float(frameCount)).rounded(.up)) }

    func state(_ f: Int) -> FrameState {
        let t: Float = u(f)
        if f >= arrivalFrame {
            if t < holdEnd { return FrameState(grown: tubeLength, fade: 0) }
            let w: Float = (t - holdEnd) / (1 - holdEnd)
            if mutant == .rewind {
                // The forbidden ending: the tube shrinks back down the style.
                return FrameState(grown: tubeLength * (1 - w), fade: 0)
            }
            return FrameState(grown: tubeLength, fade: w)
        }
        // Growth ends on the arrival frame, so scale growth to reach it there.
        let arrive: Float = u(arrivalFrame)
        let v: Float = (t - growthStart) / (arrive - growthStart)
        return FrameState(grown: tubeLength * grownFraction(v), fade: 0)
    }

    /// Real hours since the tube emerged, at a steady growth rate.
    func hours(_ s: FrameState) -> Float { s.grown / tubeLength * realGrowthHours }
}

/// Growth rate on the frame, mm per hour.
func growthRate(tubeLength: Float) -> Float { tubeLength / realGrowthHours }

/// A crossfade of two finished pictures.
func dissolve(_ a: [UInt8], _ b: [UInt8], _ w: Float) -> [UInt8] {
    var out: [UInt8] = a
    for i in 0..<a.count {
        let v: Float = Float(a[i]) * (1 - w) + Float(b[i]) * w
        out[i] = UInt8(min(max(v.rounded(), 0), 255))
    }
    return out
}

// MARK: - the frames of the loop

/// Renders the loop's frames, reusing what does not change: the pod, the
/// first frame (the dissolve's target) and the fully grown picture (every
/// frame of the hold and the dissolve).
final class Loop {
    let frames: BeanFrames
    let timeline: Timeline
    let delayCentiseconds: Int
    private var first: [UInt8]? = nil
    private var full: [UInt8]? = nil

    init(_ frames: BeanFrames, frameCount: Int, delayCentiseconds: Int, mutant: Mutant) {
        self.frames = frames
        self.delayCentiseconds = delayCentiseconds
        timeline = Timeline(frameCount: frameCount, tubeLength: frames.scene.layout.tubeLength, mutant: mutant)
    }

    /// Seconds of playback from the tube emerging to its arrival.
    var growthSeconds: Float {
        (timeline.u(timeline.arrivalFrame) - growthStart) * Float(timeline.frameCount * delayCentiseconds) / 100
    }

    /// Whether the widened tube and the following inset are drawn: always,
    /// except in the mutant that shows what step 29 first looked like.
    var visible: Bool { timeline.mutant != .invisible }

    /// What the inset looks at in a frame.
    func view(_ s: FrameState) -> InsetView {
        visible ? trackingView(frames.scene.model.tube, grown: s.grown) : stillInsetView
    }

    private func picture(_ s: FrameState) throws -> [UInt8] {
        let tube: Chain = frames.scene.model.tube
        let overlay: TubeOverlay? = visible ? TubeOverlay(points: grownTube(tube, s.grown)) : nil
        var bytes: [UInt8] = try frames.render(grown: s.grown, view: view(s), tube: overlay).bytes
        drawCaption(&bytes, width: frames.width, height: frames.height, hours: timeline.hours(s),
                    rate: growthRate(tubeLength: timeline.tubeLength), totalHours: realGrowthHours,
                    growthSeconds: growthSeconds, legend: visible)
        return bytes
    }

    func frame(_ f: Int) throws -> [UInt8] {
        let s: FrameState = timeline.state(f)
        if f == 0 {
            if first == nil { first = try picture(s) }
            return first!
        }
        if s.grown < timeline.tubeLength { return try picture(s) }
        if full == nil { full = try picture(FrameState(grown: timeline.tubeLength, fade: 0)) }
        if s.fade <= 0 { return full! }
        if first == nil { first = try picture(timeline.state(0)) }
        return dissolve(full!, first!, s.fade)
    }
}

// MARK: - the inset follows the tip

/// The field the inset shows once it is following the tip, mm across. MODEL:
/// wide enough that the tissue sliding past is not a blur at 20 frames a
/// second (the tip advances ~2 mm per second of playback), close enough that
/// the tube is drawn at its true 10 µm width (~3 px) and the style's cells of
/// tissue read as tissue.
let insetFollowField: Float = 1.4

/// The tube's grown length at which the inset starts to follow, and has fully
/// taken up the tip. Before that it is step 25's view of the stigma, in which
/// the tube leaves its grain; its first ~0.3 mm lies there. MODEL.
let insetFollowFrom: Float = 0.15
let insetFollowBy: Float = 0.9

/// Where on the tube the inset looks: a little behind the tip — the average
/// of the last 0.3 mm grown — so the tip leads, and the averaging smooths the
/// turns of the path into a steady pan.
func trackingView(_ tube: Chain, grown: Float) -> InsetView {
    var sum = SIMD3<Float>(0, 0, 0)
    let n: Int = 8
    for i in 0...n {
        let s: Float = max(grown - 0.3 * Float(i) / Float(n), 0)
        sum += grownTube(tube, s).last!
    }
    let behind: SIMD3<Float> = sum / Float(n + 1)
    let w: Float = smoothstep(insetFollowFrom, insetFollowBy, grown)
    let target: SIMD3<Float> = insetTarget + (behind - insetTarget) * w
    let field: Float = insetFieldDiameter + (insetFollowField - insetFieldDiameter) * w
    let reference: SIMD3<Float> = mainTarget + (target - mainTarget) * w
    // The cut comes down from above the flower onto the tube's own plane as
    // the inset takes up the tip, opening the style along the tube so the
    // tube shows at its true width instead of behind the filaments that run
    // in front of the style all round the coil. MODEL (a dissection).
    // The plane sits at the lowest point of the last 0.8 mm grown, so the
    // whole stretch of tube in view lies on the cut face, none buried.
    var lowZ: Float = 1e9
    for i in 0...16 {
        lowZ = min(lowZ, grownTube(tube, max(grown - 0.8 * Float(i) / 16, 0)).last!.z)
    }
    let cutZ: Float = w > 0 ? lowZ + (1 - w) * 2.0 : 1e9
    return InsetView(target: target, field: field, mainReference: reference, cutZ: cutZ)
}
