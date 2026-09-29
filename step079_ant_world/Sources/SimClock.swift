// Step 79: the film's clock — which simulated second each film frame shows.
//
// The film runs at 30 frames a second. The simulation ticks every 1/60 s of
// the ants' time, so a frame that advances 2 ticks is REAL TIME and one that
// advances 2k ticks is a k× time-lapse. The clock is a list of segments, each
// a number of frames at a whole number of ticks per frame, so every frame
// lands exactly on a tick and the mapping is exact. It only ever moves
// forward (the project's rule: the film plays once through and never
// rewinds); the `rewind` mutant breaks exactly that.

import Foundation

enum SimMutant: String {
    case none
    /// No pheromone is ever deposited: the rules otherwise unchanged.
    case noPheromone
    /// Time runs backward in part of the film clock.
    case rewind
    /// Paths may curve at 3 mm radius, tighter than the shared ant can walk.
    case tightTurn

    static func fromEnvironment() -> SimMutant {
        guard let s = ProcessInfo.processInfo.environment["SIM_MUTANT"] else { return .none }
        return SimMutant(rawValue: s) ?? .none
    }
}

struct SimClockSegment: Equatable {
    /// Film frames in this segment.
    var frames: Int
    /// Simulation ticks per film frame: 2 = real time at 30 fps / 60 Hz.
    var ticksPerFrame: Int
}

struct SimClock: Equatable {
    static let filmFPS: Int = 30
    var segments: [SimClockSegment]
    var mutant: SimMutant = .none

    var frameCount: Int { segments.reduce(0) { $0 + $1.frames } }
    var filmSeconds: Double { Double(frameCount) / Double(SimClock.filmFPS) }

    /// THE FILM'S CLOCK. Trail formation takes minutes of ant time (this
    /// simulation: a median of about 11 min over eight seeds, see the report),
    /// so a minute of film can't be real time throughout. It is:
    ///   * 0–50 s of film: TIME-LAPSE ×15 (30 ticks a frame) — 12.5 minutes of
    ///     ant time: the search, the first find, the trail forming;
    ///   * 50–60 s: REAL TIME (2 ticks a frame) — 10 s on the formed trail,
    ///     where the walking is shown as it is.
    /// The clock and its rate go on screen (`label`).
    static let film = SimClock(segments: [
        SimClockSegment(frames: 1500, ticksPerFrame: 30),
        SimClockSegment(frames: 300, ticksPerFrame: 2),
    ])

    /// Ticks per film frame at real time.
    static var realTimeTicksPerFrame: Int {
        Int((1.0 / SimConst.tick).rounded()) / filmFPS
    }

    /// The simulation tick film frame `f` shows (frame 0 shows tick 0).
    func tick(atFrame f: Int) -> Int {
        precondition(f >= 0 && f < frameCount, "frame \(f) outside the film")
        var ticks: Int = 0
        var left: Int = f
        for s in segments {
            let n: Int = min(left, s.frames)
            ticks += n * s.ticksPerFrame
            left -= n
            if left == 0 { break }
        }
        if mutant == .rewind {
            // The last fifth of the film plays the fifth before it backwards.
            let cut: Int = frameCount * 4 / 5
            if f > cut {
                let back: Int = 2 * cut - f
                return tick(atFrameForward: max(back, 0))
            }
        }
        return ticks
    }

    private func tick(atFrameForward f: Int) -> Int {
        var copy: SimClock = self
        copy.mutant = .none
        return copy.tick(atFrame: f)
    }

    /// Simulated seconds shown at film frame `f`.
    func simSeconds(atFrame f: Int) -> Double {
        let n: Double = Double(tick(atFrame: f))
        return n * SimConst.tick
    }

    /// Simulated seconds at any film second (piecewise linear between frames;
    /// exact at frames). For putting the clock on screen.
    func simSeconds(atFilmSecond s: Double) -> Double {
        let x: Double = s * Double(SimClock.filmFPS)
        let f0: Int = min(max(Int(x.rounded(.down)), 0), frameCount - 1)
        let f1: Int = min(f0 + 1, frameCount - 1)
        let u: Double = min(max(x - Double(f0), 0), 1)
        let a: Double = simSeconds(atFrame: f0)
        let b: Double = simSeconds(atFrame: f1)
        return a + (b - a) * u
    }

    /// How much faster than life the film runs at frame `f`: 1 = real time.
    func speedup(atFrame f: Int) -> Double {
        var left: Int = f
        for s in segments {
            if left < s.frames {
                return Double(s.ticksPerFrame) / Double(SimClock.realTimeTicksPerFrame)
            }
            left -= s.frames
        }
        let last: SimClockSegment = segments[segments.count - 1]
        return Double(last.ticksPerFrame) / Double(SimClock.realTimeTicksPerFrame)
    }

    /// A label for the screen, e.g. "time-lapse ×4 · 1:23 ant time".
    func label(atFrame f: Int) -> String {
        let k: Double = speedup(atFrame: f)
        let t: Double = simSeconds(atFrame: f)
        let m: Int = Int(t) / 60
        let s: Int = Int(t) % 60
        let clock: String = String(format: "%d:%02d", m, s)
        if k == 1 { return "real time · \(clock) ant time" }
        return String(format: "time-lapse ×%g · %@ ant time", k, clock)
    }
}
