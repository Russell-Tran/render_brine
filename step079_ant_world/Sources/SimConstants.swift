// Step 79, "A minute of ant world": the SIMULATION's numbers. Nothing in the
// Sim*.swift files draws anything or knows about the ant's body model; the
// film reads what they produce (SimRecord.swift).
//
// The world: MILLIMETRES on the ground plane, the same axes as steps 44–57 —
// y up, the ground is y = 0, and an ant's position is a point (x, z) on the
// ground. Heading `yaw` follows step 44's `turn`: body +x points along
// (cos yaw, 0, sin yaw). Time: SIMULATED SECONDS, i.e. seconds of the ants'
// life; the film's clock maps film seconds onto these (SimClock.swift).
//
// Every number carries where it came from, or says MODEL and why. Everything
// lives inside `SimConst` so that nothing here collides with the shared ant
// module (lib/ant/v1) or the film's own names.

import Foundation

enum SimConst {

    // MARK: - walking

    // Walking speed: step 44's, re-checked for this step. Dussutour,
    // Deneubourg & Fourcassié (2005, *J Exp Biol* 208(15): 2903–2912, doi
    // 10.1242/jeb.01711, "Temporal organization of bi-directional traffic in
    // the ant Lasius niger (L.)"), opened again 2026-09-28: bridges 210 mm long
    // with "one bottleneck (15 mm) and one entrance (15 mm) at both ends" and "a
    // central part (60 mm)"; ants crossing without meeting others took
    // "2.96±0.61 s" (wide bridge; 2.93 ± 0.56 s narrow); "room temperature
    // (25±1°C)". DERIVED as step 44 did: 90 mm (entrance + central part +
    // entrance, between the bottlenecks) / 2.96 s = 30.4 mm/s. It rests on
    // that reading of the layout (60 mm alone would give 20 mm/s); labelled
    // with its 25 °C.
    static let dussutourDistance: Float = 90.0
    static let dussutourTime: Float = 2.96
    static let walkSpeed: Float = dussutourDistance / dussutourTime   // mm/s ≈ 30.4
    static let speedTemperature: Int = 25

    // MARK: - who lays trail, and how

    // "Foragers do not exhibit trail-laying behaviour until a food source is
    // discovered. Trail laying then occurs more or less equally both to and
    // from the nest" (Beckers, Deneubourg & Goss 1992, *Insectes Sociaux* 39:
    // 59–72, doi 10.1007/BF01240531, abstract, opened). "Only successful
    // foragers deposit pheromone in L. niger" (Grüter, Schürch, Czaczkes et al.
    // 2012, *PLoS ONE* 7: e44501, PMC3440389, citing Beckers et al. 1992).
    // So: an ant lays only after it has itself fed at the sugar, and then both
    // on the way home and on its next trips out.
    //
    // Some never do: "among L. niger foragers, 14% never participate in the
    // formation of the chemical pathway and never lay a trail over successive
    // trips" (Mailleux, Detrain & Deneubourg 2005, *J Insect Physiol* 51:
    // 297–304, PMID 15749112, abstract, opened).
    static let neverLayFraction: Double = 0.14

    // At a food source bigger than its crop a scout goes straight home laying
    // trail: "When scouts discovered food volumes exceeding the capacity of
    // their crop (3 or 6 µl), 90% immediately returned to the nest laying a
    // recruitment trail" (Mailleux, Deneubourg & Detrain 2000, *Anim Behav*
    // 59: 1061–1069, PMID 10860533, abstract, opened). A sugar pile is far more
    // than a crop, so every successful ant heads home (the 14% above still
    // don't lay).

    // How often a laying ant dabs: "After leaving a food source, an ant drops
    // on the average .5 pheromone per second. This experimental value gives
    // us an upper bound" (Boissard, Degond & Motsch 2011, arXiv:1108.3495,
    // §3.1, opened, citing Beckers et al. 1992 for it). SECONDARY: Beckers
    // 1992's full text (marks per 20 cm bridge crossing) could not be opened
    // here, so the figure is theirs as Boissard et al. read it. Used as the
    // rate of a Poisson process, so the dabs fall irregularly.
    static let dabsPerSecond: Double = 0.5
    // Where on the way home: returning L. niger "lay up to five times more on
    // the segment closest to the source than that closest to the nest"
    // (Beckers et al. 1992, abstract). The 0.5/s is Boissard et al.'s figure
    // "after leaving a food source", so it is the rate at the sugar, falling
    // linearly with the fraction of the way home still to go to a fifth of
    // it at the nest. (Linear: MODEL; the abstract gives only the end ratio.)
    static let nestEndLayingFraction: Float = 0.2

    // What a dab is: "Pheromone deposition is a very stereotyped behaviour in
    // L. niger ... It involves the ant pausing for ca. 0.2 seconds, backing
    // up, and firmly pressing the tip of their abdomen onto the substrate"
    // (Czaczkes et al. 2016, *PLoS ONE*, PMC4784821, opened; step 44 used the
    // same quote). The ant stands still for 0.2 s; the "backing up" is left
    // out (the film only ever moves forward).
    static let dabPause: Float = 0.2
    /// Pheromone laid per dab, in MARKS — the unit Beckers et al. counted in.
    /// The field is in marks per mm², so one dab adds exactly 1 to its integral.
    static let marksPerDab: Float = 1.0
    /// Where the gaster tip touches the ground, mm behind the ant's origin
    /// along its body axis. MODEL: the rear of step 44's body (it is 4.3 mm
    /// long with the head's front about 2.0 mm ahead of the origin).
    static let gasterTipBehind: Float = 2.3

    // MARK: - the pheromone field

    // Lifetime. Beckers, Deneubourg & Goss (1993, *J Insect Behav* 6: 751–759,
    // doi 10.1007/BF01201674, abstract, opened on Springer 2026-09-28): "the
    // mean lifetime of the trail pheromone was estimated to be 47 min". Forster
    // et al. 2014 (*Ethology*, PMC4204274, opened) restate it as "A single dot
    // of L. niger trail pheromone has been estimated to become undetectable in
    // 47 min at room temperature" — the SAME 47 min, but Beckers call it a mean
    // lifetime, and that is what the field uses: exponential decay with time
    // constant τ = 47 min, c(t) = c₀·e^(−t/τ). (Step 44's "~47 min" citation
    // via Forster 2014 is thus right about the number and its origin; Forster's
    // "undetectable" wording is a paraphrase.) Over a few minutes this decay
    // is small — ~6% in 3 min — which is itself the finding: evaporation plays
    // almost no part in a trail's first minutes.
    static let pheromoneLifetime: Double = 47.0 * 60.0     // s, mean lifetime τ

    /// Spreading of the sensed pheromone over the ground, mm²/s. MODEL: no
    /// measured value for L. niger; the field stands for what an antenna
    /// sweeping just above the ground picks up around a dot, and this is set
    /// so a dot's scent widens by about an ant's width over a few minutes
    /// (σ² = 2Dt: 1.5 mm after a minute, 3.5 mm after five). Tried: 0.3 made
    /// the trail a band too broad to steer by; 0.05 changed little.
    static let diffusion: Float = 0.02

    /// Grid spacing, mm. MODEL: one cell per tick of walking (see `tick`).
    static let cell: Float = 0.5

    // MARK: - sensing and choosing

    // Where the antenna tips are, for touching sugar and staying inside the
    // arena: MODEL, from step 44's walking pose — tips about 3.2 mm ahead of
    // the body origin, swept up to ±0.54 mm to either side (rest ±0.28 plus a
    // ±0.26 sweep).
    static let antennaAhead: Float = 3.2
    static let antennaSide: Float = 0.54

    // What the antennae SAMPLE as they sweep: an arc about the head of radius
    // `sweepRadius`, out to `sweepHalfAngle` either side, at 2·sweepSamples+1
    // points. MODEL: L. niger workers sweep their antennae over the ground as
    // they walk (step 44/55 draw it, ~7 sweeps/s, MODEL there too); an
    // antenna (scape ≈ 0.96 mm + funiculus, step 26) reaches about 1.5 mm from
    // the head, and swung out it meets a trail the ant is crossing at an
    // angle, not only one it is already on — two antenna tips held side by
    // side cannot tell which way a trail they cross at right angles runs.
    static let sweepRadius: Float = 1.5
    static let sweepHalfAngle: Float = 1.4
    static let sweepSamples: Int = 4
    /// Outbound, a sample in a direction back towards the nest counts this
    /// much. MODEL: outbound ants keep heading out (they know where home is).
    static let backwardWeight: Float = 0.25
    /// Leaving the nest: directions considered, and how far out along each
    /// the scent is taken, mm. MODEL.
    static let exitDirections: Int = 24
    static let exitSniffRadius: Float = 6.0

    // Choosing where to go: the choice function Beckers et al. (1993,
    // abstract) fitted to L. niger at a fork — "A mathematical function
    // describing the probability that a forager chooses one of two paths in
    // relation to the amount of trail pheromone on them closely fitted
    // experimental data" — of the form Deneubourg's group uses,
    //     P(path i) = (k + C_i)ⁿ / Σ_j (k + C_j)ⁿ,
    // with n = 2 as in Deneubourg et al. 1990 and Nicolis et al. 2003 (as
    // stated by Kawakami, Sakai & Nishimori, arXiv:1805.05598, opened: "the
    // Hill coefficient (n=2 in ref [5, 6])"). Beckers' own k and n are not in
    // their abstract, and the full text could not be opened: so n = 2 is the
    // family's value and k is MODEL (in marks/mm², the field's unit; a fresh
    // dot is 1 mark spread over a cell or four, up to 4 marks/mm²). Used
    // twice: as weights over the sweep's directions (the ant bears towards
    // their weighted mean, SimWorld.sweep), and over the directions out of
    // the nest (SimWorld.exitHeading). A first version applied it to the two
    // antenna tips only, turning left or right each tick; ants crossing a
    // trail could not tell which way it ran, and no trail formed.
    static let choiceN: Float = 2
    static let choiceK: Float = 0.02
    /// Scent above which an ant counts as ON a trail — the sum of the
    /// strongest sample on each side of its sweep, marks/mm². MODEL:
    /// detection thresholds for L. niger were not found.
    static let detectThreshold: Float = 0.02
    /// After losing the scent a follower keeps its heading this long before
    /// it goes back to searching, s. MODEL.
    static let lostPatience: Float = 0.5

    // MARK: - searching and homing

    /// Searching is a correlated random walk: the heading diffuses at this
    /// rate, rad²/s. MODEL: no L. niger search statistics used; this gives a
    /// persistence length (v / D_θ·2) of about 20 mm, a few body lengths.
    static let searchTurnDiffusion: Float = 3.0
    /// Where a trail gives out, a recruit searches tightly for a while. Le
    /// Breton & Fourcassié (2004, abstract, opened): the search trajectories
    /// of L. niger workers recruited to sugar show greater sinuosity ("food-
    /// correlated search") than those recruited to prey. How: MODEL — loops
    /// at the tightest turn the ant can walk (5 mm radius), switching
    /// direction now and then (rate per second), for 8 s.
    static let localSearchSeconds: Float = 8.0
    static let loopSwitchRate: Float = 0.7
    /// Heading noise while following or homing, rad²/s. MODEL.
    static let walkTurnDiffusion: Float = 0.05
    /// Homing: ants going home steer towards the nest as sharply as the
    /// 5 mm turning radius allows (rad/s at walking speed). MODEL: ants home
    /// by path integration (not modelled in detail); turning as tightly as it
    /// can keeps the U-turn at the food, where the trail begins, small.
    static let homingTurnRate: Float = walkSpeed / minTurnRadius
    /// Going home along a trail, the homing turn counts this much beside the
    /// trail's pull. MODEL: "ants that meet scent follow the gradient (and
    /// reinforce on the way home)" — homing ants use the trail they meet.
    static let homePullOnTrail: Float = 0.3
    /// A searcher that finds nothing turns for home after this long, s. MODEL
    /// (unsuccessful scouts "returned to the nest without laying a trail",
    /// Mailleux et al. 2000).
    static let searchGiveUp: Float = 20

    // MARK: - at the sugar, and in the nest

    // Sucrose has no vapour; the ant finds it by touch (step 26). On contact
    // it taps: step 30's 4 antennal strokes per second (Lenoir 1982 via
    // O'Fallon et al. 2016, 3–6/s), three taps as in step 68 → 0.75 s.
    static let tasteSeconds: Float = 0.75
    /// Then it feeds. MODEL, shortened: a real L. niger drinks until its own
    /// "desired volume" (Mailleux et al. 2000), and dry grains must first be
    /// dissolved; no duration for that was found.
    static let feedSeconds: Float = 4.0
    /// Inside the nest a returning forager passes its load on and recruits.
    /// Recruits leave after a delay, s. MODEL: no L. niger timing found; Le
    /// Breton & Fourcassié (2004, *Behav Ecol Sociobiol* 55: 242–250, abstract,
    /// opened) show recruits respond to "the residue of sugar smeared on the
    /// body of the recruiting workers coming back to the nest".
    static let recruitDelay: Float = 4.0
    /// Nest-mates each successful return sends out. MODEL.
    static let recruitsPerReturn: Int = 2
    /// How long a forager stays in the nest before going out again, s. MODEL.
    static let unloadSeconds: Float = 6.0

    // MARK: - bodies (for spacing only)

    /// Each ant's footprint for avoiding the others: a capsule along its body
    /// axis from `bodyRear` to `bodyFront` (mm from the origin) of radius
    /// `bodyRadius`; no ant steps so that its capsule comes within
    /// 2 × bodyRadius of another's. MODEL, from step 44's ant: head front
    /// ≈ +2.0, gaster rear ≈ −2.3, body half-width ≈ 0.5 (head 0.94 mm wide).
    /// Legs (feet out to ≈ 1.6 mm) are not kept apart: ants passing on a
    /// trail may brush legs, as real ones do. (Swerving to keep feet clear too
    /// was tried and knocked ants off the trail at every meeting.)
    static let bodyFront: Float = 2.0
    static let bodyRear: Float = -2.3
    static let bodyRadius: Float = 0.5
    /// Within this far of the sugar pile's edge, mm, ants don't keep apart:
    /// at food they crowd and climb over one another and the grains. MODEL.
    /// (Keeping them apart there queued them at the pile and stalled the
    /// whole colony; the film must draw the overlaps, e.g. one ant raised.)
    static let foodCrowdMargin: Float = 7.0
    /// Waiting longer than this lets an ant squeeze past for this far. MODEL.
    static let waitPatience: Float = 0.3
    static let squeezeDistance: Float = 8.0
    /// The edge: an ant closer than this to a wall, and not heading away from
    /// it, bears away, mm. MODEL: room for a 5 mm-radius half turn plus the
    /// antennae's 3.7 mm reach and a margin.
    static let edgeLookahead: Float = 13.0

    // MARK: - what the shared ant can draw

    /// The tightest turn the shared ant's gait holds, mm of radius at its
    /// origin: lib/ant/v1's `AntV1.minimumTurnRadius`, MEASURED on that module
    /// (at 4 mm a knee dips underground). Every heading change here happens
    /// while walking and is capped at step length / this radius, so no path
    /// ever curves tighter; and nothing turns on the spot (the module poses
    /// the body by distance walked, so it cannot).
    static let minTurnRadius: Float = 5.0
    /// The cap is set 2% wider, so the record — whose distances are Floats,
    /// as the module's Track keeps them — still reads ≥ 5 mm after rounding.
    static let turnCapMargin: Float = 1.02

    // The dab's timing, as lib/ant/v1 and step 44 time it: a 0.2 s stop in
    // all (Czaczkes et al. 2016), standing still for 0.06 s of it and easing
    // down and up over the rest; the gaster tip is fully down for 80% of the
    // standing time and bends over 0.05 s. The 0.06 and 0.05 are MODEL (step 44).
    static let dabHold: Float = 0.06
    static let dabRamp: Float = (dabPause - dabHold) / 2
    static let dabBend: Float = 0.05

    // MARK: - time step

    /// One simulation tick, s. MODEL: at 30.4 mm/s an ant moves 0.51 mm, one
    /// grid cell, per tick; 2 ticks per film frame at 30 fps is real time.
    static let tick: Double = 1.0 / 60.0
}
