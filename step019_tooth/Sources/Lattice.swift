// A two-dimensional spring lattice, and the brittle fracture that falls out of
// it.
//
// Every step of this project so far has DRAWN something: fetched coordinates,
// posed them, traced rays through them. This one has to SOLVE something,
// because the subject is a stress field and nobody deposits those.
//
// The model is the simplest thing that can carry the physics honestly: a
// triangular lattice of point masses joined by linear springs. Central forces
// only — no bond-bending term — which fixes the Poisson ratio at exactly 1/3
// (see `latticePoissonRatio` below). That is a constraint, not a choice, and it
// happens to land close to the measured values for both tissues here, which is
// luck and is stated as luck.
//
// Why a triangular lattice rather than a square one: a square lattice of
// central-force springs has no resistance to shear at all — it folds flat — and
// the usual patch, adding diagonals, makes the elasticity depend on direction.
// Six neighbours at 60° gives isotropic 2D elasticity from central forces
// alone, which is what we want, since enamel's ANISOTROPY should come from the
// rods we put in, not from the mesh we happened to choose.
//
// Sources for the elastic constants are in Tooth.swift, next to the values.
//
// THE ONE THING THIS LATTICE CANNOT DO, said here because it is load-bearing:
// Poisson's ratio is 1/3 and there is no dial for it. Enamel is about 0.30 and
// dentin about 0.31, so for the tooth that is luck. For the PERIODONTAL
// LIGAMENT it is not: the ligament is a fluid-filled tissue with a Poisson ratio
// near 0.45, nearly incompressible, and 1/3 makes it far more squeezable than it
// is. That shows up directly in the answer — the tooth in this model moves about
// 270 µm under 1000 N where a real one moves tens — and there is no way to fix
// it without a bond-bending term, which would cost the isotropy the triangular
// lattice was chosen for.

import Foundation
import simd

// MARK: - what a triangular central-force lattice is, as elasticity

/// A 2D triangular lattice of central-force springs has Lamé constants
/// λ = μ = (√3/4)·k, so
///
///     E₂D = 4μ(λ+μ)/(λ+2μ) = (2/√3)·k        ν = λ/(λ+2μ) = 1/3
///
/// Both are forced by the geometry. The spring constant is the only dial.
let latticePoissonRatio: Double = 1.0 / 3.0

/// The spring constant that makes a lattice of thickness `t` behave as a sheet
/// of Young's modulus `E`. Inverting E₂D = (2/√3)·k/t.
func springConstant(youngsModulus e: Double, thickness t: Double) -> Double {
    let root3: Double = 3.0.squareRoot()
    let half: Double = root3 / 2
    return half * e * t
}

/// And back again, so a test can check the round trip against the identity
/// rather than against a number I typed twice.
func youngsModulus(springConstant k: Double, thickness t: Double) -> Double {
    let root3: Double = 3.0.squareRoot()
    let two: Double = 2
    return (two / root3) * k / t
}

// MARK: - the lattice

/// One spring. `a` and `b` index into the node arrays; `rest` is its unstrained
/// length; `k` its stiffness; `limit` the tensile strain past which it breaks.
/// A broken bond carries no force ever again — cracks do not heal, and a test
/// asserts it.
struct Bond {
    var a: Int32
    var b: Int32
    var rest: Float
    var k: Float
    var limit: Float
    var broken: Bool
}

/// What a node is made of. The index is into the material table; `.outside`
/// nodes are not part of the solid at all.
enum ToothTissue: Int32, CaseIterable {
    case outside = 0
    case enamel = 1
    case dej = 2          // the graded junction — its own material, not a line
    case dentin = 3
    case pulp = 4
    case cementum = 5
    case pdl = 6          // periodontal ligament: what the tooth actually hangs in
    case bone = 7         // alveolar bone, held rigid — see `kindAt`

    var isSolid: Bool { self != .outside && self != .pulp }

    /// The tissues a crack may be counted in. Bone and ligament are not tooth.
    var isTooth: Bool {
        switch self {
        case .enamel, .dej, .dentin, .cementum: return true
        default: return false
        }
    }

    var name: String {
        switch self {
        case .outside: return "outside"
        case .enamel: return "enamel"
        case .dej: return "dentino-enamel junction"
        case .dentin: return "dentin"
        case .pulp: return "pulp"
        case .cementum: return "cementum"
        case .pdl: return "periodontal ligament"
        case .bone: return "alveolar bone"
        }
    }
}

struct TissueProperties {
    var youngsModulus: Double     // Pa
    var tensileStrength: Double   // Pa, the stress at which it fails in tension
    var name: String

    /// What a bond entirely inside this tissue fails at.
    var criticalStrain: Double { tensileStrength / youngsModulus }
}

/// How a node is constrained.
enum NodeKind: Int32 {
    case free = 0
    case fixed = 1        // pinned: the root, held by bone
    case loaded = 2       // where the opposing tooth presses
}

final class Lattice {
    let spacing: Double            // metres between neighbouring nodes
    let thickness: Double          // metres out of plane
    let cols: Int
    let rows: Int

    private(set) var position: [SIMD2<Float>] = []     // current, metres
    private(set) var rest: [SIMD2<Float>] = []         // unloaded
    private(set) var tissue: [ToothTissue] = []
    private(set) var kind: [NodeKind] = []
    private(set) var load: [SIMD2<Float>] = []         // applied force, newtons
    private(set) var bonds: [Bond] = []
    private(set) var bondsOf: [[Int32]] = []           // node → its bonds

    /// Where a node sits in the plane when nothing is loaded. Rows are offset
    /// by half a spacing, which is what makes the neighbours six-fold.
    func restPosition(col i: Int, row j: Int) -> SIMD2<Float> {
        let a: Double = spacing
        let offset: Double = (j % 2 == 0) ? 0 : a / 2
        let x: Double = Double(i) * a + offset
        let root3: Double = 3.0.squareRoot()
        let y: Double = Double(j) * a * root3 / 2
        return SIMD2<Float>(Float(x), Float(y))
    }

    func index(col i: Int, row j: Int) -> Int { j * cols + i }

    init(cols: Int, rows: Int, spacing: Double, thickness: Double,
         properties: [ToothTissue: TissueProperties],
         tissueAt: (SIMD2<Float>) -> ToothTissue,
         kindAt: (SIMD2<Float>, ToothTissue) -> NodeKind) {
        self.cols = cols
        self.rows = rows
        self.spacing = spacing
        self.thickness = thickness

        position.reserveCapacity(cols * rows)
        for j in 0..<rows {
            for i in 0..<cols {
                let p: SIMD2<Float> = restPosition(col: i, row: j)
                rest.append(p)
                position.append(p)
                let t: ToothTissue = tissueAt(p)
                tissue.append(t)
                kind.append(kindAt(p, t))
                load.append(.zero)
            }
        }

        // Six neighbours: two along the row, and two in each adjacent row. Only
        // the forward half of each pair is walked, so every bond is made once.
        bondsOf = Array(repeating: [], count: cols * rows)
        for j in 0..<rows {
            let shift: Int = (j % 2 == 0) ? -1 : 0
            for i in 0..<cols {
                let here: Int = index(col: i, row: j)
                guard tissue[here].isSolid else { continue }
                var neighbours: [(Int, Int)] = [(i + 1, j)]
                if j + 1 < rows {
                    neighbours.append((i + shift, j + 1))
                    neighbours.append((i + shift + 1, j + 1))
                }
                for (ni, nj) in neighbours {
                    guard ni >= 0, ni < cols, nj >= 0, nj < rows else { continue }
                    let there: Int = index(col: ni, row: nj)
                    guard tissue[there].isSolid else { continue }
                    addBond(here, there, properties: properties)
                }
            }
        }
        computeCellFractions()
    }

    /// A bond spanning two tissues gets the softer stiffness and the smaller
    /// tensile STRENGTH — it is only as strong as its weaker end. That matters
    /// at the junction, where it is the whole point.
    ///
    /// Taking the smaller STRAIN instead is wrong, and it was wrong here for a
    /// while: a bond between the ligament (50 MPa) and cementum (15 GPa) got the
    /// ligament's stiffness and cementum's critical strain of 0.23%, so it
    /// failed at 0.12 MPa and the whole ligament interface tore before the tooth
    /// felt anything. Strength is the property that belongs to the material;
    /// strain is that strength divided by whatever stiffness the bond ends up
    /// with, so it has to be computed after the stiffness, not before.
    private func addBond(_ a: Int, _ b: Int, properties: [ToothTissue: TissueProperties]) {
        guard let pa = properties[tissue[a]], let pb = properties[tissue[b]] else { return }
        let ea: Double = pa.youngsModulus
        let eb: Double = pb.youngsModulus
        let softer: Double = min(ea, eb)
        let k: Double = springConstant(youngsModulus: softer, thickness: thickness)
        let weakest: Double = min(pa.tensileStrength, pb.tensileStrength)
        let limit: Double = weakest / softer
        let d: SIMD2<Float> = rest[b] - rest[a]
        let restLength: Float = simd_length(d)
        bonds.append(Bond(a: Int32(a), b: Int32(b), rest: restLength,
                          k: Float(k), limit: Float(limit), broken: false))
        let id = Int32(bonds.count - 1)
        bondsOf[a].append(id)
        bondsOf[b].append(id)
    }

    func setLoad(_ f: SIMD2<Float>, at node: Int) { load[node] = f }

    // MARK: - solving

    // The spring force is not quite linear in position, because the bond
    // direction turns as the node moves. At the strains this render works in —
    // enamel breaks below 0.1% — that second-order term is negligible, so the
    // system is linearised:
    //
    //     f = K u     with    K u |bond = k (n̂ · (u_b − u_a)) n̂
    //
    // which is symmetric and positive definite on the free nodes. That buys
    // conjugate gradient, and CG is the difference between this working and
    // not. Jacobi relaxation — even with momentum — needs iterations going as
    // the SQUARE of the longest span for a bending problem, and a bending
    // problem is precisely what a loaded tooth is. Measured on a cantilever:
    // Jacobi left a residual forty times the applied load after 120,000 sweeps.
    // CG converges the same beam to 1e−6 in a few hundred.

    /// K·u, assembled on the fly. `unit` is each bond's rest direction, which
    /// is fixed, so the operator never has to be rebuilt as the solution moves.
    func applyStiffness(_ u: [SIMD2<Float>], into out: inout [SIMD2<Float>]) {
        for i in 0..<out.count { out[i] = .zero }
        for bond in bonds where !bond.broken {
            let a = Int(bond.a)
            let b = Int(bond.b)
            let dir: SIMD2<Float> = bondDirection(bond)
            let rel: SIMD2<Float> = u[b] - u[a]
            let along: Float = simd_dot(dir, rel)
            // K, not the restoring force: the elastic energy's second
            // derivative. Getting this sign backwards makes the operator
            // negative definite and conjugate gradient quits on its first
            // step, which is exactly what it did.
            let pull: SIMD2<Float> = dir * (bond.k * along)
            out[a] -= pull
            out[b] += pull
        }
        // Constrained nodes contribute nothing and move not at all.
        for i in 0..<out.count where kind[i] == .fixed || !tissue[i].isSolid {
            out[i] = .zero
        }
    }

    func bondDirection(_ bond: Bond) -> SIMD2<Float> {
        let d: SIMD2<Float> = rest[Int(bond.b)] - rest[Int(bond.a)]
        let length: Float = simd_length(d)
        return length > 0 ? d / length : SIMD2<Float>(1, 0)
    }

    /// The per-component diagonal of K, Σ k n̂ⱼ², used to precondition. A node
    /// with no unbroken bonds gets 1 so the division is safe; it cannot move
    /// anyway because nothing pulls on it.
    func stiffnessDiagonal() -> [SIMD2<Float>] {
        var d = [SIMD2<Float>](repeating: .zero, count: position.count)
        for bond in bonds where !bond.broken {
            let dir: SIMD2<Float> = bondDirection(bond)
            let contribution = SIMD2<Float>(bond.k * dir.x * dir.x, bond.k * dir.y * dir.y)
            d[Int(bond.a)] += contribution
            d[Int(bond.b)] += contribution
        }
        for i in 0..<d.count {
            if d[i].x <= 0 { d[i].x = 1 }
            if d[i].y <= 0 { d[i].y = 1 }
        }
        return d
    }

    private func constrain(_ v: inout [SIMD2<Float>]) {
        for i in 0..<v.count where kind[i] == .fixed || !tissue[i].isSolid {
            v[i] = .zero
        }
    }

    /// Preconditioned conjugate gradient for K u = f. Returns the residual norm
    /// as a fraction of the load norm, and how many iterations it took, so a
    /// caller can assert convergence rather than hope for it.
    @discardableResult
    func solve(tolerance: Float = 1e-6, maxIterations: Int = 4000,
               warmStart: Bool = false) -> (residual: Float, iterations: Int) {
        let n = position.count
        var u = [SIMD2<Float>](repeating: .zero, count: n)
        var r: [SIMD2<Float>] = load
        // The yardstick is the APPLIED load, not the starting residual. Measuring
        // against the starting residual would make a warm start look like a
        // tighter convergence than a cold one on the same problem.
        constrain(&r)
        let loadNorm: Float = norm(r)
        guard loadNorm > 0 else { displacement_ = u; return (0, 0) }
        // Warm starting matters once cracks are running: a crack step changes a
        // handful of bonds out of forty thousand, so the previous displacement
        // is already most of the answer. r ← f − K u₀ is the only change CG
        // needs; everything below is the standard algorithm on that residual.
        if warmStart && displacement_.count == n {
            u = displacement_
            constrain(&u)
            var ku = [SIMD2<Float>](repeating: .zero, count: n)
            applyStiffness(u, into: &ku)
            for i in 0..<n { r[i] -= ku[i] }
            constrain(&r)
        }

        let precond: [SIMD2<Float>] = stiffnessDiagonal()
        var z = [SIMD2<Float>](repeating: .zero, count: n)
        for i in 0..<n { z[i] = r[i] / precond[i] }
        constrain(&z)
        var p: [SIMD2<Float>] = z
        var rz: Float = dot(r, z)
        var ap = [SIMD2<Float>](repeating: .zero, count: n)
        var iterations = 0

        breakdown = false
        for it in 0..<maxIterations {
            iterations = it + 1
            applyStiffness(p, into: &ap)
            let pap: Float = dot(p, ap)
            // pᵀKp ≤ 0 on a matrix that is supposed to be positive definite means
            // the lattice has developed a zero-energy mode — a hinge or a
            // four-bar linkage left behind by a crack. Bailing out silently here
            // is how a run full of ten-kilometre displacements got reported as a
            // converged solve, so it is recorded.
            guard pap > 0 else { breakdown = true; break }
            let alpha: Float = rz / pap
            for i in 0..<n {
                u[i] += p[i] * alpha
                r[i] -= ap[i] * alpha
            }
            constrain(&r)
            if norm(r) <= tolerance * loadNorm { break }
            for i in 0..<n { z[i] = r[i] / precond[i] }
            constrain(&z)
            let rzNext: Float = dot(r, z)
            let beta: Float = rzNext / rz
            rz = rzNext
            for i in 0..<n { p[i] = z[i] + p[i] * beta }
        }
        displacement_ = u
        for i in 0..<n { position[i] = rest[i] + u[i] }
        return (norm(r) / loadNorm, iterations)
    }

    private var displacement_: [SIMD2<Float>] = []

    /// Set when the solve ran into a non-positive pᵀKp. See `solve`.
    private(set) var breakdown: Bool = false

    /// The largest displacement anywhere, in metres. The single number that says
    /// whether the answer is elasticity or a mechanism: a tooth under 1000 N
    /// moves tens of microns, so anything orders above that is a fragment
    /// swinging about a hinge the crack left behind. A converged residual does
    /// NOT rule this out — CG converges happily on a singular system if the load
    /// happens to be nearly orthogonal to the null space.
    var maxDisplacement: Float {
        var best: Float = 0
        for u in displacement_ {
            guard u.x.isFinite, u.y.isFinite else { return .infinity }
            let m: Float = simd_length(u)
            if m > best { best = m }
        }
        return best
    }

    /// Cut a free fragment loose: the connected set of nodes, over unbroken
    /// bonds, whose displacement is above `threshold`. A cusp held on by one
    /// node is a hinge, which a central-force lattice cannot resist at all and a
    /// real solid can; calling it a fragment that has broken off is the honest
    /// reading, and it is a DIFFERENT finding from a crack that arrested.
    ///
    /// Returns how many bonds were cut.
    @discardableResult
    func detachRunaway(threshold: Float) -> Int {
        var seed: [Int] = []
        for n in 0..<position.count where tissue[n].isSolid && kind[n] != .fixed {
            let u: SIMD2<Float> = displacementOf(n)
            if !u.x.isFinite || !u.y.isFinite || simd_length(u) > threshold { seed.append(n) }
        }
        guard !seed.isEmpty else { return 0 }
        var inFragment = [Bool](repeating: false, count: position.count)
        var queue: [Int] = []
        for n in seed where !inFragment[n] { inFragment[n] = true; queue.append(n) }
        var head = 0
        while head < queue.count {
            let n = queue[head]
            head += 1
            for id in bondsOf[n] where !bonds[Int(id)].broken {
                let other: Int = Int(bonds[Int(id)].a) == n ? Int(bonds[Int(id)].b) : Int(bonds[Int(id)].a)
                guard !inFragment[other], kind[other] != .fixed else { continue }
                let u: SIMD2<Float> = displacementOf(other)
                let big: Bool = !u.x.isFinite || !u.y.isFinite || simd_length(u) > threshold * 0.25
                if big { inFragment[other] = true; queue.append(other) }
            }
        }
        var cut = 0
        for n in 0..<position.count where inFragment[n] {
            load[n] = .zero
            for id in bondsOf[n] where !bonds[Int(id)].broken {
                bonds[Int(id)].broken = true
                breakOrder.append(id)
                cut += 1
            }
        }
        detachedNodes += queue.count
        return cut
    }

    private(set) var detachedNodes: Int = 0

    private func dot(_ a: [SIMD2<Float>], _ b: [SIMD2<Float>]) -> Float {
        var s: Float = 0
        for i in 0..<a.count { s += simd_dot(a[i], b[i]) }
        return s
    }

    private func norm(_ a: [SIMD2<Float>]) -> Float {
        dot(a, a).squareRoot()
    }

    /// Break every bond past its tensile limit, all at once. Compression never
    /// breaks a bond: enamel fails in tension, and a model that let it crumble
    /// under compression would be telling a different story from the
    /// literature. Returns how many broke.
    ///
    /// This is kept because it is what the first version of this step did, and
    /// because a test uses it to show WHY it is wrong: breaking everything that
    /// is over the limit in one round is an avalanche, not a crack. The stress
    /// a crack sheds has to be redistributed before the next bond is chosen, and
    /// doing that in one round means every bond is judged against a field that
    /// no longer exists by the time it breaks. Use `breakWorst` instead.
    @discardableResult
    func breakAllOverstrained() -> Int {
        var broke = 0
        for i in 0..<bonds.count {
            guard !bonds[i].broken else { continue }
            if bondStrain(i) > bonds[i].limit {
                bonds[i].broken = true
                broke += 1
            }
        }
        return broke
    }

    /// How close a bond is to failing: tensile strain over its own limit. Above
    /// 1 it should break. Negative (compressed) bonds return 0 — this is the one
    /// place the tension-only rule is written, and everything else reads it.
    func overload(_ i: Int) -> Float {
        let bond = bonds[i]
        guard !bond.broken, bond.limit > 0 else { return 0 }
        let strain: Float = bondStrain(i)
        // NaN compares false against everything, so a blown-up solve used to
        // read as "nothing is overloaded" and the run reported itself arrested.
        // It is now infinite instead, which nothing can mistake for calm.
        guard strain.isFinite else { return .infinity }
        guard strain > 0 else { return 0 }
        return strain / bond.limit
    }

    /// Break the most overloaded bond, and any bond within `window` of it.
    ///
    /// This is the fuse-model rule and it is the difference between a crack and
    /// a demolition. Breaking every over-limit bond at once judges them all
    /// against a stress field that the first break already invalidated; the
    /// field has to be re-solved between breaks or the crack runs away. `window`
    /// of 0 is the strict one-bond-at-a-time rule; a few percent lets a crack
    /// tip that is genuinely two bonds wide open both, which costs a lot of
    /// solves to refuse and changes nothing.
    ///
    /// Returns the indices broken, worst first, and the worst overload SEEN —
    /// which is what tells a caller whether anything was over the limit at all.
    @discardableResult
    func breakWorst(window: Float = 0.02, limit maxBreaks: Int = 8) -> (broken: [Int], worst: Float) {
        var worst: Float = 0
        for i in 0..<bonds.count {
            let r: Float = overload(i)
            if r > worst { worst = r }
        }
        guard worst > 1 else { return ([], worst) }
        let cut: Float = worst * (1 - window)
        var candidates: [(Int, Float)] = []
        for i in 0..<bonds.count {
            let r: Float = overload(i)
            if r >= cut { candidates.append((i, r)) }
        }
        candidates.sort { $0.1 > $1.1 }
        var broken: [Int] = []
        for (i, _) in candidates.prefix(maxBreaks) {
            bonds[i].broken = true
            broken.append(i)
            breakOrder.append(Int32(i))
        }
        return (broken, worst)
    }

    /// Every bond that has ever broken, in the order it broke. Nothing is ever
    /// removed from this, and `heal` is not a method on this class — a test
    /// checks the broken set only ever grows.
    private(set) var breakOrder: [Int32] = []

    /// Multiply every bond's critical strain by a factor. Used to put quenched
    /// disorder into the tissues — a real solid does not fail at one strain, it
    /// fails at a distribution of them, and without some scatter a lattice
    /// fractures along a straight line of its own symmetry axes.
    func scaleLimits(_ factor: (Int) -> Float) {
        for i in 0..<bonds.count { bonds[i].limit *= factor(i) }
    }

    /// Cut loose any node the crack has left dangling, and keep cutting until
    /// nothing dangles.
    ///
    /// A node held by one spring is not part of a solid: it is free to swing
    /// about that spring, the stiffness operator is singular in that direction,
    /// and conjugate gradient cannot converge on it. Left in, it produces
    /// displacements of millimetres, strains of 700,000 and a crack that eats
    /// the surface it started on — which is exactly what it did here before this
    /// existed. Two bonds that are not opposite each other pin a point in the
    /// plane; fewer than that, and the node is spall.
    @discardableResult
    func pruneDetached() -> Int {
        var cut = 0
        var changed = true
        while changed {
            changed = false
            for n in 0..<position.count {
                guard tissue[n].isSolid, kind[n] != .fixed else { continue }
                var live: [SIMD2<Float>] = []
                var ids: [Int] = []
                for id in bondsOf[n] where !bonds[Int(id)].broken {
                    ids.append(Int(id))
                    var dir: SIMD2<Float> = bondDirection(bonds[Int(id)])
                    if Int(bonds[Int(id)].b) == n { dir = -dir }
                    live.append(dir)
                }
                var loose: Bool = live.count < 2
                if live.count == 2 {
                    let d: Float = simd_dot(live[0], live[1])
                    if d < -0.99 { loose = true }
                }
                guard loose, !ids.isEmpty else { continue }
                for id in ids {
                    bonds[id].broken = true
                    breakOrder.append(Int32(id))
                    cut += 1
                }
                changed = true
            }
        }
        return cut
    }

    /// Make every bond touching these nodes unbreakable. Used on the contact
    /// patch: a load applied to a handful of nodes has a singularity of its own,
    /// and a model that lets the loading fixture fail is measuring the fixture.
    func reinforce(around nodes: [Int]) {
        var seen = Set<Int>()
        for n in nodes {
            for id in bondsOf[n] where !seen.contains(Int(id)) {
                seen.insert(Int(id))
                bonds[Int(id)].limit = Float(1e6)
            }
        }
    }

    /// Put a bond back. Only the mutation check calls this, and test 7 exists to
    /// catch it doing so.
    func healForMutation(_ i: Int) { bonds[i].broken = false }

    /// Linearised bond strain: the relative displacement along the bond's own
    /// rest direction, over its rest length.
    func bondStrain(_ i: Int) -> Float {
        let bond = bonds[i]
        guard !bond.broken else { return 0 }
        let dir: SIMD2<Float> = bondDirection(bond)
        let rel: SIMD2<Float> = displacementOf(Int(bond.b)) - displacementOf(Int(bond.a))
        return simd_dot(dir, rel) / bond.rest
    }

    func displacementOf(_ n: Int) -> SIMD2<Float> {
        displacement_.isEmpty ? .zero : displacement_[n]
    }

    /// Displacement of a node from where it started.
    func displacement(_ n: Int) -> SIMD2<Float> { displacementOf(n) }

    func resetToRest() {
        position = rest
    }

    // MARK: - reading a stress out of the bonds

    /// The virial stress at a node: sum over its bonds of (force × separation),
    /// divided by the area each node owns. For a triangular lattice that area
    /// is (√3/2)·a².
    ///
    /// Returned as the 2D symmetric tensor (σxx, σyy, σxy) in pascals.
    func nodeStress(_ n: Int) -> SIMD3<Float> {
        var sxx: Float = 0
        var syy: Float = 0
        var sxy: Float = 0
        for id in bondsOf[n] {
            let bond = bonds[Int(id)]
            guard !bond.broken else { continue }
            let a = Int(bond.a)
            let b = Int(bond.b)
            let unit: SIMD2<Float> = bondDirection(bond)
            let d: SIMD2<Float> = rest[b] - rest[a]
            let magnitude: Float = bond.k * bondStrain(Int(id)) * bond.rest
            let fx: Float = unit.x * magnitude
            let fy: Float = unit.y * magnitude
            // Half to each end of the bond.
            sxx += 0.5 * fx * d.x
            syy += 0.5 * fy * d.y
            sxy += 0.5 * 0.5 * (fx * d.y + fy * d.x)
        }
        let root3: Double = 3.0.squareRoot()
        let cell: Double = (root3 / 2) * spacing * spacing * thickness
        // The area a node OWNS, not the area a node would own if it were in the
        // middle of the sheet. A node on a flat free surface belongs to three
        // triangles rather than six and owns half a cell, so dividing by the
        // full cell would report half the stress — and the surface is precisely
        // where a brittle solid is read. This was worth two whole days.
        let owned: Double = cell * Double(cellFraction[n])
        let scale: Float = Float(1.0 / max(owned, cell * 1e-3))
        return SIMD3<Float>(sxx * scale, syy * scale, sxy * scale)
    }

    /// The fraction of a full lattice cell each node owns: the number of
    /// triangles it belongs to, over six. Computed once from the geometry, not
    /// from the current breakage — a node inside an open crack is on a new free
    /// surface, and this does not pretend to know that.
    private(set) var cellFraction: [Float] = []

    /// The six neighbours of a node, in angular order, or −1 where the lattice
    /// or the tissue map has none. Angular order is what makes the triangle
    /// count a walk over consecutive pairs.
    func neighbours(col i: Int, row j: Int) -> [Int] {
        return neighbourSites(col: i, row: j).map { k -> Int in
            guard k >= 0 else { return -1 }
            return tissue[k].isSolid ? k : -1
        }
    }

    /// The same six, but −1 ONLY where the grid ends. The caller can then ask
    /// what tissue is there, which is not the same question as whether it is
    /// part of the solid: a node beside the pulp chamber has a free surface, a
    /// node beside the ligament has a bonded one, and `neighbours` calls both of
    /// them −1.
    func neighbourSites(col i: Int, row j: Int) -> [Int] {
        let shift: Int = (j % 2 == 0) ? -1 : 0
        let offsets: [(Int, Int)] = [(i + 1, j), (i + shift + 1, j + 1), (i + shift, j + 1),
                                     (i - 1, j), (i + shift, j - 1), (i + shift + 1, j - 1)]
        return offsets.map { (ni, nj) -> Int in
            guard ni >= 0, ni < cols, nj >= 0, nj < rows else { return -1 }
            return index(col: ni, row: nj)
        }
    }

    fileprivate func computeCellFractions() {
        var out = [Float](repeating: 0, count: cols * rows)
        for j in 0..<rows {
            for i in 0..<cols {
                let here: Int = index(col: i, row: j)
                guard tissue[here].isSolid else { continue }
                let ring: [Int] = neighbours(col: i, row: j)
                var triangles: Int = 0
                for k in 0..<6 where ring[k] >= 0 && ring[(k + 1) % 6] >= 0 { triangles += 1 }
                let six: Float = 6
                out[here] = Float(triangles) / six
            }
        }
        cellFraction = out
    }

    /// The largest tensile stress carried by any one bond at a node, E·ε along
    /// that bond. No area, no cell, no free-surface correction — it is the
    /// quantity that actually decides whether this model breaks, read straight
    /// off the thing that breaks. Zero if every bond there is in compression.
    func peakBondStress(_ n: Int) -> Float {
        var best: Float = 0
        for id in bondsOf[n] {
            let i = Int(id)
            guard !bonds[i].broken else { continue }
            let strain: Float = bondStrain(i)
            guard strain > 0 else { continue }
            let e: Float = Float(youngsModulus(springConstant: Double(bonds[i].k),
                                               thickness: thickness))
            let stress: Float = e * strain
            if stress > best { best = stress }
        }
        return best
    }

    /// The larger principal stress. Positive means tension, which is the only
    /// thing that breaks a brittle solid — so this is the field the render
    /// colours by, not the von Mises stress a metals textbook would reach for.
    func maxPrincipalStress(_ n: Int) -> Float {
        let s: SIMD3<Float> = nodeStress(n)
        let mean: Float = (s.x + s.y) / 2
        let half: Float = (s.x - s.y) / 2
        let radius: Float = (half * half + s.z * s.z).squareRoot()
        return mean + radius
    }
}
