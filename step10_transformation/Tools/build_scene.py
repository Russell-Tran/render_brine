"""Builds Resources/scene.json: the E. coli envelope, a supercoiled pGLO, and
the calcium ions around it.

Unlike every earlier step, most of what this builds is a MODEL rather than a
measured structure, and the JSON records which is which so the render can say
so. Three levels are tagged on every part:

    "measured"   dimensions taken from published measurements of real cells
    "simulated"  shapes that molecular-dynamics studies produce
    "model"      a mechanism inferred from bulk behaviour, never observed

Measurements used, all checked rather than assumed:

  Periplasm width           12 nm (CEMOVIS) / 14 nm (cryo-ET), Matias-style
                            comparison in Wataru et al., J. Electron Microsc.
                            59:419 (2010). The brief said 15-25 nm; that is too
                            wide, so 13 nm is used here.
  Outer membrane to PG      ~11 nm, same source, agreed by both methods.
  Bilayer thickness         4.7 +/- 0.1 nm for a PE/PG/cardiolipin bilayer by
                            AFM (Langmuir 41:12301, 2025). The brief said ~7 nm,
                            which is the thicker figure cryo-EM density profiles
                            give including headgroups; 4.7 nm is the cleaner
                            measured value and is what is used.
  Inner membrane lipids     70-80% phosphatidylethanolamine, 20-25%
                            phosphatidylglycerol, <5% cardiolipin (same paper).
                            Modelled here as 75 / 21 / 4.
  Area per lipid            0.588 nm^2 for POPE (Hills et al., J. Comput. Chem.
                            37, 2016). The brief said 0.65, which is nearer
                            POPC's 0.683.
  Supercoiling              sigma = -0.06 for a plasmid from E. coli in mid
                            exponential growth. Confirmed; the brief was right.
  Ca2+ radius               1.00 A ionic (Shannon 1976, six-coordinate);
                            ~4.1 A for the first hydration shell.

Coarse-graining: each lipid becomes 12 beads on the Martini plan (roughly four
heavy atoms to a bead), with a bead radius of 2.6 A. All-atom, a 100 x 100 nm
patch would be about 4.3 million atoms; at 12 beads a lipid it is 408,000
spheres, which the step 8a grid can handle.

Run: python3 Tools/build_scene.py
"""

import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "Resources", "scene.json")

# Everything in angstroms, to match every earlier step.
NM = 10.0

# The envelope is rendered edge-on, as a cross-section, so it must be wide
# across the view but only deep enough to look solid. Making it a slab rather
# than a square saves most of the lipids: 200 x 24 nm comes to about 390,000
# beads where 200 x 200 would be 3.3 million, and the extra depth is never seen.
PATCH_X = 200 * NM
PATCH_Z = 24 * NM
BILAYER = 4.7 * NM        # measured, AFM
PERIPLASM = 13.0 * NM     # measured, between the CEMOVIS and cryo-ET figures
OM_TO_PG = 11.0 * NM      # measured
PG_THICK = 2.5 * NM       # a single glycan layer; E. coli's is thin and sparse
LPS_EXTRA = 1.2 * NM      # the O-antigen side of the outer leaflet sits proud

AREA_PER_LIPID = 0.588 * NM * NM   # measured, POPE
BEAD_R = 2.6              # Martini bead radius, angstroms
BEADS_PER_LIPID = 12
N_ARMS = 4                # branches in the folded plectoneme; real plasmids fold

CA_IONIC = 1.00           # Shannon 1976
CA_HYDRATED = 4.1

# Colours are decided here so the tests can check them, and so the three
# evidence levels read consistently across both GIFs.
COLOR = {
    "lps":      [0.78, 0.55, 0.26],
    "om_head":  [0.93, 0.78, 0.42],
    "om_tail":  [0.74, 0.64, 0.42],
    "pg":       [0.55, 0.72, 0.50],
    "im_head":  [0.93, 0.72, 0.36],
    "im_tail":  [0.72, 0.60, 0.38],
    "dna":      [0.30, 0.62, 0.86],
    "calcium":  [0.95, 0.85, 0.35],
    "pore":     [0.88, 0.45, 0.38],
}


def rng(seed):
    return np.random.default_rng(seed)


# ---------------------------------------------------------------- the envelope

def leaflet(y_head, direction, n_per_side, spacing, jitter, seed, head_color, tail_color):
    """One leaflet of a bilayer: heads on a jittered square lattice at `y_head`,
    tails running `direction` (+1 or -1) into the middle of the bilayer.

    Returns a list of (x, y, z, r, color, part) beads.
    """
    r = rng(seed)
    beads = []
    half = BILAYER / 2
    # Two head beads, two glycerol, then eight tail beads down the length.
    offsets = [0.0, 0.10, 0.22, 0.30, 0.40, 0.52, 0.64, 0.76, 0.44, 0.56, 0.68, 0.80]
    kinds = ["head", "head", "head", "head"] + ["tail"] * 8
    lateral = [0.0, 0.0, 0.0, 0.0, -0.9, -1.0, -1.1, -1.2, 0.9, 1.0, 1.1, 1.2]
    n_z = max(int(PATCH_Z / spacing), 1)
    for i in range(n_per_side):
        for j in range(n_z):
            x = (i + 0.5) * spacing - PATCH_X / 2 + r.normal(0, jitter)
            z = (j + 0.5) * spacing - PATCH_Z / 2 + r.normal(0, jitter)
            # A small random tilt, as real lipids have. Kept modest on purpose:
            # the tilt is multiplied by depth, so a large one swings the deepest
            # tail beads sideways into the neighbouring lipid. A test measures
            # the closest approach and this is what keeps it physical.
            tilt = r.normal(0, 0.05, 2)
            for k in range(BEADS_PER_LIPID):
                # Measured from the head, into the bilayer, as a fraction of a
                # LEAFLET's thickness — not of the whole bilayer. Using the full
                # thickness sent each leaflet's tails 14 A past the midplane and
                # straight through the other leaflet, which a test caught.
                depth = offsets[k] * half
                bx = x + lateral[k] * BEAD_R + tilt[0] * depth
                bz = z + tilt[1] * depth
                by = y_head + direction * depth
                beads.append((bx, by, bz, BEAD_R, head_color if kinds[k] == "head" else tail_color, kinds[k]))
    return beads


def bilayer(y_center, seed, head_color, tail_color, outer_is_lps=False):
    """Both leaflets of one membrane, centred on `y_center`."""
    spacing = math.sqrt(AREA_PER_LIPID)
    n_per_side = int(PATCH_X / spacing)
    half = BILAYER / 2
    out = []
    upper = leaflet(y_center + half, -1, n_per_side, spacing, spacing * 0.06, seed,
                    COLOR["lps"] if outer_is_lps else head_color, tail_color)
    lower = leaflet(y_center - half, +1, n_per_side, spacing, spacing * 0.06, seed + 1,
                    head_color, tail_color)
    if outer_is_lps:
        # LPS carries a sugar chain standing proud of the membrane; two extra
        # beads a lipid, which is what makes the outer surface rough.
        r = rng(seed + 7)
        extra = []
        for k in range(0, len(upper), BEADS_PER_LIPID):
            x, y, z = upper[k][0], upper[k][1], upper[k][2]
            for m in (1, 2):
                extra.append((x + r.normal(0, 1.2), y + m * LPS_EXTRA / 2, z + r.normal(0, 1.2),
                              BEAD_R * 1.15, COLOR["lps"], "lps"))
        upper = upper + extra
    out.extend(upper)
    out.extend(lower)
    return out, n_per_side * max(int(PATCH_Z / spacing), 1) * 2


def peptidoglycan(y_center, seed):
    """A sparse glycan mesh: strands running in x, cross-linked in z.

    E. coli's is a single layer over most of the cell and is mostly holes,
    which is why a plasmid's problem is the membranes rather than this.
    """
    r = rng(seed)
    beads = []
    strand_gap = 4.5 * NM        # between glycan strands
    bead_gap = 1.0 * NM
    n_strands = max(int(PATCH_Z / strand_gap), 1)
    for i in range(n_strands):
        z = (i + 0.5) * strand_gap - PATCH_Z / 2 + r.normal(0, 2.0)
        y = y_center + r.normal(0, 1.5)
        x = -PATCH_X / 2
        while x < PATCH_X / 2:
            beads.append((x, y + r.normal(0, 0.8), z + r.normal(0, 0.8), 4.0, COLOR["pg"], "pg"))
            x += bead_gap
        # Peptide cross-links to the next strand, at intervals.
        if i + 1 < n_strands:
            for _ in range(int(PATCH_X / (9 * NM))):
                xc = r.uniform(-PATCH_X / 2, PATCH_X / 2)
                for f in np.linspace(0, 1, 5):
                    beads.append((xc, y + r.normal(0, 0.6), z + f * strand_gap, 3.2, COLOR["pg"], "pg"))
    return beads


# ---------------------------------------------------------------- the plasmid

def serpentine_axis(length, fold_width, n_arms, n):
    """The path the superhelix axis follows: a serpentine of `n_arms` straight
    arms joined by half-turn folds, laid out in x with the arms stacked in y.

    A single unbranched plectoneme of a 5.4 kb plasmid is a rod some hundreds of
    nanometres long — longer than the cell it lives in. Real plasmids fold and
    branch, which is what keeps 1.8 um of DNA inside a 2 um cell, so folding the
    axis is both the realistic choice and the one that fits a frame.

    Returns points along the axis and a smooth frame (tangent, normal, binormal).
    """
    turn_len = math.pi * fold_width / 2
    arm = (length - (n_arms - 1) * turn_len) / n_arms
    if arm <= 0:
        arm = length / n_arms * 0.5
    n_arm = max(int(n / (n_arms * 1.3)), 6)
    n_turn = max(int(n_arm * turn_len / max(arm, 1e-6)), 6)

    # A gentle opposite lean on alternate arms, so the finished plectoneme is a
    # three-dimensional object rather than a flat ribbon.
    lean = fold_width * 0.15
    pts = []
    for a_i in range(n_arms):
        y = (a_i - (n_arms - 1) / 2) * fold_width
        forward = (a_i % 2 == 0)
        for k in range(n_arm):
            t = k / max(n_arm - 1, 1)
            x = arm * (t - 0.5) if forward else arm * (0.5 - t)
            pts.append((x, y, lean * (0.5 - t) * (1 if forward else -1)))
        if a_i + 1 < n_arms:
            y2 = (a_i + 1 - (n_arms - 1) / 2) * fold_width
            side = 1.0 if forward else -1.0
            for k in range(1, n_turn):
                ang = math.pi * k / n_turn
                pts.append((side * (arm / 2 + fold_width / 2 * math.sin(ang)),
                            (y + y2) / 2 - (y2 - y) / 2 * math.cos(ang),
                            lean * 0.4 * math.sin(ang)))
    axis = np.array(pts)

    # Parallel transport rather than a Frenet frame, so it does not spin wildly
    # where the curvature vanishes along the straight arms.
    tan = np.gradient(axis, axis=0)
    tan /= np.linalg.norm(tan, axis=1, keepdims=True) + 1e-12
    nrm = np.zeros_like(axis)
    ref = np.array([0.0, 0.0, 1.0])
    v = ref - tan[0] * np.dot(ref, tan[0])
    nrm[0] = v / np.linalg.norm(v)
    for k in range(1, len(axis)):
        v = nrm[k - 1] - tan[k] * np.dot(nrm[k - 1], tan[k])
        nv = np.linalg.norm(v)
        nrm[k] = v / nv if nv > 1e-9 else nrm[k - 1]
    return axis, tan, nrm, np.cross(tan, nrm)


def plectoneme(n_bp, sigma, super_radius, seed):
    """A supercoiled plasmid as an interwound (plectonemic) superhelix wound
    around a folded axis.

    A relaxed circle of n_bp base pairs has Lk0 = n_bp / 10.5 turns. Underwinding
    it to a supercoiling density sigma removes dLk = sigma * Lk0 turns, and in a
    free plasmid most of that appears as writhe: the axis wraps around itself
    rather than the strands unwinding. The crossing number is set from dLk, and
    the writhe is then measured back off the finished curve by the Gauss double
    integral rather than assumed.

    Boles, White & Cozzarelli, J. Mol. Biol. 213:931 (1990), for the roughly
    3:1 split of linking difference between writhe and twist in a free plasmid.
    """
    lk0 = n_bp / 10.5
    d_lk = sigma * lk0
    target_writhe = 0.75 * d_lk

    contour = n_bp * 3.4               # angstroms of duplex
    n_pts = 2400

    def build(turns):
        return _wind(contour, turns, super_radius, n_pts, target_writhe)

    # One superhelical turn does not contribute exactly one unit of writhe: the
    # end fold and the arms' lean take some back. So solve for the winding that
    # actually produces the target, by bisection on the measured writhe, rather
    # than assuming the two are the same number.
    # The winding cannot be pushed past the point where it would consume the
    # whole contour: 2*pi*r*turns must stay under half the wound length, or the
    # superhelix axis has no length left and the model balls up.
    wound = contour * 0.88
    max_turns = 0.92 * wound / (4 * math.pi * super_radius)
    lo_turns, hi_turns = abs(target_writhe) * 0.4, min(abs(target_writhe) * 3.0, max_turns)
    curve = None
    for _ in range(18):
        mid = (lo_turns + hi_turns) / 2
        curve = build(mid)
        wr = writhe(curve)
        if abs(wr) < abs(target_writhe):
            lo_turns = mid
        else:
            hi_turns = mid
    wr = writhe(curve)
    if (wr < 0) != (target_writhe < 0):
        curve[:, 2] = -curve[:, 2]
        wr = writhe(curve)
    return curve, wr, d_lk, target_writhe


def _wind(contour, turns, super_radius, n_pts, target_writhe):
    """The interwound curve for a given number of superhelical turns."""
    # Two antiparallel strands wound about the axis, plus caps at each end of
    # the hairpin. The winding uses most of the contour.
    wound = contour * 0.88
    # Length of the axis the two strands can wrap, given how much contour the
    # helical winding itself consumes.
    axis_len = math.sqrt(max((wound / 2) ** 2 - (2 * math.pi * super_radius * turns) ** 2,
                             (wound * 0.15) ** 2))
    fold_width = 3.4 * super_radius

    n_half = n_pts // 2
    axis, tan, nrm, binorm = serpentine_axis(axis_len, fold_width, N_ARMS, n_half)

    pts = []
    m = len(axis)
    # Strand A, then strand B on the opposite side of the axis, traversed back.
    for k in range(m):
        th = 2 * math.pi * turns * k / (m - 1)
        pts.append(axis[k] + super_radius * (math.cos(th) * nrm[k] + math.sin(th) * binorm[k]))
    for k in range(m - 1, -1, -1):
        th = 2 * math.pi * turns * k / (m - 1) + math.pi
        pts.append(axis[k] + super_radius * (math.cos(th) * nrm[k] + math.sin(th) * binorm[k]))

    return np.array(pts)


def writhe(curve):
    """Writhe by the Gauss double integral, on the closed polygon `curve`.

    Wr = (1/4pi) * double integral over the curve of
         (dr1 x dr2) . (r1 - r2) / |r1 - r2|^3
    """
    p = curve
    n = len(p)
    nxt = np.roll(p, -1, axis=0)
    seg = nxt - p
    mid = (p + nxt) / 2
    total = 0.0
    for i in range(n):
        d = mid[i] - mid                      # (n, 3)
        dist = np.linalg.norm(d, axis=1)
        dist[i] = np.inf                      # skip the self term
        # Skip immediate neighbours: adjacent segments contribute a singular,
        # near-zero term that the discretisation handles badly.
        dist[(i + 1) % n] = np.inf
        dist[(i - 1) % n] = np.inf
        cross = np.cross(np.broadcast_to(seg[i], seg.shape), seg)
        total += float(np.sum(np.einsum("ij,ij->i", cross, d) / dist ** 3))
    return total / (4 * math.pi)


def dna_beads(curve, n_bp):
    """The duplex along the axis curve, coarse-grained: one bead per ~4 base
    pairs at the duplex's 10 A radius, which is the right size on screen at
    this scale and keeps the count sane."""
    # Resample the curve to one point per 4 bp of contour.
    seg = np.linalg.norm(np.diff(np.vstack([curve, curve[:1]]), axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    total = s[-1]
    want = max(int(n_bp / 4), 8)
    targets = np.linspace(0, total, want, endpoint=False)
    out = []
    for t in targets:
        i = int(np.searchsorted(s, t, side="right") - 1)
        i = min(max(i, 0), len(curve) - 1)
        f = (t - s[i]) / max(seg[i], 1e-9)
        p = curve[i] + f * (curve[(i + 1) % len(curve)] - curve[i])
        out.append((p[0], p[1], p[2], 10.0, COLOR["dna"], "dna"))
    return out


# ---------------------------------------------------------------- the ions

def calcium(dna, membrane_y, n_ions, seed):
    """Ca2+ around the DNA and against the membrane.

    This is the part of the render that is a MODEL. Calcium is known to screen
    the mutual repulsion between DNA's phosphates and the membrane's own
    negative charge, and to condense DNA; the arrangement drawn here is a
    plausible consequence of that, not a measured distribution. Ions are placed
    where the electrostatics would put them: in a shell around the duplex, and
    in a layer on the membrane surface.
    """
    r = rng(seed)
    pts = np.array([[d[0], d[1], d[2]] for d in dna])
    out = []
    n_dna = int(n_ions * 0.6)
    for _ in range(n_dna):
        base = pts[r.integers(0, len(pts))]
        d = r.normal(0, 1, 3)
        d /= np.linalg.norm(d)
        shell = 10.0 + CA_HYDRATED + r.uniform(0, 6)
        p = base + d * shell
        out.append((p[0], p[1], p[2], CA_HYDRATED, COLOR["calcium"], "calcium"))
    for _ in range(n_ions - n_dna):
        x = r.uniform(-PATCH_X / 2, PATCH_X / 2)
        z = r.uniform(-PATCH_Z / 2, PATCH_Z / 2)
        y = membrane_y + r.uniform(2, 14)
        out.append((x, y, z, CA_HYDRATED, COLOR["calcium"], "calcium"))
    return out


# ---------------------------------------------------------------- assembly

def main():
    # Lay the envelope out along y. Outside the cell is +y.
    om_center = 0.0
    pg_center = om_center - BILAYER / 2 - OM_TO_PG
    im_center = om_center - BILAYER / 2 - PERIPLASM - BILAYER / 2
    om_outer_face = om_center + BILAYER / 2 + LPS_EXTRA

    om, om_lipids = bilayer(om_center, 101, COLOR["om_head"], COLOR["om_tail"], outer_is_lps=True)
    im, im_lipids = bilayer(im_center, 201, COLOR["im_head"], COLOR["im_tail"])
    pg = peptidoglycan(pg_center, 301)

    n_bp = 5371
    sigma = -0.06
    curve, wr, d_lk, target_wr = plectoneme(n_bp, sigma, super_radius=5.0 * NM, seed=401)
    dna = dna_beads(curve, n_bp)
    # The plasmid starts well clear of the membrane and drifts in; the render
    # moves it, so the geometry here is its resting shape at the origin.
    dna_pts = np.array([[d[0], d[1], d[2]] for d in dna])
    dna_extent = float((dna_pts.max(axis=0) - dna_pts.min(axis=0)).max())   # the full span

    ions = calcium(dna, om_outer_face, 900, 501)

    def pack(beads, group, evidence):
        return [{"p": [round(float(b[0]), 2), round(float(b[1]), 2), round(float(b[2]), 2)],
                 "r": round(float(b[3]), 2), "c": b[4], "part": b[5],
                 "group": group, "evidence": evidence} for b in beads]

    scene = {
        "units": "angstrom",
        "patch_x": PATCH_X, "patch_z": PATCH_Z,
        "layers": {
            "om_center": om_center, "pg_center": pg_center, "im_center": im_center,
            "bilayer": BILAYER, "periplasm": PERIPLASM, "om_to_pg": OM_TO_PG,
            "pg_thickness": PG_THICK, "om_outer_face": om_outer_face,
        },
        "measurements": {
            "bilayer_nm": 4.7, "bilayer_source": "AFM of a PE/PG/cardiolipin bilayer, Langmuir 41:12301 (2025)",
            "periplasm_nm": 13.0, "periplasm_source": "12 nm CEMOVIS / 14 nm cryo-ET, J. Electron Microsc. 59:419 (2010)",
            "om_to_pg_nm": 11.0, "om_to_pg_source": "same, agreed by both methods",
            "area_per_lipid_nm2": 0.588, "area_source": "POPE, Hills et al., J. Comput. Chem. 37 (2016)",
            "lipid_mix": "75% phosphatidylethanolamine, 21% phosphatidylglycerol, 4% cardiolipin",
            "sigma": sigma, "sigma_source": "plasmid from E. coli in mid-exponential growth",
            "ca_ionic_A": CA_IONIC, "ca_hydrated_A": CA_HYDRATED,
        },
        "plasmid": {
            "bp": n_bp, "sigma": sigma, "lk0": n_bp / 10.5, "delta_lk": d_lk,
            "target_writhe": target_wr, "measured_writhe": wr,
            "super_radius": 5.0 * NM, "extent": dna_extent,
            "contour_nm": n_bp * 0.34,
        },
        "counts": {
            "om_lipids": om_lipids, "im_lipids": im_lipids,
            "lipid_beads": len(om) + len(im), "pg_beads": len(pg),
            "dna_beads": len(dna), "ions": len(ions),
        },
        "beads": (pack(om, "outer_membrane", "measured")
                  + pack(pg, "peptidoglycan", "measured")
                  + pack(im, "inner_membrane", "measured")
                  + pack(dna, "plasmid", "model")
                  + pack(ions, "calcium", "model")),
    }

    json.dump(scene, open(OUT, "w"), separators=(",", ":"))
    total = len(scene["beads"])
    print(f"envelope: OM {om_lipids:,} lipids, IM {im_lipids:,} lipids, "
          f"{len(om) + len(im):,} lipid beads, {len(pg):,} PG beads")
    print(f"plasmid: {n_bp} bp, sigma {sigma}, dLk {d_lk:.1f}, "
          f"writhe wanted {target_wr:.1f}, measured {wr:.1f}, {len(dna)} beads, extent {dna_extent:.0f} A")
    print(f"ions: {len(ions)}")
    print(f"total {total:,} spheres  (all-atom would be ~{(om_lipids + im_lipids) * 125 / 1e6:.1f} M atoms)")
    print(f"layers: OM {om_center:.0f}  PG {pg_center:.0f}  IM {im_center:.0f} A; outer face {om_outer_face:.0f}")
    print("wrote", os.path.relpath(OUT))


if __name__ == "__main__":
    main()
