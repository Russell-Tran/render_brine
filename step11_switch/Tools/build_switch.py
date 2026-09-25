"""Builds Resources/switch.json: the geometry of AraC's light switch.

The mechanism (Schleif, FEMS Microbiol Rev 34:779, 2010; Lobell & Schleif,
Science 250:528, 1990):

  OFF  the AraC dimer holds araO2 and araI1 at once. They are ~210 bp apart,
       so holding both forces the DNA between them into a loop, and the loop
       keeps RNA polymerase off the pBAD promoter.
  ON   L-arabinose binds a pocket in AraC's sugar-binding domain. Affinity for
       the ADJACENT pair araI1+araI2 rises ~50-fold, so AraC lets go of the
       distant araO2, takes araI2, the loop opens, and pBAD is transcribed.

What is measured here and what is not
-------------------------------------
MEASURED  the sugar-binding/dimerisation domain, with L-arabinose (2ARC,
          1.5 A) and with the anti-inducer D-fucose (2AAC, 1.6 A); the
          DNA-binding domain (2K9S, NMR); B-DNA nucleotide geometry, carried
          over from step 8's 1BNA templates via step 8a; and the positions of
          araI1 and araI2, which are found in pGLO's own sequence verbatim.
MODELLED  the whole protein bridging a loop. The two domains of AraC have only
          ever been solved separately and the linker between them is flexible,
          so no structure of the complete protein on looped DNA exists. The
          path of the loop is modelled too.
NOT SUPPORTED BY THE PAIR  2ARC and 2AAC do NOT show an arm-open vs arm-closed
          difference: superposed on their cores, the N-terminal arms differ by
          0.71 A, against 0.38 A for the core itself. Fucose is an anti-inducer
          that binds the pocket and holds the arm down much as arabinose does.
          So the "no sugar, arm lifted" state is drawn as a model and labelled
          as one; it is not read off a crystal.

Run: python3 Tools/build_switch.py      (needs numpy; reuses step 8a's pGLO data)
"""

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "Resources")
PGLO = os.path.join(HERE, "..", "..", "step8a_plasmid", "Resources", "pglo.json")
OUT = os.path.join(RES, "switch.json")

# B-DNA, as step 8a used: solution values, not the tighter crystal helix.
RISE = 3.4          # A per base pair
TWIST = 360 / 10.5  # degrees per base pair

# Published AraC half-sites from the E. coli ara regulatory region. These are
# searched for in pGLO's own sequence rather than assumed to be at the native
# coordinates, because pGLO is an engineered construct.
HALF_SITES = {
    "araI1": "TAGCATTTTTATCCATA",
    "araI2": "GGATCCTACCTGACGCT",
}
MINUS35 = "CTGACG"
MINUS10 = "TACTGT"
# araO2's centre is published at -280 relative to the transcription start
# (Schleif 2010). It is located here by that offset and then checked against
# the 210 bp araO2-araI1 spacing the looping literature reports.
ARAO2_OFFSET = -280
ARAO2_LEN = 17

VDW = {"C": 1.70, "N": 1.55, "O": 1.52, "P": 1.80, "S": 1.80, "SE": 1.90, "H": 1.20}


# ------------------------------------------------------------- pGLO sites

def map_sites(seq):
    """Where araO2, araI1 and araI2 fall in pGLO. 1-based inclusive."""
    sites = {}
    for name, motif in HALF_SITES.items():
        i = seq.find(motif)
        assert i >= 0, f"{name} not found in pGLO"
        assert seq.find(motif, i + 1) < 0, f"{name} is not unique in pGLO"
        sites[name] = (i + 1, i + len(motif))
    m35 = seq.find(MINUS35) + 1
    m10 = seq.find(MINUS10) + 1
    # Back-calculate +1 from araI1's published span (-78..-56); araI2's
    # published span (-51..-35) is then a check on it.
    plus1 = sites["araI1"][0] + 78
    check = sites["araI2"][0] + 51
    assert abs(plus1 - check) <= 3, f"the two half-sites disagree on +1: {plus1} vs {check}"
    centre = plus1 + ARAO2_OFFSET
    sites["araO2"] = (centre - ARAO2_LEN // 2, centre + ARAO2_LEN // 2)
    spacing = (sites["araI1"][0] + sites["araI1"][1]) // 2 - centre
    return sites, plus1, m35, m10, spacing


# ------------------------------------------------------------- structures

def read_pdb(path, model_one=False, keep_hetatm=None):
    """Heavy atoms as a list of dicts. `keep_hetatm` is a set of residue names."""
    out, in_model = [], True
    for line in open(path):
        if line.startswith("MODEL"):
            in_model = line.split()[1] == "1"
        if line.startswith("ENDMDL") and model_one:
            break
        rec = line[:6].strip()
        if rec == "HETATM" and keep_hetatm and line[17:20].strip() in keep_hetatm:
            pass
        elif rec != "ATOM" or not in_model:
            continue
        el = (line[76:78].strip() or line[12:16].strip()[0]).upper()
        if el == "H":
            continue
        alt = line[16]
        if alt not in (" ", "A"):
            continue
        out.append({"el": el, "chain": line[21], "res": int(line[22:26]),
                    "resn": line[17:20].strip(), "name": line[12:16].strip(),
                    "pos": [float(line[30:38]), float(line[38:46]), float(line[46:54])]})
    return out


def kabsch(moving, fixed):
    mc, fc = moving.mean(0), fixed.mean(0)
    u, _, vt = np.linalg.svd((moving - mc).T @ (fixed - fc))
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1, 1, d]) @ u.T
    return r, fc - r @ mc


def arm_comparison(arc, aac):
    """How much the N-terminal arm really differs between the two structures."""
    def ca(atoms):
        return {(a["chain"], a["res"]): np.array(a["pos"]) for a in atoms if a["name"] == "CA"}
    A, B = ca(arc), ca(aac)
    common = [k for k in A if k in B]
    core = [k for k in common if 30 <= k[1] <= 160]
    r, t = kabsch(np.array([A[k] for k in core]), np.array([B[k] for k in core]))
    def rms(lo, hi):
        sel = [k for k in common if lo <= k[1] <= hi]
        if len(sel) < 3:
            return None
        p = np.array([r @ A[k] + t for k in sel])
        q = np.array([B[k] for k in sel])
        return float(np.sqrt(((p - q) ** 2).sum(1).mean()))
    return {"arm_7_18": rms(7, 18), "core_30_160": rms(30, 160), "n_common": len(common)}


# ------------------------------------------------------------- DNA on a curve

def centreline(n_bp, loop_from, loop_to, span_turns):
    """A planar path: a straight lead-in, then the loop region bent through
    `span_turns` x 2pi, then a straight lead-out.

    Only the stretch between the two held sites bends, which is what the
    mechanism actually says: AraC grips araO2 and araI1 and the 210 bp between
    them is forced round. span=1 brings those two sites together and closes the
    loop; less than 1 lets it open. The DNA's arc length never changes, so the
    same 290 base pairs are present in every state and only the path differs.
    """
    kappa = np.zeros(n_bp)
    inside = max(loop_to - loop_from, 1)
    kappa[loop_from:loop_to] = span_turns * 2 * np.pi / inside
    pts = np.zeros((n_bp, 3))
    tangents = np.zeros((n_bp, 3))
    angle = 0.0
    p = np.zeros(3)
    for i in range(n_bp):
        tangents[i] = [np.cos(angle), np.sin(angle), 0.0]
        pts[i] = p
        p = p + tangents[i] * RISE
        angle += kappa[i]
    return pts, tangents


def frames(pts, tangents):
    """Parallel-transport frames along the curve, so the helix does not spin
    spuriously where the path bends."""
    n = len(pts)
    normals = np.zeros((n, 3))
    ref = np.array([0.0, 0.0, 1.0])
    normals[0] = ref - tangents[0] * np.dot(ref, tangents[0])
    normals[0] /= np.linalg.norm(normals[0])
    for i in range(1, n):
        v = normals[i - 1] - tangents[i] * np.dot(normals[i - 1], tangents[i])
        nv = np.linalg.norm(v)
        normals[i] = v / nv if nv > 1e-9 else normals[i - 1]
    binormals = np.cross(tangents, normals)
    return normals, binormals


def place_dna(seq, pts, tangents, twist0=0.0):
    """Stamps base-pair templates along the curve.

    Each template is one real crystal base pair; it is placed as a RIGID BODY
    (as step 8a did) so every bond length and hydrogen bond inside it survives
    the bend untouched. Only the path between base pairs is modelled.
    """
    normals, binormals = frames(pts, tangents)
    out_atoms, out_bonds, out_hbonds = [], [], []
    base = 0
    for i, ch in enumerate(seq):
        kind = {"A": "AT", "T": "TA", "G": "GC", "C": "CG"}[ch]
        t = TEMPLATES[kind]
        ang = np.radians(twist0 + TWIST * i)
        # The template's own axes: x along the helix, y/z across it.
        e1 = tangents[i]
        e2 = normals[i] * np.cos(ang) + binormals[i] * np.sin(ang)
        e3 = np.cross(e1, e2)
        rot = np.stack([e1, e2, e3], axis=1)
        for a in t["atoms"]:
            p = rot @ np.array(a["pos"]) + pts[i]
            out_atoms.append({"el": a["el"], "strand": a["strand"], "name": a["name"],
                              "bp": i, "pos": [round(float(v), 3) for v in p]})
        for b in t["bonds"]:
            out_bonds.append([base + b[0], base + b[1]])
        for hb in t["hbonds"]:
            out_hbonds.append([base + hb[0], base + hb[1]])
        base += len(t["atoms"])
    # Join consecutive base pairs: O3' of one to P of the next, per strand.
    by_bp = {}
    for i, a in enumerate(out_atoms):
        by_bp[(a["bp"], a["strand"], a["name"])] = i
    for i in range(len(seq) - 1):
        for strand, (x, y) in ((0, ("O3'", "P")), (1, ("P", "O3'"))):
            u = by_bp.get((i, strand, x)); v = by_bp.get((i + 1, strand, y))
            if u is not None and v is not None:
                out_bonds.append([u, v])
    return out_atoms, out_bonds, out_hbonds


# ------------------------------------------------------------- main

def main():
    pglo = json.load(open(PGLO))
    seq = pglo["sequence"]
    global TEMPLATES
    TEMPLATES = pglo["templates"]

    sites, plus1, m35, m10, spacing = map_sites(seq)
    print(f"araI1 {sites['araI1']}   araI2 {sites['araI2']}   araO2 {sites['araO2']}")
    print(f"-35 at {m35}, -10 at {m10}, transcription start ~{plus1}")
    print(f"araO2 centre to araI1 centre: {spacing} bp   (published 210)")
    assert 200 <= spacing <= 220, f"spacing {spacing} is nowhere near the published 210 bp"

    arc = read_pdb(os.path.join(RES, "2ARC.pdb"), keep_hetatm={"ARA"})
    aac = read_pdb(os.path.join(RES, "2AAC.pdb"), keep_hetatm={"FCB"})
    dbd = read_pdb(os.path.join(RES, "2K9S.pdb"), model_one=True)
    arms = arm_comparison([a for a in arc if a["resn"] != "ARA"],
                          [a for a in aac if a["resn"] != "FCB"])
    print(f"2ARC vs 2AAC, superposed on the core: arm {arms['arm_7_18']:.2f} A, "
          f"core {arms['core_30_160']:.2f} A over {arms['n_common']} CA")
    sugar = [a for a in arc if a["resn"] == "ARA" and a["chain"] == "A"]
    print(f"2ARC: {len([a for a in arc if a['resn'] != 'ARA'])} protein atoms, "
          f"arabinose {len(sugar)} atoms per site")
    print(f"2K9S model 1: {len(dbd)} heavy atoms")

    # The stretch of pGLO the render shows: from a little before araO2 to a
    # little after araI2.
    lo = sites["araO2"][0] - 12
    hi = sites["araI2"][1] + 26
    window = seq[lo - 1:hi]
    n_bp = len(window)
    print(f"\nDNA drawn: pGLO {lo}-{hi}, {n_bp} bp, {n_bp * RISE:.0f} A of contour length")

    # Where each half-site's middle falls in the drawn window, as a base-pair index.
    bp_of = {k: (v[0] + v[1]) // 2 - lo for k, v in sites.items()}
    print("half-site centres, as base-pair index into the window: "
          + ", ".join(f"{k} {i}" for k, i in sorted(bp_of.items(), key=lambda x: x[1])))

    loop_from, loop_to = bp_of["araO2"], bp_of["araI1"]
    # Report the two end states here, but ship only the recipe: Swift rebuilds
    # the DNA every frame from the templates, so each base pair stays a rigid
    # crystal body at every instant of the animation, not just at the ends.
    for name, span, held in (("looped", 1.0, ("araO2", "araI1")),
                             ("open", 0.80, ("araI1", "araI2"))):
        pts, tang = centreline(n_bp, loop_from, loop_to, span)
        atoms, _, _ = place_dna(window, pts, tang)
        extent = np.ptp(np.array([a["pos"] for a in atoms]), axis=0)
        gap = float(np.linalg.norm(pts[bp_of[held[1]]] - pts[bp_of[held[0]]]))
        print(f"  {name:7s} span {span:.2f} turns, extent "
              f"{extent[0]:.0f} x {extent[1]:.0f} x {extent[2]:.0f} A, {len(atoms)} atoms, "
              f"{held[0]}-{held[1]} {gap:.0f} A apart")

    # Centre each protein piece on its own centroid, so Swift can place it by
    # a frame rather than having to undo the crystal's arbitrary origin.
    def centred(atoms):
        c = np.array([a["pos"] for a in atoms]).mean(0)
        return [{"el": a["el"], "pos": [round(float(v), 3) for v in np.array(a["pos"]) - c]}
                for a in atoms], c

    core_atoms, core_c = centred([a for a in arc if a["resn"] != "ARA"])
    dbd_atoms, _ = centred(dbd)
    sugar_atoms = [{"el": a["el"], "pos": [round(float(v - c), 3) for v, c in zip(a["pos"], core_c)]}
                   for a in sugar]
    print(f"\nprotein pieces: core {len(core_atoms)} atoms, DBD {len(dbd_atoms)}, sugar {len(sugar_atoms)}")

    data = {
        "source": "pGLO sequence and base-pair templates from step 8a; AraC from PDB 2ARC, 2AAC, 2K9S",
        "sites": {k: list(v) for k, v in sites.items()},
        "bp_of": bp_of,
        "plus1": plus1, "minus35": m35, "minus10": m10, "spacing_bp": spacing,
        "window": [lo, hi], "sequence": window, "n_bp": n_bp,
        "loop_from": loop_from, "loop_to": loop_to,
        "rise": RISE, "twist": TWIST,
        "arm_comparison": arms,
        "templates": {k: {"atoms": v["atoms"], "bonds": v["bonds"], "hbonds": v["hbonds"]}
                      for k, v in TEMPLATES.items()},
        "protein": {"core": core_atoms, "dbd": dbd_atoms, "sugar": sugar_atoms},
    }
    json.dump(data, open(OUT, "w"))
    print(f"\nwrote {os.path.relpath(OUT)}  "
          f"({os.path.getsize(OUT) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
