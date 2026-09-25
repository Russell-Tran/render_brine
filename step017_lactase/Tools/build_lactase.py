"""Builds Resources/scene.json: E. coli (lacZ) beta-galactosidase cutting
lactose, from the series of structures Juers and colleagues solved along the
reaction coordinate.

Why E. coli and not a human enzyme
----------------------------------
The obvious subject for "lactose being cut" is human lactase — lactase-phlorizin
hydrolase, UniProt P09848, the brush-border enzyme whose loss is lactose
intolerance. There is no structure of it. Asking the PDB for every entry
carrying P09848 returns nothing at all: no crystal structure, no cryo-EM map,
no NMR model. `check_no_human_lactase()` below runs that query and stops the
build if it ever starts returning hits, because on the day it does, this step
is rendering the wrong molecule.

So this renders E. coli beta-galactosidase (LacZ, P00722), and the caption bar
says whose enzyme it is. Both are family GH2 retaining beta-galactosidases and
both run the same two-step mechanism, but they are different proteins and the
picture must not pretend otherwise.

LacZ is in any case the better subject, because somebody deliberately went and
solved its whole catalytic cycle:

  1JYN  E537Q + LACTOSE                       1.80 A   substrate, shallow site
  1JZ5  + D-galactonolactone                  1.80 A   transition-state mimic
  1JZ2  TRAPPED 2-F-GALACTOSYL-ENZYME         2.10 A   the covalent intermediate
  1JZ7  + GALACTOSE                           1.50 A   product in the site
  1JYV  E537Q + ONPG                          1.75 A   the chromogenic substrate
  1JYW  E537Q + PNPG                          1.55 A   the other one
  1JZ8  E537Q + ALLOLACTOSE                   1.50 A   the inducer

all from Juers, Heightman, Vasella, McCarter, Mackenzie, Withers & Matthews,
"A structural view of the action of Escherichia coli (lacZ) beta-galactosidase",
Biochemistry 40:14781 (2001).

The middle of this cycle is not interpolated. 1JZ2 is the covalent
alpha-galactosyl-enzyme itself, trapped with a 2-fluoro sugar and refined at
2.1 A. Four states of the cycle are measured; only the short paths between
them are drawn.

What this script measures rather than assumes
---------------------------------------------
  * the noise floor, from the four crystallographically independent copies in
    each entry — they were refined against the same data, so how much THEY
    differ is what "no difference" looks like (`measure_motion`);
  * the anomeric configuration at the galactosyl C1 in each state, straight
    from the coordinates, by the sign of a triple product (`anomeric_sign`);
  * whether the transition-state mimic is flat at C1, which is the whole claim
    of a transition-state mimic (`lactone_planarity`);
  * what holds the magnesium (`metal_ligands`);
  * where the water that does the second half-cycle sits, and at what angle to
    the bond it attacks (`catalytic_water`);
  * the way out of the buried active site, by flooding the free space with a
    probe and walking to the surface (`escape_channel`).

Run: python3 Tools/build_lactase.py
"""

import itertools
import json
import os
import sys
import urllib.error
import urllib.request
from collections import deque

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "Resources")
OUT = os.path.join(RES, "scene.json")

# The coordinate files are ~4 MB each and are not committed; they are fetched
# on demand and .gitignore'd, as step 12 does with its four entries.
ENTRIES = ["1JYN", "1JZ5", "1JZ2", "1JZ7", "1JYV", "1JYW", "1JZ8"]

# The structure whose protein is drawn, and whose frame everything is put into:
# the highest-resolution member of the series.
REF = "1JZ7"

HUMAN_LACTASE = "P09848"     # lactase-phlorizin hydrolase, the enzyme we cannot draw
LACZ = "P00722"              # E. coli beta-galactosidase, the one we can

# van der Waals radii (A), Bondi, J. Phys. Chem. 68:441 (1964) — every step
# since 9 has used these.
VDW = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80, "SE": 1.90,
       "F": 1.47, "MG": 1.73, "NA": 2.27, "H": 1.20}

# Covalent radii (A), Cordero et al., Dalton Trans. (2008) 2832. The renderer
# uses these to decide what counts as a bond, which is how the covalent
# intermediate is detected rather than declared; they are repeated in
# Sources/Lactase.swift, which is where the bond test actually runs.
COVALENT = {"C": 0.76, "N": 0.71, "O": 0.66, "S": 1.05, "F": 0.57, "MG": 1.41}

# What is in the crystal but is not the enzyme. The bound sugars are animated
# by this render instead of being drawn where the crystal left them, and DMSO
# and bis-tris are cryoprotectant and buffer. The magnesium and sodium stay:
# they are part of the protein's structure, and one of them is in the
# mechanism. CSO 247 is an oxidised cysteine of the chain itself, so it stays.
NOT_THE_ENZYME = {"GAL", "BGC", "GLC", "2FG", "149", "145", "147", "IPT", "2DG",
                  "GTZ", "DMS", "BTB", "HOH"}

# The residues that line the site, as Juers 2001 names them. Used only to say
# which atoms count as "the active site" when the motion is measured.
SITE_RESIDUES = [416, 418, 460, 461, 502, 503, 537, 538, 539, 540, 541, 542,
                 543, 601, 602, 796, 797, 798, 799, 800, 801, 802, 803, 804,
                 805, 806]

# The cast drawn ball-and-stick inside the porthole. Everything else in the
# tetramer stays space-filling.
CAST_RESIDUES = [416, 418, 460, 461, 502, 503, 537]

# Which ligand copy is the one in chain A's active site. Worked out by distance
# in `find_ligand`, not hard-coded; these are only the component names to look
# for, and which structure they belong to.
LIGANDS = {
    "1JYN": ("GAL", "BGC"),      # lactose: beta-D-Gal-(1->4)-D-Glc, two residues
    "1JZ5": ("149",),            # D-galactonolactone
    "1JZ2": ("2FG",),            # 2-deoxy-2-fluoro-galactosyl, covalent
    "1JZ7": ("GAL",),            # galactose
    "1JYV": ("145",),            # ONPG
    "1JYW": ("147",),            # PNPG
    "1JZ8": ("GAL", "BGC"),      # allolactose
}

RING = ["C1", "C2", "C3", "C4", "C5", "O5"]


# ----------------------------------------------------------------- fetching

def fetch(entry):
    path = os.path.join(RES, entry + ".cif")
    if os.path.exists(path) and os.path.getsize(path) > 100_000:
        return path
    url = "https://files.rcsb.org/download/%s.cif" % entry
    print("  fetching %s from RCSB..." % entry)
    os.makedirs(RES, exist_ok=True)
    with urllib.request.urlopen(url) as r, open(path, "wb") as f:
        f.write(r.read())
    return path


def rcsb_entries_for(accession):
    """Every PDB entry whose polymer maps to this UniProt accession."""
    query = {
        "query": {"type": "terminal", "service": "text", "parameters": {
            "attribute": "rcsb_polymer_entity_container_identifiers"
                         ".reference_sequence_identifiers.database_accession",
            "operator": "exact_match", "value": accession}},
        "return_type": "entry",
        "request_options": {"paginate": {"start": 0, "rows": 200},
                            "results_verbosity": "compact"},
    }
    req = urllib.request.Request(
        "https://search.rcsb.org/rcsbsearch/v2/query",
        data=json.dumps(query).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            if r.status == 204:              # the API's way of saying "none"
                return []
            return json.loads(r.read()).get("result_set", [])
    except urllib.error.HTTPError as e:
        if e.code == 204:
            return []
        raise


def check_no_human_lactase():
    """The premise of the substitution, checked against the PDB rather than
    remembered. If this ever finds something, the right thing to do is stop and
    render the human enzyme."""
    try:
        hits = rcsb_entries_for(HUMAN_LACTASE)
    except Exception as e:                   # offline: say so, do not pretend
        print("  (could not reach RCSB to re-check %s: %s)" % (HUMAN_LACTASE, e))
        return None
    if hits:
        raise SystemExit(
            "STOP: the PDB now has %d structure(s) for human lactase-phlorizin "
            "hydrolase (%s): %s.\nThis step exists because there were none. "
            "Render the human enzyme instead of substituting LacZ."
            % (len(hits), HUMAN_LACTASE, ", ".join(hits)))
    print("  RCSB: 0 entries for UniProt %s (human lactase-phlorizin hydrolase)"
          % HUMAN_LACTASE)
    return 0


# ------------------------------------------------------------------ parsing

def read_atom_site(path):
    """The _atom_site loop as (column index, rows). mmCIF because these entries
    are deposited in it and step 12 already reads this format."""
    with open(path) as f:
        lines = f.read().split("\n")
    i = 0
    while i < len(lines):
        if lines[i].startswith("_atom_site."):
            hdr = []
            while i < len(lines) and lines[i].startswith("_atom_site."):
                hdr.append(lines[i].strip())
                i += 1
            col = {h: k for k, h in enumerate(hdr)}
            rows = []
            while i < len(lines) and not lines[i].startswith("#"):
                p = lines[i].split()
                if len(p) == len(hdr):
                    rows.append(p)
                i += 1
            return col, rows
        i += 1
    raise SystemExit("no _atom_site loop in " + path)


def read_structure(entry, waters=False):
    col, rows = read_atom_site(fetch(entry))

    def g(p, k):
        return p[col["_atom_site." + k]]

    out = []
    for p in rows:
        if g(p, "label_alt_id") not in (".", "A"):
            continue                          # one conformer only, as step 9 did
        comp = g(p, "auth_comp_id")
        if comp == "HOH" and not waters:
            continue
        out.append({
            "group": g(p, "group_PDB"),
            "el": g(p, "type_symbol").upper(),
            "name": g(p, "auth_atom_id").strip('"'),
            "comp": comp,
            "chain": g(p, "auth_asym_id"),
            "seq": int(g(p, "auth_seq_id")),
            "pos": np.array([float(g(p, "Cartn_x")), float(g(p, "Cartn_y")),
                             float(g(p, "Cartn_z"))]),
            "b": float(g(p, "B_iso_or_equiv")),
        })
    return out


# ------------------------------------------------------------ superposition

def kabsch(P, Q):
    """The rotation and translation taking P onto Q, least squares.
    Kabsch, Acta Cryst. A32:922 (1976).

    (errstate: numpy 2.0 on Accelerate raises spurious divide/overflow warnings
    out of matmul on these shapes. The arithmetic is checked by the RMSDs it
    produces, which are finite and reproducible.)"""
    with np.errstate(all="ignore"):
        pc, qc = P.mean(0), Q.mean(0)
        H = (P - pc).T @ (Q - qc)
        U, _, Vt = np.linalg.svd(H)
        D = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(Vt.T @ U.T)))])
        R = Vt.T @ D @ U.T
        assert np.all(np.isfinite(R)), "superposition did not converge"
        return R, qc - R @ pc


def alpha_carbons(atoms, chain):
    return {a["seq"]: a["pos"] for a in atoms
            if a["name"] == "CA" and a["chain"] == chain and a["group"] == "ATOM"}


def superpose(atoms, ref_atoms, chain="A", ref_chain="A"):
    a, b = alpha_carbons(atoms, chain), alpha_carbons(ref_atoms, ref_chain)
    common = sorted(set(a) & set(b))
    P = np.array([a[s] for s in common])
    Q = np.array([b[s] for s in common])
    R, t = kabsch(P, Q)
    with np.errstate(all="ignore"):
        d = np.linalg.norm(P @ R.T + t - Q, axis=1)
    return R, t, float(np.sqrt((d ** 2).mean())), len(common)


def rmsd_over(a, b):
    keys = sorted(set(a) & set(b))
    d = np.array([np.linalg.norm(a[k] - b[k]) for k in keys])
    return float(np.sqrt((d ** 2).mean())), float(d.max()), len(keys)


def side_chain_atoms(atoms, chain, seqs, R=None, t=None):
    out = {}
    for a in atoms:
        if a["chain"] == chain and a["seq"] in seqs and a["group"] == "ATOM":
            p = a["pos"] if R is None else R @ a["pos"] + t
            out[(a["seq"], a["name"])] = p
    return out


# --------------------------------------------------- step 11's discipline
#
# Step 11 measured the displacement between its two structures before animating
# anything, found 0.71 A against a 0.38 A noise floor, and refused to animate
# the change it had planned. The same check has to run here before a frame is
# drawn, and the noise floor has to come from the data rather than from a
# remembered number.
#
# The honest floor is right there in each entry: LacZ crystallises with all
# four subunits in the asymmetric unit, so each entry contains four copies
# refined independently against one set of data. Whatever those four copies
# disagree about is not a conformational change.

def measure_motion(structures):
    noise_ca, noise_site = [], []
    rows = []
    for e in ["1JYN", "1JZ5", "1JZ2", "1JZ7"]:
        A = structures[e]
        for c1, c2 in itertools.combinations("ABCD", 2):
            R, t, r, n = superpose(A, A, chain=c1, ref_chain=c2)
            s1 = side_chain_atoms(A, c1, SITE_RESIDUES, R, t)
            s2 = side_chain_atoms(A, c2, SITE_RESIDUES)
            rs, mx, ns = rmsd_over(s1, s2)
            noise_ca.append(r)
            noise_site.append(rs)
            rows.append({"kind": "noise", "entry": e, "pair": c1 + c2,
                         "ca": round(r, 3), "site": round(rs, 3)})

    signal = []
    for e in ["1JYN", "1JZ5", "1JZ2"]:
        R, t, r, n = superpose(structures[e], structures[REF])
        s1 = side_chain_atoms(structures[e], "A", SITE_RESIDUES, R, t)
        s2 = side_chain_atoms(structures[REF], "A", SITE_RESIDUES)
        rs, mx, ns = rmsd_over(s1, s2)
        signal.append({"kind": "signal", "entry": e, "vs": REF,
                       "ca": round(r, 3), "site": round(rs, 3), "siteMax": round(mx, 2),
                       "nCA": n, "nSite": ns})

    floor_ca = float(np.mean(noise_ca))
    floor_site = float(np.mean(noise_site))
    worst_ca = max(s["ca"] for s in signal)
    worst_site = max(s["site"] for s in signal)
    return {
        "noiseCA": round(floor_ca, 3), "noiseCASd": round(float(np.std(noise_ca)), 3),
        "noiseSite": round(floor_site, 3), "noiseSiteSd": round(float(np.std(noise_site)), 3),
        "signal": signal, "pairs": rows,
        "proteinMovesAboveNoise": bool(worst_ca > floor_ca),
        "siteMovesAboveNoise": bool(worst_site > floor_site),
        "worstCA": round(worst_ca, 3), "worstSite": round(worst_site, 3),
        "verdict": ("the protein does not measurably move between these states: "
                    "every state-to-state difference is smaller than the difference "
                    "between the four independently refined copies inside a single "
                    "entry. Only the ligand chemistry is animated."
                    if worst_ca <= floor_ca and worst_site <= floor_site else
                    "at least one state-to-state difference exceeds the "
                    "chain-to-chain noise; see the table."),
    }


# ------------------------------------------------------------- the ligands

def find_ligand(atoms, comps, anchor):
    """The copy of `comps` closest to `anchor` — the one in chain A's site.
    Returns {atom name: position} merged across the components, tagged."""
    groups = {}
    for a in atoms:
        if a["group"] == "HETATM" and a["comp"] in comps:
            groups.setdefault((a["chain"], a["comp"], a["seq"]), []).append(a)
    best, bestd = None, 1e30
    for key, g in groups.items():
        d = min(float(np.linalg.norm(a["pos"] - anchor)) for a in g)
        if d < bestd:
            best, bestd = key, d
    if best is None:
        raise SystemExit("no %s found" % (comps,))
    # For lactose the galactosyl is the one near the anchor; the glucose is
    # whichever copy of the partner component is bonded to it.
    picked = {best: groups[best]}
    if len(comps) > 1:
        gal = groups[best]
        for key, g in groups.items():
            if key == best or key[1] == best[1]:
                continue
            d = min(float(np.linalg.norm(a["pos"] - b["pos"])) for a in g for b in gal)
            if d < 1.8:                       # covalently joined: the same sugar
                picked[key] = g
    return picked, bestd


def anomeric_sign(ring, exo):
    """The sign that says which face of the ring the exocyclic group at C1 is
    on, from coordinates alone:

        (exo - C1) . [ (O5 - C1) x (C2 - C1) ]

    The ring atoms O5, C1, C2 fix a frame; the exocyclic substituent is either
    above or below it. Nothing here depends on an atom's NAME, which matters:
    the PDB chemical component 2FG is called "2-deoxy-2-fluoro-BETA-D-
    galactopyranose", but in 1JZ2 it is bonded to Glu537 and the coordinates
    say alpha. The coordinates win."""
    c1, o5, c2 = ring["C1"], ring["O5"], ring["C2"]
    return float(np.dot(exo - c1, np.cross(o5 - c1, c2 - c1)))


def lactone_planarity(ring):
    """How far C1's exocyclic oxygen lies out of the O5-C1-C2 plane. A normal
    sugar is tetrahedral at C1 and this is ~0.9 A; a 1,5-lactone is sp2 and it
    is ~0. Being flat is the entire claim of a transition-state mimic, so it is
    measured rather than asserted."""
    c1, o5, c2, o1 = ring["C1"], ring["O5"], ring["C2"], ring["O1"]
    n = np.cross(o5 - c1, c2 - c1)
    n = n / np.linalg.norm(n)
    return float(abs(np.dot(o1 - c1, n))), float(np.linalg.norm(o1 - c1))


# ------------------------------------------------------------- the channel

def escape_channel(heavy, radii, start, outward_hint, step=1.2, reach=46.0):
    """The way out of a buried site, found rather than drawn: flood the space
    a water-sized probe can occupy, starting at the site, and walk to the first
    point that is out in bulk solvent.

    `heavy` are the protein's heavy atoms, `radii` their van der Waals radii.
    A grid point is free if no atom's surface plus a 1.4 A probe reaches it.
    Breadth-first from `start` gives the shortest such path, which is the
    channel; `outward_hint` only breaks ties between equally good exits.
    """
    probe = 1.4                                  # a water, Lee & Richards (1971)
    lo = start - reach
    n = int(2 * reach / step) + 1

    # Only the atoms that could matter, in a coarse hash for the distance test.
    near = [(p, r) for p, r in zip(heavy, radii)
            if np.all(np.abs(p - start) < reach + 6)]
    P = np.array([p for p, _ in near])
    Rr = np.array([r for _, r in near]) + probe
    cell = 6.0
    buckets = {}
    for i, p in enumerate(P):
        buckets.setdefault(tuple((p // cell).astype(int)), []).append(i)

    def free(p):
        b = (p // cell).astype(int)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for i in buckets.get((b[0] + dx, b[1] + dy, b[2] + dz), ()):
                        if np.dot(p - P[i], p - P[i]) < Rr[i] * Rr[i]:
                            return False
        return True

    def to_grid(p):
        return tuple(np.clip(((p - lo) / step).round().astype(int), 0, n - 1))

    def to_world(c):
        return lo + np.array(c, dtype=float) * step

    s = to_grid(start)
    # The site itself is full of the ligand we removed, so start from the
    # nearest free point instead of insisting the centre is free.
    if not free(to_world(s)):
        best = None
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                for dz in range(-3, 4):
                    c = (s[0] + dx, s[1] + dy, s[2] + dz)
                    if min(c) < 0 or max(c) >= n:
                        continue
                    w = to_world(c)
                    if free(w):
                        d = float(np.linalg.norm(w - start))
                        if best is None or d < best[0]:
                            best = (d, c)
        if best is None:
            return None
        s = best[1]

    seen = {s: None}
    q = deque([s])
    exit_cell = None
    steps = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    while q:
        c = q.popleft()
        w = to_world(c)
        if np.linalg.norm(w - start) > reach - step * 2:
            exit_cell = c
            break
        for d in steps:
            nc = (c[0] + d[0], c[1] + d[1], c[2] + d[2])
            if nc in seen or min(nc) < 0 or max(nc) >= n:
                continue
            if not free(to_world(nc)):
                continue
            seen[nc] = c
            q.append(nc)
    if exit_cell is None:
        return None
    path = []
    c = exit_cell
    while c is not None:
        path.append(to_world(c))
        c = seen[c]
    path.reverse()
    return np.array(path)


def smooth_path(path, keep=14):
    """Thin a lattice path down to a few waypoints and round the corners, so a
    ligand slides along it instead of stepping through a staircase."""
    if path is None or len(path) < 3:
        return path
    idx = np.linspace(0, len(path) - 1, min(keep, len(path))).round().astype(int)
    pts = path[idx]
    for _ in range(24):                      # Laplacian smoothing, ends pinned
        q = pts.copy()
        q[1:-1] = 0.5 * pts[1:-1] + 0.25 * (pts[:-2] + pts[2:])
        pts = q
    return pts


# -------------------------------------------------------------------- main

def main():
    print("checking the premise:")
    check_no_human_lactase()
    lacz = rcsb_entries_for(LACZ)
    if lacz:
        print("  RCSB: %d entries for UniProt %s (E. coli lacZ beta-galactosidase)"
              % (len(lacz), LACZ))

    print("reading the series:")
    structures = {e: read_structure(e) for e in ENTRIES}
    ref = structures[REF]
    for e in ENTRIES:
        n537 = [a["comp"] for a in structures[e]
                if a["chain"] == "A" and a["seq"] == 537 and a["name"] == "CA"]
        print("  %s  %6d heavy atoms   residue 537 is %s"
              % (e, len(structures[e]), n537[0] if n537 else "?"))

    # --- step 11's check, before anything is drawn -----------------------
    print("\nmeasuring the motion before animating any of it:")
    motion = measure_motion(structures)
    print("  noise floor (four independent copies, same data):")
    print("      whole chain  %.3f +- %.3f A" % (motion["noiseCA"], motion["noiseCASd"]))
    print("      active site  %.3f +- %.3f A" % (motion["noiseSite"], motion["noiseSiteSd"]))
    print("  signal (state against state, chain A):")
    for s in motion["signal"]:
        print("      %s vs %s   whole chain %.3f A   active site %.3f A (max %.2f)"
              % (s["vs"], s["entry"], s["ca"], s["site"], s["siteMax"]))
    print("  -> " + motion["verdict"])

    # --- put every state into the reference frame ------------------------
    frames = {}
    for e in ENTRIES:
        R, t, r, n = superpose(structures[e], ref)
        frames[e] = (R, t)

    def place(e, p):
        R, t = frames[e]
        return R @ p + t

    anchor = [a["pos"] for a in ref
              if a["chain"] == "A" and a["seq"] == 537 and a["name"] == "OE2"][0]

    # Two states have their sugar in CONTACT with Glu537 — 1JZ5 approaching it
    # and 1JZ2 bonded to it — and for those the whole-chain frame is not good
    # enough. The protein is frozen at 1JZ7, and 1JZ7's Glu537 carboxylate sits
    # about 0.3 A from 1JZ2's. That is far below the noise floor, so the data
    # cannot say which is right; but a 1.45 A bond cannot absorb 0.3 A, and
    # drawing it at 1.76 A would be drawing a bond that is not a bond.
    #
    # So for those two states the placement is corrected by a pure TRANSLATION
    # that puts their Glu537 OE2 exactly on 1JZ7's. Nothing in the protein
    # moves; the sugar and its water are placed in the frame of the residue the
    # sugar is bonded to, and every distance INSIDE the complex is untouched,
    # because a translation cannot change one.
    #
    # Fitting the side chain by rotation as well was tried first and is worse:
    # five atoms are not enough to fix an orientation, so it left the bond at
    # 1.60 A and swung the attacking water 1.3 A out of place. The numbers the
    # build prints below are that comparison.
    NUCLEOPHILE_FRAME = {"1JZ5", "1JZ2"}
    draw_frames = dict(frames)
    for e in NUCLEOPHILE_FRAME:
        own = [a["pos"] for a in structures[e]
               if a["chain"] == "A" and a["seq"] == 537 and a["name"] == "OE2"][0]
        R, t = frames[e]
        draw_frames[e] = (R, t + (anchor - (R @ own + t)))

    def draw(e, p):
        R, t = draw_frames[e]
        return R @ p + t

    # `ligands` is in the whole-chain frame and is what every MEASUREMENT below
    # uses. `drawn` is the same ligands placed for the render, and differs from
    # it only for the two states in contact with Glu537.
    ligands, drawn = {}, {}
    for e in ENTRIES:
        Ra, ta = frames[e]
        local_anchor = np.linalg.inv(Ra) @ (anchor - ta)
        picked, d = find_ligand(structures[e], LIGANDS[e], local_anchor)
        ligands[e], drawn[e] = {}, {}
        for (chain, comp, seq), g in picked.items():
            for a in g:
                key = "%s:%s" % (comp, a["name"])
                ligands[e][key] = place(e, a["pos"])
                drawn[e][key] = draw(e, a["pos"])

    # --- the anomeric centre, measured in every state --------------------
    print("\nthe anomeric carbon, from the coordinates:")

    def ring_of(e, comp):
        return {n.split(":")[1]: p for n, p in ligands[e].items() if n.startswith(comp + ":")}

    gal_n = ring_of("1JYN", "GAL")
    glc_n = ring_of("1JYN", "BGC")
    gal_2 = ring_of("1JZ2", "2FG")
    gal_7 = ring_of("1JZ7", "GAL")
    lac_5 = ring_of("1JZ5", "149")

    e537_oe2 = np.array([place("1JZ2", a["pos"]) for a in structures["1JZ2"]
                         if a["chain"] == "A" and a["seq"] == 537 and a["name"] == "OE2"][0])
    # The bond as 1JZ2 itself refined it (a rigid transform preserves the
    # distance, so this is its native length), against what the frozen protein
    # would give if the sugar were placed by the whole-chain superposition.
    bond_own_native = float(np.linalg.norm(gal_2["C1"] - e537_oe2))
    bond_chain = float(np.linalg.norm(gal_2["C1"] - anchor))

    states = [
        ("substrate", "1JYN", gal_n, glc_n["O4"], "O4 of the glucose"),
        ("covalent", "1JZ2", gal_2, e537_oe2, "OE2 of Glu537"),
        ("product", "1JZ7", gal_7, gal_7["O1"], "O1, which was the water"),
    ]
    anomeric = []
    for label, e, ring, exo, what in states:
        v = anomeric_sign(ring, exo)
        d = float(np.linalg.norm(exo - ring["C1"]))
        cfg = "beta" if v > 0 else "alpha"
        anomeric.append({"state": label, "entry": e, "signedVolume": round(v, 3),
                         "config": cfg, "exo": what, "bond": round(d, 3)})
        print("  %-10s %s  C1-%-22s %.2f A   signed volume %+7.3f  -> %s"
              % (label, e, what, d, v, cfg))
    inverts_once = anomeric[0]["config"] != anomeric[1]["config"]
    inverts_twice = anomeric[1]["config"] != anomeric[2]["config"]
    retained = anomeric[0]["config"] == anomeric[2]["config"]
    print("  -> inverts at glycosylation: %s; inverts again at deglycosylation: %s; "
          "net retention: %s" % (inverts_once, inverts_twice, retained))
    if not (inverts_once and inverts_twice and retained):
        raise SystemExit("the deposited coordinates do not show two inversions and a "
                         "retention; stop and find out why before drawing anything")

    flat, carbonyl = lactone_planarity(lac_5)
    tetra, single = lactone_planarity(gal_7)
    print("  transition-state mimic 1JZ5: C1=O1 %.2f A, O1 lies %.2f A out of the "
          "O5-C1-C2 plane (sp2, flat)" % (carbonyl, flat))
    print("  ordinary sugar     1JZ7: C1-O1 %.2f A, O1 lies %.2f A out of that plane "
          "(sp3, puckered)" % (single, tetra))

    # --- the magnesium, and what actually holds it -----------------------
    mg = None
    for a in ref:
        if a["comp"] == "MG" and a["chain"] == "A":
            d = float(np.linalg.norm(a["pos"] - anchor))
            if mg is None or d < mg[0]:
                mg = (d, a["pos"])
    metal = []
    for a in ref:
        if a["el"] in ("O", "N") and a["group"] == "ATOM":
            d = float(np.linalg.norm(a["pos"] - mg[1]))
            if d < 2.6:
                metal.append({"res": a["comp"], "seq": a["seq"], "atom": a["name"],
                              "d": round(d, 2)})
    metal.sort(key=lambda m: m["d"])
    print("\nthe magnesium (%.2f A from Glu537's nucleophilic oxygen) is held by:" % mg[0])
    for m in metal:
        print("   %s%d:%s at %.2f A" % (m["res"], m["seq"], m["atom"], m["d"]))

    # --- the water that does the second half-cycle -----------------------
    print("\nthe water that attacks in the second half-cycle (1JZ2):")
    w2 = [a for a in read_structure("1JZ2", waters=True) if a["comp"] == "HOH"]
    c1 = gal_2["C1"]
    axis = c1 - e537_oe2
    axis = axis / np.linalg.norm(axis)
    e461 = {n: place("1JZ2", a["pos"]) for a in structures["1JZ2"]
            for n in [a["name"]] if a["chain"] == "A" and a["seq"] == 461}
    cands = []
    for a in w2:
        p = place("1JZ2", a["pos"])
        d = float(np.linalg.norm(p - c1))
        if d < 5.0:
            cos = float(np.dot((p - c1) / d, axis))
            db = min(float(np.linalg.norm(p - e461[n])) for n in ("OE1", "OE2"))
            cands.append((-cos * (1.0 if d < 4.2 else 0.0), d, cos, db, p,
                          a["seq"], a["pos"]))
    cands.sort()
    _, wd, wcos, wb, wpos, wseq, wpos_raw = cands[0]
    print("   HOH %d: %.2f A from C1, cos(angle to the C1-OE2 bond) = %+.2f "
          "(+1 is straight behind it), %.2f A from Glu461"
          % (wseq, wd, wcos, wb))
    print("   -> it sits on the face the glucose left, in line with the bond it "
          "attacks, hydrogen-bonded to the base that deprotonates it")
    # and the product hydroxyl it becomes
    o1_461 = min(float(np.linalg.norm(gal_7["O1"] - place(REF, a["pos"])))
                 for a in structures[REF]
                 if a["chain"] == "A" and a["seq"] == 461 and a["name"] in ("OE1", "OE2"))
    print("   in the product (1JZ7) the O1 it became is %.2f A from Glu461 — the same "
          "residue still holding it" % o1_461)

    # --- how far the sugar slides, shallow site to deep ------------------
    def ring_centre(e, comp):
        r = ring_of(e, comp)
        return np.array([r[n] for n in RING if n in r]).mean(0)

    shallow = {e: ring_centre(e, LIGANDS[e][0]) for e in ("1JYN", "1JYV", "1JYW", "1JZ8")}
    deep = {e: ring_centre(e, LIGANDS[e][0]) for e in ("1JZ5", "1JZ2", "1JZ7")}
    sc = np.array(list(shallow.values())).mean(0)
    dc = np.array(list(deep.values())).mean(0)
    slide = float(np.linalg.norm(dc - sc))
    spread_s = float(max(np.linalg.norm(v - sc) for v in shallow.values()))
    spread_d = float(max(np.linalg.norm(v - dc) for v in deep.values()))
    print("\nthe shallow site and the deep site:")
    print("   four substrate complexes (all E537Q) cluster within %.2f A of each other"
          % spread_s)
    print("   three wild-type complexes cluster within %.2f A of each other" % spread_d)
    print("   the two clusters are %.2f A apart — the slide the sugar makes before "
          "chemistry" % slide)
    print("   CAVEAT: every shallow structure is the E537Q mutant and every deep one "
          "is wild type, so\n           'substrate binds shallow first' and 'taking the "
          "nucleophile away leaves it shallow'\n           cannot be told apart from "
          "these structures alone.")

    # --- the tetramer ----------------------------------------------------
    #
    # The enzyme itself: no waters, no cryoprotectant, and none of the ligands,
    # because the ligands are what this render animates. Leaving 1JZ7's own
    # galactose in would have the incoming substrate collide with the product
    # of the previous turn of the cycle — which is exactly what the clearance
    # check below caught the first time it was run.
    heavy = [a for a in ref if a["el"] != "H" and a["comp"] not in NOT_THE_ENZYME]
    pos = np.array([a["pos"] for a in heavy])
    site_c1 = gal_7["C1"]

    # An orientation that makes the render work: origin at the tetramer's
    # centroid, and the way out of the active site pointing at the camera.
    centre = pos.mean(0)
    radii = np.array([VDW.get(a["el"], 1.7) for a in heavy])
    print("\nfinding the way out of the site (flooding the free space with a probe):")
    path = escape_channel(pos, radii, site_c1, None)
    if path is None:
        raise SystemExit("no route from the active site to bulk solvent was found")
    path = smooth_path(path)
    out_dir = path[-1] - path[0]
    out_dir = out_dir / np.linalg.norm(out_dir)
    print("   %d waypoints, %.1f A from the site to solvent, heading %s"
          % (len(path), float(np.linalg.norm(path[-1] - path[0])), out_dir.round(2)))

    # Smoothing rounds the lattice corners, so the clearance has to be measured
    # again afterwards rather than inherited from the flood fill.
    dense = np.array([path[i] + (path[i + 1] - path[i]) * f
                      for i in range(len(path) - 1)
                      for f in np.linspace(0, 1, 12, endpoint=False)] + [path[-1]])
    clear = min(float(np.min(np.linalg.norm(pos - q, axis=1) - radii)) for q in dense)
    print("   narrowest point: %.2f A of clear space beyond the van der Waals "
          "surface (a water needs 1.4)" % clear)

    # The cross-check that says this is the real channel and not an artefact of
    # the flood fill: the glucose half of the lactose in 1JYN, which is on its
    # way out in the crystal, should already be sitting in it.
    glc_pts = np.array([glc_n[n] for n in sorted(glc_n)])
    seg = np.array([float(np.min(np.linalg.norm(dense - p, axis=1))) for p in glc_pts])
    print("   the glucose of 1JYN's lactose lies %.2f-%.2f A off this channel — "
          "the crystal\n     already has the leaving group in the way out"
          % (seg.min(), seg.max()))
    if seg.min() > 4.0:
        raise SystemExit("the channel does not pass the leaving group; it is the "
                         "wrong channel")
    channel_clearance = round(clear, 2)
    glucose_off_channel = [round(float(seg.min()), 2), round(float(seg.max()), 2)]

    # Rotate so that the channel points along +z, i.e. at the camera.
    def rotation_taking(a, b):
        a = a / np.linalg.norm(a)
        b = np.array(b, dtype=float)
        b = b / np.linalg.norm(b)
        v = np.cross(a, b)
        c = float(np.dot(a, b))
        if np.linalg.norm(v) < 1e-9:
            return np.eye(3) if c > 0 else -np.eye(3)
        kx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3) + kx + kx @ kx / (1 + c)

    rot = rotation_taking(out_dir, [0, 0, 1])

    def final(p):
        return rot @ (np.asarray(p) - centre)

    atoms_out = []
    for a, p in zip(heavy, pos):
        q = final(p)
        atoms_out.append({"el": a["el"], "pos": [round(float(v), 3) for v in q],
                          "chain": a["chain"], "seq": a["seq"], "res": a["comp"]})

    # --- the cast, in every state ----------------------------------------
    def cast_of(e):
        """The side chains drawn ball-and-stick, in entry e's pose."""
        out = []
        for a in structures[e]:
            if a["chain"] == "A" and a["seq"] in CAST_RESIDUES and a["group"] == "ATOM":
                if a["name"] in ("N", "C", "O"):      # backbone stubs, not drawn
                    continue
                out.append({"el": a["el"], "name": a["name"], "res": a["comp"],
                            "seq": a["seq"],
                            "pos": [round(float(v), 3) for v in final(place(e, a["pos"]))]})
        return out

    def ligand_of(e):
        out = []
        for key, p in sorted(ligands[e].items()):
            comp, name = key.split(":")
            el = "".join(ch for ch in name if ch.isalpha())[:1]
            if name.startswith("F"):
                el = "F"
            out.append({"el": el, "name": name, "comp": comp,
                        "pos": [round(float(v), 3) for v in final(p)]})
        return out

    # --- the eight boundary poses of the catalytic cycle -----------------
    #
    # 24 heavy atoms and 3 hydrogens with fixed identity, given a position at
    # each of nine boundaries; the ninth is the first, so the loop closes.
    # Where a boundary has a deposited structure its atoms come straight from
    # it. Where it does not, the modelling is named in `provenance` and is as
    # small as it can be made.
    #
    #   B0 parked          nothing in the site
    #   B1 shallow         1JYN: lactose bound, but 3 A short of the deep site
    #   B2 deep            MODEL: 1JYN's lactose, its galactosyl ring put on
    #                      1JZ5's deep ring. The true Michaelis complex is the
    #                      one state of this cycle nobody has solved.
    #   B3 transition      1JZ5: the flat lactone. Glucose drawn part way out.
    #   B4 covalent        1JZ2: the alpha-galactosyl-enzyme itself
    #   B5 glucose gone    1JZ2's galactosyl; glucose away down the channel
    #   B6 water in place  1JZ2's galactosyl and 1JZ2's own attacking water
    #   B7 product         1JZ7: beta-galactose, Glu537 free again
    #   B8 = B0
    GAL_NAMES = ["C1", "C2", "C3", "C4", "C5", "C6", "O2", "O3", "O4", "O5", "O6"]
    GLC_NAMES = ["C1", "C2", "C3", "C4", "C5", "C6", "O1", "O2", "O3", "O4", "O5", "O6"]
    provenance = {}

    def named(ring, names, sub=None):
        out = []
        for n in names:
            key = n if n in ring else (sub or {}).get(n)
            if key is None or key not in ring:
                raise SystemExit("missing atom %s" % n)
            out.append(np.asarray(ring[key], dtype=float))
        return np.array(out)

    def ring_drawn(e, comp):
        return {n.split(":")[1]: p for n, p in drawn[e].items() if n.startswith(comp + ":")}

    gal_shallow = named(gal_n, GAL_NAMES)
    glc_shallow = named(glc_n, GLC_NAMES)
    gal_ts = named(ring_drawn("1JZ5", "149"), GAL_NAMES)    # the flat lactone's ring
    # 2FG carries fluorine where the 2-hydroxyl belongs; the trap is what makes
    # the intermediate long-lived enough to crystallise. The oxygen is put back
    # at the fluorine's place: C-F is 1.39 A and C-O 1.43 A, so this moves one
    # atom by about 0.04 A and changes nothing about the anomeric centre.
    gal_cov = named(ring_drawn("1JZ2", "2FG"), GAL_NAMES, sub={"O2": "F2"})
    provenance["O2 of the covalent intermediate"] = (
        "MODEL: 1JZ2 is the 2-deoxy-2-FLUORO-galactosyl enzyme — the fluorine is "
        "the trap that made the intermediate last long enough to see. The hydroxyl "
        "is drawn back at the fluorine's position (C-F 1.39 A vs C-O 1.43 A).")
    gal_prod = named(gal_7, GAL_NAMES)
    prod_o1 = np.asarray(gal_7["O1"], dtype=float)

    # B2: the deep Michaelis complex, which is not deposited. The smallest
    # honest model of it: move 1JYN's lactose as a rigid body so its galactosyl
    # ring lands on the deep ring that IS deposited, and carry the glucose.
    Rm, tm = kabsch(gal_shallow[:6], gal_ts[:6])
    gal_deep = gal_shallow @ Rm.T + tm
    glc_rigid = glc_shallow @ Rm.T + tm

    def clearance(block, limit=16.0):
        """How far a block of atoms is from running into the protein: the
        smallest gap between an atom's centre and a protein atom's van der
        Waals surface. Negative means overlap."""
        worst = 1e9
        for p in block:
            m = np.abs(pos - p).max(1) < limit
            if not m.any():
                continue
            worst = min(worst, float(np.min(np.linalg.norm(pos[m] - p, axis=1) - radii[m])))
        return worst

    rigid_clash = clearance(glc_rigid)
    print("\nthe deep Michaelis complex, which nobody has solved:")
    print("   moving 1JYN's lactose onto the deep ring as ONE RIGID BODY buries the "
          "glucose\n     %.2f A inside the protein. That is not a complex, it is a "
          "collision." % rigid_clash)

    def rotate_about(block, a, b, degrees, keep):
        """Turn `block` about the axis a->b, leaving the atoms in `keep` alone."""
        k = (b - a) / np.linalg.norm(b - a)
        th = np.radians(degrees)
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        R = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)
        out = (block - a) @ R.T + a
        for i in keep:
            out[i] = block[i]
        return out

    # The one thing that MUST change when the sugar moves 2.7 A is the pair of
    # torsions about the glycosidic linkage — that is what a glycosidic bond is
    # for. So the glucose is swung about C1-O4 and about O4-C4 until it fits,
    # and how far it had to swing is reported rather than hidden.
    o4i = GLC_NAMES.index("O4")
    best = (rigid_clash, 0.0, 0.0, glc_rigid)
    for phi in range(0, 360, 6):
        a = rotate_about(glc_rigid, gal_deep[0], glc_rigid[o4i], phi, keep=[o4i])
        if clearance(a[[o4i]]) < -0.6:
            continue
        for psi in range(0, 360, 6):
            b = rotate_about(a, a[o4i], a[3], psi, keep=[o4i, 3])
            c = clearance(b)
            if c > best[0]:
                best = (c, float(phi), float(psi), b)
    clash, phi, psi, glc_deep = best
    print("   swinging the glucose about the two glycosidic torsions instead — the "
          "one degree\n     of freedom a glycosidic bond actually has — gets it to "
          "%+.2f A at phi %.0f deg, psi %.0f deg." % (clash, phi, psi))
    bond = float(np.linalg.norm(glc_deep[o4i] - gal_deep[0]))
    print("   the bond that is about to break is still %.2f A long." % bond)
    if clash < -0.7 or abs(bond - 1.39) > 0.05:
        raise SystemExit("no glycosidic torsion puts the glucose in the deep site "
                         "without a collision; the rigid slide model is wrong")
    provenance["the deep Michaelis complex (B2)"] = (
        "MODEL: no structure has substrate in the DEEP site — it is the one state "
        "of this cycle nobody has caught. 1JYN's galactosyl ring is placed on "
        "1JZ5's deep ring (measured at both ends, %.2f A apart), and the glucose "
        "is then swung about the two glycosidic torsions (phi %.0f, psi %.0f) "
        "until it stops hitting the protein. The C1-O4 bond is held at its "
        "crystallographic %.2f A throughout." % (slide, phi, psi, bond))

    # The channel, as a function of arc length from the site outward. Past the
    # point where the flood fill broke out into bulk solvent it is continued in
    # a straight line, because bulk solvent has no shape to follow.
    #
    # It runs to 92 A rather than stopping at the protein's surface for a
    # camera reason that is also a physical one: the way out of this site points
    # STRAIGHT AT the viewer, so a product parked just outside the mouth sits
    # between the camera and the site and is magnified into a balloon. A
    # molecule that has diffused away is not 40 A from where it started, and
    # parking it where it really goes puts it behind the camera, which is where
    # a molecule that has left ought to be.
    tail = [path[-1] + out_dir * f for f in np.arange(6.0, 56.0, 6.0)]
    path = np.vstack([path, np.array(tail)])
    chan = np.array(path)
    seglen = np.linalg.norm(np.diff(chan, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seglen)])

    def along(s):
        """A point s angstroms out along the channel from the site."""
        s = float(np.clip(s, 0, arc[-1]))
        i = int(np.searchsorted(arc, s, side="right") - 1)
        i = min(i, len(chan) - 2)
        f = (s - arc[i]) / max(seglen[i], 1e-6)
        return chan[i] + (chan[i + 1] - chan[i]) * f

    def shift_to(block, target):
        return block + (target - block.mean(0))

    # The park: out in bulk solvent at the far end of the channel, where a
    # product that has diffused away and a substrate that has not yet arrived
    # are the same thing — which is what lets the loop close.
    park_c = along(arc[-1])          # bulk solvent, 92 A out and behind the camera
    gal_park = shift_to(gal_prod, park_c + np.array([0.0, 4.5, 0.0]))
    glc_park = shift_to(glc_shallow, park_c + np.array([0.0, -4.5, 0.0]))
    # the product's O1 rides with the parked galactose, in the same place the
    # incoming water parks
    park_o1 = gal_park[0] + (prod_o1 - gal_prod[0])

    # Glucose on its way out: the crystallographic leaving group is already
    # part way down the channel, so its exit is the channel, not a straight line.
    def glc_at(s):
        return shift_to(glc_shallow, along(s))

    glc_out_arc = float(np.linalg.norm(glc_shallow.mean(0) - site_c1))

    heavy_labels = (["GAL:" + n for n in GAL_NAMES]
                    + ["GLC:" + n for n in GLC_NAMES] + ["WAT:O"])
    heavy_elements = ["C"] * 6 + ["O"] * 5 + ["C"] * 6 + ["O"] * 6 + ["O"]

    # 1JZ2's own attacking water, placed in the same frame as the sugar it
    # attacks so the two keep the geometry they were refined with.
    wat_attack = draw("1JZ2", wpos_raw)
    e537_ref = {a["name"]: place(REF, a["pos"]) for a in structures[REF]
                if a["chain"] == "A" and a["seq"] == 537 and a["group"] == "ATOM"}
    e461_ref = {a["name"]: place(REF, a["pos"]) for a in structures[REF]
                if a["chain"] == "A" and a["seq"] == 461 and a["group"] == "ATOM"}
    bond_drawn = float(np.linalg.norm(gal_cov[0] - e537_ref["OE2"]))
    wat_drawn = min(float(np.linalg.norm(wat_attack - e461_ref[n])) for n in ("OE1", "OE2"))
    print("\nputting the covalent state in the frozen protein:")
    print("   1JZ2's own C1-OE2 bond                          %.3f A" % bond_own_native)
    print("   drawn in 1JZ7's frame, superposed on the chain  %.3f A  <- not a bond" % bond_chain)
    print("   drawn with Glu537's oxygen translated onto 1JZ7's %.3f A" % bond_drawn)
    print("   the attacking water stays %.2f A from Glu461 (measured %.2f in 1JZ2)"
          % (wat_drawn, wb))
    if abs(bond_drawn - bond_own_native) > 0.05:
        raise SystemExit("the covalent bond is drawn at the wrong length")

    def boundary(gal, glc, wat):
        return np.vstack([gal, glc, wat[None, :]])

    B = [
        boundary(gal_park, glc_park, park_o1),                          # 0 parked
        boundary(gal_shallow, glc_shallow, park_o1),                    # 1 shallow
        boundary(gal_deep, glc_deep, park_o1),                          # 2 deep
        boundary(gal_ts, glc_at(glc_out_arc + 1.1), park_o1),           # 3 transition
        boundary(gal_cov, glc_at(glc_out_arc + 3.4), park_o1),          # 4 covalent
        boundary(gal_cov, glc_park, park_o1),                           # 5 glucose gone
        boundary(gal_cov, glc_park, wat_attack),                        # 6 water in place
        boundary(gal_prod, glc_park, prod_o1),                          # 7 product
        boundary(gal_park, glc_park, park_o1),                          # 8 = 0
    ]
    assert np.allclose(B[0], B[8]), "the heavy-atom loop does not close"

    # --- the three hydrogens that carry the mechanism --------------------
    #
    # X-rays at 1.5 A do not see hydrogens, so every one of these is placed by
    # this code and is labelled MODEL. Their POSITIONS are modelled; that they
    # move between these particular oxygens is the mechanism, and the heavy-atom
    # geometry that makes it possible is measured.
    e461 = {a["name"]: place(REF, a["pos"]) for a in structures[REF]
            if a["chain"] == "A" and a["seq"] == 461 and a["group"] == "ATOM"}
    e537 = {a["name"]: place(REF, a["pos"]) for a in structures[REF]
            if a["chain"] == "A" and a["seq"] == 537 and a["group"] == "ATOM"}

    def hydrogen_on(o, away_from, toward, length=0.98):
        """An O-H of the usual length, pointing from `o` toward `toward` but
        kept off the bond to `away_from`."""
        v = toward - o
        v = v - np.dot(v, (o - away_from) / np.linalg.norm(o - away_from)) * 0.0
        return o + length * v / np.linalg.norm(v)

    bridge_shallow = glc_shallow[GLC_NAMES.index("O4")]
    bridge_deep = glc_deep[GLC_NAMES.index("O4")]
    h_acid_site = hydrogen_on(e461["OE2"], e461["CD"], gal_deep[0])
    h_on_bridge_deep = hydrogen_on(bridge_deep, glc_deep[3], e461["OE2"])
    # Mid-transfer: on the line from Glu461's oxygen to the one it is
    # protonating, too far from either to be bonded to either. Placed
    # explicitly rather than as a midpoint, so "in flight" is a stated
    # distance and not an accident of where the midpoint happened to land.
    _v = bridge_deep - e461["OE2"]
    h_in_flight = e461["OE2"] + 1.35 * _v / np.linalg.norm(_v)
    park_glc_o4h = hydrogen_on(glc_park[GLC_NAMES.index("O4")],
                               glc_park[3], glc_park[GLC_NAMES.index("O4")]
                               + np.array([0.0, -1.0, 0.0]))
    park_gal_o1h = hydrogen_on(park_o1, gal_park[0], park_o1 + np.array([0.0, 1.0, 0.0]))
    wat_h1 = hydrogen_on(wat_attack, gal_cov[0], e461["OE2"])
    wat_h2 = hydrogen_on(wat_attack, gal_cov[0], wat_attack + (wat_attack - e461["OE2"]))
    prod_o1h = hydrogen_on(prod_o1, gal_prod[0], prod_o1 + (prod_o1 - e461["OE2"]))

    def glc_h(s):
        blk = glc_at(s)
        return hydrogen_on(blk[GLC_NAMES.index("O4")], blk[3],
                           blk[GLC_NAMES.index("O4")] + (blk.mean(0) - site_c1))

    # H_acid starts on Glu461, is handed to the oxygen that leaves, and goes out
    # with the glucose. H_w1 and H_w2 arrive on the water; one becomes the
    # product's hydroxyl, the other is the proton Glu461 gets back. The proton
    # Glu461 ends with is NOT the one it started with, which is exactly what a
    # catalytic cycle means.
    H = np.array([
        [h_acid_site, park_gal_o1h, park_glc_o4h],                      # 0 parked
        [h_acid_site, park_gal_o1h, park_glc_o4h],                      # 1 shallow
        [h_acid_site, park_gal_o1h, park_glc_o4h],                      # 2 deep
        [h_in_flight, park_gal_o1h, park_glc_o4h],                       # 3 in flight
        [glc_h(glc_out_arc + 3.4), park_gal_o1h, park_glc_o4h],         # 4 covalent
        [park_glc_o4h, park_gal_o1h, park_glc_o4h],                     # 5 glucose gone
        [park_glc_o4h, wat_h1, wat_h2],                                 # 6 water in place
        [park_glc_o4h, prod_o1h, h_acid_site],                          # 7 product
        [park_glc_o4h, park_gal_o1h, h_acid_site],                      # 8 parked again
    ])
    # The loop closes as a SET of places, not as a list of atoms: at B8 the
    # proton on Glu461 is the one the water brought, and the one Glu461 started
    # with has gone out on the glucose. So B0 and B8 hold the same three
    # positions in a different order, and the permutation is written down.
    H_PERMUTATION = [2, 1, 0]
    assert np.allclose(H[8], H[0][H_PERMUTATION]), "the proton loop does not close"

    def emit(block, labels, elements):
        return [{"el": e, "label": l, "pos": [round(float(v), 3) for v in final(p)]}
                for p, l, e in zip(block, labels, elements)]

    boundaries = []
    hlabels = ["H:acid", "H:w1", "H:w2"]
    for k in range(9):
        boundaries.append(emit(B[k], heavy_labels, heavy_elements)
                          + emit(H[k], hlabels, ["H", "H", "H"]))

    provenance["all three hydrogens"] = (
        "MODEL: X-rays at 1.5-2.1 A do not see hydrogen. Every H here is placed "
        "by build_lactase.py at 0.98 A on the oxygen it belongs to, pointing at "
        "the oxygen it is going to. Which oxygens those are is the mechanism; "
        "where the H sits between them is a drawing.")
    provenance["the paths between boundaries"] = (
        "MODEL: smoothstep interpolation between poses. Eight of the nine "
        "boundaries are a deposited structure or a rigid-body placement onto "
        "one; nothing between them is.")
    provenance["the way glucose and galactose leave"] = (
        "DERIVED: the channel found by flooding the free space around the site "
        "with a 1.4 A probe and walking out (escape_channel). It passes within "
        "%.2f A of the glucose 1JYN already has sitting in it." % glucose_off_channel[0])

    # --- why the tetramer and not one subunit -----------------------------
    #
    # The usual reason given is that a neighbouring subunit completes the active
    # site. Measured, that is nearly but not quite what happens: chain D's
    # 272-288 loop packs against chain A's site residues at van der Waals
    # contact, so a lone monomer would have a hole in the wall of its site — but
    # the loop gets no nearer than 9 A to the sugar and contributes no catalytic
    # atom. Every residue that does chemistry here belongs to one subunit.
    posA = np.array([a["pos"] for a in heavy if a["chain"] == "A"])
    site_a = np.array([a["pos"] for a in heavy
                       if a["chain"] == "A" and a["seq"] in SITE_RESIDUES])
    interface = {}
    for ch in "BCD":
        loop = np.array([a["pos"] for a in heavy
                         if a["chain"] == ch and 272 <= a["seq"] <= 288])
        if len(loop) == 0:
            continue
        interface[ch] = {
            "toSubstrate": round(float(np.linalg.norm(loop - site_c1, axis=1).min()), 1),
            "toSiteResidues": round(float(min(np.linalg.norm(site_a - p, axis=1).min()
                                              for p in loop)), 1)}
    partner = min(interface, key=lambda c: interface[c]["toSiteResidues"])
    print("\nwhy the tetramer:")
    for ch in sorted(interface):
        print("   chain %s's 272-288 loop: %.1f A from chain A's site residues, "
              "%.1f A from the sugar"
              % (ch, interface[ch]["toSiteResidues"], interface[ch]["toSubstrate"]))
    print("   -> chain %s's loop completes the WALL of chain A's site (contact), but "
          "comes no\n      nearer than %.1f A to the substrate and donates no catalytic "
          "residue." % (partner, interface[partner]["toSubstrate"]))
    complementing = [i for i, a in enumerate(heavy)
                     if a["chain"] == partner and 272 <= a["seq"] <= 288]

    mg_final = final(mg[1])
    water_final = final(wpos)
    channel = [[round(float(v), 3) for v in final(p)] for p in path]

    counts = {}
    for a in atoms_out:
        counts[a["el"]] = counts.get(a["el"], 0) + 1
    chains = sorted({a["chain"] for a in atoms_out})

    data = {
        "source": ("E. coli (lacZ) beta-galactosidase; Juers, Heightman, Vasella, "
                   "McCarter, Mackenzie, Withers & Matthews, Biochemistry 40:14781 "
                   "(2001). Protein drawn from %s (1.50 A); waters and hydrogens "
                   "omitted from the space-filling tetramer." % REF),
        "organism": "Escherichia coli",
        "gene": "lacZ",
        "uniprot": LACZ,
        "humanLactase": {"uniprot": HUMAN_LACTASE,
                         "name": "lactase-phlorizin hydrolase",
                         "structures": 0,
                         "note": ("no deposited structure of any kind; this is why the "
                                  "render substitutes the E. coli enzyme and says so "
                                  "on the caption bar")},
        "entries": {e: {"role": r} for e, r in [
            ("1JYN", "E537Q with lactose bound, shallow site"),
            ("1JZ5", "D-galactonolactone, transition-state mimic, deep site"),
            ("1JZ2", "trapped 2-F-galactosyl-enzyme, the covalent intermediate"),
            ("1JZ7", "galactose, the product in the deep site"),
            ("1JYV", "E537Q with ONPG"),
            ("1JYW", "E537Q with PNPG"),
            ("1JZ8", "E537Q with allolactose, the inducer")]},
        "mutantEntries": ["1JYN", "1JYV", "1JYW", "1JZ8"],
        "mutation": "E537Q",
        "reference": REF,
        "motion": motion,
        "anomeric": anomeric,
        "lactone": {"entry": "1JZ5", "outOfPlane": round(flat, 3),
                    "bond": round(carbonyl, 3),
                    "sugarOutOfPlane": round(tetra, 3), "sugarBond": round(single, 3)},
        "metal": {"ligands": metal,
                  "toNucleophile": round(float(mg[0]), 2),
                  "pos": [round(float(v), 3) for v in mg_final]},
        "water": {"entry": "1JZ2", "seq": int(wseq), "toC1": round(wd, 2),
                  "cosToBond": round(wcos, 3), "toGlu461": round(wb, 2),
                  "productO1ToGlu461": round(o1_461, 2),
                  "pos": [round(float(v), 3) for v in water_final]},
        "sites": {"slide": round(slide, 2), "shallowSpread": round(spread_s, 2),
                  "deepSpread": round(spread_d, 2),
                  "shallowEntries": sorted(shallow), "deepEntries": sorted(deep)},
        "channel": channel,
        "channelClearance": channel_clearance,
        "glucoseOffChannel": glucose_off_channel,
        "activeSite": [round(float(v), 3) for v in final(site_c1)],
        "nucleophile": [round(float(v), 3) for v in
                        final([place(REF, a["pos"]) for a in structures[REF]
                               if a["chain"] == "A" and a["seq"] == 537
                               and a["name"] == "OE2"][0])],
        "chains": chains,
        "elements": counts,
        "atomCount": len(atoms_out),
        "extent": [round(float(v), 1) for v in
                   (np.array([a["pos"] for a in atoms_out]).max(0)
                    - np.array([a["pos"] for a in atoms_out]).min(0))],
        "cast": {e: cast_of(e) for e in ("1JYN", "1JZ5", "1JZ2", "1JZ7")},
        "ligands": {e: ligand_of(e) for e in ENTRIES},
        "boundaries": boundaries,
        "boundaryNames": ["parked", "shallow", "deep", "transition", "covalent",
                          "glucose gone", "water in place", "product", "parked"],
        "boundarySource": [
            "MODEL: out in bulk solvent at the far end of the channel",
            "1JYN, measured", "MODEL: 1JYN's lactose rigid-placed on 1JZ5's deep ring",
            "1JZ5, measured (glucose part way out: MODEL)",
            "1JZ2, measured (2-F put back to 2-OH; glucose down the channel: MODEL)",
            "1JZ2, measured (glucose parked: MODEL)",
            "1JZ2, measured, including its own attacking water",
            "1JZ7, measured",
            "MODEL: the same bulk as boundary 0"],
        "protonPermutation": H_PERMUTATION,
        "provenance": provenance,
        "glu461": {n: [round(float(v), 3) for v in final(p)] for n, p in e461.items()},
        "glu537": {n: [round(float(v), 3) for v in final(p)] for n, p in e537.items()},
        "deepClearance": round(float(clash), 2),
        "interface": {"loops": interface, "partner": partner,
                      "complementingAtoms": complementing,
                      "note": ("chain %s's 272-288 loop packs against chain A's active-"
                               "site residues at %.1f A — van der Waals contact — so a "
                               "monomer would have a hole in the wall of its site. It "
                               "gets no closer than %.1f A to the sugar and donates no "
                               "catalytic residue."
                               % (partner, interface[partner]["toSiteResidues"],
                                  interface[partner]["toSubstrate"]))},
        "catalyticChain": "A",
        "atoms": atoms_out,
    }
    with open(OUT, "w") as f:
        json.dump(data, f, indent=0)

    print("\ntetramer: %d heavy atoms, chains %s, %s"
          % (len(atoms_out), "".join(chains), counts))
    print("  extent %.0f x %.0f x %.0f A" % tuple(data["extent"]))
    print("wrote %s (%.1f MB)" % (os.path.relpath(OUT), os.path.getsize(OUT) / 1e6))


if __name__ == "__main__":
    main()
