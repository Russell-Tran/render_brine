"""Builds Resources/grooves.json: a 40 base-pair duplex drawn space-filling,
plus the twelve real crystal base pairs it is checked against.

The point of this step is a thing ball-and-stick cannot show. Drawn as small
balls on sticks, DNA looks like an airy ladder and its grooves look like gaps
between the rungs. Drawn at van der Waals radii — every atom at the size its
electron cloud actually occupies — two facts appear:

  * the core is solid. Out to 3 Å from the helix axis the space is about 78%
    filled, which is as dense as matter gets;
  * the outside is mostly empty, and that empty volume is not nothing. It is
    the major and minor grooves, and the major groove is where proteins reach
    in to read the sequence.

Two structures come out of this file:

  1BNA      The twelve measured base pairs of the Dickerson-Drew dodecamer,
            straight from step 8 with every atom where crystallography put it.
            Used for the occupancy measurement, the groove measurements, and
            the side-by-side still that compares the two representations.

  helix     Forty base pairs of real pGLO sequence on an ideal B-DNA helix,
            stamped from step 8a's four crystal base-pair templates. 1BNA is
            only 12 bp — barely one turn — and a groove has to spiral through
            three or four turns before the eye reads it as a channel. This one
            is IDEALISED: the base pairs are measured, the helix they sit on
            is not.

Run: python3 Tools/build_grooves.py
"""

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "Resources", "grooves.json")
DNA_JSON = os.path.join(HERE, "..", "..", "step008_dna", "Resources", "dna.json")
PGLO_JSON = os.path.join(HERE, "..", "..", "step008a_plasmid", "Resources", "pglo.json")

# Van der Waals radii, Bondi, J. Phys. Chem. 68:441 (1964). These are the radii
# that make this step's pictures mean anything: a space-filling sphere is the
# volume the atom's electrons actually keep other atoms out of.
VDW = {"C": 1.70, "N": 1.55, "O": 1.52, "P": 1.80, "H": 1.20}

# The ideal helix the 40-mer is built on. This is step 8a's ring geometry —
# 512 turns in 5,371 bp — so the same closed base-pair templates apply without
# re-deriving the backbone joins. It works out at 10.49 bp per turn, which is
# the solution value of 10.5 (Wang, PNAS 76:200, 1979) to three figures.
RISE = 3.4
BP_PER_TURN = 5371 / 512
PAIRS = 40

# Where pGLO's 40 bp come from: the start of the GFP gene, so this is the same
# stretch step 8a's dive ends on.
GFP_START = 1342

# The sugar-phosphate backbone. Same chemistry in every nucleotide, and the
# ridge that runs along the outside of the duplex between the two grooves.
BACKBONE = {"P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'"}

# Which face of each base looks into which groove. Seeman, Rosenberg & Rich,
# PNAS 73:804 (1976): the major-groove edge presents a pattern of hydrogen-bond
# donors and acceptors distinctive enough to tell all four base pairs apart,
# which is why sequence-specific proteins read there. The minor-groove edge is
# far more degenerate — A:T and T:A look nearly alike from that side.
MAJOR_EDGE = {
    "A": {"N7", "C5", "C6", "N6"},
    "G": {"N7", "C5", "C6", "O6"},
    "T": {"C4", "O4", "C5", "C7", "C5M", "C6"},
    "C": {"C4", "N4", "C5", "C6"},
}
MINOR_EDGE = {
    "A": {"C2", "N3"},
    "G": {"C2", "N2", "N3"},
    "T": {"C2", "O2"},
    "C": {"C2", "O2"},
}

COMPLEMENT = str.maketrans("ACGT", "TGCA")


# --------------------------------------------------------------- the crystal

def load_1bna():
    """Step 8's dodecamer: atoms, bonds, and residues keyed by (chain, number).

    Step 8 already turned the molecule so its helix axis lies along x and
    centred it on the origin, which is what every measurement below assumes.
    """
    d = json.load(open(DNA_JSON))
    residues = {}
    for i, a in enumerate(d["atoms"]):
        chain_seq, name = a["label"].split(".", 1)
        residues.setdefault((chain_seq[0], int(chain_seq[1:])), {})[name] = i
    return d, residues


SEQ_1BNA = "CGCGAATTCGCG"


def base_of(chain, number):
    """Which base sits at this residue. A residue n pairs with B residue 25-n."""
    if chain == "A":
        return SEQ_1BNA[number - 1]
    return SEQ_1BNA[(25 - number) - 1].translate(COMPLEMENT)


# ------------------------------------------------------------- measurements

def occupancy_profile(positions, radii, shells=12, samples=200_000, seed=20260924):
    """How much of each cylindrical shell about the x axis is inside an atom.

    Monte Carlo rather than analytic, because the atoms overlap: summing sphere
    volumes would double-count every bond. Points are drawn uniformly in each
    shell — with r = sqrt(uniform(r0^2, r1^2)) so area, not radius, is uniform —
    and tested against every atom. The ends are trimmed by 3 Å because a crystal
    frays there and the fraying is not what is being measured.
    """
    lo, hi = positions[:, 0].min() + 3, positions[:, 0].max() - 3
    rng = np.random.default_rng(seed)
    out = []
    for r0 in range(shells):
        r1 = r0 + 1
        x = rng.uniform(lo, hi, samples)
        theta = rng.uniform(0, 2 * np.pi, samples)
        r = np.sqrt(rng.uniform(r0 ** 2, r1 ** 2, samples))
        pts = np.stack([x, r * np.cos(theta), r * np.sin(theta)], axis=1)
        inside = np.zeros(samples, bool)
        for i in range(len(positions)):
            inside |= np.sum((pts - positions[i]) ** 2, axis=1) < radii[i] ** 2
        out.append(float(inside.mean()))
    return out, (float(lo), float(hi))


# Twice the van der Waals radius of a phosphate group, the constant subtracted
# from a cross-strand P...P separation to get a groove width. El Hassan &
# Calladine's convention, and the one Saenger's tables are quoted in.
PHOSPHATE_ALLOWANCE = 5.8

# How many base-pair steps apart the two phosphates that face each other across
# a groove are. Found by scanning every offset and taking the two minima, not
# assumed: see `groove_offsets`.
GROOVE_OFFSET = {"minor": -4, "major": 3}


def groove_offsets(phosphorus, first_chain_residues):
    """The cross-strand offsets at which P...P separation is locally smallest.

    Scanning every offset and taking the minima is what finds the grooves
    without being told where they are: a duplex has exactly two, and which is
    which follows from their width.
    """
    means = {}
    for k in range(-9, 10):
        ds = [np.linalg.norm(phosphorus[("A", i)] - phosphorus[("B", 25 - (i + k))])
              for i in first_chain_residues if ("B", 25 - (i + k)) in phosphorus]
        if len(ds) >= 4:
            means[k] = float(np.mean(ds))
    minima = [k for k in means if all(means[k] <= means[j] for j in (k - 1, k + 1) if j in means)]
    return means, sorted(minima, key=lambda k: means[k])


def measure_grooves(phosphorus, residues_a):
    """Width of each groove, by the standard convention."""
    out = {}
    for name, k in GROOVE_OFFSET.items():
        widths = []
        for i in residues_a:
            j = 25 - (i + k)
            if ("B", j) in phosphorus:
                widths.append(float(np.linalg.norm(phosphorus[("A", i)] - phosphorus[("B", j)])))
        out[name] = {"pp": float(np.mean(widths)),
                     "width": float(np.mean(widths)) - PHOSPHATE_ALLOWANCE,
                     "n": len(widths)}
    return out


def check_edge_assignment(d, residues, phosphorus, residues_a):
    """Does the textbook edge assignment agree with where the atoms actually are?

    For each atom named as a major- or minor-groove edge atom, ask which
    groove's midline it is in fact nearer. They should agree. Where they do not
    is worth knowing: the answer turns out to be the two base pairs at each end
    of the crystal, which are disordered and stack against the neighbouring
    molecule rather than sitting in a proper duplex.
    """
    mid = {}
    for name, k in GROOVE_OFFSET.items():
        mid[name] = np.array([(phosphorus[("A", i)] + phosphorus[("B", 25 - (i + k))]) / 2
                              for i in residues_a if ("B", 25 - (i + k)) in phosphorus])
    results = {}
    for trim in (0, 1, 2):
        agree = disagree = 0
        for (chain, number), names in residues.items():
            index = number if chain == "A" else 25 - number
            if index <= trim or index > 12 - trim:
                continue
            base = base_of(chain, number)
            for name, i in names.items():
                if name in BACKBONE or name.startswith("H"):
                    continue
                want = ("major" if name in MAJOR_EDGE[base]
                        else "minor" if name in MINOR_EDGE[base] else None)
                if want is None:
                    continue
                p = np.array(d["atoms"][i]["pos"])
                near = ("major" if np.min(np.linalg.norm(mid["major"] - p, axis=1))
                        < np.min(np.linalg.norm(mid["minor"] - p, axis=1)) else "minor")
                agree, disagree = (agree + 1, disagree) if near == want else (agree, disagree + 1)
        results[trim] = {"agree": agree, "disagree": disagree,
                         "fraction": agree / (agree + disagree)}
    return results


# ------------------------------------------------------------ the 40-mer

def stamp(templates, sequence):
    """Forty base pairs on an ideal helix, from the four crystal templates.

    Each template is one real base pair from the middle of 1BNA, expressed in a
    frame local to the pair. Stamping puts pair i at x = i x RISE, turned by i
    twists about the helix axis. Step 8a already closed the backbone joins for
    exactly this twist, so neighbouring nucleotides meet at a proper O3'-P bond.
    """
    twist = 2 * np.pi / BP_PER_TURN
    atoms, bonds, hbonds = [], [], []
    for i, base in enumerate(sequence):
        kind = base + base.translate(COMPLEMENT)
        t = templates[kind]
        base0 = len(atoms)
        angle = twist * i
        c, s = np.cos(angle), np.sin(angle)
        for a in t["atoms"]:
            x, y, z = a["pos"]
            atoms.append({
                "el": a["el"],
                "strand": a["strand"],
                "name": a["name"],
                "base": base if a["strand"] == 0 else base.translate(COMPLEMENT),
                "pair": i,
                "pos": [x + i * RISE, c * y - s * z, s * y + c * z],
            })
        bonds += [[base0 + p, base0 + q, o] for p, q, o in t["bonds"]]
        hbonds += [[base0 + p, base0 + q] for p, q in t["hbonds"]]
        # The O3'-P ester joining this pair's nucleotides to the previous one's.
        if i > 0:
            prev = base0 - len(templates[sequence[i - 1] + sequence[i - 1].translate(COMPLEMENT)]["atoms"])
            for strand, back, forward in ((0, "O3'", "P"), (1, "P", "O3'")):
                a = find(atoms, prev, base0, strand, back)
                b = find(atoms, base0, len(atoms), strand, forward)
                if a is not None and b is not None:
                    bonds.append([a, b, 1] if strand == 0 else [b, a, 1])
    return atoms, bonds, hbonds


PO_BOND = 1.60          # the O3'-P ester, Å


def close_backbone(atoms, bonds):
    """Puts every phosphorus where both its bonds are the right length.

    Step 8a closed each base-pair template's backbone against one reference
    template, which is right when every neighbour is that template. Stamping a
    real sequence puts all four types next to each other, and the small
    differences between them — each is a different base pair lifted out of the
    crystal — land on the joints: the O3'-P ester comes out anywhere from 1.5
    to 1.9 Å instead of 1.60.

    So the join is closed again here, on the assembled chain rather than on
    templates in isolation. Each phosphorus is moved to the intersection of two
    spheres — 1.60 Å from its own O5' and 1.60 Å from the neighbouring O3' —
    taking whichever intersection point is nearer where it already sat, and its
    two free oxygens come with it. Nothing else moves, so the bases and their
    pairing are untouched.
    """
    index = {}
    for i, a in enumerate(atoms):
        index[(a["pair"], a["strand"], a["name"])] = i
    moved = []
    for (pair, strand, name), i in sorted(index.items()):
        if name != "P":
            continue
        # Which O3' this phosphorus has to reach: the previous pair's on
        # strand 0, the next pair's on strand 1 (the strands run opposite ways).
        neighbour = (pair - 1, strand, "O3'") if strand == 0 else (pair + 1, strand, "O3'")
        if neighbour not in index:
            continue
        o3 = np.array(atoms[index[neighbour]]["pos"])
        o5_key = (pair, strand, "O5'")
        if o5_key not in index:
            continue
        o5 = np.array(atoms[index[o5_key]]["pos"])
        p = np.array(atoms[i]["pos"])
        ra = float(np.linalg.norm(p - o5))          # keep its own P-O5' as it is
        d = float(np.linalg.norm(o3 - o5))
        if not abs(ra - PO_BOND) < d < ra + PO_BOND:
            continue                                 # no intersection; leave it
        axis = (o3 - o5) / d
        a = (ra ** 2 - PO_BOND ** 2 + d ** 2) / (2 * d)
        h = float(np.sqrt(max(ra ** 2 - a ** 2, 0)))
        centre = o5 + a * axis
        off = p - centre
        off -= np.dot(off, axis) * axis
        n = float(np.linalg.norm(off))
        new_p = centre + (off / n * h if n > 1e-6 else h * np.array([0.0, 0.0, 1.0]))
        shift = new_p - p
        moved.append(float(np.linalg.norm(shift)))
        for nm in ("P", "OP1", "OP2"):
            k = (pair, strand, nm)
            if k in index:
                atoms[k and index[k]]["pos"] = list(np.array(atoms[index[k]]["pos"]) + shift)
    return moved


def joint_lengths(atoms, bonds):
    """Every O3'-P ester length in the assembled chain."""
    out = []
    for a, b, _ in bonds:
        na, nb = atoms[a]["name"], atoms[b]["name"]
        if {na, nb} == {"O3'", "P"}:
            out.append(float(np.linalg.norm(np.array(atoms[a]["pos"]) - np.array(atoms[b]["pos"]))))
    return out


def find(atoms, lo, hi, strand, name):
    for i in range(lo, hi):
        if atoms[i]["strand"] == strand and atoms[i]["name"] == name:
            return i
    return None


def classify(atoms):
    """Every atom as backbone, major-groove edge, minor-groove edge, or interior.

    This is what the render colours by. Element colouring would say nothing
    here: the question is not what an atom is but which groove it lines.
    """
    counts = {}
    for a in atoms:
        base = a["base"]
        if a["name"] in BACKBONE:
            part = "backbone"
        elif a["name"].startswith("H"):
            part = "hydrogen"
        elif a["name"] in MAJOR_EDGE[base]:
            part = "major"
        elif a["name"] in MINOR_EDGE[base]:
            part = "minor"
        else:
            part = "interior"
        a["part"] = part
        counts[part] = counts.get(part, 0) + 1
    return counts


def helix_grooves(atoms):
    """The same groove measurement, on the built 40-mer rather than the crystal."""
    phosphorus = {}
    for a in atoms:
        if a["name"] == "P":
            phosphorus[(a["strand"], a["pair"])] = np.array(a["pos"])
    out = {}
    for name, k in GROOVE_OFFSET.items():
        widths = []
        for i in range(PAIRS):
            # strand 1 runs the other way, so its pair index counts backwards.
            if (0, i) in phosphorus and (1, i + k) in phosphorus:
                widths.append(float(np.linalg.norm(phosphorus[(0, i)] - phosphorus[(1, i + k)])))
        if widths:
            out[name] = {"pp": float(np.mean(widths)),
                         "width": float(np.mean(widths)) - PHOSPHATE_ALLOWANCE,
                         "n": len(widths)}
    return out


def handedness(atoms):
    """Positive if the helix is right-handed: the angle of each successive
    backbone atom about the axis should advance with x."""
    p = [(a["pos"], a["pair"]) for a in atoms if a["name"] == "P" and a["strand"] == 0]
    p.sort(key=lambda t: t[1])
    angles = np.unwrap([np.arctan2(q[2], q[1]) for q, _ in p])
    return float(np.degrees(np.mean(np.diff(angles))))


def main():
    d, residues = load_1bna()
    crystal_pos = np.array([a["pos"] for a in d["atoms"]])
    crystal_rad = np.array([VDW[a["el"]] for a in d["atoms"]])

    print("measuring the crystal (PDB 1BNA, via step 8)")
    profile, span = occupancy_profile(crystal_pos, crystal_rad)
    for i, v in enumerate(profile):
        print(f"  {i:2d}-{i+1:2d} Å   {100*v:5.1f}%  {'#' * int(v * 40)}")
    core = float(np.mean(profile[0:3]))
    rim = float(np.mean(profile[9:11]))
    weights = np.array([(i + 1) ** 2 - i ** 2 for i in range(10)], float)
    weights /= weights.sum()
    overall = float(np.dot(weights, profile[:10]))
    print(f"  core 0-3 Å {100*core:.1f}% · rim 9-11 Å {100*rim:.1f}% · "
          f"whole cylinder to 10 Å {100*overall:.1f}%")

    phosphorus = {}
    for (chain, number), names in residues.items():
        if "P" in names:
            phosphorus[(chain, number)] = np.array(d["atoms"][names["P"]]["pos"])
    residues_a = sorted(n for c, n in phosphorus if c == "A")

    means, minima = groove_offsets(phosphorus, residues_a)
    print(f"\ncross-strand P...P minima at offsets {minima[:2]} "
          f"(scanned {min(means)} to {max(means)})")
    crystal_grooves = measure_grooves(phosphorus, residues_a)
    for name, g in crystal_grooves.items():
        print(f"  {name}: P...P {g['pp']:.2f} Å, width {g['width']:.2f} Å (n={g['n']})")

    edges = check_edge_assignment(d, residues, phosphorus, residues_a)
    for trim, r in edges.items():
        label = "all base pairs" if trim == 0 else f"trimming {trim} at each end"
        print(f"  edge assignment, {label}: {100*r['fraction']:.0f}% agree with geometry")

    pglo = json.load(open(PGLO_JSON))
    sequence = pglo["sequence"][GFP_START - 1:GFP_START - 1 + PAIRS]
    print(f"\nbuilding {PAIRS} bp of pGLO from {GFP_START}: {sequence}")
    atoms, bonds, hbonds = stamp(pglo["templates"], sequence)
    # Stamping lays pair i at x = i x RISE, so the duplex runs from 0 upwards
    # and its middle sits two thirds of the way along. Centre it on the origin,
    # which is what the camera aims at and what every radial measurement below
    # assumes. (Caught by the test that walks the turn checking every atom is
    # in frame: the helix was simply off to one side.)
    centre = float(np.mean([a["pos"][0] for a in atoms]))
    for a in atoms:
        a["pos"] = [a["pos"][0] - centre, a["pos"][1], a["pos"][2]]
    print(f"  centred on the origin (moved {centre:.1f} Å along the axis)")
    before = joint_lengths(atoms, bonds)
    moved = close_backbone(atoms, bonds)
    after = joint_lengths(atoms, bonds)
    print(f"  backbone joins: {min(before):.2f}-{max(before):.2f} Å before, "
          f"{min(after):.2f}-{max(after):.2f} Å after "
          f"({len(moved)} phosphorus atoms moved, mean {np.mean(moved):.2f} Å)")
    assert max(abs(x - PO_BOND) for x in after) < 0.15, f"backbone still open: {min(after)}-{max(after)}"
    counts = classify(atoms)
    print(f"  {len(atoms)} atoms, {len(bonds)} bonds, {len(hbonds)} hydrogen bonds")
    print("  " + " · ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    print(f"  twist {360/BP_PER_TURN:.2f}°/bp ({BP_PER_TURN:.2f} bp per turn), "
          f"rise {RISE} Å, handedness {handedness(atoms):+.2f}°/bp")
    built = helix_grooves(atoms)
    for name, g in built.items():
        print(f"  {name}: P...P {g['pp']:.2f} Å, width {g['width']:.2f} Å (n={g['n']})")

    data = {
        "source": "atoms from PDB 1BNA via step 8; base-pair templates and pGLO sequence via step 8a; "
                  "van der Waals radii from Bondi (1964)",
        "vdw": VDW,
        "crystal": {
            "sequence": SEQ_1BNA,
            "atoms": [{"el": a["el"], "pos": a["pos"], "label": a["label"]} for a in d["atoms"]],
            "bonds": d["bonds"],
            "occupancy": profile,
            "occupancy_span": span,
            "core": core, "rim": rim, "overall": overall,
            "grooves": crystal_grooves,
            "edge_assignment": edges,
        },
        "helix": {
            "sequence": sequence,
            "origin": f"pGLO {GFP_START}..{GFP_START + PAIRS - 1}",
            "pairs": PAIRS, "rise": RISE, "bp_per_turn": BP_PER_TURN,
            "atoms": atoms, "bonds": bonds, "hbonds": hbonds,
            "counts": counts, "grooves": built,
            "twist_per_bp": 360 / BP_PER_TURN,
        },
    }
    json.dump(data, open(OUT, "w"), separators=(",", ":"))
    print(f"\nwrote {os.path.relpath(OUT)} ({os.path.getsize(OUT)//1024} KB)")


if __name__ == "__main__":
    main()
