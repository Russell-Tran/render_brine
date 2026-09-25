"""Builds Resources/pglo.json: the pGLO sequence, its feature map, and the
four base-pair templates used to draw real atoms at the bottom of the dive.

pGLO is Bio-Rad's teaching plasmid — 5,371 bp, and the reason transformed
E. coli glow green under UV. The sequence comes from NovoPro's page for the
vector (https://www.novoprolabs.com/vector/V12008), which publishes the full
GenBank record for Bio-Rad's plasmid; the feature coordinates come from the
same annotation. Everything is re-checked here rather than trusted:

  * the sequence is exactly 5,371 bp of ACGT,
  * every feature lies inside the plasmid,
  * the three protein-coding features (araC, GFP, bla) each begin with ATG,
    have a length divisible by three, contain no internal stop codon, and are
    followed immediately by one. That is a strong independent check that the
    annotation and the sequence describe the same molecule.

The atoms come from step 8's dna.json (PDB 1BNA plus hydrogens). For each of
the four base-pair types this pulls one real example out of the middle of the
dodecamer — the ends of a crystal structure are frayed — and re-expresses its
atoms in a frame local to the base pair, so the same template can be stamped
down anywhere along the plasmid at the right position and twist.

Run: python3 Tools/build_plasmid.py
"""

import json
import os
import re
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "Resources", "pglo.json")
SEQ_CACHE = os.path.join(HERE, "..", "Resources", "pglo_sequence.txt")
DNA_JSON = os.path.join(HERE, "..", "..", "step8_dna", "Resources", "dna.json")
SOURCE_URL = "https://www.novoprolabs.com/vector/V12008"

LENGTH = 5371

# NovoPro's annotation of the Bio-Rad sequence. (start, end, strand); 1-based
# and inclusive, as GenBank counts. Strand −1 means the feature reads along the
# complementary strand.
FEATURES = [
    ("araC",          99,  974, -1, "makes the AraC protein, which holds pBAD off until arabinose arrives"),
    ("pBAD",        1001, 1285,  1, "the arabinose-controlled promoter: the switch"),
    ("RBS",         1312, 1334,  1, "ribosome binding site"),
    ("GFP",         1342, 2058,  1, "green fluorescent protein, the 'cycle 3' variant: the glow"),
    ("MCS",         2063, 2119,  1, "multiple cloning site"),
    ("rrnB T1",     2321, 2407,  1, "transcription terminator"),
    ("rrnB T2",     2499, 2526,  1, "transcription terminator"),
    ("AmpR prom",   2544, 2635,  1, "promoter for bla"),
    ("bla",         2636, 3493,  1, "beta-lactamase: survives ampicillin, so only transformed cells grow"),
    ("f1 ori",      3538, 3993,  1, "phage origin, vestigial here"),
    ("ori",         4104, 4692,  1, "ColE1/pMB1 origin: the plasmid copies itself"),
    ("bom",         4878, 5018, -1, "basis of mobility, vestigial here"),
]

CODING = {"araC", "GFP", "bla"}
STOPS = {"TAA", "TAG", "TGA"}

# The Watson–Crick hydrogen bonds of each base pair, as (donor strand, donor
# atom, acceptor strand, acceptor atom) with strand 0 the first-named base.
# Three for G:C, two for A:T.
WATSON_CRICK = {
    "CG": [(0, "N4", 1, "O6"), (1, "N1", 0, "N3"), (1, "N2", 0, "O2")],
    "GC": [(1, "N4", 0, "O6"), (0, "N1", 1, "N3"), (0, "N2", 1, "O2")],
    "AT": [(0, "N6", 1, "O4"), (1, "N3", 0, "N1")],
    "TA": [(1, "N6", 0, "O4"), (0, "N3", 1, "N1")],
}

# Which features are load-bearing for "glows, survives, copies itself", and the
# color each gets. Everything else is drawn neutral grey — that contrast is the
# argument the render makes.
LIT = {
    "GFP":       ("gfp",    "the glow"),
    "pBAD":      ("switch", "the arabinose switch"),
    "araC":      ("switch", "the arabinose switch"),
    "RBS":       ("gfp",    "the glow"),
    "AmpR prom": ("bla",    "survives ampicillin"),
    "bla":       ("bla",    "survives ampicillin"),
    "ori":       ("ori",    "copies itself"),
}


def complement(s):
    return s.translate(str.maketrans("ACGT", "TGCA"))


def reverse_complement(s):
    return complement(s)[::-1]


def fetch_sequence():
    """The pGLO sequence, cached next to this script after the first download."""
    if os.path.exists(SEQ_CACHE):
        seq = "".join(l.strip() for l in open(SEQ_CACHE) if not l.startswith("#"))
        if len(seq) == LENGTH:
            return seq
    request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "Mozilla/5.0"})
    page = urllib.request.urlopen(request, timeout=60).read().decode("utf-8", "replace")
    runs = sorted(re.findall(r"[ACGT]{1000,}", page), key=len, reverse=True)
    assert runs, "no sequence found on the page"
    seq = runs[0]
    with open(SEQ_CACHE, "w") as f:
        f.write(f"# pGLO, {len(seq)} bp. Source: {SOURCE_URL}\n")
        for i in range(0, len(seq), 60):
            f.write(seq[i:i + 60] + "\n")
    return seq


def check_sequence(seq):
    assert len(seq) == LENGTH, f"expected {LENGTH} bp, got {len(seq)}"
    assert set(seq) <= set("ACGT"), "unexpected letters in the sequence"
    for name, start, end, strand, _ in FEATURES:
        assert 1 <= start <= end <= LENGTH, f"{name} is outside the plasmid"
        if name not in CODING:
            continue
        s = seq[start - 1:end]
        if strand < 0:
            s = reverse_complement(s)
        codons = [s[i:i + 3] for i in range(0, len(s), 3)]
        stop = seq[end:end + 3] if strand > 0 else reverse_complement(seq[start - 4:start - 1])
        assert len(s) % 3 == 0, f"{name} is not a whole number of codons"
        assert codons[0] == "ATG", f"{name} does not start with ATG"
        assert not any(c in STOPS for c in codons), f"{name} has an internal stop codon"
        assert stop in STOPS, f"{name} is not followed by a stop codon"
    print(f"sequence: {LENGTH} bp, GC {100 * (seq.count('G') + seq.count('C')) / LENGTH:.1f}%; "
          f"araC, GFP and bla are all clean open reading frames")


# ------------------------------------------------------- base-pair templates

def load_1bna():
    """Atoms and bonds from step 8, grouped by residue. Step 8 already turned
    the molecule so its helix axis lies along x, centered on the origin."""
    d = json.load(open(DNA_JSON))
    atoms = d["atoms"]
    residues = {}
    for i, a in enumerate(atoms):
        chain_seq, name = a["label"].split(".", 1)
        chain, number = chain_seq[0], int(chain_seq[1:])
        residues.setdefault((chain, number), {})[name] = i
    return atoms, d["bonds"], residues


def base_pair_frame(atoms, a_idx, b_idx):
    """Origin and axes for a base pair: x along the helix, y from one C1' to
    the other, z completing a right-handed set."""
    pa = np.array(atoms[a_idx]["pos"])
    pb = np.array(atoms[b_idx]["pos"])
    origin = (pa + pb) / 2
    x = np.array([1.0, 0.0, 0.0])
    y = pb - pa
    y -= np.dot(y, x) * x
    y /= np.linalg.norm(y)
    z = np.cross(x, y)
    return origin, np.array([x, y, z])


# The sugar-phosphate backbone, which is the same chemistry in every
# nucleotide and so should be the same shape in every template.
BACKBONE = ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'"]


def kabsch(moving, fixed):
    """The rotation and shift that best lays `moving` over `fixed`."""
    mc, fc = moving.mean(axis=0), fixed.mean(axis=0)
    h = (moving - mc).T @ (fixed - fc)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1, 1, d]) @ u.T
    return r, fc - r @ mc


def standardise_backbones(templates, reference="AT"):
    """Lays every template's backbone over the reference one's.

    Each template is one real base pair lifted out of a crystal, so each
    carries that position's own twist, roll and sugar pucker, and stamped side
    by side on a regular ring those differences land at the joints. Fitting
    each strand separately would tidy the joints but pull the two bases apart
    and break the Watson-Crick pairing, so the whole base pair moves as one
    rigid body, fitted on all the backbone atoms of both strands at once. That
    keeps every bond length and every hydrogen bond exactly as the crystal had
    them, and leaves only what one rigid motion cannot fix.
    """
    ref = {}
    for a in templates[reference]["atoms"]:
        if a["name"] in BACKBONE:
            ref[(a["strand"], a["name"])] = np.array(a["pos"])
    for kind, t in templates.items():
        shared = [a for a in t["atoms"] if (a["strand"], a["name"]) in ref]
        moving = np.array([a["pos"] for a in shared])
        fixed = np.array([ref[(a["strand"], a["name"])] for a in shared])
        r, shift = kabsch(moving, fixed)
        residual = float(np.sqrt(((moving @ r.T + shift - fixed) ** 2).sum(axis=1).mean()))
        for a in t["atoms"]:
            a["pos"] = [round(float(v), 3) for v in r @ np.array(a["pos"]) + shift]
        if kind != reference:
            print(f"  {kind} laid over {reference}: backbone off by {residual:.2f} Å RMS")
    return templates


def build_templates(seq):
    """One real example of each base-pair type, in base-pair-local coordinates."""
    atoms, bonds, residues = load_1bna()
    # 1BNA is CGCGAATTCGCG on chain A (residues 1–12) paired with chain B
    # (13–24); residue n on A pairs with 25 − n on B.
    chain_a = "CGCGAATTCGCG"
    # Check that chain A really does advance along +x, or the templates would
    # place the plasmid's strands the wrong way round and invert the helix.
    first = np.array(atoms[residues[("A", 2)]["C1'"]]["pos"])
    last = np.array(atoms[residues[("A", 11)]["C1'"]]["pos"])
    assert last[0] > first[0], "chain A does not run along +x"

    # Middle base pairs, away from the frayed ends, one per type.
    picks = {"CG": 9, "GC": 10, "AT": 6, "TA": 7}
    templates = {}
    for kind, n in picks.items():
        assert chain_a[n - 1] == kind[0]
        a_res, b_res = ("A", n), ("B", 25 - n)
        origin, r = base_pair_frame(atoms, residues[a_res]["C1'"], residues[b_res]["C1'"])
        members = {}          # global atom index -> (strand, local name)
        for strand, key in ((0, a_res), (1, b_res)):
            for name, idx in residues[key].items():
                members[idx] = (strand, name)
        order = sorted(members)
        local_index = {g: i for i, g in enumerate(order)}
        out_atoms = []
        for g in order:
            strand, name = members[g]
            p = r @ (np.array(atoms[g]["pos"]) - origin)
            out_atoms.append({"el": atoms[g]["el"], "strand": strand, "name": name,
                              "pos": [round(float(v), 3) for v in p]})
        out_bonds = [[local_index[a], local_index[b], o] for a, b, o in bonds
                     if a in local_index and b in local_index]

        # The Watson–Crick hydrogen bonds holding the pair together: for each,
        # the donor's hydrogen that actually points at the acceptor.
        by_name = {(a["strand"], a["name"]): i for i, a in enumerate(out_atoms)}
        hbonds, lengths = [], []
        for d_strand, d_name, a_strand, a_name in WATSON_CRICK[kind]:
            d = by_name[(d_strand, d_name)]
            acc = by_name[(a_strand, a_name)]
            hs = [b for a, b, _ in out_bonds if a == d and out_atoms[b]["el"] == "H"]
            hs += [a for a, b, _ in out_bonds if b == d and out_atoms[a]["el"] == "H"]
            assert hs, f"{kind}: donor {d_name} has no hydrogen"
            pa = np.array(out_atoms[acc]["pos"])
            h = min(hs, key=lambda i: np.linalg.norm(np.array(out_atoms[i]["pos"]) - pa))
            hbonds.append([h, acc])
            lengths.append(float(np.linalg.norm(np.array(out_atoms[d]["pos"]) - pa)))
        assert all(2.5 <= l <= 3.4 for l in lengths), f"{kind}: odd H-bond lengths {lengths}"

        templates[kind] = {"atoms": out_atoms, "bonds": out_bonds, "hbonds": hbonds}
        print(f"template {kind}: {len(out_atoms)} atoms, {len(out_bonds)} bonds, "
              f"{len(hbonds)} H-bonds at " + ", ".join(f"{l:.2f}" for l in lengths) + " Å")
    standardise_backbones(templates)
    check_joints(templates)
    return templates


RISE = 3.4                     # ideal B-DNA, and what the ring's size is built from
TURNS = 512
PO_BOND = 1.60                 # the O3'–P ester bond, Å


def one_step(p, forward=True):
    """One base-pair step along an ideal helix whose axis is local x."""
    twist = 2 * np.pi * TURNS / LENGTH * (1 if forward else -1)
    c, s = np.cos(twist), np.sin(twist)
    return np.array([p[0] + (RISE if forward else -RISE), c * p[1] - s * p[2], s * p[1] + c * p[2]])


def joint_lengths(templates):
    spot = {(a["strand"], a["name"]): np.array(a["pos"]) for a in templates["AT"]["atoms"]}
    # Strand 0 runs 5'→3' as the index grows, strand 1 the other way.
    return (float(np.linalg.norm(spot[(0, "O3'")] - one_step(spot[(0, "P")]))),
            float(np.linalg.norm(spot[(1, "P")] - one_step(spot[(1, "O3'")]))))


def close_the_backbone(templates, reference="AT"):
    """Moves each phosphorus so the backbone actually joins up.

    The nucleotides are real crystal geometry, but 1BNA is wound tighter than
    B-DNA in solution — about 10.1 base pairs per turn against 10.5 — and this
    ring is built on the solution figure, because that is what sets a plasmid's
    real size and the turn count the whole supercoiling argument rests on. Laid
    on the looser helix the crystal nucleotides no longer touch: the O3'–P bond
    between neighbours comes out at 2.5 Å instead of 1.60.

    So each phosphorus is placed where both its bonds are right — 1.60 Å from
    its own O5' and 1.60 Å from the previous nucleotide's O3' — which is the
    intersection of two spheres, taking the point nearest where the crystal put
    it. Its two free oxygens come along with it.
    """
    reference_o3 = {a["strand"]: np.array(a["pos"])
                    for a in templates[reference]["atoms"] if a["name"] == "O3'"}
    for kind, t in templates.items():
        spot = {(a["strand"], a["name"]): np.array(a["pos"]) for a in t["atoms"]}
        for strand in (0, 1):
            p = spot[(strand, "P")]
            o5 = spot[(strand, "O5'")]
            # The neighbouring O3' this phosphorus has to reach, taken from the
            # reference template so that any two base-pair types join up.
            neighbour = reference_o3[strand]
            partner = one_step(neighbour, forward=(strand == 1))
            ra = float(np.linalg.norm(p - o5))
            d = float(np.linalg.norm(partner - o5))
            if not abs(ra - PO_BOND) < d < ra + PO_BOND:
                continue
            axis = (partner - o5) / d
            a = (ra ** 2 - PO_BOND ** 2 + d ** 2) / (2 * d)
            h = float(np.sqrt(max(ra ** 2 - a ** 2, 0)))
            centre = o5 + a * axis
            off = p - centre
            off -= np.dot(off, axis) * axis
            n = np.linalg.norm(off)
            new_p = centre + (off / n * h if n > 1e-6 else h * np.array([0.0, 0.0, 1.0]))
            shift = new_p - p
            for atom in t["atoms"]:
                if atom["strand"] == strand and atom["name"] in ("P", "OP1", "OP2"):
                    atom["pos"] = [round(float(v), 3) for v in np.array(atom["pos"]) + shift]
            if kind == "AT":
                print(f"  strand {strand}: phosphorus moved {np.linalg.norm(shift):.2f} Å to close the backbone")
    return templates


def check_joints(templates):
    before = joint_lengths(templates)
    close_the_backbone(templates)
    after = joint_lengths(templates)
    print(f"O3'–P across one step: {before[0]:.2f} / {before[1]:.2f} Å before, "
          f"{after[0]:.2f} / {after[1]:.2f} Å after")
    assert all(abs(x - PO_BOND) < 0.15 for x in after), f"backbone still open: {after}"


def main():
    seq = fetch_sequence()
    check_sequence(seq)
    templates = build_templates(seq)

    features = []
    for name, start, end, strand, note in FEATURES:
        lit, role = LIT.get(name, (None, None))
        features.append({"name": name, "start": start, "end": end, "strand": strand,
                         "note": note, "color": lit or "grey", "role": role or ""})
    lit_bp = sum(f["end"] - f["start"] + 1 for f in features if f["color"] != "grey")
    print(f"load-bearing: {lit_bp} bp of {LENGTH} ({100 * lit_bp / LENGTH:.0f}%) lit, the rest grey")

    data = {
        "name": "pGLO",
        "length": LENGTH,
        "source": f"sequence and feature annotation from {SOURCE_URL} (Bio-Rad's pGLO); "
                  "atoms from PDB 1BNA via step 8",
        "sequence": seq,
        "features": features,
        "templates": templates,
    }
    json.dump(data, open(OUT, "w"), separators=(",", ":"))
    size = os.path.getsize(OUT) // 1024
    print(f"wrote {os.path.relpath(OUT)} ({size} KB)")


if __name__ == "__main__":
    main()
