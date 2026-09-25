"""Builds Resources/pair.json: GFP and mCherry side by side, aligned, with the
conjugated path through each chromophore marked.

  GFP      PDB 1EMA, the S65T variant (Ormo, Cubitt, Kallio, Gross, Tsien &
           Remington, Science 273:1392, 1996; 1.9 A)
  mCherry  PDB 2H5Q (Shu, Shaner, Yarbrough, Tsien & Remington, Biochemistry
           45:9639, 2006; 1.36 A)

The two proteins are the same object in almost every way: a barrel of eleven
strands, ~220 residues, with a chromophore each has built out of three of its
own amino acids. What differs is the conjugation, and this script measures it
rather than asserting it.

THE MEASUREMENT THAT MATTERS. A chromophore's colour is set by how far its pi
electrons can spread; a longer conjugated run means a smaller energy gap and
redder light. mCherry's run is longer because of an acylimine - the CA1-N1
bond of the first residue is oxidised from single to double, which brings CA1
and N1 into the pi system and carries it on into the preceding backbone.

That is a claim about a bond order, and bond orders show up as bond lengths:

    GFP      N1-CA1 = 1.471 A     a textbook C-N single bond (1.47)
    mCherry  N1-CA1 = 1.305 A     a textbook C=N double bond (1.28)

Both are measured from the deposited coordinates below and asserted in main().
Nothing else in either chromophore differs by anything like as much.

WHAT ISN'T THE DIFFERENCE. mCherry's chromophore has 23 atoms to GFP's 22, and
it would be easy to point at that spare atom as the cause. It isn't: GFP is
built from Thr-Tyr-Gly and mCherry from Met-Tyr-Gly, so the extra atom is just
the sulfur-bearing methionine side chain (CB1-CG1-SD-CE against CB1-CG1-OG1).
That side chain is not conjugated and has nothing to do with the colour. The
colour difference is a bond order, not an atom count.

Two housekeeping differences from step 9's single-protein script:

  * 2H5Q has alternate conformations (altloc A and B) where 1EMA has none, so
    only the blank or 'A' conformer is kept.
  * mCherry has no selenomethionine. 1EMA's four are a crystallographer's
    phasing trick, not biology, and are noted for the caption.

Run: python3 Tools/build_pair.py
"""

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "Resources")
OUT = os.path.join(RES, "pair.json")

# Covalent radii (A), Cordero et al., Dalton Trans. (2008) 2832.
COVALENT = {"C": 0.76, "N": 0.71, "O": 0.66, "S": 1.05, "SE": 1.20}

# Emission and excitation peaks in nm, from FPbase (fpbase.org), which is the
# standard reference for fluorescent-protein spectra. These are the values for
# the exact variants whose structures are rendered here - NOT wild-type avGFP,
# whose 395/509 dual-peak excitation belongs to a different molecule.
PROTEINS = {
    "GFP": {
        "pdb": "1EMA",
        "label": "GFP (S65T)",
        "chromophore": "CRO",
        "formedFrom": ["THR", "TYR", "GLY"],
        "excitation": 490,
        "emission": 510,
        "source": "PDB 1EMA, Ormo et al., Science 273:1392 (1996), 1.9 A",
        "spectra": "FPbase 'GFP (S65T)': ex 490, em 510",
    },
    "mCherry": {
        "pdb": "2H5Q",
        "label": "mCherry",
        "chromophore": "CH6",
        "formedFrom": ["MET", "TYR", "GLY"],
        "excitation": 587,
        "emission": 610,
        "source": "PDB 2H5Q, Shu et al., Biochemistry 45:9639 (2006), 1.36 A",
        "spectra": "FPbase 'mCherry': ex 587, em 610",
    },
}

# The conjugated pi system of each chromophore, by atom name. Both share the
# phenol ring, the methine bridge and the imidazolinone; mCherry continues
# through CA1 and N1 because of the acylimine. Every bond along these paths is
# checked for connectivity in main(), so a wrong name fails loudly.
SHARED_PI = ["OH", "CZ", "CE1", "CE2", "CD1", "CD2", "CG2",   # phenol
             "CB2", "CA2",                                    # methine bridge
             "N2", "C1", "N3", "C2", "O2"]                    # imidazolinone
EXTRA_PI = {"GFP": [], "mCherry": ["CA1", "N1"]}              # the acylimine

# The bond whose order is the whole story.
ACYLIMINE = ("N1", "CA1")


def read_pdb(path):
    """Every non-water atom in the first conformer, in file order."""
    atoms = []
    for line in open(path):
        if line[:6] not in ("ATOM  ", "HETATM"):
            continue
        if line[17:20].strip() == "HOH":
            continue
        if line[16] not in (" ", "A"):        # keep one conformer only
            continue
        atoms.append({
            "name": line[12:16].strip(),
            "res": line[17:20].strip(),
            "seq": int(line[22:26]),
            "pos": np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])]),
            "el": line[76:78].strip().upper(),
        })
    return atoms


def find_bonds(atoms):
    """Pairs closer than their covalent radii allow, via a coarse grid."""
    pos = np.array([a["pos"] for a in atoms])
    cell = 3.0
    grid = {}
    for i, p in enumerate(pos):
        grid.setdefault(tuple((p // cell).astype(int)), []).append(i)
    bonds = []
    for i, p in enumerate(pos):
        base = (p // cell).astype(int)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for j in grid.get((base[0] + dx, base[1] + dy, base[2] + dz), []):
                        if j <= i:
                            continue
                        limit = COVALENT[atoms[i]["el"]] + COVALENT[atoms[j]["el"]] + 0.45
                        if float(np.linalg.norm(pos[i] - pos[j])) < limit:
                            bonds.append((i, j))
    return sorted(bonds)


def read_strands(path):
    """The beta strands the depositors annotated."""
    return [(int(l[22:26]), int(l[33:37])) for l in open(path) if l.startswith("SHEET")]


def rotation_taking(axis, to):
    """A rotation matrix taking unit vector `axis` onto `to`."""
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    b = np.asarray(to, dtype=float)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else -np.eye(3)
    kx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + kx + kx @ kx / (1 + c)


def spin_about_y(degrees):
    t = np.radians(degrees)
    return np.array([[np.cos(t), 0, np.sin(t)], [0, 1, 0], [-np.sin(t), 0, np.cos(t)]])


def barrel_frame(atoms):
    """Put the barrel's long axis along y, centred on the origin, then spin
    about y until the chromophore faces +z. Both proteins get the same
    treatment, so the two end up in a common frame without needing a sequence
    alignment between folds that share only about a quarter of their residues."""
    ca = np.array([a["pos"] for a in atoms if a["name"] == "CA"])
    assert len(ca) > 200, f"expected a backbone, found {len(ca)} alpha carbons"
    center = ca.mean(axis=0)
    _, _, vt = np.linalg.svd(ca - center)
    axis = vt[0]
    half = len(ca) // 2
    if np.dot(axis, ca[half:].mean(axis=0) - ca[:half].mean(axis=0)) < 0:
        axis = -axis
    return center, rotation_taking(axis, [0, 1, 0])


def build(key):
    """One protein: atoms placed in the common frame, plus everything measured
    about it that the renderer or the tests will want."""
    spec = PROTEINS[key]
    path = os.path.join(RES, spec["pdb"] + ".pdb")
    atoms = read_pdb(path)
    chromo_name = spec["chromophore"]

    residues = sorted({(a["seq"], a["res"]) for a in atoms})
    numbers = [s for s, _ in residues]
    names = {s: r for s, r in residues}

    # The chromophore: one residue, formed by three fusing, so two numbers are
    # missing from the sequence. Which two differs between the proteins, so it
    # is found rather than assumed.
    chromo_seq = [s for s, r in residues if r == chromo_name]
    assert len(chromo_seq) == 1, f"{key}: expected one {chromo_name}, found {chromo_seq}"
    chromo_seq = chromo_seq[0]
    missing = [s for s in range(numbers[0], numbers[-1] + 1) if s not in names]
    assert len(missing) == 2, f"{key}: expected two absent residue numbers, found {missing}"
    chromo_idx = [i for i, a in enumerate(atoms) if a["res"] == chromo_name]

    center, rot = barrel_frame(atoms)
    placed = np.einsum("ij,kj->ik", np.array([a["pos"] for a in atoms]) - center, rot)
    placed -= placed.mean(axis=0)
    # Spin about the barrel axis so each chromophore faces the camera (+z).
    c = placed[chromo_idx].mean(axis=0)
    placed = np.einsum("ij,kj->ik", placed, spin_about_y(np.degrees(np.arctan2(c[0], c[2]))))

    by_name = {atoms[i]["name"]: i for i in chromo_idx}
    bonds = find_bonds(atoms)
    bondset = {(min(i, j), max(i, j)) for i, j in bonds}

    def length(n1, n2):
        i, j = by_name[n1], by_name[n2]
        return float(np.linalg.norm(placed[i] - placed[j]))

    # The conjugated path, checked for connectivity so a wrong atom name fails
    # here rather than quietly drawing a broken highlight.
    pi_names = SHARED_PI + EXTRA_PI[key]
    for n in pi_names:
        assert n in by_name, f"{key}: chromophore has no atom {n}"
    pi_idx = [by_name[n] for n in pi_names]
    pi_bonds = [[i, j] for i, j in bondset if i in set(pi_idx) and j in set(pi_idx)]
    assert len(pi_bonds) >= len(pi_names) - 1, \
        f"{key}: pi system looks disconnected ({len(pi_bonds)} bonds for {len(pi_names)} atoms)"

    strands = read_strands(path)
    in_barrel = set()
    for first, last in strands:
        in_barrel.update(range(first, last + 1))
    wall = np.array([placed[i] for i, a in enumerate(atoms)
                     if a["name"] == "CA" and a["seq"] in in_barrel])
    wall_radius = np.linalg.norm(wall[:, [0, 2]], axis=1)

    order = {s: i / (len(numbers) - 1) for i, s in enumerate(numbers)}
    out_atoms = [{
        "el": a["el"],
        "pos": [round(float(v), 4) for v in placed[i]],
        "seq": a["seq"],
        "res": a["res"],
        "name": a["name"],
        "t": round(order[a["seq"]], 6),
        "chromophore": a["res"] == chromo_name,
        "pi": a["res"] == chromo_name and a["name"] in pi_names,
    } for i, a in enumerate(atoms)]

    elements = {}
    for a in atoms:
        elements[a["el"]] = elements.get(a["el"], 0) + 1

    return {
        "key": key,
        "label": spec["label"],
        "pdb": spec["pdb"],
        "source": spec["source"],
        "spectraSource": spec["spectra"],
        "excitation": spec["excitation"],
        "emission": spec["emission"],
        "chromophoreName": chromo_name,
        "chromophoreResidue": chromo_seq,
        "formedFrom": spec["formedFrom"],
        "missingResidues": missing,
        "chromophoreAtoms": chromo_idx,
        "chromophoreAtomCount": len(chromo_idx),
        "chromophoreCenter": [round(float(v), 4) for v in placed[chromo_idx].mean(axis=0)],
        "piAtoms": pi_idx,
        "piAtomCount": len(pi_idx),
        "piBonds": sorted(pi_bonds),
        "extraPi": EXTRA_PI[key],
        "acylimineLength": round(length(*ACYLIMINE), 3),
        "residueCount": len(residues),
        "firstResidue": numbers[0],
        "lastResidue": numbers[-1],
        "selenomethionines": sorted(s for s, r in residues if r == "MSE"),
        "elements": elements,
        "strandCount": len(strands),
        "wallRadius": round(float(wall_radius.mean()), 2),
        "wallRadiusSpread": round(float(wall_radius.std()), 2),
        "barrelLength": round(float(placed[:, 1].max() - placed[:, 1].min()), 2),
        "atoms": out_atoms,
        "bonds": [[i, j] for i, j in bonds],
    }


def fold_agreement(a, b):
    """How well the two folds match once both are in the common frame: for each
    of b's alpha carbons, the distance to the nearest of a's. These proteins
    share only about a quarter of their sequence, so this measures the fold
    rather than the sequence."""
    pa = np.array([at["pos"] for at in a["atoms"] if at["name"] == "CA"])
    pb = np.array([at["pos"] for at in b["atoms"] if at["name"] == "CA"])
    d = np.linalg.norm(pb[:, None, :] - pa[None, :, :], axis=2).min(axis=1)
    return {
        "median": round(float(np.median(d)), 2),
        "mean": round(float(d.mean()), 2),
        "within2A": round(float((d < 2.0).mean()), 3),
        "within3A": round(float((d < 3.0).mean()), 3),
    }


def main():
    gfp = build("GFP")
    cherry = build("mCherry")

    # The acylimine, which is the entire point: single in GFP, double in
    # mCherry. Textbook C-N is 1.47 A and C=N is 1.28 A.
    assert gfp["acylimineLength"] > 1.42, \
        f"GFP's N1-CA1 should be a single bond, measured {gfp['acylimineLength']}"
    assert cherry["acylimineLength"] < 1.36, \
        f"mCherry's N1-CA1 should be a double bond, measured {cherry['acylimineLength']}"
    assert cherry["piAtomCount"] > gfp["piAtomCount"], "mCherry's pi system should be the longer one"
    # Redder light means lower energy means a longer wavelength.
    assert cherry["emission"] > gfp["emission"], "mCherry should emit at the longer wavelength"

    agreement = fold_agreement(gfp, cherry)
    data = {
        "note": "GFP (1EMA, S65T) and mCherry (2H5Q), aligned in a common barrel frame",
        "proteins": [gfp, cherry],
        "foldAgreement": agreement,
        "acylimine": {
            "bond": list(ACYLIMINE),
            "GFP": gfp["acylimineLength"],
            "mCherry": cherry["acylimineLength"],
            "singleReference": 1.47,
            "doubleReference": 1.28,
        },
    }
    json.dump(data, open(OUT, "w"), indent=0)

    for p in (gfp, cherry):
        print(f"{p['label']:12s} {p['pdb']}  {len(p['atoms'])} heavy atoms, "
              f"{p['residueCount']} residues ({p['firstResidue']}..{p['lastResidue']}), "
              f"{p['strandCount']} strands")
        print(f"             chromophore {p['chromophoreName']} {p['chromophoreResidue']}, "
              f"{p['chromophoreAtomCount']} atoms, from {'-'.join(p['formedFrom'])}, "
              f"residues {p['missingResidues']} absent")
        print(f"             pi system {p['piAtomCount']} atoms"
              + (f" (+{', '.join(p['extraPi'])} via the acylimine)" if p["extraPi"] else "")
              + f", N1-CA1 = {p['acylimineLength']:.3f} A")
        print(f"             ex {p['excitation']} nm, em {p['emission']} nm   [{p['spectraSource']}]")
        if p["selenomethionines"]:
            print(f"             selenomethionine at {p['selenomethionines']}")
    print(f"\nacylimine: GFP {gfp['acylimineLength']:.3f} A (single, ref 1.47) vs "
          f"mCherry {cherry['acylimineLength']:.3f} A (double, ref 1.28)")
    print(f"fold agreement: {agreement['within3A']:.0%} of mCherry's CAs within 3 A of a GFP CA "
          f"(median {agreement['median']} A)")
    print(f"wrote {os.path.relpath(OUT)}")


if __name__ == "__main__":
    main()
