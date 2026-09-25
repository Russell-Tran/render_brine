"""Builds Resources/dna.json from the crystal structure in Resources/1BNA.pdb.

1BNA is the Dickerson–Drew dodecamer, CGCGAATTCGCG paired with itself: the
first X-ray structure of a full turn of B-DNA (Drew et al., PNAS 78:2179,
1981; 1.9 Å). Downloaded from https://files.rcsb.org/download/1BNA.pdb.

What this does:
  1. Reads the 486 DNA atoms (the 80 crystal waters are left out) and keeps
     their positions exactly.
  2. Joins atoms closer than their covalent radii allow (+0.45 Å), then sets
     double bonds and charges by atom name from each base's Kekulé structure.
     Each phosphate is −1 at pH 7 (one O⁻, one P=O; really the charge is
     shared between the two).
  3. Adds hydrogens at standard positions (X-rays don't see them) with RDKit.
  4. Turns the molecule so its helix axis (the line that best fits the base-pair
     centers) lies along x, centered on the origin.
  5. Records the Watson–Crick hydrogen bonds and the measured helix: rise and
     twist per base pair.

Run: python3 Tools/build_dna.py      (needs RDKit: pip3 install --user rdkit)
"""

import json
import os

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdchem
from rdkit.Geometry import Point3D

HERE = os.path.dirname(os.path.abspath(__file__))
PDB = os.path.join(HERE, "..", "Resources", "1BNA.pdb")
OUT = os.path.join(HERE, "..", "Resources", "dna.json")

# Covalent radii (Å), Cordero et al., Dalton Trans. (2008) 2832.
COVALENT = {"C": 0.76, "N": 0.71, "O": 0.66, "P": 1.07}

# Double bonds in each base's Kekulé structure, by PDB atom name.
DOUBLE = {
    "DA": [("C4", "C5"), ("C6", "N1"), ("C2", "N3"), ("C8", "N7")],
    "DG": [("C6", "O6"), ("C4", "C5"), ("C2", "N3"), ("C8", "N7")],
    "DC": [("C2", "O2"), ("N3", "C4"), ("C5", "C6")],
    "DT": [("C2", "O2"), ("C4", "O4"), ("C5", "C6")],
}

# Watson–Crick hydrogen bonds as (donor residue's atom, acceptor residue's
# atom), keyed by (base on strand A, base on strand B).
WATSON_CRICK = {
    ("DG", "DC"): [("B", "N4", "A", "O6"), ("A", "N1", "B", "N3"), ("A", "N2", "B", "O2")],
    ("DC", "DG"): [("A", "N4", "B", "O6"), ("B", "N1", "A", "N3"), ("B", "N2", "A", "O2")],
    ("DA", "DT"): [("A", "N6", "B", "O4"), ("B", "N3", "A", "N1")],
    ("DT", "DA"): [("B", "N6", "A", "O4"), ("A", "N3", "B", "N1")],
}

# Nucleoside formulas (C, H, N, O), for the independent formula check.
NUCLEOSIDE = {"DC": (9, 13, 3, 4), "DG": (10, 13, 5, 4), "DA": (10, 13, 5, 3), "DT": (10, 14, 2, 5)}


def read_pdb(path):
    atoms = []
    for line in open(path):
        if not line.startswith("ATOM"):
            continue
        atoms.append({
            "name": line[12:16].strip(),
            "res": line[17:20].strip(),
            "chain": line[21],
            "seq": int(line[22:26]),
            "pos": np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])]),
            "el": line[76:78].strip(),
        })
    return atoms


def build_molecule(atoms):
    m = Chem.RWMol()
    for a in atoms:
        atom = Chem.Atom(a["el"])
        atom.SetNoImplicit(False)
        m.AddAtom(atom)
    n = len(atoms)
    pos = np.array([a["pos"] for a in atoms])
    for i in range(n):
        for j in range(i + 1, n):
            limit = COVALENT[atoms[i]["el"]] + COVALENT[atoms[j]["el"]] + 0.45
            if np.linalg.norm(pos[i] - pos[j]) < limit:
                m.AddBond(i, j, rdchem.BondType.SINGLE)
    index = {(a["chain"], a["seq"], a["name"]): i for i, a in enumerate(atoms)}
    residues = sorted({(a["chain"], a["seq"], a["res"]) for a in atoms})
    for chain, seq, res in residues:
        for x, y in DOUBLE[res]:
            m.GetBondBetweenAtoms(index[(chain, seq, x)], index[(chain, seq, y)]).SetBondType(rdchem.BondType.DOUBLE)
        if (chain, seq, "P") in index:
            m.GetBondBetweenAtoms(index[(chain, seq, "P")], index[(chain, seq, "OP2")]).SetBondType(rdchem.BondType.DOUBLE)
            m.GetAtomWithIdx(index[(chain, seq, "OP1")]).SetFormalCharge(-1)
    conf = Chem.Conformer(n)
    for i, p in enumerate(pos):
        conf.SetAtomPosition(i, Point3D(*p))
    m.AddConformer(conf, assignId=True)
    mol = m.GetMol()
    Chem.SanitizeMol(mol)
    return mol, index, residues


def expected_formula(residues):
    """The duplex's formula from its sequence: nucleosides, plus one phosphate
    link per neighboring pair (+P +2 O −1 H), each link ionized (−1 H, −1 charge)."""
    c = h = nn = o = p = charge = 0
    for chain in "AB":
        strand = [r for r in residues if r[0] == chain]
        for _, _, res in strand:
            dc, dh, dn, do = NUCLEOSIDE[res]
            c, h, nn, o = c + dc, h + dh, nn + dn, o + do
        links = len(strand) - 1
        p, o, h, charge = p + links, o + 2 * links, h - 2 * links, charge - links
    return {"C": c, "H": h, "N": nn, "O": o, "P": p}, charge


def main():
    atoms = read_pdb(PDB)
    mol, index, residues = build_molecule(atoms)
    molh = Chem.AddHs(mol, addCoords=True)
    # Sanitizing marks the bases aromatic (bond order 1.5); draw them as the
    # Kekulé structures set above, with real single and double bonds.
    Chem.Kekulize(molh, clearAromaticFlags=True)
    conf = molh.GetConformer()
    pos = np.array([list(conf.GetAtomPosition(i)) for i in range(molh.GetNumAtoms())])

    # Formula and charge, against the sequence.
    counts = {}
    for a in molh.GetAtoms():
        counts[a.GetSymbol()] = counts.get(a.GetSymbol(), 0) + 1
    charge = Chem.GetFormalCharge(molh)
    want, want_charge = expected_formula(residues)
    assert counts == want and charge == want_charge, (counts, charge, want, want_charge)

    # Base pairs: strand A residue i with strand B residue 25 − i.
    a_res = [r for r in residues if r[0] == "A"]
    b_res = {r[1]: r for r in residues if r[0] == "B"}
    pairs = [(ra, b_res[25 - ra[1]]) for ra in a_res]

    # Helix axis: the best-fit line through the base-pair centers (midpoints of
    # the two C1' atoms).
    centers = np.array([(pos[index[("A", ra[1], "C1'")]] + pos[index[("B", rb[1], "C1'")]]) / 2 for ra, rb in pairs])
    mid = centers.mean(axis=0)
    _, _, vt = np.linalg.svd(centers - mid)
    axis = vt[0] if np.dot(vt[0], centers[-1] - centers[0]) > 0 else -vt[0]
    # A rotation taking the axis to +x.
    x = axis
    y = np.cross([0.0, 0.0, 1.0], x)
    y /= np.linalg.norm(y)
    z = np.cross(x, y)
    r = np.array([x, y, z])
    placed = np.einsum("ij,kj->ik", pos - mid, r)
    # Center along the axis on the middle of the base pairs.
    placed[:, 0] -= np.einsum("ij,j->i", centers - mid, r[0]).mean()

    # Rise and twist per step, from the base-pair centers and C1'→C1' vectors
    # projected on the plane across the axis. Positive twist = right-handed.
    rises, twists = [], []
    c1 = [(placed[index[("A", ra[1], "C1'")]], placed[index[("B", rb[1], "C1'")]]) for ra, rb in pairs]
    for k in range(len(pairs) - 1):
        ca = (c1[k][0] + c1[k][1]) / 2
        cb = (c1[k + 1][0] + c1[k + 1][1]) / 2
        rises.append(float(cb[0] - ca[0]))
        v1 = (c1[k][1] - c1[k][0]) * np.array([0, 1, 1])
        v2 = (c1[k + 1][1] - c1[k + 1][0]) * np.array([0, 1, 1])
        twists.append(float(np.degrees(np.arctan2(np.dot([1, 0, 0], np.cross(v1, v2)), np.dot(v1, v2)))))

    # Watson–Crick hydrogen bonds: donor, its H nearest the acceptor, acceptor.
    hbonds = []
    for ra, rb in pairs:
        side = {"A": ra, "B": rb}
        for dside, dname, aside, aname in WATSON_CRICK[(ra[2], rb[2])]:
            d = index[(side[dside][0], side[dside][1], dname)]
            acc = index[(side[aside][0], side[aside][1], aname)]
            hs = [nb.GetIdx() for nb in molh.GetAtomWithIdx(d).GetNeighbors() if nb.GetSymbol() == "H"]
            h = min(hs, key=lambda i: np.linalg.norm(placed[i] - placed[acc]))
            hbonds.append({"donor": d, "h": h, "acceptor": acc,
                           "da": float(np.linalg.norm(placed[d] - placed[acc])),
                           "pair": f"{ra[2][1]}{ra[1]}–{rb[2][1]}{rb[1]}"})

    out_atoms = []
    for i, a in enumerate(molh.GetAtoms()):
        if i < len(atoms):
            src = atoms[i]
            label = f"{src['chain']}{src['seq']}.{src['name']}"
        else:
            parent = a.GetNeighbors()[0].GetIdx()
            src = atoms[parent]
            label = f"{src['chain']}{src['seq']}.H{i}"
        out_atoms.append({"el": a.GetSymbol(), "pos": [round(float(v), 4) for v in placed[i]],
                          "charge": a.GetFormalCharge(), "label": label})
    bonds = [[b.GetBeginAtomIdx(), b.GetEndAtomIdx(), int(b.GetBondTypeAsDouble())] for b in molh.GetBonds()]

    data = {
        "source": "PDB 1BNA, Drew et al., PNAS 78:2179 (1981); hydrogens added by RDKit",
        "sequence": "".join(r[2][1] for r in a_res),
        "formula": counts,
        "charge": charge,
        "rise": rises,
        "twist": twists,
        "atoms": out_atoms,
        "bonds": bonds,
        "hbonds": hbonds,
    }
    json.dump(data, open(OUT, "w"), indent=0)
    turn = 360 / np.mean(twists)
    print(f"{data['sequence']}: {len(out_atoms)} atoms {counts} charge {charge:+d}, {len(bonds)} bonds, {len(hbonds)} H-bonds")
    print(f"rise {np.mean(rises):.2f} Å (range {min(rises):.2f}–{max(rises):.2f}), twist {np.mean(twists):.1f}° "
          f"(range {min(twists):.1f}–{max(twists):.1f}) → {turn:.1f} bp per turn")
    print(f"H-bond donor–acceptor {min(h['da'] for h in hbonds):.2f}–{max(h['da'] for h in hbonds):.2f} Å")
    ext = placed.max(axis=0) - placed.min(axis=0)
    print(f"extent x {ext[0]:.1f} y {ext[1]:.1f} z {ext[2]:.1f} Å; wrote {os.path.relpath(OUT)}")


if __name__ == "__main__":
    main()
