"""Builds Resources/gfp.json from the crystal structure in Resources/1EMA.pdb.

1EMA is green fluorescent protein from Aequorea victoria, the S65T variant
(Ormo, Cubitt, Kallio, Gross, Tsien & Remington, Science 273:1392, 1996;
1.9 A). Downloaded from https://files.rcsb.org/download/1EMA.pdb.

What this does:
  1. Reads the 1,771 non-water heavy atoms (the 95 crystal waters are left
     out) and keeps their positions exactly.
  2. Joins atoms closer than their covalent radii allow (+0.45 A), as step 8
     did. Bonds are only drawn for the chromophore, but the whole protein's
     bonds are useful for the tests.
  3. Finds the beta barrel's axis from the backbone alpha carbons, and turns
     the molecule so that axis lies along y, centered on the origin.
  4. Records which atoms belong to the chromophore, and where along the chain
     each residue sits, so the renderer can color from the N terminus to the C
     terminus.

Two things worth knowing about this structure, both checked below rather than
assumed:

  * The chromophore is a single residue, CRO 66, of 22 atoms. GFP makes it
    out of three consecutive amino acids of its own chain, which fuse into
    one unit, so residues 65 and 67 are simply absent from the numbering.
    That gap is the most direct evidence of the chromophore story there is.

  * Four methionines are selenomethionine (MSE 78, 88, 153, 218), holding
    selenium rather than sulfur. That is a crystallographer's trick for
    solving the phase problem, not a feature of the real protein.

No hydrogens are added, unlike step 8. X-rays at 1.9 A cannot see them, and
here nothing depends on them: there are no hydrogen bonds to draw, a protein
surface drawn at van der Waals radii is conventionally heavy-atom, and the
~1,800 extra spheres would double the cost of the ambient occlusion that
makes the shape readable in the first place.

Run: python3 Tools/build_gfp.py
"""

import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PDB = os.path.join(HERE, "..", "Resources", "1EMA.pdb")
OUT = os.path.join(HERE, "..", "Resources", "gfp.json")

# Covalent radii (A), Cordero et al., Dalton Trans. (2008) 2832.
COVALENT = {"C": 0.76, "N": 0.71, "O": 0.66, "S": 1.05, "SE": 1.20}

CHROMOPHORE = "CRO"
SELENOMETHIONINE = "MSE"


def read_pdb(path):
    """Every non-water atom, in file order."""
    atoms = []
    for line in open(path):
        if line[:6] not in ("ATOM  ", "HETATM"):
            continue
        resn = line[17:20].strip()
        if resn == "HOH":
            continue
        assert line[16] == " ", "this structure should have no alternate locations"
        atoms.append({
            "name": line[12:16].strip(),
            "res": resn,
            "seq": int(line[22:26]),
            "pos": np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])]),
            "el": line[76:78].strip().upper(),
        })
    return atoms


def find_bonds(atoms):
    """Pairs closer than their covalent radii allow. A coarse grid keeps this
    from being 1,771 squared."""
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
    """The beta strands the depositors annotated. GFP's barrel is famously
    eleven-stranded, and the file says so rather than us assuming it."""
    strands = []
    for line in open(path):
        if line.startswith("SHEET"):
            strands.append((int(line[22:26]), int(line[33:37])))
    return strands


def barrel_frame(atoms):
    """The beta barrel's axis, from the spread of the backbone alpha carbons.
    A barrel is roughly a cylinder, so its long axis is the direction the
    alpha carbons spread out along most."""
    ca = np.array([a["pos"] for a in atoms if a["name"] == "CA"])
    assert len(ca) > 200, f"expected a backbone, found {len(ca)} alpha carbons"
    center = ca.mean(axis=0)
    _, _, vt = np.linalg.svd(ca - center)
    axis = vt[0]
    # Point the axis from the barrel's first half toward its second, so the
    # choice is reproducible rather than up to the sign numpy happens to pick.
    half = len(ca) // 2
    if np.dot(axis, ca[half:].mean(axis=0) - ca[:half].mean(axis=0)) < 0:
        axis = -axis
    return center, axis, ca


def rotation_taking(axis, to):
    """A rotation matrix that takes the unit vector `axis` onto `to`."""
    a = axis / np.linalg.norm(axis)
    b = np.array(to, dtype=float)
    b /= np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else -np.eye(3)
    kx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + kx + kx @ kx / (1 + c)


def main():
    atoms = read_pdb(PDB)
    elements = {}
    for a in atoms:
        elements[a["el"]] = elements.get(a["el"], 0) + 1

    residues = sorted({(a["seq"], a["res"]) for a in atoms})
    numbers = [s for s, _ in residues]
    names = {s: r for s, r in residues}

    # The chromophore, checked rather than assumed.
    chromo = [s for s, r in residues if r == CHROMOPHORE]
    assert chromo == [66], f"expected one chromophore at 66, found {chromo}"
    assert 65 not in names and 67 not in names, \
        "65 and 67 should be absent: they fused into the chromophore"
    chromo_atoms = [i for i, a in enumerate(atoms) if a["res"] == CHROMOPHORE]
    assert len(chromo_atoms) == 22, f"chromophore should have 22 atoms, has {len(chromo_atoms)}"

    selenomet = sorted(s for s, r in residues if r == SELENOMETHIONINE)

    center, axis, ca = barrel_frame(atoms)
    rot = rotation_taking(axis, [0, 1, 0])
    placed = np.einsum("ij,kj->ik", np.array([a["pos"] for a in atoms]) - center, rot)
    placed -= placed.mean(axis=0)

    # How far along the chain each residue sits, 0 at the N terminus to 1 at
    # the C terminus, for the color spectrum.
    order = {s: i / (len(numbers) - 1) for i, s in enumerate(numbers)}

    chromo_center = placed[chromo_atoms].mean(axis=0)
    extent = placed.max(axis=0) - placed.min(axis=0)

    # The barrel wall itself: the alpha carbons of the annotated strands. The
    # spread of their distance from the axis is how circular the barrel is.
    strands = read_strands(PDB)
    assert len(strands) == 11, f"GFP's barrel should have 11 strands, the file says {len(strands)}"
    in_barrel = set()
    for first, last in strands:
        in_barrel.update(range(first, last + 1))
    wall = np.array([placed[i] for i, a in enumerate(atoms)
                     if a["name"] == "CA" and a["seq"] in in_barrel])
    wall_radius = np.linalg.norm(wall[:, [0, 2]], axis=1)
    across = np.linalg.norm(placed[:, [0, 2]], axis=1)

    out_atoms = []
    for i, a in enumerate(atoms):
        out_atoms.append({
            "el": a["el"],
            "pos": [round(float(v), 4) for v in placed[i]],
            "seq": a["seq"],
            "res": a["res"],
            "name": a["name"],
            "t": round(order[a["seq"]], 6),
            "chromophore": a["res"] == CHROMOPHORE,
        })
    bonds = find_bonds(atoms)

    data = {
        "source": "PDB 1EMA, Ormo et al., Science 273:1392 (1996), 1.9 A; waters and hydrogens omitted",
        "variant": "S65T",
        "residueCount": len(residues),
        "firstResidue": numbers[0],
        "lastResidue": numbers[-1],
        "missingForChromophore": [65, 67],
        "chromophoreResidue": 66,
        "chromophoreAtoms": chromo_atoms,
        "chromophoreCenter": [round(float(v), 4) for v in chromo_center],
        "selenomethionines": selenomet,
        "elements": elements,
        "strandCount": len(strands),
        "strands": [[a, b] for a, b in strands],
        "barrelLength": round(float(extent[1]), 2),
        "barrelRadius": round(float(np.percentile(across, 95)), 2),
        "wallLength": round(float(wall[:, 1].max() - wall[:, 1].min()), 2),
        "wallRadius": round(float(wall_radius.mean()), 2),
        "wallRadiusSpread": round(float(wall_radius.std()), 2),
        "atoms": out_atoms,
        "bonds": [[i, j] for i, j in bonds],
    }
    json.dump(data, open(OUT, "w"), indent=0)

    print(f"GFP {data['variant']}: {len(out_atoms)} heavy atoms, {len(residues)} residues "
          f"({numbers[0]}..{numbers[-1]}), {len(bonds)} bonds")
    print(f"  elements {elements}")
    print(f"  chromophore CRO 66, {len(chromo_atoms)} atoms, "
          f"{np.linalg.norm(chromo_center):.2f} A from the center")
    print(f"  residues 65 and 67 absent: fused into the chromophore")
    print(f"  selenomethionine at {selenomet}")
    print(f"  {len(strands)} beta strands; their wall is {data['wallLength']:.1f} A long and "
          f"{2*data['wallRadius']:.1f} A across, radius varying only +-{data['wallRadiusSpread']:.1f} A")
    print(f"  whole molecule {data['barrelLength']:.1f} A long, {2*data['barrelRadius']:.1f} A across")
    print(f"  extent {extent[0]:.1f} x {extent[1]:.1f} x {extent[2]:.1f} A")
    print(f"wrote {os.path.relpath(OUT)}")


if __name__ == "__main__":
    main()
