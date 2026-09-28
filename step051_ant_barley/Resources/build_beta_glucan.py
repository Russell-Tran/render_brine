#!/usr/bin/env python3
"""Step 51: builds a short piece of barley mixed-linkage beta-glucan from two
PubChem 3-D conformers, and writes it as an MDL molfile.

    python3 build_beta_glucan.py        (run in Resources/; needs numpy)

The piece is the one Purushotham et al. 2022 (Sci Adv 8: eadd1596) describe
barley beta-glucan as made of: a cellotriosyl and a cellotetraosyl unit
separated by a single (1,3)-beta linkage —

    Glc b1-4 Glc b1-4 Glc b1-3 Glc b1-4 Glc b1-4 Glc b1-4 Glc
    r1       r2       r3       r4       r5       r6       r7 (reducing end, beta)

Nothing is invented: every residue, and every glycosidic linkage's geometry,
is copied from a PubChem conformer —
    beta-cellobiose,    CID 10712   (Glc b1-4 Glc)  -> the five 1-4 links
    beta-laminaribiose, CID 5287770 (Glc b1-3 Glc)  -> the one 1-3 link
Each link is laid down by superposing the template's acceptor ring (C1-C5,
O5) on the residue already placed (least squares, Kabsch) and carrying the
template's donor residue and bridging oxygen with it; the acceptor's own
O4-H (or O3-H) is replaced by the template's bridging O. The ring fits are
printed; the tests check the bonds and angles that result.
"""
import numpy as np

def read_sdf(path):
    lines = open(path).read().split("\n")
    na, nb = int(lines[3][0:3]), int(lines[3][3:6])
    el, xyz = [], []
    for l in lines[4:4 + na]:
        f = l.split()
        xyz.append([float(f[0]), float(f[1]), float(f[2])]); el.append(f[3])
    bonds = []
    for l in lines[4 + na:4 + na + nb]:
        bonds.append((int(l[0:3]) - 1, int(l[3:6]) - 1, int(l[6:9])))
    return el, np.array(xyz), bonds

def neighbours(n, bonds):
    nb = [[] for _ in range(n)]
    for a, b, _ in bonds:
        nb[a].append(b); nb[b].append(a)
    return nb

def label(el, nb):
    """Split a disaccharide into its two residues and name every heavy atom
    (C1..C6, O1..O6, with the bridging O named in the acceptor as O3/O4)
    and every H by the heavy atom it is on."""
    n = len(el)
    # Ring oxygens: O bonded to two C that are both ring carbons -> find by
    # 6-rings. Simple: an O with two C neighbours, one of which has 2 O neighbours
    # (anomeric) and the other a CH2OH neighbour (C5).
    def cs(i): return [j for j in nb[i] if el[j] == "C"]
    def os_(i): return [j for j in nb[i] if el[j] == "O"]
    ring_o = []
    for i in range(n):
        if el[i] != "O" or len(cs(i)) != 2: continue
        a, b = cs(i)
        # C5 is bonded to a CH2 carbon (C6: one C neighbour, two H)
        def has_c6(c): return any(len(cs(k)) == 1 and sum(el[m] == "H" for m in nb[k]) == 2 for k in cs(c))
        if has_c6(a) or has_c6(b): ring_o.append(i)
    assert len(ring_o) == 2, ring_o
    residues = []
    for o5 in ring_o:
        a, b = cs(o5)
        def has_c6(c): return any(len(cs(k)) == 1 and sum(el[m] == "H" for m in nb[k]) == 2 for k in cs(c))
        c5, c1 = (a, b) if has_c6(a) else (b, a)
        names = {"O5": o5, "C1": c1, "C5": c5}
        names["C6"] = [k for k in cs(c5) if len(cs(k)) == 1][0]
        names["C2"] = [k for k in cs(c1)][0]
        names["C3"] = [k for k in cs(names["C2"]) if k != c1][0]
        names["C4"] = [k for k in cs(names["C3"]) if k != names["C2"]][0]
        assert c5 in cs(names["C4"])
        for c, o in [("C1", "O1"), ("C2", "O2"), ("C3", "O3"), ("C4", "O4"), ("C6", "O6")]:
            ox = [k for k in os_(names[c]) if k != o5]
            assert len(ox) == 1
            names[o] = ox[0]
        residues.append(names)
    # Donor: the residue whose O1 is also bonded to a carbon of the other.
    r0, r1 = residues
    if set(cs(r0["O1"])) - {r0["C1"]}:
        donor, acceptor = r0, r1
    else:
        donor, acceptor = r1, r0
    bridge = donor["O1"]
    link = [k for k, v in acceptor.items() if v == bridge][0]   # "O3" or "O4"
    del donor["O1"]
    return donor, acceptor, bridge, link

def residue_atoms(names, el, nb, skip=()):
    """Heavy atoms of a residue plus their hydrogens, in a fixed order."""
    order = ["C1", "C2", "C3", "C4", "C5", "C6", "O1", "O2", "O3", "O4", "O5", "O6"]
    out = []
    for k in order:
        if k not in names or k in skip: continue
        i = names[k]
        out.append((k, i))
        hs = [m for m in nb[i] if el[m] == "H"]
        for j, m in enumerate(hs):
            out.append(("H" + k + ("" if j == 0 else "'" * j), m))
    return out

def kabsch(P, Q):
    """R, t minimising |R P + t - Q|."""
    pc, qc = P.mean(0), Q.mean(0)
    H = (P - pc).T @ (Q - qc)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    return R, qc - R @ pc

RING = ["C1", "C2", "C3", "C4", "C5", "O5"]

def main():
    templates = {}
    for key, path in [("4", "cellobiose_beta_CID10712_3d.sdf"), ("3", "laminaribiose_beta_CID5287770_3d.sdf")]:
        el, xyz, bonds = read_sdf(path)
        nb = neighbours(len(el), bonds)
        donor, acceptor, bridge, link = label(el, nb)
        assert link == "O" + key, (path, link)
        templates[key] = (el, xyz, nb, donor, acceptor, bridge)

    # Residues: dicts name -> (element, xyz). Start from the reducing end:
    # r7 and r6 as the cellobiose template gives them.
    el, xyz, nb, donor, acceptor, bridge = templates["4"]
    res = {}
    res[7] = {k: (el[i], xyz[i].copy()) for k, i in residue_atoms(acceptor, el, nb)}
    res[6] = {k: (el[i], xyz[i].copy()) for k, i in residue_atoms(donor, el, nb)}
    links = {5: "4", 4: "4", 3: "3", 2: "4", 1: "4"}   # residue r links 1->link into r+1
    fits = []
    for r in [5, 4, 3, 2, 1]:
        key = links[r]
        el, xyz, nb, donor, acceptor, bridge = templates[key]
        placed = np.array([res[r + 1][k][1] for k in RING])
        tmpl = np.array([xyz[acceptor[k]] for k in RING])
        R, t = kabsch(tmpl, placed)
        rms = np.sqrt((((tmpl @ R.T + t) - placed) ** 2).sum(1).mean())
        fits.append((r, key, rms))
        # The acceptor's free O-H at the linked position gives way to the
        # template's bridging O.
        o = "O" + key
        del res[r + 1][o]
        del res[r + 1]["H" + o]
        res[r + 1][o] = ("O", R @ xyz[bridge] + t)
        res[r] = {k: (el[i], R @ xyz[i] + t) for k, i in residue_atoms(donor, el, nb)}
    for r, key, rms in fits:
        print(f"r{r} -> r{r + 1}: 1-{key} link, acceptor ring fit {rms:.3f} A rms")

    # Atoms and bonds.
    atoms, index = [], {}
    for r in range(1, 8):
        for k, (e, p) in res[r].items():
            index[(r, k)] = len(atoms)
            atoms.append((e, p))
    bonds = []
    ringbonds = [("C1", "C2"), ("C2", "C3"), ("C3", "C4"), ("C4", "C5"), ("C5", "O5"), ("O5", "C1"), ("C5", "C6")]
    for r in range(1, 8):
        names = res[r]
        for a, b in ringbonds: bonds.append((index[(r, a)], index[(r, b)]))
        for c, o in [("C1", "O1"), ("C2", "O2"), ("C3", "O3"), ("C4", "O4"), ("C6", "O6")]:
            if o in names: bonds.append((index[(r, c)], index[(r, o)]))
        for k in names:
            if k.startswith("H"): bonds.append((index[(r, k[1:].rstrip("'"))], index[(r, k)]))
    for r in [1, 2, 3, 4, 5, 6]:
        o = "O" + links.get(r, "4") if r != 6 else "O4"
        bonds.append((index[(r, "C1")], index[(r + 1, o)]))
    out = ["barley beta-glucan heptasaccharide, Glc b1-4 Glc b1-4 Glc b1-3 Glc b1-4 Glc b1-4 Glc b1-4 Glc",
           "  built by build_beta_glucan.py from PubChem CID 10712 and CID 5287770 3-D conformers", "",
           f"{len(atoms):3d}{len(bonds):3d}  0     0  0  0  0  0  0999 V2000"]
    for e, p in atoms:
        out.append(f"{p[0]:10.4f}{p[1]:10.4f}{p[2]:10.4f} {e:<3} 0  0  0  0  0  0  0  0  0  0  0  0")
    for a, b in bonds:
        out.append(f"{a + 1:3d}{b + 1:3d}  1  0  0  0  0")
    out.append("M  END")
    out.append("$$$$")
    open("beta_glucan_G4G4G3G4G4G4G.sdf", "w").write("\n".join(out) + "\n")
    # A clash check, for the builder's own eyes (the tests repeat it).
    P = np.array([p for _, p in atoms]); E = [e for e, _ in atoms]
    bonded = set((min(a, b), max(a, b)) for a, b in bonds)
    nbl = neighbours(len(atoms), [(a, b, 1) for a, b in bonds])
    worst = 9
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            if (i, j) in bonded or set(nbl[i]) & set(nbl[j]): continue
            d = np.linalg.norm(P[i] - P[j])
            if d < worst: worst, pair = d, (i, j, E[i], E[j])
    print(f"{len(atoms)} atoms, {len(bonds)} bonds; closest non-bonded pair (1-3 excluded) {worst:.2f} A {pair}")

main()
