"""Builds Resources/timeline.json: keyframes for the two glycolysis GIFs.

For each GIF this writes a list of keyframes. Each keyframe has every atom's
position, visibility and charge, the bonds with their orders, the tokens
(ATP, ADP, NAD⁺, NADH) and the caption. The Swift renderer interpolates
smoothly between keyframes.

3D shapes: each species is embedded with RDKit (ETKDG) and relaxed with the
MMFF94 force field. Atoms far from the reaction keep their previous
positions as a starting point, and each new shape is rotated onto the
previous one (Kabsch), so the motion between steps is smooth.

Run: python3 Tools/build_timeline.py
"""

import copy
import json
import os

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D

import pathway as pw

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "Resources", "timeline.json")


# ---------------------------------------------------------------- 3D shapes

def bond_set(sp):
    out = {}
    for b in sp.m.GetBonds():
        a1 = b.GetBeginAtom().GetProp("lbl")
        a2 = b.GetEndAtom().GetProp("lbl")
        out[tuple(sorted((a1, a2)))] = b.GetBondTypeAsDouble()
    return out


def kabsch(moving, fixed):
    """Rotation R and translation t that best map `moving` points onto `fixed`."""
    mc, fc = moving.mean(axis=0), fixed.mean(axis=0)
    h = (moving - mc).T @ (fixed - fc)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1, 1, d]) @ u.T
    return r, fc - r @ mc


def relax(m, conf_id=-1):
    """MMFF94 with a distance-dependent dielectric (ε = 4r), a standard stand-in
    for water: in vacuum the charged phosphates pull OH hydrogens into
    unrealistically short (~1.5 Å) hydrogen bonds."""
    props = AllChem.MMFFGetMoleculeProperties(m)
    props.SetMMFFDielectricModel(2)   # 2 = distance-dependent
    props.SetMMFFDielectricConstant(4.0)
    ff = AllChem.MMFFGetMoleculeForceField(m, props, confId=conf_id)
    ff.Minimize(maxIts=5000)


def path_clear(prev, new, m):
    """True if moving in a straight (smoothstepped) line from `prev` to `new`
    never brings two heavy atoms two or more bonds apart within 1.6 Å (they
    normally sit 2.4 Å or more apart)."""
    hops = Chem.GetDistanceMatrix(m)
    idx = {a.GetProp("lbl"): a.GetIdx() for a in m.GetAtoms() if a.GetSymbol() != "H"}
    shared = [l for l in idx if l in prev and l in new]
    far = [(x, y) for x in range(len(shared)) for y in range(x + 1, len(shared))
           if hops[idx[shared[x]], idx[shared[y]]] >= 2]
    a = np.array([prev[l] for l in shared])
    b = np.array([new[l] for l in shared])
    for s in np.linspace(0, 1, 11):
        p = a + (b - a) * (s * s * (3 - 2 * s))
        if far and min(np.linalg.norm(p[x] - p[y]) for x, y in far) < 1.6:
            return False
    return True


def extended_conformer(m, seed, like=None):
    """For chains without a ring: of 20 relaxed shapes, the most stretched-out
    one, so no atoms end up stacked behind each other on screen. With `like`
    (the previous shape, by label), the one closest to it instead, so a small
    chemical change stays a small motion."""
    ids = AllChem.EmbedMultipleConfs(m, numConfs=20 if like is None else 60, randomSeed=seed)
    assert len(ids) > 0
    for cid in ids:
        relax(m, cid)
    heavy = [a.GetIdx() for a in m.GetAtoms() if a.GetSymbol() != "H"]
    labels = [a.GetProp("lbl") for a in m.GetAtoms()]
    best, best_score = ids[0], None
    for cid in ids:
        conf = m.GetConformer(cid)
        pts = np.array([list(conf.GetAtomPosition(i)) for i in heavy])
        if like is None:
            score = -float(((pts - pts.mean(axis=0)) ** 2).sum(axis=1).mean())
        else:
            shared = [k for k, i in enumerate(heavy) if labels[i] in like]
            fixed = np.array([like[labels[heavy[k]]] for k in shared])
            # Fit and score on the carbon backbone, so the carbons hold still
            # and the groups that change hands do the moving.
            carbon = [k for k, i in enumerate(shared) if labels[heavy[i]].startswith("C")]
            r, t = kabsch(pts[shared][carbon], fixed[carbon])
            moved = pts[shared] @ r.T + t
            score = float(np.sum((moved[carbon] - fixed[carbon]) ** 2)) + 0.1 * float(np.sum((moved - fixed) ** 2))
            aligned = {labels[heavy[k]]: q for k, q in zip(shared, moved)}
            if not path_clear(like, aligned, m):
                score += 1e6
        if best_score is None or score < best_score:
            best, best_score = cid, score
    return m.GetConformer(best)


def embed(sp, prev_sp=None, prev=None, seed=61, follow=False):
    """3D coordinates by label, aligned onto the previous species' coordinates.
    `follow`: for an open chain, pick the shape closest to the previous one."""
    m = Chem.Mol(sp.m)
    labels = [a.GetProp("lbl") for a in m.GetAtoms()]
    if m.GetRingInfo().NumRings() == 0:
        conf = extended_conformer(m, seed, like=prev if follow else None)
        coords = {lbl: np.array(conf.GetAtomPosition(i)) for i, lbl in enumerate(labels)}
        return align(coords, prev, m, labels)
    cmap = {}
    if prev_sp is not None:
        old, new = bond_set(prev_sp), bond_set(sp)
        changed = set()
        for key in set(old) | set(new):
            if old.get(key) != new.get(key):
                changed.update(key)
        near = set(changed)
        for lbl in changed:
            if lbl in labels:
                near.update(n.GetProp("lbl") for n in m.GetAtomWithIdx(labels.index(lbl)).GetNeighbors())
        for i, lbl in enumerate(labels):
            if lbl in prev and lbl not in near and m.GetAtomWithIdx(i).GetSymbol() != "H":
                cmap[i] = Point3D(*prev[lbl])
    ok = -1
    if cmap:
        ok = AllChem.EmbedMolecule(m, coordMap=cmap, randomSeed=seed, useRandomCoords=True)
    if ok < 0:
        params = AllChem.ETKDGv3()
        params.randomSeed = seed
        ok = AllChem.EmbedMolecule(m, params)
    assert ok >= 0, f"could not embed {sp.name}"
    relax(m)
    conf = m.GetConformer()
    coords = {lbl: np.array(conf.GetAtomPosition(i)) for i, lbl in enumerate(labels)}
    return align(coords, prev, m, labels)


def align(coords, prev, m, labels):
    """Rotates and shifts `coords` onto the previous shape's shared heavy atoms."""
    if prev is None:
        return coords
    shared = [l for l in labels if l in prev and m.GetAtomWithIdx(labels.index(l)).GetSymbol() != "H"]
    r, t = kabsch(np.array([coords[l] for l in shared]), np.array([prev[l] for l in shared]))
    return {l: r @ p + t for l, p in coords.items()}


def orient(coords):
    """Centers on the heavy atoms and turns the molecule face-on to the camera."""
    pts = np.array(list(coords.values()))
    c = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - c)
    r = vt  # rows: largest spread → x, next → y, least → z (toward the camera)
    return {l: r @ (p - c) for l, p in coords.items()}, (r, c)


def face_camera(coords, prev, sp):
    """Turns an open chain to lie flat facing the camera (its longest direction
    along x), picking whichever of the four flat orientations is closest to
    the previous shape so the turn stays small."""
    heavy = [a.GetProp("lbl") for a in sp.m.GetAtoms() if a.GetSymbol() != "H"]
    pts = np.array([coords[l] for l in heavy])
    c = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - c)
    if np.linalg.det(vt) < 0:
        vt[2] = -vt[2]      # keep it a rotation (a mirror would flip the molecule's handedness)
    target_c = np.mean([prev[l] for l in heavy if l in prev], axis=0)
    best, best_err = None, None
    for sx in (1, -1):
        for sy in (1, -1):
            r = np.diag([sx, sy, sx * sy]) @ vt
            if np.linalg.det(r) < 0:
                continue
            moved = {l: r @ (p - c) for l, p in coords.items()}
            # Keep the atoms both shapes share where they were (new atoms
            # mustn't drag the molecule sideways).
            shift = target_c - np.mean([moved[l] for l in heavy if l in prev], axis=0)
            moved = {l: q + shift for l, q in moved.items()}
            err = sum(float(np.sum((moved[l] - prev[l]) ** 2)) for l in heavy if l in prev)
            if best_err is None or err < best_err:
                best, best_err = moved, err
    return best


def map_coords(sp_from, coords_from, sp_to):
    """Copies coordinates between two species with the same structure but
    different labels (half A → half B), so the twins look identical."""
    match = sp_to.m.GetSubstructMatch(sp_from.m)
    assert len(match) == sp_from.m.GetNumAtoms(), "twins don't match"
    out = {}
    for i, j in enumerate(match):
        out[sp_to.m.GetAtomWithIdx(j).GetProp("lbl")] = coords_from[sp_from.m.GetAtomWithIdx(i).GetProp("lbl")]
    return out


# ---------------------------------------------------------------- keyframes

class Timeline:
    def __init__(self, name):
        self.name = name
        self.atoms = {}      # lbl -> {"el","pos","vis","charge","glow"}
        self.bonds = {}      # (a, b) -> order
        self.tokens = {}     # id -> {"text","pos","alpha"}
        self.caption = {"title": "", "enzyme": "", "equation": "", "ledger": [0, 0, 0]}
        self.keys = []
        self.camera_distance = 25.0   # Å; the payoff's two long halves need a little more room
        self.camera_swing = 12.0      # degrees the camera sways side to side

    def set_species(self, sp, coords, offset=(0, 0, 0)):
        off = np.array(offset, dtype=float)
        labels = set(sp.labels())
        coords = self.match_equivalents(sp, coords, off)
        for a in sp.m.GetAtoms():
            lbl = a.GetProp("lbl")
            self.atoms[lbl] = {"el": a.GetSymbol(), "pos": coords[lbl] + off, "vis": 1.0,
                               "charge": a.GetFormalCharge(), "glow": False}
        for key in list(self.bonds):
            if key[0] in labels or key[1] in labels:
                del self.bonds[key]
        for key, order in bond_set(sp).items():
            self.bonds[key] = order

    def match_equivalents(self, sp, coords, off):
        """A phosphate's three outer oxygens (and a CH₂'s or CH₃'s hydrogens)
        are interchangeable in the picture. Give each the new spot nearest its
        old one, so they never swap places through each other mid-motion."""
        from itertools import permutations
        coords = dict(coords)
        for a in sp.m.GetAtoms():
            for el in ("O", "H"):
                ends = [n.GetProp("lbl") for n in a.GetNeighbors()
                        if n.GetSymbol() == el and n.GetDegree() == 1 and n.GetProp("lbl") in self.atoms]
                if len(ends) < 2:
                    continue
                spots = [coords[l] for l in ends]
                old = [self.atoms[l]["pos"] - off for l in ends]
                best = min(permutations(range(len(ends))),
                           key=lambda p: sum(float(np.sum((spots[p[i]] - old[i]) ** 2)) for i in range(len(ends))))
                for i, l in enumerate(ends):
                    coords[l] = spots[best[i]]
        return coords

    def put(self, lbl, el, pos, vis=1.0, charge=0, glow=False):
        self.atoms[lbl] = {"el": el, "pos": np.array(pos, dtype=float), "vis": vis, "charge": charge, "glow": glow}

    def move_group(self, labels, target):
        """Moves a group rigidly so its centroid lands on `target`."""
        c = np.mean([self.atoms[l]["pos"] for l in labels], axis=0)
        for l in labels:
            self.atoms[l]["pos"] = self.atoms[l]["pos"] - c + np.array(target, dtype=float)

    def cut(self, labels):
        """Removes every bond between the group and the rest."""
        s = set(labels)
        for key in list(self.bonds):
            if (key[0] in s) != (key[1] in s):
                del self.bonds[key]

    def token(self, tid, text, pos, alpha):
        self.tokens[tid] = {"text": text, "pos": np.array(pos, dtype=float), "alpha": alpha}

    def key(self, t):
        self.keys.append({
            "t": round(t, 3),
            "atoms": {l: [*map(float, a["pos"]), a["vis"], a["charge"], 1 if a["glow"] else 0]
                      for l, a in self.atoms.items()},
            "bonds": [[a, b, o] for (a, b), o in self.bonds.items() if o > 0],
            "tokens": {i: [k["text"], *map(float, k["pos"]), k["alpha"]] for i, k in self.tokens.items()},
            **copy.deepcopy(self.caption),
        })

    def export(self):
        return {"name": self.name, "duration": self.keys[-1]["t"], "camera_distance": self.camera_distance,
                "camera_swing": self.camera_swing,
                "elements": {l: a["el"] for l, a in self.atoms.items()}, "keys": self.keys}


OFF = 30.0   # far off-screen
H_OUT = np.array([0.0, 0.0, 0.0])


def free_spot(coords, lbl, distance, skip=()):
    """The point `distance` Å from atom `lbl` that is farthest from every other
    atom (leaning toward the camera), and the direction to it."""
    p = coords[lbl]
    others = np.array([q for l, q in coords.items() if l != lbl and l not in skip])
    best, best_score, best_dir = None, -1.0, None
    n = 400
    for i in range(n):
        z = 1 - 2 * (i + 0.5) / n
        r = np.sqrt(1 - z * z)
        phi = i * np.pi * (3 - np.sqrt(5))
        d = np.array([r * np.cos(phi), r * np.sin(phi), z])
        q = p + d * distance
        score = float(np.min(np.linalg.norm(others - q, axis=1))) + 0.15 * d[2]
        if score > best_score:
            best, best_score, best_dir = q, score, d
    return best, best_dir


def outward(coords, lbl, distance):
    """A point `distance` Å further out from the molecule's center through atom `lbl`."""
    center = np.mean(list(coords.values()), axis=0)
    p = coords[lbl]
    d = p - center
    d = d / (np.linalg.norm(d) + 1e-9)
    return p + d * distance


# ---------------------------------------------------------------- GIF 1

def build_spend():
    glc = pw.labeled_glucose()
    g6p, e_hk = pw.hexokinase(glc)
    g6p_open = pw.open_glucose_ring(g6p)
    f6p_open = pw.aldose_to_ketose(g6p_open)
    f6p = pw.close_fructose_ring(f6p_open)
    f16bp, e_pfk = pw.phosphofructokinase(f6p)
    f16bp_open = pw.open_fructose_ring(f16bp)
    dhap, g3p_b = pw.aldolase(f16bp_open)
    g3p_a = pw.triose_phosphate_isomerase(dhap)

    c_glc, _ = orient(embed(glc))
    c_g6p = embed(g6p, glc, c_glc)
    c_g6p_open = face_camera(embed(g6p_open, g6p, c_g6p), c_g6p, g6p_open)
    c_f6p_open = face_camera(embed(f6p_open, g6p_open, c_g6p_open), c_g6p_open, f6p_open)
    c_f6p = embed(f6p, f6p_open, c_f6p_open)
    c_f16bp = embed(f16bp, f6p, c_f6p)
    c_f16bp_open = face_camera(embed(f16bp_open, f16bp, c_f16bp), c_f16bp, f16bp_open)
    c_dhap = embed(dhap, f16bp_open, c_f16bp_open)
    c_g3p_b = embed(g3p_b, f16bp_open, c_f16bp_open)
    c_g3p_a = embed(g3p_a, dhap, c_dhap)

    # After the split the halves drift apart along x, each toward the side it
    # already sits on in the open chain (carbons 1-3 end up on the right), so
    # they never pass through each other.
    side_a = 1.0 if np.mean([c_f16bp_open[l] for l in ("C1", "C2", "C3")], axis=0)[0] > 0 else -1.0
    side = {"A": side_a, "B": -side_a}

    def spread(coords, direction):
        cen = np.mean(list(coords.values()), axis=0)
        shift = np.array([direction * 4.8 - cen[0], -0.2 - cen[1], 0])
        return {l: p + shift for l, p in coords.items()}

    def pulled(gap):
        """The open chain with C3–C4 broken, each piece (as still bonded) slid
        `gap` Å toward its own side."""
        bonds = [k for k in bond_set(f16bp_open) if set(k) != {"C3", "C4"}]
        piece = {l: l for l in c_f16bp_open}
        def root(l):
            while piece[l] != l:
                l = piece[l]
            return l
        for a, b in bonds:
            piece[root(a)] = root(b)
        side_of = {root("C1"): side["A"], root("C6"): side["B"]}
        return {l: p + np.array([side_of[root(l)] * gap, 0, 0]) for l, p in c_f16bp_open.items()}
    c_dhap_s = spread(c_dhap, side["A"])
    c_g3p_b_s = spread(c_g3p_b, side["B"])
    c_g3p_a_s = spread(embed(g3p_a, dhap, c_dhap_s), side["A"])

    tl = Timeline("spend")
    atp1 = np.array([-5.2, 3.0, 0.5])
    atp2 = np.array([5.2, 3.0, 0.5])
    tl.set_species(glc, c_glc)
    # The phosphate groups wait at their ATP tokens (hidden until the token appears).
    for group, token, final in ((e_hk["arrive"][0][0], atp1, c_g6p), (e_pfk["arrive"][0][0], atp2, c_f16bp)):
        cen = np.mean([final[l] for l in group], axis=0)
        for l in group:
            el = "P" if l in ("Pa", "Pb") else "O"
            q = -1 if l.endswith("b") or l.endswith("c") else 0
            tl.put(l, el, final[l] - cen + token + np.array([0, 0.9, 0]), vis=0.0, charge=q)
    # The next glucose waits above the frame.
    for l, p in c_glc.items():
        tl.put("n_" + l, glc.atom(l).GetSymbol(), p + np.array([0, 14, 0]))
    for key, o in bond_set(glc).items():
        tl.bonds[("n_" + key[0], "n_" + key[1])] = o
    tl.token("ATP1", "ATP", atp1, 0)
    tl.token("ATP2", "ATP", atp2, 0)
    tl.caption = {"title": "Glucose", "enzyme": "Glycolysis begins: first the cell spends two ATP",
                  "equation": "C₆H₁₂O₆", "ledger": [0, 0, 0]}
    t = 0.0
    tl.key(t)

    # 1. Hexokinase.
    t += 1.0
    tl.caption = {"title": "Spend #1: glucose gets a phosphate", "enzyme": "hexokinase",
                  "equation": "glucose + ATP → glucose-6-phosphate + ADP + H⁺", "ledger": [0, 0, 0]}
    tl.token("ATP1", "ATP", atp1, 1)
    for l in e_hk["arrive"][0][0]:
        tl.atoms[l]["vis"] = 1.0
    tl.key(t)
    t += 2.6
    h = e_hk["leave_h"][0]
    tl.set_species(g6p, c_g6p)
    h_spot, h_dir = free_spot(c_g6p, "O6", 2.4, skip=(h,))
    tl.put(h, "H", h_spot, charge=1, glow=True)
    tl.token("ATP1", "ADP", atp1, 1)
    tl.caption["ledger"] = [1, 0, 0]
    tl.key(t)
    t += 1.2
    tl.put(h, "H", h_spot + h_dir * 7.0, charge=1, glow=True)
    tl.token("ATP1", "ADP", atp1 + np.array([-1.5, 1.0, 0]), 0)
    tl.key(t)
    tl.atoms[h]["vis"] = 0.0
    tl.put(h, "H", outward(c_g6p, "O6", OFF), vis=0.0)

    # 2. Phosphoglucose isomerase: open, shuffle, close.
    t += 0.4
    tl.caption = {"title": "The six-sided ring becomes five-sided", "enzyme": "phosphoglucose isomerase",
                  "equation": "glucose-6-phosphate → fructose-6-phosphate", "ledger": [1, 0, 0]}
    tl.key(t)
    t += 1.6
    tl.set_species(g6p_open, c_g6p_open)
    tl.key(t)
    t += 1.6
    tl.set_species(f6p_open, c_f6p_open)
    tl.key(t)
    t += 1.6
    tl.set_species(f6p, c_f6p)
    tl.key(t)

    # 3. Phosphofructokinase.
    t += 0.8
    tl.caption = {"title": "Spend #2: the point of no return", "enzyme": "phosphofructokinase-1",
                  "equation": "fructose-6-phosphate + ATP → fructose-1,6-bisphosphate + ADP + H⁺",
                  "ledger": [1, 0, 0]}
    tl.token("ATP2", "ATP", atp2, 1)
    for l in e_pfk["arrive"][0][0]:
        tl.atoms[l]["vis"] = 1.0
    tl.key(t)
    t += 2.6
    h = e_pfk["leave_h"][0]
    tl.set_species(f16bp, c_f16bp)
    h_spot, h_dir = free_spot(c_f16bp, "O1", 2.4, skip=(h,))
    tl.put(h, "H", h_spot, charge=1, glow=True)
    tl.token("ATP2", "ADP", atp2, 1)
    tl.caption["ledger"] = [2, 0, 0]
    tl.key(t)
    t += 1.2
    tl.put(h, "H", h_spot + h_dir * 7.0, charge=1, glow=True)
    tl.token("ATP2", "ADP", atp2 + np.array([1.5, 1.0, 0]), 0)
    tl.key(t)
    tl.put(h, "H", outward(c_f16bp, "O1", OFF), vis=0.0)

    # 4. Aldolase: open the ring, then split between C3 and C4.
    t += 0.5
    tl.caption = {"title": "Six carbons become three plus three", "enzyme": "aldolase",
                  "equation": "fructose-1,6-bisphosphate → DHAP + glyceraldehyde-3-phosphate",
                  "ledger": [2, 0, 0]}
    tl.key(t)
    t += 1.6
    tl.set_species(f16bp_open, c_f16bp_open)
    tl.key(t)
    t += 1.2
    tl.set_species(f16bp_open, pulled(1.0))
    del tl.bonds[("C3", "C4")]
    tl.key(t)
    t += 1.6
    tl.set_species(dhap, c_dhap_s)
    tl.set_species(g3p_b, c_g3p_b_s)
    tl.key(t)
    t += 0.6
    tl.key(t)

    # 5. Triose phosphate isomerase.
    t += 0.4
    tl.caption = {"title": "Two identical halves", "enzyme": "triose phosphate isomerase",
                  "equation": "DHAP → glyceraldehyde-3-phosphate", "ledger": [2, 0, 0]}
    tl.key(t)
    t += 2.2
    tl.set_species(g3p_a, c_g3p_a_s)
    tl.key(t)
    t += 1.2
    tl.caption = {"title": "Two glyceraldehyde-3-phosphates, ready for the payoff", "enzyme": "",
                  "equation": "1 glucose → 2 glyceraldehyde-3-phosphate  (ATP spent: 2)", "ledger": [2, 0, 0]}
    tl.key(t)

    # 6. Clear the stage: halves drift out, the next glucose arrives.
    t += 2.6
    for sp, coords, direction in ((g3p_a, c_g3p_a_s, side["A"]), (g3p_b, c_g3p_b_s, side["B"])):
        labels = sp.labels()
        cen = np.mean([coords[l] for l in labels], axis=0)
        tl.move_group(labels, cen + np.array([direction * 14, -3, 0]))
    for l, p in c_glc.items():
        tl.atoms["n_" + l]["pos"] = p.copy()
    tl.caption = {"title": "Glucose", "enzyme": "Glycolysis begins: first the cell spends two ATP",
                  "equation": "C₆H₁₂O₆", "ledger": [0, 0, 0]}
    tl.key(t)
    species = {"glucose": glc, "G6P": g6p, "F6P": f6p, "F16BP": f16bp, "DHAP": dhap, "G3P_A": g3p_a, "G3P_B": g3p_b}
    return tl, species, (g3p_a, g3p_b)


# ---------------------------------------------------------------- GIF 2

def build_payoff():
    glc = pw.labeled_glucose()
    g6p, _ = pw.hexokinase(glc)
    f6p = pw.close_fructose_ring(pw.aldose_to_ketose(pw.open_glucose_ring(g6p)))
    f16bp, _ = pw.phosphofructokinase(f6p)
    dhap, g3p_b = pw.aldolase(pw.open_fructose_ring(f16bp))
    g3p_a = pw.triose_phosphate_isomerase(dhap)

    chains = {}
    for half, g3p in (("A", g3p_a), ("B", g3p_b)):
        bpg, e1 = pw.gapdh(g3p, half)
        pg3, e2 = pw.phosphoglycerate_kinase(bpg, half)
        pg2, e3 = pw.phosphoglycerate_mutase(pg3, half)
        pep, e4 = pw.enolase(pg2, half)
        pyr, e5 = pw.pyruvate_kinase(pep, half)
        chains[half] = {"sp": [g3p, bpg, pg3, pg2, pep, pyr], "ev": [e1, e2, e3, e4, e5]}

    # Shapes for half A, copied onto half B so the twins match.
    spa = chains["A"]["sp"]
    ca = [orient(embed(spa[0]))[0]]
    for i in range(1, len(spa)):
        shape = embed(spa[i], spa[i - 1], ca[i - 1], follow=True)
        flat = face_camera(shape, ca[i - 1], spa[i])
        ca.append(flat if path_clear(ca[i - 1], flat, spa[i].m) else shape)
    cb = [map_coords(spa[i], ca[i], chains["B"]["sp"][i]) for i in range(len(spa))]
    # Turn half A half a turn about the vertical axis (a rotation, so its
    # handedness is kept): both phosphate ends then face the middle and both
    # aldehyde ends, where the new phosphates join, face outward.
    ald, phos = pw.HALVES["A"]["ald"], pw.HALVES["A"]["phos"]
    if (ca[0][ald][0] - ca[0][phos][0]) < 0:
        ca = [{l: p * np.array([-1, 1, -1]) for l, p in c.items()} for c in ca]
    else:
        cb = [{l: p * np.array([-1, 1, -1]) for l, p in c.items()} for c in cb]
    # Tilt both halves 15° (same sense) so the pair is narrower on screen and
    # their facing ends step apart vertically.
    tilt = np.radians(15)
    rz = np.array([[np.cos(tilt), -np.sin(tilt), 0], [np.sin(tilt), np.cos(tilt), 0], [0, 0, 1]])
    ca = [{l: rz @ p for l, p in c.items()} for c in ca]
    cb = [{l: rz @ p for l, p in c.items()} for c in cb]
    coords = {"A": ca, "B": cb}
    x_of = {"A": 4.5, "B": -4.5}
    HALVES_ALD = {h: pw.HALVES[h]["ald"] for h in "AB"}
    HALVES_MID_O = {h: pw.HALVES[h]["mid_o"] for h in "AB"}   # carbons 1-3 on the right, as in the spend GIF

    def at(half, i):
        return {l: p + np.array([x_of[half], 0, 0]) for l, p in coords[half][i].items()}

    def unit(v):
        return v / np.linalg.norm(v)

    def pi_shape(half):
        """HPO₄²⁻ as it will sit on 1,3-BPG, plus its own H (O–H 0.96 Å, pointing
        away from the P), and the direction it arrives from."""
        ev = chains[half]["ev"][0]
        c1 = at(half, 1)
        grp = ev["phosphate_in"]
        shape = {l: c1[l] for l in grp}
        on = ev["phosphate_h_on"]
        shape[ev["phosphate_h"]] = c1[on] + 0.96 * unit(c1[on] - c1[grp[0]])
        # It comes up from below the frame, so it hovers below and outside.
        approach = unit(unit(c1[grp[0]] - c1[HALVES_ALD[half]]) + np.array([0, -1.0, 0]))
        return shape, approach

    def leave_offset(half, i, grp, anchor):
        """A leaving phosphate first slides 1.8 Å off the O it was on, leaning
        up toward its ADP token and never toward the other half."""
        c = at(half, i - 1)
        d = unit(c[grp[0]] - c[anchor])
        if d[0] * x_of[half] < 0:
            d[0] = 0
        return unit(d + np.array([0, 1, 0])) * 1.8

    tl = Timeline("payoff")
    tl.camera_distance = 30.0
    tl.camera_swing = 7.0
    for half in "AB":
        tl.set_species(chains[half]["sp"][0], at(half, 0))
        # The next pair of halves waits above the frame.
        for l, p in at(half, 0).items():
            tl.put("n_" + l, chains[half]["sp"][0].atom(l).GetSymbol(), p + np.array([0, 14, 0]))
        for key, o in bond_set(chains[half]["sp"][0]).items():
            tl.bonds[("n_" + key[0], "n_" + key[1])] = o
    ledger = [2, 0, 0]
    tl.caption = {"title": "Two glyceraldehyde-3-phosphates", "enzyme": "The payoff: each half now earns back ATP",
                  "equation": "carbons 4–6 on the left, 1–3 on the right", "ledger": ledger[:]}
    t = 0.0
    # Free phosphates (HPO₄²⁻) wait below the frame; their final shape comes from 1,3-BPG.
    for half in "AB":
        ev = chains[half]["ev"][0]
        grp = ev["phosphate_in"]
        shape, _ = pi_shape(half)
        cen = np.mean([shape[l] for l in grp], axis=0)
        start = np.array([x_of[half] * 1.5, -11.0, 0.5])
        for l, p in shape.items():
            el = "P" if l == grp[0] else ("H" if l == ev["phosphate_h"] else "O")
            q = -1 if l in (grp[0] + "c",) else 0
            tl.put(l, el, p - cen + start, charge=q)
        tl.bonds[tuple(sorted((ev["phosphate_h"], ev["phosphate_h_on"])))] = 1
        tl.bonds[(grp[0], grp[0] + "a")] = 2
        for s in "bcd":
            tl.bonds[(grp[0], grp[0] + s)] = 1
        tl.token("NAD" + half, "NAD⁺", [x_of[half], -3.9, 0.8], 0)
        tl.token("ADP1" + half, "ADP", [x_of[half], 4.7, 0.8], 0)
        tl.token("ADP2" + half, "ADP", [x_of[half], 4.7, 0.8], 0)
    tl.key(t)

    # 5. GAPDH: phosphate in, hydride to NAD⁺, H⁺ out.
    t += 1.0
    tl.caption = {"title": "Energy captured as NADH; a free phosphate joins", "enzyme": "glyceraldehyde-3-phosphate dehydrogenase",
                  "equation": "G3P + NAD⁺ + HPO₄²⁻ → 1,3-bisphosphoglycerate + NADH + H⁺", "ledger": ledger[:]}
    for half in "AB":
        ev = chains[half]["ev"][0]
        shape, approach = pi_shape(half)
        for l, p in shape.items():
            tl.atoms[l]["pos"] = p + approach * 1.7
        tl.token("NAD" + half, "NAD⁺", [x_of[half], -3.9, 0.8], 1)
    tl.key(t)
    t += 2.6
    h_out = {}
    for half in "AB":
        ev = chains[half]["ev"][0]
        c1 = at(half, 1)
        tl.set_species(chains[half]["sp"][1], c1)
        hyd = ev["hydride"]
        tl.put(hyd, "H", [x_of[half] + 0.6, -3.5, 1.2])   # onto the NAD⁺ token
        ph = ev["phosphate_h"]
        tl.cut([ph])
        spot, direction = free_spot(c1, ev["phosphate_h_on"], 2.4)
        h_out[half] = (spot, direction)
        tl.put(ph, "H", spot, charge=1, glow=True)
        tl.token("NAD" + half, "NADH", [x_of[half], -3.9, 0.8], 1)
    ledger = [2, 0, 2]
    tl.caption["ledger"] = ledger[:]
    tl.key(t)
    t += 1.2
    for half in "AB":
        ev = chains[half]["ev"][0]
        c1 = at(half, 1)
        spot, direction = h_out[half]
        tl.put(ev["phosphate_h"], "H", spot + direction * 7.0, charge=1, glow=True)
        tl.put(ev["hydride"], "H", [x_of[half] + 0.6, -3.9, 1.2], vis=0.0)
        tl.token("NAD" + half, "NADH", [x_of[half], -4.6, 0.8], 0)
    tl.key(t)
    for half in "AB":
        ev = chains[half]["ev"][0]
        tl.atoms[ev["phosphate_h"]]["vis"] = 0.0

    # 6. Phosphoglycerate kinase: first payback.
    t += 0.4
    tl.caption = {"title": "First payback: back to even", "enzyme": "phosphoglycerate kinase",
                  "equation": "1,3-bisphosphoglycerate + ADP → 3-phosphoglycerate + ATP", "ledger": ledger[:]}
    for half in "AB":
        tl.token("ADP1" + half, "ADP", [x_of[half], 4.7, 0.8], 1)
    tl.key(t)
    t += 2.4
    for half in "AB":
        grp = chains[half]["ev"][1]["leave"]
        off = leave_offset(half, 2, grp, "Q" + half + "b")
        before = {l: tl.atoms[l]["pos"].copy() for l in grp}
        tl.set_species(chains[half]["sp"][2], at(half, 2))
        tl.cut(grp)
        for l in grp:
            tl.atoms[l]["pos"] = before[l] + off
        tl.token("ADP1" + half, "ATP", [x_of[half], 4.7, 0.8], 1)
    ledger = [2, 2, 2]
    tl.caption["ledger"] = ledger[:]
    tl.key(t)
    t += 1.0
    for half in "AB":
        grp = chains[half]["ev"][1]["leave"]
        cen = np.mean([tl.atoms[l]["pos"] for l in grp], axis=0)
        tl.move_group(grp, [cen[0], 5.4, cen[2]])   # straight up, clear of the molecule
        for l in grp:
            tl.atoms[l]["vis"] = 0.0
        tl.token("ADP1" + half, "ATP", [x_of[half], 5.8, 0.8], 0)
    tl.key(t)

    # 7a. Phosphoglycerate mutase.
    t += 0.3
    tl.caption = {"title": "A phosphate slides over", "enzyme": "phosphoglycerate mutase",
                  "equation": "3-phosphoglycerate → 2-phosphoglycerate", "ledger": ledger[:]}
    tl.key(t)
    t += 2.0
    for half in "AB":
        tl.set_species(chains[half]["sp"][3], at(half, 3))
    tl.key(t)

    # 7b. Enolase: water leaves.
    t += 0.4
    tl.caption = {"title": "Water leaves", "enzyme": "enolase",
                  "equation": "2-phosphoglycerate → phosphoenolpyruvate + H₂O", "ledger": ledger[:]}
    tl.key(t)
    t += 2.0
    water_dir = {}
    for half in "AB":
        water = chains[half]["ev"][3]["water"]
        c3 = at(half, 3)
        tl.set_species(chains[half]["sp"][4], at(half, 4))
        tl.cut(water)
        pep = at(half, 4)
        pep[water[0]] = c3[water[0]]
        o, d = free_spot(pep, water[0], 1.4)
        water_dir[half] = d
        tl.put(water[0], "O", o)
        # A real water shape: O–H 0.96 Å at 104.5°, both H's leaning away.
        side = unit(np.cross(d, [0, 0, 1]) if abs(d[2]) < 0.9 else np.cross(d, [1, 0, 0]))
        a = np.radians(104.5 / 2)
        tl.put(water[1], "H", o + 0.96 * (np.cos(a) * d + np.sin(a) * side))
        tl.put(water[2], "H", o + 0.96 * (np.cos(a) * d - np.sin(a) * side))
        tl.bonds[tuple(sorted((water[0], water[1])))] = 1
        tl.bonds[tuple(sorted((water[0], water[2])))] = 1
    tl.key(t)
    t += 1.4
    for half in "AB":
        water = chains[half]["ev"][3]["water"]
        cen = np.mean([tl.atoms[l]["pos"] for l in water], axis=0)
        tl.move_group(water, cen + water_dir[half] * 6.0)
        for l in water:
            tl.atoms[l]["vis"] = 0.0
    tl.key(t)

    # 8. Pyruvate kinase: second payback.
    t += 0.3
    tl.caption = {"title": "Second payback: pyruvate", "enzyme": "pyruvate kinase",
                  "equation": "phosphoenolpyruvate + ADP + H⁺ → pyruvate + ATP", "ledger": ledger[:]}
    for half in "AB":
        tl.token("ADP2" + half, "ADP", [x_of[half], 4.7, 0.8], 1)
        hp = chains[half]["ev"][4]["proton_in"]
        both = {**at("A", 4), **at("B", 4)}
        both[hp] = at(half, 5)[hp]
        spot, _ = free_spot(both, hp, 5.0)
        tl.put(hp, "H", spot, charge=1, glow=True)
    tl.key(t)
    t += 2.4
    for half in "AB":
        grp = chains[half]["ev"][4]["leave"]
        off = leave_offset(half, 5, grp, HALVES_MID_O[half])
        before = {l: tl.atoms[l]["pos"].copy() for l in grp}
        tl.set_species(chains[half]["sp"][5], at(half, 5))
        tl.cut(grp)
        for l in grp:
            tl.atoms[l]["pos"] = before[l] + off
        tl.token("ADP2" + half, "ATP", [x_of[half], 4.7, 0.8], 1)
    ledger = [2, 4, 2]
    tl.caption["ledger"] = ledger[:]
    tl.key(t)
    t += 1.0
    for half in "AB":
        grp = chains[half]["ev"][4]["leave"]
        cen = np.mean([tl.atoms[l]["pos"] for l in grp], axis=0)
        tl.move_group(grp, [cen[0], 5.4, cen[2]])   # straight up, clear of the molecule
        for l in grp:
            tl.atoms[l]["vis"] = 0.0
        tl.token("ADP2" + half, "ATP", [x_of[half], 5.8, 0.8], 0)
    tl.caption = {"title": "Two pyruvates", "enzyme": "",
                  "equation": "glucose → 2 pyruvate:  2 ATP spent, 4 made, net +2 ATP, plus 2 NADH",
                  "ledger": ledger[:]}
    tl.key(t)
    t += 1.6
    tl.key(t)

    # 9. Clear the stage: pyruvates fall away, the next pair arrives.
    t += 2.6
    for half in "AB":
        labels = chains[half]["sp"][5].labels()
        cen = np.mean([tl.atoms[l]["pos"] for l in labels], axis=0)
        tl.move_group(labels, cen + np.array([0, -14, 0]))
        for l, p in at(half, 0).items():
            tl.atoms["n_" + l]["pos"] = p.copy()
    tl.caption = {"title": "Two glyceraldehyde-3-phosphates", "enzyme": "The payoff: each half now earns back ATP",
                  "equation": "carbons 4–6 on the left, 1–3 on the right", "ledger": [2, 0, 0]}
    tl.key(t)
    return tl, chains


if __name__ == "__main__":
    spend, species, _ = build_spend()
    payoff, chains = build_payoff()
    data = {"gifs": [spend.export(), payoff.export()]}
    json.dump(data, open(OUT, "w"), indent=0)
    for g in data["gifs"]:
        print(f"{g['name']}: {len(g['keys'])} keyframes, {g['duration']:.1f} s, {len(g['elements'])} atoms")
    print("wrote", os.path.relpath(OUT))
