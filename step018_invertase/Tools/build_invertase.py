"""Builds Resources/invertase.json from two crystal structures it downloads.

    python3 Tools/build_invertase.py

THE ENZYME
    PDB 4EQV, Saccharomyces cerevisiae invertase (SUC2), 3.40 A.
    Sainz-Polo, Ramirez-Escudero, Lafraya, Gonzalez, Marin-Navarro, Polaina &
    Sanz-Aparicio, J. Biol. Chem. 288:9755 (2013).

    The deposited file IS the biological unit: eight chains, A to H, D4 point
    symmetry — a tetramer of dimers. Nothing has to be generated.

    NOTE ON NUMBERING, which is the first thing this script checks. The paper
    numbers the mature protein, which begins Ser-Met-Thr-Asn-... The crystal
    numbers from the Met, one earlier, so every residue in the coordinates is
    ONE LOWER than the paper's:

        paper Asp23  (nucleophile)                = 4EQV Asp22   in W-M-N-D-P-N-G
        paper Asp152 (transition-state stabiliser)= 4EQV Asp151  in R-D-P
        paper Glu204 (acid/base)                  = 4EQV Glu203  in E-C

    This script finds those three motifs in the coordinates and asserts that
    the residues they land on are the right ones. Taking the paper's numbers
    into the crystal unchecked would have put the nucleophile on a proline.

THE SUBSTRATE, and why it is not docked
    There is no structure of sucrose bound to yeast invertase. There is,
    however, a trapped Michaelis complex in a CLOSE YEAST RELATIVE:

    PDB 6S1T, beta-fructofuranosidase from Schwanniomyces occidentalis,
    D50A nucleophile mutant, with sucrose in the active site, 2.09 A.
    Miguez, Gimeno-Perez, Gonzalez-Alfonso, ... Fernandez-Lobato &
    Sanz-Aparicio, Sci. Rep. 11:7158 (2021).

    The two enzymes are 51% identical over 500 aligned residues. So the pose
    here is not docked by hand: the template's active site is superposed onto
    4EQV's and the sucrose is carried across with it. The superposition is
    computed below and its quality is written into the JSON, because that
    number is the whole claim.

    What comes out is not fitted and is worth stating: the fructose C2 sits
    2.75 A from the nucleophile's carboxylate oxygen, the glycosidic oxygen
    2.57 A from the acid/base carboxylate, and the O2-C2-OD2 angle — the line
    the nucleophile has to attack along — is 164 degrees. None of that was
    aimed at. It is still a MODEL of invertase's complex, and is labelled one.

THE FIVE STATES
    Built from those coordinates, not drawn freehand:

      free        empty site
      michaelis   sucrose bound, as transferred
      covalent    beta-fructosyl-enzyme ester on Asp23, glucose leaving
      hydrolysis  glucose gone, the attacking water in place
      product     free beta-D-fructose, enzyme restored

    The two inversions are done as real geometry. At each one the anomeric
    carbon C2 is reflected through the plane of its three retained substituents
    (C1, C3, O5) — the umbrella flip through the planar oxocarbenium-like
    transition state — and the incoming oxygen is placed on the far face along
    the line the leaving group left on. The configuration is then MEASURED from
    the coordinates rather than asserted, with beta calibrated against 6S1T's
    own ligand, which the PDB chemical dictionary annotates beta-D-fructofuranose.
"""

import json
import os
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "Resources")
OUT = os.path.join(RES, "invertase.json")

ENZYME = "4EQV"          # S. cerevisiae invertase, 3.40 A, octamer
TEMPLATE = "6S1T"        # S. occidentalis Ffase D50A + sucrose, 2.09 A

# Literature (mature-protein) numbering, from Sainz-Polo et al. (2013).
LIT_NUCLEOPHILE = 23
LIT_STABILISER = 152
LIT_ACID_BASE = 204

THREE = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
         "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
         "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}

# Bond lengths used to build the two states that no crystal shows, in angstroms.
# Allen et al., J. Chem. Soc. Perkin Trans. II S1 (1987), small-molecule means.
ESTER_C_O = 1.43         # glycosyl-ester C2-O(Asp), a C(sp3)-O(carboxylate) bond
ANOMERIC_C_O = 1.41      # C2-OH of a free fructofuranose


def fetch(code):
    path = os.path.join(RES, f"{code}.pdb")
    if not os.path.exists(path):
        url = f"https://files.rcsb.org/download/{code}.pdb"
        print(f"fetching {url}")
        with urllib.request.urlopen(url) as r, open(path, "wb") as f:
            f.write(r.read())
    return path


def read_atoms(path, want_chain=None, records=("ATOM  ", "HETATM")):
    """Every atom, waters dropped, first altloc only."""
    out = []
    for line in open(path):
        if line[:6] not in records:
            continue
        if line[17:20].strip() == "HOH":
            continue
        if line[16] not in " A":
            continue
        chain = line[21]
        if want_chain is not None and chain != want_chain:
            continue
        out.append({
            "name": line[12:16].strip(),
            "res": line[17:20].strip(),
            "ch": chain,
            "seq": int(line[22:26]),
            "el": (line[76:78].strip() or line[12:14].strip()[:1]).upper(),
            "pos": np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])]),
        })
    return out


def resolution(path):
    for line in open(path):
        if line.startswith("REMARK   2 RESOLUTION."):
            return float(line.split()[3])
    raise AssertionError("no resolution record")


def ca_chain(atoms):
    """(pdb numbers, one-letter sequence, CA coordinates) for one chain."""
    seen = {}
    for a in atoms:
        if a["name"] == "CA" and len(a["res"]) == 3 and a["res"] in THREE:
            seen[a["seq"]] = a
    keys = sorted(seen)
    seq = "".join(THREE[seen[k]["res"]] for k in keys)
    xyz = np.array([seen[k]["pos"] for k in keys])
    return keys, seq, xyz


def needleman_wunsch(a, b, match=2, mismatch=-1, gap=-4):
    """A plain global alignment. These two sequences are half identical, so a
    simple scoring scheme is enough — and it is checked below by asserting the
    three catalytic residues line up, which is the only thing riding on it."""
    n, m = len(a), len(b)
    score = np.zeros((n + 1, m + 1))
    back = np.zeros((n + 1, m + 1), dtype=np.int8)
    score[1:, 0] = gap * np.arange(1, n + 1)
    back[1:, 0] = 1
    score[0, 1:] = gap * np.arange(1, m + 1)
    back[0, 1:] = 2
    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            diag = score[i - 1, j - 1] + (match if ai == b[j - 1] else mismatch)
            up = score[i - 1, j] + gap
            left = score[i, j - 1] + gap
            best = max(diag, up, left)
            score[i, j] = best
            back[i, j] = 0 if best == diag else (1 if best == up else 2)
    i, j, pairs = n, m, []
    while i > 0 or j > 0:
        step = back[i, j]
        if step == 0:
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif step == 1:
            i -= 1
        else:
            j -= 1
    return pairs[::-1]


def kabsch(p, q):
    """The rotation and translation taking point set p onto q, least squares.
    Kabsch, Acta Cryst. A32:922 (1976)."""
    pc, qc = p.mean(0), q.mean(0)
    u, _, vt = np.linalg.svd((p - pc).T @ (q - qc))
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return r, qc - r @ pc


def find_motif(seq, keys, motif):
    hits = [keys[i] for i in range(len(seq) - len(motif) + 1) if seq[i:i + len(motif)] == motif]
    assert len(hits) == 1, f"motif {motif} should occur once, found it at {hits}"
    return hits[0]


def reflect_through_plane(point, a, b, c):
    """`point` mirrored in the plane through a, b and c. This is how the two
    inversions at the anomeric carbon are done: the carbon flips through the
    plane of the three substituents it keeps."""
    normal = np.cross(b - a, c - a)
    normal = normal / np.linalg.norm(normal)
    return point - 2.0 * float(np.dot(point - a, normal)) * normal


def anomeric_volume(ring_o, ring_c, exo, anomeric):
    """A signed volume whose SIGN says which face of the ring the exocyclic
    substituent sits on. Positive is calibrated below against 6S1T's own
    ligand, which the PDB annotates beta-D-fructofuranose."""
    a = ring_o - anomeric
    b = ring_c - anomeric
    d = exo - anomeric
    return float(np.dot(np.cross(a, b), d))


def main():
    enzyme_path = fetch(ENZYME)
    template_path = fetch(TEMPLATE)

    res = resolution(enzyme_path)
    print(f"{ENZYME}: {res:.2f} A")

    enzyme = read_atoms(enzyme_path)
    chains = sorted({a["ch"] for a in enzyme})
    assert len(chains) == 8, f"4EQV should be an octamer in the file, found {len(chains)} chains"
    print(f"  {len(enzyme)} heavy atoms in {len(chains)} chains {''.join(chains)} — "
          f"the deposited octamer, D4")

    chain_a = [a for a in enzyme if a["ch"] == "A"]
    keys, seq, ca = ca_chain(chain_a)

    # --- the numbering check ---------------------------------------------
    nuc_pdb = find_motif(seq, keys, "NDPNG") + 1        # the D of NDPNG
    stab_pdb = find_motif(seq, keys, "RDP") + 1         # the D of RDP
    acid_pdb = find_motif(seq, keys, "EC")              # the E of EC

    def residue(seq_no, chain="A"):
        hit = [a for a in enzyme if a["ch"] == chain and a["seq"] == seq_no]
        assert hit, f"no residue {seq_no} in chain {chain}"
        return hit

    assert residue(nuc_pdb)[0]["res"] == "ASP", "the NDPNG nucleophile must be an aspartate"
    assert residue(stab_pdb)[0]["res"] == "ASP", "the RDP stabiliser must be an aspartate"
    assert residue(acid_pdb)[0]["res"] == "GLU", "the EC acid/base must be a glutamate"
    offsets = {LIT_NUCLEOPHILE - nuc_pdb, LIT_STABILISER - stab_pdb, LIT_ACID_BASE - acid_pdb}
    assert offsets == {1}, f"the paper's numbering should be uniformly +1, got {offsets}"
    print(f"  catalytic residues: Asp{nuc_pdb} (paper Asp{LIT_NUCLEOPHILE}, nucleophile), "
          f"Asp{stab_pdb} (paper Asp{LIT_STABILISER}), Glu{acid_pdb} (paper Glu{LIT_ACID_BASE})")
    print(f"  crystal numbering runs one behind the paper's throughout")

    # --- the template, and the transfer -----------------------------------
    template = read_atoms(template_path)
    t_chain = [a for a in template if a["ch"] == "A"]
    t_keys, t_seq, t_ca = ca_chain(t_chain)

    sucrose = [a for a in template if a["res"] in ("GLC", "FRU")]
    by_chain = {}
    for a in sucrose:
        by_chain.setdefault(a["ch"], []).append(a)
    nuc_t = find_motif(t_seq, t_keys, "NAPNG") + 1      # D50A: the D is an alanine here
    nuc_cb = [a for a in t_chain if a["seq"] == nuc_t and a["name"] == "CB"][0]["pos"]
    near = {ch: min(float(np.linalg.norm(a["pos"] - nuc_cb)) for a in v) for ch, v in by_chain.items()}
    suc_chain = min(near, key=near.get)
    sucrose = by_chain[suc_chain]
    assert len(sucrose) == 23, f"sucrose has 23 heavy atoms, found {len(sucrose)}"
    print(f"{TEMPLATE}: sucrose (chain {suc_chain}, {len(sucrose)} heavy atoms) "
          f"{near[suc_chain]:.2f} A from the mutated nucleophile Ala{nuc_t}")

    pairs = needleman_wunsch(t_seq, seq)
    identical = sum(1 for i, j in pairs if t_seq[i] == seq[j])
    identity = 100.0 * identical / len(pairs)
    mapped = {t_keys[i]: keys[j] for i, j in pairs}
    assert mapped[nuc_t] == nuc_pdb and mapped.get(230) == acid_pdb, \
        "the alignment must put the catalytic residues in register"
    print(f"  aligned {len(pairs)} residues, {identical} identical = {identity:.1f}%")

    # Superpose on the active site alone. The whole chain is 2.3 A over 500
    # residues; the site is what the substrate sits in, and it is much tighter.
    suc_xyz = np.array([a["pos"] for a in sucrose])
    site_t = {a["seq"] for a in t_chain
              if float(np.min(np.linalg.norm(suc_xyz - a["pos"], axis=1))) < 8.0}
    sel = [(i, j) for i, j in pairs if t_keys[i] in site_t]
    p = np.array([t_ca[i] for i, j in sel])
    q = np.array([ca[j] for i, j in sel])
    rot, trans = kabsch(p, q)
    site_rmsd = float(np.sqrt((np.linalg.norm(p @ rot.T + trans - q, axis=1) ** 2).mean()))

    whole = [(i, j) for i, j in pairs]
    pw = np.array([t_ca[i] for i, j in whole])
    qw = np.array([ca[j] for i, j in whole])
    rw, tw = kabsch(pw, qw)
    whole_rmsd = float(np.sqrt((np.linalg.norm(pw @ rw.T + tw - qw, axis=1) ** 2).mean()))
    print(f"  superposition: {len(sel)} active-site CA at {site_rmsd:.2f} A, "
          f"{len(whole)} CA over the whole chain at {whole_rmsd:.2f} A")

    placed = {}
    for a in sucrose:
        placed[(a["res"], a["name"])] = rot @ a["pos"] + trans

    def P(res, name):
        return placed[(res, name)]

    def enz(seq_no, name, chain="A"):
        hit = [a for a in enzyme if a["ch"] == chain and a["seq"] == seq_no and a["name"] == name]
        assert hit, f"no atom {name} in residue {seq_no}{chain}"
        return hit[0]["pos"]

    # --- what the transfer produced, measured -----------------------------
    c2 = P("FRU", "C2")
    o_glyc = P("FRU", "O2")            # the bridging oxygen, glucose C1 on the far side
    c1, c3, o5 = P("FRU", "C1"), P("FRU", "C3"), P("FRU", "O5")
    od1, od2 = enz(nuc_pdb, "OD1"), enz(nuc_pdb, "OD2")
    nuc_o_name = "OD2" if np.linalg.norm(od2 - c2) < np.linalg.norm(od1 - c2) else "OD1"
    nuc_o = enz(nuc_pdb, nuc_o_name)
    oe1, oe2 = enz(acid_pdb, "OE1"), enz(acid_pdb, "OE2")
    acid_o_name = "OE1" if np.linalg.norm(oe1 - o_glyc) < np.linalg.norm(oe2 - o_glyc) else "OE2"
    acid_o = enz(acid_pdb, acid_o_name)

    def angle(a, b, c):
        u, v = a - b, c - b
        cosine = float(np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v))
        return float(np.degrees(np.arccos(max(-1.0, min(1.0, cosine)))))

    attack_distance = float(np.linalg.norm(nuc_o - c2))
    attack_angle = angle(o_glyc, c2, nuc_o)
    acid_distance = float(np.linalg.norm(acid_o - o_glyc))
    clashes = 0
    for pos in placed.values():
        for a in enzyme:
            if a["ch"] == "A" and float(np.linalg.norm(pos - a["pos"])) < 2.2:
                clashes += 1
    assert clashes == 0, f"the transferred sucrose clashes with the enzyme at {clashes} contacts"
    print(f"  transferred pose: {nuc_pdb}{nuc_o_name}..C2 {attack_distance:.2f} A, "
          f"O2-C2-{nuc_o_name} {attack_angle:.1f} deg, "
          f"{acid_pdb}{acid_o_name}..O(glycosidic) {acid_distance:.2f} A, {clashes} clashes")

    # --- the two inversions, as geometry ----------------------------------
    beta_reference = anomeric_volume(o5, c3, o_glyc, c2)
    assert beta_reference > 0, "calibration: 6S1T's ligand is beta, so beta is the positive sign"

    fructosyl = [a for a in sucrose if a["res"] == "FRU"]
    glucosyl = [a for a in sucrose if a["res"] == "GLC"]

    def frame(atoms_in, positions):
        return [{"el": a["el"], "name": a["name"], "res": a["res"],
                 "pos": [round(float(v), 4) for v in positions[a["name"]]]} for a in atoms_in]

    michaelis_fru = {a["name"]: P("FRU", a["name"]) for a in fructosyl}
    michaelis_glc = {a["name"]: P("GLC", a["name"]) for a in glucosyl}

    # Glycosylation. The anomeric carbon umbrellas through the plane of the
    # three substituents it keeps, the bridging oxygen leaves with the glucose,
    # and the fructosyl unit then slides down the attack line until the new
    # ester bond is the right length. `slide` solves for that exactly rather
    # than assuming the carbon sits on the line.
    attack_dir = (nuc_o - c2) / np.linalg.norm(nuc_o - c2)
    covalent_fru = {k: v.copy() for k, v in michaelis_fru.items()}
    covalent_fru.pop("O2")
    covalent_fru["C2"] = reflect_through_plane(covalent_fru["C2"], covalent_fru["C1"],
                                               covalent_fru["C3"], covalent_fru["O5"])

    def slide(point, target, direction, length):
        """How far to move along `direction` so `point` ends `length` from `target`."""
        v = target - point
        b = float(np.dot(v, direction))
        disc = b * b - float(np.dot(v, v)) + length * length
        assert disc >= 0, "the bond length is unreachable along this line"
        return b - float(np.sqrt(disc))

    step = slide(covalent_fru["C2"], nuc_o, attack_dir, ESTER_C_O)
    covalent_fru = {k: v + attack_dir * step for k, v in covalent_fru.items()}
    covalent_volume = anomeric_volume(covalent_fru["O5"], covalent_fru["C3"], nuc_o,
                                      covalent_fru["C2"])
    assert covalent_volume * beta_reference < 0, \
        "glycosylation must invert the anomeric centre"
    ester = float(np.linalg.norm(nuc_o - covalent_fru["C2"]))
    assert 1.30 <= ester <= 1.60, f"the ester bond came out {ester:.2f} A"

    # The glucose leaves along the line it was already on, away from the site.
    out_dir = (P("GLC", "C1") - c2)
    out_dir = out_dir / np.linalg.norm(out_dir)
    leaving_glc = {k: v + out_dir * 7.0 for k, v in michaelis_glc.items()}
    leaving_glc["O1"] = o_glyc + out_dir * 7.0

    # Deglycosylation. Water attacks from the face the glucose left on, so the
    # carbon umbrellas back and the product is beta again: two inversions, net
    # retention. That is what invertase's name gets wrong.
    water_dir = -attack_dir
    water_o = covalent_fru["C2"] + water_dir * 2.80
    product_fru = {k: v + water_dir * step for k, v in covalent_fru.items()}
    product_fru["C2"] = reflect_through_plane(product_fru["C2"], product_fru["C1"],
                                              product_fru["C3"], product_fru["O5"])
    product_fru["O2"] = product_fru["C2"] + water_dir * ANOMERIC_C_O
    product_volume = anomeric_volume(product_fru["O5"], product_fru["C3"], product_fru["O2"],
                                     product_fru["C2"])
    assert product_volume * beta_reference > 0, \
        "deglycosylation must invert again, back to beta"
    print(f"  anomeric volume: michaelis {beta_reference:+.2f}, covalent {covalent_volume:+.2f}, "
          f"product {product_volume:+.2f} — beta, alpha, beta")
    print(f"  fructosyl-enzyme ester C2-{nuc_o_name} {ester:.2f} A")

    # The free fructose drifts out the way the glucose went.
    free_fru = {k: v + water_dir * 1.0 for k, v in product_fru.items()}

    def bonds_of(names, extra=()):
        """Sugar bonds, found by distance inside the fragment."""
        keys_l = list(names)
        out = []
        for i in range(len(keys_l)):
            for j in range(i + 1, len(keys_l)):
                d = float(np.linalg.norm(names[keys_l[i]] - names[keys_l[j]]))
                if d < 1.75:
                    out.append([keys_l[i], keys_l[j]])
        return out + [list(e) for e in extra]

    states = {
        "free": {"fructosyl": {}, "glucosyl": {}, "water": None, "esterBond": False},
        "michaelis": {"fructosyl": michaelis_fru, "glucosyl": michaelis_glc,
                      "water": None, "esterBond": False},
        "covalent": {"fructosyl": covalent_fru, "glucosyl": leaving_glc,
                     "water": None, "esterBond": True},
        "hydrolysis": {"fructosyl": covalent_fru, "glucosyl": {},
                       "water": water_o, "esterBond": True},
        "product": {"fructosyl": free_fru, "glucosyl": {}, "water": None, "esterBond": False},
    }

    out_states = {}
    for name, s in states.items():
        fru = s["fructosyl"]
        glc = s["glucosyl"]
        entry = {
            "fructosyl": [{"el": "O" if n.startswith("O") else "C", "name": n,
                           "pos": [round(float(v), 4) for v in fru[n]]} for n in sorted(fru)],
            "glucosyl": [{"el": "O" if n.startswith("O") else "C", "name": n,
                          "pos": [round(float(v), 4) for v in glc[n]]} for n in sorted(glc)],
            "fructosylBonds": bonds_of(fru) if fru else [],
            "glucosylBonds": bonds_of(glc) if glc else [],
            "esterBond": s["esterBond"],
            "water": None if s["water"] is None else [round(float(v), 4) for v in s["water"]],
        }
        if fru and glc and name == "michaelis":
            entry["glycosidicBond"] = ["FRU:O2", "GLC:C1"]
        out_states[name] = entry

    # --- placing the scene ------------------------------------------------
    # Everything is written in the octamer's own frame, centred on its centroid,
    # so the assembly turns about itself. The porthole looks down the line from
    # the dimer's centre out through the active site.
    all_pos = np.array([a["pos"] for a in enzyme])
    centre = all_pos.mean(axis=0)

    # Which chain is A's dimer partner is decided by counting the interface,
    # not by assuming it is the next letter. The paper calls the assembly a
    # tetramer of dimers; this is that claim made checkable.
    chain_a_pos = np.array([a["pos"] for a in chain_a])
    contacts = {}
    for other in chains:
        if other == "A":
            continue
        other_pos = np.array([a["pos"] for a in enzyme if a["ch"] == other])
        cell = 5.0
        grid = {}
        for q in other_pos:
            grid.setdefault(tuple((q // cell).astype(int)), []).append(q)
        n = 0
        for p in chain_a_pos:
            base = (p // cell).astype(int)
            found = False
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        for q in grid.get((base[0] + dx, base[1] + dy, base[2] + dz), []):
                            if float(np.dot(p - q, p - q)) < 20.25:     # 4.5 A
                                found = True
                                break
                        if found:
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                n += 1
        contacts[other] = n
    partner = max(contacts, key=contacts.get)
    ranked = sorted(contacts.items(), key=lambda kv: -kv[1])
    assert ranked[0][1] > 2 * max(1, ranked[1][1]), \
        f"chain A should have one dominant partner, interfaces are {ranked}"
    dimer_chains = ["A", partner]
    print(f"  interfaces from A: {', '.join(f'{k} {v}' for k, v in ranked if v)} "
          f"-> the dimer is A{partner}, the rest is the tetramer contact")

    dimer_pos = np.array([a["pos"] for a in enzyme if a["ch"] in dimer_chains])
    dimer_centre = dimer_pos.mean(axis=0)
    site_centre = np.array([c2, o_glyc, nuc_o, acid_o]).mean(axis=0)

    # The funnel axis: the GH32 site sits at the bottom of the five-bladed
    # propeller's central cavity, so the way in is straight out from the chain's
    # own centre. The porthole opens along this line.
    site_axis = site_centre - chain_a_pos.mean(axis=0)
    site_axis = site_axis / np.linalg.norm(site_axis)
    in_front = 0
    for a in enzyme:
        if a["ch"] not in dimer_chains:
            continue
        offset = a["pos"] - site_centre
        along = float(np.dot(offset, site_axis))
        if along <= 0:
            continue
        if float(np.linalg.norm(offset - site_axis * along)) < 9.0:
            in_front += 1
    print(f"  {in_front} dimer atoms lie in the 9 A channel over the site — "
          f"what the porthole has to take away")

    print(f"  octamer {np.ptp(all_pos, axis=0)[0]:.0f} x {np.ptp(all_pos, axis=0)[1]:.0f} x "
          f"{np.ptp(all_pos, axis=0)[2]:.0f} A; dimer AB {len(dimer_pos)} atoms")

    def shifted(v):
        return [round(float(x), 4) for x in (np.asarray(v) - centre)]

    for entry in out_states.values():
        for group in ("fructosyl", "glucosyl"):
            for a in entry[group]:
                a["pos"] = shifted(np.array(a["pos"]))
        if entry["water"] is not None:
            entry["water"] = shifted(np.array(entry["water"]))

    atoms_out = []
    for a in enzyme:
        atoms_out.append({
            "el": a["el"], "ch": a["ch"], "seq": a["seq"], "res": a["res"], "name": a["name"],
            "pos": shifted(a["pos"]),
        })

    data = {
        "enzyme": {
            "pdb": ENZYME,
            "resolutionAngstroms": res,
            "organism": "Saccharomyces cerevisiae",
            "gene": "SUC2",
            "family": "glycoside hydrolase family 32 (GH32)",
            "source": ("PDB 4EQV, Sainz-Polo et al., J. Biol. Chem. 288:9755 (2013), "
                       f"{res:.2f} A; waters and hydrogens omitted"),
            "chains": chains,
            "dimerChains": dimer_chains,
            "interfaceContacts": contacts,
            "atomCount": len(atoms_out),
            "dimerAtomCount": int(len(dimer_pos)),
            "biologicalUnit": "octamer (a tetramer of dimers), D4; the deposited file is the assembly",
        },
        "numbering": {
            "note": "the crystal numbers one behind the paper's mature-protein numbering",
            "offsetToPaper": 1,
            "nucleophile": {"pdbSeq": nuc_pdb, "paperSeq": LIT_NUCLEOPHILE, "res": "ASP",
                            "motif": "NDPNG", "oxygen": nuc_o_name},
            "stabiliser": {"pdbSeq": stab_pdb, "paperSeq": LIT_STABILISER, "res": "ASP",
                           "motif": "RDP"},
            "acidBase": {"pdbSeq": acid_pdb, "paperSeq": LIT_ACID_BASE, "res": "GLU",
                         "motif": "EC", "oxygen": acid_o_name},
        },
        "template": {
            "pdb": TEMPLATE,
            "resolutionAngstroms": resolution(template_path),
            "organism": "Schwanniomyces occidentalis",
            "variant": "D50A nucleophile mutant, sucrose trapped in the site",
            "source": ("PDB 6S1T, Miguez et al., Sci. Rep. 11:7158 (2021), 2.09 A; "
                       "beta-fructofuranosidase D50A with sucrose"),
            "alignedResidues": len(pairs),
            "identicalResidues": identical,
            "identityPercent": round(identity, 1),
            "siteResidues": len(sel),
            "siteRmsdAngstroms": round(site_rmsd, 3),
            "wholeChainRmsdAngstroms": round(whole_rmsd, 3),
            "clashes": clashes,
        },
        "catalysis": {
            "attackDistanceAngstroms": round(attack_distance, 3),
            "attackAngleDegrees": round(attack_angle, 1),
            "acidToLeavingOxygenAngstroms": round(acid_distance, 3),
            "esterBondAngstroms": round(ester, 3),
            "anomericVolumes": {"michaelis": round(beta_reference, 4),
                                "covalent": round(covalent_volume, 4),
                                "product": round(product_volume, 4)},
            "betaSign": 1,
            "mechanism": "retaining, double displacement: two inversions, net retention",
        },
        "siteCentre": shifted(site_centre),
        "siteAxis": [round(float(v), 5) for v in site_axis],
        "atomsOverSite": in_front,
        "dimerCentre": shifted(dimer_centre),
        "nucleophileOxygen": shifted(nuc_o),
        "acidOxygen": shifted(acid_o),
        "states": out_states,
        "atoms": atoms_out,
    }
    with open(OUT, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    size = os.path.getsize(OUT) / 1024 / 1024
    print(f"wrote {os.path.relpath(OUT)} ({size:.1f} MB)")


if __name__ == "__main__":
    main()
