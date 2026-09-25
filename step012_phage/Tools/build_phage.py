"""Builds Resources/scene.json: a T4 phage baseplate firing, from the deposited
structures, plus the E. coli envelope it fires into.

The structures
--------------
  5IV5  T4 baseplate, hexagonal PRE-attachment   549,576 atoms, 145 chains, 4.11 Å
  5IV7  T4 baseplate, star-shaped POST-attachment 312,210 atoms,  96 chains, 6.77 Å
  3J9Q  contractile nanotube, PRE-contraction     99,648 atoms, 3.5 Å
  3J9R  the same nanotube, POST-contraction      103,824 atoms, 3.9 Å
  5W5F  T4 tail tube                              23,472 atoms, 3.4 Å
Taylor et al., Nature 533:346 (2016) for the baseplates; Ge et al., Nat. Struct.
Mol. Biol. 22:377 (2015) for the nanotube.

What the two baseplate entries actually contain
-----------------------------------------------
This was the thing to check before building anything, and the answer decided
the render. 5IV7 is a STRICT SUBSET of 5IV5:

  common to both  8 entities, 96 chains   gp6 gp7 gp8 gp9 gp10 gp11 gp25 gp53
                                          (the wedge — identical chain counts
                                           and identical atom counts per entity)
  only in 5IV5    8 entities, 49 chains   gp27 hub, gp5 peptidoglycan hydrolase
                                          (the needle), gp48, gp54, gp19 tail
                                          tube, gp12 short tail fibres, Zn, Fe

"Hubless" in 5IV7's title is literal: the whole central hub is unmodelled after
firing. That is not a gap in the data to be papered over — it is the event. The
needle and hub are driven out ahead of the tube, and short tail fibres swing
down to grip the cell. So:

  * the 96 wedge chains MORPH between the two measured states;
  * the hub parts are drawn only in the pre state and depart downward, which is
    labelled `model` because their path is inferred, not solved.

Nothing appears from nowhere, and nothing vanishes mid-flight unlabelled.

Run: python3 Tools/build_phage.py
"""

import collections
import json
import math
import os
import re
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "Resources")
OUT = os.path.join(RES, "scene.json")

# The wedge proteins, present in both baseplate states.
WEDGE = ["Baseplate wedge protein gp6", "Baseplate wedge protein gp7",
         "Baseplate wedge protein gp8", "Baseplate wedge protein gp9",
         "Baseplate wedge protein gp10", "Baseplate wedge protein gp11",
         "Baseplate wedge protein gp25", "Baseplate wedge protein gp53"]

# Everything the post-attachment structure no longer has, split by what each
# part actually does when the phage fires. Lumping them together would drive
# the short tail fibres through the cell wall, which is not what they do —
# they grip the OUTSIDE.
NEEDLE = ["Baseplate hub protein gp27", "Peptidoglycan hydrolase gp5",
          "Baseplate tail-tube protein gp48", "Baseplate tail-tube protein gp54"]
TUBE = ["Tail tube protein gp19"]            # stays put; the sheath drives it
FIBRE = ["Short tail fiber protein gp12"]    # swings down onto the LPS and grips
HUB = NEEDLE + TUBE + FIBRE

# van der Waals radii, Bondi, J. Phys. Chem. 68:441 (1964), as every step since 9.
VDW = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80,
       "SE": 1.90, "ZN": 1.39, "FE": 2.00, "MG": 1.73}

# The E. coli envelope, measured, carried over from step 10 unchanged.
BILAYER = 47.0       # Å, AFM of a PE/PG/cardiolipin bilayer
PERIPLASM = 130.0    # Å, CEMOVIS 120 and cryo-ET 140 on the same organism
PG_THICK = 45.0      # Å, peptidoglycan layer
AREA_PER_LIPID = 58.8  # Å², POPE


# The coordinate files are not committed — 5IV5 alone is 55 MB — so they are
# fetched on demand. 5IV5 and 5IV7 are too large for the legacy PDB format and
# exist only as mmCIF.
ENTRIES = ["5IV5", "5IV7", "3J9Q", "3J9R"]


def fetch(entry):
    path = os.path.join(RES, f"{entry}.cif")
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        return path
    url = f"https://files.rcsb.org/download/{entry}.cif"
    print(f"  fetching {entry} from RCSB...")
    os.makedirs(RES, exist_ok=True)
    with urllib.request.urlopen(url) as r, open(path, "wb") as f:
        f.write(r.read())
    return path


# ------------------------------------------------------------------ parsing

def cif_entities(lines):
    """entity id -> description, from the _entity loop."""
    names, i = {}, 0
    while i < len(lines):
        if lines[i].startswith("loop_"):
            hdr, j = [], i + 1
            while j < len(lines) and lines[j].lstrip().startswith("_"):
                hdr.append(lines[j].strip())
                j += 1
            if any(h.startswith("_entity.") for h in hdr):
                col = {h: k for k, h in enumerate(hdr)}
                idc, dsc = col.get("_entity.id"), col.get("_entity.pdbx_description")
                k = j
                while k < len(lines) and not lines[k].startswith("#"):
                    row = lines[k].strip()
                    if row and not row.startswith("_"):
                        p = re.findall(r"'[^']*'|\"[^\"]*\"|\S+", row)
                        if idc is not None and dsc is not None and len(p) > max(idc, dsc):
                            names[p[idc]] = p[dsc].strip("'\"")
                    k += 1
            i = j
        else:
            i += 1
    return names


def read_cif(path):
    """Chains as {auth_chain: {'entity': name, 'atoms': [(key, el, x, y, z), ...]}}
    where key is (residue number, atom name) — the label an atom keeps between
    one structure and another, so the two states can be intersected on it.

    Column indices confirmed against the file's own _atom_site header:
    2 type_symbol, 3 label_atom_id, 7 label_entity_id, 10-12 Cartn_x/y/z,
    16 auth_seq_id, 18 auth_asym_id.
    """
    lines = open(path).readlines()
    names = cif_entities(lines)
    chains = collections.OrderedDict()
    for ln in lines:
        if ln.startswith(("ATOM", "HETATM")):
            p = ln.split()
            ch = p[18]
            if ch not in chains:
                chains[ch] = {"entity": names.get(p[7], "?"), "atoms": []}
            chains[ch]["atoms"].append(((p[16], p[3]), p[2],
                                        float(p[10]), float(p[11]), float(p[12])))
    return chains


def read_pdb(path):
    """The same shape of result for a classic PDB file."""
    chains = collections.OrderedDict()
    name = "?"
    for ln in open(path):
        if ln.startswith("TITLE"):
            name = ln[10:].strip() or name
        elif ln.startswith(("ATOM", "HETATM")):
            ch = ln[21]
            if ch not in chains:
                chains[ch] = {"entity": name, "atoms": []}
            el = (ln[76:78].strip() or ln[12:16].strip()[:1]).upper()
            key = (ln[22:26].strip(), ln[12:16].strip())
            chains[ch]["atoms"].append((key, el, float(ln[30:38]), float(ln[38:46]), float(ln[46:54])))
    return chains


# ------------------------------------------------------------------ geometry

def centroid(atoms):
    n = len(atoms)
    return (sum(a[2] for a in atoms) / n, sum(a[3] for a in atoms) / n, sum(a[4] for a in atoms) / n)


def recentre(chains):
    """Puts the assembly's centre of mass at the origin in x and y, and its
    lowest point at z = 0, so the two states can be compared on equal terms."""
    pts = [a for c in chains.values() for a in c["atoms"]]
    cx, cy, _ = centroid(pts)
    zlo = min(a[4] for a in pts)
    for c in chains.values():
        c["atoms"] = [(k, e, x - cx, y - cy, z - zlo) for k, e, x, y, z in c["atoms"]]
    return chains


def match_wedge(pre, post):
    """Pairs the 96 wedge chains between the two states.

    Chain IDs do not survive the transition (only 68 of 96 are shared), so
    they cannot be trusted. What does hold is that both states contain the
    same entities with the same chain counts, and the baseplate has six-fold
    symmetry about z. So within each entity the chains are sorted by their
    angle about the axis and paired in that order — a correspondence that
    depends only on the symmetry, not on how the depositors named things.
    """
    def by_entity(chains):
        out = collections.defaultdict(list)
        for cid, c in chains.items():
            if c["entity"] in WEDGE:
                cx, cy, _ = centroid(c["atoms"])
                out[c["entity"]].append((math.atan2(cy, cx), cid))
        for e in out:
            out[e].sort()
        return out

    a, b = by_entity(pre), by_entity(post)
    pairs = []
    for e in WEDGE:
        ca, cb = a.get(e, []), b.get(e, [])
        assert len(ca) == len(cb), f"{e}: {len(ca)} chains pre, {len(cb)} post"
        # Rotate the post list to the offset that minimises total angular shift,
        # so a chain is paired with its own neighbour rather than one 60° away.
        best, best_cost = 0, None
        for k in range(len(cb)):
            cost = sum(abs((ca[i][0] - cb[(i + k) % len(cb)][0] + math.pi) % (2 * math.pi) - math.pi)
                       for i in range(len(ca)))
            if best_cost is None or cost < best_cost:
                best, best_cost = k, cost
        for i in range(len(ca)):
            pairs.append((e, ca[i][1], cb[(i + best) % len(cb)][1]))
    return pairs


def radius_and_height(chains, only=None):
    pts = [a for cid, c in chains.items() if only is None or c["entity"] in only for a in c["atoms"]]
    r = max(math.hypot(a[2], a[3]) for a in pts)
    return r, max(a[4] for a in pts) - min(a[4] for a in pts)


# ------------------------------------------------------------------ envelope

def envelope(half_x, half_y, top_z):
    """Step 10's measured E. coli envelope, coarse-grained as it was there:
    Martini beads, roughly four heavy atoms each. Built downward from top_z."""
    beads = []
    om_top = top_z
    om_mid = om_top - BILAYER / 2
    pg_mid = om_top - BILAYER - PERIPLASM / 2
    im_mid = om_top - BILAYER - PERIPLASM - BILAYER / 2
    spacing = math.sqrt(AREA_PER_LIPID)          # Å between lipid heads

    def leaflet(z, up, part, group):
        nx = int(2 * half_x / spacing)
        ny = int(2 * half_y / spacing)
        for i in range(nx):
            for j in range(ny):
                x = -half_x + (i + 0.5) * spacing
                y = -half_y + (j + 0.5) * spacing
                beads.append((x, y, z, 2.6, part, group))          # head
                for k in range(1, 5):                              # four tail beads
                    beads.append((x, y, z - up * k * BILAYER / 11, 2.3, part + "_tail", group))

    leaflet(om_mid + BILAYER / 2, 1, "lps", "outer_membrane")
    leaflet(om_mid - BILAYER / 2, -1, "lipid", "outer_membrane")
    leaflet(im_mid + BILAYER / 2, 1, "lipid", "inner_membrane")
    leaflet(im_mid - BILAYER / 2, -1, "lipid", "inner_membrane")

    # Peptidoglycan: a mesh of glycan strands cross-linked by peptides.
    step = 42.0
    nx = int(2 * half_x / step)
    ny = int(2 * half_y / step)
    for i in range(nx + 1):
        for j in range(ny + 1):
            x = -half_x + i * step
            y = -half_y + j * step
            if i < nx:
                for t in range(6):
                    beads.append((x + t * step / 6, y, pg_mid, 3.2, "pg", "peptidoglycan"))
            if j < ny:
                for t in range(1, 6):
                    beads.append((x, y + t * step / 6, pg_mid, 2.8, "pg", "peptidoglycan"))
    return beads, dict(om=om_mid, pg=pg_mid, im=im_mid, top=om_top,
                       bilayer=BILAYER, periplasm=PERIPLASM)


# ------------------------------------------------------------------ main

def main():
    for e in ENTRIES:
        fetch(e)
    print("reading the baseplate, both states...")
    pre = recentre(read_cif(os.path.join(RES, "5IV5.cif")))
    post = recentre(read_cif(os.path.join(RES, "5IV7.cif")))

    ents_pre = {c["entity"] for c in pre.values()}
    ents_post = {c["entity"] for c in post.values()}
    common = ents_pre & ents_post
    assert set(WEDGE) <= common, f"expected the wedge in both, missing {set(WEDGE) - common}"
    assert not (ents_post - ents_pre), "5IV7 should be a strict subset of 5IV5"
    print(f"  5IV5 {sum(len(c['atoms']) for c in pre.values()):>8,} atoms  {len(pre):>3} chains  {len(ents_pre)} entities")
    print(f"  5IV7 {sum(len(c['atoms']) for c in post.values()):>8,} atoms  {len(post):>3} chains  {len(ents_post)} entities")
    print(f"  common: {len(common)} entities (the wedge)   pre-only: {len(ents_pre - ents_post)} (the hub)")

    pairs = match_wedge(pre, post)
    print(f"  matched {len(pairs)} wedge chains by six-fold symmetry")

    r_pre, h_pre = radius_and_height(pre, only=set(WEDGE))
    r_post, h_post = radius_and_height(post, only=set(WEDGE))
    print(f"  wedge spreads {2*r_pre/10:.1f} nm -> {2*r_post/10:.1f} nm across "
          f"({100*(r_post/r_pre-1):+.0f}%), height {h_pre:.0f} -> {h_post:.0f} Å ({100*(h_post/h_pre-1):+.0f}%)")

    print("reading the sheath, both states...")
    sh_pre = recentre(read_cif(os.path.join(RES, "3J9Q.cif")))
    sh_post = recentre(read_cif(os.path.join(RES, "3J9R.cif")))
    sr_pre, sh_h_pre = radius_and_height(sh_pre)
    sr_post, sh_h_post = radius_and_height(sh_post)
    print(f"  sheath {sh_h_pre:.0f} -> {sh_h_post:.0f} Å long ({100*(sh_h_post/sh_h_pre-1):+.0f}%), "
          f"radius {sr_pre:.0f} -> {sr_post:.0f} Å ({100*(sr_post/sr_pre-1):+.0f}%)")

    # ---- assemble ----
    # The baseplate sits above the cell. The wedge morphs; the hub departs.
    shapes = []      # (x, y, z, radius, part, group, evidence, morph_index)
    morph = []       # paired positions: [(pre xyz, post xyz)] for the wedge

    # Wedge: one entry per atom carrying both its start and end position.
    # An atom morphs only if BOTH states resolved it. Cryo-EM models different
    # disordered stretches in each, so the chains do not agree atom for atom:
    # they are intersected on (residue number, atom name), and anything present
    # in one state alone is left out rather than being made to appear from
    # nowhere. The fraction kept is reported and asserted in the tests.
    n_wedge, n_pre_only, n_post_only = 0, 0, 0
    for entity, cid_pre, cid_post in pairs:
        a = {k: (el, x, y, z) for k, el, x, y, z in pre[cid_pre]["atoms"]}
        b = {k: (el, x, y, z) for k, el, x, y, z in post[cid_post]["atoms"]}
        shared = a.keys() & b.keys()
        n_pre_only += len(a) - len(shared)
        n_post_only += len(b) - len(shared)
        for k in sorted(shared):
            el, x, y, z = a[k]
            _, x2, y2, z2 = b[k]
            r = VDW.get(el.upper(), 1.70)
            shapes.append({"p": [round(x, 2), round(y, 2), round(z, 2)], "r": round(r, 2),
                           "part": "wedge", "group": entity, "ev": "measured",
                           "q": [round(x2, 2), round(y2, 2), round(z2, 2)]})
            n_wedge += 1
    kept = 100 * n_wedge / (n_wedge + n_pre_only)
    print(f"  wedge atoms resolved in both states: {n_wedge:,} of {n_wedge + n_pre_only:,} ({kept:.1f}%)")
    print(f"    dropped: {n_pre_only:,} modelled only before firing, {n_post_only:,} only after")

    # Hub and needle: present only before firing. They travel downward, which
    # is inferred rather than solved — hence `model`.
    n_hub = collections.Counter()
    for cid, c in pre.items():
        if c["entity"] in HUB:
            part = ("needle" if c["entity"] in NEEDLE
                    else "tube" if c["entity"] in TUBE else "fibre")
            for _k, el, x, y, z in c["atoms"]:
                r = VDW.get(el.upper(), 1.70)
                shapes.append({"p": [round(x, 2), round(y, 2), round(z, 2)], "r": round(r, 2),
                               "part": part, "group": c["entity"], "ev": "model", "q": None})
                n_hub[part] += 1

    print(f"  wedge  {n_wedge:>7,} atoms  morphing between two measured states")
    for part in ("needle", "tube", "fibre"):
        what = {"needle": "driven through the wall", "tube": "stays, the sheath drives it",
                "fibre": "swings down and grips the outside"}[part]
        print(f"  {part:<6} {n_hub[part]:>7,} atoms  {what}")

    # The envelope, below the baseplate. The gap is measured, not guessed: the
    # star spreads downward as it flips, so the clearance is taken from the
    # LOWEST point the wedge reaches in the post-attachment state. Set from the
    # pre state instead and the arms would pass straight through the membrane.
    post_low = min(z for cid, c in post.items() if c["entity"] in WEDGE
                   for _k, _e, _x, _y, z in c["atoms"])
    clearance = 30.0
    top = post_low - clearance
    print(f"  the flipped star reaches down to z = {post_low:.0f} Å; "
          f"the membrane starts {clearance:.0f} Å below that")
    half = max(r_post * 1.25, 380.0)
    env_beads, layers = envelope(half, half * 0.35, top)
    for x, y, z, r, part, group in env_beads:
        shapes.append({"p": [round(x, 2), round(y, 2), round(z, 2)], "r": r,
                       "part": part, "group": group, "ev": "measured", "q": None})
    print(f"  envelope {len(env_beads):,} beads, half-width {half:.0f} Å")

    reach = layers["im"] - BILAYER / 2          # the needle must pass this z
    data = {
        "source": "PDB 5IV5, 5IV7 (Taylor 2016); 3J9Q, 3J9R (Ge 2015); 5W5F. "
                  "Envelope dimensions as measured in step 10.",
        "counts": {"wedge": n_wedge, "needle": n_hub["needle"], "tube": n_hub["tube"],
                   "fibre": n_hub["fibre"], "envelope": len(env_beads),
                   "total": len(shapes)},
        "clearance": round(clearance, 1),
        "starLow": round(post_low, 1),
        "baseplate": {"preRadius": round(r_pre, 1), "postRadius": round(r_post, 1),
                      "preHeight": round(h_pre, 1), "postHeight": round(h_post, 1),
                      "wedgeChains": len(pairs), "wedgeAtomsKept": n_wedge,
                      "wedgeAtomsDropped": n_pre_only},
        "sheath": {"preLength": round(sh_h_pre, 1), "postLength": round(sh_h_post, 1),
                   "preRadius": round(sr_pre, 1), "postRadius": round(sr_post, 1)},
        "layers": {k: round(v, 1) for k, v in layers.items()},
        "reach": round(reach, 1),
        "entities": {"common": sorted(common), "preOnly": sorted(ents_pre - ents_post)},
        "shapes": shapes,
    }
    with open(OUT, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    mb = os.path.getsize(OUT) / 1e6
    print(f"\nwrote {os.path.relpath(OUT)}  {len(shapes):,} shapes, {mb:.0f} MB")
    print(f"  the needle must reach z = {reach:.0f} Å to cross the inner membrane")


if __name__ == "__main__":
    main()
