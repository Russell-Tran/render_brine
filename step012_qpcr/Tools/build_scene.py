"""Builds Resources/scene.json: a DNA target with two hybridisation probes
landing on it, for the FRET render.

WHY THIS CHEMISTRY AND NOT THE ONE IN THE PROPOSAL. The step was proposed
around a TaqMan hydrolysis probe: one probe carrying a reporter dye and a
quencher, cut apart by the polymerase. Working the numbers first killed that
plan, and the reason is worth keeping in the source:

  When the polymerase reaches a TaqMan probe, the probe is HYBRIDISED to the
  template. Hybridised DNA is a rigid rod - persistence length about 50 nm,
  far longer than a 25-mer - so the dye and the quencher sit at opposite ends
  of a stiff 8.5 nm stick. Forster transfer at 8.5 nm with R0 near 5 nm is
  about 4%. By FRET alone the intact probe would already be ~96% BRIGHT.

  Real TaqMan probes are dark until they are cut. So FRET cannot be what keeps
  them dark; the dominant mechanism is contact (static) quenching, where dye
  and quencher touch and form a non-fluorescent ground-state complex. That has
  no clean distance law to compute - it is a binding equilibrium, not geometry.

So a render driving TaqMan brightness off the r^-6 law would teach something
false. The LightCycler HybProbe format does not have that problem: two probes
land head-to-tail a few nucleotides apart, a donor on the 3' end of one and an
acceptor on the 5' end of the other, and FRET between them IS the mechanism -
that is the entire design of the assay. Better still, the separation is set by
base-pair geometry, which is measured, not modelled.

WHAT THIS BUILDS. A 34 bp B-DNA duplex stamped from the four base-pair
templates in step008a_plasmid (themselves lifted from the 1BNA crystal
structure), then read as three molecules:

  - the target strand, continuous, the thing being amplified
  - probe 1, hybridised, carrying the DONOR at its 3' end
  - probe 2, arriving, carrying the ACCEPTOR at its 5' end

with a gap of a few nucleotides between the probes. Probe 2 is animated in
from solution; the donor-acceptor distance closes, and the renderer computes
transfer efficiency from that distance on every frame.

Run: python3 Tools/build_scene.py
"""

import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PGLO = os.path.join(HERE, "..", "..", "step008a_plasmid", "Resources", "pglo.json")
OUT = os.path.join(HERE, "..", "Resources", "scene.json")

# The stretch of pGLO to use as the target: inside the GFP gene, so the render
# is amplifying the same gene steps 8a, 9 and 11 have been following.
TARGET_START = 1342          # GFP's start codon, from step 8a
TARGET_LENGTH = 34           # base pairs

# Probe layout along the target, in base pairs from the left end.
PROBE1 = (2, 15)             # hybridised, donor at its 3' end
GAP_NT = 2                   # nucleotides between the probes
PROBE2 = (17, 32)            # arriving, acceptor at its 5' end

# Ideal B-DNA, as steps 8a and 11 used.
RISE = 3.4                   # A per base pair
TWIST = 360.0 / 10.5         # degrees per base pair, Wang PNAS 76:200 (1979)


def load_templates():
    with open(PGLO) as f:
        pglo = json.load(f)
    return pglo["templates"], pglo["sequence"]


def step_transform(i):
    """Where base pair `i` sits: rise along x, twist about x."""
    a = math.radians(TWIST * i)
    c, s = math.cos(a), math.sin(a)
    rot = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    shift = np.array([RISE * i, 0.0, 0.0])
    return rot, shift


def build_duplex(seq, templates):
    """Stamp one base-pair template per position along an ideal helix.

    Each base pair is placed as a RIGID BODY, so every bond length and every
    Watson-Crick hydrogen bond inside it stays exactly as the crystal had it.
    This is the approach step 8a arrived at.
    """
    atoms, bonds, pairs = [], [], []
    for i, base in enumerate(seq):
        kind = base + {"A": "T", "T": "A", "C": "G", "G": "C"}[base]
        t = templates[kind]
        rot, shift = step_transform(i)
        base_index = len(atoms)
        for a in t["atoms"]:
            p = rot @ np.array(a["pos"]) + shift
            atoms.append({
                "el": a["el"],
                "pos": [round(float(v), 4) for v in p],
                "strand": a["strand"],       # 0 = target, 1 = probe side
                "name": a["name"],
                "bp": i,
            })
        for b in t["bonds"]:
            bonds.append([base_index + b[0], base_index + b[1], b[2]])
        pairs.append({"bp": i, "base": base,
                      "hbonds": [[base_index + h[0], base_index + h[1]] for h in t["hbonds"]]})
    return atoms, bonds, pairs


def backbone_joins(atoms):
    """The O3'-P bonds between neighbouring base pairs, per strand.

    Stamping templates onto an ideal helix leaves these joins slightly wrong,
    because the crystal templates were not built for a 10.5 bp/turn helix.
    Step 8a solved this by moving each phosphorus to where both of its bonds
    come out right; here the joins are simply reported, and the tests check
    they are within a believable range.
    """
    spot = {}
    for i, a in enumerate(atoms):
        spot[(a["bp"], a["strand"], a["name"])] = i
    joins = []
    n = max(a["bp"] for a in atoms) + 1
    for bp in range(n - 1):
        # strand 0 runs 5'->3' with increasing bp; strand 1 runs the other way.
        o3 = spot.get((bp, 0, "O3'"))
        p = spot.get((bp + 1, 0, "P"))
        if o3 is not None and p is not None:
            joins.append([o3, p, 0])
        o3b = spot.get((bp + 1, 1, "O3'"))
        pb = spot.get((bp, 1, "P"))
        if o3b is not None and pb is not None:
            joins.append([o3b, pb, 1])
    return joins


def dye_position(atoms, bp, strand, outward):
    """Where a dye sits: on the sugar at the end of a probe, on its linker.

    THIS IS THE STEP'S BIGGEST MODELLING CHOICE, so it is worth being explicit.
    A real dye hangs off the probe's end on a short flexible tether, usually a
    six-carbon linker. It has NO single position: it samples a roughly
    spherical volume a few angstroms across, and every measurement of such a
    system reports an average over that volume.

    What is placed here is the centre of that volume. The linker is taken as
    reaching RADIALLY OUTWARD from the helix - which is where a tether on a
    terminal sugar actually points, away from the duplex rather than along it -
    with a small push along the axis away from the probe's body. Aiming both
    linkers straight down the axis at each other would put the two dyes 1 nm
    apart and interpenetrating, which is geometrically impossible.
    """
    here = [i for i, a in enumerate(atoms) if a["bp"] == bp and a["strand"] == strand]
    if not here:
        raise SystemExit(f"no atoms at bp {bp} strand {strand}")
    c1 = next((i for i in here if atoms[i]["name"] == "C1'"), here[0])
    base = np.array(atoms[c1]["pos"])
    radial = np.array([0.0, base[1], base[2]])
    n = np.linalg.norm(radial)
    radial = radial / n if n > 1e-6 else np.array([0.0, 1.0, 0.0])
    # A C6 linker reaches about 8 A when extended; most of that goes radially.
    return base + radial * 9.5 + np.array([outward * 3.0, 0.0, 0.0])


def main():
    templates, pglo_seq = load_templates()
    seq = pglo_seq[TARGET_START - 1: TARGET_START - 1 + TARGET_LENGTH]
    assert len(seq) == TARGET_LENGTH, "target ran off the end of pGLO"

    atoms, bonds, pairs = build_duplex(seq, templates)
    joins = backbone_joins(atoms)

    # Which atoms belong to which molecule.
    for a in atoms:
        if a["strand"] == 0:
            a["part"] = "target"
        elif PROBE1[0] <= a["bp"] < PROBE1[1]:
            a["part"] = "probe1"
        elif PROBE2[0] <= a["bp"] < PROBE2[1]:
            a["part"] = "probe2"
        else:
            a["part"] = "none"          # the gap, and the flanks: not drawn

    donor = dye_position(atoms, PROBE1[1] - 1, 1, +1)
    acceptor = dye_position(atoms, PROBE2[0], 1, -1)

    # Turn the whole duplex about its axis so BOTH dyes face the camera.
    #
    # The two dyes sit three base pairs apart, which on a 10.5 bp/turn helix is
    # about 103 degrees around it - so on an arbitrary orientation one of them
    # is always round the back. Rotating so that the midpoint between their two
    # radial directions points at the viewer puts each about 51 degrees off
    # centre, which keeps both in sight. This turns the scene, it does not
    # deform it: every distance, including the one driving the transfer, is
    # unchanged.
    def radial_angle(p):
        return math.atan2(p[2], p[1])

    mid = (np.array([0.0, donor[1], donor[2]]) / np.linalg.norm(donor[1:])
           + np.array([0.0, acceptor[1], acceptor[2]]) / np.linalg.norm(acceptor[1:]))
    turn = math.pi / 2 - radial_angle(mid)      # +z is toward the camera
    c, s = math.cos(turn), math.sin(turn)
    spin = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    for a in atoms:
        a["pos"] = [round(float(v), 4) for v in spin @ np.array(a["pos"])]
    donor = spin @ donor
    acceptor = spin @ acceptor

    bound_distance = float(np.linalg.norm(donor - acceptor))

    data = {
        "source": "B-DNA stamped from step008a_plasmid base-pair templates (1BNA crystal)",
        "chemistry": "LightCycler HybProbe: adjacent donor/acceptor probes, FRET",
        "target_sequence": seq,
        "target_start_in_pglo": TARGET_START,
        "rise": RISE,
        "twist": TWIST,
        "probe1": list(PROBE1),
        "probe2": list(PROBE2),
        "gap_nt": GAP_NT,
        "donor_home": [round(float(v), 4) for v in donor],
        "acceptor_home": [round(float(v), 4) for v in acceptor],
        "bound_distance_A": round(bound_distance, 3),
        "atoms": atoms,
        "bonds": bonds + joins,
        "pairs": pairs,
    }
    with open(OUT, "w") as f:
        json.dump(data, f)

    counts = {}
    for a in atoms:
        counts[a["part"]] = counts.get(a["part"], 0) + 1
    print(f"target {seq}")
    print(f"{len(atoms)} atoms, {len(bonds)} intra-pair bonds, {len(joins)} backbone joins")
    print(f"parts: {counts}")
    print(f"donor-acceptor when both probes are bound: {bound_distance:.2f} A "
          f"= {bound_distance / 10:.2f} nm")
    print(f"wrote {os.path.relpath(OUT)} "
          f"({os.path.getsize(OUT) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
