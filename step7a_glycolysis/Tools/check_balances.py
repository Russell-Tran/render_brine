"""Checks that every glycolysis reaction in pathway.py balances, atom for atom
and charge for charge, and that the ten steps add up to the textbook net
equation.

The sugars come straight from pathway.py (built by bond edits, each checked
against PubChem). The helpers that are drawn as tokens are written out here,
in their dominant form at pH 7:

  ATP⁴⁻   C10H12N5O13P3   (PubChem CID 5957, fully ionized)
  ADP³⁻   C10H12N5O10P2   (PubChem CID 6022, fully ionized)
  NAD⁺    C21H26N7O14P2⁻  (PubChem CID 5892: nicotinamide +1, pyrophosphate −2 → net −1)
  NADH    C21H27N7O14P2²⁻ (PubChem CID 439153, pyrophosphate −2)
  HPO₄²⁻  inorganic phosphate, the main form at pH 7.2 (pKa₂ ≈ 7.2)
  H⁺, H₂O

Net (Berg, Tymoczko & Gatto, Biochemistry, ch. 16):
  glucose + 2 NAD⁺ + 2 ADP + 2 Pᵢ → 2 pyruvate + 2 NADH + 2 H⁺ + 2 ATP + 2 H₂O
"""

from collections import Counter

import pathway as pw

HELPERS = {
    "ATP": (Counter(C=10, H=12, N=5, O=13, P=3), -4),
    "ADP": (Counter(C=10, H=12, N=5, O=10, P=2), -3),
    "NAD+": (Counter(C=21, H=26, N=7, O=14, P=2), -1),
    "NADH": (Counter(C=21, H=27, N=7, O=14, P=2), -2),
    "Pi": (Counter(H=1, O=4, P=1), -2),
    "H+": (Counter(H=1), +1),
    "H2O": (Counter(H=2, O=1), 0),
}


def side(items):
    """Total atoms and charge for a list of species objects and helper names."""
    atoms, charge = Counter(), 0
    for item in items:
        if isinstance(item, str):
            f, q = HELPERS[item]
        else:
            f, q = Counter(item.formula_counts()), item.charge_total()
        atoms += f
        charge += q
    return atoms, charge


def check(name, left, right):
    la, lq = side(left)
    ra, rq = side(right)
    ok = la == ra and lq == rq
    print(f"{'ok  ' if ok else 'FAIL'} {name:32s} atoms {dict(sorted(la.items()))}  charge {lq:+d} → {rq:+d}")
    if not ok:
        print(f"       left {dict(la)} {lq:+d}\n       right {dict(ra)} {rq:+d}")
    return ok


def main():
    glc = pw.labeled_glucose()
    g6p, _ = pw.hexokinase(glc)
    g6p_open = pw.open_glucose_ring(g6p)
    f6p_open = pw.aldose_to_ketose(g6p_open)
    f6p = pw.close_fructose_ring(f6p_open)
    f16bp, _ = pw.phosphofructokinase(f6p)
    f16bp_open = pw.open_fructose_ring(f16bp)
    dhap, g3p_b = pw.aldolase(f16bp_open)
    g3p_a = pw.triose_phosphate_isomerase(dhap)

    results = [
        check("hexokinase", [glc, "ATP"], [g6p, "ADP", "H+"]),
        check("phosphoglucose isomerase", [g6p], [f6p]),
        check("phosphofructokinase-1", [f6p, "ATP"], [f16bp, "ADP", "H+"]),
        check("aldolase", [f16bp], [dhap, g3p_b]),
        check("triose phosphate isomerase", [dhap], [g3p_a]),
    ]
    pyruvates = []
    for half, g3p in (("A", g3p_a), ("B", g3p_b)):
        bpg, _ = pw.gapdh(g3p, half)
        pg3, _ = pw.phosphoglycerate_kinase(bpg, half)
        pg2, _ = pw.phosphoglycerate_mutase(pg3, half)
        pep, _ = pw.enolase(pg2, half)
        pyr, _ = pw.pyruvate_kinase(pep, half)
        pyruvates.append(pyr)
        results += [
            check(f"GAPDH ({half})", [g3p, "NAD+", "Pi"], [bpg, "NADH", "H+"]),
            check(f"phosphoglycerate kinase ({half})", [bpg, "ADP"], [pg3, "ATP"]),
            check(f"phosphoglycerate mutase ({half})", [pg3], [pg2]),
            check(f"enolase ({half})", [pg2], [pep, "H2O"]),
            check(f"pyruvate kinase ({half})", [pep, "ADP", "H+"], [pyr, "ATP"]),
        ]
    results.append(check("net glycolysis",
                         [glc, "NAD+", "NAD+", "ADP", "ADP", "Pi", "Pi"],
                         pyruvates + ["NADH", "NADH", "H+", "H+", "ATP", "ATP", "H2O", "H2O"]))

    # Carbon bookkeeping: glucose's carbons 1–3 end in pyruvate A, 4–6 in B.
    carbons = [sorted(l for l in p.labels() if l[0] == "C" and l[1:].isdigit()) for p in pyruvates]
    track = carbons == [["C1", "C2", "C3"], ["C4", "C5", "C6"]]
    print(f"{'ok  ' if track else 'FAIL'} carbon tracking: pyruvate A {carbons[0]}, pyruvate B {carbons[1]}")
    results.append(track)

    print(f"\n{sum(results)}/{len(results)} checks pass")
    raise SystemExit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
