"""The ten steps of glycolysis as edits to labeled molecules.

Each function takes the previous species and returns the next one, plus what
left or arrived: groups of atom labels and where they go or come from. Every
species is checked against its PubChem structure (connectivity and
stereochemistry).
"""

from chemistry import Species, labeled_glucose, match_reference

# The two halves after the split. Half A comes from glucose carbons 1–3 (via
# DHAP), half B from carbons 4–6. In each: the aldehyde carbon (becomes the
# carboxylate), the middle carbon, and the phosphate carbon (becomes the methyl).
HALVES = {
    "A": {"ald": "C3", "ald_o": "O3", "mid": "C2", "mid_o": "O2", "phos": "C1", "phos_o": "O1"},
    "B": {"ald": "C4", "ald_o": "O4", "mid": "C5", "mid_o": "O5", "phos": "C6", "phos_o": "O6"},
}


def checked(sp, reference):
    sp.finish()
    match_reference(sp, reference)
    return sp


# ---------------- Part 1: spend two ATP, split in half ----------------

def hexokinase(glc):
    """glucose + ATP → glucose-6-phosphate + ADP + H⁺"""
    sp = glc.copy("glucose-6-phosphate")
    h = sp.h_on("O6")[0]
    sp.remove(h)
    sp.add_phosphate("Pa", "O6")
    events = {"arrive": [(["Pa", "Paa", "Pab", "Pac"], "ATP1")], "leave_h": [h]}
    return checked(sp, "beta-D-glucose 6-phosphate"), events


def open_glucose_ring(g6p):
    """The ring opens: C1 becomes an aldehyde, O5 takes C1's hydroxyl H."""
    sp = g6p.copy("glucose-6-phosphate (open chain)")
    sp.unbond("O5", "C1")
    sp.set_order("C1", "O1", 2)
    sp.move_h(sp.h_on("O1")[0], "O5")
    return checked(sp, "aldehydo-D-glucose 6-phosphate")


def aldose_to_ketose(open_g6p):
    """The C=O moves from C1 to C2 (via an enediol): C2's H goes to C1."""
    sp = open_g6p.copy("fructose-6-phosphate (open chain)")
    sp.move_h("H2", "C1")
    sp.move_h(sp.h_on("O2")[0], "O1")
    sp.set_order("C1", "O1", 1)
    sp.set_order("C2", "O2", 2)
    return checked(sp, "keto-D-fructose 6-phosphate")


def close_fructose_ring(open_f6p):
    """O5 bonds to C2, closing fructose's five-membered ring."""
    sp = open_f6p.copy("fructose-6-phosphate")
    sp.bond("O5", "C2")
    sp.set_order("C2", "O2", 1)
    sp.move_h(sp.h_on("O5")[0], "O2")
    return checked(sp, "beta-D-fructose 6-phosphate")


def phosphofructokinase(f6p):
    """fructose-6-phosphate + ATP → fructose-1,6-bisphosphate + ADP + H⁺"""
    sp = f6p.copy("fructose-1,6-bisphosphate")
    h = sp.h_on("O1")[0]
    sp.remove(h)
    sp.add_phosphate("Pb", "O1")
    events = {"arrive": [(["Pb", "Pba", "Pbb", "Pbc"], "ATP2")], "leave_h": [h]}
    return checked(sp, "beta-D-fructose 1,6-bisphosphate"), events


def open_fructose_ring(f16bp):
    sp = f16bp.copy("fructose-1,6-bisphosphate (open chain)")
    sp.unbond("O5", "C2")
    sp.set_order("C2", "O2", 2)
    sp.move_h(sp.h_on("O2")[0], "O5")
    return checked(sp, "keto-D-fructose 1,6-bisphosphate")


def split_off(sp, keep_carbons, name):
    """Keeps only the atoms attached (through bonds) to the given carbons."""
    keep = set()
    todo = [sp.idx(c) for c in keep_carbons]
    while todo:
        i = todo.pop()
        if i in keep:
            continue
        keep.add(i)
        todo.extend(n.GetIdx() for n in sp.m.GetAtomWithIdx(i).GetNeighbors())
    half = sp.copy(name)
    for lbl in [a.GetProp("lbl") for a in sp.m.GetAtoms() if a.GetIdx() not in keep]:
        half.remove(lbl)
    return half


def aldolase(open_f16bp):
    """The C3–C4 bond breaks: DHAP (carbons 1–3) + glyceraldehyde-3-phosphate (4–6)."""
    sp = open_f16bp.copy("split")
    sp.unbond("C3", "C4")
    sp.set_order("C4", "O4", 2)
    sp.move_h(sp.h_on("O4")[0], "C3")
    dhap = checked(split_off(sp, ["C1"], "dihydroxyacetone phosphate"), "dihydroxyacetone phosphate")
    g3p = checked(split_off(sp, ["C4"], "glyceraldehyde-3-phosphate"), "D-glyceraldehyde 3-phosphate")
    return dhap, g3p


def triose_phosphate_isomerase(dhap):
    """DHAP → glyceraldehyde-3-phosphate: C3 becomes the aldehyde, C2 takes an H."""
    sp = dhap.copy("glyceraldehyde-3-phosphate (from DHAP)")
    sp.move_h("H3", "C2")
    sp.move_h(sp.h_on("O3")[0], "O2")
    sp.set_order("C2", "O2", 1)
    sp.set_order("C3", "O3", 2)
    return checked(sp, "D-glyceraldehyde 3-phosphate")


# ---------------- Part 2: the payoff, for one half ----------------

def gapdh(g3p, half):
    """G3P + NAD⁺ + HPO₄²⁻ → 1,3-bisphosphoglycerate + NADH + H⁺.
    The aldehyde H goes to NAD⁺ (as a hydride), a free phosphate joins the
    carbon, and the phosphate's own H leaves as H⁺."""
    r = HALVES[half]
    sp = g3p.copy("1,3-bisphosphoglycerate")
    hydride = sp.h_on(r["ald"])[0]
    sp.remove(hydride)
    q = "Q" + half
    sp.add("P", q)
    sp.add("O", q + "a")          # P=O
    sp.add("O", q + "b")          # bridges to the carbon
    sp.add("O", q + "c", -1)
    sp.add("O", q + "d", -1)      # its H leaves as H⁺
    sp.bond(q, q + "a", 2)
    sp.bond(q, q + "b")
    sp.bond(q, q + "c")
    sp.bond(q, q + "d")
    sp.bond(r["ald"], q + "b")
    events = {
        "hydride": hydride,
        "phosphate_in": [q, q + "a", q + "b", q + "c", q + "d"],
        "phosphate_h": q + "h",       # travels with the phosphate, then leaves
        "phosphate_h_on": q + "d",
    }
    return checked(sp, "3-phospho-D-glyceroyl phosphate"), events


def phosphoglycerate_kinase(bpg, half):
    """1,3-bisphosphoglycerate + ADP → 3-phosphoglycerate + ATP"""
    r = HALVES[half]
    sp = bpg.copy("3-phosphoglycerate")
    q = "Q" + half
    group = [q, q + "a", q + "c", q + "d"]
    for lbl in group:
        sp.remove(lbl)
    sp.charge(q + "b", -1)       # stays on the carbon: the carboxylate's O⁻
    return checked(sp, "3-phospho-D-glycerate"), {"leave": group}


def phosphoglycerate_mutase(pg3, half):
    """3-phosphoglycerate → 2-phosphoglycerate: the phosphate moves from the
    end carbon's oxygen to the middle carbon's oxygen."""
    r = HALVES[half]
    sp = pg3.copy("2-phosphoglycerate")
    group = sp.phosphate_group(r["phos_o"])
    sp.unbond(r["phos_o"], group[0])
    sp.bond(r["mid_o"], group[0])
    sp.move_h(sp.h_on(r["mid_o"])[0], r["phos_o"])
    return checked(sp, "2-phospho-D-glycerate"), {"moved": group}


def enolase(pg2, half):
    """2-phosphoglycerate → phosphoenolpyruvate + H₂O"""
    r = HALVES[half]
    sp = pg2.copy("phosphoenolpyruvate")
    water_o = r["phos_o"]
    water_h = sp.h_on(water_o) + sp.h_on(r["mid"])
    for lbl in [water_o] + water_h:
        sp.remove(lbl)
    sp.set_order(r["mid"], r["phos"], 2)
    return checked(sp, "phosphoenolpyruvate"), {"water": [water_o] + water_h}


def pyruvate_kinase(pep, half):
    """phosphoenolpyruvate + ADP + H⁺ → pyruvate + ATP"""
    r = HALVES[half]
    sp = pep.copy("pyruvate")
    group = sp.phosphate_group(r["mid_o"])
    for lbl in group:
        sp.remove(lbl)
    sp.set_order(r["mid"], r["mid_o"], 2)
    sp.set_order(r["mid"], r["phos"], 1)
    h = "Hp" + half
    sp.add("H", h)
    sp.bond(r["phos"], h)
    return checked(sp, "pyruvic acid"), {"leave": group, "proton_in": h}


def run():
    """All species in order, as a quick self-check."""
    glc = labeled_glucose()
    g6p, _ = hexokinase(glc)
    g6p_open = open_glucose_ring(g6p)
    f6p_open = aldose_to_ketose(g6p_open)
    f6p = close_fructose_ring(f6p_open)
    f16bp, _ = phosphofructokinase(f6p)
    f16bp_open = open_fructose_ring(f16bp)
    dhap, g3p_b = aldolase(f16bp_open)
    g3p_a = triose_phosphate_isomerase(dhap)
    chain = [glc, g6p, g6p_open, f6p_open, f6p, f16bp, f16bp_open, dhap, g3p_b, g3p_a]
    for half, g3p in (("A", g3p_a), ("B", g3p_b)):
        bpg, _ = gapdh(g3p, half)
        pg3, _ = phosphoglycerate_kinase(bpg, half)
        pg2, _ = phosphoglycerate_mutase(pg3, half)
        pep, _ = enolase(pg2, half)
        pyr, _ = pyruvate_kinase(pep, half)
        chain += [bpg, pg3, pg2, pep, pyr]
    return chain


if __name__ == "__main__":
    for sp in run():
        carbons = sorted(l for l in sp.labels() if l.startswith("C"))
        print(f"{sp.name:45s} charge {sp.charge_total():+d}  {sp.formula_counts()}  carbons {carbons}")
