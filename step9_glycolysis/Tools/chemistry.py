"""The chemistry of glycolysis, atom by atom.

Starts from β-D-glucose (PubChem CID 64689) with every atom given a permanent
label (C1..C6 use glucose's numbering, H's and O's are named after the atom
they start on). Each enzyme step is written as edits to the molecule: bonds
made and broken, hydrogens moved, phosphate groups added or removed. So every
atom can be followed from glucose to pyruvate.

Charges are the forms found near pH 7: phosphate monoesters −2, carboxylates
−1. Each intermediate is checked against PubChem (connectivity and
stereochemistry) after putting the acidic H's back, since PubChem stores the
neutral acids.

3D shapes come from RDKit (ETKDG embedding + MMFF94 force field).
"""

import json
import os

from rdkit import Chem
from rdkit.Chem import AllChem, rdCIPLabeler

HERE = os.path.dirname(os.path.abspath(__file__))
REFERENCES = json.load(open(os.path.join(HERE, "..", "Resources", "pubchem_references.json")))


def ref_smiles(name):
    entry = REFERENCES[name]
    return entry.get("IsomericSMILES") or entry.get("SMILES")


class Species:
    """A molecule (or a few) whose atoms all carry a permanent 'lbl'."""

    def __init__(self, mol, name=""):
        self.m = Chem.RWMol(mol)
        self.name = name

    def copy(self, name):
        return Species(Chem.Mol(self.m), name)

    # --- lookups ---
    def idx(self, lbl):
        for a in self.m.GetAtoms():
            if a.GetProp("lbl") == lbl:
                return a.GetIdx()
        raise KeyError(lbl)

    def atom(self, lbl):
        return self.m.GetAtomWithIdx(self.idx(lbl))

    def labels(self):
        return [a.GetProp("lbl") for a in self.m.GetAtoms()]

    def h_on(self, lbl):
        """Labels of hydrogens bonded to an atom."""
        return [n.GetProp("lbl") for n in self.atom(lbl).GetNeighbors() if n.GetSymbol() == "H"]

    # --- edits ---
    def add(self, element, lbl, charge=0):
        a = Chem.Atom(element)
        a.SetFormalCharge(charge)
        a.SetNoImplicit(True)
        a.SetProp("lbl", lbl)
        self.m.AddAtom(a)

    def bond(self, a, b, order=1):
        t = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE}[order]
        self.m.AddBond(self.idx(a), self.idx(b), t)

    def unbond(self, a, b):
        self.m.RemoveBond(self.idx(a), self.idx(b))

    def set_order(self, a, b, order):
        t = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE}[order]
        self.m.GetBondBetweenAtoms(self.idx(a), self.idx(b)).SetBondType(t)

    def charge(self, lbl, q):
        self.atom(lbl).SetFormalCharge(q)

    def remove(self, lbl):
        self.m.RemoveAtom(self.idx(lbl))

    def move_h(self, h, new_parent):
        old = [n.GetProp("lbl") for n in self.atom(h).GetNeighbors()][0]
        self.unbond(h, old)
        self.bond(h, new_parent)

    def add_phosphate(self, prefix, onto):
        """Attach a PO₃²⁻ group (from ATP) to oxygen `onto`: P, =O, O⁻, O⁻."""
        self.add("P", prefix)
        self.add("O", prefix + "a")
        self.add("O", prefix + "b", -1)
        self.add("O", prefix + "c", -1)
        self.bond(onto, prefix)
        self.bond(prefix, prefix + "a", 2)
        self.bond(prefix, prefix + "b")
        self.bond(prefix, prefix + "c")

    def phosphate_group(self, bridging_oxygen):
        """Labels of the PO₃ group hanging off an oxygen (P and its other three O's)."""
        p = [n for n in self.atom(bridging_oxygen).GetNeighbors() if n.GetSymbol() == "P"][0]
        others = [n.GetProp("lbl") for n in p.GetNeighbors() if n.GetProp("lbl") != bridging_oxygen]
        return [p.GetProp("lbl")] + others

    # --- chemistry checks ---
    def finish(self):
        for a in self.m.GetAtoms():
            a.SetNoImplicit(True)
            a.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
        Chem.SanitizeMol(self.m)

    def charge_total(self):
        return sum(a.GetFormalCharge() for a in self.m.GetAtoms())

    def formula_counts(self):
        counts = {}
        for a in self.m.GetAtoms():
            counts[a.GetSymbol()] = counts.get(a.GetSymbol(), 0) + 1
        return counts


def neutral_copy(mol):
    """The same molecule with H⁺ put back on every O⁻ (PubChem's neutral acid form)."""
    rw = Chem.RWMol(mol)
    for a in list(rw.GetAtoms()):
        if a.GetSymbol() == "O" and a.GetFormalCharge() == -1:
            a.SetFormalCharge(0)
            h = Chem.Atom("H")
            h.SetNoImplicit(True)
            h.SetProp("lbl", "_acidH")
            i = rw.AddAtom(h)
            rw.AddBond(a.GetIdx(), i, Chem.BondType.SINGLE)
    Chem.SanitizeMol(rw)
    return rw


def cip_codes(mol):
    rdCIPLabeler.AssignCIPLabels(mol)
    return {a.GetIdx(): a.GetProp("_CIPCode") for a in mol.GetAtoms() if a.HasProp("_CIPCode")}


def match_reference(sp, ref_name):
    """Sets stereocenters to match PubChem and checks the whole structure agrees.
    Fragments (like two separate halves) are checked one reference each."""
    ref = Chem.AddHs(Chem.MolFromSmiles(ref_smiles(ref_name)))
    # PubChem sometimes stores a charged form; neutralize it the same way.
    ref = neutral_copy(ref)
    ref_cip = cip_codes(ref)
    ref_heavy = Chem.RemoveHs(ref)

    neutral = neutral_copy(sp.m)
    match = neutral.GetSubstructMatch(ref_heavy)
    if not match:
        raise AssertionError(f"{sp.name}: connectivity doesn't match PubChem {ref_name}")
    # Map reference heavy-atom indices back to the AddHs reference to read CIP codes.
    heavy_to_full = [a.GetIdx() for a in ref.GetAtoms() if a.GetSymbol() != "H"]
    for ref_heavy_idx, my_idx in enumerate(match):
        want = ref_cip.get(heavy_to_full[ref_heavy_idx])
        if want is None:
            continue
        for tag in (Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW):
            sp.m.GetAtomWithIdx(my_idx).SetChiralTag(tag)
            if cip_codes(neutral_copy(sp.m)).get(my_idx) == want:
                break
        else:
            raise AssertionError(f"{sp.name}: couldn't match stereo at atom {my_idx}")
    mine = Chem.MolToSmiles(Chem.RemoveHs(neutral_copy(sp.m)))
    theirs = Chem.MolToSmiles(Chem.RemoveHs(ref))
    if mine != theirs:
        raise AssertionError(f"{sp.name}: {mine} != PubChem {theirs}")
    return mine


def labeled_glucose():
    m = Chem.AddHs(Chem.MolFromSmiles(ref_smiles("beta-D-glucose")))
    ring_o = [a for a in m.GetAtoms() if a.GetSymbol() == "O" and a.IsInRing()][0]
    def oxygens(a):
        return [n for n in a.GetNeighbors() if n.GetSymbol() == "O"]
    ring_cs = list(ring_o.GetNeighbors())
    c1 = [a for a in ring_cs if len(oxygens(a)) == 2][0]
    c5 = [a for a in ring_cs if a.GetIdx() != c1.GetIdx()][0]
    carbons = {1: c1, 5: c5}
    prev, cur = ring_o, c1
    for n in (2, 3, 4):
        nxt = [x for x in cur.GetNeighbors() if x.GetSymbol() == "C" and x.GetIdx() != prev.GetIdx()][0]
        carbons[n] = nxt
        prev, cur = cur, nxt
    carbons[6] = [x for x in c5.GetNeighbors() if x.GetSymbol() == "C" and not x.IsInRing()][0]
    ring_o.SetProp("lbl", "O5")
    for n, c in carbons.items():
        c.SetProp("lbl", f"C{n}")
        hs = [x for x in c.GetNeighbors() if x.GetSymbol() == "H"]
        if len(hs) == 1:
            hs[0].SetProp("lbl", f"H{n}")
        else:
            for h, s in zip(hs, "ab"):
                h.SetProp("lbl", f"H{n}{s}")
        for o in oxygens(c):
            if o.GetIdx() == ring_o.GetIdx():
                continue
            o.SetProp("lbl", f"O{n}")
            for h in o.GetNeighbors():
                if h.GetSymbol() == "H":
                    h.SetProp("lbl", f"HO{n}")
    for a in m.GetAtoms():
        assert a.HasProp("lbl"), f"unlabeled atom {a.GetSymbol()}"
    sp = Species(m, "β-D-glucose")
    sp.finish()
    match_reference(sp, "beta-D-glucose")
    return sp
