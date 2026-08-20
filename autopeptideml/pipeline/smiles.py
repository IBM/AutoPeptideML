import copy
import os.path as osp

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import xml.etree.ElementTree as ET

from .pipeline import BaseElement

try:
    import rdkit.Chem.rdmolfiles as rdm
    import rdkit.Chem.rdmolops as rdops

    from rdkit import Chem, rdBase
    from rdkit.Chem import (
        DataStructs,
        MolFromSmiles,
        MolToSmiles,
        RWMol,
        Mol,
    )
    from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect

except ImportError:
    raise ImportError(
        "You need to install rdkit to use this method. "
        "Try: `pip install rdkit`"
    )


# ============================================================================
# MONOMER LIBRARY
# ============================================================================

def read_chembl_library(path: str) -> Dict[str, Tuple[str, str]]:
    """
    Load the ChEMBL monomer library.

    Returns
    -------
    Dict[str, Tuple[str, str]]
        Monomer ID -> (SMILES, natural analogue)
    """
    tree = ET.parse(path)
    root = tree.getroot()

    if "}" in root.tag:
        namespace = root.tag[root.tag.find("{") + 1:root.tag.find("}")]
        ns = {"lmr": namespace}
        monomer_xpath = ".//lmr:Monomer"
    else:
        ns = {}
        monomer_xpath = ".//Monomer"

    monomers = {}

    for monomer in root.findall(monomer_xpath, ns):
        def text(name):
            return monomer.findtext(
                f"lmr:{name}" if ns else name,
                default="",
                namespaces=ns,
            )

        monomer_id = text("MonomerID")
        smiles = text("MonomerSmiles")
        natural_analog = text("NaturalAnalog")

        if monomer_id:
            monomers[monomer_id] = (
                smiles,
                natural_analog,
            )

    return monomers


AAs = [
    "A", "C", "D", "E", "F", "G", "H", "H2", "I", "K", "L",
    "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y",
]

AA_DICT = read_chembl_library(
    osp.join(
        osp.dirname(__file__),
        "..",
        "data",
        "chembl_monomer_library.xml",
    )
)

# Custom histidine analogue used by the original implementation.
AA_DICT.update({
    "H2": (
        "[*]N[C@@H](Cc1c[nH]cn1)C([*])=O |"
        "$_R1;;;;;;;;;;_R2;$|",
        "H",
    )
})

CACHE = {}
CACHE_ACHIRAL = {}

# Map non-canonical library names to their canonical BILN equivalents.
# lalloI (L-allo-Isoleucine) and I both represent Isoleucine-class residues;
# the biologically canonical form is "I".
_CANONICAL_NAME_MAP = {
    "lalloI": "I",
}


# ============================================================================
# BASIC HELPERS
# ============================================================================

def _strip_smiles_properties(smiles: str) -> str:
    """
    Remove CXSMILES/property information from a SMILES string.
    """
    return smiles.split(" ")[0]


def _mol_from_library_smiles(smiles: str) -> Mol:
    """
    Read a monomer SMILES, including CXSMILES atom labels.

    The full CXSMILES string (including the |$...$| block) must be
    preserved so that atomLabel properties (_R1, _R2, _R3 …) survive
    into the returned molecule.  Stripping the CXSMILES suffix before
    parsing loses those labels.
    """
    mol = MolFromSmiles(smiles, sanitize=True)

    if mol is None:
        mol = MolFromSmiles(smiles, sanitize=False)

        if mol is None:
            raise ValueError(
                f"Could not parse monomer SMILES: {smiles}"
            )

        Chem.SanitizeMol(mol)

    return mol


def _get_atom_label(atom: Chem.Atom) -> Optional[str]:
    """
    Return the atomLabel property if present.
    """
    if atom.HasProp("atomLabel"):
        return atom.GetProp("atomLabel")

    return None


def _rgroup_from_label(label: Optional[str]) -> Optional[int]:
    """
    Convert '_R1', '_R2', '_R3' ... into integer R-group numbers.
    """
    if not label:
        return None

    if label.startswith("_R"):
        try:
            return int(label[2:])
        except ValueError:
            return None

    return None


# ============================================================================
# SEQUENCE <-> SMILES
# ============================================================================

class SequenceToSmiles(BaseElement):
    name = "sequence-to-smiles"
    parallel = "processing"

    def _single_call(self, mol):
        rd_mol = rdm.MolFromFASTA(mol)

        if rd_mol is None:
            raise RuntimeError(
                f"Molecule: {mol} could not be read by RDKit.",
                "Maybe introduce a filtering step in your pipeline",
            )

        return rdm.MolToSmiles(
            rd_mol,
            canonical=True,
            isomericSmiles=True,
        )


class SmilesToSequence(BaseElement):
    name = "smiles-to-sequence"
    parallel = "processing"

    def __init__(self, keep_analog: bool = True):
        self.keep_analog = keep_analog

    def _single_call(self, mol):
        final_pep, _, _chain_sizes = break_into_monomers(mol)

        if not isinstance(final_pep, list):
            raise ValueError(mol, final_pep)

        if self.keep_analog:
            final_pep = [
                AA_DICT[r][1]
                if r not in ["X", "ac", "am"]
                else r
                for r in final_pep
                if r in AA_DICT or r in ["X", "ac", "am"]
            ]

            final_pep = [
                r for r in final_pep
                if r not in ["ac", "am"]
            ]

        else:
            final_pep = [
                r if r in AAs else "X"
                for r in final_pep
            ]

        return "".join(final_pep)


# ============================================================================
# SMILES -> BILN
# ============================================================================

def _merge_caps_human_readable(biln: str) -> str:
    """
    Merge terminal cap tokens into adjacent residue names.

    In human-readable BILN notation, the ``ac`` (N-acetyl) and ``am``
    (C-terminal amide) cap monomers are folded into the neighbouring
    residue token as a suffix rather than appearing as separate tokens:

        ac-W-Y-C-G-am  →  Wac-Y-C-Gam

    Rules applied per chain:
    * If the first token of a chain is ``ac``, remove it and append
      ``"ac"`` to the second token.
    * If the last token of a chain is ``am``, remove it and append
      ``"am"`` to the penultimate token.

    Annotations (parenthesised suffixes such as ``(1,3)``) on the cap
    token are preserved by moving them onto the merged result.
    """
    result_chains = []

    for chain_str in biln.split("."):
        tokens = chain_str.split("-")

        # --- N-terminal acetyl cap ---
        if tokens and tokens[0] == "ac":
            tokens.pop(0)
            if tokens:
                tokens[0] = tokens[0] + "ac"

        # --- C-terminal amide cap ---
        # The last token may be bare "am" or annotated like "am(1,2)".
        if tokens:
            last = tokens[-1]
            base_last = last.split("(")[0]   # strip any annotations
            suffix_last = last[len(base_last):]  # annotations portion
            if base_last == "am":
                tokens.pop()
                if tokens:
                    tokens[-1] = tokens[-1] + "am" + suffix_last

        result_chains.append("-".join(tokens))

    return ".".join(result_chains)


class SmilesToBiln(BaseElement):
    """
    Convert a peptide SMILES into topology-aware BILN.

    Ordinary peptide backbone connections are represented by '-'.

    Non-backbone connections are represented as:

        MONOMER(BOND_ID,R_GROUP)

    For example:

        C(1,3)-A-A-A-C(1,3)

    represents a disulfide bridge.

    Head-to-tail cyclization:

        C(1,1)-A-A-A-C(1,2)
    """

    name = "smiles-to-biln"
    parallel = "processing"
    human_readable = False

    def __init__(self, human_readable: bool = False, handle_errors: bool = False):
        self.human_readable = human_readable
        self.handle_errors = handle_errors

    def _single_call(self, mol):
        monomer_names, fragments, chain_sizes = break_into_monomers(mol)

        if not fragments:
            if self.handle_errors:
                return "X"
            raise ValueError(
                f"No peptide fragments found in molecule: {mol!r}"
            )

        # Reject non-peptide inputs: if every residue is unrecognised ("X")
        # AND the molecule contains no amino-acid backbone motif (N-C-C=O),
        # the input is not a peptide.
        if all(name == "X" for name in monomer_names):
            raw_mol = MolFromSmiles(mol) if isinstance(mol, str) else mol
            backbone_patt = Chem.MolFromSmarts("NCC(=O)")
            is_peptide_like = (
                raw_mol is not None
                and raw_mol.HasSubstructMatch(backbone_patt)
            )
            if not is_peptide_like:
                if self.handle_errors:
                    return "X"
                raise ValueError(
                    f"Input does not appear to be a peptide: {mol!r}"
                )

        biln = fragments_to_biln(
            fragments=fragments,
            monomer_names=monomer_names,
            chain_sizes=chain_sizes,
        )

        if self.human_readable:
            biln = _merge_caps_human_readable(biln)

        return biln


# ============================================================================
# BILN -> SMILES
# ============================================================================

class BilnToSmiles(BaseElement):
    """
    Convert BILN into an RDKit SMILES.

    Supports:
        A-G-C
        C(1,3)-A-A-A-C(1,3)
        C(1,1)-A-A-A-C(1,2)
        A-G.K-E
        A-G-K(1,3)-D.D-E-A(1,2)

    Explicit BILN bonds are reconstructed using their R-group numbers.
    """

    name = "biln-to-smiles"
    parallel = "processing"

    def _single_call(self, biln):
        return biln_to_smiles(biln)


# ============================================================================
# BILN TOKENIZER / PARSER
# ============================================================================

def tokenize_biln_chain(chain: str) -> List[str]:
    """
    Split one BILN chain into monomer tokens.

    Examples
    --------
    A-G-C
        -> ["A", "G", "C"]

    C(1,3)-A-A-C(1,3)
        -> ["C(1,3)", "A", "A", "C(1,3)"]

    K(1,3)(2,3)
        -> ["K(1,3)(2,3)"]

    [2-Cl-Phe]-A
        -> ["[2-Cl-Phe]", "A"]
    """
    tokens = []
    current = []
    bracket_depth = 0
    paren_depth = 0

    for char in chain:
        if char == "[":
            bracket_depth += 1
            current.append(char)

        elif char == "]":
            bracket_depth -= 1
            current.append(char)

        elif char == "(":
            paren_depth += 1
            current.append(char)

        elif char == ")":
            paren_depth -= 1
            current.append(char)

        elif (
            char == "-"
            and bracket_depth == 0
            and paren_depth == 0
        ):
            if current:
                tokens.append("".join(current))
                current = []

        else:
            current.append(char)

    if current:
        tokens.append("".join(current))

    if bracket_depth != 0:
        raise ValueError(
            f"Unbalanced '[' in BILN chain: {chain}"
        )

    if paren_depth != 0:
        raise ValueError(
            f"Unbalanced '(' in BILN chain: {chain}"
        )

    return tokens


def parse_biln_token(token: str):
    """
    Parse:

        A
        C(1,3)
        K(1,3)(2,3)
        [2-Cl-Phe](1,3)

    Returns
    -------
    symbol, [(bond_id, rgroup), ...]
    """
    symbol_end = token.find("(")

    if symbol_end == -1:
        return token, []

    symbol = token[:symbol_end]

    annotations = []
    pos = symbol_end

    while pos < len(token):
        if token[pos] != "(":
            raise ValueError(
                f"Invalid BILN token: {token}"
            )

        end = token.find(")", pos)

        if end == -1:
            raise ValueError(
                f"Unclosed BILN annotation: {token}"
            )

        content = token[pos + 1:end]
        parts = content.split(",")

        if len(parts) != 2:
            raise ValueError(
                f"Invalid BILN annotation: ({content})"
            )

        try:
            bond_id = int(parts[0])
            rgroup = int(parts[1])
        except ValueError:
            raise ValueError(
                f"Invalid BILN annotation: ({content})"
            )

        annotations.append((bond_id, rgroup))
        pos = end + 1

    return symbol, annotations


def parse_biln(biln: str):
    """
    Parse complete BILN.

    Dots represent independent chains/components.
    """
    chains = []

    for chain_text in biln.split("."):
        chain_text = chain_text.strip()

        if not chain_text:
            raise ValueError(
                f"Empty BILN chain in: {biln}"
            )

        tokens = tokenize_biln_chain(chain_text)

        chain = []

        for token in tokens:
            symbol, annotations = parse_biln_token(token)

            chain.append({
                "symbol": symbol,
                "annotations": annotations,
            })

        chains.append(chain)

    return chains


# ============================================================================
# BILN VALIDATION
# ============================================================================

def validate_biln(biln: str):
    """
    Validate BILN bond identifiers.

    Every explicit bond ID must occur exactly twice.
    """
    chains = parse_biln(biln)

    occurrences = defaultdict(list)

    for chain_idx, chain in enumerate(chains):
        for monomer_idx, monomer in enumerate(chain):
            for bond_id, rgroup in monomer["annotations"]:
                occurrences[bond_id].append(
                    (
                        chain_idx,
                        monomer_idx,
                        rgroup,
                    )
                )

    for bond_id, items in occurrences.items():
        if len(items) != 2:
            raise ValueError(
                f"BILN bond ID {bond_id} occurs "
                f"{len(items)} times; it must occur exactly twice."
            )

        for _, _, rgroup in items:
            if rgroup < 1:
                raise ValueError(
                    f"Invalid R-group R{rgroup} "
                    f"for bond {bond_id}."
                )

    return True


# ============================================================================
# BILN -> SMILES
# ============================================================================

def biln_to_smiles(biln: str) -> str:
    """
    Convert topology-aware BILN into SMILES.

    The monomer library supplies R1/R2/R3 attachment labels.

    Standard '-' connections are interpreted as:
        previous R2 -> next R1

    Explicit connections are interpreted from their
    (BondID,RGroup) annotations.
    """
    validate_biln(biln)

    chains = parse_biln(biln)

    # ------------------------------------------------------------
    # Instantiate every monomer.
    # ------------------------------------------------------------
    chain_mols = []

    for chain in chains:
        mols = []

        for entry in chain:
            symbol = entry["symbol"]

            if symbol == "X":
                print(
                    "Warning: Monomer X is being substituted by G."
                )
                symbol = "G"

            if symbol not in AA_DICT:
                raise KeyError(
                    f"Unknown BILN monomer: {symbol}"
                )

            mols.append(
                _mol_from_library_smiles(
                    AA_DICT[symbol][0]
                )
            )

        chain_mols.append(mols)

    # ------------------------------------------------------------
    # Build normal backbone bonds first.
    # ------------------------------------------------------------
    assembled = []

    for mols, chain in zip(chain_mols, chains):
        if not mols:
            continue

        result = mols[0]

        for idx in range(1, len(mols)):
            result = _combine_fragments(
                result,
                mols[idx],
            )

        assembled.append(result)

    # ------------------------------------------------------------
    # Add explicit BILN connections.
    # ------------------------------------------------------------
    bond_map = defaultdict(list)

    for chain_idx, chain in enumerate(chains):
        for monomer_idx, entry in enumerate(chain):
            for bond_id, rgroup in entry["annotations"]:
                bond_map[bond_id].append(
                    {
                        "chain": chain_idx,
                        "monomer": monomer_idx,
                        "rgroup": rgroup,
                    }
                )

    # ------------------------------------------------------------
    # For explicit bonds we need the original monomer atom
    # positions. Build them independently, then combine all
    # components with the requested bond.
    #
    # This is intentionally kept conservative: explicit bonds
    # that coincide with backbone bonds are skipped because the
    # backbone has already been constructed.
    # ------------------------------------------------------------
    if len(assembled) == 1:
        result = assembled[0]
    else:
        result = assembled[0]

        for component in assembled[1:]:
            result = Chem.CombineMols(result, component)

    # NOTE:
    # Full arbitrary inter-chain BILN assembly requires preserving
    # monomer atom offsets through the backbone assembly. For the
    # common single-chain cyclic/disulfide use case, use the
    # topology-aware construction below.
    #
    # Rebuild from scratch for explicit topology.
    return _build_biln_topology(biln)


def _find_rgroup_atom(mol: Mol, rgroup: int) -> Optional[int]:
    """
    Find the atom carrying _R<n>.
    """
    target = f"_R{rgroup}"

    for atom in mol.GetAtoms():
        if (
            atom.HasProp("atomLabel")
            and atom.GetProp("atomLabel") == target
        ):
            return atom.GetIdx()

    return None


def _build_biln_topology(biln: str) -> str:
    """
    Robust topology builder for BILN.

    Constructs all monomers independently, connects:
        - implicit R2 -> R1 backbone bonds
        - explicit (BondID,RGroup) bonds

    This preserves cycles and branches.
    """
    chains = parse_biln(biln)

    # ------------------------------------------------------------
    # Create one global RWMol.
    # ------------------------------------------------------------
    rw = Chem.RWMol()

    atom_offsets = {}
    monomer_mols = {}

    # Add all monomers.
    for chain_idx, chain in enumerate(chains):
        for monomer_idx, entry in enumerate(chain):
            symbol = entry["symbol"]

            if symbol == "X":
                symbol = "G"

            if symbol not in AA_DICT:
                raise KeyError(
                    f"Unknown BILN monomer: {symbol}"
                )

            mol = _mol_from_library_smiles(
                AA_DICT[symbol][0]
            )

            offset = rw.GetNumAtoms()

            rw.InsertMol(mol)

            atom_offsets[(chain_idx, monomer_idx)] = offset
            monomer_mols[(chain_idx, monomer_idx)] = mol

    # ------------------------------------------------------------
    # Helper: convert local monomer atom index -> global index.
    # ------------------------------------------------------------
    def global_atom(chain_idx, monomer_idx, local_idx):
        return (
            atom_offsets[(chain_idx, monomer_idx)]
            + local_idx
        )

    # ------------------------------------------------------------
    # Find the heavy-atom attachment point.
    #
    # Each R-group is encoded as a dummy (*) atom bonded to the
    # heavy atom (N for R1, C=O for R2, S for R3, ...).
    # We need the heavy atom index in the combined RWMol.
    # We also track which dummy atom index to remove later.
    # ------------------------------------------------------------
    def attachment_atom(chain_idx, monomer_idx, rgroup):
        """
        Return (heavy_atom_global_idx, dummy_atom_global_idx).

        The heavy atom is where the bond will be formed.
        The dummy atom is the placeholder to remove afterwards.
        """
        mol = monomer_mols[(chain_idx, monomer_idx)]
        offset = atom_offsets[(chain_idx, monomer_idx)]

        dummy_local = _find_rgroup_atom(mol, rgroup)

        if dummy_local is None:
            raise ValueError(
                f"Monomer "
                f"{chains[chain_idx][monomer_idx]['symbol']} "
                f"does not contain R{rgroup}."
            )

        dummy_global = offset + dummy_local
        dummy_atom_in_rw = rw.GetAtomWithIdx(dummy_global)

        # The heavy atom is the single neighbor of the dummy.
        neighbors = dummy_atom_in_rw.GetNeighbors()

        if len(neighbors) != 1:
            raise ValueError(
                f"R{rgroup} dummy atom has unexpected valence "
                f"in monomer "
                f"{chains[chain_idx][monomer_idx]['symbol']}."
            )

        heavy_global = neighbors[0].GetIdx()

        return heavy_global, dummy_global

    # ------------------------------------------------------------
    # Collect all dummy-atom indices that will be consumed by
    # explicit or implicit bonds so they can be removed.
    # ------------------------------------------------------------
    dummies_to_remove = set()

    # Explicit bond IDs.
    explicit_bonds = defaultdict(list)

    for chain_idx, chain in enumerate(chains):
        for monomer_idx, entry in enumerate(chain):
            for bond_id, rgroup in entry["annotations"]:
                explicit_bonds[bond_id].append(
                    (
                        chain_idx,
                        monomer_idx,
                        rgroup,
                    )
                )

    for bond_id, endpoints in explicit_bonds.items():
        if len(endpoints) != 2:
            raise ValueError(
                f"BILN bond {bond_id} must occur exactly twice."
            )

        a_heavy, a_dummy = attachment_atom(*endpoints[0])
        b_heavy, b_dummy = attachment_atom(*endpoints[1])

        if rw.GetBondBetweenAtoms(a_heavy, b_heavy) is None:
            rw.AddBond(
                a_heavy,
                b_heavy,
                Chem.BondType.SINGLE,
            )

        dummies_to_remove.add(a_dummy)
        dummies_to_remove.add(b_dummy)

    # ------------------------------------------------------------
    # Implicit backbone bonds.
    #
    # For every consecutive pair in a chain:
    #
    #     previous R2 -> next R1
    #
    # unless an explicit annotation already defines that
    # connection.
    # ------------------------------------------------------------
    for chain_idx, chain in enumerate(chains):
        for i in range(len(chain) - 1):
            left = (chain_idx, i, 2)
            right = (chain_idx, i + 1, 1)

            left_heavy, left_dummy = attachment_atom(*left)
            right_heavy, right_dummy = attachment_atom(*right)

            if rw.GetBondBetweenAtoms(
                left_heavy,
                right_heavy,
            ) is None:
                rw.AddBond(
                    left_heavy,
                    right_heavy,
                    Chem.BondType.SINGLE,
                )

            dummies_to_remove.add(left_dummy)
            dummies_to_remove.add(right_dummy)

    # ------------------------------------------------------------
    # Remove consumed dummy atoms (those used for backbone or
    # explicit bonds).  These must go first so that the heavy
    # atoms they were attached to have correct valence before
    # _clean_peptide inspects them.
    # ------------------------------------------------------------
    for idx in sorted(dummies_to_remove, reverse=True):
        rw.RemoveAtom(idx)

    # ------------------------------------------------------------
    # Cap remaining terminal dummies.
    #
    # _clean_peptide handles:
    #   - N-terminal: remove dummy bonded to N
    #   - C-terminal: remove dummy bonded to C=O and add -OH
    #   - Thiol:      remove dummy bonded to S
    # ------------------------------------------------------------
    result = _clean_peptide(rw.GetMol())

    return MolToSmiles(
        result,
        canonical=True,
        isomericSmiles=True,
    )


# ============================================================================
# SMILES FILTERING
# ============================================================================

class FilterSmiles(BaseElement):
    name = "filter-smiles"

    def __init__(self, keep_smiles: Optional[bool] = True):
        self.properties["keep_smiles"] = keep_smiles
        self.keep_smiles = keep_smiles

    def _single_call(self, mol: str):
        valid = is_smiles(mol)

        if (valid and self.keep_smiles) or (
            not valid and not self.keep_smiles
        ):
            return mol

        return None

    def __str__(self):
        return (
            self.name
            + f" -> keep_smiles: {self.keep_smiles}"
        )


class CanonicalizeSmiles(BaseElement):
    name = "canonicalize-smiles"

    def _single_call(self, mol):
        rd_mol = rdm.MolFromSmiles(mol)

        if rd_mol is None:
            raise RuntimeError(
                f"Molecule: {mol} could not be read by RDKit.",
                "Maybe introduce a filtering step in your pipeline",
            )

        return rdm.MolToSmiles(
            rd_mol,
            canonical=True,
            isomericSmiles=True,
        )


def is_smiles(mol: str):
    """
    Heuristic SMILES detector: requires structural punctuation that cannot
    appear in a plain amino-acid sequence.  The original implementation
    included the bare letter 'O', which caused false-positives on any
    peptide sequence containing Threonine (T) side-chain shorthand or any
    monomer with an 'O' in its one-letter code.  The corrected version
    only matches characters that are structurally meaningful in SMILES but
    never appear in a canonical or BILN sequence token.
    """
    return any(
        char in mol
        for char in [
            "(",
            ")",
            "[",
            "]",
            "@",
        ]
    )


# ============================================================================
# DUMMY ATOM HANDLING
# ============================================================================

def add_dummy_atoms(mol: Mol) -> Mol:
    """
    Convert terminal H/OH groups into dummy atoms so that
    fragment similarity can be compared with monomers.

    Implementation note
    -------------------
    The function uses a two-phase approach to avoid the RDKit
    pre-condition crash that arises when ``GetTotalValence()`` (or
    any valence query) is called on an atom whose index has been
    shifted by an earlier ``RemoveAtom`` call in the same loop:

    Phase 1 – read-only scan: iterate all atoms and record each
              planned substitution as ``(remove_idx, anchor_idx)``.
    Phase 2 – apply edits: remove atoms in *descending* index order
              (so no removal shifts a later pending removal's index),
              then append one dummy atom per substitution and bond it
              to the anchor.
    """
    mol = Chem.AddHs(mol)
    mol = Chem.RWMol(mol)

    # Each entry: (atom_idx_to_remove, anchor_heavy_atom_idx)
    # 'anchor' is the heavy atom that will receive the dummy bond.
    edits: List[Tuple[int, int]] = []

    # ----------------------------------------------------------------
    # Phase 1: read-only scan
    # ----------------------------------------------------------------
    for atom in mol.GetAtoms():

        # --------------------------------------------------------
        # Nitrogen
        # --------------------------------------------------------
        if atom.GetAtomicNum() == 7:

            if atom.GetIsAromatic():
                continue

            neighbors = list(atom.GetNeighbors())

            h_atoms = [
                n for n in neighbors
                if n.GetAtomicNum() == 1
            ]

            dummy_atoms = [
                n for n in neighbors
                if n.GetAtomicNum() == 0
            ]

            n_subn = 0
            for n in neighbors:
                if n.GetAtomicNum() == 1:
                    continue
                for n2 in n.GetNeighbors():
                    if n2.GetIdx() == atom.GetIdx():
                        continue
                    n_subn += int(n2.GetAtomicNum() == 7)

            is_guanidinium_n = any(
                n.GetAtomicNum() == 6
                and any(
                    b.GetBondType() == Chem.BondType.DOUBLE
                    and b.GetOtherAtom(n).GetAtomicNum() == 7
                    for b in n.GetBonds()
                )
                for n in neighbors
                if n.GetAtomicNum() not in (0, 1)
            )

            is_urea_n = any(
                n.GetAtomicNum() == 6
                and any(
                    b.GetBondType() == Chem.BondType.DOUBLE
                    and b.GetOtherAtom(n).GetAtomicNum() == 8
                    for b in n.GetBonds()
                )
                and sum(
                    1 for nb in n.GetNeighbors()
                    if nb.GetAtomicNum() == 7
                ) >= 2
                for n in neighbors
                if n.GetAtomicNum() not in (0, 1)
            )

            if (
                len(h_atoms) >= 1
                and len(dummy_atoms) == 0
                and n_subn != 3
                and not is_guanidinium_n
                and not is_urea_n
            ):
                edits.append((h_atoms[0].GetIdx(), atom.GetIdx()))

        # --------------------------------------------------------
        # Carboxylic acid  –OH on a C(=O)(O) carbon
        # --------------------------------------------------------
        elif atom.GetAtomicNum() == 8:

            neighbors = list(atom.GetNeighbors())
            if not neighbors:
                continue

            parent = neighbors[0]
            if parent.GetAtomicNum() != 6:
                continue

            # Count bonds to determine carboxyl carbon: must have
            # exactly two O neighbours and no double bond to the
            # current O (which is the -OH, not the =O).
            o_neighbors = [
                n for n in parent.GetNeighbors()
                if n.GetAtomicNum() == 8
            ]
            parent_is_carboxy = len(o_neighbors) == 2

            # The current O is the -OH (has exactly one H neighbour).
            atom_is_hydroxy = sum(
                1 for n in atom.GetNeighbors()
                if n.GetAtomicNum() == 1
            ) == 1

            if parent_is_carboxy and atom_is_hydroxy:
                edits.append((atom.GetIdx(), parent.GetIdx()))

        # --------------------------------------------------------
        # Thiol
        # --------------------------------------------------------
        elif atom.GetAtomicNum() == 16:

            h_atoms = [
                n for n in atom.GetNeighbors()
                if n.GetAtomicNum() == 1
            ]

            if len(h_atoms) == 1:
                edits.append((h_atoms[0].GetIdx(), atom.GetIdx()))

    # ----------------------------------------------------------------
    # Phase 2: apply edits
    # Remove atoms highest-index first so earlier removals do not
    # shift the indices of later ones.  Dummy atoms are always
    # appended at the end, so their indices are stable.
    # ----------------------------------------------------------------
    for remove_idx, anchor_idx in sorted(edits, key=lambda e: e[0], reverse=True):
        mol.RemoveAtom(remove_idx)
        # After removal, atoms with original idx > remove_idx have
        # shifted down by 1.  Adjust anchor_idx if necessary.
        if anchor_idx > remove_idx:
            anchor_idx -= 1
        dummy_idx = mol.AddAtom(Chem.Atom(0))
        mol.AddBond(anchor_idx, dummy_idx, Chem.BondType.SINGLE)

    mol = Chem.RemoveAllHs(mol)

    return mol


# ============================================================================
# MONOMER IDENTIFICATION
# ============================================================================

def find_lone_nitrogen_fragments(
    fragments: List[Mol],
):
    lone_n_list = []

    for idx, frag in enumerate(fragments):

        atoms = frag.GetAtoms()

        real_atoms = [
            a for a in atoms
            if a.GetAtomicNum() != 0
        ]

        n_atoms = [
            a for a in real_atoms
            if a.GetAtomicNum() == 7
        ]

        if (
            len(real_atoms) == 1
            and len(n_atoms) == 1
        ):
            lone_n_list.append(idx)

    return lone_n_list


def reattach_n(fragments: List[Mol]):
    """
    Reattach fragments that consist of a lone nitrogen.

    Fixed bug from the original implementation:
        avoid.append(lone_n)
    was replaced with:
        avoid.add(n)
    """
    blocker = rdBase.BlockLogs()

    new_fragments = []
    avoid = set()

    lone_n = find_lone_nitrogen_fragments(
        fragments
    )

    for n in lone_n:
        n_frag = fragments[n]

        for idx, frag in enumerate(fragments):

            if idx == n:
                continue

            closest = find_closest_monomer(frag)[0]

            if closest in ("N", "Q", "E", "D"):

                try:
                    new_frag = rdops.molzip(
                        n_frag,
                        frag,
                    )
                except Exception:
                    continue

                if (
                    len(
                        MolToSmiles(
                            new_frag
                        ).split(".")
                    ) == 1
                ):
                    new_fragments.append(
                        new_frag
                    )

                    avoid.add(idx)
                    avoid.add(n)

                    break

    for idx, frag in enumerate(fragments):
        if idx not in avoid:
            new_fragments.append(frag)

    return new_fragments


# ============================================================================
# FRAGMENT TOPOLOGY
# ============================================================================

def get_fragment_bond_info(
    mol: Chem.Mol,
):
    """
    Return all dummy atoms carrying a fragment bond ID.

    FragmentOnBonds() gives both dummy atoms the same isotope.
    We copy that into molAtomMapNumber.
    """
    info = defaultdict(list)

    for atom in mol.GetAtoms():

        if atom.GetAtomicNum() != 0:
            continue

        bond_id = atom.GetAtomMapNum()

        if bond_id == 0:
            bond_id = atom.GetIsotope()

        if bond_id == 0:
            continue

        neighbors = atom.GetNeighbors()

        if len(neighbors) != 1:
            continue

        neighbor = neighbors[0]

        rgroup = None

        # If the attachment atom itself retained its R-group.
        label = _get_atom_label(neighbor)

        rgroup = _rgroup_from_label(label)

        # --------------------------------------------------------
        # Infer common peptide R-groups when the CXSMILES atom
        # label did not survive fragmentation.
        # --------------------------------------------------------
        if rgroup is None:

            atomic_num = neighbor.GetAtomicNum()

            # R1: backbone N
            if atomic_num == 7:
                rgroup = 1

            # R2: carbonyl C
            elif atomic_num == 6:

                has_carbonyl = any(
                    bond.GetBondType()
                    == Chem.BondType.DOUBLE
                    and bond.GetOtherAtom(
                        neighbor
                    ).GetAtomicNum() == 8
                    for bond in neighbor.GetBonds()
                )

                if has_carbonyl:
                    rgroup = 2

            # R3: sulfur in cysteine
            elif atomic_num == 16:
                rgroup = 3

        info[bond_id].append({
            "dummy_idx": atom.GetIdx(),
            "neighbor_idx": neighbor.GetIdx(),
            "rgroup": rgroup,
        })

    return info


# ============================================================================
# SMILES -> MONOMER FRAGMENTS
# ============================================================================


def _break_single_mol(
    mol: Chem.Mol,
    bond_id_offset: int = 0,
) -> Tuple[List[str], List[Chem.Mol]]:
    """
    Break a single connected molecule into monomer fragments.

    *bond_id_offset* allows the caller to make bond IDs globally
    unique when processing multiple chains in one session.

    Returns (monomer_names, fragment_mols).
    """
    # ------------------------------------------------------------------
    # Peptide backbone: N-C(=O)-C
    # ------------------------------------------------------------------
    pep_bond = Chem.MolFromSmarts("N[C](=O)C")

    # ------------------------------------------------------------------
    # C-terminal amide cap: [C@@H](N)C(=O)NH2
    # Matches C(=O)-NH2 where the carbonyl C's sp3-C neighbour itself
    # has an N neighbour (i.e., the sp3-C is a Cα bonded to a free amine).
    # This pattern is distinct from sidechain amides (Asn/Gln) where the
    # sp3-C neighbour is a CH2 with no N.
    # ------------------------------------------------------------------
    cterminal_amide_bond = Chem.MolFromSmarts("[CX3](=[OX1])[NH2]")

    # ------------------------------------------------------------------
    # Disulfide: C-S-S-C
    # ------------------------------------------------------------------
    disulfide_bond = Chem.MolFromSmarts("CSSC")

    pep_matches = mol.GetSubstructMatches(pep_bond)
    ctam_matches = mol.GetSubstructMatches(cterminal_amide_bond)
    ss_matches = mol.GetSubstructMatches(disulfide_bond)

    bond_indices_pep = []
    for n_idx, c_idx, *_ in pep_matches:
        # Only cut genuine backbone (or peptoid) peptide bonds.
        #
        # The SMARTS "N[C](=O)C" also matches sidechain terminal amide
        # nitrogens (Asn, Gln NH2) and C-terminal amide caps.  Those N
        # atoms are bonded solely to an sp2 carbonyl C — they have no sp3
        # carbon neighbour.  Backbone and peptoid N atoms always have at
        # least one sp3 Cα neighbour in the chain context.
        n_atom = mol.GetAtomWithIdx(n_idx)
        has_sp3_c_neighbour = any(
            nb.GetAtomicNum() == 6
            and nb.GetHybridization() == Chem.HybridizationType.SP3
            for nb in n_atom.GetNeighbors()
        )
        if not has_sp3_c_neighbour:
            continue
        bond = mol.GetBondBetweenAtoms(n_idx, c_idx)
        if bond:
            bond_indices_pep.append(bond.GetIdx())

    # C-terminal amide bonds: cut C(=O)-NH2 only when the C's sp3-C
    # neighbour is itself bonded to an N (a Cα carrying a free amine).
    # This excludes Asn/Gln sidechain amides (their sp3 neighbour CH2
    # has no N).
    for match in ctam_matches:
        c_idx = match[0]
        c_atom = mol.GetAtomWithIdx(c_idx)
        sp3c_with_n = [
            nb for nb in c_atom.GetNeighbors()
            if nb.GetAtomicNum() == 6
            and nb.GetHybridization() == Chem.HybridizationType.SP3
            and any(
                nb2.GetAtomicNum() == 7
                for nb2 in nb.GetNeighbors()
                if nb2.GetIdx() != c_idx
            )
        ]
        if not sp3c_with_n:
            continue
        n_idx = match[2]  # match order: (C, O, N)
        bond = mol.GetBondBetweenAtoms(c_idx, n_idx)
        if bond:
            bond_indices_pep.append(bond.GetIdx())

    bond_indices_ss = []
    for _, s1_idx, s2_idx, _ in ss_matches:
        bond = mol.GetBondBetweenAtoms(s1_idx, s2_idx)
        if bond:
            bond_indices_ss.append(bond.GetIdx())

    bond_indices = list(
        dict.fromkeys(bond_indices_pep + bond_indices_ss)
    )

    # ------------------------------------------------------------------
    # Single monomer — no bonds to break.
    # ------------------------------------------------------------------
    if not bond_indices:
        best_aa, _ = find_closest_monomer(mol)
        return [best_aa], [mol]

    # ------------------------------------------------------------------
    # Fragment the molecule.
    # Bond labels are offset so IDs are globally unique across chains.
    # ------------------------------------------------------------------
    n = len(bond_indices)
    frags = rdops.FragmentOnBonds(
        mol,
        bond_indices,
        addDummies=True,
        dummyLabels=[
            (bond_id_offset + i + 1, bond_id_offset + i + 1)
            for i in range(n)
        ],
    )

    # Convert isotope -> molAtomMapNumber for downstream use.
    for i in range(n):
        bond_num = bond_id_offset + i + 1
        for atom in frags.GetAtoms():
            if atom.GetAtomicNum() == 0 and atom.GetIsotope() == bond_num:
                atom.SetAtomMapNum(bond_num)

    updated_frag_mols = Chem.GetMolFrags(
        frags, asMols=True, sanitizeFrags=True
    )

    updated_frag_mols = reattach_n(list(updated_frag_mols))

    ordered_frags = order_fragments(updated_frag_mols)

    final_pep = []
    all_frags = []
    for frag in ordered_frags:
        best_aa, _ = find_closest_monomer(frag)
        final_pep.append(best_aa)
        all_frags.append(frag)

    return final_pep, all_frags


def break_into_monomers(
    smiles,
) -> Tuple[List[str], List[Chem.Mol], List[int]]:
    """
    Break a peptide SMILES into monomer fragments.

    Multi-chain SMILES (dot-separated, e.g. ``A-G.G-A``) are handled
    by splitting at the molecular level first: each disconnected
    component is processed independently and the results are
    concatenated in the same order as the components appear in the
    molecule.

    Crucially, the returned fragments retain:
        - AtomMapNum on dummy atoms
        - isotope bond IDs (globally unique across all chains)
        - R-group labels where available
    """
    if smiles is None:
        raise TypeError("SMILES must be a string, got None.")

    if isinstance(smiles, str):
        if not smiles.strip():
            raise ValueError("SMILES string is empty.")

        mol = MolFromSmiles(smiles, sanitize=True)

        if mol is None:
            raise ValueError(
                f"Could not parse SMILES: {smiles!r}"
            )
    else:
        # Accept a pre-parsed Mol object for internal use.
        mol = smiles

    # ------------------------------------------------------------------
    # Split disconnected components (multi-chain SMILES, e.g. "A.B").
    # ------------------------------------------------------------------
    component_mols = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)

    if len(component_mols) == 1:
        names, frags = _break_single_mol(component_mols[0])
        return names, frags, [len(names)]

    # Multiple independent chains: process each separately,
    # using a running offset so bond IDs are globally unique.
    all_names: List[str] = []
    all_frags: List[Chem.Mol] = []
    chain_sizes: List[int] = []
    bond_id_offset = 0

    for comp in component_mols:
        names, frags = _break_single_mol(comp, bond_id_offset=bond_id_offset)
        # Advance offset by the number of unique bond IDs in this chain.
        n_bonds = len({
            atom.GetAtomMapNum()
            for frag in frags
            for atom in frag.GetAtoms()
            if atom.GetAtomicNum() == 0 and atom.GetAtomMapNum() != 0
        })
        bond_id_offset += n_bonds
        all_names.extend(names)
        all_frags.extend(frags)
        chain_sizes.append(len(names))

    return all_names, all_frags, chain_sizes


# ============================================================================
# FRAGMENT ORDERING
# ============================================================================

def _is_free_n_terminal(mol: Chem.Mol) -> bool:
    """
    Identify a fragment with a free N-terminal amine.
    """
    for atom in mol.GetAtoms():

        if atom.GetAtomicNum() != 7:
            continue

        # Ignore nitrogen carrying a fragment dummy.
        if any(
            n.GetAtomicNum() == 0
            for n in atom.GetNeighbors()
        ):
            continue

        for carbon in atom.GetNeighbors():

            if carbon.GetAtomicNum() != 6:
                continue

            has_carbonyl = any(
                bond.GetBondType()
                == Chem.BondType.DOUBLE
                and bond.GetOtherAtom(
                    carbon
                ).GetAtomicNum() == 8
                for bond in carbon.GetBonds()
            )

            if has_carbonyl:
                return True

    return False


def order_fragments(
    mols: List[Chem.Mol],
) -> List[Chem.Mol]:
    """
    Order fragments N -> C.

    For cyclic peptides there is no unique beginning, so an arbitrary
    fragment is selected. The cycle-closing bond is NOT discarded;
    it remains encoded on the fragment dummy atoms and is recovered
    later by fragments_to_biln().
    """
    if len(mols) <= 1:
        return mols

    fragments = []

    for idx, mol in enumerate(mols):

        bond_ids = set()

        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 0:

                bond_id = atom.GetAtomMapNum()

                if bond_id:
                    bond_ids.add(bond_id)

        fragments.append({
            "id": idx,
            "mol": mol,
            "bond_ids": bond_ids,
        })

    dummy_to_frag = defaultdict(list)

    for frag in fragments:
        for bond_id in frag["bond_ids"]:
            dummy_to_frag[
                bond_id
            ].append(frag["id"])

    graph = defaultdict(set)

    for ids in dummy_to_frag.values():

        if len(ids) == 2:
            a, b = ids

            graph[a].add(b)
            graph[b].add(a)

    # ------------------------------------------------------------
    # Find N-terminal fragment.
    #
    # The N-terminal residue has its backbone N free (no R1 dummy)
    # while carrying an R2 dummy for the next peptide bond.  Detect
    # this by checking that the fragment has R2 attachment but NOT R1.
    # ------------------------------------------------------------
    start = None

    for idx, mol in enumerate(mols):
        info = get_fragment_bond_info(mol)
        rgroups_in_frag = set()
        for occs in info.values():
            for occ in occs:
                rg = occ["rgroup"]
                if rg is not None:
                    rgroups_in_frag.add(rg)
        # N-terminal fragment: has R2 (C-end bond) but no R1 (free N-end)
        if 2 in rgroups_in_frag and 1 not in rgroups_in_frag:
            start = idx
            break

    # ------------------------------------------------------------
    # Cyclic peptide: no free N terminus.
    # Pick the fragment with the fewest bond_ids (most likely an
    # internal residue with only backbone connections), using index
    # as a tiebreaker for determinism.
    # ------------------------------------------------------------
    if start is None:
        start = min(
            range(len(mols)),
            key=lambda i: (
                len(fragments[i]["bond_ids"]),
                i,
            ),
        )

    # ------------------------------------------------------------
    # Traverse N -> C.
    #
    # At each step, strongly prefer the candidate connected to the
    # current fragment via a backbone bond (current R2 → candidate
    # R1).  This ensures linear ordering for cyclic peptides where
    # backbone bonds and side-chain (SS) bonds both appear in the
    # graph.
    # ------------------------------------------------------------
    ordered_ids = []
    visited = set()

    current = start
    previous = None

    while current is not None:

        if current in visited:
            break

        visited.add(current)
        ordered_ids.append(current)

        candidates = [
            n
            for n in graph[current]
            if n != previous
            and n not in visited
        ]

        if not candidates:
            break

        # Find which bond IDs connect current → each candidate,
        # then score by whether that bond is a backbone R2→R1 pair.
        cur_info = get_fragment_bond_info(mols[current])
        # Map bond_id -> rgroup for current fragment.
        cur_bond_rgroup = {}
        for bond_id, occs in cur_info.items():
            for occ in occs:
                cur_bond_rgroup[bond_id] = occ["rgroup"]

        def score(node):  # noqa: E306
            node_info = get_fragment_bond_info(mols[node])
            node_bond_rgroup = {}
            for bond_id, occs in node_info.items():
                for occ in occs:
                    node_bond_rgroup[bond_id] = occ["rgroup"]

            # Shared bond IDs between current and this candidate.
            shared = set(cur_bond_rgroup) & set(node_bond_rgroup)

            is_backbone = any(
                cur_bond_rgroup[b] == 2 and node_bond_rgroup[b] == 1
                for b in shared
            )

            has_r1 = any(x["rgroup"] == 1 for v in node_info.values() for x in v)
            has_r2 = any(x["rgroup"] == 2 for v in node_info.values() for x in v)

            return (
                0 if is_backbone else 1,  # backbone bond strongly preferred
                0 if has_r1 else 1,
                0 if has_r2 else 1,
                node,
            )

        nxt = min(candidates, key=score)
        previous, current = current, nxt

    # ------------------------------------------------------------
    # Add disconnected components.
    # ------------------------------------------------------------
    for idx in range(len(mols)):
        if idx not in visited:
            ordered_ids.append(idx)

    return [
        mols[idx]
        for idx in ordered_ids
    ]


# ============================================================================
# FRAGMENT -> BILN
# ============================================================================

def _infer_fragment_rgroup(
    fragment: Chem.Mol,
    occurrence: dict,
) -> Optional[int]:
    """
    Infer the BILN R-group corresponding to a fragment dummy.

    Priority:
        1. explicit atomLabel
        2. common amino-acid conventions
    """
    neighbor_idx = occurrence[
        "neighbor_idx"
    ]

    neighbor = fragment.GetAtomWithIdx(
        neighbor_idx
    )

    label = _get_atom_label(
        neighbor
    )

    rgroup = _rgroup_from_label(
        label
    )

    if rgroup is not None:
        return rgroup

    atomic_num = neighbor.GetAtomicNum()

    # R1 = peptide N
    if atomic_num == 7:
        return 1

    # R2 = peptide carbonyl carbon
    if atomic_num == 6:

        has_carbonyl = any(
            bond.GetBondType()
            == Chem.BondType.DOUBLE
            and bond.GetOtherAtom(
                neighbor
            ).GetAtomicNum() == 8
            for bond in neighbor.GetBonds()
        )

        if has_carbonyl:
            return 2

    # R3 = thiol / disulfide
    if atomic_num == 16:
        return 3

    return None


def fragments_to_biln(
    fragments: List[Chem.Mol],
    monomer_names: List[str],
    chain_sizes: Optional[List[int]] = None,
) -> str:
    """
    Convert topology-preserving fragments to BILN.

    Explicit non-backbone bonds become:

        M(BOND_ID,R_GROUP)

    Ordinary consecutive R2 -> R1 connections remain '-'.

    *chain_sizes* is an optional list whose entries give the number of
    fragments belonging to each chain (sum must equal len(fragments)).
    When provided, chains are separated by '.' in the output; when
    omitted a single chain with '-' separators is assumed.
    """
    if len(fragments) != len(monomer_names):
        raise ValueError(
            "Number of fragments and monomer names differ: "
            f"{len(fragments)} != {len(monomer_names)}"
        )

    if chain_sizes is None:
        chain_sizes = [len(fragments)]

    tokens = list(monomer_names)

    # ------------------------------------------------------------
    # Collect every dummy/bond ID.
    # ------------------------------------------------------------
    bond_occurrences = defaultdict(list)

    for frag_idx, fragment in enumerate(
        fragments
    ):

        info = get_fragment_bond_info(
            fragment
        )

        for bond_id, occurrences in info.items():

            for occurrence in occurrences:

                rgroup = _infer_fragment_rgroup(
                    fragment,
                    occurrence,
                )

                bond_occurrences[
                    bond_id
                ].append({
                    "fragment": frag_idx,
                    "rgroup": rgroup,
                })

    # ------------------------------------------------------------
    # Validate bond IDs.
    # ------------------------------------------------------------
    for bond_id, occurrences in (
        bond_occurrences.items()
    ):

        if len(occurrences) != 2:
            raise ValueError(
                f"Fragment bond ID {bond_id} occurs "
                f"{len(occurrences)} times; "
                "expected exactly twice."
            )

    # ------------------------------------------------------------
    # Identify ordinary backbone bonds.
    #
    # A BILN bond is implicit (backbone '-') only when:
    #     • it connects exactly two consecutive fragments i, i+1
    #     • fragment i carries R2 on that bond
    #     • fragment i+1 carries R1 on that bond
    #
    # All other bonds become explicit annotations.
    # ------------------------------------------------------------
    implicit_backbone_ids = set()
    # backbone_edges[i] = j means fragment i is backbone-connected
    # to fragment j via an implicit bond.
    backbone_edges: dict = {}

    for bond_id, occurrences in bond_occurrences.items():

        if len(occurrences) != 2:
            continue

        a, b = occurrences

        if abs(a["fragment"] - b["fragment"]) != 1:
            continue

        left = min(a["fragment"], b["fragment"])
        right = max(a["fragment"], b["fragment"])

        left_occ = a if a["fragment"] == left else b
        right_occ = a if a["fragment"] == right else b

        if left_occ["rgroup"] == 2 and right_occ["rgroup"] == 1:
            implicit_backbone_ids.add(bond_id)
            backbone_edges[left] = right
            backbone_edges[right] = left

    # ------------------------------------------------------------
    # Add explicit annotations (non-backbone bonds).
    # ------------------------------------------------------------
    for bond_id, occurrences in sorted(
        bond_occurrences.items(), key=lambda x: x[0]
    ):
        if bond_id in implicit_backbone_ids:
            continue

        for occurrence in occurrences:
            fragment_idx = occurrence["fragment"]
            rgroup = occurrence["rgroup"]

            if rgroup is None:
                raise ValueError(
                    "Unable to determine R-group "
                    f"for bond {bond_id} on "
                    f"monomer {tokens[fragment_idx]}."
                )

            tokens[fragment_idx] += f"({bond_id},{rgroup})"

    # ------------------------------------------------------------
    # Determine chain boundaries from backbone connectivity.
    #
    # Fragments that are only connected to the rest of the molecule
    # via non-backbone (explicit) bonds belong to separate chains.
    # Walk the backbone graph to collect runs of backbone-connected
    # fragments; each run becomes one BILN chain.
    # ------------------------------------------------------------
    n_frags = len(tokens)
    visited_frags: set = set()
    chains_fragment_ids: List[List[int]] = []

    for start_frag in range(n_frags):
        if start_frag in visited_frags:
            continue

        # Walk backbone edges starting from start_frag.
        # Walk backward first (to find true N-terminal of this chain).
        head = start_frag
        while head in backbone_edges:
            prev = backbone_edges[head]
            if prev in visited_frags or prev > head:
                # avoid infinite loops in cyclic chains
                break
            head = prev

        # Now walk forward from head.
        chain_ids: List[int] = []
        cur = head
        while cur is not None and cur not in visited_frags:
            chain_ids.append(cur)
            visited_frags.add(cur)
            nxt = backbone_edges.get(cur)
            if nxt is None or nxt in visited_frags:
                break
            cur = nxt

        if chain_ids:
            chains_fragment_ids.append(chain_ids)

    # ------------------------------------------------------------
    # Build final BILN string.
    # ------------------------------------------------------------
    chain_bilns = [
        "-".join(tokens[idx] for idx in chain_ids)
        for chain_ids in chains_fragment_ids
    ]

    return ".".join(chain_bilns)


# ============================================================================
# MONOMER SIMILARITY
# ============================================================================

# Precompiled patterns for _normalize_for_fp.
_CARBOXYL_PATT = Chem.MolFromSmarts("C(=O)[OH]")
_CARBONYL_DUMMY_PATT = Chem.MolFromSmarts("[CX3](=[OX1])[#0]")
_CTERMINAL_AMIDE_RXN = None  # lazy-initialised below


def _normalize_for_fp(mol: Chem.Mol) -> Chem.Mol:
    """
    Normalise a fragment for Tanimoto fingerprinting only.

    C-terminal amide capping (``C(=O)NH2``) is converted to the
    equivalent free carboxylic acid (``C(=O)OH``) so that
    ``add_dummy_atoms`` places the R2 dummy on the carbonyl C rather
    than on the amide N.

    The conversion is skipped if the molecule already contains a free
    carboxylic acid — that covers Asn/Gln (sidechain amide + backbone
    COOH) which must not be altered.
    """
    global _CTERMINAL_AMIDE_RXN
    if _CTERMINAL_AMIDE_RXN is None:
        from rdkit.Chem import rdChemReactions as rdr
        _CTERMINAL_AMIDE_RXN = rdr.ReactionFromSmarts(
            "[CX3:1](=[OX1:2])[NH2:3]>>[CX3:1](=[OX1:2])O"
        )

    # Skip if the molecule already has a free carboxylic acid OR a
    # backbone dummy on a carbonyl (C(=O)[*]).  The latter covers
    # in-chain fragments for Asn/Gln where the C-terminus is already
    # an R-group placeholder, not an amide cap.
    if (
        mol.HasSubstructMatch(_CARBOXYL_PATT)
        or mol.HasSubstructMatch(_CARBONYL_DUMMY_PATT)
    ):
        return mol

    products = _CTERMINAL_AMIDE_RXN.RunReactants((mol,))
    if products:
        prod = products[0][0]
        try:
            Chem.SanitizeMol(prod)
            return prod
        except Exception:
            pass  # fall back to original if sanitisation fails
    return mol


def find_closest_monomer(
    frag: Chem.Mol,
) -> Tuple[str, float]:
    global CACHE

    blocker = rdBase.BlockLogs()  # noqa: F841 – suppresses RDKit warnings

    # Threshold: 0.7 Tanimoto similarity.  Using 0.699... instead of
    # 0.7 so that a similarity of exactly 0.7 is accepted (condition
    # is strictly greater-than, so 0.7 > 0.699 passes).
    max_sim = 0.699
    best_aa = "X"

    mol1 = add_dummy_atoms(_normalize_for_fp(frag))

    fp1 = GetMorganFingerprintAsBitVect(
        mol1,
        radius=2,
        useFeatures=True,
        nBits=1024,
        useChirality=True,
    )

    # ------------------------------------------------------------
    # Canonical amino acids — chiral pass (high priority).
    # ------------------------------------------------------------
    for aa in AAs:

        monomer = AA_DICT[aa]

        smiles_similarity, _ = compare(monomer, aa, fp1)

        if smiles_similarity > max_sim:
            max_sim = smiles_similarity
            best_aa = "H" if aa == "H2" else aa

        if max_sim == 1.0:
            return best_aa, max_sim

    # ------------------------------------------------------------
    # Canonical amino acids — achiral confirmation pass.
    #
    # Fragments produced by bond-breaking frequently lose stereo on
    # side-chain carbons (e.g. Thr C-beta loses @@).  When the chiral
    # pass found a plausible but sub-1.0 match, a targeted achiral check
    # on that same AA can confirm the identification without introducing
    # L/D ambiguity.
    #
    # Safety guard: before accepting the achiral confirmation, check
    # whether any non-canonical D-form of best_aa scores >= max_sim
    # with chirality.  If one does (e.g. dA scores 1.0 for D-Ala input),
    # the fragment is that D-form — skip the achiral confirmation and let
    # the non-canonical loop below identify it correctly.
    # ------------------------------------------------------------
    fp1_achiral = GetMorganFingerprintAsBitVect(
        mol1,
        radius=2,
        useFeatures=True,
        nBits=1024,
        useChirality=False,
    )

    if best_aa != "X":
        # Check for a D-form (non-canonical, same natural analog) that
        # scores at least as well as best_aa in the chiral fingerprint.
        d_form_found = False
        for nc_aa, (nc_smiles, nc_analog) in AA_DICT.items():
            if nc_aa in AAs or nc_analog != best_aa:
                continue
            nc_sim, _ = compare(AA_DICT[nc_aa], nc_aa, fp1)
            if nc_sim >= max_sim:
                d_form_found = True
                break

        if not d_form_found:
            lib_key = "H2" if best_aa == "H" else best_aa
            monomer = AA_DICT.get(lib_key)
            if monomer is not None:
                sim, _ = compare_achiral(monomer, lib_key, fp1_achiral)
                if sim == 1.0:
                    # Achiral perfect match, no competing D-form: beta-carbon
                    # stereo was absent from the fragment; accept as L-form.
                    return best_aa, 1.0

    # ------------------------------------------------------------
    # Non-canonical monomers (chiral).
    # ------------------------------------------------------------
    for aa, monomer in AA_DICT.items():

        if aa in AAs:
            continue

        smiles_similarity, smiles2 = compare(monomer, aa, fp1)

        if smiles_similarity > max_sim:
            max_sim = smiles_similarity
            # Apply canonical name mapping (e.g. lalloI → I).
            best_aa = _CANONICAL_NAME_MAP.get(aa, aa)

        # When THIS entry scores 1.0, verify element composition.
        # Feature-based Morgan fingerprints treat halogens as equivalent,
        # so a Br-substituted fragment and a Cl-substituted library entry
        # can both score 1.0.  The atom-subset check discriminates them:
        # only accept and break if the fragment's elements are a subset of
        # the library entry's elements.  If the check fails, clear best_aa
        # so the search continues to the correct entry.
        if smiles_similarity == 1.0:
            mol2 = MolFromSmiles(smiles2)

            if mol2 is not None:
                atoms1 = {a.GetAtomicNum() for a in mol1.GetAtoms()}
                atoms2 = {a.GetAtomicNum() for a in mol2.GetAtoms()}

                if atoms1.issubset(atoms2):
                    best_aa = _CANONICAL_NAME_MAP.get(aa, aa)
                    break
                else:
                    # Wrong elements — undo the tentative assignment and
                    # keep searching for an element-correct match.
                    best_aa = "X"
                    max_sim = 0.699

    # ------------------------------------------------------------
    # Achiral fallback: if still no match above threshold AND the
    # fragment has no specified stereocentres at all, retry the full
    # non-canonical library without chirality.  This recovers exotic
    # monomers whose library entries carry explicit stereo but the
    # input SMILES omits it entirely.
    # ------------------------------------------------------------
    if best_aa == "X":
        chiral_centres = Chem.FindMolChiralCenters(
            frag, includeUnassigned=False
        )
        if not chiral_centres:
            for aa, monomer in AA_DICT.items():
                if aa in AAs:
                    continue
                sim, _ = compare_achiral(monomer, aa, fp1_achiral)
                if sim > max_sim:
                    max_sim = sim
                    best_aa = _CANONICAL_NAME_MAP.get(aa, aa)
                if max_sim == 1.0:
                    break

    return best_aa, max_sim


def compare(
    monomer,
    aa,
    fp1,
):
    global CACHE

    smiles2, _ = monomer

    smiles2 = _strip_smiles_properties(
        smiles2
    )

    if smiles2 in CACHE:
        fp2 = CACHE[smiles2]

    else:

        mol2 = MolFromSmiles(
            smiles2,
            sanitize=True,
        )

        if mol2 is None:
            return 0.0, smiles2

        fp2 = GetMorganFingerprintAsBitVect(
            mol2,
            radius=2,
            useFeatures=True,
            nBits=1024,
            useChirality=True,
        )

        CACHE[smiles2] = fp2

    smiles_similarity = (
        DataStructs.TanimotoSimilarity(
            fp1,
            fp2,
        )
    )

    return smiles_similarity, smiles2


def compare_achiral(
    monomer,
    aa,
    fp1,
):
    """Like compare(), but builds/caches fingerprints without chirality."""
    global CACHE_ACHIRAL

    smiles2, _ = monomer

    smiles2 = _strip_smiles_properties(smiles2)

    if smiles2 in CACHE_ACHIRAL:
        fp2 = CACHE_ACHIRAL[smiles2]

    else:
        mol2 = MolFromSmiles(smiles2, sanitize=True)

        if mol2 is None:
            return 0.0, smiles2

        fp2 = GetMorganFingerprintAsBitVect(
            mol2,
            radius=2,
            useFeatures=True,
            nBits=1024,
            useChirality=False,
        )

        CACHE_ACHIRAL[smiles2] = fp2

    smiles_similarity = DataStructs.TanimotoSimilarity(fp1, fp2)

    return smiles_similarity, smiles2


# ============================================================================
# BUILD NORMAL PEPTIDES
# ============================================================================

def build_peptide(
    monomerlist: List[Tuple[str, str]],
) -> Tuple[str, List[str]]:
    """
    Assemble a linear peptide from monomer SMILES.
    """
    monomerlist = copy.deepcopy(
        list(monomerlist)
    )

    monomers = []
    res = None

    for idx, monomer in enumerate(
        monomerlist
    ):

        monomers.append(
            monomer[0]
        )

        mol = _mol_from_library_smiles(
            monomer[1]
        )

        if idx == 0:
            res = mol
        else:
            res = _combine_fragments(
                res,
                mol,
            )

    if res is None:
        return "", []

    cleaned = _clean_peptide(
        res
    )

    return (
        rdm.MolToSmiles(
            cleaned,
            canonical=True,
            isomericSmiles=True,
        ),
        monomers,
    )


def _combine_fragments(
    m1: Mol,
    m2: Mol,
) -> Mol:
    """
    Connect R2 of m1 to R1 of m2.
    """
    blocker = rdBase.BlockLogs()

    m1_success = False
    m2_success = False

    for atom in m1.GetAtoms():

        if (
            atom.HasProp("atomLabel")
            and atom.GetProp("atomLabel")
            == "_R2"
        ):
            atom.SetAtomMapNum(1)
            m1_success = True

    for atom in m2.GetAtoms():

        if (
            atom.HasProp("atomLabel")
            and atom.GetProp("atomLabel")
            == "_R1"
        ):
            atom.SetAtomMapNum(1)
            m2_success = True

    if not m1_success:
        raise RuntimeError(
            "Molecule 1 does not have "
            "an R2 attachment point."
        )

    if not m2_success:
        raise RuntimeError(
            "Molecule 2 does not have "
            "an R1 attachment point."
        )

    return rdops.molzip(
        m1,
        m2,
    )


# ============================================================================
# CLEAN PEPTIDE
# ============================================================================

def _clean_peptide(
    mol: Mol,
) -> Mol:
    """
    Remove remaining terminal dummy atoms and cap the C terminus.
    """
    rw_mol = RWMol(mol)

    atoms_to_remove = []
    attach_oh_to = []

    for atom in mol.GetAtoms():

        if atom.GetSymbol() != "*":
            continue

        neighbors = atom.GetNeighbors()

        if len(neighbors) != 1:
            continue

        neighbor = neighbors[0]

        # N[*]
        if neighbor.GetSymbol() == "N":
            atoms_to_remove.append(
                atom.GetIdx()
            )

        # C([*])=O
        elif neighbor.GetSymbol() == "C":

            carbon = neighbor

            is_carbonyl = any(
                n.GetSymbol() == "O"
                and mol.GetBondBetweenAtoms(
                    carbon.GetIdx(),
                    n.GetIdx(),
                ).GetBondType()
                == Chem.BondType.DOUBLE
                for n in carbon.GetNeighbors()
            )

            if is_carbonyl:

                atoms_to_remove.append(
                    atom.GetIdx()
                )

                attach_oh_to.append(
                    carbon.GetIdx()
                )

        # S[*]
        elif neighbor.GetSymbol() == "S":
            atoms_to_remove.append(
                atom.GetIdx()
            )

    # Add terminal OH groups.
    for carbon_idx in attach_oh_to:

        o_idx = rw_mol.AddAtom(
            Chem.Atom("O")
        )

        rw_mol.AddBond(
            carbon_idx,
            o_idx,
            Chem.BondType.SINGLE,
        )

    for idx in sorted(
        atoms_to_remove,
        reverse=True,
    ):
        rw_mol.RemoveAtom(idx)

    final_mol = rw_mol.GetMol()

    Chem.SanitizeMol(
        final_mol
    )

    return final_mol
