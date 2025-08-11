import copy
import os.path as osp
import re

from typing import Dict, List, Optional, Tuple

import xml.etree.ElementTree as ET

from .pipeline import BaseElement

try:
    import rdkit.Chem.rdmolfiles as rdm
    import rdkit.Chem.rdmolops as rdops

    from rdkit import Chem, rdBase
    from rdkit.Chem import DataStructs, MolFromSmiles, MolToSmiles, RWMol, Mol
    from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
except ImportError:
    raise ImportError("You need to install rdkit to use this method.",
                      "Try: `pip install rdkit`")


def read_chembl_library(path: str) -> Dict[str, Tuple[str, str]]:
    """
    Loads and parses a ChEMBL monomer library XML file to extract monomer data.

    This function reads the XML file located at `<path>/chembl_monomer_library.xml` and parses
    all `<Monomer>` elements within the following structure:

        MonomerDB > PolymerList > Polymer > Monomer

    It returns a dictionary mapping monomer IDs to their corresponding SMILES strings and Natural Analog.

    :type path: str
    :param path: Path to the ChEMBL monomer library XML file (`chembl_monomer_library.xml`).

    :rtype: Dict[str, Tuple[str, str]]
    :return: A dictionary mapping `MonomerID` to `MonomerSmiles` and `NaturalAnalog`.

    :raises FileNotFoundError: If the XML file cannot be found at the given path.
    :raises xml.etree.ElementTree.ParseError: If the XML file is malformed or unreadable.
    """
    tree = ET.parse(path)
    root = tree.getroot()
    ns = {'lmr': root.tag[root.tag.find('{')+1:root.tag.find('}')]}

    monomers = {}
    for monomer in root.findall('.//lmr:Monomer', ns):
        monomer_data = {
            'MonomerID': monomer.findtext('lmr:MonomerID', default='',
                                          namespaces=ns),
            'MonomerSmiles': monomer.findtext('lmr:MonomerSmiles', default='',
                                              namespaces=ns),
            'MonomerMolFile': monomer.findtext('lmr:MonomerMolFile',
                                               default='', namespaces=ns),
            'MonomerType': monomer.findtext('lmr:MonomerType', default='',
                                            namespaces=ns),
            'PolymerType': monomer.findtext('lmr:PolymerType', default='',
                                            namespaces=ns),
            'NaturalAnalog': monomer.findtext('lmr:NaturalAnalog', default='',
                                              namespaces=ns),
            'MonomerName': monomer.findtext('lmr:MonomerName', default='',
                                            namespaces=ns),
        }
        entry = {monomer_data['MonomerID']:
                 (monomer_data['MonomerSmiles'],
                  monomer_data['NaturalAnalog'])}
        monomers.update(entry)
    return monomers


AAs = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'L', 'M',
       'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W']
AA_DICT = read_chembl_library(osp.join(
    osp.dirname(__file__), '..', 'data', 'chembl_monomer_library.xml')
)
CACHE = {}


class SequenceToSmiles(BaseElement):
    """
    Class `SequenceToSMILES` converts peptide sequences (e.g., FASTA format) into SMILES (Simplified Molecular Input Line Entry System) representations using RDKit.

    Attributes:
        :type name: str
        :param name: The name of the element. Default is `'sequence-to-smiles'`.
    """
    name = 'sequence-to-smiles'
    parallel = 'processing'

    def _single_call(self, mol):
        """
        Converts a single peptide sequence into a SMILES representation.

        :type mol: str
          :param mol: A peptide sequence (e.g., FASTA format).

        :rtype: str
          :return: The SMILES representation of the molecule.

        :raises RuntimeError: If the molecule cannot be read by RDKit.
        """
        rd_mol = rdm.MolFromFASTA(mol)
        if rd_mol is None:
            raise RuntimeError(f'Molecule: {mol} could not be read by RDKit.',
                               'Maybe introduce a filtering step in your pipeline')
        return rdm.MolToSmiles(rd_mol, canonical=True, isomericSmiles=True)


class SmilesToSequence(BaseElement):

    name = 'smiles-to-sequence'
    parallel = 'processing'

    def __init__(self, keep_analog: bool = True):
        """
        Initializes the `SmilesToSequence` keeping the natural analog of the non-canonical residues.

        :type substitution: bool
        :param keep_analog: Whether to keep the natural analog of the non-canonical residues. Otherwise, marks them as 'X'.
            Default is True.

        :rtype: None
        """
        self.keep_analog = keep_analog

    def _single_call(self, mol):
        final_pep = break_into_monomers(mol)[0]
        if not isinstance(final_pep, list):
            raise ValueError(mol, final_pep)

        if self.keep_analog:
            final_pep = [AA_DICT[r][1] if r != 'X' else r for r in final_pep]
        else:
            final_pep = [r if r in AAs else 'X' for r in final_pep]
        return ''.join(final_pep)


class SmilesToBiln(BaseElement):
    name = "smiles-to-biln"

    def _single_call(self, mol):
        final_pep = break_into_monomers(mol)[0]

        return '-'.join(final_pep)


class BilnToSmiles(BaseElement):
    name = "biln-to-smiles"

    def _single_call(self, mol):
        monomers = mol.split('-')
        monomers_as_smiles = []
        for monomer in monomers:
            if monomer == 'X':
                print("Warning: Monomer X is being substituted by G.")
                monomer = 'G'
            monomer_smiles = AA_DICT[monomer][0]
            monomers_as_smiles.append((monomer, monomer_smiles))

        if len(monomers_as_smiles) == 1:
            return monomer

        peptide, _ = build_peptide(monomers_as_smiles)
        return peptide


class FilterSmiles(BaseElement):
    """
    Class `FilterSMILES` filters molecular representations based on whether they are valid SMILES strings. 
    It can either retain or discard SMILES strings based on the configuration.

    Attributes:
        :type name: str
        :param name: The name of the element. Default is `'filter-smiles'`.

        :type keep_smiles: Optional[bool]
        :param keep_smiles: Determines whether to retain valid SMILES strings (`True`) or discard them (`False`).
                            Default is `True`.
    """
    name = 'filter-smiles'

    def __init__(self, keep_smiles: Optional[bool] = True):
        """
        Initializes the `FilterSMILES` element with a configuration to retain or discard SMILES strings.

        :type keep_smiles: Optional[bool]
          :param keep_smiles: Determines whether to retain valid SMILES strings (`True`) or discard them (`False`).
                              Default is `True`.

        :rtype: None
        """
        self.properties['keep_smiles'] = keep_smiles
        self.keep_smiles = keep_smiles

    def _single_call(self, mol: str):
        """
        Filters a single molecular representation based on its validity as a SMILES string.

        :type mol: str
          :param mol: A molecular representation to evaluate.

        :rtype: Union[str, None]
          :return: The molecule if it meets the SMILES validity condition, or `None` otherwise.
        """
        if ((is_smiles(mol) and self.keep_smiles) or
           (not is_smiles(mol) and not self.keep_smiles)):
            return mol
        else:
            return None

    def __str__(self):
        return self.name + f" -> keep_smiles: {self.keep_smiles}"


class CanonicalizeSmiles(BaseElement):
    """
    Class `CanonicalizeSmiles` converts SMILES (Simplified Molecular Input Line Entry System) strings into their canonical forms using RDKit.

    Attributes:
        :type name: str
        :param name: The name of the element. Default is `'canonicalize-smiles'`.
    """
    name = 'canonicalize-smiles'

    def _single_call(self, mol):
        """
        Converts a SMILES string into its canonical representation.

        :type mol: str
          :param mol: A SMILES string representing a molecule.

        :rtype: str
          :return: The canonical SMILES representation of the molecule.

        :raises RuntimeError: If the molecule cannot be read by RDKit.
        """
        rd_mol = rdm.MolFromSmiles(mol)
        if rd_mol is None:
            raise RuntimeError(f'Molecule: {mol} could not be read by RDKit.',
                               'Maybe introduce a filtering step in your pipeline')
        return rdm.MolToSmiles(rd_mol, canonical=True, isomericSmiles=True)


def is_smiles(mol: str):
    return (
        '(' in mol or ')' in mol or
        '[' in mol or ']' in mol or
        '@' in mol or 'O' in mol
    )


def add_dummy_atoms(mol: Mol) -> Mol:
    mol = Chem.AddHs(mol)
    mol = Chem.RWMol(mol)

    for atom in mol.GetAtoms():
        # Nitrogens in C-NH2
        if atom.GetAtomicNum() == 7:
            neighbors = atom.GetNeighbors()
            h_atoms = [n for n in neighbors if n.GetAtomicNum() == 1]
            dummy_atoms = [n for n in neighbors if n.GetAtomicNum() == 0]
            h_count = len(h_atoms)
            dummy_count = len(dummy_atoms)

            if h_count >= 2 and dummy_count == 0:
                for h in h_atoms:
                    mol.RemoveAtom(h.GetIdx())
                    break
                dummy_idx = mol.AddAtom(Chem.Atom(0))
                mol.AddBond(atom.GetIdx(), dummy_idx, Chem.BondType.SINGLE)

        # Oxygens in COOH
        elif atom.GetAtomicNum() == 8:
            parent = atom.GetNeighbors()[0]
            if parent.GetAtomicNum() == 6 and parent.GetTotalValence() == 4:
                parent_is_carboxy = len([n for n in parent.GetNeighbors() if n.GetAtomicNum() == 8]) == 2
                atom_is_hidroxy = (len([n for n in atom.GetNeighbors() if n.GetAtomicNum() == 1]) == 1)

                if parent_is_carboxy and atom_is_hidroxy:
                    mol.RemoveAtom(atom.GetIdx())
                    dummy_idx = mol.AddAtom(Chem.Atom(0))
                    mol.AddBond(parent.GetIdx(), dummy_idx, Chem.BondType.SINGLE)

        # Sulphur in C-SH2
        elif atom.GetAtomicNum() == 16:
            h_atoms = [n for n in atom.GetNeighbors() if n.GetAtomicNum() == 1]
            h_count = len(h_atoms)
            if h_count == 1:
                for h in h_atoms:
                    mol.RemoveAtom(h.GetIdx())
                dummy_idx = mol.AddAtom(Chem.Atom(0))
                mol.AddBond(atom.GetIdx(), dummy_idx, Chem.BondType.SINGLE)

    mol = Chem.RemoveAllHs(mol)
    return mol


def find_lone_nitrogen_fragments(fragments: List[Mol]):
    """
    Given a list of RDKit Mol objects (fragments),
    returns a list of (index, fragment) where the fragment
    is essentially a 'lone nitrogen' or dummy-capped nitrogen.
    """
    lone_n_list = []
    for idx, frag in enumerate(fragments):
        atoms = frag.GetAtoms()
        # Count real atoms (exclude dummies, atomicNum=0)
        real_atoms = [a for a in atoms if a.GetAtomicNum() != 0]
        n_atoms = [a for a in real_atoms if a.GetAtomicNum() == 7]

        # "Lone nitrogen" = fragment has exactly 1 real atom and it’s nitrogen
        # OR only nitrogen + dummy atoms
        if len(real_atoms) == 1 and len(n_atoms) == 1:
            lone_n_list.append(idx)

    return lone_n_list


def reattach_n(fragments: List[Mol]):
    blocker = rdBase.BlockLogs()
    new_fragments, avoid = [], []
    lone_n = find_lone_nitrogen_fragments(fragments)
    for n in lone_n:
        n_frag = fragments[n]
        for idx, frag in enumerate(fragments):
            if idx == n:
                continue

            if find_closest_monomer(frag)[0] in ('N', 'Q', 'E', 'D'):
                new_frag = rdops.molzip(n_frag, frag)
                if len(MolToSmiles(new_frag).split('.')) == 1:
                    new_fragments.append(new_frag)
                    avoid.append(idx)
                    avoid.append(lone_n)
                    break

    for idx, frag in enumerate(fragments):
        if idx not in avoid:
            new_fragments.append(frag)

    return new_fragments


def break_into_monomers(smiles: str) -> Tuple[List[str], List[Chem.Mol]]:
    """
    Breaks a peptide SMILES into its monomers and assigns matching molAtomMapNumber
    to each pair of dummy atoms resulting from bond breaking.
    """
    mol = MolFromSmiles(smiles, sanitize=True)
    if mol is None:
        raise RuntimeError(f'Molecule: {smiles} could not be read by RDKit.',
                           'Maybe introduce a filtering step in your pipeline')

    # Patterns for peptide and disulfide bonds
    patt = 'N[C](=O)C'
    patt2 = "CSSC"
    pep_bond = Chem.MolFromSmarts(patt)
    disulfide_bond = Chem.MolFromSmarts(patt2)
    pep_matches = mol.GetSubstructMatches(pep_bond)
    ss_matches = mol.GetSubstructMatches(disulfide_bond)

    bond_indices_pep = [
        mol.GetBondBetweenAtoms(n_idx, c_idx).GetIdx()
        for n_idx, c_idx, *_ in pep_matches
        if mol.GetBondBetweenAtoms(n_idx, c_idx)
    ]
    bond_indices_ss = [
        mol.GetBondBetweenAtoms(s1_idx, s2_idx).GetIdx()
        for _, s1_idx, s2_idx, _ in ss_matches
        if mol.GetBondBetweenAtoms(s1_idx, s2_idx)
    ]
    bond_indices = bond_indices_pep + bond_indices_ss

    if not bond_indices:
        best_aa, _ = find_closest_monomer(mol)
        return [best_aa], [mol]
    # Fragment and retain dummy atom pairs
    frags = rdops.FragmentOnBonds(mol, bond_indices, addDummies=True,
                                  dummyLabels=[[i+1, i+1] for i in range(len(bond_indices))])

    # Assign matching molAtomMapNumber to dummy pairs
    for bond_num, bond_idx in enumerate(bond_indices, start=1):
        dummy_atoms = [atom for atom in frags.GetAtoms()
                       if atom.GetAtomicNum() == 0 and
                       atom.GetIsotope() == bond_num]

        for atom in dummy_atoms:
            atom.SetIntProp("molAtomMapNumber", bond_num)

    # Re-extract sanitized fragments with new properties
    updated_frag_mols = Chem.GetMolFrags(frags, asMols=True,
                                         sanitizeFrags=True)
    n_fixed_frags = reattach_n(updated_frag_mols)
    ordered_frags = order_monomers(n_fixed_frags)

    final_pep, all_frags = [], []
    for frag in ordered_frags:
        best_aa, sim = find_closest_monomer(frag)
        final_pep.append(best_aa)
        all_frags.append(frag)
    return final_pep, all_frags


def order_monomers(monomers: List[Chem.Mol]) -> List[Chem.Mol]:
    """
    Orders peptide monomers from N-terminal to C-terminal.
    Supports cyclic peptides by detecting an artificial break.

    Args:
        monomers: List of RDKit Mol objects representing peptide monomers,
                  with dummy atoms carrying matching molAtomMapNumber
                  for their connection points.

    Returns:
        Ordered list of RDKit Mol objects in sequence order.
    """

    # Step 1: Build adjacency from dummy atom labels
    connections = {}
    for i, mol in enumerate(monomers):
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 0 and atom.HasProp("molAtomMapNumber"):
                label = atom.GetIntProp("molAtomMapNumber")
                if label not in connections:
                    connections[label] = []
                connections[label].append(i)

    # Step 2: Build graph of monomer connections
    graph = {i: set() for i in range(len(monomers))}
    for mol_ids in connections.values():
        if len(mol_ids) == 2:
            a, b = mol_ids
            graph[a].add(b)
            graph[b].add(a)

    # Step 3: Find N-terminal (monomer with free amine and only one neighbor)
    start_idx = None
    for idx, mol in enumerate(monomers):
        if len(graph[idx]) == 1:  # terminal candidate
            smi = MolToSmiles(mol)
            if "N" in smi and "[OH]" not in smi:  # crude free amine detection
                start_idx = idx
                break

    # If cyclic peptide, pick arbitrary start
    if start_idx is None:
        start_idx = 0

    # Step 4: Traverse graph to get sequence order
    ordered = []
    visited = set()
    current = start_idx
    prev = None
    while True:
        ordered.append(current)
        visited.add(current)
        neighbors = graph[current] - ({prev} if prev is not None else set())
        if not neighbors:
            break  # reached free C-terminal
        prev, current = current, neighbors.pop()
        if current in visited:
            break  # cyclic peptide complete

    return [monomers[i] for i in ordered]


def find_closest_monomer(frag: Chem.Mol) -> Tuple[str, float]:
    global CACHE
    blocker = rdBase.BlockLogs()
    max_sim, best_aa = 0.7, 'X'

    mol1 = add_dummy_atoms(frag)
    fp1 = GetMorganFingerprintAsBitVect(
        mol1, radius=2, useFeatures=True, nBits=1024, useChirality=True
    )
    for aa in AAs:
        monomer = AA_DICT[aa]
        smiles_similarity, _ = compare(monomer, aa, fp1)
        if smiles_similarity > max_sim:
            max_sim = smiles_similarity
            best_aa = aa
        if max_sim == 1.0:
            return best_aa, max_sim

    for aa, monomer in AA_DICT.items():
        if aa in AAs:
            continue
        smiles_similarity, smiles2 = compare(monomer, aa, fp1)

        if smiles_similarity > max_sim:
            max_sim = smiles_similarity
            best_aa = aa
        if max_sim == 1.0:
            max_sim = smiles_similarity
            best_aa = aa

            mol2 = MolFromSmiles(smiles2)
            atoms1 = set([a.GetAtomicNum() for a in mol1.GetAtoms()])
            atoms2 = set([a.GetAtomicNum() for a in mol2.GetAtoms()])
            if len(atoms1.intersection(atoms2)) == len(atoms1):
                break

    return best_aa, max_sim


def compare(monomer, aa, fp1):
    global CACHE
    smiles2, _ = monomer
    smiles2 = smiles2.split(' ')[0]

    if smiles2 in CACHE:
        fp2 = CACHE[smiles2]
    else:
        mol2 = MolFromSmiles(smiles2, sanitize=True)
        fp2 = GetMorganFingerprintAsBitVect(
            mol2, radius=2, useFeatures=True, nBits=1024, useChirality=True
        )
        CACHE[smiles2] = fp2
    smiles_similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
    return smiles_similarity, smiles2


def build_peptide(monomerlist: List[Tuple[str, str]]) -> Tuple[str, List[str]]:
    """
    Assemble a peptide from a list of monomer SMILES strings.

    This function takes a list of monomers (represented as SMILES), connects them 
    in order via peptide bonds, and returns the final product as a SMILES string.
    Dummy atoms are removed or capped appropriately.

    :param monomerlist: List of monomer SMILES strings.
    :type monomerlist: List[str]
    :return: Tuple of SMILES of the assembled peptide and List of the monomers it is comprised of.
    :rtype: Tuple[str, List[str]]
    """
    monomerlist = list(monomerlist)
    monomerlist = copy.deepcopy(monomerlist)
    monomers = []
    for idx, monomer in enumerate(monomerlist):
        monomers.append(monomer[0])
        mol = MolFromSmiles(monomer[1])
        if mol is None:
            try:
                mol = MolFromSmiles(monomer[1], sanitize=False)
                if mol is None:
                    raise ValueError("MolFromSmiles returned None")
                Chem.SanitizeMol(mol)
            except Exception as e:
                print(f"[ERROR] Failed to parse or sanitize SMILES: {monomer[1]}")
                print(f"Reason: {e}")
                raise RuntimeError
        if idx == 0:
            res = mol
        else:
            res = _combine_fragments(res, mol)
    return (rdm.MolToSmiles(_clean_peptide(res), canonical=True, isomericSmiles=True), monomers)


def _combine_fragments(m1: str, m2: str) -> Mol:
    """
    Combine two RDKit molecule fragments using labeled attachment points.

    Atom labels '_R2' and '_R1' are used to identify the carboxylic and amino 
    attachment points respectively. If these labels are missing, an error is raised.

    :param m1: RDKit molecule (as string or Mol object) with an '_R2' attachment point.
    :type m1: str
    :param m2: RDKit molecule (as string or Mol object) with an '_R1' attachment point.
    :type m2: str
    :return: Combined molecule with a peptide bond between m1 and m2.
    :rtype: Mol
    :raises RuntimeError: If required attachment points are not found in either monomer.
    """
    blocker = rdBase.BlockLogs()
    m1_success, m2_success = False, False

    for atm in m1.GetAtoms():
        if atm.HasProp('atomLabel') and atm.GetProp('atomLabel') == '_R2':
            atm.SetAtomMapNum(1)
            m1_success = True
    for atm in m2.GetAtoms():
        if atm.HasProp('atomLabel') and atm.GetProp('atomLabel') == '_R1':
            atm.SetAtomMapNum(1)
            m2_success = True
    if not m1_success:
        raise RuntimeError("Molecule 1 does not have a free amino group for attachment.")
    if not m2_success:
        raise RuntimeError("Molecule 2 does not have a free carboxy group for attachment.")
    return rdops.molzip(m1, m2)


def _clean_peptide(mol: Mol) -> Mol:
    """
    Clean a peptide by removing or replacing dummy atoms.

    - Removes dummy atoms (*) attached to nitrogen atoms (N[*]).
    - Replaces dummy atoms attached to carbonyl carbon atoms (C([*])=O) with hydroxyl groups (→ COOH).
    - Removes dummy atoms (*) attached to sulphur atoms (S[*])

    :param mol: RDKit molecule to modify.
    :type mol: Mol
    :return: Modified molecule with proper N-/C-terminal capping.
    :rtype: Mol
    """
    rw_mol = RWMol(mol)
    atoms_to_remove = []
    attach_oh_to = []

    # First, scan and collect targets
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == '*':
            neighbors = atom.GetNeighbors()
            if len(neighbors) != 1:
                continue
            neighbor = neighbors[0]

            # Case 1: dummy attached to N → mark dummy for removal
            if neighbor.GetSymbol() == 'N':
                atoms_to_remove.append(atom.GetIdx())

            # Case 2: dummy attached to carbonyl carbon (C=O)
            elif neighbor.GetSymbol() == 'C':
                carbon = neighbor
                is_carbonyl = any(
                    n.GetSymbol() == 'O' and mol.GetBondBetweenAtoms(carbon.GetIdx(), n.GetIdx()).GetBondType() == Chem.BondType.DOUBLE
                    for n in carbon.GetNeighbors()
                )
                if is_carbonyl:
                    atoms_to_remove.append(atom.GetIdx())
                    attach_oh_to.append(carbon.GetIdx())

            # Case 3: dummy attached to sulphur (CS)
            elif neighbor.GetSymbol() == 'S':
                atoms_to_remove.append(atom.GetIdx())

    # Now, modify molecule safely
    for carbon_idx in attach_oh_to:
        o_idx = rw_mol.AddAtom(Chem.Atom("O"))
        h_idx = rw_mol.AddAtom(Chem.Atom("H"))
        rw_mol.AddBond(carbon_idx, o_idx, Chem.BondType.SINGLE)
        rw_mol.AddBond(o_idx, h_idx, Chem.BondType.SINGLE)

    # Remove dummy atoms (do in reverse order to avoid reindexing issues)
    for idx in sorted(atoms_to_remove, reverse=True):
        rw_mol.RemoveAtom(idx)

    final_mol = rw_mol.GetMol()
    Chem.SanitizeMol(final_mol)
    return final_mol
