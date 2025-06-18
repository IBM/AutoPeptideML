import os.path as osp

from typing import Dict, List, Optional, Tuple

import xml.etree.ElementTree as ET

from .pipeline import BaseElement

try:
    import rdkit.Chem.rdmolfiles as rdm
    import rdkit.Chem.rdmolops as rdops
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.Chem import DataStructs
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


class SequenceToSMILES(BaseElement):
    """
    Class `SequenceToSMILES` converts peptide sequences (e.g., FASTA format) into SMILES (Simplified Molecular Input Line Entry System) representations using RDKit.

    Attributes:
        :type name: str
        :param name: The name of the element. Default is `'sequence-to-smiles'`.
    """
    name = 'sequence-to-smiles'

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
        final_pep = break_into_monomers(mol)
        if not isinstance(final_pep, list):
            raise ValueError(mol, final_pep)

        if self.keep_analog:
            final_pep = [AA_DICT[r][1] if r != 'X' else r for r in final_pep]
        else:
            final_pep = [r if r in AAs else 'X' for r in final_pep]
        return ''.join(final_pep)


class SmilesToBILN(BaseElement):
    name = "smiles-to-biln"

    def _single_call(self, mol):
        final_pep = break_into_monomers(mol)

        return '-'.join(final_pep)


class FilterSMILES(BaseElement):
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


def add_dummy_atoms(mol: Chem.Mol) -> Chem.Mol:
    mol = Chem.AddHs(mol)
    mol = Chem.RWMol(mol)

    for atom in mol.GetAtoms():
        # Nitrogens in C-NH2
        if atom.GetAtomicNum() == 7:
            neighbors = atom.GetNeighbors()
            h_atoms = [n for n in neighbors if n.GetAtomicNum() == 1]
            h_count = len(h_atoms)

            if h_count == 2:
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


def break_into_monomers(smiles: str) -> List[str]:
    """Breaks a given molecule into its constituent amino acid monomers.

    :type mol: str
        :param smiles: A peptide SMILES.

    :rtype: List[str]
        :return: A list of the monomers comprising the peptide
    """
    mol = rdm.MolFromSmiles(smiles, sanitize=True)
    fpgen = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=2048, includeChirality=True,
        countSimulation=True
    )
    if mol is None:
        raise RuntimeError(f'Molecule: {smiles} could not be read by RDKit.',
                           'Maybe introduce a filtering step in your pipeline')
    final_pep = []

    patt = 'N[C](=O)C'
    pep_bond = Chem.MolFromSmarts(patt)
    matches = mol.GetSubstructMatches(pep_bond)

    bond_indices = [
        mol.GetBondBetweenAtoms(n_idx, c_idx).GetIdx()
        for n_idx, c_idx, *_ in matches
        if mol.GetBondBetweenAtoms(n_idx, c_idx)
    ]
    if not bond_indices:
        return ['X']

    # Fragment the molecule at peptide bonds
    frags = rdops.FragmentOnBonds(mol, bond_indices, addDummies=True)
    frag_mols = Chem.GetMolFrags(frags, asMols=True, sanitizeFrags=True)
    for frag in frag_mols:
        max_sim, best_aa = 0.3, 'X'

        mol1 = add_dummy_atoms(frag)
        fp1 = fpgen.GetFingerprint(mol1)

        for aa, monomer in AA_DICT.items():
            smiles2, _ = monomer
            smiles2 = smiles2.split(' ')[0]
            mol2 = Chem.MolFromSmiles(smiles2, sanitize=True)
            mol2 = Chem.RemoveAllHs(mol2, sanitize=True)
            fp2 = fpgen.GetFingerprint(mol2)
            smiles_similarity = DataStructs.TanimotoSimilarity(fp1, fp2)

            if smiles_similarity > max_sim:
                max_sim = smiles_similarity
                best_aa = aa
            print(best_aa)

        final_pep.append(best_aa)
    return final_pep
