import os.path as osp

from itertools import combinations
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET

from .pipeline import BaseElement

try:
    import rdkit.Chem.rdmolfiles as rdm
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.Chem import DataStructs
except ImportError:
    raise ImportError("You need to install rdkit to use this method.",
                      " Try: `pip install rdkit`")


def read_chembl_library(path: str) -> Tuple[pd.DataFrame, Dict[str, str]]:
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
                 (monomer_data['MonomerSmiles'], monomer_data['NaturalAnalog'])}
        monomers.update(entry)
    return monomers


AAs = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'L', 'M',
       'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'X']
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
        """Code adapted from PepFuNN
        https://github.com/novonordisk-research/pepfunn


        Modifications include:
            - Added support for non-canonical residues
        """
        mol = rdm.MolFromSmiles(mol)
        fpgen = rdFingerprintGenerator.GetMorganGenerator(
            radius=2, fpSize=2048, includeChirality=True,
            countSimulation=True
        )
        if mol is None:
            raise RuntimeError(f'Molecule: {mol} could not be read by RDKit.',
                               'Maybe introduce a filtering step in your pipeline')
        CAatoms = mol.GetSubstructMatches(
            Chem.MolFromSmarts("[C:0](=[O:1])[C:2][N:3]")
        )

        for atoms in CAatoms:
            a = mol.GetAtomWithIdx(atoms[2])
            info = Chem.AtomPDBResidueInfo()
            info.SetName(" CA ")
            a.SetMonomerInfo(info)

        for aa, aa_smiles in AA_DICT.items():
            matches = mol.GetSubstructMatches(Chem.MolFromSmiles(aa_smiles))
            for atoms in matches:
                for atom in atoms:
                    a = mol.GetAtomWithIdx(atom)
                    info = Chem.AtomPDBResidueInfo()
                    if a.GetMonomerInfo() != None:
                        if a.GetMonomerInfo().GetName() == " CA ":
                            info.SetName(" CA ")
                            info.SetResidueName(aa)
                            a.SetMonomerInfo(info)

        # Renumber the backbone atoms so the sequence order is correct:
        mult = len(mol.GetSubstructMatches(Chem.MolFromSmiles(AA_DICT['G'])))
        bbsmiles = "O" + "C(=O)CN" * mult
        backbone = mol.GetSubstructMatches(Chem.MolFromSmiles(bbsmiles))[0]

        id_list = list(backbone)
        id_list.reverse()
        for idx in [a.GetIdx() for a in mol.GetAtoms()]:
            if idx not in id_list:
                id_list.append(idx)

        mol = Chem.RenumberAtoms(mol, newOrder=id_list)

        # Pattern of the AA backbone
        final_pep = []
        for patt in ['NCC(=O)N', 'NCC(=O)O']:
            pep_bond = Chem.MolFromSmarts(patt)
            am = np.array(Chem.GetAdjacencyMatrix(mol))

            for bond in mol.GetSubstructMatches(pep_bond):
                alpha = bond[1]
                nitrogens = set([bond[0], bond[-1]])
                aa_atom_idx = set([alpha])

                set2 = set()

                while aa_atom_idx != set2:
                    set2 = aa_atom_idx.copy()
                    temp_am = am[:, list(aa_atom_idx)]
                    aa_atom_idx = set(np.where(temp_am == 1)[0]) | aa_atom_idx
                    aa_atom_idx -= nitrogens

                aa_atom_idx.add(bond[0])

                bonds = []
                for i, j in combinations(aa_atom_idx, 2):
                    b = mol.GetBondBetweenAtoms(int(i), int(j))
                    if b:
                        bonds.append(b.GetIdx())

                mol1 = Chem.PathToSubmol(mol, bonds)
                flag = 0
                for aa, monomer in AA_DICT.items():
                    smiles2, _ = monomer
                    mol2 = Chem.MolFromSmiles(smiles2)
                    fp1 = fpgen.GetFingerprint(mol1)
                    fp2 = fpgen.GetFingerprint(mol2)
                    smiles_similarity = DataStructs.TanimotoSimilarity(fp1, fp2)

                    if smiles_similarity == 1.0:
                        final_pep.append(aa)
                        flag = 1
                        break

                if flag == 0:
                    try:
                        new_smiles1 = add_terminal_oxygen(
                            Chem.MolToSmiles(mol1)
                        )
                        mol1 = Chem.MolFromSmiles(new_smiles1)

                        for aa, monomer in AA_DICT.items():
                            smiles2, _ = monomer
                            mol2 = Chem.MolFromSmiles(smiles2)

                            fp1 = fpgen.GetFingerprint(mol1)
                            fp2 = fpgen.GetFingerprint(mol2)
                            smiles_similarity = DataStructs.TanimotoSimilarity(fp1, fp2)

                            if smiles_similarity == 1.0:
                                final_pep.append(aa)
                                flag = 1
                                break
                    except RuntimeError:
                        pass

                if flag == 0:
                    final_pep.append('X')

        if self.keep_analog:
            final_pep = [AA_DICT[r][1] for r in final_pep]
        else:
            final_pep = [r if r in AAs else 'X' for r in final_pep]
        return ''.join(final_pep)


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


def add_terminal_oxygen(aa_mol: Chem.Mol) -> Chem.Mol:
    """
    Code adapted from PepFuNN
    https://github.com/novonordisk-research/pepfunn

    Modifications include:
        - Input and output typing
        - Remove code comments

    Add terminal oxygen to an amino acid SMILES
    """
    backbone = Chem.MolFromSmarts('NCC(=O)')
    carboxyl_carbon_idx = aa_mol.GetSubstructMatch(backbone)[-2]

    mod = Chem.MolFromSmiles('O')
    new_mol = Chem.CombineMols(aa_mol, mod)
    max_atom_idx = new_mol.GetNumAtoms() - 1
    ed_mol = Chem.EditableMol(new_mol)

    ed_mol.AddBond(
        carboxyl_carbon_idx,
        max_atom_idx,
        order=Chem.rdchem.BondType.SINGLE
    )
    final_mol = ed_mol.GetMol()
    return final_mol

