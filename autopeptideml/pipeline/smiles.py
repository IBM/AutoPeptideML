from typing import *

from .pipeline import BaseElement
try:
    import rdkit.Chem.rdmolfiles as rdm
except ImportError:
    raise ImportError("You need to install rdkit to use this method.",
                      " Try: `pip install rdkit`")


def is_smiles(mol: str):
    return (
        '(' in mol or ')' in mol or
        '[' in mol or ']' in mol or
        '@' in mol or 'O' in mol
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
            raise RuntimeError(f'Molecule: {mol} could not be read by RDKit.')
        return rdm.MolToSmiles(rd_mol, canonical=True, isomericSmiles=True)


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
