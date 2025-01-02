from typing import *

from ..pipeline import BaseElement
try:
    import rdkit.Chem.rdmolfiles as rdm
except ImportError:
    raise ImportError("You need to install rdkit to use this method.",
                      " Try: `pip install rdkit`")


class SequenceToSMILES(BaseElement):
    name = 'sequence-to-smiles'

    def _single_call(self, mol):
        rd_mol = rdm.MolFromFASTA(mol)
        if rd_mol is None:
            raise RuntimeError(f'Molecule: {mol} could not be read by RDKit.')
        return rdm.MolToSmiles(rd_mol, canonical=True, isomericSmiles=True)


class FilterSMILES(BaseElement):
    name = 'filter-smiles'

    def __init__(self, keep_smiles: Optional[bool] = True):
        self.properties['keep_smiles'] = keep_smiles
        self.keep_smiles = keep_smiles

    def _single_call(self, mol):
        if ('(' in mol or
            ')' in mol or
            '[' in mol or
            ']' in mol or
           '@' in mol):
            if self.keep_smiles:
                return mol
            else:
                return None
        else:
            if self.keep_smiles:
                return None
            else:
                return mol