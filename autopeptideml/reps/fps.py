from typing import *

import numpy as np

from .engine import RepEngineBase
try:
    import rdkit.Chem.rdmolfiles as rdm
    from rdkit.Chem import rdFingerprintGenerator as rfp
except ImportError:
    raise ImportError("You need to install rdkit to use this method.",
                      " Try: `pip install rdkit`")


class RepEngineFP(RepEngineBase):
    engine = 'fp'

    def __init__(self, rep: str, nbits: int, radius: int):
        super().__init__(rep, nbits=nbits, radius=radius)
        self.nbits = nbits
        self.radius = radius
        self.name = f'{self.engine}-{self.nbits}-{self.radius}'
        self.generator = self._load_generator(rep)

    def _preprocess_batch(self, batch: List[str]) -> List[str]:
        return batch

    def _rep_batch(self, batch: List[str]) -> List[np.ndarray]:
        out = []
        for i in batch:
            mol = rdm.MolFromSmiles(i)
            if mol is None:
                fp = np.zeros((1, self.nbits))
            else:
                fp = self.generator.GetCountFingerprintAsNumPy(mol)
            out.append(fp)
        return out

    def _load_generator(self, rep: str):
        if rep == 'ecfp' or 'morgan' in rep:
            return rfp.GetMorganGenerator(radius=self.radius,
                                          includeChirality=True,
                                          fpSize=self.nbits)
        elif rep == 'fcfp':
            invgen = rfp.GetMorganFeatureAtomInvGen()
            return rfp.GetMorganGenerator(radius=self.radius,
                                          fpSize=self.nbits,
                                          includeChirality=True,
                                          atomInvariantsGenerator=invgen)
        else:
            raise NotImplementedError(
                f'Representation: {rep} is not currently implemented.',
                'Please, request this new feature in the Issues page of the',
                'github repo: https://IBM/AutoPeptideML'
            )

    def dim(self) -> int:
        return self.nbits
