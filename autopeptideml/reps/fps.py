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
    """
    Class `RepEngineFP` is a subclass of `RepEngineBase` designed for computing molecular fingerprints (FPs) 
    using popular fingerprinting algorithms such as ECFP or FCFP. This engine generates fixed-length bit vectors 
    representing molecular structures based on their topological features.

    Attributes:
        :type engine: str
        :param engine: The name of the engine. Default is `'fp'`, indicating a fingerprint-based representation.

        :type nbits: int
        :param nbits: The length of the fingerprint bit vector. This determines the number of bits in the fingerprint.

        :type radius: int
        :param radius: The radius parameter used for fingerprint generation, determining the neighborhood size around each atom.

        :type name: str
        :param name: The name of the fingerprint generator, which includes the engine type, `nbits`, and `radius`.

        :type generator: object
        :param generator: The fingerprint generator object, loaded based on the specified `rep` type.
    """
    engine = 'fp'

    def __init__(self, rep: str, nbits: int, radius: int):
        """
        Initializes the `RepEngineFP` with the specified representation type, fingerprint size, and radius.

        :type rep: str
          :param rep: The type of fingerprint to generate (e.g., 'ecfp', 'fcfp').

        :type nbits: int
          :param nbits: The length of the fingerprint bit vector.

        :type radius: int
          :param radius: The radius of the neighborhood around each atom to consider when generating the fingerprint.

        :rtype: None
        """
        super().__init__(rep, nbits=nbits, radius=radius)
        self.nbits = nbits
        self.radius = radius
        self.name = f'{self.engine}-{rep}-{self.nbits}-{self.radius}'
        self.generator = self._load_generator(rep)

    def _preprocess_batch(self, batch: List[str]) -> List[str]:
        """
        Preprocesses a batch of molecular representations. For this class, no preprocessing is required.

        :type batch: List[str]
          :param batch: A list of molecular representations (e.g., SMILES strings).

        :rtype: List[str]
          :return: The same batch of molecular representations as input.
        """
        return batch

    def _rep_batch(self, batch: List[str]) -> List[np.ndarray]:
        """
        Computes the fingerprint for each molecule in a batch and returns the results as a list of NumPy arrays.

        :type batch: List[str]
          :param batch: A list of molecular representations (e.g., SMILES strings).

        :rtype: List[np.ndarray]
          :return: A list of NumPy arrays representing the molecular fingerprints.
        """
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
        """
        Loads the appropriate fingerprint generator based on the specified representation type.

        :type rep: str
          :param rep: The type of fingerprint to generate (e.g., 'ecfp', 'fcfp').

        :rtype: object
          :return: The fingerprint generator object based on the `rep` type.

        :raises NotImplementedError: If the specified `rep` type is not supported.
        """
        if 'ecfp' in rep or 'morgan' in rep:
            return rfp.GetMorganGenerator(radius=self.radius,
                                          includeChirality=True,
                                          fpSize=self.nbits,
                                          countSimulation='count' in rep)
        elif 'fcfp' in rep:
            invgen = rfp.GetMorganFeatureAtomInvGen()
            return rfp.GetMorganGenerator(radius=self.radius,
                                          fpSize=self.nbits,
                                          includeChirality=True,
                                          atomInvariantsGenerator=invgen,
                                          countSimulation='count' in rep)
        else:
            raise NotImplementedError(
                f'Representation: {rep} is not currently implemented.',
                'Please, request this new feature in the Issues page of the',
                'github repo: https://IBM/AutoPeptideML'
            )

    def dim(self) -> int:
        """
        Returns the dimensionality (bit size) of the generated fingerprint.

        :rtype: int
          :return: The number of bits in the fingerprint (i.e., `nbits`).
        """
        return self.nbits
