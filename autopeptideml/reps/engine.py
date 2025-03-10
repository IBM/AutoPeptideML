import copy
import json
import yaml
from typing import *

import numpy as np
from tqdm import tqdm

try:
    from itertools import batched
except ImportError:
    from itertools import islice

    def batched(iterable, n, *, strict=False):
        # batched('ABCDEFG', 3) → ABC DEF G
        if n < 1:
            raise ValueError('n must be at least one')
        iterator = iter(iterable)
        while batch := tuple(islice(iterator, n)):
            if strict and len(batch) != n:
                raise ValueError('batched(): incomplete batch')
            yield batch


class RepEngineBase:
    """
    Class `RepEngineBase` is an abstract base class for implementing molecular representation engines. 
    It defines a framework for computing molecular representations in batches and includes utilities for 
    serialization and property management.

    Attributes:
        :type engine: str
        :param engine: The name of the representation engine.

        :type rep: str
        :param rep: The type of molecular representation (e.g., fingerprint, embedding).

        :type properties: dict
        :param properties: A dictionary containing the engine's properties, including configuration arguments passed during initialization.
    """
    engine: str

    def __init__(self, rep: str, **args):
        """
        Initializes the `RepEngineBase` with the specified representation type and additional configuration arguments.

        :type rep: str
          :param rep: The type of molecular representation (e.g., fingerprint, embedding).

        :type **args: dict
          :param **args: Additional arguments for configuring the representation engine.

        :rtype: None
        """
        self.rep = rep
        self.__dict__.update(args)
        self.properties = copy.deepcopy(self.__dict__)

    def compute_reps(self, mols: List[str],
                     verbose: Optional[bool] = False,
                     batch_size: Optional[int] = 12) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Computes molecular representations for a list of molecules in batches.

        :type mols: List[str]
          :param mols: A list of molecular representations (e.g., SMILES strings).

        :type verbose: Optional[bool]
          :param verbose: If `True`, displays a progress bar during batch processing. Default is `False`.

        :type batch_size: Optional[int]
          :param batch_size: The size of each batch for processing. Default is `12`.

        :rtype: Union[np.ndarray, List[np.ndarray]]
          :return: A stacked NumPy array of computed representations, or a list of arrays if pooling is disabled.
        """
        batches = batched(mols, batch_size)
        out = []

        if verbose:
            pbar = tqdm(list(batches))
        else:
            pbar = batches

        for batch in pbar:
            batch = self._preprocess_batch(batch)
            rep = self._rep_batch(batch)
            out.extend(rep)

        if 'average_pooling' in self.__dict__:
            if not self.__dict__['average_pooling']:
                return out
        return np.stack(out)

    def dim(self) -> int:
        """
        Returns the dimensionality of the molecular representations.

        :rtype: int
          :return: The dimensionality of the computed representations.

        :raises NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError

    def _rep_batch(self, batch: List[str]) -> np.ndarray:
        """
        Computes representations for a batch of molecules. Must be implemented by subclasses.

        :type batch: List[str]
          :param batch: A batch of molecular representations (e.g., SMILES strings).

        :rtype: np.ndarray
          :return: A NumPy array of computed representations for the batch.

        :raises NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError

    def _preprocess_batch(self, batch: List[str]) -> List[str]:
        """
        Preprocesses a batch of molecules before computing representations. Must be implemented by subclasses.

        :type batch: List[str]
          :param batch: A batch of molecular representations (e.g., SMILES strings).

        :rtype: List[str]
          :return: A preprocessed list of molecular representations.

        :raises NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError

    def save(self, filename: str):
        """
        Saves the engine's properties to a file in YAML format.

        :type filename: str
          :param filename: The path to the file where the properties will be saved.

        :rtype: None
        """
        yaml.safe_dump(self.properties, open(filename, 'w'))

    def __str__(self) -> str:
        """
        Returns a string representation of the engine's properties in JSON format.

        :rtype: str
          :return: A JSON string representation of the engine's properties.
        """
        return str(json.dumps(self.properties))
