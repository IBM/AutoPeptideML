from typing import *

import numpy as np

from .engine import RepEngineBase


RESIDUES = {
    'V': 0,  'I': 1,  'L': 2,  'E': 3,  'Q': 4,
    'D': 5,  'N': 6,  'H': 7,  'W': 8,  'F': 9,
    'Y': 10, 'R': 11, 'K': 12, 'S': 13, 'T': 14,
    'M': 15, 'A': 16, 'G': 17, 'P': 18, 'C': 19,
    'X': 20
}


class RepEngineOnehot(RepEngineBase):
    """
    Class `RepEngineOnehot` is a subclass of `RepEngineBase` that generates one-hot encoded representations 
    for input sequences. This representation is commonly used for tasks in machine learning and bioinformatics, 
    such as protein sequence classification, where each amino acid in the sequence is represented by a binary vector.

    Attributes:
        :type engine: str
        :param engine: The name of the engine. Default is `'one-hot'`, indicating one-hot encoding representation.

        :type max_length: int
        :param max_length: The maximum length of the input sequences. Sequences longer than this length will be truncated.

        :type name: str
        :param name: The name of the representation engine, which is set to `'one-hot'`.
    """
    engine = 'one-hot'

    def __init__(self, max_length: int):
        """
        Initializes the `RepEngineOnehot` with the specified maximum sequence length. The one-hot encoding will
        use this length to determine the size of the output vectors.

        :type max_length: int
          :param max_length: The maximum length of the input sequences. Sequences longer than this will be truncated.

        :rtype: None
        """
        super().__init__('one-hot', max_length=max_length)
        self.max_length = max_length
        self.name = f'{self.engine}'

    def _preprocess_batch(self, batch: List[str]):
        """
        Preprocesses a batch of input sequences by truncating them to the specified maximum length.

        :type batch: List[str]
          :param batch: A list of input sequences (e.g., protein sequences in FASTA format).

        :rtype: List[str]
          :return: A list of preprocessed sequences truncated to the maximum length.
        """
        return [s[:self.max_length] for s in batch]

    def _rep_batch(self, batch: List[str]) -> np.ndarray:
        """
        Converts a batch of input sequences into one-hot encoded representations. Each amino acid in the sequence 
        is represented by a binary vector where the position corresponding to the amino acid is set to 1, and 
        all other positions are set to 0.

        :type batch: List[str]
          :param batch: A list of input sequences (e.g., protein sequences in FASTA format).

        :rtype: np.ndarray
          :return: A 2D numpy array where each row corresponds to a one-hot encoded representation of a sequence.
        """
        out = np.zeros((len(batch), self.max_length * len(RESIDUES)),
                       dtype=np.int8)
        for idx, s in enumerate(batch):
            for idx2, c in enumerate(s):
                out[idx, idx2 * len(RESIDUES) + RESIDUES[c]] = 1
        return out

    def dim(self) -> int:
        """
        Returns the dimensionality of the one-hot encoded representation, which is the product of the 
        maximum sequence length and the number of possible amino acids.

        :rtype: int
          :return: The dimensionality of the one-hot encoded representation.
        """
        return int(len(RESIDUES) * self.max_length)
