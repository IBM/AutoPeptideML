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
    engine = 'one-hot'

    def __init__(self, max_length: int):
        super().__init__('one-hot', max_length=max_length)
        self.max_length = max_length

    def _preprocess_batch(self, batch: List[str]):
        return [s[:self.max_length] for s in batch]

    def _rep_batch(self, batch: List[str]) -> np.ndarray:
        out = np.zeros((len(batch), self.max_length * len(RESIDUES)),
                       dtype=np.int8)
        for idx, s in enumerate(batch):
            for idx2, c in enumerate(s):
                out[idx, idx2 * len(RESIDUES) + RESIDUES[c]] = 1
        return out

    def dim(self) -> int:
        return int(len(RESIDUES) * self.max_length)