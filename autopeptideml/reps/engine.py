import copy
import json
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
    engine: str

    def __init__(self, rep: str, **args):
        self.rep = rep
        self.__dict__.update(args)
        self.properties = copy.deepcopy(self.__dict__)

    def compute_reps(self, mols: List[str],
                     verbose: Optional[bool] = False,
                     batch_size: Optional[int] = 12) -> Union[np.ndarray, List[np.ndarray]]:
        batches = batched(mols, batch_size)
        out = []

        if verbose:
            pbar = tqdm(batches)
        else:
            pbar = batches

        for batch in pbar:
            self._preprocess_batch(batch)
            out.extend(self._rep_batch(batch))
            if 'average_pooling' in self.__dict__:
                if not self.__dict__['average_pooling']:
                    return out
    
        return np.stack(out)

    def dim(self) -> int:
        raise NotImplementedError

    def _rep_batch(self, batch: List[str]) -> np.ndarray:
        raise NotImplementedError

    def _preprocess_batch(self, batch: List[str]) -> List[str]:
        raise NotImplementedError

    def save(self, filename: str):
        json.dump(open(filename, 'w'), self.properties)

    def __str__(self) -> str:
        return str(self.properties)
