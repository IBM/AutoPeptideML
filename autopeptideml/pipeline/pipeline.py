import json
import yaml
from typing import *

from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
from tqdm import tqdm


class BaseElement:
    name: str
    properties: Dict[str, Any] = {}

    def __call__(self, mol: Union[str, List[str]],
                 n_jobs: int = cpu_count(),
                 verbose: bool = False) -> Union[str, List[str]]:
        if isinstance(mol, str):
            return self._single_call(mol)
        else:
            return self._parallel_call(mol, n_jobs=n_jobs,
                                       verbose=verbose)

    def _single_call(self, mol: str) -> str:
        raise NotImplementedError

    def _clean(self, mol: List[Optional[str]]) -> List[str]:
        return [m for m in mol if m is not None]

    def _parallel_call(self, mol: List[str], n_jobs: int,
                       verbose: bool) -> List[str]:
        if n_jobs > 1:
            jobs, out = [], []
            with ThreadPoolExecutor(n_jobs) as exec:
                for item in mol:
                    job = exec.submit(self._single_call, item)
                    jobs.append(job)

                if verbose:
                    pbar = tqdm(jobs, unit_scale=True)
                else:
                    pbar = jobs

                for job in pbar:
                    if job.exception() is not None:
                        raise RuntimeError(job.exception())
                    out.append(job.result())
        else:
            out = []
            for item in mol:
                out.append(self._single_call(item))
        return self._clean(out)


class Pipeline:
    def __init__(self, elements: Union[List[BaseElement], List["Pipeline"]],
                 name: str = 'pipeline',
                 aggregate: bool = False):
        self.elements = elements
        self.name = name
        self.properties = {name: {
                'name': name,
                'aggregate': aggregate,
                'elements': [{e.name: e.properties} for e in elements]}
        }
        self.properties['aggregate'] = aggregate
        self.aggregate = aggregate

    def __str__(self) -> str:
        return json.dumps(self.properties)

    def __call__(self, mols: List[str],
                 n_jobs: int = cpu_count(),
                 verbose: bool = False):
        original_mols = mols
        aggregation = []
        for idx, e in enumerate(self.elements):
            if verbose:
                print(f"Executing preprocessing step {idx+1} of",
                      f"{len(self.elements)}: {e.name}")
            if self.aggregate:
                mols = e(original_mols, n_jobs=n_jobs, verbose=verbose)
                aggregation.extend(mols)
            else:
                mols = e(mols, n_jobs=n_jobs, verbose=verbose)

            if verbose and not self.aggregate:
                print(f'Total molecules removed: {len(original_mols)-len(mols):,}')

        if self.aggregate:
            return aggregation
        else:
            return mols

    def save(self, filename: str):
        yaml.safe_dump(self.properties, open(filename, 'w'))

    @classmethod
    def load(self, filename: str, element_registry):
        self.properties = json.load(open(filename))
        elements = []
        for e, e_prop in self.config.items():
            elements.append(element_registry[e](**e_prop))
        return Pipeline(elements)
