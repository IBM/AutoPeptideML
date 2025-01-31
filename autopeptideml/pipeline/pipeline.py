import json
import yaml
from typing import *

from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
from tqdm import tqdm


class BaseElement:
    """
    Class `BaseElement` provides a foundation for implementing molecular processing elements.
    It supports both single and parallel processing of molecular data, making it suitable for operations
    that can be applied to molecular representations such as SMILES strings.

    Attributes:
        :type name: str
        :param name: The name of the processing element.

        :type properties: Dict[str, Any]
        :param properties: A dictionary of additional properties for the processing element.
                            Default is an empty dictionary.
    """
    name: str
    properties: Dict[str, Any] = {}

    def __call__(self, mol: Union[str, List[str]],
                 n_jobs: int = cpu_count(),
                 verbose: bool = False) -> Union[str, List[str]]:
        """
        Processes molecular data, either as a single molecule or a list of molecules.
        Automatically selects single or parallel processing based on the input type.

        :type mol: Union[str, List[str]]
          :param mol: A single molecular representation (e.g., SMILES string) or a list of such representations.

        :type n_jobs: int
          :param n_jobs: The number of parallel jobs to use for processing. Default is the number of CPU cores.

        :type verbose: bool
          :param verbose: Enables verbose output if set to `True`, displaying a progress bar for parallel processing.
                          Default is `False`.

        :rtype: Union[str, List[str]]
          :return: The processed molecular representation(s).
        """
        if isinstance(mol, str):
            return self._single_call(mol)
        elif len(mol) == 0:
            return mol
        else:
            return self._parallel_call(mol, n_jobs=n_jobs,
                                       verbose=verbose)

    def _single_call(self, mol: str) -> str:
        """
        Processes a single molecular representation. 
        Must be implemented in a subclass.

        :type mol: str
          :param mol: A single molecular representation (e.g., SMILES string).

        :rtype: str
          :return: The processed molecular representation.

        :raises NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError

    def _clean(self, mol: List[Optional[str]]) -> List[str]:
        """
        Cleans the processed molecular data by removing `None` values.

        :type mol: List[Optional[str]]
          :param mol: A list of processed molecular representations, some of which may be `None`.

        :rtype: List[str]
          :return: A cleaned list of molecular representations without `None` values.
        """
        return [m for m in mol if m is not None]

    def _parallel_call(self, mol: List[str], n_jobs: int,
                       verbose: bool) -> List[str]:
        """
        Processes a list of molecular representations in parallel using a thread pool.

        :type mol: List[str]
          :param mol: A list of molecular representations (e.g., SMILES strings) to process.

        :type n_jobs: int
          :param n_jobs: The number of parallel jobs to use for processing. If set to `1`, processes sequentially.

        :type verbose: bool
          :param verbose: Enables verbose output if set to `True`, displaying a progress bar for parallel processing.

        :rtype: List[str]
          :return: A list of processed molecular representations.

        :raises RuntimeError: If any parallel job raises an exception.
        """
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
    """
    Class `Pipeline` represents a sequence of molecular processing steps, where each step is defined by an element 
    (`BaseElement` or another `Pipeline`). The pipeline can process molecular data sequentially and optionally 
    aggregate results across all steps.

    Attributes:
        :type elements: Union[List[BaseElement], List[Pipeline]]
        :param elements: A list of `BaseElement` or `Pipeline` instances that define the processing steps.

        :type name: str
        :param name: The name of the pipeline. Default is `'pipeline'`.

        :type aggregate: bool
        :param aggregate: If `True`, the pipeline aggregates results from all steps. 
                            If `False`, the results of one step are passed to the next. Default is `False`.
    """
    def __init__(self, elements: Union[List[BaseElement], List["Pipeline"]],
                 name: str = 'pipeline',
                 aggregate: bool = False):
        """
        Initializes the pipeline with a sequence of processing elements and configuration.

        :type elements: Union[List[BaseElement], List[Pipeline]]
          :param elements: A list of `BaseElement` or `Pipeline` instances to define the processing steps.

        :type name: str
          :param name: The name of the pipeline. Default is `'pipeline'`.

        :type aggregate: bool
          :param aggregate: If `True`, results from all steps are aggregated. If `False`, results of one step 
                            are passed to the next. Default is `False`.

        :rtype: None
        """
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
        """
        Returns a JSON string representation of the pipeline's properties.

        :rtype: str
          :return: A JSON string representing the pipeline's configuration and properties.
        """
        return json.dumps(self.properties, indent=3)

    def __call__(self, mols: List[str],
                 n_jobs: int = cpu_count(),
                 verbose: bool = False):
        """
        Processes a list of molecular representations using the pipeline.

        :type mols: List[str]
          :param mols: A list of molecular representations (e.g., SMILES strings) to process.

        :type n_jobs: int
          :param n_jobs: The number of parallel jobs to use for processing. Default is the number of CPU cores.

        :type verbose: bool
          :param verbose: Enables verbose output if set to `True`. Displays progress and step information.

        :rtype: Union[List[str], List[List[str]]]
          :return: Processed molecular data. If `aggregate` is `True`, returns aggregated results from all steps.
                   Otherwise, returns the final processed molecular data.
        """
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
        """
        Saves the pipeline's properties to a YAML file.

        :type filename: str
          :param filename: The name of the file to save the pipeline's properties.

        :rtype: None
        """
        yaml.safe_dump(self.properties, open(filename, 'w'))

    @classmethod
    def load(self, filename: str, element_registry: dict):
        """
        Loads a pipeline from a YAML file and reconstructs its elements using a registry.

        :type filename: str
          :param filename: The name of the file containing the saved pipeline properties.

        :type element_registry: Dict[str, Callable]
          :param element_registry: A dictionary mapping element names to their constructor functions.

        :rtype: Pipeline
          :return: A reconstructed `Pipeline` instance based on the saved properties.
        """
        self.properties = json.load(open(filename))
        elements = []
        for e, e_prop in self.config.items():
            elements.append(element_registry[e](**e_prop))
        return Pipeline(elements)
