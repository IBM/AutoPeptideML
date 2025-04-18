from typing import *

from .pipeline import BaseElement


RESIDUES = {
    'V': 'VAL', 'I': 'ILE', 'L': 'LEU', 'E': 'GLU', 'Q': 'GLN',
    'D': 'ASP', 'N': 'ASN', 'H': 'HIS', 'W': 'TRP', 'F': 'PHE',
    'Y': 'TYR', 'R': 'ARG', 'K': 'LYS', 'S': 'SER', 'T': 'THR',
    'M': 'MET', 'A': 'ALA', 'G': 'GLY', 'P': 'PRO', 'C': 'CYS'
}


def is_canonical(sequence: str):
    if not (len(sequence) > 0):
        return False
    for char in sequence:
        if char not in RESIDUES:
            return False
    return True


class CanonicalCleaner(BaseElement):
    """
    Class `CanonicalCleaner` is a molecular processing element that standardizes molecular representations 
    by replacing non-canonical residues with a specified substitution character.

    Attributes:
        :type name: str
        :param name: The name of the element. Default is `'canonical-cleaner'`.

        :type substitution: str
        :param substitution: The character used to replace non-canonical residues. Default is `'X'`.
    """
    name = 'canonical-cleaner'

    def __init__(self, substitution: str = 'X'):
        """
        Initializes the `CanonicalCleaner` with a substitution character.

        :type substitution: str
          :param substitution: The character used to replace non-canonical residues. Default is `'X'`.

        :rtype: None
        """
        self.sub = substitution
        self.properties = {'substitution': substitution}

    def _single_call(self, mol: str) -> str:
        """
        Cleans a single molecular representation by replacing non-canonical residues.

        :type mol: str
          :param mol: A single molecular representation (e.g., a sequence of residues).

        :rtype: str
          :return: The cleaned molecular representation with non-canonical residues replaced by the substitution.
        """
        return ''.join([c if c in RESIDUES else self.sub for c in mol])


class CanonicalFilter(BaseElement):
    """
    Class `CanonicalFilter` is a molecular processing element that filters molecular representations based on 
    their canonicality. It can either keep or discard canonical molecules based on the configuration.

    Attributes:
        :type name: str
        :param name: The name of the element. Default is `'canonical-filter'`.

        :type keep_canonical: bool
        :param keep_canonical: Determines whether to keep canonical molecules (`True`) or discard them (`False`).
                                Default is `True`.
    """
    name = 'canonical-filter'

    def __init__(self, keep_canonical: bool = True):
        """
        Initializes the `CanonicalFilter` with a configuration to keep or discard canonical molecules.

        :type keep_canonical: bool
          :param keep_canonical: Determines whether to keep canonical molecules (`True`) or discard them (`False`).
                                 Default is `True`.

        :rtype: None
        """
        self.keep_canonical = keep_canonical

    def _single_call(self, mol: str) -> Union[str, None]:
        """
        Filters a single molecular representation based on its canonicality.

        :type mol: str
          :param mol: A single molecular representation (e.g., a sequence of residues).

        :rtype: Union[str, None]
          :return: The molecule if it meets the canonicality condition, or `None` otherwise.
        """
        if not (len(mol) > 0):
            return None
        if ((is_canonical(mol) and self.keep_canonical) or
           (not is_canonical(mol) and not self.keep_canonical)):
            return mol
        else:
            return None
