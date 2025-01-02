from typing import *

from ..pipeline import BaseElement


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
    name = 'canonical-cleaner'

    def __init__(self, substitution: str = 'X'):
        self.sub = substitution
        self.properties = {'substitution': substitution}

    def _single_call(self, mol: str) -> str:
        return ''.join([c if c in RESIDUES else self.sub for c in mol])


class CanonicalFilter(BaseElement):
    name = 'canonical-filter'

    def __init__(self):
        return

    def _single_call(self, mol: str) -> Union[str, None]:
        if not (len(mol) > 0):
            return None
        for char in mol:
            if char not in RESIDUES:
                return None
        return mol
