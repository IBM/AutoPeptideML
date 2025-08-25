RESIDUES = {
    'V':'VAL', 'I':'ILE', 'L':'LEU', 'E':'GLU', 'Q':'GLN', 
    'D':'ASP', 'N':'ASN', 'H':'HIS', 'W':'TRP', 'F':'PHE', 
    'Y':'TYR', 'R':'ARG', 'K':'LYS', 'S':'SER', 'T':'THR',
    'M':'MET', 'A':'ALA', 'G':'GLY', 'P':'PRO', 'C':'CYS'
}

def is_canonical(sequence: str):
    if not (len(sequence) > 0):
        return False
    for char in sequence:
        if char not in RESIDUES:
            return False
    return True
