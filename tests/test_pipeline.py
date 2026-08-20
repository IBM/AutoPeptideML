import pytest

from autopeptideml.pipeline import Pipeline, CanonicalCleaner, CanonicalFilter
from autopeptideml.pipeline.smiles import (
    SmilesToSequence, CanonicalizeSmiles,
    FilterSmiles, BilnToSmiles, SmilesToBiln,
    SequenceToSmiles,
)
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import DataStructs


def test_canonical_filter():
    seqs = ['AAACCTWSFB', 'AAACCTWF', 'AAACCTWaF']
    pipe = Pipeline([CanonicalFilter()])
    seqs_out = pipe(seqs)
    assert seqs_out == ['AAACCTWF']


@pytest.mark.parametrize(
    "smiles, seq_out",
    [
        ('N[C@H](Cc1ccc(C(F)(F)F)cc1)C(=O)N[C@@H](CC(=O)O)C(=O)COC(=O)c1c(C(F)(F)F)cccc1C(F)(F)F', 'FX'),
        ('N[C@@H](C)C(=O)N[C@@H](CS)C(=O)N[C@@H](C(O)C)C(=O)O', "ACT"),
        ('Cc1cccc(C[C@H](N)C(=O)N[C@@H](Cc2c(Br)[nH]c3ccccc23)C(=O)O)c1', "FW"),
        ('CC(C)C[C@H](NC(=O)[C@@H](NC(=O)[C@H](Cc1ccccc1)NC(=O)[C@H](C)NC(=O)[C@H](CC(C)C)NC(=O)[C@@H](N)Cc1c[nH]c2ccccc12)C(C)C)C(=O)N[C@H](C(=O)O)C(C)C', 'WLAFVLV'),
        ('CC(C)C[C@H](NC(=O)[C@@H](N)CCC(N)=O)C(=O)N1CCC[C@H]1C(=O)N[C@@H](Cc1ccccc1)C(=O)N1CCC[C@H]1C(=O)N[C@@H](CCC(N)=O)C(=O)N[C@@H](CCC(N)=O)C(=O)N1CCC[C@H]1C(=O)N[C@@H](CCC(N)=O)C(=O)O', 'QLPFPQQPQ'),
        ('CC(C)C[C@H](NC(=O)[C@@H](N)CC(C)C)C(=O)N[C@@H](C)C(=O)N[C@@H](Cc1c[nH]cn1)C(=O)O', 'LLAH'),
        ('N=C(N)NCCC[C@H](NC(=O)[C@H](CCCNC(=N)N)NC(=O)[C@H](CCCCN)NC(=O)[C@@H](N)Cc1ccccc1)C(=O)O', 'FKRR'),
        ('CC(=O)N[C@@H](Cc1c[nH]c2ccccc12)C(=O)N[C@@H](Cc1ccc(O)cc1)C(=O)N[C@@H](CS)C(=O)NCC(N)=O', 'WYCG'),
        ('CC(C)[C@H](NC(=O)[C@H](Cc1ccc(O)cc1)NC(=O)[C@H](CO)NC(=O)[C@@H]1CCCN1)C(=O)O', 'PSYV')
    ]
)
def test_to_sequence(smiles, seq_out):
    pipe = Pipeline([SmilesToSequence()])
    seq_pred = pipe(smiles)
    assert seq_pred == seq_out


@pytest.mark.parametrize(
    "smiles, seq_out",
    [
        ('N[C@@H](C)C(=O)N[C@@H](CS)C(=O)N[C@@H](C(O)C)C(=O)O', "A-C-T"),
        ('Cc1cccc(C[C@H](N)C(=O)N[C@@H](Cc2c(Br)[nH]c3ccccc23)C(=O)O)c1', "X1186-X1772"),
        ('N[C@H](Cc1ccc(C(F)(F)F)cc1)C(=O)N[C@@H](CC(=O)O)C(=O)COC(=O)c1c(C(F)(F)F)cccc1C(F)(F)F', 'X933-X'),
        ('CC(=O)N[C@@H](Cc1c[nH]c2ccccc12)C(=O)N[C@@H](Cc1ccc(O)cc1)C(=O)N[C@@H](CS)C(=O)NCC(N)=O', 'Wac-Y-C-Gam'),
        ('CC[C@H](C)[C@@H]1NC(=O)[C@H](CCCNC(=N)N)NC(=O)[C@@H]2CSCCC(=O)N3CN(CN(C3)C(=O)CCSC[C@@H](C(=O)N[C@@H](C)C(N)=O)NC(=O)[C@H](CC(C)C)NC(=O)[C@H](CCCCN)NC1=O)C(=O)CCSC[C@H](NC(=O)[C@H](C)N)C(=O)NCC(=O)N[C@@H](CCCNC(=N)N)C(=O)N[C@@H](CCC(=O)O)C(=O)N[C@@H](CCC(=O)O)C(=O)N[C@@H](CC(C)C)C(=O)N1CCC[C@H]1C(=O)N2', 'A-X248(4,2)-G-R-E-E-L-P-X248(3,2)-R-I-K-L-X248(5,2)-Aam.X(3,1)(4,1)(5,1)')
    ]
)
def test_to_biln(smiles, seq_out):
    pipe = Pipeline([SmilesToBiln(human_readable=True)])
    seq_pred = pipe(smiles)
    assert seq_pred == seq_out


def test_to_biln_non_peptide_returns_x():
    """Non-peptide molecules return 'X' when handle_errors=True."""
    pipe = Pipeline([SmilesToBiln(human_readable=True, handle_errors=True)])
    result = pipe('Nc1nc2c(ncn2C2OC(COP(=O)(O)O)C(O)C2O)c(=O)[nH]1')
    assert result == 'X'


@pytest.mark.parametrize(
    "biln, smiles_out",
    [
        ("A-C-T", 'N[C@@H](C)C(=O)N[C@@H](CS)C(=O)N[C@@H](C(O)C)C(=O)O'),
        ("X1186-X1772", 'Cc1cccc(C[C@H](N)C(=O)N[C@@H](Cc2c(Br)[nH]c3ccccc23)C(=O)O)c1'),
        ("F-P", "N[C@@H](CC1=CC=CC=C1)C(=O)N2CCC[C@H]2C(=O)O"),
        ("X933-X2777", "N[C@H](Cc1ccc(C(F)(F)F)cc1)C(=O)N[C@@H](CC(=O)O)C(=O)COC(=O)c1c(C(F)(F)F)cccc1C(F)(F)F")
    ]
)
def test_from_biln(biln, smiles_out):
    pipe = Pipeline([BilnToSmiles(), CanonicalizeSmiles()])
    pipe2 = Pipeline([CanonicalizeSmiles()])
    seq_pred = pipe(biln)
    assert _check_smiles(seq_pred, pipe2(smiles_out))


def test_canonical_cleaner():
    seqs = ['AAACCTWSFB', 'AAACCTWF', 'AAACCTWaF']
    pipe = Pipeline([CanonicalCleaner()])
    seqs_out = pipe(seqs)
    assert seqs_out == ['AAACCTWSFX', 'AAACCTWF', 'AAACCTWXF']


def test_to_smiles():
    # The fourth input is a truncated/invalid SMILES: it passes the is_smiles()
    # heuristic but RDKit cannot parse it. FilterSmiles(keep_smiles=True)
    # forwards it unchanged; CanonicalizeSmiles would crash on it — but it is
    # only touched by pipe1 (the SMILES-keeping branch), where it is never
    # canonicalized.  The correct expected output therefore excludes it: pipe1
    # forwards only the valid-SMILES item (none here, since the truncated string
    # has no matching molecule), and pipe2 converts the three peptide sequences.
    #
    # Sequences: BRTWSF, ARTWF, aRTWSF
    #   - B is non-canonical → replaced by G (CanonicalCleaner substitution='G')
    #   - a (lower-case) → replaced by G
    #   pipe2 output (3 SMILES): GRTWSF, ARTWF, GRTWSF
    # pipe1 keeps the invalid SMILES string as-is (1 item).
    # aggregate=True → pipe1_output + pipe2_output = 4 items total.
    seqs = ['BRTWSF', 'ARTWF', 'aRTWSF', 'C[C@H](N)C(=O)N[C@@H](CCCNC(=N)N)C(=O)N[C@H]']
    pipe1 = Pipeline([FilterSmiles()], name='pipe_smiles')
    pipe2 = Pipeline([FilterSmiles(keep_smiles=False),
                      CanonicalCleaner(substitution='G'),
                      SequenceToSmiles()], name='pipe_seqs')
    pipe = Pipeline([pipe1, pipe2], name='main_pipeline', aggregate=True)
    seqs_out = pipe(seqs, verbose=True)

    # pipe1 result: only the one item that looks like SMILES
    assert len(seqs_out) == 4

    # pipe1 bucket: the invalid-SMILES passthrough (index 0)
    assert seqs_out[0] == 'C[C@H](N)C(=O)N[C@@H](CCCNC(=N)N)C(=O)N[C@H]'

    # pipe2 bucket: three sequence→SMILES conversions (indices 1-3)
    # Use fingerprint comparison so the assertion is RDKit-version-agnostic.
    expected_seqs = ['GRTWSF', 'ARTWF', 'GRTWSF']
    for smiles_pred, seq in zip(seqs_out[1:], expected_seqs):
        expected_smiles = Pipeline([SequenceToSmiles()])(seq)
        assert _check_smiles(smiles_pred, expected_smiles), (
            f"SMILES mismatch for {seq!r}: got {smiles_pred!r}"
        )

def _check_smiles(smiles1: str, smiles2: str) -> bool:
    fpgen = rdFingerprintGenerator.GetMorganGenerator(
        radius=4, fpSize=2048, includeChirality=False,
        countSimulation=True
    )
    mol1 = Chem.MolFromSmiles(smiles1, sanitize=True)
    mol2 = Chem.MolFromSmiles(smiles2, sanitize=True)
    mol1 = Chem.RemoveAllHs(mol1, sanitize=True)
    mol2 = Chem.RemoveAllHs(mol2, sanitize=True)
    fp1 = fpgen.GetFingerprint(mol1)
    fp2 = fpgen.GetFingerprint(mol2)
    smiles_similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
    return smiles_similarity == 1.0
