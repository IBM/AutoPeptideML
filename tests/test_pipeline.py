import pytest

from autopeptideml.pipeline import Pipeline
from autopeptideml.pipeline import (Pipeline, CanonicalCleaner, CanonicalFilter,
                                    SequenceToSMILES, FilterSMILES, SmilesToSequence,
                                    SmilesToBILN)


def test_canonical_filter():
    seqs = ['AAACCTWSFB', 'AAACCTWF', 'AAACCTWaF']
    pipe = Pipeline([CanonicalFilter()])
    seqs_out = pipe(seqs)
    assert seqs_out == ['AAACCTWF']


@pytest.mark.parametrize(
    "smiles, seq_out",
    [
        ('c1ccc(C)cc1C[C@H](N)C(=O)N1CCC[C@H]1C(=O)O[C@@H]C(=O)O[C@H](O)C', 'FP'),
        ('N[C@@H](C)C(=O)N[C@@H](CS)C(=O)N[C@@H](C(O)C)C(=O)O', "ACT"),
        ('C(c1cccc(C)c1)[C@H](N)C(=O)N[C@H](C(=O)O)c1c2ccccc2[nH]c1(Br)', "FW"),
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
        ('C(c1cccc(C)c1)[C@H](N)C(=O)N[C@H](C(=O)O)c1c2ccccc2[nH]c1(Br)', "X1186-X1772"),
    ]
)
def test_to_biln(smiles, seq_out):
    pipe = Pipeline([SmilesToBILN()])
    seq_pred = pipe(smiles)
    assert seq_pred == seq_out


def test_canonical_cleaner():
    seqs = ['AAACCTWSFB', 'AAACCTWF', 'AAACCTWaF']
    pipe = Pipeline([CanonicalCleaner()])
    seqs_out = pipe(seqs)
    assert seqs_out == ['AAACCTWSFX', 'AAACCTWF', 'AAACCTWXF']


def test_to_smiles():
    seqs = ['BRTWSF', 'ARTWF', 'aRTWSF', 'C[C@H](N)C(=O)N[C@@H](CCCNC(=N)N)C(=O)N[C@H]']
    pipe1 = Pipeline([FilterSMILES()], name='pipe_smiles')
    pipe2 = Pipeline([FilterSMILES(keep_smiles=False),
                     CanonicalCleaner(substitution='G'),
                     SequenceToSMILES()], name='pipe_seqs')
    pipe = Pipeline([pipe1, pipe2], name='main_pipeline', aggregate=True)
    seqs_out = pipe(seqs, verbose=True)
    assert seqs_out == [
        'C[C@H](N)C(=O)N[C@@H](CCCNC(=N)N)C(=O)N[C@H]',
        'C[C@@H](O)[C@H](NC(=O)[C@H](CCCNC(=N)N)NC(=O)CN)C(=O)N[C@@H](Cc1c[nH]c2ccccc12)C(=O)N[C@@H](CO)C(=O)N[C@@H](Cc1ccccc1)C(=O)O',
        'C[C@H](N)C(=O)N[C@@H](CCCNC(=N)N)C(=O)N[C@H](C(=O)N[C@@H](Cc1c[nH]c2ccccc12)C(=O)N[C@@H](Cc1ccccc1)C(=O)O)[C@@H](C)O',
        'C[C@@H](O)[C@H](NC(=O)[C@H](CCCNC(=N)N)NC(=O)CN)C(=O)N[C@@H](Cc1c[nH]c2ccccc12)C(=O)N[C@@H](CO)C(=O)N[C@@H](Cc1ccccc1)C(=O)O'
    ]
