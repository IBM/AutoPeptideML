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
        ('NCC(=O)SCCCC[C@H]C(=O)N(CC(=O)O)c1ccccc1', 'XG'),
        ('C[C@H](N)C(=O)N[C@@H](CCCNC(=N)N)C(=O)N[C@H]', "AAACCTWSFB"),
        ('C[C@@H](O)[C@H](NC(=O)[C@H](CCCNC(=N)N)NC(=O)CN)C(=O)N[C@@H](Cc1c[nH]c2ccccc12)C(=O)N[C@@H](CO)C(=O)N[C@@H](Cc1ccccc1)C(=O)O', "AAACCTWF"),
    ]
)
def test_to_sequence(smiles, seq_out):
    pipe = Pipeline([SmilesToSequence()])
    seq_pred = pipe(smiles)
    assert seq_pred == seq_out


@pytest.mark.parametrize(
    "smiles, seq_out",
    [
        ('CC(=O)SCCCC[C@H](N)C(=O)N(CC=O)c1ccccc1', 'X2484-X414'),
        ('C[C@H](N)C(=O)N[C@@H](CCCNC(=N)N)C(=O)N[C@H]', "A-A-A-C-C-T-W-S-F-B"),
        ('C[C@@H](O)[C@H](NC(=O)[C@H](CCCNC(=N)N)NC(=O)CN)C(=O)N[C@@H](Cc1c[nH]c2ccccc12)C(=O)N[C@@H](CO)C(=O)N[C@@H](Cc1ccccc1)C(=O)O', "A-A-A-C-C-T-W-F"),
    ]
)
def test_to_biln(smiles, seq_out):
    pipe = Pipeline([SmilesToBILN])
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
