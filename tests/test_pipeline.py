import pytest

from autopeptideml.pipeline import Pipeline
from autopeptideml.pipeline import (Pipeline, CanonicalCleaner, CanonicalFilter)
from autopeptideml.pipeline.smiles import (
    SmilesToSequence, CanonicalizeSmiles,
    FilterSmiles, BilnToSmiles, SmilesToBiln,
    SequenceToSmiles
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
        ('N[C@H](Cc1ccc(C(F)(F)F)cc1)C(=O)N[C@@H](CC(=O)O)C(=O)COC(=O)c1c(C(F)(F)F)cccc1C(F)(F)F', 'FD'),
        ('N[C@@H](C)C(=O)N[C@@H](CS)C(=O)N[C@@H](C(O)C)C(=O)O', "ACT"),
        ('Cc1cccc(C[C@H](N)C(=O)N[C@@H](Cc2c(Br)[nH]c3ccccc23)C(=O)O)c1', "FW"),
        ('CC(C)C[C@H](NC(=O)[C@@H](NC(=O)[C@H](Cc1ccccc1)NC(=O)[C@H](C)NC(=O)[C@H](CC(C)C)NC(=O)[C@@H](N)Cc1c[nH]c2ccccc12)C(C)C)C(=O)N[C@H](C(=O)O)C(C)C', 'WLAFVLV'),
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
        ('Nc1nc2c(ncn2C2OC(COP(=O)(O)O)C(O)C2O)c(=O)[nH]1', "X"),
        ('N[C@H](Cc1ccc(C(F)(F)F)cc1)C(=O)N[C@@H](CC(=O)O)C(=O)COC(=O)c1c(C(F)(F)F)cccc1C(F)(F)F', 'X933-X2777'),
        ('CC(=O)N[C@@H](Cc1c[nH]c2ccccc12)C(=O)N[C@@H](Cc1ccc(O)cc1)C(=O)N[C@@H](CS)C(=O)NCC(N)=O', 'am-G-C-Y-W-ac')
    ]
)
def test_to_biln(smiles, seq_out):
    pipe = Pipeline([SmilesToBiln()])
    seq_pred = pipe(smiles)
    assert seq_pred == seq_out


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
    seqs = ['BRTWSF', 'ARTWF', 'aRTWSF', 'C[C@H](N)C(=O)N[C@@H](CCCNC(=N)N)C(=O)N[C@H]']
    pipe1 = Pipeline([FilterSmiles()], name='pipe_smiles')
    pipe2 = Pipeline([FilterSmiles(keep_smiles=False),
                     CanonicalCleaner(substitution='G'),
                     SequenceToSmiles()], name='pipe_seqs')
    pipe = Pipeline([pipe1, pipe2], name='main_pipeline', aggregate=True)
    seqs_out = pipe(seqs, verbose=True)
    assert seqs_out == [
        'C[C@H](N)C(=O)N[C@@H](CCCNC(=N)N)C(=O)N[C@H]',
        'C[C@@H](O)[C@H](NC(=O)[C@H](CCCNC(=N)N)NC(=O)CN)C(=O)N[C@@H](Cc1c[nH]c2ccccc12)C(=O)N[C@@H](CO)C(=O)N[C@@H](Cc1ccccc1)C(=O)O',
        'C[C@H](N)C(=O)N[C@@H](CCCNC(=N)N)C(=O)N[C@H](C(=O)N[C@@H](Cc1c[nH]c2ccccc12)C(=O)N[C@@H](Cc1ccccc1)C(=O)O)[C@@H](C)O',
        'C[C@@H](O)[C@H](NC(=O)[C@H](CCCNC(=N)N)NC(=O)CN)C(=O)N[C@@H](Cc1c[nH]c2ccccc12)C(=O)N[C@@H](CO)C(=O)N[C@@H](Cc1ccccc1)C(=O)O'
    ]


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
