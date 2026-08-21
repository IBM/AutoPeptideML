"""
Tests for RepEngineFP (rdkit-backed) and RepEngineSkfp (scikit-fingerprints-backed).

A small set of SMILES for a tri-peptide Ala-Arg-Gly is used throughout so that
every test stays fast and self-contained.
"""
import json

import numpy as np
import pytest

from autopeptideml.reps.fps import RepEngineFP, RepEngineSkfp

# Two valid peptide SMILES used across all tests.
SMILES = [
    'C[C@H](N)C(=O)N[C@@H](CCCNC(=N)N)C(=O)NCC(=O)O',   # Ala-Arg-Gly
    'N[C@@H](Cc1ccccc1)C(=O)N[C@@H](CS)C(=O)O',           # Phe-Cys
]


# ---------------------------------------------------------------------------
# RepEngineFP — rdkit-backed
# ---------------------------------------------------------------------------

class TestRepEngineFP:
    def test_ecfp_shape(self):
        re = RepEngineFP('ecfp', nbits=1024, radius=2)
        out = re.compute_reps(SMILES, batch_size=2)
        assert out.shape == (2, 1024)

    def test_ecfp_dim(self):
        re = RepEngineFP('ecfp', nbits=512, radius=3)
        assert re.dim() == 512

    def test_ecfp_count_shape(self):
        re = RepEngineFP('ecfp-count', nbits=512, radius=2)
        out = re.compute_reps(SMILES, batch_size=2)
        assert out.shape == (2, 512)
        assert re.dim() == 512

    def test_fcfp_shape(self):
        re = RepEngineFP('fcfp', nbits=256, radius=2)
        out = re.compute_reps(SMILES, batch_size=2)
        assert out.shape == (2, 256)

    def test_output_dtype_is_numeric(self):
        re = RepEngineFP('ecfp', nbits=128, radius=2)
        out = re.compute_reps(SMILES, batch_size=2)
        assert np.issubdtype(out.dtype, np.number)

    def test_output_is_binary(self):
        """Bit (non-count) fingerprints must only contain 0 and 1."""
        re = RepEngineFP('ecfp', nbits=1024, radius=2)
        out = re.compute_reps(SMILES, batch_size=2)
        assert set(np.unique(out)).issubset({0, 1})

    def test_deterministic(self):
        re = RepEngineFP('ecfp', nbits=1024, radius=2)
        a = re.compute_reps(SMILES, batch_size=2)
        b = re.compute_reps(SMILES, batch_size=2)
        np.testing.assert_array_equal(a, b)

    def test_different_smiles_differ(self):
        re = RepEngineFP('ecfp', nbits=1024, radius=2)
        out = re.compute_reps(SMILES, batch_size=2)
        assert not np.array_equal(out[0], out[1])

    def test_serialisation(self):
        re = RepEngineFP('ecfp', nbits=256, radius=4)
        d = json.loads(str(re))
        assert d == {'rep': 'ecfp', 'nbits': 256, 'radius': 4}

    def test_invalid_rep_raises(self):
        with pytest.raises(NotImplementedError):
            RepEngineFP('nonexistent', nbits=128, radius=2)

    def test_invalid_smiles_returns_zeros(self):
        """An unparseable SMILES must produce an all-zero fingerprint, not raise."""
        re = RepEngineFP('ecfp', nbits=512, radius=2)
        out = re.compute_reps(['not_a_smiles'], batch_size=1)
        assert out.shape == (1, 512)
        assert out.sum() == 0


# ---------------------------------------------------------------------------
# RepEngineSkfp — scikit-fingerprints-backed
# ---------------------------------------------------------------------------

skfp = pytest.importorskip('skfp', reason='scikit-fingerprints not installed')


class TestRepEngineSkfp:
    def test_ecfp_shape(self):
        re = RepEngineSkfp('ecfp', fp_size=1024, radius=2)
        out = re.compute_reps(SMILES, batch_size=2)
        assert out.shape == (2, 1024)

    def test_ecfp_dim(self):
        re = RepEngineSkfp('ecfp', fp_size=512, radius=2)
        assert re.dim() == 512

    def test_maccs_shape_and_dim(self):
        """MACCS has a fixed size of 166 bits."""
        re = RepEngineSkfp('maccs')
        out = re.compute_reps(SMILES, batch_size=2)
        assert re.dim() == 166
        assert out.shape == (2, 166)

    def test_atompair_shape(self):
        re = RepEngineSkfp('atompair', fp_size=512)
        out = re.compute_reps(SMILES, batch_size=2)
        assert out.shape == (2, 512)

    def test_topologicaltorsion_shape(self):
        re = RepEngineSkfp('topologicaltorsion', fp_size=256)
        out = re.compute_reps(SMILES, batch_size=2)
        assert out.shape == (2, 256)

    def test_count_variant(self):
        """count=True fingerprints may contain values > 1."""
        re = RepEngineSkfp('ecfp', fp_size=512, radius=2, count=True)
        out = re.compute_reps(SMILES, batch_size=2)
        assert out.shape == (2, 512)
        assert re.dim() == 512

    def test_output_is_binary_for_bit_fp(self):
        re = RepEngineSkfp('maccs')
        out = re.compute_reps(SMILES, batch_size=2)
        assert set(np.unique(out)).issubset({0, 1})

    def test_deterministic(self):
        re = RepEngineSkfp('ecfp', fp_size=1024, radius=2)
        a = re.compute_reps(SMILES, batch_size=2)
        b = re.compute_reps(SMILES, batch_size=2)
        np.testing.assert_array_equal(a, b)

    def test_different_smiles_differ(self):
        re = RepEngineSkfp('ecfp', fp_size=1024, radius=2)
        out = re.compute_reps(SMILES, batch_size=2)
        assert not np.array_equal(out[0], out[1])

    def test_name(self):
        re = RepEngineSkfp('maccs')
        assert re.name == 'skfp-maccs'

    def test_invalid_rep_raises(self):
        with pytest.raises(NotImplementedError):
            RepEngineSkfp('nonexistent_fp')
