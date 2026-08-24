"""
Integration tests for AutoPeptideML.

These tests exercise the full pipeline on synthetic in-memory data so they
run without network access or large model downloads.  They are intentionally
kept fast by:
  - Using only ECFP fingerprints (no LM downloads)
  - Limiting n_trials to a small number
  - Using 'random' split strategy to skip the Hestia similarity computation
"""
import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import pytest

from autopeptideml import AutoPeptideML


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CANONICAL_SEQS = [
    "ACDEFGHIKLM",
    "NOPQRSTVWY",   # O and U trigger non-canonical handling
    "AAAAACCCCC",
    "GGGGGTTTTT",
    "CCCCCAAAAA",
    "MMMMMLLLLL",
    "FFFFF",
    "YYYYY",
    "WWWWW",
    "RRRRR",
]

# 20 labelled peptides (10 positive, 10 negative)
LABELLED_SEQS = [
    "ACGGF", "LKRAA", "MWCDE", "HSTPQ", "NIFVW",
    "YCALR", "GMPVE", "TKWFD", "SRQNI", "ADCGM",
]
LABELLED_LABELS = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]


@pytest.fixture
def output_root(tmp_path):
    """Return a fresh temporary directory for each test."""
    return str(tmp_path)


def _make_df(seqs=None, labels=None):
    if seqs is None:
        seqs = LABELLED_SEQS
    if labels is None:
        labels = LABELLED_LABELS
    return pd.DataFrame({"sequence": seqs, "label": labels})


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestAutoPeptideMLInit:
    def test_from_dataframe(self, output_root):
        df = _make_df()
        apml = AutoPeptideML(df, output_root,
                             sequence_field="sequence",
                             label_field="label")
        assert apml.sequence_field == "sequence"
        assert apml.label_field == "label"
        assert len(apml.df) <= len(df)  # duplicates removed

    def test_from_list(self, output_root):
        seqs = ["ACGF", "LLKR", "MMCA"]
        apml = AutoPeptideML(seqs, output_root)
        assert apml.sequence_field == "peptide"
        assert apml.label_field == "label"
        assert len(apml.df) == 3
        assert (apml.df["label"] == 1).all()

    def test_output_directory_created(self, output_root):
        df = _make_df()
        apml = AutoPeptideML(df, output_root,
                             sequence_field="sequence",
                             label_field="label")
        assert os.path.isdir(apml.outputdir)

    def test_remove_duplicates_true(self, output_root):
        df = pd.DataFrame({"sequence": ["ACGF", "ACGF", "LLKR"],
                           "label": [1, 1, 0]})
        apml = AutoPeptideML(df, output_root,
                             sequence_field="sequence",
                             label_field="label",
                             remove_duplicates=True)
        assert len(apml.df) == 2

    def test_remove_duplicates_false(self, output_root):
        df = pd.DataFrame({"sequence": ["ACGF", "ACGF", "LLKR"],
                           "label": [1, 1, 0]})
        apml = AutoPeptideML(df, output_root,
                             sequence_field="sequence",
                             label_field="label",
                             remove_duplicates=False)
        assert len(apml.df) == 3

    def test_metadata_written(self, output_root):
        df = _make_df()
        apml = AutoPeptideML(df, output_root,
                             sequence_field="sequence",
                             label_field="label")
        meta_path = os.path.join(apml.meta_dir, "metadata.yml")
        assert os.path.isfile(meta_path)

    def test_start_data_written(self, output_root):
        df = _make_df()
        apml = AutoPeptideML(df, output_root,
                             sequence_field="sequence",
                             label_field="label")
        start_path = os.path.join(apml.meta_dir, "start-data.tsv")
        assert os.path.isfile(start_path)


# ---------------------------------------------------------------------------
# Preprocessing (_preprocessing_data called implicitly by build_models)
# ---------------------------------------------------------------------------

class TestPreprocessing:
    def test_apml_smiles_column_created(self, output_root):
        df = _make_df()
        apml = AutoPeptideML(df, output_root,
                             sequence_field="sequence",
                             label_field="label")
        apml._preprocessing_data(n_jobs=1, verbose=False)
        assert "apml-smiles" in apml.df.columns
        assert "apml-seqs" in apml.df.columns

    def test_apml_smiles_are_strings(self, output_root):
        df = _make_df()
        apml = AutoPeptideML(df, output_root,
                             sequence_field="sequence",
                             label_field="label")
        apml._preprocessing_data(n_jobs=1, verbose=False)
        assert apml.df["apml-smiles"].dtype == object


# ---------------------------------------------------------------------------
# build_models — full pipeline (classification, ecfp only, few trials)
# ---------------------------------------------------------------------------

class TestBuildModels:
    def test_classification_ecfp(self, output_root):
        df = _make_df()
        apml = AutoPeptideML(df, output_root,
                             sequence_field="sequence",
                             label_field="label")
        apml.build_models(
            task="class",
            reps=["ecfp-4-512"],
            models=["svm"],
            split_strategy="random",
            n_trials=4,
            n_folds_cv=2,
            verbose=False,
            n_jobs=1,
            random_state=42,
        )
        assert apml.ensemble is not None
        assert hasattr(apml, "test_result")
        assert "mcc" in apml.test_result.columns

    def test_regression_ecfp(self, output_root):
        rng = np.random.default_rng(0)
        # Use 30 unique sequences so the random 80/20 split leaves enough train samples
        aa = "ACDEFGHIKLMNPQRSTVWY"
        seqs = [aa[i % 20] * 5 + aa[(i + 1) % 20] * 3 for i in range(30)]
        values = rng.standard_normal(30).tolist()
        df = pd.DataFrame({"sequence": seqs, "value": values})
        apml = AutoPeptideML(df, output_root,
                             sequence_field="sequence",
                             label_field="value")
        apml.build_models(
            task="reg",
            reps=["ecfp-4-512"],
            models=["svm"],
            split_strategy="random",
            n_trials=4,
            n_folds_cv=2,
            verbose=False,
            n_jobs=1,
            random_state=42,
        )
        assert apml.ensemble is not None
        assert "mse" in apml.test_result.columns

    def test_invalid_task_raises(self, output_root):
        df = _make_df()
        apml = AutoPeptideML(df, output_root,
                             sequence_field="sequence",
                             label_field="label")
        with pytest.raises(ValueError):
            apml.build_models(task="invalid_task")

    def test_string_label_raises(self, output_root):
        df = pd.DataFrame({"sequence": LABELLED_SEQS,
                           "label": ["pos"] * 5 + ["neg"] * 5})
        apml = AutoPeptideML(df, output_root,
                             sequence_field="sequence",
                             label_field="label")
        with pytest.raises(ValueError):
            apml.build_models(task="class")

    def test_ensemble_saved_to_disk(self, output_root):
        df = _make_df()
        apml = AutoPeptideML(df, output_root,
                             sequence_field="sequence",
                             label_field="label")
        apml.build_models(
            task="class",
            reps=["ecfp-4-512"],
            models=["svm"],
            split_strategy="random",
            n_trials=3,
            n_folds_cv=2,
            verbose=False,
            n_jobs=1,
        )
        ens_dir = os.path.join(apml.outputdir, "ensemble")
        assert os.path.isdir(ens_dir)
        onnx_files = [f for f in os.listdir(ens_dir) if f.endswith(".onnx")]
        assert len(onnx_files) > 0

    def test_hpo_history_written(self, output_root):
        df = _make_df()
        apml = AutoPeptideML(df, output_root,
                             sequence_field="sequence",
                             label_field="label")
        apml.build_models(
            task="class",
            reps=["ecfp-4-512"],
            models=["svm"],
            split_strategy="random",
            n_trials=3,
            n_folds_cv=2,
            verbose=False,
            n_jobs=1,
        )
        hist_path = os.path.join(apml.meta_dir, "hpo_history.tsv")
        assert os.path.isfile(hist_path)


# ---------------------------------------------------------------------------
# represent() — inference on new sequences after build_models
# ---------------------------------------------------------------------------

class TestRepresent:
    def test_represent_ecfp_shape(self, output_root):
        df = _make_df()
        apml = AutoPeptideML(df, output_root,
                             sequence_field="sequence",
                             label_field="label")
        apml.build_models(
            task="class",
            reps=["ecfp-4-512"],
            models=["svm"],
            split_strategy="random",
            n_trials=3,
            n_folds_cv=2,
            verbose=False,
            n_jobs=1,
        )
        new_seqs = ["ACGF", "LLKR", "MMCA"]
        result = apml.represent(new_seqs, rep="ecfp-4-512", verbose=False, n_jobs=1)
        assert "ecfp-4-512" in result
        assert result["ecfp-4-512"].shape == (3, 512)
