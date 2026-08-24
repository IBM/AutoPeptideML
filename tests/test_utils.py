"""
Tests for autopeptideml.utils:
  - dataset_parsing (read_file, read_data, _is_string_series)
  - label_balance (discretizer)
  - yaml_format (format_numbers)
"""
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from autopeptideml.utils.dataset_parsing import (
    read_file,
    read_data,
    _is_string_series,
)
from autopeptideml.utils.label_balance import discretizer
from autopeptideml.utils.yaml_format import format_numbers


# ---------------------------------------------------------------------------
# dataset_parsing
# ---------------------------------------------------------------------------

class TestReadFile:
    def test_csv(self, tmp_path):
        p = tmp_path / "data.csv"
        p.write_text("seq,label\nAAC,1\nGGG,0\n")
        df = read_file(str(p))
        assert list(df.columns) == ["seq", "label"]
        assert len(df) == 2

    def test_tsv(self, tmp_path):
        p = tmp_path / "data.tsv"
        p.write_text("seq\tlabel\nAAC\t1\nGGG\t0\n")
        df = read_file(str(p))
        assert "label" in df.columns
        assert len(df) == 2

    def test_fasta(self, tmp_path):
        p = tmp_path / "seqs.fasta"
        p.write_text(">seq1\nAACG\n>seq2\nGGGT\n")
        df = read_file(str(p))
        assert "sequence" in df.columns
        assert len(df) == 2
        assert df["sequence"].tolist() == ["AACG", "GGGT"]

    def test_fasta_multiline(self, tmp_path):
        p = tmp_path / "multi.fasta"
        p.write_text(">seq1\nAAC\nGGG\n>seq2\nTTT\n")
        df = read_file(str(p))
        assert df["sequence"].iloc[0] == "AACGGG"

    def test_smi(self, tmp_path):
        p = tmp_path / "mols.smi"
        p.write_text("CC(=O)O\nCCO\n")
        df = read_file(str(p))
        assert "SMILES" in df.columns
        assert len(df) == 2

    def test_unsupported_format_raises(self, tmp_path):
        p = tmp_path / "data.xyz"
        p.write_text("some content")
        with pytest.raises(RuntimeError):
            read_file(str(p))


class TestReadData:
    def test_single_csv_file(self, tmp_path):
        p = tmp_path / "data.csv"
        p.write_text("seq,label\nAAC,1\n")
        df = read_data(str(p))
        assert len(df) == 1

    def test_directory_reads_all_files(self, tmp_path):
        """read_data on a directory must not recurse infinitely (regression for bug #1)."""
        (tmp_path / "a.csv").write_text("seq,label\nAAC,1\n")
        (tmp_path / "b.csv").write_text("seq,label\nGGG,0\n")
        df = read_data(str(tmp_path))
        # Two files, one row each → two total rows
        assert len(df) == 2

    def test_empty_directory(self, tmp_path):
        df = read_data(str(tmp_path))
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0


class TestIsStringSeries:
    def test_string_dtype(self):
        s = pd.Series(["a", "b"], dtype="string")
        assert _is_string_series(s) is True

    def test_object_all_strings(self):
        s = pd.Series(["a", "b", "c"])
        assert _is_string_series(s) is True

    def test_object_with_none(self):
        s = pd.Series(["a", None, "c"])
        assert _is_string_series(s) is True

    def test_object_with_int(self):
        s = pd.Series(["a", 1, "c"])
        assert _is_string_series(s) is False

    def test_numeric_series(self):
        s = pd.Series([1, 2, 3])
        assert _is_string_series(s) is False


# ---------------------------------------------------------------------------
# label_balance
# ---------------------------------------------------------------------------

class TestDiscretizer:
    def test_returns_array(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = discretizer(arr, n_bins=3)
        assert isinstance(result, np.ndarray)

    def test_shape_preserved(self):
        arr = np.linspace(0, 1, 50)
        result = discretizer(arr, n_bins=5)
        assert result.shape[0] == 50

    def test_bins_respected(self):
        arr = np.linspace(0, 10, 100)
        result = discretizer(arr, n_bins=4)
        unique_bins = np.unique(result)
        assert len(unique_bins) <= 4

    def test_return_discretizer_flag(self):
        arr = np.linspace(0, 1, 20)
        labels, disc = discretizer(arr, n_bins=3, return_discretizer=True)
        assert labels is not None
        assert hasattr(disc, "transform")

    def test_none_input(self):
        assert discretizer(None) is None

    def test_1d_input_works(self):
        arr = np.array([0.1, 0.5, 0.9])
        result = discretizer(arr, n_bins=2)
        assert result is not None


# ---------------------------------------------------------------------------
# yaml_format
# ---------------------------------------------------------------------------

class TestFormatNumbers:
    def test_int_string(self):
        assert format_numbers("42") == 42

    def test_float_string(self):
        assert format_numbers("3.14") == pytest.approx(3.14)

    def test_scientific_notation(self):
        assert format_numbers("1e3") == pytest.approx(1000.0)

    def test_plain_string_unchanged(self):
        assert format_numbers("hello") == "hello"

    def test_nested_dict(self):
        obj = {"a": "1", "b": {"c": "2.5"}}
        result = format_numbers(obj)
        assert result == {"a": 1, "b": {"c": 2.5}}

    def test_list(self):
        result = format_numbers(["1", "2.0", "abc"])
        assert result == [1, 2.0, "abc"]

    def test_passthrough_non_string(self):
        assert format_numbers(42) == 42
        assert format_numbers(3.14) == pytest.approx(3.14)
        assert format_numbers(None) is None
