"""
Tests for autopeptideml.train:
  - EarlyStoppingCallback
  - VotingEnsemble (predict, predict_proba, save, load)
  - OptunaTrainer.hpo (end-to-end on a tiny synthetic dataset)
  - NoHpoTrainer.train
"""
import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from autopeptideml.train.architectures import VotingEnsemble, OnnxModel
from autopeptideml.train.trainer import (
    EarlyStoppingCallback,
    OptunaTrainer,
    NoHpoTrainer,
)


# ---------------------------------------------------------------------------
# EarlyStoppingCallback
# ---------------------------------------------------------------------------

class TestEarlyStoppingCallback:
    def _make_study(self, best_value: float):
        study = MagicMock()
        study.best_value = best_value
        return study

    def test_stops_after_patience(self):
        cb = EarlyStoppingCallback(early_stopping_rounds=3, direction="maximize")
        study = self._make_study(0.5)
        # First call — new best
        cb(study, None)
        assert cb._iter == 0
        assert cb._score == pytest.approx(0.5)

        # Three consecutive non-improvements
        study.best_value = 0.5
        cb(study, None)
        cb(study, None)
        cb(study, None)
        study.stop.assert_called_once()

    def test_resets_on_improvement(self):
        cb = EarlyStoppingCallback(early_stopping_rounds=5, direction="maximize")
        study = self._make_study(0.5)
        # Two non-improvements
        cb(study, None)
        cb(study, None)
        assert cb._iter == 1
        # Now an improvement
        study.best_value = 0.8
        cb(study, None)
        assert cb._iter == 0

    def test_invalid_direction_raises(self):
        """Constructing with an invalid direction must raise, not silently pass."""
        with pytest.raises((ValueError, AttributeError)):
            cb = EarlyStoppingCallback(early_stopping_rounds=3, direction="sideways")
            # Trigger the operator path which would fail if unset
            study = MagicMock()
            study.best_value = 1.0
            cb(study, None)

    def test_minimize_direction(self):
        cb = EarlyStoppingCallback(early_stopping_rounds=2, direction="minimize")
        study = self._make_study(1.0)
        cb(study, None)
        assert cb._score == pytest.approx(1.0)
        # Improvement = lower value
        study.best_value = 0.5
        cb(study, None)
        assert cb._iter == 0


# ---------------------------------------------------------------------------
# VotingEnsemble
# ---------------------------------------------------------------------------

def _make_dummy_models(task="class"):
    """Return two trivial sklearn models trained on tiny data."""
    from sklearn.svm import SVC, SVR

    np.random.seed(0)
    X = np.random.randn(40, 4).astype(np.float32)
    y = np.array([0] * 20 + [1] * 20)

    if task == "class":
        m1 = SVC(probability=True).fit(X, y)
        m2 = SVC(probability=True).fit(X, y)
    else:
        m1 = SVR().fit(X, y.astype(float))
        m2 = SVR().fit(X, y.astype(float))
    return [m1, m2], X


class TestVotingEnsemble:
    def test_predict_returns_mean_and_std(self):
        models, X = _make_dummy_models("class")
        ensemble = VotingEnsemble(models, ["rep1", "rep2"])
        mean, std = ensemble.predict({"rep1": X, "rep2": X})
        assert mean.shape == (len(X),)
        assert std.shape == (len(X),)

    def test_predict_proba_returns_mean_and_std(self):
        models, X = _make_dummy_models("class")
        ensemble = VotingEnsemble(models, ["rep1", "rep2"])
        mean, std = ensemble.predict_proba({"rep1": X, "rep2": X})
        assert mean.shape[0] == len(X)
        assert std.shape[0] == len(X)

    def test_save_requires_predict_first(self, tmp_path):
        models, X = _make_dummy_models("class")
        ensemble = VotingEnsemble(models, ["rep1", "rep2"])
        with pytest.raises(RuntimeError, match="predict"):
            ensemble.save(str(tmp_path / "ens"))

    def test_save_raises_on_file_path(self, tmp_path):
        models, X = _make_dummy_models("class")
        ensemble = VotingEnsemble(models, ["rep1", "rep2"])
        ensemble.predict({"rep1": X, "rep2": X})
        file_p = tmp_path / "file.onnx"
        file_p.write_bytes(b"")
        with pytest.raises(FileExistsError):
            ensemble.save(str(file_p))

    def test_save_and_load_roundtrip(self, tmp_path):
        models, X = _make_dummy_models("class")
        ensemble = VotingEnsemble(models, ["rep1", "rep2"])
        ensemble.predict({"rep1": X, "rep2": X})
        save_dir = str(tmp_path / "ens")
        ensemble.save(save_dir)
        loaded = VotingEnsemble.load(save_dir)
        assert len(loaded.models) == 2
        mean, std = loaded.predict({"rep1": X, "rep2": X})
        assert mean.shape == (len(X),)

    def test_load_raises_on_non_directory(self, tmp_path):
        f = tmp_path / "not_a_dir.onnx"
        f.write_bytes(b"")
        with pytest.raises(NotADirectoryError):
            VotingEnsemble.load(str(f))

    def test_single_array_predict(self):
        """Passing a plain ndarray (not a dict) should also work."""
        models, X = _make_dummy_models("class")
        ensemble = VotingEnsemble(models, ["r", "r"])
        mean, std = ensemble.predict(X)
        assert mean.shape == (len(X),)


# ---------------------------------------------------------------------------
# OptunaTrainer — end-to-end on tiny synthetic data
# ---------------------------------------------------------------------------

def _make_classification_data(n=80, d=10, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)
    return X, y


def _make_regression_data(n=80, d=10, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    y = X[:, 0] + 0.1 * rng.standard_normal(n)
    return X, y.astype(float)


class TestOptunaTrainer:
    def test_classification_hpo_finds_best_model(self):
        X, y = _make_classification_data()
        trainer = OptunaTrainer(task="class", direction="maximize", metric="mcc")
        trainer.hpo(
            x={"ecfp": X},
            y=y,
            models=["svm"],
            n_folds=2,
            n_trials=4,
            patience=4,
            random_state=1,
            n_jobs=1,
            verbose=0,
        )
        assert trainer.best_model is not None
        mean, std = trainer.best_model.predict({"ecfp": X})
        assert mean.shape == (len(X),)

    def test_regression_hpo_finds_best_model(self):
        X, y = _make_regression_data()
        trainer = OptunaTrainer(task="reg", direction="maximize", metric="spcc")
        trainer.hpo(
            x={"ecfp": X},
            y=y,
            models=["svm"],
            n_folds=2,
            n_trials=4,
            patience=4,
            random_state=1,
            n_jobs=1,
            verbose=0,
        )
        assert trainer.best_model is not None

    def test_history_is_populated(self):
        X, y = _make_classification_data()
        trainer = OptunaTrainer(task="class")
        trainer.hpo(
            x={"r": X}, y=y, models=["svm"],
            n_folds=2, n_trials=3, patience=3,
            verbose=0,
        )
        assert len(trainer.history) > 0
        assert "run" in trainer.history.columns

    def test_multi_rep_hpo(self):
        X, y = _make_classification_data()
        X2 = X[:, :5]
        trainer = OptunaTrainer(task="class")
        trainer.hpo(
            x={"rep_a": X, "rep_b": X2},
            y=y,
            models=["svm"],
            n_folds=2,
            n_trials=4,
            patience=4,
            verbose=0,
        )
        assert trainer.best_model is not None

    def test_numpy_array_input(self):
        """Passing a plain ndarray should be wrapped automatically."""
        X, y = _make_classification_data()
        trainer = OptunaTrainer(task="class")
        trainer.hpo(
            x=X, y=y, models=["svm"],
            n_folds=2, n_trials=3, patience=3, verbose=0,
        )
        assert trainer.best_model is not None

    def test_early_stopping_respected(self):
        """With patience=1 and enough trials, the study should stop early."""
        X, y = _make_classification_data()
        trainer = OptunaTrainer(task="class")
        trainer.hpo(
            x={"r": X}, y=y, models=["svm"],
            n_folds=2, n_trials=50, patience=1,
            verbose=0,
        )
        # Should have stopped well before 50 trials
        assert len(trainer.history["run"].unique()) < 50


# ---------------------------------------------------------------------------
# NoHpoTrainer
# ---------------------------------------------------------------------------

class TestNoHpoTrainer:
    def test_train_returns_supensemble(self):
        X, y = _make_classification_data()
        from sklearn.model_selection import StratifiedKFold
        kf = StratifiedKFold(n_splits=2, shuffle=True, random_state=0)
        folds = list(kf.split(X, y))

        hpspace = [{"name": "svm", "representation": "r",
                    "variables": {"C": 1.0, "probability": True}}]
        trainer = NoHpoTrainer(hpspace=hpspace,
                               optim_strategy={"task": "class"})
        result = trainer.train(folds, {"r": X}, y)
        assert "models" in result
        assert "reps" in result
        assert len(result["models"]) > 0
