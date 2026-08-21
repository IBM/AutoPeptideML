"""
Tests for autopeptideml.utils.plots — specifically guarding against the
zero-width-figure crash in plot_model_vs_rep when the HPO history contains
fewer than 20 unique runs (which happens routinely with one representation
and aggressive early-stopping).
"""
import matplotlib
matplotlib.use('Agg')  # non-interactive backend — must come before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from autopeptideml.utils.plots import plot_model_vs_rep, plot_optimization_history


def _make_history(n_runs: int, reps: list, n_folds: int = 3) -> pd.DataFrame:
    """Build a minimal HPO history DataFrame with the same schema as the real pipeline."""
    rng = np.random.default_rng(0)
    rows = []
    for run in range(1, n_runs + 1):
        rep = reps[(run - 1) % len(reps)]
        for fold in range(n_folds):
            rows.append({
                'run': run,
                'fold': fold,
                'name': 'svm',
                'representation': rep,
                'variables': str({'C': 1.0}),
                'mcc': float(rng.uniform(0.3, 0.8)),
                'acc': float(rng.uniform(0.5, 0.9)),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Bug regression: plot_model_vs_rep must not crash when n_runs < 20
# (previously caused LinAlgError: Singular matrix via a zero-width figure)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_runs", [1, 2, 5, 10, 19])
def test_plot_model_vs_rep_few_runs_one_rep(n_runs):
    """plot_model_vs_rep must succeed with fewer than 20 unique runs, one rep."""
    history = _make_history(n_runs=n_runs, reps=['one-hot'])
    try:
        plot_model_vs_rep(history=history, metric='mcc')
    finally:
        plt.close('all')


@pytest.mark.parametrize("n_runs", [1, 2, 5, 10, 19])
def test_plot_model_vs_rep_few_runs_two_reps(n_runs):
    """plot_model_vs_rep must succeed with fewer than 20 unique runs, two reps."""
    history = _make_history(n_runs=n_runs, reps=['one-hot', 'ecfp-16'])
    try:
        plot_model_vs_rep(history=history, metric='mcc')
    finally:
        plt.close('all')


def test_plot_model_vs_rep_mutates_history():
    """plot_model_vs_rep adds Representation/Model/Run columns so the QMD can drop them."""
    history = _make_history(n_runs=5, reps=['one-hot'])
    assert 'Representation' not in history.columns
    plot_model_vs_rep(history=history, metric='mcc')
    plt.close('all')
    assert 'Representation' in history.columns
    assert 'Model' in history.columns
    assert 'Run' in history.columns


def test_plot_optimization_history_few_runs():
    """plot_optimization_history must succeed with very few runs."""
    history = _make_history(n_runs=2, reps=['one-hot'])
    try:
        plot_optimization_history(history=history, metric='mcc')
    finally:
        plt.close('all')
