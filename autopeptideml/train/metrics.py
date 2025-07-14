from collections import defaultdict
from typing import Dict

import numpy as np

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (matthews_corrcoef,
                             accuracy_score, f1_score,
                             precision_score, recall_score, mean_squared_error,
                             mean_absolute_error, roc_auc_score)


def _pcc(preds, truths):
    return pearsonr(preds, truths)[0]


def _spcc(preds, truths):
    return spearmanr(preds, truths)[0]


def _f1_weighted(preds, truths):
    return f1_score(preds, truths, average='weighted')


def _recall(preds, truths):
    return recall_score(preds, truths, zero_division=True)


CLASSIFICATION_METRICS = {
    'mcc': matthews_corrcoef,
    'acc': accuracy_score,
    'f1': f1_score,
    'f1_weighted': _f1_weighted,
    'precision': precision_score,
    'recall': _recall,
    'auroc': roc_auc_score
}

REGRESSION_METRICS = {
    'mse': mean_squared_error,
    'mae': mean_absolute_error,
    'pcc': _pcc,
    'spcc': _spcc
}


def evaluate(preds, truth, pred_task) -> Dict[str, float]:
    result = {}
    if pred_task == 'reg':
        metrics = REGRESSION_METRICS
    else:
        metrics = CLASSIFICATION_METRICS

    for key, value in metrics.items():
        if key in ['auroc'] or pred_task == 'reg':
            t_pred = preds
        else:
            t_pred = preds > 0.5
        try:
            result[key] = value(truth, t_pred)
        except ValueError:
            result[key] = 0.0
        if np.isnan(result[key]):
            result[key] = 0.0
    return result


def bootstrap_evaluate(
    preds: np.ndarray,
    truth: np.ndarray,
    pred_task: str,
    n_bootstrap_samples: int = 1000,
    ci: float = 0.95,
    all_results: bool = False
) -> Dict[str, Dict[str, float]]:
    if pred_task == 'reg':
        metrics = REGRESSION_METRICS
    else:
        preds = preds > 0.5
        metrics = CLASSIFICATION_METRICS

    n = len(preds)
    metric_scores = defaultdict(list)

    for _ in range(n_bootstrap_samples):
        # Sample with replacement
        indices = np.random.choice(n, n, replace=True)
        sample_preds = preds[indices]
        sample_truth = truth[indices]

        for key, metric_fn in metrics.items():
            try:
                score = metric_fn(sample_preds, sample_truth)
            except Exception:
                score = np.nan
            metric_scores[key].append(score)
    if all_results:
        return metric_scores
    results = {}
    alpha = 1 - ci
    for key, scores in metric_scores.items():
        scores = np.array(scores)
        mean = np.nanmean(scores)
        lower = np.nanpercentile(scores, 100 * alpha / 2)
        upper = np.nanpercentile(scores, 100 * (1 - alpha / 2))
        results[key] = {
            'mean': mean,
            'ci_lower': lower,
            'ci_upper': upper
        }

    return results
