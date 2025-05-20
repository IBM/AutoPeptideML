from typing import *
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
        preds = preds > 0.5
        metrics = CLASSIFICATION_METRICS

    for key, value in metrics.items():
        try:
            result[key] = value(preds, truth)
        except ValueError as e:
            result[key] = np.nan
    return result
