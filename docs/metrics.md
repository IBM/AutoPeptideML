# Metrics

**Module:** `autopeptideml.train.metrics`

## Overview

The metrics module provides evaluation functions for both classification and regression tasks. Two entry points are available:

- [`evaluate`](#evaluate) — single-pass evaluation on a fixed test set.
- [`bootstrap_evaluate`](#bootstrap_evaluate) — bootstrap resampling to compute confidence intervals.

---

## Classification Metrics

Used when `pred_task='class'`.

| Key | Metric | Notes |
|---|---|---|
| `mcc` | Matthews Correlation Coefficient | Primary optimisation metric for classification. Balanced measure that accounts for all four confusion-matrix cells. |
| `acc` | Accuracy | Fraction of correct predictions. |
| `f1` | F1 Score | Harmonic mean of precision and recall. |
| `f1_weighted` | Weighted F1 | F1 weighted by support per class. |
| `precision` | Precision | TP / (TP + FP). |
| `recall` | Recall | TP / (TP + FN). Zero-division handled (returns `0`). |
| `auroc` | Area Under ROC Curve | Computed on raw probabilities (not thresholded). |
| `log_loss` | Log Loss | Normalised cross-entropy. Computed on raw probabilities. |
| `tp` | True Positives | Raw count. |
| `tn` | True Negatives | Raw count. |
| `fp` | False Positives | Raw count. |
| `fn` | False Negatives | Raw count. |

> **Threshold:** All metrics except `auroc` and `log_loss` use a decision threshold of `0.5`.

---

## Regression Metrics

Used when `pred_task='reg'`.

| Key | Metric | Notes |
|---|---|---|
| `mse` | Mean Squared Error | |
| `mae` | Mean Absolute Error | |
| `pcc` | Pearson Correlation Coefficient | |
| `spcc` | Spearman Correlation Coefficient | Primary optimisation metric for regression. |
| `r2` | R² Score | Coefficient of determination. |

---

## `evaluate`

```python
evaluate(
    preds: np.ndarray,
    truth: np.ndarray,
    pred_task: str
) -> Dict[str, float]
```

Compute all metrics for the given task in a single pass.

| Parameter | Type | Description |
|---|---|---|
| `preds` | `np.ndarray` | Model predictions. For classification these should be probabilities in `[0, 1]`. |
| `truth` | `np.ndarray` | Ground truth labels. |
| `pred_task` | `str` | `'class'` for classification, `'reg'` for regression. |

**Returns:** A dictionary mapping metric names to their float values. Any metric that cannot be computed (e.g. single-class test sets) is set to `0.0`.

### Example

```python
import numpy as np
from autopeptideml.train.metrics import evaluate

preds = np.array([0.9, 0.3, 0.8, 0.1, 0.7])
truth = np.array([1,   0,   1,   0,   1  ])

results = evaluate(preds, truth, pred_task='class')
print(results['mcc'])    # Matthews Correlation Coefficient
print(results['auroc'])  # AUROC
```

---

## `bootstrap_evaluate`

```python
bootstrap_evaluate(
    preds: np.ndarray,
    truth: np.ndarray,
    pred_task: str,
    n_bootstrap_samples: int = 1000,
    ci: float = 0.95,
    all_results: bool = False
) -> Dict[str, Dict[str, float]]
```

Estimates confidence intervals for all metrics via bootstrap resampling (sampling with replacement).

| Parameter | Type | Default | Description |
|---|---|---|---|
| `preds` | `np.ndarray` | — | Model predictions. |
| `truth` | `np.ndarray` | — | Ground truth labels. |
| `pred_task` | `str` | — | `'class'` or `'reg'`. |
| `n_bootstrap_samples` | `int` | `1000` | Number of bootstrap iterations. |
| `ci` | `float` | `0.95` | Confidence level for intervals (e.g. `0.95` → 95% CI). |
| `all_results` | `bool` | `False` | If `True`, return all per-sample scores instead of aggregated statistics. |

**Returns (default):** A nested dictionary:

```python
{
    'mcc': {'mean': 0.82, 'ci_lower': 0.74, 'ci_upper': 0.89},
    'auroc': {'mean': 0.94, 'ci_lower': 0.89, 'ci_upper': 0.98},
    ...
}
```

**Returns (`all_results=True`):** A dictionary mapping each metric name to a list of per-bootstrap scores.

### Example

```python
from autopeptideml.train.metrics import bootstrap_evaluate

results = bootstrap_evaluate(
    preds=preds,
    truth=truth,
    pred_task='class',
    n_bootstrap_samples=2000,
    ci=0.95
)

for metric, stats in results.items():
    print(f"{metric}: {stats['mean']:.3f} [{stats['ci_lower']:.3f}, {stats['ci_upper']:.3f}]")
```

---

## Notes

- `NaN` values in metric scores (e.g. from degenerate bootstrap samples) are handled with `np.nanmean` and `np.nanpercentile`.
- For classification, `preds` are thresholded at `0.5` for all discrete metrics before bootstrap sampling.
