# Model Architectures

**Module:** `autopeptideml.train.architectures`

## Overview

This module provides the model ensemble infrastructure used to save, load, and run predictions with trained AutoPeptideML models. The two main classes are:

- [`VotingEnsemble`](#votingensemble) — the trained ensemble that averages predictions from multiple individual models.
- [`OnnxModel`](#onnxmodel) — a thin wrapper around an ONNX Runtime session for a single saved model.

Supported model families for training and export:

| Identifier | Family | Notes |
|---|---|---|
| `'knn'` | K-Nearest Neighbours | scikit-learn |
| `'svm'` | Support Vector Machine | scikit-learn |
| `'rf'` | Random Forest | scikit-learn |
| `'gradboost'` | Gradient Boosting | scikit-learn |
| `'lightgbm'` | LightGBM | requires `pip install lightgbm` |
| `'xgboost'` | XGBoost | requires `pip install xgboost` |

---

## `VotingEnsemble`

An ensemble that combines predictions from multiple models, each potentially trained on a different feature representation. Predictions are averaged across all models; both mean and standard deviation are returned.

### Attributes

| Attribute | Type | Description |
|---|---|---|
| `models` | `List[Callable]` | Individual trained models (scikit-learn API or `OnnxModel`). |
| `reps` | `List[str]` | Representation key for each model (same order as `models`). |
| `dims` | `Dict[str, int]` | Feature dimensions per representation. Populated on first `predict` / `predict_proba` call. Required for `save`. |

### Constructor

```python
VotingEnsemble(models: List[Callable], reps: List[str])
```

| Parameter | Type | Description |
|---|---|---|
| `models` | `List[Callable]` | Trained model objects. |
| `reps` | `List[str]` | Representation identifier for each model. |

---

### `predict`

```python
predict(
    x: Union[np.ndarray, Dict[str, np.ndarray]]
) -> Tuple[np.ndarray, np.ndarray]
```

Run regression or classification label prediction across the ensemble.

| Parameter | Type | Description |
|---|---|---|
| `x` | `np.ndarray` or `Dict[str, np.ndarray]` | If a dict, keys must match `self.reps`. |

**Returns:** `(mean_predictions, std_predictions)` — both of shape `(n_samples,)`.

---

### `predict_proba`

```python
predict_proba(
    x: Dict[str, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]
```

Run probability prediction for binary classification.

**Returns:** `(mean_probabilities, std_probabilities)` — both of shape `(n_samples,)`.

---

### `save`

```python
save(path: str) -> None
```

Exports each model in the ensemble to an ONNX file inside `path/`. Files are named `{index}_{rep}.onnx`.

**Requires:** `predict` or `predict_proba` must be called first to populate `self.dims`.

**Supported export backends:**

| Model type | ONNX converter |
|---|---|
| scikit-learn (`knn`, `svm`, `rf`, `gradboost`) | `skl2onnx.to_onnx` |
| LightGBM | `onnxmltools.convert_lightgbm` |
| XGBoost | `onnxmltools.convert_xgboost` |
| CatBoost | `onnxmltools.convert_catboost` |

**Raises:**

- `RuntimeError` — if `save` is called before `predict`/`predict_proba`.
- `FileExistsError` — if `path` points to an existing file.

---

### `load` *(classmethod)*

```python
VotingEnsemble.load(path: str) -> VotingEnsemble
```

Reconstructs a `VotingEnsemble` from a directory of ONNX files. Each file must be named `{index}_{rep}.onnx` as written by `save`.

**Raises:**

- `NotADirectoryError` — if `path` is not a directory.
- `RuntimeError` — if any file in `path` is not an ONNX file.

---

## `OnnxModel`

A thin wrapper around an `onnxruntime.InferenceSession` for a single ONNX model file.

### Constructor

```python
OnnxModel(path: str)
```

| Parameter | Type | Description |
|---|---|---|
| `path` | `str` | Path to the `.onnx` model file. |

Loads the model with `CPUExecutionProvider` and suppresses verbose runtime logging.

---

### `predict`

```python
predict(x: np.ndarray) -> np.ndarray
```

Run inference and return raw predictions (labels or regression values).

| Parameter | Type | Description |
|---|---|---|
| `x` | `np.ndarray` | Input array of shape `(n_samples, n_features)` as `float32`. |

---

### `predict_proba`

```python
predict_proba(x: np.ndarray) -> np.ndarray
```

Run inference and return class probabilities for the positive class.

**Returns:** Array of shape `(n_samples,)` with the probability of class `1`.

---

## Example

```python
from autopeptideml.train.architectures import VotingEnsemble
import numpy as np

# --- Load a saved ensemble ---
ensemble = VotingEnsemble.load('results/2024-01-01 12:00:00/ensemble')

# --- Predict on new data ---
x = {'ecfp': np.random.rand(10, 1024).astype(np.float32)}

# For classification
mean_proba, uncertainty = ensemble.predict_proba(x)
print(mean_proba)     # predicted probabilities
print(uncertainty)    # std across models (uncertainty)

# For regression
mean_val, uncertainty = ensemble.predict(x)
```

---

## Notes

- The ONNX export converts all models to `float32` input type. Ensure your feature arrays are cast to `float32` before passing to `predict` / `predict_proba` on a loaded ensemble.
- The `VotingEnsemble` expects the `x` dict keys to match the `reps` list exactly. Key order does not matter.
