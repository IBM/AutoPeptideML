# `AutoPeptideML` — Main Pipeline Class

**Module:** `autopeptideml.apml`  
**Version:** 2.1.0

## Overview

`AutoPeptideML` is the top-level class that orchestrates the complete peptide bioactivity ML workflow:
data ingestion, sequence preprocessing, negative sampling, dataset partitioning, feature representation, hyperparameter optimisation (HPO), evaluation, and reporting.

---

## Constructor

```python
AutoPeptideML(
    data: Union[pd.DataFrame, List[str]],
    outputdir: str,
    sequence_field: str = None,
    label_field: str = None,
    remove_duplicates: bool = True
)
```

Creates a timestamped subdirectory under `outputdir` and writes an initial `metadata/metadata.yml` file.

| Parameter           | Type                          | Default | Description                                                                                                        |
| ------------------- | ----------------------------- | ------- | ------------------------------------------------------------------------------------------------------------------ |
| `data`              | `pd.DataFrame` or `List[str]` | —       | Input data. Pass a DataFrame with sequence and label columns, or a plain list of sequences (all assumed positive). |
| `outputdir`         | `str`                         | —       | Base output directory. A timestamped sub-folder is created automatically.                                          |
| `sequence_field`    | `str`                         | `None`  | Column name containing peptide sequences or SMILES strings. Required when `data` is a DataFrame.                   |
| `label_field`       | `str`                         | `None`  | Column name containing binary (`0`/`1`) or continuous labels. Required when `data` is a DataFrame.                 |
| `remove_duplicates` | `bool`                        | `True`  | Drop rows with duplicate sequences before any processing.                                                          |

**Output directory layout (created on init):**

```
<outputdir>/<timestamp>/
    metadata/
        metadata.yml       # run metadata and status
        start-data.tsv     # copy of the raw input data
```

---

## Class Attributes

| Attribute  | Type                    | Description                                                                                       |
| ---------- | ----------------------- | ------------------------------------------------------------------------------------------------- |
| `df`       | `pd.DataFrame`          | Active working dataset. Extended with `apml-smiles` and `apml-seqs` columns during preprocessing. |
| `metadata` | `dict`                  | Run metadata written to YAML after every step.                                                    |
| `parts`    | `dict`                  | Train / test partition index arrays (keys: `'train'`, `'test'`). Populated by `build_models`.     |
| `x`        | `dict[str, np.ndarray]` | Representation arrays, keyed by representation name.                                              |
| `ensemble` | `VotingEnsemble`        | Best trained ensemble model. Available after `build_models`.                                      |

---

## Public Methods

### `sample_negatives`

```python
sample_negatives(
    target_db: Union[str, pd.DataFrame],
    activities_to_exclude: List[str] = [],
    desired_ratio: float = 1.0,
    verbose: bool = True,
    sample_by: str = 'mw',
    n_jobs: int = cpu_count(),
    random_state: int = 1
)
```

Augments the dataset with negative samples drawn from a peptide database to reach the requested negative/positive ratio. Internally delegates to [`add_negatives_from_db`](negative_sampling.md#add_negatives_from_db).

| Parameter               | Type                    | Default  | Description                                                                                              |
| ----------------------- | ----------------------- | -------- | -------------------------------------------------------------------------------------------------------- |
| `target_db`             | `str` or `pd.DataFrame` | —        | Built-in database name (`'canonical'`, `'non-canonical'`, `'both'`) or a custom DataFrame.               |
| `activities_to_exclude` | `List[str]`             | `[]`     | Column names flagging known active entries that must not appear as negatives.                            |
| `desired_ratio`         | `float`                 | `1.0`    | Target negatives-to-positives ratio.                                                                     |
| `verbose`               | `bool`                  | `True`   | Print progress information.                                                                              |
| `sample_by`             | `str`                   | `'mw'`   | Matching strategy. `'mw'` bins by molecular weight (requires RDKit); `'length'` bins by sequence length. |
| `n_jobs`                | `int`                   | all CPUs | Number of parallel workers for feature computation.                                                      |
| `random_state`          | `int`                   | `1`      | Random seed for reproducibility.                                                                         |

---

### `build_models`

```python
build_models(
    task: str = 'class',
    ensemble: bool = False,
    reps: Union[str, List[str], Dict[str, RepEngineBase]] = ['ecfp-16'],
    models: Union[str, List[str]] = ALL_MODELS,
    split_strategy: str = 'min',
    hestia_generator: HestiaGenerator = None,
    model_configs: Dict[str, dict] = {},
    partitions: Dict[str, np.ndarray] = None,
    folds: List[Tuple[np.ndarray, np.ndarray]] = None,
    n_folds_cv: int = 5,
    verbose: bool = True,
    n_trials: int = 100,
    sim_args: SimArguments = None,
    device: str = 'cpu',
    random_state: int = 1,
    extra_x: np.ndarray = None,
    n_jobs: int = cpu_count()
)
```

The main training entry-point. Internally calls `_preprocessing_data`, `_partitioning`, `_representing`, `_hpo`, and `_evaluating` in sequence.

| Parameter          | Type                                              | Default       | Description                                                                                                                                                                                 |
| ------------------ | ------------------------------------------------- | ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `task`             | `str`                                             | `'class'`     | Task type. `'class'` for binary classification, `'reg'` for regression.                                                                                                                     |
| `ensemble`         | `bool`                                            | `False`       | Build an ensemble over multiple representations.                                                                                                                                            |
| `reps`             | `str`, `List[str]`, or `Dict[str, RepEngineBase]` | `['ecfp-16']` | Representation(s) to compute. Shortcuts: `'ecfp'`, `'esm2-8m'`, `'one-hot'`, etc. Pass a dict to provide pre-built engine objects. See [Representations](repenginebase.md) for all options. |
| `models`           | `str` or `List[str]`                              | all models    | Model families to include in HPO. Supported: `'knn'`, `'svm'`, `'rf'`, `'gradboost'`, `'lightgbm'`, `'xgboost'`.                                                                            |
| `split_strategy`   | `str`                                             | `'min'`       | Data split strategy. `'random'` uses 80/20 random split; `'min'` uses Hestia similarity-based partitioning to minimise leakage.                                                             |
| `hestia_generator` | `HestiaGenerator`                                 | `None`        | Pre-computed Hestia generator to reuse existing partitions.                                                                                                                                 |
| `model_configs`    | `Dict[str, dict]`                                 | `{}`          | Custom hyperparameter search space overrides per model.                                                                                                                                     |
| `partitions`       | `Dict[str, np.ndarray]`                           | `None`        | Pre-defined index arrays (`{'train': ..., 'test': ...}`). Skips Hestia partitioning.                                                                                                        |
| `folds`            | `List[Tuple[np.ndarray, np.ndarray]]`             | `None`        | Custom cross-validation folds as `(train_idx, val_idx)` pairs.                                                                                                                              |
| `n_folds_cv`       | `int`                                             | `5`           | Number of cross-validation folds (used when `folds` is None).                                                                                                                               |
| `verbose`          | `bool`                                            | `True`        | Print progress at each stage.                                                                                                                                                               |
| `n_trials`         | `int`                                             | `100`         | Number of Optuna HPO trials.                                                                                                                                                                |
| `sim_args`         | `SimArguments`                                    | `None`        | Custom Hestia similarity arguments.                                                                                                                                                         |
| `device`           | `str`                                             | `'cpu'`       | Compute device for language model representations: `'cpu'`, `'cuda'`, or `'mps'`.                                                                                                           |
| `random_state`     | `int`                                             | `1`           | Global random seed.                                                                                                                                                                         |
| `extra_x`          | `np.ndarray`                                      | `None`        | Additional feature columns concatenated to every representation array.                                                                                                                      |
| `n_jobs`           | `int`                                             | all CPUs      | Parallelism for preprocessing and partitioning.                                                                                                                                             |

---

### `represent`

```python
represent(
    mols: List[str],
    rep: str,
    n_jobs: int = cpu_count(),
    verbose: bool = True
) -> Dict[str, np.ndarray]
```

Compute a single representation for an arbitrary list of sequences or SMILES strings using an already-initialised engine stored in `self.repengines`. Returns a dictionary `{rep: array}`.

---

### `create_report`

```python
create_report()
```

Renders a Quarto (`.qmd`) HTML report summarising evaluation metrics and results. Requires [Quarto](https://quarto.org) to be installed.

---

## Example Usage

```python
import pandas as pd
from autopeptideml import AutoPeptideML

# --- Build a classification model ---
df = pd.read_csv('peptides.csv')

apml = AutoPeptideML(
    data=df,
    outputdir='results',
    sequence_field='sequence',
    label_field='label'
)

# Optional: add negative samples from the built-in canonical peptide DB
apml.sample_negatives(
    target_db='canonical',
    activities_to_exclude=['antimicrobial'],
    desired_ratio=1.0
)

# Train models using ECFP fingerprints and ESM2-8M embeddings
apml.build_models(
    task='class',
    reps=['ecfp', 'esm2-8m'],
    models=['rf', 'lightgbm'],
    split_strategy='min',
    n_trials=50,
    device='cpu'
)

# Generate a Quarto PDF report
apml.create_report()
```

After training, the output directory contains:

```
<outputdir>/<timestamp>/
    data.tsv                  # pre-processed dataset with apml-smiles, apml-seqs columns
    ensemble/                 # ONNX models (one per rep × best-model)
    metadata/
        metadata.yml          # full run metadata
        hpo_history.tsv       # per-trial HPO scores
        cv-folds.pckl         # cross-validation fold indices
        reps.pckl             # cached representation arrays
        parts.pckl            # train/test partition indices
        preds.npy             # test-set predictions
```

---

## Representation Shortcuts

The `reps` argument of `build_models` accepts the following short identifiers:

| Shortcut         | Engine            | Notes                                                            |
| ---------------- | ----------------- | ---------------------------------------------------------------- |
| `'ecfp'`         | `RepEngineFP`     | Defaults to radius 8, 1024 bits. Format: `ecfp-<radius>-<nbits>` |
| `'fcfp'`         | `RepEngineFP`     | Feature-class Morgan. Format: `fcfp-<radius>-<nbits>`            |
| `'pepfunn'`      | `RepEngineFP`     | Requires `pepfunn` package.                                      |
| `'one-hot'`      | `RepEngineOnehot` | One-hot encoding of canonical sequences (max length 50).         |
| `'esm2-8m'`      | `RepEngineLM`     | ESM-2 8M parameter model.                                        |
| `'esm2-35m'`     | `RepEngineLM`     | ESM-2 35M parameter model.                                       |
| `'esm2-150m'`    | `RepEngineLM`     | ESM-2 150M parameter model.                                      |
| `'esm2-650m'`    | `RepEngineLM`     | ESM-2 650M parameter model.                                      |
| `'esm2-3b'`      | `RepEngineLM`     | ESM-2 3B parameter model.                                        |
| `'esm2-15b'`     | `RepEngineLM`     | ESM-2 15B parameter model.                                       |
| `'prot-t5-xl'`   | `RepEngineLM`     | ProtT5-XL encoder.                                               |
| `'ankh-base'`    | `RepEngineLM`     | ANKH base model.                                                 |
| `'molformer-xl'` | `RepEngineLM`     | IBM MoLFormer-XL (SMILES-based).                                 |
| `'chemberta-2'`  | `RepEngineLM`     | ChemBERTa 77M (SMILES-based).                                    |
| `'peptideclm'`   | `RepEngineLM`     | PeptideCLM 23M (SMILES-based). Requires `smilesPE`.              |

---

## Split Strategies

| Strategy   | Description                                                                                                                                                        |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `'random'` | 80/20 random split (no similarity awareness).                                                                                                                      |
| `'min'`    | Uses [Hestia](https://github.com/IBM/Hestia-OOD) to compute similarity-based partitions and selects the least-leaky split. Recommended for trustworthy evaluation. |

---

## Dependencies

- `pandas`, `numpy`
- `pyyaml`, `tqdm`
- `hestia` (for similarity-based partitioning)
- `rdkit` (for SMILES handling and molecular weight, when using fingerprints or negative sampling)
- `torch`, `transformers` (for language model representations)
- `optuna` (for HPO)
- `onnxmltools`, `onnxruntime`, `skl2onnx` (for model export and inference)
