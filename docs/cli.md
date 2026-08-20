# CLI Reference

**Entry point:** `autopeptideml`

AutoPeptideML exposes a command-line interface built with [Typer](https://typer.tiangolo.com/). Three commands are available:

| Command | Description |
|---|---|
| [`build-model`](#build-model) | Interactive model builder (with optional config file). |
| [`prepare-config`](#prepare-config) | Generate a YAML config file without training. |
| [`predict`](#predict) | Run predictions using a previously trained ensemble. |

---

## `build-model`

```bash
autopeptideml build-model [OPTIONS]
```

Builds, trains, and evaluates an AutoPeptideML model. If no `--config-path` is provided, an interactive prompt guides you through dataset and training setup.

### Options

| Option | Type | Default | Description |
|---|---|---|---|
| `--outdir` | `str` | `apml-result` | Directory where all output files will be saved. A timestamped subdirectory is created inside. |
| `--config-path` | `str` | `<outdir>/setup-config.yml` | Path to an existing YAML configuration file. If omitted, the interactive wizard runs and writes the config to `<outdir>/setup-config.yml`. |

### Workflow

1. If `--config-path` is not given, the interactive wizard collects:
   - Task type (classification / regression)
   - Dataset path and column mapping
   - Negative sampling strategy (for classification)
   - Model families and representations to try
   - Number of HPO trials and compute device

2. Loads the dataset from the path in `config['datasets']['main']['path']`.

3. If `config['datasets']['neg-db']` is present, calls `sample_negatives`.

4. Calls `build_models` with the config parameters.

5. Calls `create_report` to render the Quarto evaluation report.

### Config File Keys

```yaml
datasets:
  main:
    path: path/to/data.csv       # or .tsv / .fasta
    feat-fields: sequence        # column with sequences/SMILES
    label-field: label           # column with labels (or 'Assume all entries are positive')
  neg-db:                        # optional
    path: canonical              # 'canonical', 'non-canonical', 'both', or a custom path
    feat-fields: null
    activities-to-exclude: []    # list of activity columns to exclude from negatives

task: class                      # 'class' or 'reg'
pipeline: to-smiles
reps: [ecfp, esm2-8m]           # representation short names
models: [rf, lightgbm, knn]     # model families
n-trials: 100                    # Optuna HPO trials
device: cpu                      # 'cpu', 'cuda', or 'mps'
split-strategy: min
metric: mcc                      # 'mcc' (class) or 'spcc' (reg)
direction: maximize
n-jobs: -1                       # -1 = all CPUs
```

### Example

```bash
# Interactive wizard, results saved to 'my-experiment'
autopeptideml build-model --outdir my-experiment

# Use a pre-existing config
autopeptideml build-model --outdir my-experiment --config-path my-experiment/setup-config.yml
```

---

## `prepare-config`

```bash
autopeptideml prepare-config CONFIG_PATH
```

Runs the interactive wizard and saves the resulting YAML config to `CONFIG_PATH` **without** training a model. The `.yml` suffix is appended automatically if missing.

### Arguments

| Argument | Type | Description |
|---|---|---|
| `CONFIG_PATH` | `str` | Destination path for the YAML config file. |

### Example

```bash
autopeptideml prepare-config experiments/antimicrobial-config
# → writes experiments/antimicrobial-config.yml
```

---

## `predict`

```bash
autopeptideml predict RESULT_DIR FEATURES_PATH [OPTIONS]
```

Loads a trained ensemble from a previous `build-model` run and generates predictions on new data.

### Arguments

| Argument | Type | Description |
|---|---|---|
| `RESULT_DIR` | `str` | Path to the experiment output directory (the **timestamped** subdirectory). Must contain `ensemble/` and optionally `metadata/metadata.yml`. |
| `FEATURES_PATH` | `str` | Path to the input file (CSV / TSV) containing the molecules to predict. |

### Options

| Option | Type | Default | Description |
|---|---|---|---|
| `--feature-field` | `str` | auto-detected | Column name with sequences or SMILES. Auto-detects `'sequence'`, `'smiles'`, or `'SMILES'` if not provided. |
| `--output-path` | `str` | `predictions.tsv` | Where to write the predictions. Output is tab-separated with added `preds` and `uncertainty` columns. |
| `--n-jobs` | `int` | `-1` (all CPUs) | Parallelism for preprocessing. |
| `--device` | `str` | `'cpu'` | Device for language model inference: `'cpu'`, `'cuda'`, or `'mps'`. |

### Output Format

The output file is a tab-separated copy of the input data with two extra columns:

| Column | Description |
|---|---|
| `preds` | Predicted probability (classification) or continuous value (regression). |
| `uncertainty` | Standard deviation of predictions across ensemble members. Use as a calibrated uncertainty estimate. |

### Example

```bash
autopeptideml predict \
  my-experiment/2024-06-01_12:00:00 \
  new_peptides.csv \
  --feature-field sequence \
  --output-path predictions.tsv \
  --device cpu
```

### Raises

- `RuntimeError` — if the `ensemble/` subdirectory does not exist in `RESULT_DIR`.
- `FileNotFoundError` — if `FEATURES_PATH` does not exist.

---

## Python API Equivalent

All CLI commands map directly to Python calls:

```python
from autopeptideml import AutoPeptideML
from autopeptideml.train.architectures import VotingEnsemble
from autopeptideml.utils.dataset_parsing import read_data
from autopeptideml.pipeline import get_pipeline

# build-model equivalent
apml = AutoPeptideML(data=df, outputdir='my-experiment',
                     sequence_field='sequence', label_field='label')
apml.build_models(task='class', reps=['ecfp'], models=['rf'], n_trials=50)
apml.create_report()

# predict equivalent
ensemble = VotingEnsemble.load('my-experiment/2024-06-01_12:00:00/ensemble')
```

For full API details, see the [AutoPeptideML class reference](autopeptideml.md).
