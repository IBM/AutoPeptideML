# Class AutoPeptideML

## Overview

`AutoPeptideML` is a configurable machine learning workflow class designed for peptide modeling. It integrates data pipelines, representations, model training (with HPO), evaluation, and export.

---

## Class: `AutoPeptideML`

### Constructor

```python
AutoPeptideML(config: dict)
```

* Initializes the AutoPeptideML workflow with a provided configuration dictionary.
* Creates output directories and stores pipeline, representation, training, and database settings.

---

### Public Methods

#### `get_pipeline`

```python
get_pipeline(pipe_config: Optional[dict] = None) -> Pipeline
```

Load or construct the preprocessing pipeline.

#### `get_database`

```python
get_database(db_config: Optional[dict] = None) -> Database
```

Create or load the peptide database with optional negative data support.

#### `get_reps`

```python
get_reps(rep_config: Optional[dict] = None) -> Tuple[Dict[str, RepEngineBase], Dict[str, np.ndarray]]
```

Load or compute representations for the data.

#### `get_test`

```python
get_test(test_config: Optional[Dict] = None) -> HestiaGenerator
```

Partition the dataset into training/validation/test using `HestiaGenerator`.

#### `get_train`

```python
get_train(train_config: Optional[Dict] = None) -> BaseTrainer
```

Load and return the trainer based on the configuration (supports Optuna and Grid).

#### `run_hpo`

```python
run_hpo() -> Dict
```

Perform hyperparameter optimization across dataset partitions.

#### `run_evaluation`

```python
run_evaluation(models) -> pd.DataFrame
```

Run evaluation on the trained models and return a DataFrame of results.

#### `save_experiment`

```python
save_experiment(model_backend: str = 'onnx', save_reps: bool = False, save_test: bool = True, save_all_models: bool = True)
```

Save the full experiment including models, test partitions, and configuration.

#### `save_database`

```python
save_database()
```

Export the database to CSV.

#### `save_models`

```python
save_models(ensemble_path: str, backend: str = 'onnx', save_all: bool = True)
```

Save models using `onnx` or `joblib` backends.

#### `save_reps`

```python
save_reps(rep_dir: str)
```

Save precomputed representations to disk.

#### `predict`

```python
predict(df: pd.DataFrame, feature_field: str, experiment_dir: str, backend: str = 'onnx') -> np.ndarray
```

Load a saved experiment and predict using the trained ensemble on new data.

---

### Configuration Keys

The `config` dictionary passed to the constructor must include the following keys:

* `outputdir`: str
* `pipeline`: dict or str
* `representation`: dict or str
* `train`: dict or str
* `databases`: dict
* `test`: dict

---

### Dependencies

* pandas, numpy
* yaml, json
* hestia
* sklearn
* skl2onnx, onnxmltools, joblib (optional)

---

## Example Usage

```python
from autopipeline.autopeptideml import AutoPeptideML

config = yaml.safe_load(open('config.yml'))
runner = AutoPeptideML(config)
pipeline = runner.get_pipeline()
db = runner.get_database()
reps, x = runner.get_reps()
test = runner.get_test()
trainer = runner.get_train()
models = runner.run_hpo()
evaluation = runner.run_evaluation(models)
runner.save_experiment()
```

---

For detailed config templates and supported options, see the corresponding YAML schema documentation.
