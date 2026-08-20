# Negative Sampling

**Module:** `autopeptideml.db.negative_sampling`

## Overview

The negative sampling module provides utilities for augmenting a positive-only peptide dataset with negative examples drawn from curated peptide databases. It implements a class-balanced sampling strategy based on molecular weight or sequence length to ensure negatives are physically comparable to the positive samples.

---

## `get_neg_db`

```python
get_neg_db(
    target_db: str,
    verbose: bool,
    return_path: bool = False
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, str]]
```

Retrieves a precompiled negative-sample database. If the database is not present locally it is downloaded automatically using [`gdown`](https://pypi.org/project/gdown/).

| Parameter | Type | Default | Description |
|---|---|---|---|
| `target_db` | `str` | — | Database identifier. Must be one of `'canonical'`, `'non-canonical'`, or `'both'`. |
| `verbose` | `bool` | — | Print download progress. |
| `return_path` | `bool` | `False` | If `True`, also return the local file path alongside the DataFrame. |

**Returns:** `pd.DataFrame` (or `Tuple[pd.DataFrame, str]` when `return_path=True`).

**Raises:** `ImportError` if `gdown` is not installed and download is required.

### Available Databases

| ID | Contents |
|---|---|
| `'canonical'` | Bioactive peptides composed of the 20 standard amino acids. |
| `'non-canonical'` | Bioactive peptides with non-standard residues or chemical modifications. |
| `'both'` | Merged version of canonical and non-canonical databases. |

The databases are downloaded on first use from Google Drive and cached locally under `autopeptideml/data/dbs/`.

---

## `add_negatives_from_db`

```python
add_negatives_from_db(
    df: pd.DataFrame,
    target_db: Union[str, pd.DataFrame],
    sequence_field: str,
    activities_to_exclude: List[str] = [],
    label_field: str = None,
    desired_ratio: float = 1.0,
    verbose: bool = True,
    sample_by: str = 'mw',
    n_jobs: int = cpu_count(),
    random_state: int = 1
) -> pd.DataFrame
```

Augments a dataset with negative samples to reach the desired negative/positive ratio. Negatives are drawn from `target_db` using a property-matched sampling strategy so that the resulting negative set has a similar molecular-weight (or length) distribution to the positive set.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `df` | `pd.DataFrame` | — | Input DataFrame containing the positive samples. |
| `target_db` | `str` or `pd.DataFrame` | — | Source of negatives. Either a built-in database name (`'canonical'`, `'non-canonical'`, `'both'`) or a custom DataFrame with at least a `'smiles'` column. |
| `sequence_field` | `str` | — | Column in `df` containing SMILES strings of the positive samples (automatically set to `'apml-smiles'` when called through `AutoPeptideML`). |
| `activities_to_exclude` | `List[str]` | `[]` | Column names in the database that flag known bioactive entries. Rows where any of these columns equals `1` are excluded from the negative pool. |
| `label_field` | `str` | `None` | Column containing the label. If `None`, defaults to `'label'` and all input rows are assumed positive. |
| `desired_ratio` | `float` | `1.0` | Target `negatives / positives` ratio. |
| `verbose` | `bool` | `True` | Print warnings and progress. |
| `sample_by` | `str` | `'mw'` | Binning strategy. `'mw'` uses molecular weight (requires RDKit); `'length'` uses sequence length. |
| `n_jobs` | `int` | all CPUs | Parallel workers for feature computation. |
| `random_state` | `int` | `1` | Random seed for reproducibility. |

**Returns:** A new `pd.DataFrame` combining the original positive samples and the newly added negatives, shuffled.

**Raises:**

- `ValueError` — if `label_field` is not found in `df`, `sequence_field` is missing, or `target_db` / `sample_by` are invalid.

### Sampling Strategy

1. Compute a molecular property (mass or length) for both the positives and the database.
2. Discretise both into bins using `sklearn`'s `KBinsDiscretizer`.
3. For each bin, sample the required number of negatives to approximate `desired_ratio` while staying within the available pool.

If a bin in the database has fewer molecules than needed, all available molecules in that bin are used (partial fill without replacement).

---

## `setup_databases`

```python
setup_databases()
```

Downloads all three precompiled negative databases (`canonical`, `non-canonical`, `both`) to `autopeptideml/data/dbs/`. Requires `gdown`.

Useful for pre-downloading databases in offline environments.

---

## Example

```python
import pandas as pd
from autopeptideml.db.negative_sampling import add_negatives_from_db

# Positive-only dataset
df = pd.read_csv('positive_peptides.csv')
df['label'] = 1

# Augment with canonical negatives at 1:1 ratio
df_augmented = add_negatives_from_db(
    df=df,
    target_db='canonical',
    sequence_field='smiles',
    label_field='label',
    desired_ratio=1.0,
    sample_by='mw',
    random_state=42
)

print(df_augmented['label'].value_counts())
```

Alternatively, use a custom negative DataFrame:

```python
neg_db = pd.read_csv('my_negatives.csv')  # must contain 'smiles' column
df_augmented = add_negatives_from_db(
    df=df,
    target_db=neg_db,
    sequence_field='smiles',
    activities_to_exclude=['antibacterial', 'antifungal'],
    desired_ratio=2.0
)
```
