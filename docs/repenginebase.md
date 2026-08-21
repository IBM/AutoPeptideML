# `RepEngineBase` — Abstract Representation Engine

**Module:** `autopeptideml.reps.engine`

## Overview

`RepEngineBase` is the abstract base class for all molecular representation engines in AutoPeptideML. It defines the standard interface for batch-level representation computation, serialisation, and property management. All concrete engines — fingerprints, language models, one-hot encoding — inherit from this class.

Subclasses must implement:

- [`_preprocess_batch`](#_preprocess_batch)
- [`_rep_batch`](#_rep_batch)
- [`dim`](#dim)

---

## Attributes

| Attribute | Type | Description |
|---|---|---|
| `engine` | `str` | Class-level identifier for the engine type (e.g. `'fp'`, `'lm'`, `'one-hot'`). |
| `rep` | `str` | Instance-level representation name passed at construction. |
| `properties` | `dict` | Deep copy of all instance attributes captured at `__init__` time. Used for serialisation. |

---

## Constructor

```python
RepEngineBase(rep: str, **args)
```

| Parameter | Type | Description |
|---|---|---|
| `rep` | `str` | Representation identifier (e.g. `'ecfp'`, `'esm2-8m'`). |
| `**args` | `Any` | Additional keyword arguments added as instance attributes and captured in `self.properties`. |

---

## Public Methods

### `compute_reps`

```python
compute_reps(
    mols: List[str],
    verbose: Optional[bool] = False,
    batch_size: Optional[int] = 12
) -> Union[np.ndarray, List[np.ndarray]]
```

Computes representations for a list of molecules by calling `_preprocess_batch` then `_rep_batch` on successive batches.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `mols` | `List[str]` | — | Input molecules as SMILES strings or amino-acid sequences. |
| `verbose` | `bool` | `False` | Show a `tqdm` progress bar over batches. |
| `batch_size` | `int` | `12` | Number of molecules per batch. |

**Returns:**

- `np.ndarray` — stacked array of shape `(n_mols, dim)` when `average_pooling` is `True` or not set.
- `List[np.ndarray]` — list of per-molecule arrays (variable length) when `average_pooling=False`.

---

### `dim`

```python
dim() -> int
```

Returns the dimensionality of the computed representation vector.

**Raises:** `NotImplementedError` — must be implemented by subclasses.

---

### `get_num_params`

```python
get_num_params() -> int
```

Returns the total number of learnable parameters in the engine. The base implementation returns `0`; language model engines override this to return the actual parameter count.

---

### `save`

```python
save(filename: str)
```

Serialises `self.properties` to a YAML file at `filename`. This enables reloading the engine configuration later.

| Parameter | Type | Description |
|---|---|---|
| `filename` | `str` | Destination path for the YAML file. |

---

### `__str__`

```python
__str__() -> str
```

Returns a JSON string representation of `self.properties`.

---

## Abstract Methods (must be implemented by subclasses)

### `_rep_batch`

```python
_rep_batch(batch: List[str]) -> np.ndarray
```

Compute and return representations for a single batch.

**Raises:** `NotImplementedError`

---

### `_preprocess_batch`

```python
_preprocess_batch(batch: List[str]) -> List[str]
```

Apply any necessary preprocessing to a batch before representation computation (e.g. tokenisation, canonical conversion).

**Raises:** `NotImplementedError`

---

## Subclasses

| Class | Module | Description |
|---|---|---|
| [`RepEngineFP`](repenginefp.md) | `autopeptideml.reps.fps` | Molecular fingerprints via RDKit (ECFP, FCFP, PepFuNN). |
| [`RepEngineSkfp`](repengineskfp.md) | `autopeptideml.reps.fps` | 28 fingerprint families via scikit-fingerprints (MACCS, AtomPair, Avalon, PubChem, Mordred, …). |
| [`RepEngineLM`](repenginelm.md) | `autopeptideml.reps.lms` | Pre-trained language model embeddings (ESM2, ProtT5, MoLFormer, …). |
| [`RepEngineOnehot`](repengineseqbased.md) | `autopeptideml.reps.seq_based` | Fixed-length one-hot encoding for canonical amino acid sequences. |

---

## Design Notes

- **Batch processing** is handled centrally by `compute_reps`; subclasses only need to implement single-batch logic in `_rep_batch`.
- **Pooling control:** setting `self.average_pooling = False` before calling `compute_reps` causes it to return raw per-residue tensors rather than pooled vectors. This is useful for sequence-level models.
- **Python compatibility:** `batched` from `itertools` is used (Python ≥ 3.12) with a fallback `islice`-based implementation for earlier versions.
