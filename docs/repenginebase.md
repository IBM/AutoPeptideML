

# `RepEngineBase` Class Documentation

**Module:** `rep_engine_base`

## Purpose
`RepEngineBase` is an abstract base class for molecular representation engines. It defines a standard interface and utilities for computing molecular representations from a list of molecules (e.g., SMILES strings), particularly in batched processing. This class is intended to be subclassed, with core functionality like preprocessing and representation computation implemented in derived classes.

---

## Attributes

- **`engine`** (`str`):  
  Name of the representation engine. Typically defined in a subclass or passed during instantiation.

- **`rep`** (`str`):  
  Type of molecular representation (e.g., `'fingerprint'`, `'embedding'`).

- **`properties`** (`dict`):  
  A deep copy of the instance's dictionary at initialization. Captures configuration state.

---

## Constructor

```python
def __init__(self, rep: str, **args)
```

**Parameters:**
- `rep` (`str`): Type of molecular representation.
- `**args` (`dict`): Additional configuration options stored as attributes.

**Effect:**  
Initializes the object, stores `rep`, and adds all additional keyword arguments to the instance. Also creates a deep copy of all these attributes in `self.properties` for serialization.

---

## Public Methods

### `compute_reps`

```python
def compute_reps(self, mols: List[str], verbose: Optional[bool] = False, batch_size: Optional[int] = 12) -> Union[np.ndarray, List[np.ndarray]]
```

**Description:**  
Computes molecular representations in batches using `_preprocess_batch` and `_rep_batch`.

**Parameters:**
- `mols` (`List[str]`): List of molecular inputs (e.g., SMILES strings).
- `verbose` (`bool`, optional): If `True`, shows a progress bar.
- `batch_size` (`int`, optional): Number of molecules per batch.

**Returns:**  
- `np.ndarray` if `average_pooling` is `True` or unset.
- `List[np.ndarray]` if `average_pooling` is explicitly set to `False`.

---

### `dim`

```python
def dim(self) -> int
```

**Description:**  
Abstract method. Must return the dimensionality of the computed representation.

**Raises:**  
- `NotImplementedError`

---

### `_rep_batch`

```python
def _rep_batch(self, batch: List[str]) -> np.ndarray
```

**Description:**  
Abstract method. Must compute and return the representation for a batch of molecules.

**Raises:**  
- `NotImplementedError`

---

### `_preprocess_batch`

```python
def _preprocess_batch(self, batch: List[str]) -> List[str]
```

**Description:**  
Abstract method. Must return a preprocessed version of the batch for representation.

**Raises:**  
- `NotImplementedError`

---

### `save`

```python
def save(self, filename: str)
```

**Description:**  
Serializes and saves the engine’s properties to a YAML file.

**Parameters:**
- `filename` (`str`): Destination path for the YAML file.

## Design Notes

- This class provides **batch processing** support and optional **average pooling** control.
- The use of `batched` from `itertools` supports Python 3.10+ but also includes a fallback implementation for older versions.
- Intended for extension: Subclasses must implement `_rep_batch`, `_preprocess_batch`, and `dim`.
