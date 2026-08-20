# `RepEngineFP` — Fingerprint Representation Engine

**Module:** `autopeptideml.reps.fps`  
**Inherits from:** [`RepEngineBase`](repenginebase.md)

## Overview

`RepEngineFP` computes fixed-length molecular fingerprint bit vectors using [RDKit](https://www.rdkit.org/). It supports Extended-Connectivity Fingerprints (ECFP / Morgan), Feature-class Fingerprints (FCFP), and peptide-specific fingerprints via [PepFuNN](https://github.com/novonordisk-research/pepfunn).

**Requires:** `pip install rdkit`  
**PepFuNN fingerprints additionally require:** `pip install git+https://github.com/novonordisk-research/pepfunn`

---

## Attributes

| Attribute | Type | Description |
|---|---|---|
| `engine` | `str` | Fixed to `'fp'`. |
| `nbits` | `int` | Fingerprint bit-vector length. |
| `radius` | `int` | Neighbourhood radius for the Morgan algorithm. |
| `name` | `str` | Auto-generated identifier in the form `fp-<rep>-<nbits>-<radius>`. |
| `generator` | object | RDKit fingerprint generator instance (or `PepFunn_Generator` for PepFuNN). |
| `count` | `bool` | If `True`, uses count-simulation fingerprints instead of binary. |

---

## Constructor

```python
RepEngineFP(rep: str, nbits: int, radius: int)
```

| Parameter | Type | Description |
|---|---|---|
| `rep` | `str` | Fingerprint type. Accepted values: `'ecfp'`, `'ecfp-count'`, `'morgan'`, `'fcfp'`, `'fcfp-count'`, `'pepfunn'`. |
| `nbits` | `int` | Number of bits in the fingerprint vector (e.g. `1024`, `2048`). |
| `radius` | `int` | Morgan radius (e.g. `2` for ECFP4, `4` for ECFP8). |

---

## Methods

### `compute_reps` *(inherited)*

```python
compute_reps(
    mols: List[str],
    verbose: bool = False,
    batch_size: int = 12
) -> np.ndarray
```

Compute fingerprints for a list of SMILES strings. Returns an array of shape `(n_mols, nbits)`.

---

### `dim`

```python
dim() -> int
```

Returns `self.nbits`.

---

### `_preprocess_batch`

```python
_preprocess_batch(batch: List[str]) -> List[str]
```

For standard fingerprints, returns the batch unchanged. For PepFuNN fingerprints, converts SMILES to BILN notation using the [`SmilesToBiln`](pipeline.md#smilestobiln) transformer.

---

### `_rep_batch`

```python
_rep_batch(batch: List[str]) -> List[np.ndarray]
```

Converts each SMILES to an RDKit `Mol` object and computes the fingerprint. Invalid molecules (where `MolFromSmiles` returns `None`) produce zero vectors of length `nbits`.

---

### `_load_generator`

```python
_load_generator(rep: str) -> object
```

Instantiates the appropriate RDKit generator based on `rep`:

| `rep` value | Generator |
|---|---|
| `'ecfp'` / `'morgan'` | `rdFingerprintGenerator.GetMorganGenerator` |
| `'ecfp-count'` | `GetMorganGenerator` with `countSimulation=True` |
| `'fcfp'` | `GetMorganGenerator` with `GetMorganFeatureAtomInvGen()` |
| `'pepfunn'` | `PepFunn_Generator` |

---

## Representation Shortcuts in `build_models`

When passing `reps` to `AutoPeptideML.build_models`, fingerprints can be specified as:

| Format | Example | Meaning |
|---|---|---|
| `<name>` | `'ecfp'` | Uses default radius 8, 1024 bits |
| `<name>-<radius>` | `'ecfp-4'` | Uses 1024 bits, radius 4 |
| `<name>-<radius>-<nbits>` | `'ecfp-4-2048'` | Explicit radius and bit size |

---

## Example

```python
from autopeptideml.reps.fps import RepEngineFP

engine = RepEngineFP(rep='ecfp', nbits=1024, radius=4)

smiles = [
    'CC(=O)NC(CCC(N)=O)C(=O)NC(CCC(N)=O)C(=O)O',
    'CC(N)C(=O)O',
]

fps = engine.compute_reps(smiles, verbose=False)
print(fps.shape)  # (2, 1024)
print(engine.dim())  # 1024
```
