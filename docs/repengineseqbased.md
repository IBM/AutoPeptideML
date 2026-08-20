# `RepEngineOnehot` — One-Hot Sequence Encoder

**Module:** `autopeptideml.reps.seq_based`  
**Inherits from:** [`RepEngineBase`](repenginebase.md)

## Overview

`RepEngineOnehot` encodes canonical amino acid sequences as fixed-length binary one-hot vectors. Each position in the sequence maps to a 21-element binary vector (20 standard amino acids + unknown `'X'`). Sequences longer than `max_length` are truncated; shorter ones are zero-padded.

This encoder is useful as a lightweight baseline representation that requires no third-party dependencies beyond NumPy.

---

## Residue Alphabet

The encoder uses a fixed alphabet of 21 characters:

| Index | Residue | Index | Residue | Index | Residue |
|---|---|---|---|---|---|
| 0 | V (Val) | 7 | H (His) | 14 | T (Thr) |
| 1 | I (Ile) | 8 | W (Trp) | 15 | M (Met) |
| 2 | L (Leu) | 9 | F (Phe) | 16 | A (Ala) |
| 3 | E (Glu) | 10 | Y (Tyr) | 17 | G (Gly) |
| 4 | Q (Gln) | 11 | R (Arg) | 18 | P (Pro) |
| 5 | D (Asp) | 12 | K (Lys) | 19 | C (Cys) |
| 6 | N (Asn) | 13 | S (Ser) | 20 | X (unknown) |

Non-canonical residues are not mapped and should be converted to `'X'` beforehand using [`CanonicalCleaner`](pipeline.md#canonicalcleaner).

---

## Attributes

| Attribute | Type | Description |
|---|---|---|
| `engine` | `str` | Fixed to `'one-hot'`. |
| `max_length` | `int` | Maximum sequence length. Sequences are truncated to this value. |
| `name` | `str` | Fixed to `'one-hot'`. |

---

## Constructor

```python
RepEngineOnehot(max_length: int)
```

| Parameter | Type | Description |
|---|---|---|
| `max_length` | `int` | Maximum number of residues per sequence. Determines the output vector length as `max_length × 21`. |

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

Returns an `int8` array of shape `(n_seqs, max_length × 21)`.

---

### `dim`

```python
dim() -> int
```

Returns `max_length × 21`.

---

### `_preprocess_batch`

```python
_preprocess_batch(batch: List[str]) -> List[str]
```

Truncates each sequence to `max_length` characters.

---

### `_rep_batch`

```python
_rep_batch(batch: List[str]) -> np.ndarray
```

Converts each sequence in the batch into a flattened one-hot matrix of shape `(max_length × 21,)`.  
Returns a 2D `int8` array of shape `(len(batch), max_length × 21)`.

---

## Example

```python
from autopeptideml.reps.seq_based import RepEngineOnehot

engine = RepEngineOnehot(max_length=10)

sequences = ['ACDEFGHIKL', 'MWGY']
X = engine.compute_reps(sequences)

print(X.shape)   # (2, 210)  — 10 × 21
print(engine.dim())  # 210
```

---

## Notes

- In `AutoPeptideML.build_models`, pass `reps=['one-hot']` to use this encoder with the default `max_length=50`.
- For variable-length sequences, `max_length` should be set to the maximum expected length in your dataset to avoid truncation of long sequences.
- The encoder operates on canonical sequences. Use the [`to-sequences`](pipeline.md#built-in-pipelines) preprocessing pipeline to convert SMILES or mixed input to canonical sequences before passing to this engine.
