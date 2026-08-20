# Pipeline — Preprocessing Modules

**Module:** `autopeptideml.pipeline`

## Overview

The pipeline module provides composable preprocessing elements for converting between peptide sequence formats (amino-acid sequences, SMILES, BILN). It is built around two core abstractions:

- [`BaseElement`](#baseelement) — a single processing step that can be applied to one molecule or parallelised over a list.
- [`Pipeline`](#pipeline) — an ordered sequence of `BaseElement` (or nested `Pipeline`) instances.

Three ready-to-use pipelines are available via [`get_pipeline`](#get_pipeline).

---

## `BaseElement`

**Module:** `autopeptideml.pipeline.pipeline`

Abstract base class for a single molecular processing step. Subclasses implement `_single_call` and are invoked as callables.

### Attributes

| Attribute | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | — | Human-readable identifier for this element. |
| `properties` | `dict` | `{}` | Serialisable configuration dictionary. |
| `parallel` | `str` | `'threading'` | Parallelism backend: `'threading'` (default) or `'processing'`. |

### `__call__`

```python
element(
    mol: Union[str, List[str]],
    n_jobs: int = cpu_count(),
    verbose: bool = False
) -> Union[str, List[str]]
```

Dispatches to `_single_call` for a single string, or `_parallel_call` for a list.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `mol` | `str` or `List[str]` | — | Molecule(s) to process. |
| `n_jobs` | `int` | all CPUs | Parallel workers. `1` forces sequential execution. |
| `verbose` | `bool` | `False` | Show a `tqdm` progress bar. |

### Abstract: `_single_call`

```python
_single_call(mol: str) -> Optional[str]
```

Process a single molecule string. Return `None` to discard the molecule (filtered out by `_clean`).

**Raises:** `NotImplementedError`

---

## `Pipeline`

**Module:** `autopeptideml.pipeline.pipeline`

An ordered sequence of processing steps. Each step receives the output of the previous one (or the original input, if `aggregate=True`).

### Constructor

```python
Pipeline(
    elements: List[Union[BaseElement, Pipeline]],
    name: str = 'pipeline',
    aggregate: bool = False
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `elements` | `List` | — | Ordered list of `BaseElement` or nested `Pipeline` instances. |
| `name` | `str` | `'pipeline'` | Identifier for this pipeline. |
| `aggregate` | `bool` | `False` | If `True`, apply all elements independently on the original input and return their combined outputs. Used for forked processing (e.g. separate streams for SMILES and sequences that are then merged). |

### `__call__`

```python
pipeline(
    mols: List[str],
    n_jobs: int = cpu_count(),
    verbose: bool = False
) -> Union[List[str], List[List[str]]]
```

### `save`

```python
save(filename: str)
```

Serialises pipeline properties to a YAML file.

### `load` *(classmethod)*

```python
Pipeline.load(filename: str, element_registry: dict) -> Pipeline
```

Reconstructs a `Pipeline` from a saved YAML file using an element registry mapping names to constructors.

---

## Built-in Processing Elements

### Sequence Elements

**Module:** `autopeptideml.pipeline.sequence`

#### `CanonicalCleaner`

Replaces non-canonical residues in a sequence with a substitution character.

```python
CanonicalCleaner(substitution: str = 'X')
```

| Parameter | Default | Description |
|---|---|---|
| `substitution` | `'X'` | Character to use for non-canonical residues. |

**Example:**
```python
from autopeptideml.pipeline.sequence import CanonicalCleaner
cleaner = CanonicalCleaner(substitution='G')
cleaner('ACDB3L')  # → 'ACDGGL'  (non-canonical '3' → 'G')
```

#### `CanonicalFilter`

Keeps or discards sequences based on whether they consist entirely of the 20 canonical amino acids.

```python
CanonicalFilter(keep_canonical: bool = True)
```

| Parameter | Default | Description |
|---|---|---|
| `keep_canonical` | `True` | `True` → keep canonical sequences; `False` → keep non-canonical. |

Non-matching sequences are returned as `None` and removed from the output list.

---

### SMILES Elements

**Module:** `autopeptideml.pipeline.smiles`

#### `SequenceToSmiles`

Converts a canonical amino-acid sequence to a SMILES string using the ChEMBL monomer library.

```python
SequenceToSmiles()
```

Uses multiprocessing (`parallel = 'processing'`). Non-canonical residues are omitted; returns `None` for empty results.

#### `FilterSmiles`

Passes through only molecules that are (or are not) valid SMILES strings.

```python
FilterSmiles(keep_smiles: Optional[bool] = True)
```

| Parameter | Default | Description |
|---|---|---|
| `keep_smiles` | `True` | `True` → keep valid SMILES; `False` → keep non-SMILES (sequences). |

#### `CanonicalizeSmiles`

Converts a SMILES string to RDKit canonical form. Returns `None` for invalid SMILES.

```python
CanonicalizeSmiles()
```

#### `SmilesToSequence`

Converts a SMILES string back to a canonical amino-acid sequence by decomposing it into monomers and matching them against the ChEMBL library.

```python
SmilesToSequence(keep_analog: bool = True)
```

| Parameter | Default | Description |
|---|---|---|
| `keep_analog` | `True` | If `True`, non-canonical monomers are substituted by their closest canonical analogue; if `False`, they are replaced with `'X'`. |

#### `SmilesToBiln`

Converts a SMILES string to BILN (Biopolymer Identifier Language for Non-standard peptides) notation.

```python
SmilesToBiln(human_readable: bool = False, handle_errors: bool = False)
```

| Parameter | Default | Description |
|---|---|---|
| `human_readable` | `False` | If `True`, outputs a more descriptive BILN with full monomer names. |
| `handle_errors` | `False` | If `True`, returns `None` on parse errors rather than raising. |

#### `BilnToSmiles`

Converts a BILN string back to SMILES.

```python
BilnToSmiles()
```

---

## `get_pipeline`

**Module:** `autopeptideml.pipeline.default_pipelines`

```python
get_pipeline(name: str, **kwargs) -> Pipeline
```

Retrieve a pre-built pipeline by name.

| Parameter | Type | Description |
|---|---|---|
| `name` | `str` | Pipeline identifier. |
| `**kwargs` | `Any` | Arguments forwarded to the pipeline constructor. |

**Raises:** `ValueError` if `name` is not a recognised pipeline.

---

## Built-in Pipelines

### `'to-smiles'`

Converts sequences **and** existing SMILES to canonical SMILES.

**Flow:**

```
input
 ├── [non-SMILES path] FilterSmiles(keep_smiles=False) → CanonicalCleaner → SequenceToSmiles
 └── [SMILES path]     FilterSmiles(keep_smiles=True)
 └── aggregate both streams → CanonicalizeSmiles
```

**Kwargs:**

| Kwarg | Type | Default | Description |
|---|---|---|---|
| `substitution` | `str` | `'G'` | Substitution for non-canonical residues before SMILES conversion. |

---

### `'to-smiles-fast'`

Same as `'to-smiles'` but skips the final `CanonicalizeSmiles` step for speed.

---

### `'to-sequences'`

Converts sequences and SMILES to canonical amino-acid sequences.

**Flow:**

```
input
 ├── [non-SMILES path] FilterSmiles(keep_smiles=False)
 └── [SMILES path]     FilterSmiles(keep_smiles=True) → SmilesToSequence
 └── aggregate both streams → CanonicalCleaner(substitution)
```

**Kwargs:**

| Kwarg | Type | Default | Description |
|---|---|---|---|
| `substitution` | `str` | `'X'` | Substitution for residues not in the canonical alphabet. |
| `keep_analog` | `bool` | `True` | Preserve closest analogue for non-canonical monomers during SMILES → sequence. |

---

## Example

```python
from autopeptideml.pipeline import get_pipeline

# Convert a mix of sequences and SMILES to canonical SMILES
pipe = get_pipeline('to-smiles')
inputs = ['ACDEFGHIKL', 'CC(=O)NC(CCC(N)=O)C(=O)O']
smiles = pipe(inputs, n_jobs=4, verbose=True)
print(smiles)

# Convert to sequences
pipe2 = get_pipeline('to-sequences')
seqs = pipe2(inputs, n_jobs=4)
print(seqs)
```
